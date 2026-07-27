"""Create-only, no-resume runner for real PFCR CURE-Lite training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..phase_resolved_real_cache import PFCRRealCacheAdapter
from ..phase_resolved_real_states import PFCRRealStateCatalog
from ..phase_resolved_relation_decoder import (
    PhaseResolvedRelationDecoderConfig,
)
from ..phase_resolved_relation_population import (
    PFCR_FEATURE_CHANNELS,
    PFCR_FEATURE_STRIDE,
)
from ..phase_resolved_relation_training import (
    PFCR_TRAINING_ALGORITHM_VERSION,
    PFCR_TRAIN_RELATION_DIM,
    PhaseResolvedRelationTrainingConfig,
)
from .phase_resolved_real_artifacts import (
    LoadedPFCRRealDecoderArtifact,
    PFCRRealDecoderRunConfig,
    load_pfcr_real_decoder_artifact,
    save_pfcr_real_decoder_artifact,
)
from .phase_resolved_real_training import (
    PFCR_REAL_PREFLIGHT_SCHEMA,
    PFCRRealFormalTrainingConfig,
    PFCRRealPreflightConfig,
    execute_pfcr_real_formal_training,
    pfcr_real_formal_schedule_payload,
)


PFCR_REAL_FORMAL_ATTEMPT_SCHEMA = (
    "cure-lite-pfcr-real-formal-attempt-v1"
)
PFCR_REAL_FORMAL_COMPLETE_SCHEMA = (
    "cure-lite-pfcr-real-formal-complete-v1"
)
_STARTED_NAME = "STARTED.json"
_RUN_RECEIPT_NAME = "run_receipt.json"
_ARTIFACT_DIR = "decoder_artifact"
_COMPLETE_NAME = "COMPLETE.json"
_HEX = frozenset("0123456789abcdef")


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    with source.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_json(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _fingerprinted(
    value: Mapping[str, object],
    *,
    field: str,
) -> dict[str, object]:
    core = dict(value)
    if field in core:
        raise ValueError(f"{field} already exists")
    return {**core, field: stable_fingerprint(core)}


def _verify_fingerprint(
    value: Mapping[str, object],
    *,
    field: str,
    name: str,
) -> str:
    core = dict(value)
    fingerprint = core.pop(field, None)
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(core) != fingerprint
    ):
        raise ValueError(f"{name} fingerprint mismatch")
    return _digest(fingerprint, name=f"{name} fingerprint")


@dataclass(frozen=True)
class PFCRRealPreflightAuthorization:
    """A verified seed-specific authorization for formal implementation."""

    source_directory: Path
    started_file_sha256: str
    result_file_sha256: str
    complete_file_sha256: str
    seed: int
    result_fingerprint: str
    complete_fingerprint: str
    cache_contract_fingerprint: str
    state_catalog_fingerprint: str
    lineage_allowlist_fingerprint: str


@dataclass(frozen=True)
class PFCRDevelopmentAuthorization:
    """Current bounded-evidence Development results for seeds 42 and 43."""

    file_sha256_by_seed: Mapping[int, str]
    result_fingerprint_by_seed: Mapping[int, str]
    authorization_fingerprint: str

    def __post_init__(self) -> None:
        if set(self.file_sha256_by_seed) != {42, 43} or set(
            self.result_fingerprint_by_seed
        ) != {42, 43}:
            raise ValueError(
                "PFCR Development authorization requires seeds 42 and 43"
            )
        for mapping in (
            self.file_sha256_by_seed,
            self.result_fingerprint_by_seed,
        ):
            for value in mapping.values():
                _digest(value, name="Development digest")
        if stable_fingerprint(self.canonical_payload()) != (
            self.authorization_fingerprint
        ):
            raise ValueError(
                "PFCR Development authorization fingerprint mismatch"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "cure-lite-pfcr-development-authorization-v1"
            ),
            "training_algorithm_version": (
                PFCR_TRAINING_ALGORITHM_VERSION
            ),
            "seeds": [42, 43],
            "file_sha256_by_seed": {
                str(seed): self.file_sha256_by_seed[seed]
                for seed in (42, 43)
            },
            "result_fingerprint_by_seed": {
                str(seed): self.result_fingerprint_by_seed[seed]
                for seed in (42, 43)
            },
        }


def load_pfcr_development_authorization(
    seed42_result: str | Path,
    seed43_result: str | Path,
) -> PFCRDevelopmentAuthorization:
    """Require both current v2 Development seeds before formal training."""

    paths = {
        42: Path(seed42_result).expanduser(),
        43: Path(seed43_result).expanduser(),
    }
    file_hashes: dict[int, str] = {}
    result_fingerprints: dict[int, str] = {}
    expected_decoder = PhaseResolvedRelationDecoderConfig(
        feature_channels=PFCR_FEATURE_CHANNELS,
        feature_stride=PFCR_FEATURE_STRIDE,
        relation_dim=PFCR_TRAIN_RELATION_DIM,
    )
    expected_decoder_payload = {
        "feature_channels": expected_decoder.feature_channels,
        "feature_stride": expected_decoder.feature_stride,
        "relation_dim": expected_decoder.relation_dim,
        "parameter_count": expected_decoder.expected_parameter_count,
        "relation_policy": expected_decoder.relation_policy,
        "feature_normalization_policy": (
            expected_decoder.feature_normalization_policy
        ),
        "evidence_policy": expected_decoder.evidence_policy,
        "evidence_ceiling": expected_decoder.evidence_ceiling,
        "release_policy": expected_decoder.release_policy,
        "output_policy": expected_decoder.output_policy,
    }
    for seed, raw_path in paths.items():
        if raw_path.is_symlink():
            raise ValueError(
                "PFCR Development result may not be a symlink"
            )
        path = raw_path.resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                "PFCR Development result must be a regular file"
            )
        result = _strict_json(
            path,
            name=f"PFCR Development seed {seed} result",
        )
        fingerprint = _verify_fingerprint(
            result,
            field="result_fingerprint",
            name=f"PFCR Development seed {seed} result",
        )
        if (
            result.get("schema_version")
            != PFCR_TRAINING_ALGORITHM_VERSION
            or result.get("config")
            != PhaseResolvedRelationTrainingConfig(
                seed=seed
            ).manifest()
            or result.get("decoder") != expected_decoder_payload
        ):
            raise ValueError(
                f"PFCR Development seed {seed} is not current v2"
            )
        if result.get("scope") != {
            "model": "CURE-Lite",
            "stage": (
                "dataset-free learned relation decoder Development"
            ),
            "real_dataset_training": False,
            "dataset_metrics_read": False,
            "full_CURE_in_scope": False,
        }:
            raise ValueError(
                f"PFCR Development seed {seed} scope changed"
            )
        gates = result.get("gates")
        decision = result.get("decision")
        if (
            not isinstance(gates, Mapping)
            or not isinstance(gates.get("results"), Mapping)
            or any(
                value is not True
                for value in gates["results"].values()
            )
            or decision
            != {
                "development_learnability_pass": True,
                "learned_relation_mechanism_supported_on_fixed_population": (
                    True
                ),
                "real_dataset_training_implementation_authorized": True,
                "real_dataset_model_success_claimed": False,
                "full_CURE_authorized": False,
            }
        ):
            raise ValueError(
                f"PFCR Development seed {seed} did not pass"
            )
        file_hashes[seed] = file_sha256(path)
        result_fingerprints[seed] = fingerprint
    base = {
        "schema_version": "cure-lite-pfcr-development-authorization-v1",
        "training_algorithm_version": PFCR_TRAINING_ALGORITHM_VERSION,
        "seeds": [42, 43],
        "file_sha256_by_seed": {
            str(seed): file_hashes[seed] for seed in (42, 43)
        },
        "result_fingerprint_by_seed": {
            str(seed): result_fingerprints[seed]
            for seed in (42, 43)
        },
    }
    result = PFCRDevelopmentAuthorization(
        file_sha256_by_seed=file_hashes,
        result_fingerprint_by_seed=result_fingerprints,
        authorization_fingerprint=stable_fingerprint(base),
    )
    if result.canonical_payload() != base:
        raise AssertionError("PFCR Development authorization drifted")
    return result


def load_pfcr_real_preflight_authorization(
    directory: str | Path,
    *,
    expected_seed: int,
) -> PFCRRealPreflightAuthorization:
    """Verify the complete D_R preflight without reading D_V."""

    if expected_seed not in {42, 43}:
        raise ValueError("expected_seed must be 42 or 43")
    raw = Path(directory).expanduser()
    if raw.is_symlink():
        raise ValueError("PFCR preflight directory may not be a symlink")
    source = raw.resolve(strict=True)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("PFCR preflight path must be a regular directory")
    members = {path.name: path for path in source.iterdir()}
    expected_files = {
        "STARTED.json",
        "result.json",
        "COMPLETE.json",
    }
    if set(members) != expected_files or any(
        path.is_symlink() or not path.is_file()
        for path in members.values()
    ):
        raise ValueError("PFCR preflight file inventory is not canonical")
    started = _strict_json(
        members["STARTED.json"],
        name="PFCR preflight STARTED receipt",
    )
    result = _strict_json(
        members["result.json"],
        name="PFCR preflight result",
    )
    complete = _strict_json(
        members["COMPLETE.json"],
        name="PFCR preflight COMPLETE receipt",
    )
    started_fingerprint = _verify_fingerprint(
        started,
        field="receipt_fingerprint",
        name="PFCR preflight STARTED receipt",
    )
    result_fingerprint = _verify_fingerprint(
        result,
        field="result_fingerprint",
        name="PFCR preflight result",
    )
    complete_fingerprint = _verify_fingerprint(
        complete,
        field="complete_fingerprint",
        name="PFCR preflight COMPLETE receipt",
    )
    if (
        started.get("schema_version")
        != "cure-lite-pfcr-real-preflight-start-v1"
        or started.get("seed") != expected_seed
        or started.get("continuation_supported") is not False
        or started.get("incomplete_attempt_may_be_reused") is not False
    ):
        raise ValueError("PFCR preflight STARTED policy changed")
    input_files = started.get("input_files")
    if (
        not isinstance(input_files, Mapping)
        or set(input_files)
        != {"manifest", "state_index", "geometry_config", "p0_a1"}
    ):
        raise ValueError("PFCR preflight input inventory changed")
    for name, binding in input_files.items():
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str)
        ):
            raise ValueError(
                f"PFCR preflight input binding {name!r} is malformed"
            )
        path = Path(binding["path"])
        if (
            path.is_symlink()
            or not path.resolve(strict=True).is_file()
            or file_sha256(path) != binding["sha256"]
        ):
            raise ValueError(
                f"PFCR preflight input {name!r} changed"
            )
    if result.get("schema_version") != PFCR_REAL_PREFLIGHT_SCHEMA:
        raise ValueError("PFCR preflight result schema changed")
    if (
        result.get("config")
        != PFCRRealPreflightConfig(
            seed=expected_seed
        ).canonical_payload()
    ):
        raise ValueError("PFCR preflight config changed")
    if result.get("scope") != {
        "model": "CURE-Lite",
        "split_read": ["D_R"],
        "D_V_read": False,
        "D_T_read": False,
        "full_CURE_in_scope": False,
        "performance_evaluation": False,
    }:
        raise ValueError("PFCR preflight scope changed")
    if result.get("population") != {
        "D_R_samples": 160,
        "factual_targets": 32,
        "factual_sources": 24,
        "factual_no_miss_sources": 135,
        "lineage_safe_legal_targets": 206,
        "lineage_safe_legal_sources": 149,
        "excluded_legal_identities": [
            ["XDU486", 1, 1],
            ["XDU526", 1, 1],
            ["XDU965", 1, 1],
        ],
    }:
        raise ValueError("PFCR preflight real-state population changed")
    decoder = result.get("decoder")
    if (
        not isinstance(decoder, Mapping)
        or decoder.get("feature_channels") != 64
        or decoder.get("feature_stride") != 4
        or decoder.get("relation_dim") != 8
        or decoder.get("evidence_ceiling") != 10.0
    ):
        raise ValueError("PFCR preflight decoder contract changed")
    gates = result.get("gates")
    if not isinstance(gates, Mapping) or not gates or any(
        value is not True for value in gates.values()
    ):
        raise ValueError("PFCR preflight did not pass every execution gate")
    decision = result.get("decision")
    expected_decision = {
        "real_training_execution_preflight_pass": True,
        "formal_800_epoch_training_implementation_authorized": True,
        "real_dataset_model_success_claimed": False,
        "D_V_evaluation_authorized_by_this_receipt": False,
        "full_CURE_authorized": False,
    }
    if decision != expected_decision:
        raise ValueError("PFCR preflight decision is not an authorization")
    if (
        complete.get("schema_version")
        != "cure-lite-pfcr-real-preflight-complete-v1"
        or complete.get("result_file") != "result.json"
        or complete.get("result_file_sha256")
        != file_sha256(members["result.json"])
        or complete.get("result_fingerprint") != result_fingerprint
        or complete.get("decision") != expected_decision
        or complete.get("started_receipt_fingerprint")
        != started_fingerprint
    ):
        raise ValueError("PFCR preflight COMPLETE binding changed")
    return PFCRRealPreflightAuthorization(
        source_directory=source,
        started_file_sha256=file_sha256(members["STARTED.json"]),
        result_file_sha256=file_sha256(members["result.json"]),
        complete_file_sha256=file_sha256(members["COMPLETE.json"]),
        seed=expected_seed,
        result_fingerprint=result_fingerprint,
        complete_fingerprint=complete_fingerprint,
        cache_contract_fingerprint=_digest(
            result.get("cache_contract_fingerprint"),
            name="cache_contract_fingerprint",
        ),
        state_catalog_fingerprint=_digest(
            result.get("state_catalog_fingerprint"),
            name="state_catalog_fingerprint",
        ),
        lineage_allowlist_fingerprint=_digest(
            result.get("lineage_allowlist_fingerprint"),
            name="lineage_allowlist_fingerprint",
        ),
    )


@dataclass(frozen=True)
class PublishedPFCRRealFormalAttempt:
    """Strictly loaded completed PFCR training attempt."""

    root: Path
    seed: int
    artifact: LoadedPFCRRealDecoderArtifact
    run_receipt_fingerprint: str
    complete_fingerprint: str


def _normalized_binding(
    value: Mapping[str, str],
    *,
    name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not key
            or Path(key).is_absolute()
        ):
            raise ValueError(f"{name} path is invalid")
        result[key] = _digest(digest, name=f"{name}[{key!r}]")
    return dict(sorted(result.items()))


def _verify_live_binding(
    root: Path,
    binding: Mapping[str, str],
    *,
    name: str,
) -> None:
    for relative, expected_sha in binding.items():
        path = root / relative
        if (
            path.is_symlink()
            or not path.resolve(strict=True).is_file()
            or file_sha256(path) != expected_sha
        ):
            raise RuntimeError(
                f"{name} member {relative!r} changed during execution"
            )


def _attempt_inventory(root: Path) -> dict[str, str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != _COMPLETE_NAME
    )
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in paths
    }


def run_pfcr_real_formal_attempt(
    cache: PFCRRealCacheAdapter,
    catalog: PFCRRealStateCatalog,
    config: PFCRRealFormalTrainingConfig,
    authorization: PFCRRealPreflightAuthorization,
    development_authorization: PFCRDevelopmentAuthorization,
    *,
    output_dir: str | Path,
    device: torch.device | str,
    binding_root: str | Path,
    input_binding: Mapping[str, str],
    implementation_binding: Mapping[str, str],
    epoch_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> PublishedPFCRRealFormalAttempt:
    """Claim one fresh directory and publish only a finished PFCR model."""

    if not isinstance(cache, PFCRRealCacheAdapter):
        raise TypeError("cache must be PFCRRealCacheAdapter")
    if not isinstance(catalog, PFCRRealStateCatalog):
        raise TypeError("catalog must be PFCRRealStateCatalog")
    if not isinstance(config, PFCRRealFormalTrainingConfig):
        raise TypeError("config must be PFCRRealFormalTrainingConfig")
    if not isinstance(
        authorization,
        PFCRRealPreflightAuthorization,
    ):
        raise TypeError("authorization has the wrong type")
    if not isinstance(
        development_authorization,
        PFCRDevelopmentAuthorization,
    ):
        raise TypeError("development_authorization has the wrong type")
    if config.seed != authorization.seed:
        raise ValueError("PFCR formal and preflight seeds differ")
    if (
        cache.contract.contract_fingerprint
        != authorization.cache_contract_fingerprint
        or catalog.catalog_fingerprint
        != authorization.state_catalog_fingerprint
        or catalog.allowlist.allowlist_fingerprint
        != authorization.lineage_allowlist_fingerprint
    ):
        raise RuntimeError(
            "PFCR real cache/catalog differs from the passed preflight"
        )
    inputs = _normalized_binding(input_binding, name="input_binding")
    implementation = _normalized_binding(
        implementation_binding,
        name="implementation_binding",
    )
    raw_binding_root = Path(binding_root).expanduser()
    if raw_binding_root.is_symlink():
        raise ValueError("binding_root may not be a symbolic link")
    resolved_binding_root = raw_binding_root.resolve(strict=True)
    if (
        not resolved_binding_root.is_dir()
        or resolved_binding_root.is_symlink()
    ):
        raise ValueError("binding_root must be a regular directory")
    _verify_live_binding(
        resolved_binding_root,
        inputs,
        name="input binding",
    )
    _verify_live_binding(
        resolved_binding_root,
        implementation,
        name="implementation binding",
    )
    refreshed_authorization = load_pfcr_real_preflight_authorization(
        authorization.source_directory,
        expected_seed=config.seed,
    )
    if refreshed_authorization != authorization:
        raise RuntimeError(
            "PFCR real preflight changed during formal training"
        )
    schedule_payload = pfcr_real_formal_schedule_payload(catalog, config)
    schedule_fingerprint = stable_fingerprint(schedule_payload)
    target = Path(output_dir).expanduser().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"refusing to reuse PFCR formal attempt {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    started = _fingerprinted(
        {
            "schema_version": PFCR_REAL_FORMAL_ATTEMPT_SCHEMA,
            "model": "CURE-Lite",
            "seed": config.seed,
            "training_config": config.canonical_payload(),
            "cache_contract_fingerprint": (
                cache.contract.contract_fingerprint
            ),
            "state_catalog_fingerprint": (
                catalog.catalog_fingerprint
            ),
            "lineage_allowlist_fingerprint": (
                catalog.allowlist.allowlist_fingerprint
            ),
            "formal_schedule": schedule_payload,
            "formal_schedule_fingerprint": schedule_fingerprint,
            "preflight": {
                "source_directory": str(
                    authorization.source_directory
                ),
                "started_file_sha256": (
                    authorization.started_file_sha256
                ),
                "result_file_sha256": (
                    authorization.result_file_sha256
                ),
                "complete_file_sha256": (
                    authorization.complete_file_sha256
                ),
                "result_fingerprint": (
                    authorization.result_fingerprint
                ),
                "complete_fingerprint": (
                    authorization.complete_fingerprint
                ),
            },
            "development_authorization": {
                **development_authorization.canonical_payload(),
                "authorization_fingerprint": (
                    development_authorization.authorization_fingerprint
                ),
            },
            "input_binding": inputs,
            "implementation_binding": implementation,
            "execution_policy": {
                "runtime_splits": ["D_R"],
                "D_V_read": False,
                "D_T_read": False,
                "fresh_initialization": True,
                "create_only_output": True,
                "continuation_supported": False,
                "checkpoint_written": False,
                "intermediate_optimizer_state_written": False,
                "incomplete_attempt_may_be_reused": False,
                "incomplete_attempt_may_be_evaluated": False,
                "complete_written_last": True,
            },
        },
        field="started_fingerprint",
    )
    _write_new_json(target / _STARTED_NAME, started)

    training = execute_pfcr_real_formal_training(
        cache,
        catalog,
        config,
        device=device,
        epoch_callback=epoch_callback,
    )
    ledger = training.execution_ledger
    run_config = PFCRRealDecoderRunConfig(
        seed=config.seed,
        cache_contract_fingerprint=(
            cache.contract.contract_fingerprint
        ),
        state_catalog_fingerprint=catalog.catalog_fingerprint,
        lineage_allowlist_fingerprint=(
            catalog.allowlist.allowlist_fingerprint
        ),
        formal_schedule_fingerprint=schedule_fingerprint,
        preflight_result_fingerprint=(
            authorization.result_fingerprint
        ),
        initial_model_fingerprint=(
            ledger.initial_model_fingerprint
        ),
        decoder_config=training.decoder.config,
        training_config=config,
    )
    artifact_fingerprint = save_pfcr_real_decoder_artifact(
        target / _ARTIFACT_DIR,
        training.decoder,
        run_config,
        training.epoch_logs,
        ledger,
    )
    artifact = load_pfcr_real_decoder_artifact(
        target / _ARTIFACT_DIR
    )
    if artifact.artifact_fingerprint != artifact_fingerprint:
        raise RuntimeError("PFCR artifact verification changed its identity")
    _verify_live_binding(
        resolved_binding_root,
        inputs,
        name="input binding",
    )
    _verify_live_binding(
        resolved_binding_root,
        implementation,
        name="implementation binding",
    )
    refreshed_authorization = load_pfcr_real_preflight_authorization(
        authorization.source_directory,
        expected_seed=config.seed,
    )
    if refreshed_authorization != authorization:
        raise RuntimeError(
            "PFCR real preflight changed during formal training"
        )
    run_receipt = _fingerprinted(
        {
            "schema_version": PFCR_REAL_FORMAL_ATTEMPT_SCHEMA,
            "seed": config.seed,
            "started_fingerprint": started["started_fingerprint"],
            "artifact_directory": _ARTIFACT_DIR,
            "artifact_fingerprint": artifact_fingerprint,
            "artifact_receipt_sha256": artifact.receipt_sha256,
            "execution_ledger": ledger.canonical_payload(),
            "formal_training_complete": True,
            "optimizer_state_saved": False,
            "checkpoint_saved": False,
            "D_V_read": False,
            "D_T_read": False,
            "performance_success_claimed": False,
            "D_V_evaluation_authorized": True,
            "full_CURE_authorized": False,
        },
        field="run_receipt_fingerprint",
    )
    _write_new_json(target / _RUN_RECEIPT_NAME, run_receipt)
    inventory = _attempt_inventory(target)
    complete = _fingerprinted(
        {
            "schema_version": PFCR_REAL_FORMAL_COMPLETE_SCHEMA,
            "seed": config.seed,
            "started_fingerprint": started["started_fingerprint"],
            "run_receipt_fingerprint": (
                run_receipt["run_receipt_fingerprint"]
            ),
            "artifact_fingerprint": artifact_fingerprint,
            "inventory_sha256": inventory,
            "decision": {
                "formal_800_by_40_training_complete": True,
                "artifact_strictly_loadable": True,
                "D_V_evaluation_authorized": True,
                "performance_success_claimed": False,
                "full_CURE_authorized": False,
            },
            "complete_written_after_all_inventory_members": True,
        },
        field="complete_fingerprint",
    )
    _write_new_json(target / _COMPLETE_NAME, complete)
    return load_pfcr_real_formal_attempt(target)


def load_pfcr_real_formal_attempt(
    directory: str | Path,
) -> PublishedPFCRRealFormalAttempt:
    """Load only an exact completed attempt; STARTED-only attempts fail."""

    raw = Path(directory).expanduser()
    if raw.is_symlink():
        raise ValueError("PFCR formal attempt may not be a symlink")
    root = raw.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("PFCR formal attempt must be a regular directory")
    members = {path.name: path for path in root.iterdir()}
    expected = {
        _STARTED_NAME,
        _RUN_RECEIPT_NAME,
        _ARTIFACT_DIR,
        _COMPLETE_NAME,
    }
    if set(members) != expected:
        raise ValueError(
            "PFCR formal attempt is incomplete or has extra members"
        )
    if (
        members[_ARTIFACT_DIR].is_symlink()
        or not members[_ARTIFACT_DIR].is_dir()
        or any(
            members[name].is_symlink()
            or not members[name].is_file()
            for name in (
                _STARTED_NAME,
                _RUN_RECEIPT_NAME,
                _COMPLETE_NAME,
            )
        )
    ):
        raise ValueError("PFCR formal attempt member type changed")
    started = _strict_json(
        members[_STARTED_NAME],
        name="PFCR formal STARTED receipt",
    )
    run_receipt = _strict_json(
        members[_RUN_RECEIPT_NAME],
        name="PFCR formal run receipt",
    )
    complete = _strict_json(
        members[_COMPLETE_NAME],
        name="PFCR formal COMPLETE receipt",
    )
    if set(started) != {
        "schema_version",
        "model",
        "seed",
        "training_config",
        "cache_contract_fingerprint",
        "state_catalog_fingerprint",
        "lineage_allowlist_fingerprint",
        "formal_schedule",
        "formal_schedule_fingerprint",
        "preflight",
        "development_authorization",
        "input_binding",
        "implementation_binding",
        "execution_policy",
        "started_fingerprint",
    }:
        raise ValueError("PFCR formal STARTED fields are not canonical")
    if set(run_receipt) != {
        "schema_version",
        "seed",
        "started_fingerprint",
        "artifact_directory",
        "artifact_fingerprint",
        "artifact_receipt_sha256",
        "execution_ledger",
        "formal_training_complete",
        "optimizer_state_saved",
        "checkpoint_saved",
        "D_V_read",
        "D_T_read",
        "performance_success_claimed",
        "D_V_evaluation_authorized",
        "full_CURE_authorized",
        "run_receipt_fingerprint",
    }:
        raise ValueError("PFCR formal run receipt fields are not canonical")
    if set(complete) != {
        "schema_version",
        "seed",
        "started_fingerprint",
        "run_receipt_fingerprint",
        "artifact_fingerprint",
        "inventory_sha256",
        "decision",
        "complete_written_after_all_inventory_members",
        "complete_fingerprint",
    }:
        raise ValueError("PFCR formal COMPLETE fields are not canonical")
    started_fingerprint = _verify_fingerprint(
        started,
        field="started_fingerprint",
        name="PFCR formal STARTED receipt",
    )
    run_receipt_fingerprint = _verify_fingerprint(
        run_receipt,
        field="run_receipt_fingerprint",
        name="PFCR formal run receipt",
    )
    complete_fingerprint = _verify_fingerprint(
        complete,
        field="complete_fingerprint",
        name="PFCR formal COMPLETE receipt",
    )
    if (
        started.get("schema_version")
        != PFCR_REAL_FORMAL_ATTEMPT_SCHEMA
        or run_receipt.get("schema_version")
        != PFCR_REAL_FORMAL_ATTEMPT_SCHEMA
        or complete.get("schema_version")
        != PFCR_REAL_FORMAL_COMPLETE_SCHEMA
    ):
        raise ValueError("PFCR formal attempt schema changed")
    seed = started.get("seed")
    if (
        seed not in {42, 43}
        or run_receipt.get("seed") != seed
        or complete.get("seed") != seed
        or run_receipt.get("started_fingerprint")
        != started_fingerprint
        or complete.get("started_fingerprint")
        != started_fingerprint
        or complete.get("run_receipt_fingerprint")
        != run_receipt_fingerprint
    ):
        raise ValueError("PFCR formal attempt seed/receipt binding changed")
    expected_inventory = complete.get("inventory_sha256")
    if (
        not isinstance(expected_inventory, Mapping)
        or dict(expected_inventory) != _attempt_inventory(root)
    ):
        raise ValueError("PFCR formal attempt inventory changed")
    artifact = load_pfcr_real_decoder_artifact(
        members[_ARTIFACT_DIR]
    )
    if (
        started.get("model") != "CURE-Lite"
        or started.get("training_config")
        != artifact.config.training_config.canonical_payload()
        or started.get("cache_contract_fingerprint")
        != artifact.config.cache_contract_fingerprint
        or started.get("state_catalog_fingerprint")
        != artifact.config.state_catalog_fingerprint
        or started.get("lineage_allowlist_fingerprint")
        != artifact.config.lineage_allowlist_fingerprint
        or started.get("formal_schedule_fingerprint")
        != artifact.config.formal_schedule_fingerprint
    ):
        raise ValueError("PFCR STARTED and artifact run config differ")
    schedule = started.get("formal_schedule")
    if (
        not isinstance(schedule, Mapping)
        or stable_fingerprint(schedule)
        != started["formal_schedule_fingerprint"]
        or schedule.get("seed") != seed
        or schedule.get("optimizer_updates") != 32_000
        or schedule.get("decoder_forwards_per_update") != 1
        or schedule.get("decoder_states_per_update") != 12
    ):
        raise ValueError("PFCR formal schedule binding changed")
    preflight = started.get("preflight")
    if (
        not isinstance(preflight, Mapping)
        or set(preflight)
        != {
            "source_directory",
            "started_file_sha256",
            "result_file_sha256",
            "complete_file_sha256",
            "result_fingerprint",
            "complete_fingerprint",
        }
        or preflight.get("result_fingerprint")
        != artifact.config.preflight_result_fingerprint
    ):
        raise ValueError("PFCR formal preflight binding changed")
    development = started.get("development_authorization")
    if not isinstance(development, Mapping):
        raise ValueError("PFCR Development authorization is malformed")
    development_core = dict(development)
    development_fingerprint = development_core.pop(
        "authorization_fingerprint",
        None,
    )
    if (
        not isinstance(development_fingerprint, str)
        or stable_fingerprint(development_core)
        != development_fingerprint
        or development_core.get("training_algorithm_version")
        != PFCR_TRAINING_ALGORITHM_VERSION
        or development_core.get("seeds") != [42, 43]
    ):
        raise ValueError("PFCR Development authorization changed")
    _normalized_binding(
        started.get("input_binding"),
        name="STARTED input binding",
    )
    _normalized_binding(
        started.get("implementation_binding"),
        name="STARTED implementation binding",
    )
    expected_policy = {
        "runtime_splits": ["D_R"],
        "D_V_read": False,
        "D_T_read": False,
        "fresh_initialization": True,
        "create_only_output": True,
        "continuation_supported": False,
        "checkpoint_written": False,
        "intermediate_optimizer_state_written": False,
        "incomplete_attempt_may_be_reused": False,
        "incomplete_attempt_may_be_evaluated": False,
        "complete_written_last": True,
    }
    if started.get("execution_policy") != expected_policy:
        raise ValueError("PFCR formal execution policy changed")
    if (
        run_receipt.get("artifact_directory") != _ARTIFACT_DIR
        or run_receipt.get("artifact_fingerprint")
        != artifact.artifact_fingerprint
        or run_receipt.get("artifact_receipt_sha256")
        != artifact.receipt_sha256
        or complete.get("artifact_fingerprint")
        != artifact.artifact_fingerprint
        or artifact.config.seed != seed
        or run_receipt.get("execution_ledger")
        != artifact.execution_ledger.canonical_payload()
    ):
        raise ValueError("PFCR formal artifact/receipt binding changed")
    expected_decision = {
        "formal_800_by_40_training_complete": True,
        "artifact_strictly_loadable": True,
        "D_V_evaluation_authorized": True,
        "performance_success_claimed": False,
        "full_CURE_authorized": False,
    }
    if (
        complete.get("decision") != expected_decision
        or complete.get(
            "complete_written_after_all_inventory_members"
        )
        is not True
        or run_receipt.get("formal_training_complete") is not True
        or run_receipt.get("optimizer_state_saved") is not False
        or run_receipt.get("checkpoint_saved") is not False
        or run_receipt.get("D_V_read") is not False
        or run_receipt.get("D_T_read") is not False
        or run_receipt.get("performance_success_claimed") is not False
        or run_receipt.get("D_V_evaluation_authorized") is not True
        or run_receipt.get("full_CURE_authorized") is not False
    ):
        raise ValueError("PFCR formal completion decision changed")
    return PublishedPFCRRealFormalAttempt(
        root=root,
        seed=seed,
        artifact=artifact,
        run_receipt_fingerprint=run_receipt_fingerprint,
        complete_fingerprint=complete_fingerprint,
    )


__all__ = [
    "PFCR_REAL_FORMAL_ATTEMPT_SCHEMA",
    "PFCR_REAL_FORMAL_COMPLETE_SCHEMA",
    "PFCRRealPreflightAuthorization",
    "PublishedPFCRRealFormalAttempt",
    "load_pfcr_real_formal_attempt",
    "load_pfcr_real_preflight_authorization",
    "run_pfcr_real_formal_attempt",
]
