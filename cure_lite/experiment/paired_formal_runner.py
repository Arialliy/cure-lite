"""Create-only, no-resume runner for formal paired CURE-Lite training.

This module owns the execution boundary around the already-frozen formal
schedule and training engine.  It deliberately has no evaluation, calibration,
inference, wave-decision, checkpoint, resume, or shortened-horizon interface.
An attempt directory is claimed before optimization starts.  A failed attempt
remains marked incomplete and can never be reused; a new directory is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..config import DecoderConfig, LossConfig
from ..decoder import CURELiteDecoder
from ..losses import CURELiteLoss
from ..paired_losses import PairedDifferenceLoss
from ..paired_types import (
    PairCatalog,
    tensor_content_fingerprint,
)
from .artifacts import decoder_state_fingerprint
from .cache_pipeline import LoadedDRCacheBundle
from .paired_artifacts import (
    PAIRED_METHODS,
    LoadedPairedDecoderArtifact,
    PairedDecoderRunConfig,
    load_paired_decoder_artifact,
    method_objective_contract,
    save_paired_decoder_artifact,
)
from .paired_formal_controls import PairedFormalControlInputProvider
from .paired_formal_preflight import PublishedPairedFormalPreflight
from .paired_formal_schedule import (
    FORMAL_METHOD_KINDS,
    PairedFormalSchedule,
    prepared_training_catalog_fingerprint,
)
from .paired_formal_training import (
    PAIRED_DIFFERENCE_METHOD,
    PairedFormalTrainingResult,
    execute_paired_formal_training,
)
from .training_pipeline import PreparedTrainingCatalog


PAIRED_FORMAL_RUNNER_CONFIG_SCHEMA = (
    "cure-lite-paired-formal-runner-config-v1"
)
PAIRED_FORMAL_ATTEMPT_SCHEMA = "cure-lite-paired-formal-attempt-v1"
PAIRED_FORMAL_ATTEMPT_COMPLETE_SCHEMA = (
    "cure-lite-paired-formal-attempt-complete-v1"
)
PAIRED_FORMAL_PROVIDER_RECEIPT_SCHEMA = (
    "cure-lite-paired-formal-provider-use-v1"
)

FORMAL_RUNNER_SEEDS = (42, 43)
FORMAL_RUNNER_METHODS = FORMAL_METHOD_KINDS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INCOMPLETE_NAME = ".INCOMPLETE.json"
_ARTIFACT_DIR = "decoder_artifact"
_PROVIDER_NAME = "control_provider_receipt.json"
_RUN_RECEIPT_NAME = "run_receipt.json"
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
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


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
    payload: Mapping[str, object],
    *,
    field: str,
) -> dict[str, object]:
    core = dict(payload)
    if field in core:
        raise ValueError(f"{field} already exists")
    return {**core, field: stable_fingerprint(core)}


def _verify_fingerprint(
    payload: Mapping[str, object],
    *,
    field: str,
    name: str,
) -> str:
    core = dict(payload)
    fingerprint = core.pop(field, None)
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(core) != fingerprint
    ):
        raise RuntimeError(f"{name} fingerprint mismatch")
    return _digest(fingerprint, name=f"{name} fingerprint")


def _exact_mapping(
    value: object,
    *,
    expected: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise ValueError(f"{name} differs from the frozen protocol")
    return dict(value)


def validate_paired_formal_runner_config(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate the horizon-free production protocol without touching data."""

    if not isinstance(payload, Mapping):
        raise TypeError("paired formal runner config must be a mapping")
    config = dict(payload)
    expected_fields = {
        "schema_version",
        "protocol_id",
        "dataset",
        "training_split",
        "seeds",
        "methods",
        "preflight_binding",
        "control_preflight_binding",
        "paired_objective_binding",
        "catalog_reconstruction_binding",
        "evaluation_protocol_binding",
        "model",
        "optimization",
        "budget",
        "initial_decoder_fingerprints",
        "runtime_input_fingerprints",
        "implementation_binding",
        "execution_policy",
        "config_fingerprint_scope",
        "config_fingerprint",
    }
    if set(config) != expected_fields:
        raise ValueError("paired formal runner config fields are not canonical")
    unsigned = dict(config)
    fingerprint = unsigned.pop("config_fingerprint")
    if stable_fingerprint(unsigned) != fingerprint:
        raise ValueError("paired formal runner config fingerprint mismatch")
    _digest(fingerprint, name="config_fingerprint")
    if (
        config["schema_version"] != PAIRED_FORMAL_RUNNER_CONFIG_SCHEMA
        or config["protocol_id"] != "irstd1k-dr-paired-formal-runner-v1"
        or config["dataset"] != "IRSTD-1K"
        or config["training_split"] != "D_R"
        or config["seeds"] != list(FORMAL_RUNNER_SEEDS)
        or config["methods"] != list(FORMAL_RUNNER_METHODS)
        or config["config_fingerprint_scope"]
        != "all-fields-except-config_fingerprint"
    ):
        raise ValueError("paired formal runner protocol identity changed")

    preflight = config["preflight_binding"]
    control = config["control_preflight_binding"]
    objective = config["paired_objective_binding"]
    reconstruction = config["catalog_reconstruction_binding"]
    evaluation = config["evaluation_protocol_binding"]
    for section, fields, name in (
        (
            preflight,
            {
                "repo_path",
                "complete_file_sha256",
                "complete_fingerprint",
                "config_fingerprint",
                "method_bindings_fingerprint",
                "pair_catalog_fingerprint",
                "seed42_formal_schedule_fingerprint",
                "seed43_formal_schedule_fingerprint",
            },
            "preflight binding",
        ),
        (
            control,
            {
                "repo_path",
                "complete_file_sha256",
                "complete_fingerprint",
                "pair_catalog_fingerprint",
                "dct_basis_fingerprint",
                "target_permutation_plan_fingerprint",
                "target_assignment_fingerprint",
                "provider_fingerprint",
            },
            "control preflight binding",
        ),
        (
            objective,
            {"repo_path", "file_sha256", "receipt_fingerprint"},
            "paired objective binding",
        ),
        (
            reconstruction,
            {"repo_path", "file_sha256", "config_fingerprint"},
            "catalog reconstruction binding",
        ),
        (
            evaluation,
            {
                "repo_path",
                "file_sha256",
                "comparison_protocol_fingerprint",
            },
            "evaluation protocol binding",
        ),
    ):
        if not isinstance(section, Mapping) or set(section) != fields:
            raise ValueError(f"{name} fields are not canonical")
        path = section.get("repo_path")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError(f"{name} repo_path is invalid")
        for key, value in section.items():
            if key != "repo_path":
                _digest(value, name=f"{name}.{key}")
    if (
        preflight["pair_catalog_fingerprint"]
        != control["pair_catalog_fingerprint"]
    ):
        raise ValueError("formal and control preflights bind different catalogs")

    model = config["model"]
    if not isinstance(model, Mapping) or set(model) != {
        "decoder_config",
        "absolute_loss_config",
        "paired_loss_id",
    }:
        raise ValueError("formal model section is not canonical")
    if model["decoder_config"] != {
        "feature_channels": 64,
        "width": 32,
        "groups": 8,
    } or model["absolute_loss_config"] != {
        "dice_weight": 1.0,
        "epsilon": 1e-6,
    } or model["paired_loss_id"] != (
        "balanced_pre_mask_score_difference_regression_v1"
    ):
        raise ValueError("formal model/loss definition changed")

    _exact_mapping(
        config["optimization"],
        expected={
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "initialization_device": "cpu",
            "same_seed_same_initial_bytes_across_methods": True,
        },
        name="optimization",
    )
    _exact_mapping(
        config["budget"],
        expected={
            "epochs": 800,
            "steps_per_epoch": 40,
            "optimizer_updates": 32_000,
            "factual_miss_states_per_update": 4,
            "factual_no_miss_states_per_update": 4,
            "clean_pairs_per_update": 2,
            "paired_endpoint_states_per_update": 4,
            "decoder_states_per_update": 12,
            "decoder_forwards_per_update": 3,
            "backward_calls": 32_000,
            "optimizer_steps": 32_000,
        },
        name="budget",
    )
    initial = config["initial_decoder_fingerprints"]
    if not isinstance(initial, Mapping) or set(initial) != {"42", "43"}:
        raise ValueError("initial decoder fingerprint inventory changed")
    for seed, value in initial.items():
        _digest(value, name=f"seed-{seed} initial decoder fingerprint")
    runtime_inputs = config["runtime_input_fingerprints"]
    if (
        not isinstance(runtime_inputs, Mapping)
        or set(runtime_inputs) != {"42", "43"}
    ):
        raise ValueError("runtime input fingerprint inventory changed")
    for seed, value in runtime_inputs.items():
        _digest(value, name=f"seed-{seed} runtime input fingerprint")
    implementation = config["implementation_binding"]
    if not isinstance(implementation, Mapping) or not implementation:
        raise ValueError("implementation binding must be non-empty")
    for relative, digest in implementation.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise ValueError("implementation path is invalid")
        _digest(digest, name=f"implementation SHA for {relative}")
    _exact_mapping(
        config["execution_policy"],
        expected={
            "create_only_output": True,
            "failed_attempt_directory_reuse": False,
            "checkpoint_written": False,
            "resume": False,
            "overwrite": False,
            "partial_horizon": False,
            "allow_scientific_overrides": False,
            "runtime_splits": ["D_R"],
            "allow_D_V": False,
            "allow_D_T": False,
            "allow_calibration": False,
            "allow_inference": False,
            "allow_wave_decision": False,
            "complete_written_last": True,
        },
        name="execution policy",
    )
    return config


@dataclass(frozen=True, slots=True)
class _LoadedRunnerConfigSeal:
    source_path: Path
    source_sha256: str
    payload: dict[str, object]


@dataclass(frozen=True)
class LoadedPairedFormalRunnerConfig:
    """Strictly loaded immutable formal-runner protocol."""

    source_path: Path
    source_sha256: str
    payload: Mapping[str, object]
    config_fingerprint: str
    _verification_token: object

    def _verify_seal(self) -> _LoadedRunnerConfigSeal:
        seal = self._verification_token
        if type(seal) is not _LoadedRunnerConfigSeal:
            raise TypeError("formal runner config must come from the strict loader")
        if (
            seal.source_path != self.source_path
            or seal.source_sha256 != self.source_sha256
            or seal.payload is not self.payload
        ):
            raise TypeError("loaded formal runner config fields were replaced")
        return seal

    def __post_init__(self) -> None:
        self._verify_seal()
        if self.config_fingerprint != self.payload["config_fingerprint"]:
            raise ValueError("loaded config fingerprint differs from its payload")
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        seal = self._verify_seal()
        if (
            self.source_path.is_symlink()
            or file_sha256(self.source_path) != self.source_sha256
            or _strict_json(
                self.source_path,
                name="paired formal runner config",
            )
            != seal.payload
        ):
            raise RuntimeError("paired formal runner config changed on disk")
        validate_paired_formal_runner_config(self.payload)
        for relative, expected in self.payload[
            "implementation_binding"
        ].items():
            path = (_REPO_ROOT / str(relative)).resolve(strict=True)
            if (
                path.is_symlink()
                or path.relative_to(_REPO_ROOT).as_posix() != relative
                or not path.is_file()
                or file_sha256(path) != expected
            ):
                raise RuntimeError(
                    f"formal runner implementation changed: {relative}"
                )


def load_paired_formal_runner_config(
    path: str | Path,
) -> LoadedPairedFormalRunnerConfig:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError("formal runner config may not be a symbolic link")
    source = candidate.resolve(strict=True)
    payload = validate_paired_formal_runner_config(
        _strict_json(source, name="paired formal runner config")
    )
    source_sha = file_sha256(source)
    seal = _LoadedRunnerConfigSeal(source, source_sha, payload)
    return LoadedPairedFormalRunnerConfig(
        source_path=source,
        source_sha256=source_sha,
        payload=payload,
        config_fingerprint=str(payload["config_fingerprint"]),
        _verification_token=seal,
    )


@dataclass(frozen=True)
class PairedFormalRuntimeInputs:
    """Strict D_R-only objects reconstructed before an attempt starts."""

    bundle: LoadedDRCacheBundle
    pair_catalog: PairCatalog
    prepared_catalog: PreparedTrainingCatalog
    schedule: PairedFormalSchedule
    control_provider: PairedFormalControlInputProvider | None


def _factual_anchor_runtime_payload(
    anchor: object,
) -> dict[str, object]:
    """Hash one factual anchor's complete tensor state and role metadata."""

    # The exact class check is intentionally avoided here only to keep this
    # helper private to PairedFormalSchedule's already-validated populations.
    example = anchor.example
    supervision = example.supervision
    supervision.validate()
    return {
        "anchor_metadata": anchor.canonical_payload(),
        "state_metadata": {
            "sample_id": example.sample_id,
            "branch": supervision.branch,
            "positive_gt_ids": list(supervision.positive_gt_ids),
            "reachable_gt_ids": list(supervision.reachable_gt_ids),
            "unreachable_gt_ids": list(supervision.unreachable_gt_ids),
        },
        "tensor_fingerprints": {
            "feature": tensor_content_fingerprint(example.feature),
            "occupancy": tensor_content_fingerprint(
                supervision.occupancy
            ),
            "target": tensor_content_fingerprint(supervision.target),
            "valid_mask": tensor_content_fingerprint(
                supervision.valid_mask
            ),
        },
    }


def paired_formal_runtime_input_payload(
    pair_catalog: PairCatalog,
    schedule: PairedFormalSchedule,
) -> dict[str, object]:
    """Recompute the complete seed-specific training-input identity.

    This complements the historical identity-only schedule fingerprint.  It
    dynamically consumes every factual tensor and the complete PairCatalog
    canonical payload immediately before and after formal optimization.
    """

    if not isinstance(pair_catalog, PairCatalog):
        raise TypeError("pair_catalog must be PairCatalog")
    if not isinstance(schedule, PairedFormalSchedule):
        raise TypeError("schedule must be PairedFormalSchedule")
    catalog_payload = pair_catalog.canonical_payload()
    catalog_fingerprint = stable_fingerprint(catalog_payload)
    if catalog_fingerprint != pair_catalog.catalog_fingerprint:
        raise RuntimeError("PairCatalog canonical payload changed in memory")
    all_pairs = (
        *pair_catalog.clean_positive,
        *pair_catalog.component_null,
        *pair_catalog.identity_null,
    )
    pair_feature_fingerprints = []
    for pair in all_pairs:
        live = tensor_content_fingerprint(pair.feature)
        if live != pair.feature_fingerprint:
            raise RuntimeError("PairCatalog feature tensor changed in memory")
        pair_feature_fingerprints.append(
            {
                "pair_id": pair.pair_id,
                "pair_kind": pair.pair_kind,
                "sample_id": pair.sample_id,
                "feature_fingerprint": live,
            }
        )
    catalog_clean_by_id = {
        pair.pair_id: pair.canonical_payload()
        for pair in pair_catalog.clean_positive
    }
    schedule_pair_by_id: dict[str, dict[str, object]] = {}
    schedule_pair_feature_fingerprints: list[dict[str, str]] = []
    for pair in schedule.paired_schedule.pairs:
        live = tensor_content_fingerprint(pair.feature)
        if live != pair.feature_fingerprint:
            raise RuntimeError(
                "formal schedule pair feature tensor changed in memory"
            )
        if pair.pair_id in schedule_pair_by_id:
            raise RuntimeError("formal schedule contains duplicate pair IDs")
        schedule_pair_by_id[pair.pair_id] = pair.canonical_payload()
        schedule_pair_feature_fingerprints.append(
            {
                "pair_id": pair.pair_id,
                "feature_fingerprint": live,
            }
        )
    if (
        set(schedule_pair_by_id) != set(catalog_clean_by_id)
        or any(
            schedule_pair_by_id[pair_id]
            != catalog_clean_by_id[pair_id]
            for pair_id in catalog_clean_by_id
        )
    ):
        raise RuntimeError(
            "actual formal schedule pairs differ from PairCatalog.clean_positive"
        )
    schedule_pair_payloads = [
        schedule_pair_by_id[pair_id]
        for pair_id in sorted(schedule_pair_by_id)
    ]
    schedule_pair_feature_fingerprints.sort(
        key=lambda row: row["pair_id"]
    )
    return {
        "schema_version": "cure-lite-paired-formal-runtime-input-v1",
        "dataset": pair_catalog.dataset,
        "split": pair_catalog.split,
        "seed": schedule.seed,
        "formal_schedule_fingerprint": schedule.schedule_fingerprint,
        "paired_schedule_fingerprint": (
            schedule.paired_schedule.schedule_fingerprint
        ),
        "pair_catalog_fingerprint": catalog_fingerprint,
        "pair_catalog_canonical_payload": catalog_payload,
        "pair_feature_runtime_fingerprints": pair_feature_fingerprints,
        "actual_schedule_pair_payloads": schedule_pair_payloads,
        "actual_schedule_pair_feature_fingerprints": (
            schedule_pair_feature_fingerprints
        ),
        "factual_miss_anchors": [
            _factual_anchor_runtime_payload(anchor)
            for anchor in schedule.factual_miss_anchors
        ],
        "factual_no_miss_anchors": [
            _factual_anchor_runtime_payload(anchor)
            for anchor in schedule.factual_no_miss_anchors
        ],
    }


def paired_formal_runtime_input_fingerprint(
    pair_catalog: PairCatalog,
    schedule: PairedFormalSchedule,
) -> str:
    return stable_fingerprint(
        paired_formal_runtime_input_payload(pair_catalog, schedule)
    )


def _bound_repo_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is invalid")
    path = (_REPO_ROOT / value).resolve(strict=True)
    if path.is_symlink() or path.relative_to(_REPO_ROOT).as_posix() != value:
        raise RuntimeError(f"{name} is not the frozen repository path")
    return path


def validate_preflight_binding(
    config: LoadedPairedFormalRunnerConfig,
    preflight: PublishedPairedFormalPreflight,
) -> None:
    """Bind the strict loaded schedule preflight to the runner protocol."""

    config.verify_unchanged()
    if not isinstance(preflight, PublishedPairedFormalPreflight):
        raise TypeError("preflight must be a strictly loaded published preflight")
    preflight.verify_unchanged()
    binding = config.payload["preflight_binding"]
    assert isinstance(binding, Mapping)
    expected_root = _bound_repo_path(
        binding["repo_path"],
        name="formal preflight root",
    )
    if (
        preflight.root != expected_root
        or file_sha256(expected_root / "COMPLETE.json")
        != binding["complete_file_sha256"]
        or preflight.complete_fingerprint
        != binding["complete_fingerprint"]
        or preflight.config_fingerprint != binding["config_fingerprint"]
        or preflight.method_bindings_fingerprint
        != binding["method_bindings_fingerprint"]
        or preflight.pair_catalog_fingerprint
        != binding["pair_catalog_fingerprint"]
        or preflight.seed42_formal_schedule_fingerprint
        != binding["seed42_formal_schedule_fingerprint"]
        or preflight.seed43_formal_schedule_fingerprint
        != binding["seed43_formal_schedule_fingerprint"]
    ):
        raise RuntimeError("formal schedule preflight differs from the freeze")


def _validate_runtime(
    config: LoadedPairedFormalRunnerConfig,
    preflight: PublishedPairedFormalPreflight,
    runtime: PairedFormalRuntimeInputs,
    *,
    method: str,
    seed: int,
) -> str:
    validate_preflight_binding(config, preflight)
    if not isinstance(runtime, PairedFormalRuntimeInputs):
        raise TypeError("runtime must be PairedFormalRuntimeInputs")
    if method not in FORMAL_RUNNER_METHODS or method not in PAIRED_METHODS:
        raise ValueError("formal method is not in the frozen inventory")
    if seed not in FORMAL_RUNNER_SEEDS:
        raise ValueError("formal seed must be 42 or 43")
    if not isinstance(runtime.bundle, LoadedDRCacheBundle):
        raise TypeError("runtime bundle must come from the strict D_R loader")
    runtime.bundle.verify_unchanged()
    if runtime.bundle.split != "D_R":
        raise ValueError("formal training may use only D_R")
    if (
        not isinstance(runtime.pair_catalog, PairCatalog)
        or runtime.pair_catalog.dataset != "IRSTD-1K"
        or runtime.pair_catalog.split != "D_R"
        or runtime.pair_catalog.catalog_fingerprint
        != preflight.pair_catalog_fingerprint
    ):
        raise RuntimeError("runtime pair catalog differs from the formal freeze")
    if not isinstance(runtime.prepared_catalog, PreparedTrainingCatalog):
        raise TypeError("runtime prepared catalog has an invalid type")
    prepared_fingerprint = prepared_training_catalog_fingerprint(
        runtime.prepared_catalog
    )
    if (
        not isinstance(runtime.schedule, PairedFormalSchedule)
        or runtime.schedule.seed != seed
        or runtime.schedule.prepared_catalog_fingerprint
        != prepared_fingerprint
        or runtime.schedule.paired_schedule.catalog_fingerprint
        != runtime.pair_catalog.catalog_fingerprint
    ):
        raise RuntimeError("runtime schedule differs from its D_R catalogs")
    expected_schedule = (
        preflight.seed42_formal_schedule_fingerprint
        if seed == 42
        else preflight.seed43_formal_schedule_fingerprint
    )
    if runtime.schedule.schedule_fingerprint != expected_schedule:
        raise RuntimeError("runtime schedule differs from the formal preflight")
    runtime_input_fingerprint = paired_formal_runtime_input_fingerprint(
        runtime.pair_catalog,
        runtime.schedule,
    )
    if runtime_input_fingerprint != config.payload[
        "runtime_input_fingerprints"
    ][str(seed)]:
        raise RuntimeError(
            "full formal runtime input differs from the seed freeze"
        )

    control_binding = config.payload["control_preflight_binding"]
    assert isinstance(control_binding, Mapping)
    if method == PAIRED_DIFFERENCE_METHOD:
        if runtime.control_provider is not None:
            raise ValueError("paired_difference cannot receive a control provider")
    else:
        provider = runtime.control_provider
        if not isinstance(provider, PairedFormalControlInputProvider):
            raise TypeError("every matched control requires the frozen provider")
        provider.verify_unchanged()
        runtime_pair_population = [
            pair.canonical_payload()
            for pair in runtime.pair_catalog.clean_positive
        ]
        provider_pair_population = [
            provider.pair_by_id[pair.pair_id].canonical_payload()
            for pair in runtime.pair_catalog.clean_positive
            if pair.pair_id in provider.pair_by_id
        ]
        if (
            provider.pair_catalog_fingerprint
            != runtime.pair_catalog.catalog_fingerprint
            or len(provider_pair_population)
            != len(runtime_pair_population)
            or stable_fingerprint(provider_pair_population)
            != stable_fingerprint(runtime_pair_population)
            or provider.prepared_catalog_fingerprint != prepared_fingerprint
            or provider.preflight.complete_fingerprint
            != control_binding["complete_fingerprint"]
            or provider.preflight.dct_basis_fingerprint
            != control_binding["dct_basis_fingerprint"]
            or provider.preflight.target_permutation_plan_fingerprint
            != control_binding["target_permutation_plan_fingerprint"]
            or provider.preflight.target_assignment_fingerprint
            != control_binding["target_assignment_fingerprint"]
            or provider.provider_fingerprint
            != control_binding["provider_fingerprint"]
        ):
            raise RuntimeError("formal control provider differs from the freeze")
    return runtime_input_fingerprint


def build_seeded_formal_decoder(
    config: LoadedPairedFormalRunnerConfig,
    *,
    seed: int,
    device: torch.device,
) -> CURELiteDecoder:
    """Initialize on CPU so every method at one seed starts byte-identically."""

    if seed not in FORMAL_RUNNER_SEEDS:
        raise ValueError("formal seed must be 42 or 43")
    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("formal runner device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    model = config.payload["model"]
    assert isinstance(model, Mapping)
    decoder_config = DecoderConfig(**dict(model["decoder_config"]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        decoder = CURELiteDecoder(decoder_config)
    expected = config.payload["initial_decoder_fingerprints"][str(seed)]
    if decoder_state_fingerprint(decoder) != expected:
        raise RuntimeError("CPU decoder initialization differs from the freeze")
    decoder = decoder.to(device=device)
    if decoder_state_fingerprint(decoder) != expected:
        raise RuntimeError("moving decoder changed its initial bytes")
    decoder.train()
    return decoder


def _run_config(
    protocol: LoadedPairedFormalRunnerConfig,
    runtime: PairedFormalRuntimeInputs,
    decoder: CURELiteDecoder,
    *,
    method: str,
    seed: int,
    runtime_input_fingerprint: str,
) -> PairedDecoderRunConfig:
    bundle = runtime.bundle
    objective = protocol.payload["paired_objective_binding"]
    control = protocol.payload["control_preflight_binding"]
    model = protocol.payload["model"]
    assert isinstance(objective, Mapping)
    assert isinstance(control, Mapping)
    assert isinstance(model, Mapping)
    return PairedDecoderRunConfig(
        method=method,
        seed=seed,
        manifest_fingerprint=bundle.split_manifest_fingerprint,
        manifest_file_sha256=bundle.split_manifest_file_sha256,
        preprocessing_fingerprint=bundle.preprocessing_fingerprint,
        base_fingerprint=bundle.base_fingerprint,
        state_fingerprint=bundle.state_fingerprint,
        gt_fingerprint=bundle.gt_fingerprint,
        base_index_fingerprint=bundle.base_index_fingerprint,
        base_index_sha256=bundle.base_index_sha256,
        state_index_fingerprint=bundle.state_index_fingerprint,
        state_index_sha256=bundle.state_index_sha256,
        formal_protocol_fingerprint=protocol.config_fingerprint,
        paired_objective_fingerprint=str(objective["receipt_fingerprint"]),
        pair_catalog_fingerprint=runtime.pair_catalog.catalog_fingerprint,
        paired_schedule_fingerprint=(
            runtime.schedule.paired_schedule.schedule_fingerprint
        ),
        formal_schedule_fingerprint=runtime.schedule.schedule_fingerprint,
        runtime_input_fingerprint=runtime_input_fingerprint,
        control_preflight_fingerprint=str(control["complete_fingerprint"]),
        control_provider_fingerprint=(
            None
            if method == PAIRED_DIFFERENCE_METHOD
            else runtime.control_provider.provider_fingerprint
        ),
        method_contract_fingerprint=stable_fingerprint(
            method_objective_contract(method)
        ),
        initial_decoder_fingerprint=decoder_state_fingerprint(decoder),
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
        decoder_config=decoder.config,
        absolute_loss_config=LossConfig(
            **dict(model["absolute_loss_config"])
        ),
    )


def _claim_attempt(
    output_dir: str | Path,
    *,
    protocol: LoadedPairedFormalRunnerConfig,
    preflight: PublishedPairedFormalPreflight,
    method: str,
    seed: int,
) -> Path:
    requested = Path(output_dir).expanduser()
    absolute = Path(os.path.abspath(requested))
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(
            f"formal attempt output already exists and cannot be reused: {absolute}"
        )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("formal attempt path may not traverse a symbolic link")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.mkdir(exist_ok=False)
    marker = _fingerprinted(
        {
            "schema_version": "cure-lite-paired-formal-incomplete-v1",
            "execution_status": "incomplete",
            "method": method,
            "seed": seed,
            "config_fingerprint": protocol.config_fingerprint,
            "formal_preflight_complete_fingerprint": (
                preflight.complete_fingerprint
            ),
            "resume_allowed": False,
            "directory_reuse_allowed": False,
            "checkpoint_written": False,
        },
        field="attempt_fingerprint",
    )
    _write_new_json(absolute / _INCOMPLETE_NAME, marker)
    return absolute


def _provider_receipt(
    runtime: PairedFormalRuntimeInputs,
    *,
    method: str,
    control_preflight_fingerprint: str,
) -> dict[str, object]:
    provider = runtime.control_provider
    used = method != PAIRED_DIFFERENCE_METHOD
    if used:
        assert isinstance(provider, PairedFormalControlInputProvider)
        provider_payload: object = provider.canonical_receipt()
        provider_fingerprint: object = provider.provider_fingerprint
    else:
        if provider is not None:
            raise ValueError("paired_difference cannot serialize a provider")
        provider_payload = None
        provider_fingerprint = None
    return _fingerprinted(
        {
            "schema_version": PAIRED_FORMAL_PROVIDER_RECEIPT_SCHEMA,
            "method": method,
            "control_provider_used": used,
            "control_preflight_fingerprint": control_preflight_fingerprint,
            "provider_fingerprint": provider_fingerprint,
            "provider_receipt": provider_payload,
        },
        field="receipt_fingerprint",
    )


@dataclass(frozen=True)
class PublishedPairedFormalAttempt:
    root: Path
    method: str
    seed: int
    config_fingerprint: str
    formal_schedule_fingerprint: str
    runtime_input_fingerprint: str
    initial_decoder_fingerprint: str
    final_decoder_fingerprint: str
    paired_artifact_fingerprint: str
    provider_fingerprint: str | None
    run_receipt_fingerprint: str
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        if load_paired_formal_attempt(self.root) != self:
            raise RuntimeError("published paired formal attempt changed")


def _published_file_hashes(
    root: Path,
    artifact: LoadedPairedDecoderArtifact,
) -> dict[str, str]:
    return {
        f"{_ARTIFACT_DIR}/decoder.safetensors": artifact.weights_sha256,
        f"{_ARTIFACT_DIR}/train_log.json": artifact.train_log_sha256,
        f"{_ARTIFACT_DIR}/execution_ledger.json": (
            artifact.execution_ledger_sha256
        ),
        f"{_ARTIFACT_DIR}/receipt.json": artifact.receipt_sha256,
        _PROVIDER_NAME: file_sha256(root / _PROVIDER_NAME),
        _RUN_RECEIPT_NAME: file_sha256(root / _RUN_RECEIPT_NAME),
    }


def run_paired_formal_attempt(
    protocol: LoadedPairedFormalRunnerConfig,
    preflight: PublishedPairedFormalPreflight,
    runtime: PairedFormalRuntimeInputs,
    *,
    method: str,
    seed: int,
    output_dir: str | Path,
    device: torch.device,
) -> PublishedPairedFormalAttempt:
    """Run exactly one 800 x 40 attempt and publish only after completion."""

    if not isinstance(protocol, LoadedPairedFormalRunnerConfig):
        raise TypeError("protocol must come from the strict config loader")
    runtime_input_fingerprint = _validate_runtime(
        protocol,
        preflight,
        runtime,
        method=method,
        seed=seed,
    )
    root = _claim_attempt(
        output_dir,
        protocol=protocol,
        preflight=preflight,
        method=method,
        seed=seed,
    )

    # There is intentionally no recovery handler below.  Any exception leaves
    # the claimed root with its incomplete seal, making a new path mandatory.
    decoder = build_seeded_formal_decoder(
        protocol,
        seed=seed,
        device=device,
    )
    run_config = _run_config(
        protocol,
        runtime,
        decoder,
        method=method,
        seed=seed,
        runtime_input_fingerprint=runtime_input_fingerprint,
    )
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    result = execute_paired_formal_training(
        decoder,
        CURELiteLoss(run_config.absolute_loss_config),
        PairedDifferenceLoss(),
        optimizer,
        runtime.schedule,
        run_config,
        control_kwargs_provider=runtime.control_provider,
    )
    if not isinstance(result, PairedFormalTrainingResult):
        raise TypeError("formal training engine returned an invalid result")

    protocol.verify_unchanged()
    preflight.verify_unchanged()
    runtime.bundle.verify_unchanged()
    if (
        paired_formal_runtime_input_fingerprint(
            runtime.pair_catalog,
            runtime.schedule,
        )
        != runtime_input_fingerprint
    ):
        raise RuntimeError("formal runtime input changed during training")
    if runtime.control_provider is not None:
        runtime.control_provider.verify_unchanged()

    artifact_fingerprint = save_paired_decoder_artifact(
        root / _ARTIFACT_DIR,
        decoder,
        run_config,
        result.epoch_logs,
        result.execution_ledger,
    )
    artifact = load_paired_decoder_artifact(
        root / _ARTIFACT_DIR,
        expected_config=run_config,
    )
    if artifact.artifact_fingerprint != artifact_fingerprint:
        raise RuntimeError("published paired artifact fingerprint changed")

    control_binding = protocol.payload["control_preflight_binding"]
    assert isinstance(control_binding, Mapping)
    provider_receipt = _provider_receipt(
        runtime,
        method=method,
        control_preflight_fingerprint=str(
            control_binding["complete_fingerprint"]
        ),
    )
    _write_new_json(root / _PROVIDER_NAME, provider_receipt)
    run_receipt = _fingerprinted(
        {
            "schema_version": PAIRED_FORMAL_ATTEMPT_SCHEMA,
            "execution_status": "complete",
            "dataset": "IRSTD-1K",
            "training_split": "D_R",
            "method": method,
            "seed": seed,
            "runner_config_fingerprint": protocol.config_fingerprint,
            "runner_config_file_sha256": protocol.source_sha256,
            "formal_preflight_complete_fingerprint": (
                preflight.complete_fingerprint
            ),
            "formal_preflight_method_bindings_fingerprint": (
                preflight.method_bindings_fingerprint
            ),
            "pair_catalog_fingerprint": (
                runtime.pair_catalog.catalog_fingerprint
            ),
            "formal_schedule_fingerprint": (
                runtime.schedule.schedule_fingerprint
            ),
            "runtime_input_fingerprint": runtime_input_fingerprint,
            "initial_decoder_fingerprint": (
                run_config.initial_decoder_fingerprint
            ),
            "final_decoder_fingerprint": (
                artifact.decoder_state_fingerprint
            ),
            "paired_artifact_fingerprint": artifact_fingerprint,
            "provider_receipt_fingerprint": provider_receipt[
                "receipt_fingerprint"
            ],
            "provider_fingerprint": (
                runtime.control_provider.provider_fingerprint
                if runtime.control_provider is not None
                else None
            ),
            "optimizer_updates": 32_000,
            "completed_epochs": 800,
            "checkpoint_written": False,
            "resume_used": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "wave_decision_performed": False,
        },
        field="receipt_fingerprint",
    )
    _write_new_json(root / _RUN_RECEIPT_NAME, run_receipt)
    published_files = _published_file_hashes(root, artifact)
    complete = _fingerprinted(
        {
            "schema_version": PAIRED_FORMAL_ATTEMPT_COMPLETE_SCHEMA,
            "execution_status": "complete",
            "method": method,
            "seed": seed,
            "runner_config_fingerprint": protocol.config_fingerprint,
            "formal_schedule_fingerprint": (
                runtime.schedule.schedule_fingerprint
            ),
            "runtime_input_fingerprint": runtime_input_fingerprint,
            "initial_decoder_fingerprint": (
                run_config.initial_decoder_fingerprint
            ),
            "final_decoder_fingerprint": (
                artifact.decoder_state_fingerprint
            ),
            "paired_artifact_fingerprint": artifact_fingerprint,
            "provider_fingerprint": (
                runtime.control_provider.provider_fingerprint
                if runtime.control_provider is not None
                else None
            ),
            "run_receipt_fingerprint": run_receipt[
                "receipt_fingerprint"
            ],
            "artifact_files": published_files,
            "artifact_file_count": len(published_files),
            "complete_800_by_40": True,
            "checkpoint_written": False,
            "resume_used": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "wave_decision_performed": False,
        },
        field="complete_fingerprint",
    )
    _write_new_json(root / _COMPLETE_NAME, complete)
    (root / _INCOMPLETE_NAME).unlink()
    published = load_paired_formal_attempt(root)
    published.verify_unchanged()
    return published


def load_paired_formal_attempt(
    output_dir: str | Path,
) -> PublishedPairedFormalAttempt:
    """Strictly load one fully completed no-resume formal attempt."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise ValueError("formal attempt cannot be addressed through a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("formal attempt root must be a regular directory")
    if (root / _INCOMPLETE_NAME).exists():
        raise RuntimeError("formal attempt is incomplete and cannot be resumed")
    if {path.name for path in root.iterdir()} != {
        _ARTIFACT_DIR,
        _PROVIDER_NAME,
        _RUN_RECEIPT_NAME,
        _COMPLETE_NAME,
    }:
        raise RuntimeError("formal attempt file inventory changed")
    artifact = load_paired_decoder_artifact(root / _ARTIFACT_DIR)
    provider = _strict_json(
        root / _PROVIDER_NAME,
        name="formal control-provider receipt",
    )
    provider_receipt_fingerprint = _verify_fingerprint(
        provider,
        field="receipt_fingerprint",
        name="formal control-provider receipt",
    )
    run_receipt = _strict_json(
        root / _RUN_RECEIPT_NAME,
        name="formal run receipt",
    )
    run_receipt_fingerprint = _verify_fingerprint(
        run_receipt,
        field="receipt_fingerprint",
        name="formal run receipt",
    )
    complete = _strict_json(
        root / _COMPLETE_NAME,
        name="formal COMPLETE receipt",
    )
    complete_fingerprint = _verify_fingerprint(
        complete,
        field="complete_fingerprint",
        name="formal COMPLETE receipt",
    )
    actual_files = _published_file_hashes(root, artifact)
    if (
        complete.get("schema_version")
        != PAIRED_FORMAL_ATTEMPT_COMPLETE_SCHEMA
        or complete.get("execution_status") != "complete"
        or complete.get("artifact_files") != actual_files
        or complete.get("artifact_file_count") != len(actual_files)
        or complete.get("paired_artifact_fingerprint")
        != artifact.artifact_fingerprint
        or complete.get("run_receipt_fingerprint")
        != run_receipt_fingerprint
        or run_receipt.get("provider_receipt_fingerprint")
        != provider_receipt_fingerprint
        or run_receipt.get("paired_artifact_fingerprint")
        != artifact.artifact_fingerprint
        or run_receipt.get("method") != artifact.config.method
        or run_receipt.get("seed") != artifact.config.seed
        or run_receipt.get("formal_schedule_fingerprint")
        != artifact.config.formal_schedule_fingerprint
        or run_receipt.get("runtime_input_fingerprint")
        != artifact.config.runtime_input_fingerprint
        or run_receipt.get("initial_decoder_fingerprint")
        != artifact.config.initial_decoder_fingerprint
        or run_receipt.get("final_decoder_fingerprint")
        != artifact.decoder_state_fingerprint
        or complete.get("method") != artifact.config.method
        or complete.get("seed") != artifact.config.seed
        or complete.get("formal_schedule_fingerprint")
        != artifact.config.formal_schedule_fingerprint
        or complete.get("runtime_input_fingerprint")
        != artifact.config.runtime_input_fingerprint
        or complete.get("initial_decoder_fingerprint")
        != artifact.config.initial_decoder_fingerprint
        or complete.get("final_decoder_fingerprint")
        != artifact.decoder_state_fingerprint
    ):
        raise RuntimeError("formal attempt cross-file bindings changed")
    required_true = ("complete_800_by_40",)
    required_false = (
        "checkpoint_written",
        "resume_used",
        "D_V_accessed",
        "D_T_accessed",
        "calibration_performed",
        "inference_performed",
        "wave_decision_performed",
    )
    if any(complete.get(name) is not True for name in required_true) or any(
        complete.get(name) is not False for name in required_false
    ):
        raise RuntimeError("formal attempt execution boundary changed")
    provider_used = provider.get("control_provider_used")
    provider_fingerprint = provider.get("provider_fingerprint")
    expected_used = artifact.config.method != PAIRED_DIFFERENCE_METHOD
    if (
        provider.get("schema_version")
        != PAIRED_FORMAL_PROVIDER_RECEIPT_SCHEMA
        or provider.get("method") != artifact.config.method
        or provider_used is not expected_used
        or (
            expected_used
            and _digest(
                provider_fingerprint,
                name="provider_fingerprint",
            )
            != run_receipt.get("provider_fingerprint")
        )
        or (
            not expected_used
            and (
                provider_fingerprint is not None
                or provider.get("provider_receipt") is not None
                or run_receipt.get("provider_fingerprint") is not None
            )
        )
        or complete.get("provider_fingerprint")
        != run_receipt.get("provider_fingerprint")
        or artifact.config.control_provider_fingerprint
        != run_receipt.get("provider_fingerprint")
    ):
        raise RuntimeError("formal control-provider use binding changed")
    return PublishedPairedFormalAttempt(
        root=root,
        method=artifact.config.method,
        seed=artifact.config.seed,
        config_fingerprint=str(run_receipt["runner_config_fingerprint"]),
        formal_schedule_fingerprint=(
            artifact.config.formal_schedule_fingerprint
        ),
        runtime_input_fingerprint=artifact.config.runtime_input_fingerprint,
        initial_decoder_fingerprint=(
            artifact.config.initial_decoder_fingerprint
        ),
        final_decoder_fingerprint=artifact.decoder_state_fingerprint,
        paired_artifact_fingerprint=artifact.artifact_fingerprint,
        provider_fingerprint=(
            str(provider_fingerprint) if expected_used else None
        ),
        run_receipt_fingerprint=run_receipt_fingerprint,
        complete_fingerprint=complete_fingerprint,
    )


__all__ = [
    "FORMAL_RUNNER_METHODS",
    "FORMAL_RUNNER_SEEDS",
    "PAIRED_FORMAL_ATTEMPT_COMPLETE_SCHEMA",
    "PAIRED_FORMAL_ATTEMPT_SCHEMA",
    "PAIRED_FORMAL_PROVIDER_RECEIPT_SCHEMA",
    "PAIRED_FORMAL_RUNNER_CONFIG_SCHEMA",
    "LoadedPairedFormalRunnerConfig",
    "PairedFormalRuntimeInputs",
    "PublishedPairedFormalAttempt",
    "build_seeded_formal_decoder",
    "load_paired_formal_attempt",
    "load_paired_formal_runner_config",
    "run_paired_formal_attempt",
    "validate_paired_formal_runner_config",
    "validate_preflight_binding",
]
