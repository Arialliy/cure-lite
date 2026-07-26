"""Shared frozen dataset-free runner core for CURE-Lite NLCC-v12.

This root-package module is intentionally outside :mod:`cure_lite.experiment`:
importing that package executes its historical initializer and imports the
real-data pipeline.  NLCC-v12 development and holdout therefore share this
additive, dataset-free core without changing either package initializer.

The module defines execution, evaluation, and create-only publication logic.
Importing it never constructs an optimizer, runs an update, creates an
artifact directory, or accesses a dataset.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .cache.schema import canonical_json, file_sha256, stable_fingerprint
from .config import LossConfig
from .losses import CURELiteLoss
from .nlcc_dataset_free_inputs import (
    ANCHOR_NULL,
    ANCHOR_POSITIVE,
    FACTUAL_BATCH_SIZE,
    NLCCInputProfile,
    NLCCPairSpec,
    NLCCStrata,
    NLCCUpdate,
    build_factual_population,
    build_outcome_batch,
    build_pair_specs,
    build_schedule,
    build_strata,
    factual_indices_for_update,
)
from .nlcc_dataset_free_runner_config import (
    DEVELOPMENT,
    EXPECTED_ADDITIVE_PATHS,
    HOLDOUT,
    INPUT_FREEZE_FILE_SHA256,
    INPUT_FREEZE_FINGERPRINT,
    INPUT_FREEZE_REPO_PATH,
    METHOD_ID,
    NLCCDatasetFreeRunnerConfig,
    PROFILE_INDEPENDENCE_FILE_SHA256,
    PROFILE_INDEPENDENCE_FINGERPRINT,
    PROFILE_INDEPENDENCE_REPO_PATH,
    RUNNER_CLARIFICATION_FILE_SHA256,
    RUNNER_CLARIFICATION_FINGERPRINT,
    RUNNER_CLARIFICATION_REPO_PATH,
    RUNNER_PREREGISTRATION_FILE_SHA256,
    RUNNER_PREREGISTRATION_FINGERPRINT,
    RUNNER_PREREGISTRATION_REPO_PATH,
)
from .null_anchored_local_count_crossing_decoder import (
    CURELiteNullAnchoredLocalCountCrossingDecoder,
    NullAnchoredLocalCountCrossingDecoderFields,
)
from .paired_endpoint_crossing_losses import PairedEndpointCrossingLoss
from .paired_outcome_types import OutcomePairBatch
from .paired_types import PairBatch
from .train.paired_outcome_step import outcome_complete_train_step
from .train.step import BranchBatch


PRE_RUN_AUTHORIZATION_SCHEMA = (
    "cure-lite.nlcc-v12.dataset-free-pre-run-authorization.v1"
)
ATTEMPT_SCHEMA = "cure-lite.nlcc-v12.dataset-free-attempt.v1"
RESULT_SCHEMA = "cure-lite.nlcc-v12.dataset-free-result.v1"
FAILURE_SCHEMA = "cure-lite.nlcc-v12.dataset-free-failure.v1"
DECISION_SCHEMA = "cure-lite.nlcc-v12.dataset-free-decision.v1"
COMPLETE_SCHEMA = "cure-lite.nlcc-v12.dataset-free-complete.v1"

_ROOT = Path(__file__).resolve().parents[1]
_INCOMPLETE = ".incomplete"
_AUTHORITY_NONCES: set[object] = set()

_FORBIDDEN_RUNTIME_MODULES = {
    "cure_lite.experiment.cache_pipeline",
    "cure_lite.experiment.stage_a_runner",
    "cure_lite.experiment.stage_a_m_runner",
    "cure_lite.experiment.training_pipeline",
}

REQUIRED_AUTH_SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            *EXPECTED_ADDITIVE_PATHS,
            "CURE_Lite_NLCC_v12_模型与代码设计.md",
            "cure_lite/cache/schema.py",
            "cure_lite/config.py",
            "cure_lite/losses.py",
            "cure_lite/nlcc_dataset_free_inputs.py",
            "cure_lite/nlcc_development_inputs.py",
            "cure_lite/nlcc_holdout_inputs.py",
            "cure_lite/null_anchored_local_count_crossing_config.py",
            "cure_lite/null_anchored_local_count_crossing_decoder.py",
            "cure_lite/paired_endpoint_crossing_losses.py",
            "cure_lite/paired_outcome_losses.py",
            "cure_lite/paired_outcome_types.py",
            "cure_lite/paired_types.py",
            "cure_lite/train/paired_outcome_step.py",
            "cure_lite/train/paired_step.py",
            "cure_lite/train/step.py",
            INPUT_FREEZE_REPO_PATH,
            RUNNER_PREREGISTRATION_REPO_PATH,
            RUNNER_CLARIFICATION_REPO_PATH,
            PROFILE_INDEPENDENCE_REPO_PATH,
        )
    )
)


def runtime_import_boundary() -> dict[str, object]:
    """Fail if the dataset or real-data pipeline entered this process."""

    forbidden = sorted(
        name
        for name in sys.modules
        if name in _FORBIDDEN_RUNTIME_MODULES
        or name == "datasets"
        or name.startswith("datasets.")
    )
    if forbidden:
        raise RuntimeError(
            "NLCC dataset-free runtime import boundary was crossed: "
            f"{forbidden}"
        )
    return {
        "forbidden_exact_modules": sorted(_FORBIDDEN_RUNTIME_MODULES),
        "datasets_prefix_forbidden": True,
        "observed_forbidden_modules": forbidden,
        "all_pass": True,
    }


def _profile_input(config: NLCCDatasetFreeRunnerConfig) -> NLCCInputProfile:
    """Import only the selected input profile.

    In particular, the holdout path never imports the development input
    module.  Development artifacts are an authorization record only.
    """

    if config.profile.kind == DEVELOPMENT:
        from .nlcc_development_inputs import DEVELOPMENT_PROFILE

        return DEVELOPMENT_PROFILE
    if config.profile.kind == HOLDOUT:
        from .nlcc_holdout_inputs import HOLDOUT_PROFILE

        return HOLDOUT_PROFILE
    raise ValueError("unknown profile kind")


def _select_pair_batch(batch: PairBatch, indices: Tensor) -> PairBatch:
    selected = tuple(int(value) for value in indices.detach().cpu().tolist())
    return PairBatch(
        feature=batch.feature.index_select(0, indices),
        occupancy_plus=batch.occupancy_plus.index_select(0, indices),
        occupancy_minus=batch.occupancy_minus.index_select(0, indices),
        label_increment=batch.label_increment.index_select(0, indices),
        image_valid_mask=batch.image_valid_mask.index_select(0, indices),
        pair_ids=tuple(batch.pair_ids[index] for index in selected),
        sample_ids=tuple(batch.sample_ids[index] for index in selected),
        group_ids=tuple(batch.group_ids[index] for index in selected),
        pair_kinds=tuple(batch.pair_kinds[index] for index in selected),
        projection_visible=tuple(
            batch.projection_visible[index] for index in selected
        ),
    )


def _select_outcome_batch(
    population: OutcomePairBatch,
    indices: Tensor,
) -> OutcomePairBatch:
    return OutcomePairBatch(
        pair_batch=_select_pair_batch(population.pair_batch, indices),
        completion_plus=population.completion_plus.index_select(0, indices),
        completion_minus=population.completion_minus.index_select(0, indices),
        gt_union=population.gt_union.index_select(0, indices),
        intervention_footprint=(
            population.intervention_footprint.index_select(0, indices)
        ),
    )


def _select_branch_batch(batch: BranchBatch, indices: Tensor) -> BranchBatch:
    return BranchBatch(
        feature=batch.feature.index_select(0, indices),
        occupancy=batch.occupancy.index_select(0, indices),
        target=batch.target.index_select(0, indices),
        valid_mask=batch.valid_mask.index_select(0, indices),
    )


@dataclass(frozen=True)
class NLCCMaterializedProfile:
    """One scientific cache, built once before all optimizer updates."""

    config: NLCCDatasetFreeRunnerConfig
    input_profile: NLCCInputProfile
    specs: tuple[NLCCPairSpec, ...]
    schedule: tuple[NLCCUpdate, ...]
    pair_population: OutcomePairBatch
    factual_population: Mapping[str, BranchBatch]
    strata: NLCCStrata
    pair_index_schedule: tuple[Tensor, ...]
    factual_index_schedule: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.schedule) != self.config.profile.updates:
            raise ValueError("materialized update count differs")
        if len(self.specs) * 2 == 0:
            raise ValueError("materialized pair population is empty")
        if len(self.pair_index_schedule) != len(self.schedule):
            raise ValueError("pair index schedule differs")
        if len(self.factual_index_schedule) != len(self.schedule):
            raise ValueError("factual index schedule differs")

    def training_batches(
        self,
        update_index: int,
    ) -> tuple[dict[str, BranchBatch], OutcomePairBatch]:
        """Index cached tensors only; never re-enter an input builder."""

        if (
            isinstance(update_index, bool)
            or not isinstance(update_index, int)
            or not 0 <= update_index < len(self.schedule)
        ):
            raise ValueError("update_index is outside the frozen schedule")
        factual_indices = self.factual_index_schedule[update_index]
        factual = {
            name: _select_branch_batch(batch, factual_indices)
            for name, batch in self.factual_population.items()
        }
        outcome = _select_outcome_batch(
            self.pair_population,
            self.pair_index_schedule[update_index],
        )
        return factual, outcome

    def manifest(self) -> dict[str, object]:
        group_rows = Counter(spec.group_id for spec in self.specs)
        role_rows = Counter(spec.anchor_role for spec in self.specs)
        role_slots = Counter()
        for spec in self.specs:
            role_slots[spec.anchor_role] += spec.exposure_count
        return {
            "profile_id": self.input_profile.profile_id,
            "pair_rows": len(self.specs),
            "updates": len(self.schedule),
            "pair_slots": sum(
                len(update.population_indices) for update in self.schedule
            ),
            "group_rows": dict(sorted(group_rows.items())),
            "anchor_role_rows": dict(sorted(role_rows.items())),
            "anchor_role_slots": dict(sorted(role_slots.items())),
            "factual_rows_per_branch": {
                name: int(batch.feature.shape[0])
                for name, batch in sorted(self.factual_population.items())
            },
            "scientific_pair_population_builder_calls": 1,
            "scientific_factual_population_builder_calls": 1,
            "scientific_schedule_builder_calls": 1,
            "per_update_builder_reentry": False,
            "per_update_operation": "cached_tensor_index_select",
            "frozen_fingerprints": {
                "input": self.config.profile.input_fingerprint,
                "catalog": self.config.profile.catalog_fingerprint,
                "schedule": self.config.profile.schedule_fingerprint,
                "factual_population": (
                    self.config.profile.factual_population_fingerprint
                ),
                "factual_schedule": (
                    self.config.profile.factual_schedule_fingerprint
                ),
            },
        }


def materialize_profile(
    config: NLCCDatasetFreeRunnerConfig,
) -> NLCCMaterializedProfile:
    """Build the selected full populations and schedules exactly once."""

    if not isinstance(config, NLCCDatasetFreeRunnerConfig):
        raise TypeError("config must be NLCCDatasetFreeRunnerConfig")
    runtime_import_boundary()
    profile = _profile_input(config)
    if (
        profile.profile_id != config.profile.profile_id
        or profile.update_count != config.profile.updates
    ):
        raise RuntimeError("selected frozen input profile differs from config")
    specs = build_pair_specs(profile)
    schedule = build_schedule(profile, specs)
    pair_population = build_outcome_batch(profile, specs, device="cpu")
    factual_population = build_factual_population(profile, device="cpu")
    strata = build_strata(pair_population)

    # These are diagnostic leaves.  The frozen step detaches every feature;
    # their gradients must therefore remain None after a complete run.
    pair_population.pair_batch.feature.requires_grad_(True)
    for batch in factual_population.values():
        batch.feature.requires_grad_(True)

    pair_indices = tuple(
        torch.tensor(
            update.population_indices,
            dtype=torch.int64,
            device="cpu",
        )
        for update in schedule
    )
    factual_indices = tuple(
        torch.tensor(
            factual_indices_for_update(profile, update_index),
            dtype=torch.int64,
            device="cpu",
        )
        for update_index in range(profile.update_count)
    )
    cache = NLCCMaterializedProfile(
        config=config,
        input_profile=profile,
        specs=specs,
        schedule=schedule,
        pair_population=pair_population,
        factual_population=factual_population,
        strata=strata,
        pair_index_schedule=pair_indices,
        factual_index_schedule=factual_indices,
    )
    manifest = cache.manifest()
    if manifest["pair_slots"] != config.profile.pair_slots:
        raise RuntimeError("materialized pair slot count differs")
    if set(factual_population) != {"factual_miss", "factual_no_miss"}:
        raise RuntimeError("materialized factual branches differ")
    return cache


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return (
        str(tensor.dtype).encode("utf-8")
        + repr(tuple(tensor.shape)).encode("utf-8")
        + tensor.numpy().tobytes()
    )


def decoder_fingerprint(decoder: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in decoder.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


@dataclass(frozen=True)
class PreRunAuthorization:
    profile_id: str
    profile_kind: str
    attempt_ordinal: int
    repo_path: str
    file_sha256: str
    authorization_fingerprint: str
    source_bindings: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.profile_kind not in {DEVELOPMENT, HOLDOUT}:
            raise ValueError("authorization profile kind differs")
        if self.attempt_ordinal != 1:
            raise ValueError("only attempt ordinal one is allowed")
        for name in ("file_sha256", "authorization_fingerprint"):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA256")

    def binding(self) -> dict[str, object]:
        return {
            "repo_path": self.repo_path,
            "file_sha256": self.file_sha256,
            "authorization_fingerprint": self.authorization_fingerprint,
            "profile_id": self.profile_id,
            "profile_kind": self.profile_kind,
            "attempt_ordinal": self.attempt_ordinal,
        }


def _source_bindings(repo_root: Path) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for relative in REQUIRED_AUTH_SOURCE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required runner source is absent: {relative}")
        bindings[relative] = file_sha256(path)
    return bindings


def pre_run_authorization_payload(
    config: NLCCDatasetFreeRunnerConfig,
    *,
    repo_root: Path = _ROOT,
) -> dict[str, object]:
    """Return, but never publish, the exact future authorization payload."""

    source_bindings = _source_bindings(Path(repo_root))
    payload: dict[str, object] = {
        "schema_version": PRE_RUN_AUTHORIZATION_SCHEMA,
        "method_id": METHOD_ID,
        "profile_kind": config.profile.kind,
        "profile_id": config.profile.profile_id,
        "attempt_ordinal": 1,
        "authorized": True,
        "runner_contract": {
            "config_fingerprint": stable_fingerprint(config.manifest()),
            "preregistration_file_sha256": (
                RUNNER_PREREGISTRATION_FILE_SHA256
            ),
            "preregistration_fingerprint": (
                RUNNER_PREREGISTRATION_FINGERPRINT
            ),
            "path_metric_clarification_file_sha256": (
                RUNNER_CLARIFICATION_FILE_SHA256
            ),
            "path_metric_clarification_fingerprint": (
                RUNNER_CLARIFICATION_FINGERPRINT
            ),
            "profile_independence_file_sha256": (
                PROFILE_INDEPENDENCE_FILE_SHA256
            ),
            "profile_independence_fingerprint": (
                PROFILE_INDEPENDENCE_FINGERPRINT
            ),
            "input_freeze_file_sha256": INPUT_FREEZE_FILE_SHA256,
            "input_freeze_fingerprint": INPUT_FREEZE_FINGERPRINT,
        },
        "profile_binding": config.profile.manifest(),
        "source_bindings": source_bindings,
        "execution_contract": {
            "one_attempt": True,
            "automatic_retry": False,
            "from_scratch_decoder_seed": config.decoder_seed,
            "fresh_empty_adam_state": True,
            "development_state_carry_into_holdout": False,
        },
    }
    payload["authorization_fingerprint"] = stable_fingerprint(payload)
    return payload


def _load_json_object(path: Path, *, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value


def load_pre_run_authorization(
    config: NLCCDatasetFreeRunnerConfig,
    *,
    repo_root: Path = _ROOT,
) -> PreRunAuthorization:
    """Load and independently verify authority before any attempt claim."""

    root = Path(repo_root)
    path = root / config.profile.pre_run_authorization
    if not path.is_file():
        raise FileNotFoundError(
            "the frozen pre-run authorization is absent; no attempt was claimed"
        )
    payload = _load_json_object(path, name="pre-run authorization")
    observed = payload.get("authorization_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("authorization_fingerprint", None)
    if not isinstance(observed, str) or stable_fingerprint(unsigned) != observed:
        raise RuntimeError("pre-run authorization fingerprint differs")
    expected = pre_run_authorization_payload(config, repo_root=root)
    if payload != expected:
        raise RuntimeError("pre-run authorization differs from frozen sources")
    bindings = payload["source_bindings"]
    if not isinstance(bindings, dict):
        raise TypeError("authorization source_bindings must be an object")
    return PreRunAuthorization(
        profile_id=config.profile.profile_id,
        profile_kind=config.profile.kind,
        attempt_ordinal=1,
        repo_path=str(path.relative_to(root)),
        file_sha256=file_sha256(path),
        authorization_fingerprint=observed,
        source_bindings={str(key): str(value) for key, value in bindings.items()},
    )


@dataclass(frozen=True)
class ExecutionAuthority:
    """Process-local token issued only after durable attempt reload."""

    config_fingerprint: str
    profile_id: str
    profile_kind: str
    artifact_directory: Path
    attempt_binding: Mapping[str, object]
    development_authorization_binding: Mapping[str, object] | None
    _nonce: object


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_create_only(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_json_create_only(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes_create_only(
        path,
        (canonical_json(payload) + "\n").encode("utf-8"),
    )


def _attempt_payload(
    config: NLCCDatasetFreeRunnerConfig,
    authorization: PreRunAuthorization,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": ATTEMPT_SCHEMA,
        "method_id": METHOD_ID,
        "profile_kind": config.profile.kind,
        "profile_id": config.profile.profile_id,
        "attempt_ordinal": 1,
        "status": "STARTED_CREATE_ONLY",
        "automatic_retry_allowed": False,
        "config_fingerprint": stable_fingerprint(config.manifest()),
        "authorization_binding": authorization.binding(),
    }
    payload["attempt_fingerprint"] = stable_fingerprint(payload)
    return payload


def _verify_complete_inventory(directory: Path) -> dict[str, object]:
    complete_path = directory / "COMPLETE.json"
    complete = _load_json_object(complete_path, name="COMPLETE")
    observed = complete.get("complete_fingerprint")
    unsigned = dict(complete)
    unsigned.pop("complete_fingerprint", None)
    if not isinstance(observed, str) or stable_fingerprint(unsigned) != observed:
        raise RuntimeError("COMPLETE fingerprint differs")
    inventory = complete.get("files")
    if not isinstance(inventory, dict):
        raise TypeError("COMPLETE files must be an object")
    for name, expected_sha in inventory.items():
        if not isinstance(name, str) or not isinstance(expected_sha, str):
            raise TypeError("COMPLETE inventory entries must be strings")
        path = directory / name
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise RuntimeError(f"COMPLETE inventory differs for {name}")
    return complete


def verify_development_authorization_artifacts(
    *,
    repo_root: Path = _ROOT,
) -> dict[str, object]:
    """Verify development PASS for holdout authorization, never its state."""

    directory = (
        Path(repo_root)
        / "protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/"
        "development_regression_r1"
    )
    required = {"attempt.json", "result.json", "decision.json", "COMPLETE.json"}
    observed = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
    if observed != required:
        raise RuntimeError(
            "holdout requires exactly the sealed development PASS artifacts"
        )
    if any(
        "checkpoint" in name.lower() or "optimizer" in name.lower()
        for name in observed
    ):
        raise RuntimeError("holdout may not consume development state")
    complete = _verify_complete_inventory(directory)
    attempt = _load_json_object(directory / "attempt.json", name="development attempt")
    result = _load_json_object(directory / "result.json", name="development result")
    decision = _load_json_object(
        directory / "decision.json", name="development decision"
    )
    if set(complete.get("files", {})) != {
        "attempt.json",
        "result.json",
        "decision.json",
    }:
        raise RuntimeError("development COMPLETE inventory is not exact")
    if (
        attempt.get("schema_version") != ATTEMPT_SCHEMA
        or attempt.get("method_id") != METHOD_ID
        or attempt.get("profile_kind") != DEVELOPMENT
        or result.get("schema_version") != RESULT_SCHEMA
        or result.get("method_id") != METHOD_ID
        or result.get("profile_kind") != DEVELOPMENT
        or decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("method_id") != METHOD_ID
        or decision.get("profile_kind") != DEVELOPMENT
    ):
        raise RuntimeError("development artifact identity differs")
    if (
        result.get("all_pass") is not True
        or decision.get("all_pass") is not True
        or decision.get("decision") != "NLCC_V12_DEVELOPMENT_PASS"
    ):
        raise RuntimeError("development artifacts do not authorize holdout")
    attempt_unsigned = dict(attempt)
    attempt_fingerprint = attempt_unsigned.pop("attempt_fingerprint", None)
    if (
        not isinstance(attempt_fingerprint, str)
        or stable_fingerprint(attempt_unsigned) != attempt_fingerprint
    ):
        raise RuntimeError("development attempt fingerprint differs")
    result_unsigned = dict(result)
    result_fingerprint = result_unsigned.pop("result_fingerprint", None)
    if (
        not isinstance(result_fingerprint, str)
        or stable_fingerprint(result_unsigned) != result_fingerprint
    ):
        raise RuntimeError("development result fingerprint differs")
    decision_unsigned = dict(decision)
    decision_fingerprint = decision_unsigned.pop("decision_fingerprint", None)
    if (
        not isinstance(decision_fingerprint, str)
        or stable_fingerprint(decision_unsigned) != decision_fingerprint
    ):
        raise RuntimeError("development decision fingerprint differs")
    initial_decoder_fingerprint = result.get("initial_decoder_fingerprint")
    if (
        not isinstance(initial_decoder_fingerprint, str)
        or len(initial_decoder_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in initial_decoder_fingerprint
        )
    ):
        raise RuntimeError("development initial decoder fingerprint is absent")
    return {
        "purpose": "authorization_only",
        "directory": str(directory.relative_to(Path(repo_root))),
        "result_file_sha256": file_sha256(directory / "result.json"),
        "result_fingerprint": result_fingerprint,
        "attempt_fingerprint": attempt_fingerprint,
        "decision_fingerprint": decision_fingerprint,
        "complete_fingerprint": complete["complete_fingerprint"],
        "initial_decoder_fingerprint": initial_decoder_fingerprint,
        "checkpoint_loaded": False,
        "optimizer_state_loaded": False,
        "training_trace_used_as_input": False,
        "all_pass": True,
    }


def claim_execution(
    config: NLCCDatasetFreeRunnerConfig,
    authorization: PreRunAuthorization,
    *,
    repo_root: Path = _ROOT,
) -> ExecutionAuthority:
    """Atomically claim one profile and durably reload attempt.json."""

    if authorization.profile_id != config.profile.profile_id:
        raise ValueError("authorization profile differs from config")
    if authorization.profile_kind != config.profile.kind:
        raise ValueError("authorization kind differs from config")
    development_binding = None
    if config.profile.kind == HOLDOUT:
        development_binding = verify_development_authorization_artifacts(
            repo_root=repo_root
        )
    directory = Path(repo_root) / config.profile.canonical_artifact_directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(directory)
    _fsync_directory(directory.parent)
    _write_bytes_create_only(directory / _INCOMPLETE, b"INCOMPLETE\n")
    attempt = _attempt_payload(config, authorization)
    _write_json_create_only(directory / "attempt.json", attempt)
    reloaded = _load_json_object(directory / "attempt.json", name="attempt")
    if reloaded != attempt:
        raise RuntimeError("durably reloaded attempt differs")
    attempt_unsigned = dict(reloaded)
    observed = attempt_unsigned.pop("attempt_fingerprint", None)
    if not isinstance(observed, str) or stable_fingerprint(attempt_unsigned) != observed:
        raise RuntimeError("attempt fingerprint differs after reload")
    nonce = object()
    _AUTHORITY_NONCES.add(nonce)
    return ExecutionAuthority(
        config_fingerprint=stable_fingerprint(config.manifest()),
        profile_id=config.profile.profile_id,
        profile_kind=config.profile.kind,
        artifact_directory=directory,
        attempt_binding={
            "repo_path": str((directory / "attempt.json").relative_to(repo_root)),
            "file_sha256": file_sha256(directory / "attempt.json"),
            "attempt_fingerprint": observed,
            "attempt_ordinal": 1,
        },
        development_authorization_binding=development_binding,
        _nonce=nonce,
    )


def _require_authority(
    authority: ExecutionAuthority,
    config: NLCCDatasetFreeRunnerConfig,
) -> None:
    if not isinstance(authority, ExecutionAuthority):
        raise TypeError("authority must be ExecutionAuthority")
    if authority._nonce not in _AUTHORITY_NONCES:
        raise RuntimeError("execution authority is not process-local and live")
    if authority.config_fingerprint != stable_fingerprint(config.manifest()):
        raise RuntimeError("execution authority config differs")
    if authority.profile_id != config.profile.profile_id:
        raise RuntimeError("execution authority profile differs")
    attempt = authority.artifact_directory / "attempt.json"
    if not attempt.is_file() or not (authority.artifact_directory / _INCOMPLETE).is_file():
        raise RuntimeError("attempt must be durable before optimizer construction")


@dataclass(frozen=True)
class NLCCTrainingComponents:
    decoder: CURELiteNullAnchoredLocalCountCrossingDecoder
    absolute_criterion: CURELiteLoss
    outcome_criterion: PairedEndpointCrossingLoss
    optimizer: torch.optim.Adam
    optimizer_contract: Mapping[str, object]
    initial_decoder_fingerprint: str
    optimizer_state_initially_empty: bool


def _optimizer_contract(optimizer: torch.optim.Adam) -> dict[str, object]:
    defaults = optimizer.defaults
    return {
        "class": "torch.optim.Adam",
        "learning_rate": float(defaults["lr"]),
        "betas": [float(value) for value in defaults["betas"]],
        "epsilon": float(defaults["eps"]),
        "weight_decay": float(defaults["weight_decay"]),
        "amsgrad": bool(defaults["amsgrad"]),
        "maximize": bool(defaults["maximize"]),
        "foreach": defaults["foreach"],
        "capturable": bool(defaults["capturable"]),
        "differentiable": bool(defaults["differentiable"]),
        "fused": defaults["fused"],
        "decoupled_weight_decay": bool(
            defaults.get("decoupled_weight_decay", False)
        ),
    }


def build_training_components(
    authority: ExecutionAuthority,
    config: NLCCDatasetFreeRunnerConfig,
) -> NLCCTrainingComponents:
    """Construct a from-scratch decoder and fresh Adam after attempt claim."""

    _require_authority(authority, config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.decoder_seed)
        decoder = CURELiteNullAnchoredLocalCountCrossingDecoder(
            feature_channels=config.feature_channels,
            feature_stride=config.feature_stride,
        )
    named_parameters = tuple(decoder.named_parameters())
    if len(named_parameters) != config.parameter_tensors:
        raise RuntimeError("decoder parameter tensor count differs")
    if sum(value.numel() for _, value in named_parameters) != config.parameters:
        raise RuntimeError("decoder parameter count differs")
    initial = decoder_fingerprint(decoder)
    loss_config = LossConfig(
        dice_weight=config.loss_dice_weight,
        epsilon=config.loss_epsilon,
    )
    absolute = CURELiteLoss(loss_config)
    outcome = PairedEndpointCrossingLoss(loss_config)
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
        maximize=config.maximize,
        foreach=config.foreach,
        capturable=config.capturable,
        differentiable=config.differentiable,
        fused=config.fused,
        decoupled_weight_decay=config.decoupled_weight_decay,
    )
    optimizer_contract = _optimizer_contract(optimizer)
    expected_optimizer_contract = {
        "class": "torch.optim.Adam",
        "learning_rate": config.learning_rate,
        "betas": list(config.betas),
        "epsilon": config.optimizer_epsilon,
        "weight_decay": config.weight_decay,
        "amsgrad": config.amsgrad,
        "maximize": config.maximize,
        "foreach": config.foreach,
        "capturable": config.capturable,
        "differentiable": config.differentiable,
        "fused": config.fused,
        "decoupled_weight_decay": config.decoupled_weight_decay,
    }
    if optimizer_contract != expected_optimizer_contract:
        raise RuntimeError("fresh Adam defaults differ from frozen contract")
    if optimizer.state:
        raise RuntimeError("fresh Adam must have empty optimizer state")
    return NLCCTrainingComponents(
        decoder=decoder,
        absolute_criterion=absolute,
        outcome_criterion=outcome,
        optimizer=optimizer,
        optimizer_contract=optimizer_contract,
        initial_decoder_fingerprint=initial,
        optimizer_state_initially_empty=True,
    )


def _minimum(value: Tensor, *, name: str) -> float:
    if value.numel() == 0:
        raise ValueError(f"{name} mask is empty")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return float(value.min().detach().cpu())


def _maximum(value: Tensor, *, name: str) -> float:
    if value.numel() == 0:
        raise ValueError(f"{name} mask is empty")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return float(value.max().detach().cpu())


def _mean(value: Tensor, *, name: str) -> float:
    if value.numel() == 0:
        raise ValueError(f"{name} mask is empty")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return float(value.mean().detach().cpu())


def _anchor_null_mask(cache: NLCCMaterializedProfile) -> Tensor:
    mask = torch.zeros_like(cache.pair_population.completion_plus)
    stride = cache.config.feature_stride
    for index, spec in enumerate(cache.specs):
        if spec.anchor_role != ANCHOR_NULL:
            continue
        row = stride * spec.anchor_cell[0] + spec.anchor_phase[0]
        column = stride * spec.anchor_cell[1] + spec.anchor_phase[1]
        mask[index, 0, row, column] = True
    return mask


def _row_mask(values: Sequence[bool], reference: Tensor) -> Tensor:
    return torch.tensor(
        values,
        dtype=torch.bool,
        device=reference.device,
    ).reshape(-1, 1, 1, 1)


def _twin_gaps(
    cache: NLCCMaterializedProfile,
    score_plus: Tensor,
) -> dict[str, dict[str, object]]:
    by_match: dict[str, list[tuple[int, NLCCPairSpec]]] = defaultdict(list)
    for index, spec in enumerate(cache.specs):
        by_match[spec.match_id].append((index, spec))
    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    stride = cache.config.feature_stride
    for match_id in sorted(by_match):
        rows = by_match[match_id]
        if len(rows) != 2:
            raise RuntimeError("matched-twin join is not bijective")
        by_role = {spec.anchor_role: (index, spec) for index, spec in rows}
        if set(by_role) != {ANCHOR_POSITIVE, ANCHOR_NULL}:
            raise RuntimeError("matched-twin roles differ")
        positive_index, positive = by_role[ANCHOR_POSITIVE]
        null_index, null = by_role[ANCHOR_NULL]
        if positive.group_id != null.group_id:
            raise RuntimeError("matched twins cross groups")
        row = stride * positive.anchor_cell[0] + positive.anchor_phase[0]
        column = stride * positive.anchor_cell[1] + positive.anchor_phase[1]
        positive_score = float(score_plus[positive_index, 0, row, column])
        null_score = float(score_plus[null_index, 0, row, column])
        by_group[positive.group_id].append(
            {
                "match_id": match_id,
                "positive_plus_score": positive_score,
                "null_plus_score": null_score,
                "gap": positive_score - null_score,
            }
        )
    result: dict[str, dict[str, object]] = {}
    for group_id, rows in sorted(by_group.items()):
        values = torch.tensor([float(row["gap"]) for row in rows])
        result[group_id] = {
            "matches": rows,
            "minimum": _minimum(values, name=f"{group_id}/twin_gap"),
            "mean": _mean(values, name=f"{group_id}/twin_gap"),
            "maximum": _maximum(values, name=f"{group_id}/twin_gap"),
            "is_gate": False,
        }
    return result


def evaluate_cached_logits(
    cache: NLCCMaterializedProfile,
    *,
    logits_plus: Tensor,
    logits_minus: Tensor,
    factual_miss_logits: Tensor,
    factual_no_miss_logits: Tensor,
    structural_training_contract: Mapping[str, object],
    operator_field_diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Apply all frozen gates to one already-computed full-population output."""

    population = cache.pair_population
    factual_miss = cache.factual_population["factual_miss"]
    factual_no_miss = cache.factual_population["factual_no_miss"]
    expected_pair_shape = population.pair_batch.label_increment.shape
    expected_miss_shape = factual_miss.target.shape
    expected_no_miss_shape = factual_no_miss.target.shape
    if logits_plus.shape != expected_pair_shape or logits_minus.shape != expected_pair_shape:
        raise ValueError("full pair logits shape differs")
    if factual_miss_logits.shape != expected_miss_shape:
        raise ValueError("full factual-miss logits shape differs")
    if factual_no_miss_logits.shape != expected_no_miss_shape:
        raise ValueError("full factual-no-miss logits shape differs")
    for name, value in (
        ("logits_plus", logits_plus),
        ("logits_minus", logits_minus),
        ("factual_miss_logits", factual_miss_logits),
        ("factual_no_miss_logits", factual_no_miss_logits),
    ):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} must be finite")

    score_plus = torch.sigmoid(logits_plus)
    score_minus = torch.sigmoid(logits_minus)
    delta = score_minus - score_plus
    factual_miss_score = torch.sigmoid(factual_miss_logits)
    factual_no_miss_score = torch.sigmoid(factual_no_miss_logits)

    loss_config = LossConfig(
        dice_weight=cache.config.loss_dice_weight,
        epsilon=cache.config.loss_epsilon,
    )
    absolute = CURELiteLoss(loss_config)
    outcome = PairedEndpointCrossingLoss(loss_config)
    factual_miss_loss = absolute(
        factual_miss_logits,
        factual_miss.target,
        factual_miss.valid_mask,
    )["total"]
    factual_no_miss_loss = absolute(
        factual_no_miss_logits,
        factual_no_miss.target,
        factual_no_miss.valid_mask,
    )["total"]
    pair_loss = outcome(
        logits_plus,
        logits_minus,
        population.completion_plus,
        population.pair_batch.occupancy_plus,
        population.gt_union,
        population.pair_batch.label_increment,
        population.pair_batch.image_valid_mask,
        population.intervention_footprint,
    )["total"]
    total_loss = factual_miss_loss + factual_no_miss_loss + pair_loss
    if not bool(torch.isfinite(total_loss)):
        raise FloatingPointError("population total loss must be finite")

    thresholds = cache.config.thresholds
    miss_target = factual_miss.target > 0.5
    miss_background = factual_miss.valid_mask & ~miss_target
    no_miss_valid = factual_no_miss.valid_mask
    global_metrics = {
        "population_total_loss": float(total_loss.detach().cpu()),
        "factual_miss_target_min": _minimum(
            factual_miss_score[miss_target], name="factual_miss_target"
        ),
        "factual_miss_background_max": _maximum(
            factual_miss_score[miss_background], name="factual_miss_background"
        ),
        "factual_no_miss_max": _maximum(
            factual_no_miss_score[no_miss_valid], name="factual_no_miss"
        ),
        "factual_miss_loss": float(factual_miss_loss.detach().cpu()),
        "factual_no_miss_loss": float(factual_no_miss_loss.detach().cpu()),
        "pair_loss": float(pair_loss.detach().cpu()),
    }
    global_checks = {
        "population_total_loss": global_metrics["population_total_loss"]
        < thresholds.population_total_loss_max_exclusive,
        "factual_miss_target": global_metrics["factual_miss_target_min"]
        > thresholds.factual_miss_target_min_exclusive,
        "factual_miss_background": global_metrics[
            "factual_miss_background_max"
        ]
        < thresholds.factual_miss_background_max_exclusive,
        "factual_no_miss": global_metrics["factual_no_miss_max"]
        < thresholds.factual_no_miss_max_exclusive,
    }

    positive_role = _row_mask(
        [spec.anchor_role == ANCHOR_POSITIVE for spec in cache.specs],
        score_plus,
    )
    null_anchor = _anchor_null_mask(cache)
    plus_background = (
        population.pair_batch.image_valid_mask
        & ~population.pair_batch.occupancy_plus
        & ~population.gt_union
    )
    twin_gaps = _twin_gaps(cache, score_plus)
    group_ids = tuple(sorted({spec.group_id for spec in cache.specs}))
    if len(group_ids) != 8:
        raise RuntimeError("final evaluation requires exactly eight groups")
    groups: list[dict[str, object]] = []
    numeric_gate_count = len(global_checks)
    for group_id in group_ids:
        group_rows = _row_mask(
            [spec.group_id == group_id for spec in cache.specs],
            score_plus,
        )
        group_specs = tuple(spec for spec in cache.specs if spec.group_id == group_id)
        pair_kinds = {spec.pair_kind for spec in group_specs}
        if len(pair_kinds) != 1:
            raise RuntimeError("one group mixes pair kinds")
        pair_kind = next(iter(pair_kinds))
        positive_mask = group_rows & positive_role & population.completion_plus
        null_mask = group_rows & null_anchor
        background_mask = group_rows & plus_background
        H_mask = group_rows & cache.strata.H
        G_near_mask = group_rows & cache.strata.G_near
        G_tail_mask = group_rows & cache.strata.G_norm_tail
        metrics: dict[str, object] = {
            "row_count": len(group_specs),
            "slot_count": sum(spec.exposure_count for spec in group_specs),
            "positive_anchor_min": _minimum(
                score_plus[positive_mask], name=f"{group_id}/positive_anchor"
            ),
            "matched_anchor_null_max": _maximum(
                score_plus[null_mask], name=f"{group_id}/matched_anchor_null"
            ),
            "plus_background_max": _maximum(
                score_plus[background_mask], name=f"{group_id}/plus_background"
            ),
            "zero_H_max_abs": _maximum(
                delta[H_mask].abs(), name=f"{group_id}/H"
            ),
            "zero_G_near_max_abs": _maximum(
                delta[G_near_mask].abs(), name=f"{group_id}/G_near"
            ),
            "zero_G_norm_tail_max_abs": _maximum(
                delta[G_tail_mask].abs(), name=f"{group_id}/G_norm_tail"
            ),
            "matched_twin_gap": twin_gaps[group_id],
        }
        checks: dict[str, bool] = {
            "positive_anchor": metrics["positive_anchor_min"]
            > thresholds.positive_anchor_min_exclusive,
            "matched_anchor_null": metrics["matched_anchor_null_max"]
            < thresholds.matched_anchor_null_max_exclusive,
            "plus_background": metrics["plus_background_max"]
            < thresholds.plus_background_max_exclusive,
            "zero_H": metrics["zero_H_max_abs"]
            <= thresholds.zero_H_max_abs_max_inclusive,
            "zero_G_near": metrics["zero_G_near_max_abs"]
            <= thresholds.zero_G_near_max_abs_max_inclusive,
            "zero_G_norm_tail": metrics["zero_G_norm_tail_max_abs"]
            <= thresholds.zero_G_norm_tail_max_abs_max_inclusive,
        }
        numeric_gate_count += len(checks)
        D_mask = group_rows & cache.strata.D
        if pair_kind == "clean_positive":
            metrics.update(
                {
                    "clean_D_pixel_count": int(D_mask.sum()),
                    "clean_D_delta_mean": _mean(
                        delta[D_mask], name=f"{group_id}/D_delta"
                    ),
                    "clean_D_plus_max": _maximum(
                        score_plus[D_mask], name=f"{group_id}/D_plus"
                    ),
                    "clean_D_minus_min": _minimum(
                        score_minus[D_mask], name=f"{group_id}/D_minus"
                    ),
                    "D_wrong_direction_pixel_count": int(
                        (delta[D_mask] < 0.0).sum()
                    ),
                }
            )
            D_checks = {
                "clean_D_delta_mean": metrics["clean_D_delta_mean"]
                >= thresholds.clean_D_delta_mean_min_inclusive,
                "clean_D_plus": metrics["clean_D_plus_max"]
                < thresholds.clean_D_plus_max_exclusive,
                "clean_D_minus": metrics["clean_D_minus_min"]
                > thresholds.clean_D_minus_min_exclusive,
                "D_wrong_direction": metrics[
                    "D_wrong_direction_pixel_count"
                ]
                <= thresholds.D_wrong_direction_pixel_count_max_inclusive,
            }
            checks.update(D_checks)
            numeric_gate_count += len(D_checks)
            D_status = "APPLICABLE"
        else:
            if bool(D_mask.any()):
                raise RuntimeError("component-null group unexpectedly has D")
            D_status = "NOT_APPLICABLE_EMPTY_D"
            metrics["clean_D_pixel_count"] = 0
            metrics["clean_D_delta_mean"] = None
            metrics["clean_D_plus_max"] = None
            metrics["clean_D_minus_min"] = None
            metrics["D_wrong_direction_pixel_count"] = None
        groups.append(
            {
                "group_id": group_id,
                "pair_kind": pair_kind,
                "D_gate_status": D_status,
                "metrics": metrics,
                "checks": checks,
                "all_pass": all(checks.values()),
            }
        )

    if numeric_gate_count != 76:
        raise RuntimeError("NLCC-v12 numeric gate algebra differs from 76")
    structural_pass = structural_training_contract.get("all_pass") is True
    all_pass = (
        structural_pass
        and all(global_checks.values())
        and all(group["all_pass"] is True for group in groups)
    )
    return {
        "global_metrics": global_metrics,
        "global_checks": global_checks,
        "groups": groups,
        "numeric_gate_count": numeric_gate_count,
        "structural_training_contract": dict(structural_training_contract),
        "operator_field_diagnostics": (
            {} if operator_field_diagnostics is None else dict(operator_field_diagnostics)
        ),
        "final_forward_contract": {
            "pair_endpoint_forward_calls": 1,
            "pair_endpoint_states": 2 * len(cache.specs),
            "factual_miss_forward_calls": 1,
            "factual_no_miss_forward_calls": 1,
            "total_decoder_calls": 3,
            "unique_pair_rows_equal_weight": True,
            "exposure_weighted": False,
            "repeated_group_forwards": False,
        },
        "all_pass": all_pass,
    }


def _field_range(
    fields: Sequence[NullAnchoredLocalCountCrossingDecoderFields],
    name: str,
) -> dict[str, float]:
    values = torch.cat([getattr(field, name).reshape(-1) for field in fields])
    return {
        "minimum": _minimum(values, name=name),
        "maximum": _maximum(values, name=name),
    }


def evaluate_trained_decoder(
    decoder: CURELiteNullAnchoredLocalCountCrossingDecoder,
    cache: NLCCMaterializedProfile,
    *,
    structural_training_contract: Mapping[str, object],
) -> dict[str, object]:
    """Perform exactly the frozen three final full-population calls."""

    decoder.eval()
    population = cache.pair_population
    miss = cache.factual_population["factual_miss"]
    no_miss = cache.factual_population["factual_no_miss"]
    with torch.no_grad():
        pair_fields = decoder.forward_fields(
            torch.cat(
                (population.pair_batch.feature, population.pair_batch.feature),
                dim=0,
            ),
            torch.cat(
                (
                    population.pair_batch.occupancy_plus,
                    population.pair_batch.occupancy_minus,
                ),
                dim=0,
            ),
        )
        miss_fields = decoder.forward_fields(miss.feature, miss.occupancy)
        no_miss_fields = decoder.forward_fields(
            no_miss.feature, no_miss.occupancy
        )
    row_count = len(cache.specs)
    logits_plus = pair_fields.logits[:row_count]
    logits_minus = pair_fields.logits[row_count:]
    diagnostics = {
        "crossing_margin": _field_range(
            (pair_fields, miss_fields, no_miss_fields), "crossing_margin"
        ),
        "recovery_factor": _field_range(
            (pair_fields, miss_fields, no_miss_fields), "recovery_factor"
        ),
        "all_finite": True,
        "same_three_final_forward_fields_calls": True,
    }
    return evaluate_cached_logits(
        cache,
        logits_plus=logits_plus,
        logits_minus=logits_minus,
        factual_miss_logits=miss_fields.logits,
        factual_no_miss_logits=no_miss_fields.logits,
        structural_training_contract=structural_training_contract,
        operator_field_diagnostics=diagnostics,
    )


def execute_authorized_profile(
    authority: ExecutionAuthority,
    cache: NLCCMaterializedProfile,
) -> dict[str, object]:
    """Run the exact frozen profile; callers must publish its terminal state."""

    config = cache.config
    _require_authority(authority, config)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(config.torch_threads)
        torch.use_deterministic_algorithms(config.deterministic_algorithms)
        components = build_training_components(authority, config)
        same_cross_profile_initialization = True
        if config.profile.kind == HOLDOUT:
            development_binding = authority.development_authorization_binding
            if not isinstance(development_binding, Mapping):
                raise RuntimeError(
                    "holdout lacks verified development authorization binding"
                )
            same_cross_profile_initialization = (
                components.initial_decoder_fingerprint
                == development_binding.get("initial_decoder_fingerprint")
            )
            if not same_cross_profile_initialization:
                raise RuntimeError(
                    "holdout from-scratch seed-42 initialization differs from development"
                )
        decoder = components.decoder
        named_parameters = tuple(decoder.named_parameters())
        forward_sizes: list[int] = []

        def observe(_module: object, args: tuple[object, ...]) -> None:
            forward_sizes.append(int(args[0].shape[0]))

        gradient_failures: list[dict[str, object]] = []
        gradient_minimum = float("inf")
        gradient_maximum = 0.0
        step_contract_failures: list[dict[str, object]] = []
        first_logs: dict[str, float | int] | None = None
        last_logs: dict[str, float | int] | None = None
        handle = decoder.register_forward_pre_hook(observe)
        try:
            for update_index in range(config.profile.updates):
                factual, outcome = cache.training_batches(update_index)
                logs = outcome_complete_train_step(
                    decoder,
                    components.absolute_criterion,
                    components.outcome_criterion,
                    components.optimizer,
                    factual,
                    outcome,
                )
                if update_index == 0:
                    first_logs = dict(logs)
                if update_index == config.profile.updates - 1:
                    last_logs = dict(logs)
                expected = {
                    "factual_miss/states": 4,
                    "factual_no_miss/states": 4,
                    "outcome/pairs": 2,
                    "outcome/endpoints": 4,
                    "decoder_forward_calls_per_update": 3,
                    "decoder_states_per_update": 12,
                    "backward_calls": 1,
                    "optimizer_steps": 1,
                }
                observed = {name: logs.get(name) for name in expected}
                if observed != expected:
                    step_contract_failures.append(
                        {"update_index": update_index, "observed": observed}
                    )
                if int(logs["outcome/clean_pairs"]) + int(
                    logs["outcome/component_null_pairs"]
                ) != 2:
                    step_contract_failures.append(
                        {
                            "update_index": update_index,
                            "observed_pair_kind_total": (
                                int(logs["outcome/clean_pairs"])
                                + int(logs["outcome/component_null_pairs"])
                            ),
                        }
                    )
                for name, parameter in named_parameters:
                    gradient = parameter.grad
                    finite = gradient is not None and bool(
                        torch.isfinite(gradient).all()
                    )
                    norm = (
                        0.0
                        if gradient is None
                        else float(gradient.detach().double().norm().cpu())
                    )
                    gradient_minimum = min(gradient_minimum, norm)
                    gradient_maximum = max(gradient_maximum, norm)
                    if not finite or norm <= 0.0:
                        gradient_failures.append(
                            {
                                "update_index": update_index,
                                "parameter": name,
                                "finite": finite,
                                "l2_norm": norm,
                            }
                        )
        finally:
            handle.remove()
        if first_logs is None or last_logs is None:
            raise RuntimeError("frozen profile executed no updates")
        patterns = tuple(
            tuple(forward_sizes[offset : offset + 3])
            for offset in range(0, len(forward_sizes), 3)
        )
        feature_detach = (
            cache.pair_population.pair_batch.feature.grad is None
            and all(
                batch.feature.grad is None
                for batch in cache.factual_population.values()
            )
        )
        structural = {
            "updates_executed": config.profile.updates,
            "expected_updates": config.profile.updates,
            "training_forward_call_count": len(forward_sizes),
            "expected_training_forward_call_count": 3 * config.profile.updates,
            "all_update_forward_patterns_4_4_4": (
                len(patterns) == config.profile.updates
                and all(pattern == (4, 4, 4) for pattern in patterns)
            ),
            "step_contract_failure_count": len(step_contract_failures),
            "gradient_failure_count": len(gradient_failures),
            "all_six_gradients_finite_nonzero_every_update": not gradient_failures,
            "feature_cache_leaves_remain_without_grad": feature_detach,
            "one_backward_and_one_step_per_update": not step_contract_failures,
            "population_builder_reentry": False,
            "from_scratch_seed_42": True,
            "fresh_adam_state_before_first_update": (
                components.optimizer_state_initially_empty
            ),
            "development_checkpoint_loaded": False,
            "development_optimizer_state_loaded": False,
            "all_pass": (
                len(forward_sizes) == 3 * config.profile.updates
                and len(patterns) == config.profile.updates
                and all(pattern == (4, 4, 4) for pattern in patterns)
                and not step_contract_failures
                and not gradient_failures
                and feature_detach
                and components.optimizer_state_initially_empty
            ),
        }
        final_evaluation = evaluate_trained_decoder(
            decoder,
            cache,
            structural_training_contract=structural,
        )
        all_pass = final_evaluation["all_pass"] is True
        decision_prefix = (
            "NLCC_V12_DEVELOPMENT"
            if config.profile.kind == DEVELOPMENT
            else "NLCC_V12_HOLDOUT"
        )
        result: dict[str, object] = {
            "schema_version": RESULT_SCHEMA,
            "method_id": METHOD_ID,
            "profile_kind": config.profile.kind,
            "profile_id": config.profile.profile_id,
            "evidentiary_role": config.profile.evidentiary_role,
            "decision": f"{decision_prefix}_{'PASS' if all_pass else 'FAIL'}",
            "all_pass": all_pass,
            "attempt_binding": dict(authority.attempt_binding),
            "config": config.manifest(),
            "materialized_cache": cache.manifest(),
            "optimizer_contract": dict(components.optimizer_contract),
            "initial_decoder_fingerprint": (
                components.initial_decoder_fingerprint
            ),
            "final_decoder_fingerprint": decoder_fingerprint(decoder),
            "training": {
                "structural_contract": structural,
                "gradient_minimum_l2": gradient_minimum,
                "gradient_maximum_l2": gradient_maximum,
                "gradient_failures": gradient_failures,
                "step_contract_failures": step_contract_failures,
                "first_update_logs": first_logs,
                "last_update_logs": last_logs,
            },
            "final_evaluation": final_evaluation,
            "profile_independence": {
                "decoder_from_scratch_seed_42": True,
                "optimizer_fresh_empty_state": True,
                "checkpoint_loaded": False,
                "optimizer_state_loaded": False,
                "continued_training": False,
                "same_initial_decoder_fingerprint_as_development": (
                    same_cross_profile_initialization
                ),
                "development_artifacts_used_for_authorization_only": (
                    None
                    if authority.development_authorization_binding is None
                    else dict(authority.development_authorization_binding)
                ),
            },
            "runtime": {
                "device": "cpu",
                "torch_threads": config.torch_threads,
                "deterministic_algorithms": True,
                "torch_version": str(torch.__version__),
                "runtime_import_boundary": runtime_import_boundary(),
            },
            "execution_boundary": {
                "dataset_accessed": False,
                "D_R_accessed": False,
                "DV_accessed": False,
                "DT_accessed": False,
                "detection_performance_evaluated": False,
                "model_success_claim_authorized": False,
                "real_performance_claim_authorized": False,
                "automatic_retry_allowed": False,
            },
        }
        result["result_fingerprint"] = stable_fingerprint(result)
        return result
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)


def _seal_directory(
    authority: ExecutionAuthority,
    *,
    terminal_name: str,
) -> dict[str, object]:
    directory = authority.artifact_directory
    terminal = _load_json_object(directory / terminal_name, name=terminal_name)
    all_pass = terminal_name == "result.json" and terminal.get("all_pass") is True
    if terminal_name == "result.json":
        unsigned = dict(terminal)
        observed = unsigned.pop("result_fingerprint", None)
        if not isinstance(observed, str) or stable_fingerprint(unsigned) != observed:
            raise RuntimeError("result fingerprint differs before decision")
    decision_prefix = (
        "NLCC_V12_DEVELOPMENT"
        if authority.profile_kind == DEVELOPMENT
        else "NLCC_V12_HOLDOUT"
    )
    decision: dict[str, object] = {
        "schema_version": DECISION_SCHEMA,
        "method_id": METHOD_ID,
        "profile_id": authority.profile_id,
        "profile_kind": authority.profile_kind,
        "terminal_file": terminal_name,
        "terminal_file_sha256": file_sha256(directory / terminal_name),
        "decision": f"{decision_prefix}_{'PASS' if all_pass else 'FAIL'}",
        "all_pass": all_pass,
        "recomputed_from_reloaded_terminal": True,
    }
    decision["decision_fingerprint"] = stable_fingerprint(decision)
    _write_json_create_only(directory / "decision.json", decision)
    inventory_names = ("attempt.json", terminal_name, "decision.json")
    complete: dict[str, object] = {
        "schema_version": COMPLETE_SCHEMA,
        "method_id": METHOD_ID,
        "profile_id": authority.profile_id,
        "files": {
            name: file_sha256(directory / name) for name in inventory_names
        },
        "temporary_incomplete_marker_excluded": True,
    }
    complete["complete_fingerprint"] = stable_fingerprint(complete)
    _write_json_create_only(directory / "COMPLETE.json", complete)
    reloaded_complete = _verify_complete_inventory(directory)
    if reloaded_complete != complete:
        raise RuntimeError("independently reloaded COMPLETE differs")
    observed_names = {path.name for path in directory.iterdir()}
    expected_names = {*inventory_names, "COMPLETE.json", _INCOMPLETE}
    if observed_names != expected_names:
        raise RuntimeError("published artifact inventory contains extra files")
    (directory / _INCOMPLETE).unlink()
    _fsync_directory(directory)
    return {
        "decision": decision,
        "complete": complete,
        "artifact_directory": str(directory),
    }


def publish_result(
    authority: ExecutionAuthority,
    result: Mapping[str, object],
) -> dict[str, object]:
    if (authority.artifact_directory / "result.json").exists():
        raise FileExistsError("result.json already exists")
    payload = dict(result)
    unsigned = dict(payload)
    observed = unsigned.pop("result_fingerprint", None)
    if not isinstance(observed, str) or stable_fingerprint(unsigned) != observed:
        raise ValueError("result fingerprint differs")
    _write_json_create_only(authority.artifact_directory / "result.json", payload)
    return _seal_directory(authority, terminal_name="result.json")


def publish_failure(
    authority: ExecutionAuthority,
    error: BaseException,
) -> dict[str, object]:
    failure: dict[str, object] = {
        "schema_version": FAILURE_SCHEMA,
        "method_id": METHOD_ID,
        "profile_id": authority.profile_id,
        "profile_kind": authority.profile_kind,
        "status": "EXECUTION_EXCEPTION_NO_RETRY",
        "exception_type": type(error).__name__,
        "message": str(error),
        "automatic_retry_allowed": False,
    }
    failure["failure_fingerprint"] = stable_fingerprint(failure)
    _write_json_create_only(authority.artifact_directory / "failure.json", failure)
    return _seal_directory(authority, terminal_name="failure.json")


def preflight_profile(config: NLCCDatasetFreeRunnerConfig) -> dict[str, object]:
    """Run source/input structure checks only; never claim or optimize."""

    boundary = runtime_import_boundary()
    bindings = {
        RUNNER_PREREGISTRATION_REPO_PATH: RUNNER_PREREGISTRATION_FILE_SHA256,
        RUNNER_CLARIFICATION_REPO_PATH: RUNNER_CLARIFICATION_FILE_SHA256,
        PROFILE_INDEPENDENCE_REPO_PATH: PROFILE_INDEPENDENCE_FILE_SHA256,
        INPUT_FREEZE_REPO_PATH: INPUT_FREEZE_FILE_SHA256,
    }
    for relative, expected in bindings.items():
        path = _ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"frozen runner binding differs: {relative}")
    cache = materialize_profile(config)
    return {
        "method_id": METHOD_ID,
        "profile_id": config.profile.profile_id,
        "config_fingerprint": stable_fingerprint(config.manifest()),
        "materialized_cache": cache.manifest(),
        "runtime_import_boundary": boundary,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "gate_metrics_observed": False,
        "artifact_directory_created": False,
        "all_pass": True,
    }


def run_canonical_profile(
    config: NLCCDatasetFreeRunnerConfig,
    *,
    repo_root: Path = _ROOT,
) -> tuple[dict[str, object], int]:
    """Canonical CLI transaction; no output override and no automatic retry."""

    authorization = load_pre_run_authorization(config, repo_root=repo_root)
    cache = materialize_profile(config)
    authority = claim_execution(
        config,
        authorization,
        repo_root=repo_root,
    )
    try:
        result = execute_authorized_profile(authority, cache)
        sealed = publish_result(authority, result)
        return sealed, 0 if result["all_pass"] is True else 2
    except BaseException as error:
        sealed = publish_failure(authority, error)
        return sealed, 3


__all__ = [
    "ATTEMPT_SCHEMA",
    "COMPLETE_SCHEMA",
    "DECISION_SCHEMA",
    "ExecutionAuthority",
    "FAILURE_SCHEMA",
    "NLCCMaterializedProfile",
    "NLCCTrainingComponents",
    "PRE_RUN_AUTHORIZATION_SCHEMA",
    "PreRunAuthorization",
    "REQUIRED_AUTH_SOURCE_PATHS",
    "RESULT_SCHEMA",
    "build_training_components",
    "claim_execution",
    "decoder_fingerprint",
    "evaluate_cached_logits",
    "evaluate_trained_decoder",
    "execute_authorized_profile",
    "load_pre_run_authorization",
    "materialize_profile",
    "pre_run_authorization_payload",
    "preflight_profile",
    "publish_failure",
    "publish_result",
    "run_canonical_profile",
    "runtime_import_boundary",
    "verify_development_authorization_artifacts",
]
