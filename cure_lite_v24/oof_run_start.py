"""Persistent one-shot authorization for one real v24 OOF-4 execution.

The metadata-only authorization is created from a verified real-D_R
structural PASS receipt and the frozen OOF split.  Each fold then publishes
one create-only run-start marker after its train-only caches and schedule
have been sealed, but before either optimizer is allowed to step.

An existing marker is never adopted as a fresh capability.  Consequently a
crashed process cannot retry a fold, choose another output directory, or
silently replace the pre-bound schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
from typing import Final

from cure_lite.cache.schema import canonical_json, file_sha256, stable_fingerprint
from cure_lite.coverage_state_schedule import CoverageStateTrainingSchedule
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRSourceBinding,
)
from tools.gcr_pacre_v24_protocol import (
    VerifiedOOF4Split,
    require_verified_oof4_split,
)

from .artifact_io import (
    atomic_write_new_canonical_json,
    read_canonical_json,
    regular_file_receipt,
)
from .dr_gate import (
    GCR_PACRE_DR_RECEIPT_PATH,
    GCR_PACRE_DR_SOURCE_PATHS,
    GCR_PACRE_DR_PASS_DECISION,
    verify_gcr_pacre_dr_receipt,
)
from .oof_cache import (
    VerifiedOOFCacheArtifact,
    require_verified_oof_cache_artifact,
)
from .oof_split import (
    VerifiedOOFFoldClosure,
    require_verified_oof_fold_closure,
)
from .source_closure import (
    gcr_pacre_v24_source_closure_fingerprint,
    gcr_pacre_v24_source_closure_hashes,
)


OOF_EXECUTION_AUTHORIZATION_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-real-execution-authorization-v2"
)
OOF_TRAINING_RUN_START_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-fold-persistent-run-start-v1"
)
OOF_RUNTIME_RELATIVE_PATH: Final = (
    "runs/irstd1k_stage_a_seed42/"
    "cure_lite_gcr_pacre_v24_oof4_seed42_r1"
)
OOF_DR_RECEIPT_RELATIVE_PATH: Final = GCR_PACRE_DR_RECEIPT_PATH
OOF_DR_SOURCE_RELATIVE_PATHS: Final = dict(GCR_PACRE_DR_SOURCE_PATHS)
OOF_SEED: Final = 42
OOF_EPOCHS: Final = 10
OOF_STEPS_PER_EPOCH: Final = 40
OOF_UPDATES: Final = 400
OOF_PROCESS_INSTANCE_FINGERPRINT: Final = stable_fingerprint(
    {
        "schema_version": (
            "cure-lite-v24-oof-run-fold-process-instance-v1"
        ),
        "process_id": os.getpid(),
        "process_nonce": secrets.token_hex(32),
    }
)

_TOKEN_ISSUER = object()
_TOKEN_REGISTRY: dict[int, object] = {}


def _register(value: object) -> object:
    if getattr(value, "_issuer", None) is not _TOKEN_ISSUER:
        raise AssertionError("attempted to register unsigned OOF run token")
    identity = id(value)
    prior = _TOKEN_REGISTRY.get(identity)
    if prior is not None and prior is not value:
        raise RuntimeError("OOF run token identity collision")
    _TOKEN_REGISTRY[identity] = value
    return value


def _require(value: object, expected: type, *, name: str) -> object:
    if (
        type(value) is not expected
        or getattr(value, "_issuer", None) is not _TOKEN_ISSUER
        or _TOKEN_REGISTRY.get(id(value)) is not value
    ):
        raise TypeError(f"{name} must be an exact verifier-issued capability")
    return value


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def required_oof_runtime_root() -> Path:
    """Return the single frozen real OOF runtime directory."""

    return (_repository_root() / OOF_RUNTIME_RELATIVE_PATH).absolute()


def required_oof_dr_receipt_path() -> Path:
    return (_repository_root() / OOF_DR_RECEIPT_RELATIVE_PATH).absolute()


def required_oof_dr_source_paths() -> dict[str, Path]:
    return {
        name: (_repository_root() / relative).absolute()
        for name, relative in OOF_DR_SOURCE_RELATIVE_PATHS.items()
    }


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA256")
    return value


def _strict_payload(token_json: str, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(token_json)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{name} payload is invalid") from error
    if not isinstance(value, dict) or canonical_json(value) != token_json:
        raise RuntimeError(f"{name} payload is not canonical")
    return value


@dataclass(frozen=True, slots=True)
class VerifiedOOFExecutionAuthorization:
    payload_json: str
    authorization_fingerprint: str
    split_receipt_fingerprint: str
    source_binding_fingerprint: str
    d_r_receipt_fingerprint: str
    d_r_receipt_path: str
    d_r_receipt_file_sha256: str
    expected_real_inputs_fingerprint: str
    expected_full_cache_fingerprint: str
    source_closure_fingerprint: str
    runtime_root: str
    artifact_path: str
    artifact_file_sha256: str
    _issuer: object

    @property
    def payload(self) -> dict[str, object]:
        return _strict_payload(
            self.payload_json,
            name="OOF execution authorization",
        )


@dataclass(frozen=True, slots=True)
class VerifiedOOFTrainingRunStart:
    payload_json: str
    marker_fingerprint: str
    marker_path: str
    marker_file_sha256: str
    event_index: int
    fold_id: int
    closure_fingerprint: str
    authorization_fingerprint: str
    source_closure_fingerprint: str
    schedule_fingerprint: str
    training_population_fingerprint: str
    output_directory: str
    process_instance_fingerprint: str
    _issuer: object

    @property
    def payload(self) -> dict[str, object]:
        return _strict_payload(self.payload_json, name="OOF run-start")


def require_verified_oof_execution_authorization(
    value: object,
) -> VerifiedOOFExecutionAuthorization:
    result = _require(
        value,
        VerifiedOOFExecutionAuthorization,
        name="execution_authorization",
    )
    assert isinstance(result, VerifiedOOFExecutionAuthorization)
    d_r_receipt = read_canonical_json(result.d_r_receipt_path)
    if (
        result.payload.get("authorization_fingerprint")
        != result.authorization_fingerprint
        or file_sha256(Path(result.artifact_path))
        != result.artifact_file_sha256
        or read_canonical_json(result.artifact_path) != result.payload
        or result.source_closure_fingerprint
        != gcr_pacre_v24_source_closure_fingerprint()
        or file_sha256(Path(result.d_r_receipt_path))
        != result.d_r_receipt_file_sha256
        or verify_gcr_pacre_dr_receipt(d_r_receipt)
        != result.d_r_receipt_fingerprint
    ):
        raise RuntimeError("OOF execution authorization changed")
    return result


def require_verified_oof_training_run_start(
    value: object,
) -> VerifiedOOFTrainingRunStart:
    result = _require(
        value,
        VerifiedOOFTrainingRunStart,
        name="run_start_token",
    )
    assert isinstance(result, VerifiedOOFTrainingRunStart)
    if (
        result.payload.get("marker_fingerprint")
        != result.marker_fingerprint
        or result.payload.get("event_index") != 2
        or result.payload.get("event") != "training_claimed"
        or result.payload.get("process_instance_fingerprint")
        != result.process_instance_fingerprint
        or result.event_index != 2
        or file_sha256(Path(result.marker_path))
        != result.marker_file_sha256
        or read_canonical_json(result.marker_path) != result.payload
        or result.source_closure_fingerprint
        != gcr_pacre_v24_source_closure_fingerprint()
    ):
        raise RuntimeError("OOF persistent run-start changed")
    return result


def authorize_real_oof4_execution_new(
    *,
    verified_split: VerifiedOOF4Split,
    source_binding: CoverageStateRealDRSourceBinding,
    runtime_root: str | Path | None = None,
) -> VerifiedOOFExecutionAuthorization:
    """Create the one real OOF authorization before any fold attempt."""

    split = require_verified_oof4_split(verified_split)
    if type(source_binding) is not CoverageStateRealDRSourceBinding:
        raise TypeError(
            "source_binding must be exact CoverageStateRealDRSourceBinding"
        )
    source_binding.verify_unchanged()
    fixed_source_paths = required_oof_dr_source_paths()
    if {
        "manifest_path": source_binding.manifest_path,
        "state_index_path": source_binding.state_index_path,
        "geometry_config_path": source_binding.geometry_config_path,
        "geometry_receipt_path": source_binding.geometry_receipt_path,
        "observability_config_path": source_binding.observability_config_path,
    } != fixed_source_paths:
        raise PermissionError("OOF D_R metadata source paths are frozen")
    receipt_path = required_oof_dr_receipt_path()
    d_r_receipt = read_canonical_json(receipt_path)
    receipt_artifact = regular_file_receipt(receipt_path)
    receipt_fingerprint = verify_gcr_pacre_dr_receipt(d_r_receipt)
    decision = d_r_receipt.get("decision")
    input_binding = d_r_receipt.get("input_binding")
    if not isinstance(decision, dict) or (
        decision.get("status") != GCR_PACRE_DR_PASS_DECISION
        and decision.get("decision") != GCR_PACRE_DR_PASS_DECISION
    ):
        raise PermissionError("OOF requires a real D_R structural PASS")
    if not isinstance(input_binding, dict):
        raw = d_r_receipt.get("raw_observations")
        input_binding = (
            raw.get("input_binding")
            if isinstance(raw, dict)
            else None
        )
    if (
        not isinstance(input_binding, dict)
        or input_binding.get("source_binding_fingerprint")
        != source_binding.binding_fingerprint
    ):
        raise PermissionError("D_R receipt and metadata binding differ")
    expected_inputs = _sha256(
        d_r_receipt.get("real_inputs_fingerprint"),
        name="real_inputs_fingerprint",
    )
    expected_cache = _sha256(
        d_r_receipt.get("cache_fingerprint"),
        name="cache_fingerprint",
    )
    requested = (
        required_oof_runtime_root()
        if runtime_root is None
        else Path(runtime_root).absolute()
    )
    if requested != required_oof_runtime_root():
        raise PermissionError("real OOF runtime root is frozen")
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(
            "real OOF authorization/output already exists; no retry or "
            "alternate output is permitted"
        )
    requested.mkdir(parents=False, exist_ok=False)
    sources = gcr_pacre_v24_source_closure_hashes()
    body = {
        "schema_version": OOF_EXECUTION_AUTHORIZATION_SCHEMA,
        "stage": "D_R_source_group_OOF4",
        "seed": OOF_SEED,
        "epochs": OOF_EPOCHS,
        "steps_per_epoch": OOF_STEPS_PER_EPOCH,
        "updates_per_arm_per_fold": OOF_UPDATES,
        "split_receipt_fingerprint": split.receipt_fingerprint,
        "plan_fingerprint": split.plan_fingerprint,
        "root_by_sample_fingerprint": split.root_by_sample_fingerprint,
        "D_R_structural_receipt_fingerprint": receipt_fingerprint,
        "D_R_structural_receipt_artifact": {
            **receipt_artifact,
            "receipt_fingerprint": receipt_fingerprint,
        },
        "source_binding_fingerprint": source_binding.binding_fingerprint,
        "source_binding": source_binding.canonical_payload(),
        "expected_real_inputs_fingerprint": expected_inputs,
        "expected_full_cache_fingerprint": expected_cache,
        "source_hashes": dict(sources),
        "source_closure_fingerprint": (
            gcr_pacre_v24_source_closure_fingerprint(sources)
        ),
        "runtime_root": str(requested),
        "fold_output_directories": [
            str(requested / f"fold_{fold_id}") for fold_id in range(4)
        ],
        "from_scratch": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "checkpoint_policy": "final_only",
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    fingerprint = stable_fingerprint(body)
    payload = {**body, "authorization_fingerprint": fingerprint}
    artifact = atomic_write_new_canonical_json(
        requested / "authorization.json",
        payload,
    )
    for fold_id in range(4):
        fold = requested / f"fold_{fold_id}"
        fold.mkdir()
        for partition in ("train", "holdout"):
            for arm in ("base_eval", "v23_control", "candidate"):
                (fold / partition / arm).mkdir(parents=True)
        (fold / "terminal").mkdir()
        (fold / "evaluation").mkdir()
    receipt = regular_file_receipt(artifact)
    return _register(VerifiedOOFExecutionAuthorization(
        payload_json=canonical_json(payload),
        authorization_fingerprint=fingerprint,
        split_receipt_fingerprint=split.receipt_fingerprint,
        source_binding_fingerprint=source_binding.binding_fingerprint,
        d_r_receipt_fingerprint=receipt_fingerprint,
        d_r_receipt_path=str(receipt_path),
        d_r_receipt_file_sha256=str(receipt_artifact["file_sha256"]),
        expected_real_inputs_fingerprint=expected_inputs,
        expected_full_cache_fingerprint=expected_cache,
        source_closure_fingerprint=str(
            payload["source_closure_fingerprint"]
        ),
        runtime_root=str(requested),
        artifact_path=str(artifact),
        artifact_file_sha256=str(receipt["file_sha256"]),
        _issuer=_TOKEN_ISSUER,
    ))


def load_and_verify_real_oof4_execution_authorization(
    *,
    verified_split: VerifiedOOF4Split,
) -> VerifiedOOFExecutionAuthorization:
    """Reissue the live authorization solely from its frozen disk chain.

    The caller supplies no receipt/output path.  The authorization artifact
    fixes the D_R receipt path at issuance, and this verifier rereads that
    canonical receipt and reconstructs the metadata-only source binding from
    paths sealed inside the authorization.  It never opens D_V or D_T.
    """

    split = require_verified_oof4_split(verified_split)
    runtime_root = required_oof_runtime_root()
    artifact_path = runtime_root / "authorization.json"
    payload = read_canonical_json(artifact_path)
    expected_fields = {
        "schema_version",
        "stage",
        "seed",
        "epochs",
        "steps_per_epoch",
        "updates_per_arm_per_fold",
        "split_receipt_fingerprint",
        "plan_fingerprint",
        "root_by_sample_fingerprint",
        "D_R_structural_receipt_fingerprint",
        "D_R_structural_receipt_artifact",
        "source_binding_fingerprint",
        "source_binding",
        "expected_real_inputs_fingerprint",
        "expected_full_cache_fingerprint",
        "source_hashes",
        "source_closure_fingerprint",
        "runtime_root",
        "fold_output_directories",
        "from_scratch",
        "resume_allowed",
        "automatic_retry_allowed",
        "checkpoint_policy",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "authorization_fingerprint",
    }
    if set(payload) != expected_fields:
        raise ValueError("OOF authorization artifact fields changed")
    body = dict(payload)
    authorization_fingerprint = body.pop(
        "authorization_fingerprint",
        None,
    )
    if (
        not isinstance(authorization_fingerprint, str)
        or authorization_fingerprint != stable_fingerprint(body)
    ):
        raise ValueError("OOF authorization fingerprint changed")
    receipt_artifact = payload.get("D_R_structural_receipt_artifact")
    expected_receipt_fields = {
        "path",
        "size_bytes",
        "file_sha256",
        "device",
        "inode",
        "hardlink_count",
        "receipt_fingerprint",
    }
    if (
        not isinstance(receipt_artifact, dict)
        or set(receipt_artifact) != expected_receipt_fields
        or receipt_artifact.get("hardlink_count") != 1
    ):
        raise ValueError("OOF D_R receipt artifact binding changed")
    d_r_path = Path(str(receipt_artifact["path"]))
    if d_r_path != required_oof_dr_receipt_path():
        raise PermissionError("OOF D_R structural receipt path is frozen")
    d_r_receipt = read_canonical_json(d_r_path)
    d_r_regular = regular_file_receipt(d_r_path)
    d_r_fingerprint = verify_gcr_pacre_dr_receipt(d_r_receipt)
    source_binding_payload = payload.get("source_binding")
    if not isinstance(source_binding_payload, dict):
        raise TypeError("OOF authorization source binding must be a mapping")
    source_paths = source_binding_payload.get("paths")
    if not isinstance(source_paths, dict) or set(source_paths) != {
        "manifest",
        "state_index",
        "geometry_config",
        "geometry_receipt",
        "observability_config",
    }:
        raise ValueError("OOF authorization source paths changed")
    source_path_key_map = {
        "manifest_path": "manifest",
        "state_index_path": "state_index",
        "geometry_config_path": "geometry_config",
        "geometry_receipt_path": "geometry_receipt",
        "observability_config_path": "observability_config",
    }
    if {
        argument: Path(str(source_paths[payload_key]))
        for argument, payload_key in source_path_key_map.items()
    } != required_oof_dr_source_paths():
        raise PermissionError("OOF authorization source paths are not frozen")
    from cure_lite.experiment.coverage_state_real_dr_inputs import (
        bind_coverage_state_real_dr_sources,
    )

    source_binding, _, _, _ = bind_coverage_state_real_dr_sources(
        manifest_path=str(source_paths["manifest"]),
        state_index_path=str(source_paths["state_index"]),
        geometry_config_path=str(source_paths["geometry_config"]),
        geometry_receipt_path=str(source_paths["geometry_receipt"]),
        observability_config_path=str(source_paths["observability_config"]),
    )
    if source_binding.canonical_payload() != source_binding_payload:
        raise PermissionError("OOF authorization source binding changed")
    decision = d_r_receipt.get("decision")
    input_binding = d_r_receipt.get("input_binding")
    if not isinstance(input_binding, dict):
        raw = d_r_receipt.get("raw_observations")
        input_binding = (
            raw.get("input_binding") if isinstance(raw, dict) else None
        )
    source_rows = gcr_pacre_v24_source_closure_hashes()
    expected_folds = [
        str(runtime_root / f"fold_{fold_id}") for fold_id in range(4)
    ]
    if (
        payload.get("schema_version")
        != OOF_EXECUTION_AUTHORIZATION_SCHEMA
        or payload.get("stage") != "D_R_source_group_OOF4"
        or (
            payload.get("seed"),
            payload.get("epochs"),
            payload.get("steps_per_epoch"),
            payload.get("updates_per_arm_per_fold"),
        )
        != (OOF_SEED, OOF_EPOCHS, OOF_STEPS_PER_EPOCH, OOF_UPDATES)
        or payload.get("split_receipt_fingerprint")
        != split.receipt_fingerprint
        or payload.get("plan_fingerprint") != split.plan_fingerprint
        or payload.get("root_by_sample_fingerprint")
        != split.root_by_sample_fingerprint
        or payload.get("D_R_structural_receipt_fingerprint")
        != d_r_fingerprint
        or receipt_artifact.get("receipt_fingerprint")
        != d_r_fingerprint
        or {
            key: receipt_artifact[key]
            for key in (
                "path",
                "size_bytes",
                "file_sha256",
                "device",
                "inode",
                "hardlink_count",
            )
        }
        != d_r_regular
        or not isinstance(decision, dict)
        or (
            decision.get("status") != GCR_PACRE_DR_PASS_DECISION
            and decision.get("decision") != GCR_PACRE_DR_PASS_DECISION
        )
        or not isinstance(input_binding, dict)
        or input_binding.get("source_binding_fingerprint")
        != source_binding.binding_fingerprint
        or payload.get("source_binding_fingerprint")
        != source_binding.binding_fingerprint
        or payload.get("expected_real_inputs_fingerprint")
        != d_r_receipt.get("real_inputs_fingerprint")
        or payload.get("expected_full_cache_fingerprint")
        != d_r_receipt.get("cache_fingerprint")
        or payload.get("source_hashes") != dict(source_rows)
        or payload.get("source_closure_fingerprint")
        != gcr_pacre_v24_source_closure_fingerprint(source_rows)
        or payload.get("runtime_root") != str(runtime_root)
        or payload.get("fold_output_directories") != expected_folds
        or payload.get("from_scratch") is not True
        or payload.get("resume_allowed") is not False
        or payload.get("automatic_retry_allowed") is not False
        or payload.get("checkpoint_policy") != "final_only"
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("OOF persisted execution authorization changed")
    artifact_receipt = regular_file_receipt(artifact_path)
    if (
        artifact_path != runtime_root / "authorization.json"
        or artifact_path.stat().st_mode & 0o777 != 0o444
    ):
        raise PermissionError("OOF authorization path/mode changed")
    for fold_id, expected in enumerate(expected_folds):
        fold = Path(expected)
        if (
            not fold.is_dir()
            or fold.is_symlink()
            or fold.resolve(strict=True) != fold
            or fold.name != f"fold_{fold_id}"
        ):
            raise RuntimeError("OOF fold output directory changed")
    return _register(VerifiedOOFExecutionAuthorization(
        payload_json=canonical_json(payload),
        authorization_fingerprint=authorization_fingerprint,
        split_receipt_fingerprint=split.receipt_fingerprint,
        source_binding_fingerprint=source_binding.binding_fingerprint,
        d_r_receipt_fingerprint=d_r_fingerprint,
        d_r_receipt_path=str(d_r_path),
        d_r_receipt_file_sha256=str(
            receipt_artifact["file_sha256"]
        ),
        expected_real_inputs_fingerprint=str(
            payload["expected_real_inputs_fingerprint"]
        ),
        expected_full_cache_fingerprint=str(
            payload["expected_full_cache_fingerprint"]
        ),
        source_closure_fingerprint=str(
            payload["source_closure_fingerprint"]
        ),
        runtime_root=str(runtime_root),
        artifact_path=str(artifact_path),
        artifact_file_sha256=str(artifact_receipt["file_sha256"]),
        _issuer=_TOKEN_ISSUER,
    ))


def create_oof_training_run_start_new(
    execution_authorization: VerifiedOOFExecutionAuthorization,
    fold_closure: VerifiedOOFFoldClosure,
    *,
    schedule: CoverageStateTrainingSchedule,
    control_cache_artifact: VerifiedOOFCacheArtifact,
    candidate_cache_artifact: VerifiedOOFCacheArtifact,
) -> VerifiedOOFTrainingRunStart:
    """Persist one non-repeatable fold intent immediately before training."""

    authorization = require_verified_oof_execution_authorization(
        execution_authorization
    )
    closure = require_verified_oof_fold_closure(fold_closure)
    control = require_verified_oof_cache_artifact(control_cache_artifact)
    candidate = require_verified_oof_cache_artifact(candidate_cache_artifact)
    if type(schedule) is not CoverageStateTrainingSchedule:
        raise TypeError("schedule must be exact CoverageStateTrainingSchedule")
    if (
        closure.split_receipt_fingerprint
        != authorization.split_receipt_fingerprint
        or control.fold_id != closure.fold_id
        or candidate.fold_id != closure.fold_id
        or control.partition != "train"
        or candidate.partition != "train"
        or control.arm != "PACRE_VC_v23_control"
        or candidate.arm != "GCR_PACRE_v24"
        or (
            schedule.config.seed,
            schedule.config.epochs,
            schedule.config.steps_per_epoch,
        )
        != (OOF_SEED, OOF_EPOCHS, OOF_STEPS_PER_EPOCH)
        or schedule.cache_fingerprint
        != control.payload.get("semantic_payload_fingerprint")
        or schedule.cache_fingerprint
        != candidate.payload.get("semantic_payload_fingerprint")
    ):
        raise PermissionError("OOF run-start cache/schedule binding changed")
    output = Path(authorization.runtime_root) / f"fold_{closure.fold_id}"
    marker_path = output / "run_start.json"
    body = {
        "schema_version": OOF_TRAINING_RUN_START_SCHEMA,
        "fold_id": closure.fold_id,
        "closure_fingerprint": closure.closure_fingerprint,
        "split_receipt_fingerprint": closure.split_receipt_fingerprint,
        "authorization_fingerprint": (
            authorization.authorization_fingerprint
        ),
        "authorization_artifact_file_sha256": (
            authorization.artifact_file_sha256
        ),
        "source_binding_fingerprint": (
            authorization.source_binding_fingerprint
        ),
        "source_closure_fingerprint": (
            authorization.source_closure_fingerprint
        ),
        "seed": OOF_SEED,
        "epochs": OOF_EPOCHS,
        "steps_per_epoch": OOF_STEPS_PER_EPOCH,
        "updates_per_arm": OOF_UPDATES,
        "event_index": 2,
        "event": "training_claimed",
        "process_instance_fingerprint": (
            OOF_PROCESS_INSTANCE_FINGERPRINT
        ),
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "batch_sequence_fingerprint": stable_fingerprint(
            [
                selection.selection_fingerprint
                for selection in schedule.selections
            ]
        ),
        "training_population_fingerprint": schedule.cache_fingerprint,
        "control_cache_artifact_fingerprint": (
            control.artifact_fingerprint
        ),
        "candidate_cache_artifact_fingerprint": (
            candidate.artifact_fingerprint
        ),
        "output_directory": str(output),
        "marker_path": str(marker_path),
        "from_scratch": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "checkpoint_policy": "final_only",
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    fingerprint = stable_fingerprint(body)
    payload = {**body, "marker_fingerprint": fingerprint}
    marker = atomic_write_new_canonical_json(marker_path, payload)
    marker_receipt = regular_file_receipt(marker)
    return _register(VerifiedOOFTrainingRunStart(
        payload_json=canonical_json(payload),
        marker_fingerprint=fingerprint,
        marker_path=str(marker),
        marker_file_sha256=str(marker_receipt["file_sha256"]),
        event_index=2,
        fold_id=closure.fold_id,
        closure_fingerprint=closure.closure_fingerprint,
        authorization_fingerprint=authorization.authorization_fingerprint,
        source_closure_fingerprint=authorization.source_closure_fingerprint,
        schedule_fingerprint=schedule.schedule_fingerprint,
        training_population_fingerprint=schedule.cache_fingerprint,
        output_directory=str(output),
        process_instance_fingerprint=OOF_PROCESS_INSTANCE_FINGERPRINT,
        _issuer=_TOKEN_ISSUER,
    ))


def run_start_artifact_receipt(
    token: VerifiedOOFTrainingRunStart,
) -> dict[str, object]:
    """Return the exact persistent wrapper consumed by fold verification."""

    value = require_verified_oof_training_run_start(token)
    receipt = regular_file_receipt(value.marker_path)
    return {
        **receipt,
        "marker_fingerprint": value.marker_fingerprint,
    }


__all__ = [
    "OOF_EPOCHS",
    "OOF_DR_RECEIPT_RELATIVE_PATH",
    "OOF_DR_SOURCE_RELATIVE_PATHS",
    "OOF_EXECUTION_AUTHORIZATION_SCHEMA",
    "OOF_RUNTIME_RELATIVE_PATH",
    "OOF_PROCESS_INSTANCE_FINGERPRINT",
    "OOF_SEED",
    "OOF_STEPS_PER_EPOCH",
    "OOF_TRAINING_RUN_START_SCHEMA",
    "OOF_UPDATES",
    "VerifiedOOFExecutionAuthorization",
    "VerifiedOOFTrainingRunStart",
    "authorize_real_oof4_execution_new",
    "create_oof_training_run_start_new",
    "load_and_verify_real_oof4_execution_authorization",
    "require_verified_oof_execution_authorization",
    "require_verified_oof_training_run_start",
    "required_oof_runtime_root",
    "required_oof_dr_receipt_path",
    "required_oof_dr_source_paths",
    "run_start_artifact_receipt",
]
