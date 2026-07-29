"""Create-only terminal artifacts and receipts for v24 Formal800.

Only the final GCR-PACRE float32 model state is serialized.  Optimizer state,
intermediate checkpoints, pickle payloads, D_V data, and D_T data are outside
this module's contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final, Mapping

import torch

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.experiment.coverage_state_training import (
    coverage_state_model_fingerprint,
)
from tools.gcr_pacre_v24_protocol import (
    VerifiedFormalTraining,
    validate_formal_training_receipt,
)

from .artifact_io import (
    atomic_write_new_canonical_json,
    load_terminal_safetensors_strict,
    read_canonical_json,
    save_terminal_safetensors_new,
)
from .factory import (
    GCR_PACRE_FORMAL_PARAMETER_COUNT,
    GCR_PACRE_PARAMETER_NAMES,
    build_formal_gcr_pacre_training_model,
)
from .formal_cache_artifacts import (
    require_verified_formal_cache_artifact,
    verify_formal_cache_artifact,
)
from .formal_training import (
    GCR_PACRE_FORMAL_EPOCHS,
    GCR_PACRE_FORMAL_STEPS_PER_EPOCH,
    GCR_PACRE_FORMAL_UPDATES,
    GCRPACREFormalAuthorization,
    GCRPACREFormalRunResult,
)
from .formal_run_start import (
    verify_gcr_pacre_formal_run_start_token,
)
from .gcr_pacre import (
    CURELiteGatedCommonResidualPACRELevelSet,
)
from .source_closure import (
    GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
    gcr_pacre_v24_source_closure_fingerprint,
)
from .training_trace import trace_finite_audit


GCR_PACRE_FORMAL_MODEL_ARTIFACT_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-formal800-final-model-v1"
)
GCR_PACRE_FORMAL_EVIDENCE_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-formal800-evidence-v6"
)
GCR_PACRE_FORMAL_MODEL_FILE: Final = "model.safetensors"
GCR_PACRE_FORMAL_ARTIFACT_RECEIPT_FILE: Final = "artifact.json"
GCR_PACRE_FORMAL_SCHEDULE_POLICY_SCHEMA: Final = (
    "cure-lite-v24-formal800-schedule-policy-without-seed-v1"
)
_ARTIFACT_FIELDS: Final = {
    "schema_version",
    "serialization",
    "seed",
    "role",
    "model_file",
    "model_file_absolute_path",
    "model_file_size_bytes",
    "model_file_sha256",
    "model_fingerprint",
    "model_config",
    "state_keys",
    "state_shapes",
    "state_dtypes",
    "parameter_count",
    "authorization_fingerprint",
    "training_receipt_fingerprint",
    "training_result_fingerprint",
    "formal_result_fingerprint",
    "terminal_D_R_evaluation_fingerprint",
    "source_hashes",
    "final_checkpoint_only",
    "optimizer_state_saved",
    "intermediate_checkpoint_saved",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "artifact_fingerprint",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _formal_schedule_policy_without_seed(
    semantic_cache_fingerprint: str,
) -> dict[str, object]:
    if not _is_sha256(semantic_cache_fingerprint):
        raise ValueError("semantic cache fingerprint is malformed")
    return {
        "schema_version": GCR_PACRE_FORMAL_SCHEDULE_POLICY_SCHEMA,
        "semantic_cache_fingerprint": semantic_cache_fingerprint,
        "epochs": GCR_PACRE_FORMAL_EPOCHS,
        "steps_per_epoch": GCR_PACRE_FORMAL_STEPS_PER_EPOCH,
        "updates": GCR_PACRE_FORMAL_UPDATES,
        "logical_states_per_update": 12,
        "objective_invariant": True,
        "optimizer_exposure_accounting": (
            "recomputed_against_current_cache_before_use"
        ),
    }


def save_gcr_pacre_formal_schedule_atomic(
    path: str | Path,
    *,
    authorization: GCRPACREFormalAuthorization,
) -> dict[str, object]:
    """Seal the exact schedule bytes used by one Formal authorization."""

    if type(authorization) is not GCRPACREFormalAuthorization:
        raise TypeError("authorization must be exact Formal authorization")
    authorization.verify_unchanged()
    supplied = Path(path)
    if not supplied.is_absolute():
        supplied = (Path.cwd() / supplied).resolve(strict=False)
    required = Path(
        str(authorization.chain_run_binding["schedule_artifact_path"])
    )
    if supplied != required:
        raise PermissionError(
            f"Formal schedule path must be exactly {required}"
        )
    target = atomic_write_new_canonical_json(
        supplied,
        authorization.schedule.canonical_payload(),
    )
    stat_result = target.stat()
    policy = _formal_schedule_policy_without_seed(
        authorization.cache.cache_fingerprint
    )
    return {
        "path": str(target.resolve(strict=True)),
        "size_bytes": stat_result.st_size,
        "file_sha256": file_sha256(target),
        "schedule_fingerprint": (
            authorization.schedule.schedule_fingerprint
        ),
        "seed": authorization.seed,
        "epochs": GCR_PACRE_FORMAL_EPOCHS,
        "steps_per_epoch": GCR_PACRE_FORMAL_STEPS_PER_EPOCH,
        "updates": GCR_PACRE_FORMAL_UPDATES,
        "semantic_cache_fingerprint": (
            authorization.cache.cache_fingerprint
        ),
        "policy_without_seed_fingerprint": stable_fingerprint(policy),
    }


def _validate_schedule_artifact(
    value: Mapping[str, object],
    authorization: GCRPACREFormalAuthorization,
) -> dict[str, object]:
    expected_fields = {
        "path",
        "size_bytes",
        "file_sha256",
        "schedule_fingerprint",
        "seed",
        "epochs",
        "steps_per_epoch",
        "updates",
        "semantic_cache_fingerprint",
        "policy_without_seed_fingerprint",
    }
    payload = dict(value)
    path = Path(str(payload.get("path")))
    policy = _formal_schedule_policy_without_seed(
        authorization.cache.cache_fingerprint
    )
    if (
        set(payload) != expected_fields
        or not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or path.stat().st_size != payload.get("size_bytes")
        or file_sha256(path) != payload.get("file_sha256")
        or read_canonical_json(path)
        != authorization.schedule.canonical_payload()
        or payload.get("schedule_fingerprint")
        != authorization.schedule.schedule_fingerprint
        or payload.get("seed") != authorization.seed
        or payload.get("epochs") != GCR_PACRE_FORMAL_EPOCHS
        or payload.get("steps_per_epoch")
        != GCR_PACRE_FORMAL_STEPS_PER_EPOCH
        or payload.get("updates") != GCR_PACRE_FORMAL_UPDATES
        or payload.get("semantic_cache_fingerprint")
        != authorization.cache.cache_fingerprint
        or payload.get("policy_without_seed_fingerprint")
        != stable_fingerprint(policy)
    ):
        raise ValueError("Formal schedule artifact binding changed")
    return payload


def _validate_artifact_payload(
    directory: Path,
    payload: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    value = dict(payload)
    fingerprint = value.get("artifact_fingerprint")
    body = dict(value)
    body.pop("artifact_fingerprint", None)
    model_path = directory / GCR_PACRE_FORMAL_MODEL_FILE
    if (
        set(value) != _ARTIFACT_FIELDS
        or value.get("schema_version")
        != GCR_PACRE_FORMAL_MODEL_ARTIFACT_SCHEMA
        or value.get("serialization") != "safetensors"
        or value.get("seed") not in {42, 43}
        or value.get("role")
        not in {"primary", "training_integrity_only"}
        or (value.get("seed"), value.get("role"))
        not in {
            (42, "primary"),
            (43, "training_integrity_only"),
        }
        or value.get("model_file") != GCR_PACRE_FORMAL_MODEL_FILE
        or value.get("model_file_absolute_path")
        != str(model_path.resolve(strict=True))
        or model_path.is_symlink()
        or not model_path.is_file()
        or model_path.resolve(strict=True) != model_path
        or model_path.stat().st_size
        != value.get("model_file_size_bytes")
        or file_sha256(model_path) != value.get("model_file_sha256")
        or not _is_sha256(value.get("model_fingerprint"))
        or value.get("model_config")
        != {
            "feature_channels": 64,
            "feature_stride": 4,
            "width": 32,
            "parameter_count": GCR_PACRE_FORMAL_PARAMETER_COUNT,
        }
        or value.get("state_keys") != list(GCR_PACRE_PARAMETER_NAMES)
        or value.get("parameter_count")
        != GCR_PACRE_FORMAL_PARAMETER_COUNT
        or not all(
            _is_sha256(value.get(name))
            for name in (
                "authorization_fingerprint",
                "training_receipt_fingerprint",
                "training_result_fingerprint",
                "formal_result_fingerprint",
                "terminal_D_R_evaluation_fingerprint",
            )
        )
        or not isinstance(value.get("source_hashes"), dict)
        or not value.get("source_hashes")
        or value.get("final_checkpoint_only") is not True
        or value.get("optimizer_state_saved") is not False
        or value.get("intermediate_checkpoint_saved") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or not _is_sha256(fingerprint)
        or stable_fingerprint(body) != fingerprint
    ):
        raise ValueError("Formal terminal artifact receipt is invalid")
    return model_path, value


@dataclass(frozen=True, eq=False)
class LoadedGCRPACREFormalTerminal:
    directory: Path
    model: CURELiteGatedCommonResidualPACRELevelSet
    receipt_json: str

    @property
    def receipt(self) -> dict[str, object]:
        value = json.loads(self.receipt_json)
        if not isinstance(value, dict):
            raise AssertionError("validated Formal receipt changed")
        return value

    @property
    def artifact_fingerprint(self) -> str:
        value = self.receipt["artifact_fingerprint"]
        if not isinstance(value, str):
            raise AssertionError("validated artifact fingerprint changed")
        return value

    def verify_unchanged(self) -> None:
        _load_terminal_direct(self.directory, expected_receipt=self.receipt)
        if (
            coverage_state_model_fingerprint(self.model)
            != self.receipt["model_fingerprint"]
        ):
            raise RuntimeError("loaded Formal model changed in memory")


def _load_terminal_direct(
    directory: str | Path,
    *,
    expected_receipt: Mapping[str, object] | None,
) -> LoadedGCRPACREFormalTerminal:
    root = Path(directory)
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
    ):
        raise RuntimeError("Formal artifact directory is not canonical")
    receipt_path = root / GCR_PACRE_FORMAL_ARTIFACT_RECEIPT_FILE
    payload = read_canonical_json(receipt_path)
    if expected_receipt is not None and payload != dict(expected_receipt):
        raise PermissionError("Formal artifact receipt differs from expected")
    model_path, payload = _validate_artifact_payload(root, payload)
    state = load_terminal_safetensors_strict(model_path)
    if (
        set(state) != set(GCR_PACRE_PARAMETER_NAMES)
        or {
            name: list(tensor.shape) for name, tensor in state.items()
        }
        != payload.get("state_shapes")
        or {
            name: str(tensor.dtype) for name, tensor in state.items()
        }
        != payload.get("state_dtypes")
        or sum(tensor.numel() for tensor in state.values())
        != GCR_PACRE_FORMAL_PARAMETER_COUNT
    ):
        raise ValueError("Formal safetensors state contract changed")
    model = build_formal_gcr_pacre_training_model()
    model.load_state_dict(state, strict=True)
    model.eval()
    if coverage_state_model_fingerprint(model) != payload["model_fingerprint"]:
        raise RuntimeError("Formal safetensors model fingerprint changed")
    return LoadedGCRPACREFormalTerminal(
        directory=root,
        model=model,
        receipt_json=canonical_json(payload),
    )


def save_gcr_pacre_formal_terminal_atomic(
    directory: str | Path,
    *,
    formal_result: GCRPACREFormalRunResult,
) -> dict[str, object]:
    """Create one final-only model directory with no replacement path."""

    if type(formal_result) is not GCRPACREFormalRunResult:
        raise TypeError("formal_result must be exact Formal run result")
    formal_result.verify_unchanged()
    root = Path(directory)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve(strict=False)
    required = Path(
        str(
            formal_result.authorization.chain_run_binding[
                "terminal_artifact_directory"
            ]
        )
    )
    if root != required:
        raise PermissionError(
            f"Formal terminal path must be exactly {required}"
        )
    root.mkdir(parents=True, exist_ok=False)
    root = root.resolve(strict=True)
    model = formal_result.model
    final_fingerprint = coverage_state_model_fingerprint(model)
    saved = save_terminal_safetensors_new(
        root / GCR_PACRE_FORMAL_MODEL_FILE,
        model,
        metadata={
            "schema": GCR_PACRE_FORMAL_MODEL_ARTIFACT_SCHEMA,
            "run": f"formal800-seed{formal_result.seed}",
            "seed": str(formal_result.seed),
            "role": formal_result.role,
            "model_fingerprint": final_fingerprint,
            "epochs": str(GCR_PACRE_FORMAL_EPOCHS),
            "updates": str(GCR_PACRE_FORMAL_UPDATES),
            "checkpoint_policy": "final_only",
        },
    )
    body = {
        "schema_version": GCR_PACRE_FORMAL_MODEL_ARTIFACT_SCHEMA,
        "serialization": "safetensors",
        "seed": formal_result.seed,
        "role": formal_result.role,
        "model_file": GCR_PACRE_FORMAL_MODEL_FILE,
        "model_file_absolute_path": saved["path"],
        "model_file_size_bytes": saved["size_bytes"],
        "model_file_sha256": saved["file_sha256"],
        "model_fingerprint": final_fingerprint,
        "model_config": {
            "feature_channels": 64,
            "feature_stride": 4,
            "width": 32,
            "parameter_count": GCR_PACRE_FORMAL_PARAMETER_COUNT,
        },
        "state_keys": saved["state_keys"],
        "state_shapes": saved["state_shapes"],
        "state_dtypes": saved["state_dtypes"],
        "parameter_count": saved["parameter_count"],
        "authorization_fingerprint": (
            formal_result.authorization.authorization_fingerprint
        ),
        "training_receipt_fingerprint": (
            formal_result.training_receipt.receipt_fingerprint
        ),
        "training_result_fingerprint": (
            formal_result.training_result.result_fingerprint
        ),
        "formal_result_fingerprint": formal_result.result_fingerprint,
        "terminal_D_R_evaluation_fingerprint": (
            formal_result.terminal_evaluation.evaluation_fingerprint
        ),
        "source_hashes": dict(formal_result.source_hashes_after),
        "final_checkpoint_only": True,
        "optimizer_state_saved": False,
        "intermediate_checkpoint_saved": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    payload = {
        **body,
        "artifact_fingerprint": stable_fingerprint(body),
    }
    atomic_write_new_canonical_json(
        root / GCR_PACRE_FORMAL_ARTIFACT_RECEIPT_FILE,
        payload,
    )
    loaded = _load_terminal_direct(root, expected_receipt=payload)
    if (
        coverage_state_model_fingerprint(loaded.model)
        != final_fingerprint
    ):
        raise RuntimeError("Formal terminal roundtrip changed model state")
    return payload


def load_and_verify_gcr_pacre_formal_terminal(
    directory: str | Path,
    *,
    expected_receipt: Mapping[str, object] | None = None,
) -> LoadedGCRPACREFormalTerminal:
    """Strict verify-only load of one final-only safetensors directory."""

    loaded = _load_terminal_direct(
        directory,
        expected_receipt=expected_receipt,
    )
    loaded.verify_unchanged()
    return loaded


def _terminal_protocol_artifact(
    artifact_receipt: Mapping[str, object],
) -> dict[str, object]:
    return {
        "path": artifact_receipt["model_file_absolute_path"],
        "size_bytes": artifact_receipt["model_file_size_bytes"],
        "file_sha256": artifact_receipt["model_file_sha256"],
        "model_fingerprint": artifact_receipt["model_fingerprint"],
    }


def build_formal_evidence_receipt(
    formal_result: GCRPACREFormalRunResult,
    *,
    schedule_artifact: Mapping[str, object],
    terminal_artifact_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact protocol v6 outer Formal evidence receipt."""

    if type(formal_result) is not GCRPACREFormalRunResult:
        raise TypeError("formal_result must be exact Formal result")
    formal_result.verify_unchanged()
    authorization = formal_result.authorization
    schedule = _validate_schedule_artifact(
        schedule_artifact,
        authorization,
    )
    artifact_root = Path(
        str(terminal_artifact_receipt["model_file_absolute_path"])
    ).parent
    loaded = _load_terminal_direct(
        artifact_root,
        expected_receipt=terminal_artifact_receipt,
    )
    if (
        loaded.receipt["formal_result_fingerprint"]
        != formal_result.result_fingerprint
        or loaded.receipt["model_fingerprint"]
        != formal_result.training_receipt.final_model_fingerprint
    ):
        raise PermissionError("Formal result/artifact binding changed")
    cache_token = require_verified_formal_cache_artifact(
        authorization.cache_artifact
    )
    reverified_cache = verify_formal_cache_artifact(
        cache_token.path,
        cache_id=cache_token.cache_id,
        expected_semantic_cache_fingerprint=(
            cache_token.semantic_cache_fingerprint
        ),
        expected_neutral_payload_fingerprint=(
            cache_token.neutral_payload_fingerprint
        ),
    )
    if (
        reverified_cache.receipt_fingerprint
        != cache_token.receipt_fingerprint
    ):
        raise RuntimeError("Formal cache artifact changed")
    training_receipt = formal_result.training_receipt.canonical_payload()
    training_receipt_fp = stable_fingerprint(training_receipt)
    if training_receipt_fp != formal_result.training_receipt.receipt_fingerprint:
        raise RuntimeError("Formal core training receipt changed")
    source_closure_fp = gcr_pacre_v24_source_closure_fingerprint(
        formal_result.source_hashes_after
    )
    run_start_artifact = verify_gcr_pacre_formal_run_start_token(
        authorization,
        formal_result.run_start_token,
    )
    trace_payload = read_canonical_json(
        str(formal_result.training_trace_artifact["path"])
    )
    body = {
        "schema_version": GCR_PACRE_FORMAL_EVIDENCE_SCHEMA,
        "seed": formal_result.seed,
        "evaluation_role": formal_result.role,
        "prerequisites": {
            "dataset_free_receipt_fingerprint": (
                authorization.dataset_free_receipt_fingerprint
            ),
            "D_R_structural_receipt_fingerprint": (
                authorization.d_r_structural_receipt_fingerprint
            ),
            "OOF4_decision_fingerprint": (
                authorization.oof_decision.decision_fingerprint
            ),
            "paired_bounded400_decision_fingerprint": (
                authorization.bounded_decision.decision_fingerprint
            ),
        },
        "access_audit_receipt_fingerprint": (
            authorization.access_audit.receipt_fingerprint
        ),
        "training_receipt": training_receipt,
        "training_receipt_fingerprint": training_receipt_fp,
        "finite_audit": trace_finite_audit(
            trace_payload,
            arm=formal_result.role,
        ),
        "cache_artifact": reverified_cache.payload,
        "run_start_artifact": run_start_artifact,
        "schedule_artifact": schedule,
        "training_trace_artifact": dict(
            formal_result.training_trace_artifact
        ),
        "terminal_artifact": _terminal_protocol_artifact(
            terminal_artifact_receipt
        ),
        "terminal_D_R_evaluation": (
            formal_result.terminal_evaluation.canonical_payload()
        ),
        "terminal_D_R_evaluation_fingerprint": (
            formal_result.terminal_evaluation.evaluation_fingerprint
        ),
        "source_closure": {
            "schema_version": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
            "source_hashes": dict(formal_result.source_hashes_after),
            "source_closure_fingerprint": source_closure_fp,
        },
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {**body, "receipt_fingerprint": stable_fingerprint(body)}


def validate_and_issue_formal_evidence(
    receipt: Mapping[str, object],
    *,
    authorization: GCRPACREFormalAuthorization,
    repository_root: str | Path,
) -> VerifiedFormalTraining:
    """Mechanically load the terminal directory, then issue the Formal token."""

    if type(authorization) is not GCRPACREFormalAuthorization:
        raise TypeError("authorization must be exact Formal authorization")
    authorization.verify_unchanged()
    raw_artifact = receipt.get("terminal_artifact")
    if not isinstance(raw_artifact, Mapping):
        raise TypeError("Formal terminal_artifact must be a mapping")
    raw_path = raw_artifact.get("path")
    if not isinstance(raw_path, str):
        raise TypeError("Formal terminal artifact path must be text")
    loaded = load_and_verify_gcr_pacre_formal_terminal(
        Path(raw_path).parent,
    )
    raw_terminal_evaluation = receipt.get("terminal_D_R_evaluation")
    raw_terminal_evaluation_fp = receipt.get(
        "terminal_D_R_evaluation_fingerprint"
    )
    if (
        loaded.receipt.get("model_file_absolute_path") != raw_path
        or loaded.receipt.get("model_file_size_bytes")
        != raw_artifact.get("size_bytes")
        or loaded.receipt.get("model_file_sha256")
        != raw_artifact.get("file_sha256")
        or loaded.receipt.get("model_fingerprint")
        != raw_artifact.get("model_fingerprint")
        or loaded.receipt.get("terminal_D_R_evaluation_fingerprint")
        != raw_terminal_evaluation_fp
        or not isinstance(raw_terminal_evaluation, Mapping)
        or stable_fingerprint(dict(raw_terminal_evaluation))
        != raw_terminal_evaluation_fp
    ):
        raise PermissionError(
            "Formal outer evidence differs from the strict terminal "
            "artifact receipt"
        )
    token = validate_formal_training_receipt(
        receipt,
        expected_seed=authorization.seed,
        expected_role=authorization.role,
        oof_decision=authorization.oof_decision,
        bounded_decision=authorization.bounded_decision,
        access_audit=authorization.access_audit,
        cache_artifact=authorization.cache_artifact,
        dataset_free_receipt_fingerprint=(
            authorization.dataset_free_receipt_fingerprint
        ),
        d_r_structural_receipt_fingerprint=(
            authorization.d_r_structural_receipt_fingerprint
        ),
        repository_root=repository_root,
    )
    if (
        token.final_model_fingerprint
        != loaded.receipt["model_fingerprint"]
        or token.terminal_artifact_path
        != loaded.receipt["model_file_absolute_path"]
        or token.terminal_artifact_sha256
        != loaded.receipt["model_file_sha256"]
    ):
        raise PermissionError(
            "Formal issued token differs from the loaded terminal model"
        )
    return token


__all__ = [
    "GCR_PACRE_FORMAL_ARTIFACT_RECEIPT_FILE",
    "GCR_PACRE_FORMAL_EVIDENCE_SCHEMA",
    "GCR_PACRE_FORMAL_MODEL_ARTIFACT_SCHEMA",
    "GCR_PACRE_FORMAL_MODEL_FILE",
    "GCR_PACRE_FORMAL_SCHEDULE_POLICY_SCHEMA",
    "LoadedGCRPACREFormalTerminal",
    "build_formal_evidence_receipt",
    "load_and_verify_gcr_pacre_formal_terminal",
    "save_gcr_pacre_formal_schedule_atomic",
    "save_gcr_pacre_formal_terminal_atomic",
    "validate_and_issue_formal_evidence",
]
