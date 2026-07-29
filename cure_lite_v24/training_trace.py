"""Create-only, hash-chained training traces for protected v24 stages.

The trace is an auditable accidental-tamper and execution-consistency record.
Without an external signer or trusted hardware it cannot prove that a
same-user malicious producer actually executed every claimed update; the
mechanical verifier therefore makes no stronger claim.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Final, Mapping, Sequence

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    CoverageStateTrainingSchedule,
    build_coverage_state_training_schedule,
)

from .artifact_io import (
    atomic_write_new_canonical_json,
    read_canonical_json,
)
from .formal_cache_artifacts import (
    VerifiedFormalCacheArtifact,
    load_formal_scalar_cache_artifact,
)


GCR_PACRE_TRAINING_TRACE_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-step-finite-hash-chain-v1"
)
GCR_PACRE_TRAINING_TRACE_ARTIFACT_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-step-trace-artifact-v1"
)
GCR_PACRE_TRACE_FINITE_AUDIT_SCHEMA: Final = (
    "cure-lite-v24-training-finite-trace-audit-v2"
)
GCR_PACRE_TRACE_LIMITATION: Final = (
    "accidental_tamper_and_internal_consistency_evidence_not_"
    "cryptographic_proof_against_same_user_malicious_fabrication_"
    "without_external_signature_or_trusted_hardware"
)

_ARM_STEP_FIELDS: Final = {
    "loss",
    "gradient_l2_norm",
    "optimizer_step_counter",
    "parameter_state_digest",
    "optimizer_state_digest",
    "loss_finite",
    "gradients_finite",
    "parameters_finite",
    "optimizer_state_finite",
}


def _sha(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise FloatingPointError(f"{name} must be finite")
    return result


def mechanically_rebuild_schedule_artifact(
    *,
    schedule_artifact_path: str,
    cache_artifact: VerifiedFormalCacheArtifact,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    expected_schedule_fingerprint: str,
) -> CoverageStateTrainingSchedule:
    """Read canonical schedule bytes and rebuild every selection from cache."""

    cache = load_formal_scalar_cache_artifact(cache_artifact)
    rebuilt = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=seed,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        ),
    )
    payload = read_canonical_json(schedule_artifact_path)
    expected_fp = _sha(
        expected_schedule_fingerprint,
        name="expected_schedule_fingerprint",
    )
    if (
        payload != rebuilt.canonical_payload()
        or stable_fingerprint(payload) != expected_fp
        or rebuilt.schedule_fingerprint != expected_fp
    ):
        raise PermissionError(
            "canonical schedule bytes differ from deterministic "
            "full-cache reconstruction"
        )
    return rebuilt


def _normalize_arm_step(
    value: object,
    *,
    expected_step: int,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ARM_STEP_FIELDS:
        raise ValueError(f"{name} fields changed")
    loss = _finite(value.get("loss"), name=f"{name}.loss")
    gradient = _finite(
        value.get("gradient_l2_norm"),
        name=f"{name}.gradient_l2_norm",
    )
    counter = value.get("optimizer_step_counter")
    if (
        loss < 0.0
        or gradient < 0.0
        or isinstance(counter, bool)
        or not isinstance(counter, int)
        or counter != expected_step
        or any(
            value.get(flag) is not True
            for flag in (
                "loss_finite",
                "gradients_finite",
                "parameters_finite",
                "optimizer_state_finite",
            )
        )
    ):
        raise ValueError(f"{name} finite/optimizer-step evidence changed")
    return {
        "loss": loss,
        "gradient_l2_norm": gradient,
        "optimizer_step_counter": counter,
        "parameter_state_digest": _sha(
            value.get("parameter_state_digest"),
            name=f"{name}.parameter_state_digest",
        ),
        "optimizer_state_digest": _sha(
            value.get("optimizer_state_digest"),
            name=f"{name}.optimizer_state_digest",
        ),
        "loss_finite": True,
        "gradients_finite": True,
        "parameters_finite": True,
        "optimizer_state_finite": True,
    }


def build_training_trace_payload(
    *,
    stage_id: str,
    authorization_fingerprint: str,
    schedule: CoverageStateTrainingSchedule,
    arm_names: Sequence[str],
    terminal_model_fingerprints: Mapping[str, str],
    raw_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Seal exact ordered step rows into a one-way canonical hash chain."""

    if not isinstance(stage_id, str) or not stage_id:
        raise ValueError("trace stage_id must be non-empty")
    authorization_fp = _sha(
        authorization_fingerprint,
        name="authorization_fingerprint",
    )
    arms = tuple(arm_names)
    if (
        not arms
        or len(set(arms)) != len(arms)
        or any(not isinstance(arm, str) or not arm for arm in arms)
        or set(terminal_model_fingerprints) != set(arms)
        or len(raw_rows) != schedule.config.updates
    ):
        raise ValueError("training trace arm/update inventory changed")
    terminal_fps = {
        arm: _sha(
            terminal_model_fingerprints[arm],
            name=f"terminal_model_fingerprints[{arm}]",
        )
        for arm in arms
    }
    batch_sequence_fp = stable_fingerprint(
        [
            selection.canonical_payload()
            for selection in schedule.selections
        ]
    )
    header = {
        "schema_version": GCR_PACRE_TRAINING_TRACE_SCHEMA,
        "stage_id": stage_id,
        "authorization_fingerprint": authorization_fp,
        "seed": schedule.config.seed,
        "epochs": schedule.config.epochs,
        "steps_per_epoch": schedule.config.steps_per_epoch,
        "updates": schedule.config.updates,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "batch_sequence_fingerprint": batch_sequence_fp,
        "arm_names": list(arms),
        "terminal_model_fingerprints": terminal_fps,
        "evidence_limitation": GCR_PACRE_TRACE_LIMITATION,
    }
    previous = stable_fingerprint(
        {"trace_genesis": header}
    )
    rows: list[dict[str, object]] = []
    for update, (raw, selection) in enumerate(
        zip(raw_rows, schedule.selections, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise TypeError(f"training trace row {update} is not a mapping")
        expected_fields = {
            "update",
            "epoch",
            "step",
            "selection_fingerprint",
            "arms",
        }
        if set(raw) != expected_fields:
            raise ValueError(f"training trace row {update} fields changed")
        raw_arms = raw.get("arms")
        if (
            raw.get("update") != update
            or raw.get("epoch") != selection.epoch
            or raw.get("step") != selection.step
            or raw.get("selection_fingerprint")
            != selection.selection_fingerprint
            or not isinstance(raw_arms, Mapping)
            or set(raw_arms) != set(arms)
        ):
            raise ValueError(f"training trace row {update} identity changed")
        normalized_arms = {
            arm: _normalize_arm_step(
                raw_arms[arm],
                expected_step=update + 1,
                name=f"trace[{update}].arms[{arm}]",
            )
            for arm in arms
        }
        body = {
            "update": update,
            "epoch": selection.epoch,
            "step": selection.step,
            "selection_fingerprint": selection.selection_fingerprint,
            "arms": normalized_arms,
            "previous_row_fingerprint": previous,
        }
        row_fp = stable_fingerprint(body)
        rows.append({**body, "row_fingerprint": row_fp})
        previous = row_fp
    body = {
        **header,
        "genesis_fingerprint": stable_fingerprint(
            {"trace_genesis": header}
        ),
        "rows": rows,
        "final_row_fingerprint": previous,
    }
    return {**body, "trace_fingerprint": stable_fingerprint(body)}


def save_training_trace_new(
    path: str | Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Persist one immutable canonical trace and return its file receipt."""

    target = atomic_write_new_canonical_json(path, payload)
    stat_result = target.stat()
    trace_fp = _sha(
        payload.get("trace_fingerprint"),
        name="trace_fingerprint",
    )
    final_fp = _sha(
        payload.get("final_row_fingerprint"),
        name="final_row_fingerprint",
    )
    return {
        "schema_version": GCR_PACRE_TRAINING_TRACE_ARTIFACT_SCHEMA,
        "path": str(target.resolve(strict=True)),
        "size_bytes": stat_result.st_size,
        "file_sha256": file_sha256(target),
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "hardlink_count": stat_result.st_nlink,
        "trace_fingerprint": trace_fp,
        "final_row_fingerprint": final_fp,
    }


def verify_training_trace_artifact(
    artifact: object,
    *,
    expected_path: str | Path,
    stage_id: str,
    authorization_fingerprint: str,
    schedule: CoverageStateTrainingSchedule,
    arm_names: Sequence[str],
    terminal_model_fingerprints: Mapping[str, str],
) -> dict[str, object]:
    """Re-read, normalize, and verify every persisted trace row."""

    if not isinstance(artifact, Mapping):
        raise TypeError("training trace artifact must be a mapping")
    expected_artifact_fields = {
        "schema_version",
        "path",
        "size_bytes",
        "file_sha256",
        "device",
        "inode",
        "hardlink_count",
        "trace_fingerprint",
        "final_row_fingerprint",
    }
    path = Path(str(artifact.get("path")))
    required = Path(expected_path)
    if (
        set(artifact) != expected_artifact_fields
        or artifact.get("schema_version")
        != GCR_PACRE_TRAINING_TRACE_ARTIFACT_SCHEMA
        or not required.is_absolute()
        or path != required
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or path.stat().st_nlink != 1
        or path.stat().st_mode & 0o222
        or path.stat().st_size != artifact.get("size_bytes")
        or path.stat().st_dev != artifact.get("device")
        or path.stat().st_ino != artifact.get("inode")
        or path.stat().st_nlink != artifact.get("hardlink_count")
        or file_sha256(path) != artifact.get("file_sha256")
    ):
        raise PermissionError("training trace artifact bytes/path changed")
    payload = read_canonical_json(path)
    if payload.get("schema_version") != GCR_PACRE_TRAINING_TRACE_SCHEMA:
        raise ValueError("training trace schema changed")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("training trace rows must be a list")
    raw_rows: list[dict[str, object]] = []
    previous = payload.get("genesis_fingerprint")
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"training trace row {index} is invalid")
        body = dict(raw)
        row_fp = body.pop("row_fingerprint", None)
        if (
            body.get("previous_row_fingerprint") != previous
            or stable_fingerprint(body) != row_fp
        ):
            raise PermissionError(
                f"training trace hash chain broke at row {index}"
            )
        raw_rows.append(
            {
                key: body[key]
                for key in (
                    "update",
                    "epoch",
                    "step",
                    "selection_fingerprint",
                    "arms",
                )
            }
        )
        previous = row_fp
    rebuilt = build_training_trace_payload(
        stage_id=stage_id,
        authorization_fingerprint=authorization_fingerprint,
        schedule=schedule,
        arm_names=arm_names,
        terminal_model_fingerprints=terminal_model_fingerprints,
        raw_rows=raw_rows,
    )
    if (
        payload != rebuilt
        or payload.get("trace_fingerprint")
        != artifact.get("trace_fingerprint")
        or payload.get("final_row_fingerprint")
        != artifact.get("final_row_fingerprint")
    ):
        raise PermissionError(
            "training trace differs from schedule-bound reconstruction"
        )
    return payload


def trace_finite_audit(
    trace_payload: Mapping[str, object],
    *,
    arm: str,
    parameter_count: int = 3,
) -> dict[str, object]:
    """Derive, rather than assert, one arm's finite audit from trace rows."""

    rows = trace_payload.get("rows")
    if (
        not isinstance(rows, list)
        or not rows
        or isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count < 1
    ):
        raise ValueError("finite trace audit input changed")
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"finite trace row {index} changed")
        arms = raw.get("arms")
        if not isinstance(arms, Mapping) or arm not in arms:
            raise ValueError(f"finite trace lacks arm {arm}")
        _normalize_arm_step(
            arms[arm],
            expected_step=index + 1,
            name=f"finite_trace[{index}].arms[{arm}]",
        )
    body = {
        "schema_version": GCR_PACRE_TRACE_FINITE_AUDIT_SCHEMA,
        "trace_fingerprint": trace_payload["trace_fingerprint"],
        "arm": arm,
        "expected_updates": len(rows),
        "trace_rows_checked": len(rows),
        "loss_values_checked": len(rows),
        "gradient_steps_checked": len(rows),
        "parameter_state_steps_checked": len(rows),
        "optimizer_state_steps_checked": len(rows),
        "gradient_tensors_checked": len(rows) * parameter_count,
        "parameter_tensors_checked": len(rows) * parameter_count,
        "nonfinite_values": 0,
        "evidence_limitation": GCR_PACRE_TRACE_LIMITATION,
    }
    return {**body, "audit_fingerprint": stable_fingerprint(body)}


__all__ = [
    "GCR_PACRE_TRACE_FINITE_AUDIT_SCHEMA",
    "GCR_PACRE_TRACE_LIMITATION",
    "GCR_PACRE_TRAINING_TRACE_ARTIFACT_SCHEMA",
    "GCR_PACRE_TRAINING_TRACE_SCHEMA",
    "build_training_trace_payload",
    "mechanically_rebuild_schedule_artifact",
    "save_training_trace_new",
    "trace_finite_audit",
    "verify_training_trace_artifact",
]
