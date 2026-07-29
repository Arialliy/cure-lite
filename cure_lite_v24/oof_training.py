"""One-shot paired training core for the v24 source-disjoint OOF gate.

This module has no full-``D_R`` input entry point.  It accepts only two
verifier-issued train-cache readers for one exact fold, plus the persistent
run-start capability published before materialization.  PACRE-VC v23 and
GCR-PACRE v24 are constructed independently from seed 42, compared
byte-for-byte before either optimizer can step, and then trained in lockstep
for exactly ``10 * 40`` unchanged PMOPE updates.

Only final safetensors are published.  A completed-400 capability is issued
after both create-only terminal files have been flushed, fsynced, and
round-trip checked.  Holdout, ``D_V``, and ``D_T`` inputs are neither accepted
nor imported by the execution API.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from math import isfinite
import os
from pathlib import Path
from threading import Lock
from typing import Final, Mapping

import torch

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.coverage_state_device_cache import (
    CoverageStateDeviceCache,
    prepare_coverage_state_device_cache,
)
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateTrainingSchedule,
    coverage_state_schedule_exposure_report,
)
from cure_lite.experiment.coverage_state_training import (
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
    audit_coverage_state_training_state,
    coverage_state_fused_train_step,
    coverage_state_pair_objective_policy,
)
from cure_lite_v23.factory import build_pacre_vc_training_model
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)

from .artifact_io import (
    load_terminal_safetensors_strict,
    regular_file_receipt,
    save_terminal_safetensors_new,
)
from .factory import (
    GCR_PACRE_PARAMETER_NAMES,
    build_gcr_pacre_training_model,
)
from .gcr_pacre import (
    CURELiteGatedCommonResidualPACRELevelSet,
    CoverageStateGCRPACREConfig,
)
from .oof_cache import (
    OOF_EVENT_TRAIN_CACHE_CREATED,
    OOF_EVENT_TRAINING_RUN_START,
    VerifiedOOFCacheReader,
    load_oof_cache_payload,
    require_verified_oof_cache_reader,
)
from .oof_run_start import (
    OOF_EPOCHS,
    OOF_SEED,
    OOF_STEPS_PER_EPOCH,
    OOF_UPDATES,
    VerifiedOOFTrainingRunStart,
    require_verified_oof_training_run_start,
)
from .oof_split import (
    VerifiedOOFFoldClosure,
    require_verified_oof_fold_closure,
)


OOF_PAIRED_TRAINING_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-paired-training-result-v1"
)
OOF_COMPLETED_400_CAPABILITY_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-completed-400-capability-v1"
)
OOF_TERMINAL_ARTIFACT_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-terminal-safetensors-v1"
)
OOF_CONTROL_ARM: Final = "PACRE_VC_v23_control"
OOF_CANDIDATE_ARM: Final = "GCR_PACRE_v24"
OOF_TRAINING_ARMS: Final = (OOF_CONTROL_ARM, OOF_CANDIDATE_ARM)
OOF_OBJECTIVE: Final = CoverageStatePairObjective.PMOPE_JOINT.value
OOF_OPTIMIZER_FQCN: Final = "torch.optim.adam.Adam"
OOF_TERMINAL_FILES: Final = {
    OOF_CONTROL_ARM: "v23_control_terminal.safetensors",
    OOF_CANDIDATE_ARM: "candidate_terminal.safetensors",
}
OOF_READER_IDS: Final = {
    OOF_CONTROL_ARM: "PACRE_VC_v23_control_train_runner",
    OOF_CANDIDATE_ARM: "GCR_PACRE_v24_train_runner",
}
OOF_TRAINING_SOURCE_PATHS: Final = tuple(sorted((
    "cure_lite/coverage_state_device_cache.py",
    "cure_lite/coverage_state_precomputed_cache.py",
    "cure_lite/coverage_state_schedule.py",
    "cure_lite/experiment/coverage_state_training.py",
    "cure_lite/train/coverage_state_fused_step.py",
    "cure_lite_v23/factory.py",
    "cure_lite_v23/pacre_vc.py",
    "cure_lite_v24/artifact_io.py",
    "cure_lite_v24/factory.py",
    "cure_lite_v24/gcr_pacre.py",
    "cure_lite_v24/oof_cache.py",
    "cure_lite_v24/oof_run_start.py",
    "cure_lite_v24/oof_split.py",
    "cure_lite_v24/oof_training.py",
)))

_CAPABILITY_ISSUER = object()
_CAPABILITY_REGISTRY: dict[int, object] = {}
_CLAIM_LOCK = Lock()
_CLAIMED_RUN_STARTS: set[tuple[str, str]] = set()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_oof_terminal_model_strict(
    terminal_artifact: Mapping[str, object],
    *,
    arm: str,
    expected_path: str | Path | None = None,
) -> torch.nn.Module:
    """Reconstruct the exact OOF model from terminal safetensors.

    This is the cross-process mechanical boundary used by both the protocol
    validator and the final evidence replay.  Receipt-declared state
    metadata is compared to the actual safetensors state before a strict
    factory reconstruction and model-fingerprint recomputation.
    """

    if not isinstance(terminal_artifact, Mapping):
        raise TypeError("OOF terminal artifact must be a mapping")
    if arm not in OOF_TRAINING_ARMS:
        raise ValueError("unknown OOF terminal arm")
    path_value = terminal_artifact.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("OOF terminal artifact path is invalid")
    path = Path(path_value)
    if expected_path is not None and path != Path(expected_path):
        raise PermissionError("OOF terminal is not at its fixed runtime path")
    state = load_terminal_safetensors_strict(path)
    expected_keys = list(GCR_PACRE_PARAMETER_NAMES)
    actual_shapes = {
        name: list(tensor.shape) for name, tensor in state.items()
    }
    actual_dtypes = {
        name: str(tensor.dtype) for name, tensor in state.items()
    }
    if (
        set(state) != set(expected_keys)
        or len(state) != len(expected_keys)
        or terminal_artifact.get("state_keys") != expected_keys
        or terminal_artifact.get("state_shapes") != actual_shapes
        or terminal_artifact.get("state_dtypes") != actual_dtypes
        or terminal_artifact.get("parameter_count")
        != sum(tensor.numel() for tensor in state.values())
        or terminal_artifact.get("parameter_count") != 64_064
    ):
        raise RuntimeError("OOF terminal state schema differs from the model")
    if arm == OOF_CONTROL_ARM:
        model = build_pacre_vc_training_model(
            CoverageStatePACREVerifierCorrectedConfig(
                feature_channels=64,
                feature_stride=4,
                width=32,
            )
        )
        if type(model) is not CURELitePACREVerifierCorrectedLevelSet:
            raise AssertionError("OOF v23 factory type changed")
    else:
        model = build_gcr_pacre_training_model(
            CoverageStateGCRPACREConfig(
                feature_channels=64,
                feature_stride=4,
                width=32,
            )
        )
        if type(model) is not CURELiteGatedCommonResidualPACRELevelSet:
            raise AssertionError("OOF v24 factory type changed")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("OOF terminal strict state load was incomplete")
    model_fingerprint = coverage_state_model_fingerprint(model)
    if terminal_artifact.get("model_fingerprint") != model_fingerprint:
        raise RuntimeError("OOF terminal model fingerprint is not factual")
    return model


def _source_hashes() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[1]
    rows: list[tuple[str, str]] = []
    for relative in OOF_TRAINING_SOURCE_PATHS:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
        ):
            raise RuntimeError(f"OOF training source is invalid: {relative}")
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


def _claim_run_start(token: VerifiedOOFTrainingRunStart) -> None:
    key = (token.marker_fingerprint, token.marker_path)
    with _CLAIM_LOCK:
        if key in _CLAIMED_RUN_STARTS:
            raise PermissionError(
                "OOF persistent run-start is already consumed in this process"
            )
        _CLAIMED_RUN_STARTS.add(key)


def _parameter_rows(
    model: torch.nn.Module,
) -> tuple[dict[str, object], ...]:
    rows = tuple(
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "byte_count": parameter.numel() * parameter.element_size(),
            "content_fingerprint": tensor_content_fingerprint(parameter),
        }
        for name, parameter in model.named_parameters()
    )
    if tuple(str(row["name"]) for row in rows) != (
        GCR_PACRE_PARAMETER_NAMES
    ):
        raise RuntimeError("OOF parameter inventory changed")
    return rows


def _require_byte_identical_initialization(
    control: torch.nn.Module,
    candidate: torch.nn.Module,
) -> tuple[tuple[dict[str, object], ...], str]:
    control_rows = _parameter_rows(control)
    candidate_rows = _parameter_rows(candidate)
    if control_rows != candidate_rows:
        raise RuntimeError(
            "OOF initial parameter names/shapes/dtypes/bytes differ"
        )
    for (control_name, control_parameter), (
        candidate_name,
        candidate_parameter,
    ) in zip(
        control.named_parameters(),
        candidate.named_parameters(),
        strict=True,
    ):
        if (
            control_name != candidate_name
            or not torch.equal(
                control_parameter.detach().cpu().contiguous().view(
                    torch.uint8
                ),
                candidate_parameter.detach().cpu().contiguous().view(
                    torch.uint8
                ),
            )
        ):
            raise RuntimeError("OOF initial parameter raw bytes differ")
    control_fp = coverage_state_model_fingerprint(control)
    candidate_fp = coverage_state_model_fingerprint(candidate)
    if control_fp != candidate_fp:
        raise RuntimeError("OOF initial model states differ")
    return control_rows, stable_fingerprint(list(control_rows))


def _module_storage_addresses(
    model: torch.nn.Module,
) -> set[tuple[str, int, int]]:
    return {
        (
            str(value.device),
            int(value.untyped_storage().data_ptr()),
            int(value.untyped_storage().nbytes()),
        )
        for value in model.parameters()
    }


def _runtime_instance_fingerprint(
    value: object,
    *,
    kind: str,
) -> str:
    return stable_fingerprint(
        {
            "schema_version": (
                "cure-lite-v24-oof-process-local-object-identity-v1"
            ),
            "kind": kind,
            "process_id": os.getpid(),
            "python_object_id": id(value),
            "fqcn": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    )


def _parameter_storage_rows(
    model: torch.nn.Module,
) -> tuple[dict[str, object], ...]:
    rows = tuple(
        {
            "name": name,
            "device": str(parameter.device),
            "nbytes": int(parameter.untyped_storage().nbytes()),
            "storage_identity_fingerprint": stable_fingerprint(
                {
                    "schema_version": (
                        "cure-lite-v24-oof-process-local-storage-identity-v1"
                    ),
                    "process_id": os.getpid(),
                    "device": str(parameter.device),
                    "data_ptr": int(
                        parameter.untyped_storage().data_ptr()
                    ),
                    "nbytes": int(
                        parameter.untyped_storage().nbytes()
                    ),
                }
            ),
        }
        for name, parameter in model.named_parameters()
    )
    if (
        tuple(str(row["name"]) for row in rows)
        != GCR_PACRE_PARAMETER_NAMES
        or len({
            str(row["storage_identity_fingerprint"]) for row in rows
        })
        != len(rows)
    ):
        raise RuntimeError("OOF parameter storage identity is not disjoint")
    return rows


def _cache_storage_addresses(
    cache: CoverageStateScalarCache,
) -> set[tuple[str, int, int]]:
    addresses: set[tuple[str, int, int]] = set()
    seen: set[int] = set()

    def walk(value: object) -> None:
        identity = id(value)
        if identity in seen:
            return
        if isinstance(value, torch.Tensor):
            seen.add(identity)
            addresses.add((
                str(value.device),
                int(value.untyped_storage().data_ptr()),
                int(value.untyped_storage().nbytes()),
            ))
            return
        if hasattr(value, "__dataclass_fields__") and not isinstance(
            value,
            type,
        ):
            seen.add(identity)
            for name in value.__dataclass_fields__:
                walk(getattr(value, name))
            return
        if isinstance(value, Mapping):
            seen.add(identity)
            for key, item in value.items():
                walk(key)
                walk(item)
            return
        if isinstance(value, (tuple, list)):
            seen.add(identity)
            for item in value:
                walk(item)

    walk(cache)
    if not addresses:
        raise RuntimeError("OOF scalar cache contains no tensor storage")
    return addresses


def _device_cache_storage_addresses(
    cache: CoverageStateDeviceCache,
) -> set[tuple[str, int, int]]:
    return {
        (
            str(tensor.device),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        )
        for _, tensor in cache.named_tensors()
    }


def _cache_sample_ids(
    cache: CoverageStateScalarCache,
) -> set[str]:
    return {
        value.record.sample_id for value in cache.natural_records
    } | {
        value.record.sample_id for value in cache.pair_records
    }


def _resolved_device(value: torch.device | str) -> torch.device:
    try:
        result = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise TypeError("device must identify one torch device") from error
    if result.type == "cuda" and result.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        result = torch.device("cuda", torch.cuda.current_device())
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("OOF training supports only CPU or CUDA")
    return result


@contextmanager
def _deterministic_execution(device: torch.device):
    old_algorithms = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        yield
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        torch.use_deterministic_algorithms(
            old_algorithms,
            warn_only=old_warn_only,
        )
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def _fresh_seeded_model(
    factory,
    config: object,
) -> torch.nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(OOF_SEED)
        return factory(config)


def _new_terminal_path(
    value: str | Path | None,
    *,
    output_directory: Path,
    arm: str,
) -> Path:
    expected = (
        output_directory / "terminal" / OOF_TERMINAL_FILES[arm]
    )
    requested = expected if value is None else Path(value)
    if not requested.is_absolute():
        requested = (Path.cwd() / requested).absolute()
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    if (
        target != expected
        or target.exists()
        or target.is_symlink()
        or parent.is_symlink()
    ):
        raise FileExistsError(
            "OOF terminal output must be the exact new frozen path"
        )
    return target


def _reader_cache(
    reader: VerifiedOOFCacheReader,
    *,
    closure: VerifiedOOFFoldClosure,
    arm: str,
    schedule: CoverageStateTrainingSchedule,
) -> CoverageStateScalarCache:
    token = require_verified_oof_cache_reader(reader)
    if (
        token.fold_id != closure.fold_id
        or token.partition != "train"
        or token.arm != arm
        or token.closure_fingerprint != closure.closure_fingerprint
        or token.reader_id != OOF_READER_IDS[arm]
        or token.authorization_event
        != OOF_EVENT_TRAIN_CACHE_CREATED
        or token.holdout_terminal_seal_fingerprint is not None
        or token.semantic_payload_fingerprint
        != schedule.cache_fingerprint
    ):
        raise PermissionError("OOF training reader binding changed")
    payload = load_oof_cache_payload(token)
    if type(payload) is not CoverageStateScalarCache:
        raise TypeError(
            "OOF training requires an exact train-only "
            "CoverageStateScalarCache payload"
        )
    payload.verify_unchanged()
    if (
        payload.cache_fingerprint != schedule.cache_fingerprint
        or _cache_sample_ids(payload) != set(closure.train_sample_ids)
    ):
        raise PermissionError("OOF scalar cache crosses its train closure")
    return payload


def _training_result(
    *,
    model: torch.nn.Module,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    device_cache: CoverageStateDeviceCache,
    optimizer_fingerprint: str,
    initial_model_fingerprint: str,
    epoch_logs: list[dict[str, object]],
    first_nonzero: Mapping[str, int],
    counters: Mapping[str, int],
    device: torch.device,
) -> CoverageStateTrainingResult:
    if set(first_nonzero) != set(GCR_PACRE_PARAMETER_NAMES):
        raise RuntimeError("OOF model has an incomplete PMOPE gradient path")
    if dict(counters) != {
        "forward_calls": OOF_UPDATES,
        "backward_calls": OOF_UPDATES,
        "optimizer_steps": OOF_UPDATES,
        "logical_state_evaluations": OOF_UPDATES * 12,
        "finite_state_audits": OOF_UPDATES + 1,
    }:
        raise RuntimeError("OOF per-arm compute ledger is not exact 400")
    return CoverageStateTrainingResult(
        objective=OOF_OBJECTIVE,
        objective_policy=coverage_state_pair_objective_policy(
            CoverageStatePairObjective.PMOPE_JOINT
        ),
        seed=OOF_SEED,
        epochs=OOF_EPOCHS,
        steps_per_epoch=OOF_STEPS_PER_EPOCH,
        completed_updates=OOF_UPDATES,
        schedule_fingerprint=schedule.schedule_fingerprint,
        cache_fingerprint=cache.cache_fingerprint,
        execution_device=str(device),
        device_cache_fingerprint=device_cache.device_cache_fingerprint,
        device_cache_resident_bytes=device_cache.resident_tensor_bytes,
        optimizer_config_fingerprint=optimizer_fingerprint,
        initial_model_fingerprint=initial_model_fingerprint,
        final_model_fingerprint=coverage_state_model_fingerprint(model),
        epoch_logs=tuple(epoch_logs),
        first_nonzero_gradient_update=tuple(sorted(first_nonzero.items())),
        forward_calls=counters["forward_calls"],
        backward_calls=counters["backward_calls"],
        optimizer_steps=counters["optimizer_steps"],
        logical_state_evaluations=counters[
            "logical_state_evaluations"
        ],
        finite_state_audits=counters["finite_state_audits"],
    )


def _save_terminal(
    path: Path,
    *,
    arm: str,
    fold_id: int,
    model: torch.nn.Module,
    training_result: CoverageStateTrainingResult,
    run_start: VerifiedOOFTrainingRunStart,
) -> dict[str, object]:
    final_fingerprint = coverage_state_model_fingerprint(model)
    if (
        training_result.completed_updates != OOF_UPDATES
        or training_result.final_model_fingerprint != final_fingerprint
    ):
        raise RuntimeError("OOF terminal is not an exact completed-400 model")
    saved = save_terminal_safetensors_new(
        path,
        model,
        metadata={
            "schema": OOF_TERMINAL_ARTIFACT_SCHEMA,
            "run": f"oof4-fold-{fold_id}",
            "seed": str(OOF_SEED),
            "role": arm,
            "arm": arm,
            "model_fingerprint": final_fingerprint,
            "epochs": str(OOF_EPOCHS),
            "steps_per_epoch": str(OOF_STEPS_PER_EPOCH),
            "updates": str(OOF_UPDATES),
            "checkpoint_policy": "final_only",
            "run_start_marker_fingerprint": (
                run_start.marker_fingerprint
            ),
        },
    )
    loaded = load_terminal_safetensors_strict(path)
    expected_state = {
        name: tensor.detach().to("cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    if (
        set(loaded) != set(expected_state)
        or any(
            not torch.equal(loaded[name], expected_state[name])
            for name in loaded
        )
    ):
        raise RuntimeError("OOF terminal safetensors roundtrip changed")
    body = {
        "schema_version": OOF_TERMINAL_ARTIFACT_SCHEMA,
        "fold_id": fold_id,
        "arm": arm,
        "seed": OOF_SEED,
        "epochs": OOF_EPOCHS,
        "steps_per_epoch": OOF_STEPS_PER_EPOCH,
        "completed_updates": OOF_UPDATES,
        "path": saved["path"],
        "size_bytes": saved["size_bytes"],
        "file_sha256": saved["file_sha256"],
        "device": saved["device"],
        "inode": saved["inode"],
        "hardlink_count": saved["hardlink_count"],
        "state_keys": saved["state_keys"],
        "state_shapes": saved["state_shapes"],
        "state_dtypes": saved["state_dtypes"],
        "parameter_count": saved["parameter_count"],
        "model_fingerprint": final_fingerprint,
        "training_result_fingerprint": (
            training_result.result_fingerprint
        ),
        "run_start_marker_fingerprint": (
            run_start.marker_fingerprint
        ),
        "serialization": "safetensors",
        "final_checkpoint_only": True,
        "optimizer_state_saved": False,
        "intermediate_checkpoint_saved": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {
        **body,
        "terminal_artifact_fingerprint": stable_fingerprint(body),
    }


@dataclass(frozen=True, slots=True)
class VerifiedOOFCompleted400Capability:
    """Strong exact-instance proof of one sealed 400-update train arm."""

    payload_json: str
    capability_fingerprint: str
    fold_id: int
    arm: str
    closure_fingerprint: str
    run_start_marker_fingerprint: str
    run_start_marker_path: str
    run_start_marker_file_sha256: str
    cache_artifact_fingerprint: str
    semantic_cache_fingerprint: str
    schedule_fingerprint: str
    batch_sequence_fingerprint: str
    shared_initial_parameter_fingerprint: str
    initial_parameters_json: str
    module_instance_id: str
    optimizer_instance_id: str
    parameter_storage_ledger_json: str
    parameter_storage_ledger_fingerprint: str
    optimizer_config_fingerprint: str
    objective_policy_fingerprint: str
    completed_updates: int
    terminal_artifact_fingerprint: str
    terminal_artifact_path: str
    terminal_artifact_file_sha256: str
    terminal_model_fingerprint: str
    training_result_fingerprint: str
    source_fingerprint: str
    _run_start_token: VerifiedOOFTrainingRunStart = field(repr=False)
    _model: torch.nn.Module = field(repr=False)
    _optimizer: torch.optim.Adam = field(repr=False)
    _issuer: object = field(repr=False)

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.payload_json)
        if (
            not isinstance(value, dict)
            or canonical_json(value) != self.payload_json
        ):
            raise RuntimeError("completed-400 capability payload changed")
        return value

    @property
    def initial_parameters(self) -> tuple[dict[str, object], ...]:
        value = json.loads(self.initial_parameters_json)
        if not isinstance(value, list):
            raise RuntimeError("completed-400 initial ledger changed")
        return tuple(dict(row) for row in value)

    @property
    def parameter_storage_ledger(
        self,
    ) -> tuple[dict[str, object], ...]:
        value = json.loads(self.parameter_storage_ledger_json)
        if not isinstance(value, list):
            raise RuntimeError("completed-400 storage ledger changed")
        return tuple(dict(row) for row in value)


def _register_capability(
    value: VerifiedOOFCompleted400Capability,
) -> VerifiedOOFCompleted400Capability:
    if value._issuer is not _CAPABILITY_ISSUER:
        raise AssertionError("attempted to register unsigned completed-400")
    identity = id(value)
    prior = _CAPABILITY_REGISTRY.get(identity)
    if prior is not None and prior is not value:
        raise RuntimeError("completed-400 capability identity collision")
    _CAPABILITY_REGISTRY[identity] = value
    return value


def _terminal_unchanged(
    capability: VerifiedOOFCompleted400Capability,
) -> None:
    receipt = regular_file_receipt(capability.terminal_artifact_path)
    if (
        receipt["file_sha256"]
        != capability.terminal_artifact_file_sha256
        or receipt["path"] != capability.terminal_artifact_path
        or receipt["hardlink_count"] != 1
    ):
        raise RuntimeError("OOF completed-400 terminal artifact changed")
    state = load_terminal_safetensors_strict(
        capability.terminal_artifact_path
    )
    ledger = capability.payload.get("terminal_artifact")
    if (
        not isinstance(ledger, dict)
        or ledger.get("file_sha256")
        != capability.terminal_artifact_file_sha256
        or ledger.get("terminal_artifact_fingerprint")
        != capability.terminal_artifact_fingerprint
        or not isinstance(ledger.get("state_keys"), list)
        or set(ledger["state_keys"]) != set(state)
        or len(ledger["state_keys"]) != len(state)
        or ledger.get("state_shapes")
        != {name: list(tensor.shape) for name, tensor in state.items()}
        or ledger.get("state_dtypes")
        != {name: str(tensor.dtype) for name, tensor in state.items()}
    ):
        raise RuntimeError("OOF completed-400 terminal ledger changed")


def require_verified_oof_completed_400_capability(
    value: object,
    *,
    fold_closure: VerifiedOOFFoldClosure | None = None,
    arm: str | None = None,
) -> VerifiedOOFCompleted400Capability:
    if (
        type(value) is not VerifiedOOFCompleted400Capability
        or value._issuer is not _CAPABILITY_ISSUER
        or _CAPABILITY_REGISTRY.get(id(value)) is not value
    ):
        raise TypeError(
            "completed_400_capability must be issued by the paired OOF runner"
        )
    capability = value
    payload = capability.payload
    body = dict(payload)
    fingerprint = body.pop("capability_fingerprint", None)
    current_sources = _source_hashes()
    initial_rows = payload.get("initial_parameters")
    model_config = payload.get("model_config")
    storage_rows = _parameter_storage_rows(capability._model)
    optimizer_parameters = tuple(
        parameter
        for group in capability._optimizer.param_groups
        for parameter in group["params"]
    )
    model_parameters = tuple(capability._model.parameters())
    optimizer_parameters_differ = (
        len(optimizer_parameters) != len(model_parameters)
        or any(
            optimizer_parameter is not model_parameter
            for optimizer_parameter, model_parameter in zip(
                optimizer_parameters,
                model_parameters,
                strict=True,
            )
        )
    )
    model_type_wrong = (
        capability.arm == OOF_CONTROL_ARM
        and type(capability._model)
        is not CURELitePACREVerifierCorrectedLevelSet
    ) or (
        capability.arm == OOF_CANDIDATE_ARM
        and type(capability._model)
        is not CURELiteGatedCommonResidualPACRELevelSet
    )
    if (
        fingerprint != capability.capability_fingerprint
        or stable_fingerprint(body) != capability.capability_fingerprint
        or payload.get("completed_updates") != OOF_UPDATES
        or capability.completed_updates != OOF_UPDATES
        or payload.get("source_hashes") != dict(current_sources)
        or capability.source_fingerprint
        != stable_fingerprint(dict(current_sources))
        or not isinstance(initial_rows, list)
        or canonical_json(initial_rows)
        != capability.initial_parameters_json
        or stable_fingerprint(initial_rows)
        != capability.shared_initial_parameter_fingerprint
        or not isinstance(model_config, dict)
        or set(model_config)
        != {
            "feature_channels",
            "feature_stride",
            "width",
            "parameter_count",
        }
        or capability.module_instance_id
        != _runtime_instance_fingerprint(
            capability._model,
            kind="module",
        )
        or capability.optimizer_instance_id
        != _runtime_instance_fingerprint(
            capability._optimizer,
            kind="optimizer",
        )
        or canonical_json(list(storage_rows))
        != capability.parameter_storage_ledger_json
        or stable_fingerprint(list(storage_rows))
        != capability.parameter_storage_ledger_fingerprint
        or payload.get("module_instance_id")
        != capability.module_instance_id
        or payload.get("optimizer_instance_id")
        != capability.optimizer_instance_id
        or payload.get("parameter_storage_ledger")
        != list(storage_rows)
        or payload.get("parameter_storage_ledger_fingerprint")
        != capability.parameter_storage_ledger_fingerprint
        or type(capability._optimizer) is not torch.optim.Adam
        or optimizer_parameters_differ
        or model_type_wrong
        or coverage_state_optimizer_config_fingerprint(
            capability._model,
            capability._optimizer,
        )
        != capability.optimizer_config_fingerprint
        or coverage_state_model_fingerprint(capability._model)
        != capability.terminal_model_fingerprint
    ):
        raise RuntimeError("completed-400 capability binding changed")
    if capability.arm == OOF_CONTROL_ARM:
        normative_config = CoverageStatePACREVerifierCorrectedConfig(
            feature_channels=int(model_config["feature_channels"]),
            feature_stride=int(model_config["feature_stride"]),
            width=int(model_config["width"]),
        )
        normative_model = _fresh_seeded_model(
            build_pacre_vc_training_model,
            normative_config,
        )
    elif capability.arm == OOF_CANDIDATE_ARM:
        normative_config = CoverageStateGCRPACREConfig(
            feature_channels=int(model_config["feature_channels"]),
            feature_stride=int(model_config["feature_stride"]),
            width=int(model_config["width"]),
        )
        normative_model = _fresh_seeded_model(
            build_gcr_pacre_training_model,
            normative_config,
        )
    else:
        raise PermissionError("completed-400 capability has an unknown arm")
    normative_rows = _parameter_rows(normative_model)
    if (
        list(normative_rows) != initial_rows
        or sum(int(row["numel"]) for row in normative_rows)
        != model_config["parameter_count"]
    ):
        raise RuntimeError(
            "completed-400 initial ledger differs from the seed42 "
            "fresh-factory normative ledger"
        )
    run_start = require_verified_oof_training_run_start(
        capability._run_start_token
    )
    if (
        run_start.marker_fingerprint
        != capability.run_start_marker_fingerprint
        or run_start.marker_path != capability.run_start_marker_path
        or run_start.marker_file_sha256
        != capability.run_start_marker_file_sha256
        or run_start.fold_id != capability.fold_id
        or run_start.closure_fingerprint
        != capability.closure_fingerprint
    ):
        raise PermissionError("completed-400 run-start binding changed")
    if arm is not None and capability.arm != arm:
        raise PermissionError("completed-400 capability belongs to another arm")
    if fold_closure is not None:
        closure = require_verified_oof_fold_closure(fold_closure)
        if (
            closure.fold_id != capability.fold_id
            or closure.closure_fingerprint
            != capability.closure_fingerprint
        ):
            raise PermissionError(
                "completed-400 capability belongs to another fold"
            )
    _terminal_unchanged(capability)
    return capability


def _issue_capability(
    *,
    closure: VerifiedOOFFoldClosure,
    run_start: VerifiedOOFTrainingRunStart,
    reader: VerifiedOOFCacheReader,
    arm: str,
    schedule: CoverageStateTrainingSchedule,
    shared_initial_parameter_fingerprint: str,
    initial_parameters: tuple[dict[str, object], ...],
    model_config: object,
    optimizer_config_fingerprint: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    training_result: CoverageStateTrainingResult,
    terminal_artifact: Mapping[str, object],
    source_hashes: tuple[tuple[str, str], ...],
) -> VerifiedOOFCompleted400Capability:
    objective_policy_fp = stable_fingerprint(
        coverage_state_pair_objective_policy(
            CoverageStatePairObjective.PMOPE_JOINT
        )
    )
    batch_sequence_fp = stable_fingerprint(
        [
            selection.selection_fingerprint
            for selection in schedule.selections
        ]
    )
    module_instance_id = _runtime_instance_fingerprint(
        model,
        kind="module",
    )
    optimizer_instance_id = _runtime_instance_fingerprint(
        optimizer,
        kind="optimizer",
    )
    storage_rows = _parameter_storage_rows(model)
    storage_ledger_fp = stable_fingerprint(list(storage_rows))
    body = {
        "schema_version": OOF_COMPLETED_400_CAPABILITY_SCHEMA,
        "fold_id": closure.fold_id,
        "arm": arm,
        "closure_fingerprint": closure.closure_fingerprint,
        "run_start": {
            "marker_fingerprint": run_start.marker_fingerprint,
            "marker_path": run_start.marker_path,
            "marker_file_sha256": run_start.marker_file_sha256,
            "authorization_fingerprint": (
                run_start.authorization_fingerprint
            ),
            "source_closure_fingerprint": (
                run_start.source_closure_fingerprint
            ),
        },
        "train_cache": {
            "artifact_fingerprint": (
                reader.cache_artifact_fingerprint
            ),
            "semantic_payload_fingerprint": (
                reader.semantic_payload_fingerprint
            ),
            "reader_authorization_fingerprint": (
                reader.authorization_fingerprint
            ),
        },
        "seed": OOF_SEED,
        "epochs": OOF_EPOCHS,
        "steps_per_epoch": OOF_STEPS_PER_EPOCH,
        "completed_updates": OOF_UPDATES,
        "training_invocations": 1,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "batch_sequence_fingerprint": batch_sequence_fp,
        "shared_initial_parameter_fingerprint": (
            shared_initial_parameter_fingerprint
        ),
        "initial_parameters": list(initial_parameters),
        "model_config": {
            "feature_channels": model_config.feature_channels,
            "feature_stride": model_config.feature_stride,
            "width": model_config.width,
            "parameter_count": model_config.expected_parameter_count,
        },
        "module_instance_id": module_instance_id,
        "optimizer_instance_id": optimizer_instance_id,
        "parameter_storage_ledger": list(storage_rows),
        "parameter_storage_ledger_fingerprint": storage_ledger_fp,
        "optimizer_fqcn": OOF_OPTIMIZER_FQCN,
        "optimizer_config_fingerprint": optimizer_config_fingerprint,
        "objective": OOF_OBJECTIVE,
        "objective_policy_fingerprint": objective_policy_fp,
        "training_result_fingerprint": training_result.result_fingerprint,
        "terminal_artifact": dict(terminal_artifact),
        "source_hashes": dict(source_hashes),
        "from_scratch": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "checkpoint_policy": "final_only",
        "holdout_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    fingerprint = stable_fingerprint(body)
    payload = {**body, "capability_fingerprint": fingerprint}
    return _register_capability(VerifiedOOFCompleted400Capability(
        payload_json=canonical_json(payload),
        capability_fingerprint=fingerprint,
        fold_id=closure.fold_id,
        arm=arm,
        closure_fingerprint=closure.closure_fingerprint,
        run_start_marker_fingerprint=run_start.marker_fingerprint,
        run_start_marker_path=run_start.marker_path,
        run_start_marker_file_sha256=run_start.marker_file_sha256,
        cache_artifact_fingerprint=reader.cache_artifact_fingerprint,
        semantic_cache_fingerprint=(
            reader.semantic_payload_fingerprint
        ),
        schedule_fingerprint=schedule.schedule_fingerprint,
        batch_sequence_fingerprint=batch_sequence_fp,
        shared_initial_parameter_fingerprint=(
            shared_initial_parameter_fingerprint
        ),
        initial_parameters_json=canonical_json(list(initial_parameters)),
        module_instance_id=module_instance_id,
        optimizer_instance_id=optimizer_instance_id,
        parameter_storage_ledger_json=canonical_json(
            list(storage_rows)
        ),
        parameter_storage_ledger_fingerprint=storage_ledger_fp,
        optimizer_config_fingerprint=optimizer_config_fingerprint,
        objective_policy_fingerprint=objective_policy_fp,
        completed_updates=OOF_UPDATES,
        terminal_artifact_fingerprint=str(
            terminal_artifact["terminal_artifact_fingerprint"]
        ),
        terminal_artifact_path=str(terminal_artifact["path"]),
        terminal_artifact_file_sha256=str(
            terminal_artifact["file_sha256"]
        ),
        terminal_model_fingerprint=(
            training_result.final_model_fingerprint
        ),
        training_result_fingerprint=training_result.result_fingerprint,
        source_fingerprint=stable_fingerprint(dict(source_hashes)),
        _run_start_token=run_start,
        _model=model,
        _optimizer=optimizer,
        _issuer=_CAPABILITY_ISSUER,
    ))


@dataclass(frozen=True, eq=False)
class OOFPairedTrainingResult:
    fold_id: int
    closure_fingerprint: str
    run_start_marker_fingerprint: str
    control_model: CURELitePACREVerifierCorrectedLevelSet
    candidate_model: CURELiteGatedCommonResidualPACRELevelSet
    control_training_result: CoverageStateTrainingResult
    candidate_training_result: CoverageStateTrainingResult
    shared_initial_parameters: tuple[dict[str, object], ...]
    shared_initial_parameter_fingerprint: str
    control_terminal_artifact: Mapping[str, object]
    candidate_terminal_artifact: Mapping[str, object]
    control_capability: VerifiedOOFCompleted400Capability
    candidate_capability: VerifiedOOFCompleted400Capability
    paired_update_fingerprint: str
    result_fingerprint: str

    @property
    def completed_400_capabilities(
        self,
    ) -> dict[str, VerifiedOOFCompleted400Capability]:
        return {
            OOF_CONTROL_ARM: self.control_capability,
            OOF_CANDIDATE_ARM: self.candidate_capability,
        }

    def verify_unchanged(self) -> None:
        control = require_verified_oof_completed_400_capability(
            self.control_capability,
            arm=OOF_CONTROL_ARM,
        )
        candidate = require_verified_oof_completed_400_capability(
            self.candidate_capability,
            arm=OOF_CANDIDATE_ARM,
        )
        if (
            self.fold_id != control.fold_id
            or self.fold_id != candidate.fold_id
            or self.closure_fingerprint
            != control.closure_fingerprint
            or self.closure_fingerprint
            != candidate.closure_fingerprint
            or self.run_start_marker_fingerprint
            != control.run_start_marker_fingerprint
            or self.run_start_marker_fingerprint
            != candidate.run_start_marker_fingerprint
            or control.module_instance_id
            == candidate.module_instance_id
            or control.optimizer_instance_id
            == candidate.optimizer_instance_id
            or {
                str(row["storage_identity_fingerprint"])
                for row in control.parameter_storage_ledger
            }
            & {
                str(row["storage_identity_fingerprint"])
                for row in candidate.parameter_storage_ledger
            }
            or coverage_state_model_fingerprint(self.control_model)
            != self.control_training_result.final_model_fingerprint
            or coverage_state_model_fingerprint(self.candidate_model)
            != self.candidate_training_result.final_model_fingerprint
            or self.control_training_result.completed_updates != OOF_UPDATES
            or self.candidate_training_result.completed_updates
            != OOF_UPDATES
        ):
            raise RuntimeError("OOF paired training result changed")
        body = {
            "schema_version": OOF_PAIRED_TRAINING_SCHEMA,
            "fold_id": self.fold_id,
            "closure_fingerprint": self.closure_fingerprint,
            "run_start_marker_fingerprint": (
                self.run_start_marker_fingerprint
            ),
            "shared_initial_parameters": list(
                self.shared_initial_parameters
            ),
            "shared_initial_parameter_fingerprint": (
                self.shared_initial_parameter_fingerprint
            ),
            "control_capability_fingerprint": (
                control.capability_fingerprint
            ),
            "candidate_capability_fingerprint": (
                candidate.capability_fingerprint
            ),
            "paired_update_fingerprint": self.paired_update_fingerprint,
            "fixed_relative_promotion_threshold": None,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        if stable_fingerprint(body) != self.result_fingerprint:
            raise RuntimeError("OOF paired result fingerprint changed")


def run_paired_oof_training_400(
    *,
    fold_closure: VerifiedOOFFoldClosure,
    run_start_token: VerifiedOOFTrainingRunStart,
    control_cache_reader: VerifiedOOFCacheReader,
    candidate_cache_reader: VerifiedOOFCacheReader,
    schedule: CoverageStateTrainingSchedule,
    candidate_config: CoverageStateGCRPACREConfig,
    control_output_path: str | Path | None = None,
    candidate_output_path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> OOFPairedTrainingResult:
    """Execute one exact paired fold attempt and publish final-only models."""

    closure = require_verified_oof_fold_closure(fold_closure)
    run_start = require_verified_oof_training_run_start(run_start_token)
    if type(schedule) is not CoverageStateTrainingSchedule:
        raise TypeError("schedule must be exact CoverageStateTrainingSchedule")
    if type(candidate_config) is not CoverageStateGCRPACREConfig:
        raise TypeError("candidate_config must be exact v24 config")
    batch_sequence_fp = stable_fingerprint(
        [
            selection.selection_fingerprint
            for selection in schedule.selections
        ]
    )
    if (
        run_start.fold_id != closure.fold_id
        or run_start.closure_fingerprint != closure.closure_fingerprint
        or run_start.event_index != OOF_EVENT_TRAINING_RUN_START
        or run_start.payload.get("event_index")
        != OOF_EVENT_TRAINING_RUN_START
        or (
            schedule.config.seed,
            schedule.config.epochs,
            schedule.config.steps_per_epoch,
            schedule.config.updates,
        )
        != (
            OOF_SEED,
            OOF_EPOCHS,
            OOF_STEPS_PER_EPOCH,
            OOF_UPDATES,
        )
        or run_start.schedule_fingerprint
        != schedule.schedule_fingerprint
        or run_start.training_population_fingerprint
        != schedule.cache_fingerprint
        or run_start.payload.get("batch_sequence_fingerprint")
        != batch_sequence_fp
        or run_start.payload.get("updates_per_arm") != OOF_UPDATES
    ):
        raise PermissionError("OOF run-start/schedule/closure binding changed")
    _claim_run_start(run_start)
    output_directory = Path(run_start.output_directory).resolve(strict=True)
    control_terminal_path = _new_terminal_path(
        control_output_path,
        output_directory=output_directory,
        arm=OOF_CONTROL_ARM,
    )
    candidate_terminal_path = _new_terminal_path(
        candidate_output_path,
        output_directory=output_directory,
        arm=OOF_CANDIDATE_ARM,
    )
    control_reader = require_verified_oof_cache_reader(
        control_cache_reader
    )
    candidate_reader = require_verified_oof_cache_reader(
        candidate_cache_reader
    )
    if (
        run_start.payload.get("control_cache_artifact_fingerprint")
        != control_reader.cache_artifact_fingerprint
        or run_start.payload.get("candidate_cache_artifact_fingerprint")
        != candidate_reader.cache_artifact_fingerprint
    ):
        raise PermissionError("OOF run-start does not bind these cache readers")
    control_cache = _reader_cache(
        control_reader,
        closure=closure,
        arm=OOF_CONTROL_ARM,
        schedule=schedule,
    )
    candidate_cache = _reader_cache(
        candidate_reader,
        closure=closure,
        arm=OOF_CANDIDATE_ARM,
        schedule=schedule,
    )
    if (
        control_cache is candidate_cache
        or control_cache.cache_fingerprint
        != candidate_cache.cache_fingerprint
        or _cache_storage_addresses(control_cache)
        & _cache_storage_addresses(candidate_cache)
    ):
        raise PermissionError("OOF train scalar caches are not independent")
    coverage_state_schedule_exposure_report(control_cache, schedule)
    coverage_state_schedule_exposure_report(candidate_cache, schedule)
    resolved = _resolved_device(device)
    source_hashes_before = _source_hashes()
    control_config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=candidate_config.feature_channels,
        feature_stride=candidate_config.feature_stride,
        width=candidate_config.width,
    )

    with _deterministic_execution(resolved):
        control_model = _fresh_seeded_model(
            build_pacre_vc_training_model,
            control_config,
        )
        candidate_model = _fresh_seeded_model(
            build_gcr_pacre_training_model,
            candidate_config,
        )
        if (
            type(control_model)
            is not CURELitePACREVerifierCorrectedLevelSet
            or type(candidate_model)
            is not CURELiteGatedCommonResidualPACRELevelSet
        ):
            raise AssertionError("OOF model factory returned a wrong arm")
        initial_rows, shared_initial_fp = (
            _require_byte_identical_initialization(
                control_model,
                candidate_model,
            )
        )
        initial_model_fp = coverage_state_model_fingerprint(control_model)
        control_model = control_model.to(
            device=resolved,
            dtype=torch.float32,
        )
        candidate_model = candidate_model.to(
            device=resolved,
            dtype=torch.float32,
        )
        if (
            coverage_state_model_fingerprint(control_model)
            != initial_model_fp
            or coverage_state_model_fingerprint(candidate_model)
            != initial_model_fp
            or _module_storage_addresses(control_model)
            & _module_storage_addresses(candidate_model)
        ):
            raise RuntimeError("OOF model storage is shared or changed")

        control_optimizer = torch.optim.Adam(
            control_model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
        )
        candidate_optimizer = torch.optim.Adam(
            candidate_model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
        )
        if (
            type(control_optimizer) is not torch.optim.Adam
            or type(candidate_optimizer) is not torch.optim.Adam
            or control_optimizer is candidate_optimizer
            or control_optimizer.state
            or candidate_optimizer.state
        ):
            raise RuntimeError("OOF arms require two fresh independent Adam")
        control_optimizer_fp = (
            coverage_state_optimizer_config_fingerprint(
                control_model,
                control_optimizer,
            )
        )
        candidate_optimizer_fp = (
            coverage_state_optimizer_config_fingerprint(
                candidate_model,
                candidate_optimizer,
            )
        )
        if control_optimizer_fp != candidate_optimizer_fp:
            raise RuntimeError("OOF Adam policies differ")

        control_device_cache = prepare_coverage_state_device_cache(
            control_cache,
            device=resolved,
        )
        candidate_device_cache = prepare_coverage_state_device_cache(
            candidate_cache,
            device=resolved,
        )
        if (
            control_device_cache is candidate_device_cache
            or _device_cache_storage_addresses(control_device_cache)
            & _device_cache_storage_addresses(candidate_device_cache)
        ):
            raise PermissionError("OOF packed device caches share storage")
        control_device_cache.verify_unchanged()
        candidate_device_cache.verify_unchanged()
        audit_coverage_state_training_state(
            control_model,
            control_optimizer,
        )
        audit_coverage_state_training_state(
            candidate_model,
            candidate_optimizer,
        )

        first_nonzero: dict[str, dict[str, int]] = {
            OOF_CONTROL_ARM: {},
            OOF_CANDIDATE_ARM: {},
        }
        epoch_logs: dict[str, list[dict[str, object]]] = {
            OOF_CONTROL_ARM: [],
            OOF_CANDIDATE_ARM: [],
        }
        counters: dict[str, dict[str, int]] = {
            arm: {
                "forward_calls": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "logical_state_evaluations": 0,
                "finite_state_audits": 1,
            }
            for arm in OOF_TRAINING_ARMS
        }
        epoch_sums = {
            arm: {
                "factual_miss/loss": 0.0,
                "factual_no_miss/loss": 0.0,
                "pair/loss": 0.0,
                "total": 0.0,
                "gradient_l2_norm": 0.0,
            }
            for arm in OOF_TRAINING_ARMS
        }
        paired_updates: list[dict[str, object]] = []
        epoch_selections: list[str] = []
        for update, selection in enumerate(schedule.selections):
            control_batch = control_device_cache.materialize(
                selection,
                verify=False,
                validate=False,
            )
            candidate_batch = candidate_device_cache.materialize(
                selection,
                verify=False,
                validate=False,
            )
            if (
                control_batch.selection_fingerprint
                != candidate_batch.selection_fingerprint
            ):
                raise RuntimeError("OOF paired batch sequence changed")
            logs: dict[str, Mapping[str, object]] = {}
            for arm, model, optimizer, batch, cache in (
                (
                    OOF_CONTROL_ARM,
                    control_model,
                    control_optimizer,
                    control_batch,
                    control_cache,
                ),
                (
                    OOF_CANDIDATE_ARM,
                    candidate_model,
                    candidate_optimizer,
                    candidate_batch,
                    candidate_cache,
                ),
            ):
                log = coverage_state_fused_train_step(
                    model,
                    optimizer,
                    batch,
                    config=cache.sobolev_config,
                    pair_objective=(
                        CoverageStatePairObjective.PMOPE_JOINT
                    ),
                    audit=False,
                    track_nonzero_gradients=(
                        len(first_nonzero[arm])
                        < len(GCR_PACRE_PARAMETER_NAMES)
                    ),
                )
                if log.get("selection_fingerprint") != (
                    batch.selection_fingerprint
                ):
                    raise RuntimeError("OOF PMOPE step used another batch")
                for metric in epoch_sums[arm]:
                    epoch_sums[arm][metric] += _finite(
                        log[metric],
                        name=f"{arm}.{metric}",
                    )
                for name in filter(
                    None,
                    str(log["nonzero_gradient_parameters"]).split(","),
                ):
                    first_nonzero[arm].setdefault(name, update)
                counters[arm]["forward_calls"] += int(
                    log["model_forward_calls"]
                )
                counters[arm]["backward_calls"] += int(
                    log["backward_calls"]
                )
                counters[arm]["optimizer_steps"] += int(
                    log["optimizer_steps"]
                )
                counters[arm]["logical_state_evaluations"] += int(
                    log["logical_states"]
                )
                counters[arm]["finite_state_audits"] += int(
                    log["post_step_finite_audits"]
                )
                logs[arm] = log
            paired_updates.append({
                "update": update,
                "schedule_selection_fingerprint": (
                    selection.selection_fingerprint
                ),
                "materialized_batch_fingerprint": (
                    control_batch.selection_fingerprint
                ),
                "control_total": _finite(
                    logs[OOF_CONTROL_ARM]["total"],
                    name="control total",
                ),
                "candidate_total": _finite(
                    logs[OOF_CANDIDATE_ARM]["total"],
                    name="candidate total",
                ),
            })
            epoch_selections.append(selection.selection_fingerprint)
            if (update + 1) % OOF_STEPS_PER_EPOCH == 0:
                epoch = update // OOF_STEPS_PER_EPOCH
                selection_fp = stable_fingerprint(epoch_selections)
                for arm in OOF_TRAINING_ARMS:
                    epoch_logs[arm].append({
                        "epoch": epoch,
                        "completed_updates": update + 1,
                        "objective": OOF_OBJECTIVE,
                        "selection_sequence_fingerprint": selection_fp,
                        **{
                            f"mean_{name}": (
                                value / OOF_STEPS_PER_EPOCH
                            )
                            for name, value in epoch_sums[arm].items()
                        },
                    })
                    epoch_sums[arm] = {
                        name: 0.0 for name in epoch_sums[arm]
                    }
                epoch_selections = []

        audit_coverage_state_training_state(
            control_model,
            control_optimizer,
        )
        audit_coverage_state_training_state(
            candidate_model,
            candidate_optimizer,
        )
        control_device_cache.verify_unchanged()
        candidate_device_cache.verify_unchanged()
        control_training = _training_result(
            model=control_model,
            cache=control_cache,
            schedule=schedule,
            device_cache=control_device_cache,
            optimizer_fingerprint=control_optimizer_fp,
            initial_model_fingerprint=initial_model_fp,
            epoch_logs=epoch_logs[OOF_CONTROL_ARM],
            first_nonzero=first_nonzero[OOF_CONTROL_ARM],
            counters=counters[OOF_CONTROL_ARM],
            device=resolved,
        )
        candidate_training = _training_result(
            model=candidate_model,
            cache=candidate_cache,
            schedule=schedule,
            device_cache=candidate_device_cache,
            optimizer_fingerprint=candidate_optimizer_fp,
            initial_model_fingerprint=initial_model_fp,
            epoch_logs=epoch_logs[OOF_CANDIDATE_ARM],
            first_nonzero=first_nonzero[OOF_CANDIDATE_ARM],
            counters=counters[OOF_CANDIDATE_ARM],
            device=resolved,
        )

    if _source_hashes() != source_hashes_before:
        raise RuntimeError("OOF training source bytes changed during execution")
    control_artifact = _save_terminal(
        control_terminal_path,
        arm=OOF_CONTROL_ARM,
        fold_id=closure.fold_id,
        model=control_model,
        training_result=control_training,
        run_start=run_start,
    )
    candidate_artifact = _save_terminal(
        candidate_terminal_path,
        arm=OOF_CANDIDATE_ARM,
        fold_id=closure.fold_id,
        model=candidate_model,
        training_result=candidate_training,
        run_start=run_start,
    )
    if (
        control_artifact["device"],
        control_artifact["inode"],
    ) == (
        candidate_artifact["device"],
        candidate_artifact["inode"],
    ):
        raise PermissionError("OOF terminal arms share one physical file")
    if _source_hashes() != source_hashes_before:
        raise RuntimeError("OOF training source bytes changed before sealing")
    control_capability = _issue_capability(
        closure=closure,
        run_start=run_start,
        reader=control_reader,
        arm=OOF_CONTROL_ARM,
        schedule=schedule,
        shared_initial_parameter_fingerprint=shared_initial_fp,
        initial_parameters=initial_rows,
        model_config=control_config,
        optimizer_config_fingerprint=control_optimizer_fp,
        model=control_model,
        optimizer=control_optimizer,
        training_result=control_training,
        terminal_artifact=control_artifact,
        source_hashes=source_hashes_before,
    )
    candidate_capability = _issue_capability(
        closure=closure,
        run_start=run_start,
        reader=candidate_reader,
        arm=OOF_CANDIDATE_ARM,
        schedule=schedule,
        shared_initial_parameter_fingerprint=shared_initial_fp,
        initial_parameters=initial_rows,
        model_config=candidate_config,
        optimizer_config_fingerprint=candidate_optimizer_fp,
        model=candidate_model,
        optimizer=candidate_optimizer,
        training_result=candidate_training,
        terminal_artifact=candidate_artifact,
        source_hashes=source_hashes_before,
    )
    paired_update_fp = stable_fingerprint(paired_updates)
    result_body = {
        "schema_version": OOF_PAIRED_TRAINING_SCHEMA,
        "fold_id": closure.fold_id,
        "closure_fingerprint": closure.closure_fingerprint,
        "run_start_marker_fingerprint": run_start.marker_fingerprint,
        "shared_initial_parameters": list(initial_rows),
        "shared_initial_parameter_fingerprint": shared_initial_fp,
        "control_capability_fingerprint": (
            control_capability.capability_fingerprint
        ),
        "candidate_capability_fingerprint": (
            candidate_capability.capability_fingerprint
        ),
        "paired_update_fingerprint": paired_update_fp,
        "fixed_relative_promotion_threshold": None,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    result = OOFPairedTrainingResult(
        fold_id=closure.fold_id,
        closure_fingerprint=closure.closure_fingerprint,
        run_start_marker_fingerprint=run_start.marker_fingerprint,
        control_model=control_model,
        candidate_model=candidate_model,
        control_training_result=control_training,
        candidate_training_result=candidate_training,
        shared_initial_parameters=initial_rows,
        shared_initial_parameter_fingerprint=shared_initial_fp,
        control_terminal_artifact=control_artifact,
        candidate_terminal_artifact=candidate_artifact,
        control_capability=control_capability,
        candidate_capability=candidate_capability,
        paired_update_fingerprint=paired_update_fp,
        result_fingerprint=stable_fingerprint(result_body),
    )
    result.verify_unchanged()
    return result


__all__ = [
    "OOF_CANDIDATE_ARM",
    "OOF_COMPLETED_400_CAPABILITY_SCHEMA",
    "OOF_CONTROL_ARM",
    "OOF_OBJECTIVE",
    "OOF_PAIRED_TRAINING_SCHEMA",
    "OOF_TERMINAL_ARTIFACT_SCHEMA",
    "OOF_TRAINING_ARMS",
    "OOFPairedTrainingResult",
    "VerifiedOOFCompleted400Capability",
    "load_oof_terminal_model_strict",
    "require_verified_oof_completed_400_capability",
    "run_paired_oof_training_400",
]
