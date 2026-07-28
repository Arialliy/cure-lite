#!/usr/bin/env python3
"""Validate or consume the unique CURE-Lite v22 PACRE bounded-400 run.

``--validate-create-only`` checks only static/generated prerequisites.  It
does not claim output, load real ``D_R`` tensors, construct an optimizer, or
enter the real-data gate.  ``--run-once`` is fixed to visible GPU 0, seed 42,
10 epochs by 40 updates, one PMOPE candidate, and one output directory.

There is no output, seed, budget, retry, resume, calibration, ``D_V``, or
``D_T`` option.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_sobolev import CSLF_PMOPE_POLICY
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_EPOCHS,
    COVERAGE_STATE_BOUNDED_SEED,
    COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH,
    COVERAGE_STATE_BOUNDED_UPDATES,
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite.frozen_base import module_state_fingerprint
from cure_lite_v22.bounded_runner import (
    PACRE_BOUNDED_ATTEMPT_FIELDS,
    PACRE_BOUNDED_ATTEMPT_RUNTIME_FIELDS,
    PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG,
    PACRE_BOUNDED_DEVICE,
    PACRE_BOUNDED_OUTPUT_REPO_PATH,
    PACRE_BOUNDED_PAUSE_TEMPERATURE_C,
    PACRE_BOUNDED_RESUME_TEMPERATURE_C,
    PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256,
    PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH,
    PACRE_BOUNDED_VISIBLE_GPU,
    _implementation_binding as _bounded_implementation_binding,
    load_pacre_bounded_output_claim,
    pacre_bounded_process_identity,
    prepare_pacre_bounded_run_authorization,
    run_pacre_pmope_bounded_400,
)
from cure_lite_v22.dataset_free import (
    PACRE_FORMAL_FEATURE_CHANNELS,
    PACRE_FORMAL_FEATURE_STRIDE,
    PACRE_FORMAL_PARAMETER_COUNT,
    PACRE_FORMAL_WIDTH,
    run_pacre_dataset_free_gate,
)
from cure_lite_v22.decision import PACRE_BOUNDED_RUN_ID
from cure_lite_v22.dr_gate import (
    PACRE_DR_FAIL_DECISION,
    PACRE_DR_PASS_DECISION,
    run_pacre_dr_gate,
)
from cure_lite_v22.pacre import (
    CSLF_PACRE_CENTERING_POLICY,
    CSLF_PACRE_EQUATION_POLICY,
    CSLF_PACRE_FIELD_POLICY,
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
)
from cure_lite_v22.training import PACRE_PMOPE_OBJECTIVE


_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = PACRE_BOUNDED_RUN_ID
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = _ROOT / OUTPUT_REPO_PATH
if OUTPUT_REPO_PATH != PACRE_BOUNDED_OUTPUT_REPO_PATH:
    raise RuntimeError("PACRE bounded output path binding changed")

FROZEN_DEVICE = PACRE_BOUNDED_DEVICE
FROZEN_VISIBLE_GPU = PACRE_BOUNDED_VISIBLE_GPU
FROZEN_CUBLAS_WORKSPACE_CONFIG = (
    PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG
)
FROZEN_PAUSE_TEMPERATURE_C = PACRE_BOUNDED_PAUSE_TEMPERATURE_C
FROZEN_RESUME_TEMPERATURE_C = PACRE_BOUNDED_RESUME_TEMPERATURE_C
FROZEN_SEED = COVERAGE_STATE_BOUNDED_SEED
FROZEN_EPOCHS = COVERAGE_STATE_BOUNDED_EPOCHS
FROZEN_STEPS_PER_EPOCH = COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
FROZEN_UPDATES = COVERAGE_STATE_BOUNDED_UPDATES
FROZEN_FEATURE_CHANNELS = PACRE_FORMAL_FEATURE_CHANNELS
FROZEN_FEATURE_STRIDE = PACRE_FORMAL_FEATURE_STRIDE
FROZEN_MODEL_WIDTH = PACRE_FORMAL_WIDTH
FROZEN_PARAMETER_COUNT = PACRE_FORMAL_PARAMETER_COUNT
FROZEN_CHECKPOINT_SERIALIZATION = "safetensors"

TEMPERATURE_WRAPPER_REPO_PATH = (
    PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH
)
TEMPERATURE_WRAPPER_FILE_SHA256 = (
    PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256
)
CLI_REPO_PATH = "tools/run_cure_lite_v22_pacre_bounded_400.py"

FROZEN_REAL_DR_INPUTS = (
    (
        "manifest_path",
        "protocols/IRSTD-1K/stage_a_seed42/manifest.json",
        "aa8e33529bd86f564ce6e163e0f9a7b1b3053e9c15054a59c6702a1523f35c02",
    ),
    (
        "state_index_path",
        (
            "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3/"
            "d_r/state_cache/index.json"
        ),
        "075fc1ad217f365df85b1d29568ad215f06ce6e0b691ef78a5dd85f0affe6298",
    ),
    (
        "geometry_config_path",
        "protocols/IRSTD-1K/geometry_safe_p0_v2/config.json",
        "719e956b7c51b2b2c8294699fe26c2d36d5c8190b0d8bb5c1d5665a0f4344558",
    ),
    (
        "geometry_receipt_path",
        (
            "runs/irstd1k_stage_a_seed42/"
            "cure_lite_geometry_safe_p0_v2_r1/"
            "receipts/geometry_catalog.json"
        ),
        "e2a9a986f8819433f3f5efd5c4f627504d10fb32d20f62769b2235b803209283",
    ),
    (
        "observability_config_path",
        (
            "protocols/IRSTD-1K/"
            "coverage_state_observability_v1/config.json"
        ),
        "60d42e657f1daed3cb01c7ee93c8f3fe17417542931d853756ccbbeda1f95713",
    ),
)

RUN_SCHEMA = "cure-lite-v22-pacre-pmope-bounded-400-run-v1"
VALIDATION_SCHEMA = (
    "cure-lite-v22-pacre-pmope-bounded-400-create-only-validation-v1"
)
ATTEMPT_SCHEMA = (
    "cure-lite-v22-pacre-pmope-bounded-400-attempt-v1"
)
AUTHORIZATION_WRAPPER_SCHEMA = (
    "cure-lite-v22-pacre-pmope-bounded-400-authorization-wrapper-v1"
)
RESULT_WRAPPER_SCHEMA = (
    "cure-lite-v22-pacre-pmope-bounded-400-result-wrapper-v1"
)
DATASET_FREE_WRAPPER_SCHEMA = (
    "cure-lite-v22-pacre-bounded-400-dataset-free-v1"
)
INPUTS_SCHEMA = "cure-lite-v22-pacre-bounded-400-inputs-v1"
PREFLIGHT_SCHEMA = "cure-lite-v22-pacre-bounded-400-preflight-v1"
DR_GATE_WRAPPER_SCHEMA = (
    "cure-lite-v22-pacre-bounded-400-real-D_R-gate-v1"
)
TRAINING_WRAPPER_SCHEMA = (
    "cure-lite-v22-pacre-bounded-400-training-v1"
)
ZERO_LEVEL_WRAPPER_SCHEMA = (
    "cure-lite-v22-pacre-bounded-400-zero-level-v1"
)
FAILURE_SCHEMA = (
    "cure-lite-v22-pacre-pmope-bounded-400-failure-v1"
)
CHECKPOINT_SCHEMA = (
    "cure-lite-v22-pacre-pmope-bounded-400-checkpoint-v1"
)
DECISION_SCHEMA = (
    "cure-lite-v22-pacre-pmope-bounded-400-terminal-decision-v1"
)
FROZEN_DR_GATE_STOP_ARTIFACT_FILE_COUNT = 7
FROZEN_TERMINAL_ARTIFACT_FILE_COUNT = 13
_INCOMPLETE = ".incomplete"
_CHECKPOINT_FILE = f"checkpoints/{PACRE_PMOPE_OBJECTIVE}.safetensors"
_CHECKPOINT_RECEIPT_FILE = (
    f"checkpoints/{PACRE_PMOPE_OBJECTIVE}.checkpoint.json"
)
_COMMON_TERMINAL_FILES = frozenset(
    {
        "attempt.json",
        "receipts/config.json",
        "receipts/dataset_free.json",
        "receipts/inputs.json",
        "receipts/preflight.json",
        "receipts/dr_gate.json",
        "receipts/decision.json",
    }
)
_PASS_TERMINAL_FILES = frozenset(
    {
        *_COMMON_TERMINAL_FILES,
        "receipts/authorization.json",
        "receipts/training.json",
        "receipts/zero_level.json",
        "receipts/bounded_result.json",
        _CHECKPOINT_FILE,
        _CHECKPOINT_RECEIPT_FILE,
    }
)
_TERMINAL_JSON_CONTRACTS = {
    "receipts/config.json": (RUN_SCHEMA, "receipt_fingerprint"),
    "receipts/dataset_free.json": (
        DATASET_FREE_WRAPPER_SCHEMA,
        "wrapper_fingerprint",
    ),
    "receipts/inputs.json": (INPUTS_SCHEMA, "receipt_fingerprint"),
    "receipts/preflight.json": (
        PREFLIGHT_SCHEMA,
        "receipt_fingerprint",
    ),
    "receipts/dr_gate.json": (
        DR_GATE_WRAPPER_SCHEMA,
        "wrapper_fingerprint",
    ),
    "receipts/decision.json": (
        DECISION_SCHEMA,
        "receipt_fingerprint",
    ),
    "receipts/authorization.json": (
        AUTHORIZATION_WRAPPER_SCHEMA,
        "receipt_fingerprint",
    ),
    "receipts/training.json": (
        TRAINING_WRAPPER_SCHEMA,
        "receipt_fingerprint",
    ),
    "receipts/zero_level.json": (
        ZERO_LEVEL_WRAPPER_SCHEMA,
        "receipt_fingerprint",
    ),
    "receipts/bounded_result.json": (
        RESULT_WRAPPER_SCHEMA,
        "receipt_fingerprint",
    ),
    _CHECKPOINT_RECEIPT_FILE: (
        CHECKPOINT_SCHEMA,
        "receipt_fingerprint",
    ),
}
_ATTEMPT_FINGERPRINT_FIELD = "receipt_fingerprint"
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")
_ATTEMPT_FIELDS = PACRE_BOUNDED_ATTEMPT_FIELDS
_ATTEMPT_RUNTIME_FIELDS = PACRE_BOUNDED_ATTEMPT_RUNTIME_FIELDS
_VERIFIED_RUNTIME_FIELDS = frozenset(
    {
        *_ATTEMPT_RUNTIME_FIELDS,
        "visible_device_count",
        "visible_device_index",
        "device_name",
        "device_total_memory_bytes",
        "cuda_runtime_verified_after_output_claim",
    }
)


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    result = dict(payload)
    if field in result:
        raise ValueError(f"payload already contains {field}")
    result[field] = stable_fingerprint(result)
    return result


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_HEX_DIGITS
    )


def _validate_attempt_receipt(
    attempt: Mapping[str, object],
) -> str:
    """Validate the exact immutable attempt schema and self fingerprint."""

    if not isinstance(attempt, Mapping):
        raise TypeError("PACRE attempt must be a mapping")
    payload = dict(attempt)
    if frozenset(payload) != _ATTEMPT_FIELDS:
        raise ValueError("PACRE attempt fields differ from the fixed schema")
    fingerprint = payload.pop(_ATTEMPT_FINGERPRINT_FIELD)
    runtime = payload.get("runtime")
    budget = payload.get("budget")
    process_identity = payload.get("process_identity")
    if (
        payload.get("schema_version") != ATTEMPT_SCHEMA
        or payload.get("run_id") != RUN_ID
        or payload.get("output_repo_path") != OUTPUT_REPO_PATH
        or not _is_sha256(payload.get("config_fingerprint"))
        or payload.get("candidate") != "PACRE-v22"
        or payload.get("objective") != PACRE_PMOPE_OBJECTIVE
        or process_identity != pacre_bounded_process_identity()
        or budget
        != {
            "seed": FROZEN_SEED,
            "epochs": FROZEN_EPOCHS,
            "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
            "updates": FROZEN_UPDATES,
        }
        or not _is_sha256(
            payload.get("dataset_free_receipt_fingerprint")
        )
        or payload.get("dataset_free_invocations_before_claim") != 1
        or payload.get("single_attempt") is not True
        or payload.get("resume_allowed") is not False
        or payload.get("automatic_retry_allowed") is not False
        or payload.get("formal_800_authorized") is not False
        or payload.get("D_V_accessed") is not False
        or payload.get("D_T_accessed") is not False
        or not isinstance(runtime, Mapping)
        or frozenset(runtime) != _ATTEMPT_RUNTIME_FIELDS
        or runtime.get("device") != FROZEN_DEVICE
        or runtime.get("CUDA_VISIBLE_DEVICES")
        != FROZEN_VISIBLE_GPU
        or runtime.get("CUBLAS_WORKSPACE_CONFIG")
        != FROZEN_CUBLAS_WORKSPACE_CONFIG
        or runtime.get("temperature_wrapper_repo_path")
        != TEMPERATURE_WRAPPER_REPO_PATH
        or runtime.get("temperature_wrapper_file_sha256")
        != TEMPERATURE_WRAPPER_FILE_SHA256
        or runtime.get("pause_temperature_c")
        != FROZEN_PAUSE_TEMPERATURE_C
        or runtime.get("resume_temperature_c")
        != FROZEN_RESUME_TEMPERATURE_C
        or not _is_sha256(fingerprint)
        or stable_fingerprint(payload) != fingerprint
    ):
        raise ValueError("PACRE attempt receipt is invalid")
    return str(fingerprint)


def _read_canonical_json_object(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} is not canonical JSON") from error
    if (
        not isinstance(payload, dict)
        or _json_bytes(payload) != raw
    ):
        raise ValueError(f"{path.name} is not a canonical JSON object")
    return payload


def _validate_self_fingerprinted_receipt(
    payload: Mapping[str, object],
    *,
    schema: str,
    field: str = "receipt_fingerprint",
    name: str,
) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping")
    body = dict(payload)
    fingerprint = body.pop(field, None)
    if (
        body.get("schema_version") != schema
        or body.get("run_id") != RUN_ID
        or not _is_sha256(fingerprint)
        or stable_fingerprint(body) != fingerprint
    ):
        raise ValueError(f"{name} receipt is invalid")
    return str(fingerprint)


def _write_new_bytes(path: Path, payload: bytes) -> None:
    """Create one file exclusively, flush it, and never replace a path."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    _write_new_bytes(path, _json_bytes(payload))


def _write_run_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    expected_artifacts: dict[str, bytes],
) -> None:
    """Write and retain the exact intended bytes for terminal verification."""

    raw = _json_bytes(payload)
    relative = str(path.relative_to(OUTPUT_PATH))
    if relative in expected_artifacts:
        raise RuntimeError(f"PACRE artifact was registered twice: {relative}")
    _write_new_bytes(path, raw)
    expected_artifacts[relative] = raw


def _claim_output(
    output: Path,
    *,
    attempt: Mapping[str, object],
) -> tuple[Path, Path]:
    """Atomically claim the sole directory and establish its marker."""

    _validate_attempt_receipt(attempt)
    output.mkdir(parents=True, exist_ok=False)
    (output / _INCOMPLETE).open("xb").close()
    _write_new_json(output / "attempt.json", attempt)
    receipts = output / "receipts"
    checkpoints = output / "checkpoints"
    receipts.mkdir(exist_ok=False)
    checkpoints.mkdir(exist_ok=False)
    return receipts, checkpoints


def _artifact_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("PACRE artifact may not be a symbolic link")
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if relative in {_INCOMPLETE, "COMPLETE.json"}:
            continue
        result[relative] = file_sha256(path)
    return result


def _canonical_repo_file(relative: str, *, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name} repository path is invalid")
    path = _ROOT / relative
    absolute = Path(os.path.abspath(path))
    if path.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = path.resolve(strict=True)
    if (
        resolved != absolute
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise ValueError(f"{name} must be a canonical regular file")
    return resolved


def _verify_frozen_sources() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for argument, relative, expected_sha256 in FROZEN_REAL_DR_INPUTS:
        path = _canonical_repo_file(relative, name=argument)
        if file_sha256(path) != expected_sha256:
            raise RuntimeError(
                f"frozen real D_R input changed: {argument}"
            )
        result[argument] = path
    return result


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    result = dict(_bounded_implementation_binding())
    for relative in (CLI_REPO_PATH, TEMPERATURE_WRAPPER_REPO_PATH):
        path = _canonical_repo_file(relative, name="implementation file")
        result[relative] = file_sha256(path)
    if (
        result[TEMPERATURE_WRAPPER_REPO_PATH]
        != TEMPERATURE_WRAPPER_FILE_SHA256
    ):
        raise RuntimeError("temperature wrapper changed")
    return tuple(sorted(result.items()))


def _flag_value(tokens: Sequence[str], name: str) -> str:
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == name:
            if index + 1 >= len(tokens):
                raise RuntimeError(f"temperature wrapper lacks {name}")
            values.append(tokens[index + 1])
        elif token.startswith(f"{name}="):
            values.append(token.split("=", maxsplit=1)[1])
    if len(values) != 1:
        raise RuntimeError(
            f"temperature wrapper must specify {name} exactly once"
        )
    return values[0]


def _read_parent_command() -> tuple[str, ...]:
    path = Path("/proc") / str(os.getppid()) / "cmdline"
    tokens = tuple(
        item.decode("utf-8", errors="strict")
        for item in path.read_bytes().split(b"\0")
        if item
    )
    if not tokens:
        raise RuntimeError("temperature wrapper command is unavailable")
    return tokens


def _verify_runtime_contract() -> dict[str, object]:
    """Verify the non-CUDA envelope before claiming the fixed output."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != FROZEN_VISIBLE_GPU:
        raise RuntimeError("PACRE bounded run fixes visible GPU 0")
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != FROZEN_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "PACRE bounded run fixes CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    wrapper = _canonical_repo_file(
        TEMPERATURE_WRAPPER_REPO_PATH,
        name="temperature wrapper",
    )
    tokens = _read_parent_command()
    candidates: list[Path] = []
    for token in tokens:
        if not token.endswith("run_with_gpu_temperature_control.py"):
            continue
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            candidates.append(candidate.resolve(strict=True))
        except OSError:
            continue
    if candidates != [wrapper]:
        raise RuntimeError(
            "PACRE bounded run requires the fixed temperature wrapper"
        )
    expected = {
        "--gpu": FROZEN_VISIBLE_GPU,
        "--pause-temp": str(FROZEN_PAUSE_TEMPERATURE_C),
        "--resume-temp": str(FROZEN_RESUME_TEMPERATURE_C),
    }
    for name, value in expected.items():
        if _flag_value(tokens, name) != value:
            raise RuntimeError(
                f"temperature wrapper {name} differs from protocol"
            )
    if file_sha256(wrapper) != TEMPERATURE_WRAPPER_FILE_SHA256:
        raise RuntimeError("temperature wrapper changed")
    return {
        "device": FROZEN_DEVICE,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "CUBLAS_WORKSPACE_CONFIG": os.environ[
            "CUBLAS_WORKSPACE_CONFIG"
        ],
        "temperature_wrapper_repo_path": (
            TEMPERATURE_WRAPPER_REPO_PATH
        ),
        "temperature_wrapper_file_sha256": (
            TEMPERATURE_WRAPPER_FILE_SHA256
        ),
        "pause_temperature_c": FROZEN_PAUSE_TEMPERATURE_C,
        "resume_temperature_c": FROZEN_RESUME_TEMPERATURE_C,
    }


def _verify_cuda_runtime_contract(
    runtime_envelope: Mapping[str, object],
) -> dict[str, object]:
    """Touch CUDA only after the exclusive output claim exists."""

    if (
        not isinstance(runtime_envelope, Mapping)
        or frozenset(runtime_envelope) != _ATTEMPT_RUNTIME_FIELDS
        or runtime_envelope.get("device") != FROZEN_DEVICE
        or runtime_envelope.get("CUDA_VISIBLE_DEVICES")
        != FROZEN_VISIBLE_GPU
        or runtime_envelope.get("CUBLAS_WORKSPACE_CONFIG")
        != FROZEN_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError("PACRE runtime envelope changed before CUDA")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "PACRE bounded run requires one visible CUDA device"
        )
    if torch.cuda.current_device() != 0:
        raise RuntimeError(
            "PACRE bounded run requires logical cuda:0"
        )
    properties = torch.cuda.get_device_properties(0)
    return {
        **dict(runtime_envelope),
        "visible_device_count": torch.cuda.device_count(),
        "visible_device_index": torch.cuda.current_device(),
        "device_name": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
        "cuda_runtime_verified_after_output_claim": True,
    }


def _validate_dataset_free_receipt(
    receipt: Mapping[str, object],
) -> str:
    if not isinstance(receipt, Mapping):
        raise TypeError("dataset-free receipt must be a mapping")
    body = dict(receipt)
    fingerprint = body.pop("receipt_fingerprint", None)
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or stable_fingerprint(body) != fingerprint
        or body.get("candidate") != "PACRE-v22"
        or body.get("gate_passed") is not True
        or body.get("parameter_count") != FROZEN_PARAMETER_COUNT
        or body.get("D_R_accessed") is not False
        or body.get("D_V_accessed") is not False
        or body.get("D_T_accessed") is not False
        or body.get("training_performed") is not False
    ):
        raise PermissionError("PACRE dataset-free prerequisite differs")
    return fingerprint


def _static_config_payload(
    *,
    source_paths: Mapping[str, Path],
    implementation: tuple[tuple[str, str], ...],
    dataset_free_receipt_fingerprint: str,
) -> dict[str, object]:
    if len(dataset_free_receipt_fingerprint) != 64:
        raise ValueError("dataset-free fingerprint is malformed")
    frozen_hashes = {
        name: digest for name, _, digest in FROZEN_REAL_DR_INPUTS
    }
    return {
        "schema_version": RUN_SCHEMA,
        "run_id": RUN_ID,
        "output_repo_path": OUTPUT_REPO_PATH,
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "real_inputs": {
            name: {
                "repo_path": str(path.relative_to(_ROOT)),
                "file_sha256": frozen_hashes[name],
            }
            for name, path in sorted(source_paths.items())
        },
        "model": {
            "class": (
                "CURELitePhaseAlignedCenteredResidualCompatibility"
                "EnergyLevelSet"
            ),
            "candidate": "PACRE-v22",
            "input_interface": ["F_b", "O"],
            "input_representation": "phase_preserving",
            "field_policy": CSLF_PACRE_FIELD_POLICY,
            "equation_policy": CSLF_PACRE_EQUATION_POLICY,
            "centering_policy": CSLF_PACRE_CENTERING_POLICY,
            "feature_channels": FROZEN_FEATURE_CHANNELS,
            "feature_stride": FROZEN_FEATURE_STRIDE,
            "width": FROZEN_MODEL_WIDTH,
            "parameter_count": FROZEN_PARAMETER_COUNT,
            "parameter_tensor_count": 3,
            "single_completion_field": True,
            "additional_heads": 0,
            "additional_branches": 0,
            "field_threshold": 0.0,
            "threshold_search_performed": False,
            "objective": PACRE_PMOPE_OBJECTIVE,
            "objective_policy": CSLF_PMOPE_POLICY,
        },
        "budget": {
            "seed": FROZEN_SEED,
            "epochs": FROZEN_EPOCHS,
            "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
            "updates": FROZEN_UPDATES,
            "objectives": 1,
        },
        "execution": {
            "device": FROZEN_DEVICE,
            "CUDA_VISIBLE_DEVICES": FROZEN_VISIBLE_GPU,
            "CUBLAS_WORKSPACE_CONFIG": (
                FROZEN_CUBLAS_WORKSPACE_CONFIG
            ),
            "temperature_wrapper_repo_path": (
                TEMPERATURE_WRAPPER_REPO_PATH
            ),
            "temperature_wrapper_file_sha256": (
                TEMPERATURE_WRAPPER_FILE_SHA256
            ),
            "pause_temperature_c": FROZEN_PAUSE_TEMPERATURE_C,
            "resume_temperature_c": FROZEN_RESUME_TEMPERATURE_C,
            "checkpoint_serialization": (
                FROZEN_CHECKPOINT_SERIALIZATION
            ),
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        },
        "dataset_free_gate": {
            "receipt_fingerprint": (
                dataset_free_receipt_fingerprint
            ),
            "invocations": 1,
        },
        "real_D_R_gate": {
            "status": "not_run_in_static_config",
            "invocations_if_run_once": 1,
        },
        "implementation": {
            "files": dict(implementation),
            "implementation_fingerprint": stable_fingerprint(
                dict(implementation)
            ),
        },
        "evidence_scope": {
            "D_R_cached_tensor_payload_accessed": False,
            "D_R_gate_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
            "formal_800_executed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
        },
    }


def _expected_model_config(preflight: object) -> CoverageStatePACREConfig:
    population = getattr(preflight, "population")
    cache = population.cache
    first = cache.raw_catalog.natural_records[0]
    feature_channels = int(first.feature.shape[1])
    feature_stride = int(cache.raw_catalog.feature_stride)
    if (
        feature_channels != FROZEN_FEATURE_CHANNELS
        or feature_stride != FROZEN_FEATURE_STRIDE
    ):
        raise RuntimeError("real D_R PACRE feature contract differs")
    config = CoverageStatePACREConfig(
        feature_channels=feature_channels,
        feature_stride=feature_stride,
        width=FROZEN_MODEL_WIDTH,
    )
    if config.expected_parameter_count != FROZEN_PARAMETER_COUNT:
        raise RuntimeError("real D_R PACRE parameter count differs")
    return config


def _write_checkpoint_new(
    directory: Path,
    *,
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    expected_artifacts: dict[str, bytes] | None = None,
) -> dict[str, object]:
    if type(model) is not (
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
    ):
        raise TypeError("checkpoint requires the exact PACRE model")
    state = {
        name: value.detach().to("cpu").contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    from safetensors.torch import load_file, save

    path = directory / f"{PACRE_PMOPE_OBJECTIVE}.safetensors"
    serialized = save(state)
    _write_new_bytes(path, serialized)
    if expected_artifacts is not None:
        relative = str(path.relative_to(OUTPUT_PATH))
        if relative in expected_artifacts:
            raise RuntimeError("PACRE checkpoint was registered twice")
        expected_artifacts[relative] = serialized
    loaded = load_file(str(path), device="cpu")
    if set(loaded) != set(state) or any(
        not torch.equal(loaded[name], state[name]) for name in state
    ):
        raise RuntimeError("PACRE checkpoint roundtrip changed")
    config = model.config
    payload = _fingerprinted(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "run_id": RUN_ID,
            "objective": PACRE_PMOPE_OBJECTIVE,
            "objective_policy": CSLF_PMOPE_POLICY,
            "model_class": type(model).__name__,
            "model_config": {
                "feature_channels": config.feature_channels,
                "feature_stride": config.feature_stride,
                "width": config.width,
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                ),
                "field_policy": config.field_policy,
                "equation_policy": config.equation_policy,
                "centering_policy": config.centering_policy,
            },
            "repo_relative_path": str(path.relative_to(_ROOT)),
            "serialization": FROZEN_CHECKPOINT_SERIALIZATION,
            "tensor_only_state_dict": True,
            "weights_only_roundtrip_verified": True,
            "checkpoint_file_sha256": file_sha256(path),
            "module_state_fingerprint": (
                module_state_fingerprint(model)
            ),
            "state_keys": list(state),
            "device_policy": "cpu_checkpoint",
        }
    )
    receipt_path = (
        directory / f"{PACRE_PMOPE_OBJECTIVE}.checkpoint.json"
    )
    if expected_artifacts is None:
        _write_new_json(receipt_path, payload)
    else:
        _write_run_json(
            receipt_path,
            payload,
            expected_artifacts=expected_artifacts,
        )
    return payload


def _failure_payload(
    error: BaseException,
    *,
    attempt_receipt_fingerprint: str,
    artifact_files: Mapping[str, str],
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": FAILURE_SCHEMA,
            "run_id": RUN_ID,
            "status": "failed_incomplete_attempt",
            "exception_type": type(error).__name__,
            "message": str(error),
            "attempt_receipt_fingerprint": (
                attempt_receipt_fingerprint
            ),
            "artifact_files_before_failure": dict(artifact_files),
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def validate_create_only() -> dict[str, object]:
    """Validate generated/static bindings without entering real ``D_R``."""

    source_paths = _verify_frozen_sources()
    implementation = _implementation_binding()
    dataset_free = run_pacre_dataset_free_gate()
    dataset_fingerprint = _validate_dataset_free_receipt(dataset_free)
    config = _static_config_payload(
        source_paths=source_paths,
        implementation=implementation,
        dataset_free_receipt_fingerprint=dataset_fingerprint,
    )
    return _fingerprinted(
        {
            "schema_version": VALIDATION_SCHEMA,
            "run_id": RUN_ID,
            "mode": "create_only_protocol_validation",
            "static_contract_valid": True,
            "config_fingerprint": stable_fingerprint(config),
            "implementation_fingerprint": stable_fingerprint(
                dict(implementation)
            ),
            "dataset_free_receipt_fingerprint": (
                dataset_fingerprint
            ),
            "dataset_free_gate_passed": True,
            "dataset_free_invocations": 1,
            "bounded_output_exists": (
                OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink()
            ),
            "output_claimed": False,
            "D_R_cached_tensor_payload_accessed": False,
            "D_R_gate_performed": False,
            "authorization_created": False,
            "optimizer_constructed": False,
            "training_performed": False,
            "checkpoint_written": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "formal800_eligible": False,
            "formal_800_authorized": False,
            "not_a_formal_result": True,
        }
    )


def _validate_terminal_artifact_population(
    *,
    expected_artifact_bytes: Mapping[str, bytes],
    pass_path: bool,
) -> dict[str, dict[str, object]]:
    """Re-read every fixed artifact and compare it with intended bytes."""

    expected_names = (
        _PASS_TERMINAL_FILES if pass_path else _COMMON_TERMINAL_FILES
    )
    if (
        not isinstance(expected_artifact_bytes, Mapping)
        or frozenset(expected_artifact_bytes) != expected_names
        or len(expected_names)
        != (
            FROZEN_TERMINAL_ARTIFACT_FILE_COUNT
            if pass_path
            else FROZEN_DR_GATE_STOP_ARTIFACT_FILE_COUNT
        )
    ):
        raise RuntimeError(
            "PACRE intended terminal artifact population is incomplete"
        )
    current_hashes = _artifact_hashes(OUTPUT_PATH)
    if frozenset(current_hashes) != expected_names:
        raise RuntimeError(
            "PACRE persisted terminal artifact names differ"
        )

    parsed: dict[str, dict[str, object]] = {}
    for relative in sorted(expected_names):
        intended = expected_artifact_bytes[relative]
        if not isinstance(intended, bytes):
            raise TypeError("PACRE intended artifact bytes are invalid")
        path = OUTPUT_PATH / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != intended
            or file_sha256(path)
            != sha256(intended).hexdigest()
        ):
            raise RuntimeError(
                f"PACRE persisted artifact changed: {relative}"
            )
        if relative == _CHECKPOINT_FILE:
            continue
        payload = _read_canonical_json_object(path)
        parsed[relative] = payload
        if relative == "attempt.json":
            _validate_attempt_receipt(payload)
            continue
        schema, fingerprint_field = _TERMINAL_JSON_CONTRACTS[relative]
        _validate_self_fingerprinted_receipt(
            payload,
            schema=schema,
            field=fingerprint_field,
            name=relative,
        )

    attempt = parsed["attempt.json"]
    config = parsed["receipts/config.json"]
    dataset_free = parsed["receipts/dataset_free.json"]
    inputs = parsed["receipts/inputs.json"]
    preflight = parsed["receipts/preflight.json"]
    dr_gate = parsed["receipts/dr_gate.json"]
    decision = parsed["receipts/decision.json"]

    attempt_fingerprint = attempt["receipt_fingerprint"]
    config_fingerprint = config["receipt_fingerprint"]
    dataset_receipt = dataset_free.get("receipt")
    bounded_population = inputs.get("bounded_population")
    preflight_payload = preflight.get("preflight")
    schedule_payload = preflight.get("schedule")
    dr_receipt = dr_gate.get("receipt")
    runtime = inputs.get("runtime")
    if not all(
        isinstance(value, Mapping)
        for value in (
            dataset_receipt,
            bounded_population,
            preflight_payload,
            schedule_payload,
            dr_receipt,
            runtime,
        )
    ):
        raise TypeError("PACRE persisted prerequisite payload is invalid")
    dataset_receipt = dict(dataset_receipt)
    dataset_inner_fingerprint = dataset_receipt.pop(
        "receipt_fingerprint",
        None,
    )
    config_dataset_gate = config.get("dataset_free_gate")
    if not isinstance(config_dataset_gate, Mapping):
        raise TypeError("PACRE config dataset-free binding is invalid")
    attempt_runtime = attempt.get("runtime")
    if not isinstance(attempt_runtime, Mapping):
        raise TypeError("PACRE attempt runtime binding is invalid")
    if (
        dataset_free.get("receipt_fingerprint")
        != dataset_inner_fingerprint
        or stable_fingerprint(dataset_receipt)
        != dataset_inner_fingerprint
        or attempt.get("dataset_free_receipt_fingerprint")
        != dataset_inner_fingerprint
        or config_dataset_gate.get("receipt_fingerprint")
        != dataset_inner_fingerprint
        or inputs.get("attempt_receipt_fingerprint")
        != attempt_fingerprint
        or inputs.get("config_fingerprint") != config_fingerprint
        or frozenset(runtime) != _VERIFIED_RUNTIME_FIELDS
        or runtime.get("cuda_runtime_verified_after_output_claim")
        is not True
        or any(
            runtime.get(name) != value
            for name, value in attempt_runtime.items()
        )
        or stable_fingerprint(dict(bounded_population))
        != inputs.get("population_fingerprint")
        or stable_fingerprint(dict(preflight_payload))
        != preflight.get("preflight_fingerprint")
        or stable_fingerprint(dict(schedule_payload))
        != preflight_payload.get("schedule_fingerprint")
        or preflight_payload.get("population_fingerprint")
        != inputs.get("population_fingerprint")
        or preflight_payload.get("bounded_cache_fingerprint")
        != inputs.get("bounded_cache_fingerprint")
        or dr_gate.get("receipt_fingerprint")
        != stable_fingerprint(dict(dr_receipt))
        or dr_receipt.get("dataset_free_receipt_fingerprint")
        != dataset_inner_fingerprint
        or dr_receipt.get("real_inputs_fingerprint")
        != inputs.get("real_inputs_fingerprint")
        or dr_receipt.get("population_fingerprint")
        != inputs.get("population_fingerprint")
        or dr_receipt.get("cache_fingerprint")
        != inputs.get("bounded_cache_fingerprint")
        or dr_gate.get("decision") != dr_receipt.get("decision")
        or dr_gate.get("gate_passed")
        is not bool(dr_receipt.get("gate_passed"))
    ):
        raise RuntimeError(
            "PACRE persisted prerequisite artifact association changed"
        )

    if not pass_path:
        if (
            decision.get("D_R_gate_receipt_fingerprint")
            != dr_gate.get("receipt_fingerprint")
            or decision.get("D_R_gate_decision")
            != dr_gate.get("decision")
            or decision.get("authorization_created") is not False
            or decision.get("bounded_training_performed") is not False
        ):
            raise RuntimeError(
                "PACRE D_R-stop artifact association changed"
            )
        return parsed

    authorization = parsed["receipts/authorization.json"]
    training = parsed["receipts/training.json"]
    zero_level = parsed["receipts/zero_level.json"]
    bounded_result = parsed["receipts/bounded_result.json"]
    checkpoint = parsed[_CHECKPOINT_RECEIPT_FILE]
    authorization_payload = authorization.get("authorization")
    training_receipt = training.get("training_receipt")
    training_result = training.get("training_result")
    diagnostic = zero_level.get("diagnostic")
    result_payload = bounded_result.get("result")
    pacre_decision = decision.get("PACRE_decision")
    output_claim = (
        authorization_payload.get("output_claim")
        if isinstance(authorization_payload, Mapping)
        else None
    )
    if not all(
        isinstance(value, Mapping)
        for value in (
            authorization_payload,
            training_receipt,
            training_result,
            diagnostic,
            result_payload,
            pacre_decision,
            output_claim,
        )
    ):
        raise TypeError("PACRE persisted pass payload is invalid")
    authorization_payload = dict(authorization_payload)
    result_payload = dict(result_payload)
    weights_path = OUTPUT_PATH / _CHECKPOINT_FILE
    result_training = result_payload.get("training")
    if not isinstance(result_training, Mapping):
        raise TypeError("PACRE result training payload is invalid")
    if (
        stable_fingerprint(authorization_payload)
        != authorization.get("authorization_fingerprint")
        or authorization_payload.get("output_claim_fingerprint")
        != inputs.get("output_claim_fingerprint")
        or output_claim.get("attempt_receipt_fingerprint")
        != attempt_fingerprint
        or output_claim.get("config_fingerprint")
        != config_fingerprint
        or output_claim.get("dataset_free_receipt_fingerprint")
        != dataset_inner_fingerprint
        or output_claim.get("runtime") != attempt_runtime
        or output_claim.get("process_identity")
        != attempt.get("process_identity")
        or authorization_payload.get("dataset_free_receipt_fingerprint")
        != dataset_inner_fingerprint
        or authorization_payload.get("D_R_gate_receipt_fingerprint")
        != dr_gate.get("receipt_fingerprint")
        or authorization_payload.get("preflight_fingerprint")
        != preflight.get("preflight_fingerprint")
        or stable_fingerprint(dict(training_receipt))
        != training.get("training_receipt_fingerprint")
        or stable_fingerprint(dict(training_result))
        != training.get("training_result_fingerprint")
        or training.get("checkpoint_receipt_fingerprint")
        != checkpoint.get("receipt_fingerprint")
        or checkpoint.get("checkpoint_file_sha256")
        != file_sha256(weights_path)
        or checkpoint.get("repo_relative_path")
        != f"{OUTPUT_REPO_PATH}/{_CHECKPOINT_FILE}"
        or stable_fingerprint(dict(diagnostic))
        != zero_level.get("diagnostic_result_fingerprint")
        or diagnostic.get("checkpoint_fingerprint")
        != checkpoint.get("module_state_fingerprint")
        or stable_fingerprint(result_payload)
        != bounded_result.get("result_fingerprint")
        or result_payload.get("authorization_fingerprint")
        != authorization.get("authorization_fingerprint")
        or result_training.get("receipt") != training_receipt
        or result_training.get("result") != training_result
        or result_payload.get("diagnostic") != diagnostic
        or result_payload.get("decision") != pacre_decision
        or decision.get("result_fingerprint")
        != bounded_result.get("result_fingerprint")
        or decision.get("checkpoint_receipt_fingerprint")
        != checkpoint.get("receipt_fingerprint")
        or decision.get("PACRE_decision_fingerprint")
        != stable_fingerprint(dict(pacre_decision))
    ):
        raise RuntimeError(
            "PACRE persisted pass artifact association changed"
        )
    return parsed


def _validate_terminal_associations(
    *,
    decision: Mapping[str, object],
    expected_attempt_receipt_fingerprint: str,
    expected_config_fingerprint: str,
    expected_authorization_fingerprint: str | None,
    expected_result_fingerprint: str | None,
) -> dict[str, object]:
    """Re-read and cross-check the persisted terminal evidence graph."""

    if (
        not _is_sha256(expected_attempt_receipt_fingerprint)
        or not _is_sha256(expected_config_fingerprint)
        or (
            expected_authorization_fingerprint is not None
            and not _is_sha256(expected_authorization_fingerprint)
        )
        or (
            expected_result_fingerprint is not None
            and not _is_sha256(expected_result_fingerprint)
        )
        or (
            (expected_authorization_fingerprint is None)
            != (expected_result_fingerprint is None)
        )
    ):
        raise ValueError("PACRE expected terminal links are malformed")

    attempt_path = OUTPUT_PATH / "attempt.json"
    config_path = OUTPUT_PATH / "receipts" / "config.json"
    decision_path = OUTPUT_PATH / "receipts" / "decision.json"
    authorization_path = (
        OUTPUT_PATH / "receipts" / "authorization.json"
    )
    result_path = (
        OUTPUT_PATH / "receipts" / "bounded_result.json"
    )

    attempt = _read_canonical_json_object(attempt_path)
    attempt_fingerprint = _validate_attempt_receipt(attempt)
    config = _read_canonical_json_object(config_path)
    config_fingerprint = _validate_self_fingerprinted_receipt(
        config,
        schema=RUN_SCHEMA,
        name="config",
    )
    persisted_decision = _read_canonical_json_object(decision_path)
    decision_fingerprint = _validate_self_fingerprinted_receipt(
        persisted_decision,
        schema=DECISION_SCHEMA,
        name="decision",
    )
    supplied_decision = dict(decision)

    if (
        attempt_fingerprint != expected_attempt_receipt_fingerprint
        or config_fingerprint != expected_config_fingerprint
        or attempt.get("config_fingerprint")
        != expected_config_fingerprint
        or config.get("run_id") != attempt.get("run_id")
        or persisted_decision != supplied_decision
        or persisted_decision.get("run_id")
        != attempt.get("run_id")
        or persisted_decision.get("attempt_receipt_fingerprint")
        != expected_attempt_receipt_fingerprint
        or persisted_decision.get("config_fingerprint")
        != expected_config_fingerprint
    ):
        raise RuntimeError(
            "PACRE attempt/config/decision association changed"
        )

    if expected_authorization_fingerprint is None:
        if (
            authorization_path.exists()
            or authorization_path.is_symlink()
            or result_path.exists()
            or result_path.is_symlink()
            or "authorization_fingerprint" in persisted_decision
            or "result_fingerprint" in persisted_decision
            or persisted_decision.get("authorization_created")
            is not False
            or persisted_decision.get("bounded_training_performed")
            is not False
        ):
            raise RuntimeError(
                "PACRE stopped decision has forbidden terminal links"
            )
        return {
            "attempt_receipt_fingerprint": attempt_fingerprint,
            "config_fingerprint": config_fingerprint,
            "decision_receipt_fingerprint": decision_fingerprint,
        }

    authorization = _read_canonical_json_object(authorization_path)
    authorization_wrapper_fingerprint = (
        _validate_self_fingerprinted_receipt(
            authorization,
            schema=AUTHORIZATION_WRAPPER_SCHEMA,
            name="authorization wrapper",
        )
    )
    authorization_payload = authorization.get("authorization")
    if not isinstance(authorization_payload, Mapping):
        raise TypeError("PACRE authorization payload is invalid")
    authorization_payload = dict(authorization_payload)

    bounded_result = _read_canonical_json_object(result_path)
    result_wrapper_fingerprint = (
        _validate_self_fingerprinted_receipt(
            bounded_result,
            schema=RESULT_WRAPPER_SCHEMA,
            name="bounded result wrapper",
        )
    )
    result_payload = bounded_result.get("result")
    if not isinstance(result_payload, Mapping):
        raise TypeError("PACRE bounded result payload is invalid")
    result_payload = dict(result_payload)
    pacre_decision = persisted_decision.get("PACRE_decision")
    if not isinstance(pacre_decision, Mapping):
        raise TypeError("PACRE terminal decision payload is invalid")
    pacre_decision = dict(pacre_decision)

    if (
        authorization.get("run_id") != RUN_ID
        or authorization.get("attempt_receipt_fingerprint")
        != expected_attempt_receipt_fingerprint
        or authorization.get("config_fingerprint")
        != expected_config_fingerprint
        or authorization.get("authorization_fingerprint")
        != expected_authorization_fingerprint
        or authorization_payload.get("run_id") != RUN_ID
        or authorization.get("authorization_attempt_fingerprint")
        != authorization_payload.get("attempt_fingerprint")
        or not _is_sha256(
            authorization.get("authorization_attempt_fingerprint")
        )
        or stable_fingerprint(authorization_payload)
        != expected_authorization_fingerprint
        or bounded_result.get("run_id") != RUN_ID
        or bounded_result.get("attempt_receipt_fingerprint")
        != expected_attempt_receipt_fingerprint
        or bounded_result.get("config_fingerprint")
        != expected_config_fingerprint
        or bounded_result.get("authorization_fingerprint")
        != expected_authorization_fingerprint
        or bounded_result.get("result_fingerprint")
        != expected_result_fingerprint
        or result_payload.get("run_id") != RUN_ID
        or result_payload.get("authorization_fingerprint")
        != expected_authorization_fingerprint
        or stable_fingerprint(result_payload)
        != expected_result_fingerprint
        or bounded_result.get("authorization_attempt_fingerprint")
        != authorization_payload.get("attempt_fingerprint")
        or persisted_decision.get("attempt_receipt_fingerprint")
        != expected_attempt_receipt_fingerprint
        or persisted_decision.get("authorization_attempt_fingerprint")
        != authorization_payload.get("attempt_fingerprint")
        or persisted_decision.get("authorization_fingerprint")
        != expected_authorization_fingerprint
        or persisted_decision.get("result_fingerprint")
        != expected_result_fingerprint
        or persisted_decision.get("PACRE_decision_fingerprint")
        != stable_fingerprint(pacre_decision)
        or result_payload.get("decision") != pacre_decision
    ):
        raise RuntimeError(
            "PACRE authorization/result/decision association changed"
        )
    return {
        "attempt_receipt_fingerprint": attempt_fingerprint,
        "config_fingerprint": config_fingerprint,
        "authorization_wrapper_fingerprint": (
            authorization_wrapper_fingerprint
        ),
        "authorization_fingerprint": (
            expected_authorization_fingerprint
        ),
        "authorization_attempt_fingerprint": (
            authorization_payload["attempt_fingerprint"]
        ),
        "result_wrapper_fingerprint": result_wrapper_fingerprint,
        "result_fingerprint": expected_result_fingerprint,
        "decision_receipt_fingerprint": decision_fingerprint,
    }


def _complete_run(
    *,
    decision: Mapping[str, object],
    expected_artifact_count: int,
    fields: Mapping[str, object],
    expected_attempt_receipt_fingerprint: str,
    expected_config_fingerprint: str,
    expected_authorization_fingerprint: str | None,
    expected_result_fingerprint: str | None,
    expected_artifact_bytes: Mapping[str, bytes],
) -> dict[str, object]:
    if _implementation_binding() != _RUN_IMPLEMENTATION_BINDING:
        raise RuntimeError("PACRE implementation changed during execution")
    _validate_terminal_artifact_population(
        expected_artifact_bytes=expected_artifact_bytes,
        pass_path=expected_authorization_fingerprint is not None,
    )
    terminal_links = _validate_terminal_associations(
        decision=decision,
        expected_attempt_receipt_fingerprint=(
            expected_attempt_receipt_fingerprint
        ),
        expected_config_fingerprint=expected_config_fingerprint,
        expected_authorization_fingerprint=(
            expected_authorization_fingerprint
        ),
        expected_result_fingerprint=expected_result_fingerprint,
    )
    artifacts = _artifact_hashes(OUTPUT_PATH)
    if len(artifacts) != expected_artifact_count:
        raise RuntimeError(
            "PACRE terminal artifact population is incomplete"
        )
    complete = _fingerprinted(
        {
            "schema_version": RUN_SCHEMA,
            "run_id": RUN_ID,
            "status": "complete",
            "decision": decision["status"],
            "artifact_files": artifacts,
            "artifact_file_count": len(artifacts),
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            **terminal_links,
            **dict(fields),
        },
        field="complete_fingerprint",
    )
    incomplete_path = OUTPUT_PATH / _INCOMPLETE
    complete_path = OUTPUT_PATH / "COMPLETE.json"
    if (
        incomplete_path.is_symlink()
        or not incomplete_path.is_file()
        or complete_path.exists()
        or complete_path.is_symlink()
    ):
        raise RuntimeError("PACRE terminal publication marker is invalid")
    # Turn the existing incomplete marker into the complete receipt, then
    # atomically rename it.  A failure before rename leaves only
    # ``.incomplete``; successful rename creates COMPLETE and removes the
    # incomplete state in one filesystem operation.
    with incomplete_path.open("r+b") as handle:
        handle.seek(0)
        handle.truncate()
        handle.write(_json_bytes(complete))
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(incomplete_path, complete_path)
    return complete


# Set only for the duration of ``run_once``.  Keeping this process-local
# avoids recomputing a large source binding merely to finalize the artifact.
_RUN_IMPLEMENTATION_BINDING: tuple[tuple[str, str], ...] = ()


def run_once() -> dict[str, object]:
    """Consume the sole seed-42 PACRE bounded-400 attempt."""

    global _RUN_IMPLEMENTATION_BINDING
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError(
            f"single-use bounded output already exists: {OUTPUT_PATH}"
        )

    source_paths = _verify_frozen_sources()
    implementation = _implementation_binding()
    runtime_envelope = _verify_runtime_contract()
    dataset_free = run_pacre_dataset_free_gate()
    dataset_fingerprint = _validate_dataset_free_receipt(dataset_free)
    config = _fingerprinted(
        _static_config_payload(
            source_paths=source_paths,
            implementation=implementation,
            dataset_free_receipt_fingerprint=dataset_fingerprint,
        )
    )
    attempt = _fingerprinted(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "run_id": RUN_ID,
            "output_repo_path": OUTPUT_REPO_PATH,
            "config_fingerprint": config["receipt_fingerprint"],
            "runtime": runtime_envelope,
            "candidate": "PACRE-v22",
            "objective": PACRE_PMOPE_OBJECTIVE,
            "budget": {
                "seed": FROZEN_SEED,
                "epochs": FROZEN_EPOCHS,
                "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
                "updates": FROZEN_UPDATES,
            },
            "process_identity": pacre_bounded_process_identity(),
            "dataset_free_receipt_fingerprint": (
                dataset_fingerprint
            ),
            "dataset_free_invocations_before_claim": 1,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    attempt_receipt_fingerprint = str(
        attempt["receipt_fingerprint"]
    )
    config_fingerprint = str(config["receipt_fingerprint"])
    try:
        receipts, checkpoints_dir = _claim_output(
            OUTPUT_PATH,
            attempt=attempt,
        )
    except BaseException:
        raise

    expected_artifact_bytes: dict[str, bytes] = {
        "attempt.json": _json_bytes(attempt)
    }
    _RUN_IMPLEMENTATION_BINDING = implementation
    try:
        output_claim = load_pacre_bounded_output_claim()
        runtime = _verify_cuda_runtime_contract(runtime_envelope)
        _write_run_json(
            receipts / "config.json",
            config,
            expected_artifacts=expected_artifact_bytes,
        )
        _write_run_json(
            receipts / "dataset_free.json",
            _fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-v22-pacre-bounded-400-"
                        "dataset-free-v1"
                    ),
                    "run_id": RUN_ID,
                    "receipt": dict(dataset_free),
                    "receipt_fingerprint": dataset_fingerprint,
                    "gate_passed": True,
                    "invocations": 1,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                },
                field="wrapper_fingerprint",
            ),
            expected_artifacts=expected_artifact_bytes,
        )

        real_inputs = build_coverage_state_real_dr_inputs(**source_paths)
        population = build_coverage_state_bounded_population(
            real_inputs.scalar_cache,
            seed=FROZEN_SEED,
        )
        preflight = prepare_coverage_state_bounded_preflight(population)
        _write_run_json(
            receipts / "inputs.json",
            _fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-v22-pacre-bounded-400-inputs-v1"
                    ),
                    "run_id": RUN_ID,
                    "attempt_receipt_fingerprint": (
                        attempt_receipt_fingerprint
                    ),
                    "config_fingerprint": config_fingerprint,
                    "output_claim_fingerprint": (
                        output_claim.claim_fingerprint
                    ),
                    "runtime": runtime,
                    "real_D_R_inputs": real_inputs.canonical_payload(),
                    "source_binding": (
                        real_inputs.source_binding.canonical_payload()
                    ),
                    "real_inputs_fingerprint": (
                        real_inputs.build_fingerprint
                    ),
                    "bounded_population": (
                        population.canonical_payload()
                    ),
                    "population_fingerprint": (
                        population.population_fingerprint
                    ),
                    "bounded_cache_fingerprint": (
                        population.cache.cache_fingerprint
                    ),
                    "construction_invocations": {
                        "real_inputs": 1,
                        "population": 1,
                    },
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                }
            ),
            expected_artifacts=expected_artifact_bytes,
        )
        _write_run_json(
            receipts / "preflight.json",
            _fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-v22-pacre-bounded-400-preflight-v1"
                    ),
                    "run_id": RUN_ID,
                    "preflight": preflight.canonical_payload(),
                    "preflight_fingerprint": (
                        preflight.preflight_fingerprint
                    ),
                    "schedule": preflight.schedule.canonical_payload(),
                    "preflight_invocations": 1,
                    "training_authorized": (
                        preflight.training_authorized
                    ),
                    "formal_800_authorized": False,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                }
            ),
            expected_artifacts=expected_artifact_bytes,
        )
        if not preflight.training_authorized:
            raise PermissionError("bounded D_R preflight did not pass")

        model_config = _expected_model_config(preflight)
        dr_gate = run_pacre_dr_gate(
            dataset_free_receipt=dataset_free,
            real_inputs=real_inputs,
            bounded_population=population,
            device=FROZEN_DEVICE,
        )
        _write_run_json(
            receipts / "dr_gate.json",
            _fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-v22-pacre-bounded-400-real-D_R-"
                        "gate-v1"
                    ),
                    "run_id": RUN_ID,
                    "receipt": dr_gate.canonical_payload(),
                    "receipt_fingerprint": (
                        dr_gate.receipt_fingerprint
                    ),
                    "decision": dr_gate.decision,
                    "failed_checks": list(dr_gate.failed_checks),
                    "gate_passed": dr_gate.gate_passed,
                    "gate_invocations": 1,
                    "optimizer_steps": 0,
                    "training_performed": False,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                },
                field="wrapper_fingerprint",
            ),
            expected_artifacts=expected_artifact_bytes,
        )

        if not dr_gate.gate_passed:
            if dr_gate.decision != PACRE_DR_FAIL_DECISION:
                raise RuntimeError("PACRE D_R failure decision differs")
            decision = _fingerprinted(
                {
                    "schema_version": DECISION_SCHEMA,
                    "run_id": RUN_ID,
                    "attempt_receipt_fingerprint": (
                        attempt_receipt_fingerprint
                    ),
                    "config_fingerprint": config_fingerprint,
                    "status": "PACRE_V22_D_R_GATE_FAIL",
                    "D_R_gate_decision": dr_gate.decision,
                    "D_R_gate_receipt_fingerprint": (
                        dr_gate.receipt_fingerprint
                    ),
                    "bounded_gate_passed": False,
                    "authorization_created": False,
                    "bounded_training_performed": False,
                    "zero_level_evaluation_performed": False,
                    "checkpoint_count": 0,
                    "formal800_eligible": False,
                    "formal_800_authorized": False,
                    "next_action": (
                        "freeze_pacre_v22_D_R_negative_and_stop_before_"
                        "bounded_training"
                    ),
                    "resume_allowed": False,
                    "automatic_retry_allowed": False,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                }
            )
            _write_run_json(
                receipts / "decision.json",
                decision,
                expected_artifacts=expected_artifact_bytes,
            )
            complete = _complete_run(
                decision=decision,
                expected_artifact_count=(
                    FROZEN_DR_GATE_STOP_ARTIFACT_FILE_COUNT
                ),
                fields={
                    "bounded_gate_passed": False,
                    "D_R_gate_passed": False,
                    "authorization_created": False,
                    "bounded_training_performed": False,
                    "zero_level_evaluation_performed": False,
                    "checkpoint_count": 0,
                    "formal800_eligible": False,
                    "formal_800_authorized": False,
                    "dataset_free_invocations": 1,
                    "real_inputs_construction_invocations": 1,
                    "population_construction_invocations": 1,
                    "preflight_invocations": 1,
                    "D_R_gate_invocations": 1,
                    "bounded_runner_invocations": 0,
                },
                expected_attempt_receipt_fingerprint=(
                    attempt_receipt_fingerprint
                ),
                expected_config_fingerprint=config_fingerprint,
                expected_authorization_fingerprint=None,
                expected_result_fingerprint=None,
                expected_artifact_bytes=expected_artifact_bytes,
            )
            return {
                "run_id": RUN_ID,
                "output": str(OUTPUT_PATH),
                "decision": decision["status"],
                "bounded_gate_passed": False,
                "formal800_eligible": False,
                "complete_fingerprint": (
                    complete["complete_fingerprint"]
                ),
                "D_V_accessed": False,
                "D_T_accessed": False,
            }

        if dr_gate.decision != PACRE_DR_PASS_DECISION:
            raise RuntimeError("PACRE D_R pass decision differs")
        authorization = prepare_pacre_bounded_run_authorization(
            preflight,
            dataset_free,
            dr_gate,
            real_inputs,
            model_config,
            output_claim=output_claim,
            run_id=RUN_ID,
        )
        if (
            authorization.run_id != RUN_ID
            or not authorization.prerequisites_passed
            or not authorization.available
        ):
            raise PermissionError("PACRE bounded authorization did not pass")
        _write_run_json(
            receipts / "authorization.json",
            _fingerprinted(
                {
                    "schema_version": AUTHORIZATION_WRAPPER_SCHEMA,
                    "run_id": RUN_ID,
                    "attempt_receipt_fingerprint": (
                        attempt_receipt_fingerprint
                    ),
                    "config_fingerprint": config_fingerprint,
                    "authorization_attempt_fingerprint": (
                        authorization.attempt_fingerprint
                    ),
                    "authorization": (
                        authorization.canonical_payload()
                    ),
                    "authorization_fingerprint": (
                        authorization.authorization_fingerprint
                    ),
                    "training_authorized": True,
                    "available_before_runner": True,
                    "formal_800_authorized": False,
                }
            ),
            expected_artifacts=expected_artifact_bytes,
        )

        result = run_pacre_pmope_bounded_400(
            authorization,
            model_config,
            run_id=RUN_ID,
            device=FROZEN_DEVICE,
        )
        if (
            result.run_id != RUN_ID
            or result.training_invocations != 1
            or result.zero_level_evaluation_invocations != 1
            or result.decision_invocations != 1
        ):
            raise RuntimeError(
                "PACRE bounded runner did not execute exactly once"
            )
        bundle = result.training
        model = bundle.model
        training_result = bundle.training_result
        if (
            type(model)
            is not
            CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
            or training_result.seed != FROZEN_SEED
            or training_result.epochs != FROZEN_EPOCHS
            or training_result.steps_per_epoch
            != FROZEN_STEPS_PER_EPOCH
            or training_result.completed_updates != FROZEN_UPDATES
            or training_result.forward_calls != FROZEN_UPDATES
            or training_result.backward_calls != FROZEN_UPDATES
            or training_result.optimizer_steps != FROZEN_UPDATES
            or training_result.objective != PACRE_PMOPE_OBJECTIVE
        ):
            raise RuntimeError("PACRE bounded training ledger differs")

        checkpoint = _write_checkpoint_new(
            checkpoints_dir,
            model=model,
            expected_artifacts=expected_artifact_bytes,
        )
        _write_run_json(
            receipts / "training.json",
            _fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-v22-pacre-bounded-400-training-v1"
                    ),
                    "run_id": RUN_ID,
                    "bundle_fingerprint": bundle.bundle_fingerprint,
                    "training_receipt": (
                        bundle.receipt.canonical_payload()
                    ),
                    "training_receipt_fingerprint": (
                        bundle.receipt.receipt_fingerprint
                    ),
                    "training_result": (
                        training_result.canonical_payload()
                    ),
                    "training_result_fingerprint": (
                        training_result.result_fingerprint
                    ),
                    "checkpoint_receipt_fingerprint": (
                        checkpoint["receipt_fingerprint"]
                    ),
                    "training_invocations": 1,
                    "bounded_training_performed": True,
                    "formal_training_performed": False,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                }
            ),
            expected_artifacts=expected_artifact_bytes,
        )
        _write_run_json(
            receipts / "zero_level.json",
            _fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-v22-pacre-bounded-400-zero-level-v1"
                    ),
                    "run_id": RUN_ID,
                    "threshold": 0.0,
                    "threshold_search_performed": False,
                    "diagnostic": (
                        result.diagnostic.canonical_payload()
                    ),
                    "diagnostic_result_fingerprint": (
                        result.diagnostic.result_fingerprint
                    ),
                    "evaluation_invocations": 1,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                }
            ),
            expected_artifacts=expected_artifact_bytes,
        )
        bounded_payload = result.canonical_payload()
        _write_run_json(
            receipts / "bounded_result.json",
            _fingerprinted(
                {
                    "schema_version": RESULT_WRAPPER_SCHEMA,
                    "run_id": RUN_ID,
                    "attempt_receipt_fingerprint": (
                        attempt_receipt_fingerprint
                    ),
                    "config_fingerprint": config_fingerprint,
                    "authorization_attempt_fingerprint": (
                        authorization.attempt_fingerprint
                    ),
                    "authorization_fingerprint": (
                        authorization.authorization_fingerprint
                    ),
                    "result": bounded_payload,
                    "result_fingerprint": result.result_fingerprint,
                }
            ),
            expected_artifacts=expected_artifact_bytes,
        )
        passed = bool(result.bounded_gate_passed)
        decision = _fingerprinted(
            {
                "schema_version": DECISION_SCHEMA,
                "run_id": RUN_ID,
                "attempt_receipt_fingerprint": (
                    attempt_receipt_fingerprint
                ),
                "config_fingerprint": config_fingerprint,
                "authorization_attempt_fingerprint": (
                    authorization.attempt_fingerprint
                ),
                "authorization_fingerprint": (
                    authorization.authorization_fingerprint
                ),
                "status": (
                    "PACRE_V22_BOUNDED_400_GATE_PASS"
                    if passed
                    else "PACRE_V22_BOUNDED_400_GATE_FAIL"
                ),
                "bounded_gate_passed": passed,
                "failed_checks": list(result.failed_checks),
                "PACRE_decision": (
                    result.decision.canonical_payload()
                ),
                "PACRE_decision_fingerprint": (
                    result.decision.decision_fingerprint
                ),
                "result_fingerprint": result.result_fingerprint,
                "checkpoint_receipt_fingerprint": (
                    checkpoint["receipt_fingerprint"]
                ),
                "training_invocations": 1,
                "zero_level_evaluation_invocations": 1,
                "decision_invocations": 1,
                "formal800_eligible": result.formal800_eligible,
                "formal_800_authorized": False,
                "formal_800_executed": False,
                "next_action": (
                    "freeze_pacre_v22_and_design_formal800_protocol"
                    if passed
                    else "freeze_pacre_v22_bounded_negative_and_review"
                ),
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_run_json(
            receipts / "decision.json",
            decision,
            expected_artifacts=expected_artifact_bytes,
        )
        complete = _complete_run(
            decision=decision,
            expected_artifact_count=(
                FROZEN_TERMINAL_ARTIFACT_FILE_COUNT
            ),
            fields={
                "bounded_gate_passed": passed,
                "D_R_gate_passed": True,
                "authorization_created": True,
                "bounded_training_performed": True,
                "zero_level_evaluation_performed": True,
                "checkpoint_count": 1,
                "formal800_eligible": result.formal800_eligible,
                "formal_800_authorized": False,
                "dataset_free_invocations": 1,
                "real_inputs_construction_invocations": 1,
                "population_construction_invocations": 1,
                "preflight_invocations": 1,
                "D_R_gate_invocations": 1,
                "bounded_runner_invocations": 1,
            },
            expected_attempt_receipt_fingerprint=(
                attempt_receipt_fingerprint
            ),
            expected_config_fingerprint=config_fingerprint,
            expected_authorization_fingerprint=(
                authorization.authorization_fingerprint
            ),
            expected_result_fingerprint=(
                result.result_fingerprint
            ),
            expected_artifact_bytes=expected_artifact_bytes,
        )
        return {
            "run_id": RUN_ID,
            "output": str(OUTPUT_PATH),
            "decision": decision["status"],
            "bounded_gate_passed": passed,
            "formal800_eligible": result.formal800_eligible,
            "complete_fingerprint": complete["complete_fingerprint"],
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    except BaseException as error:
        try:
            _write_new_json(
                OUTPUT_PATH / "FAILURE.json",
                _failure_payload(
                    error,
                    attempt_receipt_fingerprint=(
                        attempt_receipt_fingerprint
                    ),
                    artifact_files=_artifact_hashes(OUTPUT_PATH),
                ),
            )
        except BaseException:
            pass
        raise
    finally:
        _RUN_IMPLEMENTATION_BINDING = ()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-create-only",
        action="store_true",
        help="validate without claiming output or loading real D_R",
    )
    mode.add_argument(
        "--run-once",
        action="store_true",
        help="consume the unique temperature-controlled bounded attempt",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = (
        validate_create_only()
        if args.validate_create_only
        else run_once()
    )
    print(
        json.dumps(
            result,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
