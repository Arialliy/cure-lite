#!/usr/bin/env python3
"""Execute the single frozen real-D_R CSLF bounded-400 run.

The formal entry point has no configurable data path, output path, device,
seed, budget, retry, or resume surface.  ``--validate-create-only`` checks the
static contract without claiming the run directory or loading cached tensors.
``--run-once`` is accepted only as a child of the bound GPU-temperature
wrapper and consumes the unique r1 output path before real D_R reconstruction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_device_cache import (
    prepare_coverage_state_device_cache,
)
from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
)
from cure_lite.coverage_state_schedule import (
    coverage_state_schedule_exposure_report,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_EPOCHS,
    COVERAGE_STATE_BOUNDED_SEED,
    COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH,
    COVERAGE_STATE_BOUNDED_UPDATES,
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_bounded_runner import (
    COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
    COVERAGE_STATE_BOUNDED_MODEL_WIDTH,
    CoverageStateBoundedRunResult,
    prepare_coverage_state_bounded_run_authorization,
    run_coverage_state_bounded_400,
)
from cure_lite.experiment.coverage_state_dataset_free import (
    run_coverage_state_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    bind_coverage_state_real_dr_sources,
    build_coverage_state_real_dr_inputs,
)
from cure_lite.frozen_base import module_state_fingerprint


_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "cure_lite_cslf_v15_bounded_400_r1"
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = _ROOT / OUTPUT_REPO_PATH
FROZEN_DEVICE = "cuda:0"
FROZEN_VISIBLE_GPU = "0"
FROZEN_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FROZEN_PAUSE_TEMPERATURE_C = 82
FROZEN_RESUME_TEMPERATURE_C = 75
TEMPERATURE_WRAPPER_REPO_PATH = "tools/run_with_gpu_temperature_control.py"
TEMPERATURE_WRAPPER_FILE_SHA256 = (
    "026b751fbb59530721da1436af32f3bc924c9ed2ab3576df062a45bca7ec5e86"
)

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

RUN_SCHEMA = "cure-lite-cslf-v15-bounded-400-run-v1"
ATTEMPT_SCHEMA = "cure-lite-cslf-v15-bounded-400-attempt-v1"
FAILURE_SCHEMA = "cure-lite-cslf-v15-bounded-400-failure-v1"
DECISION_SCHEMA = "cure-lite-cslf-v15-bounded-400-decision-v1"
VALIDATION_SCHEMA = "cure-lite-cslf-v15-bounded-400-create-only-validation-v1"
MEMORY_PREFLIGHT_SCHEMA = (
    "cure-lite-cslf-v15-bounded-400-device-memory-preflight-v1"
)
CHECKPOINT_SCHEMA = "cure-lite-cslf-v15-bounded-400-checkpoint-v1"

_INCOMPLETE = ".incomplete"
_ACTIVATION_RESERVE_BYTES = 2 * 1024**3

_EXTRA_IMPLEMENTATION_PATHS = (
    "cure_lite/cache/base_cache.py",
    "cure_lite/cache/state_cache.py",
    "cure_lite/config.py",
    "cure_lite/data.py",
    "cure_lite/experiment/cache_pipeline.py",
    "cure_lite/experiment/coverage_state_observability_protocol.py",
    "cure_lite/experiment/coverage_state_raw_catalog.py",
    "cure_lite/experiment/coverage_state_real_dr_inputs.py",
    "cure_lite/experiment/geometry_catalog_protocol.py",
    "cure_lite/experiment/geometry_safe_catalog.py",
    "cure_lite/experiment/training_pipeline.py",
    "cure_lite/intervention.py",
    "cure_lite/matching.py",
    "cure_lite/occupancy.py",
    "cure_lite/splits.py",
    "cure_lite/supervision.py",
    "tools/run_coverage_state_cslf_bounded_400.py",
    TEMPERATURE_WRAPPER_REPO_PATH,
)
IMPLEMENTATION_REPO_PATHS = tuple(
    dict.fromkeys(
        (
            *COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
            *_EXTRA_IMPLEMENTATION_PATHS,
        )
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    value = dict(payload)
    if field in value:
        raise ValueError(f"payload already contains {field}")
    value[field] = stable_fingerprint(value)
    return value


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


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _json_bytes(payload)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


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
    resolved: dict[str, Path] = {}
    for argument, relative, expected_sha256 in FROZEN_REAL_DR_INPUTS:
        path = _canonical_repo_file(relative, name=argument)
        if file_sha256(path) != expected_sha256:
            raise RuntimeError(f"frozen real D_R input changed: {argument}")
        resolved[argument] = path
    return resolved


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for relative in IMPLEMENTATION_REPO_PATHS:
        path = _canonical_repo_file(relative, name="implementation file")
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _static_config_payload(
    *,
    source_paths: Mapping[str, Path],
    implementation: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    return {
        "schema_version": RUN_SCHEMA,
        "run_id": RUN_ID,
        "output_repo_path": OUTPUT_REPO_PATH,
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "real_inputs": {
            name: {
                "repo_path": str(path.relative_to(_ROOT)),
                "file_sha256": dict(
                    (key, digest)
                    for key, _, digest in FROZEN_REAL_DR_INPUTS
                )[name],
            }
            for name, path in sorted(source_paths.items())
        },
        "model": {
            "representation": "scalar_max",
            "width": COVERAGE_STATE_BOUNDED_MODEL_WIDTH,
            "field_threshold": 0.0,
            "threshold_search_performed": False,
        },
        "budget": {
            "seed": COVERAGE_STATE_BOUNDED_SEED,
            "epochs": COVERAGE_STATE_BOUNDED_EPOCHS,
            "steps_per_epoch": (
                COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
            ),
            "updates_per_objective": COVERAGE_STATE_BOUNDED_UPDATES,
            "objectives": 3,
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
            "device_memory_preflight": {
                "packed_payload_accounting": (
                    "exact_cpu_projection_of_device_cache"
                ),
                "retained_model_optimizer_accounting": (
                    "six_parameter_copies_plus_three_buffer_copies"
                ),
                "fixed_activation_reserve_bytes": (
                    _ACTIVATION_RESERVE_BYTES
                ),
                "free_memory_must_meet_requirement": True,
            },
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        },
        "implementation": {
            "files": dict(implementation),
            "implementation_fingerprint": stable_fingerprint(
                dict(implementation)
            ),
        },
        "evidence_scope": {
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "performance_evaluation_performed": False,
            "formal_800_authorized": False,
        },
    }


def _flag_value(tokens: Sequence[str], name: str) -> str:
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == name:
            if index + 1 >= len(tokens):
                raise RuntimeError(f"parent wrapper lacks value for {name}")
            values.append(tokens[index + 1])
        elif token.startswith(f"{name}="):
            values.append(token.split("=", maxsplit=1)[1])
    if len(values) != 1:
        raise RuntimeError(f"parent wrapper must specify {name} exactly once")
    return values[0]


def _validate_wrapper_command(tokens: Sequence[str]) -> None:
    wrapper = _canonical_repo_file(
        TEMPERATURE_WRAPPER_REPO_PATH,
        name="temperature wrapper",
    )
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
        raise RuntimeError("formal run must be launched by the bound wrapper")
    if file_sha256(wrapper) != TEMPERATURE_WRAPPER_FILE_SHA256:
        raise RuntimeError("temperature wrapper file changed")
    expected = {
        "--gpu": str(int(FROZEN_VISIBLE_GPU)),
        "--pause-temp": str(FROZEN_PAUSE_TEMPERATURE_C),
        "--resume-temp": str(FROZEN_RESUME_TEMPERATURE_C),
    }
    for name, value in expected.items():
        if _flag_value(tokens, name) != value:
            raise RuntimeError(f"parent wrapper {name} differs from protocol")


def _read_parent_command() -> tuple[str, ...]:
    path = Path("/proc") / str(os.getppid()) / "cmdline"
    value = path.read_bytes()
    tokens = tuple(
        item.decode("utf-8", errors="strict")
        for item in value.split(b"\0")
        if item
    )
    if not tokens:
        raise RuntimeError("parent wrapper command is unavailable")
    return tokens


def _verify_runtime_contract() -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != FROZEN_VISIBLE_GPU:
        raise RuntimeError("formal run fixes CUDA_VISIBLE_DEVICES=0")
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != FROZEN_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "formal run fixes CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    _validate_wrapper_command(_read_parent_command())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "formal run requires exactly one visible CUDA device"
        )
    if torch.cuda.current_device() != 0:
        raise RuntimeError("formal run requires visible cuda:0")
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": FROZEN_DEVICE,
        "visible_device_count": torch.cuda.device_count(),
        "visible_device_index": torch.cuda.current_device(),
        "device_name": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "CUBLAS_WORKSPACE_CONFIG": os.environ[
            "CUBLAS_WORKSPACE_CONFIG"
        ],
        "temperature_wrapper_repo_path": TEMPERATURE_WRAPPER_REPO_PATH,
        "temperature_wrapper_file_sha256": (
            TEMPERATURE_WRAPPER_FILE_SHA256
        ),
        "pause_temperature_c": FROZEN_PAUSE_TEMPERATURE_C,
        "resume_temperature_c": FROZEN_RESUME_TEMPERATURE_C,
    }


def _claim_output(
    output: Path,
    *,
    attempt: Mapping[str, object],
) -> tuple[Path, Path]:
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
            raise RuntimeError("bounded artifact may not be a symbolic link")
        if not path.is_file() or path.name in {
            _INCOMPLETE,
            "COMPLETE.json",
        }:
            continue
        result[str(path.relative_to(root))] = file_sha256(path)
    return result


def _model_config(preflight: object) -> CoverageStateLevelSetConfig:
    population = preflight.population
    first = population.cache.raw_catalog.natural_records[0]
    return CoverageStateLevelSetConfig(
        feature_channels=int(first.feature.shape[1]),
        feature_stride=population.cache.raw_catalog.feature_stride,
        width=COVERAGE_STATE_BOUNDED_MODEL_WIDTH,
    )


def _model_config_payload(
    value: CoverageStateLevelSetConfig,
) -> dict[str, object]:
    return {
        "feature_channels": value.feature_channels,
        "feature_stride": value.feature_stride,
        "width": value.width,
        "normalization_epsilon_hex": (
            value.normalization_epsilon.hex()
        ),
        "field_amplitude_hex": value.field_amplitude.hex(),
        "initial_field_value_hex": value.initial_field_value.hex(),
        "field_policy": value.field_policy,
        "target_policy": value.target_policy,
        "output_policy": value.output_policy,
        "feature_policy": value.feature_policy,
        "numerical_policy": value.numerical_policy,
    }


def _device_memory_preflight(
    cache: object,
    model_config: CoverageStateLevelSetConfig,
) -> dict[str, object]:
    """Check free CUDA memory against packed payload plus a fixed reserve."""

    projected = prepare_coverage_state_device_cache(cache, device="cpu")
    projected.verify_unchanged(verify_content=True, verify_source=False)
    projected_payload = projected.resident_tensor_bytes
    projected_report = projected.memory_report()
    projected_fingerprint = projected.device_cache_fingerprint
    source_cache_fingerprint = projected.source_cache_fingerprint
    del projected
    gc.collect()

    model = CURELiteCoverageStateLevelSet(model_config)
    parameter_bytes = sum(
        value.numel() * value.element_size()
        for value in model.parameters()
    )
    buffer_bytes = sum(
        value.numel() * value.element_size()
        for value in model.buffers()
    )
    del model
    # Three retained checkpoints plus one gradient and two Adam moments for
    # the currently trained model.
    model_optimizer_bytes = 6 * parameter_bytes + 3 * buffer_bytes
    required = (
        projected_payload
        + model_optimizer_bytes
        + _ACTIVATION_RESERVE_BYTES
    )
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "exactly_one_visible_device": torch.cuda.device_count() == 1,
        "visible_cuda_zero": torch.cuda.current_device() == 0,
        "free_memory_meets_requirement": int(free_bytes) >= required,
        "total_memory_meets_requirement": int(total_bytes) >= required,
    }
    payload = {
        "schema_version": MEMORY_PREFLIGHT_SCHEMA,
        "device": FROZEN_DEVICE,
        "model_config": _model_config_payload(model_config),
        "source_cache_fingerprint": source_cache_fingerprint,
        "projected_cpu_pack_fingerprint": projected_fingerprint,
        "projected_target_device": FROZEN_DEVICE,
        "projected_device_cache": projected_report,
        "model_parameter_bytes": parameter_bytes,
        "model_buffer_bytes": buffer_bytes,
        "model_optimizer_retention_bytes": model_optimizer_bytes,
        "fixed_activation_reserve_bytes": _ACTIVATION_RESERVE_BYTES,
        "required_free_bytes": required,
        "observed_free_bytes": int(free_bytes),
        "observed_total_bytes": int(total_bytes),
        "checks": checks,
        "all_pass": all(checks.values()),
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    result = _fingerprinted(payload)
    if not result["all_pass"]:
        raise RuntimeError("device memory preflight did not pass")
    return result


def _checkpoint_backend() -> str:
    try:
        from safetensors.torch import save as _  # noqa: F401
    except ImportError:
        return "torch_tensor_only_state_dict"
    return "safetensors"


def _write_checkpoint_new(
    directory: Path,
    *,
    objective: str,
    model: CURELiteCoverageStateLevelSet,
) -> dict[str, object]:
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    backend = _checkpoint_backend()
    if backend == "safetensors":
        from safetensors.torch import load_file, save

        path = directory / f"{objective}.safetensors"
        encoded = save(state)
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        loaded = load_file(str(path), device="cpu")
    else:
        path = directory / f"{objective}.state_dict.pt"
        with path.open("xb") as handle:
            torch.save(
                state,
                handle,
                _use_new_zipfile_serialization=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        loaded = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    if (
        set(loaded) != set(state)
        or any(
            not torch.equal(loaded[name], state[name])
            for name in state
        )
    ):
        raise RuntimeError("checkpoint tensor-only round trip changed")
    payload = _fingerprinted(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "objective": objective,
            "repo_relative_path": str(path.relative_to(_ROOT)),
            "serialization": backend,
            "tensor_only_state_dict": True,
            "weights_only_roundtrip_verified": True,
            "checkpoint_file_sha256": file_sha256(path),
            "module_state_fingerprint": module_state_fingerprint(model),
            "state_keys": list(state),
            "dtype_policy": "preserve_trained_state",
            "device_policy": "cpu_checkpoint",
        }
    )
    _write_new_json(
        directory / f"{objective}.checkpoint.json",
        payload,
    )
    return payload


def _decision_payload(
    result: CoverageStateBoundedRunResult,
    checkpoints: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    passed = result.bounded_gate_passed
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "status": (
                "BOUNDED_CSLF_GATE_PASS"
                if passed
                else "BOUNDED_CSLF_GATE_FAIL"
            ),
            "bounded_gate_passed": passed,
            "failed_checks": list(result.failed_checks),
            "result_fingerprint": result.result_fingerprint,
            "checkpoint_receipt_fingerprints": {
                str(value["objective"]): str(
                    value["receipt_fingerprint"]
                )
                for value in checkpoints
            },
            "next_action": (
                "freeze_bounded_result_and_review_formal_800_prerequisites"
                if passed
                else "freeze_bounded_result_and_modify_cslf_model"
            ),
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "automatic_retry_allowed": False,
        }
    )


def validate_create_only() -> dict[str, object]:
    """Validate static bindings without claiming output or loading D_R tensors."""

    source_paths = _verify_frozen_sources()
    binding, protocol, _, _ = bind_coverage_state_real_dr_sources(
        **source_paths
    )
    implementation = _implementation_binding()
    config = _static_config_payload(
        source_paths=source_paths,
        implementation=implementation,
    )
    output_exists = OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink()
    return _fingerprinted(
        {
            "schema_version": VALIDATION_SCHEMA,
            "mode": "create_only_static_validation",
            "static_contract_valid": True,
            "source_binding_fingerprint": binding.binding_fingerprint,
            "observability_protocol_fingerprint": protocol.fingerprint,
            "config_fingerprint": stable_fingerprint(config),
            "implementation_fingerprint": stable_fingerprint(
                dict(implementation)
            ),
            "formal_output_exists": output_exists,
            "run_once_available": not output_exists,
            "output_claimed": False,
            "D_R_cached_tensor_payload_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "dataset_free_gate_executed": False,
            "authorization_created": False,
            "training_performed": False,
            "formal_800_authorized": False,
            "not_a_formal_result": True,
        }
    )


def _failure_payload(
    error: BaseException,
    *,
    attempt_fingerprint: str,
    artifact_files: Mapping[str, str],
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed_incomplete_attempt",
            "exception_type": type(error).__name__,
            "message": str(error),
            "time_utc": _utc_now(),
            "attempt_fingerprint": attempt_fingerprint,
            "artifact_files_before_failure": dict(artifact_files),
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def run_once() -> dict[str, object]:
    """Claim and execute the unique bounded run; never resume or retry."""

    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError(
            f"single-use bounded output already exists: {OUTPUT_PATH}"
        )
    source_paths = _verify_frozen_sources()
    implementation = _implementation_binding()
    runtime = _verify_runtime_contract()
    config = _fingerprinted(
        _static_config_payload(
            source_paths=source_paths,
            implementation=implementation,
        )
    )
    attempt = _fingerprinted(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "run_id": RUN_ID,
            "time_utc": _utc_now(),
            "output_repo_path": OUTPUT_REPO_PATH,
            "config_fingerprint": config["receipt_fingerprint"],
            "runtime": runtime,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    try:
        receipts, checkpoints_dir = _claim_output(
            OUTPUT_PATH,
            attempt=attempt,
        )
    except BaseException as error:
        if OUTPUT_PATH.is_dir() and (OUTPUT_PATH / "attempt.json").is_file():
            try:
                failure = _failure_payload(
                    error,
                    attempt_fingerprint=str(
                        attempt["receipt_fingerprint"]
                    ),
                    artifact_files=_artifact_hashes(OUTPUT_PATH),
                )
                _write_new_json(OUTPUT_PATH / "FAILURE.json", failure)
            except BaseException:
                pass
        raise
    try:
        _write_new_json(receipts / "config.json", config)

        real_inputs = build_coverage_state_real_dr_inputs(**source_paths)
        population = build_coverage_state_bounded_population(
            real_inputs.scalar_cache
        )
        preflight = prepare_coverage_state_bounded_preflight(population)
        exposure = coverage_state_schedule_exposure_report(
            population.cache,
            preflight.schedule,
        )
        input_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v15-bounded-400-inputs-v1"
                ),
                "real_D_R_inputs": real_inputs.canonical_payload(),
                "source_binding": (
                    real_inputs.source_binding.canonical_payload()
                ),
                "bounded_population": population.canonical_payload(),
                "population_fingerprint": (
                    population.population_fingerprint
                ),
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(receipts / "inputs.json", input_receipt)

        preflight_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v15-bounded-400-preflight-v1"
                ),
                "preflight": preflight.canonical_payload(),
                "schedule": preflight.schedule.canonical_payload(),
                "schedule_selections": [
                    value.canonical_payload()
                    for value in preflight.schedule.selections
                ],
                "exposure": exposure,
                "training_authorized": (
                    preflight.training_authorized
                ),
                "formal_800_authorized": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(
            receipts / "preflight.json",
            preflight_receipt,
        )
        if not preflight.training_authorized:
            raise PermissionError("bounded D_R preflight did not authorize")

        dataset_free = run_coverage_state_dataset_free_gate()
        dataset_free_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v15-bounded-400-dataset-free-v1"
                ),
                "dataset_free": dataset_free.canonical_payload(),
                "dataset_free_receipt_fingerprint": (
                    dataset_free.receipt_fingerprint
                ),
                "all_pass": dataset_free.all_pass,
                "formal_800_authorized": False,
            }
        )
        _write_new_json(
            receipts / "dataset_free.json",
            dataset_free_receipt,
        )
        authorization = prepare_coverage_state_bounded_run_authorization(
            preflight,
            dataset_free,
        )
        authorization_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v15-bounded-400-authorization-v1"
                ),
                "authorization": authorization.canonical_payload(),
                "authorization_fingerprint": (
                    authorization.authorization_fingerprint
                ),
                "runtime_implementation_fingerprint": (
                    stable_fingerprint(dict(implementation))
                ),
                "config_receipt_fingerprint": config[
                    "receipt_fingerprint"
                ],
                "model_config": _model_config_payload(
                    _model_config(preflight)
                ),
                "training_authorized": (
                    authorization.training_authorized
                ),
                "formal_800_authorized": False,
            }
        )
        _write_new_json(
            receipts / "authorization.json",
            authorization_receipt,
        )
        if not authorization.training_authorized:
            raise PermissionError("bounded authorization did not pass")

        model_config = _model_config(preflight)
        authorization.verify_model_config(model_config)
        memory = _device_memory_preflight(
            population.cache,
            model_config,
        )
        _write_new_json(
            receipts / "device_memory_preflight.json",
            memory,
        )

        result = run_coverage_state_bounded_400(
            authorization,
            model_config,
            device=FROZEN_DEVICE,
        )
        checkpoint_receipts = tuple(
            _write_checkpoint_new(
                checkpoints_dir,
                objective=name,
                model=model,
            )
            for name, model in result.training.models
        )
        training_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v15-bounded-400-training-v1"
                ),
                "training": result.training.canonical_payload(),
                "training_result_fingerprint": (
                    result.training.result_fingerprint
                ),
                "checkpoint_receipt_fingerprints": {
                    str(value["objective"]): str(
                        value["receipt_fingerprint"]
                    )
                    for value in checkpoint_receipts
                },
                "formal_training_performed": False,
                "bounded_training_performed": True,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(
            receipts / "training.json",
            training_receipt,
        )
        zero_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v15-bounded-400-zero-level-v1"
                ),
                "threshold": 0.0,
                "threshold_search_performed": False,
                "diagnostics": {
                    name: value.canonical_payload()
                    for name, value in result.diagnostics
                },
                "all_bounded_gates_passed": all(
                    value.bounded_gate_passed
                    for _, value in result.diagnostics
                ),
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(
            receipts / "zero_level.json",
            zero_receipt,
        )
        bounded_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v15-bounded-400-result-v1"
                ),
                "result": result.canonical_payload(),
                "result_fingerprint": result.result_fingerprint,
            }
        )
        _write_new_json(
            receipts / "bounded_result.json",
            bounded_receipt,
        )
        decision = _decision_payload(result, checkpoint_receipts)
        _write_new_json(receipts / "decision.json", decision)

        # Recheck all mutable bindings after the last model read.
        real_inputs.verify_unchanged()
        authorization.verify_unchanged()
        result.verify_unchanged()
        if _implementation_binding() != implementation:
            raise RuntimeError(
                "bounded implementation changed during execution"
            )
        if _verify_frozen_sources() != source_paths:
            raise RuntimeError("frozen D_R source paths changed")

        artifacts = _artifact_hashes(OUTPUT_PATH)
        complete = _fingerprinted(
            {
                "schema_version": RUN_SCHEMA,
                "status": "complete",
                "run_id": RUN_ID,
                "decision": decision["status"],
                "bounded_gate_passed": result.bounded_gate_passed,
                "formal_800_authorized": False,
                "full_CURE_authorized": False,
                "cross_backbone_authorized": False,
                "performance_claim_supported": False,
                "split": "D_R",
                "runtime_splits": ["D_R"],
                "config_fingerprint": config["receipt_fingerprint"],
                "input_receipt_fingerprint": (
                    input_receipt["receipt_fingerprint"]
                ),
                "preflight_receipt_fingerprint": (
                    preflight_receipt["receipt_fingerprint"]
                ),
                "dataset_free_receipt_fingerprint": (
                    dataset_free_receipt["receipt_fingerprint"]
                ),
                "authorization_receipt_fingerprint": (
                    authorization_receipt["receipt_fingerprint"]
                ),
                "training_receipt_fingerprint": (
                    training_receipt["receipt_fingerprint"]
                ),
                "zero_level_receipt_fingerprint": (
                    zero_receipt["receipt_fingerprint"]
                ),
                "decision_fingerprint": decision[
                    "receipt_fingerprint"
                ],
                "artifact_files": artifacts,
                "artifact_file_count": len(artifacts),
                "single_attempt": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "calibration_performed": False,
                "performance_evaluation_performed": False,
            },
            field="complete_fingerprint",
        )
        _write_new_json(OUTPUT_PATH / "COMPLETE.json", complete)
        (OUTPUT_PATH / _INCOMPLETE).unlink()
        return {
            "output": str(OUTPUT_PATH),
            "decision": decision["status"],
            "bounded_gate_passed": result.bounded_gate_passed,
            "complete_fingerprint": complete["complete_fingerprint"],
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    except BaseException as error:
        try:
            failure = _failure_payload(
                error,
                attempt_fingerprint=str(
                    attempt["receipt_fingerprint"]
                ),
                artifact_files=_artifact_hashes(OUTPUT_PATH),
            )
            _write_new_json(OUTPUT_PATH / "FAILURE.json", failure)
        except BaseException:
            # Preserve the original exception and every already-created byte.
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-create-only",
        action="store_true",
        help=(
            "validate static bindings only; never claim output, load cached "
            "D_R tensors, authorize, or train"
        ),
    )
    mode.add_argument(
        "--run-once",
        action="store_true",
        help="consume the unique wrapper-controlled bounded-400 attempt",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
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
            indent=2,
        )
    )
    if args.run_once and result["bounded_gate_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
