#!/usr/bin/env python3
"""Validate or execute the unique PPCE/SORR bounded-400 protocol.

``--validate-create-only`` verifies frozen bindings without loading D_R
tensors or claiming output.  ``--run-once`` is the explicit, wrapper-controlled
single-use execution path that writes the complete bounded receipt graph.
"""

from __future__ import annotations

import argparse
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
from cure_lite.coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
    CURELitePhasePreservingCoverageStateLevelSet,
    CoverageStatePhasePreservingConfig,
)
from cure_lite.experiment.coverage_state_dataset_free import (
    COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT,
    run_coverage_state_phase_preserving_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_ppce_bounded_runner import (
    COVERAGE_STATE_V15B_PARENT_COMPLETE_FINGERPRINT,
    COVERAGE_STATE_V15B_PARENT_COMPLETE_SHA256,
    COVERAGE_STATE_V15B_PARENT_SOURCE_ARCHIVE_SHA256,
    COVERAGE_STATE_V15B_PARENT_SOURCE_MANIFEST_SHA256,
    CoverageStatePPCEBoundedRunAuthorization,
    CoverageStatePPCEBoundedRunResult,
    expected_coverage_state_ppce_config,
    prepare_coverage_state_ppce_bounded_run_authorization,
    run_coverage_state_ppce_support_oriented_bounded_400,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite.coverage_state_schedule import (
    coverage_state_schedule_exposure_report,
)
from cure_lite.frozen_base import module_state_fingerprint
from cure_lite.train.coverage_state_fused_step import (
    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
    coverage_state_pair_objective_policy,
)
from tools import (
    run_coverage_state_cslf_support_oriented_bounded_400 as _v15b_cli,
)


_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "cure_lite_cslf_v16_ppce_support_oriented_bounded_400_r1"
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = _ROOT / OUTPUT_REPO_PATH
RUN_SCHEMA = (
    "cure-lite-cslf-v16-ppce-support-oriented-bounded-400-run-v1"
)
VALIDATION_SCHEMA = (
    "cure-lite-cslf-v16-ppce-support-oriented-bounded-400-"
    "create-only-validation-v1"
)
ATTEMPT_SCHEMA = (
    "cure-lite-cslf-v16-ppce-support-oriented-bounded-400-attempt-v1"
)
FAILURE_SCHEMA = (
    "cure-lite-cslf-v16-ppce-support-oriented-bounded-400-failure-v1"
)
CHECKPOINT_SCHEMA = (
    "cure-lite-cslf-v16-ppce-support-oriented-bounded-400-checkpoint-v1"
)
DECISION_SCHEMA = (
    "cure-lite-cslf-v16-ppce-support-oriented-bounded-400-decision-v1"
)
FROZEN_DEVICE = _v15b_cli.FROZEN_DEVICE
FROZEN_VISIBLE_GPU = _v15b_cli.FROZEN_VISIBLE_GPU
FROZEN_CUBLAS_WORKSPACE_CONFIG = (
    _v15b_cli.FROZEN_CUBLAS_WORKSPACE_CONFIG
)
FROZEN_PAUSE_TEMPERATURE_C = (
    _v15b_cli.FROZEN_PAUSE_TEMPERATURE_C
)
FROZEN_RESUME_TEMPERATURE_C = (
    _v15b_cli.FROZEN_RESUME_TEMPERATURE_C
)
FROZEN_CHECKPOINT_SERIALIZATION = "safetensors"
_ACTIVATION_RESERVE_BYTES = 2 * 1024**3
_INCOMPLETE = ".incomplete"
PARENT_V15B_RUN_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    "cure_lite_cslf_v15b_support_oriented_bounded_400_r1"
)
PARENT_V15B_SOURCE_MANIFEST_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_cslf_v15b_support_oriented_bounded_400_13cc94f4f514.json"
)
PARENT_V15B_SOURCE_ARCHIVE_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_cslf_v15b_support_oriented_bounded_400_13cc94f4f514.tar"
)
PARENT_V15B_ARTIFACT_FILE_COUNT = 17
FROZEN_FEATURE_CHANNELS = 64
FROZEN_FEATURE_STRIDE = 4
FROZEN_MODEL_WIDTH = 32
FROZEN_PARAMETER_COUNT = 23856
FROZEN_SEED = 42
FROZEN_EPOCHS = 10
FROZEN_STEPS_PER_EPOCH = 40
FROZEN_UPDATES_PER_OBJECTIVE = 400
FROZEN_ARTIFACT_FILE_COUNT = 17
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
    result: dict[str, Path] = {}
    for name, relative, expected_sha256 in FROZEN_REAL_DR_INPUTS:
        path = _canonical_repo_file(relative, name=name)
        if file_sha256(path) != expected_sha256:
            raise RuntimeError(f"frozen real D_R input changed: {name}")
        result[name] = path
    return result


def _verify_parent_v15b_closure() -> dict[str, object]:
    parent = (_ROOT / PARENT_V15B_RUN_REPO_PATH).resolve(strict=True)
    if (
        parent
        != Path(os.path.abspath(_ROOT / PARENT_V15B_RUN_REPO_PATH))
        or not parent.is_dir()
        or parent.is_symlink()
    ):
        raise RuntimeError("parent v15B result directory changed")
    complete_path = _canonical_repo_file(
        f"{PARENT_V15B_RUN_REPO_PATH}/COMPLETE.json",
        name="parent v15B COMPLETE",
    )
    if (
        file_sha256(complete_path)
        != COVERAGE_STATE_V15B_PARENT_COMPLETE_SHA256
    ):
        raise RuntimeError("parent v15B COMPLETE changed")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        complete.get("complete_fingerprint")
        != COVERAGE_STATE_V15B_PARENT_COMPLETE_FINGERPRINT
        or complete.get("status") != "complete"
        or complete.get("decision")
        != "BOUNDED_SUPPORT_ORIENTED_CSLF_GATE_FAIL"
        or complete.get("bounded_gate_passed") is not False
    ):
        raise RuntimeError("parent v15B decision binding changed")
    artifacts = complete.get("artifact_files")
    if (
        not isinstance(artifacts, dict)
        or len(artifacts) != PARENT_V15B_ARTIFACT_FILE_COUNT
    ):
        raise RuntimeError("parent v15B artifact map is missing")
    for relative, expected_sha256 in artifacts.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise RuntimeError("parent v15B artifact map is invalid")
        path = parent / relative
        resolved = path.resolve(strict=True)
        if (
            resolved != Path(os.path.abspath(path))
            or not resolved.is_file()
            or resolved.is_symlink()
            or file_sha256(resolved) != expected_sha256
        ):
            raise RuntimeError(
                f"parent v15B artifact changed: {relative}"
            )
    actual = {
        str(path.relative_to(parent))
        for path in parent.rglob("*")
        if path.is_file() and path.name != "COMPLETE.json"
    }
    if actual != set(artifacts):
        raise RuntimeError("parent v15B artifact population changed")
    manifest = _canonical_repo_file(
        PARENT_V15B_SOURCE_MANIFEST_REPO_PATH,
        name="parent v15B source manifest",
    )
    archive = _canonical_repo_file(
        PARENT_V15B_SOURCE_ARCHIVE_REPO_PATH,
        name="parent v15B source archive",
    )
    if (
        file_sha256(manifest)
        != COVERAGE_STATE_V15B_PARENT_SOURCE_MANIFEST_SHA256
        or file_sha256(archive)
        != COVERAGE_STATE_V15B_PARENT_SOURCE_ARCHIVE_SHA256
    ):
        raise RuntimeError("parent v15B source closure changed")
    return {
        "run_repo_path": PARENT_V15B_RUN_REPO_PATH,
        "complete_fingerprint": (
            COVERAGE_STATE_V15B_PARENT_COMPLETE_FINGERPRINT
        ),
        "complete_sha256": COVERAGE_STATE_V15B_PARENT_COMPLETE_SHA256,
        "artifact_file_count": len(artifacts),
        "source_manifest_repo_path": (
            PARENT_V15B_SOURCE_MANIFEST_REPO_PATH
        ),
        "source_manifest_sha256": (
            COVERAGE_STATE_V15B_PARENT_SOURCE_MANIFEST_SHA256
        ),
        "source_archive_repo_path": (
            PARENT_V15B_SOURCE_ARCHIVE_REPO_PATH
        ),
        "source_archive_sha256": (
            COVERAGE_STATE_V15B_PARENT_SOURCE_ARCHIVE_SHA256
        ),
    }


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    paths = tuple(
        sorted(
            str(path.relative_to(_ROOT))
            for path in (_ROOT / "cure_lite").rglob("*.py")
            if path.is_file()
        )
    ) + (
        "tools/run_coverage_state_cslf_ppce_support_oriented_bounded_400.py",
        "tools/run_coverage_state_cslf_support_oriented_bounded_400.py",
        "tools/run_with_gpu_temperature_control.py",
    )
    return tuple(
        (
            relative,
            file_sha256(
                _canonical_repo_file(
                    relative,
                    name="implementation file",
                )
            ),
        )
        for relative in paths
    )


def _static_config_payload(
    *,
    source_paths: Mapping[str, Path],
    implementation: tuple[tuple[str, str], ...],
    dataset_free_receipt_fingerprint: str,
) -> dict[str, object]:
    if (
        not isinstance(dataset_free_receipt_fingerprint, str)
        or len(dataset_free_receipt_fingerprint) != 64
        or dataset_free_receipt_fingerprint
        != COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT
    ):
        raise ValueError(
            "dataset-free receipt fingerprint differs from the frozen "
            "PPCE gate"
        )
    config = CoverageStatePhasePreservingConfig(
        feature_channels=FROZEN_FEATURE_CHANNELS,
        feature_stride=FROZEN_FEATURE_STRIDE,
        width=FROZEN_MODEL_WIDTH,
    )
    model = CURELitePhasePreservingCoverageStateLevelSet(config)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if (
        parameter_count != config.expected_parameter_count
        or parameter_count != FROZEN_PARAMETER_COUNT
    ):
        raise RuntimeError("frozen PPCE parameter count changed")
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
            "class": (
                "CURELitePhasePreservingCoverageStateLevelSet"
            ),
            "coverage_policy": CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
            "feature_channels": config.feature_channels,
            "feature_stride": config.feature_stride,
            "phase_occupancy_channels": (
                config.phase_occupancy_channels
            ),
            "width": config.width,
            "parameter_count": parameter_count,
            "field_threshold": 0.0,
            "threshold_search_performed": False,
            "objective_suite": [
                value.value
                for value in (
                    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
                )
            ],
            "candidate_objective": (
                "support_oriented_response_joint"
            ),
            "candidate_objective_policy": (
                coverage_state_pair_objective_policy(
                    "support_oriented_response_joint"
                )
            ),
        },
        "budget": {
            "seed": FROZEN_SEED,
            "epochs": FROZEN_EPOCHS,
            "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
            "updates_per_objective": FROZEN_UPDATES_PER_OBJECTIVE,
            "objectives": 3,
        },
        "execution": {
            "device": FROZEN_DEVICE,
            "CUDA_VISIBLE_DEVICES": FROZEN_VISIBLE_GPU,
            "CUBLAS_WORKSPACE_CONFIG": (
                FROZEN_CUBLAS_WORKSPACE_CONFIG
            ),
            "pause_temperature_c": (
                FROZEN_PAUSE_TEMPERATURE_C
            ),
            "resume_temperature_c": (
                FROZEN_RESUME_TEMPERATURE_C
            ),
            "checkpoint_serialization": (
                FROZEN_CHECKPOINT_SERIALIZATION
            ),
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        },
        "dataset_free_gate": {
            "binding_mode": "actual_runtime_receipt_fingerprint",
            "receipt_fingerprint": (
                dataset_free_receipt_fingerprint
            ),
        },
        "parent_v15b_negative_result": (
            _verify_parent_v15b_closure()
        ),
        "implementation": {
            "files": dict(implementation),
            "implementation_fingerprint": stable_fingerprint(
                dict(implementation)
            ),
        },
        "evidence_scope": {
            "D_R_cached_tensor_payload_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
        },
    }


def _device_memory_preflight(
    cache: object,
    model_config: CoverageStatePhasePreservingConfig,
) -> dict[str, object]:
    """Account for one packed cache and three retained PPCE models."""

    projected = prepare_coverage_state_device_cache(
        cache,
        device="cpu",
    )
    projected.verify_unchanged(verify_content=True, verify_source=False)
    projected_payload = projected.resident_tensor_bytes
    projected_report = projected.memory_report()
    projected_fingerprint = projected.device_cache_fingerprint
    source_cache_fingerprint = projected.source_cache_fingerprint
    del projected
    gc.collect()
    model = CURELitePhasePreservingCoverageStateLevelSet(model_config)
    parameter_bytes = sum(
        value.numel() * value.element_size()
        for value in model.parameters()
    )
    buffer_bytes = sum(
        value.numel() * value.element_size()
        for value in model.buffers()
    )
    del model
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
    result = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-cslf-v16-ppce-support-oriented-"
                "bounded-400-device-memory-preflight-v1"
            ),
            "device": FROZEN_DEVICE,
            "model_class": (
                "CURELitePhasePreservingCoverageStateLevelSet"
            ),
            "model_parameter_count": (
                model_config.expected_parameter_count
            ),
            "source_cache_fingerprint": source_cache_fingerprint,
            "projected_cpu_pack_fingerprint": projected_fingerprint,
            "projected_device_cache": projected_report,
            "model_parameter_bytes": parameter_bytes,
            "model_buffer_bytes": buffer_bytes,
            "model_optimizer_retention_bytes": model_optimizer_bytes,
            "fixed_activation_reserve_bytes": (
                _ACTIVATION_RESERVE_BYTES
            ),
            "required_free_bytes": required,
            "observed_free_bytes": int(free_bytes),
            "observed_total_bytes": int(total_bytes),
            "checks": checks,
            "all_pass": all(checks.values()),
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
        }
    )
    if not result["all_pass"]:
        raise RuntimeError("PPCE device memory preflight did not pass")
    return result


def _write_checkpoint_new(
    directory: Path,
    *,
    objective: str,
    objective_policy: str,
    model: CURELitePhasePreservingCoverageStateLevelSet,
) -> dict[str, object]:
    if type(model) is not CURELitePhasePreservingCoverageStateLevelSet:
        raise TypeError("PPCE checkpoint requires the exact model class")
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    from safetensors.torch import load_file, save

    path = directory / f"{objective}.safetensors"
    with path.open("xb") as handle:
        handle.write(save(state))
        handle.flush()
        os.fsync(handle.fileno())
    loaded = load_file(str(path), device="cpu")
    if (
        set(loaded) != set(state)
        or any(
            not torch.equal(loaded[name], state[name])
            for name in state
        )
    ):
        raise RuntimeError("PPCE checkpoint roundtrip changed")
    result = _fingerprinted(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "objective": objective,
            "objective_policy": objective_policy,
            "model_class": (
                "CURELitePhasePreservingCoverageStateLevelSet"
            ),
            "model_config": {
                "feature_channels": model.config.feature_channels,
                "feature_stride": model.config.feature_stride,
                "width": model.config.width,
                "coverage_policy": model.config.coverage_policy,
                "parameter_count": sum(
                    value.numel() for value in model.parameters()
                ),
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
    _write_new_json(
        directory / f"{objective}.checkpoint.json",
        result,
    )
    return result


def _zero_level_payload(
    result: CoverageStatePPCEBoundedRunResult,
    authorization: CoverageStatePPCEBoundedRunAuthorization,
) -> dict[str, object]:
    candidate_gate = dict(result.checks)[
        "candidate_original_zero_level_gates"
    ]
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-cslf-v16-ppce-support-oriented-"
                "bounded-400-zero-level-v1"
            ),
            "input_representation": "phase_preserving",
            "threshold": 0.0,
            "threshold_search_performed": False,
            "diagnostics": {
                name: value.canonical_payload()
                for name, value in result.diagnostics
            },
            "candidate_objective": authorization.candidate_objective,
            "candidate_bounded_gate_passed": candidate_gate,
            "control_bounded_gate_outcomes": {
                name: value.bounded_gate_passed
                for name, value in result.diagnostics
                if name != authorization.candidate_objective
            },
            "control_outcomes_are_not_candidate_gates": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def _decision_payload(
    result: CoverageStatePPCEBoundedRunResult,
    checkpoints: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    passed = result.bounded_gate_passed
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "status": (
                "BOUNDED_PPCE_SUPPORT_ORIENTED_CSLF_GATE_PASS"
                if passed
                else "BOUNDED_PPCE_SUPPORT_ORIENTED_CSLF_GATE_FAIL"
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
            "candidate_objective": (
                result.authorization.candidate_objective
            ),
            "candidate_gate_passed": dict(result.checks)[
                "candidate_original_zero_level_gates"
            ],
            "control_outcomes_are_not_candidate_gates": True,
            "next_action": (
                "freeze_ppce_result_and_review_confirmation_prerequisites"
                if passed
                else (
                    "freeze_ppce_negative_result_and_stop_current_"
                    "absolute_zero_level_route"
                )
            ),
            "parent_v15b_complete_fingerprint": (
                result.authorization.parent_v15b_complete_fingerprint
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
            "attempt_fingerprint": attempt_fingerprint,
            "artifact_files_before_failure": dict(artifact_files),
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def validate_create_only() -> dict[str, object]:
    """Validate protocol construction without claiming or training D_R."""

    source_paths = _verify_frozen_sources()
    parent = _verify_parent_v15b_closure()
    dataset_free = (
        run_coverage_state_phase_preserving_dataset_free_gate()
    )
    if not dataset_free.all_pass:
        raise RuntimeError("PPCE dataset-free gate did not pass")
    if (
        dataset_free.receipt_fingerprint
        != COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT
    ):
        raise RuntimeError("frozen PPCE dataset-free receipt changed")
    implementation = _implementation_binding()
    config = _static_config_payload(
        source_paths=source_paths,
        implementation=implementation,
        dataset_free_receipt_fingerprint=(
            dataset_free.receipt_fingerprint
        ),
    )
    output_exists = OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink()
    return _fingerprinted(
        {
            "schema_version": VALIDATION_SCHEMA,
            "mode": "create_only_protocol_validation",
            "static_contract_valid": True,
            "config_fingerprint": stable_fingerprint(config),
            "implementation_fingerprint": stable_fingerprint(
                dict(implementation)
            ),
            "dataset_free_receipt_fingerprint": (
                dataset_free.receipt_fingerprint
            ),
            "dataset_free_gate_passed": dataset_free.all_pass,
            "parent_v15b_negative_result": parent,
            "bounded_output_exists": output_exists,
            "run_once_implemented": True,
            "output_claimed": False,
            "D_R_cached_tensor_payload_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "authorization_created": False,
            "training_performed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
            "not_a_formal_result": True,
        }
    )


def run_once() -> dict[str, object]:
    """Execute the unique wrapper-controlled D_R attempt; never resume."""

    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError(
            f"single-use bounded output already exists: {OUTPUT_PATH}"
        )
    source_paths = _verify_frozen_sources()
    parent = _verify_parent_v15b_closure()
    implementation = _implementation_binding()
    runtime = _v15b_cli._verify_runtime_contract()
    dataset_free = (
        run_coverage_state_phase_preserving_dataset_free_gate()
    )
    if not dataset_free.all_pass:
        raise PermissionError("PPCE dataset-free gate did not authorize")
    if (
        dataset_free.receipt_fingerprint
        != COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT
    ):
        raise RuntimeError("frozen PPCE dataset-free receipt changed")
    config = _fingerprinted(
        _static_config_payload(
            source_paths=source_paths,
            implementation=implementation,
            dataset_free_receipt_fingerprint=(
                dataset_free.receipt_fingerprint
            ),
        )
    )
    attempt = _fingerprinted(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "run_id": RUN_ID,
            "output_repo_path": OUTPUT_REPO_PATH,
            "config_fingerprint": config["receipt_fingerprint"],
            "runtime": runtime,
            "parent_v15b_negative_result": parent,
            "dataset_free_receipt_fingerprint": (
                dataset_free.receipt_fingerprint
            ),
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    try:
        receipts, checkpoints_dir = _v15b_cli._claim_output(
            OUTPUT_PATH,
            attempt=attempt,
        )
    except BaseException as error:
        if (
            OUTPUT_PATH.is_dir()
            and (OUTPUT_PATH / "attempt.json").is_file()
        ):
            try:
                _write_new_json(
                    OUTPUT_PATH / "FAILURE.json",
                    _failure_payload(
                        error,
                        attempt_fingerprint=str(
                            attempt["receipt_fingerprint"]
                        ),
                        artifact_files=_v15b_cli._artifact_hashes(
                            OUTPUT_PATH
                        ),
                    ),
                )
            except BaseException:
                pass
        raise
    try:
        _write_new_json(receipts / "config.json", config)
        real_inputs = build_coverage_state_real_dr_inputs(
            **source_paths
        )
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
                    "cure-lite-cslf-v16-ppce-support-oriented-"
                    "bounded-400-inputs-v1"
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
                    "cure-lite-cslf-v16-ppce-support-oriented-"
                    "bounded-400-preflight-v1"
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
        dataset_free_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v16-ppce-support-oriented-"
                    "bounded-400-dataset-free-v1"
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
        authorization = (
            prepare_coverage_state_ppce_bounded_run_authorization(
                preflight,
                dataset_free,
            )
        )
        model_config = expected_coverage_state_ppce_config(preflight)
        if (
            model_config.feature_channels != FROZEN_FEATURE_CHANNELS
            or model_config.feature_stride != FROZEN_FEATURE_STRIDE
            or model_config.width != FROZEN_MODEL_WIDTH
            or model_config.expected_parameter_count
            != FROZEN_PARAMETER_COUNT
        ):
            raise RuntimeError("real D_R PPCE model config changed")
        authorization_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v16-ppce-support-oriented-"
                    "bounded-400-authorization-v1"
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
                "training_authorized": (
                    authorization.training_authorized
                ),
                "formal_800_authorized": False,
                "parent_v15b_negative_result": parent,
            }
        )
        _write_new_json(
            receipts / "authorization.json",
            authorization_receipt,
        )
        if not authorization.training_authorized:
            raise PermissionError("PPCE authorization did not pass")
        memory = _device_memory_preflight(
            population.cache,
            model_config,
        )
        _write_new_json(
            receipts / "device_memory_preflight.json",
            memory,
        )
        result = (
            run_coverage_state_ppce_support_oriented_bounded_400(
                authorization,
                model_config,
                device=FROZEN_DEVICE,
            )
        )
        checkpoint_receipts = tuple(
            _write_checkpoint_new(
                checkpoints_dir,
                objective=result_row.objective,
                objective_policy=result_row.objective_policy,
                model=model,
            )
            for result_row, (_, model) in zip(
                result.training.results,
                result.training.models,
                strict=True,
            )
        )
        training_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v16-ppce-support-oriented-"
                    "bounded-400-training-v1"
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
                "all_models_exact_ppce_class": True,
                "all_models_parameter_count": (
                    FROZEN_PARAMETER_COUNT
                ),
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(
            receipts / "training.json",
            training_receipt,
        )
        zero_receipt = _zero_level_payload(result, authorization)
        _write_new_json(
            receipts / "zero_level.json",
            zero_receipt,
        )
        bounded_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cslf-v16-ppce-support-oriented-"
                    "bounded-400-result-v1"
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

        real_inputs.verify_unchanged()
        authorization.verify_unchanged()
        result.verify_unchanged()
        if _implementation_binding() != implementation:
            raise RuntimeError(
                "PPCE implementation changed during execution"
            )
        if _verify_frozen_sources() != source_paths:
            raise RuntimeError("frozen D_R source paths changed")
        if _verify_parent_v15b_closure() != parent:
            raise RuntimeError("parent v15B closure changed")
        replay_dataset_free = (
            run_coverage_state_phase_preserving_dataset_free_gate()
        )
        if (
            replay_dataset_free.receipt_fingerprint
            != dataset_free.receipt_fingerprint
        ):
            raise RuntimeError("PPCE dataset-free receipt changed")

        artifacts = _v15b_cli._artifact_hashes(OUTPUT_PATH)
        if len(artifacts) != FROZEN_ARTIFACT_FILE_COUNT:
            raise RuntimeError(
                "PPCE terminal artifact population is incomplete"
            )
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
                "parent_v15b_negative_result": parent,
                **_v15b_cli._complete_receipt_fingerprints(
                    config=config,
                    input_receipt=input_receipt,
                    preflight_receipt=preflight_receipt,
                    dataset_free_receipt=dataset_free_receipt,
                    authorization_receipt=authorization_receipt,
                    training_receipt=training_receipt,
                    zero_receipt=zero_receipt,
                    bounded_receipt=bounded_receipt,
                    decision=decision,
                ),
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
                artifact_files=_v15b_cli._artifact_hashes(
                    OUTPUT_PATH
                ),
            )
            _write_new_json(OUTPUT_PATH / "FAILURE.json", failure)
        except BaseException:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-create-only",
        action="store_true",
        help="validate without claiming output or loading D_R tensors",
    )
    mode.add_argument(
        "--run-once",
        action="store_true",
        help="consume the unique wrapper-controlled bounded-400 attempt",
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
    if args.run_once and result["bounded_gate_passed"] is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
