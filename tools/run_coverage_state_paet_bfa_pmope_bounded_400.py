#!/usr/bin/env python3
"""Validate or execute the unique v21 PAET-BFA bounded-400 attempt.

``--validate-create-only`` checks generated/static prerequisites without
claiming output, loading cached ``D_R`` tensors, constructing an optimizer,
or running the real-``D_R`` gate.  ``--run-once`` consumes the fixed
seed-42, 10-by-40 attempt.  There is no seed, budget, output, retry, or
resume option and the command never reads ``D_V`` or ``D_T``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CSLF_PAET_EQUATION_POLICY,
    CSLF_PAET_FIELD_POLICY,
    CSLF_PAET_FLIP_POLICY,
    CSLF_PAET_TRANSPORT_POLICY,
    PAET_INPUT_REPRESENTATION,
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.coverage_state_schedule import (
    coverage_state_schedule_exposure_report,
)
from cure_lite.coverage_state_sobolev import CSLF_PMOPE_POLICY
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_paet_bounded_runner import (
    COVERAGE_STATE_PAET_HISTORICAL_CACHE_FINGERPRINT,
    COVERAGE_STATE_PAET_HISTORICAL_DEVICE_CACHE_FINGERPRINT,
    COVERAGE_STATE_PAET_HISTORICAL_OPTIMIZER_FINGERPRINT,
    COVERAGE_STATE_PAET_MEMORY_RATIO_LIMIT,
    COVERAGE_STATE_PAET_OFFICIAL_RUN_ID,
    COVERAGE_STATE_PAET_STEP_TIME_RATIO_LIMIT,
    COVERAGE_STATE_PAET_V20_RESOURCE_COMPARISON_STATUS,
    _current_implementation_binding,
    expected_coverage_state_paet_config,
    prepare_coverage_state_paet_bounded_run_authorization,
    run_coverage_state_paet_bfa_pmope_bounded_400,
    verify_repository_coverage_state_bfa_v20_reference,
)
from cure_lite.experiment.coverage_state_paet_dataset_free import (
    COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_PAET_FORMAL_WIDTH,
    COVERAGE_STATE_PAET_MARGIN,
    run_coverage_state_paet_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_paet_dr_gate import (
    run_coverage_state_paet_dr_gate,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite.frozen_base import module_state_fingerprint
from tools import run_coverage_state_cmif_pmope_bounded_400 as _pmope_cli
from tools import (
    run_coverage_state_cslf_support_oriented_bounded_400 as _v15b_cli,
)


_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = COVERAGE_STATE_PAET_OFFICIAL_RUN_ID
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = _ROOT / OUTPUT_REPO_PATH
RUN_SCHEMA = "cure-lite-paet-bfa-v21-pmope-bounded-400-run-v1"
VALIDATION_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-bounded-400-"
    "create-only-validation-v1"
)
ATTEMPT_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-bounded-400-attempt-v1"
)
FAILURE_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-bounded-400-failure-v1"
)
CHECKPOINT_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-bounded-400-checkpoint-v1"
)
DECISION_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-bounded-400-decision-v1"
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
FROZEN_FEATURE_CHANNELS = COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS
FROZEN_FEATURE_STRIDE = COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
FROZEN_MODEL_WIDTH = COVERAGE_STATE_PAET_FORMAL_WIDTH
FROZEN_PARAMETER_COUNT = COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
FROZEN_SEED = 42
FROZEN_EPOCHS = 10
FROZEN_STEPS_PER_EPOCH = 40
FROZEN_UPDATES_PER_OBJECTIVE = 400
FROZEN_ARTIFACT_FILE_COUNT = 17
FROZEN_DR_GATE_STOP_ARTIFACT_FILE_COUNT = 8
FROZEN_REAL_DR_INPUTS = _pmope_cli.FROZEN_REAL_DR_INPUTS
_MINIMUM_ACTIVATION_ADMISSION_RESERVE_BYTES = 2 * 1024**3
_FROZEN_RESIDENT_DEVICE_CACHE_BYTES = 205_521_408
_INCOMPLETE = ".incomplete"

_fingerprinted = _pmope_cli._fingerprinted
_write_new_json = _pmope_cli._write_new_json


def _assert_run_id(value: object, *, layer: str) -> str:
    if value != RUN_ID:
        raise RuntimeError(f"{layer} run_id differs from {RUN_ID}")
    return RUN_ID


def _verify_frozen_sources() -> dict[str, Path]:
    return _pmope_cli._verify_frozen_sources()


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    """Bind the PAET closure, this entrypoint, and execution wrapper."""

    core = dict(_current_implementation_binding())
    extras = (
        "tools/run_coverage_state_paet_bfa_pmope_bounded_400.py",
        "tools/run_coverage_state_cmif_pmope_bounded_400.py",
        (
            "tools/"
            "run_coverage_state_cslf_ppce_support_oriented_bounded_400.py"
        ),
        "tools/run_coverage_state_cslf_support_oriented_bounded_400.py",
        "tools/run_with_gpu_temperature_control.py",
    )
    for relative in extras:
        path = _ROOT / relative
        absolute = Path(os.path.abspath(path))
        if (
            path.is_symlink()
            or path.resolve(strict=True) != absolute
            or not absolute.is_file()
        ):
            raise RuntimeError(
                f"PAET bounded implementation path changed: {relative}"
            )
        core[relative] = file_sha256(absolute)
    return tuple(sorted(core.items()))


def _static_config_payload(
    *,
    source_paths: Mapping[str, Path],
    implementation: tuple[tuple[str, str], ...],
    dataset_free_receipt_fingerprint: str,
    sealed_v20_reference_fingerprint: str,
) -> dict[str, object]:
    """Construct the immutable singleton seed-42 PAET configuration."""

    if (
        len(dataset_free_receipt_fingerprint) != 64
        or len(sealed_v20_reference_fingerprint) != 64
    ):
        raise ValueError("PAET prerequisite fingerprint changed")
    config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=FROZEN_FEATURE_CHANNELS,
        feature_stride=FROZEN_FEATURE_STRIDE,
        width=FROZEN_MODEL_WIDTH,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FROZEN_SEED)
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(config)
        parameter_count = sum(
            value.numel() for value in model.parameters()
        )
    if (
        parameter_count != config.expected_parameter_count
        or parameter_count != FROZEN_PARAMETER_COUNT
    ):
        raise RuntimeError("frozen PAET-BFA parameter count changed")
    frozen_hashes = {
        key: digest for key, _, digest in FROZEN_REAL_DR_INPUTS
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
                "CURELitePhaseAlignedEvidenceTransportLevelSet"
            ),
            "version": "v21",
            "candidate": "PAET-BFA",
            "input_interface": ["F_b", "O"],
            "input_representation": PAET_INPUT_REPRESENTATION,
            "field_policy": CSLF_PAET_FIELD_POLICY,
            "equation_policy": CSLF_PAET_EQUATION_POLICY,
            "flip_policy": CSLF_PAET_FLIP_POLICY,
            "transport_policy": CSLF_PAET_TRANSPORT_POLICY,
            "feature_channels": config.feature_channels,
            "feature_stride": config.feature_stride,
            "phase_occupancy_channels": (
                config.phase_occupancy_channels
            ),
            "width": config.width,
            "parameter_count": parameter_count,
            "parameter_tensor_count": 3,
            "field_threshold": 0.0,
            "threshold_search_performed": False,
            "objective_suite": ["pmope_joint"],
            "candidate_objective": "pmope_joint",
            "candidate_objective_policy": CSLF_PMOPE_POLICY,
            "fixed_margin_hex": COVERAGE_STATE_PAET_MARGIN.hex(),
            "allowed_difference_from_v20": (
                "predeclared_phase_aligned_evidence_transport_only"
            ),
            "single_completion_field": True,
            "additional_learned_components": 0,
        },
        "budget": {
            "seed": FROZEN_SEED,
            "epochs": FROZEN_EPOCHS,
            "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
            "updates_per_objective": FROZEN_UPDATES_PER_OBJECTIVE,
            "objectives": 1,
        },
        "execution": {
            "device": FROZEN_DEVICE,
            "CUDA_VISIBLE_DEVICES": FROZEN_VISIBLE_GPU,
            "CUBLAS_WORKSPACE_CONFIG": (
                FROZEN_CUBLAS_WORKSPACE_CONFIG
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
        "resource_measurement": {
            "scope": (
                "single_training_invocation_including_device_cache_setup_"
                "and_post_verification"
            ),
            "actual_fields": [
                "baseline_allocated_bytes",
                "baseline_reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "incremental_peak_allocated_bytes",
                "incremental_peak_reserved_bytes",
                "elapsed_ns",
                "ns_per_update",
            ],
            "v20_comparison_status": (
                COVERAGE_STATE_PAET_V20_RESOURCE_COMPARISON_STATUS
            ),
            "v20_measured_reference_available": False,
            "working_memory_ratio_limit": {
                "numerator": COVERAGE_STATE_PAET_MEMORY_RATIO_LIMIT[0],
                "denominator": COVERAGE_STATE_PAET_MEMORY_RATIO_LIMIT[1],
            },
            "step_time_ratio_limit": {
                "numerator": (
                    COVERAGE_STATE_PAET_STEP_TIME_RATIO_LIMIT[0]
                ),
                "denominator": (
                    COVERAGE_STATE_PAET_STEP_TIME_RATIO_LIMIT[1]
                ),
            },
            "ratio_is_scientific_gate": False,
            "ratio_claim_supported": False,
        },
        "dataset_free_gate": {
            "binding_mode": "actual_runtime_receipt_fingerprint",
            "receipt_fingerprint": dataset_free_receipt_fingerprint,
        },
        "real_D_R_gate": {
            "status": "not_run_in_static_config",
            "receipt_fingerprint": None,
            "run_once_only": True,
        },
        "sealed_v20_reference": {
            "reference_fingerprint": (
                sealed_v20_reference_fingerprint
            ),
            "read_only": True,
            "retrained": False,
            "reevaluated": False,
            "measured_resource_reference_available": False,
        },
        "post_training_certificate": {
            "status": "not_run_in_static_config",
            "run_once_only": True,
            "pair_result_is_bounded_gate": False,
        },
        "bounded_decision": {
            "status": "not_run_in_static_config",
            "clean_target_negative_minimum": [124, 149],
            "clean_outside_completion_maximum": 46,
            "factual_recovered_required": [16, 16],
            "factual_strict_minimum": [14, 16],
            "factual_target_negative_minimum": [310, 335],
            "component_null_required": [16, 16],
            "factual_no_miss_required": [16, 16],
            "identity_null_required": [16, 16],
            "diagnostic_null_required": [1, 1],
            "invalid_completion_required": 0,
            "compact_support_minimum": [1, 16],
            "threshold_search_performed": False,
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


def _device_memory_preflight(
    cache: object,
    model_config: CoverageStatePhaseAlignedEvidenceTransportConfig,
) -> dict[str, object]:
    """Minimum capacity admission; the runner records actual peak usage."""

    if (
        type(model_config)
        is not CoverageStatePhaseAlignedEvidenceTransportConfig
    ):
        raise TypeError("PAET memory preflight requires exact config class")
    if (
        model_config.feature_channels != FROZEN_FEATURE_CHANNELS
        or model_config.feature_stride != FROZEN_FEATURE_STRIDE
        or model_config.width != FROZEN_MODEL_WIDTH
        or model_config.expected_parameter_count
        != FROZEN_PARAMETER_COUNT
    ):
        raise RuntimeError("PAET static memory model contract changed")
    source_cache_fingerprint = getattr(cache, "cache_fingerprint", None)
    parameter_bytes = FROZEN_PARAMETER_COUNT * 4
    model_optimizer_bytes = 4 * parameter_bytes
    required = (
        _FROZEN_RESIDENT_DEVICE_CACHE_BYTES
        + model_optimizer_bytes
        + _MINIMUM_ACTIVATION_ADMISSION_RESERVE_BYTES
    )
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "exactly_one_visible_device": torch.cuda.device_count() == 1,
        "visible_cuda_zero": torch.cuda.current_device() == 0,
        "source_cache_fingerprint_exact": (
            source_cache_fingerprint
            == COVERAGE_STATE_PAET_HISTORICAL_CACHE_FINGERPRINT
        ),
        "device_cache_fingerprint_exact": (
            len(
                COVERAGE_STATE_PAET_HISTORICAL_DEVICE_CACHE_FINGERPRINT
            )
            == 64
        ),
        "optimizer_fingerprint_exact": (
            len(COVERAGE_STATE_PAET_HISTORICAL_OPTIMIZER_FINGERPRINT)
            == 64
        ),
        "free_memory_meets_minimum_admission": (
            int(free_bytes) >= required
        ),
        "total_memory_meets_minimum_admission": (
            int(total_bytes) >= required
        ),
    }
    result = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                "device-memory-preflight-v1"
            ),
            "run_id": RUN_ID,
            "device": FROZEN_DEVICE,
            "model_class": (
                "CURELitePhaseAlignedEvidenceTransportLevelSet"
            ),
            "model_parameter_count": (
                model_config.expected_parameter_count
            ),
            "source_cache_fingerprint": source_cache_fingerprint,
            "projected_device_cache_fingerprint": (
                COVERAGE_STATE_PAET_HISTORICAL_DEVICE_CACHE_FINGERPRINT
            ),
            "device_cache_resident_bytes": (
                _FROZEN_RESIDENT_DEVICE_CACHE_BYTES
            ),
            "optimizer_config_fingerprint": (
                COVERAGE_STATE_PAET_HISTORICAL_OPTIMIZER_FINGERPRINT
            ),
            "model_parameter_bytes": parameter_bytes,
            "model_optimizer_retention_bytes": model_optimizer_bytes,
            "minimum_activation_admission_reserve_bytes": (
                _MINIMUM_ACTIVATION_ADMISSION_RESERVE_BYTES
            ),
            "minimum_required_free_bytes": required,
            "observed_free_bytes": int(free_bytes),
            "observed_total_bytes": int(total_bytes),
            "preflight_is_not_peak_measurement": True,
            "actual_peak_recorded_by_runner": True,
            "checks": checks,
            "all_pass": all(checks.values()),
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
        }
    )
    if not result["all_pass"]:
        raise RuntimeError("PAET device memory preflight did not pass")
    return result


def _write_checkpoint_new(
    directory: Path,
    *,
    objective: str,
    objective_policy: str,
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
) -> dict[str, object]:
    """Write and roundtrip the sole tensor-only PAET checkpoint."""

    if type(model) is not CURELitePhaseAlignedEvidenceTransportLevelSet:
        raise TypeError("PAET checkpoint requires the exact model class")
    if objective != "pmope_joint" or objective_policy != CSLF_PMOPE_POLICY:
        raise ValueError("checkpoint must be the singleton PAET candidate")
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
    if set(loaded) != set(state) or any(
        not torch.equal(loaded[name], state[name]) for name in state
    ):
        raise RuntimeError("PAET checkpoint roundtrip changed")
    result = _fingerprinted(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "run_id": RUN_ID,
            "objective": objective,
            "objective_policy": objective_policy,
            "model_class": (
                "CURELitePhaseAlignedEvidenceTransportLevelSet"
            ),
            "model_config": {
                "feature_channels": model.config.feature_channels,
                "feature_stride": model.config.feature_stride,
                "width": model.config.width,
                "field_policy": model.config.field_policy,
                "equation_policy": model.config.equation_policy,
                "flip_policy": model.config.flip_policy,
                "transport_policy": model.config.transport_policy,
                "parameter_count": sum(
                    value.numel() for value in model.parameters()
                ),
                "fixed_margin_hex": COVERAGE_STATE_PAET_MARGIN.hex(),
            },
            "repo_relative_path": str(path.relative_to(_ROOT)),
            "serialization": FROZEN_CHECKPOINT_SERIALIZATION,
            "tensor_only_state_dict": True,
            "weights_only_roundtrip_verified": True,
            "checkpoint_file_sha256": file_sha256(path),
            "module_state_fingerprint": module_state_fingerprint(model),
            "state_keys": list(state),
            "device_policy": "cpu_checkpoint",
        }
    )
    _write_new_json(
        directory / f"{objective}.checkpoint.json",
        result,
    )
    return result


def _failure_payload(
    error: BaseException,
    *,
    attempt_fingerprint: str,
    artifact_files: Mapping[str, str],
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": FAILURE_SCHEMA,
            "run_id": RUN_ID,
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
    """Validate static bindings without claiming output or reading D_R."""

    source_paths = _verify_frozen_sources()
    dataset_free = run_coverage_state_paet_dataset_free_gate()
    if not dataset_free.all_pass:
        raise RuntimeError("PAET dataset-free gate did not pass")
    sealed_v20 = (
        verify_repository_coverage_state_bfa_v20_reference(_ROOT)
    )
    implementation = _implementation_binding()
    config = _static_config_payload(
        source_paths=source_paths,
        implementation=implementation,
        dataset_free_receipt_fingerprint=(
            dataset_free.receipt_fingerprint
        ),
        sealed_v20_reference_fingerprint=(
            sealed_v20.reference_fingerprint
        ),
    )
    _assert_run_id(config["run_id"], layer="static config")
    output_exists = OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink()
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
                dataset_free.receipt_fingerprint
            ),
            "dataset_free_gate_passed": dataset_free.all_pass,
            "sealed_v20_reference_fingerprint": (
                sealed_v20.reference_fingerprint
            ),
            "sealed_v20_reference_read_only": True,
            "D_R_gate_status": "not_run",
            "D_R_gate_performed": False,
            "bounded_400_authorized": False,
            "run_once_static_prerequisites_valid": True,
            "bounded_output_exists": output_exists,
            "run_once_implemented": True,
            "output_claimed": False,
            "D_R_cached_tensor_payload_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "authorization_created": False,
            "training_performed": False,
            "resource_measurement_performed": False,
            "post_training_certificate_performed": False,
            "zero_level_evaluation_performed": False,
            "formal800_eligible": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
            "not_a_formal_result": True,
        }
    )


def _dataset_free_payload(dataset_free: object) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                "dataset-free-v1"
            ),
            "run_id": RUN_ID,
            "dataset_free": dataset_free.canonical_payload(),
            "dataset_free_receipt_fingerprint": (
                dataset_free.receipt_fingerprint
            ),
            "all_pass": dataset_free.all_pass,
            "invocations": 1,
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def _sealed_v20_payload(sealed_v20: object) -> dict[str, object]:
    payload = sealed_v20.canonical_payload()
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                "sealed-v20-reference-v1"
            ),
            "run_id": RUN_ID,
            "sealed_v20_reference": payload,
            "sealed_v20_reference_fingerprint": (
                sealed_v20.reference_fingerprint
            ),
            "read_only": True,
            "historical_result_retrained": False,
            "historical_result_reevaluated": False,
            "historical_result_is_candidate_gate": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def _prepare_paet_authorization(
    preflight: object,
    dataset_free: object,
    dr_gate: object,
    *,
    sealed_v20_reference: object,
) -> object:
    return prepare_coverage_state_paet_bounded_run_authorization(
        preflight,
        dataset_free,
        dr_gate,
        run_id=RUN_ID,
        sealed_v20_reference=sealed_v20_reference,
    )


def _expected_paet_config(
    preflight: object,
) -> CoverageStatePhaseAlignedEvidenceTransportConfig:
    return expected_coverage_state_paet_config(preflight)


def _run_paet_bounded(
    authorization: object,
    model_config: CoverageStatePhaseAlignedEvidenceTransportConfig,
) -> object:
    return run_coverage_state_paet_bfa_pmope_bounded_400(
        authorization,
        model_config,
        run_id=RUN_ID,
        device=FROZEN_DEVICE,
    )


def _certificate_payload(result: object) -> dict[str, object]:
    _assert_run_id(result.run_id, layer="certificate source result")
    certificate = result.certificate
    certificate.verify()
    payload = certificate.canonical_payload()
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                "post-training-certificate-v1"
            ),
            "run_id": RUN_ID,
            "certificate": payload,
            "certificate_receipt_fingerprint": (
                stable_fingerprint(payload)
            ),
            "certificate_integrity_passed": (
                certificate.integrity_passed
            ),
            "all_pairs_passed": certificate.all_pairs_passed,
            "pair_result_is_bounded_gate": False,
            "certificate_invocations": result.certificate_invocations,
            "optimizer_constructed": False,
            "backward_performed": False,
            "training_performed": False,
            "external_data_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def _zero_level_payload(result: object) -> dict[str, object]:
    _assert_run_id(result.run_id, layer="zero-level source result")
    diagnostic_payload = result.diagnostic.canonical_payload()
    decision_payload = result.decision.canonical_payload()
    _assert_run_id(
        decision_payload.get("run_id"),
        layer="zero-level decision",
    )
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                "zero-level-v1"
            ),
            "run_id": RUN_ID,
            "input_representation": PAET_INPUT_REPRESENTATION,
            "threshold": 0.0,
            "threshold_search_performed": False,
            "candidate_diagnostic": diagnostic_payload,
            "diagnostic_result_fingerprint": (
                stable_fingerprint(diagnostic_payload)
            ),
            "paet_zero_level_decision": decision_payload,
            "paet_zero_level_decision_fingerprint": (
                stable_fingerprint(decision_payload)
            ),
            "candidate_zero_level_gate_passed": (
                result.decision.bounded_gate_passed
            ),
            "same_sign_response_is_gate": False,
            "zero_level_evaluation_invocations": (
                result.zero_level_evaluation_invocations
            ),
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def _decision_payload(
    result: object,
    checkpoints: Sequence[Mapping[str, object]],
    *,
    result_fingerprint: str | None = None,
) -> dict[str, object]:
    _assert_run_id(result.run_id, layer="decision source result")
    if (
        len(checkpoints) != 1
        or checkpoints[0].get("objective") != "pmope_joint"
        or checkpoints[0].get("run_id") != RUN_ID
    ):
        raise ValueError("PAET decision requires one bound checkpoint")
    passed = bool(result.bounded_gate_passed)
    bound_result_fingerprint = (
        result.result_fingerprint
        if result_fingerprint is None
        else result_fingerprint
    )
    if (
        not isinstance(bound_result_fingerprint, str)
        or len(bound_result_fingerprint) != 64
    ):
        raise ValueError("PAET result fingerprint is invalid")
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "run_id": RUN_ID,
            "status": (
                "PAET_BFA_V21_BOUNDED_400_GATE_PASS"
                if passed
                else "PAET_BFA_V21_BOUNDED_400_GATE_FAIL"
            ),
            "bounded_gate_passed": passed,
            "failed_checks": list(result.failed_checks),
            "result_fingerprint": bound_result_fingerprint,
            "checkpoint_receipt_fingerprints": {
                str(value["objective"]): str(
                    value["receipt_fingerprint"]
                )
                for value in checkpoints
            },
            "candidate_model": "PAET-BFA",
            "candidate_objective": "pmope_joint",
            "post_training_certificate_integrity_passed": (
                result.certificate.integrity_passed
            ),
            "pair_certificate_result_is_bounded_gate": False,
            "zero_level_gate_passed": (
                result.decision.bounded_gate_passed
            ),
            "zero_level_failed_checks": list(
                result.decision.failed_checks
            ),
            "same_sign_response_is_gate": False,
            "training_invocations": result.training_invocations,
            "certificate_invocations": result.certificate_invocations,
            "zero_level_evaluation_invocations": (
                result.zero_level_evaluation_invocations
            ),
            "resource_measurement_fingerprint": (
                result
                .resource_measurement.measurement_fingerprint
            ),
            "resource_ratio_claim_supported": False,
            "resource_ratio_is_scientific_gate": False,
            "sealed_v20_reference_fingerprint": (
                result.authorization.sealed_v20_reference_fingerprint
            ),
            "historical_result_retrained": False,
            "historical_result_reevaluated": False,
            "historical_result_is_candidate_gate": False,
            "next_action": (
                "freeze_paet_v21_and_design_formal800_protocol"
                if passed
                else "freeze_paet_v21_negative_and_review_structure"
            ),
            "formal800_eligible": passed,
            "formal_800_authorized": False,
            "formal_800_executed": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        }
    )


def run_once() -> dict[str, object]:
    """Consume the sole fixed seed-42 PAET-BFA bounded attempt."""

    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError(
            f"single-use bounded output already exists: {OUTPUT_PATH}"
        )

    dataset_free = run_coverage_state_paet_dataset_free_gate()
    if not dataset_free.all_pass:
        raise PermissionError("PAET dataset-free gate did not pass")
    sealed_v20 = (
        verify_repository_coverage_state_bfa_v20_reference(_ROOT)
    )
    source_paths = _verify_frozen_sources()
    implementation = _implementation_binding()
    runtime = _v15b_cli._verify_runtime_contract()
    config = _fingerprinted(
        _static_config_payload(
            source_paths=source_paths,
            implementation=implementation,
            dataset_free_receipt_fingerprint=(
                dataset_free.receipt_fingerprint
            ),
            sealed_v20_reference_fingerprint=(
                sealed_v20.reference_fingerprint
            ),
        )
    )
    _assert_run_id(config["run_id"], layer="config")
    attempt = _fingerprinted(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "run_id": RUN_ID,
            "output_repo_path": OUTPUT_REPO_PATH,
            "config_fingerprint": config["receipt_fingerprint"],
            "runtime": runtime,
            "dataset_free_receipt_fingerprint": (
                dataset_free.receipt_fingerprint
            ),
            "sealed_v20_reference_fingerprint": (
                sealed_v20.reference_fingerprint
            ),
            "candidate_model": "PAET-BFA",
            "candidate_objective": "pmope_joint",
            "objectives": 1,
            "dataset_free_invocations_before_claim": 1,
            "D_R_gate_run_count_before_claim": 0,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "formal_800_executed": False,
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
        dataset_free_receipt = _dataset_free_payload(dataset_free)
        _write_new_json(
            receipts / "dataset_free.json",
            dataset_free_receipt,
        )
        sealed_receipt = _sealed_v20_payload(sealed_v20)
        _write_new_json(
            receipts / "sealed_v20_reference.json",
            sealed_receipt,
        )

        real_inputs = build_coverage_state_real_dr_inputs(**source_paths)
        population = build_coverage_state_bounded_population(
            real_inputs.scalar_cache,
            seed=FROZEN_SEED,
        )
        preflight = prepare_coverage_state_bounded_preflight(population)
        exposure = coverage_state_schedule_exposure_report(
            population.cache,
            preflight.schedule,
        )
        input_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                    "inputs-v1"
                ),
                "run_id": RUN_ID,
                "real_D_R_inputs": real_inputs.canonical_payload(),
                "source_binding": (
                    real_inputs.source_binding.canonical_payload()
                ),
                "bounded_population": population.canonical_payload(),
                "population_fingerprint": (
                    population.population_fingerprint
                ),
                "construction_invocations": {
                    "real_inputs": 1,
                    "population": 1,
                },
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(receipts / "inputs.json", input_receipt)
        preflight_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                    "preflight-v1"
                ),
                "run_id": RUN_ID,
                "preflight": preflight.canonical_payload(),
                "schedule": preflight.schedule.canonical_payload(),
                "schedule_selections": [
                    value.canonical_payload()
                    for value in preflight.schedule.selections
                ],
                "exposure": exposure,
                "preflight_invocations": 1,
                "training_authorized": preflight.training_authorized,
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
            raise PermissionError("bounded D_R preflight did not pass")

        dr_gate = run_coverage_state_paet_dr_gate(
            dataset_free_receipt=dataset_free,
            real_inputs=real_inputs,
            bounded_population=population,
            device=FROZEN_DEVICE,
        )
        dr_gate_payload = dr_gate.canonical_payload()
        dr_gate_passed = bool(dr_gate.all_pass)
        dr_gate_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                    "real-D_R-gate-v1"
                ),
                "run_id": RUN_ID,
                "D_R_gate": dr_gate_payload,
                "D_R_gate_evidence_fingerprint": (
                    dr_gate.evidence_fingerprint
                ),
                "all_pass": dr_gate_passed,
                "gate_run_count": 1,
                "optimizer_steps": 0,
                "training_performed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(receipts / "dr_gate.json", dr_gate_receipt)

        if not dr_gate_passed:
            decision = _fingerprinted(
                {
                    "schema_version": DECISION_SCHEMA,
                    "run_id": RUN_ID,
                    "status": "PAET_BFA_V21_DR_GATE_FAIL",
                    "bounded_gate_passed": False,
                    "D_R_gate_passed": False,
                    "D_R_gate_evidence_fingerprint": (
                        dr_gate.evidence_fingerprint
                    ),
                    "authorization_created": False,
                    "bounded_training_performed": False,
                    "resource_measurement_performed": False,
                    "post_training_certificate_performed": False,
                    "zero_level_evaluation_performed": False,
                    "checkpoint_count": 0,
                    "formal800_eligible": False,
                    "formal_800_authorized": False,
                    "formal_800_executed": False,
                    "full_CURE_authorized": False,
                    "cross_backbone_authorized": False,
                    "performance_claim_supported": False,
                    "next_action": (
                        "freeze_paet_D_R_gate_negative_and_stop_before_"
                        "bounded_training"
                    ),
                    "resume_allowed": False,
                    "automatic_retry_allowed": False,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                }
            )
            _write_new_json(receipts / "decision.json", decision)
            artifacts = _v15b_cli._artifact_hashes(OUTPUT_PATH)
            if len(artifacts) != FROZEN_DR_GATE_STOP_ARTIFACT_FILE_COUNT:
                raise RuntimeError(
                    "PAET D_R gate-stop artifact population is incomplete"
                )
            complete = _fingerprinted(
                {
                    "schema_version": RUN_SCHEMA,
                    "run_id": RUN_ID,
                    "status": "complete",
                    "decision": decision["status"],
                    "bounded_gate_passed": False,
                    "D_R_gate_passed": False,
                    "artifact_files": artifacts,
                    "artifact_file_count": len(artifacts),
                    "authorization_created": False,
                    "bounded_training_performed": False,
                    "resource_measurement_performed": False,
                    "post_training_certificate_performed": False,
                    "zero_level_evaluation_performed": False,
                    "checkpoint_count": 0,
                    "formal800_eligible": False,
                    "formal_800_authorized": False,
                    "formal_800_executed": False,
                    "full_CURE_authorized": False,
                    "cross_backbone_authorized": False,
                    "performance_claim_supported": False,
                    "single_attempt": True,
                    "resume_allowed": False,
                    "automatic_retry_allowed": False,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                },
                field="complete_fingerprint",
            )
            _write_new_json(OUTPUT_PATH / "COMPLETE.json", complete)
            (OUTPUT_PATH / _INCOMPLETE).unlink()
            return {
                "run_id": RUN_ID,
                "output": str(OUTPUT_PATH),
                "decision": decision["status"],
                "bounded_gate_passed": False,
                "complete_fingerprint": complete[
                    "complete_fingerprint"
                ],
                "formal_800_authorized": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }

        authorization = _prepare_paet_authorization(
            preflight,
            dataset_free,
            dr_gate,
            sealed_v20_reference=sealed_v20,
        )
        _assert_run_id(authorization.run_id, layer="authorization")
        model_config = _expected_paet_config(preflight)
        if (
            type(model_config)
            is not CoverageStatePhaseAlignedEvidenceTransportConfig
            or model_config.expected_parameter_count
            != FROZEN_PARAMETER_COUNT
        ):
            raise RuntimeError("real D_R PAET model config changed")
        authorization_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                    "authorization-v1"
                ),
                "run_id": RUN_ID,
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
                "D_R_gate_evidence_fingerprint": (
                    dr_gate.evidence_fingerprint
                ),
                "sealed_v20_reference_fingerprint": (
                    sealed_v20.reference_fingerprint
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
            raise PermissionError("PAET authorization did not pass")

        memory = _device_memory_preflight(
            population.cache,
            model_config,
        )
        _write_new_json(
            receipts / "device_memory_preflight.json",
            memory,
        )
        result = _run_paet_bounded(authorization, model_config)
        _assert_run_id(result.run_id, layer="bounded result")
        if (
            result.training_invocations != 1
            or result.certificate_invocations != 1
            or result.zero_level_evaluation_invocations != 1
            or len(result.training.results) != 1
            or len(result.training.models) != 1
            or result.training.results[0].objective != "pmope_joint"
            or result.training.models[0][0] != "pmope_joint"
        ):
            raise RuntimeError("PAET run returned a non-singleton result")

        checkpoint_receipts = (
            _write_checkpoint_new(
                checkpoints_dir,
                objective="pmope_joint",
                objective_policy=(
                    result.training.results[0].objective_policy
                ),
                model=result.training.models[0][1],
            ),
        )
        resource_payload = (
            result.resource_measurement.canonical_payload()
        )
        resource_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                    "training-resource-measurement-v1"
                ),
                "run_id": RUN_ID,
                "measurement": resource_payload,
                "measurement_fingerprint": (
                    result
                    .resource_measurement.measurement_fingerprint
                ),
                "v20_ratio_claim_supported": False,
                "resource_ratio_is_scientific_gate": False,
            }
        )
        _write_new_json(
            receipts / "training_resource_measurement.json",
            resource_receipt,
        )
        training_payload = result.training.canonical_payload()
        training_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                    "training-v1"
                ),
                "run_id": RUN_ID,
                "training": training_payload,
                "training_result_fingerprint": (
                    stable_fingerprint(training_payload)
                ),
                "resource_measurement_receipt_fingerprint": (
                    resource_receipt["receipt_fingerprint"]
                ),
                "checkpoint_receipt_fingerprints": {
                    str(value["objective"]): str(
                        value["receipt_fingerprint"]
                    )
                    for value in checkpoint_receipts
                },
                "training_invocations": result.training_invocations,
                "formal_training_performed": False,
                "bounded_training_performed": True,
                "candidate_count": 1,
                "candidate_model": "PAET-BFA",
                "candidate_objective": "pmope_joint",
                "historical_result_retrained": False,
                "all_models_exact_paet_class": True,
                "all_models_parameter_count": FROZEN_PARAMETER_COUNT,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(receipts / "training.json", training_receipt)
        certificate_receipt = _certificate_payload(result)
        _write_new_json(
            receipts / "post_training_certificate.json",
            certificate_receipt,
        )
        zero_receipt = _zero_level_payload(result)
        _write_new_json(receipts / "zero_level.json", zero_receipt)
        result_payload = result.canonical_payload()
        _assert_run_id(result_payload["run_id"], layer="result payload")
        result_fingerprint = stable_fingerprint(result_payload)
        bounded_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-pmope-bounded-400-"
                    "result-v1"
                ),
                "run_id": RUN_ID,
                "result": result_payload,
                "result_fingerprint": result_fingerprint,
            }
        )
        _write_new_json(
            receipts / "bounded_result.json",
            bounded_receipt,
        )
        decision = _decision_payload(
            result,
            checkpoint_receipts,
            result_fingerprint=result_fingerprint,
        )
        _write_new_json(receipts / "decision.json", decision)

        if _implementation_binding() != implementation:
            raise RuntimeError(
                "PAET implementation changed during execution"
            )
        artifacts = _v15b_cli._artifact_hashes(OUTPUT_PATH)
        if len(artifacts) != FROZEN_ARTIFACT_FILE_COUNT:
            raise RuntimeError(
                "PAET terminal artifact population is incomplete"
            )
        complete = _fingerprinted(
            {
                "schema_version": RUN_SCHEMA,
                "run_id": RUN_ID,
                "status": "complete",
                "decision": decision["status"],
                "bounded_gate_passed": result.bounded_gate_passed,
                "formal800_eligible": result.formal800_eligible,
                "formal_800_authorized": False,
                "formal_800_executed": False,
                "full_CURE_authorized": False,
                "cross_backbone_authorized": False,
                "performance_claim_supported": False,
                "split": "D_R",
                "runtime_splits": ["D_R"],
                "resource_measurement_fingerprint": (
                    result
                    .resource_measurement.measurement_fingerprint
                ),
                "resource_ratio_claim_supported": False,
                "pair_certificate_result_is_bounded_gate": False,
                "artifact_files": artifacts,
                "artifact_file_count": len(artifacts),
                "dataset_free_invocations": 1,
                "real_inputs_construction_invocations": 1,
                "population_construction_invocations": 1,
                "preflight_invocations": 1,
                "D_R_gate_invocations": 1,
                "training_invocations": result.training_invocations,
                "resource_measurement_invocations": 1,
                "post_training_certificate_invocations": (
                    result.certificate_invocations
                ),
                "zero_level_evaluation_invocations": (
                    result.zero_level_evaluation_invocations
                ),
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
            "run_id": RUN_ID,
            "output": str(OUTPUT_PATH),
            "decision": decision["status"],
            "bounded_gate_passed": result.bounded_gate_passed,
            "formal800_eligible": result.formal800_eligible,
            "complete_fingerprint": complete["complete_fingerprint"],
            "formal_800_authorized": False,
            "formal_800_executed": False,
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
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-create-only",
        action="store_true",
        help="validate without claiming output or loading D_R tensors",
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
