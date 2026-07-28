#!/usr/bin/env python3
"""Validate or execute the unique CMIF/USCOPE v19 bounded-400 protocol.

``--validate-create-only`` checks only generated/static bindings.  It does
not claim output, load real-``D_R`` tensors, construct an optimizer, evaluate
a checkpoint, or train.  ``--run-once`` consumes one fixed seed-42,
400-update attempt.  The command exposes no seed, update-count, output,
resume, or retry option and never accesses ``D_V`` or ``D_T``.
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
from cure_lite.coverage_state_centered_mixed_interaction import (
    CMIF_ENERGY_POLICY,
    CMIF_INPUT_REPRESENTATION,
    CMIF_INTERACTION_POLICY,
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from cure_lite.coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
)
from cure_lite.coverage_state_schedule import (
    coverage_state_schedule_exposure_report,
)
from cure_lite.coverage_state_supremal_projection import (
    CSLF_USCOPE_POLICY,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_cmif_dataset_free import (
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_CMIF_FORMAL_WIDTH,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite.experiment.coverage_state_uscope_dataset_free import (
    COVERAGE_STATE_USCOPE_MARGIN,
    run_coverage_state_uscope_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_uscope_dr_gate import (
    run_coverage_state_uscope_dr_gate,
)
from cure_lite.experiment.coverage_state_uscope_sealed_v18 import (
    verify_repository_coverage_state_uscope_sealed_v18,
)
from cure_lite.frozen_base import module_state_fingerprint
from tools import run_coverage_state_cmif_pmope_bounded_400 as _pmope_cli
from tools import (
    run_coverage_state_cslf_support_oriented_bounded_400 as _v15b_cli,
)


_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "cure_lite_cmif_v19_uscope_bounded_400_r1"
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = _ROOT / OUTPUT_REPO_PATH
RUN_SCHEMA = "cure-lite-cmif-v19-uscope-bounded-400-run-v1"
VALIDATION_SCHEMA = (
    "cure-lite-cmif-v19-uscope-bounded-400-create-only-validation-v1"
)
ATTEMPT_SCHEMA = "cure-lite-cmif-v19-uscope-bounded-400-attempt-v1"
FAILURE_SCHEMA = "cure-lite-cmif-v19-uscope-bounded-400-failure-v1"
CHECKPOINT_SCHEMA = (
    "cure-lite-cmif-v19-uscope-bounded-400-checkpoint-v1"
)
DECISION_SCHEMA = "cure-lite-cmif-v19-uscope-bounded-400-decision-v1"
FROZEN_DEVICE = _v15b_cli.FROZEN_DEVICE
FROZEN_VISIBLE_GPU = _v15b_cli.FROZEN_VISIBLE_GPU
FROZEN_CUBLAS_WORKSPACE_CONFIG = _v15b_cli.FROZEN_CUBLAS_WORKSPACE_CONFIG
FROZEN_PAUSE_TEMPERATURE_C = _v15b_cli.FROZEN_PAUSE_TEMPERATURE_C
FROZEN_RESUME_TEMPERATURE_C = _v15b_cli.FROZEN_RESUME_TEMPERATURE_C
FROZEN_CHECKPOINT_SERIALIZATION = "safetensors"
FROZEN_FEATURE_CHANNELS = COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS
FROZEN_FEATURE_STRIDE = COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE
FROZEN_MODEL_WIDTH = COVERAGE_STATE_CMIF_FORMAL_WIDTH
FROZEN_PARAMETER_COUNT = 64064
FROZEN_SEED = 42
FROZEN_EPOCHS = 10
FROZEN_STEPS_PER_EPOCH = 40
FROZEN_UPDATES_PER_OBJECTIVE = 400
FROZEN_ARTIFACT_FILE_COUNT = 16
FROZEN_DR_GATE_STOP_ARTIFACT_FILE_COUNT = 8
FROZEN_REAL_DR_INPUTS = _pmope_cli.FROZEN_REAL_DR_INPUTS
_ACTIVATION_RESERVE_BYTES = 2 * 1024**3
_FROZEN_RESIDENT_DEVICE_CACHE_BYTES = 205_521_408
_FROZEN_DEVICE_CACHE_FINGERPRINT = (
    "76ed2f94b4187154bad62896b93d637f131865f2e4e8dad38becb4bebc71119f"
)
_FROZEN_OPTIMIZER_FINGERPRINT = (
    "2d058b1cad606e3c1b723aab05925efb2e873c2b3bf021aeaf0f7df40e0690f0"
)
_FROZEN_SOURCE_CACHE_FINGERPRINT = (
    "c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca00754c206092aaf66d8"
)
_INCOMPLETE = ".incomplete"

_fingerprinted = _pmope_cli._fingerprinted
_write_new_json = _pmope_cli._write_new_json


def _verify_frozen_sources() -> dict[str, Path]:
    """Reuse the frozen real-D_R file paths and exact hashes from v18."""

    return _pmope_cli._verify_frozen_sources()


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    """Bind every implementation file used by the v19 command."""

    from cure_lite.experiment.coverage_state_uscope_bounded_runner import (
        _current_implementation_binding,
    )

    core = dict(_current_implementation_binding())
    extras = (
        "tools/run_coverage_state_cmif_uscope_bounded_400.py",
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
                f"USCOPE bounded implementation path changed: {relative}"
            )
        core[relative] = file_sha256(absolute)
    return tuple(sorted(core.items()))


def _static_config_payload(
    *,
    source_paths: Mapping[str, Path],
    implementation: tuple[tuple[str, str], ...],
    dataset_free_receipt_fingerprint: str,
    sealed_v18_receipt_fingerprint: str,
) -> dict[str, object]:
    """Construct the immutable singleton seed-42 USCOPE configuration."""

    if (
        len(dataset_free_receipt_fingerprint) != 64
        or len(sealed_v18_receipt_fingerprint) != 64
    ):
        raise ValueError("USCOPE prerequisite fingerprint changed")
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=FROZEN_FEATURE_CHANNELS,
        feature_stride=FROZEN_FEATURE_STRIDE,
        width=FROZEN_MODEL_WIDTH,
    )
    parameter_count = config.expected_parameter_count
    if (
        parameter_count != FROZEN_PARAMETER_COUNT
    ):
        raise RuntimeError("frozen CMIF parameter count changed")
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
            "class": "CURELiteCenteredMixedInteractionLevelSet",
            "input_representation": CMIF_INPUT_REPRESENTATION,
            "coverage_policy": CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
            "interaction_policy": CMIF_INTERACTION_POLICY,
            "energy_policy": CMIF_ENERGY_POLICY,
            "feature_channels": config.feature_channels,
            "feature_stride": config.feature_stride,
            "phase_occupancy_channels": config.phase_occupancy_channels,
            "width": config.width,
            "parameter_count": parameter_count,
            "field_threshold": 0.0,
            "threshold_search_performed": False,
            "objective_suite": ["uscope_joint"],
            "candidate_objective": "uscope_joint",
            "candidate_objective_policy": CSLF_USCOPE_POLICY,
            "fixed_margin_hex": COVERAGE_STATE_USCOPE_MARGIN.hex(),
            "same_sign_response_is_gate": False,
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
            "CUBLAS_WORKSPACE_CONFIG": FROZEN_CUBLAS_WORKSPACE_CONFIG,
            "pause_temperature_c": FROZEN_PAUSE_TEMPERATURE_C,
            "resume_temperature_c": FROZEN_RESUME_TEMPERATURE_C,
            "checkpoint_serialization": FROZEN_CHECKPOINT_SERIALIZATION,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
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
        "sealed_v18_negative_result": {
            "receipt_fingerprint": sealed_v18_receipt_fingerprint,
            "read_only": True,
            "checkpoint_deserialized": False,
            "retrained": False,
            "reevaluated": False,
            "candidate_gate": False,
        },
        "post_training_certificate": {
            "status": "not_run_in_static_config",
            "run_once_only": True,
            "same_sign_response_is_gate": False,
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
            "post_training_certificate_performed": False,
            "zero_level_evaluation_performed": False,
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
        },
    }


def _device_memory_preflight(
    cache: object,
    model_config: CoverageStateCenteredMixedInteractionConfig,
) -> dict[str, object]:
    """Check static sealed memory bounds without repacking the cache."""

    if type(model_config) is not CoverageStateCenteredMixedInteractionConfig:
        raise TypeError("CMIF memory preflight requires exact config class")
    if FROZEN_DEVICE != "cuda:0":
        raise RuntimeError("USCOPE bounded execution requires cuda:0")
    if (
        model_config.feature_channels != FROZEN_FEATURE_CHANNELS
        or model_config.feature_stride != FROZEN_FEATURE_STRIDE
        or model_config.width != FROZEN_MODEL_WIDTH
        or model_config.expected_parameter_count
        != FROZEN_PARAMETER_COUNT
    ):
        raise RuntimeError("USCOPE static memory model contract changed")
    source_cache_fingerprint = getattr(cache, "cache_fingerprint", None)
    parameter_bytes = FROZEN_PARAMETER_COUNT * 4
    buffer_bytes = 0
    model_optimizer_bytes = 4 * parameter_bytes + buffer_bytes
    required = (
        _FROZEN_RESIDENT_DEVICE_CACHE_BYTES
        + model_optimizer_bytes
        + _ACTIVATION_RESERVE_BYTES
    )
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "exactly_one_visible_device": torch.cuda.device_count() == 1,
        "visible_cuda_zero": torch.cuda.current_device() == 0,
        "source_cache_fingerprint_exact": (
            source_cache_fingerprint
            == _FROZEN_SOURCE_CACHE_FINGERPRINT
        ),
        "device_cache_fingerprint_exact": (
            _FROZEN_DEVICE_CACHE_FINGERPRINT
            == _pmope_cli.COVERAGE_STATE_PMOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT
        ),
        "optimizer_fingerprint_exact": (
            _FROZEN_OPTIMIZER_FINGERPRINT
            == _pmope_cli.COVERAGE_STATE_PMOPE_HISTORICAL_OPTIMIZER_FINGERPRINT
        ),
        "free_memory_meets_requirement": int(free_bytes) >= required,
        "total_memory_meets_requirement": int(total_bytes) >= required,
    }
    result = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-cmif-v19-uscope-bounded-400-"
                "device-memory-preflight-v1"
            ),
            "device": FROZEN_DEVICE,
            "model_class": "CURELiteCenteredMixedInteractionLevelSet",
            "model_parameter_count": model_config.expected_parameter_count,
            "source_cache_fingerprint": source_cache_fingerprint,
            "projected_device_cache_fingerprint": (
                _FROZEN_DEVICE_CACHE_FINGERPRINT
            ),
            "projected_device_cache": {
                "resident_tensor_bytes": (
                    _FROZEN_RESIDENT_DEVICE_CACHE_BYTES
                ),
                "binding_mode": (
                    "sealed_v18_static_budget_no_runtime_repack"
                ),
                "runtime_pack_count": 0,
            },
            "optimizer_config_fingerprint": (
                _FROZEN_OPTIMIZER_FINGERPRINT
            ),
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
    )
    if not result["all_pass"]:
        raise RuntimeError("CMIF device memory preflight did not pass")
    return result


def _write_checkpoint_new(
    directory: Path,
    *,
    objective: str,
    objective_policy: str,
    model: CURELiteCenteredMixedInteractionLevelSet,
) -> dict[str, object]:
    """Persist the one tensor-only USCOPE checkpoint and verify roundtrip."""

    if type(model) is not CURELiteCenteredMixedInteractionLevelSet:
        raise TypeError("CMIF checkpoint requires the exact model class")
    if objective != "uscope_joint" or objective_policy != CSLF_USCOPE_POLICY:
        raise ValueError("checkpoint must be the singleton USCOPE candidate")
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
        raise RuntimeError("CMIF checkpoint roundtrip changed")
    result = _fingerprinted(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "objective": objective,
            "objective_policy": objective_policy,
            "model_class": "CURELiteCenteredMixedInteractionLevelSet",
            "model_config": {
                "feature_channels": model.config.feature_channels,
                "feature_stride": model.config.feature_stride,
                "width": model.config.width,
                "coverage_policy": model.config.coverage_policy,
                "interaction_policy": model.config.interaction_policy,
                "energy_policy": model.config.energy_policy,
                "parameter_count": sum(
                    value.numel() for value in model.parameters()
                ),
                "fixed_margin_hex": COVERAGE_STATE_USCOPE_MARGIN.hex(),
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
    _write_new_json(directory / f"{objective}.checkpoint.json", result)
    return result


def _sealed_v18_is_read_only(payload: Mapping[str, object]) -> bool:
    return (
        payload.get("historical_negative_result") is True
        and payload.get("contemporaneous_candidate_result") is False
        and payload.get("checkpoint_treated_as_opaque_bytes") is True
        and payload.get("model_deserialization_performed") is False
        and payload.get("evaluator_called") is False
        and payload.get("training_performed") is False
        and payload.get("D_R_cached_tensor_payload_accessed") is False
        and payload.get("D_V_accessed") is False
        and payload.get("D_T_accessed") is False
        and payload.get("runtime_splits") == []
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
    """Validate static bindings without claiming output or reading real D_R."""

    source_paths = _verify_frozen_sources()
    dataset_free = run_coverage_state_uscope_dataset_free_gate()
    if not dataset_free.all_pass:
        raise RuntimeError("USCOPE dataset-free gate did not pass")
    sealed_v18 = verify_repository_coverage_state_uscope_sealed_v18(_ROOT)
    sealed_payload = sealed_v18.canonical_payload()
    if not _sealed_v18_is_read_only(sealed_payload):
        raise RuntimeError("sealed v18 negative result is not read-only")
    implementation = _implementation_binding()
    config = _static_config_payload(
        source_paths=source_paths,
        implementation=implementation,
        dataset_free_receipt_fingerprint=(
            dataset_free.receipt_fingerprint
        ),
        sealed_v18_receipt_fingerprint=(
            sealed_v18.receipt_fingerprint
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
            "sealed_v18_receipt_fingerprint": (
                sealed_v18.receipt_fingerprint
            ),
            "sealed_v18_negative_result_read_only": True,
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
            "post_training_certificate_performed": False,
            "zero_level_evaluation_performed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
            "not_a_formal_result": True,
        }
    )


def _prepare_uscope_authorization(
    preflight: object,
    dataset_free: object,
    dr_gate: object,
    *,
    sealed_v18_receipt: object,
) -> object:
    from cure_lite.experiment.coverage_state_uscope_bounded_runner import (
        prepare_coverage_state_uscope_bounded_run_authorization,
    )

    return prepare_coverage_state_uscope_bounded_run_authorization(
        preflight,
        dataset_free,
        dr_gate,
        sealed_v18_receipt=sealed_v18_receipt,
    )


def _expected_uscope_config(
    preflight: object,
) -> CoverageStateCenteredMixedInteractionConfig:
    from cure_lite.experiment.coverage_state_uscope_bounded_runner import (
        expected_coverage_state_uscope_config,
    )

    return expected_coverage_state_uscope_config(preflight)


def _run_uscope_bounded(
    authorization: object,
    model_config: CoverageStateCenteredMixedInteractionConfig,
) -> object:
    from cure_lite.experiment.coverage_state_uscope_bounded_runner import (
        run_coverage_state_cmif_uscope_bounded_400,
    )

    return run_coverage_state_cmif_uscope_bounded_400(
        authorization,
        model_config,
        device=FROZEN_DEVICE,
    )


def _dataset_free_payload(dataset_free: object) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-cmif-v19-uscope-bounded-400-"
                "dataset-free-v1"
            ),
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


def _sealed_v18_payload(sealed_v18: object) -> dict[str, object]:
    payload = sealed_v18.canonical_payload()
    if not _sealed_v18_is_read_only(payload):
        raise RuntimeError("sealed v18 negative result is not read-only")
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-cmif-v19-uscope-bounded-400-"
                "sealed-v18-negative-result-v1"
            ),
            "sealed_v18": payload,
            "sealed_v18_receipt_fingerprint": (
                sealed_v18.receipt_fingerprint
            ),
            "checkpoint_treated_as_opaque_bytes": True,
            "historical_result_retrained": False,
            "historical_result_reevaluated": False,
            "historical_result_is_candidate_gate": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def _certificate_payload(result: object) -> dict[str, object]:
    certificate = result.certificate
    certificate.verify()
    certificate_payload = certificate.canonical_payload()
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-cmif-v19-uscope-bounded-400-"
                "post-training-certificate-v1"
            ),
            "certificate": certificate_payload,
            "certificate_receipt_fingerprint": (
                stable_fingerprint(certificate_payload)
            ),
            "certificate_gate_passed": certificate.gate_passed,
            "certificate_invocations": result.certificate_invocations,
            "same_sign_response_is_gate": False,
            "optimizer_constructed": False,
            "backward_performed": False,
            "training_performed": False,
            "external_data_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def _zero_level_payload(result: object) -> dict[str, object]:
    diagnostic = result.diagnostic
    decision = result.decision
    diagnostic_payload = diagnostic.canonical_payload()
    decision_payload = decision.canonical_payload()
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-cmif-v19-uscope-bounded-400-"
                "zero-level-v1"
            ),
            "input_representation": "phase_preserving",
            "threshold": 0.0,
            "threshold_search_performed": False,
            "candidate_diagnostic": diagnostic_payload,
            "diagnostic_result_fingerprint": (
                stable_fingerprint(diagnostic_payload)
            ),
            "uscope_zero_level_decision": (
                decision_payload
            ),
            "uscope_zero_level_decision_fingerprint": (
                stable_fingerprint(decision_payload)
            ),
            "candidate_zero_level_gate_passed": (
                decision.zero_level_gate_passed
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
    if (
        len(checkpoints) != 1
        or checkpoints[0].get("objective") != "uscope_joint"
    ):
        raise ValueError("USCOPE decision requires one candidate checkpoint")
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
        raise ValueError("USCOPE result fingerprint is invalid")
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "status": (
                "USCOPE_V19_BOUNDED_400_GATE_PASS"
                if passed
                else "USCOPE_V19_BOUNDED_400_GATE_FAIL"
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
            "candidate_objective": "uscope_joint",
            "post_training_certificate_passed": (
                result.certificate.gate_passed
            ),
            "zero_level_gate_passed": (
                result.decision.zero_level_gate_passed
            ),
            "same_sign_response_is_gate": False,
            "training_invocations": result.training_invocations,
            "certificate_invocations": result.certificate_invocations,
            "zero_level_evaluation_invocations": (
                result.zero_level_evaluation_invocations
            ),
            "sealed_v18_receipt_fingerprint": (
                result.authorization.sealed_v18_receipt_fingerprint
            ),
            "historical_result_retrained": False,
            "historical_result_reevaluated": False,
            "historical_result_is_candidate_gate": False,
            "next_action": (
                "freeze_uscope_bounded_result_and_design_formal800_protocol"
                if passed
                else "freeze_uscope_v19_negative_result_and_review_structure"
            ),
            "formal800_eligible": passed,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        }
    )


def _complete_receipt_fingerprints(
    *,
    config: Mapping[str, object],
    input_receipt: Mapping[str, object],
    preflight_receipt: Mapping[str, object],
    dataset_free_receipt: Mapping[str, object],
    dr_gate_receipt: Mapping[str, object],
    sealed_v18_receipt: Mapping[str, object],
    authorization_receipt: Mapping[str, object],
    memory_receipt: Mapping[str, object],
    training_receipt: Mapping[str, object],
    certificate_receipt: Mapping[str, object],
    zero_receipt: Mapping[str, object],
    bounded_receipt: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    return {
        "config_fingerprint": config["receipt_fingerprint"],
        "input_receipt_fingerprint": input_receipt[
            "receipt_fingerprint"
        ],
        "preflight_receipt_fingerprint": preflight_receipt[
            "receipt_fingerprint"
        ],
        "dataset_free_receipt_fingerprint": dataset_free_receipt[
            "receipt_fingerprint"
        ],
        "D_R_gate_receipt_fingerprint": dr_gate_receipt[
            "receipt_fingerprint"
        ],
        "sealed_v18_receipt_fingerprint": sealed_v18_receipt[
            "receipt_fingerprint"
        ],
        "authorization_receipt_fingerprint": authorization_receipt[
            "receipt_fingerprint"
        ],
        "device_memory_preflight_receipt_fingerprint": memory_receipt[
            "receipt_fingerprint"
        ],
        "training_receipt_fingerprint": training_receipt[
            "receipt_fingerprint"
        ],
        "post_training_certificate_receipt_fingerprint": (
            certificate_receipt["receipt_fingerprint"]
        ),
        "zero_level_receipt_fingerprint": zero_receipt[
            "receipt_fingerprint"
        ],
        "bounded_result_receipt_fingerprint": bounded_receipt[
            "receipt_fingerprint"
        ],
        "decision_fingerprint": decision["receipt_fingerprint"],
    }


def run_once() -> dict[str, object]:
    """Consume the sole fixed seed-42 USCOPE bounded attempt."""

    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError(
            f"single-use bounded output already exists: {OUTPUT_PATH}"
        )

    dataset_free = run_coverage_state_uscope_dataset_free_gate()
    if not dataset_free.all_pass:
        raise PermissionError("USCOPE dataset-free gate did not pass")
    sealed_v18 = verify_repository_coverage_state_uscope_sealed_v18(_ROOT)
    if not _sealed_v18_is_read_only(sealed_v18.canonical_payload()):
        raise PermissionError("sealed v18 negative result is not read-only")
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
            sealed_v18_receipt_fingerprint=(
                sealed_v18.receipt_fingerprint
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
            "dataset_free_receipt_fingerprint": (
                dataset_free.receipt_fingerprint
            ),
            "sealed_v18_receipt_fingerprint": (
                sealed_v18.receipt_fingerprint
            ),
            "candidate_objective": "uscope_joint",
            "objectives": 1,
            "dataset_free_invocations_before_claim": 1,
            "D_R_gate_run_count_before_claim": 0,
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
        dataset_free_receipt = _dataset_free_payload(dataset_free)
        _write_new_json(
            receipts / "dataset_free.json",
            dataset_free_receipt,
        )
        sealed_receipt = _sealed_v18_payload(sealed_v18)
        _write_new_json(
            receipts / "sealed_v18_negative_result.json",
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
                    "cure-lite-cmif-v19-uscope-bounded-400-"
                    "inputs-v1"
                ),
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
                    "cure-lite-cmif-v19-uscope-bounded-400-"
                    "preflight-v1"
                ),
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
        _write_new_json(receipts / "preflight.json", preflight_receipt)
        if not preflight.training_authorized:
            raise PermissionError("bounded D_R preflight did not pass")

        dr_gate = run_coverage_state_uscope_dr_gate(
            dataset_free_receipt=dataset_free,
            real_inputs=real_inputs,
            population=population,
            device=FROZEN_DEVICE,
        )
        dr_gate_payload = dr_gate.canonical_payload()
        dr_gate_passed = bool(dr_gate.all_pass)
        dr_gate_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v19-uscope-bounded-400-"
                    "real-D_R-gate-v1"
                ),
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
                    "status": "USCOPE_V19_DR_GATE_FAIL",
                    "bounded_gate_passed": False,
                    "D_R_gate_passed": False,
                    "D_R_gate_evidence_fingerprint": (
                        dr_gate.evidence_fingerprint
                    ),
                    "authorization_created": False,
                    "bounded_training_performed": False,
                    "post_training_certificate_performed": False,
                    "zero_level_evaluation_performed": False,
                    "checkpoint_count": 0,
                    "formal800_eligible": False,
                    "formal_800_authorized": False,
                    "full_CURE_authorized": False,
                    "cross_backbone_authorized": False,
                    "performance_claim_supported": False,
                    "next_action": (
                        "freeze_D_R_gate_negative_result_and_stop_before_"
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
                    "USCOPE D_R gate-stop artifact population is incomplete"
                )
            complete = _fingerprinted(
                {
                    "schema_version": RUN_SCHEMA,
                    "status": "complete",
                    "run_id": RUN_ID,
                    "decision": decision["status"],
                    "bounded_gate_passed": False,
                    "D_R_gate_passed": False,
                    "D_R_gate_evidence_fingerprint": (
                        dr_gate.evidence_fingerprint
                    ),
                    "config_fingerprint": config[
                        "receipt_fingerprint"
                    ],
                    "input_receipt_fingerprint": input_receipt[
                        "receipt_fingerprint"
                    ],
                    "preflight_receipt_fingerprint": preflight_receipt[
                        "receipt_fingerprint"
                    ],
                    "dataset_free_receipt_fingerprint": (
                        dataset_free_receipt["receipt_fingerprint"]
                    ),
                    "sealed_v18_receipt_fingerprint": sealed_receipt[
                        "receipt_fingerprint"
                    ],
                    "D_R_gate_receipt_fingerprint": dr_gate_receipt[
                        "receipt_fingerprint"
                    ],
                    "decision_fingerprint": decision[
                        "receipt_fingerprint"
                    ],
                    "artifact_files": artifacts,
                    "artifact_file_count": len(artifacts),
                    "dataset_free_invocations": 1,
                    "real_inputs_construction_invocations": 1,
                    "population_construction_invocations": 1,
                    "preflight_invocations": 1,
                    "D_R_gate_invocations": 1,
                    "authorization_created": False,
                    "bounded_training_performed": False,
                    "post_training_certificate_performed": False,
                    "zero_level_evaluation_performed": False,
                    "checkpoint_count": 0,
                    "formal800_eligible": False,
                    "formal_800_authorized": False,
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

        authorization = _prepare_uscope_authorization(
            preflight,
            dataset_free,
            dr_gate,
            sealed_v18_receipt=sealed_v18,
        )
        model_config = _expected_uscope_config(preflight)
        if (
            type(model_config)
            is not CoverageStateCenteredMixedInteractionConfig
            or model_config.feature_channels != FROZEN_FEATURE_CHANNELS
            or model_config.feature_stride != FROZEN_FEATURE_STRIDE
            or model_config.width != FROZEN_MODEL_WIDTH
            or model_config.expected_parameter_count
            != FROZEN_PARAMETER_COUNT
        ):
            raise RuntimeError("real D_R USCOPE model config changed")
        authorization_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v19-uscope-bounded-400-"
                    "authorization-v1"
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
                "D_R_gate_evidence_fingerprint": (
                    dr_gate.evidence_fingerprint
                ),
                "sealed_v18_receipt_fingerprint": (
                    sealed_v18.receipt_fingerprint
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
            raise PermissionError("USCOPE authorization did not pass")

        memory = _device_memory_preflight(
            population.cache,
            model_config,
        )
        _write_new_json(
            receipts / "device_memory_preflight.json",
            memory,
        )
        result = _run_uscope_bounded(authorization, model_config)
        if (
            result.training_invocations != 1
            or result.certificate_invocations != 1
            or result.zero_level_evaluation_invocations != 1
            or len(result.training.results) != 1
            or len(result.training.models) != 1
            or result.training.results[0].objective != "uscope_joint"
            or result.training.models[0][0] != "uscope_joint"
        ):
            raise RuntimeError("USCOPE run returned a non-singleton result")

        checkpoint_receipts = (
            _write_checkpoint_new(
                checkpoints_dir,
                objective="uscope_joint",
                objective_policy=(
                    result.training.results[0].objective_policy
                ),
                model=result.training.models[0][1],
            ),
        )
        training_payload = result.training.canonical_payload()
        training_result_fingerprint = stable_fingerprint(training_payload)
        training_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v19-uscope-bounded-400-"
                    "training-v1"
                ),
                "training": training_payload,
                "training_result_fingerprint": (
                    training_result_fingerprint
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
                "candidate_objective": "uscope_joint",
                "historical_result_retrained": False,
                "all_models_exact_cmif_class": True,
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
        result_fingerprint = stable_fingerprint(result_payload)
        bounded_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v19-uscope-bounded-400-"
                    "result-v1"
                ),
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
                "USCOPE implementation changed during execution"
            )

        artifacts = _v15b_cli._artifact_hashes(OUTPUT_PATH)
        if len(artifacts) != FROZEN_ARTIFACT_FILE_COUNT:
            raise RuntimeError(
                "USCOPE terminal artifact population is incomplete"
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
                "D_R_gate_evidence_fingerprint": (
                    dr_gate.evidence_fingerprint
                ),
                "sealed_v18_evidence_fingerprint": (
                    sealed_v18.receipt_fingerprint
                ),
                "post_training_certificate_fingerprint": (
                    certificate_receipt[
                        "certificate_receipt_fingerprint"
                    ]
                ),
                "zero_level_decision_fingerprint": (
                    zero_receipt[
                        "uscope_zero_level_decision_fingerprint"
                    ]
                ),
                "formal800_eligible": result.bounded_gate_passed,
                **_complete_receipt_fingerprints(
                    config=config,
                    input_receipt=input_receipt,
                    preflight_receipt=preflight_receipt,
                    dataset_free_receipt=dataset_free_receipt,
                    dr_gate_receipt=dr_gate_receipt,
                    sealed_v18_receipt=sealed_receipt,
                    authorization_receipt=authorization_receipt,
                    memory_receipt=memory,
                    training_receipt=training_receipt,
                    certificate_receipt=certificate_receipt,
                    zero_receipt=zero_receipt,
                    bounded_receipt=bounded_receipt,
                    decision=decision,
                ),
                "artifact_files": artifacts,
                "artifact_file_count": len(artifacts),
                "dataset_free_invocations": 1,
                "real_inputs_construction_invocations": 1,
                "population_construction_invocations": 1,
                "preflight_invocations": 1,
                "D_R_gate_invocations": 1,
                "training_invocations": result.training_invocations,
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
                attempt_fingerprint=str(attempt["receipt_fingerprint"]),
                artifact_files=_v15b_cli._artifact_hashes(OUTPUT_PATH),
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
        validate_create_only() if args.validate_create_only else run_once()
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
