#!/usr/bin/env python3
"""Validate or execute the unique CMIF/SORR bounded-400 protocol.

``--validate-create-only`` verifies the frozen seed-42 protocol, generated
CMIF receipt, and persisted real-``D_R`` P0 authorization without loading
cached tensors or claiming output.  ``--run-once`` is the explicit,
wrapper-controlled single-use execution path.  Neither mode accesses
``D_V``/``D_T`` or authorizes Formal800.
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
from cure_lite.coverage_state_centered_mixed_interaction import (
    CMIF_ENERGY_POLICY,
    CMIF_INPUT_REPRESENTATION,
    CMIF_INTERACTION_POLICY,
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from cure_lite.coverage_state_device_cache import (
    prepare_coverage_state_device_cache,
)
from cure_lite.coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
)
from cure_lite.coverage_state_schedule import (
    coverage_state_schedule_exposure_report,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_cmif_bounded_runner import (
    COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT,
    COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT,
    COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT,
    CoverageStateCMIFBoundedRunAuthorization,
    CoverageStateCMIFBoundedRunResult,
    _current_cmif_implementation_binding,
    _verify_persisted_cmif_p0_authorization,
    expected_coverage_state_cmif_config,
    prepare_coverage_state_cmif_bounded_run_authorization,
    run_coverage_state_cmif_support_oriented_bounded_400,
)
from cure_lite.experiment.coverage_state_cmif_dataset_free import (
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_CMIF_FORMAL_WIDTH,
    run_coverage_state_cmif_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite.frozen_base import module_state_fingerprint
from cure_lite.train.coverage_state_fused_step import (
    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
    coverage_state_pair_objective_policy,
)
from tools import (
    run_coverage_state_cslf_ppce_support_oriented_bounded_400
    as _ppce_cli,
)
from tools import (
    run_coverage_state_cslf_support_oriented_bounded_400 as _v15b_cli,
)


_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "cure_lite_cmif_v17_support_oriented_bounded_400_r1"
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = _ROOT / OUTPUT_REPO_PATH
RUN_SCHEMA = "cure-lite-cmif-v17-support-oriented-bounded-400-run-v1"
VALIDATION_SCHEMA = (
    "cure-lite-cmif-v17-support-oriented-bounded-400-"
    "create-only-validation-v1"
)
ATTEMPT_SCHEMA = (
    "cure-lite-cmif-v17-support-oriented-bounded-400-attempt-v1"
)
FAILURE_SCHEMA = (
    "cure-lite-cmif-v17-support-oriented-bounded-400-failure-v1"
)
CHECKPOINT_SCHEMA = (
    "cure-lite-cmif-v17-support-oriented-bounded-400-checkpoint-v1"
)
DECISION_SCHEMA = (
    "cure-lite-cmif-v17-support-oriented-bounded-400-decision-v1"
)
FROZEN_DEVICE = _v15b_cli.FROZEN_DEVICE
FROZEN_VISIBLE_GPU = _v15b_cli.FROZEN_VISIBLE_GPU
FROZEN_CUBLAS_WORKSPACE_CONFIG = _v15b_cli.FROZEN_CUBLAS_WORKSPACE_CONFIG
FROZEN_PAUSE_TEMPERATURE_C = _v15b_cli.FROZEN_PAUSE_TEMPERATURE_C
FROZEN_RESUME_TEMPERATURE_C = _v15b_cli.FROZEN_RESUME_TEMPERATURE_C
FROZEN_CHECKPOINT_SERIALIZATION = "safetensors"
FROZEN_FEATURE_CHANNELS = COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS
FROZEN_FEATURE_STRIDE = COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE
FROZEN_MODEL_WIDTH = COVERAGE_STATE_CMIF_FORMAL_WIDTH
FROZEN_PARAMETER_COUNT = COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
FROZEN_SEED = 42
FROZEN_EPOCHS = 10
FROZEN_STEPS_PER_EPOCH = 40
FROZEN_UPDATES_PER_OBJECTIVE = 400
FROZEN_ARTIFACT_FILE_COUNT = 17
FROZEN_REAL_DR_INPUTS = _ppce_cli.FROZEN_REAL_DR_INPUTS
_ACTIVATION_RESERVE_BYTES = 2 * 1024**3
_INCOMPLETE = ".incomplete"


_fingerprinted = _ppce_cli._fingerprinted
_write_new_json = _ppce_cli._write_new_json


def _verify_frozen_sources() -> dict[str, Path]:
    """Reuse the unchanged, hash-bound real-D_R source contract."""

    return _ppce_cli._verify_frozen_sources()


def _verify_cmif_p0_authorization() -> dict[str, object]:
    """Verify the persisted four-layer P0 closure and bounded permission."""

    value = _verify_persisted_cmif_p0_authorization()
    if (
        value.get("training_authorized") is not True
        or value.get("r2_complete_fingerprint")
        != COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT
        or value.get("p0_core_receipt_fingerprint")
        != COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT
        or value.get("bounded_population_fingerprint")
        != COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT
        or value.get("D_V_accessed") is not False
        or value.get("D_T_accessed") is not False
        or value.get("training_performed") is not False
    ):
        raise PermissionError("persisted CMIF P0 did not authorize bounded-400")
    return value


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    """Bind the core closure plus this CLI and temperature wrapper."""

    core = dict(_current_cmif_implementation_binding())
    extras = (
        "tools/run_coverage_state_cmif_support_oriented_bounded_400.py",
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
                f"CMIF bounded implementation path changed: {relative}"
            )
        core[relative] = file_sha256(absolute)
    return tuple(sorted(core.items()))


def _static_config_payload(
    *,
    source_paths: Mapping[str, Path],
    implementation: tuple[tuple[str, str], ...],
    dataset_free_receipt_fingerprint: str,
    p0_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Construct the immutable seed-42 CMIF bounded configuration."""

    if (
        dataset_free_receipt_fingerprint
        != p0_evidence.get("dataset_free_receipt_fingerprint")
        or p0_evidence.get("training_authorized") is not True
    ):
        raise ValueError("CMIF dataset-free/P0 receipt binding changed")
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=FROZEN_FEATURE_CHANNELS,
        feature_stride=FROZEN_FEATURE_STRIDE,
        width=FROZEN_MODEL_WIDTH,
    )
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    parameter_count = sum(value.numel() for value in model.parameters())
    if (
        parameter_count != config.expected_parameter_count
        or parameter_count != FROZEN_PARAMETER_COUNT
    ):
        raise RuntimeError("frozen CMIF parameter count changed")
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
            "objective_suite": [
                value.value
                for value in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
            ],
            "candidate_objective": "support_oriented_response_joint",
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
        "persisted_p0_authorization": dict(p0_evidence),
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
            "bounded_400_authorized": True,
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
    """Account for one packed cache and three retained CMIF models."""

    if type(model_config) is not CoverageStateCenteredMixedInteractionConfig:
        raise TypeError("CMIF memory preflight requires exact config class")
    projected = prepare_coverage_state_device_cache(cache, device="cpu")
    projected.verify_unchanged(verify_content=True, verify_source=False)
    projected_payload = projected.resident_tensor_bytes
    projected_report = projected.memory_report()
    projected_fingerprint = projected.device_cache_fingerprint
    source_cache_fingerprint = projected.source_cache_fingerprint
    del projected
    gc.collect()
    model = CURELiteCenteredMixedInteractionLevelSet(model_config)
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
    required = projected_payload + model_optimizer_bytes + _ACTIVATION_RESERVE_BYTES
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
                "cure-lite-cmif-v17-support-oriented-bounded-400-"
                "device-memory-preflight-v1"
            ),
            "device": FROZEN_DEVICE,
            "model_class": "CURELiteCenteredMixedInteractionLevelSet",
            "model_parameter_count": model_config.expected_parameter_count,
            "source_cache_fingerprint": source_cache_fingerprint,
            "projected_cpu_pack_fingerprint": projected_fingerprint,
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
    """Persist one tensor-only CMIF checkpoint and verify roundtrip."""

    if type(model) is not CURELiteCenteredMixedInteractionLevelSet:
        raise TypeError("CMIF checkpoint requires the exact model class")
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


def _zero_level_payload(
    result: CoverageStateCMIFBoundedRunResult,
    authorization: CoverageStateCMIFBoundedRunAuthorization,
) -> dict[str, object]:
    candidate_gate = dict(result.checks)[
        "candidate_original_zero_level_gates"
    ]
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-cmif-v17-support-oriented-bounded-400-"
                "zero-level-v1"
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
    result: CoverageStateCMIFBoundedRunResult,
    checkpoints: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Qualify only the frozen candidate; controls remain diagnostic."""

    passed = result.bounded_gate_passed
    candidate = dict(result.diagnostics)[
        result.authorization.candidate_objective
    ]
    compact_only_failure = (
        not passed
        and candidate.factual_miss_gate_passed
        and candidate.factual_no_miss_gate_passed
        and candidate.clean_defined_metrics_passed
        and not candidate.clean_compact_support_gate_passed
        and candidate.component_null_gate_passed
        and candidate.identity_null_gate_passed
        and candidate.diagnostic_null_gate_passed
    )
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "status": (
                "CMIF_V17_BOUNDED_400_GATE_PASS"
                if passed
                else "CMIF_V17_BOUNDED_400_GATE_FAIL"
            ),
            "bounded_gate_passed": passed,
            "failed_checks": list(result.failed_checks),
            "result_fingerprint": result.result_fingerprint,
            "checkpoint_receipt_fingerprints": {
                str(value["objective"]): str(value["receipt_fingerprint"])
                for value in checkpoints
            },
            "candidate_objective": result.authorization.candidate_objective,
            "candidate_gate_passed": dict(result.checks)[
                "candidate_original_zero_level_gates"
            ],
            "control_outcomes_are_not_candidate_gates": True,
            "persisted_p0_evidence_fingerprint": (
                result.authorization.p0_evidence_fingerprint
            ),
            "next_action": (
                "freeze_cmif_bounded_result_and_review_formal800_prerequisites"
                if passed
                else (
                    "freeze_v17_and_review_objective_only_in_independent_v18"
                    if compact_only_failure
                    else (
                        "freeze_cmif_v17_negative_result_and_attribute_"
                        "failed_gates"
                    )
                )
            ),
            "compact_gate_only_failure": compact_only_failure,
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
    """Validate all static bindings without claiming or loading D_R."""

    source_paths = _verify_frozen_sources()
    p0 = _verify_cmif_p0_authorization()
    dataset_free = run_coverage_state_cmif_dataset_free_gate()
    if not dataset_free.all_pass:
        raise RuntimeError("CMIF dataset-free gate did not pass")
    implementation = _implementation_binding()
    config = _static_config_payload(
        source_paths=source_paths,
        implementation=implementation,
        dataset_free_receipt_fingerprint=dataset_free.receipt_fingerprint,
        p0_evidence=p0,
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
            "persisted_p0_authorization": p0,
            "bounded_400_authorized": True,
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
    """Execute the sole wrapper-controlled D_R attempt; never resume."""

    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError(
            f"single-use bounded output already exists: {OUTPUT_PATH}"
        )
    source_paths = _verify_frozen_sources()
    p0 = _verify_cmif_p0_authorization()
    implementation = _implementation_binding()
    runtime = _v15b_cli._verify_runtime_contract()
    dataset_free = run_coverage_state_cmif_dataset_free_gate()
    if not dataset_free.all_pass:
        raise PermissionError("CMIF dataset-free gate did not authorize")
    config = _fingerprinted(
        _static_config_payload(
            source_paths=source_paths,
            implementation=implementation,
            dataset_free_receipt_fingerprint=(
                dataset_free.receipt_fingerprint
            ),
            p0_evidence=p0,
        )
    )
    attempt = _fingerprinted(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "run_id": RUN_ID,
            "output_repo_path": OUTPUT_REPO_PATH,
            "config_fingerprint": config["receipt_fingerprint"],
            "runtime": runtime,
            "persisted_p0_authorization": p0,
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
                    "cure-lite-cmif-v17-support-oriented-bounded-400-"
                    "inputs-v1"
                ),
                "real_D_R_inputs": real_inputs.canonical_payload(),
                "source_binding": (
                    real_inputs.source_binding.canonical_payload()
                ),
                "bounded_population": population.canonical_payload(),
                "population_fingerprint": population.population_fingerprint,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(receipts / "inputs.json", input_receipt)
        preflight_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v17-support-oriented-bounded-400-"
                    "preflight-v1"
                ),
                "preflight": preflight.canonical_payload(),
                "schedule": preflight.schedule.canonical_payload(),
                "schedule_selections": [
                    value.canonical_payload()
                    for value in preflight.schedule.selections
                ],
                "exposure": exposure,
                "training_authorized": preflight.training_authorized,
                "formal_800_authorized": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(receipts / "preflight.json", preflight_receipt)
        if not preflight.training_authorized:
            raise PermissionError("bounded D_R preflight did not authorize")
        dataset_free_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v17-support-oriented-bounded-400-"
                    "dataset-free-v1"
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
            prepare_coverage_state_cmif_bounded_run_authorization(
                preflight,
                dataset_free,
            )
        )
        model_config = expected_coverage_state_cmif_config(preflight)
        if (
            type(model_config)
            is not CoverageStateCenteredMixedInteractionConfig
            or model_config.feature_channels != FROZEN_FEATURE_CHANNELS
            or model_config.feature_stride != FROZEN_FEATURE_STRIDE
            or model_config.width != FROZEN_MODEL_WIDTH
            or model_config.expected_parameter_count
            != FROZEN_PARAMETER_COUNT
        ):
            raise RuntimeError("real D_R CMIF model config changed")
        authorization_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v17-support-oriented-bounded-400-"
                    "authorization-v1"
                ),
                "authorization": authorization.canonical_payload(),
                "authorization_fingerprint": (
                    authorization.authorization_fingerprint
                ),
                "runtime_implementation_fingerprint": stable_fingerprint(
                    dict(implementation)
                ),
                "config_receipt_fingerprint": config[
                    "receipt_fingerprint"
                ],
                "persisted_p0_authorization": p0,
                "training_authorized": authorization.training_authorized,
                "formal_800_authorized": False,
            }
        )
        _write_new_json(
            receipts / "authorization.json",
            authorization_receipt,
        )
        if not authorization.training_authorized:
            raise PermissionError("CMIF authorization did not pass")
        memory = _device_memory_preflight(population.cache, model_config)
        _write_new_json(
            receipts / "device_memory_preflight.json",
            memory,
        )
        result = run_coverage_state_cmif_support_oriented_bounded_400(
            authorization,
            model_config,
            device=FROZEN_DEVICE,
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
                    "cure-lite-cmif-v17-support-oriented-bounded-400-"
                    "training-v1"
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
                "all_models_exact_cmif_class": True,
                "all_models_parameter_count": FROZEN_PARAMETER_COUNT,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(receipts / "training.json", training_receipt)
        zero_receipt = _zero_level_payload(result, authorization)
        _write_new_json(receipts / "zero_level.json", zero_receipt)
        bounded_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v17-support-oriented-bounded-400-"
                    "result-v1"
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
            raise RuntimeError("CMIF implementation changed during execution")
        if _verify_frozen_sources() != source_paths:
            raise RuntimeError("frozen D_R source paths changed")
        if _verify_cmif_p0_authorization() != p0:
            raise RuntimeError("persisted CMIF P0 closure changed")
        replay_dataset_free = run_coverage_state_cmif_dataset_free_gate()
        if (
            replay_dataset_free.receipt_fingerprint
            != dataset_free.receipt_fingerprint
        ):
            raise RuntimeError("CMIF dataset-free receipt changed")

        artifacts = _v15b_cli._artifact_hashes(OUTPUT_PATH)
        if len(artifacts) != FROZEN_ARTIFACT_FILE_COUNT:
            raise RuntimeError(
                "CMIF terminal artifact population is incomplete"
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
                "persisted_p0_authorization": p0,
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
        help="consume the unique wrapper-controlled bounded-400 attempt",
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
    if args.run_once and result["bounded_gate_passed"] is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
