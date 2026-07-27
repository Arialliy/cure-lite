#!/usr/bin/env python3
"""Validate or execute the unique CMIF/PMOPE bounded-400 protocol.

``--validate-create-only`` verifies only static seed-42 bindings without
loading real-``D_R`` tensors, running the real-``D_R`` gate, or claiming
output. ``--run-once`` computes the real-``D_R`` gate exactly once in memory,
authorizes exactly one ``pmope_joint`` candidate, and consumes the sole
wrapper-controlled bounded attempt. Neither mode accesses ``D_V``/``D_T`` or
executes or authorizes Formal800.
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
from cure_lite.coverage_state_sobolev import (
    CSLF_PMOPE_POLICY,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_pmope_bounded_runner import (
    COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_PMOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT,
    COVERAGE_STATE_PMOPE_HISTORICAL_OPTIMIZER_FINGERPRINT,
    CoverageStatePMOPEBoundedRunAuthorization,
    CoverageStatePMOPEBoundedRunResult,
    _current_implementation_binding,
    expected_coverage_state_pmope_config,
    prepare_coverage_state_pmope_bounded_run_authorization,
    run_coverage_state_cmif_pmope_bounded_400,
    verify_current_sealed_v17_controls,
)
from cure_lite.experiment.coverage_state_pmope_dataset_free import (
    COVERAGE_STATE_PMOPE_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_PMOPE_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_PMOPE_FORMAL_WIDTH,
    COVERAGE_STATE_PMOPE_MARGIN,
    run_coverage_state_pmope_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_pmope_dr_gate import (
    run_coverage_state_pmope_dr_gate,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite.experiment.coverage_state_training import (
    CoverageStateMatchedTrainingConfig,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite.frozen_base import module_state_fingerprint
from tools import (
    run_coverage_state_cslf_ppce_support_oriented_bounded_400
    as _ppce_cli,
)
from tools import (
    run_coverage_state_cslf_support_oriented_bounded_400 as _v15b_cli,
)


_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "cure_lite_cmif_v18_pmope_bounded_400_r1"
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = _ROOT / OUTPUT_REPO_PATH
RUN_SCHEMA = "cure-lite-cmif-v18-pmope-bounded-400-run-v1"
VALIDATION_SCHEMA = (
    "cure-lite-cmif-v18-pmope-bounded-400-create-only-validation-v1"
)
ATTEMPT_SCHEMA = "cure-lite-cmif-v18-pmope-bounded-400-attempt-v1"
FAILURE_SCHEMA = "cure-lite-cmif-v18-pmope-bounded-400-failure-v1"
CHECKPOINT_SCHEMA = "cure-lite-cmif-v18-pmope-bounded-400-checkpoint-v1"
DECISION_SCHEMA = "cure-lite-cmif-v18-pmope-bounded-400-decision-v1"
FROZEN_DEVICE = _v15b_cli.FROZEN_DEVICE
FROZEN_VISIBLE_GPU = _v15b_cli.FROZEN_VISIBLE_GPU
FROZEN_CUBLAS_WORKSPACE_CONFIG = _v15b_cli.FROZEN_CUBLAS_WORKSPACE_CONFIG
FROZEN_PAUSE_TEMPERATURE_C = _v15b_cli.FROZEN_PAUSE_TEMPERATURE_C
FROZEN_RESUME_TEMPERATURE_C = _v15b_cli.FROZEN_RESUME_TEMPERATURE_C
FROZEN_CHECKPOINT_SERIALIZATION = "safetensors"
FROZEN_FEATURE_CHANNELS = COVERAGE_STATE_PMOPE_FORMAL_FEATURE_CHANNELS
FROZEN_FEATURE_STRIDE = COVERAGE_STATE_PMOPE_FORMAL_FEATURE_STRIDE
FROZEN_MODEL_WIDTH = COVERAGE_STATE_PMOPE_FORMAL_WIDTH
FROZEN_PARAMETER_COUNT = COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT
FROZEN_SEED = 42
FROZEN_EPOCHS = 10
FROZEN_STEPS_PER_EPOCH = 40
FROZEN_UPDATES_PER_OBJECTIVE = 400
FROZEN_ARTIFACT_FILE_COUNT = 15
FROZEN_DR_GATE_STOP_ARTIFACT_FILE_COUNT = 8
FROZEN_REAL_DR_INPUTS = _ppce_cli.FROZEN_REAL_DR_INPUTS
_ACTIVATION_RESERVE_BYTES = 2 * 1024**3
_INCOMPLETE = ".incomplete"


_fingerprinted = _ppce_cli._fingerprinted
_write_new_json = _ppce_cli._write_new_json


def _verify_frozen_sources() -> dict[str, Path]:
    """Reuse the unchanged, hash-bound real-D_R source contract."""

    return _ppce_cli._verify_frozen_sources()


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    """Bind the core closure plus this CLI and temperature wrapper."""

    core = dict(_current_implementation_binding())
    extras = (
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
                f"CMIF bounded implementation path changed: {relative}"
            )
        core[relative] = file_sha256(absolute)
    return tuple(sorted(core.items()))


def _static_config_payload(
    *,
    source_paths: Mapping[str, Path],
    implementation: tuple[tuple[str, str], ...],
    dataset_free_receipt_fingerprint: str,
    sealed_v17_receipt_fingerprint: str,
) -> dict[str, object]:
    """Construct the immutable singleton seed-42 PMOPE configuration."""

    if (
        len(dataset_free_receipt_fingerprint) != 64
        or len(sealed_v17_receipt_fingerprint) != 64
    ):
        raise ValueError("PMOPE prerequisite fingerprint changed")
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=FROZEN_FEATURE_CHANNELS,
        feature_stride=FROZEN_FEATURE_STRIDE,
        width=FROZEN_MODEL_WIDTH,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FROZEN_SEED)
        model = CURELiteCenteredMixedInteractionLevelSet(config)
        parameter_count = sum(
            value.numel() for value in model.parameters()
        )
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
            "objective_suite": ["pmope_joint"],
            "candidate_objective": "pmope_joint",
            "candidate_objective_policy": CSLF_PMOPE_POLICY,
            "fixed_margin_hex": COVERAGE_STATE_PMOPE_MARGIN.hex(),
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
        "historical_v17_controls": {
            "receipt_fingerprint": sealed_v17_receipt_fingerprint,
            "read_only": True,
            "retrained": False,
            "reevaluated": False,
            "candidate_gates": False,
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
    """Account for one packed cache and one PMOPE model/optimizer."""

    if type(model_config) is not CoverageStateCenteredMixedInteractionConfig:
        raise TypeError("CMIF memory preflight requires exact config class")
    if FROZEN_DEVICE != "cuda:0":
        raise RuntimeError("PMOPE bounded execution requires cuda:0")
    projected = prepare_coverage_state_device_cache(
        cache,
        device=FROZEN_DEVICE,
    )
    projected.verify_unchanged(verify_content=True, verify_source=False)
    projected_payload = projected.resident_tensor_bytes
    projected_report = projected.memory_report()
    projected_fingerprint = projected.device_cache_fingerprint
    source_cache_fingerprint = projected.source_cache_fingerprint
    del projected
    gc.collect()
    model = CURELiteCenteredMixedInteractionLevelSet(model_config)
    training_config = CoverageStateMatchedTrainingConfig(
        seed=FROZEN_SEED
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(
            training_config.adam_beta1,
            training_config.adam_beta2,
        ),
        eps=training_config.adam_epsilon,
        weight_decay=training_config.weight_decay,
    )
    optimizer_fingerprint = coverage_state_optimizer_config_fingerprint(
        model,
        optimizer,
    )
    parameter_bytes = sum(
        value.numel() * value.element_size()
        for value in model.parameters()
    )
    buffer_bytes = sum(
        value.numel() * value.element_size()
        for value in model.buffers()
    )
    del optimizer
    del model
    model_optimizer_bytes = 4 * parameter_bytes + buffer_bytes
    required = projected_payload + model_optimizer_bytes + _ACTIVATION_RESERVE_BYTES
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "exactly_one_visible_device": torch.cuda.device_count() == 1,
        "visible_cuda_zero": torch.cuda.current_device() == 0,
        "device_cache_fingerprint_exact": (
            projected_fingerprint
            == COVERAGE_STATE_PMOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT
        ),
        "optimizer_fingerprint_exact": (
            optimizer_fingerprint
            == COVERAGE_STATE_PMOPE_HISTORICAL_OPTIMIZER_FINGERPRINT
        ),
        "free_memory_meets_requirement": int(free_bytes) >= required,
        "total_memory_meets_requirement": int(total_bytes) >= required,
    }
    result = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-"
                "device-memory-preflight-v1"
            ),
            "device": FROZEN_DEVICE,
            "model_class": "CURELiteCenteredMixedInteractionLevelSet",
            "model_parameter_count": model_config.expected_parameter_count,
            "source_cache_fingerprint": source_cache_fingerprint,
            "projected_device_cache_fingerprint": projected_fingerprint,
            "projected_device_cache": projected_report,
            "optimizer_config_fingerprint": optimizer_fingerprint,
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
    if objective != "pmope_joint" or objective_policy != CSLF_PMOPE_POLICY:
        raise ValueError("checkpoint must be the singleton PMOPE candidate")
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
                "fixed_margin_hex": COVERAGE_STATE_PMOPE_MARGIN.hex(),
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
    result: CoverageStatePMOPEBoundedRunResult,
    authorization: CoverageStatePMOPEBoundedRunAuthorization,
) -> dict[str, object]:
    candidate_gate = dict(result.checks)[
        "candidate_seven_zero_level_gates"
    ]
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-"
                "zero-level-v1"
            ),
            "input_representation": "phase_preserving",
            "threshold": 0.0,
            "threshold_search_performed": False,
            "candidate_diagnostic": result.diagnostic.canonical_payload(),
            "candidate_objective": authorization.candidate_objective,
            "candidate_bounded_gate_passed": candidate_gate,
            "historical_v17_controls": (
                authorization.sealed_v17_receipt.canonical_payload()
            ),
            "historical_controls_retrained": False,
            "historical_controls_reevaluated": False,
            "historical_control_outcomes_are_candidate_gates": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def _decision_payload(
    result: CoverageStatePMOPEBoundedRunResult,
    checkpoints: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Qualify only PMOPE; historical v17 controls remain read-only."""

    passed = result.bounded_gate_passed
    if (
        len(checkpoints) != 1
        or checkpoints[0].get("objective") != "pmope_joint"
    ):
        raise ValueError("PMOPE decision requires one candidate checkpoint")
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "status": (
                "PMOPE_V18_BOUNDED_400_GATE_PASS"
                if passed
                else "PMOPE_V18_BOUNDED_400_GATE_FAIL"
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
                "candidate_seven_zero_level_gates"
            ],
            "sealed_v17_receipt_fingerprint": (
                result.authorization.sealed_v17_receipt_fingerprint
            ),
            "historical_controls_retrained": False,
            "historical_controls_reevaluated": False,
            "historical_control_outcomes_are_candidate_gates": False,
            "next_action": (
                "freeze_pmope_bounded_result_and_design_formal800_protocol"
                if passed
                else "freeze_pmope_v18_negative_result_and_review_structure"
            ),
            "formal800_eligible": passed,
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


def _complete_receipt_fingerprints(
    *,
    config: Mapping[str, object],
    input_receipt: Mapping[str, object],
    preflight_receipt: Mapping[str, object],
    dataset_free_receipt: Mapping[str, object],
    dr_gate_receipt: Mapping[str, object],
    sealed_v17_receipt: Mapping[str, object],
    authorization_receipt: Mapping[str, object],
    memory_receipt: Mapping[str, object],
    training_receipt: Mapping[str, object],
    zero_receipt: Mapping[str, object],
    bounded_receipt: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    """Return the complete singleton PMOPE receipt graph."""

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
        "sealed_v17_receipt_fingerprint": sealed_v17_receipt[
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
        "zero_level_receipt_fingerprint": zero_receipt[
            "receipt_fingerprint"
        ],
        "bounded_result_receipt_fingerprint": bounded_receipt[
            "receipt_fingerprint"
        ],
        "decision_fingerprint": decision["receipt_fingerprint"],
    }


def validate_create_only() -> dict[str, object]:
    """Validate static bindings without claiming or touching real D_R."""

    source_paths = _verify_frozen_sources()
    dataset_free = run_coverage_state_pmope_dataset_free_gate()
    if not dataset_free.all_pass:
        raise RuntimeError("PMOPE dataset-free gate did not pass")
    sealed_v17 = verify_current_sealed_v17_controls()
    sealed_payload = sealed_v17.canonical_payload()
    if (
        sealed_payload["historical_frozen_controls"] is not True
        or sealed_payload["contemporaneous_controls"] is not False
        or sealed_payload[
            "control_outcomes_are_not_candidate_gates"
        ]
        is not True
        or sealed_payload["model_deserialization_performed"] is not False
        or sealed_payload["evaluator_called"] is not False
        or sealed_payload["training_performed"] is not False
    ):
        raise RuntimeError("sealed v17 controls are not read-only")
    implementation = _implementation_binding()
    config = _static_config_payload(
        source_paths=source_paths,
        implementation=implementation,
        dataset_free_receipt_fingerprint=dataset_free.receipt_fingerprint,
        sealed_v17_receipt_fingerprint=(
            sealed_v17.receipt_fingerprint
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
            "sealed_v17_receipt_fingerprint": (
                sealed_v17.receipt_fingerprint
            ),
            "historical_v17_controls_read_only": True,
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
    implementation = _implementation_binding()
    runtime = _v15b_cli._verify_runtime_contract()
    dataset_free = run_coverage_state_pmope_dataset_free_gate()
    if not dataset_free.all_pass:
        raise PermissionError("PMOPE dataset-free gate did not pass")
    sealed_v17 = verify_current_sealed_v17_controls()
    config = _fingerprinted(
        _static_config_payload(
            source_paths=source_paths,
            implementation=implementation,
            dataset_free_receipt_fingerprint=(
                dataset_free.receipt_fingerprint
            ),
            sealed_v17_receipt_fingerprint=(
                sealed_v17.receipt_fingerprint
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
            "sealed_v17_receipt_fingerprint": (
                sealed_v17.receipt_fingerprint
            ),
            "candidate_objective": "pmope_joint",
            "objectives": 1,
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
                    "cure-lite-cmif-v18-pmope-bounded-400-"
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
                    "cure-lite-cmif-v18-pmope-bounded-400-"
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
                    "cure-lite-cmif-v18-pmope-bounded-400-"
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
        sealed_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v18-pmope-bounded-400-"
                    "sealed-v17-controls-v1"
                ),
                "sealed_v17": sealed_v17.canonical_payload(),
                "sealed_v17_receipt_fingerprint": (
                    sealed_v17.receipt_fingerprint
                ),
                "historical_controls_retrained": False,
                "historical_controls_reevaluated": False,
                "historical_control_outcomes_are_candidate_gates": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(
            receipts / "sealed_v17_controls.json",
            sealed_receipt,
        )
        dr_gate = run_coverage_state_pmope_dr_gate(
            dataset_free_receipt=dataset_free,
            real_inputs=real_inputs,
            bounded_population=population,
        )
        dr_gate_payload = dr_gate.canonical_payload()
        dr_gate_evidence_fingerprint = stable_fingerprint(
            dr_gate_payload
        )
        dr_gate_passed = bool(dr_gate.checks) and all(
            value for _, value in dr_gate.checks
        )
        dr_gate_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v18-pmope-bounded-400-"
                    "real-D_R-gate-v1"
                ),
                "D_R_gate": dr_gate_payload,
                "D_R_gate_evidence_fingerprint": (
                    dr_gate_evidence_fingerprint
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
                    "status": "PMOPE_V18_DR_GATE_FAIL",
                    "bounded_gate_passed": False,
                    "D_R_gate_passed": False,
                    "D_R_gate_evidence_fingerprint": (
                        dr_gate_evidence_fingerprint
                    ),
                    "authorization_created": False,
                    "bounded_training_performed": False,
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
            if (
                len(artifacts)
                != FROZEN_DR_GATE_STOP_ARTIFACT_FILE_COUNT
            ):
                raise RuntimeError(
                    "PMOPE D_R gate-stop artifact population is incomplete"
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
                        dr_gate_evidence_fingerprint
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
                    "sealed_v17_receipt_fingerprint": sealed_receipt[
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
                    "authorization_created": False,
                    "bounded_training_performed": False,
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
        authorization = (
            prepare_coverage_state_pmope_bounded_run_authorization(
                preflight,
                dataset_free,
                dr_gate,
                sealed_v17_receipt=sealed_v17,
                dr_gate_canonical_payload=dr_gate_payload,
            )
        )
        model_config = expected_coverage_state_pmope_config(preflight)
        if (
            type(model_config)
            is not CoverageStateCenteredMixedInteractionConfig
            or model_config.feature_channels != FROZEN_FEATURE_CHANNELS
            or model_config.feature_stride != FROZEN_FEATURE_STRIDE
            or model_config.width != FROZEN_MODEL_WIDTH
            or model_config.expected_parameter_count
            != FROZEN_PARAMETER_COUNT
        ):
            raise RuntimeError("real D_R PMOPE model config changed")
        authorization_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v18-pmope-bounded-400-"
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
                "D_R_gate_evidence_fingerprint": (
                    dr_gate_evidence_fingerprint
                ),
                "sealed_v17_receipt_fingerprint": (
                    sealed_v17.receipt_fingerprint
                ),
                "training_authorized": authorization.training_authorized,
                "formal_800_authorized": False,
            }
        )
        _write_new_json(
            receipts / "authorization.json",
            authorization_receipt,
        )
        if not authorization.training_authorized:
            raise PermissionError("PMOPE authorization did not pass")
        memory = _device_memory_preflight(population.cache, model_config)
        _write_new_json(
            receipts / "device_memory_preflight.json",
            memory,
        )
        result = run_coverage_state_cmif_pmope_bounded_400(
            authorization,
            model_config,
            device=FROZEN_DEVICE,
        )
        if (
            len(result.training.results) != 1
            or len(result.training.models) != 1
            or result.training.results[0].objective != "pmope_joint"
            or result.training.models[0][0] != "pmope_joint"
        ):
            raise RuntimeError("PMOPE run returned a non-singleton result")
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
        training_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cmif-v18-pmope-bounded-400-"
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
                "candidate_count": 1,
                "candidate_objective": "pmope_joint",
                "historical_controls_retrained": False,
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
                    "cure-lite-cmif-v18-pmope-bounded-400-"
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
        result.verify_unchanged()
        if _implementation_binding() != implementation:
            raise RuntimeError("PMOPE implementation changed during execution")
        if _verify_frozen_sources() != source_paths:
            raise RuntimeError("frozen D_R source paths changed")
        replay_dataset_free = run_coverage_state_pmope_dataset_free_gate()
        if (
            replay_dataset_free.receipt_fingerprint
            != dataset_free.receipt_fingerprint
        ):
            raise RuntimeError("PMOPE dataset-free receipt changed")
        replay_sealed = verify_current_sealed_v17_controls()
        if (
            replay_sealed.receipt_fingerprint
            != sealed_v17.receipt_fingerprint
        ):
            raise RuntimeError("sealed v17 controls changed")

        artifacts = _v15b_cli._artifact_hashes(OUTPUT_PATH)
        if len(artifacts) != FROZEN_ARTIFACT_FILE_COUNT:
            raise RuntimeError(
                "PMOPE terminal artifact population is incomplete"
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
                    dr_gate_evidence_fingerprint
                ),
                "sealed_v17_evidence_fingerprint": (
                    sealed_v17.receipt_fingerprint
                ),
                "formal800_eligible": result.bounded_gate_passed,
                **_complete_receipt_fingerprints(
                    config=config,
                    input_receipt=input_receipt,
                    preflight_receipt=preflight_receipt,
                    dataset_free_receipt=dataset_free_receipt,
                    dr_gate_receipt=dr_gate_receipt,
                    sealed_v17_receipt=sealed_receipt,
                    authorization_receipt=authorization_receipt,
                    memory_receipt=memory,
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
