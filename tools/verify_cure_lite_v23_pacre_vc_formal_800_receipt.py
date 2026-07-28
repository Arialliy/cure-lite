#!/usr/bin/env python3
"""Independently verify the sole PACRE-VC v23 Formal800 terminal graph.

The verifier is intentionally stricter than a checksum reader.  It checks the
closed artifact population, replays the terminal v23 D_R verifier, verifies the
live source/runtime prerequisites, reconstructs the full D_R inputs and the
seed-42 Formal800 authorization, reconstructs the common training-result
object, and strictly reloads the final-only safetensors artifact.  It never
opens D_V or D_T and cannot make a performance claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import isfinite
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_sobolev import CSLF_PMOPE_POLICY
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_TRAINING_RESULT_SCHEMA,
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite_v23.authorization import (
    frozen_real_dr_source_paths,
    protocol_root,
)
from cure_lite_v23.bounded_runner import (
    PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG,
    PACRE_BOUNDED_OUTPUT_PATH,
    PACRE_BOUNDED_PAUSE_TEMPERATURE_C,
    PACRE_BOUNDED_RESUME_TEMPERATURE_C,
    PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256,
    PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH,
    PACRE_BOUNDED_VISIBLE_GPU,
)
from cure_lite_v23.dr_gate import (
    PACRE_VC_DR_CHECK_NAMES,
    PACRE_VC_DR_PASS_DECISION,
    CoverageStatePACREDRGateReceipt,
    pacre_vc_dr_receipt_from_payload,
)
from cure_lite_v23.environment import (
    stabilize_pacre_vc_numerical_runtime,
)
from cure_lite_v23.factory import PACRE_VC_PARAMETER_NAMES
from cure_lite_v23.formal_artifacts import (
    PACRE_VC_FORMAL_MODEL_ARTIFACT_SCHEMA,
    LoadedPACREVCFormalArtifact,
    VerifiedPACREVCFormalTerminal,
    _issue_verified_pacre_vc_formal_terminal,
    formal_training_ledger_payload,
    load_pacre_vc_formal_final_model,
)
from cure_lite_v23.formal_training import (
    PACRE_VC_FORMAL_AUTHORIZATION_SCHEMA,
    PACRE_VC_FORMAL_DEVICE,
    PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT,
    PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT,
    PACRE_VC_FORMAL_RESULT_SCHEMA,
    PACRE_VC_FORMAL_RUN_ID,
    PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT,
    PACRE_VC_FORMAL_SEED,
    PACRE_VC_FORMAL_UPDATES,
    expected_pacre_vc_formal_config,
    prepare_pacre_vc_formal_800_authorization,
)
from cure_lite_v23.protocol import (
    read_strict_json,
    strict_json_bytes,
    verify_fingerprinted,
    verify_source_closure,
)
from cure_lite_v23.training import (
    PACRE_PMOPE_OBJECTIVE,
    PACRE_PMOPE_TRAINING_CONFIG,
)
from tools.verify_cure_lite_v23_pacre_vc_dr_receipt import (
    verify_terminal as verify_dr_terminal,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = PACRE_VC_FORMAL_RUN_ID
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = ROOT / OUTPUT_REPO_PATH
BOUNDED_OUTPUT_PATH = PACRE_BOUNDED_OUTPUT_PATH

FROZEN_DEVICE = PACRE_VC_FORMAL_DEVICE
FROZEN_VISIBLE_GPU = PACRE_BOUNDED_VISIBLE_GPU
FROZEN_CUBLAS_WORKSPACE_CONFIG = PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG
FROZEN_PYTHONHASHSEED = "0"
FROZEN_PAUSE_TEMPERATURE_C = PACRE_BOUNDED_PAUSE_TEMPERATURE_C
FROZEN_RESUME_TEMPERATURE_C = PACRE_BOUNDED_RESUME_TEMPERATURE_C
FROZEN_SEED = PACRE_VC_FORMAL_SEED
FROZEN_EPOCHS = 800
FROZEN_STEPS_PER_EPOCH = 40
FROZEN_UPDATES = PACRE_VC_FORMAL_UPDATES
FROZEN_PARAMETER_COUNT = 64_064

ATTEMPT_SCHEMA = "cure-lite-v23-pacre-vc-formal800-cli-attempt-v1"
DR_TERMINAL_SCHEMA = (
    "cure-lite-v23-pacre-vc-formal800-D_R-terminal-binding-v1"
)
CONFIG_SCHEMA = "cure-lite-v23-pacre-vc-formal800-config-v1"
INPUTS_SCHEMA = "cure-lite-v23-pacre-vc-formal800-inputs-v1"
AUTHORIZATION_WRAPPER_SCHEMA = (
    "cure-lite-v23-pacre-vc-formal800-authorization-wrapper-v1"
)
EPOCH_PROGRESS_SCHEMA = (
    "cure-lite-v23-pacre-vc-formal800-epoch-progress-v1"
)
TRAINING_SCHEMA = "cure-lite-v23-pacre-vc-formal800-training-v1"
DECISION_SCHEMA = "cure-lite-v23-pacre-vc-formal800-decision-v1"
COMPLETE_SCHEMA = "cure-lite-v23-pacre-vc-formal800-complete-v1"
VERIFICATION_SCHEMA = (
    "cure-lite-v23-pacre-vc-formal800-terminal-verification-v1"
)
TERMINAL_STATUS = (
    "FORMAL800_TRAINING_COMPLETE_D_V_PREREGISTRATION_ELIGIBLE"
)
NEXT_ACTION = (
    "validate_and_run_fixed_adaptive_D_V_under_"
    "existing_preregistration"
)

EXPECTED_DIRECTORIES = frozenset({"receipts", "final_model"})
EXPECTED_ARTIFACT_FILES = frozenset(
    {
        "attempt.json",
        "receipts/dr_terminal_verification.json",
        "receipts/config.json",
        "receipts/inputs.json",
        "receipts/authorization.json",
        "receipts/epoch_progress.jsonl",
        "receipts/training.json",
        "receipts/decision.json",
        "final_model/model.safetensors",
        "final_model/artifact.json",
    }
)

_ATTEMPT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "output_repo_path",
        "candidate",
        "objective",
        "runtime",
        "budget",
        "source_closure_fingerprint",
        "dataset_free_receipt_fingerprint",
        "D_R_terminal_verification_fingerprint",
        "D_R_gate_receipt_fingerprint",
        "D_R_gate_check_count",
        "D_R_gate_passed",
        "bounded_400_output_absent",
        "bounded_400_required",
        "bounded_400_authorization_effect",
        "single_attempt",
        "resume_allowed",
        "automatic_retry_allowed",
        "overwrite_allowed",
        "checkpoint_policy",
        "D_R_receipt_metadata_read",
        "D_R_tensor_payload_accessed",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_claim_supported",
        "receipt_fingerprint",
    }
)
_DR_TERMINAL_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "attempt_fingerprint",
        "verification",
        "verification_fingerprint",
        "D_R_gate_receipt_fingerprint",
        "D_R_gate_check_count",
        "D_R_gate_passed",
        "D_R_reopened",
        "D_R_tensor_payload_accessed",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "training_performed",
        "receipt_fingerprint",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "candidate",
        "split",
        "runtime_splits",
        "objective",
        "model_config",
        "parameter_count",
        "field_threshold_hex",
        "threshold_search_performed",
        "budget",
        "runtime",
        "checkpoint_policy",
        "optimizer_state_saved",
        "intermediate_checkpoint_saved",
        "bounded_400_required",
        "bounded_400_authorization_effect",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_claim_supported",
        "receipt_fingerprint",
    }
)
_INPUTS_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "attempt_fingerprint",
        "real_inputs",
        "real_inputs_fingerprint",
        "full_D_R_scalar_cache_fingerprint",
        "source_binding_fingerprint",
        "source_files",
        "construction_invocations",
        "D_R_accessed",
        "D_V_accessed",
        "D_T_accessed",
        "split_manifest_metadata_read",
        "D_R_tensor_payload_accessed",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "training_performed",
        "receipt_fingerprint",
    }
)
_AUTHORIZATION_WRAPPER_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "attempt_fingerprint",
        "authorization",
        "authorization_fingerprint",
        "formal_D_R_training_authorized",
        "bounded_400_receipt_consumed",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "receipt_fingerprint",
    }
)
_TRAINING_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "attempt_fingerprint",
        "authorization_fingerprint",
        "formal_result",
        "formal_result_fingerprint",
        "training_result_fingerprint",
        "final_model_artifact_fingerprint",
        "compute_ledger",
        "final_checkpoint_only",
        "optimizer_state_saved",
        "intermediate_checkpoint_saved",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_claim_supported",
        "receipt_fingerprint",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "status",
        "attempt_fingerprint",
        "authorization_fingerprint",
        "formal_result_fingerprint",
        "final_model_artifact_fingerprint",
        "formal_training_complete",
        "compute_ledger_complete",
        "bounded_400_required",
        "bounded_400_authorization_effect",
        "D_V_preregistration_eligible",
        "D_V_execution_authorized",
        "D_T_execution_authorized",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_gate_passed",
        "performance_claim_supported",
        "final_model_performance_success_established",
        "next_action",
        "receipt_fingerprint",
    }
)
_COMPLETE_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "status",
        "attempt_fingerprint",
        "authorization_fingerprint",
        "formal_result_fingerprint",
        "final_model_artifact_fingerprint",
        "decision_fingerprint",
        "artifact_files",
        "artifact_count",
        "seed",
        "epochs",
        "steps_per_epoch",
        "updates",
        "training_invocations",
        "final_checkpoint_only",
        "optimizer_state_saved",
        "intermediate_checkpoint_saved",
        "bounded_400_required",
        "bounded_400_authorization_effect",
        "D_V_preregistration_eligible",
        "D_V_execution_authorized",
        "D_T_execution_authorized",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_claim_supported",
        "complete_fingerprint",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "candidate",
        "serialization",
        "model_file",
        "model_file_sha256",
        "model_config",
        "model_config_fingerprint",
        "state_keys",
        "state_shapes",
        "state_dtypes",
        "parameter_count",
        "coverage_state_model_fingerprint",
        "module_state_fingerprint",
        "formal_result_fingerprint",
        "training_result_fingerprint",
        "formal_training_ledger",
        "authorization_fingerprint",
        "source_closure_fingerprint",
        "final_checkpoint_only",
        "optimizer_state_saved",
        "intermediate_checkpoint_saved",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "performance_evaluation_performed",
        "artifact_fingerprint",
    }
)
_TRAINING_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "objective",
        "objective_policy",
        "seed",
        "epochs",
        "steps_per_epoch",
        "completed_updates",
        "schedule_fingerprint",
        "cache_fingerprint",
        "execution_device",
        "device_cache_fingerprint",
        "device_cache_resident_bytes",
        "optimizer_config_fingerprint",
        "initial_model_fingerprint",
        "final_model_fingerprint",
        "epoch_logs",
        "first_nonzero_gradient_update",
        "compute",
    }
)
_FORMAL_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "runtime_splits",
        "authorization_fingerprint",
        "training_result",
        "training_result_fingerprint",
        "final_model_fingerprint",
        "optimizer_config_fingerprint",
        "source_closure_fingerprint_after",
        "training_invocations",
        "checks",
        "training_complete",
        "output_contract",
        "bounded_400_required",
        "bounded_400_is_final_success",
        "D_V_accessed",
        "D_T_accessed",
        "calibration_performed",
        "inference_performed",
        "performance_evaluation_performed",
        "performance_claim_supported",
        "full_CURE_authorized",
        "cross_backbone_authorized",
    }
)
_FORMAL_RESULT_CHECKS = frozenset(
    {
        "authorization_claimed_and_consumed_once",
        "bounded400_not_prerequisite_or_success",
        "D_R_only_without_evaluation",
        "direct_D_R_13_of_13_authorization",
        "exact_v23_single_PMOPE_model",
        "fixed_seed42_800x40_compute_ledger",
        "fresh_exact_Adam_policy",
        "from_scratch_initial_to_one_final_model",
        "full_D_R_cache_and_formal_schedule",
        "no_resume_or_retry",
    }
)
_EPOCH_RESULT_FIELDS = frozenset(
    {
        "epoch",
        "completed_updates",
        "objective",
        "selection_sequence_fingerprint",
        "mean_factual_miss/loss",
        "mean_factual_no_miss/loss",
        "mean_pair/loss",
        "mean_total",
        "mean_gradient_l2_norm",
    }
)


@dataclass(frozen=True)
class _LivePrerequisites:
    runtime: dict[str, object]
    source_closure: dict[str, object]
    source_closure_fingerprint: str
    dataset_free: dict[str, object]
    dataset_free_fingerprint: str
    dr_verification: dict[str, object]
    dr_verification_fingerprint: str
    dr_receipt: CoverageStatePACREDRGateReceipt


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_fields(
    payload: Mapping[str, object],
    fields: frozenset[str],
    *,
    name: str,
) -> None:
    if frozenset(payload) != fields:
        raise ValueError(f"{name} fields differ from the frozen schema")


def _require_false(
    payload: Mapping[str, object],
    *names: str,
) -> None:
    if any(payload.get(name) is not False for name in names):
        raise PermissionError(
            "Formal800 terminal records forbidden evaluation or D_V/D_T access"
        )


def _expected_runtime_contract() -> dict[str, object]:
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": FROZEN_VISIBLE_GPU,
        "CUBLAS_WORKSPACE_CONFIG": FROZEN_CUBLAS_WORKSPACE_CONFIG,
        "PYTHONHASHSEED": FROZEN_PYTHONHASHSEED,
    }
    for name, expected in expected_environment.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"Formal800 verification fixes {name}={expected}")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal800 verification requires available CUDA")
    visible_count = int(torch.cuda.device_count())
    current_device = int(torch.cuda.current_device())
    if (
        visible_count != 1
        or current_device != 0
        or str(torch.device("cuda", current_device)) != FROZEN_DEVICE
    ):
        raise RuntimeError(
            "Formal800 verification fixes one visible logical cuda:0"
        )
    wrapper = ROOT / PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH
    if (
        not wrapper.is_file()
        or wrapper.is_symlink()
        or wrapper.resolve(strict=True) != wrapper
        or file_sha256(wrapper)
        != PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256
    ):
        raise RuntimeError("fixed Formal800 temperature wrapper changed")
    return {
        "device": FROZEN_DEVICE,
        **expected_environment,
        "cuda_available": True,
        "visible_cuda_device_count": visible_count,
        "current_cuda_logical_device": current_device,
        "temperature_wrapper_repo_path": (
            PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH
        ),
        "temperature_wrapper_file_sha256": (
            PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256
        ),
        "pause_temperature_c": FROZEN_PAUSE_TEMPERATURE_C,
        "resume_temperature_c": FROZEN_RESUME_TEMPERATURE_C,
        "temperature_wrapper_parent_verified": True,
    }


def _verify_output_population(output: Path) -> dict[str, str]:
    if (
        not output.is_dir()
        or output.is_symlink()
        or (output / ".incomplete").exists()
        or (output / ".incomplete").is_symlink()
        or (output / "FAILURE.json").exists()
        or (output / "FAILURE.json").is_symlink()
    ):
        raise RuntimeError("PACRE-VC Formal800 output is not terminal")
    directories: set[str] = set()
    files: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        relative = str(path.relative_to(output))
        if path.is_symlink():
            raise RuntimeError("Formal800 terminal contains a symbolic link")
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files[relative] = file_sha256(path)
        else:
            raise RuntimeError("Formal800 terminal contains a special file")
    expected_files = EXPECTED_ARTIFACT_FILES | {"COMPLETE.json"}
    if directories != EXPECTED_DIRECTORIES or set(files) != expected_files:
        raise RuntimeError("Formal800 terminal artifact population differs")
    return files


def _load_dr_receipt(
    verification: Mapping[str, object],
) -> CoverageStatePACREDRGateReceipt:
    output_value = verification.get("output")
    if not isinstance(output_value, str):
        raise TypeError("terminal D_R verification lacks its output path")
    wrapper = read_strict_json(
        Path(output_value) / "receipts/dr_gate.json"
    )
    wrapper_fingerprint = verify_fingerprinted(
        wrapper,
        field="wrapper_fingerprint",
    )
    receipt_payload = wrapper.get("receipt")
    if not isinstance(receipt_payload, Mapping):
        raise TypeError("terminal D_R wrapper lacks its canonical receipt")
    receipt = pacre_vc_dr_receipt_from_payload(receipt_payload)
    if (
        verification.get("decision") != PACRE_VC_DR_PASS_DECISION
        or verification.get("gate_passed") is not True
        or verification.get("failed_checks") != []
        or verification.get("receipt_fingerprint")
        != receipt.receipt_fingerprint
        or verification.get("wrapper_fingerprint")
        != wrapper_fingerprint
        or verification.get("formal_800_route_granted") is not True
        or verification.get("bounded_400_required") is not False
        or verification.get("bounded_400_authorization_effect") is not False
        or receipt.decision != PACRE_VC_DR_PASS_DECISION
        or receipt.gate_passed is not True
        or tuple(name for name, _ in receipt.checks)
        != PACRE_VC_DR_CHECK_NAMES
        or len(receipt.checks) != 13
        or not all(passed for _, passed in receipt.checks)
    ):
        raise PermissionError(
            "Formal800 terminal lacks the exact v23 D_R 13/13 PASS"
        )
    return receipt


def _verify_live_prerequisites() -> _LivePrerequisites:
    runtime = _expected_runtime_contract()
    closure = read_strict_json(
        protocol_root() / "implementation_closure.json"
    )
    closure_fingerprint = verify_source_closure(closure)
    dataset_free = read_strict_json(
        protocol_root() / "dataset_free_receipt.json"
    )
    dataset_free_fingerprint = verify_fingerprinted(dataset_free)
    dr_verification = dict(verify_dr_terminal())
    dr_verification_fingerprint = stable_fingerprint(dr_verification)
    dr_receipt = _load_dr_receipt(dr_verification)
    if (
        not _is_sha256(closure_fingerprint)
        or not _is_sha256(dataset_free_fingerprint)
        or dr_receipt.source_closure_fingerprint
        != closure_fingerprint
        or dr_receipt.dataset_free_receipt_fingerprint
        != dataset_free_fingerprint
    ):
        raise PermissionError(
            "live source/runtime/D_R prerequisites changed after Formal800"
        )
    return _LivePrerequisites(
        runtime=runtime,
        source_closure=closure,
        source_closure_fingerprint=closure_fingerprint,
        dataset_free=dataset_free,
        dataset_free_fingerprint=dataset_free_fingerprint,
        dr_verification=dr_verification,
        dr_verification_fingerprint=dr_verification_fingerprint,
        dr_receipt=dr_receipt,
    )


def _verify_attempt(
    attempt: Mapping[str, object],
    live: _LivePrerequisites,
) -> str:
    _require_exact_fields(attempt, _ATTEMPT_FIELDS, name="attempt")
    fingerprint = verify_fingerprinted(attempt)
    if (
        attempt.get("schema_version") != ATTEMPT_SCHEMA
        or attempt.get("run_id") != RUN_ID
        or attempt.get("output_repo_path") != OUTPUT_REPO_PATH
        or attempt.get("candidate") != "PACRE-VC-v23"
        or attempt.get("objective") != PACRE_PMOPE_OBJECTIVE
        or attempt.get("runtime") != live.runtime
        or attempt.get("budget")
        != {
            "seed": FROZEN_SEED,
            "epochs": FROZEN_EPOCHS,
            "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
            "updates": FROZEN_UPDATES,
            "from_scratch": True,
            "training_invocations": 1,
        }
        or attempt.get("source_closure_fingerprint")
        != live.source_closure_fingerprint
        or attempt.get("dataset_free_receipt_fingerprint")
        != live.dataset_free_fingerprint
        or attempt.get("D_R_terminal_verification_fingerprint")
        != live.dr_verification_fingerprint
        or attempt.get("D_R_gate_receipt_fingerprint")
        != live.dr_receipt.receipt_fingerprint
        or attempt.get("D_R_gate_check_count") != 13
        or attempt.get("D_R_gate_passed") is not True
        or attempt.get("bounded_400_output_absent") is not True
        or attempt.get("bounded_400_required") is not False
        or attempt.get("bounded_400_authorization_effect") is not False
        or attempt.get("single_attempt") is not True
        or attempt.get("resume_allowed") is not False
        or attempt.get("automatic_retry_allowed") is not False
        or attempt.get("overwrite_allowed") is not False
        or attempt.get("checkpoint_policy") != "final_model_only"
        or attempt.get("D_R_receipt_metadata_read") is not True
        or attempt.get("D_R_tensor_payload_accessed") is not False
    ):
        raise ValueError("Formal800 attempt claim changed")
    _require_false(
        attempt,
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_claim_supported",
    )
    return fingerprint


def _verify_dr_terminal_binding(
    payload: Mapping[str, object],
    *,
    attempt_fingerprint: str,
    live: _LivePrerequisites,
) -> None:
    _require_exact_fields(
        payload,
        _DR_TERMINAL_FIELDS,
        name="D_R terminal binding",
    )
    verify_fingerprinted(payload)
    if (
        payload.get("schema_version") != DR_TERMINAL_SCHEMA
        or payload.get("run_id") != RUN_ID
        or payload.get("attempt_fingerprint") != attempt_fingerprint
        or payload.get("verification") != live.dr_verification
        or payload.get("verification_fingerprint")
        != live.dr_verification_fingerprint
        or payload.get("D_R_gate_receipt_fingerprint")
        != live.dr_receipt.receipt_fingerprint
        or payload.get("D_R_gate_check_count") != 13
        or payload.get("D_R_gate_passed") is not True
        or payload.get("D_R_reopened") is not False
        or payload.get("D_R_tensor_payload_accessed") is not False
        or payload.get("training_performed") is not False
    ):
        raise ValueError("persisted D_R terminal replay changed")
    _require_false(
        payload,
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
    )


def _verify_config(
    payload: Mapping[str, object],
    *,
    model_config: object,
    live: _LivePrerequisites,
) -> None:
    _require_exact_fields(payload, _CONFIG_FIELDS, name="config")
    verify_fingerprinted(payload)
    if (
        payload.get("schema_version") != CONFIG_SCHEMA
        or payload.get("run_id") != RUN_ID
        or payload.get("candidate") != "PACRE-VC-v23"
        or payload.get("split") != "D_R"
        or payload.get("runtime_splits") != ["D_R"]
        or payload.get("objective") != PACRE_PMOPE_OBJECTIVE
        or payload.get("model_config") != asdict(model_config)
        or payload.get("parameter_count") != FROZEN_PARAMETER_COUNT
        or payload.get("field_threshold_hex") != 0.0.hex()
        or payload.get("threshold_search_performed") is not False
        or payload.get("budget")
        != {
            "seed": FROZEN_SEED,
            "epochs": FROZEN_EPOCHS,
            "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
            "updates": FROZEN_UPDATES,
            "from_scratch": True,
            "training_invocations": 1,
        }
        or payload.get("runtime") != live.runtime
        or payload.get("checkpoint_policy") != "final_model_only"
        or payload.get("optimizer_state_saved") is not False
        or payload.get("intermediate_checkpoint_saved") is not False
        or payload.get("bounded_400_required") is not False
        or payload.get("bounded_400_authorization_effect") is not False
    ):
        raise ValueError("Formal800 frozen config changed")
    _require_false(
        payload,
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_claim_supported",
    )


def _source_file_receipt(paths: Mapping[str, Path]) -> dict[str, object]:
    return {
        name: {
            "repo_path": str(path.relative_to(ROOT)),
            "file_sha256": file_sha256(path),
        }
        for name, path in sorted(paths.items())
    }


def _rebuild_authorization(
    *,
    attempt_fingerprint: str,
    inputs: Mapping[str, object],
    config: Mapping[str, object],
    authorization_wrapper: Mapping[str, object],
    live: _LivePrerequisites,
):
    source_paths = frozen_real_dr_source_paths()
    real_inputs = build_coverage_state_real_dr_inputs(**source_paths)
    real_inputs.verify_unchanged()
    model_config = expected_pacre_vc_formal_config(real_inputs)
    if (
        real_inputs.source_binding.split != "D_R"
        or real_inputs.scalar_cache.raw_catalog.split != "D_R"
        or real_inputs.scalar_cache.cache_fingerprint
        != PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
        or model_config.expected_parameter_count
        != FROZEN_PARAMETER_COUNT
    ):
        raise PermissionError("rebuilt full D_R Formal800 input changed")

    _require_exact_fields(inputs, _INPUTS_FIELDS, name="inputs")
    verify_fingerprinted(inputs)
    if (
        inputs.get("schema_version") != INPUTS_SCHEMA
        or inputs.get("run_id") != RUN_ID
        or inputs.get("attempt_fingerprint") != attempt_fingerprint
        or inputs.get("real_inputs") != real_inputs.canonical_payload()
        or inputs.get("real_inputs_fingerprint")
        != real_inputs.build_fingerprint
        or inputs.get("full_D_R_scalar_cache_fingerprint")
        != real_inputs.scalar_cache.cache_fingerprint
        or inputs.get("source_binding_fingerprint")
        != real_inputs.source_binding.binding_fingerprint
        or inputs.get("source_files") != _source_file_receipt(source_paths)
        or inputs.get("construction_invocations") != 1
        or inputs.get("D_R_accessed") is not True
        or inputs.get("D_V_accessed") is not False
        or inputs.get("D_T_accessed") is not False
        or inputs.get("split_manifest_metadata_read") is not True
        or inputs.get("D_R_tensor_payload_accessed") is not True
        or inputs.get("training_performed") is not False
    ):
        raise ValueError("persisted full D_R inputs changed")
    _require_false(
        inputs,
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
    )
    _verify_config(config, model_config=model_config, live=live)

    authorization = prepare_pacre_vc_formal_800_authorization(
        real_inputs,
        model_config,
        dataset_free_receipt=live.dataset_free,
        dr_gate_receipt=live.dr_receipt,
        source_closure=live.source_closure,
        output_claim_fingerprint=attempt_fingerprint,
        run_id=RUN_ID,
    )
    authorization.verify_unchanged()
    _require_exact_fields(
        authorization_wrapper,
        _AUTHORIZATION_WRAPPER_FIELDS,
        name="authorization wrapper",
    )
    verify_fingerprinted(authorization_wrapper)
    expected_payload = authorization.canonical_payload()
    if (
        authorization_wrapper.get("schema_version")
        != AUTHORIZATION_WRAPPER_SCHEMA
        or authorization_wrapper.get("run_id") != RUN_ID
        or authorization_wrapper.get("attempt_fingerprint")
        != attempt_fingerprint
        or authorization_wrapper.get("authorization")
        != expected_payload
        or authorization_wrapper.get("authorization_fingerprint")
        != authorization.authorization_fingerprint
        or authorization_wrapper.get("formal_D_R_training_authorized")
        is not True
        or authorization_wrapper.get("bounded_400_receipt_consumed")
        is not False
        or authorization.output_claim_fingerprint
        != attempt_fingerprint
        or authorization.prerequisites_passed is not True
        or authorization.available is not True
        or expected_payload.get("schema_version")
        != PACRE_VC_FORMAL_AUTHORIZATION_SCHEMA
    ):
        raise PermissionError("persisted Formal800 authorization changed")
    _require_false(
        authorization_wrapper,
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
    )
    return authorization, model_config


def _parse_training_result(
    payload: Mapping[str, object],
) -> CoverageStateTrainingResult:
    _require_exact_fields(
        payload,
        _TRAINING_RESULT_FIELDS,
        name="training result",
    )
    compute = payload.get("compute")
    first = payload.get("first_nonzero_gradient_update")
    epoch_logs = payload.get("epoch_logs")
    if (
        payload.get("schema_version")
        != COVERAGE_STATE_TRAINING_RESULT_SCHEMA
        or not isinstance(compute, Mapping)
        or frozenset(compute)
        != {
            "forward_calls",
            "backward_calls",
            "optimizer_steps",
            "logical_state_evaluations",
            "finite_state_audits",
        }
        or not isinstance(first, Mapping)
        or not isinstance(epoch_logs, list)
    ):
        raise ValueError("common Formal800 training result schema changed")
    if set(first) != set(PACRE_VC_PARAMETER_NAMES) or any(
        type(value) is not int or value < 0 or value >= FROZEN_UPDATES
        for value in first.values()
    ):
        raise ValueError("Formal800 gradient-coverage ledger changed")
    integer_fields = (
        payload.get("seed"),
        payload.get("epochs"),
        payload.get("steps_per_epoch"),
        payload.get("completed_updates"),
        payload.get("device_cache_resident_bytes"),
        compute.get("forward_calls"),
        compute.get("backward_calls"),
        compute.get("optimizer_steps"),
        compute.get("logical_state_evaluations"),
        compute.get("finite_state_audits"),
    )
    digest_fields = (
        payload.get("schedule_fingerprint"),
        payload.get("cache_fingerprint"),
        payload.get("device_cache_fingerprint"),
        payload.get("optimizer_config_fingerprint"),
        payload.get("initial_model_fingerprint"),
        payload.get("final_model_fingerprint"),
    )
    if (
        any(type(value) is not int for value in integer_fields)
        or payload.get("device_cache_resident_bytes", 0) < 1
        or not all(_is_sha256(value) for value in digest_fields)
        or type(payload.get("objective")) is not str
        or type(payload.get("objective_policy")) is not str
        or type(payload.get("execution_device")) is not str
    ):
        raise ValueError("Formal800 training result scalar types changed")
    result = CoverageStateTrainingResult(
        objective=payload.get("objective"),
        objective_policy=payload.get("objective_policy"),
        seed=payload.get("seed"),
        epochs=payload.get("epochs"),
        steps_per_epoch=payload.get("steps_per_epoch"),
        completed_updates=payload.get("completed_updates"),
        schedule_fingerprint=payload.get("schedule_fingerprint"),
        cache_fingerprint=payload.get("cache_fingerprint"),
        execution_device=payload.get("execution_device"),
        device_cache_fingerprint=payload.get(
            "device_cache_fingerprint"
        ),
        device_cache_resident_bytes=payload.get(
            "device_cache_resident_bytes"
        ),
        optimizer_config_fingerprint=payload.get(
            "optimizer_config_fingerprint"
        ),
        initial_model_fingerprint=payload.get(
            "initial_model_fingerprint"
        ),
        final_model_fingerprint=payload.get(
            "final_model_fingerprint"
        ),
        epoch_logs=tuple(epoch_logs),
        first_nonzero_gradient_update=tuple(sorted(first.items())),
        forward_calls=compute.get("forward_calls"),
        backward_calls=compute.get("backward_calls"),
        optimizer_steps=compute.get("optimizer_steps"),
        logical_state_evaluations=compute.get(
            "logical_state_evaluations"
        ),
        finite_state_audits=compute.get("finite_state_audits"),
    )
    if result.canonical_payload() != dict(payload):
        raise ValueError("training result is not exact canonical reconstruction")
    return result


def _read_epoch_progress(
    path: Path,
) -> tuple[dict[str, object], ...]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError("Formal800 epoch progress is not a regular file")
    try:
        raw_lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise RuntimeError("cannot read Formal800 epoch progress") from error
    if len(raw_lines) != FROZEN_EPOCHS:
        raise ValueError("Formal800 epoch progress must contain 800 rows")
    rows: list[dict[str, object]] = []
    for expected_epoch, raw in enumerate(raw_lines):
        try:
            event = json.loads(
                raw.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Formal800 epoch progress is invalid JSONL") from error
        if (
            not isinstance(event, dict)
            or strict_json_bytes(event) != raw
            or frozenset(event)
            != {
                "schema_version",
                "run_id",
                "epoch_result",
                "receipt_fingerprint",
            }
        ):
            raise ValueError("Formal800 epoch event schema changed")
        verify_fingerprinted(event)
        row = event.get("epoch_result")
        if not isinstance(row, dict):
            raise TypeError("Formal800 epoch event lacks its result")
        _require_exact_fields(
            row,
            _EPOCH_RESULT_FIELDS,
            name="epoch result",
        )
        metrics = (
            row["mean_factual_miss/loss"],
            row["mean_factual_no_miss/loss"],
            row["mean_pair/loss"],
            row["mean_total"],
            row["mean_gradient_l2_norm"],
        )
        if (
            event.get("schema_version") != EPOCH_PROGRESS_SCHEMA
            or event.get("run_id") != RUN_ID
            or type(row.get("epoch")) is not int
            or row.get("epoch") != expected_epoch
            or type(row.get("completed_updates")) is not int
            or row.get("completed_updates")
            != (expected_epoch + 1) * FROZEN_STEPS_PER_EPOCH
            or row.get("objective") != PACRE_PMOPE_OBJECTIVE
            or not _is_sha256(row.get("selection_sequence_fingerprint"))
            or any(
                type(value) is not float
                or not isfinite(float(value))
                for value in metrics
            )
        ):
            raise ValueError("Formal800 epoch result changed")
        rows.append(row)
    return tuple(rows)


def _fresh_optimizer_fingerprint(
    artifact: LoadedPACREVCFormalArtifact,
) -> str:
    config = PACRE_PMOPE_TRAINING_CONFIG
    optimizer = torch.optim.Adam(
        artifact.model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    if type(optimizer) is not torch.optim.Adam or optimizer.state:
        raise RuntimeError("fresh exact Adam policy changed")
    return coverage_state_optimizer_config_fingerprint(
        artifact.model,
        optimizer,
    )


def _verify_training(
    payload: Mapping[str, object],
    *,
    attempt_fingerprint: str,
    authorization: object,
    live: _LivePrerequisites,
    epoch_rows: tuple[dict[str, object], ...],
    artifact: LoadedPACREVCFormalArtifact,
) -> tuple[CoverageStateTrainingResult, str]:
    _require_exact_fields(payload, _TRAINING_FIELDS, name="training")
    verify_fingerprinted(payload)
    formal = payload.get("formal_result")
    if not isinstance(formal, Mapping):
        raise TypeError("Formal800 training wrapper lacks its formal result")
    _require_exact_fields(
        formal,
        _FORMAL_RESULT_FIELDS,
        name="formal result",
    )
    training_payload = formal.get("training_result")
    if not isinstance(training_payload, Mapping):
        raise TypeError("formal result lacks the common training result")
    result = _parse_training_result(training_payload)
    result_fingerprint = result.result_fingerprint
    artifact_receipt = artifact.receipt
    optimizer_fingerprint = _fresh_optimizer_fingerprint(artifact)
    checks = formal.get("checks")
    ledger = payload.get("compute_ledger")
    if (
        not isinstance(checks, Mapping)
        or frozenset(checks) != _FORMAL_RESULT_CHECKS
        or not all(value is True for value in checks.values())
        or not isinstance(ledger, Mapping)
        or frozenset(ledger)
        != {
            "seed",
            "epochs",
            "steps_per_epoch",
            "completed_updates",
            "forward_calls",
            "backward_calls",
            "optimizer_steps",
            "logical_state_evaluations",
            "finite_state_audits",
            "epoch_progress_rows",
            "training_invocations",
        }
    ):
        raise ValueError("Formal800 result checks/ledger changed")
    if (
        payload.get("schema_version") != TRAINING_SCHEMA
        or payload.get("run_id") != RUN_ID
        or payload.get("attempt_fingerprint") != attempt_fingerprint
        or payload.get("authorization_fingerprint")
        != authorization.authorization_fingerprint
        or payload.get("formal_result_fingerprint")
        != stable_fingerprint(dict(formal))
        or payload.get("training_result_fingerprint")
        != result_fingerprint
        or payload.get("final_model_artifact_fingerprint")
        != artifact.artifact_fingerprint
        or payload.get("final_checkpoint_only") is not True
        or payload.get("optimizer_state_saved") is not False
        or payload.get("intermediate_checkpoint_saved") is not False
        or formal.get("schema_version") != PACRE_VC_FORMAL_RESULT_SCHEMA
        or formal.get("run_id") != RUN_ID
        or formal.get("runtime_splits") != ["D_R"]
        or formal.get("authorization_fingerprint")
        != authorization.authorization_fingerprint
        or dict(training_payload) != result.canonical_payload()
        or formal.get("training_result_fingerprint")
        != result_fingerprint
        or formal.get("final_model_fingerprint")
        != result.final_model_fingerprint
        or formal.get("optimizer_config_fingerprint")
        != optimizer_fingerprint
        or formal.get("source_closure_fingerprint_after")
        != live.source_closure_fingerprint
        or formal.get("training_invocations") != 1
        or formal.get("training_complete") is not True
        or formal.get("output_contract")
        != {
            "single_final_model": True,
            "checkpoint_written": False,
            "optimizer_state_returned": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        }
        or formal.get("bounded_400_required") is not False
        or formal.get("bounded_400_is_final_success") is not False
        or result.objective != PACRE_PMOPE_OBJECTIVE
        or result.objective_policy != CSLF_PMOPE_POLICY
        or result.seed != FROZEN_SEED
        or result.epochs != FROZEN_EPOCHS
        or result.steps_per_epoch != FROZEN_STEPS_PER_EPOCH
        or result.completed_updates != FROZEN_UPDATES
        or result.schedule_fingerprint
        != PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT
        or result.schedule_fingerprint
        != authorization.schedule.schedule_fingerprint
        or result.cache_fingerprint
        != PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
        or result.cache_fingerprint
        != authorization.real_inputs.scalar_cache.cache_fingerprint
        or result.execution_device != FROZEN_DEVICE
        or result.optimizer_config_fingerprint != optimizer_fingerprint
        or result.initial_model_fingerprint
        != PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT
        or result.initial_model_fingerprint
        != authorization.initial_model_fingerprint
        or result.final_model_fingerprint
        != coverage_state_model_fingerprint(artifact.model)
        or result.final_model_fingerprint
        == result.initial_model_fingerprint
        or result.epoch_logs != epoch_rows
        or result.forward_calls != FROZEN_UPDATES
        or result.backward_calls != FROZEN_UPDATES
        or result.optimizer_steps != FROZEN_UPDATES
        or result.logical_state_evaluations != 12 * FROZEN_UPDATES
        or result.finite_state_audits != FROZEN_UPDATES + 1
        or tuple(name for name, _ in artifact.model.named_parameters())
        != PACRE_VC_PARAMETER_NAMES
        or sum(value.numel() for value in artifact.model.parameters())
        != FROZEN_PARAMETER_COUNT
        or artifact_receipt.get("training_result_fingerprint")
        != result_fingerprint
        or artifact_receipt.get("formal_result_fingerprint")
        != payload.get("formal_result_fingerprint")
        or artifact_receipt.get("formal_training_ledger")
        != formal_training_ledger_payload(artifact.model, result)
        or artifact_receipt.get("authorization_fingerprint")
        != authorization.authorization_fingerprint
        or artifact_receipt.get("source_closure_fingerprint")
        != live.source_closure_fingerprint
        or ledger
        != {
            "seed": FROZEN_SEED,
            "epochs": FROZEN_EPOCHS,
            "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
            "completed_updates": FROZEN_UPDATES,
            "forward_calls": FROZEN_UPDATES,
            "backward_calls": FROZEN_UPDATES,
            "optimizer_steps": FROZEN_UPDATES,
            "logical_state_evaluations": 12 * FROZEN_UPDATES,
            "finite_state_audits": FROZEN_UPDATES + 1,
            "epoch_progress_rows": FROZEN_EPOCHS,
            "training_invocations": 1,
        }
    ):
        raise ValueError("Formal800 training/result/artifact binding changed")
    _require_false(
        payload,
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_claim_supported",
    )
    _require_false(
        formal,
        "D_V_accessed",
        "D_T_accessed",
        "calibration_performed",
        "inference_performed",
        "performance_evaluation_performed",
        "performance_claim_supported",
        "full_CURE_authorized",
        "cross_backbone_authorized",
    )
    return result, str(payload["formal_result_fingerprint"])


def _verify_artifact(
    output: Path,
    *,
    authorization_fingerprint: str,
    source_closure_fingerprint: str,
) -> LoadedPACREVCFormalArtifact:
    receipt = read_strict_json(output / "final_model/artifact.json")
    _require_exact_fields(receipt, _ARTIFACT_FIELDS, name="final artifact")
    verify_fingerprinted(receipt, field="artifact_fingerprint")
    if (
        receipt.get("schema_version")
        != PACRE_VC_FORMAL_MODEL_ARTIFACT_SCHEMA
        or receipt.get("authorization_fingerprint")
        != authorization_fingerprint
        or receipt.get("source_closure_fingerprint")
        != source_closure_fingerprint
        or receipt.get("final_checkpoint_only") is not True
        or receipt.get("optimizer_state_saved") is not False
        or receipt.get("intermediate_checkpoint_saved") is not False
    ):
        raise ValueError("Formal800 final-only artifact binding changed")
    _require_false(
        receipt,
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "performance_evaluation_performed",
    )
    loaded = load_pacre_vc_formal_final_model(
        output / "final_model",
        receipt,
    )
    loaded.verify_unchanged()
    return loaded


def _verify_decision(
    payload: Mapping[str, object],
    *,
    attempt_fingerprint: str,
    authorization_fingerprint: str,
    formal_result_fingerprint: str,
    artifact_fingerprint: str,
) -> str:
    _require_exact_fields(payload, _DECISION_FIELDS, name="decision")
    fingerprint = verify_fingerprinted(payload)
    if (
        payload.get("schema_version") != DECISION_SCHEMA
        or payload.get("run_id") != RUN_ID
        or payload.get("status") != TERMINAL_STATUS
        or payload.get("attempt_fingerprint") != attempt_fingerprint
        or payload.get("authorization_fingerprint")
        != authorization_fingerprint
        or payload.get("formal_result_fingerprint")
        != formal_result_fingerprint
        or payload.get("final_model_artifact_fingerprint")
        != artifact_fingerprint
        or payload.get("formal_training_complete") is not True
        or payload.get("compute_ledger_complete") is not True
        or payload.get("bounded_400_required") is not False
        or payload.get("bounded_400_authorization_effect") is not False
        or payload.get("D_V_preregistration_eligible") is not True
        or payload.get("D_V_execution_authorized") is not False
        or payload.get("D_T_execution_authorized") is not False
        or payload.get("performance_gate_passed") is not None
        or payload.get("final_model_performance_success_established")
        is not False
        or payload.get("next_action") != NEXT_ACTION
    ):
        raise PermissionError("Formal800 terminal decision exceeds training")
    _require_false(
        payload,
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_claim_supported",
    )
    return fingerprint


def _verify_complete(
    complete: Mapping[str, object],
    *,
    live_files: Mapping[str, str],
    attempt_fingerprint: str,
    authorization_fingerprint: str,
    formal_result_fingerprint: str,
    artifact_fingerprint: str,
    decision_fingerprint: str,
) -> str:
    _require_exact_fields(complete, _COMPLETE_FIELDS, name="COMPLETE")
    fingerprint = verify_fingerprinted(
        complete,
        field="complete_fingerprint",
    )
    artifacts = complete.get("artifact_files")
    live_artifacts = {
        name: digest
        for name, digest in live_files.items()
        if name != "COMPLETE.json"
    }
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != EXPECTED_ARTIFACT_FILES
        or artifacts != live_artifacts
        or complete.get("artifact_count") != len(EXPECTED_ARTIFACT_FILES)
        or complete.get("schema_version") != COMPLETE_SCHEMA
        or complete.get("run_id") != RUN_ID
        or complete.get("status") != TERMINAL_STATUS
        or complete.get("attempt_fingerprint") != attempt_fingerprint
        or complete.get("authorization_fingerprint")
        != authorization_fingerprint
        or complete.get("formal_result_fingerprint")
        != formal_result_fingerprint
        or complete.get("final_model_artifact_fingerprint")
        != artifact_fingerprint
        or complete.get("decision_fingerprint") != decision_fingerprint
        or complete.get("seed") != FROZEN_SEED
        or complete.get("epochs") != FROZEN_EPOCHS
        or complete.get("steps_per_epoch") != FROZEN_STEPS_PER_EPOCH
        or complete.get("updates") != FROZEN_UPDATES
        or complete.get("training_invocations") != 1
        or complete.get("final_checkpoint_only") is not True
        or complete.get("optimizer_state_saved") is not False
        or complete.get("intermediate_checkpoint_saved") is not False
        or complete.get("bounded_400_required") is not False
        or complete.get("bounded_400_authorization_effect") is not False
        or complete.get("D_V_preregistration_eligible") is not True
        or complete.get("D_V_execution_authorized") is not False
        or complete.get("D_T_execution_authorized") is not False
    ):
        raise ValueError("Formal800 COMPLETE graph binding changed")
    _require_false(
        complete,
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_claim_supported",
    )
    return fingerprint


def verify_terminal(output: Path = OUTPUT_PATH) -> dict[str, object]:
    """Verify and return the exact final-only Formal800 binding.

    The optional path exists for strict downstream composition and synthetic
    tests; the command-line entry point always verifies ``OUTPUT_PATH``.
    """

    stabilize_pacre_vc_numerical_runtime()
    if (
        BOUNDED_OUTPUT_PATH.exists()
        or BOUNDED_OUTPUT_PATH.is_symlink()
    ):
        raise PermissionError(
            "v23 bounded-400 output must be absent and has no Formal800 effect"
        )
    output = Path(output)
    live_files = _verify_output_population(output)
    complete = read_strict_json(output / "COMPLETE.json")
    # Check the self-contained hash graph before any expensive D_R rebuild.
    preliminary_artifacts = complete.get("artifact_files")
    if (
        not isinstance(preliminary_artifacts, dict)
        or preliminary_artifacts
        != {
            name: digest
            for name, digest in live_files.items()
            if name != "COMPLETE.json"
        }
    ):
        raise RuntimeError("Formal800 COMPLETE artifact hashes changed")
    verify_fingerprinted(complete, field="complete_fingerprint")

    live = _verify_live_prerequisites()
    attempt = read_strict_json(output / "attempt.json")
    attempt_fingerprint = _verify_attempt(attempt, live)
    dr_terminal = read_strict_json(
        output / "receipts/dr_terminal_verification.json"
    )
    _verify_dr_terminal_binding(
        dr_terminal,
        attempt_fingerprint=attempt_fingerprint,
        live=live,
    )
    config = read_strict_json(output / "receipts/config.json")
    inputs = read_strict_json(output / "receipts/inputs.json")
    authorization_wrapper = read_strict_json(
        output / "receipts/authorization.json"
    )
    authorization, _ = _rebuild_authorization(
        attempt_fingerprint=attempt_fingerprint,
        inputs=inputs,
        config=config,
        authorization_wrapper=authorization_wrapper,
        live=live,
    )
    artifact = _verify_artifact(
        output,
        authorization_fingerprint=(
            authorization.authorization_fingerprint
        ),
        source_closure_fingerprint=live.source_closure_fingerprint,
    )
    epoch_rows = _read_epoch_progress(
        output / "receipts/epoch_progress.jsonl"
    )
    training = read_strict_json(output / "receipts/training.json")
    training_result, formal_result_fingerprint = _verify_training(
        training,
        attempt_fingerprint=attempt_fingerprint,
        authorization=authorization,
        live=live,
        epoch_rows=epoch_rows,
        artifact=artifact,
    )
    decision = read_strict_json(output / "receipts/decision.json")
    decision_fingerprint = _verify_decision(
        decision,
        attempt_fingerprint=attempt_fingerprint,
        authorization_fingerprint=(
            authorization.authorization_fingerprint
        ),
        formal_result_fingerprint=formal_result_fingerprint,
        artifact_fingerprint=artifact.artifact_fingerprint,
    )
    complete_fingerprint = _verify_complete(
        complete,
        live_files=live_files,
        attempt_fingerprint=attempt_fingerprint,
        authorization_fingerprint=(
            authorization.authorization_fingerprint
        ),
        formal_result_fingerprint=formal_result_fingerprint,
        artifact_fingerprint=artifact.artifact_fingerprint,
        decision_fingerprint=decision_fingerprint,
    )
    artifact.verify_unchanged()
    authorization.real_inputs.verify_unchanged()
    authorization.verify_unchanged()
    closing_source_fingerprint = verify_source_closure(
        live.source_closure
    )
    if closing_source_fingerprint != live.source_closure_fingerprint:
        raise RuntimeError("PACRE-VC source closure changed during verification")

    return {
        "schema_version": VERIFICATION_SCHEMA,
        "run_id": RUN_ID,
        "output": str(output),
        "status": TERMINAL_STATUS,
        "complete_fingerprint": complete_fingerprint,
        "attempt_fingerprint": attempt_fingerprint,
        "authorization_fingerprint": (
            authorization.authorization_fingerprint
        ),
        "training_result_fingerprint": (
            training_result.result_fingerprint
        ),
        "formal_result_fingerprint": formal_result_fingerprint,
        "source_closure_fingerprint": live.source_closure_fingerprint,
        "D_R_gate_receipt_fingerprint": (
            live.dr_receipt.receipt_fingerprint
        ),
        "artifact_fingerprint": artifact.artifact_fingerprint,
        "model_file_sha256": artifact.receipt["model_file_sha256"],
        "final_model_fingerprint": (
            training_result.final_model_fingerprint
        ),
        "seed": FROZEN_SEED,
        "epochs": FROZEN_EPOCHS,
        "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
        "updates": FROZEN_UPDATES,
        "from_scratch": True,
        "training_invocations": 1,
        "artifact_count": len(EXPECTED_ARTIFACT_FILES),
        "D_V_preregistration_eligible": True,
        "D_V_execution_authorized": False,
        "D_T_execution_authorized": False,
        "D_V_tensor_payload_accessed": False,
        "D_T_tensor_payload_accessed": False,
        "performance_evaluation_performed": False,
        "performance_claim_supported": False,
        "bounded_400_required": False,
        "bounded_400_authorization_effect": False,
    }


def verify_terminal_sealed(
    output: Path = OUTPUT_PATH,
) -> VerifiedPACREVCFormalTerminal:
    """Return the exact loaded artifact with a verifier-issued terminal seal."""

    output = Path(output)
    verification = verify_terminal(output)
    artifact_payload = read_strict_json(
        output / "final_model/artifact.json"
    )
    artifact = load_pacre_vc_formal_final_model(
        output / "final_model",
        artifact_payload,
    )
    return _issue_verified_pacre_vc_formal_terminal(
        artifact,
        verification,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    result = verify_terminal()
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
