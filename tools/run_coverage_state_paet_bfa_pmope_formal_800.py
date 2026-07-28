#!/usr/bin/env python3
"""Validate or consume the unique v21 PAET-BFA Formal800 attempt.

``--validate-create-only`` checks only static files and the completed v21
bounded evidence.  It does not deserialize the full D_R tensor cache, claim
the output directory, construct an optimizer, or access D_V/D_T.

``--run-once`` is fixed to seed 42, 800 x 40 updates, visible ``cuda:0``,
``CUBLAS_WORKSPACE_CONFIG=:4096:8``, and the 82/75 C temperature wrapper.
The output directory is the cross-process attempt claim.  A started or failed
directory is never resumed or reused.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import BinaryIO, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_paet_formal_source_closure import (
    verify_coverage_state_paet_formal_source_closure,
)
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
)
from cure_lite.experiment.coverage_state_paet_dataset_free import (
    COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_PAET_FORMAL_WIDTH,
)
from cure_lite.experiment.coverage_state_paet_formal_artifacts import (
    load_coverage_state_paet_formal_artifact,
    save_coverage_state_paet_formal_artifact,
)
from cure_lite.experiment.coverage_state_paet_formal_structural import (
    evaluate_coverage_state_paet_formal_structural_retention,
)
from cure_lite.experiment.coverage_state_paet_formal_training import (
    COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT,
    COVERAGE_STATE_PAET_FORMAL_RUN_ID,
    COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT,
    COVERAGE_STATE_PAET_FORMAL_SEED,
    COVERAGE_STATE_PAET_FORMAL_UPDATES,
    _current_formal_implementation_binding,
    expected_coverage_state_paet_formal_config,
    load_repository_coverage_state_paet_bounded_artifact_seal,
    prepare_coverage_state_paet_formal_800_authorization,
    run_coverage_state_paet_bfa_pmope_formal_800,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from tools import (
    run_coverage_state_cslf_support_oriented_bounded_400 as _base_cli,
)
from tools import (
    run_coverage_state_paet_bfa_pmope_bounded_400 as _bounded_cli,
)


_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = COVERAGE_STATE_PAET_FORMAL_RUN_ID
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = _ROOT / OUTPUT_REPO_PATH
RUN_SCHEMA = "cure-lite-paet-bfa-v21-pmope-formal800-run-v1"
VALIDATION_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-formal800-"
    "create-only-validation-v1"
)
ATTEMPT_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-formal800-attempt-v1"
)
STARTED_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-formal800-started-v1"
)
FAILURE_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-formal800-failure-v1"
)
DECISION_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-formal800-structural-decision-v1"
)
FROZEN_DEVICE = _base_cli.FROZEN_DEVICE
FROZEN_VISIBLE_GPU = _base_cli.FROZEN_VISIBLE_GPU
FROZEN_CUBLAS_WORKSPACE_CONFIG = (
    _base_cli.FROZEN_CUBLAS_WORKSPACE_CONFIG
)
FROZEN_PAUSE_TEMPERATURE_C = (
    _base_cli.FROZEN_PAUSE_TEMPERATURE_C
)
FROZEN_RESUME_TEMPERATURE_C = (
    _base_cli.FROZEN_RESUME_TEMPERATURE_C
)
FROZEN_FEATURE_CHANNELS = (
    COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS
)
FROZEN_FEATURE_STRIDE = COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
FROZEN_MODEL_WIDTH = COVERAGE_STATE_PAET_FORMAL_WIDTH
FROZEN_PARAMETER_COUNT = COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
FROZEN_SEED = COVERAGE_STATE_PAET_FORMAL_SEED
FROZEN_EPOCHS = 800
FROZEN_STEPS_PER_EPOCH = 40
FROZEN_UPDATES = COVERAGE_STATE_PAET_FORMAL_UPDATES
FROZEN_ARTIFACT_FILE_COUNT = 14
FROZEN_REAL_DR_INPUTS = _bounded_cli.FROZEN_REAL_DR_INPUTS
FINAL_MODEL_DIRECTORY_NAME = "final_model"
_INCOMPLETE = ".incomplete"
_CONTROL_FILE_NAMES = frozenset(
    {
        "attempt.json",
        "STARTED.json",
        "COMPLETE.json",
        "FAILURE.json",
        _INCOMPLETE,
    }
)

_fingerprinted = _base_cli._fingerprinted
_write_new_json = _base_cli._write_new_json


def _verify_frozen_sources() -> dict[str, Path]:
    """Hash the frozen D_R source files without deserializing their tensors."""

    return _bounded_cli._verify_frozen_sources()


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    """Bind formal training, final artifact, replay, CLI, and wrapper code."""

    result = dict(_current_formal_implementation_binding())
    extras = (
        "cure_lite/experiment/coverage_state_bounded_protocol.py",
        (
            "cure_lite/experiment/"
            "coverage_state_paet_formal_artifacts.py"
        ),
        (
            "cure_lite/experiment/"
            "coverage_state_paet_formal_structural.py"
        ),
        "cure_lite/experiment/coverage_state_paet_decision.py",
        (
            "cure_lite/experiment/"
            "coverage_state_zero_level_evaluation.py"
        ),
        (
            "tools/"
            "run_coverage_state_paet_bfa_pmope_formal_800.py"
        ),
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
                f"Formal800 implementation path changed: {relative}"
            )
        result[relative] = file_sha256(absolute)
    return tuple(sorted(result.items()))


def _formal_model_config(
) -> CoverageStatePhaseAlignedEvidenceTransportConfig:
    config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=FROZEN_FEATURE_CHANNELS,
        feature_stride=FROZEN_FEATURE_STRIDE,
        width=FROZEN_MODEL_WIDTH,
    )
    if config.expected_parameter_count != FROZEN_PARAMETER_COUNT:
        raise RuntimeError("Formal800 PAET parameter contract changed")
    return config


def _static_config_payload(
    *,
    source_paths: Mapping[str, Path],
    implementation: tuple[tuple[str, str], ...],
    bounded_artifact_seal_fingerprint: str,
    source_closure: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if (
        not isinstance(bounded_artifact_seal_fingerprint, str)
        or len(bounded_artifact_seal_fingerprint) != 64
    ):
        raise ValueError("bounded artifact seal fingerprint is invalid")
    config = _formal_model_config()
    frozen_source_hashes = {
        name: digest for name, _, digest in FROZEN_REAL_DR_INPUTS
    }
    payload = {
        "schema_version": RUN_SCHEMA,
        "run_id": RUN_ID,
        "output_repo_path": OUTPUT_REPO_PATH,
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "real_inputs": {
            name: {
                "repo_path": str(path.relative_to(_ROOT)),
                "file_sha256": frozen_source_hashes[name],
            }
            for name, path in sorted(source_paths.items())
        },
        "bounded_artifact_seal_fingerprint": (
            bounded_artifact_seal_fingerprint
        ),
        "bounded_evidence_interpretation": (
            "structural_advancement_only_not_performance"
        ),
        "model": {
            "class": (
                "CURELitePhaseAlignedEvidenceTransportLevelSet"
            ),
            "candidate": "PAET-BFA-v21",
            "input_interface": ["F_b", "O"],
            "config": asdict(config),
            "feature_channels": config.feature_channels,
            "feature_stride": config.feature_stride,
            "width": config.width,
            "parameter_count": config.expected_parameter_count,
            "parameter_tensor_count": 3,
            "candidate_objective": "pmope_joint",
            "single_completion_field": True,
            "field_threshold_hex": 0.0.hex(),
            "threshold_search_performed": False,
        },
        "budget": {
            "seed": FROZEN_SEED,
            "epochs": FROZEN_EPOCHS,
            "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
            "updates": FROZEN_UPDATES,
            "objectives": 1,
            "from_scratch": True,
        },
        "full_D_R_contract": {
            "scalar_cache_fingerprint": (
                COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
            ),
            "formal_schedule_fingerprint": (
                COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
            ),
        },
        "execution": {
            "device": FROZEN_DEVICE,
            "CUDA_VISIBLE_DEVICES": FROZEN_VISIBLE_GPU,
            "CUBLAS_WORKSPACE_CONFIG": (
                FROZEN_CUBLAS_WORKSPACE_CONFIG
            ),
            "temperature_wrapper_repo_path": (
                "tools/run_with_gpu_temperature_control.py"
            ),
            "pause_temperature_c": FROZEN_PAUSE_TEMPERATURE_C,
            "resume_temperature_c": FROZEN_RESUME_TEMPERATURE_C,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        },
        "final_artifact": {
            "directory": FINAL_MODEL_DIRECTORY_NAME,
            "serialization": "safetensors",
            "checkpoint_policy": "final_model_only",
            "optimizer_state_saved": False,
            "intermediate_checkpoint_saved": False,
            "strict_loader_required": True,
            "training_and_module_state_fingerprints_separate": True,
        },
        "post_training_structural_replay": {
            "source": "same_full_D_R_cache_then_fixed_bounded_population",
            "population_seed": 42,
            "policy": "frozen_v21_bounded_structural_policy",
            "threshold_search_performed": False,
            "performance_evaluation": False,
            "D_V_authorized_only_if_structural_retention_passes": True,
            "generic_population_gate_reported_separately": True,
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
            "inference_performed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        },
    }
    if source_closure is not None:
        payload.update(_source_closure_fields(source_closure))
    return payload


def _source_closure_fields(receipt: Mapping[str, object]) -> dict[str, object]:
    """Normalize the sealed source-closure receipt for durable run records."""

    expected = {
        "sealed": True,
        "manifest_sha256": 64,
        "archive_sha256": 64,
        "content_fingerprint": 64,
    }
    for field, requirement in expected.items():
        value = receipt.get(field)
        if isinstance(requirement, bool):
            if value is not requirement:
                raise RuntimeError("Formal800 source closure is not sealed")
        elif not isinstance(value, str) or len(value) != requirement:
            raise RuntimeError(f"Formal800 source closure {field} is invalid")
    count = receipt.get("file_count")
    if not isinstance(count, int) or count < 1:
        raise RuntimeError("Formal800 source closure file count is invalid")
    return {
        "source_closure_manifest_sha256": receipt["manifest_sha256"],
        "source_closure_archive_sha256": receipt["archive_sha256"],
        "source_closure_content_fingerprint": receipt["content_fingerprint"],
        "source_closure_file_count": count,
    }


def validate_create_only() -> dict[str, object]:
    """Validate static bindings without claiming output or loading D_R."""

    source_closure = verify_coverage_state_paet_formal_source_closure()
    source_paths = _verify_frozen_sources()
    bounded_seal = (
        load_repository_coverage_state_paet_bounded_artifact_seal()
    )
    bounded_seal.verify_unchanged()
    implementation = _implementation_binding()
    config = _static_config_payload(
        source_paths=source_paths,
        implementation=implementation,
        bounded_artifact_seal_fingerprint=(
            bounded_seal.audit_fingerprint
        ),
        source_closure=source_closure,
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
            "bounded_artifact_seal_fingerprint": (
                bounded_seal.audit_fingerprint
            ),
            "bounded_structural_advancement_passed": (
                bounded_seal.structural_advancement_passed
            ),
            "bounded_generic_population_gate_passed": (
                bounded_seal.generic_population_gate_passed
            ),
            "output_exists": (
                OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink()
            ),
            "output_claimed": False,
            "D_R_source_files_hashed": True,
            "D_R_cached_tensor_payload_accessed": False,
            "real_inputs_constructed": False,
            "authorization_created": False,
            "training_performed": False,
            "final_artifact_saved": False,
            "structural_replay_performed": False,
            "D_V_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
            "not_a_formal_result": True,
            **_source_closure_fields(source_closure),
        }
    )


def _claim_output(
    output: Path,
) -> Path:
    """Claim the output before any source or D_R inspection."""

    output.mkdir(parents=True, exist_ok=False)
    (output / _INCOMPLETE).open("xb").close()
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    return receipts


def _scientific_artifact_hashes(root: Path) -> dict[str, str]:
    """Hash only completed scientific records, never attempt controls."""

    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(
                "Formal800 output may not contain a symbolic link"
            )
        if not path.is_file() or path.name in _CONTROL_FILE_NAMES:
            continue
        result[str(path.relative_to(root))] = file_sha256(path)
    return result


def _claim_payload(
    source_closure: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Return the D_R-free records written as the unique attempt claim."""

    attempt = _fingerprinted(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "run_id": RUN_ID,
            "output_repo_path": OUTPUT_REPO_PATH,
            "seed": FROZEN_SEED,
            "epochs": FROZEN_EPOCHS,
            "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
            "updates": FROZEN_UPDATES,
            "device": FROZEN_DEVICE,
            "CUDA_VISIBLE_DEVICES": FROZEN_VISIBLE_GPU,
            "CUBLAS_WORKSPACE_CONFIG": (
                FROZEN_CUBLAS_WORKSPACE_CONFIG
            ),
            "pause_temperature_c": FROZEN_PAUSE_TEMPERATURE_C,
            "resume_temperature_c": FROZEN_RESUME_TEMPERATURE_C,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            **_source_closure_fields(source_closure),
        }
    )
    started = _fingerprinted(
        {
            "schema_version": STARTED_SCHEMA,
            "run_id": RUN_ID,
            "status": "started_single_attempt",
            "attempt_fingerprint": attempt["receipt_fingerprint"],
            "output_directory_reusable": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    return attempt, started


def _failure_payload(
    error: BaseException,
    *,
    attempt_fingerprint: str | None,
    artifact_files: Mapping[str, str],
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": FAILURE_SCHEMA,
            "run_id": RUN_ID,
            "status": "failed_incomplete_single_attempt",
            "exception_type": type(error).__name__,
            "message": str(error),
            "attempt_fingerprint": attempt_fingerprint,
            "artifact_files_before_failure": dict(artifact_files),
            "output_directory_reusable": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_V_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
        }
    )


class _EpochProgressRecorder:
    """Durably mirror every trainer epoch callback without model state."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: BinaryIO | None = path.open("xb")
        self.rows: list[dict[str, object]] = []

    def __call__(
        self,
        objective: str,
        row: Mapping[str, object],
    ) -> None:
        if self._handle is None:
            raise RuntimeError("Formal800 epoch recorder is closed")
        expected_epoch = len(self.rows)
        normalized = json.loads(
            json.dumps(
                dict(row),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        if (
            objective != "pmope_joint"
            or normalized.get("objective") != "pmope_joint"
            or normalized.get("epoch") != expected_epoch
            or normalized.get("completed_updates")
            != (expected_epoch + 1) * FROZEN_STEPS_PER_EPOCH
        ):
            raise RuntimeError("Formal800 epoch progress is not canonical")
        event = {
            "schema_version": (
                "cure-lite-paet-bfa-v21-formal800-epoch-progress-v1"
            ),
            "run_id": RUN_ID,
            "objective": objective,
            "epoch_result": normalized,
        }
        encoded = (
            json.dumps(
                event,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        self._handle.write(encoded)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.rows.append(normalized)
        print(
            json.dumps(
                {
                    "event": "formal800_epoch_complete",
                    "run_id": RUN_ID,
                    "epoch": expected_epoch,
                    "completed_updates": normalized[
                        "completed_updates"
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )

    def close_and_verify(
        self,
        expected_rows: Sequence[Mapping[str, object]],
    ) -> None:
        if self._handle is None:
            raise RuntimeError("Formal800 epoch recorder closed twice")
        self._handle.close()
        self._handle = None
        normalized_expected = [
            json.loads(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            for row in expected_rows
        ]
        if (
            len(self.rows) != FROZEN_EPOCHS
            or self.rows != normalized_expected
        ):
            raise RuntimeError(
                "Formal800 callback rows differ from the final epoch ledger"
            )

    def close_after_failure(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _measure_formal_training(
    authorization: object,
    model_config: CoverageStatePhaseAlignedEvidenceTransportConfig,
    *,
    epoch_callback: _EpochProgressRecorder,
) -> tuple[object, dict[str, object]]:
    """Measure the single training invocation, including CUDA peak memory."""

    torch.cuda.synchronize(0)
    baseline_allocated = int(torch.cuda.memory_allocated(0))
    baseline_reserved = int(torch.cuda.memory_reserved(0))
    torch.cuda.reset_peak_memory_stats(0)
    started_ns = time.perf_counter_ns()
    result = run_coverage_state_paet_bfa_pmope_formal_800(
        authorization,
        model_config,
        device=FROZEN_DEVICE,
        epoch_callback=epoch_callback,
    )
    torch.cuda.synchronize(0)
    elapsed_ns = time.perf_counter_ns() - started_ns
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    if (
        elapsed_ns < 1
        or peak_allocated < baseline_allocated
        or peak_reserved < baseline_reserved
    ):
        raise RuntimeError("Formal800 resource measurement is invalid")
    measurement = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-paet-bfa-v21-formal800-"
                "training-resource-measurement-v1"
            ),
            "run_id": RUN_ID,
            "device": FROZEN_DEVICE,
            "scope": "single_formal_training_invocation",
            "updates": FROZEN_UPDATES,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": (
                peak_allocated - baseline_allocated
            ),
            "incremental_peak_reserved_bytes": (
                peak_reserved - baseline_reserved
            ),
            "elapsed_ns": elapsed_ns,
            "ns_per_update": elapsed_ns / FROZEN_UPDATES,
            "oom_observed": False,
            "training_invocations": 1,
            "performance_measurement": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    return result, measurement


def _final_model_member_hashes(directory: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("final-model artifact inventory changed")
        members[path.name] = file_sha256(path)
    if set(members) != {
        "model.safetensors",
        "formal_result.json",
        "training.json",
        "epoch_log.json",
        "receipt.json",
    }:
        raise RuntimeError("final-model artifact inventory is incomplete")
    return members


def _structural_decision_payload(
    *,
    formal_result: object,
    loaded_artifact: object,
    structural: object,
) -> dict[str, object]:
    training_complete = bool(formal_result.training_complete)
    artifact_bound = (
        loaded_artifact.formal_result_fingerprint
        == formal_result.result_fingerprint
        and loaded_artifact.authorization_fingerprint
        == formal_result.authorization.authorization_fingerprint
        and loaded_artifact.training_model_fingerprint
        == formal_result.training.results[0].final_model_fingerprint
        and loaded_artifact.module_state_fingerprint
        == structural.final_model_fingerprint
    )
    structural_passed = bool(
        structural.post_formal_structural_retention_passed
    )
    generic_passed = bool(
        structural.generic_population_gate_passed
    )
    d_v_authorized = (
        training_complete and artifact_bound and structural_passed
    )
    checks = {
        "formal_training_complete": training_complete,
        "strict_final_artifact_bound": artifact_bound,
        "structural_replay_invoked_once": (
            structural.evaluation_invocations == 1
        ),
        "frozen_paet_structural_retention_gate_passed": (
            structural_passed
        ),
        "D_V_and_D_T_not_accessed": True,
    }
    payload = {
        "schema_version": DECISION_SCHEMA,
        "run_id": RUN_ID,
        "status": (
            "PAET_BFA_V21_FORMAL800_STRUCTURAL_PASS_AUTHORIZE_D_V"
            if d_v_authorized
            else "PAET_BFA_V21_FORMAL800_STRUCTURAL_FAIL_STOP"
        ),
        "formal_training_complete": training_complete,
        "strict_final_artifact_bound": artifact_bound,
        "paet_structural_retention_gate_passed": structural_passed,
        "generic_zero_level_population_gate_passed": generic_passed,
        "generic_gate_is_D_V_prerequisite": False,
        "structural_gate_and_generic_gate_are_separate": True,
        "checks": checks,
        "failed_checks": [
            name for name, passed in checks.items() if not passed
        ],
        "D_V_authorized": d_v_authorized,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "performance_evaluation_performed": False,
        "performance_gate_passed": None,
        "performance_claim_supported": False,
        "final_model_success_established": False,
        "full_CURE_authorized": False,
        "cross_backbone_authorized": False,
        "next_action": (
            "RUN_ONE_SEPARATE_STRICT_D_V_REVEAL"
            if d_v_authorized
            else "STOP_AND_PRESERVE_FORMAL800_STRUCTURAL_RESULT"
        ),
        "bindings": {
            "authorization_fingerprint": (
                formal_result.authorization.authorization_fingerprint
            ),
            "formal_training_result_fingerprint": (
                formal_result.result_fingerprint
            ),
            "final_artifact_fingerprint": (
                loaded_artifact.artifact_fingerprint
            ),
            "training_model_fingerprint": (
                loaded_artifact.training_model_fingerprint
            ),
            "module_state_fingerprint": (
                loaded_artifact.module_state_fingerprint
            ),
            "structural_result_fingerprint": (
                structural.result_fingerprint
            ),
        },
    }
    return _fingerprinted(payload, field="decision_fingerprint")


def run_once() -> dict[str, object]:
    """Consume the sole fixed seed-42 Formal800 D_R attempt."""

    # mkdir(exist_ok=False) is the cross-process lock.  It must precede every
    # frozen-input read, cache construction, optimizer construction, and
    # training action: a rejected contender may not even inspect D_R.
    receipts = _claim_output(OUTPUT_PATH)
    attempt: dict[str, object] | None = None
    started: dict[str, object] | None = None
    progress: _EpochProgressRecorder | None = None
    try:
        # The output directory is already the cross-process attempt claim;
        # closure verification remains strictly before any D_R read.
        source_closure = verify_coverage_state_paet_formal_source_closure()
        attempt, started = _claim_payload(source_closure)
        _write_new_json(OUTPUT_PATH / "attempt.json", attempt)
        _write_new_json(OUTPUT_PATH / "STARTED.json", started)
        source_paths = _verify_frozen_sources()
        bounded_seal = (
            load_repository_coverage_state_paet_bounded_artifact_seal()
        )
        bounded_seal.verify_unchanged()
        implementation = _implementation_binding()
        runtime = _base_cli._verify_runtime_contract()
        config = _fingerprinted(
            _static_config_payload(
                source_paths=source_paths,
                implementation=implementation,
                bounded_artifact_seal_fingerprint=(
                    bounded_seal.audit_fingerprint
                ),
                source_closure=source_closure,
            )
        )
        _write_new_json(receipts / "config.json", config)
        real_inputs = build_coverage_state_real_dr_inputs(**source_paths)
        real_inputs.verify_unchanged()
        if (
            real_inputs.scalar_cache.cache_fingerprint
            != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        ):
            raise RuntimeError("Formal800 full D_R cache changed")
        inputs_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-formal800-inputs-v1"
                ),
                "run_id": RUN_ID,
                "real_D_R_inputs": real_inputs.canonical_payload(),
                "real_inputs_build_fingerprint": (
                    real_inputs.build_fingerprint
                ),
                "full_D_R_scalar_cache_fingerprint": (
                    real_inputs.scalar_cache.cache_fingerprint
                ),
                "construction_invocations": 1,
                "bounded_population_constructed_before_training": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(receipts / "inputs.json", inputs_receipt)

        model_config = expected_coverage_state_paet_formal_config(
            real_inputs
        )
        if model_config != _formal_model_config():
            raise RuntimeError("Formal800 model configuration changed")
        authorization = (
            prepare_coverage_state_paet_formal_800_authorization(
                real_inputs,
                model_config,
                bounded_artifact_seal=bounded_seal,
                run_id=RUN_ID,
            )
        )
        authorization.verify_unchanged()
        if not authorization.formal_training_authorized:
            raise PermissionError("Formal800 training was not authorized")
        authorization_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-formal800-authorization-v1"
                ),
                "run_id": RUN_ID,
                "authorization": authorization.canonical_payload(),
                "authorization_fingerprint": (
                    authorization.authorization_fingerprint
                ),
                "config_fingerprint": config["receipt_fingerprint"],
                "implementation_fingerprint": stable_fingerprint(
                    dict(implementation)
                ),
                "formal_training_authorized": True,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(
            receipts / "authorization.json",
            authorization_receipt,
        )

        progress = _EpochProgressRecorder(
            receipts / "epoch_progress.jsonl"
        )
        formal_result, resource = _measure_formal_training(
            authorization,
            model_config,
            epoch_callback=progress,
        )
        if (
            not formal_result.training_complete
            or formal_result.training_invocations != 1
            or len(formal_result.training.results) != 1
            or len(formal_result.training.models) != 1
        ):
            raise RuntimeError("Formal800 returned an incomplete result")
        progress.close_and_verify(
            formal_result.training.results[0].epoch_logs
        )
        _write_new_json(
            receipts / "training_resource.json",
            resource,
        )

        final_directory = OUTPUT_PATH / FINAL_MODEL_DIRECTORY_NAME
        artifact_fingerprint = (
            save_coverage_state_paet_formal_artifact(
                final_directory,
                formal_result,
            )
        )
        loaded = load_coverage_state_paet_formal_artifact(
            final_directory,
            expected_authorization_fingerprint=(
                authorization.authorization_fingerprint
            ),
            expected_result_fingerprint=(
                formal_result.result_fingerprint
            ),
        )
        loaded.verify_unchanged()
        if artifact_fingerprint != loaded.artifact_fingerprint:
            raise RuntimeError(
                "Formal800 final-artifact save/load fingerprints differ"
            )
        training_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-formal800-training-v1"
                ),
                "run_id": RUN_ID,
                "formal_result": formal_result.canonical_payload(),
                "formal_training_result_fingerprint": (
                    formal_result.result_fingerprint
                ),
                "authorization_fingerprint": (
                    authorization.authorization_fingerprint
                ),
                "training_invocations": 1,
                "completed_updates": FROZEN_UPDATES,
                "epoch_callback_rows": len(progress.rows),
                "resource_measurement_fingerprint": (
                    resource["receipt_fingerprint"]
                ),
                "final_artifact_fingerprint": artifact_fingerprint,
                "training_model_fingerprint": (
                    loaded.training_model_fingerprint
                ),
                "module_state_fingerprint": (
                    loaded.module_state_fingerprint
                ),
                "D_V_accessed": False,
                "D_T_accessed": False,
                "performance_evaluation_performed": False,
            }
        )
        _write_new_json(
            receipts / "formal_training.json",
            training_receipt,
        )
        artifact_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-formal800-"
                    "final-artifact-binding-v1"
                ),
                "run_id": RUN_ID,
                "artifact_repo_path": str(
                    final_directory.relative_to(_ROOT)
                ),
                "artifact_fingerprint": (
                    loaded.artifact_fingerprint
                ),
                "artifact_receipt_sha256": loaded.receipt_sha256,
                "authorization_fingerprint": (
                    loaded.authorization_fingerprint
                ),
                "formal_result_fingerprint": (
                    loaded.formal_result_fingerprint
                ),
                "training_model_fingerprint": (
                    loaded.training_model_fingerprint
                ),
                "module_state_fingerprint": (
                    loaded.module_state_fingerprint
                ),
                "member_files": _final_model_member_hashes(
                    final_directory
                ),
                "strict_loader_verified": True,
                "checkpoint_policy": "final_model_only",
                "optimizer_state_saved": False,
                "intermediate_checkpoint_saved": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(
            receipts / "final_artifact.json",
            artifact_receipt,
        )

        population = build_coverage_state_bounded_population(
            real_inputs.scalar_cache,
            seed=FROZEN_SEED,
        )
        structural = (
            evaluate_coverage_state_paet_formal_structural_retention(
                formal_result,
                population,
                device=FROZEN_DEVICE,
            )
        )
        structural_payload = structural.canonical_payload()
        structural_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-paet-bfa-v21-formal800-"
                    "structural-replay-v1"
                ),
                "run_id": RUN_ID,
                "source_full_D_R_scalar_cache_fingerprint": (
                    real_inputs.scalar_cache.cache_fingerprint
                ),
                "bounded_population": population.canonical_payload(),
                "bounded_population_fingerprint": (
                    population.population_fingerprint
                ),
                "structural_result": structural_payload,
                "structural_result_fingerprint": (
                    structural.result_fingerprint
                ),
                "training_model_fingerprint": (
                    loaded.training_model_fingerprint
                ),
                "module_state_fingerprint": (
                    loaded.module_state_fingerprint
                ),
                "evaluation_invocations": (
                    structural.evaluation_invocations
                ),
                "paet_structural_retention_gate_passed": (
                    structural.post_formal_structural_retention_passed
                ),
                "generic_zero_level_population_gate_passed": (
                    structural.generic_population_gate_passed
                ),
                "performance_evaluation_performed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        _write_new_json(
            receipts / "structural_replay.json",
            structural_receipt,
        )
        decision = _structural_decision_payload(
            formal_result=formal_result,
            loaded_artifact=loaded,
            structural=structural,
        )
        _write_new_json(receipts / "decision.json", decision)

        loaded.verify_unchanged()
        real_inputs.verify_unchanged()
        population.verify_unchanged()
        bounded_seal.verify_unchanged()
        if _implementation_binding() != implementation:
            raise RuntimeError(
                "Formal800 implementation changed during execution"
            )
        closing_source_closure = (
            verify_coverage_state_paet_formal_source_closure()
        )
        if closing_source_closure != source_closure:
            raise RuntimeError(
                "Formal800 source closure changed during execution"
            )
        artifact_files = _scientific_artifact_hashes(OUTPUT_PATH)
        if len(artifact_files) != FROZEN_ARTIFACT_FILE_COUNT:
            raise RuntimeError(
                "Formal800 terminal artifact population is incomplete"
            )
        complete = _fingerprinted(
            {
                "schema_version": RUN_SCHEMA,
                "run_id": RUN_ID,
                "status": "complete",
                "decision": decision["status"],
                "formal_training_complete": (
                    formal_result.training_complete
                ),
                "formal_training_result_fingerprint": (
                    formal_result.result_fingerprint
                ),
                "final_artifact_fingerprint": (
                    loaded.artifact_fingerprint
                ),
                "artifact_receipt_sha256": loaded.receipt_sha256,
                "training_model_fingerprint": (
                    loaded.training_model_fingerprint
                ),
                "module_state_fingerprint": (
                    loaded.module_state_fingerprint
                ),
                "structural_result_fingerprint": (
                    structural.result_fingerprint
                ),
                "paet_structural_retention_gate_passed": (
                    structural.post_formal_structural_retention_passed
                ),
                "generic_zero_level_population_gate_passed": (
                    structural.generic_population_gate_passed
                ),
                "structural_gate_and_generic_gate_are_separate": True,
                "D_V_authorized": decision["D_V_authorized"],
                "D_V_accessed": False,
                "D_T_accessed": False,
                "performance_evaluation_performed": False,
                "performance_gate_passed": None,
                "performance_claim_supported": False,
                "artifact_files": artifact_files,
                "artifact_file_count": len(artifact_files),
                "single_attempt": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "full_CURE_authorized": False,
                "cross_backbone_authorized": False,
                "attempt_fingerprint": attempt["receipt_fingerprint"],
                "started_fingerprint": started["receipt_fingerprint"],
                **_source_closure_fields(source_closure),
            },
            field="complete_fingerprint",
        )
        _write_new_json(OUTPUT_PATH / "COMPLETE.json", complete)
        (OUTPUT_PATH / _INCOMPLETE).unlink()
        return {
            "run_id": RUN_ID,
            "output": str(OUTPUT_PATH),
            "decision": decision["status"],
            "formal_training_complete": (
                formal_result.training_complete
            ),
            "paet_structural_retention_gate_passed": (
                structural.post_formal_structural_retention_passed
            ),
            "generic_zero_level_population_gate_passed": (
                structural.generic_population_gate_passed
            ),
            "D_V_authorized": decision["D_V_authorized"],
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
            "complete_fingerprint": complete["complete_fingerprint"],
        }
    except BaseException as error:
        if progress is not None:
            progress.close_after_failure()
        try:
            failure = _failure_payload(
                error,
                attempt_fingerprint=(
                    None
                    if attempt is None
                    else str(attempt["receipt_fingerprint"])
                ),
                artifact_files=_scientific_artifact_hashes(OUTPUT_PATH),
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
        help="consume the unique temperature-controlled Formal800 attempt",
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
