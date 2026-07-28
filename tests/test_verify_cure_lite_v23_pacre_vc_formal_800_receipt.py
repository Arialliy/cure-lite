from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest
import torch

import cure_lite_v23.formal_artifacts as formal_artifacts
from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_sobolev import CSLF_PMOPE_POLICY
from cure_lite.experiment.coverage_state_training import (
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite_v23.factory import (
    PACRE_VC_PARAMETER_NAMES,
    build_pacre_vc_training_model,
)
from cure_lite_v23.formal_artifacts import (
    save_pacre_vc_formal_final_model,
)
from cure_lite_v23.formal_training import (
    PACRE_VC_FORMAL_AUTHORIZATION_SCHEMA,
    PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT,
)
from cure_lite_v23.pacre_vc import (
    CoverageStatePACREVerifierCorrectedConfig,
)
from cure_lite_v23.protocol import (
    fingerprinted,
    strict_json_bytes,
    write_new_json,
)
from cure_lite_v23.training import (
    PACRE_PMOPE_OBJECTIVE,
    PACRE_PMOPE_TRAINING_CONFIG,
)
from tools import (
    verify_cure_lite_v23_pacre_vc_formal_800_receipt as verifier,
)


class _FakeDRReceipt:
    receipt_fingerprint = "d" * 64
    source_closure_fingerprint = "c" * 64
    dataset_free_receipt_fingerprint = "e" * 64
    checks = tuple((name, True) for name in verifier.PACRE_VC_DR_CHECK_NAMES)
    gate_passed = True
    decision = verifier.PACRE_VC_DR_PASS_DECISION


class _FakeRealInputs:
    def __init__(self) -> None:
        self.source_binding = SimpleNamespace(
            split="D_R",
            binding_fingerprint="b" * 64,
        )
        self.scalar_cache = SimpleNamespace(
            cache_fingerprint=(
                verifier.PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
            ),
            raw_catalog=SimpleNamespace(split="D_R"),
        )
        self.build_fingerprint = "i" * 64
        self.verify_count = 0

    def verify_unchanged(self) -> None:
        self.verify_count += 1

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "synthetic-full-D_R-input-v1",
            "split": "D_R",
            "cache_fingerprint": (
                verifier.PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
            ),
            "D_V_accessed": False,
            "D_T_accessed": False,
        }


class _FakeAuthorization:
    def __init__(
        self,
        *,
        attempt_fingerprint: str,
        real_inputs: _FakeRealInputs,
    ) -> None:
        self.output_claim_fingerprint = attempt_fingerprint
        self.real_inputs = real_inputs
        self.schedule = SimpleNamespace(
            schedule_fingerprint=(
                verifier.PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT
            )
        )
        self.initial_model_fingerprint = (
            PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT
        )
        self.prerequisites_passed = True
        self.available = True
        self.verify_count = 0
        self._payload = {
            "schema_version": PACRE_VC_FORMAL_AUTHORIZATION_SCHEMA,
            "run_id": verifier.RUN_ID,
            "output_claim_fingerprint": attempt_fingerprint,
            "scope": "D_R_formal_800",
            "runtime_splits": ["D_R"],
            "budget": {
                "seed": 42,
                "epochs": 800,
                "steps_per_epoch": 40,
                "updates": 32_000,
                "objectives": 1,
            },
            "training_contract": {
                "from_scratch": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
            },
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
        }
        self.authorization_fingerprint = stable_fingerprint(self._payload)
        self.source_closure_fingerprint = "c" * 64

    def canonical_payload(self) -> dict[str, object]:
        return dict(self._payload)

    def verify_unchanged(self) -> None:
        self.verify_count += 1


class _SyntheticFormalRunResult:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        training_result: CoverageStateTrainingResult,
        authorization: _FakeAuthorization,
        result_fingerprint: str,
    ) -> None:
        self.model = model
        self.final_model = model
        self.training_result = training_result
        self.authorization = authorization
        self.source_closure_fingerprint_after = "c" * 64
        self.training_complete = True
        self.result_fingerprint = result_fingerprint

    def verify_unchanged(self) -> None:
        return None


def _runtime() -> dict[str, object]:
    return {
        "device": "cuda:0",
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "cuda_available": True,
        "visible_cuda_device_count": 1,
        "current_cuda_logical_device": 0,
        "temperature_wrapper_repo_path": "tools/wrapper.py",
        "temperature_wrapper_file_sha256": "w" * 64,
        "pause_temperature_c": 82,
        "resume_temperature_c": 75,
        "temperature_wrapper_parent_verified": True,
    }


def _epoch_row(epoch: int) -> dict[str, object]:
    return {
        "epoch": epoch,
        "completed_updates": (epoch + 1) * 40,
        "objective": PACRE_PMOPE_OBJECTIVE,
        "selection_sequence_fingerprint": stable_fingerprint(
            {"epoch": epoch}
        ),
        "mean_factual_miss/loss": 1.0,
        "mean_factual_no_miss/loss": 1.0,
        "mean_pair/loss": 2.0,
        "mean_total": 2.0,
        "mean_gradient_l2_norm": 0.5,
    }


def _replace_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_bytes(strict_json_bytes(payload))


def _rehash_complete(output: Path) -> None:
    path = output / "COMPLETE.json"
    complete = json.loads(path.read_text(encoding="utf-8"))
    complete.pop("complete_fingerprint", None)
    complete["artifact_files"] = {
        name: file_sha256(output / name)
        for name in sorted(verifier.EXPECTED_ARTIFACT_FILES)
    }
    complete["artifact_count"] = len(complete["artifact_files"])
    _replace_json(
        path,
        fingerprinted(complete, field="complete_fingerprint"),
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, _FakeAuthorization]:
    output = tmp_path / verifier.RUN_ID
    receipts = output / "receipts"
    receipts.mkdir(parents=True)
    real_inputs = _FakeRealInputs()
    config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    live = verifier._LivePrerequisites(
        runtime=_runtime(),
        source_closure={
            "closure_fingerprint": "c" * 64,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
        },
        source_closure_fingerprint="c" * 64,
        dataset_free={
            "receipt_fingerprint": "e" * 64,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
        },
        dataset_free_fingerprint="e" * 64,
        dr_verification={
            "run_id": "pacre_v23_verifier_corrected_D_R_structural_r1",
            "output": "/synthetic/D_R",
            "decision": verifier.PACRE_VC_DR_PASS_DECISION,
            "gate_passed": True,
            "failed_checks": [],
            "receipt_fingerprint": "d" * 64,
            "formal_800_route_granted": True,
            "bounded_400_required": False,
            "bounded_400_authorization_effect": False,
        },
        dr_verification_fingerprint=stable_fingerprint(
            {"synthetic": "D_R-terminal"}
        ),
        dr_receipt=_FakeDRReceipt(),  # type: ignore[arg-type]
    )
    attempt = fingerprinted(
        {
            "schema_version": verifier.ATTEMPT_SCHEMA,
            "run_id": verifier.RUN_ID,
            "output_repo_path": verifier.OUTPUT_REPO_PATH,
            "candidate": "PACRE-VC-v23",
            "objective": PACRE_PMOPE_OBJECTIVE,
            "runtime": live.runtime,
            "budget": {
                "seed": 42,
                "epochs": 800,
                "steps_per_epoch": 40,
                "updates": 32_000,
                "from_scratch": True,
                "training_invocations": 1,
            },
            "source_closure_fingerprint": "c" * 64,
            "dataset_free_receipt_fingerprint": "e" * 64,
            "D_R_terminal_verification_fingerprint": (
                live.dr_verification_fingerprint
            ),
            "D_R_gate_receipt_fingerprint": "d" * 64,
            "D_R_gate_check_count": 13,
            "D_R_gate_passed": True,
            "bounded_400_output_absent": True,
            "bounded_400_required": False,
            "bounded_400_authorization_effect": False,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "overwrite_allowed": False,
            "checkpoint_policy": "final_model_only",
            "D_R_receipt_metadata_read": True,
            "D_R_tensor_payload_accessed": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
        }
    )
    write_new_json(output / "attempt.json", attempt)
    attempt_fingerprint = str(attempt["receipt_fingerprint"])

    dr_terminal = fingerprinted(
        {
            "schema_version": verifier.DR_TERMINAL_SCHEMA,
            "run_id": verifier.RUN_ID,
            "attempt_fingerprint": attempt_fingerprint,
            "verification": live.dr_verification,
            "verification_fingerprint": live.dr_verification_fingerprint,
            "D_R_gate_receipt_fingerprint": "d" * 64,
            "D_R_gate_check_count": 13,
            "D_R_gate_passed": True,
            "D_R_reopened": False,
            "D_R_tensor_payload_accessed": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "training_performed": False,
        }
    )
    write_new_json(
        receipts / "dr_terminal_verification.json",
        dr_terminal,
    )
    config_receipt = fingerprinted(
        {
            "schema_version": verifier.CONFIG_SCHEMA,
            "run_id": verifier.RUN_ID,
            "candidate": "PACRE-VC-v23",
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "objective": PACRE_PMOPE_OBJECTIVE,
            "model_config": asdict(config),
            "parameter_count": 64_064,
            "field_threshold_hex": 0.0.hex(),
            "threshold_search_performed": False,
            "budget": {
                "seed": 42,
                "epochs": 800,
                "steps_per_epoch": 40,
                "updates": 32_000,
                "from_scratch": True,
                "training_invocations": 1,
            },
            "runtime": live.runtime,
            "checkpoint_policy": "final_model_only",
            "optimizer_state_saved": False,
            "intermediate_checkpoint_saved": False,
            "bounded_400_required": False,
            "bounded_400_authorization_effect": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
        }
    )
    write_new_json(receipts / "config.json", config_receipt)
    inputs = fingerprinted(
        {
            "schema_version": verifier.INPUTS_SCHEMA,
            "run_id": verifier.RUN_ID,
            "attempt_fingerprint": attempt_fingerprint,
            "real_inputs": real_inputs.canonical_payload(),
            "real_inputs_fingerprint": real_inputs.build_fingerprint,
            "full_D_R_scalar_cache_fingerprint": (
                real_inputs.scalar_cache.cache_fingerprint
            ),
            "source_binding_fingerprint": (
                real_inputs.source_binding.binding_fingerprint
            ),
            "source_files": {},
            "construction_invocations": 1,
            "D_R_accessed": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "split_manifest_metadata_read": True,
            "D_R_tensor_payload_accessed": True,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "training_performed": False,
        }
    )
    write_new_json(receipts / "inputs.json", inputs)
    authorization = _FakeAuthorization(
        attempt_fingerprint=attempt_fingerprint,
        real_inputs=real_inputs,
    )
    authorization_wrapper = fingerprinted(
        {
            "schema_version": verifier.AUTHORIZATION_WRAPPER_SCHEMA,
            "run_id": verifier.RUN_ID,
            "attempt_fingerprint": attempt_fingerprint,
            "authorization": authorization.canonical_payload(),
            "authorization_fingerprint": (
                authorization.authorization_fingerprint
            ),
            "formal_D_R_training_authorized": True,
            "bounded_400_receipt_consumed": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "performance_evaluation_performed": False,
        }
    )
    write_new_json(
        receipts / "authorization.json",
        authorization_wrapper,
    )

    torch.manual_seed(123)
    model = build_pacre_vc_training_model(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=PACRE_PMOPE_TRAINING_CONFIG.learning_rate,
        betas=(
            PACRE_PMOPE_TRAINING_CONFIG.adam_beta1,
            PACRE_PMOPE_TRAINING_CONFIG.adam_beta2,
        ),
        eps=PACRE_PMOPE_TRAINING_CONFIG.adam_epsilon,
        weight_decay=PACRE_PMOPE_TRAINING_CONFIG.weight_decay,
    )
    optimizer_fingerprint = coverage_state_optimizer_config_fingerprint(
        model,
        optimizer,
    )
    epoch_rows = tuple(_epoch_row(epoch) for epoch in range(800))
    training_result = CoverageStateTrainingResult(
        objective=PACRE_PMOPE_OBJECTIVE,
        objective_policy=CSLF_PMOPE_POLICY,
        seed=42,
        epochs=800,
        steps_per_epoch=40,
        completed_updates=32_000,
        schedule_fingerprint=(
            verifier.PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT
        ),
        cache_fingerprint=(
            verifier.PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
        ),
        execution_device="cuda:0",
        device_cache_fingerprint="f" * 64,
        device_cache_resident_bytes=1,
        optimizer_config_fingerprint=optimizer_fingerprint,
        initial_model_fingerprint=(
            PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT
        ),
        final_model_fingerprint=coverage_state_model_fingerprint(model),
        epoch_logs=epoch_rows,
        first_nonzero_gradient_update=tuple(
            (name, 0) for name in PACRE_VC_PARAMETER_NAMES
        ),
        forward_calls=32_000,
        backward_calls=32_000,
        optimizer_steps=32_000,
        logical_state_evaluations=384_000,
        finite_state_audits=32_001,
    )
    formal = {
        "schema_version": verifier.PACRE_VC_FORMAL_RESULT_SCHEMA,
        "run_id": verifier.RUN_ID,
        "runtime_splits": ["D_R"],
        "authorization_fingerprint": (
            authorization.authorization_fingerprint
        ),
        "training_result": training_result.canonical_payload(),
        "training_result_fingerprint": training_result.result_fingerprint,
        "final_model_fingerprint": training_result.final_model_fingerprint,
        "optimizer_config_fingerprint": optimizer_fingerprint,
        "source_closure_fingerprint_after": "c" * 64,
        "training_invocations": 1,
        "checks": {
            name: True for name in sorted(verifier._FORMAL_RESULT_CHECKS)
        },
        "training_complete": True,
        "output_contract": {
            "single_final_model": True,
            "checkpoint_written": False,
            "optimizer_state_returned": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        },
        "bounded_400_required": False,
        "bounded_400_is_final_success": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "calibration_performed": False,
        "inference_performed": False,
        "performance_evaluation_performed": False,
        "performance_claim_supported": False,
        "full_CURE_authorized": False,
        "cross_backbone_authorized": False,
    }
    formal_fingerprint = stable_fingerprint(formal)
    monkeypatch.setattr(
        formal_artifacts,
        "CoverageStatePACREVCFormal800RunResult",
        _SyntheticFormalRunResult,
    )
    artifact = save_pacre_vc_formal_final_model(
        output / "final_model",
        formal_result=_SyntheticFormalRunResult(
            model=model,
            training_result=training_result,
            authorization=authorization,
            result_fingerprint=formal_fingerprint,
        ),
    )
    with (receipts / "epoch_progress.jsonl").open("xb") as handle:
        for row in epoch_rows:
            handle.write(
                strict_json_bytes(
                    fingerprinted(
                        {
                            "schema_version": (
                                verifier.EPOCH_PROGRESS_SCHEMA
                            ),
                            "run_id": verifier.RUN_ID,
                            "epoch_result": row,
                        }
                    )
                )
            )
    training = fingerprinted(
        {
            "schema_version": verifier.TRAINING_SCHEMA,
            "run_id": verifier.RUN_ID,
            "attempt_fingerprint": attempt_fingerprint,
            "authorization_fingerprint": (
                authorization.authorization_fingerprint
            ),
            "formal_result": formal,
            "formal_result_fingerprint": formal_fingerprint,
            "training_result_fingerprint": (
                training_result.result_fingerprint
            ),
            "final_model_artifact_fingerprint": (
                artifact["artifact_fingerprint"]
            ),
            "compute_ledger": {
                "seed": 42,
                "epochs": 800,
                "steps_per_epoch": 40,
                "completed_updates": 32_000,
                "forward_calls": 32_000,
                "backward_calls": 32_000,
                "optimizer_steps": 32_000,
                "logical_state_evaluations": 384_000,
                "finite_state_audits": 32_001,
                "epoch_progress_rows": 800,
                "training_invocations": 1,
            },
            "final_checkpoint_only": True,
            "optimizer_state_saved": False,
            "intermediate_checkpoint_saved": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
        }
    )
    write_new_json(receipts / "training.json", training)
    decision = fingerprinted(
        {
            "schema_version": verifier.DECISION_SCHEMA,
            "run_id": verifier.RUN_ID,
            "status": verifier.TERMINAL_STATUS,
            "attempt_fingerprint": attempt_fingerprint,
            "authorization_fingerprint": (
                authorization.authorization_fingerprint
            ),
            "formal_result_fingerprint": formal_fingerprint,
            "final_model_artifact_fingerprint": (
                artifact["artifact_fingerprint"]
            ),
            "formal_training_complete": True,
            "compute_ledger_complete": True,
            "bounded_400_required": False,
            "bounded_400_authorization_effect": False,
            "D_V_preregistration_eligible": True,
            "D_V_execution_authorized": False,
            "D_T_execution_authorized": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "performance_evaluation_performed": False,
            "performance_gate_passed": None,
            "performance_claim_supported": False,
            "final_model_performance_success_established": False,
            "next_action": verifier.NEXT_ACTION,
        }
    )
    write_new_json(receipts / "decision.json", decision)
    artifact_files = {
        name: file_sha256(output / name)
        for name in sorted(verifier.EXPECTED_ARTIFACT_FILES)
    }
    complete = fingerprinted(
        {
            "schema_version": verifier.COMPLETE_SCHEMA,
            "run_id": verifier.RUN_ID,
            "status": verifier.TERMINAL_STATUS,
            "attempt_fingerprint": attempt_fingerprint,
            "authorization_fingerprint": (
                authorization.authorization_fingerprint
            ),
            "formal_result_fingerprint": formal_fingerprint,
            "final_model_artifact_fingerprint": (
                artifact["artifact_fingerprint"]
            ),
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifact_files,
            "artifact_count": len(artifact_files),
            "seed": 42,
            "epochs": 800,
            "steps_per_epoch": 40,
            "updates": 32_000,
            "training_invocations": 1,
            "final_checkpoint_only": True,
            "optimizer_state_saved": False,
            "intermediate_checkpoint_saved": False,
            "bounded_400_required": False,
            "bounded_400_authorization_effect": False,
            "D_V_preregistration_eligible": True,
            "D_V_execution_authorized": False,
            "D_T_execution_authorized": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
        },
        field="complete_fingerprint",
    )
    write_new_json(output / "COMPLETE.json", complete)

    monkeypatch.setattr(
        verifier,
        "BOUNDED_OUTPUT_PATH",
        tmp_path / "bounded-absent",
    )
    monkeypatch.setattr(
        verifier,
        "stabilize_pacre_vc_numerical_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        verifier,
        "_verify_live_prerequisites",
        lambda: live,
    )
    monkeypatch.setattr(
        verifier,
        "verify_source_closure",
        lambda closure: "c" * 64,
    )
    monkeypatch.setattr(
        verifier,
        "frozen_real_dr_source_paths",
        lambda: {},
    )
    monkeypatch.setattr(
        verifier,
        "build_coverage_state_real_dr_inputs",
        lambda **kwargs: real_inputs,
    )
    monkeypatch.setattr(
        verifier,
        "expected_pacre_vc_formal_config",
        lambda rebuilt: config,
    )
    monkeypatch.setattr(
        verifier,
        "prepare_pacre_vc_formal_800_authorization",
        lambda *args, **kwargs: authorization,
    )
    return output, authorization


def test_verify_terminal_accepts_exact_synthetic_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, authorization = _fixture(tmp_path, monkeypatch)
    result = verifier.verify_terminal(output)
    assert result["schema_version"] == verifier.VERIFICATION_SCHEMA
    assert result["run_id"] == verifier.RUN_ID
    assert result["epochs"] == 800
    assert result["updates"] == 32_000
    assert result["authorization_fingerprint"] == (
        authorization.authorization_fingerprint
    )
    assert result["D_V_preregistration_eligible"] is True
    assert result["D_V_execution_authorized"] is False
    assert result["D_T_execution_authorized"] is False
    assert result["performance_claim_supported"] is False


def test_verify_terminal_issues_artifact_identity_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _fixture(tmp_path, monkeypatch)
    sealed = verifier.verify_terminal_sealed(output)
    assert (
        type(sealed)
        is formal_artifacts.VerifiedPACREVCFormalTerminal
    )
    assert sealed.artifact.directory == output / "final_model"
    assert sealed.verification["epochs"] == 800
    assert sealed.verification["updates"] == 32_000
    sealed.verify_unchanged()


def test_verify_terminal_rejects_extra_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _fixture(tmp_path, monkeypatch)
    (output / "extra.txt").write_text("forbidden", encoding="utf-8")
    with pytest.raises(RuntimeError, match="population"):
        verifier.verify_terminal(output)


def test_verify_terminal_rejects_rehashed_D_T_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _fixture(tmp_path, monkeypatch)
    path = output / "receipts/decision.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision.pop("receipt_fingerprint")
    decision["D_T_execution_authorized"] = True
    decision = fingerprinted(decision)
    _replace_json(path, decision)
    complete_path = output / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete.pop("complete_fingerprint")
    complete["decision_fingerprint"] = decision["receipt_fingerprint"]
    _replace_json(
        complete_path,
        fingerprinted(complete, field="complete_fingerprint"),
    )
    _rehash_complete(output)
    with pytest.raises(PermissionError, match="exceeds training"):
        verifier.verify_terminal(output)


def test_verify_terminal_rejects_rehashed_short_epoch_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _fixture(tmp_path, monkeypatch)
    path = output / "receipts/epoch_progress.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))
    _rehash_complete(output)
    with pytest.raises(ValueError, match="800 rows"):
        verifier.verify_terminal(output)


def test_verify_terminal_rejects_rehashed_performance_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _fixture(tmp_path, monkeypatch)
    path = output / "receipts/training.json"
    training = json.loads(path.read_text(encoding="utf-8"))
    training.pop("receipt_fingerprint")
    training["performance_claim_supported"] = True
    _replace_json(path, fingerprinted(training))
    _rehash_complete(output)
    with pytest.raises(PermissionError, match="forbidden"):
        verifier.verify_terminal(output)
