from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite_v23.pacre_vc import (
    CoverageStatePACREVerifierCorrectedConfig,
)
from cure_lite_v23.protocol import fingerprinted, write_new_json
from tools import run_cure_lite_v23_pacre_vc_formal_800 as cli


class _FakeDRReceipt:
    receipt_fingerprint = "d" * 64
    source_closure_fingerprint = "c" * 64
    dataset_free_receipt_fingerprint = "e" * 64
    checks = tuple((f"check_{index:02d}", True) for index in range(13))
    gate_passed = True


def _static_context() -> cli._StaticContext:
    return cli._StaticContext(
        runtime={
            "device": "cuda:0",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "pause_temperature_c": 82,
            "resume_temperature_c": 75,
            "temperature_wrapper_parent_verified": True,
        },
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
            "gate_passed": True,
            "decision": "PACRE_VC_D_R_13_OF_13_PASS",
            "failed_checks": [],
            "receipt_fingerprint": "d" * 64,
            "formal_800_route_granted": True,
            "bounded_400_required": False,
            "bounded_400_authorization_effect": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
        dr_verification_fingerprint=stable_fingerprint(
            {"terminal_D_R": "13_of_13_PASS"}
        ),
        dr_receipt=_FakeDRReceipt(),  # type: ignore[arg-type]
    )


class _FakeRealInputs:
    def __init__(self) -> None:
        self.source_binding = SimpleNamespace(
            split="D_R",
            binding_fingerprint="b" * 64,
        )
        self.scalar_cache = SimpleNamespace(
            cache_fingerprint=(
                cli.PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
            ),
            raw_catalog=SimpleNamespace(split="D_R"),
        )
        self.build_fingerprint = "i" * 64
        self.verify_calls = 0

    def verify_unchanged(self) -> None:
        self.verify_calls += 1

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fake-real-D_R-inputs-v1",
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "execution_policy": {
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
        }


class _FakeAuthorization:
    def __init__(self, output_claim_fingerprint: str) -> None:
        self.output_claim_fingerprint = output_claim_fingerprint
        self.authorization_fingerprint = "a" * 64
        self.source_closure_fingerprint = "c" * 64
        self.prerequisites_passed = True
        self.available = True
        self.verify_calls = 0

    def verify_unchanged(self) -> None:
        self.verify_calls += 1

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fake-formal-authorization-v1",
            "run_id": cli.RUN_ID,
            "output_claim_fingerprint": self.output_claim_fingerprint,
            "budget": {
                "seed": 42,
                "epochs": 800,
                "steps_per_epoch": 40,
                "updates": 32_000,
            },
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
        }


class _FakeTrainingResult:
    def __init__(self, epoch_logs: tuple[dict[str, object], ...]) -> None:
        self.seed = 42
        self.epochs = 800
        self.steps_per_epoch = 40
        self.completed_updates = 32_000
        self.forward_calls = 32_000
        self.backward_calls = 32_000
        self.optimizer_steps = 32_000
        self.logical_state_evaluations = 384_000
        self.finite_state_audits = 32_001
        self.epoch_logs = epoch_logs
        self.objective = cli.PACRE_PMOPE_OBJECTIVE
        self.execution_device = "cuda:0"
        self.result_fingerprint = "t" * 64

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fake-training-result-v1",
            "seed": self.seed,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "completed_updates": self.completed_updates,
            "epoch_logs": list(self.epoch_logs),
            "compute": {
                "forward_calls": self.forward_calls,
                "backward_calls": self.backward_calls,
                "optimizer_steps": self.optimizer_steps,
                "logical_state_evaluations": (
                    self.logical_state_evaluations
                ),
                "finite_state_audits": self.finite_state_audits,
            },
        }


class _FakeFormalResult:
    def __init__(
        self,
        epoch_logs: tuple[dict[str, object], ...],
        authorization: _FakeAuthorization,
    ) -> None:
        self.authorization = authorization
        self.training_complete = True
        self.training_invocations = 1
        self.training_result = _FakeTrainingResult(epoch_logs)
        self.final_model = object()
        self.result_fingerprint = "r" * 64
        self.verify_calls = 0

    def verify_unchanged(self) -> None:
        self.verify_calls += 1

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fake-formal-result-v1",
            "run_id": cli.RUN_ID,
            "training_complete": True,
            "training_invocations": 1,
            "training_result": (
                self.training_result.canonical_payload()
            ),
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
        }


def _install_success_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[str], dict[str, object]]:
    output = tmp_path / cli.RUN_ID
    bounded = tmp_path / "bounded-must-stay-absent"
    calls: list[str] = []
    captured: dict[str, object] = {}
    context = _static_context()
    real_inputs = _FakeRealInputs()
    config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )

    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "BOUNDED_OUTPUT_PATH", bounded)

    def validate_static(
        *,
        formal_output_may_exist: bool,
    ) -> cli._StaticContext:
        calls.append(
            "static_after_claim"
            if formal_output_may_exist
            else "static_before_claim"
        )
        assert output.exists() is formal_output_may_exist
        return context

    def build_inputs(**kwargs: object) -> _FakeRealInputs:
        calls.append("build_real_inputs")
        assert (output / cli.INCOMPLETE_FILE).is_file()
        assert (output / "attempt.json").is_file()
        captured["build_kwargs"] = kwargs
        return real_inputs

    def prepare(
        passed_inputs: object,
        passed_config: object,
        **kwargs: object,
    ) -> _FakeAuthorization:
        calls.append("prepare_authorization")
        assert passed_inputs is real_inputs
        assert passed_config is config
        claim = kwargs["output_claim_fingerprint"]
        attempt = json.loads(
            (output / "attempt.json").read_text(encoding="utf-8")
        )
        assert claim == attempt["receipt_fingerprint"]
        captured["output_claim_fingerprint"] = claim
        return _FakeAuthorization(str(claim))

    def train(
        authorization: object,
        passed_config: object,
        **kwargs: object,
    ) -> _FakeFormalResult:
        calls.append("train_once")
        assert isinstance(authorization, _FakeAuthorization)
        assert passed_config is config
        assert kwargs["device"] == "cuda:0"
        callback = kwargs["epoch_callback"]
        rows: list[dict[str, object]] = []
        for epoch in range(800):
            row = {
                "epoch": epoch,
                "objective": cli.PACRE_PMOPE_OBJECTIVE,
                "completed_updates": (epoch + 1) * 40,
                "mean_total_loss": 1.0 / (epoch + 1),
            }
            callback(row)
            rows.append(row)
        return _FakeFormalResult(tuple(rows), authorization)

    def save(
        directory: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append("save_final_only")
        formal_result = kwargs["formal_result"]
        assert isinstance(formal_result, _FakeFormalResult)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "model.safetensors").write_bytes(b"safe-tensors")
        artifact = fingerprinted(
            {
                "schema_version": "fake-formal-artifact-v1",
                "candidate": "PACRE-VC-v23",
                "authorization_fingerprint": (
                    formal_result.authorization
                    .authorization_fingerprint
                ),
                "source_closure_fingerprint": (
                    formal_result.authorization
                    .source_closure_fingerprint
                ),
                "final_checkpoint_only": True,
                "optimizer_state_saved": False,
                "intermediate_checkpoint_saved": False,
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
                "performance_evaluation_performed": False,
            },
            field="artifact_fingerprint",
        )
        write_new_json(directory / "artifact.json", artifact)
        return artifact

    monkeypatch.setattr(cli, "_validate_static", validate_static)
    monkeypatch.setattr(
        cli,
        "frozen_real_dr_source_paths",
        lambda: {},
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        build_inputs,
    )
    monkeypatch.setattr(
        cli,
        "expected_pacre_vc_formal_config",
        lambda inputs: config,
    )
    monkeypatch.setattr(
        cli,
        "prepare_pacre_vc_formal_800_authorization",
        prepare,
    )
    monkeypatch.setattr(
        cli,
        "run_pacre_vc_pmope_formal_800",
        train,
    )
    monkeypatch.setattr(
        cli,
        "save_pacre_vc_formal_final_model",
        save,
    )
    return output, calls, captured


def _json_artifacts(output: Path) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for path in output.rglob("*.json"):
        artifacts.append(json.loads(path.read_text(encoding="utf-8")))
    return artifacts


def _walk_key_values(value: object):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_key_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_values(child)


def test_frozen_contract_is_seed42_800x40_final_only():
    assert cli.RUN_ID == (
        "cure_lite_pacre_v23_vc_pmope_formal_800_seed42_r1"
    )
    assert cli.OUTPUT_REPO_PATH.endswith(cli.RUN_ID)
    assert cli.FROZEN_SEED == 42
    assert cli.FROZEN_EPOCHS == 800
    assert cli.FROZEN_STEPS_PER_EPOCH == 40
    assert cli.FROZEN_UPDATES == 32_000
    assert cli.FROZEN_DEVICE == "cuda:0"
    assert cli.FROZEN_VISIBLE_GPU == "0"
    assert cli.FROZEN_CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert cli.FROZEN_PYTHONHASHSEED == "0"
    assert cli.FROZEN_PAUSE_TEMPERATURE_C == 82
    assert cli.FROZEN_RESUME_TEMPERATURE_C == 75
    assert cli.EXPECTED_SCIENTIFIC_FILES == {
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


def test_visible_cuda_contract_requires_only_logical_cuda_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cli.torch.cuda, "current_device", lambda: 0)
    assert cli._visible_cuda_contract() == {
        "cuda_available": True,
        "visible_cuda_device_count": 1,
        "current_cuda_logical_device": 0,
    }

    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="exactly one visible"):
        cli._visible_cuda_contract()
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cli.torch.cuda, "current_device", lambda: 1)
    with pytest.raises(RuntimeError, match="logical device"):
        cli._visible_cuda_contract()
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available CUDA"):
        cli._visible_cuda_contract()


def test_dr_loader_verifies_the_gate_wrapper_custom_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    receipt = SimpleNamespace(
        receipt_fingerprint="d" * 64,
        checks=tuple(
            (name, True) for name in cli.PACRE_VC_DR_CHECK_NAMES
        ),
        gate_passed=True,
        decision=cli.PACRE_VC_DR_PASS_DECISION,
        canonical_payload=lambda: {
            "run_id": (
                "pacre_v23_verifier_corrected_D_R_structural_r1"
            )
        },
    )
    wrapper = fingerprinted(
        {
            "receipt": {"persisted": "receipt"},
            "receipt_fingerprint": receipt.receipt_fingerprint,
        },
        field="wrapper_fingerprint",
    )
    monkeypatch.setattr(cli, "read_strict_json", lambda path: wrapper)
    monkeypatch.setattr(
        cli,
        "pacre_vc_dr_receipt_from_payload",
        lambda payload: receipt,
    )
    verification = {
        "output": str(tmp_path / "D_R"),
        "run_id": (
            "pacre_v23_verifier_corrected_D_R_structural_r1"
        ),
        "decision": cli.PACRE_VC_DR_PASS_DECISION,
        "gate_passed": True,
        "failed_checks": [],
        "receipt_fingerprint": receipt.receipt_fingerprint,
        "wrapper_fingerprint": wrapper["wrapper_fingerprint"],
        "formal_800_route_granted": True,
        "bounded_400_required": False,
        "bounded_400_authorization_effect": False,
    }
    assert cli._load_dr_receipt(verification) is receipt

    corrupted = dict(wrapper)
    corrupted["wrapper_fingerprint"] = "0" * 64
    monkeypatch.setattr(
        cli,
        "read_strict_json",
        lambda path: corrupted,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        cli._load_dr_receipt(verification)


def test_validate_create_only_never_claims_or_builds_or_authorizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "formal"
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "BOUNDED_OUTPUT_PATH", tmp_path / "bounded")
    monkeypatch.setattr(
        cli,
        "_validate_static",
        lambda *, formal_output_may_exist: _static_context(),
    )

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("create-only entered a tensor/run path")

    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        forbidden,
    )
    monkeypatch.setattr(
        cli,
        "prepare_pacre_vc_formal_800_authorization",
        forbidden,
    )
    monkeypatch.setattr(
        cli,
        "run_pacre_vc_pmope_formal_800",
        forbidden,
    )
    result = cli.validate_create_only()
    assert result["D_R_gate_check_count"] == 13
    assert result["output_claimed"] is False
    assert result["real_inputs_constructed"] is False
    assert result["formal_authorization_created"] is False
    assert result["training_performed"] is False
    assert result["D_R_tensor_payload_accessed"] is False
    assert result["D_V_tensor_payload_accessed"] is False
    assert result["D_T_tensor_payload_accessed"] is False
    assert not output.exists()


def test_run_once_claims_before_inputs_and_seals_800_rows_32000_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, calls, captured = _install_success_path(
        tmp_path,
        monkeypatch,
    )

    complete = cli.run_once()

    assert calls[:3] == [
        "static_before_claim",
        "static_after_claim",
        "build_real_inputs",
    ]
    assert calls.count("train_once") == 1
    assert calls[-1] == "save_final_only"
    assert len(str(captured["output_claim_fingerprint"])) == 64
    assert not (output / cli.INCOMPLETE_FILE).exists()
    assert not (output / "FAILURE.json").exists()
    assert (output / "COMPLETE.json").is_file()
    assert complete["seed"] == 42
    assert complete["epochs"] == 800
    assert complete["steps_per_epoch"] == 40
    assert complete["updates"] == 32_000
    assert complete["training_invocations"] == 1
    assert complete["D_V_preregistration_eligible"] is True
    assert complete["D_V_execution_authorized"] is False
    assert complete["performance_claim_supported"] is False
    assert complete["artifact_count"] == 10

    progress_path = output / "receipts/epoch_progress.jsonl"
    rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 800
    assert rows[0]["epoch_result"]["epoch"] == 0
    assert rows[-1]["epoch_result"]["epoch"] == 799
    assert rows[-1]["epoch_result"]["completed_updates"] == 32_000
    assert all(
        row["receipt_fingerprint"]
        == stable_fingerprint(
            {
                key: value
                for key, value in row.items()
                if key != "receipt_fingerprint"
            }
        )
        for row in rows
    )

    training = json.loads(
        (output / "receipts/training.json").read_text(encoding="utf-8")
    )
    assert training["compute_ledger"] == {
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
    }
    live_files = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    }
    assert live_files == set(cli.EXPECTED_SCIENTIFIC_FILES) | {
        "COMPLETE.json"
    }


def test_run_once_never_accesses_d_v_or_d_t_or_claims_performance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, _, _ = _install_success_path(tmp_path, monkeypatch)
    cli.run_once()

    for artifact in _json_artifacts(output):
        for key, value in _walk_key_values(artifact):
            if key in {
                "D_V_accessed",
                "D_T_accessed",
                "D_V_tensor_payload_accessed",
                "D_T_tensor_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
                "performance_evaluation_performed",
                "performance_claim_supported",
            }:
                assert value is False
    decision = json.loads(
        (output / "receipts/decision.json").read_text(encoding="utf-8")
    )
    assert decision["D_V_preregistration_eligible"] is True
    assert decision["D_V_execution_authorized"] is False
    assert decision["D_T_execution_authorized"] is False
    assert decision["performance_gate_passed"] is None
    assert decision["final_model_performance_success_established"] is False


def test_d_r_failure_occurs_before_claim_or_input_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "formal"
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "BOUNDED_OUTPUT_PATH", tmp_path / "bounded")
    monkeypatch.setattr(
        cli,
        "_validate_static",
        lambda **kwargs: (_ for _ in ()).throw(
            PermissionError("D_R is 12/13")
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("input graph must not be built")
        ),
    )

    with pytest.raises(PermissionError, match="12/13"):
        cli.run_once()
    assert not output.exists()


def test_existing_formal_output_fails_closed_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "formal"
    output.mkdir()
    sentinel = output / "owned-by-earlier-attempt"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "BOUNDED_OUTPUT_PATH", tmp_path / "bounded")

    with pytest.raises(FileExistsError, match="already exists"):
        cli._ensure_attempt_paths_absent(
            formal_output_may_exist=False
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert set(output.iterdir()) == {sentinel}


def test_existing_bounded_output_has_no_authorization_effect_and_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bounded = tmp_path / "bounded"
    bounded.mkdir()
    monkeypatch.setattr(cli, "OUTPUT_PATH", tmp_path / "formal")
    monkeypatch.setattr(cli, "BOUNDED_OUTPUT_PATH", bounded)

    with pytest.raises(
        PermissionError,
        match="bounded-400 output must be absent",
    ):
        cli._ensure_attempt_paths_absent(
            formal_output_may_exist=False
        )
    assert not cli.OUTPUT_PATH.exists()


def test_training_exception_writes_failure_and_keeps_incomplete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, _, _ = _install_success_path(tmp_path, monkeypatch)

    def fail_training(
        authorization: object,
        config: object,
        **kwargs: object,
    ):
        callback = kwargs["epoch_callback"]
        callback(
            {
                "epoch": 0,
                "objective": cli.PACRE_PMOPE_OBJECTIVE,
                "completed_updates": 40,
            }
        )
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(
        cli,
        "run_pacre_vc_pmope_formal_800",
        fail_training,
    )

    with pytest.raises(RuntimeError, match="synthetic optimizer"):
        cli.run_once()
    assert (output / cli.INCOMPLETE_FILE).is_file()
    assert (output / "FAILURE.json").is_file()
    assert not (output / "COMPLETE.json").exists()
    failure = json.loads(
        (output / "FAILURE.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed_incomplete_single_attempt"
    assert failure["failed_stage"] == "formal_800_training"
    assert failure["real_inputs_constructed"] is True
    assert failure["training_invocations"] == 1
    assert failure["output_directory_reusable"] is False
    assert failure["resume_allowed"] is False
    assert failure["automatic_retry_allowed"] is False
    assert failure["D_V_tensor_payload_accessed"] is False
    assert failure["D_T_tensor_payload_accessed"] is False
    assert failure["performance_claim_supported"] is False
