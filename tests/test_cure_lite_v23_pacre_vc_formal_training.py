from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest
import torch

import cure_lite_v23.formal_training as formal
from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.coverage_state_schedule import CoverageStateScheduleConfig
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    COVERAGE_STATE_FORMAL_SCOPE,
    CoverageStateRunAuthorization,
    CoverageStateTrainingResult,
    _verify_run_authorization,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite.train.coverage_state_fused_step import (
    coverage_state_pair_objective_policy,
)
from cure_lite_v23.pacre_vc import (
    PACRE_VC_CANDIDATE,
    CoverageStatePACREVerifierCorrectedConfig,
)


class _FakeCache:
    def __init__(self, fingerprint: str) -> None:
        self.cache_fingerprint = fingerprint
        self.raw_catalog = SimpleNamespace(
            dataset="IRSTD-1K",
            split="D_R",
            feature_stride=2,
            natural_records=(
                SimpleNamespace(feature=torch.zeros(1, 2, 2, 2)),
            ),
        )
        self.sobolev_config = SimpleNamespace(truncation_radius=2)

    def verify_unchanged(self) -> None:
        if self.raw_catalog.split != "D_R":
            raise RuntimeError("fake cache split changed")


class _FakeRealInputs:
    def __init__(self, cache: _FakeCache) -> None:
        self.scalar_cache = cache
        self.scalar_cache_fingerprint = cache.cache_fingerprint
        self.source_binding = SimpleNamespace(
            dataset="IRSTD-1K",
            split="D_R",
            binding_fingerprint=stable_fingerprint(
                {"fake_source_binding": cache.cache_fingerprint}
            ),
        )
        self.build_fingerprint = stable_fingerprint(
            {
                "fake_real_inputs": cache.cache_fingerprint,
                "source": self.source_binding.binding_fingerprint,
            }
        )
        self.verify_calls = 0

    def verify_unchanged(self) -> None:
        self.verify_calls += 1
        self.scalar_cache.verify_unchanged()


class _FakeSchedule:
    def __init__(
        self,
        cache: _FakeCache,
        config: CoverageStateScheduleConfig,
        fingerprint: str,
    ) -> None:
        self.cache_fingerprint = cache.cache_fingerprint
        self.config = config
        self.schedule_fingerprint = fingerprint


class _FakeDRReceipt:
    def __init__(
        self,
        *,
        dataset_fingerprint: str,
        closure_fingerprint: str,
        real_inputs_fingerprint: str,
        passed: bool = True,
    ) -> None:
        self.dataset_free_receipt_fingerprint = dataset_fingerprint
        self.source_closure_fingerprint = closure_fingerprint
        self.real_inputs_fingerprint = real_inputs_fingerprint
        self.cache_fingerprint = "b" * 64
        self.checks = tuple(
            (name, passed) for name in formal.PACRE_VC_DR_CHECK_NAMES
        )
        self.probe = {
            "D_R_accessed": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "parameter_updates": 0,
        }
        self.verify_source_calls = 0

    @property
    def gate_passed(self) -> bool:
        return all(value for _, value in self.checks)

    @property
    def decision(self) -> str:
        return (
            formal.PACRE_VC_DR_PASS_DECISION
            if self.gate_passed
            else "PACRE_V23_D_R_STRUCTURAL_FAIL"
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fake-pacre-v23-dr-receipt-v1",
            "candidate": PACRE_VC_CANDIDATE,
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "source_closure_fingerprint": (
                self.source_closure_fingerprint
            ),
            "real_inputs_fingerprint": self.real_inputs_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "probe": dict(self.probe),
            "checks": dict(self.checks),
            "failed_checks": [
                name for name, passed in self.checks if not passed
            ],
            "gate_passed": self.gate_passed,
            "decision": self.decision,
            "identifiability_only": True,
            "performance_claim_supported": False,
            "training_performed": False,
            "D_R_accessed": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "formal_800_authorized": False,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_sources_unchanged(self) -> None:
        self.verify_source_calls += 1


@dataclass
class _Environment:
    cache: _FakeCache
    real_inputs: _FakeRealInputs
    config: CoverageStatePACREVerifierCorrectedConfig
    dataset_receipt: dict[str, object]
    source_closure: dict[str, object]
    dr_receipt: _FakeDRReceipt
    output_claim_fingerprint: str
    schedule_fingerprint: str
    exposure_fingerprint: str
    build_calls: list[tuple[object, CoverageStateScheduleConfig]]
    exposure_calls: list[tuple[object, object]]

    def prepare(
        self,
        *,
        receipt: _FakeDRReceipt | None = None,
        model_config: CoverageStatePACREVerifierCorrectedConfig | None = None,
        output_claim_fingerprint: str | None = None,
    ) -> formal.CoverageStatePACREVCFormal800Authorization:
        return formal.prepare_pacre_vc_formal_800_authorization(
            self.real_inputs,
            self.config if model_config is None else model_config,
            dataset_free_receipt=self.dataset_receipt,
            dr_gate_receipt=self.dr_receipt if receipt is None else receipt,
            source_closure=self.source_closure,
            output_claim_fingerprint=(
                self.output_claim_fingerprint
                if output_claim_fingerprint is None
                else output_claim_fingerprint
            ),
        )


@pytest.fixture(autouse=True)
def _clear_formal_attempt_registry():
    with formal._FORMAL_ATTEMPT_REGISTRY_LOCK:
        formal._FORMAL_ATTEMPT_REGISTRY.clear()
    yield
    with formal._FORMAL_ATTEMPT_REGISTRY_LOCK:
        formal._FORMAL_ATTEMPT_REGISTRY.clear()


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> _Environment:
    cache_fingerprint = stable_fingerprint({"fake_full_D_R_cache": 1})
    schedule_fingerprint = stable_fingerprint(
        {"fake_seed42_formal_schedule": cache_fingerprint}
    )
    cache = _FakeCache(cache_fingerprint)
    real_inputs = _FakeRealInputs(cache)
    config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )

    dataset_body: dict[str, object] = {
        "schema_version": "fake-pacre-v23-dataset-free-v1",
        "candidate": PACRE_VC_CANDIDATE,
        "parameter_count": config.expected_parameter_count,
        "gate_passed": True,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    dataset_receipt = {
        **dataset_body,
        "receipt_fingerprint": stable_fingerprint(dataset_body),
    }

    closure_body: dict[str, object] = {
        "schema_version": "fake-pacre-v23-source-closure-v1",
        "file_count": 1,
        "binding_fingerprint": "c" * 64,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    source_closure = {
        **closure_body,
        "closure_fingerprint": stable_fingerprint(closure_body),
    }
    dr_receipt = _FakeDRReceipt(
        dataset_fingerprint=str(dataset_receipt["receipt_fingerprint"]),
        closure_fingerprint=str(source_closure["closure_fingerprint"]),
        real_inputs_fingerprint=real_inputs.build_fingerprint,
    )
    output_claim_fingerprint = stable_fingerprint(
        {"exclusive_formal_attempt_json": 1}
    )

    monkeypatch.setattr(formal, "CoverageStateScalarCache", _FakeCache)
    monkeypatch.setattr(formal, "CoverageStateRealDRInputs", _FakeRealInputs)
    monkeypatch.setattr(
        formal,
        "CoverageStateTrainingSchedule",
        _FakeSchedule,
    )
    monkeypatch.setattr(
        formal,
        "CoverageStatePACREDRGateReceipt",
        _FakeDRReceipt,
    )
    monkeypatch.setattr(formal, "PACRE_FORMAL_FEATURE_CHANNELS", 2)
    monkeypatch.setattr(formal, "PACRE_FORMAL_FEATURE_STRIDE", 2)
    monkeypatch.setattr(formal, "PACRE_FORMAL_WIDTH", 4)
    monkeypatch.setattr(
        formal,
        "PACRE_FORMAL_PARAMETER_COUNT",
        config.expected_parameter_count,
    )
    monkeypatch.setattr(
        formal,
        "PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT",
        cache_fingerprint,
    )
    monkeypatch.setattr(
        formal,
        "PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT",
        schedule_fingerprint,
    )
    monkeypatch.setattr(formal, "PACRE_VC_FORMAL_DEVICE", "cpu")

    def validate_dataset(payload):
        body = dict(payload)
        fingerprint = body.pop("receipt_fingerprint", None)
        if (
            fingerprint != stable_fingerprint(body)
            or body.get("candidate") != PACRE_VC_CANDIDATE
            or body.get("gate_passed") is not True
            or body.get("D_R_accessed") is not False
            or body.get("D_V_accessed") is not False
            or body.get("D_T_accessed") is not False
            or body.get("training_performed") is not False
        ):
            raise PermissionError("fake dataset receipt invalid")
        return fingerprint

    def verify_closure(payload):
        if dict(payload) != source_closure:
            raise RuntimeError("fake source closure changed")
        return source_closure["closure_fingerprint"]

    build_calls: list[tuple[object, CoverageStateScheduleConfig]] = []

    def build_schedule(actual_cache, schedule_config):
        build_calls.append((actual_cache, schedule_config))
        return _FakeSchedule(
            actual_cache,
            schedule_config,
            schedule_fingerprint,
        )

    exposure_body: dict[str, object] = {
        "schema_version": "fake-formal-exposure-gate-v1",
        "cache_fingerprint": cache_fingerprint,
        "schedule_fingerprint": schedule_fingerprint,
        "checks": {
            "selection_exact_budget": True,
            "full_support_exposed": True,
        },
        "failed_checks": [],
        "all_pass": True,
        "formal_training_authorized": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    exposure_fingerprint = stable_fingerprint(exposure_body)
    exposure = {
        **exposure_body,
        "gate_fingerprint": exposure_fingerprint,
    }
    exposure_calls: list[tuple[object, object]] = []

    def exposure_gate(actual_cache, actual_schedule):
        exposure_calls.append((actual_cache, actual_schedule))
        return json.loads(canonical_json(exposure))

    monkeypatch.setattr(
        formal,
        "_validate_dataset_free_receipt",
        validate_dataset,
    )
    monkeypatch.setattr(formal, "verify_source_closure", verify_closure)
    monkeypatch.setattr(
        formal,
        "build_coverage_state_training_schedule",
        build_schedule,
    )
    monkeypatch.setattr(
        formal,
        "coverage_state_formal_exposure_gate",
        exposure_gate,
    )
    monkeypatch.setattr(
        formal,
        "PACRE_VC_FORMAL_EXPOSURE_GATE_FINGERPRINT",
        exposure_fingerprint,
    )
    initial_fingerprint = formal._formal_model_binding(config)[2]
    monkeypatch.setattr(
        formal,
        "PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT",
        initial_fingerprint,
    )
    monkeypatch.setattr(
        formal,
        "_deterministic_execution",
        lambda _device: nullcontext(),
    )
    return _Environment(
        cache=cache,
        real_inputs=real_inputs,
        config=config,
        dataset_receipt=dataset_receipt,
        source_closure=source_closure,
        dr_receipt=dr_receipt,
        output_claim_fingerprint=output_claim_fingerprint,
        schedule_fingerprint=schedule_fingerprint,
        exposure_fingerprint=exposure_fingerprint,
        build_calls=build_calls,
        exposure_calls=exposure_calls,
    )


def _completed_fake_training(
    model,
    optimizer,
    cache,
    schedule,
    **kwargs,
) -> CoverageStateTrainingResult:
    authorization = kwargs["authorization"]
    assert isinstance(authorization, CoverageStateRunAuthorization)
    assert schedule.config == CoverageStateScheduleConfig.formal(seed=42)
    assert schedule.config.updates == 32_000
    assert cache is authorization.real_inputs.scalar_cache
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.state == {}
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert optimizer.param_groups[0]["eps"] == 1.0e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    # Exercise the real generic dispatcher: it must derive the Formal800 scope
    # from 800 x 40 and invoke this exact authorization subclass.
    _verify_run_authorization(authorization, cache, schedule)
    initial = coverage_state_model_fingerprint(model)
    assert initial == kwargs["expected_initial_model_fingerprint"]
    optimizer_fingerprint = coverage_state_optimizer_config_fingerprint(
        model,
        optimizer,
    )
    with torch.no_grad():
        model.scalar_energy_weight.fill_(0.125)
    final = coverage_state_model_fingerprint(model)
    objective = formal.PACRE_PMOPE_OBJECTIVE
    return CoverageStateTrainingResult(
        objective=objective,
        objective_policy=coverage_state_pair_objective_policy(objective),
        seed=42,
        epochs=800,
        steps_per_epoch=40,
        completed_updates=32_000,
        schedule_fingerprint=schedule.schedule_fingerprint,
        cache_fingerprint=cache.cache_fingerprint,
        execution_device="cpu",
        device_cache_fingerprint="d" * 64,
        device_cache_resident_bytes=1,
        optimizer_config_fingerprint=optimizer_fingerprint,
        initial_model_fingerprint=initial,
        final_model_fingerprint=final,
        epoch_logs=tuple(
            {
                "epoch": epoch,
                "completed_updates": (epoch + 1) * 40,
                "objective": objective,
            }
            for epoch in range(800)
        ),
        first_nonzero_gradient_update=(
            ("joint_hidden_bias", 1),
            ("joint_state_weight", 1),
            ("scalar_energy_weight", 0),
        ),
        forward_calls=32_000,
        backward_calls=32_000,
        optimizer_steps=32_000,
        logical_state_evaluations=384_000,
        finite_state_audits=32_001,
    )


def test_prepare_binds_direct_dr_pass_full_cache_and_formal_schedule(
    environment: _Environment,
) -> None:
    authorization = environment.prepare()
    payload = authorization.canonical_payload()

    assert type(authorization) is (
        formal.CoverageStatePACREVCFormal800Authorization
    )
    assert isinstance(authorization, CoverageStateRunAuthorization)
    assert authorization.available
    assert len(environment.build_calls) >= 1
    assert all(
        cache is environment.cache
        and config == CoverageStateScheduleConfig.formal(seed=42)
        for cache, config in environment.build_calls
    )
    assert authorization.schedule.config.updates == 32_000
    assert payload["D_R_gate_check_count"] == 13
    assert payload["output_claim_fingerprint"] == (
        environment.output_claim_fingerprint
    )
    assert all(payload["D_R_gate_checks"].values())
    assert payload["budget"] == {
        "seed": 42,
        "epochs": 800,
        "steps_per_epoch": 40,
        "updates": 32_000,
        "objectives": 1,
    }
    assert payload["authorization_policy"] == {
        "directly_authorized_by_v23_D_R_13_of_13_PASS": True,
        "bounded_400_required": False,
        "bounded_400_receipt_consumed": False,
        "bounded_400_is_final_success": False,
    }
    assert payload["formal_D_R_training_authorized"] is True
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["performance_evaluation_performed"] is False


def test_runner_reaches_real_800_scope_once_via_generic_trainer(
    environment: _Environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = environment.prepare()
    calls: list[dict[str, object]] = []

    def trainer(model, optimizer, cache, schedule, **kwargs):
        calls.append(
            {
                "model": model,
                "optimizer": optimizer,
                "cache": cache,
                "schedule": schedule,
                **kwargs,
            }
        )
        return _completed_fake_training(
            model,
            optimizer,
            cache,
            schedule,
            **kwargs,
        )

    monkeypatch.setattr(formal, "train_coverage_state_objective", trainer)
    result = formal.run_pacre_vc_pmope_formal_800(
        authorization,
        environment.config,
        device="cpu",
    )

    assert len(calls) == 1
    assert calls[0]["cache"] is environment.cache
    assert calls[0]["schedule"] is authorization.schedule
    assert calls[0]["authorization"] is authorization
    assert calls[0]["objective"].value == "pmope_joint"
    assert calls[0]["expected_initial_model_fingerprint"] == (
        authorization.initial_model_fingerprint
    )
    assert type(result) is formal.CoverageStatePACREVCFormal800RunResult
    assert result.training_complete
    assert result.final_model is result.model
    assert result.training_result.completed_updates == 32_000
    assert result.training_invocations == 1
    assert result.authorization.attempt_execution_ledger == {
        "state": "consumed",
        "claim_count": 1,
        "consume_count": 1,
        "failure_count": 0,
        "training_binding_fingerprint": (
            result.authorization.attempt_execution_ledger[
                "training_binding_fingerprint"
            ]
        ),
    }
    result_payload = result.canonical_payload()
    assert result_payload["output_contract"]["single_final_model"] is True
    assert result_payload["output_contract"]["checkpoint_written"] is False
    assert result_payload["bounded_400_required"] is False
    assert result_payload["D_V_accessed"] is False
    assert result_payload["D_T_accessed"] is False
    assert result_payload["performance_evaluation_performed"] is False

    with pytest.raises(PermissionError, match="no longer available"):
        formal.run_pacre_vc_pmope_formal_800(
            authorization,
            environment.config,
            device="cpu",
        )


def test_dr_failure_and_binding_drift_reject_before_schedule_or_model(
    environment: _Environment,
) -> None:
    failed = _FakeDRReceipt(
        dataset_fingerprint=str(
            environment.dataset_receipt["receipt_fingerprint"]
        ),
        closure_fingerprint=str(
            environment.source_closure["closure_fingerprint"]
        ),
        real_inputs_fingerprint=environment.real_inputs.build_fingerprint,
        passed=False,
    )
    initial_build_calls = len(environment.build_calls)
    with pytest.raises(PermissionError, match="13/13"):
        environment.prepare(receipt=failed)
    assert len(environment.build_calls) == initial_build_calls

    wrong_source = _FakeDRReceipt(
        dataset_fingerprint=str(
            environment.dataset_receipt["receipt_fingerprint"]
        ),
        closure_fingerprint="e" * 64,
        real_inputs_fingerprint=environment.real_inputs.build_fingerprint,
    )
    with pytest.raises(PermissionError, match="different formal prerequisites"):
        environment.prepare(receipt=wrong_source)
    assert len(environment.build_calls) == initial_build_calls

    wrong_model = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=2,
        feature_stride=2,
        width=5,
    )
    with pytest.raises(PermissionError, match="model config"):
        environment.prepare(model_config=wrong_model)
    assert len(environment.build_calls) == initial_build_calls


def test_equivalent_preparations_share_one_process_local_attempt(
    environment: _Environment,
) -> None:
    first = environment.prepare()
    second = environment.prepare()
    assert first is not second
    assert first._attempt_token is second._attempt_token

    first.claim_for_training(
        model_config=first.model_config,
        cache=first.real_inputs.scalar_cache,
        schedule=first.schedule,
        scope=COVERAGE_STATE_FORMAL_SCOPE,
        device="cpu",
    )
    with pytest.raises(PermissionError, match="no longer available"):
        second.claim_for_training(
            model_config=second.model_config,
            cache=second.real_inputs.scalar_cache,
            schedule=second.schedule,
            scope=COVERAGE_STATE_FORMAL_SCOPE,
            device="cpu",
        )
    assert first.attempt_execution_ledger["claim_count"] == 1

    distinct_claim = environment.prepare(
        output_claim_fingerprint=stable_fingerprint(
            {"exclusive_formal_attempt_json": 2}
        )
    )
    assert distinct_claim._attempt_token is not first._attempt_token
    assert distinct_claim.attempt_fingerprint != first.attempt_fingerprint


def test_wrong_scope_or_schedule_cannot_claim_formal_attempt(
    environment: _Environment,
) -> None:
    authorization = environment.prepare()
    with pytest.raises(PermissionError, match="training binding"):
        authorization.claim_for_training(
            model_config=authorization.model_config,
            cache=authorization.real_inputs.scalar_cache,
            schedule=authorization.schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
            device="cpu",
        )
    assert authorization.available

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        environment.prepare(output_claim_fingerprint="A" * 64)

    equivalent = _FakeSchedule(
        environment.cache,
        CoverageStateScheduleConfig.formal(seed=42),
        environment.schedule_fingerprint,
    )
    with pytest.raises(PermissionError, match="training binding"):
        authorization.claim_for_training(
            model_config=authorization.model_config,
            cache=authorization.real_inputs.scalar_cache,
            schedule=equivalent,
            scope=COVERAGE_STATE_FORMAL_SCOPE,
            device="cpu",
        )
    assert authorization.available


def test_training_exception_is_terminal_and_not_retryable(
    environment: _Environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = environment.prepare()
    calls = 0

    def failing_trainer(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic trainer failure")

    monkeypatch.setattr(
        formal,
        "train_coverage_state_objective",
        failing_trainer,
    )
    with pytest.raises(RuntimeError, match="synthetic trainer failure"):
        formal.run_pacre_vc_pmope_formal_800(
            authorization,
            environment.config,
            device="cpu",
        )
    assert calls == 1
    assert authorization.attempt_execution_ledger["state"] == "failed"
    assert authorization.attempt_execution_ledger["claim_count"] == 1
    assert authorization.attempt_execution_ledger["consume_count"] == 1
    assert authorization.attempt_execution_ledger["failure_count"] == 1

    with pytest.raises(PermissionError, match="no longer available"):
        formal.run_pacre_vc_pmope_formal_800(
            authorization,
            environment.config,
            device="cpu",
        )
    assert calls == 1
