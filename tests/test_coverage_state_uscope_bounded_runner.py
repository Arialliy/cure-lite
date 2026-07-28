from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import cure_lite.experiment.coverage_state_uscope_bounded_runner as runner
from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_uscope_dataset_free import (
    run_coverage_state_uscope_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_uscope_dr_gate import (
    CoverageStateUSCOPEDRGateReceipt,
)
from cure_lite.frozen_base import module_state_fingerprint
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)


@pytest.fixture(scope="module")
def preflight():
    population = build_coverage_state_bounded_population(
        make_bounded_training_scalar_cache()
    )
    return prepare_coverage_state_bounded_preflight(population)


@pytest.fixture(scope="module")
def dataset_free():
    return run_coverage_state_uscope_dataset_free_gate()


@pytest.fixture(scope="module")
def sealed_v18():
    return runner.verify_current_sealed_v18_negative()


class _FakeDRGate(CoverageStateUSCOPEDRGateReceipt):
    def __init__(
        self,
        *,
        dataset_free_receipt,
        population,
        model_config: dict[str, object],
        initial_model_fingerprint: str,
        passed: bool,
    ) -> None:
        object.__setattr__(
            self,
            "dataset_free_receipt",
            dataset_free_receipt,
        )
        object.__setattr__(
            self,
            "real_inputs",
            SimpleNamespace(
                build_fingerprint="1" * 64,
                source_binding=SimpleNamespace(
                    binding_fingerprint="2" * 64,
                ),
            ),
        )
        object.__setattr__(self, "population", population)
        object.__setattr__(
            self,
            "implementation_binding",
            (("fake-uscope-dr.py", "3" * 64),),
        )
        object.__setattr__(
            self,
            "geometry",
            {"fixed_real_D_R_geometry": True},
        )
        object.__setattr__(
            self,
            "probe",
            {
                "execution_seed": 42,
                "runtime_splits": ["D_R"],
                "model_config": model_config,
                "initial_model_fingerprint": (
                    initial_model_fingerprint
                ),
            },
        )
        object.__setattr__(
            self,
            "checks",
            (("fixed_real_D_R_gate", passed),),
        )
        object.__setattr__(self, "evidence_fingerprint", "4" * 64)
        object.__setattr__(self, "verify_calls", 0)

    @property
    def all_pass(self) -> bool:
        return all(value for _, value in self.checks)

    def verify_unchanged(self) -> None:
        object.__setattr__(
            self,
            "verify_calls",
            self.verify_calls + 1,
        )

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": (
                runner.COVERAGE_STATE_USCOPE_DR_GATE_SCHEMA
            ),
            "evidence_fingerprint": self.evidence_fingerprint,
            "checks": dict(self.checks),
            "all_pass": self.all_pass,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _patch_toy_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
) -> None:
    monkeypatch.setattr(
        runner,
        "COVERAGE_STATE_USCOPE_HISTORICAL_SCHEDULE_FINGERPRINT",
        preflight.schedule.schedule_fingerprint,
    )
    monkeypatch.setattr(
        runner,
        "COVERAGE_STATE_USCOPE_HISTORICAL_CACHE_FINGERPRINT",
        preflight.population.bounded_cache_fingerprint,
    )


def _fake_dr_gate(preflight, dataset_free, *, passed: bool = True):
    config = runner.expected_coverage_state_uscope_config(preflight)
    _, raw_initial = runner._common_initial_model_fingerprints(config)
    return _FakeDRGate(
        dataset_free_receipt=dataset_free,
        population=preflight.population,
        model_config=runner._dr_model_config_payload(config),
        initial_model_fingerprint=raw_initial,
        passed=passed,
    )


def _authorization(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free,
    sealed_v18,
):
    _patch_toy_coordinates(monkeypatch, preflight)
    monkeypatch.setattr(
        runner,
        "verify_current_sealed_v18_negative",
        lambda: sealed_v18,
    )
    return runner.prepare_coverage_state_uscope_bounded_run_authorization(
        preflight,
        dataset_free,
        _fake_dr_gate(preflight, dataset_free),
    )


def test_authorization_binds_single_uscope_candidate_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free,
    sealed_v18,
) -> None:
    _patch_toy_coordinates(monkeypatch, preflight)
    sealed_calls = 0

    def sealed_once():
        nonlocal sealed_calls
        sealed_calls += 1
        return sealed_v18

    monkeypatch.setattr(
        runner,
        "verify_current_sealed_v18_negative",
        sealed_once,
    )
    dr_gate = _fake_dr_gate(preflight, dataset_free)
    authorization = (
        runner.prepare_coverage_state_uscope_bounded_run_authorization(
            preflight,
            dataset_free,
            dr_gate,
        )
    )
    calls_after_prepare = dr_gate.verify_calls
    authorization.verify_unchanged()
    authorization.verify_for_run(
        cache=preflight.population.cache,
        schedule=preflight.schedule,
        scope=runner.COVERAGE_STATE_BOUNDED_SCOPE,
    )

    assert sealed_calls == 1
    assert dr_gate.verify_calls == calls_after_prepare
    assert authorization.training_authorized
    assert authorization.objective_suite == ("uscope_joint",)
    assert authorization.candidate_objective_policy == (
        runner.CSLF_USCOPE_POLICY
    )
    payload = authorization.canonical_payload()
    assert payload["budget"]["seed"] == 42
    assert payload["budget"]["updates"] == 400
    assert payload["formal_800_authorized"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False


def test_authorization_rejects_failed_dr_gate(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free,
    sealed_v18,
) -> None:
    _patch_toy_coordinates(monkeypatch, preflight)
    monkeypatch.setattr(
        runner,
        "verify_current_sealed_v18_negative",
        lambda: sealed_v18,
    )
    with pytest.raises(PermissionError, match="D_R gate"):
        runner.prepare_coverage_state_uscope_bounded_run_authorization(
            preflight,
            dataset_free,
            _fake_dr_gate(
                preflight,
                dataset_free,
                passed=False,
            ),
        )


def test_provided_sealed_receipt_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free,
    sealed_v18,
) -> None:
    _patch_toy_coordinates(monkeypatch, preflight)

    def forbidden_replay():
        raise AssertionError("sealed v18 verifier was replayed")

    monkeypatch.setattr(
        runner,
        "verify_current_sealed_v18_negative",
        forbidden_replay,
    )
    authorization = (
        runner.prepare_coverage_state_uscope_bounded_run_authorization(
            preflight,
            dataset_free,
            _fake_dr_gate(preflight, dataset_free),
            sealed_v18_receipt=sealed_v18,
        )
    )
    authorization.verify_unchanged()
    assert authorization.training_authorized


class _FakeTraining:
    def __init__(self, authorization, model) -> None:
        row = SimpleNamespace(
            objective="uscope_joint",
            objective_policy=runner.CSLF_USCOPE_POLICY,
            seed=42,
            epochs=10,
            steps_per_epoch=40,
            completed_updates=400,
            forward_calls=400,
            backward_calls=400,
            optimizer_steps=400,
            logical_state_evaluations=4800,
            finite_state_audits=401,
            execution_device="cuda:0",
            initial_model_fingerprint=(
                runner
                .COVERAGE_STATE_USCOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            ),
            final_model_fingerprint=runner.coverage_state_model_fingerprint(
                model
            ),
            schedule_fingerprint=(
                authorization.preflight.schedule.schedule_fingerprint
            ),
            cache_fingerprint=(
                authorization
                .preflight.population.bounded_cache_fingerprint
            ),
            optimizer_config_fingerprint=(
                runner
                .COVERAGE_STATE_USCOPE_HISTORICAL_OPTIMIZER_FINGERPRINT
            ),
            device_cache_fingerprint=(
                runner
                .COVERAGE_STATE_USCOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT
            ),
            first_nonzero_gradient_update=(
                ("joint_hidden_bias", 1),
                ("joint_state_weight", 1),
                ("scalar_energy_weight", 0),
            ),
        )
        self.results = (row,)
        self.models = (("uscope_joint", model),)
        self.common_initial_model_fingerprint = (
            row.initial_model_fingerprint
        )
        self.schedule_fingerprint = row.schedule_fingerprint
        self.cache_fingerprint = row.cache_fingerprint

    def verify_unchanged(self) -> None:
        return None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "objective_suite": ["uscope_joint"],
            "fairness": {
                "allowed_difference_from_sealed_v18": (
                    "predeclared_pair_objective_only"
                )
            },
        }


class _FakeCertificate:
    def __init__(self, authorization, model) -> None:
        self.cache_fingerprint = (
            authorization.preflight.population.bounded_cache_fingerprint
        )
        self.model_fingerprint_before = (
            runner._certificate_model_fingerprint(model)
        )
        self.model_fingerprint_after = self.model_fingerprint_before
        self.pair_batch_size = 32
        self.model_forward_invocations = 1
        self.pair_certificates = tuple(
            SimpleNamespace(
                gamma_strictly_below_margin=True,
                raw_sign_error_pixels=0,
                pair_certificate_passed=True,
            )
            for _ in range(32)
        )
        self.all_pass = True
        self.same_sign_response_evaluated = False
        self.same_sign_response_is_gate = False
        self.optimizer_constructed = False
        self.backward_performed = False
        self.training_performed = False

    def verify(self) -> None:
        return None

    def canonical_payload(self) -> dict[str, object]:
        return {"all_pass": self.all_pass}


class _FakeDiagnostic:
    def __init__(self, authorization, model) -> None:
        self.checkpoint_fingerprint = module_state_fingerprint(model)
        self.cache_fingerprint = (
            authorization.preflight.population.bounded_cache_fingerprint
        )
        self.split = "D_R"
        self.backward_calls = 0
        self.optimizer_steps = 0
        self.config = SimpleNamespace(
            d_v_accessed=False,
            d_t_accessed=False,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {"split": self.split}


class _FakeDecision:
    def __init__(self, diagnostic) -> None:
        self.diagnostic = diagnostic
        self.zero_level_gate_passed = True
        self.response_sign_correct_pixels = 0
        self.response_sign_pixels = 999

    def canonical_payload(self) -> dict[str, object]:
        return {
            "same_sign_response_policy": (
                "legacy_all_response_ordering_is_"
                "diagnostic_not_binary_gate"
            ),
            "zero_level_gate_passed": self.zero_level_gate_passed,
            "response_diagnostic": {
                "correct": self.response_sign_correct_pixels,
                "total": self.response_sign_pixels,
            },
        }


def _fake_candidate(authorization):
    config = runner.expected_coverage_state_uscope_config(
        authorization.preflight
    )
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    with torch.no_grad():
        model.scalar_energy_weight.fill_(0.125)
    training = _FakeTraining(authorization, model)
    certificate = _FakeCertificate(authorization, model)
    diagnostic = _FakeDiagnostic(authorization, model)
    decision = _FakeDecision(diagnostic)
    return model, training, certificate, diagnostic, decision


def test_result_gates_certificate_but_not_response_accuracy(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free,
    sealed_v18,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        dataset_free,
        sealed_v18,
    )
    _, training, certificate, diagnostic, decision = _fake_candidate(
        authorization
    )
    checks = runner._uscope_bounded_result_checks(
        authorization,
        training,
        certificate,
        diagnostic,
        decision,
        training_invocations=1,
        certificate_invocations=1,
        zero_level_evaluation_invocations=1,
    )
    assert all(dict(checks).values())
    assert decision.response_sign_correct_pixels == 0

    certificate.pair_certificates[0].raw_sign_error_pixels = 1
    failed = dict(
        runner._uscope_bounded_result_checks(
            authorization,
            training,
            certificate,
            diagnostic,
            decision,
            training_invocations=1,
            certificate_invocations=1,
            zero_level_evaluation_invocations=1,
        )
    )
    assert failed["every_pair_uniform_certificate"] is False


def test_runner_calls_training_certificate_then_zero_level_once(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free,
    sealed_v18,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        dataset_free,
        sealed_v18,
    )
    model, training, certificate, diagnostic, decision = _fake_candidate(
        authorization
    )
    order: list[str] = []

    def train(*args, **kwargs):
        order.append("training")
        return training

    def certify(*args, **kwargs):
        order.append("certificate")
        assert kwargs["pair_batch_size"] == 32
        return certificate

    def evaluate(*args, **kwargs):
        order.append("zero_level")
        return diagnostic

    def decide(actual):
        order.append("decision")
        assert actual is diagnostic
        return decision

    monkeypatch.setattr(
        runner,
        "train_matched_coverage_state_cmif_uscope_objectives",
        train,
    )
    monkeypatch.setattr(
        runner,
        "audit_coverage_state_uscope_pair_certificate",
        certify,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_coverage_state_zero_level_checkpoint",
        evaluate,
    )
    monkeypatch.setattr(
        runner,
        "decide_coverage_state_uscope_zero_level",
        decide,
    )
    monkeypatch.setattr(
        runner,
        "_deterministic_execution",
        lambda device: nullcontext(),
    )

    with pytest.raises(PermissionError, match="cuda:0"):
        runner.run_coverage_state_cmif_uscope_bounded_400(
            authorization,
            model.config,
            device="cpu",
        )
    assert order == []
    result = runner.run_coverage_state_cmif_uscope_bounded_400(
        authorization,
        model.config,
        device="cuda:0",
    )
    assert order == [
        "training",
        "certificate",
        "zero_level",
        "decision",
    ]
    assert result.bounded_gate_passed
    assert result.training is training
    assert result.certificate is certificate
    assert result.diagnostic is diagnostic
    assert result.decision is decision
