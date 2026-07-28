from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import cure_lite.experiment.coverage_state_paet_bounded_runner as runner
from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_paet_dataset_free import (
    CoverageStatePAETDatasetFreeReceipt,
)
from cure_lite.experiment.coverage_state_paet_dr_gate import (
    CoverageStatePAETDRGateReceipt,
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
def sealed_v20():
    return (
        runner.verify_repository_coverage_state_bfa_v20_reference()
    )


class _FakeDatasetFree(CoverageStatePAETDatasetFreeReceipt):
    def __init__(self, *, passed: bool = True) -> None:
        object.__setattr__(self, "_passed", passed)

    def verify_unchanged(self) -> None:
        return None

    @property
    def all_pass(self) -> bool:
        return self._passed

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fake-paet-dataset-free-v1",
            "all_pass": self.all_pass,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


class _FakeDRGate(CoverageStatePAETDRGateReceipt):
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
        object.__setattr__(self, "bounded_population", population)
        object.__setattr__(
            self,
            "implementation_binding",
            (("fake-paet-dr.py", "3" * 64),),
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

    @property
    def all_pass(self) -> bool:
        return all(value for _, value in self.checks)

    def verify_unchanged(self) -> None:
        return None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": runner.COVERAGE_STATE_PAET_DR_GATE_SCHEMA,
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
        "COVERAGE_STATE_PAET_HISTORICAL_SCHEDULE_FINGERPRINT",
        preflight.schedule.schedule_fingerprint,
    )
    monkeypatch.setattr(
        runner,
        "COVERAGE_STATE_PAET_HISTORICAL_CACHE_FINGERPRINT",
        preflight.population.bounded_cache_fingerprint,
    )


def _fake_dr_gate(preflight, dataset_free, *, passed: bool = True):
    config = runner.expected_coverage_state_paet_config(preflight)
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
    sealed_v20,
):
    _patch_toy_coordinates(monkeypatch, preflight)
    dataset_free = _FakeDatasetFree()
    return (
        runner.prepare_coverage_state_paet_bounded_run_authorization(
            preflight,
            dataset_free,
            _fake_dr_gate(preflight, dataset_free),
            run_id=runner.COVERAGE_STATE_PAET_OFFICIAL_RUN_ID,
            sealed_v20_reference=sealed_v20,
        )
    )


def test_sealed_v20_reference_is_read_only_and_has_no_fake_resource(
    sealed_v20,
) -> None:
    payload = sealed_v20.canonical_payload()
    assert payload["run_id"].endswith("_r2")
    assert payload["known_historical_internal_run_id_defect"][
        "internal_result_run_id"
    ].endswith("_r1")
    assert payload["observed"]["factual_target_negative_pixels"] == 310
    assert payload["observed"]["factual_target_pixels"] == 335
    resource = payload["resource_reference"]
    assert resource["measured_reference_available"] is False
    assert resource["working_memory_bytes"] is None
    assert resource["elapsed_ns"] is None
    assert resource["ns_per_update"] is None
    assert resource["ratio_claim_supported"] is False


def test_authorization_binds_run_id_paet_pmope_and_fixed_budget(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    sealed_v20,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        sealed_v20,
    )
    authorization.verify_unchanged()
    payload = authorization.canonical_payload()

    assert authorization.training_authorized
    assert authorization.run_id == (
        runner.COVERAGE_STATE_PAET_OFFICIAL_RUN_ID
    )
    assert authorization.objective_suite == ("pmope_joint",)
    assert payload["run_id"] == authorization.run_id
    assert payload["expected_parameter_count"] == 64064
    assert payload["budget"] == {
        "seed": 42,
        "epochs": 10,
        "steps_per_epoch": 40,
        "updates": 400,
        "objectives": 1,
    }
    assert payload["historical_comparison_coordinates"][
        "allowed_difference"
    ] == "predeclared_phase_aligned_evidence_transport_only"
    assert payload["formal_800_authorized"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["resume_allowed"] is False
    assert payload["automatic_retry_allowed"] is False


def test_authorization_rejects_failed_dr_gate_or_wrong_run_id(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    sealed_v20,
) -> None:
    _patch_toy_coordinates(monkeypatch, preflight)
    dataset_free = _FakeDatasetFree()
    with pytest.raises(PermissionError, match="D_R gate"):
        runner.prepare_coverage_state_paet_bounded_run_authorization(
            preflight,
            dataset_free,
            _fake_dr_gate(
                preflight,
                dataset_free,
                passed=False,
            ),
            run_id=runner.COVERAGE_STATE_PAET_OFFICIAL_RUN_ID,
            sealed_v20_reference=sealed_v20,
        )
    with pytest.raises(PermissionError, match="run_id"):
        runner.prepare_coverage_state_paet_bounded_run_authorization(
            preflight,
            dataset_free,
            _fake_dr_gate(preflight, dataset_free),
            run_id="cure_lite_paet_bfa_v21_wrong",
            sealed_v20_reference=sealed_v20,
        )


def test_resource_measurement_records_real_fields_without_fake_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    clock_values = iter((1_000, 5_000))

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: events.append(f"sync:{device}"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: 100,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: 200,
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: events.append(f"reset:{device}"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: 600,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda device: 900,
    )

    def operation() -> str:
        events.append("training")
        return "trained"

    value, receipt = runner._measure_training_invocation(
        operation,
        device=torch.device("cuda:0"),
        clock=lambda: next(clock_values),
    )

    assert value == "trained"
    assert events == [
        "sync:cuda:0",
        "reset:cuda:0",
        "training",
        "sync:cuda:0",
    ]
    assert receipt.baseline_allocated_bytes == 100
    assert receipt.baseline_reserved_bytes == 200
    assert receipt.peak_allocated_bytes == 600
    assert receipt.peak_reserved_bytes == 900
    assert receipt.incremental_peak_allocated_bytes == 500
    assert receipt.incremental_peak_reserved_bytes == 700
    assert receipt.elapsed_ns == 4_000
    payload = receipt.canonical_payload()
    assert payload["updates"] == 400
    assert payload["parameter_count"] == 64064
    assert payload["ns_per_update"] == {
        "numerator": 4_000,
        "denominator": 400,
    }
    comparison = payload["v20_comparison"]
    assert comparison["status"] == (
        "NOT_EVALUATED_NO_MATCHED_V20_MEASUREMENT"
    )
    assert comparison["working_memory_ratio"] is None
    assert comparison["step_time_ratio"] is None
    assert comparison["working_memory_gate_passed"] is None
    assert comparison["step_time_gate_passed"] is None
    assert comparison["not_a_scientific_gate"] is True


class _FakeTraining:
    def __init__(self, authorization, model) -> None:
        row = SimpleNamespace(
            objective="pmope_joint",
            objective_policy=runner.CSLF_PMOPE_POLICY,
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
                .COVERAGE_STATE_PAET_HISTORICAL_INITIAL_MODEL_FINGERPRINT
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
                .COVERAGE_STATE_PAET_HISTORICAL_OPTIMIZER_FINGERPRINT
            ),
            device_cache_fingerprint=(
                runner
                .COVERAGE_STATE_PAET_HISTORICAL_DEVICE_CACHE_FINGERPRINT
            ),
            first_nonzero_gradient_update=(
                ("joint_hidden_bias", 1),
                ("joint_state_weight", 1),
                ("scalar_energy_weight", 0),
            ),
        )
        self.results = (row,)
        self.models = (("pmope_joint", model),)
        self.common_initial_model_fingerprint = row.initial_model_fingerprint
        self.schedule_fingerprint = row.schedule_fingerprint
        self.cache_fingerprint = row.cache_fingerprint

    def verify_unchanged(self) -> None:
        return None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "objective_suite": ["pmope_joint"],
            "fairness": {
                "allowed_difference_from_sealed_v20": (
                    "predeclared_field_equation_only"
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
        self.pair_batch_size = 4
        self.model_forward_invocations = 8
        self.pair_certificates = tuple(
            SimpleNamespace(pair_certificate_passed=False)
            for _ in range(32)
        )
        self.integrity_passed = True
        self.optimizer_constructed = False
        self.backward_performed = False
        self.training_performed = False
        self.external_data_accessed = False

    def verify(self) -> None:
        return None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "integrity_passed": self.integrity_passed,
            "diagnostic_summary": {
                "all_pairs_passed": False,
                "total_raw_sign_error_pixels": 999,
                "pair_result_is_bounded_gate": False,
            },
        }


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
    def __init__(self, diagnostic, *, passed: bool = True) -> None:
        self.run_id = runner.COVERAGE_STATE_PAET_OFFICIAL_RUN_ID
        self.diagnostic = diagnostic
        self.bounded_gate_passed = passed

    def canonical_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "bounded_gate_passed": self.bounded_gate_passed,
            "same_sign_response_diagnostic": {
                "is_gate": False,
            },
        }

    @property
    def decision_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _fake_measurement():
    return runner.CoverageStatePAETTrainingResourceMeasurement(
        device="cuda:0",
        measurement_scope=(
            "single_training_invocation_including_device_cache_setup_"
            "and_post_verification"
        ),
        baseline_allocated_bytes=100,
        baseline_reserved_bytes=200,
        peak_allocated_bytes=600,
        peak_reserved_bytes=900,
        elapsed_ns=4_000,
        updates=400,
        parameter_count=64064,
        v20_reference_status=(
            runner.COVERAGE_STATE_PAET_V20_RESOURCE_COMPARISON_STATUS
        ),
    )


def _fake_candidate(authorization):
    config = runner.expected_coverage_state_paet_config(
        authorization.preflight
    )
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(config)
    with torch.no_grad():
        model.scalar_energy_weight.fill_(0.125)
    training = _FakeTraining(authorization, model)
    certificate = _FakeCertificate(authorization, model)
    diagnostic = _FakeDiagnostic(authorization, model)
    decision = _FakeDecision(diagnostic)
    return model, training, certificate, diagnostic, decision


def test_run_id_mismatch_is_rejected_before_training(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    sealed_v20,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        sealed_v20,
    )
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("training was reached")

    monkeypatch.setattr(
        runner,
        "train_matched_coverage_state_paet_bfa_pmope_objectives",
        forbidden,
    )
    with pytest.raises(PermissionError, match="run_id"):
        runner.run_coverage_state_paet_bfa_pmope_bounded_400(
            authorization,
            runner.expected_coverage_state_paet_config(preflight),
            run_id="cure_lite_paet_bfa_v21_wrong",
            device="cuda:0",
        )
    assert called is False


def test_runner_calls_each_stage_once_and_keeps_resource_ratio_non_gating(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    sealed_v20,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        sealed_v20,
    )
    model, training, certificate, diagnostic, decision = _fake_candidate(
        authorization
    )
    order: list[str] = []

    def train(*args, **kwargs):
        order.append("training")
        return training

    def measure(operation, *, device):
        value = operation()
        order.append("measurement")
        return value, _fake_measurement()

    def certify(*args, **kwargs):
        order.append("certificate")
        assert kwargs["pair_batch_size"] == 4
        return certificate

    def evaluate(*args, **kwargs):
        order.append("zero_level")
        return diagnostic

    def decide(actual, *, run_id):
        order.append("decision")
        assert actual is diagnostic
        assert run_id == authorization.run_id
        return decision

    monkeypatch.setattr(
        runner,
        "train_matched_coverage_state_paet_bfa_pmope_objectives",
        train,
    )
    monkeypatch.setattr(runner, "_measure_training_invocation", measure)
    monkeypatch.setattr(
        runner,
        "audit_coverage_state_paet_pair_certificate",
        certify,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_coverage_state_zero_level_checkpoint",
        evaluate,
    )
    monkeypatch.setattr(
        runner,
        "decide_coverage_state_paet_bounded",
        decide,
    )
    monkeypatch.setattr(
        runner,
        "_deterministic_execution",
        lambda device: nullcontext(),
    )

    with pytest.raises(PermissionError, match="cuda:0"):
        runner.run_coverage_state_paet_bfa_pmope_bounded_400(
            authorization,
            model.config,
            run_id=runner.COVERAGE_STATE_PAET_OFFICIAL_RUN_ID,
            device="cpu",
        )
    assert order == []

    result = runner.run_coverage_state_paet_bfa_pmope_bounded_400(
        authorization,
        model.config,
        run_id=runner.COVERAGE_STATE_PAET_OFFICIAL_RUN_ID,
        device="cuda:0",
    )
    assert order == [
        "training",
        "measurement",
        "certificate",
        "zero_level",
        "decision",
    ]
    assert result.run_id == authorization.run_id
    assert result.canonical_payload()["run_id"] == authorization.run_id
    assert result.bounded_gate_passed
    assert result.formal800_eligible
    assert result.canonical_payload()["formal_800_authorized"] is False
    assert result.resource_measurement.canonical_payload()[
        "v20_comparison"
    ]["not_a_scientific_gate"] is True
