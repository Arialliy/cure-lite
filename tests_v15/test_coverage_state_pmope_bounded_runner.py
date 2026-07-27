from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

import cure_lite.experiment.coverage_state_pmope_bounded_runner as runner
from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_pmope_dataset_free import (
    CoverageStatePMOPEDatasetFreeReceipt,
    run_coverage_state_pmope_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_pmope_dr_gate import (
    CoverageStatePMOPEDRGateReceipt,
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
def dataset_free() -> CoverageStatePMOPEDatasetFreeReceipt:
    return run_coverage_state_pmope_dataset_free_gate()


class _FakeDRGateReceipt(CoverageStatePMOPEDRGateReceipt):
    def __init__(self, **values: object) -> None:
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "full_verify_calls", 0)
        object.__setattr__(
            self,
            "evidence_fingerprint",
            stable_fingerprint(self._evidence_payload()),
        )

    def _evidence_payload(self) -> dict[str, object]:
        return {
            "dataset_free": self.dataset_free_receipt_fingerprint,
            "real_inputs": self.real_inputs_build_fingerprint,
            "source": self.source_binding_fingerprint,
            "population": self.bounded_population_fingerprint,
            "cache": self.bounded_cache_fingerprint,
            "v17": self.v17_binding_fingerprint,
            "execution_seed": self.execution_seed,
            "model": self.model_config_payload,
            "initial": self.initial_model_fingerprint,
            "final": self.final_model_fingerprint,
            "gradient_rows": list(self.gradient_rows),
            "geometry": self.geometry,
            "checks": dict(self.checks),
            "passed": self.passed,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": runner.COVERAGE_STATE_PMOPE_DR_GATE_SCHEMA,
            **self._evidence_payload(),
            "checks": dict(self.checks),
            "all_pass": self.all_pass,
            "evidence_fingerprint": self.evidence_fingerprint,
        }

    @property
    def all_pass(self) -> bool:
        return self.passed and all(value for _, value in self.checks)

    def verify_unchanged(self) -> None:
        object.__setattr__(
            self,
            "full_verify_calls",
            self.full_verify_calls + 1,
        )
        if (
            stable_fingerprint(self._evidence_payload())
            != self.evidence_fingerprint
        ):
            raise RuntimeError("fake D_R receipt changed")

    def with_changes(self, **changes: object) -> _FakeDRGateReceipt:
        values = {
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "real_inputs_build_fingerprint": (
                self.real_inputs_build_fingerprint
            ),
            "source_binding_fingerprint": self.source_binding_fingerprint,
            "bounded_population_fingerprint": (
                self.bounded_population_fingerprint
            ),
            "bounded_cache_fingerprint": self.bounded_cache_fingerprint,
            "v17_binding_fingerprint": self.v17_binding_fingerprint,
            "execution_seed": self.execution_seed,
            "model_config_payload": self.model_config_payload,
            "initial_model_fingerprint": self.initial_model_fingerprint,
            "final_model_fingerprint": self.final_model_fingerprint,
            "gradient_rows": self.gradient_rows,
            "geometry": self.geometry,
            "checks": self.checks,
            "passed": self.passed,
        }
        values.update(changes)
        return _FakeDRGateReceipt(**values)


def _fake_dr_gate(
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
    *,
    passed: bool = True,
) -> _FakeDRGateReceipt:
    model_config = runner.expected_coverage_state_pmope_config(
        preflight
    )
    _, dr_initial = runner._common_initial_model_fingerprints(
        model_config
    )
    return _FakeDRGateReceipt(
        dataset_free_receipt_fingerprint=(
            dataset_free.receipt_fingerprint
        ),
        real_inputs_build_fingerprint="1" * 64,
        source_binding_fingerprint="2" * 64,
        bounded_population_fingerprint=(
            preflight.population.population_fingerprint
        ),
        bounded_cache_fingerprint=(
            preflight.population.bounded_cache_fingerprint
        ),
        v17_binding_fingerprint=(
            "851a67d97a89f8bdf03bf09875648bcf3cd844bde9db884ba55730d0d10cf42d"
        ),
        execution_seed=42,
        model_config_payload=runner._dr_gate_model_config_payload(
            model_config
        ),
        initial_model_fingerprint=dr_initial,
        final_model_fingerprint=dr_initial,
        gradient_rows=(
            {"role": "clean", "finite": True, "nonzero": True},
        ),
        geometry={
            "mass_rows": [
                {"role": "clean", "positive_measure": True},
            ],
        },
        checks=(("all_real_D_R_checks", passed),),
        passed=passed,
    )


def _patch_toy_historical_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
) -> None:
    monkeypatch.setattr(
        runner,
        "COVERAGE_STATE_PMOPE_HISTORICAL_SCHEDULE_FINGERPRINT",
        preflight.schedule.schedule_fingerprint,
    )
    monkeypatch.setattr(
        runner,
        "COVERAGE_STATE_PMOPE_HISTORICAL_CACHE_FINGERPRINT",
        preflight.population.bounded_cache_fingerprint,
    )


def _authorization(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
):
    _patch_toy_historical_coordinates(monkeypatch, preflight)
    return runner.prepare_coverage_state_pmope_bounded_run_authorization(
        preflight,
        dataset_free,
        _fake_dr_gate(preflight, dataset_free),
    )


def test_sealed_v17_controls_are_read_only_and_exact() -> None:
    sealed = runner.verify_current_sealed_v17_controls()
    payload = sealed.canonical_payload()

    assert sealed.receipt_fingerprint == (
        "851a67d97a89f8bdf03bf09875648bcf3cd844bde9db884ba55730d0d10cf42d"
    )
    assert tuple(value.objective for value in sealed.controls) == (
        "support_oriented_response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    assert payload["historical_frozen_controls"] is True
    assert payload["contemporaneous_controls"] is False
    assert payload["control_outcomes_are_not_candidate_gates"] is True
    assert payload["model_deserialization_performed"] is False
    assert payload["evaluator_called"] is False
    assert payload["training_performed"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False


def test_authorization_is_singleton_seed42_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        dataset_free,
    )
    payload = authorization.canonical_payload()

    assert authorization.training_authorized
    assert authorization.objective_suite == ("pmope_joint",)
    assert authorization.candidate_objective == "pmope_joint"
    assert authorization.candidate_objective_policy == (
        runner.CSLF_PMOPE_POLICY
    )
    assert authorization.expected_parameter_count == 64064
    assert authorization.dr_gate_receipt.full_verify_calls == 1
    assert payload["budget"] == {
        "seed": 42,
        "epochs": 10,
        "steps_per_epoch": 40,
        "updates": 400,
        "objectives": 1,
    }
    assert payload["fixed_margin_hex"] == float(0.225).hex()
    assert payload["formal_800_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["resume_allowed"] is False
    assert payload["automatic_retry_allowed"] is False


def test_authorization_rejects_failed_D_R_gate(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    _patch_toy_historical_coordinates(monkeypatch, preflight)
    with pytest.raises(PermissionError, match="D_R gate"):
        runner.prepare_coverage_state_pmope_bounded_run_authorization(
            preflight,
            dataset_free,
            _fake_dr_gate(
                preflight,
                dataset_free,
                passed=False,
            ),
        )


def test_authorization_rejects_missing_D_R_mass_evidence(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    _patch_toy_historical_coordinates(monkeypatch, preflight)
    receipt = _fake_dr_gate(preflight, dataset_free).with_changes(
        geometry={"mass_rows": []},
    )
    with pytest.raises(PermissionError, match="D_R gate"):
        runner.prepare_coverage_state_pmope_bounded_run_authorization(
            preflight,
            dataset_free,
            receipt,
        )


@pytest.mark.parametrize(
    "drift",
    ["config", "initial_state", "sealed_v17", "execution_seed"],
)
def test_authorization_rejects_D_R_model_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
    drift: str,
) -> None:
    _patch_toy_historical_coordinates(monkeypatch, preflight)
    receipt = _fake_dr_gate(preflight, dataset_free)
    if drift == "config":
        payload = dict(receipt.model_config_payload)
        payload["width"] = 31
        receipt = receipt.with_changes(model_config_payload=payload)
    elif drift == "initial_state":
        receipt = receipt.with_changes(
            initial_model_fingerprint="3" * 64,
            final_model_fingerprint="3" * 64,
        )
    elif drift == "sealed_v17":
        receipt = receipt.with_changes(
            v17_binding_fingerprint="4" * 64,
        )
    else:
        receipt = receipt.with_changes(execution_seed=43)
    with pytest.raises(ValueError, match="authorization binding"):
        runner.prepare_coverage_state_pmope_bounded_run_authorization(
            preflight,
            dataset_free,
            receipt,
        )


def test_authorization_rejects_historical_three_objective_suite(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        dataset_free,
    )
    with pytest.raises(ValueError, match="authorization binding"):
        replace(
            authorization,
            objective_suite=(
                "pmope_joint",
                "identity_joint",
                "separable_endpoint",
            ),
        )


class _FakeTraining:
    def __init__(
        self,
        *,
        model: CURELiteCenteredMixedInteractionLevelSet,
        schedule_fingerprint: str,
        cache_fingerprint: str,
    ) -> None:
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
            first_nonzero_gradient_update=(
                ("joint_hidden_bias", 1),
                ("joint_state_weight", 1),
                ("scalar_energy_weight", 0),
            ),
            initial_model_fingerprint=(
                runner
                .COVERAGE_STATE_PMOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            ),
            schedule_fingerprint=schedule_fingerprint,
            cache_fingerprint=cache_fingerprint,
            optimizer_config_fingerprint=(
                runner
                .COVERAGE_STATE_PMOPE_HISTORICAL_OPTIMIZER_FINGERPRINT
            ),
            device_cache_fingerprint=(
                runner
                .COVERAGE_STATE_PMOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT
            ),
        )
        self.results = (row,)
        self.models = (("pmope_joint", model),)
        self.common_initial_model_fingerprint = (
            row.initial_model_fingerprint
        )
        self.schedule_fingerprint = schedule_fingerprint
        self.cache_fingerprint = cache_fingerprint

    def verify_unchanged(self) -> None:
        return None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "objective_suite": ["pmope_joint"],
            "fairness": {
                "single_candidate_only": True,
                "historical_controls_retrained": False,
            },
        }


class _FakeDiagnostic:
    def __init__(
        self,
        *,
        model: CURELiteCenteredMixedInteractionLevelSet,
        cache_fingerprint: str,
        dataset: str,
        gates_pass: bool = True,
    ) -> None:
        self.checkpoint_fingerprint = module_state_fingerprint(model)
        self.cache_fingerprint = cache_fingerprint
        self.dataset = dataset
        self.split = "D_R"
        self.factual_miss_gate_passed = gates_pass
        self.factual_no_miss_gate_passed = gates_pass
        self.clean_defined_metrics_passed = gates_pass
        self.clean_compact_support_gate_passed = gates_pass
        self.component_null_gate_passed = gates_pass
        self.identity_null_gate_passed = gates_pass
        self.diagnostic_null_gate_passed = gates_pass
        self.bounded_gate_passed = gates_pass
        self.backward_calls = 0
        self.optimizer_steps = 0
        self.config = SimpleNamespace(
            residual_threshold=0.0,
            threshold_search_performed=False,
            input_representation="phase_preserving",
            d_v_accessed=False,
            d_t_accessed=False,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "gates": {
                "bounded_gate_passed": self.bounded_gate_passed,
            },
            "threshold": 0.0,
        }


def _fake_candidate_evidence(authorization):
    model = CURELiteCenteredMixedInteractionLevelSet(
        runner.expected_coverage_state_pmope_config(
            authorization.preflight
        )
    )
    training = _FakeTraining(
        model=model,
        schedule_fingerprint=(
            authorization.preflight.schedule.schedule_fingerprint
        ),
        cache_fingerprint=(
            authorization
            .preflight.population.bounded_cache_fingerprint
        ),
    )
    diagnostic = _FakeDiagnostic(
        model=model,
        cache_fingerprint=(
            authorization
            .preflight.population.bounded_cache_fingerprint
        ),
        dataset=(
            authorization
            .preflight.population.cache.raw_catalog.dataset
        ),
    )
    return model, training, diagnostic


def test_candidate_pass_does_not_depend_on_failed_v17_controls(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        dataset_free,
    )
    _, training, diagnostic = _fake_candidate_evidence(
        authorization
    )
    checks = runner._pmope_bounded_result_checks(
        authorization,
        training,
        diagnostic,
    )
    result = runner.CoverageStatePMOPEBoundedRunResult(
        authorization=authorization,
        training=training,
        diagnostic=diagnostic,
        checks=checks,
    )

    assert all(dict(checks).values())
    assert result.bounded_gate_passed
    assert all(
        control.bounded_gate_passed is False
        for control in authorization.sealed_v17_receipt.controls
    )
    payload = result.canonical_payload()
    assert payload[
        "historical_control_outcomes_are_candidate_gates"
    ] is False
    assert payload["formal800_eligible"] is True
    assert payload["formal_800_authorized"] is False


def test_one_failed_candidate_gate_fails_the_result(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        dataset_free,
    )
    model, training, _ = _fake_candidate_evidence(authorization)
    diagnostic = _FakeDiagnostic(
        model=model,
        cache_fingerprint=(
            preflight.population.bounded_cache_fingerprint
        ),
        dataset=preflight.population.cache.raw_catalog.dataset,
        gates_pass=False,
    )
    checks = runner._pmope_bounded_result_checks(
        authorization,
        training,
        diagnostic,
    )
    result = runner.CoverageStatePMOPEBoundedRunResult(
        authorization=authorization,
        training=training,
        diagnostic=diagnostic,
        checks=checks,
    )

    assert dict(checks)["candidate_seven_zero_level_gates"] is False
    assert result.bounded_gate_passed is False
    assert result.canonical_payload()["formal800_eligible"] is False


def test_runner_dispatches_one_candidate_without_real_training(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        dataset_free,
    )
    model, training, diagnostic = _fake_candidate_evidence(
        authorization
    )
    captured: dict[str, object] = {}

    def train(actual_config, actual_cache, actual_schedule, **kwargs):
        captured.update(
            {
                "config": actual_config,
                "cache": actual_cache,
                "schedule": actual_schedule,
                **kwargs,
            }
        )
        return training

    def evaluate(actual_model, actual_cache, **kwargs):
        captured["evaluated_model"] = actual_model
        captured["evaluated_cache"] = actual_cache
        captured["evaluation_kwargs"] = kwargs
        return diagnostic

    monkeypatch.setattr(
        runner,
        "train_matched_coverage_state_cmif_pmope_objectives",
        train,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_coverage_state_zero_level_checkpoint",
        evaluate,
    )
    monkeypatch.setattr(
        runner,
        "_deterministic_execution",
        lambda device: nullcontext(),
    )
    with pytest.raises(PermissionError, match="cuda:0"):
        runner.run_coverage_state_cmif_pmope_bounded_400(
            authorization,
            model.config,
            device="cpu",
        )
    assert captured == {}
    result = runner.run_coverage_state_cmif_pmope_bounded_400(
        authorization,
        model.config,
        device="cuda:0",
    )

    assert result.bounded_gate_passed
    assert captured["authorization"] is authorization
    assert captured["config"].seed == 42
    assert captured["cache"] is preflight.population.cache
    assert captured["schedule"] is preflight.schedule
    assert captured["evaluated_model"] is model
    assert captured["evaluated_cache"] is preflight.population.cache


def test_implementation_binding_contains_only_current_singleton_path(
    monkeypatch: pytest.MonkeyPatch,
    preflight,
    dataset_free: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    authorization = _authorization(
        monkeypatch,
        preflight,
        dataset_free,
    )
    paths = tuple(
        path for path, _ in authorization.implementation_binding
    )

    assert len(paths) == 37
    assert (
        "cure_lite/experiment/coverage_state_pmope_bounded_runner.py"
        in paths
    )
    assert (
        "cure_lite/experiment/coverage_state_pmope_sealed_v17.py"
        in paths
    )
    assert (
        "cure_lite/experiment/coverage_state_pmope_dr_gate.py"
        in paths
    )
    assert (
        "cure_lite/experiment/coverage_state_cmif_bounded_runner.py"
        not in paths
    )
    assert all(len(digest) == 64 for _, digest in (
        authorization.implementation_binding
    ))
