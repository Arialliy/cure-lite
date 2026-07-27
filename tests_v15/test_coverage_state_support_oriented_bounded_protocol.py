from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import cure_lite.experiment.coverage_state_bounded_runner as bounded_runner
from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
)
from cure_lite.coverage_state_sobolev import (
    CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_SEED,
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_bounded_runner import (
    COVERAGE_STATE_SUPPORT_ORIENTED_BOUNDED_AUTHORIZATION_SCHEMA,
    COVERAGE_STATE_SUPPORT_ORIENTED_BOUNDED_RESULT_SCHEMA,
    COVERAGE_STATE_V15A_PARENT_COMPLETE_FINGERPRINT,
    CoverageStateSupportOrientedBoundedRunAuthorization,
    prepare_coverage_state_support_oriented_bounded_run_authorization,
    run_coverage_state_support_oriented_bounded_400,
)
from cure_lite.experiment.coverage_state_dataset_free import (
    run_coverage_state_support_oriented_dataset_free_gate,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)


@pytest.fixture(scope="module")
def support_oriented_authorization():
    population = build_coverage_state_bounded_population(
        make_bounded_training_scalar_cache()
    )
    preflight = prepare_coverage_state_bounded_preflight(population)
    receipt = run_coverage_state_support_oriented_dataset_free_gate()
    return prepare_coverage_state_support_oriented_bounded_run_authorization(
        preflight,
        receipt,
    )


def _model_config(
    authorization: CoverageStateSupportOrientedBoundedRunAuthorization,
) -> CoverageStateLevelSetConfig:
    raw_catalog = authorization.preflight.population.cache.raw_catalog
    return CoverageStateLevelSetConfig(
        feature_channels=int(
            raw_catalog.natural_records[0].feature.shape[1]
        ),
        feature_stride=raw_catalog.feature_stride,
        width=bounded_runner.COVERAGE_STATE_BOUNDED_MODEL_WIDTH,
    )


def test_support_oriented_authorization_is_independent_and_fully_bound(
    support_oriented_authorization,
) -> None:
    authorization = support_oriented_authorization
    assert isinstance(
        authorization,
        CoverageStateSupportOrientedBoundedRunAuthorization,
    )
    assert authorization.training_authorized
    assert authorization.objective_suite == (
        "support_oriented_response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    assert authorization.candidate_objective == (
        "support_oriented_response_joint"
    )
    assert (
        authorization.candidate_objective_policy
        == CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
    )
    assert authorization.parent_v15a_complete_fingerprint == (
        "f925ece389a96cd6e8ef5487d91428d7981764b12601133cc3eaf9d11b782d35"
    )
    assert authorization.parent_v15a_complete_fingerprint == (
        COVERAGE_STATE_V15A_PARENT_COMPLETE_FINGERPRINT
    )
    payload = authorization.canonical_payload()
    assert payload["schema_version"] == (
        COVERAGE_STATE_SUPPORT_ORIENTED_BOUNDED_AUTHORIZATION_SCHEMA
    )
    assert payload["runtime_splits"] == ["D_R"]
    assert payload["formal_training_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("candidate_objective", "completion_rooted_response_joint"),
        ("candidate_objective_policy", "changed"),
        ("parent_v15a_complete_fingerprint", "0" * 64),
        (
            "objective_suite",
            (
                "support_oriented_response_joint",
                "separable_endpoint",
                "identity_joint",
            ),
        ),
    ),
)
def test_support_oriented_authorization_rejects_protocol_drift(
    support_oriented_authorization,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="binding changed"):
        replace(
            support_oriented_authorization,
            **{field: value},
        )


def test_support_oriented_candidate_gates_do_not_consume_control_outcomes(
    support_oriented_authorization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = support_oriented_authorization
    training = SimpleNamespace(
        results=tuple(
            SimpleNamespace(objective=name)
            for name in authorization.objective_suite
        )
    )
    diagnostics = (
        (
            "support_oriented_response_joint",
            SimpleNamespace(bounded_gate_passed=True),
        ),
        (
            "identity_joint",
            SimpleNamespace(bounded_gate_passed=False),
        ),
        (
            "separable_endpoint",
            SimpleNamespace(bounded_gate_passed=False),
        ),
    )
    monkeypatch.setattr(
        bounded_runner,
        "_bounded_result_checks",
        lambda *args, **kwargs: (
            ("execution_and_fairness", True),
            ("zero_level_gates", False),
        ),
    )
    checks = dict(
        bounded_runner._support_oriented_bounded_result_checks(
            authorization,
            training,
            diagnostics,
        )
    )
    assert "zero_level_gates" not in checks
    assert checks["candidate_original_zero_level_gates"] is True
    assert checks["control_diagnostics_complete"] is True
    assert all(checks.values())

    failed_candidate = (
        (
            "support_oriented_response_joint",
            SimpleNamespace(bounded_gate_passed=False),
        ),
        *diagnostics[1:],
    )
    failed = dict(
        bounded_runner._support_oriented_bounded_result_checks(
            authorization,
            training,
            failed_candidate,
        )
    )
    assert failed["candidate_original_zero_level_gates"] is False


def test_support_oriented_bounded_runner_dispatch_and_result_scope(
    support_oriented_authorization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = support_oriented_authorization
    config = _model_config(authorization)
    models = tuple(
        (
            name,
            CURELiteCoverageStateLevelSet(config).train(),
        )
        for name in authorization.objective_suite
    )
    training = SimpleNamespace(
        results=tuple(
            SimpleNamespace(objective=name)
            for name in authorization.objective_suite
        ),
        models=models,
        verify_unchanged=lambda: None,
        canonical_payload=lambda: {"training": "stub"},
    )
    observed: dict[str, object] = {}

    def train(*args, **kwargs):
        observed["config"] = kwargs["config"]
        observed["authorization"] = kwargs["authorization"]
        return training

    monkeypatch.setattr(
        bounded_runner,
        "train_matched_coverage_state_support_oriented_objectives",
        train,
    )

    def evaluate(model, cache, *, device):
        assert model.training is False
        assert cache is authorization.preflight.population.cache
        return SimpleNamespace(
            bounded_gate_passed=True,
            canonical_payload=lambda: {"bounded_gate_passed": True},
        )

    monkeypatch.setattr(
        bounded_runner,
        "evaluate_coverage_state_zero_level_checkpoint",
        evaluate,
    )
    monkeypatch.setattr(
        bounded_runner,
        "_support_oriented_bounded_result_checks",
        lambda *args, **kwargs: (("orchestration_stub", True),),
    )
    result = run_coverage_state_support_oriented_bounded_400(
        authorization,
        config,
        device="cpu",
    )
    assert observed["authorization"] is authorization
    assert observed["config"].seed == COVERAGE_STATE_BOUNDED_SEED
    assert all(not model.training for _, model in result.training.models)
    payload = result.canonical_payload()
    assert payload["schema_version"] == (
        COVERAGE_STATE_SUPPORT_ORIENTED_BOUNDED_RESULT_SCHEMA
    )
    assert payload["runtime_splits"] == ["D_R"]
    assert payload["candidate_qualification_uses_original_gates_only"]
    assert payload["control_outcomes_are_not_candidate_gates"]
    assert payload["formal_training_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
