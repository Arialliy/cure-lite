from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import cure_lite.experiment.coverage_state_cmif_bounded_runner as runner
from cure_lite.coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from cure_lite.coverage_state_phase_preserving import (
    CURELitePhasePreservingCoverageStateLevelSet,
    CoverageStatePhasePreservingConfig,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_SEED,
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_cmif_dataset_free import (
    run_coverage_state_cmif_dataset_free_gate,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)


def _toy_authorization(
    monkeypatch: pytest.MonkeyPatch,
):
    population = build_coverage_state_bounded_population(
        make_bounded_training_scalar_cache()
    )
    preflight = prepare_coverage_state_bounded_preflight(population)
    dataset_free = run_coverage_state_cmif_dataset_free_gate()
    actual_p0 = runner._verify_persisted_cmif_p0_authorization()
    monkeypatch.setattr(
        runner,
        "COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT",
        population.population_fingerprint,
    )
    toy_p0 = dict(actual_p0)
    toy_p0["bounded_population_fingerprint"] = (
        population.population_fingerprint
    )
    without_fingerprint = dict(toy_p0)
    without_fingerprint.pop("evidence_fingerprint")
    toy_p0["evidence_fingerprint"] = runner.stable_fingerprint(
        without_fingerprint
    )
    monkeypatch.setattr(
        runner,
        "COVERAGE_STATE_CMIF_P0_EVIDENCE_FINGERPRINT",
        toy_p0["evidence_fingerprint"],
    )
    monkeypatch.setattr(
        runner,
        "_verify_persisted_cmif_p0_authorization",
        lambda: dict(toy_p0),
    )
    authorization = (
        runner.prepare_coverage_state_cmif_bounded_run_authorization(
            preflight,
            dataset_free,
        )
    )
    return authorization


def test_fixed_persisted_p0_authorization_is_complete() -> None:
    payload = runner._verify_persisted_cmif_p0_authorization()

    assert payload["training_authorized"] is True
    assert payload["r1_complete_fingerprint"] == (
        runner.COVERAGE_STATE_CMIF_P0_R1_COMPLETE_FINGERPRINT
    )
    assert payload["r2_complete_fingerprint"] == (
        runner.COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT
    )
    assert payload["p0_core_receipt_fingerprint"] == (
        runner.COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT
    )
    assert payload["bounded_population_fingerprint"] == (
        runner.COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT
    )
    assert payload["dataset_free_receipt_fingerprint"] == (
        runner.COVERAGE_STATE_CMIF_P0_DATASET_FREE_RECEIPT_FINGERPRINT
    )
    assert payload["checks"]
    assert all(payload["checks"].values())


def test_cmif_authorization_binds_p0_model_suite_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = _toy_authorization(monkeypatch)
    config = runner.expected_coverage_state_cmif_config(
        authorization.preflight
    )
    payload = authorization.canonical_payload()

    assert authorization.training_authorized
    assert config.expected_parameter_count == 64064
    assert authorization.expected_parameter_count == 64064
    assert authorization.objective_suite == (
        "support_oriented_response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    assert authorization.preflight.schedule.config.seed == (
        COVERAGE_STATE_BOUNDED_SEED
    )
    assert authorization.preflight.schedule.config.updates == 400
    assert payload["model_class"] == (
        "CURELiteCenteredMixedInteractionLevelSet"
    )
    assert payload["input_representation"] == "phase_preserving"
    assert payload["checks"]["persisted_p0_authorized"] is True
    assert payload["formal_800_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False


def test_cmif_authorization_rejects_p0_or_model_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = _toy_authorization(monkeypatch)

    with pytest.raises(ValueError, match="binding changed"):
        replace(
            authorization,
            p0_r2_complete_fingerprint="0" * 64,
        )
    with pytest.raises(ValueError, match="binding changed"):
        replace(authorization, expected_parameter_count=1)
    with pytest.raises(PermissionError, match="model config"):
        authorization.verify_model_config(
            CoverageStateCenteredMixedInteractionConfig(
                feature_channels=64,
                feature_stride=4,
                width=31,
            )
        )
    with pytest.raises(PermissionError, match="model config"):
        authorization.verify_model_config(
            CoverageStatePhasePreservingConfig(
                feature_channels=64,
                feature_stride=4,
                width=32,
            )
        )


def test_cmif_result_checks_require_exact_models_and_candidate_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = _toy_authorization(monkeypatch)
    config = runner.expected_coverage_state_cmif_config(
        authorization.preflight
    )
    models = tuple(
        (
            name,
            CURELiteCenteredMixedInteractionLevelSet(config),
        )
        for name in authorization.objective_suite
    )
    training = SimpleNamespace(
        results=tuple(
            SimpleNamespace(objective=name)
            for name in authorization.objective_suite
        ),
        models=models,
    )
    diagnostics = tuple(
        (
            name,
            SimpleNamespace(
                bounded_gate_passed=(index == 0),
                config=SimpleNamespace(
                    input_representation="phase_preserving"
                ),
            ),
        )
        for index, name in enumerate(authorization.objective_suite)
    )
    monkeypatch.setattr(
        runner,
        "_bounded_result_checks",
        lambda *args, **kwargs: (
            ("execution_and_fairness", True),
            ("authorized_model_config", False),
            ("zero_level_gates", False),
        ),
    )

    checks = dict(
        runner._cmif_bounded_result_checks(
            authorization,
            training,
            diagnostics,
        )
    )

    assert checks["candidate_original_zero_level_gates"] is True
    assert checks["control_diagnostics_complete"] is True
    assert checks["all_models_exact_cmif_class"] is True
    assert checks["all_models_same_cmif_config"] is True
    assert checks["all_models_expected_parameter_count"] is True
    assert checks["authorized_model_config"] is True
    assert checks["persisted_p0_authorization_bound"] is True
    assert all(checks.values())

    changed_models = (
        models[0],
        (
            "identity_joint",
            CURELitePhasePreservingCoverageStateLevelSet(
                CoverageStatePhasePreservingConfig(
                    feature_channels=64,
                    feature_stride=4,
                    width=32,
                )
            ),
        ),
        models[2],
    )
    changed = dict(
        runner._cmif_bounded_result_checks(
            authorization,
            SimpleNamespace(
                results=training.results,
                models=changed_models,
            ),
            diagnostics,
        )
    )
    assert changed["all_models_exact_cmif_class"] is False
    assert changed["authorized_model_config"] is False


def test_cmif_runner_dispatches_fixed_suite_without_real_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = _toy_authorization(monkeypatch)
    config = runner.expected_coverage_state_cmif_config(
        authorization.preflight
    )
    models = tuple(
        (
            name,
            CURELiteCenteredMixedInteractionLevelSet(config).train(),
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
        observed["model_config"] = args[0]
        observed["training_config"] = kwargs["config"]
        observed["authorization"] = kwargs["authorization"]
        return training

    monkeypatch.setattr(
        runner,
        "train_matched_coverage_state_cmif_support_oriented_objectives",
        train,
    )

    def evaluate(model, cache, *, device, config):
        assert type(model) is CURELiteCenteredMixedInteractionLevelSet
        assert model.training is False
        assert config.input_representation == "phase_preserving"
        return SimpleNamespace(
            bounded_gate_passed=True,
            canonical_payload=lambda: {"bounded_gate_passed": True},
        )

    monkeypatch.setattr(
        runner,
        "evaluate_coverage_state_zero_level_checkpoint",
        evaluate,
    )
    monkeypatch.setattr(
        runner,
        "_cmif_bounded_result_checks",
        lambda *args, **kwargs: (("orchestration_stub", True),),
    )

    result = runner.run_coverage_state_cmif_support_oriented_bounded_400(
        authorization,
        config,
        device="cpu",
    )

    assert observed["model_config"] is config
    assert observed["authorization"] is authorization
    assert observed["training_config"].seed == (
        COVERAGE_STATE_BOUNDED_SEED
    )
    payload = result.canonical_payload()
    assert payload["expected_parameter_count"] == 64064
    assert payload["runtime_splits"] == ["D_R"]
    assert payload["formal_800_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False
