from __future__ import annotations

import torch

import cure_lite.experiment.coverage_state_training as training_module
from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from cure_lite.coverage_state_phase_preserving import (
    build_coverage_state_level_set,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.experiment.coverage_state_training import (
    CoverageStateMatchedTrainingConfig,
    train_matched_coverage_state_bfa_pmope_objectives,
)
from cure_lite.train.coverage_state_fused_step import (
    COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES,
)
from tests_v15.coverage_state_test_helpers import TOY_STRIDE
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


def _toy_config() -> CoverageStateBinaryFlipAntisymmetricConfig:
    return CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=2,
        feature_stride=TOY_STRIDE,
        width=4,
    )


def test_factory_registers_the_exact_bfa_model() -> None:
    config = _toy_config()
    model = build_coverage_state_level_set(config)

    assert type(model) is CURELiteBinaryFlipAntisymmetricLevelSet
    assert model.config is config
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        config.expected_parameter_count
    )


def test_bfa_pmope_wrapper_routes_only_the_frozen_objective(
    monkeypatch,
) -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=2,
        ),
    )
    model_config = _toy_config()
    matched_config = CoverageStateMatchedTrainingConfig(seed=42)
    sentinel = object()
    captured: dict[str, object] = {}

    def train_suite(
        actual_model_config,
        actual_cache,
        actual_schedule,
        **kwargs,
    ):
        captured.update(
            {
                "model_config": actual_model_config,
                "cache": actual_cache,
                "schedule": actual_schedule,
                **kwargs,
            }
        )
        return sentinel

    monkeypatch.setattr(
        training_module,
        "_train_matched_coverage_state_objective_suite",
        train_suite,
    )
    result = train_matched_coverage_state_bfa_pmope_objectives(
        model_config,
        cache,
        schedule,
        config=matched_config,
        device="cpu",
    )

    assert result is sentinel
    assert captured["model_config"] is model_config
    assert captured["cache"] is cache
    assert captured["schedule"] is schedule
    assert captured["config"] is matched_config
    assert captured["objectives"] == (
        COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES
    )
    assert captured["device"] == "cpu"
    assert captured["authorization"] is None
    assert captured["epoch_callback"] is None


def test_bfa_pmope_toy_training_closes_the_compute_ledger() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=3,
        ),
    )
    result = train_matched_coverage_state_bfa_pmope_objectives(
        _toy_config(),
        cache,
        schedule,
        config=CoverageStateMatchedTrainingConfig(seed=42),
        device="cpu",
    )

    assert tuple(value.objective for value in result.results) == (
        "pmope_joint",
    )
    assert tuple(name for name, _ in result.models) == ("pmope_joint",)
    assert type(result.models[0][1]) is (
        CURELiteBinaryFlipAntisymmetricLevelSet
    )
    trained = result.results[0]
    assert trained.completed_updates == 3
    assert trained.forward_calls == 3
    assert trained.backward_calls == 3
    assert trained.optimizer_steps == 3
    assert trained.logical_state_evaluations == 36
    assert trained.finite_state_audits == 4
    latency = dict(trained.first_nonzero_gradient_update)
    assert latency["scalar_energy_weight"] == 0
    assert latency["joint_state_weight"] <= 2
    assert latency["joint_hidden_bias"] <= 2

    fairness = result.canonical_payload()["fairness"]
    assert fairness == {
        "single_candidate_only": True,
        "same_initial_state": True,
        "same_schedule": True,
        "same_endpoints": True,
        "same_model": True,
        "same_optimizer": True,
        "same_device_cache": True,
        "same_compute_budget": True,
        "same_natural_branches": True,
        "historical_controls_retrained": False,
        "historical_v18_objective_reused": True,
        "allowed_difference_from_sealed_v18": (
            "predeclared_field_equation_only"
        ),
        "same_model_class": True,
        "same_model_config": True,
        "same_parameter_count": True,
        "same_parameter_shapes": True,
    }


def test_bfa_training_wrapper_rejects_non_bfa_config() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=1,
        ),
    )

    try:
        train_matched_coverage_state_bfa_pmope_objectives(
            object(),  # type: ignore[arg-type]
            cache,
            schedule,
            config=CoverageStateMatchedTrainingConfig(seed=42),
            device=torch.device("cpu"),
        )
    except TypeError as error:
        assert "CoverageStateBinaryFlipAntisymmetricConfig" in str(error)
    else:
        raise AssertionError("non-BFA config was accepted")
