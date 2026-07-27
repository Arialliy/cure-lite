from __future__ import annotations

from types import SimpleNamespace

import torch

import cure_lite.experiment.coverage_state_training as training_module
import cure_lite.train.coverage_state_fused_step as fused_step_module
from cure_lite.coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.coverage_state_sobolev import (
    CSLF_PMOPE_POLICY,
    CoverageStateSobolevConfig,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES,
    COVERAGE_STATE_REGISTERED_MATCHED_OBJECTIVE_SUITES,
    CoverageStateMatchedTrainingConfig,
    train_matched_coverage_state_cmif_pmope_objectives,
)
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
    coverage_state_fused_train_step,
    coverage_state_pair_objective_policy,
)
from tests_v15.coverage_state_test_helpers import TOY_STRIDE
from tests_v15.coverage_state_training_test_helpers import (
    make_training_fused_batch,
    make_training_scalar_cache,
)


def test_pmope_policy_and_single_candidate_suite_are_fixed() -> None:
    assert CoverageStatePairObjective.PMOPE_JOINT.value == "pmope_joint"
    assert coverage_state_pair_objective_policy(
        CoverageStatePairObjective.PMOPE_JOINT
    ) == CSLF_PMOPE_POLICY
    assert COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES == (
        CoverageStatePairObjective.PMOPE_JOINT,
    )
    assert (
        COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES
        in COVERAGE_STATE_REGISTERED_MATCHED_OBJECTIVE_SUITES
    )


def test_fused_step_dispatches_pmope_once(
    monkeypatch,
) -> None:
    calls: list[tuple[bool, object]] = []

    def pmope(
        field_plus,
        field_minus,
        targets,
        *,
        config,
        validate,
    ):
        calls.append((validate, targets))
        return SimpleNamespace(
            loss=field_plus.square().mean() + field_minus.square().mean()
        )

    monkeypatch.setattr(
        fused_step_module,
        "coverage_state_pmope_pair_loss_from_targets",
        pmope,
    )
    torch.manual_seed(712)
    model = CURELiteCoverageStateLevelSet(
        CoverageStateLevelSetConfig(
            feature_channels=2,
            feature_stride=TOY_STRIDE,
            width=4,
        )
    )
    batch = make_training_fused_batch()
    logs = coverage_state_fused_train_step(
        model,
        torch.optim.SGD(model.parameters(), lr=1.0e-2),
        batch,
        config=CoverageStateSobolevConfig(
            truncation_radius=TOY_STRIDE
        ),
        pair_objective=CoverageStatePairObjective.PMOPE_JOINT,
    )
    assert calls == [(False, batch.pairs.joint_targets)]
    assert logs["pair_objective"] == "pmope_joint"
    assert logs["pair_objective_policy"] == CSLF_PMOPE_POLICY


def test_cmif_pmope_wrapper_routes_only_the_fixed_candidate(
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
    model_config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=2,
        feature_stride=TOY_STRIDE,
        width=4,
    )
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
    result = train_matched_coverage_state_cmif_pmope_objectives(
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
        CoverageStatePairObjective.PMOPE_JOINT,
    )
    assert captured["device"] == "cpu"
    assert captured["authorization"] is None
    assert captured["epoch_callback"] is None


def test_cmif_pmope_single_candidate_closes_toy_training_ledger() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=3,
        ),
    )
    model_config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=2,
        feature_stride=TOY_STRIDE,
        width=4,
    )
    result = train_matched_coverage_state_cmif_pmope_objectives(
        model_config,
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
        CURELiteCenteredMixedInteractionLevelSet
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
    payload = result.canonical_payload()
    assert payload["objective_suite"] == ["pmope_joint"]
    assert payload["fairness"] == {
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
        "allowed_difference_from_sealed_v17": (
            "predeclared_pair_objective_only"
        ),
        "same_model_class": True,
        "same_model_config": True,
        "same_parameter_count": True,
        "same_parameter_shapes": True,
    }
    assert "response_identity_share_joint_measure" not in payload["fairness"]
    assert "separable_uses_endpoint_absolute_measures" not in payload[
        "fairness"
    ]
