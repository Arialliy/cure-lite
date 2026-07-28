from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from torch import Tensor

from cure_lite.coverage_state_centered_mixed_interaction import (
    CoverageStateCenteredMixedInteractionConfig,
    CURELiteCenteredMixedInteractionLevelSet,
)
from cure_lite.coverage_state_sobolev import (
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    coverage_state_pmope_pair_loss_from_targets,
    prepare_coverage_state_pair_targets,
)
from cure_lite.coverage_state_supremal_projection import (
    CSLF_USCOPE_POLICY,
    CoverageStateUSCOPELossFields,
    coverage_state_uscope_pair_loss_from_targets,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_REGISTERED_MATCHED_OBJECTIVE_SUITES,
    COVERAGE_STATE_USCOPE_MATCHED_OBJECTIVES,
    CoverageStateMatchedTrainingConfig,
    CoverageStateMatchedTrainingResult,
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
    train_matched_coverage_state_cmif_uscope_objectives,
)
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
    coverage_state_pair_objective_policy,
)


_RADIUS = 4
_EXPECTED_POLICY = (
    "uniform_sobolev_chebyshev_orthant_projection_energy_v1"
)


def _mask(
    size: int,
    coordinates: Iterable[tuple[int, int]] = (),
) -> Tensor:
    result = torch.zeros(1, 1, size, size, dtype=torch.bool)
    for row, column in coordinates:
        result[..., row, column] = True
    return result


def _pair_targets(
    *,
    size: int = 9,
    invalid: Iterable[tuple[int, int]] = (),
) -> CoverageStatePairTargets:
    target_plus = _mask(size, ((2, 2),))
    target_minus = target_plus | _mask(size, ((4, 4),))
    occupancy_plus = _mask(size, ((4, 4),))
    occupancy_minus = _mask(size)
    valid = torch.ones_like(target_plus)
    for row, column in invalid:
        valid[..., row, column] = False
    return prepare_coverage_state_pair_targets(
        occupancy_plus,
        occupancy_minus,
        target_plus,
        target_minus,
        valid,
        config=CoverageStateSobolevConfig(
            truncation_radius=_RADIUS,
        ),
    )


def _orthant_field(target_field: Tensor, magnitude: float) -> Tensor:
    return torch.sign(target_field) * magnitude


def _repeat_targets(
    targets: CoverageStatePairTargets,
    count: int,
) -> CoverageStatePairTargets:
    values = {
        name: getattr(targets, name).repeat(count, 1, 1, 1)
        for name in (
            "target_field_plus",
            "target_field_minus",
            "focus_support",
            "focus_support_field",
            "integration_measure",
            "valid_mask",
        )
    }
    result = CoverageStatePairTargets(**values)
    result.validate()
    return result


def test_uscope_exact_product_gauge_and_pmope_violations() -> None:
    targets = _pair_targets()
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)
    field_plus = torch.zeros_like(targets.target_field_plus)
    field_minus = torch.zeros_like(targets.target_field_minus)

    result = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )
    pmope = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )
    expected_sobolev = 0.5 * (
        pmope.per_state_value_power
        + pmope.per_state_spatial_power
    )
    expected_gamma = torch.maximum(
        pmope.violation_plus.flatten(1).amax(dim=1),
        pmope.violation_minus.flatten(1).amax(dim=1),
    )
    expected_product = 0.5 * (
        expected_sobolev + expected_gamma.pow(config.norm_order)
    )
    expected_loss = (
        expected_product + config.norm_epsilon**config.norm_order
    ).pow(1.0 / float(config.norm_order)) - config.norm_epsilon

    assert isinstance(result, CoverageStateUSCOPELossFields)
    assert torch.equal(result.violation_plus, pmope.violation_plus)
    assert torch.equal(result.violation_minus, pmope.violation_minus)
    assert torch.equal(result.per_state_sobolev_power, expected_sobolev)
    assert torch.equal(
        result.per_state_chebyshev_violation,
        expected_gamma,
    )
    assert torch.equal(result.per_state_product_power, expected_product)
    assert torch.equal(result.per_state_loss, expected_loss)
    assert result.loss.item() == pytest.approx(expected_loss.mean().item())


def test_uscope_reduces_the_supremum_per_state_then_means_the_batch() -> None:
    single = _pair_targets()
    targets = _repeat_targets(single, 2)
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)
    margin = config.field_amplitude / float(config.truncation_radius)
    field_plus = _orthant_field(
        targets.target_field_plus,
        2.0 * margin,
    )
    field_minus = _orthant_field(
        targets.target_field_minus,
        2.0 * margin,
    )
    field_plus[0, ..., 0, 0] = 0.0
    field_minus[1, ..., 0, 0] = 0.5 * margin

    result = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )

    assert result.per_state_chebyshev_violation.tolist() == pytest.approx(
        [margin, 0.5 * margin]
    )
    assert result.loss.item() == pytest.approx(
        result.per_state_loss.mean().item()
    )
    assert result.per_state_loss[0] > result.per_state_loss[1] > 0.0


def test_uscope_has_the_same_zero_set_as_pmope() -> None:
    targets = _pair_targets()
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)
    margin = config.field_amplitude / float(config.truncation_radius)
    field_plus = _orthant_field(targets.target_field_plus, 2.0 * margin)
    field_minus = _orthant_field(targets.target_field_minus, 3.0 * margin)

    result = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )
    pmope = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )

    assert result.loss.item() == 0.0
    assert pmope.loss.item() == 0.0
    assert result.per_state_chebyshev_violation.item() == 0.0

    violated = field_plus.clone()
    violated[..., 0, 0] = 0.0
    result = coverage_state_uscope_pair_loss_from_targets(
        violated,
        field_minus,
        targets,
        config=config,
    )
    pmope = coverage_state_pmope_pair_loss_from_targets(
        violated,
        field_minus,
        targets,
        config=config,
    )
    assert result.loss.item() > 0.0
    assert pmope.loss.item() > 0.0
    assert result.per_state_chebyshev_violation.item() == pytest.approx(
        margin
    )


def test_uscope_worst_background_pixel_has_the_correct_gradient() -> None:
    targets = _pair_targets()
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)
    margin = config.field_amplitude / float(config.truncation_radius)
    field_plus = _orthant_field(
        targets.target_field_plus,
        2.0 * margin,
    ).requires_grad_()
    field_minus = _orthant_field(
        targets.target_field_minus,
        2.0 * margin,
    ).requires_grad_()
    with torch.no_grad():
        field_plus[..., 0, 0] = 0.0

    result = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )
    gradient_plus, gradient_minus = torch.autograd.grad(
        result.loss,
        (field_plus, field_minus),
    )

    assert result.per_state_chebyshev_violation.item() == pytest.approx(
        margin
    )
    assert gradient_plus[..., 0, 0].item() < 0.0
    assert torch.isfinite(gradient_plus).all()
    assert torch.isfinite(gradient_minus).all()


def test_uscope_ignores_invalid_pixels_in_both_gauge_factors() -> None:
    targets = _pair_targets(invalid=((0, 0),))
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)
    margin = config.field_amplitude / float(config.truncation_radius)
    field_plus = _orthant_field(
        targets.target_field_plus,
        2.0 * margin,
    )
    field_minus = _orthant_field(
        targets.target_field_minus,
        2.0 * margin,
    )
    field_plus[..., 0, 0] = -100.0
    field_minus[..., 0, 0] = 100.0

    result = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )

    assert result.loss.item() == 0.0
    assert not bool(result.valid_mask[..., 0, 0])
    assert result.violation_plus[..., 0, 0].item() == 0.0
    assert result.violation_minus[..., 0, 0].item() == 0.0


def test_uscope_policy_and_singleton_training_registration() -> None:
    assert CSLF_USCOPE_POLICY == _EXPECTED_POLICY
    assert CoverageStatePairObjective.USCOPE_JOINT.value == "uscope_joint"
    assert (
        coverage_state_pair_objective_policy(
            CoverageStatePairObjective.USCOPE_JOINT
        )
        == CSLF_USCOPE_POLICY
    )
    assert COVERAGE_STATE_USCOPE_MATCHED_OBJECTIVES == (
        CoverageStatePairObjective.USCOPE_JOINT,
    )
    assert (
        COVERAGE_STATE_USCOPE_MATCHED_OBJECTIVES
        in COVERAGE_STATE_REGISTERED_MATCHED_OBJECTIVE_SUITES
    )
    assert callable(train_matched_coverage_state_cmif_uscope_objectives)


def test_uscope_singleton_uses_its_own_canonical_fairness_payload() -> None:
    torch.manual_seed(19)
    model = CURELiteCenteredMixedInteractionLevelSet(
        CoverageStateCenteredMixedInteractionConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    model_fingerprint = coverage_state_model_fingerprint(model)
    result = CoverageStateTrainingResult(
        objective=CoverageStatePairObjective.USCOPE_JOINT.value,
        objective_policy=CSLF_USCOPE_POLICY,
        seed=42,
        epochs=1,
        steps_per_epoch=1,
        completed_updates=1,
        schedule_fingerprint="1" * 64,
        cache_fingerprint="2" * 64,
        execution_device="cpu",
        device_cache_fingerprint="3" * 64,
        device_cache_resident_bytes=1,
        optimizer_config_fingerprint="4" * 64,
        initial_model_fingerprint=model_fingerprint,
        final_model_fingerprint=model_fingerprint,
        epoch_logs=({"epoch": 0},),
        first_nonzero_gradient_update=(),
        forward_calls=1,
        backward_calls=1,
        optimizer_steps=1,
        logical_state_evaluations=12,
        finite_state_audits=2,
    )
    matched = CoverageStateMatchedTrainingResult(
        config=CoverageStateMatchedTrainingConfig(seed=42),
        common_initial_model_fingerprint=model_fingerprint,
        schedule_fingerprint=result.schedule_fingerprint,
        cache_fingerprint=result.cache_fingerprint,
        results=(result,),
        models=((result.objective, model),),
    )

    payload = matched.canonical_payload()
    assert payload["objective_suite"] == ["uscope_joint"]
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
        "allowed_difference_from_sealed_v18": (
            "predeclared_pair_objective_only"
        ),
        "same_model_class": True,
        "same_model_config": True,
        "same_parameter_count": True,
        "same_parameter_shapes": True,
    }
    assert "response_identity_share_joint_measure" not in payload[
        "fairness"
    ]
