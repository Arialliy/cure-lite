from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import math

import pytest
import torch
from torch import Tensor

from cure_lite.coverage_state_sobolev import (
    CSLF_PMOPE_POLICY,
    CoverageStatePMOPELossFields,
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    _pair_energy,
    coverage_state_identity_joint_loss_from_targets,
    coverage_state_pair_sobolev_loss_from_targets,
    coverage_state_pmope_pair_loss_from_targets,
    coverage_state_support_oriented_pair_sobolev_loss_from_targets,
    prepare_coverage_state_pair_targets,
)


_RADIUS = 4
_EXPECTED_POLICY = (
    "paired_minimum_sdf_margin_target_orthant_projection_"
    "joint_w1p4_energy_v1"
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
    target_plus: Iterable[tuple[int, int]] = (),
    added_target: Iterable[tuple[int, int]] = ((4, 4),),
    occupancy_plus: Iterable[tuple[int, int]] | None = None,
    occupancy_minus: Iterable[tuple[int, int]] = (),
    invalid: Iterable[tuple[int, int]] = (),
) -> CoverageStatePairTargets:
    plus_coordinates = tuple(target_plus)
    added_coordinates = tuple(added_target)
    target_plus_mask = _mask(size, plus_coordinates)
    target_minus_mask = target_plus_mask | _mask(size, added_coordinates)
    occupancy_plus_mask = _mask(
        size,
        (
            added_coordinates
            if occupancy_plus is None
            else tuple(occupancy_plus)
        ),
    )
    valid_mask = torch.ones_like(target_plus_mask)
    for row, column in invalid:
        valid_mask[..., row, column] = False
    return prepare_coverage_state_pair_targets(
        occupancy_plus_mask,
        _mask(size, occupancy_minus),
        target_plus_mask,
        target_minus_mask,
        valid_mask,
        config=CoverageStateSobolevConfig(
            truncation_radius=_RADIUS,
        ),
    )


def _orthant_field(target_field: Tensor, magnitude: float) -> Tensor:
    return torch.sign(target_field) * magnitude


def test_policy_and_fixed_margin_are_frozen() -> None:
    targets = _pair_targets()
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)
    result = coverage_state_pmope_pair_loss_from_targets(
        targets.target_field_plus,
        targets.target_field_minus,
        targets,
        config=config,
    )

    assert CSLF_PMOPE_POLICY == _EXPECTED_POLICY
    assert isinstance(result, CoverageStatePMOPELossFields)
    assert result.margin.ndim == 0
    assert result.margin.item() == pytest.approx(0.9 / 4.0)
    assert torch.equal(result.valid_mask, targets.valid_mask)


def test_zero_loss_guarantees_full_valid_domain_sign_and_completion() -> None:
    targets = _pair_targets(
        target_plus=((2, 2),),
        added_target=((4, 4), (4, 5)),
    )
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)
    margin = config.field_amplitude / float(config.truncation_radius)
    field_plus = _orthant_field(targets.target_field_plus, margin)
    field_minus = _orthant_field(targets.target_field_minus, 2.0 * margin)

    result = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )
    valid = targets.valid_mask

    assert result.loss.item() == 0.0
    assert int(torch.count_nonzero(result.violation_plus)) == 0
    assert int(torch.count_nonzero(result.violation_minus)) == 0
    assert torch.equal(
        (field_plus < 0.0) & valid,
        (targets.target_field_plus < 0.0) & valid,
    )
    assert torch.equal(
        (field_minus < 0.0) & valid,
        (targets.target_field_minus < 0.0) & valid,
    )

    violating = field_minus.clone()
    violating[..., 0, 0] = 0.0
    violated = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        violating,
        targets,
        config=config,
    )
    assert bool(valid[..., 0, 0])
    assert violated.violation_minus[..., 0, 0].item() == pytest.approx(
        margin
    )
    assert violated.loss.item() > 0.0


def test_gradient_descent_pushes_inside_down_and_outside_up() -> None:
    targets = _pair_targets(
        target_plus=((2, 2),),
        added_target=((4, 4),),
    )
    field_plus = torch.zeros_like(
        targets.target_field_plus,
        requires_grad=True,
    )
    field_minus = torch.zeros_like(
        targets.target_field_minus,
        requires_grad=True,
    )
    result = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=CoverageStateSobolevConfig(
            truncation_radius=_RADIUS,
        ),
    )
    result.loss.backward()
    assert field_plus.grad is not None
    assert field_minus.grad is not None

    for gradient, target_field in (
        (field_plus.grad, targets.target_field_plus),
        (field_minus.grad, targets.target_field_minus),
    ):
        inside = targets.valid_mask & (target_field < 0.0)
        outside = targets.valid_mask & (target_field > 0.0)
        assert bool(torch.all(gradient[inside] > 0.0))
        assert bool(torch.all(gradient[outside] < 0.0))
        assert bool(torch.isfinite(gradient).all())


def test_null_target_states_receive_the_correct_uniform_direction() -> None:
    targets = _pair_targets(
        target_plus=(),
        added_target=(),
        occupancy_plus=((4, 4),),
        occupancy_minus=(),
    )
    assert bool(torch.all(targets.target_field_plus > 0.0))
    assert bool(torch.all(targets.target_field_minus > 0.0))
    field_plus = torch.zeros_like(
        targets.target_field_plus,
        requires_grad=True,
    )
    field_minus = torch.zeros_like(
        targets.target_field_minus,
        requires_grad=True,
    )
    result = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=CoverageStateSobolevConfig(
            truncation_radius=_RADIUS,
        ),
    )
    gradients = torch.autograd.grad(
        result.loss,
        (field_plus, field_minus),
    )

    assert result.loss.item() > 0.0
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
    assert all(bool(torch.all(value[targets.valid_mask] < 0.0)) for value in gradients)


def test_orthant_cone_is_not_identity_response_sorr_or_omco() -> None:
    targets = _pair_targets(
        target_plus=((2, 2),),
        added_target=((4, 4),),
    )
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)
    magnitude = 2.0 * config.field_amplitude
    field_plus = _orthant_field(targets.target_field_plus, magnitude)
    field_minus = _orthant_field(targets.target_field_minus, magnitude)

    pmope = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )
    identity = coverage_state_identity_joint_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )
    response = coverage_state_pair_sobolev_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )
    sorr = coverage_state_support_oriented_pair_sobolev_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )
    error_plus = field_plus - targets.target_field_plus
    error_minus = field_minus - targets.target_field_minus
    root_two = math.sqrt(2.0)
    omco_first = (error_plus + error_minus) / root_two
    omco_second = (error_minus - error_plus) / root_two
    omco_loss = _pair_energy(
        (omco_first, omco_second),
        targets,
        config=config,
    )[0]

    assert pmope.loss.item() == 0.0
    assert identity.loss.item() > 0.0
    assert response.loss.item() > 0.0
    assert sorr.loss.item() > 0.0
    assert omco_loss.item() > 0.0


def test_invalid_pixels_are_zero_and_valid_forward_backward_are_finite() -> None:
    invalid = tuple((row, 4) for row in range(11))
    targets = _pair_targets(
        size=11,
        target_plus=((3, 2),),
        added_target=((7, 2), (7, 7)),
        invalid=invalid,
    )
    generator = torch.Generator().manual_seed(1801)
    field_plus = torch.randn(
        targets.target_field_plus.shape,
        generator=generator,
        dtype=torch.float32,
    ).requires_grad_()
    field_minus = torch.randn(
        targets.target_field_minus.shape,
        generator=generator,
        dtype=torch.float32,
    ).requires_grad_()
    result = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=CoverageStateSobolevConfig(
            truncation_radius=_RADIUS,
        ),
    )
    gradients = torch.autograd.grad(
        result.loss,
        (field_plus, field_minus),
    )

    assert not bool(torch.any(result.violation_plus[~targets.valid_mask]))
    assert not bool(torch.any(result.violation_minus[~targets.valid_mask]))
    for value in (
        result.loss,
        result.value_power,
        result.spatial_power,
        result.per_state_loss,
        result.per_state_value_power,
        result.per_state_spatial_power,
        result.violation_plus,
        result.violation_minus,
        *gradients,
    ):
        assert bool(torch.isfinite(value).all())


def test_validation_rejects_zero_target_field_and_nonfinite_prediction() -> None:
    targets = _pair_targets()
    zero_target = targets.target_field_plus.clone()
    zero_target[..., 0, 0] = 0.0
    invalid_targets = replace(targets, target_field_plus=zero_target)
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)

    with pytest.raises(ValueError, match="strictly nonzero"):
        coverage_state_pmope_pair_loss_from_targets(
            targets.target_field_plus,
            targets.target_field_minus,
            invalid_targets,
            config=config,
        )

    nonfinite = targets.target_field_plus.clone()
    nonfinite[..., 0, 0] = float("nan")
    with pytest.raises(ValueError, match="must align"):
        coverage_state_pmope_pair_loss_from_targets(
            nonfinite,
            targets.target_field_minus,
            targets,
            config=config,
        )
