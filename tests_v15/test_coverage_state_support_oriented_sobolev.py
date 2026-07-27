from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from cure_lite.coverage_state_level_set import CSLF_FIELD_AMPLITUDE
from cure_lite.coverage_state_sobolev import (
    CoverageStatePairLossFields,
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    coverage_state_added_target_support_from_targets,
    coverage_state_pair_sobolev_loss_from_targets,
    coverage_state_support_oriented_pair_sobolev_loss_from_targets,
    prepare_coverage_state_pair_targets,
)


_RADIUS = 4


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
) -> tuple[CoverageStatePairTargets, Tensor, Tensor, Tensor]:
    plus_coordinates = tuple(target_plus)
    added_coordinates = tuple(added_target)
    target_plus_mask = _mask(size, plus_coordinates)
    added_mask = _mask(size, added_coordinates)
    target_minus_mask = target_plus_mask | added_mask
    occupancy_plus_mask = _mask(
        size,
        (
            added_coordinates
            if occupancy_plus is None
            else tuple(occupancy_plus)
        ),
    )
    occupancy_minus_mask = _mask(size, occupancy_minus)
    valid_mask = torch.ones_like(target_plus_mask)
    for row, column in invalid:
        valid_mask[..., row, column] = False
    targets = prepare_coverage_state_pair_targets(
        occupancy_plus_mask,
        occupancy_minus_mask,
        target_plus_mask,
        target_minus_mask,
        valid_mask,
        config=CoverageStateSobolevConfig(
            truncation_radius=_RADIUS,
        ),
    )
    return targets, target_plus_mask, target_minus_mask, valid_mask


def _nonuniform_fields(
    targets: CoverageStatePairTargets,
) -> tuple[Tensor, Tensor]:
    count = targets.target_field_plus.numel()
    ramp = torch.arange(
        count,
        dtype=torch.float32,
    ).reshape_as(targets.target_field_plus) / float(count)
    field_plus = targets.target_field_plus + 0.125 * ramp
    field_minus = (
        targets.target_field_minus
        - 0.0625 * torch.flip(ramp, dims=(-1,))
    )
    return field_plus.contiguous(), field_minus.contiguous()


def _assert_pair_results_bitwise_equal(
    actual: CoverageStatePairLossFields,
    expected: CoverageStatePairLossFields,
) -> None:
    names = (
        "loss",
        "value_power",
        "spatial_power",
        "per_state_loss",
        "per_state_value_power",
        "per_state_spatial_power",
        "target_field_plus",
        "target_field_minus",
        "predicted_coverage_response",
        "target_coverage_response",
        "anchor_error",
        "response_error",
        "focus_support",
        "focus_support_field",
        "integration_measure",
    )
    for name in names:
        assert torch.equal(
            getattr(actual, name),
            getattr(expected, name),
        ), name


def test_selector_recovers_exact_added_target_support_with_invalid_barrier(
) -> None:
    invalid = tuple((row, 4) for row in range(9))
    targets, target_plus, target_minus, valid = _pair_targets(
        target_plus=((2, 2),),
        added_target=((5, 2), (5, 6)),
        invalid=invalid,
    )
    selector = coverage_state_added_target_support_from_targets(targets)
    expected = target_minus & ~target_plus

    assert selector.dtype == torch.bool
    assert not selector.requires_grad
    assert torch.equal(selector, expected)
    assert int(torch.count_nonzero(selector)) == 2
    assert not bool(torch.any(selector & ~valid))
    assert not bool(torch.any(selector[..., :, 4]))


def test_selector_rejects_a_reversed_target_transition() -> None:
    targets, _, _, _ = _pair_targets()
    reversed_targets = replace(
        targets,
        target_field_plus=targets.target_field_minus,
        target_field_minus=targets.target_field_plus,
    )
    with pytest.raises(ValueError, match="must not remove target support"):
        coverage_state_added_target_support_from_targets(reversed_targets)


def test_partitioned_root_response_and_coordinate_inverse_are_exact() -> None:
    targets, _, _, _ = _pair_targets(
        target_plus=((2, 2),),
        added_target=((4, 4), (4, 5)),
    )
    selector = coverage_state_added_target_support_from_targets(targets)
    error_plus = torch.full_like(targets.target_field_plus, 0.25)
    error_minus = torch.full_like(targets.target_field_minus, -0.5)
    error_minus = error_minus.masked_fill(selector, 0.5)
    field_plus = targets.target_field_plus + error_plus
    field_minus = targets.target_field_minus + error_minus

    support_oriented = (
        coverage_state_support_oriented_pair_sobolev_loss_from_targets(
            field_plus,
            field_minus,
            targets,
            config=CoverageStateSobolevConfig(
                truncation_radius=_RADIUS,
            ),
        )
    )
    legacy = coverage_state_pair_sobolev_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=CoverageStateSobolevConfig(
            truncation_radius=_RADIUS,
        ),
    )

    assert torch.equal(
        support_oriented.anchor_error[selector],
        error_minus[selector],
    )
    assert torch.equal(
        support_oriented.anchor_error[~selector],
        error_plus[~selector],
    )
    assert torch.equal(
        support_oriented.predicted_coverage_response,
        legacy.predicted_coverage_response,
    )
    assert torch.equal(
        support_oriented.target_coverage_response,
        legacy.target_coverage_response,
    )
    assert torch.equal(
        support_oriented.response_error,
        legacy.response_error,
    )

    delta_error = error_minus - error_plus
    reconstructed_plus = torch.where(
        selector,
        support_oriented.anchor_error - delta_error,
        support_oriented.anchor_error,
    )
    reconstructed_minus = torch.where(
        selector,
        support_oriented.anchor_error,
        support_oriented.anchor_error + delta_error,
    )
    assert torch.equal(reconstructed_plus, error_plus)
    assert torch.equal(reconstructed_minus, error_minus)

    plus_leaf = error_plus.clone().requires_grad_()
    minus_leaf = error_minus.clone().requires_grad_()
    root = torch.where(selector, minus_leaf, plus_leaf)
    response = minus_leaf - plus_leaf
    root_plus, root_minus = torch.autograd.grad(
        root.sum(),
        (plus_leaf, minus_leaf),
        retain_graph=True,
    )
    response_plus, response_minus = torch.autograd.grad(
        response.sum(),
        (plus_leaf, minus_leaf),
    )
    assert torch.equal(root_plus, (~selector).to(torch.float32))
    assert torch.equal(root_minus, selector.to(torch.float32))
    assert torch.equal(response_plus, -torch.ones_like(response_plus))
    assert torch.equal(response_minus, torch.ones_like(response_minus))


def test_support_oriented_root_has_direct_added_target_gradient() -> None:
    size = 17
    targets, _, _, valid = _pair_targets(
        size=size,
        target_plus=(),
        added_target=((size // 2, size // 2),),
    )
    selector = coverage_state_added_target_support_from_targets(targets)
    offset = 0.001 + CSLF_FIELD_AMPLITUDE / float(_RADIUS)
    field_plus = (
        targets.target_field_plus + offset
    ).detach().requires_grad_()
    field_minus = (
        targets.target_field_minus + offset
    ).detach().requires_grad_()

    result = coverage_state_support_oriented_pair_sobolev_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=CoverageStateSobolevConfig(
            truncation_radius=_RADIUS,
        ),
    )
    result.loss.backward()

    assert bool(torch.all(field_minus.detach()[selector] > 0.0))
    assert (
        result.response_error.detach().abs().max().item()
        <= 1.0e-6
    )
    assert field_plus.grad is not None
    assert field_minus.grad is not None
    assert bool(torch.isfinite(field_plus.grad).all())
    assert bool(torch.isfinite(field_minus.grad).all())
    assert float(field_minus.grad[selector].min()) >= 1.0e-4
    exterior_plus_gradient = field_plus.grad[(~selector) & valid]
    assert int(torch.count_nonzero(exterior_plus_gradient)) > 0


@pytest.mark.parametrize(
    ("target", "occupancy_plus", "occupancy_minus"),
    (
        (((2, 2),), (), ()),
        ((), ((4, 4),), ()),
    ),
    ids=("identity_null", "component_null"),
)
def test_null_pairs_reduce_bitwise_to_legacy_response_joint(
    target: tuple[tuple[int, int], ...],
    occupancy_plus: tuple[tuple[int, int], ...],
    occupancy_minus: tuple[tuple[int, int], ...],
) -> None:
    targets, _, _, _ = _pair_targets(
        target_plus=target,
        added_target=(),
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
    )
    assert not bool(
        coverage_state_added_target_support_from_targets(targets).any()
    )
    field_plus, field_minus = _nonuniform_fields(targets)
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)

    support_plus = field_plus.detach().clone().requires_grad_()
    support_minus = field_minus.detach().clone().requires_grad_()
    support_oriented = (
        coverage_state_support_oriented_pair_sobolev_loss_from_targets(
            support_plus,
            support_minus,
            targets,
            config=config,
        )
    )
    support_gradients = torch.autograd.grad(
        support_oriented.loss,
        (support_plus, support_minus),
    )

    legacy_plus = field_plus.detach().clone().requires_grad_()
    legacy_minus = field_minus.detach().clone().requires_grad_()
    legacy = coverage_state_pair_sobolev_loss_from_targets(
        legacy_plus,
        legacy_minus,
        targets,
        config=config,
    )
    legacy_gradients = torch.autograd.grad(
        legacy.loss,
        (legacy_plus, legacy_minus),
    )

    _assert_pair_results_bitwise_equal(support_oriented, legacy)
    assert torch.equal(support_gradients[0], legacy_gradients[0])
    assert torch.equal(support_gradients[1], legacy_gradients[1])


def test_exact_endpoint_fixed_point_has_zero_loss_and_gradient() -> None:
    targets, _, _, _ = _pair_targets(
        target_plus=((2, 2),),
        added_target=((4, 4),),
    )
    field_plus = (
        targets.target_field_plus.detach().clone().requires_grad_()
    )
    field_minus = (
        targets.target_field_minus.detach().clone().requires_grad_()
    )
    result = coverage_state_support_oriented_pair_sobolev_loss_from_targets(
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

    assert result.loss.detach().item() == 0.0
    assert int(torch.count_nonzero(result.anchor_error)) == 0
    assert int(torch.count_nonzero(result.response_error)) == 0
    assert int(torch.count_nonzero(gradients[0])) == 0
    assert int(torch.count_nonzero(gradients[1])) == 0
    assert bool(torch.isfinite(gradients[0]).all())
    assert bool(torch.isfinite(gradients[1]).all())


def test_boundary_is_finite_but_not_legacy_energy_equivalent() -> None:
    targets, _, _, _ = _pair_targets(
        target_plus=(),
        added_target=((4, 4),),
    )
    selector = coverage_state_added_target_support_from_targets(targets)
    error_plus = torch.zeros_like(targets.target_field_plus)
    error_minus = torch.zeros_like(targets.target_field_minus)
    error_minus[..., 4, 4] = 1.0
    field_plus = (
        targets.target_field_plus + error_plus
    ).detach().requires_grad_()
    field_minus = (
        targets.target_field_minus + error_minus
    ).detach().requires_grad_()
    config = CoverageStateSobolevConfig(truncation_radius=_RADIUS)

    support_oriented = (
        coverage_state_support_oriented_pair_sobolev_loss_from_targets(
            field_plus,
            field_minus,
            targets,
            config=config,
        )
    )
    legacy = coverage_state_pair_sobolev_loss_from_targets(
        field_plus.detach(),
        field_minus.detach(),
        targets,
        config=config,
    )
    gradients = torch.autograd.grad(
        support_oriented.loss,
        (field_plus, field_minus),
    )

    inside = (4, 4)
    outside = (4, 5)
    assert bool(selector[..., inside[0], inside[1]])
    assert not bool(selector[..., outside[0], outside[1]])
    assert support_oriented.anchor_error[
        ..., outside[0], outside[1]
    ].item() == error_plus[..., outside[0], outside[1]].item()
    support_boundary_difference = (
        support_oriented.anchor_error[..., outside[0], outside[1]]
        - support_oriented.anchor_error[..., inside[0], inside[1]]
    )
    legacy_boundary_difference = (
        legacy.anchor_error[..., outside[0], outside[1]]
        - legacy.anchor_error[..., inside[0], inside[1]]
    )
    assert support_boundary_difference.item() == -1.0
    assert legacy_boundary_difference.item() == 0.0
    assert not torch.equal(
        support_oriented.spatial_power,
        legacy.spatial_power,
    )
    assert not torch.equal(support_oriented.loss, legacy.loss)
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
    assert all(int(torch.count_nonzero(value)) > 0 for value in gradients)


def test_invalid_barrier_keeps_selector_loss_and_gradients_valid() -> None:
    invalid = tuple((row, 4) for row in range(11))
    targets, target_plus, target_minus, valid = _pair_targets(
        size=11,
        target_plus=((3, 2),),
        added_target=((7, 2), (7, 7)),
        invalid=invalid,
    )
    selector = coverage_state_added_target_support_from_targets(targets)
    field_plus, field_minus = _nonuniform_fields(targets)
    field_plus = field_plus.detach().requires_grad_()
    field_minus = field_minus.detach().requires_grad_()
    result = coverage_state_support_oriented_pair_sobolev_loss_from_targets(
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

    assert torch.equal(selector, target_minus & ~target_plus)
    assert not bool(torch.any(selector & ~valid))
    assert not bool(torch.any(selector[..., :, 4]))
    assert bool(torch.isfinite(result.loss))
    assert bool(torch.isfinite(result.value_power))
    assert bool(torch.isfinite(result.spatial_power))
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
