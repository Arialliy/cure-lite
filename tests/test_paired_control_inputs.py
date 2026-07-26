from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.decoder import project_occupancy_to_feature_grid
from cure_lite.paired_control_inputs import (
    TARGET_PERMUTATION_INCONCLUSIVE,
    TARGET_PERMUTATION_READY,
    build_dct_coordinate_basis,
    build_target_permutation,
    capacity_active_dct_feature_like,
    feature_only_zero_occupancy,
    materialize_permuted_label_increments,
    nominal_zero_feature_like,
    target_permutation_compatible,
)
from cure_lite.paired_types import PairExample, tensor_content_fingerprint
from cure_lite.sampling import stable_hash


def _clean_pair(
    *,
    sample_id: str,
    target_id: int,
    target_pixels: tuple[tuple[int, int], ...],
    valid_pixels: tuple[tuple[int, int], ...] | None = None,
    size: int = 4,
) -> PairExample:
    feature = torch.full(
        (1, 2, 2, 2),
        float(target_id),
        dtype=torch.float32,
    )
    valid = torch.zeros((1, size, size), dtype=torch.bool)
    if valid_pixels is None:
        valid.fill_(True)
    else:
        for row, column in valid_pixels:
            valid[0, row, column] = True
    plus = torch.zeros_like(valid)
    removed_row, removed_column = target_pixels[0]
    plus[0, removed_row, removed_column] = True
    minus = torch.zeros_like(plus)
    clean = torch.zeros_like(valid)
    for row, column in target_pixels:
        clean[0, row, column] = True
    if torch.any(clean & ~valid):
        raise ValueError("test target must lie inside its own valid mask")
    completion_plus = torch.zeros_like(valid)
    completion_minus = clean.clone()
    projected_plus = project_occupancy_to_feature_grid(
        plus.unsqueeze(0),
        (2, 2),
    )
    projected_minus = project_occupancy_to_feature_grid(
        minus.unsqueeze(0),
        (2, 2),
    )
    pair_id = stable_fingerprint(
        {
            "sample_id": sample_id,
            "target_id": target_id,
            "target_pixels": target_pixels,
            "valid_pixels": valid_pixels,
            "size": size,
        }
    )
    return PairExample(
        pair_id=pair_id,
        pair_kind="clean_positive",
        sample_id=sample_id,
        group_id=f"group-{sample_id}",
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        removed_component=plus.clone(),
        image_valid_mask=valid,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        label_increment=clean.to(torch.float32),
        clean_increment=clean,
        evaluation_gt_id=target_id,
        native_gt_id=target_id,
        pred_id=target_id,
        feature_fingerprint=tensor_content_fingerprint(feature),
        before_match_fingerprint=stable_fingerprint(
            {"sample_id": sample_id, "target_id": target_id, "state": "plus"}
        ),
        after_match_fingerprint=stable_fingerprint(
            {"sample_id": sample_id, "target_id": target_id, "state": "minus"}
        ),
        projected_occupancy_plus_fingerprint=tensor_content_fingerprint(
            projected_plus
        ),
        projected_occupancy_minus_fingerprint=tensor_content_fingerprint(
            projected_minus
        ),
        projection_visible=True,
        geometry_safe_bijective_lineage=True,
        selected_gt_is_only_new_unmatched=True,
        other_match_identities_unchanged=True,
        preexisting_unmatched_gt_noninterference=True,
    )


def test_nominal_zero_feature_preserves_only_shape_dtype_and_device() -> None:
    first = torch.randn((2, 3, 4, 5), dtype=torch.float32, requires_grad=True)
    second = torch.randn((2, 3, 4, 5), dtype=torch.float32)
    first_zero = nominal_zero_feature_like(first)
    second_zero = nominal_zero_feature_like(second)

    assert first_zero.shape == first.shape
    assert first_zero.dtype == first.dtype
    assert first_zero.device == first.device
    assert not first_zero.requires_grad
    assert torch.count_nonzero(first_zero) == 0
    torch.testing.assert_close(first_zero, second_zero)


def test_dct_basis_is_source_independent_ordered_and_fingerprinted() -> None:
    first = torch.randn((2, 5, 3, 4), dtype=torch.float32)
    second = torch.randn((2, 5, 3, 4), dtype=torch.float32) + 1000.0
    first_control, first_basis = capacity_active_dct_feature_like(first)
    second_control, second_basis = capacity_active_dct_feature_like(second)

    assert tuple(first_control.shape) == (2, 5, 3, 4)
    assert first_control.dtype == torch.float32
    assert first_basis.modes == ((0, 1), (1, 0), (0, 2), (1, 1), (2, 0))
    assert (0, 0) not in first_basis.modes
    assert first_basis.tensor_fingerprint == second_basis.tensor_fingerprint
    assert first_basis.basis_fingerprint == second_basis.basis_fingerprint
    assert first_basis.canonical_payload["source_independent"] is True
    torch.testing.assert_close(first_control, second_control)
    torch.testing.assert_close(first_control[0], first_control[1])

    values = first_basis.tensor.to(torch.float64)
    means = values.mean(dim=(-2, -1))
    rms = torch.sqrt(values.square().mean(dim=(-2, -1)))
    torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-6, rtol=0)
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-6, rtol=0)
    assert torch.all(first_basis.tensor.flatten(2).abs().sum(dim=2) > 0)


def test_dct_basis_keeps_every_feature_project_channel_gradient_activatable() -> None:
    basis = build_dct_coordinate_basis(
        channels=6,
        height=3,
        width=3,
        dtype=torch.float32,
    )
    control = basis.expand(1).requires_grad_(True)
    projection = torch.nn.Conv2d(6, 6, kernel_size=1, bias=False)
    output = projection(control)
    loss = (output * control.detach()).sum()
    loss.backward()

    assert projection.weight.grad is not None
    per_input_channel = projection.weight.grad.abs().sum(dim=(0, 2, 3))
    assert torch.all(per_input_channel > 0)
    assert control.grad is not None
    assert torch.isfinite(control.grad).all()


def test_dct_basis_rejects_more_channels_than_non_dc_modes() -> None:
    with pytest.raises(ValueError, match=r"C <= h\*w-1"):
        build_dct_coordinate_basis(channels=4, height=2, width=2)


def test_feature_only_control_uses_one_fixed_zero_occupancy() -> None:
    plus = torch.ones((2, 1, 5, 6), dtype=torch.bool)
    minus = torch.zeros_like(plus)
    zero_plus, zero_minus = feature_only_zero_occupancy(plus, minus)

    assert zero_plus is zero_minus
    assert zero_plus.shape == plus.shape
    assert zero_plus.dtype == torch.bool
    assert torch.count_nonzero(zero_plus) == 0


def test_target_permutation_is_deterministic_source_disjoint_and_fixed_point_free() -> None:
    pairs = (
        _clean_pair(sample_id="source-a", target_id=1, target_pixels=((0, 0),)),
        _clean_pair(sample_id="source-b", target_id=2, target_pixels=((1, 1),)),
        _clean_pair(sample_id="source-c", target_id=3, target_pixels=((2, 2),)),
    )
    plan = build_target_permutation(pairs)
    replay = build_target_permutation(tuple(reversed(pairs)))

    assert plan.status == TARGET_PERMUTATION_READY
    assert plan.ready
    assert plan.plan_fingerprint == replay.plan_fingerprint
    assert plan.canonical_pair_ids == replay.canonical_pair_ids
    assert len(plan.assignments) == len(pairs)
    assert {
        assignment.donor_pair_id for assignment in plan.assignments
    } == set(plan.canonical_pair_ids)
    assert all(
        assignment.recipient_pair_id != assignment.donor_pair_id
        and assignment.recipient_sample_id != assignment.donor_sample_id
        for assignment in plan.assignments
    )

    donor_indices = {
        pair_id: index
        for index, pair_id in enumerate(plan.canonical_pair_ids)
    }
    actual = tuple(
        donor_indices[assignment.donor_pair_id]
        for assignment in plan.assignments
    )
    assert actual == (1, 2, 0)

    targets = materialize_permuted_label_increments(pairs, plan)
    assert len(targets) == len(pairs)
    assert all(target.dtype == torch.float32 for target in targets)


def test_target_permutation_compatibility_checks_shape_support_and_identity() -> None:
    recipient = _clean_pair(
        sample_id="recipient",
        target_id=1,
        target_pixels=((0, 0),),
        valid_pixels=((0, 0), (0, 1)),
    )
    compatible = _clean_pair(
        sample_id="donor-good",
        target_id=2,
        target_pixels=((0, 1),),
    )
    outside = _clean_pair(
        sample_id="donor-outside",
        target_id=3,
        target_pixels=((3, 3),),
    )
    wrong_shape = _clean_pair(
        sample_id="donor-shape",
        target_id=4,
        target_pixels=((0, 1),),
        size=5,
    )

    assert target_permutation_compatible(recipient, compatible)
    assert not target_permutation_compatible(recipient, recipient)
    assert not target_permutation_compatible(
        recipient,
        replace(compatible, sample_id=recipient.sample_id),
    )
    assert not target_permutation_compatible(recipient, outside)
    assert not target_permutation_compatible(recipient, wrong_shape)


def test_no_compatible_perfect_matching_is_explicitly_inconclusive() -> None:
    pairs = (
        _clean_pair(sample_id="source-a", target_id=1, target_pixels=((0, 0),)),
        _clean_pair(sample_id="source-a", target_id=2, target_pixels=((1, 1),)),
        _clean_pair(sample_id="source-b", target_id=3, target_pixels=((2, 2),)),
    )
    plan = build_target_permutation(pairs)

    assert plan.status == TARGET_PERMUTATION_INCONCLUSIVE
    assert not plan.ready
    assert plan.reason_code == "no_compatible_perfect_matching"
    assert plan.assignments == ()
    assert plan.canonical_pair_ids == tuple(
        pair.pair_id
        for pair in sorted(
            pairs,
            key=lambda pair: (
                stable_hash(
                    "target-permutation-canonical-v1",
                    pair.pair_id,
                ),
                pair.pair_id,
            ),
        )
    )
    with pytest.raises(RuntimeError, match="COMPUTATIONALLY_INCONCLUSIVE"):
        materialize_permuted_label_increments(pairs, plan)
