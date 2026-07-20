from __future__ import annotations

import pytest
import torch

from cure_lite.instances import instances_from_binary_mask, union_instance_masks
from cure_lite.matching import match_components
from cure_lite.supervision import (
    build_factual_supervision,
    factual_oracle_reachable,
    full_gt_recoverable,
)
from cure_lite.types import BranchSupervision


def _jointly_unreachable_scene():
    # Both anchor misses are independently recoverable.  Adding both full GT
    # masks changes CC8/matching and leaves GT 4 unmatched, so their union is
    # not a valid factual training target.
    gt_mask = torch.tensor(
        [
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    occupancy = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    gt = instances_from_binary_mask(gt_mask)
    match = match_components(instances_from_binary_mask(occupancy), gt)
    return occupancy, gt, match


def test_factual_state_selects_one_individually_reachable_miss() -> None:
    occupancy, gt, match = _jointly_unreachable_scene()
    assert match.unmatched_gt_ids == frozenset({1, 4})
    assert full_gt_recoverable(occupancy, gt, 1, match)
    assert full_gt_recoverable(occupancy, gt, 4, match)
    assert factual_oracle_reachable(occupancy, gt, 1, match)

    joint = match_components(
        instances_from_binary_mask(
            occupancy | union_instance_masks(gt, match.unmatched_gt_ids)
        ),
        gt,
    )
    assert not match.unmatched_gt_ids <= joint.matched_gt_ids

    first = build_factual_supervision(occupancy, gt, match)
    repeated = build_factual_supervision(occupancy, gt, match)
    assert first.positive_gt_ids == (1,)
    assert first.reachable_gt_ids == (1, 4)
    assert torch.equal(first.target, repeated.target)
    assert torch.equal(
        first.target[0].to(torch.bool),
        gt.by_id(1).mask & ~occupancy,
    )

    selected = build_factual_supervision(
        occupancy, gt, match, selected_gt_id=4
    )
    assert selected.positive_gt_ids == (4,)
    assert selected.reachable_gt_ids == (1, 4)
    assert torch.equal(
        selected.target[0].to(torch.bool),
        gt.by_id(4).mask & ~occupancy,
    )


def test_factual_selection_rejects_non_reachable_id() -> None:
    occupancy, gt, match = _jointly_unreachable_scene()
    with pytest.raises(ValueError, match="individually reachable"):
        build_factual_supervision(occupancy, gt, match, selected_gt_id=2)


@pytest.mark.parametrize(
    ("branch", "target_value", "valid_value", "kwargs", "message"),
    [
        ("factual_miss", 0.0, True, {}, "non-empty positive target"),
        ("factual_no_miss", 1.0, True, {}, "empty target"),
        (
            "factual_unreachable",
            0.0,
            True,
            {"unreachable_gt_ids": (1,)},
            "diagnostics-only",
        ),
        ("synthetic", 0.0, True, {}, "non-empty target"),
    ],
)
def test_branch_specific_target_semantics_are_enforced(
    branch: str,
    target_value: float,
    valid_value: bool,
    kwargs: dict[str, tuple[int, ...]],
    message: str,
) -> None:
    occupancy = torch.zeros(1, 3, 3, dtype=torch.bool)
    target = torch.zeros(1, 3, 3)
    target[0, 1, 1] = target_value
    valid = torch.full_like(occupancy, valid_value)
    with pytest.raises(ValueError, match=message):
        BranchSupervision(occupancy, target, valid, branch, **kwargs)


def test_factual_unreachable_diagnostics_only_shape_is_valid() -> None:
    occupancy = torch.zeros(1, 3, 3, dtype=torch.bool)
    BranchSupervision(
        occupancy,
        torch.zeros(1, 3, 3),
        torch.zeros_like(occupancy),
        "factual_unreachable",
        unreachable_gt_ids=(2,),
    )


@pytest.mark.parametrize("branch", ["factual_miss", "synthetic"])
def test_positive_branches_keep_legacy_optional_metadata(branch: str) -> None:
    occupancy = torch.zeros(1, 3, 3, dtype=torch.bool)
    target = torch.zeros(1, 3, 3)
    target[0, 1, 1] = 1.0
    BranchSupervision(
        occupancy,
        target,
        torch.ones_like(occupancy),
        branch,
    )


def test_synthetic_rejects_multiple_positive_gt_ids() -> None:
    occupancy = torch.zeros(1, 3, 3, dtype=torch.bool)
    target = torch.zeros(1, 3, 3)
    target[0, 1, 1] = 1.0
    with pytest.raises(ValueError, match="exactly one"):
        BranchSupervision(
            occupancy,
            target,
            torch.ones_like(occupancy),
            "synthetic",
            positive_gt_ids=(1, 2),
        )


def test_authoritative_reachable_catalog_requires_one_selected_member() -> None:
    occupancy = torch.zeros(1, 3, 3, dtype=torch.bool)
    target = torch.zeros(1, 3, 3)
    target[0, 1, 1] = 1.0
    valid = torch.ones_like(occupancy)
    with pytest.raises(ValueError, match="exactly one selected"):
        BranchSupervision(
            occupancy,
            target,
            valid,
            "factual_miss",
            reachable_gt_ids=(1, 2),
        )
    with pytest.raises(ValueError, match="belong to reachable"):
        BranchSupervision(
            occupancy,
            target,
            valid,
            "factual_miss",
            positive_gt_ids=(3,),
            reachable_gt_ids=(1, 2),
        )
