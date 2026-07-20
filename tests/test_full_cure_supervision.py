from __future__ import annotations

import pytest
import torch

from cure_lite.cure import (
    CounterfactualBackgroundPolicy,
    CounterfactualTargetPolicy,
    build_counterfactual_residual_set_supervision,
    build_factual_residual_set_supervision,
)
from cure_lite.instances import instances_from_binary_mask
from cure_lite.intervention import enumerate_legal_deletions
from cure_lite.matching import match_components
from cure_lite.types import LegalDeletion


def _three_target_state():
    gt_mask = torch.zeros(9, 9, dtype=torch.bool)
    gt_mask[1, 1] = True
    gt_mask[4, 4] = True
    gt_mask[7, 7] = True
    occupancy = torch.zeros_like(gt_mask)
    occupancy[1, 1] = True
    occupancy[7, 7] = True
    gt = instances_from_binary_mask(gt_mask)
    pred = instances_from_binary_mask(occupancy)
    before = match_components(pred, gt)
    deletions = enumerate_legal_deletions(pred, gt, before, occupancy)
    return occupancy, gt, before, deletions


def test_factual_supervision_contains_all_editable_misses() -> None:
    occupancy, gt, before, _ = _three_target_state()
    assert before.unmatched_gt_ids == frozenset({2})
    state = build_factual_residual_set_supervision(
        occupancy, gt, before, suppression_radius=0
    )
    assert state.branch == "factual"
    assert state.positive_gt_ids == (2,)
    assert state.uneditable_gt_ids == ()
    assert state.object_masks.shape == (1, 9, 9)
    assert torch.equal(state.object_masks[0], gt.by_id(2).mask)


def test_counterfactual_state_supervises_only_the_atomic_uncensored_target() -> None:
    occupancy, gt, before, deletions = _three_target_state()
    deletion = next(item for item in deletions if item.gt_id == 1)
    state = build_counterfactual_residual_set_supervision(
        deletion, gt, before, occupancy, suppression_radius=0
    )
    assert state.branch == "counterfactual"
    assert state.positive_gt_ids == (1,)
    assert state.ignored_gt_ids == (2,)
    assert state.object_masks.shape[0] == 1
    assert torch.equal(state.target.to(torch.bool)[0], gt.by_id(1).mask)
    assert not torch.any(state.background_mask)


def test_all_uncovered_counterfactual_targets_remain_an_explicit_ablation() -> None:
    occupancy, gt, before, deletions = _three_target_state()
    deletion = next(item for item in deletions if item.gt_id == 1)
    state = build_counterfactual_residual_set_supervision(
        deletion,
        gt,
        before,
        occupancy,
        suppression_radius=0,
        target_policy=CounterfactualTargetPolicy.ALL_UNCOVERED_TARGETS,
    )
    assert state.positive_gt_ids == (1, 2)
    assert state.ignored_gt_ids == ()
    assert torch.equal(
        state.target.to(torch.bool)[0],
        gt.by_id(1).mask | gt.by_id(2).mask,
    )


def test_counterfactual_background_bce_remains_an_explicit_ablation() -> None:
    occupancy, gt, before, deletions = _three_target_state()
    deletion = next(item for item in deletions if item.gt_id == 1)
    default = build_counterfactual_residual_set_supervision(
        deletion, gt, before, occupancy, suppression_radius=0
    )
    ablation = build_counterfactual_residual_set_supervision(
        deletion,
        gt,
        before,
        occupancy,
        suppression_radius=0,
        background_policy=CounterfactualBackgroundPolicy.BCE,
    )
    assert not torch.any(default.background_mask)
    assert torch.any(ablation.background_mask)
    gt_union = torch.stack([item.mask for item in gt.instances]).any(dim=0)
    assert torch.equal(
        ablation.background_mask[0],
        ablation.editable_mask[0] & ~gt_union,
    )


def test_counterfactual_builder_rejects_forged_before_state() -> None:
    occupancy, gt, before, deletions = _three_target_state()
    deletion = next(item for item in deletions if item.gt_id == 1)
    with pytest.raises(ValueError, match="inconsistent with occupancy_before"):
        build_counterfactual_residual_set_supervision(
            deletion,
            gt,
            deletion.match_after,
            occupancy,
            suppression_radius=0,
        )


def test_dilation_reports_uneditable_factual_miss() -> None:
    gt_mask = torch.zeros(7, 7, dtype=torch.bool)
    gt_mask[2, 2] = True
    gt_mask[2, 4] = True
    occupancy = torch.zeros_like(gt_mask)
    occupancy[2, 2] = True
    gt = instances_from_binary_mask(gt_mask)
    match = match_components(instances_from_binary_mask(occupancy), gt)
    assert match.unmatched_gt_ids == frozenset({2})
    state = build_factual_residual_set_supervision(
        occupancy, gt, match, suppression_radius=2
    )
    assert state.positive_gt_ids == ()
    assert state.uneditable_gt_ids == (2,)
    assert state.object_masks.shape[0] == 0


def test_counterfactual_builder_rejects_full_gt_bridge_failure() -> None:
    gt_mask = torch.zeros(7, 7, dtype=torch.bool)
    gt_mask[1, 1:3] = True
    gt_mask[1, 5] = True
    occupancy = torch.zeros_like(gt_mask)
    occupancy[1, 1] = True
    occupancy[1, 3:6] = True
    gt = instances_from_binary_mask(gt_mask)
    pred_before = instances_from_binary_mask(occupancy)
    before = match_components(pred_before, gt)
    pair = next(item for item in before.pairs if item.gt_id == 1)
    pred_after = pred_before.without(pair.pred_id)
    occupancy_after = pred_after.occupancy
    forged = LegalDeletion(
        gt_id=pair.gt_id,
        pred_id=pair.pred_id,
        occupancy_after=occupancy_after,
        pred_after=pred_after,
        match_after=match_components(pred_after, gt),
    )
    with pytest.raises(ValueError, match="full-GT coverage restoration"):
        build_counterfactual_residual_set_supervision(
            forged,
            gt,
            before,
            occupancy,
            suppression_radius=0,
        )
