"""Residual-set supervision for factual and counterfactual coverage states."""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import MatchConfig
from ..instances import instances_from_binary_mask, union_instance_masks
from ..intervention import full_gt_restores_base_coverage
from ..matching import match_components
from ..types import InstanceMap, LegalDeletion, MatchResult
from .descriptors import dilate_mask
from .experiment import (
    CounterfactualBackgroundPolicy,
    CounterfactualTargetPolicy,
)
from .types import ResidualSetSupervision


def _occupancy_2d(occupancy: Tensor, shape: tuple[int, int]) -> Tensor:
    if not isinstance(occupancy, Tensor) or occupancy.dtype != torch.bool:
        raise TypeError("occupancy must be a bool tensor")
    result = occupancy.detach().to(device="cpu", dtype=torch.bool)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 2 or tuple(result.shape) != shape:
        raise ValueError("occupancy and GT must share a [H,W] grid")
    return result.contiguous()


def _validate_match(
    occupancy: Tensor,
    gt: InstanceMap,
    match: MatchResult,
    match_config: MatchConfig,
) -> Tensor:
    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be an InstanceMap")
    if not isinstance(match, MatchResult):
        raise TypeError("match must be a MatchResult")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be MatchConfig")
    occupied = _occupancy_2d(occupancy, gt.shape)
    prediction = instances_from_binary_mask(occupied, connectivity=8, min_area=1)
    expected = match_components(prediction, gt, match_config)
    if match != expected:
        raise ValueError("match is stale or inconsistent with occupancy and GT")
    return occupied


def _build(
    occupancy: Tensor,
    gt: InstanceMap,
    match: MatchResult,
    *,
    suppression_radius: int,
    branch: str,
    required_gt_id: int | None = None,
    supervised_gt_ids: tuple[int, ...] | None = None,
    counterfactual_background_policy: CounterfactualBackgroundPolicy = (
        CounterfactualBackgroundPolicy.EMPTY
    ),
) -> ResidualSetSupervision:
    exclusion = dilate_mask(occupancy, suppression_radius)
    editable = ~exclusion
    positive_ids: list[int] = []
    uneditable_ids: list[int] = []
    object_masks: list[Tensor] = []
    unmatched_ids = tuple(sorted(match.unmatched_gt_ids))
    if supervised_gt_ids is None:
        supervised_ids = unmatched_ids
    else:
        supervised_ids = tuple(sorted(set(supervised_gt_ids)))
        if set(supervised_ids) - set(unmatched_ids):
            raise ValueError("supervised_gt_ids must be currently unmatched")
    ignored_ids = tuple(sorted(set(unmatched_ids) - set(supervised_ids)))
    for gt_id in supervised_ids:
        writable = gt.by_id(gt_id).mask & editable
        if bool(torch.any(writable)):
            positive_ids.append(gt_id)
            object_masks.append(writable)
        else:
            uneditable_ids.append(gt_id)
    if required_gt_id is not None and required_gt_id not in positive_ids:
        raise ValueError(
            "the counterfactually deleted target has no residual-writable pixels "
            "after occupancy suppression"
        )
    stacked = (
        torch.stack(object_masks)
        if object_masks
        else torch.zeros((0, *gt.shape), dtype=torch.bool)
    )
    target = (
        stacked.any(dim=0)
        if object_masks
        else torch.zeros(gt.shape, dtype=torch.bool)
    )
    gt_union = union_instance_masks(gt, gt.ids)
    if not isinstance(
        counterfactual_background_policy, CounterfactualBackgroundPolicy
    ):
        raise TypeError(
            "counterfactual_background_policy must be "
            "CounterfactualBackgroundPolicy"
        )
    # Counterfactual target sampling must not also resample the host image's
    # background gradient in the default method.  Host-background BCE remains
    # available only as an explicit, receipt-bound ablation.
    if branch == "factual" or (
        branch == "counterfactual"
        and counterfactual_background_policy
        is CounterfactualBackgroundPolicy.BCE
    ):
        background = editable & ~gt_union
    else:
        background = torch.zeros_like(editable)
    return ResidualSetSupervision(
        occupancy=occupancy.unsqueeze(0),
        editable_mask=editable.unsqueeze(0),
        target=target.to(torch.float32).unsqueeze(0),
        background_mask=background.unsqueeze(0),
        object_masks=stacked,
        positive_gt_ids=tuple(positive_ids),
        uneditable_gt_ids=tuple(uneditable_ids),
        ignored_gt_ids=ignored_ids,
        branch=branch,
    )


def build_factual_residual_set_supervision(
    occupancy: Tensor,
    gt: InstanceMap,
    match: MatchResult,
    match_config: MatchConfig = MatchConfig(),
    *,
    suppression_radius: int,
) -> ResidualSetSupervision:
    """Supervise every writable factual miss in one frozen-base state."""

    occupied = _validate_match(occupancy, gt, match, match_config)
    return _build(
        occupied,
        gt,
        match,
        suppression_radius=suppression_radius,
        branch="factual",
    )


def build_counterfactual_residual_set_supervision(
    deletion: LegalDeletion,
    gt: InstanceMap,
    before: MatchResult,
    occupancy_before: Tensor,
    match_config: MatchConfig = MatchConfig(),
    *,
    suppression_radius: int,
    target_policy: CounterfactualTargetPolicy = (
        CounterfactualTargetPolicy.SELECTED_DELETED_TARGET_ONLY
    ),
    background_policy: CounterfactualBackgroundPolicy = (
        CounterfactualBackgroundPolicy.EMPTY
    ),
) -> ResidualSetSupervision:
    """Supervise one atomically uncensored target after a legal deletion.

    Factual misses already receive exposure in the factual branch.  By default
    they are ignored in this state so odds sampling changes only exposure of the
    selected target, rather than silently changing factual-miss exposure through
    the host-image distribution.  ``ALL_UNCOVERED_TARGETS`` and counterfactual
    background BCE are explicit ablation axes that are later bound into the
    sealed training-policy and state-pool receipts.
    """

    if not isinstance(deletion, LegalDeletion):
        raise TypeError("deletion must be LegalDeletion")
    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be an InstanceMap")
    if not isinstance(before, MatchResult):
        raise TypeError("before must be a MatchResult")
    if not isinstance(target_policy, CounterfactualTargetPolicy):
        raise TypeError("target_policy must be CounterfactualTargetPolicy")
    if not isinstance(background_policy, CounterfactualBackgroundPolicy):
        raise TypeError("background_policy must be CounterfactualBackgroundPolicy")
    occupied_before = _occupancy_2d(occupancy_before, gt.shape)
    pred_before = instances_from_binary_mask(
        occupied_before, connectivity=8, min_area=1
    )
    expected_before = match_components(pred_before, gt, match_config)
    if before != expected_before:
        raise ValueError("before matching is inconsistent with occupancy_before and GT")
    try:
        removed_component = pred_before.by_id(deletion.pred_id).mask
    except KeyError as error:
        raise ValueError("deleted prediction component did not exist before") from error
    expected_occupancy_after = occupied_before & ~removed_component
    if not torch.equal(expected_occupancy_after, deletion.occupancy_after):
        raise ValueError(
            "counterfactual state must delete exactly one complete prediction component"
        )
    if not full_gt_restores_base_coverage(
        deletion.occupancy_after,
        gt,
        deletion.gt_id,
        before,
        match_config,
    ):
        raise ValueError("counterfactual deletion fails full-GT coverage restoration")
    if deletion.pred_after.shape != gt.shape:
        raise ValueError("deletion and GT grids differ")
    occupied = _occupancy_2d(deletion.occupancy_after, gt.shape)
    if not torch.equal(occupied, deletion.pred_after.occupancy):
        raise ValueError("deletion occupancy is inconsistent with pred_after")
    expected_after = match_components(deletion.pred_after, gt, match_config)
    if deletion.match_after != expected_after:
        raise ValueError("deletion match_after is stale or inconsistent")
    if deletion.gt_id not in deletion.match_after.unmatched_gt_ids:
        raise ValueError("deleted target is not unmatched in the counterfactual state")
    before_pairs = {(pair.gt_id, pair.pred_id) for pair in before.pairs}
    removed_pair = (deletion.gt_id, deletion.pred_id)
    after_pairs = {(pair.gt_id, pair.pred_id) for pair in deletion.match_after.pairs}
    if removed_pair not in before_pairs:
        raise ValueError("deleted target/component was not paired before intervention")
    if set(deletion.match_after.unmatched_gt_ids) != set(before.unmatched_gt_ids) | {
        deletion.gt_id
    }:
        raise ValueError("counterfactual state must introduce exactly one new miss")
    if after_pairs != before_pairs - {removed_pair}:
        raise ValueError("counterfactual deletion changed another match identity")
    return _build(
        occupied,
        gt,
        deletion.match_after,
        suppression_radius=suppression_radius,
        branch="counterfactual",
        required_gt_id=deletion.gt_id,
        supervised_gt_ids=(
            None
            if target_policy is CounterfactualTargetPolicy.ALL_UNCOVERED_TARGETS
            else (deletion.gt_id,)
        ),
        counterfactual_background_policy=background_policy,
    )


__all__ = [
    "build_counterfactual_residual_set_supervision",
    "build_factual_residual_set_supervision",
]
