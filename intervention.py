"""Strict legal single-target coverage interventions."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import InterventionConfig, MatchConfig
from .instances import instances_from_binary_mask
from .matching import match_components
from .types import InstanceMap, LegalDeletion, MatchResult


def _as_occupancy(mask: Tensor, shape: tuple[int, int]) -> Tensor:
    if not isinstance(mask, Tensor):
        raise TypeError("occupancy must be a torch.Tensor")
    if mask.dtype != torch.bool:
        raise TypeError("occupancy must be bool")
    occupancy = mask.detach().to(device="cpu", dtype=torch.bool)
    if occupancy.ndim == 3 and occupancy.shape[0] == 1:
        occupancy = occupancy[0]
    if occupancy.ndim != 2 or tuple(occupancy.shape) != shape:
        raise ValueError(f"occupancy must have shape {shape} or (1,{shape[0]},{shape[1]})")
    return occupancy.contiguous()


def _resolve_configs(
    positional: InterventionConfig | MatchConfig | None,
    match_config: MatchConfig | None,
    intervention_config: InterventionConfig | None,
) -> tuple[MatchConfig, InterventionConfig]:
    """Support the documented four/five-positional call forms without guessing."""

    if positional is not None:
        if isinstance(positional, InterventionConfig):
            if intervention_config is not None:
                raise TypeError("intervention configuration was supplied twice")
            intervention_config = positional
        elif isinstance(positional, MatchConfig):
            if match_config is not None:
                raise TypeError("matching configuration was supplied twice")
            match_config = positional
        else:
            raise TypeError("fifth positional argument must be InterventionConfig or MatchConfig")
    if match_config is None:
        match_config = MatchConfig()
    if intervention_config is None:
        intervention_config = InterventionConfig()
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be MatchConfig")
    if not isinstance(intervention_config, InterventionConfig):
        raise TypeError("intervention_config must be InterventionConfig")
    return match_config, intervention_config


def _identity_pairs(result: MatchResult) -> tuple[tuple[int, int], ...]:
    return tuple((pair.gt_id, pair.pred_id) for pair in result.pairs)


def full_gt_restores_base_coverage(
    occupancy_after: Tensor,
    gt: InstanceMap,
    gt_id: int,
    before: MatchResult,
    match_config: MatchConfig,
    *,
    connectivity: int = 8,
    min_area: int = 1,
) -> bool:
    """Whether adding the complete GT restores pre-deletion coverage.

    The full-GT mask is deliberately decomposed into connected components
    again.  This is a conservative legality condition, not a theoretical
    oracle over all possible partial residual masks.  Merely exposing writable
    pixels is insufficient because adding a target can merge it with another
    component and change one-to-one matching.
    """

    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be an InstanceMap")
    if not isinstance(before, MatchResult):
        raise TypeError("before must be a MatchResult")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be MatchConfig")
    if isinstance(gt_id, bool) or not isinstance(gt_id, int) or gt_id < 1:
        raise ValueError("gt_id must be a positive integer")
    if before.gt_ids != tuple(sorted(gt.ids)):
        raise ValueError("before matching is inconsistent with the GT instance map")

    occupied = _as_occupancy(occupancy_after, gt.shape)
    oracle_mask = occupied | gt.by_id(gt_id).mask
    oracle_pred = instances_from_binary_mask(
        oracle_mask,
        connectivity=connectivity,
        min_area=min_area,
    )
    oracle_match = match_components(oracle_pred, gt, match_config)
    return oracle_match.matched_gt_ids == before.matched_gt_ids


def oracle_restores_base_coverage(
    occupancy_after: Tensor,
    gt: InstanceMap,
    gt_id: int,
    before: MatchResult,
    match_config: MatchConfig,
    *,
    connectivity: int = 8,
    min_area: int = 1,
) -> bool:
    """Backward-compatible alias for :func:`full_gt_restores_base_coverage`."""

    return full_gt_restores_base_coverage(
        occupancy_after,
        gt,
        gt_id,
        before,
        match_config,
        connectivity=connectivity,
        min_area=min_area,
    )


def enumerate_legal_deletions(
    pred: InstanceMap,
    gt: InstanceMap,
    before: MatchResult,
    occupancy: Tensor,
    config: InterventionConfig | MatchConfig | None = None,
    *,
    match_config: MatchConfig | None = None,
    intervention_config: InterventionConfig | None = None,
) -> tuple[LegalDeletion, ...]:
    """Enumerate deletions satisfying A/B/C and full-GT restoration.

    The fifth positional argument is normally :class:`InterventionConfig`, as
    shown by the main-body API.  Passing a :class:`MatchConfig` there is also
    accepted for backward-compatible four/five-positional usage; the other
    configuration can always be supplied by keyword.
    """

    if not isinstance(pred, InstanceMap) or not isinstance(gt, InstanceMap):
        raise TypeError("pred and gt must be InstanceMap objects")
    if not isinstance(before, MatchResult):
        raise TypeError("before must be a MatchResult")
    if pred.shape != gt.shape:
        raise ValueError("prediction and GT instance maps must have the same shape")
    match_config, intervention_config = _resolve_configs(
        config, match_config, intervention_config
    )
    occupancy_2d = _as_occupancy(occupancy, pred.shape)
    if not torch.equal(occupancy_2d, pred.occupancy):
        raise ValueError("occupancy must exactly equal pred.occupancy")

    expected_before = match_components(pred, gt, match_config)
    if before != expected_before:
        raise ValueError("before matching is stale or inconsistent with pred, gt, or match_config")

    original_misses = set(before.unmatched_gt_ids)
    original_pairs = set(_identity_pairs(before))
    legal: list[LegalDeletion] = []

    for pair in before.pairs:
        pred_after = pred.without(pair.pred_id)
        occupancy_after = occupancy_2d & (pred.labels != pair.pred_id)
        if not torch.equal(occupancy_after, pred_after.occupancy):
            raise AssertionError("component deletion produced inconsistent occupancy")
        after = match_components(pred_after, gt, match_config)

        # A: exactly the selected GT is added to the original miss set.
        if set(after.unmatched_gt_ids) != original_misses | {pair.gt_id}:
            continue
        # B: every remaining component-target identity is unchanged.
        expected_after_pairs = original_pairs - {(pair.gt_id, pair.pred_id)}
        if set(_identity_pairs(after)) != expected_after_pairs:
            continue
        # C: the selected target retains at least one residual-writable pixel.
        target_mask = gt.by_id(pair.gt_id).mask
        writable_pixels = int(torch.count_nonzero(target_mask & ~occupancy_after))
        if writable_pixels < intervention_config.min_writable_pixels:
            continue
        # A writable target is legal only when perfect residual output survives
        # the complete CC8 + one-to-one matching pipeline and restores exactly
        # the GT coverage that existed before deletion.
        if not full_gt_restores_base_coverage(
            occupancy_after,
            gt,
            pair.gt_id,
            before,
            match_config,
        ):
            continue

        legal.append(
            LegalDeletion(
                gt_id=pair.gt_id,
                pred_id=pair.pred_id,
                occupancy_after=occupancy_after,
                pred_after=pred_after,
                match_after=after,
            )
        )

    return tuple(sorted(legal, key=lambda item: (item.gt_id, item.pred_id)))
