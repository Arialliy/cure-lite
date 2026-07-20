"""Legality gates for model-consistent CURE input interventions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor
from torch.nn import functional as F

from ..config import MatchConfig, OccupancyConfig
from ..instances import mask_iou, union_instance_masks
from ..intervention import full_gt_restores_base_coverage
from ..matching import match_components
from ..occupancy import build_occupancy
from ..types import InstanceMap, MatchResult


REJECTION_REASONS = (
    "target_not_matched_before",
    "non_atomic_coverage_change",
    "retained_component_lineage_changed",
    "no_writable_target_pixels",
    "raw_background_occupancy_increased",
    "unmatched_component_count_increased",
    "unmatched_component_pixels_increased",
    "base_output_changed_outside_guard",
    "input_max_delta_exceeded",
    "input_mean_delta_exceeded",
    "input_changed_outside_roi",
    "full_gt_not_recoverable",
)


@dataclass(frozen=True)
class AcceptanceConfig:
    """Pre-registered semantic and output-stability limits.

    The defaults are conservative reference values for the deterministic toy
    closure.  A real experiment must freeze them before looking at test data.
    """

    min_writable_pixels: int = 1
    min_retained_component_iou: float = 0.5
    output_guard_radius: int = 2
    max_outside_probability_delta: float = 1e-6
    max_input_abs_delta: float = 0.75
    max_input_mean_delta: float = 0.05
    max_outside_input_delta: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_writable_pixels, bool)
            or not isinstance(self.min_writable_pixels, int)
            or self.min_writable_pixels < 1
        ):
            raise ValueError("min_writable_pixels must be a positive integer")
        if (
            isinstance(self.output_guard_radius, bool)
            or not isinstance(self.output_guard_radius, int)
            or self.output_guard_radius < 0
        ):
            raise ValueError("output_guard_radius must be a non-negative integer")
        for name in (
            "min_retained_component_iou",
            "max_outside_probability_delta",
            "max_input_abs_delta",
            "max_input_mean_delta",
            "max_outside_input_delta",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(value)
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.min_retained_component_iou <= 1.0:
            raise ValueError("min_retained_component_iou must lie in [0,1]")
        if any(
            value < 0.0
            for value in (
                self.max_outside_probability_delta,
                self.max_input_abs_delta,
                self.max_input_mean_delta,
                self.max_outside_input_delta,
            )
        ):
            raise ValueError("intervention tolerances must be non-negative")


@dataclass(frozen=True)
class AcceptanceDecision:
    """Complete, auditable result of applying the legality gates once."""

    accepted: bool
    reasons: tuple[str, ...]
    retained_lineage_ious: tuple[tuple[int, float], ...]
    writable_pixels: int
    full_gt_recoverable: bool
    outside_probability_max_delta: float
    raw_background_pixels_before: int
    raw_background_pixels_after: int
    unmatched_component_pixels_before: int
    unmatched_component_pixels_after: int

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool")
        if not isinstance(self.reasons, tuple) or any(
            reason not in REJECTION_REASONS for reason in self.reasons
        ):
            raise ValueError("reasons contains an unknown rejection code")
        if self.accepted != (len(self.reasons) == 0):
            raise ValueError("accepted must be equivalent to an empty reason list")
        if self.reasons != tuple(dict.fromkeys(self.reasons)):
            raise ValueError("rejection reasons must be unique and ordered")
        if not isinstance(self.retained_lineage_ious, tuple):
            raise TypeError("retained_lineage_ious must be a tuple")
        lineage_ids: list[int] = []
        for item in self.retained_lineage_ious:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("every lineage record must be a pair")
            gt_id, overlap = item
            if isinstance(gt_id, bool) or not isinstance(gt_id, int) or gt_id < 1:
                raise ValueError("lineage GT IDs must be positive integers")
            if not isfinite(float(overlap)) or not 0.0 <= float(overlap) <= 1.0:
                raise ValueError("lineage IoU must lie in [0,1]")
            lineage_ids.append(gt_id)
        if lineage_ids != sorted(set(lineage_ids)):
            raise ValueError("lineage GT IDs must be sorted and unique")
        for name in (
            "writable_pixels",
            "raw_background_pixels_before",
            "raw_background_pixels_after",
            "unmatched_component_pixels_before",
            "unmatched_component_pixels_after",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.full_gt_recoverable, bool):
            raise TypeError("full_gt_recoverable must be bool")
        if (
            not isfinite(float(self.outside_probability_max_delta))
            or self.outside_probability_max_delta < 0.0
        ):
            raise ValueError(
                "outside_probability_max_delta must be finite and non-negative"
            )


def _probability_2d(probability: Tensor, shape: tuple[int, int]) -> Tensor:
    if not isinstance(probability, Tensor) or probability.dtype != torch.float32:
        raise TypeError("probability must be a float32 tensor")
    value = probability.detach().to(device="cpu")
    if value.ndim == 4 and value.shape[:2] == (1, 1):
        value = value[0, 0]
    elif value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or tuple(value.shape) != shape:
        raise ValueError("probability must contain one map on the GT grid")
    if not torch.isfinite(value).all() or torch.any((value < 0.0) | (value > 1.0)):
        raise ValueError("probability must be finite and lie in [0,1]")
    return value.contiguous()


def _support_2d(mask: Tensor, shape: tuple[int, int]) -> Tensor:
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise TypeError("changed_support must be a bool tensor")
    value = mask.detach().to(device="cpu")
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or tuple(value.shape) != shape:
        raise ValueError("changed_support must have the GT grid shape")
    if not bool(torch.any(value)):
        raise ValueError("changed_support cannot be empty")
    return value.contiguous()


def _dilate(mask: Tensor, radius: int) -> Tensor:
    if radius == 0:
        return mask
    width = 2 * radius + 1
    return (
        F.max_pool2d(
            mask.to(torch.float32)[None, None],
            kernel_size=width,
            stride=1,
            padding=radius,
        )[0, 0]
        > 0.0
    )


def _pairs_by_gt(result: MatchResult) -> dict[int, int]:
    return {pair.gt_id: pair.pred_id for pair in result.pairs}


def _unmatched_component_pixels(
    prediction: InstanceMap,
    matching: MatchResult,
) -> int:
    return sum(
        prediction.by_id(pred_id).area
        for pred_id in matching.unmatched_pred_ids
    )


def assess_legal_intervention(
    *,
    gt: InstanceMap,
    target_gt_id: int,
    pred_before: InstanceMap,
    match_before: MatchResult,
    probability_before: Tensor,
    pred_after: InstanceMap,
    match_after: MatchResult,
    probability_after: Tensor,
    changed_support: Tensor,
    transform_max_abs_delta: float,
    transform_mean_abs_delta: float,
    transform_outside_max_delta: float,
    occupancy_config: OccupancyConfig = OccupancyConfig(),
    match_config: MatchConfig = MatchConfig(),
    config: AcceptanceConfig = AcceptanceConfig(),
) -> AcceptanceDecision:
    """Accept only an atomic, recoverable, locally bounded coverage change."""

    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be an InstanceMap")
    if not isinstance(pred_before, InstanceMap) or not isinstance(pred_after, InstanceMap):
        raise TypeError("pred_before and pred_after must be InstanceMap objects")
    if not isinstance(match_before, MatchResult) or not isinstance(match_after, MatchResult):
        raise TypeError("match_before and match_after must be MatchResult objects")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be MatchConfig")
    if not isinstance(occupancy_config, OccupancyConfig):
        raise TypeError("occupancy_config must be OccupancyConfig")
    if not isinstance(config, AcceptanceConfig):
        raise TypeError("config must be AcceptanceConfig")
    if (
        isinstance(target_gt_id, bool)
        or not isinstance(target_gt_id, int)
        or target_gt_id < 1
    ):
        raise ValueError("target_gt_id must be a positive integer")
    gt.by_id(target_gt_id)
    if pred_before.shape != gt.shape or pred_after.shape != gt.shape:
        raise ValueError("prediction and GT grids must match")
    if match_before != match_components(pred_before, gt, match_config):
        raise ValueError("match_before is stale or inconsistent")
    if match_after != match_components(pred_after, gt, match_config):
        raise ValueError("match_after is stale or inconsistent")

    before_probability = _probability_2d(probability_before, gt.shape)
    after_probability = _probability_2d(probability_after, gt.shape)
    _, expected_pred_before = build_occupancy(
        before_probability, occupancy_config
    )
    _, expected_pred_after = build_occupancy(after_probability, occupancy_config)
    if not torch.equal(expected_pred_before.labels, pred_before.labels):
        raise ValueError("pred_before was not built from probability_before")
    if not torch.equal(expected_pred_after.labels, pred_after.labels):
        raise ValueError("pred_after was not built from probability_after")
    support = _support_2d(changed_support, gt.shape)
    for name, value in (
        ("transform_max_abs_delta", transform_max_abs_delta),
        ("transform_mean_abs_delta", transform_mean_abs_delta),
        ("transform_outside_max_delta", transform_outside_max_delta),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    reasons: list[str] = []
    if target_gt_id not in match_before.matched_gt_ids:
        reasons.append("target_not_matched_before")

    expected_covered = match_before.matched_gt_ids - {target_gt_id}
    if match_after.matched_gt_ids != expected_covered:
        reasons.append("non_atomic_coverage_change")

    before_by_gt = _pairs_by_gt(match_before)
    after_by_gt = _pairs_by_gt(match_after)
    lineage: list[tuple[int, float]] = []
    for retained_gt_id in sorted(expected_covered & match_after.matched_gt_ids):
        overlap = mask_iou(
            pred_before.by_id(before_by_gt[retained_gt_id]).mask,
            pred_after.by_id(after_by_gt[retained_gt_id]).mask,
        )
        lineage.append((retained_gt_id, overlap))
        if overlap < config.min_retained_component_iou:
            reasons.append("retained_component_lineage_changed")

    writable_pixels = int(
        torch.count_nonzero(gt.by_id(target_gt_id).mask & ~pred_after.occupancy)
    )
    if writable_pixels < config.min_writable_pixels:
        reasons.append("no_writable_target_pixels")

    background = ~union_instance_masks(gt, gt.ids)
    raw_before = int(torch.count_nonzero(pred_before.occupancy & background))
    raw_after = int(torch.count_nonzero(pred_after.occupancy & background))
    if raw_after > raw_before:
        reasons.append("raw_background_occupancy_increased")
    if len(match_after.unmatched_pred_ids) > len(match_before.unmatched_pred_ids):
        reasons.append("unmatched_component_count_increased")
    unmatched_pixels_before = _unmatched_component_pixels(
        pred_before, match_before
    )
    unmatched_pixels_after = _unmatched_component_pixels(pred_after, match_after)
    if unmatched_pixels_after > unmatched_pixels_before:
        reasons.append("unmatched_component_pixels_increased")

    guard = _dilate(support, config.output_guard_radius)
    outside = ~guard
    outside_probability_max_delta = (
        float(torch.max(torch.abs(after_probability - before_probability)[outside]))
        if bool(torch.any(outside))
        else 0.0
    )
    if outside_probability_max_delta > config.max_outside_probability_delta:
        reasons.append("base_output_changed_outside_guard")

    if float(transform_max_abs_delta) > config.max_input_abs_delta:
        reasons.append("input_max_delta_exceeded")
    if float(transform_mean_abs_delta) > config.max_input_mean_delta:
        reasons.append("input_mean_delta_exceeded")
    if float(transform_outside_max_delta) > config.max_outside_input_delta:
        reasons.append("input_changed_outside_roi")

    recoverable = full_gt_restores_base_coverage(
        pred_after.occupancy,
        gt,
        target_gt_id,
        match_before,
        match_config,
    )
    if not recoverable:
        reasons.append("full_gt_not_recoverable")

    ordered_reasons = tuple(dict.fromkeys(reasons))
    return AcceptanceDecision(
        accepted=not ordered_reasons,
        reasons=ordered_reasons,
        retained_lineage_ious=tuple(lineage),
        writable_pixels=writable_pixels,
        full_gt_recoverable=recoverable,
        outside_probability_max_delta=outside_probability_max_delta,
        raw_background_pixels_before=raw_before,
        raw_background_pixels_after=raw_after,
        unmatched_component_pixels_before=unmatched_pixels_before,
        unmatched_component_pixels_after=unmatched_pixels_after,
    )


__all__ = [
    "AcceptanceConfig",
    "AcceptanceDecision",
    "REJECTION_REASONS",
    "assess_legal_intervention",
]
