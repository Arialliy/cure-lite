"""Deterministic CURE-Lite object, false-alarm, and IoU metrics.

All object quantities use the same CC8 decomposition and deterministic matcher
as state construction.  The standard pixel false-alarm rate counts pixels in
unmatched components.  ``raw_background_fa`` is reported separately because a
background appendage connected to a matched component is invisible to the
standard false-alarm definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from .config import MatchConfig
from .instances import instances_from_binary_mask
from .matching import match_components
from .types import InstanceMap


def _as_2d_bool(value: Tensor, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value, device="cpu")
    if tensor.ndim == 4 and tensor.shape[:2] == (1, 1):
        tensor = tensor[0, 0]
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be [H,W], [1,H,W], or [1,1,H,W]")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    if torch.any((tensor != 0) & (tensor != 1)):
        raise ValueError(f"{name} must be binary")
    return tensor.to(torch.bool).contiguous()


def full_pipeline_reachable_anchor_miss_ids(
    anchor_prediction: Tensor,
    gt_mask: Tensor,
    match_config: MatchConfig,
) -> frozenset[int]:
    """Return individually full-GT-recoverable misses of the fixed anchor.

    A miss is reachable only when adding its complete GT mask to the anchor,
    followed by a fresh CC8 decomposition and matching, matches that target and
    retains every target covered by the anchor.  The resulting set supplies a
    conservative diagnostic denominator for reachable-RMR.  It is not a
    theoretical upper bound over arbitrary partial residual masks and never
    replaces the all-anchor-miss denominator of gross/net RMR.
    """

    anchor_bool = _as_2d_bool(anchor_prediction, name="anchor_prediction")
    gt_bool = _as_2d_bool(gt_mask, name="gt_mask")
    if anchor_bool.shape != gt_bool.shape:
        raise ValueError("anchor_prediction and gt_mask must have identical shapes")

    anchor_instances = instances_from_binary_mask(
        anchor_bool, connectivity=8, min_area=1
    )
    gt_instances = instances_from_binary_mask(gt_bool, connectivity=8, min_area=1)
    return full_pipeline_reachable_anchor_miss_ids_from_instances(
        anchor_bool,
        gt_bool,
        anchor_instances,
        gt_instances,
        match_config,
    )


def full_pipeline_reachable_anchor_miss_ids_from_instances(
    anchor_prediction: Tensor,
    gt_mask: Tensor,
    anchor_instances: InstanceMap,
    gt_instances: InstanceMap,
    match_config: MatchConfig,
) -> frozenset[int]:
    """Evaluate reachability with exact, already decomposed anchor/GT masks.

    The explicit occupancy checks make this a safe package API: stale or
    mismatched precomputed components fail closed.  This lets calibration build
    fixed components once without changing the public metric semantics.
    """

    anchor_bool = _as_2d_bool(anchor_prediction, name="anchor_prediction")
    gt_bool = _as_2d_bool(gt_mask, name="gt_mask")
    if anchor_bool.shape != gt_bool.shape:
        raise ValueError("anchor_prediction and gt_mask must have identical shapes")
    if not isinstance(anchor_instances, InstanceMap):
        raise TypeError("anchor_instances must be an InstanceMap")
    if not isinstance(gt_instances, InstanceMap):
        raise TypeError("gt_instances must be an InstanceMap")
    if anchor_instances.shape != tuple(anchor_bool.shape) or not torch.equal(
        anchor_instances.occupancy, anchor_bool
    ):
        raise ValueError("anchor instance map differs from anchor_prediction")
    if gt_instances.shape != tuple(gt_bool.shape) or not torch.equal(
        gt_instances.occupancy, gt_bool
    ):
        raise ValueError("GT instance map differs from gt_mask")
    anchor_match = match_components(anchor_instances, gt_instances, match_config)
    covered = anchor_match.matched_gt_ids
    reachable: list[int] = []
    for gt_id in sorted(anchor_match.unmatched_gt_ids):
        oracle_prediction = anchor_bool | gt_instances.by_id(gt_id).mask
        oracle_instances = instances_from_binary_mask(
            oracle_prediction, connectivity=8, min_area=1
        )
        oracle_match = match_components(oracle_instances, gt_instances, match_config)
        if gt_id in oracle_match.matched_gt_ids and covered <= oracle_match.matched_gt_ids:
            reachable.append(gt_id)
    return frozenset(reachable)


@dataclass(frozen=True)
class ImageEvaluation:
    """Sufficient statistics for one evaluation image."""

    matched_gt: int
    total_gt: int
    recovered_anchor_misses: int
    overlap_supported_recovered_anchor_misses: int
    total_anchor_misses: int
    retained_anchor_covered: int
    total_anchor_covered: int
    recovered_reachable_anchor_misses: int
    total_reachable_anchor_misses: int
    unmatched_pred_pixels: int
    unmatched_pred_components: int
    raw_background_fp: int
    total_pixels: int
    intersection: int
    union: int

    @property
    def pd(self) -> float:
        return self.matched_gt / self.total_gt if self.total_gt else 1.0

    @property
    def gross_rmr(self) -> float:
        if not self.total_anchor_misses:
            return 0.0
        return self.recovered_anchor_misses / self.total_anchor_misses

    @property
    def rmr(self) -> float:
        """Backward-compatible name for gross RMR."""

        return self.gross_rmr

    @property
    def net_recovered_anchor_misses(self) -> int:
        return self.matched_gt - self.total_anchor_covered

    @property
    def net_rmr(self) -> float:
        if not self.total_anchor_misses:
            return 0.0
        return self.net_recovered_anchor_misses / self.total_anchor_misses

    @property
    def retention(self) -> float:
        if not self.total_anchor_covered:
            return 1.0
        return self.retained_anchor_covered / self.total_anchor_covered

    @property
    def reachable_rmr(self) -> float:
        if not self.total_reachable_anchor_misses:
            return 0.0
        return (
            self.recovered_reachable_anchor_misses
            / self.total_reachable_anchor_misses
        )

    @property
    def oracle_upper_bound(self) -> float:
        """Legacy name for the conservative full-GT recoverability ratio."""

        return self.full_gt_recoverability

    @property
    def full_gt_recoverability(self) -> float:
        """Fraction of anchor misses recoverable by adding their complete GT."""

        if not self.total_anchor_misses:
            return 0.0
        return self.total_reachable_anchor_misses / self.total_anchor_misses

    @property
    def overlap_supported_rmr(self) -> float:
        if not self.total_anchor_misses:
            return 0.0
        return (
            self.overlap_supported_recovered_anchor_misses
            / self.total_anchor_misses
        )

    @property
    def pixel_fa(self) -> float:
        return self.unmatched_pred_pixels / self.total_pixels if self.total_pixels else 0.0

    @property
    def raw_background_fa(self) -> float:
        return self.raw_background_fp / self.total_pixels if self.total_pixels else 0.0

    @property
    def fp_components_per_mp(self) -> float:
        megapixels = self.total_pixels / 1_000_000.0
        return self.unmatched_pred_components / megapixels if megapixels else 0.0

    @property
    def iou(self) -> float:
        return self.intersection / self.union if self.union else 1.0


@dataclass(frozen=True)
class AggregateEvaluation:
    """Dataset-level micro metrics plus the per-image normalized IoU.

    ``miou`` is the global intersection divided by global union.  ``niou`` is
    the unweighted mean of image IoUs.  ``rmr`` is retained only as the legacy
    spelling of ``gross_rmr``; calibration decisions use ``net_rmr``.
    """

    pd: float
    rmr: float
    gross_rmr: float
    net_rmr: float
    retention: float
    reachable_rmr: float
    oracle_upper_bound: float
    overlap_supported_rmr: float
    pixel_fa: float
    raw_background_fa: float
    fp_components_per_mp: float
    miou: float
    niou: float
    images: int
    recovered_anchor_misses: int
    net_recovered_anchor_misses: int
    total_anchor_misses: int
    retained_anchor_covered: int
    total_anchor_covered: int
    recovered_reachable_anchor_misses: int
    total_reachable_anchor_misses: int
    budget_violation: bool = False

    @property
    def global_miou(self) -> float:
        return self.miou

    @property
    def full_gt_recoverability(self) -> float:
        """Conservative diagnostic; ``oracle_upper_bound`` is a legacy field."""

        return self.oracle_upper_bound


def evaluate_binary_prediction(
    prediction: Tensor,
    gt_mask: Tensor,
    match_config: MatchConfig,
    *,
    anchor_miss_ids: set[int] | frozenset[int] = frozenset(),
    reachable_anchor_miss_ids: set[int] | frozenset[int] = frozenset(),
    residual_mask: Tensor | None = None,
) -> ImageEvaluation:
    """Evaluate one fixed-grid binary prediction with the shared matcher.

    GT IDs not in ``anchor_miss_ids`` are the anchor-covered set ``K0``.
    Reachability is deliberately explicit: callers that have not certified an
    full-GT-recoverable set get a conservative empty diagnostic denominator
    rather than an invented claim that every miss is reachable.
    """

    pred_bool = _as_2d_bool(prediction, name="prediction")
    gt_bool = _as_2d_bool(gt_mask, name="gt_mask")
    if pred_bool.shape != gt_bool.shape:
        raise ValueError("prediction and gt_mask must have identical shapes")

    pred_instances = instances_from_binary_mask(
        pred_bool, connectivity=8, min_area=1
    )
    gt_instances = instances_from_binary_mask(gt_bool, connectivity=8, min_area=1)
    return evaluate_binary_prediction_from_instances(
        pred_bool,
        gt_bool,
        pred_instances,
        gt_instances,
        match_config,
        anchor_miss_ids=anchor_miss_ids,
        reachable_anchor_miss_ids=reachable_anchor_miss_ids,
        residual_mask=residual_mask,
    )


def evaluate_binary_prediction_from_instances(
    prediction: Tensor,
    gt_mask: Tensor,
    pred_instances: InstanceMap,
    gt_instances: InstanceMap,
    match_config: MatchConfig,
    *,
    anchor_miss_ids: set[int] | frozenset[int] = frozenset(),
    reachable_anchor_miss_ids: set[int] | frozenset[int] = frozenset(),
    residual_mask: Tensor | None = None,
) -> ImageEvaluation:
    """Evaluate with exact, already decomposed prediction and GT masks.

    The public evaluator and the accelerated calibration ledger both delegate
    here.  Explicit occupancy checks make a stale or mismatched component map
    fail closed rather than silently changing metric semantics.
    """

    pred_bool = _as_2d_bool(prediction, name="prediction")
    gt_bool = _as_2d_bool(gt_mask, name="gt_mask")
    if pred_bool.shape != gt_bool.shape:
        raise ValueError("prediction and gt_mask must have identical shapes")
    if not isinstance(pred_instances, InstanceMap):
        raise TypeError("pred_instances must be an InstanceMap")
    if not isinstance(gt_instances, InstanceMap):
        raise TypeError("gt_instances must be an InstanceMap")
    if pred_instances.shape != tuple(pred_bool.shape) or not torch.equal(
        pred_instances.occupancy, pred_bool
    ):
        raise ValueError("prediction instance map differs from prediction")
    if gt_instances.shape != tuple(gt_bool.shape) or not torch.equal(
        gt_instances.occupancy, gt_bool
    ):
        raise ValueError("GT instance map differs from gt_mask")
    match = match_components(pred_instances, gt_instances, match_config)

    anchor_ids = frozenset(int(item) for item in anchor_miss_ids)
    reachable_ids = frozenset(int(item) for item in reachable_anchor_miss_ids)
    gt_ids = frozenset(gt_instances.ids)
    if not anchor_ids <= gt_ids:
        raise ValueError("anchor_miss_ids do not belong to this GT instance map")
    if not reachable_ids <= anchor_ids:
        raise ValueError("reachable_anchor_miss_ids must be a subset of anchor_miss_ids")
    anchor_covered_ids = gt_ids - anchor_ids

    unmatched_pixels = sum(
        pred_instances.by_id(pred_id).area for pred_id in match.unmatched_pred_ids
    )
    recovered_ids = anchor_ids & match.matched_gt_ids
    retained_ids = anchor_covered_ids & match.matched_gt_ids
    recovered_reachable_ids = reachable_ids & match.matched_gt_ids
    overlap_supported = 0
    if residual_mask is not None:
        residual_bool = _as_2d_bool(residual_mask, name="residual_mask")
        if residual_bool.shape != gt_bool.shape:
            raise ValueError("residual_mask and gt_mask must have identical shapes")
        overlap_supported = sum(
            bool(torch.any(residual_bool & gt_instances.by_id(gt_id).mask))
            for gt_id in recovered_ids
        )
    intersection = int(torch.count_nonzero(pred_bool & gt_bool))
    union = int(torch.count_nonzero(pred_bool | gt_bool))
    raw_background_fp = int(torch.count_nonzero(pred_bool & ~gt_bool))
    return ImageEvaluation(
        matched_gt=match.cardinality,
        total_gt=len(gt_instances.instances),
        recovered_anchor_misses=len(recovered_ids),
        overlap_supported_recovered_anchor_misses=overlap_supported,
        total_anchor_misses=len(anchor_ids),
        retained_anchor_covered=len(retained_ids),
        total_anchor_covered=len(anchor_covered_ids),
        recovered_reachable_anchor_misses=len(recovered_reachable_ids),
        total_reachable_anchor_misses=len(reachable_ids),
        unmatched_pred_pixels=unmatched_pixels,
        unmatched_pred_components=len(match.unmatched_pred_ids),
        raw_background_fp=raw_background_fp,
        total_pixels=pred_bool.numel(),
        intersection=intersection,
        union=union,
    )


def aggregate_evaluations(records: Iterable[ImageEvaluation]) -> AggregateEvaluation:
    """Micro-average object/FA counts and report both IoU aggregations."""

    items = tuple(records)
    if not items:
        raise ValueError("at least one evaluation record is required")

    matched = sum(item.matched_gt for item in items)
    total_gt = sum(item.total_gt for item in items)
    recovered = sum(item.recovered_anchor_misses for item in items)
    net_recovered = sum(item.net_recovered_anchor_misses for item in items)
    overlap_supported = sum(
        item.overlap_supported_recovered_anchor_misses for item in items
    )
    total_misses = sum(item.total_anchor_misses for item in items)
    retained = sum(item.retained_anchor_covered for item in items)
    total_covered = sum(item.total_anchor_covered for item in items)
    recovered_reachable = sum(
        item.recovered_reachable_anchor_misses for item in items
    )
    total_reachable = sum(item.total_reachable_anchor_misses for item in items)
    unmatched_pixels = sum(item.unmatched_pred_pixels for item in items)
    unmatched_components = sum(item.unmatched_pred_components for item in items)
    raw_background_fp = sum(item.raw_background_fp for item in items)
    pixels = sum(item.total_pixels for item in items)
    intersection = sum(item.intersection for item in items)
    union = sum(item.union for item in items)
    gross_rmr = recovered / total_misses if total_misses else 0.0

    return AggregateEvaluation(
        pd=matched / total_gt if total_gt else 1.0,
        rmr=gross_rmr,
        gross_rmr=gross_rmr,
        net_rmr=net_recovered / total_misses if total_misses else 0.0,
        retention=retained / total_covered if total_covered else 1.0,
        reachable_rmr=(
            recovered_reachable / total_reachable if total_reachable else 0.0
        ),
        oracle_upper_bound=(
            total_reachable / total_misses if total_misses else 0.0
        ),
        overlap_supported_rmr=(
            overlap_supported / total_misses if total_misses else 0.0
        ),
        pixel_fa=unmatched_pixels / pixels if pixels else 0.0,
        raw_background_fa=raw_background_fp / pixels if pixels else 0.0,
        fp_components_per_mp=(
            unmatched_components / (pixels / 1_000_000.0) if pixels else 0.0
        ),
        miou=intersection / union if union else 1.0,
        niou=sum(item.iou for item in items) / len(items),
        images=len(items),
        recovered_anchor_misses=recovered,
        net_recovered_anchor_misses=net_recovered,
        total_anchor_misses=total_misses,
        retained_anchor_covered=retained,
        total_anchor_covered=total_covered,
        recovered_reachable_anchor_misses=recovered_reachable,
        total_reachable_anchor_misses=total_reachable,
    )
