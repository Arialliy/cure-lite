"""Validation-only threshold calibration for CURE-Lite v0.1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, isnan
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .config import MatchConfig, OccupancyConfig
from .instances import instances_from_binary_mask
from .matching import match_components
from .metrics import (
    AggregateEvaluation,
    aggregate_evaluations,
    evaluate_binary_prediction,
    full_pipeline_reachable_anchor_miss_ids,
)
from .occupancy import build_occupancy


_VALIDATION_ROLES = frozenset({"D_V", "validation"})
_UNSET = object()


def _require_validation(split_role: str) -> None:
    if split_role not in _VALIDATION_ROLES:
        raise RuntimeError("threshold selection is permitted only on D_V/validation")


def _as_probability(value: Tensor, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tensor.ndim == 4 and tensor.shape[:2] == (1, 1):
        tensor = tensor[0, 0]
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must contain one 2D probability map")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    if torch.any((tensor < 0) | (tensor > 1)):
        raise ValueError(f"{name} must lie in [0,1]")
    return tensor.contiguous()


def _as_gt(value: Tensor) -> Tensor:
    tensor = torch.as_tensor(value, device="cpu")
    if tensor.ndim == 4 and tensor.shape[:2] == (1, 1):
        tensor = tensor[0, 0]
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError("gt_mask must contain one 2D mask")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError("gt_mask contains non-finite values")
    if torch.any((tensor != 0) & (tensor != 1)):
        raise ValueError("gt_mask must be binary")
    return tensor.to(torch.bool).contiguous()


def _thresholds(
    values: Iterable[float], *, allow_empty: bool = False
) -> tuple[float, ...]:
    result = tuple(sorted(set(float(value) for value in values)))
    if not result and not allow_empty:
        raise ValueError("threshold candidates are empty")
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in result):
        raise ValueError("threshold candidates must be finite and in [0,1]")
    return result


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    base_probability: Tensor
    residual_probability: Tensor
    gt_mask: Tensor

    def normalized(self) -> tuple[Tensor, Tensor, Tensor]:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        base = _as_probability(self.base_probability, name="base_probability")
        residual = _as_probability(
            self.residual_probability, name="residual_probability"
        )
        gt = _as_gt(self.gt_mask)
        if base.shape != residual.shape or base.shape != gt.shape:
            raise ValueError("base, residual, and GT grids must be identical")
        return base, residual, gt


@dataclass(frozen=True)
class FalseAlarmBudget:
    pixel_fa_budget: float
    component_fa_per_mp_budget: float = float("inf")
    raw_background_fa_budget: float = float("inf")
    minimum_retention: float = 0.0

    def __post_init__(self) -> None:
        if self.pixel_fa_budget < 0 or not isfinite(self.pixel_fa_budget):
            raise ValueError("pixel_fa_budget must be finite and non-negative")
        if isnan(self.component_fa_per_mp_budget) or self.component_fa_per_mp_budget < 0:
            raise ValueError("component_fa_per_mp_budget must be non-negative")
        if isnan(self.raw_background_fa_budget) or self.raw_background_fa_budget < 0:
            raise ValueError("raw_background_fa_budget must be non-negative")
        if not isfinite(self.minimum_retention) or not 0.0 <= self.minimum_retention <= 1.0:
            raise ValueError("minimum_retention must be finite and in [0,1]")

    def accepts(self, metrics: AggregateEvaluation) -> bool:
        return (
            metrics.pixel_fa <= self.pixel_fa_budget
            and metrics.fp_components_per_mp <= self.component_fa_per_mp_budget
            and metrics.raw_background_fa <= self.raw_background_fa_budget
            and metrics.retention >= self.minimum_retention
        )


def _coerce_budget(value: object) -> FalseAlarmBudget:
    if isinstance(value, FalseAlarmBudget):
        return value
    try:
        pixel = float(getattr(value, "pixel_fa_budget"))
        component = float(getattr(value, "component_fa_per_mp_budget"))
        raw_background = float(
            getattr(value, "raw_background_fa_budget", float("inf"))
        )
        minimum_retention = float(getattr(value, "minimum_retention", 0.0))
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "budget must expose valid false-alarm and retention limits"
        ) from error
    return FalseAlarmBudget(pixel, component, raw_background, minimum_retention)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float | None
    metrics: AggregateEvaluation | None
    feasible: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.feasible:
            if self.metrics is None:
                raise ValueError("feasible selection must contain metrics")
            if self.reason is not None:
                raise ValueError("feasible selection cannot contain a failure reason")
        elif self.threshold is not None or self.metrics is not None:
            raise ValueError("infeasible selection cannot contain a threshold or metrics")

    @property
    def is_null_residual(self) -> bool:
        """Whether a feasible selection explicitly disables the residual."""

        return self.feasible and self.threshold is None


def anchor_miss_ids(
    base_probability: Tensor,
    gt_mask: Tensor,
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
) -> frozenset[int]:
    base = _as_probability(base_probability, name="base_probability")
    gt_bool = _as_gt(gt_mask)
    if base.shape != gt_bool.shape:
        raise ValueError("base probability and GT grids must be identical")
    _, pred = build_occupancy(base, occupancy_config)
    gt = instances_from_binary_mask(gt_bool, connectivity=8, min_area=1)
    return frozenset(match_components(pred, gt, match_config).unmatched_gt_ids)


def _fixed_anchor_misses(
    samples: Sequence[CalibrationSample],
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
) -> dict[str, frozenset[int]]:
    misses, _ = _fixed_anchor_state(samples, occupancy_config, match_config)
    return misses


def _fixed_anchor_state(
    samples: Sequence[CalibrationSample],
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
) -> tuple[dict[str, frozenset[int]], dict[str, frozenset[int]]]:
    misses: dict[str, frozenset[int]] = {}
    reachable: dict[str, frozenset[int]] = {}
    for sample in samples:
        base, _, gt = sample.normalized()
        if sample.sample_id in misses:
            raise ValueError(f"duplicate sample_id {sample.sample_id!r}")
        sample_misses = anchor_miss_ids(
            base, gt, occupancy_config, match_config
        )
        occupancy, _ = build_occupancy(base, occupancy_config)
        misses[sample.sample_id] = sample_misses
        reachable[sample.sample_id] = full_pipeline_reachable_anchor_miss_ids(
            occupancy, gt, match_config
        )
    return misses, reachable


def _validated_anchor_state(
    samples: Sequence[CalibrationSample],
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    anchor_miss_ids_by_sample: Mapping[str, frozenset[int]] | None,
    reachable_anchor_miss_ids_by_sample: Mapping[str, frozenset[int]] | None,
) -> tuple[dict[str, frozenset[int]], dict[str, frozenset[int]]]:
    expected_anchors, expected_reachable = _fixed_anchor_state(
        samples, occupancy_config, match_config
    )
    anchors = (
        expected_anchors
        if anchor_miss_ids_by_sample is None
        else {
            str(sample_id): frozenset(int(item) for item in ids)
            for sample_id, ids in anchor_miss_ids_by_sample.items()
        }
    )
    reachable = (
        expected_reachable
        if reachable_anchor_miss_ids_by_sample is None
        else {
            str(sample_id): frozenset(int(item) for item in ids)
            for sample_id, ids in reachable_anchor_miss_ids_by_sample.items()
        }
    )
    if anchors != expected_anchors:
        raise ValueError("anchor miss sets differ from the fixed occupancy anchor")
    if reachable != expected_reachable:
        raise ValueError(
            "reachable miss sets differ from the full-GT recoverability diagnostic"
        )
    return anchors, reachable


def evaluate_residual_threshold(
    samples: Sequence[CalibrationSample],
    threshold: float | None,
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    *,
    anchor_miss_ids_by_sample: Mapping[str, frozenset[int]] | None = None,
    reachable_anchor_miss_ids_by_sample: Mapping[str, frozenset[int]] | None = None,
) -> AggregateEvaluation:
    """Evaluate one residual operating point without selecting or changing it.

    ``threshold=None`` is the explicit residual-off candidate.  A numeric
    threshold, including 1.0, retains its literal inclusive-threshold meaning.
    """

    if threshold is not None:
        threshold = float(threshold)
        if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("residual threshold must be finite and in [0,1]")
    if not samples:
        raise ValueError("at least one calibration sample is required")
    anchors, reachable = _validated_anchor_state(
        samples,
        occupancy_config,
        match_config,
        anchor_miss_ids_by_sample,
        reachable_anchor_miss_ids_by_sample,
    )

    records = []
    seen: set[str] = set()
    for sample in samples:
        base, residual_probability, gt = sample.normalized()
        if sample.sample_id in seen:
            raise ValueError(f"duplicate sample_id {sample.sample_id!r}")
        seen.add(sample.sample_id)
        if sample.sample_id not in anchors:
            raise ValueError(f"missing anchor miss set for {sample.sample_id!r}")
        occupancy, _ = build_occupancy(base, occupancy_config)
        residual = (
            torch.zeros_like(occupancy)
            if threshold is None
            else (residual_probability >= threshold) & ~occupancy
        )
        final = occupancy | residual
        records.append(
            evaluate_binary_prediction(
                final,
                gt,
                match_config,
                anchor_miss_ids=anchors[sample.sample_id],
                reachable_anchor_miss_ids=reachable[sample.sample_id],
                residual_mask=residual,
            )
        )
    return aggregate_evaluations(records)


def select_residual_threshold(
    samples: Sequence[CalibrationSample],
    thresholds: Iterable[float],
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    budget: object,
    *,
    split_role: str = "D_V",
    anchor_miss_ids_by_sample: Mapping[str, frozenset[int]] | None = None,
    reachable_anchor_miss_ids_by_sample: Mapping[str, frozenset[int]] | None = None,
) -> ThresholdSelection:
    """On D_V, maximize total Pd under every preregistered constraint.

    For one fixed anchor, total Pd and net RMR have the same ordering because
    the number of anchor-covered and anchor-missed targets is constant.  Pd is
    nevertheless placed first because it is the standard IRSTD metric; net and
    gross RMR remain diagnostic tie-breakers.  The explicit null candidate is
    residual-off and sorts above every numeric threshold when all metric terms
    tie, avoiding any assumption that numeric ``1.0`` disables the residual.
    """

    _require_validation(split_role)
    candidates: tuple[float | None, ...] = (
        None,
        *_thresholds(thresholds, allow_empty=True),
    )
    fa_budget = _coerce_budget(budget)
    anchors, reachable = _validated_anchor_state(
        samples,
        occupancy_config,
        match_config,
        anchor_miss_ids_by_sample,
        reachable_anchor_miss_ids_by_sample,
    )
    anchor_metrics = evaluate_base_threshold(
        samples,
        occupancy_config.threshold,
        occupancy_config,
        match_config,
        anchor_miss_ids_by_sample=anchors,
        reachable_anchor_miss_ids_by_sample=reachable,
    )
    if not fa_budget.accepts(anchor_metrics):
        raise ValueError(
            "the preregistered FA budget cannot be below fixed anchor occupancy FA"
        )
    feasible: list[tuple[float | None, AggregateEvaluation]] = []
    for threshold in candidates:
        metrics = evaluate_residual_threshold(
            samples,
            threshold,
            occupancy_config,
            match_config,
            anchor_miss_ids_by_sample=anchors,
            reachable_anchor_miss_ids_by_sample=reachable,
        )
        if fa_budget.accepts(metrics):
            feasible.append((threshold, metrics))
    if not feasible:
        return ThresholdSelection(
            threshold=None,
            metrics=None,
            feasible=False,
            reason="no residual threshold satisfies the preregistered FA budget",
        )
    threshold, metrics = max(
        feasible,
        key=lambda item: (
            item[1].pd,
            item[1].net_rmr,
            item[1].gross_rmr,
            float("inf") if item[0] is None else item[0],
        ),
    )
    return ThresholdSelection(threshold, metrics, True)


def select_anchor_threshold_by_miou(
    base_probabilities: Sequence[Tensor],
    gt_masks: Sequence[Tensor],
    thresholds: Iterable[float],
    *,
    split_role: str = "D_V",
) -> float:
    """On D_V, maximize global base-only mIoU; ties choose higher tau_o."""

    _require_validation(split_role)
    if not base_probabilities or len(base_probabilities) != len(gt_masks):
        raise ValueError("base_probabilities and gt_masks must be non-empty and aligned")
    normalized = []
    for probability, gt_mask in zip(base_probabilities, gt_masks):
        probability_2d = _as_probability(probability, name="base_probability")
        gt_2d = _as_gt(gt_mask)
        if probability_2d.shape != gt_2d.shape:
            raise ValueError("base probability and GT grids must be identical")
        normalized.append((probability_2d, gt_2d))

    scored: list[tuple[float, float]] = []
    for threshold in _thresholds(thresholds):
        intersection = 0
        union = 0
        for probability, gt in normalized:
            prediction = probability >= threshold
            intersection += int(torch.count_nonzero(prediction & gt))
            union += int(torch.count_nonzero(prediction | gt))
        miou = intersection / union if union else 1.0
        scored.append((miou, threshold))
    return max(scored, key=lambda item: (item[0], item[1]))[1]


def evaluate_base_threshold(
    samples: Sequence[CalibrationSample],
    threshold: float,
    anchor_occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    *,
    anchor_miss_ids_by_sample: Mapping[str, frozenset[int]] | None = None,
    reachable_anchor_miss_ids_by_sample: Mapping[str, frozenset[int]] | None = None,
) -> AggregateEvaluation:
    """Evaluate Base@B while retaining the fixed tau_o anchor miss set."""

    threshold = float(threshold)
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("base threshold must be finite and in [0,1]")
    if not samples:
        raise ValueError("at least one calibration sample is required")
    anchors, reachable = _validated_anchor_state(
        samples,
        anchor_occupancy_config,
        match_config,
        anchor_miss_ids_by_sample,
        reachable_anchor_miss_ids_by_sample,
    )
    records = []
    seen: set[str] = set()
    for sample in samples:
        base, _, gt = sample.normalized()
        if sample.sample_id in seen:
            raise ValueError(f"duplicate sample_id {sample.sample_id!r}")
        seen.add(sample.sample_id)
        if sample.sample_id not in anchors:
            raise ValueError(f"missing anchor miss set for {sample.sample_id!r}")
        records.append(
            evaluate_binary_prediction(
                base >= threshold,
                gt,
                match_config,
                anchor_miss_ids=anchors[sample.sample_id],
                reachable_anchor_miss_ids=reachable[sample.sample_id],
            )
        )
    return aggregate_evaluations(records)


def select_base_threshold_at_budget(
    samples: Sequence[CalibrationSample],
    thresholds: Iterable[float],
    anchor_occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    budget: object,
    *,
    split_role: str = "D_V",
    anchor_miss_ids_by_sample: Mapping[str, frozenset[int]] | None = None,
    reachable_anchor_miss_ids_by_sample: Mapping[str, frozenset[int]] | None = None,
) -> ThresholdSelection:
    """Select Base@B with the shared matcher, anchor sets, and total budget.

    Base@B is the threshold-relaxation control.  A threshold above the fixed
    anchor threshold can remove anchor pixels and therefore is not a valid
    Base@B candidate, even when it happens to improve component matching.
    """

    _require_validation(split_role)
    tau_o = anchor_occupancy_config.threshold
    candidates = tuple(
        threshold
        for threshold in sorted(
            set((*_thresholds(thresholds), tau_o))
        )
        if threshold <= tau_o
    )
    fa_budget = _coerce_budget(budget)
    anchors, reachable = _validated_anchor_state(
        samples,
        anchor_occupancy_config,
        match_config,
        anchor_miss_ids_by_sample,
        reachable_anchor_miss_ids_by_sample,
    )
    anchor_metrics = evaluate_base_threshold(
        samples,
        anchor_occupancy_config.threshold,
        anchor_occupancy_config,
        match_config,
        anchor_miss_ids_by_sample=anchors,
        reachable_anchor_miss_ids_by_sample=reachable,
    )
    if not fa_budget.accepts(anchor_metrics):
        raise ValueError(
            "the preregistered FA budget cannot be below fixed anchor occupancy FA"
        )
    feasible: list[tuple[float, AggregateEvaluation]] = []
    for threshold in candidates:
        metrics = evaluate_base_threshold(
            samples,
            threshold,
            anchor_occupancy_config,
            match_config,
            anchor_miss_ids_by_sample=anchors,
            reachable_anchor_miss_ids_by_sample=reachable,
        )
        if fa_budget.accepts(metrics):
            feasible.append((threshold, metrics))
    if not feasible:
        return ThresholdSelection(
            threshold=None,
            metrics=None,
            feasible=False,
            reason="no base threshold satisfies the preregistered FA budget",
        )
    threshold, metrics = max(
        feasible,
        key=lambda item: (item[1].pd, item[1].net_rmr, item[0]),
    )
    return ThresholdSelection(threshold, metrics, True)


select_base_threshold_under_budget = select_base_threshold_at_budget


@dataclass(frozen=True)
class FrozenThresholdProtocol:
    """A validation-frozen (tau_o, tau_r) pair for untouched D_T evaluation."""

    occupancy_config: OccupancyConfig
    residual_threshold: float | None
    budget: object | None = None

    def __post_init__(self) -> None:
        if self.residual_threshold is not None:
            value = float(self.residual_threshold)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("residual_threshold must be finite and in [0,1]")
            object.__setattr__(self, "residual_threshold", value)
        if self.budget is not None:
            _coerce_budget(self.budget)

    def evaluate_test(
        self,
        samples: Sequence[CalibrationSample],
        match_config: MatchConfig,
        *,
        proposed_occupancy_threshold: float | None = None,
        proposed_residual_threshold: object = _UNSET,
    ) -> AggregateEvaluation:
        if (
            proposed_occupancy_threshold is not None
            and float(proposed_occupancy_threshold) != self.occupancy_config.threshold
        ):
            raise RuntimeError("test-time occupancy-threshold retuning is forbidden")
        if proposed_residual_threshold is not _UNSET:
            proposed = (
                None
                if proposed_residual_threshold is None
                else float(proposed_residual_threshold)
            )
            if proposed != self.residual_threshold:
                raise RuntimeError("test-time residual-threshold retuning is forbidden")
        metrics = evaluate_residual_threshold(
            samples,
            self.residual_threshold,
            self.occupancy_config,
            match_config,
        )
        violation = self.budget is not None and not _coerce_budget(self.budget).accepts(
            metrics
        )
        return replace(metrics, budget_violation=violation)


@dataclass(frozen=True)
class FrozenBaseThresholdProtocol:
    """Validation-frozen Base@B threshold for untouched D_T evaluation."""

    base_threshold: float
    anchor_occupancy_config: OccupancyConfig
    budget: object | None = None

    def __post_init__(self) -> None:
        value = float(self.base_threshold)
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("base_threshold must be finite and in [0,1]")
        if value > self.anchor_occupancy_config.threshold:
            raise ValueError(
                "Base@B base_threshold must not exceed the anchor occupancy threshold"
            )
        if self.budget is not None:
            _coerce_budget(self.budget)

    def evaluate_test(
        self,
        samples: Sequence[CalibrationSample],
        match_config: MatchConfig,
        *,
        proposed_base_threshold: float | None = None,
    ) -> AggregateEvaluation:
        if (
            proposed_base_threshold is not None
            and float(proposed_base_threshold) != self.base_threshold
        ):
            raise RuntimeError("test-time base-threshold retuning is forbidden")
        metrics = evaluate_base_threshold(
            samples,
            self.base_threshold,
            self.anchor_occupancy_config,
            match_config,
        )
        violation = self.budget is not None and not _coerce_budget(self.budget).accepts(
            metrics
        )
        return replace(metrics, budget_violation=violation)
