"""Immutable value objects used by the CURE-Lite v0.1 core."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class FrozenBaseOutput:
    """Detached output of a frozen dense detector adapter."""

    probability: Tensor
    feature: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.probability, Tensor) or not isinstance(self.feature, Tensor):
            raise TypeError("probability and feature must be tensors")
        if self.probability.ndim != 4 or self.probability.shape[1] != 1:
            raise ValueError("probability must have shape [B,1,H,W]")
        if self.feature.ndim != 4 or self.feature.shape[0] != self.probability.shape[0]:
            raise ValueError("feature must have shape [B,C,h,w] with the same batch size")
        if self.probability.dtype != torch.float32:
            raise TypeError("probability must be float32")
        if not torch.isfinite(self.probability).all():
            raise ValueError("probability contains non-finite values")
        if torch.any((self.probability < 0.0) | (self.probability > 1.0)):
            raise ValueError("probability must lie in [0,1]")
        if self.probability.requires_grad or self.feature.requires_grad:
            raise ValueError("frozen base outputs must be detached")


@dataclass(frozen=True, eq=False)
class Instance:
    """One deterministically numbered connected component."""

    instance_id: int
    mask: Tensor
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]

    def __post_init__(self) -> None:
        if isinstance(self.instance_id, bool) or not isinstance(self.instance_id, int):
            raise TypeError("instance_id must be an integer")
        if self.instance_id < 1:
            raise ValueError("instance_id must be positive")
        if not isinstance(self.mask, Tensor) or self.mask.ndim != 2:
            raise ValueError("instance mask must be a [H,W] tensor")
        if self.mask.device.type != "cpu" or self.mask.dtype != torch.bool:
            raise TypeError("instance mask must be a CPU bool tensor")
        if isinstance(self.area, bool) or not isinstance(self.area, int) or self.area < 1:
            raise ValueError("instance area must be a positive integer")
        if int(torch.count_nonzero(self.mask)) != self.area:
            raise ValueError("instance area does not equal its mask area")
        if len(self.bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in self.bbox):
            raise TypeError("bbox must contain four integer coordinates")
        ymin, xmin, ymax, xmax = self.bbox
        height, width = self.mask.shape
        if not (0 <= ymin < ymax <= height and 0 <= xmin < xmax <= width):
            raise ValueError("bbox is outside the instance mask")
        if len(self.centroid) != 2 or not all(isfinite(float(v)) for v in self.centroid):
            raise ValueError("centroid must contain two finite coordinates")
        coordinates = torch.nonzero(self.mask, as_tuple=False)
        actual_bbox = (
            int(coordinates[:, 0].min()),
            int(coordinates[:, 1].min()),
            int(coordinates[:, 0].max()) + 1,
            int(coordinates[:, 1].max()) + 1,
        )
        if self.bbox != actual_bbox:
            raise ValueError("bbox is inconsistent with the instance mask")
        actual_centroid = (
            float(coordinates[:, 0].to(torch.float64).mean()),
            float(coordinates[:, 1].to(torch.float64).mean()),
        )
        if any(abs(float(given) - actual) > 1e-12 for given, actual in zip(self.centroid, actual_centroid)):
            raise ValueError("centroid is inconsistent with the instance mask")


@dataclass(frozen=True, eq=False)
class InstanceMap:
    """A label image and its component records.

    IDs are contiguous when first constructed, but may contain gaps after a
    legal deletion.  The class therefore never assumes ``id == tuple index``.
    """

    labels: Tensor
    instances: tuple[Instance, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.labels, Tensor) or self.labels.ndim != 2:
            raise ValueError("labels must be a [H,W] tensor")
        if self.labels.device.type != "cpu" or self.labels.dtype != torch.int64:
            raise TypeError("labels must be a CPU int64 tensor")
        if torch.any(self.labels < 0):
            raise ValueError("labels may not be negative")
        if not isinstance(self.instances, tuple):
            raise TypeError("instances must be a tuple")
        ids = tuple(item.instance_id for item in self.instances)
        if len(set(ids)) != len(ids) or ids != tuple(sorted(ids)):
            raise ValueError("instance IDs must be unique and sorted")
        represented = torch.zeros_like(self.labels, dtype=torch.bool)
        for item in self.instances:
            if item.mask.shape != self.labels.shape:
                raise ValueError("instance and label-map shapes differ")
            expected = self.labels == item.instance_id
            if not torch.equal(item.mask, expected):
                raise ValueError("instance mask is inconsistent with labels")
            represented |= item.mask
        if not torch.equal(represented, self.labels > 0):
            raise ValueError("positive labels are not represented by instances")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.labels.shape[0]), int(self.labels.shape[1])

    @property
    def occupancy(self) -> Tensor:
        return self.labels > 0

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(item.instance_id for item in self.instances)

    def by_id(self, instance_id: int) -> Instance:
        for item in self.instances:
            if item.instance_id == instance_id:
                return item
        raise KeyError(f"unknown instance_id={instance_id}")

    def without(self, instance_id: int) -> "InstanceMap":
        """Return a map with one complete component removed, without renumbering."""

        self.by_id(instance_id)
        labels = self.labels.clone()
        labels[labels == instance_id] = 0
        remaining = tuple(item for item in self.instances if item.instance_id != instance_id)
        return InstanceMap(labels=labels, instances=remaining)


@dataclass(frozen=True, order=True)
class MatchPair:
    """One prediction/GT match, ordered by ``(gt_id, pred_id, ...)``."""

    gt_id: int
    pred_id: int
    distance: float
    iou: float

    def __post_init__(self) -> None:
        for name, value in (("gt_id", self.gt_id), ("pred_id", self.pred_id)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isfinite(float(self.distance)) or self.distance < 0.0:
            raise ValueError("distance must be finite and non-negative")
        if not isfinite(float(self.iou)) or not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou must lie in [0,1]")


@dataclass(frozen=True)
class MatchResult:
    """Complete deterministic matching result, including unmatched IDs."""

    pairs: tuple[MatchPair, ...]
    pred_ids: tuple[int, ...]
    gt_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(value, tuple) for value in (self.pairs, self.pred_ids, self.gt_ids)):
            raise TypeError("pairs, pred_ids, and gt_ids must be tuples")
        for name, values in (("pred_ids", self.pred_ids), ("gt_ids", self.gt_ids)):
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
                raise ValueError(f"{name} must contain positive integer IDs")
        if self.pred_ids != tuple(sorted(set(self.pred_ids))):
            raise ValueError("pred_ids must be sorted and unique")
        if self.gt_ids != tuple(sorted(set(self.gt_ids))):
            raise ValueError("gt_ids must be sorted and unique")
        identity = tuple((pair.gt_id, pair.pred_id) for pair in self.pairs)
        if identity != tuple(sorted(identity)) or len(set(identity)) != len(identity):
            raise ValueError("match pairs must be uniquely sorted by (gt_id,pred_id)")
        if len({pair.gt_id for pair in self.pairs}) != len(self.pairs):
            raise ValueError("matching contains a repeated GT")
        if len({pair.pred_id for pair in self.pairs}) != len(self.pairs):
            raise ValueError("matching contains a repeated prediction")
        if any(pair.gt_id not in self.gt_ids or pair.pred_id not in self.pred_ids for pair in self.pairs):
            raise ValueError("matching references an ID outside its instance maps")

    @property
    def matched_pred_ids(self) -> frozenset[int]:
        return frozenset(pair.pred_id for pair in self.pairs)

    @property
    def matched_gt_ids(self) -> frozenset[int]:
        return frozenset(pair.gt_id for pair in self.pairs)

    @property
    def unmatched_pred_ids(self) -> frozenset[int]:
        return frozenset(self.pred_ids) - self.matched_pred_ids

    @property
    def unmatched_gt_ids(self) -> frozenset[int]:
        return frozenset(self.gt_ids) - self.matched_gt_ids

    @property
    def cardinality(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True, eq=False)
class LegalDeletion:
    """A prediction-component deletion that satisfies every legality gate."""

    gt_id: int
    pred_id: int
    occupancy_after: Tensor
    pred_after: InstanceMap
    match_after: MatchResult

    def __post_init__(self) -> None:
        for name, value in (("gt_id", self.gt_id), ("pred_id", self.pred_id)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.occupancy_after, Tensor) or self.occupancy_after.ndim != 2:
            raise ValueError("occupancy_after must be a [H,W] tensor")
        if self.occupancy_after.device.type != "cpu" or self.occupancy_after.dtype != torch.bool:
            raise TypeError("occupancy_after must be a CPU bool tensor")
        if not isinstance(self.pred_after, InstanceMap) or not isinstance(self.match_after, MatchResult):
            raise TypeError("pred_after/match_after have invalid types")
        if tuple(self.occupancy_after.shape) != self.pred_after.shape:
            raise ValueError("occupancy_after and pred_after shapes differ")
        if not torch.equal(self.occupancy_after, self.pred_after.occupancy):
            raise ValueError("occupancy_after is inconsistent with pred_after")
        if self.match_after.pred_ids != tuple(sorted(self.pred_after.ids)):
            raise ValueError("match_after is inconsistent with pred_after")


@dataclass(frozen=True, eq=False)
class BranchSupervision:
    """One factual or synthetic residual-supervision state.

    ``positive_gt_ids`` records only targets that are valid positive training
    supervision in this state.  ``reachable_gt_ids`` records the complete
    individually full-GT-recoverable factual catalog before one target is
    selected.  ``unreachable_gt_ids`` is diagnostic metadata: those IDs remain
    part of the primary miss denominator even though they must not enter either
    factual training pool.

    The metadata fields remain optional for backwards-compatible construction
    of manually assembled training-control states.  The normative builders
    always populate them.  When ``reachable_gt_ids`` is supplied for a
    factual-miss state, it is treated as authoritative and must contain the
    selected positive ID.
    """

    occupancy: Tensor
    target: Tensor
    valid_mask: Tensor
    branch: str
    positive_gt_ids: tuple[int, ...] = ()
    unreachable_gt_ids: tuple[int, ...] = ()
    reachable_gt_ids: tuple[int, ...] = ()

    def validate(self) -> None:
        if not all(isinstance(value, Tensor) for value in (self.occupancy, self.target, self.valid_mask)):
            raise TypeError("occupancy, target, and valid_mask must be tensors")
        if self.occupancy.shape != self.target.shape or self.target.shape != self.valid_mask.shape:
            raise ValueError("occupancy, target, and valid_mask must have identical shapes")
        if self.occupancy.ndim != 3 or self.occupancy.shape[0] != 1:
            raise ValueError("a supervision state must have shape [1,H,W]")
        if self.occupancy.dtype != torch.bool or self.valid_mask.dtype != torch.bool:
            raise TypeError("occupancy and valid_mask must be bool")
        if self.target.dtype != torch.float32:
            raise TypeError("target must be float32")
        if not (self.occupancy.device == self.target.device == self.valid_mask.device):
            raise ValueError("supervision tensors must share a device")
        if not torch.isfinite(self.target).all():
            raise ValueError("target contains non-finite values")
        if torch.any((self.target != 0.0) & (self.target != 1.0)):
            raise ValueError("target must be binary")
        if self.branch not in {
            "factual_miss",
            "factual_no_miss",
            "factual_unreachable",
            "synthetic",
        }:
            raise ValueError(f"unknown supervision branch {self.branch!r}")
        for name, values in (
            ("positive_gt_ids", self.positive_gt_ids),
            ("unreachable_gt_ids", self.unreachable_gt_ids),
            ("reachable_gt_ids", self.reachable_gt_ids),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in values
            ):
                raise ValueError(f"{name} must contain positive integer IDs")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if set(self.positive_gt_ids) & set(self.unreachable_gt_ids):
            raise ValueError("positive and unreachable GT IDs must be disjoint")
        if set(self.reachable_gt_ids) & set(self.unreachable_gt_ids):
            raise ValueError("reachable and unreachable GT IDs must be disjoint")
        positive = self.target.to(torch.bool)
        if torch.any(positive & ~self.valid_mask):
            raise ValueError("positive target lies outside valid_mask")
        if torch.any(self.valid_mask & self.occupancy):
            raise ValueError("valid_mask overlaps occupancy")

        has_target = bool(torch.any(positive))
        has_valid = bool(torch.any(self.valid_mask))
        if self.branch == "factual_miss":
            if not has_target:
                raise ValueError("factual_miss requires a non-empty positive target")
            if self.reachable_gt_ids:
                if len(self.positive_gt_ids) != 1:
                    raise ValueError(
                        "factual_miss with reachable metadata requires exactly one "
                        "selected positive GT"
                    )
                if not set(self.positive_gt_ids) <= set(self.reachable_gt_ids):
                    raise ValueError(
                        "selected factual positive must belong to reachable_gt_ids"
                    )
        elif self.branch == "factual_no_miss":
            if has_target:
                raise ValueError("factual_no_miss requires an empty target")
            if self.positive_gt_ids or self.reachable_gt_ids or self.unreachable_gt_ids:
                raise ValueError("factual_no_miss cannot carry miss GT metadata")
        elif self.branch == "factual_unreachable":
            if has_target or has_valid:
                raise ValueError(
                    "factual_unreachable must be diagnostics-only with empty target "
                    "and valid_mask"
                )
            if self.positive_gt_ids or self.reachable_gt_ids:
                raise ValueError(
                    "factual_unreachable cannot carry positive or reachable GT IDs"
                )
            if not self.unreachable_gt_ids:
                raise ValueError(
                    "factual_unreachable requires at least one unreachable GT ID"
                )
        else:  # synthetic
            if not has_target:
                raise ValueError("synthetic supervision requires a non-empty target")
            if self.positive_gt_ids and len(self.positive_gt_ids) != 1:
                raise ValueError(
                    "synthetic supervision permits exactly one positive GT when "
                    "metadata is supplied"
                )
            if self.reachable_gt_ids or self.unreachable_gt_ids:
                raise ValueError(
                    "synthetic supervision cannot carry factual reachability metadata"
                )

    def __post_init__(self) -> None:
        self.validate()
