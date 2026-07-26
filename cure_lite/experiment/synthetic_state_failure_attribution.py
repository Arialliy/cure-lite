"""D_R-only predictive attribution for failed synthetic-state support.

This module is intentionally diagnostic.  It separates variables that are
defined for both factual and legal targets from a legal-only intervention
ledger, and it never constructs a transformed state, a sampling distribution,
or a training example.

The statistical outputs describe *predictive separability*.  Because target
role is selected by the frozen Base detector and several blocks are dependent
on the same Base output, no result from this module identifies an independent
causal contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from math import hypot, isfinite, log, log1p
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..instances import instances_from_binary_mask
from ..splits import SplitManifest
from .cache_pipeline import LoadedDRCacheBundle
from .geometry_safe_catalog import GeometrySafeTrainingCatalogView
from .p0_protocol import P0OverlapConfig, P0SeparabilityConfig
from .p0_support import (
    _background_ring,
    _clipped_logit,
    _coverage_receipt,
    _feature_moments,
    _fit_feature_projector,
    _fit_logistic_irls,
    _fold_assignment,
    _group_balanced_weights,
    _higher_quantile,
    _mmd_receipt,
    _nearest_component_centroid_distance,
    _occupancy_patch,
    _project_feature,
    _robust_scale,
    _robust_scale_fit,
    _weighted_auc,
)
from .training_pipeline import PreparedTrainingCatalog


FAILURE_ATTRIBUTION_SCHEMA = "cure-lite-synthetic-state-failure-attribution-v1"
BLOCK_OOF_SCHEMA = "cure-lite-failure-attribution-block-oof-v1"
COMPOSITE_OOF_SCHEMA = "cure-lite-failure-attribution-composite-oof-v1"
BLOCK_SUPPORT_SCHEMA = "cure-lite-failure-attribution-block-support-v1"
SHARED_GROUP_SCHEMA = "cure-lite-failure-attribution-shared-group-v1"
SAME_SOURCE_SCHEMA = "cure-lite-failure-attribution-same-source-v1"

COMMON_BLOCKS = (
    "G_full",
    "W",
    "P",
    "F_local",
    "F_background_global",
    "O",
)
FEATURE_BLOCKS = frozenset({"F_local", "F_background_global"})

_INTERPRETATION = (
    "predictive-non-causal; correlated blocks and Base-defined role selection "
    "preclude independent causal attribution"
)


def block_definition(block: str) -> dict[str, object]:
    """Return the frozen field order and measurement support for one block."""

    definitions: dict[str, dict[str, object]] = {
        "G_full": {
            "support": "full-GT",
            "fields_in_order": [
                "log1p_gt_area",
                "log_gt_aspect_ratio",
                "gt_bbox_fill_fraction",
                "border_distance_normalized",
                "gt_centroid_y_normalized",
                "gt_centroid_x_normalized",
            ],
        },
        "W": {
            "support": "writable-supervision-mask",
            "fields_in_order": [
                "log1p_supervision_area",
                "supervision_fraction_of_gt",
                "supervision_component_count",
                "supervision_to_gt_centroid_distance_normalized",
            ],
            "interpretation": "occupancy/intervention-derived; not pure geometry",
        },
        "P": {
            "support": "full-GT-and-fixed-background-ring",
            "fields_in_order": [
                "clipped_logit_base_gt_mean",
                "clipped_logit_base_ring_mean",
                "base_gt_std",
                "base_ring_std",
                "clipped_logit_base_gt_max",
                "clipped_logit_base_ring_max",
                "base_gt_minus_ring_mean",
            ],
            "decoder_input": False,
        },
        "F_local": {
            "support": "full-GT-and-fixed-background-ring",
            "layout_in_order": [
                "target_channel_mean[C]",
                "target_channel_std[C]",
                "target_minus_ring_channel_mean[C]",
                "target_rms[1]",
            ],
            "dimensions": "3*C+1",
            "interpretation": (
                "overlaps F_background_global through ring-derived terms; "
                "not a mutually exclusive factor"
            ),
        },
        "F_background_global": {
            "support": "fixed-background-ring-and-whole-feature-grid",
            "layout_in_order": [
                "ring_channel_mean[C]",
                "ring_channel_std[C]",
                "global_channel_mean[C]",
                "global_channel_std[C]",
            ],
            "dimensions": "4*C",
            "interpretation": (
                "overlaps F_local through ring-derived terms; not a "
                "mutually exclusive factor"
            ),
        },
        "O": {
            "support": "full-GT-fixed-ring-and-projected-conditioning-occupancy",
            "layout_in_order": [
                "conditioning_gt_fraction[1]",
                "conditioning_ring_fraction[1]",
                "nearest_component_centroid_distance_normalized[1]",
                "projected_local_patch_row_major[(2*r+1)^2]",
                "projected_global_fraction[1]",
            ],
            "dimensions": "4+(2*r+1)^2",
        },
    }
    if block not in definitions:
        raise ValueError(f"block must be one of {COMMON_BLOCKS}")
    return dict(definitions[block])


def _authority() -> dict[str, bool]:
    return {
        "authorizes_transformation": False,
        "authorizes_candidate_s": False,
        "authorizes_p0_d": False,
        "authorizes_training": False,
        "authorizes_calibration": False,
        "authorizes_inference": False,
        "authorizes_d_v_access": False,
        "authorizes_d_t_access": False,
        "authorizes_full_cure": False,
        "authorizes_other_backbone": False,
    }


def _fingerprinted(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["receipt_fingerprint"] = stable_fingerprint(result)
    return result


def _tensor_fingerprint(value: Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _vector(value: Tensor, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if tensor.ndim != 1 or tensor.numel() < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains a non-finite value")
    return tensor.detach().clone().contiguous()


def _identity_key(
    identity: tuple[str, int, int | None],
) -> tuple[str, int, int]:
    return identity[0], identity[1], -1 if identity[2] is None else identity[2]


@dataclass(frozen=True, eq=False)
class CommonStateRecord:
    """One common-role observation with strictly separated variable blocks."""

    identity: tuple[str, int, int | None]
    sample_id: str
    group_id: str
    role: str
    G_full: Tensor
    W: Tensor
    P: Tensor
    F_local: Tensor
    F_background_global: Tensor
    O: Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, tuple)
            or len(self.identity) != 3
            or self.identity[0] != self.sample_id
        ):
            raise ValueError("identity must be (sample_id, gt_id, pred_id)")
        _, gt_id, pred_id = self.identity
        if isinstance(gt_id, bool) or not isinstance(gt_id, int) or gt_id < 1:
            raise ValueError("identity gt_id must be a positive integer")
        if pred_id is not None and (
            isinstance(pred_id, bool)
            or not isinstance(pred_id, int)
            or pred_id < 1
        ):
            raise ValueError("identity pred_id must be positive or None")
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id
            or not isinstance(self.group_id, str)
            or not self.group_id
        ):
            raise ValueError("sample_id and group_id must be non-empty")
        if self.role not in {"factual", "legal"}:
            raise ValueError("role must be factual or legal")
        if self.role == "factual" and pred_id is not None:
            raise ValueError("factual identity cannot carry pred_id")
        if self.role == "legal" and pred_id is None:
            raise ValueError("legal identity requires pred_id")
        for block in COMMON_BLOCKS:
            object.__setattr__(
                self,
                block,
                _vector(getattr(self, block), name=block),
            )
        fixed_dimensions = {
            "G_full": 6,
            "W": 4,
            "P": 7,
            "O": 29,
        }
        for block, expected in fixed_dimensions.items():
            if self.block(block).numel() != expected:
                raise ValueError(
                    f"{block} must have the frozen dimension {expected}"
                )
        local_dimensions = int(self.F_local.numel())
        if (local_dimensions - 1) % 3:
            raise ValueError("F_local must have dimension 3*C+1")
        channels = (local_dimensions - 1) // 3
        if channels < 1 or self.F_background_global.numel() != 4 * channels:
            raise ValueError(
                "F_local and F_background_global do not share one channel count"
            )

    def block(self, name: str) -> Tensor:
        if name not in COMMON_BLOCKS:
            raise ValueError(f"block must be one of {COMMON_BLOCKS}")
        return getattr(self, name)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "identity": list(self.identity),
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "role": self.role,
            "blocks": {
                block: [float(value) for value in self.block(block).tolist()]
                for block in COMMON_BLOCKS
            },
        }


@dataclass(frozen=True)
class LegalOccupancyLedger:
    """Legal-only pre/post deletion facts, excluded from common-role models."""

    identity: tuple[str, int, int]
    sample_id: str
    group_id: str
    before_target_fraction: float
    after_target_fraction: float
    before_ring_fraction: float
    after_ring_fraction: float
    removed_target_fraction: float
    removed_pixels: int
    projected_changed_cells: int
    deletion_equals_frozen_pred_component: bool
    source_feature_is_synthetic_feature: bool
    before_occupancy_fingerprint: str
    after_occupancy_fingerprint: str
    deletion_mask_fingerprint: str
    source_feature_fingerprint: str
    synthetic_feature_fingerprint: str
    target_fingerprint: str
    supervision_fingerprint: str
    valid_mask_fingerprint: str

    def __post_init__(self) -> None:
        if (
            len(self.identity) != 3
            or self.identity[0] != self.sample_id
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.identity[1:]
            )
        ):
            raise ValueError("legal ledger identity is invalid")
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id
            or not isinstance(self.group_id, str)
            or not self.group_id
        ):
            raise ValueError("legal ledger sample_id/group_id must be non-empty")
        for name in (
            "before_target_fraction",
            "after_target_fraction",
            "before_ring_fraction",
            "after_ring_fraction",
            "removed_target_fraction",
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if self.removed_pixels < 1 or self.projected_changed_cells < 1:
            raise ValueError("legal deletion must change occupancy")
        if self.deletion_equals_frozen_pred_component is not True:
            raise ValueError("deletion must equal the frozen prediction component")
        if self.source_feature_is_synthetic_feature is not True:
            raise ValueError("synthetic feature must preserve source object identity")
        if self.source_feature_fingerprint != self.synthetic_feature_fingerprint:
            raise ValueError("source and synthetic feature fingerprints differ")
        for field in (
            "before_occupancy_fingerprint",
            "after_occupancy_fingerprint",
            "deletion_mask_fingerprint",
            "source_feature_fingerprint",
            "synthetic_feature_fingerprint",
            "target_fingerprint",
            "supervision_fingerprint",
            "valid_mask_fingerprint",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA256 digest")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "identity": list(self.identity),
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "before_target_fraction": self.before_target_fraction,
            "after_target_fraction": self.after_target_fraction,
            "before_ring_fraction": self.before_ring_fraction,
            "after_ring_fraction": self.after_ring_fraction,
            "removed_target_fraction": self.removed_target_fraction,
            "removed_pixels": self.removed_pixels,
            "projected_changed_cells": self.projected_changed_cells,
            "deletion_equals_frozen_pred_component": (
                self.deletion_equals_frozen_pred_component
            ),
            "source_feature_is_synthetic_feature": (
                self.source_feature_is_synthetic_feature
            ),
            "before_occupancy_fingerprint": self.before_occupancy_fingerprint,
            "after_occupancy_fingerprint": self.after_occupancy_fingerprint,
            "deletion_mask_fingerprint": self.deletion_mask_fingerprint,
            "source_feature_fingerprint": self.source_feature_fingerprint,
            "synthetic_feature_fingerprint": (
                self.synthetic_feature_fingerprint
            ),
            "target_fingerprint": self.target_fingerprint,
            "supervision_fingerprint": self.supervision_fingerprint,
            "valid_mask_fingerprint": self.valid_mask_fingerprint,
            "classifier_eligible": False,
        }


@dataclass(frozen=True)
class PopulationExpectation:
    factual_targets: int = 32
    legal_targets: int = 206

    def __post_init__(self) -> None:
        if self.factual_targets < 1 or self.legal_targets < 1:
            raise ValueError("population expectations must be positive")


@dataclass(frozen=True)
class SameSourceExpectation:
    sources: int = 14
    factual_targets: int = 18
    legal_targets: int = 21

    def __post_init__(self) -> None:
        if min(self.sources, self.factual_targets, self.legal_targets) < 1:
            raise ValueError("same-source expectations must be positive")


@dataclass(frozen=True)
class SharedGroupExpectation:
    groups: int = 14
    factual_targets: int = 18
    legal_targets: int = 25

    def __post_init__(self) -> None:
        if min(self.groups, self.factual_targets, self.legal_targets) < 1:
            raise ValueError("shared-group expectations must be positive")


@dataclass(frozen=True)
class FailureAttributionPopulation:
    common_records: tuple[CommonStateRecord, ...]
    legal_occupancy_ledger: tuple[LegalOccupancyLedger, ...]
    split: str = "D_R"

    def __post_init__(self) -> None:
        if self.split != "D_R":
            raise ValueError("failure attribution permits only D_R")
        records = _canonical_records(self.common_records)
        if records != self.common_records:
            raise ValueError("common records must be in canonical identity order")
        ledgers = tuple(
            sorted(
                self.legal_occupancy_ledger,
                key=lambda item: _identity_key(item.identity),
            )
        )
        if ledgers != self.legal_occupancy_ledger:
            raise ValueError("legal occupancy ledger must be canonically ordered")
        legal_identities = {
            item.identity for item in self.common_records if item.role == "legal"
        }
        ledger_identities = [item.identity for item in ledgers]
        if (
            len(ledger_identities) != len(set(ledger_identities))
            or len(ledger_identities) != len(legal_identities)
            or set(ledger_identities) != legal_identities
        ):
            raise ValueError("legal occupancy ledger must align one-to-one with legal records")

    def canonical_payload(self) -> dict[str, object]:
        factual = sum(item.role == "factual" for item in self.common_records)
        legal = len(self.common_records) - factual
        dimensions = {
            block: int(self.common_records[0].block(block).numel())
            for block in COMMON_BLOCKS
        }
        feature_channels = (dimensions["F_local"] - 1) // 3
        return {
            "schema_version": FAILURE_ATTRIBUTION_SCHEMA,
            "split": self.split,
            "counts": {
                "factual_targets": factual,
                "legal_targets": legal,
                "legal_occupancy_ledger": len(self.legal_occupancy_ledger),
            },
            "common_blocks": list(COMMON_BLOCKS),
            "realized_dimensions": {
                "blocks": dimensions,
                "feature_channels": feature_channels,
                "occupancy_patch_radius": 2,
            },
            "block_definitions": {
                block: block_definition(block) for block in COMMON_BLOCKS
            },
            "measurement_contract": {
                "support_by_block": {
                    block: block_definition(block)["support"]
                    for block in COMMON_BLOCKS
                },
                "writable_geometry_isolated_as": "W",
                "legal_pre_post_occupancy": (
                    "separate-ledger-never-role-classifier-input"
                ),
            },
            "interpretation": _INTERPRETATION,
            "authority": _authority(),
            "common_records": [
                item.canonical_payload() for item in self.common_records
            ],
            "legal_occupancy_ledger": [
                item.canonical_payload() for item in self.legal_occupancy_ledger
            ],
        }

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _mask2d(value: Tensor, *, name: str, shape: tuple[int, int] | None = None) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.bool, device="cpu")
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or (shape is not None and tuple(tensor.shape) != shape):
        raise ValueError(f"{name} must be a two-dimensional mask on the common grid")
    return tensor.contiguous()


def _probability2d(value: Tensor, shape: tuple[int, int]) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if tensor.ndim == 4 and tensor.shape[:2] == (1, 1):
        tensor = tensor[0, 0]
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or tuple(tensor.shape) != shape:
        raise ValueError("probability must use the common target grid")
    if not torch.isfinite(tensor).all() or torch.any((tensor < 0.0) | (tensor > 1.0)):
        raise ValueError("probability must be finite and lie in [0,1]")
    return tensor.contiguous()


def _fraction(mask: Tensor, support: Tensor) -> float:
    denominator = int(torch.count_nonzero(support))
    if denominator < 1:
        raise RuntimeError("fraction support is empty")
    return float(torch.count_nonzero(mask & support)) / denominator


def build_common_state_record(
    *,
    sample_id: str,
    group_id: str,
    role: str,
    gt_id: int,
    pred_id: int | None,
    target_mask: Tensor,
    supervision_mask: Tensor,
    conditioning_occupancy: Tensor,
    probability: Tensor,
    feature: Tensor,
    gt_labels: Tensor,
    valid_mask: Tensor,
    overlap: P0OverlapConfig,
) -> CommonStateRecord:
    """Build full-GT-aligned common blocks for one factual or legal target."""

    if not isinstance(overlap, P0OverlapConfig):
        raise TypeError("overlap must be P0OverlapConfig")
    gt = _mask2d(target_mask, name="target_mask")
    shape = tuple(int(value) for value in gt.shape)
    supervision = _mask2d(
        supervision_mask,
        name="supervision_mask",
        shape=shape,
    )
    occupancy = _mask2d(
        conditioning_occupancy,
        name="conditioning_occupancy",
        shape=shape,
    )
    labels = torch.as_tensor(gt_labels, dtype=torch.int64, device="cpu")
    valid = _mask2d(valid_mask, name="valid_mask", shape=shape)
    p = _probability2d(probability, shape)
    if labels.ndim != 2 or tuple(labels.shape) != shape:
        raise ValueError("gt_labels must use the common target grid")
    if not torch.any(gt) or torch.any(supervision & ~gt):
        raise ValueError("target must be nonempty and contain supervision")
    if torch.any(gt & ~valid):
        raise ValueError("target extends outside valid_mask")

    ring = _background_ring(
        gt,
        labels,
        valid,
        inner=overlap.ring_inner_radius,
        outer=overlap.ring_outer_radius,
    )
    coordinates = torch.nonzero(gt, as_tuple=False).to(torch.float64)
    area = int(coordinates.shape[0])
    ymin = int(coordinates[:, 0].min())
    xmin = int(coordinates[:, 1].min())
    ymax = int(coordinates[:, 0].max()) + 1
    xmax = int(coordinates[:, 1].max()) + 1
    height, width = shape
    centroid = (
        float(coordinates[:, 0].mean()),
        float(coordinates[:, 1].mean()),
    )
    bbox_area = (ymax - ymin) * (xmax - xmin)
    border = min(ymin, xmin, height - ymax, width - xmax) / max(height, width)
    G_full = torch.tensor(
        (
            log1p(area),
            log((ymax - ymin) / (xmax - xmin)),
            area / bbox_area,
            border,
            (centroid[0] + 0.5) / height,
            (centroid[1] + 0.5) / width,
        ),
        dtype=torch.float64,
    )

    supervision_coordinates = torch.nonzero(supervision, as_tuple=False)
    if not supervision_coordinates.numel():
        raise ValueError("common target supervision must be nonempty")
    supervision_centroid = (
        float(supervision_coordinates[:, 0].to(torch.float64).mean()),
        float(supervision_coordinates[:, 1].to(torch.float64).mean()),
    )
    supervision_instances = instances_from_binary_mask(
        supervision,
        connectivity=8,
        min_area=1,
    )
    supervision_area = int(supervision_coordinates.shape[0])
    W = torch.tensor(
        (
            log1p(supervision_area),
            supervision_area / area,
            float(len(supervision_instances.instances)),
            hypot(
                supervision_centroid[0] - centroid[0],
                supervision_centroid[1] - centroid[1],
            )
            / hypot(height, width),
        ),
        dtype=torch.float64,
    )

    target_p = p[gt]
    ring_p = p[ring]
    target_p_mean = float(target_p.mean())
    ring_p_mean = float(ring_p.mean())
    P = torch.tensor(
        (
            _clipped_logit(target_p_mean, overlap.probability_clip),
            _clipped_logit(ring_p_mean, overlap.probability_clip),
            float(target_p.std(unbiased=False)),
            float(ring_p.std(unbiased=False)),
            _clipped_logit(float(target_p.max()), overlap.probability_clip),
            _clipped_logit(float(ring_p.max()), overlap.probability_clip),
            target_p_mean - ring_p_mean,
        ),
        dtype=torch.float64,
    )

    target_mean, target_std, target_rms = _feature_moments(feature, gt)
    ring_mean, ring_std, _ = _feature_moments(feature, ring)
    feature64 = torch.as_tensor(feature, dtype=torch.float64, device="cpu")
    if feature64.ndim != 4 or feature64.shape[0] != 1:
        raise ValueError("feature must have shape [1,C,h,w]")
    global_mean = feature64[0].mean(dim=(1, 2))
    global_std = feature64[0].std(dim=(1, 2), unbiased=False)
    F_local = torch.cat(
        (
            target_mean,
            target_std,
            target_mean - ring_mean,
            torch.tensor((target_rms,), dtype=torch.float64),
        )
    )
    F_background_global = torch.cat(
        (ring_mean, ring_std, global_mean, global_std)
    )

    patch = _occupancy_patch(
        occupancy,
        centroid,
        tuple(int(value) for value in feature64.shape[-2:]),
        overlap.joint_occupancy_patch_radius,
    )
    O = torch.cat(
        (
            torch.tensor(
                (
                    _fraction(occupancy, gt),
                    _fraction(occupancy, ring),
                    _nearest_component_centroid_distance(centroid, occupancy),
                ),
                dtype=torch.float64,
            ),
            patch,
        )
    )
    return CommonStateRecord(
        identity=(sample_id, gt_id, pred_id),
        sample_id=sample_id,
        group_id=group_id,
        role=role,
        G_full=G_full,
        W=W,
        P=P,
        F_local=F_local,
        F_background_global=F_background_global,
        O=O,
    )


def build_legal_occupancy_ledger(
    *,
    identity: tuple[str, int, int],
    group_id: str,
    target_mask: Tensor,
    ring_mask: Tensor,
    before_occupancy: Tensor,
    after_occupancy: Tensor,
    pred_labels: Tensor,
    source_feature: Tensor,
    synthetic_feature: Tensor,
    supervision_mask: Tensor,
    valid_mask: Tensor,
    feature_size: tuple[int, int],
) -> LegalOccupancyLedger:
    """Build a legal-only deletion ledger that cannot enter block models."""

    target = _mask2d(target_mask, name="target_mask")
    shape = tuple(int(value) for value in target.shape)
    ring = _mask2d(ring_mask, name="ring_mask", shape=shape)
    before = _mask2d(before_occupancy, name="before_occupancy", shape=shape)
    after = _mask2d(after_occupancy, name="after_occupancy", shape=shape)
    supervision = _mask2d(
        supervision_mask,
        name="supervision_mask",
        shape=shape,
    )
    valid = _mask2d(valid_mask, name="valid_mask", shape=shape)
    labels = torch.as_tensor(pred_labels, dtype=torch.int64, device="cpu")
    if labels.ndim != 2 or tuple(labels.shape) != shape:
        raise ValueError("pred_labels must use the occupancy grid")
    if synthetic_feature is not source_feature:
        raise ValueError("synthetic feature must be the exact source feature object")
    if not torch.equal(synthetic_feature, source_feature):
        raise ValueError("synthetic and source feature tensor values differ")
    if torch.any(after & ~before):
        raise ValueError("legal post-delete occupancy cannot add pixels")
    removed = before & ~after
    if not torch.any(removed):
        raise ValueError("legal occupancy ledger requires a nonempty deletion")
    frozen_component = labels == identity[2]
    deletion_exact = torch.equal(removed, frozen_component)
    if not deletion_exact:
        raise ValueError(
            "before & ~after does not equal the frozen prediction component"
        )
    from ..decoder import project_occupancy_to_feature_grid

    before_small = project_occupancy_to_feature_grid(
        before.unsqueeze(0).unsqueeze(0),
        feature_size,
    )
    after_small = project_occupancy_to_feature_grid(
        after.unsqueeze(0).unsqueeze(0),
        feature_size,
    )
    changed = int(torch.count_nonzero(before_small ^ after_small))
    return LegalOccupancyLedger(
        identity=identity,
        sample_id=identity[0],
        group_id=group_id,
        before_target_fraction=_fraction(before, target),
        after_target_fraction=_fraction(after, target),
        before_ring_fraction=_fraction(before, ring),
        after_ring_fraction=_fraction(after, ring),
        removed_target_fraction=_fraction(removed, target),
        removed_pixels=int(torch.count_nonzero(removed)),
        projected_changed_cells=changed,
        deletion_equals_frozen_pred_component=True,
        source_feature_is_synthetic_feature=True,
        before_occupancy_fingerprint=_tensor_fingerprint(before),
        after_occupancy_fingerprint=_tensor_fingerprint(after),
        deletion_mask_fingerprint=_tensor_fingerprint(removed),
        source_feature_fingerprint=_tensor_fingerprint(source_feature),
        synthetic_feature_fingerprint=_tensor_fingerprint(synthetic_feature),
        target_fingerprint=_tensor_fingerprint(target),
        supervision_fingerprint=_tensor_fingerprint(supervision),
        valid_mask_fingerprint=_tensor_fingerprint(valid),
    )


def build_failure_attribution_population(
    bundle: LoadedDRCacheBundle,
    catalog: PreparedTrainingCatalog | GeometrySafeTrainingCatalogView,
    manifest: SplitManifest,
    overlap: P0OverlapConfig,
    *,
    expectation: PopulationExpectation = PopulationExpectation(),
) -> FailureAttributionPopulation:
    """Build the exact geometry-safe D_R factual/legal analysis population."""

    if not isinstance(bundle, LoadedDRCacheBundle) or bundle.split != "D_R":
        raise TypeError("bundle must be a strictly loaded D_R cache bundle")
    if not isinstance(
        catalog,
        (PreparedTrainingCatalog, GeometrySafeTrainingCatalogView),
    ):
        raise TypeError(
            "catalog must be PreparedTrainingCatalog or "
            "GeometrySafeTrainingCatalogView"
        )
    if not isinstance(manifest, SplitManifest):
        raise TypeError("manifest must be SplitManifest")
    if not isinstance(expectation, PopulationExpectation):
        raise TypeError("expectation must be PopulationExpectation")
    bundle.verify_unchanged()
    group_by_sample = {
        item.sample_id: item.group_id for item in manifest.records_for("D_R")
    }
    if set(group_by_sample) != set(catalog.source_ids):
        raise RuntimeError("manifest D_R membership differs from prepared catalog")
    row_by_sample = {row.sample_id: row for row in bundle.rows}
    if set(row_by_sample) != set(catalog.source_ids):
        raise RuntimeError("D_R bundle membership differs from prepared catalog")

    records: list[CommonStateRecord] = []
    ledgers: list[LegalOccupancyLedger] = []
    for entry in catalog.entries:
        row = row_by_sample[entry.sample_id]
        group_id = group_by_sample[entry.sample_id]
        gt = entry.gt
        for gt_id, example in zip(
            entry.reachable_gt_ids,
            entry.factual_examples,
            strict=True,
        ):
            records.append(
                build_common_state_record(
                    sample_id=entry.sample_id,
                    group_id=group_id,
                    role="factual",
                    gt_id=gt_id,
                    pred_id=None,
                    target_mask=gt.by_id(gt_id).mask,
                    supervision_mask=example.supervision.target[0] > 0,
                    conditioning_occupancy=example.supervision.occupancy[0],
                    probability=row.base_output.probability,
                    feature=row.base_output.feature,
                    gt_labels=row.state.gt_labels,
                    valid_mask=row.state.image_valid_mask,
                    overlap=overlap,
                )
            )
        for candidate, example in zip(
            entry.decoder_visible_legal_candidates,
            entry.synthetic_examples,
            strict=True,
        ):
            target = gt.by_id(candidate.gt_id).mask
            record = build_common_state_record(
                sample_id=entry.sample_id,
                group_id=group_id,
                role="legal",
                gt_id=candidate.gt_id,
                pred_id=candidate.pred_id,
                target_mask=target,
                supervision_mask=example.supervision.target[0] > 0,
                conditioning_occupancy=example.supervision.occupancy[0],
                probability=row.base_output.probability,
                feature=row.base_output.feature,
                gt_labels=row.state.gt_labels,
                valid_mask=row.state.image_valid_mask,
                overlap=overlap,
            )
            ring = _background_ring(
                target,
                row.state.gt_labels,
                row.state.image_valid_mask,
                inner=overlap.ring_inner_radius,
                outer=overlap.ring_outer_radius,
            )
            identity = (
                entry.sample_id,
                candidate.gt_id,
                candidate.pred_id,
            )
            records.append(record)
            ledgers.append(
                build_legal_occupancy_ledger(
                    identity=identity,
                    group_id=group_id,
                    target_mask=target,
                    ring_mask=ring,
                    before_occupancy=row.state.occupancy,
                    after_occupancy=example.supervision.occupancy[0],
                    pred_labels=row.state.pred_labels,
                    source_feature=entry.source.feature,
                    synthetic_feature=example.feature,
                    supervision_mask=example.supervision.target[0] > 0,
                    valid_mask=example.supervision.valid_mask[0],
                    feature_size=tuple(row.base_output.feature.shape[-2:]),
                )
            )
    ordered = _canonical_records(tuple(records))
    factual_count = sum(item.role == "factual" for item in ordered)
    legal_count = len(ordered) - factual_count
    if (factual_count, legal_count) != (
        expectation.factual_targets,
        expectation.legal_targets,
    ):
        raise RuntimeError(
            "failure-attribution population differs from the frozen expectation: "
            f"{factual_count}/{legal_count} != "
            f"{expectation.factual_targets}/{expectation.legal_targets}"
        )
    population = FailureAttributionPopulation(
        common_records=ordered,
        legal_occupancy_ledger=tuple(
            sorted(ledgers, key=lambda item: _identity_key(item.identity))
        ),
    )
    bundle.verify_unchanged()
    return population


def _canonical_records(
    records: Sequence[CommonStateRecord],
) -> tuple[CommonStateRecord, ...]:
    values = tuple(records)
    if not values or any(not isinstance(item, CommonStateRecord) for item in values):
        raise TypeError("records must contain CommonStateRecord values")
    identities = [item.identity for item in values]
    if len(identities) != len(set(identities)):
        raise ValueError("common attribution identities must be unique")
    dimensions = {
        block: {int(item.block(block).numel()) for item in values}
        for block in COMMON_BLOCKS
    }
    inconsistent = [block for block, sizes in dimensions.items() if len(sizes) != 1]
    if inconsistent:
        raise ValueError(f"block dimensions differ across records: {inconsistent}")
    return tuple(sorted(values, key=lambda item: _identity_key(item.identity)))


def _project_fold_block(
    records: tuple[CommonStateRecord, ...],
    train: Sequence[int],
    test: Sequence[int],
    block: str,
    feature_components: int,
) -> tuple[Tensor, Tensor, dict[str, object] | None]:
    train_raw = torch.stack([records[index].block(block) for index in train])
    test_raw = torch.stack([records[index].block(block) for index in test])
    if block not in FEATURE_BLOCKS:
        return train_raw, test_raw, None
    legal_train = [index for index in train if records[index].role == "legal"]
    if not legal_train:
        raise RuntimeError("feature projection fold has no legal training targets")
    projector = _fit_feature_projector(
        torch.stack([records[index].block(block) for index in legal_train]),
        feature_components,
    )
    fit_population = [
        {
            "identity": list(records[index].identity),
            "group_id": records[index].group_id,
        }
        for index in legal_train
    ]
    parameters = {
        "raw_median": [
            float(value) for value in projector.raw_median.tolist()
        ],
        "raw_scale": [
            float(value) for value in projector.raw_scale.tolist()
        ],
        "raw_constant_floor_dimensions": [
            index
            for index, flag in enumerate(projector.raw_constant.tolist())
            if flag
        ],
        "raw_maxdev_fallback_dimensions": [
            index
            for index, flag in enumerate(
                projector.raw_maxdev_fallback.tolist()
            )
            if flag
        ],
        "pca_mean": [
            float(value) for value in projector.pca_mean.tolist()
        ],
        "basis": [
            [float(value) for value in row]
            for row in projector.basis.tolist()
        ],
        "singular_values": [
            float(value) for value in projector.singular_values.tolist()
        ],
    }
    audit = {
        "fit_role": "training-fold-legal-targets-only",
        "fit_targets": len(legal_train),
        "fit_groups": len({records[index].group_id for index in legal_train}),
        "components": feature_components,
        "fit_population_fingerprint": stable_fingerprint(fit_population),
        "parameters": parameters,
        "parameter_fingerprint": stable_fingerprint(parameters),
    }
    return (
        _project_feature(train_raw, projector),
        _project_feature(test_raw, projector),
        audit,
    )


def _fixed_oof_group_bootstrap(
    scores: Sequence[float],
    labels: Sequence[int],
    groups: Sequence[str],
    config: P0SeparabilityConfig,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.bootstrap_seed)
    unique_groups = sorted(set(groups))
    by_group = {
        group: [
            index for index, candidate in enumerate(groups) if candidate == group
        ]
        for group in unique_groups
    }
    values: list[float] = []
    skipped = 0
    for _ in range(config.bootstrap_replicates):
        sampled = torch.randint(
            len(unique_groups),
            (len(unique_groups),),
            generator=generator,
        ).tolist()
        indices: list[int] = []
        occurrence_groups: list[str] = []
        for occurrence, group_index in enumerate(sampled):
            group = unique_groups[group_index]
            indices.extend(by_group[group])
            occurrence_groups.extend(
                [f"{group}#{occurrence}"] * len(by_group[group])
            )
        selected_labels = [labels[index] for index in indices]
        if len(set(selected_labels)) < 2:
            skipped += 1
            continue
        weights = _group_balanced_weights(selected_labels, occurrence_groups)
        values.append(
            _weighted_auc(
                [scores[index] for index in indices],
                selected_labels,
                weights,
            )
        )
    if not values:
        raise RuntimeError("all grouped AUC bootstrap replicates were uninformative")
    lower_probability, upper_probability = config.bootstrap_interval
    return {
        "interpretation": config.bootstrap_interpretation,
        "requested_replicates": config.bootstrap_replicates,
        "valid_replicates": len(values),
        "skipped_replicates": skipped,
        "interval_probability": [lower_probability, upper_probability],
        "lower": _higher_quantile(values, lower_probability),
        "upper": _higher_quantile(values, upper_probability),
    }


def run_composite_group_oof(
    records: Sequence[CommonStateRecord],
    *,
    blocks: Sequence[str],
    separability: P0SeparabilityConfig,
    feature_components: int = 6,
) -> dict[str, object]:
    """Return group-OOF metrics for one fixed, ordered block composition."""

    values = _canonical_records(records)
    requested = tuple(blocks)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(block not in COMMON_BLOCKS for block in requested)
    ):
        raise ValueError("blocks must be nonempty unique members of COMMON_BLOCKS")
    if not isinstance(separability, P0SeparabilityConfig):
        raise TypeError("separability must be P0SeparabilityConfig")
    if (
        isinstance(feature_components, bool)
        or not isinstance(feature_components, int)
        or feature_components < 1
    ):
        raise ValueError("feature_components must be a positive integer")
    if {item.role for item in values} != {"factual", "legal"}:
        raise RuntimeError("block OOF requires both target roles")
    assignment = _fold_assignment(values, separability.folds)
    scores = [float("nan")] * len(values)
    labels = [1 if item.role == "factual" else 0 for item in values]
    groups = [item.group_id for item in values]
    fold_receipts: list[dict[str, object]] = []
    for fold in range(separability.folds):
        train = [
            index for index, group in enumerate(groups) if assignment[group] != fold
        ]
        test = [
            index for index, group in enumerate(groups) if assignment[group] == fold
        ]
        if not train or not test:
            raise RuntimeError("group OOF produced an empty fold")
        projected = [
            _project_fold_block(
                values,
                train,
                test,
                block,
                feature_components,
            )
            for block in requested
        ]
        train_raw = torch.cat([item[0] for item in projected], dim=1)
        test_raw = torch.cat([item[1] for item in projected], dim=1)
        projection = {
            block: item[2]
            for block, item in zip(requested, projected, strict=True)
        }
        median, scale, constant, maxdev = _robust_scale_fit(train_raw)
        train_scaled = _robust_scale(train_raw, median, scale)
        test_scaled = _robust_scale(test_raw, median, scale)
        beta, fit = _fit_logistic_irls(
            train_scaled,
            torch.tensor(
                [labels[index] for index in train],
                dtype=torch.float64,
            ),
            [groups[index] for index in train],
            separability,
        )
        design = torch.cat(
            (
                torch.ones((len(test), 1), dtype=torch.float64),
                test_scaled,
            ),
            dim=1,
        )
        probability = torch.sigmoid(design @ beta)
        for index, score in zip(test, probability.tolist(), strict=True):
            scores[index] = float(score)
        train_groups = sorted({groups[index] for index in train})
        test_groups = sorted({groups[index] for index in test})
        if set(train_groups) & set(test_groups):
            raise AssertionError("group OOF leaked a group across train/test")
        fold_receipts.append(
            {
                "fold": fold,
                "train_groups": train_groups,
                "test_groups": test_groups,
                "train_targets": len(train),
                "test_targets": len(test),
                "raw_dimensions_by_block": {
                    block: int(values[0].block(block).numel())
                    for block in requested
                },
                "model_dimensions": int(train_scaled.shape[1]),
                "projection_fit_by_block": projection,
                "scale_fit": {
                    "fit_population": "training-fold-all-roles",
                    "median": [float(value) for value in median.tolist()],
                    "scale": [float(value) for value in scale.tolist()],
                    "constant_floor_dimensions": [
                        index
                        for index, flag in enumerate(constant.tolist())
                        if flag
                    ],
                    "maxdev_fallback_dimensions": [
                        index
                        for index, flag in enumerate(maxdev.tolist())
                        if flag
                    ],
                },
                "classifier_fit": fit,
                "classifier_parameter_fingerprint": stable_fingerprint(
                    [float(value) for value in beta.tolist()]
                ),
            }
        )
    if any(not isfinite(value) for value in scores):
        raise RuntimeError("block OOF left a score undefined")
    weights = _group_balanced_weights(labels, groups)
    auc = _weighted_auc(scores, labels, weights)
    bootstrap = _fixed_oof_group_bootstrap(
        scores,
        labels,
        groups,
        separability,
    )
    epsilon = 1e-12
    losses = [
        -(
            label * log(max(epsilon, min(1.0 - epsilon, score)))
            + (1 - label)
            * log(max(epsilon, min(1.0 - epsilon, 1.0 - score)))
        )
        for label, score in zip(labels, scores, strict=True)
    ]
    total_weight = sum(weights)
    balanced_log_loss = sum(
        weight * loss for weight, loss in zip(weights, losses, strict=True)
    ) / total_weight
    return _fingerprinted(
        {
            "schema_version": COMPOSITE_OOF_SCHEMA,
            "split": "D_R",
            "blocks": list(requested),
            "block_definitions": {
                block: block_definition(block) for block in requested
            },
            "records": len(values),
            "groups": len(set(groups)),
            "group_aware": True,
            "cross_fitted": True,
            "estimands": {
                "group_balanced_oof_auc": auc,
                "group_balanced_oof_auc_bootstrap_lower": bootstrap["lower"],
                "group_balanced_oof_auc_bootstrap_upper": bootstrap["upper"],
                "group_balanced_cross_fitted_log_loss": balanced_log_loss,
            },
            "auc_bootstrap": bootstrap,
            "folds": fold_receipts,
            "oof_predictions": [
                {
                    "identity": list(item.identity),
                    "group_id": item.group_id,
                    "role": item.role,
                    "score_factual": score,
                }
                for item, score in zip(values, scores, strict=True)
            ],
            "interpretation": _INTERPRETATION,
            "not_an_independent_causal_effect": True,
            "authority": _authority(),
        }
    )


def run_block_only_group_oof(
    records: Sequence[CommonStateRecord],
    *,
    block: str,
    separability: P0SeparabilityConfig,
    feature_components: int = 6,
) -> dict[str, object]:
    """Return deterministic group-OOF predictive metrics for one block."""

    composite = run_composite_group_oof(
        records,
        blocks=(block,),
        separability=separability,
        feature_components=feature_components,
    )
    result = dict(composite)
    result.pop("receipt_fingerprint")
    result.pop("blocks")
    result.pop("block_definitions")
    result["schema_version"] = BLOCK_OOF_SCHEMA
    result["block"] = block
    result["block_definition"] = block_definition(block)
    return _fingerprinted(result)


def _support_values(
    factual: tuple[CommonStateRecord, ...],
    legal: tuple[CommonStateRecord, ...],
    block: str,
    feature_components: int,
    *,
    legal_fit_filter: frozenset[str] | None = None,
) -> tuple[Tensor, Tensor, dict[str, object] | None]:
    factual_raw = torch.stack([item.block(block) for item in factual])
    legal_raw = torch.stack([item.block(block) for item in legal])
    if block not in FEATURE_BLOCKS:
        return factual_raw, legal_raw, None
    fit_indices = [
        index
        for index, item in enumerate(legal)
        if legal_fit_filter is None or item.group_id in legal_fit_filter
    ]
    if not fit_indices:
        raise RuntimeError("support feature projection has an empty legal fit set")
    projector = _fit_feature_projector(
        legal_raw[fit_indices],
        feature_components,
    )
    fit_population = [
        {
            "identity": list(legal[index].identity),
            "group_id": legal[index].group_id,
        }
        for index in fit_indices
    ]
    parameters = {
        "raw_median": [
            float(value) for value in projector.raw_median.tolist()
        ],
        "raw_scale": [
            float(value) for value in projector.raw_scale.tolist()
        ],
        "raw_constant_floor_dimensions": [
            index
            for index, flag in enumerate(projector.raw_constant.tolist())
            if flag
        ],
        "raw_maxdev_fallback_dimensions": [
            index
            for index, flag in enumerate(
                projector.raw_maxdev_fallback.tolist()
            )
            if flag
        ],
        "pca_mean": [
            float(value) for value in projector.pca_mean.tolist()
        ],
        "basis": [
            [float(value) for value in row]
            for row in projector.basis.tolist()
        ],
        "singular_values": [
            float(value) for value in projector.singular_values.tolist()
        ],
    }
    return (
        _project_feature(factual_raw, projector),
        _project_feature(legal_raw, projector),
        {
            "fit_role": "legal-targets-only",
            "fit_targets": len(fit_indices),
            "fit_groups": len(
                {legal[index].group_id for index in fit_indices}
            ),
            "components": feature_components,
            "fit_population_fingerprint": stable_fingerprint(fit_population),
            "parameters": parameters,
            "parameter_fingerprint": stable_fingerprint(parameters),
        },
    )


def run_block_coverage_mmd(
    records: Sequence[CommonStateRecord],
    *,
    block: str,
    overlap: P0OverlapConfig,
    separability: P0SeparabilityConfig,
    feature_components: int = 6,
) -> dict[str, object]:
    """Wrap the frozen group-aware P0 coverage/MMD estimators for one block."""

    values = _canonical_records(records)
    if block not in COMMON_BLOCKS:
        raise ValueError(f"block must be one of {COMMON_BLOCKS}")
    if not isinstance(overlap, P0OverlapConfig):
        raise TypeError("overlap must be P0OverlapConfig")
    if not isinstance(separability, P0SeparabilityConfig):
        raise TypeError("separability must be P0SeparabilityConfig")
    if (
        isinstance(feature_components, bool)
        or not isinstance(feature_components, int)
        or feature_components < 1
    ):
        raise ValueError("feature_components must be a positive integer")
    factual = tuple(item for item in values if item.role == "factual")
    legal = tuple(item for item in values if item.role == "legal")
    if not factual or not legal:
        raise RuntimeError("coverage/MMD requires both target roles")
    factual_cov, legal_cov, coverage_projection = _support_values(
        factual,
        legal,
        block,
        feature_components,
    )
    coverage = _coverage_receipt(
        factual,
        legal,
        factual_cov,
        legal_cov,
        overlap,
    )
    coverage["descriptive_threshold_crossed"] = coverage.pop("pass")

    factual_groups = {item.group_id for item in factual}
    legal_exclusive_groups = frozenset(
        item.group_id for item in legal if item.group_id not in factual_groups
    )
    factual_mmd, legal_mmd, mmd_projection = _support_values(
        factual,
        legal,
        block,
        feature_components,
        legal_fit_filter=legal_exclusive_groups,
    )
    mmd = _mmd_receipt(
        factual,
        legal,
        factual_mmd,
        legal_mmd,
        space=block,
        config=separability,
    )
    mmd["observed_within_legal_reference_q95"] = mmd.pop("pass")
    return _fingerprinted(
        {
            "schema_version": BLOCK_SUPPORT_SCHEMA,
            "split": "D_R",
            "block": block,
            "coverage": coverage,
            "mmd": mmd,
            "coverage_projection_fit": coverage_projection,
            "mmd_projection_fit": mmd_projection,
            "interpretation": (
                "group-aware support/separability screen; not proof of equal or "
                "different full decoder-input distributions"
            ),
            "formal_p0_gate": False,
            "authority": _authority(),
        }
    )


def exact_same_source_subset(
    records: Sequence[CommonStateRecord],
    *,
    expectation: SameSourceExpectation = SameSourceExpectation(),
) -> tuple[CommonStateRecord, ...]:
    """Return only sample IDs that contain both roles, with exact count checks."""

    values = _canonical_records(records)
    if not isinstance(expectation, SameSourceExpectation):
        raise TypeError("expectation must be SameSourceExpectation")
    roles_by_source: dict[str, set[str]] = {}
    for item in values:
        roles_by_source.setdefault(item.sample_id, set()).add(item.role)
    sources = {
        sample_id
        for sample_id, roles in roles_by_source.items()
        if roles == {"factual", "legal"}
    }
    subset = tuple(item for item in values if item.sample_id in sources)
    factual = sum(item.role == "factual" for item in subset)
    legal = len(subset) - factual
    actual = len(sources), factual, legal
    frozen = (
        expectation.sources,
        expectation.factual_targets,
        expectation.legal_targets,
    )
    if actual != frozen:
        raise RuntimeError(
            f"exact same-source population differs from freeze: {actual} != {frozen}"
        )
    return subset


def shared_manifest_group_subset(
    records: Sequence[CommonStateRecord],
    *,
    expectation: SharedGroupExpectation = SharedGroupExpectation(),
) -> tuple[CommonStateRecord, ...]:
    """Return targets in manifest groups containing both roles."""

    values = _canonical_records(records)
    if not isinstance(expectation, SharedGroupExpectation):
        raise TypeError("expectation must be SharedGroupExpectation")
    roles_by_group: dict[str, set[str]] = {}
    for item in values:
        roles_by_group.setdefault(item.group_id, set()).add(item.role)
    groups = {
        group_id
        for group_id, roles in roles_by_group.items()
        if roles == {"factual", "legal"}
    }
    subset = tuple(item for item in values if item.group_id in groups)
    factual = sum(item.role == "factual" for item in subset)
    legal = len(subset) - factual
    actual = len(groups), factual, legal
    frozen = (
        expectation.groups,
        expectation.factual_targets,
        expectation.legal_targets,
    )
    if actual != frozen:
        raise RuntimeError(
            f"shared manifest-group population differs from freeze: {actual} != {frozen}"
        )
    return subset


def run_shared_group_sensitivity(
    records: Sequence[CommonStateRecord],
    *,
    blocks: Sequence[str],
    separability: P0SeparabilityConfig,
    expectation: SharedGroupExpectation = SharedGroupExpectation(),
    feature_components: int = 6,
) -> dict[str, object]:
    """Run uncentered block OOF screens on exact shared manifest groups."""

    requested = tuple(blocks)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(block not in COMMON_BLOCKS for block in requested)
    ):
        raise ValueError("blocks must be unique members of COMMON_BLOCKS")
    subset = shared_manifest_group_subset(records, expectation=expectation)
    results = {
        block: run_block_only_group_oof(
            subset,
            block=block,
            separability=separability,
            feature_components=feature_components,
        )
        for block in requested
    }
    return _fingerprinted(
        {
            "schema_version": SHARED_GROUP_SCHEMA,
            "split": "D_R",
            "stratum": "shared-manifest-group",
            "population": {
                "groups": expectation.groups,
                "factual_targets": expectation.factual_targets,
                "legal_targets": expectation.legal_targets,
            },
            "source_centered": False,
            "results": results,
            "interpretation": (
                "shared-group predictive sensitivity; distinct from exact "
                "dual-role-source sensitivity and non-causal"
            ),
            "authority": _authority(),
        }
    )


def source_center_common_blocks(
    records: Sequence[CommonStateRecord],
) -> tuple[CommonStateRecord, ...]:
    """Center within selected dual-role sources without using role labels.

    This is a transductive sensitivity calculation on the predeclared overlap
    subset.  It does not estimate or remove a population-wide source effect.
    """

    values = _canonical_records(records)
    by_source: dict[str, list[CommonStateRecord]] = {}
    for item in values:
        by_source.setdefault(item.sample_id, []).append(item)
    centered: list[CommonStateRecord] = []
    for item in values:
        peers = by_source[item.sample_id]
        updates = {
            block: item.block(block)
            - torch.stack([peer.block(block) for peer in peers]).mean(dim=0)
            for block in COMMON_BLOCKS
        }
        centered.append(replace(item, **updates))
    return _canonical_records(centered)


def run_exact_same_source_sensitivity(
    records: Sequence[CommonStateRecord],
    *,
    blocks: Sequence[str],
    separability: P0SeparabilityConfig,
    expectation: SameSourceExpectation = SameSourceExpectation(),
    feature_components: int = 6,
) -> dict[str, object]:
    """Run source-centered group-OOF screens on the exact overlap subset."""

    requested = tuple(blocks)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(block not in COMMON_BLOCKS for block in requested)
    ):
        raise ValueError("blocks must be unique members of COMMON_BLOCKS")
    subset = exact_same_source_subset(records, expectation=expectation)
    centered = source_center_common_blocks(subset)
    results = {
        block: run_block_only_group_oof(
            centered,
            block=block,
            separability=separability,
            feature_components=feature_components,
        )
        for block in requested
    }
    return _fingerprinted(
        {
            "schema_version": SAME_SOURCE_SCHEMA,
            "split": "D_R",
            "stratum": "exact-dual-role-source",
            "population": {
                "sources": expectation.sources,
                "factual_targets": expectation.factual_targets,
                "legal_targets": expectation.legal_targets,
            },
            "source_centering": "label-blind-within-sample-mean-v1",
            "results": results,
            "interpretation": (
                "transductive selected-overlap sensitivity only; centering "
                "does not remove a population-wide source/background effect "
                "or identify a causal effect"
            ),
            "authority": _authority(),
        }
    )


__all__ = [
    "BLOCK_OOF_SCHEMA",
    "BLOCK_SUPPORT_SCHEMA",
    "COMMON_BLOCKS",
    "COMPOSITE_OOF_SCHEMA",
    "FAILURE_ATTRIBUTION_SCHEMA",
    "FEATURE_BLOCKS",
    "SAME_SOURCE_SCHEMA",
    "SHARED_GROUP_SCHEMA",
    "CommonStateRecord",
    "FailureAttributionPopulation",
    "LegalOccupancyLedger",
    "PopulationExpectation",
    "SameSourceExpectation",
    "SharedGroupExpectation",
    "block_definition",
    "build_common_state_record",
    "build_failure_attribution_population",
    "build_legal_occupancy_ledger",
    "exact_same_source_subset",
    "run_block_coverage_mmd",
    "run_block_only_group_oof",
    "run_composite_group_oof",
    "run_exact_same_source_sensitivity",
    "run_shared_group_sensitivity",
    "shared_manifest_group_subset",
    "source_center_common_blocks",
]
