"""Shared target descriptors for miss-propensity cross-fitting."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from ..instances import instances_from_binary_mask
from ..intervention import enumerate_legal_deletions
from ..matching import match_components
from ..splits import SplitManifest
from ..types import InstanceMap, MatchResult
from .config import DescriptorConfig
from .protocol import CUREProtocol
from .source import FrozenSourceRecord
from .types import (
    _ELIGIBLE_CATALOG_SEAL,
    EligibleSampleCatalog,
    TargetDescriptor,
)


def dilate_mask(mask: Tensor, radius: int) -> Tensor:
    """Chebyshev dilation on a 2D or ``[B,1,H,W]`` boolean mask."""

    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise TypeError("mask must be a bool tensor")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    original_ndim = mask.ndim
    if original_ndim == 2:
        batched = mask[None, None]
    elif original_ndim == 3 and mask.shape[0] == 1:
        batched = mask[None]
    elif original_ndim == 4 and mask.shape[1] == 1:
        batched = mask
    else:
        raise ValueError("mask must have shape [H,W], [1,H,W], or [B,1,H,W]")
    if radius == 0:
        result = batched.clone()
    else:
        result = F.max_pool2d(
            batched.to(torch.float32),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        ).to(torch.bool)
    if original_ndim == 2:
        return result[0, 0]
    if original_ndim == 3:
        return result[0]
    return result


def _map_2d(value: Tensor, *, name: str, probability: bool = False) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if result.ndim == 4 and result.shape[:2] == (1, 1):
        result = result[0, 0]
    elif result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape [H,W] or [1,H,W]")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    if probability and torch.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must lie in [0,1]")
    return result.contiguous()


def extract_target_descriptor(
    *,
    sample_id: str,
    group_id: str,
    gt: InstanceMap,
    gt_id: int,
    match: MatchResult,
    eligible_factual_gt_ids: tuple[int, ...],
    legal_covered_gt_ids: tuple[int, ...],
    probability: Tensor,
    intensity: Tensor,
    config: DescriptorConfig = DescriptorConfig(),
) -> TargetDescriptor:
    """Extract the seven source-only fields used by the CURE propensity model.

    The background ring excludes every annotated target, so the descriptor is
    defined in exactly the same way for covered and missed instances.  The
    intensity map must already be expressed in the frozen preprocessing space;
    no dataset-specific inverse normalization is guessed here.
    """

    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be an InstanceMap")
    if not isinstance(match, MatchResult):
        raise TypeError("match must be a MatchResult")
    if not isinstance(config, DescriptorConfig):
        raise TypeError("config must be DescriptorConfig")
    if match.gt_ids != tuple(sorted(gt.ids)):
        raise ValueError("match is inconsistent with the GT instance map")
    factual_ids = set(eligible_factual_gt_ids)
    legal_ids = set(legal_covered_gt_ids)
    if factual_ids & legal_ids:
        raise ValueError("factual and legal eligible pools must be disjoint")
    if gt_id in factual_ids and gt_id in match.unmatched_gt_ids:
        role = "factual_miss"
    elif gt_id in legal_ids and gt_id in match.matched_gt_ids:
        role = "legal_covered"
    else:
        raise ValueError("gt_id is not in the common eligible propensity universe")

    probability_2d = _map_2d(probability, name="probability", probability=True)
    intensity_2d = _map_2d(intensity, name="intensity")
    if tuple(probability_2d.shape) != gt.shape or tuple(intensity_2d.shape) != gt.shape:
        raise ValueError("probability, intensity, and GT must share a grid")

    target = gt.by_id(gt_id).mask
    all_gt = gt.occupancy
    outer = dilate_mask(target, config.ring_outer_radius)
    inner = dilate_mask(target, config.ring_inner_radius)
    ring = outer & ~inner & ~all_gt
    if not bool(torch.any(ring)):
        raise ValueError("target has no uncontaminated local background ring")

    target_scores = probability_2d[target]
    background_scores = probability_2d[ring]
    target_intensity = intensity_2d[target]
    background_intensity = intensity_2d[ring]
    local_scr = (
        target_intensity.mean() - background_intensity.mean()
    ) / (background_intensity.std(unbiased=False) + config.epsilon)

    ymin, xmin, ymax, xmax = gt.by_id(gt_id).bbox
    height, width = gt.shape
    boundary_distance = min(ymin, xmin, height - ymax, width - xmax)
    normalized_boundary_distance = boundary_distance / float(max(height, width))

    values = torch.tensor(
        (
            float(target_scores.min()),
            float(target_scores.mean()),
            float(target_scores.max()),
            float(local_scr),
            math.log(float(gt.by_id(gt_id).area)),
            float(background_scores.max()),
            normalized_boundary_distance,
        ),
        dtype=torch.float64,
        device="cpu",
    )
    return TargetDescriptor(
        sample_id=sample_id,
        group_id=group_id,
        gt_id=gt_id,
        role=role,
        values=values,
    )


def build_eligible_sample_catalog(
    *,
    source: FrozenSourceRecord,
    protocol: CUREProtocol,
    manifest: SplitManifest,
) -> EligibleSampleCatalog:
    """Build the one authoritative factual/legal propensity universe.

    The function derives factual occupancy from the same frozen probability and
    :class:`CUREResidualConfig` used by inference, then recomputes matching and
    legal deletions.  Targets enter the universe only when their descriptor is
    valid and the shared suppression radius leaves writable support.  Callers
    cannot splice an arbitrary coverage map into a different probability state.
    The public construction route accepts only a source record issued by an
    actual execution of the protocol-bound frozen adapter; raw outputs and
    caller-supplied matches are deliberately not accepted here.
    """

    if not isinstance(source, FrozenSourceRecord):
        raise TypeError("source must be FrozenSourceRecord")
    if not isinstance(protocol, CUREProtocol):
        raise TypeError("protocol must be CUREProtocol")
    source.validate_against(protocol, manifest)
    sample_id = source.sample_id
    group_id = source.group_id
    gt = source.gt
    probability = source.probability
    intensity = source.intensity
    residual_config = protocol.residual_config
    match_config = protocol.match_config
    intervention_config = protocol.intervention_config
    descriptor_config = protocol.descriptor_config
    probability_2d = _map_2d(
        probability, name="probability", probability=True
    )
    if tuple(probability_2d.shape) != gt.shape:
        raise ValueError("probability and GT must share a grid")
    occupied = probability_2d >= residual_config.occupancy_threshold
    pred = instances_from_binary_mask(occupied, connectivity=8, min_area=1)
    match = match_components(pred, gt, match_config)
    legal_all = enumerate_legal_deletions(
        pred,
        gt,
        match,
        occupied,
        match_config=match_config,
        intervention_config=intervention_config,
    )

    all_gt = gt.occupancy

    def descriptor_valid(gt_id: int) -> bool:
        target = gt.by_id(gt_id).mask
        outer = dilate_mask(target, descriptor_config.ring_outer_radius)
        inner = dilate_mask(target, descriptor_config.ring_inner_radius)
        return bool(torch.any(outer & ~inner & ~all_gt))

    factual_editable = ~dilate_mask(
        occupied, residual_config.suppression_radius
    )
    factual_ids = tuple(
        gt_id
        for gt_id in sorted(match.unmatched_gt_ids)
        if bool(torch.any(gt.by_id(gt_id).mask & factual_editable))
        and descriptor_valid(gt_id)
    )
    eligible_deletions = tuple(
        deletion
        for deletion in legal_all
        if bool(
            torch.any(
                gt.by_id(deletion.gt_id).mask
                & ~dilate_mask(
                    deletion.occupancy_after,
                    residual_config.suppression_radius,
                )
            )
        )
        and descriptor_valid(deletion.gt_id)
    )
    legal_ids = tuple(deletion.gt_id for deletion in eligible_deletions)
    descriptors = tuple(
        sorted(
            (
                extract_target_descriptor(
                    sample_id=sample_id,
                    group_id=group_id,
                    gt=gt,
                    gt_id=gt_id,
                    match=match,
                    eligible_factual_gt_ids=factual_ids,
                    legal_covered_gt_ids=legal_ids,
                    probability=probability,
                    intensity=intensity,
                    config=descriptor_config,
                )
                for gt_id in (*factual_ids, *legal_ids)
            ),
            key=lambda item: item.key,
        )
    )
    return EligibleSampleCatalog(
        sample_id=sample_id,
        group_id=group_id,
        base_fingerprint=protocol.base_fingerprint,
        protocol_fingerprint=protocol.fingerprint,
        frozen_output_fingerprint=source.frozen_output_fingerprint,
        source_fingerprint=source.fingerprint,
        gt_fingerprint=source.gt_fingerprint,
        descriptors=descriptors,
        legal_deletions=eligible_deletions,
        excluded_factual_gt_ids=tuple(
            sorted(set(match.unmatched_gt_ids) - set(factual_ids))
        ),
        excluded_covered_gt_ids=tuple(
            sorted(set(match.matched_gt_ids) - set(legal_ids))
        ),
        occupancy_threshold=residual_config.occupancy_threshold,
        suppression_radius=residual_config.suppression_radius,
        _seal=_ELIGIBLE_CATALOG_SEAL,
    )


__all__ = [
    "build_eligible_sample_catalog",
    "dilate_mask",
    "extract_target_descriptor",
]
