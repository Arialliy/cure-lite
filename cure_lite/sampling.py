"""Deterministic target selection for CURE-Lite training states."""

from __future__ import annotations

import hashlib
from math import floor, isfinite, log1p

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import MissAlignmentConfig
from .types import LegalDeletion


def _validate_key(*, sample_id: str, epoch: int, global_seed: int) -> None:
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")


def stable_hash(*parts: object) -> int:
    """Return a process-independent 64-bit hash of typed, length-prefixed parts."""

    digest = hashlib.sha256()
    for part in parts:
        type_name = type(part).__qualname__.encode("utf-8")
        value = str(part).encode("utf-8")
        digest.update(len(type_name).to_bytes(4, "big"))
        digest.update(type_name)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def _identity_schedule(
    identities: tuple[tuple[int, ...], ...],
    *,
    namespace: str,
    sample_id: str,
    global_seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the canonical seed-specific order of immutable identities."""

    return tuple(
        sorted(
            identities,
            key=lambda identity: (
                stable_hash(namespace, sample_id, global_seed, *identity),
                *identity,
            ),
        )
    )


def choose_uniform_legal_identity(
    legal_identities: tuple[tuple[int, int], ...],
    *,
    sample_id: str,
    epoch: int,
    global_seed: int,
) -> tuple[int, int] | None:
    """Choose one compact legal-deletion identity on the original v2 cycle."""

    if not isinstance(legal_identities, tuple):
        raise TypeError("legal_identities must be a tuple")
    if any(
        not isinstance(identity, tuple)
        or len(identity) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in identity
        )
        for identity in legal_identities
    ):
        raise ValueError(
            "legal_identities must contain positive integer (gt_id, pred_id) pairs"
        )
    if len(legal_identities) != len(set(legal_identities)):
        raise ValueError("legal_identities must be unique")
    _validate_key(sample_id=sample_id, epoch=epoch, global_seed=global_seed)
    if not legal_identities:
        return None
    schedule = _identity_schedule(
        legal_identities,
        namespace="legal-deletion-cycle-v2",
        sample_id=sample_id,
        global_seed=global_seed,
    )
    selected = schedule[epoch % len(schedule)]
    return selected[0], selected[1]


def choose_uniform_legal_deletion(
    legal_candidates: tuple[LegalDeletion, ...],
    *,
    sample_id: str,
    epoch: int,
    global_seed: int,
) -> LegalDeletion | None:
    """Choose from a seed-specific without-replacement catalog cycle.

    Every candidate appears exactly once in each contiguous catalog-length
    cycle.  The seed changes the deterministic permutation, while ``epoch``
    advances through it.  This avoids silently repeating an easy candidate
    while another legal intervention is never exposed during a fixed run.
    """

    if not isinstance(legal_candidates, tuple):
        raise TypeError("legal_candidates must be a tuple")
    if any(not isinstance(item, LegalDeletion) for item in legal_candidates):
        raise TypeError("legal_candidates contains a non-LegalDeletion item")
    _validate_key(sample_id=sample_id, epoch=epoch, global_seed=global_seed)
    if not legal_candidates:
        return None
    identities = tuple((item.gt_id, item.pred_id) for item in legal_candidates)
    if len(identities) != len(set(identities)):
        raise ValueError("legal_candidates must have unique (gt_id, pred_id) identities")
    selected_identity = choose_uniform_legal_identity(
        identities,
        sample_id=sample_id,
        epoch=epoch,
        global_seed=global_seed,
    )
    if selected_identity is None:
        raise AssertionError("a non-empty legal catalog must select one identity")
    return next(
        item
        for item in legal_candidates
        if (item.gt_id, item.pred_id) == selected_identity
    )


def choose_uniform_factual_gt_id(
    reachable_gt_ids: tuple[int, ...],
    *,
    sample_id: str,
    epoch: int,
    global_seed: int,
) -> int | None:
    """Select one atomic factual target with deterministic cyclic coverage.

    The key deliberately excludes the training variant and model scores, so
    Factual-only, exposure-matched factual, and Uniform-Legal runs share the
    same factual-target schedule when their split, epoch, and seed agree.  A
    sample/seed-specific hash permutation fixes the order, while ``epoch``
    advances through that order.  Consequently, every target is selected
    exactly once in every contiguous ``len(reachable_gt_ids)`` epochs; unlike
    independent per-epoch hashing, the selector cannot repeatedly starve one
    member of a finite catalog.
    """

    if not isinstance(reachable_gt_ids, tuple):
        raise TypeError("reachable_gt_ids must be a tuple")
    if any(
        isinstance(gt_id, bool) or not isinstance(gt_id, int) or gt_id < 1
        for gt_id in reachable_gt_ids
    ):
        raise ValueError("reachable_gt_ids must contain positive integer IDs")
    if reachable_gt_ids != tuple(sorted(set(reachable_gt_ids))):
        raise ValueError("reachable_gt_ids must be sorted and unique")
    _validate_key(sample_id=sample_id, epoch=epoch, global_seed=global_seed)
    if not reachable_gt_ids:
        return None
    schedule = _identity_schedule(
        tuple((gt_id,) for gt_id in reachable_gt_ids),
        namespace="factual-target-cycle-v2",
        sample_id=sample_id,
        global_seed=global_seed,
    )
    return schedule[epoch % len(schedule)][0]


def positive_region_feature_rms_many(
    feature: Tensor,
    positive_masks: tuple[Tensor, ...],
) -> tuple[float, ...]:
    """Summarize one frozen feature over several positive supervision masks.

    The source-grid mask is projected with adaptive average pooling so each
    feature cell is weighted by its covered source-grid fraction.  Channel
    count is normalized explicitly, making the scalar comparable within a
    detector even when adapters expose different feature widths.  Converting
    the frozen feature to CPU float64 happens once per source, not once per
    target.
    """

    if not isinstance(feature, Tensor) or feature.ndim != 4:
        raise ValueError("feature must have shape [1,C,h,w]")
    if feature.shape[0] != 1 or feature.shape[1] < 1:
        raise ValueError("feature must contain one non-empty feature map")
    if not feature.is_floating_point() or not torch.isfinite(feature).all():
        raise ValueError("feature must be finite and floating point")
    if feature.requires_grad:
        raise ValueError("miss alignment requires a detached frozen feature")
    if not isinstance(positive_masks, tuple) or not positive_masks:
        raise ValueError("positive_masks must be a non-empty tuple")

    normalized_masks: list[Tensor] = []
    source_shape: tuple[int, int] | None = None
    for positive_mask in positive_masks:
        mask = torch.as_tensor(positive_mask, device="cpu")
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError("positive_mask must have shape [H,W] or [1,H,W]")
        if source_shape is None:
            source_shape = tuple(int(value) for value in mask.shape)
        elif tuple(mask.shape) != source_shape:
            raise ValueError("positive masks for one source must share a grid")
        if mask.dtype == torch.bool:
            mask_float = mask.to(torch.float64)
        elif mask.is_floating_point():
            if not torch.isfinite(mask).all() or torch.any(
                (mask != 0.0) & (mask != 1.0)
            ):
                raise ValueError("positive_mask must be binary")
            mask_float = mask.to(torch.float64)
        else:
            raise TypeError("positive_mask must be bool or floating point")
        if not torch.any(mask_float):
            raise ValueError("positive_mask must be non-empty")
        normalized_masks.append(mask_float)

    source = feature.detach().to(device="cpu", dtype=torch.float64)
    weights = F.adaptive_avg_pool2d(
        torch.stack(normalized_masks).unsqueeze(1),
        tuple(int(value) for value in source.shape[-2:]),
    )[:, 0]
    weight_sums = weights.sum(dim=(-2, -1))
    if torch.any(weight_sums <= 0.0):
        raise RuntimeError("positive mask has zero feature-grid weight")
    mean_squares = (
        source[0].square().unsqueeze(0) * weights.unsqueeze(1)
    ).sum(dim=(1, 2, 3)) / (weight_sums * source.shape[1])
    values = tuple(float(value) for value in torch.sqrt(mean_squares))
    if any(not isfinite(value) or value < 0.0 for value in values):
        raise RuntimeError("feature RMS values must be finite and non-negative")
    return values


def positive_region_feature_rms(feature: Tensor, positive_mask: Tensor) -> float:
    """Summarize one frozen feature over one positive supervision mask."""

    return positive_region_feature_rms_many(feature, (positive_mask,))[0]


def miss_alignment_descriptor(feature: Tensor, positive_mask: Tensor) -> float:
    """Return the fixed v0.2 descriptor ``log1p(feature_rms)``."""

    return log1p(positive_region_feature_rms(feature, positive_mask))


def miss_alignment_descriptors(
    feature: Tensor,
    positive_masks: tuple[Tensor, ...],
) -> tuple[float, ...]:
    """Return fixed v0.2 descriptors with one feature conversion per source."""

    return tuple(
        log1p(value)
        for value in positive_region_feature_rms_many(feature, positive_masks)
    )


def quantized_miss_alignment_distance(
    factual_descriptor: float,
    legal_descriptor: float,
    config: MissAlignmentConfig = MissAlignmentConfig(),
) -> int:
    """Return a half-up fixed-point absolute descriptor distance."""

    if not isinstance(config, MissAlignmentConfig):
        raise TypeError("config must be MissAlignmentConfig")
    for name, value in (
        ("factual_descriptor", factual_descriptor),
        ("legal_descriptor", legal_descriptor),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    distance = abs(float(factual_descriptor) - float(legal_descriptor))
    return int(floor(distance * config.distance_quantization + 0.5))


def quantized_miss_alignment_descriptor(
    descriptor: float,
    config: MissAlignmentConfig = MissAlignmentConfig(),
) -> int:
    """Return the canonical fixed-point representation used in receipts."""

    if not isinstance(config, MissAlignmentConfig):
        raise TypeError("config must be MissAlignmentConfig")
    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, (int, float))
        or not isfinite(float(descriptor))
        or float(descriptor) < 0.0
    ):
        raise ValueError("descriptor must be finite and non-negative")
    return int(floor(float(descriptor) * config.distance_quantization + 0.5))


def choose_miss_aligned_legal_identity(
    factual_descriptor: float,
    legal_descriptors: tuple[tuple[str, int, int, float], ...],
    *,
    config: MissAlignmentConfig = MissAlignmentConfig(),
) -> tuple[str, int, int, int]:
    """Choose the globally nearest legal target with a canonical tie-break.

    Each legal tuple is ``(sample_id, gt_id, pred_id, descriptor)``.  The
    returned tuple appends the quantized distance.  Candidate reuse across
    factual targets is intentional: repeated mass represents the empirical
    factual target distribution rather than an arbitrary diversity constraint.
    """

    if not isinstance(config, MissAlignmentConfig):
        raise TypeError("config must be MissAlignmentConfig")
    if not isinstance(legal_descriptors, tuple) or not legal_descriptors:
        raise ValueError("legal_descriptors must be a non-empty tuple")
    normalized: list[tuple[int, str, int, int]] = []
    identities: list[tuple[str, int, int]] = []
    for item in legal_descriptors:
        if not isinstance(item, tuple) or len(item) != 4:
            raise ValueError(
                "legal descriptors must be (sample_id,gt_id,pred_id,value)"
            )
        sample_id, gt_id, pred_id, descriptor = item
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("legal sample_id must be non-empty")
        for name, value in (("gt_id", gt_id), ("pred_id", pred_id)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"legal {name} must be a positive integer")
        identity = (sample_id, gt_id, pred_id)
        identities.append(identity)
        normalized.append(
            (
                quantized_miss_alignment_distance(
                    factual_descriptor,
                    descriptor,
                    config,
                ),
                sample_id,
                gt_id,
                pred_id,
            )
        )
    if len(identities) != len(set(identities)):
        raise ValueError("legal descriptor identities must be unique")
    best = min(normalized)
    return best[1], best[2], best[3], best[0]


__all__ = [
    "choose_miss_aligned_legal_identity",
    "choose_uniform_factual_gt_id",
    "choose_uniform_legal_deletion",
    "choose_uniform_legal_identity",
    "miss_alignment_descriptor",
    "miss_alignment_descriptors",
    "positive_region_feature_rms",
    "positive_region_feature_rms_many",
    "quantized_miss_alignment_descriptor",
    "quantized_miss_alignment_distance",
    "stable_hash",
]
