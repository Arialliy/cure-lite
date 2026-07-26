"""Matched-control losses for the CURE-Lite paired-objective route.

The proposed objective remains :class:`~cure_lite.paired_losses.PairedDifferenceLoss`.
This module contains only the frozen controls from protocol sections 6.1,
6.2, and 6.5.  It deliberately does not own a decoder forward, a training
step, or pair metadata.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from .losses import CURELiteLoss
from .paired_losses import PairedDifferenceLoss


def _validate_bool_state_tensors(
    **values: Tensor,
) -> tuple[int, int, int, int]:
    if not values:
        raise ValueError("at least one state tensor is required")
    if any(not isinstance(value, Tensor) for value in values.values()):
        raise TypeError("all state values must be tensors")
    first = next(iter(values.values()))
    if first.ndim != 4 or first.shape[1] != 1:
        raise ValueError("state tensors must have shape [B,1,H,W]")
    if first.shape[0] < 1 or min(first.shape[-2:]) < 1:
        raise ValueError("state tensors must have non-empty batch and spatial axes")
    expected_shape = tuple(int(value) for value in first.shape)
    expected_device = first.device
    for name, value in values.items():
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must be bool")
        if tuple(value.shape) != expected_shape:
            raise ValueError("all state tensors must have identical shapes")
        if value.device != expected_device:
            raise ValueError("all state tensors must share a device")
    return expected_shape


def _validate_endpoint_logits(
    logits_plus: Tensor,
    logits_minus: Tensor,
    *,
    state_shape: tuple[int, int, int, int] | None = None,
) -> None:
    if not isinstance(logits_plus, Tensor) or not isinstance(logits_minus, Tensor):
        raise TypeError("logits_plus and logits_minus must be tensors")
    if logits_plus.ndim != 4 or logits_plus.shape[1] != 1:
        raise ValueError("endpoint logits must have shape [B,1,H,W]")
    if logits_plus.shape != logits_minus.shape:
        raise ValueError("endpoint logits must have identical shapes")
    if logits_plus.shape[0] < 1 or min(logits_plus.shape[-2:]) < 1:
        raise ValueError("endpoint logits must have non-empty batch and spatial axes")
    if not logits_plus.is_floating_point() or not logits_minus.is_floating_point():
        raise TypeError("endpoint logits must be floating point")
    if logits_plus.dtype != logits_minus.dtype:
        raise TypeError("endpoint logits must share a dtype")
    if logits_plus.device != logits_minus.device:
        raise ValueError("endpoint logits must share a device")
    if not torch.isfinite(logits_plus).all() or not torch.isfinite(logits_minus).all():
        raise ValueError("endpoint logits must be finite")
    if state_shape is not None and tuple(logits_plus.shape) != state_shape:
        raise ValueError("endpoint logits and supervision tensors must share a shape")


def _absolute_criterion(
    criterion: CURELiteLoss | None,
) -> CURELiteLoss:
    if criterion is None:
        return CURELiteLoss()
    if not isinstance(criterion, CURELiteLoss):
        raise TypeError("criterion must be CURELiteLoss")
    return criterion


def _paired_criterion(
    criterion: PairedDifferenceLoss | None,
) -> PairedDifferenceLoss:
    if criterion is None:
        return PairedDifferenceLoss()
    if not isinstance(criterion, PairedDifferenceLoss):
        raise TypeError("criterion must be PairedDifferenceLoss")
    return criterion


def _criterion_total(result: Mapping[str, Tensor], *, name: str) -> Tensor:
    if not isinstance(result, Mapping) or "total" not in result:
        raise TypeError(f"{name} must return a mapping containing 'total'")
    total = result["total"]
    if not isinstance(total, Tensor) or total.ndim != 0:
        raise ValueError(f"{name} total must be a scalar tensor")
    if not torch.isfinite(total):
        raise ValueError(f"{name} total must be finite")
    return total


def build_geometry_matched_endpoint_supervision(
    completion_field: Tensor,
    occupancy: Tensor,
    gt_union: Tensor,
    image_valid_mask: Tensor,
) -> dict[str, Tensor]:
    """Construct the exact ``(T, B, M)`` tensors for one endpoint batch.

    ``completion_field`` is the already audited instance-level field
    ``R_{G,V}(O)``.  It is *not* reconstructed as all uncovered GT pixels:
    targets that the fixed matcher regards as covered must remain absent.

    The frozen control semantics are

    ``T = completion_field``,
    ``B = V & ~O & ~union(G_j)``, and
    ``M = T | B``.
    """

    shape = _validate_bool_state_tensors(
        completion_field=completion_field,
        occupancy=occupancy,
        gt_union=gt_union,
        image_valid_mask=image_valid_mask,
    )
    del shape
    if torch.any(occupancy & ~image_valid_mask):
        raise ValueError("occupancy lies outside image_valid_mask")
    writable = image_valid_mask & ~occupancy
    if torch.any(completion_field & ~writable):
        raise ValueError("completion_field must be valid and writable")
    if torch.any(completion_field & ~gt_union):
        raise ValueError("completion_field must contain only GT pixels")

    background = writable & ~gt_union
    valid_mask = completion_field | background
    if torch.any(completion_field & background):
        raise AssertionError("target and writable background must be disjoint")
    return {
        "target": completion_field.to(dtype=torch.float32),
        "background": background,
        "valid_mask": valid_mask,
    }


def geometry_matched_independent_endpoint_loss(
    logits_plus: Tensor,
    logits_minus: Tensor,
    *,
    completion_plus: Tensor,
    completion_minus: Tensor,
    occupancy_plus: Tensor,
    occupancy_minus: Tensor,
    gt_union: Tensor,
    image_valid_mask: Tensor,
    criterion: CURELiteLoss | None = None,
) -> dict[str, Tensor]:
    """Return protocol-6.1 independent absolute endpoint ERM.

    Each endpoint is reduced by ``CURELiteLoss`` per state.  The two endpoint
    losses are then averaged inside each pair, followed by an arithmetic mean
    over pairs.  No ``Q_minus - Q_plus`` term is formed.
    """

    shape = _validate_bool_state_tensors(
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        gt_union=gt_union,
        image_valid_mask=image_valid_mask,
    )
    _validate_endpoint_logits(logits_plus, logits_minus, state_shape=shape)
    absolute = _absolute_criterion(criterion)
    plus = build_geometry_matched_endpoint_supervision(
        completion_plus,
        occupancy_plus,
        gt_union,
        image_valid_mask,
    )
    minus = build_geometry_matched_endpoint_supervision(
        completion_minus,
        occupancy_minus,
        gt_union,
        image_valid_mask,
    )
    plus_result = absolute(
        logits_plus,
        plus["target"],
        plus["valid_mask"],
    )
    minus_result = absolute(
        logits_minus,
        minus["target"],
        minus["valid_mask"],
    )
    plus_total = _criterion_total(plus_result, name="plus endpoint criterion")
    minus_total = _criterion_total(minus_result, name="minus endpoint criterion")
    per_pair_plus = plus_result["per_state_total"]
    per_pair_minus = minus_result["per_state_total"]
    if per_pair_plus.ndim != 1 or per_pair_minus.shape != per_pair_plus.shape:
        raise ValueError("CURELiteLoss must return one loss per endpoint state")
    per_pair_total = 0.5 * (per_pair_plus + per_pair_minus)
    total = per_pair_total.mean()
    return {
        "total": total,
        "loss": total,
        "plus_endpoint_loss": plus_total,
        "minus_endpoint_loss": minus_total,
        "per_pair_total": per_pair_total,
        "per_pair_plus": per_pair_plus,
        "per_pair_minus": per_pair_minus,
        "target_plus": plus["target"],
        "target_minus": minus["target"],
        "background_plus": plus["background"],
        "background_minus": minus["background"],
        "valid_mask_plus": plus["valid_mask"],
        "valid_mask_minus": minus["valid_mask"],
        "pair_count": torch.tensor(logits_plus.shape[0], device=logits_plus.device),
    }


def build_after_only_synthetic_supervision(
    selected_completion: Tensor,
    occupancy_minus: Tensor,
    gt_union: Tensor,
    image_valid_mask: Tensor,
) -> dict[str, Tensor]:
    """Construct the old atomic synthetic target and writable background.

    Unlike the geometry-matched independent control, ``selected_completion``
    contains only the deleted target's clean increment.  Pre-existing factual
    misses are excluded from both the target and the writable valid domain,
    reproducing the historical atomic synthetic-state supervision.
    """

    return build_geometry_matched_endpoint_supervision(
        selected_completion,
        occupancy_minus,
        gt_union,
        image_valid_mask,
    )


def after_only_absolute_synthetic_loss(
    logits_plus: Tensor,
    logits_minus: Tensor,
    *,
    selected_completion: Tensor,
    occupancy_minus: Tensor,
    gt_union: Tensor,
    image_valid_mask: Tensor,
    criterion: CURELiteLoss | None = None,
) -> dict[str, Tensor]:
    """Return protocol-6.2 after-only absolute synthetic loss.

    Both endpoint logits are required so the caller cannot silently change the
    matched forward contract.  Only ``logits_minus`` participates in the loss
    graph; ``logits_plus`` receives no gradient from this control.
    """

    shape = _validate_bool_state_tensors(
        selected_completion=selected_completion,
        occupancy_minus=occupancy_minus,
        gt_union=gt_union,
        image_valid_mask=image_valid_mask,
    )
    _validate_endpoint_logits(logits_plus, logits_minus, state_shape=shape)
    absolute = _absolute_criterion(criterion)
    supervision = build_after_only_synthetic_supervision(
        selected_completion,
        occupancy_minus,
        gt_union,
        image_valid_mask,
    )
    result = absolute(
        logits_minus,
        supervision["target"],
        supervision["valid_mask"],
    )
    total = _criterion_total(result, name="after-only criterion")
    return {
        **result,
        "total": total,
        "loss": total,
        "after_endpoint_loss": total,
        "target_minus": supervision["target"],
        "background_minus": supervision["background"],
        "valid_mask_minus": supervision["valid_mask"],
        "pair_count": torch.tensor(logits_plus.shape[0], device=logits_plus.device),
    }


def plus_detached_paired_difference_loss(
    logits_plus: Tensor,
    logits_minus: Tensor,
    label_increment: Tensor,
    image_valid_mask: Tensor,
    *,
    criterion: PairedDifferenceLoss | None = None,
) -> dict[str, Tensor]:
    """Evaluate the paired difference after detaching ``Q_plus``'s logit path."""

    _validate_endpoint_logits(logits_plus, logits_minus)
    paired = _paired_criterion(criterion)
    return paired(
        logits_plus.detach(),
        logits_minus,
        label_increment,
        image_valid_mask,
    )


def minus_detached_paired_difference_loss(
    logits_plus: Tensor,
    logits_minus: Tensor,
    label_increment: Tensor,
    image_valid_mask: Tensor,
    *,
    criterion: PairedDifferenceLoss | None = None,
) -> dict[str, Tensor]:
    """Evaluate the paired difference after detaching ``Q_minus``'s logit path."""

    _validate_endpoint_logits(logits_plus, logits_minus)
    paired = _paired_criterion(criterion)
    return paired(
        logits_plus,
        logits_minus.detach(),
        label_increment,
        image_valid_mask,
    )


__all__ = [
    "after_only_absolute_synthetic_loss",
    "build_after_only_synthetic_supervision",
    "build_geometry_matched_endpoint_supervision",
    "geometry_matched_independent_endpoint_loss",
    "minus_detached_paired_difference_loss",
    "plus_detached_paired_difference_loss",
]
