"""Detector-independent deterministic CURE counterfactual transforms."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import TransformConfig, TransformDiagnostics, TransformSpec


class InvalidTransformSupportError(ValueError):
    """Raised when a valid transform request has insufficient local support."""


def _validated_target_mask(target_mask: Tensor) -> Tensor:
    if not isinstance(target_mask, Tensor):
        raise TypeError("target_mask must be a torch.Tensor")
    if target_mask.device.type != "cpu" or target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be a CPU bool tensor")
    if target_mask.ndim != 2:
        raise ValueError("target_mask must have shape [H,W]")
    if target_mask.shape[0] < 1 or target_mask.shape[1] < 1:
        raise ValueError("target_mask spatial dimensions must be non-empty")
    if not torch.any(target_mask):
        raise ValueError("target_mask must contain at least one target pixel")
    return target_mask.detach().contiguous()


def _validated_protected_mask(
    protected_mask: Tensor | None,
    target: Tensor,
) -> Tensor:
    if protected_mask is None:
        return torch.zeros_like(target)
    if not isinstance(protected_mask, Tensor):
        raise TypeError("protected_mask must be a torch.Tensor or None")
    if protected_mask.device.type != "cpu" or protected_mask.dtype != torch.bool:
        raise TypeError("protected_mask must be a CPU bool tensor")
    if protected_mask.ndim != 2 or protected_mask.shape != target.shape:
        raise ValueError("protected_mask must have the target grid shape")
    if torch.any(protected_mask & target):
        raise ValueError("protected_mask must not overlap the selected target")
    return protected_mask.detach().contiguous()


def _validated_image(image: Tensor, target: Tensor) -> Tensor:
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape [1,C,H,W]")
    if image.shape[1] < 1 or image.shape[2] < 1 or image.shape[3] < 1:
        raise ValueError("image channel and spatial dimensions must be non-empty")
    if not image.is_floating_point():
        raise TypeError("image must have a floating-point dtype")
    if not torch.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    if tuple(image.shape[-2:]) != tuple(target.shape):
        raise ValueError("image and target_mask spatial shapes must agree")
    return image


def _dilate(mask: Tensor, radius: int) -> Tensor:
    if radius == 0:
        return mask.clone()
    pooled = F.max_pool2d(
        mask.to(torch.float32).unsqueeze(0).unsqueeze(0),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return pooled[0, 0].to(torch.bool)


def build_soft_roi(
    target_mask: Tensor,
    config: TransformConfig = TransformConfig(),
) -> Tensor:
    """Return a CPU float32 soft ROI with finite, exactly zero support outside.

    Target pixels have weight one.  Successive Chebyshev shells decay linearly
    to ``1 / (roi_radius + 1)`` at the outer editable boundary.
    """

    target = _validated_target_mask(target_mask)
    if not isinstance(config, TransformConfig):
        raise TypeError("config must be TransformConfig")
    soft = target.to(torch.float32)
    previous = target
    for radius in range(1, config.roi_radius + 1):
        current = _dilate(target, radius)
        shell = current & ~previous
        weight = 1.0 - radius / (config.roi_radius + 1.0)
        soft[shell] = weight
        previous = current
    return soft.contiguous()


def build_background_ring(
    target_mask: Tensor,
    config: TransformConfig = TransformConfig(),
    *,
    protected_mask: Tensor | None = None,
) -> Tensor:
    """Return the local-background ring, excluding any protected GT pixels."""

    target = _validated_target_mask(target_mask)
    if not isinstance(config, TransformConfig):
        raise TypeError("config must be TransformConfig")
    protected = _validated_protected_mask(protected_mask, target)
    outer = _dilate(target, config.ring_outer_radius)
    inner = _dilate(target, config.ring_inner_radius)
    return (outer & ~inner & ~protected).contiguous()


def apply_counterfactual_transform(
    image: Tensor,
    target_mask: Tensor,
    spec: TransformSpec,
    *,
    protected_mask: Tensor | None = None,
) -> tuple[Tensor, TransformDiagnostics]:
    """Attenuate target-local contrast toward a channel-wise ring background.

    For a soft ROI weight ``w`` and attenuation strength ``lambda``, the local
    value is

    ``x' = x + w * lambda * (ring_mean - x)``.

    Boolean indexed assignment is used so image elements outside the hard ROI
    are copied bit-for-bit rather than merely multiplied by a numerical zero.
    """

    if not isinstance(spec, TransformSpec):
        raise TypeError("spec must be TransformSpec")
    target = _validated_target_mask(target_mask)
    protected = _validated_protected_mask(protected_mask, target)
    source = _validated_image(image, target)
    soft_roi_cpu = build_soft_roi(target, spec.config)
    # Other annotated targets are neither edited nor used to estimate the
    # local background.  This keeps the intervention target-specific when
    # instances are spatially close.
    soft_roi_cpu[protected] = 0.0
    hard_roi_cpu = soft_roi_cpu > 0.0
    ring_cpu = build_background_ring(
        target,
        spec.config,
        protected_mask=protected,
    )
    ring_pixels = int(torch.count_nonzero(ring_cpu))
    if ring_pixels < spec.config.minimum_ring_pixels:
        raise InvalidTransformSupportError(
            "local background ring contains fewer pixels than minimum_ring_pixels"
        )
    if torch.any(ring_cpu & hard_roi_cpu):
        raise AssertionError("background ring overlaps the editable ROI")

    hard_roi = hard_roi_cpu.to(device=source.device)
    ring = ring_cpu.to(device=source.device)
    soft_roi = soft_roi_cpu.to(device=source.device, dtype=source.dtype)
    # Accumulate the small ring in float64, then return to the input dtype.  The
    # transform output itself always preserves the input dtype and device.
    ring_values = source[0, :, ring]
    ring_background = ring_values.to(torch.float64).mean(dim=1).to(source.dtype)

    transformed = source.clone()
    local_source = source[0, :, hard_roi]
    local_weight = soft_roi[hard_roi].unsqueeze(0)
    local_background = ring_background.unsqueeze(1)
    transformed[0, :, hard_roi] = local_source + (
        spec.strength * local_weight * (local_background - local_source)
    )

    if not torch.isfinite(transformed).all():
        raise RuntimeError("counterfactual transform produced non-finite values")
    outside = ~hard_roi
    if not torch.equal(transformed[0, :, outside], source[0, :, outside]):
        raise RuntimeError("counterfactual transform changed pixels outside the ROI")

    absolute_delta = (transformed - source).abs()
    max_abs_delta = float(absolute_delta.max().detach().cpu())
    mean_abs_delta = float(absolute_delta.mean().detach().cpu())
    outside_delta = absolute_delta[0, :, outside]
    outside_roi_max_delta = (
        float(outside_delta.max().detach().cpu()) if outside_delta.numel() else 0.0
    )
    diagnostics = TransformDiagnostics(
        max_abs_delta=max_abs_delta,
        mean_abs_delta=mean_abs_delta,
        outside_roi_max_delta=outside_roi_max_delta,
        ring_pixels=ring_pixels,
    )
    return transformed, diagnostics


__all__ = [
    "InvalidTransformSupportError",
    "apply_counterfactual_transform",
    "build_background_ring",
    "build_soft_roi",
]
