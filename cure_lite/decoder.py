"""The fixed CURE-Lite v0.1 residual decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DecoderConfig


def project_occupancy_to_feature_grid(
    occupancy: Tensor,
    output_size: tuple[int, int],
) -> Tensor:
    """Project binary occupancy without dropping sub-stride positive pixels.

    IRSTD components can be smaller than the feature stride.  Nearest-neighbor
    downsampling can therefore turn a non-empty component into an all-zero
    conditioning map.  Adaptive max pooling preserves whether every feature
    cell contains at least one occupied source pixel and keeps the projected
    state binary.
    """

    if not isinstance(occupancy, Tensor):
        raise TypeError("occupancy must be a tensor")
    if occupancy.ndim != 4 or occupancy.shape[1] != 1 or occupancy.shape[0] < 1:
        raise ValueError("occupancy must have shape [B,1,H,W]")
    if occupancy.dtype != torch.bool:
        raise TypeError("occupancy must be bool")
    if (
        not isinstance(output_size, tuple)
        or len(output_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in output_size)
    ):
        raise ValueError("output_size must contain two positive integers")
    if any(
        projected > source
        for projected, source in zip(output_size, occupancy.shape[-2:], strict=True)
    ):
        raise ValueError("occupancy projection may not upsample the source grid")
    if tuple(occupancy.shape[-2:]) == output_size:
        return occupancy.contiguous()
    projected = F.adaptive_max_pool2d(
        occupancy.to(dtype=torch.float32),
        output_size,
    )
    return projected.to(dtype=torch.bool).contiguous()


class CURELiteDecoder(nn.Module):
    """Decode detached base features conditioned only on binary occupancy."""

    def __init__(
        self,
        config: DecoderConfig | int | None = None,
        width: int | None = None,
        groups: int | None = None,
        *,
        feature_channels: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(config, DecoderConfig):
            if feature_channels is not None or width is not None or groups is not None:
                raise ValueError("do not override fields of an explicit DecoderConfig")
            resolved = config
        else:
            if config is not None and feature_channels is not None:
                raise ValueError("feature_channels was supplied twice")
            channels = config if config is not None else feature_channels
            if channels is None:
                raise TypeError("feature_channels or DecoderConfig is required")
            resolved = DecoderConfig(
                feature_channels=channels,
                width=32 if width is None else width,
                groups=8 if groups is None else groups,
            )
        self.config = resolved
        self.feature_channels = resolved.feature_channels

        self.project = nn.Sequential(
            nn.Conv2d(resolved.feature_channels, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(33, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, feature: Tensor, occupancy: Tensor) -> Tensor:
        if not isinstance(feature, Tensor) or not isinstance(occupancy, Tensor):
            raise TypeError("feature and occupancy must be tensors")
        if feature.ndim != 4:
            raise ValueError("feature must have shape [B,C,h,w]")
        if occupancy.ndim != 4 or occupancy.shape[1] != 1:
            raise ValueError("occupancy must have shape [B,1,H,W]")
        if feature.shape[0] != occupancy.shape[0] or feature.shape[0] < 1:
            raise ValueError("feature and occupancy batch sizes must agree and be non-empty")
        if feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"expected {self.feature_channels} feature channels, got {feature.shape[1]}"
            )
        if not feature.is_floating_point():
            raise TypeError("feature must be floating point")
        if feature.dtype != self.project[0].weight.dtype:
            raise TypeError(
                "feature dtype must match decoder weights "
                f"({feature.dtype} != {self.project[0].weight.dtype})"
            )
        if occupancy.dtype != torch.bool:
            raise TypeError("occupancy must be bool")
        if feature.device != occupancy.device:
            raise ValueError("feature and occupancy must share a device")

        output_size = occupancy.shape[-2:]
        occupancy_small = project_occupancy_to_feature_grid(
            occupancy,
            tuple(feature.shape[-2:]),
        ).to(dtype=feature.dtype)
        projected = self.project(feature.detach())
        logits_small = self.decode(torch.cat((projected, occupancy_small), dim=1))
        return F.interpolate(
            logits_small,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


__all__ = ["CURELiteDecoder", "project_occupancy_to_feature_grid"]
