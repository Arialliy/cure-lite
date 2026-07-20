"""The fixed CURE-Lite v0.1 residual decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DecoderConfig


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
        occupancy_small = F.interpolate(
            occupancy.to(dtype=feature.dtype),
            size=feature.shape[-2:],
            mode="nearest",
        )
        projected = self.project(feature.detach())
        logits_small = self.decode(torch.cat((projected, occupancy_small), dim=1))
        return F.interpolate(
            logits_small,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


__all__ = ["CURELiteDecoder"]
