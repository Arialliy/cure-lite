"""A compact standard U-Net used only to provide frozen Stage-A evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ReferenceBaseModelConfig


@dataclass(frozen=True)
class ReferenceBaseNetworkOutput:
    """Train-time logits and the quarter-resolution feature exposed to CURE."""

    logits: Tensor
    feature: Tensor


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip: nn.Module
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.activation = nn.SiLU()

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(self.body(value) + self.skip(value))


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.block = _ResidualBlock(out_channels, out_channels, groups)

    def forward(self, value: Tensor) -> Tensor:
        return self.block(self.down(value))


class _Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.block = _ResidualBlock(
            in_channels + skip_channels,
            out_channels,
            groups,
        )

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(
            value,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.block(torch.cat((value, skip), dim=1))


class ReferenceBaseNetwork(nn.Module):
    """Standard compact U-Net; not part of the CURE method definition."""

    def __init__(
        self,
        config: ReferenceBaseModelConfig = ReferenceBaseModelConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(config, ReferenceBaseModelConfig):
            raise TypeError("config must be ReferenceBaseModelConfig")
        self.config = config
        groups = config.norm_groups
        self.stem = _ResidualBlock(
            config.in_channels,
            config.stem_channels,
            groups,
        )
        self.down_half = _Down(
            config.stem_channels,
            config.half_channels,
            groups,
        )
        self.down_quarter = _Down(
            config.half_channels,
            config.feature_channels,
            groups,
        )
        self.down_eighth = _Down(
            config.feature_channels,
            config.eighth_channels,
            groups,
        )
        self.down_sixteenth = _Down(
            config.eighth_channels,
            config.bottleneck_channels,
            groups,
        )
        self.up_eighth = _Up(
            config.bottleneck_channels,
            config.eighth_channels,
            config.eighth_channels,
            groups,
        )
        self.up_quarter = _Up(
            config.eighth_channels,
            config.feature_channels,
            config.feature_channels,
            groups,
        )
        self.up_half = _Up(
            config.feature_channels,
            config.half_channels,
            config.half_channels,
            groups,
        )
        self.up_full = _Up(
            config.half_channels,
            config.stem_channels,
            config.stem_channels,
            groups,
        )
        self.head = nn.Conv2d(config.stem_channels, 1, kernel_size=1)

    @property
    def feature_channels(self) -> int:
        return self.config.feature_channels

    def forward_with_feature(self, images: Tensor) -> ReferenceBaseNetworkOutput:
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("images must have shape [B,C,H,W]")
        if images.shape[0] < 1 or images.shape[1] != self.config.in_channels:
            raise ValueError("images have an invalid batch or channel count")
        if images.shape[-2] % 16 or images.shape[-1] % 16:
            raise ValueError("reference-base image dimensions must be divisible by 16")
        full = self.stem(images)
        half = self.down_half(full)
        quarter = self.down_quarter(half)
        eighth = self.down_eighth(quarter)
        bottleneck = self.down_sixteenth(eighth)
        decoded = self.up_eighth(bottleneck, eighth)
        decoded = self.up_quarter(decoded, quarter)
        decoded = self.up_half(decoded, half)
        decoded = self.up_full(decoded, full)
        logits = self.head(decoded)
        if logits.shape[-2:] != images.shape[-2:]:
            raise RuntimeError("reference-base logits do not use the input grid")
        return ReferenceBaseNetworkOutput(logits=logits, feature=quarter)

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_feature(images).logits


__all__ = ["ReferenceBaseNetwork", "ReferenceBaseNetworkOutput"]
