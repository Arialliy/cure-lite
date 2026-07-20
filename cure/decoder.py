"""Lightweight feature-and-coverage-conditioned residual decoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import CUREResidualConfig


class CUREResidualDecoder(nn.Module):
    """The only trainable inference module added by full CURE.

    The core path uses only frozen features and the intervened coverage state.
    Base probability conditioning is retained solely as a shortcut-risk
    ablation and is disabled by default.
    """

    def __init__(self, config: CUREResidualConfig) -> None:
        super().__init__()
        if not isinstance(config, CUREResidualConfig):
            raise TypeError("config must be CUREResidualConfig")
        self.config = config
        self.feature_channels = config.feature_channels
        self.project = nn.Sequential(
            nn.Conv2d(config.feature_channels, config.width, kernel_size=1),
            nn.GroupNorm(config.groups, config.width),
            nn.SiLU(),
        )
        conditioning_channels = 2 if config.condition_on_probability else 1
        input_channels = config.width + conditioning_channels
        self.decode = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                padding=1,
                groups=input_channels,
            ),
            nn.Conv2d(input_channels, config.width, kernel_size=1),
            nn.GroupNorm(config.groups, config.width),
            nn.SiLU(),
            nn.Conv2d(
                config.width,
                config.width,
                kernel_size=3,
                padding=1,
                groups=config.width,
            ),
            nn.Conv2d(config.width, config.width, kernel_size=1),
            nn.GroupNorm(config.groups, config.width),
            nn.SiLU(),
            nn.Conv2d(config.width, 1, kernel_size=1),
        )
        output_layer = self.decode[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(
            output_layer.bias,
            math.log(
                config.initial_residual_probability
                / (1.0 - config.initial_residual_probability)
            ),
        )

    def forward(
        self,
        feature: Tensor,
        base_probability: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        if any(not isinstance(value, Tensor) for value in (feature, base_probability, occupancy)):
            raise TypeError("feature, base_probability, and occupancy must be tensors")
        if feature.ndim != 4 or feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"feature must have shape [B,{self.feature_channels},h,w]"
            )
        if base_probability.ndim != 4 or base_probability.shape[1] != 1:
            raise ValueError("base_probability must have shape [B,1,H,W]")
        if occupancy.shape != base_probability.shape or occupancy.dtype != torch.bool:
            raise ValueError("occupancy must be bool with the base-probability shape")
        if feature.shape[0] != base_probability.shape[0] or feature.shape[0] < 1:
            raise ValueError("decoder inputs must have the same non-empty batch size")
        if feature.dtype != self.project[0].weight.dtype:
            raise TypeError("feature dtype must match decoder weights")
        if base_probability.dtype != feature.dtype:
            raise TypeError("base_probability and feature must share a floating dtype")
        if not (
            feature.device == base_probability.device == occupancy.device
        ):
            raise ValueError("decoder inputs must share a device")
        if not torch.isfinite(feature).all() or not torch.isfinite(base_probability).all():
            raise ValueError("decoder inputs must be finite")
        if torch.any((base_probability < 0.0) | (base_probability > 1.0)):
            raise ValueError("base_probability must lie in [0,1]")

        projected = self.project(feature.detach())
        projected_full = F.interpolate(
            projected,
            size=base_probability.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        conditioning = [projected_full]
        if self.config.condition_on_probability:
            conditioning.append(base_probability.detach())
        conditioning.append(occupancy.detach().to(feature.dtype))
        return self.decode(torch.cat(conditioning, dim=1))


__all__ = ["CUREResidualDecoder"]
