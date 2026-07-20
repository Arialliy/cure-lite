"""Frozen detector adapter contract for CURE-Lite v0.1."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .types import FrozenBaseOutput


class FrozenBaseAdapter(nn.Module, ABC):
    """Expose a probability map and one feature from a permanently frozen base.

    Subclasses implement :meth:`extract`; callers should use ``torch.no_grad``
    rather than inference mode so the detached feature remains consumable by a
    trainable convolution.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if not isinstance(base, nn.Module):
            raise TypeError("base must be a torch.nn.Module")
        self.base = base
        self._freeze_base()

    @property
    def model(self) -> nn.Module:
        """Read-only compatibility alias; the module is registered only as ``base``."""

        return self.base

    @property
    @abstractmethod
    def feature_channels(self) -> int:
        """Number of channels in the exposed feature tensor."""

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Stable identity of weights, preprocessing, and extraction choices."""

    @abstractmethod
    def extract(self, images: Tensor) -> FrozenBaseOutput:
        """Return detached base outputs for an NCHW image batch."""

    def _freeze_base(self) -> None:
        self.base.requires_grad_(False)
        self.base.eval()

    def validate_output(
        self,
        output: FrozenBaseOutput,
        batch_size_or_images: int | Tensor,
    ) -> None:
        """Validate the frozen-output contract.

        An integer batch size is accepted for adapters that validate their own
        spatial grid. Passing the input tensor additionally verifies grid and
        device equality.
        """

        if not isinstance(output, FrozenBaseOutput):
            raise TypeError("extract() must return FrozenBaseOutput")
        probability, feature = output.probability, output.feature
        if not isinstance(probability, Tensor) or not isinstance(feature, Tensor):
            raise TypeError("base probability and feature must be tensors")
        if probability.ndim != 4 or probability.shape[1] != 1:
            raise ValueError("base probability must have shape [B,1,H,W]")
        if feature.ndim != 4:
            raise ValueError("base feature must have shape [B,C,h,w]")

        images: Tensor | None
        if isinstance(batch_size_or_images, Tensor):
            images = batch_size_or_images
            if images.ndim != 4:
                raise ValueError("images must have shape [B,C,H,W]")
            batch_size = int(images.shape[0])
        elif isinstance(batch_size_or_images, int) and not isinstance(batch_size_or_images, bool):
            images = None
            batch_size = batch_size_or_images
        else:
            raise TypeError("batch_size_or_images must be an integer or image tensor")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if probability.shape[0] != batch_size or feature.shape[0] != batch_size:
            raise ValueError("base output batch size does not match the input")
        if feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"expected {self.feature_channels} feature channels, got {feature.shape[1]}"
            )
        if probability.dtype != torch.float32:
            raise TypeError("base probability must be float32")
        if not feature.is_floating_point():
            raise TypeError("base feature must be floating point")
        if probability.requires_grad or feature.requires_grad:
            raise ValueError("frozen base outputs must be detached")
        if probability.device != feature.device:
            raise ValueError("base probability and feature must share a device")
        if not torch.isfinite(probability).all() or not torch.isfinite(feature).all():
            raise ValueError("base outputs must be finite")
        if torch.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("base probability must lie in [0,1]")
        if images is not None:
            if probability.shape[-2:] != images.shape[-2:]:
                raise ValueError("base probability must use the input evaluation grid")
            if probability.device != images.device:
                raise ValueError("base outputs and images must share a device")

    def forward(self, images: Tensor) -> FrozenBaseOutput:
        self._freeze_base()
        with torch.no_grad():
            output = self.extract(images)
        self.validate_output(output, images)
        return output

    def train(self, mode: bool = True) -> "FrozenBaseAdapter":
        # The adapter is part of the frozen base, so even adapter.train(True)
        # must not enable stochastic or running-statistics updates.
        super().train(False)
        self._freeze_base()
        return self


__all__ = ["FrozenBaseAdapter"]
