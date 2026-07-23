"""Frozen-base CURE-Lite composition and monotone hard-union inference."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import OccupancyConfig
from .decoder import CURELiteDecoder
from .frozen_base import FrozenBaseAdapter
from .types import FrozenBaseOutput


@dataclass(frozen=True)
class CURELiteOutput(Mapping[str, Tensor]):
    base_probability: Tensor
    occupancy: Tensor
    residual_logits: Tensor
    residual_probability: Tensor
    residual_mask: Tensor
    final_mask: Tensor

    _KEYS = (
        "base_probability",
        "occupancy",
        "residual_logits",
        "residual_probability",
        "residual_mask",
        "final_mask",
    )

    def __getitem__(self, key: str) -> Tensor:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)


class CURELiteModel(nn.Module):
    """Compose a frozen detector adapter with the trainable residual decoder."""

    def __init__(
        self,
        base: FrozenBaseAdapter,
        decoder: CURELiteDecoder,
        occupancy_config: OccupancyConfig | float = OccupancyConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(base, FrozenBaseAdapter):
            raise TypeError("base must implement FrozenBaseAdapter")
        if not isinstance(decoder, CURELiteDecoder):
            raise TypeError("decoder must be CURELiteDecoder")
        if isinstance(occupancy_config, (int, float)) and not isinstance(occupancy_config, bool):
            occupancy_config = OccupancyConfig(threshold=float(occupancy_config))
        if not isinstance(occupancy_config, OccupancyConfig):
            raise TypeError("occupancy_config must be OccupancyConfig or a threshold")
        if base.feature_channels != decoder.feature_channels:
            raise ValueError(
                "base feature channels and decoder input channels must agree "
                f"({base.feature_channels} != {decoder.feature_channels})"
            )
        self.base = base
        self.decoder = decoder
        self.occupancy_config = occupancy_config
        self._freeze_base()

    @property
    def occupancy_threshold(self) -> float:
        return self.occupancy_config.threshold

    def _freeze_base(self) -> None:
        self.base.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True) -> "CURELiteModel":
        super().train(mode)
        self._freeze_base()
        self.decoder.train(mode)
        return self

    def trainable_parameters(self):
        """Return the only parameters permitted in the optimizer."""

        return self.decoder.parameters()

    def forward(
        self,
        images: Tensor,
        residual_threshold: float | None = None,
    ) -> CURELiteOutput:
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("images must have shape [B,C,H,W]")
        if images.shape[0] < 1 or not images.is_floating_point():
            raise ValueError("images must be a non-empty floating-point batch")
        if residual_threshold is not None:
            if isinstance(residual_threshold, bool) or not isinstance(
                residual_threshold, (int, float)
            ):
                raise TypeError("residual_threshold must be numeric or None")
            residual_threshold = float(residual_threshold)
            if not 0.0 <= residual_threshold <= 1.0:
                raise ValueError("residual_threshold must lie in [0,1]")

        self._freeze_base()
        with torch.no_grad():
            base_output = self.base.extract(images)
        return self._compose_from_base_output(
            images,
            base_output,
            residual_threshold=residual_threshold,
        )

    def _compose_from_base_output(
        self,
        images: Tensor,
        base_output: FrozenBaseOutput,
        *,
        residual_threshold: float | None,
    ) -> CURELiteOutput:
        """Compose one already-computed frozen output without rerunning Base."""

        self.base.validate_output(base_output, images)
        base_probability = base_output.probability.detach()
        feature = base_output.feature.detach()
        occupancy = base_probability >= self.occupancy_threshold
        residual_logits = self.decoder(feature, occupancy)
        residual_probability = torch.sigmoid(residual_logits)
        residual_probability = residual_probability.masked_fill(occupancy, 0.0)
        if residual_threshold is None:
            residual_mask = torch.zeros_like(occupancy)
        else:
            residual_mask = residual_probability >= residual_threshold
            residual_mask &= ~occupancy
        final_mask = occupancy | residual_mask
        return CURELiteOutput(
            base_probability=base_probability,
            occupancy=occupancy,
            residual_logits=residual_logits,
            residual_probability=residual_probability,
            residual_mask=residual_mask,
            final_mask=final_mask,
        )

    def infer(
        self,
        images: Tensor,
        residual_threshold: float | None,
    ) -> CURELiteOutput:
        return self.forward(images, residual_threshold=residual_threshold)


__all__ = ["CURELiteModel", "CURELiteOutput"]
