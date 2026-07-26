"""Frozen-base hard-union composition for the additive SVEF v4 decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import OccupancyConfig
from .factorized_decoder import CURELiteFactorizedDecoder
from .frozen_base import FrozenBaseAdapter
from .model import CURELiteOutput
from .types import FrozenBaseOutput


class CURELiteFactorizedModel(nn.Module):
    """Compose any receipt-bound frozen adapter with one SVEF decoder."""

    def __init__(
        self,
        base: FrozenBaseAdapter,
        decoder: CURELiteFactorizedDecoder,
        occupancy_config: OccupancyConfig | float = OccupancyConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(base, FrozenBaseAdapter):
            raise TypeError("base must implement FrozenBaseAdapter")
        if not isinstance(decoder, CURELiteFactorizedDecoder):
            raise TypeError("decoder must be CURELiteFactorizedDecoder")
        if isinstance(occupancy_config, (int, float)) and not isinstance(
            occupancy_config,
            bool,
        ):
            occupancy_config = OccupancyConfig(
                threshold=float(occupancy_config)
            )
        if not isinstance(occupancy_config, OccupancyConfig):
            raise TypeError(
                "occupancy_config must be OccupancyConfig or a threshold"
            )
        if base.feature_channels != decoder.feature_channels:
            raise ValueError(
                "base feature channels and decoder input channels must agree "
                f"({base.feature_channels} != {decoder.feature_channels})"
            )
        declared_stride = getattr(base, "feature_stride", None)
        if declared_stride is not None:
            if (
                isinstance(declared_stride, bool)
                or not isinstance(declared_stride, int)
                or declared_stride < 1
            ):
                raise ValueError(
                    "an adapter feature_stride declaration must be a "
                    "positive integer"
                )
            if declared_stride != decoder.feature_stride:
                raise ValueError(
                    "adapter and decoder feature strides must agree "
                    f"({declared_stride} != {decoder.feature_stride})"
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

    def train(self, mode: bool = True) -> "CURELiteFactorizedModel":
        super().train(mode)
        self._freeze_base()
        self.decoder.train(mode)
        return self

    def trainable_parameters(self):
        return self.decoder.parameters()

    def forward(
        self,
        images: Tensor,
        residual_threshold: float | None = None,
    ) -> CURELiteOutput:
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("images must have shape [B,C,H,W]")
        if images.shape[0] < 1 or not images.is_floating_point():
            raise ValueError(
                "images must be a non-empty floating-point batch"
            )
        if residual_threshold is not None:
            if isinstance(residual_threshold, bool) or not isinstance(
                residual_threshold,
                (int, float),
            ):
                raise TypeError("residual_threshold must be numeric or None")
            residual_threshold = float(residual_threshold)
            if not 0.0 <= residual_threshold <= 1.0:
                raise ValueError(
                    "residual_threshold must lie in [0,1]"
                )

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
        self.base.validate_output(base_output, images)
        base_probability = base_output.probability.detach()
        feature = base_output.feature.detach()
        feature_height, feature_width = (
            int(value) for value in feature.shape[-2:]
        )
        output_height, output_width = (
            int(value) for value in base_probability.shape[-2:]
        )
        if (
            output_height % feature_height == 0
            and output_width % feature_width == 0
        ):
            height_stride = output_height // feature_height
            width_stride = output_width // feature_width
            if height_stride == width_stride:
                if height_stride != self.decoder.feature_stride:
                    raise ValueError(
                        "runtime feature/evaluation ratio differs from the "
                        "decoder feature_stride "
                        f"({height_stride} != {self.decoder.feature_stride})"
                    )
        occupancy = base_probability >= self.occupancy_threshold
        residual_logits = self.decoder(feature, occupancy)
        residual_probability = torch.sigmoid(residual_logits)
        residual_probability = residual_probability.masked_fill(
            occupancy,
            0.0,
        )
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
        return self.forward(
            images,
            residual_threshold=residual_threshold,
        )


__all__ = ["CURELiteFactorizedModel"]
