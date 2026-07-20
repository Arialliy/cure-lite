"""A completely self-contained deterministic frozen base for toy CURE tests."""

from __future__ import annotations

import hashlib
import json

import torch
from torch import Tensor, nn

from ..frozen_base import FrozenBaseAdapter
from ..types import FrozenBaseOutput


_TOY_BASE_SCHEMA = "cure-lite-toy-frozen-base-v1"


class _ToyFrozenBase(nn.Module):
    """Fixed local filters plus a monotone single-pixel probability head."""

    def __init__(self) -> None:
        super().__init__()
        self.probability_head = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.feature_head = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.probability_head.weight.fill_(0.8)
            self.probability_head.bias.fill_(0.1)

            kernels = torch.zeros((3, 1, 3, 3), dtype=torch.float32)
            kernels[0, 0, 1, 1] = 1.0
            kernels[1, 0].fill_(1.0 / 9.0)
            kernels[2, 0, 1, 1] = 1.0
            kernels[2, 0] -= 1.0 / 9.0
            self.feature_head.weight.copy_(kernels)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        probability = self.probability_head(images).clamp(0.0, 1.0)
        feature = self.feature_head(images)
        return probability, feature


def _module_fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    digest.update(_TOY_BASE_SCHEMA.encode("ascii"))
    digest.update(
        json.dumps(
            {
                "input_channels": 1,
                "feature_channels": 3,
                "feature_stride": 1,
                "probability": "clamp(0.1 + 0.8*x, 0, 1)",
                "feature_filters": ("identity", "mean3x3", "highpass3x3"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ToyFrozenBaseAdapter(FrozenBaseAdapter):
    """Expose a frozen same-grid probability and three-channel toy feature."""

    def __init__(self) -> None:
        base = _ToyFrozenBase()
        super().__init__(base)
        self._fingerprint = _module_fingerprint(base)

    @property
    def feature_channels(self) -> int:
        return 3

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def extract(self, images: Tensor) -> FrozenBaseOutput:
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("images must have shape [B,1,H,W]")
        if images.shape[0] < 1 or images.shape[1] != 1:
            raise ValueError("images must be a non-empty single-channel batch")
        if images.dtype != torch.float32:
            raise TypeError("toy base images must be float32")
        if not torch.isfinite(images).all() or torch.any(
            (images < 0.0) | (images > 1.0)
        ):
            raise ValueError("toy base images must be finite and lie in [0,1]")

        self._freeze_base()
        with torch.no_grad():
            probability, feature = self.base(images)
        return FrozenBaseOutput(
            probability=probability.detach().to(torch.float32),
            feature=feature.detach(),
        )


__all__ = ["ToyFrozenBaseAdapter"]
