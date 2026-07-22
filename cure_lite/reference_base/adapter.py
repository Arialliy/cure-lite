"""Frozen-output adapter for the project-owned reference detector."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from ..data import PreprocessConfig
from ..frozen_base import FrozenBaseAdapter
from ..types import FrozenBaseOutput
from .model import ReferenceBaseNetwork


def _sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


class ReferenceBaseAdapter(FrozenBaseAdapter):
    """Expose the reference model through the detector-neutral Base contract."""

    def __init__(
        self,
        model: ReferenceBaseNetwork,
        preprocessing: PreprocessConfig,
        base_fingerprint: str,
    ) -> None:
        if not isinstance(model, ReferenceBaseNetwork):
            raise TypeError("model must be ReferenceBaseNetwork")
        if not isinstance(preprocessing, PreprocessConfig):
            raise TypeError("preprocessing must be PreprocessConfig")
        self._preprocessing = preprocessing
        self._fingerprint = _sha256(
            base_fingerprint,
            name="base_fingerprint",
        )
        super().__init__(model)

    @property
    def reference_model(self) -> ReferenceBaseNetwork:
        return self.base  # type: ignore[return-value]

    @property
    def feature_channels(self) -> int:
        return self.reference_model.feature_channels

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def preprocessing(self) -> PreprocessConfig:
        return self._preprocessing

    def validate_preprocessing(self, preprocessing: object) -> None:
        if preprocessing != self._preprocessing:
            raise ValueError("preprocessing differs from the trained reference base")

    def extract(self, images: Tensor) -> FrozenBaseOutput:
        if images.dtype != torch.float32:
            raise TypeError("reference-base images must be float32")
        output = self.reference_model.forward_with_feature(images)
        return FrozenBaseOutput(
            probability=torch.sigmoid(output.logits).to(torch.float32).detach(),
            feature=output.feature.to(torch.float32).detach(),
        )


def load_reference_base_adapter(
    run_dir: str | Path,
    *,
    device: str | torch.device,
) -> ReferenceBaseAdapter:
    """Load a completed reference-base run without any external model code."""

    from .training import load_reference_base_run

    loaded = load_reference_base_run(run_dir, device=device)
    return ReferenceBaseAdapter(
        loaded.model,
        loaded.config.preprocess,
        loaded.base_fingerprint,
    )


__all__ = ["ReferenceBaseAdapter", "load_reference_base_adapter"]
