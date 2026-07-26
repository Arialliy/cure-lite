from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from cure_lite.directed_factorized_decoder import (
    CURELiteDirectedFactorizedDecoder,
)
from cure_lite.factorized_model import CURELiteFactorizedModel
from cure_lite.frozen_base import FrozenBaseAdapter
from cure_lite.types import FrozenBaseOutput


class _BaseModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.weight


class _Adapter(FrozenBaseAdapter):
    def __init__(self) -> None:
        self.extract_calls = 0
        super().__init__(_BaseModule())

    @property
    def feature_channels(self) -> int:
        return 2

    @property
    def feature_stride(self) -> int:
        return 2

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(b"directed-factorized-adapter").hexdigest()

    def validate_preprocessing(self, preprocessing: object) -> None:
        del preprocessing

    def extract(self, images: torch.Tensor) -> FrozenBaseOutput:
        self.extract_calls += 1
        probability = torch.zeros(
            images.shape[0],
            1,
            images.shape[-2],
            images.shape[-1],
            device=images.device,
        )
        probability[:, :, 1, 2] = 0.9
        feature = F.avg_pool2d(
            images[:, :1],
            kernel_size=2,
        ).repeat(1, 2, 1, 1)
        return FrozenBaseOutput(
            probability=probability.detach(),
            feature=feature.detach(),
        )


def test_existing_factorized_model_accepts_v5_without_new_wrapper() -> None:
    adapter = _Adapter()
    decoder = CURELiteDirectedFactorizedDecoder(
        feature_channels=2,
        feature_stride=2,
    )
    model = CURELiteFactorizedModel(adapter, decoder)
    decoder_calls = 0

    def count_decoder(*_args) -> None:
        nonlocal decoder_calls
        decoder_calls += 1

    handle = model.decoder.register_forward_hook(count_decoder)
    try:
        output = model(torch.randn(2, 1, 8, 8), residual_threshold=0.0)
    finally:
        handle.remove()

    assert adapter.extract_calls == 1
    assert decoder_calls == 1
    assert output.occupancy[:, :, 1, 2].all()
    assert not torch.any(output.residual_mask & output.occupancy)
    assert torch.equal(
        output.final_mask,
        output.occupancy | output.residual_mask,
    )
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    assert tuple(model.trainable_parameters()) == tuple(decoder.parameters())
