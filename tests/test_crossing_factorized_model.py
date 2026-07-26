from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cure_lite.crossing_factorized_decoder import (
    CURELiteCrossingFactorizedDecoder,
)
from cure_lite.factorized_model import CURELiteFactorizedModel
from cure_lite.frozen_base import FrozenBaseAdapter
from cure_lite.types import FrozenBaseOutput


class _CountingBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return images * self.weight


class _ContractAdapter(FrozenBaseAdapter):
    """A detector-independent adapter exercising only the public contract."""

    def __init__(
        self,
        *,
        channels: int = 2,
        stride: int = 2,
    ) -> None:
        self._channels = channels
        self._stride = stride
        self.extract_calls = 0
        super().__init__(_CountingBase())

    @property
    def feature_channels(self) -> int:
        return self._channels

    @property
    def feature_stride(self) -> int:
        return self._stride

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            b"crossing-factorized-contract-adapter"
        ).hexdigest()

    def validate_preprocessing(self, preprocessing: object) -> None:
        del preprocessing

    def extract(self, images: torch.Tensor) -> FrozenBaseOutput:
        self.extract_calls += 1
        base_feature = self.base(images)
        probability = torch.zeros(
            images.shape[0],
            1,
            images.shape[-2],
            images.shape[-1],
            dtype=torch.float32,
            device=images.device,
        )
        probability[:, :, 1, 2] = 0.9
        feature = F.avg_pool2d(
            base_feature[:, :1],
            kernel_size=self._stride,
        ).repeat(1, self._channels, 1, 1)
        return FrozenBaseOutput(
            probability=probability.detach(),
            feature=feature.detach(),
        )


class _NoStrideContractAdapter(FrozenBaseAdapter):
    """The base contract does not require a declared feature stride."""

    def __init__(self, *, actual_stride: int = 4) -> None:
        self._actual_stride = actual_stride
        self.extract_calls = 0
        super().__init__(_CountingBase())

    @property
    def feature_channels(self) -> int:
        return 2

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            b"crossing-factorized-no-stride-contract-adapter"
        ).hexdigest()

    def validate_preprocessing(self, preprocessing: object) -> None:
        del preprocessing

    def extract(self, images: torch.Tensor) -> FrozenBaseOutput:
        self.extract_calls += 1
        base_feature = self.base(images)
        probability = torch.zeros(
            images.shape[0],
            1,
            images.shape[-2],
            images.shape[-1],
            dtype=torch.float32,
            device=images.device,
        )
        feature = F.avg_pool2d(
            base_feature[:, :1],
            kernel_size=self._actual_stride,
        ).repeat(1, 2, 1, 1)
        return FrozenBaseOutput(
            probability=probability.detach(),
            feature=feature.detach(),
        )


def _model() -> tuple[
    _ContractAdapter,
    CURELiteCrossingFactorizedDecoder,
    CURELiteFactorizedModel,
]:
    adapter = _ContractAdapter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7701)
        decoder = CURELiteCrossingFactorizedDecoder(
            feature_channels=adapter.feature_channels,
            feature_stride=adapter.feature_stride,
        )
    return adapter, decoder, CURELiteFactorizedModel(adapter, decoder)


def _module_graph(
    model: nn.Module,
) -> tuple[tuple[str, type[nn.Module]], ...]:
    return tuple(
        (name, type(module)) for name, module in model.named_modules()
    )


def _assert_hard_union(output: object) -> None:
    occupancy = output.occupancy
    residual_probability = output.residual_probability
    residual_mask = output.residual_mask
    final_mask = output.final_mask

    assert occupancy[:, :, 1, 2].all()
    assert torch.equal(
        residual_probability[occupancy],
        torch.zeros_like(residual_probability[occupancy]),
    )
    assert not torch.any(residual_mask & occupancy)
    assert torch.equal(final_mask, occupancy | residual_mask)


def test_existing_factorized_model_accepts_v7_contract_directly() -> None:
    adapter, decoder, model = _model()
    decoder_calls = 0

    def count_decoder(*_args: object) -> None:
        nonlocal decoder_calls
        decoder_calls += 1

    handle = decoder.register_forward_hook(count_decoder)
    try:
        output = model(
            torch.randn(2, 1, 8, 8),
            residual_threshold=0.0,
        )
    finally:
        handle.remove()

    assert type(model) is CURELiteFactorizedModel
    assert model.base is adapter
    assert model.decoder is decoder
    assert tuple(model._modules) == ("base", "decoder")
    assert adapter.extract_calls == 1
    assert adapter.base.forward_calls == 1
    assert decoder_calls == 1
    _assert_hard_union(output)

    assert adapter.training is False
    assert all(
        not parameter.requires_grad for parameter in adapter.parameters()
    )
    assert tuple(model.trainable_parameters()) == tuple(
        decoder.parameters()
    )
    trainable_names = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert trainable_names
    assert all(name.startswith("decoder.") for name in trainable_names)


def test_train_and_eval_keep_one_shared_graph_and_forward_path() -> None:
    adapter, decoder, model = _model()
    graph = _module_graph(model)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    decoder_calls = 0

    def count_decoder(*_args: object) -> None:
        nonlocal decoder_calls
        decoder_calls += 1

    handle = decoder.register_forward_hook(count_decoder)
    images = torch.randn(1, 1, 8, 8)
    try:
        model.train()
        assert model.training is True
        assert decoder.training is True
        assert adapter.training is False
        train_output = model(images, residual_threshold=0.0)

        model.eval()
        assert model.training is False
        assert decoder.training is False
        assert adapter.training is False
        eval_output = model(images, residual_threshold=0.0)
    finally:
        handle.remove()

    assert _module_graph(model) == graph
    assert tuple(id(parameter) for parameter in model.parameters()) == (
        parameter_ids
    )
    assert tuple(model._modules) == ("base", "decoder")
    assert adapter.extract_calls == 2
    assert adapter.base.forward_calls == 2
    assert decoder_calls == 2
    _assert_hard_union(train_output)
    _assert_hard_union(eval_output)
    torch.testing.assert_close(
        train_output.residual_logits,
        eval_output.residual_logits,
    )


def test_v7_model_rejects_declared_channel_or_stride_mismatch() -> None:
    adapter = _ContractAdapter(channels=2, stride=2)
    wrong_channels = CURELiteCrossingFactorizedDecoder(
        feature_channels=3,
        feature_stride=2,
    )
    wrong_stride = CURELiteCrossingFactorizedDecoder(
        feature_channels=2,
        feature_stride=1,
    )

    with pytest.raises(ValueError, match="channels"):
        CURELiteFactorizedModel(adapter, wrong_channels)
    with pytest.raises(ValueError, match="strides"):
        CURELiteFactorizedModel(adapter, wrong_stride)


def test_v7_model_runtime_validates_an_undeclared_feature_stride() -> None:
    matching_adapter = _NoStrideContractAdapter(actual_stride=4)
    matching = CURELiteFactorizedModel(
        matching_adapter,
        CURELiteCrossingFactorizedDecoder(
            feature_channels=2,
            feature_stride=4,
        ),
    )
    output = matching(torch.randn(1, 1, 16, 16))
    assert output.residual_logits.shape == (1, 1, 16, 16)
    assert matching_adapter.extract_calls == 1
    assert matching_adapter.base.forward_calls == 1

    mismatched_adapter = _NoStrideContractAdapter(actual_stride=4)
    mismatched = CURELiteFactorizedModel(
        mismatched_adapter,
        CURELiteCrossingFactorizedDecoder(
            feature_channels=2,
            feature_stride=2,
        ),
    )
    with pytest.raises(ValueError, match="runtime feature/evaluation ratio"):
        mismatched(torch.randn(1, 1, 16, 16))
    assert mismatched_adapter.extract_calls == 1
    assert mismatched_adapter.base.forward_calls == 1
