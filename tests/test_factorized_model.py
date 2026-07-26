from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cure_lite.factorized_decoder import CURELiteFactorizedDecoder
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
    def __init__(
        self,
        *,
        channels: int = 2,
        stride: int = 2,
    ) -> None:
        self._channels = channels
        self._stride = stride
        self.extract_calls = 0
        super().__init__(_BaseModule())

    @property
    def feature_channels(self) -> int:
        return self._channels

    @property
    def feature_stride(self) -> int:
        return self._stride

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(b"factorized-test-adapter").hexdigest()

    def validate_preprocessing(self, preprocessing: object) -> None:
        del preprocessing

    def extract(self, images: torch.Tensor) -> FrozenBaseOutput:
        self.extract_calls += 1
        probability = torch.zeros(
            images.shape[0],
            1,
            images.shape[-2],
            images.shape[-1],
            dtype=torch.float32,
            device=images.device,
        )
        probability[:, :, 1, 2] = 0.9
        pooled = F.avg_pool2d(images[:, :1], kernel_size=self._stride)
        feature = pooled.repeat(1, self._channels, 1, 1)
        return FrozenBaseOutput(
            probability=probability.detach(),
            feature=feature.detach(),
        )


class _NoStrideAdapter(FrozenBaseAdapter):
    def __init__(self, *, actual_stride: int = 4) -> None:
        self._actual_stride = actual_stride
        self.extract_calls = 0
        super().__init__(_BaseModule())

    @property
    def feature_channels(self) -> int:
        return 2

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(b"no-stride-test-adapter").hexdigest()

    def validate_preprocessing(self, preprocessing: object) -> None:
        del preprocessing

    def extract(self, images: torch.Tensor) -> FrozenBaseOutput:
        self.extract_calls += 1
        probability = torch.zeros(
            images.shape[0],
            1,
            images.shape[-2],
            images.shape[-1],
            dtype=torch.float32,
            device=images.device,
        )
        feature = F.avg_pool2d(
            images[:, :1],
            kernel_size=self._actual_stride,
        ).repeat(1, 2, 1, 1)
        return FrozenBaseOutput(
            probability=probability.detach(),
            feature=feature.detach(),
        )


def _model() -> tuple[_Adapter, CURELiteFactorizedModel]:
    adapter = _Adapter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(553)
        decoder = CURELiteFactorizedDecoder(
            feature_channels=adapter.feature_channels,
            feature_stride=adapter.feature_stride,
        )
    return adapter, CURELiteFactorizedModel(adapter, decoder)


def test_model_runs_base_once_and_preserves_hard_union() -> None:
    adapter, model = _model()
    images = torch.randn(2, 1, 8, 8)
    decoder_calls = 0

    def _count_decoder(*_args) -> None:
        nonlocal decoder_calls
        decoder_calls += 1

    handle = model.decoder.register_forward_hook(_count_decoder)
    try:
        output = model(images, residual_threshold=0.0)
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
    assert torch.equal(
        output.residual_probability[output.occupancy],
        torch.zeros_like(output.residual_probability[output.occupancy]),
    )
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    assert adapter.training is False


def test_none_threshold_preserves_single_score_path_without_residual_mask() -> None:
    adapter, model = _model()
    output = model(torch.randn(1, 1, 8, 8))

    assert adapter.extract_calls == 1
    assert not output.residual_mask.any()
    assert torch.equal(output.final_mask, output.occupancy)
    assert output.residual_logits.shape == (1, 1, 8, 8)


def test_train_keeps_base_frozen_and_only_exposes_decoder_parameters() -> None:
    adapter, model = _model()
    model.train(True)

    assert model.training is True
    assert model.decoder.training is True
    assert adapter.training is False
    assert tuple(model.trainable_parameters()) == tuple(
        model.decoder.parameters()
    )
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


def test_adapter_channel_or_stride_mismatch_is_rejected() -> None:
    adapter = _Adapter(channels=2, stride=2)
    wrong_channels = CURELiteFactorizedDecoder(
        feature_channels=3,
        feature_stride=2,
    )
    wrong_stride = CURELiteFactorizedDecoder(
        feature_channels=2,
        feature_stride=1,
    )

    with pytest.raises(ValueError, match="channels"):
        CURELiteFactorizedModel(adapter, wrong_channels)
    with pytest.raises(ValueError, match="strides"):
        CURELiteFactorizedModel(adapter, wrong_stride)


def test_runtime_ratio_binds_adapter_without_stride_property() -> None:
    adapter = _NoStrideAdapter(actual_stride=4)
    matching = CURELiteFactorizedModel(
        adapter,
        CURELiteFactorizedDecoder(
            feature_channels=2,
            feature_stride=4,
        ),
    )
    output = matching(torch.randn(1, 1, 16, 16))
    assert output.residual_logits.shape == (1, 1, 16, 16)

    mismatched_adapter = _NoStrideAdapter(actual_stride=4)
    mismatched = CURELiteFactorizedModel(
        mismatched_adapter,
        CURELiteFactorizedDecoder(
            feature_channels=2,
            feature_stride=2,
        ),
    )
    with pytest.raises(ValueError, match="runtime feature/evaluation ratio"):
        mismatched(torch.randn(1, 1, 16, 16))


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, "0.5"])
def test_invalid_residual_threshold_is_rejected(threshold: object) -> None:
    _, model = _model()
    with pytest.raises((TypeError, ValueError)):
        model(  # type: ignore[arg-type]
            torch.randn(1, 1, 8, 8),
            residual_threshold=threshold,
        )
