from __future__ import annotations

import torch

from cure_lite.factorized_model import CURELiteFactorizedModel
from cure_lite.phase_balanced_null_surplus_factorized_decoder import (
    CURELitePhaseBalancedNullSurplusFactorizedDecoder,
)
from tests.test_crossing_factorized_model import (
    _ContractAdapter,
    _assert_hard_union,
)


def test_existing_detector_independent_model_accepts_v9_directly() -> None:
    adapter = _ContractAdapter(channels=2, stride=2)
    decoder = CURELitePhaseBalancedNullSurplusFactorizedDecoder(
        feature_channels=2,
        feature_stride=2,
    )
    model = CURELiteFactorizedModel(adapter, decoder)
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
    assert all(
        not parameter.requires_grad for parameter in adapter.parameters()
    )
    assert tuple(model.trainable_parameters()) == tuple(
        decoder.parameters()
    )
