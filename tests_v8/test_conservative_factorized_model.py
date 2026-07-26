from __future__ import annotations

import torch

from cure_lite.conservative_factorized_decoder import (
    CURELiteConservativeFactorizedDecoder,
)
from cure_lite.factorized_model import CURELiteFactorizedModel
from tests.test_crossing_factorized_model import (
    _ContractAdapter,
    _assert_hard_union,
)


def test_existing_detector_independent_model_accepts_v8_directly() -> None:
    adapter = _ContractAdapter(channels=2, stride=2)
    decoder = CURELiteConservativeFactorizedDecoder(
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


def test_stride_one_is_the_exact_single_phase_degeneracy() -> None:
    decoder = CURELiteConservativeFactorizedDecoder(
        feature_channels=3,
        feature_stride=1,
    )
    feature = torch.randn(2, 3, 7, 9)
    occupancy = torch.zeros(2, 1, 7, 9, dtype=torch.bool)
    fields = decoder.forward_fields(feature, occupancy)

    assert fields.raw_phase_evidence.shape == (2, 1, 7, 9)
    assert torch.equal(
        fields.phase_allocation,
        torch.ones_like(fields.phase_allocation),
    )
    assert torch.equal(
        fields.allocated_phase_evidence,
        fields.evidence_budget,
    )
    assert fields.logits.shape == occupancy.shape
