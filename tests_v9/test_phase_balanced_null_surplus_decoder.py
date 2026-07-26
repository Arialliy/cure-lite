from __future__ import annotations

import torch
from torch.nn import functional as F

from cure_lite.factorized_config import FactorizedDecoderConfig
from cure_lite.factorized_decoder import CURELiteFactorizedDecoder
from cure_lite.phase_balanced_null_surplus_factorized_config import (
    PhaseBalancedNullSurplusFactorizedDecoderConfig,
)
from cure_lite.phase_balanced_null_surplus_factorized_decoder import (
    CURELitePhaseBalancedNullSurplusFactorizedDecoder,
)


def _decoder(
    *,
    channels: int = 3,
    stride: int = 4,
    seed: int = 9117,
) -> CURELitePhaseBalancedNullSurplusFactorizedDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CURELitePhaseBalancedNullSurplusFactorizedDecoder(
            PhaseBalancedNullSurplusFactorizedDecoderConfig(
                feature_channels=channels,
                feature_stride=stride,
            )
        )


def test_decoder_keeps_exact_v4_topology_state_and_parameter_budget() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2027)
        v4 = CURELiteFactorizedDecoder(
            FactorizedDecoderConfig(64, 4)
        )
        torch.manual_seed(2027)
        v9 = CURELitePhaseBalancedNullSurplusFactorizedDecoder(
            PhaseBalancedNullSurplusFactorizedDecoderConfig(64, 4)
        )

    assert tuple(v4.state_dict()) == tuple(v9.state_dict())
    for name, expected in v4.state_dict().items():
        assert torch.equal(v9.state_dict()[name], expected)
    assert sum(parameter.numel() for parameter in v9.parameters()) == 4385
    assert len(tuple(v9.parameters())) == 6
    assert tuple(type(module) for module in tuple(v9.modules())[1:]) == tuple(
        type(module) for module in tuple(v4.modules())[1:]
    )


def test_native_phase_surplus_is_pixel_shuffled_before_composition() -> None:
    decoder = _decoder(channels=3, stride=4)
    feature = torch.randn(2, 3, 5, 7)
    occupancy = torch.zeros(2, 1, 20, 28, dtype=torch.bool)
    occupancy[0, 0, 9, 13] = True

    fields = decoder.forward_fields(feature, occupancy)

    assert fields.raw_phase_evidence.shape == (2, 16, 5, 7)
    assert fields.implicit_null_threshold.shape == (2, 1, 5, 7)
    assert fields.native_phase_evidence.shape == (2, 16, 5, 7)
    assert fields.evidence.shape == (2, 1, 20, 28)
    assert fields.field_resize_applied is False
    assert torch.equal(
        fields.evidence,
        decoder.pixel_shuffle(fields.native_phase_evidence),
    )
    assert torch.equal(fields.logits, fields.baseline_logits + fields.evidence)


def test_resize_occurs_only_after_native_pixel_shuffle() -> None:
    decoder = _decoder(channels=3, stride=4)
    feature = torch.randn(1, 3, 3, 5)
    occupancy = torch.zeros(1, 1, 13, 21, dtype=torch.bool)

    fields = decoder.forward_fields(feature, occupancy)
    native = decoder.pixel_shuffle(fields.native_phase_evidence)
    expected = F.interpolate(
        native,
        size=(13, 21),
        mode="bilinear",
        align_corners=False,
    )

    assert fields.native_subpixel_size == (12, 20)
    assert fields.output_size == (13, 21)
    assert fields.field_resize_applied is True
    assert torch.equal(fields.evidence, expected)


def test_endpoint_selection_is_invariant_and_count_effect_is_local() -> None:
    decoder = _decoder(channels=2, stride=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(22)
        feature = torch.randn(1, 2, 4, 4)
    plus = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    plus[0, 0, 2, 2] = True
    minus = plus.clone()
    minus[0, 0, 2, 2] = False

    plus_fields = decoder.forward_fields(feature, plus)
    minus_fields = decoder.forward_fields(feature, minus)
    assert torch.equal(
        plus_fields.active_phase_mask,
        minus_fields.active_phase_mask,
    )

    count_release = (
        plus_fields.local_occupancy_count
        - minus_fields.local_occupancy_count
    )
    native_support = count_release > 0.0
    output_support = F.pixel_shuffle(
        native_support.expand(-1, 4, -1, -1).to(torch.float32),
        2,
    ).to(torch.bool)
    delta = minus_fields.logits - plus_fields.logits

    assert torch.all(delta >= 0.0)
    assert torch.any(delta[output_support] > 0.0)
    assert torch.equal(
        delta[~output_support],
        torch.zeros_like(delta[~output_support]),
    )


def test_identical_endpoints_are_bitwise_identical() -> None:
    decoder = _decoder(channels=2, stride=2)
    feature = torch.randn(2, 2, 4, 4)
    occupancy = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    occupancy[:, :, 2, 2] = True

    first = decoder.forward_fields(feature, occupancy)
    second = decoder.forward_fields(
        feature.clone(),
        occupancy.clone(),
    )
    for name in (
        "baseline_logits",
        "raw_phase_evidence",
        "phase_intensity",
        "implicit_null_threshold",
        "signed_phase_surplus",
        "active_phase_mask",
        "native_phase_evidence",
        "evidence",
        "logits",
        "projected_occupancy",
        "local_occupancy_count",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_zero_feature_is_exactly_null_for_every_occupancy() -> None:
    decoder = _decoder(channels=3, stride=4)
    feature = torch.zeros(1, 3, 3, 3)
    plus = torch.zeros(1, 1, 12, 12, dtype=torch.bool)
    plus[0, 0, 4, 4] = True
    minus = torch.zeros_like(plus)

    plus_fields = decoder.forward_fields(feature, plus)
    minus_fields = decoder.forward_fields(feature, minus)

    assert torch.equal(
        plus_fields.raw_phase_evidence,
        torch.zeros_like(plus_fields.raw_phase_evidence),
    )
    assert torch.equal(
        minus_fields.raw_phase_evidence,
        torch.zeros_like(minus_fields.raw_phase_evidence),
    )
    assert torch.equal(
        plus_fields.evidence,
        torch.zeros_like(plus_fields.evidence),
    )
    assert torch.equal(
        minus_fields.evidence,
        torch.zeros_like(minus_fields.evidence),
    )
    assert torch.equal(plus_fields.logits, minus_fields.logits)


def test_paired_endpoints_reach_all_six_parameter_tensors() -> None:
    decoder = _decoder(channels=3, stride=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        feature = torch.randn(2, 3, 5, 5, requires_grad=True)
    plus = torch.zeros(2, 1, 10, 10, dtype=torch.bool)
    plus[:, :, 4:6, 4:6] = True
    minus = torch.zeros_like(plus)

    delta = torch.sigmoid(decoder(feature, minus)) - torch.sigmoid(
        decoder(feature, plus)
    )
    loss = (delta - 0.5).square().mean()
    loss.backward()

    parameters = tuple(decoder.named_parameters())
    assert len(parameters) == 6
    for _, parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.any(parameter.grad != 0.0)
    assert feature.grad is None


def test_stride_one_decoder_retains_a_live_single_phase_operator() -> None:
    decoder = _decoder(channels=3, stride=1)
    feature = torch.randn(2, 3, 7, 9)
    occupancy = torch.zeros(2, 1, 7, 9, dtype=torch.bool)
    fields = decoder.forward_fields(feature, occupancy)

    assert fields.raw_phase_evidence.shape == (2, 1, 7, 9)
    assert fields.native_phase_evidence.shape == (2, 1, 7, 9)
    assert fields.logits.shape == occupancy.shape
