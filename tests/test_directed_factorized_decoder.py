from __future__ import annotations

from math import log

import pytest
import torch
from torch.nn import functional as F

from cure_lite.directed_factorized_config import (
    DirectedFactorizedDecoderConfig,
)
from cure_lite.directed_factorized_decoder import (
    CURELiteDirectedFactorizedDecoder,
    directed_evidence_activation,
)
from cure_lite.factorized_config import FactorizedDecoderConfig
from cure_lite.factorized_decoder import CURELiteFactorizedDecoder
from cure_lite.train.paired_step import _paired_endpoint_logits


def _decoder(
    *,
    channels: int = 3,
    stride: int = 4,
    seed: int = 7301,
) -> CURELiteDirectedFactorizedDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CURELiteDirectedFactorizedDecoder(
            DirectedFactorizedDecoderConfig(
                feature_channels=channels,
                feature_stride=stride,
            )
        )


def _v4_and_v5(
    *,
    channels: int = 3,
    stride: int = 2,
    seed: int = 901,
) -> tuple[CURELiteFactorizedDecoder, CURELiteDirectedFactorizedDecoder]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        v4 = CURELiteFactorizedDecoder(
            FactorizedDecoderConfig(channels, stride)
        )
        torch.manual_seed(seed)
        v5 = CURELiteDirectedFactorizedDecoder(
            DirectedFactorizedDecoderConfig(channels, stride)
        )
    return v4, v5


def _raw_fields(
    decoder: CURELiteFactorizedDecoder,
    feature: torch.Tensor,
    *,
    output_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    trunk0 = F.silu(decoder.stem_norm(decoder.stem(feature.detach())))
    residual = decoder.pointwise(
        F.silu(decoder.depthwise_norm(decoder.depthwise(trunk0)))
    )
    trunk = trunk0 + 0.5 * residual
    baseline = decoder.pixel_shuffle(decoder.baseline_head(trunk))
    evidence = decoder.pixel_shuffle(decoder.evidence_head(trunk))
    if tuple(baseline.shape[-2:]) != output_size:
        baseline = F.interpolate(
            baseline,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        evidence = F.interpolate(
            evidence,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
    return baseline, evidence


def test_directed_activation_has_exact_polarity_and_gradient_contract() -> None:
    raw = torch.tensor(
        [-20.0, -1.0, -1.0e-6, 0.0, 1.0e-6, 0.5, 20.0],
        requires_grad=True,
    )
    observed = directed_evidence_activation(raw)
    expected_positive = F.softplus(raw[4:]) - log(2.0)

    assert torch.equal(observed[:4], torch.zeros_like(observed[:4]))
    torch.testing.assert_close(
        observed[4:],
        expected_positive,
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert torch.all(observed >= 0.0)
    assert torch.isfinite(observed).all()

    observed.sum().backward()
    assert torch.equal(raw.grad[:4], torch.zeros_like(raw.grad[:4]))
    torch.testing.assert_close(
        raw.grad[4:],
        torch.sigmoid(raw.detach()[4:]),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert float(raw.grad[4]) == pytest.approx(0.5, abs=1.0e-6)
    assert float(observed[-1].detach()) == pytest.approx(
        20.0 - log(2.0),
        abs=1.0e-5,
    )


def test_v4_and_v5_share_exact_initial_state_but_not_signed_evidence() -> None:
    v4, v5 = _v4_and_v5()
    state4 = v4.state_dict()
    state5 = v5.state_dict()

    assert state4.keys() == state5.keys()
    for name in state4:
        assert state4[name].shape == state5[name].shape
        assert torch.equal(state4[name], state5[name]), name
    assert tuple(type(module) for module in v4.modules())[1:] == tuple(
        type(module) for module in v5.modules()
    )[1:]
    expected = FactorizedDecoderConfig(3, 2).expected_parameter_count
    assert sum(parameter.numel() for parameter in v4.parameters()) == expected
    assert sum(parameter.numel() for parameter in v5.parameters()) == expected

    signed = torch.tensor([-0.75, 0.75])
    v4_evidence = F.softplus(signed.square()) - log(2.0)
    v5_evidence = directed_evidence_activation(signed)
    assert torch.equal(v4_evidence[0], v4_evidence[1])
    assert float(v5_evidence[0]) == 0.0
    assert float(v5_evidence[1]) > 0.0


def test_only_evidence_and_logits_differ_for_same_state_and_input() -> None:
    v4, v5 = _v4_and_v5(channels=4, stride=2)
    feature = torch.randn(2, 4, 4, 5)
    occupancy = torch.rand(2, 1, 8, 10) > 0.8

    fields4 = v4.forward_fields(feature, occupancy)
    fields5 = v5.forward_fields(feature, occupancy)

    assert torch.equal(fields4.baseline_logits, fields5.baseline_logits)
    assert torch.equal(fields4.vacancy, fields5.vacancy)
    assert torch.equal(
        fields4.projected_occupancy,
        fields5.projected_occupancy,
    )
    assert torch.equal(
        fields4.local_occupancy_count,
        fields5.local_occupancy_count,
    )
    assert fields4.native_subpixel_size == fields5.native_subpixel_size
    assert fields4.output_size == fields5.output_size
    assert fields4.field_resize_applied == fields5.field_resize_applied
    assert not torch.equal(fields4.evidence, fields5.evidence)

    _, raw = _raw_fields(v4, feature, output_size=(8, 10))
    expected5 = directed_evidence_activation(raw)
    torch.testing.assert_close(
        fields5.evidence,
        expected5,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fields5.logits,
        fields5.baseline_logits + expected5 * fields5.vacancy,
        rtol=0.0,
        atol=0.0,
    )


def test_non_native_path_resizes_raw_before_directed_activation() -> None:
    decoder = _decoder(channels=3, stride=2)
    feature = torch.randn(1, 3, 2, 3)
    occupancy = torch.zeros(1, 1, 5, 7, dtype=torch.bool)
    occupancy[0, 0, 2, 3] = True

    fields = decoder.forward_fields(feature, occupancy)
    baseline_raw, evidence_raw = _raw_fields(
        decoder,
        feature,
        output_size=(5, 7),
    )
    expected_baseline = -F.softplus(
        decoder.baseline_raw.reshape(1, 1, 1, 1)
        + baseline_raw
    )
    expected_evidence = directed_evidence_activation(evidence_raw)

    assert fields.native_subpixel_size == (4, 6)
    assert fields.output_size == (5, 7)
    assert fields.field_resize_applied is True
    torch.testing.assert_close(
        fields.baseline_logits,
        expected_baseline,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fields.evidence,
        expected_evidence,
        rtol=0.0,
        atol=0.0,
    )


def test_zero_identity_deletion_support_and_detach_invariants_hold() -> None:
    decoder = _decoder(channels=4, stride=2)
    zero = torch.zeros(2, 4, 4, 5)
    occupancy_a = torch.zeros(2, 1, 8, 10, dtype=torch.bool)
    occupancy_b = torch.rand(2, 1, 8, 10) > 0.6
    assert torch.equal(
        decoder(zero, occupancy_a),
        decoder(zero, occupancy_b),
    )

    feature = torch.randn(2, 4, 4, 5, requires_grad=True)
    plus = torch.rand(2, 1, 8, 10) > 0.75
    minus = plus & ~(torch.rand(2, 1, 8, 10) > 0.7)
    plus_fields = decoder.forward_fields(feature, plus)
    minus_fields = decoder.forward_fields(feature, minus)
    gate_delta = minus_fields.vacancy - plus_fields.vacancy
    logit_delta = minus_fields.logits - plus_fields.logits

    assert float(gate_delta.min()) >= 0.0
    assert float(logit_delta.detach().min()) >= -1.0e-7
    unchanged = gate_delta == 0.0
    assert torch.count_nonzero(logit_delta[unchanged]) == 0
    torch.testing.assert_close(
        logit_delta,
        minus_fields.evidence * gate_delta,
        rtol=1.0e-4,
        atol=3.0e-7,
    )

    identity_plus, identity_minus = _paired_endpoint_logits(
        decoder,
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=plus,
    )
    assert torch.equal(identity_plus, identity_minus)
    identity_plus.square().mean().backward()
    assert feature.grad is None
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )
