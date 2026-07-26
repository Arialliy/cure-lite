from __future__ import annotations

from math import log

import pytest
import torch
from torch.nn import functional as F

from cure_lite.factorized_config import FactorizedDecoderConfig
from cure_lite.factorized_decoder import CURELiteFactorizedDecoder
from cure_lite.recoverable_factorized_config import (
    RecoverableFactorizedDecoderConfig,
)
from cure_lite.recoverable_factorized_decoder import (
    CURELiteRecoverableFactorizedDecoder,
    polarity_recoverable_evidence,
)
from cure_lite.train.paired_step import _paired_endpoint_logits


def _decoder(
    *,
    channels: int = 3,
    stride: int = 4,
    seed: int = 7301,
) -> CURELiteRecoverableFactorizedDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CURELiteRecoverableFactorizedDecoder(
            RecoverableFactorizedDecoderConfig(
                feature_channels=channels,
                feature_stride=stride,
            )
        )


def _v4_and_v6(
    *,
    channels: int = 3,
    stride: int = 2,
    seed: int = 901,
) -> tuple[
    CURELiteFactorizedDecoder,
    CURELiteRecoverableFactorizedDecoder,
]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        v4 = CURELiteFactorizedDecoder(
            FactorizedDecoderConfig(channels, stride)
        )
        torch.manual_seed(seed)
        v6 = CURELiteRecoverableFactorizedDecoder(
            RecoverableFactorizedDecoderConfig(channels, stride)
        )
    return v4, v6


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


def test_operator_has_exact_forward_and_declared_surrogate_gradient() -> None:
    raw = torch.tensor(
        [-20.0, -1.0, -1.0e-6, 0.0, 1.0e-4, 0.5, 4.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    observed = polarity_recoverable_evidence(raw)
    positive = raw.detach()[4:]
    expected_positive = F.softplus(positive.square()) - log(2.0)

    assert torch.equal(observed[:4], torch.zeros_like(observed[:4]))
    torch.testing.assert_close(
        observed[4:],
        expected_positive,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    assert torch.all(observed >= 0.0)
    assert torch.isfinite(observed).all()

    observed.sum().backward()
    expected_negative = torch.sigmoid(raw.detach()[:4])
    expected_positive_gradient = (
        2.0
        * positive
        * torch.sigmoid(positive.square())
    )
    torch.testing.assert_close(
        raw.grad[:4],
        expected_negative,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        raw.grad[4:],
        expected_positive_gradient,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    assert float(raw.grad[3]) == 0.5


def test_surrogate_gradient_is_intentionally_not_negative_finite_difference() -> None:
    raw = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
    output = polarity_recoverable_evidence(raw)
    output.backward()
    surrogate_gradient = float(raw.grad)

    epsilon = 1.0e-5
    left = polarity_recoverable_evidence(
        torch.tensor(-0.5 - epsilon, dtype=torch.float64)
    )
    right = polarity_recoverable_evidence(
        torch.tensor(-0.5 + epsilon, dtype=torch.float64)
    )
    finite_difference = float((right - left) / (2.0 * epsilon))

    assert finite_difference == 0.0
    assert surrogate_gradient == pytest.approx(
        float(torch.sigmoid(torch.tensor(-0.5, dtype=torch.float64))),
        abs=1.0e-12,
    )


def test_negative_phase_can_cross_zero_under_a_fixed_target_objective() -> None:
    raw = torch.tensor(-0.13205, requires_grad=True)
    optimizer = torch.optim.SGD([raw], lr=0.2)

    initial = float(raw.detach())
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        evidence = polarity_recoverable_evidence(raw)
        loss = (evidence - 1.0).square()
        loss.backward()
        optimizer.step()

    assert initial < 0.0
    assert float(raw.detach()) > 0.0
    assert float(polarity_recoverable_evidence(raw).detach()) > 0.0


@pytest.mark.parametrize(
    "invalid",
    [
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
    ],
)
def test_operator_rejects_invalid_tensor_values(invalid: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        polarity_recoverable_evidence(invalid)
    with pytest.raises(TypeError):
        polarity_recoverable_evidence([0.0])  # type: ignore[arg-type]


def test_v4_and_v6_share_exact_state_modules_and_positive_forward() -> None:
    v4, v6 = _v4_and_v6()
    state4 = v4.state_dict()
    state6 = v6.state_dict()

    assert state4.keys() == state6.keys()
    for name in state4:
        assert state4[name].shape == state6[name].shape
        assert torch.equal(state4[name], state6[name]), name
    assert tuple(type(module) for module in v4.modules())[1:] == tuple(
        type(module) for module in v6.modules()
    )[1:]
    expected = FactorizedDecoderConfig(3, 2).expected_parameter_count
    assert sum(parameter.numel() for parameter in v4.parameters()) == expected
    assert sum(parameter.numel() for parameter in v6.parameters()) == expected

    positive = torch.tensor([0.1, 0.75, 3.0])
    v4_evidence = F.softplus(positive.square()) - log(2.0)
    v6_evidence = polarity_recoverable_evidence(positive)
    torch.testing.assert_close(
        v6_evidence,
        v4_evidence,
        rtol=0.0,
        atol=0.0,
    )


def test_only_evidence_and_logits_differ_for_same_state_and_input() -> None:
    v4, v6 = _v4_and_v6(channels=4, stride=2)
    feature = torch.randn(2, 4, 4, 5)
    occupancy = torch.rand(2, 1, 8, 10) > 0.8

    fields4 = v4.forward_fields(feature, occupancy)
    fields6 = v6.forward_fields(feature, occupancy)

    assert torch.equal(fields4.baseline_logits, fields6.baseline_logits)
    assert torch.equal(fields4.vacancy, fields6.vacancy)
    assert torch.equal(
        fields4.projected_occupancy,
        fields6.projected_occupancy,
    )
    assert torch.equal(
        fields4.local_occupancy_count,
        fields6.local_occupancy_count,
    )
    assert fields4.native_subpixel_size == fields6.native_subpixel_size
    assert fields4.output_size == fields6.output_size
    assert fields4.field_resize_applied == fields6.field_resize_applied

    _, raw = _raw_fields(v4, feature, output_size=(8, 10))
    expected6 = polarity_recoverable_evidence(raw)
    torch.testing.assert_close(
        fields6.evidence,
        expected6,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fields6.logits,
        fields6.baseline_logits + expected6 * fields6.vacancy,
        rtol=0.0,
        atol=0.0,
    )


def test_non_native_path_resizes_raw_before_recoverable_operator() -> None:
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
    expected_evidence = polarity_recoverable_evidence(evidence_raw)

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


def test_zero_identity_deletion_batch_and_detach_invariants_hold() -> None:
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

    logits_plus, logits_minus = _paired_endpoint_logits(
        decoder,
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
    )
    separate_plus = decoder(feature, plus)
    separate_minus = decoder(feature, minus)
    torch.testing.assert_close(
        logits_plus,
        separate_plus,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        logits_minus,
        separate_minus,
        rtol=0.0,
        atol=0.0,
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
