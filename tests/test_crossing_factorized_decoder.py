from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from cure_lite.crossing_factorized_config import (
    CrossingFactorizedDecoderConfig,
)
from cure_lite.crossing_factorized_decoder import (
    CURELiteCrossingFactorizedDecoder,
    CrossingFactorizedDecoderFields,
    crossing_recoverable_evidence,
)
from cure_lite.factorized_config import FactorizedDecoderConfig
from cure_lite.factorized_decoder import CURELiteFactorizedDecoder
from cure_lite.train.paired_step import _paired_endpoint_logits


def _decoder(
    *,
    channels: int = 3,
    stride: int = 4,
    seed: int = 7301,
) -> CURELiteCrossingFactorizedDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CURELiteCrossingFactorizedDecoder(
            CrossingFactorizedDecoderConfig(
                feature_channels=channels,
                feature_stride=stride,
            )
        )


def _v4_and_v7(
    *,
    channels: int = 3,
    stride: int = 2,
    seed: int = 901,
) -> tuple[
    CURELiteFactorizedDecoder,
    CURELiteCrossingFactorizedDecoder,
]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        v4 = CURELiteFactorizedDecoder(
            FactorizedDecoderConfig(channels, stride)
        )
        torch.manual_seed(seed)
        v7 = CURELiteCrossingFactorizedDecoder(
            CrossingFactorizedDecoderConfig(channels, stride)
        )
    return v4, v7


def _raw_fields(
    decoder: CURELiteCrossingFactorizedDecoder,
    feature: torch.Tensor,
    *,
    output_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    trunk0 = F.silu(decoder.stem_norm(decoder.stem(feature.detach())))
    residual = decoder.pointwise(
        F.silu(decoder.depthwise_norm(decoder.depthwise(trunk0)))
    )
    trunk = trunk0 + decoder.config.trunk_residual_scale * residual
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


def test_operator_has_exact_forward_and_full_axis_recovery_gradient() -> None:
    margin = torch.tensor(
        [-20.0, -1.0, -1.0e-6, 0.0, 1.0e-6, 0.5, 4.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    observed = crossing_recoverable_evidence(margin)
    positive = margin.detach()[4:]
    expected_positive = torch.expm1(positive)

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
    expected_negative = torch.exp(margin.detach()[:4])
    expected_positive_gradient = torch.exp(positive)
    torch.testing.assert_close(
        margin.grad[:4],
        expected_negative,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        margin.grad[4:],
        expected_positive_gradient,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    assert float(margin.grad[3]) == 1.0


def test_float32_positive_forward_is_bit_exact_at_multiple_scales() -> None:
    margin = torch.tensor(
        [
            -80.0,
            -1.0,
            0.0,
            1.0e-7,
            1.0e-4,
            0.25,
            1.0,
            8.0,
            40.0,
            88.0,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    observed = crossing_recoverable_evidence(margin)
    expected = torch.where(
        margin.detach() <= 0.0,
        torch.zeros_like(margin.detach()),
        torch.expm1(margin.detach()),
    )

    assert torch.equal(observed, expected)
    observed.sum().backward()
    assert margin.grad is not None
    assert torch.equal(margin.grad, torch.exp(margin.detach()))


def test_recovery_gradient_is_continuous_across_zero_boundary() -> None:
    epsilon = 1.0e-7
    margin = torch.tensor(
        [-epsilon, 0.0, epsilon],
        dtype=torch.float64,
        requires_grad=True,
    )
    crossing_recoverable_evidence(margin).sum().backward()

    assert float(margin.grad[0]) < 1.0
    assert float(margin.grad[1]) == 1.0
    assert float(margin.grad[2]) > 1.0
    assert float((margin.grad[1] - margin.grad[0]).abs()) < 2.0e-7
    assert float((margin.grad[2] - margin.grad[1]).abs()) < 2.0e-7


def test_float32_negative_recovery_uses_exp_not_expm1_backward() -> None:
    margin = torch.tensor(
        [-80.0, -20.0, -17.0, -16.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    crossing_recoverable_evidence(margin).sum().backward()

    expected = torch.exp(margin.detach())
    assert margin.grad is not None
    assert torch.all(margin.grad > 0.0)
    torch.testing.assert_close(
        margin.grad,
        expected,
        rtol=0.0,
        atol=0.0,
    )


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
        crossing_recoverable_evidence(invalid)
    with pytest.raises(TypeError):
        crossing_recoverable_evidence([0.0])  # type: ignore[arg-type]


def test_operator_has_finite_safe_domain_and_fails_fast_on_overflow() -> None:
    safe = torch.tensor(
        [-80.0, -1.0, 0.0, 1.0, 88.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    observed = crossing_recoverable_evidence(safe)

    assert torch.isfinite(observed).all()
    assert torch.equal(observed[:3], torch.zeros_like(observed[:3]))
    assert torch.all(observed[3:] > 0.0)
    observed.sum().backward()
    assert safe.grad is not None
    assert torch.isfinite(safe.grad).all()
    assert torch.all(safe.grad > 0.0)
    torch.testing.assert_close(
        safe.grad,
        torch.exp(safe.detach()),
        rtol=0.0,
        atol=0.0,
    )

    finite_but_overflowing = torch.tensor(89.0, dtype=torch.float32)
    with pytest.raises(
        ValueError,
        match="margin, continuation, and recovery must remain",
    ):
        crossing_recoverable_evidence(finite_but_overflowing)

    underflowing_recovery = torch.tensor(-104.0, dtype=torch.float32)
    with pytest.raises(
        ValueError,
        match="margin, continuation, and recovery must remain",
    ):
        crossing_recoverable_evidence(underflowing_recovery)


def test_v4_and_v7_share_exact_state_modules_and_parameter_count() -> None:
    v4, v7 = _v4_and_v7()
    state4 = v4.state_dict()
    state7 = v7.state_dict()

    assert state4.keys() == state7.keys()
    for name in state4:
        assert state4[name].shape == state7[name].shape
        assert torch.equal(state4[name], state7[name]), name
    assert tuple(type(module) for module in v4.modules())[1:] == tuple(
        type(module) for module in v7.modules()
    )[1:]
    expected = FactorizedDecoderConfig(3, 2).expected_parameter_count
    assert sum(parameter.numel() for parameter in v4.parameters()) == expected
    assert sum(parameter.numel() for parameter in v7.parameters()) == expected


def test_fields_follow_the_frozen_crossing_equation_and_are_auditable() -> None:
    decoder = _decoder(channels=4, stride=2).double()
    feature = torch.randn(2, 4, 3, 5, dtype=torch.float64)
    occupancy = torch.rand(2, 1, 6, 10) > 0.8

    fields = decoder.forward_fields(feature, occupancy)
    assert isinstance(fields, CrossingFactorizedDecoderFields)
    baseline_raw, expected_raw = _raw_fields(
        decoder,
        feature,
        output_size=(6, 10),
    )
    expected_baseline = -F.softplus(
        decoder.baseline_raw.reshape(1, 1, 1, 1)
        + baseline_raw
    )
    projected, count, expected_burden = decoder.burden_field(
        occupancy,
        feature_size=(3, 5),
    )
    count_output = F.interpolate(
        count,
        size=(6, 10),
        mode="nearest",
    )
    expected_margin = expected_raw - expected_burden
    expected_evidence = crossing_recoverable_evidence(expected_margin)
    expected_ratio = torch.clamp_min(
        torch.exp(expected_raw) / (1.0 + count_output) - 1.0,
        0.0,
    )

    assert torch.equal(fields.projected_occupancy, projected)
    assert torch.equal(fields.local_occupancy_count, count)
    assert torch.equal(fields.raw_evidence, expected_raw)
    assert torch.equal(fields.occupancy_burden, expected_burden)
    assert torch.equal(fields.crossing_margin, expected_margin)
    assert torch.equal(fields.evidence, expected_evidence)
    torch.testing.assert_close(
        fields.evidence,
        expected_ratio,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    assert torch.equal(fields.baseline_logits, expected_baseline)
    assert torch.equal(
        fields.logits,
        fields.baseline_logits + fields.evidence,
    )
    for value in (
        fields.baseline_logits,
        fields.raw_evidence,
        fields.occupancy_burden,
        fields.crossing_margin,
        fields.evidence,
        fields.logits,
        fields.local_occupancy_count,
    ):
        assert value.dtype == torch.float64
        assert value.device == feature.device
        assert torch.isfinite(value).all()


def test_non_native_path_resizes_raw_and_burden_by_frozen_modes() -> None:
    decoder = _decoder(channels=3, stride=2)
    feature = torch.randn(1, 3, 2, 3)
    occupancy = torch.zeros(1, 1, 5, 7, dtype=torch.bool)
    occupancy[0, 0, 2, 3] = True

    fields = decoder.forward_fields(feature, occupancy)
    baseline_raw, expected_raw = _raw_fields(
        decoder,
        feature,
        output_size=(5, 7),
    )
    expected_baseline = -F.softplus(
        decoder.baseline_raw.reshape(1, 1, 1, 1)
        + baseline_raw
    )
    _, _count, expected_burden = decoder.burden_field(
        occupancy,
        feature_size=(2, 3),
    )

    assert fields.native_subpixel_size == (4, 6)
    assert fields.output_size == (5, 7)
    assert fields.field_resize_applied is True
    assert torch.equal(fields.baseline_logits, expected_baseline)
    assert torch.equal(fields.raw_evidence, expected_raw)
    assert torch.equal(fields.occupancy_burden, expected_burden)


def test_forward_never_calls_parent_reciprocal_vacancy_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _decoder(channels=3, stride=2)
    feature = torch.randn(1, 3, 2, 3)
    occupancy = torch.zeros(1, 1, 4, 6, dtype=torch.bool)
    occupancy[0, 0, 1, 2] = True

    def _forbidden_parent_path(*args: object, **kwargs: object) -> None:
        raise AssertionError("parent reciprocal vacancy path was called")

    monkeypatch.setattr(
        CURELiteFactorizedDecoder,
        "vacancy_field",
        _forbidden_parent_path,
    )
    fields = decoder.forward_fields(feature, occupancy)

    assert fields.logits.shape == (1, 1, 4, 6)
    assert torch.count_nonzero(fields.local_occupancy_count) > 0


def test_occupancy_deletion_is_monotone_and_exactly_local_to_count_change() -> None:
    decoder = _decoder(channels=4, stride=2, seed=447)
    feature = torch.randn(1, 4, 4, 5)
    occupancy_minus = torch.zeros(1, 1, 8, 10, dtype=torch.bool)
    empty_fields = decoder.forward_fields(feature, occupancy_minus)
    maximum = int(empty_fields.raw_evidence.reshape(-1).argmax())
    assert (
        float(
            empty_fields.raw_evidence.detach().reshape(-1)[maximum]
        )
        > 0.0
    )
    y, x = divmod(maximum, 10)

    occupancy_plus = occupancy_minus.clone()
    occupancy_plus[0, 0, y, x] = True
    plus_fields = decoder.forward_fields(feature, occupancy_plus)
    minus_fields = decoder.forward_fields(feature, occupancy_minus)
    changed = (
        plus_fields.local_occupancy_count
        != minus_fields.local_occupancy_count
    )
    changed_output = (
        plus_fields.occupancy_burden
        != minus_fields.occupancy_burden
    )

    assert torch.count_nonzero(changed) > 0
    assert torch.count_nonzero(changed_output) > 0
    assert torch.equal(
        plus_fields.baseline_logits,
        minus_fields.baseline_logits,
    )
    assert torch.equal(
        plus_fields.raw_evidence,
        minus_fields.raw_evidence,
    )
    assert torch.all(
        minus_fields.local_occupancy_count
        <= plus_fields.local_occupancy_count
    )
    assert torch.all(
        minus_fields.occupancy_burden
        <= plus_fields.occupancy_burden
    )
    assert torch.all(
        minus_fields.logits >= plus_fields.logits
    )
    assert torch.equal(
        minus_fields.logits[~changed_output],
        plus_fields.logits[~changed_output],
    )
    assert float(
        (
            minus_fields.logits[0, 0, y, x]
            - plus_fields.logits[0, 0, y, x]
        ).detach()
    ) > 0.0


def test_identity_two_endpoint_batch_and_detached_gradients_hold() -> None:
    decoder = _decoder(channels=4, stride=2)
    feature = torch.randn(2, 4, 4, 5, requires_grad=True)
    occupancy = torch.rand(2, 1, 8, 10) > 0.75

    identity_plus, identity_minus = _paired_endpoint_logits(
        decoder,
        feature=feature,
        occupancy_plus=occupancy,
        occupancy_minus=occupancy,
    )
    assert torch.equal(identity_plus, identity_minus)
    identity_plus.square().mean().backward()
    assert feature.grad is None
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in decoder.parameters()
    )


def test_two_endpoint_batched_forward_matches_separate_forwards() -> None:
    decoder = _decoder(channels=4, stride=2)
    feature = torch.randn(2, 4, 4, 5)
    occupancy_plus = torch.rand(2, 1, 8, 10) > 0.72
    occupancy_minus = occupancy_plus & ~(
        torch.rand(2, 1, 8, 10) > 0.8
    )

    batched_plus, batched_minus = _paired_endpoint_logits(
        decoder,
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
    )
    separate_plus = decoder(feature, occupancy_plus)
    separate_minus = decoder(feature, occupancy_minus)

    torch.testing.assert_close(
        batched_plus,
        separate_plus,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        batched_minus,
        separate_minus,
        rtol=0.0,
        atol=0.0,
    )


def test_forward_rejects_nonfinite_margin_or_continuation() -> None:
    decoder = _decoder(channels=3, stride=2)
    feature = torch.zeros(1, 3, 2, 3)
    occupancy = torch.zeros(1, 1, 4, 6, dtype=torch.bool)
    feature[0, 0, 0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match="margin, continuation, and recovery must remain",
    ):
        decoder.forward_fields(feature, occupancy)
