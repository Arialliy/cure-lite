from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from cure_lite.factorized_config import FactorizedDecoderConfig
from cure_lite.factorized_decoder import CURELiteFactorizedDecoder
from cure_lite.train.paired_step import _paired_endpoint_logits


def _decoder(
    *,
    channels: int = 3,
    stride: int = 4,
    seed: int = 7301,
) -> CURELiteFactorizedDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CURELiteFactorizedDecoder(
            FactorizedDecoderConfig(
                feature_channels=channels,
                feature_stride=stride,
            )
        )


def test_reference_forward_has_exact_fields_and_native_subpixel_shape() -> None:
    decoder = _decoder(channels=64, stride=4)
    feature = torch.randn(2, 64, 64, 64)
    occupancy = torch.zeros(2, 1, 256, 256, dtype=torch.bool)
    occupancy[0, 0, 120:124, 80:84] = True

    fields = decoder.forward_fields(feature, occupancy)

    assert fields.logits.shape == (2, 1, 256, 256)
    assert fields.baseline_logits.shape == fields.logits.shape
    assert fields.evidence.shape == fields.logits.shape
    assert fields.vacancy.shape == fields.logits.shape
    assert fields.projected_occupancy.shape == (2, 1, 64, 64)
    assert fields.local_occupancy_count.shape == (2, 1, 64, 64)
    assert fields.native_subpixel_size == (256, 256)
    assert fields.output_size == (256, 256)
    assert fields.field_resize_applied is False
    assert torch.all(fields.baseline_logits < 0.0)
    assert torch.all(fields.evidence >= 0.0)
    assert float(fields.vacancy.min()) >= 0.1
    assert float(fields.vacancy.max()) <= 1.0
    assert torch.isfinite(fields.logits).all()
    torch.testing.assert_close(
        fields.logits,
        fields.baseline_logits + fields.evidence * fields.vacancy,
        rtol=0.0,
        atol=0.0,
    )


def test_zero_feature_is_exactly_invariant_to_any_occupancy() -> None:
    decoder = _decoder(channels=5, stride=2)
    feature = torch.zeros(3, 5, 4, 5)
    empty = torch.zeros(3, 1, 8, 10, dtype=torch.bool)
    occupied = torch.rand(3, 1, 8, 10) > 0.55

    empty_fields = decoder.forward_fields(feature, empty)
    occupied_fields = decoder.forward_fields(feature, occupied)

    assert torch.equal(
        empty_fields.baseline_logits,
        occupied_fields.baseline_logits,
    )
    assert torch.count_nonzero(empty_fields.evidence) == 0
    assert torch.count_nonzero(occupied_fields.evidence) == 0
    assert torch.equal(empty_fields.logits, occupied_fields.logits)
    expected = math.log(0.1 / 0.9)
    torch.testing.assert_close(
        empty_fields.logits,
        torch.full_like(empty_fields.logits, expected),
        rtol=0.0,
        atol=3.0e-7,
    )


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []),
)
def test_zero_feature_invariance_survives_parameter_updates_and_device(
    device: str,
) -> None:
    decoder = _decoder(channels=3, stride=2).to(device)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.uniform_(-1.5, 1.5)
    feature = torch.zeros(2, 3, 3, 4, device=device)
    occupancy_a = torch.zeros(
        2,
        1,
        6,
        8,
        dtype=torch.bool,
        device=device,
    )
    occupancy_b = torch.rand(2, 1, 6, 8, device=device) > 0.4

    fields_a = decoder.forward_fields(feature, occupancy_a)
    fields_b = decoder.forward_fields(feature, occupancy_b)

    assert torch.count_nonzero(fields_a.evidence) == 0
    assert torch.count_nonzero(fields_b.evidence) == 0
    assert torch.equal(fields_a.logits, fields_b.logits)


def test_inverse_count_gate_has_frozen_dense_overlap_delta() -> None:
    decoder = _decoder(channels=2, stride=1)
    plus = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    minus = plus.clone()
    minus[0, 0, 1, 1] = False

    _, plus_count, plus_vacancy = decoder.vacancy_field(
        plus,
        feature_size=(3, 3),
    )
    _, minus_count, minus_vacancy = decoder.vacancy_field(
        minus,
        feature_size=(3, 3),
    )

    assert float(plus_count[0, 0, 1, 1]) == 9.0
    assert float(minus_count[0, 0, 1, 1]) == 8.0
    observed = (
        minus_vacancy[0, 0, 1, 1]
        - plus_vacancy[0, 0, 1, 1]
    )
    torch.testing.assert_close(
        observed,
        torch.tensor(1.0 / 90.0),
        rtol=0.0,
        atol=1.0e-8,
    )


def test_deletion_is_monotone_and_delta_has_exact_gate_support() -> None:
    decoder = _decoder(channels=4, stride=4)
    feature = torch.randn(2, 4, 4, 4)
    plus = torch.rand(2, 1, 16, 16) > 0.82
    removal = (torch.rand_like(plus, dtype=torch.float32) > 0.6) & plus
    minus = plus & ~removal

    plus_fields = decoder.forward_fields(feature, plus)
    minus_fields = decoder.forward_fields(feature, minus)
    gate_delta = minus_fields.vacancy - plus_fields.vacancy
    logit_delta = minus_fields.logits - plus_fields.logits
    probability_delta = (
        torch.sigmoid(minus_fields.logits)
        - torch.sigmoid(plus_fields.logits)
    )

    assert float(gate_delta.detach().min()) >= 0.0
    assert float(logit_delta.detach().min()) >= -1.0e-7
    assert float(probability_delta.detach().min()) >= -1.0e-7
    unchanged = gate_delta == 0.0
    assert torch.count_nonzero(gate_delta) > 0
    assert torch.count_nonzero(logit_delta[unchanged]) == 0
    assert torch.count_nonzero(probability_delta[unchanged]) == 0
    torch.testing.assert_close(
        logit_delta,
        minus_fields.evidence * gate_delta,
        rtol=1.0e-4,
        atol=3.0e-7,
    )


def test_identity_endpoints_are_bitwise_equal_and_2b_matches_separate() -> None:
    decoder = _decoder(channels=4, stride=2)
    feature = torch.randn(2, 4, 3, 5)
    plus = torch.rand(2, 1, 6, 10) > 0.8
    minus = plus.clone()

    logits_plus, logits_minus = _paired_endpoint_logits(
        decoder,
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
    )
    separate_plus = decoder(feature, plus)
    separate_minus = decoder(feature, minus)

    assert torch.equal(logits_plus, logits_minus)
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


def test_pixel_shuffle_maps_every_phase_to_one_subpixel() -> None:
    decoder = _decoder(channels=2, stride=4)
    phases = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1, 1)

    observed = decoder.pixel_shuffle(phases)
    expected = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    assert torch.equal(observed, expected)


def test_non_native_output_resizes_fields_before_final_gate() -> None:
    decoder = _decoder(channels=3, stride=2)
    feature = torch.randn(1, 3, 2, 3)
    occupancy = torch.zeros(1, 1, 5, 7, dtype=torch.bool)
    occupancy[0, 0, 2, 3] = True

    fields = decoder.forward_fields(feature, occupancy)

    with torch.no_grad():
        trunk0 = F.silu(
            decoder.stem_norm(decoder.stem(feature.detach()))
        )
        trunk = trunk0 + 0.5 * decoder.pointwise(
            F.silu(
                decoder.depthwise_norm(
                    decoder.depthwise(trunk0)
                )
            )
        )
        baseline_native = decoder.pixel_shuffle(
            decoder.baseline_head(trunk)
        )
        evidence_native = decoder.pixel_shuffle(
            decoder.evidence_head(trunk)
        )
        baseline_raw = F.interpolate(
            baseline_native,
            size=(5, 7),
            mode="bilinear",
            align_corners=False,
        )
        evidence_raw = F.interpolate(
            evidence_native,
            size=(5, 7),
            mode="bilinear",
            align_corners=False,
        )
        expected_baseline = -F.softplus(
            decoder.baseline_raw.reshape(1, 1, 1, 1)
            + baseline_raw
        )
        expected_evidence = (
            F.softplus(evidence_raw.square())
            - F.softplus(evidence_raw.new_zeros(()))
        )
        expected = expected_baseline + expected_evidence * fields.vacancy

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
    torch.testing.assert_close(
        fields.logits,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_feature_is_detached_but_all_decoder_parameters_receive_gradients() -> None:
    decoder = _decoder(channels=4, stride=2)
    feature = torch.randn(2, 4, 4, 4, requires_grad=True)
    occupancy = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    occupancy[0, 0, 1:3, 1:3] = True

    loss = decoder(feature, occupancy).square().mean()
    loss.backward()

    assert feature.grad is None
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in decoder.parameters()
    )


@pytest.mark.parametrize(
    ("feature", "occupancy", "error"),
    [
        (
            torch.ones(1, 3, 2, 2, dtype=torch.int64),
            torch.zeros(1, 1, 8, 8, dtype=torch.bool),
            TypeError,
        ),
        (
            torch.ones(1, 3, 2, 2),
            torch.zeros(1, 1, 8, 8),
            TypeError,
        ),
        (
            torch.ones(1, 2, 2, 2),
            torch.zeros(1, 1, 8, 8, dtype=torch.bool),
            ValueError,
        ),
        (
            torch.ones(1, 3, 9, 9),
            torch.zeros(1, 1, 8, 8, dtype=torch.bool),
            ValueError,
        ),
    ],
)
def test_invalid_forward_inputs_are_rejected(
    feature: torch.Tensor,
    occupancy: torch.Tensor,
    error: type[Exception],
) -> None:
    decoder = _decoder(channels=3, stride=4)
    with pytest.raises(error):
        decoder(feature, occupancy)
