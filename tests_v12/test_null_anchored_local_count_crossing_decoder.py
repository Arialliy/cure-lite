from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from cure_lite.factorized_config import FactorizedDecoderConfig
from cure_lite.factorized_decoder import CURELiteFactorizedDecoder
from cure_lite.null_anchored_local_count_crossing_config import (
    NullAnchoredLocalCountCrossingDecoderConfig,
)
from cure_lite.null_anchored_local_count_crossing_decoder import (
    CURELiteNullAnchoredLocalCountCrossingDecoder,
    null_anchored_local_count_crossing,
)


def _decoder(
    *,
    channels: int = 3,
    stride: int = 4,
    seed: int = 12117,
) -> CURELiteNullAnchoredLocalCountCrossingDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CURELiteNullAnchoredLocalCountCrossingDecoder(
            NullAnchoredLocalCountCrossingDecoderConfig(
                feature_channels=channels,
                feature_stride=stride,
            )
        )


def _raw_from_phase_relative(phase_relative: torch.Tensor) -> torch.Tensor:
    """Invert q = r - sum(r)/(P + 1) for one phase vector."""

    return phase_relative + phase_relative.sum()


def test_decoder_keeps_exact_v4_topology_state_initialization_and_budget() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2027)
        v4 = CURELiteFactorizedDecoder(FactorizedDecoderConfig(64, 4))
        torch.manual_seed(2027)
        v12 = CURELiteNullAnchoredLocalCountCrossingDecoder(
            NullAnchoredLocalCountCrossingDecoderConfig(64, 4)
        )

    assert tuple(v12.state_dict()) == tuple(v4.state_dict())
    assert len(v12.state_dict()) == 7
    for name, expected in v4.state_dict().items():
        observed = v12.state_dict()[name]
        assert observed.shape == expected.shape
        assert torch.equal(observed, expected), name
    assert sum(parameter.numel() for parameter in v12.parameters()) == 4385
    assert len(tuple(v12.parameters())) == 6
    assert tuple(type(module) for module in tuple(v12.modules())[1:]) == tuple(
        type(module) for module in tuple(v4.modules())[1:]
    )

    v4.load_state_dict(v12.state_dict(), strict=True)
    v12.load_state_dict(v4.state_dict(), strict=True)


def test_one_forward_calls_each_trainable_module_once_and_pixel_shuffle_twice() -> None:
    decoder = _decoder(channels=3, stride=4)
    calls = {
        "stem": 0,
        "stem_norm": 0,
        "depthwise": 0,
        "depthwise_norm": 0,
        "pointwise": 0,
        "baseline_head": 0,
        "evidence_head": 0,
        "pixel_shuffle": 0,
    }

    def record(name: str):
        def hook(*_args: object) -> None:
            calls[name] += 1

        return hook

    handles = [
        getattr(decoder, name).register_forward_hook(record(name))
        for name in calls
    ]
    try:
        decoder(
            torch.randn(2, 3, 5, 5),
            torch.zeros(2, 1, 20, 20, dtype=torch.bool),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {
        "stem": 1,
        "stem_norm": 1,
        "depthwise": 1,
        "depthwise_norm": 1,
        "pointwise": 1,
        "baseline_head": 1,
        "evidence_head": 1,
        "pixel_shuffle": 2,
    }


def test_joint_operator_matches_the_frozen_equation_and_recovery_jacobian() -> None:
    raw_vector = torch.tensor(
        [5.0, 1.0, -2.0, 0.5],
        dtype=torch.float64,
        requires_grad=True,
    )
    count = torch.tensor([[[[1.0]]]], dtype=torch.float64)

    def evidence(vector: torch.Tensor) -> torch.Tensor:
        fields = null_anchored_local_count_crossing(
            vector.reshape(1, 4, 1, 1),
            count,
        )
        return fields[-1].reshape(4)

    (
        reference,
        phase_relative,
        count_boundary,
        margin,
        active,
        recovery,
        observed,
    ) = null_anchored_local_count_crossing(
        raw_vector.reshape(1, 4, 1, 1),
        count,
    )
    expected_reference = raw_vector.sum().reshape(1, 1, 1, 1) / 5.0
    expected_relative = raw_vector.reshape(1, 4, 1, 1) - expected_reference
    expected_margin = expected_relative - count
    expected_active = expected_margin > 0.0
    expected_recovery = torch.exp(expected_margin)
    expected_forward = torch.where(
        expected_active,
        torch.expm1(expected_margin),
        torch.zeros_like(expected_margin),
    )

    torch.testing.assert_close(reference, expected_reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        phase_relative,
        expected_relative,
        rtol=0.0,
        atol=0.0,
    )
    assert count_boundary is count
    torch.testing.assert_close(margin, expected_margin, rtol=0.0, atol=0.0)
    assert torch.equal(active, expected_active)
    torch.testing.assert_close(
        recovery,
        expected_recovery,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        observed,
        expected_forward,
        rtol=0.0,
        atol=0.0,
    )

    jacobian = torch.autograd.functional.jacobian(evidence, raw_vector)
    recovery_vector = expected_recovery.detach().reshape(4)
    expected_jacobian = torch.diag(recovery_vector) - (
        recovery_vector[:, None] / 5.0
    )
    torch.testing.assert_close(
        jacobian,
        expected_jacobian,
        rtol=1.0e-12,
        atol=1.0e-15,
    )


def test_linear_count_crossings_are_analytically_unreachable() -> None:
    required_probability_delta = 0.8
    required_logit_gap = 2.0 * math.log(19.0)

    normalized_delta = 1.0 / 9.0
    raw_count_delta = 1.0
    normalized_probability_bound = math.tanh(normalized_delta / 4.0)
    raw_count_probability_bound = math.tanh(raw_count_delta / 4.0)

    assert normalized_probability_bound == pytest.approx(0.0277706, abs=1e-7)
    assert raw_count_probability_bound == pytest.approx(0.2449187, abs=1e-7)
    assert normalized_probability_bound < required_probability_delta
    assert raw_count_probability_bound < required_probability_delta
    assert normalized_delta < required_logit_gap
    assert raw_count_delta < required_logit_gap
    assert required_logit_gap == pytest.approx(5.88887796, abs=1e-8)


def test_raw_count_exponential_has_an_explicit_dual_active_05_95_witness() -> None:
    plus_count = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    minus_count = torch.zeros_like(plus_count)
    plus_margin = 1.3
    phase_relative = plus_margin + float(plus_count.item())
    raw = torch.tensor(
        [[[[2.0 * phase_relative]]]],
        dtype=torch.float64,
    )

    plus_fields = null_anchored_local_count_crossing(raw, plus_count)
    minus_fields = null_anchored_local_count_crossing(raw, minus_count)
    plus_evidence = plus_fields[-1]
    minus_evidence = minus_fields[-1]
    baseline = -5.8
    plus_probability = torch.sigmoid(plus_evidence + baseline)
    minus_probability = torch.sigmoid(minus_evidence + baseline)

    assert bool(plus_fields[4].item())
    assert bool(minus_fields[4].item())
    assert float(plus_probability) < 0.05
    assert float(minus_probability) > 0.95
    assert float(minus_probability - plus_probability) > 0.8
    torch.testing.assert_close(
        (minus_evidence + 1.0) / (plus_evidence + 1.0),
        torch.full_like(plus_evidence, math.e),
        rtol=1.0e-12,
        atol=1.0e-15,
    )


@pytest.mark.parametrize("plus_count_value", [1.0, 2.0, 3.0])
def test_every_one_count_release_has_the_same_exact_exponential_ratio(
    plus_count_value: float,
) -> None:
    raw = torch.tensor([[[[10.0]]]], dtype=torch.float64)
    plus_count = torch.full(
        (1, 1, 1, 1),
        plus_count_value,
        dtype=torch.float64,
    )
    minus_count = plus_count - 1.0

    plus_fields = null_anchored_local_count_crossing(raw, plus_count)
    minus_fields = null_anchored_local_count_crossing(raw, minus_count)
    plus_evidence = plus_fields[-1]
    minus_evidence = minus_fields[-1]

    assert bool(plus_fields[4].item())
    assert bool(minus_fields[4].item())
    torch.testing.assert_close(
        minus_fields[3] - plus_fields[3],
        torch.ones_like(plus_fields[3]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        (minus_evidence + 1.0) / (plus_evidence + 1.0),
        torch.full_like(plus_evidence, math.e),
        rtol=1.0e-12,
        atol=1.0e-15,
    )


@pytest.mark.parametrize("active_phase_count", [0, 1, 2, 3])
def test_fixed_null_supports_zero_one_two_and_three_active_phases(
    active_phase_count: int,
) -> None:
    desired_relative = torch.tensor(
        [1.0] * active_phase_count
        + [-1.0] * (4 - active_phase_count),
        dtype=torch.float64,
    )
    raw = _raw_from_phase_relative(desired_relative).reshape(1, 4, 1, 1)
    count = torch.zeros(1, 1, 1, 1, dtype=torch.float64)

    fields = null_anchored_local_count_crossing(raw, count)

    torch.testing.assert_close(
        fields[1].reshape(4),
        desired_relative,
        rtol=0.0,
        atol=0.0,
    )
    assert int(fields[4].sum()) == active_phase_count
    assert int((fields[-1].detach() > 0.0).sum()) == active_phase_count


def test_deleting_counts_only_expands_the_active_set() -> None:
    desired_relative = torch.tensor(
        [3.5, 2.5, 1.5, 0.5],
        dtype=torch.float64,
    )
    raw = _raw_from_phase_relative(desired_relative).reshape(1, 4, 1, 1)

    active_sets = []
    for count_value in (3.0, 2.0, 1.0, 0.0):
        count = torch.full(
            (1, 1, 1, 1),
            count_value,
            dtype=torch.float64,
        )
        active_sets.append(
            null_anchored_local_count_crossing(raw, count)[4].reshape(4)
        )

    for before, after in zip(
        active_sets[:-1],
        active_sets[1:],
        strict=True,
    ):
        assert torch.all(~before | after)
    assert [int(active.sum()) for active in active_sets] == [1, 2, 3, 4]


def test_inactive_counts_zero_through_nine_keep_exact_finite_gradients() -> None:
    raw = torch.zeros(10, 1, 1, 1, dtype=torch.float64, requires_grad=True)
    count = torch.arange(10, dtype=torch.float64).reshape(10, 1, 1, 1)

    native_evidence = null_anchored_local_count_crossing(raw, count)[-1]

    assert torch.equal(
        native_evidence.detach(),
        torch.zeros_like(native_evidence),
    )
    native_evidence.sum().backward()
    expected = 0.5 * torch.exp(-count)
    torch.testing.assert_close(
        raw.grad,
        expected,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    assert torch.isfinite(raw.grad).all()
    assert torch.all(raw.grad > 0.0)


def test_zero_feature_is_exactly_null_for_every_occupancy() -> None:
    decoder = _decoder(channels=3, stride=4)
    feature = torch.zeros(2, 3, 3, 3)
    empty = torch.zeros(2, 1, 12, 12, dtype=torch.bool)
    occupied = torch.ones_like(empty)

    empty_fields = decoder.forward_fields(feature, empty)
    occupied_fields = decoder.forward_fields(feature, occupied)

    for fields in (empty_fields, occupied_fields):
        assert torch.equal(
            fields.raw_phase_evidence,
            torch.zeros_like(fields.raw_phase_evidence),
        )
        assert torch.equal(
            fields.native_phase_evidence.detach(),
            torch.zeros_like(fields.native_phase_evidence),
        )
        assert torch.equal(fields.evidence.detach(), torch.zeros_like(fields.evidence))
    assert torch.equal(empty_fields.baseline_logits, occupied_fields.baseline_logits)
    assert torch.equal(empty_fields.logits, occupied_fields.logits)


def test_same_projected_count_is_an_exact_local_state_collision() -> None:
    decoder = _decoder(channels=2, stride=2)
    feature = torch.randn(1, 2, 4, 4)
    first = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    second = torch.zeros_like(first)
    first[0, 0, 0, 0] = True
    second[0, 0, 1, 1] = True

    first_fields = decoder.forward_fields(feature, first)
    second_fields = decoder.forward_fields(feature, second)

    assert not torch.equal(first, second)
    assert torch.equal(
        first_fields.projected_occupancy,
        second_fields.projected_occupancy,
    )
    assert torch.equal(
        first_fields.local_occupancy_count,
        second_fields.local_occupancy_count,
    )
    for name in (
        "stem_feature",
        "trunk_feature",
        "baseline_logits",
        "raw_phase_evidence",
        "null_anchored_reference",
        "phase_relative_evidence",
        "crossing_margin",
        "active_phase_mask",
        "recovery_factor",
        "native_phase_evidence",
        "evidence",
        "logits",
    ):
        assert torch.equal(
            getattr(first_fields, name),
            getattr(second_fields, name),
        ), name


def test_count_intervention_has_exact_native_and_subpixel_locality() -> None:
    decoder = _decoder(channels=2, stride=2)
    plus = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    plus[0, 0, 2, 2] = True
    minus = torch.zeros_like(plus)
    plus_projected, plus_count = decoder.native_count_field(
        plus,
        feature_size=(4, 4),
    )
    minus_projected, minus_count = decoder.native_count_field(
        minus,
        feature_size=(4, 4),
    )
    del plus_projected, minus_projected

    desired_relative = torch.full((1, 4, 4, 4), 2.0, dtype=torch.float32)
    raw = desired_relative + desired_relative.sum(dim=1, keepdim=True)
    plus_evidence = null_anchored_local_count_crossing(raw, plus_count)[-1]
    minus_evidence = null_anchored_local_count_crossing(raw, minus_count)[-1]
    native_support = plus_count > minus_count
    native_delta = minus_evidence.detach() - plus_evidence.detach()

    assert torch.all(native_delta[native_support.expand_as(native_delta)] > 0.0)
    assert torch.equal(
        native_delta[~native_support.expand_as(native_delta)],
        torch.zeros_like(native_delta[~native_support.expand_as(native_delta)]),
    )

    output_delta = decoder.pixel_shuffle(native_delta)
    output_support = decoder.pixel_shuffle(
        native_support.expand_as(native_delta).to(torch.float32)
    ).to(torch.bool)
    assert torch.all(output_delta[output_support] > 0.0)
    assert torch.equal(
        output_delta[~output_support],
        torch.zeros_like(output_delta[~output_support]),
    )


def test_forward_detaches_feature_and_reaches_all_six_parameter_tensors() -> None:
    decoder = _decoder(channels=3, stride=2)
    feature = torch.randn(2, 3, 5, 5, requires_grad=True)
    occupancy = torch.zeros(2, 1, 10, 10, dtype=torch.bool)
    occupancy[:, :, 4:6, 4:6] = True

    loss = decoder(feature, occupancy).square().mean()
    loss.backward()

    parameters = tuple(decoder.named_parameters())
    assert len(parameters) == 6
    for name, parameter in parameters:
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.any(parameter.grad != 0.0), name
    assert feature.grad is None


def test_resize_occurs_after_native_pixel_shuffle() -> None:
    decoder = _decoder(channels=3, stride=4)
    feature = torch.randn(1, 3, 3, 5)
    occupancy = torch.zeros(1, 1, 13, 21, dtype=torch.bool)

    fields = decoder.forward_fields(feature, occupancy)
    expected_evidence = F.interpolate(
        decoder.pixel_shuffle(fields.native_phase_evidence),
        size=(13, 21),
        mode="bilinear",
        align_corners=False,
    )

    assert fields.native_subpixel_size == (12, 20)
    assert fields.output_size == (13, 21)
    assert fields.field_resize_applied is True
    torch.testing.assert_close(
        fields.evidence,
        expected_evidence,
        rtol=0.0,
        atol=0.0,
    )
    assert fields.logits.shape == occupancy.shape
    assert torch.equal(fields.logits, fields.baseline_logits + fields.evidence)


def test_stride_one_retains_one_live_phase_and_native_shape() -> None:
    decoder = _decoder(channels=3, stride=1)
    feature = torch.randn(2, 3, 7, 9)
    occupancy = torch.zeros(2, 1, 7, 9, dtype=torch.bool)

    fields = decoder.forward_fields(feature, occupancy)

    assert fields.raw_phase_evidence.shape == (2, 1, 7, 9)
    assert fields.native_phase_evidence.shape == (2, 1, 7, 9)
    assert fields.logits.shape == occupancy.shape
    assert fields.field_resize_applied is False


@pytest.mark.parametrize(
    ("raw", "count", "error"),
    [
        (
            torch.zeros(1, 4, 1),
            torch.zeros(1, 1, 1, 1),
            ValueError,
        ),
        (
            torch.zeros(1, 4, 2, 2),
            torch.zeros(1, 1, 1, 1),
            ValueError,
        ),
        (
            torch.zeros(1, 4, 1, 1, dtype=torch.int64),
            torch.zeros(1, 1, 1, 1),
            TypeError,
        ),
        (
            torch.zeros(1, 4, 1, 1, dtype=torch.float64),
            torch.zeros(1, 1, 1, 1, dtype=torch.float32),
            TypeError,
        ),
    ],
)
def test_joint_operator_rejects_invalid_shapes_and_types(
    raw: torch.Tensor,
    count: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        null_anchored_local_count_crossing(raw, count)

    with pytest.raises(TypeError):
        null_anchored_local_count_crossing(  # type: ignore[arg-type]
            [0.0],
            count,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_joint_operator_rejects_nonfinite_raw_values(
    invalid: float,
) -> None:
    raw = torch.zeros(1, 4, 1, 1)
    raw[0, 0, 0, 0] = invalid
    count = torch.zeros(1, 1, 1, 1)

    with pytest.raises(
        FloatingPointError,
        match="raw_phase_evidence must be finite",
    ):
        null_anchored_local_count_crossing(raw, count)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_joint_operator_rejects_nonfinite_count_values(
    invalid: float,
) -> None:
    raw = torch.zeros(1, 4, 1, 1)
    count = torch.full((1, 1, 1, 1), invalid)

    with pytest.raises(
        FloatingPointError,
        match="local_occupancy_count must be finite",
    ):
        null_anchored_local_count_crossing(raw, count)


@pytest.mark.parametrize("invalid", [-1.0, 0.5, 9.5, 10.0])
def test_joint_operator_rejects_non_count_domain_values(
    invalid: float,
) -> None:
    raw = torch.zeros(1, 4, 1, 1)
    count = torch.full((1, 1, 1, 1), invalid)

    with pytest.raises(
        ValueError,
        match=r"integers in \[0,9\]",
    ):
        null_anchored_local_count_crossing(raw, count)


def test_joint_operator_fails_before_exponential_overflow_is_consumed() -> None:
    raw = torch.full((1, 1, 1, 1), 200.0)
    count = torch.zeros(1, 1, 1, 1)

    with pytest.raises(
        FloatingPointError,
        match="evidence and recovery factor must be finite",
    ):
        null_anchored_local_count_crossing(raw, count)


def test_decoder_rejects_nonfinite_feature_and_parameter_states() -> None:
    decoder = _decoder(channels=2, stride=2)
    occupancy = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    feature = torch.zeros(1, 2, 4, 4)
    feature[0, 0, 0, 0] = float("nan")

    with pytest.raises(FloatingPointError):
        decoder(feature, occupancy)

    feature.zero_()
    with torch.no_grad():
        decoder.baseline_raw.fill_(float("inf"))
    with pytest.raises(
        FloatingPointError,
        match="forward fields must all be finite",
    ):
        decoder(feature, occupancy)


@pytest.mark.parametrize(
    ("occupancy", "feature_size", "error"),
    [
        (torch.zeros(1, 8, 8, dtype=torch.bool), (4, 4), ValueError),
        (torch.zeros(1, 1, 8, 8), (4, 4), TypeError),
        (torch.zeros(1, 1, 8, 8, dtype=torch.bool), (0, 4), ValueError),
        (torch.zeros(1, 1, 0, 8, dtype=torch.bool), (4, 4), ValueError),
    ],
)
def test_native_count_field_rejects_invalid_inputs(
    occupancy: torch.Tensor,
    feature_size: tuple[int, int],
    error: type[Exception],
) -> None:
    decoder = _decoder(channels=2, stride=2)

    with pytest.raises(error):
        decoder.native_count_field(occupancy, feature_size=feature_size)


def test_decoder_constructor_rejects_ambiguous_or_incomplete_config() -> None:
    config = NullAnchoredLocalCountCrossingDecoderConfig(3, 2)

    with pytest.raises(ValueError):
        CURELiteNullAnchoredLocalCountCrossingDecoder(
            config,
            feature_channels=3,
        )
    with pytest.raises(TypeError):
        CURELiteNullAnchoredLocalCountCrossingDecoder(feature_channels=3)
    with pytest.raises(TypeError):
        CURELiteNullAnchoredLocalCountCrossingDecoder(  # type: ignore[arg-type]
            FactorizedDecoderConfig(3, 2)
        )
