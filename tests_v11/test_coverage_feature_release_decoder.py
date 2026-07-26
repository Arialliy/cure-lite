from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from cure_lite.conservative_factorized_decoder import (
    coverage_conserving_phase_evidence,
)
from cure_lite.coverage_feature_release_config import (
    CoverageFeatureReleaseDecoderConfig,
)
from cure_lite.coverage_feature_release_decoder import (
    CURELiteCoverageFeatureReleaseDecoder,
    occupancy_free_conserving_phase_evidence,
)
from cure_lite.factorized_config import FactorizedDecoderConfig
from cure_lite.factorized_decoder import CURELiteFactorizedDecoder


def _decoder(
    *,
    channels: int = 3,
    stride: int = 4,
    seed: int = 11117,
) -> CURELiteCoverageFeatureReleaseDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CURELiteCoverageFeatureReleaseDecoder(
            CoverageFeatureReleaseDecoderConfig(
                feature_channels=channels,
                feature_stride=stride,
            )
        )


def test_decoder_keeps_exact_topology_state_and_parameter_budget() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2027)
        v4 = CURELiteFactorizedDecoder(
            FactorizedDecoderConfig(64, 4)
        )
        torch.manual_seed(2027)
        v11 = CURELiteCoverageFeatureReleaseDecoder(
            CoverageFeatureReleaseDecoderConfig(64, 4)
        )

    assert tuple(v4.state_dict()) == tuple(v11.state_dict())
    for name, expected in v4.state_dict().items():
        assert torch.equal(v11.state_dict()[name], expected)
    assert sum(parameter.numel() for parameter in v11.parameters()) == 4385
    assert len(tuple(v11.parameters())) == 6
    assert tuple(type(module) for module in tuple(v11.modules())[1:]) == tuple(
        type(module) for module in tuple(v4.modules())[1:]
    )

    v4.load_state_dict(v11.state_dict(), strict=True)
    v11.load_state_dict(v4.state_dict(), strict=True)


def test_hot_path_phase_operator_matches_v8_zero_burden_forward_and_gradient() -> None:
    raw_v8 = torch.tensor(
        [[[[1.5]], [[0.25]], [[-0.5]], [[-1.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    raw_v11 = raw_v8.detach().clone().requires_grad_()
    burden = torch.zeros((1, 1, 1, 1), dtype=torch.float64)

    expected = coverage_conserving_phase_evidence(raw_v8, burden)
    observed = occupancy_free_conserving_phase_evidence(raw_v11)

    for actual, reference in zip(observed, expected, strict=True):
        torch.testing.assert_close(
            actual,
            reference,
            rtol=0.0,
            atol=0.0,
        )
    expected[-1].sum().backward()
    observed[-1].sum().backward()
    torch.testing.assert_close(
        raw_v11.grad,
        raw_v8.grad,
        rtol=0.0,
        atol=0.0,
    )


def test_release_field_is_the_frozen_inverse_local_count() -> None:
    decoder = _decoder(channels=2, stride=2)
    occupancy = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    occupancy[0, 0, 2, 2] = True

    projected, count, release = decoder.native_release_field(
        occupancy,
        feature_size=(4, 4),
    )
    expected_count = F.conv2d(
        projected.to(torch.float32),
        torch.ones(1, 1, 3, 3),
        padding=1,
    )

    assert projected.dtype == torch.bool
    assert torch.equal(count, expected_count)
    assert torch.equal(release, expected_count.add(1.0).reciprocal())
    assert torch.all(release > 0.0)
    assert torch.all(release <= 1.0)


def test_forward_fields_apply_release_inside_the_shared_trunk() -> None:
    decoder = _decoder(channels=3, stride=2)
    feature = torch.randn(2, 3, 4, 5)
    occupancy = torch.zeros(2, 1, 8, 10, dtype=torch.bool)
    occupancy[0, 0, 2, 2] = True

    fields = decoder.forward_fields(feature, occupancy)

    assert fields.stem_feature.shape == (2, 32, 4, 5)
    assert fields.feature_release.shape == (2, 1, 4, 5)
    assert fields.released_stem_feature.shape == (2, 32, 4, 5)
    assert fields.trunk_feature.shape == (2, 32, 4, 5)
    assert fields.logits.shape == occupancy.shape
    assert torch.equal(
        fields.released_stem_feature,
        fields.stem_feature * fields.feature_release,
    )
    expected_residual = decoder.pointwise(
        F.silu(
            decoder.depthwise_norm(
                decoder.depthwise(fields.released_stem_feature)
            )
        )
    )
    torch.testing.assert_close(
        fields.trunk_feature,
        fields.released_stem_feature
        + decoder.config.trunk_residual_scale * expected_residual,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        fields.common_mode_phase_evidence,
        fields.budget_margin,
    )
    torch.testing.assert_close(
        fields.phase_allocation.sum(dim=1, keepdim=True),
        torch.ones_like(fields.evidence_budget),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        fields.allocated_phase_evidence.sum(dim=1, keepdim=True),
        fields.evidence_budget,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert torch.equal(
        fields.evidence,
        decoder.pixel_shuffle(fields.allocated_phase_evidence),
    )
    assert torch.equal(
        fields.logits,
        fields.baseline_logits + fields.evidence,
    )


def test_empty_occupancy_releases_the_complete_stem_feature() -> None:
    decoder = _decoder(channels=2, stride=4)
    feature = torch.randn(2, 2, 3, 5)
    occupancy = torch.zeros(2, 1, 12, 20, dtype=torch.bool)

    fields = decoder.forward_fields(feature, occupancy)

    assert torch.equal(
        fields.feature_release,
        torch.ones_like(fields.feature_release),
    )
    assert torch.equal(
        fields.released_stem_feature,
        fields.stem_feature,
    )


def test_zero_feature_cannot_create_an_occupancy_only_response() -> None:
    decoder = _decoder(channels=3, stride=4)
    feature = torch.zeros(1, 3, 3, 3)
    plus = torch.zeros(1, 1, 12, 12, dtype=torch.bool)
    plus[0, 0, 4, 4] = True
    minus = torch.zeros_like(plus)

    plus_fields = decoder.forward_fields(feature, plus)
    minus_fields = decoder.forward_fields(feature, minus)

    assert torch.equal(
        plus_fields.stem_feature,
        torch.zeros_like(plus_fields.stem_feature),
    )
    assert torch.equal(
        plus_fields.released_stem_feature,
        torch.zeros_like(plus_fields.released_stem_feature),
    )
    assert torch.equal(
        minus_fields.released_stem_feature,
        torch.zeros_like(minus_fields.released_stem_feature),
    )
    assert torch.equal(plus_fields.logits, minus_fields.logits)


def test_same_projected_count_gives_exactly_the_same_state() -> None:
    decoder = _decoder(channels=2, stride=2)
    feature = torch.randn(1, 2, 4, 4)
    first = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    second = torch.zeros_like(first)
    first[0, 0, 0, 0] = True
    second[0, 0, 1, 1] = True

    first_fields = decoder.forward_fields(feature, first)
    second_fields = decoder.forward_fields(feature, second)

    assert torch.equal(
        first_fields.projected_occupancy,
        second_fields.projected_occupancy,
    )
    assert torch.equal(
        first_fields.local_occupancy_count,
        second_fields.local_occupancy_count,
    )
    assert torch.equal(
        first_fields.feature_release,
        second_fields.feature_release,
    )
    assert torch.equal(first_fields.logits, second_fields.logits)


def test_identity_forward_is_exact_and_feature_is_detached() -> None:
    decoder = _decoder(channels=2, stride=2)
    feature = torch.randn(1, 2, 4, 4, requires_grad=True)
    occupancy = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    occupancy[0, 0, 2, 2] = True

    first = decoder(feature, occupancy)
    second = decoder(feature, occupancy.clone())
    assert torch.equal(first, second)

    first.square().mean().backward()
    assert feature.grad is None
    assert all(parameter.grad is not None for parameter in decoder.parameters())


def test_nonempty_intervention_changes_the_joint_latent_state() -> None:
    decoder = _decoder(channels=3, stride=4)
    feature = torch.randn(1, 3, 5, 5)
    plus = torch.zeros(1, 1, 20, 20, dtype=torch.bool)
    plus[0, 0, 8, 8] = True
    minus = torch.zeros_like(plus)

    plus_fields = decoder.forward_fields(feature, plus)
    minus_fields = decoder.forward_fields(feature, minus)

    assert torch.any(
        minus_fields.feature_release > plus_fields.feature_release
    )
    assert not torch.equal(
        plus_fields.released_stem_feature,
        minus_fields.released_stem_feature,
    )
    assert not torch.equal(
        plus_fields.trunk_feature,
        minus_fields.trunk_feature,
    )


def test_occupancy_is_absent_from_the_conserved_output_budget() -> None:
    decoder = _decoder(channels=3, stride=4)
    feature = torch.randn(1, 3, 5, 5)
    occupancy = torch.zeros(1, 1, 20, 20, dtype=torch.bool)
    occupancy[0, 0, 8, 8] = True
    fields = decoder.forward_fields(feature, occupancy)

    assert torch.equal(
        fields.budget_margin,
        fields.common_mode_phase_evidence,
    )
    torch.testing.assert_close(
        fields.allocated_phase_evidence.sum(dim=1, keepdim=True),
        fields.evidence_budget,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


@pytest.mark.parametrize(
    ("occupancy", "feature_size", "error"),
    [
        (torch.zeros(1, 8, 8), (4, 4), ValueError),
        (torch.zeros(1, 1, 8, 8), (4, 4), TypeError),
        (
            torch.zeros(1, 1, 8, 8, dtype=torch.bool),
            (0, 4),
            ValueError,
        ),
        (
            torch.zeros(1, 1, 0, 8, dtype=torch.bool),
            (4, 4),
            ValueError,
        ),
    ],
)
def test_release_field_rejects_invalid_inputs(
    occupancy: torch.Tensor,
    feature_size: tuple[int, int],
    error: type[Exception],
) -> None:
    decoder = _decoder(channels=2, stride=2)
    with pytest.raises(error):
        decoder.native_release_field(
            occupancy,
            feature_size=feature_size,
        )
