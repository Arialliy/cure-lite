from __future__ import annotations

import pytest

from cure_lite.crossing_factorized_config import (
    CROSSING_BACKWARD_SURROGATE_POLICY,
    CROSSING_FORWARD_EVIDENCE_TRANSFORM,
    CROSSING_LOGIT_COMPOSITION_POLICY,
    CROSSING_OCCUPANCY_BURDEN_POLICY,
    CROSSING_RESIZE_POLICY,
    CROSSING_ZERO_BOUNDARY_POLICY,
    CrossingFactorizedDecoderConfig,
)
from cure_lite.crossing_factorized_decoder import (
    CURELiteCrossingFactorizedDecoder,
)


def test_reference_config_preserves_v4_topology_and_parameter_count() -> None:
    config = CrossingFactorizedDecoderConfig(
        feature_channels=64,
        feature_stride=4,
    )

    assert config.width == 32
    assert config.groups == 8
    assert config.trunk_residual_scale == 0.5
    assert config.baseline_probability == 0.1
    assert config.vacancy_kernel_size == 3
    assert (
        config.occupancy_burden_policy
        == CROSSING_OCCUPANCY_BURDEN_POLICY
    )
    assert (
        config.forward_evidence_transform
        == CROSSING_FORWARD_EVIDENCE_TRANSFORM
    )
    assert (
        config.backward_surrogate_policy
        == CROSSING_BACKWARD_SURROGATE_POLICY
    )
    assert config.zero_boundary_policy == CROSSING_ZERO_BOUNDARY_POLICY
    assert (
        config.logit_composition_policy
        == CROSSING_LOGIT_COMPOSITION_POLICY
    )
    assert config.resize_policy == CROSSING_RESIZE_POLICY
    assert config.phase_channels == 16
    assert config.expected_parameter_count == 4385
    assert config.to_v4_topology_config().expected_parameter_count == 4385

    decoder = CURELiteCrossingFactorizedDecoder(config)
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 4385


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_channels": True, "feature_stride": 4},
        {"feature_channels": 0, "feature_stride": 4},
        {"feature_channels": 4, "feature_stride": False},
        {"feature_channels": 4, "feature_stride": 0},
        {"feature_channels": 4, "feature_stride": 2, "width": 16},
        {"feature_channels": 4, "feature_stride": 2, "groups": 4},
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "trunk_residual_scale": 1.0,
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "baseline_probability": 0.2,
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "vacancy_kernel_size": 5,
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "occupancy_burden_policy": "other",
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "forward_evidence_transform": "other",
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "backward_surrogate_policy": "other",
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "zero_boundary_policy": "other",
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "logit_composition_policy": "other",
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "resize_policy": "other",
        },
    ],
)
def test_config_rejects_any_equation_or_topology_variant(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CrossingFactorizedDecoderConfig(**kwargs)


def test_decoder_constructor_requires_one_truthful_v7_config() -> None:
    config = CrossingFactorizedDecoderConfig(
        feature_channels=3,
        feature_stride=2,
    )
    with pytest.raises(ValueError):
        CURELiteCrossingFactorizedDecoder(
            config,
            feature_channels=3,
        )
    with pytest.raises(TypeError):
        CURELiteCrossingFactorizedDecoder(feature_channels=3)
    with pytest.raises(TypeError):
        CURELiteCrossingFactorizedDecoder(3)  # type: ignore[arg-type]
