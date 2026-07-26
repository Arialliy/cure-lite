from __future__ import annotations

import pytest

from cure_lite.directed_factorized_config import (
    DIRECTED_EVIDENCE_TRANSFORM,
    DirectedFactorizedDecoderConfig,
)
from cure_lite.directed_factorized_decoder import (
    CURELiteDirectedFactorizedDecoder,
)
from cure_lite.factorized_config import SVEF_RESIZE_POLICY


def test_reference_config_preserves_v4_topology_and_parameter_count() -> None:
    config = DirectedFactorizedDecoderConfig(
        feature_channels=64,
        feature_stride=4,
    )

    assert config.width == 32
    assert config.groups == 8
    assert config.trunk_residual_scale == 0.5
    assert config.baseline_probability == 0.1
    assert config.vacancy_kernel_size == 3
    assert config.evidence_transform == DIRECTED_EVIDENCE_TRANSFORM
    assert config.resize_policy == SVEF_RESIZE_POLICY
    assert config.phase_channels == 16
    assert config.expected_parameter_count == 4385
    assert config.to_v4_topology_config().expected_parameter_count == 4385

    decoder = CURELiteDirectedFactorizedDecoder(config)
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
            "evidence_transform": "other",
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "resize_policy": "other",
        },
    ],
)
def test_config_rejects_any_non_activation_or_topology_variant(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        DirectedFactorizedDecoderConfig(**kwargs)


def test_decoder_constructor_requires_one_truthful_v5_config() -> None:
    config = DirectedFactorizedDecoderConfig(
        feature_channels=3,
        feature_stride=2,
    )
    with pytest.raises(ValueError):
        CURELiteDirectedFactorizedDecoder(
            config,
            feature_channels=3,
        )
    with pytest.raises(TypeError):
        CURELiteDirectedFactorizedDecoder(feature_channels=3)
    with pytest.raises(TypeError):
        CURELiteDirectedFactorizedDecoder(3)  # type: ignore[arg-type]
