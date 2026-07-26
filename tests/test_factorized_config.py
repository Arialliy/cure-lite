from __future__ import annotations

import pytest

from cure_lite.factorized_config import (
    FactorizedDecoderConfig,
    SVEF_EVIDENCE_TRANSFORM,
    SVEF_RESIZE_POLICY,
)
from cure_lite.factorized_decoder import CURELiteFactorizedDecoder


def test_reference_config_has_frozen_topology_and_parameter_count() -> None:
    config = FactorizedDecoderConfig(
        feature_channels=64,
        feature_stride=4,
    )

    assert config.width == 32
    assert config.groups == 8
    assert config.trunk_residual_scale == 0.5
    assert config.baseline_probability == 0.1
    assert config.vacancy_kernel_size == 3
    assert config.evidence_transform == SVEF_EVIDENCE_TRANSFORM
    assert config.resize_policy == SVEF_RESIZE_POLICY
    assert config.phase_channels == 16
    assert config.expected_parameter_count == 4385

    decoder = CURELiteFactorizedDecoder(config)
    assert sum(
        parameter.numel() for parameter in decoder.parameters()
    ) == 4385


def test_adapter_fields_define_one_decoder_family() -> None:
    full_resolution = FactorizedDecoderConfig(
        feature_channels=16,
        feature_stride=1,
    )
    half_resolution = FactorizedDecoderConfig(
        feature_channels=7,
        feature_stride=2,
    )

    assert full_resolution.phase_channels == 1
    assert half_resolution.phase_channels == 4
    assert full_resolution.expected_parameter_count == 1889
    assert half_resolution.expected_parameter_count == 1793


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
def test_config_rejects_non_method_variants(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        FactorizedDecoderConfig(**kwargs)


def test_decoder_constructor_rejects_ambiguous_or_incomplete_config() -> None:
    config = FactorizedDecoderConfig(
        feature_channels=3,
        feature_stride=2,
    )
    with pytest.raises(ValueError):
        CURELiteFactorizedDecoder(
            config,
            feature_channels=3,
        )
    with pytest.raises(TypeError):
        CURELiteFactorizedDecoder(feature_channels=3)
    with pytest.raises(TypeError):
        CURELiteFactorizedDecoder(3)  # type: ignore[arg-type]

