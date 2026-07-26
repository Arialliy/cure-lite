from __future__ import annotations

from dataclasses import replace

import pytest

from cure_lite.conservative_factorized_config import (
    CONSERVATIVE_ALLOCATION_POLICY,
    CONSERVATIVE_MASS_POLICY,
    CONSERVATIVE_PHASE_AGGREGATION_POLICY,
    CONSERVATIVE_RESIZE_POLICY,
)
from cure_lite.coverage_feature_release_config import (
    CCFR_FEATURE_RELEASE_POLICY,
    CCFR_JOINT_TRUNK_POLICY,
    CCFR_LOGIT_COMPOSITION_POLICY,
    CCFR_OCCUPANCY_PROJECTION_POLICY,
    CCFR_OUTPUT_BUDGET_POLICY,
    CoverageFeatureReleaseDecoderConfig,
)
from cure_lite.factorized_config import FactorizedDecoderConfig


def test_config_has_only_adapter_bound_shape_and_frozen_method_constants() -> None:
    config = CoverageFeatureReleaseDecoderConfig(
        feature_channels=64,
        feature_stride=4,
    )

    assert config.width == 32
    assert config.groups == 8
    assert config.trunk_residual_scale == 0.5
    assert config.baseline_probability == 0.1
    assert config.vacancy_kernel_size == 3
    assert config.occupancy_projection_policy == (
        CCFR_OCCUPANCY_PROJECTION_POLICY
    )
    assert config.feature_release_policy == CCFR_FEATURE_RELEASE_POLICY
    assert config.joint_trunk_policy == CCFR_JOINT_TRUNK_POLICY
    assert config.logit_composition_policy == (
        CCFR_LOGIT_COMPOSITION_POLICY
    )
    assert config.phase_aggregation_policy == (
        CONSERVATIVE_PHASE_AGGREGATION_POLICY
    )
    assert config.output_budget_policy == CCFR_OUTPUT_BUDGET_POLICY
    assert config.allocation_policy == CONSERVATIVE_ALLOCATION_POLICY
    assert config.mass_policy == CONSERVATIVE_MASS_POLICY
    assert config.resize_policy == CONSERVATIVE_RESIZE_POLICY
    assert config.phase_channels == 16
    assert config.expected_parameter_count == 4385


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("width", 16),
        ("groups", 4),
        ("trunk_residual_scale", 1.0),
        ("baseline_probability", 0.2),
        ("vacancy_kernel_size", 5),
        ("occupancy_projection_policy", "changed"),
        ("feature_release_policy", "changed"),
        ("joint_trunk_policy", "changed"),
        ("phase_aggregation_policy", "changed"),
        ("output_budget_policy", "changed"),
        ("allocation_policy", "changed"),
        ("mass_policy", "changed"),
        ("logit_composition_policy", "changed"),
        ("resize_policy", "changed"),
    ],
)
def test_config_rejects_every_method_change(
    name: str,
    value: object,
) -> None:
    config = CoverageFeatureReleaseDecoderConfig(8, 4)
    with pytest.raises(ValueError):
        replace(config, **{name: value})


def test_topology_conversion_is_exact_and_has_no_extra_parameter_budget() -> None:
    config = CoverageFeatureReleaseDecoderConfig(8, 4)
    topology = config.to_v4_topology_config()

    assert type(topology) is FactorizedDecoderConfig
    assert topology == FactorizedDecoderConfig(8, 4)
    assert config.expected_parameter_count == 2593


@pytest.mark.parametrize(
    ("channels", "stride", "error"),
    [
        (True, 4, TypeError),
        (0, 4, ValueError),
        (8, True, TypeError),
        (8, 0, ValueError),
    ],
)
def test_adapter_shape_validation_is_inherited(
    channels: object,
    stride: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        CoverageFeatureReleaseDecoderConfig(
            feature_channels=channels,  # type: ignore[arg-type]
            feature_stride=stride,  # type: ignore[arg-type]
        )
