from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from cure_lite.factorized_config import (
    SVEF_RESIZE_POLICY,
    FactorizedDecoderConfig,
)
from cure_lite.null_anchored_local_count_crossing_config import (
    NLCC_BASELINE_POLICY,
    NLCC_COUNT_BOUNDARY_POLICY,
    NLCC_CROSSING_POLICY,
    NLCC_LOGIT_COMPOSITION_POLICY,
    NLCC_OCCUPANCY_PROJECTION_POLICY,
    NLCC_PHASE_REFERENCE_POLICY,
    NullAnchoredLocalCountCrossingDecoderConfig,
)


def test_config_freezes_the_complete_nlcc_method() -> None:
    config = NullAnchoredLocalCountCrossingDecoderConfig(
        feature_channels=64,
        feature_stride=4,
    )

    assert config.width == 32
    assert config.groups == 8
    assert config.trunk_residual_scale == 0.5
    assert config.baseline_probability == 0.1
    assert config.vacancy_kernel_size == 3
    assert config.occupancy_projection_policy == (
        NLCC_OCCUPANCY_PROJECTION_POLICY
    )
    assert config.phase_reference_policy == NLCC_PHASE_REFERENCE_POLICY
    assert config.count_boundary_policy == NLCC_COUNT_BOUNDARY_POLICY
    assert config.crossing_policy == NLCC_CROSSING_POLICY
    assert config.baseline_policy == NLCC_BASELINE_POLICY
    assert config.logit_composition_policy == (
        NLCC_LOGIT_COMPOSITION_POLICY
    )
    assert config.resize_policy == SVEF_RESIZE_POLICY
    assert config.phase_channels == 16
    assert config.expected_parameter_count == 4385

    with pytest.raises(FrozenInstanceError):
        config.width = 16  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("width", 16),
        ("groups", 4),
        ("trunk_residual_scale", 1.0),
        ("baseline_probability", 0.2),
        ("vacancy_kernel_size", 5),
        ("occupancy_projection_policy", "changed"),
        ("phase_reference_policy", "changed"),
        ("count_boundary_policy", "changed"),
        ("crossing_policy", "changed"),
        ("baseline_policy", "changed"),
        ("logit_composition_policy", "changed"),
        ("resize_policy", "changed"),
    ],
)
def test_config_rejects_every_method_change(
    name: str,
    value: object,
) -> None:
    config = NullAnchoredLocalCountCrossingDecoderConfig(8, 4)

    with pytest.raises(ValueError):
        replace(config, **{name: value})


def test_v4_topology_conversion_is_exact_and_adds_no_budget() -> None:
    config = NullAnchoredLocalCountCrossingDecoderConfig(8, 4)
    topology = config.to_v4_topology_config()

    assert type(topology) is FactorizedDecoderConfig
    assert topology == FactorizedDecoderConfig(8, 4)
    assert config.phase_channels == topology.phase_channels == 16
    assert config.expected_parameter_count == 2593
    assert config.expected_parameter_count == topology.expected_parameter_count


@pytest.mark.parametrize(
    ("channels", "stride", "error"),
    [
        (True, 4, TypeError),
        (0, 4, ValueError),
        (8, True, TypeError),
        (8, 0, ValueError),
    ],
)
def test_adapter_shape_validation_is_inherited_from_v4(
    channels: object,
    stride: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        NullAnchoredLocalCountCrossingDecoderConfig(
            feature_channels=channels,  # type: ignore[arg-type]
            feature_stride=stride,  # type: ignore[arg-type]
        )
