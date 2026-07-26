from __future__ import annotations

from dataclasses import replace

import pytest

from cure_lite.conservative_factorized_config import (
    ConservativeFactorizedDecoderConfig,
)


def test_config_has_no_search_surface_and_keeps_v4_topology() -> None:
    config = ConservativeFactorizedDecoderConfig(
        feature_channels=64,
        feature_stride=4,
    )

    assert config.phase_channels == 16
    assert config.expected_parameter_count == 4385
    assert (
        config.to_v4_topology_config().expected_parameter_count
        == config.expected_parameter_count
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("phase_aggregation_policy", "phase_logmeanexp_v0"),
        ("allocation_policy", "phase_sigmoid_v0"),
        ("mass_policy", "unconstrained_v0"),
        ("resize_policy", "allocate_after_resize_v0"),
    ],
)
def test_config_rejects_method_changes(
    name: str,
    value: str,
) -> None:
    config = ConservativeFactorizedDecoderConfig(3, 2)
    with pytest.raises(ValueError):
        replace(config, **{name: value})
