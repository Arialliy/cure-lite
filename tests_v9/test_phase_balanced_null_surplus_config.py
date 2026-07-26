from __future__ import annotations

from dataclasses import replace

import pytest

from cure_lite.phase_balanced_null_surplus_factorized_config import (
    PB_NAES_METHOD_ID,
    PhaseBalancedNullSurplusFactorizedDecoderConfig,
)


def test_config_freezes_the_method_and_v4_topology() -> None:
    config = PhaseBalancedNullSurplusFactorizedDecoderConfig(
        feature_channels=64,
        feature_stride=4,
    )

    assert config.method_id == PB_NAES_METHOD_ID
    assert config.phase_channels == 16
    assert config.expected_parameter_count == 4385
    topology = config.to_v4_topology_config()
    assert topology.feature_channels == 64
    assert topology.feature_stride == 4
    assert topology.expected_parameter_count == 4385


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("method_id", "pb_naes_v10"),
        ("phase_intensity_policy", "learned_intensity"),
        ("null_mass_policy", "one_global_null"),
        ("balance_policy", "learned_balance"),
        ("occupancy_policy", "learned_occupancy"),
        ("forward_policy", "soft_selection"),
        ("backward_policy", "positive_only"),
        ("logit_composition_policy", "extra_head"),
        ("resize_policy", "resize_before_surplus"),
    ],
)
def test_config_rejects_every_method_mutation(
    name: str,
    value: str,
) -> None:
    config = PhaseBalancedNullSurplusFactorizedDecoderConfig(3, 4)
    with pytest.raises(ValueError, match=f"fixes {name}"):
        replace(config, **{name: value})
