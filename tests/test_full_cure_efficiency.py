from __future__ import annotations

import torch

from cure_lite.cure.config import CUREResidualConfig
from cure_lite.cure.decoder import CUREResidualDecoder
from cure_lite.cure.efficiency import measure_cure_incremental_efficiency


def test_full_cure_incremental_efficiency_report_has_exact_conv_cost() -> None:
    config = CUREResidualConfig(feature_channels=3, width=8, groups=4)
    decoder = CUREResidualDecoder(config)
    feature = torch.zeros(1, 3, 3, 3)
    probability = torch.full((1, 1, 9, 9), 0.1)
    occupancy = torch.zeros_like(probability, dtype=torch.bool)
    occupancy[0, 0, 4, 4] = True

    report = measure_cure_incremental_efficiency(
        decoder,
        feature,
        probability,
        occupancy,
        warmup=0,
        repetitions=2,
    )

    assert report.parameters == sum(
        parameter.numel() for parameter in decoder.parameters()
    )
    assert report.trainable_parameters == report.parameters
    # project + two depthwise/pointwise blocks + output projection
    expected_macs = (
        3 * 8 * 3 * 3
        + (9 * 9 + 9 * 8 + 8 * 9 + 8 * 8 + 8) * 9 * 9
    )
    assert report.conv_macs == expected_macs
    assert report.conv_flops == 2 * expected_macs
    assert report.repetitions == 2
    assert report.median_incremental_latency_ms >= 0.0
    assert report.p95_incremental_latency_ms >= 0.0
    assert report.feature_shape == (1, 3, 3, 3)
    assert report.evaluation_shape == (1, 1, 9, 9)


def test_efficiency_measurement_restores_decoder_mode() -> None:
    decoder = CUREResidualDecoder(
        CUREResidualConfig(feature_channels=3, width=8, groups=4)
    ).train()
    feature = torch.zeros(1, 3, 2, 2)
    probability = torch.zeros(1, 1, 5, 5)
    occupancy = torch.zeros_like(probability, dtype=torch.bool)
    measure_cure_incremental_efficiency(
        decoder,
        feature,
        probability,
        occupancy,
        warmup=0,
        repetitions=1,
    )
    assert decoder.training
