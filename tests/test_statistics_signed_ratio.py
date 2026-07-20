from __future__ import annotations

import pytest

from cure_lite.statistics import (
    PairedRatioObservation,
    PairedSignedRatioObservation,
    formal_paired_hierarchical_ratio_bootstrap,
    formal_paired_hierarchical_signed_ratio_bootstrap,
)


def test_signed_ratio_pools_zero_denominator_scene_before_dividing() -> None:
    observations = [
        PairedSignedRatioObservation(
            f"seed-{seed}",
            "no-anchor-miss",
            baseline_numerator=0,
            baseline_denominator=0,
            method_numerator=-1,
            method_denominator=0,
        )
        for seed in range(5)
    ] + [
        PairedSignedRatioObservation(
            f"seed-{seed}",
            "anchor-misses",
            baseline_numerator=0,
            baseline_denominator=2,
            method_numerator=2,
            method_denominator=2,
        )
        for seed in range(5)
    ]

    result = formal_paired_hierarchical_signed_ratio_bootstrap(
        observations, replicates=200, random_seed=17
    )

    # Per seed: method=(-1 + 2)/(0 + 2), baseline=0/2.
    assert result.estimate == pytest.approx(0.5)
    assert result.pipeline_seeds == 5
    assert result.scenes == 2


def test_signed_ratio_requires_positive_denominator_after_seed_pooling() -> None:
    observations = [
        PairedSignedRatioObservation(
            f"seed-{seed}",
            "no-anchor-miss",
            baseline_numerator=0,
            baseline_denominator=0,
            method_numerator=-1,
            method_denominator=0,
        )
        for seed in range(5)
    ]

    with pytest.raises(ValueError, match="positive aggregate"):
        formal_paired_hierarchical_signed_ratio_bootstrap(
            observations, replicates=10
        )


def test_signed_ratio_keeps_denominators_non_negative() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        PairedSignedRatioObservation("seed", "scene", -1, -1, 1, 1)


def test_ordinary_ratio_contract_remains_strict() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        PairedRatioObservation("seed", "scene", -1, 1, 0, 1)
    with pytest.raises(ValueError, match="zero baseline denominator"):
        PairedRatioObservation("seed", "scene", 1, 0, 0, 1)

    signed_rows = [
        PairedSignedRatioObservation(
            f"seed-{seed}", "scene", -1, 1, 0, 1
        )
        for seed in range(5)
    ]
    with pytest.raises(TypeError, match="PairedRatioObservation"):
        formal_paired_hierarchical_ratio_bootstrap(
            signed_rows,  # type: ignore[arg-type]
            replicates=10,
        )
