from __future__ import annotations

from dataclasses import replace

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.paired_formal_decision import (
    FORMAL_SEEDS,
    FORMAL_WAVES,
    HISTORICAL_COMPARATORS,
    PROPOSED_METHOD,
    FormalMethodEvidence,
    assess_formal_wave,
    expected_methods_for_wave,
)

_COMPARISON_PROTOCOL_FINGERPRINT = "9" * 64


def _row(
    method: str,
    seed: int,
    *,
    true_targets: int,
    recovered: int,
) -> FormalMethodEvidence:
    payload = {
        "method": method,
        "seed": seed,
        "true_targets": true_targets,
        "recovered": recovered,
    }
    return FormalMethodEvidence(
        method=method,
        seed=seed,
        total_targets=170,
        true_targets=true_targets,
        pd=true_targets / 170,
        total_anchor_misses=23,
        recovered_anchor_misses=recovered,
        retention=1.0,
        pixel_fa=1e-5,
        raw_background_fa=2e-5,
        fp_components_per_mp=10.0,
        budget_violation=False,
        comparison_protocol_fingerprint=(
            _COMPARISON_PROTOCOL_FINGERPRINT
        ),
        result_fingerprint=stable_fingerprint(payload),
    )


def _wave_rows(
    wave: str,
    *,
    proposed_true_targets: int = 157,
    proposed_recovered: int = 10,
) -> list[FormalMethodEvidence]:
    return [
        _row(
            method,
            seed,
            true_targets=(
                proposed_true_targets
                if method == PROPOSED_METHOD
                else 155
            ),
            recovered=(
                proposed_recovered if method == PROPOSED_METHOD else 8
            ),
        )
        for seed in FORMAL_SEEDS
        for method in expected_methods_for_wave(wave)
    ]


def test_wave_method_sets_are_cumulative_and_exact() -> None:
    assert expected_methods_for_wave("A") == (
        PROPOSED_METHOD,
        *HISTORICAL_COMPARATORS,
        *FORMAL_WAVES["A"],
    )
    assert expected_methods_for_wave("C") == (
        PROPOSED_METHOD,
        *HISTORICAL_COMPARATORS,
        *FORMAL_WAVES["A"],
        *FORMAL_WAVES["B"],
        *FORMAL_WAVES["C"],
    )
    with pytest.raises(ValueError, match="wave"):
        expected_methods_for_wave("D")


def test_formal_wave_pass_is_per_seed_and_requires_both_margins() -> None:
    decision = assess_formal_wave(
        _wave_rows("A"),
        wave="A",
        protocol_fingerprint="a" * 64,
        comparison_protocol_fingerprint=(
            _COMPARISON_PROTOCOL_FINGERPRINT
        ),
    )

    assert decision["status"] == "FORMAL_WAVE_PASS"
    assert decision["next_action"] == "RUN_PRE_FROZEN_WAVE_B"
    assert decision["all_seeds_pass"] is True
    assert [row["pass"] for row in decision["seed_decisions"]] == [True, True]
    assert decision["per_seed_not_mean"] is True
    assert decision["authorizes_full_cure"] is False


def test_one_seed_cannot_be_compensated_by_the_other_seed() -> None:
    rows = _wave_rows("A", proposed_true_targets=160, proposed_recovered=12)
    for index, row in enumerate(rows):
        if row.seed == 43 and row.method == PROPOSED_METHOD:
            rows[index] = replace(
                row,
                true_targets=156,
                pd=156 / 170,
                recovered_anchor_misses=9,
                result_fingerprint="b" * 64,
            )
            break

    decision = assess_formal_wave(
        rows,
        wave="A",
        protocol_fingerprint="a" * 64,
        comparison_protocol_fingerprint=(
            _COMPARISON_PROTOCOL_FINGERPRINT
        ),
    )

    assert decision["status"] == "PERFORMANCE_FAIL"
    assert decision["next_action"] == "STOP_AND_PRESERVE_EVIDENCE"
    assert [row["pass"] for row in decision["seed_decisions"]] == [True, False]


def test_constraint_failure_stops_even_when_performance_margins_pass() -> None:
    rows = _wave_rows("A", proposed_true_targets=160, proposed_recovered=12)
    for index, row in enumerate(rows):
        if row.seed == 42 and row.method == "independent_endpoint":
            rows[index] = replace(
                row,
                pixel_fa=2e-4,
                result_fingerprint="c" * 64,
            )
            break

    decision = assess_formal_wave(
        rows,
        wave="A",
        protocol_fingerprint="a" * 64,
        comparison_protocol_fingerprint=(
            _COMPARISON_PROTOCOL_FINGERPRINT
        ),
    )
    assert decision["status"] == "PERFORMANCE_FAIL"
    assert decision["seed_decisions"][0]["checks"][
        "all_methods_satisfy_constraints"
    ] is False


def test_comparator_uses_calibration_retention_but_proposed_requires_one() -> None:
    rows = _wave_rows("A", proposed_true_targets=160, proposed_recovered=12)
    for index, row in enumerate(rows):
        if row.seed == 42 and row.method == "independent_endpoint":
            rows[index] = replace(
                row,
                retention=0.99,
                result_fingerprint="f" * 64,
            )
        if row.seed == 43 and row.method == PROPOSED_METHOD:
            rows[index] = replace(
                row,
                retention=0.99,
                result_fingerprint="1" * 64,
            )

    decision = assess_formal_wave(
        rows,
        wave="A",
        protocol_fingerprint="a" * 64,
        comparison_protocol_fingerprint=(
            _COMPARISON_PROTOCOL_FINGERPRINT
        ),
    )
    assert decision["seed_decisions"][0]["pass"] is True
    assert decision["seed_decisions"][1]["pass"] is False
    assert decision["seed_decisions"][1]["constraint_checks"][
        PROPOSED_METHOD
    ]["proposed_retention_equal_1"] is False


def test_final_wave_only_authorizes_frozen_confirmation() -> None:
    decision = assess_formal_wave(
        _wave_rows("C"),
        wave="C",
        protocol_fingerprint="d" * 64,
        comparison_protocol_fingerprint=(
            _COMPARISON_PROTOCOL_FINGERPRINT
        ),
    )
    assert decision["status"] == "FORMAL_MATCHED_CONTROL_GATE_PASS"
    assert decision["next_action"] == "FROZEN_CONFIRMATION_ONLY"
    assert decision["authorizes_only_frozen_confirmation"] is True
    assert decision["authorizes_full_cure"] is False
    assert decision["authorizes_cross_backbone"] is False


def test_missing_duplicate_extra_or_inconsistent_evidence_is_rejected() -> None:
    rows = _wave_rows("A")
    with pytest.raises(ValueError, match="exactly once"):
        assess_formal_wave(
            rows[:-1],
            wave="A",
            protocol_fingerprint="a" * 64,
            comparison_protocol_fingerprint=(
                _COMPARISON_PROTOCOL_FINGERPRINT
            ),
        )
    with pytest.raises(ValueError, match="exactly once"):
        assess_formal_wave(
            [*rows, rows[0]],
            wave="A",
            protocol_fingerprint="a" * 64,
            comparison_protocol_fingerprint=(
                _COMPARISON_PROTOCOL_FINGERPRINT
            ),
        )

    inconsistent = list(rows)
    inconsistent[0] = replace(
        inconsistent[0],
        total_anchor_misses=22,
        result_fingerprint="e" * 64,
    )
    with pytest.raises(ValueError, match="populations"):
        assess_formal_wave(
            inconsistent,
            wave="A",
            protocol_fingerprint="a" * 64,
            comparison_protocol_fingerprint=(
                _COMPARISON_PROTOCOL_FINGERPRINT
            ),
        )


def test_pd_must_equal_the_exact_target_count_fraction() -> None:
    with pytest.raises(ValueError, match="exact true-target fraction"):
        replace(_row("F", 42, true_targets=150, recovered=4), pd=0.5)


def test_all_methods_and_seeds_must_share_comparison_protocol() -> None:
    rows = _wave_rows("A")
    rows[0] = replace(
        rows[0],
        comparison_protocol_fingerprint="8" * 64,
        result_fingerprint="7" * 64,
    )
    with pytest.raises(ValueError, match="common comparison"):
        assess_formal_wave(
            rows,
            wave="A",
            protocol_fingerprint="a" * 64,
            comparison_protocol_fingerprint=(
                _COMPARISON_PROTOCOL_FINGERPRINT
            ),
        )
