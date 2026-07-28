from __future__ import annotations

from dataclasses import replace

import pytest

from cure_lite.experiment.coverage_state_paet_decision import (
    COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    decide_coverage_state_paet_bounded,
)
from cure_lite.experiment.coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION,
    CoverageStateNaturalZeroLevelDiagnostic,
    CoverageStatePairZeroLevelDiagnostic,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
)


def _natural(
    index: int,
    *,
    state_kind: str,
    passed: bool,
    recovered: bool | None,
    target_pixels: int = 0,
    target_negative: int = 0,
) -> CoverageStateNaturalZeroLevelDiagnostic:
    return CoverageStateNaturalZeroLevelDiagnostic(
        record_id=f"{state_kind}-{index}",
        sample_id=f"sample-{state_kind}-{index}",
        state_kind=state_kind,
        field_valid_pixels=512,
        invalid_completion_pixels=0,
        negative_pixels=target_negative,
        negative_components=1 if target_negative else 0,
        focus_target_pixels=target_pixels,
        focus_target_negative_pixels=target_negative,
        target_negative_fraction_hex=(
            (target_negative / target_pixels).hex()
            if target_pixels
            else None
        ),
        target_recovered=recovered,
        connected_support_components=1 if target_pixels else 0,
        connected_support_components_hit=(
            1 if recovered is True else 0
        ),
        connected_support_recall_hex=(
            (1.0 if recovered else 0.0).hex()
            if target_pixels
            else None
        ),
        gate_passed=passed,
    )


def _pair(
    index: int,
    *,
    pair_kind: str,
    optimizer_role: str,
    passed: bool,
    added_pixels: int = 0,
    target_negative: int = 0,
    outside: int | None = None,
    compact: bool | None = None,
) -> CoverageStatePairZeroLevelDiagnostic:
    return CoverageStatePairZeroLevelDiagnostic(
        pair_id=f"{optimizer_role}-{index}",
        sample_id=f"sample-{optimizer_role}-{index}",
        pair_kind=pair_kind,
        optimizer_role=optimizer_role,
        scalar_hidden=False,
        actual_inputs_equal=pair_kind == "identity_null",
        invalid_completion_pixels_plus=0,
        invalid_completion_pixels_minus=0,
        field_exact_equal=pair_kind == "identity_null",
        completion_exact_equal=pair_kind != "clean_positive",
        final_exact_equal=pair_kind != "clean_positive",
        maximum_abs_field_difference_hex=(0.0).hex(),
        added_target_pixels=added_pixels,
        added_target_components=1 if added_pixels else 0,
        minus_added_target_negative_pixels=target_negative,
        minus_added_target_all_negative=(
            target_negative == added_pixels if added_pixels else None
        ),
        response_sign_pixels=added_pixels,
        response_sign_correct_pixels=target_negative,
        response_sign_all_correct=(
            target_negative == added_pixels if added_pixels else None
        ),
        plus_writable_false_island_components=(
            0 if pair_kind == "clean_positive" else None
        ),
        new_negative_pixels=target_negative + (outside or 0),
        new_negative_components=1 if target_negative else 0,
        removed_footprint_negative_pixels=0,
        new_completion_pixels=target_negative + (outside or 0),
        new_completion_outside_added_target_pixels=outside,
        new_completion_components=1 if target_negative else 0,
        compact_support_exact_equal=compact,
        compact_support_component_match=compact,
        compact_support_passed=compact,
        defined_metrics_passed=passed,
        gate_passed=passed,
    )


def _passing_diagnostic() -> CoverageStateZeroLevelEvaluationResult:
    target_sizes = (20,) * 15 + (35,)
    negative_sizes = (19,) * 15 + (25,)
    assert sum(target_sizes) == 335
    assert sum(negative_sizes) == 310
    natural = tuple(
        _natural(
            index,
            state_kind="factual_miss",
            passed=index < 14,
            recovered=True,
            target_pixels=target_sizes[index],
            target_negative=negative_sizes[index],
        )
        for index in range(16)
    ) + tuple(
        _natural(
            index,
            state_kind="factual_no_miss",
            passed=True,
            recovered=None,
        )
        for index in range(16)
    )

    clean_sizes = (14,) + (9,) * 15
    remaining_negative = 124
    remaining_outside = 46
    clean = []
    for index, size in enumerate(clean_sizes):
        negative = min(size, remaining_negative)
        remaining_negative -= negative
        outside = min(3, remaining_outside)
        remaining_outside -= outside
        clean.append(
            _pair(
                index,
                pair_kind="clean_positive",
                optimizer_role="clean_positive",
                passed=index == 0,
                added_pixels=size,
                target_negative=negative,
                outside=outside,
                compact=index == 0,
            )
        )
    assert remaining_negative == remaining_outside == 0
    component = tuple(
        _pair(
            index,
            pair_kind="component_null",
            optimizer_role="component_null",
            passed=True,
        )
        for index in range(16)
    )
    identity = tuple(
        _pair(
            index,
            pair_kind="identity_null",
            optimizer_role="diagnostic_identity",
            passed=True,
        )
        for index in range(16)
    )
    diagnostic = (
        _pair(
            0,
            pair_kind="component_null",
            optimizer_role="diagnostic_only",
            passed=True,
        ),
    )
    return CoverageStateZeroLevelEvaluationResult(
        config=CoverageStateZeroLevelEvaluationConfig(
            input_representation=(
                COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
            ),
        ),
        dataset="IRSTD-1K",
        split="D_R",
        cache_fingerprint="a" * 64,
        checkpoint_fingerprint="b" * 64,
        state_ledger=(),
        natural_diagnostics=natural,
        pair_diagnostics=tuple(clean) + component + identity + diagnostic,
        diagnostic_state_references=97,
        unique_actual_input_states=97,
        model_forward_invocations=97,
        exact_replay_forward_invocations=0,
        reused_state_references=0,
        backward_calls=0,
        optimizer_steps=0,
        factual_miss_gate_passed=False,
        factual_no_miss_gate_passed=True,
        clean_defined_metrics_passed=False,
        clean_compact_support_gate_passed=False,
        component_null_gate_passed=True,
        identity_null_gate_passed=True,
        scalar_hidden_diagnostic_gate_passed=True,
        bounded_gate_passed=False,
        fail_closed_reasons=(),
    )


def _decide(
    diagnostic: CoverageStateZeroLevelEvaluationResult,
):
    return decide_coverage_state_paet_bounded(
        diagnostic,
        run_id=COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    )


def test_paet_decision_accepts_every_inclusive_boundary_together() -> None:
    decision = _decide(_passing_diagnostic())

    assert decision.bounded_gate_passed
    assert decision.formal800_eligible
    assert decision.run_id == COVERAGE_STATE_PAET_BOUNDED_RUN_ID
    assert decision.factual_target_pixels == 335
    assert decision.factual_target_negative_pixels == 310
    assert decision.factual_strict_count == 14
    assert decision.factual_recovered_count == 16
    assert decision.clean_target_pixels == 149
    assert decision.clean_target_negative_pixels == 124
    assert decision.clean_outside_completion_pixels == 46
    assert decision.clean_compact_support_passed_count == 1
    assert decision.component_null_passed_count == 16
    assert decision.factual_no_miss_passed_count == 16
    assert decision.identity_null_passed_count == 16
    assert decision.diagnostic_null_passed_count == 1
    assert decision.invalid_completion_pixels == 0
    payload = decision.canonical_payload()
    assert payload["same_sign_response_diagnostic"]["is_gate"] is False
    assert payload["formal_800_authorized"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("factual_recovered", "factual_recovered_16_of_16"),
        ("factual_strict", "factual_strict_ge_14_of_16"),
        (
            "factual_negative",
            "factual_target_negative_ge_310_of_335",
        ),
        (
            "clean_negative",
            "clean_target_negative_ge_124_of_149",
        ),
        ("clean_outside", "clean_outside_completion_le_46"),
        ("clean_compact", "clean_compact_support_ge_1_of_16"),
        ("component", "component_null_16_of_16"),
        ("no_miss", "factual_no_miss_16_of_16"),
        ("identity", "identity_null_16_of_16"),
        ("diagnostic", "diagnostic_null_1_of_1"),
        ("invalid", "invalid_completion_zero"),
    ),
)
def test_paet_decision_rejects_each_one_unit_violation(
    mutation: str,
    failed_check: str,
) -> None:
    diagnostic = _passing_diagnostic()
    natural = list(diagnostic.natural_diagnostics)
    pairs = list(diagnostic.pair_diagnostics)
    if mutation == "factual_recovered":
        natural[0] = replace(natural[0], target_recovered=False)
    elif mutation == "factual_strict":
        natural[13] = replace(natural[13], gate_passed=False)
    elif mutation == "factual_negative":
        natural[0] = replace(
            natural[0],
            focus_target_negative_pixels=(
                natural[0].focus_target_negative_pixels - 1
            ),
        )
    elif mutation == "clean_negative":
        pairs[0] = replace(
            pairs[0],
            minus_added_target_negative_pixels=(
                pairs[0].minus_added_target_negative_pixels - 1
            ),
        )
    elif mutation == "clean_outside":
        pairs[0] = replace(
            pairs[0],
            new_completion_outside_added_target_pixels=4,
        )
    elif mutation == "clean_compact":
        pairs[0] = replace(pairs[0], compact_support_passed=False)
    elif mutation == "component":
        pairs[16] = replace(pairs[16], gate_passed=False)
    elif mutation == "no_miss":
        natural[16] = replace(natural[16], gate_passed=False)
    elif mutation == "identity":
        pairs[32] = replace(pairs[32], gate_passed=False)
    elif mutation == "diagnostic":
        pairs[-1] = replace(pairs[-1], gate_passed=False)
    elif mutation == "invalid":
        natural[0] = replace(
            natural[0],
            invalid_completion_pixels=1,
        )
    else:
        raise AssertionError(mutation)
    decision = _decide(
        replace(
            diagnostic,
            natural_diagnostics=tuple(natural),
            pair_diagnostics=tuple(pairs),
        )
    )

    assert not decision.bounded_gate_passed
    assert failed_check in decision.failed_checks


def test_paet_decision_binds_population_and_zero_threshold() -> None:
    diagnostic = _passing_diagnostic()
    missing = replace(
        diagnostic,
        pair_diagnostics=diagnostic.pair_diagnostics[:-1],
    )
    decision = _decide(missing)
    assert not decision.bounded_gate_passed
    assert "fixed_D_R_population" in decision.failed_checks
    assert "diagnostic_null_1_of_1" in decision.failed_checks

    wrong_representation = replace(
        diagnostic,
        config=replace(
            diagnostic.config,
            input_representation=(
                COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION
            ),
        ),
    )
    decision = _decide(wrong_representation)
    assert not decision.bounded_gate_passed
    assert decision.failed_checks == ("threshold_zero_without_search",)


def test_paet_decision_rejects_undefined_clean_metrics() -> None:
    diagnostic = _passing_diagnostic()
    pairs = list(diagnostic.pair_diagnostics)
    pairs[0] = replace(
        pairs[0],
        new_completion_outside_added_target_pixels=None,
    )
    decision = _decide(
        replace(diagnostic, pair_diagnostics=tuple(pairs))
    )
    assert not decision.bounded_gate_passed
    assert "fixed_D_R_population" in decision.failed_checks


def test_paet_decision_rejects_wrong_run_id() -> None:
    with pytest.raises(PermissionError, match="run_id"):
        decide_coverage_state_paet_bounded(
            _passing_diagnostic(),
            run_id="cure_lite_paet_bfa_v21_wrong",
        )
