from __future__ import annotations

from dataclasses import replace

from cure_lite.experiment.coverage_state_bfa_decision import (
    decide_coverage_state_bfa_bounded,
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
) -> CoverageStateNaturalZeroLevelDiagnostic:
    return CoverageStateNaturalZeroLevelDiagnostic(
        record_id=f"{state_kind}-{index}",
        sample_id=f"sample-{state_kind}-{index}",
        state_kind=state_kind,
        field_valid_pixels=64,
        invalid_completion_pixels=0,
        negative_pixels=1 if recovered else 0,
        negative_components=1 if recovered else 0,
        focus_target_pixels=1 if state_kind == "factual_miss" else 0,
        focus_target_negative_pixels=1 if recovered else 0,
        target_negative_fraction_hex=(
            (1.0 if recovered else 0.0).hex()
            if state_kind == "factual_miss"
            else None
        ),
        target_recovered=recovered,
        connected_support_components=(
            1 if state_kind == "factual_miss" else 0
        ),
        connected_support_components_hit=(
            1 if recovered else 0
        ),
        connected_support_recall_hex=(
            (1.0 if recovered else 0.0).hex()
            if state_kind == "factual_miss"
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
    natural = tuple(
        _natural(
            index,
            state_kind="factual_miss",
            passed=index < 12,
            recovered=True,
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
            passed=index < 15,
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
        component_null_gate_passed=False,
        identity_null_gate_passed=True,
        scalar_hidden_diagnostic_gate_passed=True,
        bounded_gate_passed=False,
        fail_closed_reasons=(),
    )


def test_bfa_decision_applies_all_predeclared_inequalities() -> None:
    decision = decide_coverage_state_bfa_bounded(_passing_diagnostic())

    assert decision.bounded_gate_passed
    assert decision.formal800_eligible
    assert decision.factual_strict_count == 12
    assert decision.factual_recovered_count == 16
    assert decision.clean_target_pixels == 149
    assert decision.clean_target_negative_pixels == 124
    assert decision.clean_outside_completion_pixels == 46
    assert decision.clean_compact_support_passed_count == 1
    assert decision.component_null_passed_count == 15
    assert decision.identity_null_passed_count == 16
    assert decision.diagnostic_null_passed_count == 1
    assert decision.invalid_completion_pixels == 0
    payload = decision.canonical_payload()
    assert payload["same_sign_response_diagnostic"]["is_gate"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False


def test_bfa_decision_does_not_allow_one_metric_to_offset_another() -> None:
    diagnostic = _passing_diagnostic()
    factual = list(diagnostic.natural_diagnostics)
    factual[11] = replace(factual[11], gate_passed=False)

    failed = decide_coverage_state_bfa_bounded(
        replace(diagnostic, natural_diagnostics=tuple(factual))
    )

    assert not failed.bounded_gate_passed
    assert failed.failed_checks == ("factual_strict_gt_11_of_16",)
    assert failed.factual_recovered_count == 16
    assert failed.clean_target_negative_pixels == 124


def test_bfa_decision_binds_fixed_population_and_zero_threshold() -> None:
    diagnostic = _passing_diagnostic()
    missing = replace(
        diagnostic,
        pair_diagnostics=diagnostic.pair_diagnostics[:-1],
    )
    decision = decide_coverage_state_bfa_bounded(missing)

    assert not decision.bounded_gate_passed
    assert "fixed_D_R_population" in decision.failed_checks
    assert "diagnostic_null_pass" in decision.failed_checks

    wrong_representation = replace(
        diagnostic,
        config=replace(
            diagnostic.config,
            input_representation=(
                COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION
            ),
        ),
    )
    decision = decide_coverage_state_bfa_bounded(wrong_representation)
    assert not decision.bounded_gate_passed
    assert decision.failed_checks == ("threshold_zero_without_search",)


def test_bfa_decision_rejects_undefined_clean_metrics() -> None:
    diagnostic = _passing_diagnostic()
    pairs = list(diagnostic.pair_diagnostics)
    pairs[0] = replace(
        pairs[0],
        new_completion_outside_added_target_pixels=None,
    )

    decision = decide_coverage_state_bfa_bounded(
        replace(diagnostic, pair_diagnostics=tuple(pairs))
    )

    assert not decision.bounded_gate_passed
    assert "fixed_D_R_population" in decision.failed_checks
