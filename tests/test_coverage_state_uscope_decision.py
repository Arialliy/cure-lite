from __future__ import annotations

from dataclasses import replace

from cure_lite.experiment.coverage_state_uscope_decision import (
    decide_coverage_state_uscope_zero_level,
)
from cure_lite.experiment.coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStatePairZeroLevelDiagnostic,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
)


def _pair(*, response_all: bool) -> CoverageStatePairZeroLevelDiagnostic:
    return CoverageStatePairZeroLevelDiagnostic(
        pair_id="pair",
        sample_id="sample",
        pair_kind="clean_positive",
        optimizer_role="clean_positive",
        scalar_hidden=False,
        actual_inputs_equal=False,
        invalid_completion_pixels_plus=0,
        invalid_completion_pixels_minus=0,
        field_exact_equal=False,
        completion_exact_equal=False,
        final_exact_equal=False,
        maximum_abs_field_difference_hex=(1.0).hex(),
        added_target_pixels=1,
        added_target_components=1,
        minus_added_target_negative_pixels=1,
        minus_added_target_all_negative=True,
        response_sign_pixels=4,
        response_sign_correct_pixels=4 if response_all else 3,
        response_sign_all_correct=response_all,
        plus_writable_false_island_components=0,
        new_negative_pixels=1,
        new_negative_components=1,
        removed_footprint_negative_pixels=0,
        new_completion_pixels=1,
        new_completion_outside_added_target_pixels=0,
        new_completion_components=1,
        compact_support_exact_equal=True,
        compact_support_component_match=True,
        compact_support_passed=True,
        defined_metrics_passed=response_all,
        gate_passed=response_all,
    )


def _diagnostic() -> CoverageStateZeroLevelEvaluationResult:
    return CoverageStateZeroLevelEvaluationResult(
        config=CoverageStateZeroLevelEvaluationConfig(
            input_representation=(
                COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
            ),
        ),
        dataset="IRSTD-1K",
        split="D_R",
        cache_fingerprint="c" * 64,
        checkpoint_fingerprint="d" * 64,
        state_ledger=(),
        natural_diagnostics=(),
        pair_diagnostics=(_pair(response_all=False),),
        diagnostic_state_references=1,
        unique_actual_input_states=1,
        model_forward_invocations=1,
        exact_replay_forward_invocations=0,
        reused_state_references=0,
        backward_calls=0,
        optimizer_steps=0,
        factual_miss_gate_passed=True,
        factual_no_miss_gate_passed=True,
        clean_defined_metrics_passed=False,
        clean_compact_support_gate_passed=True,
        component_null_gate_passed=True,
        identity_null_gate_passed=True,
        scalar_hidden_diagnostic_gate_passed=True,
        bounded_gate_passed=False,
        fail_closed_reasons=("defined_metric_gate_failed:clean_positive",),
    )


def test_uscope_response_order_is_diagnostic_not_binary_gate() -> None:
    decision = decide_coverage_state_uscope_zero_level(_diagnostic())

    assert decision.bounded_gate_passed
    assert decision.zero_level_gate_passed
    assert decision.response_sign_pixels == 4
    assert decision.response_sign_correct_pixels == 3
    assert decision.response_sign_all_correct_pair_count == 0
    assert decision.canonical_payload()["same_sign_response_policy"] == (
        "legacy_all_response_ordering_is_diagnostic_not_binary_gate"
    )
    assert decision.canonical_payload()["response_diagnostic"][
        "available"
    ] is True


def test_uscope_keeps_detection_facing_gates_strict() -> None:
    diagnostic = replace(
        _diagnostic(),
        clean_compact_support_gate_passed=False,
    )

    decision = decide_coverage_state_uscope_zero_level(diagnostic)

    assert not decision.bounded_gate_passed
    assert decision.failed_checks == ("clean_compact_support",)
