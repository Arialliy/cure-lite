from __future__ import annotations

from types import SimpleNamespace

import pytest

import cure_lite_v22.decision as pacre_decision
from cure_lite.experiment.coverage_state_paet_decision import (
    COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    decide_coverage_state_paet_bounded,
)
from cure_lite.experiment.coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
)
from cure_lite_v22.decision import (
    PACRE_BOUNDED_RUN_ID,
    decide_coverage_state_pacre_bounded,
)


def _empty_diagnostic() -> CoverageStateZeroLevelEvaluationResult:
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
        natural_diagnostics=(),
        pair_diagnostics=(),
        diagnostic_state_references=0,
        unique_actual_input_states=0,
        model_forward_invocations=0,
        exact_replay_forward_invocations=0,
        reused_state_references=0,
        backward_calls=0,
        optimizer_steps=0,
        factual_miss_gate_passed=False,
        factual_no_miss_gate_passed=False,
        clean_defined_metrics_passed=False,
        clean_compact_support_gate_passed=False,
        component_null_gate_passed=False,
        identity_null_gate_passed=False,
        scalar_hidden_diagnostic_gate_passed=False,
        bounded_gate_passed=False,
        fail_closed_reasons=("empty_fixture",),
    )


def test_decision_reuses_exact_frozen_inequality_checks() -> None:
    diagnostic = _empty_diagnostic()
    reference = decide_coverage_state_paet_bounded(
        diagnostic,
        run_id=COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    )
    decision = decide_coverage_state_pacre_bounded(
        diagnostic,
        run_id=PACRE_BOUNDED_RUN_ID,
    )

    assert decision.checks == reference.checks
    assert decision.reference_decision_fingerprint == (
        reference.decision_fingerprint
    )
    assert not decision.bounded_gate_passed
    assert "fixed_D_R_population" in decision.failed_checks
    payload = decision.canonical_payload()
    assert payload["candidate"] == "PACRE-v22"
    assert payload["reference_inequality_policy"][
        "only_numeric_inequalities_reused"
    ] is True
    assert payload["reference_inequality_policy"][
        "historical_model_reused"
    ] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False


def test_decision_can_pass_only_when_every_reference_check_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = _empty_diagnostic()
    fake = SimpleNamespace(
        checks=(("a", True), ("b", True)),
        decision_fingerprint="c" * 64,
        factual_miss_count=16,
        factual_target_pixels=335,
        factual_no_miss_count=16,
        clean_pair_count=16,
        clean_target_pixels=149,
        component_null_count=16,
        identity_null_count=16,
        diagnostic_null_count=1,
        factual_target_negative_pixels=310,
        factual_strict_count=14,
        factual_recovered_count=16,
        factual_no_miss_passed_count=16,
        clean_target_negative_pixels=124,
        clean_outside_completion_pixels=46,
        clean_compact_support_passed_count=1,
        component_null_passed_count=16,
        identity_null_passed_count=16,
        diagnostic_null_passed_count=1,
        invalid_completion_pixels=0,
        response_sign_pixels=149,
        response_sign_correct_pixels=124,
        response_sign_all_correct_pair_count=1,
    )
    monkeypatch.setattr(
        pacre_decision,
        "decide_coverage_state_paet_bounded",
        lambda actual, *, run_id: (
            fake
            if actual is diagnostic
            and run_id == COVERAGE_STATE_PAET_BOUNDED_RUN_ID
            else None
        ),
    )

    decision = decide_coverage_state_pacre_bounded(
        diagnostic,
        run_id=PACRE_BOUNDED_RUN_ID,
    )
    assert decision.bounded_gate_passed
    assert decision.formal800_eligible
    assert not decision.failed_checks


def test_decision_rejects_any_other_run_id() -> None:
    with pytest.raises(PermissionError, match="run_id"):
        decide_coverage_state_pacre_bounded(
            _empty_diagnostic(),
            run_id="different",
        )
