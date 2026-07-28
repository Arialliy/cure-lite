from __future__ import annotations

from types import SimpleNamespace

import pytest

import cure_lite_v23.decision as decision_module
from cure_lite.experiment.coverage_state_paet_decision import (
    COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    decide_coverage_state_paet_bounded,
)
from cure_lite.experiment.coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
)
from cure_lite_v23.decision import (
    PACRE_VC_BOUNDED_RUN_ID,
    decide_coverage_state_pacre_vc_bounded,
)
from cure_lite_v23.pacre_vc import PACRE_VC_CANDIDATE


def _empty_diagnostic() -> CoverageStateZeroLevelEvaluationResult:
    return CoverageStateZeroLevelEvaluationResult(
        config=CoverageStateZeroLevelEvaluationConfig(
            input_representation=COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
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


def test_v23_reuses_every_frozen_v21_inequality_without_model_reuse() -> None:
    diagnostic = _empty_diagnostic()
    reference = decide_coverage_state_paet_bounded(
        diagnostic,
        run_id=COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    )
    decision = decide_coverage_state_pacre_vc_bounded(
        diagnostic,
        run_id=PACRE_VC_BOUNDED_RUN_ID,
    )

    assert decision.checks == reference.checks
    assert decision.reference_decision_fingerprint == (
        reference.decision_fingerprint
    )
    payload = decision.canonical_payload()
    assert payload["candidate"] == PACRE_VC_CANDIDATE
    assert payload["threshold"] == 0.0
    assert payload["threshold_search_performed"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["reference_inequality_policy"][
        "only_numeric_inequalities_reused"
    ] is True
    assert payload["reference_inequality_policy"][
        "historical_model_reused"
    ] is False
    assert payload["formal800_status"] == (
        "BLOCKED_PENDING_SEPARATE_PREREGISTRATION"
    )
    assert payload["formal_800_authorized"] is False


def test_all_checks_pass_means_eligible_but_never_formal_authorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = _empty_diagnostic()
    reference = SimpleNamespace(
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
        decision_module,
        "decide_coverage_state_paet_bounded",
        lambda actual, *, run_id: (
            reference
            if actual is diagnostic
            and run_id == COVERAGE_STATE_PAET_BOUNDED_RUN_ID
            else pytest.fail("unexpected decision input")
        ),
    )

    decision = decide_coverage_state_pacre_vc_bounded(
        diagnostic,
        run_id=PACRE_VC_BOUNDED_RUN_ID,
    )
    assert decision.bounded_gate_passed
    assert decision.formal800_eligible
    assert decision.canonical_payload()["formal_800_authorized"] is False


def test_v23_decision_rejects_every_nonfrozen_run_id() -> None:
    with pytest.raises(PermissionError, match="run_id"):
        decide_coverage_state_pacre_vc_bounded(
            _empty_diagnostic(),
            run_id="cure_lite_pacre_v22_pmope_bounded_400_seed42_r1",
        )
