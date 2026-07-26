from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "continuously_recoverable_log_vacancy_evidence_crossing_v7"
)
_CLOSURE = _PROTOCOL / "toy_gate_closure_receipt.json"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_v7_toy_closure_fingerprint_and_all_file_bindings() -> None:
    closure = _load(_CLOSURE)
    unsigned = dict(closure)
    fingerprint = unsigned.pop("receipt_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)
    assert (
        fingerprint
        == "f95573edd8b842980d5b175b1aac8caf753f6c279342da8e29f54e165b1e255f"
    )

    result_binding = closure["result_binding"]
    result_path = _ROOT / result_binding["repo_path"]
    assert file_sha256(result_path) == result_binding["file_sha256"]
    result = _load(result_path)
    result_unsigned = dict(result)
    result_fingerprint = result_unsigned.pop("result_fingerprint")
    assert result_fingerprint == stable_fingerprint(result_unsigned)
    assert result_fingerprint == result_binding["result_fingerprint"]
    assert result_binding["process_replay_count"] == 3
    assert result_binding["byte_identical"] is True
    assert result_binding["byte_count_per_result"] == 36334
    assert result_binding["cli_exit_codes"] == [0, 0, 0]

    for binding in closure["protocol_bindings"].values():
        assert file_sha256(_ROOT / binding["repo_path"]) == (
            binding["file_sha256"]
        )
    for group_name in (
        "implementation_bindings",
        "shared_training_dependency_bindings",
        "verification_test_bindings",
    ):
        for repo_path, expected_sha256 in closure[group_name].items():
            assert file_sha256(_ROOT / repo_path) == expected_sha256


def test_v7_toy_closure_authorizes_code_not_real_data_execution() -> None:
    closure = _load(_CLOSURE)
    assert closure["phase_status"] == "FROZEN_TOY_GATE_PASS"
    assert closure["decision"] == "CR_LVEC_V7_TOY_GATE_PASS"

    gate = closure["gate_summary"]
    assert gate["toy_gate_pass"] is True
    assert gate["passed_case_count"] == 6
    assert gate["failed_case_count"] == 0
    assert gate["passed_family_count"] == 2
    assert gate["nonvacuous_locality_audit_pass"] is True
    assert gate["numerical_contract_audit_pass"] is True
    assert gate["all_parameter_gradients_pass"] is True
    assert gate["bounded_code_creation_authorized"] is True
    assert gate["real_D_R_bounded_authorized"] is False

    boundary = closure["boundary"]
    assert boundary["D_R_accessed_by_toy_gate"] is False
    assert boundary["D_V_accessed"] is False
    assert boundary["D_T_accessed"] is False
    assert boundary["bounded_code_creation_authorized"] is True
    assert boundary["real_D_R_bounded_authorized"] is False
    assert boundary["real_D_R_status"] == "NOT_RUN_TOY_PHASE"
    assert boundary["detection_performance_evaluated"] is False
    assert boundary["formal_800_authorized"] is False
    assert boundary["full_cure_authorized"] is False
    assert boundary["other_detector_integration_authorized"] is False
    assert boundary["automatic_method_retry_performed"] is False


def test_v7_toy_closure_records_bug_fix_and_pre_bounded_sync_gate() -> None:
    closure = _load(_CLOSURE)
    bug = closure["development_bug_record"]
    assert bug["status"] == "FIXED_BEFORE_FORMAL_RESULT_PUBLICATION"
    assert bug["failed_check"] == "forward_crossing_exact"
    assert bug["affected_case_count"] == 4
    assert bug["code_change"] == (
        "restore_parentheses_around_recovery_minus_detached_recovery"
    )
    for field in (
        "formula_changed",
        "seed_changed",
        "updates_changed",
        "learning_rate_changed",
        "loss_changed",
        "thresholds_changed",
        "formal_result_or_closure_existed_before_fix",
    ):
        assert bug[field] is False
    assert bug["regression_test_added"] is True

    issue = closure["known_pre_bounded_performance_issue"]
    assert issue["status"] == (
        "MUST_BE_MEASURED_OR_RESOLVED_BEFORE_REAL_D_R_EXECUTION"
    )
    assert issue["estimated_decoder_calls_for_32000_updates"] == 96000
    assert issue["claim_of_zero_cuda_synchronization_allowed"] is False

