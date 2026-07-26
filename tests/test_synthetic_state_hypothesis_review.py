from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


_ROOT = Path(__file__).resolve().parents[1]
_RECEIPT = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "synthetic_state_hypothesis_review_v1"
    / "proposal_receipt.json"
)
_RECEIPT_SHA256 = (
    "25197bbc52016c9eec2b891745b6eb421797c47566a7d4e0d41261c7453888ad"
)
_RECEIPT_FINGERPRINT = (
    "4addd47f9e9bae221b0e100be105ad03fbaf865bfa86e6dd2b209d9c014a2c35"
)
_RUN_ROOT = _ROOT / "runs" / "irstd1k_stage_a_seed42"
_ATTRIBUTION_R1 = (
    _RUN_ROOT / "cure_lite_synthetic_state_failure_attribution_v1_r1"
)
_ATTRIBUTION_R2 = (
    _RUN_ROOT / "cure_lite_synthetic_state_failure_attribution_v1_r2"
)


def _payload() -> dict[str, object]:
    return json.loads(_RECEIPT.read_text(encoding="utf-8"))


def test_hypothesis_review_receipt_is_canonical_and_frozen() -> None:
    payload = _payload()
    fingerprint = payload.pop("receipt_fingerprint")

    assert set(payload) == {
        "schema_version",
        "proposal_id",
        "stage_role",
        "dataset",
        "evidence_split",
        "new_runtime_data_access",
        "base_commit",
        "evidence_binding",
        "reviewed_source_binding",
        "verified_tensor_flow",
        "candidate_hypothesis",
        "competing_explanations",
        "hard_gate_review",
        "scientific_status",
        "required_next_read_only_gates",
        "next_action",
        "performed",
        "authorization",
        "receipt_fingerprint_scope",
    }
    assert fingerprint == _RECEIPT_FINGERPRINT
    assert stable_fingerprint(payload) == fingerprint
    assert file_sha256(_RECEIPT) == _RECEIPT_SHA256


def test_hypothesis_review_binds_exact_evidence_and_source_code() -> None:
    payload = _payload()
    evidence = payload["evidence_binding"]

    config = (
        _ROOT
        / "protocols"
        / "IRSTD-1K"
        / "synthetic_state_failure_attribution_v1"
        / "config.json"
    )
    r1_complete = _ATTRIBUTION_R1 / "COMPLETE.json"
    r2_complete = _ATTRIBUTION_R2 / "COMPLETE.json"
    authority = _ATTRIBUTION_R1 / "receipts" / "authority_binding.json"
    decision = _ATTRIBUTION_R1 / "receipts" / "diagnostic_decision.json"
    inventory = (
        _ATTRIBUTION_R1 / "receipts" / "population_factor_inventory.json"
    )

    assert file_sha256(config) == evidence["attribution_config_file_sha256"]
    assert file_sha256(r1_complete) == evidence["r1_complete_file_sha256"]
    assert file_sha256(r2_complete) == evidence["r2_complete_file_sha256"]
    assert r1_complete.read_bytes() == r2_complete.read_bytes()
    assert evidence["r1_r2_byte_identical"] is True
    assert json.loads(r1_complete.read_text())["complete_fingerprint"] == (
        evidence["r1_complete_fingerprint"]
    )
    assert json.loads(r2_complete.read_text())["complete_fingerprint"] == (
        evidence["r2_complete_fingerprint"]
    )
    assert file_sha256(authority) == evidence["authority_binding_file_sha256"]
    assert json.loads(authority.read_text())["receipt_fingerprint"] == (
        evidence["authority_binding_receipt_fingerprint"]
    )
    assert file_sha256(decision) == evidence["diagnostic_decision_file_sha256"]
    assert json.loads(decision.read_text())["receipt_fingerprint"] == (
        evidence["diagnostic_decision_receipt_fingerprint"]
    )
    assert json.loads(inventory.read_text())["population_fingerprint"] == (
        evidence["population_fingerprint"]
    )

    for relative, expected in payload["reviewed_source_binding"].items():
        assert file_sha256(_ROOT / relative) == expected


def test_hypothesis_review_records_h0_without_method_overreach() -> None:
    payload = _payload()
    review = payload["hard_gate_review"]
    status = payload["scientific_status"]
    candidate = payload["candidate_hypothesis"]

    assert review["review_outcome"] == "H0"
    assert review["review_outcome_meaning"] == (
        "no_candidate_satisfies_all_five_transformation_review_gates"
    )
    assert review["eligible_to_draft_separate_transformation_protocol"] is False
    assert review["single_mechanism"]["state"] == "fail"
    assert candidate["epistemic_status"] == (
        "structural_hypothesis_not_causally_identified"
    )
    assert candidate["causal_claim_established"] is False
    assert candidate["transformation_specified"] is False
    assert candidate["performance_claim_authorized"] is False
    assert candidate["novelty_claim_authorized"] is False
    assert status["asymmetric_feature_occupancy_edit_in_code"] == "verified"
    assert (
        status["joint_state_incompatibility_as_failure_cause"]
        == "not_established"
    )
    assert status["cure_lite_mechanism"] == "not_established"
    assert status["full_cure"] == "not_started"
    assert status["paper_core"] == "not_established"


def test_hypothesis_review_authorizes_only_read_only_redefinition() -> None:
    payload = _payload()

    assert payload["evidence_split"] == "D_R"
    assert payload["new_runtime_data_access"] == []
    assert all(value is False for value in payload["performed"].values())
    assert payload["authorization"] == {
        "authorizes_read_only_core_hypothesis_redefinition": True,
        "authorizes_new_metric_computation": False,
        "authorizes_transformation_construction": False,
        "authorizes_candidate_s_construction": False,
        "authorizes_p0_d": False,
        "authorizes_training": False,
        "authorizes_calibration": False,
        "authorizes_inference": False,
        "authorizes_d_v_evaluation": False,
        "authorizes_d_t_access": False,
        "authorizes_full_cure": False,
        "authorizes_backbone_integration": False,
    }
    assert payload["next_action"] == (
        "redefine_cure_lite_core_learning_object_before_any_"
        "transformation_protocol"
    )
    assert payload["evidence_binding"]["formal_p0_b"] == "fail"
    assert payload["evidence_binding"]["formal_p0_c"] == "fail"
    assert payload["evidence_binding"]["formal_p0_d"] == "not_evaluated"
    assert payload["evidence_binding"]["p0_d_nonexecution_reason"] == (
        "blocked_by_frozen_p0_b_c_stop_rule"
    )


def test_hypothesis_review_documents_preserve_the_cure_mainline() -> None:
    next_plan = (_ROOT / "CURE_Lite_下一步方案.md").read_text(encoding="utf-8")
    results = (
        _ROOT / "CURE_Lite_全部结果与当前研究结论.md"
    ).read_text(encoding="utf-8")
    review = (_ROOT / "CURE_Lite_机制假设审查.md").read_text(encoding="utf-8")

    for text in (next_plan, results):
        assert "完成 CURE-Lite 最小核心机制" in text
        assert "设计 Full CURE" in text
        assert "跨 IRSTD backbone 与三数据集验证" in text
    for text in (next_plan, results, review):
        assert "review outcome = H0" in text
        assert "不否定 CURE 总方向" in text
    assert "已完成：CURE-Lite 核心学习对象重定义" in next_plan
    assert "paired-objective protocol       = frozen" in next_plan
    assert "Wave A decision                   = PERFORMANCE_FAIL" in next_plan
    assert "current paired version            = stopped and preserved" in next_plan
    assert (
        "next_route                        = failure attribution before any new version"
        in next_plan
    )
    assert "STOP_AND_PRESERVE_EVIDENCE" in results
