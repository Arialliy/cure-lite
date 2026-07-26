from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


_ROOT = Path(__file__).resolve().parents[1]
_RECEIPT = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "core_learning_object_redefinition_v1"
    / "proposal_receipt.json"
)
_RECEIPT_FINGERPRINT = (
    "62461b5514b45d4082ea4001c4e8324b2f5ad0542a4ae11891b3db4fda980ef9"
)
_RECEIPT_SHA256 = (
    "18d99a53dd6b364210449acd03ac3a8bc2608a97be6f841818fb63178a2911bd"
)


def _payload() -> dict[str, object]:
    return json.loads(_RECEIPT.read_text(encoding="utf-8"))


def test_core_object_receipt_is_canonical_and_frozen() -> None:
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
        "problem_redefinition",
        "core_learning_object",
        "why_this_is_one_mechanism",
        "non_reduction_contract",
        "known_nonidentifiabilities_and_shortcuts",
        "candidate_review",
        "required_next_protocol_freezes",
        "predeclared_falsification_conditions",
        "future_stage_entry_gate",
        "scientific_status",
        "next_action",
        "performed",
        "authorization",
        "receipt_fingerprint_scope",
    }
    assert fingerprint == _RECEIPT_FINGERPRINT
    assert stable_fingerprint(payload) == fingerprint
    assert file_sha256(_RECEIPT) == _RECEIPT_SHA256


def test_core_object_receipt_binds_prior_evidence_and_current_source() -> None:
    payload = _payload()
    evidence = payload["evidence_binding"]

    prior = (
        _ROOT
        / "protocols"
        / "IRSTD-1K"
        / "synthetic_state_hypothesis_review_v1"
        / "proposal_receipt.json"
    )
    p0_decision = (
        _ROOT
        / "runs"
        / "irstd1k_stage_a_seed42"
        / "cure_lite_geometry_safe_p0_bc_v1_r1"
        / "receipts"
        / "decision.json"
    )
    attribution_decision = (
        _ROOT
        / "runs"
        / "irstd1k_stage_a_seed42"
        / "cure_lite_synthetic_state_failure_attribution_v1_r1"
        / "receipts"
        / "diagnostic_decision.json"
    )

    assert file_sha256(prior) == evidence["prior_h0_receipt_file_sha256"]
    assert json.loads(prior.read_text())["receipt_fingerprint"] == (
        evidence["prior_h0_receipt_fingerprint"]
    )
    assert file_sha256(p0_decision) == (
        evidence["geometry_safe_p0_bc_decision_file_sha256"]
    )
    assert json.loads(p0_decision.read_text())["receipt_fingerprint"] == (
        evidence["geometry_safe_p0_bc_decision_receipt_fingerprint"]
    )
    assert file_sha256(attribution_decision) == (
        evidence["failure_attribution_decision_file_sha256"]
    )
    assert json.loads(attribution_decision.read_text())["receipt_fingerprint"] == (
        evidence["failure_attribution_decision_receipt_fingerprint"]
    )
    count_results = _ROOT / evidence["historical_d_v_count_results_path"]
    assert file_sha256(count_results) == (
        evidence["historical_d_v_count_results_file_sha256"]
    )
    count_payload = json.loads(count_results.read_text(encoding="utf-8"))
    assert count_payload["results_fingerprint"] == (
        evidence["historical_d_v_count_results_fingerprint"]
    )
    count_diagnostics = count_payload["recovery_diagnostics"]["M"]
    assert count_diagnostics["total_anchor_covered"] == 147
    assert count_diagnostics["total_anchor_misses"] == 23

    budget_config = _ROOT / evidence["frozen_stage_a_budget_config_path"]
    assert file_sha256(budget_config) == (
        evidence["frozen_stage_a_budget_config_file_sha256"]
    )
    budget = json.loads(budget_config.read_text(encoding="utf-8"))["budget"]
    assert budget["pixel_fa_budget"] == 0.0001
    assert budget["raw_background_fa_budget"] == 0.0001
    assert budget["component_fa_per_mp_budget"] == 100.0
    assert budget["minimum_retention"] == 0.99

    for relative, expected in payload["reviewed_source_binding"].items():
        assert file_sha256(_ROOT / relative) == expected


def test_core_object_is_a_pre_mask_same_source_pair_not_a_surrogate_match() -> None:
    payload = _payload()
    problem = payload["problem_redefinition"]
    core = payload["core_learning_object"]
    contract = payload["non_reduction_contract"]

    assert problem["new_learning_unit"] == "same_source_coverage_before_after_pair"
    assert problem["does_not_claim_deleted_endpoint_is_a_factual_miss"] is True
    assert problem["does_not_require_factual_legal_exchangeability"] is True
    assert core["candidate_id"] == "same_source_discrete_coverage_response"
    assert core["operator_output"] == (
        "sigmoid_decoder_output_before_hard_mask_Q"
    )
    assert core["operator_output_range"] == "[0,1]"
    assert core["actual_label_increment"] == (
        "D_g=R_G_V(O_minus)_setminus_R_G_V(O_plus)"
    )
    assert core["exact_clean_pair_identity"] == (
        "D_g=A_g_only_when_the_explicit_preexisting_unmatched_gt_"
        "noninterference_requirement_holds"
    )
    assert (
        "deleted_component_has_zero_valid_domain_intersection_with_every_"
        "preexisting_unmatched_gt_instance"
        in core["legal_pair_requirements"]
    )
    assert core["positive_pair_eligibility_invariant"] == "D_g_equals_A_g"
    assert core["null_pair_label_rule"] == (
        "a_null_pair_is_admissible_only_when_its_actual_valid_domain_"
        "label_increment_D_g_is_empty"
    )
    assert core["pairwise_gradient_requirement"] == (
        "both_pre_mask_endpoints_share_parameters_remain_attached_and_"
        "enter_one_nonseparable_objective"
    )
    assert core["factual_legal_pairwise_claim"] is False
    assert core["feature_occupancy_interaction_identified"] is False
    assert contract["hard_mask_before_difference_forbidden"] is True
    assert contract["stop_gradient_on_either_endpoint_forbidden"] is True
    assert contract["independent_endpoint_losses_alone_are_not_pairwise"] is True


def test_core_object_records_shortcuts_and_strict_non_authorization() -> None:
    payload = _payload()
    shortcut_ids = {
        item["id"] for item in payload["known_nonidentifiabilities_and_shortcuts"]
    }

    assert shortcut_ids == {
        "additive_feature_offset",
        "hard_mask_mechanical_response",
        "occupancy_only_hole_copy",
        "detected_to_factual_transport",
        "nonlocal_decoder_response",
        "cross_target_uncovering",
    }
    assert payload["candidate_review"]["implementation_readiness"] == (
        "fail_not_yet_specified"
    )
    assert payload["candidate_review"]["empirical_mechanism_support"] == (
        "not_evaluated"
    )
    assert payload["performed"] == {
        "new_metric_computation_performed": False,
        "new_data_access_performed": False,
        "pairwise_objective_implemented": False,
        "decoder_modified": False,
        "loss_modified": False,
        "transformation_constructed": False,
        "candidate_s_constructed": False,
        "p0_d_executed": False,
        "training_performed": False,
        "calibration_performed": False,
        "inference_performed": False,
        "d_v_accessed": False,
        "d_t_accessed": False,
        "full_cure_started": False,
        "backbone_integration_performed": False,
    }
    assert payload["authorization"] == {
        "authorizes_separate_read_only_paired_objective_protocol_draft": True,
        "authorizes_new_metric_computation": False,
        "authorizes_pairwise_objective_implementation": False,
        "authorizes_decoder_modification": False,
        "authorizes_loss_modification": False,
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


def test_match_stability_alone_does_not_make_a_clean_label_increment() -> None:
    """A removed component may expose pixels of an existing missed target."""

    selected_gt = {1, 2, 3}
    preexisting_unmatched_gt = {9, 10}
    removed_component = {1, 2, 3, 9}
    occupancy_plus = set(removed_component)
    occupancy_minus = occupancy_plus - removed_component

    completion_plus = preexisting_unmatched_gt - occupancy_plus
    completion_minus = (
        (selected_gt - occupancy_minus)
        | (preexisting_unmatched_gt - occupancy_minus)
    )
    actual_increment = completion_minus - completion_plus
    selected_increment = selected_gt - occupancy_minus

    assert actual_increment == {1, 2, 3, 9}
    assert selected_increment == {1, 2, 3}
    assert actual_increment != selected_increment
    assert removed_component & preexisting_unmatched_gt == {9}


def test_full_cure_entry_gate_is_per_seed_and_requires_two_targets() -> None:
    payload = _payload()
    gate = payload["future_stage_entry_gate"]
    evidence = payload["evidence_binding"]
    count_payload = json.loads(
        (_ROOT / evidence["historical_d_v_count_results_path"]).read_text(
            encoding="utf-8"
        )
    )
    diagnostics = count_payload["recovery_diagnostics"]["M"]

    assert gate["historical_d_v_target_count"] == 170
    assert gate["historical_d_v_anchor_miss_count"] == 23
    assert gate["historical_d_v_target_count"] == (
        diagnostics["total_anchor_covered"]
        + diagnostics["total_anchor_misses"]
    )
    assert gate["historical_d_v_anchor_miss_count"] == (
        diagnostics["total_anchor_misses"]
    )
    assert gate["development_seeds"] == [42, 43]
    assert gate["per_seed_gate_not_mean_gate"] is True
    assert (
        gate[
            "minimum_additional_true_targets_over_the_best_matched_"
            "comparator_per_seed"
        ]
        == 2
    )
    assert gate["equivalent_minimum_pd_margin_on_the_current_d_v"] == (
        2 / 170
    )
    assert (
        gate[
            "minimum_additional_recovered_anchor_misses_over_the_best_"
            "matched_comparator_per_seed"
        ]
        == 2
    )
    assert gate["equivalent_minimum_anchor_recovery_rate_margin"] == 2 / 23
    assert gate["constraints"] == {
        "retention": 1.0,
        "pixel_fa_max": 0.0001,
        "raw_background_fa_max": 0.0001,
        "fp_components_per_mp_max": 100.0,
        "budget_violation": False,
    }
    assert gate["historical_stage_a_minimum_retention"] == 0.99
    assert (
        gate[
            "retention_one_is_a_stricter_new_stage_gate_not_the_"
            "historical_budget"
        ]
        is True
    )
    assert gate["development_pass_authorizes_only"] == (
        "freeze_and_confirmation_not_full_cure"
    )
    assert gate["one_seed_or_mean_only_improvement_is_insufficient"] is True
    assert gate["one_additional_target_per_seed_is_insufficient"] is True


def test_core_object_documents_preserve_stage_and_mainline_boundaries() -> None:
    next_plan = (_ROOT / "CURE_Lite_下一步方案.md").read_text(encoding="utf-8")
    results = (
        _ROOT / "CURE_Lite_全部结果与当前研究结论.md"
    ).read_text(encoding="utf-8")
    h0_review = (_ROOT / "CURE_Lite_机制假设审查.md").read_text(
        encoding="utf-8"
    )
    core_review = (_ROOT / "CURE_Lite_核心学习对象重定义.md").read_text(
        encoding="utf-8"
    )

    for text in (next_plan, results, core_review):
        assert "完成 CURE-Lite 最小核心机制" in text
        assert "设计 Full CURE" in text
        assert "跨 IRSTD backbone 与三数据集验证" in text
    for text in (next_plan, results, h0_review, core_review):
        assert "Delta" in text or "\\Delta" in text
    assert "pairwise implementation = false" in next_plan
    assert "training authorization = false" in core_review
    assert "未授权实现或训练" in results
    assert "H0 保持不变" in results
