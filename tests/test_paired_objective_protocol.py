from __future__ import annotations

import json
from pathlib import Path

import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint


_ROOT = Path(__file__).resolve().parents[1]
_RECEIPT = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_objective_v1"
    / "proposal_receipt.json"
)
_RECEIPT_FINGERPRINT = "5a2f357911fb5f1dc1a946b3dbad429d256c390677d238b2f395fe90ce91fac8"
_RECEIPT_SHA256 = "e4f289a7d960df1c778ae88f20cc66d13e2062194b8300c0b7a257ad20b5c7b2"


def _payload() -> dict[str, object]:
    return json.loads(_RECEIPT.read_text(encoding="utf-8"))


def _paired_loss(
    q_minus: torch.Tensor,
    q_plus: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if not (
        q_minus.shape == q_plus.shape == target.shape == valid.shape
    ):
        raise ValueError("paired toy tensors must share a shape")
    positive = valid & target
    zero = valid & ~target
    if not bool(positive.any()) or not bool(zero.any()):
        raise ValueError("clean positive toy needs nonempty positive and zero domains")
    delta = q_minus - q_plus
    positive_loss = (((delta[positive] - 1.0) / 2.0) ** 2).mean()
    zero_loss = (delta[zero] ** 2).mean()
    return 0.5 * positive_loss + 0.5 * zero_loss


def test_paired_objective_receipt_is_canonical_and_frozen() -> None:
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
        "venue_context",
        "upstream_binding",
        "reviewed_source_binding",
        "frozen_semantics",
        "pair_catalog_contract",
        "objective_contract",
        "nonfactorization_contract",
        "future_schedule_contract",
        "matched_control_contract",
        "future_interface_contract",
        "static_toy_gate_contract",
        "future_decision_contract",
        "future_performance_gate",
        "candidate_review",
        "scientific_status",
        "next_action",
        "performed",
        "authorization",
        "receipt_fingerprint_scope",
    }
    assert fingerprint == _RECEIPT_FINGERPRINT
    assert stable_fingerprint(payload) == fingerprint
    assert file_sha256(_RECEIPT) == _RECEIPT_SHA256


def test_receipt_binds_core_geometry_sources_and_training_budget() -> None:
    payload = _payload()
    upstream = payload["upstream_binding"]

    for prefix in (
        "core_object_receipt",
        "geometry_catalog",
        "p0_a1",
        "eligible_view",
        "seed42_stage_config",
        "seed43_stage_config",
    ):
        path = _ROOT / upstream[f"{prefix}_path"]
        assert file_sha256(path) == upstream[f"{prefix}_file_sha256"]

    core = json.loads(
        (_ROOT / upstream["core_object_receipt_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert core["receipt_fingerprint"] == (
        upstream["core_object_receipt_fingerprint"]
    )
    geometry = json.loads(
        (_ROOT / upstream["geometry_catalog_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert geometry["receipt_fingerprint"] == (
        upstream["geometry_catalog_receipt_fingerprint"]
    )
    a1 = json.loads(
        (_ROOT / upstream["p0_a1_path"]).read_text(encoding="utf-8")
    )
    assert a1["receipt_fingerprint"] == upstream["p0_a1_receipt_fingerprint"]
    eligible = json.loads(
        (_ROOT / upstream["eligible_view_path"]).read_text(encoding="utf-8")
    )
    assert eligible["receipt_fingerprint"] == (
        upstream["eligible_view_receipt_fingerprint"]
    )
    assert eligible["eligible_catalog_fingerprint"] == (
        upstream["eligible_catalog_fingerprint"]
    )

    for key, seed in (
        ("seed42_stage_config_path", 42),
        ("seed43_stage_config_path", 43),
    ):
        training = json.loads(
            (_ROOT / upstream[key]).read_text(encoding="utf-8")
        )["training"]
        assert training["global_seed"] == seed
        assert training["epochs"] == 800
        assert training["steps_per_epoch"] == 40
        assert training["factual_miss_batch"] == 4
        assert training["factual_no_miss_batch"] == 4
        assert training["synthetic_batch"] == 4

    for relative, expected in payload["reviewed_source_binding"].items():
        assert file_sha256(_ROOT / relative) == expected


def test_objective_is_one_fixed_pre_mask_difference_mechanism() -> None:
    payload = _payload()
    objective = payload["objective_contract"]
    semantics = payload["frozen_semantics"]
    catalog = payload["pair_catalog_contract"]

    assert semantics["target_is_not_redefined_as_G_g_intersect_C_g"] is True
    assert objective["difference_domain"] == "raw_pre_hard_mask_score"
    assert objective["positive_and_zero_domain_weights"] == [0.5, 0.5]
    assert objective["positive_error_is_range_normalized_by_two"] is True
    assert objective["fixed_branch_coefficients"] == {
        "factual_miss": 1.0,
        "factual_no_miss": 1.0,
        "paired_difference": 1.0,
    }
    assert objective["legal_endpoint_absolute_loss_in_proposed_objective"] is False
    assert objective["stop_gradient_on_either_endpoint_forbidden"] is True
    assert catalog["null_pairs_enter_proposed_training_objective"] is False
    assert catalog["null_pairs_are_control_and_evaluation_only"] is True


def test_matched_controls_are_uniquely_implementable() -> None:
    controls = _payload()["matched_control_contract"]

    assert controls["independent_endpoint_target_rule"] == (
        "T_endpoint_equals_R_G_V_O_endpoint"
    )
    assert controls["independent_endpoint_valid_rule"] == (
        "M_endpoint_equals_T_endpoint_union_B_endpoint"
    )
    assert controls["independent_endpoint_pair_reduction"] == (
        "one_half_times_CURELiteLoss_plus_plus_CURELiteLoss_minus_then_"
        "macro_mean_over_pairs"
    )
    assert controls["independent_endpoint_fixed_coefficients"] == [1.0, 1.0, 1.0]
    assert controls["independent_endpoint_rule_is_frozen"] is True

    pair_matched = set(controls["pair_matched_controls"])
    baselines = set(controls["baseline_comparators_with_native_ledgers"])
    assert "factual_absolute_only" not in pair_matched
    assert "factual_absolute_only" in baselines
    assert controls["factual_absolute_only_states_per_update"] == 8
    assert controls["factual_exposure_matched_states_per_update"] == 12

    assert controls["both_occupancy_only_controls_required_before_mechanism_freeze"]
    assert controls["capacity_active_occupancy_control_preflight"] == (
        "require_C_less_than_or_equal_to_h_times_w_minus_one_and_all_channels_"
        "standardizable"
    )
    assert controls["permutation_no_perfect_matching_status"] == (
        "COMPUTATIONALLY_INCONCLUSIVE"
    )
    assert controls["permutation_silent_singleton_exclusion_forbidden"] is True
    assert controls["missing_permutation_control_forbids_mechanism_freeze"] is True


def test_balanced_loss_normalizes_strata_and_has_bounded_terms() -> None:
    q_minus = torch.tensor([0.9, 0.3, 0.3], dtype=torch.float64)
    q_plus = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)
    target = torch.tensor([True, False, False])
    valid = torch.ones(3, dtype=torch.bool)

    loss = _paired_loss(q_minus, q_plus, target, valid)
    delta = q_minus - q_plus
    expected_positive = ((delta[0] - 1.0) / 2.0) ** 2
    expected_zero = (delta[1:] ** 2).mean()

    assert torch.allclose(loss, 0.5 * expected_positive + 0.5 * expected_zero)
    assert 0.0 <= float(expected_positive) <= 1.0
    assert 0.0 <= float(expected_zero) <= 1.0

    duplicated_zero_loss = _paired_loss(
        torch.tensor([0.9, 0.3, 0.3, 0.3, 0.3], dtype=torch.float64),
        torch.tensor([0.1, 0.2, 0.4, 0.2, 0.4], dtype=torch.float64),
        torch.tensor([True, False, False, False, False]),
        torch.ones(5, dtype=torch.bool),
    )
    assert torch.allclose(loss, duplicated_zero_loss)


def test_pair_loss_has_two_attached_gradients_and_nonzero_mixed_partial() -> None:
    q_minus = torch.tensor([0.7, 0.4], dtype=torch.float64, requires_grad=True)
    q_plus = torch.tensor([0.2, 0.3], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([True, False])
    valid = torch.ones(2, dtype=torch.bool)

    loss = _paired_loss(q_minus, q_plus, target, valid)
    grad_minus, grad_plus = torch.autograd.grad(
        loss,
        (q_minus, q_plus),
        create_graph=True,
    )
    mixed = torch.autograd.grad(
        grad_minus.sum(),
        q_plus,
        retain_graph=True,
    )[0]

    assert torch.isfinite(grad_minus).all()
    assert torch.isfinite(grad_plus).all()
    assert torch.count_nonzero(grad_minus) == grad_minus.numel()
    assert torch.count_nonzero(grad_plus) == grad_plus.numel()
    assert torch.allclose(grad_minus, -grad_plus)
    assert torch.count_nonzero(mixed) == mixed.numel()
    assert torch.all(mixed < 0)


def test_pair_identity_changes_loss_with_fixed_endpoint_marginals() -> None:
    q_minus = (
        torch.tensor([0.9, 0.2], dtype=torch.float64),
        torch.tensor([0.2, 0.9], dtype=torch.float64),
    )
    q_plus = (
        torch.tensor([0.1, 0.2], dtype=torch.float64),
        torch.tensor([0.2, 0.1], dtype=torch.float64),
    )
    targets = (
        torch.tensor([True, False]),
        torch.tensor([False, True]),
    )
    valid = torch.ones(2, dtype=torch.bool)

    aligned = sum(
        _paired_loss(minus, plus, target, valid)
        for minus, plus, target in zip(
            q_minus,
            q_plus,
            targets,
            strict=True,
        )
    )
    permuted = sum(
        _paired_loss(minus, plus, target, valid)
        for minus, plus, target in zip(
            q_minus,
            reversed(q_plus),
            targets,
            strict=True,
        )
    )

    assert not torch.allclose(aligned, permuted)


def test_raw_constant_score_has_no_difference_but_post_mask_does() -> None:
    constant = 0.4
    q_plus = torch.full((4,), constant, dtype=torch.float64)
    q_minus = torch.full((4,), constant, dtype=torch.float64)
    occupancy_plus = torch.tensor([True, True, False, False])
    occupancy_minus = torch.tensor([False, False, False, False])

    raw_delta = q_minus - q_plus
    post_mask_delta = (
        q_minus * (~occupancy_minus)
        - q_plus * (~occupancy_plus)
    )

    assert torch.equal(raw_delta, torch.zeros_like(raw_delta))
    assert torch.equal(
        post_mask_delta,
        constant * (occupancy_plus & ~occupancy_minus).to(torch.float64),
    )


def test_instance_completion_increment_can_extend_beyond_removed_component() -> None:
    selected_gt = {1, 2, 3, 4}
    removed_component = {1, 2}
    occupancy_plus = set(removed_component)
    occupancy_minus = occupancy_plus - removed_component

    completion_plus: set[int] = set()
    completion_minus = selected_gt - occupancy_minus
    actual_increment = completion_minus - completion_plus
    newly_uncovered_target_pixels = selected_gt & removed_component

    assert actual_increment == selected_gt
    assert newly_uncovered_target_pixels == {1, 2}
    assert actual_increment > newly_uncovered_target_pixels


def test_protocol_is_strictly_non_authorizing_and_preserves_mainline() -> None:
    payload = _payload()
    assert payload["performed"] == {
        "new_metric_computation_performed": False,
        "new_data_access_performed": False,
        "pair_catalog_constructed": False,
        "exposure_simulation_performed": False,
        "paired_objective_implemented": False,
        "decoder_modified": False,
        "loss_modified": False,
        "training_step_modified": False,
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
        "authorizes_independent_read_only_protocol_integrity_audit": True,
        "authorizes_pair_catalog_construction": False,
        "authorizes_exposure_simulation": False,
        "authorizes_paired_objective_implementation": False,
        "authorizes_decoder_modification": False,
        "authorizes_loss_modification": False,
        "authorizes_training_step_modification": False,
        "authorizes_training": False,
        "authorizes_calibration": False,
        "authorizes_inference": False,
        "authorizes_d_v_evaluation": False,
        "authorizes_d_t_access": False,
        "authorizes_transformation_construction": False,
        "authorizes_candidate_s_construction": False,
        "authorizes_p0_d": False,
        "authorizes_full_cure": False,
        "authorizes_backbone_integration": False,
    }

    protocol = (
        _ROOT / "CURE_Lite_Paired_Objective_协议.md"
    ).read_text(encoding="utf-8")
    for marker in (
        "完成 CURE-Lite 最小核心机制",
        "设计 Full CURE",
        "跨 IRSTD backbone 与三数据集验证",
        "pairwise implementation = false",
        "training authorization = false",
    ):
        assert marker in protocol


def test_future_gate_remains_per_seed_and_requires_two_targets() -> None:
    gate = _payload()["future_performance_gate"]

    assert gate["historical_d_v_target_count"] == 170
    assert gate["historical_d_v_anchor_miss_count"] == 23
    assert gate["development_seeds"] == [42, 43]
    assert gate["per_seed_not_mean"] is True
    assert gate["minimum_additional_true_targets_over_best_matched_comparator"] == 2
    assert gate["minimum_pd_margin"] == 2 / 170
    assert gate["minimum_additional_recovered_anchor_misses"] == 2
    assert gate["minimum_anchor_recovery_rate_margin"] == 2 / 23
    assert gate["passing_authorizes_only"] == (
        "freeze_and_confirmation_not_full_cure"
    )
