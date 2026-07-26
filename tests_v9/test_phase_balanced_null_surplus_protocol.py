from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


ROOT = Path(__file__).resolve().parents[1]
V9 = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "phase_balanced_null_anchored_evidence_surplus_v9"
)
V8 = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conserving_subpixel_evidence_allocation_v8"
)
DESIGN = ROOT / "CURE_Lite_PBNAES_v9_模型与代码设计.md"
PROPOSAL = V9 / "proposal_receipt.json"
TOY_CONFIG = V9 / "toy_config.json"
V8_NEGATIVE_CLOSURE = (
    V8 / "verifier_correction_r2" / "bounded_negative_closure_receipt.json"
)
V8_TOY_CONFIG = V8 / "toy_config.json"

DESIGN_SHA256 = (
    "89f8340ffae4b9150b15d432b6833eda64ff66ccdebe1f4ef095db97d2763f8c"
)
PROPOSAL_SHA256 = (
    "7bb563907f8b037ca2feac5ae356c5b3e3ed52cf945dfdaa2d7fa503131e02c5"
)
PROPOSAL_FINGERPRINT = (
    "90dd2931d2e8aa5ed76e86165b8d68363d7735e522b0a7cca228cc04269de5e6"
)
TOY_CONFIG_SHA256 = (
    "1089829e88d54b314bb94ba1f536130c37baa5ed2d75a0a8eff7f87d6336fddc"
)
TOY_CONFIG_FINGERPRINT = (
    "8af91f01925c258d1be9d9ec590d7aedabbc17d7ad3028e7f581731fa3925406"
)
V8_NEGATIVE_CLOSURE_SHA256 = (
    "c3264fc93ff6329184e72f8ce4f2b37d26eba98968cea0b6db000740c1a36aa1"
)
V8_NEGATIVE_CLOSURE_FINGERPRINT = (
    "99df195fe4906b71f2dd63444f84b6bcd84820b3e40d020828a5ed02d04925f9"
)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _verify_internal_fingerprint(
    payload: dict[str, object],
    *,
    field: str,
    expected: str,
) -> None:
    assert payload[field] == expected
    unsigned = dict(payload)
    del unsigned[field]
    assert stable_fingerprint(unsigned) == expected


def test_v9_design_proposal_and_v8_negative_closure_are_exactly_bound() -> None:
    proposal = _load(PROPOSAL)
    v8_closure = _load(V8_NEGATIVE_CLOSURE)

    assert file_sha256(DESIGN) == DESIGN_SHA256
    assert file_sha256(PROPOSAL) == PROPOSAL_SHA256
    assert file_sha256(V8_NEGATIVE_CLOSURE) == V8_NEGATIVE_CLOSURE_SHA256
    _verify_internal_fingerprint(
        proposal,
        field="proposal_fingerprint",
        expected=PROPOSAL_FINGERPRINT,
    )
    _verify_internal_fingerprint(
        v8_closure,
        field="receipt_fingerprint",
        expected=V8_NEGATIVE_CLOSURE_FINGERPRINT,
    )

    assert proposal["schema_version"] == (
        "cure-lite-pb-naes-v9-proposal-v1"
    )
    assert proposal["method_id"] == "pb_naes_v9"
    assert proposal["method_name"] == (
        "Phase-Balanced Null-Anchored Evidence Surplus"
    )
    assert proposal["status"] == (
        "FROZEN_BEFORE_DATASET_FREE_IMPLEMENTATION_GATE"
    )
    assert proposal["design_document"] == {
        "repo_path": "CURE_Lite_PBNAES_v9_模型与代码设计.md",
        "file_sha256": DESIGN_SHA256,
    }
    assert proposal["predecessor_v8"] == {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "coverage_conserving_subpixel_evidence_allocation_v8/"
            "verifier_correction_r2/bounded_negative_closure_receipt.json"
        ),
        "file_sha256": V8_NEGATIVE_CLOSURE_SHA256,
        "receipt_fingerprint": V8_NEGATIVE_CLOSURE_FINGERPRINT,
        "decision": "CC_SEA_V8_R2_BOUNDED_NEGATIVE_CLOSED",
        "frozen_interpretation": (
            "the fixed shared budget improved restraint but diluted clean "
            "response and forced every finite softmax phase to consume "
            "positive mass"
        ),
    }
    assert v8_closure["method_id"] == "cc_sea_v8"
    assert v8_closure["decision"] == "CC_SEA_V8_R2_BOUNDED_NEGATIVE_CLOSED"
    assert v8_closure["phase_status"] == (
        "FROZEN_R2_BOUNDED_MODEL_CODE_NONPASS"
    )


def test_v9_toy_config_is_canonical_and_bound_to_the_proposal() -> None:
    proposal = _load(PROPOSAL)
    config = _load(TOY_CONFIG)

    assert file_sha256(TOY_CONFIG) == TOY_CONFIG_SHA256
    _verify_internal_fingerprint(
        config,
        field="config_fingerprint",
        expected=TOY_CONFIG_FINGERPRINT,
    )
    assert config["schema_version"] == (
        "cure-lite-pb-naes-v9-toy-config-v1"
    )
    assert config["method_id"] == proposal["method_id"] == "pb_naes_v9"
    assert config["proposal_binding"] == {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "phase_balanced_null_anchored_evidence_surplus_v9/"
            "proposal_receipt.json"
        ),
        "file_sha256": PROPOSAL_SHA256,
        "proposal_fingerprint": PROPOSAL_FINGERPRINT,
    }
    assert config["operator"] == proposal["single_mechanism"]


def test_v9_toy_catalog_contains_the_six_frozen_cases() -> None:
    config = _load(TOY_CONFIG)

    assert config["cases"] == [
        {
            "family_id": "component_contains_response",
            "case_id": "legacy_one_pixel",
            "clean_pixels": [[1, 2]],
        },
        {
            "family_id": "component_contains_response",
            "case_id": "legacy_two_pixels",
            "clean_pixels": [[1, 2], [2, 1]],
        },
        {
            "family_id": "component_contains_response",
            "case_id": "legacy_three_pixels",
            "clean_pixels": [[1, 2], [2, 1], [2, 2]],
        },
        {
            "family_id": "response_outside_component_inside_count_support",
            "case_id": "support_one_pixel",
            "clean_pixels": [[1, 6]],
        },
        {
            "family_id": "response_outside_component_inside_count_support",
            "case_id": "support_two_pixels",
            "clean_pixels": [[1, 6], [2, 5]],
        },
        {
            "family_id": "response_outside_component_inside_count_support",
            "case_id": "support_three_pixels",
            "clean_pixels": [[1, 6], [2, 5], [2, 6]],
        },
    ]
    decision = config["decision_rule"]
    assert isinstance(decision, dict)
    assert decision == {
        "required_passed_case_count": 6,
        "required_passed_family_count": 2,
        "per_case_all_checks_required": True,
        "mean_cannot_override_case_failure": True,
        "counterexample_audit_required": True,
        "numerical_audit_required": True,
        "pass_decision": "PB_NAES_V9_TOY_GATE_PASS",
        "fail_decision": "PB_NAES_V9_TOY_GATE_FAIL",
    }


def test_v9_preserves_optimization_thresholds_and_closed_execution_boundary() -> None:
    config = _load(TOY_CONFIG)
    v8_config = _load(V8_TOY_CONFIG)

    assert config["optimization"] == v8_config["optimization"] == {
        "seed": 7817,
        "optimizer": "adam",
        "updates_per_case": 320,
        "learning_rate": 0.004,
        "weight_decay": 0.0,
        "loss": (
            "unchanged_outcome_complete_transition_plus_absolute_factual"
        ),
        "training_step": "unchanged_outcome_complete_train_step",
        "automatic_retry_allowed": False,
    }
    assert config["thresholds"] == v8_config["thresholds"] == {
        "total_loss_max_exclusive": 0.1,
        "plus_completion_min_exclusive": 0.95,
        "plus_background_max_exclusive": 0.05,
        "factual_miss_target_min_exclusive": 0.95,
        "factual_miss_background_max_exclusive": 0.05,
        "factual_no_miss_max_exclusive": 0.05,
        "clean_D_mean_min_inclusive": 0.8,
        "clean_H_max_abs_max_inclusive": 0.05,
        "clean_G_max_abs_max_inclusive": 0.05,
        "component_H_max_abs_max_inclusive": 0.05,
        "component_G_max_abs_max_inclusive": 0.05,
    }
    expected_boundary = {
        "D_R_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
        "detection_performance_allowed": False,
        "real_bounded_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "cross_detector_authorized": False,
    }
    assert config["execution_boundary"] == expected_boundary
    assert config["execution_boundary"] == v8_config["execution_boundary"]

    proposal = _load(PROPOSAL)
    assert proposal["current_authorization"] == {
        "dataset_free_unit_code": True,
        "dataset_free_toy_code": True,
        "dataset_free_dry_run_code": False,
        "real_D_R": False,
        "D_V": False,
        "D_T": False,
        "formal_800": False,
        "full_CURE": False,
        "cross_detector": False,
    }
