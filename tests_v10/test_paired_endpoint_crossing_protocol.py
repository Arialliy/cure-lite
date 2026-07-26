from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


ROOT = Path(__file__).resolve().parents[1]
V10 = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_endpoint_crossing_objective_v10"
)
V9 = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "phase_balanced_null_anchored_evidence_surplus_v9"
)

DESIGN = ROOT / "CURE_Lite_PECO_v10_模型与代码设计.md"
PROPOSAL = V10 / "proposal_receipt.json"
V9_NEGATIVE_CLOSURE = V9 / "toy_negative_closure_receipt.json"
FROZEN_V8_DECODER = ROOT / "cure_lite" / "conservative_factorized_decoder.py"
PREDECESSOR_LOSS = ROOT / "cure_lite" / "paired_outcome_losses.py"
TRAIN_STEP = ROOT / "cure_lite" / "train" / "paired_outcome_step.py"

DESIGN_SHA256 = (
    "2aae545beb7cef818c64edde12ba21a14d52d9ff42b52cdd15a7355d700a8203"
)
PROPOSAL_SHA256 = (
    "74eb7196944135fa8c620dca8c6593460fc7b7086d08ce6104071dad9d88e47a"
)
PROPOSAL_FINGERPRINT = (
    "377d3b5e5cdf7fdb2b903bd423897b9aea436ee943e00206cd26865b95599365"
)
V9_NEGATIVE_CLOSURE_SHA256 = (
    "b86efe20a4ac737b37b2c55f3af601c44e30a78da8bf20a5053f8e3445df152a"
)
V9_NEGATIVE_CLOSURE_FINGERPRINT = (
    "8df51743d87b002d045ac14ea9702340326e297ba740eab427a990b9c86f28c1"
)
FROZEN_V8_DECODER_SHA256 = (
    "fb7b4aeb16934218d5add300a3be2350d6c77615064486cb92cf93399ab05528"
)
PREDECESSOR_LOSS_SHA256 = (
    "c873b23afe76038f72a93ed99ef9023c090a7fda321c6ac5f725938d774b5c0e"
)
TRAIN_STEP_SHA256 = (
    "479cc663779a48ff7eee447e9582850d8431ccc633e970452ad4c35f526a2265"
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


def test_v10_design_proposal_and_v9_negative_closure_are_exactly_bound() -> None:
    proposal = _load(PROPOSAL)
    v9_closure = _load(V9_NEGATIVE_CLOSURE)

    assert file_sha256(DESIGN) == DESIGN_SHA256
    assert file_sha256(PROPOSAL) == PROPOSAL_SHA256
    assert file_sha256(V9_NEGATIVE_CLOSURE) == V9_NEGATIVE_CLOSURE_SHA256
    _verify_internal_fingerprint(
        proposal,
        field="proposal_fingerprint",
        expected=PROPOSAL_FINGERPRINT,
    )
    _verify_internal_fingerprint(
        v9_closure,
        field="receipt_fingerprint",
        expected=V9_NEGATIVE_CLOSURE_FINGERPRINT,
    )

    assert proposal["schema_version"] == "cure-lite-peco-v10-proposal-v1"
    assert proposal["method_id"] == "peco_v10"
    assert proposal["method_name"] == "Paired Endpoint Crossing Objective"
    assert proposal["status"] == (
        "FROZEN_BEFORE_DATASET_FREE_IMPLEMENTATION_GATE"
    )
    assert proposal["design_document"] == {
        "repo_path": "CURE_Lite_PECO_v10_模型与代码设计.md",
        "file_sha256": DESIGN_SHA256,
    }
    assert proposal["predecessor_v9"] == {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "phase_balanced_null_anchored_evidence_surplus_v9/"
            "toy_negative_closure_receipt.json"
        ),
        "file_sha256": V9_NEGATIVE_CLOSURE_SHA256,
        "receipt_fingerprint": V9_NEGATIVE_CLOSURE_FINGERPRINT,
        "decision": "PB_NAES_V9_DATASET_FREE_TOY_NEGATIVE_CLOSED",
    }
    assert v9_closure["method_id"] == "pb_naes_v9"
    assert v9_closure["decision"] == (
        "PB_NAES_V9_DATASET_FREE_TOY_NEGATIVE_CLOSED"
    )
    assert v9_closure["phase_status"] == "FROZEN_TOY_MODEL_CODE_NONPASS"


def test_v10_binds_the_frozen_decoder_loss_and_train_step_sources() -> None:
    proposal = _load(PROPOSAL)

    assert file_sha256(FROZEN_V8_DECODER) == FROZEN_V8_DECODER_SHA256
    assert file_sha256(PREDECESSOR_LOSS) == PREDECESSOR_LOSS_SHA256
    assert file_sha256(TRAIN_STEP) == TRAIN_STEP_SHA256
    assert proposal["frozen_decoder_binding"] == {
        "method_id": "cc_sea_v8",
        "repo_path": "cure_lite/conservative_factorized_decoder.py",
        "file_sha256": FROZEN_V8_DECODER_SHA256,
        "policy": "read_only_forward_equation_and_topology",
    }
    assert proposal["predecessor_loss_binding"] == {
        "repo_path": "cure_lite/paired_outcome_losses.py",
        "file_sha256": PREDECESSOR_LOSS_SHA256,
        "policy": (
            "read_only_inherited_validation_anchor_zero_strata_and_hierarchy"
        ),
    }
    assert proposal["training_step_binding"] == {
        "repo_path": "cure_lite/train/paired_outcome_step.py",
        "file_sha256": TRAIN_STEP_SHA256,
        "policy": "unchanged",
    }


def test_v10_single_mechanism_is_the_exact_paired_endpoint_crossing_formula() -> None:
    proposal = _load(PROPOSAL)

    assert proposal["single_mechanism"] == {
        "name": "paired_endpoint_crossing_response_risk",
        "response_stratum": "D=bool(label_increment)&image_valid_mask",
        "covered_plus_target": 0,
        "uncovered_minus_target": 1,
        "response_risk": (
            "0.5*(softplus(logits_plus)+softplus(-logits_minus))"
        ),
        "local_zero_risk": (
            "mean_H((sigmoid(logits_minus)-sigmoid(logits_plus))^2)"
        ),
        "global_zero_risk": (
            "mean_G((sigmoid(logits_minus)-sigmoid(logits_plus))^2)"
        ),
        "plus_anchor": "unchanged",
        "hierarchical_active_means": "unchanged",
        "factual_branches": "unchanged",
        "trainable_parameters_added": 0,
        "inference_operations_added": 0,
        "search_parameters_added": 0,
    }
    assert proposal["analytic_contract"] == {
        "plus_response_gradient_sign": "strictly_positive",
        "minus_response_gradient_sign": "strictly_negative",
        "both_high_saturation_is_not_stationary": True,
        "both_low_saturation_is_not_stationary": True,
        "empty_response_uses_no_empty_reduction": True,
        "component_null_uses_zero_risk_only": True,
        "pair_kind_dispatch": False,
        "decoder_forward_inputs": [
            "detached_feature",
            "occupancy",
        ],
        "decoder_topology_changed": False,
        "inference_changed": False,
    }


def test_v10_discloses_candidate_selection_and_keeps_execution_closed() -> None:
    proposal = _load(PROPOSAL)

    assert proposal["development_screen_disclosure"] == {
        "performed_before_proposal_freeze": True,
        "dataset_free": True,
        "cases_reused_from_v8": 6,
        "all_six_passed": True,
        "clean_D_min": 0.991625964641571,
        "clean_D_max": 0.9931955337524414,
        "evidentiary_status": (
            "candidate_selection_only_not_independent_confirmation"
        ),
    }
    assert proposal["current_authorization"] == {
        "dataset_free_unit_code": True,
        "development_regression_code": True,
        "exposure_matched_confirmation_design": True,
        "exposure_matched_confirmation_run": False,
        "dry_run": False,
        "real_D_R": False,
        "D_V": False,
        "D_T": False,
        "formal_800": False,
        "full_CURE": False,
        "cross_detector": False,
    }
