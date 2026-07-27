from __future__ import annotations

import json

from cure_lite.phase_resolved_relation_preflight import (
    PFCR_PREFLIGHT_ALGORITHM_VERSION,
    build_phase_resolved_relation_preflight_receipt,
)


def test_preflight_passes_relation_input_but_not_decoder_training() -> None:
    receipt = build_phase_resolved_relation_preflight_receipt(
        max_examples=2
    )

    assert receipt["schema_version"] == (
        PFCR_PREFLIGHT_ALGORITHM_VERSION
    )
    assert receipt["scope"]["model"] == "CURE-Lite"
    assert receipt["scope"]["full_CURE_in_scope"] is False
    assert receipt["analytic_reference"]["state_count"] == 32
    assert receipt["analytic_reference"]["maximum_absolute_error"] == 0.0
    assert receipt["analytic_reference"]["mismatch_pixel_count"] == 0
    assert receipt["analytic_reference"]["exact_completion_match"] is True
    role = receipt["relation_role_identifiability"]
    assert role["absolute_origin_used"] is False
    assert role["raw_feature_value_used"] is False
    assert role["prototype_id_used"] is False
    assert role["metadata_used"] is False
    assert role["conflict_key_count"] == 0
    matched = receipt["matched_same_geometry_relevance"]
    assert matched["matched_group_count"] == 8
    assert matched["passed_group_count"] == 8
    assert matched["all_passed"] is True
    assert receipt["phase_sufficiency"][
        "phase_patterns_have_distinct_global_encoder_outputs"
    ] is True
    decision = receipt["decision"]
    assert decision["input_contract_v2_pass"] is True
    assert decision["relation_state_implementation_authorized"] is True
    assert decision["full_decoder_training_authorized"] is False
    assert decision["old_nlcc_training_authorized"] is False


def test_preflight_receipt_is_byte_deterministic() -> None:
    first = build_phase_resolved_relation_preflight_receipt(
        max_examples=2
    )
    second = build_phase_resolved_relation_preflight_receipt(
        max_examples=2
    )
    first_bytes = json.dumps(
        first,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    second_bytes = json.dumps(
        second,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert first_bytes == second_bytes
    assert first["receipt_fingerprint"] == second["receipt_fingerprint"]
