from __future__ import annotations

from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.experiment.geometry_safe_p0_bc_protocol import (
    GEOMETRY_SAFE_P0_BC_CONFIG_SCHEMA,
    GeometrySafeP0BCProtocol,
    load_geometry_safe_p0_bc_protocol,
)
from cure_lite.experiment.p0_protocol import load_p0_config
from tools.run_geometry_safe_p0_bc import (
    GEOMETRY_SAFE_P0_BC_FROZEN_CONFIG_FILE_SHA256,
    _decision,
    _load_and_verify_upstream,
    _three_valued_conjunction,
    build_parser,
)


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "geometry_safe_p0_bc_v1"
    / "config.json"
)
_P0_V1_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "p0_v1"
    / "p0_config.json"
)
_GEOMETRY_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "geometry_safe_p0_v2"
    / "config.json"
)
_UPSTREAM = (
    _ROOT
    / "runs"
    / "irstd1k_stage_a_seed42"
    / "cure_lite_geometry_safe_p0_v2_r1"
)
_UPSTREAM_GEOMETRY_CATALOG = (
    _UPSTREAM / "receipts" / "geometry_catalog.json"
)
_UPSTREAM_P0_A1 = (
    _UPSTREAM / "receipts" / "p0_a1_population_eligibility.json"
)
_UPSTREAM_ELIGIBLE_VIEW = (
    _UPSTREAM / "receipts" / "eligible_view.json"
)
_UPSTREAM_COMPLETE = _UPSTREAM / "COMPLETE.json"


def _payload() -> dict[str, object]:
    return json.loads(_CONFIG.read_text(encoding="utf-8"))


def _runtime_validate_upstream(config: GeometrySafeP0BCProtocol) -> None:
    _load_and_verify_upstream(
        config,
        geometry_config_path=_GEOMETRY_CONFIG,
        geometry_catalog_path=_UPSTREAM_GEOMETRY_CATALOG,
        p0_a1_path=_UPSTREAM_P0_A1,
        eligible_view_path=_UPSTREAM_ELIGIBLE_VIEW,
        geometry_complete_path=_UPSTREAM_COMPLETE,
    )


def _mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_mapping_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_mapping_keys(item))
        return keys
    return set()


def test_geometry_safe_p0_bc_config_is_canonical_and_fingerprinted() -> None:
    payload = _payload()
    config = load_geometry_safe_p0_bc_protocol(_CONFIG)
    p0_v1 = load_p0_config(_P0_V1_CONFIG)

    assert config.schema_version == GEOMETRY_SAFE_P0_BC_CONFIG_SCHEMA
    assert config.canonical_payload() == payload
    assert config.fingerprint == stable_fingerprint(payload)
    assert len(config.fingerprint) == 64
    assert (
        file_sha256(_CONFIG)
        == GEOMETRY_SAFE_P0_BC_FROZEN_CONFIG_FILE_SHA256
    )
    assert (
        payload["overlap"]
        == p0_v1.canonical_payload()["overlap"]
    )
    assert (
        payload["separability"]
        == p0_v1.canonical_payload()["separability"]
    )


def test_geometry_safe_p0_bc_is_d_r_only_and_authorizes_no_mutation() -> None:
    config = load_geometry_safe_p0_bc_protocol(_CONFIG)

    assert config.split == "D_R"
    assert config.execution_policy.allowed_runtime_splits == ("D_R",)
    assert config.execution_policy.create_only_output is True
    assert config.execution_policy.allow_training is False
    assert config.execution_policy.allow_calibration is False
    assert config.execution_policy.allow_inference is False
    assert config.execution_policy.allow_d_v_access is False
    assert config.execution_policy.allow_d_t_access is False
    assert config.execution_policy.allow_candidate_s_construction is False
    assert config.execution_policy.allow_backbone_integration is False
    assert config.decision_policy.authorizes_candidate_s_construction is False
    assert config.decision_policy.authorizes_training is False
    assert config.decision_policy.authorizes_d_v_evaluation is False
    assert config.decision_policy.authorizes_full_cure is False

    options = {
        action.dest
        for action in build_parser()._actions
        if action.dest != "help"
    }
    assert options == {
        "manifest",
        "state_index",
        "config",
        "geometry_config",
        "geometry_catalog_receipt",
        "p0_a1_receipt",
        "eligible_view_receipt",
        "geometry_complete",
        "p0_v1_config",
        "output",
    }
    forbidden_fragments = {
        "train",
        "calibr",
        "infer",
        "d_v",
        "d_t",
        "candidate_s",
    }
    assert not any(
        fragment in option
        for option in options
        for fragment in forbidden_fragments
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    (
        ("factual_targets", 32),
        ("factual_groups", 24),
        ("legal_targets", 206),
        ("legal_source_images", 149),
        ("legal_groups", 145),
        ("role_overlap_groups", 14),
        ("role_overlap_factual_targets", 18),
        ("role_overlap_legal_targets", 25),
        ("legal_exclusive_groups", 131),
    ),
)
def test_geometry_safe_p0_bc_population_is_strict(
    field: str,
    expected: int,
) -> None:
    config = load_geometry_safe_p0_bc_protocol(_CONFIG)
    assert getattr(config.population_binding, field) == expected

    payload = _payload()
    payload["population_binding"][field] = expected + 1
    with pytest.raises(ValueError, match=f"population_binding.{field}"):
        GeometrySafeP0BCProtocol.from_mapping(payload)


def test_geometry_safe_p0_bc_config_has_no_target_identity_allowlist() -> None:
    keys = {key.lower() for key in _mapping_keys(_payload())}
    assert not any("allowlist" in key for key in keys)
    assert not any("hardcoded_identity" in key for key in keys)
    assert {
        "target_identity_allowlist",
        "target_id_allowlist",
        "sample_id_allowlist",
        "legal_target_ids",
        "factual_target_ids",
        "identity_exclusions",
    }.isdisjoint(keys)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("execution_policy", "allowed_runtime_splits"),
        ("execution_policy", "allow_training"),
        ("execution_policy", "allow_calibration"),
        ("execution_policy", "allow_inference"),
        ("execution_policy", "allow_d_v_access"),
        ("execution_policy", "allow_d_t_access"),
        ("execution_policy", "allow_candidate_s_construction"),
        ("decision_policy", "authorizes_candidate_s_construction"),
        ("decision_policy", "authorizes_training"),
        ("decision_policy", "authorizes_d_v_evaluation"),
        ("decision_policy", "authorizes_full_cure"),
    ),
)
def test_geometry_safe_p0_bc_rejects_scope_expansion(
    section: str,
    field: str,
) -> None:
    payload = _payload()
    if field == "allowed_runtime_splits":
        payload[section][field] = ["D_R", "D_V"]
    else:
        payload[section][field] = True
    with pytest.raises(ValueError):
        GeometrySafeP0BCProtocol.from_mapping(payload)


def test_geometry_safe_p0_bc_rejects_d_v_and_unknown_fields() -> None:
    payload = _payload()
    payload["split"] = "D_V"
    with pytest.raises(ValueError, match="only D_R"):
        GeometrySafeP0BCProtocol.from_mapping(payload)

    payload = _payload()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="fields are not canonical"):
        GeometrySafeP0BCProtocol.from_mapping(payload)


@pytest.mark.parametrize(
    "field",
    (
        "p0_a1_file_sha256",
        "p0_a1_receipt_fingerprint",
        "eligible_view_file_sha256",
        "eligible_view_receipt_fingerprint",
        "eligible_catalog_fingerprint",
    ),
)
def test_geometry_safe_p0_bc_rejects_malformed_upstream_digests(
    field: str,
) -> None:
    payload = _payload()
    payload["upstream_binding"][field] = "not-a-sha256"
    with pytest.raises(ValueError, match="lowercase SHA256"):
        GeometrySafeP0BCProtocol.from_mapping(payload)


@pytest.mark.parametrize(
    "field",
    (
        "p0_a1_file_sha256",
        "p0_a1_receipt_fingerprint",
        "eligible_view_file_sha256",
        "eligible_view_receipt_fingerprint",
        "eligible_catalog_fingerprint",
    ),
)
def test_geometry_safe_p0_bc_runtime_rejects_wrong_upstream_binding(
    field: str,
) -> None:
    payload = _payload()
    payload["upstream_binding"][field] = "0" * 64
    config = GeometrySafeP0BCProtocol.from_mapping(payload)
    with pytest.raises(RuntimeError, match="upstream|changed"):
        _runtime_validate_upstream(config)


def test_three_valued_conjunction_truth_table() -> None:
    states = ("pass", "fail", "inconclusive")
    for combination in product(states, repeat=3):
        expected = (
            "fail"
            if "fail" in combination
            else "inconclusive"
            if "inconclusive" in combination
            else "pass"
        )
        assert _three_valued_conjunction(combination) == expected

    with pytest.raises(RuntimeError, match="unknown"):
        _three_valued_conjunction(("pass", "not_evaluated"))


@pytest.mark.parametrize(
    ("p0_b_state", "p0_c_state", "expected_state", "expected_route"),
    tuple(
        (
            p0_b_state,
            p0_c_state,
            (
                "fail"
                if "fail" in (p0_b_state, p0_c_state)
                else "inconclusive"
                if "inconclusive" in (p0_b_state, p0_c_state)
                else "pass"
            ),
                (
                    "redesign_synthetic_state"
                    if "fail" in (p0_b_state, p0_c_state)
                    else "resolve_p0_bc_inconclusive"
                    if "inconclusive" in (p0_b_state, p0_c_state)
                    else "eligible_to_design_candidate_s"
                ),
        )
        for p0_b_state, p0_c_state in product(
            ("pass", "fail", "inconclusive"),
            repeat=2,
        )
    ),
)
def test_geometry_safe_p0_bc_decision_truth_table(
    p0_b_state: str,
    p0_c_state: str,
    expected_state: str,
    expected_route: str,
) -> None:
    config = load_geometry_safe_p0_bc_protocol(_CONFIG)
    decision = _decision(
        config,
        population={"receipt_fingerprint": "population"},
        p0_b={
            "receipt_fingerprint": "p0-b",
            "diagnostic_status": p0_b_state,
        },
        p0_c={
            "receipt_fingerprint": "p0-c",
            "diagnostic_status": p0_c_state,
        },
    )

    expected_bool = (
        True
        if expected_state == "pass"
        else False
        if expected_state == "fail"
        else None
    )
    assert decision["p0_a1_b_c_gate_state"] == expected_state
    assert decision["p0_a1_b_c_pass"] is expected_bool
    assert decision["next_route"] == expected_route
    assert decision["eligible_to_design_candidate_s"] is (
        expected_state == "pass"
    )
    assert decision["eligible_to_construct_candidate_s"] is False
    assert decision["authorizes_candidate_s_construction"] is False
    assert decision["authorizes_training"] is False
    assert decision["authorizes_d_v_evaluation"] is False
    assert decision["authorizes_full_cure"] is False
    assert decision["candidate_distribution_constructed"] is False
    assert decision["p0_d_executed"] is False
    assert decision["training_performed"] is False
    assert decision["d_v_accessed"] is False
    assert decision["d_t_accessed"] is False

    assert decision["all_p0_gate_state"] == (
        "fail" if expected_state == "fail" else "inconclusive"
    )
    assert decision["all_p0_pass"] is None
    assert (
        decision["all_p0_completion_status"]
        == "not_complete_p0_d_not_evaluated"
    )
