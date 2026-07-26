from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


_ROOT = Path(__file__).resolve().parents[1]
_V7 = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "continuously_recoverable_log_vacancy_evidence_crossing_v7"
)
_V6 = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "polarity_recoverable_subpixel_vacancy_evidence_factorization_v6"
)
_METHOD_PROPOSAL = _V7 / "proposal_receipt.json"
_TOY_RESULT = _V7 / "toy_gate_result.json"
_TOY_CLOSURE = _V7 / "toy_gate_closure_receipt.json"
_BOUNDED_PROPOSAL = _V7 / "bounded_implementation_proposal_receipt.json"
_BOUNDED_CONFIG = _V7 / "bounded_config.json"
_DRY_CONFIG = _V7 / "bounded_dry_run_config.json"
_V6_BOUNDED_CONFIG = _V6 / "bounded_config.json"
_V6_NEGATIVE_CLOSURE = _V6 / "bounded_negative_closure_receipt.json"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _verify_optional_fingerprint(
    payload: dict[str, object],
    *,
    field: str,
) -> None:
    fingerprint = payload.get(field)
    if fingerprint is None:
        return
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64
    unsigned = dict(payload)
    unsigned.pop(field)
    assert fingerprint == stable_fingerprint(unsigned)


def _verify_optional_binding(
    binding: dict[str, object],
    *,
    path: Path,
    fingerprint_field: str,
    payload_fingerprint_field: str,
) -> None:
    if "file_sha256" in binding:
        assert binding["file_sha256"] == file_sha256(path)
    if fingerprint_field in binding:
        payload = _load(path)
        assert binding[fingerprint_field] == payload[payload_fingerprint_field]


def test_v7_bounded_protocol_binds_the_frozen_toy_gate() -> None:
    method = _load(_METHOD_PROPOSAL)
    toy_result = _load(_TOY_RESULT)
    toy_closure = _load(_TOY_CLOSURE)
    proposal = _load(_BOUNDED_PROPOSAL)
    config = _load(_BOUNDED_CONFIG)
    dry = _load(_DRY_CONFIG)

    assert method["proposal_fingerprint"] == (
        "9d291e6ad9ec0869aa0ab0eaebcb219cd62678420375f56af480ba105208dbf2"
    )
    assert file_sha256(_METHOD_PROPOSAL) == (
        "fa72f4ef850f72a65003e913db1b1230d7b0b45046faf61950fb1e4ef80d3c4f"
    )
    assert toy_result["result_fingerprint"] == (
        "ff5e76894066eb953773c444060ce6783dd12cdc2d99aa9d46700086db75ffaa"
    )
    assert file_sha256(_TOY_RESULT) == (
        "01a582dbd645e47be64601ae3b42e62b889f5c838e926ed5b09c76af3f5c24b6"
    )
    assert toy_closure["receipt_fingerprint"] == (
        "f95573edd8b842980d5b175b1aac8caf753f6c279342da8e29f54e165b1e255f"
    )
    assert file_sha256(_TOY_CLOSURE) == (
        "25c3317045533f4116b8873d892fcd2c0e866d3e991843a4c0c8e872142f0fe5"
    )
    assert toy_closure["gate_summary"]["bounded_code_creation_authorized"]
    assert not toy_closure["gate_summary"]["real_D_R_bounded_authorized"]

    for payload in (proposal, config, dry):
        assert payload["method_id"] == "cr_lvec_v7"

    proposal_toy = proposal["protocol_bindings"]["toy_gate_closure"]
    config_toy = config["toy_gate_authorization"]
    dry_toy = dry["toy_gate_closure_binding"]
    assert proposal_toy["file_sha256"] == file_sha256(_TOY_CLOSURE)
    assert config_toy["closure_file_sha256"] == file_sha256(_TOY_CLOSURE)
    assert dry_toy["file_sha256"] == file_sha256(_TOY_CLOSURE)
    assert proposal_toy["receipt_fingerprint"] == (
        toy_closure["receipt_fingerprint"]
    )
    assert config_toy["closure_receipt_fingerprint"] == (
        toy_closure["receipt_fingerprint"]
    )
    assert dry_toy["receipt_fingerprint"] == (
        toy_closure["receipt_fingerprint"]
    )


def test_bounded_implementation_proposal_is_additive_and_not_run_authority() -> None:
    proposal = _load(_BOUNDED_PROPOSAL)
    _verify_optional_fingerprint(proposal, field="receipt_fingerprint")

    assert proposal["schema_version"] == (
        "cure-lite-cr-lvec-v7-bounded-implementation-proposal-v1"
    )
    assert proposal["phase_status"] == "SPECIFIED_NOT_IMPLEMENTED"
    assert proposal["decision"] == (
        "CR_LVEC_V7_BOUNDED_IMPLEMENTATION_CREATION_AUTHORIZED"
    )
    scope = proposal["implementation_scope"]
    assert scope["additive_runtime_files"] == [
        "cure_lite/experiment/crossing_factorized_outcome_bounded.py",
        "tools/dry_run_crossing_factorized_outcome_bounded.py",
        "tools/run_crossing_factorized_outcome_bounded.py",
    ]
    assert scope["additive_test_files"] == [
        "tests/test_crossing_factorized_outcome_bounded.py",
        "tests/test_dry_run_crossing_factorized_outcome_bounded_cli.py",
        "tests/test_run_crossing_factorized_outcome_bounded_cli.py",
        "tests/test_crossing_factorized_bounded_implementation_closure.py",
    ]
    assert scope["real_run_authorization_created_in_this_code_stage"] is False
    assert scope["existing_decoder_may_be_modified"] is False
    assert scope["existing_loss_may_be_modified"] is False
    assert scope["existing_train_step_may_be_modified"] is False
    assert scope["base_or_backbone_may_be_modified"] is False

    boundary = proposal["execution_boundary"]
    assert boundary["D_R_protocol_metadata_may_be_read"] is True
    assert (
        boundary["D_R_dataset_or_cached_tensor_payload_access_allowed"]
        is False
    )
    assert boundary["real_D_R_bounded_execution_authorized"] is False
    assert boundary["D_V_access_allowed"] is False
    assert boundary["D_T_access_allowed"] is False
    assert boundary["formal_800_allowed"] is False
    assert boundary["full_CURE_allowed"] is False
    assert boundary["other_detector_integration_allowed"] is False

    closure = proposal["implementation_closure_contract"]
    assert closure["must_record_D_R_payload_accessed_false"] is True
    assert closure["must_record_real_D_R_bounded_authorized_false"] is True
    assert closure["may_directly_authorize_real_D_R_execution"] is False
    assert (
        closure["may_make_single_real_D_R_authorization_eligible"] is True
    )


def test_bounded_config_preserves_v6_population_budget_and_gates() -> None:
    config = _load(_BOUNDED_CONFIG)
    v6 = _load(_V6_BOUNDED_CONFIG)
    _verify_optional_fingerprint(config, field="config_fingerprint")

    assert config["schema_version"] == (
        "cure-lite-cr-lvec-v7-bounded-config-v1"
    )
    assert config["dataset"] == "IRSTD-1K"
    assert config["split"] == "D_R"
    assert config["anchor_population"] == v6["anchor_population"]
    assert config["outcome_population"] == v6["outcome_population"]
    assert config["budget"] == v6["budget"]
    assert config["bounded_gates"] == v6["bounded_gates"]
    assert config["optimization"]["loss"] == v6["optimization"]["loss"]
    for key in ("optimizer", "learning_rate", "weight_decay", "seed"):
        assert config["optimization"][key] == v6["optimization"][key]

    decoder = config["optimization"]["decoder"]
    assert decoder["feature_channels"] == 64
    assert decoder["feature_stride"] == 4
    assert decoder["occupancy_burden_policy"] == (
        "nearest_log1p_local_occupancy_count_v1"
    )
    assert decoder["forward_evidence_transform"] == (
        "positive_exponential_ratio_crossing_v1"
    )
    assert decoder["backward_surrogate_policy"] == (
        "full_axis_exponential_continuation_v1"
    )
    assert decoder["zero_boundary_policy"] == (
        "full_axis_unit_gradient_v1"
    )
    assert decoder["logit_composition_policy"] == (
        "baseline_plus_crossing_evidence_v1"
    )
    assert decoder["resize_policy"] == (
        "bilinear_raw_nearest_burden_then_crossing_v1"
    )

    gates = config["structural_gates"]
    assert gates["decoder_parameter_count"] == 4385
    assert gates["decoder_parameter_tensor_count"] == 6
    assert gates["parent_reciprocal_vacancy_path_call_count"] == 0
    assert gates["float32_negative_finite_nonzero_probe"] == -80.0
    assert gates["float32_first_zero_recovery_probe"] == -104.0
    assert gates["float32_largest_finite_positive_probe"] == 88.0
    assert gates["float32_first_nonfinite_positive_probe"] == 89.0
    assert gates["identity_endpoint_exact"] is True
    assert (
        gates["local_count_changed_and_unchanged_support_both_nonempty"]
        is True
    )
    assert gates["two_endpoint_batch_forward_required"] is True


def test_sync_policy_allows_only_bounded_400_not_formal_800() -> None:
    proposal = _load(_BOUNDED_PROPOSAL)
    config = _load(_BOUNDED_CONFIG)
    dry = _load(_DRY_CONFIG)

    p_sync = proposal["cuda_synchronization_policy"]
    c_sync = config["cuda_synchronization_policy"]
    d_sync = dry["cuda_synchronization_boundary"]
    assert p_sync["strict_finite_and_nonzero_recovery_check_retained"] is True
    assert p_sync[
        "potential_host_synchronization_check_sites_per_decoder_forward"
    ] == 1
    assert p_sync["bounded_potential_host_synchronization_check_sites"] == (
        400 * 3
    )
    assert p_sync["formal_800_potential_host_synchronization_check_sites"] == (
        32000 * 3
    )
    assert p_sync["bounded_400_may_be_authorized_after_implementation_closure"]
    assert p_sync["formal_800_authorized"] is False
    assert p_sync["zero_synchronization_claim_allowed"] is False

    assert c_sync["strict_finite_and_nonzero_recovery_check_retained"] is True
    assert c_sync["bounded_potential_host_synchronization_check_sites"] == 1200
    assert c_sync["formal_800_potential_host_synchronization_check_sites"] == (
        96000
    )
    assert c_sync["formal_800_allowed"] is False
    assert d_sync["strict_runtime_check_retained"] is True
    assert d_sync["bounded_potential_host_synchronization_check_sites"] == 1200
    assert d_sync["formal_800_potential_host_synchronization_check_sites"] == (
        96000
    )
    assert d_sync["formal_800_authorized"] is False


def test_dry_run_is_synthetic_deterministic_and_cannot_read_real_data() -> None:
    dry = _load(_DRY_CONFIG)
    _verify_optional_fingerprint(dry, field="config_fingerprint")

    assert dry["schema_version"] == (
        "cure-lite-cr-lvec-v7-bounded-dry-run-config-v1"
    )
    assert dry["mode"] == "synthetic_bounded_implementation_dry_run"
    data = dry["data_contract"]
    assert data["provider"] == "fixed_in_memory_synthetic_pair_provider_v1"
    assert data["dataset_name"] is None
    assert data["split"] is None
    assert data["allowed_dataset_splits"] == []
    assert data["filesystem_dataset_root_allowed"] is False
    assert data["real_catalog_loader_allowed"] is False
    assert data["real_catalog_loader_call_count_required"] == 0
    assert (
        data["D_R_dataset_or_cached_tensor_payload_access_allowed"] is False
    )
    assert data["D_V_access_allowed"] is False
    assert data["D_T_access_allowed"] is False

    optimization = dry["optimization"]
    assert optimization["device"] == "cpu"
    assert optimization["optimizer_updates"] == 8
    assert optimization["decoder_forward_calls_per_update"] == 3
    assert optimization["decoder_states_per_update"] == 12
    assert optimization["automatic_retry_allowed"] is False
    assert optimization["resume_allowed"] is False

    replay = dry["replay_contract"]
    assert replay["independent_process_replay_count"] == 2
    assert replay["temporary_outputs_must_match_before_canonical_write"]
    assert replay["canonical_payload_byte_identity_required"]
    assert replay["canonical_write_policy"] == (
        "create_only_after_two_temporary_replays_match"
    )

    decision = dry["decision_rule"]
    assert decision["all_required_checks_must_pass"] is True
    assert decision["dry_run_can_authorize_real_D_R_execution"] is False
    boundary = dry["execution_boundary"]
    assert boundary["D_R_payload_access_allowed"] is False
    assert boundary["real_D_R_bounded_allowed"] is False
    assert (
        boundary["real_run_authorization_receipt_required_or_created"]
        is False
    )
    assert boundary["formal_800_allowed"] is False


def test_chain_placeholders_are_paths_only_until_root_signs_them() -> None:
    proposal = _load(_BOUNDED_PROPOSAL)
    config = _load(_BOUNDED_CONFIG)
    dry = _load(_DRY_CONFIG)

    proposal_binding = config["bounded_implementation_proposal_binding"]
    dry_proposal_binding = dry["bounded_implementation_proposal_binding"]
    dry_config_binding = dry["bounded_config_binding"]
    assert proposal_binding["repo_path"] == (
        "protocols/IRSTD-1K/"
        "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
        "bounded_implementation_proposal_receipt.json"
    )
    assert dry_proposal_binding["repo_path"] == proposal_binding["repo_path"]
    assert dry_config_binding["repo_path"] == (
        "protocols/IRSTD-1K/"
        "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
        "bounded_config.json"
    )

    _verify_optional_binding(
        proposal_binding,
        path=_BOUNDED_PROPOSAL,
        fingerprint_field="receipt_fingerprint",
        payload_fingerprint_field="receipt_fingerprint",
    )
    _verify_optional_binding(
        dry_proposal_binding,
        path=_BOUNDED_PROPOSAL,
        fingerprint_field="receipt_fingerprint",
        payload_fingerprint_field="receipt_fingerprint",
    )
    _verify_optional_binding(
        dry_config_binding,
        path=_BOUNDED_CONFIG,
        fingerprint_field="config_fingerprint",
        payload_fingerprint_field="config_fingerprint",
    )

    assert file_sha256(_V6_BOUNDED_CONFIG) == (
        proposal["protocol_bindings"]["predecessor_v6_bounded_config"][
            "file_sha256"
        ]
    )
    assert file_sha256(_V6_NEGATIVE_CLOSURE) == (
        proposal["protocol_bindings"]["predecessor_v6_negative_closure"][
            "file_sha256"
        ]
    )
