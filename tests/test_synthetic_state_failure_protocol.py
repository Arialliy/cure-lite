from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.experiment.synthetic_state_failure_protocol import (
    SYNTHETIC_STATE_FAILURE_CONFIG_SCHEMA,
    SyntheticStateFailureProtocol,
    load_synthetic_state_failure_protocol,
)


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "synthetic_state_failure_attribution_v1"
    / "config.json"
)
_CONFIG_SHA256 = (
    "8933113e745ab42119e90a0a3f2b4366290f38a6b523251d94d39bc5665e6161"
)
_RUN_ROOT = (
    _ROOT / "runs" / "irstd1k_stage_a_seed42"
)
_R1 = _RUN_ROOT / "cure_lite_geometry_safe_p0_bc_v1_r1"
_R2 = _RUN_ROOT / "cure_lite_geometry_safe_p0_bc_v1_r2"
_OLD_P0_B = (
    _RUN_ROOT
    / "cure_lite_p0_v1_r3"
    / "receipts"
    / "p0_b_support.json"
)


def _payload() -> dict[str, object]:
    return json.loads(_CONFIG.read_text(encoding="utf-8"))


def test_failure_protocol_config_is_canonical_and_frozen() -> None:
    payload = _payload()
    config = load_synthetic_state_failure_protocol(_CONFIG)

    assert config.schema_version == SYNTHETIC_STATE_FAILURE_CONFIG_SCHEMA
    assert config.protocol_id == (
        "irstd1k-dr-synthetic-state-failure-attribution-v1"
    )
    assert config.stage_role == "descriptive-diagnostic-only"
    assert config.canonical_payload() == payload
    assert config.fingerprint == stable_fingerprint(payload)
    assert file_sha256(_CONFIG) == _CONFIG_SHA256


def test_failure_protocol_binds_formal_209_and_206_evidence() -> None:
    config = load_synthetic_state_failure_protocol(_CONFIG)
    authority = config.authority_binding

    assert file_sha256(_R1 / "COMPLETE.json") == (
        authority.p0_bc_r1_complete_file_sha256
    )
    assert file_sha256(_R2 / "COMPLETE.json") == (
        authority.p0_bc_r2_complete_file_sha256
    )
    assert (
        authority.p0_bc_r1_complete_file_sha256
        == authority.p0_bc_r2_complete_file_sha256
    )
    assert file_sha256(_R1 / "receipts" / "population_binding.json") == (
        authority.p0_bc_population_file_sha256
    )
    assert file_sha256(_R1 / "receipts" / "p0_b_support.json") == (
        authority.p0_bc_p0_b_file_sha256
    )
    assert file_sha256(_R1 / "receipts" / "p0_c_screening.json") == (
        authority.p0_bc_p0_c_file_sha256
    )
    assert file_sha256(_R1 / "receipts" / "decision.json") == (
        authority.p0_bc_decision_file_sha256
    )
    assert file_sha256(_OLD_P0_B) == (
        authority.legacy_209_p0_b_file_sha256
    )


def test_failure_protocol_is_d_r_only_and_authorizes_no_action() -> None:
    config = load_synthetic_state_failure_protocol(_CONFIG)
    execution = config.execution_policy
    decision = config.decision_policy

    assert config.split == "D_R"
    assert execution.allowed_runtime_splits == ("D_R",)
    assert execution.create_only_output is True
    assert execution.allow_training is False
    assert execution.allow_calibration is False
    assert execution.allow_inference is False
    assert execution.allow_d_v_access is False
    assert execution.allow_d_t_access is False
    assert execution.allow_candidate_s_construction is False
    assert execution.allow_p0_d is False
    assert execution.allow_transformation_construction is False
    assert execution.allow_full_cure is False
    assert execution.allow_backbone_integration is False
    assert decision.overall_scientific_gate == (
        "none-descriptive-diagnostic-only"
    )
    assert decision.separate_transformation_protocol_required is True
    assert decision.authorizes_transformation_construction is False
    assert decision.authorizes_candidate_s_construction is False
    assert decision.authorizes_p0_d is False
    assert decision.authorizes_training is False
    assert decision.authorizes_d_v_evaluation is False
    assert decision.authorizes_full_cure is False


@pytest.mark.parametrize(
    "field",
    (
        "allow_training",
        "allow_calibration",
        "allow_inference",
        "allow_d_v_access",
        "allow_d_t_access",
        "allow_candidate_s_construction",
        "allow_p0_d",
        "allow_transformation_construction",
        "allow_full_cure",
        "allow_backbone_integration",
    ),
)
def test_failure_protocol_rejects_execution_scope_expansion(
    field: str,
) -> None:
    payload = _payload()
    payload["execution_policy"][field] = True
    with pytest.raises(ValueError):
        SyntheticStateFailureProtocol.from_mapping(payload)


@pytest.mark.parametrize(
    "field",
    (
        "authorizes_transformation_construction",
        "authorizes_candidate_s_construction",
        "authorizes_p0_d",
        "authorizes_training",
        "authorizes_d_v_evaluation",
        "authorizes_full_cure",
    ),
)
def test_failure_protocol_rejects_decision_scope_expansion(
    field: str,
) -> None:
    payload = _payload()
    payload["decision_policy"][field] = True
    with pytest.raises(ValueError):
        SyntheticStateFailureProtocol.from_mapping(payload)


def test_failure_protocol_rejects_non_d_r_and_unknown_fields() -> None:
    payload = _payload()
    payload["split"] = "D_V"
    with pytest.raises(ValueError, match="split"):
        SyntheticStateFailureProtocol.from_mapping(payload)

    payload = _payload()
    payload["execution_policy"]["allowed_runtime_splits"] = ["D_R", "D_T"]
    with pytest.raises(ValueError, match="exactly D_R"):
        SyntheticStateFailureProtocol.from_mapping(payload)

    payload = _payload()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="fields are not canonical"):
        SyntheticStateFailureProtocol.from_mapping(payload)


def test_failure_protocol_population_and_exclusions_are_exact() -> None:
    population = load_synthetic_state_failure_protocol(
        _CONFIG
    ).population_binding

    assert (
        population.factual_targets,
        population.factual_groups,
        population.legacy_legal_targets,
        population.geometry_safe_legal_targets,
    ) == (32, 24, 209, 206)
    assert (
        population.geometry_safe_legal_source_images,
        population.geometry_safe_legal_groups,
        population.role_overlap_groups,
        population.legal_exclusive_groups,
    ) == (149, 145, 14, 131)
    assert (
        population.dual_role_source_images,
        population.dual_role_source_factual_targets,
        population.dual_role_source_legal_targets,
    ) == (14, 18, 21)
    assert [
        (item.sample_id, item.gt_id, item.pred_id)
        for item in population.excluded_legal_identities
    ] == [
        ("XDU486", 1, 1),
        ("XDU526", 1, 1),
        ("XDU965", 1, 1),
    ]

    payload = _payload()
    payload["population_binding"]["geometry_safe_legal_targets"] = 205
    with pytest.raises(ValueError, match="geometry_safe_legal_targets"):
        SyntheticStateFailureProtocol.from_mapping(payload)

    payload = _payload()
    payload["population_binding"]["excluded_legal_identities"][0][
        "sample_id"
    ] = "XDU000"
    with pytest.raises(ValueError, match="excluded_legal_identities"):
        SyntheticStateFailureProtocol.from_mapping(payload)


def test_failure_protocol_factor_roles_do_not_confuse_proxy_and_input() -> None:
    taxonomy = load_synthetic_state_failure_protocol(
        _CONFIG
    ).factor_taxonomy
    by_id = {item.factor_id: item for item in taxonomy.factors}

    assert taxonomy.decoder_input_factors == (
        "F_local",
        "F_background_global",
        "O",
    )
    assert taxonomy.proxy_factors == ("P",)
    assert by_id["P"].decoder_observed is False
    assert by_id["P"].claim_scope == (
        "fixed-seven-dimensional-proxy-not-decoder-input"
    )
    assert all(
        by_id[factor].decoder_observed
        for factor in taxonomy.decoder_input_factors
    )
    assert "do-not-prove" in taxonomy.interpretation_limit


@pytest.mark.parametrize(
    "field",
    (
        "arbitrary_subset_search_allowed",
        "descriptor_selection_allowed",
        "hyperparameter_search_allowed",
    ),
)
def test_failure_protocol_forbids_descriptor_search(field: str) -> None:
    config = load_synthetic_state_failure_protocol(_CONFIG)
    assert getattr(config.probe_freeze, field) is False

    payload = _payload()
    payload["probe_freeze"][field] = True
    with pytest.raises(ValueError, match=field):
        SyntheticStateFailureProtocol.from_mapping(payload)


def test_failure_protocol_probe_set_and_fit_are_exact() -> None:
    probes = load_synthetic_state_failure_protocol(_CONFIG).probe_freeze
    assert probes.primary_single_blocks == (
        "G_full",
        "W",
        "P",
        "F_local",
        "F_background_global",
        "O",
    )
    assert probes.decoder_input_probe_union == (
        "F_local",
        "F_background_global",
        "O",
    )
    assert probes.fixed_drop_one_probes == (
        "drop_F_local",
        "drop_F_background_global",
        "drop_O",
    )
    assert probes.strata == (
        "all-geometry-safe-population",
        "shared-manifest-groups-only",
        "selected-dual-role-source-images-transductive-sensitivity",
    )
    assert probes.feature_components == 6
    assert probes.feature_projection_rule == (
        "robust-scaled-legal-only-pca-plus-residual-v1"
    )
    assert probes.oof_feature_projection_fit_population == (
        "training-fold-legal-targets-only"
    )
    assert probes.coverage_feature_projection_fit_population == (
        "all-geometry-safe-legal-targets"
    )
    assert probes.mmd_feature_projection_fit_population == (
        "legal-exclusive-manifest-groups-only"
    )
    assert probes.fixed_fit_failure_policy == (
        "record-block-inconclusive-no-refit-no-override-v1"
    )
    assert probes.partial_stratum_completion_allowed is True
    assert probes.oof_fit_rule == (
        "group-oof-projection-scale-and-classifier-fold-local-v1"
    )
    assert probes.source_centering_interpretation == (
        "selected-overlap-transductive-sensitivity-not-source-elimination"
    )
    assert probes.mmd_rule == (
        "descriptive-group-u-multiscale-rbf-matched-legal-reference-v1"
    )
    assert probes.multiple_comparison_rule == (
        "none-descriptive-q95-crossing-six-primary-blocks-v1"
    )

    payload = _payload()
    payload["probe_freeze"]["primary_single_blocks"].append("best-result")
    with pytest.raises(ValueError, match="primary_single_blocks"):
        SyntheticStateFailureProtocol.from_mapping(payload)

    payload = _payload()
    payload["probe_freeze"]["classifier_l2"] = True
    with pytest.raises(ValueError, match="classifier_l2"):
        SyntheticStateFailureProtocol.from_mapping(payload)

    payload = _payload()
    payload["probe_freeze"]["folds"] = 5.0
    with pytest.raises(ValueError, match="folds"):
        SyntheticStateFailureProtocol.from_mapping(payload)


def test_failure_protocol_freezes_two_by_two_coverage_replay() -> None:
    decomposition = load_synthetic_state_failure_protocol(
        _CONFIG
    ).coverage_transition_decomposition

    assert decomposition.population_axis == ("legal209", "legal206")
    assert decomposition.fit_axis == ("fit209", "fit206")
    assert decomposition.cells == (
        "legal209-fit209",
        "legal206-fit209",
        "legal209-fit206",
        "legal206-fit206",
    )
    assert decomposition.factual_targets == 32
    assert decomposition.legacy_expected_covered == 23
    assert decomposition.geometry_safe_expected_covered == 16
    assert decomposition.per_factual_transition_required is True
    assert decomposition.descriptive_shapley_allowed is True
    assert decomposition.causal_attribution_allowed is False

    payload = _payload()
    payload["coverage_transition_decomposition"][
        "causal_attribution_allowed"
    ] = True
    with pytest.raises(ValueError, match="causal_attribution_allowed"):
        SyntheticStateFailureProtocol.from_mapping(payload)


def test_failure_protocol_decision_language_preserves_evidence_limits() -> None:
    decision = load_synthetic_state_failure_protocol(
        _CONFIG
    ).decision_policy

    assert decision.execution_states == (
        "complete",
        "partial_inconclusive",
        "invalid",
    )
    assert decision.metric_states == (
        "strong",
        "not_strong",
        "inconclusive",
    )
    assert decision.block_states == (
        "strong_role_signal",
        "no_strong_role_signal_detected",
        "mixed_or_inconclusive",
    )
    assert "not-distribution-equality" in (
        decision.no_strong_signal_interpretation
    )
    assert "not-causal-proof" in decision.strong_signal_interpretation


def test_failure_protocol_receipt_contract_is_diagnostic_only() -> None:
    contract = load_synthetic_state_failure_protocol(
        _CONFIG
    ).receipt_contract
    assert contract.receipt_files[-1] == "diagnostic_decision.json"
    assert "frozen_feature_evidence.json" in contract.receipt_files
    assert "coverage_transition_decomposition.json" in contract.receipt_files
    assert contract.require_complete_marker is True
    assert contract.require_per_target_ledger is True
    assert contract.require_fold_fit_audit is True
    assert contract.require_two_run_byte_identity is True
    forbidden = ("train", "checkpoint", "transformation", "candidate_s")
    assert not any(
        fragment in name
        for name in contract.receipt_files
        for fragment in forbidden
    )


def test_failure_protocol_rejects_wrong_valid_authority_digest() -> None:
    payload = _payload()
    payload["authority_binding"]["p0_bc_p0_b_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the freeze"):
        SyntheticStateFailureProtocol.from_mapping(payload)


def test_failure_protocol_loader_rejects_duplicate_keys_and_symlink(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"x","schema_version":"y"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_synthetic_state_failure_protocol(duplicate)

    link = tmp_path / "config-link.json"
    link.symlink_to(_CONFIG)
    with pytest.raises(ValueError, match="symlink"):
        load_synthetic_state_failure_protocol(link)


def test_failure_protocol_rejects_any_semantic_mutation() -> None:
    original = _payload()
    mutations = [
        ("factor_taxonomy", "interpretation_limit", "we-prove-equality"),
        ("probe_freeze", "folds", 4),
        ("probe_freeze", "auc_effect_boundary", 0.8),
        (
            "coverage_transition_decomposition",
            "reference_radius_rule",
            "reuse-best-radius",
        ),
        (
            "decision_policy",
            "overall_scientific_gate",
            "pass-on-best-block",
        ),
    ]
    for section, field, replacement in mutations:
        payload = deepcopy(original)
        payload[section][field] = replacement
        with pytest.raises(ValueError):
            SyntheticStateFailureProtocol.from_mapping(payload)
