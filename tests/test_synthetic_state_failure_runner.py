from __future__ import annotations

import json
from pathlib import Path

import pytest

from cure_lite.cache.schema import file_sha256
from cure_lite.experiment.synthetic_state_failure_attribution import (
    COMMON_BLOCKS,
)
from cure_lite.experiment.geometry_safe_p0_bc_protocol import (
    load_geometry_safe_p0_bc_protocol,
)
from cure_lite.experiment.synthetic_state_failure_protocol import (
    load_synthetic_state_failure_protocol,
)
from tools.run_synthetic_state_failure_attribution import (
    SYNTHETIC_STATE_FAILURE_CONFIG_FILE_SHA256,
    _RECEIPT_NAMES,
    _decision_receipt,
    _drop_one_log_loss_summaries,
    _factual_signatures,
    _fixed_fit_failure_code,
    _inconclusive_probe,
    _load_authority,
    _mmd_descriptive_crossing,
    _predictive_signature_profile,
    _screen_blocks,
    _stratum_auc_states,
    _verify_projection_freeze,
    build_parser,
)


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "synthetic_state_failure_attribution_v1"
    / "config.json"
)
_P0_BC_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "geometry_safe_p0_bc_v1"
    / "config.json"
)
_RUN_ROOT = _ROOT / "runs" / "irstd1k_stage_a_seed42"
_R1 = _RUN_ROOT / "cure_lite_geometry_safe_p0_bc_v1_r1"
_R2 = _RUN_ROOT / "cure_lite_geometry_safe_p0_bc_v1_r2"
_LEGACY_P0_B = (
    _RUN_ROOT
    / "cure_lite_p0_v1_r3"
    / "receipts"
    / "p0_b_support.json"
)


def _authority_paths() -> dict[str, Path]:
    return {
        "p0_bc_config": _P0_BC_CONFIG,
        "p0_bc_r1_complete": _R1 / "COMPLETE.json",
        "p0_bc_r2_complete": _R2 / "COMPLETE.json",
        "p0_bc_population_receipt": (
            _R1 / "receipts" / "population_binding.json"
        ),
        "p0_bc_p0_b_receipt": _R1 / "receipts" / "p0_b_support.json",
        "p0_bc_p0_c_receipt": _R1 / "receipts" / "p0_c_screening.json",
        "p0_bc_decision_receipt": _R1 / "receipts" / "decision.json",
        "legacy_209_p0_b_receipt": _LEGACY_P0_B,
    }


def _oof(lower: float, upper: float) -> dict[str, object]:
    return {
        "estimands": {
            "group_balanced_oof_auc_bootstrap_lower": lower,
            "group_balanced_oof_auc_bootstrap_upper": upper,
        }
    }


def _support(
    *,
    observed: float,
    q95: float,
    reference: list[float],
) -> dict[str, object]:
    return {
        "mmd": {
            "observed_factual_vs_matched_legal": {
                "summary_quantile": observed,
            },
            "legal_vs_legal_reference": {
                "quantile": q95,
                "values": reference,
            },
        }
    }


def _screening_fixture() -> dict[str, dict[str, object]]:
    oof = {
        block: _oof(0.50, 0.65)
        for block in COMMON_BLOCKS
    }
    support = {
        block: _support(
            observed=0.0,
            q95=1.0,
            reference=[0.0] * 999,
        )
        for block in COMMON_BLOCKS
    }
    oof["G_full"] = _oof(0.80, 0.90)
    support["G_full"] = _support(
        observed=2.0,
        q95=1.0,
        reference=[0.0] * 999,
    )
    oof["P"] = _oof(0.65, 0.75)
    return _screen_blocks(oof, support, auc_boundary=0.70)


def _sensitivity_fixture(
    lower: float,
    upper: float,
) -> dict[str, object]:
    return {
        "results": {
            block: {
                "estimands": {
                    "group_balanced_oof_auc": (lower + upper) / 2.0,
                    "group_balanced_oof_auc_bootstrap_lower": lower,
                    "group_balanced_oof_auc_bootstrap_upper": upper,
                }
            }
            for block in COMMON_BLOCKS
        }
    }


def test_runner_config_digest_and_receipt_contract_are_frozen() -> None:
    config = load_synthetic_state_failure_protocol(_CONFIG)

    assert file_sha256(_CONFIG) == (
        SYNTHETIC_STATE_FAILURE_CONFIG_FILE_SHA256
    )
    assert tuple(config.receipt_contract.receipt_files) == _RECEIPT_NAMES
    assert len(_RECEIPT_NAMES) == 9
    assert len(set(_RECEIPT_NAMES)) == 9


def test_runner_cli_has_only_path_bindings_and_no_statistical_override() -> None:
    options = {
        action.dest
        for action in build_parser()._actions
        if action.dest != "help"
    }

    assert options == {
        "manifest",
        "state_index",
        "config",
        "p0_bc_config",
        "geometry_config",
        "geometry_catalog_receipt",
        "p0_a1_receipt",
        "eligible_view_receipt",
        "geometry_complete",
        "p0_v1_config",
        "p0_bc_r1_complete",
        "p0_bc_r2_complete",
        "p0_bc_population_receipt",
        "p0_bc_p0_b_receipt",
        "p0_bc_p0_c_receipt",
        "p0_bc_decision_receipt",
        "legacy_209_p0_b_receipt",
        "output",
    }
    forbidden = {
        "auc",
        "bootstrap",
        "component",
        "fold",
        "holm",
        "mmd",
        "quantile",
        "seed",
        "threshold",
    }
    assert not any(
        fragment in option
        for option in options
        for fragment in forbidden
    )


def test_runner_loads_exact_formal_authority_and_209_206_evidence() -> None:
    config = load_synthetic_state_failure_protocol(_CONFIG)
    p0_bc, payloads = _load_authority(config, _authority_paths())

    assert p0_bc.fingerprint == (
        config.authority_binding.p0_bc_config_fingerprint
    )
    assert (
        payloads["p0_bc_r1_complete"]
        == payloads["p0_bc_r2_complete"]
    )
    assert (
        payloads["legacy_209_p0_b_receipt"]["counts"][
            "decoder_visible_legal_targets"
        ]
        == 209
    )
    assert (
        payloads["p0_bc_p0_b_receipt"]["legacy_raw"]["counts"][
            "decoder_visible_legal_targets"
        ]
        == 206
    )
    assert payloads["p0_bc_decision_receipt"]["formal_gates"] == {
        "p0_a1": "pass",
        "p0_b": "fail",
        "p0_c": "fail",
        "p0_d": "not_evaluated",
    }


def test_runner_rejects_any_authority_file_change(
    tmp_path: Path,
) -> None:
    config = load_synthetic_state_failure_protocol(_CONFIG)
    paths = _authority_paths()
    payload = json.loads(
        paths["p0_bc_decision_receipt"].read_text(encoding="utf-8")
    )
    payload["next_route"] = "changed"
    changed = tmp_path / "decision.json"
    changed.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    paths["p0_bc_decision_receipt"] = changed

    with pytest.raises(
        RuntimeError,
        match="differs from the frozen authority binding",
    ):
        _load_authority(config, paths)


def test_mmd_screen_is_descriptive_q95_crossing_without_p_value() -> None:
    result = _mmd_descriptive_crossing(
        {
            "observed_factual_vs_matched_legal": {
                "summary_quantile": 2.0,
            },
            "legal_vs_legal_reference": {
                "quantile": 1.0,
                "values": [0.0, 1.0, 3.0],
            },
        }
    )

    assert result["observed_above_legal_reference_q95"] is True
    assert result["descriptive_crossing_state"] == "above_reference_q95"
    assert result["reference_replicates"] == 3
    assert result["statistical_significance_claimed"] is False
    assert result["inferential_test"] is None
    assert result["multiplicity_correction"] is None
    assert "inferential" in result["interpretation"]
    assert not any("holm" in key.lower() for key in result)

    with pytest.raises(RuntimeError, match="reference is empty"):
        _mmd_descriptive_crossing(
            {
                "observed_factual_vs_matched_legal": {
                    "summary_quantile": 2.0,
                },
                "legal_vs_legal_reference": {
                    "quantile": 1.0,
                    "values": [],
                },
            }
        )


def test_auc_is_three_valued_and_combined_with_descriptive_crossing() -> None:
    screening = _screening_fixture()

    assert screening["G_full"]["auc"]["state"] == "strong"
    assert screening["G_full"]["mmd_descriptive_crossing"][
        "descriptive_crossing_state"
    ] == "above_reference_q95"
    assert screening["G_full"]["block_state"] == "strong_role_signal"

    assert screening["W"]["auc"]["state"] == "not_strong"
    assert screening["W"]["mmd_descriptive_crossing"][
        "descriptive_crossing_state"
    ] == "not_above_reference_q95"
    assert (
        screening["W"]["block_state"]
        == "no_strong_role_signal_detected"
    )

    assert screening["P"]["auc"]["state"] == "inconclusive"
    assert screening["P"]["block_state"] == "mixed_or_inconclusive"


def test_diagnostic_decision_never_authorizes_a_next_mechanism() -> None:
    config = load_synthetic_state_failure_protocol(_CONFIG)
    screening = _screening_fixture()
    shared_states = _stratum_auc_states(
        _sensitivity_fixture(0.50, 0.65),
        auc_boundary=0.70,
    )
    exact_states = _stratum_auc_states(
        _sensitivity_fixture(0.65, 0.75),
        auc_boundary=0.70,
    )
    signature = _predictive_signature_profile(
        screening,
        shared_states,
        exact_states,
    )
    decision = _decision_receipt(
        config,
        screening,
        authority_fingerprint="a" * 64,
        state_contract_fingerprint="b" * 64,
        profile_fingerprint="c" * 64,
        composition_fingerprint="d" * 64,
        transition_fingerprint="e" * 64,
        signature_fingerprint="f" * 64,
        predictive_signature_profile=signature,
    )

    assert decision["scientific_gate"] is None
    assert decision["formal_p0_b"] == "fail"
    assert decision["formal_p0_c"] == "fail"
    assert decision["formal_p0_d"] == "not_evaluated"
    assert decision["transformation_constructed"] is False
    assert decision["candidate_s_constructed"] is False
    assert decision["p0_d_executed"] is False
    assert decision["training_performed"] is False
    assert decision["d_v_accessed"] is False
    assert decision["d_t_accessed"] is False
    assert decision["predictive_signature_profile"] == signature
    assert decision["next_stage"] == (
        "separate_hypothesis_review_required_before_any_"
        "transformation_proposal"
    )
    assert decision["authorizes_transformation_construction"] is False
    assert decision["authorizes_candidate_s_construction"] is False
    assert decision["authorizes_p0_d"] is False
    assert decision["authorizes_training"] is False
    assert decision["authorizes_d_v_evaluation"] is False
    assert decision["authorizes_full_cure"] is False
    assert decision["authorizes_backbone_integration"] is False
    assert decision["execution_state"] == "complete"
    assert decision["stratum_has_three_state_inconclusive_auc"] is True
    assert decision["computationally_inconclusive_probes"] == []

    partial = _decision_receipt(
        config,
        screening,
        authority_fingerprint="a" * 64,
        state_contract_fingerprint="b" * 64,
        profile_fingerprint="c" * 64,
        composition_fingerprint="d" * 64,
        transition_fingerprint="e" * 64,
        signature_fingerprint="f" * 64,
        predictive_signature_profile=signature,
        computationally_inconclusive_probes=("shared:P",),
    )
    assert partial["execution_state"] == "partial_inconclusive"
    assert partial["computationally_inconclusive_probes"] == ["shared:P"]


def test_stratum_auc_states_form_a_non_authorizing_signature() -> None:
    screening = _screening_fixture()
    shared = _stratum_auc_states(
        _sensitivity_fixture(0.80, 0.90),
        auc_boundary=0.70,
    )
    exact_source = _stratum_auc_states(
        _sensitivity_fixture(0.65, 0.75),
        auc_boundary=0.70,
    )
    profile = _predictive_signature_profile(
        screening,
        shared,
        exact_source,
    )

    assert all(row["state"] == "strong" for row in shared.values())
    assert all(
        row["state"] == "inconclusive" for row in exact_source.values()
    )
    assert profile["blocks"]["G_full"] == {
        "all_geometry_safe_population": "strong",
        "shared_manifest_groups": "strong",
        "selected_dual_role_sources_transductive_source_centered": (
            "inconclusive"
        ),
    }
    assert profile["authorizes_transformation_construction"] is False
    assert profile["authorizes_candidate_s_construction"] is False
    assert profile["authorizes_p0_d"] is False
    assert profile["authorizes_training"] is False
    assert profile["authorizes_d_v_evaluation"] is False
    assert profile["authorizes_full_cure"] is False


def test_frozen_fit_failure_becomes_inconclusive_without_refit() -> None:
    error = RuntimeError(
        "grouped logistic IRLS did not converge within the frozen limit"
    )
    assert _fixed_fit_failure_code(error) == (
        "frozen_logistic_irls_nonconvergence"
    )
    failure = _inconclusive_probe(
        block_or_probe="G_full",
        error=error,
    )
    sensitivity = _sensitivity_fixture(0.80, 0.90)
    sensitivity["results"]["G_full"] = failure

    states = _stratum_auc_states(
        sensitivity,
        auc_boundary=0.70,
    )

    assert states["G_full"]["state"] == "inconclusive"
    assert states["G_full"]["point"] is None
    assert states["G_full"]["failure_code"] == (
        "frozen_logistic_irls_nonconvergence"
    )
    assert states["G_full"]["statistical_override_applied"] is False
    assert failure["frozen_fit_retained"] is True
    assert failure["refit_with_modified_parameters"] is False
    assert failure["authorizes_training"] is False

    with pytest.raises(RuntimeError, match="unexpected failure"):
        _inconclusive_probe(
            block_or_probe="G_full",
            error=RuntimeError("unexpected failure"),
        )


def test_full_population_screen_preserves_fixed_fit_inconclusive() -> None:
    oof = {
        block: _oof(0.50, 0.65)
        for block in COMMON_BLOCKS
    }
    oof["F_local"] = _inconclusive_probe(
        block_or_probe="F_local",
        error=RuntimeError(
            "grouped logistic IRLS did not converge within the frozen limit"
        ),
    )
    support = {
        block: _support(
            observed=0.0,
            q95=1.0,
            reference=[0.0] * 1000,
        )
        for block in COMMON_BLOCKS
    }

    screening = _screen_blocks(oof, support, auc_boundary=0.70)

    assert screening["F_local"]["auc"]["state"] == "inconclusive"
    assert screening["F_local"]["auc"]["bootstrap_lower"] is None
    assert screening["F_local"]["auc"]["computational_failure_code"] == (
        "frozen_logistic_irls_nonconvergence"
    )
    assert (
        screening["F_local"]["block_state"]
        == "mixed_or_inconclusive"
    )


def test_drop_one_summary_is_fixed_paired_log_loss_not_selection() -> None:
    composites = {
        "decoder_input_probe_union": {
            "estimands": {
                "group_balanced_cross_fitted_log_loss": 0.40,
            }
        },
        "drop_F_local": {
            "estimands": {
                "group_balanced_cross_fitted_log_loss": 0.55,
            }
        },
        "drop_F_background_global": {
            "estimands": {
                "group_balanced_cross_fitted_log_loss": 0.35,
            }
        },
        "drop_O": {
            "estimands": {
                "group_balanced_cross_fitted_log_loss": 0.40,
            }
        },
    }

    result = _drop_one_log_loss_summaries(composites)

    assert result["summaries"]["drop_F_local"][
        "drop_loss_minus_union_loss"
    ] == pytest.approx(0.15)
    assert result["summaries"]["drop_F_background_global"][
        "drop_loss_minus_union_loss"
    ] == pytest.approx(-0.05)
    assert result["summaries"]["drop_O"][
        "drop_loss_minus_union_loss"
    ] == pytest.approx(0.0)
    assert result["selection_rule"] is None
    assert result["threshold"] is None
    assert result["winner_selected"] is False
    assert result["authorizes_transformation_construction"] is False
    assert result["authorizes_training"] is False

    composites["drop_F_local"] = _inconclusive_probe(
        block_or_probe="drop_F_local",
        error=RuntimeError(
            "grouped logistic IRLS did not converge within the frozen limit"
        ),
    )
    partial = _drop_one_log_loss_summaries(composites)
    assert partial["execution_status"] == "partial_inconclusive"
    assert partial["summaries"]["drop_F_local"][
        "drop_loss_minus_union_loss"
    ] is None
    assert partial["summaries"]["drop_F_local"]["execution_status"] == (
        "inconclusive"
    )


def test_projection_fit_populations_and_pca_residual_are_verified() -> None:
    config = load_synthetic_state_failure_protocol(_CONFIG)
    p0_bc = load_geometry_safe_p0_bc_protocol(_P0_BC_CONFIG)
    components = config.probe_freeze.feature_components
    raw_dimensions = {
        "G_full": 6,
        "W": 4,
        "P": 7,
        "F_local": 20,
        "F_background_global": 24,
        "O": 29,
    }

    def oof(blocks: tuple[str, ...]) -> dict[str, object]:
        dimensions = sum(
            components + 1
            if block in {"F_local", "F_background_global"}
            else raw_dimensions[block]
            for block in blocks
        )
        projections = {
            block: (
                {
                    "fit_role": "training-fold-legal-targets-only",
                    "components": components,
                }
                if block in {"F_local", "F_background_global"}
                else None
            )
            for block in blocks
        }
        return {
            "blocks": list(blocks),
            "folds": [
                {
                    "raw_dimensions_by_block": {
                        block: raw_dimensions[block] for block in blocks
                    },
                    "model_dimensions": dimensions,
                    "projection_fit_by_block": projections,
                }
            ],
        }

    block_oof = {
        block: oof((block,))
        for block in COMMON_BLOCKS
    }
    composites = {
        "decoder_input_probe_union": oof(
            ("F_local", "F_background_global", "O")
        ),
        "drop_F_local": oof(("F_background_global", "O")),
        "drop_F_background_global": oof(("F_local", "O")),
        "drop_O": oof(("F_local", "F_background_global")),
    }
    block_support: dict[str, dict[str, object]] = {}
    for block in COMMON_BLOCKS:
        if block in {"F_local", "F_background_global"}:
            block_support[block] = {
                "coverage_projection_fit": {
                    "fit_role": "legal-targets-only",
                    "fit_targets": 206,
                    "fit_groups": 145,
                    "components": components,
                },
                "mmd_projection_fit": {
                    "fit_role": "legal-targets-only",
                    "fit_groups": 131,
                    "components": components,
                },
            }
        else:
            block_support[block] = {
                "coverage_projection_fit": None,
                "mmd_projection_fit": None,
            }

    audit = _verify_projection_freeze(
        config,
        p0_bc,
        block_oof,
        block_support,
        composites,
    )

    assert audit["verified"] is True
    assert audit["feature_components"] == 6
    assert audit[
        "PCA_plus_residual_output_dimensions_per_feature_block"
    ] == 7
    assert audit["oof_feature_projection_fit_population"] == (
        "training-fold-legal-targets-only"
    )
    assert audit["coverage_feature_projection_fit_population"] == (
        "all-geometry-safe-legal-targets"
    )
    assert audit["mmd_feature_projection_fit_population"] == (
        "legal-exclusive-manifest-groups-only"
    )

    block_oof["F_local"] = _inconclusive_probe(
        block_or_probe="F_local",
        error=RuntimeError(
            "grouped logistic IRLS did not converge within the frozen limit"
        ),
    )
    skipped = _verify_projection_freeze(
        config,
        p0_bc,
        block_oof,
        block_support,
        composites,
    )
    assert skipped["verified_completed_oof_results"] is True
    assert "single-block:F_local" in skipped["inconclusive_oof_results"]

    block_support["F_local"]["mmd_projection_fit"]["fit_groups"] = 130
    with pytest.raises(RuntimeError, match="legal-exclusive groups"):
        _verify_projection_freeze(
            config,
            p0_bc,
            block_oof,
            block_support,
            composites,
        )


def test_factual_signatures_are_fixed_six_block_descriptions() -> None:
    config = load_synthetic_state_failure_protocol(_CONFIG)
    support: dict[str, dict[str, object]] = {}
    for block_index, block in enumerate(COMMON_BLOCKS):
        support[block] = {
            "coverage": {
                "factual_targets": [
                    {
                        "identity": [f"sample-{index:02d}", 1, None],
                        "group_id": f"group-{index % 24:02d}",
                        "covered": (index + block_index) % 2 == 0,
                    }
                    for index in range(32)
                ]
            }
        }

    receipt = _factual_signatures(
        config,
        support,
        profile_fingerprint="1" * 64,
    )

    assert receipt["factual_targets"] == 32
    assert receipt["block_order"] == list(COMMON_BLOCKS)
    assert len(receipt["rows"]) == 32
    assert sum(
        item["targets"] for item in receipt["signature_counts"]
    ) == 32
    assert all(len(row["signature"]) == 6 for row in receipt["rows"])
    assert receipt["authorizes_transformation"] is False
