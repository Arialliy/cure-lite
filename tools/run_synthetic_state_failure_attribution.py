#!/usr/bin/env python3
"""Run the create-only D_R synthetic-state failure-attribution protocol.

The command reconstructs the frozen 209-target and geometry-safe 206-target
populations, runs only predeclared descriptive probes, and writes auditable
receipts.  It never constructs a transformed state or candidate S, runs P0-D,
trains, calibrates, infers, reads D_V/D_T, or starts Full CURE.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import PIL
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.data import ManifestImageDataset  # noqa: E402
from cure_lite.experiment.cache_pipeline import load_d_r_cache_bundle  # noqa: E402
from cure_lite.experiment.coverage_transition import (  # noqa: E402
    build_coverage_transition,
)
from cure_lite.experiment.geometry_safe_catalog import (  # noqa: E402
    build_geometry_safe_catalog,
    build_geometry_safe_p0_view,
    build_p0_a1_receipt,
)
from cure_lite.experiment.geometry_safe_p0_bc_protocol import (  # noqa: E402
    GeometrySafeP0BCProtocol,
    load_geometry_safe_p0_bc_protocol,
)
from cure_lite.experiment.p0_support import _extract_targets  # noqa: E402
from cure_lite.experiment.synthetic_state_failure_attribution import (  # noqa: E402
    COMMON_BLOCKS,
    PopulationExpectation,
    SameSourceExpectation,
    SharedGroupExpectation,
    build_failure_attribution_population,
    exact_same_source_subset,
    run_block_coverage_mmd,
    run_block_only_group_oof,
    run_composite_group_oof,
    run_exact_same_source_sensitivity,
    run_shared_group_sensitivity,
    shared_manifest_group_subset,
    source_center_common_blocks,
)
from cure_lite.experiment.synthetic_state_failure_protocol import (  # noqa: E402
    SyntheticStateFailureProtocol,
    load_synthetic_state_failure_protocol,
)
from cure_lite.experiment.training_pipeline import (  # noqa: E402
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402
from tools.run_geometry_safe_p0_bc import (  # noqa: E402
    _canonical_existing_file,
    _fingerprinted,
    _load_and_verify_upstream,
    _prepare_output,
    _reconstructed_eligible_view_receipt,
    _strict_json,
    _verify_fingerprinted,
    _verify_input_binding,
    _verify_statistical_freeze,
    _write_new_json,
)


SYNTHETIC_STATE_FAILURE_RUN_SCHEMA = (
    "cure-lite-synthetic-state-failure-attribution-run-v1"
)
SYNTHETIC_STATE_FAILURE_CONFIG_FILE_SHA256 = (
    "8933113e745ab42119e90a0a3f2b4366290f38a6b523251d94d39bc5665e6161"
)
_ROOT = Path(__file__).resolve().parents[1]
_RECEIPT_NAMES = (
    "authority_binding.json",
    "population_factor_inventory.json",
    "state_contract_audit.json",
    "frozen_feature_evidence.json",
    "factor_probe_profile.json",
    "composition_strata.json",
    "coverage_transition_decomposition.json",
    "factual_miss_signatures.json",
    "diagnostic_decision.json",
)
_COMPOSITE_BLOCKS = {
    "decoder_input_probe_union": ("F_local", "F_background_global", "O"),
    "drop_F_local": ("F_background_global", "O"),
    "drop_F_background_global": ("F_local", "O"),
    "drop_O": ("F_local", "F_background_global"),
}


def _no_authority() -> dict[str, bool]:
    return {
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--p0-bc-config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--geometry-catalog-receipt", type=Path, required=True)
    parser.add_argument("--p0-a1-receipt", type=Path, required=True)
    parser.add_argument("--eligible-view-receipt", type=Path, required=True)
    parser.add_argument("--geometry-complete", type=Path, required=True)
    parser.add_argument("--p0-v1-config", type=Path, required=True)
    parser.add_argument("--p0-bc-r1-complete", type=Path, required=True)
    parser.add_argument("--p0-bc-r2-complete", type=Path, required=True)
    parser.add_argument("--p0-bc-population-receipt", type=Path, required=True)
    parser.add_argument("--p0-bc-p0-b-receipt", type=Path, required=True)
    parser.add_argument("--p0-bc-p0-c-receipt", type=Path, required=True)
    parser.add_argument("--p0-bc-decision-receipt", type=Path, required=True)
    parser.add_argument("--legacy-209-p0-b-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _canonical_inputs(args: argparse.Namespace) -> dict[str, Path]:
    names = {
        "manifest": "manifest",
        "state_index": "D_R state index",
        "config": "failure-attribution config",
        "p0_bc_config": "geometry-safe P0-B/C config",
        "geometry_config": "geometry-safe P0-A1 config",
        "geometry_catalog_receipt": "geometry catalog receipt",
        "p0_a1_receipt": "P0-A1 receipt",
        "eligible_view_receipt": "eligible-view receipt",
        "geometry_complete": "geometry COMPLETE",
        "p0_v1_config": "P0-v1 config",
        "p0_bc_r1_complete": "P0-B/C r1 COMPLETE",
        "p0_bc_r2_complete": "P0-B/C r2 COMPLETE",
        "p0_bc_population_receipt": "P0-B/C population receipt",
        "p0_bc_p0_b_receipt": "P0-B/C P0-B receipt",
        "p0_bc_p0_c_receipt": "P0-B/C P0-C receipt",
        "p0_bc_decision_receipt": "P0-B/C decision receipt",
        "legacy_209_p0_b_receipt": "legacy 209-target P0-B receipt",
    }
    return {
        field: _canonical_existing_file(getattr(args, field), name=name)
        for field, name in names.items()
    }


def _verify_file_digest(
    path: Path,
    expected: str,
    *,
    name: str,
) -> None:
    if file_sha256(path) != expected:
        raise RuntimeError(f"{name} differs from the frozen authority binding")


def _verify_embedded_fingerprint(
    payload: Mapping[str, Any],
    expected: str,
    *,
    name: str,
    field: str = "receipt_fingerprint",
) -> None:
    _verify_fingerprinted(payload, name=name, field=field)
    if payload.get(field) != expected:
        raise RuntimeError(f"{name} fingerprint differs from the freeze")


def _load_authority(
    config: SyntheticStateFailureProtocol,
    paths: Mapping[str, Path],
) -> tuple[
    GeometrySafeP0BCProtocol,
    dict[str, dict[str, Any]],
]:
    authority = config.authority_binding
    digest_bindings = {
        "p0_bc_config": authority.p0_bc_config_file_sha256,
        "p0_bc_r1_complete": authority.p0_bc_r1_complete_file_sha256,
        "p0_bc_r2_complete": authority.p0_bc_r2_complete_file_sha256,
        "p0_bc_population_receipt": authority.p0_bc_population_file_sha256,
        "p0_bc_p0_b_receipt": authority.p0_bc_p0_b_file_sha256,
        "p0_bc_p0_c_receipt": authority.p0_bc_p0_c_file_sha256,
        "p0_bc_decision_receipt": authority.p0_bc_decision_file_sha256,
        "legacy_209_p0_b_receipt": (
            authority.legacy_209_p0_b_file_sha256
        ),
    }
    for field, expected in digest_bindings.items():
        _verify_file_digest(paths[field], expected, name=field)

    p0_bc = load_geometry_safe_p0_bc_protocol(paths["p0_bc_config"])
    if p0_bc.fingerprint != authority.p0_bc_config_fingerprint:
        raise RuntimeError("P0-B/C config fingerprint differs from the freeze")
    if p0_bc.input_binding != config.input_binding:
        raise RuntimeError(
            "failure-attribution and P0-B/C input bindings differ"
        )
    payloads = {
        field: _strict_json(paths[field], name=field)
        for field in (
            "p0_bc_r1_complete",
            "p0_bc_r2_complete",
            "p0_bc_population_receipt",
            "p0_bc_p0_b_receipt",
            "p0_bc_p0_c_receipt",
            "p0_bc_decision_receipt",
            "legacy_209_p0_b_receipt",
        )
    }
    expected_fingerprints = {
        "p0_bc_r1_complete": (
            authority.p0_bc_r1_complete_fingerprint,
            "complete_fingerprint",
        ),
        "p0_bc_r2_complete": (
            authority.p0_bc_r2_complete_fingerprint,
            "complete_fingerprint",
        ),
        "p0_bc_population_receipt": (
            authority.p0_bc_population_receipt_fingerprint,
            "receipt_fingerprint",
        ),
        "p0_bc_p0_b_receipt": (
            authority.p0_bc_p0_b_receipt_fingerprint,
            "receipt_fingerprint",
        ),
        "p0_bc_p0_c_receipt": (
            authority.p0_bc_p0_c_receipt_fingerprint,
            "receipt_fingerprint",
        ),
        "p0_bc_decision_receipt": (
            authority.p0_bc_decision_receipt_fingerprint,
            "receipt_fingerprint",
        ),
        "legacy_209_p0_b_receipt": (
            authority.legacy_209_p0_b_receipt_fingerprint,
            "receipt_fingerprint",
        ),
    }
    for field, (expected, fingerprint_field) in expected_fingerprints.items():
        _verify_embedded_fingerprint(
            payloads[field],
            expected,
            name=field,
            field=fingerprint_field,
        )
    if payloads["p0_bc_r1_complete"] != payloads["p0_bc_r2_complete"]:
        raise RuntimeError("formal P0-B/C r1/r2 COMPLETE files differ")
    decision = payloads["p0_bc_decision_receipt"]
    if (
        decision.get("next_route") != "redesign_synthetic_state"
        or decision.get("formal_gates", {}).get("p0_b") != "fail"
        or decision.get("formal_gates", {}).get("p0_c") != "fail"
        or decision.get("authorizes_training") is not False
        or decision.get("authorizes_candidate_s_construction") is not False
    ):
        raise RuntimeError(
            "formal P0-B/C decision does not bind the failure-attribution route"
        )
    current_b = payloads["p0_bc_p0_b_receipt"]
    old_b = payloads["legacy_209_p0_b_receipt"]
    if (
        current_b.get("legacy_raw", {}).get("counts", {}).get(
            "decoder_visible_legal_targets"
        )
        != 206
        or current_b.get("legacy_raw", {}).get("coverage", {}).get(
            "decoder_joint", {}
        ).get("covered_factual_targets")
        != 16
        or old_b.get("counts", {}).get("decoder_visible_legal_targets")
        != 209
        or old_b.get("coverage", {}).get("decoder_joint", {}).get(
            "covered_factual_targets"
        )
        != 23
    ):
        raise RuntimeError("formal 209/206 P0-B evidence differs from the freeze")
    return p0_bc, payloads


def _mmd_descriptive_crossing(
    mmd: Mapping[str, Any],
) -> dict[str, object]:
    observed = float(
        mmd["observed_factual_vs_matched_legal"]["summary_quantile"]
    )
    reference = [
        float(value)
        for value in mmd["legal_vs_legal_reference"]["values"]
    ]
    q95 = float(mmd["legal_vs_legal_reference"]["quantile"])
    if not reference:
        raise RuntimeError("MMD legal reference is empty")
    crossing = observed > q95
    return {
        "observed_summary": observed,
        "legal_reference_q95": q95,
        "observed_above_legal_reference_q95": crossing,
        "descriptive_crossing_state": (
            "above_reference_q95"
            if crossing
            else "not_above_reference_q95"
        ),
        "reference_replicates": len(reference),
        "statistical_significance_claimed": False,
        "inferential_test": None,
        "multiplicity_correction": None,
        "interpretation": (
            "descriptive comparison of like frozen receipt summaries only; "
            "not an inferential significance test"
        ),
    }


def _screen_blocks(
    block_oof: Mapping[str, Mapping[str, Any]],
    block_support: Mapping[str, Mapping[str, Any]],
    *,
    auc_boundary: float,
) -> dict[str, dict[str, object]]:
    mmd_rows = {
        block: _mmd_descriptive_crossing(result["mmd"])
        for block, result in block_support.items()
    }
    result: dict[str, dict[str, object]] = {}
    for block in COMMON_BLOCKS:
        oof = block_oof[block]
        if oof.get("execution_status") == "inconclusive":
            lower = None
            upper = None
            auc_state = "inconclusive"
            auc_failure_code = oof["failure_code"]
        else:
            estimands = oof["estimands"]
            lower = float(
                estimands["group_balanced_oof_auc_bootstrap_lower"]
            )
            upper = float(
                estimands["group_balanced_oof_auc_bootstrap_upper"]
            )
            auc_state = (
                "strong"
                if lower > auc_boundary
                else "not_strong"
                if upper <= auc_boundary
                else "inconclusive"
            )
            auc_failure_code = None
        mmd = dict(mmd_rows[block])
        crossing = bool(mmd["observed_above_legal_reference_q95"])
        block_state = (
            "strong_role_signal"
            if auc_state == "strong" and crossing
            else "no_strong_role_signal_detected"
            if auc_state == "not_strong" and not crossing
            else "mixed_or_inconclusive"
        )
        result[block] = {
            "auc": {
                "boundary": auc_boundary,
                "bootstrap_lower": lower,
                "bootstrap_upper": upper,
                "state": auc_state,
                "computational_failure_code": auc_failure_code,
            },
            "mmd_descriptive_crossing": mmd,
            "block_state": block_state,
            "interpretation": (
                "AUC is a three-valued predictive screen and MMD is a "
                "descriptive q95 crossing only; neither is a causal effect "
                "or a full-distribution statement"
            ),
        }
    return result


def _auc_three_state(
    result: Mapping[str, Any],
    *,
    auc_boundary: float,
) -> dict[str, object]:
    estimands = result["estimands"]
    point = float(estimands["group_balanced_oof_auc"])
    lower = float(estimands["group_balanced_oof_auc_bootstrap_lower"])
    upper = float(estimands["group_balanced_oof_auc_bootstrap_upper"])
    state = (
        "strong"
        if lower > auc_boundary
        else "not_strong"
        if upper <= auc_boundary
        else "inconclusive"
    )
    return {
        "boundary": auc_boundary,
        "point": point,
        "bootstrap_lower": lower,
        "bootstrap_upper": upper,
        "state": state,
    }


def _stratum_auc_states(
    sensitivity: Mapping[str, Any],
    *,
    auc_boundary: float,
) -> dict[str, dict[str, object]]:
    results = sensitivity["results"]
    if set(results) != set(COMMON_BLOCKS):
        raise RuntimeError("stratum AUC results do not contain the six blocks")
    states: dict[str, dict[str, object]] = {}
    for block in COMMON_BLOCKS:
        result = results[block]
        if result.get("execution_status") == "inconclusive":
            states[block] = {
                "boundary": auc_boundary,
                "point": None,
                "bootstrap_lower": None,
                "bootstrap_upper": None,
                "state": "inconclusive",
                "failure_code": result["failure_code"],
                "statistical_override_applied": False,
            }
        else:
            states[block] = _auc_three_state(
                result,
                auc_boundary=auc_boundary,
            )
    return states


def _fixed_fit_failure_code(error: RuntimeError) -> str | None:
    messages = {
        "grouped logistic IRLS did not converge within the frozen limit": (
            "frozen_logistic_irls_nonconvergence"
        ),
        "all grouped AUC bootstrap replicates were uninformative": (
            "frozen_group_bootstrap_uninformative"
        ),
        "group OOF produced an empty fold": "frozen_group_oof_empty_fold",
        "feature projection fold has no legal training targets": (
            "frozen_feature_projection_has_no_legal_training_target"
        ),
        "block OOF requires both target roles": (
            "frozen_probe_population_lacks_a_role"
        ),
        "block OOF left a score undefined": (
            "frozen_group_oof_score_undefined"
        ),
    }
    return messages.get(str(error))


def _inconclusive_probe(
    *,
    block_or_probe: str,
    error: RuntimeError,
) -> dict[str, object]:
    code = _fixed_fit_failure_code(error)
    if code is None:
        raise error
    return _fingerprinted(
        {
            "schema_version": "cure-lite-fixed-probe-inconclusive-v1",
            "split": "D_R",
            "block_or_probe": block_or_probe,
            "execution_status": "inconclusive",
            "failure_code": code,
            "failure_message": str(error),
            "frozen_fit_retained": True,
            "refit_with_modified_parameters": False,
            "statistical_override_applied": False,
            "estimand_available": False,
            "interpretation": (
                "the frozen probe was not numerically evaluable; this is not "
                "evidence for or against role separability"
            ),
            **_no_authority(),
        }
    )


def _safe_block_results(
    records: Sequence[Any],
    *,
    separability: Any,
    feature_components: int,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for block in COMMON_BLOCKS:
        try:
            results[block] = run_block_only_group_oof(
                records,
                block=block,
                separability=separability,
                feature_components=feature_components,
            )
        except RuntimeError as error:
            results[block] = _inconclusive_probe(
                block_or_probe=block,
                error=error,
            )
    return results


def _safe_shared_group_sensitivity(
    records: Sequence[Any],
    *,
    separability: Any,
    expectation: SharedGroupExpectation,
    feature_components: int,
) -> dict[str, object]:
    try:
        return run_shared_group_sensitivity(
            records,
            blocks=COMMON_BLOCKS,
            separability=separability,
            expectation=expectation,
            feature_components=feature_components,
        )
    except RuntimeError as error:
        code = _fixed_fit_failure_code(error)
        if code is None:
            raise
        subset = shared_manifest_group_subset(
            records,
            expectation=expectation,
        )
        results = _safe_block_results(
            subset,
            separability=separability,
            feature_components=feature_components,
        )
        failures = [
            block
            for block in COMMON_BLOCKS
            if results[block].get("execution_status") == "inconclusive"
        ]
        return _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-failure-attribution-shared-group-partial-v1"
                ),
                "split": "D_R",
                "stratum": "shared-manifest-group",
                "execution_status": "partial_inconclusive",
                "population": {
                    "groups": expectation.groups,
                    "factual_targets": expectation.factual_targets,
                    "legal_targets": expectation.legal_targets,
                },
                "initial_full_core_api_failure_code": code,
                "inconclusive_blocks": failures,
                "source_centered": False,
                "results": results,
                "statistical_override_applied": False,
                "interpretation": (
                    "fixed per-block continuation after a frozen-fit failure; "
                    "no parameter was changed and failed blocks remain "
                    "inconclusive"
                ),
                **_no_authority(),
            }
        )


def _safe_exact_source_sensitivity(
    records: Sequence[Any],
    *,
    separability: Any,
    expectation: SameSourceExpectation,
    feature_components: int,
) -> dict[str, object]:
    try:
        return run_exact_same_source_sensitivity(
            records,
            blocks=COMMON_BLOCKS,
            separability=separability,
            expectation=expectation,
            feature_components=feature_components,
        )
    except RuntimeError as error:
        code = _fixed_fit_failure_code(error)
        if code is None:
            raise
        subset = source_center_common_blocks(
            exact_same_source_subset(records, expectation=expectation)
        )
        results = _safe_block_results(
            subset,
            separability=separability,
            feature_components=feature_components,
        )
        failures = [
            block
            for block in COMMON_BLOCKS
            if results[block].get("execution_status") == "inconclusive"
        ]
        return _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-failure-attribution-same-source-partial-v1"
                ),
                "split": "D_R",
                "stratum": "exact-dual-role-source",
                "execution_status": "partial_inconclusive",
                "population": {
                    "sources": expectation.sources,
                    "factual_targets": expectation.factual_targets,
                    "legal_targets": expectation.legal_targets,
                },
                "initial_full_core_api_failure_code": code,
                "inconclusive_blocks": failures,
                "source_centering": "label-blind-within-sample-mean-v1",
                "source_centering_scope": (
                    "selected-overlap-transductive-sensitivity-not-source-"
                    "elimination"
                ),
                "results": results,
                "statistical_override_applied": False,
                "interpretation": (
                    "fixed per-block continuation on the selected-overlap "
                    "transductive sensitivity after a frozen-fit failure; no "
                    "parameter was changed and failed blocks remain "
                    "inconclusive"
                ),
                **_no_authority(),
            }
        )


def _safe_composite_probe(
    records: Sequence[Any],
    *,
    blocks: Sequence[str],
    name: str,
    separability: Any,
    feature_components: int,
) -> dict[str, object]:
    try:
        return run_composite_group_oof(
            records,
            blocks=blocks,
            separability=separability,
            feature_components=feature_components,
        )
    except RuntimeError as error:
        return _inconclusive_probe(
            block_or_probe=name,
            error=error,
        )


def _predictive_signature_profile(
    screening: Mapping[str, Mapping[str, Any]],
    shared_states: Mapping[str, Mapping[str, Any]],
    exact_source_states: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    return {
        "state_order": [
            "all_geometry_safe_population",
            "shared_manifest_groups",
            "selected_dual_role_sources_transductive_source_centered",
        ],
        "blocks": {
            block: {
                "all_geometry_safe_population": screening[block]["auc"][
                    "state"
                ],
                "shared_manifest_groups": shared_states[block]["state"],
                "selected_dual_role_sources_transductive_source_centered": (
                    exact_source_states[block]["state"]
                ),
            }
            for block in COMMON_BLOCKS
        },
        "interpretation": (
            "three-valued predictive AUC signatures across frozen strata; "
            "the source-centered stratum is a selected-overlap transductive "
            "sensitivity and does not eliminate source effects; persistence "
            "or attenuation is descriptive, non-causal, and does not "
            "authorize a state transformation"
        ),
        **_no_authority(),
    }


def _drop_one_log_loss_summaries(
    composites: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    if set(composites) != set(_COMPOSITE_BLOCKS):
        raise RuntimeError("fixed decoder-input composites differ from freeze")
    field = "group_balanced_cross_fitted_log_loss"
    union_result = composites["decoder_input_probe_union"]
    union_inconclusive = (
        union_result.get("execution_status") == "inconclusive"
    )
    union = (
        None
        if union_inconclusive
        else float(union_result["estimands"][field])
    )
    rows: dict[str, dict[str, object]] = {}
    for name in ("drop_F_local", "drop_F_background_global", "drop_O"):
        drop_result = composites[name]
        drop_inconclusive = (
            drop_result.get("execution_status") == "inconclusive"
        )
        loss = (
            None
            if drop_inconclusive
            else float(drop_result["estimands"][field])
        )
        estimand_available = union is not None and loss is not None
        rows[name] = {
            "dropped_block": name.removeprefix("drop_"),
            "decoder_input_probe_union_log_loss": union,
            "drop_one_log_loss": loss,
            "drop_loss_minus_union_loss": (
                loss - union if estimand_available else None
            ),
            "estimand_name": (
                "conditional_predictive_cross_fitted_log_loss_difference"
            ),
            "execution_status": (
                "complete" if estimand_available else "inconclusive"
            ),
            "failure_codes": [
                result["failure_code"]
                for result in (union_result, drop_result)
                if result.get("execution_status") == "inconclusive"
            ],
        }
    return {
        "paired_by_identical_oof_population_and_fold_assignment": True,
        "execution_status": (
            "partial_inconclusive"
            if union_inconclusive
            or any(
                item.get("execution_status") == "inconclusive"
                for name, item in composites.items()
                if name != "decoder_input_probe_union"
            )
            else "complete"
        ),
        "summaries": rows,
        "selection_rule": None,
        "threshold": None,
        "winner_selected": False,
        "interpretation": (
            "fixed conditional predictive log-loss point differences only; "
            "the union is a summary probe rather than the complete decoder "
            "input, and the differences are not causal ablations or a "
            "transformation ranking"
        ),
        **_no_authority(),
    }


def _verify_projection_freeze(
    config: SyntheticStateFailureProtocol,
    p0_bc: GeometrySafeP0BCProtocol,
    block_oof: Mapping[str, Mapping[str, Any]],
    block_support: Mapping[str, Mapping[str, Any]],
    composites: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    components = config.probe_freeze.feature_components
    if p0_bc.overlap.joint_feature_components != components:
        raise RuntimeError(
            "P0-B/C feature-component count differs from attribution freeze"
        )

    def verify_oof(
        result: Mapping[str, Any],
        blocks: Sequence[str],
        *,
        name: str,
    ) -> bool:
        if result.get("execution_status") == "inconclusive":
            return False
        expected_dimensions = sum(
            components + 1
            if block in {"F_local", "F_background_global"}
            else int(result["folds"][0]["raw_dimensions_by_block"][block])
            for block in blocks
        )
        for fold in result["folds"]:
            if int(fold["model_dimensions"]) != expected_dimensions:
                raise RuntimeError(
                    f"{name} does not preserve PCA-plus-residual dimensions"
                )
            projections = fold["projection_fit_by_block"]
            for block in blocks:
                projection = projections[block]
                if block in {"F_local", "F_background_global"}:
                    if (
                        projection is None
                        or projection.get("fit_role")
                        != config.probe_freeze.oof_feature_projection_fit_population
                        or projection.get("components") != components
                    ):
                        raise RuntimeError(
                            f"{name} feature projection differs from freeze"
                        )
                elif projection is not None:
                    raise RuntimeError(
                        f"{name} projected a non-feature block"
                    )
        return True

    completed_oof: list[str] = []
    inconclusive_oof: list[str] = []
    for block in COMMON_BLOCKS:
        completed = verify_oof(
            block_oof[block],
            (block,),
            name=f"single-block {block}",
        )
        (completed_oof if completed else inconclusive_oof).append(
            f"single-block:{block}"
        )
    for name, result in composites.items():
        completed = verify_oof(
            result,
            _COMPOSITE_BLOCKS[name],
            name=name,
        )
        (completed_oof if completed else inconclusive_oof).append(
            f"composite:{name}"
        )

    population = config.population_binding
    for block in ("F_local", "F_background_global"):
        coverage = block_support[block]["coverage_projection_fit"]
        mmd = block_support[block]["mmd_projection_fit"]
        if (
            coverage is None
            or coverage.get("fit_role") != "legal-targets-only"
            or coverage.get("components") != components
            or coverage.get("fit_targets")
            != population.geometry_safe_legal_targets
            or coverage.get("fit_groups")
            != population.geometry_safe_legal_groups
        ):
            raise RuntimeError(
                f"{block} coverage projection is not fit on all legal targets"
            )
        if (
            mmd is None
            or mmd.get("fit_role") != "legal-targets-only"
            or mmd.get("components") != components
            or mmd.get("fit_groups") != population.legal_exclusive_groups
        ):
            raise RuntimeError(
                f"{block} MMD projection is not fit on legal-exclusive groups"
            )
    for block in ("G_full", "W", "P", "O"):
        if (
            block_support[block]["coverage_projection_fit"] is not None
            or block_support[block]["mmd_projection_fit"] is not None
        ):
            raise RuntimeError(f"{block} unexpectedly uses PCA projection")

    return {
        "verified": True,
        "verified_completed_oof_results": True,
        "completed_oof_results": completed_oof,
        "inconclusive_oof_results": inconclusive_oof,
        "inconclusive_results_have_no_projection_to_verify": True,
        "feature_components": components,
        "feature_projection_rule": (
            config.probe_freeze.feature_projection_rule
        ),
        "oof_feature_projection_fit_population": (
            config.probe_freeze.oof_feature_projection_fit_population
        ),
        "coverage_feature_projection_fit_population": (
            config.probe_freeze.coverage_feature_projection_fit_population
        ),
        "mmd_feature_projection_fit_population": (
            config.probe_freeze.mmd_feature_projection_fit_population
        ),
        "oof_fit_rule": config.probe_freeze.oof_fit_rule,
        "PCA_plus_residual_output_dimensions_per_feature_block": (
            components + 1
        ),
        **_no_authority(),
    }


def _state_contract_receipt(
    config: SyntheticStateFailureProtocol,
    population: Any,
    authority_fingerprint: str,
) -> dict[str, object]:
    rows = [
        item.canonical_payload()
        for item in population.legal_occupancy_ledger
    ]
    all_valid = all(
        bool(row["deletion_equals_frozen_pred_component"])
        and bool(row["source_feature_is_synthetic_feature"])
        and row["source_feature_fingerprint"]
        == row["synthetic_feature_fingerprint"]
        and row["classifier_eligible"] is False
        for row in rows
    )
    if len(rows) != config.population_binding.geometry_safe_legal_targets:
        raise RuntimeError("state-contract ledger count differs from the freeze")
    if not all_valid:
        raise RuntimeError("a frozen synthetic-state invariant failed")
    return _fingerprinted(
        {
            "schema_version": "cure-lite-failure-state-contract-audit-v1",
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "authority_binding_receipt_fingerprint": authority_fingerprint,
            "status": "complete",
            "invariants_pass": True,
            "legal_targets": len(rows),
            "classifier_eligible_ledger_rows": 0,
            "rows": rows,
            "interpretation": (
                "exact state-construction invariants; not a statistical "
                "common-support conclusion"
            ),
            "authorizes_transformation": False,
            **_no_authority(),
        }
    )


def _frozen_feature_receipt(
    config: SyntheticStateFailureProtocol,
    population: Any,
    state_contract_fingerprint: str,
) -> dict[str, object]:
    records = {
        item.identity: item
        for item in population.common_records
        if item.role == "legal"
    }
    rows: list[dict[str, object]] = []
    for ledger in population.legal_occupancy_ledger:
        record = records[ledger.identity]
        rows.append(
            {
                "identity": list(ledger.identity),
                "sample_id": record.sample_id,
                "group_id": record.group_id,
                "source_feature_fingerprint": (
                    ledger.source_feature_fingerprint
                ),
                "synthetic_feature_fingerprint": (
                    ledger.synthetic_feature_fingerprint
                ),
                "exact_feature_identity": (
                    ledger.source_feature_fingerprint
                    == ledger.synthetic_feature_fingerprint
                ),
                "F_local_fingerprint": stable_fingerprint(
                    [float(value) for value in record.F_local.tolist()]
                ),
                "F_background_global_fingerprint": stable_fingerprint(
                    [
                        float(value)
                        for value in record.F_background_global.tolist()
                    ]
                ),
                "P_values": [
                    float(value) for value in record.P.tolist()
                ],
                "classifier_eligible": False,
            }
        )
    if not all(bool(row["exact_feature_identity"]) for row in rows):
        raise RuntimeError("a legal synthetic feature differs from frozen Base")
    return _fingerprinted(
        {
            "schema_version": "cure-lite-frozen-feature-evidence-v1",
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "state_contract_receipt_fingerprint": state_contract_fingerprint,
            "legal_targets": len(rows),
            "exact_feature_identity_targets": len(rows),
            "rows": rows,
            "interpretation": (
                "occupancy deletion preserves frozen feature exactly; "
                "whether its summaries predict role is evaluated separately"
            ),
            "authorizes_feature_transformation": False,
            **_no_authority(),
        }
    )


def _factual_signatures(
    config: SyntheticStateFailureProtocol,
    block_support: Mapping[str, Mapping[str, Any]],
    profile_fingerprint: str,
) -> dict[str, object]:
    by_identity: dict[tuple[str, int], dict[str, bool]] = {}
    groups: dict[tuple[str, int], str] = {}
    for block in COMMON_BLOCKS:
        rows = block_support[block]["coverage"]["factual_targets"]
        for row in rows:
            identity = (str(row["identity"][0]), int(row["identity"][1]))
            by_identity.setdefault(identity, {})[block] = bool(row["covered"])
            groups[identity] = str(row["group_id"])
    if len(by_identity) != config.population_binding.factual_targets:
        raise RuntimeError("factual signature count differs from the freeze")
    rows = []
    signature_counts: dict[str, int] = {}
    for identity in sorted(by_identity):
        inside = by_identity[identity]
        signature = "".join(
            "1" if inside[block] else "0" for block in COMMON_BLOCKS
        )
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        rows.append(
            {
                "identity": [identity[0], identity[1], None],
                "group_id": groups[identity],
                "block_order": list(COMMON_BLOCKS),
                "inside_legal_reference": {
                    block: inside[block] for block in COMMON_BLOCKS
                },
                "signature": signature,
            }
        )
    return _fingerprinted(
        {
            "schema_version": "cure-lite-factual-miss-signatures-v1",
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "factor_probe_profile_receipt_fingerprint": profile_fingerprint,
            "factual_targets": len(rows),
            "block_order": list(COMMON_BLOCKS),
            "rows": rows,
            "signature_counts": [
                {"signature": key, "targets": signature_counts[key]}
                for key in sorted(signature_counts)
            ],
            "interpretation": (
                "fixed support signatures, not learned clusters and not a "
                "descriptor-selection mechanism"
            ),
            "authorizes_transformation": False,
            **_no_authority(),
        }
    )


def _decision_receipt(
    config: SyntheticStateFailureProtocol,
    screening: Mapping[str, Mapping[str, Any]],
    *,
    authority_fingerprint: str,
    state_contract_fingerprint: str,
    profile_fingerprint: str,
    composition_fingerprint: str,
    transition_fingerprint: str,
    signature_fingerprint: str,
    predictive_signature_profile: Mapping[str, Any],
    computationally_inconclusive_probes: Sequence[str] = (),
) -> dict[str, object]:
    strong = [
        block
        for block in COMMON_BLOCKS
        if screening[block]["block_state"] == "strong_role_signal"
    ]
    mixed = [
        block
        for block in COMMON_BLOCKS
        if screening[block]["block_state"] == "mixed_or_inconclusive"
    ]
    stratum_three_state_inconclusive = any(
        state == "inconclusive"
        for row in predictive_signature_profile["blocks"].values()
        for state in row.values()
    )
    computational_failures = sorted(
        set(str(value) for value in computationally_inconclusive_probes)
    )
    return _fingerprinted(
        {
            "schema_version": "cure-lite-failure-attribution-decision-v1",
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "execution_state": (
                "partial_inconclusive"
                if computational_failures
                else "complete"
            ),
            "scientific_gate": None,
            "formal_p0_b": "fail",
            "formal_p0_c": "fail",
            "formal_p0_d": "not_evaluated",
            "strong_role_signal_blocks": strong,
            "mixed_or_inconclusive_blocks": mixed,
            "stratum_has_three_state_inconclusive_auc": (
                stratum_three_state_inconclusive
            ),
            "computationally_inconclusive_probes": computational_failures,
            "factor_states": {
                block: screening[block]["block_state"]
                for block in COMMON_BLOCKS
            },
            "predictive_signature_profile": dict(
                predictive_signature_profile
            ),
            "input_receipt_fingerprints": {
                "authority_binding": authority_fingerprint,
                "state_contract_audit": state_contract_fingerprint,
                "factor_probe_profile": profile_fingerprint,
                "composition_strata": composition_fingerprint,
                "coverage_transition_decomposition": transition_fingerprint,
                "factual_miss_signatures": signature_fingerprint,
            },
            "interpretation": (
                "descriptive predictive attribution only; no block is an "
                "independent causal effect and no low-dimensional result "
                "proves equality or inequality of complete state distributions"
            ),
            "next_stage": (
                "separate_hypothesis_review_required_before_any_"
                "transformation_proposal"
            ),
            "transformation_constructed": False,
            "candidate_s_constructed": False,
            "p0_d_executed": False,
            "training_performed": False,
            "d_v_accessed": False,
            "d_t_accessed": False,
            **_no_authority(),
        }
    )


def _implementation_binding() -> dict[str, str]:
    files = (
        _ROOT / "tools" / "run_synthetic_state_failure_attribution.py",
        _ROOT / "tools" / "run_geometry_safe_p0_bc.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "synthetic_state_failure_protocol.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "synthetic_state_failure_attribution.py",
        _ROOT / "cure_lite" / "experiment" / "coverage_transition.py",
        _ROOT / "cure_lite" / "experiment" / "p0_support.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "geometry_safe_catalog.py",
        _ROOT / "cure_lite" / "experiment" / "training_pipeline.py",
    )
    return {
        str(path.relative_to(_ROOT)): file_sha256(path) for path in files
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = _canonical_inputs(args)
    output = _prepare_output(args.output)
    _verify_file_digest(
        paths["config"],
        SYNTHETIC_STATE_FAILURE_CONFIG_FILE_SHA256,
        name="failure-attribution config",
    )
    config = load_synthetic_state_failure_protocol(paths["config"])
    p0_bc, formal = _load_authority(config, paths)
    _verify_statistical_freeze(p0_bc, paths["p0_v1_config"])
    (
        geometry_protocol,
        upstream_geometry_catalog,
        upstream_p0_a1,
        upstream_eligible_view,
        _,
    ) = _load_and_verify_upstream(
        p0_bc,
        geometry_config_path=paths["geometry_config"],
        geometry_catalog_path=paths["geometry_catalog_receipt"],
        p0_a1_path=paths["p0_a1_receipt"],
        eligible_view_path=paths["eligible_view_receipt"],
        geometry_complete_path=paths["geometry_complete"],
    )
    manifest = load_and_validate_manifest(paths["manifest"])
    if manifest.dataset != config.dataset:
        raise RuntimeError("manifest dataset differs from protocol")
    state_index = _strict_json(paths["state_index"], name="D_R state index")
    preprocess = _verify_input_binding(
        p0_bc,
        geometry_protocol,
        paths["manifest"],
        paths["state_index"],
        state_index,
    )
    dataset = ManifestImageDataset(
        manifest,
        "D_R",
        preprocess,
        manifest_path=paths["manifest"],
    )
    bundle = load_d_r_cache_bundle(
        paths["state_index"],
        dataset,
        expected_base_fingerprint=config.input_binding.base_fingerprint,
    )
    sources = tuple(
        CachedTrainingSource(
            row.sample_id,
            row.base_output.feature,
            row.base_output.probability,
            row.state,
        )
        for row in bundle.rows
    )
    legacy = prepare_training_catalog(
        sources,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )
    geometry = build_geometry_safe_catalog(
        bundle,
        legacy,
        manifest,
        geometry_protocol,
    )
    reconstructed_geometry = _fingerprinted(geometry.canonical_payload())
    if reconstructed_geometry != upstream_geometry_catalog:
        raise RuntimeError("reconstructed geometry catalog differs from A1")
    reconstructed_a1 = _fingerprinted(
        build_p0_a1_receipt(
            geometry,
            geometry_protocol,
            a0_receipt_fingerprint=upstream_p0_a1[
                "a0_receipt_fingerprint"
            ],
        )
    )
    if reconstructed_a1 != upstream_p0_a1:
        raise RuntimeError("reconstructed P0-A1 receipt differs from authority")
    view = build_geometry_safe_p0_view(legacy, geometry)
    reconstructed_view = _reconstructed_eligible_view_receipt(
        geometry,
        view,
        config.authority_binding.eligible_catalog_fingerprint,
    )
    if reconstructed_view != upstream_eligible_view:
        raise RuntimeError("reconstructed eligible view differs from authority")

    immutable = {str(path): file_sha256(path) for path in paths.values()}
    implementation = _implementation_binding()
    authority_receipt = _fingerprinted(
        {
            "schema_version": "cure-lite-failure-authority-binding-v1",
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "config_fingerprint": config.fingerprint,
            "config_file_sha256": file_sha256(paths["config"]),
            "authority_binding": config.canonical_payload()[
                "authority_binding"
            ],
            "input_binding": config.canonical_payload()["input_binding"],
            "formal_p0_bc_r1_r2_byte_identical": (
                formal["p0_bc_r1_complete"]
                == formal["p0_bc_r2_complete"]
            ),
            "formal_p0_bc_replay_semantics": (
                "same-input-filesystem-replays-not-statistically-"
                "independent-repeats"
            ),
            "formal_decision": {
                "p0_b": "fail",
                "p0_c": "fail",
                "candidate_s_authorized": False,
                "training_authorized": False,
            },
            "verified": True,
            **_no_authority(),
        }
    )

    population = build_failure_attribution_population(
        bundle,
        view,
        manifest,
        p0_bc.overlap,
        expectation=PopulationExpectation(
            factual_targets=config.population_binding.factual_targets,
            legal_targets=(
                config.population_binding.geometry_safe_legal_targets
            ),
        ),
    )
    inventory = _fingerprinted(
        {
            "schema_version": "cure-lite-failure-population-inventory-v1",
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "authority_binding_receipt_fingerprint": (
                authority_receipt["receipt_fingerprint"]
            ),
            "population_binding": config.canonical_payload()[
                "population_binding"
            ],
            "factor_taxonomy": config.canonical_payload()[
                "factor_taxonomy"
            ],
            "population": population.canonical_payload(),
            "population_fingerprint": population.fingerprint,
            "authorizes_transformation": False,
            **_no_authority(),
        }
    )
    state_contract = _state_contract_receipt(
        config,
        population,
        str(authority_receipt["receipt_fingerprint"]),
    )
    feature_evidence = _frozen_feature_receipt(
        config,
        population,
        str(state_contract["receipt_fingerprint"]),
    )

    records = population.common_records
    feature_components = config.probe_freeze.feature_components
    if p0_bc.overlap.joint_feature_components != feature_components:
        raise RuntimeError(
            "P0-B/C feature-component count differs from attribution freeze"
        )
    block_oof = _safe_block_results(
        records,
        separability=p0_bc.separability,
        feature_components=feature_components,
    )
    block_support = {
        block: run_block_coverage_mmd(
            records,
            block=block,
            overlap=p0_bc.overlap,
            separability=p0_bc.separability,
            feature_components=feature_components,
        )
        for block in COMMON_BLOCKS
    }
    screening = _screen_blocks(
        block_oof,
        block_support,
        auc_boundary=config.probe_freeze.auc_effect_boundary,
    )
    composites = {
        name: _safe_composite_probe(
            records,
            blocks=blocks,
            name=name,
            separability=p0_bc.separability,
            feature_components=feature_components,
        )
        for name, blocks in _COMPOSITE_BLOCKS.items()
    }
    projection_fit_audit = _verify_projection_freeze(
        config,
        p0_bc,
        block_oof,
        block_support,
        composites,
    )
    profile = _fingerprinted(
        {
            "schema_version": "cure-lite-failure-factor-probe-profile-v1",
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "population_factor_inventory_receipt_fingerprint": (
                inventory["receipt_fingerprint"]
            ),
            "probe_freeze": config.canonical_payload()["probe_freeze"],
            "block_oof": block_oof,
            "block_support_mmd": block_support,
            "fixed_composite_oof": composites,
            "projection_fit_audit": projection_fit_audit,
            "fixed_drop_one_paired_point_summaries": (
                _drop_one_log_loss_summaries(composites)
            ),
            "screening": screening,
            "descriptive_mmd_q95_crossing_blocks": list(COMMON_BLOCKS),
            "interpretation": (
                "fixed predictive probes only; decoder_input_probe_union is "
                "a low-dimensional summary probe rather than the complete "
                "decoder input, no best descriptor is selected, and no MMD "
                "significance or multiplicity claim is made"
            ),
            "authorizes_transformation": False,
            "authorizes_candidate_s": False,
            **_no_authority(),
        }
    )

    shared_expectation = SharedGroupExpectation(
        groups=config.population_binding.role_overlap_groups,
        factual_targets=(
            config.population_binding.role_overlap_factual_targets
        ),
        legal_targets=config.population_binding.role_overlap_legal_targets,
    )
    source_expectation = SameSourceExpectation(
        sources=config.population_binding.dual_role_source_images,
        factual_targets=(
            config.population_binding.dual_role_source_factual_targets
        ),
        legal_targets=(
            config.population_binding.dual_role_source_legal_targets
        ),
    )
    shared = _safe_shared_group_sensitivity(
        records,
        separability=p0_bc.separability,
        expectation=shared_expectation,
        feature_components=feature_components,
    )
    exact_source = _safe_exact_source_sensitivity(
        records,
        separability=p0_bc.separability,
        expectation=source_expectation,
        feature_components=feature_components,
    )
    shared_records = shared_manifest_group_subset(
        records,
        expectation=shared_expectation,
    )
    source_records = source_center_common_blocks(
        exact_same_source_subset(records, expectation=source_expectation)
    )
    shared_auc_states = _stratum_auc_states(
        shared,
        auc_boundary=config.probe_freeze.auc_effect_boundary,
    )
    exact_source_auc_states = _stratum_auc_states(
        exact_source,
        auc_boundary=config.probe_freeze.auc_effect_boundary,
    )
    signature_profile = _predictive_signature_profile(
        screening,
        shared_auc_states,
        exact_source_auc_states,
    )
    shared_composite = _safe_composite_probe(
        shared_records,
        blocks=_COMPOSITE_BLOCKS["decoder_input_probe_union"],
        name="shared_group_decoder_input_probe_union",
        separability=p0_bc.separability,
        feature_components=feature_components,
    )
    source_composite = _safe_composite_probe(
        source_records,
        blocks=_COMPOSITE_BLOCKS["decoder_input_probe_union"],
        name="selected_source_centered_decoder_input_probe_union",
        separability=p0_bc.separability,
        feature_components=feature_components,
    )
    composition = _fingerprinted(
        {
            "schema_version": "cure-lite-failure-composition-strata-v1",
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "factor_probe_profile_receipt_fingerprint": (
                profile["receipt_fingerprint"]
            ),
            "shared_manifest_groups": shared,
            "selected_dual_role_sources_transductive_sensitivity": (
                exact_source
            ),
            "auc_three_state_by_stratum": {
                "shared_manifest_groups": shared_auc_states,
                "selected_dual_role_sources_transductive_source_centered": (
                    exact_source_auc_states
                ),
            },
            "predictive_signature_profile": signature_profile,
            "shared_group_decoder_input_probe_union": shared_composite,
            "selected_source_centered_decoder_input_probe_union": (
                source_composite
            ),
            "source_centering_scope": (
                config.probe_freeze.source_centering_interpretation
            ),
            "interpretation": (
                "predeclared composition sensitivities; the source-centered "
                "analysis is selected-overlap and transductive, does not "
                "eliminate source effects, and persistence or attenuation "
                "does not establish causal mediation"
            ),
            "authorizes_transformation": False,
            **_no_authority(),
        }
    )

    old_factual, old_legal, _ = _extract_targets(
        bundle,
        legacy,
        manifest,
        p0_bc.overlap,
    )
    new_factual, new_legal, _ = _extract_targets(
        bundle,
        view,
        manifest,
        p0_bc.overlap,
    )
    transition_raw = build_coverage_transition(
        old_factual,
        old_legal,
        new_factual,
        new_legal,
        p0_bc.overlap,
    )
    transition = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-failure-coverage-transition-receipt-v1"
            ),
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "authority_binding_receipt_fingerprint": (
                authority_receipt["receipt_fingerprint"]
            ),
            "decomposition_freeze": config.canonical_payload()[
                "coverage_transition_decomposition"
            ],
            "result": transition_raw,
            "causal_attribution_allowed": False,
            "authorizes_transformation": False,
            **_no_authority(),
        }
    )
    signatures = _factual_signatures(
        config,
        block_support,
        str(profile["receipt_fingerprint"]),
    )
    computationally_inconclusive_probes = [
        f"all_population_single_block:{block}"
        for block, result in block_oof.items()
        if result.get("execution_status") == "inconclusive"
    ]
    computationally_inconclusive_probes.extend(
        f"all_population_composite:{name}"
        for name, result in composites.items()
        if result.get("execution_status") == "inconclusive"
    )
    computationally_inconclusive_probes.extend(
        f"shared_manifest_groups_single_block:{block}"
        for block, result in shared["results"].items()
        if result.get("execution_status") == "inconclusive"
    )
    computationally_inconclusive_probes.extend(
        f"selected_dual_role_sources_single_block:{block}"
        for block, result in exact_source["results"].items()
        if result.get("execution_status") == "inconclusive"
    )
    if shared_composite.get("execution_status") == "inconclusive":
        computationally_inconclusive_probes.append(
            "shared_manifest_groups_composite:decoder_input_probe_union"
        )
    if source_composite.get("execution_status") == "inconclusive":
        computationally_inconclusive_probes.append(
            "selected_dual_role_sources_composite:"
            "decoder_input_probe_union"
        )
    decision = _decision_receipt(
        config,
        screening,
        authority_fingerprint=str(
            authority_receipt["receipt_fingerprint"]
        ),
        state_contract_fingerprint=str(
            state_contract["receipt_fingerprint"]
        ),
        profile_fingerprint=str(profile["receipt_fingerprint"]),
        composition_fingerprint=str(
            composition["receipt_fingerprint"]
        ),
        transition_fingerprint=str(transition["receipt_fingerprint"]),
        signature_fingerprint=str(signatures["receipt_fingerprint"]),
        predictive_signature_profile=signature_profile,
        computationally_inconclusive_probes=(
            computationally_inconclusive_probes
        ),
    )
    receipt_payloads = {
        "authority_binding.json": authority_receipt,
        "population_factor_inventory.json": inventory,
        "state_contract_audit.json": state_contract,
        "frozen_feature_evidence.json": feature_evidence,
        "factor_probe_profile.json": profile,
        "composition_strata.json": composition,
        "coverage_transition_decomposition.json": transition,
        "factual_miss_signatures.json": signatures,
        "diagnostic_decision.json": decision,
    }
    if tuple(receipt_payloads) != _RECEIPT_NAMES:
        raise AssertionError("failure-attribution receipt order changed")
    if tuple(config.receipt_contract.receipt_files) != _RECEIPT_NAMES:
        raise RuntimeError("receipt contract differs from runner outputs")

    bundle.verify_unchanged()
    if _implementation_binding() != implementation:
        raise RuntimeError("failure-attribution implementation changed during run")
    if any(
        file_sha256(Path(path)) != digest
        for path, digest in immutable.items()
    ):
        raise RuntimeError("a frozen failure-attribution input changed during run")

    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / ".incomplete"
    incomplete.touch(exist_ok=False)
    receipts_dir = output / "receipts"
    receipts_dir.mkdir(exist_ok=False)
    for name, payload in receipt_payloads.items():
        _write_new_json(receipts_dir / name, payload)
    receipt_sha256 = {
        name: file_sha256(receipts_dir / name)
        for name in _RECEIPT_NAMES
    }
    complete = _fingerprinted(
        {
            "schema_version": SYNTHETIC_STATE_FAILURE_RUN_SCHEMA,
            "status": "complete",
            "execution_status": decision["execution_state"],
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "config_fingerprint": config.fingerprint,
            "config_file_sha256": file_sha256(paths["config"]),
            "implementation_files": implementation,
            "environment": {
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pillow": PIL.__version__,
                "platform": platform.platform(),
            },
            "receipt_files": list(_RECEIPT_NAMES),
            "receipt_sha256": receipt_sha256,
            "decision_fingerprint": decision["receipt_fingerprint"],
            "formal_p0_b": "fail",
            "formal_p0_c": "fail",
            "formal_p0_d": "not_evaluated",
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
            **_no_authority(),
            "complete_fingerprint_scope": (
                "all-fields-except-complete-fingerprint"
            ),
        },
        field="complete_fingerprint",
    )
    _write_new_json(output / "COMPLETE.json", complete)
    incomplete.unlink()
    return complete


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
