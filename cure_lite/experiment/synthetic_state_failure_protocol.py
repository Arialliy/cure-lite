"""Frozen protocol for D_R-only synthetic-state failure attribution.

This module defines a configuration contract only.  It deliberately contains
no diagnostic runner, state transformation, candidate-S construction, P0-D
simulation, training, calibration, inference, or D_V/D_T access.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ..cache.schema import stable_fingerprint
from .geometry_catalog_protocol import GeometryCatalogInputBinding


SYNTHETIC_STATE_FAILURE_CONFIG_SCHEMA = (
    "cure-lite-synthetic-state-failure-attribution-config-v1"
)
_SHA256 = frozenset("0123456789abcdef")


def _mapping(
    value: object,
    fields: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields are not canonical")
    return dict(value)


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _true(value: object, *, name: str) -> bool:
    if value is not True:
        raise ValueError(f"{name} must be true")
    return True


def _false(value: object, *, name: str) -> bool:
    if value is not False:
        raise ValueError(f"{name} must be false")
    return False


def _fixed_string(
    value: object,
    expected: str,
    *,
    name: str,
) -> str:
    if value != expected:
        raise ValueError(f"{name} must be {expected!r}")
    return expected


def _fixed_scalar(
    value: object,
    expected: str | int | float,
    *,
    name: str,
) -> str | int | float:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} must be {expected!r}")
    return expected


def _fixed_list(
    value: object,
    expected: tuple[str, ...],
    *,
    name: str,
) -> tuple[str, ...]:
    if value != list(expected):
        raise ValueError(f"{name} must equal the frozen list")
    return expected


_AUTHORITY_DIGESTS = {
    "p0_bc_config_file_sha256": (
        "3c61a9839c33bd517e86c4c47a48e4b404397f19697e6a1c519948d4081ef047"
    ),
    "p0_bc_config_fingerprint": (
        "707a9d50d4b707988a90a7220162d2f47c6f219380eee60e7c262a9c11cd8099"
    ),
    "p0_bc_r1_complete_file_sha256": (
        "97fa9f8e603c07eb34bd541de4d7284718af9c97657965d44692c1a6c7b7e379"
    ),
    "p0_bc_r1_complete_fingerprint": (
        "de1ad2b460db48c2ec28d21814be0cbc7190e121ee5be5321bcbf178ba6ff997"
    ),
    "p0_bc_r2_complete_file_sha256": (
        "97fa9f8e603c07eb34bd541de4d7284718af9c97657965d44692c1a6c7b7e379"
    ),
    "p0_bc_r2_complete_fingerprint": (
        "de1ad2b460db48c2ec28d21814be0cbc7190e121ee5be5321bcbf178ba6ff997"
    ),
    "p0_bc_population_file_sha256": (
        "5d080d32e9515a9faecc957a9d7b4441253e04680a8a6599cafd8fb496b2b72f"
    ),
    "p0_bc_population_receipt_fingerprint": (
        "a477157d4608cb6392fbdf37dace6e27747b49aad2114a1df5b288640782fe20"
    ),
    "p0_bc_p0_b_file_sha256": (
        "ed080830c271f672b250453bb5b7c4f8cbf956f92b3769c30a053616f36809e8"
    ),
    "p0_bc_p0_b_receipt_fingerprint": (
        "44b2757d3e9a01ecda264557b00c264090da6ea69b911f53e4d3fadb5fcda744"
    ),
    "p0_bc_p0_c_file_sha256": (
        "cd977ec1fa3a5e0bfa9375aea698e53daec6d5f366be1f27b65724d9615e42dd"
    ),
    "p0_bc_p0_c_receipt_fingerprint": (
        "ff79f07a22923c961ca328ba2417ddac95b27a3fc8ffd68cac7ac37a4bc474f0"
    ),
    "p0_bc_decision_file_sha256": (
        "a3c00cb0c628413d14b00377a0ed50e9156f72c0d0cb24e3fdb638632c1febb0"
    ),
    "p0_bc_decision_receipt_fingerprint": (
        "b1b00f7d520a992bcfc348a2ecca9033d479f44baedbe4140c519a6b277987c1"
    ),
    "legacy_209_p0_b_file_sha256": (
        "d1c257eb886e07da2a1682096417d2e03ea76d3ed01ffc5946f2b18eef6f7502"
    ),
    "legacy_209_p0_b_receipt_fingerprint": (
        "b6ee8ca8aad4cac314269222a10e16671fcbebeaf27733cbf19fd64a068f7928"
    ),
    "p0_a1_receipt_fingerprint": (
        "8ce8d3b00fbc6569a5aa5272a343c9293c0da4b06d592ae2b3c9d2066a93e351"
    ),
    "eligible_catalog_fingerprint": (
        "a7f3862e41272edb8cafa50398f104c5fab6a8ae55c78bf90c1f2ef34bf1bcd9"
    ),
}

_INPUT_DIGESTS = {
    "manifest_file_sha256": (
        "aa8e33529bd86f564ce6e163e0f9a7b1b3053e9c15054a59c6702a1523f35c02"
    ),
    "state_index_sha256": (
        "075fc1ad217f365df85b1d29568ad215f06ce6e0b691ef78a5dd85f0affe6298"
    ),
    "state_index_fingerprint": (
        "d06e05151b2d7e9e4829d52fca0037e1c42c8a552c015312a03afd26c3fbf59d"
    ),
    "base_fingerprint": (
        "5f69986b95d11a89c5a5e91d6bdd63add865eda102be8ce486722fee8cd00dce"
    ),
    "base_state_fingerprint": (
        "1e17bc11465bf4fd63b5a697dd466cd2d78505d44ba83f9862ffbce3bd39f3c4"
    ),
    "state_fingerprint": (
        "eedfc898171e1e6ebd3ef29e17f52da93e89a9a4a33351ea14cddb838d651e72"
    ),
    "gt_fingerprint": (
        "5705d1ac381f9744d13222396d064a2bc034edea93e657c5f62705cf269823a8"
    ),
}


@dataclass(frozen=True)
class FailureAuthorityBinding:
    p0_bc_config_file_sha256: str
    p0_bc_config_fingerprint: str
    p0_bc_r1_complete_file_sha256: str
    p0_bc_r1_complete_fingerprint: str
    p0_bc_r2_complete_file_sha256: str
    p0_bc_r2_complete_fingerprint: str
    p0_bc_population_file_sha256: str
    p0_bc_population_receipt_fingerprint: str
    p0_bc_p0_b_file_sha256: str
    p0_bc_p0_b_receipt_fingerprint: str
    p0_bc_p0_c_file_sha256: str
    p0_bc_p0_c_receipt_fingerprint: str
    p0_bc_decision_file_sha256: str
    p0_bc_decision_receipt_fingerprint: str
    legacy_209_p0_b_file_sha256: str
    legacy_209_p0_b_receipt_fingerprint: str
    p0_a1_receipt_fingerprint: str
    eligible_catalog_fingerprint: str

    @classmethod
    def from_mapping(cls, value: object) -> "FailureAuthorityBinding":
        payload = _mapping(
            value,
            set(_AUTHORITY_DIGESTS),
            name="authority_binding",
        )
        for field, expected in _AUTHORITY_DIGESTS.items():
            actual = _digest(
                payload[field],
                name=f"authority_binding.{field}",
            )
            if actual != expected:
                raise ValueError(
                    f"authority_binding.{field} differs from the freeze"
                )
        return cls(**payload)


@dataclass(frozen=True)
class FailureExecutionPolicy:
    allowed_runtime_splits: tuple[str, ...]
    create_only_output: bool
    allow_training: bool
    allow_calibration: bool
    allow_inference: bool
    allow_d_v_access: bool
    allow_d_t_access: bool
    allow_candidate_s_construction: bool
    allow_p0_d: bool
    allow_transformation_construction: bool
    allow_full_cure: bool
    allow_backbone_integration: bool

    @classmethod
    def from_mapping(cls, value: object) -> "FailureExecutionPolicy":
        fields = {
            "allowed_runtime_splits",
            "create_only_output",
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
        }
        payload = _mapping(value, fields, name="execution_policy")
        if payload["allowed_runtime_splits"] != ["D_R"]:
            raise ValueError("execution_policy permits exactly D_R")
        return cls(
            allowed_runtime_splits=("D_R",),
            create_only_output=_true(
                payload["create_only_output"],
                name="execution_policy.create_only_output",
            ),
            **{
                field: _false(
                    payload[field],
                    name=f"execution_policy.{field}",
                )
                for field in sorted(fields - {
                    "allowed_runtime_splits",
                    "create_only_output",
                })
            },
        )


@dataclass(frozen=True)
class ExcludedLegalIdentity:
    sample_id: str
    gt_id: int
    pred_id: int

    @classmethod
    def from_mapping(cls, value: object) -> "ExcludedLegalIdentity":
        payload = _mapping(
            value,
            {"sample_id", "gt_id", "pred_id"},
            name="excluded_legal_identity",
        )
        if not isinstance(payload["sample_id"], str) or not payload["sample_id"]:
            raise ValueError("excluded legal sample_id must be non-empty")
        return cls(
            sample_id=payload["sample_id"],
            gt_id=_positive_int(
                payload["gt_id"],
                name="excluded_legal_identity.gt_id",
            ),
            pred_id=_positive_int(
                payload["pred_id"],
                name="excluded_legal_identity.pred_id",
            ),
        )


_EXCLUDED_IDENTITIES = (
    ExcludedLegalIdentity("XDU486", 1, 1),
    ExcludedLegalIdentity("XDU526", 1, 1),
    ExcludedLegalIdentity("XDU965", 1, 1),
)


@dataclass(frozen=True)
class FailurePopulationBinding:
    factual_targets: int
    factual_groups: int
    legacy_legal_targets: int
    legacy_legal_source_images: int
    legacy_legal_groups: int
    geometry_safe_legal_targets: int
    geometry_safe_legal_source_images: int
    geometry_safe_legal_groups: int
    geometry_excluded_legal_targets: int
    role_overlap_groups: int
    role_overlap_factual_targets: int
    role_overlap_legal_targets: int
    legal_exclusive_groups: int
    dual_role_source_images: int
    dual_role_source_factual_targets: int
    dual_role_source_legal_targets: int
    group_key: str
    excluded_legal_identities: tuple[ExcludedLegalIdentity, ...]

    @classmethod
    def from_mapping(cls, value: object) -> "FailurePopulationBinding":
        fields = {
            "factual_targets",
            "factual_groups",
            "legacy_legal_targets",
            "legacy_legal_source_images",
            "legacy_legal_groups",
            "geometry_safe_legal_targets",
            "geometry_safe_legal_source_images",
            "geometry_safe_legal_groups",
            "geometry_excluded_legal_targets",
            "role_overlap_groups",
            "role_overlap_factual_targets",
            "role_overlap_legal_targets",
            "legal_exclusive_groups",
            "dual_role_source_images",
            "dual_role_source_factual_targets",
            "dual_role_source_legal_targets",
            "group_key",
            "excluded_legal_identities",
        }
        payload = _mapping(value, fields, name="population_binding")
        expected = {
            "factual_targets": 32,
            "factual_groups": 24,
            "legacy_legal_targets": 209,
            "legacy_legal_source_images": 150,
            "legacy_legal_groups": 146,
            "geometry_safe_legal_targets": 206,
            "geometry_safe_legal_source_images": 149,
            "geometry_safe_legal_groups": 145,
            "geometry_excluded_legal_targets": 3,
            "role_overlap_groups": 14,
            "role_overlap_factual_targets": 18,
            "role_overlap_legal_targets": 25,
            "legal_exclusive_groups": 131,
            "dual_role_source_images": 14,
            "dual_role_source_factual_targets": 18,
            "dual_role_source_legal_targets": 21,
            "group_key": "manifest.group_id",
        }
        for field, required in expected.items():
            if payload[field] != required:
                raise ValueError(
                    f"population_binding.{field} must be {required!r}"
                )
        identities_raw = payload["excluded_legal_identities"]
        if not isinstance(identities_raw, list):
            raise ValueError(
                "population_binding.excluded_legal_identities must be a list"
            )
        identities = tuple(
            ExcludedLegalIdentity.from_mapping(item)
            for item in identities_raw
        )
        if identities != _EXCLUDED_IDENTITIES:
            raise ValueError(
                "population_binding.excluded_legal_identities differ "
                "from the freeze"
            )
        result = cls(
            **{
                field: payload[field]
                for field in expected
            },
            excluded_legal_identities=identities,
        )
        if (
            result.geometry_safe_legal_targets
            + result.geometry_excluded_legal_targets
            != result.legacy_legal_targets
            or result.legal_exclusive_groups + result.role_overlap_groups
            != result.geometry_safe_legal_groups
        ):
            raise ValueError("population_binding arithmetic is inconsistent")
        return result


@dataclass(frozen=True)
class FailureFactor:
    factor_id: str
    representation: tuple[str, ...]
    system_role: str
    decoder_observed: bool
    claim_scope: str

    @classmethod
    def from_mapping(cls, value: object) -> "FailureFactor":
        payload = _mapping(
            value,
            {
                "factor_id",
                "representation",
                "system_role",
                "decoder_observed",
                "claim_scope",
            },
            name="factor",
        )
        representation = payload["representation"]
        if (
            not isinstance(representation, list)
            or not representation
            or any(not isinstance(item, str) or not item for item in representation)
        ):
            raise ValueError("factor representation must be non-empty strings")
        if not isinstance(payload["decoder_observed"], bool):
            raise TypeError("factor decoder_observed must be boolean")
        return cls(
            factor_id=payload["factor_id"],
            representation=tuple(representation),
            system_role=payload["system_role"],
            decoder_observed=payload["decoder_observed"],
            claim_scope=payload["claim_scope"],
        )


_FACTOR_PAYLOADS = (
    {
        "factor_id": "G_full",
        "representation": [
            "log1p_gt_area",
            "log_gt_aspect_ratio",
            "gt_bbox_fill_fraction",
            "border_distance_normalized",
            "gt_centroid_y_normalized",
            "gt_centroid_x_normalized",
        ],
        "system_role": "full-gt-target-geometry",
        "decoder_observed": False,
        "claim_scope": "fixed-six-dimensional-full-gt-summary-only",
    },
    {
        "factor_id": "W",
        "representation": [
            "log1p_supervision_area",
            "supervision_fraction_of_gt",
            "supervision_component_count",
            "supervision_to_gt_centroid_distance_normalized",
        ],
        "system_role": "writable-supervision-geometry",
        "decoder_observed": False,
        "claim_scope": "fixed-four-dimensional-loss-support-summary-only",
    },
    {
        "factor_id": "P",
        "representation": [
            "clipped_logit_base_gt_mean",
            "clipped_logit_base_ring_mean",
            "base_gt_std",
            "base_ring_std",
            "clipped_logit_base_gt_max",
            "clipped_logit_base_ring_max",
            "base_gt_minus_ring_mean",
        ],
        "system_role": "base-selection-and-construction-proxy",
        "decoder_observed": False,
        "claim_scope": "fixed-seven-dimensional-proxy-not-decoder-input",
    },
    {
        "factor_id": "F_local",
        "representation": [
            "target_channel_mean",
            "target_channel_std",
            "target_minus_ring_channel_mean",
            "target_rms",
        ],
        "system_role": "decoder-input-local-feature-summary",
        "decoder_observed": True,
        "claim_scope": "fixed-local-moment-vector-probe-only",
    },
    {
        "factor_id": "F_background_global",
        "representation": [
            "ring_channel_mean",
            "ring_channel_std",
            "global_channel_mean",
            "global_channel_std",
        ],
        "system_role": "decoder-input-context-feature-summary",
        "decoder_observed": True,
        "claim_scope": "fixed-background-global-moment-vector-probe-only",
    },
    {
        "factor_id": "O",
        "representation": [
            "conditioning_gt_fraction",
            "conditioning_ring_fraction",
            "nearest_component_centroid_distance_normalized",
            "projected_local_patch_row_major",
            "projected_global_fraction",
        ],
        "system_role": "decoder-input-occupancy-summary",
        "decoder_observed": True,
        "claim_scope": "fixed-twenty-nine-dimensional-occupancy-probe-only",
    },
)


@dataclass(frozen=True)
class FailureFactorTaxonomy:
    factors: tuple[FailureFactor, ...]
    decoder_input_factors: tuple[str, ...]
    proxy_factors: tuple[str, ...]
    interpretation_limit: str

    @classmethod
    def from_mapping(cls, value: object) -> "FailureFactorTaxonomy":
        payload = _mapping(
            value,
            {
                "factors",
                "decoder_input_factors",
                "proxy_factors",
                "interpretation_limit",
            },
            name="factor_taxonomy",
        )
        factors_raw = payload["factors"]
        if not isinstance(factors_raw, list):
            raise ValueError("factor_taxonomy.factors must be a list")
        factors = tuple(
            FailureFactor.from_mapping(item) for item in factors_raw
        )
        factor_payloads = tuple(
            {
                **asdict(item),
                "representation": list(item.representation),
            }
            for item in factors
        )
        if factor_payloads != _FACTOR_PAYLOADS:
            raise ValueError("factor_taxonomy.factors differ from the freeze")
        decoder = _fixed_list(
            payload["decoder_input_factors"],
            ("F_local", "F_background_global", "O"),
            name="factor_taxonomy.decoder_input_factors",
        )
        proxy = _fixed_list(
            payload["proxy_factors"],
            ("P",),
            name="factor_taxonomy.proxy_factors",
        )
        limit = _fixed_string(
            payload["interpretation_limit"],
            (
                "fixed-low-dimensional-probes-do-not-prove-full-state-"
                "distribution-equality-or-causal-dominance"
            ),
            name="factor_taxonomy.interpretation_limit",
        )
        return cls(factors, decoder, proxy, limit)


@dataclass(frozen=True)
class FailureProbeFreeze:
    strata: tuple[str, ...]
    primary_single_blocks: tuple[str, ...]
    decoder_input_probe_union: tuple[str, ...]
    fixed_drop_one_probes: tuple[str, ...]
    group_key: str
    folds: int
    classifier: str
    classifier_l2: float
    bootstrap_replicates: int
    bootstrap_seed: int
    auc_effect_boundary: float
    mmd_rule: str
    mmd_reference_replicates: int
    mmd_reference_seed: int
    mmd_reference_quantile: float
    multiple_comparison_rule: str
    feature_components: int
    feature_projection_rule: str
    oof_feature_projection_fit_population: str
    coverage_feature_projection_fit_population: str
    mmd_feature_projection_fit_population: str
    fixed_fit_failure_policy: str
    partial_stratum_completion_allowed: bool
    arbitrary_subset_search_allowed: bool
    descriptor_selection_allowed: bool
    hyperparameter_search_allowed: bool
    oof_fit_rule: str
    source_centering_interpretation: str
    single_block_interpretation: str
    drop_one_interpretation: str

    @classmethod
    def from_mapping(cls, value: object) -> "FailureProbeFreeze":
        fields = set(cls.__dataclass_fields__)
        payload = _mapping(value, fields, name="probe_freeze")
        frozen_lists = {
            "strata": (
                "all-geometry-safe-population",
                "shared-manifest-groups-only",
                "selected-dual-role-source-images-transductive-sensitivity",
            ),
            "primary_single_blocks": (
                "G_full",
                "W",
                "P",
                "F_local",
                "F_background_global",
                "O",
            ),
            "decoder_input_probe_union": (
                "F_local",
                "F_background_global",
                "O",
            ),
            "fixed_drop_one_probes": (
                "drop_F_local",
                "drop_F_background_global",
                "drop_O",
            ),
        }
        converted: dict[str, Any] = {}
        for field, expected in frozen_lists.items():
            converted[field] = _fixed_list(
                payload[field],
                expected,
                name=f"probe_freeze.{field}",
            )
        expected_scalars = {
            "group_key": "manifest.group_id",
            "folds": 5,
            "classifier": "class-balanced-l2-logistic-irls-v1",
            "classifier_l2": 1.0,
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 1729,
            "auc_effect_boundary": 0.7,
            "mmd_rule": (
                "descriptive-group-u-multiscale-rbf-matched-legal-"
                "reference-v1"
            ),
            "mmd_reference_replicates": 1000,
            "mmd_reference_seed": 2718,
            "mmd_reference_quantile": 0.95,
            "multiple_comparison_rule": (
                "none-descriptive-q95-crossing-six-primary-blocks-v1"
            ),
            "feature_components": 6,
            "feature_projection_rule": (
                "robust-scaled-legal-only-pca-plus-residual-v1"
            ),
            "oof_feature_projection_fit_population": (
                "training-fold-legal-targets-only"
            ),
            "coverage_feature_projection_fit_population": (
                "all-geometry-safe-legal-targets"
            ),
            "mmd_feature_projection_fit_population": (
                "legal-exclusive-manifest-groups-only"
            ),
            "fixed_fit_failure_policy": (
                "record-block-inconclusive-no-refit-no-override-v1"
            ),
            "oof_fit_rule": (
                "group-oof-projection-scale-and-classifier-fold-local-v1"
            ),
            "source_centering_interpretation": (
                "selected-overlap-transductive-sensitivity-not-source-"
                "elimination"
            ),
            "single_block_interpretation": (
                "predictive-sufficiency-screen-not-causal-attribution"
            ),
            "drop_one_interpretation": (
                "conditional-predictive-contribution-not-causal-ablation"
            ),
        }
        for field, expected in expected_scalars.items():
            converted[field] = _fixed_scalar(
                payload[field],
                expected,
                name=f"probe_freeze.{field}",
            )
        for field in (
            "partial_stratum_completion_allowed",
        ):
            converted[field] = _true(
                payload[field],
                name=f"probe_freeze.{field}",
            )
        for field in (
            "arbitrary_subset_search_allowed",
            "descriptor_selection_allowed",
            "hyperparameter_search_allowed",
        ):
            converted[field] = _false(
                payload[field],
                name=f"probe_freeze.{field}",
            )
        return cls(**converted)


@dataclass(frozen=True)
class CoverageTransitionDecomposition:
    factual_targets: int
    population_axis: tuple[str, ...]
    fit_axis: tuple[str, ...]
    cells: tuple[str, ...]
    legacy_expected_covered: int
    geometry_safe_expected_covered: int
    reference_radius_rule: str
    individual_exclusion_replay: str
    per_factual_transition_required: bool
    descriptive_shapley_allowed: bool
    causal_attribution_allowed: bool

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> "CoverageTransitionDecomposition":
        fields = set(cls.__dataclass_fields__)
        payload = _mapping(
            value,
            fields,
            name="coverage_transition_decomposition",
        )
        expected_lists = {
            "population_axis": ("legal209", "legal206"),
            "fit_axis": ("fit209", "fit206"),
            "cells": (
                "legal209-fit209",
                "legal206-fit209",
                "legal209-fit206",
                "legal206-fit206",
            ),
        }
        converted: dict[str, Any] = {}
        for field, expected in expected_lists.items():
            converted[field] = _fixed_list(
                payload[field],
                expected,
                name=f"coverage_transition_decomposition.{field}",
            )
        expected_scalars = {
            "factual_targets": 32,
            "legacy_expected_covered": 23,
            "geometry_safe_expected_covered": 16,
            "reference_radius_rule": (
                "recompute-within-each-population-fit-cell-v1"
            ),
            "individual_exclusion_replay": (
                "fit209-and-legacy-fixed-radius-delete-each-and-all-v1"
            ),
        }
        for field, expected in expected_scalars.items():
            converted[field] = _fixed_scalar(
                payload[field],
                expected,
                name=f"coverage_transition_decomposition.{field}",
            )
        converted["per_factual_transition_required"] = _true(
            payload["per_factual_transition_required"],
            name=(
                "coverage_transition_decomposition."
                "per_factual_transition_required"
            ),
        )
        converted["descriptive_shapley_allowed"] = _true(
            payload["descriptive_shapley_allowed"],
            name=(
                "coverage_transition_decomposition."
                "descriptive_shapley_allowed"
            ),
        )
        converted["causal_attribution_allowed"] = _false(
            payload["causal_attribution_allowed"],
            name=(
                "coverage_transition_decomposition."
                "causal_attribution_allowed"
            ),
        )
        return cls(**converted)


@dataclass(frozen=True)
class FailureReceiptContract:
    receipt_files: tuple[str, ...]
    require_complete_marker: bool
    require_per_target_ledger: bool
    require_fold_fit_audit: bool
    require_two_run_byte_identity: bool

    @classmethod
    def from_mapping(cls, value: object) -> "FailureReceiptContract":
        fields = set(cls.__dataclass_fields__)
        payload = _mapping(value, fields, name="receipt_contract")
        files = _fixed_list(
            payload["receipt_files"],
            (
                "authority_binding.json",
                "population_factor_inventory.json",
                "state_contract_audit.json",
                "frozen_feature_evidence.json",
                "factor_probe_profile.json",
                "composition_strata.json",
                "coverage_transition_decomposition.json",
                "factual_miss_signatures.json",
                "diagnostic_decision.json",
            ),
            name="receipt_contract.receipt_files",
        )
        return cls(
            receipt_files=files,
            require_complete_marker=_true(
                payload["require_complete_marker"],
                name="receipt_contract.require_complete_marker",
            ),
            require_per_target_ledger=_true(
                payload["require_per_target_ledger"],
                name="receipt_contract.require_per_target_ledger",
            ),
            require_fold_fit_audit=_true(
                payload["require_fold_fit_audit"],
                name="receipt_contract.require_fold_fit_audit",
            ),
            require_two_run_byte_identity=_true(
                payload["require_two_run_byte_identity"],
                name="receipt_contract.require_two_run_byte_identity",
            ),
        )


@dataclass(frozen=True)
class FailureDecisionPolicy:
    execution_states: tuple[str, ...]
    metric_states: tuple[str, ...]
    block_states: tuple[str, ...]
    overall_scientific_gate: str
    no_strong_signal_interpretation: str
    strong_signal_interpretation: str
    separate_transformation_protocol_required: bool
    authorizes_transformation_construction: bool
    authorizes_candidate_s_construction: bool
    authorizes_p0_d: bool
    authorizes_training: bool
    authorizes_d_v_evaluation: bool
    authorizes_full_cure: bool

    @classmethod
    def from_mapping(cls, value: object) -> "FailureDecisionPolicy":
        fields = set(cls.__dataclass_fields__)
        payload = _mapping(value, fields, name="decision_policy")
        converted: dict[str, Any] = {}
        frozen_lists = {
            "execution_states": (
                "complete",
                "partial_inconclusive",
                "invalid",
            ),
            "metric_states": ("strong", "not_strong", "inconclusive"),
            "block_states": (
                "strong_role_signal",
                "no_strong_role_signal_detected",
                "mixed_or_inconclusive",
            ),
        }
        for field, expected in frozen_lists.items():
            converted[field] = _fixed_list(
                payload[field],
                expected,
                name=f"decision_policy.{field}",
            )
        expected_strings = {
            "overall_scientific_gate": (
                "none-descriptive-diagnostic-only"
            ),
            "no_strong_signal_interpretation": (
                "absence-of-strong-signal-in-frozen-probes-not-"
                "distribution-equality"
            ),
            "strong_signal_interpretation": (
                "predictive-role-signal-in-frozen-probes-not-causal-proof"
            ),
        }
        for field, expected in expected_strings.items():
            converted[field] = _fixed_string(
                payload[field],
                expected,
                name=f"decision_policy.{field}",
            )
        converted["separate_transformation_protocol_required"] = _true(
            payload["separate_transformation_protocol_required"],
            name=(
                "decision_policy."
                "separate_transformation_protocol_required"
            ),
        )
        for field in (
            "authorizes_transformation_construction",
            "authorizes_candidate_s_construction",
            "authorizes_p0_d",
            "authorizes_training",
            "authorizes_d_v_evaluation",
            "authorizes_full_cure",
        ):
            converted[field] = _false(
                payload[field],
                name=f"decision_policy.{field}",
            )
        return cls(**converted)


@dataclass(frozen=True)
class SyntheticStateFailureProtocol:
    schema_version: str
    protocol_id: str
    stage_role: str
    dataset: str
    split: str
    authority_binding: FailureAuthorityBinding
    input_binding: GeometryCatalogInputBinding
    execution_policy: FailureExecutionPolicy
    population_binding: FailurePopulationBinding
    factor_taxonomy: FailureFactorTaxonomy
    probe_freeze: FailureProbeFreeze
    coverage_transition_decomposition: CoverageTransitionDecomposition
    receipt_contract: FailureReceiptContract
    decision_policy: FailureDecisionPolicy

    @classmethod
    def from_mapping(cls, value: object) -> "SyntheticStateFailureProtocol":
        fields = set(cls.__dataclass_fields__)
        payload = _mapping(
            value,
            fields,
            name="synthetic-state failure-attribution config",
        )
        _fixed_string(
            payload["schema_version"],
            SYNTHETIC_STATE_FAILURE_CONFIG_SCHEMA,
            name="schema_version",
        )
        _fixed_string(
            payload["protocol_id"],
            "irstd1k-dr-synthetic-state-failure-attribution-v1",
            name="protocol_id",
        )
        _fixed_string(
            payload["stage_role"],
            "descriptive-diagnostic-only",
            name="stage_role",
        )
        _fixed_string(payload["dataset"], "IRSTD-1K", name="dataset")
        _fixed_string(payload["split"], "D_R", name="split")
        input_binding = GeometryCatalogInputBinding.from_mapping(
            payload["input_binding"]
        )
        if asdict(input_binding) != _INPUT_DIGESTS:
            raise ValueError("input_binding differs from the D_R freeze")
        return cls(
            schema_version=payload["schema_version"],
            protocol_id=payload["protocol_id"],
            stage_role=payload["stage_role"],
            dataset=payload["dataset"],
            split="D_R",
            authority_binding=FailureAuthorityBinding.from_mapping(
                payload["authority_binding"]
            ),
            input_binding=input_binding,
            execution_policy=FailureExecutionPolicy.from_mapping(
                payload["execution_policy"]
            ),
            population_binding=FailurePopulationBinding.from_mapping(
                payload["population_binding"]
            ),
            factor_taxonomy=FailureFactorTaxonomy.from_mapping(
                payload["factor_taxonomy"]
            ),
            probe_freeze=FailureProbeFreeze.from_mapping(
                payload["probe_freeze"]
            ),
            coverage_transition_decomposition=(
                CoverageTransitionDecomposition.from_mapping(
                    payload["coverage_transition_decomposition"]
                )
            ),
            receipt_contract=FailureReceiptContract.from_mapping(
                payload["receipt_contract"]
            ),
            decision_policy=FailureDecisionPolicy.from_mapping(
                payload["decision_policy"]
            ),
        )

    def canonical_payload(self) -> dict[str, object]:
        return json.loads(
            json.dumps(
                asdict(self),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def load_synthetic_state_failure_protocol(
    path: str | Path,
) -> SyntheticStateFailureProtocol:
    """Load one duplicate-key-free, non-symlink frozen protocol."""

    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError(
            "synthetic-state failure-attribution config may not be a symlink"
        )
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(
            "synthetic-state failure-attribution config must be a regular file"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(
                    "synthetic-state failure-attribution config contains "
                    f"duplicate key {key!r}"
                )
            result[key] = item
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(
            "synthetic-state failure-attribution config contains "
            f"non-finite number {value}"
        )

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    config = SyntheticStateFailureProtocol.from_mapping(payload)
    if config.canonical_payload() != payload:
        raise ValueError(
            "synthetic-state failure-attribution config JSON is not canonical"
        )
    return config


__all__ = [
    "SYNTHETIC_STATE_FAILURE_CONFIG_SCHEMA",
    "CoverageTransitionDecomposition",
    "ExcludedLegalIdentity",
    "FailureAuthorityBinding",
    "FailureDecisionPolicy",
    "FailureExecutionPolicy",
    "FailureFactor",
    "FailureFactorTaxonomy",
    "FailurePopulationBinding",
    "FailureProbeFreeze",
    "FailureReceiptContract",
    "SyntheticStateFailureProtocol",
    "load_synthetic_state_failure_protocol",
]
