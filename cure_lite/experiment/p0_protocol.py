"""Frozen, D_R-only protocol objects for CURE-Lite P0 diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from ..cache.schema import stable_fingerprint


P0_CONFIG_SCHEMA = "cure-lite-p0-diagnostic-config-v1"
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


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, *, name: str, positive: bool = False) -> float:
    result = _finite(value, name=name)
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not lower_ok or result > 1.0:
        interval = "(0,1]" if positive else "[0,1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _true(value: object, *, name: str) -> bool:
    if value is not True:
        raise ValueError(f"{name} must be true")
    return True


def _size(value: object, *, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item JSON list")
    return (
        _positive_int(value[0], name=f"{name}[0]"),
        _positive_int(value[1], name=f"{name}[1]"),
    )


@dataclass(frozen=True)
class P0InputBinding:
    manifest_file_sha256: str
    state_index_sha256: str
    state_index_fingerprint: str
    base_fingerprint: str
    base_state_fingerprint: str
    state_fingerprint: str

    @classmethod
    def from_mapping(cls, value: object) -> "P0InputBinding":
        payload = _mapping(
            value,
            {
                "manifest_file_sha256",
                "state_index_sha256",
                "state_index_fingerprint",
                "base_fingerprint",
                "base_state_fingerprint",
                "state_fingerprint",
            },
            name="input_binding",
        )
        return cls(
            **{
                key: _digest(item, name=f"input_binding.{key}")
                for key, item in payload.items()
            }
        )


@dataclass(frozen=True)
class P0GeometryConfig:
    expected_native_size: tuple[int, int]
    expected_evaluation_size: tuple[int, int]
    foreground_rule: str
    resize_rule: str
    connectivity: int
    min_component_area: int
    require_zero_native_disappearances: bool
    require_zero_resized_merges: bool
    require_zero_native_splits: bool
    require_one_to_one_legal_lineage: bool
    legal_area_ratio_min_inclusive: float
    legal_area_ratio_max_inclusive: float
    legal_centroid_shift_max_px256_inclusive: float
    geometry_quantization: int

    @classmethod
    def from_mapping(cls, value: object) -> "P0GeometryConfig":
        fields = {
            "expected_native_size",
            "expected_evaluation_size",
            "foreground_rule",
            "resize_rule",
            "connectivity",
            "min_component_area",
            "require_zero_native_disappearances",
            "require_zero_resized_merges",
            "require_zero_native_splits",
            "require_one_to_one_legal_lineage",
            "legal_area_ratio_min_inclusive",
            "legal_area_ratio_max_inclusive",
            "legal_centroid_shift_max_px256_inclusive",
            "geometry_quantization",
        }
        payload = _mapping(value, fields, name="geometry")
        if payload["foreground_rule"] != "uint8-positive-v1":
            raise ValueError("geometry foreground_rule is not supported")
        if payload["resize_rule"] != "pil-nearest-v1":
            raise ValueError("geometry resize_rule is not supported")
        connectivity = _positive_int(
            payload["connectivity"], name="geometry.connectivity"
        )
        min_area = _positive_int(
            payload["min_component_area"], name="geometry.min_component_area"
        )
        if connectivity != 8 or min_area != 1:
            raise ValueError("P0 geometry must reproduce 8-connected min-area-1 GT")
        minimum = _finite(
            payload["legal_area_ratio_min_inclusive"],
            name="geometry.legal_area_ratio_min_inclusive",
        )
        maximum = _finite(
            payload["legal_area_ratio_max_inclusive"],
            name="geometry.legal_area_ratio_max_inclusive",
        )
        centroid = _finite(
            payload["legal_centroid_shift_max_px256_inclusive"],
            name="geometry.legal_centroid_shift_max_px256_inclusive",
        )
        if minimum <= 0.0 or maximum < minimum or centroid < 0.0:
            raise ValueError("geometry distortion thresholds are invalid")
        return cls(
            expected_native_size=_size(
                payload["expected_native_size"],
                name="geometry.expected_native_size",
            ),
            expected_evaluation_size=_size(
                payload["expected_evaluation_size"],
                name="geometry.expected_evaluation_size",
            ),
            foreground_rule=payload["foreground_rule"],
            resize_rule=payload["resize_rule"],
            connectivity=connectivity,
            min_component_area=min_area,
            require_zero_native_disappearances=_true(
                payload["require_zero_native_disappearances"],
                name="geometry.require_zero_native_disappearances",
            ),
            require_zero_resized_merges=_true(
                payload["require_zero_resized_merges"],
                name="geometry.require_zero_resized_merges",
            ),
            require_zero_native_splits=_true(
                payload["require_zero_native_splits"],
                name="geometry.require_zero_native_splits",
            ),
            require_one_to_one_legal_lineage=_true(
                payload["require_one_to_one_legal_lineage"],
                name="geometry.require_one_to_one_legal_lineage",
            ),
            legal_area_ratio_min_inclusive=minimum,
            legal_area_ratio_max_inclusive=maximum,
            legal_centroid_shift_max_px256_inclusive=centroid,
            geometry_quantization=_positive_int(
                payload["geometry_quantization"],
                name="geometry.geometry_quantization",
            ),
        )


@dataclass(frozen=True)
class P0OverlapConfig:
    factual_population: str
    legal_population: str
    group_key: str
    exclude_same_group_neighbors: bool
    handcrafted_descriptor_fields: tuple[str, ...]
    probability_clip: float
    ring_inner_radius: int
    ring_outer_radius: int
    joint_feature_components: int
    joint_feature_residual: str
    joint_occupancy_representation: str
    joint_occupancy_patch_radius: int
    knn_k: int
    legal_reference_quantile: float
    coverage_minimum: float
    robust_scale_rule: str
    quantile_rule: str

    @classmethod
    def from_mapping(cls, value: object) -> "P0OverlapConfig":
        fields = {
            "factual_population",
            "legal_population",
            "group_key",
            "exclude_same_group_neighbors",
            "handcrafted_descriptor_fields",
            "probability_clip",
            "ring_inner_radius",
            "ring_outer_radius",
            "joint_feature_components",
            "joint_feature_residual",
            "joint_occupancy_representation",
            "joint_occupancy_patch_radius",
            "knn_k",
            "legal_reference_quantile",
            "coverage_minimum",
            "robust_scale_rule",
            "quantile_rule",
        }
        payload = _mapping(value, fields, name="overlap")
        expected_strings = {
            "factual_population": "reachable-factual-misses",
            "legal_population": "decoder-visible-legal-targets",
            "group_key": "manifest.group_id",
            "joint_feature_residual": (
                "legal-subspace-reconstruction-l2-per-sqrt-dimension-v1"
            ),
            "joint_occupancy_representation": (
                "raw-local-patch-plus-global-fraction-v1"
            ),
            "robust_scale_rule": "median-mad-maxdev-constant-floor-v1",
            "quantile_rule": "sorted-higher-v1",
        }
        for key, expected in expected_strings.items():
            if payload[key] != expected:
                raise ValueError(f"overlap.{key} must be {expected!r}")
        descriptor_fields = payload["handcrafted_descriptor_fields"]
        if (
            not isinstance(descriptor_fields, list)
            or not descriptor_fields
            or any(not isinstance(item, str) or not item for item in descriptor_fields)
            or len(set(descriptor_fields)) != len(descriptor_fields)
        ):
            raise ValueError("overlap handcrafted_descriptor_fields are invalid")
        inner = _positive_int(
            payload["ring_inner_radius"], name="overlap.ring_inner_radius"
        )
        outer = _positive_int(
            payload["ring_outer_radius"], name="overlap.ring_outer_radius"
        )
        if outer <= inner:
            raise ValueError("overlap outer ring radius must exceed inner radius")
        if payload["exclude_same_group_neighbors"] is not True:
            raise ValueError("P0 requires source-group-disjoint neighbours")
        probability_clip = _probability(
            payload["probability_clip"],
            name="overlap.probability_clip",
            positive=True,
        )
        if probability_clip >= 0.5:
            raise ValueError("overlap.probability_clip must be below 0.5")
        return cls(
            factual_population=payload["factual_population"],
            legal_population=payload["legal_population"],
            group_key=payload["group_key"],
            exclude_same_group_neighbors=True,
            handcrafted_descriptor_fields=tuple(descriptor_fields),
            probability_clip=probability_clip,
            ring_inner_radius=inner,
            ring_outer_radius=outer,
            joint_feature_components=_positive_int(
                payload["joint_feature_components"],
                name="overlap.joint_feature_components",
            ),
            joint_feature_residual=payload["joint_feature_residual"],
            joint_occupancy_representation=payload[
                "joint_occupancy_representation"
            ],
            joint_occupancy_patch_radius=_positive_int(
                payload["joint_occupancy_patch_radius"],
                name="overlap.joint_occupancy_patch_radius",
            ),
            knn_k=_positive_int(payload["knn_k"], name="overlap.knn_k"),
            legal_reference_quantile=_probability(
                payload["legal_reference_quantile"],
                name="overlap.legal_reference_quantile",
                positive=True,
            ),
            coverage_minimum=_probability(
                payload["coverage_minimum"],
                name="overlap.coverage_minimum",
                positive=True,
            ),
            robust_scale_rule=payload["robust_scale_rule"],
            quantile_rule=payload["quantile_rule"],
        )


@dataclass(frozen=True)
class P0SeparabilityConfig:
    folds: int
    classifier: str
    classifier_l2: float
    classifier_max_iterations: int
    classifier_tolerance: float
    auc_maximum: float
    auc_gate_rule: str
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_interval: tuple[float, float]
    bootstrap_interpretation: str
    mmd: str
    mmd_group_overlap_policy: str
    mmd_observed_summary_quantile: float
    mmd_kernel_scales: tuple[float, ...]
    mmd_bandwidth_rule: str
    mmd_reference_replicates: int
    mmd_reference_seed: int
    mmd_reference_quantile: float
    require_mmd_within_legal_reference: bool

    @classmethod
    def from_mapping(cls, value: object) -> "P0SeparabilityConfig":
        fields = {
            "folds",
            "classifier",
            "classifier_l2",
            "classifier_max_iterations",
            "classifier_tolerance",
            "auc_maximum",
            "auc_gate_rule",
            "bootstrap_replicates",
            "bootstrap_seed",
            "bootstrap_interval",
            "bootstrap_interpretation",
            "mmd",
            "mmd_group_overlap_policy",
            "mmd_observed_summary_quantile",
            "mmd_kernel_scales",
            "mmd_bandwidth_rule",
            "mmd_reference_replicates",
            "mmd_reference_seed",
            "mmd_reference_quantile",
            "require_mmd_within_legal_reference",
        }
        payload = _mapping(value, fields, name="separability")
        if payload["classifier"] != "class-balanced-l2-logistic-irls-v1":
            raise ValueError("separability classifier is not supported")
        if (
            payload["auc_gate_rule"]
            != "group-balanced-oof-point-estimate-v1"
        ):
            raise ValueError("separability AUC gate rule is not supported")
        if (
            payload["bootstrap_interpretation"]
            != "conditional-group-bootstrap-of-fixed-oof-scores-v1"
        ):
            raise ValueError(
                "separability bootstrap interpretation is not supported"
            )
        if payload["mmd"] != "group-u-multiscale-rbf-matched-legal-null-v1":
            raise ValueError("separability MMD rule is not supported")
        if (
            payload["mmd_group_overlap_policy"]
            != "remove-overlap-from-legal-reference-v1"
        ):
            raise ValueError("separability MMD overlap policy is not supported")
        if (
            payload["mmd_bandwidth_rule"]
            != "legal-exclusive-source-disjoint-positive-distance-median-v1"
        ):
            raise ValueError(
                "separability MMD bandwidth rule is not supported"
            )
        kernel_scales = payload["mmd_kernel_scales"]
        if (
            not isinstance(kernel_scales, list)
            or not kernel_scales
            or any(
                _finite(value, name="separability.mmd_kernel_scales") <= 0.0
                for value in kernel_scales
            )
            or tuple(float(value) for value in kernel_scales)
            != tuple(sorted(set(float(value) for value in kernel_scales)))
        ):
            raise ValueError(
                "separability MMD kernel scales must be sorted unique positives"
            )
        l2 = _finite(payload["classifier_l2"], name="separability.classifier_l2")
        tolerance = _finite(
            payload["classifier_tolerance"],
            name="separability.classifier_tolerance",
        )
        interval = payload["bootstrap_interval"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("separability.bootstrap_interval must have two values")
        lower = _probability(interval[0], name="bootstrap_interval[0]")
        upper = _probability(interval[1], name="bootstrap_interval[1]")
        if l2 <= 0.0 or tolerance <= 0.0 or lower >= upper:
            raise ValueError("separability numeric protocol is invalid")
        for seed_name in ("bootstrap_seed", "mmd_reference_seed"):
            if (
                isinstance(payload[seed_name], bool)
                or not isinstance(payload[seed_name], int)
                or payload[seed_name] < 0
            ):
                raise ValueError(f"separability.{seed_name} must be non-negative")
        return cls(
            folds=_positive_int(payload["folds"], name="separability.folds"),
            classifier=payload["classifier"],
            classifier_l2=l2,
            classifier_max_iterations=_positive_int(
                payload["classifier_max_iterations"],
                name="separability.classifier_max_iterations",
            ),
            classifier_tolerance=tolerance,
            auc_maximum=_probability(
                payload["auc_maximum"], name="separability.auc_maximum"
            ),
            auc_gate_rule=payload["auc_gate_rule"],
            bootstrap_replicates=_positive_int(
                payload["bootstrap_replicates"],
                name="separability.bootstrap_replicates",
            ),
            bootstrap_seed=payload["bootstrap_seed"],
            bootstrap_interval=(lower, upper),
            bootstrap_interpretation=payload["bootstrap_interpretation"],
            mmd=payload["mmd"],
            mmd_group_overlap_policy=payload["mmd_group_overlap_policy"],
            mmd_observed_summary_quantile=_probability(
                payload["mmd_observed_summary_quantile"],
                name="separability.mmd_observed_summary_quantile",
                positive=True,
            ),
            mmd_kernel_scales=tuple(
                float(value) for value in kernel_scales
            ),
            mmd_bandwidth_rule=payload["mmd_bandwidth_rule"],
            mmd_reference_replicates=_positive_int(
                payload["mmd_reference_replicates"],
                name="separability.mmd_reference_replicates",
            ),
            mmd_reference_seed=payload["mmd_reference_seed"],
            mmd_reference_quantile=_probability(
                payload["mmd_reference_quantile"],
                name="separability.mmd_reference_quantile",
                positive=True,
            ),
            require_mmd_within_legal_reference=_true(
                payload["require_mmd_within_legal_reference"],
                name="separability.require_mmd_within_legal_reference",
            ),
        )


@dataclass(frozen=True)
class P0CandidateProposalConfig:
    status: str
    descriptor_space: str
    kernel: str
    factual_weighting: str
    bandwidth_rule: str
    uniform_base: str
    lambda_rule: str
    lambda_grid_denominator: int
    minimum_uniform_floor: float
    integer_mass_total: int

    @classmethod
    def from_mapping(cls, value: object) -> "P0CandidateProposalConfig":
        fields = {
            "status",
            "descriptor_space",
            "kernel",
            "factual_weighting",
            "bandwidth_rule",
            "uniform_base",
            "lambda_rule",
            "lambda_grid_denominator",
            "minimum_uniform_floor",
            "integer_mass_total",
        }
        payload = _mapping(value, fields, name="candidate_marginal_proposal")
        expected = {
            "status": "diagnostic-only-until-p0-a-b-c-pass",
            "descriptor_space": "handcrafted",
            "kernel": "rbf",
            "factual_weighting": "uniform-over-reachable-factual-targets",
            "bandwidth_rule": (
                "median-source-disjoint-factual-to-legal-kth-distance"
            ),
            "uniform_base": "historical-source-balanced-U",
            "lambda_rule": (
                "largest-grid-value-satisfying-analytic-exposure-constraints"
            ),
        }
        for key, required in expected.items():
            if payload[key] != required:
                raise ValueError(
                    f"candidate_marginal_proposal.{key} must be {required!r}"
                )
        floor = _probability(
            payload["minimum_uniform_floor"],
            name="candidate_marginal_proposal.minimum_uniform_floor",
            positive=True,
        )
        if floor >= 1.0:
            raise ValueError("candidate uniform floor must be below one")
        return cls(
            status=payload["status"],
            descriptor_space=payload["descriptor_space"],
            kernel=payload["kernel"],
            factual_weighting=payload["factual_weighting"],
            bandwidth_rule=payload["bandwidth_rule"],
            uniform_base=payload["uniform_base"],
            lambda_rule=payload["lambda_rule"],
            lambda_grid_denominator=_positive_int(
                payload["lambda_grid_denominator"],
                name="candidate_marginal_proposal.lambda_grid_denominator",
            ),
            minimum_uniform_floor=floor,
            integer_mass_total=_positive_int(
                payload["integer_mass_total"],
                name="candidate_marginal_proposal.integer_mass_total",
            ),
        )


@dataclass(frozen=True)
class P0ExposureConfig:
    epochs: int
    steps_per_epoch: int
    synthetic_batch: int
    seeds: tuple[int, ...]
    require_all_targets_positive: bool
    target_ess_minimum_fraction_of_legal: float
    target_maximum_uniform_multiple: float
    source_ess_minimum_fraction_of_legal_sources: float
    source_maximum_uniform_multiple: float
    source_top5_maximum_share: float
    source_top10_maximum_share: float
    candidate_marginal_proposal: P0CandidateProposalConfig

    @classmethod
    def from_mapping(cls, value: object) -> "P0ExposureConfig":
        fields = {
            "epochs",
            "steps_per_epoch",
            "synthetic_batch",
            "seeds",
            "require_all_targets_positive",
            "target_ess_minimum_fraction_of_legal",
            "target_maximum_uniform_multiple",
            "source_ess_minimum_fraction_of_legal_sources",
            "source_maximum_uniform_multiple",
            "source_top5_maximum_share",
            "source_top10_maximum_share",
            "candidate_marginal_proposal",
        }
        payload = _mapping(value, fields, name="exposure")
        seeds = payload["seeds"]
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds)
            or tuple(seeds) != tuple(sorted(set(seeds)))
        ):
            raise ValueError("exposure.seeds must be sorted unique integers")
        target_multiple = _finite(
            payload["target_maximum_uniform_multiple"],
            name="exposure.target_maximum_uniform_multiple",
        )
        source_multiple = _finite(
            payload["source_maximum_uniform_multiple"],
            name="exposure.source_maximum_uniform_multiple",
        )
        if target_multiple < 1.0 or source_multiple < 1.0:
            raise ValueError("exposure maximum multiples must be at least one")
        top5 = _probability(
            payload["source_top5_maximum_share"],
            name="exposure.source_top5_maximum_share",
        )
        top10 = _probability(
            payload["source_top10_maximum_share"],
            name="exposure.source_top10_maximum_share",
        )
        if top10 < top5:
            raise ValueError("source top10 cap cannot be below top5 cap")
        return cls(
            epochs=_positive_int(payload["epochs"], name="exposure.epochs"),
            steps_per_epoch=_positive_int(
                payload["steps_per_epoch"], name="exposure.steps_per_epoch"
            ),
            synthetic_batch=_positive_int(
                payload["synthetic_batch"], name="exposure.synthetic_batch"
            ),
            seeds=tuple(seeds),
            require_all_targets_positive=_true(
                payload["require_all_targets_positive"],
                name="exposure.require_all_targets_positive",
            ),
            target_ess_minimum_fraction_of_legal=_probability(
                payload["target_ess_minimum_fraction_of_legal"],
                name="exposure.target_ess_minimum_fraction_of_legal",
                positive=True,
            ),
            target_maximum_uniform_multiple=target_multiple,
            source_ess_minimum_fraction_of_legal_sources=_probability(
                payload["source_ess_minimum_fraction_of_legal_sources"],
                name="exposure.source_ess_minimum_fraction_of_legal_sources",
                positive=True,
            ),
            source_maximum_uniform_multiple=source_multiple,
            source_top5_maximum_share=top5,
            source_top10_maximum_share=top10,
            candidate_marginal_proposal=P0CandidateProposalConfig.from_mapping(
                payload["candidate_marginal_proposal"]
            ),
        )


@dataclass(frozen=True)
class P0DiagnosticConfig:
    schema_version: str
    dataset: str
    split: str
    input_binding: P0InputBinding
    geometry: P0GeometryConfig
    overlap: P0OverlapConfig
    separability: P0SeparabilityConfig
    exposure: P0ExposureConfig

    @classmethod
    def from_mapping(cls, value: object) -> "P0DiagnosticConfig":
        payload = _mapping(
            value,
            {
                "schema_version",
                "dataset",
                "split",
                "input_binding",
                "geometry",
                "overlap",
                "separability",
                "exposure",
            },
            name="P0 config",
        )
        if payload["schema_version"] != P0_CONFIG_SCHEMA:
            raise ValueError("unsupported P0 config schema")
        if not isinstance(payload["dataset"], str) or not payload["dataset"]:
            raise ValueError("P0 dataset must be non-empty")
        if payload["split"] != "D_R":
            raise ValueError("P0 diagnostics permit only D_R")
        return cls(
            schema_version=payload["schema_version"],
            dataset=payload["dataset"],
            split=payload["split"],
            input_binding=P0InputBinding.from_mapping(payload["input_binding"]),
            geometry=P0GeometryConfig.from_mapping(payload["geometry"]),
            overlap=P0OverlapConfig.from_mapping(payload["overlap"]),
            separability=P0SeparabilityConfig.from_mapping(
                payload["separability"]
            ),
            exposure=P0ExposureConfig.from_mapping(payload["exposure"]),
        )

    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["geometry"]["expected_native_size"] = list(
            self.geometry.expected_native_size
        )
        payload["geometry"]["expected_evaluation_size"] = list(
            self.geometry.expected_evaluation_size
        )
        payload["overlap"]["handcrafted_descriptor_fields"] = list(
            self.overlap.handcrafted_descriptor_fields
        )
        payload["separability"]["bootstrap_interval"] = list(
            self.separability.bootstrap_interval
        )
        payload["separability"]["mmd_kernel_scales"] = list(
            self.separability.mmd_kernel_scales
        )
        payload["exposure"]["seeds"] = list(self.exposure.seeds)
        return payload

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def load_p0_config(path: str | Path) -> P0DiagnosticConfig:
    """Load one strict, duplicate-key-free P0 JSON configuration."""

    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("P0 config may not be a symbolic link")
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("P0 config must be a regular file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"P0 config contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"P0 config contains non-finite number {value}")

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    config = P0DiagnosticConfig.from_mapping(payload)
    if config.canonical_payload() != payload:
        raise ValueError("P0 config JSON is not canonical")
    return config


__all__ = [
    "P0_CONFIG_SCHEMA",
    "P0CandidateProposalConfig",
    "P0DiagnosticConfig",
    "P0ExposureConfig",
    "P0GeometryConfig",
    "P0InputBinding",
    "P0OverlapConfig",
    "P0SeparabilityConfig",
    "load_p0_config",
]
