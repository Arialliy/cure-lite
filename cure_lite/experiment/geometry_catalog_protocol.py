"""Strict D_R-only protocol for the geometry-safe P0-v2 sidecar catalog.

This protocol does not reinterpret the frozen P0-v1 result.  It separates the
descriptive, dataset-wide resize audit (A0) from the formal eligibility gate
for the factual/legal analysis populations (A1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from ..cache.schema import stable_fingerprint


GEOMETRY_CATALOG_CONFIG_SCHEMA = "cure-lite-geometry-safe-p0-config-v2"
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


def _size(value: object, *, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item JSON list")
    return (
        _positive_int(value[0], name=f"{name}[0]"),
        _positive_int(value[1], name=f"{name}[1]"),
    )


def _true(value: object, *, name: str) -> bool:
    if value is not True:
        raise ValueError(f"{name} must be true")
    return True


def _false(value: object, *, name: str) -> bool:
    if value is not False:
        raise ValueError(f"{name} must be false")
    return False


@dataclass(frozen=True)
class GeometryPredecessorBinding:
    predecessor_schema_version: str
    predecessor_config_fingerprint: str
    predecessor_p0_a_receipt_fingerprint: str
    predecessor_p0_a_file_sha256: str
    predecessor_decision_fingerprint: str
    predecessor_complete_fingerprint: str
    predecessor_complete_file_sha256: str
    predecessor_p0_a_pass: bool
    predecessor_next_route: str
    relationship: str
    v1_outcome_remains_valid: bool
    result_reinterpretation: bool
    geometry_thresholds_changed: bool
    hardcoded_target_exclusions_added: bool
    d_v_metrics_consulted_for_revision: bool

    @classmethod
    def from_mapping(cls, value: object) -> "GeometryPredecessorBinding":
        fields = {
            "predecessor_schema_version",
            "predecessor_config_fingerprint",
            "predecessor_p0_a_receipt_fingerprint",
            "predecessor_p0_a_file_sha256",
            "predecessor_decision_fingerprint",
            "predecessor_complete_fingerprint",
            "predecessor_complete_file_sha256",
            "predecessor_p0_a_pass",
            "predecessor_next_route",
            "relationship",
            "v1_outcome_remains_valid",
            "result_reinterpretation",
            "geometry_thresholds_changed",
            "hardcoded_target_exclusions_added",
            "d_v_metrics_consulted_for_revision",
        }
        payload = _mapping(value, fields, name="predecessor")
        expected = {
            "predecessor_schema_version": "cure-lite-p0-diagnostic-config-v1",
            "predecessor_next_route": "rebuild_synthetic_target_extraction",
            "relationship": "new-protocol-scope-split-not-v1-reinterpretation",
        }
        for name, required in expected.items():
            if payload[name] != required:
                raise ValueError(f"predecessor.{name} must be {required!r}")
        return cls(
            predecessor_schema_version=payload["predecessor_schema_version"],
            predecessor_config_fingerprint=_digest(
                payload["predecessor_config_fingerprint"],
                name="predecessor.predecessor_config_fingerprint",
            ),
            predecessor_p0_a_receipt_fingerprint=_digest(
                payload["predecessor_p0_a_receipt_fingerprint"],
                name="predecessor.predecessor_p0_a_receipt_fingerprint",
            ),
            predecessor_p0_a_file_sha256=_digest(
                payload["predecessor_p0_a_file_sha256"],
                name="predecessor.predecessor_p0_a_file_sha256",
            ),
            predecessor_decision_fingerprint=_digest(
                payload["predecessor_decision_fingerprint"],
                name="predecessor.predecessor_decision_fingerprint",
            ),
            predecessor_complete_fingerprint=_digest(
                payload["predecessor_complete_fingerprint"],
                name="predecessor.predecessor_complete_fingerprint",
            ),
            predecessor_complete_file_sha256=_digest(
                payload["predecessor_complete_file_sha256"],
                name="predecessor.predecessor_complete_file_sha256",
            ),
            predecessor_p0_a_pass=_false(
                payload["predecessor_p0_a_pass"],
                name="predecessor.predecessor_p0_a_pass",
            ),
            predecessor_next_route=payload["predecessor_next_route"],
            relationship=payload["relationship"],
            v1_outcome_remains_valid=_true(
                payload["v1_outcome_remains_valid"],
                name="predecessor.v1_outcome_remains_valid",
            ),
            result_reinterpretation=_false(
                payload["result_reinterpretation"],
                name="predecessor.result_reinterpretation",
            ),
            geometry_thresholds_changed=_false(
                payload["geometry_thresholds_changed"],
                name="predecessor.geometry_thresholds_changed",
            ),
            hardcoded_target_exclusions_added=_false(
                payload["hardcoded_target_exclusions_added"],
                name="predecessor.hardcoded_target_exclusions_added",
            ),
            d_v_metrics_consulted_for_revision=_false(
                payload["d_v_metrics_consulted_for_revision"],
                name="predecessor.d_v_metrics_consulted_for_revision",
            ),
        )


@dataclass(frozen=True)
class GeometryCatalogInputBinding:
    manifest_file_sha256: str
    state_index_sha256: str
    state_index_fingerprint: str
    base_fingerprint: str
    base_state_fingerprint: str
    state_fingerprint: str
    gt_fingerprint: str

    @classmethod
    def from_mapping(cls, value: object) -> "GeometryCatalogInputBinding":
        fields = {
            "manifest_file_sha256",
            "state_index_sha256",
            "state_index_fingerprint",
            "base_fingerprint",
            "base_state_fingerprint",
            "state_fingerprint",
            "gt_fingerprint",
        }
        payload = _mapping(value, fields, name="input_binding")
        return cls(
            **{
                key: _digest(item, name=f"input_binding.{key}")
                for key, item in payload.items()
            }
        )


@dataclass(frozen=True)
class GeometryExecutionPolicy:
    allowed_runtime_splits: tuple[str, ...]
    create_only_output: bool
    allow_training: bool
    allow_calibration: bool
    allow_inference: bool
    allow_d_v_evaluation: bool
    allow_backbone_integration: bool

    @classmethod
    def from_mapping(cls, value: object) -> "GeometryExecutionPolicy":
        fields = {
            "allowed_runtime_splits",
            "create_only_output",
            "allow_training",
            "allow_calibration",
            "allow_inference",
            "allow_d_v_evaluation",
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
            allow_training=_false(
                payload["allow_training"],
                name="execution_policy.allow_training",
            ),
            allow_calibration=_false(
                payload["allow_calibration"],
                name="execution_policy.allow_calibration",
            ),
            allow_inference=_false(
                payload["allow_inference"],
                name="execution_policy.allow_inference",
            ),
            allow_d_v_evaluation=_false(
                payload["allow_d_v_evaluation"],
                name="execution_policy.allow_d_v_evaluation",
            ),
            allow_backbone_integration=_false(
                payload["allow_backbone_integration"],
                name="execution_policy.allow_backbone_integration",
            ),
        )


@dataclass(frozen=True)
class GeometryTransformConfig:
    expected_native_size: tuple[int, int]
    expected_evaluation_size: tuple[int, int]
    foreground_rule: str
    resize_rule: str
    connectivity: int
    min_component_area: int
    centroid_coordinate_rule: str
    lineage_rule: str
    require_exact_component_projection: bool
    area_ratio_min_inclusive: float
    area_ratio_max_inclusive: float
    centroid_shift_max_evaluation_px_inclusive: float
    require_synthetic_positive_equals_evaluation_target: bool
    geometry_quantization: int

    @classmethod
    def from_mapping(cls, value: object) -> "GeometryTransformConfig":
        fields = {
            "expected_native_size",
            "expected_evaluation_size",
            "foreground_rule",
            "resize_rule",
            "connectivity",
            "min_component_area",
            "centroid_coordinate_rule",
            "lineage_rule",
            "require_exact_component_projection",
            "area_ratio_min_inclusive",
            "area_ratio_max_inclusive",
            "centroid_shift_max_evaluation_px_inclusive",
            "require_synthetic_positive_equals_evaluation_target",
            "geometry_quantization",
        }
        payload = _mapping(value, fields, name="geometry")
        expected = {
            "foreground_rule": "uint8-positive-v1",
            "resize_rule": "pil-nearest-v1",
            "centroid_coordinate_rule": "pixel-center-v1",
            "lineage_rule": "bidirectional-one-to-one-native-evaluation-v1",
        }
        for name, required in expected.items():
            if payload[name] != required:
                raise ValueError(f"geometry.{name} must be {required!r}")
        connectivity = _positive_int(
            payload["connectivity"], name="geometry.connectivity"
        )
        min_area = _positive_int(
            payload["min_component_area"], name="geometry.min_component_area"
        )
        if connectivity != 8 or min_area != 1:
            raise ValueError("geometry must use 8-connectivity and min-area 1")
        minimum = _finite(
            payload["area_ratio_min_inclusive"],
            name="geometry.area_ratio_min_inclusive",
        )
        maximum = _finite(
            payload["area_ratio_max_inclusive"],
            name="geometry.area_ratio_max_inclusive",
        )
        centroid = _finite(
            payload["centroid_shift_max_evaluation_px_inclusive"],
            name="geometry.centroid_shift_max_evaluation_px_inclusive",
        )
        if minimum <= 0.0 or maximum < minimum or centroid < 0.0:
            raise ValueError("geometry thresholds are invalid")
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
            centroid_coordinate_rule=payload["centroid_coordinate_rule"],
            lineage_rule=payload["lineage_rule"],
            require_exact_component_projection=_true(
                payload["require_exact_component_projection"],
                name="geometry.require_exact_component_projection",
            ),
            area_ratio_min_inclusive=minimum,
            area_ratio_max_inclusive=maximum,
            centroid_shift_max_evaluation_px_inclusive=centroid,
            require_synthetic_positive_equals_evaluation_target=_true(
                payload[
                    "require_synthetic_positive_equals_evaluation_target"
                ],
                name=(
                    "geometry."
                    "require_synthetic_positive_equals_evaluation_target"
                ),
            ),
            geometry_quantization=_positive_int(
                payload["geometry_quantization"],
                name="geometry.geometry_quantization",
            ),
        )


@dataclass(frozen=True)
class GeometryPopulationConfig:
    a0_role: str
    a0_population: str
    a0_require_exact_preservation_for_downstream: bool
    factual_candidate_population: str
    legal_candidate_population: str
    group_key: str
    apply_rules_to_roles: tuple[str, ...]
    invalid_candidate_action: str
    hardcoded_identity_exclusions: tuple[object, ...]
    retention_threshold_policy: str
    unreachable_factual_policy: str
    require_complete_candidate_accounting: bool
    require_zero_invalid_retained_targets: bool
    require_zero_duplicate_retained_identities: bool
    require_all_reachable_factual_geometry_eligible: bool
    require_nonempty_factual_eligible: bool
    require_nonempty_legal_eligible: bool

    @classmethod
    def from_mapping(cls, value: object) -> "GeometryPopulationConfig":
        fields = {
            "a0_role",
            "a0_population",
            "a0_require_exact_preservation_for_downstream",
            "factual_candidate_population",
            "legal_candidate_population",
            "group_key",
            "apply_rules_to_roles",
            "invalid_candidate_action",
            "hardcoded_identity_exclusions",
            "retention_threshold_policy",
            "unreachable_factual_policy",
            "require_complete_candidate_accounting",
            "require_zero_invalid_retained_targets",
            "require_zero_duplicate_retained_identities",
            "require_all_reachable_factual_geometry_eligible",
            "require_nonempty_factual_eligible",
            "require_nonempty_legal_eligible",
        }
        payload = _mapping(value, fields, name="analysis_population")
        expected = {
            "a0_role": "descriptive-non-gating",
            "a0_population": "all-native-gt-components",
            "factual_candidate_population": "reachable-factual-misses",
            "legal_candidate_population": (
                "decoder-visible-legal-targets-before-geometry-filter"
            ),
            "group_key": "manifest.group_id",
            "invalid_candidate_action": "exclude-with-complete-receipt-v1",
            "retention_threshold_policy": (
                "report-only-no-posthoc-retention-threshold"
            ),
            "unreachable_factual_policy": (
                "outside-reachable-population-but-mandatory-ledger"
            ),
        }
        for name, required in expected.items():
            if payload[name] != required:
                raise ValueError(
                    f"analysis_population.{name} must be {required!r}"
                )
        if payload["apply_rules_to_roles"] != ["factual", "legal"]:
            raise ValueError(
                "analysis_population.apply_rules_to_roles is not canonical"
            )
        exclusions = payload["hardcoded_identity_exclusions"]
        if not isinstance(exclusions, list) or exclusions:
            raise ValueError(
                "analysis_population hardcoded_identity_exclusions must be empty"
            )
        return cls(
            a0_role=payload["a0_role"],
            a0_population=payload["a0_population"],
            a0_require_exact_preservation_for_downstream=_false(
                payload["a0_require_exact_preservation_for_downstream"],
                name=(
                    "analysis_population."
                    "a0_require_exact_preservation_for_downstream"
                ),
            ),
            factual_candidate_population=payload[
                "factual_candidate_population"
            ],
            legal_candidate_population=payload["legal_candidate_population"],
            group_key=payload["group_key"],
            apply_rules_to_roles=("factual", "legal"),
            invalid_candidate_action=payload["invalid_candidate_action"],
            hardcoded_identity_exclusions=(),
            retention_threshold_policy=payload["retention_threshold_policy"],
            unreachable_factual_policy=payload[
                "unreachable_factual_policy"
            ],
            require_complete_candidate_accounting=_true(
                payload["require_complete_candidate_accounting"],
                name=(
                    "analysis_population."
                    "require_complete_candidate_accounting"
                ),
            ),
            require_zero_invalid_retained_targets=_true(
                payload["require_zero_invalid_retained_targets"],
                name=(
                    "analysis_population."
                    "require_zero_invalid_retained_targets"
                ),
            ),
            require_zero_duplicate_retained_identities=_true(
                payload["require_zero_duplicate_retained_identities"],
                name=(
                    "analysis_population."
                    "require_zero_duplicate_retained_identities"
                ),
            ),
            require_all_reachable_factual_geometry_eligible=_true(
                payload[
                    "require_all_reachable_factual_geometry_eligible"
                ],
                name=(
                    "analysis_population."
                    "require_all_reachable_factual_geometry_eligible"
                ),
            ),
            require_nonempty_factual_eligible=_true(
                payload["require_nonempty_factual_eligible"],
                name=(
                    "analysis_population."
                    "require_nonempty_factual_eligible"
                ),
            ),
            require_nonempty_legal_eligible=_true(
                payload["require_nonempty_legal_eligible"],
                name=(
                    "analysis_population."
                    "require_nonempty_legal_eligible"
                ),
            ),
        )


@dataclass(frozen=True)
class GeometryCatalogProtocol:
    schema_version: str
    protocol_id: str
    dataset: str
    split: str
    predecessor: GeometryPredecessorBinding
    input_binding: GeometryCatalogInputBinding
    execution_policy: GeometryExecutionPolicy
    geometry: GeometryTransformConfig
    analysis_population: GeometryPopulationConfig

    @classmethod
    def from_mapping(cls, value: object) -> "GeometryCatalogProtocol":
        fields = {
            "schema_version",
            "protocol_id",
            "dataset",
            "split",
            "predecessor",
            "input_binding",
            "execution_policy",
            "geometry",
            "analysis_population",
        }
        payload = _mapping(value, fields, name="geometry catalog config")
        if payload["schema_version"] != GEOMETRY_CATALOG_CONFIG_SCHEMA:
            raise ValueError("unsupported geometry catalog config schema")
        if (
            not isinstance(payload["protocol_id"], str)
            or not payload["protocol_id"]
        ):
            raise ValueError("protocol_id must be non-empty")
        if not isinstance(payload["dataset"], str) or not payload["dataset"]:
            raise ValueError("dataset must be non-empty")
        if payload["split"] != "D_R":
            raise ValueError("geometry-safe P0 permits only D_R")
        return cls(
            schema_version=payload["schema_version"],
            protocol_id=payload["protocol_id"],
            dataset=payload["dataset"],
            split="D_R",
            predecessor=GeometryPredecessorBinding.from_mapping(
                payload["predecessor"]
            ),
            input_binding=GeometryCatalogInputBinding.from_mapping(
                payload["input_binding"]
            ),
            execution_policy=GeometryExecutionPolicy.from_mapping(
                payload["execution_policy"]
            ),
            geometry=GeometryTransformConfig.from_mapping(payload["geometry"]),
            analysis_population=GeometryPopulationConfig.from_mapping(
                payload["analysis_population"]
            ),
        )

    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["execution_policy"]["allowed_runtime_splits"] = list(
            self.execution_policy.allowed_runtime_splits
        )
        payload["geometry"]["expected_native_size"] = list(
            self.geometry.expected_native_size
        )
        payload["geometry"]["expected_evaluation_size"] = list(
            self.geometry.expected_evaluation_size
        )
        payload["analysis_population"]["apply_rules_to_roles"] = list(
            self.analysis_population.apply_rules_to_roles
        )
        payload["analysis_population"]["hardcoded_identity_exclusions"] = []
        return payload

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def load_geometry_catalog_protocol(
    path: str | Path,
) -> GeometryCatalogProtocol:
    """Load one strict, duplicate-key-free geometry-safe P0-v2 config."""

    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("geometry config may not be a symbolic link")
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("geometry config must be a regular file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"geometry config contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"geometry config contains non-finite number {value}")

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    config = GeometryCatalogProtocol.from_mapping(payload)
    if config.canonical_payload() != payload:
        raise ValueError("geometry config JSON is not canonical")
    return config


__all__ = [
    "GEOMETRY_CATALOG_CONFIG_SCHEMA",
    "GeometryCatalogInputBinding",
    "GeometryCatalogProtocol",
    "GeometryExecutionPolicy",
    "GeometryPopulationConfig",
    "GeometryPredecessorBinding",
    "GeometryTransformConfig",
    "load_geometry_catalog_protocol",
]
