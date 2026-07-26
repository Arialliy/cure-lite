"""Frozen D_R-only protocol for geometry-safe P0-B/C diagnostics.

This follow-on protocol consumes the already passed geometry-safe P0-A1
population contract.  It does not alter A0/A1, construct a candidate
distribution, train a decoder, inspect D_V/D_T, or authorize Full CURE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ..cache.schema import stable_fingerprint
from .geometry_catalog_protocol import GeometryCatalogInputBinding
from .p0_protocol import P0OverlapConfig, P0SeparabilityConfig


GEOMETRY_SAFE_P0_BC_CONFIG_SCHEMA = (
    "cure-lite-geometry-safe-p0-bc-config-v1"
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


@dataclass(frozen=True)
class GeometrySafeP0BCUpstreamBinding:
    geometry_protocol_file_sha256: str
    geometry_protocol_fingerprint: str
    geometry_catalog_file_sha256: str
    geometry_catalog_fingerprint: str
    p0_a1_file_sha256: str
    p0_a1_receipt_fingerprint: str
    eligible_view_file_sha256: str
    eligible_view_receipt_fingerprint: str
    eligible_catalog_fingerprint: str
    geometry_complete_file_sha256: str
    geometry_complete_fingerprint: str
    p0_v1_config_file_sha256: str
    p0_v1_config_fingerprint: str

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> "GeometrySafeP0BCUpstreamBinding":
        fields = {
            "geometry_protocol_file_sha256",
            "geometry_protocol_fingerprint",
            "geometry_catalog_file_sha256",
            "geometry_catalog_fingerprint",
            "p0_a1_file_sha256",
            "p0_a1_receipt_fingerprint",
            "eligible_view_file_sha256",
            "eligible_view_receipt_fingerprint",
            "eligible_catalog_fingerprint",
            "geometry_complete_file_sha256",
            "geometry_complete_fingerprint",
            "p0_v1_config_file_sha256",
            "p0_v1_config_fingerprint",
        }
        payload = _mapping(value, fields, name="upstream_binding")
        return cls(
            **{
                key: _digest(item, name=f"upstream_binding.{key}")
                for key, item in payload.items()
            }
        )


@dataclass(frozen=True)
class GeometrySafeP0BCExecutionPolicy:
    allowed_runtime_splits: tuple[str, ...]
    create_only_output: bool
    allow_training: bool
    allow_calibration: bool
    allow_inference: bool
    allow_d_v_access: bool
    allow_d_t_access: bool
    allow_candidate_s_construction: bool
    allow_backbone_integration: bool

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> "GeometrySafeP0BCExecutionPolicy":
        fields = {
            "allowed_runtime_splits",
            "create_only_output",
            "allow_training",
            "allow_calibration",
            "allow_inference",
            "allow_d_v_access",
            "allow_d_t_access",
            "allow_candidate_s_construction",
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
            allow_d_v_access=_false(
                payload["allow_d_v_access"],
                name="execution_policy.allow_d_v_access",
            ),
            allow_d_t_access=_false(
                payload["allow_d_t_access"],
                name="execution_policy.allow_d_t_access",
            ),
            allow_candidate_s_construction=_false(
                payload["allow_candidate_s_construction"],
                name="execution_policy.allow_candidate_s_construction",
            ),
            allow_backbone_integration=_false(
                payload["allow_backbone_integration"],
                name="execution_policy.allow_backbone_integration",
            ),
        )


@dataclass(frozen=True)
class GeometrySafeP0BCPopulationBinding:
    factual_discovered: int
    factual_unreachable_outside_population: int
    factual_targets: int
    factual_groups: int
    legal_candidates_before_geometry_filter: int
    legal_geometry_excluded: int
    legal_targets: int
    legal_source_images: int
    legal_groups: int
    role_overlap_groups: int
    role_overlap_factual_targets: int
    role_overlap_legal_targets: int
    legal_exclusive_groups: int
    group_key: str

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> "GeometrySafeP0BCPopulationBinding":
        fields = {
            "factual_discovered",
            "factual_unreachable_outside_population",
            "factual_targets",
            "factual_groups",
            "legal_candidates_before_geometry_filter",
            "legal_geometry_excluded",
            "legal_targets",
            "legal_source_images",
            "legal_groups",
            "role_overlap_groups",
            "role_overlap_factual_targets",
            "role_overlap_legal_targets",
            "legal_exclusive_groups",
            "group_key",
        }
        payload = _mapping(value, fields, name="population_binding")
        expected = {
            "factual_discovered": 33,
            "factual_unreachable_outside_population": 1,
            "factual_targets": 32,
            "factual_groups": 24,
            "legal_candidates_before_geometry_filter": 209,
            "legal_geometry_excluded": 3,
            "legal_targets": 206,
            "legal_source_images": 149,
            "legal_groups": 145,
            "role_overlap_groups": 14,
            "role_overlap_factual_targets": 18,
            "role_overlap_legal_targets": 25,
            "legal_exclusive_groups": 131,
            "group_key": "manifest.group_id",
        }
        for field, required in expected.items():
            if payload[field] != required:
                raise ValueError(
                    f"population_binding.{field} must be {required!r}"
                )
        result = cls(
            factual_discovered=_positive_int(
                payload["factual_discovered"],
                name="population_binding.factual_discovered",
            ),
            factual_unreachable_outside_population=_positive_int(
                payload["factual_unreachable_outside_population"],
                name=(
                    "population_binding."
                    "factual_unreachable_outside_population"
                ),
            ),
            factual_targets=_positive_int(
                payload["factual_targets"],
                name="population_binding.factual_targets",
            ),
            factual_groups=_positive_int(
                payload["factual_groups"],
                name="population_binding.factual_groups",
            ),
            legal_candidates_before_geometry_filter=_positive_int(
                payload["legal_candidates_before_geometry_filter"],
                name=(
                    "population_binding."
                    "legal_candidates_before_geometry_filter"
                ),
            ),
            legal_geometry_excluded=_positive_int(
                payload["legal_geometry_excluded"],
                name="population_binding.legal_geometry_excluded",
            ),
            legal_targets=_positive_int(
                payload["legal_targets"],
                name="population_binding.legal_targets",
            ),
            legal_source_images=_positive_int(
                payload["legal_source_images"],
                name="population_binding.legal_source_images",
            ),
            legal_groups=_positive_int(
                payload["legal_groups"],
                name="population_binding.legal_groups",
            ),
            role_overlap_groups=_positive_int(
                payload["role_overlap_groups"],
                name="population_binding.role_overlap_groups",
            ),
            role_overlap_factual_targets=_positive_int(
                payload["role_overlap_factual_targets"],
                name="population_binding.role_overlap_factual_targets",
            ),
            role_overlap_legal_targets=_positive_int(
                payload["role_overlap_legal_targets"],
                name="population_binding.role_overlap_legal_targets",
            ),
            legal_exclusive_groups=_positive_int(
                payload["legal_exclusive_groups"],
                name="population_binding.legal_exclusive_groups",
            ),
            group_key=payload["group_key"],
        )
        if (
            result.legal_targets + result.legal_geometry_excluded
            != result.legal_candidates_before_geometry_filter
            or result.legal_exclusive_groups + result.role_overlap_groups
            != result.legal_groups
        ):
            raise ValueError("population_binding arithmetic is inconsistent")
        return result


@dataclass(frozen=True)
class GeometrySafeP0BCDecisionPolicy:
    p0_b_rule: str
    p0_c_rule: str
    inconclusive_rule: str
    conjunction_rule: str
    pass_route: str
    fail_route: str
    inconclusive_route: str
    requires_matched_geometry_safe_uniform_control: bool
    authorizes_candidate_s_construction: bool
    authorizes_training: bool
    authorizes_d_v_evaluation: bool
    authorizes_full_cure: bool

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> "GeometrySafeP0BCDecisionPolicy":
        fields = {
            "p0_b_rule",
            "p0_c_rule",
            "inconclusive_rule",
            "conjunction_rule",
            "pass_route",
            "fail_route",
            "inconclusive_route",
            "requires_matched_geometry_safe_uniform_control",
            "authorizes_candidate_s_construction",
            "authorizes_training",
            "authorizes_d_v_evaluation",
            "authorizes_full_cure",
        }
        payload = _mapping(value, fields, name="decision_policy")
        expected = {
            "p0_b_rule": (
                "both-handcrafted-and-decoder-joint-knn-coverage-pass"
            ),
            "p0_c_rule": (
                "both-spaces-bootstrap-interval-auc-and-grouped-mmd-"
                "three-valued-screen"
            ),
            "inconclusive_rule": (
                "numeric-or-statistical-noncompletion-or-auc-interval-"
                "overlap-is-inconclusive"
            ),
            "conjunction_rule": (
                "three-valued-conjunction-pass-fail-inconclusive"
            ),
            "pass_route": "eligible_to_design_candidate_s",
            "fail_route": "redesign_synthetic_state",
            "inconclusive_route": "resolve_p0_bc_inconclusive",
        }
        for field, required in expected.items():
            if payload[field] != required:
                raise ValueError(
                    f"decision_policy.{field} must be {required!r}"
                )
        return cls(
            p0_b_rule=payload["p0_b_rule"],
            p0_c_rule=payload["p0_c_rule"],
            inconclusive_rule=payload["inconclusive_rule"],
            conjunction_rule=payload["conjunction_rule"],
            pass_route=payload["pass_route"],
            fail_route=payload["fail_route"],
            inconclusive_route=payload["inconclusive_route"],
            requires_matched_geometry_safe_uniform_control=_true(
                payload["requires_matched_geometry_safe_uniform_control"],
                name=(
                    "decision_policy."
                    "requires_matched_geometry_safe_uniform_control"
                ),
            ),
            authorizes_candidate_s_construction=_false(
                payload["authorizes_candidate_s_construction"],
                name=(
                    "decision_policy."
                    "authorizes_candidate_s_construction"
                ),
            ),
            authorizes_training=_false(
                payload["authorizes_training"],
                name="decision_policy.authorizes_training",
            ),
            authorizes_d_v_evaluation=_false(
                payload["authorizes_d_v_evaluation"],
                name="decision_policy.authorizes_d_v_evaluation",
            ),
            authorizes_full_cure=_false(
                payload["authorizes_full_cure"],
                name="decision_policy.authorizes_full_cure",
            ),
        )


@dataclass(frozen=True)
class GeometrySafeP0BCProtocol:
    schema_version: str
    protocol_id: str
    dataset: str
    split: str
    upstream_binding: GeometrySafeP0BCUpstreamBinding
    input_binding: GeometryCatalogInputBinding
    execution_policy: GeometrySafeP0BCExecutionPolicy
    population_binding: GeometrySafeP0BCPopulationBinding
    overlap: P0OverlapConfig
    separability: P0SeparabilityConfig
    decision_policy: GeometrySafeP0BCDecisionPolicy

    @classmethod
    def from_mapping(cls, value: object) -> "GeometrySafeP0BCProtocol":
        fields = {
            "schema_version",
            "protocol_id",
            "dataset",
            "split",
            "upstream_binding",
            "input_binding",
            "execution_policy",
            "population_binding",
            "overlap",
            "separability",
            "decision_policy",
        }
        payload = _mapping(value, fields, name="geometry-safe P0-B/C config")
        if payload["schema_version"] != GEOMETRY_SAFE_P0_BC_CONFIG_SCHEMA:
            raise ValueError("unsupported geometry-safe P0-B/C config schema")
        if (
            not isinstance(payload["protocol_id"], str)
            or not payload["protocol_id"]
        ):
            raise ValueError("protocol_id must be non-empty")
        if payload["dataset"] != "IRSTD-1K":
            raise ValueError("geometry-safe P0-B/C dataset must be IRSTD-1K")
        if payload["split"] != "D_R":
            raise ValueError("geometry-safe P0-B/C permits only D_R")
        return cls(
            schema_version=payload["schema_version"],
            protocol_id=payload["protocol_id"],
            dataset=payload["dataset"],
            split="D_R",
            upstream_binding=GeometrySafeP0BCUpstreamBinding.from_mapping(
                payload["upstream_binding"]
            ),
            input_binding=GeometryCatalogInputBinding.from_mapping(
                payload["input_binding"]
            ),
            execution_policy=GeometrySafeP0BCExecutionPolicy.from_mapping(
                payload["execution_policy"]
            ),
            population_binding=(
                GeometrySafeP0BCPopulationBinding.from_mapping(
                    payload["population_binding"]
                )
            ),
            overlap=P0OverlapConfig.from_mapping(payload["overlap"]),
            separability=P0SeparabilityConfig.from_mapping(
                payload["separability"]
            ),
            decision_policy=GeometrySafeP0BCDecisionPolicy.from_mapping(
                payload["decision_policy"]
            ),
        )

    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["execution_policy"]["allowed_runtime_splits"] = list(
            self.execution_policy.allowed_runtime_splits
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
        return payload

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def load_geometry_safe_p0_bc_protocol(
    path: str | Path,
) -> GeometrySafeP0BCProtocol:
    """Load one strict, duplicate-key-free geometry-safe P0-B/C config."""

    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("geometry-safe P0-B/C config may not be a symlink")
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("geometry-safe P0-B/C config must be a regular file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    "geometry-safe P0-B/C config contains duplicate key "
                    f"{key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(
            "geometry-safe P0-B/C config contains non-finite number "
            f"{value}"
        )

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    config = GeometrySafeP0BCProtocol.from_mapping(payload)
    if config.canonical_payload() != payload:
        raise ValueError("geometry-safe P0-B/C config JSON is not canonical")
    return config


__all__ = [
    "GEOMETRY_SAFE_P0_BC_CONFIG_SCHEMA",
    "GeometrySafeP0BCDecisionPolicy",
    "GeometrySafeP0BCExecutionPolicy",
    "GeometrySafeP0BCPopulationBinding",
    "GeometrySafeP0BCProtocol",
    "GeometrySafeP0BCUpstreamBinding",
    "load_geometry_safe_p0_bc_protocol",
]
