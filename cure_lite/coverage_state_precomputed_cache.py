"""Training-ready scalar CSLF geometry over the frozen raw population.

The raw catalog remains representation neutral.  This module is the first
representation-specific layer: it accepts only an observability receipt that
authorized ``scalar_max`` and precomputes every target field and integration
measure needed by the three matched objectives.

No image, label, matcher, component extractor, distance transform, or target
builder is called by the later training step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor

from .cache.schema import stable_fingerprint
from .coverage_state_level_set import (
    CSLF_FEATURE_POLICY,
    CSLF_NUMERICAL_POLICY,
    CSLF_TARGET_POLICY,
    normalize_cslf_feature,
)
from .coverage_state_observability import (
    CoverageStateObservabilityDecision,
    CoverageStatePairObservabilityAudit,
    CoverageStatePopulationObservabilityReceipt,
    actual_input_fingerprint,
    occupancy_to_scalar_grid,
)
from .coverage_state_raw_catalog import (
    COVERAGE_STATE_NATURAL_FOCUS_POLICY,
    COVERAGE_STATE_SCENE_TARGET_POLICY,
    CoverageStateNaturalRecord,
    CoverageStatePairRecord,
    CoverageStateRawCatalog,
)
from .coverage_state_sobolev import (
    CoverageStateAbsoluteTargets,
    CoverageStatePairTargets,
    CoverageStateSceneFieldTargets,
    CoverageStateSobolevConfig,
    prepare_coverage_state_focused_absolute_targets_from_scene,
    prepare_coverage_state_pair_targets,
    prepare_coverage_state_scene_field_targets,
)
from .paired_types import tensor_content_fingerprint


COVERAGE_STATE_SCALAR_CACHE_SCHEMA = (
    "cure-lite-scalar-coverage-state-precomputed-cache-v1"
)
COVERAGE_STATE_SCALAR_REPRESENTATION = "scalar_max"
COVERAGE_STATE_ENDPOINT_MEASURE_POLICY = (
    "separable_endpoint_writable_domain_absolute_measure_v1"
)
COVERAGE_STATE_PAIR_OPTIMIZER_ROLES = (
    "clean_positive",
    "component_null",
    "diagnostic_only",
    "identity_diagnostic",
)


def _record_fingerprint(
    record: CoverageStateNaturalRecord | CoverageStatePairRecord,
) -> str:
    return stable_fingerprint(record.canonical_payload())


def _config_payload(
    config: CoverageStateSobolevConfig,
) -> dict[str, object]:
    if not isinstance(config, CoverageStateSobolevConfig):
        raise TypeError("config must be CoverageStateSobolevConfig")
    return {
        "truncation_radius": config.truncation_radius,
        "field_amplitude_hex": config.field_amplitude.hex(),
        "norm_order": config.norm_order,
        "norm_epsilon_hex": config.norm_epsilon.hex(),
        "objective_policy": config.objective_policy,
        "measure_policy": config.measure_policy,
        "target_policy": CSLF_TARGET_POLICY,
        "feature_policy": CSLF_FEATURE_POLICY,
        "numerical_policy": CSLF_NUMERICAL_POLICY,
        "scene_target_policy": COVERAGE_STATE_SCENE_TARGET_POLICY,
        "natural_focus_policy": COVERAGE_STATE_NATURAL_FOCUS_POLICY,
        "endpoint_measure_policy": COVERAGE_STATE_ENDPOINT_MEASURE_POLICY,
    }


def _absolute_targets_payload(
    targets: CoverageStateAbsoluteTargets,
) -> dict[str, str]:
    targets.validate()
    return {
        "target_field": tensor_content_fingerprint(targets.target_field),
        "integration_measure": tensor_content_fingerprint(
            targets.integration_measure
        ),
        "field_valid_mask": tensor_content_fingerprint(
            targets.field_valid_mask
        ),
        "loss_valid_mask": tensor_content_fingerprint(
            targets.loss_valid_mask
        ),
        "focus_support": tensor_content_fingerprint(targets.focus_support),
        "focus_support_field": tensor_content_fingerprint(
            targets.focus_support_field
        ),
    }


def _pair_targets_payload(
    targets: CoverageStatePairTargets,
) -> dict[str, str]:
    targets.validate()
    return {
        "target_field_plus": tensor_content_fingerprint(
            targets.target_field_plus
        ),
        "target_field_minus": tensor_content_fingerprint(
            targets.target_field_minus
        ),
        "focus_support": tensor_content_fingerprint(targets.focus_support),
        "focus_support_field": tensor_content_fingerprint(
            targets.focus_support_field
        ),
        "integration_measure": tensor_content_fingerprint(
            targets.integration_measure
        ),
        "valid_mask": tensor_content_fingerprint(targets.valid_mask),
    }


def _scalar_actual_input_fingerprint(
    feature: Tensor,
    occupancy: Tensor,
    *,
    stride: int,
) -> str:
    feature_size = tuple(int(value) for value in feature.shape[-2:])
    encoded = normalize_cslf_feature(feature)
    projected = occupancy_to_scalar_grid(
        occupancy,
        feature_size=feature_size,
    )
    return actual_input_fingerprint(
        encoded,
        projected,
        representation="scalar_max",
        stride=stride,
    )


@dataclass(frozen=True, eq=False)
class CoverageStateCachedNatural:
    """One raw natural record and its immutable focused field geometry."""

    record: CoverageStateNaturalRecord
    targets: CoverageStateAbsoluteTargets
    raw_record_fingerprint: str
    actual_scalar_input_fingerprint: str
    geometry_fingerprints: dict[str, str]

    def validate(self, *, stride: int) -> None:
        if not isinstance(self.record, CoverageStateNaturalRecord):
            raise TypeError("cached natural record has an invalid type")
        if not isinstance(self.targets, CoverageStateAbsoluteTargets):
            raise TypeError("cached natural targets have an invalid type")
        self.targets.validate()
        if _record_fingerprint(self.record) != self.raw_record_fingerprint:
            raise RuntimeError("cached natural raw record changed")
        if _absolute_targets_payload(self.targets) != self.geometry_fingerprints:
            raise RuntimeError("cached natural target geometry changed")
        if not torch.equal(
            self.targets.field_valid_mask,
            self.record.valid_mask,
        ):
            raise ValueError("natural field-valid mask differs from raw record")
        if not torch.equal(
            self.targets.loss_valid_mask,
            self.record.loss_valid_mask,
        ):
            raise ValueError("natural loss-valid mask differs from raw record")
        expected_focus = self.record.target & self.record.loss_valid_mask
        if not torch.equal(self.targets.focus_support, expected_focus):
            raise ValueError("natural focus support differs from raw record")
        if not bool(
            torch.all(self.targets.target_field[self.record.target] < 0.0)
        ):
            raise ValueError("scene-complete target is not negative in the field")
        actual = _scalar_actual_input_fingerprint(
            self.record.feature,
            self.record.occupancy,
            stride=stride,
        )
        if actual != self.actual_scalar_input_fingerprint:
            raise RuntimeError("cached natural actual input changed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "record_id": self.record.record_id,
            "sample_id": self.record.sample_id,
            "state_kind": self.record.state_kind,
            "raw_record_fingerprint": self.raw_record_fingerprint,
            "actual_scalar_input_fingerprint": (
                self.actual_scalar_input_fingerprint
            ),
            "geometry": dict(sorted(self.geometry_fingerprints.items())),
        }


@dataclass(frozen=True, eq=False)
class CoverageStateCachedPair:
    """One raw pair with joint and separable endpoint target geometries."""

    record: CoverageStatePairRecord
    joint_targets: CoverageStatePairTargets
    absolute_targets_plus: CoverageStateAbsoluteTargets
    absolute_targets_minus: CoverageStateAbsoluteTargets
    scalar_visible: bool
    optimizer_role: str
    raw_record_fingerprint: str
    actual_input_plus_fingerprint: str
    actual_input_minus_fingerprint: str
    joint_geometry_fingerprints: dict[str, str]
    absolute_plus_fingerprints: dict[str, str]
    absolute_minus_fingerprints: dict[str, str]

    @property
    def optimization_eligible(self) -> bool:
        return self.optimizer_role in {
            "clean_positive",
            "component_null",
        }

    def validate(self, *, stride: int) -> None:
        if not isinstance(self.record, CoverageStatePairRecord):
            raise TypeError("cached pair record has an invalid type")
        if self.optimizer_role not in COVERAGE_STATE_PAIR_OPTIMIZER_ROLES:
            raise ValueError("cached pair has an unknown optimizer role")
        self.joint_targets.validate()
        self.absolute_targets_plus.validate()
        self.absolute_targets_minus.validate()
        if _record_fingerprint(self.record) != self.raw_record_fingerprint:
            raise RuntimeError("cached pair raw record changed")
        if (
            _pair_targets_payload(self.joint_targets)
            != self.joint_geometry_fingerprints
            or _absolute_targets_payload(self.absolute_targets_plus)
            != self.absolute_plus_fingerprints
            or _absolute_targets_payload(self.absolute_targets_minus)
            != self.absolute_minus_fingerprints
        ):
            raise RuntimeError("cached pair target geometry changed")
        kind = self.record.pair_kind
        expected_role = (
            "clean_positive"
            if kind == "clean_positive" and self.scalar_visible
            else "component_null"
            if kind == "component_null" and self.scalar_visible
            else "diagnostic_only"
            if kind == "component_null"
            else "identity_diagnostic"
            if kind == "identity_null"
            else ""
        )
        if not expected_role or self.optimizer_role != expected_role:
            raise ValueError("pair kind, visibility, and optimizer role differ")
        if kind == "clean_positive" and not self.scalar_visible:
            raise ValueError(
                "scalar-authorized cache cannot contain a hidden clean pair"
            )
        if not torch.equal(
            self.absolute_targets_plus.loss_valid_mask,
            self.record.valid_mask & ~self.record.occupancy_plus,
        ) or not torch.equal(
            self.absolute_targets_minus.loss_valid_mask,
            self.record.valid_mask & ~self.record.occupancy_minus,
        ):
            raise ValueError("separable endpoint writable domains changed")
        actual_plus = _scalar_actual_input_fingerprint(
            self.record.feature,
            self.record.occupancy_plus,
            stride=stride,
        )
        actual_minus = _scalar_actual_input_fingerprint(
            self.record.feature,
            self.record.occupancy_minus,
            stride=stride,
        )
        if (
            actual_plus != self.actual_input_plus_fingerprint
            or actual_minus != self.actual_input_minus_fingerprint
        ):
            raise RuntimeError("cached pair actual input changed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "pair_id": self.record.pair_id,
            "sample_id": self.record.sample_id,
            "pair_kind": self.record.pair_kind,
            "scalar_visible": self.scalar_visible,
            "optimizer_role": self.optimizer_role,
            "raw_record_fingerprint": self.raw_record_fingerprint,
            "actual_input_plus_fingerprint": (
                self.actual_input_plus_fingerprint
            ),
            "actual_input_minus_fingerprint": (
                self.actual_input_minus_fingerprint
            ),
            "joint_geometry": dict(
                sorted(self.joint_geometry_fingerprints.items())
            ),
            "absolute_plus": dict(
                sorted(self.absolute_plus_fingerprints.items())
            ),
            "absolute_minus": dict(
                sorted(self.absolute_minus_fingerprints.items())
            ),
        }


@dataclass(frozen=True, eq=False)
class CoverageStateScalarCache:
    """Complete scalar-specific target cache for one frozen raw population."""

    raw_catalog: CoverageStateRawCatalog
    observability: CoverageStatePopulationObservabilityReceipt
    sobolev_config: CoverageStateSobolevConfig
    natural_records: tuple[CoverageStateCachedNatural, ...]
    pair_records: tuple[CoverageStateCachedPair, ...]
    raw_catalog_fingerprint: str
    observability_receipt_fingerprint: str
    sobolev_config_fingerprint: str

    @property
    def clean_positive_records(self) -> tuple[CoverageStateCachedPair, ...]:
        return tuple(
            value
            for value in self.pair_records
            if value.optimizer_role == "clean_positive"
        )

    @property
    def component_null_records(self) -> tuple[CoverageStateCachedPair, ...]:
        return tuple(
            value
            for value in self.pair_records
            if value.optimizer_role == "component_null"
        )

    @property
    def diagnostic_pair_records(self) -> tuple[CoverageStateCachedPair, ...]:
        return tuple(
            value
            for value in self.pair_records
            if not value.optimization_eligible
        )

    @cached_property
    def cache_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_SCALAR_CACHE_SCHEMA,
            "dataset": self.raw_catalog.dataset,
            "split": self.raw_catalog.split,
            "representation": COVERAGE_STATE_SCALAR_REPRESENTATION,
            "raw_catalog_fingerprint": self.raw_catalog_fingerprint,
            "observability_receipt_fingerprint": (
                self.observability_receipt_fingerprint
            ),
            "sobolev_config_fingerprint": (
                self.sobolev_config_fingerprint
            ),
            "sobolev_config": _config_payload(self.sobolev_config),
            "counts": {
                "natural_total": len(self.natural_records),
                "pair_total": len(self.pair_records),
                "clean_positive_optimization_eligible": len(
                    self.clean_positive_records
                ),
                "component_null_total": sum(
                    value.record.pair_kind == "component_null"
                    for value in self.pair_records
                ),
                "component_null_optimization_eligible": len(
                    self.component_null_records
                ),
                "component_null_diagnostic_only": sum(
                    value.record.pair_kind == "component_null"
                    and value.optimizer_role == "diagnostic_only"
                    for value in self.pair_records
                ),
                "identity_null_diagnostic": sum(
                    value.optimizer_role == "identity_diagnostic"
                    for value in self.pair_records
                ),
            },
            "natural_records": [
                value.canonical_payload() for value in self.natural_records
            ],
            "pair_records": [
                value.canonical_payload() for value in self.pair_records
            ],
        }

    def verify_unchanged(self) -> None:
        if self.raw_catalog.catalog_fingerprint != self.raw_catalog_fingerprint:
            raise RuntimeError("raw catalog changed after scalar cache creation")
        if (
            self.observability.receipt_fingerprint
            != self.observability_receipt_fingerprint
        ):
            raise RuntimeError(
                "observability receipt changed after scalar cache creation"
            )
        if stable_fingerprint(
            _config_payload(self.sobolev_config)
        ) != self.sobolev_config_fingerprint:
            raise RuntimeError("Sobolev config changed after cache creation")
        for value in self.natural_records:
            value.validate(stride=self.raw_catalog.feature_stride)
        for value in self.pair_records:
            value.validate(stride=self.raw_catalog.feature_stride)


def _scene_key(
    target: Tensor,
    valid_mask: Tensor,
) -> tuple[str, str]:
    return (
        tensor_content_fingerprint(target),
        tensor_content_fingerprint(valid_mask),
    )


def _absolute_key(
    scene_key: tuple[str, str],
    loss_valid_mask: Tensor,
) -> tuple[str, str, str]:
    return (
        scene_key[0],
        scene_key[1],
        tensor_content_fingerprint(loss_valid_mask),
    )


def prepare_scalar_coverage_state_cache(
    catalog: CoverageStateRawCatalog,
    observability: CoverageStatePopulationObservabilityReceipt,
    config: CoverageStateSobolevConfig,
) -> CoverageStateScalarCache:
    """Precompute the complete scalar CSLF geometry after representation choice."""

    if not isinstance(catalog, CoverageStateRawCatalog):
        raise TypeError("catalog must be CoverageStateRawCatalog")
    if not isinstance(
        observability,
        CoverageStatePopulationObservabilityReceipt,
    ):
        raise TypeError(
            "observability must be a population observability receipt"
        )
    if not isinstance(config, CoverageStateSobolevConfig):
        raise TypeError("config must be CoverageStateSobolevConfig")
    if observability.raw_catalog_fingerprint != catalog.catalog_fingerprint:
        raise ValueError("observability receipt and raw catalog differ")
    if (
        observability.decision
        is not CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
        or not observability.scalar_authorized
        or observability.pp_authorized
    ):
        raise PermissionError(
            "scalar cache requires AUTHORIZE_SCALAR_CSLF"
        )
    if (
        config.truncation_radius != catalog.feature_stride
        or observability.scalar_duplicate_input_target_conflicts != 0
        or observability.target_response_outside_scalar_rf_pixels != 0
        or observability.identity_null_nonidentical_count != 0
    ):
        raise ValueError("scalar cache authorization conditions changed")
    if (
        observability.natural_record_count != len(catalog.natural_records)
        or observability.pair_record_count != len(catalog.pair_records)
    ):
        raise ValueError("observability population counts differ from catalog")

    audits = {value.pair_id: value for value in observability.pair_audits}
    if set(audits) != {value.pair_id for value in catalog.pair_records}:
        raise ValueError("observability pair universe differs from raw catalog")

    scene_cache: dict[
        tuple[str, str],
        CoverageStateSceneFieldTargets,
    ] = {}
    absolute_cache: dict[
        tuple[str, str, str],
        CoverageStateAbsoluteTargets,
    ] = {}

    def scene_geometry(
        target: Tensor,
        valid_mask: Tensor,
    ) -> tuple[
        tuple[str, str],
        CoverageStateSceneFieldTargets,
    ]:
        key = _scene_key(target, valid_mask)
        value = scene_cache.get(key)
        if value is None:
            value = prepare_coverage_state_scene_field_targets(
                target,
                valid_mask,
                config=config,
            )
            scene_cache[key] = value
        return key, value

    def absolute_geometry(
        target: Tensor,
        field_valid_mask: Tensor,
        loss_valid_mask: Tensor,
    ) -> CoverageStateAbsoluteTargets:
        key, scene = scene_geometry(target, field_valid_mask)
        absolute_key = _absolute_key(key, loss_valid_mask)
        value = absolute_cache.get(absolute_key)
        if value is None:
            value = (
                prepare_coverage_state_focused_absolute_targets_from_scene(
                    scene,
                    loss_valid_mask,
                    config=config,
                )
            )
            absolute_cache[absolute_key] = value
        return value

    naturals: list[CoverageStateCachedNatural] = []
    for record in catalog.natural_records:
        targets = absolute_geometry(
            record.target,
            record.valid_mask,
            record.loss_valid_mask,
        )
        cached = CoverageStateCachedNatural(
            record=record,
            targets=targets,
            raw_record_fingerprint=_record_fingerprint(record),
            actual_scalar_input_fingerprint=(
                _scalar_actual_input_fingerprint(
                    record.feature,
                    record.occupancy,
                    stride=catalog.feature_stride,
                )
            ),
            geometry_fingerprints=_absolute_targets_payload(targets),
        )
        cached.validate(stride=catalog.feature_stride)
        naturals.append(cached)

    pairs: list[CoverageStateCachedPair] = []
    for record in catalog.pair_records:
        audit: CoverageStatePairObservabilityAudit = audits[record.pair_id]
        scalar_visible = bool(audit.scalar.changed_feature_cells)
        if record.pair_kind == "clean_positive":
            role = "clean_positive" if scalar_visible else ""
        elif record.pair_kind == "component_null":
            role = "component_null" if scalar_visible else "diagnostic_only"
        else:
            role = "identity_diagnostic"
        if not role:
            raise ValueError(
                "scalar-authorized catalog contains a hidden clean pair"
            )
        joint = prepare_coverage_state_pair_targets(
            record.occupancy_plus,
            record.occupancy_minus,
            record.target_plus,
            record.target_minus,
            record.valid_mask,
            config=config,
        )
        absolute_plus = absolute_geometry(
            record.target_plus,
            record.valid_mask,
            record.valid_mask & ~record.occupancy_plus,
        )
        absolute_minus = absolute_geometry(
            record.target_minus,
            record.valid_mask,
            record.valid_mask & ~record.occupancy_minus,
        )
        cached = CoverageStateCachedPair(
            record=record,
            joint_targets=joint,
            absolute_targets_plus=absolute_plus,
            absolute_targets_minus=absolute_minus,
            scalar_visible=scalar_visible,
            optimizer_role=role,
            raw_record_fingerprint=_record_fingerprint(record),
            actual_input_plus_fingerprint=audit.scalar.input_plus_sha256,
            actual_input_minus_fingerprint=audit.scalar.input_minus_sha256,
            joint_geometry_fingerprints=_pair_targets_payload(joint),
            absolute_plus_fingerprints=_absolute_targets_payload(
                absolute_plus
            ),
            absolute_minus_fingerprints=_absolute_targets_payload(
                absolute_minus
            ),
        )
        cached.validate(stride=catalog.feature_stride)
        pairs.append(cached)

    result = CoverageStateScalarCache(
        raw_catalog=catalog,
        observability=observability,
        sobolev_config=config,
        natural_records=tuple(naturals),
        pair_records=tuple(pairs),
        raw_catalog_fingerprint=catalog.catalog_fingerprint,
        observability_receipt_fingerprint=(
            observability.receipt_fingerprint
        ),
        sobolev_config_fingerprint=stable_fingerprint(
            _config_payload(config)
        ),
    )
    result.verify_unchanged()
    if not result.clean_positive_records or not result.component_null_records:
        raise ValueError(
            "scalar optimizer requires clean and visible component-null pools"
        )
    return result


__all__ = [
    "COVERAGE_STATE_ENDPOINT_MEASURE_POLICY",
    "COVERAGE_STATE_PAIR_OPTIMIZER_ROLES",
    "COVERAGE_STATE_SCALAR_CACHE_SCHEMA",
    "COVERAGE_STATE_SCALAR_REPRESENTATION",
    "CoverageStateCachedNatural",
    "CoverageStateCachedPair",
    "CoverageStateScalarCache",
    "prepare_scalar_coverage_state_cache",
]
