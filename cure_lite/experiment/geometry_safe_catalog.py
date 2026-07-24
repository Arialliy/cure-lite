"""Native-first lineage audit and geometry-safe P0 diagnostic catalog.

The module is a read-only sidecar over the frozen D_R cache and the legacy
``PreparedTrainingCatalog``.  It never rewrites the evaluation-grid GT,
synthetic state, feature tensor, supervision tensor, or training catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from math import hypot, isfinite
from typing import Any

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..instances import instances_from_binary_mask
from ..splits import SplitManifest
from ..train.pools import StateExample
from ..types import InstanceMap
from .cache_pipeline import LoadedDRCacheBundle
from .geometry_catalog_protocol import (
    GeometryCatalogProtocol,
    GeometryTransformConfig,
)
from .p0_geometry import _native_mask, _resize_mask, _scaled_centroid
from .training_pipeline import (
    PreparedLegalCandidate,
    PreparedTrainingCatalog,
    PreparedTrainingSource,
    TrainingSupportSummary,
)


GEOMETRY_SAFE_CATALOG_SCHEMA = "cure-lite-geometry-safe-catalog-v2"
P0_A0_SCHEMA = "cure-lite-p0-a0-dataset-geometry-audit-v2"
P0_A1_SCHEMA = "cure-lite-p0-a1-population-eligibility-v2"

_GEOMETRY_REASON_ORDER = (
    "multiple_native_ancestors",
    "native_has_multiple_evaluation_descendants",
    "projected_component_mismatch",
    "area_ratio_below_minimum",
    "area_ratio_above_maximum",
    "centroid_shift_above_maximum",
)


def _quantized(value: float, scale: int) -> float:
    return round(float(value) * scale) / scale


def _tensor_content_fingerprint(tensor: Tensor) -> str:
    source = torch.as_tensor(tensor).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(source.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(source.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(source.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class NativeEvaluationTrace:
    """One exact native-component to evaluation-component incidence graph."""

    native: InstanceMap
    evaluation: InstanceMap
    projected_by_native: tuple[tuple[int, Tensor], ...]
    descendants: tuple[tuple[int, tuple[int, ...]], ...]
    ancestors: tuple[tuple[int, tuple[int, ...]], ...]

    def projected_mask(self, native_id: int) -> Tensor:
        for candidate_id, mask in self.projected_by_native:
            if candidate_id == native_id:
                return mask
        raise KeyError(f"unknown native target {native_id}")

    def descendant_ids(self, native_id: int) -> tuple[int, ...]:
        for candidate_id, values in self.descendants:
            if candidate_id == native_id:
                return values
        raise KeyError(f"unknown native target {native_id}")

    def ancestor_ids(self, evaluation_id: int) -> tuple[int, ...]:
        for candidate_id, values in self.ancestors:
            if candidate_id == evaluation_id:
                return values
        raise KeyError(f"unknown evaluation target {evaluation_id}")


def trace_native_to_evaluation(
    native_mask: Tensor,
    evaluation_labels: Tensor,
    config: GeometryTransformConfig,
) -> NativeEvaluationTrace:
    """Construct the frozen Pillow-nearest lineage incidence graph."""

    if not isinstance(config, GeometryTransformConfig):
        raise TypeError("config must be GeometryTransformConfig")
    if (
        not isinstance(native_mask, Tensor)
        or native_mask.dtype != torch.bool
        or native_mask.device.type != "cpu"
        or native_mask.ndim != 2
    ):
        raise TypeError("native_mask must be a two-dimensional CPU bool tensor")
    if tuple(native_mask.shape) != config.expected_native_size:
        raise ValueError("native_mask shape differs from the geometry protocol")
    if (
        not isinstance(evaluation_labels, Tensor)
        or evaluation_labels.dtype != torch.int64
        or evaluation_labels.device.type != "cpu"
        or evaluation_labels.ndim != 2
    ):
        raise TypeError(
            "evaluation_labels must be a two-dimensional CPU int64 tensor"
        )
    if tuple(evaluation_labels.shape) != config.expected_evaluation_size:
        raise ValueError(
            "evaluation_labels shape differs from the geometry protocol"
        )
    if torch.any(evaluation_labels < 0):
        raise ValueError("evaluation_labels cannot contain negative values")

    native = instances_from_binary_mask(
        native_mask,
        connectivity=config.connectivity,
        min_area=config.min_component_area,
    )
    evaluation = instances_from_binary_mask(
        evaluation_labels > 0,
        connectivity=config.connectivity,
        min_area=config.min_component_area,
    )
    if not torch.equal(evaluation.labels, evaluation_labels):
        raise RuntimeError("evaluation labels are not canonical component IDs")

    projected_rows: list[tuple[int, Tensor]] = []
    descendant_rows: list[tuple[int, tuple[int, ...]]] = []
    projected_union = torch.zeros_like(evaluation_labels, dtype=torch.bool)
    projected_count = torch.zeros_like(evaluation_labels, dtype=torch.int16)
    for target in native.instances:
        projected = _resize_mask(
            target.mask,
            config.expected_evaluation_size,
        )
        projected_rows.append((target.instance_id, projected))
        projected_union |= projected
        projected_count += projected.to(torch.int16)
        child_ids = tuple(
            sorted(
                int(value)
                for value in torch.unique(evaluation_labels[projected]).tolist()
                if int(value) > 0
            )
        )
        descendant_rows.append((target.instance_id, child_ids))
    if torch.any(projected_count > 1):
        raise RuntimeError("native component projections overlap")
    if not torch.equal(projected_union, evaluation_labels > 0):
        raise RuntimeError(
            "per-component native projections do not reproduce evaluation GT"
        )

    ancestor_rows: list[tuple[int, tuple[int, ...]]] = []
    for target in evaluation.instances:
        parent_ids = tuple(
            native_id
            for native_id, projected in projected_rows
            if bool(torch.any(projected & target.mask))
        )
        if not parent_ids:
            raise RuntimeError("evaluation target has no native ancestor")
        ancestor_rows.append((target.instance_id, parent_ids))

    descendant_map = dict(descendant_rows)
    ancestor_map = dict(ancestor_rows)
    for native_id, evaluation_ids in descendant_rows:
        for evaluation_id in evaluation_ids:
            if native_id not in ancestor_map[evaluation_id]:
                raise RuntimeError("lineage incidence is not reciprocal")
    for evaluation_id, native_ids in ancestor_rows:
        for native_id in native_ids:
            if evaluation_id not in descendant_map[native_id]:
                raise RuntimeError("lineage incidence is not reciprocal")

    return NativeEvaluationTrace(
        native=native,
        evaluation=evaluation,
        projected_by_native=tuple(projected_rows),
        descendants=tuple(descendant_rows),
        ancestors=tuple(ancestor_rows),
    )


@dataclass(frozen=True)
class NativeLineageRecord:
    native_id: int
    area: int
    centroid: tuple[float, float]
    projected_area: int
    descendant_evaluation_ids: tuple[int, ...]
    projected_mask_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "native_id": self.native_id,
            "area": self.area,
            "centroid": list(self.centroid),
            "projected_area": self.projected_area,
            "descendant_evaluation_ids": list(
                self.descendant_evaluation_ids
            ),
            "projected_mask_fingerprint": self.projected_mask_fingerprint,
        }


@dataclass(frozen=True)
class EvaluationLineageRecord:
    evaluation_id: int
    area: int
    centroid: tuple[float, float]
    ancestor_native_ids: tuple[int, ...]
    evaluation_mask_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "evaluation_id": self.evaluation_id,
            "area": self.area,
            "centroid": list(self.centroid),
            "ancestor_native_ids": list(self.ancestor_native_ids),
            "evaluation_mask_fingerprint": (
                self.evaluation_mask_fingerprint
            ),
        }


@dataclass(frozen=True)
class GeometrySampleAudit:
    sample_id: str
    group_id: str
    mask_file_sha256: str
    native_targets: tuple[NativeLineageRecord, ...]
    evaluation_targets: tuple[EvaluationLineageRecord, ...]
    disappeared_native_ids: tuple[int, ...]
    merged_evaluation_ids: tuple[int, ...]
    split_native_ids: tuple[int, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "mask_file_sha256": self.mask_file_sha256,
            "native_targets": [
                item.canonical_payload() for item in self.native_targets
            ],
            "evaluation_targets": [
                item.canonical_payload() for item in self.evaluation_targets
            ],
            "disappeared_native_ids": list(self.disappeared_native_ids),
            "merged_evaluation_ids": list(self.merged_evaluation_ids),
            "split_native_ids": list(self.split_native_ids),
        }


@dataclass(frozen=True)
class GeometryTargetRecord:
    sample_id: str
    group_id: str
    role: str
    evaluation_gt_id: int
    pred_id: int | None
    candidate_ordinal: int
    analysis_candidate: bool
    native_ancestor_ids: tuple[int, ...]
    native_gt_id: int | None
    native_descendant_evaluation_ids: tuple[int, ...]
    reciprocal_one_to_one: bool
    exact_component_projection: bool
    native_area: int | None
    projected_area: int | None
    evaluation_area: int
    expected_scaled_area: float | None
    area_ratio: float | None
    native_centroid: tuple[float, float] | None
    expected_evaluation_centroid: tuple[float, float] | None
    evaluation_centroid: tuple[float, float]
    centroid_shift_evaluation_px: float | None
    geometry_eligible: bool
    geometry_reason_codes: tuple[str, ...]
    analysis_eligible: bool
    analysis_exclusion_reasons: tuple[str, ...]
    evaluation_mask_fingerprint: str
    projected_mask_fingerprint: str | None
    candidate_occupancy_fingerprint: str | None
    synthetic_occupancy_fingerprint: str | None
    synthetic_target_fingerprint: str | None
    synthetic_valid_mask_fingerprint: str | None
    source_state_content_fingerprint: str

    @property
    def identity(self) -> tuple[str, int, int | None]:
        return self.sample_id, self.evaluation_gt_id, self.pred_id

    def canonical_payload(self) -> dict[str, object]:
        return {
            "identity": list(self.identity),
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "role": self.role,
            "evaluation_gt_id": self.evaluation_gt_id,
            "pred_id": self.pred_id,
            "candidate_ordinal": self.candidate_ordinal,
            "analysis_candidate": self.analysis_candidate,
            "native_ancestor_ids": list(self.native_ancestor_ids),
            "native_gt_id": self.native_gt_id,
            "native_descendant_evaluation_ids": list(
                self.native_descendant_evaluation_ids
            ),
            "reciprocal_one_to_one": self.reciprocal_one_to_one,
            "exact_component_projection": self.exact_component_projection,
            "native_area": self.native_area,
            "projected_area": self.projected_area,
            "evaluation_area": self.evaluation_area,
            "expected_scaled_area": self.expected_scaled_area,
            "area_ratio": self.area_ratio,
            "native_centroid": (
                None
                if self.native_centroid is None
                else list(self.native_centroid)
            ),
            "expected_evaluation_centroid": (
                None
                if self.expected_evaluation_centroid is None
                else list(self.expected_evaluation_centroid)
            ),
            "evaluation_centroid": list(self.evaluation_centroid),
            "centroid_shift_evaluation_px": (
                self.centroid_shift_evaluation_px
            ),
            "geometry_eligible": self.geometry_eligible,
            "geometry_reason_codes": list(self.geometry_reason_codes),
            "analysis_eligible": self.analysis_eligible,
            "analysis_exclusion_reasons": list(
                self.analysis_exclusion_reasons
            ),
            "evaluation_mask_fingerprint": (
                self.evaluation_mask_fingerprint
            ),
            "projected_mask_fingerprint": (
                self.projected_mask_fingerprint
            ),
            "candidate_occupancy_fingerprint": (
                self.candidate_occupancy_fingerprint
            ),
            "synthetic_occupancy_fingerprint": (
                self.synthetic_occupancy_fingerprint
            ),
            "synthetic_target_fingerprint": (
                self.synthetic_target_fingerprint
            ),
            "synthetic_valid_mask_fingerprint": (
                self.synthetic_valid_mask_fingerprint
            ),
            "source_state_content_fingerprint": (
                self.source_state_content_fingerprint
            ),
        }


@dataclass(frozen=True)
class _EvaluationGeometry:
    ancestor_ids: tuple[int, ...]
    native_id: int | None
    descendant_ids: tuple[int, ...]
    reciprocal: bool
    exact_projection: bool
    native_area: int | None
    projected_area: int | None
    evaluation_area: int
    expected_scaled_area: float | None
    area_ratio: float | None
    native_centroid: tuple[float, float] | None
    expected_centroid: tuple[float, float] | None
    evaluation_centroid: tuple[float, float]
    centroid_shift: float | None
    reason_codes: tuple[str, ...]
    evaluation_mask_fingerprint: str
    projected_mask_fingerprint: str | None

    @property
    def eligible(self) -> bool:
        return not self.reason_codes


def _evaluation_geometry(
    trace: NativeEvaluationTrace,
    evaluation_id: int,
    config: GeometryTransformConfig,
) -> _EvaluationGeometry:
    try:
        evaluation_target = trace.evaluation.by_id(evaluation_id)
    except KeyError as error:
        raise RuntimeError(
            f"catalog references unknown evaluation GT {evaluation_id}"
        ) from error
    ancestor_ids = trace.ancestor_ids(evaluation_id)
    if not ancestor_ids:
        raise RuntimeError("evaluation target has no native ancestor")

    native_id: int | None = None
    descendant_ids: tuple[int, ...] = ()
    exact_projection = False
    native_area: int | None = None
    projected_area: int | None = None
    expected_area: float | None = None
    area_ratio: float | None = None
    native_centroid: tuple[float, float] | None = None
    expected_centroid: tuple[float, float] | None = None
    centroid_shift: float | None = None
    projected_fingerprint: str | None = None
    reasons: list[str] = []

    if len(ancestor_ids) > 1:
        reasons.append("multiple_native_ancestors")
    else:
        native_id = ancestor_ids[0]
        descendant_ids = trace.descendant_ids(native_id)
        if len(descendant_ids) > 1:
            reasons.append("native_has_multiple_evaluation_descendants")
        elif descendant_ids != (evaluation_id,):
            raise RuntimeError("one-parent lineage points to an inconsistent child")
        else:
            native_target = trace.native.by_id(native_id)
            projected = trace.projected_mask(native_id)
            projected_fingerprint = _tensor_content_fingerprint(projected)
            exact_projection = torch.equal(
                projected,
                evaluation_target.mask,
            )
            if not exact_projection:
                reasons.append("projected_component_mismatch")
            else:
                native_area = native_target.area
                projected_area = int(torch.count_nonzero(projected))
                native_height, native_width = config.expected_native_size
                evaluation_height, evaluation_width = (
                    config.expected_evaluation_size
                )
                expected_area = native_area * (
                    evaluation_height / native_height
                ) * (evaluation_width / native_width)
                area_ratio = evaluation_target.area / expected_area
                native_centroid = native_target.centroid
                expected_centroid = _scaled_centroid(
                    native_centroid,
                    config.expected_native_size,
                    config.expected_evaluation_size,
                )
                centroid_shift = hypot(
                    evaluation_target.centroid[0] - expected_centroid[0],
                    evaluation_target.centroid[1] - expected_centroid[1],
                )
                if (
                    not isfinite(area_ratio)
                    or area_ratio <= 0.0
                    or not isfinite(centroid_shift)
                ):
                    raise RuntimeError("geometry statistic is not finite positive")
                if area_ratio < config.area_ratio_min_inclusive:
                    reasons.append("area_ratio_below_minimum")
                if area_ratio > config.area_ratio_max_inclusive:
                    reasons.append("area_ratio_above_maximum")
                if (
                    centroid_shift
                    > config.centroid_shift_max_evaluation_px_inclusive
                ):
                    reasons.append("centroid_shift_above_maximum")

    if tuple(
        reason for reason in _GEOMETRY_REASON_ORDER if reason in reasons
    ) != tuple(reasons):
        raise AssertionError("geometry exclusion reasons are not canonical")
    scale = config.geometry_quantization
    return _EvaluationGeometry(
        ancestor_ids=ancestor_ids,
        native_id=native_id,
        descendant_ids=descendant_ids,
        reciprocal=(
            len(ancestor_ids) == 1
            and descendant_ids == (evaluation_id,)
        ),
        exact_projection=exact_projection,
        native_area=native_area,
        projected_area=projected_area,
        evaluation_area=evaluation_target.area,
        expected_scaled_area=(
            None if expected_area is None else _quantized(expected_area, scale)
        ),
        area_ratio=(
            None if area_ratio is None else _quantized(area_ratio, scale)
        ),
        native_centroid=(
            None
            if native_centroid is None
            else tuple(_quantized(value, scale) for value in native_centroid)
        ),
        expected_centroid=(
            None
            if expected_centroid is None
            else tuple(
                _quantized(value, scale) for value in expected_centroid
            )
        ),
        evaluation_centroid=tuple(
            _quantized(value, scale)
            for value in evaluation_target.centroid
        ),
        centroid_shift=(
            None
            if centroid_shift is None
            else _quantized(centroid_shift, scale)
        ),
        reason_codes=tuple(reasons),
        evaluation_mask_fingerprint=_tensor_content_fingerprint(
            evaluation_target.mask
        ),
        projected_mask_fingerprint=projected_fingerprint,
    )


def _target_record(
    *,
    sample_id: str,
    group_id: str,
    role: str,
    evaluation_gt_id: int,
    pred_id: int | None,
    candidate_ordinal: int,
    analysis_candidate: bool,
    geometry: _EvaluationGeometry,
    source_state_content_fingerprint: str,
    candidate: PreparedLegalCandidate | None = None,
    example: StateExample | None = None,
) -> GeometryTargetRecord:
    candidate_occupancy = None
    synthetic_occupancy = None
    synthetic_target = None
    synthetic_valid = None
    if role == "legal":
        if candidate is None or example is None:
            raise TypeError("legal target requires candidate and synthetic example")
        candidate_occupancy = _tensor_content_fingerprint(
            candidate.occupancy_after
        )
        synthetic_occupancy = _tensor_content_fingerprint(
            example.supervision.occupancy
        )
        synthetic_target = _tensor_content_fingerprint(
            example.supervision.target
        )
        synthetic_valid = _tensor_content_fingerprint(
            example.supervision.valid_mask
        )
    elif candidate is not None or example is not None:
        raise TypeError("non-legal target cannot carry a synthetic candidate")

    analysis_reasons = list(geometry.reason_codes)
    if not analysis_candidate:
        analysis_reasons.insert(0, "unreachable_factual_state")
    return GeometryTargetRecord(
        sample_id=sample_id,
        group_id=group_id,
        role=role,
        evaluation_gt_id=evaluation_gt_id,
        pred_id=pred_id,
        candidate_ordinal=candidate_ordinal,
        analysis_candidate=analysis_candidate,
        native_ancestor_ids=geometry.ancestor_ids,
        native_gt_id=geometry.native_id,
        native_descendant_evaluation_ids=geometry.descendant_ids,
        reciprocal_one_to_one=geometry.reciprocal,
        exact_component_projection=geometry.exact_projection,
        native_area=geometry.native_area,
        projected_area=geometry.projected_area,
        evaluation_area=geometry.evaluation_area,
        expected_scaled_area=geometry.expected_scaled_area,
        area_ratio=geometry.area_ratio,
        native_centroid=geometry.native_centroid,
        expected_evaluation_centroid=geometry.expected_centroid,
        evaluation_centroid=geometry.evaluation_centroid,
        centroid_shift_evaluation_px=geometry.centroid_shift,
        geometry_eligible=geometry.eligible,
        geometry_reason_codes=geometry.reason_codes,
        analysis_eligible=analysis_candidate and geometry.eligible,
        analysis_exclusion_reasons=tuple(analysis_reasons),
        evaluation_mask_fingerprint=(
            geometry.evaluation_mask_fingerprint
        ),
        projected_mask_fingerprint=(
            geometry.projected_mask_fingerprint
        ),
        candidate_occupancy_fingerprint=candidate_occupancy,
        synthetic_occupancy_fingerprint=synthetic_occupancy,
        synthetic_target_fingerprint=synthetic_target,
        synthetic_valid_mask_fingerprint=synthetic_valid,
        source_state_content_fingerprint=source_state_content_fingerprint,
    )


def fingerprint_prepared_analysis_population(
    bundle: LoadedDRCacheBundle,
    catalog: PreparedTrainingCatalog,
    manifest: SplitManifest,
) -> str:
    """Fingerprint the exact factual/legal objects consumed by the sidecar."""

    if catalog.source_ids != tuple(row.sample_id for row in bundle.rows):
        raise ValueError("catalog and D_R bundle source identities differ")
    group_by_sample = {
        record.sample_id: record.group_id
        for record in manifest.records_for("D_R")
    }
    if set(group_by_sample) != set(catalog.source_ids):
        raise ValueError("manifest D_R groups differ from the catalog")
    row_by_id = {row.sample_id: row for row in bundle.rows}
    rows: list[dict[str, object]] = []
    for entry in catalog.entries:
        row = row_by_id[entry.sample_id]
        factual = [
            {
                "identity": [entry.sample_id, gt_id, None],
                "target": _tensor_content_fingerprint(
                    example.supervision.target
                ),
                "occupancy": _tensor_content_fingerprint(
                    example.supervision.occupancy
                ),
                "valid_mask": _tensor_content_fingerprint(
                    example.supervision.valid_mask
                ),
            }
            for gt_id, example in zip(
                entry.reachable_gt_ids,
                entry.factual_examples,
                strict=True,
            )
        ]
        legal = [
            {
                "identity": [
                    entry.sample_id,
                    candidate.gt_id,
                    candidate.pred_id,
                ],
                "candidate_occupancy": _tensor_content_fingerprint(
                    candidate.occupancy_after
                ),
                "target": _tensor_content_fingerprint(
                    example.supervision.target
                ),
                "occupancy": _tensor_content_fingerprint(
                    example.supervision.occupancy
                ),
                "valid_mask": _tensor_content_fingerprint(
                    example.supervision.valid_mask
                ),
            }
            for candidate, example in zip(
                entry.decoder_visible_legal_candidates,
                entry.synthetic_examples,
                strict=True,
            )
        ]
        rows.append(
            {
                "sample_id": entry.sample_id,
                "group_id": group_by_sample[entry.sample_id],
                "mask_sha256": row.mask_sha256,
                "state_cache_sha256": row.state_cache_sha256,
                "source_state_content_fingerprint": row.content_fingerprint,
                "real_miss_ids": list(entry.real_miss_ids),
                "reachable_gt_ids": list(entry.reachable_gt_ids),
                "factual": factual,
                "legal": legal,
            }
        )
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-prepared-analysis-population-v2",
            "split": "D_R",
            "rows": rows,
        }
    )


@dataclass(frozen=True)
class GeometrySafeCatalog:
    protocol_fingerprint: str
    source_catalog_fingerprint: str
    sample_audits: tuple[GeometrySampleAudit, ...]
    factual_records: tuple[GeometryTargetRecord, ...]
    legal_records: tuple[GeometryTargetRecord, ...]
    outside_population_records: tuple[GeometryTargetRecord, ...]
    catalog_fingerprint: str

    @property
    def eligible_factual_identities(
        self,
    ) -> tuple[tuple[str, int, int | None], ...]:
        return tuple(
            record.identity
            for record in self.factual_records
            if record.analysis_eligible
        )

    @property
    def eligible_legal_identities(
        self,
    ) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (record.sample_id, record.evaluation_gt_id, int(record.pred_id))
            for record in self.legal_records
            if record.analysis_eligible
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": GEOMETRY_SAFE_CATALOG_SCHEMA,
            "split": "D_R",
            "protocol_fingerprint": self.protocol_fingerprint,
            "source_catalog_fingerprint": self.source_catalog_fingerprint,
            "sample_audits": [
                item.canonical_payload() for item in self.sample_audits
            ],
            "factual_records": [
                item.canonical_payload() for item in self.factual_records
            ],
            "legal_records": [
                item.canonical_payload() for item in self.legal_records
            ],
            "outside_population_records": [
                item.canonical_payload()
                for item in self.outside_population_records
            ],
        }


def _sample_audit(
    *,
    sample_id: str,
    group_id: str,
    mask_sha256: str,
    trace: NativeEvaluationTrace,
    quantization: int,
) -> GeometrySampleAudit:
    native_records = tuple(
        NativeLineageRecord(
            native_id=target.instance_id,
            area=target.area,
            centroid=tuple(
                _quantized(value, quantization)
                for value in target.centroid
            ),
            projected_area=int(
                torch.count_nonzero(
                    trace.projected_mask(target.instance_id)
                )
            ),
            descendant_evaluation_ids=trace.descendant_ids(
                target.instance_id
            ),
            projected_mask_fingerprint=_tensor_content_fingerprint(
                trace.projected_mask(target.instance_id)
            ),
        )
        for target in trace.native.instances
    )
    evaluation_records = tuple(
        EvaluationLineageRecord(
            evaluation_id=target.instance_id,
            area=target.area,
            centroid=tuple(
                _quantized(value, quantization)
                for value in target.centroid
            ),
            ancestor_native_ids=trace.ancestor_ids(target.instance_id),
            evaluation_mask_fingerprint=_tensor_content_fingerprint(
                target.mask
            ),
        )
        for target in trace.evaluation.instances
    )
    return GeometrySampleAudit(
        sample_id=sample_id,
        group_id=group_id,
        mask_file_sha256=mask_sha256,
        native_targets=native_records,
        evaluation_targets=evaluation_records,
        disappeared_native_ids=tuple(
            native_id
            for native_id, descendants in trace.descendants
            if not descendants
        ),
        merged_evaluation_ids=tuple(
            evaluation_id
            for evaluation_id, ancestors in trace.ancestors
            if len(ancestors) > 1
        ),
        split_native_ids=tuple(
            native_id
            for native_id, descendants in trace.descendants
            if len(descendants) > 1
        ),
    )


def build_geometry_safe_catalog(
    bundle: LoadedDRCacheBundle,
    catalog: PreparedTrainingCatalog,
    manifest: SplitManifest,
    protocol: GeometryCatalogProtocol,
) -> GeometrySafeCatalog:
    """Build the complete A0/A1 ledger without mutating legacy objects."""

    if not isinstance(bundle, LoadedDRCacheBundle):
        raise TypeError("bundle must be LoadedDRCacheBundle")
    if not isinstance(catalog, PreparedTrainingCatalog):
        raise TypeError("catalog must be PreparedTrainingCatalog")
    if not isinstance(manifest, SplitManifest):
        raise TypeError("manifest must be SplitManifest")
    if not isinstance(protocol, GeometryCatalogProtocol):
        raise TypeError("protocol must be GeometryCatalogProtocol")
    if protocol.split != "D_R" or bundle.split != "D_R":
        raise ValueError("geometry-safe catalog permits only D_R")
    if protocol.dataset != manifest.dataset:
        raise ValueError("protocol and manifest datasets differ")
    if catalog.source_ids != tuple(row.sample_id for row in bundle.rows):
        raise ValueError("catalog and D_R bundle source identities differ")

    bundle.verify_unchanged()
    group_by_sample = {
        record.sample_id: record.group_id
        for record in manifest.records_for("D_R")
    }
    if set(group_by_sample) != set(catalog.source_ids):
        raise ValueError("manifest D_R groups differ from the catalog")
    row_by_id = {row.sample_id: row for row in bundle.rows}
    source_fingerprint = fingerprint_prepared_analysis_population(
        bundle,
        catalog,
        manifest,
    )

    sample_audits: list[GeometrySampleAudit] = []
    factual_records: list[GeometryTargetRecord] = []
    legal_records: list[GeometryTargetRecord] = []
    outside_records: list[GeometryTargetRecord] = []
    for entry in catalog.entries:
        row = row_by_id[entry.sample_id]
        native_mask = _native_mask(row.mask_path)
        if native_mask.dtype != torch.bool or native_mask.device.type != "cpu":
            raise TypeError("native GT loader did not produce a CPU bool tensor")
        trace = trace_native_to_evaluation(
            native_mask,
            row.state.gt_labels,
            protocol.geometry,
        )
        sample_audits.append(
            _sample_audit(
                sample_id=entry.sample_id,
                group_id=group_by_sample[entry.sample_id],
                mask_sha256=row.mask_sha256,
                trace=trace,
                quantization=protocol.geometry.geometry_quantization,
            )
        )

        for ordinal, (gt_id, example) in enumerate(
            zip(
                entry.reachable_gt_ids,
                entry.factual_examples,
                strict=True,
            )
        ):
            if example.supervision.positive_gt_ids != (gt_id,):
                raise RuntimeError("factual target identity differs from its example")
            factual_records.append(
                _target_record(
                    sample_id=entry.sample_id,
                    group_id=group_by_sample[entry.sample_id],
                    role="factual",
                    evaluation_gt_id=gt_id,
                    pred_id=None,
                    candidate_ordinal=ordinal,
                    analysis_candidate=True,
                    geometry=_evaluation_geometry(
                        trace,
                        gt_id,
                        protocol.geometry,
                    ),
                    source_state_content_fingerprint=row.content_fingerprint,
                )
            )

        unreachable = tuple(
            sorted(set(entry.real_miss_ids) - set(entry.reachable_gt_ids))
        )
        for ordinal, gt_id in enumerate(unreachable):
            outside_records.append(
                _target_record(
                    sample_id=entry.sample_id,
                    group_id=group_by_sample[entry.sample_id],
                    role="factual_unreachable",
                    evaluation_gt_id=gt_id,
                    pred_id=None,
                    candidate_ordinal=ordinal,
                    analysis_candidate=False,
                    geometry=_evaluation_geometry(
                        trace,
                        gt_id,
                        protocol.geometry,
                    ),
                    source_state_content_fingerprint=row.content_fingerprint,
                )
            )

        for ordinal, (candidate, example) in enumerate(
            zip(
                entry.decoder_visible_legal_candidates,
                entry.synthetic_examples,
                strict=True,
            )
        ):
            if (
                example.sample_id != entry.sample_id
                or example.feature is not entry.source.feature
                or example.supervision.branch != "synthetic"
                or example.supervision.positive_gt_ids != (candidate.gt_id,)
            ):
                raise RuntimeError(
                    "synthetic candidate/example metadata differs"
                )
            if not torch.equal(
                candidate.occupancy_after,
                example.supervision.occupancy[0],
            ):
                raise RuntimeError(
                    "candidate and synthetic conditioning occupancy differ"
                )
            evaluation_mask = entry.gt.by_id(candidate.gt_id).mask
            if not torch.equal(
                evaluation_mask,
                row.state.gt_labels == candidate.gt_id,
            ):
                raise RuntimeError(
                    "prepared GT and cached evaluation component differ"
                )
            actual_positive = example.supervision.target[0] > 0
            if (
                protocol.geometry
                .require_synthetic_positive_equals_evaluation_target
                and not torch.equal(actual_positive, evaluation_mask)
            ):
                raise RuntimeError(
                    "synthetic positive differs from its evaluation GT target"
                )
            legal_records.append(
                _target_record(
                    sample_id=entry.sample_id,
                    group_id=group_by_sample[entry.sample_id],
                    role="legal",
                    evaluation_gt_id=candidate.gt_id,
                    pred_id=candidate.pred_id,
                    candidate_ordinal=ordinal,
                    analysis_candidate=True,
                    geometry=_evaluation_geometry(
                        trace,
                        candidate.gt_id,
                        protocol.geometry,
                    ),
                    source_state_content_fingerprint=row.content_fingerprint,
                    candidate=candidate,
                    example=example,
                )
            )

    sample_audit_tuple = tuple(
        sorted(sample_audits, key=lambda item: item.sample_id)
    )
    factual_tuple = tuple(
        sorted(
            factual_records,
            key=lambda item: (item.sample_id, item.evaluation_gt_id),
        )
    )
    legal_tuple = tuple(
        sorted(
            legal_records,
            key=lambda item: (
                item.sample_id,
                item.evaluation_gt_id,
                int(item.pred_id),
            ),
        )
    )
    outside_tuple = tuple(
        sorted(
            outside_records,
            key=lambda item: (item.sample_id, item.evaluation_gt_id),
        )
    )
    if len({item.identity for item in factual_tuple}) != len(factual_tuple):
        raise RuntimeError("factual candidate identities are not unique")
    if len({item.identity for item in legal_tuple}) != len(legal_tuple):
        raise RuntimeError("legal candidate identities are not unique")
    if len(factual_tuple) != catalog.support_summary.reachable_miss_targets:
        raise RuntimeError("factual candidate ledger is incomplete")
    if (
        len(legal_tuple)
        != catalog.support_summary.decoder_visible_legal_candidates
    ):
        raise RuntimeError("legal candidate ledger is incomplete")
    if len(outside_tuple) != (
        catalog.support_summary.real_miss_targets
        - catalog.support_summary.reachable_miss_targets
    ):
        raise RuntimeError("outside-population factual ledger is incomplete")

    unsealed = GeometrySafeCatalog(
        protocol_fingerprint=protocol.fingerprint,
        source_catalog_fingerprint=source_fingerprint,
        sample_audits=sample_audit_tuple,
        factual_records=factual_tuple,
        legal_records=legal_tuple,
        outside_population_records=outside_tuple,
        catalog_fingerprint="",
    )
    fingerprint = stable_fingerprint(unsealed.canonical_payload())
    result = replace(unsealed, catalog_fingerprint=fingerprint)
    bundle.verify_unchanged()
    return result


@dataclass(frozen=True, eq=False)
class GeometrySafeEntryView:
    """Non-trainable view that filters candidate/example pairs by one index."""

    base: PreparedTrainingSource
    selected_legal_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base, PreparedTrainingSource):
            raise TypeError("base must be PreparedTrainingSource")
        indices = self.selected_legal_indices
        if (
            not isinstance(indices, tuple)
            or indices != tuple(sorted(set(indices)))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(
                    self.base.decoder_visible_legal_candidates
                )
                for index in indices
            )
        ):
            raise ValueError("selected legal indices are not canonical")

    @property
    def sample_id(self) -> str:
        return self.base.sample_id

    @property
    def source(self):
        return self.base.source

    @property
    def gt(self):
        return self.base.gt

    @property
    def real_miss_ids(self) -> tuple[int, ...]:
        return self.base.real_miss_ids

    @property
    def reachable_gt_ids(self) -> tuple[int, ...]:
        return self.base.reachable_gt_ids

    @property
    def factual_examples(self) -> tuple[StateExample, ...]:
        return self.base.factual_examples

    @property
    def decoder_visible_legal_candidates(
        self,
    ) -> tuple[PreparedLegalCandidate, ...]:
        return tuple(
            self.base.decoder_visible_legal_candidates[index]
            for index in self.selected_legal_indices
        )

    @property
    def synthetic_examples(self) -> tuple[StateExample, ...]:
        return tuple(
            self.base.synthetic_examples[index]
            for index in self.selected_legal_indices
        )


@dataclass(frozen=True, eq=False)
class GeometrySafeTrainingCatalogView:
    """P0-only zero-copy projection of a legacy prepared catalog."""

    base: PreparedTrainingCatalog
    entries: tuple[GeometrySafeEntryView, ...]
    support_summary: TrainingSupportSummary
    geometry_catalog_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.base, PreparedTrainingCatalog):
            raise TypeError("base must be PreparedTrainingCatalog")
        if len(self.entries) != len(self.base.entries) or any(
            view.base is not entry
            for view, entry in zip(
                self.entries,
                self.base.entries,
                strict=True,
            )
        ):
            raise ValueError("geometry view entries do not bind the base catalog")
        if not isinstance(self.support_summary, TrainingSupportSummary):
            raise TypeError("support_summary must be TrainingSupportSummary")
        visible = sum(
            len(entry.selected_legal_indices) for entry in self.entries
        )
        if (
            self.support_summary.decoder_visible_legal_candidates
            != visible
        ):
            raise ValueError("geometry view support summary is inconsistent")

    @property
    def source_ids(self) -> tuple[str, ...]:
        return self.base.source_ids


def build_geometry_safe_p0_view(
    legacy: PreparedTrainingCatalog,
    geometry: GeometrySafeCatalog,
) -> GeometrySafeTrainingCatalogView:
    """Project an eligible legal allowlist without rebuilding any tensors."""

    if not isinstance(legacy, PreparedTrainingCatalog):
        raise TypeError("legacy must be PreparedTrainingCatalog")
    if not isinstance(geometry, GeometrySafeCatalog):
        raise TypeError("geometry must be GeometrySafeCatalog")
    if any(
        not record.analysis_eligible for record in geometry.factual_records
    ):
        raise RuntimeError(
            "geometry-safe P0 view requires every reachable factual target"
        )
    safe = frozenset(geometry.eligible_legal_identities)
    raw = {
        (
            entry.sample_id,
            candidate.gt_id,
            candidate.pred_id,
        )
        for entry in legacy.entries
        for candidate in entry.decoder_visible_legal_candidates
    }
    ledger = {
        (
            record.sample_id,
            record.evaluation_gt_id,
            int(record.pred_id),
        )
        for record in geometry.legal_records
    }
    if raw != ledger:
        raise RuntimeError("geometry ledger and legacy legal population differ")
    if not safe or not safe <= raw:
        raise RuntimeError("geometry-safe legal allowlist is invalid")

    entries: list[GeometrySafeEntryView] = []
    selected_identities: set[tuple[str, int, int]] = set()
    for entry in legacy.entries:
        selected = tuple(
            index
            for index, candidate in enumerate(
                entry.decoder_visible_legal_candidates
            )
            if (
                entry.sample_id,
                candidate.gt_id,
                candidate.pred_id,
            )
            in safe
        )
        view = GeometrySafeEntryView(
            base=entry,
            selected_legal_indices=selected,
        )
        for index, candidate, example in zip(
            selected,
            view.decoder_visible_legal_candidates,
            view.synthetic_examples,
            strict=True,
        ):
            if (
                candidate
                is not entry.decoder_visible_legal_candidates[index]
                or example is not entry.synthetic_examples[index]
                or example.feature is not entry.source.feature
            ):
                raise RuntimeError(
                    "geometry view reconstructed a candidate or example"
                )
            selected_identities.add(
                (entry.sample_id, candidate.gt_id, candidate.pred_id)
            )
        entries.append(view)
    if selected_identities != safe:
        raise RuntimeError("geometry view did not project the full allowlist")

    support = replace(
        legacy.support_summary,
        decoder_visible_legal_candidates=len(safe),
        synthetic_images=sum(bool(entry.selected_legal_indices) for entry in entries),
    )
    return GeometrySafeTrainingCatalogView(
        base=legacy,
        entries=tuple(entries),
        support_summary=support,
        geometry_catalog_fingerprint=geometry.catalog_fingerprint,
    )


def build_p0_a0_receipt(
    geometry: GeometrySafeCatalog,
    protocol: GeometryCatalogProtocol,
) -> dict[str, object]:
    """Create the non-gating, dataset-wide A0 receipt."""

    native_count = sum(
        len(sample.native_targets) for sample in geometry.sample_audits
    )
    evaluation_count = sum(
        len(sample.evaluation_targets) for sample in geometry.sample_audits
    )
    disappearance_events = [
        {
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "native_id": native_id,
            "native_area": next(
                item.area
                for item in sample.native_targets
                if item.native_id == native_id
            ),
            "projected_area": 0,
        }
        for sample in geometry.sample_audits
        for native_id in sample.disappeared_native_ids
    ]
    merge_events = [
        {
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "evaluation_id": evaluation_id,
            "ancestor_native_ids": list(
                next(
                    item.ancestor_native_ids
                    for item in sample.evaluation_targets
                    if item.evaluation_id == evaluation_id
                )
            ),
        }
        for sample in geometry.sample_audits
        for evaluation_id in sample.merged_evaluation_ids
    ]
    split_events = [
        {
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "native_id": native_id,
            "descendant_evaluation_ids": list(
                next(
                    item.descendant_evaluation_ids
                    for item in sample.native_targets
                    if item.native_id == native_id
                )
            ),
        }
        for sample in geometry.sample_audits
        for native_id in sample.split_native_ids
    ]
    exact = not (disappearance_events or merge_events or split_events)
    return {
        "schema_version": P0_A0_SCHEMA,
        "protocol_id": protocol.protocol_id,
        "split": "D_R",
        "execution_status": "completed",
        "formal_gate_role": "none",
        "protocol_fingerprint": protocol.fingerprint,
        "source_catalog_fingerprint": (
            geometry.source_catalog_fingerprint
        ),
        "coordinate_contract": {
            "native_size": list(protocol.geometry.expected_native_size),
            "evaluation_size": list(
                protocol.geometry.expected_evaluation_size
            ),
            "resize_rule": protocol.geometry.resize_rule,
            "connectivity": protocol.geometry.connectivity,
            "centroid_coordinate_rule": (
                protocol.geometry.centroid_coordinate_rule
            ),
        },
        "counts": {
            "source_images": len(geometry.sample_audits),
            "native_targets": native_count,
            "evaluation_targets": evaluation_count,
            "native_disappearances": len(disappearance_events),
            "evaluation_merges": len(merge_events),
            "native_splits": len(split_events),
        },
        "rates": {
            "native_disappearance_rate": {
                "numerator": len(disappearance_events),
                "denominator": native_count,
                "value": (
                    len(disappearance_events) / native_count
                    if native_count
                    else 0.0
                ),
            }
        },
        "events": {
            "disappearances": disappearance_events,
            "merges": merge_events,
            "splits": split_events,
        },
        "dataset_exact_preservation": exact,
        "audit_status": (
            "exact_preservation" if exact else "anomalies_present"
        ),
        "downstream_gate_effect": "none",
        "claim_restrictions": [
            "does-not-establish-exact-preservation-of-all-native-targets"
        ]
        if not exact
        else [],
        "sample_audits": [
            sample.canonical_payload()
            for sample in geometry.sample_audits
        ],
        "does_not_reinterpret_p0_v1": True,
    }


def build_p0_a1_receipt(
    geometry: GeometrySafeCatalog,
    protocol: GeometryCatalogProtocol,
    *,
    a0_receipt_fingerprint: str,
) -> dict[str, object]:
    """Create the formal analysis-population eligibility receipt."""

    factual = geometry.factual_records
    legal = geometry.legal_records
    factual_eligible = tuple(
        item for item in factual if item.analysis_eligible
    )
    legal_eligible = tuple(item for item in legal if item.analysis_eligible)
    factual_excluded = tuple(
        item for item in factual if not item.analysis_eligible
    )
    legal_excluded = tuple(
        item for item in legal if not item.analysis_eligible
    )
    all_candidates = (*factual, *legal)
    all_identities = [
        (item.role, *item.identity) for item in all_candidates
    ]
    eligible_identities = [
        (item.role, *item.identity)
        for item in (*factual_eligible, *legal_eligible)
    ]
    duplicate_candidates = len(all_identities) - len(set(all_identities))
    duplicate_eligible = len(eligible_identities) - len(
        set(eligible_identities)
    )
    complete = (
        len(factual) + len(legal)
        == len(factual_eligible)
        + len(factual_excluded)
        + len(legal_eligible)
        + len(legal_excluded)
    )
    invalid_retained = sum(
        not item.geometry_eligible
        for item in (*factual_eligible, *legal_eligible)
    )
    factual_groups = {item.group_id for item in factual}
    factual_eligible_groups = {
        item.group_id for item in factual_eligible
    }
    legal_groups = {item.group_id for item in legal}
    legal_eligible_groups = {item.group_id for item in legal_eligible}
    candidate_fp = stable_fingerprint(
        [item.canonical_payload() for item in all_candidates]
    )
    eligible_fp = stable_fingerprint(
        {
            "protocol_fingerprint": protocol.fingerprint,
            "source_catalog_fingerprint": (
                geometry.source_catalog_fingerprint
            ),
            "factual": [
                item.canonical_payload() for item in factual_eligible
            ],
            "legal": [
                item.canonical_payload() for item in legal_eligible
            ],
        }
    )
    excluded_fp = stable_fingerprint(
        [
            item.canonical_payload()
            for item in (*factual_excluded, *legal_excluded)
        ]
    )
    p0_a1_pass = bool(
        complete
        and duplicate_candidates == 0
        and duplicate_eligible == 0
        and invalid_retained == 0
        and factual_eligible
        and legal_eligible
        and (
            len(factual_eligible) == len(factual)
            if (
                protocol.analysis_population
                .require_all_reachable_factual_geometry_eligible
            )
            else True
        )
    )
    failure_reasons: list[str] = []
    if not complete:
        failure_reasons.append("incomplete_candidate_accounting")
    if duplicate_candidates:
        failure_reasons.append("duplicate_candidate_identities")
    if duplicate_eligible:
        failure_reasons.append("duplicate_eligible_identities")
    if invalid_retained:
        failure_reasons.append("invalid_retained_targets")
    if not factual_eligible:
        failure_reasons.append("empty_factual_eligible_population")
    if not legal_eligible:
        failure_reasons.append("empty_legal_eligible_population")
    if (
        protocol.analysis_population
        .require_all_reachable_factual_geometry_eligible
        and len(factual_eligible) != len(factual)
    ):
        failure_reasons.append(
            "reachable_factual_geometry_exclusion_not_allowed"
        )

    return {
        "schema_version": P0_A1_SCHEMA,
        "protocol_id": protocol.protocol_id,
        "split": "D_R",
        "execution_status": "completed",
        "formal_gate_role": "p0-geometry-eligibility",
        "a0_receipt_fingerprint": a0_receipt_fingerprint,
        "protocol_fingerprint": protocol.fingerprint,
        "source_catalog_fingerprint": (
            geometry.source_catalog_fingerprint
        ),
        "geometry_catalog_fingerprint": geometry.catalog_fingerprint,
        "candidate_catalog_fingerprint": candidate_fp,
        "eligible_catalog_fingerprint": eligible_fp,
        "excluded_catalog_fingerprint": excluded_fp,
        "population_contract": {
            "factual": (
                protocol.analysis_population.factual_candidate_population
            ),
            "legal": protocol.analysis_population.legal_candidate_population,
            "group_key": protocol.analysis_population.group_key,
            "retention_threshold_policy": (
                protocol.analysis_population.retention_threshold_policy
            ),
            "invalid_candidate_action": (
                protocol.analysis_population.invalid_candidate_action
            ),
            "hardcoded_identity_exclusions": [],
        },
        "counts": {
            "factual_discovered": (
                len(factual) + len(geometry.outside_population_records)
            ),
            "factual_unreachable_outside_population": len(
                geometry.outside_population_records
            ),
            "factual_candidates": len(factual),
            "factual_eligible": len(factual_eligible),
            "factual_geometry_excluded": len(factual_excluded),
            "factual_candidate_groups": len(factual_groups),
            "factual_eligible_groups": len(factual_eligible_groups),
            "legal_candidates": len(legal),
            "legal_eligible": len(legal_eligible),
            "legal_geometry_excluded": len(legal_excluded),
            "legal_candidate_source_images": len(
                {item.sample_id for item in legal}
            ),
            "legal_eligible_source_images": len(
                {item.sample_id for item in legal_eligible}
            ),
            "legal_candidate_groups": len(legal_groups),
            "legal_eligible_groups": len(legal_eligible_groups),
        },
        "accounting": {
            "all_candidates_classified_exactly_once": complete,
            "duplicate_candidate_identities": duplicate_candidates,
            "duplicate_eligible_identities": duplicate_eligible,
            "invalid_retained_targets": invalid_retained,
            "unaccounted_targets": 0 if complete else 1,
        },
        "rule_outcomes": {
            "all_reachable_factual_geometry_eligible": (
                len(factual_eligible) == len(factual)
            ),
            "all_retained_bidirectional_one_to_one_lineage": all(
                item.reciprocal_one_to_one
                for item in (*factual_eligible, *legal_eligible)
            ),
            "all_retained_exact_component_projection": all(
                item.exact_component_projection
                for item in (*factual_eligible, *legal_eligible)
            ),
            "all_retained_area_ratio_within_gate": all(
                item.area_ratio is not None
                and "area_ratio_below_minimum"
                not in item.geometry_reason_codes
                and "area_ratio_above_maximum"
                not in item.geometry_reason_codes
                for item in (*factual_eligible, *legal_eligible)
            ),
            "all_retained_centroid_shift_within_gate": all(
                item.centroid_shift_evaluation_px is not None
                and "centroid_shift_above_maximum"
                not in item.geometry_reason_codes
                for item in (*factual_eligible, *legal_eligible)
            ),
        },
        "candidate_targets": [
            item.canonical_payload() for item in all_candidates
        ],
        "eligible_target_identities": {
            "factual": [list(item.identity) for item in factual_eligible],
            "legal": [list(item.identity) for item in legal_eligible],
        },
        "excluded_targets": [
            item.canonical_payload()
            for item in (*factual_excluded, *legal_excluded)
        ],
        "outside_population_ledger": [
            item.canonical_payload()
            for item in geometry.outside_population_records
        ],
        "p0_a1_pass": p0_a1_pass,
        "formal_status": "pass" if p0_a1_pass else "fail",
        "failure_reason_codes": failure_reasons,
        "next_route_if_failed": (
            None
            if p0_a1_pass
            else "rebuild-analysis-population-extraction"
        ),
        "analysis_claim_scope": (
            "geometry-eligible-evaluation-grid-targets"
        ),
        "does_not_reinterpret_p0_v1": True,
        "requires_matched_uniform_control": True,
    }


__all__ = [
    "GEOMETRY_SAFE_CATALOG_SCHEMA",
    "P0_A0_SCHEMA",
    "P0_A1_SCHEMA",
    "EvaluationLineageRecord",
    "GeometrySafeCatalog",
    "GeometrySafeEntryView",
    "GeometrySafeTrainingCatalogView",
    "GeometrySampleAudit",
    "GeometryTargetRecord",
    "NativeEvaluationTrace",
    "NativeLineageRecord",
    "build_geometry_safe_catalog",
    "build_geometry_safe_p0_view",
    "build_p0_a0_receipt",
    "build_p0_a1_receipt",
    "fingerprint_prepared_analysis_population",
    "trace_native_to_evaluation",
]
