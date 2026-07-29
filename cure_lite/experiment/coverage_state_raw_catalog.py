"""Strict ``D_R`` adapter for the representation-neutral CSLF population.

The adapter starts from the verified Base/state cache bundle and full-grid
component semantics.  It does not call ``prepare_training_catalog``,
``decoder_visible_legal_deletions``, ``PairExample``, or any scalar-specific
batch validator.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..coverage_state_raw_catalog import (
    CoverageStateNaturalRecord,
    CoverageStatePairRecord,
    CoverageStateRawCatalog,
    CoverageStateRawExclusion,
    make_coverage_state_raw_catalog,
)
from ..instances import instances_from_binary_mask
from ..intervention import enumerate_legal_deletions
from ..matching import match_components
from ..splits import SplitManifest
from ..types import InstanceMap, MatchResult
from .cache_pipeline import LoadedDRCacheBundle, LoadedDRCacheRow
from .geometry_safe_catalog import GeometrySafeCatalog, GeometryTargetRecord


def _positive_id_tuple(values: Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in values.tolist())


def _pair_tuple(values: Tensor) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(int(item) for item in row)
        for row in values.tolist()
    )


def _match_identity(match: MatchResult) -> tuple[tuple[int, int], ...]:
    return tuple((pair.gt_id, pair.pred_id) for pair in match.pairs)


def _match_fingerprint(match: MatchResult) -> str:
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-coverage-state-match-v1",
            "pairs": [
                {
                    "gt_id": pair.gt_id,
                    "pred_id": pair.pred_id,
                    "distance_hex": float(pair.distance).hex(),
                    "iou_hex": float(pair.iou).hex(),
                }
                for pair in match.pairs
            ],
            "pred_ids": list(match.pred_ids),
            "gt_ids": list(match.gt_ids),
            "unmatched_pred_ids": sorted(match.unmatched_pred_ids),
            "unmatched_gt_ids": sorted(match.unmatched_gt_ids),
        }
    )


def _record_fingerprint(record: GeometryTargetRecord) -> str:
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-geometry-lineage-record-binding-v1",
            "record": record.canonical_payload(),
        }
    )


def _raw_id(
    *,
    kind: str,
    sample_id: str,
    evaluation_gt_id: int | None,
    pred_id: int | None,
) -> str:
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-coverage-state-raw-record-id-v1",
            "kind": kind,
            "sample_id": sample_id,
            "evaluation_gt_id": evaluation_gt_id,
            "pred_id": pred_id,
        }
    )


def _completion_field(
    *,
    gt: InstanceMap,
    match: MatchResult,
    occupancy: Tensor,
    image_valid_mask: Tensor,
    allowed_gt_ids: tuple[int, ...] | None = None,
) -> Tensor:
    """Return completion truth for every currently unmatched GT component."""

    if (
        occupancy.dtype != torch.bool
        or image_valid_mask.dtype != torch.bool
        or occupancy.device.type != "cpu"
        or image_valid_mask.device.type != "cpu"
        or occupancy.ndim != 2
        or occupancy.shape != image_valid_mask.shape
        or tuple(occupancy.shape) != gt.shape
    ):
        raise TypeError("completion occupancy/valid masks must be aligned CPU bool")
    if match.gt_ids != tuple(sorted(gt.ids)):
        raise ValueError("completion match and GT identities differ")
    if allowed_gt_ids is None:
        selected_ids = tuple(sorted(match.unmatched_gt_ids))
    else:
        if (
            not isinstance(allowed_gt_ids, tuple)
            or allowed_gt_ids != tuple(sorted(set(allowed_gt_ids)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in allowed_gt_ids
            )
            or not set(allowed_gt_ids) <= set(match.gt_ids)
        ):
            raise ValueError("allowed_gt_ids must be sorted GT identities")
        selected_ids = tuple(
            gt_id
            for gt_id in allowed_gt_ids
            if gt_id in match.unmatched_gt_ids
        )
    result = torch.zeros_like(occupancy)
    for gt_id in selected_ids:
        result |= (
            gt.by_id(gt_id).mask
            & image_valid_mask
            & ~occupancy
        )
    return result.contiguous()


def _source_state(
    row: LoadedDRCacheRow,
    bundle: LoadedDRCacheBundle,
) -> tuple[Tensor, Tensor, InstanceMap, InstanceMap, MatchResult]:
    state = row.state.normalized()
    occupancy = state.occupancy
    valid = state.image_valid_mask
    pred = instances_from_binary_mask(occupancy)
    gt = instances_from_binary_mask(state.gt_labels > 0)
    if not torch.equal(pred.labels, state.pred_labels):
        raise RuntimeError(
            f"cached prediction labels are not canonical for {row.sample_id!r}"
        )
    if not torch.equal(gt.labels, state.gt_labels):
        raise RuntimeError(
            f"cached GT labels are not canonical for {row.sample_id!r}"
        )
    before = match_components(pred, gt, bundle.match_config)
    if _match_identity(before) != _pair_tuple(state.base_match_pairs):
        raise RuntimeError(
            f"cached Base matches changed for {row.sample_id!r}"
        )
    legal = enumerate_legal_deletions(
        pred,
        gt,
        before,
        occupancy,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )
    legal_identity = tuple(
        (value.gt_id, value.pred_id) for value in legal
    )
    if legal_identity != _pair_tuple(state.legal_pairs):
        raise RuntimeError(
            f"cached legal deletions changed for {row.sample_id!r}"
        )
    return occupancy, valid, pred, gt, before


def _geometry_maps(
    bundle: LoadedDRCacheBundle,
    geometry: GeometrySafeCatalog,
) -> tuple[
    dict[tuple[str, int], GeometryTargetRecord],
    dict[tuple[str, int, int], GeometryTargetRecord],
]:
    if stable_fingerprint(geometry.canonical_payload()) != (
        geometry.catalog_fingerprint
    ):
        raise ValueError("geometry catalog fingerprint does not match its contents")
    row_fingerprints = {
        row.sample_id: row.content_fingerprint for row in bundle.rows
    }
    factual: dict[tuple[str, int], GeometryTargetRecord] = {}
    for record in geometry.factual_records:
        if (
            record.role != "factual"
            or record.pred_id is not None
            or record.source_state_content_fingerprint
            != row_fingerprints.get(record.sample_id)
        ):
            raise ValueError("geometry factual lineage binding is invalid")
        key = record.sample_id, record.evaluation_gt_id
        if key in factual:
            raise ValueError("geometry factual identities are duplicated")
        factual[key] = record
    legal: dict[tuple[str, int, int], GeometryTargetRecord] = {}
    for record in geometry.legal_records:
        if (
            record.role != "legal"
            or record.pred_id is None
            or record.source_state_content_fingerprint
            != row_fingerprints.get(record.sample_id)
        ):
            raise ValueError("geometry legal lineage binding is invalid")
        key = (
            record.sample_id,
            record.evaluation_gt_id,
            int(record.pred_id),
        )
        if key in legal:
            raise ValueError("geometry legal identities are duplicated")
        legal[key] = record
    expected_factual = {
        (row.sample_id, gt_id)
        for row in bundle.rows
        for gt_id in _positive_id_tuple(row.state.reachable_miss_ids)
    }
    expected_legal = {
        (row.sample_id, gt_id, pred_id)
        for row in bundle.rows
        for gt_id, pred_id in _pair_tuple(row.state.legal_pairs)
    }
    if set(factual) != expected_factual:
        raise ValueError(
            "geometry factual lineage does not close over cached reachable misses"
        )
    if set(legal) != expected_legal:
        raise ValueError(
            "geometry legal lineage does not close over full-grid legal pairs"
        )
    outside = {
        (record.sample_id, record.evaluation_gt_id)
        for record in geometry.outside_population_records
    }
    expected_outside = {
        (row.sample_id, gt_id)
        for row in bundle.rows
        for gt_id in (
            set(_positive_id_tuple(row.state.real_miss_ids))
            - set(_positive_id_tuple(row.state.reachable_miss_ids))
        )
    }
    if outside != expected_outside:
        raise ValueError(
            "geometry outside-population lineage does not close over cached misses"
        )
    return factual, legal


def _feature_stride(bundle: LoadedDRCacheBundle) -> int:
    strides: set[int] = set()
    for row in bundle.rows:
        feature = row.base_output.feature
        height, width = row.state.occupancy.shape
        feature_height, feature_width = feature.shape[-2:]
        if (
            height % feature_height
            or width % feature_width
            or height // feature_height != width // feature_width
        ):
            raise ValueError(
                "Base feature and evaluation grids require one exact stride"
            )
        strides.add(height // feature_height)
    if len(strides) != 1:
        raise ValueError("D_R rows do not share one feature stride")
    stride, = strides
    if stride < 1:
        raise ValueError("feature stride must be positive")
    return stride


def _natural_records(
    *,
    row: LoadedDRCacheRow,
    group_id: str,
    occupancy: Tensor,
    valid: Tensor,
    gt: InstanceMap,
    factual_geometry: dict[tuple[str, int], GeometryTargetRecord],
) -> tuple[
    tuple[CoverageStateNaturalRecord, ...],
    tuple[CoverageStateRawExclusion, ...],
]:
    state = row.state
    feature = row.base_output.feature
    real_misses = _positive_id_tuple(state.real_miss_ids)
    reachable = _positive_id_tuple(state.reachable_miss_ids)
    background = state.gt_labels == 0
    writable = ~occupancy
    records: list[CoverageStateNaturalRecord] = []
    exclusions: list[CoverageStateRawExclusion] = []
    if not real_misses:
        empty = torch.zeros_like(occupancy)
        loss_valid = valid & writable & background
        records.append(
            CoverageStateNaturalRecord(
                record_id=_raw_id(
                    kind="factual_no_miss",
                    sample_id=row.sample_id,
                    evaluation_gt_id=None,
                    pred_id=None,
                ),
                sample_id=row.sample_id,
                group_id=group_id,
                state_kind="factual_no_miss",
                feature=feature,
                occupancy=occupancy.unsqueeze(0).unsqueeze(0),
                target=empty.unsqueeze(0).unsqueeze(0),
                valid_mask=valid.unsqueeze(0).unsqueeze(0),
                loss_valid_mask=loss_valid.unsqueeze(0).unsqueeze(0),
                target_ids=(),
                focus_target_ids=(),
                source_row_fingerprint=row.content_fingerprint,
                evaluation_gt_ids=(),
                native_gt_ids=(),
                lineage_record_fingerprint=None,
            )
        )
    eligible_geometry: list[GeometryTargetRecord] = []
    for gt_id in reachable:
        geometry = factual_geometry[(row.sample_id, gt_id)]
        if not geometry.analysis_eligible:
            exclusions.append(
                CoverageStateRawExclusion(
                    candidate_kind="factual_geometry",
                    sample_id=row.sample_id,
                    evaluation_gt_id=gt_id,
                    pred_id=None,
                    reason_codes=tuple(
                        sorted(
                            set(geometry.analysis_exclusion_reasons)
                            or {"geometry_not_analysis_eligible"}
                        )
                    ),
                )
            )
            continue
        if geometry.native_gt_id is None:
            raise ValueError("eligible factual geometry lacks native identity")
        eligible_geometry.append(geometry)
    if eligible_geometry:
        evaluation_ids = tuple(
            sorted(record.evaluation_gt_id for record in eligible_geometry)
        )
        native_ids = tuple(
            sorted(int(record.native_gt_id) for record in eligible_geometry)
        )
        if len(set(native_ids)) != len(native_ids):
            raise ValueError("eligible factual lineage repeats a native identity")
        scene_target = torch.zeros_like(occupancy)
        for gt_id in evaluation_ids:
            scene_target |= (
                gt.by_id(gt_id).mask
                & valid
                & writable
            )
        if not torch.any(scene_target):
            raise RuntimeError("eligible factual miss has no writable target")
        target_ids = tuple(
            sorted(f"evaluation_gt:{gt_id}" for gt_id in evaluation_ids)
        )
        scene_lineage_fingerprint = stable_fingerprint(
            {
                "schema_version": (
                    "cure-lite-scene-complete-factual-lineage-v1"
                ),
                "sample_id": row.sample_id,
                "records": [
                    _record_fingerprint(record)
                    for record in sorted(
                        eligible_geometry,
                        key=lambda value: value.evaluation_gt_id,
                    )
                ],
            }
        )
    else:
        evaluation_ids = ()
        native_ids = ()
        scene_target = torch.zeros_like(occupancy)
        target_ids = ()
        scene_lineage_fingerprint = None
    for geometry in sorted(
        eligible_geometry,
        key=lambda value: value.evaluation_gt_id,
    ):
        gt_id = geometry.evaluation_gt_id
        focus_target = gt.by_id(gt_id).mask & scene_target
        loss_valid = valid & writable & (background | focus_target)
        records.append(
            CoverageStateNaturalRecord(
                record_id=_raw_id(
                    kind="factual_miss",
                    sample_id=row.sample_id,
                    evaluation_gt_id=gt_id,
                    pred_id=None,
                ),
                sample_id=row.sample_id,
                group_id=group_id,
                state_kind="factual_miss",
                feature=feature,
                occupancy=occupancy.unsqueeze(0).unsqueeze(0),
                target=scene_target.unsqueeze(0).unsqueeze(0),
                valid_mask=valid.unsqueeze(0).unsqueeze(0),
                loss_valid_mask=loss_valid.unsqueeze(0).unsqueeze(0),
                target_ids=target_ids,
                focus_target_ids=(f"evaluation_gt:{gt_id}",),
                source_row_fingerprint=row.content_fingerprint,
                evaluation_gt_ids=evaluation_ids,
                native_gt_ids=native_ids,
                lineage_record_fingerprint=scene_lineage_fingerprint,
            )
        )
    for gt_id in sorted(set(real_misses) - set(reachable)):
        exclusions.append(
            CoverageStateRawExclusion(
                candidate_kind="factual_unreachable",
                sample_id=row.sample_id,
                evaluation_gt_id=gt_id,
                pred_id=None,
                reason_codes=("not_individually_reachable",),
            )
        )
    return tuple(records), tuple(exclusions)


def _pair_records(
    *,
    bundle: LoadedDRCacheBundle,
    row: LoadedDRCacheRow,
    group_id: str,
    occupancy: Tensor,
    valid: Tensor,
    pred: InstanceMap,
    gt: InstanceMap,
    before: MatchResult,
    legal_geometry: dict[tuple[str, int, int], GeometryTargetRecord],
    base_target_ids: tuple[int, ...],
) -> tuple[
    tuple[CoverageStatePairRecord, ...],
    tuple[CoverageStateRawExclusion, ...],
]:
    feature = row.base_output.feature
    before_field = _completion_field(
        gt=gt,
        match=before,
        occupancy=occupancy,
        image_valid_mask=valid,
        allowed_gt_ids=base_target_ids,
    )
    before_full_field = _completion_field(
        gt=gt,
        match=before,
        occupancy=occupancy,
        image_valid_mask=valid,
    )
    before_fingerprint = _match_fingerprint(before)
    records: list[CoverageStatePairRecord] = []
    exclusions: list[CoverageStateRawExclusion] = []
    empty = torch.zeros_like(occupancy)
    records.append(
        CoverageStatePairRecord(
            pair_id=_raw_id(
                kind="identity_null",
                sample_id=row.sample_id,
                evaluation_gt_id=None,
                pred_id=None,
            ),
            sample_id=row.sample_id,
            group_id=group_id,
            pair_kind="identity_null",
            feature=feature,
            occupancy_plus=occupancy.unsqueeze(0).unsqueeze(0),
            occupancy_minus=occupancy.unsqueeze(0).unsqueeze(0),
            target_plus=before_field.unsqueeze(0).unsqueeze(0),
            target_minus=before_field.unsqueeze(0).unsqueeze(0),
            valid_mask=valid.unsqueeze(0).unsqueeze(0),
            removed_component=empty.unsqueeze(0).unsqueeze(0),
            removed_component_ids=(),
            target_ids_added=(),
            source_row_fingerprint=row.content_fingerprint,
            evaluation_gt_id=None,
            native_gt_id=None,
            pred_id=None,
            before_match_fingerprint=before_fingerprint,
            after_match_fingerprint=before_fingerprint,
            lineage_record_fingerprint=None,
        )
    )
    legal = enumerate_legal_deletions(
        pred,
        gt,
        before,
        occupancy,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )
    for deletion in legal:
        geometry = legal_geometry[
            (row.sample_id, deletion.gt_id, deletion.pred_id)
        ]
        if not geometry.analysis_eligible:
            exclusions.append(
                CoverageStateRawExclusion(
                    candidate_kind="clean_positive",
                    sample_id=row.sample_id,
                    evaluation_gt_id=deletion.gt_id,
                    pred_id=deletion.pred_id,
                    reason_codes=tuple(
                        sorted(
                            set(geometry.analysis_exclusion_reasons)
                            or {"geometry_not_analysis_eligible"}
                        )
                    ),
                )
            )
            continue
        if geometry.native_gt_id is None:
            raise ValueError("eligible legal geometry lacks native identity")
        component = pred.by_id(deletion.pred_id).mask
        occupancy_minus = deletion.occupancy_after
        after = deletion.match_after
        after_field = _completion_field(
            gt=gt,
            match=after,
            occupancy=occupancy_minus,
            image_valid_mask=valid,
            allowed_gt_ids=tuple(
                sorted((*base_target_ids, deletion.gt_id))
            ),
        )
        increment = after_field & ~before_field
        expected_increment = (
            valid
            & gt.by_id(deletion.gt_id).mask
            & ~occupancy_minus
        )
        reasons: list[str] = []
        if not torch.equal(increment, expected_increment):
            reasons.append("actual_increment_differs_from_selected_target")
        if not torch.any(increment):
            reasons.append("empty_actual_increment")
        if not torch.any(valid & ~increment):
            reasons.append("empty_zero_response_domain")
        if before.unmatched_gt_ids:
            preexisting = torch.stack(
                [
                    gt.by_id(gt_id).mask
                    for gt_id in sorted(before.unmatched_gt_ids)
                ],
                dim=0,
            ).any(dim=0)
            if torch.any(component & valid & preexisting):
                reasons.append("preexisting_unmatched_gt_interference")
        if reasons:
            exclusions.append(
                CoverageStateRawExclusion(
                    candidate_kind="clean_positive",
                    sample_id=row.sample_id,
                    evaluation_gt_id=deletion.gt_id,
                    pred_id=deletion.pred_id,
                    reason_codes=tuple(sorted(set(reasons))),
                )
            )
            continue
        records.append(
            CoverageStatePairRecord(
                pair_id=_raw_id(
                    kind="clean_positive",
                    sample_id=row.sample_id,
                    evaluation_gt_id=deletion.gt_id,
                    pred_id=deletion.pred_id,
                ),
                sample_id=row.sample_id,
                group_id=group_id,
                pair_kind="clean_positive",
                feature=feature,
                occupancy_plus=occupancy.unsqueeze(0).unsqueeze(0),
                occupancy_minus=occupancy_minus.unsqueeze(0).unsqueeze(0),
                target_plus=before_field.unsqueeze(0).unsqueeze(0),
                target_minus=after_field.unsqueeze(0).unsqueeze(0),
                valid_mask=valid.unsqueeze(0).unsqueeze(0),
                removed_component=component.unsqueeze(0).unsqueeze(0),
                removed_component_ids=(
                    f"prediction_component:{deletion.pred_id}",
                ),
                target_ids_added=(
                    f"evaluation_gt:{deletion.gt_id}",
                ),
                source_row_fingerprint=row.content_fingerprint,
                evaluation_gt_id=deletion.gt_id,
                native_gt_id=int(geometry.native_gt_id),
                pred_id=deletion.pred_id,
                before_match_fingerprint=before_fingerprint,
                after_match_fingerprint=_match_fingerprint(after),
                lineage_record_fingerprint=_record_fingerprint(geometry),
            )
        )
    for pred_id in pred.ids:
        component = pred.by_id(pred_id).mask
        pred_after = pred.without(pred_id)
        occupancy_minus = occupancy & ~component
        if not torch.equal(occupancy_minus, pred_after.occupancy):
            raise RuntimeError("component deletion changed prediction identities")
        after = match_components(pred_after, gt, bundle.match_config)
        after_field = _completion_field(
            gt=gt,
            match=after,
            occupancy=occupancy_minus,
            image_valid_mask=valid,
            allowed_gt_ids=base_target_ids,
        )
        after_full_field = _completion_field(
            gt=gt,
            match=after,
            occupancy=occupancy_minus,
            image_valid_mask=valid,
        )
        if not torch.equal(after_full_field, before_full_field):
            exclusions.append(
                CoverageStateRawExclusion(
                    candidate_kind="component_null",
                    sample_id=row.sample_id,
                    evaluation_gt_id=None,
                    pred_id=pred_id,
                    reason_codes=("actual_increment_nonempty",),
                )
            )
            continue
        records.append(
            CoverageStatePairRecord(
                pair_id=_raw_id(
                    kind="component_null",
                    sample_id=row.sample_id,
                    evaluation_gt_id=None,
                    pred_id=pred_id,
                ),
                sample_id=row.sample_id,
                group_id=group_id,
                pair_kind="component_null",
                feature=feature,
                occupancy_plus=occupancy.unsqueeze(0).unsqueeze(0),
                occupancy_minus=occupancy_minus.unsqueeze(0).unsqueeze(0),
                target_plus=before_field.unsqueeze(0).unsqueeze(0),
                target_minus=after_field.unsqueeze(0).unsqueeze(0),
                valid_mask=valid.unsqueeze(0).unsqueeze(0),
                removed_component=component.unsqueeze(0).unsqueeze(0),
                removed_component_ids=(
                    f"prediction_component:{pred_id}",
                ),
                target_ids_added=(),
                source_row_fingerprint=row.content_fingerprint,
                evaluation_gt_id=None,
                native_gt_id=None,
                pred_id=pred_id,
                before_match_fingerprint=before_fingerprint,
                after_match_fingerprint=_match_fingerprint(after),
                lineage_record_fingerprint=None,
            )
        )
    return tuple(records), tuple(exclusions)


def build_coverage_state_raw_catalog(
    bundle: LoadedDRCacheBundle,
    manifest: SplitManifest,
    geometry: GeometrySafeCatalog,
) -> CoverageStateRawCatalog:
    """Build the complete representation-neutral population from strict D_R."""

    if not isinstance(bundle, LoadedDRCacheBundle):
        raise TypeError("bundle must be LoadedDRCacheBundle")
    if not isinstance(manifest, SplitManifest):
        raise TypeError("manifest must be SplitManifest")
    if not isinstance(geometry, GeometrySafeCatalog):
        raise TypeError("geometry must be GeometrySafeCatalog")
    if bundle.split != "D_R":
        raise ValueError("raw coverage-state catalog permits only D_R")
    if manifest.dataset == "" or manifest.fingerprint != (
        bundle.split_manifest_fingerprint
    ):
        raise ValueError("manifest and strict D_R bundle bindings differ")
    allowed_samples = {row.sample_id for row in bundle.rows}
    group_by_sample = {
        record.sample_id: record.group_id
        for record in manifest.records_for("D_R")
        if record.sample_id in allowed_samples
    }
    if set(group_by_sample) != {row.sample_id for row in bundle.rows}:
        raise ValueError("manifest D_R identities differ from the cache bundle")
    bundle.verify_unchanged()
    factual_geometry, legal_geometry = _geometry_maps(bundle, geometry)
    natural_records: list[CoverageStateNaturalRecord] = []
    pair_records: list[CoverageStatePairRecord] = []
    exclusions: list[CoverageStateRawExclusion] = []
    for row in bundle.rows:
        occupancy, valid, pred, gt, before = _source_state(row, bundle)
        natural, natural_exclusions = _natural_records(
            row=row,
            group_id=group_by_sample[row.sample_id],
            occupancy=occupancy,
            valid=valid,
            gt=gt,
            factual_geometry=factual_geometry,
        )
        pairs, pair_exclusions = _pair_records(
            bundle=bundle,
            row=row,
            group_id=group_by_sample[row.sample_id],
            occupancy=occupancy,
            valid=valid,
            pred=pred,
            gt=gt,
            before=before,
            legal_geometry=legal_geometry,
            base_target_ids=tuple(
                sorted(
                    gt_id
                    for gt_id in _positive_id_tuple(
                        row.state.reachable_miss_ids
                    )
                    if factual_geometry[
                        (row.sample_id, gt_id)
                    ].analysis_eligible
                )
            ),
        )
        natural_records.extend(natural)
        pair_records.extend(pairs)
        exclusions.extend(natural_exclusions)
        exclusions.extend(pair_exclusions)
    source_fingerprint = stable_fingerprint(
        {
            "schema_version": "cure-lite-coverage-state-raw-source-v1",
            "split": "D_R",
            "dataset": manifest.dataset,
            "manifest_fingerprint": manifest.fingerprint,
            "base_fingerprint": bundle.base_fingerprint,
            "base_state_fingerprint": bundle.base_state_fingerprint,
            "state_fingerprint": bundle.state_fingerprint,
            "gt_fingerprint": bundle.gt_fingerprint,
            "geometry_catalog_fingerprint": geometry.catalog_fingerprint,
            "row_content_fingerprints": [
                [row.sample_id, row.content_fingerprint]
                for row in bundle.rows
            ],
            "representation_selected": False,
            "scalar_visibility_used_for_inclusion": False,
            "phase_visibility_used_for_inclusion": False,
            "training_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    result = make_coverage_state_raw_catalog(
        dataset=manifest.dataset,
        feature_stride=_feature_stride(bundle),
        source_fingerprint=source_fingerprint,
        natural_records=tuple(natural_records),
        pair_records=tuple(pair_records),
        exclusions=tuple(exclusions),
    )
    bundle.verify_unchanged()
    return result


__all__ = ["build_coverage_state_raw_catalog"]
