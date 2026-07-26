"""Deterministic clean/null pair construction over the frozen D_R catalog.

This is an additive catalog layer.  It consumes already prepared cache
objects, the geometry-safe allowlist, and the frozen split manifest; it does
not load images, evaluate D_V/D_T, alter the decoder, or touch the legacy
single-state training path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..config import MatchConfig
from ..decoder import project_occupancy_to_feature_grid
from ..instances import instances_from_binary_mask
from ..matching import match_components
from ..paired_types import (
    PairCatalog,
    PairCatalogExclusion,
    PairExample,
    tensor_content_fingerprint,
)
from ..splits import SplitManifest
from ..types import InstanceMap, MatchResult
from .geometry_safe_catalog import GeometrySafeCatalog, GeometryTargetRecord
from .training_pipeline import PreparedTrainingCatalog, PreparedTrainingSource


PAIR_CATALOG_SCHEMA = "cure-lite-pair-catalog-v1"


def _pair_id(
    *,
    kind: str,
    sample_id: str,
    evaluation_gt_id: int | None,
    native_gt_id: int | None,
    pred_id: int | None,
) -> str:
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-pair-id-v1",
            "pair_kind": kind,
            "sample_id": sample_id,
            "evaluation_gt_id": evaluation_gt_id,
            "native_gt_id": native_gt_id,
            "pred_id": pred_id,
        }
    )


def _match_fingerprint(match: MatchResult) -> str:
    if not isinstance(match, MatchResult):
        raise TypeError("match must be a MatchResult")
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-pair-match-v1",
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


def _completion_field(
    *,
    gt: InstanceMap,
    match: MatchResult,
    occupancy: Tensor,
    image_valid_mask: Tensor,
) -> Tensor:
    """Materialize R_G,V(O) under the frozen instance-matcher semantics."""

    if not isinstance(gt, InstanceMap) or not isinstance(match, MatchResult):
        raise TypeError("gt and match must be InstanceMap/MatchResult")
    if (
        not isinstance(occupancy, Tensor)
        or not isinstance(image_valid_mask, Tensor)
        or occupancy.device.type != "cpu"
        or image_valid_mask.device.type != "cpu"
        or occupancy.dtype != torch.bool
        or image_valid_mask.dtype != torch.bool
        or occupancy.ndim != 2
        or image_valid_mask.ndim != 2
        or occupancy.shape != image_valid_mask.shape
        or tuple(occupancy.shape) != gt.shape
    ):
        raise TypeError("occupancy and image_valid_mask must be aligned CPU bool masks")
    if match.gt_ids != tuple(sorted(gt.ids)):
        raise ValueError("match and GT instance identities differ")
    result = torch.zeros_like(occupancy)
    for gt_id in sorted(match.unmatched_gt_ids):
        result |= gt.by_id(gt_id).mask & image_valid_mask & ~occupancy
    return result.contiguous()


def _projected_fingerprints(
    occupancy_plus: Tensor,
    occupancy_minus: Tensor,
    *,
    feature_size: tuple[int, int],
) -> tuple[str, str, bool]:
    plus = project_occupancy_to_feature_grid(
        occupancy_plus.unsqueeze(0).unsqueeze(0),
        feature_size,
    )
    minus = project_occupancy_to_feature_grid(
        occupancy_minus.unsqueeze(0).unsqueeze(0),
        feature_size,
    )
    return (
        tensor_content_fingerprint(plus),
        tensor_content_fingerprint(minus),
        not torch.equal(plus, minus),
    )


def _source_state(
    entry: PreparedTrainingSource,
    match_config: MatchConfig,
) -> tuple[Tensor, InstanceMap, MatchResult, Tensor]:
    state = entry.source.state
    occupancy = state.occupancy
    pred = instances_from_binary_mask(occupancy)
    if not torch.equal(pred.labels, state.pred_labels):
        raise RuntimeError(
            f"cached prediction labels are not canonical for {entry.sample_id!r}"
        )
    if not torch.equal(entry.gt.labels, state.gt_labels):
        raise RuntimeError(
            f"prepared GT differs from cached GT for {entry.sample_id!r}"
        )
    before = match_components(pred, entry.gt, match_config)
    cached_pairs = tuple(
        tuple(int(value) for value in row)
        for row in state.base_match_pairs.tolist()
    )
    before_pairs = tuple((pair.gt_id, pair.pred_id) for pair in before.pairs)
    if before_pairs != cached_pairs:
        raise RuntimeError(
            f"recomputed matches differ from cache for {entry.sample_id!r}"
        )
    valid = state.image_valid_mask
    if torch.any(occupancy & ~valid):
        raise RuntimeError("cached occupancy extends outside image_valid_mask")
    return occupancy, pred, before, valid


def _deletion_truth(
    *,
    occupancy_plus: Tensor,
    pred: InstanceMap,
    pred_id: int,
    gt: InstanceMap,
    before: MatchResult,
    image_valid_mask: Tensor,
    match_config: MatchConfig,
) -> dict[str, object]:
    component = pred.by_id(pred_id).mask
    pred_after = pred.without(pred_id)
    occupancy_minus = occupancy_plus & ~component
    if not torch.equal(occupancy_minus, pred_after.occupancy):
        raise RuntimeError("component deletion did not preserve prediction identities")
    after = match_components(pred_after, gt, match_config)
    r_plus = _completion_field(
        gt=gt,
        match=before,
        occupancy=occupancy_plus,
        image_valid_mask=image_valid_mask,
    )
    r_minus = _completion_field(
        gt=gt,
        match=after,
        occupancy=occupancy_minus,
        image_valid_mask=image_valid_mask,
    )
    increment = r_minus & ~r_plus
    return {
        "component": component,
        "occupancy_minus": occupancy_minus,
        "after": after,
        "completion_plus": r_plus,
        "completion_minus": r_minus,
        "increment": increment,
    }


def _geometry_rows(
    catalog: PreparedTrainingCatalog,
    geometry: GeometrySafeCatalog,
) -> dict[tuple[str, int, int], GeometryTargetRecord]:
    raw = {
        (entry.sample_id, candidate.gt_id, candidate.pred_id)
        for entry in catalog.entries
        for candidate in entry.decoder_visible_legal_candidates
    }
    rows = {
        (
            record.sample_id,
            record.evaluation_gt_id,
            int(record.pred_id),
        ): record
        for record in geometry.legal_records
    }
    if len(rows) != len(geometry.legal_records):
        raise RuntimeError("geometry legal ledger contains duplicate identities")
    if set(rows) != raw:
        raise RuntimeError(
            "geometry legal ledger and prepared visible candidates differ"
        )
    return rows


def _reason_tuple(reasons: Iterable[str]) -> tuple[str, ...]:
    values = tuple(sorted(set(reasons)))
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("pair exclusion reasons must be non-empty strings")
    return values


def _positive_example(
    *,
    entry: PreparedTrainingSource,
    group_id: str,
    record: GeometryTargetRecord,
    candidate_index: int,
    match_config: MatchConfig,
    occupancy_plus: Tensor,
    pred: InstanceMap,
    before: MatchResult,
    image_valid_mask: Tensor,
) -> tuple[PairExample | None, PairCatalogExclusion | None]:
    candidate = entry.decoder_visible_legal_candidates[candidate_index]
    if (
        record.sample_id != entry.sample_id
        or record.group_id != group_id
        or record.evaluation_gt_id != candidate.gt_id
        or record.pred_id != candidate.pred_id
    ):
        raise RuntimeError("geometry and prepared legal identities differ")
    truth = _deletion_truth(
        occupancy_plus=occupancy_plus,
        pred=pred,
        pred_id=candidate.pred_id,
        gt=entry.gt,
        before=before,
        image_valid_mask=image_valid_mask,
        match_config=match_config,
    )
    occupancy_minus = truth["occupancy_minus"]
    if not isinstance(occupancy_minus, Tensor):
        raise AssertionError("deletion truth returned a non-tensor occupancy")
    if not torch.equal(occupancy_minus, candidate.occupancy_after):
        raise RuntimeError("prepared candidate occupancy differs from exact deletion")
    after = truth["after"]
    component = truth["component"]
    increment = truth["increment"]
    if not isinstance(after, MatchResult) or not isinstance(
        component, Tensor
    ) or not isinstance(increment, Tensor):
        raise AssertionError("deletion truth contains invalid objects")

    reasons: list[str] = []
    if not record.analysis_eligible:
        reasons.extend(
            f"geometry:{reason}"
            for reason in record.analysis_exclusion_reasons
        )
    if record.native_gt_id is None:
        reasons.append("geometry:missing_native_gt_id")
    expected_after_pairs = {
        (pair.gt_id, pair.pred_id) for pair in before.pairs
    } - {(candidate.gt_id, candidate.pred_id)}
    after_pairs = {(pair.gt_id, pair.pred_id) for pair in after.pairs}
    if after_pairs != expected_after_pairs:
        reasons.append("other_match_identity_changed")
    if set(after.unmatched_gt_ids) != set(before.unmatched_gt_ids) | {
        candidate.gt_id
    }:
        reasons.append("selected_gt_not_unique_new_unmatched")
    if before.unmatched_gt_ids:
        preexisting_unmatched = torch.stack(
            [
                entry.gt.by_id(gt_id).mask
                for gt_id in sorted(before.unmatched_gt_ids)
            ],
            dim=0,
        ).any(dim=0)
        if torch.any(component & image_valid_mask & preexisting_unmatched):
            reasons.append("preexisting_unmatched_gt_interference")
    clean = (
        image_valid_mask
        & entry.gt.by_id(candidate.gt_id).mask
        & ~occupancy_minus
    )
    if not torch.equal(increment, clean):
        reasons.append("actual_increment_differs_from_clean_increment")
    if not torch.any(increment):
        reasons.append("empty_actual_increment")
    if not torch.any(image_valid_mask & ~increment):
        reasons.append("empty_zero_response_domain")
    projected_plus, projected_minus, visible = _projected_fingerprints(
        occupancy_plus,
        occupancy_minus,
        feature_size=tuple(entry.source.feature.shape[-2:]),
    )
    if not visible:
        reasons.append("projection_invisible")

    if reasons:
        return None, PairCatalogExclusion(
            pair_kind="clean_positive",
            sample_id=entry.sample_id,
            group_id=group_id,
            evaluation_gt_id=candidate.gt_id,
            native_gt_id=record.native_gt_id,
            pred_id=candidate.pred_id,
            reason_codes=_reason_tuple(reasons),
        )
    native_gt_id = record.native_gt_id
    if native_gt_id is None:
        raise AssertionError("eligible geometry row lost native_gt_id")
    example = PairExample(
        pair_id=_pair_id(
            kind="clean_positive",
            sample_id=entry.sample_id,
            evaluation_gt_id=candidate.gt_id,
            native_gt_id=native_gt_id,
            pred_id=candidate.pred_id,
        ),
        pair_kind="clean_positive",
        sample_id=entry.sample_id,
        group_id=group_id,
        feature=entry.source.feature,
        occupancy_plus=occupancy_plus.unsqueeze(0),
        occupancy_minus=occupancy_minus.unsqueeze(0),
        removed_component=component.unsqueeze(0),
        image_valid_mask=image_valid_mask.unsqueeze(0),
        completion_plus=truth["completion_plus"].unsqueeze(0),
        completion_minus=truth["completion_minus"].unsqueeze(0),
        label_increment=increment.unsqueeze(0).to(torch.float32),
        clean_increment=clean.unsqueeze(0),
        evaluation_gt_id=candidate.gt_id,
        native_gt_id=native_gt_id,
        pred_id=candidate.pred_id,
        feature_fingerprint=tensor_content_fingerprint(entry.source.feature),
        before_match_fingerprint=_match_fingerprint(before),
        after_match_fingerprint=_match_fingerprint(after),
        projected_occupancy_plus_fingerprint=projected_plus,
        projected_occupancy_minus_fingerprint=projected_minus,
        projection_visible=visible,
        geometry_safe_bijective_lineage=True,
        selected_gt_is_only_new_unmatched=True,
        other_match_identities_unchanged=True,
        preexisting_unmatched_gt_noninterference=True,
    )
    return example, None


def _component_null_example(
    *,
    entry: PreparedTrainingSource,
    group_id: str,
    pred_id: int,
    match_config: MatchConfig,
    occupancy_plus: Tensor,
    pred: InstanceMap,
    before: MatchResult,
    image_valid_mask: Tensor,
) -> tuple[PairExample | None, PairCatalogExclusion | None]:
    truth = _deletion_truth(
        occupancy_plus=occupancy_plus,
        pred=pred,
        pred_id=pred_id,
        gt=entry.gt,
        before=before,
        image_valid_mask=image_valid_mask,
        match_config=match_config,
    )
    occupancy_minus = truth["occupancy_minus"]
    after = truth["after"]
    component = truth["component"]
    increment = truth["increment"]
    if not isinstance(occupancy_minus, Tensor) or not isinstance(
        after, MatchResult
    ) or not isinstance(component, Tensor) or not isinstance(increment, Tensor):
        raise AssertionError("deletion truth contains invalid objects")
    projected_plus, projected_minus, visible = _projected_fingerprints(
        occupancy_plus,
        occupancy_minus,
        feature_size=tuple(entry.source.feature.shape[-2:]),
    )
    reasons: list[str] = []
    if not visible:
        reasons.append("projection_invisible")
    if torch.any(increment):
        reasons.append("actual_increment_nonempty")
    if reasons:
        return None, PairCatalogExclusion(
            pair_kind="component_null",
            sample_id=entry.sample_id,
            group_id=group_id,
            evaluation_gt_id=None,
            native_gt_id=None,
            pred_id=pred_id,
            reason_codes=_reason_tuple(reasons),
        )
    empty = torch.zeros_like(occupancy_plus)
    return PairExample(
        pair_id=_pair_id(
            kind="component_null",
            sample_id=entry.sample_id,
            evaluation_gt_id=None,
            native_gt_id=None,
            pred_id=pred_id,
        ),
        pair_kind="component_null",
        sample_id=entry.sample_id,
        group_id=group_id,
        feature=entry.source.feature,
        occupancy_plus=occupancy_plus.unsqueeze(0),
        occupancy_minus=occupancy_minus.unsqueeze(0),
        removed_component=component.unsqueeze(0),
        image_valid_mask=image_valid_mask.unsqueeze(0),
        completion_plus=truth["completion_plus"].unsqueeze(0),
        completion_minus=truth["completion_minus"].unsqueeze(0),
        label_increment=empty.unsqueeze(0).to(torch.float32),
        clean_increment=empty.unsqueeze(0),
        evaluation_gt_id=None,
        native_gt_id=None,
        pred_id=pred_id,
        feature_fingerprint=tensor_content_fingerprint(entry.source.feature),
        before_match_fingerprint=_match_fingerprint(before),
        after_match_fingerprint=_match_fingerprint(after),
        projected_occupancy_plus_fingerprint=projected_plus,
        projected_occupancy_minus_fingerprint=projected_minus,
        projection_visible=visible,
        geometry_safe_bijective_lineage=None,
        selected_gt_is_only_new_unmatched=None,
        other_match_identities_unchanged=None,
        preexisting_unmatched_gt_noninterference=None,
    ), None


def _identity_null_example(
    *,
    entry: PreparedTrainingSource,
    group_id: str,
    occupancy: Tensor,
    before: MatchResult,
    image_valid_mask: Tensor,
) -> PairExample:
    completion = _completion_field(
        gt=entry.gt,
        match=before,
        occupancy=occupancy,
        image_valid_mask=image_valid_mask,
    )
    projected = project_occupancy_to_feature_grid(
        occupancy.unsqueeze(0).unsqueeze(0),
        tuple(entry.source.feature.shape[-2:]),
    )
    projected_fingerprint = tensor_content_fingerprint(projected)
    empty = torch.zeros_like(occupancy)
    match_fingerprint = _match_fingerprint(before)
    return PairExample(
        pair_id=_pair_id(
            kind="identity_null",
            sample_id=entry.sample_id,
            evaluation_gt_id=None,
            native_gt_id=None,
            pred_id=None,
        ),
        pair_kind="identity_null",
        sample_id=entry.sample_id,
        group_id=group_id,
        feature=entry.source.feature,
        occupancy_plus=occupancy.unsqueeze(0),
        occupancy_minus=occupancy.unsqueeze(0),
        removed_component=empty.unsqueeze(0),
        image_valid_mask=image_valid_mask.unsqueeze(0),
        completion_plus=completion.unsqueeze(0),
        completion_minus=completion.unsqueeze(0),
        label_increment=empty.unsqueeze(0).to(torch.float32),
        clean_increment=empty.unsqueeze(0),
        evaluation_gt_id=None,
        native_gt_id=None,
        pred_id=None,
        feature_fingerprint=tensor_content_fingerprint(entry.source.feature),
        before_match_fingerprint=match_fingerprint,
        after_match_fingerprint=match_fingerprint,
        projected_occupancy_plus_fingerprint=projected_fingerprint,
        projected_occupancy_minus_fingerprint=projected_fingerprint,
        projection_visible=False,
        geometry_safe_bijective_lineage=None,
        selected_gt_is_only_new_unmatched=None,
        other_match_identities_unchanged=None,
        preexisting_unmatched_gt_noninterference=None,
    )


def build_pair_catalog(
    prepared: PreparedTrainingCatalog,
    geometry: GeometrySafeCatalog,
    manifest: SplitManifest,
    *,
    paired_protocol_fingerprint: str,
    match_config: MatchConfig = MatchConfig(),
) -> PairCatalog:
    """Construct clean-positive and null pairs from D_R only.

    Clean positives are restricted to the geometry-safe legal allowlist and
    independently revalidate matching, noninterference, label increment, and
    feature-grid visibility.  Component/identity nulls are control objects and
    never enter :attr:`PairCatalog.trainable_pairs`.
    """

    if not isinstance(prepared, PreparedTrainingCatalog):
        raise TypeError("prepared must be a PreparedTrainingCatalog")
    if not isinstance(geometry, GeometrySafeCatalog):
        raise TypeError("geometry must be a GeometrySafeCatalog")
    if not isinstance(manifest, SplitManifest):
        raise TypeError("manifest must be a SplitManifest")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be a MatchConfig")
    if (
        not isinstance(paired_protocol_fingerprint, str)
        or len(paired_protocol_fingerprint) != 64
    ):
        raise ValueError("paired_protocol_fingerprint must be a SHA256 string")
    records = manifest.records_for("D_R")
    group_by_sample = {
        record.sample_id: record.group_id
        for record in records
    }
    if set(group_by_sample) != set(prepared.source_ids):
        raise ValueError("manifest D_R identities differ from the prepared catalog")
    geometry_by_identity = _geometry_rows(prepared, geometry)

    positives: list[PairExample] = []
    component_nulls: list[PairExample] = []
    identity_nulls: list[PairExample] = []
    exclusions: list[PairCatalogExclusion] = []
    for entry in prepared.entries:
        group_id = group_by_sample[entry.sample_id]
        occupancy, pred, before, valid = _source_state(entry, match_config)
        identity_nulls.append(
            _identity_null_example(
                entry=entry,
                group_id=group_id,
                occupancy=occupancy,
                before=before,
                image_valid_mask=valid,
            )
        )
        for index, candidate in enumerate(
            entry.decoder_visible_legal_candidates
        ):
            record = geometry_by_identity[
                (entry.sample_id, candidate.gt_id, candidate.pred_id)
            ]
            example, exclusion = _positive_example(
                entry=entry,
                group_id=group_id,
                record=record,
                candidate_index=index,
                match_config=match_config,
                occupancy_plus=occupancy,
                pred=pred,
                before=before,
                image_valid_mask=valid,
            )
            if example is not None:
                positives.append(example)
            if exclusion is not None:
                exclusions.append(exclusion)
        for pred_id in pred.ids:
            example, exclusion = _component_null_example(
                entry=entry,
                group_id=group_id,
                pred_id=pred_id,
                match_config=match_config,
                occupancy_plus=occupancy,
                pred=pred,
                before=before,
                image_valid_mask=valid,
            )
            if example is not None:
                component_nulls.append(example)
            if exclusion is not None:
                exclusions.append(exclusion)

    def pair_order(item: PairExample) -> tuple[str, int, int, str]:
        return (
            item.sample_id,
            -1
            if item.evaluation_gt_id is None
            else item.evaluation_gt_id,
            -1 if item.pred_id is None else item.pred_id,
            item.pair_id,
        )

    positives_tuple = tuple(sorted(positives, key=pair_order))
    component_null_tuple = tuple(
        sorted(component_nulls, key=pair_order)
    )
    identity_null_tuple = tuple(
        sorted(identity_nulls, key=pair_order)
    )
    exclusions_tuple = tuple(
        sorted(
            exclusions,
            key=lambda item: (
                item.pair_kind,
                item.sample_id,
                -1
                if item.evaluation_gt_id is None
                else item.evaluation_gt_id,
                -1 if item.pred_id is None else item.pred_id,
            ),
        )
    )
    unsealed = PairCatalog(
        dataset=manifest.dataset,
        split="D_R",
        paired_protocol_fingerprint=paired_protocol_fingerprint,
        geometry_catalog_fingerprint=geometry.catalog_fingerprint,
        source_catalog_fingerprint=geometry.source_catalog_fingerprint,
        manifest_fingerprint=manifest.fingerprint,
        clean_positive=positives_tuple,
        component_null=component_null_tuple,
        identity_null=identity_null_tuple,
        exclusions=exclusions_tuple,
        catalog_fingerprint="",
    )
    result = replace(
        unsealed,
        catalog_fingerprint=stable_fingerprint(unsealed.canonical_payload()),
    )
    if result.catalog_fingerprint != stable_fingerprint(result.canonical_payload()):
        raise AssertionError("pair catalog fingerprint is unstable")
    return result


__all__ = [
    "PAIR_CATALOG_SCHEMA",
    "build_pair_catalog",
]
