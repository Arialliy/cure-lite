"""Representation-neutral coverage-state records for CURE-Lite.

This module deliberately stops before scalar max projection or phase
preservation is chosen.  It binds frozen Base features, full-resolution
coverage states, completion targets, and source identities on ``D_R`` so the
two candidate representations can be audited over exactly the same
population.

The older :mod:`cure_lite.paired_types` objects require a scalar-projection
visible deletion.  Reusing them here would remove the very examples needed
to decide whether scalar projection is adequate, so the raw contract is kept
separate and additive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cache.schema import stable_fingerprint
from .paired_types import tensor_content_fingerprint


COVERAGE_STATE_RAW_CATALOG_SCHEMA = "cure-lite-coverage-state-raw-catalog-v2"
COVERAGE_STATE_SCENE_TARGET_POLICY = (
    "single_scene_complete_completion_field_per_actual_input_v1"
)
COVERAGE_STATE_NATURAL_FOCUS_POLICY = (
    "per_target_focus_changes_measure_not_absolute_field_v1"
)
COVERAGE_STATE_NATURAL_KINDS = ("factual_miss", "factual_no_miss")
COVERAGE_STATE_PAIR_KINDS = (
    "clean_positive",
    "component_null",
    "identity_null",
)


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    text = _nonempty_text(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA256 fingerprint")
    return text


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _freeze_feature(value: Tensor, *, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 4
        or value.shape[0] != 1
        or value.shape[1] < 1
        or min(value.shape[-2:]) < 1
    ):
        raise TypeError(f"{name} must be detached CPU float32 [1,C,h,w]")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    # One source feature is intentionally shared by all of its natural and
    # paired records.  Copying it per endpoint would recreate the redundant
    # cache cost this layer is meant to remove.
    return value.detach().contiguous()


def _freeze_mask(value: Tensor, *, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.bool
        or value.ndim != 4
        or value.shape[:2] != (1, 1)
        or min(value.shape[-2:]) < 1
    ):
        raise TypeError(f"{name} must be CPU bool [1,1,H,W]")
    return value.detach().clone().contiguous()


def _identity_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{name} must be sorted and unique")
    return value


def _positive_id(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be null or a positive integer")
    return value


def _positive_id_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in value
    ):
        raise ValueError(f"{name} must contain positive integers")
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{name} must be sorted and unique")
    return value


def _validate_state_grid(
    *,
    feature: Tensor,
    occupancy: Tensor,
    target: Tensor,
    valid_mask: Tensor,
) -> None:
    if not (
        occupancy.shape == target.shape == valid_mask.shape
    ):
        raise ValueError("occupancy, target, and valid_mask shapes must match")
    if not bool(torch.any(valid_mask)):
        raise ValueError("valid_mask cannot be empty")
    if bool(torch.any(target & ~valid_mask)):
        raise ValueError("target must remain inside valid_mask")
    if bool(torch.any(occupancy & target)):
        raise ValueError("completion target must be writable under occupancy")
    if feature.shape[0] != occupancy.shape[0]:
        raise ValueError("feature and output-grid batch sizes differ")


@dataclass(frozen=True, eq=False)
class CoverageStateNaturalRecord:
    """One natural factual completion state from the frozen ``D_R`` cache."""

    record_id: str
    sample_id: str
    group_id: str
    state_kind: str
    feature: Tensor
    occupancy: Tensor
    target: Tensor
    valid_mask: Tensor
    loss_valid_mask: Tensor
    target_ids: tuple[str, ...]
    focus_target_ids: tuple[str, ...]
    source_row_fingerprint: str
    evaluation_gt_ids: tuple[int, ...]
    native_gt_ids: tuple[int, ...]
    lineage_record_fingerprint: str | None

    def __post_init__(self) -> None:
        _nonempty_text(self.record_id, name="record_id")
        _nonempty_text(self.sample_id, name="sample_id")
        _nonempty_text(self.group_id, name="group_id")
        if self.state_kind not in COVERAGE_STATE_NATURAL_KINDS:
            raise ValueError(f"unknown natural state_kind {self.state_kind!r}")
        feature = _freeze_feature(self.feature, name="feature")
        occupancy = _freeze_mask(self.occupancy, name="occupancy")
        target = _freeze_mask(self.target, name="target")
        valid = _freeze_mask(self.valid_mask, name="valid_mask")
        loss_valid = _freeze_mask(
            self.loss_valid_mask,
            name="loss_valid_mask",
        )
        if loss_valid.shape != valid.shape:
            raise ValueError("loss_valid_mask and field valid_mask shapes differ")
        _validate_state_grid(
            feature=feature,
            occupancy=occupancy,
            target=target,
            valid_mask=valid,
        )
        if bool(torch.any(occupancy & ~valid)):
            raise ValueError("natural occupancy must remain inside field valid_mask")
        if (
            not bool(torch.any(loss_valid))
            or bool(torch.any(loss_valid & ~valid))
            or bool(torch.any(loss_valid & occupancy))
        ):
            raise ValueError(
                "loss_valid_mask must be nonempty, field-valid, and writable"
            )
        target_ids = _identity_tuple(self.target_ids, name="target_ids")
        focus_ids = _identity_tuple(
            self.focus_target_ids,
            name="focus_target_ids",
        )
        _sha256(
            self.source_row_fingerprint,
            name="source_row_fingerprint",
        )
        evaluation_ids = _positive_id_tuple(
            self.evaluation_gt_ids,
            name="evaluation_gt_ids",
        )
        native_ids = _positive_id_tuple(
            self.native_gt_ids,
            name="native_gt_ids",
        )
        lineage = self.lineage_record_fingerprint
        if lineage is not None:
            _sha256(lineage, name="lineage_record_fingerprint")
        has_target = bool(torch.any(target))
        if self.state_kind == "factual_miss":
            if (
                not has_target
                or not target_ids
                or len(target_ids) != len(evaluation_ids)
                or len(target_ids) != len(native_ids)
                or len(focus_ids) != 1
                or not set(focus_ids) <= set(target_ids)
                or lineage is None
            ):
                raise ValueError(
                    "factual_miss requires a scene-complete target set, one "
                    "focus target, aligned evaluation/native IDs, and lineage"
                )
        elif (
            has_target
            or target_ids
            or focus_ids
            or evaluation_ids
            or native_ids
            or lineage is not None
        ):
            raise ValueError("factual_no_miss requires an empty target and no target IDs")
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "loss_valid_mask", loss_valid)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "focus_target_ids", focus_ids)
        object.__setattr__(self, "evaluation_gt_ids", evaluation_ids)
        object.__setattr__(self, "native_gt_ids", native_ids)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "state_kind": self.state_kind,
            "target_ids": list(self.target_ids),
            "focus_target_ids": list(self.focus_target_ids),
            "source_row_fingerprint": self.source_row_fingerprint,
            "evaluation_gt_ids": list(self.evaluation_gt_ids),
            "native_gt_ids": list(self.native_gt_ids),
            "lineage_record_fingerprint": self.lineage_record_fingerprint,
            "tensors": {
                "feature": tensor_content_fingerprint(self.feature),
                "occupancy": tensor_content_fingerprint(self.occupancy),
                "target": tensor_content_fingerprint(self.target),
                "valid_mask": tensor_content_fingerprint(self.valid_mask),
                "loss_valid_mask": tensor_content_fingerprint(
                    self.loss_valid_mask
                ),
            },
        }


@dataclass(frozen=True, eq=False)
class CoverageStatePairRecord:
    """One full-grid coverage intervention before representation selection."""

    pair_id: str
    sample_id: str
    group_id: str
    pair_kind: str
    feature: Tensor
    occupancy_plus: Tensor
    occupancy_minus: Tensor
    target_plus: Tensor
    target_minus: Tensor
    valid_mask: Tensor
    removed_component: Tensor
    removed_component_ids: tuple[str, ...]
    target_ids_added: tuple[str, ...]
    source_row_fingerprint: str
    evaluation_gt_id: int | None
    native_gt_id: int | None
    pred_id: int | None
    before_match_fingerprint: str
    after_match_fingerprint: str
    lineage_record_fingerprint: str | None

    def __post_init__(self) -> None:
        _nonempty_text(self.pair_id, name="pair_id")
        _nonempty_text(self.sample_id, name="sample_id")
        _nonempty_text(self.group_id, name="group_id")
        if self.pair_kind not in COVERAGE_STATE_PAIR_KINDS:
            raise ValueError(f"unknown pair_kind {self.pair_kind!r}")
        feature = _freeze_feature(self.feature, name="feature")
        plus = _freeze_mask(self.occupancy_plus, name="occupancy_plus")
        minus = _freeze_mask(self.occupancy_minus, name="occupancy_minus")
        target_plus = _freeze_mask(self.target_plus, name="target_plus")
        target_minus = _freeze_mask(self.target_minus, name="target_minus")
        valid = _freeze_mask(self.valid_mask, name="valid_mask")
        removed = _freeze_mask(self.removed_component, name="removed_component")
        if not (
            plus.shape
            == minus.shape
            == target_plus.shape
            == target_minus.shape
            == valid.shape
            == removed.shape
        ):
            raise ValueError("all pair output-grid tensors must share one shape")
        _validate_state_grid(
            feature=feature,
            occupancy=plus,
            target=target_plus,
            valid_mask=valid,
        )
        _validate_state_grid(
            feature=feature,
            occupancy=minus,
            target=target_minus,
            valid_mask=valid,
        )
        if bool(torch.any(minus & ~plus)):
            raise ValueError("occupancy_minus must be a subset of occupancy_plus")
        if bool(torch.any((plus | minus) & ~valid)):
            raise ValueError("paired occupancy endpoints must remain inside valid_mask")
        if not torch.equal(removed, plus & ~minus):
            raise ValueError(
                "removed_component must equal occupancy_plus minus occupancy_minus"
            )
        if bool(torch.any(target_plus & ~target_minus)):
            raise ValueError("a deletion pair may only add completion support")
        removed_ids = _identity_tuple(
            self.removed_component_ids,
            name="removed_component_ids",
        )
        target_ids = _identity_tuple(
            self.target_ids_added,
            name="target_ids_added",
        )
        _sha256(
            self.source_row_fingerprint,
            name="source_row_fingerprint",
        )
        evaluation_gt_id = _positive_id(
            self.evaluation_gt_id,
            name="evaluation_gt_id",
        )
        native_gt_id = _positive_id(
            self.native_gt_id,
            name="native_gt_id",
        )
        pred_id = _positive_id(self.pred_id, name="pred_id")
        _sha256(
            self.before_match_fingerprint,
            name="before_match_fingerprint",
        )
        _sha256(
            self.after_match_fingerprint,
            name="after_match_fingerprint",
        )
        lineage = self.lineage_record_fingerprint
        if lineage is not None:
            _sha256(lineage, name="lineage_record_fingerprint")
        occupancy_equal = torch.equal(plus, minus)
        targets_equal = torch.equal(target_plus, target_minus)
        if self.pair_kind == "clean_positive":
            if (
                occupancy_equal
                or targets_equal
                or len(removed_ids) != 1
                or len(target_ids) != 1
                or evaluation_gt_id is None
                or native_gt_id is None
                or pred_id is None
                or lineage is None
            ):
                raise ValueError(
                    "clean_positive requires a deletion, one target, and "
                    "evaluation/native/pred lineage identities"
                )
        elif self.pair_kind == "component_null":
            if (
                occupancy_equal
                or not targets_equal
                or len(removed_ids) != 1
                or target_ids
                or evaluation_gt_id is not None
                or native_gt_id is not None
                or pred_id is None
                or lineage is not None
            ):
                raise ValueError(
                    "component_null requires a deletion with unchanged target"
                )
        else:
            if (
                not occupancy_equal
                or not targets_equal
                or bool(torch.any(removed))
                or removed_ids
                or target_ids
                or evaluation_gt_id is not None
                or native_gt_id is not None
                or pred_id is not None
                or lineage is not None
                or self.before_match_fingerprint
                != self.after_match_fingerprint
            ):
                raise ValueError(
                    "identity_null must preserve feature-independent state exactly"
                )
        for name, value in (
            ("feature", feature),
            ("occupancy_plus", plus),
            ("occupancy_minus", minus),
            ("target_plus", target_plus),
            ("target_minus", target_minus),
            ("valid_mask", valid),
            ("removed_component", removed),
            ("removed_component_ids", removed_ids),
            ("target_ids_added", target_ids),
            ("evaluation_gt_id", evaluation_gt_id),
            ("native_gt_id", native_gt_id),
            ("pred_id", pred_id),
        ):
            object.__setattr__(self, name, value)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "pair_kind": self.pair_kind,
            "removed_component_ids": list(self.removed_component_ids),
            "target_ids_added": list(self.target_ids_added),
            "source_row_fingerprint": self.source_row_fingerprint,
            "evaluation_gt_id": self.evaluation_gt_id,
            "native_gt_id": self.native_gt_id,
            "pred_id": self.pred_id,
            "before_match_fingerprint": self.before_match_fingerprint,
            "after_match_fingerprint": self.after_match_fingerprint,
            "lineage_record_fingerprint": self.lineage_record_fingerprint,
            "tensors": {
                "feature": tensor_content_fingerprint(self.feature),
                "occupancy_plus": tensor_content_fingerprint(self.occupancy_plus),
                "occupancy_minus": tensor_content_fingerprint(self.occupancy_minus),
                "target_plus": tensor_content_fingerprint(self.target_plus),
                "target_minus": tensor_content_fingerprint(self.target_minus),
                "valid_mask": tensor_content_fingerprint(self.valid_mask),
                "removed_component": tensor_content_fingerprint(
                    self.removed_component
                ),
            },
        }


@dataclass(frozen=True)
class CoverageStateRawExclusion:
    """One explicitly rejected raw candidate with representation-free reasons."""

    candidate_kind: str
    sample_id: str
    evaluation_gt_id: int | None
    pred_id: int | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.candidate_kind not in {
            "factual_unreachable",
            "factual_geometry",
            "clean_positive",
            "component_null",
        }:
            raise ValueError("unknown raw candidate_kind")
        _nonempty_text(self.sample_id, name="sample_id")
        _positive_id(self.evaluation_gt_id, name="evaluation_gt_id")
        _positive_id(self.pred_id, name="pred_id")
        if (
            not isinstance(self.reason_codes, tuple)
            or not self.reason_codes
            or self.reason_codes != tuple(sorted(set(self.reason_codes)))
            or any(not isinstance(value, str) or not value for value in self.reason_codes)
        ):
            raise ValueError("reason_codes must be a sorted unique non-empty tuple")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "candidate_kind": self.candidate_kind,
            "sample_id": self.sample_id,
            "evaluation_gt_id": self.evaluation_gt_id,
            "pred_id": self.pred_id,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, eq=False)
class CoverageStateRawCatalog:
    """Complete representation-neutral population for one frozen detector."""

    dataset: str
    split: str
    feature_stride: int
    source_fingerprint: str
    natural_records: tuple[CoverageStateNaturalRecord, ...]
    pair_records: tuple[CoverageStatePairRecord, ...]
    exclusions: tuple[CoverageStateRawExclusion, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_text(self.dataset, name="dataset")
        if self.split != "D_R":
            raise ValueError("coverage-state raw catalog permits only D_R")
        stride = _positive_int(self.feature_stride, name="feature_stride")
        _sha256(self.source_fingerprint, name="source_fingerprint")
        if not isinstance(self.natural_records, tuple) or any(
            not isinstance(value, CoverageStateNaturalRecord)
            for value in self.natural_records
        ):
            raise TypeError("natural_records must contain natural record values")
        if not isinstance(self.pair_records, tuple) or any(
            not isinstance(value, CoverageStatePairRecord)
            for value in self.pair_records
        ):
            raise TypeError("pair_records must contain pair record values")
        if not self.natural_records or not self.pair_records:
            raise ValueError("raw catalog requires natural and paired populations")
        if not isinstance(self.exclusions, tuple) or any(
            not isinstance(value, CoverageStateRawExclusion)
            for value in self.exclusions
        ):
            raise TypeError("exclusions must contain CoverageStateRawExclusion values")
        natural_keys = tuple(value.record_id for value in self.natural_records)
        pair_keys = tuple(value.pair_id for value in self.pair_records)
        if natural_keys != tuple(sorted(set(natural_keys))):
            raise ValueError("natural records must be canonically ordered and unique")
        if pair_keys != tuple(sorted(set(pair_keys))):
            raise ValueError("pair records must be canonically ordered and unique")
        exclusion_keys = tuple(
            (
                value.candidate_kind,
                value.sample_id,
                -1
                if value.evaluation_gt_id is None
                else value.evaluation_gt_id,
                -1 if value.pred_id is None else value.pred_id,
            )
            for value in self.exclusions
        )
        if exclusion_keys != tuple(sorted(set(exclusion_keys))):
            raise ValueError("exclusions must be canonically ordered and unique")
        grids = {
            (
                tuple(value.feature.shape[1:]),
                tuple(value.occupancy.shape[1:]),
            )
            for value in self.natural_records
        }
        grids.update(
            (
                tuple(value.feature.shape[1:]),
                tuple(value.occupancy_plus.shape[1:]),
            )
            for value in self.pair_records
        )
        if len(grids) != 1:
            raise ValueError("all raw records must share feature and output grids")
        (feature_grid, output_grid), = grids
        expected_output = (
            1,
            int(feature_grid[-2]) * stride,
            int(feature_grid[-1]) * stride,
        )
        if output_grid != expected_output:
            raise ValueError(
                "output grid must equal feature grid times feature_stride"
            )
        object.__setattr__(self, "feature_stride", stride)

    @property
    def catalog_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    @property
    def pair_kind_counts(self) -> dict[str, int]:
        return {
            kind: sum(record.pair_kind == kind for record in self.pair_records)
            for kind in COVERAGE_STATE_PAIR_KINDS
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_RAW_CATALOG_SCHEMA,
            "dataset": self.dataset,
            "split": self.split,
            "feature_stride": self.feature_stride,
            "source_fingerprint": self.source_fingerprint,
            "scene_target_policy": COVERAGE_STATE_SCENE_TARGET_POLICY,
            "natural_focus_policy": COVERAGE_STATE_NATURAL_FOCUS_POLICY,
            "counts": {
                "natural": len(self.natural_records),
                "pairs": len(self.pair_records),
                "exclusions": len(self.exclusions),
                **self.pair_kind_counts,
            },
            "natural_records": [
                value.canonical_payload() for value in self.natural_records
            ],
            "pair_records": [
                value.canonical_payload() for value in self.pair_records
            ],
            "exclusions": [
                value.canonical_payload() for value in self.exclusions
            ],
        }


def make_coverage_state_raw_catalog(
    *,
    dataset: str,
    feature_stride: int,
    source_fingerprint: str,
    natural_records: tuple[CoverageStateNaturalRecord, ...],
    pair_records: tuple[CoverageStatePairRecord, ...],
    exclusions: tuple[CoverageStateRawExclusion, ...] = (),
) -> CoverageStateRawCatalog:
    """Canonicalize an already materialized in-memory ``D_R`` population.

    This low-level constructor performs no file access and is suitable for
    deterministic unit tests.  The later real-cache adapter must create the
    same records directly from the strict ``D_R`` bundle, before any scalar
    visibility filter.
    """

    if not isinstance(natural_records, tuple) or not isinstance(
        pair_records,
        tuple,
    ):
        raise TypeError("natural_records and pair_records must be tuples")
    return CoverageStateRawCatalog(
        dataset=dataset,
        split="D_R",
        feature_stride=feature_stride,
        source_fingerprint=source_fingerprint,
        natural_records=tuple(
            sorted(natural_records, key=lambda value: value.record_id)
        ),
        pair_records=tuple(
            sorted(pair_records, key=lambda value: value.pair_id)
        ),
        exclusions=tuple(
            sorted(
                exclusions,
                key=lambda value: (
                    value.candidate_kind,
                    value.sample_id,
                    -1
                    if value.evaluation_gt_id is None
                    else value.evaluation_gt_id,
                    -1 if value.pred_id is None else value.pred_id,
                ),
            )
        ),
    )


__all__ = [
    "COVERAGE_STATE_NATURAL_KINDS",
    "COVERAGE_STATE_PAIR_KINDS",
    "COVERAGE_STATE_RAW_CATALOG_SCHEMA",
    "COVERAGE_STATE_SCENE_TARGET_POLICY",
    "COVERAGE_STATE_NATURAL_FOCUS_POLICY",
    "CoverageStateNaturalRecord",
    "CoverageStatePairRecord",
    "CoverageStateRawExclusion",
    "CoverageStateRawCatalog",
    "make_coverage_state_raw_catalog",
]
