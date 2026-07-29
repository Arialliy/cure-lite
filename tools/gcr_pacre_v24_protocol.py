"""Pure protocol utilities for the CURE-Lite v24 GCR-PACRE evidence chain.

This module deliberately has no model, dataset, tensor, optimizer, training,
or evaluation entry point.  It operates only on already materialized metadata
and sufficient statistics.  In particular, importing it cannot read D_V or
D_T and cannot authorize either split.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


PROTOCOL_ID = "irstd1k-gcr-pacre-v24-evidence-v1"
OOF_SCHEMA = "cure-lite-v24-gcr-pacre-oof4-plan-v1"
OOF_NAMESPACE = "cure-lite-v24-gcr-pacre-root-oof4-v1"
OOF_FOLD_COUNT = 4
OOF_SEED = 42
OOF_ARMS = (
    "BaseA",
    "BaseB_train_fold_selected",
    "PACRE_VC_v23_control",
    "GCR_PACRE_v24",
    "GCR_PACRE_v24_forced_G1",
)
BASE_A_THRESHOLD = 0.72
BASE_B_THRESHOLD_GRID = tuple(index / 50 for index in range(51))
FORMAL_EPOCHS = 800
FORMAL_STEPS_PER_EPOCH = 40
FORMAL_UPDATES = FORMAL_EPOCHS * FORMAL_STEPS_PER_EPOCH
BOUNDED_EPOCHS = 10
BOUNDED_STEPS_PER_EPOCH = 40
BOUNDED_UPDATES = BOUNDED_EPOCHS * BOUNDED_STEPS_PER_EPOCH
PIXEL_FA_LIMIT = 1.0e-4
RAW_BACKGROUND_FA_LIMIT = 1.0e-4
FP_COMPONENTS_PER_MP_LIMIT = 100.0
ROOT_GROUP_FIELDS = (
    "group_id",
    "scene_id",
    "sequence_id",
    "crop_source_id",
    "near_duplicate_group",
)

_HEX = frozenset("0123456789abcdef")
_STAT_INTEGER_FIELDS = (
    "images",
    "matched_gt",
    "total_gt",
    "recovered_anchor_misses",
    "overlap_supported_recovered_anchor_misses",
    "total_anchor_misses",
    "retained_anchor_covered",
    "total_anchor_covered",
    "recovered_reachable_anchor_misses",
    "total_reachable_anchor_misses",
    "unmatched_pred_pixels",
    "unmatched_pred_components",
    "raw_background_fp",
    "total_pixels",
    "intersection",
    "union",
)


def canonical_json(value: object) -> str:
    """Return strict deterministic JSON used by protocol fingerprints."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_fingerprint(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return dict(value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _text(value, name=name)
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return digest


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _real(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be real")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_json(path: Path) -> dict[str, object]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError(f"invalid protocol JSON file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read strict JSON: {path}") from error
    return _mapping(value, name=str(path))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def derive_root_source_ids(
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    split: str = "D_R",
) -> dict[str, str]:
    """Derive transitive root IDs from every provenance key in the manifest.

    Rows are connected when they share an equal value under the same grouping
    field.  Connected components, rather than a one-field precedence rule,
    close transitive aliases such as sample A sharing a scene with B while B
    shares a crop source with C.  The resulting root ID is content-derived and
    does not expose a filesystem path.
    """

    split = _text(split, name="split")
    normalized: list[dict[str, object]] = []
    sample_ids: set[str] = set()
    grouping_owner: dict[tuple[str, str], str] = {}
    for index, raw in enumerate(manifest_rows):
        row = _mapping(raw, name=f"manifest_rows[{index}]")
        sample_id = _text(row.get("sample_id"), name=f"row[{index}].sample_id")
        row_split = _text(row.get("split"), name=f"row[{index}].split")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate manifest sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        normalized.append(row)
        for field_name in ROOT_GROUP_FIELDS:
            raw_value = row.get(field_name)
            if raw_value is None:
                continue
            value = _text(
                raw_value,
                name=f"row[{index}].{field_name}",
            )
            key = (field_name, value)
            owner = grouping_owner.setdefault(key, row_split)
            if owner != row_split:
                raise ValueError(
                    f"{field_name}={value!r} crosses {owner}/{row_split}"
                )

    selected = [
        row for row in normalized if str(row["split"]) == split
    ]
    if len(selected) < OOF_FOLD_COUNT:
        raise ValueError(
            f"{split} needs at least {OOF_FOLD_COUNT} samples for OOF4"
        )
    selected_ids = tuple(str(row["sample_id"]) for row in selected)
    union_find = _UnionFind(selected_ids)
    owners: dict[tuple[str, str], str] = {}
    for row in selected:
        sample_id = str(row["sample_id"])
        for field_name in ROOT_GROUP_FIELDS:
            raw_value = row.get(field_name)
            if raw_value is None:
                continue
            key = (field_name, str(raw_value))
            prior = owners.setdefault(key, sample_id)
            union_find.union(prior, sample_id)

    components: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        components[union_find.find(str(row["sample_id"]))].append(row)

    root_by_sample: dict[str, str] = {}
    for rows in components.values():
        members = sorted(str(row["sample_id"]) for row in rows)
        keys = sorted(
            {
                f"{field_name}:{row[field_name]}"
                for row in rows
                for field_name in ROOT_GROUP_FIELDS
                if row.get(field_name) is not None
            }
        )
        root_id = "root-source-" + stable_fingerprint(
            {
                "schema_version": "cure-lite-root-source-component-v1",
                "split": split,
                "member_sample_ids": members,
                "provenance_keys": keys,
            }
        )
        for sample_id in members:
            root_by_sample[sample_id] = root_id
    if set(root_by_sample) != set(selected_ids):
        raise AssertionError("root-source derivation lost manifest samples")
    return dict(sorted(root_by_sample.items()))


def propagate_root_source_ids(
    rows: Sequence[Mapping[str, object]],
    root_by_sample: Mapping[str, str],
) -> tuple[dict[str, object], ...]:
    """Attach the authoritative root ID to derived metadata rows."""

    roots = {
        _text(sample_id, name="root_by_sample key"): _text(
            root_id,
            name=f"root_by_sample[{sample_id!r}]",
        )
        for sample_id, root_id in root_by_sample.items()
    }
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, name=f"rows[{index}]")
        unit_id = _text(row.get("unit_id"), name=f"rows[{index}].unit_id")
        sample_id = _text(
            row.get("sample_id"),
            name=f"rows[{index}].sample_id",
        )
        if unit_id in seen:
            raise ValueError(f"duplicate derived unit_id {unit_id!r}")
        seen.add(unit_id)
        if sample_id not in roots:
            raise ValueError(f"unknown root source for sample {sample_id!r}")
        expected = roots[sample_id]
        supplied = row.get("root_source_id")
        if supplied is not None and supplied != expected:
            raise ValueError(f"root_source_id changed for {unit_id!r}")
        result.append({**row, "root_source_id": expected})
    validate_root_source_closure(result)
    return tuple(result)


def validate_root_source_closure(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reject missing roots, cross-root parent edges, and sample rebinding."""

    normalized: dict[str, dict[str, object]] = {}
    root_by_sample: dict[str, str] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, name=f"rows[{index}]")
        unit_id = _text(row.get("unit_id"), name=f"rows[{index}].unit_id")
        sample_id = _text(
            row.get("sample_id"),
            name=f"rows[{index}].sample_id",
        )
        root_id = _text(
            row.get("root_source_id"),
            name=f"rows[{index}].root_source_id",
        )
        if unit_id in normalized:
            raise ValueError(f"duplicate unit_id {unit_id!r}")
        prior = root_by_sample.setdefault(sample_id, root_id)
        if prior != root_id:
            raise ValueError(f"sample {sample_id!r} was rebound across roots")
        parents = row.get("parent_unit_ids", [])
        if not isinstance(parents, list) or any(
            not isinstance(value, str) or not value for value in parents
        ):
            raise TypeError(f"{unit_id}.parent_unit_ids must be a text list")
        if len(parents) != len(set(parents)) or unit_id in parents:
            raise ValueError(f"{unit_id} has invalid parent edges")
        normalized[unit_id] = {**row, "parent_unit_ids": parents}

    for unit_id, row in normalized.items():
        for parent_id in row["parent_unit_ids"]:
            parent = normalized.get(str(parent_id))
            if parent is None:
                raise ValueError(
                    f"{unit_id!r} references absent parent {parent_id!r}"
                )
            if parent["root_source_id"] != row["root_source_id"]:
                raise ValueError(
                    f"{unit_id!r} crosses root_source_id through {parent_id!r}"
                )
    roots = sorted(
        {str(row["root_source_id"]) for row in normalized.values()}
    )
    body = {
        "schema_version": "cure-lite-v24-root-source-closure-v1",
        "unit_count": len(normalized),
        "sample_count": len(root_by_sample),
        "root_source_count": len(roots),
        "root_source_ids": roots,
        "unit_roots": {
            unit_id: str(row["root_source_id"])
            for unit_id, row in sorted(normalized.items())
        },
    }
    return {**body, "closure_fingerprint": stable_fingerprint(body)}


def deterministic_oof4_plan(
    root_by_sample: Mapping[str, str],
    *,
    seed: int = OOF_SEED,
) -> dict[str, object]:
    """Create the exact root-disjoint four-fold split.

    Roots are ordered by a namespaced SHA256 rank and assigned round-robin.
    This guarantees fold root counts differ by at most one and avoids Python
    hash randomization.
    """

    seed = _integer(seed, name="seed")
    normalized = {
        _text(sample_id, name="sample_id"): _text(
            root_id,
            name=f"root_by_sample[{sample_id!r}]",
        )
        for sample_id, root_id in root_by_sample.items()
    }
    roots = sorted(set(normalized.values()))
    if len(roots) < OOF_FOLD_COUNT:
        raise ValueError("OOF4 requires at least four distinct root sources")
    ordered_roots = sorted(
        roots,
        key=lambda root_id: (
            sha256(
                f"{OOF_NAMESPACE}|{seed}|{root_id}".encode("utf-8")
            ).hexdigest(),
            root_id,
        ),
    )
    held_out_by_fold = [
        ordered_roots[index::OOF_FOLD_COUNT]
        for index in range(OOF_FOLD_COUNT)
    ]
    folds: list[dict[str, object]] = []
    all_roots = set(roots)
    for fold_index, held_out_roots in enumerate(held_out_by_fold):
        held_out_set = set(held_out_roots)
        train_roots = sorted(all_roots - held_out_set)
        held_out_samples = sorted(
            sample_id
            for sample_id, root_id in normalized.items()
            if root_id in held_out_set
        )
        train_samples = sorted(set(normalized) - set(held_out_samples))
        if set(train_roots) & held_out_set:
            raise AssertionError("OOF root split leaked")
        folds.append(
            {
                "fold_id": fold_index,
                "held_out_root_source_ids": sorted(held_out_roots),
                "train_root_source_ids": train_roots,
                "held_out_sample_ids": held_out_samples,
                "train_sample_ids": train_samples,
            }
        )
    body = {
        "schema_version": OOF_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "namespace": OOF_NAMESPACE,
        "seed": seed,
        "fold_count": OOF_FOLD_COUNT,
        "assignment_policy": (
            "sha256_ranked_root_source_round_robin_no_python_hash_v1"
        ),
        "root_source_count": len(roots),
        "sample_count": len(normalized),
        "folds": folds,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {**body, "plan_fingerprint": stable_fingerprint(body)}


def verify_oof4_plan(
    plan: Mapping[str, object],
    root_by_sample: Mapping[str, str],
) -> str:
    payload = _mapping(plan, name="plan")
    expected = deterministic_oof4_plan(
        root_by_sample,
        seed=_integer(payload.get("seed"), name="plan.seed"),
    )
    if payload != expected:
        raise ValueError("OOF4 plan differs from deterministic root split")
    return _sha256(
        payload.get("plan_fingerprint"),
        name="plan.plan_fingerprint",
    )


@dataclass(frozen=True, slots=True)
class FactualSufficientStatistics:
    """Additive statistics that exactly recover the shared aggregate metrics."""

    images: int
    matched_gt: int
    total_gt: int
    recovered_anchor_misses: int
    overlap_supported_recovered_anchor_misses: int
    total_anchor_misses: int
    retained_anchor_covered: int
    total_anchor_covered: int
    recovered_reachable_anchor_misses: int
    total_reachable_anchor_misses: int
    unmatched_pred_pixels: int
    unmatched_pred_components: int
    raw_background_fp: int
    total_pixels: int
    intersection: int
    union: int
    sum_image_iou: float

    def __post_init__(self) -> None:
        for name in _STAT_INTEGER_FIELDS:
            _integer(getattr(self, name), name=name)
        iou_sum = _real(
            self.sum_image_iou,
            name="sum_image_iou",
            minimum=0.0,
        )
        object.__setattr__(self, "sum_image_iou", iou_sum)
        if self.images < 1:
            raise ValueError("statistics must contain at least one image")
        comparisons = (
            (self.matched_gt, self.total_gt, "matched_gt"),
            (
                self.recovered_anchor_misses,
                self.total_anchor_misses,
                "recovered_anchor_misses",
            ),
            (
                self.overlap_supported_recovered_anchor_misses,
                self.recovered_anchor_misses,
                "overlap-supported recovery",
            ),
            (
                self.retained_anchor_covered,
                self.total_anchor_covered,
                "retained_anchor_covered",
            ),
            (
                self.recovered_reachable_anchor_misses,
                self.total_reachable_anchor_misses,
                "reachable recovery",
            ),
            (
                self.total_reachable_anchor_misses,
                self.total_anchor_misses,
                "reachable misses",
            ),
            (
                self.unmatched_pred_pixels,
                self.total_pixels,
                "unmatched_pred_pixels",
            ),
            (
                self.raw_background_fp,
                self.total_pixels,
                "raw_background_fp",
            ),
            (self.intersection, self.union, "intersection"),
            (self.union, self.total_pixels, "union"),
        )
        for numerator, denominator, name in comparisons:
            if numerator > denominator:
                raise ValueError(f"{name} exceeds its denominator")
        if self.sum_image_iou > self.images:
            raise ValueError("sum_image_iou exceeds image count")
        # These are not optional metric conventions: they are identities of
        # the fixed instance matcher.  Enforcing them prevents an internally
        # self-hashed ledger from independently inventing detection and
        # recovery counts that no prediction/GT pairing could produce.
        if self.total_gt != (
            self.total_anchor_misses + self.total_anchor_covered
        ):
            raise ValueError(
                "total_gt must equal anchor misses plus anchor covered"
            )
        if self.matched_gt != (
            self.recovered_anchor_misses
            + self.retained_anchor_covered
        ):
            raise ValueError(
                "matched_gt must equal recovered misses plus retained covered"
            )
        if (
            self.recovered_reachable_anchor_misses
            > self.recovered_anchor_misses
        ):
            raise ValueError(
                "reachable recovery exceeds all recovered anchor misses"
            )
        # Every persisted OOF row is one image.  Its nIoU contribution must
        # therefore be exactly determined by the same intersection/union
        # integers, including the fixed empty-union convention.
        if self.images == 1:
            expected_iou = (
                self.intersection / self.union
                if self.union
                else 1.0
            )
            if self.sum_image_iou != expected_iou:
                raise ValueError(
                    "single-image IoU sum differs from intersection/union"
                )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "FactualSufficientStatistics":
        payload = _mapping(value, name="sufficient_statistics")
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                "sufficient-statistic fields differ: "
                f"expected={sorted(expected)}, observed={sorted(payload)}"
            )
        return cls(**payload)  # type: ignore[arg-type]

    def plus(
        self,
        other: "FactualSufficientStatistics",
    ) -> "FactualSufficientStatistics":
        if type(other) is not FactualSufficientStatistics:
            raise TypeError("can pool only exact factual statistics")
        values = {
            name: getattr(self, name) + getattr(other, name)
            for name in _STAT_INTEGER_FIELDS
        }
        values["sum_image_iou"] = self.sum_image_iou + other.sum_image_iou
        return FactualSufficientStatistics(**values)

    def metrics(self) -> dict[str, object]:
        pixels = self.total_pixels
        total_misses = self.total_anchor_misses
        total_covered = self.total_anchor_covered
        total_reachable = self.total_reachable_anchor_misses
        pixel_fa = self.unmatched_pred_pixels / pixels
        raw_background_fa = self.raw_background_fp / pixels
        fp_components_per_mp = (
            self.unmatched_pred_components / (pixels / 1_000_000.0)
        )
        return {
            "images": self.images,
            "true_targets": self.matched_gt,
            "total_targets": self.total_gt,
            "pd": (
                self.matched_gt / self.total_gt
                if self.total_gt
                else 1.0
            ),
            "recovered_anchor_misses": (
                self.recovered_anchor_misses
            ),
            "retention": (
                self.retained_anchor_covered / total_covered
                if total_covered
                else 1.0
            ),
            "mIoU": self.intersection / self.union if self.union else 1.0,
            "nIoU": self.sum_image_iou / self.images,
            "pixel_fa": pixel_fa,
            "raw_background_fa": raw_background_fa,
            "fp_components_per_mp": fp_components_per_mp,
            "gross_rmr": (
                self.recovered_anchor_misses / total_misses
                if total_misses
                else 0.0
            ),
            "reachable_rmr": (
                self.recovered_reachable_anchor_misses / total_reachable
                if total_reachable
                else 0.0
            ),
            "budget_violation": not (
                pixel_fa <= PIXEL_FA_LIMIT
                and raw_background_fa <= RAW_BACKGROUND_FA_LIMIT
                and fp_components_per_mp <= FP_COMPONENTS_PER_MP_LIMIT
                and (
                    self.retained_anchor_covered / total_covered
                    if total_covered
                    else 1.0
                )
                == 1.0
            ),
            "sufficient_statistics": asdict(self),
        }


def _metric(value: Mapping[str, object], name: str) -> float:
    return _real(value.get(name), name=name)


def _count(value: Mapping[str, object], name: str) -> int:
    return _integer(value.get(name), name=name)


def safety_checks(metrics: Mapping[str, object]) -> dict[str, bool]:
    value = _mapping(metrics, name="metrics")
    budget_violation = value.get("budget_violation")
    if not isinstance(budget_violation, bool):
        raise TypeError("budget_violation must be bool")
    return {
        "retention_equal_1": _metric(value, "retention") == 1.0,
        "pixel_fa_le_limit": (
            _metric(value, "pixel_fa") <= PIXEL_FA_LIMIT
        ),
        "raw_background_fa_le_limit": (
            _metric(value, "raw_background_fa")
            <= RAW_BACKGROUND_FA_LIMIT
        ),
        "fp_components_per_mp_le_limit": (
            _metric(value, "fp_components_per_mp")
            <= FP_COMPONENTS_PER_MP_LIMIT
        ),
        "budget_violation_false": budget_violation is False,
    }


def select_base_b_train_fold_threshold(
    candidate_rows: Sequence[Mapping[str, object]],
) -> float:
    """Select BaseB only from a train-fold 51-point ledger."""

    by_threshold: dict[float, dict[str, object]] = {}
    for index, raw in enumerate(candidate_rows):
        row = _mapping(raw, name=f"candidate_rows[{index}]")
        if row.get("selection_split_role") != "OOF_train_fold":
            raise ValueError("BaseB threshold selection may use train fold only")
        if row.get("D_V_payload_accessed") is not False:
            raise PermissionError("BaseB OOF selection touched D_V")
        if row.get("D_T_payload_accessed") is not False:
            raise PermissionError("BaseB OOF selection touched D_T")
        threshold = _real(row.get("threshold"), name="threshold")
        if threshold in by_threshold:
            raise ValueError("duplicate BaseB threshold row")
        metrics = _mapping(row.get("metrics"), name="metrics")
        by_threshold[threshold] = metrics
    if tuple(sorted(by_threshold)) != BASE_B_THRESHOLD_GRID:
        raise ValueError("BaseB must evaluate the complete frozen 51-point grid")
    feasible = [
        (threshold, metrics)
        for threshold, metrics in by_threshold.items()
        if all(safety_checks(metrics).values())
    ]
    if not feasible:
        raise ValueError("no train-fold BaseB threshold satisfies safety")

    def selection_key(
        item: tuple[float, Mapping[str, object]],
    ) -> tuple[float, float, float, float, float, float]:
        threshold, metrics = item
        return (
            _metric(metrics, "pd"),
            _metric(metrics, "retention"),
            -_metric(metrics, "pixel_fa"),
            -_metric(metrics, "raw_background_fa"),
            -_metric(metrics, "fp_components_per_mp"),
            threshold,
        )

    return max(feasible, key=selection_key)[0]


# ---------------------------------------------------------------------------
# v24 fail-closed evidence objects
# ---------------------------------------------------------------------------
#
# The first implementation of this module returned bare SHA strings and let
# downstream decision helpers consume caller-created dictionaries/booleans.
# That was useful while drafting the protocol, but it is not a sufficient
# authorization boundary: a caller could accidentally validate one receipt
# and decide on a different payload, or could simply assert that D_V/D_T had
# not been read.  The definitions below intentionally supersede those draft
# entry points.  Every decision-relevant value is now carried by an immutable
# object that can only be issued by a verifier in this process.

_TOKEN_ISSUER = object()
_TOKEN_REGISTRY: dict[int, object] = {}


def _register_token(value: object) -> object:
    """Keep a strong exact-instance capability record for issued tokens."""

    if getattr(value, "_issuer", None) is not _TOKEN_ISSUER:
        raise AssertionError("protocol attempted to register an unsigned token")
    identity = id(value)
    previous = _TOKEN_REGISTRY.get(identity)
    if previous is not None and previous is not value:
        raise RuntimeError("protocol token identity collision")
    _TOKEN_REGISTRY[identity] = value
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: Iterable[str],
    *,
    name: str,
) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        missing = sorted(expected_set - observed)
        extra = sorted(observed - expected_set)
        raise ValueError(
            f"{name} fields changed; missing={missing}, extra={extra}"
        )


def _self_fingerprint(
    payload: Mapping[str, object],
    *,
    field_name: str,
    name: str,
) -> str:
    body = dict(payload)
    fingerprint = body.pop(field_name, None)
    if fingerprint != stable_fingerprint(body):
        raise ValueError(f"{name} {field_name} is invalid")
    return _sha256(fingerprint, name=f"{name}.{field_name}")


def _token(value: object, token_type: type, *, name: str) -> object:
    if (
        type(value) is not token_type
        or getattr(value, "_issuer", None) is not _TOKEN_ISSUER
        or _TOKEN_REGISTRY.get(id(value)) is not value
    ):
        raise TypeError(f"{name} must be issued by its protocol verifier")
    return value


def _strict_relative_path(value: object, *, name: str) -> Path:
    relative = Path(_text(value, name=name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be repository-relative")
    return relative


def _verify_regular_file(
    path_value: object,
    *,
    name: str,
    expected_sha256: object,
    expected_size: object | None = None,
    expected_device: object | None = None,
    expected_inode: object | None = None,
    require_single_link: bool = True,
) -> tuple[Path, os.stat_result]:
    path = Path(_text(path_value, name=f"{name}.path"))
    if not path.is_absolute():
        raise ValueError(f"{name}.path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} is not a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise RuntimeError(f"{name}.path is not canonical")
    stat_result = path.stat()
    if require_single_link and stat_result.st_nlink != 1:
        raise PermissionError(f"{name} has a hard-link alias")
    if expected_size is not None and _integer(
        expected_size,
        name=f"{name}.size_bytes",
    ) != stat_result.st_size:
        raise RuntimeError(f"{name} size changed")
    if expected_device is not None and _integer(
        expected_device,
        name=f"{name}.device",
    ) != stat_result.st_dev:
        raise RuntimeError(f"{name} device changed")
    if expected_inode is not None and _integer(
        expected_inode,
        name=f"{name}.inode",
    ) != stat_result.st_ino:
        raise RuntimeError(f"{name} inode changed")
    if _file_sha256(path) != _sha256(
        expected_sha256,
        name=f"{name}.file_sha256",
    ):
        raise RuntimeError(f"{name} bytes changed")
    return path, stat_result


def _verify_repo_source_hashes(
    source_hashes: object,
    *,
    repository_root: Path,
    name: str,
    exact_paths: Sequence[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    values = _mapping(source_hashes, name=name)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if exact_paths is not None and set(values) != set(exact_paths):
        raise ValueError(f"{name} source inventory changed")
    result: list[tuple[str, str]] = []
    for raw_path, raw_digest in values.items():
        relative = _strict_relative_path(raw_path, name=f"{name} path")
        path = repository_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
        ):
            raise RuntimeError(f"{name} source is not regular: {relative}")
        digest = _sha256(raw_digest, name=f"{name}[{raw_path}]")
        if _file_sha256(path) != digest:
            raise RuntimeError(f"{name} source bytes changed: {relative}")
        result.append((raw_path, digest))
    return tuple(result)


class _PayloadToken(Mapping[str, object]):
    """Read-only mapping view used for decision/evidence tokens."""

    payload_json: str

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.payload_json)
        return _mapping(value, name=type(self).__name__)

    def __getitem__(self, key: str) -> object:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)


@dataclass(frozen=True, slots=True)
class VerifiedAccessAudit(_PayloadToken):
    payload_json: str
    stage_id: str
    allowed_splits: tuple[str, ...]
    receipt_fingerprint: str
    _issuer: object

    @property
    def observed_payloads(self) -> tuple[dict[str, object], ...]:
        raw = self.payload["observed_payloads"]
        if not isinstance(raw, list):
            raise AssertionError("verified access payload changed")
        return tuple(_mapping(row, name="observed payload") for row in raw)


@dataclass(frozen=True, slots=True)
class VerifiedOOF4Split:
    receipt_fingerprint: str
    protocol_preregistration_fingerprint: str
    root_by_sample_items: tuple[tuple[str, str], ...]
    plan_json: str
    root_by_sample_fingerprint: str
    plan_fingerprint: str
    _issuer: object

    @property
    def root_by_sample(self) -> dict[str, str]:
        return dict(self.root_by_sample_items)

    @property
    def plan(self) -> dict[str, object]:
        return _mapping(json.loads(self.plan_json), name="verified OOF plan")


@dataclass(frozen=True, slots=True)
class VerifiedOOFFold:
    receipt_fingerprint: str
    split_receipt_fingerprint: str
    fold_id: int
    train_sample_roots: tuple[tuple[str, str], ...]
    holdout_sample_roots: tuple[tuple[str, str], ...]
    evaluation_artifact_fingerprints: tuple[tuple[str, str], ...]
    evaluation_ledger_fingerprints: tuple[tuple[str, str], ...]
    evaluation_rows_json: str
    factual_rows_json: str
    evaluation_dataset_fingerprint: str
    evaluator_fingerprint: str
    process_instance_fingerprint: str
    access_audit_receipt_fingerprint: str
    base_b_ledger_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedFactualPool(_PayloadToken):
    payload_json: str
    evidence_fingerprint: str
    fold_receipt_fingerprint: str
    split_receipt_fingerprint: str
    fold_id: int
    access_audit_receipt_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedOOFPooledEvidence(_PayloadToken):
    payload_json: str
    evidence_fingerprint: str
    split_receipt_fingerprint: str
    fold_receipt_fingerprints: tuple[str, ...]
    access_audit_receipt_fingerprints: tuple[str, ...]
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedGatePathEvidence(_PayloadToken):
    payload_json: str
    receipt_fingerprint: str
    pooled_evidence_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedOOFDecision(_PayloadToken):
    payload_json: str
    decision_fingerprint: str
    pooled_evidence_fingerprint: str
    gate_path_receipt_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedBoundedEvidence(_PayloadToken):
    payload_json: str
    receipt_fingerprint: str
    oof_decision_fingerprint: str
    access_audit_receipt_fingerprint: str
    full_d_r_semantic_cache_fingerprint: str
    full_d_r_neutral_payload_fingerprint: str
    full_d_r_materialization_receipt_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedBoundedDecision(_PayloadToken):
    payload_json: str
    decision_fingerprint: str
    bounded_receipt_fingerprint: str
    oof_decision_fingerprint: str
    full_d_r_semantic_cache_fingerprint: str
    full_d_r_neutral_payload_fingerprint: str
    full_d_r_materialization_receipt_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedFormalTraining(_PayloadToken):
    payload_json: str
    receipt_fingerprint: str
    training_receipt_fingerprint: str
    seed: int
    role: str
    semantic_cache_fingerprint: str
    cache_artifact_path: str
    cache_artifact_sha256: str
    cache_artifact_device: int
    cache_artifact_inode: int
    cache_artifact_receipt_fingerprint: str
    cache_neutral_payload_fingerprint: str
    schedule_fingerprint: str
    schedule_policy_without_seed_fingerprint: str
    final_model_fingerprint: str
    terminal_artifact_path: str
    terminal_artifact_sha256: str
    run_start_artifact_path: str
    run_start_artifact_sha256: str
    run_start_artifact_device: int
    run_start_artifact_inode: int
    run_start_marker_fingerprint: str
    process_instance_fingerprint: str
    unified_source_closure_fingerprint: str
    model_contract_fingerprint: str
    access_audit_receipt_fingerprint: str
    _cache_artifact_token: object
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedFormalTrainingPair(_PayloadToken):
    payload_json: str
    pair_fingerprint: str
    seed42_receipt_fingerprint: str
    seed43_receipt_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedDTPreregistration(_PayloadToken):
    payload_json: str
    preregistration_fingerprint: str
    protocol_preregistration_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedDTSeed42ModelBinding(_PayloadToken):
    payload_json: str
    receipt_fingerprint: str
    formal_receipt_fingerprint: str
    final_model_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedBaselineEnvelope(_PayloadToken):
    payload_json: str
    binding_fingerprint: str
    source_receipt_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedDVCandidateEvidence(_PayloadToken):
    payload_json: str
    receipt_fingerprint: str
    model_binding_receipt_fingerprint: str
    access_audit_receipt_fingerprint: str
    _issuer: object


def require_verified_access_audit(
    value: object,
) -> VerifiedAccessAudit:
    """Return only an access token issued by this module's verifier."""

    token = _token(value, VerifiedAccessAudit, name="access_audit")
    assert isinstance(token, VerifiedAccessAudit)
    return token


def require_verified_oof4_split(
    value: object,
) -> VerifiedOOF4Split:
    """Return only an OOF4 split issued by this module's verifier."""

    token = _token(value, VerifiedOOF4Split, name="verified_split")
    assert isinstance(token, VerifiedOOF4Split)
    return token


def require_verified_oof_decision(
    value: object,
) -> VerifiedOOFDecision:
    """Return only an OOF decision issued by this module's verifier."""

    token = _token(value, VerifiedOOFDecision, name="oof_decision")
    assert isinstance(token, VerifiedOOFDecision)
    return token


def require_verified_bounded_decision(
    value: object,
) -> VerifiedBoundedDecision:
    """Return only a bounded decision issued by this module's verifier."""

    token = _token(value, VerifiedBoundedDecision, name="bounded_decision")
    assert isinstance(token, VerifiedBoundedDecision)
    return token


def verify_access_audit_receipt(
    receipt: Mapping[str, object],
    *,
    expected_stage_id: str,
    allowed_splits: Sequence[str],
) -> VerifiedAccessAudit:
    """Issue a split-access token from an exact, self-fingerprinted ledger."""

    payload = _mapping(receipt, name="access audit receipt")
    _exact_keys(
        payload,
        {
            "schema_version",
            "stage_id",
            "allowed_splits",
            "observed_payloads",
            "source_manifest_fingerprint",
            "event_log_fingerprint",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "receipt_fingerprint",
        },
        name="access audit receipt",
    )
    expected_stage = _text(expected_stage_id, name="expected_stage_id")
    expected_allowed = tuple(
        _text(value, name="allowed split") for value in allowed_splits
    )
    if len(expected_allowed) != len(set(expected_allowed)):
        raise ValueError("allowed_splits contains duplicates")
    raw_allowed = payload.get("allowed_splits")
    if not isinstance(raw_allowed, list) or tuple(raw_allowed) != (
        expected_allowed
    ):
        raise PermissionError("access-audit split allowlist changed")
    if (
        payload.get("schema_version")
        != "cure-lite-v24-split-access-audit-v1"
        or payload.get("stage_id") != expected_stage
        or payload.get("D_V_payload_accessed")
        is not ("D_V" in expected_allowed)
        or payload.get("D_T_payload_accessed")
        is not ("D_T" in expected_allowed)
    ):
        raise PermissionError("access-audit stage/firewall changed")
    _sha256(
        payload.get("source_manifest_fingerprint"),
        name="source_manifest_fingerprint",
    )
    _sha256(
        payload.get("event_log_fingerprint"),
        name="event_log_fingerprint",
    )
    rows = payload.get("observed_payloads")
    if not isinstance(rows, list):
        raise TypeError("observed_payloads must be a list")
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, name=f"observed_payloads[{index}]")
        _exact_keys(
            row,
            {"split", "logical_id", "purpose", "source_fingerprint"},
            name=f"observed_payloads[{index}]",
        )
        split = _text(row.get("split"), name=f"observed[{index}].split")
        logical_id = _text(
            row.get("logical_id"),
            name=f"observed[{index}].logical_id",
        )
        _text(row.get("purpose"), name=f"observed[{index}].purpose")
        _sha256(
            row.get("source_fingerprint"),
            name=f"observed[{index}].source_fingerprint",
        )
        if split not in expected_allowed:
            raise PermissionError(f"observed forbidden split {split!r}")
        if (split, logical_id) in seen:
            raise ValueError("duplicate split/logical_id access event")
        seen.add((split, logical_id))
    fingerprint = _self_fingerprint(
        payload,
        field_name="receipt_fingerprint",
        name="access audit",
    )
    return _register_token(VerifiedAccessAudit(
        payload_json=canonical_json(payload),
        stage_id=expected_stage,
        allowed_splits=expected_allowed,
        receipt_fingerprint=fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


def _verified_main_preregistration(
    repository_root: Path,
) -> tuple[dict[str, object], str]:
    path = (
        repository_root
        / "protocols/IRSTD-1K/gcr_pacre_v24/preregistration.json"
    )
    payload = _strict_json(path)
    if (
        payload.get("schema_version")
        != "cure-lite-v24-gcr-pacre-evidence-preregistration-v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status")
        != "FROZEN_DESIGN_EXECUTION_NOT_AUTHORIZED"
    ):
        raise ValueError("main protocol preregistration identity changed")
    authorization = _mapping(
        payload.get("authorization"),
        name="main preregistration authorization",
    )
    if not authorization or any(value is not False for value in authorization.values()):
        raise PermissionError("main preregistration already authorizes execution")
    fingerprint = _self_fingerprint(
        payload,
        field_name="preregistration_fingerprint",
        name="main preregistration",
    )
    return payload, fingerprint


def validate_protocol_artifact_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path,
) -> str:
    """Verify the one-way byte/self-fingerprint closure of protocol files."""

    root = Path(repository_root).resolve(strict=True)
    manifest = _strict_json(Path(manifest_path).resolve(strict=True))
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_preregistration_fingerprint",
            "status",
            "artifacts",
            "OOF4_authorized",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "manifest_fingerprint",
        },
        name="protocol artifact manifest",
    )
    _, protocol_fp = _verified_main_preregistration(root)
    if (
        manifest.get("schema_version")
        != "cure-lite-v24-gcr-pacre-protocol-artifact-manifest-v1"
        or manifest.get("protocol_preregistration_fingerprint") != protocol_fp
        or manifest.get("status")
        != "FROZEN_METADATA_ONLY_EXECUTION_NOT_AUTHORIZED"
        or manifest.get("OOF4_authorized") is not False
        or manifest.get("D_V_payload_accessed") is not False
        or manifest.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("protocol artifact manifest identity changed")
    fingerprint = _self_fingerprint(
        manifest,
        field_name="manifest_fingerprint",
        name="protocol artifact manifest",
    )
    artifacts = _mapping(manifest.get("artifacts"), name="manifest artifacts")
    expected_names = (
        "preregistration.json",
        "D_R_OOF4_split_preregistration.json",
        "D_T_preregistration.json",
        "exact_baseline_ledger_binding.json",
        "protocol.schema.json",
        "D_T_seed42_model_binding.template.json",
    )
    if set(artifacts) != set(expected_names):
        raise ValueError("protocol artifact inventory changed")
    protocol_dir = root / "protocols/IRSTD-1K/gcr_pacre_v24"
    for filename, raw in artifacts.items():
        row = _mapping(raw, name=f"manifest artifacts[{filename}]")
        _exact_keys(
            row,
            {
                "file_sha256",
                "self_fingerprint_field",
                "self_fingerprint",
            },
            name=f"manifest artifacts[{filename}]",
        )
        path = protocol_dir / filename
        expected_sha = _sha256(
            row.get("file_sha256"),
            name=f"{filename}.file_sha256",
        )
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
            or _file_sha256(path) != expected_sha
        ):
            raise RuntimeError(f"protocol artifact bytes changed: {filename}")
        self_field = row.get("self_fingerprint_field")
        self_fp = row.get("self_fingerprint")
        if self_field is None:
            if self_fp is not None:
                raise ValueError(f"{filename} has an invalid null self-binding")
            continue
        field_name = _text(
            self_field,
            name=f"{filename}.self_fingerprint_field",
        )
        file_payload = _strict_json(path)
        observed_self = _self_fingerprint(
            file_payload,
            field_name=field_name,
            name=filename,
        )
        if observed_self != _sha256(
            self_fp,
            name=f"{filename}.self_fingerprint",
        ):
            raise ValueError(f"{filename} self-fingerprint anchor changed")
    return fingerprint


def verify_oof4_split_preregistration(
    preregistration: Mapping[str, object] | str | Path,
    *,
    repository_root: str | Path,
) -> VerifiedOOF4Split:
    """Verify sources, exact sample→root mapping, full plan, and compact receipt."""

    root = Path(repository_root).resolve(strict=True)
    if isinstance(preregistration, (str, Path)):
        payload = _strict_json(Path(preregistration).resolve(strict=True))
    else:
        payload = _mapping(
            preregistration,
            name="OOF4 split preregistration",
        )
    _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "protocol_preregistration_fingerprint",
            "status",
            "source_bindings",
            "root_source_derivation",
            "plan",
            "root_counts_differ_by_at_most_one",
            "each_root_held_out_exactly_once",
            "derived_role_closure",
            "OOF4_authorized",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "receipt_fingerprint",
        },
        name="OOF4 split preregistration",
    )
    main, protocol_fingerprint = _verified_main_preregistration(root)
    if payload.get("protocol_preregistration_fingerprint") != (
        protocol_fingerprint
    ):
        raise ValueError("OOF split is not anchored to the main preregistration")
    fingerprint = _self_fingerprint(
        payload,
        field_name="receipt_fingerprint",
        name="OOF4 split preregistration",
    )
    if (
        payload.get("schema_version")
        != "cure-lite-v24-gcr-pacre-D_R-OOF4-split-preregistration-v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status")
        != "SPLIT_FROZEN_EXECUTION_NOT_AUTHORIZED"
        or payload.get("OOF4_authorized") is not False
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("OOF split authorization identity changed")
    bindings = _mapping(payload.get("source_bindings"), name="source_bindings")
    _exact_keys(
        bindings,
        {
            "manifest_repo_path",
            "manifest_file_sha256",
            "state_index_repo_path",
            "state_index_file_sha256",
        },
        name="source_bindings",
    )
    main_sources = _mapping(
        main.get("frozen_D_R_sources"),
        name="main frozen_D_R_sources",
    )

    def bound_json(
        path_key: str,
        digest_key: str,
        main_key: str,
    ) -> dict[str, object]:
        relative = _strict_relative_path(
            bindings.get(path_key),
            name=f"source_bindings.{path_key}",
        )
        main_binding = _mapping(
            main_sources.get(main_key),
            name=f"main frozen source {main_key}",
        )
        if (
            main_binding.get("repo_path") != str(relative)
            or main_binding.get("file_sha256") != bindings.get(digest_key)
        ):
            raise ValueError("split/main frozen source anchors differ")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
            or _file_sha256(path)
            != _sha256(
                bindings.get(digest_key),
                name=f"source_bindings.{digest_key}",
            )
        ):
            raise RuntimeError(f"frozen D_R source changed: {path_key}")
        return _strict_json(path)

    manifest = bound_json(
        "manifest_repo_path",
        "manifest_file_sha256",
        "manifest",
    )
    state_index = bound_json(
        "state_index_repo_path",
        "state_index_file_sha256",
        "state_index",
    )
    manifest_rows = manifest.get("samples")
    state_rows = state_index.get("records")
    if not isinstance(manifest_rows, list) or not isinstance(state_rows, list):
        raise TypeError("frozen D_R metadata rows must be lists")
    root_by_sample = derive_root_source_ids(manifest_rows)
    indexed_samples: list[str] = []
    for index, raw in enumerate(state_rows):
        row = _mapping(raw, name=f"state index records[{index}]")
        indexed_samples.append(
            _text(row.get("sample_id"), name=f"state record {index}.sample_id")
        )
    if (
        len(indexed_samples) != len(set(indexed_samples))
        or set(root_by_sample) != set(indexed_samples)
        or state_index.get("dataset") != "IRSTD-1K"
        or state_index.get("split") != "D_R"
        or state_index.get("sample_count") != len(indexed_samples)
    ):
        raise RuntimeError("manifest/state-index sample closure changed")
    plan = deterministic_oof4_plan(root_by_sample)
    derivation = _mapping(
        payload.get("root_source_derivation"),
        name="root_source_derivation",
    )
    _exact_keys(
        derivation,
        {
            "algorithm",
            "grouping_fields",
            "sample_count",
            "root_source_count",
            "root_by_sample_fingerprint",
        },
        name="root_source_derivation",
    )
    if (
        derivation.get("algorithm")
        != "transitive_connected_components_over_same-field_grouping_keys_v1"
        or derivation.get("grouping_fields") != list(ROOT_GROUP_FIELDS)
        or derivation.get("sample_count") != len(root_by_sample)
        or derivation.get("root_source_count")
        != len(set(root_by_sample.values()))
        or derivation.get("root_by_sample_fingerprint")
        != stable_fingerprint(dict(sorted(root_by_sample.items())))
    ):
        raise ValueError("frozen sample→root mapping changed")
    compact = _mapping(payload.get("plan"), name="compact OOF4 plan")
    expected_summaries = [
        {
            "fold_id": fold["fold_id"],
            "held_out_root_source_count": len(
                fold["held_out_root_source_ids"]
            ),
            "held_out_sample_count": len(fold["held_out_sample_ids"]),
            "held_out_root_source_ids_fingerprint": stable_fingerprint(
                fold["held_out_root_source_ids"]
            ),
            "held_out_sample_ids_fingerprint": stable_fingerprint(
                fold["held_out_sample_ids"]
            ),
        }
        for fold in plan["folds"]  # type: ignore[index]
    ]
    _exact_keys(
        compact,
        {
            "schema_version",
            "namespace",
            "seed",
            "fold_count",
            "assignment_policy",
            "plan_fingerprint",
            "fold_summaries",
        },
        name="compact OOF4 plan",
    )
    if compact != {
        "schema_version": plan["schema_version"],
        "namespace": plan["namespace"],
        "seed": plan["seed"],
        "fold_count": plan["fold_count"],
        "assignment_policy": plan["assignment_policy"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "fold_summaries": expected_summaries,
    }:
        raise ValueError("compact OOF4 plan differs from full recomputation")
    role_closure = _mapping(
        payload.get("derived_role_closure"),
        name="derived_role_closure",
    )
    if (
        role_closure.get("status")
        != "PENDING_D_R_AUTHORIZED_MATERIALIZATION"
        or role_closure.get("per_fold_role_counts") is not None
        or role_closure.get("four_fold_role_union_intersection_proof")
        is not None
        or role_closure.get("materialization_authorized") is not False
        or role_closure.get(
            "fabricated_or_evenly_apportioned_counts_forbidden"
        )
        is not True
        or payload.get("root_counts_differ_by_at_most_one") is not True
        or payload.get("each_root_held_out_exactly_once") is not True
    ):
        raise PermissionError("unmaterialized role-closure contract changed")
    return _register_token(VerifiedOOF4Split(
        receipt_fingerprint=fingerprint,
        protocol_preregistration_fingerprint=protocol_fingerprint,
        root_by_sample_items=tuple(sorted(root_by_sample.items())),
        plan_json=canonical_json(plan),
        root_by_sample_fingerprint=stable_fingerprint(
            dict(sorted(root_by_sample.items()))
        ),
        plan_fingerprint=_sha256(
            plan["plan_fingerprint"],
            name="plan_fingerprint",
        ),
        _issuer=_TOKEN_ISSUER,
    ))


_BASE_B_SELECTOR_POLICY = (
    "maximize_pd",
    "maximize_retention",
    "minimize_pixel_fa",
    "minimize_raw_background_fa",
    "minimize_fp_components_per_mp",
    "maximize_threshold",
)
_TRAIN_CACHE_READERS = {
    "base_eval": ("BaseB_train_fold_selector",),
    "PACRE_VC_v23_control": ("PACRE_VC_v23_control_train_runner",),
    "GCR_PACRE_v24": ("GCR_PACRE_v24_train_runner",),
}
_HOLDOUT_CACHE_READERS = {
    arm: ("OOF4_read_only_holdout_evaluator",)
    for arm in ("base_eval", "PACRE_VC_v23_control", "GCR_PACRE_v24")
}
_OOF_TRAINING_ARM_FIELDS = {
    "seed",
    "epochs",
    "steps_per_epoch",
    "completed_updates",
    "training_invocations",
    "from_scratch",
    "resume_allowed",
    "automatic_retry_allowed",
    "checkpoint_policy",
    "optimizer_state_initial_empty",
    "train_root_source_ids",
    "train_sample_ids",
    "schedule_fingerprint",
    "batch_sequence_fingerprint",
    "training_population_fingerprint",
    "initial_shared_parameter_fingerprint",
    "initial_parameters",
    "completed_400_capability",
    "completed_400_capability_fingerprint",
    "run_start_marker_fingerprint",
    "PMOPE_fingerprint",
    "Adam_policy_fingerprint",
    "dtype_device_policy_fingerprint",
    "source_hashes",
    "module_instance_id",
    "optimizer_instance_id",
    "parameter_storage_ledger",
    "parameter_storage_ledger_fingerprint",
    "initial_model_fingerprint",
    "final_model_fingerprint",
    "terminal_artifact_fingerprint",
    "terminal_artifact",
}


def _validate_artifact(
    artifact: object,
    *,
    name: str,
    expected_model_fingerprint: str | None = None,
) -> tuple[str, str, os.stat_result]:
    value = _mapping(artifact, name=name)
    _exact_keys(
        value,
        {"path", "size_bytes", "file_sha256", "model_fingerprint"},
        name=name,
    )
    model_fingerprint = _sha256(
        value.get("model_fingerprint"),
        name=f"{name}.model_fingerprint",
    )
    if (
        expected_model_fingerprint is not None
        and model_fingerprint != expected_model_fingerprint
    ):
        raise ValueError(f"{name} model fingerprint changed")
    path, stat_result = _verify_regular_file(
        value.get("path"),
        name=name,
        expected_sha256=value.get("file_sha256"),
        expected_size=value.get("size_bytes"),
    )
    return str(path), model_fingerprint, stat_result


def _validate_base_b_train_fold_ledger(
    selection: object,
    *,
    train_sample_roots: Mapping[str, str],
    holdout_sample_roots: Mapping[str, str],
    base_train_cache_sha256: str,
    access_audit: VerifiedAccessAudit,
    repository_root: Path,
    protocol_preregistration_fingerprint: str,
) -> tuple[str, float]:
    value = _mapping(selection, name="BaseB_train_fold_selection")
    _exact_keys(
        value,
        {
            "selection_root_source_ids",
            "selection_sample_ids",
            "evaluation_root_source_ids",
            "evaluation_sample_ids",
            "holdout_labels_used_for_selection",
            "complete_51_point_grid_evaluated",
            "D_V_threshold_reused",
            "grid_source_repo_path",
            "grid_source_file_sha256",
            "threshold_grid",
            "candidate_rows",
            "candidate_ledger_fingerprint",
            "selector_policy",
            "selector_policy_fingerprint",
            "selected_threshold",
            "input_train_cache_fingerprint",
            "access_audit_receipt_fingerprint",
        },
        name="BaseB_train_fold_selection",
    )
    train_samples = sorted(train_sample_roots)
    holdout_samples = sorted(holdout_sample_roots)
    train_roots = sorted(set(train_sample_roots.values()))
    holdout_roots = sorted(set(holdout_sample_roots.values()))
    if (
        value.get("selection_sample_ids") != train_samples
        or value.get("selection_root_source_ids") != train_roots
        or value.get("evaluation_sample_ids") != holdout_samples
        or value.get("evaluation_root_source_ids") != holdout_roots
        or value.get("holdout_labels_used_for_selection") is not False
        or value.get("complete_51_point_grid_evaluated") is not True
        or value.get("D_V_threshold_reused") is not False
        or value.get("input_train_cache_fingerprint")
        != base_train_cache_sha256
        or value.get("access_audit_receipt_fingerprint")
        != access_audit.receipt_fingerprint
    ):
        raise PermissionError("BaseB train-only population/cache binding changed")
    main, main_fingerprint = _verified_main_preregistration(repository_root)
    if main_fingerprint != protocol_preregistration_fingerprint:
        raise ValueError("fold/main preregistration token mismatch")
    base_contract = _mapping(
        _mapping(main.get("OOF4"), name="main OOF4").get(
            "BaseB_train_fold_selected"
        ),
        name="main BaseB_train_fold_selected",
    )
    relative = _strict_relative_path(
        value.get("grid_source_repo_path"),
        name="grid_source_repo_path",
    )
    if (
        base_contract.get("threshold_grid_source_repo_path") != str(relative)
        or base_contract.get("threshold_grid_source_file_sha256")
        != value.get("grid_source_file_sha256")
    ):
        raise ValueError("BaseB grid source is not preregistered")
    source = repository_root / relative
    if (
        not source.is_file()
        or source.is_symlink()
        or source.resolve(strict=True) != source
        or _file_sha256(source)
        != _sha256(
            value.get("grid_source_file_sha256"),
            name="grid_source_file_sha256",
        )
    ):
        raise RuntimeError("BaseB frozen grid source changed")
    source_payload = _strict_json(source)
    if (
        source_payload.get("base_thresholds") != list(BASE_B_THRESHOLD_GRID)
        or source_payload.get("anchor_thresholds")
        != list(BASE_B_THRESHOLD_GRID)
    ):
        raise ValueError("BaseB source no longer contains the exact 51 grid")
    if value.get("threshold_grid") != list(BASE_B_THRESHOLD_GRID):
        raise ValueError("BaseB receipt threshold grid changed")
    policy = value.get("selector_policy")
    if policy != list(_BASE_B_SELECTOR_POLICY):
        raise ValueError("BaseB selector policy changed")
    if value.get("selector_policy_fingerprint") != stable_fingerprint(policy):
        raise ValueError("BaseB selector policy fingerprint changed")
    rows = value.get("candidate_rows")
    if not isinstance(rows, list) or len(rows) != len(BASE_B_THRESHOLD_GRID):
        raise ValueError("BaseB ledger must contain exactly 51 rows")
    expected_row_keys = {
        "threshold",
        "selection_split_role",
        "train_sample_ids",
        "train_root_source_ids",
        "input_train_cache_fingerprint",
        "access_audit_receipt_fingerprint",
        "metrics",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
    }
    normalized_rows: list[dict[str, object]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, name=f"BaseB candidate_rows[{index}]")
        _exact_keys(row, expected_row_keys, name=f"BaseB row {index}")
        expected_threshold = BASE_B_THRESHOLD_GRID[index]
        if (
            _real(row.get("threshold"), name=f"row[{index}].threshold")
            != expected_threshold
            or row.get("selection_split_role") != "OOF_train_fold"
            or row.get("train_sample_ids") != train_samples
            or row.get("train_root_source_ids") != train_roots
            or set(row.get("train_sample_ids", [])) & set(holdout_samples)
            or set(row.get("train_root_source_ids", [])) & set(holdout_roots)
            or row.get("input_train_cache_fingerprint")
            != base_train_cache_sha256
            or row.get("access_audit_receipt_fingerprint")
            != access_audit.receipt_fingerprint
            or row.get("D_V_payload_accessed") is not False
            or row.get("D_T_payload_accessed") is not False
        ):
            raise PermissionError(f"BaseB row {index} is not train-only")
        metrics = _mapping(row.get("metrics"), name=f"BaseB row {index} metrics")
        _exact_keys(
            metrics,
            {
                "pd",
                "retention",
                "pixel_fa",
                "raw_background_fa",
                "fp_components_per_mp",
                "budget_violation",
            },
            name=f"BaseB row {index} metrics",
        )
        # Parse every field now so NaN/Inf and bool-as-number cannot survive.
        _metric(metrics, "pd")
        safety_checks(metrics)
        normalized_rows.append(row)
    ledger_fingerprint = stable_fingerprint(normalized_rows)
    if value.get("candidate_ledger_fingerprint") != ledger_fingerprint:
        raise ValueError("BaseB candidate ledger fingerprint changed")
    selected = select_base_b_train_fold_threshold(normalized_rows)
    if _real(value.get("selected_threshold"), name="selected_threshold") != (
        selected
    ):
        raise ValueError("BaseB selected threshold is not selector-derived")
    return ledger_fingerprint, selected


_OOF_CACHE_ARTIFACT_SCHEMA = (
    "cure-lite-v24-gcr-pacre-oof4-cache-artifact-v1"
)
_OOF_CACHE_SET_SCHEMA = (
    "cure-lite-v24-gcr-pacre-oof4-six-cache-independence-v1"
)
_OOF_RUN_START_SCHEMA = (
    "cure-lite-v24-gcr-pacre-oof4-fold-persistent-run-start-v1"
)
_OOF_TERMINAL_ARTIFACT_SCHEMA = (
    "cure-lite-v24-gcr-pacre-oof4-terminal-safetensors-v1"
)
_OOF_TERMINAL_SEAL_SCHEMA = (
    "cure-lite-v24-gcr-pacre-oof4-training-terminal-seal-v1"
)
_OOF_COMPLETED_CAPABILITY_SCHEMA = (
    "cure-lite-v24-gcr-pacre-oof4-completed-400-capability-v1"
)
_OOF_EVALUATION_LEDGER_SCHEMA = (
    "cure-lite-v24-gcr-pacre-oof4-evaluation-ledger-v1"
)
_OOF_SOURCE_CLOSURE_SCHEMA = (
    "cure-lite-v24-gcr-pacre-unified-source-closure-v1"
)
_OOF_PARAMETER_CONTRACT = (
    ("joint_state_weight", (32, 80, 5, 5)),
    ("joint_hidden_bias", (32,)),
    ("scalar_energy_weight", (32,)),
)


def _verify_oof_json_artifact(
    artifact: object,
    *,
    name: str,
    fingerprint_field: str,
) -> tuple[dict[str, object], Path, os.stat_result]:
    value = _mapping(artifact, name=name)
    _exact_keys(
        value,
        {
            "path",
            "size_bytes",
            "file_sha256",
            "device",
            "inode",
            "hardlink_count",
            fingerprint_field,
            "payload",
        },
        name=name,
    )
    if value.get("hardlink_count") != 1:
        raise PermissionError(f"{name} must have exactly one hard link")
    path, stat_result = _verify_regular_file(
        value.get("path"),
        name=name,
        expected_sha256=value.get("file_sha256"),
        expected_size=value.get("size_bytes"),
        expected_device=value.get("device"),
        expected_inode=value.get("inode"),
    )
    if (
        stat_result.st_nlink != 1
        or stat_result.st_mode & 0o777 != 0o444
    ):
        raise PermissionError(f"{name} is not immutable mode-0444 evidence")
    payload = _mapping(value.get("payload"), name=f"{name}.payload")
    if (
        _strict_json(path) != payload
        or path.read_bytes()
        != (canonical_json(payload) + "\n").encode("utf-8")
        or value.get(fingerprint_field)
        != payload.get(fingerprint_field)
    ):
        raise RuntimeError(f"{name} canonical artifact binding changed")
    return payload, path, stat_result


def _validate_oof_initial_parameter_ledger(
    value: object,
    *,
    name: str,
) -> tuple[list[dict[str, object]], str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty parameter ledger")
    rows: list[dict[str, object]] = []
    names: set[str] = set()
    for index, raw in enumerate(value):
        row = _mapping(raw, name=f"{name}[{index}]")
        _exact_keys(
            row,
            {
                "name",
                "shape",
                "dtype",
                "numel",
                "byte_count",
                "content_fingerprint",
            },
            name=f"{name}[{index}]",
        )
        parameter_name = _text(
            row.get("name"),
            name=f"{name}[{index}].name",
        )
        shape = row.get("shape")
        if (
            parameter_name in names
            or not isinstance(shape, list)
            or not shape
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 1
                for item in shape
            )
            or row.get("dtype") != "torch.float32"
        ):
            raise ValueError(f"{name}[{index}] identity changed")
        names.add(parameter_name)
        numel = 1
        for dimension in shape:
            numel *= dimension
        if (
            row.get("numel") != numel
            or row.get("byte_count") != 4 * numel
        ):
            raise ValueError(f"{name}[{index}] byte geometry changed")
        _sha256(
            row.get("content_fingerprint"),
            name=f"{name}[{index}].content_fingerprint",
        )
        rows.append(row)
    observed_contract = tuple(
        (str(row["name"]), tuple(row["shape"])) for row in rows
    )
    if observed_contract != _OOF_PARAMETER_CONTRACT:
        raise ValueError(f"{name} is not the frozen 64/4/32 topology")
    if sum(int(row["numel"]) for row in rows) != 64_064:
        raise ValueError(f"{name} parameter count is not 64064")
    return rows, stable_fingerprint(rows)


def _validate_oof_parameter_storage_ledger(
    value: object,
    *,
    initial_parameters: Sequence[Mapping[str, object]],
    name: str,
) -> tuple[list[dict[str, object]], str, set[str]]:
    if not isinstance(value, list) or len(value) != len(initial_parameters):
        raise ValueError(f"{name} length changed")
    rows: list[dict[str, object]] = []
    identities: set[str] = set()
    expected_names = [str(row["name"]) for row in initial_parameters]
    for index, raw in enumerate(value):
        row = _mapping(raw, name=f"{name}[{index}]")
        _exact_keys(
            row,
            {
                "name",
                "device",
                "nbytes",
                "storage_identity_fingerprint",
            },
            name=f"{name}[{index}]",
        )
        identity = _sha256(
            row.get("storage_identity_fingerprint"),
            name=f"{name}[{index}].storage_identity_fingerprint",
        )
        if (
            row.get("name") != expected_names[index]
            or not isinstance(row.get("device"), str)
            or not str(row["device"])
            or row.get("nbytes")
            != initial_parameters[index].get("byte_count")
            or identity in identities
        ):
            raise ValueError(f"{name}[{index}] storage identity changed")
        identities.add(identity)
        rows.append(row)
    return rows, stable_fingerprint(rows), identities


def _validate_oof_terminal_artifact(
    artifact: object,
    *,
    fold_id: int,
    arm: str,
    final_model_fingerprint: str,
    run_start_marker_fingerprint: str,
    runtime_root: str | Path,
) -> tuple[str, os.stat_result]:
    value = _mapping(artifact, name=f"{arm}.terminal_artifact")
    _exact_keys(
        value,
        {
            "schema_version",
            "fold_id",
            "arm",
            "seed",
            "epochs",
            "steps_per_epoch",
            "completed_updates",
            "path",
            "size_bytes",
            "file_sha256",
            "device",
            "inode",
            "hardlink_count",
            "state_keys",
            "state_shapes",
            "state_dtypes",
            "parameter_count",
            "model_fingerprint",
            "training_result_fingerprint",
            "run_start_marker_fingerprint",
            "serialization",
            "final_checkpoint_only",
            "optimizer_state_saved",
            "intermediate_checkpoint_saved",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "terminal_artifact_fingerprint",
        },
        name=f"{arm}.terminal_artifact",
    )
    artifact_fp = _self_fingerprint(
        value,
        field_name="terminal_artifact_fingerprint",
        name=f"{arm}.terminal_artifact",
    )
    state_keys = value.get("state_keys")
    state_shapes = _mapping(
        value.get("state_shapes"),
        name=f"{arm}.terminal_artifact.state_shapes",
    )
    state_dtypes = _mapping(
        value.get("state_dtypes"),
        name=f"{arm}.terminal_artifact.state_dtypes",
    )
    if (
        value.get("schema_version") != _OOF_TERMINAL_ARTIFACT_SCHEMA
        or value.get("fold_id") != fold_id
        or value.get("arm") != arm
        or value.get("seed") != OOF_SEED
        or value.get("epochs") != BOUNDED_EPOCHS
        or value.get("steps_per_epoch") != BOUNDED_STEPS_PER_EPOCH
        or value.get("completed_updates") != BOUNDED_UPDATES
        or value.get("hardlink_count") != 1
        or value.get("model_fingerprint") != final_model_fingerprint
        or value.get("run_start_marker_fingerprint")
        != run_start_marker_fingerprint
        or value.get("serialization") != "safetensors"
        or value.get("final_checkpoint_only") is not True
        or value.get("optimizer_state_saved") is not False
        or value.get("intermediate_checkpoint_saved") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or not isinstance(state_keys, list)
        or not state_keys
        or len(state_keys) != len(set(state_keys))
        or set(state_shapes) != set(state_keys)
        or set(state_dtypes) != set(state_keys)
        or any(value != "torch.float32" for value in state_dtypes.values())
    ):
        raise PermissionError(f"{arm} terminal artifact policy changed")
    parameter_count = 0
    for key in state_keys:
        shape = state_shapes.get(key)
        if not isinstance(shape, list) or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 1
            for item in shape
        ):
            raise ValueError(f"{arm} terminal state shape changed")
        count = 1
        for dimension in shape:
            count *= dimension
        parameter_count += count
    if value.get("parameter_count") != parameter_count:
        raise ValueError(f"{arm} terminal parameter count changed")
    _sha256(
        value.get("training_result_fingerprint"),
        name=f"{arm}.training_result_fingerprint",
    )
    terminal_name = {
        "PACRE_VC_v23_control": "v23_control_terminal.safetensors",
        "GCR_PACRE_v24": "candidate_terminal.safetensors",
    }.get(arm)
    if terminal_name is None:
        raise ValueError(f"unknown OOF terminal arm {arm!r}")
    expected_path = (
        Path(runtime_root)
        / f"fold_{fold_id}"
        / "terminal"
        / terminal_name
    )
    if value.get("path") != str(expected_path):
        raise PermissionError(f"{arm} terminal path is not the fixed slot")
    _, stat_result = _verify_regular_file(
        value.get("path"),
        name=f"{arm}.terminal_artifact",
        expected_sha256=value.get("file_sha256"),
        expected_size=value.get("size_bytes"),
        expected_device=value.get("device"),
        expected_inode=value.get("inode"),
    )
    if stat_result.st_nlink != 1:
        raise PermissionError(f"{arm} terminal artifact has a hard-link alias")
    if stat_result.st_mode & 0o777 != 0o444:
        raise PermissionError(f"{arm} terminal artifact must be mode 0444")
    from cure_lite_v24.oof_training import (
        load_oof_terminal_model_strict,
    )

    load_oof_terminal_model_strict(
        value,
        arm=arm,
        expected_path=expected_path,
    )
    return artifact_fp, stat_result


def _validate_oof_evaluation_ledger_artifact(
    artifact: object,
    *,
    fold_id: int,
    arm: str,
    held_out_sample_roots: Mapping[str, str],
    expected_model_fingerprint: str | None,
    expected_operating_point: float,
    expected_path: Path,
) -> tuple[str, dict[str, object], os.stat_result]:
    value = _mapping(artifact, name=f"evaluation ledger {arm}")
    _exact_keys(
        value,
        {
            "path",
            "size_bytes",
            "file_sha256",
            "device",
            "inode",
            "hardlink_count",
            "ledger_fingerprint",
        },
        name=f"evaluation ledger {arm}",
    )
    if (
        value.get("hardlink_count") != 1
        or value.get("path") != str(expected_path)
    ):
        raise PermissionError("evaluation ledger has a hard-link alias")
    path, stat_result = _verify_regular_file(
        value.get("path"),
        name=f"evaluation ledger {arm}",
        expected_sha256=value.get("file_sha256"),
        expected_size=value.get("size_bytes"),
        expected_device=value.get("device"),
        expected_inode=value.get("inode"),
    )
    if stat_result.st_mode & 0o777 != 0o444:
        raise PermissionError("evaluation ledger must be immutable mode 0444")
    payload = _strict_json(path)
    if path.read_bytes() != (canonical_json(payload) + "\n").encode("utf-8"):
        raise ValueError("evaluation ledger JSON is not canonical")
    _exact_keys(
        payload,
        {
            "schema_version",
            "fold_id",
            "partition",
            "arm",
            "operating_point",
            "dataset_fingerprint",
            "model_fingerprint",
            "per_sample_rows",
            "pooled_statistics",
            "field_ledger_fingerprint",
            "prediction_ledger_fingerprint",
            "role_ledger_fingerprint",
            "evaluator_fingerprint",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "ledger_fingerprint",
        },
        name=f"evaluation ledger payload {arm}",
    )
    ledger_fp = _self_fingerprint(
        payload,
        field_name="ledger_fingerprint",
        name=f"evaluation ledger payload {arm}",
    )
    rows = payload.get("per_sample_rows")
    expected_sample_ids = list(held_out_sample_roots)
    if (
        value.get("ledger_fingerprint") != ledger_fp
        or payload.get("schema_version") != _OOF_EVALUATION_LEDGER_SCHEMA
        or payload.get("fold_id") != fold_id
        or payload.get("partition") != "holdout"
        or payload.get("arm") != arm
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
        or not isinstance(rows, list)
        or [row.get("sample_id") for row in rows if isinstance(row, Mapping)]
        != expected_sample_ids
        or payload.get("model_fingerprint")
        != expected_model_fingerprint
        or _real(
            payload.get("operating_point"),
            name=f"evaluation ledger {arm}.operating_point",
        )
        != expected_operating_point
    ):
        raise ValueError(f"evaluation ledger {arm} identity changed")
    normalized_rows = [
        _mapping(row, name=f"evaluation ledger {arm} row")
        for row in rows
    ]
    for row in normalized_rows:
        _exact_keys(
            row,
            {
                "sample_id",
                "root_source_id",
                "statistics",
                "field_fingerprint",
                "prediction_fingerprint",
                "role_statistics",
            },
            name=f"evaluation ledger {arm} row",
        )
        FactualSufficientStatistics.from_mapping(
            _mapping(row.get("statistics"), name="evaluation statistics")
        )
        if row.get("root_source_id") != held_out_sample_roots[
            str(row["sample_id"])
        ]:
            raise ValueError(
                f"evaluation ledger {arm} sample/root closure changed"
            )
        _sha256(row.get("field_fingerprint"), name="field_fingerprint")
        _sha256(
            row.get("prediction_fingerprint"),
            name="prediction_fingerprint",
        )
        _mapping(row.get("role_statistics"), name="role_statistics")
    if (
        payload.get("field_ledger_fingerprint")
        != stable_fingerprint(
            [row["field_fingerprint"] for row in normalized_rows]
        )
        or payload.get("prediction_ledger_fingerprint")
        != stable_fingerprint(
            [row["prediction_fingerprint"] for row in normalized_rows]
        )
        or payload.get("role_ledger_fingerprint")
        != stable_fingerprint(
            [row["role_statistics"] for row in normalized_rows]
        )
    ):
        raise ValueError(f"evaluation ledger {arm} internal seal changed")
    pooled = FactualSufficientStatistics.from_mapping(
        _mapping(
            payload.get("pooled_statistics"),
            name=f"evaluation ledger {arm} pooled statistics",
        )
    )
    aggregate: FactualSufficientStatistics | None = None
    for row in normalized_rows:
        item = FactualSufficientStatistics.from_mapping(
            _mapping(row["statistics"], name="evaluation row statistics")
        )
        aggregate = item if aggregate is None else aggregate.plus(item)
    if aggregate != pooled:
        raise ValueError(f"evaluation ledger {arm} pooled statistics changed")
    _sha256(payload.get("dataset_fingerprint"), name="dataset_fingerprint")
    _sha256(payload.get("evaluator_fingerprint"), name="evaluator_fingerprint")
    if expected_model_fingerprint is not None:
        _sha256(
            payload.get("model_fingerprint"),
            name=f"evaluation ledger {arm}.model_fingerprint",
        )
    return ledger_fp, payload, stat_result


def validate_oof_fold_execution_receipt(
    receipt: Mapping[str, object],
    verified_split: VerifiedOOF4Split,
    *,
    access_audit: VerifiedAccessAudit,
    execution_authorization: object,
    repository_root: str | Path,
) -> VerifiedOOFFold:
    """Validate one OOF fold against the verifier-issued frozen split."""

    split = _token(
        verified_split,
        VerifiedOOF4Split,
        name="verified_split",
    )
    access = _token(
        access_audit,
        VerifiedAccessAudit,
        name="access_audit",
    )
    assert isinstance(split, VerifiedOOF4Split)
    assert isinstance(access, VerifiedAccessAudit)
    from cure_lite_v24.oof_run_start import (
        require_verified_oof_execution_authorization,
    )

    authorization = require_verified_oof_execution_authorization(
        execution_authorization
    )
    if (
        authorization.split_receipt_fingerprint
        != split.receipt_fingerprint
    ):
        raise PermissionError("OOF authorization and frozen split differ")
    payload = _mapping(receipt, name="OOF fold receipt")
    _exact_keys(
        payload,
        {
            "schema_version",
            "split_preregistration_fingerprint",
            "root_by_sample_fingerprint",
            "plan_fingerprint",
            "fold_id",
            "train_root_source_ids",
            "held_out_root_source_ids",
            "train_sample_ids",
            "held_out_sample_ids",
            "access_audit_receipt_fingerprint",
            "events",
            "run_start_artifact",
            "terminal_seal",
            "cache_set_fingerprint",
            "cache_entries",
            "training_arms",
            "BaseB_train_fold_selection",
            "evaluation_artifact_fingerprints",
            "evaluation_ledger_artifacts",
            "factual_rows_artifact",
            "held_out_prediction_role",
            "source_closure",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "receipt_fingerprint",
        },
        name="OOF fold receipt",
    )
    if payload.get("schema_version") != "cure-lite-v24-oof-fold-execution-v3":
        raise ValueError("unexpected OOF fold schema")
    if (
        payload.get("split_preregistration_fingerprint")
        != split.receipt_fingerprint
        or payload.get("root_by_sample_fingerprint")
        != split.root_by_sample_fingerprint
        or payload.get("plan_fingerprint") != split.plan_fingerprint
    ):
        raise ValueError("fold is not bound to the verified frozen split")
    fold_id = _integer(payload.get("fold_id"), name="fold_id")
    folds = split.plan.get("folds")
    if not isinstance(folds, list):
        raise AssertionError("verified split lost folds")
    matches = [
        _mapping(row, name="verified plan fold")
        for row in folds
        if isinstance(row, Mapping) and row.get("fold_id") == fold_id
    ]
    if len(matches) != 1:
        raise ValueError("fold_id is absent from the frozen plan")
    planned = matches[0]
    root_by_sample = split.root_by_sample
    train_samples = list(planned["train_sample_ids"])  # type: ignore[arg-type]
    holdout_samples = list(
        planned["held_out_sample_ids"]  # type: ignore[arg-type]
    )
    train_roots = list(
        planned["train_root_source_ids"]  # type: ignore[arg-type]
    )
    holdout_roots = list(
        planned["held_out_root_source_ids"]  # type: ignore[arg-type]
    )
    if (
        payload.get("train_sample_ids") != train_samples
        or payload.get("held_out_sample_ids") != holdout_samples
        or payload.get("train_root_source_ids") != train_roots
        or payload.get("held_out_root_source_ids") != holdout_roots
    ):
        raise ValueError("fold sample/root allowlists differ from the full plan")
    train_sample_roots = {
        sample_id: root_by_sample[sample_id] for sample_id in train_samples
    }
    holdout_sample_roots = {
        sample_id: root_by_sample[sample_id] for sample_id in holdout_samples
    }
    if (
        set(train_sample_roots.values()) != set(train_roots)
        or set(holdout_sample_roots.values()) != set(holdout_roots)
        or set(train_samples) & set(holdout_samples)
        or set(train_roots) & set(holdout_roots)
    ):
        raise AssertionError("verified fold root closure failed")
    fold_closure_fingerprint = stable_fingerprint(
        {
            "schema_version": (
                "cure-lite-v24-gcr-pacre-oof4-fold-closure-v1"
            ),
            "fold_id": fold_id,
            "split_receipt_fingerprint": split.receipt_fingerprint,
            "plan_fingerprint": split.plan_fingerprint,
            "root_by_sample_fingerprint": split.root_by_sample_fingerprint,
            "train_root_source_ids": train_roots,
            "held_out_root_source_ids": holdout_roots,
            "train_sample_ids": train_samples,
            "held_out_sample_ids": holdout_samples,
            "root_by_sample": dict(sorted(root_by_sample.items())),
            "checks": {
                "root_sets_disjoint": True,
                "root_union_exact": True,
                "sample_sets_disjoint": True,
                "sample_union_exact": True,
                "sample_to_root_closure_exact": True,
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            },
        }
    )
    expected_stage = f"oof4_fold_{fold_id}"
    if (
        access.stage_id != expected_stage
        or access.allowed_splits != ("D_R",)
        or payload.get("access_audit_receipt_fingerprint")
        != access.receipt_fingerprint
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("fold split-access evidence changed")

    events = _mapping(payload.get("events"), name="OOF fold events")
    event_names = (
        "train_cache_materialized",
        "training_claimed",
        "training_terminals_sealed",
        "holdout_cache_materialized",
        "holdout_cache_first_open",
    )
    _exact_keys(events, event_names, name="OOF fold events")
    event_values = tuple(
        _integer(events.get(name), name=f"events.{name}")
        for name in event_names
    )
    if event_values != (1, 2, 3, 4, 5):
        raise PermissionError("OOF cache/training event ledger is not frozen")

    source_closure = _mapping(
        payload.get("source_closure"),
        name="OOF source_closure",
    )
    _exact_keys(
        source_closure,
        {
            "schema_version",
            "source_hashes",
            "source_closure_fingerprint",
        },
        name="OOF source_closure",
    )
    if source_closure.get("schema_version") != _OOF_SOURCE_CLOSURE_SCHEMA:
        raise ValueError("OOF source closure schema changed")
    from cure_lite_v24.source_closure import (
        GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
        gcr_pacre_v24_source_closure_fingerprint,
    )

    repository = Path(repository_root).resolve(strict=True)
    closure_rows = _verify_repo_source_hashes(
        source_closure.get("source_hashes"),
        repository_root=repository,
        name="OOF source_closure.source_hashes",
        exact_paths=GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
    )
    source_closure_fp = gcr_pacre_v24_source_closure_fingerprint(
        tuple(sorted(closure_rows))
    )
    if (
        source_closure.get("source_closure_fingerprint")
        != source_closure_fp
    ):
        raise ValueError("OOF source closure fingerprint changed")

    terminal_seal = _mapping(
        payload.get("terminal_seal"),
        name="OOF terminal_seal",
    )
    _exact_keys(
        terminal_seal,
        {
            "schema_version",
            "fold_id",
            "closure_fingerprint",
            "terminal_artifact_fingerprints",
            "completed_400_capability_fingerprints",
            "run_start_marker_fingerprint",
            "shared_initial_parameter_fingerprint",
            "initial_parameters",
            "schedule_fingerprint",
            "batch_sequence_fingerprint",
            "semantic_cache_fingerprint",
            "optimizer_config_fingerprint",
            "objective_policy_fingerprint",
            "event_index",
            "seal_fingerprint",
        },
        name="OOF terminal_seal",
    )
    terminal_seal_fp = _self_fingerprint(
        terminal_seal,
        field_name="seal_fingerprint",
        name="OOF terminal_seal",
    )
    if (
        terminal_seal.get("schema_version") != _OOF_TERMINAL_SEAL_SCHEMA
        or terminal_seal.get("fold_id") != fold_id
        or terminal_seal.get("closure_fingerprint")
        != fold_closure_fingerprint
        or terminal_seal.get("event_index") != 3
    ):
        raise PermissionError("OOF terminal seal identity changed")
    seal_initial_parameters, seal_initial_fp = (
        _validate_oof_initial_parameter_ledger(
            terminal_seal.get("initial_parameters"),
            name="OOF terminal_seal.initial_parameters",
        )
    )
    if (
        terminal_seal.get("shared_initial_parameter_fingerprint")
        != seal_initial_fp
    ):
        raise ValueError("OOF terminal seal initial ledger changed")

    raw_caches = payload.get("cache_entries")
    if not isinstance(raw_caches, list) or len(raw_caches) != 6:
        raise ValueError("OOF fold requires exactly six physical caches")
    cache_fields = {
        "cache_id",
        "artifact_fingerprint",
        "tensor_ledger_fingerprint",
        "partition",
        "arm",
        "closure_fingerprint",
        "terminal_seal_fingerprint",
        "semantic_payload_fingerprint",
        "root_source_ids",
        "sample_ids",
        "realpath",
        "device",
        "inode",
        "size_bytes",
        "file_sha256",
        "creation_phase",
        "creation_event",
        "reader_allowlist",
        "is_symlink",
        "hardlink_count",
        "fiemap_extent_flags",
        "is_reflink",
        "shared_tensor_storage",
        "mmap_reused",
        "process_cache_reused",
    }
    cache_by_key: dict[tuple[str, str], dict[str, object]] = {}
    physical: set[tuple[int, int]] = set()
    realpaths: set[str] = set()
    observed_access = {
        (
            str(row["logical_id"]),
            str(row["source_fingerprint"]),
            str(row["purpose"]),
        )
        for row in access.observed_payloads
        if row.get("split") == "D_R"
    }
    for index, raw in enumerate(raw_caches):
        cache = _mapping(raw, name=f"cache_entries[{index}]")
        _exact_keys(cache, cache_fields, name=f"cache_entries[{index}]")
        cache_id = _text(cache.get("cache_id"), name=f"cache[{index}].cache_id")
        partition = cache.get("partition")
        arm = cache.get("arm")
        if partition not in {"train", "holdout"} or arm not in (
            "base_eval",
            "PACRE_VC_v23_control",
            "GCR_PACRE_v24",
        ):
            raise ValueError(f"invalid cache identity {cache_id!r}")
        key = (str(partition), str(arm))
        if key in cache_by_key:
            raise ValueError("duplicate partition/arm cache")
        expected_samples = (
            train_samples if partition == "train" else holdout_samples
        )
        expected_roots = (
            train_roots if partition == "train" else holdout_roots
        )
        expected_phase = (
            "pre_training_train_only"
            if partition == "train"
            else "post_terminal_seal_holdout_only"
        )
        expected_event_name = (
            "train_cache_materialized"
            if partition == "train"
            else "holdout_cache_materialized"
        )
        readers = (
            _TRAIN_CACHE_READERS[str(arm)]
            if partition == "train"
            else _HOLDOUT_CACHE_READERS[str(arm)]
        )
        expected_terminal_seal = (
            None if partition == "train" else terminal_seal_fp
        )
        arm_directory = {
            "base_eval": "base_eval",
            "PACRE_VC_v23_control": "v23_control",
            "GCR_PACRE_v24": "candidate",
        }[str(arm)]
        expected_cache_path = (
            Path(authorization.runtime_root)
            / f"fold_{fold_id}"
            / str(partition)
            / arm_directory
            / "cache.pt"
        )
        fiemap_flags = cache.get("fiemap_extent_flags")
        if (
            cache_id
            != f"oof4-fold-{fold_id}-{partition}-{arm}"
            or
            cache.get("sample_ids") != expected_samples
            or cache.get("root_source_ids") != expected_roots
            or cache.get("closure_fingerprint")
            != fold_closure_fingerprint
            or cache.get("terminal_seal_fingerprint")
            != expected_terminal_seal
            or cache.get("realpath") != str(expected_cache_path)
            or cache.get("creation_phase") != expected_phase
            or cache.get("creation_event") != events[expected_event_name]
            or cache.get("reader_allowlist") != list(readers)
            or cache.get("is_symlink") is not False
            or cache.get("hardlink_count") != 1
            or cache.get("is_reflink") is not False
            or cache.get("shared_tensor_storage") is not False
            or cache.get("mmap_reused") is not False
            or cache.get("process_cache_reused") is not False
            or not isinstance(fiemap_flags, list)
            or not fiemap_flags
            or any(
                isinstance(flag, bool)
                or not isinstance(flag, int)
                or flag not in {0, 1}
                for flag in fiemap_flags
            )
            or fiemap_flags[-1] != 1
            or any(flag == 1 for flag in fiemap_flags[:-1])
        ):
            raise PermissionError(f"{cache_id} isolation metadata changed")
        artifact_body = {
            "schema_version": _OOF_CACHE_ARTIFACT_SCHEMA,
            "cache_id": cache_id,
            "fold_id": fold_id,
            "partition": partition,
            "arm": arm,
            "closure_fingerprint": fold_closure_fingerprint,
            "terminal_seal_fingerprint": expected_terminal_seal,
            "semantic_payload_fingerprint": cache.get(
                "semantic_payload_fingerprint"
            ),
            "root_source_ids": expected_roots,
            "sample_ids": expected_samples,
            "realpath": cache.get("realpath"),
            "device": cache.get("device"),
            "inode": cache.get("inode"),
            "size_bytes": cache.get("size_bytes"),
            "file_sha256": cache.get("file_sha256"),
            "creation_phase": expected_phase,
            "creation_event": events[expected_event_name],
            "reader_allowlist": list(readers),
            "tensor_ledger_fingerprint": cache.get(
                "tensor_ledger_fingerprint"
            ),
            "fiemap_extent_flags": fiemap_flags,
            "loader": {
                "torch_load": True,
                "weights_only": True,
                "mmap_used": False,
                "neutral_object_graph": True,
            },
        }
        artifact_fp = _sha256(
            cache.get("artifact_fingerprint"),
            name=f"{cache_id}.artifact_fingerprint",
        )
        _sha256(
            cache.get("tensor_ledger_fingerprint"),
            name=f"{cache_id}.tensor_ledger_fingerprint",
        )
        _sha256(
            cache.get("semantic_payload_fingerprint"),
            name=f"{cache_id}.semantic_payload_fingerprint",
        )
        if artifact_fp != stable_fingerprint(artifact_body):
            raise ValueError(f"{cache_id} artifact fingerprint changed")
        path, stat_result = _verify_regular_file(
            cache.get("realpath"),
            name=f"cache {cache_id}",
            expected_sha256=cache.get("file_sha256"),
            expected_size=cache.get("size_bytes"),
            expected_device=cache.get("device"),
            expected_inode=cache.get("inode"),
        )
        physical_key = (stat_result.st_dev, stat_result.st_ino)
        if physical_key in physical or str(path) in realpaths:
            raise PermissionError("OOF caches share physical storage identity")
        physical.add(physical_key)
        realpaths.add(str(path))
        from cure_lite_v24.formal_cache_artifacts import _fiemap_flags

        if list(_fiemap_flags(path)) != fiemap_flags:
            raise RuntimeError(f"{cache_id} FIEMAP ledger changed")
        purpose = (
            "train_cache_materialization"
            if partition == "train"
            else "read_only_holdout_cache_materialization"
        )
        if (
            cache_id,
            str(cache["file_sha256"]),
            purpose,
        ) not in observed_access:
            raise PermissionError(f"{cache_id} lacks access-audit evidence")
        cache_by_key[key] = cache
    if set(cache_by_key) != {
        (partition, arm)
        for partition in ("train", "holdout")
        for arm in (
            "base_eval",
            "PACRE_VC_v23_control",
            "GCR_PACRE_v24",
        )
    }:
        raise ValueError("OOF cache matrix is incomplete")
    if (
        cache_by_key[
            ("train", "PACRE_VC_v23_control")
        ]["semantic_payload_fingerprint"]
        != cache_by_key[
            ("train", "GCR_PACRE_v24")
        ]["semantic_payload_fingerprint"]
        or len(
            {
                cache_by_key[
                    ("holdout", arm)
                ]["semantic_payload_fingerprint"]
                for arm in (
                    "base_eval",
                    "PACRE_VC_v23_control",
                    "GCR_PACRE_v24",
                )
            }
        )
        != 1
        or len(
            {
                cache_by_key[
                    ("holdout", arm)
                ]["tensor_ledger_fingerprint"]
                for arm in (
                    "base_eval",
                    "PACRE_VC_v23_control",
                    "GCR_PACRE_v24",
                )
            }
        )
        != 1
    ):
        raise PermissionError("OOF semantic cache population pairing changed")
    expected_cache_order = sorted(
        raw_caches,
        key=lambda row: (
            str(row["partition"]),
            str(row["arm"]),
        ),
    )
    if raw_caches != expected_cache_order:
        raise ValueError("OOF cache entry order changed")
    cache_set_body = {
        "schema_version": _OOF_CACHE_SET_SCHEMA,
        "fold_id": fold_id,
        "cache_artifact_fingerprints": sorted(
            str(row["artifact_fingerprint"]) for row in raw_caches
        ),
        "entries": raw_caches,
    }
    if payload.get("cache_set_fingerprint") != stable_fingerprint(
        cache_set_body
    ):
        raise ValueError("OOF six-cache set fingerprint changed")

    run_start, run_start_path, _ = _verify_oof_json_artifact(
        payload.get("run_start_artifact"),
        name="OOF run_start_artifact",
        fingerprint_field="marker_fingerprint",
    )
    _exact_keys(
        run_start,
        {
            "schema_version",
            "fold_id",
            "closure_fingerprint",
            "split_receipt_fingerprint",
            "authorization_fingerprint",
            "authorization_artifact_file_sha256",
            "source_binding_fingerprint",
            "source_closure_fingerprint",
            "seed",
            "epochs",
            "steps_per_epoch",
            "updates_per_arm",
            "event_index",
            "event",
            "process_instance_fingerprint",
            "schedule_fingerprint",
            "batch_sequence_fingerprint",
            "training_population_fingerprint",
            "control_cache_artifact_fingerprint",
            "candidate_cache_artifact_fingerprint",
            "output_directory",
            "marker_path",
            "from_scratch",
            "resume_allowed",
            "automatic_retry_allowed",
            "checkpoint_policy",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "marker_fingerprint",
        },
        name="OOF run_start payload",
    )
    run_start_fp = _self_fingerprint(
        run_start,
        field_name="marker_fingerprint",
        name="OOF run_start payload",
    )
    train_semantic_fp = str(
        cache_by_key[
            ("train", "PACRE_VC_v23_control")
        ]["semantic_payload_fingerprint"]
    )
    expected_run_start_path = (
        Path(authorization.runtime_root)
        / f"fold_{fold_id}"
        / "run_start.json"
    )
    if (
        run_start.get("schema_version") != _OOF_RUN_START_SCHEMA
        or run_start.get("fold_id") != fold_id
        or run_start.get("closure_fingerprint")
        != fold_closure_fingerprint
        or run_start.get("split_receipt_fingerprint")
        != split.receipt_fingerprint
        or run_start.get("authorization_fingerprint")
        != authorization.authorization_fingerprint
        or run_start.get("authorization_artifact_file_sha256")
        != authorization.artifact_file_sha256
        or run_start.get("source_binding_fingerprint")
        != authorization.source_binding_fingerprint
        or run_start.get("source_closure_fingerprint")
        != source_closure_fp
        or source_closure_fp
        != authorization.source_closure_fingerprint
        or run_start.get("seed") != OOF_SEED
        or run_start.get("epochs") != BOUNDED_EPOCHS
        or run_start.get("steps_per_epoch") != BOUNDED_STEPS_PER_EPOCH
        or run_start.get("updates_per_arm") != BOUNDED_UPDATES
        or run_start.get("event_index") != 2
        or run_start.get("event") != "training_claimed"
        or run_start.get("training_population_fingerprint")
        != train_semantic_fp
        or run_start.get("control_cache_artifact_fingerprint")
        != cache_by_key[
            ("train", "PACRE_VC_v23_control")
        ]["artifact_fingerprint"]
        or run_start.get("candidate_cache_artifact_fingerprint")
        != cache_by_key[
            ("train", "GCR_PACRE_v24")
        ]["artifact_fingerprint"]
        or run_start.get("output_directory") != str(run_start_path.parent)
        or run_start.get("marker_path") != str(run_start_path)
        or run_start_path != expected_run_start_path
        or run_start.get("from_scratch") is not True
        or run_start.get("resume_allowed") is not False
        or run_start.get("automatic_retry_allowed") is not False
        or run_start.get("checkpoint_policy") != "final_only"
        or run_start.get("D_V_payload_accessed") is not False
        or run_start.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("OOF persistent run-start binding changed")
    process_instance_fp = _sha256(
        run_start.get("process_instance_fingerprint"),
        name="run_start.process_instance_fingerprint",
    )
    for field_name in (
        "authorization_fingerprint",
        "authorization_artifact_file_sha256",
        "source_binding_fingerprint",
        "schedule_fingerprint",
        "batch_sequence_fingerprint",
    ):
        _sha256(run_start.get(field_name), name=f"run_start.{field_name}")

    training_arms = _mapping(payload.get("training_arms"), name="training_arms")
    expected_training_arms = (
        "PACRE_VC_v23_control",
        "GCR_PACRE_v24",
    )
    if set(training_arms) != set(expected_training_arms):
        raise ValueError("OOF training arm inventory changed")
    paired_names = (
        "seed",
        "schedule_fingerprint",
        "batch_sequence_fingerprint",
        "training_population_fingerprint",
        "initial_shared_parameter_fingerprint",
        "PMOPE_fingerprint",
        "Adam_policy_fingerprint",
        "dtype_device_policy_fingerprint",
    )
    paired_values: dict[str, set[object]] = defaultdict(set)
    modules: set[str] = set()
    optimizers: set[str] = set()
    storages: dict[str, set[str]] = {}
    final_fingerprints: dict[str, str] = {}
    terminal_artifact_fingerprints: dict[str, str] = {}
    completed_capability_fingerprints: dict[str, str] = {}
    artifact_physical: set[tuple[int, int]] = set()
    repository = Path(repository_root).resolve(strict=True)
    for arm, raw in training_arms.items():
        value = _mapping(raw, name=f"training_arms[{arm}]")
        _exact_keys(value, _OOF_TRAINING_ARM_FIELDS, name=f"training arm {arm}")
        if (
            value.get("seed") != OOF_SEED
            or value.get("epochs") != BOUNDED_EPOCHS
            or value.get("steps_per_epoch") != BOUNDED_STEPS_PER_EPOCH
            or value.get("completed_updates") != BOUNDED_UPDATES
            or value.get("training_invocations") != 1
            or value.get("from_scratch") is not True
            or value.get("resume_allowed") is not False
            or value.get("automatic_retry_allowed") is not False
            or value.get("checkpoint_policy") != "final_only"
            or value.get("optimizer_state_initial_empty") is not True
            or value.get("train_root_source_ids") != train_roots
            or value.get("train_sample_ids") != train_samples
        ):
            raise ValueError(f"OOF training arm {arm} identity changed")
        for name in paired_names:
            raw_value = value.get(name)
            if name != "seed":
                _sha256(raw_value, name=f"{arm}.{name}")
            paired_values[name].add(raw_value)
        arm_source_rows = _verify_repo_source_hashes(
            value.get("source_hashes"),
            repository_root=repository,
            name=f"{arm}.source_hashes",
            exact_paths=GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
        )
        if tuple(sorted(arm_source_rows)) != tuple(sorted(closure_rows)):
            raise ValueError(f"{arm} source closure differs from fold closure")
        module_id = _text(value.get("module_instance_id"), name=f"{arm}.module")
        optimizer_id = _text(
            value.get("optimizer_instance_id"),
            name=f"{arm}.optimizer",
        )
        _sha256(module_id, name=f"{arm}.module_instance_id")
        _sha256(optimizer_id, name=f"{arm}.optimizer_instance_id")
        modules.add(module_id)
        optimizers.add(optimizer_id)
        initial_parameters, initial_ledger_fp = (
            _validate_oof_initial_parameter_ledger(
                value.get("initial_parameters"),
                name=f"{arm}.initial_parameters",
            )
        )
        if (
            value.get("initial_shared_parameter_fingerprint")
            != initial_ledger_fp
            or initial_parameters != seal_initial_parameters
        ):
            raise ValueError(f"{arm} shared initial parameter ledger changed")
        storage_rows, storage_ledger_fp, storage_ids = (
            _validate_oof_parameter_storage_ledger(
                value.get("parameter_storage_ledger"),
                initial_parameters=initial_parameters,
                name=f"{arm}.parameter_storage_ledger",
            )
        )
        if (
            value.get("parameter_storage_ledger_fingerprint")
            != storage_ledger_fp
        ):
            raise ValueError(f"{arm} parameter storage ledger changed")
        storages[arm] = storage_ids
        initial = _sha256(
            value.get("initial_model_fingerprint"),
            name=f"{arm}.initial_model_fingerprint",
        )
        final = _sha256(
            value.get("final_model_fingerprint"),
            name=f"{arm}.final_model_fingerprint",
        )
        if initial == final:
            raise ValueError(f"{arm} parameter state did not change")
        capability = _mapping(
            value.get("completed_400_capability"),
            name=f"{arm}.completed_400_capability",
        )
        _exact_keys(
            capability,
            {
                "schema_version",
                "fold_id",
                "arm",
                "closure_fingerprint",
                "run_start",
                "train_cache",
                "seed",
                "epochs",
                "steps_per_epoch",
                "completed_updates",
                "training_invocations",
                "schedule_fingerprint",
                "batch_sequence_fingerprint",
                "shared_initial_parameter_fingerprint",
                "initial_parameters",
                "model_config",
                "module_instance_id",
                "optimizer_instance_id",
                "parameter_storage_ledger",
                "parameter_storage_ledger_fingerprint",
                "optimizer_fqcn",
                "optimizer_config_fingerprint",
                "objective",
                "objective_policy_fingerprint",
                "training_result_fingerprint",
                "terminal_artifact",
                "source_hashes",
                "from_scratch",
                "resume_allowed",
                "automatic_retry_allowed",
                "checkpoint_policy",
                "holdout_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
                "capability_fingerprint",
            },
            name=f"{arm}.completed_400_capability",
        )
        capability_fp = _self_fingerprint(
            capability,
            field_name="capability_fingerprint",
            name=f"{arm}.completed_400_capability",
        )
        capability_run_start = _mapping(
            capability.get("run_start"),
            name=f"{arm}.capability.run_start",
        )
        _exact_keys(
            capability_run_start,
            {
                "marker_fingerprint",
                "marker_path",
                "marker_file_sha256",
                "authorization_fingerprint",
                "source_closure_fingerprint",
            },
            name=f"{arm}.capability.run_start",
        )
        capability_cache = _mapping(
            capability.get("train_cache"),
            name=f"{arm}.capability.train_cache",
        )
        _exact_keys(
            capability_cache,
            {
                "artifact_fingerprint",
                "semantic_payload_fingerprint",
                "reader_authorization_fingerprint",
            },
            name=f"{arm}.capability.train_cache",
        )
        model_config = _mapping(
            capability.get("model_config"),
            name=f"{arm}.capability.model_config",
        )
        _sha256(
            capability_cache.get("reader_authorization_fingerprint"),
            name=f"{arm}.capability.reader_authorization_fingerprint",
        )
        _sha256(
            capability.get("training_result_fingerprint"),
            name=f"{arm}.capability.training_result_fingerprint",
        )
        _exact_keys(
            model_config,
            {
                "feature_channels",
                "feature_stride",
                "width",
                "parameter_count",
            },
            name=f"{arm}.capability.model_config",
        )
        expected_train_cache = cache_by_key[("train", arm)]
        if (
            value.get("completed_400_capability_fingerprint")
            != capability_fp
            or value.get("run_start_marker_fingerprint") != run_start_fp
            or capability.get("schema_version")
            != _OOF_COMPLETED_CAPABILITY_SCHEMA
            or capability.get("fold_id") != fold_id
            or capability.get("arm") != arm
            or capability.get("closure_fingerprint")
            != fold_closure_fingerprint
            or capability_run_start.get("marker_fingerprint")
            != run_start_fp
            or capability_run_start.get("marker_path")
            != str(run_start_path)
            or capability_run_start.get("marker_file_sha256")
            != _mapping(
                payload.get("run_start_artifact"),
                name="run_start_artifact",
            ).get("file_sha256")
            or capability_run_start.get("authorization_fingerprint")
            != run_start.get("authorization_fingerprint")
            or capability_run_start.get("source_closure_fingerprint")
            != source_closure_fp
            or capability_cache.get("artifact_fingerprint")
            != expected_train_cache["artifact_fingerprint"]
            or capability_cache.get("semantic_payload_fingerprint")
            != train_semantic_fp
            or capability.get("seed") != OOF_SEED
            or capability.get("epochs") != BOUNDED_EPOCHS
            or capability.get("steps_per_epoch")
            != BOUNDED_STEPS_PER_EPOCH
            or capability.get("completed_updates") != BOUNDED_UPDATES
            or capability.get("training_invocations") != 1
            or capability.get("schedule_fingerprint")
            != run_start.get("schedule_fingerprint")
            or capability.get("batch_sequence_fingerprint")
            != run_start.get("batch_sequence_fingerprint")
            or capability.get("shared_initial_parameter_fingerprint")
            != initial_ledger_fp
            or capability.get("initial_parameters") != initial_parameters
            or capability.get("module_instance_id") != module_id
            or capability.get("optimizer_instance_id") != optimizer_id
            or capability.get("parameter_storage_ledger") != storage_rows
            or capability.get("parameter_storage_ledger_fingerprint")
            != storage_ledger_fp
            or capability.get("optimizer_fqcn")
            != "torch.optim.adam.Adam"
            or capability.get("optimizer_config_fingerprint")
            != value.get("Adam_policy_fingerprint")
            or capability.get("objective") != "pmope_joint"
            or capability.get("objective_policy_fingerprint")
            != value.get("PMOPE_fingerprint")
            or capability.get("terminal_artifact")
            != value.get("terminal_artifact")
            or capability.get("training_result_fingerprint")
            != _mapping(
                value.get("terminal_artifact"),
                name=f"{arm}.terminal_artifact",
            ).get("training_result_fingerprint")
            or model_config
            != {
                "feature_channels": 64,
                "feature_stride": 4,
                "width": 32,
                "parameter_count": 64_064,
            }
            or capability.get("from_scratch") is not True
            or capability.get("resume_allowed") is not False
            or capability.get("automatic_retry_allowed") is not False
            or capability.get("checkpoint_policy") != "final_only"
            or capability.get("holdout_payload_accessed") is not False
            or capability.get("D_V_payload_accessed") is not False
            or capability.get("D_T_payload_accessed") is not False
        ):
            raise PermissionError(f"{arm} completed-400 capability changed")
        capability_sources = _verify_repo_source_hashes(
            capability.get("source_hashes"),
            repository_root=repository,
            name=f"{arm}.capability.source_hashes",
        )
        if not capability_sources:
            raise ValueError(f"{arm} capability source closure is empty")
        terminal_artifact_fp, stat_result = (
            _validate_oof_terminal_artifact(
                value.get("terminal_artifact"),
                fold_id=fold_id,
                arm=arm,
                final_model_fingerprint=final,
                run_start_marker_fingerprint=run_start_fp,
                runtime_root=authorization.runtime_root,
            )
        )
        if (
            value.get("terminal_artifact_fingerprint")
            != terminal_artifact_fp
        ):
            raise ValueError(f"{arm} terminal artifact seal changed")
        physical_key = (stat_result.st_dev, stat_result.st_ino)
        if physical_key in artifact_physical:
            raise PermissionError("OOF terminal artifacts share storage")
        artifact_physical.add(physical_key)
        final_fingerprints[arm] = final
        terminal_artifact_fingerprints[arm] = terminal_artifact_fp
        completed_capability_fingerprints[arm] = capability_fp
    if (
        any(len(values) != 1 for values in paired_values.values())
        or len(modules) != 2
        or len(optimizers) != 2
        or storages[expected_training_arms[0]]
        & storages[expected_training_arms[1]]
        or terminal_seal.get("terminal_artifact_fingerprints")
        != terminal_artifact_fingerprints
        or terminal_seal.get("completed_400_capability_fingerprints")
        != completed_capability_fingerprints
        or terminal_seal.get("run_start_marker_fingerprint")
        != run_start_fp
        or terminal_seal.get("schedule_fingerprint")
        != run_start.get("schedule_fingerprint")
        or terminal_seal.get("batch_sequence_fingerprint")
        != run_start.get("batch_sequence_fingerprint")
        or terminal_seal.get("semantic_cache_fingerprint")
        != train_semantic_fp
        or terminal_seal.get("optimizer_config_fingerprint")
        != training_arms[
            "PACRE_VC_v23_control"
        ]["Adam_policy_fingerprint"]
        or terminal_seal.get("objective_policy_fingerprint")
        != training_arms[
            "PACRE_VC_v23_control"
        ]["PMOPE_fingerprint"]
    ):
        raise PermissionError("OOF paired identity/independence changed")

    base_cache_sha = _sha256(
        cache_by_key[("train", "base_eval")].get("file_sha256"),
        name="base train cache sha",
    )
    base_b_ledger_fp, selected_threshold = (
        _validate_base_b_train_fold_ledger(
            payload.get("BaseB_train_fold_selection"),
            train_sample_roots=train_sample_roots,
            holdout_sample_roots=holdout_sample_roots,
            base_train_cache_sha256=base_cache_sha,
            access_audit=access,
            repository_root=repository,
            protocol_preregistration_fingerprint=(
                split.protocol_preregistration_fingerprint
            ),
        )
    )
    eval_fps = _mapping(
        payload.get("evaluation_artifact_fingerprints"),
        name="evaluation_artifact_fingerprints",
    )
    if set(eval_fps) != set(OOF_ARMS):
        raise ValueError("OOF evaluation artifact inventory changed")
    normalized_eval_fps = {
        arm: _sha256(eval_fps.get(arm), name=f"evaluation artifact {arm}")
        for arm in OOF_ARMS
    }
    expected_base_a = stable_fingerprint(
        {"arm": "BaseA", "threshold": BASE_A_THRESHOLD}
    )
    expected_base_b = stable_fingerprint(
        {
            "arm": "BaseB_train_fold_selected",
            "candidate_ledger_fingerprint": base_b_ledger_fp,
            "selected_threshold": selected_threshold,
        }
    )
    if (
        normalized_eval_fps["BaseA"] != expected_base_a
        or normalized_eval_fps["BaseB_train_fold_selected"]
        != expected_base_b
        or normalized_eval_fps["PACRE_VC_v23_control"]
        != final_fingerprints["PACRE_VC_v23_control"]
        or normalized_eval_fps["GCR_PACRE_v24"]
        != final_fingerprints["GCR_PACRE_v24"]
        or normalized_eval_fps["GCR_PACRE_v24_forced_G1"]
        != final_fingerprints["GCR_PACRE_v24"]
        or payload.get("held_out_prediction_role") != "factual_only"
    ):
        raise ValueError("OOF evaluation artifact binding changed")
    raw_evaluation_ledgers = _mapping(
        payload.get("evaluation_ledger_artifacts"),
        name="evaluation_ledger_artifacts",
    )
    if set(raw_evaluation_ledgers) != set(OOF_ARMS):
        raise ValueError("OOF evaluation ledger inventory changed")
    evaluation_ledger_fingerprints: dict[str, str] = {}
    evaluation_rows_by_arm: dict[str, object] = {}
    evaluation_dataset_fingerprints: set[str] = set()
    evaluator_fingerprints: set[str] = set()
    evaluation_physical: set[tuple[int, int]] = set()
    for arm in OOF_ARMS:
        expected_model_fp = (
            None
            if arm in {"BaseA", "BaseB_train_fold_selected"}
            else (
                final_fingerprints["PACRE_VC_v23_control"]
                if arm == "PACRE_VC_v23_control"
                else final_fingerprints["GCR_PACRE_v24"]
            )
        )
        expected_operating_point = (
            selected_threshold
            if arm == "BaseB_train_fold_selected"
            else BASE_A_THRESHOLD
        )
        ledger_fp, ledger_payload, ledger_stat = (
            _validate_oof_evaluation_ledger_artifact(
                raw_evaluation_ledgers[arm],
                fold_id=fold_id,
                arm=arm,
                held_out_sample_roots=holdout_sample_roots,
                expected_model_fingerprint=expected_model_fp,
                expected_operating_point=expected_operating_point,
                expected_path=(
                    Path(authorization.runtime_root)
                    / f"fold_{fold_id}"
                    / "evaluation"
                    / f"{arm}.json"
                ),
            )
        )
        physical_key = (ledger_stat.st_dev, ledger_stat.st_ino)
        if physical_key in evaluation_physical:
            raise PermissionError("OOF evaluation ledgers share storage")
        evaluation_physical.add(physical_key)
        evaluation_ledger_fingerprints[arm] = ledger_fp
        evaluation_rows_by_arm[arm] = ledger_payload["per_sample_rows"]
        evaluation_dataset_fingerprints.add(
            str(ledger_payload["dataset_fingerprint"])
        )
        evaluator_fingerprints.add(
            str(ledger_payload["evaluator_fingerprint"])
        )
    expected_holdout_dataset_fingerprint = str(
        cache_by_key[
            ("holdout", "base_eval")
        ]["semantic_payload_fingerprint"]
    )
    from cure_lite_v24.oof_evaluation import OOFConcreteEvaluator

    expected_evaluator_fingerprint = (
        OOFConcreteEvaluator.fixed().evaluator_fingerprint
    )
    if (
        evaluation_dataset_fingerprints
        != {expected_holdout_dataset_fingerprint}
        or evaluator_fingerprints != {expected_evaluator_fingerprint}
    ):
        raise PermissionError(
            "OOF evaluation ledgers are not bound to the verified held-out "
            "cache and frozen evaluator"
        )
    factual_artifact, factual_path, _ = _verify_oof_json_artifact(
        payload.get("factual_rows_artifact"),
        name="OOF factual_rows_artifact",
        fingerprint_field="ledger_fingerprint",
    )
    expected_factual_path = (
        Path(authorization.runtime_root)
        / f"fold_{fold_id}"
        / "factual_rows.json"
    )
    if factual_path != expected_factual_path:
        raise PermissionError("OOF factual rows path is not the fixed slot")
    _exact_keys(
        factual_artifact,
        {
            "schema_version",
            "fold_id",
            "closure_fingerprint",
            "rows",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "ledger_fingerprint",
        },
        name="OOF factual rows payload",
    )
    _self_fingerprint(
        factual_artifact,
        field_name="ledger_fingerprint",
        name="OOF factual rows payload",
    )
    factual_rows = factual_artifact.get("rows")
    if (
        factual_artifact.get("schema_version")
        != "cure-lite-v24-gcr-pacre-oof4-factual-rows-artifact-v1"
        or factual_artifact.get("fold_id") != fold_id
        or factual_artifact.get("closure_fingerprint")
        != fold_closure_fingerprint
        or factual_artifact.get("D_V_payload_accessed") is not False
        or factual_artifact.get("D_T_payload_accessed") is not False
        or not isinstance(factual_rows, list)
        or len(factual_rows) != len(OOF_ARMS) * len(holdout_samples)
    ):
        raise ValueError("OOF factual rows artifact identity changed")
    factual_by_key: dict[tuple[str, str], dict[str, object]] = {}
    shared_factual: dict[str, tuple[object, object]] = {}
    for index, raw in enumerate(factual_rows):
        row = _mapping(raw, name=f"OOF factual rows[{index}]")
        _exact_keys(row, _FACTUAL_ROW_FIELDS, name=f"OOF factual row {index}")
        arm = _text(row.get("arm"), name=f"factual[{index}].arm")
        sample_id = _text(
            row.get("sample_id"),
            name=f"factual[{index}].sample_id",
        )
        key = (arm, sample_id)
        ledger_rows = evaluation_rows_by_arm.get(arm)
        expected_row = next(
            (
                candidate
                for candidate in ledger_rows
                if isinstance(candidate, Mapping)
                and candidate.get("sample_id") == sample_id
            ),
            None,
        ) if isinstance(ledger_rows, list) else None
        if (
            arm not in OOF_ARMS
            or sample_id not in holdout_sample_roots
            or key in factual_by_key
            or row.get("split") != "D_R"
            or row.get("evidence_role") != "factual_only"
            or row.get("fold_id") != fold_id
            or row.get("root_source_id")
            != holdout_sample_roots[sample_id]
            or row.get("terminal_artifact_fingerprint")
            != normalized_eval_fps[arm]
            or row.get("evaluation_contract_fingerprint")
            != next(iter(evaluator_fingerprints))
            or not isinstance(expected_row, Mapping)
            or row.get("sufficient_statistics")
            != expected_row.get("statistics")
        ):
            raise ValueError("OOF factual row/evaluation ledger binding changed")
        _sha256(row.get("gt_fingerprint"), name="factual gt fingerprint")
        _sha256(
            row.get("anchor_state_fingerprint"),
            name="factual anchor fingerprint",
        )
        shared = (
            row.get("gt_fingerprint"),
            row.get("anchor_state_fingerprint"),
        )
        if shared_factual.setdefault(sample_id, shared) != shared:
            raise ValueError("OOF factual GT/anchor identity differs by arm")
        factual_by_key[key] = row
    if set(factual_by_key) != {
        (arm, sample_id)
        for arm in OOF_ARMS
        for sample_id in holdout_samples
    }:
        raise ValueError("OOF factual rows do not cover the exact fold matrix")
    from cure_lite_v24.oof_evaluation import (
        mechanically_replay_oof_fold_evidence,
    )

    mechanically_replay_oof_fold_evidence(
        payload,
        runtime_root=authorization.runtime_root,
    )
    receipt_fingerprint = _self_fingerprint(
        payload,
        field_name="receipt_fingerprint",
        name="OOF fold receipt",
    )
    return _register_token(VerifiedOOFFold(
        receipt_fingerprint=receipt_fingerprint,
        split_receipt_fingerprint=split.receipt_fingerprint,
        fold_id=fold_id,
        train_sample_roots=tuple(sorted(train_sample_roots.items())),
        holdout_sample_roots=tuple(sorted(holdout_sample_roots.items())),
        evaluation_artifact_fingerprints=tuple(
            (arm, normalized_eval_fps[arm]) for arm in OOF_ARMS
        ),
        evaluation_ledger_fingerprints=tuple(
            (arm, evaluation_ledger_fingerprints[arm])
            for arm in OOF_ARMS
        ),
        evaluation_rows_json=canonical_json(evaluation_rows_by_arm),
        factual_rows_json=canonical_json(factual_rows),
        evaluation_dataset_fingerprint=next(
            iter(evaluation_dataset_fingerprints)
        ),
        evaluator_fingerprint=next(iter(evaluator_fingerprints)),
        process_instance_fingerprint=process_instance_fp,
        access_audit_receipt_fingerprint=access.receipt_fingerprint,
        base_b_ledger_fingerprint=base_b_ledger_fp,
        _issuer=_TOKEN_ISSUER,
    ))


_FACTUAL_ROW_FIELDS = {
    "split",
    "evidence_role",
    "fold_id",
    "arm",
    "sample_id",
    "root_source_id",
    "gt_fingerprint",
    "anchor_state_fingerprint",
    "evaluation_contract_fingerprint",
    "terminal_artifact_fingerprint",
    "sufficient_statistics",
}
_SHARED_FACTUAL_FIELDS = (
    "total_gt",
    "total_anchor_misses",
    "total_anchor_covered",
    "total_reachable_anchor_misses",
    "total_pixels",
)


def pool_factual_only_rows(
    rows: Sequence[Mapping[str, object]],
    verified_fold: VerifiedOOFFold,
    *,
    access_audit: VerifiedAccessAudit,
) -> VerifiedFactualPool:
    """Pool one fold only after exact sample/root and denominator matching."""

    fold = _token(verified_fold, VerifiedOOFFold, name="verified_fold")
    access = _token(access_audit, VerifiedAccessAudit, name="access_audit")
    assert isinstance(fold, VerifiedOOFFold)
    assert isinstance(access, VerifiedAccessAudit)
    persisted_rows = json.loads(fold.factual_rows_json)
    if not isinstance(persisted_rows, list) or list(rows) != persisted_rows:
        raise PermissionError(
            "OOF pooling rows differ from the persisted factual artifact"
        )
    if (
        access.receipt_fingerprint
        != fold.access_audit_receipt_fingerprint
        or access.stage_id != f"oof4_fold_{fold.fold_id}"
        or access.allowed_splits != ("D_R",)
    ):
        raise PermissionError("factual pooling access receipt changed")
    sample_roots = dict(fold.holdout_sample_roots)
    artifact_fps = dict(fold.evaluation_artifact_fingerprints)
    raw_evaluation_rows = _mapping(
        json.loads(fold.evaluation_rows_json),
        name="verified fold evaluation rows",
    )
    evaluation_rows: dict[str, dict[str, dict[str, object]]] = {}
    for arm in OOF_ARMS:
        arm_rows = raw_evaluation_rows.get(arm)
        if not isinstance(arm_rows, list):
            raise AssertionError("verified fold evaluation rows changed")
        normalized_arm_rows = [
            _mapping(row, name=f"verified evaluation row {arm}")
            for row in arm_rows
        ]
        evaluation_rows[arm] = {
            str(row["sample_id"]): row for row in normalized_arm_rows
        }
    expected_samples = set(sample_roots)
    observed: dict[str, dict[str, FactualSufficientStatistics]] = {
        arm: {} for arm in OOF_ARMS
    }
    shared_by_sample: dict[str, tuple[object, ...]] = {}
    normalized_rows: list[dict[str, object]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, name=f"factual rows[{index}]")
        _exact_keys(row, _FACTUAL_ROW_FIELDS, name=f"factual row {index}")
        if (
            row.get("split") != "D_R"
            or row.get("evidence_role") != "factual_only"
            or row.get("fold_id") != fold.fold_id
        ):
            raise ValueError("OOF evidence must be factual D_R held-out data")
        arm = _text(row.get("arm"), name=f"row[{index}].arm")
        if arm not in OOF_ARMS:
            raise ValueError(f"unexpected OOF arm {arm!r}")
        sample_id = _text(
            row.get("sample_id"),
            name=f"row[{index}].sample_id",
        )
        root_id = _text(
            row.get("root_source_id"),
            name=f"row[{index}].root_source_id",
        )
        if sample_roots.get(sample_id) != root_id:
            raise ValueError("OOF row sample→root mapping changed")
        if sample_id in observed[arm]:
            raise ValueError(f"duplicate OOF arm/sample row {(arm, sample_id)!r}")
        digests = tuple(
            _sha256(row.get(name), name=f"row[{index}].{name}")
            for name in (
                "gt_fingerprint",
                "anchor_state_fingerprint",
                "evaluation_contract_fingerprint",
                "terminal_artifact_fingerprint",
            )
        )
        if digests[-1] != artifact_fps[arm]:
            raise ValueError("OOF row is bound to the wrong arm artifact")
        if digests[2] != fold.evaluator_fingerprint:
            raise ValueError("OOF row evaluator contract changed")
        stats = FactualSufficientStatistics.from_mapping(
            _mapping(
                row.get("sufficient_statistics"),
                name=f"row[{index}].sufficient_statistics",
            )
        )
        if stats.images != 1:
            raise ValueError("each OOF factual row must describe exactly one image")
        expected_evaluation_row = evaluation_rows[arm].get(sample_id)
        if (
            expected_evaluation_row is None
            or expected_evaluation_row.get("root_source_id") != root_id
            or expected_evaluation_row.get("statistics")
            != asdict(stats)
        ):
            raise ValueError(
                "OOF factual row differs from the sealed evaluation ledger"
            )
        shared = (
            digests[0],
            digests[1],
            digests[2],
            *(getattr(stats, name) for name in _SHARED_FACTUAL_FIELDS),
        )
        prior = shared_by_sample.setdefault(sample_id, shared)
        if prior != shared:
            raise ValueError(
                "OOF arms differ on GT/anchor/contract or denominators"
            )
        observed[arm][sample_id] = stats
        normalized_rows.append(row)
    for arm in OOF_ARMS:
        if set(observed[arm]) != expected_samples:
            raise ValueError(f"OOF arm {arm} does not cover exact held-out samples")
    natural_rows = evaluation_rows["GCR_PACRE_v24"]
    forced_rows = evaluation_rows["GCR_PACRE_v24_forced_G1"]
    field_difference_ledger = [
        {
            "fold_id": fold.fold_id,
            "sample_id": sample_id,
            "natural_output_fingerprint": natural_rows[sample_id][
                "field_fingerprint"
            ],
            "forced_G1_output_fingerprint": forced_rows[sample_id][
                "field_fingerprint"
            ],
        }
        for sample_id in sorted(expected_samples)
        if natural_rows[sample_id]["field_fingerprint"]
        != forced_rows[sample_id]["field_fingerprint"]
    ]
    prediction_difference_ledger = [
        {
            "fold_id": fold.fold_id,
            "sample_id": sample_id,
            "natural_output_fingerprint": natural_rows[sample_id][
                "prediction_fingerprint"
            ],
            "forced_G1_output_fingerprint": forced_rows[sample_id][
                "prediction_fingerprint"
            ],
        }
        for sample_id in sorted(expected_samples)
        if natural_rows[sample_id]["prediction_fingerprint"]
        != forced_rows[sample_id]["prediction_fingerprint"]
    ]
    pooled_statistics: dict[str, FactualSufficientStatistics] = {}
    for arm in OOF_ARMS:
        values = [observed[arm][sample] for sample in sorted(expected_samples)]
        total = values[0]
        for value in values[1:]:
            total = total.plus(value)
        pooled_statistics[arm] = total
    body = {
        "schema_version": "cure-lite-v24-oof-fold-factual-pool-v2",
        "split_preregistration_fingerprint": (
            fold.split_receipt_fingerprint
        ),
        "fold_receipt_fingerprint": fold.receipt_fingerprint,
        "fold_id": fold.fold_id,
        "access_audit_receipt_fingerprint": access.receipt_fingerprint,
        "held_out_sample_roots": dict(sorted(sample_roots.items())),
        "evaluation_artifact_fingerprints": dict(
            fold.evaluation_artifact_fingerprints
        ),
        "evaluation_ledger_fingerprints": dict(
            fold.evaluation_ledger_fingerprints
        ),
        "evaluation_dataset_fingerprint": (
            fold.evaluation_dataset_fingerprint
        ),
        "evaluator_fingerprint": fold.evaluator_fingerprint,
        "process_instance_fingerprint": (
            fold.process_instance_fingerprint
        ),
        "verified_field_difference_ledger": field_difference_ledger,
        "verified_field_difference_ledger_fingerprint": stable_fingerprint(
            field_difference_ledger
        ),
        "verified_prediction_difference_ledger": (
            prediction_difference_ledger
        ),
        "verified_prediction_difference_ledger_fingerprint": (
            stable_fingerprint(prediction_difference_ledger)
        ),
        "arm_sufficient_statistics": {
            arm: asdict(pooled_statistics[arm]) for arm in OOF_ARMS
        },
        "arm_metrics": {
            arm: pooled_statistics[arm].metrics() for arm in OOF_ARMS
        },
        "factual_row_ledger_fingerprint": stable_fingerprint(
            sorted(
                normalized_rows,
                key=lambda row: (str(row["arm"]), str(row["sample_id"])),
            )
        ),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    fingerprint = stable_fingerprint(body)
    payload = {**body, "evidence_fingerprint": fingerprint}
    return _register_token(VerifiedFactualPool(
        payload_json=canonical_json(payload),
        evidence_fingerprint=fingerprint,
        fold_receipt_fingerprint=fold.receipt_fingerprint,
        split_receipt_fingerprint=fold.split_receipt_fingerprint,
        fold_id=fold.fold_id,
        access_audit_receipt_fingerprint=access.receipt_fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


def combine_oof4_factual_pools(
    fold_pools: Sequence[VerifiedFactualPool],
    verified_split: VerifiedOOF4Split,
) -> VerifiedOOFPooledEvidence:
    """Add the four verified fold statistics into one Formal-equivalent pool."""

    split = _token(
        verified_split,
        VerifiedOOF4Split,
        name="verified_split",
    )
    assert isinstance(split, VerifiedOOF4Split)
    if len(fold_pools) != OOF_FOLD_COUNT:
        raise ValueError("OOF4 pooling requires exactly four verified folds")
    pools: list[VerifiedFactualPool] = []
    for index, raw in enumerate(fold_pools):
        value = _token(raw, VerifiedFactualPool, name=f"fold_pools[{index}]")
        assert isinstance(value, VerifiedFactualPool)
        if value.split_receipt_fingerprint != split.receipt_fingerprint:
            raise ValueError("OOF fold pool belongs to another split")
        pools.append(value)
    if sorted(pool.fold_id for pool in pools) != list(range(OOF_FOLD_COUNT)):
        raise ValueError("OOF4 fold IDs are incomplete or duplicated")
    pools.sort(key=lambda pool: pool.fold_id)
    sample_roots: dict[str, str] = {}
    pooled_stats: dict[str, FactualSufficientStatistics] = {}
    artifact_fps_by_fold: dict[str, object] = {}
    ledger_fps_by_fold: dict[str, object] = {}
    dataset_fps_by_fold: dict[str, object] = {}
    evaluator_fps_by_fold: dict[str, object] = {}
    candidate_metrics_by_fold: dict[str, dict[str, object]] = {}
    field_difference_ledger: list[dict[str, object]] = []
    prediction_difference_ledger: list[dict[str, object]] = []
    process_instance_fingerprints: list[str] = []
    held_out_samples_by_fold: dict[str, list[str]] = {}
    for pool in pools:
        payload = pool.payload
        fold_samples = _mapping(
            payload.get("held_out_sample_roots"),
            name=f"fold {pool.fold_id} held_out_sample_roots",
        )
        if set(sample_roots) & set(fold_samples):
            raise ValueError("OOF held-out sample appears in multiple folds")
        sample_roots.update(
            {
                _text(key, name="sample_id"): _text(value, name="root_id")
                for key, value in fold_samples.items()
            }
        )
        held_out_samples_by_fold[str(pool.fold_id)] = sorted(fold_samples)
        stats = _mapping(
            payload.get("arm_sufficient_statistics"),
            name=f"fold {pool.fold_id} statistics",
        )
        for arm in OOF_ARMS:
            row = FactualSufficientStatistics.from_mapping(
                _mapping(stats.get(arm), name=f"fold {pool.fold_id}/{arm}")
            )
            pooled_stats[arm] = (
                row if arm not in pooled_stats else pooled_stats[arm].plus(row)
            )
        artifact_fps_by_fold[str(pool.fold_id)] = payload[
            "evaluation_artifact_fingerprints"
        ]
        ledger_fps_by_fold[str(pool.fold_id)] = payload[
            "evaluation_ledger_fingerprints"
        ]
        dataset_fps_by_fold[str(pool.fold_id)] = payload[
            "evaluation_dataset_fingerprint"
        ]
        evaluator_fps_by_fold[str(pool.fold_id)] = payload[
            "evaluator_fingerprint"
        ]
        fold_metrics = _mapping(
            payload.get("arm_metrics"),
            name=f"fold {pool.fold_id} arm metrics",
        )
        candidate_metrics_by_fold[str(pool.fold_id)] = dict(
            _mapping(
                fold_metrics.get("GCR_PACRE_v24"),
                name=f"fold {pool.fold_id} candidate metrics",
            )
        )
        raw_field_differences = payload.get(
            "verified_field_difference_ledger"
        )
        raw_prediction_differences = payload.get(
            "verified_prediction_difference_ledger"
        )
        if (
            not isinstance(raw_field_differences, list)
            or not isinstance(raw_prediction_differences, list)
            or payload.get(
                "verified_field_difference_ledger_fingerprint"
            )
            != stable_fingerprint(raw_field_differences)
            or payload.get(
                "verified_prediction_difference_ledger_fingerprint"
            )
            != stable_fingerprint(raw_prediction_differences)
        ):
            raise AssertionError("verified fold gate-difference ledger changed")
        field_difference_ledger.extend(
            _mapping(row, name="verified field difference")
            for row in raw_field_differences
        )
        prediction_difference_ledger.extend(
            _mapping(row, name="verified prediction difference")
            for row in raw_prediction_differences
        )
        process_instance_fingerprints.append(
            _sha256(
                payload.get("process_instance_fingerprint"),
                name=f"fold {pool.fold_id} process instance",
            )
        )
    if sample_roots != split.root_by_sample:
        raise ValueError("four fold pools do not cover the exact frozen D_R map")
    if len(set(process_instance_fingerprints)) != OOF_FOLD_COUNT:
        raise PermissionError(
            "OOF folds were not executed in four independent processes"
        )
    body = {
        "schema_version": "cure-lite-v24-oof4-factual-pool-v2",
        "split_preregistration_fingerprint": split.receipt_fingerprint,
        "fold_receipt_fingerprints": [
            pool.fold_receipt_fingerprint for pool in pools
        ],
        "fold_evidence_fingerprints": [
            pool.evidence_fingerprint for pool in pools
        ],
        "access_audit_receipt_fingerprints": [
            pool.access_audit_receipt_fingerprint for pool in pools
        ],
        "sample_roots_fingerprint": stable_fingerprint(sample_roots),
        "sample_count": len(sample_roots),
        "evaluation_artifact_fingerprints_by_fold": artifact_fps_by_fold,
        "evaluation_ledger_fingerprints_by_fold": ledger_fps_by_fold,
        "evaluation_dataset_fingerprints_by_fold": dataset_fps_by_fold,
        "evaluator_fingerprints_by_fold": evaluator_fps_by_fold,
        "candidate_metrics_by_fold": candidate_metrics_by_fold,
        "verified_field_difference_ledger": field_difference_ledger,
        "verified_field_difference_ledger_fingerprint": stable_fingerprint(
            field_difference_ledger
        ),
        "verified_prediction_difference_ledger": (
            prediction_difference_ledger
        ),
        "verified_prediction_difference_ledger_fingerprint": (
            stable_fingerprint(prediction_difference_ledger)
        ),
        "held_out_sample_ids_by_fold": held_out_samples_by_fold,
        "fold_process_instance_fingerprints": (
            process_instance_fingerprints
        ),
        "arm_sufficient_statistics": {
            arm: asdict(pooled_stats[arm]) for arm in OOF_ARMS
        },
        "arm_metrics": {
            arm: pooled_stats[arm].metrics() for arm in OOF_ARMS
        },
        "aggregation": "additive_sufficient_statistics",
        "fold_metric_arithmetic_mean_used": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    fingerprint = stable_fingerprint(body)
    payload = {**body, "evidence_fingerprint": fingerprint}
    return _register_token(VerifiedOOFPooledEvidence(
        payload_json=canonical_json(payload),
        evidence_fingerprint=fingerprint,
        split_receipt_fingerprint=split.receipt_fingerprint,
        fold_receipt_fingerprints=tuple(
            pool.fold_receipt_fingerprint for pool in pools
        ),
        access_audit_receipt_fingerprints=tuple(
            pool.access_audit_receipt_fingerprint for pool in pools
        ),
        _issuer=_TOKEN_ISSUER,
    ))


def verify_gate_path_receipt(
    receipt: Mapping[str, object],
    pooled_evidence: VerifiedOOFPooledEvidence,
) -> VerifiedGatePathEvidence:
    """Verify that forced-G1 is a same-terminal, non-trivial read-only path."""

    pooled = _token(
        pooled_evidence,
        VerifiedOOFPooledEvidence,
        name="pooled_evidence",
    )
    assert isinstance(pooled, VerifiedOOFPooledEvidence)
    payload = _mapping(receipt, name="gate path receipt")
    _exact_keys(
        payload,
        {
            "schema_version",
            "pooled_evidence_fingerprint",
            "sample_roots_fingerprint",
            "v24_terminal_artifact_fingerprints",
            "forced_G1_terminal_artifact_fingerprints",
            "forced_G1_retrained",
            "field_difference_count",
            "prediction_difference_count",
            "field_difference_ledger",
            "field_difference_ledger_fingerprint",
            "prediction_difference_ledger",
            "prediction_difference_ledger_fingerprint",
            "access_audit_receipt_fingerprints",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "receipt_fingerprint",
        },
        name="gate path receipt",
    )
    pooled_payload = pooled.payload
    artifacts_by_fold = _mapping(
        pooled_payload.get("evaluation_artifact_fingerprints_by_fold"),
        name="pooled artifacts by fold",
    )
    expected_artifacts = [
        _sha256(
            _mapping(
                artifacts_by_fold[str(fold_id)],
                name=f"fold {fold_id} artifacts",
            ).get("GCR_PACRE_v24"),
            name=f"fold {fold_id} v24 artifact",
        )
        for fold_id in range(OOF_FOLD_COUNT)
    ]
    samples_by_fold = _mapping(
        pooled_payload.get("held_out_sample_ids_by_fold"),
        name="pooled held-out samples by fold",
    )

    def difference_ledger(
        raw_rows: object,
        *,
        kind: str,
    ) -> list[dict[str, object]]:
        if not isinstance(raw_rows, list):
            raise ValueError(f"{kind} difference ledger must be a list")
        normalized: list[dict[str, object]] = []
        seen: set[tuple[int, str]] = set()
        for index, raw_row in enumerate(raw_rows):
            row = _mapping(raw_row, name=f"{kind} differences[{index}]")
            _exact_keys(
                row,
                {
                    "fold_id",
                    "sample_id",
                    "natural_output_fingerprint",
                    "forced_G1_output_fingerprint",
                },
                name=f"{kind} differences[{index}]",
            )
            fold_id = _integer(
                row.get("fold_id"),
                name=f"{kind} differences[{index}].fold_id",
            )
            sample_id = _text(
                row.get("sample_id"),
                name=f"{kind} differences[{index}].sample_id",
            )
            natural = _sha256(
                row.get("natural_output_fingerprint"),
                name=f"{kind} differences[{index}].natural",
            )
            forced = _sha256(
                row.get("forced_G1_output_fingerprint"),
                name=f"{kind} differences[{index}].forced",
            )
            fold_samples = samples_by_fold.get(str(fold_id))
            if (
                not isinstance(fold_samples, list)
                or sample_id not in fold_samples
                or natural == forced
                or (fold_id, sample_id) in seen
            ):
                raise ValueError(f"{kind} difference evidence is invalid")
            seen.add((fold_id, sample_id))
            normalized.append(row)
        return normalized

    field_rows = difference_ledger(
        payload.get("field_difference_ledger"),
        kind="field",
    )
    prediction_rows = difference_ledger(
        payload.get("prediction_difference_ledger"),
        kind="prediction",
    )
    expected_field_rows = pooled_payload.get(
        "verified_field_difference_ledger"
    )
    expected_prediction_rows = pooled_payload.get(
        "verified_prediction_difference_ledger"
    )
    if (
        payload.get("schema_version")
        != "cure-lite-v24-gcr-pacre-forced-G1-path-v2"
        or payload.get("pooled_evidence_fingerprint")
        != pooled.evidence_fingerprint
        or payload.get("sample_roots_fingerprint")
        != pooled_payload.get("sample_roots_fingerprint")
        or payload.get("v24_terminal_artifact_fingerprints")
        != expected_artifacts
        or payload.get("forced_G1_terminal_artifact_fingerprints")
        != expected_artifacts
        or payload.get("forced_G1_retrained") is not False
        or _integer(
            payload.get("field_difference_count"),
            name="field_difference_count",
            minimum=0,
        )
        != len(field_rows)
        or _integer(
            payload.get("prediction_difference_count"),
            name="prediction_difference_count",
            minimum=0,
        )
        != len(prediction_rows)
        or payload.get("field_difference_ledger_fingerprint")
        != stable_fingerprint(field_rows)
        or payload.get("prediction_difference_ledger_fingerprint")
        != stable_fingerprint(prediction_rows)
        or field_rows != expected_field_rows
        or prediction_rows != expected_prediction_rows
        or stable_fingerprint(field_rows)
        != pooled_payload.get(
            "verified_field_difference_ledger_fingerprint"
        )
        or stable_fingerprint(prediction_rows)
        != pooled_payload.get(
            "verified_prediction_difference_ledger_fingerprint"
        )
        or payload.get("access_audit_receipt_fingerprints")
        != list(pooled.access_audit_receipt_fingerprints)
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("forced-G1 gate-path evidence changed")
    fingerprint = _self_fingerprint(
        payload,
        field_name="receipt_fingerprint",
        name="gate path receipt",
    )
    return _register_token(VerifiedGatePathEvidence(
        payload_json=canonical_json(payload),
        receipt_fingerprint=fingerprint,
        pooled_evidence_fingerprint=pooled.evidence_fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


def decide_oof4_pooled(
    pooled_evidence: VerifiedOOFPooledEvidence,
    *,
    gate_path_evidence: VerifiedGatePathEvidence,
) -> VerifiedOOFDecision:
    """Decide the relative OOF gate only from verifier-issued evidence."""

    pooled = _token(
        pooled_evidence,
        VerifiedOOFPooledEvidence,
        name="pooled_evidence",
    )
    mechanism = _token(
        gate_path_evidence,
        VerifiedGatePathEvidence,
        name="gate_path_evidence",
    )
    assert isinstance(pooled, VerifiedOOFPooledEvidence)
    assert isinstance(mechanism, VerifiedGatePathEvidence)
    if mechanism.pooled_evidence_fingerprint != pooled.evidence_fingerprint:
        raise ValueError("gate-path receipt belongs to different OOF evidence")
    metrics = _mapping(pooled.payload.get("arm_metrics"), name="OOF metrics")
    values = {
        arm: _mapping(metrics.get(arm), name=f"metrics[{arm}]")
        for arm in OOF_ARMS
    }
    candidate = values["GCR_PACRE_v24"]
    raw_candidate_metrics_by_fold = _mapping(
        pooled.payload.get("candidate_metrics_by_fold"),
        name="OOF candidate_metrics_by_fold",
    )
    if set(raw_candidate_metrics_by_fold) != {
        str(index) for index in range(OOF_FOLD_COUNT)
    }:
        raise ValueError("OOF candidate per-fold metric inventory changed")
    candidate_safety_by_fold = {
        str(fold_id): safety_checks(
            _mapping(
                raw_candidate_metrics_by_fold[str(fold_id)],
                name=f"OOF fold {fold_id} candidate metrics",
            )
        )
        for fold_id in range(OOF_FOLD_COUNT)
    }
    base_names = ("BaseA", "BaseB_train_fold_selected")
    base_safety = {
        name: safety_checks(values[name])
        for name in base_names
    }
    valid_base_names = tuple(
        name
        for name in base_names
        if all(base_safety[name].values())
    )
    excluded_base_names = tuple(
        name for name in base_names if name not in valid_base_names
    )
    valid_base_rows = tuple(values[name] for name in valid_base_names)
    envelope = (
        {
            "true_targets": max(
                _count(row, "true_targets") for row in valid_base_rows
            ),
            "recovered_anchor_misses": max(
                _count(row, "recovered_anchor_misses")
                for row in valid_base_rows
            ),
            "mIoU": max(
                _metric(row, "mIoU") for row in valid_base_rows
            ),
            "nIoU": max(
                _metric(row, "nIoU") for row in valid_base_rows
            ),
        }
        if valid_base_rows
        else None
    )
    control = values["PACRE_VC_v23_control"]
    forced_g1 = values["GCR_PACRE_v24_forced_G1"]
    primary_names = (
        "true_targets",
        "recovered_anchor_misses",
        "mIoU",
        "nIoU",
    )

    def primary(metric_row: Mapping[str, object], name: str) -> float | int:
        return (
            _count(metric_row, name)
            if name in {"true_targets", "recovered_anchor_misses"}
            else _metric(metric_row, name)
        )

    checks = {
        "at_least_one_safety_valid_Base_row": envelope is not None,
        "strict_true_targets_above_Base_envelope": (
            envelope is not None
            and _count(candidate, "true_targets") > envelope["true_targets"]
        ),
        "strict_recovery_above_Base_envelope": (
            envelope is not None
            and _count(candidate, "recovered_anchor_misses")
            > envelope["recovered_anchor_misses"]
        ),
        "mIoU_not_below_Base_envelope": (
            envelope is not None
            and _metric(candidate, "mIoU") >= envelope["mIoU"]
        ),
        "nIoU_not_below_Base_envelope": (
            envelope is not None
            and _metric(candidate, "nIoU") >= envelope["nIoU"]
        ),
        "strict_true_targets_above_PACRE_VC_v23_control": (
            _count(candidate, "true_targets")
            > _count(control, "true_targets")
        ),
        "strict_recovery_above_PACRE_VC_v23_control": (
            _count(candidate, "recovered_anchor_misses")
            > _count(control, "recovered_anchor_misses")
        ),
        "mIoU_not_below_PACRE_VC_v23_control": (
            _metric(candidate, "mIoU") >= _metric(control, "mIoU")
        ),
        "nIoU_not_below_PACRE_VC_v23_control": (
            _metric(candidate, "nIoU") >= _metric(control, "nIoU")
        ),
        "no_primary_regression_vs_forced_G1": all(
            primary(candidate, name) >= primary(forced_g1, name)
            for name in primary_names
        ),
        "at_least_one_primary_improvement_vs_forced_G1": any(
            primary(candidate, name) > primary(forced_g1, name)
            for name in primary_names
        ),
        "verified_nontrivial_same_terminal_gate_path": (
            mechanism.pooled_evidence_fingerprint
            == pooled.evidence_fingerprint
            and mechanism.payload.get("field_difference_count", 0) >= 1
            and mechanism.payload.get("prediction_difference_count", 0)
            >= 1
        ),
        **{
            f"candidate_{name}": passed
            for name, passed in safety_checks(candidate).items()
        },
        **{
            f"fold_{fold_id}_candidate_{name}": passed
            for fold_id, fold_checks in candidate_safety_by_fold.items()
            for name, passed in fold_checks.items()
        },
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    body = {
        "schema_version": "cure-lite-v24-gcr-pacre-oof4-decision-v3",
        "pooled_evidence_fingerprint": pooled.evidence_fingerprint,
        "gate_path_receipt_fingerprint": mechanism.receipt_fingerprint,
        "comparison": "pooled_factual_only_sufficient_statistics",
        "Base_validity": {
            name: {
                "safety_checks": base_safety[name],
                "valid": name in valid_base_names,
                "used_in_envelope": name in valid_base_names,
            }
            for name in base_names
        },
        "Base_envelope_used_arms": list(valid_base_names),
        "Base_envelope_excluded_arms": list(excluded_base_names),
        "Base_envelope": envelope,
        "per_fold_candidate_safety": candidate_safety_by_fold,
        "checks": checks,
        "failed_checks": failed,
        "gate_passed": not failed,
        "per_fold_performance_vote_used": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    fingerprint = stable_fingerprint(body)
    payload = {**body, "decision_fingerprint": fingerprint}
    return _register_token(VerifiedOOFDecision(
        payload_json=canonical_json(payload),
        decision_fingerprint=fingerprint,
        pooled_evidence_fingerprint=pooled.evidence_fingerprint,
        gate_path_receipt_fingerprint=mechanism.receipt_fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


def _validate_finite_audit(
    raw: object,
    *,
    updates: int,
    name: str,
) -> str:
    value = _mapping(raw, name=name)
    _exact_keys(
        value,
        {
            "schema_version",
            "expected_updates",
            "loss_values_checked",
            "gradient_tensors_checked",
            "parameter_tensors_checked",
            "nonfinite_values",
            "audit_fingerprint",
        },
        name=name,
    )
    body = dict(value)
    fingerprint = body.pop("audit_fingerprint", None)
    if (
        value.get("schema_version")
        != "cure-lite-v24-training-finite-audit-v1"
        or value.get("expected_updates") != updates
        or value.get("loss_values_checked") != updates
        or value.get("gradient_tensors_checked") != updates * 3
        or value.get("parameter_tensors_checked") != (updates + 1) * 3
        or value.get("nonfinite_values") != 0
        or fingerprint != stable_fingerprint(body)
    ):
        raise ValueError(f"{name} is incomplete or non-finite")
    return _sha256(fingerprint, name=f"{name}.audit_fingerprint")


def _validate_schedule_artifact(
    raw: object,
    *,
    expected_schedule_fingerprint: str,
    name: str,
) -> tuple[str, os.stat_result]:
    value = _mapping(raw, name=name)
    _exact_keys(
        value,
        {"path", "size_bytes", "file_sha256", "schedule_fingerprint"},
        name=name,
    )
    schedule_fp = _sha256(
        value.get("schedule_fingerprint"),
        name=f"{name}.schedule_fingerprint",
    )
    if schedule_fp != expected_schedule_fingerprint:
        raise ValueError(f"{name} semantic schedule changed")
    path, stat_result = _verify_regular_file(
        value.get("path"),
        name=name,
        expected_sha256=value.get("file_sha256"),
        expected_size=value.get("size_bytes"),
    )
    return str(path), stat_result


def _formal_schedule_policy_without_seed(
    semantic_cache_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": (
            "cure-lite-v24-formal800-schedule-policy-without-seed-v1"
        ),
        "semantic_cache_fingerprint": semantic_cache_fingerprint,
        "epochs": FORMAL_EPOCHS,
        "steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
        "updates": FORMAL_UPDATES,
        "logical_states_per_update": 12,
        "objective_invariant": True,
        "optimizer_exposure_accounting": (
            "recomputed_against_current_cache_before_use"
        ),
    }


def _validate_formal_schedule_artifact(
    raw: object,
    *,
    expected_schedule_fingerprint: str,
    expected_semantic_cache_fingerprint: str,
    expected_seed: int,
    name: str,
) -> tuple[str, os.stat_result, str]:
    value = _mapping(raw, name=name)
    _exact_keys(
        value,
        {
            "path",
            "size_bytes",
            "file_sha256",
            "schedule_fingerprint",
            "seed",
            "epochs",
            "steps_per_epoch",
            "updates",
            "semantic_cache_fingerprint",
            "policy_without_seed_fingerprint",
        },
        name=name,
    )
    schedule_fp = _sha256(
        value.get("schedule_fingerprint"),
        name=f"{name}.schedule_fingerprint",
    )
    semantic_cache_fp = _sha256(
        value.get("semantic_cache_fingerprint"),
        name=f"{name}.semantic_cache_fingerprint",
    )
    expected_policy = _formal_schedule_policy_without_seed(
        expected_semantic_cache_fingerprint
    )
    expected_policy_fp = stable_fingerprint(expected_policy)
    if (
        schedule_fp != expected_schedule_fingerprint
        or semantic_cache_fp != expected_semantic_cache_fingerprint
        or value.get("seed") != expected_seed
        or value.get("epochs") != FORMAL_EPOCHS
        or value.get("steps_per_epoch") != FORMAL_STEPS_PER_EPOCH
        or value.get("updates") != FORMAL_UPDATES
        or value.get("policy_without_seed_fingerprint")
        != expected_policy_fp
    ):
        raise ValueError(f"{name} semantic schedule policy changed")
    path, stat_result = _verify_regular_file(
        value.get("path"),
        name=name,
        expected_sha256=value.get("file_sha256"),
        expected_size=value.get("size_bytes"),
    )
    return str(path), stat_result, expected_policy_fp


_BOUNDED_METRIC_FIELDS = {
    "true_targets",
    "recovered_anchor_misses",
    "mIoU",
    "nIoU",
    "pd",
    "retention",
    "pixel_fa",
    "raw_background_fa",
    "fp_components_per_mp",
    "budget_violation",
    "initial_PMOPE",
    "terminal_PMOPE",
    "terminal_target_role_violation",
    "terminal_background_role_violation",
    "terminal_zero_crossed_target_states",
    "terminal_false_completion_states",
    "terminal_field_fingerprint",
    "terminal_role_prediction_fingerprint",
    "G1_PMOPE",
    "G1_target_role_violation",
    "G1_background_role_violation",
    "G1_zero_crossed_target_states",
    "G1_false_completion_states",
    "G1_field_fingerprint",
    "G1_role_prediction_fingerprint",
    "terminal_gate_distribution",
    "G1_gate_distribution",
    "gate_role_distributions_present",
}
_BOUNDED_ARM_FIELDS = {
    "role",
    "seed",
    "epochs",
    "steps_per_epoch",
    "completed_updates",
    "training_invocations",
    "from_scratch",
    "resume_allowed",
    "automatic_retry_allowed",
    "checkpoint_policy",
    "optimizer_state_initial_empty",
    "population_fingerprint",
    "schedule_fingerprint",
    "batch_sequence_fingerprint",
    "initial_shared_parameter_fingerprint",
    "PMOPE_fingerprint",
    "Adam_policy_fingerprint",
    "dtype_device_policy_fingerprint",
    "source_hashes",
    "cache_fingerprint",
    "neutral_payload_fingerprint",
    "cache_instance_id",
    "rng_instance_id",
    "module_instance_id",
    "optimizer_instance_id",
    "parameter_storage_ids",
    "initial_model_fingerprint",
    "final_model_fingerprint",
    "terminal_artifact",
    "finite_audit",
    "metrics",
}

_BOUNDED_DIAGNOSTIC_DELTA_FIELDS = {
    "PMOPE",
    "target_role_violation",
    "background_role_violation",
    "zero_crossed_target_states",
    "false_completion_states",
}
_BOUNDED_UPDATE_DIAGNOSTIC_FIELDS = {
    "update",
    "selection_fingerprint",
    "control_loss",
    "candidate_loss",
    "candidate_minus_control_loss",
    "control_gradient_l2_norm",
    "candidate_gradient_l2_norm",
    "candidate_minus_control_gradient_l2_norm",
}


def _signed_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _validate_gate_role_distribution(
    raw: object,
    *,
    name: str,
    required: bool,
) -> None:
    if not required:
        if raw is not None:
            raise ValueError(f"{name} must be exact null for the v23 arm")
        return
    value = _mapping(raw, name=name)
    _exact_keys(
        value,
        {
            "schema_version",
            "endpoint_counts",
            "target_G",
            "background_G",
            "target_E",
            "background_E",
        },
        name=name,
    )
    if (
        value.get("schema_version")
        != "cure-lite-v24-gcr-pacre-gate-role-summary-v1"
    ):
        raise ValueError(f"{name} schema changed")
    endpoint = _mapping(
        value.get("endpoint_counts"),
        name=f"{name}.endpoint_counts",
    )
    _exact_keys(
        endpoint,
        {"G_equal_0", "G_equal_2", "G_strict_interior"},
        name=f"{name}.endpoint_counts",
    )
    endpoint_total = sum(
        _integer(endpoint.get(key), name=f"{name}.{key}")
        for key in ("G_equal_0", "G_equal_2", "G_strict_interior")
    )
    if required and _integer(
        endpoint.get("G_strict_interior"),
        name=f"{name}.G_strict_interior",
    ) < 1:
        raise PermissionError(
            f"{name} violates the preregistered strict-interior gate "
            "requirement"
        )

    summaries: dict[str, tuple[int, float, float, float]] = {}
    for summary_name in (
        "target_G",
        "background_G",
        "target_E",
        "background_E",
    ):
        summary = _mapping(
            value.get(summary_name),
            name=f"{name}.{summary_name}",
        )
        _exact_keys(
            summary,
            {"count", "minimum", "maximum", "mean"},
            name=f"{name}.{summary_name}",
        )
        count = _integer(
            summary.get("count"),
            name=f"{name}.{summary_name}.count",
            minimum=1,
        )
        minimum = _real(
            summary.get("minimum"),
            name=f"{name}.{summary_name}.minimum",
        )
        maximum = _real(
            summary.get("maximum"),
            name=f"{name}.{summary_name}.maximum",
        )
        mean = _real(
            summary.get("mean"),
            name=f"{name}.{summary_name}.mean",
        )
        if not minimum <= mean <= maximum:
            raise ValueError(f"{name}.{summary_name} ordering changed")
        if summary_name.endswith("_G") and not (
            0.0 <= minimum <= maximum <= 2.0
        ):
            raise ValueError(f"{name}.{summary_name} left G in [0,2]")
        summaries[summary_name] = (count, minimum, maximum, mean)
    if (
        summaries["target_G"][0] != summaries["target_E"][0]
        or summaries["background_G"][0]
        != summaries["background_E"][0]
        or endpoint_total
        != summaries["target_G"][0] + summaries["background_G"][0]
    ):
        raise ValueError(f"{name} role/endpoint counts are inconsistent")


def _bounded_metric_delta(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    left_prefix: str,
    right_prefix: str,
) -> dict[str, int | float]:
    return {
        "PMOPE": (
            _real(left.get(f"{left_prefix}PMOPE"), name="left PMOPE")
            - _real(right.get(f"{right_prefix}PMOPE"), name="right PMOPE")
        ),
        "target_role_violation": (
            _real(
                left.get(f"{left_prefix}target_role_violation"),
                name="left target_role_violation",
            )
            - _real(
                right.get(f"{right_prefix}target_role_violation"),
                name="right target_role_violation",
            )
        ),
        "background_role_violation": (
            _real(
                left.get(f"{left_prefix}background_role_violation"),
                name="left background_role_violation",
            )
            - _real(
                right.get(f"{right_prefix}background_role_violation"),
                name="right background_role_violation",
            )
        ),
        "zero_crossed_target_states": (
            _integer(
                left.get(f"{left_prefix}zero_crossed_target_states"),
                name="left zero_crossed_target_states",
            )
            - _integer(
                right.get(f"{right_prefix}zero_crossed_target_states"),
                name="right zero_crossed_target_states",
            )
        ),
        "false_completion_states": (
            _integer(
                left.get(f"{left_prefix}false_completion_states"),
                name="left false_completion_states",
            )
            - _integer(
                right.get(f"{right_prefix}false_completion_states"),
                name="right false_completion_states",
            )
        ),
    }


def _validate_bounded_delta(
    raw: object,
    *,
    expected: Mapping[str, int | float],
    name: str,
) -> None:
    value = _mapping(raw, name=name)
    _exact_keys(value, _BOUNDED_DIAGNOSTIC_DELTA_FIELDS, name=name)
    normalized = {
        "PMOPE": _real(value.get("PMOPE"), name=f"{name}.PMOPE"),
        "target_role_violation": _real(
            value.get("target_role_violation"),
            name=f"{name}.target_role_violation",
        ),
        "background_role_violation": _real(
            value.get("background_role_violation"),
            name=f"{name}.background_role_violation",
        ),
        "zero_crossed_target_states": _signed_integer(
            value.get("zero_crossed_target_states"),
            name=f"{name}.zero_crossed_target_states",
        ),
        "false_completion_states": _signed_integer(
            value.get("false_completion_states"),
            name=f"{name}.false_completion_states",
        ),
    }
    if normalized != dict(expected):
        raise ValueError(f"{name} is not the exact terminal metric difference")


def _validate_paired_bounded_diagnostics(
    raw: object,
    *,
    arms: Mapping[str, object],
) -> tuple[str, tuple[dict[str, object], ...]]:
    value = _mapping(raw, name="paired_diagnostics")
    _exact_keys(
        value,
        {
            "interpretation",
            "candidate_minus_control",
            "candidate_minus_same_weight_G1",
            "per_update_fingerprint",
            "per_update",
        },
        name="paired_diagnostics",
    )
    if value.get("interpretation") != (
        "paired_deltas_are_diagnostic_only_without_a_fixed_threshold"
    ):
        raise PermissionError(
            "bounded paired deltas acquired a fixed gate interpretation"
        )
    control_arm = _mapping(
        arms.get("PACRE_VC_v23_control"),
        name="bounded control arm",
    )
    candidate_arm = _mapping(
        arms.get("GCR_PACRE_v24"),
        name="bounded candidate arm",
    )
    control = _mapping(control_arm.get("metrics"), name="control metrics")
    candidate = _mapping(
        candidate_arm.get("metrics"),
        name="candidate metrics",
    )
    _validate_bounded_delta(
        value.get("candidate_minus_control"),
        expected=_bounded_metric_delta(
            candidate,
            control,
            left_prefix="terminal_",
            right_prefix="terminal_",
        ),
        name="candidate_minus_control",
    )
    candidate_g1 = _mapping(
        value.get("candidate_minus_same_weight_G1"),
        name="candidate_minus_same_weight_G1",
    )
    _exact_keys(
        candidate_g1,
        _BOUNDED_DIAGNOSTIC_DELTA_FIELDS
        | {
            "field_nonidentity_witness",
            "role_prediction_nonidentity_witness",
        },
        name="candidate_minus_same_weight_G1",
    )
    _validate_bounded_delta(
        {
            key: candidate_g1[key]
            for key in _BOUNDED_DIAGNOSTIC_DELTA_FIELDS
        },
        expected=_bounded_metric_delta(
            candidate,
            candidate,
            left_prefix="terminal_",
            right_prefix="G1_",
        ),
        name="candidate_minus_same_weight_G1 deltas",
    )
    terminal_field = _sha256(
        candidate.get("terminal_field_fingerprint"),
        name="candidate terminal_field_fingerprint",
    )
    g1_field = _sha256(
        candidate.get("G1_field_fingerprint"),
        name="candidate G1_field_fingerprint",
    )
    terminal_role = _sha256(
        candidate.get("terminal_role_prediction_fingerprint"),
        name="candidate terminal_role_prediction_fingerprint",
    )
    g1_role = _sha256(
        candidate.get("G1_role_prediction_fingerprint"),
        name="candidate G1_role_prediction_fingerprint",
    )
    if (
        candidate_g1.get("field_nonidentity_witness") is not True
        or candidate_g1.get("role_prediction_nonidentity_witness") is not True
        or terminal_field == g1_field
        or terminal_role == g1_role
    ):
        raise PermissionError(
            "candidate/same-weight-G1 nonidentity witnesses are absent"
        )
    rows = value.get("per_update")
    if not isinstance(rows, list) or len(rows) != BOUNDED_UPDATES:
        raise ValueError(
            "paired_diagnostics.per_update must contain exactly 400 rows"
        )
    normalized_rows: list[dict[str, object]] = []
    for expected_update, raw_row in enumerate(rows):
        row = _mapping(
            raw_row,
            name=f"paired_diagnostics.per_update[{expected_update}]",
        )
        _exact_keys(
            row,
            _BOUNDED_UPDATE_DIAGNOSTIC_FIELDS,
            name=f"paired_diagnostics.per_update[{expected_update}]",
        )
        update = _integer(
            row.get("update"),
            name=f"per_update[{expected_update}].update",
        )
        selection_fp = _sha256(
            row.get("selection_fingerprint"),
            name=f"per_update[{expected_update}].selection_fingerprint",
        )
        control_loss = _real(
            row.get("control_loss"),
            name=f"per_update[{expected_update}].control_loss",
            minimum=0.0,
        )
        candidate_loss = _real(
            row.get("candidate_loss"),
            name=f"per_update[{expected_update}].candidate_loss",
            minimum=0.0,
        )
        loss_delta = _real(
            row.get("candidate_minus_control_loss"),
            name=f"per_update[{expected_update}].loss_delta",
        )
        control_gradient = _real(
            row.get("control_gradient_l2_norm"),
            name=f"per_update[{expected_update}].control_gradient",
            minimum=0.0,
        )
        candidate_gradient = _real(
            row.get("candidate_gradient_l2_norm"),
            name=f"per_update[{expected_update}].candidate_gradient",
            minimum=0.0,
        )
        gradient_delta = _real(
            row.get("candidate_minus_control_gradient_l2_norm"),
            name=f"per_update[{expected_update}].gradient_delta",
        )
        if (
            update != expected_update
            or loss_delta != candidate_loss - control_loss
            or gradient_delta != candidate_gradient - control_gradient
        ):
            raise ValueError(
                "paired per-update ordering or exact deltas changed"
            )
        normalized_rows.append(
            {
                "update": update,
                "selection_fingerprint": selection_fp,
                "control_loss": control_loss,
                "candidate_loss": candidate_loss,
                "candidate_minus_control_loss": loss_delta,
                "control_gradient_l2_norm": control_gradient,
                "candidate_gradient_l2_norm": candidate_gradient,
                "candidate_minus_control_gradient_l2_norm": gradient_delta,
            }
        )
    ledger_fp = stable_fingerprint(normalized_rows)
    if _sha256(
        value.get("per_update_fingerprint"),
        name="paired_diagnostics.per_update_fingerprint",
    ) != ledger_fp:
        raise ValueError("paired per-update diagnostic fingerprint changed")
    return ledger_fp, tuple(normalized_rows)


def _validate_bounded_diagnostics_trace_binding(
    diagnostic_rows: Sequence[Mapping[str, object]],
    trace_payload: Mapping[str, object],
) -> None:
    """Bind the diagnostic ledger to the persisted per-step state trace."""

    raw_trace_rows = trace_payload.get("rows")
    if (
        not isinstance(raw_trace_rows, list)
        or len(diagnostic_rows) != BOUNDED_UPDATES
        or len(raw_trace_rows) != BOUNDED_UPDATES
    ):
        raise ValueError(
            "bounded diagnostic/trace update inventories changed"
        )
    for update, (diagnostic, raw_trace) in enumerate(
        zip(diagnostic_rows, raw_trace_rows, strict=True)
    ):
        trace = _mapping(
            raw_trace,
            name=f"bounded training trace row {update}",
        )
        trace_arms = _mapping(
            trace.get("arms"),
            name=f"bounded training trace row {update}.arms",
        )
        control = _mapping(
            trace_arms.get("PACRE_VC_v23_control"),
            name=f"bounded training trace row {update}.control",
        )
        candidate = _mapping(
            trace_arms.get("GCR_PACRE_v24"),
            name=f"bounded training trace row {update}.candidate",
        )
        expected = {
            "update": update,
            "selection_fingerprint": trace.get(
                "selection_fingerprint"
            ),
            "control_loss": control.get("loss"),
            "candidate_loss": candidate.get("loss"),
            "candidate_minus_control_loss": (
                float(candidate["loss"]) - float(control["loss"])
            ),
            "control_gradient_l2_norm": control.get(
                "gradient_l2_norm"
            ),
            "candidate_gradient_l2_norm": candidate.get(
                "gradient_l2_norm"
            ),
            "candidate_minus_control_gradient_l2_norm": (
                float(candidate["gradient_l2_norm"])
                - float(control["gradient_l2_norm"])
            ),
        }
        if dict(diagnostic) != expected:
            raise PermissionError(
                f"bounded diagnostic row {update} differs from the "
                "persisted training trace"
            )


def _validate_gcr_pacre_persistent_run_start(
    artifact: object,
    *,
    name: str,
    expected_schema: str,
    expected_path: Path,
    expected_stage_id: str,
    expected_access_fingerprint: str,
    repository_root: Path,
) -> tuple[dict[str, object], os.stat_result]:
    """Re-read and validate one immutable bounded/Formal attempt marker."""

    wrapper = _mapping(artifact, name=name)
    _exact_keys(
        wrapper,
        {
            "path",
            "size_bytes",
            "file_sha256",
            "device",
            "inode",
            "hardlink_count",
            "marker_fingerprint",
            "payload",
        },
        name=name,
    )
    path, stat_result = _verify_regular_file(
        wrapper.get("path"),
        name=name,
        expected_sha256=wrapper.get("file_sha256"),
        expected_size=wrapper.get("size_bytes"),
        expected_device=wrapper.get("device"),
        expected_inode=wrapper.get("inode"),
    )
    if (
        path != expected_path
        or _integer(
            wrapper.get("hardlink_count"),
            name=f"{name}.hardlink_count",
        )
        != 1
        or stat_result.st_nlink != 1
        or stat_result.st_mode & 0o222
    ):
        raise PermissionError(f"{name} physical/path policy changed")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{name} must be newline-terminated canonical JSON")
    try:
        stored = json.loads(raw[:-1].decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} JSON is invalid") from error
    payload = _mapping(wrapper.get("payload"), name=f"{name}.payload")
    if (
        not isinstance(stored, dict)
        or canonical_json(stored) != raw[:-1].decode("utf-8")
        or stored != payload
    ):
        raise PermissionError(f"{name} embedded payload differs from bytes")
    body = dict(payload)
    marker_fp = body.pop("marker_fingerprint", None)
    if (
        payload.get("schema_version") != expected_schema
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("marker_path") != str(expected_path)
        or payload.get("stage_id") != expected_stage_id
        or payload.get("access_audit_receipt_fingerprint")
        != expected_access_fingerprint
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
        or _sha256(
            marker_fp,
            name=f"{name}.payload.marker_fingerprint",
        )
        != stable_fingerprint(body)
        or wrapper.get("marker_fingerprint") != marker_fp
    ):
        raise PermissionError(f"{name} logical identity changed")
    chain = _mapping(
        payload.get("chain_config"),
        name=f"{name}.chain_config",
    )
    _exact_keys(
        chain,
        {"path", "file_sha256", "config_fingerprint"},
        name=f"{name}.chain_config",
    )
    chain_path, chain_stat = _verify_regular_file(
        chain.get("path"),
        name=f"{name}.chain_config",
        expected_sha256=chain.get("file_sha256"),
    )
    if chain_stat.st_mode & 0o222:
        raise PermissionError(f"{name} chain config is writable")
    chain_raw = chain_path.read_bytes()
    if not chain_raw.endswith(b"\n"):
        raise ValueError(f"{name} chain config is not canonical")
    try:
        chain_payload = json.loads(
            chain_raw[:-1].decode("utf-8", errors="strict")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} chain config JSON is invalid") from error
    if (
        not isinstance(chain_payload, dict)
        or canonical_json(chain_payload)
        != chain_raw[:-1].decode("utf-8")
    ):
        raise ValueError(f"{name} chain config is not canonical")
    chain_body = dict(chain_payload)
    chain_fp = chain_body.pop("config_fingerprint", None)
    if (
        _sha256(
            chain.get("config_fingerprint"),
            name=f"{name}.chain_config.config_fingerprint",
        )
        != chain_fp
        or chain_fp != stable_fingerprint(chain_body)
        or chain_payload.get("repository_root") != str(repository_root)
        or chain_payload.get("chain_config_path") != str(chain_path)
        or chain_payload.get("D_V_payload_accessed") is not False
        or chain_payload.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError(f"{name} chain config binding changed")
    source = _mapping(
        payload.get("source_closure"),
        name=f"{name}.source_closure",
    )
    _exact_keys(
        source,
        {"schema_version", "fingerprint", "source_hashes"},
        name=f"{name}.source_closure",
    )
    from cure_lite_v24.source_closure import (
        GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
        GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
        gcr_pacre_v24_source_closure_fingerprint,
    )

    source_rows = _verify_repo_source_hashes(
        source.get("source_hashes"),
        repository_root=repository_root,
        name=f"{name}.source_closure.source_hashes",
        exact_paths=GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
    )
    expected_source_fp = gcr_pacre_v24_source_closure_fingerprint(
        source_rows
    )
    if (
        source.get("schema_version")
        != GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA
        or source.get("fingerprint") != expected_source_fp
        or chain_payload.get("source_hashes") != dict(source_rows)
        or chain_payload.get("source_closure_fingerprint")
        != expected_source_fp
    ):
        raise RuntimeError(f"{name} unified source closure changed")
    return payload, stat_result


def _gcr_pacre_v24_evidence_runtime_root(
    repository_root: Path,
) -> Path:
    return (
        repository_root
        / "runs/irstd1k_stage_a_seed42/gcr_pacre_v24_evidence_r1"
    )


def _validate_current_authorization_artifact(
    path: Path,
    *,
    expected_authorization_fingerprint: str,
    name: str,
) -> dict[str, object]:
    """Bind a run-start marker to the current immutable authorization bytes."""

    from cure_lite_v24.artifact_io import read_canonical_json

    expected_fp = _sha256(
        expected_authorization_fingerprint,
        name=f"{name}.expected_authorization_fingerprint",
    )
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or path.stat().st_nlink != 1
        or path.stat().st_mode & 0o222
    ):
        raise PermissionError(
            f"{name} authorization artifact is not immutable/canonical"
        )
    payload = read_canonical_json(path)
    if stable_fingerprint(payload) != expected_fp:
        raise PermissionError(
            f"{name} run-start authorization_fingerprint differs from "
            "the current authorization artifact"
        )
    return payload


def validate_paired_bounded_receipt(
    receipt: Mapping[str, object],
    *,
    oof_decision: VerifiedOOFDecision,
    access_audit: VerifiedAccessAudit,
    full_d_r_cache_artifact: object,
    dataset_free_receipt_fingerprint: str,
    d_r_structural_receipt_fingerprint: str,
    repository_root: str | Path,
) -> VerifiedBoundedEvidence:
    """Validate the exact paired 10×40 smoke evidence and its prerequisites."""

    oof = _token(oof_decision, VerifiedOOFDecision, name="oof_decision")
    access = _token(access_audit, VerifiedAccessAudit, name="access_audit")
    assert isinstance(oof, VerifiedOOFDecision)
    assert isinstance(access, VerifiedAccessAudit)
    if oof.payload.get("gate_passed") is not True:
        raise PermissionError("bounded stage requires a passed verified OOF gate")
    dataset_free_fp = _sha256(
        dataset_free_receipt_fingerprint,
        name="dataset_free_receipt_fingerprint",
    )
    structural_fp = _sha256(
        d_r_structural_receipt_fingerprint,
        name="d_r_structural_receipt_fingerprint",
    )
    repository = Path(repository_root).resolve(strict=True)
    payload = _mapping(receipt, name="paired bounded receipt")
    _exact_keys(
        payload,
        {
            "schema_version",
            "budget",
            "prerequisites",
            "access_audit_receipt_fingerprint",
            "paired_population_fingerprint",
            "full_D_R_cache_materialization",
            "run_start_artifact",
            "schedule_artifact",
            "training_trace_artifact",
            "arms",
            "paired_diagnostics",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "receipt_fingerprint",
        },
        name="paired bounded receipt",
    )
    if (
        payload.get("schema_version")
        != "cure-lite-v24-gcr-pacre-paired-bounded400-receipt-v6"
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
        or access.stage_id != "paired_bounded400"
        or access.allowed_splits != ("D_R",)
        or payload.get("access_audit_receipt_fingerprint")
        != access.receipt_fingerprint
    ):
        raise PermissionError("bounded stage split/firewall identity changed")
    budget = _mapping(payload.get("budget"), name="bounded budget")
    if budget != {
        "epochs": BOUNDED_EPOCHS,
        "steps_per_epoch": BOUNDED_STEPS_PER_EPOCH,
        "updates": BOUNDED_UPDATES,
        "training_invocations_per_arm": 1,
    }:
        raise ValueError("bounded budget must be exactly 10×40 once per arm")
    prerequisites = _mapping(
        payload.get("prerequisites"),
        name="bounded prerequisites",
    )
    _exact_keys(
        prerequisites,
        {
            "dataset_free_receipt_fingerprint",
            "D_R_structural_receipt_fingerprint",
            "OOF4_decision_fingerprint",
        },
        name="bounded prerequisites",
    )
    if prerequisites != {
        "dataset_free_receipt_fingerprint": dataset_free_fp,
        "D_R_structural_receipt_fingerprint": structural_fp,
        "OOF4_decision_fingerprint": oof.decision_fingerprint,
    }:
        raise PermissionError("bounded prerequisites changed")
    population_fp = _sha256(
        payload.get("paired_population_fingerprint"),
        name="paired_population_fingerprint",
    )
    from cure_lite_v24.formal_cache_artifacts import (
        require_verified_formal_cache_artifact,
        verify_formal_cache_artifact,
    )

    supplied_full_cache = require_verified_formal_cache_artifact(
        full_d_r_cache_artifact
    )
    expected_full_cache_id = (
        "paired-bounded400-full-D_R-materialization"
    )
    if supplied_full_cache.cache_id != expected_full_cache_id:
        raise PermissionError("bounded full-D_R cache token identity changed")
    verified_full_cache = verify_formal_cache_artifact(
        supplied_full_cache.path,
        cache_id=expected_full_cache_id,
        expected_semantic_cache_fingerprint=(
            supplied_full_cache.semantic_cache_fingerprint
        ),
    )
    if (
        supplied_full_cache.receipt_fingerprint
        != verified_full_cache.receipt_fingerprint
        or payload.get("full_D_R_cache_materialization")
        != verified_full_cache.payload
    ):
        raise PermissionError(
            "bounded receipt lacks verified full-D_R materialization"
        )
    run_start, _ = _validate_gcr_pacre_persistent_run_start(
        payload.get("run_start_artifact"),
        name="bounded run_start_artifact",
        expected_schema=(
            "cure-lite-v24-gcr-pacre-paired-bounded400-"
            "persistent-run-start-v1"
        ),
        expected_path=(
            _gcr_pacre_v24_evidence_runtime_root(repository)
            / "bounded/paired_bounded400/run_start.json"
        ),
        expected_stage_id="paired_bounded400",
        expected_access_fingerprint=access.receipt_fingerprint,
        repository_root=repository,
    )
    _exact_keys(
        run_start,
        {
            "schema_version",
            "protocol_id",
            "path_policy",
            "marker_path",
            "stage_id",
            "chain_config",
            "authorization_fingerprint",
            "OOF4_decision_fingerprint",
            "access_audit_receipt_fingerprint",
            "full_D_R_cache_artifact",
            "source_closure",
            "intent",
            "intent_fingerprint",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "marker_fingerprint",
        },
        name="bounded run-start payload",
    )
    run_cache = _mapping(
        run_start.get("full_D_R_cache_artifact"),
        name="bounded run-start cache",
    )
    intent = _mapping(
        run_start.get("intent"),
        name="bounded run-start intent",
    )
    bounded_runtime = _gcr_pacre_v24_evidence_runtime_root(repository)
    marker_chain = _mapping(
        run_start.get("chain_config"),
        name="bounded run-start chain_config",
    )
    authorization_fp = _sha256(
        run_start.get("authorization_fingerprint"),
        name="bounded run-start authorization_fingerprint",
    )
    if (
        run_start.get("path_policy")
        != "fixed_runtime_root_bounded_paired_bounded400_run_start_json_v1"
        or marker_chain.get("path")
        != str(bounded_runtime / "bounded/execution_chain_config.json")
        or run_start.get("OOF4_decision_fingerprint")
        != oof.decision_fingerprint
        or run_cache
        != {
            "receipt_fingerprint": (
                verified_full_cache.receipt_fingerprint
            ),
            "cache_id": verified_full_cache.cache_id,
            "path": verified_full_cache.path,
            "file_sha256": verified_full_cache.file_sha256,
            "device": verified_full_cache.device,
            "inode": verified_full_cache.inode,
            "hardlink_count": verified_full_cache.hardlink_count,
            "semantic_cache_fingerprint": (
                verified_full_cache.semantic_cache_fingerprint
            ),
            "neutral_payload_fingerprint": (
                verified_full_cache.neutral_payload_fingerprint
            ),
        }
        or intent
        != {
            "execution_kind": "paired_bounded400_D_R_training",
            "split": "D_R",
            "requested_device": intent.get("requested_device"),
            "output_directory": str(
                bounded_runtime / "bounded/paired_bounded400"
            ),
            "seed": 42,
            "epochs": BOUNDED_EPOCHS,
            "steps_per_epoch": BOUNDED_STEPS_PER_EPOCH,
            "optimizer_steps_authorized_per_arm": BOUNDED_UPDATES,
            "parameter_updates_authorized_per_arm": BOUNDED_UPDATES,
            "training_invocations_authorized_per_arm": 1,
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_V_materialization_intended": False,
            "D_T_materialization_intended": False,
        }
        or not isinstance(intent.get("requested_device"), str)
        or not intent.get("requested_device")
        or run_start.get("intent_fingerprint")
        != stable_fingerprint(intent)
    ):
        raise PermissionError("bounded persistent run-start binding changed")
    _validate_current_authorization_artifact(
        bounded_runtime
        / "bounded/paired_bounded400/authorization.json",
        expected_authorization_fingerprint=authorization_fp,
        name="bounded",
    )
    observed_full_cache = {
        (
            str(row["logical_id"]),
            str(row["source_fingerprint"]),
            str(row["purpose"]),
        )
        for row in access.observed_payloads
        if row.get("split") == "D_R"
    }
    if (
        expected_full_cache_id,
        verified_full_cache.file_sha256,
        "paired_bounded400_full_D_R_materialization",
    ) not in observed_full_cache:
        raise PermissionError(
            "bounded full-D_R materialization lacks access evidence"
        )
    arms = _mapping(payload.get("arms"), name="bounded arms")
    arm_names = ("PACRE_VC_v23_control", "GCR_PACRE_v24")
    if set(arms) != set(arm_names):
        raise ValueError("bounded arm inventory changed")
    paired_names = (
        "seed",
        "population_fingerprint",
        "schedule_fingerprint",
        "batch_sequence_fingerprint",
        "initial_shared_parameter_fingerprint",
        "PMOPE_fingerprint",
        "Adam_policy_fingerprint",
        "dtype_device_policy_fingerprint",
    )
    paired_values: dict[str, set[object]] = defaultdict(set)
    modules: set[str] = set()
    optimizers: set[str] = set()
    caches: set[str] = set()
    rngs: set[str] = set()
    storage_by_arm: dict[str, set[str]] = {}
    source_hashes_by_arm: dict[str, dict[str, object]] = {}
    artifact_physical: set[tuple[int, int]] = set()
    claimed_finite_audits: dict[str, dict[str, object]] = {}
    terminal_model_fingerprints: dict[str, str] = {}
    schedule_fp: str | None = None
    from cure_lite_v24.source_closure import (
        GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
    )

    for arm, raw in arms.items():
        value = _mapping(raw, name=f"bounded arms[{arm}]")
        _exact_keys(value, _BOUNDED_ARM_FIELDS, name=f"bounded arm {arm}")
        expected_role = (
            "control" if arm == "PACRE_VC_v23_control" else "candidate"
        )
        if (
            value.get("role") != expected_role
            or value.get("seed") != OOF_SEED
            or value.get("epochs") != BOUNDED_EPOCHS
            or value.get("steps_per_epoch") != BOUNDED_STEPS_PER_EPOCH
            or value.get("completed_updates") != BOUNDED_UPDATES
            or value.get("training_invocations") != 1
            or value.get("from_scratch") is not True
            or value.get("resume_allowed") is not False
            or value.get("automatic_retry_allowed") is not False
            or value.get("checkpoint_policy") != "final_only"
            or value.get("optimizer_state_initial_empty") is not True
            or value.get("population_fingerprint") != population_fp
        ):
            raise ValueError(f"bounded arm {arm} execution identity changed")
        for name in paired_names:
            raw_value = value.get(name)
            if name != "seed":
                _sha256(raw_value, name=f"{arm}.{name}")
            paired_values[name].add(raw_value)
        schedule_fp = _sha256(
            value.get("schedule_fingerprint"),
            name=f"{arm}.schedule_fingerprint",
        )
        _verify_repo_source_hashes(
            value.get("source_hashes"),
            repository_root=repository,
            name=f"{arm}.source_hashes",
            exact_paths=GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
        )
        source_hashes_by_arm[arm] = dict(
            _mapping(
                value.get("source_hashes"),
                name=f"{arm}.source_hashes",
            )
        )
        if (
            _sha256(
                value.get("cache_fingerprint"),
                name=f"{arm}.cache_fingerprint",
            )
            != verified_full_cache.semantic_cache_fingerprint
            or _sha256(
                value.get("neutral_payload_fingerprint"),
                name=f"{arm}.neutral_payload_fingerprint",
            )
            != verified_full_cache.neutral_payload_fingerprint
        ):
            raise ValueError(
                f"{arm} is not bound to the verified full-D_R content"
            )
        caches.add(
            _text(value.get("cache_instance_id"), name=f"{arm}.cache_instance")
        )
        rngs.add(_text(value.get("rng_instance_id"), name=f"{arm}.rng_instance"))
        modules.add(
            _text(value.get("module_instance_id"), name=f"{arm}.module_instance")
        )
        optimizers.add(
            _text(
                value.get("optimizer_instance_id"),
                name=f"{arm}.optimizer_instance",
            )
        )
        raw_storage = value.get("parameter_storage_ids")
        if (
            not isinstance(raw_storage, list)
            or not raw_storage
            or len(raw_storage) != len(set(raw_storage))
        ):
            raise ValueError(f"{arm}.parameter_storage_ids is invalid")
        storage_by_arm[arm] = {
            _text(item, name=f"{arm}.parameter storage")
            for item in raw_storage
        }
        initial = _sha256(
            value.get("initial_model_fingerprint"),
            name=f"{arm}.initial_model_fingerprint",
        )
        final = _sha256(
            value.get("final_model_fingerprint"),
            name=f"{arm}.final_model_fingerprint",
        )
        if initial == final:
            raise ValueError(f"{arm} parameter state did not change")
        artifact_path, _, stat_result = _validate_artifact(
            value.get("terminal_artifact"),
            name=f"{arm}.terminal_artifact",
            expected_model_fingerprint=final,
        )
        expected_terminal_name = (
            "control_terminal.safetensors"
            if arm == "PACRE_VC_v23_control"
            else "candidate_terminal.safetensors"
        )
        if artifact_path != str(
            bounded_runtime
            / "bounded/paired_bounded400"
            / expected_terminal_name
        ):
            raise PermissionError(
                f"{arm} terminal artifact left its fixed runtime path"
            )
        physical_key = (stat_result.st_dev, stat_result.st_ino)
        if physical_key in artifact_physical:
            raise PermissionError("bounded arm artifacts share physical storage")
        artifact_physical.add(physical_key)
        claimed_finite_audits[arm] = dict(
            _mapping(
                value.get("finite_audit"),
                name=f"{arm}.finite_audit",
            )
        )
        terminal_model_fingerprints[arm] = final
        metrics = _mapping(value.get("metrics"), name=f"{arm}.metrics")
        _exact_keys(metrics, _BOUNDED_METRIC_FIELDS, name=f"{arm}.metrics")
        for metric_name in (
            "initial_PMOPE",
            "terminal_PMOPE",
            "terminal_target_role_violation",
            "terminal_background_role_violation",
            "G1_PMOPE",
            "G1_target_role_violation",
            "G1_background_role_violation",
        ):
            _real(metrics.get(metric_name), name=f"{arm}.{metric_name}")
        for count_name in (
            "terminal_zero_crossed_target_states",
            "terminal_false_completion_states",
            "G1_zero_crossed_target_states",
            "G1_false_completion_states",
        ):
            _integer(metrics.get(count_name), name=f"{arm}.{count_name}")
        for fingerprint_name in (
            "terminal_field_fingerprint",
            "terminal_role_prediction_fingerprint",
            "G1_field_fingerprint",
            "G1_role_prediction_fingerprint",
        ):
            _sha256(
                metrics.get(fingerprint_name),
                name=f"{arm}.{fingerprint_name}",
            )
        gate_present = metrics.get("gate_role_distributions_present")
        if not isinstance(gate_present, bool):
            raise TypeError(f"{arm}.gate_role_distributions_present must be bool")
        candidate_gate_arm = arm == "GCR_PACRE_v24"
        _validate_gate_role_distribution(
            metrics.get("terminal_gate_distribution"),
            name=f"{arm}.terminal_gate_distribution",
            required=candidate_gate_arm,
        )
        _validate_gate_role_distribution(
            metrics.get("G1_gate_distribution"),
            name=f"{arm}.G1_gate_distribution",
            required=candidate_gate_arm,
        )
        if gate_present is not candidate_gate_arm:
            raise ValueError(
                f"{arm} gate distribution presence/mapping changed"
            )
        _metric(metrics, "pd")
        _count(metrics, "true_targets")
        _count(metrics, "recovered_anchor_misses")
        _metric(metrics, "mIoU")
        _metric(metrics, "nIoU")
        safety_checks(metrics)
        from cure_lite_v24.terminal_evidence import (
            mechanically_recompute_bounded_arm,
        )

        mechanical_metrics = mechanically_recompute_bounded_arm(
            arm=arm,
            terminal_artifact_path=artifact_path,
            expected_initial_model_fingerprint=initial,
            expected_final_model_fingerprint=final,
            expected_initial_parameter_fingerprint=_sha256(
                value.get("initial_shared_parameter_fingerprint"),
                name=f"{arm}.initial_shared_parameter_fingerprint",
            ),
            full_d_r_cache_artifact=verified_full_cache,
            requested_device=str(intent["requested_device"]),
        )
        if mechanical_metrics != metrics:
            raise PermissionError(
                f"{arm} receipt metrics differ from strict terminal "
                "safetensors + verified full-D_R cache recomputation"
            )
    if (
        any(len(values) != 1 for values in paired_values.values())
        or len(modules) != 2
        or len(optimizers) != 2
        or len(caches) != 2
        or len(rngs) != 2
        or source_hashes_by_arm[arm_names[0]]
        != source_hashes_by_arm[arm_names[1]]
        or storage_by_arm[arm_names[0]] & storage_by_arm[arm_names[1]]
    ):
        raise PermissionError("bounded pairing or execution independence changed")
    _, diagnostic_rows = _validate_paired_bounded_diagnostics(
        payload.get("paired_diagnostics"),
        arms=arms,
    )
    assert schedule_fp is not None
    schedule_path, _ = _validate_schedule_artifact(
        payload.get("schedule_artifact"),
        expected_schedule_fingerprint=schedule_fp,
        name="bounded schedule_artifact",
    )
    if schedule_path != str(
        bounded_runtime / "bounded/paired_bounded400/schedule.json"
    ):
        raise PermissionError(
            "bounded schedule artifact left its fixed runtime path"
        )
    from cure_lite_v24.training_trace import (
        mechanically_rebuild_schedule_artifact,
        trace_finite_audit,
        verify_training_trace_artifact,
    )

    rebuilt_schedule = mechanically_rebuild_schedule_artifact(
        schedule_artifact_path=schedule_path,
        cache_artifact=verified_full_cache,
        seed=OOF_SEED,
        epochs=BOUNDED_EPOCHS,
        steps_per_epoch=BOUNDED_STEPS_PER_EPOCH,
        expected_schedule_fingerprint=schedule_fp,
    )
    rebuilt_batch_sequence_fp = stable_fingerprint(
        [
            selection.canonical_payload()
            for selection in rebuilt_schedule.selections
        ]
    )
    if paired_values["batch_sequence_fingerprint"] != {
        rebuilt_batch_sequence_fp
    }:
        raise PermissionError(
            "bounded batch sequence differs from mechanical schedule "
            "reconstruction"
        )
    trace_payload = verify_training_trace_artifact(
        payload.get("training_trace_artifact"),
        expected_path=(
            bounded_runtime
            / "bounded/paired_bounded400/training_trace.json"
        ),
        stage_id="paired_bounded400",
        authorization_fingerprint=authorization_fp,
        schedule=rebuilt_schedule,
        arm_names=arm_names,
        terminal_model_fingerprints=terminal_model_fingerprints,
    )
    _validate_bounded_diagnostics_trace_binding(
        diagnostic_rows,
        trace_payload,
    )
    for arm in arm_names:
        if claimed_finite_audits[arm] != trace_finite_audit(
            trace_payload,
            arm=arm,
        ):
            raise PermissionError(
                f"{arm} finite audit is not derived from its exact "
                "persisted step trace"
            )
    fingerprint = _self_fingerprint(
        payload,
        field_name="receipt_fingerprint",
        name="paired bounded receipt",
    )
    return _register_token(VerifiedBoundedEvidence(
        payload_json=canonical_json(payload),
        receipt_fingerprint=fingerprint,
        oof_decision_fingerprint=oof.decision_fingerprint,
        access_audit_receipt_fingerprint=access.receipt_fingerprint,
        full_d_r_semantic_cache_fingerprint=(
            verified_full_cache.semantic_cache_fingerprint
        ),
        full_d_r_neutral_payload_fingerprint=(
            verified_full_cache.neutral_payload_fingerprint
        ),
        full_d_r_materialization_receipt_fingerprint=(
            verified_full_cache.receipt_fingerprint
        ),
        _issuer=_TOKEN_ISSUER,
    ))


def decide_paired_bounded400(
    bounded_evidence: VerifiedBoundedEvidence,
) -> VerifiedBoundedDecision:
    """Apply the bounded smoke vector to the exact verified arm receipt."""

    evidence = _token(
        bounded_evidence,
        VerifiedBoundedEvidence,
        name="bounded_evidence",
    )
    assert isinstance(evidence, VerifiedBoundedEvidence)
    arms = _mapping(evidence.payload.get("arms"), name="bounded arms")
    control_arm = _mapping(
        arms.get("PACRE_VC_v23_control"),
        name="bounded control",
    )
    candidate_arm = _mapping(
        arms.get("GCR_PACRE_v24"),
        name="bounded candidate",
    )
    control = _mapping(control_arm.get("metrics"), name="control metrics")
    candidate = _mapping(candidate_arm.get("metrics"), name="candidate metrics")
    checks = {
        "control_terminal_PMOPE_not_above_own_initial": (
            _metric(control, "terminal_PMOPE")
            <= _metric(control, "initial_PMOPE")
        ),
        "candidate_terminal_PMOPE_not_above_own_initial": (
            _metric(candidate, "terminal_PMOPE")
            <= _metric(candidate, "initial_PMOPE")
        ),
        "control_parameter_state_changed": (
            control_arm["initial_model_fingerprint"]
            != control_arm["final_model_fingerprint"]
        ),
        "candidate_parameter_state_changed": (
            candidate_arm["initial_model_fingerprint"]
            != candidate_arm["final_model_fingerprint"]
        ),
        "gate_endpoint_interior_and_role_distributions_present": (
            candidate.get("gate_role_distributions_present") is True
        ),
        **{
            f"candidate_{name}": passed
            for name, passed in safety_checks(candidate).items()
        },
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    diagnostics = {
        "gate_eligible": False,
        "paired_terminal_PMOPE_delta_candidate_minus_control": (
            _metric(candidate, "terminal_PMOPE")
            - _metric(control, "terminal_PMOPE")
        ),
        "paired_terminal_target_role_violation_delta": (
            _metric(candidate, "terminal_target_role_violation")
            - _metric(control, "terminal_target_role_violation")
        ),
        "paired_terminal_background_role_violation_delta": (
            _metric(candidate, "terminal_background_role_violation")
            - _metric(control, "terminal_background_role_violation")
        ),
        "candidate_zero_crossed_target_states": _count(
            candidate,
            "terminal_zero_crossed_target_states",
        ),
        "control_zero_crossed_target_states": _count(
            control,
            "terminal_zero_crossed_target_states",
        ),
        "candidate_false_completion_states": _count(
            candidate,
            "terminal_false_completion_states",
        ),
        "control_false_completion_states": _count(
            control,
            "terminal_false_completion_states",
        ),
        "candidate_forced_G1_zero_crossed_target_states": _count(
            candidate,
            "G1_zero_crossed_target_states",
        ),
        "interpretation": (
            "paired_deltas_and_discrete_mechanism_values_are_diagnostic_only"
        ),
    }
    body = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-paired-bounded400-decision-v2"
        ),
        "bounded_receipt_fingerprint": evidence.receipt_fingerprint,
        "OOF4_decision_fingerprint": evidence.oof_decision_fingerprint,
        "full_D_R_cache_binding": {
            "semantic_cache_fingerprint": (
                evidence.full_d_r_semantic_cache_fingerprint
            ),
            "neutral_payload_fingerprint": (
                evidence.full_d_r_neutral_payload_fingerprint
            ),
            "materialization_receipt_fingerprint": (
                evidence.full_d_r_materialization_receipt_fingerprint
            ),
        },
        "budget": {
            "epochs": BOUNDED_EPOCHS,
            "steps_per_epoch": BOUNDED_STEPS_PER_EPOCH,
            "updates": BOUNDED_UPDATES,
        },
        "status_semantics": "optimization_smoke_not_generalization_evidence",
        "discrete_diagnostics": diagnostics,
        "checks": checks,
        "failed_checks": failed,
        "gate_passed": not failed,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    fingerprint = stable_fingerprint(body)
    payload = {**body, "decision_fingerprint": fingerprint}
    return _register_token(VerifiedBoundedDecision(
        payload_json=canonical_json(payload),
        decision_fingerprint=fingerprint,
        bounded_receipt_fingerprint=evidence.receipt_fingerprint,
        oof_decision_fingerprint=evidence.oof_decision_fingerprint,
        full_d_r_semantic_cache_fingerprint=(
            evidence.full_d_r_semantic_cache_fingerprint
        ),
        full_d_r_neutral_payload_fingerprint=(
            evidence.full_d_r_neutral_payload_fingerprint
        ),
        full_d_r_materialization_receipt_fingerprint=(
            evidence.full_d_r_materialization_receipt_fingerprint
        ),
        _issuer=_TOKEN_ISSUER,
    ))


_GCR_PACRE_SOURCE_PATHS = (
    "cure_lite_v24/gcr_pacre.py",
    "cure_lite_v24/factory.py",
    "cure_lite_v24/training.py",
)
_GCR_PACRE_PARAMETER_NAMES = (
    "joint_state_weight",
    "joint_hidden_bias",
    "scalar_energy_weight",
)
_FORMAL_PARAMETER_SHAPES = {
    "joint_hidden_bias": [32],
    "joint_state_weight": [32, 80, 5, 5],
    "scalar_energy_weight": [32],
}
_FORMAL_MODEL_CONTRACT = {
    "model_class": (
        "cure_lite_v24.gcr_pacre."
        "CURELiteGatedCommonResidualPACRELevelSet"
    ),
    "config_class": (
        "cure_lite_v24.gcr_pacre.CoverageStateGCRPACREConfig"
    ),
    "config": {
        "feature_channels": 64,
        "feature_stride": 4,
        "width": 32,
        "normalization_epsilon": {
            "float_hex": "0x1.0c6f7a0b5ed8dp-20",
        },
        "field_amplitude": {
            "float_hex": "0x1.ccccccccccccdp-1",
        },
        "initial_field_value": {
            "float_hex": "0x1.ccccccccccccdp-1",
        },
        "field_policy": "gcr_pacre_single_zero_level_set_field_v1",
        "target_policy": (
            "fixed_amplitude_truncated_signed_chessboard_distance_"
            "on_masked_grid_v3"
        ),
        "output_policy": (
            "negative_zero_level_set_then_occupancy_exclusion_"
            "and_hard_union_v1"
        ),
        "feature_policy": (
            "samplewise_global_rms_normalized_relative_spatial_amplitude_v2"
        ),
        "numerical_policy": (
            "finite_closed_gate_interval_with_saturation_audit_v1"
        ),
        "coarse_radius": 2,
        "coverage_policy": (
            "lossless_bool_pixel_unshuffle_row_major_phase_coverage_v1"
        ),
        "equation_policy": (
            "flip_even_common_gate_times_flip_odd_residual_v1"
        ),
        "flip_policy": "exact_binary_current_center_phase_involution_v1",
        "transport_policy": (
            "align_corners_false_bilinear_then_row_major_phase_pack_v1"
        ),
        "input_representation": "phase_preserving",
        "interaction_policy": (
            "bounded_even_gate_times_binary_flip_odd_residual_v1"
        ),
        "energy_policy": (
            "shared_readout_residual_and_common_compatibility_v1"
        ),
        "method_id": "cure_lite_gcr_pacre_v24",
        "centering_policy": (
            "exact_per_cell_hidden_channel_phase_mean_quotient_v1"
        ),
    },
    "parameter_count": 64064,
    "parameter_shapes": _FORMAL_PARAMETER_SHAPES,
}
_FORMAL_TRAINING_FIELDS = {
    "schema_version",
    "role",
    "evaluation_role",
    "seed",
    "scope",
    "objective",
    "optimizer_fqcn",
    "policy_json",
    "policy_fingerprint",
    "model",
    "source_hashes",
    "cache_fingerprint",
    "schedule_fingerprint",
    "optimizer_config_fingerprint",
    "training_result_fingerprint",
    "budget",
    "compute",
    "from_scratch",
    "resume_allowed",
    "automatic_retry_allowed",
    "checkpoint_policy",
    "eligible_for_future_D_V_authorization_after_all_external_prerequisites",
    "eligible_for_future_D_T_authorization_after_all_external_prerequisites",
    "D_V_execution_authorized",
    "D_T_execution_authorized",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "selection_effect",
    "may_replace_seed42_primary",
}


def _formal_policy_payload(role: str, seed: int) -> dict[str, object]:
    future_primary = role == "primary" and seed == 42
    return {
        "role": role,
        "seed": seed,
        "scope": "D_R_formal_800",
        "budget": {
            "epochs": FORMAL_EPOCHS,
            "steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
            "updates": FORMAL_UPDATES,
        },
        "objective": "pmope_joint",
        "optimizer_fqcn": "torch.optim.adam.Adam",
        "learning_rate_hex": (0.001).hex(),
        "weight_decay_hex": (0.0).hex(),
        "betas_hex": [(0.9).hex(), (0.999).hex()],
        "epsilon_hex": (1.0e-8).hex(),
        "training_invocations": 1,
        "from_scratch": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "checkpoint_policy": "final_only",
        "D_V_execution_authorized": False,
        "D_T_execution_authorized": False,
        "eligible_for_future_D_V_authorization_after_all_external_prerequisites": (
            future_primary
        ),
        "eligible_for_future_D_T_authorization_after_all_external_prerequisites": (
            future_primary
        ),
    }


def _validate_core_formal_training_receipt(
    raw: object,
    *,
    expected_seed: int,
    expected_role: str,
    repository_root: Path,
) -> tuple[str, str, str, str]:
    payload = _mapping(raw, name="core Formal training receipt")
    _exact_keys(
        payload,
        _FORMAL_TRAINING_FIELDS,
        name="core Formal training receipt",
    )
    future_primary = expected_seed == 42 and expected_role == "primary"
    if (
        payload.get("schema_version")
        != "cure-lite-v24-gcr-pacre-pmope-training-receipt-v1"
        or payload.get("role") != expected_role
        or payload.get("evaluation_role") != expected_role
        or payload.get("seed") != expected_seed
        or payload.get("scope") != "D_R_formal_800"
        or payload.get("objective") != "pmope_joint"
        or payload.get("optimizer_fqcn") != "torch.optim.adam.Adam"
        or payload.get("from_scratch") is not True
        or payload.get("resume_allowed") is not False
        or payload.get("automatic_retry_allowed") is not False
        or payload.get("checkpoint_policy") != "final_only"
        or payload.get(
            "eligible_for_future_D_V_authorization_after_all_external_prerequisites"
        )
        is not future_primary
        or payload.get(
            "eligible_for_future_D_T_authorization_after_all_external_prerequisites"
        )
        is not future_primary
        or payload.get("D_V_execution_authorized") is not False
        or payload.get("D_T_execution_authorized") is not False
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
        or payload.get("selection_effect")
        != ("predeclared_primary" if future_primary else "none")
        or payload.get("may_replace_seed42_primary") is not False
    ):
        raise PermissionError("core Formal role/evaluation firewall changed")
    expected_policy = _formal_policy_payload(expected_role, expected_seed)
    policy_json = _text(payload.get("policy_json"), name="policy_json")
    try:
        parsed_policy = json.loads(policy_json)
    except json.JSONDecodeError as error:
        raise ValueError("policy_json is invalid") from error
    policy_fingerprint = _sha256(
        payload.get("policy_fingerprint"),
        name="policy_fingerprint",
    )
    if (
        canonical_json(parsed_policy) != policy_json
        or parsed_policy != expected_policy
        or sha256(policy_json.encode("utf-8")).hexdigest()
        != policy_fingerprint
    ):
        raise ValueError("core Formal policy binding changed")
    budget = _mapping(payload.get("budget"), name="Formal training budget")
    compute = _mapping(payload.get("compute"), name="Formal compute")
    if budget != {
        "epochs": FORMAL_EPOCHS,
        "steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
        "updates": FORMAL_UPDATES,
        "training_invocations": 1,
    } or compute != {
        "completed_updates": FORMAL_UPDATES,
        "forward_calls": FORMAL_UPDATES,
        "backward_calls": FORMAL_UPDATES,
        "optimizer_steps": FORMAL_UPDATES,
    }:
        raise ValueError("Formal training/compute ledger is not exact 800×40")
    source_rows = _verify_repo_source_hashes(
        payload.get("source_hashes"),
        repository_root=repository_root,
        name="Formal source_hashes",
        exact_paths=_GCR_PACRE_SOURCE_PATHS,
    )
    source_closure_fp = stable_fingerprint(dict(source_rows))
    for name in (
        "cache_fingerprint",
        "schedule_fingerprint",
        "optimizer_config_fingerprint",
        "training_result_fingerprint",
    ):
        _sha256(payload.get(name), name=name)
    model = _mapping(payload.get("model"), name="Formal model")
    _exact_keys(
        model,
        {
            "model_fqcn",
            "config_fqcn",
            "contract_json",
            "contract_fingerprint",
            "parameter_count",
            "initial_parameters",
            "initial_parameter_state_fingerprint",
            "initial_fingerprint",
            "final_fingerprint",
        },
        name="Formal model",
    )
    if (
        model.get("model_fqcn")
        != (
            "cure_lite_v24.gcr_pacre."
            "CURELiteGatedCommonResidualPACRELevelSet"
        )
        or model.get("config_fqcn")
        != "cure_lite_v24.gcr_pacre.CoverageStateGCRPACREConfig"
    ):
        raise ValueError("Formal model/config identity changed")
    contract_json = _text(model.get("contract_json"), name="contract_json")
    try:
        contract_payload = json.loads(contract_json)
    except json.JSONDecodeError as error:
        raise ValueError("model contract_json is invalid") from error
    contract_fp = _sha256(
        model.get("contract_fingerprint"),
        name="model.contract_fingerprint",
    )
    if (
        canonical_json(contract_payload) != contract_json
        or contract_payload != _FORMAL_MODEL_CONTRACT
        or sha256(contract_json.encode("utf-8")).hexdigest() != contract_fp
    ):
        raise ValueError("Formal model contract binding changed")
    raw_parameters = model.get("initial_parameters")
    if not isinstance(raw_parameters, list) or len(raw_parameters) != 3:
        raise ValueError("Formal initial parameter ledger is incomplete")
    normalized_parameters: list[dict[str, object]] = []
    total_parameters = 0
    for index, raw_parameter in enumerate(raw_parameters):
        parameter = _mapping(
            raw_parameter,
            name=f"initial_parameters[{index}]",
        )
        _exact_keys(
            parameter,
            {
                "name",
                "shape",
                "numel",
                "dtype",
                "byte_count",
                "content_fingerprint",
            },
            name=f"initial_parameters[{index}]",
        )
        if parameter.get("name") != _GCR_PACRE_PARAMETER_NAMES[index]:
            raise ValueError("Formal initial parameter order changed")
        shape = parameter.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
                for dim in shape
            )
        ):
            raise ValueError("Formal parameter shape is invalid")
        expected_numel = 1
        for dim in shape:
            expected_numel *= dim
        numel = _integer(parameter.get("numel"), name="parameter.numel", minimum=1)
        if (
            shape
            != _FORMAL_PARAMETER_SHAPES[
                _GCR_PACRE_PARAMETER_NAMES[index]
            ]
            or
            numel != expected_numel
            or parameter.get("dtype") != "torch.float32"
            or parameter.get("byte_count") != numel * 4
        ):
            raise ValueError("Formal parameter byte contract changed")
        _sha256(
            parameter.get("content_fingerprint"),
            name="parameter.content_fingerprint",
        )
        total_parameters += numel
        normalized_parameters.append(parameter)
    if (
        model.get("parameter_count") != 64064
        or total_parameters != 64064
        or model.get("initial_parameter_state_fingerprint")
        != stable_fingerprint(normalized_parameters)
    ):
        raise ValueError("Formal initial parameter state binding changed")
    initial_fp = _sha256(
        model.get("initial_fingerprint"),
        name="model.initial_fingerprint",
    )
    final_fp = _sha256(
        model.get("final_fingerprint"),
        name="model.final_fingerprint",
    )
    if initial_fp == final_fp:
        raise ValueError("Formal parameters did not change")
    training_receipt_fp = stable_fingerprint(payload)
    return training_receipt_fp, final_fp, contract_fp, source_closure_fp


def validate_formal_training_receipt(
    receipt: Mapping[str, object],
    *,
    expected_seed: int,
    expected_role: str,
    oof_decision: VerifiedOOFDecision,
    bounded_decision: VerifiedBoundedDecision,
    access_audit: VerifiedAccessAudit,
    cache_artifact: object,
    dataset_free_receipt_fingerprint: str,
    d_r_structural_receipt_fingerprint: str,
    repository_root: str | Path,
) -> VerifiedFormalTraining:
    """Validate exact outer evidence plus training.py's canonical receipt."""

    if (expected_seed, expected_role) not in {
        (42, "primary"),
        (43, "training_integrity_only"),
    }:
        raise ValueError("Formal accepts only frozen seed/role pairs")
    oof = _token(oof_decision, VerifiedOOFDecision, name="oof_decision")
    bounded = _token(
        bounded_decision,
        VerifiedBoundedDecision,
        name="bounded_decision",
    )
    access = _token(access_audit, VerifiedAccessAudit, name="access_audit")
    assert isinstance(oof, VerifiedOOFDecision)
    assert isinstance(bounded, VerifiedBoundedDecision)
    assert isinstance(access, VerifiedAccessAudit)
    if (
        oof.payload.get("gate_passed") is not True
        or bounded.payload.get("gate_passed") is not True
        or bounded.oof_decision_fingerprint != oof.decision_fingerprint
    ):
        raise PermissionError("Formal prerequisites have not passed coherently")
    dataset_free_fp = _sha256(
        dataset_free_receipt_fingerprint,
        name="dataset_free_receipt_fingerprint",
    )
    structural_fp = _sha256(
        d_r_structural_receipt_fingerprint,
        name="d_r_structural_receipt_fingerprint",
    )
    payload = _mapping(receipt, name="Formal evidence receipt")
    _exact_keys(
        payload,
        {
            "schema_version",
            "seed",
            "evaluation_role",
            "prerequisites",
            "access_audit_receipt_fingerprint",
            "training_receipt",
            "training_receipt_fingerprint",
            "finite_audit",
            "cache_artifact",
            "run_start_artifact",
            "schedule_artifact",
            "training_trace_artifact",
            "terminal_artifact",
            "terminal_D_R_evaluation",
            "terminal_D_R_evaluation_fingerprint",
            "source_closure",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "receipt_fingerprint",
        },
        name="Formal evidence receipt",
    )
    expected_stage = f"formal800_seed{expected_seed}_{expected_role}"
    if (
        payload.get("schema_version")
        != "cure-lite-v24-gcr-pacre-formal800-evidence-v6"
        or payload.get("seed") != expected_seed
        or payload.get("evaluation_role") != expected_role
        or access.stage_id != expected_stage
        or access.allowed_splits != ("D_R",)
        or payload.get("access_audit_receipt_fingerprint")
        != access.receipt_fingerprint
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("Formal outer role/access identity changed")
    prerequisites = _mapping(
        payload.get("prerequisites"),
        name="Formal prerequisites",
    )
    _exact_keys(
        prerequisites,
        {
            "dataset_free_receipt_fingerprint",
            "D_R_structural_receipt_fingerprint",
            "OOF4_decision_fingerprint",
            "paired_bounded400_decision_fingerprint",
        },
        name="Formal prerequisites",
    )
    if prerequisites != {
        "dataset_free_receipt_fingerprint": dataset_free_fp,
        "D_R_structural_receipt_fingerprint": structural_fp,
        "OOF4_decision_fingerprint": oof.decision_fingerprint,
        "paired_bounded400_decision_fingerprint": (
            bounded.decision_fingerprint
        ),
    }:
        raise PermissionError("Formal prerequisite binding changed")
    repository = Path(repository_root).resolve(strict=True)
    (
        training_receipt_fp,
        final_model_fp,
        contract_fp,
        core_source_closure_fp,
    ) = _validate_core_formal_training_receipt(
        payload.get("training_receipt"),
        expected_seed=expected_seed,
        expected_role=expected_role,
        repository_root=repository,
    )
    if payload.get("training_receipt_fingerprint") != training_receipt_fp:
        raise ValueError("Formal core training fingerprint changed")
    # The inner training receipt still binds its exact three-file execution
    # core.  The v6 outer receipt separately binds the complete unified
    # repository runtime closure.
    _sha256(
        core_source_closure_fp,
        name="Formal core source closure fingerprint",
    )
    source_closure = _mapping(
        payload.get("source_closure"),
        name="Formal unified source_closure",
    )
    _exact_keys(
        source_closure,
        {
            "schema_version",
            "source_hashes",
            "source_closure_fingerprint",
        },
        name="Formal unified source_closure",
    )
    from cure_lite_v24.source_closure import (
        GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
        GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
        gcr_pacre_v24_source_closure_fingerprint,
    )

    unified_rows = _verify_repo_source_hashes(
        source_closure.get("source_hashes"),
        repository_root=repository,
        name="Formal unified source_hashes",
        exact_paths=GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
    )
    unified_fp = gcr_pacre_v24_source_closure_fingerprint(unified_rows)
    if (
        source_closure.get("schema_version")
        != GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA
        or source_closure.get("source_closure_fingerprint") != unified_fp
    ):
        raise RuntimeError("Formal unified source closure changed")
    claimed_finite_audit = dict(
        _mapping(
            payload.get("finite_audit"),
            name="Formal finite_audit",
        )
    )
    inner = _mapping(
        payload.get("training_receipt"),
        name="Formal training receipt",
    )
    semantic_cache_fp = _sha256(
        inner.get("cache_fingerprint"),
        name="Formal cache_fingerprint",
    )
    from cure_lite_v24.formal_cache_artifacts import (
        require_verified_formal_cache_artifact,
        verify_formal_cache_artifact,
    )

    supplied_cache = require_verified_formal_cache_artifact(cache_artifact)
    expected_cache_id = (
        f"formal800-seed{expected_seed}-{expected_role}-full-D_R-cache"
    )
    if (
        supplied_cache.cache_id != expected_cache_id
        or supplied_cache.semantic_cache_fingerprint != semantic_cache_fp
    ):
        raise PermissionError("Formal cache token identity changed")
    verified_cache = verify_formal_cache_artifact(
        supplied_cache.path,
        cache_id=expected_cache_id,
        expected_semantic_cache_fingerprint=semantic_cache_fp,
        expected_neutral_payload_fingerprint=(
            bounded.full_d_r_neutral_payload_fingerprint
        ),
    )
    if (
        supplied_cache.receipt_fingerprint
        != verified_cache.receipt_fingerprint
        or payload.get("cache_artifact") != verified_cache.payload
    ):
        raise PermissionError(
            "Formal outer receipt lacks the mechanically verified cache token"
        )
    if (
        verified_cache.semantic_cache_fingerprint
        != bounded.full_d_r_semantic_cache_fingerprint
        or verified_cache.neutral_payload_fingerprint
        != bounded.full_d_r_neutral_payload_fingerprint
    ):
        raise PermissionError(
            "Formal cache content differs from verified bounded full-D_R "
            "materialization"
        )
    run_key = (
        "seed42_primary"
        if expected_seed == 42
        else "seed43_training_integrity_only"
    )
    formal_runtime = _gcr_pacre_v24_evidence_runtime_root(repository)
    run_start, run_start_stat = (
        _validate_gcr_pacre_persistent_run_start(
            payload.get("run_start_artifact"),
            name="Formal run_start_artifact",
            expected_schema=(
                "cure-lite-v24-gcr-pacre-formal800-"
                "persistent-run-start-v2"
            ),
            expected_path=(
                formal_runtime
                / "formal"
                / run_key
                / "run_start.json"
            ),
            expected_stage_id=expected_stage,
            expected_access_fingerprint=access.receipt_fingerprint,
            repository_root=repository,
        )
    )
    _exact_keys(
        run_start,
        {
            "schema_version",
            "protocol_id",
            "path_policy",
            "marker_path",
            "seed",
            "role",
            "stage_id",
            "process_instance_fingerprint",
            "chain_config",
            "authorization_fingerprint",
            "access_audit_receipt_fingerprint",
            "cache_artifact",
            "source_closure",
            "intent",
            "intent_fingerprint",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "marker_fingerprint",
        },
        name="Formal run-start payload",
    )
    marker_chain = _mapping(
        run_start.get("chain_config"),
        name="Formal run-start chain_config",
    )
    marker_cache = _mapping(
        run_start.get("cache_artifact"),
        name="Formal run-start cache_artifact",
    )
    marker_source = _mapping(
        run_start.get("source_closure"),
        name="Formal run-start source_closure",
    )
    intent = _mapping(
        run_start.get("intent"),
        name="Formal run-start intent",
    )
    authorization_fp = _sha256(
        run_start.get("authorization_fingerprint"),
        name="Formal run-start authorization_fingerprint",
    )
    process_instance_fp = _sha256(
        run_start.get("process_instance_fingerprint"),
        name="Formal run-start process_instance_fingerprint",
    )
    expected_output = formal_runtime / "formal" / run_key
    if (
        run_start.get("path_policy")
        != "fixed_runtime_root_seed_role_directory_run_start_json_v1"
        or run_start.get("seed") != expected_seed
        or run_start.get("role") != expected_role
        or marker_chain.get("path")
        != str(formal_runtime / "formal/execution_chain_config.json")
        or marker_cache
        != {
            "path": verified_cache.path,
            "file_sha256": verified_cache.file_sha256,
            "receipt_fingerprint": verified_cache.receipt_fingerprint,
            "semantic_cache_fingerprint": (
                verified_cache.semantic_cache_fingerprint
            ),
            "neutral_payload_fingerprint": (
                verified_cache.neutral_payload_fingerprint
            ),
        }
        or marker_source
        != {
            "schema_version": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
            "fingerprint": unified_fp,
            "source_hashes": dict(unified_rows),
        }
        or intent
        != {
            "execution_kind": "Formal800_D_R_training",
            "split": "D_R",
            "requested_device": intent.get("requested_device"),
            "output_directory": str(expected_output),
            "epochs": FORMAL_EPOCHS,
            "steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
            "optimizer_steps_authorized": FORMAL_UPDATES,
            "parameter_updates_authorized": FORMAL_UPDATES,
            "training_invocations_authorized": 1,
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_V_materialization_intended": False,
            "D_T_materialization_intended": False,
        }
        or not isinstance(intent.get("requested_device"), str)
        or not intent.get("requested_device")
        or run_start.get("intent_fingerprint")
        != stable_fingerprint(intent)
    ):
        raise PermissionError("Formal persistent run-start binding changed")
    _validate_current_authorization_artifact(
        expected_output / "authorization.json",
        expected_authorization_fingerprint=authorization_fp,
        name=f"Formal seed{expected_seed}",
    )
    purpose = (
        "Formal800_seed42_primary_training_cache"
        if (expected_seed, expected_role) == (42, "primary")
        else "Formal800_seed43_training_integrity_cache"
    )
    observed = {
        (
            str(row["logical_id"]),
            str(row["source_fingerprint"]),
            str(row["purpose"]),
        )
        for row in access.observed_payloads
        if row.get("split") == "D_R"
    }
    if (
        expected_cache_id,
        verified_cache.file_sha256,
        purpose,
    ) not in observed:
        raise PermissionError(
            "Formal cache lacks exact access-audit evidence"
        )
    schedule_fp = _sha256(
        inner.get("schedule_fingerprint"),
        name="Formal schedule_fingerprint",
    )
    schedule_path, _, schedule_policy_fp = (
        _validate_formal_schedule_artifact(
            payload.get("schedule_artifact"),
            expected_schedule_fingerprint=schedule_fp,
            expected_semantic_cache_fingerprint=semantic_cache_fp,
            expected_seed=expected_seed,
            name="Formal schedule_artifact",
        )
    )
    if schedule_path != str(expected_output / "schedule.json"):
        raise PermissionError(
            "Formal schedule artifact left its fixed runtime path"
        )
    from cure_lite_v24.training_trace import (
        mechanically_rebuild_schedule_artifact,
        trace_finite_audit,
        verify_training_trace_artifact,
    )

    rebuilt_schedule = mechanically_rebuild_schedule_artifact(
        schedule_artifact_path=schedule_path,
        cache_artifact=verified_cache,
        seed=expected_seed,
        epochs=FORMAL_EPOCHS,
        steps_per_epoch=FORMAL_STEPS_PER_EPOCH,
        expected_schedule_fingerprint=schedule_fp,
    )
    artifact_path, _, _ = _validate_artifact(
        payload.get("terminal_artifact"),
        name="Formal terminal_artifact",
        expected_model_fingerprint=final_model_fp,
    )
    if artifact_path != str(
        expected_output / "terminal/model.safetensors"
    ):
        raise PermissionError(
            "Formal terminal artifact left its fixed runtime path"
        )
    artifact = _mapping(
        payload.get("terminal_artifact"),
        name="Formal terminal_artifact",
    )
    artifact_sha = _sha256(
        artifact.get("file_sha256"),
        name="Formal terminal artifact sha",
    )
    trace_payload = verify_training_trace_artifact(
        payload.get("training_trace_artifact"),
        expected_path=expected_output / "training_trace.json",
        stage_id=expected_stage,
        authorization_fingerprint=authorization_fp,
        schedule=rebuilt_schedule,
        arm_names=(expected_role,),
        terminal_model_fingerprints={
            expected_role: final_model_fp,
        },
    )
    if claimed_finite_audit != trace_finite_audit(
        trace_payload,
        arm=expected_role,
    ):
        raise PermissionError(
            "Formal finite audit is not derived from the exact persisted "
            "step trace"
        )
    claimed_terminal_evaluation = _mapping(
        payload.get("terminal_D_R_evaluation"),
        name="Formal terminal_D_R_evaluation",
    )
    claimed_terminal_evaluation_fp = _sha256(
        payload.get("terminal_D_R_evaluation_fingerprint"),
        name="Formal terminal_D_R_evaluation_fingerprint",
    )
    from cure_lite_v24.terminal_evidence import (
        mechanically_recompute_formal_terminal,
    )

    (
        mechanical_terminal_evaluation,
        mechanical_terminal_evaluation_fp,
    ) = mechanically_recompute_formal_terminal(
        terminal_artifact_path=artifact_path,
        expected_final_model_fingerprint=final_model_fp,
        cache_artifact=verified_cache,
        requested_device=str(intent["requested_device"]),
        seed=expected_seed,
        role=expected_role,
    )
    if (
        claimed_terminal_evaluation != mechanical_terminal_evaluation
        or claimed_terminal_evaluation_fp
        != mechanical_terminal_evaluation_fp
    ):
        raise PermissionError(
            "Formal terminal D_R evidence differs from strict "
            "safetensors + verified cache recomputation"
        )
    fingerprint = _self_fingerprint(
        payload,
        field_name="receipt_fingerprint",
        name="Formal evidence receipt",
    )
    return _register_token(VerifiedFormalTraining(
        payload_json=canonical_json(payload),
        receipt_fingerprint=fingerprint,
        training_receipt_fingerprint=training_receipt_fp,
        seed=expected_seed,
        role=expected_role,
        semantic_cache_fingerprint=semantic_cache_fp,
        cache_artifact_path=verified_cache.path,
        cache_artifact_sha256=verified_cache.file_sha256,
        cache_artifact_device=verified_cache.device,
        cache_artifact_inode=verified_cache.inode,
        cache_artifact_receipt_fingerprint=(
            verified_cache.receipt_fingerprint
        ),
        cache_neutral_payload_fingerprint=(
            verified_cache.neutral_payload_fingerprint
        ),
        schedule_fingerprint=schedule_fp,
        schedule_policy_without_seed_fingerprint=schedule_policy_fp,
        final_model_fingerprint=final_model_fp,
        terminal_artifact_path=artifact_path,
        terminal_artifact_sha256=artifact_sha,
        run_start_artifact_path=str(
            payload["run_start_artifact"]["path"]
        ),
        run_start_artifact_sha256=str(
            payload["run_start_artifact"]["file_sha256"]
        ),
        run_start_artifact_device=run_start_stat.st_dev,
        run_start_artifact_inode=run_start_stat.st_ino,
        run_start_marker_fingerprint=str(
            run_start["marker_fingerprint"]
        ),
        process_instance_fingerprint=process_instance_fp,
        unified_source_closure_fingerprint=unified_fp,
        model_contract_fingerprint=contract_fp,
        access_audit_receipt_fingerprint=access.receipt_fingerprint,
        _cache_artifact_token=verified_cache,
        _issuer=_TOKEN_ISSUER,
    ))


def verify_formal800_training_independence(
    seed42_primary: VerifiedFormalTraining,
    seed43_integrity: VerifiedFormalTraining,
) -> VerifiedFormalTrainingPair:
    """Derive seed42/43 independence and the seed43 non-selection firewall."""

    primary = _token(
        seed42_primary,
        VerifiedFormalTraining,
        name="seed42_primary",
    )
    integrity = _token(
        seed43_integrity,
        VerifiedFormalTraining,
        name="seed43_integrity",
    )
    assert isinstance(primary, VerifiedFormalTraining)
    assert isinstance(integrity, VerifiedFormalTraining)
    if (
        (primary.seed, primary.role) != (42, "primary")
        or (integrity.seed, integrity.role)
        != (43, "training_integrity_only")
    ):
        raise PermissionError("Formal pair seed/role binding changed")
    primary_inner = _mapping(
        primary.payload.get("training_receipt"),
        name="seed42 training receipt",
    )
    integrity_inner = _mapping(
        integrity.payload.get("training_receipt"),
        name="seed43 training receipt",
    )
    primary_model = _mapping(primary_inner.get("model"), name="seed42 model")
    integrity_model = _mapping(
        integrity_inner.get("model"),
        name="seed43 model",
    )
    primary_path = Path(primary.terminal_artifact_path)
    integrity_path = Path(integrity.terminal_artifact_path)
    primary_stat = primary_path.stat()
    integrity_stat = integrity_path.stat()
    primary_run_start = Path(primary.run_start_artifact_path)
    integrity_run_start = Path(integrity.run_start_artifact_path)
    primary_run_start_stat = primary_run_start.stat()
    integrity_run_start_stat = integrity_run_start.stat()
    if (
        _file_sha256(primary_run_start)
        != primary.run_start_artifact_sha256
        or _file_sha256(integrity_run_start)
        != integrity.run_start_artifact_sha256
        or primary_run_start_stat.st_mode & 0o222
        or integrity_run_start_stat.st_mode & 0o222
    ):
        raise RuntimeError("Formal pair run-start bytes changed")
    from cure_lite_v24.formal_cache_artifacts import (
        verify_formal_cache_pair_independence,
    )

    cache_pair = verify_formal_cache_pair_independence(
        primary._cache_artifact_token,
        integrity._cache_artifact_token,
    )
    cache_pair_checks = _mapping(
        cache_pair.payload.get("checks"),
        name="Formal cache pair checks",
    )
    checks = {
        "same_source_closure": (
            primary.unified_source_closure_fingerprint
            == integrity.unified_source_closure_fingerprint
            and primary.payload.get("source_closure")
            == integrity.payload.get("source_closure")
        ),
        "different_run_start_paths": (
            primary.run_start_artifact_path
            != integrity.run_start_artifact_path
        ),
        "different_run_start_storage": (
            (
                primary.run_start_artifact_device,
                primary.run_start_artifact_inode,
            )
            != (
                integrity.run_start_artifact_device,
                integrity.run_start_artifact_inode,
            )
            == (
                integrity_run_start_stat.st_dev,
                integrity_run_start_stat.st_ino,
            )
            and (
                primary.run_start_artifact_device,
                primary.run_start_artifact_inode,
            )
            == (
                primary_run_start_stat.st_dev,
                primary_run_start_stat.st_ino,
            )
        ),
        "different_run_start_markers": (
            primary.run_start_marker_fingerprint
            != integrity.run_start_marker_fingerprint
        ),
        "different_interpreter_process_instances": (
            primary.process_instance_fingerprint
            != integrity.process_instance_fingerprint
        ),
        "same_model_contract": (
            primary.model_contract_fingerprint
            == integrity.model_contract_fingerprint
        ),
        "different_initial_parameter_states": (
            primary_model.get("initial_parameter_state_fingerprint")
            != integrity_model.get("initial_parameter_state_fingerprint")
        ),
        "different_initial_model_fingerprints": (
            primary_model.get("initial_fingerprint")
            != integrity_model.get("initial_fingerprint")
        ),
        "different_final_model_fingerprints": (
            primary.final_model_fingerprint
            != integrity.final_model_fingerprint
        ),
        "same_semantic_cache_fingerprint": (
            primary.semantic_cache_fingerprint
            == integrity.semantic_cache_fingerprint
            == primary_inner.get("cache_fingerprint")
            == integrity_inner.get("cache_fingerprint")
        ),
        "different_cache_artifact_paths": (
            primary.cache_artifact_path != integrity.cache_artifact_path
        ),
        "different_cache_artifact_storage": (
            (
                primary.cache_artifact_device,
                primary.cache_artifact_inode,
            )
            != (
                integrity.cache_artifact_device,
                integrity.cache_artifact_inode,
            )
        ),
        "both_cache_artifacts_single_link": (
            cache_pair_checks.get("both_single_link") is True
        ),
        "same_cache_tensor_payload": (
            cache_pair_checks.get("same_neutral_payload_fingerprint") is True
        ),
        "cache_tensor_storage_instances_disjoint": (
            cache_pair_checks.get(
                "actual_loaded_tensor_storages_disjoint"
            )
            is True
        ),
        "cache_fiemap_and_non_mmap_mechanically_verified": (
            cache_pair_checks.get(
                "both_fiemap_no_shared_unknown_delalloc_encoded"
            )
            is True
            and cache_pair_checks.get("both_fixed_non_mmap_loads") is True
        ),
        "different_schedule_fingerprints": (
            primary.schedule_fingerprint
            != integrity.schedule_fingerprint
        ),
        "same_schedule_policy_except_seed": (
            primary.schedule_policy_without_seed_fingerprint
            == integrity.schedule_policy_without_seed_fingerprint
        ),
        "different_terminal_paths": (
            primary.terminal_artifact_path
            != integrity.terminal_artifact_path
        ),
        "different_terminal_storage": (
            (primary_stat.st_dev, primary_stat.st_ino)
            != (integrity_stat.st_dev, integrity_stat.st_ino)
        ),
        "seed43_future_D_V_ineligible": (
            integrity_inner.get(
                "eligible_for_future_D_V_authorization_after_all_external_prerequisites"
            )
            is False
        ),
        "seed43_future_D_T_ineligible": (
            integrity_inner.get(
                "eligible_for_future_D_T_authorization_after_all_external_prerequisites"
            )
            is False
        ),
        "seed43_D_V_D_T_unread_and_unauthorized": (
            integrity_inner.get("D_V_execution_authorized") is False
            and integrity_inner.get("D_T_execution_authorized") is False
            and integrity_inner.get("D_V_payload_accessed") is False
            and integrity_inner.get("D_T_payload_accessed") is False
        ),
        "seed43_no_selection_or_replacement": (
            integrity_inner.get("selection_effect") == "none"
            and integrity_inner.get("may_replace_seed42_primary") is False
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise PermissionError(
            "Formal seed42/43 independence failed: " + ", ".join(failed)
        )
    body = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-formal800-training-pair-v2"
        ),
        "seed42_primary_receipt_fingerprint": primary.receipt_fingerprint,
        "seed43_training_integrity_receipt_fingerprint": (
            integrity.receipt_fingerprint
        ),
        "cache_artifact_pair_fingerprint": cache_pair.pair_fingerprint,
        "checks": checks,
        "seed43_selection_effect": "none",
        "seed43_may_replace_seed42_primary": False,
        "D_V_payload_accessed_by_seed43": False,
        "D_T_payload_accessed_by_seed43": False,
    }
    fingerprint = stable_fingerprint(body)
    payload = {**body, "pair_fingerprint": fingerprint}
    return _register_token(VerifiedFormalTrainingPair(
        payload_json=canonical_json(payload),
        pair_fingerprint=fingerprint,
        seed42_receipt_fingerprint=primary.receipt_fingerprint,
        seed43_receipt_fingerprint=integrity.receipt_fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


_D_T_ACTIVATION_PREREQUISITES = [
    "v24_dataset_free_PASS",
    "v24_D_R_structural_PASS",
    "v24_D_R_OOF4_PASS",
    "v24_paired_bounded400_PASS",
    "v24_Formal800_seed42_terminal_PASS",
    "v24_Formal800_seed43_training_integrity_complete",
    "seed43_D_V_and_D_T_unread",
    "this_preregistration_predates_D_V_authorization",
    "v24_D_V_adaptive_PASS",
    "all_source_verifier_artifact_checks_PASS",
]


def validate_d_t_preregistration(
    preregistration: Mapping[str, object],
    *,
    repository_root: str | Path,
) -> VerifiedDTPreregistration:
    """Verify every frozen one-shot D_T field without opening D_T."""

    payload = _mapping(preregistration, name="D_T preregistration")
    _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "protocol_preregistration_fingerprint",
            "status",
            "must_predate",
            "candidate_model_role",
            "candidate_model_binding",
            "candidate_attempts",
            "automatic_retry_allowed",
            "resume_allowed",
            "checkpoint_selection_allowed",
            "threshold_search_allowed",
            "seed43_evaluation_allowed",
            "Base_validity_and_envelope",
            "safety",
            "decision_table",
            "activation_prerequisites",
            "authorization_created",
            "D_T_payload_accessed",
            "D_V_payload_accessed_by_this_preregistration",
            "preregistration_fingerprint",
        },
        name="D_T preregistration",
    )
    repository = Path(repository_root).resolve(strict=True)
    _, protocol_fingerprint = _verified_main_preregistration(repository)
    if payload.get("protocol_preregistration_fingerprint") != (
        protocol_fingerprint
    ):
        raise ValueError("D_T preregistration is not anchored to main protocol")
    fingerprint = _self_fingerprint(
        payload,
        field_name="preregistration_fingerprint",
        name="D_T preregistration",
    )
    if (
        payload.get("schema_version")
        != "cure-lite-v24-gcr-pacre-D_T-preregistration-v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "RULES_FROZEN_RUN_NOT_AUTHORIZED"
        or payload.get("must_predate") != "v24_D_V_authorization"
        or payload.get("candidate_model_role")
        != "Formal800_seed42_primary_final_only"
        or payload.get("candidate_attempts") != 1
        or payload.get("automatic_retry_allowed") is not False
        or payload.get("resume_allowed") is not False
        or payload.get("checkpoint_selection_allowed") is not False
        or payload.get("threshold_search_allowed") is not False
        or payload.get("seed43_evaluation_allowed") is not False
        or payload.get("authorization_created") is not False
        or payload.get("D_T_payload_accessed") is not False
        or payload.get("D_V_payload_accessed_by_this_preregistration")
        is not False
        or payload.get("activation_prerequisites")
        != _D_T_ACTIVATION_PREREQUISITES
    ):
        raise PermissionError("D_T one-shot identity/authorization changed")
    model_binding = _mapping(
        payload.get("candidate_model_binding"),
        name="D_T candidate_model_binding",
    )
    _exact_keys(
        model_binding,
        {
            "status",
            "final_model_fingerprint",
            "binding_artifact",
            "binding_artifact_must_be_sealed_before",
            "binding_may_not_change_these_preregistered_rules",
        },
        name="D_T candidate_model_binding",
    )
    if model_binding != {
        "status": "PENDING_FORMAL800_SEED42_VERIFIED_TERMINAL",
        "final_model_fingerprint": None,
        "binding_artifact": "D_T_seed42_model_binding.json",
        "binding_artifact_must_be_sealed_before": "v24_D_V_authorization",
        "binding_may_not_change_these_preregistered_rules": True,
    }:
        raise PermissionError("D_T pending model-binding contract changed")
    envelope = _mapping(
        payload.get("Base_validity_and_envelope"),
        name="D_T Base_validity_and_envelope",
    )
    _exact_keys(
        envelope,
        {
            "procedure",
            "operating_points",
            "D_T_threshold_grid_evaluation_allowed",
            "D_T_threshold_selection_allowed",
            "D_T_evaluated_Base_operating_point_count",
            "values",
            "rounding_before_comparison",
            "integer_endpoints",
            "mIoU_nIoU",
            "minimum_fixed_uplift_margin",
        },
        name="D_T Base_validity_and_envelope",
    )
    if (
        envelope.get("procedure")
        != "same_schema_and_validity_policy_as_frozen_D_V_comparison"
        or envelope.get("D_T_threshold_grid_evaluation_allowed") is not False
        or envelope.get("D_T_threshold_selection_allowed") is not False
        or envelope.get("D_T_evaluated_Base_operating_point_count") != 2
        or envelope.get("values")
        != (
            "one_shot_D_T_metrics_at_only_the_two_fixed_operating_points_"
            "then_endpoint_wise_valid_envelope"
        )
        or envelope.get("rounding_before_comparison") is not False
        or envelope.get("integer_endpoints")
        != "strictly_greater_than_each_endpoint_wise_valid_Base_maximum"
        or envelope.get("mIoU_nIoU")
        != "not_below_each_endpoint_wise_valid_Base_maximum"
        or envelope.get("minimum_fixed_uplift_margin") is not None
    ):
        raise ValueError("D_T Base envelope policy changed")
    operating_points = _mapping(
        envelope.get("operating_points"),
        name="D_T operating_points",
    )
    if set(operating_points) != {"Base@A", "Base@B"}:
        raise ValueError("D_T must contain exactly Base@A/Base@B")
    base_a = _mapping(operating_points["Base@A"], name="D_T Base@A")
    _exact_keys(base_a, {"threshold", "source"}, name="D_T Base@A")
    if base_a != {
        "threshold": BASE_A_THRESHOLD,
        "source": "preexisting_fixed_anchor",
    }:
        raise ValueError("D_T Base@A contract changed")
    base_b = _mapping(operating_points["Base@B"], name="D_T Base@B")
    _exact_keys(
        base_b,
        {
            "threshold",
            "threshold_source_repo_path",
            "threshold_source_file_sha256",
            "threshold_selector",
            "selected_on",
            "D_T_selection_effect",
        },
        name="D_T Base@B",
    )
    if (
        _real(base_b.get("threshold"), name="D_T Base@B.threshold") != 0.14
        or base_b.get("threshold_selector")
        != ["evaluation_result", "Base@B_selection", "selected_threshold"]
        or base_b.get("selected_on") != "frozen_D_V"
        or base_b.get("D_T_selection_effect") != "none"
    ):
        raise ValueError("D_T Base@B selector contract changed")
    relative = _strict_relative_path(
        base_b.get("threshold_source_repo_path"),
        name="D_T Base@B source path",
    )
    source = repository / relative
    if (
        not source.is_file()
        or source.is_symlink()
        or source.resolve(strict=True) != source
        or _file_sha256(source)
        != _sha256(
            base_b.get("threshold_source_file_sha256"),
            name="D_T Base@B source sha",
        )
    ):
        raise RuntimeError("D_T frozen Base@B source changed")
    source_payload = _strict_json(source)
    _self_fingerprint(
        source_payload,
        field_name="receipt_fingerprint",
        name="D_T Base@B source receipt",
    )
    selected: object = source_payload
    for part in base_b["threshold_selector"]:  # type: ignore[index]
        selected = _mapping(selected, name="D_T Base@B selector").get(part)
    if _real(selected, name="sealed Base@B threshold") != 0.14:
        raise RuntimeError("sealed D_V-selected Base@B threshold changed")
    safety = _mapping(payload.get("safety"), name="D_T safety")
    if safety != {
        "retention": 1.0,
        "pixel_fa_max": PIXEL_FA_LIMIT,
        "raw_background_fa_max": RAW_BACKGROUND_FA_LIMIT,
        "fp_components_per_megapixel_max": FP_COMPONENTS_PER_MP_LIMIT,
        "budget_violation": False,
    }:
        raise ValueError("D_T safety contract changed")
    decision_table = _mapping(
        payload.get("decision_table"),
        name="D_T decision_table",
    )
    if decision_table != {
        "PASS": "all_identity_source_metric_relative_and_safety_checks_pass",
        "FAIL": (
            "valid_execution_completes_and_any_performance_or_safety_check_fails"
        ),
        "STOP": (
            "identity_source_schema_integrity_or_one_shot_contract_is_invalid"
        ),
    }:
        raise ValueError("D_T decision table changed")
    return _register_token(VerifiedDTPreregistration(
        payload_json=canonical_json(payload),
        preregistration_fingerprint=fingerprint,
        protocol_preregistration_fingerprint=protocol_fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


def validate_d_t_seed42_model_binding(
    receipt: Mapping[str, object],
    *,
    d_t_preregistration: VerifiedDTPreregistration,
    formal_seed42: VerifiedFormalTraining,
    formal_pair: VerifiedFormalTrainingPair,
    access_audit: VerifiedAccessAudit,
) -> VerifiedDTSeed42ModelBinding:
    """Seal the one future D_T model identity before any D_V authorization."""

    prereg = _token(
        d_t_preregistration,
        VerifiedDTPreregistration,
        name="d_t_preregistration",
    )
    formal = _token(
        formal_seed42,
        VerifiedFormalTraining,
        name="formal_seed42",
    )
    pair = _token(
        formal_pair,
        VerifiedFormalTrainingPair,
        name="formal_pair",
    )
    access = _token(access_audit, VerifiedAccessAudit, name="access_audit")
    assert isinstance(prereg, VerifiedDTPreregistration)
    assert isinstance(formal, VerifiedFormalTraining)
    assert isinstance(pair, VerifiedFormalTrainingPair)
    assert isinstance(access, VerifiedAccessAudit)
    if (
        formal.seed != 42
        or formal.role != "primary"
        or pair.seed42_receipt_fingerprint != formal.receipt_fingerprint
    ):
        raise PermissionError("D_T model binding requires Formal seed42 primary")
    if (
        access.stage_id != "d_t_seed42_model_binding"
        or access.allowed_splits
        or access.observed_payloads
    ):
        raise PermissionError("model binding must be a payload-free stage")
    payload = _mapping(receipt, name="D_T seed42 model binding")
    _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_preregistration_fingerprint",
            "D_T_preregistration_fingerprint",
            "Formal800_evidence_receipt_fingerprint",
            "Formal800_pair_fingerprint",
            "training_receipt_fingerprint",
            "seed",
            "role",
            "final_model_fingerprint",
            "model_contract_fingerprint",
            "terminal_artifact",
            "access_audit_receipt_fingerprint",
            "events",
            "D_V_authorization_created",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "receipt_fingerprint",
        },
        name="D_T seed42 model binding",
    )
    if (
        payload.get("schema_version")
        != "cure-lite-v24-D_T-seed42-model-binding-v1"
        or payload.get("protocol_preregistration_fingerprint")
        != prereg.protocol_preregistration_fingerprint
        or payload.get("D_T_preregistration_fingerprint")
        != prereg.preregistration_fingerprint
        or payload.get("Formal800_evidence_receipt_fingerprint")
        != formal.receipt_fingerprint
        or payload.get("Formal800_pair_fingerprint") != pair.pair_fingerprint
        or payload.get("training_receipt_fingerprint")
        != formal.training_receipt_fingerprint
        or payload.get("seed") != 42
        or payload.get("role") != "primary"
        or payload.get("final_model_fingerprint")
        != formal.final_model_fingerprint
        or payload.get("model_contract_fingerprint")
        != formal.model_contract_fingerprint
        or payload.get("access_audit_receipt_fingerprint")
        != access.receipt_fingerprint
        or payload.get("D_V_authorization_created") is not False
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("D_T seed42 model binding identity changed")
    events = _mapping(payload.get("events"), name="D_T binding events")
    _exact_keys(
        events,
        {
            "Formal800_seed42_terminal_sealed",
            "D_T_model_binding_sealed",
            "D_V_authorization_created",
        },
        name="D_T binding events",
    )
    terminal_event = _integer(
        events.get("Formal800_seed42_terminal_sealed"),
        name="Formal terminal event",
    )
    binding_event = _integer(
        events.get("D_T_model_binding_sealed"),
        name="D_T binding event",
    )
    if terminal_event >= binding_event or events.get(
        "D_V_authorization_created"
    ) is not None:
        raise PermissionError("D_T model binding did not predate D_V authorization")
    artifact = _mapping(
        payload.get("terminal_artifact"),
        name="D_T bound terminal artifact",
    )
    path, model_fp, _ = _validate_artifact(
        artifact,
        name="D_T bound terminal artifact",
        expected_model_fingerprint=formal.final_model_fingerprint,
    )
    if (
        path != formal.terminal_artifact_path
        or artifact.get("file_sha256") != formal.terminal_artifact_sha256
        or model_fp != formal.final_model_fingerprint
    ):
        raise ValueError("D_T binding artifact differs from verified Formal model")
    fingerprint = _self_fingerprint(
        payload,
        field_name="receipt_fingerprint",
        name="D_T seed42 model binding",
    )
    return _register_token(VerifiedDTSeed42ModelBinding(
        payload_json=canonical_json(payload),
        receipt_fingerprint=fingerprint,
        formal_receipt_fingerprint=formal.receipt_fingerprint,
        final_model_fingerprint=formal.final_model_fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


def _select_json_path(
    value: Mapping[str, object],
    path: object,
    *,
    name: str,
) -> object:
    if (
        not isinstance(path, list)
        or not path
        or any(not isinstance(part, str) or not part for part in path)
    ):
        raise ValueError(f"{name} must be a non-empty string path")
    selected: object = value
    for part in path:
        selected = _mapping(selected, name=name).get(part)
    return selected


def load_exact_baseline_envelope(
    binding_path: str | Path,
    *,
    repository_root: str | Path,
    access_audit: VerifiedAccessAudit,
) -> VerifiedBaselineEnvelope:
    """Load a checksum-bound aggregate-only D_V Base envelope."""

    access = _token(access_audit, VerifiedAccessAudit, name="access_audit")
    assert isinstance(access, VerifiedAccessAudit)
    if (
        access.stage_id != "exact_baseline_envelope"
        or access.allowed_splits != ("SEALED_D_V_AGGREGATE_METADATA",)
    ):
        raise PermissionError("baseline loader requires aggregate-only access")
    root = Path(repository_root).resolve(strict=True)
    binding = _strict_json(Path(binding_path).resolve(strict=True))
    _exact_keys(
        binding,
        {
            "schema_version",
            "protocol_preregistration_fingerprint",
            "source_kind",
            "source_repo_path",
            "source_file_sha256",
            "source_schema_version",
            "selector",
            "valid_base_names_selector",
            "required_fields",
            "comparison_policy",
            "existing_sealed_aggregate_metadata_read",
            "new_D_V_inference_performed",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "binding_fingerprint",
        },
        name="exact baseline binding",
    )
    _, protocol_fp = _verified_main_preregistration(root)
    if binding.get("protocol_preregistration_fingerprint") != protocol_fp:
        raise ValueError("baseline binding is not anchored to main protocol")
    binding_fp = _self_fingerprint(
        binding,
        field_name="binding_fingerprint",
        name="exact baseline binding",
    )
    if (
        binding.get("schema_version")
        != "cure-lite-v24-exact-baseline-ledger-binding-v1"
        or binding.get("source_kind") != "sealed_aggregate_metadata_only"
        or binding.get("required_fields")
        != [
            "true_targets",
            "recovered_anchor_misses",
            "mIoU",
            "nIoU",
        ]
        or binding.get("comparison_policy")
        != {
            "integer_endpoints": "strictly_greater",
            "mIoU_nIoU": "not_below",
            "rounding_before_comparison": False,
            "minimum_fixed_uplift_margin": None,
        }
        or binding.get("existing_sealed_aggregate_metadata_read") is not True
        or binding.get("new_D_V_inference_performed") is not False
        or binding.get("D_V_payload_accessed") is not False
        or binding.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("baseline binding is not frozen aggregate metadata")
    relative = _strict_relative_path(
        binding.get("source_repo_path"),
        name="baseline source_repo_path",
    )
    source = root / relative
    source_sha = _sha256(
        binding.get("source_file_sha256"),
        name="baseline source_file_sha256",
    )
    if (
        not source.is_file()
        or source.is_symlink()
        or source.resolve(strict=True) != source
        or _file_sha256(source) != source_sha
    ):
        raise RuntimeError("sealed baseline source bytes changed")
    if not any(
        row.get("logical_id") == str(relative)
        and row.get("source_fingerprint") == source_sha
        and row.get("purpose") == "sealed_aggregate_baseline_envelope"
        for row in access.observed_payloads
    ):
        raise PermissionError("sealed baseline source lacks access-audit binding")
    payload = _strict_json(source)
    if payload.get("schema_version") != binding.get("source_schema_version"):
        raise RuntimeError("sealed baseline schema changed")
    source_receipt_fp = _self_fingerprint(
        payload,
        field_name="receipt_fingerprint",
        name="sealed baseline source",
    )
    if binding.get("selector") != ["evaluation_result", "operating_points"]:
        raise ValueError("baseline operating-point selector changed")
    operating_points = _mapping(
        _select_json_path(payload, binding["selector"], name="selector"),
        name="baseline operating_points",
    )
    valid_names = _select_json_path(
        payload,
        binding.get("valid_base_names_selector"),
        name="valid_base_names_selector",
    )
    if (
        not isinstance(valid_names, list)
        or not valid_names
        or len(valid_names) != len(set(valid_names))
        or any(
            not isinstance(name, str) or not name.startswith("Base@")
            for name in valid_names
        )
    ):
        raise ValueError("baseline valid Base name ledger changed")
    valid_rows: list[dict[str, object]] = []
    for base_name in valid_names:
        point = _mapping(
            operating_points.get(base_name),
            name=f"operating_points[{base_name}]",
        )
        summary = _mapping(point.get("summary"), name=f"{base_name}.summary")
        aggregate = _mapping(
            point.get("aggregate_evaluation"),
            name=f"{base_name}.aggregate_evaluation",
        )
        normalized = {
            "true_targets": _count(summary, "true_targets"),
            "recovered_anchor_misses": _count(
                summary,
                "recovered_anchor_misses",
            ),
            "mIoU": _metric(summary, "mIoU"),
            "nIoU": _metric(summary, "nIoU"),
            "retention": _metric(summary, "retention"),
            "pixel_fa": _metric(summary, "pixel_Fa"),
            "raw_background_fa": _metric(summary, "raw_background_Fa"),
            "fp_components_per_mp": _metric(
                summary,
                "false_positive_components_per_megapixel",
            ),
            "budget_violation": summary.get("budget_violation"),
        }
        if (
            aggregate.get("budget_violation") is not False
            or normalized["budget_violation"] is not False
            or not all(safety_checks(normalized).values())
            or normalized["mIoU"] != _metric(aggregate, "miou")
            or normalized["nIoU"] != _metric(aggregate, "niou")
            or normalized["retention"] != _metric(aggregate, "retention")
            or normalized["pixel_fa"] != _metric(aggregate, "pixel_fa")
            or normalized["raw_background_fa"]
            != _metric(aggregate, "raw_background_fa")
            or normalized["fp_components_per_mp"]
            != _metric(aggregate, "fp_components_per_mp")
            or normalized["recovered_anchor_misses"]
            != _count(aggregate, "recovered_anchor_misses")
            or normalized["true_targets"]
            != (
                _count(aggregate, "retained_anchor_covered")
                + _count(aggregate, "recovered_anchor_misses")
            )
        ):
            raise PermissionError(f"invalid sealed Base row {base_name!r}")
        valid_rows.append(normalized)
    envelope = {
        "true_targets": max(_count(row, "true_targets") for row in valid_rows),
        "recovered_anchor_misses": max(
            _count(row, "recovered_anchor_misses") for row in valid_rows
        ),
        "mIoU": max(_metric(row, "mIoU") for row in valid_rows),
        "nIoU": max(_metric(row, "nIoU") for row in valid_rows),
    }
    body = {
        "schema_version": "cure-lite-v24-exact-baseline-envelope-v2",
        "binding_fingerprint": binding_fp,
        "source_receipt_fingerprint": source_receipt_fp,
        "access_audit_receipt_fingerprint": access.receipt_fingerprint,
        "envelope": envelope,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    evidence_fp = stable_fingerprint(body)
    token_payload = {**body, "evidence_fingerprint": evidence_fp}
    return _register_token(VerifiedBaselineEnvelope(
        payload_json=canonical_json(token_payload),
        binding_fingerprint=binding_fp,
        source_receipt_fingerprint=source_receipt_fp,
        _issuer=_TOKEN_ISSUER,
    ))


def verify_dv_candidate_evidence(
    receipt: Mapping[str, object],
    *,
    model_binding: VerifiedDTSeed42ModelBinding,
    access_audit: VerifiedAccessAudit,
) -> VerifiedDVCandidateEvidence:
    """Bind authorized D_V aggregate metrics to the pre-D_V model seal."""

    binding = _token(
        model_binding,
        VerifiedDTSeed42ModelBinding,
        name="model_binding",
    )
    access = _token(access_audit, VerifiedAccessAudit, name="access_audit")
    assert isinstance(binding, VerifiedDTSeed42ModelBinding)
    assert isinstance(access, VerifiedAccessAudit)
    if access.stage_id != "v24_D_V_one_shot" or access.allowed_splits != ("D_V",):
        raise PermissionError("D_V evidence access role changed")
    payload = _mapping(receipt, name="D_V candidate evidence")
    _exact_keys(
        payload,
        {
            "schema_version",
            "model_binding_receipt_fingerprint",
            "final_model_fingerprint",
            "access_audit_receipt_fingerprint",
            "candidate_metrics",
            "D_T_payload_accessed",
            "receipt_fingerprint",
        },
        name="D_V candidate evidence",
    )
    if (
        payload.get("schema_version")
        != "cure-lite-v24-gcr-pacre-D_V-candidate-evidence-v1"
        or payload.get("model_binding_receipt_fingerprint")
        != binding.receipt_fingerprint
        or payload.get("final_model_fingerprint")
        != binding.final_model_fingerprint
        or payload.get("access_audit_receipt_fingerprint")
        != access.receipt_fingerprint
        or payload.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("D_V candidate/model/firewall binding changed")
    metrics = _mapping(
        payload.get("candidate_metrics"),
        name="D_V candidate_metrics",
    )
    _exact_keys(
        metrics,
        {
            "true_targets",
            "recovered_anchor_misses",
            "mIoU",
            "nIoU",
            "retention",
            "pixel_fa",
            "raw_background_fa",
            "fp_components_per_mp",
            "budget_violation",
        },
        name="D_V candidate_metrics",
    )
    _count(metrics, "true_targets")
    _count(metrics, "recovered_anchor_misses")
    _metric(metrics, "mIoU")
    _metric(metrics, "nIoU")
    safety_checks(metrics)
    fingerprint = _self_fingerprint(
        payload,
        field_name="receipt_fingerprint",
        name="D_V candidate evidence",
    )
    return _register_token(VerifiedDVCandidateEvidence(
        payload_json=canonical_json(payload),
        receipt_fingerprint=fingerprint,
        model_binding_receipt_fingerprint=binding.receipt_fingerprint,
        access_audit_receipt_fingerprint=access.receipt_fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


def decide_relative_dv_gate(
    candidate_evidence: VerifiedDVCandidateEvidence,
    baseline_envelope: VerifiedBaselineEnvelope,
) -> dict[str, object]:
    """Apply the exact relative D_V gate only to verifier-issued evidence."""

    candidate_token = _token(
        candidate_evidence,
        VerifiedDVCandidateEvidence,
        name="candidate_evidence",
    )
    baseline_token = _token(
        baseline_envelope,
        VerifiedBaselineEnvelope,
        name="baseline_envelope",
    )
    assert isinstance(candidate_token, VerifiedDVCandidateEvidence)
    assert isinstance(baseline_token, VerifiedBaselineEnvelope)
    candidate = _mapping(
        candidate_token.payload.get("candidate_metrics"),
        name="candidate_metrics",
    )
    baseline = _mapping(
        baseline_token.payload.get("envelope"),
        name="baseline_envelope",
    )
    checks = {
        "true_targets_strictly_above_exact_Base_envelope": (
            _count(candidate, "true_targets")
            > _count(baseline, "true_targets")
        ),
        "recovered_anchor_misses_strictly_above_exact_Base_envelope": (
            _count(candidate, "recovered_anchor_misses")
            > _count(baseline, "recovered_anchor_misses")
        ),
        "mIoU_not_below_exact_Base_envelope": (
            _metric(candidate, "mIoU") >= _metric(baseline, "mIoU")
        ),
        "nIoU_not_below_exact_Base_envelope": (
            _metric(candidate, "nIoU") >= _metric(baseline, "nIoU")
        ),
        "D_T_payload_not_accessed": (
            candidate_token.payload.get("D_T_payload_accessed") is False
        ),
        **{
            f"candidate_{name}": passed
            for name, passed in safety_checks(candidate).items()
        },
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    body = {
        "schema_version": "cure-lite-v24-gcr-pacre-relative-d-v-decision-v2",
        "candidate_evidence_fingerprint": (
            candidate_token.receipt_fingerprint
        ),
        "baseline_binding_fingerprint": baseline_token.binding_fingerprint,
        "comparison": "exact_endpoint_wise_frozen_valid_Base_envelope",
        "minimum_fixed_uplift_margin": None,
        "plus_one_count_improvement_is_sufficient": True,
        "baseline_envelope": dict(baseline),
        "checks": checks,
        "failed_checks": failed,
        "gate_passed": not failed,
        "authorizes_D_T": False,
    }
    return {**body, "decision_fingerprint": stable_fingerprint(body)}


__all__ = [
    "BASE_A_THRESHOLD",
    "BASE_B_THRESHOLD_GRID",
    "BOUNDED_EPOCHS",
    "BOUNDED_STEPS_PER_EPOCH",
    "BOUNDED_UPDATES",
    "FORMAL_EPOCHS",
    "FORMAL_STEPS_PER_EPOCH",
    "FORMAL_UPDATES",
    "FactualSufficientStatistics",
    "OOF_ARMS",
    "OOF_FOLD_COUNT",
    "OOF_SEED",
    "VerifiedAccessAudit",
    "VerifiedBaselineEnvelope",
    "VerifiedBoundedDecision",
    "VerifiedBoundedEvidence",
    "VerifiedDVCandidateEvidence",
    "VerifiedDTPreregistration",
    "VerifiedDTSeed42ModelBinding",
    "VerifiedFactualPool",
    "VerifiedFormalTraining",
    "VerifiedFormalTrainingPair",
    "VerifiedGatePathEvidence",
    "VerifiedOOF4Split",
    "VerifiedOOFDecision",
    "VerifiedOOFFold",
    "VerifiedOOFPooledEvidence",
    "canonical_json",
    "combine_oof4_factual_pools",
    "decide_oof4_pooled",
    "decide_paired_bounded400",
    "decide_relative_dv_gate",
    "derive_root_source_ids",
    "deterministic_oof4_plan",
    "load_exact_baseline_envelope",
    "pool_factual_only_rows",
    "propagate_root_source_ids",
    "require_verified_access_audit",
    "require_verified_bounded_decision",
    "require_verified_oof4_split",
    "require_verified_oof_decision",
    "safety_checks",
    "select_base_b_train_fold_threshold",
    "stable_fingerprint",
    "validate_d_t_seed42_model_binding",
    "validate_formal_training_receipt",
    "validate_d_t_preregistration",
    "validate_oof_fold_execution_receipt",
    "validate_paired_bounded_receipt",
    "validate_protocol_artifact_manifest",
    "validate_root_source_closure",
    "verify_oof4_plan",
    "verify_oof4_split_preregistration",
    "verify_access_audit_receipt",
    "verify_dv_candidate_evidence",
    "verify_formal800_training_independence",
    "verify_gate_path_receipt",
]
