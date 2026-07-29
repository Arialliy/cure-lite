"""Verifier-bound source-disjoint folds for the v24 D_R OOF-4 gate.

This module never derives a new split from results.  It expands one
``VerifiedOOF4Split`` capability into an exact per-fold closure and rejects a
missing, unknown, duplicated, or cross-root sample before any cache is built.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Final, Iterable, Mapping

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from tools.gcr_pacre_v24_protocol import (
    VerifiedOOF4Split,
    require_verified_oof4_split,
)


OOF_FOLD_CLOSURE_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-fold-closure-v1"
)
_TOKEN_ISSUER = object()
_TOKEN_REGISTRY: dict[int, object] = {}


def _register(value: object) -> object:
    if getattr(value, "_issuer", None) is not _TOKEN_ISSUER:
        raise AssertionError("attempted to register an unsigned OOF closure")
    identity = id(value)
    prior = _TOKEN_REGISTRY.get(identity)
    if prior is not None and prior is not value:
        raise RuntimeError("OOF closure token identity collision")
    _TOKEN_REGISTRY[identity] = value
    return value


def _ordered_unique_text(
    values: Iterable[object],
    *,
    name: str,
) -> tuple[str, ...]:
    rows = tuple(values)
    if (
        any(not isinstance(value, str) or not value for value in rows)
        or len(rows) != len(set(rows))
    ):
        raise ValueError(f"{name} must be unique nonempty text")
    return rows


@dataclass(frozen=True, slots=True)
class VerifiedOOFFoldClosure:
    """Exact train/holdout root and sample capability for one frozen fold."""

    payload_json: str
    fold_id: int
    split_receipt_fingerprint: str
    plan_fingerprint: str
    root_by_sample_fingerprint: str
    train_root_source_ids: tuple[str, ...]
    held_out_root_source_ids: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    held_out_sample_ids: tuple[str, ...]
    sample_root_pairs: tuple[tuple[str, str], ...]
    closure_fingerprint: str
    _issuer: object

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise AssertionError("verified OOF closure payload changed")
        return value

    @property
    def root_by_sample(self) -> dict[str, str]:
        return dict(self.sample_root_pairs)


def require_verified_oof_fold_closure(
    value: object,
) -> VerifiedOOFFoldClosure:
    if (
        type(value) is not VerifiedOOFFoldClosure
        or value._issuer is not _TOKEN_ISSUER
        or _TOKEN_REGISTRY.get(id(value)) is not value
    ):
        raise TypeError(
            "fold_closure must be issued by the fixed OOF split expander"
        )
    return value


def verify_oof_fold_closure(
    verified_split: VerifiedOOF4Split,
    *,
    fold_id: int,
    available_sample_ids: Iterable[str],
) -> VerifiedOOFFoldClosure:
    """Expand one preregistered fold against the exact materialized universe."""

    split = require_verified_oof4_split(verified_split)
    if isinstance(fold_id, bool) or not isinstance(fold_id, int):
        raise TypeError("fold_id must be an integer")
    plan = split.plan
    if plan.get("fold_count") != 4 or fold_id not in range(4):
        raise ValueError("OOF-4 fold_id must be in [0,3]")
    folds = plan.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise RuntimeError("verified OOF plan lost its four folds")
    raw_fold = next(
        (
            value
            for value in folds
            if isinstance(value, Mapping)
            and value.get("fold_id") == fold_id
        ),
        None,
    )
    if raw_fold is None:
        raise RuntimeError("verified OOF plan lacks the requested fold")

    root_by_sample = split.root_by_sample
    available = _ordered_unique_text(
        available_sample_ids,
        name="available_sample_ids",
    )
    if set(available) != set(root_by_sample):
        missing = sorted(set(root_by_sample) - set(available))
        unknown = sorted(set(available) - set(root_by_sample))
        raise PermissionError(
            "materialized D_R sample universe differs from frozen OOF split; "
            f"missing={missing}, unknown={unknown}"
        )
    train_roots = _ordered_unique_text(
        raw_fold.get("train_root_source_ids", ()),
        name="train_root_source_ids",
    )
    holdout_roots = _ordered_unique_text(
        raw_fold.get("held_out_root_source_ids", ()),
        name="held_out_root_source_ids",
    )
    train_samples = _ordered_unique_text(
        raw_fold.get("train_sample_ids", ()),
        name="train_sample_ids",
    )
    holdout_samples = _ordered_unique_text(
        raw_fold.get("held_out_sample_ids", ()),
        name="held_out_sample_ids",
    )
    expected_roots = set(root_by_sample.values())
    if (
        not train_roots
        or not holdout_roots
        or set(train_roots) & set(holdout_roots)
        or set(train_roots) | set(holdout_roots) != expected_roots
        or set(train_samples) & set(holdout_samples)
        or set(train_samples) | set(holdout_samples) != set(root_by_sample)
    ):
        raise PermissionError("OOF fold root/sample partition is not exact")
    for sample_id in train_samples:
        if root_by_sample[sample_id] not in set(train_roots):
            raise PermissionError("train sample crosses its root closure")
    for sample_id in holdout_samples:
        if root_by_sample[sample_id] not in set(holdout_roots):
            raise PermissionError("holdout sample crosses its root closure")
    body = {
        "schema_version": OOF_FOLD_CLOSURE_SCHEMA,
        "fold_id": fold_id,
        "split_receipt_fingerprint": split.receipt_fingerprint,
        "plan_fingerprint": split.plan_fingerprint,
        "root_by_sample_fingerprint": (
            split.root_by_sample_fingerprint
        ),
        "train_root_source_ids": list(train_roots),
        "held_out_root_source_ids": list(holdout_roots),
        "train_sample_ids": list(train_samples),
        "held_out_sample_ids": list(holdout_samples),
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
    fingerprint = stable_fingerprint(body)
    payload = {**body, "closure_fingerprint": fingerprint}
    return _register(VerifiedOOFFoldClosure(
        payload_json=canonical_json(payload),
        fold_id=fold_id,
        split_receipt_fingerprint=split.receipt_fingerprint,
        plan_fingerprint=split.plan_fingerprint,
        root_by_sample_fingerprint=split.root_by_sample_fingerprint,
        train_root_source_ids=train_roots,
        held_out_root_source_ids=holdout_roots,
        train_sample_ids=train_samples,
        held_out_sample_ids=holdout_samples,
        sample_root_pairs=tuple(sorted(root_by_sample.items())),
        closure_fingerprint=fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


def verify_all_oof_fold_closures(
    verified_split: VerifiedOOF4Split,
    *,
    available_sample_ids: Iterable[str],
) -> tuple[VerifiedOOFFoldClosure, ...]:
    """Issue all four closures and recheck once-only holdout coverage."""

    available = tuple(available_sample_ids)
    closures = tuple(
        verify_oof_fold_closure(
            verified_split,
            fold_id=fold_id,
            available_sample_ids=available,
        )
        for fold_id in range(4)
    )
    held_out = [
        sample_id
        for closure in closures
        for sample_id in closure.held_out_sample_ids
    ]
    roots = [
        root_id
        for closure in closures
        for root_id in closure.held_out_root_source_ids
    ]
    split = require_verified_oof4_split(verified_split)
    if (
        len(held_out) != len(set(held_out))
        or set(held_out) != set(split.root_by_sample)
        or len(roots) != len(set(roots))
        or set(roots) != set(split.root_by_sample.values())
    ):
        raise PermissionError(
            "four OOF closures do not hold out each sample/root exactly once"
        )
    return closures


__all__ = [
    "OOF_FOLD_CLOSURE_SCHEMA",
    "VerifiedOOFFoldClosure",
    "require_verified_oof_fold_closure",
    "verify_all_oof_fold_closures",
    "verify_oof_fold_closure",
]
