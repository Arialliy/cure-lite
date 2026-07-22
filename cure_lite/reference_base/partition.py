"""Deterministic group-disjoint D_B fit/select partitioning."""

from __future__ import annotations

from dataclasses import dataclass

from ..cache.schema import stable_fingerprint
from ..splits import SplitManifest


DB_PARTITION_SCHEMA = "cure-lite-reference-base-d-b-partition-v1"


@dataclass(frozen=True)
class DBPartition:
    fit_sample_ids: tuple[str, ...]
    select_sample_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    select_group_ids: tuple[str, ...]
    manifest_fingerprint: str
    selection_fraction: float
    selection_seed: int
    fingerprint: str

    def canonical_payload(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": DB_PARTITION_SCHEMA,
            "manifest_fingerprint": self.manifest_fingerprint,
            "selection_fraction": self.selection_fraction,
            "selection_seed": self.selection_seed,
            "fit_sample_ids": list(self.fit_sample_ids),
            "select_sample_ids": list(self.select_sample_ids),
            "fit_group_ids": list(self.fit_group_ids),
            "select_group_ids": list(self.select_group_ids),
        }
        if include_fingerprint:
            payload["partition_fingerprint"] = self.fingerprint
        return payload


def _rank_group(group_id: str, seed: int) -> tuple[str, str]:
    return (
        stable_fingerprint(
            {
                "schema_version": "cure-lite-reference-base-group-rank-v1",
                "seed": seed,
                "group_id": group_id,
            }
        ),
        group_id,
    )


def build_d_b_partition(
    manifest: SplitManifest,
    *,
    selection_fraction: float = 0.2,
    seed: int = 42,
) -> DBPartition:
    """Split only D_B into deterministic, group-disjoint fit/select subsets."""

    if not isinstance(manifest, SplitManifest):
        raise TypeError("manifest must be SplitManifest")
    if (
        isinstance(selection_fraction, bool)
        or not isinstance(selection_fraction, (int, float))
        or not 0.0 < float(selection_fraction) < 1.0
    ):
        raise ValueError("selection_fraction must lie strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    records = tuple(manifest.records_for("D_B"))
    groups: dict[str, list[str]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(record.sample_id)
    if len(groups) < 2:
        raise ValueError("D_B needs at least two groups for fit/select partitioning")
    ordered_groups = tuple(sorted(groups, key=lambda item: _rank_group(item, seed)))
    target = max(1, min(len(records) - 1, round(len(records) * selection_fraction)))
    cumulative = 0
    candidates: list[tuple[int, int]] = []
    for prefix_length, group_id in enumerate(ordered_groups[:-1], start=1):
        cumulative += len(groups[group_id])
        candidates.append((abs(cumulative - target), prefix_length))
    _, selected_prefix = min(candidates)
    select_groups = frozenset(ordered_groups[:selected_prefix])
    fit_groups = frozenset(ordered_groups[selected_prefix:])
    if not select_groups or not fit_groups or select_groups & fit_groups:
        raise RuntimeError("invalid D_B group partition")

    select_ids = tuple(
        sorted(
            record.sample_id
            for record in records
            if record.group_id in select_groups
        )
    )
    fit_ids = tuple(
        sorted(
            record.sample_id
            for record in records
            if record.group_id in fit_groups
        )
    )
    if set(fit_ids) & set(select_ids) or set(fit_ids) | set(select_ids) != {
        record.sample_id for record in records
    }:
        raise RuntimeError("D_B fit/select sample membership is inconsistent")
    payload = {
        "schema_version": DB_PARTITION_SCHEMA,
        "manifest_fingerprint": manifest.fingerprint,
        "selection_fraction": float(selection_fraction),
        "selection_seed": seed,
        "fit_sample_ids": list(fit_ids),
        "select_sample_ids": list(select_ids),
        "fit_group_ids": sorted(fit_groups),
        "select_group_ids": sorted(select_groups),
    }
    return DBPartition(
        fit_sample_ids=fit_ids,
        select_sample_ids=select_ids,
        fit_group_ids=tuple(sorted(fit_groups)),
        select_group_ids=tuple(sorted(select_groups)),
        manifest_fingerprint=manifest.fingerprint,
        selection_fraction=float(selection_fraction),
        selection_seed=seed,
        fingerprint=stable_fingerprint(payload),
    )


__all__ = ["DB_PARTITION_SCHEMA", "DBPartition", "build_d_b_partition"]
