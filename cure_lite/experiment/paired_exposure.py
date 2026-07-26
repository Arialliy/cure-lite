"""Read-only exposure audit for the frozen clean-pair schedule.

No model forward, optimizer step, calibration, or evaluation split is touched.
The returned receipt accounts for every one of the 64,000 ``D_R`` pair
exposures in the 800 x 40 plan.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping

from ..cache.schema import stable_fingerprint
from ..train.paired_pools import (
    PAIRED_EPOCHS,
    PAIRED_EXPOSURES,
    PAIRED_OPTIMIZER_UPDATES,
    PAIRED_STEPS_PER_EPOCH,
    PAIRS_PER_UPDATE,
    PairedSchedule,
)


PAIRED_EXPOSURE_SCHEMA = "cure-lite-clean-pair-exposure-v1"


def _exposure_statistics(
    counts: Mapping[str, int],
    *,
    top_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, object]:
    if not counts:
        raise ValueError("exposure population cannot be empty")
    ordered = sorted(
        ((identity, int(count)) for identity, count in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if any(count < 0 for _, count in ordered):
        raise ValueError("exposure counts cannot be negative")
    values = [count for _, count in ordered]
    total = sum(values)
    square_sum = sum(count * count for count in values)
    result: dict[str, object] = {
        "population": len(values),
        "total_exposures": total,
        "unique_exposed": sum(count > 0 for count in values),
        "zero_exposure": sum(count == 0 for count in values),
        "minimum_count": min(values),
        "maximum_count": max(values),
        "maximum_share": max(values) / total if total else 0.0,
        "ess": total * total / square_sum if square_sum else 0.0,
        "normalized_ess": (
            total * total / square_sum / len(values)
            if square_sum
            else 0.0
        ),
    }
    for top in top_values:
        selected = ordered[: min(top, len(ordered))]
        result[f"top{top}_concentration"] = {
            "share": (
                sum(count for _, count in selected) / total
                if total
                else 0.0
            ),
            "items": [
                {"identity": identity, "count": count}
                for identity, count in selected
            ],
        }
    return result


def build_paired_exposure_receipt(
    schedule: PairedSchedule,
) -> dict[str, object]:
    """Return complete target/source exposure accounting for one schedule."""

    if not isinstance(schedule, PairedSchedule):
        raise TypeError("schedule must be a PairedSchedule")
    target_counts = Counter(
        {pair.pair_id: 0 for pair in schedule.pairs}
    )
    source_counts = Counter(
        {pair.sample_id: 0 for pair in schedule.pairs}
    )
    source_distinct_violations = 0
    for first_index, second_index in schedule.batch_pair_indices:
        first = schedule.pairs[first_index]
        second = schedule.pairs[second_index]
        target_counts[first.pair_id] += 1
        target_counts[second.pair_id] += 1
        source_counts[first.sample_id] += 1
        source_counts[second.sample_id] += 1
        if first.sample_id == second.sample_id:
            source_distinct_violations += 1

    if sum(target_counts.values()) != PAIRED_EXPOSURES:
        raise RuntimeError("target exposure replay has the wrong event count")
    if sum(source_counts.values()) != PAIRED_EXPOSURES:
        raise RuntimeError("source exposure replay has the wrong event count")
    target = _exposure_statistics(target_counts)
    source = _exposure_statistics(source_counts)
    maximum_pair_count_difference = (
        int(target["maximum_count"]) - int(target["minimum_count"])
    )
    payload: dict[str, object] = {
        "schema_version": PAIRED_EXPOSURE_SCHEMA,
        "evidence_split": "D_R",
        "read_only": True,
        "catalog_fingerprint": schedule.catalog_fingerprint,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "sequence_fingerprint": schedule.sequence_fingerprint,
        "seed": schedule.seed,
        "schedule": {
            "epochs": PAIRED_EPOCHS,
            "steps_per_epoch": PAIRED_STEPS_PER_EPOCH,
            "optimizer_updates": PAIRED_OPTIMIZER_UPDATES,
            "pairs_per_update": PAIRS_PER_UPDATE,
            "pair_exposures": PAIRED_EXPOSURES,
        },
        "gates": {
            "all_targets_exposed": target["zero_exposure"] == 0,
            "all_sources_exposed": source["zero_exposure"] == 0,
            "maximum_pair_exposure_count_difference": (
                maximum_pair_count_difference
            ),
            "maximum_pair_exposure_count_difference_at_most_one": (
                maximum_pair_count_difference <= 1
            ),
            "source_distinct_violations": source_distinct_violations,
            "every_update_uses_two_distinct_sources": (
                source_distinct_violations == 0
            ),
        },
        "target": target,
        "source_image": source,
        "target_counts": [
            {
                "pair_id": pair.pair_id,
                "sample_id": pair.sample_id,
                "group_id": pair.group_id,
                "evaluation_gt_id": pair.evaluation_gt_id,
                "native_gt_id": pair.native_gt_id,
                "pred_id": pair.pred_id,
                "count": target_counts[pair.pair_id],
            }
            for pair in schedule.pairs
        ],
        "source_counts": [
            {"sample_id": sample_id, "count": source_counts[sample_id]}
            for sample_id in sorted(source_counts)
        ],
        "forbidden_actions_performed": {
            "model_forward": False,
            "optimizer_step": False,
            "training": False,
            "calibration": False,
            "D_V_read": False,
            "D_T_read": False,
        },
    }
    payload["receipt_fingerprint"] = stable_fingerprint(payload)
    return payload


__all__ = [
    "PAIRED_EXPOSURE_SCHEMA",
    "build_paired_exposure_receipt",
]
