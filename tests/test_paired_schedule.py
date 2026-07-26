from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.decoder import project_occupancy_to_feature_grid
from cure_lite.experiment.paired_exposure import (
    PAIRED_EXPOSURE_SCHEMA,
    build_paired_exposure_receipt,
)
from cure_lite.paired_types import (
    PairCatalog,
    PairExample,
    tensor_content_fingerprint,
)
from cure_lite.train.paired_pools import (
    PAIRED_EPOCHS,
    PAIRED_EXPOSURES,
    PAIRED_OPTIMIZER_UPDATES,
    PAIRED_SCHEDULE_SCHEMA,
    PAIRED_STEPS_PER_EPOCH,
    PAIRS_PER_UPDATE,
    build_paired_schedule,
    iter_paired_batches,
    pair_batch_for_update,
)


def _clean_pair(
    *,
    sample_id: str,
    target_id: int,
) -> PairExample:
    feature = torch.full(
        (1, 2, 2, 2),
        float(target_id) / 10.0,
        dtype=torch.float32,
    )
    valid = torch.ones((1, 4, 4), dtype=torch.bool)
    plus = torch.zeros_like(valid)
    row = target_id % 4
    column = (target_id * 3) % 4
    plus[0, row, column] = True
    minus = torch.zeros_like(plus)
    clean = torch.zeros_like(valid)
    clean[0, row, column] = True
    clean[0, (row + 1) % 4, column] = True
    completion_plus = torch.zeros_like(valid)
    completion_minus = clean.clone()
    projected_plus = project_occupancy_to_feature_grid(
        plus.unsqueeze(0),
        (2, 2),
    )
    projected_minus = project_occupancy_to_feature_grid(
        minus.unsqueeze(0),
        (2, 2),
    )
    pair_id = stable_fingerprint(
        {
            "kind": "clean_positive",
            "sample_id": sample_id,
            "target_id": target_id,
        }
    )
    return PairExample(
        pair_id=pair_id,
        pair_kind="clean_positive",
        sample_id=sample_id,
        group_id=f"group-{sample_id}",
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        removed_component=plus.clone(),
        image_valid_mask=valid,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        label_increment=clean.to(torch.float32),
        clean_increment=clean,
        evaluation_gt_id=target_id,
        native_gt_id=target_id,
        pred_id=target_id,
        feature_fingerprint=tensor_content_fingerprint(feature),
        before_match_fingerprint=stable_fingerprint(
            {"sample_id": sample_id, "target_id": target_id, "state": "before"}
        ),
        after_match_fingerprint=stable_fingerprint(
            {"sample_id": sample_id, "target_id": target_id, "state": "after"}
        ),
        projected_occupancy_plus_fingerprint=tensor_content_fingerprint(
            projected_plus
        ),
        projected_occupancy_minus_fingerprint=tensor_content_fingerprint(
            projected_minus
        ),
        projection_visible=True,
        geometry_safe_bijective_lineage=True,
        selected_gt_is_only_new_unmatched=True,
        other_match_identities_unchanged=True,
        preexisting_unmatched_gt_noninterference=True,
    )


def _catalog(source_ids: tuple[str, ...]) -> PairCatalog:
    pairs = tuple(
        sorted(
            (
                _clean_pair(sample_id=sample_id, target_id=index + 1)
                for index, sample_id in enumerate(source_ids)
            ),
            key=lambda pair: (
                pair.sample_id,
                int(pair.evaluation_gt_id),
                int(pair.pred_id),
                pair.pair_id,
            ),
        )
    )
    return PairCatalog(
        dataset="paired-toy",
        split="D_R",
        paired_protocol_fingerprint="1" * 64,
        geometry_catalog_fingerprint="2" * 64,
        source_catalog_fingerprint="3" * 64,
        manifest_fingerprint="4" * 64,
        clean_positive=pairs,
        component_null=(),
        identity_null=(),
        exclusions=(),
        catalog_fingerprint="5" * 64,
    )


def test_schedule_freezes_the_complete_800_by_40_by_2_budget() -> None:
    assert PAIRED_EPOCHS == 800
    assert PAIRED_STEPS_PER_EPOCH == 40
    assert PAIRS_PER_UPDATE == 2
    assert PAIRED_OPTIMIZER_UPDATES == 32_000
    assert PAIRED_EXPOSURES == 64_000

    schedule = build_paired_schedule(
        _catalog(("source-a", "source-a", "source-b", "source-c")),
        seed=42,
    )
    assert schedule.optimizer_updates == 32_000
    assert schedule.exposures == 64_000
    assert len(schedule.batch_pair_indices) == 32_000
    assert sum(schedule.canonical_cycle_counts) == 64_000
    assert max(schedule.canonical_cycle_counts) - min(
        schedule.canonical_cycle_counts
    ) <= 1
    assert all(
        schedule.pairs[first].sample_id
        != schedule.pairs[second].sample_id
        for first, second in schedule.batch_pair_indices
    )


def test_canonical_cycle_preserves_support_when_division_has_remainder() -> None:
    schedule = build_paired_schedule(
        _catalog(("source-a", "source-b", "source-c")),
        seed=43,
    )
    assert sorted(schedule.canonical_cycle_counts) == [21_333, 21_333, 21_334]
    assert all(count > 0 for count in schedule.canonical_cycle_counts)


def test_schedule_is_exactly_repeatable_and_seed_specific() -> None:
    catalog = _catalog(("source-a", "source-a", "source-b", "source-c"))
    first = build_paired_schedule(catalog, seed=42)
    replay = build_paired_schedule(catalog, seed=42)
    other_seed = build_paired_schedule(catalog, seed=43)
    assert first.schedule_fingerprint == replay.schedule_fingerprint
    assert first.sequence_fingerprint == replay.sequence_fingerprint
    assert first.batch_pair_indices == replay.batch_pair_indices
    assert first.schedule_fingerprint != other_seed.schedule_fingerprint


def test_pair_batch_materialization_stores_one_feature_per_pair() -> None:
    schedule = build_paired_schedule(
        _catalog(("source-a", "source-a", "source-b", "source-c")),
        seed=42,
    )
    batch = pair_batch_for_update(
        schedule,
        epoch=0,
        step=0,
        device="cpu",
    )
    batch.validate()
    assert tuple(batch.feature.shape) == (2, 2, 2, 2)
    assert tuple(batch.occupancy_plus.shape) == (2, 1, 4, 4)
    assert len(set(batch.sample_ids)) == 2
    assert batch.pair_kinds == ("clean_positive", "clean_positive")

    epoch_batches = tuple(
        iter_paired_batches(schedule, epoch=799, device="cpu")
    )
    assert len(epoch_batches) == 40
    assert all(len(set(value.sample_ids)) == 2 for value in epoch_batches)


def test_schedule_fails_explicitly_when_distinct_sources_are_impossible() -> None:
    with pytest.raises(
        RuntimeError,
        match="at least two distinct source images",
    ):
        build_paired_schedule(
            _catalog(("source-a", "source-a")),
            seed=42,
        )

    with pytest.raises(
        RuntimeError,
        match="more than half",
    ):
        build_paired_schedule(
            _catalog(("source-a", "source-a", "source-a", "source-b")),
            seed=42,
        )


def test_exposure_receipt_accounts_for_every_target_and_source() -> None:
    schedule = build_paired_schedule(
        _catalog(("source-a", "source-a", "source-b", "source-c")),
        seed=42,
    )
    receipt = build_paired_exposure_receipt(schedule)
    assert receipt["schema_version"] == PAIRED_EXPOSURE_SCHEMA
    assert receipt["evidence_split"] == "D_R"
    assert receipt["read_only"] is True
    assert receipt["schedule"] == {
        "epochs": 800,
        "steps_per_epoch": 40,
        "optimizer_updates": 32_000,
        "pairs_per_update": 2,
        "pair_exposures": 64_000,
    }
    assert receipt["gates"] == {
        "all_targets_exposed": True,
        "all_sources_exposed": True,
        "maximum_pair_exposure_count_difference": 0,
        "maximum_pair_exposure_count_difference_at_most_one": True,
        "source_distinct_violations": 0,
        "every_update_uses_two_distinct_sources": True,
    }
    target = receipt["target"]
    assert target["population"] == 4
    assert target["total_exposures"] == 64_000
    assert target["zero_exposure"] == 0
    assert target["ess"] == pytest.approx(4.0)
    assert target["maximum_share"] == pytest.approx(0.25)
    assert target["top1_concentration"]["share"] == pytest.approx(0.25)
    source = receipt["source_image"]
    assert source["population"] == 3
    assert source["total_exposures"] == 64_000
    assert source["zero_exposure"] == 0
    assert source["maximum_share"] == pytest.approx(0.5)
    assert source["ess"] == pytest.approx(8.0 / 3.0)
    assert len(receipt["target_counts"]) == 4
    assert len(receipt["source_counts"]) == 3
    assert receipt["forbidden_actions_performed"] == {
        "model_forward": False,
        "optimizer_step": False,
        "training": False,
        "calibration": False,
        "D_V_read": False,
        "D_T_read": False,
    }

    canonical = dict(receipt)
    fingerprint = canonical.pop("receipt_fingerprint")
    assert fingerprint == stable_fingerprint(canonical)
    assert json.dumps(receipt, sort_keys=True) == json.dumps(
        build_paired_exposure_receipt(schedule),
        sort_keys=True,
    )


def test_schedule_rejects_tampering_with_a_frozen_plan() -> None:
    schedule = build_paired_schedule(
        _catalog(("source-a", "source-a", "source-b", "source-c")),
        seed=42,
    )
    duplicated = (
        (schedule.batch_pair_indices[0][0],) * 2,
        *schedule.batch_pair_indices[1:],
    )
    with pytest.raises(ValueError, match="canonical-cycle allocation|duplicate source"):
        replace(schedule, batch_pair_indices=duplicated)


def test_schema_names_are_pair_route_specific() -> None:
    assert PAIRED_SCHEDULE_SCHEMA == "cure-lite-clean-pair-schedule-v1"
    assert PAIRED_EXPOSURE_SCHEMA == "cure-lite-clean-pair-exposure-v1"
