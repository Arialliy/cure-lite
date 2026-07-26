from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.decoder import project_occupancy_to_feature_grid
from cure_lite.experiment.paired_outcome_schedule import (
    PAIRED_OUTCOME_SCHEDULE_SCHEMA,
    build_outcome_pair_schedule,
)
from cure_lite.paired_types import (
    PairCatalog,
    PairExample,
    tensor_content_fingerprint,
)


def _pair(
    *,
    pair_kind: str,
    index: int,
    sample_id: str,
) -> PairExample:
    feature = torch.full(
        (1, 2, 2, 2),
        float(index + 1) / 1000.0,
        dtype=torch.float32,
    )
    valid = torch.ones((1, 4, 4), dtype=torch.bool)
    plus = torch.zeros_like(valid)
    row = index % 4
    column = (index // 4) % 4
    plus[0, row, column] = True
    minus = torch.zeros_like(plus)
    projected_plus = project_occupancy_to_feature_grid(
        plus.unsqueeze(0),
        (2, 2),
    )
    projected_minus = project_occupancy_to_feature_grid(
        minus.unsqueeze(0),
        (2, 2),
    )
    empty = torch.zeros_like(valid)
    if pair_kind == "clean_positive":
        increment = empty.clone()
        increment[0, row, column] = True
        increment[0, (row + 1) % 4, column] = True
        completion_plus = empty
        completion_minus = increment
        evaluation_gt_id = index + 1
        native_gt_id = index + 1
        pred_id = index + 1
        clean_checks: tuple[bool | None, ...] = (True, True, True, True)
    elif pair_kind == "component_null":
        increment = empty
        completion_plus = empty
        completion_minus = empty
        evaluation_gt_id = None
        native_gt_id = None
        pred_id = index + 1
        clean_checks = (None, None, None, None)
    else:
        raise ValueError("test pair kind is invalid")
    pair_id = stable_fingerprint(
        {
            "pair_kind": pair_kind,
            "index": index,
            "sample_id": sample_id,
        }
    )
    return PairExample(
        pair_id=pair_id,
        pair_kind=pair_kind,
        sample_id=sample_id,
        group_id=f"group-{sample_id}",
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        removed_component=plus.clone(),
        image_valid_mask=valid,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        label_increment=increment.to(torch.float32),
        clean_increment=increment,
        evaluation_gt_id=evaluation_gt_id,
        native_gt_id=native_gt_id,
        pred_id=pred_id,
        feature_fingerprint=tensor_content_fingerprint(feature),
        before_match_fingerprint=stable_fingerprint(
            {"pair_id": pair_id, "endpoint": "plus"}
        ),
        after_match_fingerprint=stable_fingerprint(
            {"pair_id": pair_id, "endpoint": "minus"}
        ),
        projected_occupancy_plus_fingerprint=tensor_content_fingerprint(
            projected_plus
        ),
        projected_occupancy_minus_fingerprint=tensor_content_fingerprint(
            projected_minus
        ),
        projection_visible=True,
        geometry_safe_bijective_lineage=clean_checks[0],
        selected_gt_is_only_new_unmatched=clean_checks[1],
        other_match_identities_unchanged=clean_checks[2],
        preexisting_unmatched_gt_noninterference=clean_checks[3],
    )


def _catalog(
    *,
    clean_sources: tuple[str, ...],
    component_sources: tuple[str, ...],
) -> PairCatalog:
    clean = tuple(
        sorted(
            (
                _pair(
                    pair_kind="clean_positive",
                    index=index,
                    sample_id=sample_id,
                )
                for index, sample_id in enumerate(clean_sources)
            ),
            key=lambda pair: (
                pair.sample_id,
                int(pair.evaluation_gt_id),
                int(pair.pred_id),
                pair.pair_id,
            ),
        )
    )
    offset = len(clean_sources)
    component = tuple(
        sorted(
            (
                _pair(
                    pair_kind="component_null",
                    index=offset + index,
                    sample_id=sample_id,
                )
                for index, sample_id in enumerate(component_sources)
            ),
            key=lambda pair: (
                pair.sample_id,
                -1,
                int(pair.pred_id),
                pair.pair_id,
            ),
        )
    )
    unsealed = PairCatalog(
        dataset="outcome-toy",
        split="D_R",
        paired_protocol_fingerprint="1" * 64,
        geometry_catalog_fingerprint="2" * 64,
        source_catalog_fingerprint="3" * 64,
        manifest_fingerprint="4" * 64,
        clean_positive=clean,
        component_null=component,
        identity_null=(),
        exclusions=(),
        catalog_fingerprint="",
    )
    return replace(
        unsealed,
        catalog_fingerprint=stable_fingerprint(
            unsealed.canonical_payload()
        ),
    )


def test_real_population_bounded_and_formal_exposures_are_pair_uniform() -> None:
    catalog = _catalog(
        clean_sources=tuple(f"clean-{index:03d}" for index in range(206)),
        component_sources=tuple(
            f"component-{index:03d}" for index in range(16)
        ),
    )
    bounded = build_outcome_pair_schedule(
        catalog,
        seed=42,
        optimizer_updates=400,
        steps_per_epoch=40,
    )
    assert bounded.epochs == 10
    assert bounded.exposures == 800
    assert len(bounded.pairs) == 222
    assert sorted(set(bounded.pair_exposure_counts)) == [3, 4]
    assert bounded.pair_exposure_counts.count(4) == 134
    assert bounded.pair_exposure_counts.count(3) == 88

    formal = build_outcome_pair_schedule(
        catalog,
        seed=42,
        optimizer_updates=32_000,
        steps_per_epoch=40,
    )
    assert formal.epochs == 800
    assert formal.exposures == 64_000
    assert sorted(set(formal.pair_exposure_counts)) == [288, 289]
    assert formal.pair_exposure_counts.count(289) == 64
    assert formal.pair_exposure_counts.count(288) == 158
    assert all(
        formal.pairs[first].sample_id != formal.pairs[second].sample_id
        for first, second in formal.batch_pair_indices
    )
    clean_counts = [
        count
        for pair, count in zip(
            formal.pairs,
            formal.pair_exposure_counts,
            strict=True,
        )
        if pair.pair_kind == "clean_positive"
    ]
    null_counts = [
        count
        for pair, count in zip(
            formal.pairs,
            formal.pair_exposure_counts,
            strict=True,
        )
        if pair.pair_kind == "component_null"
    ]
    assert max(clean_counts + null_counts) - min(clean_counts + null_counts) == 1


def test_outcome_schedule_is_repeatable_seed_specific_and_tensor_free() -> None:
    catalog = _catalog(
        clean_sources=("a", "a", "b", "c"),
        component_sources=("d", "e"),
    )
    first = build_outcome_pair_schedule(
        catalog,
        seed=42,
        optimizer_updates=40,
        steps_per_epoch=40,
    )
    replay = build_outcome_pair_schedule(
        catalog,
        seed=42,
        optimizer_updates=40,
        steps_per_epoch=40,
    )
    other = build_outcome_pair_schedule(
        catalog,
        seed=43,
        optimizer_updates=40,
        steps_per_epoch=40,
    )
    assert first.batch_pair_indices == replay.batch_pair_indices
    assert first.schedule_fingerprint == replay.schedule_fingerprint
    assert first.sequence_fingerprint == replay.sequence_fingerprint
    assert first.schedule_fingerprint != other.schedule_fingerprint
    receipt = first.canonical_receipt()
    assert receipt["schema_version"] == PAIRED_OUTCOME_SCHEDULE_SCHEMA
    assert receipt["population_counts"] == {
        "clean_positive": 4,
        "component_null": 2,
        "outcome_union": 6,
    }
    assert receipt["selection_contract"]["clean_null_stratified_sampling"] is False
    assert "tensor(" not in repr(receipt)


def test_outcome_schedule_rejects_infeasible_source_packing_and_tampering() -> None:
    catalog = _catalog(
        clean_sources=("a", "a", "a"),
        component_sources=("b",),
    )
    with pytest.raises(RuntimeError, match="more than half"):
        build_outcome_pair_schedule(
            catalog,
            seed=42,
            optimizer_updates=40,
            steps_per_epoch=40,
        )

    valid = build_outcome_pair_schedule(
        _catalog(
            clean_sources=("a", "b", "c"),
            component_sources=("d",),
        ),
        seed=42,
        optimizer_updates=40,
        steps_per_epoch=40,
    )
    duplicated = (
        (valid.batch_pair_indices[0][0], valid.batch_pair_indices[0][0]),
        *valid.batch_pair_indices[1:],
    )
    with pytest.raises(ValueError, match="balanced allocation|duplicate sources"):
        replace(valid, batch_pair_indices=duplicated)


def test_outcome_schedule_requires_complete_outcome_population_and_full_coverage() -> None:
    clean_only = _catalog(
        clean_sources=("a", "b"),
        component_sources=(),
    )
    with pytest.raises(RuntimeError, match="non-empty clean and component-null"):
        build_outcome_pair_schedule(
            clean_only,
            seed=42,
            optimizer_updates=40,
            steps_per_epoch=40,
        )
    too_small_budget = _catalog(
        clean_sources=tuple(f"c-{index}" for index in range(8)),
        component_sources=("n",),
    )
    with pytest.raises(RuntimeError, match="cannot expose every"):
        build_outcome_pair_schedule(
            too_small_budget,
            seed=42,
            optimizer_updates=4,
            steps_per_epoch=4,
        )


def test_outcome_schedule_rejects_stale_catalog_fingerprint() -> None:
    catalog = _catalog(
        clean_sources=("a", "b"),
        component_sources=("c",),
    )
    changed = replace(catalog, catalog_fingerprint="0" * 64)

    with pytest.raises(
        RuntimeError,
        match="catalog fingerprint does not reproduce",
    ):
        build_outcome_pair_schedule(
            changed,
            seed=42,
            optimizer_updates=40,
            steps_per_epoch=40,
        )
