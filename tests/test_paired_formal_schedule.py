from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.cache.state_cache import StateCacheRecord
from cure_lite.decoder import project_occupancy_to_feature_grid
from cure_lite.experiment.paired_formal_schedule import (
    DECODER_FORWARDS_PER_UPDATE,
    FACTUAL_MISS_STATES_PER_UPDATE,
    FACTUAL_NO_MISS_STATES_PER_UPDATE,
    FORMAL_METHOD_KINDS,
    PAIRED_ENDPOINT_STATES_PER_UPDATE,
    PAIRED_FORMAL_SCHEDULE_SCHEMA,
    bind_paired_formal_schedule,
    build_paired_formal_schedule,
    build_paired_formal_schedule_from_epoch_pool_builder,
    formal_batches_for_update,
    formal_factual_anchor_id,
    prepared_training_catalog_fingerprint,
)
from cure_lite.experiment.training_pipeline import (
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.instances import instances_from_binary_mask
from cure_lite.intervention import enumerate_legal_deletions
from cure_lite.matching import match_components
from cure_lite.occupancy import build_occupancy
from cure_lite.paired_types import (
    PairCatalog,
    PairExample,
    tensor_content_fingerprint,
)
from cure_lite.sampling import stable_hash
from cure_lite.supervision import build_factual_supervision
from cure_lite.toy import (
    ToyFrozenBaseAdapter,
    make_factual_miss_scene,
    make_two_target_scene,
)
from cure_lite.train.paired_pools import (
    PAIRED_EPOCHS,
    PAIRED_EXPOSURES,
    PAIRED_OPTIMIZER_UPDATES,
    PAIRED_STEPS_PER_EPOCH,
    build_paired_schedule,
)
from cure_lite.train.paired_step import DECODER_STATES_PER_UPDATE
from cure_lite.train.pools import BranchPools, StateExample
from cure_lite.types import BranchSupervision


def _clean_pair(*, sample_id: str, target_id: int) -> PairExample:
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
            "kind": "formal-clean-positive",
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
        completion_plus=torch.zeros_like(valid),
        completion_minus=clean.clone(),
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


def _pair_catalog(source_ids: tuple[str, ...]) -> PairCatalog:
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
        dataset="formal-schedule-toy",
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


def _factual_example(
    *,
    branch: str,
    sample_id: str,
    target_id: int | None,
) -> StateExample:
    feature_value = float(target_id or len(sample_id)) / 10.0
    feature = torch.full((1, 2, 2, 2), feature_value, dtype=torch.float32)
    occupancy = torch.zeros((1, 4, 4), dtype=torch.bool)
    target = torch.zeros((1, 4, 4), dtype=torch.float32)
    if target_id is not None:
        target[0, target_id % 4, (target_id * 3) % 4] = 1.0
    supervision = BranchSupervision(
        occupancy=occupancy,
        target=target,
        valid_mask=torch.ones_like(occupancy),
        branch=branch,
        positive_gt_ids=(() if target_id is None else (target_id,)),
        reachable_gt_ids=(() if target_id is None else (target_id,)),
    )
    return StateExample(sample_id, feature, supervision)


@pytest.fixture(scope="module")
def fake_factual_populations() -> tuple[
    tuple[StateExample, ...],
    tuple[StateExample, ...],
]:
    factual_miss = tuple(
        _factual_example(
            branch="factual_miss",
            sample_id=f"miss-source-{index}",
            target_id=index + 1,
        )
        for index in range(5)
    )
    factual_no_miss = tuple(
        _factual_example(
            branch="factual_no_miss",
            sample_id=f"no-miss-source-{index}",
            target_id=None,
        )
        for index in range(4)
    )
    return factual_miss, factual_no_miss


def _fake_formal_schedule(
    seed: int,
    populations: tuple[
        tuple[StateExample, ...],
        tuple[StateExample, ...],
    ],
):
    factual_miss, factual_no_miss = populations
    paired = build_paired_schedule(
        _pair_catalog(("pair-a", "pair-a", "pair-b", "pair-c")),
        seed=seed,
    )
    return build_paired_formal_schedule_from_epoch_pool_builder(
        paired,
        prepared_catalog_fingerprint="a" * 64,
        expected_factual_miss=factual_miss,
        expected_factual_no_miss=factual_no_miss,
        epoch_pool_builder=lambda _epoch: BranchPools(
            factual_miss=factual_miss,
            factual_no_miss=factual_no_miss,
        ),
    )


@pytest.fixture(scope="module")
def fake_schedule_42(fake_factual_populations):
    return _fake_formal_schedule(42, fake_factual_populations)


def test_full_32k_schedule_binds_pair_and_exact_factual_draws(
    fake_factual_populations,
    fake_schedule_42,
) -> None:
    factual_miss, factual_no_miss = fake_factual_populations
    schedule = fake_schedule_42

    assert PAIRED_FORMAL_SCHEDULE_SCHEMA == (
        "cure-lite-paired-formal-schedule-v1"
    )
    assert PAIRED_EPOCHS == 800
    assert PAIRED_STEPS_PER_EPOCH == 40
    assert PAIRED_OPTIMIZER_UPDATES == 32_000
    assert FACTUAL_MISS_STATES_PER_UPDATE == 4
    assert FACTUAL_NO_MISS_STATES_PER_UPDATE == 4
    assert PAIRED_ENDPOINT_STATES_PER_UPDATE == 4
    assert DECODER_STATES_PER_UPDATE == 12
    assert DECODER_FORWARDS_PER_UPDATE == 3
    assert len(schedule.factual_miss_indices) == 32_000
    assert len(schedule.factual_no_miss_indices) == 32_000
    assert all(len(selected) == 4 for selected in schedule.factual_miss_indices)
    assert all(
        len(selected) == 4 for selected in schedule.factual_no_miss_indices
    )
    assert sum(schedule.pair_exposure_counts) == PAIRED_EXPOSURES
    assert sum(schedule.factual_miss_exposure_counts) == 128_000
    assert sum(schedule.factual_no_miss_exposure_counts) == 128_000
    assert min(schedule.pair_exposure_counts) > 0
    assert min(schedule.factual_miss_exposure_counts) > 0
    assert min(schedule.factual_no_miss_exposure_counts) > 0
    assert schedule.decoder_state_evaluations == 384_000
    assert schedule.decoder_forward_calls == 96_000
    assert sum(row.total_exposures for row in schedule.source_exposure_ledger) == (
        64_000 + 128_000 + 128_000
    )

    epoch, step = 17, 9
    actual = schedule.factual_examples_for_update(epoch=epoch, step=step)
    expected_miss = tuple(
        factual_miss[
            stable_hash("factual_miss", epoch, step, draw, 42)
            % len(factual_miss)
        ]
        for draw in range(4)
    )
    expected_no_miss = tuple(
        factual_no_miss[
            stable_hash("factual_no_miss", epoch, step, draw, 42)
            % len(factual_no_miss)
        ]
        for draw in range(4)
    )
    assert actual["factual_miss"] == expected_miss
    assert actual["factual_no_miss"] == expected_no_miss
    assert tuple(
        formal_factual_anchor_id("factual_miss", item)
        for item in actual["factual_miss"]
    ) == tuple(
        formal_factual_anchor_id("factual_miss", item)
        for item in expected_miss
    )

    factual_batches, pair_batch = formal_batches_for_update(
        schedule,
        epoch=epoch,
        step=step,
        device="cpu",
    )
    assert tuple(factual_batches) == (
        "factual_miss",
        "factual_no_miss",
    )
    assert factual_batches["factual_miss"].feature.shape[0] == 4
    assert factual_batches["factual_no_miss"].feature.shape[0] == 4
    assert pair_batch.feature.shape[0] == 2
    assert pair_batch.pair_ids == tuple(
        pair.pair_id
        for pair in schedule.paired_schedule.batch_examples(
            epoch=epoch,
            step=step,
        )
    )


def test_schedule_is_deterministic_and_seed_specific(
    fake_factual_populations,
    fake_schedule_42,
) -> None:
    replay = _fake_formal_schedule(42, fake_factual_populations)
    other_seed = _fake_formal_schedule(43, fake_factual_populations)
    assert replay.schedule_fingerprint == fake_schedule_42.schedule_fingerprint
    assert replay.combined_sequence_fingerprint == (
        fake_schedule_42.combined_sequence_fingerprint
    )
    assert replay.factual_miss_indices == fake_schedule_42.factual_miss_indices
    assert replay.factual_no_miss_indices == (
        fake_schedule_42.factual_no_miss_indices
    )
    assert other_seed.schedule_fingerprint != (
        fake_schedule_42.schedule_fingerprint
    )
    assert other_seed.combined_sequence_fingerprint != (
        fake_schedule_42.combined_sequence_fingerprint
    )


def test_schedule_rejects_sequence_ledger_and_fingerprint_tampering(
    fake_schedule_42,
) -> None:
    schedule = fake_schedule_42
    changed = list(schedule.factual_miss_indices)
    differing = next(
        index
        for index in range(1, len(changed))
        if changed[index] != changed[0]
    )
    changed[0], changed[differing] = changed[differing], changed[0]
    with pytest.raises(
        ValueError,
        match="factual-miss sequence fingerprint",
    ):
        replace(schedule, factual_miss_indices=tuple(changed))

    invalid = (
        (len(schedule.factual_no_miss_anchors),) * 4,
        *schedule.factual_no_miss_indices[1:],
    )
    with pytest.raises(ValueError, match="invalid anchor index"):
        replace(schedule, factual_no_miss_indices=invalid)

    altered_counts = (
        schedule.factual_miss_exposure_counts[0] + 1,
        *schedule.factual_miss_exposure_counts[1:],
    )
    with pytest.raises(ValueError, match="exposure ledger"):
        replace(
            schedule,
            factual_miss_exposure_counts=altered_counts,
        )

    with pytest.raises(ValueError, match="schedule_fingerprint"):
        replace(schedule, schedule_fingerprint="0" * 64)


def test_proposed_and_all_controls_reuse_the_same_schedule_object(
    fake_schedule_42,
) -> None:
    bindings = tuple(
        bind_paired_formal_schedule(
            fake_schedule_42,
            method_kind=method_kind,
        )
        for method_kind in FORMAL_METHOD_KINDS
    )
    assert FORMAL_METHOD_KINDS[0] == "paired_difference"
    assert all(binding.schedule is fake_schedule_42 for binding in bindings)
    assert {
        binding.shared_schedule_fingerprint for binding in bindings
    } == {fake_schedule_42.schedule_fingerprint}
    assert all(
        binding.canonical_payload()["method_label_affects_schedule"] is False
        for binding in bindings
    )
    assert len({binding.binding_fingerprint for binding in bindings}) == len(
        FORMAL_METHOD_KINDS
    )
    with pytest.raises(ValueError, match="method_kind"):
        bind_paired_formal_schedule(
            fake_schedule_42,
            method_kind="not-a-formal-method",
        )


def _pairs(rows) -> torch.Tensor:
    return torch.tensor(
        [[row.gt_id, row.pred_id] for row in rows],
        dtype=torch.int64,
    ).reshape(-1, 2)


def _cached_source(scene) -> CachedTrainingSource:
    output = ToyFrozenBaseAdapter()(scene.image_batch())
    occupancy, pred = build_occupancy(output.probability)
    gt = instances_from_binary_mask(scene.gt_mask)
    match = match_components(pred, gt)
    factual = build_factual_supervision(occupancy, gt, match)
    legal = enumerate_legal_deletions(pred, gt, match, occupancy)
    image_valid_mask = torch.ones_like(occupancy)
    state = StateCacheRecord(
        sample_id=scene.sample_id,
        occupancy=occupancy,
        pred_labels=pred.labels,
        gt_labels=gt.labels,
        base_match_pairs=_pairs(match.pairs),
        real_miss_ids=torch.tensor(
            sorted(match.unmatched_gt_ids),
            dtype=torch.int64,
        ),
        reachable_miss_ids=torch.tensor(
            factual.reachable_gt_ids,
            dtype=torch.int64,
        ),
        legal_pairs=_pairs(legal),
        image_valid_mask=image_valid_mask,
    )
    return CachedTrainingSource(
        scene.sample_id,
        output.feature,
        output.probability,
        state,
    )


def test_production_builder_uses_prepared_catalog_epoch_pools() -> None:
    sources = (
        _cached_source(make_factual_miss_scene(missed_gt_id=2)),
        _cached_source(make_two_target_scene()),
        _cached_source(make_factual_miss_scene(missed_gt_id=1)),
    )
    prepared = prepare_training_catalog(sources)
    paired = build_paired_schedule(
        _pair_catalog(tuple(source.sample_id for source in sources)),
        seed=42,
    )
    schedule = build_paired_formal_schedule(paired, prepared)

    assert schedule.prepared_catalog_fingerprint == (
        prepared_training_catalog_fingerprint(prepared)
    )
    expected_miss_ids = {
        formal_factual_anchor_id("factual_miss", example)
        for entry in prepared.entries
        for example in entry.factual_examples
    }
    expected_no_miss_ids = {
        formal_factual_anchor_id(
            "factual_no_miss",
            entry.factual_no_miss_example,
        )
        for entry in prepared.entries
        if entry.factual_no_miss_example is not None
    }
    assert {anchor.anchor_id for anchor in schedule.factual_miss_anchors} == (
        expected_miss_ids
    )
    assert {
        anchor.anchor_id for anchor in schedule.factual_no_miss_anchors
    } == expected_no_miss_ids
    assert min(schedule.factual_miss_exposure_counts) > 0
    assert min(schedule.factual_no_miss_exposure_counts) > 0
