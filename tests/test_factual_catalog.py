from __future__ import annotations

import torch

from cure_lite.cache.schema import STATE_CACHE_SCHEMA
from cure_lite.cache.state_cache import (
    StateCacheRecord,
    load_state_cache,
    save_state_cache,
)
from cure_lite.instances import instances_from_binary_mask
from cure_lite.matching import match_components
from cure_lite.sampling import choose_uniform_factual_gt_id
from cure_lite.supervision import (
    build_epoch_factual_supervision_from_catalog,
    build_factual_supervision,
    build_factual_supervision_from_catalog,
)


def _multi_reachable_source():
    gt_mask = torch.tensor(
        [
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    occupancy = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    pred = instances_from_binary_mask(occupancy)
    gt = instances_from_binary_mask(gt_mask)
    match = match_components(pred, gt)
    oracle = build_factual_supervision(occupancy, gt, match)
    assert oracle.reachable_gt_ids == (1, 4)
    return occupancy, pred, gt, match, oracle


def test_factual_selector_is_deterministic_atomic_and_variant_independent() -> None:
    catalog = (1, 4)
    first = tuple(
        choose_uniform_factual_gt_id(
            catalog,
            sample_id="scene-a",
            epoch=epoch,
            global_seed=23,
        )
        for epoch in range(64)
    )
    repeated = tuple(
        choose_uniform_factual_gt_id(
            catalog,
            sample_id="scene-a",
            epoch=epoch,
            global_seed=23,
        )
        for epoch in range(64)
    )
    assert first == repeated
    assert set(first) == {1, 4}
    assert choose_uniform_factual_gt_id(
        (), sample_id="scene-a", epoch=0, global_seed=23
    ) is None


def test_factual_selector_covers_each_catalog_member_once_per_cycle() -> None:
    catalog = (2, 5, 9, 13)
    cycle = tuple(
        choose_uniform_factual_gt_id(
            catalog,
            sample_id="scene-cycle",
            epoch=epoch,
            global_seed=41,
        )
        for epoch in range(len(catalog))
    )
    assert len(set(cycle)) == len(catalog)
    assert set(cycle) == set(catalog)

    repeated_cycle = tuple(
        choose_uniform_factual_gt_id(
            catalog,
            sample_id="scene-cycle",
            epoch=epoch,
            global_seed=41,
        )
        for epoch in range(len(catalog), 2 * len(catalog))
    )
    assert repeated_cycle == cycle

    # The guarantee also holds for a contiguous cycle crossing the wrap point.
    wrapped = tuple(
        choose_uniform_factual_gt_id(
            catalog,
            sample_id="scene-cycle",
            epoch=epoch,
            global_seed=41,
        )
        for epoch in range(2, 2 + len(catalog))
    )
    assert set(wrapped) == set(catalog)


def test_catalog_materializer_keeps_all_metadata_but_one_positive() -> None:
    occupancy, _, gt, match, oracle = _multi_reachable_source()
    selected = build_factual_supervision_from_catalog(
        occupancy,
        gt,
        real_miss_ids=tuple(sorted(match.unmatched_gt_ids)),
        reachable_gt_ids=oracle.reachable_gt_ids,
        selected_gt_id=4,
    )
    assert selected.positive_gt_ids == (4,)
    assert selected.reachable_gt_ids == (1, 4)
    assert selected.unreachable_gt_ids == ()
    assert torch.equal(
        selected.target[0].to(torch.bool),
        gt.by_id(4).mask & ~occupancy,
    )
    assert not torch.any(selected.valid_mask[0] & gt.by_id(1).mask)


def test_epoch_materializer_is_atomic_and_preserves_full_catalog() -> None:
    occupancy, _, gt, match, oracle = _multi_reachable_source()
    real_miss_ids = tuple(sorted(match.unmatched_gt_ids))
    states = tuple(
        build_epoch_factual_supervision_from_catalog(
            occupancy,
            gt,
            real_miss_ids=real_miss_ids,
            reachable_gt_ids=oracle.reachable_gt_ids,
            sample_id="multi-reachable",
            epoch=epoch,
            global_seed=17,
        )
        for epoch in range(len(oracle.reachable_gt_ids))
    )

    assert {state.positive_gt_ids[0] for state in states} == set(
        oracle.reachable_gt_ids
    )
    for epoch, state in enumerate(states):
        expected_gt_id = choose_uniform_factual_gt_id(
            oracle.reachable_gt_ids,
            sample_id="multi-reachable",
            epoch=epoch,
            global_seed=17,
        )
        assert state.branch == "factual_miss"
        assert state.positive_gt_ids == (expected_gt_id,)
        assert state.reachable_gt_ids == oracle.reachable_gt_ids
        assert state.unreachable_gt_ids == ()
        assert torch.equal(
            state.target[0].to(torch.bool),
            gt.by_id(expected_gt_id).mask & ~occupancy,
        )
        for unselected_gt_id in set(oracle.reachable_gt_ids) - {expected_gt_id}:
            assert not torch.any(
                state.valid_mask[0] & gt.by_id(unselected_gt_id).mask
            )


def test_state_cache_round_trip_preserves_complete_reachable_catalog(tmp_path) -> None:
    occupancy, pred, gt, match, oracle = _multi_reachable_source()
    record = StateCacheRecord(
        sample_id="multi-reachable",
        occupancy=occupancy,
        pred_labels=pred.labels,
        gt_labels=gt.labels,
        base_match_pairs=torch.tensor(
            [[pair.gt_id, pair.pred_id] for pair in match.pairs],
            dtype=torch.int64,
        ),
        real_miss_ids=torch.tensor(sorted(match.unmatched_gt_ids)),
        reachable_miss_ids=torch.tensor(oracle.reachable_gt_ids),
        legal_pairs=torch.empty((0, 2), dtype=torch.int64),
        image_valid_mask=torch.ones_like(occupancy),
    )
    path = tmp_path / "state.npz"
    fingerprint = "f" * 64
    save_state_cache(path, record, fingerprint=fingerprint)
    loaded = load_state_cache(
        path,
        expected_fingerprint=fingerprint,
        expected_sample_id="multi-reachable",
    )
    assert STATE_CACHE_SCHEMA == "cure-lite-state-cache-v3"
    assert loaded.reachable_miss_ids.tolist() == [1, 4]
