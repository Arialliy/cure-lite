from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from cure_lite.cache.state_cache import StateCacheRecord
from cure_lite.decoder import CURELiteDecoder
from cure_lite.experiment.training_pipeline import (
    CachedTrainingSource,
    TrainingSupportRequirements,
    build_epoch_branch_pools,
    require_training_branch_support,
    run_fixed_training,
    summarize_training_support,
)
from cure_lite.experiment.cache_pipeline import build_state_record
from cure_lite.instances import instances_from_binary_mask
from cure_lite.intervention import enumerate_legal_deletions
from cure_lite.losses import CURELiteLoss
from cure_lite.matching import match_components
from cure_lite.occupancy import build_occupancy
from cure_lite.sampling import choose_uniform_legal_deletion
from cure_lite.toy import (
    ToyFrozenBaseAdapter,
    make_factual_miss_scene,
    make_two_target_scene,
)
from cure_lite.supervision import build_factual_supervision
from cure_lite.types import FrozenBaseOutput


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
    image_valid_mask[0] = False
    image_valid_mask[-1] = False
    image_valid_mask[:, 0] = False
    image_valid_mask[:, -1] = False
    state = StateCacheRecord(
        sample_id=scene.sample_id,
        occupancy=occupancy,
        pred_labels=pred.labels,
        gt_labels=gt.labels,
        base_match_pairs=_pairs(match.pairs),
        real_miss_ids=torch.tensor(
            sorted(match.unmatched_gt_ids), dtype=torch.int64
        ),
        reachable_miss_ids=torch.tensor(
            factual.reachable_gt_ids, dtype=torch.int64
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


def _sources() -> tuple[CachedTrainingSource, CachedTrainingSource]:
    return (
        _cached_source(make_factual_miss_scene(missed_gt_id=1)),
        _cached_source(make_two_target_scene()),
    )


def test_epoch_pools_share_factual_schedule_and_only_u_has_synthetic() -> None:
    sources = _sources()
    factual = build_epoch_branch_pools(
        sources,
        variant="factual_only",
        epoch=3,
        global_seed=19,
    )
    uniform = build_epoch_branch_pools(
        sources,
        variant="uniform_legal",
        epoch=3,
        global_seed=19,
    )

    assert len(factual.factual_miss) == len(uniform.factual_miss) == 1
    assert len(factual.factual_no_miss) == len(uniform.factual_no_miss) == 1
    assert factual.synthetic == ()
    assert len(uniform.synthetic) == 2

    factual_state = factual.factual_miss[0].supervision
    uniform_state = uniform.factual_miss[0].supervision
    assert factual_state.positive_gt_ids == uniform_state.positive_gt_ids == (1,)
    assert factual_state.reachable_gt_ids == uniform_state.reachable_gt_ids == (1,)
    assert factual_state.unreachable_gt_ids == uniform_state.unreachable_gt_ids == ()
    assert torch.equal(factual_state.target, uniform_state.target)
    assert not torch.any(factual_state.valid_mask[:, 0])
    assert not torch.any(factual_state.valid_mask[:, -1])

    for example in uniform.synthetic:
        assert example.supervision.branch == "synthetic"
        assert len(example.supervision.positive_gt_ids) == 1
        assert example.supervision.reachable_gt_ids == ()


def test_uniform_legal_selector_covers_each_candidate_once_per_cycle() -> None:
    scene = make_two_target_scene()
    output = ToyFrozenBaseAdapter()(scene.image_batch())
    occupancy, prediction = build_occupancy(output.probability)
    gt = instances_from_binary_mask(scene.gt_mask)
    matching = match_components(prediction, gt)
    legal = enumerate_legal_deletions(prediction, gt, matching, occupancy)
    assert len(legal) == 2

    def identity(epoch: int) -> tuple[int, int]:
        selected = choose_uniform_legal_deletion(
            legal,
            sample_id=scene.sample_id,
            epoch=epoch,
            global_seed=23,
        )
        assert selected is not None
        return selected.gt_id, selected.pred_id

    expected = {(item.gt_id, item.pred_id) for item in legal}
    cycle = tuple(identity(epoch) for epoch in range(len(legal)))
    repeated = tuple(identity(epoch + len(legal)) for epoch in range(len(legal)))
    crossing = tuple(identity(epoch) for epoch in range(1, 1 + len(legal)))

    assert len(set(cycle)) == len(legal)
    assert set(cycle) == expected
    assert repeated == cycle
    assert set(crossing) == expected


def test_epoch_pool_rejects_probability_or_legal_catalog_drift() -> None:
    factual_source, covered_source = _sources()
    changed_probability = factual_source.probability.clone()
    changed_probability[0, 0, 1, 1] = 0.9
    probability_drift = CachedTrainingSource(
        factual_source.sample_id,
        factual_source.feature,
        changed_probability,
        factual_source.state,
    )
    with pytest.raises(RuntimeError, match="cached occupancy disagrees"):
        build_epoch_branch_pools(
            (probability_drift,),
            variant="factual_only",
            epoch=0,
            global_seed=0,
        )

    missing_legal = replace(
        covered_source.state,
        legal_pairs=torch.empty((0, 2), dtype=torch.int64),
    )
    legal_drift = CachedTrainingSource(
        covered_source.sample_id,
        covered_source.feature,
        covered_source.probability,
        missing_legal,
    )
    with pytest.raises(RuntimeError, match="cached legal catalog disagrees"):
        build_epoch_branch_pools(
            (legal_drift,),
            variant="uniform_legal",
            epoch=0,
            global_seed=0,
        )


def test_epoch_pool_recomputes_after_masking_invalid_padded_positives() -> None:
    scene = make_two_target_scene()
    output = ToyFrozenBaseAdapter()(scene.image_batch())
    probability = output.probability.clone()
    probability[0, 0, 0, 0] = 0.9
    padded_output = FrozenBaseOutput(
        probability=probability,
        feature=output.feature,
    )
    valid = torch.ones_like(probability, dtype=torch.bool)
    valid[:, :, 0, :] = False
    record = build_state_record(
        scene.sample_id,
        padded_output,
        scene.gt_mask,
        image_valid_mask=valid,
    )
    assert probability[0, 0, 0, 0] >= 0.5
    assert not record.occupancy[0, 0]

    source = CachedTrainingSource(
        scene.sample_id,
        padded_output.feature,
        padded_output.probability,
        record,
    )
    pools = build_epoch_branch_pools(
        (source,),
        variant="uniform_legal",
        epoch=0,
        global_seed=3,
    )
    assert len(pools.factual_no_miss) == 1
    assert len(pools.synthetic) == 1
    assert not pools.factual_no_miss[0].supervision.occupancy[0, 0, 0]


def test_fixed_training_returns_immutable_training_only_logs() -> None:
    sources = _sources()
    torch.manual_seed(5)
    decoder = CURELiteDecoder(feature_channels=3)
    criterion = CURELiteLoss()
    optimizer = torch.optim.SGD(decoder.parameters(), lr=1e-3)

    result = run_fixed_training(
        decoder,
        criterion,
        optimizer,
        sources,
        variant="uniform_legal",
        epochs=2,
        steps_per_epoch=1,
        branch_batch_sizes={
            "factual_miss": 1,
            "factual_no_miss": 1,
            "synthetic": 1,
        },
        global_seed=7,
    )

    assert result.variant == "uniform_legal"
    assert result.epochs == 2
    assert result.steps_per_epoch == 1
    assert tuple(log.epoch for log in result.epoch_logs) == (0, 1)
    for log in result.epoch_logs:
        assert dict(log.pool_sizes) == {
            "factual_miss": 1,
            "factual_no_miss": 1,
            "synthetic": 2,
        }
        metrics = dict(log.metrics)
        assert metrics["steps"] == 1
        assert metrics["synthetic/active"] == 1.0
        assert "validation" not in metrics
        assert "checkpoint" not in metrics

    with pytest.raises(FrozenInstanceError):
        result.epochs = 3
    with pytest.raises(FrozenInstanceError):
        result.epoch_logs[0].epoch = 4


def test_formal_branch_support_rejects_missing_identifying_pool() -> None:
    sources = _sources()
    pools = build_epoch_branch_pools(
        sources,
        variant="uniform_legal",
        epoch=0,
        global_seed=7,
    )
    require_training_branch_support(pools, variant="uniform_legal")
    require_training_branch_support(
        replace(pools, synthetic=()),
        variant="factual_only",
    )

    with pytest.raises(RuntimeError, match="synthetic"):
        require_training_branch_support(
            replace(pools, synthetic=()),
            variant="uniform_legal",
        )
    with pytest.raises(RuntimeError, match="factual_miss"):
        require_training_branch_support(
            replace(pools, factual_miss=()),
            variant="factual_only",
        )
    with pytest.raises(RuntimeError, match="factual_no_miss"):
        require_training_branch_support(
            replace(pools, factual_no_miss=()),
            variant="uniform_legal",
        )


def test_fixed_training_does_not_update_when_real_branch_support_is_absent() -> None:
    covered_source = _sources()[1]
    torch.manual_seed(13)
    decoder = CURELiteDecoder(feature_channels=3)
    before = {
        name: value.detach().clone() for name, value in decoder.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="factual_miss"):
        run_fixed_training(
            decoder,
            CURELiteLoss(),
            torch.optim.SGD(decoder.parameters(), lr=1e-3),
            (covered_source,),
            variant="uniform_legal",
            epochs=1,
            steps_per_epoch=1,
            branch_batch_sizes={
                "factual_miss": 1,
                "factual_no_miss": 1,
                "synthetic": 1,
            },
            global_seed=7,
        )

    for name, value in decoder.state_dict().items():
        assert torch.equal(value, before[name])


def test_training_support_summary_counts_independent_real_sources() -> None:
    summary = summarize_training_support(_sources())

    assert summary.source_images == 2
    assert summary.factual_miss_images == 1
    assert summary.factual_no_miss_images == 1
    assert summary.factual_unreachable_images == 0
    assert summary.reachable_miss_targets == 1
    assert summary.decoder_visible_legal_candidates > 0
    assert summary.synthetic_images == 2
    assert 0.0 < summary.visible_legal_fraction <= 1.0
    TrainingSupportRequirements().require(summary)

    with pytest.raises(RuntimeError, match="factual_miss_images"):
        TrainingSupportRequirements(
            minimum_factual_miss_images=2,
        ).require(summary)
