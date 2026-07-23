from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

import cure_lite.experiment.training_pipeline as training_pipeline
from cure_lite.cache.state_cache import StateCacheRecord
from cure_lite.config import InterventionConfig, MatchConfig, OccupancyConfig
from cure_lite.decoder import CURELiteDecoder
from cure_lite.experiment.artifacts import decoder_state_fingerprint
from cure_lite.instances import instances_from_binary_mask
from cure_lite.intervention import enumerate_legal_deletions
from cure_lite.losses import CURELiteLoss
from cure_lite.matching import match_components
from cure_lite.occupancy import build_occupancy
from cure_lite.sampling import choose_uniform_legal_deletion
from cure_lite.supervision import (
    build_epoch_factual_supervision_from_catalog,
    build_factual_supervision,
    build_synthetic_supervision,
)
from cure_lite.toy import (
    ToyFrozenBaseAdapter,
    make_factual_miss_scene,
    make_two_target_scene,
)
from cure_lite.train.engine import CURELiteTrainEngine
from cure_lite.train.pools import (
    BranchPools,
    StateExample,
    iter_factual_exposure_matched_batches,
    iter_fixed_branch_batches,
)
from cure_lite.types import BranchSupervision


_VARIANTS = (
    "factual_only",
    "factual_exposure_matched",
    "uniform_legal",
)
_ALL_VARIANTS = (*_VARIANTS, "miss_aligned_legal")
_BRANCHES = ("factual_miss", "factual_no_miss", "synthetic")


def _pairs(rows: Sequence[object]) -> torch.Tensor:
    return torch.tensor(
        [[row.gt_id, row.pred_id] for row in rows],
        dtype=torch.int64,
    ).reshape(-1, 2)


def _cached_source(scene) -> training_pipeline.CachedTrainingSource:
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
    return training_pipeline.CachedTrainingSource(
        scene.sample_id,
        output.feature,
        output.probability,
        state,
    )


@pytest.fixture(scope="module")
def toy_sources() -> tuple[training_pipeline.CachedTrainingSource, ...]:
    return (
        _cached_source(make_factual_miss_scene(missed_gt_id=2)),
        _cached_source(make_two_target_scene()),
        _cached_source(make_factual_miss_scene(missed_gt_id=1)),
    )


@pytest.fixture(scope="module")
def toy_catalogs(toy_sources):
    return {
        False: training_pipeline.prepare_training_catalog(toy_sources),
        True: training_pipeline.prepare_training_catalog(
            tuple(reversed(toy_sources))
        ),
    }


def _ids(values: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in values.tolist())


def _pair_ids(values: torch.Tensor) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(int(value) for value in row)
        for row in values.tolist()
    )


def _masked(
    supervision: BranchSupervision,
    image_valid_mask: torch.Tensor,
) -> BranchSupervision:
    """Test-local copy of the legacy image-valid-mask materializer."""

    valid_2d = torch.as_tensor(
        image_valid_mask,
        dtype=torch.bool,
        device="cpu",
    )
    if valid_2d.ndim == 3 and valid_2d.shape[0] == 1:
        valid_2d = valid_2d[0]
    valid = supervision.valid_mask & valid_2d.unsqueeze(0)
    return BranchSupervision(
        occupancy=supervision.occupancy,
        target=supervision.target * valid.to(supervision.target.dtype),
        valid_mask=valid,
        branch=supervision.branch,
        positive_gt_ids=supervision.positive_gt_ids,
        unreachable_gt_ids=supervision.unreachable_gt_ids,
        reachable_gt_ids=supervision.reachable_gt_ids,
    )


def _legacy_epoch_pool_oracle(
    sources: Sequence[training_pipeline.CachedTrainingSource],
    *,
    variant: str,
    epoch: int,
    global_seed: int,
    occupancy_config: OccupancyConfig = OccupancyConfig(),
    match_config: MatchConfig = MatchConfig(),
    intervention_config: InterventionConfig = InterventionConfig(),
) -> BranchPools:
    """Frozen copy of the pre-catalog per-epoch semantic reconstruction."""

    factual_miss: list[StateExample] = []
    factual_no_miss: list[StateExample] = []
    synthetic: list[StateExample] = []
    for source in sorted(sources, key=lambda item: item.sample_id):
        state = source.state
        raw_occupancy, _ = build_occupancy(
            source.probability,
            occupancy_config,
        )
        occupancy = (raw_occupancy & state.image_valid_mask).contiguous()
        pred = instances_from_binary_mask(
            occupancy,
            connectivity=occupancy_config.connectivity,
            min_area=occupancy_config.min_component_area,
        )
        assert torch.equal(occupancy, state.occupancy)
        assert torch.equal(pred.labels, state.pred_labels)

        gt = instances_from_binary_mask(
            state.gt_labels > 0,
            connectivity=occupancy_config.connectivity,
            min_area=occupancy_config.min_component_area,
        )
        assert torch.equal(gt.labels, state.gt_labels)
        match = match_components(pred, gt, match_config)
        assert tuple(
            (pair.gt_id, pair.pred_id) for pair in match.pairs
        ) == _pair_ids(state.base_match_pairs)
        assert tuple(sorted(match.unmatched_gt_ids)) == _ids(
            state.real_miss_ids
        )
        factual_oracle = build_factual_supervision(
            occupancy,
            gt,
            match,
            match_config,
        )
        assert factual_oracle.reachable_gt_ids == _ids(
            state.reachable_miss_ids
        )

        factual = build_epoch_factual_supervision_from_catalog(
            state.occupancy,
            gt,
            real_miss_ids=_ids(state.real_miss_ids),
            reachable_gt_ids=_ids(state.reachable_miss_ids),
            sample_id=source.sample_id,
            epoch=epoch,
            global_seed=global_seed,
        )
        factual = _masked(factual, state.image_valid_mask)
        if factual.branch == "factual_miss":
            factual_miss.append(
                StateExample(source.sample_id, source.feature, factual)
            )
        elif factual.branch == "factual_no_miss":
            factual_no_miss.append(
                StateExample(source.sample_id, source.feature, factual)
            )
        else:
            assert factual.branch == "factual_unreachable"

        if variant != "uniform_legal":
            continue
        legal = enumerate_legal_deletions(
            pred,
            gt,
            match,
            occupancy,
            match_config=match_config,
            intervention_config=intervention_config,
        )
        assert tuple(
            (item.gt_id, item.pred_id) for item in legal
        ) == _pair_ids(state.legal_pairs)
        visible = training_pipeline.decoder_visible_legal_deletions(
            state.occupancy,
            legal,
            feature_size=tuple(source.feature.shape[-2:]),
        )
        selected = choose_uniform_legal_deletion(
            visible,
            sample_id=source.sample_id,
            epoch=epoch,
            global_seed=global_seed,
        )
        if selected is not None:
            supervision = _masked(
                build_synthetic_supervision(selected, gt),
                state.image_valid_mask,
            )
            synthetic.append(
                StateExample(source.sample_id, source.feature, supervision)
            )

    return BranchPools(
        factual_miss=tuple(factual_miss),
        factual_no_miss=tuple(factual_no_miss),
        synthetic=tuple(synthetic),
    )


def _assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.equal(actual, expected)


def _assert_pools_equal(actual: BranchPools, expected: BranchPools) -> None:
    for branch in _BRANCHES:
        actual_items = actual.get(branch)
        expected_items = expected.get(branch)
        assert tuple(item.sample_id for item in actual_items) == tuple(
            item.sample_id for item in expected_items
        )
        assert len(actual_items) == len(expected_items)
        for actual_item, expected_item in zip(
            actual_items,
            expected_items,
            strict=True,
        ):
            assert actual_item.sample_id == expected_item.sample_id
            _assert_tensor_equal(actual_item.feature, expected_item.feature)
            actual_supervision = actual_item.supervision
            expected_supervision = expected_item.supervision
            for name in ("occupancy", "target", "valid_mask"):
                _assert_tensor_equal(
                    getattr(actual_supervision, name),
                    getattr(expected_supervision, name),
                )
            for name in (
                "branch",
                "positive_gt_ids",
                "unreachable_gt_ids",
                "reachable_gt_ids",
            ):
                assert getattr(actual_supervision, name) == getattr(
                    expected_supervision,
                    name,
                )


@pytest.mark.parametrize("reverse_inputs", (False, True))
@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("global_seed", (0, 19, 2**31 - 1))
@pytest.mark.parametrize("epoch", (0, 1, 2, 19, 799))
def test_prepared_catalog_is_bit_exact_to_legacy_epoch_semantics(
    toy_sources: tuple[training_pipeline.CachedTrainingSource, ...],
    toy_catalogs,
    *,
    reverse_inputs: bool,
    variant: str,
    global_seed: int,
    epoch: int,
) -> None:
    sources = tuple(reversed(toy_sources)) if reverse_inputs else toy_sources
    expected = _legacy_epoch_pool_oracle(
        sources,
        variant=variant,
        epoch=epoch,
        global_seed=global_seed,
    )
    actual = training_pipeline.build_epoch_branch_pools_from_catalog(
        toy_catalogs[reverse_inputs],
        variant=variant,
        epoch=epoch,
        global_seed=global_seed,
    )
    _assert_pools_equal(actual, expected)


def test_eight_hundred_epochs_and_all_variants_reuse_prepared_semantics(
    toy_sources: tuple[training_pipeline.CachedTrainingSource, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = training_pipeline.prepare_training_catalog(toy_sources)
    expensive_names = (
        "threshold_occupancy",
        "instances_from_binary_mask",
        "match_components",
        "_factual_reachability_catalog_validated",
        "build_factual_supervision_from_catalog",
        "build_synthetic_supervision_from_catalog",
        "_enumerate_legal_deletions_validated",
        "decoder_visible_legal_deletions",
        "project_occupancy_to_feature_grid",
        "_apply_image_valid_mask",
        "miss_alignment_descriptors",
        "choose_miss_aligned_legal_identity",
        "_build_miss_aligned_choices",
    )
    calls = dict.fromkeys(expensive_names, 0)

    for name in expensive_names:
        original = getattr(training_pipeline, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(training_pipeline, name, counted)

    for variant in _ALL_VARIANTS:
        for epoch in range(800):
            pools = training_pipeline.build_epoch_branch_pools_from_catalog(
                catalog,
                variant=variant,
                epoch=epoch,
                global_seed=42,
            )
            assert pools.factual_miss
            assert pools.factual_no_miss
            assert bool(pools.synthetic) == (
                variant in {"uniform_legal", "miss_aligned_legal"}
            )

    assert calls == dict.fromkeys(expensive_names, 0)


def test_alignment_descriptors_are_built_once_per_supported_source(
    toy_sources: tuple[training_pipeline.CachedTrainingSource, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = training_pipeline.miss_alignment_descriptors

    def counted(feature, masks):
        calls.append(str(id(feature)))
        return original(feature, masks)

    monkeypatch.setattr(
        training_pipeline,
        "miss_alignment_descriptors",
        counted,
    )
    catalog = training_pipeline.prepare_training_catalog(toy_sources)
    expected_calls = sum(
        bool(entry.factual_examples or entry.synthetic_examples)
        for entry in catalog.entries
    )
    assert len(calls) == expected_calls
    assert len(calls) == len(set(calls))


def test_epoch_pools_reuse_only_prevalidated_state_templates(toy_catalogs) -> None:
    catalog = toy_catalogs[False]
    factual_templates = {
        id(example)
        for entry in catalog.entries
        for example in (
            *entry.factual_examples,
            *(
                (entry.factual_no_miss_example,)
                if entry.factual_no_miss_example is not None
                else ()
            ),
        )
    }
    synthetic_templates = {
        id(example)
        for entry in catalog.entries
        for example in entry.synthetic_examples
    }

    for variant in _ALL_VARIANTS:
        for epoch in range(800):
            pools = training_pipeline.build_epoch_branch_pools_from_catalog(
                catalog,
                variant=variant,
                epoch=epoch,
                global_seed=42,
            )
            assert {
                id(example)
                for branch in ("factual_miss", "factual_no_miss")
                for example in pools.get(branch)
            } <= factual_templates
            assert {id(example) for example in pools.synthetic} <= (
                synthetic_templates
                if variant in {"uniform_legal", "miss_aligned_legal"}
                else set()
            )


def _train_with_epoch_builder(
    sources: tuple[training_pipeline.CachedTrainingSource, ...],
    *,
    variant: str,
    use_catalog: bool,
) -> tuple[CURELiteDecoder, tuple[tuple[dict[str, int], dict[str, float | int]], ...]]:
    """Run the same deterministic optimizer path through legacy/new pools."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(314159)
        decoder = CURELiteDecoder(feature_channels=3)
    optimizer = torch.optim.SGD(decoder.parameters(), lr=1e-3)
    engine = CURELiteTrainEngine(decoder, CURELiteLoss(), optimizer)
    catalog = (
        training_pipeline.prepare_training_catalog(sources)
        if use_catalog
        else None
    )
    logs: list[tuple[dict[str, int], dict[str, float | int]]] = []
    batch_sizes = {
        "factual_miss": 1,
        "factual_no_miss": 1,
        "synthetic": 1,
    }
    for epoch in range(2):
        pools = (
            training_pipeline.build_epoch_branch_pools_from_catalog(
                catalog,
                variant=variant,
                epoch=epoch,
                global_seed=23,
            )
            if catalog is not None
            else (
                training_pipeline.build_epoch_branch_pools(
                    sources,
                    variant=variant,
                    epoch=epoch,
                    global_seed=23,
                )
                if variant == "miss_aligned_legal"
                else _legacy_epoch_pool_oracle(
                    sources,
                    variant=variant,
                    epoch=epoch,
                    global_seed=23,
                )
            )
        )
        if variant == "factual_exposure_matched":
            batches = iter_factual_exposure_matched_batches(
                pools,
                batch_sizes,
                replacement_count=1,
                epoch=epoch,
                global_seed=23,
                device="cpu",
                steps=2,
            )
        else:
            batches = iter_fixed_branch_batches(
                pools,
                batch_sizes,
                epoch=epoch,
                global_seed=23,
                device="cpu",
                steps=2,
            )
        logs.append(
            (
                {branch: len(pools.get(branch)) for branch in _BRANCHES},
                engine.run_epoch(batches),
            )
        )
    return decoder, tuple(logs)


@pytest.mark.parametrize("variant", _ALL_VARIANTS)
def test_prepared_catalog_preserves_training_log_and_final_decoder_bytes(
    toy_sources: tuple[training_pipeline.CachedTrainingSource, ...],
    variant: str,
) -> None:
    legacy_decoder, legacy_logs = _train_with_epoch_builder(
        toy_sources,
        variant=variant,
        use_catalog=False,
    )
    prepared_decoder, prepared_logs = _train_with_epoch_builder(
        toy_sources,
        variant=variant,
        use_catalog=True,
    )

    assert prepared_logs == legacy_logs
    assert decoder_state_fingerprint(prepared_decoder) == (
        decoder_state_fingerprint(legacy_decoder)
    )
    for name, legacy_tensor in legacy_decoder.state_dict().items():
        assert torch.equal(prepared_decoder.state_dict()[name], legacy_tensor)
