from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cure_lite.cache.state_cache import StateCacheRecord
from cure_lite.experiment.geometry_catalog_protocol import (
    load_geometry_catalog_protocol,
)
from cure_lite.experiment.geometry_safe_catalog import (
    GeometrySafeEntryView,
    _evaluation_geometry,
    trace_native_to_evaluation,
)
from cure_lite.experiment.training_pipeline import (
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.instances import instances_from_binary_mask
from cure_lite.intervention import enumerate_legal_deletions
from cure_lite.matching import match_components
from cure_lite.occupancy import build_occupancy
from cure_lite.supervision import build_factual_supervision
from cure_lite.toy import (
    ToyFrozenBaseAdapter,
    make_factual_miss_scene,
    make_two_target_scene,
)
from cure_lite.experiment.p0_geometry import _resize_mask


_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "IRSTD-1K"
    / "geometry_safe_p0_v2"
    / "config.json"
)


def _geometry(
    native: tuple[int, int],
    evaluation: tuple[int, int],
):
    return replace(
        load_geometry_catalog_protocol(_CONFIG).geometry,
        expected_native_size=native,
        expected_evaluation_size=evaluation,
    )


def _evaluation_labels(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return instances_from_binary_mask(_resize_mask(mask, size)).labels


def test_pillow_nearest_4_to_2_sampling_lookup_is_frozen() -> None:
    retained = set()
    for y in range(4):
        for x in range(4):
            mask = torch.zeros((4, 4), dtype=torch.bool)
            mask[y, x] = True
            if torch.any(_resize_mask(mask, (2, 2))):
                retained.add((y, x))
    assert retained == {(1, 1), (1, 3), (3, 1), (3, 3)}


def test_trace_records_native_disappearance() -> None:
    native = torch.zeros((4, 4), dtype=torch.bool)
    native[0, 0] = True
    trace = trace_native_to_evaluation(
        native,
        torch.zeros((2, 2), dtype=torch.int64),
        _geometry((4, 4), (2, 2)),
    )
    assert trace.descendants == ((1, ()),)
    assert trace.ancestors == ()


def test_trace_detects_merge_with_reciprocal_incidence() -> None:
    native = torch.zeros((4, 4), dtype=torch.bool)
    native[1, 1] = True
    native[1, 3] = True
    labels = _evaluation_labels(native, (2, 2))
    trace = trace_native_to_evaluation(
        native,
        labels,
        _geometry((4, 4), (2, 2)),
    )
    assert trace.descendants == ((1, (1,)), (2, (1,)))
    assert trace.ancestors == ((1, (1, 2)),)
    geometry = _evaluation_geometry(
        trace,
        1,
        _geometry((4, 4), (2, 2)),
    )
    assert geometry.reason_codes == ("multiple_native_ancestors",)


def test_trace_detects_split_after_thin_bridge_is_lost() -> None:
    native = torch.zeros((8, 8), dtype=torch.bool)
    for y, x in ((1, 1), (0, 2), (0, 3), (0, 4), (1, 5)):
        native[y, x] = True
    labels = _evaluation_labels(native, (4, 4))
    trace = trace_native_to_evaluation(
        native,
        labels,
        _geometry((8, 8), (4, 4)),
    )
    assert trace.descendants == ((1, (1, 2)),)
    assert trace.ancestors == ((1, (1,)), (2, (1,)))
    for evaluation_id in (1, 2):
        geometry = _evaluation_geometry(
            trace,
            evaluation_id,
            _geometry((8, 8), (4, 4)),
        )
        assert geometry.reason_codes == (
            "native_has_multiple_evaluation_descendants",
        )


def test_unrelated_disappearance_does_not_reject_safe_evaluation_target() -> None:
    native = torch.zeros((4, 4), dtype=torch.bool)
    native[0, 0] = True
    native[2:4, 2:4] = True
    labels = _evaluation_labels(native, (2, 2))
    config = _geometry((4, 4), (2, 2))
    trace = trace_native_to_evaluation(native, labels, config)
    assert trace.descendants == ((1, ()), (2, (1,)))
    geometry = _evaluation_geometry(trace, 1, config)
    assert geometry.reciprocal
    assert geometry.exact_projection
    assert geometry.eligible


@pytest.mark.parametrize(
    ("pixels", "expected_ratio"),
    [
        (
            (
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
            ),
            0.5,
        ),
        (((0, 0), (1, 1)), 2.0),
    ],
)
def test_area_ratio_boundaries_are_inclusive(
    pixels: tuple[tuple[int, int], ...],
    expected_ratio: float,
) -> None:
    native = torch.zeros((4, 4), dtype=torch.bool)
    for y, x in pixels:
        native[y, x] = True
    config = _geometry((4, 4), (2, 2))
    labels = _evaluation_labels(native, (2, 2))
    geometry = _evaluation_geometry(
        trace_native_to_evaluation(native, labels, config),
        1,
        config,
    )
    assert geometry.area_ratio == expected_ratio
    assert "area_ratio_below_minimum" not in geometry.reason_codes
    assert "area_ratio_above_maximum" not in geometry.reason_codes


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
        image_valid_mask=torch.ones_like(occupancy),
    )
    return CachedTrainingSource(
        scene.sample_id,
        output.feature,
        output.probability,
        state,
    )


def test_geometry_entry_view_filters_candidate_and_example_by_same_index() -> None:
    catalog = prepare_training_catalog(
        (
            _cached_source(make_factual_miss_scene(missed_gt_id=1)),
            _cached_source(make_two_target_scene()),
        )
    )
    base = next(
        entry
        for entry in catalog.entries
        if len(entry.decoder_visible_legal_candidates) == 2
    )
    view = GeometrySafeEntryView(base=base, selected_legal_indices=(1,))
    assert (
        view.decoder_visible_legal_candidates[0]
        is base.decoder_visible_legal_candidates[1]
    )
    assert view.synthetic_examples[0] is base.synthetic_examples[1]
    assert view.synthetic_examples[0].feature is base.source.feature
    assert view.factual_examples is base.factual_examples
