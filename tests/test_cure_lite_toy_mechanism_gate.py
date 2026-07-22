"""Executable mechanism gate for the active Uniform-Legal CURE-Lite path.

This test deliberately does not import or emulate an input-level
counterfactual method.  Synthetic states keep the frozen feature of a covered
source target and delete only its legal occupancy component.  Thresholds are
selected on a validation fixture and then frozen for spatially distinct held-
out fixtures.

The toy result is a software/mechanism witness, not evidence of real IRSTD
performance.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from cure_lite.calibration import (
    CalibrationSample,
    FalseAlarmBudget,
    FrozenBaseThresholdProtocol,
    FrozenThresholdProtocol,
    select_base_threshold_at_budget,
    select_residual_threshold,
)
from cure_lite.config import MatchConfig, OccupancyConfig
from cure_lite.decoder import CURELiteDecoder
from cure_lite.instances import instances_from_binary_mask
from cure_lite.intervention import enumerate_legal_deletions
from cure_lite.losses import CURELiteLoss
from cure_lite.matching import match_components
from cure_lite.model import CURELiteModel
from cure_lite.occupancy import build_occupancy
from cure_lite.sampling import choose_uniform_legal_deletion
from cure_lite.supervision import (
    build_factual_supervision,
    build_synthetic_supervision,
)
from cure_lite.toy import (
    ToyFrozenBaseAdapter,
    ToyScene,
    attenuate_target,
    make_custom_two_target_scene,
)
from cure_lite.train.step import BranchBatch, multi_branch_train_step


_POSITIONS = (
    ((5, 5), (23, 23)),
    ((6, 20), (22, 4)),
    ((9, 5), (20, 21)),
    ((4, 24), (24, 7)),
    ((12, 3), (18, 24)),
)
_DISTRACTORS = ((1, 1), (15, 15), (30, 30))
_RESIDUAL_THRESHOLDS = (
    0.0,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    1e-3,
    1e-2,
    0.1,
    0.5,
    0.7,
    0.8,
    0.9,
    0.95,
    0.98,
    0.99,
)


def _covered_scene(
    sample_id: str,
    positions: tuple[tuple[int, int], tuple[int, int]],
    contrast: float,
) -> ToyScene:
    scene = make_custom_two_target_scene(
        sample_id=sample_id,
        target_top_lefts=positions,
        distractor_points=_DISTRACTORS,
    )
    image = scene.image.clone()
    for target in scene.target_masks:
        image[0][target] = contrast
    return ToyScene(
        sample_id=sample_id,
        image=image,
        gt_mask=scene.gt_mask,
        target_masks=scene.target_masks,
    )


def _stack_batches(items: Sequence[BranchBatch]) -> BranchBatch:
    return BranchBatch(
        feature=torch.cat([item.feature for item in items], dim=0),
        occupancy=torch.stack([item.occupancy for item in items], dim=0),
        target=torch.stack([item.target for item in items], dim=0),
        valid_mask=torch.stack([item.valid_mask for item in items], dim=0),
    )


def _train_uniform_legal(
    contrasts: Sequence[float],
) -> tuple[ToyFrozenBaseAdapter, CURELiteModel, float, frozenset[int]]:
    if len(contrasts) != len(_POSITIONS):
        raise ValueError("one source contrast is required for every toy layout")

    torch.manual_seed(7)
    base = ToyFrozenBaseAdapter()
    frozen_before = {
        name: value.detach().clone() for name, value in base.state_dict().items()
    }
    no_miss_batches: list[BranchBatch] = []
    synthetic_batches: list[BranchBatch] = []
    selected_gt_ids: set[int] = set()

    for index, (positions, contrast) in enumerate(zip(_POSITIONS, contrasts)):
        scene = _covered_scene(f"toy-R-{index}", positions, float(contrast))
        output = base(scene.image_batch())
        gt = instances_from_binary_mask(scene.gt_mask)
        occupancy, prediction = build_occupancy(output.probability)
        matching = match_components(prediction, gt)
        legal = enumerate_legal_deletions(
            prediction,
            gt,
            matching,
            occupancy,
        )
        assert len(legal) == 2
        deletion = choose_uniform_legal_deletion(
            legal,
            sample_id=scene.sample_id,
            epoch=0,
            global_seed=17,
        )
        assert deletion is not None
        selected_gt_ids.add(deletion.gt_id)

        no_miss = build_factual_supervision(occupancy, gt, matching)
        assert no_miss.branch == "factual_no_miss"
        synthetic = build_synthetic_supervision(deletion, gt)
        no_miss_batches.append(BranchBatch.from_supervision(output.feature, no_miss))
        synthetic_batches.append(
            BranchBatch.from_supervision(output.feature, synthetic)
        )

    decoder = CURELiteDecoder(feature_channels=base.feature_channels)
    criterion = CURELiteLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
    batches = {
        "factual_no_miss": _stack_batches(no_miss_batches),
        "synthetic": _stack_batches(synthetic_batches),
    }
    first = multi_branch_train_step(decoder, criterion, optimizer, batches)
    final = first
    for _ in range(49):
        final = multi_branch_train_step(decoder, criterion, optimizer, batches)

    assert first["factual_miss/active"] == 0
    assert float(final["total"]) < 0.01 * float(first["total"])
    assert all(
        torch.equal(value, frozen_before[name])
        for name, value in base.state_dict().items()
    )
    assert all(not parameter.requires_grad for parameter in base.parameters())
    model = CURELiteModel(base, decoder)
    model.eval()
    return base, model, float(final["total"]), frozenset(selected_gt_ids)


def _factual_scene(
    sample_id: str,
    positions: tuple[tuple[int, int], tuple[int, int]],
    distractors: tuple[tuple[int, int], ...],
    missed_gt_id: int,
) -> ToyScene:
    source = make_custom_two_target_scene(
        sample_id=f"{sample_id}-source",
        target_top_lefts=positions,
        distractor_points=distractors,
    )
    return attenuate_target(
        source,
        missed_gt_id,
        sample_id=sample_id,
    )


def _calibration_sample(model: CURELiteModel, scene: ToyScene) -> CalibrationSample:
    with torch.no_grad():
        output = model(scene.image_batch(), residual_threshold=None)
    return CalibrationSample(
        sample_id=scene.sample_id,
        base_probability=output.base_probability[0, 0],
        residual_probability=output.residual_probability[0, 0],
        gt_mask=scene.gt_mask[0],
    )


def _validation_and_holdout() -> tuple[ToyScene, tuple[ToyScene, ...]]:
    validation = _factual_scene(
        "toy-V-miss",
        ((7, 20), (21, 4)),
        ((3, 3), (15, 15), (27, 27)),
        1,
    )
    holdout = (
        _factual_scene(
            "toy-H-miss-1",
            ((8, 19), (20, 5)),
            ((2, 25), (14, 3), (28, 15)),
            1,
        ),
        _factual_scene(
            "toy-H-miss-2",
            ((4, 7), (23, 19)),
            ((1, 29), (16, 16), (29, 2)),
            2,
        ),
        _factual_scene(
            "toy-H-miss-3",
            ((13, 4), (6, 23)),
            ((2, 2), (20, 15), (29, 29)),
            1,
        ),
    )
    masks = (validation.gt_mask, *(scene.gt_mask for scene in holdout))
    assert len({mask.numpy().tobytes() for mask in masks}) == len(masks)
    return validation, holdout


def _zero_increment_budget() -> FalseAlarmBudget:
    return FalseAlarmBudget(
        pixel_fa_budget=0.0,
        component_fa_per_mp_budget=0.0,
        raw_background_fa_budget=0.0,
        minimum_retention=1.0,
    )


def test_uniform_legal_closes_held_out_toy_gate_at_the_base_fa_budget() -> None:
    """With source support near the anchor, Lite transfers to factual misses."""

    _, model, _, selected_gt_ids = _train_uniform_legal(
        (0.51, 0.55, 0.60, 0.70, 1.00)
    )
    assert selected_gt_ids == frozenset({1, 2})
    validation, holdout = _validation_and_holdout()
    validation_samples = [_calibration_sample(model, validation)]
    holdout_samples = [_calibration_sample(model, scene) for scene in holdout]
    occupancy = OccupancyConfig(threshold=0.5)
    matching = MatchConfig()
    budget = _zero_increment_budget()

    residual_selection = select_residual_threshold(
        validation_samples,
        _RESIDUAL_THRESHOLDS,
        occupancy,
        matching,
        budget,
    )
    base_selection = select_base_threshold_at_budget(
        validation_samples,
        (0.1, 0.2, 0.3, 0.38, 0.4, 0.45, 0.5),
        occupancy,
        matching,
        budget,
    )
    assert residual_selection.feasible
    assert residual_selection.threshold is not None
    assert base_selection.feasible
    assert base_selection.threshold == 0.5

    cure_metrics = FrozenThresholdProtocol(
        occupancy,
        residual_selection.threshold,
        budget,
    ).evaluate_test(holdout_samples, matching)
    base_metrics = FrozenBaseThresholdProtocol(
        base_selection.threshold,
        occupancy,
        budget,
    ).evaluate_test(holdout_samples, matching)

    assert cure_metrics.total_anchor_misses == 3
    assert cure_metrics.net_rmr == 1.0
    assert cure_metrics.gross_rmr == 1.0
    assert cure_metrics.pd == 1.0
    assert cure_metrics.retention == 1.0
    assert cure_metrics.pixel_fa == 0.0
    assert cure_metrics.raw_background_fa == 0.0
    assert cure_metrics.fp_components_per_mp == 0.0
    assert not cure_metrics.budget_violation
    assert base_metrics.net_rmr == 0.0
    assert base_metrics.pd == 0.5
    assert not base_metrics.budget_violation


def test_uniform_legal_gate_stops_when_source_feature_support_is_too_easy() -> None:
    """Low training loss cannot hide a synthetic/factual support failure."""

    _, model, final_loss, _ = _train_uniform_legal((1.0,) * len(_POSITIONS))
    assert final_loss < 0.01
    validation, _ = _validation_and_holdout()
    selection = select_residual_threshold(
        [_calibration_sample(model, validation)],
        _RESIDUAL_THRESHOLDS,
        OccupancyConfig(threshold=0.5),
        MatchConfig(),
        _zero_increment_budget(),
    )

    assert selection.feasible
    assert selection.threshold is None
    assert selection.metrics is not None
    assert selection.metrics.net_rmr == 0.0
    assert selection.metrics.pd == 0.5

