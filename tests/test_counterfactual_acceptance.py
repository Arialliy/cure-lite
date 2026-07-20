from __future__ import annotations

import torch

from cure_lite.counterfactual import (
    AcceptanceConfig,
    TransformConfig,
    TransformSpec,
    apply_counterfactual_transform,
    assess_legal_intervention,
)
from cure_lite.instances import instances_from_binary_mask
from cure_lite.matching import match_components
from cure_lite.occupancy import build_occupancy
from cure_lite.toy import ToyFrozenBaseAdapter, make_two_target_scene


def _state(adapter, image, gt):
    output = adapter(image)
    _, pred = build_occupancy(output.probability)
    return output, pred, match_components(pred, gt)


def _valid_candidate():
    scene = make_two_target_scene()
    gt = instances_from_binary_mask(scene.gt_mask)
    adapter = ToyFrozenBaseAdapter()
    before_output, pred_before, match_before = _state(
        adapter, scene.image_batch(), gt
    )
    config = TransformConfig()
    transformed, diagnostics = apply_counterfactual_transform(
        scene.image_batch(),
        gt.by_id(1).mask,
        TransformSpec(config=config, strength=0.75),
    )
    after_output, pred_after, match_after = _state(adapter, transformed, gt)
    changed = torch.any(transformed != scene.image_batch(), dim=(0, 1))
    return (
        scene,
        gt,
        adapter,
        before_output,
        pred_before,
        match_before,
        transformed,
        diagnostics,
        after_output,
        pred_after,
        match_after,
        changed,
    )


def _assess(values, **overrides):
    (
        _,
        gt,
        _,
        before_output,
        pred_before,
        match_before,
        _,
        diagnostics,
        after_output,
        pred_after,
        match_after,
        changed,
    ) = values
    arguments = dict(
        gt=gt,
        target_gt_id=1,
        pred_before=pred_before,
        match_before=match_before,
        probability_before=before_output.probability,
        pred_after=pred_after,
        match_after=match_after,
        probability_after=after_output.probability,
        changed_support=changed,
        transform_max_abs_delta=diagnostics.max_abs_delta,
        transform_mean_abs_delta=diagnostics.mean_abs_delta,
        transform_outside_max_delta=diagnostics.outside_roi_max_delta,
    )
    arguments.update(overrides)
    return assess_legal_intervention(**arguments)


def test_atomic_model_consistent_candidate_passes_every_gate() -> None:
    decision = _assess(_valid_candidate())

    assert decision.accepted
    assert decision.reasons == ()
    assert decision.retained_lineage_ious == ((2, 1.0),)
    assert decision.writable_pixels == 9
    assert decision.full_gt_recoverable
    assert decision.raw_background_pixels_after == 0


def test_two_new_misses_are_rejected_as_non_atomic() -> None:
    values = _valid_candidate()
    scene, gt, adapter = values[:3]
    config = TransformConfig()
    first, _ = apply_counterfactual_transform(
        scene.image_batch(), gt.by_id(1).mask, TransformSpec(config, 0.75)
    )
    second, _ = apply_counterfactual_transform(
        first, gt.by_id(2).mask, TransformSpec(config, 0.75)
    )
    output, pred, match = _state(adapter, second, gt)
    changed = torch.any(second != scene.image_batch(), dim=(0, 1))

    decision = _assess(
        values,
        probability_after=output.probability,
        pred_after=pred,
        match_after=match,
        changed_support=changed,
        transform_mean_abs_delta=float(
            torch.mean(torch.abs(second - scene.image_batch()))
        ),
    )

    assert not decision.accepted
    assert "non_atomic_coverage_change" in decision.reasons
    assert "full_gt_not_recoverable" in decision.reasons


def test_retained_component_identity_change_is_rejected() -> None:
    values = _valid_candidate()
    scene, gt = values[:2]
    after_probability = values[8].probability.clone()
    retained = gt.by_id(2).mask
    shifted = torch.zeros_like(retained)
    coordinates = torch.nonzero(retained, as_tuple=False)
    shifted[coordinates[:, 0], coordinates[:, 1] - 2] = True
    after_probability[0, 0][retained] = 0.1
    after_probability[0, 0][shifted] = 0.9
    _, pred_after = build_occupancy(after_probability)
    match_after = match_components(pred_after, gt)
    changed = values[11] | retained | shifted

    decision = _assess(
        values,
        probability_after=after_probability,
        pred_after=pred_after,
        match_after=match_after,
        changed_support=changed,
        config=AcceptanceConfig(min_retained_component_iou=0.5),
    )

    assert match_after.matched_gt_ids == frozenset({2})
    assert not decision.accepted
    assert "retained_component_lineage_changed" in decision.reasons


def test_probability_change_outside_local_guard_is_rejected() -> None:
    values = _valid_candidate()
    after_probability = values[8].probability.clone()
    after_probability[0, 0, 0, 0] = 0.2
    _, pred_after = build_occupancy(after_probability)
    match_after = match_components(pred_after, values[1])

    decision = _assess(
        values,
        probability_after=after_probability,
        pred_after=pred_after,
        match_after=match_after,
    )

    assert not decision.accepted
    assert "base_output_changed_outside_guard" in decision.reasons


def test_probability_and_component_state_must_be_coherent() -> None:
    values = _valid_candidate()

    try:
        _assess(values, probability_after=values[3].probability)
    except ValueError as error:
        assert "pred_after was not built" in str(error)
    else:
        raise AssertionError("incoherent probability/prediction state was accepted")


def test_unmatched_component_pixel_fa_cannot_grow_at_fixed_component_count() -> None:
    gt_mask = torch.zeros(32, 32, dtype=torch.bool)
    gt_mask[5:15, 5:15] = True
    gt = instances_from_binary_mask(gt_mask)
    probability_before = torch.full((1, 1, 32, 32), 0.1)
    probability_before[0, 0][gt_mask] = 0.9
    probability_before[0, 0, 25, 25] = 0.9
    _, pred_before = build_occupancy(probability_before)
    match_before = match_components(pred_before, gt)
    assert match_before.matched_gt_ids == frozenset({1})
    assert len(match_before.unmatched_pred_ids) == 1

    probability_after = torch.full_like(probability_before, 0.1)
    probability_after[0, 0, 5:9, 5:9] = 0.9
    _, pred_after = build_occupancy(probability_after)
    match_after = match_components(pred_after, gt)
    assert match_after.matched_gt_ids == frozenset()
    assert len(match_after.unmatched_pred_ids) == 1
    changed = probability_before[0, 0] != probability_after[0, 0]

    decision = assess_legal_intervention(
        gt=gt,
        target_gt_id=1,
        pred_before=pred_before,
        match_before=match_before,
        probability_before=probability_before,
        pred_after=pred_after,
        match_after=match_after,
        probability_after=probability_after,
        changed_support=changed,
        transform_max_abs_delta=0.7,
        transform_mean_abs_delta=0.04,
        transform_outside_max_delta=0.0,
        config=AcceptanceConfig(
            max_input_abs_delta=1.0,
            max_input_mean_delta=0.1,
        ),
    )

    assert decision.unmatched_component_pixels_before == 1
    assert decision.unmatched_component_pixels_after == 16
    assert decision.raw_background_pixels_after < decision.raw_background_pixels_before
    assert "unmatched_component_pixels_increased" in decision.reasons
