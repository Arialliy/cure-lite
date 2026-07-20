from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.counterfactual import (
    TransformConfig,
    search_minimal_legal_intervention,
)
from cure_lite.instances import instances_from_binary_mask
from cure_lite.supervision import build_atomic_intervention_supervision
from cure_lite.toy import (
    ToyFrozenBaseAdapter,
    make_factual_miss_scene,
    make_two_target_scene,
)
from cure_lite.types import MatchPair, MatchResult


class _CountingToyAdapter(ToyFrozenBaseAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.extract_calls = 0

    def extract(self, images):
        self.extract_calls += 1
        return super().extract(images)


def test_search_selects_the_weakest_legal_model_consistent_state() -> None:
    scene = make_two_target_scene()
    gt = instances_from_binary_mask(scene.gt_mask)
    base = _CountingToyAdapter()

    result = search_minimal_legal_intervention(
        sample_id=scene.sample_id,
        image=scene.image_batch(),
        gt=gt,
        target_gt_id=1,
        base=base,
        transform_config=TransformConfig(
            strength_grid=(0.25, 0.5, 0.75, 0.9)
        ),
    )

    assert result.failure_reason is None
    assert result.state is not None
    assert [attempt.spec.strength for attempt in result.attempts] == [
        0.25,
        0.5,
        0.75,
    ]
    assert [attempt.accepted for attempt in result.attempts] == [False, False, True]
    # The stronger 0.9 candidate is never evaluated after 0.75 is accepted.
    assert base.extract_calls == 4  # original state plus three candidates

    state = result.state
    assert state.receipt.before_covered_gt_ids == (1, 2)
    assert state.receipt.after_covered_gt_ids == (2,)
    assert state.receipt.transform.strength == 0.75
    assert len(state.receipt.source_image_fingerprint) == 64
    assert len(state.receipt.target_mask_fingerprint) == 64
    assert len(state.receipt.transformed_image_fingerprint) == 64
    assert len(state.receipt.probability_fingerprint) == 64
    assert len(state.receipt.feature_fingerprint) == 64
    assert len(state.receipt.match_config_fingerprint) == 64
    assert state.receipt.source_image_fingerprint != (
        state.receipt.transformed_image_fingerprint
    )
    assert state.matching.unmatched_gt_ids == frozenset({1})
    assert state.supervision.positive_gt_ids == (1,)
    assert torch.equal(
        state.supervision.target[0].to(torch.bool), gt.by_id(1).mask
    )
    assert torch.equal(
        state.occupancy,
        state.probability[0, 0].cpu() >= state.receipt.occupancy_threshold,
    )
    batch = state.branch_batch()
    assert batch.feature.shape == (1, 3, 32, 32)
    assert batch.occupancy.shape == (1, 1, 32, 32)
    batch.validate(expected_branch="synthetic")


def test_search_is_deterministic_and_receipt_binds_state() -> None:
    scene = make_two_target_scene()
    gt = instances_from_binary_mask(scene.gt_mask)

    first = search_minimal_legal_intervention(
        sample_id=scene.sample_id,
        image=scene.image_batch(),
        gt=gt,
        target_gt_id=2,
        base=ToyFrozenBaseAdapter(),
    )
    second = search_minimal_legal_intervention(
        sample_id=scene.sample_id,
        image=scene.image_batch(),
        gt=gt,
        target_gt_id=2,
        base=ToyFrozenBaseAdapter(),
    )

    assert first.state is not None and second.state is not None
    assert first.state.receipt == second.state.receipt
    assert torch.equal(first.state.transformed_image, second.state.transformed_image)
    assert torch.equal(first.state.probability, second.state.probability)
    assert torch.equal(first.state.feature, second.state.feature)


def test_model_consistent_state_rejects_cross_state_tensor_substitution() -> None:
    scene = make_two_target_scene()
    gt = instances_from_binary_mask(scene.gt_mask)
    result = search_minimal_legal_intervention(
        sample_id=scene.sample_id,
        image=scene.image_batch(),
        gt=gt,
        target_gt_id=1,
        base=ToyFrozenBaseAdapter(),
    )
    assert result.state is not None
    state = result.state

    with pytest.raises(ValueError, match="feature does not match"):
        replace(state, feature=torch.zeros_like(state.feature))

    changed_probability = state.probability * 0.9
    assert torch.equal(
        changed_probability[0, 0] >= state.receipt.occupancy_threshold,
        state.occupancy,
    )
    with pytest.raises(ValueError, match="probability does not match"):
        replace(state, probability=changed_probability)

    with pytest.raises(ValueError, match="matching is stale"):
        replace(state, matching=state.matching_before)

    with pytest.raises(ValueError, match="transform config"):
        replace(
            state.receipt,
            transform_config_fingerprint="forged-transform-config",
        )

    with pytest.raises(ValueError, match="sample IDs disagree"):
        replace(result, sample_id="another-sample")
    with pytest.raises(ValueError, match="target IDs disagree"):
        replace(result, target_gt_id=2)
    with pytest.raises(ValueError, match="pre-intervention matching disagree"):
        replace(result, match_before=state.matching)


def test_state_owns_input_storage_and_revalidates_before_training() -> None:
    scene = make_two_target_scene()
    image = scene.image_batch().clone()
    gt = instances_from_binary_mask(scene.gt_mask)
    result = search_minimal_legal_intervention(
        sample_id=scene.sample_id,
        image=image,
        gt=gt,
        target_gt_id=1,
        base=ToyFrozenBaseAdapter(),
    )
    assert result.state is not None
    state = result.state
    stored_source = state.source_image.clone()
    stored_gt_labels = state.gt.labels.clone()

    image.fill_(0.123)
    gt.labels.zero_()
    assert torch.equal(state.source_image, stored_source)
    assert torch.equal(state.gt.labels, stored_gt_labels)

    state.feature.zero_()
    with pytest.raises(ValueError, match="feature does not match"):
        state.branch_batch()


def test_atomic_intervention_supervision_rejects_forged_before_matching() -> None:
    scene = make_two_target_scene()
    gt = instances_from_binary_mask(scene.gt_mask)
    result = search_minimal_legal_intervention(
        sample_id=scene.sample_id,
        image=scene.image_batch(),
        gt=gt,
        target_gt_id=1,
        base=ToyFrozenBaseAdapter(),
    )
    assert result.state is not None
    state = result.state
    forged = MatchResult(
        pairs=(
            MatchPair(gt_id=1, pred_id=97, distance=0.0, iou=1.0),
            MatchPair(gt_id=2, pred_id=98, distance=0.0, iou=1.0),
        ),
        pred_ids=(97, 98),
        gt_ids=(1, 2),
    )

    with pytest.raises(ValueError, match="before matching is stale"):
        build_atomic_intervention_supervision(
            state.occupancy,
            gt,
            1,
            state.prediction_before,
            forged,
            state.matching,
            state.match_config,
        )


def test_search_returns_explicit_none_when_strength_grid_has_no_legal_state() -> None:
    scene = make_two_target_scene()
    gt = instances_from_binary_mask(scene.gt_mask)
    base = _CountingToyAdapter()
    config = TransformConfig(strength_grid=(0.25, 0.5))

    result = search_minimal_legal_intervention(
        sample_id=scene.sample_id,
        image=scene.image_batch(),
        gt=gt,
        target_gt_id=1,
        base=base,
        transform_config=config,
    )

    assert result.state is None
    assert result.failure_reason == "no_legal_candidate"
    assert len(result.attempts) == 2
    assert not any(attempt.accepted for attempt in result.attempts)
    assert base.extract_calls == 3


def test_search_does_not_fabricate_counterfactual_for_an_existing_miss() -> None:
    scene = make_factual_miss_scene(missed_gt_id=1)
    gt = instances_from_binary_mask(scene.gt_mask)
    base = _CountingToyAdapter()

    result = search_minimal_legal_intervention(
        sample_id=scene.sample_id,
        image=scene.image_batch(),
        gt=gt,
        target_gt_id=1,
        base=base,
    )

    assert result.state is None
    assert result.failure_reason == "target_not_matched_before"
    assert result.attempts == ()
    assert base.extract_calls == 1
