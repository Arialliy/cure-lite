"""Held-out toy closure for the model-consistent CURE training mechanism.

This is an executable contract test, not evidence of real-data performance.
The counterfactual branch is deliberately isolated from factual-miss training,
and validation/test layouts are spatially disjoint from the training layout.
"""

from __future__ import annotations

import torch

from cure_lite.counterfactual import search_minimal_legal_intervention
from cure_lite.decoder import CURELiteDecoder
from cure_lite.instances import instances_from_binary_mask
from cure_lite.losses import CURELiteLoss
from cure_lite.matching import match_components
from cure_lite.model import CURELiteModel
from cure_lite.occupancy import build_occupancy
from cure_lite.supervision import build_factual_supervision
from cure_lite.toy import (
    ToyFrozenBaseAdapter,
    attenuate_target,
    make_custom_two_target_scene,
)
from cure_lite.train.step import BranchBatch, multi_branch_train_step


def _scene(sample_id, target_top_lefts, distractor_points):
    return make_custom_two_target_scene(
        sample_id=sample_id,
        target_top_lefts=target_top_lefts,
        distractor_points=distractor_points,
    )


def _select_validation_threshold(model, scene, gt) -> float:
    """Choose the most conservative pre-declared threshold meeting toy gates."""

    background = ~scene.gt_mask[0]
    candidates = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
    valid: list[float] = []
    with torch.no_grad():
        raw = model(scene.image_batch(), residual_threshold=None)
    for threshold in candidates:
        residual = (raw.residual_probability[0, 0] >= threshold) & ~raw.occupancy[
            0, 0
        ]
        final = raw.occupancy[0, 0] | residual
        matching = match_components(instances_from_binary_mask(final.cpu()), gt)
        if (
            matching.matched_gt_ids == frozenset(gt.ids)
            and not torch.any(residual[background])
        ):
            valid.append(threshold)
    if not valid:
        raise AssertionError("validation found no threshold satisfying toy gates")
    return max(valid)


def test_model_consistent_counterfactual_branch_closes_on_held_out_toy() -> None:
    torch.manual_seed(7)
    train_scene = _scene(
        "toy-train-layout",
        ((5, 5), (23, 23)),
        ((4, 18), (16, 14), (28, 8)),
    )
    validation_scene = attenuate_target(
        _scene(
            "toy-validation-layout",
            ((7, 20), (21, 4)),
            ((3, 3), (15, 15), (27, 27)),
        ),
        1,
        sample_id="toy-validation-factual-miss",
    )
    test_scene = attenuate_target(
        _scene(
            "toy-test-layout",
            ((8, 19), (20, 5)),
            ((2, 25), (14, 3), (28, 15)),
        ),
        1,
        sample_id="toy-test-factual-miss",
    )
    assert not torch.equal(train_scene.gt_mask, validation_scene.gt_mask)
    assert not torch.equal(train_scene.gt_mask, test_scene.gt_mask)
    assert not torch.equal(validation_scene.gt_mask, test_scene.gt_mask)

    base = ToyFrozenBaseAdapter()
    frozen_before = {
        name: value.detach().clone() for name, value in base.state_dict().items()
    }
    train_gt = instances_from_binary_mask(train_scene.gt_mask)
    train_output = base(train_scene.image_batch())
    train_occupancy, train_prediction = build_occupancy(train_output.probability)
    train_matching = match_components(train_prediction, train_gt)
    no_miss_supervision = build_factual_supervision(
        train_occupancy, train_gt, train_matching
    )
    no_miss_batch = BranchBatch.from_supervision(
        train_output.feature, no_miss_supervision
    )

    search = search_minimal_legal_intervention(
        sample_id=train_scene.sample_id,
        image=train_scene.image_batch(),
        gt=train_gt,
        target_gt_id=1,
        base=base,
    )
    assert search.state is not None
    synthetic_state = search.state
    synthetic_batch = synthetic_state.branch_batch()

    # The transformed feature is re-evaluated and weaker than the successfully
    # detected source feature; occupancy was not directly state-deleted.
    selected_train = train_gt.by_id(1).mask
    assert float(synthetic_state.feature[0, 0][selected_train].mean()) < float(
        train_output.feature[0, 0][selected_train].mean()
    )
    rerun = base(synthetic_state.transformed_image)
    assert torch.equal(synthetic_state.probability, rerun.probability)
    assert torch.equal(synthetic_state.feature, rerun.feature)

    decoder = CURELiteDecoder(feature_channels=base.feature_channels)
    criterion = CURELiteLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
    # Omit factual_miss on purpose: this test isolates whether the generated
    # counterfactual branch itself can transfer to a different factual scene.
    batches = {
        "factual_no_miss": no_miss_batch,
        "synthetic": synthetic_batch,
    }
    first_logs = multi_branch_train_step(decoder, criterion, optimizer, batches)
    final_logs = first_logs
    for _ in range(49):
        final_logs = multi_branch_train_step(decoder, criterion, optimizer, batches)

    assert first_logs["factual_miss/active"] == 0
    assert float(final_logs["total"]) < 0.02 * float(first_logs["total"])
    assert all(
        torch.equal(value, frozen_before[name])
        for name, value in base.state_dict().items()
    )
    assert all(not parameter.requires_grad for parameter in base.parameters())

    model = CURELiteModel(base, decoder)
    model.eval()
    validation_gt = instances_from_binary_mask(validation_scene.gt_mask)
    residual_threshold = _select_validation_threshold(
        model, validation_scene, validation_gt
    )
    assert residual_threshold == 0.98

    test_gt = instances_from_binary_mask(test_scene.gt_mask)
    selected_test = test_gt.by_id(1).mask
    background = ~test_scene.gt_mask[0]
    with torch.no_grad():
        repaired = model(
            test_scene.image_batch(), residual_threshold=residual_threshold
        )
    assert not torch.any(repaired.residual_mask & repaired.occupancy)
    assert int(torch.count_nonzero(repaired.residual_mask[0, 0][background])) == 0
    assert int(torch.count_nonzero(repaired.residual_mask[0, 0][selected_test])) == 9
    final_matching = match_components(
        instances_from_binary_mask(repaired.final_mask[0, 0].cpu()), test_gt
    )
    assert final_matching.matched_gt_ids == frozenset({1, 2})

    # A global threshold low enough to recover the missed 3x3 target also
    # accepts all equal-intensity point distractors, unlike the toy repair.
    base_probability = base(test_scene.image_batch()).probability[0, 0]
    low_threshold = float(base_probability[selected_test].mean())
    low_threshold_mask = base_probability >= low_threshold
    assert int(torch.count_nonzero(low_threshold_mask[background])) == 3
    assert int(torch.count_nonzero(repaired.final_mask[0, 0][background])) == 0
