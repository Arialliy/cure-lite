from __future__ import annotations

import hashlib

import torch

from cure_lite.decoder import CURELiteDecoder
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_transition_losses import AnchoredTransitionLoss
from cure_lite.paired_transition_types import AnchoredPairBatch
from cure_lite.paired_types import PairBatch
from cure_lite.train.paired_step import paired_endpoint_logits
from cure_lite.train.paired_transition_step import (
    anchored_transition_train_step,
)
from cure_lite.train.step import BranchBatch


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _learnable_anchored_toy() -> tuple[
    AnchoredPairBatch,
    dict[str, BranchBatch],
]:
    height = width = 4
    channels = 3
    increment_locations = ((1, 1), (2, 2))
    completion_locations = ((2, 0), (1, 3))

    feature = torch.zeros(2, channels, height, width)
    occupancy_plus = torch.zeros(
        2,
        1,
        height,
        width,
        dtype=torch.bool,
    )
    occupancy_minus = torch.zeros_like(occupancy_plus)
    increment = torch.zeros(2, 1, height, width)
    completion_plus = torch.zeros_like(occupancy_plus)
    gt_union = torch.zeros_like(occupancy_plus)
    for index, ((new_row, new_column), (old_row, old_column)) in enumerate(
        zip(
            increment_locations,
            completion_locations,
            strict=True,
        )
    ):
        # Channel identities make the absolute plus state and the transition
        # jointly learnable without giving either endpoint a separate feature.
        feature[index, 0, new_row, new_column] = 4.0
        feature[index, 1, old_row, old_column] = 4.0
        feature[index, 2] = 0.15 * (index + 1)
        occupancy_plus[index, 0, new_row, new_column] = True
        increment[index, 0, new_row, new_column] = 1.0
        completion_plus[index, 0, old_row, old_column] = True
        gt_union[index] = (
            completion_plus[index]
            | increment[index].to(dtype=torch.bool)
        )

    pair_batch = PairBatch(
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        label_increment=increment,
        image_valid_mask=torch.ones_like(occupancy_plus),
        pair_ids=(_sha("apto-pair-0"), _sha("apto-pair-1")),
        sample_ids=("apto-source-0", "apto-source-1"),
        group_ids=("apto-group-0", "apto-group-1"),
        pair_kinds=("clean_positive", "clean_positive"),
        projection_visible=(True, True),
    )
    anchored = AnchoredPairBatch(
        pair_batch=pair_batch,
        completion_plus=completion_plus,
        gt_union=gt_union,
    )

    factual_feature = torch.stack(
        [feature[index % 2] for index in range(4)],
    )
    factual_target = torch.stack(
        [
            (
                completion_plus[index % 2]
                | increment[index % 2].to(dtype=torch.bool)
            ).to(dtype=torch.float32)
            for index in range(4)
        ],
    )
    factual_occupancy = torch.zeros(
        4,
        1,
        height,
        width,
        dtype=torch.bool,
    )
    factual_valid = torch.ones_like(factual_occupancy)
    factual = {
        "factual_miss": BranchBatch(
            feature=factual_feature,
            occupancy=factual_occupancy,
            target=factual_target,
            valid_mask=factual_valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=torch.zeros(4, channels, height, width),
            occupancy=factual_occupancy.clone(),
            target=torch.zeros_like(factual_target),
            valid_mask=factual_valid.clone(),
        ),
    }
    return anchored, factual


def _joint_objective_and_outputs(
    decoder: CURELiteDecoder,
    anchored: AnchoredPairBatch,
    factual: dict[str, BranchBatch],
    absolute: CURELiteLoss,
    transition: AnchoredTransitionLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    miss = factual["factual_miss"]
    no_miss = factual["factual_no_miss"]
    miss_loss = absolute(
        decoder(miss.feature, miss.occupancy),
        miss.target,
        miss.valid_mask,
    )["total"]
    no_miss_loss = absolute(
        decoder(no_miss.feature, no_miss.occupancy),
        no_miss.target,
        no_miss.valid_mask,
    )["total"]
    logits_plus, logits_minus = paired_endpoint_logits(
        decoder,
        anchored.pair_batch,
    )
    pair_loss = transition(
        logits_plus,
        logits_minus,
        anchored.completion_plus,
        anchored.occupancy_plus,
        anchored.gt_union,
        anchored.label_increment,
        anchored.image_valid_mask,
    )["total"]
    return miss_loss + no_miss_loss + pair_loss, logits_plus, logits_minus


def test_anchored_transition_joint_objective_overfits_deterministic_toy() -> None:
    """APTO must learn absolute state and transition with both endpoints active."""

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 2))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1701)
            anchored, factual = _learnable_anchored_toy()
            decoder = CURELiteDecoder(feature_channels=3)
            absolute = CURELiteLoss()
            transition = AnchoredTransitionLoss()
            optimizer = torch.optim.Adam(decoder.parameters(), lr=2e-3)

            initial_total, initial_plus, initial_minus = (
                _joint_objective_and_outputs(
                    decoder,
                    anchored,
                    factual,
                    absolute,
                    transition,
                )
            )
            initial_total_value = float(initial_total.detach())
            initial_pair_loss = transition(
                initial_plus,
                initial_minus,
                anchored.completion_plus,
                anchored.occupancy_plus,
                anchored.gt_union,
                anchored.label_increment,
                anchored.image_valid_mask,
            )["total"]
            plus_grad, minus_grad = torch.autograd.grad(
                initial_pair_loss,
                (initial_plus, initial_minus),
            )
            assert torch.isfinite(plus_grad).all()
            assert torch.isfinite(minus_grad).all()
            assert torch.count_nonzero(plus_grad) > 0
            assert torch.count_nonzero(minus_grad) > 0

            logs: dict[str, float | int] = {}
            for _ in range(220):
                logs = anchored_transition_train_step(
                    decoder,
                    absolute,
                    transition,
                    optimizer,
                    factual,
                    anchored,
                )

            decoder.eval()
            with torch.no_grad():
                final_total, logits_plus, logits_minus = (
                    _joint_objective_and_outputs(
                        decoder,
                        anchored,
                        factual,
                        absolute,
                        transition,
                    )
                )
                score_plus = torch.sigmoid(logits_plus)
                score_minus = torch.sigmoid(logits_minus)
                delta = score_minus - score_plus

            positive = anchored.label_increment.to(dtype=torch.bool)
            plus_anchor = anchored.completion_plus
            transition_background = anchored.image_valid_mask & ~positive
            anchor_background = (
                anchored.image_valid_mask
                & ~anchored.occupancy_plus
                & ~anchored.gt_union
            )

            assert float(logs["total"]) < 0.08
            assert float(final_total) < 0.05 * initial_total_value
            assert float(score_plus[plus_anchor].min()) > 0.95
            assert float(score_plus[anchor_background].mean()) < 0.01
            assert float(delta[positive].min()) > 0.90
            assert float(delta[transition_background].abs().mean()) < 0.01
            # The positive transition cannot be supplied by only one endpoint:
            # plus is suppressed while minus is activated at the same pixels.
            assert float(score_plus[positive].max()) < 0.05
            assert float(score_minus[positive].min()) > 0.95
    finally:
        torch.set_num_threads(previous_threads)
