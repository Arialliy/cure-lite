from __future__ import annotations

import hashlib

import torch

from cure_lite.decoder import CURELiteDecoder
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_losses import PairedDifferenceLoss
from cure_lite.paired_types import PairBatch
from cure_lite.train.paired_step import paired_endpoint_logits, paired_train_step
from cure_lite.train.step import BranchBatch


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _learnable_toy() -> tuple[PairBatch, dict[str, BranchBatch]]:
    height = width = 4
    channels = 2
    locations = ((1, 1), (2, 2))
    feature = torch.zeros(2, channels, height, width)
    occupancy_plus = torch.zeros(2, 1, height, width, dtype=torch.bool)
    occupancy_minus = torch.zeros_like(occupancy_plus)
    increment = torch.zeros(2, 1, height, width)
    for index, (row, column) in enumerate(locations):
        feature[index, 0, row, column] = 4.0
        feature[index, 1] = 0.2
        occupancy_plus[index, 0, row, column] = True
        increment[index, 0, row, column] = 1.0
    pair_batch = PairBatch(
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        label_increment=increment,
        image_valid_mask=torch.ones_like(occupancy_plus),
        pair_ids=(_sha("toy-pair-0"), _sha("toy-pair-1")),
        sample_ids=("toy-source-0", "toy-source-1"),
        group_ids=("toy-group-0", "toy-group-1"),
        pair_kinds=("clean_positive", "clean_positive"),
        projection_visible=(True, True),
    )

    factual_occupancy = torch.zeros(
        4,
        1,
        height,
        width,
        dtype=torch.bool,
    )
    factual_valid = torch.ones_like(factual_occupancy)
    factual_batches = {
        "factual_miss": BranchBatch(
            feature=torch.stack(
                [feature[index % 2] for index in range(4)],
            ),
            occupancy=factual_occupancy,
            target=torch.stack(
                [increment[index % 2] for index in range(4)],
            ),
            valid_mask=factual_valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=torch.zeros(4, channels, height, width),
            occupancy=factual_occupancy.clone(),
            target=torch.zeros(4, 1, height, width),
            valid_mask=factual_valid.clone(),
        ),
    }
    return pair_batch, factual_batches


def test_frozen_joint_objective_overfits_a_two_source_toy() -> None:
    """The exact 4/4/2, 1:1:1 objective must be jointly learnable."""

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 2))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            pair_batch, factual_batches = _learnable_toy()
            decoder = CURELiteDecoder(feature_channels=2)
            absolute = CURELiteLoss()
            paired = PairedDifferenceLoss()
            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

            with torch.no_grad():
                initial_plus, initial_minus = paired_endpoint_logits(
                    decoder,
                    pair_batch,
                )
                initial_pair_loss = paired(
                    initial_plus,
                    initial_minus,
                    pair_batch.label_increment,
                    pair_batch.image_valid_mask,
                )["total"]

            logs: dict[str, float | int] = {}
            for _ in range(220):
                logs = paired_train_step(
                    decoder,
                    absolute,
                    paired,
                    optimizer,
                    factual_batches,
                    pair_batch,
                )

            decoder.eval()
            with torch.no_grad():
                logits_plus, logits_minus = paired_endpoint_logits(
                    decoder,
                    pair_batch,
                )
                result = paired(
                    logits_plus,
                    logits_minus,
                    pair_batch.label_increment,
                    pair_batch.image_valid_mask,
                )
                delta = torch.sigmoid(logits_minus) - torch.sigmoid(logits_plus)
                response = pair_batch.label_increment.to(torch.bool)
                background = pair_batch.image_valid_mask & ~response

            assert float(logs["total"]) < 0.1
            assert float(result["total"]) < 1e-3
            assert float(result["total"]) < 0.01 * float(initial_pair_loss)
            assert float(delta[response].min()) > 0.95
            assert float(delta[background].abs().mean()) < 0.01
    finally:
        torch.set_num_threads(previous_threads)
