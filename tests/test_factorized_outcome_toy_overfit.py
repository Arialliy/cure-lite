from __future__ import annotations

import hashlib

import pytest
import torch

from cure_lite.config import LossConfig
from cure_lite.factorized_decoder import CURELiteFactorizedDecoder
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_outcome_losses import OutcomeCompleteTransitionLoss
from cure_lite.paired_outcome_types import (
    OutcomePairBatch,
    direct_projected_intervention_footprint,
)
from cure_lite.paired_types import PairBatch
from cure_lite.train.paired_outcome_step import outcome_complete_train_step
from cure_lite.train.paired_step import _paired_endpoint_logits
from cure_lite.train.step import BranchBatch


_TOY_UPDATES = 320
_TOY_LEARNING_RATE = 4.0e-3


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _subpixel_outcome_toy(
    clean_pixels: tuple[tuple[int, int], ...] = ((1, 2),),
) -> tuple[
    OutcomePairBatch,
    dict[str, BranchBatch],
]:
    """Return one clean and one component-null 4x subpixel problem."""

    feature = torch.zeros(2, 8, 2, 2)
    # Clean source: one deleted-target feature, plus one independent factual
    # completion.  The desired D occupies one phase of the top-left cell.
    feature[0, 0, 0, 0] = 5.0
    feature[0, 1, 1, 0] = 4.0
    feature[0, 6] = 0.5
    # Component-null source: the removed component has a distinct feature,
    # while a valid completion occupies another cell/phase.
    feature[1, 2, 1, 1] = 5.0
    feature[1, 3, 0, 1] = 4.0
    feature[1, 7] = 0.5

    occupancy_plus = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    occupancy_plus[0, 0, 0:4, 0:4] = True
    occupancy_plus[1, 0, 4:8, 4:8] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)

    completion_plus = torch.zeros_like(occupancy_plus)
    completion_plus[0, 0, 5, 1] = True
    completion_plus[1, 0, 1, 6] = True
    completion_minus = completion_plus.clone()
    for row, column in clean_pixels:
        completion_minus[0, 0, row, column] = True
    increment = (completion_minus & ~completion_plus).to(torch.float32)
    valid = torch.ones_like(occupancy_plus)

    pair_batch = PairBatch(
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        label_increment=increment,
        image_valid_mask=valid,
        pair_ids=(_sha("svef-clean"), _sha("svef-component")),
        sample_ids=("svef-clean-source", "svef-component-source"),
        group_ids=("svef-clean-group", "svef-component-group"),
        pair_kinds=("clean_positive", "component_null"),
        projection_visible=(True, True),
    )
    outcome = OutcomePairBatch(
        pair_batch=pair_batch,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        gt_union=completion_minus.clone(),
        intervention_footprint=direct_projected_intervention_footprint(
            pair_batch
        ),
    )

    factual_occupancy = torch.zeros(4, 1, 8, 8, dtype=torch.bool)
    factual_valid = torch.ones_like(factual_occupancy)
    no_miss_feature = torch.zeros(4, 8, 2, 2)
    no_miss_feature[:, 4, 0, 1] = 3.0
    no_miss_feature[:, 5] = -0.5
    factual = {
        "factual_miss": BranchBatch(
            feature=feature[0:1].repeat(4, 1, 1, 1),
            occupancy=factual_occupancy,
            target=completion_minus[0:1]
            .to(torch.float32)
            .repeat(4, 1, 1, 1),
            valid_mask=factual_valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=no_miss_feature,
            occupancy=factual_occupancy.clone(),
            target=torch.zeros(4, 1, 8, 8),
            valid_mask=factual_valid.clone(),
        ),
    }
    return outcome, factual


@pytest.mark.parametrize(
    "clean_pixels",
    [
        ((1, 2),),
        ((1, 2), (2, 1)),
        ((1, 2), (2, 1), (2, 2)),
    ],
    ids=("one_pixel", "two_pixels", "three_pixels"),
)
def test_svef_overfits_subpixel_clean_and_component_null_together(
    clean_pixels: tuple[tuple[int, int], ...],
) -> None:
    """One decoder must learn 1--3 subcell pixels without a null halo."""

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 2))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7817)
            outcome, factual = _subpixel_outcome_toy(clean_pixels)
            decoder = CURELiteFactorizedDecoder(
                feature_channels=8,
                feature_stride=4,
            )
            absolute = CURELiteLoss()
            criterion = OutcomeCompleteTransitionLoss(LossConfig())
            optimizer = torch.optim.Adam(
                decoder.parameters(),
                lr=_TOY_LEARNING_RATE,
            )

            initial_plus, initial_minus = _paired_endpoint_logits(
                decoder,
                feature=outcome.pair_batch.feature,
                occupancy_plus=outcome.pair_batch.occupancy_plus,
                occupancy_minus=outcome.pair_batch.occupancy_minus,
            )
            initial_result = criterion(
                initial_plus,
                initial_minus,
                outcome.completion_plus,
                outcome.pair_batch.occupancy_plus,
                outcome.gt_union,
                outcome.pair_batch.label_increment,
                outcome.pair_batch.image_valid_mask,
                outcome.intervention_footprint,
            )
            plus_gradient, minus_gradient = torch.autograd.grad(
                initial_result["total"],
                (initial_plus, initial_minus),
            )
            assert torch.isfinite(plus_gradient).all()
            assert torch.isfinite(minus_gradient).all()
            assert torch.count_nonzero(plus_gradient) > 0
            assert torch.count_nonzero(minus_gradient) > 0

            for _ in range(_TOY_UPDATES):
                logs = outcome_complete_train_step(
                    decoder,
                    absolute,
                    criterion,
                    optimizer,
                    factual,
                    outcome,
                )

            decoder.eval()
            with torch.no_grad():
                logits_plus, logits_minus = _paired_endpoint_logits(
                    decoder,
                    feature=outcome.pair_batch.feature,
                    occupancy_plus=outcome.pair_batch.occupancy_plus,
                    occupancy_minus=outcome.pair_batch.occupancy_minus,
                )
                score_plus = torch.sigmoid(logits_plus)
                score_minus = torch.sigmoid(logits_minus)
                delta = score_minus - score_plus
                factual_miss_score = torch.sigmoid(
                    decoder(
                        factual["factual_miss"].feature,
                        factual["factual_miss"].occupancy,
                    )
                )
                factual_no_miss_score = torch.sigmoid(
                    decoder(
                        factual["factual_no_miss"].feature,
                        factual["factual_no_miss"].occupancy,
                    )
                )

            clean = slice(0, 1)
            component = slice(1, 2)
            clean_D = outcome.response_stratum[clean]
            clean_H = outcome.local_zero_stratum[clean]
            clean_G = outcome.global_zero_stratum[clean]
            component_H = outcome.local_zero_stratum[component]
            component_G = outcome.global_zero_stratum[component]
            anchor_background = (
                outcome.pair_batch.image_valid_mask
                & ~outcome.pair_batch.occupancy_plus
                & ~outcome.gt_union
            )

            assert float(logs["total"]) < 0.10
            assert float(score_plus[outcome.completion_plus].min()) > 0.95
            assert float(score_plus[anchor_background].max()) < 0.05
            factual_miss_target = factual["factual_miss"].target > 0.5
            factual_miss_background = (
                factual["factual_miss"].valid_mask
                & ~factual_miss_target
            )
            assert float(
                factual_miss_score[factual_miss_target].min()
            ) > 0.95
            assert float(
                factual_miss_score[factual_miss_background].max()
            ) < 0.05
            assert float(factual_no_miss_score.max()) < 0.05
            assert float(delta[clean][clean_D].mean()) >= 0.8
            assert float(delta[clean][clean_H].abs().max()) <= 0.05
            assert float(delta[clean][clean_G].abs().max()) <= 0.05
            assert not bool(outcome.response_stratum[component].any())
            assert float(delta[component][component_H].abs().max()) <= 0.05
            assert float(delta[component][component_G].abs().max()) <= 0.05
    finally:
        torch.set_num_threads(previous_threads)
