from __future__ import annotations

import hashlib

import torch

from cure_lite.config import LossConfig
from cure_lite.decoder import CURELiteDecoder
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


_TRAIN_UPDATES = 160
_LEARNING_RATE = 2.0e-3


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _learnable_outcome_toy() -> tuple[
    OutcomePairBatch,
    dict[str, BranchBatch],
]:
    """Return the fixed 4/4/2 toy with one clean and one component outcome."""

    height = width = 6
    channels = 6
    feature = torch.zeros(2, channels, height, width)

    # The clean source has separate frozen evidence for its deleted target and
    # an already-completable target.  The component source instead identifies
    # a false component and a separate completable target.  Pair kind is never
    # passed to the decoder or criterion.
    feature[0, 0, 1, 1] = 5.0
    feature[0, 1, 4, 0:2] = 5.0
    feature[0, 4] = 1.0
    feature[1, 2, 3:5, 3:5] = 3.0
    feature[1, 3, 0, 4:6] = 5.0
    feature[1, 5] = 1.0

    occupancy_plus = torch.zeros(
        2,
        1,
        height,
        width,
        dtype=torch.bool,
    )
    occupancy_plus[0, 0, 1:3, 1:3] = True
    occupancy_plus[1, 0, 3:5, 3:5] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)

    increment = torch.zeros(2, 1, height, width)
    increment[0, 0, 1, 1] = 1.0
    valid = torch.ones_like(occupancy_plus)
    pair_batch = PairBatch(
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        label_increment=increment,
        image_valid_mask=valid,
        pair_ids=(_sha("oc-apto-clean"), _sha("oc-apto-component")),
        sample_ids=("oc-apto-clean-source", "oc-apto-component-source"),
        group_ids=("oc-apto-clean-group", "oc-apto-component-group"),
        pair_kinds=("clean_positive", "component_null"),
        projection_visible=(True, True),
    )

    completion_plus = torch.zeros_like(occupancy_plus)
    completion_plus[0, 0, 4, 0:2] = True
    completion_plus[1, 0, 0, 4:6] = True
    completion_minus = completion_plus.clone()
    completion_minus[0, 0, 1, 1] = True
    gt_union = completion_minus.clone()
    outcome = OutcomePairBatch(
        pair_batch=pair_batch,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        gt_union=gt_union,
        intervention_footprint=direct_projected_intervention_footprint(
            pair_batch
        ),
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
            feature=feature[0:1].repeat(4, 1, 1, 1),
            occupancy=factual_occupancy,
            target=completion_minus[0:1].to(dtype=torch.float32).repeat(
                4,
                1,
                1,
                1,
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
    return outcome, factual


def _endpoint_logits(
    decoder: CURELiteDecoder,
    outcome: OutcomePairBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _paired_endpoint_logits(
        decoder,
        feature=outcome.pair_batch.feature,
        occupancy_plus=outcome.pair_batch.occupancy_plus,
        occupancy_minus=outcome.pair_batch.occupancy_minus,
    )


def test_outcome_complete_objective_overfits_mixed_outcome_toy() -> None:
    """The real v3 step must learn anchors, clean response, and both null strata."""

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 2))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4431)
            outcome, factual = _learnable_outcome_toy()
            decoder = CURELiteDecoder(feature_channels=6)
            absolute = CURELiteLoss()
            criterion = OutcomeCompleteTransitionLoss(LossConfig())
            optimizer = torch.optim.Adam(
                decoder.parameters(),
                lr=_LEARNING_RATE,
            )

            assert factual["factual_miss"].feature.shape[0] == 4
            assert factual["factual_no_miss"].feature.shape[0] == 4
            assert outcome.pair_batch.feature.shape[0] == 2
            assert outcome.pair_batch.pair_kinds == (
                "clean_positive",
                "component_null",
            )
            assert outcome.response_stratum.flatten(1).sum().tolist() == 1
            assert outcome.local_zero_stratum.flatten(1).sum().tolist() == 7

            initial_plus, initial_minus = _endpoint_logits(decoder, outcome)
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

            logs: dict[str, float | int] = {}
            for _ in range(_TRAIN_UPDATES):
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
                logits_plus, logits_minus = _endpoint_logits(decoder, outcome)
                score_plus = torch.sigmoid(logits_plus)
                score_minus = torch.sigmoid(logits_minus)
                delta = score_minus - score_plus

            clean = slice(0, 1)
            component = slice(1, 2)
            anchor_background = (
                outcome.pair_batch.image_valid_mask
                & ~outcome.pair_batch.occupancy_plus
                & ~outcome.gt_union
            )

            assert float(logs["total"]) < 0.06
            assert {
                int(state["step"].item())
                for state in optimizer.state.values()
            } == {_TRAIN_UPDATES}

            # Plus-state absolute anchor: retain both completions and suppress
            # all writable non-GT background.
            assert float(score_plus[outcome.completion_plus].min()) > 0.98
            assert float(score_plus[anchor_background].max()) < 0.04

            # The clean deletion must cause a strong positive response on D,
            # while the rest of its direct footprint H and remote field G stay
            # nearly invariant.
            assert float(
                delta[clean][outcome.response_stratum[clean]].min()
            ) > 0.90
            assert float(
                delta[clean][outcome.local_zero_stratum[clean]].abs().max()
            ) < 0.04
            assert float(
                delta[clean][outcome.global_zero_stratum[clean]].abs().max()
            ) < 0.025

            # A component-null deletion has no D.  The same unified loss must
            # keep both its local intervention footprint and global field at
            # zero response without dispatching on pair kind.
            assert not bool(outcome.response_stratum[component].any())
            assert float(
                delta[component][
                    outcome.local_zero_stratum[component]
                ].abs().max()
            ) < 0.04
            assert float(
                delta[component][
                    outcome.global_zero_stratum[component]
                ].abs().max()
            ) < 0.025
    finally:
        torch.set_num_threads(previous_threads)
