from __future__ import annotations

import pytest
import torch

from cure_lite.config import LossConfig
from cure_lite.conservative_factorized_decoder import (
    CURELiteConservativeFactorizedDecoder,
)
from cure_lite.experiment.conservative_toy_inputs import (
    LEGACY_FAMILY,
    build_conservative_toy_case,
)
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_outcome_losses import (
    OutcomeCompleteTransitionLoss,
)
from cure_lite.train.paired_outcome_step import (
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import _paired_endpoint_logits


@pytest.mark.parametrize(
    "clean_pixels",
    [
        ((1, 2),),
        ((1, 2), (2, 1)),
        ((1, 2), (2, 1), (2, 2)),
    ],
    ids=("one_pixel", "two_pixels", "three_pixels"),
)
def test_cc_sea_overfits_subpixel_clean_and_component_null_together(
    clean_pixels: tuple[tuple[int, int], ...],
) -> None:
    """One conserved budget must select 1--3 phases without a null halo."""

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 2))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7817)
            outcome, factual = build_conservative_toy_case(
                LEGACY_FAMILY,
                clean_pixels,
            )
            decoder = CURELiteConservativeFactorizedDecoder(
                feature_channels=8,
                feature_stride=4,
            )
            absolute = CURELiteLoss()
            criterion = OutcomeCompleteTransitionLoss(LossConfig())
            optimizer = torch.optim.Adam(
                decoder.parameters(),
                lr=4.0e-3,
            )

            for _ in range(320):
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
            factual_target = factual["factual_miss"].target > 0.5
            factual_background = (
                factual["factual_miss"].valid_mask & ~factual_target
            )
            assert float(
                factual_miss_score[factual_target].min()
            ) > 0.95
            assert float(
                factual_miss_score[factual_background].max()
            ) < 0.05
            assert float(factual_no_miss_score.max()) < 0.05
            assert float(delta[clean][clean_D].mean()) >= 0.80
            assert float(delta[clean][clean_H].abs().max()) <= 0.05
            assert float(delta[clean][clean_G].abs().max()) <= 0.05
            assert not bool(outcome.response_stratum[component].any())
            assert (
                float(delta[component][component_H].abs().max())
                <= 0.05
            )
            assert (
                float(delta[component][component_G].abs().max())
                <= 0.05
            )
    finally:
        torch.set_num_threads(previous_threads)
