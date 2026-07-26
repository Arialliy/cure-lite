from __future__ import annotations

import pytest
import torch

from cure_lite.config import LossConfig
from cure_lite.conservative_factorized_decoder import (
    CURELiteConservativeFactorizedDecoder,
)
from cure_lite.experiment.conservative_toy_inputs import (
    SUPPORT_FAMILY,
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
        ((1, 6),),
        ((1, 6), (2, 5)),
        ((1, 6), (2, 5), (2, 6)),
    ],
    ids=("one_pixel", "two_pixels", "three_pixels"),
)
def test_cc_sea_recovers_response_outside_removed_component(
    clean_pixels: tuple[tuple[int, int], ...],
) -> None:
    """The conserved budget must retain v7's complete count support."""

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 2))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7817)
            outcome, factual = build_conservative_toy_case(
                SUPPORT_FAMILY,
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
                delta = (
                    torch.sigmoid(logits_minus)
                    - torch.sigmoid(logits_plus)
                )

            clean = slice(0, 1)
            component = slice(1, 2)
            clean_D = outcome.response_stratum[clean]
            clean_H = outcome.local_zero_stratum[clean]
            clean_G = outcome.global_zero_stratum[clean]
            component_H = outcome.local_zero_stratum[component]
            component_G = outcome.global_zero_stratum[component]

            assert float(logs["total"]) < 0.10
            assert float(delta[clean][clean_D].mean()) >= 0.80
            assert float(delta[clean][clean_H].abs().max()) <= 0.05
            assert float(delta[clean][clean_G].abs().max()) <= 0.05
            assert (
                float(delta[component][component_H].abs().max())
                <= 0.05
            )
            assert (
                float(delta[component][component_G].abs().max())
                <= 0.05
            )
            assert not bool(
                outcome.response_stratum[component].any()
            )
            assert not bool(
                (
                    outcome.response_stratum[clean]
                    & outcome.removed_component[clean]
                ).any()
            )
    finally:
        torch.set_num_threads(previous_threads)
