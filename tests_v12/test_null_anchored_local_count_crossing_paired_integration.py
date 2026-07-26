from __future__ import annotations

import inspect

import torch

from cure_lite.ccfr_development_inputs import (
    SUPPORT_FAMILY,
    build_conservative_toy_case,
)
from cure_lite.config import LossConfig
from cure_lite.experiment.peco_exposure_confirmation import (
    build_identical_input_conflict_control,
)
from cure_lite.losses import CURELiteLoss
from cure_lite.null_anchored_local_count_crossing_decoder import (
    CURELiteNullAnchoredLocalCountCrossingDecoder,
)
from cure_lite.paired_endpoint_crossing_losses import (
    PairedEndpointCrossingLoss,
)
from cure_lite.train.paired_outcome_step import (
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import _paired_endpoint_logits


class _RecordingNLCCDecoder(
    CURELiteNullAnchoredLocalCountCrossingDecoder
):
    def __init__(self) -> None:
        super().__init__(feature_channels=8, feature_stride=4)
        self.forward_batch_sizes: list[int] = []
        self.forward_outputs: list[torch.Tensor] = []

    def forward(
        self,
        feature: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_batch_sizes.append(int(feature.shape[0]))
        output = super().forward(feature, occupancy)
        output.retain_grad()
        self.forward_outputs.append(output)
        return output


def test_frozen_4_4_2_step_uses_one_2b_pair_forward() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12012)
        outcome, factual = build_conservative_toy_case(
            SUPPORT_FAMILY,
            ((1, 6),),
        )
        pair_feature = outcome.pair_batch.feature.clone().requires_grad_()
        outcome = type(outcome)(
            pair_batch=type(outcome.pair_batch)(
                feature=pair_feature,
                occupancy_plus=outcome.pair_batch.occupancy_plus,
                occupancy_minus=outcome.pair_batch.occupancy_minus,
                label_increment=outcome.pair_batch.label_increment,
                image_valid_mask=outcome.pair_batch.image_valid_mask,
                pair_ids=outcome.pair_batch.pair_ids,
                sample_ids=outcome.pair_batch.sample_ids,
                group_ids=outcome.pair_batch.group_ids,
                pair_kinds=outcome.pair_batch.pair_kinds,
                projection_visible=outcome.pair_batch.projection_visible,
            ),
            completion_plus=outcome.completion_plus,
            completion_minus=outcome.completion_minus,
            gt_union=outcome.gt_union,
            intervention_footprint=outcome.intervention_footprint,
        )
        for batch in factual.values():
            batch.feature.requires_grad_()

        decoder = _RecordingNLCCDecoder()
        optimizer = torch.optim.Adam(decoder.parameters(), lr=1.0e-3)
        logs = outcome_complete_train_step(
            decoder,
            CURELiteLoss(),
            PairedEndpointCrossingLoss(LossConfig()),
            optimizer,
            factual,
            outcome,
        )

    assert decoder.forward_batch_sizes == [4, 4, 4]
    assert logs["decoder_forward_calls_per_update"] == 3
    assert logs["decoder_states_per_update"] == 12
    assert logs["outcome/endpoints"] == 4
    endpoint_gradient = decoder.forward_outputs[-1].grad
    assert endpoint_gradient is not None
    assert torch.isfinite(endpoint_gradient).all()
    assert torch.count_nonzero(endpoint_gradient[:2]) > 0
    assert torch.count_nonzero(endpoint_gradient[2:]) > 0

    parameters = tuple(decoder.parameters())
    assert len(parameters) == 6
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in parameters
        if parameter.grad is not None
    )
    assert pair_feature.grad is None
    assert all(batch.feature.grad is None for batch in factual.values())


def test_identical_inputs_cannot_be_dispatched_by_pair_metadata() -> None:
    outcome = build_identical_input_conflict_control()
    torch.manual_seed(42)
    decoder = CURELiteNullAnchoredLocalCountCrossingDecoder(
        feature_channels=8,
        feature_stride=4,
    )

    with torch.no_grad():
        plus, minus = _paired_endpoint_logits(
            decoder,
            feature=outcome.pair_batch.feature,
            occupancy_plus=outcome.pair_batch.occupancy_plus,
            occupancy_minus=outcome.pair_batch.occupancy_minus,
        )

    assert torch.equal(plus[0], plus[1])
    assert torch.equal(minus[0], minus[1])
    assert "pair_kind" not in inspect.signature(decoder.forward).parameters
    assert "pair_kind" not in inspect.getsource(decoder.forward_fields)


def test_identity_pair_has_exactly_zero_endpoint_difference() -> None:
    decoder = CURELiteNullAnchoredLocalCountCrossingDecoder(
        feature_channels=8,
        feature_stride=4,
    )
    feature = torch.randn(2, 8, 2, 2)
    occupancy = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    occupancy[0, 0, 0, 0] = True

    plus, minus = _paired_endpoint_logits(
        decoder,
        feature=feature,
        occupancy_plus=occupancy,
        occupancy_minus=occupancy.clone(),
    )

    assert torch.equal(plus, minus)
    assert torch.equal(
        torch.sigmoid(minus) - torch.sigmoid(plus),
        torch.zeros_like(plus),
    )


def test_pair_gradient_is_the_sum_of_both_endpoint_vjps() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12012)
        outcome, _ = build_conservative_toy_case(
            SUPPORT_FAMILY,
            ((1, 6),),
        )
        decoder = CURELiteNullAnchoredLocalCountCrossingDecoder(
            feature_channels=8,
            feature_stride=4,
        )
        plus, minus = _paired_endpoint_logits(
            decoder,
            feature=outcome.pair_batch.feature,
            occupancy_plus=outcome.pair_batch.occupancy_plus,
            occupancy_minus=outcome.pair_batch.occupancy_minus,
        )
        result = PairedEndpointCrossingLoss(LossConfig())(
            plus,
            minus,
            outcome.completion_plus,
            outcome.pair_batch.occupancy_plus,
            outcome.gt_union,
            outcome.pair_batch.label_increment,
            outcome.pair_batch.image_valid_mask,
            outcome.intervention_footprint,
        )
        loss = result["response_stratum_loss"]
        endpoint_plus, endpoint_minus = torch.autograd.grad(
            loss,
            (plus, minus),
            retain_graph=True,
        )
        parameters = tuple(decoder.parameters())
        plus_vjp = torch.autograd.grad(
            plus,
            parameters,
            grad_outputs=endpoint_plus.detach(),
            retain_graph=True,
        )
        minus_vjp = torch.autograd.grad(
            minus,
            parameters,
            grad_outputs=endpoint_minus.detach(),
            retain_graph=True,
        )
        direct = torch.autograd.grad(loss, parameters)

    response = result["response_stratum"]
    assert torch.isfinite(endpoint_plus).all()
    assert torch.isfinite(endpoint_minus).all()
    assert torch.all(endpoint_plus[response] > 0.0)
    assert torch.all(endpoint_minus[response] < 0.0)
    for gradients in (plus_vjp, minus_vjp, direct):
        assert all(torch.isfinite(value).all() for value in gradients)
    assert torch.linalg.vector_norm(
        torch.cat([value.reshape(-1) for value in plus_vjp])
    ) > 0.0
    assert torch.linalg.vector_norm(
        torch.cat([value.reshape(-1) for value in minus_vjp])
    ) > 0.0
    for observed, plus_part, minus_part in zip(
        direct,
        plus_vjp,
        minus_vjp,
        strict=True,
    ):
        torch.testing.assert_close(
            observed,
            plus_part + minus_part,
            rtol=1.0e-5,
            atol=2.0e-8,
        )
