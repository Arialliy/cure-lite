from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.factorized_decoder import CURELiteFactorizedDecoder
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_outcome_types import OutcomePairBatch
from cure_lite.train.paired_outcome_step import outcome_complete_train_step
from tests.test_paired_outcome_step import (
    _CountingAdam,
    _EndpointAttachmentCriterion,
    _factual,
    _outcome,
)


class _RecordingFactorizedDecoder(CURELiteFactorizedDecoder):
    def __init__(self) -> None:
        super().__init__(feature_channels=2, feature_stride=4)
        self.forward_inputs: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []

    def forward(
        self,
        feature: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_inputs.append(
            (
                feature.detach().clone(),
                occupancy.detach().clone(),
            )
        )
        return super().forward(feature, occupancy)


def test_existing_outcome_step_accepts_factorized_decoder_unchanged() -> None:
    outcome = _outcome(("clean_positive", "component_null"))
    factual = _factual()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4901)
        decoder = _RecordingFactorizedDecoder()
    optimizer = _CountingAdam(decoder.parameters(), lr=1.0e-3)
    criterion = _EndpointAttachmentCriterion()
    before = [
        parameter.detach().clone()
        for parameter in decoder.parameters()
    ]

    logs = outcome_complete_train_step(
        decoder,
        CURELiteLoss(),
        criterion,
        optimizer,
        factual,
        outcome,
    )

    assert len(decoder.forward_inputs) == 3
    assert [value[0].shape[0] for value in decoder.forward_inputs] == [
        4,
        4,
        4,
    ]
    assert logs["decoder_forward_calls_per_update"] == 3
    assert logs["decoder_states_per_update"] == 12
    assert logs["backward_calls"] == 1
    assert logs["optimizer_steps"] == 1
    assert optimizer.step_calls == 1
    assert criterion.endpoint_attachment == (True, True)
    assert any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(
            before,
            decoder.parameters(),
            strict=True,
        )
    )
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in decoder.parameters()
    )

    pair_feature, pair_occupancy = decoder.forward_inputs[2]
    torch.testing.assert_close(
        pair_feature,
        torch.cat(
            (
                outcome.pair_batch.feature,
                outcome.pair_batch.feature,
            ),
            dim=0,
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        pair_occupancy,
        torch.cat(
            (
                outcome.pair_batch.occupancy_plus,
                outcome.pair_batch.occupancy_minus,
            ),
            dim=0,
        ),
    )


def test_factorized_preflight_failure_has_zero_training_side_effects() -> None:
    outcome = _outcome(("clean_positive", "component_null"))
    duplicate_sources = replace(
        outcome.pair_batch,
        sample_ids=(
            outcome.pair_batch.sample_ids[0],
            outcome.pair_batch.sample_ids[0],
        ),
    )
    malformed = OutcomePairBatch(
        pair_batch=duplicate_sources,
        completion_plus=outcome.completion_plus,
        completion_minus=outcome.completion_minus,
        gt_union=outcome.gt_union,
        intervention_footprint=outcome.intervention_footprint,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4902)
        decoder = _RecordingFactorizedDecoder()
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    optimizer = _CountingAdam(decoder.parameters(), lr=1.0e-3)
    parameters_before = [
        parameter.detach().clone()
        for parameter in decoder.parameters()
    ]
    gradients_before = [
        parameter.grad.detach().clone()
        for parameter in decoder.parameters()
    ]

    with pytest.raises(ValueError, match="distinct source"):
        outcome_complete_train_step(
            decoder,
            CURELiteLoss(),
            _EndpointAttachmentCriterion(),
            optimizer,
            _factual(),
            malformed,
        )

    assert decoder.training is False
    assert decoder.forward_inputs == []
    assert optimizer.step_calls == 0
    assert optimizer.state == {}
    assert all(
        torch.equal(before, parameter.detach())
        for before, parameter in zip(
            parameters_before,
            decoder.parameters(),
            strict=True,
        )
    )
    assert all(
        torch.equal(before, parameter.grad)
        for before, parameter in zip(
            gradients_before,
            decoder.parameters(),
            strict=True,
        )
    )
