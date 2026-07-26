from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.config import LossConfig
from cure_lite.decoder import CURELiteDecoder
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_outcome_losses import OutcomeCompleteTransitionLoss
from cure_lite.paired_outcome_types import (
    OutcomePairBatch,
    stack_outcome_pair_examples,
)
from cure_lite.paired_types import stack_pair_examples
from cure_lite.train.paired_outcome_step import outcome_complete_train_step
from cure_lite.train.step import BranchBatch
from tests.test_paired_outcome_types import _pair


class _RecordingDecoder(CURELiteDecoder):
    def __init__(self) -> None:
        super().__init__(feature_channels=2)
        self.forward_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

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


class _CountingAdam(torch.optim.Adam):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _EndpointAttachmentCriterion(OutcomeCompleteTransitionLoss):
    def __init__(self) -> None:
        super().__init__(LossConfig())
        self.endpoint_attachment: tuple[bool, bool] | None = None

    def forward(
        self,
        logits_plus: torch.Tensor,
        logits_minus: torch.Tensor,
        completion_plus: torch.Tensor,
        occupancy_plus: torch.Tensor,
        gt_union: torch.Tensor,
        label_increment: torch.Tensor,
        image_valid_mask: torch.Tensor,
        intervention_footprint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.endpoint_attachment = (
            bool(logits_plus.requires_grad and logits_plus.grad_fn is not None),
            bool(logits_minus.requires_grad and logits_minus.grad_fn is not None),
        )
        return super().forward(
            logits_plus,
            logits_minus,
            completion_plus,
            occupancy_plus,
            gt_union,
            label_increment,
            image_valid_mask,
            intervention_footprint,
        )


def _outcome(kinds: tuple[str, str]) -> OutcomePairBatch:
    coordinates = ((1, 1), (6, 6))
    pairs = []
    unions: dict[str, torch.Tensor] = {}
    for index, (kind, component) in enumerate(
        zip(kinds, coordinates, strict=True)
    ):
        sample_id = f"{kind}-source-{index}"
        pair, gt_union = _pair(
            kind=kind,
            sample_id=sample_id,
            component=component,
        )
        pairs.append(pair)
        unions[sample_id] = gt_union
    return stack_outcome_pair_examples(
        tuple(pairs),
        gt_union_by_sample=unions,
        device="cpu",
    )


def _factual() -> dict[str, BranchBatch]:
    torch.manual_seed(2201)
    occupancy = torch.zeros(4, 1, 8, 8, dtype=torch.bool)
    valid = torch.ones_like(occupancy)
    miss_target = torch.zeros(4, 1, 8, 8)
    miss_target[:, 0, 3, 4] = 1.0
    return {
        "factual_miss": BranchBatch(
            feature=torch.randn(4, 2, 2, 2),
            occupancy=occupancy,
            target=miss_target,
            valid_mask=valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=torch.randn(4, 2, 2, 2),
            occupancy=occupancy.clone(),
            target=torch.zeros_like(miss_target),
            valid_mask=valid.clone(),
        ),
    }


def _assert_no_side_effects(
    decoder: _RecordingDecoder,
    optimizer: _CountingAdam,
    *,
    parameters_before: list[torch.Tensor],
    gradients_before: list[torch.Tensor],
) -> None:
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


def test_mixed_outcome_step_has_fixed_budget_and_decoder_input_boundary() -> None:
    outcome = _outcome(("clean_positive", "component_null"))
    decoder = _RecordingDecoder()
    optimizer = _CountingAdam(decoder.parameters(), lr=1.0e-3)
    criterion = _EndpointAttachmentCriterion()
    before = [parameter.detach().clone() for parameter in decoder.parameters()]

    logs = outcome_complete_train_step(
        decoder,
        CURELiteLoss(),
        criterion,
        optimizer,
        _factual(),
        outcome,
    )

    assert len(decoder.forward_inputs) == 3
    assert [feature.shape[0] for feature, _ in decoder.forward_inputs] == [
        4,
        4,
        4,
    ]
    assert optimizer.step_calls == 1
    assert logs["decoder_forward_calls_per_update"] == 3
    assert logs["decoder_states_per_update"] == 12
    assert logs["backward_calls"] == 1
    assert logs["optimizer_steps"] == 1
    assert logs["factual_miss/states"] == 4
    assert logs["factual_no_miss/states"] == 4
    assert logs["outcome/pairs"] == 2
    assert logs["outcome/endpoints"] == 4
    assert logs["outcome/clean_pairs"] == 1
    assert logs["outcome/component_null_pairs"] == 1
    assert logs["outcome/loss"] == pytest.approx(
        0.5
        * (
            logs["outcome/plus_anchor_loss"]
            + logs["outcome/transition_loss"]
        ),
        abs=2.0e-7,
    )
    assert logs["total"] == pytest.approx(
        logs["factual_miss/loss"]
        + logs["factual_no_miss/loss"]
        + logs["outcome/loss"],
        abs=3.0e-7,
    )
    assert criterion.endpoint_attachment == (True, True)

    # The one pair call consumes only a repeated frozen feature and the two
    # occupancy endpoints.  GT, R+/R-, D, and J never enter the decoder.
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
    assert any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, decoder.parameters(), strict=True)
    )
    assert all(parameter.grad is not None for parameter in decoder.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )


@pytest.mark.parametrize(
    ("kinds", "expected_clean", "expected_component"),
    [
        (("clean_positive", "clean_positive"), 2, 0),
        (("component_null", "component_null"), 0, 2),
    ],
)
def test_optimizer_accepts_any_clean_component_combination(
    kinds: tuple[str, str],
    expected_clean: int,
    expected_component: int,
) -> None:
    decoder = _RecordingDecoder()
    optimizer = _CountingAdam(decoder.parameters(), lr=1.0e-3)

    logs = outcome_complete_train_step(
        decoder,
        CURELiteLoss(),
        OutcomeCompleteTransitionLoss(LossConfig()),
        optimizer,
        _factual(),
        _outcome(kinds),
    )

    assert logs["outcome/clean_pairs"] == expected_clean
    assert logs["outcome/component_null_pairs"] == expected_component
    assert len(decoder.forward_inputs) == 3
    assert optimizer.step_calls == 1


def test_duplicate_source_is_rejected_with_zero_training_side_effects() -> None:
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
    decoder = _RecordingDecoder().eval()
    optimizer = _CountingAdam(decoder.parameters(), lr=1.0e-3)
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.625)
    parameters_before = [
        parameter.detach().clone() for parameter in decoder.parameters()
    ]
    gradients_before = [
        parameter.grad.detach().clone() for parameter in decoder.parameters()
    ]

    with pytest.raises(ValueError, match="distinct source samples"):
        outcome_complete_train_step(
            decoder,
            CURELiteLoss(),
            OutcomeCompleteTransitionLoss(LossConfig()),
            optimizer,
            _factual(),
            malformed,
        )

    _assert_no_side_effects(
        decoder,
        optimizer,
        parameters_before=parameters_before,
        gradients_before=gradients_before,
    )


def test_duplicate_pair_id_is_rejected_with_zero_training_side_effects() -> None:
    outcome = _outcome(("clean_positive", "component_null"))
    duplicate_ids = replace(
        outcome.pair_batch,
        pair_ids=(
            outcome.pair_batch.pair_ids[0],
            outcome.pair_batch.pair_ids[0],
        ),
    )
    malformed = OutcomePairBatch(
        pair_batch=duplicate_ids,
        completion_plus=outcome.completion_plus,
        completion_minus=outcome.completion_minus,
        gt_union=outcome.gt_union,
        intervention_footprint=outcome.intervention_footprint,
    )
    decoder = _RecordingDecoder().eval()
    optimizer = _CountingAdam(decoder.parameters(), lr=1.0e-3)
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.6875)
    parameters_before = [
        parameter.detach().clone() for parameter in decoder.parameters()
    ]
    gradients_before = [
        parameter.grad.detach().clone() for parameter in decoder.parameters()
    ]

    with pytest.raises(ValueError, match="pair_ids must be unique"):
        outcome_complete_train_step(
            decoder,
            CURELiteLoss(),
            OutcomeCompleteTransitionLoss(LossConfig()),
            optimizer,
            _factual(),
            malformed,
        )

    _assert_no_side_effects(
        decoder,
        optimizer,
        parameters_before=parameters_before,
        gradients_before=gradients_before,
    )


def test_tampered_footprint_is_rejected_with_zero_training_side_effects() -> None:
    outcome = _outcome(("clean_positive", "component_null"))
    outcome.intervention_footprint[0, 0, 0, 0] = (
        ~outcome.intervention_footprint[0, 0, 0, 0]
    )
    decoder = _RecordingDecoder().eval()
    optimizer = _CountingAdam(decoder.parameters(), lr=1.0e-3)
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.75)
    parameters_before = [
        parameter.detach().clone() for parameter in decoder.parameters()
    ]
    gradients_before = [
        parameter.grad.detach().clone() for parameter in decoder.parameters()
    ]

    with pytest.raises(ValueError, match="nearest_lift"):
        outcome_complete_train_step(
            decoder,
            CURELiteLoss(),
            OutcomeCompleteTransitionLoss(LossConfig()),
            optimizer,
            _factual(),
            outcome,
        )

    _assert_no_side_effects(
        decoder,
        optimizer,
        parameters_before=parameters_before,
        gradients_before=gradients_before,
    )


def test_empty_plus_anchor_domain_is_rejected_before_forward() -> None:
    outcome = _outcome(("clean_positive", "component_null"))
    gt_union = outcome.gt_union.clone()
    # The clean helper has empty R+.  Covering its complete valid domain with
    # GT leaves neither an anchor target nor writable anchor background.
    gt_union[0] = outcome.pair_batch.image_valid_mask[0]
    malformed = OutcomePairBatch(
        pair_batch=outcome.pair_batch,
        completion_plus=outcome.completion_plus,
        completion_minus=outcome.completion_minus,
        gt_union=gt_union,
        intervention_footprint=outcome.intervention_footprint,
    )
    decoder = _RecordingDecoder().eval()
    optimizer = _CountingAdam(decoder.parameters(), lr=1.0e-3)
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.8125)
    parameters_before = [
        parameter.detach().clone() for parameter in decoder.parameters()
    ]
    gradients_before = [
        parameter.grad.detach().clone() for parameter in decoder.parameters()
    ]

    with pytest.raises(ValueError, match="plus-anchor supervision"):
        outcome_complete_train_step(
            decoder,
            CURELiteLoss(),
            OutcomeCompleteTransitionLoss(LossConfig()),
            optimizer,
            _factual(),
            malformed,
        )

    _assert_no_side_effects(
        decoder,
        optimizer,
        parameters_before=parameters_before,
        gradients_before=gradients_before,
    )


def test_bypassed_identity_null_is_rejected_before_forward() -> None:
    first, _ = _pair(
        kind="identity_null",
        sample_id="identity-source-0",
        component=(1, 1),
    )
    second, _ = _pair(
        kind="identity_null",
        sample_id="identity-source-1",
        component=(6, 6),
    )
    pair_batch = stack_pair_examples((first, second), device="cpu")
    bypassed = object.__new__(OutcomePairBatch)
    object.__setattr__(bypassed, "pair_batch", pair_batch)
    object.__setattr__(
        bypassed,
        "completion_plus",
        torch.stack((first.completion_plus, second.completion_plus)),
    )
    object.__setattr__(
        bypassed,
        "completion_minus",
        torch.stack((first.completion_minus, second.completion_minus)),
    )
    object.__setattr__(
        bypassed,
        "gt_union",
        torch.stack((first.completion_plus, second.completion_plus)),
    )
    object.__setattr__(
        bypassed,
        "intervention_footprint",
        torch.zeros_like(pair_batch.image_valid_mask),
    )
    decoder = _RecordingDecoder().eval()
    optimizer = _CountingAdam(decoder.parameters(), lr=1.0e-3)
    parameters_before = [
        parameter.detach().clone() for parameter in decoder.parameters()
    ]
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.875)
    gradients_before = [
        parameter.grad.detach().clone() for parameter in decoder.parameters()
    ]

    with pytest.raises(ValueError, match="clean_positive/component_null"):
        outcome_complete_train_step(
            decoder,
            CURELiteLoss(),
            OutcomeCompleteTransitionLoss(LossConfig()),
            optimizer,
            _factual(),
            bypassed,
        )

    _assert_no_side_effects(
        decoder,
        optimizer,
        parameters_before=parameters_before,
        gradients_before=gradients_before,
    )
