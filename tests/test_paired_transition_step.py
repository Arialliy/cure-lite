from __future__ import annotations

import hashlib

import pytest
import torch

from cure_lite.decoder import CURELiteDecoder
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_transition_losses import AnchoredTransitionLoss
from cure_lite.paired_transition_types import AnchoredPairBatch
from cure_lite.paired_types import PairBatch
from cure_lite.train.paired_transition_step import (
    anchored_transition_train_step,
)
from cure_lite.train.step import BranchBatch


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _CountingDecoder(CURELiteDecoder):
    def __init__(self) -> None:
        super().__init__(feature_channels=2)
        self.forward_calls = 0

    def forward(
        self,
        feature: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(feature, occupancy)


class _CountingAdam(torch.optim.Adam):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def _anchored_batch(*, batch_size: int = 2) -> AnchoredPairBatch:
    torch.manual_seed(901)
    feature = torch.randn(batch_size, 2, 4, 4)
    plus = torch.zeros(batch_size, 1, 8, 8, dtype=torch.bool)
    minus = torch.zeros_like(plus)
    increment = torch.zeros(batch_size, 1, 8, 8)
    completion_plus = torch.zeros_like(plus)
    gt_union = torch.zeros_like(plus)
    locations = ((1, 1), (6, 6))
    existing = ((6, 1), (1, 6))
    for index in range(batch_size):
        row, column = locations[index]
        plus[index, 0, row, column] = True
        increment[index, 0, row, column] = 1.0
        gt_union[index, 0, row, column] = True
        old_row, old_column = existing[index]
        completion_plus[index, 0, old_row, old_column] = True
        gt_union[index, 0, old_row, old_column] = True
    pair = PairBatch(
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        label_increment=increment,
        image_valid_mask=torch.ones_like(plus),
        pair_ids=tuple(_sha(f"transition-{index}") for index in range(batch_size)),
        sample_ids=tuple(f"source-{index}" for index in range(batch_size)),
        group_ids=tuple(f"group-{index}" for index in range(batch_size)),
        pair_kinds=("clean_positive",) * batch_size,
        projection_visible=(True,) * batch_size,
    )
    return AnchoredPairBatch(
        pair_batch=pair,
        completion_plus=completion_plus,
        gt_union=gt_union,
    )


def _factual() -> dict[str, BranchBatch]:
    torch.manual_seed(902)
    occupancy = torch.zeros(4, 1, 8, 8, dtype=torch.bool)
    valid = torch.ones_like(occupancy)
    miss = torch.zeros(4, 1, 8, 8)
    miss[:, 0, 2, 2] = 1.0
    return {
        "factual_miss": BranchBatch(
            feature=torch.randn(4, 2, 4, 4),
            occupancy=occupancy,
            target=miss,
            valid_mask=valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=torch.randn(4, 2, 4, 4),
            occupancy=occupancy.clone(),
            target=torch.zeros_like(miss),
            valid_mask=valid.clone(),
        ),
    }


def test_anchored_step_keeps_the_frozen_forward_and_state_budget() -> None:
    decoder = _CountingDecoder()
    optimizer = _CountingAdam(decoder.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in decoder.parameters()]

    logs = anchored_transition_train_step(
        decoder,
        CURELiteLoss(),
        AnchoredTransitionLoss(),
        optimizer,
        _factual(),
        _anchored_batch(),
    )

    assert decoder.forward_calls == 3
    assert optimizer.step_calls == 1
    assert logs["optimizer_steps"] == 1
    assert logs["decoder_forward_calls_per_update"] == 3
    assert logs["decoder_states_per_update"] == 12
    assert logs["factual_miss/states"] == 4
    assert logs["factual_no_miss/states"] == 4
    assert logs["paired/pairs"] == 2
    assert logs["paired/endpoints"] == 4
    assert logs["paired/loss"] == pytest.approx(
        0.5
        * (
            logs["paired/plus_anchor_loss"]
            + logs["paired/transition_loss"]
        ),
        abs=2e-7,
    )
    assert logs["total"] == pytest.approx(
        logs["factual_miss/loss"]
        + logs["factual_no_miss/loss"]
        + logs["paired/loss"],
        abs=3e-7,
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


def test_invalid_anchor_has_no_training_side_effects() -> None:
    anchored = _anchored_batch()
    invalid_completion = anchored.completion_plus.clone()
    invalid_completion[0, 0, 0, 0] = True
    invalid = object.__new__(AnchoredPairBatch)
    object.__setattr__(invalid, "pair_batch", anchored.pair_batch)
    object.__setattr__(invalid, "completion_plus", invalid_completion)
    object.__setattr__(invalid, "gt_union", anchored.gt_union)
    decoder = _CountingDecoder().eval()
    optimizer = _CountingAdam(decoder.parameters(), lr=1e-3)
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    gradients_before = [
        parameter.grad.detach().clone() for parameter in decoder.parameters()
    ]

    with pytest.raises(ValueError, match="only gt_union pixels"):
        anchored_transition_train_step(
            decoder,
            CURELiteLoss(),
            AnchoredTransitionLoss(),
            optimizer,
            _factual(),
            invalid,
        )

    assert decoder.training is False
    assert decoder.forward_calls == 0
    assert optimizer.step_calls == 0
    assert all(
        torch.equal(before, parameter.grad)
        for before, parameter in zip(
            gradients_before,
            decoder.parameters(),
            strict=True,
        )
    )


def test_anchored_step_rejects_non_frozen_pair_batch_size() -> None:
    decoder = _CountingDecoder()
    optimizer = _CountingAdam(decoder.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="exactly 2 clean pairs"):
        anchored_transition_train_step(
            decoder,
            CURELiteLoss(),
            AnchoredTransitionLoss(),
            optimizer,
            _factual(),
            _anchored_batch(batch_size=1),
        )

    assert decoder.forward_calls == 0
    assert optimizer.step_calls == 0


def test_broadcastable_anchor_shape_is_rejected_before_training_side_effects() -> None:
    anchored = _anchored_batch()
    malformed = object.__new__(AnchoredPairBatch)
    object.__setattr__(malformed, "pair_batch", anchored.pair_batch)
    object.__setattr__(
        malformed,
        "completion_plus",
        anchored.completion_plus[:1].clone(),
    )
    object.__setattr__(malformed, "gt_union", anchored.gt_union)
    decoder = _CountingDecoder().eval()
    optimizer = _CountingAdam(decoder.parameters(), lr=1e-3)
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.375)
    gradients_before = [
        parameter.grad.detach().clone() for parameter in decoder.parameters()
    ]
    parameters_before = [
        parameter.detach().clone() for parameter in decoder.parameters()
    ]

    with pytest.raises(ValueError, match="evaluation shape"):
        anchored_transition_train_step(
            decoder,
            CURELiteLoss(),
            AnchoredTransitionLoss(),
            optimizer,
            _factual(),
            malformed,
        )

    assert decoder.training is False
    assert decoder.forward_calls == 0
    assert optimizer.step_calls == 0
    assert optimizer.state == {}
    assert all(
        torch.equal(before, parameter.grad)
        for before, parameter in zip(
            gradients_before,
            decoder.parameters(),
            strict=True,
        )
    )
    assert all(
        torch.equal(before, parameter.detach())
        for before, parameter in zip(
            parameters_before,
            decoder.parameters(),
            strict=True,
        )
    )


def test_post_construction_anchor_resize_is_rejected_before_forward() -> None:
    anchored = _anchored_batch()
    anchored.completion_plus.resize_(1, *anchored.completion_plus.shape[1:])
    decoder = _CountingDecoder().eval()
    optimizer = _CountingAdam(decoder.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="evaluation shape"):
        anchored_transition_train_step(
            decoder,
            CURELiteLoss(),
            AnchoredTransitionLoss(),
            optimizer,
            _factual(),
            anchored,
        )

    assert decoder.training is False
    assert decoder.forward_calls == 0
    assert optimizer.step_calls == 0
    assert optimizer.state == {}
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_bypassed_null_pair_is_rejected_before_forward() -> None:
    anchored = _anchored_batch()
    clean = anchored.pair_batch
    identity_pair = PairBatch(
        feature=clean.feature,
        occupancy_plus=clean.occupancy_plus,
        occupancy_minus=clean.occupancy_plus.clone(),
        label_increment=torch.zeros_like(clean.label_increment),
        image_valid_mask=clean.image_valid_mask,
        pair_ids=clean.pair_ids,
        sample_ids=clean.sample_ids,
        group_ids=clean.group_ids,
        pair_kinds=("identity_null", "identity_null"),
        projection_visible=(False, False),
    )
    bypassed = object.__new__(AnchoredPairBatch)
    object.__setattr__(bypassed, "pair_batch", identity_pair)
    object.__setattr__(
        bypassed,
        "completion_plus",
        torch.zeros_like(identity_pair.image_valid_mask),
    )
    object.__setattr__(
        bypassed,
        "gt_union",
        torch.zeros_like(identity_pair.image_valid_mask),
    )
    decoder = _CountingDecoder().eval()
    optimizer = _CountingAdam(decoder.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="only clean_positive"):
        anchored_transition_train_step(
            decoder,
            CURELiteLoss(),
            AnchoredTransitionLoss(),
            optimizer,
            _factual(),
            bypassed,
        )

    assert decoder.training is False
    assert decoder.forward_calls == 0
    assert optimizer.step_calls == 0
    assert optimizer.state == {}
    assert all(parameter.grad is None for parameter in decoder.parameters())
