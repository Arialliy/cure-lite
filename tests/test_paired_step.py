from __future__ import annotations

import hashlib

import pytest
import torch

from cure_lite.decoder import CURELiteDecoder
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_losses import PairedDifferenceLoss
from cure_lite.paired_types import PairBatch
from cure_lite.train.paired_step import (
    DECODER_STATES_PER_UPDATE,
    FACTUAL_ANCHOR_BATCH_SIZE,
    PAIRED_BATCH_SIZE,
    diagnose_null_pairs,
    paired_endpoint_logits,
    paired_train_step,
)
from cure_lite.train.step import BranchBatch


def _pair_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _CountingDecoder(CURELiteDecoder):
    def __init__(self, feature_channels: int) -> None:
        super().__init__(feature_channels=feature_channels)
        self.forward_calls = 0

    def forward(
        self,
        feature: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(feature, occupancy)


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def _positive_pair_batch(*, batch_size: int = 2) -> PairBatch:
    torch.manual_seed(17)
    feature = torch.randn(batch_size, 4, 4, 4)
    plus = torch.zeros(batch_size, 1, 8, 8, dtype=torch.bool)
    minus = torch.zeros_like(plus)
    increment = torch.zeros(batch_size, 1, 8, 8)
    locations = ((1, 1), (6, 6), (1, 6), (6, 1))
    for index in range(batch_size):
        row, column = locations[index]
        plus[index, 0, row, column] = True
        increment[index, 0, row, column] = 1.0
    return PairBatch(
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        label_increment=increment,
        image_valid_mask=torch.ones_like(plus),
        pair_ids=tuple(_pair_id(f"positive-{index}") for index in range(batch_size)),
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
        group_ids=tuple(f"group-{index}" for index in range(batch_size)),
        pair_kinds=("clean_positive",) * batch_size,
        projection_visible=(True,) * batch_size,
    )


def _factual_anchor_batches() -> dict[str, BranchBatch]:
    torch.manual_seed(23)
    feature_miss = torch.randn(4, 4, 4, 4)
    feature_no_miss = torch.randn(4, 4, 4, 4)
    occupancy = torch.zeros(4, 1, 8, 8, dtype=torch.bool)
    miss_target = torch.zeros(4, 1, 8, 8)
    miss_target[:, 0, 2, 2] = 1.0
    valid = torch.ones_like(occupancy)
    return {
        "factual_miss": BranchBatch(
            feature=feature_miss,
            occupancy=occupancy,
            target=miss_target,
            valid_mask=valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=feature_no_miss,
            occupancy=occupancy.clone(),
            target=torch.zeros_like(miss_target),
            valid_mask=valid.clone(),
        ),
    }


def test_one_2b_forward_matches_two_endpoint_forwards() -> None:
    decoder = _CountingDecoder(feature_channels=4).eval()
    batch = _positive_pair_batch()

    logits_plus, logits_minus = paired_endpoint_logits(decoder, batch)

    assert decoder.forward_calls == 1
    separate_plus = CURELiteDecoder.forward(
        decoder,
        batch.feature,
        batch.occupancy_plus,
    )
    separate_minus = CURELiteDecoder.forward(
        decoder,
        batch.feature,
        batch.occupancy_minus,
    )
    torch.testing.assert_close(logits_plus, separate_plus, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(logits_minus, separate_minus, rtol=1e-6, atol=1e-7)


def test_pair_forward_detaches_the_single_frozen_feature_copy() -> None:
    batch = _positive_pair_batch(batch_size=1)
    feature = batch.feature.clone().requires_grad_(True)
    batch = PairBatch(
        feature=feature,
        occupancy_plus=batch.occupancy_plus,
        occupancy_minus=batch.occupancy_minus,
        label_increment=batch.label_increment,
        image_valid_mask=batch.image_valid_mask,
        pair_ids=batch.pair_ids,
        sample_ids=batch.sample_ids,
        group_ids=batch.group_ids,
        pair_kinds=batch.pair_kinds,
        projection_visible=batch.projection_visible,
    )
    decoder = _CountingDecoder(feature_channels=4)

    logits_plus, logits_minus = paired_endpoint_logits(decoder, batch)
    PairedDifferenceLoss()(
        logits_plus,
        logits_minus,
        batch.label_increment,
        batch.image_valid_mask,
    )["total"].backward()

    assert feature.grad is None
    assert all(parameter.grad is not None for parameter in decoder.parameters())


def test_paired_step_runs_three_terms_one_backward_and_one_optimizer_step() -> None:
    decoder = _CountingDecoder(feature_channels=4)
    optimizer = _CountingSGD(decoder.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in decoder.parameters()]

    logs = paired_train_step(
        decoder,
        CURELiteLoss(),
        PairedDifferenceLoss(),
        optimizer,
        _factual_anchor_batches(),
        _positive_pair_batch(),
    )

    assert decoder.forward_calls == 3
    assert optimizer.step_calls == 1
    assert logs["optimizer_steps"] == 1
    assert logs["factual_miss/states"] == 4
    assert logs["factual_no_miss/states"] == 4
    assert logs["paired/pairs"] == 2
    assert logs["paired/endpoints"] == 4
    assert (
        logs["factual_miss/states"]
        + logs["factual_no_miss/states"]
        + logs["paired/endpoints"]
        == DECODER_STATES_PER_UPDATE
        == 12
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


def test_training_schedule_rejects_non_frozen_batch_sizes() -> None:
    assert FACTUAL_ANCHOR_BATCH_SIZE == 4
    assert PAIRED_BATCH_SIZE == 2
    decoder = _CountingDecoder(feature_channels=4)
    optimizer = _CountingSGD(decoder.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="exactly 2 clean pairs"):
        paired_train_step(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            _factual_anchor_batches(),
            _positive_pair_batch(batch_size=1),
        )

    same_source = _positive_pair_batch()
    same_source = PairBatch(
        feature=same_source.feature,
        occupancy_plus=same_source.occupancy_plus,
        occupancy_minus=same_source.occupancy_minus,
        label_increment=same_source.label_increment,
        image_valid_mask=same_source.image_valid_mask,
        pair_ids=same_source.pair_ids,
        sample_ids=(same_source.sample_ids[0], same_source.sample_ids[0]),
        group_ids=same_source.group_ids,
        pair_kinds=same_source.pair_kinds,
        projection_visible=same_source.projection_visible,
    )
    with pytest.raises(ValueError, match="distinct source samples"):
        paired_train_step(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            _factual_anchor_batches(),
            same_source,
        )

    factual = _factual_anchor_batches()
    miss = factual["factual_miss"]
    factual["factual_miss"] = BranchBatch(
        feature=miss.feature[:1],
        occupancy=miss.occupancy[:1],
        target=miss.target[:1],
        valid_mask=miss.valid_mask[:1],
    )
    with pytest.raises(ValueError, match="exactly 4 states"):
        paired_train_step(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            factual,
            _positive_pair_batch(),
        )

    assert decoder.forward_calls == 0
    assert optimizer.step_calls == 0


def test_invalid_second_anchor_has_zero_training_side_effects() -> None:
    decoder = _CountingDecoder(feature_channels=4).eval()
    optimizer = _CountingSGD(decoder.parameters(), lr=1e-3)
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    gradients_before = [
        parameter.grad.detach().clone() for parameter in decoder.parameters()
    ]
    factual = _factual_anchor_batches()
    no_miss = factual["factual_no_miss"]
    invalid_target = no_miss.target.clone()
    invalid_target[:, 0, 1, 1] = 1.0
    factual["factual_no_miss"] = BranchBatch(
        feature=no_miss.feature,
        occupancy=no_miss.occupancy,
        target=invalid_target,
        valid_mask=no_miss.valid_mask,
    )

    with pytest.raises(ValueError, match="empty target"):
        paired_train_step(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            factual,
            _positive_pair_batch(),
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


def test_optimizer_path_rejects_null_pairs_before_mutating_gradients() -> None:
    clean = _positive_pair_batch(batch_size=1)
    null = PairBatch(
        feature=clean.feature,
        occupancy_plus=clean.occupancy_plus,
        occupancy_minus=clean.occupancy_plus.clone(),
        label_increment=torch.zeros_like(clean.label_increment),
        image_valid_mask=clean.image_valid_mask,
        pair_ids=(_pair_id("identity-null"),),
        sample_ids=clean.sample_ids,
        group_ids=clean.group_ids,
        pair_kinds=("identity_null",),
        projection_visible=(False,),
    )
    decoder = _CountingDecoder(feature_channels=4)
    optimizer = _CountingSGD(decoder.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="only clean_positive"):
        paired_train_step(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            _factual_anchor_batches(),
            null,
        )

    assert decoder.forward_calls == 0
    assert optimizer.step_calls == 0
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_identity_null_is_read_only_and_has_exact_zero_difference() -> None:
    clean = _positive_pair_batch(batch_size=1)
    identity = PairBatch(
        feature=clean.feature,
        occupancy_plus=clean.occupancy_plus,
        occupancy_minus=clean.occupancy_plus.clone(),
        label_increment=torch.zeros_like(clean.label_increment),
        image_valid_mask=clean.image_valid_mask,
        pair_ids=(_pair_id("identity-diagnostic"),),
        sample_ids=clean.sample_ids,
        group_ids=clean.group_ids,
        pair_kinds=("identity_null",),
        projection_visible=(False,),
    )
    decoder = _CountingDecoder(feature_channels=4)
    before = [parameter.detach().clone() for parameter in decoder.parameters()]
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    gradients_before = [
        parameter.grad.detach().clone() for parameter in decoder.parameters()
    ]
    assert decoder.training

    result = diagnose_null_pairs(decoder, identity)

    assert decoder.forward_calls == 1
    assert decoder.training
    assert int(result["pair_count"]) == 1
    assert torch.equal(
        result["per_pair_mean_abs_delta"],
        torch.zeros(1),
    )
    assert torch.equal(
        result["per_pair_max_abs_delta"],
        torch.zeros(1),
    )
    assert torch.equal(result["per_pair_rms_delta"], torch.zeros(1))
    assert all(
        torch.equal(old, parameter.grad)
        for old, parameter in zip(
            gradients_before,
            decoder.parameters(),
            strict=True,
        )
    )
    assert all(
        torch.equal(old, parameter.detach())
        for old, parameter in zip(before, decoder.parameters(), strict=True)
    )


def test_component_null_diagnostic_is_finite_but_never_a_training_loss() -> None:
    clean = _positive_pair_batch(batch_size=1)
    component_null = PairBatch(
        feature=clean.feature,
        occupancy_plus=clean.occupancy_plus,
        occupancy_minus=clean.occupancy_minus,
        label_increment=torch.zeros_like(clean.label_increment),
        image_valid_mask=clean.image_valid_mask,
        pair_ids=(_pair_id("component-diagnostic"),),
        sample_ids=clean.sample_ids,
        group_ids=clean.group_ids,
        pair_kinds=("component_null",),
        projection_visible=(True,),
    )
    decoder = _CountingDecoder(feature_channels=4)

    result = diagnose_null_pairs(decoder, component_null)

    assert set(result) == {
        "pair_count",
        "per_pair_mean_abs_delta",
        "per_pair_max_abs_delta",
        "per_pair_rms_delta",
    }
    assert all(torch.isfinite(value).all() for value in result.values())
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_projected_invisible_edit_is_rejected_before_decoder_forward() -> None:
    clean = _positive_pair_batch(batch_size=1)
    plus = torch.zeros_like(clean.occupancy_plus)
    plus[0, 0, 0, 0] = True
    plus[0, 0, 0, 1] = True
    minus = torch.zeros_like(plus)
    minus[0, 0, 0, 0] = True
    increment = torch.zeros_like(clean.label_increment)
    increment[0, 0, 0, 1] = 1.0
    invisible = PairBatch(
        feature=clean.feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        label_increment=increment,
        image_valid_mask=clean.image_valid_mask,
        pair_ids=clean.pair_ids,
        sample_ids=clean.sample_ids,
        group_ids=clean.group_ids,
        pair_kinds=clean.pair_kinds,
        projection_visible=(False,),
    )
    decoder = _CountingDecoder(feature_channels=4)

    with pytest.raises(ValueError, match="visible projection"):
        paired_endpoint_logits(decoder, invisible)
    assert decoder.forward_calls == 0


def test_increment_must_be_writable_under_minus_endpoint() -> None:
    clean = _positive_pair_batch(batch_size=1)
    minus = clean.occupancy_minus.clone()
    minus[0, 0, 3, 3] = True
    plus = clean.occupancy_plus.clone()
    plus[0, 0, 3, 3] = True
    increment = clean.label_increment.clone()
    increment[0, 0, 3, 3] = 1.0
    invalid = PairBatch(
        feature=clean.feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        label_increment=increment,
        image_valid_mask=clean.image_valid_mask,
        pair_ids=clean.pair_ids,
        sample_ids=clean.sample_ids,
        group_ids=clean.group_ids,
        pair_kinds=clean.pair_kinds,
        projection_visible=clean.projection_visible,
    )

    with pytest.raises(ValueError, match="writable under occupancy_minus"):
        paired_endpoint_logits(_CountingDecoder(feature_channels=4), invalid)


def test_duplicate_pair_identity_is_rejected() -> None:
    batch = _positive_pair_batch()
    duplicated = PairBatch(
        feature=batch.feature,
        occupancy_plus=batch.occupancy_plus,
        occupancy_minus=batch.occupancy_minus,
        label_increment=batch.label_increment,
        image_valid_mask=batch.image_valid_mask,
        pair_ids=(batch.pair_ids[0], batch.pair_ids[0]),
        sample_ids=batch.sample_ids,
        group_ids=batch.group_ids,
        pair_kinds=batch.pair_kinds,
        projection_visible=batch.projection_visible,
    )

    with pytest.raises(ValueError, match="unique"):
        paired_endpoint_logits(_CountingDecoder(feature_channels=4), duplicated)
