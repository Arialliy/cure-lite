from __future__ import annotations

import hashlib

import pytest
import torch

from cure_lite.decoder import CURELiteDecoder
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_control_inputs import capacity_active_dct_feature_like
from cure_lite.paired_losses import PairedDifferenceLoss
from cure_lite.paired_types import PairBatch
from cure_lite.train.paired_control_step import (
    CONTROL_KINDS,
    paired_control_train_step,
)
from cure_lite.train.paired_step import (
    DECODER_STATES_PER_UPDATE,
    paired_train_step,
)
from cure_lite.train.step import BranchBatch


def _pair_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _CountingDecoder(CURELiteDecoder):
    def __init__(self) -> None:
        super().__init__(feature_channels=3)
        self.forward_calls = 0
        self.forward_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        feature: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls += 1
        self.forward_inputs.append(
            (
                feature.detach().clone(),
                occupancy.detach().clone(),
            )
        )
        return super().forward(feature, occupancy)


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _RecordingAbsoluteLoss(CURELiteLoss):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.calls.append((target.detach().clone(), valid_mask.detach().clone()))
        return super().forward(logits, target, valid_mask)


class _RecordingPairedLoss(PairedDifferenceLoss):
    def __init__(self) -> None:
        super().__init__()
        self.labels: list[torch.Tensor] = []

    def forward(
        self,
        logits_plus: torch.Tensor,
        logits_minus: torch.Tensor,
        label_increment: torch.Tensor,
        image_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.labels.append(label_increment.detach().clone())
        return super().forward(
            logits_plus,
            logits_minus,
            label_increment,
            image_valid_mask,
        )


def _pair_batch() -> PairBatch:
    torch.manual_seed(701)
    feature = torch.randn(2, 3, 3, 3)
    plus = torch.zeros(2, 1, 6, 6, dtype=torch.bool)
    minus = torch.zeros_like(plus)
    increment = torch.zeros(2, 1, 6, 6)
    plus[0, 0, 1, 1] = True
    plus[1, 0, 4, 4] = True
    increment[0, 0, 1, 1] = 1.0
    increment[1, 0, 4, 4] = 1.0
    return PairBatch(
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        label_increment=increment,
        image_valid_mask=torch.ones_like(plus),
        pair_ids=(_pair_id("pair-a"), _pair_id("pair-b")),
        sample_ids=("sample-a", "sample-b"),
        group_ids=("group-a", "group-b"),
        pair_kinds=("clean_positive", "clean_positive"),
        projection_visible=(True, True),
    )


def _factual_batches() -> dict[str, BranchBatch]:
    torch.manual_seed(702)
    occupancy = torch.zeros(4, 1, 6, 6, dtype=torch.bool)
    valid = torch.ones_like(occupancy)
    miss_target = torch.zeros(4, 1, 6, 6)
    miss_target[:, 0, 2, 2] = 1.0
    return {
        "factual_miss": BranchBatch(
            feature=torch.randn(4, 3, 3, 3),
            occupancy=occupancy,
            target=miss_target,
            valid_mask=valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=torch.randn(4, 3, 3, 3),
            occupancy=occupancy.clone(),
            target=torch.zeros_like(miss_target),
            valid_mask=valid.clone(),
        ),
    }


def _geometry(
    pair: PairBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preexisting = torch.zeros_like(pair.occupancy_plus)
    preexisting[0, 0, 2, 3] = True
    preexisting[1, 0, 3, 2] = True
    completion_plus = preexisting
    completion_minus = preexisting | pair.label_increment.to(torch.bool)
    gt_union = completion_minus.clone()
    return gt_union, completion_plus, completion_minus


def _control_kwargs(
    kind: str,
    pair: PairBatch,
) -> dict[str, torch.Tensor]:
    gt_union, completion_plus, completion_minus = _geometry(pair)
    if kind == "independent_endpoint":
        return {
            "gt_union": gt_union,
            "completion_plus": completion_plus,
            "completion_minus": completion_minus,
        }
    if kind == "after_only":
        return {"gt_union": gt_union}
    if kind == "target_permutation":
        return {"permuted_label_increment": pair.label_increment.flip(0)}
    return {}


@pytest.mark.parametrize("control_kind", CONTROL_KINDS)
def test_every_control_keeps_the_frozen_budget_and_single_step(
    control_kind: str,
) -> None:
    pair = _pair_batch()
    decoder = _CountingDecoder()
    optimizer = _CountingSGD(decoder.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in decoder.parameters()]

    logs = paired_control_train_step(
        decoder,
        CURELiteLoss(),
        PairedDifferenceLoss(),
        optimizer,
        _factual_batches(),
        pair,
        control_kind=control_kind,
        **_control_kwargs(control_kind, pair),
    )

    assert logs["control_kind"] == control_kind
    assert decoder.forward_calls == 3
    assert [value[0].shape[0] for value in decoder.forward_inputs] == [4, 4, 4]
    assert logs["control/endpoint_forward_batches"] == 1
    assert logs["factual_miss/states"] == 4
    assert logs["factual_no_miss/states"] == 4
    assert logs["control/pairs"] == 2
    assert logs["control/endpoints"] == 4
    assert logs["decoder/states"] == DECODER_STATES_PER_UPDATE == 12
    assert logs["factual_anchor_batch_size"] == 4
    assert logs["paired_batch_size"] == 2
    assert optimizer.step_calls == logs["optimizer_steps"] == 1
    assert logs["total"] == pytest.approx(
        logs["factual_miss/loss"]
        + logs["factual_no_miss/loss"]
        + logs["control/loss"],
        abs=3e-7,
    )
    assert any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, decoder.parameters(), strict=True)
    )


@pytest.mark.parametrize(
    "control_kind",
    ("zero_feature", "coordinate_basis", "feature_only"),
)
def test_control_input_transforms_are_exact(control_kind: str) -> None:
    pair = _pair_batch()
    decoder = _CountingDecoder()
    optimizer = _CountingSGD(decoder.parameters(), lr=1e-3)

    logs = paired_control_train_step(
        decoder,
        CURELiteLoss(),
        PairedDifferenceLoss(),
        optimizer,
        _factual_batches(),
        pair,
        control_kind=control_kind,
    )
    pair_feature, pair_occupancy = decoder.forward_inputs[2]

    if control_kind == "zero_feature":
        assert torch.count_nonzero(pair_feature) == 0
        assert torch.equal(
            pair_occupancy,
            torch.cat((pair.occupancy_plus, pair.occupancy_minus), dim=0),
        )
    elif control_kind == "coordinate_basis":
        expected, basis = capacity_active_dct_feature_like(pair.feature)
        assert torch.equal(pair_feature, torch.cat((expected, expected), dim=0))
        assert logs["control/basis_fingerprint"] == basis.basis_fingerprint
        assert torch.equal(pair_feature[0], pair_feature[1])
    else:
        assert torch.equal(
            pair_feature,
            torch.cat((pair.feature, pair.feature), dim=0),
        )
        assert torch.count_nonzero(pair_occupancy) == 0


def test_independent_and_after_only_dispatch_exact_absolute_targets() -> None:
    pair = _pair_batch()
    gt_union, completion_plus, completion_minus = _geometry(pair)

    independent_absolute = _RecordingAbsoluteLoss()
    independent_paired = _RecordingPairedLoss()
    decoder = _CountingDecoder()
    paired_control_train_step(
        decoder,
        independent_absolute,
        independent_paired,
        _CountingSGD(decoder.parameters(), lr=1e-3),
        _factual_batches(),
        pair,
        control_kind="independent_endpoint",
        gt_union=gt_union,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
    )
    assert len(independent_absolute.calls) == 4
    assert independent_paired.labels == []
    assert torch.equal(
        independent_absolute.calls[2][0],
        completion_plus.to(torch.float32),
    )
    assert torch.equal(
        independent_absolute.calls[3][0],
        completion_minus.to(torch.float32),
    )

    after_absolute = _RecordingAbsoluteLoss()
    after_paired = _RecordingPairedLoss()
    decoder = _CountingDecoder()
    paired_control_train_step(
        decoder,
        after_absolute,
        after_paired,
        _CountingSGD(decoder.parameters(), lr=1e-3),
        _factual_batches(),
        pair,
        control_kind="after_only",
        gt_union=gt_union,
    )
    assert len(after_absolute.calls) == 3
    assert after_paired.labels == []
    assert torch.equal(after_absolute.calls[2][0], pair.label_increment)
    assert not torch.any(
        after_absolute.calls[2][1] & completion_plus
    )


def test_target_permutation_consumes_explicit_permuted_labels() -> None:
    pair = _pair_batch()
    permuted = pair.label_increment.flip(0)
    criterion = _RecordingPairedLoss()
    decoder = _CountingDecoder()

    paired_control_train_step(
        decoder,
        CURELiteLoss(),
        criterion,
        _CountingSGD(decoder.parameters(), lr=1e-3),
        _factual_batches(),
        pair,
        control_kind="target_permutation",
        permuted_label_increment=permuted,
    )

    assert len(criterion.labels) == 1
    assert torch.equal(criterion.labels[0], permuted)
    assert not torch.equal(criterion.labels[0], pair.label_increment)


@pytest.mark.parametrize("control_kind", ("plus_detach", "minus_detach"))
def test_stop_gradient_control_updates_differ_from_main_objective(
    control_kind: str,
) -> None:
    torch.manual_seed(703)
    main_decoder = _CountingDecoder()
    control_decoder = _CountingDecoder()
    control_decoder.load_state_dict(main_decoder.state_dict())
    pair = _pair_batch()
    factual = _factual_batches()

    paired_train_step(
        main_decoder,
        CURELiteLoss(),
        PairedDifferenceLoss(),
        _CountingSGD(main_decoder.parameters(), lr=1e-2),
        factual,
        pair,
    )
    paired_control_train_step(
        control_decoder,
        CURELiteLoss(),
        PairedDifferenceLoss(),
        _CountingSGD(control_decoder.parameters(), lr=1e-2),
        factual,
        pair,
        control_kind=control_kind,
    )

    assert any(
        not torch.equal(main, control)
        for main, control in zip(
            main_decoder.parameters(),
            control_decoder.parameters(),
            strict=True,
        )
    )


def test_invalid_control_preflight_has_zero_training_side_effects() -> None:
    pair = _pair_batch()
    gt_union, completion_plus, completion_minus = _geometry(pair)
    decoder = _CountingDecoder().eval()
    optimizer = _CountingSGD(decoder.parameters(), lr=1e-3)
    for parameter in decoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.375)
    gradients_before = [
        parameter.grad.detach().clone() for parameter in decoder.parameters()
    ]

    with pytest.raises(TypeError, match="gt_union"):
        paired_control_train_step(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            _factual_batches(),
            pair,
            control_kind="independent_endpoint",
            gt_union=gt_union.to(torch.float32),
            completion_plus=completion_plus,
            completion_minus=completion_minus,
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

