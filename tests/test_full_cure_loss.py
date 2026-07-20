from __future__ import annotations

import torch

from cure_lite.cure import CURELossConfig, CUREUncensoringLoss
from cure_lite.cure.types import ResidualSetSupervision


def _supervision(object_masks: torch.Tensor) -> ResidualSetSupervision:
    height, width = object_masks.shape[-2:]
    occupancy = torch.zeros(1, height, width, dtype=torch.bool)
    editable = torch.ones_like(occupancy)
    target = (
        object_masks.any(dim=0, keepdim=True)
        if object_masks.shape[0]
        else torch.zeros_like(occupancy)
    )
    background = editable & ~target
    return ResidualSetSupervision(
        occupancy=occupancy,
        editable_mask=editable,
        target=target.to(torch.float32),
        background_mask=background,
        object_masks=object_masks,
        positive_gt_ids=tuple(range(1, object_masks.shape[0] + 1)),
        uneditable_gt_ids=(),
        ignored_gt_ids=(),
        branch="factual",
    )


def test_object_loss_gives_small_and_large_targets_equal_weight() -> None:
    masks = torch.zeros(2, 6, 6, dtype=torch.bool)
    masks[0, 0, 0] = True
    masks[1, 2:5, 2:5] = True
    supervision = _supervision(masks)
    criterion = CUREUncensoringLoss(
        CURELossConfig(background_bce_weight=0.0)
    )
    result = criterion(torch.zeros(1, 1, 6, 6), (supervision,))
    torch.testing.assert_close(result["object"], torch.tensor(1.0 / 3.0))


def test_background_bce_penalizes_high_false_positive_logits() -> None:
    supervision = _supervision(torch.zeros(0, 3, 3, dtype=torch.bool))
    criterion = CUREUncensoringLoss(CURELossConfig(background_bce_weight=1.0))
    low = criterion(torch.full((1, 1, 3, 3), -2.0), (supervision,))
    high = criterion(torch.full((1, 1, 3, 3), 2.0), (supervision,))
    assert high["bce"] > low["bce"]


def test_no_miss_background_state_has_finite_silence_gradient() -> None:
    supervision = _supervision(torch.zeros(0, 3, 3, dtype=torch.bool))
    criterion = CUREUncensoringLoss(CURELossConfig(background_bce_weight=1.0))
    logits = torch.full((1, 1, 3, 3), -1000.0)
    logits[0, 0, 1, 1] = 1000.0
    logits.requires_grad_()
    result = criterion(logits, (supervision,))
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_empty_background_and_perfect_target_have_finite_loss() -> None:
    masks = torch.ones(1, 2, 2, dtype=torch.bool)
    supervision = _supervision(masks)
    criterion = CUREUncensoringLoss(CURELossConfig(background_bce_weight=1.0))
    result = criterion(torch.full((1, 1, 2, 2), 20.0), (supervision,))
    assert torch.isfinite(result["total"])
    assert result["bce"] < 1e-6


def test_counterfactual_loss_has_no_host_background_gradient() -> None:
    target = torch.zeros(1, 5, 5, dtype=torch.bool)
    target[0, 2, 2] = True
    supervision = ResidualSetSupervision(
        occupancy=torch.zeros_like(target),
        editable_mask=torch.ones_like(target),
        target=target.to(torch.float32),
        background_mask=torch.zeros_like(target),
        object_masks=target.clone(),
        positive_gt_ids=(1,),
        uneditable_gt_ids=(),
        ignored_gt_ids=(2,),
        branch="counterfactual",
    )
    logits = torch.zeros(1, 1, 5, 5, requires_grad=True)
    result = CUREUncensoringLoss(CURELossConfig())(logits, (supervision,))
    result["total"].backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 2, 2] != 0.0
    outside = torch.ones_like(target)
    outside[0, 2, 2] = False
    assert torch.count_nonzero(logits.grad[0][outside]) == 0
