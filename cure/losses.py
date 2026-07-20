"""Object-balanced residual-set loss for full CURE."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import CURELossConfig
from .types import ResidualSetSupervision


class CUREUncensoringLoss(nn.Module):
    """Fixed equal-object recovery and dense background-silence objective."""

    def __init__(self, config: CURELossConfig = CURELossConfig()) -> None:
        super().__init__()
        if not isinstance(config, CURELossConfig):
            raise TypeError("config must be CURELossConfig")
        self.config = config

    def _state_loss(
        self,
        logits: Tensor,
        supervision: ResidualSetSupervision,
    ) -> tuple[Tensor, Tensor, Tensor]:
        probability = torch.sigmoid(logits)
        zero = logits.sum() * 0.0
        object_terms: list[Tensor] = []
        for object_mask in supervision.object_masks:
            mask = object_mask.to(dtype=probability.dtype)
            predicted_mass = (probability[0] * mask).sum()
            object_terms.append(
                1.0
                - (2.0 * predicted_mass + self.config.epsilon)
                / (predicted_mass + mask.sum() + self.config.epsilon)
            )
        object_loss = torch.stack(object_terms).mean() if object_terms else zero

        target = supervision.target
        valid = supervision.background_mask | target.to(torch.bool)
        segmentation_bce = (
            F.binary_cross_entropy_with_logits(logits[valid], target[valid])
            if bool(torch.any(valid))
            else zero
        )

        total = (
            object_loss
            + self.config.background_bce_weight * segmentation_bce
        )
        return total, object_loss, segmentation_bce

    def forward(
        self,
        residual_logits: Tensor,
        supervisions: Sequence[ResidualSetSupervision],
    ) -> dict[str, Tensor]:
        if not isinstance(residual_logits, Tensor) or residual_logits.ndim != 4:
            raise ValueError("residual_logits must have shape [B,1,H,W]")
        if residual_logits.shape[1] != 1 or residual_logits.shape[0] < 1:
            raise ValueError("residual_logits must be a non-empty single-channel batch")
        if not residual_logits.is_floating_point() or not torch.isfinite(residual_logits).all():
            raise ValueError("residual_logits must be finite and floating point")
        items = tuple(supervisions)
        if len(items) != residual_logits.shape[0] or any(
            not isinstance(item, ResidualSetSupervision) for item in items
        ):
            raise ValueError("one ResidualSetSupervision is required per batch item")
        for item in items:
            if tuple(item.target.shape[-2:]) != tuple(residual_logits.shape[-2:]):
                raise ValueError("logit and supervision grids differ")
            if item.target.device != residual_logits.device:
                raise ValueError("logits and supervisions must share a device")

        states = [
            self._state_loss(residual_logits[index], item)
            for index, item in enumerate(items)
        ]
        totals = torch.stack([state[0] for state in states])
        objects = torch.stack([state[1] for state in states])
        bces = torch.stack([state[2] for state in states])
        total = totals.mean()
        return {
            "total": total,
            "loss": total,
            "object": objects.mean(),
            "bce": bces.mean(),
            "state_count": torch.tensor(len(items), device=residual_logits.device),
            "object_count": torch.tensor(
                sum(item.object_masks.shape[0] for item in items),
                device=residual_logits.device,
            ),
            "per_state_total": totals,
            "per_state_object": objects,
            "per_state_bce": bces,
        }


__all__ = ["CUREUncensoringLoss"]
