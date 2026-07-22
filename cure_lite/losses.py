"""Masked, per-state CURE-Lite v0.1 training loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import LossConfig


class CURELiteLoss(nn.Module):
    """Balanced logistic plus positive-only soft Dice.

    Every state is reduced independently. The returned scalar is the mean of
    state losses, so images with many valid pixels cannot dominate a branch.
    """

    def __init__(
        self,
        config: LossConfig | None = None,
        *,
        dice_weight: float | None = None,
        epsilon: float | None = None,
    ) -> None:
        super().__init__()
        if config is not None and (dice_weight is not None or epsilon is not None):
            raise ValueError("do not override fields of an explicit LossConfig")
        self.config = config or LossConfig(
            dice_weight=1.0 if dice_weight is None else dice_weight,
            epsilon=1e-6 if epsilon is None else epsilon,
        )

    @property
    def dice_weight(self) -> float:
        return self.config.dice_weight

    @property
    def epsilon(self) -> float:
        return self.config.epsilon

    def _state_loss(
        self,
        logits: Tensor,
        target: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        positive = valid_mask & target.to(torch.bool)
        negative = valid_mask & ~target.to(torch.bool)
        zero = logits.sum() * 0.0

        positive_loss = F.softplus(-logits[positive]).mean() if positive.any() else zero
        negative_loss = F.softplus(logits[negative]).mean() if negative.any() else zero
        if positive.any() and negative.any():
            balanced_bce = 0.5 * (positive_loss + negative_loss)
        elif positive.any():
            balanced_bce = positive_loss
        elif negative.any():
            balanced_bce = negative_loss
        else:
            balanced_bce = zero

        if positive.any():
            probability = torch.sigmoid(logits)
            valid = valid_mask.to(dtype=probability.dtype)
            binary_target = target.to(dtype=probability.dtype)
            numerator = 2.0 * (valid * binary_target * probability).sum() + self.epsilon
            denominator = (
                (valid * binary_target).sum()
                + (valid * probability).sum()
                + self.epsilon
            )
            dice = 1.0 - numerator / denominator
        else:
            dice = zero
        total = balanced_bce + self.dice_weight * dice
        return total, balanced_bce, dice, positive.sum(), negative.sum()

    def forward(self, logits: Tensor, target: Tensor, valid_mask: Tensor) -> dict[str, Tensor]:
        if not all(isinstance(value, Tensor) for value in (logits, target, valid_mask)):
            raise TypeError("logits, target, and valid_mask must be tensors")
        if logits.ndim == 3:
            logits, target, valid_mask = (
                value.unsqueeze(0) for value in (logits, target, valid_mask)
            )
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("loss tensors must have shape [B,1,H,W] or [1,H,W]")
        if logits.shape != target.shape or target.shape != valid_mask.shape:
            raise ValueError("logits, target, and valid_mask shapes must match")
        if logits.shape[0] < 1:
            raise ValueError("loss batch must contain at least one state")
        if not logits.is_floating_point():
            raise TypeError("logits must be floating point")
        if target.dtype != torch.float32:
            raise TypeError("target must be float32")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be bool")
        if not (logits.device == target.device == valid_mask.device):
            raise ValueError("loss tensors must share a device")
        if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
            raise ValueError("logits and target must be finite")
        if torch.any((target != 0.0) & (target != 1.0)):
            raise ValueError("target must be binary")
        if torch.any(target.to(torch.bool) & ~valid_mask):
            raise ValueError("positive target lies outside valid_mask")

        states = [
            self._state_loss(logits[index], target[index], valid_mask[index])
            for index in range(logits.shape[0])
        ]
        per_state_total = torch.stack([state[0] for state in states])
        per_state_bce = torch.stack([state[1] for state in states])
        per_state_dice = torch.stack([state[2] for state in states])
        total = per_state_total.mean()
        return {
            "total": total,
            "loss": total,
            "bce": per_state_bce.mean(),
            "dice": per_state_dice.mean(),
            "positive_pixels": torch.stack([state[3] for state in states]).sum(),
            "negative_pixels": torch.stack([state[4] for state in states]).sum(),
            "state_count": torch.tensor(logits.shape[0], device=logits.device),
            "per_state_total": per_state_total,
            "per_state_bce": per_state_bce,
            "per_state_dice": per_state_dice,
        }


__all__ = ["CURELiteLoss"]
