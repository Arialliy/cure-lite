"""Coupled pre-mask score-difference objective for CURE-Lite pairs.

This module is intentionally additive.  It does not replace
``CURELiteLoss``: the latter remains the zero-order factual anchor, while this
criterion consumes the two endpoints of one clean positive pair jointly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PairedDifferenceLoss(nn.Module):
    """Balanced finite-difference regression on raw residual scores.

    Given pre-hard-mask endpoint logits, the criterion first computes

    ``delta = sigmoid(logits_minus) - sigmoid(logits_plus)``.

    For every pair independently, ``label_increment`` is the positive response
    stratum and its complement inside ``image_valid_mask`` is the zero-response
    stratum.  Both strata must be non-empty.  The frozen per-pair objective is

    ``0.5 * mean_P(((delta - 1) / 2) ** 2)
       + 0.5 * mean_Z(delta ** 2)``.

    Neither endpoint is detached, and no occupancy hard mask or threshold is
    applied here.
    """

    @staticmethod
    def _validate(
        logits_plus: Tensor,
        logits_minus: Tensor,
        label_increment: Tensor,
        image_valid_mask: Tensor,
    ) -> None:
        values = (
            logits_plus,
            logits_minus,
            label_increment,
            image_valid_mask,
        )
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError(
                "logits_plus, logits_minus, label_increment, and "
                "image_valid_mask must be tensors"
            )
        if logits_plus.ndim != 4 or logits_plus.shape[1] != 1:
            raise ValueError("paired tensors must have shape [B,1,H,W]")
        if not (
            logits_plus.shape
            == logits_minus.shape
            == label_increment.shape
            == image_valid_mask.shape
        ):
            raise ValueError("paired tensors must have identical shapes")
        if logits_plus.shape[0] < 1:
            raise ValueError("paired batch must contain at least one pair")
        if not logits_plus.is_floating_point() or not logits_minus.is_floating_point():
            raise TypeError("endpoint logits must be floating point")
        if logits_plus.dtype != logits_minus.dtype:
            raise TypeError("endpoint logits must share a dtype")
        if label_increment.dtype != torch.float32:
            raise TypeError("label_increment must be float32")
        if image_valid_mask.dtype != torch.bool:
            raise TypeError("image_valid_mask must be bool")
        if not (
            logits_plus.device
            == logits_minus.device
            == label_increment.device
            == image_valid_mask.device
        ):
            raise ValueError("paired tensors must share a device")
        if not all(
            torch.isfinite(value).all()
            for value in (logits_plus, logits_minus, label_increment)
        ):
            raise ValueError("paired floating-point tensors must be finite")
        if torch.any((label_increment != 0.0) & (label_increment != 1.0)):
            raise ValueError("label_increment must be binary")

        positive = label_increment.to(dtype=torch.bool)
        if torch.any(positive & ~image_valid_mask):
            raise ValueError("label_increment lies outside image_valid_mask")
        positive_by_pair = positive.flatten(1).any(dim=1)
        zero_by_pair = (image_valid_mask & ~positive).flatten(1).any(dim=1)
        if not torch.all(positive_by_pair):
            raise ValueError(
                "every positive pair must contain a non-empty response stratum"
            )
        if not torch.all(zero_by_pair):
            raise ValueError(
                "every positive pair must contain a non-empty zero-response stratum"
            )

    def forward(
        self,
        logits_plus: Tensor,
        logits_minus: Tensor,
        label_increment: Tensor,
        image_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Return the equally pair- and stratum-balanced coupled objective."""

        self._validate(
            logits_plus,
            logits_minus,
            label_increment,
            image_valid_mask,
        )
        score_plus = torch.sigmoid(logits_plus)
        score_minus = torch.sigmoid(logits_minus)
        delta = score_minus - score_plus
        positive = label_increment.to(dtype=torch.bool)
        zero = image_valid_mask & ~positive

        states: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        for index in range(delta.shape[0]):
            positive_mse = (((delta[index][positive[index]] - 1.0) / 2.0) ** 2).mean()
            zero_mse = (delta[index][zero[index]] ** 2).mean()
            positive_term = 0.5 * positive_mse
            zero_term = 0.5 * zero_mse
            states.append(
                (
                    positive_term + zero_term,
                    positive_mse,
                    zero_mse,
                    positive[index].sum(),
                    zero[index].sum(),
                )
            )

        per_pair_total = torch.stack([state[0] for state in states])
        per_pair_positive_mse = torch.stack([state[1] for state in states])
        per_pair_zero_mse = torch.stack([state[2] for state in states])
        total = per_pair_total.mean()
        return {
            "total": total,
            "loss": total,
            "positive_stratum_mse": per_pair_positive_mse.mean(),
            "zero_stratum_mse": per_pair_zero_mse.mean(),
            "positive_response_pixels": torch.stack(
                [state[3] for state in states]
            ).sum(),
            "zero_response_pixels": torch.stack([state[4] for state in states]).sum(),
            "pair_count": torch.tensor(delta.shape[0], device=delta.device),
            "per_pair_total": per_pair_total,
            "per_pair_positive_stratum_mse": per_pair_positive_mse,
            "per_pair_zero_stratum_mse": per_pair_zero_mse,
        }


__all__ = ["PairedDifferenceLoss"]
