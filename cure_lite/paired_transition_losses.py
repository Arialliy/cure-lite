"""Anchored state-transition objective for CURE-Lite paired examples.

The loss keeps the existing coupled pre-mask transition objective, while
anchoring the ``plus`` endpoint as a valid absolute residual prediction.  It
does not own a decoder forward and does not change the inference graph.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .losses import CURELiteLoss
from .paired_losses import PairedDifferenceLoss


class AnchoredTransitionLoss(nn.Module):
    """Combine a geometry-matched plus anchor with paired score transition.

    For each pair, the plus-endpoint supervision is

    ``target = completion_plus``,
    ``background = V & ~O_plus & ~G``, and
    ``valid = target | background``.

    The transition term is the existing :class:`PairedDifferenceLoss`.  Pair
    reductions are performed before the batch mean, and the frozen objective
    is

    ``0.5 * plus_anchor + 0.5 * transition``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.plus_anchor_criterion = CURELiteLoss()
        self.transition_criterion = PairedDifferenceLoss()

    @staticmethod
    def _validate(
        logits_plus: Tensor,
        logits_minus: Tensor,
        completion_plus: Tensor,
        occupancy_plus: Tensor,
        gt_union: Tensor,
        label_increment: Tensor,
        image_valid_mask: Tensor,
    ) -> None:
        values = {
            "logits_plus": logits_plus,
            "logits_minus": logits_minus,
            "completion_plus": completion_plus,
            "occupancy_plus": occupancy_plus,
            "gt_union": gt_union,
            "label_increment": label_increment,
            "image_valid_mask": image_valid_mask,
        }
        if any(not isinstance(value, Tensor) for value in values.values()):
            raise TypeError("all anchored-transition inputs must be tensors")
        if logits_plus.ndim != 4 or logits_plus.shape[1] != 1:
            raise ValueError("anchored-transition tensors must have shape [B,1,H,W]")
        if logits_plus.shape[0] < 1 or min(logits_plus.shape[-2:]) < 1:
            raise ValueError("anchored-transition tensors must be non-empty")
        if any(value.shape != logits_plus.shape for value in values.values()):
            raise ValueError("all anchored-transition tensors must have identical shapes")

        if not logits_plus.is_floating_point() or not logits_minus.is_floating_point():
            raise TypeError("endpoint logits must be floating point")
        if logits_plus.dtype != logits_minus.dtype:
            raise TypeError("endpoint logits must share a dtype")
        for name in (
            "completion_plus",
            "occupancy_plus",
            "gt_union",
            "image_valid_mask",
        ):
            if values[name].dtype != torch.bool:
                raise TypeError(f"{name} must be bool")
        if label_increment.dtype != torch.float32:
            raise TypeError("label_increment must be float32")

        devices = {value.device for value in values.values()}
        if len(devices) != 1:
            raise ValueError("all anchored-transition tensors must share a device")
        if not torch.isfinite(logits_plus).all():
            raise ValueError("logits_plus must be finite")
        if not torch.isfinite(logits_minus).all():
            raise ValueError("logits_minus must be finite")
        if not torch.isfinite(label_increment).all():
            raise ValueError("label_increment must be finite")
        if torch.any((label_increment != 0.0) & (label_increment != 1.0)):
            raise ValueError("label_increment must be binary")

        valid = image_valid_mask
        increment = label_increment.to(dtype=torch.bool)
        if torch.any(occupancy_plus & ~valid):
            raise ValueError("occupancy_plus lies outside image_valid_mask")
        if torch.any(gt_union & ~valid):
            raise ValueError("gt_union lies outside image_valid_mask")
        if torch.any(completion_plus & (~valid | occupancy_plus)):
            raise ValueError(
                "completion_plus must be valid and writable under occupancy_plus"
            )
        if torch.any(completion_plus & ~gt_union):
            raise ValueError("completion_plus must contain only GT pixels")
        if torch.any(increment & ~valid):
            raise ValueError("label_increment lies outside image_valid_mask")
        if torch.any(increment & ~gt_union):
            raise ValueError("label_increment must contain only GT pixels")
        if torch.any(increment & completion_plus):
            raise ValueError(
                "label_increment must be disjoint from completion_plus"
            )

        anchor_background = valid & ~occupancy_plus & ~gt_union
        anchor_valid = completion_plus | anchor_background
        if not torch.all(anchor_valid.flatten(1).any(dim=1)):
            raise ValueError("every pair requires non-empty plus-anchor supervision")

        # Delegate the remaining positive/zero-response transition semantics to
        # the frozen transition criterion, but validate them before either
        # component loss is evaluated.
        PairedDifferenceLoss._validate(
            logits_plus,
            logits_minus,
            label_increment,
            image_valid_mask,
        )

    def forward(
        self,
        logits_plus: Tensor,
        logits_minus: Tensor,
        completion_plus: Tensor,
        occupancy_plus: Tensor,
        gt_union: Tensor,
        label_increment: Tensor,
        image_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Return the equally pair-weighted anchored transition objective."""

        self._validate(
            logits_plus,
            logits_minus,
            completion_plus,
            occupancy_plus,
            gt_union,
            label_increment,
            image_valid_mask,
        )
        anchor_background = image_valid_mask & ~occupancy_plus & ~gt_union
        anchor_valid = completion_plus | anchor_background
        anchor_result = self.plus_anchor_criterion(
            logits_plus,
            completion_plus.to(dtype=torch.float32),
            anchor_valid,
        )
        transition_result = self.transition_criterion(
            logits_plus,
            logits_minus,
            label_increment,
            image_valid_mask,
        )

        per_pair_anchor = anchor_result["per_state_total"]
        per_pair_transition = transition_result["per_pair_total"]
        if (
            per_pair_anchor.ndim != 1
            or per_pair_transition.shape != per_pair_anchor.shape
        ):
            raise RuntimeError("component criteria did not return one value per pair")
        per_pair_total = 0.5 * per_pair_anchor + 0.5 * per_pair_transition
        total = per_pair_total.mean()
        return {
            "total": total,
            "loss": total,
            "plus_anchor_loss": per_pair_anchor.mean(),
            "transition_loss": per_pair_transition.mean(),
            "per_pair_total": per_pair_total,
            "per_pair_plus_anchor": per_pair_anchor,
            "per_pair_transition": per_pair_transition,
            "plus_anchor_target": completion_plus.to(dtype=torch.float32),
            "plus_anchor_background": anchor_background,
            "plus_anchor_valid_mask": anchor_valid,
            "pair_count": torch.tensor(
                logits_plus.shape[0],
                device=logits_plus.device,
            ),
        }


__all__ = ["AnchoredTransitionLoss"]
