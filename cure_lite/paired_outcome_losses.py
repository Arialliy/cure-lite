"""Unified hierarchical transition risk for OC-APTO v3.

The criterion is additive to the frozen v1/v2 implementations.  It consumes
one score-difference field and three tensor-defined semantic strata; it never
receives or dispatches on a pair kind.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import LossConfig
from .losses import CURELiteLoss


def _masked_means(
    values: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return safe per-pair means, counts, and active flags.

    Empty strata return a differentiable zero.  No empty tensor reduction is
    evaluated, so inactive strata cannot introduce NaNs into either the
    forward value or backward graph.
    """

    flat_values = values.flatten(1)
    flat_mask = mask.flatten(1)
    counts = flat_mask.sum(dim=1)
    sums = (flat_values * flat_mask.to(dtype=values.dtype)).sum(dim=1)
    means = sums / counts.clamp_min(1).to(dtype=values.dtype)
    active = counts > 0
    return means, counts, active


def _active_mean(
    values: Tensor,
    active: Tensor,
    *,
    name: str,
) -> tuple[Tensor, Tensor]:
    """Average active group values independently for every pair."""

    if values.ndim != 2 or active.shape != values.shape:
        raise ValueError(f"{name} values/active must have shape [B,K]")
    active_count = active.sum(dim=1)
    if not torch.all(active_count > 0):
        raise ValueError(f"every pair requires an active {name} group")
    weighted = values * active.to(dtype=values.dtype)
    result = weighted.sum(dim=1) / active_count.to(dtype=values.dtype)
    return result, active_count


def _active_population_mean(
    values: Tensor,
    active: Tensor,
) -> Tensor:
    """Return a finite scalar diagnostic over active pairs only."""

    count = active.sum()
    return (
        (values * active.to(dtype=values.dtype)).sum()
        / count.clamp_min(1).to(dtype=values.dtype)
    )


class OutcomeCompleteTransitionLoss(nn.Module):
    """OC-APTO's fixed baseline plus hierarchical transition objective.

    For each pair, the response stratum is ``D``; the local zero-response
    stratum is ``H = J \\ D``; and the global zero-response stratum is
    ``G = V \\ (D union J)``.  The zero risk is the active mean of ``H`` and
    ``G``.  The transition risk is then the active mean of ``D`` and the zero
    risk.  Thus an empty-``D`` pair automatically uses only the same zero-risk
    group, without a pair-kind branch.
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        if not isinstance(config, LossConfig):
            raise TypeError("config must be LossConfig")
        self.config = config
        self.plus_anchor_criterion = CURELiteLoss(config)

    @staticmethod
    def _validate(
        logits_plus: Tensor,
        logits_minus: Tensor,
        completion_plus: Tensor,
        occupancy_plus: Tensor,
        gt_union: Tensor,
        label_increment: Tensor,
        image_valid_mask: Tensor,
        intervention_footprint: Tensor,
    ) -> None:
        values = {
            "logits_plus": logits_plus,
            "logits_minus": logits_minus,
            "completion_plus": completion_plus,
            "occupancy_plus": occupancy_plus,
            "gt_union": gt_union,
            "label_increment": label_increment,
            "image_valid_mask": image_valid_mask,
            "intervention_footprint": intervention_footprint,
        }
        if any(not isinstance(value, Tensor) for value in values.values()):
            raise TypeError("all OC-APTO loss inputs must be tensors")
        if logits_plus.ndim != 4 or logits_plus.shape[1] != 1:
            raise ValueError("OC-APTO tensors must have shape [B,1,H,W]")
        if logits_plus.shape[0] < 1 or min(logits_plus.shape[-2:]) < 1:
            raise ValueError("OC-APTO tensors must be non-empty")
        if any(value.shape != logits_plus.shape for value in values.values()):
            raise ValueError("all OC-APTO tensors must have identical shapes")

        if not logits_plus.is_floating_point() or not logits_minus.is_floating_point():
            raise TypeError("endpoint logits must be floating point")
        if logits_plus.dtype != logits_minus.dtype:
            raise TypeError("endpoint logits must share a dtype")
        for name in (
            "completion_plus",
            "occupancy_plus",
            "gt_union",
            "image_valid_mask",
            "intervention_footprint",
        ):
            if values[name].dtype != torch.bool:
                raise TypeError(f"{name} must be bool")
        if label_increment.dtype != torch.float32:
            raise TypeError("label_increment must be float32")

        devices = {value.device for value in values.values()}
        if len(devices) != 1:
            raise ValueError("all OC-APTO tensors must share a device")
        if not torch.isfinite(logits_plus).all():
            raise ValueError("logits_plus must be finite")
        if not torch.isfinite(logits_minus).all():
            raise ValueError("logits_minus must be finite")
        if not torch.isfinite(label_increment).all():
            raise ValueError("label_increment must be finite")
        if torch.any((label_increment != 0.0) & (label_increment != 1.0)):
            raise ValueError("label_increment must be binary")

        valid = image_valid_mask
        response = label_increment.to(dtype=torch.bool)
        if not torch.all(valid.flatten(1).any(dim=1)):
            raise ValueError("every pair requires a non-empty valid domain")
        if not torch.all(intervention_footprint.flatten(1).any(dim=1)):
            raise ValueError(
                "every pair requires a non-empty intervention footprint"
            )
        if torch.any(occupancy_plus & ~valid):
            raise ValueError("occupancy_plus lies outside image_valid_mask")
        if torch.any(gt_union & ~valid):
            raise ValueError("gt_union lies outside image_valid_mask")
        if torch.any(intervention_footprint & ~valid):
            raise ValueError(
                "intervention_footprint lies outside image_valid_mask"
            )
        if torch.any(response & ~valid):
            raise ValueError("label_increment lies outside image_valid_mask")
        if torch.any(response & ~gt_union):
            raise ValueError("label_increment must contain only GT pixels")
        if torch.any(completion_plus & (~valid | occupancy_plus)):
            raise ValueError(
                "completion_plus must be valid and writable under occupancy_plus"
            )
        if torch.any(completion_plus & ~gt_union):
            raise ValueError("completion_plus must contain only GT pixels")
        if torch.any(completion_plus & response):
            raise ValueError(
                "completion_plus and label_increment must be disjoint"
            )

        local_zero = intervention_footprint & ~response
        global_zero = valid & ~response & ~intervention_footprint
        if not torch.all(
            (local_zero | global_zero).flatten(1).any(dim=1)
        ):
            raise ValueError("every pair requires a zero-response stratum")

        anchor_background = valid & ~occupancy_plus & ~gt_union
        anchor_valid = completion_plus | anchor_background
        if not torch.all(anchor_valid.flatten(1).any(dim=1)):
            raise ValueError("every pair requires non-empty plus-anchor supervision")

    def forward(
        self,
        logits_plus: Tensor,
        logits_minus: Tensor,
        completion_plus: Tensor,
        occupancy_plus: Tensor,
        gt_union: Tensor,
        label_increment: Tensor,
        image_valid_mask: Tensor,
        intervention_footprint: Tensor,
    ) -> dict[str, Tensor]:
        """Return equally pair-weighted OC-APTO risks and stratum diagnostics."""

        self._validate(
            logits_plus,
            logits_minus,
            completion_plus,
            occupancy_plus,
            gt_union,
            label_increment,
            image_valid_mask,
            intervention_footprint,
        )
        response = label_increment.to(dtype=torch.bool) & image_valid_mask
        local_zero = intervention_footprint & image_valid_mask & ~response
        global_zero = (
            image_valid_mask
            & ~response
            & ~intervention_footprint
        )

        anchor_background = (
            image_valid_mask & ~occupancy_plus & ~gt_union
        )
        anchor_valid = completion_plus | anchor_background
        anchor_result = self.plus_anchor_criterion(
            logits_plus,
            completion_plus.to(dtype=torch.float32),
            anchor_valid,
        )
        per_pair_anchor = anchor_result["per_state_total"]

        score_plus = torch.sigmoid(logits_plus)
        score_minus = torch.sigmoid(logits_minus)
        delta = score_minus - score_plus
        response_error = ((delta - 1.0) / 2.0).square()
        zero_error = delta.square()

        response_mean, response_count, response_active = _masked_means(
            response_error,
            response,
        )
        local_mean, local_count, local_active = _masked_means(
            zero_error,
            local_zero,
        )
        global_mean, global_count, global_active = _masked_means(
            zero_error,
            global_zero,
        )
        per_pair_zero, zero_active_strata = _active_mean(
            torch.stack((local_mean, global_mean), dim=1),
            torch.stack((local_active, global_active), dim=1),
            name="zero-response",
        )
        per_pair_transition, transition_active_groups = _active_mean(
            torch.stack((response_mean, per_pair_zero), dim=1),
            torch.stack(
                (
                    response_active,
                    torch.ones_like(response_active),
                ),
                dim=1,
            ),
            name="transition",
        )

        per_pair_total = (
            0.5 * per_pair_anchor + 0.5 * per_pair_transition
        )
        total = per_pair_total.mean()
        return {
            "total": total,
            "loss": total,
            "plus_anchor_loss": per_pair_anchor.mean(),
            "transition_loss": per_pair_transition.mean(),
            "zero_risk": per_pair_zero.mean(),
            "response_stratum_loss": _active_population_mean(
                response_mean,
                response_active,
            ),
            "local_zero_stratum_loss": _active_population_mean(
                local_mean,
                local_active,
            ),
            "global_zero_stratum_loss": _active_population_mean(
                global_mean,
                global_active,
            ),
            "per_pair_total": per_pair_total,
            "per_pair_plus_anchor": per_pair_anchor,
            "per_pair_transition": per_pair_transition,
            "per_pair_zero_risk": per_pair_zero,
            "per_pair_response_stratum": response_mean,
            "per_pair_local_zero_stratum": local_mean,
            "per_pair_global_zero_stratum": global_mean,
            "response_pixels_per_pair": response_count,
            "local_zero_pixels_per_pair": local_count,
            "global_zero_pixels_per_pair": global_count,
            "response_active_per_pair": response_active,
            "local_zero_active_per_pair": local_active,
            "global_zero_active_per_pair": global_active,
            "zero_active_strata_per_pair": zero_active_strata,
            "transition_active_groups_per_pair": transition_active_groups,
            "response_stratum": response,
            "local_zero_stratum": local_zero,
            "global_zero_stratum": global_zero,
            "plus_anchor_target": completion_plus.to(dtype=torch.float32),
            "plus_anchor_background": anchor_background,
            "plus_anchor_valid_mask": anchor_valid,
            "pair_count": torch.tensor(
                logits_plus.shape[0],
                device=logits_plus.device,
            ),
        }


__all__ = ["OutcomeCompleteTransitionLoss"]
