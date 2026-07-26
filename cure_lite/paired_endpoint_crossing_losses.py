"""Paired endpoint-crossing objective for CURE-Lite v10.

This module is additive to the frozen outcome-complete loss.  PECO changes
only the response-stratum risk: a response pixel must cross the decision
boundary in the correct direction, with the plus endpoint below zero and the
minus endpoint above zero.  The anchor, zero-response strata, hierarchical
pair weighting, input contract, and returned diagnostics remain identical to
``OutcomeCompleteTransitionLoss``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .paired_outcome_losses import (
    OutcomeCompleteTransitionLoss,
    _active_mean,
    _active_population_mean,
    _masked_means,
)


class PairedEndpointCrossingLoss(OutcomeCompleteTransitionLoss):
    """Outcome-complete loss with an endpoint-identifiable response risk.

    On the response stratum ``D``, PECO uses

    ``0.5 * (softplus(logits_plus) + softplus(-logits_minus))``.

    Consequently, minimizing this term moves the plus endpoint below zero
    and the minus endpoint above zero independently.  This rules out the
    both-high and both-low endpoint states that a probability-difference-only
    response objective cannot identify.  Outside ``D``, PECO is exactly the
    frozen parent objective.
    """

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
        """Return the frozen hierarchy with PECO response-stratum risk."""

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
        probability_delta = score_minus - score_plus
        response_error = 0.5 * (
            F.softplus(logits_plus) + F.softplus(-logits_minus)
        )
        zero_error = probability_delta.square()

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


__all__ = ["PairedEndpointCrossingLoss"]
