"""Uniform Sobolev--Chebyshev orthant projection energy.

USCOPE preserves the full-valid-domain PMOPE violation fields and combines
their existing joint ``W1,p4`` power with one statewise Chebyshev gauge.  It
adds no role masks, response coordinate, learned weight, or observed-data
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .coverage_state_sobolev import (
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    coverage_state_pmope_pair_loss_from_targets,
)


CSLF_USCOPE_POLICY = (
    "uniform_sobolev_chebyshev_orthant_projection_energy_v1"
)


@dataclass(frozen=True)
class CoverageStateUSCOPELossFields:
    """One joint full-domain Sobolev--Chebyshev projection gauge."""

    loss: Tensor
    per_state_loss: Tensor
    per_state_sobolev_power: Tensor
    per_state_chebyshev_violation: Tensor
    per_state_chebyshev_power: Tensor
    per_state_product_power: Tensor
    per_state_value_power: Tensor
    per_state_spatial_power: Tensor
    violation_plus: Tensor
    violation_minus: Tensor
    margin: Tensor
    target_field_plus: Tensor
    target_field_minus: Tensor
    integration_measure: Tensor
    valid_mask: Tensor


def coverage_state_uscope_pair_loss_from_targets(
    field_plus: Tensor,
    field_minus: Tensor,
    targets: CoverageStatePairTargets,
    *,
    config: CoverageStateSobolevConfig,
    validate: bool = True,
) -> CoverageStateUSCOPELossFields:
    """Evaluate the fixed USCOPE product gauge for one pair batch.

    Let ``q_plus`` and ``q_minus`` be the existing PMOPE minimum-margin
    orthant violations over the complete valid domain.  For every pair state,

    ``A = 0.5 * (value_power + spatial_power)``,

    ``gamma = max(q_plus, q_minus)`` over both endpoints and all valid pixels,

    ``C = gamma**4``,

    and the loss is the rooted, epsilon-stabilized uniform product gauge

    ``[0.5 * (A + C) + epsilon**4]**(1/4) - epsilon``.
    """

    if not isinstance(validate, bool):
        raise TypeError("validate must be bool")
    pmope = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
        validate=validate,
    )
    sobolev_power = 0.5 * (
        pmope.per_state_value_power
        + pmope.per_state_spatial_power
    )
    chebyshev_violation = torch.maximum(
        pmope.violation_plus.flatten(1).amax(dim=1),
        pmope.violation_minus.flatten(1).amax(dim=1),
    )
    chebyshev_power = chebyshev_violation.pow(config.norm_order)
    product_power = 0.5 * (sobolev_power + chebyshev_power)
    per_state_loss = (
        product_power + config.norm_epsilon**config.norm_order
    ).pow(1.0 / float(config.norm_order)) - config.norm_epsilon
    loss = per_state_loss.mean()
    if validate and not bool(
        torch.stack(
            (
                torch.isfinite(loss),
                torch.isfinite(per_state_loss).all(),
                torch.isfinite(sobolev_power).all(),
                torch.isfinite(chebyshev_violation).all(),
                torch.isfinite(chebyshev_power).all(),
                torch.isfinite(product_power).all(),
            )
        ).all()
    ):
        raise FloatingPointError("USCOPE is non-finite")
    return CoverageStateUSCOPELossFields(
        loss=loss,
        per_state_loss=per_state_loss,
        per_state_sobolev_power=sobolev_power,
        per_state_chebyshev_violation=chebyshev_violation,
        per_state_chebyshev_power=chebyshev_power,
        per_state_product_power=product_power,
        per_state_value_power=pmope.per_state_value_power,
        per_state_spatial_power=pmope.per_state_spatial_power,
        violation_plus=pmope.violation_plus,
        violation_minus=pmope.violation_minus,
        margin=pmope.margin,
        target_field_plus=pmope.target_field_plus,
        target_field_minus=pmope.target_field_minus,
        integration_measure=pmope.integration_measure,
        valid_mask=pmope.valid_mask,
    )


__all__ = [
    "CSLF_USCOPE_POLICY",
    "CoverageStateUSCOPELossFields",
    "coverage_state_uscope_pair_loss_from_targets",
]
