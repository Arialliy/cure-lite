"""Versioned PACRE wrapper with a verifier-corrected research contract.

PACRE-VC v23 deliberately inherits the complete numerical forward from v22.
It adds only a versioned configuration and model FQCN so that new verifier
receipts cannot be confused with the sealed v22 attempt.

The inherited ``forward_fields`` method continues to return
``cure_lite_v22.pacre.CoverageStatePACREFields``.  No v23 fields dataclass is
introduced because changing that construction path would weaken the exact
forward-parity contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
    CoverageStatePACREFields,
)


PACRE_VC_CANDIDATE: Final = "PACRE-VC-v23"
PACRE_VC_VERIFIER_POLICY: Final = (
    "v22_forward_unchanged_exact_replay_"
    "analytic_phase_roundoff_verifier_v1"
)
PACRE_VC_FIELDS_FQCN: Final = (
    "cure_lite_v22.pacre.CoverageStatePACREFields"
)


@dataclass(frozen=True)
class CoverageStatePACREVerifierCorrectedConfig(CoverageStatePACREConfig):
    """The frozen v22 PACRE model config plus one verifier policy."""

    verifier_policy: str = PACRE_VC_VERIFIER_POLICY

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.verifier_policy != PACRE_VC_VERIFIER_POLICY:
            raise ValueError("PACRE-VC fixes verifier_policy")


class CURELitePACREVerifierCorrectedLevelSet(
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
):
    """Versioned wrapper whose numerical forward is inherited from v22."""

    config: CoverageStatePACREVerifierCorrectedConfig

    def __init__(
        self,
        config: CoverageStatePACREVerifierCorrectedConfig,
    ) -> None:
        if type(config) is not CoverageStatePACREVerifierCorrectedConfig:
            raise TypeError(
                "config must be "
                "CoverageStatePACREVerifierCorrectedConfig"
            )

        # The sealed v22 constructor intentionally accepts only the exact v22
        # config type.  All inherited model fields are already frozen by the
        # v22 ``__post_init__``, so rebuilding that exact config changes no
        # numerical input to parameter initialization or the forward.
        base_config = CoverageStatePACREConfig(
            feature_channels=config.feature_channels,
            feature_stride=config.feature_stride,
            width=config.width,
        )
        super().__init__(base_config)

        # Bind receipts and model contracts to the new FQCN/policy while every
        # numerical method remains inherited unchanged.
        self.config = config


__all__ = [
    "PACRE_VC_CANDIDATE",
    "PACRE_VC_FIELDS_FQCN",
    "PACRE_VC_VERIFIER_POLICY",
    "CURELitePACREVerifierCorrectedLevelSet",
    "CoverageStatePACREFields",
    "CoverageStatePACREVerifierCorrectedConfig",
]
