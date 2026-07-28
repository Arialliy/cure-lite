"""CURE-Lite v22 append-only research package.

The package is intentionally separate from :mod:`cure_lite`: the sealed v21
Formal800 closure dynamically inventories that package, so v22 must not
change its live source boundary.
"""

from .pacre import (
    CSLF_PACRE_CENTERING_POLICY,
    CSLF_PACRE_EQUATION_POLICY,
    CSLF_PACRE_FIELD_POLICY,
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyField,
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
    CoverageStatePACREFields,
    phase_centered_feature_affine,
)
from .factory import (
    PACRE_MINIMUM_TRAINING_FEATURE_STRIDE,
    PACRE_PARAMETER_NAMES,
    PACRE_TRAINING_MODEL_FACTORY,
    build_pacre_training_model,
)

__all__ = [
    "CSLF_PACRE_CENTERING_POLICY",
    "CSLF_PACRE_EQUATION_POLICY",
    "CSLF_PACRE_FIELD_POLICY",
    "CURELitePhaseAlignedCenteredResidualCompatibilityEnergyField",
    "CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet",
    "CoverageStatePACREConfig",
    "CoverageStatePACREFields",
    "PACRE_MINIMUM_TRAINING_FEATURE_STRIDE",
    "PACRE_PARAMETER_NAMES",
    "PACRE_TRAINING_MODEL_FACTORY",
    "build_pacre_training_model",
    "phase_centered_feature_affine",
]
