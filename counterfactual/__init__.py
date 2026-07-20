"""Independent input-level counterfactual primitives for full CURE."""

from .acceptance import (
    AcceptanceConfig,
    AcceptanceDecision,
    assess_legal_intervention,
)
from .contracts import (
    LOCAL_CONTRAST_ATTENUATION,
    TransformConfig,
    TransformDiagnostics,
    TransformSpec,
)
from .transforms import (
    InvalidTransformSupportError,
    apply_counterfactual_transform,
    build_background_ring,
    build_soft_roi,
)
from .search import (
    CounterfactualSearchResult,
    InterventionAttempt,
    LegalInterventionReceipt,
    ModelConsistentState,
    search_minimal_legal_intervention,
)

__all__ = [
    "AcceptanceConfig",
    "AcceptanceDecision",
    "CounterfactualSearchResult",
    "InterventionAttempt",
    "InvalidTransformSupportError",
    "LOCAL_CONTRAST_ATTENUATION",
    "LegalInterventionReceipt",
    "ModelConsistentState",
    "TransformConfig",
    "TransformDiagnostics",
    "TransformSpec",
    "apply_counterfactual_transform",
    "assess_legal_intervention",
    "build_background_ring",
    "build_soft_roi",
    "search_minimal_legal_intervention",
]
