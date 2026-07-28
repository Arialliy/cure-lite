"""Append-only PACRE verifier-corrected candidate package for v23."""

from .pacre_vc import (
    PACRE_VC_CANDIDATE,
    PACRE_VC_FIELDS_FQCN,
    PACRE_VC_VERIFIER_POLICY,
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREFields,
    CoverageStatePACREVerifierCorrectedConfig,
)
from .factory import (
    PACRE_VC_MINIMUM_TRAINING_FEATURE_STRIDE,
    PACRE_VC_PARAMETER_NAMES,
    PACRE_VC_TRAINING_MODEL_FACTORY,
    PACREVCTrainingModel,
    PACREVCTrainingModelFactory,
    build_pacre_vc_training_model,
)
from .algebra_verifier import (
    PACRE_VC_ALGEBRA_POLICY,
    verify_pacre_forward_fields,
)
from .dataset_free import run_pacre_vc_dataset_free_gate
from .dr_gate import (
    PACRE_VC_DR_FAIL_DECISION,
    PACRE_VC_DR_PASS_DECISION,
    run_pacre_vc_dr_gate,
)
from .environment import stabilize_pacre_vc_numerical_runtime
from .numeric_stress import (
    run_pacre_vc_formal_numeric_stress_receipt,
    run_pacre_vc_scalar_counterexample_receipt,
)
from .parity import run_pacre_vc_generated_parity_receipt


__all__ = [
    "PACRE_VC_CANDIDATE",
    "PACRE_VC_ALGEBRA_POLICY",
    "PACRE_VC_DR_FAIL_DECISION",
    "PACRE_VC_DR_PASS_DECISION",
    "PACRE_VC_FIELDS_FQCN",
    "PACRE_VC_MINIMUM_TRAINING_FEATURE_STRIDE",
    "PACRE_VC_PARAMETER_NAMES",
    "PACRE_VC_TRAINING_MODEL_FACTORY",
    "PACRE_VC_VERIFIER_POLICY",
    "PACREVCTrainingModel",
    "PACREVCTrainingModelFactory",
    "CURELitePACREVerifierCorrectedLevelSet",
    "CoverageStatePACREFields",
    "CoverageStatePACREVerifierCorrectedConfig",
    "build_pacre_vc_training_model",
    "run_pacre_vc_dataset_free_gate",
    "run_pacre_vc_dr_gate",
    "run_pacre_vc_formal_numeric_stress_receipt",
    "run_pacre_vc_generated_parity_receipt",
    "run_pacre_vc_scalar_counterexample_receipt",
    "stabilize_pacre_vc_numerical_runtime",
    "verify_pacre_forward_fields",
]
