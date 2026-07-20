"""Full CURE: one miss-odds-matched coverage-intervention distribution."""

from .calibration import (
    CURECalibrationSample,
    CURETestEvaluationLedger,
    CURETestEvaluationResult,
    CUREThresholdSelection,
    FrozenCUREBaseThresholdProtocol,
    FrozenCUREThresholdProtocol,
    evaluate_cure_threshold,
    select_cure_threshold,
)
from .config import (
    CURELossConfig,
    CUREResidualConfig,
    DescriptorConfig,
    PropensityConfig,
)
from .decoder import CUREResidualDecoder
from .efficiency import CUREEfficiencyReport, measure_cure_incremental_efficiency
from .experiment import (
    CURETrainingPolicy,
    CounterfactualBackgroundPolicy,
    CounterfactualSamplingPolicy,
    CounterfactualTargetPolicy,
    build_fair_sampling_policy_family,
)
from .descriptors import (
    build_eligible_sample_catalog,
    dilate_mask,
)
from .losses import CUREUncensoringLoss
from .model import CUREModel, CUREOutput, noisy_or
from .protocol import (
    CUREProtocol,
    decoder_state_fingerprint,
    module_state_fingerprint,
)
from .propensity import (
    bind_weighted_candidates,
    choose_weighted_candidate,
    cross_fit_miss_propensity,
    miss_odds,
)
from .source import FrozenSourceRecord, extract_frozen_source_record
from .supervision import (
    build_counterfactual_residual_set_supervision,
    build_factual_residual_set_supervision,
)
from .training import (
    CUREBatch,
    CUREStateExample,
    CUREStatePool,
    build_cure_state_pool,
    draw_fixed_exposure_batch,
    train_cure_step,
)
from .types import (
    DESCRIPTOR_FIELDS,
    CUREInterventionCatalog,
    EligibleSampleCatalog,
    OOFPropensityResult,
    PropensityEstimate,
    ResidualSetSupervision,
)

__all__ = [
    "CUREBatch",
    "CUREInterventionCatalog",
    "CURECalibrationSample",
    "CUREEfficiencyReport",
    "CURETestEvaluationLedger",
    "CURETestEvaluationResult",
    "CUREThresholdSelection",
    "CURETrainingPolicy",
    "CURELossConfig",
    "CUREModel",
    "CUREOutput",
    "CUREProtocol",
    "CUREResidualConfig",
    "CUREResidualDecoder",
    "CUREStateExample",
    "CUREStatePool",
    "CUREUncensoringLoss",
    "DESCRIPTOR_FIELDS",
    "CounterfactualBackgroundPolicy",
    "CounterfactualSamplingPolicy",
    "CounterfactualTargetPolicy",
    "DescriptorConfig",
    "EligibleSampleCatalog",
    "FrozenCUREBaseThresholdProtocol",
    "FrozenCUREThresholdProtocol",
    "FrozenSourceRecord",
    "OOFPropensityResult",
    "PropensityConfig",
    "PropensityEstimate",
    "ResidualSetSupervision",
    "bind_weighted_candidates",
    "build_eligible_sample_catalog",
    "build_counterfactual_residual_set_supervision",
    "build_cure_state_pool",
    "build_factual_residual_set_supervision",
    "build_fair_sampling_policy_family",
    "choose_weighted_candidate",
    "cross_fit_miss_propensity",
    "dilate_mask",
    "decoder_state_fingerprint",
    "draw_fixed_exposure_batch",
    "evaluate_cure_threshold",
    "extract_frozen_source_record",
    "miss_odds",
    "measure_cure_incremental_efficiency",
    "module_state_fingerprint",
    "noisy_or",
    "select_cure_threshold",
    "train_cure_step",
]
