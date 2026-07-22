"""Public, backbone-independent CURE-Lite v0.1 API."""

__version__ = "0.1.0"

from .calibration import (
    CalibrationSample,
    FrozenThresholdProtocol,
    ThresholdSelection,
    select_anchor_threshold_by_miou,
    select_residual_threshold,
)
from .config import (
    CalibrationConfig,
    DecoderConfig,
    InterventionConfig,
    LossConfig,
    MatchConfig,
    OccupancyConfig,
    TrainingConfig,
)
from .decoder import CURELiteDecoder
from .frozen_base import FrozenBaseAdapter
from .instances import instances_from_binary_mask
from .intervention import enumerate_legal_deletions
from .losses import CURELiteLoss
from .matching import match_components
from .model import CURELiteModel, CURELiteOutput
from .occupancy import build_occupancy
from .sampling import choose_uniform_factual_gt_id, choose_uniform_legal_deletion
from .supervision import (
    build_epoch_factual_supervision_from_catalog,
    build_factual_supervision,
    build_factual_supervision_from_catalog,
    build_synthetic_supervision,
)
from .train import (
    BranchBatch,
    BranchPools,
    CURELiteTrainEngine,
    StateExample,
    iter_fixed_branch_batches,
    run_training_epoch,
)
from .types import (
    BranchSupervision,
    FrozenBaseOutput,
    InstanceMap,
    LegalDeletion,
    MatchResult,
)

__all__ = [
    "BranchBatch",
    "BranchPools",
    "BranchSupervision",
    "CURELiteDecoder",
    "CURELiteLoss",
    "CURELiteModel",
    "CURELiteOutput",
    "CURELiteTrainEngine",
    "CalibrationConfig",
    "CalibrationSample",
    "DecoderConfig",
    "FrozenBaseAdapter",
    "FrozenBaseOutput",
    "FrozenThresholdProtocol",
    "InstanceMap",
    "InterventionConfig",
    "LegalDeletion",
    "LossConfig",
    "MatchConfig",
    "MatchResult",
    "OccupancyConfig",
    "StateExample",
    "ThresholdSelection",
    "TrainingConfig",
    "build_epoch_factual_supervision_from_catalog",
    "build_factual_supervision",
    "build_factual_supervision_from_catalog",
    "build_occupancy",
    "build_synthetic_supervision",
    "choose_uniform_factual_gt_id",
    "choose_uniform_legal_deletion",
    "enumerate_legal_deletions",
    "instances_from_binary_mask",
    "iter_fixed_branch_batches",
    "match_components",
    "run_training_epoch",
    "select_anchor_threshold_by_miou",
    "select_residual_threshold",
]
