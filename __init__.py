"""Backbone-independent CURE-Lite v0.1 core primitives."""

from .config import (
    CalibrationConfig,
    DecoderConfig,
    InterventionConfig,
    LossConfig,
    MatchConfig,
    OccupancyConfig,
    TrainingConfig,
    config_to_dict,
)
from .instances import (
    centroid_distance,
    instances_from_binary_mask,
    mask_iou,
    union_instance_masks,
)
from .intervention import enumerate_legal_deletions, oracle_restores_base_coverage
from .matching import match_components
from .occupancy import (
    build_occupancy,
    build_occupancy_batch,
    decompose_occupancy,
    threshold_occupancy,
)
from .sampling import choose_uniform_legal_deletion, stable_hash
from .provenance import (
    BaseTrainingProvenance,
    BaseTrainingProvenanceError,
    BaseTrainingSample,
    FormalBaseTrainingIdentity,
    validate_base_training_provenance,
    validate_formal_base_training_run,
)
from .supervision import (
    build_factual_supervision,
    build_synthetic_supervision,
    factual_oracle_reachable,
)
from .types import (
    BranchSupervision,
    FrozenBaseOutput,
    Instance,
    InstanceMap,
    LegalDeletion,
    MatchPair,
    MatchResult,
)

__all__ = [
    "BranchSupervision",
    "BaseTrainingProvenance",
    "BaseTrainingProvenanceError",
    "BaseTrainingSample",
    "FormalBaseTrainingIdentity",
    "CalibrationConfig",
    "DecoderConfig",
    "FrozenBaseOutput",
    "Instance",
    "InstanceMap",
    "InterventionConfig",
    "LegalDeletion",
    "LossConfig",
    "MatchConfig",
    "MatchPair",
    "MatchResult",
    "OccupancyConfig",
    "TrainingConfig",
    "build_factual_supervision",
    "build_occupancy",
    "build_occupancy_batch",
    "build_synthetic_supervision",
    "centroid_distance",
    "choose_uniform_legal_deletion",
    "config_to_dict",
    "decompose_occupancy",
    "enumerate_legal_deletions",
    "factual_oracle_reachable",
    "instances_from_binary_mask",
    "mask_iou",
    "match_components",
    "oracle_restores_base_coverage",
    "stable_hash",
    "threshold_occupancy",
    "union_instance_masks",
    "validate_base_training_provenance",
    "validate_formal_base_training_run",
]
