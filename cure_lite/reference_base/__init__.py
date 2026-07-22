"""Project-owned experiment provider for the first CURE development stage.

Nothing in this namespace is imported by the CURE-Lite method core.  It exists
only to create a real, reproducible frozen Base condition before integrations
with external IRSTD detectors are evaluated.
"""

from .adapter import ReferenceBaseAdapter, load_reference_base_adapter
from .config import (
    REFERENCE_BASE_CONFIG_SCHEMA,
    ReferenceBaseModelConfig,
    ReferenceBaseTrainingConfig,
)
from .model import ReferenceBaseNetwork, ReferenceBaseNetworkOutput
from .partition import DBPartition, build_d_b_partition
from .training import (
    LoadedReferenceBaseRun,
    load_reference_base_run,
    reference_base_loss,
    train_reference_base,
)

__all__ = [
    "DBPartition",
    "LoadedReferenceBaseRun",
    "REFERENCE_BASE_CONFIG_SCHEMA",
    "ReferenceBaseAdapter",
    "ReferenceBaseModelConfig",
    "ReferenceBaseNetwork",
    "ReferenceBaseNetworkOutput",
    "ReferenceBaseTrainingConfig",
    "build_d_b_partition",
    "load_reference_base_adapter",
    "load_reference_base_run",
    "reference_base_loss",
    "train_reference_base",
]
