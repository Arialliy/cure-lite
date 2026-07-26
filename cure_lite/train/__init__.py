"""Training primitives for CURE-Lite v0.1."""

from .engine import CURELiteTrainEngine, run_training_epoch
from .paired_step import (
    DECODER_STATES_PER_UPDATE,
    FACTUAL_ANCHOR_BRANCHES,
    FACTUAL_ANCHOR_BATCH_SIZE,
    NULL_PAIR_KINDS,
    PAIRED_BATCH_SIZE,
    diagnose_null_pairs,
    paired_endpoint_logits,
    paired_train_step,
)
from .pools import (
    BranchPools,
    StateExample,
    iter_factual_exposure_matched_batches,
    iter_fixed_branch_batches,
)
from .step import BRANCHES, BranchBatch, combine_branch_means, multi_branch_train_step

__all__ = [
    "BRANCHES",
    "BranchBatch",
    "BranchPools",
    "CURELiteTrainEngine",
    "DECODER_STATES_PER_UPDATE",
    "FACTUAL_ANCHOR_BRANCHES",
    "FACTUAL_ANCHOR_BATCH_SIZE",
    "NULL_PAIR_KINDS",
    "PAIRED_BATCH_SIZE",
    "StateExample",
    "combine_branch_means",
    "diagnose_null_pairs",
    "iter_factual_exposure_matched_batches",
    "iter_fixed_branch_batches",
    "multi_branch_train_step",
    "paired_endpoint_logits",
    "paired_train_step",
    "run_training_epoch",
]
