"""One-update training primitive for outcome-complete CURE-Lite v3.

This module is additive.  The paired-v1 and anchored-transition-v2 training
paths remain frozen.  OC-APTO keeps the same two factual anchors, one shared
decoder, three decoder calls, twelve evaluated states, and one optimizer
update, while admitting both clean-positive and component-null outcomes into
one pair branch.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..losses import CURELiteLoss
from ..paired_outcome_losses import OutcomeCompleteTransitionLoss
from ..paired_outcome_types import OutcomePairBatch
from .paired_step import (
    FACTUAL_ANCHOR_BATCH_SIZE,
    FACTUAL_ANCHOR_BRANCHES,
    _criterion_total,
    _paired_endpoint_logits,
)
from .step import BranchBatch, _validate_optimizer_scope


OUTCOME_PAIR_BATCH_SIZE = 2
OUTCOME_ENDPOINT_STATES_PER_UPDATE = 2 * OUTCOME_PAIR_BATCH_SIZE
DECODER_STATES_PER_UPDATE = (
    2 * FACTUAL_ANCHOR_BATCH_SIZE + OUTCOME_ENDPOINT_STATES_PER_UPDATE
)
DECODER_FORWARD_CALLS_PER_UPDATE = 3
OUTCOME_OPTIMIZER_PAIR_KINDS = ("clean_positive", "component_null")


def _validate_outcome_batch(batch: OutcomePairBatch) -> None:
    """Revalidate the complete mutable-tensor outcome object."""

    if not isinstance(batch, OutcomePairBatch):
        raise TypeError("outcome_batch must be OutcomePairBatch")
    batch.validate()
    pair_batch = batch.pair_batch
    if int(pair_batch.feature.shape[0]) != OUTCOME_PAIR_BATCH_SIZE:
        raise ValueError(
            "outcome training requires exactly "
            f"{OUTCOME_PAIR_BATCH_SIZE} outcome pairs"
        )
    if len(set(pair_batch.pair_ids)) != OUTCOME_PAIR_BATCH_SIZE:
        raise ValueError("pair_ids must be unique within one outcome batch")
    if len(set(pair_batch.sample_ids)) != OUTCOME_PAIR_BATCH_SIZE:
        raise ValueError(
            "the two outcome pairs must come from distinct source samples"
        )
    if any(
        kind not in OUTCOME_OPTIMIZER_PAIR_KINDS
        for kind in pair_batch.pair_kinds
    ):
        raise ValueError(
            "outcome optimizer accepts only clean_positive/component_null pairs"
        )
    anchor_background = (
        pair_batch.image_valid_mask
        & ~pair_batch.occupancy_plus
        & ~batch.gt_union
    )
    anchor_valid = batch.completion_plus | anchor_background
    if not torch.all(anchor_valid.flatten(1).any(dim=1)):
        raise ValueError(
            "every outcome pair requires non-empty plus-anchor supervision"
        )


def _preflight_outcome_training(
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    factual_batches: Mapping[str, BranchBatch],
    outcome_batch: OutcomePairBatch,
) -> dict[str, BranchBatch]:
    """Validate a complete update before mode, gradients, or state can change."""

    _validate_outcome_batch(outcome_batch)
    _validate_optimizer_scope(decoder, optimizer)

    normalized: dict[str, BranchBatch] = {}
    for branch in FACTUAL_ANCHOR_BRANCHES:
        batch = factual_batches[branch]
        if not isinstance(batch, BranchBatch):
            raise TypeError(f"{branch} batch must be BranchBatch")
        batch = batch.batched()
        batch.validate(expected_branch=branch)
        if int(batch.feature.shape[0]) != FACTUAL_ANCHOR_BATCH_SIZE:
            raise ValueError(
                f"{branch} requires exactly "
                f"{FACTUAL_ANCHOR_BATCH_SIZE} states"
            )
        normalized[branch] = batch

    all_features = (
        normalized["factual_miss"].feature,
        normalized["factual_no_miss"].feature,
        outcome_batch.pair_batch.feature,
    )
    all_occupancies = (
        normalized["factual_miss"].occupancy,
        normalized["factual_no_miss"].occupancy,
        outcome_batch.pair_batch.occupancy_plus,
    )
    if len({feature.device for feature in all_features}) != 1:
        raise ValueError("all training features must share a device")
    if len({feature.dtype for feature in all_features}) != 1:
        raise TypeError("all training features must share a dtype")
    if len({tuple(feature.shape[1:]) for feature in all_features}) != 1:
        raise ValueError("all training features must share [C,h,w]")
    if len({tuple(value.shape[1:]) for value in all_occupancies}) != 1:
        raise ValueError("all training state tensors must share [1,H,W]")

    parameters = tuple(decoder.parameters())
    parameter_devices = {parameter.device for parameter in parameters}
    parameter_dtypes = {parameter.dtype for parameter in parameters}
    if len(parameter_devices) != 1 or len(parameter_dtypes) != 1:
        raise ValueError("decoder parameters must share one device and dtype")
    feature_device = all_features[0].device
    feature_dtype = all_features[0].dtype
    if feature_device != next(iter(parameter_devices)):
        raise ValueError(
            "training features and decoder parameters must share a device"
        )
    if feature_dtype != next(iter(parameter_dtypes)):
        raise TypeError(
            "training feature dtype must match decoder parameter dtype"
        )

    expected_channels = getattr(decoder, "feature_channels", None)
    if (
        expected_channels is not None
        and int(all_features[0].shape[1]) != int(expected_channels)
    ):
        raise ValueError("training feature channels do not match decoder")
    feature_grid = tuple(int(value) for value in all_features[0].shape[-2:])
    evaluation_grid = tuple(
        int(value) for value in all_occupancies[0].shape[-2:]
    )
    if any(
        feature_size > evaluation_size
        for feature_size, evaluation_size in zip(
            feature_grid,
            evaluation_grid,
            strict=True,
        )
    ):
        raise ValueError("decoder occupancy projection may not upsample")
    return normalized


def outcome_complete_train_step(
    decoder: nn.Module,
    absolute_criterion: CURELiteLoss,
    outcome_criterion: OutcomeCompleteTransitionLoss,
    optimizer: torch.optim.Optimizer,
    factual_batches: Mapping[str, BranchBatch],
    outcome_batch: OutcomePairBatch,
) -> dict[str, float | int]:
    """Run the fixed 4/4/2 OC-APTO update with one shared decoder.

    The objective is

    ``L_factual_miss + L_factual_no_miss + L_outcome_complete``.

    GT, completion truth, and the intervention footprint are supplied only to
    the outcome loss.  Decoder calls consume exactly ``(feature, occupancy)``.
    Both pair endpoint logits remain attached to the same backward graph.
    """

    if not isinstance(decoder, nn.Module):
        raise TypeError("decoder must be an nn.Module")
    if not isinstance(absolute_criterion, CURELiteLoss):
        raise TypeError("absolute_criterion must be CURELiteLoss")
    if not isinstance(outcome_criterion, OutcomeCompleteTransitionLoss):
        raise TypeError(
            "outcome_criterion must be OutcomeCompleteTransitionLoss"
        )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(factual_batches, Mapping):
        raise TypeError("factual_batches must be a mapping")
    if set(factual_batches) != set(FACTUAL_ANCHOR_BRANCHES):
        raise ValueError(
            "factual_batches must contain exactly factual_miss and "
            "factual_no_miss"
        )

    normalized_factual = _preflight_outcome_training(
        decoder,
        optimizer,
        factual_batches,
        outcome_batch,
    )

    decoder.train()
    optimizer.zero_grad(set_to_none=True)
    losses: dict[str, Tensor] = {}
    logs: dict[str, float | int] = {}

    for branch in FACTUAL_ANCHOR_BRANCHES:
        batch = normalized_factual[branch]
        logits = decoder(batch.feature.detach(), batch.occupancy)
        result = absolute_criterion(
            logits,
            batch.target,
            batch.valid_mask,
        )
        loss = _criterion_total(result, name=f"{branch} criterion")
        losses[branch] = loss
        logs[f"{branch}/states"] = int(batch.feature.shape[0])
        logs[f"{branch}/loss"] = float(loss.detach().cpu())

    pair_batch = outcome_batch.pair_batch
    logits_plus, logits_minus = _paired_endpoint_logits(
        decoder,
        feature=pair_batch.feature,
        occupancy_plus=pair_batch.occupancy_plus,
        occupancy_minus=pair_batch.occupancy_minus,
    )
    outcome_result = outcome_criterion(
        logits_plus,
        logits_minus,
        outcome_batch.completion_plus,
        pair_batch.occupancy_plus,
        outcome_batch.gt_union,
        pair_batch.label_increment,
        pair_batch.image_valid_mask,
        outcome_batch.intervention_footprint,
    )
    outcome_loss = _criterion_total(
        outcome_result,
        name="outcome-complete criterion",
    )
    losses["outcome_complete"] = outcome_loss

    clean_pairs = sum(
        kind == "clean_positive" for kind in pair_batch.pair_kinds
    )
    component_pairs = sum(
        kind == "component_null" for kind in pair_batch.pair_kinds
    )
    logs["outcome/pairs"] = int(pair_batch.feature.shape[0])
    logs["outcome/endpoints"] = 2 * int(pair_batch.feature.shape[0])
    logs["outcome/clean_pairs"] = int(clean_pairs)
    logs["outcome/component_null_pairs"] = int(component_pairs)
    logs["outcome/plus_anchor_loss"] = float(
        outcome_result["plus_anchor_loss"].detach().cpu()
    )
    logs["outcome/transition_loss"] = float(
        outcome_result["transition_loss"].detach().cpu()
    )
    logs["outcome/loss"] = float(outcome_loss.detach().cpu())

    devices = {loss.device for loss in losses.values()}
    if len(devices) != 1:
        raise ValueError("all objective terms must share a device")
    total = (
        losses["factual_miss"]
        + losses["factual_no_miss"]
        + losses["outcome_complete"]
    )
    total.backward()

    parameters = list(decoder.parameters())
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("every decoder parameter must receive a gradient")
    if any(
        not torch.isfinite(parameter.grad).all()
        for parameter in parameters
    ):
        raise FloatingPointError("decoder gradients must be finite")
    optimizer.step()

    logs["total"] = float(total.detach().cpu())
    logs["decoder_forward_calls_per_update"] = (
        DECODER_FORWARD_CALLS_PER_UPDATE
    )
    logs["decoder_states_per_update"] = DECODER_STATES_PER_UPDATE
    logs["backward_calls"] = 1
    logs["optimizer_steps"] = 1
    return logs


__all__ = [
    "DECODER_FORWARD_CALLS_PER_UPDATE",
    "DECODER_STATES_PER_UPDATE",
    "OUTCOME_ENDPOINT_STATES_PER_UPDATE",
    "OUTCOME_OPTIMIZER_PAIR_KINDS",
    "OUTCOME_PAIR_BATCH_SIZE",
    "outcome_complete_train_step",
]
