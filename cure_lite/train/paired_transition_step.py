"""Training step for the additive CURE-Lite anchored-transition route.

Paired-v1 remains frozen in :mod:`cure_lite.train.paired_step`.  This module
keeps the same factual anchors, decoder topology, three-forward budget, and
single optimizer update, but replaces the under-identified pair branch with a
pair-local plus-state anchor followed by the coupled transition objective.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..losses import CURELiteLoss
from ..paired_transition_losses import AnchoredTransitionLoss
from ..paired_transition_types import AnchoredPairBatch
from .paired_step import (
    DECODER_STATES_PER_UPDATE,
    FACTUAL_ANCHOR_BATCH_SIZE,
    FACTUAL_ANCHOR_BRANCHES,
    PAIRED_BATCH_SIZE,
    _criterion_total,
    _paired_endpoint_logits,
    _preflight_training_batches,
)
from .step import BranchBatch


def _validate_anchored_batch(batch: AnchoredPairBatch) -> None:
    if not isinstance(batch, AnchoredPairBatch):
        raise TypeError("pair_batch must be AnchoredPairBatch")
    batch.validate()
    if int(batch.feature.shape[0]) != PAIRED_BATCH_SIZE:
        raise ValueError(
            f"anchored training requires exactly {PAIRED_BATCH_SIZE} clean pairs"
        )
    if len(set(batch.pair_ids)) != PAIRED_BATCH_SIZE:
        raise ValueError("pair_ids must be unique within one anchored batch")
    if len(set(batch.sample_ids)) != PAIRED_BATCH_SIZE:
        raise ValueError("the two anchored pairs must come from distinct samples")


def _preflight_anchored_training(
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    factual_batches: Mapping[str, BranchBatch],
    pair_batch: AnchoredPairBatch,
) -> dict[str, BranchBatch]:
    """Validate one complete update before mode, gradients, or weights change."""

    _validate_anchored_batch(pair_batch)
    return _preflight_training_batches(
        decoder,
        optimizer,
        factual_batches,
        pair_batch.pair_batch,
    )


def anchored_transition_train_step(
    decoder: nn.Module,
    absolute_criterion: CURELiteLoss,
    transition_criterion: AnchoredTransitionLoss,
    optimizer: torch.optim.Optimizer,
    factual_batches: Mapping[str, BranchBatch],
    pair_batch: AnchoredPairBatch,
) -> dict[str, float | int]:
    """Run the fixed factual anchors and one anchored transition update.

    The update objective is

    ``L_factual_miss + L_factual_no_miss + L_anchored_transition``.

    ``L_anchored_transition`` already contains the fixed equal weighting of
    the pair-local plus anchor and the coupled endpoint transition.  Both
    endpoint logits are produced by one ``2B`` decoder call and remain in the
    same backward graph.
    """

    if not isinstance(decoder, nn.Module):
        raise TypeError("decoder must be an nn.Module")
    if not isinstance(absolute_criterion, CURELiteLoss):
        raise TypeError("absolute_criterion must be CURELiteLoss")
    if not isinstance(transition_criterion, AnchoredTransitionLoss):
        raise TypeError(
            "transition_criterion must be AnchoredTransitionLoss"
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
    normalized_factual = _preflight_anchored_training(
        decoder,
        optimizer,
        factual_batches,
        pair_batch,
    )

    decoder.train()
    optimizer.zero_grad(set_to_none=True)
    logs: dict[str, float | int] = {}
    losses: dict[str, Tensor] = {}

    for branch in FACTUAL_ANCHOR_BRANCHES:
        batch = normalized_factual[branch]
        logits = decoder(batch.feature.detach(), batch.occupancy)
        result = absolute_criterion(logits, batch.target, batch.valid_mask)
        loss = _criterion_total(result, name=f"{branch} criterion")
        losses[branch] = loss
        logs[f"{branch}/states"] = int(batch.feature.shape[0])
        logs[f"{branch}/loss"] = float(loss.detach().cpu())

    logits_plus, logits_minus = _paired_endpoint_logits(
        decoder,
        feature=pair_batch.feature,
        occupancy_plus=pair_batch.occupancy_plus,
        occupancy_minus=pair_batch.occupancy_minus,
    )
    transition_result = transition_criterion(
        logits_plus,
        logits_minus,
        pair_batch.completion_plus,
        pair_batch.occupancy_plus,
        pair_batch.gt_union,
        pair_batch.label_increment,
        pair_batch.image_valid_mask,
    )
    transition_loss = _criterion_total(
        transition_result,
        name="anchored transition criterion",
    )
    losses["anchored_transition"] = transition_loss
    logs["paired/pairs"] = int(pair_batch.feature.shape[0])
    logs["paired/endpoints"] = 2 * int(pair_batch.feature.shape[0])
    logs["paired/plus_anchor_loss"] = float(
        transition_result["plus_anchor_loss"].detach().cpu()
    )
    logs["paired/transition_loss"] = float(
        transition_result["transition_loss"].detach().cpu()
    )
    logs["paired/loss"] = float(transition_loss.detach().cpu())

    devices = {loss.device for loss in losses.values()}
    if len(devices) != 1:
        raise ValueError("all objective terms must share a device")
    total = (
        losses["factual_miss"]
        + losses["factual_no_miss"]
        + losses["anchored_transition"]
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
    logs["decoder_forward_calls_per_update"] = 3
    logs["decoder_states_per_update"] = DECODER_STATES_PER_UPDATE
    logs["optimizer_steps"] = 1
    return logs


__all__ = [
    "FACTUAL_ANCHOR_BATCH_SIZE",
    "PAIRED_BATCH_SIZE",
    "anchored_transition_train_step",
]
