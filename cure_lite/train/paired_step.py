"""Additive paired-objective training primitives for CURE-Lite.

The legacy single-state ``multi_branch_train_step`` remains untouched.  This
module binds two occupancy endpoints to one feature and one optimizer update,
while continuing to use ``CURELiteLoss`` for the factual absolute anchors.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..losses import CURELiteLoss
from ..paired_losses import PairedDifferenceLoss
from ..paired_types import PairBatch
from .step import BranchBatch, _validate_optimizer_scope


FACTUAL_ANCHOR_BRANCHES = ("factual_miss", "factual_no_miss")
FACTUAL_ANCHOR_BATCH_SIZE = 4
PAIRED_BATCH_SIZE = 2
DECODER_STATES_PER_UPDATE = 12
NULL_PAIR_KINDS = ("identity_null", "component_null")


def _validate_pair_identities(batch: PairBatch) -> None:
    if len(set(batch.pair_ids)) != len(batch.pair_ids):
        raise ValueError("pair_ids must be unique within one batch")


def _validate_clean_positive_batch(batch: PairBatch) -> None:
    batch.validate()
    _validate_pair_identities(batch)
    if any(kind != "clean_positive" for kind in batch.pair_kinds):
        raise ValueError("the optimizer path accepts only clean_positive pairs")
    increment = batch.label_increment.to(dtype=torch.bool)
    if torch.any(increment & batch.occupancy_minus):
        raise ValueError("label_increment must be writable under occupancy_minus")


def _validate_null_batch(batch: PairBatch) -> None:
    batch.validate()
    _validate_pair_identities(batch)
    if any(kind not in NULL_PAIR_KINDS for kind in batch.pair_kinds):
        raise ValueError("null diagnostics accept only identity_null/component_null")


def _preflight_training_batches(
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    factual_batches: Mapping[str, BranchBatch],
    pair_batch: PairBatch,
) -> dict[str, BranchBatch]:
    """Validate the complete update before changing mode, gradients, or state."""

    _validate_clean_positive_batch(pair_batch)
    if int(pair_batch.feature.shape[0]) != PAIRED_BATCH_SIZE:
        raise ValueError(
            f"paired training requires exactly {PAIRED_BATCH_SIZE} clean pairs"
        )
    if len(set(pair_batch.sample_ids)) != PAIRED_BATCH_SIZE:
        raise ValueError("the two clean pairs must come from distinct source samples")
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
                f"{branch} requires exactly {FACTUAL_ANCHOR_BATCH_SIZE} states"
            )
        normalized[branch] = batch

    all_features = (
        normalized["factual_miss"].feature,
        normalized["factual_no_miss"].feature,
        pair_batch.feature,
    )
    all_occupancies = (
        normalized["factual_miss"].occupancy,
        normalized["factual_no_miss"].occupancy,
        pair_batch.occupancy_plus,
    )
    if len({feature.device for feature in all_features}) != 1:
        raise ValueError("all training features must share a device")
    if len({feature.dtype for feature in all_features}) != 1:
        raise TypeError("all training features must share a dtype")
    if len({tuple(feature.shape[1:]) for feature in all_features}) != 1:
        raise ValueError("all training features must share [C,h,w]")
    if len({tuple(value.shape[1:]) for value in all_occupancies}) != 1:
        raise ValueError("all training state tensors must share [1,H,W]")

    decoder_parameters = tuple(decoder.parameters())
    parameter_devices = {parameter.device for parameter in decoder_parameters}
    parameter_dtypes = {parameter.dtype for parameter in decoder_parameters}
    if len(parameter_devices) != 1 or len(parameter_dtypes) != 1:
        raise ValueError("decoder parameters must share one device and dtype")
    feature_device = all_features[0].device
    feature_dtype = all_features[0].dtype
    if feature_device != next(iter(parameter_devices)):
        raise ValueError("training features and decoder parameters must share a device")
    if feature_dtype != next(iter(parameter_dtypes)):
        raise TypeError("training feature dtype must match decoder parameter dtype")

    expected_channels = getattr(decoder, "feature_channels", None)
    if expected_channels is not None and int(all_features[0].shape[1]) != int(
        expected_channels
    ):
        raise ValueError("training feature channels do not match decoder")
    feature_grid = tuple(int(value) for value in all_features[0].shape[-2:])
    evaluation_grid = tuple(int(value) for value in all_occupancies[0].shape[-2:])
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


def _paired_endpoint_logits(
    decoder: nn.Module,
    *,
    feature: Tensor,
    occupancy_plus: Tensor,
    occupancy_minus: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute all 2B endpoint logits in exactly one decoder call."""

    batch_size = int(feature.shape[0])
    endpoint_feature = torch.cat((feature.detach(), feature.detach()), dim=0)
    endpoint_occupancy = torch.cat((occupancy_plus, occupancy_minus), dim=0)
    endpoint_logits = decoder(endpoint_feature, endpoint_occupancy)
    if not isinstance(endpoint_logits, Tensor):
        raise TypeError("decoder must return one logits tensor")
    expected_shape = (
        2 * batch_size,
        1,
        int(occupancy_plus.shape[-2]),
        int(occupancy_plus.shape[-1]),
    )
    if tuple(endpoint_logits.shape) != expected_shape:
        raise ValueError(
            "decoder returned an invalid paired endpoint shape "
            f"{tuple(endpoint_logits.shape)} != {expected_shape}"
        )
    if not endpoint_logits.is_floating_point():
        raise TypeError("decoder endpoint logits must be floating point")
    if not torch.isfinite(endpoint_logits).all():
        raise ValueError("decoder endpoint logits must be finite")
    return endpoint_logits[:batch_size], endpoint_logits[batch_size:]


def paired_endpoint_logits(
    decoder: nn.Module,
    batch: PairBatch,
) -> tuple[Tensor, Tensor]:
    """Public one-call endpoint forward for a clean positive pair batch."""

    if not isinstance(decoder, nn.Module):
        raise TypeError("decoder must be an nn.Module")
    if not isinstance(batch, PairBatch):
        raise TypeError("batch must be PairBatch")
    _validate_clean_positive_batch(batch)
    return _paired_endpoint_logits(
        decoder,
        feature=batch.feature,
        occupancy_plus=batch.occupancy_plus,
        occupancy_minus=batch.occupancy_minus,
    )


def _criterion_total(
    result: Mapping[str, Tensor],
    *,
    name: str,
) -> Tensor:
    if not isinstance(result, Mapping) or "total" not in result:
        raise TypeError(f"{name} must return a mapping containing 'total'")
    total = result["total"]
    if not isinstance(total, Tensor) or total.ndim != 0:
        raise ValueError(f"{name} total must be a scalar tensor")
    if not torch.isfinite(total):
        raise ValueError(f"{name} total must be finite")
    return total


def paired_train_step(
    decoder: nn.Module,
    absolute_criterion: CURELiteLoss,
    paired_criterion: PairedDifferenceLoss,
    optimizer: torch.optim.Optimizer,
    factual_batches: Mapping[str, BranchBatch],
    pair_batch: PairBatch,
) -> dict[str, float | int]:
    """Run two factual anchors and one coupled pair loss in one update.

    The fixed objective is

    ``L_factual_miss + L_factual_no_miss + L_paired_difference``.

    Null pair kinds are rejected and therefore cannot enter this optimizer
    path.
    """

    if not isinstance(decoder, nn.Module):
        raise TypeError("decoder must be an nn.Module")
    if not isinstance(absolute_criterion, CURELiteLoss):
        raise TypeError("absolute_criterion must be the existing CURELiteLoss")
    if not isinstance(paired_criterion, PairedDifferenceLoss):
        raise TypeError("paired_criterion must be PairedDifferenceLoss")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(factual_batches, Mapping):
        raise TypeError("factual_batches must be a mapping")
    if set(factual_batches) != set(FACTUAL_ANCHOR_BRANCHES):
        raise ValueError(
            "factual_batches must contain exactly factual_miss and "
            "factual_no_miss"
        )
    if not isinstance(pair_batch, PairBatch):
        raise TypeError("pair_batch must be PairBatch")
    normalized_factual = _preflight_training_batches(
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

    # The batch was validated above.  Call the internal forward directly so
    # projection visibility is not recomputed a second time in every update.
    logits_plus, logits_minus = _paired_endpoint_logits(
        decoder,
        feature=pair_batch.feature,
        occupancy_plus=pair_batch.occupancy_plus,
        occupancy_minus=pair_batch.occupancy_minus,
    )
    paired_result = paired_criterion(
        logits_plus,
        logits_minus,
        pair_batch.label_increment,
        pair_batch.image_valid_mask,
    )
    paired_loss = _criterion_total(paired_result, name="paired criterion")
    losses["paired_difference"] = paired_loss
    logs["paired/pairs"] = int(pair_batch.feature.shape[0])
    logs["paired/endpoints"] = 2 * int(pair_batch.feature.shape[0])
    logs["paired/loss"] = float(paired_loss.detach().cpu())

    devices = {loss.device for loss in losses.values()}
    if len(devices) != 1:
        raise ValueError("all objective terms must share a device")
    total = (
        losses["factual_miss"]
        + losses["factual_no_miss"]
        + losses["paired_difference"]
    )
    total.backward()

    parameters = list(decoder.parameters())
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("every decoder parameter must receive a gradient")
    if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise FloatingPointError("decoder gradients must be finite")
    optimizer.step()

    logs["total"] = float(total.detach().cpu())
    logs["optimizer_steps"] = 1
    return logs


def diagnose_null_pairs(
    decoder: nn.Module,
    batch: PairBatch,
) -> dict[str, Tensor]:
    """Evaluate null-pair score differences without creating an optimizer path."""

    if not isinstance(decoder, nn.Module):
        raise TypeError("decoder must be an nn.Module")
    if not isinstance(batch, PairBatch):
        raise TypeError("batch must be PairBatch")
    _validate_null_batch(batch)
    with torch.no_grad():
        logits_plus, logits_minus = _paired_endpoint_logits(
            decoder,
            feature=batch.feature,
            occupancy_plus=batch.occupancy_plus,
            occupancy_minus=batch.occupancy_minus,
        )
        delta = torch.sigmoid(logits_minus) - torch.sigmoid(logits_plus)
        valid = batch.image_valid_mask
        per_pair_mean_abs = torch.stack(
            [delta[index][valid[index]].abs().mean() for index in range(delta.shape[0])]
        )
        per_pair_max_abs = torch.stack(
            [delta[index][valid[index]].abs().max() for index in range(delta.shape[0])]
        )
        per_pair_rms = torch.stack(
            [
                torch.sqrt((delta[index][valid[index]] ** 2).mean())
                for index in range(delta.shape[0])
            ]
        )
    return {
        "pair_count": torch.tensor(delta.shape[0], device=delta.device),
        "per_pair_mean_abs_delta": per_pair_mean_abs,
        "per_pair_max_abs_delta": per_pair_max_abs,
        "per_pair_rms_delta": per_pair_rms,
    }


__all__ = [
    "DECODER_STATES_PER_UPDATE",
    "FACTUAL_ANCHOR_BRANCHES",
    "FACTUAL_ANCHOR_BATCH_SIZE",
    "NULL_PAIR_KINDS",
    "PAIRED_BATCH_SIZE",
    "diagnose_null_pairs",
    "paired_endpoint_logits",
    "paired_train_step",
]
