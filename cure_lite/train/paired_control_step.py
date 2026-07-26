"""Production train step for frozen CURE-Lite matched controls.

This module is parallel to, and deliberately does not modify, the proposed
``paired_train_step``.  Every control keeps the frozen 4 + 4 + 2-pair schedule,
uses one 2B endpoint forward, applies branch weights 1:1:1, and performs one
backward followed by one optimizer step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..losses import CURELiteLoss
from ..paired_control_inputs import (
    DCTCoordinateBasis,
    capacity_active_dct_feature_like,
    feature_only_zero_occupancy,
    nominal_zero_feature_like,
)
from ..paired_control_losses import (
    after_only_absolute_synthetic_loss,
    build_after_only_synthetic_supervision,
    build_geometry_matched_endpoint_supervision,
    geometry_matched_independent_endpoint_loss,
    minus_detached_paired_difference_loss,
    plus_detached_paired_difference_loss,
)
from ..paired_losses import PairedDifferenceLoss
from ..paired_types import PairBatch
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


CONTROL_KINDS = (
    "independent_endpoint",
    "after_only",
    "zero_feature",
    "coordinate_basis",
    "feature_only",
    "target_permutation",
    "plus_detach",
    "minus_detach",
)


@dataclass(frozen=True)
class _PreparedControl:
    feature: Tensor
    occupancy_plus: Tensor
    occupancy_minus: Tensor
    label_increment: Tensor
    basis_fingerprint: str | None


def _require_control_tensor(
    value: Tensor | None,
    *,
    name: str,
    reference: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != reference.shape:
        raise ValueError(f"{name} must match the paired evaluation shape")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.device != reference.device:
        raise ValueError(f"{name} must share the paired evaluation device")
    return value


def _validate_response_label(
    label: Tensor,
    *,
    image_valid_mask: Tensor,
    name: str,
) -> None:
    if not torch.isfinite(label).all():
        raise ValueError(f"{name} must be finite")
    if torch.any((label != 0.0) & (label != 1.0)):
        raise ValueError(f"{name} must be binary")
    positive = label.to(dtype=torch.bool)
    if torch.any(positive & ~image_valid_mask):
        raise ValueError(f"{name} extends outside image_valid_mask")
    if not torch.all(positive.flatten(1).any(dim=1)):
        raise ValueError(f"every {name} state requires a positive response")
    zero = image_valid_mask & ~positive
    if not torch.all(zero.flatten(1).any(dim=1)):
        raise ValueError(f"every {name} state requires a zero-response domain")


def _coordinate_feature(
    feature: Tensor,
    basis: DCTCoordinateBasis | None,
) -> tuple[Tensor, DCTCoordinateBasis]:
    if basis is None:
        return capacity_active_dct_feature_like(feature)
    if not isinstance(basis, DCTCoordinateBasis):
        raise TypeError("coordinate_basis must be DCTCoordinateBasis")
    expected = tuple(int(value) for value in feature.shape[1:])
    if tuple(basis.tensor.shape[1:]) != expected:
        raise ValueError("coordinate_basis shape does not match pair feature")
    if basis.tensor.dtype != feature.dtype:
        raise TypeError("coordinate_basis dtype does not match pair feature")
    return (
        basis.expand(int(feature.shape[0]), device=feature.device),
        basis,
    )


def _prepare_control(
    *,
    control_kind: str,
    pair_batch: PairBatch,
    gt_union: Tensor | None,
    completion_plus: Tensor | None,
    completion_minus: Tensor | None,
    permuted_label_increment: Tensor | None,
    coordinate_basis: DCTCoordinateBasis | None,
) -> _PreparedControl:
    if not isinstance(control_kind, str) or control_kind not in CONTROL_KINDS:
        raise ValueError(f"control_kind must be one of {CONTROL_KINDS}")
    reference = pair_batch.image_valid_mask
    feature = pair_batch.feature
    occupancy_plus = pair_batch.occupancy_plus
    occupancy_minus = pair_batch.occupancy_minus
    label_increment = pair_batch.label_increment
    basis_fingerprint: str | None = None

    if control_kind == "independent_endpoint":
        union = _require_control_tensor(
            gt_union,
            name="gt_union",
            reference=reference,
            dtype=torch.bool,
        )
        plus = _require_control_tensor(
            completion_plus,
            name="completion_plus",
            reference=reference,
            dtype=torch.bool,
        )
        minus = _require_control_tensor(
            completion_minus,
            name="completion_minus",
            reference=reference,
            dtype=torch.bool,
        )
        # Execute every pixel-semantic check before decoder mode or gradients
        # can be changed.
        build_geometry_matched_endpoint_supervision(
            plus,
            occupancy_plus,
            union,
            reference,
        )
        build_geometry_matched_endpoint_supervision(
            minus,
            occupancy_minus,
            union,
            reference,
        )
    elif control_kind == "after_only":
        union = _require_control_tensor(
            gt_union,
            name="gt_union",
            reference=reference,
            dtype=torch.bool,
        )
        build_after_only_synthetic_supervision(
            label_increment.to(dtype=torch.bool),
            occupancy_minus,
            union,
            reference,
        )
    elif control_kind == "zero_feature":
        feature = nominal_zero_feature_like(feature)
    elif control_kind == "coordinate_basis":
        feature, resolved_basis = _coordinate_feature(feature, coordinate_basis)
        basis_fingerprint = resolved_basis.basis_fingerprint
    elif control_kind == "feature_only":
        occupancy_plus, occupancy_minus = feature_only_zero_occupancy(
            occupancy_plus,
            occupancy_minus,
        )
    elif control_kind == "target_permutation":
        label_increment = _require_control_tensor(
            permuted_label_increment,
            name="permuted_label_increment",
            reference=reference,
            dtype=torch.float32,
        )
        _validate_response_label(
            label_increment,
            image_valid_mask=reference,
            name="permuted_label_increment",
        )

    if control_kind != "target_permutation" and permuted_label_increment is not None:
        raise ValueError(
            "permuted_label_increment is valid only for target_permutation"
        )
    if control_kind != "coordinate_basis" and coordinate_basis is not None:
        raise ValueError("coordinate_basis is valid only for coordinate_basis")

    # Paired controls other than endpoint absolute ERM and after-only use the
    # difference criterion and therefore require both frozen response strata.
    if control_kind not in ("independent_endpoint", "after_only"):
        _validate_response_label(
            label_increment,
            image_valid_mask=reference,
            name="label_increment",
        )
    return _PreparedControl(
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        label_increment=label_increment,
        basis_fingerprint=basis_fingerprint,
    )


def _control_loss(
    *,
    control_kind: str,
    logits_plus: Tensor,
    logits_minus: Tensor,
    pair_batch: PairBatch,
    prepared: _PreparedControl,
    absolute_criterion: CURELiteLoss,
    paired_criterion: PairedDifferenceLoss,
    gt_union: Tensor | None,
    completion_plus: Tensor | None,
    completion_minus: Tensor | None,
) -> Tensor:
    if control_kind == "independent_endpoint":
        result = geometry_matched_independent_endpoint_loss(
            logits_plus,
            logits_minus,
            completion_plus=completion_plus,
            completion_minus=completion_minus,
            occupancy_plus=prepared.occupancy_plus,
            occupancy_minus=prepared.occupancy_minus,
            gt_union=gt_union,
            image_valid_mask=pair_batch.image_valid_mask,
            criterion=absolute_criterion,
        )
    elif control_kind == "after_only":
        result = after_only_absolute_synthetic_loss(
            logits_plus,
            logits_minus,
            selected_completion=pair_batch.label_increment.to(dtype=torch.bool),
            occupancy_minus=prepared.occupancy_minus,
            gt_union=gt_union,
            image_valid_mask=pair_batch.image_valid_mask,
            criterion=absolute_criterion,
        )
    elif control_kind == "plus_detach":
        result = plus_detached_paired_difference_loss(
            logits_plus,
            logits_minus,
            prepared.label_increment,
            pair_batch.image_valid_mask,
            criterion=paired_criterion,
        )
    elif control_kind == "minus_detach":
        result = minus_detached_paired_difference_loss(
            logits_plus,
            logits_minus,
            prepared.label_increment,
            pair_batch.image_valid_mask,
            criterion=paired_criterion,
        )
    else:
        result = paired_criterion(
            logits_plus,
            logits_minus,
            prepared.label_increment,
            pair_batch.image_valid_mask,
        )
    return _criterion_total(result, name=f"{control_kind} criterion")


def paired_control_train_step(
    decoder: nn.Module,
    absolute_criterion: CURELiteLoss,
    paired_criterion: PairedDifferenceLoss,
    optimizer: torch.optim.Optimizer,
    factual_batches: Mapping[str, BranchBatch],
    pair_batch: PairBatch,
    *,
    control_kind: str,
    gt_union: Tensor | None = None,
    completion_plus: Tensor | None = None,
    completion_minus: Tensor | None = None,
    permuted_label_increment: Tensor | None = None,
    coordinate_basis: DCTCoordinateBasis | None = None,
) -> dict[str, float | int | str]:
    """Run one fixed-budget matched-control optimizer update.

    Shared initialization, optimizer hyperparameters, and schedule identity are
    experiment-runner responsibilities.  This function enforces the per-update
    computational contract and records the selected control.
    """

    if not isinstance(decoder, nn.Module):
        raise TypeError("decoder must be an nn.Module")
    if not isinstance(absolute_criterion, CURELiteLoss):
        raise TypeError("absolute_criterion must be CURELiteLoss")
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

    # The shared and control-specific preflights both finish before train(),
    # zero_grad(), or any decoder forward.
    normalized_factual = _preflight_training_batches(
        decoder,
        optimizer,
        factual_batches,
        pair_batch,
    )
    prepared = _prepare_control(
        control_kind=control_kind,
        pair_batch=pair_batch,
        gt_union=gt_union,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        permuted_label_increment=permuted_label_increment,
        coordinate_basis=coordinate_basis,
    )

    decoder.train()
    optimizer.zero_grad(set_to_none=True)
    logs: dict[str, float | int | str] = {
        "control_kind": control_kind,
    }
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
        feature=prepared.feature,
        occupancy_plus=prepared.occupancy_plus,
        occupancy_minus=prepared.occupancy_minus,
    )
    control_loss = _control_loss(
        control_kind=control_kind,
        logits_plus=logits_plus,
        logits_minus=logits_minus,
        pair_batch=pair_batch,
        prepared=prepared,
        absolute_criterion=absolute_criterion,
        paired_criterion=paired_criterion,
        gt_union=gt_union,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
    )
    losses["control"] = control_loss
    logs["control/pairs"] = int(pair_batch.feature.shape[0])
    logs["control/endpoints"] = 2 * int(pair_batch.feature.shape[0])
    logs["control/loss"] = float(control_loss.detach().cpu())
    logs["decoder/states"] = (
        int(normalized_factual["factual_miss"].feature.shape[0])
        + int(normalized_factual["factual_no_miss"].feature.shape[0])
        + 2 * int(pair_batch.feature.shape[0])
    )
    if logs["decoder/states"] != DECODER_STATES_PER_UPDATE:
        raise AssertionError("matched-control state budget drifted")
    if prepared.basis_fingerprint is not None:
        logs["control/basis_fingerprint"] = prepared.basis_fingerprint

    if len({loss.device for loss in losses.values()}) != 1:
        raise ValueError("all objective terms must share a device")
    total = (
        losses["factual_miss"]
        + losses["factual_no_miss"]
        + losses["control"]
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
    logs["control/endpoint_forward_batches"] = 1
    logs["factual_anchor_batch_size"] = FACTUAL_ANCHOR_BATCH_SIZE
    logs["paired_batch_size"] = PAIRED_BATCH_SIZE
    return logs


__all__ = [
    "CONTROL_KINDS",
    "paired_control_train_step",
]
