"""One-update PFCR training with a single fused decoder forward."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import torch
from torch import Tensor

from ..phase_resolved_relation_decoder import (
    CURELitePhaseResolvedRelationDecoder,
)
from ..phase_resolved_relation_training import (
    phase_resolved_worst_endpoint_loss,
)
from .step import BRANCHES, BranchBatch


def _validate_optimizer(
    decoder: CURELitePhaseResolvedRelationDecoder,
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.nn.Parameter, ...]:
    parameters = tuple(decoder.parameters())
    parameter_ids = {id(parameter) for parameter in parameters}
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )
    optimizer_ids = tuple(id(parameter) for parameter in optimizer_parameters)
    if (
        not parameters
        or any(not parameter.requires_grad for parameter in parameters)
        or len(parameter_ids) != len(parameters)
        or not optimizer_parameters
        or len(optimizer_ids) != len(set(optimizer_ids))
        or set(optimizer_ids) != parameter_ids
    ):
        raise ValueError(
            "optimizer must contain every PFCR parameter exactly once"
        )
    return parameters


def _validated_batches(
    batches: Mapping[str, BranchBatch],
) -> tuple[tuple[str, BranchBatch], ...]:
    if not isinstance(batches, Mapping):
        raise TypeError("batches must be a mapping")
    if set(batches) != set(BRANCHES):
        raise ValueError(
            "PFCR real training requires factual_miss, "
            "factual_no_miss and synthetic on every update"
        )
    normalized: list[tuple[str, BranchBatch]] = []
    common_feature: tuple[int, int, int] | None = None
    common_output: tuple[int, int, int] | None = None
    common_device: torch.device | None = None
    for branch in BRANCHES:
        batch = batches[branch]
        if not isinstance(batch, BranchBatch):
            raise TypeError(f"{branch} batch must be BranchBatch")
        batch = batch.batched()
        batch.validate(expected_branch=branch)
        feature_shape = tuple(int(value) for value in batch.feature.shape[1:])
        output_shape = tuple(
            int(value) for value in batch.occupancy.shape[1:]
        )
        if common_feature is None:
            common_feature = feature_shape
            common_output = output_shape
            common_device = batch.feature.device
        elif (
            feature_shape != common_feature
            or output_shape != common_output
            or batch.feature.device != common_device
        ):
            raise ValueError(
                "all PFCR branches must share feature/output shapes and device"
            )
        normalized.append((branch, batch))
    return tuple(normalized)


def phase_resolved_real_train_step(
    decoder: CURELitePhaseResolvedRelationDecoder,
    optimizer: torch.optim.Optimizer,
    batches: Mapping[str, BranchBatch],
    *,
    logit_margin: float,
    audit: bool = True,
) -> dict[str, float | int]:
    """Run all three branches through PFCR once, then make one update.

    Branch losses remain independent statewise endpoint risks and are combined
    with the fixed ``1:1:1`` weights.  Concatenating the states changes only
    execution efficiency; it does not introduce cross-branch pairs.
    """

    if not isinstance(
        decoder,
        CURELitePhaseResolvedRelationDecoder,
    ):
        raise TypeError("decoder must be the PFCR decoder")
    if (
        isinstance(logit_margin, bool)
        or not isinstance(logit_margin, float)
        or not isfinite(logit_margin)
        or logit_margin <= 0.0
    ):
        raise ValueError("logit_margin must be finite and positive")
    if not isinstance(audit, bool):
        raise TypeError("audit must be bool")
    parameters = _validate_optimizer(decoder, optimizer)
    normalized = _validated_batches(batches)
    features = torch.cat(
        [batch.feature.detach() for _, batch in normalized],
        dim=0,
    )
    occupancy = torch.cat(
        [batch.occupancy for _, batch in normalized],
        dim=0,
    )
    target_float = torch.cat(
        [batch.target for _, batch in normalized],
        dim=0,
    )
    valid_mask = torch.cat(
        [batch.valid_mask for _, batch in normalized],
        dim=0,
    )
    if torch.any(
        (target_float != 0.0) & (target_float != 1.0)
    ):
        raise ValueError("PFCR target must be binary before bool conversion")
    target = target_float.to(torch.bool)

    decoder.train()
    optimizer.zero_grad(set_to_none=True)
    logits = (
        decoder(features, occupancy)
        if audit
        else decoder.forward_training_logits(features, occupancy)
    )
    if tuple(logits.shape) != tuple(occupancy.shape):
        raise RuntimeError("PFCR logits and occupancy shapes differ")
    offset = 0
    branch_losses: list[Tensor] = []
    logs: dict[str, float | int] = {
        "decoder_forward_calls": 1,
    }
    for branch, batch in normalized:
        count = int(batch.feature.shape[0])
        section = slice(offset, offset + count)
        fields = phase_resolved_worst_endpoint_loss(
            logits[section],
            target[section],
            valid_mask[section],
            occupancy[section],
            logit_margin=logit_margin,
            audit=audit,
        )
        branch_losses.append(fields.loss)
        logs[f"{branch}/active"] = 1
        logs[f"{branch}/states"] = count
        if audit:
            logs[f"{branch}/loss"] = float(
                fields.loss.detach().cpu()
            )
            if bool(fields.positive_state_mask.any()):
                logs[f"{branch}/positive_min_logit"] = float(
                    fields.positive_min_logit[
                        fields.positive_state_mask
                    ].min().detach().cpu()
                )
            if bool(fields.negative_state_mask.any()):
                logs[f"{branch}/negative_max_logit"] = float(
                    fields.negative_max_logit[
                        fields.negative_state_mask
                    ].max().detach().cpu()
                )
        offset += count
    if offset != int(logits.shape[0]):
        raise AssertionError("PFCR branch slicing did not cover the batch")
    total = torch.stack(branch_losses).sum()
    if audit and not bool(torch.isfinite(total)):
        raise FloatingPointError("PFCR total loss is not finite")
    total.backward()
    squared_grad_norm = torch.zeros(
        (),
        device=total.device,
        dtype=torch.float32,
    )
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("a PFCR parameter received no gradient")
        if audit and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("PFCR gradient is not finite")
        squared_grad_norm = (
            squared_grad_norm
            + parameter.grad.detach().float().square().sum()
        )
    optimizer.step()
    if audit and any(
        not bool(torch.isfinite(parameter).all())
        for parameter in parameters
    ):
        raise FloatingPointError("PFCR parameter became non-finite")
    gradient_l2_norm = torch.sqrt(squared_grad_norm)
    if audit:
        logs["gradient_l2_norm"] = float(
            gradient_l2_norm.detach().cpu()
        )
        logs["total"] = float(total.detach().cpu())
    else:
        values = torch.stack(
            (*branch_losses, total, gradient_l2_norm)
        ).detach().cpu().tolist()
        for (branch, _), value in zip(
            normalized,
            values[: len(normalized)],
            strict=True,
        ):
            logs[f"{branch}/loss"] = float(value)
        logs["total"] = float(values[-2])
        logs["gradient_l2_norm"] = float(values[-1])
    logs["total_states"] = int(logits.shape[0])
    return logs


__all__ = ["phase_resolved_real_train_step"]
