"""One-step, three-branch optimization for CURE-Lite v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from ..config import TrainingConfig
from ..types import BranchSupervision


BRANCHES = ("factual_miss", "factual_no_miss", "synthetic")


@dataclass(frozen=True)
class BranchBatch:
    feature: Tensor
    occupancy: Tensor
    target: Tensor
    valid_mask: Tensor

    @classmethod
    def from_supervision(
        cls,
        feature: Tensor,
        supervision: BranchSupervision,
    ) -> "BranchBatch":
        supervision.validate()
        return cls(
            feature=feature,
            occupancy=supervision.occupancy,
            target=supervision.target,
            valid_mask=supervision.valid_mask,
        )

    def batched(self) -> "BranchBatch":
        if self.occupancy.ndim == 3:
            return BranchBatch(
                feature=self.feature,
                occupancy=self.occupancy.unsqueeze(0),
                target=self.target.unsqueeze(0),
                valid_mask=self.valid_mask.unsqueeze(0),
            )
        return self

    def validate(self, *, expected_branch: str | None = None) -> None:
        if not all(
            isinstance(value, Tensor)
            for value in (self.feature, self.occupancy, self.target, self.valid_mask)
        ):
            raise TypeError("all branch-batch fields must be tensors")
        if self.feature.ndim != 4:
            raise ValueError("feature must have shape [B,C,h,w]")
        if self.occupancy.ndim != 4 or self.occupancy.shape[1] != 1:
            raise ValueError("supervision tensors must have shape [B,1,H,W]")
        if not (self.occupancy.shape == self.target.shape == self.valid_mask.shape):
            raise ValueError("branch supervision tensor shapes must match")
        if self.feature.shape[0] != self.occupancy.shape[0] or self.feature.shape[0] < 1:
            raise ValueError("branch batch sizes must agree and be non-empty")
        if not self.feature.is_floating_point():
            raise TypeError("feature must be floating point")
        if self.occupancy.dtype != torch.bool or self.valid_mask.dtype != torch.bool:
            raise TypeError("occupancy and valid_mask must be bool")
        if self.target.dtype != torch.float32:
            raise TypeError("target must be float32")
        if not (
            self.feature.device
            == self.occupancy.device
            == self.target.device
            == self.valid_mask.device
        ):
            raise ValueError("all branch tensors must share a device")
        if not torch.isfinite(self.feature).all() or not torch.isfinite(self.target).all():
            raise ValueError("feature and target must be finite")
        if torch.any((self.target != 0.0) & (self.target != 1.0)):
            raise ValueError("target must be binary")
        if torch.any(self.target.to(torch.bool) & ~self.valid_mask):
            raise ValueError("positive target lies outside valid_mask")
        if torch.any(self.valid_mask & self.occupancy):
            raise ValueError("valid_mask overlaps occupancy")
        if expected_branch is not None:
            if expected_branch not in BRANCHES:
                raise ValueError(f"unknown expected branch {expected_branch!r}")
            positive_by_state = self.target.to(torch.bool).flatten(1).any(dim=1)
            valid_by_state = self.valid_mask.flatten(1).any(dim=1)
            if expected_branch == "factual_no_miss":
                if torch.any(positive_by_state):
                    raise ValueError("factual_no_miss batch must have an empty target")
                if not torch.all(valid_by_state):
                    raise ValueError(
                        "every factual_no_miss state must contain valid negative "
                        "supervision"
                    )
            elif not torch.all(positive_by_state):
                raise ValueError(
                    f"every {expected_branch} state must contain positive supervision"
                )


def _resolve_training_config(
    config: TrainingConfig | None,
    lambda_no_miss: float | None,
    lambda_synthetic: float | None,
) -> TrainingConfig:
    if config is not None and (lambda_no_miss is not None or lambda_synthetic is not None):
        raise ValueError("do not override fields of an explicit TrainingConfig")
    return config or TrainingConfig(
        lambda_no_miss=1.0 if lambda_no_miss is None else lambda_no_miss,
        lambda_synthetic=1.0 if lambda_synthetic is None else lambda_synthetic,
    )


def combine_branch_means(
    branch_losses: Mapping[str, Tensor],
    *,
    config: TrainingConfig | None = None,
    lambda_no_miss: float | None = None,
    lambda_synthetic: float | None = None,
) -> Tensor:
    """Combine already-independent branch means according to the frozen loss."""

    unknown = set(branch_losses) - set(BRANCHES)
    if unknown:
        raise ValueError(f"unknown branches: {sorted(unknown)}")
    if not branch_losses:
        raise ValueError("at least one non-empty branch is required")
    resolved = _resolve_training_config(config, lambda_no_miss, lambda_synthetic)
    first = next(iter(branch_losses.values()))
    if not isinstance(first, Tensor) or first.ndim != 0:
        raise ValueError("branch losses must be scalar tensors")
    total = first * 0.0
    weights = {
        "factual_miss": 1.0,
        "factual_no_miss": resolved.lambda_no_miss,
        "synthetic": resolved.lambda_synthetic,
    }
    for branch, loss in branch_losses.items():
        if not isinstance(loss, Tensor) or loss.ndim != 0:
            raise ValueError("branch losses must be scalar tensors")
        if loss.device != first.device:
            raise ValueError("branch losses must share a device")
        total = total + weights[branch] * loss
    return total


def _validate_optimizer_scope(decoder: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    decoder_parameters = list(decoder.parameters())
    if not decoder_parameters:
        raise ValueError("decoder contains no parameters")
    frozen = [parameter for parameter in decoder_parameters if not parameter.requires_grad]
    if frozen:
        raise ValueError("every CURE-Lite decoder parameter must require gradients")
    decoder_ids = {id(parameter) for parameter in decoder_parameters}
    if len(decoder_ids) != len(decoder_parameters):
        raise ValueError("decoder exposes duplicate parameter references")
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    ]
    if not optimizer_parameters:
        raise ValueError("optimizer contains no parameters")
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise ValueError("optimizer contains duplicate decoder parameters")
    optimizer_id_set = set(optimizer_ids)
    foreign = optimizer_id_set - decoder_ids
    if foreign:
        raise ValueError("CURE-Lite optimizer may contain only decoder parameters")
    missing = decoder_ids - optimizer_id_set
    if missing:
        raise ValueError("optimizer must contain every decoder parameter exactly once")


def multi_branch_train_step(
    decoder: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Mapping[str, BranchBatch],
    *,
    config: TrainingConfig | None = None,
    lambda_no_miss: float | None = None,
    lambda_synthetic: float | None = None,
) -> dict[str, float | int]:
    """Run all present branches, then exactly one backward and optimizer step."""

    unknown = set(batches) - set(BRANCHES)
    if unknown:
        raise ValueError(f"unknown branches: {sorted(unknown)}")
    if not batches:
        raise ValueError("at least one non-empty branch is required")
    resolved = _resolve_training_config(config, lambda_no_miss, lambda_synthetic)
    _validate_optimizer_scope(decoder, optimizer)

    decoder.train()
    optimizer.zero_grad(set_to_none=True)
    branch_losses: dict[str, Tensor] = {}
    logs: dict[str, float | int] = {}
    for branch in BRANCHES:
        batch = batches.get(branch)
        if batch is None:
            logs[f"{branch}/active"] = 0
            logs[f"{branch}/states"] = 0
            logs[f"{branch}/loss"] = 0.0
            continue
        if not isinstance(batch, BranchBatch):
            raise TypeError(f"{branch} batch must be BranchBatch")
        batch = batch.batched()
        batch.validate(expected_branch=branch)
        logits = decoder(batch.feature.detach(), batch.occupancy)
        result = criterion(logits, batch.target, batch.valid_mask)
        if not isinstance(result, Mapping) or "total" not in result:
            raise TypeError("criterion must return a mapping containing 'total'")
        loss = result["total"]
        if not isinstance(loss, Tensor) or loss.ndim != 0:
            raise ValueError("criterion total must be a scalar tensor")
        branch_losses[branch] = loss
        logs[f"{branch}/active"] = 1
        logs[f"{branch}/states"] = int(batch.feature.shape[0])
        logs[f"{branch}/loss"] = float(loss.detach().cpu())

    total = combine_branch_means(branch_losses, config=resolved)
    total.backward()
    optimizer.step()
    logs["total"] = float(total.detach().cpu())
    return logs


__all__ = [
    "BRANCHES",
    "BranchBatch",
    "combine_branch_means",
    "multi_branch_train_step",
]
