"""Small orchestration layer around the normative multi-branch train step."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from torch import nn

from ..config import TrainingConfig
from .step import BranchBatch, multi_branch_train_step


class CURELiteTrainEngine:
    """Execute preassembled F+/F0/S branch batches without changing exposure."""

    def __init__(
        self,
        decoder: nn.Module,
        criterion: nn.Module,
        optimizer,
        config: TrainingConfig = TrainingConfig(),
    ) -> None:
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be TrainingConfig")
        self.decoder = decoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

    def step(self, batches: Mapping[str, BranchBatch]) -> dict[str, float | int]:
        return multi_branch_train_step(
            self.decoder,
            self.criterion,
            self.optimizer,
            batches,
            config=self.config,
        )

    def run_epoch(
        self,
        step_batches: Iterable[Mapping[str, BranchBatch]],
    ) -> dict[str, float | int]:
        totals: dict[str, float] = {}
        steps = 0
        for batches in step_batches:
            logs = self.step(batches)
            steps += 1
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        if steps == 0:
            raise ValueError("an epoch must contain at least one optimizer step")
        summary: dict[str, float | int] = {
            key: value / steps for key, value in totals.items()
        }
        summary["steps"] = steps
        return summary


def run_training_epoch(
    decoder: nn.Module,
    criterion: nn.Module,
    optimizer,
    step_batches: Iterable[Mapping[str, BranchBatch]],
    *,
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, float | int]:
    return CURELiteTrainEngine(decoder, criterion, optimizer, config).run_epoch(step_batches)


__all__ = ["CURELiteTrainEngine", "run_training_epoch"]
