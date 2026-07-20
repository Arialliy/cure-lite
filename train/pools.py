"""Deterministic fixed-count F+/F0/S pools for normative branch balancing."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterator, Mapping

import torch
from torch import Tensor

from ..sampling import stable_hash
from ..types import BranchSupervision
from .step import BRANCHES, BranchBatch


@dataclass(frozen=True)
class StateExample:
    sample_id: str
    feature: Tensor
    supervision: BranchSupervision

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if self.feature.ndim != 4 or self.feature.shape[0] != 1:
            raise ValueError("one state example requires feature [1,C,h,w]")
        if self.feature.requires_grad:
            raise ValueError("cached base feature must be detached")
        self.supervision.validate()


@dataclass(frozen=True)
class BranchPools:
    factual_miss: tuple[StateExample, ...] = ()
    factual_no_miss: tuple[StateExample, ...] = ()
    synthetic: tuple[StateExample, ...] = ()

    def __post_init__(self) -> None:
        for branch in BRANCHES:
            for item in getattr(self, branch):
                if item.supervision.branch != branch:
                    raise ValueError(
                        f"{item.sample_id!r} has {item.supervision.branch}, expected {branch}"
                    )

    def get(self, branch: str) -> tuple[StateExample, ...]:
        if branch not in BRANCHES:
            raise ValueError(f"unknown branch {branch!r}")
        return getattr(self, branch)


def stack_state_examples(
    items: tuple[StateExample, ...],
    *,
    device: torch.device | str,
) -> BranchBatch:
    if not items:
        raise ValueError("cannot stack an empty state selection")
    feature_shapes = {tuple(item.feature.shape[1:]) for item in items}
    supervision_shapes = {tuple(item.supervision.occupancy.shape) for item in items}
    if len(feature_shapes) != 1 or len(supervision_shapes) != 1:
        raise ValueError("selected states must have compatible feature/evaluation grids")
    return BranchBatch(
        feature=torch.cat([item.feature for item in items], dim=0).to(device),
        occupancy=torch.stack(
            [item.supervision.occupancy for item in items], dim=0
        ).to(device),
        target=torch.stack([item.supervision.target for item in items], dim=0).to(
            device
        ),
        valid_mask=torch.stack(
            [item.supervision.valid_mask for item in items], dim=0
        ).to(device),
    )


def _draw(
    pool: tuple[StateExample, ...],
    count: int,
    *,
    branch: str,
    epoch: int,
    step: int,
    global_seed: int,
) -> tuple[StateExample, ...]:
    if count < 1:
        raise ValueError("active branch draw count must be positive")
    return tuple(
        pool[stable_hash(branch, epoch, step, draw, global_seed) % len(pool)]
        for draw in range(count)
    )


def default_steps_per_epoch(
    pools: BranchPools,
    branch_batch_sizes: Mapping[str, int],
) -> int:
    unknown = set(branch_batch_sizes) - set(BRANCHES)
    if unknown:
        raise ValueError(f"unknown branch batch sizes: {sorted(unknown)}")
    lengths = []
    for branch in BRANCHES:
        pool = pools.get(branch)
        if not pool:
            continue
        count = branch_batch_sizes.get(branch, 0)
        if count < 1:
            raise ValueError(f"non-empty {branch} pool requires a positive batch size")
        lengths.append(ceil(len(pool) / count))
    if not lengths:
        raise ValueError("all training pools are empty")
    return max(lengths)


def iter_fixed_branch_batches(
    pools: BranchPools,
    branch_batch_sizes: Mapping[str, int],
    *,
    epoch: int,
    global_seed: int,
    device: torch.device | str,
    steps: int | None = None,
) -> Iterator[dict[str, BranchBatch]]:
    """Yield one fixed-size batch from every non-empty global pool per step."""

    if epoch < 0 or isinstance(epoch, bool):
        raise ValueError("epoch must be a non-negative integer")
    if steps is None:
        steps = default_steps_per_epoch(pools, branch_batch_sizes)
    if steps < 1:
        raise ValueError("steps must be positive")
    for step in range(steps):
        batches: dict[str, BranchBatch] = {}
        for branch in BRANCHES:
            pool = pools.get(branch)
            if not pool:
                continue
            count = branch_batch_sizes.get(branch, 0)
            selected = _draw(
                pool,
                count,
                branch=branch,
                epoch=epoch,
                step=step,
                global_seed=global_seed,
            )
            batches[branch] = stack_state_examples(selected, device=device)
        yield batches


def iter_factual_exposure_matched_batches(
    pools: BranchPools,
    branch_batch_sizes: Mapping[str, int],
    *,
    replacement_count: int,
    epoch: int,
    global_seed: int,
    device: torch.device | str,
    steps: int,
) -> Iterator[dict[str, BranchBatch]]:
    """F×: add an independently averaged replacement factual-positive batch.

    The replacement batch occupies the synthetic *loss slot* so it receives
    U's ``lambda_synthetic`` coefficient and independent branch mean.  Its
    masks are still factual-miss supervision; no deletion is performed.
    """

    if replacement_count < 1:
        raise ValueError("replacement_count must be positive")
    positive_factual = tuple(
        example
        for example in pools.factual_miss
        if bool(torch.any(example.supervision.target))
    )
    if not positive_factual:
        raise RuntimeError(
            "F× cannot exposure-match U without positive factual-miss states"
        )
    ordinary = iter_fixed_branch_batches(
        pools,
        branch_batch_sizes,
        epoch=epoch,
        global_seed=global_seed,
        device=device,
        steps=steps,
    )
    replacement = iter_fixed_branch_batches(
        BranchPools(factual_miss=positive_factual),
        {"factual_miss": replacement_count},
        epoch=epoch,
        global_seed=global_seed ^ 0x46585F5245504C,
        device=device,
        steps=steps,
    )
    for regular_batches, replacement_batches in zip(
        ordinary, replacement, strict=True
    ):
        if "synthetic" in regular_batches:
            raise AssertionError(
                "F× must not contain deletion-based synthetic states"
            )
        regular_batches["synthetic"] = replacement_batches["factual_miss"]
        yield regular_batches


__all__ = [
    "BranchPools",
    "StateExample",
    "default_steps_per_epoch",
    "iter_factual_exposure_matched_batches",
    "iter_fixed_branch_batches",
    "stack_state_examples",
]
