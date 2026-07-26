"""Deterministic D_R micro-population and bounded paired learnability gate.

This module contains the in-memory computation only.  It does not load a
dataset split, write artifacts, calibrate thresholds, or evaluate detection
performance.  The fixed micro-run trains a fresh CURE-Lite decoder on two
factual anchors and the coupled paired objective, then measures the complete
selected micro-population before and after the bounded update budget.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite, sqrt
import os
from typing import Any, Mapping

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..config import DecoderConfig, LossConfig
from ..decoder import CURELiteDecoder
from ..losses import CURELiteLoss
from ..paired_losses import PairedDifferenceLoss
from ..paired_types import PairBatch, PairCatalog, PairExample, stack_pair_examples
from ..sampling import stable_hash
from ..train.paired_step import (
    diagnose_null_pairs,
    paired_endpoint_logits,
    paired_train_step,
)
from ..train.pools import StateExample, stack_state_examples
from .artifacts import decoder_state_fingerprint
from .training_pipeline import PreparedTrainingCatalog


BOUNDED_MICRO_POPULATION_SCHEMA = (
    "cure-lite-paired-bounded-micro-population-v1"
)
BOUNDED_MICRO_SCHEDULE_SCHEMA = "cure-lite-paired-bounded-schedule-v1"
BOUNDED_EXECUTION_SCHEMA = "cure-lite-paired-bounded-execution-v1"
_CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@contextmanager
def _deterministic_torch_runtime(
    device: torch.device,
    specification: Mapping[str, object],
):
    """Apply the frozen deterministic policy and restore process flags.

    CUDA's workspace contract is process scoped.  The CLI sets it before
    importing torch-backed project modules; this context additionally checks
    it at execution time and restores the caller's environment on exit.
    """

    expected = {
        "torch_use_deterministic_algorithms": True,
        "torch_deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
        "restore_process_torch_flags_after_execution": True,
        "exact_replay_required_under_same_frozen_environment": True,
    }
    if dict(specification) != expected:
        raise RuntimeError("bounded deterministic policy differs from the freeze")

    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cublas = os.environ.get(_CUBLAS_WORKSPACE_ENV)
    required_cublas = str(specification["cublas_workspace_config"])
    if device.type == "cuda":
        active_cublas = os.environ.get(_CUBLAS_WORKSPACE_ENV)
        if active_cublas != required_cublas:
            raise RuntimeError(
                "CUDA bounded execution requires the frozen "
                "CUBLAS_WORKSPACE_CONFIG before torch initialization"
            )

    evidence: dict[str, object] = {
        "contract_satisfied": False,
        "torch_use_deterministic_algorithms": True,
        "torch_deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": (
            required_cublas if device.type == "cuda" else "not-applicable-cpu"
        ),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": device.type,
        "device_index": device.index,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "flags_restored_after_execution": False,
    }
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        evidence["contract_satisfied"] = (
            torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled()
            and torch.backends.cudnn.deterministic is True
            and torch.backends.cudnn.benchmark is False
            and (
                device.type != "cuda"
                or os.environ.get(_CUBLAS_WORKSPACE_ENV) == required_cublas
            )
        )
        if evidence["contract_satisfied"] is not True:
            raise RuntimeError("failed to activate deterministic runtime policy")
        yield evidence
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms,
            warn_only=previous_warn_only,
        )
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        if previous_cublas is None:
            os.environ.pop(_CUBLAS_WORKSPACE_ENV, None)
        else:
            os.environ[_CUBLAS_WORKSPACE_ENV] = previous_cublas
        evidence["flags_restored_after_execution"] = (
            torch.are_deterministic_algorithms_enabled()
            is previous_algorithms
            and torch.is_deterministic_algorithms_warn_only_enabled()
            is previous_warn_only
            and torch.backends.cudnn.deterministic
            is previous_cudnn_deterministic
            and torch.backends.cudnn.benchmark is previous_cudnn_benchmark
            and os.environ.get(_CUBLAS_WORKSPACE_ENV) == previous_cublas
        )


def _anchor_id(branch: str, example: StateExample) -> str:
    supervision = example.supervision
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-bounded-anchor-id-v1",
            "branch": branch,
            "sample_id": example.sample_id,
            "positive_gt_ids": list(supervision.positive_gt_ids),
        }
    )


def _stable_select(
    values: tuple[Any, ...],
    count: int,
    *,
    namespace: str,
    seed: int,
    identity,
) -> tuple[Any, ...]:
    _require_positive_int(count, name=f"{namespace}.count")
    if len(values) < count:
        raise RuntimeError(
            f"{namespace} population is too small ({len(values)} < {count})"
        )
    return tuple(
        sorted(
            values,
            key=lambda item: (
                stable_hash(namespace, seed, identity(item)),
                identity(item),
            ),
        )[:count]
    )


def _select_distinct_source_pairs(
    pairs: tuple[PairExample, ...],
    count: int,
    *,
    seed: int,
) -> tuple[PairExample, ...]:
    by_source: dict[str, list[PairExample]] = {}
    for pair in pairs:
        by_source.setdefault(pair.sample_id, []).append(pair)
    if len(by_source) < count:
        raise RuntimeError(
            "bounded clean-pair population cannot provide the frozen number "
            "of distinct source images"
        )
    selected_sources = sorted(
        by_source,
        key=lambda sample_id: (
            stable_hash(
                "bounded-clean-pair-source-first-v1",
                seed,
                sample_id,
            ),
            sample_id,
        ),
    )[:count]
    selected = tuple(
        min(
            by_source[sample_id],
            key=lambda pair: (
                stable_hash(
                    "bounded-clean-pair-within-source-v1",
                    seed,
                    pair.pair_id,
                ),
                pair.pair_id,
            ),
        )
        for sample_id in selected_sources
    )
    return selected


@dataclass(frozen=True)
class BoundedMicroPopulation:
    """The complete fixed 16-unit populations used by the bounded gate."""

    seed: int
    pair_catalog_fingerprint: str
    prepared_catalog_fingerprint: str
    clean_pairs: tuple[PairExample, ...]
    factual_miss: tuple[StateExample, ...]
    factual_no_miss: tuple[StateExample, ...]
    component_null: tuple[PairExample, ...]
    identity_null: tuple[PairExample, ...]
    factual_miss_ids: tuple[str, ...]
    factual_no_miss_ids: tuple[str, ...]
    population_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "clean_pairs",
            "factual_miss",
            "factual_no_miss",
            "component_null",
            "identity_null",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a non-empty tuple")
        if len({pair.sample_id for pair in self.clean_pairs}) != len(
            self.clean_pairs
        ):
            raise ValueError("bounded clean pairs must use distinct sources")
        if any(pair.pair_kind != "clean_positive" for pair in self.clean_pairs):
            raise ValueError("clean_pairs contains a non-positive pair")
        if any(
            pair.pair_kind != "component_null"
            for pair in self.component_null
        ):
            raise ValueError("component_null contains another pair kind")
        if any(
            pair.pair_kind != "identity_null" for pair in self.identity_null
        ):
            raise ValueError("identity_null contains another pair kind")
        if len(self.factual_miss_ids) != len(self.factual_miss):
            raise ValueError("factual-miss identities do not align")
        if len(self.factual_no_miss_ids) != len(self.factual_no_miss):
            raise ValueError("factual-no-miss identities do not align")
        if len(set(self.factual_miss_ids)) != len(self.factual_miss_ids):
            raise ValueError("factual-miss identities are not unique")
        if len(set(self.factual_no_miss_ids)) != len(
            self.factual_no_miss_ids
        ):
            raise ValueError("factual-no-miss identities are not unique")
        if any(
            example.supervision.branch != "factual_miss"
            for example in self.factual_miss
        ):
            raise ValueError("factual_miss contains another branch")
        if any(
            example.supervision.branch != "factual_no_miss"
            for example in self.factual_no_miss
        ):
            raise ValueError("factual_no_miss contains another branch")
        if stable_fingerprint(self.canonical_payload()) != (
            self.population_fingerprint
        ):
            raise ValueError("bounded micro-population fingerprint changed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": BOUNDED_MICRO_POPULATION_SCHEMA,
            "seed": self.seed,
            "pair_catalog_fingerprint": self.pair_catalog_fingerprint,
            "prepared_catalog_fingerprint": (
                self.prepared_catalog_fingerprint
            ),
            "selection_rule": (
                "stable-hash-over-identities-source-first-without-"
                "feature-loss-or-result-access-v1"
            ),
            "clean_pairs": [
                {
                    "pair_id": pair.pair_id,
                    "sample_id": pair.sample_id,
                }
                for pair in self.clean_pairs
            ],
            "factual_miss": [
                {
                    "anchor_id": anchor_id,
                    "sample_id": example.sample_id,
                    "positive_gt_ids": list(
                        example.supervision.positive_gt_ids
                    ),
                }
                for anchor_id, example in zip(
                    self.factual_miss_ids,
                    self.factual_miss,
                    strict=True,
                )
            ],
            "factual_no_miss": [
                {
                    "anchor_id": anchor_id,
                    "sample_id": example.sample_id,
                    "positive_gt_ids": list(
                        example.supervision.positive_gt_ids
                    ),
                }
                for anchor_id, example in zip(
                    self.factual_no_miss_ids,
                    self.factual_no_miss,
                    strict=True,
                )
            ],
            "component_null": [
                {
                    "pair_id": pair.pair_id,
                    "sample_id": pair.sample_id,
                }
                for pair in self.component_null
            ],
            "identity_null": [
                {
                    "pair_id": pair.pair_id,
                    "sample_id": pair.sample_id,
                }
                for pair in self.identity_null
            ],
        }


def build_bounded_micro_population(
    pair_catalog: PairCatalog,
    prepared: PreparedTrainingCatalog,
    specification: Mapping[str, object],
) -> BoundedMicroPopulation:
    """Select the fixed micro-population using identities only."""

    if not isinstance(pair_catalog, PairCatalog):
        raise TypeError("pair_catalog must be a PairCatalog")
    if not isinstance(prepared, PreparedTrainingCatalog):
        raise TypeError("prepared must be PreparedTrainingCatalog")
    if not isinstance(specification, Mapping):
        raise TypeError("specification must be a mapping")
    if pair_catalog.split != "D_R":
        raise ValueError("bounded micro-population permits only D_R")
    seed = specification.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("micro-population seed must be an integer")
    clean_count = _require_positive_int(
        specification.get("clean_pairs"),
        name="clean_pairs",
    )
    required_distinct = _require_positive_int(
        specification.get("require_distinct_clean_pair_sources"),
        name="require_distinct_clean_pair_sources",
    )
    if clean_count != required_distinct:
        raise ValueError(
            "this frozen selector requires every clean pair source distinct"
        )

    factual_miss_all = tuple(
        example
        for entry in prepared.entries
        for example in entry.factual_examples
    )
    factual_no_miss_all = tuple(
        entry.factual_no_miss_example
        for entry in prepared.entries
        if entry.factual_no_miss_example is not None
    )
    clean = _select_distinct_source_pairs(
        pair_catalog.clean_positive,
        clean_count,
        seed=seed,
    )
    factual_miss = _stable_select(
        factual_miss_all,
        _require_positive_int(
            specification.get("factual_miss_anchors"),
            name="factual_miss_anchors",
        ),
        namespace="bounded-factual-miss-v1",
        seed=seed,
        identity=lambda example: _anchor_id("factual_miss", example),
    )
    factual_no_miss = _stable_select(
        factual_no_miss_all,
        _require_positive_int(
            specification.get("factual_no_miss_anchors"),
            name="factual_no_miss_anchors",
        ),
        namespace="bounded-factual-no-miss-v1",
        seed=seed,
        identity=lambda example: _anchor_id(
            "factual_no_miss",
            example,
        ),
    )
    component_null = _stable_select(
        pair_catalog.component_null,
        _require_positive_int(
            specification.get("component_null_pairs"),
            name="component_null_pairs",
        ),
        namespace="bounded-component-null-v1",
        seed=seed,
        identity=lambda pair: pair.pair_id,
    )
    identity_null = _stable_select(
        pair_catalog.identity_null,
        _require_positive_int(
            specification.get("identity_null_pairs"),
            name="identity_null_pairs",
        ),
        namespace="bounded-identity-null-v1",
        seed=seed,
        identity=lambda pair: pair.pair_id,
    )
    miss_ids = tuple(
        _anchor_id("factual_miss", example) for example in factual_miss
    )
    no_miss_ids = tuple(
        _anchor_id("factual_no_miss", example)
        for example in factual_no_miss
    )
    prepared_fingerprint = stable_fingerprint(
        {
            "source_ids": list(prepared.source_ids),
            "support_summary": prepared.support_summary.canonical_payload(),
            "occupancy_config": prepared.occupancy_config,
            "match_config": prepared.match_config,
            "intervention_config": prepared.intervention_config,
        }
    )
    canonical = {
        "schema_version": BOUNDED_MICRO_POPULATION_SCHEMA,
        "seed": seed,
        "pair_catalog_fingerprint": pair_catalog.catalog_fingerprint,
        "prepared_catalog_fingerprint": prepared_fingerprint,
        "selection_rule": (
            "stable-hash-over-identities-source-first-without-"
            "feature-loss-or-result-access-v1"
        ),
        "clean_pairs": [
            {"pair_id": pair.pair_id, "sample_id": pair.sample_id}
            for pair in clean
        ],
        "factual_miss": [
            {
                "anchor_id": anchor_id,
                "sample_id": example.sample_id,
                "positive_gt_ids": list(
                    example.supervision.positive_gt_ids
                ),
            }
            for anchor_id, example in zip(
                miss_ids,
                factual_miss,
                strict=True,
            )
        ],
        "factual_no_miss": [
            {
                "anchor_id": anchor_id,
                "sample_id": example.sample_id,
                "positive_gt_ids": list(
                    example.supervision.positive_gt_ids
                ),
            }
            for anchor_id, example in zip(
                no_miss_ids,
                factual_no_miss,
                strict=True,
            )
        ],
        "component_null": [
            {"pair_id": pair.pair_id, "sample_id": pair.sample_id}
            for pair in component_null
        ],
        "identity_null": [
            {"pair_id": pair.pair_id, "sample_id": pair.sample_id}
            for pair in identity_null
        ],
    }
    return BoundedMicroPopulation(
        seed=seed,
        pair_catalog_fingerprint=pair_catalog.catalog_fingerprint,
        prepared_catalog_fingerprint=prepared_fingerprint,
        clean_pairs=clean,
        factual_miss=factual_miss,
        factual_no_miss=factual_no_miss,
        component_null=component_null,
        identity_null=identity_null,
        factual_miss_ids=miss_ids,
        factual_no_miss_ids=no_miss_ids,
        population_fingerprint=stable_fingerprint(canonical),
    )


@dataclass(frozen=True)
class BoundedMicroSchedule:
    optimizer_updates: int
    steps_per_epoch: int
    pair_indices: tuple[tuple[int, int], ...]
    factual_miss_indices: tuple[tuple[int, int, int, int], ...]
    factual_no_miss_indices: tuple[tuple[int, int, int, int], ...]
    pair_counts: tuple[int, ...]
    factual_miss_counts: tuple[int, ...]
    factual_no_miss_counts: tuple[int, ...]
    schedule_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": BOUNDED_MICRO_SCHEDULE_SCHEMA,
            "optimizer_updates": self.optimizer_updates,
            "steps_per_epoch": self.steps_per_epoch,
            "pair_indices": [list(value) for value in self.pair_indices],
            "factual_miss_indices": [
                list(value) for value in self.factual_miss_indices
            ],
            "factual_no_miss_indices": [
                list(value) for value in self.factual_no_miss_indices
            ],
            "pair_counts": list(self.pair_counts),
            "factual_miss_counts": list(self.factual_miss_counts),
            "factual_no_miss_counts": list(self.factual_no_miss_counts),
        }

    def __post_init__(self) -> None:
        if len(self.pair_indices) != self.optimizer_updates:
            raise ValueError("pair schedule length differs from update budget")
        if len(self.factual_miss_indices) != self.optimizer_updates:
            raise ValueError("factual-miss schedule length differs")
        if len(self.factual_no_miss_indices) != self.optimizer_updates:
            raise ValueError("factual-no-miss schedule length differs")
        if stable_fingerprint(self.canonical_payload()) != (
            self.schedule_fingerprint
        ):
            raise ValueError("bounded schedule fingerprint changed")


def build_bounded_micro_schedule(
    population: BoundedMicroPopulation,
    budget: Mapping[str, object],
) -> BoundedMicroSchedule:
    """Build exact cyclic 4/4/2 exposure ledgers for every update."""

    if not isinstance(population, BoundedMicroPopulation):
        raise TypeError("population must be BoundedMicroPopulation")
    updates = _require_positive_int(
        budget.get("optimizer_updates"),
        name="optimizer_updates",
    )
    steps_per_epoch = _require_positive_int(
        budget.get("steps_per_epoch"),
        name="steps_per_epoch",
    )
    if budget.get("factual_miss_states_per_update") != 4:
        raise ValueError("bounded factual-miss batch must remain 4")
    if budget.get("factual_no_miss_states_per_update") != 4:
        raise ValueError("bounded factual-no-miss batch must remain 4")
    if budget.get("clean_pairs_per_update") != 2:
        raise ValueError("bounded clean-pair batch must remain 2")

    pair_indices = tuple(
        (
            (2 * update) % len(population.clean_pairs),
            (2 * update + 1) % len(population.clean_pairs),
        )
        for update in range(updates)
    )
    miss_indices = tuple(
        tuple(
            (4 * update + draw) % len(population.factual_miss)
            for draw in range(4)
        )
        for update in range(updates)
    )
    no_miss_indices = tuple(
        tuple(
            (4 * update + draw) % len(population.factual_no_miss)
            for draw in range(4)
        )
        for update in range(updates)
    )
    if any(
        population.clean_pairs[first].sample_id
        == population.clean_pairs[second].sample_id
        for first, second in pair_indices
    ):
        raise RuntimeError("one bounded update contains duplicate pair sources")

    def counts(
        size: int,
        rows: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        ledger = Counter(index for row in rows for index in row)
        return tuple(ledger[index] for index in range(size))

    pair_counts = counts(len(population.clean_pairs), pair_indices)
    miss_counts = counts(len(population.factual_miss), miss_indices)
    no_miss_counts = counts(
        len(population.factual_no_miss),
        no_miss_indices,
    )
    canonical = {
        "schema_version": BOUNDED_MICRO_SCHEDULE_SCHEMA,
        "optimizer_updates": updates,
        "steps_per_epoch": steps_per_epoch,
        "pair_indices": [list(value) for value in pair_indices],
        "factual_miss_indices": [
            list(value) for value in miss_indices
        ],
        "factual_no_miss_indices": [
            list(value) for value in no_miss_indices
        ],
        "pair_counts": list(pair_counts),
        "factual_miss_counts": list(miss_counts),
        "factual_no_miss_counts": list(no_miss_counts),
    }
    return BoundedMicroSchedule(
        optimizer_updates=updates,
        steps_per_epoch=steps_per_epoch,
        pair_indices=pair_indices,
        factual_miss_indices=miss_indices,
        factual_no_miss_indices=no_miss_indices,
        pair_counts=pair_counts,
        factual_miss_counts=miss_counts,
        factual_no_miss_counts=no_miss_counts,
        schedule_fingerprint=stable_fingerprint(canonical),
    )


def _pair_batch(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    update: int,
    *,
    device: torch.device | str,
) -> PairBatch:
    first, second = schedule.pair_indices[update]
    return stack_pair_examples(
        (
            population.clean_pairs[first],
            population.clean_pairs[second],
        ),
        device=device,
    )


def _factual_batches(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    update: int,
    *,
    device: torch.device | str,
) -> dict[str, object]:
    return {
        "factual_miss": stack_state_examples(
            tuple(
                population.factual_miss[index]
                for index in schedule.factual_miss_indices[update]
            ),
            device=device,
        ),
        "factual_no_miss": stack_state_examples(
            tuple(
                population.factual_no_miss[index]
                for index in schedule.factual_no_miss_indices[update]
            ),
            device=device,
        ),
    }


class _ForwardLedger:
    def __init__(self, decoder: CURELiteDecoder) -> None:
        self.calls = 0
        self.states = 0
        self._handle = decoder.register_forward_hook(self._hook)

    def _hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise RuntimeError("decoder forward ledger received invalid output")
        self.calls += 1
        self.states += int(output.shape[0])

    def snapshot(self) -> tuple[int, int]:
        return self.calls, self.states

    def close(self) -> None:
        self._handle.remove()


def _macro_pair_metrics(
    logits_plus: Tensor,
    logits_minus: Tensor,
    batch: PairBatch,
    paired_criterion: PairedDifferenceLoss,
) -> dict[str, object]:
    result = paired_criterion(
        logits_plus,
        logits_minus,
        batch.label_increment,
        batch.image_valid_mask,
    )
    delta = torch.sigmoid(logits_minus) - torch.sigmoid(logits_plus)
    positive = batch.label_increment.to(torch.bool)
    zero = batch.image_valid_mask & ~positive
    positive_means = torch.stack(
        [delta[index][positive[index]].mean() for index in range(delta.shape[0])]
    )
    zero_abs_means = torch.stack(
        [
            delta[index][zero[index]].abs().mean()
            for index in range(delta.shape[0])
        ]
    )
    return {
        "paired_loss": float(result["total"].cpu()),
        "positive_macro_mean_delta": float(positive_means.mean().cpu()),
        "positive_unit_fraction_ge_0_25": float(
            (positive_means >= 0.25).to(torch.float32).mean().cpu()
        ),
        "zero_macro_mean_abs_delta": float(
            zero_abs_means.mean().cpu()
        ),
        "per_pair_positive_mean_delta": [
            {
                "pair_id": pair_id,
                "value": float(value.cpu()),
            }
            for pair_id, value in zip(
                batch.pair_ids,
                positive_means,
                strict=True,
            )
        ],
        "per_pair_zero_mean_abs_delta": [
            {
                "pair_id": pair_id,
                "value": float(value.cpu()),
            }
            for pair_id, value in zip(
                batch.pair_ids,
                zero_abs_means,
                strict=True,
            )
        ],
    }


def evaluate_bounded_micro_population(
    decoder: CURELiteDecoder,
    population: BoundedMicroPopulation,
    absolute_criterion: CURELiteLoss,
    paired_criterion: PairedDifferenceLoss,
    *,
    device: torch.device | str,
) -> dict[str, object]:
    """Evaluate every fixed unit once with macro reductions."""

    decoder.eval()
    with torch.no_grad():
        pair_batch = stack_pair_examples(
            population.clean_pairs,
            device=device,
        )
        logits_plus, logits_minus = paired_endpoint_logits(
            decoder,
            pair_batch,
        )
        paired = _macro_pair_metrics(
            logits_plus,
            logits_minus,
            pair_batch,
            paired_criterion,
        )

        anchor_metrics: dict[str, object] = {}
        for branch, values, identities in (
            (
                "factual_miss",
                population.factual_miss,
                population.factual_miss_ids,
            ),
            (
                "factual_no_miss",
                population.factual_no_miss,
                population.factual_no_miss_ids,
            ),
        ):
            batch = stack_state_examples(values, device=device)
            logits = decoder(batch.feature.detach(), batch.occupancy)
            result = absolute_criterion(
                logits,
                batch.target,
                batch.valid_mask,
            )
            anchor_metrics[branch] = {
                "loss": float(result["total"].cpu()),
                "per_unit_loss": [
                    {
                        "anchor_id": anchor_id,
                        "value": float(value.cpu()),
                    }
                    for anchor_id, value in zip(
                        identities,
                        result["per_state_total"],
                        strict=True,
                    )
                ],
            }

        null_metrics: dict[str, object] = {}
        for name, values in (
            ("component_null", population.component_null),
            ("identity_null", population.identity_null),
        ):
            batch = stack_pair_examples(values, device=device)
            result = diagnose_null_pairs(decoder, batch)
            mean_abs = result["per_pair_mean_abs_delta"]
            max_abs = result["per_pair_max_abs_delta"]
            null_metrics[name] = {
                "macro_mean_abs_delta": float(mean_abs.mean().cpu()),
                "maximum_abs_delta": float(max_abs.max().cpu()),
                "per_pair_mean_abs_delta": [
                    {
                        "pair_id": pair_id,
                        "value": float(value.cpu()),
                    }
                    for pair_id, value in zip(
                        batch.pair_ids,
                        mean_abs,
                        strict=True,
                    )
                ],
            }
    return {
        "paired": paired,
        "anchors": anchor_metrics,
        "nulls": null_metrics,
    }


def _ratio(final: float, initial: float) -> float:
    if not isfinite(initial) or not isfinite(final):
        raise ValueError("bounded gate ratios require finite values")
    if initial <= 0.0:
        raise ValueError("bounded gate ratio denominator must be positive")
    return final / initial


def _computational_gates(
    initial: Mapping[str, object],
    final: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    initial_paired = initial["paired"]
    final_paired = final["paired"]
    initial_anchors = initial["anchors"]
    final_anchors = final["anchors"]
    final_nulls = final["nulls"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            initial_paired,
            final_paired,
            initial_anchors,
            final_anchors,
            final_nulls,
        )
    ):
        raise TypeError("bounded metric payload is malformed")

    observed = {
        "paired_loss_final_over_initial": _ratio(
            float(final_paired["paired_loss"]),
            float(initial_paired["paired_loss"]),
        ),
        "paired_positive_macro_mean_delta": float(
            final_paired["positive_macro_mean_delta"]
        ),
        "paired_units_mean_delta_at_least_0_25_fraction": float(
            final_paired["positive_unit_fraction_ge_0_25"]
        ),
        "paired_zero_macro_mean_abs_delta": float(
            final_paired["zero_macro_mean_abs_delta"]
        ),
        "factual_miss_anchor_loss_final_over_initial": _ratio(
            float(final_anchors["factual_miss"]["loss"]),
            float(initial_anchors["factual_miss"]["loss"]),
        ),
        "factual_no_miss_anchor_loss_final_over_initial": _ratio(
            float(final_anchors["factual_no_miss"]["loss"]),
            float(initial_anchors["factual_no_miss"]["loss"]),
        ),
        "component_null_macro_mean_abs_delta": float(
            final_nulls["component_null"]["macro_mean_abs_delta"]
        ),
        "identity_null_max_abs_delta": float(
            final_nulls["identity_null"]["maximum_abs_delta"]
        ),
    }
    rules = {
        "paired_loss_final_over_initial": (
            "max",
            "paired_loss_final_over_initial_max",
        ),
        "paired_positive_macro_mean_delta": (
            "min",
            "paired_positive_macro_mean_delta_min",
        ),
        "paired_units_mean_delta_at_least_0_25_fraction": (
            "min",
            "paired_units_mean_delta_at_least_0_25_fraction_min",
        ),
        "paired_zero_macro_mean_abs_delta": (
            "max",
            "paired_zero_macro_mean_abs_delta_max",
        ),
        "factual_miss_anchor_loss_final_over_initial": (
            "max",
            "factual_miss_anchor_loss_final_over_initial_max",
        ),
        "factual_no_miss_anchor_loss_final_over_initial": (
            "max",
            "factual_no_miss_anchor_loss_final_over_initial_max",
        ),
        "component_null_macro_mean_abs_delta": (
            "max",
            "component_null_macro_mean_abs_delta_max",
        ),
        "identity_null_max_abs_delta": (
            "max",
            "identity_null_max_abs_delta_max",
        ),
    }
    checks: dict[str, object] = {}
    for name, (direction, threshold_name) in rules.items():
        threshold = float(thresholds[threshold_name])
        value = observed[name]
        if not isfinite(threshold) or not isfinite(value):
            raise ValueError(
                f"bounded computational gate {name} must be finite"
            )
        passed = value >= threshold if direction == "min" else value <= threshold
        checks[name] = {
            "value": value,
            "direction": direction,
            "threshold": threshold,
            "pass": passed,
        }
    return {
        "observed": observed,
        "checks": checks,
        "all_pass": all(value["pass"] for value in checks.values()),
    }


def execute_bounded_learnability(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    config: Mapping[str, object],
    *,
    device: torch.device | str,
) -> dict[str, object]:
    """Execute the fixed micro-run and return a JSON-compatible evidence map."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    optimization = config["optimization"]
    budget = config["budget"]
    gates = config["gates"]
    determinism = config["determinism"]
    if not isinstance(optimization, Mapping) or not isinstance(
        budget,
        Mapping,
    ) or not isinstance(gates, Mapping) or not isinstance(
        determinism,
        Mapping,
    ):
        raise TypeError(
            "bounded config optimization/budget/gates/determinism are malformed"
        )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    seed = int(optimization["seed"])
    decoder_config = DecoderConfig(**dict(optimization["decoder"]))
    loss_config = LossConfig(**dict(optimization["loss"]))

    cuda_devices: list[int] = []
    if target_device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device()
            if target_device.index is None
            else target_device.index
        ]
    with _deterministic_torch_runtime(
        target_device,
        determinism,
    ) as deterministic_runtime, torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        decoder = CURELiteDecoder(decoder_config).to(target_device)
        absolute = CURELiteLoss(loss_config)
        paired = PairedDifferenceLoss()
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        initial_decoder_fingerprint = decoder_state_fingerprint(decoder)
        initial_parameter_norm = sqrt(
            sum(
                float(parameter.detach().double().square().sum().cpu())
                for parameter in decoder.parameters()
            )
        )
        parameter_count = sum(
            parameter.numel() for parameter in decoder.parameters()
        )

        ledger = _ForwardLedger(decoder)
        try:
            before_initial = ledger.snapshot()
            initial = evaluate_bounded_micro_population(
                decoder,
                population,
                absolute,
                paired,
                device=target_device,
            )
            after_initial = ledger.snapshot()
            initial_forward = {
                "calls": after_initial[0] - before_initial[0],
                "state_evaluations": after_initial[1] - before_initial[1],
            }

            trace: list[dict[str, object]] = []
            minimum_gradient_norm = float("inf")
            maximum_gradient_norm = 0.0
            nonfinite_gradient_updates = 0
            zero_gradient_updates = 0
            training_start = ledger.snapshot()
            for update in range(schedule.optimizer_updates):
                before_update = ledger.snapshot()
                logs = paired_train_step(
                    decoder,
                    absolute,
                    paired,
                    optimizer,
                    _factual_batches(
                        population,
                        schedule,
                        update,
                        device=target_device,
                    ),
                    _pair_batch(
                        population,
                        schedule,
                        update,
                        device=target_device,
                    ),
                )
                squared_norm = sum(
                    float(parameter.grad.detach().double().square().sum().cpu())
                    for parameter in decoder.parameters()
                    if parameter.grad is not None
                )
                gradient_norm = sqrt(squared_norm)
                finite_gradient = torch.isfinite(
                    torch.tensor(gradient_norm)
                ).item()
                if not finite_gradient:
                    nonfinite_gradient_updates += 1
                if gradient_norm <= 0.0:
                    zero_gradient_updates += 1
                minimum_gradient_norm = min(
                    minimum_gradient_norm,
                    gradient_norm,
                )
                maximum_gradient_norm = max(
                    maximum_gradient_norm,
                    gradient_norm,
                )
                after_update = ledger.snapshot()
                trace.append(
                    {
                        "update": update,
                        "epoch": update // schedule.steps_per_epoch,
                        "step": update % schedule.steps_per_epoch,
                        "losses": logs,
                        "gradient_l2_norm": gradient_norm,
                        "decoder_forward_calls": (
                            after_update[0] - before_update[0]
                        ),
                        "decoder_state_evaluations": (
                            after_update[1] - before_update[1]
                        ),
                    }
                )
            training_end = ledger.snapshot()
            training_forward = {
                "calls": training_end[0] - training_start[0],
                "state_evaluations": training_end[1] - training_start[1],
            }

            before_final = ledger.snapshot()
            final = evaluate_bounded_micro_population(
                decoder,
                population,
                absolute,
                paired,
                device=target_device,
            )
            after_final = ledger.snapshot()
            final_forward = {
                "calls": after_final[0] - before_final[0],
                "state_evaluations": after_final[1] - before_final[1],
            }
            total_forward = {
                "calls": after_final[0],
                "state_evaluations": after_final[1],
            }
        finally:
            ledger.close()

        final_decoder_fingerprint = decoder_state_fingerprint(decoder)
        final_parameter_norm = sqrt(
            sum(
                float(parameter.detach().double().square().sum().cpu())
                for parameter in decoder.parameters()
            )
        )

    exposure = {
        "clean_pairs": [
            {
                "pair_id": pair.pair_id,
                "sample_id": pair.sample_id,
                "count": schedule.pair_counts[index],
            }
            for index, pair in enumerate(population.clean_pairs)
        ],
        "factual_miss": [
            {
                "anchor_id": anchor_id,
                "sample_id": example.sample_id,
                "count": schedule.factual_miss_counts[index],
            }
            for index, (anchor_id, example) in enumerate(
                zip(
                    population.factual_miss_ids,
                    population.factual_miss,
                    strict=True,
                )
            )
        ],
        "factual_no_miss": [
            {
                "anchor_id": anchor_id,
                "sample_id": example.sample_id,
                "count": schedule.factual_no_miss_counts[index],
            }
            for index, (anchor_id, example) in enumerate(
                zip(
                    population.factual_no_miss_ids,
                    population.factual_no_miss,
                    strict=True,
                )
            )
        ],
    }
    per_update_forward_exact = all(
        row["decoder_forward_calls"] == 3
        and row["decoder_state_evaluations"] == 12
        for row in trace
    )
    exposure_complete = all(
        min(counts) > 0 and max(counts) == min(counts)
        for counts in (
            schedule.pair_counts,
            schedule.factual_miss_counts,
            schedule.factual_no_miss_counts,
        )
    )
    structural_checks = {
        "deterministic_runtime_contract_satisfied": (
            deterministic_runtime["contract_satisfied"] is True
            and deterministic_runtime["flags_restored_after_execution"] is True
        ),
        "micro_population_counts_exact": (
            len(population.clean_pairs)
            == len(population.factual_miss)
            == len(population.factual_no_miss)
            == len(population.component_null)
            == len(population.identity_null)
            == 16
        ),
        "clean_pair_sources_distinct": (
            len({pair.sample_id for pair in population.clean_pairs}) == 16
        ),
        "all_optimizer_updates_completed": (
            len(trace) == int(budget["optimizer_updates"]) == 400
        ),
        "all_gradients_finite": nonfinite_gradient_updates == 0,
        "every_update_total_gradient_norm_positive": (
            zero_gradient_updates == 0
        ),
        "decoder_parameters_changed": (
            final_decoder_fingerprint != initial_decoder_fingerprint
        ),
        "training_forward_budget_exact": (
            training_forward["calls"]
            == int(budget["training_decoder_forward_calls"])
            and training_forward["state_evaluations"]
            == int(budget["training_decoder_state_evaluations"])
            and per_update_forward_exact
        ),
        "evaluation_forward_budget_exact": (
            initial_forward["calls"]
            == final_forward["calls"]
            == int(
                budget["evaluation_decoder_forward_calls_per_snapshot"]
            )
            and initial_forward["state_evaluations"]
            == final_forward["state_evaluations"]
            == int(
                budget[
                    "evaluation_decoder_state_evaluations_per_snapshot"
                ]
            )
        ),
        "total_forward_budget_exact": (
            total_forward["calls"]
            == int(budget["total_decoder_forward_calls"])
            and total_forward["state_evaluations"]
            == int(budget["total_decoder_state_evaluations"])
        ),
        "all_exposure_ledgers_complete": exposure_complete,
    }
    structural_execution_pass = all(structural_checks.values())
    computational = _computational_gates(
        initial,
        final,
        gates["computational_learnability"],
    )
    computational_pass = (
        structural_execution_pass and computational["all_pass"] is True
    )
    return {
        "schema_version": BOUNDED_EXECUTION_SCHEMA,
        "execution_status": "completed",
        "device": str(target_device),
        "population_fingerprint": population.population_fingerprint,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "optimizer_updates_completed": len(trace),
        "initial": initial,
        "final": final,
        "computational_gates": computational,
        "structural_checks": structural_checks,
        "structural_execution_pass": structural_execution_pass,
        "computational_learnability_pass": computational_pass,
        "parameters": {
            "trainable_parameter_count": parameter_count,
            "initial_decoder_fingerprint": (
                initial_decoder_fingerprint
            ),
            "final_decoder_fingerprint": final_decoder_fingerprint,
            "initial_l2_norm": initial_parameter_norm,
            "final_l2_norm": final_parameter_norm,
        },
        "gradients": {
            "minimum_update_l2_norm": minimum_gradient_norm,
            "maximum_update_l2_norm": maximum_gradient_norm,
            "nonfinite_updates": nonfinite_gradient_updates,
            "zero_norm_updates": zero_gradient_updates,
        },
        "forward_budget": {
            "initial_evaluation": initial_forward,
            "training": training_forward,
            "final_evaluation": final_forward,
            "total": total_forward,
        },
        "deterministic_runtime": deterministic_runtime,
        "exposure": exposure,
        "trace": trace,
        "interpretation": {
            "not_performance_evidence": True,
            "authorizes_formal_800": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
        },
    }


__all__ = [
    "BOUNDED_EXECUTION_SCHEMA",
    "BOUNDED_MICRO_POPULATION_SCHEMA",
    "BOUNDED_MICRO_SCHEDULE_SCHEMA",
    "BoundedMicroPopulation",
    "BoundedMicroSchedule",
    "build_bounded_micro_population",
    "build_bounded_micro_schedule",
    "evaluate_bounded_micro_population",
    "execute_bounded_learnability",
]
