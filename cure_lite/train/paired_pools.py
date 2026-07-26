"""Deterministic clean-pair schedule for the additive paired training route.

The schedule is deliberately constructed before training.  It consumes only
``PairCatalog.clean_positive`` from ``D_R`` and fixes the complete
800 x 40 plan.  Pair counts come from one seed-specific canonical cycle and
are then packed into source-disjoint two-pair updates without changing those
counts.

This module does not run an optimizer and does not alter the legacy
single-state pool/schedule implementation.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import heapq
import json

import torch

from ..cache.schema import stable_fingerprint
from ..paired_types import PairBatch, PairCatalog, PairExample, stack_pair_examples
from ..sampling import stable_hash


PAIRED_SCHEDULE_SCHEMA = "cure-lite-clean-pair-schedule-v1"
PAIRED_EPOCHS = 800
PAIRED_STEPS_PER_EPOCH = 40
PAIRS_PER_UPDATE = 2
PAIRED_OPTIMIZER_UPDATES = PAIRED_EPOCHS * PAIRED_STEPS_PER_EPOCH
PAIRED_EXPOSURES = PAIRED_OPTIMIZER_UPDATES * PAIRS_PER_UPDATE


def _require_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return seed


def _require_hex_fingerprint(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 fingerprint")
    return value


def _canonical_pairs(
    catalog: PairCatalog,
    *,
    seed: int,
) -> tuple[PairExample, ...]:
    pairs = catalog.clean_positive
    if not pairs:
        raise RuntimeError("paired schedule requires at least one clean_positive pair")
    if any(pair.pair_kind != "clean_positive" for pair in pairs):
        raise RuntimeError("trainable paired schedule accepts only clean_positive pairs")
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise RuntimeError("clean_positive pair IDs must be unique")
    feature_shapes = {tuple(pair.feature.shape[1:]) for pair in pairs}
    evaluation_shapes = {tuple(pair.occupancy_plus.shape) for pair in pairs}
    if len(feature_shapes) != 1 or len(evaluation_shapes) != 1:
        raise RuntimeError(
            "all clean_positive pairs must share feature and evaluation grids"
        )
    return tuple(
        sorted(
            pairs,
            key=lambda pair: (
                stable_hash(
                    "clean-positive-canonical-cycle-v1",
                    seed,
                    pair.pair_id,
                ),
                pair.pair_id,
            ),
        )
    )


def _canonical_cycle_counts(pair_count: int) -> tuple[int, ...]:
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    quotient, remainder = divmod(PAIRED_EXPOSURES, pair_count)
    counts = tuple(
        quotient + int(index < remainder)
        for index in range(pair_count)
    )
    if min(counts) < 1:
        raise RuntimeError(
            "the frozen schedule cannot expose every clean pair at least once"
        )
    if max(counts) - min(counts) > 1:
        raise AssertionError("canonical-cycle allocation is not support preserving")
    return counts


def _source_occurrence_queues(
    pairs: tuple[PairExample, ...],
    counts: tuple[int, ...],
) -> dict[str, deque[int]]:
    """Build target-balanced cyclic queues within every source image."""

    indices_by_source: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        indices_by_source.setdefault(pair.sample_id, []).append(index)
    queues: dict[str, deque[int]] = {}
    for source, indices in sorted(indices_by_source.items()):
        occurrences: list[int] = []
        maximum = max(counts[index] for index in indices)
        # Repeated canonical sweeps avoid emitting all copies of one target as
        # one contiguous block while preserving the exact global allocation.
        for cycle in range(maximum):
            occurrences.extend(
                index for index in indices if counts[index] > cycle
            )
        queues[source] = deque(occurrences)
    return queues


def _pack_source_disjoint_batches(
    pairs: tuple[PairExample, ...],
    counts: tuple[int, ...],
    *,
    seed: int,
) -> tuple[tuple[int, int], ...]:
    queues = _source_occurrence_queues(pairs, counts)
    if len(queues) < 2:
        raise RuntimeError(
            "paired schedule requires at least two distinct source images"
        )
    source_counts = {
        source: len(queue) for source, queue in queues.items()
    }
    largest = max(source_counts.values())
    if largest > PAIRED_OPTIMIZER_UPDATES:
        raise RuntimeError(
            "source-disjoint two-pair batches are impossible: one source owns "
            "more than half of all scheduled pair exposures"
        )

    source_rank = {
        source: stable_hash(
            "clean-positive-source-packing-v1",
            seed,
            source,
        )
        for source in queues
    }
    heap = [
        (-count, source_rank[source], source)
        for source, count in source_counts.items()
        if count
    ]
    heapq.heapify(heap)
    batches: list[tuple[int, int]] = []
    while heap:
        if len(heap) < 2:
            raise RuntimeError(
                "source-disjoint pair packing reached an infeasible remainder"
            )
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        _, _, first_source = first
        _, _, second_source = second
        if first_source == second_source:
            raise AssertionError("source heap returned a duplicate source")
        first_index = queues[first_source].popleft()
        second_index = queues[second_source].popleft()
        batches.append((first_index, second_index))

        first_remaining = -first[0] - 1
        second_remaining = -second[0] - 1
        if first_remaining:
            heapq.heappush(
                heap,
                (-first_remaining, source_rank[first_source], first_source),
            )
        if second_remaining:
            heapq.heappush(
                heap,
                (-second_remaining, source_rank[second_source], second_source),
            )

    if any(queues[source] for source in queues):
        raise AssertionError("source-disjoint packing left scheduled exposures")
    if len(batches) != PAIRED_OPTIMIZER_UPDATES:
        raise AssertionError("source-disjoint packing produced the wrong update count")
    return tuple(batches)


def _sequence_fingerprint(
    pairs: tuple[PairExample, ...],
    batches: tuple[tuple[int, int], ...],
) -> str:
    digest = hashlib.sha256()
    for update, (first, second) in enumerate(batches):
        digest.update(
            json.dumps(
                {
                    "epoch": update // PAIRED_STEPS_PER_EPOCH,
                    "step": update % PAIRED_STEPS_PER_EPOCH,
                    "pair_ids": [
                        pairs[first].pair_id,
                        pairs[second].pair_id,
                    ],
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _schedule_payload(
    *,
    seed: int,
    catalog_fingerprint: str,
    pairs: tuple[PairExample, ...],
    counts: tuple[int, ...],
    sequence_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": PAIRED_SCHEDULE_SCHEMA,
        "catalog_fingerprint": catalog_fingerprint,
        "seed": seed,
        "epochs": PAIRED_EPOCHS,
        "steps_per_epoch": PAIRED_STEPS_PER_EPOCH,
        "pairs_per_update": PAIRS_PER_UPDATE,
        "optimizer_updates": PAIRED_OPTIMIZER_UPDATES,
        "exposures": PAIRED_EXPOSURES,
        "canonical_pair_ids": [pair.pair_id for pair in pairs],
        "canonical_cycle_counts": list(counts),
        "sequence_fingerprint": sequence_fingerprint,
        "source_disjoint_within_every_update": True,
    }


@dataclass(frozen=True)
class PairedSchedule:
    """The complete immutable 32,000-update clean-pair plan."""

    seed: int
    catalog_fingerprint: str
    pairs: tuple[PairExample, ...]
    canonical_cycle_counts: tuple[int, ...]
    batch_pair_indices: tuple[tuple[int, int], ...]
    sequence_fingerprint: str
    schedule_fingerprint: str

    def __post_init__(self) -> None:
        _require_seed(self.seed)
        _require_hex_fingerprint(
            self.catalog_fingerprint,
            name="catalog_fingerprint",
        )
        _require_hex_fingerprint(
            self.sequence_fingerprint,
            name="sequence_fingerprint",
        )
        _require_hex_fingerprint(
            self.schedule_fingerprint,
            name="schedule_fingerprint",
        )
        if not self.pairs or any(
            not isinstance(pair, PairExample)
            or pair.pair_kind != "clean_positive"
            for pair in self.pairs
        ):
            raise TypeError("pairs must be non-empty clean_positive PairExample values")
        expected_order = tuple(
            sorted(
                self.pairs,
                key=lambda pair: (
                    stable_hash(
                        "clean-positive-canonical-cycle-v1",
                        self.seed,
                        pair.pair_id,
                    ),
                    pair.pair_id,
                ),
            )
        )
        if self.pairs != expected_order:
            raise ValueError("pairs are not in the seed-specific canonical order")
        if len(self.canonical_cycle_counts) != len(self.pairs):
            raise ValueError("pair counts and pair population differ")
        if self.canonical_cycle_counts != _canonical_cycle_counts(len(self.pairs)):
            raise ValueError("pair counts differ from the frozen canonical cycle")
        if len(self.batch_pair_indices) != PAIRED_OPTIMIZER_UPDATES:
            raise ValueError("paired schedule must contain exactly 32,000 updates")
        flattened = [
            index for batch in self.batch_pair_indices for index in batch
        ]
        if len(flattened) != PAIRED_EXPOSURES:
            raise ValueError("paired schedule must contain exactly 64,000 exposures")
        if any(index < 0 or index >= len(self.pairs) for index in flattened):
            raise ValueError("paired schedule contains an invalid pair index")
        actual = Counter(flattened)
        expected = {
            index: count
            for index, count in enumerate(self.canonical_cycle_counts)
        }
        if actual != expected:
            raise ValueError("actual schedule differs from canonical-cycle allocation")
        if max(actual.values()) - min(actual.values()) > 1:
            raise ValueError("pair exposure counts differ by more than one")
        for first, second in self.batch_pair_indices:
            if self.pairs[first].sample_id == self.pairs[second].sample_id:
                raise ValueError("one paired update contains duplicate source images")
        expected_sequence = _sequence_fingerprint(
            self.pairs,
            self.batch_pair_indices,
        )
        if self.sequence_fingerprint != expected_sequence:
            raise ValueError("sequence_fingerprint does not bind the update sequence")
        expected_schedule = stable_fingerprint(
            _schedule_payload(
                seed=self.seed,
                catalog_fingerprint=self.catalog_fingerprint,
                pairs=self.pairs,
                counts=self.canonical_cycle_counts,
                sequence_fingerprint=self.sequence_fingerprint,
            )
        )
        if self.schedule_fingerprint != expected_schedule:
            raise ValueError("schedule_fingerprint does not bind the frozen schedule")

    @property
    def optimizer_updates(self) -> int:
        return PAIRED_OPTIMIZER_UPDATES

    @property
    def exposures(self) -> int:
        return PAIRED_EXPOSURES

    def batch_examples(
        self,
        *,
        epoch: int,
        step: int,
    ) -> tuple[PairExample, PairExample]:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if epoch < 0 or epoch >= PAIRED_EPOCHS:
            raise ValueError("epoch must lie in [0, 800)")
        if step < 0 or step >= PAIRED_STEPS_PER_EPOCH:
            raise ValueError("step must lie in [0, 40)")
        first, second = self.batch_pair_indices[
            epoch * PAIRED_STEPS_PER_EPOCH + step
        ]
        return self.pairs[first], self.pairs[second]


def build_paired_schedule(
    catalog: PairCatalog,
    *,
    seed: int,
) -> PairedSchedule:
    """Freeze the complete seed-specific schedule from ``clean_positive``."""

    if not isinstance(catalog, PairCatalog):
        raise TypeError("catalog must be a PairCatalog")
    seed = _require_seed(seed)
    catalog_fingerprint = _require_hex_fingerprint(
        catalog.catalog_fingerprint,
        name="catalog.catalog_fingerprint",
    )
    pairs = _canonical_pairs(catalog, seed=seed)
    counts = _canonical_cycle_counts(len(pairs))
    batches = _pack_source_disjoint_batches(pairs, counts, seed=seed)
    sequence_fingerprint = _sequence_fingerprint(pairs, batches)
    payload = _schedule_payload(
        seed=seed,
        catalog_fingerprint=catalog_fingerprint,
        pairs=pairs,
        counts=counts,
        sequence_fingerprint=sequence_fingerprint,
    )
    return PairedSchedule(
        seed=seed,
        catalog_fingerprint=catalog_fingerprint,
        pairs=pairs,
        canonical_cycle_counts=counts,
        batch_pair_indices=batches,
        sequence_fingerprint=sequence_fingerprint,
        schedule_fingerprint=stable_fingerprint(payload),
    )


def pair_batch_for_update(
    schedule: PairedSchedule,
    *,
    epoch: int,
    step: int,
    device: torch.device | str,
) -> PairBatch:
    """Materialize one two-pair batch without duplicating endpoint features."""

    if not isinstance(schedule, PairedSchedule):
        raise TypeError("schedule must be a PairedSchedule")
    return stack_pair_examples(
        schedule.batch_examples(epoch=epoch, step=step),
        device=device,
    )


def iter_paired_batches(
    schedule: PairedSchedule,
    *,
    epoch: int,
    device: torch.device | str,
) -> Iterator[PairBatch]:
    """Yield the 40 frozen two-source ``PairBatch`` values for one epoch."""

    if not isinstance(schedule, PairedSchedule):
        raise TypeError("schedule must be a PairedSchedule")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("epoch must be an integer")
    if epoch < 0 or epoch >= PAIRED_EPOCHS:
        raise ValueError("epoch must lie in [0, 800)")
    for step in range(PAIRED_STEPS_PER_EPOCH):
        yield pair_batch_for_update(
            schedule,
            epoch=epoch,
            step=step,
            device=device,
        )


__all__ = [
    "PAIRED_EPOCHS",
    "PAIRED_EXPOSURES",
    "PAIRED_OPTIMIZER_UPDATES",
    "PAIRED_SCHEDULE_SCHEMA",
    "PAIRED_STEPS_PER_EPOCH",
    "PAIRS_PER_UPDATE",
    "PairedSchedule",
    "build_paired_schedule",
    "iter_paired_batches",
    "pair_batch_for_update",
]
