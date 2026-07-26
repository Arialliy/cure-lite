"""Deterministic full-population schedules for OC-APTO outcome pairs.

The schedule is built from the union of clean-positive and component-null
interventions.  Every pair receives the same number of exposures up to one
slot, and the two pairs in an update always come from different source
images.  Construction is identity-only and does not inspect tensors, losses,
or evaluation results.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import heapq
import json

from ..cache.schema import stable_fingerprint
from ..paired_types import PairCatalog, PairExample
from ..sampling import stable_hash


PAIRED_OUTCOME_SCHEDULE_SCHEMA = "cure-lite-paired-outcome-schedule-v1"
OUTCOME_PAIRS_PER_UPDATE = 2
OUTCOME_PAIR_KINDS = ("clean_positive", "component_null")


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer")
    return value


def _require_fingerprint(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 fingerprint")
    return value


def _canonical_outcome_pairs(
    catalog: PairCatalog,
    *,
    seed: int,
) -> tuple[PairExample, ...]:
    values = (*catalog.clean_positive, *catalog.component_null)
    if not catalog.clean_positive or not catalog.component_null:
        raise RuntimeError(
            "outcome schedule requires non-empty clean and component-null populations"
        )
    if any(pair.pair_kind not in OUTCOME_PAIR_KINDS for pair in values):
        raise RuntimeError("outcome population contains an invalid pair kind")
    if len({pair.pair_id for pair in values}) != len(values):
        raise RuntimeError("outcome pair IDs must be globally unique")
    feature_shapes = {tuple(pair.feature.shape[1:]) for pair in values}
    evaluation_shapes = {tuple(pair.occupancy_plus.shape) for pair in values}
    if len(feature_shapes) != 1 or len(evaluation_shapes) != 1:
        raise RuntimeError(
            "all outcome pairs must share feature and evaluation grids"
        )
    return tuple(
        sorted(
            values,
            key=lambda pair: (
                stable_hash(
                    "oc-apto-outcome-canonical-cycle-v1",
                    seed,
                    pair.pair_id,
                ),
                pair.pair_id,
            ),
        )
    )


def _balanced_counts(
    pair_count: int,
    *,
    optimizer_updates: int,
) -> tuple[int, ...]:
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    exposures = optimizer_updates * OUTCOME_PAIRS_PER_UPDATE
    quotient, remainder = divmod(exposures, pair_count)
    if quotient < 1:
        raise RuntimeError("the schedule cannot expose every outcome pair")
    counts = tuple(
        quotient + int(index < remainder)
        for index in range(pair_count)
    )
    if sum(counts) != exposures or max(counts) - min(counts) > 1:
        raise AssertionError("outcome exposure allocation is not balanced")
    return counts


def _source_occurrence_queues(
    pairs: tuple[PairExample, ...],
    counts: tuple[int, ...],
) -> dict[str, deque[int]]:
    by_source: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        by_source.setdefault(pair.sample_id, []).append(index)
    queues: dict[str, deque[int]] = {}
    for source, indices in sorted(by_source.items()):
        occurrences: list[int] = []
        maximum = max(counts[index] for index in indices)
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
    optimizer_updates: int,
) -> tuple[tuple[int, int], ...]:
    queues = _source_occurrence_queues(pairs, counts)
    if len(queues) < 2:
        raise RuntimeError("outcome schedule requires two distinct sources")
    source_counts = {source: len(queue) for source, queue in queues.items()}
    if max(source_counts.values()) > optimizer_updates:
        raise RuntimeError(
            "source-disjoint packing is impossible because one source owns "
            "more than half of all pair exposures"
        )
    source_rank = {
        source: stable_hash(
            "oc-apto-outcome-source-packing-v1",
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
                "source-disjoint outcome packing reached an infeasible remainder"
            )
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        first_source = first[2]
        second_source = second[2]
        if first_source == second_source:
            raise AssertionError("source heap returned a duplicate source")
        batches.append(
            (
                queues[first_source].popleft(),
                queues[second_source].popleft(),
            )
        )
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
        raise AssertionError("outcome packing left scheduled exposures")
    if len(batches) != optimizer_updates:
        raise AssertionError("outcome packing produced the wrong update count")
    return tuple(batches)


def _sequence_fingerprint(
    pairs: tuple[PairExample, ...],
    batches: tuple[tuple[int, int], ...],
    *,
    steps_per_epoch: int,
) -> str:
    digest = hashlib.sha256(b"cure-lite-oc-apto-outcome-sequence-v1\n")
    for update, (first, second) in enumerate(batches):
        digest.update(
            json.dumps(
                {
                    "epoch": update // steps_per_epoch,
                    "step": update % steps_per_epoch,
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


def _source_counts(
    pairs: tuple[PairExample, ...],
    counts: tuple[int, ...],
) -> tuple[tuple[str, int], ...]:
    ledger: Counter[str] = Counter()
    for pair, count in zip(pairs, counts, strict=True):
        ledger[pair.sample_id] += count
    return tuple(sorted(ledger.items()))


def _schedule_payload(
    *,
    seed: int,
    catalog_fingerprint: str,
    optimizer_updates: int,
    steps_per_epoch: int,
    pairs: tuple[PairExample, ...],
    counts: tuple[int, ...],
    source_counts: tuple[tuple[str, int], ...],
    sequence_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": PAIRED_OUTCOME_SCHEDULE_SCHEMA,
        "catalog_fingerprint": catalog_fingerprint,
        "seed": seed,
        "optimizer_updates": optimizer_updates,
        "steps_per_epoch": steps_per_epoch,
        "epochs": optimizer_updates // steps_per_epoch,
        "pairs_per_update": OUTCOME_PAIRS_PER_UPDATE,
        "exposures": optimizer_updates * OUTCOME_PAIRS_PER_UPDATE,
        "population_counts": {
            "clean_positive": sum(
                pair.pair_kind == "clean_positive" for pair in pairs
            ),
            "component_null": sum(
                pair.pair_kind == "component_null" for pair in pairs
            ),
            "outcome_union": len(pairs),
        },
        "canonical_pairs": [
            {
                "pair_id": pair.pair_id,
                "pair_kind": pair.pair_kind,
                "sample_id": pair.sample_id,
                "exposures": counts[index],
            }
            for index, pair in enumerate(pairs)
        ],
        "source_exposures": [
            {"sample_id": sample_id, "exposures": count}
            for sample_id, count in source_counts
        ],
        "selection_contract": {
            "pair_level_uniform": True,
            "clean_null_stratified_sampling": False,
            "loss_or_result_dependent": False,
            "source_disjoint_within_every_update": True,
            "exposure_difference_max": 1,
        },
        "sequence_fingerprint": sequence_fingerprint,
    }


@dataclass(frozen=True, eq=False)
class OutcomePairSchedule:
    """One immutable bounded or formal OC-APTO pair schedule."""

    seed: int
    catalog_fingerprint: str
    optimizer_updates: int
    steps_per_epoch: int
    pairs: tuple[PairExample, ...]
    pair_exposure_counts: tuple[int, ...]
    source_exposure_counts: tuple[tuple[str, int], ...]
    batch_pair_indices: tuple[tuple[int, int], ...]
    sequence_fingerprint: str
    schedule_fingerprint: str

    def __post_init__(self) -> None:
        _require_seed(self.seed)
        _require_fingerprint(
            self.catalog_fingerprint,
            name="catalog_fingerprint",
        )
        _require_fingerprint(
            self.sequence_fingerprint,
            name="sequence_fingerprint",
        )
        _require_fingerprint(
            self.schedule_fingerprint,
            name="schedule_fingerprint",
        )
        _require_positive_int(
            self.optimizer_updates,
            name="optimizer_updates",
        )
        _require_positive_int(self.steps_per_epoch, name="steps_per_epoch")
        if self.optimizer_updates % self.steps_per_epoch:
            raise ValueError("optimizer_updates must be divisible by steps_per_epoch")
        if not self.pairs or any(
            not isinstance(pair, PairExample)
            or pair.pair_kind not in OUTCOME_PAIR_KINDS
            for pair in self.pairs
        ):
            raise TypeError("pairs must be clean/component PairExample values")
        expected_order = tuple(
            sorted(
                self.pairs,
                key=lambda pair: (
                    stable_hash(
                        "oc-apto-outcome-canonical-cycle-v1",
                        self.seed,
                        pair.pair_id,
                    ),
                    pair.pair_id,
                ),
            )
        )
        if self.pairs != expected_order:
            raise ValueError("outcome pairs are not in canonical seed order")
        expected_counts = _balanced_counts(
            len(self.pairs),
            optimizer_updates=self.optimizer_updates,
        )
        if self.pair_exposure_counts != expected_counts:
            raise ValueError("pair counts differ from balanced allocation")
        if len(self.batch_pair_indices) != self.optimizer_updates:
            raise ValueError("outcome schedule has the wrong update count")
        flattened = tuple(
            index for batch in self.batch_pair_indices for index in batch
        )
        if len(flattened) != 2 * self.optimizer_updates:
            raise ValueError("outcome schedule has the wrong exposure count")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(self.pairs)
            for index in flattened
        ):
            raise ValueError("outcome schedule contains an invalid pair index")
        actual = Counter(flattened)
        if tuple(actual[index] for index in range(len(self.pairs))) != expected_counts:
            raise ValueError("outcome schedule differs from balanced allocation")
        for first, second in self.batch_pair_indices:
            if self.pairs[first].sample_id == self.pairs[second].sample_id:
                raise ValueError("one outcome update contains duplicate sources")
        expected_sources = _source_counts(self.pairs, expected_counts)
        if self.source_exposure_counts != expected_sources:
            raise ValueError("source exposure ledger is inconsistent")
        expected_sequence = _sequence_fingerprint(
            self.pairs,
            self.batch_pair_indices,
            steps_per_epoch=self.steps_per_epoch,
        )
        if self.sequence_fingerprint != expected_sequence:
            raise ValueError("sequence_fingerprint does not bind the sequence")
        expected_schedule = stable_fingerprint(self._canonical_payload())
        if self.schedule_fingerprint != expected_schedule:
            raise ValueError("schedule_fingerprint does not bind the schedule")

    @property
    def epochs(self) -> int:
        return self.optimizer_updates // self.steps_per_epoch

    @property
    def exposures(self) -> int:
        return self.optimizer_updates * OUTCOME_PAIRS_PER_UPDATE

    def _canonical_payload(self) -> dict[str, object]:
        return _schedule_payload(
            seed=self.seed,
            catalog_fingerprint=self.catalog_fingerprint,
            optimizer_updates=self.optimizer_updates,
            steps_per_epoch=self.steps_per_epoch,
            pairs=self.pairs,
            counts=self.pair_exposure_counts,
            source_counts=self.source_exposure_counts,
            sequence_fingerprint=self.sequence_fingerprint,
        )

    def canonical_receipt(self) -> dict[str, object]:
        result = self._canonical_payload()
        result["schedule_fingerprint"] = self.schedule_fingerprint
        return result

    def pair_ids_for_update(self, update: int) -> tuple[str, str]:
        if isinstance(update, bool) or not isinstance(update, int):
            raise TypeError("update must be an integer")
        if update < 0 or update >= self.optimizer_updates:
            raise ValueError("update lies outside the schedule")
        first, second = self.batch_pair_indices[update]
        return self.pairs[first].pair_id, self.pairs[second].pair_id


def build_outcome_pair_schedule(
    catalog: PairCatalog,
    *,
    seed: int,
    optimizer_updates: int,
    steps_per_epoch: int,
) -> OutcomePairSchedule:
    """Build one balanced, source-disjoint outcome schedule from D_R."""

    if not isinstance(catalog, PairCatalog):
        raise TypeError("catalog must be a PairCatalog")
    if catalog.split != "D_R":
        raise ValueError("outcome schedules permit only D_R")
    seed = _require_seed(seed)
    optimizer_updates = _require_positive_int(
        optimizer_updates,
        name="optimizer_updates",
    )
    steps_per_epoch = _require_positive_int(
        steps_per_epoch,
        name="steps_per_epoch",
    )
    if optimizer_updates % steps_per_epoch:
        raise ValueError("optimizer_updates must be divisible by steps_per_epoch")
    catalog_fingerprint = _require_fingerprint(
        catalog.catalog_fingerprint,
        name="catalog.catalog_fingerprint",
    )
    if stable_fingerprint(catalog.canonical_payload()) != catalog_fingerprint:
        raise RuntimeError("pair catalog fingerprint does not reproduce")
    pairs = _canonical_outcome_pairs(catalog, seed=seed)
    counts = _balanced_counts(
        len(pairs),
        optimizer_updates=optimizer_updates,
    )
    batches = _pack_source_disjoint_batches(
        pairs,
        counts,
        seed=seed,
        optimizer_updates=optimizer_updates,
    )
    sequence_fingerprint = _sequence_fingerprint(
        pairs,
        batches,
        steps_per_epoch=steps_per_epoch,
    )
    source_counts = _source_counts(pairs, counts)
    schedule_fingerprint = stable_fingerprint(
        _schedule_payload(
            seed=seed,
            catalog_fingerprint=catalog_fingerprint,
            optimizer_updates=optimizer_updates,
            steps_per_epoch=steps_per_epoch,
            pairs=pairs,
            counts=counts,
            source_counts=source_counts,
            sequence_fingerprint=sequence_fingerprint,
        )
    )
    return OutcomePairSchedule(
        seed=seed,
        catalog_fingerprint=catalog_fingerprint,
        optimizer_updates=optimizer_updates,
        steps_per_epoch=steps_per_epoch,
        pairs=pairs,
        pair_exposure_counts=counts,
        source_exposure_counts=source_counts,
        batch_pair_indices=batches,
        sequence_fingerprint=sequence_fingerprint,
        schedule_fingerprint=schedule_fingerprint,
    )


__all__ = [
    "OUTCOME_PAIRS_PER_UPDATE",
    "OUTCOME_PAIR_KINDS",
    "PAIRED_OUTCOME_SCHEDULE_SCHEMA",
    "OutcomePairSchedule",
    "build_outcome_pair_schedule",
]
