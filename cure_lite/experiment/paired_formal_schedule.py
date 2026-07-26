"""Immutable full-horizon schedule for paired CURE-Lite formal training.

The clean-pair sequence is owned by :class:`PairedSchedule`.  This module adds
the two factual-anchor sequences required by the frozen paired objective:
four factual-miss states, four factual-no-miss states, and two clean pairs
(four endpoints) in every update.

Construction is identity-only and read-only.  It uses the existing
``PreparedTrainingCatalog`` epoch materializer and the existing stable-hash
draw implementation; it performs no model forward, optimization, calibration,
or evaluation-split access.  The resulting object is shared unchanged by the
proposed method and every matched control.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import hashlib

import torch

from ..cache.schema import stable_fingerprint
from ..config import config_to_dict
from ..paired_types import PairBatch
from ..train.paired_control_step import CONTROL_KINDS
from ..train.paired_pools import (
    PAIRED_EPOCHS,
    PAIRED_EXPOSURES,
    PAIRED_OPTIMIZER_UPDATES,
    PAIRED_STEPS_PER_EPOCH,
    PAIRS_PER_UPDATE,
    PairedSchedule,
    pair_batch_for_update,
)
from ..train.paired_step import (
    DECODER_STATES_PER_UPDATE,
    FACTUAL_ANCHOR_BATCH_SIZE,
)
from ..train.pools import (
    BranchPools,
    StateExample,
    _draw,
    stack_state_examples,
)
from ..train.step import BranchBatch
from .training_pipeline import (
    PreparedTrainingCatalog,
    build_epoch_branch_pools_from_catalog,
)


PAIRED_FORMAL_SCHEDULE_SCHEMA = "cure-lite-paired-formal-schedule-v1"
PAIRED_FORMAL_ANCHOR_SCHEMA = "cure-lite-paired-formal-anchor-v1"
PAIRED_FORMAL_CATALOG_SCHEMA = "cure-lite-paired-formal-catalog-v1"
PAIRED_FORMAL_BINDING_SCHEMA = "cure-lite-paired-formal-method-binding-v1"

FACTUAL_MISS_STATES_PER_UPDATE = FACTUAL_ANCHOR_BATCH_SIZE
FACTUAL_NO_MISS_STATES_PER_UPDATE = FACTUAL_ANCHOR_BATCH_SIZE
PAIRED_ENDPOINT_STATES_PER_UPDATE = 2 * PAIRS_PER_UPDATE
DECODER_FORWARDS_PER_UPDATE = 3
FORMAL_METHOD_KINDS = ("paired_difference", *CONTROL_KINDS)

if (
    FACTUAL_MISS_STATES_PER_UPDATE
    + FACTUAL_NO_MISS_STATES_PER_UPDATE
    + PAIRED_ENDPOINT_STATES_PER_UPDATE
) != DECODER_STATES_PER_UPDATE:
    raise RuntimeError("formal schedule constants disagree with paired_train_step")


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


def _require_update_coordinates(*, epoch: int, step: int) -> int:
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("epoch must be an integer")
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer")
    if epoch < 0 or epoch >= PAIRED_EPOCHS:
        raise ValueError("epoch must lie in [0, 800)")
    if step < 0 or step >= PAIRED_STEPS_PER_EPOCH:
        raise ValueError("step must lie in [0, 40)")
    return epoch * PAIRED_STEPS_PER_EPOCH + step


def _anchor_identity_payload(
    *,
    branch: str,
    example: StateExample,
) -> dict[str, object]:
    if not isinstance(example, StateExample):
        raise TypeError("factual anchor must be a StateExample")
    supervision = example.supervision
    if supervision.branch != branch:
        raise ValueError(
            f"anchor branch is {supervision.branch!r}, expected {branch!r}"
        )
    return {
        "schema_version": PAIRED_FORMAL_ANCHOR_SCHEMA,
        "branch": branch,
        "sample_id": example.sample_id,
        "positive_gt_ids": list(supervision.positive_gt_ids),
    }


def formal_factual_anchor_id(
    branch: str,
    example: StateExample,
) -> str:
    """Return the canonical source/target identity of one factual anchor."""

    if branch not in {"factual_miss", "factual_no_miss"}:
        raise ValueError("formal factual branch is invalid")
    return stable_fingerprint(
        _anchor_identity_payload(branch=branch, example=example)
    )


@dataclass(frozen=True, eq=False)
class FormalFactualAnchor:
    """One authoritative prepared factual state and its sealed identity."""

    branch: str
    sample_id: str
    positive_gt_ids: tuple[int, ...]
    anchor_id: str
    example: StateExample

    def __post_init__(self) -> None:
        if self.branch not in {"factual_miss", "factual_no_miss"}:
            raise ValueError("formal factual anchor has an invalid branch")
        if not isinstance(self.example, StateExample):
            raise TypeError("example must be a StateExample")
        supervision = self.example.supervision
        if (
            self.sample_id != self.example.sample_id
            or self.branch != supervision.branch
            or self.positive_gt_ids != supervision.positive_gt_ids
        ):
            raise ValueError("formal anchor metadata differs from its example")
        if self.branch == "factual_miss":
            if len(self.positive_gt_ids) != 1:
                raise ValueError(
                    "a formal factual-miss anchor requires one target identity"
                )
        elif self.positive_gt_ids:
            raise ValueError("a factual-no-miss anchor cannot carry a target")
        expected = formal_factual_anchor_id(self.branch, self.example)
        if self.anchor_id != expected:
            raise ValueError("anchor_id does not bind the factual identity")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "branch": self.branch,
            "sample_id": self.sample_id,
            "positive_gt_ids": list(self.positive_gt_ids),
        }


def _formal_anchor(branch: str, example: StateExample) -> FormalFactualAnchor:
    return FormalFactualAnchor(
        branch=branch,
        sample_id=example.sample_id,
        positive_gt_ids=example.supervision.positive_gt_ids,
        anchor_id=formal_factual_anchor_id(branch, example),
        example=example,
    )


def _canonical_anchors(
    branch: str,
    examples: tuple[StateExample, ...],
) -> tuple[FormalFactualAnchor, ...]:
    if not isinstance(examples, tuple) or not examples:
        raise ValueError(f"{branch} expected population must be a non-empty tuple")
    anchors = tuple(_formal_anchor(branch, example) for example in examples)
    ordered = tuple(
        sorted(
            anchors,
            key=lambda anchor: (
                anchor.sample_id,
                anchor.positive_gt_ids,
                anchor.anchor_id,
            ),
        )
    )
    if len({anchor.anchor_id for anchor in ordered}) != len(ordered):
        raise ValueError(f"{branch} expected population has duplicate identities")
    return ordered


def prepared_training_catalog_fingerprint(
    catalog: PreparedTrainingCatalog,
) -> str:
    """Seal the prepared catalog semantics and all formal anchor identities."""

    if not isinstance(catalog, PreparedTrainingCatalog):
        raise TypeError("catalog must be a PreparedTrainingCatalog")
    factual_miss = tuple(
        example
        for entry in catalog.entries
        for example in entry.factual_examples
    )
    factual_no_miss = tuple(
        entry.factual_no_miss_example
        for entry in catalog.entries
        if entry.factual_no_miss_example is not None
    )
    miss_anchors = _canonical_anchors("factual_miss", factual_miss)
    no_miss_anchors = _canonical_anchors(
        "factual_no_miss",
        factual_no_miss,
    )
    return stable_fingerprint(
        {
            "schema_version": PAIRED_FORMAL_CATALOG_SCHEMA,
            "source_ids": list(catalog.source_ids),
            "support_summary": catalog.support_summary.canonical_payload(),
            "semantic_configs": {
                "occupancy": config_to_dict(catalog.occupancy_config),
                "matching": config_to_dict(catalog.match_config),
                "intervention": config_to_dict(
                    catalog.intervention_config
                ),
                "miss_alignment": config_to_dict(
                    catalog.miss_alignment_config
                ),
            },
            "factual_miss_anchors": [
                anchor.canonical_payload() for anchor in miss_anchors
            ],
            "factual_no_miss_anchors": [
                anchor.canonical_payload() for anchor in no_miss_anchors
            ],
        }
    )


@dataclass(frozen=True)
class FormalSourceExposure:
    """Complete pair/factual exposure counts for one source image."""

    sample_id: str
    pair_exposures: int
    factual_miss_exposures: int
    factual_no_miss_exposures: int

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("source exposure sample_id must be non-empty")
        for name in (
            "pair_exposures",
            "factual_miss_exposures",
            "factual_no_miss_exposures",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.total_exposures < 1:
            raise ValueError("a source exposure row cannot be empty")

    @property
    def total_exposures(self) -> int:
        return (
            self.pair_exposures
            + self.factual_miss_exposures
            + self.factual_no_miss_exposures
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "pair_exposures": self.pair_exposures,
            "factual_miss_exposures": self.factual_miss_exposures,
            "factual_no_miss_exposures": self.factual_no_miss_exposures,
            "total_exposures": self.total_exposures,
        }


def _sequence_fingerprints(
    *,
    paired_schedule: PairedSchedule,
    factual_miss_anchors: tuple[FormalFactualAnchor, ...],
    factual_no_miss_anchors: tuple[FormalFactualAnchor, ...],
    factual_miss_indices: tuple[tuple[int, int, int, int], ...],
    factual_no_miss_indices: tuple[tuple[int, int, int, int], ...],
) -> tuple[str, str, str]:
    """Hash all three identity sequences in one deterministic replay."""

    miss_digest = hashlib.sha256(b"cure-lite-formal-factual-miss-sequence-v1")
    no_miss_digest = hashlib.sha256(
        b"cure-lite-formal-factual-no-miss-sequence-v1"
    )
    combined_digest = hashlib.sha256(
        b"cure-lite-formal-combined-sequence-v1"
    )
    for update, (pair_indices, miss_indices, no_miss_indices) in enumerate(
        zip(
            paired_schedule.batch_pair_indices,
            factual_miss_indices,
            factual_no_miss_indices,
            strict=True,
        )
    ):
        update_bytes = update.to_bytes(4, byteorder="big", signed=False)
        miss_ids = tuple(
            bytes.fromhex(factual_miss_anchors[index].anchor_id)
            for index in miss_indices
        )
        no_miss_ids = tuple(
            bytes.fromhex(factual_no_miss_anchors[index].anchor_id)
            for index in no_miss_indices
        )
        pair_ids = tuple(
            bytes.fromhex(paired_schedule.pairs[index].pair_id)
            for index in pair_indices
        )
        miss_digest.update(update_bytes)
        no_miss_digest.update(update_bytes)
        combined_digest.update(update_bytes)
        for value in miss_ids:
            miss_digest.update(value)
            combined_digest.update(value)
        for value in no_miss_ids:
            no_miss_digest.update(value)
            combined_digest.update(value)
        for value in pair_ids:
            combined_digest.update(value)
    return (
        miss_digest.hexdigest(),
        no_miss_digest.hexdigest(),
        combined_digest.hexdigest(),
    )


def _index_exposure_counts(
    indices: tuple[tuple[int, int, int, int], ...],
    *,
    population: int,
) -> tuple[int, ...]:
    counts = Counter(index for selected in indices for index in selected)
    return tuple(counts[index] for index in range(population))


def _source_exposure_ledger(
    *,
    paired_schedule: PairedSchedule,
    factual_miss_anchors: tuple[FormalFactualAnchor, ...],
    factual_no_miss_anchors: tuple[FormalFactualAnchor, ...],
    factual_miss_counts: tuple[int, ...],
    factual_no_miss_counts: tuple[int, ...],
) -> tuple[FormalSourceExposure, ...]:
    pair_counts: Counter[str] = Counter()
    for pair, count in zip(
        paired_schedule.pairs,
        paired_schedule.canonical_cycle_counts,
        strict=True,
    ):
        pair_counts[pair.sample_id] += count
    miss_counts: Counter[str] = Counter()
    for anchor, count in zip(
        factual_miss_anchors,
        factual_miss_counts,
        strict=True,
    ):
        miss_counts[anchor.sample_id] += count
    no_miss_counts: Counter[str] = Counter()
    for anchor, count in zip(
        factual_no_miss_anchors,
        factual_no_miss_counts,
        strict=True,
    ):
        no_miss_counts[anchor.sample_id] += count
    sample_ids = sorted(
        pair_counts.keys() | miss_counts.keys() | no_miss_counts.keys()
    )
    return tuple(
        FormalSourceExposure(
            sample_id=sample_id,
            pair_exposures=pair_counts[sample_id],
            factual_miss_exposures=miss_counts[sample_id],
            factual_no_miss_exposures=no_miss_counts[sample_id],
        )
        for sample_id in sample_ids
    )


def _schedule_payload(
    *,
    seed: int,
    prepared_catalog_fingerprint: str,
    paired_schedule: PairedSchedule,
    factual_miss_anchors: tuple[FormalFactualAnchor, ...],
    factual_no_miss_anchors: tuple[FormalFactualAnchor, ...],
    pair_exposure_counts: tuple[int, ...],
    factual_miss_exposure_counts: tuple[int, ...],
    factual_no_miss_exposure_counts: tuple[int, ...],
    source_exposure_ledger: tuple[FormalSourceExposure, ...],
    pair_sequence_fingerprint: str,
    factual_miss_sequence_fingerprint: str,
    factual_no_miss_sequence_fingerprint: str,
    combined_sequence_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": PAIRED_FORMAL_SCHEDULE_SCHEMA,
        "seed": seed,
        "prepared_catalog_fingerprint": prepared_catalog_fingerprint,
        "paired_catalog_fingerprint": paired_schedule.catalog_fingerprint,
        "paired_schedule_fingerprint": paired_schedule.schedule_fingerprint,
        "budget": {
            "epochs": PAIRED_EPOCHS,
            "steps_per_epoch": PAIRED_STEPS_PER_EPOCH,
            "optimizer_updates": PAIRED_OPTIMIZER_UPDATES,
            "factual_miss_states_per_update": (
                FACTUAL_MISS_STATES_PER_UPDATE
            ),
            "factual_no_miss_states_per_update": (
                FACTUAL_NO_MISS_STATES_PER_UPDATE
            ),
            "clean_pairs_per_update": PAIRS_PER_UPDATE,
            "paired_endpoint_states_per_update": (
                PAIRED_ENDPOINT_STATES_PER_UPDATE
            ),
            "decoder_states_per_update": DECODER_STATES_PER_UPDATE,
            "decoder_forwards_per_update": DECODER_FORWARDS_PER_UPDATE,
            "total_decoder_state_evaluations": (
                PAIRED_OPTIMIZER_UPDATES * DECODER_STATES_PER_UPDATE
            ),
            "total_decoder_forward_calls": (
                PAIRED_OPTIMIZER_UPDATES * DECODER_FORWARDS_PER_UPDATE
            ),
        },
        "selection_contract": {
            "epoch_pool_builder": (
                "build_epoch_branch_pools_from_catalog:factual_only"
            ),
            "within_epoch_draw": (
                "train.pools._draw:stable_hash("
                "branch,epoch,step,draw,global_seed)"
            ),
            "global_seed_equals_paired_schedule_seed": True,
            "same_schedule_for_proposed_and_all_matched_controls": True,
            "method_or_control_label_affects_schedule": False,
        },
        "factual_miss_anchors": [
            anchor.canonical_payload() for anchor in factual_miss_anchors
        ],
        "factual_no_miss_anchors": [
            anchor.canonical_payload()
            for anchor in factual_no_miss_anchors
        ],
        "exposure_ledgers": {
            "pair_counts": list(pair_exposure_counts),
            "factual_miss_counts": list(
                factual_miss_exposure_counts
            ),
            "factual_no_miss_counts": list(
                factual_no_miss_exposure_counts
            ),
            "source_counts": [
                row.canonical_payload() for row in source_exposure_ledger
            ],
            "zero_exposure_pairs": sum(
                count == 0 for count in pair_exposure_counts
            ),
            "zero_exposure_factual_miss_anchors": sum(
                count == 0 for count in factual_miss_exposure_counts
            ),
            "zero_exposure_factual_no_miss_anchors": sum(
                count == 0 for count in factual_no_miss_exposure_counts
            ),
        },
        "sequence_fingerprints": {
            "pair": pair_sequence_fingerprint,
            "factual_miss": factual_miss_sequence_fingerprint,
            "factual_no_miss": factual_no_miss_sequence_fingerprint,
            "combined": combined_sequence_fingerprint,
        },
    }


@dataclass(frozen=True, eq=False)
class PairedFormalSchedule:
    """Complete immutable 800 x 40 pair-and-anchor training schedule."""

    seed: int
    prepared_catalog_fingerprint: str
    paired_schedule: PairedSchedule
    factual_miss_anchors: tuple[FormalFactualAnchor, ...]
    factual_no_miss_anchors: tuple[FormalFactualAnchor, ...]
    factual_miss_indices: tuple[tuple[int, int, int, int], ...]
    factual_no_miss_indices: tuple[tuple[int, int, int, int], ...]
    pair_exposure_counts: tuple[int, ...]
    factual_miss_exposure_counts: tuple[int, ...]
    factual_no_miss_exposure_counts: tuple[int, ...]
    source_exposure_ledger: tuple[FormalSourceExposure, ...]
    pair_sequence_fingerprint: str
    factual_miss_sequence_fingerprint: str
    factual_no_miss_sequence_fingerprint: str
    combined_sequence_fingerprint: str
    schedule_fingerprint: str

    def __post_init__(self) -> None:
        _require_seed(self.seed)
        _require_fingerprint(
            self.prepared_catalog_fingerprint,
            name="prepared_catalog_fingerprint",
        )
        if not isinstance(self.paired_schedule, PairedSchedule):
            raise TypeError("paired_schedule must be a PairedSchedule")
        if self.seed != self.paired_schedule.seed:
            raise ValueError("formal and paired schedule seeds differ")
        if (
            self.pair_sequence_fingerprint
            != self.paired_schedule.sequence_fingerprint
        ):
            raise ValueError("pair sequence differs from PairedSchedule")
        for name in (
            "pair_sequence_fingerprint",
            "factual_miss_sequence_fingerprint",
            "factual_no_miss_sequence_fingerprint",
            "combined_sequence_fingerprint",
            "schedule_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), name=name)

        for branch, anchors in (
            ("factual_miss", self.factual_miss_anchors),
            ("factual_no_miss", self.factual_no_miss_anchors),
        ):
            if not anchors or any(
                not isinstance(anchor, FormalFactualAnchor)
                or anchor.branch != branch
                for anchor in anchors
            ):
                raise ValueError(f"{branch} anchors are invalid")
            expected_order = tuple(
                sorted(
                    anchors,
                    key=lambda anchor: (
                        anchor.sample_id,
                        anchor.positive_gt_ids,
                        anchor.anchor_id,
                    ),
                )
            )
            if anchors != expected_order:
                raise ValueError(f"{branch} anchors are not canonical")
            if len({anchor.anchor_id for anchor in anchors}) != len(anchors):
                raise ValueError(f"{branch} anchor identities are not unique")

        for branch, indices, anchors in (
            (
                "factual_miss",
                self.factual_miss_indices,
                self.factual_miss_anchors,
            ),
            (
                "factual_no_miss",
                self.factual_no_miss_indices,
                self.factual_no_miss_anchors,
            ),
        ):
            if len(indices) != PAIRED_OPTIMIZER_UPDATES:
                raise ValueError(f"{branch} sequence must contain 32,000 updates")
            if any(
                not isinstance(selected, tuple)
                or len(selected) != FACTUAL_ANCHOR_BATCH_SIZE
                for selected in indices
            ):
                raise ValueError(
                    f"{branch} update must select exactly four anchors"
                )
            flattened = [
                index for selected in indices for index in selected
            ]
            if any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(anchors)
                for index in flattened
            ):
                raise ValueError(f"{branch} sequence has an invalid anchor index")

        expected_pair_counts = self.paired_schedule.canonical_cycle_counts
        if self.pair_exposure_counts != expected_pair_counts:
            raise ValueError("formal pair ledger differs from PairedSchedule")
        expected_miss_counts = _index_exposure_counts(
            self.factual_miss_indices,
            population=len(self.factual_miss_anchors),
        )
        expected_no_miss_counts = _index_exposure_counts(
            self.factual_no_miss_indices,
            population=len(self.factual_no_miss_anchors),
        )
        if self.factual_miss_exposure_counts != expected_miss_counts:
            raise ValueError("factual-miss exposure ledger differs from sequence")
        if self.factual_no_miss_exposure_counts != expected_no_miss_counts:
            raise ValueError(
                "factual-no-miss exposure ledger differs from sequence"
            )
        if (
            any(count < 1 for count in self.pair_exposure_counts)
            or any(count < 1 for count in self.factual_miss_exposure_counts)
            or any(
                count < 1
                for count in self.factual_no_miss_exposure_counts
            )
        ):
            raise ValueError("formal schedule omits a pair or factual anchor")
        if sum(self.pair_exposure_counts) != PAIRED_EXPOSURES:
            raise ValueError("formal pair exposure total is not 64,000")
        factual_total = (
            PAIRED_OPTIMIZER_UPDATES * FACTUAL_ANCHOR_BATCH_SIZE
        )
        if sum(self.factual_miss_exposure_counts) != factual_total:
            raise ValueError("factual-miss exposure total is not 128,000")
        if sum(self.factual_no_miss_exposure_counts) != factual_total:
            raise ValueError("factual-no-miss exposure total is not 128,000")

        expected_source_ledger = _source_exposure_ledger(
            paired_schedule=self.paired_schedule,
            factual_miss_anchors=self.factual_miss_anchors,
            factual_no_miss_anchors=self.factual_no_miss_anchors,
            factual_miss_counts=self.factual_miss_exposure_counts,
            factual_no_miss_counts=self.factual_no_miss_exposure_counts,
        )
        if self.source_exposure_ledger != expected_source_ledger:
            raise ValueError("source exposure ledger differs from sequences")
        (
            expected_miss_sequence,
            expected_no_miss_sequence,
            expected_combined,
        ) = _sequence_fingerprints(
            paired_schedule=self.paired_schedule,
            factual_miss_anchors=self.factual_miss_anchors,
            factual_no_miss_anchors=self.factual_no_miss_anchors,
            factual_miss_indices=self.factual_miss_indices,
            factual_no_miss_indices=self.factual_no_miss_indices,
        )
        if self.factual_miss_sequence_fingerprint != expected_miss_sequence:
            raise ValueError("factual-miss sequence fingerprint changed")
        if (
            self.factual_no_miss_sequence_fingerprint
            != expected_no_miss_sequence
        ):
            raise ValueError("factual-no-miss sequence fingerprint changed")
        if self.combined_sequence_fingerprint != expected_combined:
            raise ValueError("combined formal sequence fingerprint changed")
        expected_schedule = stable_fingerprint(self.canonical_payload())
        if self.schedule_fingerprint != expected_schedule:
            raise ValueError("schedule_fingerprint does not bind formal schedule")

    @property
    def optimizer_updates(self) -> int:
        return PAIRED_OPTIMIZER_UPDATES

    @property
    def decoder_state_evaluations(self) -> int:
        return PAIRED_OPTIMIZER_UPDATES * DECODER_STATES_PER_UPDATE

    @property
    def decoder_forward_calls(self) -> int:
        return PAIRED_OPTIMIZER_UPDATES * DECODER_FORWARDS_PER_UPDATE

    def canonical_payload(self) -> dict[str, object]:
        return _schedule_payload(
            seed=self.seed,
            prepared_catalog_fingerprint=(
                self.prepared_catalog_fingerprint
            ),
            paired_schedule=self.paired_schedule,
            factual_miss_anchors=self.factual_miss_anchors,
            factual_no_miss_anchors=self.factual_no_miss_anchors,
            pair_exposure_counts=self.pair_exposure_counts,
            factual_miss_exposure_counts=(
                self.factual_miss_exposure_counts
            ),
            factual_no_miss_exposure_counts=(
                self.factual_no_miss_exposure_counts
            ),
            source_exposure_ledger=self.source_exposure_ledger,
            pair_sequence_fingerprint=self.pair_sequence_fingerprint,
            factual_miss_sequence_fingerprint=(
                self.factual_miss_sequence_fingerprint
            ),
            factual_no_miss_sequence_fingerprint=(
                self.factual_no_miss_sequence_fingerprint
            ),
            combined_sequence_fingerprint=(
                self.combined_sequence_fingerprint
            ),
        )

    def factual_examples_for_update(
        self,
        *,
        epoch: int,
        step: int,
    ) -> dict[str, tuple[StateExample, ...]]:
        update = _require_update_coordinates(epoch=epoch, step=step)
        return {
            "factual_miss": tuple(
                self.factual_miss_anchors[index].example
                for index in self.factual_miss_indices[update]
            ),
            "factual_no_miss": tuple(
                self.factual_no_miss_anchors[index].example
                for index in self.factual_no_miss_indices[update]
            ),
        }


def build_paired_formal_schedule_from_epoch_pool_builder(
    paired_schedule: PairedSchedule,
    *,
    prepared_catalog_fingerprint: str,
    expected_factual_miss: tuple[StateExample, ...],
    expected_factual_no_miss: tuple[StateExample, ...],
    epoch_pool_builder: Callable[[int], BranchPools],
) -> PairedFormalSchedule:
    """Build the full frozen horizon from an injected factual-pool builder.

    This dependency-injected form is useful for focused tests with tiny fake
    populations.  It deliberately exposes no epoch/step/batch-size override:
    tests exercise the same 800 x 40 x (4, 4, 2) constants as formal runs.
    Production callers should normally use :func:`build_paired_formal_schedule`.
    """

    if not isinstance(paired_schedule, PairedSchedule):
        raise TypeError("paired_schedule must be a PairedSchedule")
    prepared_catalog_fingerprint = _require_fingerprint(
        prepared_catalog_fingerprint,
        name="prepared_catalog_fingerprint",
    )
    if not callable(epoch_pool_builder):
        raise TypeError("epoch_pool_builder must be callable")
    miss_anchors = _canonical_anchors(
        "factual_miss",
        expected_factual_miss,
    )
    no_miss_anchors = _canonical_anchors(
        "factual_no_miss",
        expected_factual_no_miss,
    )
    miss_by_id = {anchor.anchor_id: anchor for anchor in miss_anchors}
    no_miss_by_id = {
        anchor.anchor_id: anchor for anchor in no_miss_anchors
    }
    miss_index = {
        anchor.anchor_id: index for index, anchor in enumerate(miss_anchors)
    }
    no_miss_index = {
        anchor.anchor_id: index
        for index, anchor in enumerate(no_miss_anchors)
    }
    miss_sequence: list[tuple[int, int, int, int]] = []
    no_miss_sequence: list[tuple[int, int, int, int]] = []
    for epoch in range(PAIRED_EPOCHS):
        pools = epoch_pool_builder(epoch)
        if not isinstance(pools, BranchPools):
            raise TypeError("epoch_pool_builder must return BranchPools")
        if pools.synthetic:
            raise ValueError("formal factual schedule requires factual_only pools")
        if not pools.factual_miss or not pools.factual_no_miss:
            raise RuntimeError("every formal epoch requires both factual pools")
        for branch, pool, authoritative in (
            ("factual_miss", pools.factual_miss, miss_by_id),
            ("factual_no_miss", pools.factual_no_miss, no_miss_by_id),
        ):
            for example in pool:
                anchor_id = formal_factual_anchor_id(branch, example)
                if anchor_id not in authoritative:
                    raise ValueError(
                        f"{branch} epoch pool contains an undeclared anchor"
                    )
                if example is not authoritative[anchor_id].example:
                    raise ValueError(
                        f"{branch} epoch pool substituted an anchor object"
                    )
        for step in range(PAIRED_STEPS_PER_EPOCH):
            selected_miss = _draw(
                pools.factual_miss,
                FACTUAL_MISS_STATES_PER_UPDATE,
                branch="factual_miss",
                epoch=epoch,
                step=step,
                global_seed=paired_schedule.seed,
            )
            selected_no_miss = _draw(
                pools.factual_no_miss,
                FACTUAL_NO_MISS_STATES_PER_UPDATE,
                branch="factual_no_miss",
                epoch=epoch,
                step=step,
                global_seed=paired_schedule.seed,
            )
            miss_sequence.append(
                tuple(
                    miss_index[
                        formal_factual_anchor_id("factual_miss", example)
                    ]
                    for example in selected_miss
                )
            )
            no_miss_sequence.append(
                tuple(
                    no_miss_index[
                        formal_factual_anchor_id(
                            "factual_no_miss",
                            example,
                        )
                    ]
                    for example in selected_no_miss
                )
            )
    factual_miss_indices = tuple(miss_sequence)
    factual_no_miss_indices = tuple(no_miss_sequence)
    factual_miss_counts = _index_exposure_counts(
        factual_miss_indices,
        population=len(miss_anchors),
    )
    factual_no_miss_counts = _index_exposure_counts(
        factual_no_miss_indices,
        population=len(no_miss_anchors),
    )
    source_ledger = _source_exposure_ledger(
        paired_schedule=paired_schedule,
        factual_miss_anchors=miss_anchors,
        factual_no_miss_anchors=no_miss_anchors,
        factual_miss_counts=factual_miss_counts,
        factual_no_miss_counts=factual_no_miss_counts,
    )
    (
        miss_sequence_fingerprint,
        no_miss_sequence_fingerprint,
        combined_sequence_fingerprint,
    ) = _sequence_fingerprints(
        paired_schedule=paired_schedule,
        factual_miss_anchors=miss_anchors,
        factual_no_miss_anchors=no_miss_anchors,
        factual_miss_indices=factual_miss_indices,
        factual_no_miss_indices=factual_no_miss_indices,
    )
    pair_counts = paired_schedule.canonical_cycle_counts
    payload = _schedule_payload(
        seed=paired_schedule.seed,
        prepared_catalog_fingerprint=prepared_catalog_fingerprint,
        paired_schedule=paired_schedule,
        factual_miss_anchors=miss_anchors,
        factual_no_miss_anchors=no_miss_anchors,
        pair_exposure_counts=pair_counts,
        factual_miss_exposure_counts=factual_miss_counts,
        factual_no_miss_exposure_counts=factual_no_miss_counts,
        source_exposure_ledger=source_ledger,
        pair_sequence_fingerprint=paired_schedule.sequence_fingerprint,
        factual_miss_sequence_fingerprint=(
            miss_sequence_fingerprint
        ),
        factual_no_miss_sequence_fingerprint=(
            no_miss_sequence_fingerprint
        ),
        combined_sequence_fingerprint=combined_sequence_fingerprint,
    )
    return PairedFormalSchedule(
        seed=paired_schedule.seed,
        prepared_catalog_fingerprint=prepared_catalog_fingerprint,
        paired_schedule=paired_schedule,
        factual_miss_anchors=miss_anchors,
        factual_no_miss_anchors=no_miss_anchors,
        factual_miss_indices=factual_miss_indices,
        factual_no_miss_indices=factual_no_miss_indices,
        pair_exposure_counts=pair_counts,
        factual_miss_exposure_counts=factual_miss_counts,
        factual_no_miss_exposure_counts=factual_no_miss_counts,
        source_exposure_ledger=source_ledger,
        pair_sequence_fingerprint=paired_schedule.sequence_fingerprint,
        factual_miss_sequence_fingerprint=(
            miss_sequence_fingerprint
        ),
        factual_no_miss_sequence_fingerprint=(
            no_miss_sequence_fingerprint
        ),
        combined_sequence_fingerprint=combined_sequence_fingerprint,
        schedule_fingerprint=stable_fingerprint(payload),
    )


def build_paired_formal_schedule(
    paired_schedule: PairedSchedule,
    prepared_catalog: PreparedTrainingCatalog,
) -> PairedFormalSchedule:
    """Bind the real prepared ``D_R`` factual pools to ``PairedSchedule``."""

    if not isinstance(paired_schedule, PairedSchedule):
        raise TypeError("paired_schedule must be a PairedSchedule")
    if not isinstance(prepared_catalog, PreparedTrainingCatalog):
        raise TypeError("prepared_catalog must be PreparedTrainingCatalog")
    prepared_sources = set(prepared_catalog.source_ids)
    pair_sources = {pair.sample_id for pair in paired_schedule.pairs}
    if not pair_sources <= prepared_sources:
        raise ValueError(
            "paired schedule contains sources absent from prepared catalog"
        )
    factual_miss = tuple(
        example
        for entry in prepared_catalog.entries
        for example in entry.factual_examples
    )
    factual_no_miss = tuple(
        entry.factual_no_miss_example
        for entry in prepared_catalog.entries
        if entry.factual_no_miss_example is not None
    )
    return build_paired_formal_schedule_from_epoch_pool_builder(
        paired_schedule,
        prepared_catalog_fingerprint=(
            prepared_training_catalog_fingerprint(prepared_catalog)
        ),
        expected_factual_miss=factual_miss,
        expected_factual_no_miss=factual_no_miss,
        epoch_pool_builder=lambda epoch: (
            build_epoch_branch_pools_from_catalog(
                prepared_catalog,
                variant="factual_only",
                epoch=epoch,
                global_seed=paired_schedule.seed,
            )
        ),
    )


def formal_batches_for_update(
    schedule: PairedFormalSchedule,
    *,
    epoch: int,
    step: int,
    device: torch.device | str,
) -> tuple[dict[str, BranchBatch], PairBatch]:
    """Materialize one shared formal update through existing batch APIs."""

    if not isinstance(schedule, PairedFormalSchedule):
        raise TypeError("schedule must be a PairedFormalSchedule")
    factual_examples = schedule.factual_examples_for_update(
        epoch=epoch,
        step=step,
    )
    factual_batches = {
        branch: stack_state_examples(examples, device=device)
        for branch, examples in factual_examples.items()
    }
    pair_batch = pair_batch_for_update(
        schedule.paired_schedule,
        epoch=epoch,
        step=step,
        device=device,
    )
    return factual_batches, pair_batch


@dataclass(frozen=True, eq=False)
class PairedFormalMethodBinding:
    """A method label bound to one shared, method-independent schedule."""

    method_kind: str
    schedule: PairedFormalSchedule
    binding_fingerprint: str

    def __post_init__(self) -> None:
        if self.method_kind not in FORMAL_METHOD_KINDS:
            raise ValueError(
                f"method_kind must be one of {FORMAL_METHOD_KINDS}"
            )
        if not isinstance(self.schedule, PairedFormalSchedule):
            raise TypeError("schedule must be a PairedFormalSchedule")
        expected = stable_fingerprint(self.canonical_payload())
        if self.binding_fingerprint != expected:
            raise ValueError("binding_fingerprint changed")

    @property
    def shared_schedule_fingerprint(self) -> str:
        return self.schedule.schedule_fingerprint

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PAIRED_FORMAL_BINDING_SCHEMA,
            "method_kind": self.method_kind,
            "shared_schedule_fingerprint": (
                self.schedule.schedule_fingerprint
            ),
            "method_label_affects_schedule": False,
        }


def bind_paired_formal_schedule(
    schedule: PairedFormalSchedule,
    *,
    method_kind: str,
) -> PairedFormalMethodBinding:
    """Bind a label without rebuilding or changing the shared schedule."""

    if not isinstance(schedule, PairedFormalSchedule):
        raise TypeError("schedule must be a PairedFormalSchedule")
    payload = {
        "schema_version": PAIRED_FORMAL_BINDING_SCHEMA,
        "method_kind": method_kind,
        "shared_schedule_fingerprint": schedule.schedule_fingerprint,
        "method_label_affects_schedule": False,
    }
    return PairedFormalMethodBinding(
        method_kind=method_kind,
        schedule=schedule,
        binding_fingerprint=stable_fingerprint(payload),
    )


__all__ = [
    "DECODER_FORWARDS_PER_UPDATE",
    "FACTUAL_MISS_STATES_PER_UPDATE",
    "FACTUAL_NO_MISS_STATES_PER_UPDATE",
    "FORMAL_METHOD_KINDS",
    "PAIRED_ENDPOINT_STATES_PER_UPDATE",
    "PAIRED_FORMAL_ANCHOR_SCHEMA",
    "PAIRED_FORMAL_BINDING_SCHEMA",
    "PAIRED_FORMAL_CATALOG_SCHEMA",
    "PAIRED_FORMAL_SCHEDULE_SCHEMA",
    "FormalFactualAnchor",
    "FormalSourceExposure",
    "PairedFormalMethodBinding",
    "PairedFormalSchedule",
    "bind_paired_formal_schedule",
    "build_paired_formal_schedule",
    "build_paired_formal_schedule_from_epoch_pool_builder",
    "formal_batches_for_update",
    "formal_factual_anchor_id",
    "prepared_training_catalog_fingerprint",
]
