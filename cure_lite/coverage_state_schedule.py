"""Deterministic objective-invariant schedule for scalar CSLF training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property, lru_cache
from math import fsum
from types import MappingProxyType
from typing import Mapping

from .cache.schema import stable_fingerprint
from .coverage_state_batches import (
    COVERAGE_STATE_FUSED_BATCH_SCHEMA,
    CoverageStateFusedBatch,
    make_coverage_state_natural_train_batch,
    make_coverage_state_pair_train_batch,
)
from .coverage_state_precomputed_cache import (
    CoverageStateCachedNatural,
    CoverageStateCachedPair,
    CoverageStateScalarCache,
)
from .sampling import stable_hash


COVERAGE_STATE_SCHEDULE_SCHEMA = "cure-lite-cslf-training-schedule-v1"
COVERAGE_STATE_FORMAL_EPOCHS = 800
COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH = 40
COVERAGE_STATE_EXPOSURE_GATE_POLICY = (
    "branch_stratified_record_source_and_role_target_exposure_gate_v1"
)
COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION = 0.90
COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE = 2.0
COVERAGE_STATE_SOURCE_MINIMUM_ESS_FRACTION = 0.50
COVERAGE_STATE_SOURCE_MAXIMUM_UNIFORM_MULTIPLE = 4.0


def _exposure_gate_policy_payload() -> dict[str, object]:
    """Return the frozen distinction between gates and descriptions."""

    return {
        "policy": COVERAGE_STATE_EXPOSURE_GATE_POLICY,
        "record_minimum_ess_fraction_hex": (
            COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION.hex()
        ),
        "record_maximum_uniform_multiple_hex": (
            COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE.hex()
        ),
        "source_minimum_ess_fraction_hex": (
            COVERAGE_STATE_SOURCE_MINIMUM_ESS_FRACTION.hex()
        ),
        "source_maximum_uniform_multiple_hex": (
            COVERAGE_STATE_SOURCE_MAXIMUM_UNIFORM_MULTIPLE.hex()
        ),
        "zero_exposure_count": 0,
        "formal_gate_statistics": [
            "branch/record",
            "branch/source",
            "factual_focus_target",
            "clean_added_target",
        ],
        "descriptive_only_statistics": [
            "positive_target",
            "logical_state_source",
        ],
        "target_roles_are_not_pooled_for_formal_gating": True,
    }


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class CoverageStateScheduleConfig:
    seed: int
    epochs: int
    steps_per_epoch: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        object.__setattr__(
            self,
            "epochs",
            _positive_int(self.epochs, name="epochs"),
        )
        object.__setattr__(
            self,
            "steps_per_epoch",
            _positive_int(
                self.steps_per_epoch,
                name="steps_per_epoch",
            ),
        )

    @property
    def updates(self) -> int:
        return self.epochs * self.steps_per_epoch

    @classmethod
    def formal(cls, *, seed: int) -> "CoverageStateScheduleConfig":
        return cls(
            seed=seed,
            epochs=COVERAGE_STATE_FORMAL_EPOCHS,
            steps_per_epoch=COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH,
        )


@dataclass(frozen=True)
class CoverageStateUpdateSelection:
    epoch: int
    step: int
    factual_miss_record_ids: tuple[str, str, str, str]
    factual_no_miss_record_ids: tuple[str, str, str, str]
    clean_positive_pair_id: str
    component_null_pair_id: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "factual_miss_record_ids": list(
                self.factual_miss_record_ids
            ),
            "factual_no_miss_record_ids": list(
                self.factual_no_miss_record_ids
            ),
            "clean_positive_pair_id": self.clean_positive_pair_id,
            "component_null_pair_id": self.component_null_pair_id,
        }

    @property
    def selection_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


@dataclass(frozen=True)
class CoverageStateTrainingSchedule:
    cache_fingerprint: str
    config: CoverageStateScheduleConfig
    selections: tuple[CoverageStateUpdateSelection, ...]

    def __post_init__(self) -> None:
        if len(self.selections) != self.config.updates:
            raise ValueError("schedule selection count differs from horizon")
        for update, selection in enumerate(self.selections):
            epoch, step = divmod(
                update,
                self.config.steps_per_epoch,
            )
            if selection.epoch != epoch or selection.step != step:
                raise ValueError(
                    "schedule selections must be complete and update ordered"
                )

    @cached_property
    def schedule_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def exposure_counts(self) -> dict[str, dict[str, int]]:
        counters = {
            "factual_miss": Counter[str](),
            "factual_no_miss": Counter[str](),
            "clean_positive": Counter[str](),
            "component_null": Counter[str](),
        }
        for value in self.selections:
            counters["factual_miss"].update(
                value.factual_miss_record_ids
            )
            counters["factual_no_miss"].update(
                value.factual_no_miss_record_ids
            )
            counters["clean_positive"][value.clean_positive_pair_id] += 1
            counters["component_null"][value.component_null_pair_id] += 1
        return {
            name: dict(sorted(values.items()))
            for name, values in counters.items()
        }

    def canonical_payload(self) -> dict[str, object]:
        exposure = self.exposure_counts()
        return {
            "schema_version": COVERAGE_STATE_SCHEDULE_SCHEMA,
            "cache_fingerprint": self.cache_fingerprint,
            "seed": self.config.seed,
            "epochs": self.config.epochs,
            "steps_per_epoch": self.config.steps_per_epoch,
            "updates": self.config.updates,
            "logical_states_per_update": 12,
            "objective_invariant": True,
            "optimizer_exposure_accounting": (
                "recomputed_against_current_cache_before_use"
            ),
            "exposure_counts": exposure,
            "selection_sequence_fingerprint": stable_fingerprint(
                [
                    value.canonical_payload()
                    for value in self.selections
                ]
            ),
        }


@lru_cache(maxsize=16)
def _cache_record_lookups(
    cache: CoverageStateScalarCache,
) -> tuple[
    Mapping[str, CoverageStateCachedNatural],
    Mapping[str, CoverageStateCachedPair],
]:
    """Build immutable ID lookups once per process-local scalar cache."""

    naturals = {
        value.record.record_id: value
        for value in cache.natural_records
    }
    pairs = {
        value.record.pair_id: value
        for value in cache.pair_records
    }
    if (
        len(naturals) != len(cache.natural_records)
        or len(pairs) != len(cache.pair_records)
    ):
        raise ValueError("scalar cache record identities are not unique")
    return MappingProxyType(naturals), MappingProxyType(pairs)


def _base_ranked(
    values: tuple[
        CoverageStateCachedNatural | CoverageStateCachedPair,
        ...,
    ],
    *,
    namespace: str,
    seed: int,
) -> tuple[
    CoverageStateCachedNatural | CoverageStateCachedPair,
    ...,
]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                stable_hash(
                    COVERAGE_STATE_SCHEDULE_SCHEMA,
                    namespace,
                    seed,
                    (
                        value.record.record_id
                        if isinstance(value, CoverageStateCachedNatural)
                        else value.record.pair_id
                    ),
                ),
                (
                    value.record.record_id
                    if isinstance(value, CoverageStateCachedNatural)
                    else value.record.pair_id
                ),
            ),
        )
    )


def _build_natural_selection_sequence(
    ordered: tuple[CoverageStateCachedNatural, ...],
    *,
    namespace: str,
    updates: int,
) -> tuple[
    tuple[
        CoverageStateCachedNatural,
        CoverageStateCachedNatural,
        CoverageStateCachedNatural,
        CoverageStateCachedNatural,
    ],
    ...,
]:
    """Build a record-balanced sequence under a four-input uniqueness rule.

    A stateless ``(update * 4) % N`` cursor can repeatedly skip one focus
    record when several records share the same actual model input.  The
    persistent exposure ledger below instead selects the least-exposed
    eligible record at every slot.  Seed-specific base rank breaks ties, while
    per-update input exclusion preserves the fixed 4-state diversity.
    """

    if not ordered:
        raise ValueError(f"{namespace} pool cannot be empty")
    updates = _positive_int(updates, name=f"{namespace}_updates")
    input_counts = Counter(
        value.actual_scalar_input_fingerprint for value in ordered
    )
    if len(input_counts) < 4:
        raise ValueError(
            f"{namespace} pool lacks four unique actual inputs"
        )
    if max(input_counts.values()) * 4 > len(ordered):
        raise ValueError(
            f"{namespace} population cannot support record-uniform exposure "
            "with four unique actual inputs per update"
        )
    rank = {
        value.record.record_id: index
        for index, value in enumerate(ordered)
    }
    exposure = {
        value.record.record_id: 0 for value in ordered
    }
    sequence: list[
        tuple[
            CoverageStateCachedNatural,
            CoverageStateCachedNatural,
            CoverageStateCachedNatural,
            CoverageStateCachedNatural,
        ]
    ] = []
    for _ in range(updates):
        selected: list[CoverageStateCachedNatural] = []
        actual_inputs: set[str] = set()
        for _slot in range(4):
            eligible = (
                value
                for value in ordered
                if value.actual_scalar_input_fingerprint
                not in actual_inputs
            )
            try:
                value = min(
                    eligible,
                    key=lambda candidate: (
                        exposure[candidate.record.record_id],
                        rank[candidate.record.record_id],
                        candidate.record.record_id,
                    ),
                )
            except ValueError as error:
                raise ValueError(
                    f"{namespace} cannot construct a four-input update"
                ) from error
            selected.append(value)
            actual_inputs.add(value.actual_scalar_input_fingerprint)
            exposure[value.record.record_id] += 1
        sequence.append(tuple(selected))  # type: ignore[arg-type]
    return tuple(sequence)


def build_coverage_state_training_schedule(
    cache: CoverageStateScalarCache,
    config: CoverageStateScheduleConfig,
) -> CoverageStateTrainingSchedule:
    """Freeze all selections before any objective-specific model is created."""

    if not isinstance(cache, CoverageStateScalarCache):
        raise TypeError("cache must be CoverageStateScalarCache")
    if not isinstance(config, CoverageStateScheduleConfig):
        raise TypeError("config must be CoverageStateScheduleConfig")
    cache.verify_unchanged()
    miss = tuple(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_miss"
    )
    no_miss = tuple(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_no_miss"
    )
    clean = cache.clean_positive_records
    component = cache.component_null_records
    if (
        len({value.actual_scalar_input_fingerprint for value in miss}) < 4
        or len(
            {
                value.actual_scalar_input_fingerprint
                for value in no_miss
            }
        )
        < 4
        or not clean
        or not component
    ):
        raise ValueError("cache population is insufficient for fixed schedule")

    ordered_miss = tuple(
        value
        for value in _base_ranked(
            miss,
            namespace="factual_miss",
            seed=config.seed,
        )
        if isinstance(value, CoverageStateCachedNatural)
    )
    ordered_no_miss = tuple(
        value
        for value in _base_ranked(
            no_miss,
            namespace="factual_no_miss",
            seed=config.seed,
        )
        if isinstance(value, CoverageStateCachedNatural)
    )
    ordered_clean = tuple(
        value
        for value in _base_ranked(
            clean,
            namespace="clean_positive",
            seed=config.seed,
        )
        if isinstance(value, CoverageStateCachedPair)
    )
    ordered_component = tuple(
        value
        for value in _base_ranked(
            component,
            namespace="component_null",
            seed=config.seed,
        )
        if isinstance(value, CoverageStateCachedPair)
    )
    miss_sequence = _build_natural_selection_sequence(
        ordered_miss,
        namespace="factual_miss",
        updates=config.updates,
    )
    no_miss_sequence = _build_natural_selection_sequence(
        ordered_no_miss,
        namespace="factual_no_miss",
        updates=config.updates,
    )
    selections: list[CoverageStateUpdateSelection] = []
    for epoch in range(config.epochs):
        for step in range(config.steps_per_epoch):
            update = epoch * config.steps_per_epoch + step
            selected_miss = miss_sequence[update]
            selected_no_miss = no_miss_sequence[update]
            selected_clean = ordered_clean[
                update % len(ordered_clean)
            ]
            selected_component: CoverageStateCachedPair | None = None
            start = update % len(ordered_component)
            for offset in range(len(ordered_component)):
                value = ordered_component[
                    (start + offset) % len(ordered_component)
                ]
                if (
                    value.record.sample_id
                    != selected_clean.record.sample_id
                ):
                    selected_component = value
                    break
            if selected_component is None:
                raise ValueError(
                    "component-null pool cannot avoid the clean source"
                )
            selections.append(
                CoverageStateUpdateSelection(
                    epoch=epoch,
                    step=step,
                    factual_miss_record_ids=tuple(
                        value.record.record_id
                        for value in selected_miss
                    ),
                    factual_no_miss_record_ids=tuple(
                        value.record.record_id
                        for value in selected_no_miss
                    ),
                    clean_positive_pair_id=selected_clean.record.pair_id,
                    component_null_pair_id=(
                        selected_component.record.pair_id
                    ),
                )
            )
    return CoverageStateTrainingSchedule(
        cache_fingerprint=cache.cache_fingerprint,
        config=config,
        selections=tuple(selections),
    )


def materialize_coverage_state_fused_batch(
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    epoch: int,
    step: int,
    device: str,
    _cache_already_verified: bool = False,
) -> CoverageStateFusedBatch:
    """Materialize one predeclared update without rebuilding geometry."""

    if not isinstance(cache, CoverageStateScalarCache):
        raise TypeError("cache must be CoverageStateScalarCache")
    if not isinstance(schedule, CoverageStateTrainingSchedule):
        raise TypeError("schedule must be CoverageStateTrainingSchedule")
    if not isinstance(_cache_already_verified, bool):
        raise TypeError("_cache_already_verified must be bool")
    if not _cache_already_verified:
        cache.verify_unchanged()
    if cache.cache_fingerprint != schedule.cache_fingerprint:
        raise ValueError("schedule and scalar cache differ")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
        or epoch >= schedule.config.epochs
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or step >= schedule.config.steps_per_epoch
    ):
        raise ValueError("epoch/step lies outside the schedule")
    selection = schedule.selections[
        epoch * schedule.config.steps_per_epoch + step
    ]
    naturals, pairs = _cache_record_lookups(cache)
    miss = tuple(
        naturals[value] for value in selection.factual_miss_record_ids
    )
    no_miss = tuple(
        naturals[value]
        for value in selection.factual_no_miss_record_ids
    )
    result = CoverageStateFusedBatch(
        factual_miss=make_coverage_state_natural_train_batch(
            miss,
            state_kind="factual_miss",
            device=device,
            validate=not _cache_already_verified,
        ),
        factual_no_miss=make_coverage_state_natural_train_batch(
            no_miss,
            state_kind="factual_no_miss",
            device=device,
            validate=not _cache_already_verified,
        ),
        pairs=make_coverage_state_pair_train_batch(
            pairs[selection.clean_positive_pair_id],
            pairs[selection.component_null_pair_id],
            device=device,
            validate=not _cache_already_verified,
        ),
    )
    if not _cache_already_verified:
        result.validate()
    if result.selection_fingerprint != stable_fingerprint(
        {
            "schema_version": COVERAGE_STATE_FUSED_BATCH_SCHEMA,
            "factual_miss_record_ids": list(
                selection.factual_miss_record_ids
            ),
            "factual_no_miss_record_ids": list(
                selection.factual_no_miss_record_ids
            ),
            "pair_ids": [
                selection.clean_positive_pair_id,
                selection.component_null_pair_id,
            ],
            "pair_kinds": [
                "clean_positive",
                "component_null",
            ],
            "input_order": [
                "factual_miss",
                "factual_no_miss",
                "pair_plus",
                "pair_minus",
            ],
        }
    ):
        raise AssertionError("materialized batch differs from schedule")
    return result


def _audit_schedule_selections(
    schedule: CoverageStateTrainingSchedule,
    natural: Mapping[str, CoverageStateCachedNatural],
    pairs: Mapping[str, CoverageStateCachedPair],
) -> dict[str, object]:
    """Validate every scheduled identity against the current cache universe."""

    branch_totals = Counter[str]()
    identity_exposure = 0
    diagnostic_exposure = 0
    logical_states = 0
    for selection in schedule.selections:
        for branch, identities in (
            ("factual_miss", selection.factual_miss_record_ids),
            (
                "factual_no_miss",
                selection.factual_no_miss_record_ids,
            ),
        ):
            if len(identities) != 4 or len(set(identities)) != 4:
                raise ValueError(
                    f"{branch} selection must contain four unique records"
                )
            try:
                records = tuple(natural[identity] for identity in identities)
            except KeyError as error:
                raise ValueError(
                    f"{branch} selection contains an ID outside the cache"
                ) from error
            if any(
                value.record.state_kind != branch for value in records
            ):
                raise ValueError(
                    f"{branch} selection contains the wrong natural role"
                )
            if len(
                {
                    value.actual_scalar_input_fingerprint
                    for value in records
                }
            ) != 4:
                raise ValueError(
                    f"{branch} selection repeats an actual model input"
                )
            branch_totals[branch] += 4
            logical_states += 4

        try:
            clean = pairs[selection.clean_positive_pair_id]
            component = pairs[selection.component_null_pair_id]
        except KeyError as error:
            raise ValueError(
                "pair selection contains an ID outside the cache"
            ) from error
        for value in (clean, component):
            if value.optimizer_role == "identity_diagnostic":
                identity_exposure += 1
            elif value.optimizer_role == "diagnostic_only":
                diagnostic_exposure += 1
        if clean.optimizer_role != "clean_positive":
            raise ValueError(
                "clean-positive slot contains the wrong pair role"
            )
        if component.optimizer_role != "component_null":
            raise ValueError(
                "component-null slot contains the wrong pair role"
            )
        if clean.record.sample_id == component.record.sample_id:
            raise ValueError(
                "clean and component-null selections share a source"
            )
        branch_totals["clean_positive"] += 1
        branch_totals["component_null"] += 1
        logical_states += 4

    updates = schedule.config.updates
    expected_branch_totals = {
        "factual_miss": updates * 4,
        "factual_no_miss": updates * 4,
        "clean_positive": updates,
        "component_null": updates,
    }
    actual_branch_totals = {
        name: int(branch_totals.get(name, 0))
        for name in expected_branch_totals
    }
    expected_logical_states = updates * 12
    if (
        actual_branch_totals != expected_branch_totals
        or logical_states != expected_logical_states
    ):
        raise ValueError("schedule exposure totals differ from the fixed budget")
    return {
        "selection_count": len(schedule.selections),
        "branch_exposure_totals": actual_branch_totals,
        "logical_state_evaluations": logical_states,
        "identity_null_optimizer_exposure": identity_exposure,
        "diagnostic_only_optimizer_exposure": diagnostic_exposure,
        "optimizer_exposure_unit": "pair_selection",
        "exact_budget_closed": True,
    }


def _exposure_statistics(
    universe: tuple[str, ...],
    counts: Counter[str],
) -> dict[str, object]:
    if not universe or universe != tuple(sorted(set(universe))):
        raise ValueError("exposure universe must be sorted, unique, nonempty")
    values = [int(counts.get(identity, 0)) for identity in universe]
    total = sum(values)
    if total < 1:
        raise ValueError("exposure count total must be positive")
    probabilities = [value / float(total) for value in values]
    ess = 1.0 / fsum(value * value for value in probabilities)
    ordered = sorted(values, reverse=True)
    return {
        "support_size": len(universe),
        "total_exposures": total,
        "zero_exposure_count": sum(value == 0 for value in values),
        "minimum_count": min(values),
        "maximum_count": max(values),
        "maximum_share": max(probabilities),
        "top5_share": sum(ordered[:5]) / float(total),
        "empirical_ess": ess,
        "counts": {
            identity: int(counts.get(identity, 0))
            for identity in universe
        },
    }


def coverage_state_schedule_exposure_report(
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
) -> dict[str, object]:
    """Report target/record and source exposure before any training starts."""

    if not isinstance(cache, CoverageStateScalarCache):
        raise TypeError("cache must be CoverageStateScalarCache")
    if not isinstance(schedule, CoverageStateTrainingSchedule):
        raise TypeError("schedule must be CoverageStateTrainingSchedule")
    cache.verify_unchanged()
    if cache.cache_fingerprint != schedule.cache_fingerprint:
        raise ValueError("schedule and scalar cache differ")
    natural, pairs = _cache_record_lookups(cache)
    selection_audit = _audit_schedule_selections(
        schedule,
        natural,
        pairs,
    )
    branch_counts = {
        "factual_miss": Counter[str](),
        "factual_no_miss": Counter[str](),
        "clean_positive": Counter[str](),
        "component_null": Counter[str](),
    }
    branch_sources = {
        name: Counter[str]() for name in branch_counts
    }
    factual_target_counts: Counter[str] = Counter()
    clean_target_counts: Counter[str] = Counter()
    logical_source_counts: Counter[str] = Counter()
    for selection in schedule.selections:
        for branch, identities in (
            ("factual_miss", selection.factual_miss_record_ids),
            (
                "factual_no_miss",
                selection.factual_no_miss_record_ids,
            ),
        ):
            for identity in identities:
                record = natural[identity].record
                branch_counts[branch][identity] += 1
                branch_sources[branch][record.sample_id] += 1
                logical_source_counts[record.sample_id] += 1
                if branch == "factual_miss":
                    if len(record.focus_target_ids) != 1:
                        raise ValueError(
                            "factual miss schedule lost one-target focus"
                        )
                    factual_target_counts[
                        f"factual:{record.sample_id}:"
                        f"{record.focus_target_ids[0]}"
                    ] += 1
        for branch, identity in (
            ("clean_positive", selection.clean_positive_pair_id),
            ("component_null", selection.component_null_pair_id),
        ):
            record = pairs[identity].record
            branch_counts[branch][identity] += 1
            branch_sources[branch][record.sample_id] += 1
            logical_source_counts[record.sample_id] += 2
            if branch == "clean_positive":
                if len(record.target_ids_added) != 1:
                    raise ValueError(
                        "clean schedule lost one-target lineage"
                    )
                clean_target_counts[
                    f"clean:{record.sample_id}:"
                    f"{record.target_ids_added[0]}"
                ] += 1

    branch_universes = {
        "factual_miss": tuple(
            sorted(
                value.record.record_id
                for value in cache.natural_records
                if value.record.state_kind == "factual_miss"
            )
        ),
        "factual_no_miss": tuple(
            sorted(
                value.record.record_id
                for value in cache.natural_records
                if value.record.state_kind == "factual_no_miss"
            )
        ),
        "clean_positive": tuple(
            sorted(
                value.record.pair_id
                for value in cache.clean_positive_records
            )
        ),
        "component_null": tuple(
            sorted(
                value.record.pair_id
                for value in cache.component_null_records
            )
        ),
    }
    source_universes = {
        branch: tuple(
            sorted(
                {
                    (
                        natural[identity].record.sample_id
                        if branch
                        in {"factual_miss", "factual_no_miss"}
                        else pairs[identity].record.sample_id
                    )
                    for identity in universe
                }
            )
        )
        for branch, universe in branch_universes.items()
    }
    factual_target_universe = tuple(
        sorted(
            {
                f"factual:{value.record.sample_id}:"
                f"{value.record.focus_target_ids[0]}"
                for value in cache.natural_records
                if value.record.state_kind == "factual_miss"
            }
        )
    )
    clean_target_universe = tuple(
        sorted(
            {
                f"clean:{value.record.sample_id}:"
                f"{value.record.target_ids_added[0]}"
                for value in cache.clean_positive_records
            }
        )
    )
    target_universe = tuple(
        sorted(
            {
                *factual_target_universe,
                *clean_target_universe,
            }
        )
    )
    target_counts = factual_target_counts + clean_target_counts
    logical_source_universe = tuple(
        sorted(
            {
                value.record.sample_id
                for value in cache.natural_records
            }
            | {
                value.record.sample_id
                for value in cache.clean_positive_records
            }
            | {
                value.record.sample_id
                for value in cache.component_null_records
            }
        )
    )
    payload: dict[str, object] = {
        "schema_version": "cure-lite-cslf-schedule-exposure-v1",
        "cache_fingerprint": cache.cache_fingerprint,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "seed": schedule.config.seed,
        "epochs": schedule.config.epochs,
        "steps_per_epoch": schedule.config.steps_per_epoch,
        "updates": schedule.config.updates,
        "gate_policy": _exposure_gate_policy_payload(),
        "selection_audit": selection_audit,
        "branches": {
            branch: {
                "record": _exposure_statistics(
                    branch_universes[branch],
                    branch_counts[branch],
                ),
                "source": _exposure_statistics(
                    source_universes[branch],
                    branch_sources[branch],
                ),
            }
            for branch in branch_counts
        },
        "factual_focus_target": _exposure_statistics(
            factual_target_universe,
            factual_target_counts,
        ),
        "clean_added_target": _exposure_statistics(
            clean_target_universe,
            clean_target_counts,
        ),
        "positive_target": {
            **_exposure_statistics(
                target_universe,
                target_counts,
            ),
            "formal_gate_role": "descriptive_only",
        },
        "logical_state_source": {
            **_exposure_statistics(
                logical_source_universe,
                logical_source_counts,
            ),
            "formal_gate_role": "descriptive_only",
        },
        "identity_null_optimizer_exposure": selection_audit[
            "identity_null_optimizer_exposure"
        ],
        "diagnostic_only_optimizer_exposure": selection_audit[
            "diagnostic_only_optimizer_exposure"
        ],
    }
    payload["report_fingerprint"] = stable_fingerprint(payload)
    return payload


def coverage_state_formal_exposure_gate(
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
) -> dict[str, object]:
    """Apply the frozen support-preserving gate to a full 800 x 40 schedule."""

    if (
        schedule.config.epochs != COVERAGE_STATE_FORMAL_EPOCHS
        or schedule.config.steps_per_epoch
        != COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH
    ):
        raise ValueError("formal exposure gate requires an 800 x 40 schedule")
    report = coverage_state_schedule_exposure_report(cache, schedule)
    gate_policy = _exposure_gate_policy_payload()
    checks: dict[str, bool] = {}

    def check_statistics(
        name: str,
        statistics: dict[str, object],
        *,
        minimum_ess_fraction: float,
        maximum_uniform_multiple: float,
    ) -> None:
        support = int(statistics["support_size"])
        checks[f"{name}/zero_exposure"] = (
            int(statistics["zero_exposure_count"]) == 0
        )
        checks[f"{name}/ess"] = (
            float(statistics["empirical_ess"])
            >= minimum_ess_fraction * support
        )
        checks[f"{name}/maximum_share"] = (
            float(statistics["maximum_share"])
            <= maximum_uniform_multiple / float(support)
        )

    for branch, payload in report["branches"].items():
        check_statistics(
            f"{branch}/record",
            payload["record"],
            minimum_ess_fraction=(
                COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION
            ),
            maximum_uniform_multiple=(
                COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE
            ),
        )
        check_statistics(
            f"{branch}/source",
            payload["source"],
            minimum_ess_fraction=(
                COVERAGE_STATE_SOURCE_MINIMUM_ESS_FRACTION
            ),
            maximum_uniform_multiple=(
                COVERAGE_STATE_SOURCE_MAXIMUM_UNIFORM_MULTIPLE
            ),
        )
    check_statistics(
        "factual_focus_target",
        report["factual_focus_target"],
        minimum_ess_fraction=(
            COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION
        ),
        maximum_uniform_multiple=(
            COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE
        ),
    )
    check_statistics(
        "clean_added_target",
        report["clean_added_target"],
        minimum_ess_fraction=(
            COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION
        ),
        maximum_uniform_multiple=(
            COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE
        ),
    )
    checks["selection_exact_budget"] = bool(
        report["selection_audit"]["exact_budget_closed"]
    )
    checks["identity_null_optimizer_exposure"] = (
        report["identity_null_optimizer_exposure"] == 0
    )
    checks["diagnostic_only_optimizer_exposure"] = (
        report["diagnostic_only_optimizer_exposure"] == 0
    )
    result: dict[str, object] = {
        "schema_version": "cure-lite-cslf-formal-exposure-gate-v1",
        "cache_fingerprint": cache.cache_fingerprint,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "gate_policy": gate_policy,
        "gate_policy_fingerprint": stable_fingerprint(gate_policy),
        "thresholds": {
            "record_minimum_ess_fraction": (
                COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION
            ),
            "record_maximum_uniform_multiple": (
                COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE
            ),
            "source_minimum_ess_fraction": (
                COVERAGE_STATE_SOURCE_MINIMUM_ESS_FRACTION
            ),
            "source_maximum_uniform_multiple": (
                COVERAGE_STATE_SOURCE_MAXIMUM_UNIFORM_MULTIPLE
            ),
            "zero_exposure_count": 0,
        },
        "checks": checks,
        "failed_checks": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "all_pass": all(checks.values()),
        "report": report,
        "training_authorized": False,
        "formal_training_authorized": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    result["gate_fingerprint"] = stable_fingerprint(result)
    return result


__all__ = [
    "COVERAGE_STATE_EXPOSURE_GATE_POLICY",
    "COVERAGE_STATE_FORMAL_EPOCHS",
    "COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH",
    "COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE",
    "COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION",
    "COVERAGE_STATE_SCHEDULE_SCHEMA",
    "COVERAGE_STATE_SOURCE_MAXIMUM_UNIFORM_MULTIPLE",
    "COVERAGE_STATE_SOURCE_MINIMUM_ESS_FRACTION",
    "CoverageStateScheduleConfig",
    "CoverageStateTrainingSchedule",
    "CoverageStateUpdateSelection",
    "build_coverage_state_training_schedule",
    "coverage_state_formal_exposure_gate",
    "coverage_state_schedule_exposure_report",
    "materialize_coverage_state_fused_batch",
]
