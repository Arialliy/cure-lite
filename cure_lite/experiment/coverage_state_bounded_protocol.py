"""Deterministic D_R-only bounded population for CSLF pre-formal training."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property

from ..cache.schema import stable_fingerprint
from ..coverage_state_observability import (
    CoverageStateObservabilityDecision,
    audit_population_observability,
)
from ..coverage_state_precomputed_cache import (
    CoverageStateCachedNatural,
    CoverageStateCachedPair,
    CoverageStateScalarCache,
)
from ..coverage_state_raw_catalog import make_coverage_state_raw_catalog
from ..coverage_state_schedule import (
    COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE,
    COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION,
    COVERAGE_STATE_SOURCE_MAXIMUM_UNIFORM_MULTIPLE,
    COVERAGE_STATE_SOURCE_MINIMUM_ESS_FRACTION,
    CoverageStateScheduleConfig,
    CoverageStateTrainingSchedule,
    build_coverage_state_training_schedule,
    coverage_state_schedule_exposure_report,
)
from ..sampling import stable_hash


COVERAGE_STATE_BOUNDED_POPULATION_SCHEMA = (
    "cure-lite-cslf-dr-bounded-population-v1"
)
COVERAGE_STATE_BOUNDED_SELECTION_POLICY = (
    "deterministic_source_round_robin_then_record_rank_v1"
)
COVERAGE_STATE_BOUNDED_SEED = 42
COVERAGE_STATE_BOUNDED_ROLE_COUNT = 16
COVERAGE_STATE_BOUNDED_UPDATES = 400
COVERAGE_STATE_BOUNDED_EPOCHS = 10
COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH = 40


def _identity(
    value: CoverageStateCachedNatural | CoverageStateCachedPair,
) -> str:
    if isinstance(value, CoverageStateCachedNatural):
        return value.record.record_id
    return value.record.pair_id


def _source_balanced_select(
    values: tuple[
        CoverageStateCachedNatural | CoverageStateCachedPair,
        ...,
    ],
    *,
    role: str,
    count: int,
    seed: int,
) -> tuple[
    CoverageStateCachedNatural | CoverageStateCachedPair,
    ...,
]:
    """Choose records deterministically while exhausting sources by rounds."""

    if len(values) < count:
        raise ValueError(f"{role} has fewer than {count} eligible records")
    groups: dict[
        str,
        list[CoverageStateCachedNatural | CoverageStateCachedPair],
    ] = defaultdict(list)
    for value in values:
        groups[value.record.sample_id].append(value)
    for source, group in groups.items():
        group.sort(
            key=lambda value: (
                stable_hash(
                    COVERAGE_STATE_BOUNDED_POPULATION_SCHEMA,
                    role,
                    "record",
                    seed,
                    source,
                    _identity(value),
                ),
                _identity(value),
            )
        )
    ordered_sources = sorted(
        groups,
        key=lambda source: (
            stable_hash(
                COVERAGE_STATE_BOUNDED_POPULATION_SCHEMA,
                role,
                "source",
                seed,
                source,
            ),
            source,
        ),
    )
    selected: list[
        CoverageStateCachedNatural | CoverageStateCachedPair
    ] = []
    depth = 0
    while len(selected) < count:
        added = False
        for source in ordered_sources:
            group = groups[source]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise AssertionError("bounded source-round-robin selection stalled")
        depth += 1
    return tuple(selected)


@dataclass(frozen=True, eq=False)
class CoverageStateBoundedPopulation:
    """A 16-per-role training view bound to one complete scalar cache."""

    source_cache: CoverageStateScalarCache
    cache: CoverageStateScalarCache
    seed: int
    factual_miss_record_ids: tuple[str, ...]
    factual_no_miss_record_ids: tuple[str, ...]
    clean_positive_pair_ids: tuple[str, ...]
    component_null_pair_ids: tuple[str, ...]
    identity_null_pair_ids: tuple[str, ...]
    scalar_hidden_diagnostic_pair_ids: tuple[str, ...]
    source_cache_fingerprint: str
    bounded_cache_fingerprint: str

    def __post_init__(self) -> None:
        role_values = (
            self.factual_miss_record_ids,
            self.factual_no_miss_record_ids,
            self.clean_positive_pair_ids,
            self.component_null_pair_ids,
            self.identity_null_pair_ids,
        )
        if any(
            len(values) != COVERAGE_STATE_BOUNDED_ROLE_COUNT
            or len(set(values)) != len(values)
            for values in role_values
        ):
            raise ValueError("bounded population must contain 16 unique records per role")
        if (
            self.seed != COVERAGE_STATE_BOUNDED_SEED
            or self.source_cache.cache_fingerprint
            != self.source_cache_fingerprint
            or self.cache.cache_fingerprint
            != self.bounded_cache_fingerprint
        ):
            raise ValueError("bounded population binding changed")

    def canonical_payload(self) -> dict[str, object]:
        source_counts = Counter(
            value.record.sample_id
            for value in (
                *self.cache.natural_records,
                *self.cache.pair_records,
            )
        )
        return {
            "schema_version": COVERAGE_STATE_BOUNDED_POPULATION_SCHEMA,
            "selection_policy": COVERAGE_STATE_BOUNDED_SELECTION_POLICY,
            "seed": self.seed,
            "split": "D_R",
            "source_cache_fingerprint": self.source_cache_fingerprint,
            "bounded_cache_fingerprint": self.bounded_cache_fingerprint,
            "role_count": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
            "factual_miss_record_ids": list(
                self.factual_miss_record_ids
            ),
            "factual_no_miss_record_ids": list(
                self.factual_no_miss_record_ids
            ),
            "clean_positive_pair_ids": list(
                self.clean_positive_pair_ids
            ),
            "component_null_pair_ids": list(
                self.component_null_pair_ids
            ),
            "identity_null_pair_ids": list(
                self.identity_null_pair_ids
            ),
            "scalar_hidden_diagnostic_pair_ids": list(
                self.scalar_hidden_diagnostic_pair_ids
            ),
            "source_counts": dict(sorted(source_counts.items())),
            "D_V_accessed": False,
            "D_T_accessed": False,
        }

    @cached_property
    def population_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        self.source_cache.verify_unchanged()
        self.cache.verify_unchanged()
        if (
            self.source_cache.cache_fingerprint
            != self.source_cache_fingerprint
            or self.cache.cache_fingerprint
            != self.bounded_cache_fingerprint
            or stable_fingerprint(self.canonical_payload())
            != self.population_fingerprint
        ):
            raise RuntimeError("bounded population changed after creation")


def _bounded_exposure_checks(
    population: CoverageStateBoundedPopulation,
    schedule: CoverageStateTrainingSchedule,
    report: dict[str, object],
) -> tuple[tuple[str, bool], ...]:
    checks: dict[str, bool] = {
        "population_seed": population.seed == COVERAGE_STATE_BOUNDED_SEED,
        "schedule_seed": (
            schedule.config.seed == COVERAGE_STATE_BOUNDED_SEED
        ),
        "schedule_horizon": (
            schedule.config.epochs == COVERAGE_STATE_BOUNDED_EPOCHS
            and schedule.config.steps_per_epoch
            == COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
            and schedule.config.updates == COVERAGE_STATE_BOUNDED_UPDATES
        ),
        "selection_exact_budget": bool(
            report["selection_audit"]["exact_budget_closed"]
        ),
        "identity_null_optimizer_exposure": (
            int(report["identity_null_optimizer_exposure"]) == 0
        ),
        "diagnostic_only_optimizer_exposure": (
            int(report["diagnostic_only_optimizer_exposure"]) == 0
        ),
        "D_V_not_accessed": True,
        "D_T_not_accessed": True,
    }

    def add_statistics(
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
            <= maximum_uniform_multiple / support
        )

    for branch, payload in report["branches"].items():
        add_statistics(
            f"{branch}/record",
            payload["record"],
            minimum_ess_fraction=(
                COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION
            ),
            maximum_uniform_multiple=(
                COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE
            ),
        )
        add_statistics(
            f"{branch}/source",
            payload["source"],
            minimum_ess_fraction=(
                COVERAGE_STATE_SOURCE_MINIMUM_ESS_FRACTION
            ),
            maximum_uniform_multiple=(
                COVERAGE_STATE_SOURCE_MAXIMUM_UNIFORM_MULTIPLE
            ),
        )
    for name in ("factual_focus_target", "clean_added_target"):
        add_statistics(
            name,
            report[name],
            minimum_ess_fraction=(
                COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION
            ),
            maximum_uniform_multiple=(
                COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE
            ),
        )
    return tuple(sorted(checks.items()))


@dataclass(frozen=True, eq=False)
class CoverageStateBoundedPreflight:
    """Create-only authorization for exactly one bounded-400 run."""

    population: CoverageStateBoundedPopulation
    schedule: CoverageStateTrainingSchedule
    exposure_report_fingerprint: str
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if (
            self.schedule.cache_fingerprint
            != self.population.bounded_cache_fingerprint
            or self.schedule.config.updates
            != COVERAGE_STATE_BOUNDED_UPDATES
        ):
            raise ValueError("bounded preflight binding changed")
        if (
            self.checks != tuple(sorted(self.checks))
            or len({name for name, _ in self.checks}) != len(self.checks)
            or any(
                not isinstance(name, str) or not isinstance(value, bool)
                for name, value in self.checks
            )
        ):
            raise ValueError("bounded preflight checks are malformed")

    @property
    def training_authorized(self) -> bool:
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks if not passed)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "cure-lite-cslf-bounded-preflight-v1",
            "population_fingerprint": (
                self.population.population_fingerprint
            ),
            "bounded_cache_fingerprint": (
                self.population.bounded_cache_fingerprint
            ),
            "schedule_fingerprint": self.schedule.schedule_fingerprint,
            "exposure_report_fingerprint": (
                self.exposure_report_fingerprint
            ),
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "training_authorized": self.training_authorized,
            "formal_training_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }

    @cached_property
    def preflight_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        self.population.verify_unchanged()
        report = coverage_state_schedule_exposure_report(
            self.population.cache,
            self.schedule,
        )
        if (
            report["report_fingerprint"]
            != self.exposure_report_fingerprint
            or _bounded_exposure_checks(
                self.population,
                self.schedule,
                report,
            )
            != self.checks
            or stable_fingerprint(self.canonical_payload())
            != self.preflight_fingerprint
        ):
            raise RuntimeError("bounded preflight changed after creation")


def prepare_coverage_state_bounded_preflight(
    population: CoverageStateBoundedPopulation,
) -> CoverageStateBoundedPreflight:
    """Build the 400-update schedule and evaluate all pre-training gates."""

    if not isinstance(population, CoverageStateBoundedPopulation):
        raise TypeError("population must be CoverageStateBoundedPopulation")
    population.verify_unchanged()
    schedule = build_coverage_state_training_schedule(
        population.cache,
        CoverageStateScheduleConfig(
            seed=COVERAGE_STATE_BOUNDED_SEED,
            epochs=COVERAGE_STATE_BOUNDED_EPOCHS,
            steps_per_epoch=COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH,
        ),
    )
    report = coverage_state_schedule_exposure_report(
        population.cache,
        schedule,
    )
    result = CoverageStateBoundedPreflight(
        population=population,
        schedule=schedule,
        exposure_report_fingerprint=str(
            report["report_fingerprint"]
        ),
        checks=_bounded_exposure_checks(population, schedule, report),
    )
    result.verify_unchanged()
    return result


def build_coverage_state_bounded_population(
    cache: CoverageStateScalarCache,
    *,
    seed: int = COVERAGE_STATE_BOUNDED_SEED,
) -> CoverageStateBoundedPopulation:
    """Select and seal the fixed 16-per-role D_R bounded population."""

    if not isinstance(cache, CoverageStateScalarCache):
        raise TypeError("cache must be CoverageStateScalarCache")
    if seed != COVERAGE_STATE_BOUNDED_SEED:
        raise ValueError("bounded CSLF protocol fixes seed=42")
    cache.verify_unchanged()
    if (
        cache.observability.decision
        is not CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
    ):
        raise ValueError("bounded population requires scalar observability authorization")

    miss_pool = tuple(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_miss"
    )
    no_miss_pool = tuple(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_no_miss"
    )
    clean_pool = cache.clean_positive_records
    component_pool = cache.component_null_records
    identity_pool = tuple(
        value
        for value in cache.pair_records
        if value.optimizer_role == "identity_diagnostic"
    )
    hidden_diagnostics = tuple(
        value
        for value in cache.pair_records
        if value.optimizer_role == "diagnostic_only"
    )

    miss = _source_balanced_select(
        miss_pool,
        role="factual_miss",
        count=COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        seed=seed,
    )
    no_miss = _source_balanced_select(
        no_miss_pool,
        role="factual_no_miss",
        count=COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        seed=seed,
    )
    clean = _source_balanced_select(
        clean_pool,
        role="clean_positive",
        count=COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        seed=seed,
    )
    component = _source_balanced_select(
        component_pool,
        role="component_null",
        count=COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        seed=seed,
    )
    identity = _source_balanced_select(
        identity_pool,
        role="identity_null",
        count=COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        seed=seed,
    )
    selected_naturals = tuple(
        sorted((*miss, *no_miss), key=_identity)
    )
    selected_pairs = tuple(
        sorted(
            (*clean, *component, *identity, *hidden_diagnostics),
            key=_identity,
        )
    )
    source_fingerprint = stable_fingerprint(
        {
            "schema_version": COVERAGE_STATE_BOUNDED_POPULATION_SCHEMA,
            "source_cache_fingerprint": cache.cache_fingerprint,
            "selection_policy": COVERAGE_STATE_BOUNDED_SELECTION_POLICY,
            "seed": seed,
            "natural_ids": [_identity(value) for value in selected_naturals],
            "pair_ids": [_identity(value) for value in selected_pairs],
        }
    )
    raw = make_coverage_state_raw_catalog(
        dataset=cache.raw_catalog.dataset,
        feature_stride=cache.raw_catalog.feature_stride,
        source_fingerprint=source_fingerprint,
        natural_records=tuple(
            value.record for value in selected_naturals
        ),
        pair_records=tuple(value.record for value in selected_pairs),
    )
    observability = audit_population_observability(raw)
    if (
        observability.decision
        is not CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
    ):
        raise RuntimeError(
            "bounded selection changed the scalar representation decision"
        )
    bounded_cache = CoverageStateScalarCache(
        raw_catalog=raw,
        observability=observability,
        sobolev_config=cache.sobolev_config,
        natural_records=selected_naturals,
        pair_records=selected_pairs,
        raw_catalog_fingerprint=raw.catalog_fingerprint,
        observability_receipt_fingerprint=(
            observability.receipt_fingerprint
        ),
        sobolev_config_fingerprint=cache.sobolev_config_fingerprint,
    )
    bounded_cache.verify_unchanged()
    result = CoverageStateBoundedPopulation(
        source_cache=cache,
        cache=bounded_cache,
        seed=seed,
        factual_miss_record_ids=tuple(
            _identity(value) for value in miss
        ),
        factual_no_miss_record_ids=tuple(
            _identity(value) for value in no_miss
        ),
        clean_positive_pair_ids=tuple(
            _identity(value) for value in clean
        ),
        component_null_pair_ids=tuple(
            _identity(value) for value in component
        ),
        identity_null_pair_ids=tuple(
            _identity(value) for value in identity
        ),
        scalar_hidden_diagnostic_pair_ids=tuple(
            _identity(value) for value in hidden_diagnostics
        ),
        source_cache_fingerprint=cache.cache_fingerprint,
        bounded_cache_fingerprint=bounded_cache.cache_fingerprint,
    )
    result.verify_unchanged()
    return result


__all__ = [
    "COVERAGE_STATE_BOUNDED_EPOCHS",
    "COVERAGE_STATE_BOUNDED_POPULATION_SCHEMA",
    "COVERAGE_STATE_BOUNDED_ROLE_COUNT",
    "COVERAGE_STATE_BOUNDED_SEED",
    "COVERAGE_STATE_BOUNDED_SELECTION_POLICY",
    "COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH",
    "COVERAGE_STATE_BOUNDED_UPDATES",
    "CoverageStateBoundedPreflight",
    "CoverageStateBoundedPopulation",
    "build_coverage_state_bounded_population",
    "prepare_coverage_state_bounded_preflight",
]
