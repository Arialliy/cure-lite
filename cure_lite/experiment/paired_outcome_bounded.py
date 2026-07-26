"""Bounded D_R-only execution for outcome-complete CURE-Lite v3.

This runner is the first full-population model-code gate for OC-APTO.  It
optimizes one fresh decoder for the frozen 400 updates with two factual
anchor batches and two outcome pairs per update.  All 206 clean-positive and
16 component-null pairs participate in the same pair-uniform schedule.

The execution is intentionally not a detector benchmark: it never reads
``D_V`` or ``D_T``, never updates a Base/backbone, and does not calibrate or
run the hard-union inference path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import ceil, isfinite, sqrt
from typing import Mapping

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..config import DecoderConfig, LossConfig
from ..decoder import CURELiteDecoder
from ..losses import CURELiteLoss
from ..paired_outcome_losses import OutcomeCompleteTransitionLoss
from ..paired_outcome_types import OutcomePairBatch
from ..paired_types import PairCatalog, PairExample, stack_pair_examples
from ..sampling import stable_hash
from ..train.paired_outcome_step import outcome_complete_train_step
from ..train.paired_step import _paired_endpoint_logits, diagnose_null_pairs
from ..train.pools import StateExample, stack_state_examples
from .artifacts import decoder_state_fingerprint
from .paired_bounded_learnability import (
    _deterministic_torch_runtime,
)
from .paired_formal_schedule import prepared_training_catalog_fingerprint
from .paired_outcome_inputs import PairedOutcomeInputMaterializer
from .paired_outcome_schedule import OutcomePairSchedule
from .training_pipeline import PreparedTrainingCatalog


PAIRED_OUTCOME_BOUNDED_SCHEMA = (
    "cure-lite-paired-outcome-bounded-execution-v1"
)
OUTCOME_BOUNDED_ANCHOR_POPULATION_SCHEMA = (
    "cure-lite-outcome-bounded-anchor-population-v1"
)
OUTCOME_FACTUAL_ANCHOR_SCHEDULE_SCHEMA = (
    "cure-lite-outcome-factual-anchor-schedule-v1"
)

DETERMINISM_SPECIFICATION: Mapping[str, object] = {
    "torch_use_deterministic_algorithms": True,
    "torch_deterministic_warn_only": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cublas_workspace_config": ":4096:8",
    "restore_process_torch_flags_after_execution": True,
    "exact_replay_required_under_same_frozen_environment": True,
}

# These thresholds are the frozen OC-APTO v3 bounded model-code gates.  Macro
# means are within-pair means followed by an ordinary pair mean.  The
# component footprint maximum is one maximum over every component pair and
# every H pixel, not a mean of per-pair maxima.
COMPUTATIONAL_THRESHOLDS: Mapping[str, float] = {
    "factual_miss_anchor_final_over_initial_max": 0.75,
    "factual_no_miss_anchor_final_over_initial_max": 0.75,
    "plus_baseline_final_over_initial_max": 0.75,
    "clean_transition_final_over_initial_max": 0.50,
    "clean_mean_delta_on_D_min": 0.50,
    "clean_pairs_delta_at_least_0_25_fraction_min": 0.75,
    "clean_zero_macro_mean_abs_delta_max": 0.05,
    "component_null_footprint_macro_mean_abs_delta_max": 0.05,
    "component_null_footprint_global_max_abs_delta_max": 0.25,
    "component_null_context_macro_mean_abs_delta_max": 0.05,
    "identity_null_max_abs_delta_max": 1.0e-7,
}

_REQUIRED_BUDGET_KEYS = {
    "seed",
    "optimizer_updates",
    "steps_per_epoch",
    "factual_miss_states_per_update",
    "factual_no_miss_states_per_update",
    "outcome_pairs_per_update",
    "learning_rate",
    "weight_decay",
}


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _require_fingerprint(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 fingerprint")
    return value


def _anchor_id(branch: str, example: StateExample) -> str:
    """Preserve the already frozen factual-anchor identity definition."""

    return stable_fingerprint(
        {
            "schema_version": "cure-lite-bounded-anchor-id-v1",
            "branch": branch,
            "sample_id": example.sample_id,
            "positive_gt_ids": list(
                example.supervision.positive_gt_ids
            ),
        }
    )


def _stable_select(
    values: tuple[object, ...],
    count: int,
    *,
    namespace: str,
    seed: int,
    identity,
) -> tuple[object, ...]:
    count = _positive_int(count, name=f"{namespace}.count")
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


@dataclass(frozen=True, eq=False)
class OutcomeBoundedAnchorPopulation:
    """The only v3 bounded auxiliary population consumed by the executor."""

    seed: int
    pair_catalog_fingerprint: str
    prepared_catalog_fingerprint: str
    factual_miss: tuple[StateExample, ...]
    factual_no_miss: tuple[StateExample, ...]
    identity_null: tuple[PairExample, ...]
    factual_miss_ids: tuple[str, ...]
    factual_no_miss_ids: tuple[str, ...]
    population_fingerprint: str

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name in (
            "pair_catalog_fingerprint",
            "prepared_catalog_fingerprint",
            "population_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), name=name)
        if len(self.factual_miss) != 16:
            raise ValueError("factual_miss must contain exactly 16 anchors")
        if len(self.factual_no_miss) != 16:
            raise ValueError(
                "factual_no_miss must contain exactly 16 anchors"
            )
        if len(self.identity_null) != 16:
            raise ValueError(
                "identity_null must contain exactly 16 diagnostic pairs"
            )
        if any(
            not isinstance(example, StateExample)
            or example.supervision.branch != "factual_miss"
            for example in self.factual_miss
        ):
            raise TypeError("factual_miss contains an invalid anchor")
        if any(
            not isinstance(example, StateExample)
            or example.supervision.branch != "factual_no_miss"
            for example in self.factual_no_miss
        ):
            raise TypeError("factual_no_miss contains an invalid anchor")
        if any(
            not isinstance(pair, PairExample)
            or pair.pair_kind != "identity_null"
            for pair in self.identity_null
        ):
            raise TypeError("identity_null contains an invalid pair")
        if (
            len(self.factual_miss_ids) != 16
            or len(set(self.factual_miss_ids)) != 16
            or any(
                _anchor_id("factual_miss", example) != anchor_id
                for anchor_id, example in zip(
                    self.factual_miss_ids,
                    self.factual_miss,
                    strict=True,
                )
            )
        ):
            raise ValueError("factual_miss_ids do not bind the anchors")
        if (
            len(self.factual_no_miss_ids) != 16
            or len(set(self.factual_no_miss_ids)) != 16
            or any(
                _anchor_id("factual_no_miss", example) != anchor_id
                for anchor_id, example in zip(
                    self.factual_no_miss_ids,
                    self.factual_no_miss,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "factual_no_miss_ids do not bind the anchors"
            )
        if stable_fingerprint(self.canonical_payload()) != (
            self.population_fingerprint
        ):
            raise ValueError(
                "outcome anchor population fingerprint changed"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": OUTCOME_BOUNDED_ANCHOR_POPULATION_SCHEMA,
            "seed": self.seed,
            "pair_catalog_fingerprint": self.pair_catalog_fingerprint,
            "prepared_catalog_fingerprint": (
                self.prepared_catalog_fingerprint
            ),
            "selection_rule": (
                "stable-hash-over-consumed-anchor-identities-without-"
                "feature-loss-or-result-access-v1"
            ),
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
            "identity_null": [
                {
                    "pair_id": pair.pair_id,
                    "sample_id": pair.sample_id,
                }
                for pair in self.identity_null
            ],
        }

    def canonical_receipt(self) -> dict[str, object]:
        payload = self.canonical_payload()
        payload["population_fingerprint"] = self.population_fingerprint
        return payload


def build_outcome_bounded_anchor_population(
    pair_catalog: PairCatalog,
    prepared: PreparedTrainingCatalog,
    specification: Mapping[str, object],
) -> OutcomeBoundedAnchorPopulation:
    """Select exactly the 16/16 factual anchors and 16 identity diagnostics."""

    if not isinstance(pair_catalog, PairCatalog):
        raise TypeError("pair_catalog must be PairCatalog")
    if not isinstance(prepared, PreparedTrainingCatalog):
        raise TypeError("prepared must be PreparedTrainingCatalog")
    if not isinstance(specification, Mapping):
        raise TypeError("specification must be a mapping")
    required = {
        "seed",
        "factual_miss_anchors",
        "factual_no_miss_anchors",
        "identity_null_pairs",
    }
    if set(specification) != required:
        raise ValueError(
            "anchor specification must contain exactly "
            f"{sorted(required)}"
        )
    if pair_catalog.split != "D_R":
        raise ValueError("outcome anchor population permits only D_R")
    if stable_fingerprint(pair_catalog.canonical_payload()) != (
        pair_catalog.catalog_fingerprint
    ):
        raise RuntimeError("pair catalog fingerprint does not reproduce")
    seed = specification["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("anchor population seed must be an integer")
    for name in (
        "factual_miss_anchors",
        "factual_no_miss_anchors",
        "identity_null_pairs",
    ):
        if specification[name] != 16:
            raise ValueError(f"{name} is frozen at 16")

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
    factual_miss = _stable_select(
        factual_miss_all,
        16,
        namespace="bounded-factual-miss-v1",
        seed=seed,
        identity=lambda example: _anchor_id("factual_miss", example),
    )
    factual_no_miss = _stable_select(
        factual_no_miss_all,
        16,
        namespace="bounded-factual-no-miss-v1",
        seed=seed,
        identity=lambda example: _anchor_id(
            "factual_no_miss",
            example,
        ),
    )
    identity_null = _stable_select(
        pair_catalog.identity_null,
        16,
        namespace="bounded-identity-null-v1",
        seed=seed,
        identity=lambda pair: pair.pair_id,
    )
    if not all(
        isinstance(value, StateExample)
        for value in (*factual_miss, *factual_no_miss)
    ):
        raise TypeError("selected factual anchors have invalid types")
    if not all(
        isinstance(value, PairExample) for value in identity_null
    ):
        raise TypeError("selected identity diagnostics have invalid types")
    miss_values = tuple(factual_miss)
    no_miss_values = tuple(factual_no_miss)
    identity_values = tuple(identity_null)
    miss_ids = tuple(
        _anchor_id("factual_miss", example)
        for example in miss_values
    )
    no_miss_ids = tuple(
        _anchor_id("factual_no_miss", example)
        for example in no_miss_values
    )
    population = object.__new__(OutcomeBoundedAnchorPopulation)
    for name, value in {
        "seed": seed,
        "pair_catalog_fingerprint": pair_catalog.catalog_fingerprint,
        "prepared_catalog_fingerprint": (
            prepared_training_catalog_fingerprint(prepared)
        ),
        "factual_miss": miss_values,
        "factual_no_miss": no_miss_values,
        "identity_null": identity_values,
        "factual_miss_ids": miss_ids,
        "factual_no_miss_ids": no_miss_ids,
    }.items():
        object.__setattr__(population, name, value)
    object.__setattr__(
        population,
        "population_fingerprint",
        stable_fingerprint(population.canonical_payload()),
    )
    population.__post_init__()
    return population


@dataclass(frozen=True, eq=False)
class OutcomeFactualAnchorSchedule:
    """A v3-only 4/4 factual schedule with no unused pair index table."""

    population_fingerprint: str
    optimizer_updates: int
    steps_per_epoch: int
    factual_miss_indices: tuple[tuple[int, int, int, int], ...]
    factual_no_miss_indices: tuple[tuple[int, int, int, int], ...]
    factual_miss_counts: tuple[int, ...]
    factual_no_miss_counts: tuple[int, ...]
    schedule_fingerprint: str

    def __post_init__(self) -> None:
        _require_fingerprint(
            self.population_fingerprint,
            name="population_fingerprint",
        )
        _require_fingerprint(
            self.schedule_fingerprint,
            name="schedule_fingerprint",
        )
        _positive_int(self.optimizer_updates, name="optimizer_updates")
        _positive_int(self.steps_per_epoch, name="steps_per_epoch")
        if self.optimizer_updates % self.steps_per_epoch:
            raise ValueError(
                "optimizer_updates must be divisible by steps_per_epoch"
            )
        for name in (
            "factual_miss_indices",
            "factual_no_miss_indices",
        ):
            rows = getattr(self, name)
            if (
                len(rows) != self.optimizer_updates
                or any(len(row) != 4 for row in rows)
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < 16
                    for row in rows
                    for index in row
                )
            ):
                raise ValueError(f"{name} is not a valid 4-per-update table")
        for rows, counts, name in (
            (
                self.factual_miss_indices,
                self.factual_miss_counts,
                "factual_miss_counts",
            ),
            (
                self.factual_no_miss_indices,
                self.factual_no_miss_counts,
                "factual_no_miss_counts",
            ),
        ):
            actual = Counter(index for row in rows for index in row)
            if (
                len(counts) != 16
                or tuple(actual[index] for index in range(16)) != counts
            ):
                raise ValueError(f"{name} differs from its index table")
        if stable_fingerprint(self.canonical_payload()) != (
            self.schedule_fingerprint
        ):
            raise ValueError("factual anchor schedule fingerprint changed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": OUTCOME_FACTUAL_ANCHOR_SCHEDULE_SCHEMA,
            "population_fingerprint": self.population_fingerprint,
            "optimizer_updates": self.optimizer_updates,
            "steps_per_epoch": self.steps_per_epoch,
            "factual_states_per_update": {
                "factual_miss": 4,
                "factual_no_miss": 4,
            },
            "factual_miss_indices": [
                list(row) for row in self.factual_miss_indices
            ],
            "factual_no_miss_indices": [
                list(row) for row in self.factual_no_miss_indices
            ],
            "factual_miss_counts": list(self.factual_miss_counts),
            "factual_no_miss_counts": list(
                self.factual_no_miss_counts
            ),
        }

    def canonical_receipt(self) -> dict[str, object]:
        payload = self.canonical_payload()
        payload["schedule_fingerprint"] = self.schedule_fingerprint
        return payload


def build_outcome_factual_anchor_schedule(
    population: OutcomeBoundedAnchorPopulation,
    *,
    optimizer_updates: int,
    steps_per_epoch: int,
) -> OutcomeFactualAnchorSchedule:
    """Build the exact cyclic 4/4 schedule over consumed factual anchors."""

    if not isinstance(population, OutcomeBoundedAnchorPopulation):
        raise TypeError(
            "population must be OutcomeBoundedAnchorPopulation"
        )
    optimizer_updates = _positive_int(
        optimizer_updates,
        name="optimizer_updates",
    )
    steps_per_epoch = _positive_int(
        steps_per_epoch,
        name="steps_per_epoch",
    )
    if optimizer_updates % steps_per_epoch:
        raise ValueError(
            "optimizer_updates must be divisible by steps_per_epoch"
        )
    indices = tuple(
        tuple((4 * update + draw) % 16 for draw in range(4))
        for update in range(optimizer_updates)
    )
    ledger = Counter(index for row in indices for index in row)
    counts = tuple(ledger[index] for index in range(16))
    schedule = object.__new__(OutcomeFactualAnchorSchedule)
    for name, value in {
        "population_fingerprint": population.population_fingerprint,
        "optimizer_updates": optimizer_updates,
        "steps_per_epoch": steps_per_epoch,
        "factual_miss_indices": indices,
        "factual_no_miss_indices": indices,
        "factual_miss_counts": counts,
        "factual_no_miss_counts": counts,
    }.items():
        object.__setattr__(schedule, name, value)
    object.__setattr__(
        schedule,
        "schedule_fingerprint",
        stable_fingerprint(schedule.canonical_payload()),
    )
    schedule.__post_init__()
    return schedule


def _validate_execution_inputs(
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: DecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    device: torch.device | str,
    evaluation_chunk_size: int,
) -> tuple[torch.device, int, float, float]:
    if not isinstance(population, OutcomeBoundedAnchorPopulation):
        raise TypeError(
            "population must be OutcomeBoundedAnchorPopulation"
        )
    if not isinstance(factual_schedule, OutcomeFactualAnchorSchedule):
        raise TypeError(
            "factual_schedule must be OutcomeFactualAnchorSchedule"
        )
    if not isinstance(schedule, OutcomePairSchedule):
        raise TypeError("schedule must be OutcomePairSchedule")
    if not isinstance(materializer, PairedOutcomeInputMaterializer):
        raise TypeError(
            "materializer must be PairedOutcomeInputMaterializer"
        )
    if not isinstance(decoder_config, DecoderConfig):
        raise TypeError("decoder_config must be DecoderConfig")
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss_config must be LossConfig")
    if not isinstance(optimization_budget, Mapping):
        raise TypeError("optimization_budget must be a mapping")
    if set(optimization_budget) != _REQUIRED_BUDGET_KEYS:
        raise ValueError(
            "optimization_budget must contain exactly "
            f"{sorted(_REQUIRED_BUDGET_KEYS)}"
        )

    seed = optimization_budget["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("optimization_budget.seed must be an integer")
    updates = _positive_int(
        optimization_budget["optimizer_updates"],
        name="optimization_budget.optimizer_updates",
    )
    steps_per_epoch = _positive_int(
        optimization_budget["steps_per_epoch"],
        name="optimization_budget.steps_per_epoch",
    )
    if updates != 400:
        raise ValueError("OC-APTO bounded execution fixes 400 optimizer updates")
    if steps_per_epoch != 40:
        raise ValueError("OC-APTO bounded execution fixes 40 steps per epoch")
    if (
        updates != schedule.optimizer_updates
        or steps_per_epoch != schedule.steps_per_epoch
    ):
        raise ValueError("optimization budget and outcome schedule disagree")
    if (
        updates != factual_schedule.optimizer_updates
        or steps_per_epoch != factual_schedule.steps_per_epoch
        or factual_schedule.population_fingerprint
        != population.population_fingerprint
    ):
        raise ValueError(
            "optimization budget and factual anchor schedule disagree"
        )
    if seed != schedule.seed or seed != population.seed:
        raise ValueError("population, schedule, and optimization seed disagree")
    if optimization_budget["factual_miss_states_per_update"] != 4:
        raise ValueError("OC-APTO requires four factual-miss states per update")
    if optimization_budget["factual_no_miss_states_per_update"] != 4:
        raise ValueError(
            "OC-APTO requires four factual-no-miss states per update"
        )
    if optimization_budget["outcome_pairs_per_update"] != 2:
        raise ValueError("OC-APTO requires two outcome pairs per update")
    learning_rate = _finite_float(
        optimization_budget["learning_rate"],
        name="optimization_budget.learning_rate",
        positive=True,
    )
    weight_decay = _finite_float(
        optimization_budget["weight_decay"],
        name="optimization_budget.weight_decay",
        nonnegative=True,
    )
    evaluation_chunk_size = _positive_int(
        evaluation_chunk_size,
        name="evaluation_chunk_size",
    )

    if len(population.factual_miss) != 16:
        raise ValueError("bounded execution requires 16 factual-miss anchors")
    if len(population.factual_no_miss) != 16:
        raise ValueError("bounded execution requires 16 factual-no-miss anchors")
    if len(population.identity_null) != 16:
        raise ValueError("bounded execution requires 16 identity-null pairs")
    if len(set(population.factual_miss_ids)) != 16:
        raise ValueError("factual-miss anchor IDs must be unique")
    if len(set(population.factual_no_miss_ids)) != 16:
        raise ValueError("factual-no-miss anchor IDs must be unique")

    clean_count = sum(
        pair.pair_kind == "clean_positive" for pair in schedule.pairs
    )
    component_count = sum(
        pair.pair_kind == "component_null" for pair in schedule.pairs
    )
    if (clean_count, component_count, len(schedule.pairs)) != (206, 16, 222):
        raise ValueError(
            "bounded outcome schedule must bind the frozen 206+16 population"
        )
    if set(schedule.pair_exposure_counts) != {3, 4}:
        raise ValueError("bounded outcome pair exposures must be exactly 3 or 4")
    if schedule.exposures != 800:
        raise ValueError("bounded outcome schedule must contain 800 pair slots")
    if (
        schedule.catalog_fingerprint
        != materializer.pair_catalog_fingerprint
        or population.pair_catalog_fingerprint
        != materializer.pair_catalog_fingerprint
    ):
        raise ValueError("population, schedule, and materializer catalogs disagree")
    schedule_ids = tuple(pair.pair_id for pair in schedule.pairs)
    if set(schedule_ids) != set(materializer.canonical_pair_ids):
        raise ValueError("materializer must bind exactly all 222 scheduled pairs")
    for pair in schedule.pairs:
        bound = materializer.pair_by_id[pair.pair_id]
        if pair.canonical_payload() != bound.canonical_payload():
            raise ValueError("scheduled pair differs from materialized binding")

    if decoder_config.feature_channels != materializer.feature_shape[1]:
        raise ValueError("decoder feature channels differ from outcome inputs")
    expected_feature = tuple(materializer.feature_shape[1:])
    expected_state = tuple(materializer.evaluation_shape)
    for branch_name, values in (
        ("factual_miss", population.factual_miss),
        ("factual_no_miss", population.factual_no_miss),
    ):
        for example in values:
            if tuple(example.feature.shape[1:]) != expected_feature:
                raise ValueError(f"{branch_name} feature shape differs")
            if tuple(example.supervision.occupancy.shape) != expected_state:
                raise ValueError(f"{branch_name} state shape differs")
    for pair in population.identity_null:
        if tuple(pair.feature.shape[1:]) != expected_feature:
            raise ValueError("identity-null feature shape differs")
        if tuple(pair.image_valid_mask.shape) != expected_state:
            raise ValueError("identity-null evaluation shape differs")

    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must be a valid torch device") from error
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("OC-APTO bounded execution supports only CPU or CUDA")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")

    # First and only pre-training full-population hash verification.
    materializer.verify_unchanged()
    return (
        target_device,
        seed,
        learning_rate,
        weight_decay,
    )


class _ForwardLedger:
    def __init__(self, decoder: CURELiteDecoder) -> None:
        self.calls = 0
        self.states = 0
        self._handle = decoder.register_forward_hook(self._record)

    def _record(
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


def _factual_batches(
    population: OutcomeBoundedAnchorPopulation,
    schedule: OutcomeFactualAnchorSchedule,
    update: int,
    *,
    device: torch.device,
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


def _factual_metrics(
    decoder: CURELiteDecoder,
    criterion: CURELiteLoss,
    population: OutcomeBoundedAnchorPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for branch, values, anchor_ids in (
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
        result = criterion(
            decoder(batch.feature.detach(), batch.occupancy),
            batch.target,
            batch.valid_mask,
        )
        per_state = result["per_state_total"]
        output[branch] = {
            "state_count": len(values),
            "loss": float(result["total"].cpu()),
            "per_state": [
                {
                    "anchor_id": anchor_id,
                    "sample_id": values[index].sample_id,
                    "loss": float(per_state[index].cpu()),
                }
                for index, anchor_id in enumerate(anchor_ids)
            ],
        }
    return output


def _mean(values: list[float], *, name: str) -> float:
    if not values:
        raise RuntimeError(f"{name} population is empty")
    if not all(isfinite(value) for value in values):
        raise FloatingPointError(f"{name} contains non-finite values")
    return sum(values) / len(values)


def _outcome_metrics(
    decoder: CURELiteDecoder,
    materializer: PairedOutcomeInputMaterializer,
    schedule: OutcomePairSchedule,
    criterion: OutcomeCompleteTransitionLoss,
    *,
    device: torch.device,
    chunk_size: int,
) -> dict[str, object]:
    pair_ids = tuple(pair.pair_id for pair in schedule.pairs)
    records: list[dict[str, object]] = []
    all_plus: list[float] = []
    clean_transition: list[float] = []
    clean_d: list[float] = []
    clean_zero: list[float] = []
    component_transition: list[float] = []
    component_h: list[float] = []
    component_g: list[float] = []
    component_footprint_global_max = 0.0
    strata_counts = {
        "clean_positive": {"D": 0, "H": 0, "G": 0},
        "component_null": {"D": 0, "H": 0, "G": 0},
    }

    for start in range(0, len(pair_ids), chunk_size):
        selected_ids = pair_ids[start : start + chunk_size]
        batch = materializer.materialize(selected_ids, device=device)
        logits_plus, logits_minus = _paired_endpoint_logits(
            decoder,
            feature=batch.pair_batch.feature,
            occupancy_plus=batch.pair_batch.occupancy_plus,
            occupancy_minus=batch.pair_batch.occupancy_minus,
        )
        result = criterion(
            logits_plus,
            logits_minus,
            batch.completion_plus,
            batch.pair_batch.occupancy_plus,
            batch.gt_union,
            batch.pair_batch.label_increment,
            batch.pair_batch.image_valid_mask,
            batch.intervention_footprint,
        )
        delta = torch.sigmoid(logits_minus) - torch.sigmoid(logits_plus)
        absolute_delta = delta.abs()
        for index, pair_id in enumerate(selected_ids):
            kind = batch.pair_batch.pair_kinds[index]
            response = batch.response_stratum[index]
            local = batch.local_zero_stratum[index]
            global_ = batch.global_zero_stratum[index]
            d_count = int(response.sum().cpu())
            h_count = int(local.sum().cpu())
            g_count = int(global_.sum().cpu())
            strata_counts[kind]["D"] += d_count
            strata_counts[kind]["H"] += h_count
            strata_counts[kind]["G"] += g_count

            d_mean = (
                None
                if d_count == 0
                else float(delta[index][response].mean().cpu())
            )
            h_mean = (
                None
                if h_count == 0
                else float(absolute_delta[index][local].mean().cpu())
            )
            h_max = (
                None
                if h_count == 0
                else float(absolute_delta[index][local].max().cpu())
            )
            g_mean = (
                None
                if g_count == 0
                else float(absolute_delta[index][global_].mean().cpu())
            )
            active_zero = [
                value for value in (h_mean, g_mean) if value is not None
            ]
            zero_mean = _mean(
                active_zero,
                name=f"zero strata for {pair_id}",
            )
            plus_loss = float(result["per_pair_plus_anchor"][index].cpu())
            transition_loss = float(
                result["per_pair_transition"][index].cpu()
            )
            all_plus.append(plus_loss)
            if kind == "clean_positive":
                if d_mean is None:
                    raise RuntimeError("clean-positive pair has an empty D")
                clean_transition.append(transition_loss)
                clean_d.append(d_mean)
                clean_zero.append(zero_mean)
            else:
                if d_mean is not None or h_mean is None or g_mean is None:
                    raise RuntimeError(
                        "component-null D/H/G strata violate the frozen contract"
                    )
                component_transition.append(transition_loss)
                component_h.append(h_mean)
                component_g.append(g_mean)
                component_footprint_global_max = max(
                    component_footprint_global_max,
                    float(h_max),
                )

            records.append(
                {
                    "pair_id": pair_id,
                    "pair_kind": kind,
                    "sample_id": batch.pair_batch.sample_ids[index],
                    "D_pixels": d_count,
                    "H_pixels": h_count,
                    "G_pixels": g_count,
                    "plus_baseline_loss": plus_loss,
                    "transition_loss": transition_loss,
                    "pair_total_loss": float(
                        result["per_pair_total"][index].cpu()
                    ),
                    "D_mean_delta": d_mean,
                    "H_mean_abs_delta": h_mean,
                    "H_max_abs_delta": h_max,
                    "G_mean_abs_delta": g_mean,
                    "zero_strata_active_mean_abs_delta": zero_mean,
                }
            )

    if len(records) != 222:
        raise RuntimeError("outcome evaluation did not cover all 222 pairs")
    return {
        "pair_count": len(records),
        "clean_positive_count": len(clean_transition),
        "component_null_count": len(component_transition),
        "evaluation_chunk_size": chunk_size,
        "evaluation_chunk_count": ceil(len(records) / chunk_size),
        "pair_ids": list(pair_ids),
        "strata_pixel_counts": strata_counts,
        "plus_baseline_loss": _mean(all_plus, name="plus baseline"),
        "clean": {
            "transition_loss": _mean(
                clean_transition,
                name="clean transition",
            ),
            "D_pair_macro_mean_delta": _mean(clean_d, name="clean D"),
            "D_pair_fraction_mean_delta_ge_0_25": (
                sum(value >= 0.25 for value in clean_d) / len(clean_d)
            ),
            "zero_strata_pair_macro_mean_abs_delta": _mean(
                clean_zero,
                name="clean zero strata",
            ),
        },
        "component_null": {
            "transition_loss": _mean(
                component_transition,
                name="component transition",
            ),
            "footprint_pair_macro_mean_abs_delta": _mean(
                component_h,
                name="component footprint",
            ),
            "footprint_global_max_abs_delta": (
                component_footprint_global_max
            ),
            "context_pair_macro_mean_abs_delta": _mean(
                component_g,
                name="component context",
            ),
        },
        "per_pair": records,
    }


def _identity_metrics(
    decoder: CURELiteDecoder,
    population: OutcomeBoundedAnchorPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    batch = stack_pair_examples(population.identity_null, device=device)
    result = diagnose_null_pairs(decoder, batch)
    mean_abs = result["per_pair_mean_abs_delta"]
    max_abs = result["per_pair_max_abs_delta"]
    rms = result["per_pair_rms_delta"]
    return {
        "pair_count": len(population.identity_null),
        "maximum_abs_delta": float(max_abs.max().cpu()),
        "macro_mean_abs_delta": float(mean_abs.mean().cpu()),
        "macro_rms_delta": float(rms.mean().cpu()),
        "per_pair": [
            {
                "pair_id": pair_id,
                "mean_abs_delta": float(mean_abs[index].cpu()),
                "maximum_abs_delta": float(max_abs[index].cpu()),
                "rms_delta": float(rms[index].cpu()),
            }
            for index, pair_id in enumerate(batch.pair_ids)
        ],
        "diagnostic_only": True,
        "autograd_enabled": False,
        "optimizer_exposure_count": 0,
    }


def _evaluate_snapshot(
    decoder: CURELiteDecoder,
    population: OutcomeBoundedAnchorPopulation,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    absolute_criterion: CURELiteLoss,
    outcome_criterion: OutcomeCompleteTransitionLoss,
    *,
    device: torch.device,
    chunk_size: int,
) -> dict[str, object]:
    decoder.eval()
    with torch.no_grad():
        factual = _factual_metrics(
            decoder,
            absolute_criterion,
            population,
            device=device,
        )
        outcome = _outcome_metrics(
            decoder,
            materializer,
            schedule,
            outcome_criterion,
            device=device,
            chunk_size=chunk_size,
        )
        identity = _identity_metrics(decoder, population, device=device)
    return {
        "factual_anchors": factual,
        "outcome_population": outcome,
        "identity_null": identity,
    }


def _ratio(final: float, initial: float, *, name: str) -> float:
    if (
        not isfinite(initial)
        or not isfinite(final)
        or initial <= 0.0
    ):
        raise ValueError(
            f"{name} requires finite values and a positive denominator"
        )
    return final / initial


def _computational_gates(
    initial: Mapping[str, object],
    final: Mapping[str, object],
) -> dict[str, object]:
    initial_factual = initial["factual_anchors"]
    final_factual = final["factual_anchors"]
    initial_outcome = initial["outcome_population"]
    final_outcome = final["outcome_population"]
    identity = final["identity_null"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            initial_factual,
            final_factual,
            initial_outcome,
            final_outcome,
            identity,
        )
    ):
        raise TypeError("OC-APTO snapshot metrics are malformed")
    initial_clean = initial_outcome["clean"]
    final_clean = final_outcome["clean"]
    final_component = final_outcome["component_null"]
    if not all(
        isinstance(value, Mapping)
        for value in (initial_clean, final_clean, final_component)
    ):
        raise TypeError("OC-APTO outcome metrics are malformed")

    observed = {
        "factual_miss_anchor_final_over_initial": _ratio(
            float(final_factual["factual_miss"]["loss"]),
            float(initial_factual["factual_miss"]["loss"]),
            name="factual-miss anchor loss ratio",
        ),
        "factual_no_miss_anchor_final_over_initial": _ratio(
            float(final_factual["factual_no_miss"]["loss"]),
            float(initial_factual["factual_no_miss"]["loss"]),
            name="factual-no-miss anchor loss ratio",
        ),
        "plus_baseline_final_over_initial": _ratio(
            float(final_outcome["plus_baseline_loss"]),
            float(initial_outcome["plus_baseline_loss"]),
            name="plus baseline loss ratio",
        ),
        "clean_transition_final_over_initial": _ratio(
            float(final_clean["transition_loss"]),
            float(initial_clean["transition_loss"]),
            name="clean transition loss ratio",
        ),
        "clean_mean_delta_on_D": float(
            final_clean["D_pair_macro_mean_delta"]
        ),
        "clean_pairs_delta_at_least_0_25_fraction": float(
            final_clean["D_pair_fraction_mean_delta_ge_0_25"]
        ),
        "clean_zero_macro_mean_abs_delta": float(
            final_clean["zero_strata_pair_macro_mean_abs_delta"]
        ),
        "component_null_footprint_macro_mean_abs_delta": float(
            final_component["footprint_pair_macro_mean_abs_delta"]
        ),
        "component_null_footprint_global_max_abs_delta": float(
            final_component["footprint_global_max_abs_delta"]
        ),
        "component_null_context_macro_mean_abs_delta": float(
            final_component["context_pair_macro_mean_abs_delta"]
        ),
        "identity_null_max_abs_delta": float(
            identity["maximum_abs_delta"]
        ),
    }
    rules = {
        "factual_miss_anchor_final_over_initial": (
            "max",
            "factual_miss_anchor_final_over_initial_max",
        ),
        "factual_no_miss_anchor_final_over_initial": (
            "max",
            "factual_no_miss_anchor_final_over_initial_max",
        ),
        "plus_baseline_final_over_initial": (
            "max",
            "plus_baseline_final_over_initial_max",
        ),
        "clean_transition_final_over_initial": (
            "max",
            "clean_transition_final_over_initial_max",
        ),
        "clean_mean_delta_on_D": (
            "min",
            "clean_mean_delta_on_D_min",
        ),
        "clean_pairs_delta_at_least_0_25_fraction": (
            "min",
            "clean_pairs_delta_at_least_0_25_fraction_min",
        ),
        "clean_zero_macro_mean_abs_delta": (
            "max",
            "clean_zero_macro_mean_abs_delta_max",
        ),
        "component_null_footprint_macro_mean_abs_delta": (
            "max",
            "component_null_footprint_macro_mean_abs_delta_max",
        ),
        "component_null_footprint_global_max_abs_delta": (
            "max",
            "component_null_footprint_global_max_abs_delta_max",
        ),
        "component_null_context_macro_mean_abs_delta": (
            "max",
            "component_null_context_macro_mean_abs_delta_max",
        ),
        "identity_null_max_abs_delta": (
            "max",
            "identity_null_max_abs_delta_max",
        ),
    }
    checks: dict[str, object] = {}
    for name, (direction, threshold_name) in rules.items():
        value = float(observed[name])
        if not isfinite(value):
            raise FloatingPointError(f"non-finite OC-APTO gate value for {name}")
        threshold = COMPUTATIONAL_THRESHOLDS[threshold_name]
        checks[name] = {
            "value": value,
            "direction": direction,
            "threshold": threshold,
            "applicable": True,
            "status": "EVALUATED",
            "pass": value >= threshold if direction == "min" else value <= threshold,
        }
    return {
        "scope": "bounded_D_R_full_outcome_population_model_code_gate",
        "not_detection_performance": True,
        "thresholds": dict(COMPUTATIONAL_THRESHOLDS),
        "observed": observed,
        "checks": checks,
        "all_pass": all(bool(check["pass"]) for check in checks.values()),
    }


def execute_paired_outcome_bounded(
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: DecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    *,
    device: torch.device | str,
    evaluation_chunk_size: int = 32,
) -> dict[str, object]:
    """Train and audit a fresh OC-APTO decoder on the sealed D_R population."""

    (
        target_device,
        seed,
        learning_rate,
        weight_decay,
    ) = _validate_execution_inputs(
        population,
        factual_schedule,
        schedule,
        materializer,
        decoder_config,
        loss_config,
        optimization_budget,
        device,
        evaluation_chunk_size,
    )
    evaluation_chunk_size = int(evaluation_chunk_size)
    cuda_devices: list[int] = []
    if target_device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device()
            if target_device.index is None
            else target_device.index
        ]

    with _deterministic_torch_runtime(
        target_device,
        DETERMINISM_SPECIFICATION,
    ) as deterministic_runtime, torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        decoder = CURELiteDecoder(decoder_config).to(target_device)
        absolute_criterion = CURELiteLoss(loss_config).to(target_device)
        outcome_criterion = OutcomeCompleteTransitionLoss(
            loss_config
        ).to(target_device)
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        parameter_count = sum(
            parameter.numel() for parameter in decoder.parameters()
        )
        initial_decoder_fingerprint = decoder_state_fingerprint(decoder)
        initial_parameter_norm = sqrt(
            sum(
                float(parameter.detach().double().square().sum().cpu())
                for parameter in decoder.parameters()
            )
        )

        ledger = _ForwardLedger(decoder)
        pair_exposure: Counter[str] = Counter()
        source_exposure: Counter[str] = Counter()
        miss_exposure: Counter[str] = Counter()
        no_miss_exposure: Counter[str] = Counter()
        trace: list[dict[str, object]] = []
        minimum_gradient_norm = float("inf")
        maximum_gradient_norm = 0.0
        nonfinite_gradient_updates = 0
        zero_gradient_updates = 0
        optimizer_steps = 0
        backward_calls = 0
        try:
            before_initial = ledger.snapshot()
            initial = _evaluate_snapshot(
                decoder,
                population,
                schedule,
                materializer,
                absolute_criterion,
                outcome_criterion,
                device=target_device,
                chunk_size=evaluation_chunk_size,
            )
            after_initial = ledger.snapshot()
            initial_forward = {
                "calls": after_initial[0] - before_initial[0],
                "state_evaluations": after_initial[1] - before_initial[1],
            }

            training_start = ledger.snapshot()
            for update in range(schedule.optimizer_updates):
                pair_ids = schedule.pair_ids_for_update(update)
                miss_indices = factual_schedule.factual_miss_indices[update]
                no_miss_indices = (
                    factual_schedule.factual_no_miss_indices[update]
                )
                before_update = ledger.snapshot()
                logs = outcome_complete_train_step(
                    decoder,
                    absolute_criterion,
                    outcome_criterion,
                    optimizer,
                    _factual_batches(
                        population,
                        factual_schedule,
                        update,
                        device=target_device,
                    ),
                    materializer.materialize(
                        pair_ids,
                        device=target_device,
                    ),
                )
                squared_gradient_norm = sum(
                    float(
                        parameter.grad.detach().double().square().sum().cpu()
                    )
                    for parameter in decoder.parameters()
                    if parameter.grad is not None
                )
                gradient_norm = sqrt(squared_gradient_norm)
                if not isfinite(gradient_norm):
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
                optimizer_steps += int(logs["optimizer_steps"])
                backward_calls += int(logs["backward_calls"])
                pair_exposure.update(pair_ids)
                source_exposure.update(
                    materializer.pair_by_id[pair_id].sample_id
                    for pair_id in pair_ids
                )
                miss_exposure.update(
                    population.factual_miss_ids[index]
                    for index in miss_indices
                )
                no_miss_exposure.update(
                    population.factual_no_miss_ids[index]
                    for index in no_miss_indices
                )
                after_update = ledger.snapshot()
                trace.append(
                    {
                        "update": update,
                        "epoch": update // schedule.steps_per_epoch,
                        "step": update % schedule.steps_per_epoch,
                        "outcome_pair_ids": list(pair_ids),
                        "outcome_pair_kinds": [
                            materializer.pair_by_id[pair_id].pair_kind
                            for pair_id in pair_ids
                        ],
                        "factual_miss_ids": [
                            population.factual_miss_ids[index]
                            for index in miss_indices
                        ],
                        "factual_no_miss_ids": [
                            population.factual_no_miss_ids[index]
                            for index in no_miss_indices
                        ],
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

            # Second and final full-population hash verification.  The 400
            # update hot path above uses sealed pair bindings without a full
            # population re-hash.
            materializer.verify_unchanged()
            before_final = ledger.snapshot()
            final = _evaluate_snapshot(
                decoder,
                population,
                schedule,
                materializer,
                absolute_criterion,
                outcome_criterion,
                device=target_device,
                chunk_size=evaluation_chunk_size,
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

    scheduled_ids = tuple(pair.pair_id for pair in schedule.pairs)
    actual_pair_counts = tuple(
        pair_exposure[pair_id] for pair_id in scheduled_ids
    )
    actual_miss_counts = tuple(
        miss_exposure[anchor_id]
        for anchor_id in population.factual_miss_ids
    )
    actual_no_miss_counts = tuple(
        no_miss_exposure[anchor_id]
        for anchor_id in population.factual_no_miss_ids
    )
    actual_source_counts = tuple(sorted(source_exposure.items()))
    expected_snapshot_forward = {
        "calls": ceil(222 / evaluation_chunk_size) + 3,
        "state_evaluations": 2 * 222 + 2 * 16 + 2 * 16,
    }
    expected_training_forward = {
        "calls": 3 * schedule.optimizer_updates,
        "state_evaluations": 12 * schedule.optimizer_updates,
    }
    expected_total_forward = {
        "calls": (
            expected_training_forward["calls"]
            + 2 * expected_snapshot_forward["calls"]
        ),
        "state_evaluations": (
            expected_training_forward["state_evaluations"]
            + 2 * expected_snapshot_forward["state_evaluations"]
        ),
    }
    per_update_forward_exact = all(
        row["decoder_forward_calls"] == 3
        and row["decoder_state_evaluations"] == 12
        for row in trace
    )
    exposure = {
        "outcome_pairs": [
            {
                "pair_id": pair.pair_id,
                "pair_kind": pair.pair_kind,
                "sample_id": pair.sample_id,
                "count": pair_exposure[pair.pair_id],
            }
            for pair in schedule.pairs
        ],
        "source_images": [
            {"sample_id": sample_id, "count": count}
            for sample_id, count in actual_source_counts
        ],
        "factual_miss": [
            {
                "anchor_id": anchor_id,
                "sample_id": example.sample_id,
                "count": miss_exposure[anchor_id],
            }
            for anchor_id, example in zip(
                population.factual_miss_ids,
                population.factual_miss,
                strict=True,
            )
        ],
        "factual_no_miss": [
            {
                "anchor_id": anchor_id,
                "sample_id": example.sample_id,
                "count": no_miss_exposure[anchor_id],
            }
            for anchor_id, example in zip(
                population.factual_no_miss_ids,
                population.factual_no_miss,
                strict=True,
            )
        ],
        "outcome_pair_exposure_values": sorted(set(actual_pair_counts)),
        "identity_null_optimizer_exposure": 0,
    }
    structural_checks = {
        "deterministic_runtime_contract_satisfied": (
            deterministic_runtime["contract_satisfied"] is True
            and deterministic_runtime["flags_restored_after_execution"] is True
        ),
        "factual_anchor_and_identity_counts_exact": (
            len(population.factual_miss) == 16
            and len(population.factual_no_miss) == 16
            and len(population.identity_null) == 16
        ),
        "all_222_outcome_pairs_bound": (
            len(scheduled_ids) == 222
            and set(scheduled_ids) == set(materializer.canonical_pair_ids)
        ),
        "all_222_outcome_pairs_evaluated_initial": (
            initial["outcome_population"]["pair_ids"] == list(scheduled_ids)
        ),
        "all_222_outcome_pairs_evaluated_final": (
            final["outcome_population"]["pair_ids"] == list(scheduled_ids)
        ),
        "all_optimizer_updates_completed": len(trace) == 400,
        "one_backward_per_update": backward_calls == 400,
        "one_optimizer_step_per_update": optimizer_steps == 400,
        "all_gradients_finite": nonfinite_gradient_updates == 0,
        "every_update_total_gradient_norm_positive": (
            zero_gradient_updates == 0
        ),
        "decoder_parameters_changed": (
            final_decoder_fingerprint != initial_decoder_fingerprint
        ),
        "training_forward_budget_exact": (
            training_forward == expected_training_forward
            and per_update_forward_exact
        ),
        "evaluation_forward_budget_exact": (
            initial_forward == expected_snapshot_forward
            and final_forward == expected_snapshot_forward
        ),
        "total_forward_budget_exact": total_forward == expected_total_forward,
        "pair_exposure_ledger_exact": (
            actual_pair_counts == schedule.pair_exposure_counts
            and set(actual_pair_counts) == {3, 4}
        ),
        "source_exposure_ledger_exact": (
            actual_source_counts == schedule.source_exposure_counts
        ),
        "factual_exposure_ledgers_exact": (
            actual_miss_counts == factual_schedule.factual_miss_counts
            and actual_no_miss_counts
            == factual_schedule.factual_no_miss_counts
        ),
        "identity_null_excluded_from_optimizer": (
            exposure["identity_null_optimizer_exposure"] == 0
        ),
        "identity_null_diagnosed_without_autograd": all(
            snapshot["identity_null"]["autograd_enabled"] is False
            for snapshot in (initial, final)
        ),
    }
    structural_execution_pass = all(structural_checks.values())
    computational = _computational_gates(initial, final)
    computational_pass = (
        structural_execution_pass and computational["all_pass"] is True
    )
    decision = (
        "BOUNDED_MODEL_CODE_GATE_PASS"
        if computational_pass
        else (
            "STRUCTURAL_EXECUTION_FAIL"
            if not structural_execution_pass
            else "BOUNDED_MODEL_CODE_GATE_FAIL"
        )
    )
    result: dict[str, object] = {
        "schema_version": PAIRED_OUTCOME_BOUNDED_SCHEMA,
        "execution_status": "completed",
        "decision": decision,
        "device": str(target_device),
        "population_fingerprint": population.population_fingerprint,
        "outcome_schedule_fingerprint": schedule.schedule_fingerprint,
        "factual_schedule_fingerprint": (
            factual_schedule.schedule_fingerprint
        ),
        "materializer_fingerprint": materializer.materializer_fingerprint,
        "decoder_config": asdict(decoder_config),
        "loss_config": asdict(loss_config),
        "optimization_budget": dict(optimization_budget),
        "evaluation_chunk_size": evaluation_chunk_size,
        "optimizer_updates_completed": len(trace),
        "initial": initial,
        "final": final,
        "computational_gates": computational,
        "structural_checks": structural_checks,
        "structural_execution_pass": structural_execution_pass,
        "computational_model_code_gate_pass": computational_pass,
        "parameters": {
            "trainable_parameter_count": parameter_count,
            "initial_decoder_fingerprint": initial_decoder_fingerprint,
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
        "execution_ledger": {
            "backward_calls": backward_calls,
            "optimizer_steps": optimizer_steps,
            "expected_backward_calls": 400,
            "expected_optimizer_steps": 400,
        },
        "forward_budget": {
            "initial_evaluation": initial_forward,
            "training": training_forward,
            "final_evaluation": final_forward,
            "total": total_forward,
            "expected_initial_evaluation": expected_snapshot_forward,
            "expected_training": expected_training_forward,
            "expected_final_evaluation": expected_snapshot_forward,
            "expected_total": expected_total_forward,
        },
        "deterministic_runtime": deterministic_runtime,
        "exposure": exposure,
        "trace": trace,
        "interpretation": {
            "evidence_scope": (
                "fresh_decoder_bounded_D_R_full_outcome_population_model_code"
            ),
            "not_detection_performance_evidence": True,
            "does_not_establish_Pd_or_FA": True,
            "does_not_authorize_formal_training": (
                computational_pass is not True
            ),
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "base_or_backbone_updated": False,
            "identity_null_optimizer_exposure": 0,
        },
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


__all__ = [
    "COMPUTATIONAL_THRESHOLDS",
    "DETERMINISM_SPECIFICATION",
    "OUTCOME_BOUNDED_ANCHOR_POPULATION_SCHEMA",
    "OUTCOME_FACTUAL_ANCHOR_SCHEDULE_SCHEMA",
    "PAIRED_OUTCOME_BOUNDED_SCHEMA",
    "OutcomeBoundedAnchorPopulation",
    "OutcomeFactualAnchorSchedule",
    "build_outcome_bounded_anchor_population",
    "build_outcome_factual_anchor_schedule",
    "execute_paired_outcome_bounded",
]
