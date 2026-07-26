"""Dataset-free exposure-matched confirmation inputs for PECO v10.

This module contains no dataset, cache, detector, or evaluation-set access.
It freezes a synthetic population with the exact role and exposure counts of
the v8 bounded run:

* 206 clean-positive pairs and 16 component-null pairs;
* 739 clean slots and 61 component-null slots;
* 400 updates of two pairs;
* 340/59/1 updates with zero/one/two component-null pairs.

Pair kind is metadata used only to audit the population.  Decoder inputs are
always ``(detached_feature, occupancy)`` and outcome losses receive tensor
truth, never pair-kind metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..paired_outcome_types import (
    OutcomePairBatch,
    direct_projected_intervention_footprint,
)
from ..paired_types import PairBatch
from ..train.step import BranchBatch


CONFIRMATION_SEED = 42
PAIR_BATCH_SIZE = 2
UPDATE_COUNT = 400
CLEAN_PAIR_COUNT = 206
COMPONENT_PAIR_COUNT = 16
CLEAN_SLOT_COUNT = 739
COMPONENT_SLOT_COUNT = 61
FACTUAL_POPULATION_SIZE = 16
FACTUAL_BATCH_SIZE = 4
FACTUAL_SLOTS_PER_BRANCH = UPDATE_COUNT * FACTUAL_BATCH_SIZE
FACTUAL_EXPOSURES_PER_STATE = (
    FACTUAL_SLOTS_PER_BRANCH // FACTUAL_POPULATION_SIZE
)

CONTAINS_FAMILY = "component_contains_response"
OUTSIDE_FAMILY = "response_outside_component_inside_count_support"
COMPONENT_BLOCK_FAMILY = "component_null_block_geometry"
COMPONENT_SPARSE_FAMILY = "component_null_sparse_geometry"

_CLEAN_PIXEL_PATTERNS = {
    CONTAINS_FAMILY: {
        1: ((1, 2),),
        2: ((1, 2), (2, 1)),
        3: ((1, 2), (2, 1), (2, 2)),
    },
    OUTSIDE_FAMILY: {
        1: ((1, 6),),
        2: ((1, 6), (2, 5)),
        3: ((1, 6), (2, 5), (2, 6)),
    },
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConfirmationPairSpec:
    """Immutable identity, role, geometry, and exposure contract for one pair."""

    population_index: int
    pair_id: str
    sample_id: str
    group_id: str
    pair_kind: str
    geometry_family: str
    response_pixel_count: int
    exposure_count: int

    def __post_init__(self) -> None:
        if self.population_index < 0:
            raise ValueError("population_index must be nonnegative")
        if len(self.pair_id) != 64:
            raise ValueError("pair_id must be a SHA256 fingerprint")
        if self.pair_kind not in {"clean_positive", "component_null"}:
            raise ValueError("confirmation pair kind is invalid")
        if self.pair_kind == "clean_positive":
            if self.geometry_family not in {
                CONTAINS_FAMILY,
                OUTSIDE_FAMILY,
            }:
                raise ValueError("clean pair geometry family is invalid")
            if self.response_pixel_count not in {1, 2, 3}:
                raise ValueError("clean response count must be 1, 2, or 3")
        else:
            if self.geometry_family not in {
                COMPONENT_BLOCK_FAMILY,
                COMPONENT_SPARSE_FAMILY,
            }:
                raise ValueError("component-null geometry family is invalid")
            if self.response_pixel_count != 0:
                raise ValueError("component-null response count must be zero")
        if self.exposure_count not in {3, 4}:
            raise ValueError("every confirmation pair requires 3 or 4 exposures")

    def manifest(self) -> dict[str, object]:
        return {
            "population_index": self.population_index,
            "pair_id": self.pair_id,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "pair_kind": self.pair_kind,
            "geometry_family": self.geometry_family,
            "response_pixel_count": self.response_pixel_count,
            "exposure_count": self.exposure_count,
        }


@dataclass(frozen=True)
class ConfirmationUpdate:
    """One deterministic two-pair optimizer update."""

    update_index: int
    population_indices: tuple[int, int]
    pair_ids: tuple[str, str]
    pair_kinds: tuple[str, str]
    sample_ids: tuple[str, str]

    def __post_init__(self) -> None:
        if self.update_index < 0:
            raise ValueError("update_index must be nonnegative")
        for name, values in (
            ("population_indices", self.population_indices),
            ("pair_ids", self.pair_ids),
            ("pair_kinds", self.pair_kinds),
            ("sample_ids", self.sample_ids),
        ):
            if not isinstance(values, tuple) or len(values) != 2:
                raise ValueError(f"{name} must contain exactly two values")
        if len(set(self.population_indices)) != 2:
            raise ValueError("one update cannot repeat a population pair")
        if len(set(self.pair_ids)) != 2:
            raise ValueError("one update cannot repeat a pair_id")
        if len(set(self.sample_ids)) != 2:
            raise ValueError("one update requires distinct source samples")

    @property
    def component_count(self) -> int:
        return sum(kind == "component_null" for kind in self.pair_kinds)

    def manifest(self) -> dict[str, object]:
        return {
            "update_index": self.update_index,
            "population_indices": list(self.population_indices),
            "pair_ids": list(self.pair_ids),
            "pair_kinds": list(self.pair_kinds),
            "sample_ids": list(self.sample_ids),
            "component_count": self.component_count,
        }


def build_confirmation_pair_specs() -> tuple[ConfirmationPairSpec, ...]:
    """Return the fixed 222-pair population in canonical order."""

    clean_records: list[tuple[int, str, int, str]] = []
    clean_by_group: dict[str, list[int]] = {}
    for clean_index in range(CLEAN_PAIR_COUNT):
        family = (
            CONTAINS_FAMILY
            if clean_index < CLEAN_PAIR_COUNT // 2
            else OUTSIDE_FAMILY
        )
        family_index = (
            clean_index
            if family == CONTAINS_FAMILY
            else clean_index - CLEAN_PAIR_COUNT // 2
        )
        pixels = 1 + family_index % 3
        group = (
            f"clean_contains_{pixels}px"
            if family == CONTAINS_FAMILY
            else f"clean_outside_{pixels}px"
        )
        clean_records.append((clean_index, family, pixels, group))
        clean_by_group.setdefault(group, []).append(clean_index)

    # The 121 four-exposure pairs are distributed across every geometry
    # stratum (21 in the first canonical group and 20 in each remaining
    # group), rather than confounding exposure count with geometry family.
    clean_four_exposures: set[int] = set()
    for group_index, group in enumerate(sorted(clean_by_group)):
        quota = 21 if group_index == 0 else 20
        ordered = sorted(
            clean_by_group[group],
            key=lambda index: _sha(
                f"{CONFIRMATION_SEED}|clean-exposure|{group}|{index}"
            ),
        )
        clean_four_exposures.update(ordered[:quota])
    if len(clean_four_exposures) != 121:
        raise AssertionError("clean four-exposure allocation differs")

    specs: list[ConfirmationPairSpec] = []
    for clean_index, family, pixels, group in clean_records:
        identity = f"peco-v10-confirmation-clean-{clean_index:03d}"
        specs.append(
            ConfirmationPairSpec(
                population_index=len(specs),
                pair_id=_sha(identity),
                sample_id=f"{identity}-source",
                group_id=group,
                pair_kind="clean_positive",
                geometry_family=family,
                response_pixel_count=pixels,
                exposure_count=(
                    4 if clean_index in clean_four_exposures else 3
                ),
            )
        )

    component_four_exposures = {
        index
        for parity, quota in ((0, 7), (1, 6))
        for index in sorted(
            range(parity, COMPONENT_PAIR_COUNT, 2),
            key=lambda value: _sha(
                f"{CONFIRMATION_SEED}|component-exposure|{parity}|{value}"
            ),
        )[:quota]
    }
    if len(component_four_exposures) != 13:
        raise AssertionError("component four-exposure allocation differs")
    for component_index in range(COMPONENT_PAIR_COUNT):
        family = (
            COMPONENT_BLOCK_FAMILY
            if component_index % 2 == 0
            else COMPONENT_SPARSE_FAMILY
        )
        group = (
            "component_null_block"
            if family == COMPONENT_BLOCK_FAMILY
            else "component_null_sparse"
        )
        identity = (
            f"peco-v10-confirmation-component-{component_index:03d}"
        )
        specs.append(
            ConfirmationPairSpec(
                population_index=len(specs),
                pair_id=_sha(identity),
                sample_id=f"{identity}-source",
                group_id=group,
                pair_kind="component_null",
                geometry_family=family,
                response_pixel_count=0,
                exposure_count=(
                    4
                    if component_index in component_four_exposures
                    else 3
                ),
            )
        )
    _validate_population(tuple(specs))
    return tuple(specs)


def _validate_population(
    specs: tuple[ConfirmationPairSpec, ...],
) -> None:
    if len(specs) != CLEAN_PAIR_COUNT + COMPONENT_PAIR_COUNT:
        raise AssertionError("confirmation population size differs")
    if tuple(spec.population_index for spec in specs) != tuple(
        range(len(specs))
    ):
        raise AssertionError("confirmation population indices are not canonical")
    if len({spec.pair_id for spec in specs}) != len(specs):
        raise AssertionError("confirmation pair IDs are not unique")
    if len({spec.sample_id for spec in specs}) != len(specs):
        raise AssertionError("confirmation source IDs are not unique")
    clean = tuple(
        spec for spec in specs if spec.pair_kind == "clean_positive"
    )
    component = tuple(
        spec for spec in specs if spec.pair_kind == "component_null"
    )
    if len(clean) != CLEAN_PAIR_COUNT or len(component) != COMPONENT_PAIR_COUNT:
        raise AssertionError("confirmation role counts differ")
    if sum(spec.exposure_count for spec in clean) != CLEAN_SLOT_COUNT:
        raise AssertionError("confirmation clean slot count differs")
    if sum(spec.exposure_count for spec in component) != (
        COMPONENT_SLOT_COUNT
    ):
        raise AssertionError("confirmation component slot count differs")


def catalog_manifest(
    specs: tuple[ConfirmationPairSpec, ...] | None = None,
) -> list[dict[str, object]]:
    values = build_confirmation_pair_specs() if specs is None else specs
    _validate_population(values)
    return [spec.manifest() for spec in values]


def catalog_fingerprint(
    specs: tuple[ConfirmationPairSpec, ...] | None = None,
) -> str:
    return stable_fingerprint(catalog_manifest(specs))


def _slot_key(role: str, round_index: int, population_index: int) -> str:
    return _sha(
        f"{CONFIRMATION_SEED}|{role}|{round_index}|{population_index}"
    )


def _role_slots(
    specs: tuple[ConfirmationPairSpec, ...],
    *,
    pair_kind: str,
) -> list[int]:
    role_specs = tuple(spec for spec in specs if spec.pair_kind == pair_kind)
    slots: list[int] = []
    for round_index in range(4):
        active = [
            spec
            for spec in role_specs
            if spec.exposure_count > round_index
        ]
        active.sort(
            key=lambda spec: _slot_key(
                pair_kind,
                round_index,
                spec.population_index,
            )
        )
        slots.extend(spec.population_index for spec in active)
    return slots


def _take_distinct_pair(
    slots: list[int],
    first: int,
) -> int:
    for index, value in enumerate(slots):
        if value != first:
            return slots.pop(index)
    raise AssertionError("no distinct pair remains for a two-pair update")


def build_confirmation_schedule(
    specs: tuple[ConfirmationPairSpec, ...] | None = None,
) -> tuple[ConfirmationUpdate, ...]:
    """Return the exact, sparse-component 400-update schedule."""

    population = build_confirmation_pair_specs() if specs is None else specs
    _validate_population(population)
    by_index = {spec.population_index: spec for spec in population}
    clean_slots = _role_slots(population, pair_kind="clean_positive")
    component_slots = _role_slots(population, pair_kind="component_null")
    if len(clean_slots) != CLEAN_SLOT_COUNT:
        raise AssertionError("clean slot construction differs")
    if len(component_slots) != COMPONENT_SLOT_COUNT:
        raise AssertionError("component slot construction differs")

    # Sixty component-bearing updates are spread over all 400 updates.  The
    # first selected location receives two component slots; the other 59
    # receive one.  All remaining slots are clean.
    component_positions = tuple(
        (rank * UPDATE_COUNT) // 60 for rank in range(60)
    )
    if len(set(component_positions)) != 60:
        raise AssertionError("component update positions are not unique")
    double_component_position = component_positions[0]
    component_position_set = set(component_positions)

    raw_updates: list[tuple[int, int]] = []
    for update_index in range(UPDATE_COUNT):
        if update_index == double_component_position:
            first = component_slots.pop(0)
            second = _take_distinct_pair(component_slots, first)
        elif update_index in component_position_set:
            first = component_slots.pop(0)
            second = clean_slots.pop(0)
        else:
            first = clean_slots.pop(0)
            second = _take_distinct_pair(clean_slots, first)
        raw_updates.append((first, second))
    if clean_slots or component_slots:
        raise AssertionError("confirmation schedule left unused slots")

    updates = tuple(
        ConfirmationUpdate(
            update_index=update_index,
            population_indices=indices,
            pair_ids=tuple(by_index[index].pair_id for index in indices),
            pair_kinds=tuple(
                by_index[index].pair_kind for index in indices
            ),
            sample_ids=tuple(
                by_index[index].sample_id for index in indices
            ),
        )
        for update_index, indices in enumerate(raw_updates)
    )
    _validate_schedule(population, updates)
    return updates


def _validate_schedule(
    specs: tuple[ConfirmationPairSpec, ...],
    updates: tuple[ConfirmationUpdate, ...],
) -> None:
    if len(updates) != UPDATE_COUNT:
        raise AssertionError("confirmation update count differs")
    if tuple(update.update_index for update in updates) != tuple(
        range(UPDATE_COUNT)
    ):
        raise AssertionError("confirmation update order is not canonical")
    component_histogram = {
        count: sum(update.component_count == count for update in updates)
        for count in (0, 1, 2)
    }
    if component_histogram != {0: 340, 1: 59, 2: 1}:
        raise AssertionError("component update histogram differs")
    observed = {spec.population_index: 0 for spec in specs}
    for update in updates:
        for population_index in update.population_indices:
            observed[population_index] += 1
    expected = {
        spec.population_index: spec.exposure_count for spec in specs
    }
    if observed != expected:
        raise AssertionError("per-pair exposure counts differ")


def schedule_manifest(
    specs: tuple[ConfirmationPairSpec, ...] | None = None,
) -> list[dict[str, object]]:
    population = build_confirmation_pair_specs() if specs is None else specs
    return [
        update.manifest()
        for update in build_confirmation_schedule(population)
    ]


def schedule_fingerprint(
    specs: tuple[ConfirmationPairSpec, ...] | None = None,
) -> str:
    return stable_fingerprint(schedule_manifest(specs))


def _pair_feature(spec: ConfirmationPairSpec) -> Tensor:
    """Construct a deterministic feature-only role signal.

    The pair role is never encoded as an explicit categorical value.  The
    feature families differ because the confirmation asks whether the frozen
    decoder can learn role-associated evidence under the sparse schedule.
    The conflicting-input control separately proves that metadata cannot
    select a role-specific decoder output.
    """

    feature = torch.zeros(1, 8, 2, 2, dtype=torch.float32)
    jitter_code = int(spec.pair_id[:8], 16) % 17 - 8
    jitter = float(jitter_code) * 0.0125
    if spec.pair_kind == "clean_positive":
        if spec.geometry_family == CONTAINS_FAMILY:
            feature[0, 0, 0, 0] = 5.0 + jitter
            feature[0, 1, 1, 0] = 4.0 - jitter
        else:
            feature[0, 0, 0, 1] = 5.0 + jitter
            feature[0, 1, 1, 1] = 4.0 - jitter
        feature[0, 4, 0, 0] = float(spec.response_pixel_count) / 3.0
        feature[0, 5, 1, 1] = float(spec.response_pixel_count - 2)
        feature[0, 6] = 0.5 + 0.25 * jitter
    else:
        if spec.geometry_family == COMPONENT_BLOCK_FAMILY:
            feature[0, 2, 1, 1] = 5.0 + jitter
            feature[0, 3, 0, 1] = 4.0 - jitter
        else:
            feature[0, 2, 0, 0] = 5.0 + jitter
            feature[0, 3, 1, 0] = 4.0 - jitter
        feature[0, 7] = 0.5 + 0.25 * jitter
    return feature.contiguous()


def _pair_state(
    spec: ConfirmationPairSpec,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    occupancy_plus = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    if spec.geometry_family in {
        CONTAINS_FAMILY,
        COMPONENT_BLOCK_FAMILY,
    }:
        occupancy_plus[0, 0, 0:4, 0:4] = True
    else:
        occupancy_plus[0, 0, 0, 0] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)

    completion_plus = torch.zeros_like(occupancy_plus)
    completion_plus[0, 0, 5, 1] = True
    completion_minus = completion_plus.clone()
    if spec.pair_kind == "clean_positive":
        pixels = _CLEAN_PIXEL_PATTERNS[spec.geometry_family][
            spec.response_pixel_count
        ]
        for row, column in pixels:
            completion_minus[0, 0, row, column] = True
    increment = (completion_minus & ~completion_plus).to(torch.float32)
    return (
        occupancy_plus,
        occupancy_minus,
        completion_plus,
        completion_minus,
        increment,
    )


def build_confirmation_outcome_batch(
    specs: Iterable[ConfirmationPairSpec],
    *,
    device: torch.device | str = "cpu",
) -> OutcomePairBatch:
    """Materialize exactly two frozen confirmation specs on one device."""

    values = tuple(specs)
    if len(values) != PAIR_BATCH_SIZE:
        raise ValueError("confirmation outcome batch requires exactly two specs")
    if len({spec.pair_id for spec in values}) != PAIR_BATCH_SIZE:
        raise ValueError("confirmation outcome batch requires distinct pairs")
    if len({spec.sample_id for spec in values}) != PAIR_BATCH_SIZE:
        raise ValueError("confirmation outcome batch requires distinct sources")
    states = tuple(_pair_state(spec) for spec in values)
    target_device = torch.device(device)
    pair_batch = PairBatch(
        feature=torch.cat(
            [_pair_feature(spec) for spec in values],
            dim=0,
        ).to(target_device),
        occupancy_plus=torch.cat(
            [state[0] for state in states],
            dim=0,
        ).to(target_device),
        occupancy_minus=torch.cat(
            [state[1] for state in states],
            dim=0,
        ).to(target_device),
        label_increment=torch.cat(
            [state[4] for state in states],
            dim=0,
        ).to(target_device),
        image_valid_mask=torch.ones(
            PAIR_BATCH_SIZE,
            1,
            8,
            8,
            dtype=torch.bool,
            device=target_device,
        ),
        pair_ids=tuple(spec.pair_id for spec in values),
        sample_ids=tuple(spec.sample_id for spec in values),
        group_ids=tuple(spec.group_id for spec in values),
        pair_kinds=tuple(spec.pair_kind for spec in values),
        projection_visible=(True, True),
    )
    completion_plus = torch.cat(
        [state[2] for state in states],
        dim=0,
    ).to(target_device)
    completion_minus = torch.cat(
        [state[3] for state in states],
        dim=0,
    ).to(target_device)
    return OutcomePairBatch(
        pair_batch=pair_batch,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        gt_union=completion_minus.clone(),
        intervention_footprint=direct_projected_intervention_footprint(
            pair_batch
        ),
    )


def _build_confirmation_factual_populations(
    *,
    device: torch.device | str = "cpu",
) -> dict[str, BranchBatch]:
    """Return the fixed 16/16 factual populations before update slicing."""

    target_device = torch.device(device)
    valid = torch.ones(
        FACTUAL_POPULATION_SIZE,
        1,
        8,
        8,
        dtype=torch.bool,
    )
    occupancy = torch.zeros_like(valid)

    miss_feature = torch.zeros(FACTUAL_POPULATION_SIZE, 8, 2, 2)
    miss_target = torch.zeros(FACTUAL_POPULATION_SIZE, 1, 8, 8)
    miss_pixels = (
        ((1, 2),),
        ((1, 2), (2, 1)),
        ((1, 6),),
        ((1, 6), (2, 5), (2, 6)),
    )
    for index in range(FACTUAL_POPULATION_SIZE):
        template = index % len(miss_pixels)
        pixels = miss_pixels[template]
        cell_column = 0 if template < 2 else 1
        jitter = float(index // len(miss_pixels) - 1.5) * 0.05
        miss_feature[index, 0, 0, cell_column] = 5.0 + jitter
        miss_feature[index, 1, 1, cell_column] = 4.0 - jitter
        miss_feature[index, 4, 0, 0] = float(len(pixels)) / 3.0
        miss_feature[index, 5, 1, 1] = float(len(pixels) - 2)
        miss_feature[index, 6] = 0.5 + 0.25 * jitter
        for row, column in pixels:
            miss_target[index, 0, row, column] = 1.0

    no_miss_feature = torch.zeros(
        FACTUAL_POPULATION_SIZE,
        8,
        2,
        2,
    )
    for index in range(FACTUAL_POPULATION_SIZE):
        jitter = float(index % 4 - 1.5) * 0.05
        row = (index // 4) % 2
        column = index % 2
        no_miss_feature[index, 4, row, column] = 3.0 + jitter
        no_miss_feature[index, 5] = -0.5 + 0.25 * jitter
    return {
        "factual_miss": BranchBatch(
            feature=miss_feature.to(target_device),
            occupancy=occupancy.to(target_device),
            target=miss_target.to(target_device),
            valid_mask=valid.to(target_device),
        ),
        "factual_no_miss": BranchBatch(
            feature=no_miss_feature.to(target_device),
            occupancy=occupancy.clone().to(target_device),
            target=torch.zeros_like(miss_target).to(target_device),
            valid_mask=valid.clone().to(target_device),
        ),
    }


def factual_indices_for_update(update_index: int) -> tuple[int, ...]:
    """Return the frozen four-state slice ``(4u+i) mod 16``."""

    if (
        isinstance(update_index, bool)
        or not isinstance(update_index, int)
        or not 0 <= update_index < UPDATE_COUNT
    ):
        raise ValueError(
            f"update_index must be an integer in [0,{UPDATE_COUNT})"
        )
    return tuple(
        (FACTUAL_BATCH_SIZE * update_index + offset)
        % FACTUAL_POPULATION_SIZE
        for offset in range(FACTUAL_BATCH_SIZE)
    )


def factual_schedule_manifest() -> list[dict[str, object]]:
    return [
        {
            "update_index": update_index,
            "factual_miss_indices": list(
                factual_indices_for_update(update_index)
            ),
            "factual_no_miss_indices": list(
                factual_indices_for_update(update_index)
            ),
        }
        for update_index in range(UPDATE_COUNT)
    ]


def factual_schedule_fingerprint() -> str:
    return stable_fingerprint(factual_schedule_manifest())


def build_confirmation_factual_population(
    *,
    device: torch.device | str = "cpu",
) -> dict[str, BranchBatch]:
    """Return all 16 states of both factual branches for final evaluation."""

    return _build_confirmation_factual_populations(device=device)


def build_confirmation_factual_batches(
    *,
    update_index: int,
    device: torch.device | str = "cpu",
) -> dict[str, BranchBatch]:
    """Return the exact 4/4 factual slice for one optimizer update."""

    populations = _build_confirmation_factual_populations(device=device)
    indices = torch.tensor(
        factual_indices_for_update(update_index),
        dtype=torch.int64,
        device=torch.device(device),
    )
    selected: dict[str, BranchBatch] = {}
    for name, population in populations.items():
        selected[name] = BranchBatch(
            feature=population.feature.index_select(0, indices),
            occupancy=population.occupancy.index_select(0, indices),
            target=population.target.index_select(0, indices),
            valid_mask=population.valid_mask.index_select(0, indices),
        )
    return selected


def build_identical_input_conflict_control(
    *,
    device: torch.device | str = "cpu",
) -> OutcomePairBatch:
    """Return identical model inputs with mutually exclusive outcome gates.

    Row zero is clean-positive and requires a positive endpoint delta at
    ``(1, 2)``.  Row one is component-null and requires zero delta at that
    same footprint pixel.  Both rows have bitwise-identical feature and
    occupancy endpoints, so a shared decoder cannot satisfy both gates by
    consulting role metadata.
    """

    clean = ConfirmationPairSpec(
        population_index=0,
        pair_id=_sha("peco-v10-identical-conflict-clean"),
        sample_id="peco-v10-identical-conflict-clean-source",
        group_id="identical_input_conflict",
        pair_kind="clean_positive",
        geometry_family=CONTAINS_FAMILY,
        response_pixel_count=1,
        exposure_count=3,
    )
    null = ConfirmationPairSpec(
        population_index=1,
        pair_id=_sha("peco-v10-identical-conflict-null"),
        sample_id="peco-v10-identical-conflict-null-source",
        group_id="identical_input_conflict",
        pair_kind="component_null",
        geometry_family=COMPONENT_BLOCK_FAMILY,
        response_pixel_count=0,
        exposure_count=3,
    )
    clean_state = _pair_state(clean)
    null_state = _pair_state(null)
    target_device = torch.device(device)
    feature = _pair_feature(clean)
    pair_batch = PairBatch(
        feature=torch.cat((feature, feature.clone()), dim=0).to(target_device),
        occupancy_plus=torch.cat(
            (clean_state[0], clean_state[0].clone()),
            dim=0,
        ).to(target_device),
        occupancy_minus=torch.cat(
            (clean_state[1], clean_state[1].clone()),
            dim=0,
        ).to(target_device),
        label_increment=torch.cat(
            (clean_state[4], null_state[4]),
            dim=0,
        ).to(target_device),
        image_valid_mask=torch.ones(
            2,
            1,
            8,
            8,
            dtype=torch.bool,
            device=target_device,
        ),
        pair_ids=(clean.pair_id, null.pair_id),
        sample_ids=(clean.sample_id, null.sample_id),
        group_ids=(clean.group_id, null.group_id),
        pair_kinds=("clean_positive", "component_null"),
        projection_visible=(True, True),
    )
    completion_plus = torch.cat(
        (clean_state[2], null_state[2]),
        dim=0,
    ).to(target_device)
    completion_minus = torch.cat(
        (clean_state[3], null_state[3]),
        dim=0,
    ).to(target_device)
    return OutcomePairBatch(
        pair_batch=pair_batch,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        gt_union=completion_minus.clone(),
        intervention_footprint=direct_projected_intervention_footprint(
            pair_batch
        ),
    )


__all__ = [
    "CLEAN_PAIR_COUNT",
    "CLEAN_SLOT_COUNT",
    "COMPONENT_BLOCK_FAMILY",
    "COMPONENT_PAIR_COUNT",
    "COMPONENT_SLOT_COUNT",
    "COMPONENT_SPARSE_FAMILY",
    "CONFIRMATION_SEED",
    "CONTAINS_FAMILY",
    "ConfirmationPairSpec",
    "ConfirmationUpdate",
    "FACTUAL_BATCH_SIZE",
    "FACTUAL_EXPOSURES_PER_STATE",
    "FACTUAL_POPULATION_SIZE",
    "FACTUAL_SLOTS_PER_BRANCH",
    "OUTSIDE_FAMILY",
    "PAIR_BATCH_SIZE",
    "UPDATE_COUNT",
    "build_confirmation_factual_batches",
    "build_confirmation_factual_population",
    "build_confirmation_outcome_batch",
    "build_confirmation_pair_specs",
    "build_confirmation_schedule",
    "build_identical_input_conflict_control",
    "catalog_fingerprint",
    "catalog_manifest",
    "factual_indices_for_update",
    "factual_schedule_fingerprint",
    "factual_schedule_manifest",
    "schedule_fingerprint",
    "schedule_manifest",
]
