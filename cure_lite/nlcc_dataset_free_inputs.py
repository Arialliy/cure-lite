"""Reachability-aware dataset-free inputs for NLCC-v12.

This additive module defines input populations only.  It does not import a
dataset, detector, NLCC candidate decoder, optimizer, training runner, or
result artifact.  It reuses only the frozen generic occupancy-projection and
paired value-object helpers.
Every base geometry is represented by two matched rows.  The rows share one
source identity and differ only by a local channels-6/7 anchor witness and the
corresponding absolute completion target.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from .cache.schema import stable_fingerprint
from .decoder import project_occupancy_to_feature_grid
from .paired_outcome_types import (
    OutcomePairBatch,
    direct_projected_intervention_footprint,
)
from .paired_types import PairBatch, tensor_content_fingerprint
from .train.step import BranchBatch


FEATURE_CHANNELS = 8
FEATURE_HEIGHT = 7
FEATURE_WIDTH = 7
FEATURE_STRIDE = 4
OUTPUT_HEIGHT = 28
OUTPUT_WIDTH = 28
PAIR_BATCH_SIZE = 2
FACTUAL_POPULATION_SIZE = 16
FACTUAL_BATCH_SIZE = 4

ANCHOR_POSITIVE = "anchor_positive"
ANCHOR_NULL = "anchor_null"
ANCHOR_ROLES = (ANCHOR_POSITIVE, ANCHOR_NULL)

SAME_CELL_FAMILY = "same_cell"
ADJACENT_CELL_FAMILY = "adjacent_cell"
MULTICOUNT_2TO1_FAMILY = "multicount_2to1"
MULTICOUNT_3TO2_FAMILY = "multicount_3to2"
COMPONENT_NULL_BLOCK_FAMILY = "component_null_block"
COMPONENT_NULL_SPARSE_FAMILY = "component_null_sparse"

ONE_PIXEL_PHASE_PATTERN = ((0, 3),)
THREE_PIXEL_PHASE_PATTERN = ((0, 3), (1, 2), (3, 0))
ANCHOR_PHASE = (1, 1)
ANCHOR_CANDIDATE_CELLS = ((1, 1), (1, 5), (5, 1), (5, 5))

CLEAN_TARGET_SIGNAL_CHANNELS = (0, 1)
COMPONENT_NULL_SIGNAL_CHANNELS = (2, 3)
FACTUAL_MISS_SIGNAL_CHANNELS = (0, 1)
FACTUAL_NO_MISS_SIGNAL_CHANNELS = (4, 5)
COMPLETION_WITNESS_CHANNELS = (6, 7)

_GROUP_BASE = (
    ("clean_same_cell_1px", "clean_positive", SAME_CELL_FAMILY, 1),
    ("clean_same_cell_3px", "clean_positive", SAME_CELL_FAMILY, 3),
    (
        "clean_adjacent_cell_1px",
        "clean_positive",
        ADJACENT_CELL_FAMILY,
        1,
    ),
    (
        "clean_adjacent_cell_3px",
        "clean_positive",
        ADJACENT_CELL_FAMILY,
        3,
    ),
    (
        "clean_multicount_2to1",
        "clean_positive",
        MULTICOUNT_2TO1_FAMILY,
        None,
    ),
    (
        "clean_multicount_3to2",
        "clean_positive",
        MULTICOUNT_3TO2_FAMILY,
        None,
    ),
    (
        "component_null_block",
        "component_null",
        COMPONENT_NULL_BLOCK_FAMILY,
        0,
    ),
    (
        "component_null_sparse",
        "component_null",
        COMPONENT_NULL_SPARSE_FAMILY,
        0,
    ),
)

_PAIR_TENSOR_NAMES = (
    "feature",
    "occupancy_plus",
    "occupancy_minus",
    "label_increment",
    "image_valid_mask",
    "completion_plus",
    "completion_minus",
    "gt_union",
    "intervention_footprint",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bounded_nonzero_value(
    *,
    seed: int,
    stage: str,
    group: str,
    within_group_index: int,
    match_id: str,
    signal_role: str,
    channel: int,
    row: int,
    column: int,
) -> float:
    """Return a deterministic nonzero float on the frozen 1/64 lattice."""

    digest = hashlib.sha256(
        (
            f"{seed}|{stage}|{group}|{within_group_index}|{match_id}|"
            f"{signal_role}|{channel}|{row}|{column}"
        ).encode("utf-8")
    ).digest()
    unsigned = int.from_bytes(digest[:2], byteorder="big", signed=False)
    magnitude = 32 + unsigned % 97
    sign = 1.0 if digest[2] & 1 else -1.0
    return sign * float(magnitude) / 64.0


@dataclass(frozen=True)
class NLCCInputProfile:
    """One fully specified development or independent-holdout population."""

    profile_id: str
    design_seed: int
    update_count: int
    group_dyad_counts: tuple[int, ...]
    group_low_exposures: tuple[int, ...]
    group_high_exposures: tuple[int, ...]
    group_high_quotas: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if (
            isinstance(self.design_seed, bool)
            or not isinstance(self.design_seed, int)
            or not 0 <= self.design_seed <= 0xFFFFFFFF
        ):
            raise ValueError("design_seed must be a uint32")
        if (
            isinstance(self.update_count, bool)
            or not isinstance(self.update_count, int)
            or self.update_count < 1
        ):
            raise ValueError("update_count must be positive")
        values = (
            self.group_dyad_counts,
            self.group_low_exposures,
            self.group_high_exposures,
            self.group_high_quotas,
        )
        if any(not isinstance(value, tuple) or len(value) != 8 for value in values):
            raise ValueError("every group profile field must contain eight values")
        for index in range(8):
            count = self.group_dyad_counts[index]
            low = self.group_low_exposures[index]
            high = self.group_high_exposures[index]
            quota = self.group_high_quotas[index]
            if (
                any(isinstance(value, bool) or not isinstance(value, int) for value in
                    (count, low, high, quota))
                or count < 1
                or low < 1
                or high < low
                or not 0 <= quota <= count
            ):
                raise ValueError("invalid group exposure contract")
        slots_per_role = sum(
            quota * high + (count - quota) * low
            for count, low, high, quota in zip(
                self.group_dyad_counts,
                self.group_low_exposures,
                self.group_high_exposures,
                self.group_high_quotas,
                strict=True,
            )
        )
        if 2 * slots_per_role != self.update_count * PAIR_BATCH_SIZE:
            raise ValueError("profile exposure counts do not fill the update schedule")

    @property
    def dyad_count(self) -> int:
        return sum(self.group_dyad_counts)

    @property
    def row_count(self) -> int:
        return 2 * self.dyad_count

    @property
    def factual_exposures_per_state(self) -> int:
        slots = self.update_count * FACTUAL_BATCH_SIZE
        if slots % FACTUAL_POPULATION_SIZE:
            raise AssertionError("factual schedule is not population-balanced")
        return slots // FACTUAL_POPULATION_SIZE

    def manifest(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "design_seed": self.design_seed,
            "update_count": self.update_count,
            "group_dyad_counts": list(self.group_dyad_counts),
            "group_low_exposures": list(self.group_low_exposures),
            "group_high_exposures": list(self.group_high_exposures),
            "group_high_quotas": list(self.group_high_quotas),
            "dyad_count": self.dyad_count,
            "row_count": self.row_count,
            "factual_exposures_per_state": self.factual_exposures_per_state,
        }


@dataclass(frozen=True)
class NLCCPairSpec:
    """One immutable matched-twin population row."""

    population_index: int
    global_dyad_index: int
    within_group_dyad_index: int
    group_offset: int
    pair_id: str
    match_id: str
    sample_id: str
    group_id: str
    pair_kind: str
    anchor_role: str
    geometry_family: str
    response_pixel_count: int
    component_cell: tuple[int, int]
    response_cell: tuple[int, int]
    fixed_occupancy_cells: tuple[tuple[int, int], ...]
    anchor_cell: tuple[int, int]
    anchor_phase: tuple[int, int]
    exposure_count: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.population_index,
            self.global_dyad_index,
            self.within_group_dyad_index,
            self.group_offset,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_fields
        ):
            raise ValueError("pair indices must be nonnegative integers")
        for name, value in (("pair_id", self.pair_id), ("match_id", self.match_id)):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 value")
        if not self.sample_id or not self.group_id:
            raise ValueError("sample_id and group_id must be non-empty")
        if self.pair_kind not in {"clean_positive", "component_null"}:
            raise ValueError("pair_kind is outside the frozen outcome contract")
        if self.anchor_role not in ANCHOR_ROLES:
            raise ValueError("anchor_role is invalid")
        if self.response_pixel_count not in {0, 1, 3}:
            raise ValueError("response_pixel_count must be zero, one, or three")
        if (
            not 0 <= self.group_offset < len(_GROUP_BASE)
            or self.exposure_count < 1
        ):
            raise ValueError("group offset or exposure count is invalid")
        for name, cell in (
            ("component_cell", self.component_cell),
            ("response_cell", self.response_cell),
            ("anchor_cell", self.anchor_cell),
        ):
            if (
                not isinstance(cell, tuple)
                or len(cell) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < FEATURE_HEIGHT
                    for value in cell
                )
            ):
                raise ValueError(f"{name} must be a feature-grid cell")
        if not all(2 <= value <= 4 for value in self.component_cell):
            raise ValueError("component_cell must lie in the central 3x3 grid")
        if self.anchor_cell not in ANCHOR_CANDIDATE_CELLS:
            raise ValueError("anchor_cell is outside the frozen anchor ring")
        if self.anchor_phase != ANCHOR_PHASE:
            raise ValueError("anchor_phase differs from the frozen phase")
        primary_cells = (
            self.component_cell,
            self.response_cell,
            *self.fixed_occupancy_cells,
        )
        if any(
            max(
                abs(self.anchor_cell[0] - cell[0]),
                abs(self.anchor_cell[1] - cell[1]),
            )
            <= 1
            for cell in primary_cells
        ):
            raise ValueError("anchor overlaps a primary local feature/count support")
        expected_fixed = {
            MULTICOUNT_2TO1_FAMILY: 1,
            MULTICOUNT_3TO2_FAMILY: 2,
        }.get(self.geometry_family, 0)
        if len(self.fixed_occupancy_cells) != expected_fixed:
            raise ValueError("fixed occupancy count disagrees with geometry")
        if len(set(self.fixed_occupancy_cells)) != len(
            self.fixed_occupancy_cells
        ):
            raise ValueError("fixed occupancy cells must be unique")
        if self.component_cell in self.fixed_occupancy_cells:
            raise ValueError("fixed occupancy reuses the removed cell")
        if self.geometry_family == ADJACENT_CELL_FAMILY:
            distance = sum(
                abs(left - right)
                for left, right in zip(
                    self.component_cell,
                    self.response_cell,
                    strict=True,
                )
            )
            if distance != 1:
                raise ValueError("adjacent response is not cardinally adjacent")
        elif self.response_cell != self.component_cell:
            raise ValueError("non-adjacent response must use the component cell")
        if self.pair_kind == "clean_positive":
            if self.response_pixel_count not in {1, 3}:
                raise ValueError("clean rows require a nonempty response")
        elif self.response_pixel_count != 0 or self.fixed_occupancy_cells:
            raise ValueError("component-null rows require empty D and no fixed cells")

    def manifest(self) -> dict[str, object]:
        return {
            "population_index": self.population_index,
            "global_dyad_index": self.global_dyad_index,
            "within_group_dyad_index": self.within_group_dyad_index,
            "group_offset": self.group_offset,
            "pair_id": self.pair_id,
            "match_id": self.match_id,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "pair_kind": self.pair_kind,
            "anchor_role": self.anchor_role,
            "geometry_family": self.geometry_family,
            "response_pixel_count": self.response_pixel_count,
            "component_cell": list(self.component_cell),
            "response_cell": list(self.response_cell),
            "fixed_occupancy_cells": [
                list(cell) for cell in self.fixed_occupancy_cells
            ],
            "anchor_cell": list(self.anchor_cell),
            "anchor_phase": list(self.anchor_phase),
            "exposure_count": self.exposure_count,
        }


@dataclass(frozen=True)
class NLCCUpdate:
    """One deterministic positive/null two-row update."""

    update_index: int
    population_indices: tuple[int, int]
    pair_ids: tuple[str, str]
    match_ids: tuple[str, str]
    sample_ids: tuple[str, str]
    anchor_roles: tuple[str, str]

    def __post_init__(self) -> None:
        if (
            isinstance(self.update_index, bool)
            or not isinstance(self.update_index, int)
            or self.update_index < 0
        ):
            raise ValueError("update_index must be nonnegative")
        values = (
            self.population_indices,
            self.pair_ids,
            self.match_ids,
            self.sample_ids,
            self.anchor_roles,
        )
        if any(not isinstance(value, tuple) or len(value) != 2 for value in values):
            raise ValueError("every update field must contain exactly two values")
        if set(self.anchor_roles) != set(ANCHOR_ROLES):
            raise ValueError("each update requires one positive and one null row")
        if len(set(self.match_ids)) != 2 or len(set(self.sample_ids)) != 2:
            raise ValueError("matched twins cannot share one update")

    def manifest(self) -> dict[str, object]:
        return {
            "update_index": self.update_index,
            "population_indices": list(self.population_indices),
            "pair_ids": list(self.pair_ids),
            "match_ids": list(self.match_ids),
            "sample_ids": list(self.sample_ids),
            "anchor_roles": list(self.anchor_roles),
        }


@dataclass(frozen=True)
class NLCCStrata:
    """The frozen D/H/G-near/G-tail partition."""

    D: Tensor
    H: Tensor
    G_near: Tensor
    G_norm_tail: Tensor
    changed_count_support: Tensor

    def __post_init__(self) -> None:
        fields = (
            self.D,
            self.H,
            self.G_near,
            self.G_norm_tail,
            self.changed_count_support,
        )
        reference = self.D
        if any(
            not isinstance(value, Tensor)
            or value.dtype != torch.bool
            or value.shape != reference.shape
            or value.device != reference.device
            for value in fields
        ):
            raise TypeError("all strata must be aligned bool tensors")
        partition = (self.D, self.H, self.G_near, self.G_norm_tail)
        for index, left in enumerate(partition):
            for right in partition[index + 1 :]:
                if torch.any(left & right):
                    raise ValueError("D/H/G strata overlap")
        if not torch.all(self.G_norm_tail.flatten(1).any(dim=1)):
            raise ValueError("every row requires a nonempty normalization tail")


def _ranked_cardinal_neighbors(
    profile: NLCCInputProfile,
    *,
    group_id: str,
    dyad_index: int,
    cell: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    row, column = cell
    candidates = (
        (row, column + 1),
        (row + 1, column),
        (row, column - 1),
        (row - 1, column),
    )
    valid = tuple(
        value
        for value in candidates
        if 1 <= value[0] <= 5 and 1 <= value[1] <= 5
    )
    return tuple(
        sorted(
            valid,
            key=lambda value: (
                _sha(
                    f"{profile.design_seed}|neighbor|{group_id}|"
                    f"{dyad_index}|{value[0]}|{value[1]}"
                ),
                value,
            ),
        )
    )


def _anchor_cell(
    profile: NLCCInputProfile,
    *,
    group_id: str,
    dyad_index: int,
    primary_cells: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    eligible = tuple(
        candidate
        for candidate in ANCHOR_CANDIDATE_CELLS
        if all(
            max(
                abs(candidate[0] - cell[0]),
                abs(candidate[1] - cell[1]),
            )
            > 1
            for cell in primary_cells
        )
    )
    if not eligible:
        raise AssertionError("no reachability-safe anchor cell is available")
    return min(
        eligible,
        key=lambda value: (
            _sha(
                f"{profile.design_seed}|anchor|{group_id}|{dyad_index}|"
                f"{value[0]}|{value[1]}"
            ),
            value,
        ),
    )


def _high_exposure_indices(
    profile: NLCCInputProfile,
    *,
    group_id: str,
    count: int,
    quota: int,
) -> set[int]:
    ranked = sorted(
        range(count),
        key=lambda index: (
            _sha(
                f"{profile.design_seed}|exposure|{group_id}|{index}"
            ),
            index,
        ),
    )
    return set(ranked[:quota])


def build_pair_specs(profile: NLCCInputProfile) -> tuple[NLCCPairSpec, ...]:
    """Build the exact matched-twin population for ``profile``."""

    if not isinstance(profile, NLCCInputProfile):
        raise TypeError("profile must be NLCCInputProfile")
    specs: list[NLCCPairSpec] = []
    global_dyad_index = 0
    population_index = 0
    for group_offset, (
        group_id,
        pair_kind,
        family,
        frozen_pixels,
    ) in enumerate(_GROUP_BASE):
        count = profile.group_dyad_counts[group_offset]
        high_indices = _high_exposure_indices(
            profile,
            group_id=group_id,
            count=count,
            quota=profile.group_high_quotas[group_offset],
        )
        for within_group_index in range(count):
            component_cell = (
                2 + within_group_index % 3,
                2 + ((within_group_index // 3 + group_offset) % 3),
            )
            neighbors = _ranked_cardinal_neighbors(
                profile,
                group_id=group_id,
                dyad_index=within_group_index,
                cell=component_cell,
            )
            response_cell = (
                neighbors[0]
                if family == ADJACENT_CELL_FAMILY
                else component_cell
            )
            fixed_count = {
                MULTICOUNT_2TO1_FAMILY: 1,
                MULTICOUNT_3TO2_FAMILY: 2,
            }.get(family, 0)
            fixed_cells = neighbors[:fixed_count]
            anchor_cell = _anchor_cell(
                profile,
                group_id=group_id,
                dyad_index=within_group_index,
                primary_cells=(
                    component_cell,
                    response_cell,
                    *fixed_cells,
                ),
            )
            response_pixel_count = (
                (1 if within_group_index % 2 == 0 else 3)
                if frozen_pixels is None
                else frozen_pixels
            )
            dyad_identity = (
                f"nlcc-v12|{profile.profile_id}|{profile.design_seed}|"
                f"{group_id}|{within_group_index:03d}"
            )
            match_id = _sha(f"{dyad_identity}|match")
            sample_id = (
                f"nlcc-v12-{profile.profile_id}-source-{match_id[:24]}"
            )
            exposure_count = (
                profile.group_high_exposures[group_offset]
                if within_group_index in high_indices
                else profile.group_low_exposures[group_offset]
            )
            for role in ANCHOR_ROLES:
                specs.append(
                    NLCCPairSpec(
                        population_index=population_index,
                        global_dyad_index=global_dyad_index,
                        within_group_dyad_index=within_group_index,
                        group_offset=group_offset,
                        pair_id=_sha(f"{dyad_identity}|{role}|pair"),
                        match_id=match_id,
                        sample_id=sample_id,
                        group_id=group_id,
                        pair_kind=pair_kind,
                        anchor_role=role,
                        geometry_family=family,
                        response_pixel_count=int(response_pixel_count),
                        component_cell=component_cell,
                        response_cell=response_cell,
                        fixed_occupancy_cells=fixed_cells,
                        anchor_cell=anchor_cell,
                        anchor_phase=ANCHOR_PHASE,
                        exposure_count=exposure_count,
                    )
                )
                population_index += 1
            global_dyad_index += 1
    result = tuple(specs)
    _validate_population(profile, result)
    return result


def _validate_population(
    profile: NLCCInputProfile,
    specs: tuple[NLCCPairSpec, ...],
) -> None:
    if len(specs) != profile.row_count:
        raise AssertionError("population row count differs")
    if tuple(spec.population_index for spec in specs) != tuple(range(len(specs))):
        raise AssertionError("population indices are not canonical")
    if len({spec.pair_id for spec in specs}) != len(specs):
        raise AssertionError("pair IDs are not unique")
    by_match: dict[str, list[NLCCPairSpec]] = {}
    for spec in specs:
        by_match.setdefault(spec.match_id, []).append(spec)
    if len(by_match) != profile.dyad_count:
        raise AssertionError("dyad count differs")
    for values in by_match.values():
        if (
            len(values) != 2
            or {value.anchor_role for value in values} != set(ANCHOR_ROLES)
            or len({value.sample_id for value in values}) != 1
            or len({value.exposure_count for value in values}) != 1
        ):
            raise AssertionError("matched-twin population contract differs")
    expected_groups = {
        _GROUP_BASE[index][0]: 2 * count
        for index, count in enumerate(profile.group_dyad_counts)
    }
    if Counter(spec.group_id for spec in specs) != Counter(expected_groups):
        raise AssertionError("group row counts differ")
    if sum(spec.exposure_count for spec in specs) != (
        profile.update_count * PAIR_BATCH_SIZE
    ):
        raise AssertionError("population exposure total differs")
    role_slots = Counter()
    for spec in specs:
        role_slots[spec.anchor_role] += spec.exposure_count
    if role_slots != Counter(
        {
            ANCHOR_POSITIVE: profile.update_count,
            ANCHOR_NULL: profile.update_count,
        }
    ):
        raise AssertionError("positive/null exposure balance differs")


def _signal_feature(profile: NLCCInputProfile, spec: NLCCPairSpec) -> Tensor:
    feature = torch.zeros(
        1,
        FEATURE_CHANNELS,
        FEATURE_HEIGHT,
        FEATURE_WIDTH,
        dtype=torch.float32,
    )
    if spec.pair_kind == "clean_positive":
        channels = CLEAN_TARGET_SIGNAL_CHANNELS
        signal_cell = spec.response_cell
    else:
        channels = COMPONENT_NULL_SIGNAL_CHANNELS
        signal_cell = spec.component_cell
    for channel in channels:
        feature[0, channel, signal_cell[0], signal_cell[1]] = (
            _bounded_nonzero_value(
                seed=profile.design_seed,
                stage="pair",
                group=spec.group_id,
                within_group_index=spec.within_group_dyad_index,
                match_id=spec.match_id,
                signal_role="primary",
                channel=channel,
                row=signal_cell[0],
                column=signal_cell[1],
            )
        )
    if spec.anchor_role == ANCHOR_POSITIVE:
        for channel in COMPLETION_WITNESS_CHANNELS:
            feature[0, channel, spec.anchor_cell[0], spec.anchor_cell[1]] = (
                _bounded_nonzero_value(
                    seed=profile.design_seed,
                    stage="pair",
                    group=spec.group_id,
                    within_group_index=spec.within_group_dyad_index,
                    match_id=spec.match_id,
                    signal_role="anchor_witness",
                    channel=channel,
                    row=spec.anchor_cell[0],
                    column=spec.anchor_cell[1],
                )
            )
    return feature.contiguous()


def _cell_pixel(
    cell: tuple[int, int],
    phase: tuple[int, int],
) -> tuple[int, int]:
    return (
        FEATURE_STRIDE * cell[0] + phase[0],
        FEATURE_STRIDE * cell[1] + phase[1],
    )


def _set_projected_cell(
    mask: Tensor,
    cell: tuple[int, int],
    phases: Iterable[tuple[int, int]],
) -> None:
    for phase in phases:
        row, column = _cell_pixel(cell, phase)
        mask[0, 0, row, column] = True


def _pair_state(
    spec: NLCCPairSpec,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    occupancy_minus = torch.zeros(
        1,
        1,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
        dtype=torch.bool,
    )
    for cell in spec.fixed_occupancy_cells:
        _set_projected_cell(occupancy_minus, cell, ((2, 2),))
    occupancy_plus = occupancy_minus.clone()
    if spec.geometry_family == COMPONENT_NULL_BLOCK_FAMILY:
        component_phases = ((1, 1), (1, 2), (2, 1), (2, 2))
    elif spec.geometry_family == COMPONENT_NULL_SPARSE_FAMILY:
        component_phases = ((0, 0), (3, 3))
    else:
        component_phases = ((2, 2),)
    _set_projected_cell(
        occupancy_plus,
        spec.component_cell,
        component_phases,
    )

    completion_plus = torch.zeros_like(occupancy_plus)
    if spec.anchor_role == ANCHOR_POSITIVE:
        _set_projected_cell(
            completion_plus,
            spec.anchor_cell,
            (spec.anchor_phase,),
        )
    completion_minus = completion_plus.clone()
    if spec.pair_kind == "clean_positive":
        phases = (
            ONE_PIXEL_PHASE_PATTERN
            if spec.response_pixel_count == 1
            else THREE_PIXEL_PHASE_PATTERN
        )
        _set_projected_cell(completion_minus, spec.response_cell, phases)
    increment = (completion_minus & ~completion_plus).to(torch.float32)
    return (
        occupancy_plus.contiguous(),
        occupancy_minus.contiguous(),
        completion_plus.contiguous(),
        completion_minus.contiguous(),
        increment.contiguous(),
    )


def build_outcome_batch(
    profile: NLCCInputProfile,
    specs: Iterable[NLCCPairSpec],
    *,
    device: torch.device | str = "cpu",
) -> OutcomePairBatch:
    """Materialize any nonempty selection without exposing anchor role to loss."""

    if not isinstance(profile, NLCCInputProfile):
        raise TypeError("profile must be NLCCInputProfile")
    values = tuple(specs)
    if not values or any(not isinstance(spec, NLCCPairSpec) for spec in values):
        raise ValueError("specs must be a nonempty NLCCPairSpec selection")
    if len({spec.pair_id for spec in values}) != len(values):
        raise ValueError("one batch cannot repeat a pair")
    target_device = torch.device(device)
    states = tuple(_pair_state(spec) for spec in values)
    pair_batch = PairBatch(
        feature=torch.cat(
            [_signal_feature(profile, spec) for spec in values],
            dim=0,
        ).to(target_device),
        occupancy_plus=torch.cat([state[0] for state in states], dim=0).to(
            target_device
        ),
        occupancy_minus=torch.cat([state[1] for state in states], dim=0).to(
            target_device
        ),
        label_increment=torch.cat([state[4] for state in states], dim=0).to(
            target_device
        ),
        image_valid_mask=torch.ones(
            len(values),
            1,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
            dtype=torch.bool,
            device=target_device,
        ),
        pair_ids=tuple(spec.pair_id for spec in values),
        sample_ids=tuple(spec.sample_id for spec in values),
        group_ids=tuple(spec.group_id for spec in values),
        pair_kinds=tuple(spec.pair_kind for spec in values),
        projection_visible=tuple(True for _ in values),
    )
    completion_plus = torch.cat([state[2] for state in states], dim=0).to(
        target_device
    )
    completion_minus = torch.cat([state[3] for state in states], dim=0).to(
        target_device
    )
    return OutcomePairBatch(
        pair_batch=pair_batch,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        gt_union=completion_minus.clone(),
        intervention_footprint=direct_projected_intervention_footprint(
            pair_batch
        ),
    )


def build_strata(outcome: OutcomePairBatch) -> NLCCStrata:
    """Return D/H and the count-local/global split of G."""

    if not isinstance(outcome, OutcomePairBatch):
        raise TypeError("outcome must be OutcomePairBatch")
    outcome.validate()
    batch = outcome.pair_batch
    feature_size = tuple(int(value) for value in batch.feature.shape[-2:])
    plus = project_occupancy_to_feature_grid(
        batch.occupancy_plus,
        feature_size,
    ).float()
    minus = project_occupancy_to_feature_grid(
        batch.occupancy_minus,
        feature_size,
    ).float()
    kernel = torch.ones(1, 1, 3, 3, dtype=plus.dtype, device=plus.device)
    changed_native = (
        F.conv2d(plus, kernel, padding=1)
        != F.conv2d(minus, kernel, padding=1)
    )
    changed_count_support = F.interpolate(
        changed_native.float(),
        size=(OUTPUT_HEIGHT, OUTPUT_WIDTH),
        mode="nearest",
    ).bool()
    D = outcome.response_stratum
    H = outcome.local_zero_stratum
    G = outcome.global_zero_stratum
    G_near = G & changed_count_support
    G_norm_tail = G & ~changed_count_support
    if not torch.equal(
        D | H | G_near | G_norm_tail,
        batch.image_valid_mask,
    ):
        raise AssertionError("D/H/G do not partition the valid domain")
    return NLCCStrata(
        D=D.contiguous(),
        H=H.contiguous(),
        G_near=G_near.contiguous(),
        G_norm_tail=G_norm_tail.contiguous(),
        changed_count_support=changed_count_support.contiguous(),
    )


def _ranked_role_slots(
    profile: NLCCInputProfile,
    specs: tuple[NLCCPairSpec, ...],
    role: str,
) -> list[tuple[NLCCPairSpec, int]]:
    slots = [
        (spec, round_index)
        for spec in specs
        if spec.anchor_role == role
        for round_index in range(spec.exposure_count)
    ]
    return sorted(
        slots,
        key=lambda value: (
            _sha(
                f"{profile.design_seed}|schedule|{role}|"
                f"{value[1]}|{value[0].pair_id}"
            ),
            value[0].pair_id,
            value[1],
        ),
    )


def build_schedule(
    profile: NLCCInputProfile,
    specs: tuple[NLCCPairSpec, ...] | None = None,
) -> tuple[NLCCUpdate, ...]:
    """Pair one positive and one null slot from different matched dyads."""

    values = build_pair_specs(profile) if specs is None else tuple(specs)
    _validate_population(profile, values)
    positive = _ranked_role_slots(profile, values, ANCHOR_POSITIVE)
    null = _ranked_role_slots(profile, values, ANCHOR_NULL)
    if len(positive) != profile.update_count or len(null) != profile.update_count:
        raise AssertionError("role slot count differs")
    remaining_null = list(null)
    updates: list[NLCCUpdate] = []
    for update_index, positive_slot in enumerate(positive):
        null_index = next(
            (
                index
                for index, candidate in enumerate(remaining_null)
                if candidate[0].match_id != positive_slot[0].match_id
                and candidate[0].sample_id != positive_slot[0].sample_id
            ),
            None,
        )
        if null_index is None:
            raise AssertionError(
                "stable earliest-distinct null selection is exhausted"
            )
        null_slot = remaining_null.pop(null_index)
        first, second = positive_slot[0], null_slot[0]
        updates.append(
            NLCCUpdate(
                update_index=update_index,
                population_indices=(
                    first.population_index,
                    second.population_index,
                ),
                pair_ids=(first.pair_id, second.pair_id),
                match_ids=(first.match_id, second.match_id),
                sample_ids=(first.sample_id, second.sample_id),
                anchor_roles=(first.anchor_role, second.anchor_role),
            )
        )
    result = tuple(updates)
    if remaining_null:
        raise AssertionError("stable null schedule has unconsumed slots")
    observed = Counter(
        index for update in result for index in update.population_indices
    )
    expected = Counter(
        {
            spec.population_index: spec.exposure_count
            for spec in values
        }
    )
    if observed != expected:
        raise AssertionError("schedule exposures differ from the population")
    return result


def _factual_feature(
    profile: NLCCInputProfile,
    *,
    branch: str,
    state_index: int,
    signal_cell: tuple[int, int],
) -> Tensor:
    feature = torch.zeros(
        1,
        FEATURE_CHANNELS,
        FEATURE_HEIGHT,
        FEATURE_WIDTH,
        dtype=torch.float32,
    )
    if branch == "factual_miss":
        channels = FACTUAL_MISS_SIGNAL_CHANNELS
    elif branch == "factual_no_miss":
        channels = FACTUAL_NO_MISS_SIGNAL_CHANNELS
    else:
        raise ValueError("unknown factual branch")
    match_id = _sha(
        f"nlcc-v12|{profile.profile_id}|{profile.design_seed}|"
        f"factual|{branch}|{state_index:03d}"
    )
    for channel in channels:
        feature[0, channel, signal_cell[0], signal_cell[1]] = (
            _bounded_nonzero_value(
                seed=profile.design_seed,
                stage="factual",
                group=branch,
                within_group_index=state_index,
                match_id=match_id,
                signal_role=branch,
                channel=channel,
                row=signal_cell[0],
                column=signal_cell[1],
            )
        )
    return feature.contiguous()


def build_factual_population(
    profile: NLCCInputProfile,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, BranchBatch]:
    """Return sixteen factual states for each frozen absolute branch."""

    miss_features: list[Tensor] = []
    no_miss_features: list[Tensor] = []
    miss_targets: list[Tensor] = []
    for state_index in range(FACTUAL_POPULATION_SIZE):
        signal_cell = (
            2 + state_index % 3,
            2 + ((state_index // 3 + 1) % 3),
        )
        miss_features.append(
            _factual_feature(
                profile,
                branch="factual_miss",
                state_index=state_index,
                signal_cell=signal_cell,
            )
        )
        no_miss_features.append(
            _factual_feature(
                profile,
                branch="factual_no_miss",
                state_index=state_index,
                signal_cell=signal_cell,
            )
        )
        target = torch.zeros(
            1,
            1,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
            dtype=torch.float32,
        )
        phases = (
            ONE_PIXEL_PHASE_PATTERN
            if state_index % 2 == 0
            else THREE_PIXEL_PHASE_PATTERN
        )
        for phase in phases:
            row, column = _cell_pixel(signal_cell, phase)
            target[0, 0, row, column] = 1.0
        miss_targets.append(target)
    target_device = torch.device(device)
    miss_feature = torch.cat(miss_features, dim=0).to(target_device)
    no_miss_feature = torch.cat(no_miss_features, dim=0).to(target_device)
    miss_target = torch.cat(miss_targets, dim=0).to(target_device)
    occupancy = torch.zeros(
        FACTUAL_POPULATION_SIZE,
        1,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
        dtype=torch.bool,
        device=target_device,
    )
    valid = torch.ones_like(occupancy)
    return {
        "factual_miss": BranchBatch(
            feature=miss_feature,
            occupancy=occupancy,
            target=miss_target,
            valid_mask=valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=no_miss_feature,
            occupancy=occupancy.clone(),
            target=torch.zeros_like(miss_target),
            valid_mask=valid.clone(),
        ),
    }


def factual_indices_for_update(
    profile: NLCCInputProfile,
    update_index: int,
) -> tuple[int, ...]:
    if (
        isinstance(update_index, bool)
        or not isinstance(update_index, int)
        or not 0 <= update_index < profile.update_count
    ):
        raise ValueError("update_index is outside the profile schedule")
    return tuple(
        (update_index + offset) % FACTUAL_POPULATION_SIZE
        for offset in range(FACTUAL_BATCH_SIZE)
    )


def build_factual_batches(
    profile: NLCCInputProfile,
    *,
    update_index: int,
    device: torch.device | str = "cpu",
) -> dict[str, BranchBatch]:
    populations = build_factual_population(profile, device=device)
    target_device = torch.device(device)
    indices = torch.tensor(
        factual_indices_for_update(profile, update_index),
        dtype=torch.int64,
        device=target_device,
    )
    return {
        branch: BranchBatch(
            feature=batch.feature.index_select(0, indices),
            occupancy=batch.occupancy.index_select(0, indices),
            target=batch.target.index_select(0, indices),
            valid_mask=batch.valid_mask.index_select(0, indices),
        )
        for branch, batch in populations.items()
    }


def pair_tensor_manifest(
    profile: NLCCInputProfile,
    spec: NLCCPairSpec,
) -> dict[str, object]:
    outcome = build_outcome_batch(profile, (spec,))
    tensors = {
        "feature": outcome.pair_batch.feature,
        "occupancy_plus": outcome.pair_batch.occupancy_plus,
        "occupancy_minus": outcome.pair_batch.occupancy_minus,
        "label_increment": outcome.pair_batch.label_increment,
        "image_valid_mask": outcome.pair_batch.image_valid_mask,
        "completion_plus": outcome.completion_plus,
        "completion_minus": outcome.completion_minus,
        "gt_union": outcome.gt_union,
        "intervention_footprint": outcome.intervention_footprint,
    }
    return {
        **spec.manifest(),
        "tensor_fingerprints": {
            name: tensor_content_fingerprint(tensors[name])
            for name in _PAIR_TENSOR_NAMES
        },
    }


def catalog_manifest(
    profile: NLCCInputProfile,
    specs: tuple[NLCCPairSpec, ...] | None = None,
) -> list[dict[str, object]]:
    values = build_pair_specs(profile) if specs is None else tuple(specs)
    return [pair_tensor_manifest(profile, spec) for spec in values]


def catalog_fingerprint(
    profile: NLCCInputProfile,
    specs: tuple[NLCCPairSpec, ...] | None = None,
) -> str:
    return stable_fingerprint(catalog_manifest(profile, specs))


def schedule_manifest(
    profile: NLCCInputProfile,
    specs: tuple[NLCCPairSpec, ...] | None = None,
) -> list[dict[str, object]]:
    return [update.manifest() for update in build_schedule(profile, specs)]


def schedule_fingerprint(
    profile: NLCCInputProfile,
    specs: tuple[NLCCPairSpec, ...] | None = None,
) -> str:
    return stable_fingerprint(schedule_manifest(profile, specs))


def factual_population_manifest(
    profile: NLCCInputProfile,
) -> list[dict[str, object]]:
    population = build_factual_population(profile)
    rows: list[dict[str, object]] = []
    for branch in ("factual_miss", "factual_no_miss"):
        batch = population[branch]
        for state_index in range(FACTUAL_POPULATION_SIZE):
            rows.append(
                {
                    "branch": branch,
                    "state_index": state_index,
                    "tensor_fingerprints": {
                        "feature": tensor_content_fingerprint(
                            batch.feature[state_index : state_index + 1]
                        ),
                        "occupancy": tensor_content_fingerprint(
                            batch.occupancy[state_index : state_index + 1]
                        ),
                        "target": tensor_content_fingerprint(
                            batch.target[state_index : state_index + 1]
                        ),
                        "valid_mask": tensor_content_fingerprint(
                            batch.valid_mask[state_index : state_index + 1]
                        ),
                    },
                }
            )
    return rows


def factual_population_fingerprint(profile: NLCCInputProfile) -> str:
    return stable_fingerprint(factual_population_manifest(profile))


def factual_schedule_manifest(
    profile: NLCCInputProfile,
) -> list[dict[str, object]]:
    return [
        {
            "update_index": update_index,
            "factual_miss_indices": list(
                factual_indices_for_update(profile, update_index)
            ),
            "factual_no_miss_indices": list(
                factual_indices_for_update(profile, update_index)
            ),
        }
        for update_index in range(profile.update_count)
    ]


def factual_schedule_fingerprint(profile: NLCCInputProfile) -> str:
    return stable_fingerprint(factual_schedule_manifest(profile))


def _native_count(
    occupancy: Tensor,
) -> Tensor:
    projected = project_occupancy_to_feature_grid(
        occupancy,
        (FEATURE_HEIGHT, FEATURE_WIDTH),
    ).float()
    kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
    return F.conv2d(projected, kernel, padding=1)


def _patch_digests(feature: Tensor) -> dict[tuple[int, int], str]:
    padded = F.pad(feature, (1, 1, 1, 1))
    return {
        (row, column): tensor_content_fingerprint(
            padded[:, :, row : row + 3, column : column + 3]
        )
        for row in range(FEATURE_HEIGHT)
        for column in range(FEATURE_WIDTH)
    }


def _add_label(
    table: dict[tuple[object, ...], int],
    key: tuple[object, ...],
    label: bool,
) -> None:
    table[key] = table.get(key, 0) | (2 if label else 1)


def _collision_count(table: dict[tuple[object, ...], int]) -> int:
    return sum(value == 3 for value in table.values())


def reachability_audit(
    profile: NLCCInputProfile,
    specs: tuple[NLCCPairSpec, ...] | None = None,
) -> dict[str, object]:
    """Audit exact twins and weight-free local reachability before training."""

    values = build_pair_specs(profile) if specs is None else tuple(specs)
    _validate_population(profile, values)
    population_outcome = build_outcome_batch(profile, values)
    population_batch = population_outcome.pair_batch
    population_D = population_outcome.response_stratum
    population_strata = build_strata(population_outcome)
    by_match: dict[str, dict[str, NLCCPairSpec]] = {}
    for spec in values:
        by_match.setdefault(spec.match_id, {})[spec.anchor_role] = spec

    twin_counts = Counter(
        {
            "orphan_twin_count": 0,
            "shared_sample_id_failure_count": 0,
            "feature_difference_mismatch_count": 0,
            "occupancy_difference_count": 0,
            "transition_truth_difference_count": 0,
            "completion_difference_mismatch_count": 0,
            "footprint_difference_count": 0,
        }
    )
    for roles in by_match.values():
        if set(roles) != set(ANCHOR_ROLES):
            twin_counts["orphan_twin_count"] += 1
            continue
        positive = roles[ANCHOR_POSITIVE]
        null = roles[ANCHOR_NULL]
        if positive.sample_id != null.sample_id:
            twin_counts["shared_sample_id_failure_count"] += 1
        positive_index = positive.population_index
        null_index = null.population_index
        feature_diff = (
            population_batch.feature[positive_index : positive_index + 1]
            != population_batch.feature[null_index : null_index + 1]
        )
        expected_feature_diff = torch.zeros_like(feature_diff)
        for channel in COMPLETION_WITNESS_CHANNELS:
            expected_feature_diff[
                0,
                channel,
                positive.anchor_cell[0],
                positive.anchor_cell[1],
            ] = True
        if not torch.equal(feature_diff, expected_feature_diff):
            twin_counts["feature_difference_mismatch_count"] += 1
        for field in ("occupancy_plus", "occupancy_minus"):
            if not torch.equal(
                getattr(population_batch, field)[positive_index],
                getattr(population_batch, field)[null_index],
            ):
                twin_counts["occupancy_difference_count"] += 1
        if not torch.equal(
            population_batch.label_increment[positive_index],
            population_batch.label_increment[null_index],
        ):
            twin_counts["transition_truth_difference_count"] += 1
        expected_anchor = torch.zeros_like(
            population_outcome.completion_plus[positive_index]
        )
        anchor_row, anchor_column = _cell_pixel(
            positive.anchor_cell,
            positive.anchor_phase,
        )
        expected_anchor[0, anchor_row, anchor_column] = True
        for field in ("completion_plus", "completion_minus", "gt_union"):
            if not torch.equal(
                getattr(population_outcome, field)[positive_index]
                ^ getattr(population_outcome, field)[null_index],
                expected_anchor,
            ):
                twin_counts["completion_difference_mismatch_count"] += 1
        if not torch.equal(
            population_outcome.intervention_footprint[positive_index],
            population_outcome.intervention_footprint[null_index],
        ):
            twin_counts["footprint_difference_count"] += 1

    schedule = build_schedule(profile, values)
    integrity_counts = Counter(
        {
            "same_match_within_update_count": sum(
                len(set(update.match_ids)) != 2 for update in schedule
            ),
            "same_sample_within_update_count": sum(
                len(set(update.sample_ids)) != 2 for update in schedule
            ),
            "same_pair_within_update_count": sum(
                len(set(update.pair_ids)) != 2 for update in schedule
            ),
            "nonfinite_input_count": 0,
            "twin_D_H_G_partition_difference_count": 0,
        }
    )
    for roles in by_match.values():
        if set(roles) != set(ANCHOR_ROLES):
            continue
        positive_index = roles[ANCHOR_POSITIVE].population_index
        null_index = roles[ANCHOR_NULL].population_index
        if any(
            not torch.equal(
                getattr(population_strata, field)[positive_index],
                getattr(population_strata, field)[null_index],
            )
            for field in ("D", "H", "G_near", "G_norm_tail")
        ):
            integrity_counts[
                "twin_D_H_G_partition_difference_count"
            ] += 1

    floating_inputs = (
        population_batch.feature,
        population_batch.label_increment,
    )
    integrity_counts["nonfinite_input_count"] += sum(
        int((~torch.isfinite(tensor)).sum().item())
        for tensor in floating_inputs
    )

    required = Counter(
        {
            "unwitnessed_completion_count": 0,
            "completion_changed_support_overlap_count": 0,
            "opposite_label_identical_input_count": 0,
            "opposite_label_local_signature_conflict_count": 0,
            "clean_D_without_feature_witness_count": 0,
            "clean_D_without_count_difference_count": 0,
        }
    )
    exact_labels: dict[tuple[object, ...], int] = {}
    local_labels: dict[tuple[object, ...], int] = {}

    for spec in values:
        index = spec.population_index
        feature = population_batch.feature[index : index + 1]
        occupancy_plus = population_batch.occupancy_plus[index : index + 1]
        occupancy_minus = population_batch.occupancy_minus[index : index + 1]
        count_plus = _native_count(occupancy_plus)
        count_minus = _native_count(occupancy_minus)
        changed = count_plus != count_minus
        feature_hash = tensor_content_fingerprint(feature)
        plus_hash = tensor_content_fingerprint(occupancy_plus)
        minus_hash = tensor_content_fingerprint(occupancy_minus)
        patch_hashes = _patch_digests(feature)
        plus_counts = count_plus[0, 0].to(torch.int64).tolist()
        minus_counts = count_minus[0, 0].to(torch.int64).tolist()

        completion_pixels = torch.nonzero(
            population_outcome.completion_plus[index, 0],
            as_tuple=False,
        ).tolist()
        for output_row, output_column in completion_pixels:
            cell = (
                output_row // FEATURE_STRIDE,
                output_column // FEATURE_STRIDE,
            )
            witness = feature[
                0,
                list(COMPLETION_WITNESS_CHANNELS),
                cell[0],
                cell[1],
            ]
            if not bool(torch.all(witness != 0.0)):
                required["unwitnessed_completion_count"] += 1
            if bool(changed[0, 0, cell[0], cell[1]]):
                required["completion_changed_support_overlap_count"] += 1

        D = population_D[index : index + 1]
        if spec.pair_kind == "clean_positive":
            for output_row, output_column in torch.nonzero(
                D[0, 0],
                as_tuple=False,
            ).tolist():
                cell = (
                    output_row // FEATURE_STRIDE,
                    output_column // FEATURE_STRIDE,
                )
                primary = feature[
                    0,
                    list(CLEAN_TARGET_SIGNAL_CHANNELS),
                    cell[0],
                    cell[1],
                ]
                if not bool(torch.all(primary != 0.0)):
                    required["clean_D_without_feature_witness_count"] += 1
                if not bool(changed[0, 0, cell[0], cell[1]]):
                    required["clean_D_without_count_difference_count"] += 1

        anchor_background = (
            population_batch.image_valid_mask[index : index + 1]
            & ~occupancy_plus
            & ~population_outcome.gt_union[index : index + 1]
        )
        completion_plus = population_outcome.completion_plus[
            index : index + 1
        ]
        absolute_valid = completion_plus | anchor_background
        for output_row, output_column in torch.nonzero(
            absolute_valid[0, 0],
            as_tuple=False,
        ).tolist():
            label = bool(
                completion_plus[
                    0,
                    0,
                    output_row,
                    output_column,
                ]
            )
            cell = (
                output_row // FEATURE_STRIDE,
                output_column // FEATURE_STRIDE,
            )
            phase = (
                output_row % FEATURE_STRIDE,
                output_column % FEATURE_STRIDE,
            )
            _add_label(
                exact_labels,
                ("absolute", feature_hash, plus_hash, output_row, output_column),
                label,
            )
            _add_label(
                local_labels,
                (
                    "absolute",
                    phase,
                    patch_hashes[cell],
                    plus_counts[cell[0]][cell[1]],
                ),
                label,
            )
        D_values = D[0, 0].tolist()
        for output_row in range(OUTPUT_HEIGHT):
            for output_column in range(OUTPUT_WIDTH):
                label = bool(D_values[output_row][output_column])
                cell = (
                    output_row // FEATURE_STRIDE,
                    output_column // FEATURE_STRIDE,
                )
                phase = (
                    output_row % FEATURE_STRIDE,
                    output_column % FEATURE_STRIDE,
                )
                _add_label(
                    exact_labels,
                    (
                        "transition",
                        feature_hash,
                        plus_hash,
                        minus_hash,
                        output_row,
                        output_column,
                    ),
                    label,
                )
                _add_label(
                    local_labels,
                    (
                        "transition",
                        phase,
                        patch_hashes[cell],
                        plus_counts[cell[0]][cell[1]],
                        minus_counts[cell[0]][cell[1]],
                    ),
                    label,
                )

    factual = build_factual_population(profile)
    for branch, batch in factual.items():
        integrity_counts["nonfinite_input_count"] += int(
            (~torch.isfinite(batch.feature)).sum().item()
            + (~torch.isfinite(batch.target)).sum().item()
        )
        for state_index in range(FACTUAL_POPULATION_SIZE):
            feature = batch.feature[state_index : state_index + 1]
            occupancy = batch.occupancy[state_index : state_index + 1]
            target = batch.target[state_index : state_index + 1]
            count = _native_count(occupancy)
            feature_hash = tensor_content_fingerprint(feature)
            occupancy_hash = tensor_content_fingerprint(occupancy)
            patch_hashes = _patch_digests(feature)
            count_values = count[0, 0].to(torch.int64).tolist()
            target_values = target[0, 0].tolist()
            for output_row in range(OUTPUT_HEIGHT):
                for output_column in range(OUTPUT_WIDTH):
                    label = bool(target_values[output_row][output_column])
                    cell = (
                        output_row // FEATURE_STRIDE,
                        output_column // FEATURE_STRIDE,
                    )
                    phase = (
                        output_row % FEATURE_STRIDE,
                        output_column % FEATURE_STRIDE,
                    )
                    _add_label(
                        exact_labels,
                        (
                            "absolute",
                            feature_hash,
                            occupancy_hash,
                            output_row,
                            output_column,
                        ),
                        label,
                    )
                    _add_label(
                        local_labels,
                        (
                            "absolute",
                            phase,
                            patch_hashes[cell],
                            count_values[cell[0]][cell[1]],
                        ),
                        label,
                    )

    required["opposite_label_identical_input_count"] = _collision_count(
        exact_labels
    )
    required[
        "opposite_label_local_signature_conflict_count"
    ] = _collision_count(local_labels)
    required_dict = dict(required)
    twin_dict = dict(twin_counts)
    integrity_dict = dict(integrity_counts)
    return {
        "schema_version": "cure-lite.nlcc-v12.reachability-audit.v1",
        "profile_id": profile.profile_id,
        "design_seed": profile.design_seed,
        "population_counts": {
            "dyads": profile.dyad_count,
            "rows": profile.row_count,
            "anchor_positive_rows": sum(
                spec.anchor_role == ANCHOR_POSITIVE for spec in values
            ),
            "anchor_null_rows": sum(
                spec.anchor_role == ANCHOR_NULL for spec in values
            ),
        },
        "required_zero_counts": required_dict,
        "twin_integrity_counts": twin_dict,
        "input_integrity_counts": integrity_dict,
        "all_pass": (
            all(value == 0 for value in required_dict.values())
            and all(value == 0 for value in twin_dict.values())
            and all(value == 0 for value in integrity_dict.values())
        ),
    }


def input_manifest(
    profile: NLCCInputProfile,
    specs: tuple[NLCCPairSpec, ...] | None = None,
) -> dict[str, object]:
    values = build_pair_specs(profile) if specs is None else tuple(specs)
    return {
        "schema_version": "cure-lite.nlcc-v12.dataset-free-inputs.v1",
        "profile": profile.manifest(),
        "input_contract": {
            "feature_shape": [
                1,
                FEATURE_CHANNELS,
                FEATURE_HEIGHT,
                FEATURE_WIDTH,
            ],
            "output_shape": [1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH],
            "feature_stride": FEATURE_STRIDE,
            "pair_batch_size": PAIR_BATCH_SIZE,
            "factual_population_size": FACTUAL_POPULATION_SIZE,
            "factual_batch_size": FACTUAL_BATCH_SIZE,
            "anchor_phase": list(ANCHOR_PHASE),
            "completion_witness_channels": list(
                COMPLETION_WITNESS_CHANNELS
            ),
            "pair_kinds": ["clean_positive", "component_null"],
            "anchor_roles_are_model_inputs": False,
            "match_ids_are_model_inputs": False,
        },
        "group_row_counts": dict(Counter(spec.group_id for spec in values)),
        "anchor_role_counts": dict(
            Counter(spec.anchor_role for spec in values)
        ),
        "catalog_fingerprint": catalog_fingerprint(profile, values),
        "schedule_fingerprint": schedule_fingerprint(profile, values),
        "factual_population_fingerprint": factual_population_fingerprint(
            profile
        ),
        "factual_schedule_fingerprint": factual_schedule_fingerprint(profile),
        "reachability_audit": reachability_audit(profile, values),
    }


def input_fingerprint(
    profile: NLCCInputProfile,
    specs: tuple[NLCCPairSpec, ...] | None = None,
) -> str:
    return stable_fingerprint(input_manifest(profile, specs))


__all__ = [
    "ADJACENT_CELL_FAMILY",
    "ANCHOR_NULL",
    "ANCHOR_PHASE",
    "ANCHOR_POSITIVE",
    "ANCHOR_ROLES",
    "COMPONENT_NULL_BLOCK_FAMILY",
    "COMPONENT_NULL_SPARSE_FAMILY",
    "COMPLETION_WITNESS_CHANNELS",
    "FACTUAL_BATCH_SIZE",
    "FACTUAL_POPULATION_SIZE",
    "FEATURE_CHANNELS",
    "FEATURE_HEIGHT",
    "FEATURE_STRIDE",
    "FEATURE_WIDTH",
    "MULTICOUNT_2TO1_FAMILY",
    "MULTICOUNT_3TO2_FAMILY",
    "NLCCInputProfile",
    "NLCCPairSpec",
    "NLCCStrata",
    "NLCCUpdate",
    "ONE_PIXEL_PHASE_PATTERN",
    "OUTPUT_HEIGHT",
    "OUTPUT_WIDTH",
    "PAIR_BATCH_SIZE",
    "SAME_CELL_FAMILY",
    "THREE_PIXEL_PHASE_PATTERN",
    "build_factual_batches",
    "build_factual_population",
    "build_outcome_batch",
    "build_pair_specs",
    "build_schedule",
    "build_strata",
    "catalog_fingerprint",
    "catalog_manifest",
    "factual_indices_for_update",
    "factual_population_fingerprint",
    "factual_population_manifest",
    "factual_schedule_fingerprint",
    "factual_schedule_manifest",
    "input_fingerprint",
    "input_manifest",
    "pair_tensor_manifest",
    "reachability_audit",
    "schedule_fingerprint",
    "schedule_manifest",
]
