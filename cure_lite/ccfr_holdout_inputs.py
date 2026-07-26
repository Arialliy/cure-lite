"""Frozen dataset-free exposure holdout inputs for CCFR v11.

This module materializes the population that was frozen in
``exposure_holdout_design_receipt.json`` before the CCFR development result:

* a new 5x5 feature grid and 20x20 output grid;
* 206 clean-positive and 16 component-null pairs in eight exact groups;
* a new deterministic 800-slot/400-update schedule;
* independent 16-state factual-miss and factual-no-miss populations.

No dataset, detector, cache, training runner, or result file is imported.
Pair kind remains audit metadata: model inputs are only the feature tensor and
the two occupancy endpoints, while supervision is carried by tensor truth.
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


DESIGN_SEED = 3987573916
FEATURE_CHANNELS = 8
FEATURE_HEIGHT = 5
FEATURE_WIDTH = 5
FEATURE_STRIDE = 4
OUTPUT_HEIGHT = 20
OUTPUT_WIDTH = 20

PAIR_BATCH_SIZE = 2
UPDATE_COUNT = 400
TOTAL_PAIR_SLOTS = 800
CLEAN_PAIR_COUNT = 206
COMPONENT_NULL_PAIR_COUNT = 16
CLEAN_SLOT_COUNT = 739
COMPONENT_NULL_SLOT_COUNT = 61

FACTUAL_POPULATION_SIZE = 16
FACTUAL_BATCH_SIZE = 4
FACTUAL_EXPOSURES_PER_STATE = 100

SAME_CELL_FAMILY = "same_cell"
ADJACENT_CELL_FAMILY = "adjacent_cell"
MULTICOUNT_2TO1_FAMILY = "multicount_2to1"
MULTICOUNT_3TO2_FAMILY = "multicount_3to2"
COMPONENT_NULL_BLOCK_FAMILY = "component_null_block"
COMPONENT_NULL_SPARSE_FAMILY = "component_null_sparse"

ONE_PIXEL_PHASE_PATTERN = ((0, 3),)
THREE_PIXEL_PHASE_PATTERN = ((0, 3), (1, 2), (3, 0))

CLEAN_TARGET_SIGNAL_CHANNELS = (0, 1)
COMPONENT_NULL_SIGNAL_CHANNELS = (2, 3)
FACTUAL_MISS_SIGNAL_CHANNELS = (0, 1)
FACTUAL_NO_MISS_SIGNAL_CHANNELS = (4, 5)

_GROUP_CONTRACT = (
    ("clean_same_cell_1px", "clean_positive", SAME_CELL_FAMILY, 35, 1),
    ("clean_same_cell_3px", "clean_positive", SAME_CELL_FAMILY, 34, 3),
    (
        "clean_adjacent_cell_1px",
        "clean_positive",
        ADJACENT_CELL_FAMILY,
        35,
        1,
    ),
    (
        "clean_adjacent_cell_3px",
        "clean_positive",
        ADJACENT_CELL_FAMILY,
        34,
        3,
    ),
    (
        "clean_multicount_2to1",
        "clean_positive",
        MULTICOUNT_2TO1_FAMILY,
        34,
        None,
    ),
    (
        "clean_multicount_3to2",
        "clean_positive",
        MULTICOUNT_3TO2_FAMILY,
        34,
        None,
    ),
    (
        "component_null_block",
        "component_null",
        COMPONENT_NULL_BLOCK_FAMILY,
        8,
        0,
    ),
    (
        "component_null_sparse",
        "component_null",
        COMPONENT_NULL_SPARSE_FAMILY,
        8,
        0,
    ),
)

GROUP_COUNTS = {
    group_id: count
    for group_id, _kind, _family, count, _pixels in _GROUP_CONTRACT
}

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


def _lattice_value(
    *,
    group_id: str,
    within_group_index: int,
    channel: int,
    row: int,
    column: int,
) -> float:
    """Return one frozen SHA256-indexed integer-lattice value."""

    key = (
        f"{DESIGN_SEED}|{group_id}|{within_group_index}|"
        f"{channel}|{row}|{column}"
    )
    unsigned = int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:2],
        byteorder="big",
        signed=False,
    )
    return float((unsigned % 257) - 128) / 64.0


def _feature_cell(
    *,
    within_group_index: int,
    group_offset: int,
) -> tuple[int, int]:
    """Apply the receipt's exact interior-cell index rule."""

    return (
        1 + ((3 * within_group_index + group_offset) % 3),
        1 + ((2 * within_group_index + 1 + group_offset) % 3),
    )


def _cardinal_neighbors(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    row, column = cell
    values = (
        (row, column + 1),
        (row + 1, column),
        (row, column - 1),
        (row - 1, column),
    )
    return tuple(
        value
        for value in values
        if 1 <= value[0] < FEATURE_HEIGHT - 1
        and 1 <= value[1] < FEATURE_WIDTH - 1
    )


def _ranked_neighbors(
    *,
    group_id: str,
    within_group_index: int,
    cell: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            _cardinal_neighbors(cell),
            key=lambda value: (
                _sha(
                    f"{DESIGN_SEED}|neighbor|{group_id}|"
                    f"{within_group_index}|{value[0]}|{value[1]}"
                ),
                value,
            ),
        )
    )


@dataclass(frozen=True)
class CCFRHoldoutPairSpec:
    """One immutable holdout population row."""

    population_index: int
    within_group_index: int
    group_offset: int
    pair_id: str
    sample_id: str
    group_id: str
    pair_kind: str
    geometry_family: str
    response_pixel_count: int
    component_cell: tuple[int, int]
    response_cell: tuple[int, int]
    fixed_occupancy_cells: tuple[tuple[int, int], ...]
    exposure_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.population_index, bool)
            or not isinstance(self.population_index, int)
            or self.population_index < 0
        ):
            raise ValueError("population_index must be nonnegative")
        if (
            isinstance(self.within_group_index, bool)
            or not isinstance(self.within_group_index, int)
            or self.within_group_index < 0
        ):
            raise ValueError("within_group_index must be nonnegative")
        if (
            isinstance(self.group_offset, bool)
            or not isinstance(self.group_offset, int)
            or not 0 <= self.group_offset < len(_GROUP_CONTRACT)
        ):
            raise ValueError("group_offset is outside the frozen group range")
        if len(self.pair_id) != 64:
            raise ValueError("pair_id must be a SHA256 fingerprint")
        if not self.sample_id or not self.group_id:
            raise ValueError("sample_id and group_id must be non-empty")
        if self.pair_kind not in {"clean_positive", "component_null"}:
            raise ValueError("pair_kind is outside the holdout role contract")
        if self.exposure_count not in {3, 4}:
            raise ValueError("every holdout pair requires 3 or 4 exposures")
        for name, cell in (
            ("component_cell", self.component_cell),
            ("response_cell", self.response_cell),
        ):
            if (
                not isinstance(cell, tuple)
                or len(cell) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < 5
                    for value in cell
                )
            ):
                raise ValueError(f"{name} must be a 5x5-grid cell")
        if not all(1 <= value <= 3 for value in self.component_cell):
            raise ValueError("component_cell must be an interior feature cell")
        if len(set(self.fixed_occupancy_cells)) != len(
            self.fixed_occupancy_cells
        ):
            raise ValueError("fixed occupancy cells must be unique")
        if self.component_cell in self.fixed_occupancy_cells:
            raise ValueError("fixed occupancy cannot reuse the removed cell")
        if any(
            len(cell) != 2
            or any(not 0 <= value < 5 for value in cell)
            for cell in self.fixed_occupancy_cells
        ):
            raise ValueError("fixed occupancy lies outside the feature grid")

        expected_fixed = {
            MULTICOUNT_2TO1_FAMILY: 1,
            MULTICOUNT_3TO2_FAMILY: 2,
        }.get(self.geometry_family, 0)
        if len(self.fixed_occupancy_cells) != expected_fixed:
            raise ValueError("fixed occupancy count disagrees with geometry")
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
                raise ValueError("adjacent response must be cardinally adjacent")
        elif self.response_cell != self.component_cell:
            raise ValueError(
                "non-adjacent groups require response/component cell identity"
            )

        if self.pair_kind == "clean_positive":
            if self.response_pixel_count not in {1, 3}:
                raise ValueError("clean response must contain one or three pixels")
        else:
            if self.response_pixel_count != 0:
                raise ValueError("component-null response must be empty")
            if self.fixed_occupancy_cells:
                raise ValueError("component-null rows cannot carry fixed occupancy")

    def manifest(self) -> dict[str, object]:
        return {
            "population_index": self.population_index,
            "within_group_index": self.within_group_index,
            "group_offset": self.group_offset,
            "pair_id": self.pair_id,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "pair_kind": self.pair_kind,
            "geometry_family": self.geometry_family,
            "response_pixel_count": self.response_pixel_count,
            "component_cell": list(self.component_cell),
            "response_cell": list(self.response_cell),
            "fixed_occupancy_cells": [
                list(cell) for cell in self.fixed_occupancy_cells
            ],
            "exposure_count": self.exposure_count,
        }


@dataclass(frozen=True)
class CCFRHoldoutUpdate:
    """One deterministic two-pair optimizer update."""

    update_index: int
    population_indices: tuple[int, int]
    pair_ids: tuple[str, str]
    pair_kinds: tuple[str, str]
    sample_ids: tuple[str, str]

    def __post_init__(self) -> None:
        if (
            isinstance(self.update_index, bool)
            or not isinstance(self.update_index, int)
            or self.update_index < 0
        ):
            raise ValueError("update_index must be nonnegative")
        for name, values in (
            ("population_indices", self.population_indices),
            ("pair_ids", self.pair_ids),
            ("pair_kinds", self.pair_kinds),
            ("sample_ids", self.sample_ids),
        ):
            if not isinstance(values, tuple) or len(values) != PAIR_BATCH_SIZE:
                raise ValueError(f"{name} must contain exactly two values")
        if len(set(self.population_indices)) != PAIR_BATCH_SIZE:
            raise ValueError("one update cannot repeat a population pair")
        if len(set(self.pair_ids)) != PAIR_BATCH_SIZE:
            raise ValueError("one update cannot repeat a pair_id")
        if len(set(self.sample_ids)) != PAIR_BATCH_SIZE:
            raise ValueError("one update cannot repeat a source")

    def manifest(self) -> dict[str, object]:
        return {
            "update_index": self.update_index,
            "population_indices": list(self.population_indices),
            "pair_ids": list(self.pair_ids),
            "pair_kinds": list(self.pair_kinds),
            "sample_ids": list(self.sample_ids),
        }


@dataclass(frozen=True)
class CCFRHoldoutStrata:
    """Frozen D/H/G-near/G-normalization-tail partition."""

    D: Tensor
    H: Tensor
    G_near: Tensor
    G_norm_tail: Tensor
    changed_count_support: Tensor

    def __post_init__(self) -> None:
        reference = self.D
        values = (
            self.D,
            self.H,
            self.G_near,
            self.G_norm_tail,
            self.changed_count_support,
        )
        if any(
            not isinstance(value, Tensor)
            or value.dtype != torch.bool
            or value.shape != reference.shape
            or value.device != reference.device
            for value in values
        ):
            raise TypeError("all CCFR holdout strata must be aligned bool tensors")
        partition = (self.D, self.H, self.G_near, self.G_norm_tail)
        for left_index, left in enumerate(partition):
            for right in partition[left_index + 1 :]:
                if torch.any(left & right):
                    raise ValueError("D/H/G_near/G_norm_tail must be disjoint")
        if not torch.all(self.G_norm_tail.flatten(1).any(dim=1)):
            raise ValueError("every holdout row requires non-empty G_norm_tail")


def _response_pixel_count(
    *,
    frozen_count: int | None,
    within_group_index: int,
) -> int:
    if frozen_count is not None:
        return frozen_count
    # The two multicount groups are count-transition strata.  Alternating the
    # two receipt-frozen phase patterns prevents transition and phase count
    # from being confounded.
    return 1 if within_group_index % 2 == 0 else 3


def _four_exposure_population_indices(
    rows: list[dict[str, object]],
) -> set[int]:
    selected: set[int] = set()
    by_group: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_group.setdefault(str(row["group_id"]), []).append(row)

    clean_group_ids = [
        group_id
        for group_id, pair_kind, _family, _count, _pixels in _GROUP_CONTRACT
        if pair_kind == "clean_positive"
    ]
    for group_index, group_id in enumerate(clean_group_ids):
        quota = 21 if group_index == 0 else 20
        ordered = sorted(
            by_group[group_id],
            key=lambda row: (
                _sha(
                    f"{DESIGN_SEED}|clean-exposure|{group_id}|"
                    f"{row['within_group_index']}"
                ),
                int(row["population_index"]),
            ),
        )
        selected.update(
            int(row["population_index"]) for row in ordered[:quota]
        )

    null_quotas = {
        "component_null_block": 7,
        "component_null_sparse": 6,
    }
    for group_id, quota in null_quotas.items():
        ordered = sorted(
            by_group[group_id],
            key=lambda row: (
                _sha(
                    f"{DESIGN_SEED}|component-exposure|{group_id}|"
                    f"{row['within_group_index']}"
                ),
                int(row["population_index"]),
            ),
        )
        selected.update(
            int(row["population_index"]) for row in ordered[:quota]
        )
    return selected


def build_ccfr_holdout_pair_specs() -> tuple[CCFRHoldoutPairSpec, ...]:
    """Return the exact new 222-row holdout population."""

    rows: list[dict[str, object]] = []
    population_index = 0
    for (
        group_offset,
        (group_id, pair_kind, family, count, frozen_pixels),
    ) in enumerate(_GROUP_CONTRACT):
        for within_group_index in range(count):
            component_cell = _feature_cell(
                within_group_index=within_group_index,
                group_offset=group_offset,
            )
            ranked_neighbors = _ranked_neighbors(
                group_id=group_id,
                within_group_index=within_group_index,
                cell=component_cell,
            )
            response_cell = (
                ranked_neighbors[0]
                if family == ADJACENT_CELL_FAMILY
                else component_cell
            )
            fixed_count = {
                MULTICOUNT_2TO1_FAMILY: 1,
                MULTICOUNT_3TO2_FAMILY: 2,
            }.get(family, 0)
            identity = (
                f"ccfr-v11-holdout|{DESIGN_SEED}|{group_id}|"
                f"{within_group_index:03d}"
            )
            rows.append(
                {
                    "population_index": population_index,
                    "within_group_index": within_group_index,
                    "group_offset": group_offset,
                    "pair_id": _sha(f"{identity}|pair"),
                    "sample_id": f"ccfr-v11-holdout-source-{_sha(identity)[:24]}",
                    "group_id": group_id,
                    "pair_kind": pair_kind,
                    "geometry_family": family,
                    "response_pixel_count": _response_pixel_count(
                        frozen_count=frozen_pixels,
                        within_group_index=within_group_index,
                    ),
                    "component_cell": component_cell,
                    "response_cell": response_cell,
                    "fixed_occupancy_cells": ranked_neighbors[:fixed_count],
                }
            )
            population_index += 1

    four_exposures = _four_exposure_population_indices(rows)
    specs = tuple(
        CCFRHoldoutPairSpec(
            **row,
            exposure_count=(
                4 if int(row["population_index"]) in four_exposures else 3
            ),
        )
        for row in rows
    )
    _validate_population(specs)
    return specs


def _validate_population(specs: tuple[CCFRHoldoutPairSpec, ...]) -> None:
    if len(specs) != CLEAN_PAIR_COUNT + COMPONENT_NULL_PAIR_COUNT:
        raise AssertionError("CCFR holdout population size differs")
    if tuple(spec.population_index for spec in specs) != tuple(
        range(len(specs))
    ):
        raise AssertionError("CCFR population indices are not canonical")
    if len({spec.pair_id for spec in specs}) != len(specs):
        raise AssertionError("CCFR pair IDs are not unique")
    if len({spec.sample_id for spec in specs}) != len(specs):
        raise AssertionError("CCFR source IDs are not unique")
    if Counter(spec.group_id for spec in specs) != Counter(GROUP_COUNTS):
        raise AssertionError("CCFR eight-group counts differ")
    clean = tuple(
        spec for spec in specs if spec.pair_kind == "clean_positive"
    )
    component_null = tuple(
        spec for spec in specs if spec.pair_kind == "component_null"
    )
    if (
        len(clean) != CLEAN_PAIR_COUNT
        or len(component_null) != COMPONENT_NULL_PAIR_COUNT
    ):
        raise AssertionError("CCFR role counts differ")
    if Counter(spec.exposure_count for spec in clean) != Counter({3: 85, 4: 121}):
        raise AssertionError("CCFR clean exposure histogram differs")
    if Counter(spec.exposure_count for spec in component_null) != Counter(
        {3: 3, 4: 13}
    ):
        raise AssertionError("CCFR component exposure histogram differs")
    if sum(spec.exposure_count for spec in clean) != CLEAN_SLOT_COUNT:
        raise AssertionError("CCFR clean slot count differs")
    if sum(spec.exposure_count for spec in component_null) != (
        COMPONENT_NULL_SLOT_COUNT
    ):
        raise AssertionError("CCFR component-null slot count differs")


def _signal_feature(
    spec: CCFRHoldoutPairSpec,
) -> Tensor:
    """Return the frozen sparse, role-associated feature signal.

    Only the two protocol-listed channels at one designated cell receive
    lattice values.  Every other entry is exactly zero.  This controlled
    association is fixed before training and is not a claim of statistical
    independence between feature and outcome role.
    """

    feature = torch.zeros(
        1,
        FEATURE_CHANNELS,
        FEATURE_HEIGHT,
        FEATURE_WIDTH,
        dtype=torch.float32,
    )
    channels = (
        CLEAN_TARGET_SIGNAL_CHANNELS
        if spec.pair_kind == "clean_positive"
        else COMPONENT_NULL_SIGNAL_CHANNELS
    )
    signal_cell = (
        spec.response_cell
        if spec.pair_kind == "clean_positive"
        else spec.component_cell
    )
    for channel in channels:
        feature[
            0,
            channel,
            signal_cell[0],
            signal_cell[1],
        ] = _lattice_value(
            group_id=spec.group_id,
            within_group_index=spec.within_group_index,
            channel=channel,
            row=signal_cell[0],
            column=signal_cell[1],
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
    spec: CCFRHoldoutPairSpec,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    occupancy_minus = torch.zeros(
        1,
        1,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
        dtype=torch.bool,
    )
    for fixed_cell in spec.fixed_occupancy_cells:
        _set_projected_cell(occupancy_minus, fixed_cell, ((2, 2),))
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
    far_row = OUTPUT_HEIGHT - 3 if spec.component_cell[0] <= 2 else 1
    far_column = OUTPUT_WIDTH - 3 if spec.component_cell[1] <= 2 else 1
    completion_plus[0, 0, far_row, far_column] = True
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


def build_ccfr_holdout_outcome_batch(
    specs: Iterable[CCFRHoldoutPairSpec],
    *,
    device: torch.device | str = "cpu",
) -> OutcomePairBatch:
    """Materialize a non-empty, unique holdout selection on one device."""

    values = tuple(specs)
    if not values:
        raise ValueError("CCFR holdout batch cannot be empty")
    if any(not isinstance(spec, CCFRHoldoutPairSpec) for spec in values):
        raise TypeError("holdout batch requires CCFRHoldoutPairSpec values")
    if len({spec.pair_id for spec in values}) != len(values):
        raise ValueError("holdout batch cannot repeat a pair")
    if len({spec.sample_id for spec in values}) != len(values):
        raise ValueError("holdout batch cannot repeat a source")

    target_device = torch.device(device)
    states = tuple(_pair_state(spec) for spec in values)
    pair_batch = PairBatch(
        feature=torch.cat([_signal_feature(spec) for spec in values], dim=0).to(
            target_device
        ),
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


def build_ccfr_holdout_strata(
    outcome: OutcomePairBatch,
) -> CCFRHoldoutStrata:
    """Split zero-response pixels by changed-count proximity.

    ``G_near`` is the portion of the ordinary global zero stratum inside the
    lifted 3x3 local-count change support.  ``G_norm_tail`` is its complement
    in the valid global zero stratum and therefore audits any far-field
    effect outside the direct release-change support.  The historical field
    name does not attribute that effect uniquely to GroupNorm: the existing
    depthwise trunk can also propagate into this conservative audit region.
    """

    if not isinstance(outcome, OutcomePairBatch):
        raise TypeError("outcome must be an OutcomePairBatch")
    outcome.validate()
    pair_batch = outcome.pair_batch
    feature_size = tuple(int(value) for value in pair_batch.feature.shape[-2:])
    plus = project_occupancy_to_feature_grid(
        pair_batch.occupancy_plus,
        feature_size,
    ).to(dtype=torch.float32)
    minus = project_occupancy_to_feature_grid(
        pair_batch.occupancy_minus,
        feature_size,
    ).to(dtype=torch.float32)
    kernel = torch.ones(1, 1, 3, 3, dtype=plus.dtype, device=plus.device)
    plus_count = F.conv2d(plus, kernel, padding=1)
    minus_count = F.conv2d(minus, kernel, padding=1)
    changed_native = plus_count != minus_count
    changed_count_support = F.interpolate(
        changed_native.to(dtype=torch.float32),
        size=(OUTPUT_HEIGHT, OUTPUT_WIDTH),
        mode="nearest",
    ).to(dtype=torch.bool)

    D = outcome.response_stratum
    H = outcome.local_zero_stratum
    G = outcome.global_zero_stratum
    G_near = G & changed_count_support
    G_norm_tail = G & ~changed_count_support
    valid = pair_batch.image_valid_mask
    if not torch.equal(D | H | G_near | G_norm_tail, valid):
        raise AssertionError("CCFR holdout strata do not partition valid pixels")
    return CCFRHoldoutStrata(
        D=D.contiguous(),
        H=H.contiguous(),
        G_near=G_near.contiguous(),
        G_norm_tail=G_norm_tail.contiguous(),
        changed_count_support=changed_count_support.contiguous(),
    )


def pair_tensor_manifest(spec: CCFRHoldoutPairSpec) -> dict[str, object]:
    """Return exact tensor fingerprints for one population row."""

    outcome = build_ccfr_holdout_outcome_batch((spec,))
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
        "pair_id": spec.pair_id,
        "tensor_fingerprints": {
            name: tensor_content_fingerprint(tensors[name])
            for name in _PAIR_TENSOR_NAMES
        },
    }


def catalog_manifest(
    specs: tuple[CCFRHoldoutPairSpec, ...] | None = None,
) -> list[dict[str, object]]:
    population = build_ccfr_holdout_pair_specs() if specs is None else specs
    _validate_population(population)
    return [
        {
            **spec.manifest(),
            **pair_tensor_manifest(spec),
        }
        for spec in population
    ]


def catalog_fingerprint(
    specs: tuple[CCFRHoldoutPairSpec, ...] | None = None,
) -> str:
    return stable_fingerprint(catalog_manifest(specs))


def _slot_rank(
    *,
    pair_kind: str,
    round_index: int,
    pair_id: str,
) -> str:
    return _sha(
        f"{DESIGN_SEED}|{pair_kind}|{round_index}|{pair_id}"
    )


def _ranked_slots(
    specs: tuple[CCFRHoldoutPairSpec, ...],
) -> list[int]:
    slots: list[tuple[str, str, int, str, int]] = []
    for spec in specs:
        for round_index in range(spec.exposure_count):
            slots.append(
                (
                    _slot_rank(
                        pair_kind=spec.pair_kind,
                        round_index=round_index,
                        pair_id=spec.pair_id,
                    ),
                    spec.pair_kind,
                    round_index,
                    spec.pair_id,
                    spec.population_index,
                )
            )
    slots.sort()
    return [record[-1] for record in slots]


def _pop_earliest_distinct(
    slots: list[int],
    *,
    first_index: int,
    by_index: dict[int, CCFRHoldoutPairSpec],
) -> int:
    first = by_index[first_index]
    for slot_index, candidate_index in enumerate(slots):
        candidate = by_index[candidate_index]
        if (
            candidate.pair_id != first.pair_id
            and candidate.sample_id != first.sample_id
        ):
            return slots.pop(slot_index)
    raise AssertionError("no distinct pair/source remains for final update")


def build_ccfr_holdout_schedule(
    specs: tuple[CCFRHoldoutPairSpec, ...] | None = None,
) -> tuple[CCFRHoldoutUpdate, ...]:
    """Return the new stable-SHA-ranked 400-update schedule."""

    population = build_ccfr_holdout_pair_specs() if specs is None else specs
    _validate_population(population)
    by_index = {spec.population_index: spec for spec in population}
    slots = _ranked_slots(population)
    if len(slots) != TOTAL_PAIR_SLOTS:
        raise AssertionError("CCFR holdout slot population differs")

    raw_updates: list[tuple[int, int]] = []
    while slots:
        first = slots.pop(0)
        second = _pop_earliest_distinct(
            slots,
            first_index=first,
            by_index=by_index,
        )
        raw_updates.append((first, second))
    updates = tuple(
        CCFRHoldoutUpdate(
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
    specs: tuple[CCFRHoldoutPairSpec, ...],
    updates: tuple[CCFRHoldoutUpdate, ...],
) -> None:
    if len(updates) != UPDATE_COUNT:
        raise AssertionError("CCFR holdout update count differs")
    if tuple(update.update_index for update in updates) != tuple(
        range(UPDATE_COUNT)
    ):
        raise AssertionError("CCFR update indices are not canonical")
    observed = Counter(
        population_index
        for update in updates
        for population_index in update.population_indices
    )
    expected = Counter(
        {
            spec.population_index: spec.exposure_count
            for spec in specs
        }
    )
    if observed != expected:
        raise AssertionError("CCFR per-pair exposures differ")
    by_index = {spec.population_index: spec for spec in specs}
    clean_slots = sum(
        by_index[index].pair_kind == "clean_positive"
        for update in updates
        for index in update.population_indices
    )
    if clean_slots != CLEAN_SLOT_COUNT:
        raise AssertionError("CCFR clean schedule slots differ")
    if TOTAL_PAIR_SLOTS - clean_slots != COMPONENT_NULL_SLOT_COUNT:
        raise AssertionError("CCFR component-null schedule slots differ")


def schedule_manifest(
    specs: tuple[CCFRHoldoutPairSpec, ...] | None = None,
) -> list[dict[str, object]]:
    population = build_ccfr_holdout_pair_specs() if specs is None else specs
    return [
        update.manifest()
        for update in build_ccfr_holdout_schedule(population)
    ]


def schedule_fingerprint(
    specs: tuple[CCFRHoldoutPairSpec, ...] | None = None,
) -> str:
    return stable_fingerprint(schedule_manifest(specs))


def _factual_signal_feature(
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
        group_id = "ccfr_holdout_factual_miss"
        channels = FACTUAL_MISS_SIGNAL_CHANNELS
    elif branch == "factual_no_miss":
        group_id = "ccfr_holdout_factual_no_miss"
        channels = FACTUAL_NO_MISS_SIGNAL_CHANNELS
    else:
        raise ValueError("unknown factual branch")
    for channel in channels:
        feature[0, channel, signal_cell[0], signal_cell[1]] = _lattice_value(
            group_id=group_id,
            within_group_index=state_index,
            channel=channel,
            row=signal_cell[0],
            column=signal_cell[1],
        )
    return feature


def _build_ccfr_holdout_factual_populations(
    *,
    device: torch.device | str = "cpu",
) -> dict[str, BranchBatch]:
    target_device = torch.device(device)
    miss_features: list[Tensor] = []
    no_miss_features: list[Tensor] = []
    miss_targets: list[Tensor] = []
    for state_index in range(FACTUAL_POPULATION_SIZE):
        signal_cell = (
            1 + (state_index % 3),
            1 + ((2 * state_index + 1) % 3),
        )
        miss_features.append(
            _factual_signal_feature(
                branch="factual_miss",
                state_index=state_index,
                signal_cell=signal_cell,
            )
        )
        no_miss_features.append(
            _factual_signal_feature(
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

    miss_feature = torch.cat(miss_features, dim=0)
    no_miss_feature = torch.cat(no_miss_features, dim=0)
    miss_target = torch.cat(miss_targets, dim=0)
    occupancy = torch.zeros(
        FACTUAL_POPULATION_SIZE,
        1,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
        dtype=torch.bool,
    )
    valid = torch.ones_like(occupancy)
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
    """Return four contiguous indices with a one-state rotation per update."""

    if (
        isinstance(update_index, bool)
        or not isinstance(update_index, int)
        or not 0 <= update_index < UPDATE_COUNT
    ):
        raise ValueError(
            f"update_index must be an integer in [0,{UPDATE_COUNT})"
        )
    return tuple(
        (update_index + offset) % FACTUAL_POPULATION_SIZE
        for offset in range(FACTUAL_BATCH_SIZE)
    )


def build_ccfr_holdout_factual_population(
    *,
    device: torch.device | str = "cpu",
) -> dict[str, BranchBatch]:
    """Return all 16 new states for both factual branches."""

    return _build_ccfr_holdout_factual_populations(device=device)


def build_ccfr_holdout_factual_batches(
    *,
    update_index: int,
    device: torch.device | str = "cpu",
) -> dict[str, BranchBatch]:
    """Return the exact 4/4 factual slice for one update."""

    populations = _build_ccfr_holdout_factual_populations(device=device)
    target_device = torch.device(device)
    indices = torch.tensor(
        factual_indices_for_update(update_index),
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


def factual_population_manifest() -> list[dict[str, object]]:
    populations = build_ccfr_holdout_factual_population()
    rows: list[dict[str, object]] = []
    for branch in ("factual_miss", "factual_no_miss"):
        batch = populations[branch]
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


def factual_population_fingerprint() -> str:
    return stable_fingerprint(factual_population_manifest())


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


def holdout_manifest() -> dict[str, object]:
    """Return the tensor- and schedule-bound top-level holdout manifest."""

    return {
        "schema_version": "cure-lite-ccfr-v11-holdout-inputs-v1",
        "design_seed": DESIGN_SEED,
        "input_contract": {
            "feature_shape": [
                1,
                FEATURE_CHANNELS,
                FEATURE_HEIGHT,
                FEATURE_WIDTH,
            ],
            "output_shape": [1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH],
            "feature_stride": FEATURE_STRIDE,
        },
        "counts": {
            "clean_pairs": CLEAN_PAIR_COUNT,
            "component_null_pairs": COMPONENT_NULL_PAIR_COUNT,
            "pair_slots": TOTAL_PAIR_SLOTS,
            "updates": UPDATE_COUNT,
            "factual_miss_states": FACTUAL_POPULATION_SIZE,
            "factual_no_miss_states": FACTUAL_POPULATION_SIZE,
        },
        "group_counts": dict(GROUP_COUNTS),
        "catalog_fingerprint": catalog_fingerprint(),
        "schedule_fingerprint": schedule_fingerprint(),
        "factual_population_fingerprint": factual_population_fingerprint(),
        "factual_schedule_fingerprint": factual_schedule_fingerprint(),
    }


def holdout_fingerprint() -> str:
    return stable_fingerprint(holdout_manifest())


__all__ = [
    "ADJACENT_CELL_FAMILY",
    "CCFRHoldoutPairSpec",
    "CCFRHoldoutStrata",
    "CCFRHoldoutUpdate",
    "CLEAN_PAIR_COUNT",
    "CLEAN_SLOT_COUNT",
    "COMPONENT_NULL_BLOCK_FAMILY",
    "COMPONENT_NULL_PAIR_COUNT",
    "COMPONENT_NULL_SLOT_COUNT",
    "COMPONENT_NULL_SPARSE_FAMILY",
    "DESIGN_SEED",
    "FACTUAL_BATCH_SIZE",
    "FACTUAL_EXPOSURES_PER_STATE",
    "FACTUAL_POPULATION_SIZE",
    "FEATURE_CHANNELS",
    "FEATURE_HEIGHT",
    "FEATURE_STRIDE",
    "FEATURE_WIDTH",
    "GROUP_COUNTS",
    "MULTICOUNT_2TO1_FAMILY",
    "MULTICOUNT_3TO2_FAMILY",
    "ONE_PIXEL_PHASE_PATTERN",
    "OUTPUT_HEIGHT",
    "OUTPUT_WIDTH",
    "PAIR_BATCH_SIZE",
    "SAME_CELL_FAMILY",
    "THREE_PIXEL_PHASE_PATTERN",
    "TOTAL_PAIR_SLOTS",
    "UPDATE_COUNT",
    "build_ccfr_holdout_factual_batches",
    "build_ccfr_holdout_factual_population",
    "build_ccfr_holdout_outcome_batch",
    "build_ccfr_holdout_pair_specs",
    "build_ccfr_holdout_schedule",
    "build_ccfr_holdout_strata",
    "catalog_fingerprint",
    "catalog_manifest",
    "factual_indices_for_update",
    "factual_population_fingerprint",
    "factual_population_manifest",
    "factual_schedule_fingerprint",
    "factual_schedule_manifest",
    "holdout_fingerprint",
    "holdout_manifest",
    "pair_tensor_manifest",
    "schedule_fingerprint",
    "schedule_manifest",
]
