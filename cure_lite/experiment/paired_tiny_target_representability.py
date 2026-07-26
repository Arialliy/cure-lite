"""Static stride-four representability audit for 1--3 pixel targets.

This module is deliberately outside the model, loss, and training packages.
It treats the frozen decoder's final 64x64-to-256x256 bilinear interpolation
as a spatial-capacity upper-bound oracle.  No decoder weights are loaded or
optimized.

For one binary target, the oracle asks for the minimum number of background
pixels that must be non-negative while every target pixel remains
non-negative, matching the frozen inclusive residual-threshold rule.  The
problem uses a numerically certified mixed-integer linear construction:

1. minimize a threshold-tie relaxation of the background count;
2. construct an attainable positive-margin upper count and search downward;
3. close a count gap only when the scaled primal objective, its dual bound,
   and the reconstructed margin are all exactly zero in the frozen float64
   solver result.  Every non-zero sub-threshold margin is inconclusive.

All bilinear coefficients are non-negative.  Consequently, a low-resolution
node with zero coefficient on every target row can be fixed to the minimum
normalized logit (-1): doing so leaves the target unchanged and cannot
increase any background response.  Stage 1 is therefore a monotone-localized
lower bound on the free-grid count.  The descending Stage-2 search can close
the lower/upper count gap only under the explicitly frozen numerical contract
above.  The resulting count is a tolerance-certified irreducible count for
this stress gate, not a symbolic real-arithmetic theorem.  The selected
Stage-2 margin is only a localized feasibility certificate; it is not claimed
to be the free-grid maximum.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..config import MatchConfig
from ..metrics import evaluate_binary_prediction


LOW_GRID_SIZE = 64
OUTPUT_GRID_SIZE = 256
AXIS_WEIGHT_DENOMINATOR = 8
BILINEAR_WEIGHT_DENOMINATOR = 64
TARGET_AREAS = (1, 2, 3)
EXPECTED_SHAPE_COUNTS = {1: 1, 2: 4, 3: 20}
EXPECTED_TOTAL_SHAPES = 25
EXPECTED_CONCRETE_PLACEMENTS = 1_622_566
NORMALIZED_LOGIT_MIN = -1.0
NORMALIZED_LOGIT_MAX = 1.0
CERTIFYING_MARGIN_MIN = 1e-6
GAMMA_MAX = 1.0
MARGIN_OBJECTIVE_SCALE = 1_000_000_000.0
BIG_M = 2.0
VERIFY_TOLERANCE = 1e-8
PIXEL_FA_MAX = 1e-4
RAW_BACKGROUND_FA_MAX = 1e-4
FP_COMPONENTS_PER_MP_MAX = 100.0
RETENTION_REQUIRED = 1.0
MAX_FALSE_ADDITION_PIXELS = floor(
    RAW_BACKGROUND_FA_MAX * OUTPUT_GRID_SIZE * OUTPUT_GRID_SIZE
)
DEFAULT_SOLVER_OPTIONS: Mapping[str, object] = {
    "presolve": True,
    "mip_rel_gap": 0.0,
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
    "mip_feasibility_tolerance": 1e-9,
    "threads": 1,
    "parallel": False,
    "random_seed": 0,
    "output_flag": False,
}

TINY_TARGET_CASE_SCHEMA = "cure-lite-tiny-target-case-v2"
TINY_TARGET_COMPACT_PROBLEM_SCHEMA = (
    "cure-lite-tiny-target-compact-problem-v1"
)
TINY_TARGET_CATALOG_SCHEMA = "cure-lite-tiny-target-case-catalog-v1"
TINY_TARGET_CERTIFICATE_SCHEMA = (
    "cure-lite-tiny-target-representability-certificate-v2"
)
TINY_TARGET_DECISION_SCHEMA = (
    "cure-lite-tiny-target-representability-decision-v1"
)


Pixel = tuple[int, int]
ShapePixels = tuple[Pixel, ...]


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 string")
    return value


def _is_cc8_connected(pixels: Iterable[Pixel]) -> bool:
    remaining = set(pixels)
    if not remaining:
        return False
    visited = {next(iter(remaining))}
    frontier = list(visited)
    while frontier:
        row, column = frontier.pop()
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                if delta_row == 0 and delta_column == 0:
                    continue
                candidate = (row + delta_row, column + delta_column)
                if candidate in remaining and candidate not in visited:
                    visited.add(candidate)
                    frontier.append(candidate)
    return visited == remaining


def _normalize_shape(pixels: Iterable[Pixel]) -> ShapePixels:
    values = tuple(pixels)
    if not values:
        raise ValueError("shape must be non-empty")
    if any(
        isinstance(row, bool)
        or isinstance(column, bool)
        or not isinstance(row, int)
        or not isinstance(column, int)
        for row, column in values
    ):
        raise TypeError("shape coordinates must be integer pairs")
    minimum_row = min(row for row, _ in values)
    minimum_column = min(column for _, column in values)
    normalized = tuple(
        sorted(
            {
                (row - minimum_row, column - minimum_column)
                for row, column in values
            }
        )
    )
    if len(normalized) != len(values):
        raise ValueError("shape coordinates must be unique")
    return normalized


@dataclass(frozen=True)
class TinyTargetShape:
    """One oriented, translation-normalized CC8 target shape."""

    pixels: ShapePixels
    shape_id: str

    def __post_init__(self) -> None:
        normalized = _normalize_shape(self.pixels)
        if normalized != self.pixels:
            raise ValueError("shape pixels must already be normalized and sorted")
        if len(normalized) not in TARGET_AREAS or not _is_cc8_connected(normalized):
            raise ValueError("shape must be a connected 1--3 pixel CC8 component")
        expected = stable_fingerprint(
            {
                "area": len(normalized),
                "pixels": [list(pixel) for pixel in normalized],
            }
        )
        if self.shape_id != expected:
            raise ValueError("shape_id differs from the normalized shape")

    @property
    def area(self) -> int:
        return len(self.pixels)

    @property
    def height(self) -> int:
        return max(row for row, _ in self.pixels) + 1

    @property
    def width(self) -> int:
        return max(column for _, column in self.pixels) + 1

    def payload(self) -> dict[str, object]:
        return {
            "shape_id": self.shape_id,
            "area": self.area,
            "height": self.height,
            "width": self.width,
            "pixels": [list(pixel) for pixel in self.pixels],
        }


def enumerate_tiny_target_shapes() -> tuple[TinyTargetShape, ...]:
    """Enumerate every oriented CC8 shape of area 1, 2, or 3.

    Starting from the singleton, every unoccupied CC8 neighbour is added,
    followed by translation normalization and exact deduplication.  This is
    the construction frozen by the protocol.  Rotations and reflections
    remain distinct; only translation is factored out.
    """

    by_area: dict[int, set[ShapePixels]] = {
        1: {((0, 0),)},
        2: set(),
        3: set(),
    }
    for area in (2, 3):
        for prior in by_area[area - 1]:
            prior_set = set(prior)
            for row, column in prior:
                for delta_row in (-1, 0, 1):
                    for delta_column in (-1, 0, 1):
                        if delta_row == 0 and delta_column == 0:
                            continue
                        candidate = (
                            row + delta_row,
                            column + delta_column,
                        )
                        if candidate in prior_set:
                            continue
                        grown = _normalize_shape((*prior, candidate))
                        if len(grown) == area and _is_cc8_connected(grown):
                            by_area[area].add(grown)
    for area in TARGET_AREAS:
        if len(by_area[area]) != EXPECTED_SHAPE_COUNTS[area]:
            raise RuntimeError(
                f"area-{area} shape count changed: {len(by_area[area])}"
            )
    shapes = tuple(
        TinyTargetShape(
            pixels=pixels,
            shape_id=stable_fingerprint(
                {
                    "area": area,
                    "pixels": [list(pixel) for pixel in pixels],
                }
            ),
        )
        for area in TARGET_AREAS
        for pixels in sorted(by_area[area])
    )
    if len(shapes) != EXPECTED_TOTAL_SHAPES:
        raise RuntimeError("tiny-target shape catalog count changed")
    return shapes


def axis_bilinear_weights(
    output_index: int,
    *,
    low_size: int = LOW_GRID_SIZE,
    output_size: int = OUTPUT_GRID_SIZE,
) -> tuple[tuple[int, int], ...]:
    """Return exact coalesced 1-D weights as integer numerators over eight."""

    output_index = _require_int(
        output_index, name="output_index", minimum=0
    )
    if output_index >= output_size:
        raise ValueError("output_index lies outside the output grid")
    if (low_size, output_size) != (LOW_GRID_SIZE, OUTPUT_GRID_SIZE):
        raise ValueError("the frozen audit requires a 64-to-256 axis")

    # align_corners=False:
    #   source = (output + 0.5) / 4 - 0.5 = (2*output - 3) / 8.
    numerator = 2 * output_index - 3
    lower = numerator // AXIS_WEIGHT_DENOMINATOR
    remainder = numerator - lower * AXIS_WEIGHT_DENOMINATOR
    coalesced: dict[int, int] = {}
    for source_index, weight in (
        (lower, AXIS_WEIGHT_DENOMINATOR - remainder),
        (lower + 1, remainder),
    ):
        if weight == 0:
            continue
        clamped = min(low_size - 1, max(0, source_index))
        coalesced[clamped] = coalesced.get(clamped, 0) + weight
    result = tuple(sorted(coalesced.items()))
    if sum(weight for _, weight in result) != AXIS_WEIGHT_DENOMINATOR:
        raise RuntimeError("bilinear axis weights do not preserve constants")
    return result


_AXIS_ROWS = tuple(
    axis_bilinear_weights(index) for index in range(OUTPUT_GRID_SIZE)
)
_AXIS_ROW_MAPS = tuple(dict(row) for row in _AXIS_ROWS)
_AXIS_SUPPORTS = tuple(
    tuple(
        output_index
        for output_index, row in enumerate(_AXIS_ROWS)
        if any(source_index == low_index for source_index, _ in row)
    )
    for low_index in range(LOW_GRID_SIZE)
)


def _axis_equivalence_signature(
    origin: int,
    offsets: Sequence[int],
) -> tuple[object, ...]:
    target_indices = tuple(sorted({origin + int(offset) for offset in offsets}))
    active = tuple(
        sorted(
            {
                source_index
                for target_index in target_indices
                for source_index, _ in _AXIS_ROWS[target_index]
            }
        )
    )
    affected = tuple(
        sorted(
            {
                output_index
                for source_index in active
                for output_index in _AXIS_SUPPORTS[source_index]
            }
        )
    )
    active_origin = min(active)
    output_origin = min(affected)
    matrix = tuple(
        tuple(_AXIS_ROW_MAPS[index].get(source, 0) for source in active)
        for index in affected
    )
    return (
        tuple(source - active_origin for source in active),
        tuple(index - output_origin for index in affected),
        matrix,
        tuple(index - output_origin for index in target_indices),
    )


@dataclass(frozen=True)
class AxisEquivalenceClass:
    """Exact translation-equivalence class for one shape axis."""

    representative_origin: int
    origins: tuple[int, ...]
    signature_fingerprint: str

    def __post_init__(self) -> None:
        _require_int(
            self.representative_origin,
            name="representative_origin",
            minimum=0,
        )
        if not self.origins or tuple(sorted(set(self.origins))) != self.origins:
            raise ValueError("axis origins must be non-empty, sorted, and unique")
        for origin in self.origins:
            _require_int(origin, name="axis origin", minimum=0)
        if self.representative_origin != self.origins[0]:
            raise ValueError("axis representative must be the smallest origin")
        _require_sha256(
            self.signature_fingerprint,
            name="axis signature_fingerprint",
        )


def build_axis_equivalence_classes(
    offsets: Sequence[int],
    *,
    output_size: int = OUTPUT_GRID_SIZE,
) -> tuple[AxisEquivalenceClass, ...]:
    """Partition every valid origin by exact translated operator identity."""

    normalized_offsets = tuple(sorted({int(value) for value in offsets}))
    if not normalized_offsets or normalized_offsets[0] != 0:
        raise ValueError("axis offsets must be normalized and include zero")
    extent = normalized_offsets[-1] + 1
    groups: dict[tuple[object, ...], list[int]] = {}
    for origin in range(output_size - extent + 1):
        signature = _axis_equivalence_signature(origin, normalized_offsets)
        groups.setdefault(signature, []).append(origin)
    classes = tuple(
        AxisEquivalenceClass(
            representative_origin=origins[0],
            origins=tuple(origins),
            signature_fingerprint=stable_fingerprint(signature),
        )
        for signature, origins in sorted(
            groups.items(), key=lambda item: item[1][0]
        )
    )
    covered = tuple(
        origin for equivalence in classes for origin in equivalence.origins
    )
    if tuple(sorted(covered)) != tuple(range(output_size - extent + 1)):
        raise RuntimeError("axis equivalence classes do not exactly cover origins")
    return classes


def _compact_problem_fingerprint(
    shape: TinyTargetShape,
    row_axis_signature_fingerprint: str,
    column_axis_signature_fingerprint: str,
) -> str:
    """Bind an exact case without materializing its dense MILP matrix.

    The two axis signatures contain every affected coordinate and every exact
    integer interpolation coefficient, including clipped boundary rows.  The
    oriented shape supplies the joint 2-D target indicator.  Their product,
    together with the frozen monotone-localization constants below, therefore
    determines the dense problem exactly.
    """

    return stable_fingerprint(
        {
            "schema_version": TINY_TARGET_COMPACT_PROBLEM_SCHEMA,
            "shape_id": shape.shape_id,
            "shape_pixels": [list(pixel) for pixel in shape.pixels],
            "row_axis_signature_fingerprint": (
                row_axis_signature_fingerprint
            ),
            "column_axis_signature_fingerprint": (
                column_axis_signature_fingerprint
            ),
            "low_grid_size": LOW_GRID_SIZE,
            "output_grid_size": OUTPUT_GRID_SIZE,
            "mode": "bilinear",
            "align_corners": False,
            "axis_weight_denominator": AXIS_WEIGHT_DENOMINATOR,
            "bilinear_weight_denominator": BILINEAR_WEIGHT_DENOMINATOR,
            "active_node_rule": (
                "union-of-nonzero-low-nodes-on-target-output-rows"
            ),
            "inactive_low_resolution_value": NORMALIZED_LOGIT_MIN,
            "active_value_bounds": [
                NORMALIZED_LOGIT_MIN,
                NORMALIZED_LOGIT_MAX,
            ],
        }
    )


@dataclass(frozen=True)
class TinyTargetCase:
    """One exact operator-equivalence class of concrete placements."""

    case_id: str
    shape: TinyTargetShape
    representative_origin: Pixel
    row_origins: tuple[int, ...]
    column_origins: tuple[int, ...]
    row_axis_signature_fingerprint: str
    column_axis_signature_fingerprint: str
    problem_fingerprint: str
    schema_version: str = TINY_TARGET_CASE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TINY_TARGET_CASE_SCHEMA:
            raise ValueError("unsupported tiny-target case schema")
        if not isinstance(self.shape, TinyTargetShape):
            raise TypeError("shape must be TinyTargetShape")
        if (
            tuple(sorted(set(self.row_origins))) != self.row_origins
            or tuple(sorted(set(self.column_origins))) != self.column_origins
            or not self.row_origins
            or not self.column_origins
        ):
            raise ValueError("case origin sets must be sorted, unique, and non-empty")
        if self.representative_origin != (
            self.row_origins[0],
            self.column_origins[0],
        ):
            raise ValueError("case representative must be the smallest origin pair")
        maximum_row_origin = OUTPUT_GRID_SIZE - self.shape.height
        maximum_column_origin = OUTPUT_GRID_SIZE - self.shape.width
        if any(
            origin < 0 or origin > maximum_row_origin
            for origin in self.row_origins
        ) or any(
            origin < 0 or origin > maximum_column_origin
            for origin in self.column_origins
        ):
            raise ValueError("case origin lies outside the valid shape domain")
        for name, fingerprint in (
            (
                "row_axis_signature_fingerprint",
                self.row_axis_signature_fingerprint,
            ),
            (
                "column_axis_signature_fingerprint",
                self.column_axis_signature_fingerprint,
            ),
            ("problem_fingerprint", self.problem_fingerprint),
            ("case_id", self.case_id),
        ):
            _require_sha256(fingerprint, name=name)
        expected = stable_fingerprint(
            {
                "schema_version": self.schema_version,
                "shape_id": self.shape.shape_id,
                "representative_origin": list(self.representative_origin),
                "row_origins": list(self.row_origins),
                "column_origins": list(self.column_origins),
                "row_axis_signature_fingerprint": (
                    self.row_axis_signature_fingerprint
                ),
                "column_axis_signature_fingerprint": (
                    self.column_axis_signature_fingerprint
                ),
                "problem_fingerprint": self.problem_fingerprint,
            }
        )
        if self.case_id != expected:
            raise ValueError("case_id differs from the exact case identity")
        expected_problem = _compact_problem_fingerprint(
            self.shape,
            self.row_axis_signature_fingerprint,
            self.column_axis_signature_fingerprint,
        )
        if self.problem_fingerprint != expected_problem:
            raise ValueError("case compact problem fingerprint is inconsistent")

    @property
    def multiplicity(self) -> int:
        return len(self.row_origins) * len(self.column_origins)

    @property
    def target_pixels(self) -> tuple[Pixel, ...]:
        origin_row, origin_column = self.representative_origin
        return tuple(
            (origin_row + row, origin_column + column)
            for row, column in self.shape.pixels
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "shape_id": self.shape.shape_id,
            "representative_origin": list(self.representative_origin),
            "row_origins": list(self.row_origins),
            "column_origins": list(self.column_origins),
            "multiplicity": self.multiplicity,
            "row_axis_signature_fingerprint": (
                self.row_axis_signature_fingerprint
            ),
            "column_axis_signature_fingerprint": (
                self.column_axis_signature_fingerprint
            ),
            "problem_fingerprint": self.problem_fingerprint,
        }


@dataclass(frozen=True)
class RepresentabilityProblem:
    """Dense, small exact-localized MILP input for one case."""

    active_nodes: tuple[Pixel, ...]
    affected_pixels: tuple[Pixel, ...]
    target_pixels: tuple[Pixel, ...]
    coefficient_numerators: tuple[tuple[int, ...], ...]
    constant_numerators: tuple[int, ...]
    target_indicator: tuple[bool, ...]
    problem_fingerprint: str

    def __post_init__(self) -> None:
        if not self.active_nodes or not self.affected_pixels or not self.target_pixels:
            raise ValueError("representability problem sets must be non-empty")
        if len(self.coefficient_numerators) != len(self.affected_pixels):
            raise ValueError("coefficient row count changed")
        if len(self.constant_numerators) != len(self.affected_pixels):
            raise ValueError("constant row count changed")
        if len(self.target_indicator) != len(self.affected_pixels):
            raise ValueError("target-indicator row count changed")
        if any(len(row) != len(self.active_nodes) for row in self.coefficient_numerators):
            raise ValueError("coefficient column count changed")
        if sum(self.target_indicator) != len(self.target_pixels):
            raise ValueError("target indicator differs from target pixels")
        if self.problem_fingerprint != stable_fingerprint(self.payload(unsigned=True)):
            raise ValueError("problem fingerprint is inconsistent")

    def payload(self, *, unsigned: bool = False) -> dict[str, object]:
        active_origin = (
            min(row for row, _ in self.active_nodes),
            min(column for _, column in self.active_nodes),
        )
        output_origin = (
            min(row for row, _ in self.affected_pixels),
            min(column for _, column in self.affected_pixels),
        )
        payload: dict[str, object] = {
            "active_nodes_relative": [
                [row - active_origin[0], column - active_origin[1]]
                for row, column in self.active_nodes
            ],
            "affected_pixels_relative": [
                [row - output_origin[0], column - output_origin[1]]
                for row, column in self.affected_pixels
            ],
            "target_pixels_relative": [
                [row - output_origin[0], column - output_origin[1]]
                for row, column in self.target_pixels
            ],
            "coefficient_numerators": [
                list(row) for row in self.coefficient_numerators
            ],
            "constant_numerators": list(self.constant_numerators),
            "target_indicator": list(self.target_indicator),
            "common_denominator": BILINEAR_WEIGHT_DENOMINATOR,
            "inactive_low_resolution_value": NORMALIZED_LOGIT_MIN,
        }
        if not unsigned:
            payload["problem_fingerprint"] = self.problem_fingerprint
        return payload


def build_representability_problem(
    shape: TinyTargetShape,
    origin: Pixel,
) -> RepresentabilityProblem:
    """Build the exact monotone-localized free-grid problem."""

    origin_row = _require_int(origin[0], name="origin row", minimum=0)
    origin_column = _require_int(origin[1], name="origin column", minimum=0)
    if (
        origin_row + shape.height > OUTPUT_GRID_SIZE
        or origin_column + shape.width > OUTPUT_GRID_SIZE
    ):
        raise ValueError("shape origin lies outside the output grid")
    target_pixels = tuple(
        (origin_row + row, origin_column + column)
        for row, column in shape.pixels
    )
    target_set = set(target_pixels)
    active_nodes = tuple(
        sorted(
            {
                (low_row, low_column)
                for row, column in target_pixels
                for low_row, _ in _AXIS_ROWS[row]
                for low_column, _ in _AXIS_ROWS[column]
            }
        )
    )
    affected_rows = tuple(
        sorted(
            {
                output_row
                for low_row, _ in active_nodes
                for output_row in _AXIS_SUPPORTS[low_row]
            }
        )
    )
    affected_columns = tuple(
        sorted(
            {
                output_column
                for _, low_column in active_nodes
                for output_column in _AXIS_SUPPORTS[low_column]
            }
        )
    )
    affected_pixels: list[Pixel] = []
    coefficients: list[tuple[int, ...]] = []
    constants: list[int] = []
    indicators: list[bool] = []
    for row in affected_rows:
        row_weights = _AXIS_ROW_MAPS[row]
        for column in affected_columns:
            column_weights = _AXIS_ROW_MAPS[column]
            coefficient_row = tuple(
                row_weights.get(low_row, 0)
                * column_weights.get(low_column, 0)
                for low_row, low_column in active_nodes
            )
            active_weight = sum(coefficient_row)
            if active_weight == 0:
                continue
            if active_weight > BILINEAR_WEIGHT_DENOMINATOR:
                raise RuntimeError("affected output row has invalid active weight")
            affected_pixels.append((row, column))
            coefficients.append(coefficient_row)
            # u = sum_active w*x + sum_inactive w*(-1).
            constants.append(
                -(BILINEAR_WEIGHT_DENOMINATOR - active_weight)
            )
            indicators.append((row, column) in target_set)
    affected_pixels_tuple = tuple(affected_pixels)
    active_origin_row = min(row for row, _ in active_nodes)
    active_origin_column = min(column for _, column in active_nodes)
    output_origin_row = min(row for row, _ in affected_pixels_tuple)
    output_origin_column = min(column for _, column in affected_pixels_tuple)
    unsigned = {
        "active_nodes_relative": [
            [
                row - active_origin_row,
                column - active_origin_column,
            ]
            for row, column in active_nodes
        ],
        "affected_pixels_relative": [
            [
                row - output_origin_row,
                column - output_origin_column,
            ]
            for row, column in affected_pixels_tuple
        ],
        "target_pixels_relative": [
            [
                row - output_origin_row,
                column - output_origin_column,
            ]
            for row, column in target_pixels
        ],
        "coefficient_numerators": [list(value) for value in coefficients],
        "constant_numerators": constants,
        "target_indicator": indicators,
        "common_denominator": BILINEAR_WEIGHT_DENOMINATOR,
        "inactive_low_resolution_value": NORMALIZED_LOGIT_MIN,
    }
    return RepresentabilityProblem(
        active_nodes=active_nodes,
        affected_pixels=affected_pixels_tuple,
        target_pixels=target_pixels,
        coefficient_numerators=tuple(coefficients),
        constant_numerators=tuple(constants),
        target_indicator=tuple(indicators),
        problem_fingerprint=stable_fingerprint(unsigned),
    )


@dataclass(frozen=True)
class TinyTargetCaseCatalog:
    shapes: tuple[TinyTargetShape, ...]
    cases: tuple[TinyTargetCase, ...]
    concrete_placement_count: int
    catalog_fingerprint: str
    schema_version: str = TINY_TARGET_CATALOG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TINY_TARGET_CATALOG_SCHEMA:
            raise ValueError("unsupported tiny-target catalog schema")
        expected_shapes = enumerate_tiny_target_shapes()
        if self.shapes != expected_shapes:
            raise ValueError("tiny-target catalog shape identities changed")
        canonical_cases = tuple(
            sorted(
                self.cases,
                key=lambda item: (
                    item.shape.area,
                    item.shape.pixels,
                    item.case_id,
                ),
            )
        )
        if self.cases != canonical_cases:
            raise ValueError("tiny-target cases are not canonically ordered")
        if not self.cases or len({case.case_id for case in self.cases}) != len(
            self.cases
        ):
            raise ValueError("tiny-target cases must be non-empty and unique")

        expected_case_keys: set[tuple[object, ...]] = set()
        expected_concrete_count = 0
        for shape in expected_shapes:
            row_offsets = tuple(row for row, _ in shape.pixels)
            column_offsets = tuple(column for _, column in shape.pixels)
            row_classes = build_axis_equivalence_classes(row_offsets)
            column_classes = build_axis_equivalence_classes(column_offsets)
            for equivalence, offsets, axis_name in (
                (row_classes, row_offsets, "row"),
                (column_classes, column_offsets, "column"),
            ):
                for axis_class in equivalence:
                    for origin in axis_class.origins:
                        actual_fingerprint = stable_fingerprint(
                            _axis_equivalence_signature(origin, offsets)
                        )
                        if (
                            actual_fingerprint
                            != axis_class.signature_fingerprint
                        ):
                            raise ValueError(
                                f"{axis_name} equivalence member changed"
                            )
            expected_concrete_count += (
                OUTPUT_GRID_SIZE - shape.height + 1
            ) * (OUTPUT_GRID_SIZE - shape.width + 1)
            for row_class in row_classes:
                for column_class in column_classes:
                    compact = _compact_problem_fingerprint(
                        shape,
                        row_class.signature_fingerprint,
                        column_class.signature_fingerprint,
                    )
                    expected_case_keys.add(
                        (
                            shape.shape_id,
                            row_class.origins,
                            column_class.origins,
                            row_class.signature_fingerprint,
                            column_class.signature_fingerprint,
                            compact,
                        )
                    )
        actual_case_keys = {
            (
                case.shape.shape_id,
                case.row_origins,
                case.column_origins,
                case.row_axis_signature_fingerprint,
                case.column_axis_signature_fingerprint,
                case.problem_fingerprint,
            )
            for case in self.cases
        }
        if (
            actual_case_keys != expected_case_keys
            or len(self.cases) != len(expected_case_keys)
        ):
            raise ValueError(
                "tiny-target cases do not equal the derived exact partition"
            )
        if sum(case.multiplicity for case in self.cases) != (
            self.concrete_placement_count
        ):
            raise ValueError(
                "tiny-target case multiplicities do not cover placements"
            )
        if (
            self.concrete_placement_count != expected_concrete_count
            or expected_concrete_count != EXPECTED_CONCRETE_PLACEMENTS
        ):
            raise ValueError("concrete tiny-target placement count changed")
        _require_sha256(
            self.catalog_fingerprint,
            name="catalog_fingerprint",
        )
        if self.catalog_fingerprint != stable_fingerprint(
            self.payload(unsigned=True)
        ):
            raise ValueError("tiny-target catalog fingerprint is inconsistent")

    def payload(self, *, unsigned: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "shapes": [shape.payload() for shape in self.shapes],
            "cases": [case.payload() for case in self.cases],
            "shape_count": len(self.shapes),
            "equivalence_class_count": len(self.cases),
            "concrete_placement_count": self.concrete_placement_count,
        }
        if not unsigned:
            payload["catalog_fingerprint"] = self.catalog_fingerprint
        return payload


def build_tiny_target_case_catalog() -> TinyTargetCaseCatalog:
    """Build an exact, exhaustive, translation-deduplicated toy catalog."""

    shapes = enumerate_tiny_target_shapes()
    cases: list[TinyTargetCase] = []
    concrete_count = 0
    for shape in shapes:
        row_classes = build_axis_equivalence_classes(
            [row for row, _ in shape.pixels]
        )
        column_classes = build_axis_equivalence_classes(
            [column for _, column in shape.pixels]
        )
        concrete_count += (
            OUTPUT_GRID_SIZE - shape.height + 1
        ) * (
            OUTPUT_GRID_SIZE - shape.width + 1
        )
        for row_class in row_classes:
            for column_class in column_classes:
                origin = (
                    row_class.representative_origin,
                    column_class.representative_origin,
                )
                problem_fingerprint = _compact_problem_fingerprint(
                    shape,
                    row_class.signature_fingerprint,
                    column_class.signature_fingerprint,
                )
                unsigned_case = {
                    "schema_version": TINY_TARGET_CASE_SCHEMA,
                    "shape_id": shape.shape_id,
                    "representative_origin": list(origin),
                    "row_origins": list(row_class.origins),
                    "column_origins": list(column_class.origins),
                    "row_axis_signature_fingerprint": (
                        row_class.signature_fingerprint
                    ),
                    "column_axis_signature_fingerprint": (
                        column_class.signature_fingerprint
                    ),
                    "problem_fingerprint": problem_fingerprint,
                }
                cases.append(
                    TinyTargetCase(
                        case_id=stable_fingerprint(unsigned_case),
                        shape=shape,
                        representative_origin=origin,
                        row_origins=row_class.origins,
                        column_origins=column_class.origins,
                        row_axis_signature_fingerprint=(
                            row_class.signature_fingerprint
                        ),
                        column_axis_signature_fingerprint=(
                            column_class.signature_fingerprint
                        ),
                        problem_fingerprint=problem_fingerprint,
                    )
                )
    cases_tuple = tuple(
        sorted(cases, key=lambda item: (item.shape.area, item.shape.pixels, item.case_id))
    )
    unsigned_catalog = {
        "schema_version": TINY_TARGET_CATALOG_SCHEMA,
        "shapes": [shape.payload() for shape in shapes],
        "cases": [case.payload() for case in cases_tuple],
        "shape_count": len(shapes),
        "equivalence_class_count": len(cases_tuple),
        "concrete_placement_count": concrete_count,
    }
    return TinyTargetCaseCatalog(
        shapes=shapes,
        cases=cases_tuple,
        concrete_placement_count=concrete_count,
        catalog_fingerprint=stable_fingerprint(unsigned_catalog),
    )


def reconstruct_output_logits(
    problem: RepresentabilityProblem,
    active_values: Sequence[float],
) -> np.ndarray:
    """Reconstruct the full normalized 256x256 logit field."""

    values = np.asarray(active_values, dtype=np.float64)
    if values.shape != (len(problem.active_nodes),) or not np.isfinite(
        values
    ).all():
        raise ValueError("active_values have the wrong shape or contain non-finite values")
    if np.any(values < NORMALIZED_LOGIT_MIN - VERIFY_TOLERANCE) or np.any(
        values > NORMALIZED_LOGIT_MAX + VERIFY_TOLERANCE
    ):
        raise ValueError("active_values lie outside the certified bound tolerance")
    values = np.clip(
        values,
        NORMALIZED_LOGIT_MIN,
        NORMALIZED_LOGIT_MAX,
    )
    output = np.full(
        (OUTPUT_GRID_SIZE, OUTPUT_GRID_SIZE),
        NORMALIZED_LOGIT_MIN,
        dtype=np.float64,
    )
    coefficients = (
        np.asarray(problem.coefficient_numerators, dtype=np.float64)
        / BILINEAR_WEIGHT_DENOMINATOR
    )
    constants = (
        np.asarray(problem.constant_numerators, dtype=np.float64)
        / BILINEAR_WEIGHT_DENOMINATOR
    )
    affected = constants + coefficients @ values
    for (row, column), value in zip(
        problem.affected_pixels, affected, strict=True
    ):
        output[row, column] = value
    return output


def torch_bilinear_output_from_active_values(
    problem: RepresentabilityProblem,
    active_values: Sequence[float],
) -> Tensor:
    """Reference reconstruction through the exact frozen PyTorch operator."""

    values = np.asarray(active_values, dtype=np.float64)
    low = torch.full(
        (1, 1, LOW_GRID_SIZE, LOW_GRID_SIZE),
        NORMALIZED_LOGIT_MIN,
        dtype=torch.float64,
    )
    for (row, column), value in zip(
        problem.active_nodes, values.tolist(), strict=True
    ):
        low[0, 0, row, column] = float(value)
    return torch.nn.functional.interpolate(
        low,
        size=(OUTPUT_GRID_SIZE, OUTPUT_GRID_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


@dataclass(frozen=True)
class TinyTargetCaseCertificate:
    case_id: str
    case_status: str
    reason: str | None
    irreducible_false_addition_pixels: int | None
    localized_certifying_margin: float | None
    target_pixel_recall: float | None
    target_matched: bool | None
    retention: float | None
    pixel_fa: float | None
    raw_background_fa: float | None
    fp_components_per_mp: float | None
    active_value_hex: tuple[str, ...]
    positive_background_pixels: tuple[Pixel, ...]
    budget_violations: tuple[str, ...]
    bound_normalization_max_abs: float | None
    dense_problem_fingerprint: str | None
    witness_fingerprint: str | None
    stage_1_solver: Mapping[str, object]
    stage_2_solver: Mapping[str, object]
    certificate_fingerprint: str
    schema_version: str = TINY_TARGET_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TINY_TARGET_CERTIFICATE_SCHEMA:
            raise ValueError("unsupported tiny-target certificate schema")
        _require_sha256(self.case_id, name="certificate case_id")
        if self.case_status not in {
            "PASS",
            "STRUCTURAL_FAIL",
            "INCONCLUSIVE",
        }:
            raise ValueError("unsupported tiny-target case status")
        if tuple(sorted(set(self.budget_violations))) != (
            self.budget_violations
        ):
            raise ValueError(
                "budget_violations must be sorted, unique, and canonical"
            )
        allowed_violations = {
            "target_pixel_recall",
            "target_matching",
            "pixel_fa",
            "raw_background_fa",
            "fp_components_per_mp",
        }
        if not set(self.budget_violations) <= allowed_violations:
            raise ValueError("unknown tiny-target budget violation")
        if self.case_status == "INCONCLUSIVE":
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("INCONCLUSIVE certificate requires a reason")
            if any(
                value is not None
                for value in (
                    self.irreducible_false_addition_pixels,
                    self.localized_certifying_margin,
                    self.target_pixel_recall,
                    self.target_matched,
                    self.retention,
                    self.pixel_fa,
                    self.raw_background_fa,
                    self.fp_components_per_mp,
                    self.bound_normalization_max_abs,
                    self.witness_fingerprint,
                )
            ):
                raise ValueError(
                    "INCONCLUSIVE certificate may not claim solved metrics"
                )
            if (
                self.active_value_hex
                or self.positive_background_pixels
                or self.budget_violations
            ):
                raise ValueError(
                    "INCONCLUSIVE certificate may not contain a witness"
                )
        else:
            if self.case_status == "PASS":
                if self.reason is not None or self.budget_violations:
                    raise ValueError("PASS certificate contains a failure claim")
            elif (
                not isinstance(self.reason, str)
                or not self.reason
                or not self.budget_violations
            ):
                raise ValueError(
                    "STRUCTURAL_FAIL certificate requires violations"
                )
            count = _require_int(
                self.irreducible_false_addition_pixels,
                name="irreducible_false_addition_pixels",
                minimum=0,
            )
            numeric_values = (
                self.localized_certifying_margin,
                self.target_pixel_recall,
                self.retention,
                self.pixel_fa,
                self.raw_background_fa,
                self.fp_components_per_mp,
                self.bound_normalization_max_abs,
            )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                for value in numeric_values
            ):
                raise ValueError("solved certificate metrics must be finite")
            if (
                float(self.localized_certifying_margin)
                < CERTIFYING_MARGIN_MIN - VERIFY_TOLERANCE
                or not 0.0 <= float(self.target_pixel_recall) <= 1.0
                or self.retention != self.target_pixel_recall
                or float(self.bound_normalization_max_abs)
                > VERIFY_TOLERANCE
                or not isinstance(self.target_matched, bool)
            ):
                raise ValueError("solved certificate metric semantics changed")
            if (
                len(self.positive_background_pixels) != count
                or len(set(self.positive_background_pixels))
                != len(self.positive_background_pixels)
                or any(
                    not 0 <= row < OUTPUT_GRID_SIZE
                    or not 0 <= column < OUTPUT_GRID_SIZE
                    for row, column in self.positive_background_pixels
                )
            ):
                raise ValueError(
                    "positive-background witness differs from its count"
                )
            if not self.active_value_hex:
                raise ValueError("solved certificate requires active values")
            try:
                active_values = tuple(
                    float.fromhex(value) for value in self.active_value_hex
                )
            except (TypeError, ValueError) as error:
                raise ValueError("active_value_hex is malformed") from error
            if any(
                not isfinite(value)
                or value < NORMALIZED_LOGIT_MIN
                or value > NORMALIZED_LOGIT_MAX
                for value in active_values
            ):
                raise ValueError("active_value_hex lies outside frozen bounds")
            _require_sha256(
                self.dense_problem_fingerprint,
                name="dense_problem_fingerprint",
            )
            _require_sha256(
                self.witness_fingerprint,
                name="witness_fingerprint",
            )
        if self.dense_problem_fingerprint is not None:
            _require_sha256(
                self.dense_problem_fingerprint,
                name="dense_problem_fingerprint",
            )
        if not isinstance(self.stage_1_solver, Mapping) or not isinstance(
            self.stage_2_solver, Mapping
        ):
            raise TypeError("solver certificate payloads must be mappings")
        if self.case_status != "INCONCLUSIVE" and (
            not self.stage_1_solver or not self.stage_2_solver
        ):
            raise ValueError(
                "solved certificate requires non-empty solver payloads"
            )
        _require_sha256(
            self.certificate_fingerprint,
            name="certificate_fingerprint",
        )
        if self.certificate_fingerprint != stable_fingerprint(
            self.payload(unsigned=True)
        ):
            raise ValueError("certificate fingerprint is inconsistent")

    def payload(self, *, unsigned: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "case_status": self.case_status,
            "reason": self.reason,
            "irreducible_false_addition_pixels": (
                self.irreducible_false_addition_pixels
            ),
            "localized_certifying_margin": (
                self.localized_certifying_margin
            ),
            "target_pixel_recall": self.target_pixel_recall,
            "target_matched": self.target_matched,
            "retention": self.retention,
            "retention_semantics": "target_pixel_recall",
            "stage_a_anchor_retention_applicable": False,
            "pixel_fa": self.pixel_fa,
            "raw_background_fa": self.raw_background_fa,
            "fp_components_per_mp": self.fp_components_per_mp,
            "active_value_hex": list(self.active_value_hex),
            "positive_background_pixels": [
                list(pixel) for pixel in self.positive_background_pixels
            ],
            "budget_violations": list(self.budget_violations),
            "bound_normalization_max_abs": (
                self.bound_normalization_max_abs
            ),
            "dense_problem_fingerprint": self.dense_problem_fingerprint,
            "witness_fingerprint": self.witness_fingerprint,
            "stage_1_solver": dict(self.stage_1_solver),
            "stage_2_solver": dict(self.stage_2_solver),
        }
        if not unsigned:
            payload["certificate_fingerprint"] = self.certificate_fingerprint
        return payload


def _solver_payload(result: Any) -> dict[str, object]:
    def optional_number(name: str) -> float | int | None:
        value = getattr(result, name, None)
        if value is None:
            return None
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return int(value)
        number = float(value)
        return number if isfinite(number) else None

    return {
        "status": int(getattr(result, "status", -1)),
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "objective": optional_number("fun"),
        "mip_gap": optional_number("mip_gap"),
        "mip_node_count": optional_number("mip_node_count"),
        "mip_dual_bound": optional_number("mip_dual_bound"),
    }


def _require_verified_solver_payload(
    payload: Mapping[str, object],
    *,
    name: str,
) -> None:
    """Reject a claimed optimal stage that lacks reconstructed evidence."""

    required = {
        "status",
        "success",
        "objective",
        "reconstructed_objective",
        "certificate_verification",
        "certificate_failure_reason",
        "reconstructed_optimality_error",
        "optimality_certificate_kind",
        "optimality_tolerance",
        "maximum_bound_violation_abs",
        "maximum_integrality_violation_abs",
        "maximum_linear_constraint_violation_abs",
        "objective_reconstruction_error_abs",
        "maximum_bound_normalization_abs",
    }
    if not required <= set(payload):
        missing = sorted(required - set(payload))
        raise ValueError(f"{name} is missing verified fields: {missing}")
    if (
        payload["status"] != 0
        or payload["success"] is not True
        or payload["certificate_verification"] != "passed"
        or payload["certificate_failure_reason"] is not None
    ):
        raise ValueError(f"{name} does not contain a passed solver certificate")
    for field in (
        "objective",
        "reconstructed_objective",
        "reconstructed_optimality_error",
        "optimality_tolerance",
        "maximum_bound_violation_abs",
        "maximum_integrality_violation_abs",
        "maximum_linear_constraint_violation_abs",
        "objective_reconstruction_error_abs",
        "maximum_bound_normalization_abs",
    ):
        value = payload[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
        ):
            raise ValueError(f"{name}.{field} must be finite")
    if float(payload["reconstructed_optimality_error"]) > float(
        payload["optimality_tolerance"]
    ):
        raise ValueError(f"{name} optimality certificate exceeds tolerance")
    if max(
        float(payload[field])
        for field in (
            "maximum_bound_violation_abs",
            "maximum_integrality_violation_abs",
            "maximum_linear_constraint_violation_abs",
            "objective_reconstruction_error_abs",
            "maximum_bound_normalization_abs",
        )
    ) > VERIFY_TOLERANCE:
        raise ValueError(f"{name} reconstruction exceeds tolerance")


def _search_has_frozen_zero_certificate(
    *,
    margin: float,
    payload: Mapping[str, object],
) -> bool:
    """Apply the fail-closed numerical zero convention for count exclusion."""

    return (
        margin == 0.0
        and payload.get("objective") == 0.0
        and payload.get("mip_dual_bound") == 0.0
        and payload.get("reconstructed_objective") == 0.0
    )


def _inconclusive_certificate(
    case: TinyTargetCase,
    *,
    reason: str,
    dense_problem_fingerprint: str | None = None,
    stage_1: Mapping[str, object] | None = None,
    stage_2: Mapping[str, object] | None = None,
) -> TinyTargetCaseCertificate:
    unsigned = {
        "schema_version": TINY_TARGET_CERTIFICATE_SCHEMA,
        "case_id": case.case_id,
        "case_status": "INCONCLUSIVE",
        "reason": reason,
        "irreducible_false_addition_pixels": None,
        "localized_certifying_margin": None,
        "target_pixel_recall": None,
        "target_matched": None,
        "retention": None,
        "retention_semantics": "target_pixel_recall",
        "stage_a_anchor_retention_applicable": False,
        "pixel_fa": None,
        "raw_background_fa": None,
        "fp_components_per_mp": None,
        "active_value_hex": [],
        "positive_background_pixels": [],
        "budget_violations": [],
        "bound_normalization_max_abs": None,
        "dense_problem_fingerprint": dense_problem_fingerprint,
        "witness_fingerprint": None,
        "stage_1_solver": dict(stage_1 or {}),
        "stage_2_solver": dict(stage_2 or {}),
    }
    return TinyTargetCaseCertificate(
        case_id=case.case_id,
        case_status="INCONCLUSIVE",
        reason=reason,
        irreducible_false_addition_pixels=None,
        localized_certifying_margin=None,
        target_pixel_recall=None,
        target_matched=None,
        retention=None,
        pixel_fa=None,
        raw_background_fa=None,
        fp_components_per_mp=None,
        active_value_hex=(),
        positive_background_pixels=(),
        budget_violations=(),
        bound_normalization_max_abs=None,
        dense_problem_fingerprint=dense_problem_fingerprint,
        witness_fingerprint=None,
        stage_1_solver=dict(stage_1 or {}),
        stage_2_solver=dict(stage_2 or {}),
        certificate_fingerprint=stable_fingerprint(unsigned),
    )


def _build_milp_constraints(
    problem: RepresentabilityProblem,
    *,
    fixed_background_count: int | None,
    maximum_background_count: int | None = None,
    certifying_margin: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, ...]]:
    target_rows = tuple(
        index for index, value in enumerate(problem.target_indicator) if value
    )
    background_rows = tuple(
        index for index, value in enumerate(problem.target_indicator) if not value
    )
    active_count = len(problem.active_nodes)
    background_count = len(background_rows)
    variable_count = active_count + background_count + 1
    gamma_index = variable_count - 1
    coefficients = (
        np.asarray(problem.coefficient_numerators, dtype=np.float64)
        / BILINEAR_WEIGHT_DENOMINATOR
    )
    constants = (
        np.asarray(problem.constant_numerators, dtype=np.float64)
        / BILINEAR_WEIGHT_DENOMINATOR
    )
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for output_row in target_rows:
        row = np.zeros(variable_count, dtype=np.float64)
        row[:active_count] = coefficients[output_row]
        if certifying_margin:
            row[gamma_index] = -1.0
        rows.append(row)
        lower.append(-constants[output_row])
        upper.append(np.inf)
    for binary_index, output_row in enumerate(background_rows):
        binary_column = active_count + binary_index
        # Stage 1: u <= M*z.  Stage 2: u <= -gamma + M*z.
        # Thus z=0 is a relaxed non-positive row in Stage 1 and a
        # certified strictly-negative row in Stage 2.
        upper_row = np.zeros(variable_count, dtype=np.float64)
        upper_row[:active_count] = coefficients[output_row]
        upper_row[binary_column] = -BIG_M
        if certifying_margin:
            upper_row[gamma_index] = 1.0
        rows.append(upper_row)
        lower.append(-np.inf)
        upper.append(-constants[output_row])
        # u >= -M*(1-z).  A z=1 row is non-negative, matching the
        # frozen inclusive threshold.  No positive-background margin is
        # required.
        lower_row = np.zeros(variable_count, dtype=np.float64)
        lower_row[:active_count] = coefficients[output_row]
        lower_row[binary_column] = -BIG_M
        rows.append(lower_row)
        lower.append(-constants[output_row] - BIG_M)
        upper.append(np.inf)
    if (
        fixed_background_count is not None
        and maximum_background_count is not None
    ):
        raise ValueError(
            "background count may be fixed or upper-bounded, not both"
        )
    if fixed_background_count is not None:
        count_row = np.zeros(variable_count, dtype=np.float64)
        count_row[active_count : active_count + background_count] = 1.0
        rows.append(count_row)
        lower.append(float(fixed_background_count))
        upper.append(float(fixed_background_count))
    elif maximum_background_count is not None:
        count_row = np.zeros(variable_count, dtype=np.float64)
        count_row[active_count : active_count + background_count] = 1.0
        rows.append(count_row)
        lower.append(0.0)
        upper.append(float(maximum_background_count))
    return (
        np.stack(rows),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        gamma_index,
        background_rows,
    )


def _verify_solver_result(
    result: Any,
    *,
    objective: np.ndarray,
    integrality: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    matrix: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, object], str | None, float | None]:
    """Independently reconstruct one complete MILP certificate."""

    payload = _solver_payload(result)
    raw = getattr(result, "x", None)
    if (
        payload["status"] != 0
        or payload["success"] is not True
        or raw is None
        or payload["objective"] is None
    ):
        payload["certificate_verification"] = "failed"
        payload["certificate_failure_reason"] = (
            "solver_not_optimal_or_missing_required_fields"
        )
        return None, payload, "solver_not_optimal_or_missing_fields", None
    if payload["mip_dual_bound"] is not None:
        optimality_error = abs(
            float(payload["objective"])
            - float(payload["mip_dual_bound"])
        )
        optimality_tolerance = VERIFY_TOLERANCE
        optimality_kind = "absolute_primal_dual_objective_gap"
    elif payload["mip_gap"] is not None:
        optimality_error = float(payload["mip_gap"])
        optimality_tolerance = 1e-9
        optimality_kind = "reported_relative_mip_gap"
    else:
        payload["certificate_verification"] = "failed"
        payload["certificate_failure_reason"] = (
            "missing_relative_and_absolute_optimality_certificate"
        )
        return None, payload, "missing_optimality_certificate", None
    payload["reconstructed_optimality_error"] = optimality_error
    payload["optimality_certificate_kind"] = optimality_kind
    payload["optimality_tolerance"] = optimality_tolerance
    if optimality_error > optimality_tolerance:
        payload["certificate_verification"] = "failed"
        payload["certificate_failure_reason"] = (
            "solver_optimality_error_exceeded_tolerance"
        )
        return None, payload, "solver_optimality_error", None
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != objective.shape or not np.isfinite(values).all():
        payload["certificate_verification"] = "failed"
        payload["certificate_failure_reason"] = "nonfinite_or_wrong_shape_primal"
        return None, payload, "nonfinite_or_wrong_shape_primal", None

    lower_bound_violation = float(
        np.max(np.maximum(bounds_lower - values, 0.0), initial=0.0)
    )
    upper_bound_violation = float(
        np.max(np.maximum(values - bounds_upper, 0.0), initial=0.0)
    )
    integral_indices = np.flatnonzero(integrality)
    integrality_violation = (
        float(
            np.max(
                np.abs(
                    values[integral_indices]
                    - np.rint(values[integral_indices])
                ),
                initial=0.0,
            )
        )
        if integral_indices.size
        else 0.0
    )
    normalized = np.clip(values, bounds_lower, bounds_upper)
    if integral_indices.size:
        normalized[integral_indices] = np.rint(
            normalized[integral_indices]
        )
    normalization = float(
        np.max(np.abs(normalized - values), initial=0.0)
    )
    activity = matrix @ normalized
    finite_lower = np.isfinite(constraint_lower)
    finite_upper = np.isfinite(constraint_upper)
    lower_constraint_violation = float(
        np.max(
            np.maximum(
                constraint_lower[finite_lower] - activity[finite_lower],
                0.0,
            ),
            initial=0.0,
        )
    )
    upper_constraint_violation = float(
        np.max(
            np.maximum(
                activity[finite_upper] - constraint_upper[finite_upper],
                0.0,
            ),
            initial=0.0,
        )
    )
    reconstructed_objective = float(objective @ normalized)
    objective_error = abs(
        reconstructed_objective - float(payload["objective"])
    )
    maximum_violation = max(
        lower_bound_violation,
        upper_bound_violation,
        integrality_violation,
        lower_constraint_violation,
        upper_constraint_violation,
        objective_error,
    )
    payload.update(
        {
            "reconstructed_objective": reconstructed_objective,
            "maximum_bound_violation_abs": max(
                lower_bound_violation,
                upper_bound_violation,
            ),
            "maximum_integrality_violation_abs": integrality_violation,
            "maximum_linear_constraint_violation_abs": max(
                lower_constraint_violation,
                upper_constraint_violation,
            ),
            "objective_reconstruction_error_abs": objective_error,
            "maximum_bound_normalization_abs": normalization,
        }
    )
    if maximum_violation > VERIFY_TOLERANCE:
        payload["certificate_verification"] = "failed"
        payload["certificate_failure_reason"] = (
            "independent_reconstruction_exceeded_tolerance"
        )
        return (
            None,
            payload,
            "independent_solver_reconstruction_failed",
            normalization,
        )
    payload["certificate_verification"] = "passed"
    payload["certificate_failure_reason"] = None
    return normalized, payload, None, normalization


def solve_tiny_target_case(
    case: TinyTargetCase,
    *,
    solver_options: Mapping[str, object] | None = None,
) -> TinyTargetCaseCertificate:
    """Solve and independently reconstruct one fail-closed case certificate."""

    if not isinstance(case, TinyTargetCase):
        raise TypeError("case must be TinyTargetCase")
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception as error:  # pragma: no cover - environment-specific
        return _inconclusive_certificate(
            case, reason=f"scipy_milp_unavailable:{type(error).__name__}"
        )

    row_offsets = tuple(row for row, _ in case.shape.pixels)
    column_offsets = tuple(column for _, column in case.shape.pixels)
    if any(
        stable_fingerprint(_axis_equivalence_signature(origin, row_offsets))
        != case.row_axis_signature_fingerprint
        for origin in case.row_origins
    ) or any(
        stable_fingerprint(
            _axis_equivalence_signature(origin, column_offsets)
        )
        != case.column_axis_signature_fingerprint
        for origin in case.column_origins
    ):
        return _inconclusive_certificate(
            case, reason="case_equivalence_member_signature_mismatch"
        )
    compact_fingerprint = _compact_problem_fingerprint(
        case.shape,
        case.row_axis_signature_fingerprint,
        case.column_axis_signature_fingerprint,
    )
    if compact_fingerprint != case.problem_fingerprint:
        return _inconclusive_certificate(
            case, reason="case_compact_problem_fingerprint_mismatch"
        )

    problem = build_representability_problem(
        case.shape, case.representative_origin
    )
    dense_problem_fingerprint = problem.problem_fingerprint
    active_count = len(problem.active_nodes)
    background_rows = tuple(
        index
        for index, is_target in enumerate(problem.target_indicator)
        if not is_target
    )
    background_count = len(background_rows)
    variable_count = active_count + background_count + 1
    gamma_index = variable_count - 1
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[active_count : active_count + background_count] = 1
    options: dict[str, object] = dict(DEFAULT_SOLVER_OPTIONS)
    if solver_options is not None:
        if set(solver_options) - {
            *DEFAULT_SOLVER_OPTIONS,
            "time_limit",
            "node_limit",
        }:
            raise ValueError("unsupported MILP solver option")
        options.update(dict(solver_options))

    def run_milp(*args: object, **kwargs: object) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected:.*",
                category=RuntimeWarning,
            )
            return milp(*args, **kwargs)

    # Stage 1 computes a threshold-tie relaxation: targets/background z=1 are
    # non-negative and background z=0 rows are non-positive.  Its minimum is a
    # certified lower bound on the inclusive-threshold sign-count optimum.
    matrix_1, lower_1, upper_1, gamma_index_1, rows_1 = (
        _build_milp_constraints(
            problem,
            fixed_background_count=None,
            certifying_margin=False,
        )
    )
    if gamma_index_1 != gamma_index or rows_1 != background_rows:
        return _inconclusive_certificate(
            case,
            reason="stage_1_variable_identity_changed",
            dense_problem_fingerprint=dense_problem_fingerprint,
        )
    bounds_lower_1 = np.concatenate(
        (
            np.full(active_count, NORMALIZED_LOGIT_MIN),
            np.zeros(background_count),
            np.asarray([0.0]),
        )
    )
    bounds_upper_1 = np.concatenate(
        (
            np.full(active_count, NORMALIZED_LOGIT_MAX),
            np.ones(background_count),
            np.asarray([0.0]),
        )
    )
    bounds_upper = np.concatenate(
        (
            np.full(active_count, NORMALIZED_LOGIT_MAX),
            np.ones(background_count),
            np.asarray([GAMMA_MAX]),
        )
    )
    objective_1 = np.zeros(variable_count, dtype=np.float64)
    objective_1[
        active_count : active_count + background_count
    ] = 1.0
    try:
        stage_1 = run_milp(
            objective_1,
            integrality=integrality,
            bounds=Bounds(bounds_lower_1, bounds_upper_1),
            constraints=LinearConstraint(matrix_1, lower_1, upper_1),
            options=options,
        )
    except Exception as error:
        return _inconclusive_certificate(
            case,
            reason=f"stage_1_solver_exception:{type(error).__name__}",
            dense_problem_fingerprint=dense_problem_fingerprint,
        )
    (
        stage_1_values,
        stage_1_payload,
        stage_1_error,
        stage_1_normalization,
    ) = _verify_solver_result(
        stage_1,
        objective=objective_1,
        integrality=integrality,
        bounds_lower=bounds_lower_1,
        bounds_upper=bounds_upper_1,
        matrix=matrix_1,
        constraint_lower=lower_1,
        constraint_upper=upper_1,
    )
    if stage_1_error is not None or stage_1_values is None:
        return _inconclusive_certificate(
            case,
            reason=f"stage_1_{stage_1_error}",
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
        )
    relaxed_lower_count = int(
        np.sum(
            stage_1_values[
                active_count : active_count + background_count
            ]
        )
    )
    stage_1_payload.update(
        {
            "stage_kind": "threshold_tie_relaxed_lower_bound",
            "reconstructed_background_count": relaxed_lower_count,
            "reconstructed_gamma": 0.0,
        }
    )
    if abs(
        float(stage_1_payload["reconstructed_objective"])
        - relaxed_lower_count
    ) > VERIFY_TOLERANCE:
        return _inconclusive_certificate(
            case,
            reason="stage_1_count_objective_mismatch",
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
        )

    # Build a valid margin-separated upper bound.  Unlike Stage 1, this
    # solution already has the exact inclusive-threshold interpretation.
    matrix_upper, lower_upper, upper_upper, gamma_upper, rows_upper = (
        _build_milp_constraints(
            problem,
            fixed_background_count=None,
            certifying_margin=True,
        )
    )
    bounds_lower_upper = np.concatenate(
        (
            np.full(active_count, NORMALIZED_LOGIT_MIN),
            np.zeros(background_count),
            np.asarray([CERTIFYING_MARGIN_MIN]),
        )
    )
    try:
        upper_result = run_milp(
            objective_1,
            integrality=integrality,
            bounds=Bounds(bounds_lower_upper, bounds_upper),
            constraints=LinearConstraint(
                matrix_upper,
                lower_upper,
                upper_upper,
            ),
            options=options,
        )
    except Exception as error:
        return _inconclusive_certificate(
            case,
            reason=f"upper_bound_solver_exception:{type(error).__name__}",
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
        )
    (
        upper_values,
        upper_payload,
        upper_error,
        upper_normalization,
    ) = _verify_solver_result(
        upper_result,
        objective=objective_1,
        integrality=integrality,
        bounds_lower=bounds_lower_upper,
        bounds_upper=bounds_upper,
        matrix=matrix_upper,
        constraint_lower=lower_upper,
        constraint_upper=upper_upper,
    )
    if (
        gamma_upper != gamma_index
        or rows_upper != background_rows
        or upper_error is not None
        or upper_values is None
    ):
        return _inconclusive_certificate(
            case,
            reason=f"upper_bound_{upper_error or 'variable_identity_changed'}",
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
            stage_2={"initial_upper_solver": upper_payload},
        )
    current_count = int(
        np.sum(
            upper_values[
                active_count : active_count + background_count
            ]
        )
    )
    upper_payload.update(
        {
            "stage_kind": "margin_separated_count_upper_bound",
            "reconstructed_background_count": current_count,
            "reconstructed_gamma": float(upper_values[gamma_index]),
        }
    )
    if (
        current_count < relaxed_lower_count
        or abs(
            float(upper_payload["reconstructed_objective"])
            - current_count
        )
        > VERIFY_TOLERANCE
    ):
        return _inconclusive_certificate(
            case,
            reason="upper_bound_count_reconstruction_failed",
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
            stage_2={"initial_upper_solver": upper_payload},
        )

    # Exactness search.  If a lower-count inclusive-threshold solution exists,
    # monotone localization and a finite constant shift imply a localized
    # solution with a strictly positive symmetric margin.  We therefore
    # maximize that margin under successively smaller count caps.  A certified
    # zero optimum excludes all lower sign-count solutions under the frozen
    # numerical convention.  Any positive optimum is INCONCLUSIVE: a margin
    # at least CERTIFYING_MARGIN_MIN would also contradict the already
    # certified optimal valid upper, while a smaller positive margin leaves
    # the exact inclusive-threshold count unresolved.
    search_attempts: list[dict[str, object]] = []
    selected_values = upper_values
    selected_witness_source = "initial_upper_solver"
    maximum_normalization = max(
        float(stage_1_normalization or 0.0),
        float(upper_normalization or 0.0),
    )
    while current_count > relaxed_lower_count:
        cap = current_count - 1
        matrix_search, lower_search, upper_search, gamma_search, rows_search = (
            _build_milp_constraints(
                problem,
                fixed_background_count=None,
                maximum_background_count=cap,
                certifying_margin=True,
            )
        )
        if gamma_search != gamma_index or rows_search != background_rows:
            return _inconclusive_certificate(
                case,
                reason="exactness_search_variable_identity_changed",
                dense_problem_fingerprint=dense_problem_fingerprint,
                stage_1=stage_1_payload,
                stage_2={"attempts": search_attempts},
            )
        search_bounds_lower = np.concatenate(
            (
                np.full(active_count, NORMALIZED_LOGIT_MIN),
                np.zeros(background_count),
                np.asarray([0.0]),
            )
        )
        search_objective = np.zeros(variable_count, dtype=np.float64)
        search_objective[gamma_index] = -MARGIN_OBJECTIVE_SCALE
        try:
            search_result = run_milp(
                search_objective,
                integrality=integrality,
                bounds=Bounds(search_bounds_lower, bounds_upper),
                constraints=LinearConstraint(
                    matrix_search,
                    lower_search,
                    upper_search,
                ),
                options=options,
            )
        except Exception as error:
            return _inconclusive_certificate(
                case,
                reason=(
                    "exactness_search_solver_exception:"
                    f"{type(error).__name__}"
                ),
                dense_problem_fingerprint=dense_problem_fingerprint,
                stage_1=stage_1_payload,
                stage_2={"attempts": search_attempts},
            )
        (
            search_values,
            search_payload,
            search_error,
            search_normalization,
        ) = _verify_solver_result(
            search_result,
            objective=search_objective,
            integrality=integrality,
            bounds_lower=search_bounds_lower,
            bounds_upper=bounds_upper,
            matrix=matrix_search,
            constraint_lower=lower_search,
            constraint_upper=upper_search,
        )
        if search_error is not None or search_values is None:
            search_payload["count_cap"] = cap
            search_attempts.append(search_payload)
            return _inconclusive_certificate(
                case,
                reason=f"exactness_search_{search_error}",
                dense_problem_fingerprint=dense_problem_fingerprint,
                stage_1=stage_1_payload,
                stage_2={"attempts": search_attempts},
            )
        maximum_normalization = max(
            maximum_normalization,
            float(search_normalization or 0.0),
        )
        search_margin = float(search_values[gamma_index])
        lower_count = int(
            np.sum(
                search_values[
                    active_count : active_count + background_count
                ]
            )
        )
        zero_certificate = _search_has_frozen_zero_certificate(
            margin=search_margin,
            payload=search_payload,
        )
        search_payload.update(
            {
                "stage_kind": "descending_lower_count_exclusion",
                "count_cap": cap,
                "reconstructed_background_count": lower_count,
                "reconstructed_gamma": search_margin,
                "margin_objective_scale": MARGIN_OBJECTIVE_SCALE,
                "frozen_zero_certificate": zero_certificate,
            }
        )
        search_attempts.append(search_payload)
        if zero_certificate:
            break
        return _inconclusive_certificate(
            case,
            reason=(
                "exactness_search_positive_margin_conflicts_with_"
                "valid_upper_or_leaves_inclusive_count_unresolved"
            ),
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
            stage_2={"attempts": search_attempts},
        )

    stage_2_payload: dict[str, object] = {
        "initial_upper_solver": upper_payload,
        "exactness_search_attempts": search_attempts,
        "selected_witness_source": selected_witness_source,
        "selected_margin_is_free_grid_maximum_claim": False,
    }
    active_values = selected_values[:active_count]
    rounded_background = selected_values[
        active_count : active_count + background_count
    ]
    margin = float(selected_values[gamma_index])
    final_count = int(np.sum(rounded_background))
    if (
        final_count != current_count
        or margin < CERTIFYING_MARGIN_MIN - VERIFY_TOLERANCE
    ):
        return _inconclusive_certificate(
            case,
            reason="selected_witness_count_or_margin_reconstruction_failed",
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
            stage_2=stage_2_payload,
        )

    try:
        output = reconstruct_output_logits(problem, active_values)
    except Exception as error:
        return _inconclusive_certificate(
            case,
            reason=f"direct_logit_reconstruction_exception:{type(error).__name__}",
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
            stage_2=stage_2_payload,
        )
    target_set = set(problem.target_pixels)
    target_values = np.asarray(
        [output[pixel] for pixel in problem.target_pixels],
        dtype=np.float64,
    )
    expected_positive = rounded_background.astype(bool)
    background_values = np.asarray(
        [
            output[problem.affected_pixels[row]]
            for row in background_rows
        ],
        dtype=np.float64,
    )
    actual_positive = background_values >= 0.0
    positive_background = tuple(
        problem.affected_pixels[output_row]
        for is_positive, output_row in zip(
            actual_positive,
            background_rows,
            strict=True,
        )
        if is_positive
    )
    if (
        np.any(target_values < margin - VERIFY_TOLERANCE)
        or np.any(target_values < 0.0)
        or not np.array_equal(actual_positive, expected_positive)
        or np.any(
            background_values[expected_positive]
            < 0.0
        )
        or np.any(
            background_values[~expected_positive]
            > -margin + VERIFY_TOLERANCE
        )
        or np.any(background_values[~expected_positive] >= 0.0)
        or len(positive_background) != current_count
    ):
        return _inconclusive_certificate(
            case,
            reason="direct_logit_reconstruction_failed",
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
            stage_2=stage_2_payload,
        )

    prediction = torch.from_numpy(output >= 0.0)
    target = torch.zeros_like(prediction)
    for row, column in target_set:
        target[row, column] = True
    metrics = evaluate_binary_prediction(
        prediction,
        target,
        MatchConfig(),
        anchor_miss_ids=frozenset({1}),
        reachable_anchor_miss_ids=frozenset({1}),
        residual_mask=prediction,
    )
    target_pixel_recall = float(
        torch.count_nonzero(prediction & target)
    ) / float(torch.count_nonzero(target))
    target_matched = metrics.matched_gt == 1
    violations: list[str] = []
    if target_pixel_recall != RETENTION_REQUIRED:
        violations.append("target_pixel_recall")
    if not target_matched:
        violations.append("target_matching")
    if metrics.pixel_fa > PIXEL_FA_MAX:
        violations.append("pixel_fa")
    if metrics.raw_background_fa > RAW_BACKGROUND_FA_MAX:
        violations.append("raw_background_fa")
    if metrics.fp_components_per_mp > FP_COMPONENTS_PER_MP_MAX:
        violations.append("fp_components_per_mp")
    violations_tuple = tuple(sorted(violations))
    if current_count <= MAX_FALSE_ADDITION_PIXELS and violations_tuple:
        stage_2_payload["secondary_budget_diagnostic"] = {
            "minimum_count": current_count,
            "budget_violations": list(violations_tuple),
            "interpretation": (
                "selected_max_margin_witness_does_not_prove_all_"
                "minimum_count_witnesses_fail"
            ),
        }
        return _inconclusive_certificate(
            case,
            reason=(
                "minimum_count_witness_does_not_certify_all_secondary_budgets"
            ),
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
            stage_2=stage_2_payload,
        )
    status = (
        "STRUCTURAL_FAIL"
        if current_count > MAX_FALSE_ADDITION_PIXELS
        else "PASS"
    )
    if status == "STRUCTURAL_FAIL" and (
        "raw_background_fa" not in violations_tuple
    ):
        return _inconclusive_certificate(
            case,
            reason="exact_count_and_raw_background_metric_disagree",
            dense_problem_fingerprint=dense_problem_fingerprint,
            stage_1=stage_1_payload,
            stage_2=stage_2_payload,
        )
    reason = (
        "tolerance_certified_minimum_false_additions_exceed_prefrozen_raw_budget"
        if status == "STRUCTURAL_FAIL"
        else None
    )
    active_hex = tuple(
        float(value).hex() for value in active_values.tolist()
    )
    witness_core = {
        "compact_problem_fingerprint": case.problem_fingerprint,
        "dense_problem_fingerprint": dense_problem_fingerprint,
        "active_nodes": [list(value) for value in problem.active_nodes],
        "active_value_hex": list(active_hex),
        "positive_background_pixels": [
            list(value) for value in positive_background
        ],
        "tolerance_certified_minimum_false_addition_count": current_count,
        "localized_certifying_margin_hex": margin.hex(),
        "bound_normalization_max_abs_hex": (
            maximum_normalization.hex()
        ),
    }
    witness_fingerprint = stable_fingerprint(witness_core)
    metric_values = {
        "target_pixel_recall": target_pixel_recall,
        "target_matched": target_matched,
        "retention": target_pixel_recall,
        "retention_semantics": "target_pixel_recall",
        "stage_a_anchor_retention_applicable": False,
        "pixel_fa": metrics.pixel_fa,
        "raw_background_fa": metrics.raw_background_fa,
        "fp_components_per_mp": metrics.fp_components_per_mp,
    }
    unsigned = {
        "schema_version": TINY_TARGET_CERTIFICATE_SCHEMA,
        "case_id": case.case_id,
        "case_status": status,
        "reason": reason,
        "irreducible_false_addition_pixels": current_count,
        "localized_certifying_margin": margin,
        **metric_values,
        "active_value_hex": list(active_hex),
        "positive_background_pixels": [
            list(value) for value in positive_background
        ],
        "budget_violations": list(violations_tuple),
        "bound_normalization_max_abs": maximum_normalization,
        "dense_problem_fingerprint": dense_problem_fingerprint,
        "witness_fingerprint": witness_fingerprint,
        "stage_1_solver": stage_1_payload,
        "stage_2_solver": stage_2_payload,
    }
    return TinyTargetCaseCertificate(
        case_id=case.case_id,
        case_status=status,
        reason=reason,
        irreducible_false_addition_pixels=current_count,
        localized_certifying_margin=margin,
        target_pixel_recall=target_pixel_recall,
        target_matched=target_matched,
        retention=target_pixel_recall,
        pixel_fa=metrics.pixel_fa,
        raw_background_fa=metrics.raw_background_fa,
        fp_components_per_mp=metrics.fp_components_per_mp,
        active_value_hex=active_hex,
        positive_background_pixels=positive_background,
        budget_violations=violations_tuple,
        bound_normalization_max_abs=maximum_normalization,
        dense_problem_fingerprint=dense_problem_fingerprint,
        witness_fingerprint=witness_fingerprint,
        stage_1_solver=stage_1_payload,
        stage_2_solver=stage_2_payload,
        certificate_fingerprint=stable_fingerprint(unsigned),
    )


def _verify_case_solver_trace(
    certificate: TinyTargetCaseCertificate,
) -> tuple[int, float, float]:
    """Replay the recorded count-search state machine without trusting status."""

    stage_1 = certificate.stage_1_solver
    _require_verified_solver_payload(stage_1, name="stage_1_solver")
    if stage_1.get("stage_kind") != "threshold_tie_relaxed_lower_bound":
        raise ValueError("stage_1_solver kind changed")
    relaxed_count = _require_int(
        stage_1.get("reconstructed_background_count"),
        name="stage_1 reconstructed_background_count",
    )
    if (
        stage_1.get("reconstructed_gamma") != 0.0
        or float(stage_1["reconstructed_objective"]) != relaxed_count
    ):
        raise ValueError("stage_1 count or gamma reconstruction changed")

    stage_2 = certificate.stage_2_solver
    if set(stage_2) != {
        "initial_upper_solver",
        "exactness_search_attempts",
        "selected_witness_source",
        "selected_margin_is_free_grid_maximum_claim",
    }:
        raise ValueError("stage_2_solver fields changed")
    if stage_2["selected_margin_is_free_grid_maximum_claim"] is not False:
        raise ValueError("stage_2_solver makes an unsupported margin claim")
    initial = stage_2["initial_upper_solver"]
    if not isinstance(initial, Mapping):
        raise ValueError("initial_upper_solver must be a mapping")
    _require_verified_solver_payload(initial, name="initial_upper_solver")
    if initial.get("stage_kind") != "margin_separated_count_upper_bound":
        raise ValueError("initial_upper_solver kind changed")
    current_count = _require_int(
        initial.get("reconstructed_background_count"),
        name="initial_upper reconstructed_background_count",
    )
    initial_margin = initial.get("reconstructed_gamma")
    if (
        isinstance(initial_margin, bool)
        or not isinstance(initial_margin, (int, float))
        or not isfinite(float(initial_margin))
        or float(initial_margin)
        < CERTIFYING_MARGIN_MIN - VERIFY_TOLERANCE
        or float(initial["reconstructed_objective"]) != current_count
        or current_count < relaxed_count
    ):
        raise ValueError("initial_upper_solver count or margin is invalid")
    selected_margin = float(initial_margin)
    expected_source = "initial_upper_solver"
    normalization_values = [
        float(stage_1["maximum_bound_normalization_abs"]),
        float(initial["maximum_bound_normalization_abs"]),
    ]

    attempts = stage_2["exactness_search_attempts"]
    if not isinstance(attempts, (tuple, list)):
        raise ValueError("exactness_search_attempts must be a sequence")
    if current_count == relaxed_count and attempts:
        raise ValueError("closed count gap may not contain search attempts")
    zero_seen = False
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            raise ValueError("exactness-search attempt must be a mapping")
        _require_verified_solver_payload(
            attempt,
            name=f"exactness_search_attempt_{index}",
        )
        if (
            attempt.get("stage_kind")
            != "descending_lower_count_exclusion"
            or attempt.get("margin_objective_scale")
            != MARGIN_OBJECTIVE_SCALE
            or attempt.get("count_cap") != current_count - 1
        ):
            raise ValueError("exactness-search identity or count cap changed")
        count = _require_int(
            attempt.get("reconstructed_background_count"),
            name=f"exactness_search_attempt_{index} count",
        )
        margin_value = attempt.get("reconstructed_gamma")
        if (
            isinstance(margin_value, bool)
            or not isinstance(margin_value, (int, float))
            or not isfinite(float(margin_value))
            or float(margin_value) < 0.0
            or count > int(attempt["count_cap"])
        ):
            raise ValueError("exactness-search count or margin is invalid")
        margin = float(margin_value)
        if abs(
            float(attempt["reconstructed_objective"])
            + MARGIN_OBJECTIVE_SCALE * margin
        ) > VERIFY_TOLERANCE:
            raise ValueError("scaled margin objective does not reconstruct")
        is_zero = _search_has_frozen_zero_certificate(
            margin=margin,
            payload=attempt,
        )
        if attempt.get("frozen_zero_certificate") is not is_zero:
            raise ValueError("exactness-search zero certificate changed")
        normalization_values.append(
            float(attempt["maximum_bound_normalization_abs"])
        )
        if is_zero:
            if index != len(attempts):
                raise ValueError("zero search certificate must be final")
            zero_seen = True
            break
        raise ValueError(
            "solved trace may not contain a positive exactness-search margin"
        )

    if current_count > relaxed_count and not zero_seen:
        raise ValueError("count gap lacks a frozen zero certificate")
    if stage_2["selected_witness_source"] != expected_source:
        raise ValueError("selected witness source differs from search trace")
    if certificate.irreducible_false_addition_pixels != current_count:
        raise ValueError("certificate count differs from solver trace")
    if certificate.localized_certifying_margin != selected_margin:
        raise ValueError("certificate margin differs from selected witness")
    return current_count, selected_margin, max(normalization_values)


def verify_tiny_target_case_certificate(
    case: TinyTargetCase,
    certificate: TinyTargetCaseCertificate,
) -> None:
    """Independently replay one case witness before it may be aggregated."""

    if not isinstance(case, TinyTargetCase):
        raise TypeError("case must be TinyTargetCase")
    if not isinstance(certificate, TinyTargetCaseCertificate):
        raise TypeError("certificate must be TinyTargetCaseCertificate")
    if certificate.case_id != case.case_id:
        raise ValueError("certificate case_id differs from its case")

    row_offsets = tuple(row for row, _ in case.shape.pixels)
    column_offsets = tuple(column for _, column in case.shape.pixels)
    if any(
        stable_fingerprint(_axis_equivalence_signature(origin, row_offsets))
        != case.row_axis_signature_fingerprint
        for origin in case.row_origins
    ) or any(
        stable_fingerprint(
            _axis_equivalence_signature(origin, column_offsets)
        )
        != case.column_axis_signature_fingerprint
        for origin in case.column_origins
    ):
        raise ValueError("case equivalence member signature mismatch")
    if _compact_problem_fingerprint(
        case.shape,
        case.row_axis_signature_fingerprint,
        case.column_axis_signature_fingerprint,
    ) != case.problem_fingerprint:
        raise ValueError("case compact problem fingerprint mismatch")

    problem = build_representability_problem(
        case.shape,
        case.representative_origin,
    )
    if (
        certificate.dense_problem_fingerprint is not None
        and certificate.dense_problem_fingerprint
        != problem.problem_fingerprint
    ):
        raise ValueError("certificate dense problem fingerprint mismatch")
    if certificate.case_status == "INCONCLUSIVE":
        # An inconclusive record cannot authorize anything.  Early failures
        # may legitimately occur before a dense problem or solver trace exists.
        return

    if certificate.dense_problem_fingerprint != problem.problem_fingerprint:
        raise ValueError("solved certificate lacks the exact dense problem")
    count, margin, maximum_normalization = _verify_case_solver_trace(
        certificate
    )
    if certificate.bound_normalization_max_abs != maximum_normalization:
        raise ValueError("certificate bound normalization differs from trace")

    try:
        active_values = np.asarray(
            [
                float.fromhex(value)
                for value in certificate.active_value_hex
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("certificate active values are malformed") from error
    if active_values.shape != (len(problem.active_nodes),):
        raise ValueError("certificate active-value count changed")
    output = reconstruct_output_logits(problem, active_values)
    target_set = set(problem.target_pixels)
    background_rows = tuple(
        index
        for index, is_target in enumerate(problem.target_indicator)
        if not is_target
    )
    target_values = np.asarray(
        [output[pixel] for pixel in problem.target_pixels],
        dtype=np.float64,
    )
    background_values = np.asarray(
        [output[problem.affected_pixels[index]] for index in background_rows],
        dtype=np.float64,
    )
    actual_positive = background_values >= 0.0
    positive_background = tuple(
        problem.affected_pixels[output_row]
        for is_positive, output_row in zip(
            actual_positive,
            background_rows,
            strict=True,
        )
        if is_positive
    )
    if (
        np.any(target_values < margin - VERIFY_TOLERANCE)
        or np.any(target_values < 0.0)
        or np.any(background_values[~actual_positive] >= 0.0)
        or np.any(
            background_values[~actual_positive]
            > -margin + VERIFY_TOLERANCE
        )
        or positive_background
        != certificate.positive_background_pixels
        or len(positive_background) != count
    ):
        raise ValueError("certificate inclusive-threshold witness mismatch")

    active_hex = tuple(float(value).hex() for value in active_values.tolist())
    if active_hex != certificate.active_value_hex:
        raise ValueError("certificate active-value encoding is non-canonical")
    witness_core = {
        "compact_problem_fingerprint": case.problem_fingerprint,
        "dense_problem_fingerprint": problem.problem_fingerprint,
        "active_nodes": [list(value) for value in problem.active_nodes],
        "active_value_hex": list(active_hex),
        "positive_background_pixels": [
            list(value) for value in positive_background
        ],
        "tolerance_certified_minimum_false_addition_count": count,
        "localized_certifying_margin_hex": margin.hex(),
        "bound_normalization_max_abs_hex": maximum_normalization.hex(),
    }
    if certificate.witness_fingerprint != stable_fingerprint(witness_core):
        raise ValueError("certificate witness fingerprint mismatch")

    prediction = torch.from_numpy(output >= 0.0)
    target = torch.zeros_like(prediction)
    for row, column in target_set:
        target[row, column] = True
    metrics = evaluate_binary_prediction(
        prediction,
        target,
        MatchConfig(),
        anchor_miss_ids=frozenset({1}),
        reachable_anchor_miss_ids=frozenset({1}),
        residual_mask=prediction,
    )
    target_pixel_recall = float(
        torch.count_nonzero(prediction & target)
    ) / float(torch.count_nonzero(target))
    target_matched = metrics.matched_gt == 1
    violations: list[str] = []
    if target_pixel_recall != RETENTION_REQUIRED:
        violations.append("target_pixel_recall")
    if not target_matched:
        violations.append("target_matching")
    if metrics.pixel_fa > PIXEL_FA_MAX:
        violations.append("pixel_fa")
    if metrics.raw_background_fa > RAW_BACKGROUND_FA_MAX:
        violations.append("raw_background_fa")
    if metrics.fp_components_per_mp > FP_COMPONENTS_PER_MP_MAX:
        violations.append("fp_components_per_mp")
    violations_tuple = tuple(sorted(violations))
    expected_status = (
        "STRUCTURAL_FAIL"
        if count > MAX_FALSE_ADDITION_PIXELS
        else "PASS"
    )
    if expected_status == "PASS" and violations_tuple:
        raise ValueError(
            "minimum-count secondary-budget failure must be inconclusive"
        )
    expected_reason = (
        "tolerance_certified_minimum_false_additions_exceed_prefrozen_raw_budget"
        if expected_status == "STRUCTURAL_FAIL"
        else None
    )
    expected_fields = (
        certificate.case_status == expected_status,
        certificate.reason == expected_reason,
        certificate.target_pixel_recall == target_pixel_recall,
        certificate.target_matched == target_matched,
        certificate.retention == target_pixel_recall,
        certificate.pixel_fa == metrics.pixel_fa,
        certificate.raw_background_fa == metrics.raw_background_fa,
        certificate.fp_components_per_mp == metrics.fp_components_per_mp,
        certificate.budget_violations == violations_tuple,
    )
    if not all(expected_fields):
        raise ValueError("certificate status, metrics, or budgets do not replay")


def _aggregate_tiny_target_case_statuses(
    catalog: TinyTargetCaseCatalog,
    values: tuple[TinyTargetCaseCertificate, ...],
) -> dict[str, object]:
    """Aggregate already-validated records; kept separate for unit coverage."""

    counts = {
        status: sum(certificate.case_status == status for certificate in values)
        for status in ("PASS", "STRUCTURAL_FAIL", "INCONCLUSIVE")
    }
    if sum(counts.values()) != len(values):
        raise ValueError("case status aggregation is not exhaustive")
    if counts["INCONCLUSIVE"]:
        status = "COMPUTATIONALLY_INCONCLUSIVE"
    elif counts["STRUCTURAL_FAIL"]:
        status = "STRUCTURAL_FAIL"
    else:
        status = "LATE_STATIC_GATE_PASS"
    case_by_id = {case.case_id: case for case in catalog.cases}
    failing_details = tuple(
        {
            "case_id": certificate.case_id,
            "shape_id": case_by_id[certificate.case_id].shape.shape_id,
            "representative_origin": list(
                case_by_id[certificate.case_id].representative_origin
            ),
            "multiplicity": case_by_id[certificate.case_id].multiplicity,
            "irreducible_false_addition_pixels": (
                certificate.irreducible_false_addition_pixels
            ),
            "budget_violations": list(certificate.budget_violations),
        }
        for certificate in values
        if certificate.case_status == "STRUCTURAL_FAIL"
    )
    inconclusive = tuple(
        certificate.case_id
        for certificate in values
        if certificate.case_status == "INCONCLUSIVE"
    )
    maximum_false_additions = max(
        (
            certificate.irreducible_false_addition_pixels
            for certificate in values
            if certificate.irreducible_false_addition_pixels is not None
        ),
        default=None,
    )
    minimum_margin = min(
        (
            certificate.localized_certifying_margin
            for certificate in values
            if certificate.localized_certifying_margin is not None
        ),
        default=None,
    )
    placement_status_counts = {
        state: sum(
            case_by_id[certificate.case_id].multiplicity
            for certificate in values
            if certificate.case_status == state
        )
        for state in ("PASS", "STRUCTURAL_FAIL", "INCONCLUSIVE")
    }
    if sum(placement_status_counts.values()) != (
        catalog.concrete_placement_count
    ):
        raise ValueError("placement status aggregation is not exhaustive")
    core: dict[str, object] = {
        "schema_version": TINY_TARGET_DECISION_SCHEMA,
        "status": status,
        "catalog_fingerprint": catalog.catalog_fingerprint,
        "case_count": len(catalog.cases),
        "concrete_placement_count": catalog.concrete_placement_count,
        "case_status_counts": counts,
        "concrete_placement_status_counts": placement_status_counts,
        "failing_cases": list(failing_details),
        "failing_concrete_placement_count": placement_status_counts[
            "STRUCTURAL_FAIL"
        ],
        "inconclusive_case_ids": list(inconclusive),
        "maximum_irreducible_false_addition_pixels": maximum_false_additions,
        "minimum_localized_certifying_margin": minimum_margin,
        "localized_margin_is_free_grid_maximum_claim": False,
        "all_case_conjunction": status == "LATE_STATIC_GATE_PASS",
        "historical_pretraining_gate_satisfied": False,
        "late_compliance_audit": True,
        "historical_wave_a_decision_may_change": False,
        "current_paired_version_innovation_established": False,
        "training_authorized": False,
        "D_V_or_D_T_authorized": False,
        "full_cure_authorized": False,
        "cross_backbone_authorized": False,
    }
    core["decision_fingerprint"] = stable_fingerprint(core)
    return core


def replay_tiny_target_case_certificate(
    case: TinyTargetCase,
    certificate: TinyTargetCaseCertificate,
) -> None:
    """Re-run the bound solver and require the complete certificate to match."""

    verify_tiny_target_case_certificate(case, certificate)
    replayed = solve_tiny_target_case(case)
    if replayed.payload() != certificate.payload():
        raise ValueError(
            "certificate differs from an independent bound-solver replay"
        )


def _build_tiny_target_decision_without_solver_replay(
    catalog: TinyTargetCaseCatalog,
    certificates: Sequence[TinyTargetCaseCertificate],
) -> dict[str, object]:
    """Aggregate after full witness replay but without re-running the solver."""

    if not isinstance(catalog, TinyTargetCaseCatalog):
        raise TypeError("catalog must be TinyTargetCaseCatalog")
    values = tuple(certificates)
    if any(
        not isinstance(certificate, TinyTargetCaseCertificate)
        or certificate.certificate_fingerprint
        != stable_fingerprint(certificate.payload(unsigned=True))
        for certificate in values
    ):
        raise ValueError("certificate type or fingerprint is invalid")
    if len(values) != len(catalog.cases):
        raise ValueError("every exact case must have one certificate")
    if tuple(certificate.case_id for certificate in values) != tuple(
        case.case_id for case in catalog.cases
    ):
        raise ValueError("certificates are not in canonical catalog order")
    by_case = {certificate.case_id: certificate for certificate in values}
    if len(by_case) != len(values) or set(by_case) != {
        case.case_id for case in catalog.cases
    }:
        raise ValueError("certificate identities do not match the case catalog")
    for case, certificate in zip(catalog.cases, values, strict=True):
        verify_tiny_target_case_certificate(case, certificate)
    return _aggregate_tiny_target_case_statuses(catalog, values)


def build_tiny_target_decision(
    catalog: TinyTargetCaseCatalog,
    certificates: Sequence[TinyTargetCaseCertificate],
) -> dict[str, object]:
    """Re-solve every case, then aggregate the all-case conjunction."""

    decision = _build_tiny_target_decision_without_solver_replay(
        catalog,
        certificates,
    )
    for case, certificate in zip(
        catalog.cases,
        tuple(certificates),
        strict=True,
    ):
        replay_tiny_target_case_certificate(case, certificate)
    return decision


__all__ = [
    "AXIS_WEIGHT_DENOMINATOR",
    "BIG_M",
    "BILINEAR_WEIGHT_DENOMINATOR",
    "EXPECTED_CONCRETE_PLACEMENTS",
    "EXPECTED_SHAPE_COUNTS",
    "EXPECTED_TOTAL_SHAPES",
    "CERTIFYING_MARGIN_MIN",
    "DEFAULT_SOLVER_OPTIONS",
    "GAMMA_MAX",
    "MARGIN_OBJECTIVE_SCALE",
    "LOW_GRID_SIZE",
    "MAX_FALSE_ADDITION_PIXELS",
    "OUTPUT_GRID_SIZE",
    "VERIFY_TOLERANCE",
    "RepresentabilityProblem",
    "TinyTargetCase",
    "TinyTargetCaseCatalog",
    "TinyTargetCaseCertificate",
    "TinyTargetShape",
    "axis_bilinear_weights",
    "build_axis_equivalence_classes",
    "build_representability_problem",
    "build_tiny_target_case_catalog",
    "build_tiny_target_decision",
    "enumerate_tiny_target_shapes",
    "reconstruct_output_logits",
    "replay_tiny_target_case_certificate",
    "solve_tiny_target_case",
    "torch_bilinear_output_from_active_values",
    "verify_tiny_target_case_certificate",
]
