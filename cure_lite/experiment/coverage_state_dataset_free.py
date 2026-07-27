"""Dataset-free expanded gate for the scalar coverage-state level set.

The gate has two deliberately separate layers:

* a complete 16-case x 2-resolution x 3-seed geometry/representation matrix;
* a completion-root counterexample at 2 resolutions x 3 seeds;
* a small 3-objective x 3-seed, three-update computational matrix.

The second layer is only a code learnability and early-gradient check.  It is
not a performance experiment and it never reads D_R, D_V, or D_T.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..coverage_state_batches import (
    CoverageStateFusedBatch,
    CoverageStateNaturalTrainBatch,
    CoverageStatePairTrainBatch,
)
from ..coverage_state_level_set import (
    CSLF_FIELD_AMPLITUDE,
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
    normalize_cslf_feature,
    truncated_signed_distance_field,
)
from ..coverage_state_observability import (
    changed_feature_cells,
    occupancy_to_phase_grid,
    occupancy_to_scalar_grid,
    structural_output_support,
    target_response_support,
)
from ..coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
    CURELitePhasePreservingCoverageStateLevelSet,
    CoverageStatePhasePreservingConfig,
    pixel_unshuffle_bool_occupancy,
)
from ..coverage_state_sobolev import (
    CSLF_COMPLETION_ROOTED_RESPONSE_POLICY,
    CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
    CoverageStateAbsoluteTargets,
    CoverageStatePairLossFields,
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    coverage_state_added_target_support_from_targets,
    coverage_state_completion_rooted_pair_sobolev_loss_from_targets,
    coverage_state_pair_sobolev_loss_from_targets,
    coverage_state_support_oriented_pair_sobolev_loss_from_targets,
    prepare_coverage_state_focused_absolute_targets,
    prepare_coverage_state_pair_targets,
)
from ..train.coverage_state_fused_step import (
    COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES,
    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
    CoverageStatePairObjective,
    coverage_state_fused_train_step,
)
from ..paired_types import tensor_content_fingerprint
from .coverage_state_training import coverage_state_model_fingerprint


COVERAGE_STATE_DATASET_FREE_SCHEMA = (
    "cure-lite-cslf-expanded-dataset-free-gate-v2"
)
COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_SCHEMA = (
    "cure-lite-cslf-support-oriented-dataset-free-gate-v1"
)
COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SCHEMA = (
    "cure-lite-cslf-phase-preserving-dataset-free-gate-v1"
)
COVERAGE_STATE_LEGACY_DATASET_FREE_FINGERPRINT = (
    "44f85b45adc42eaefc79278d4c519aac40a5ed17034b6eb"
    "51576452ff4db935d"
)
COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_FINGERPRINT = (
    "56f56912359c5b12e10110323f01aeced279c1934f04080e5fe473f82c4d7c35"
)
COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT = (
    "3d0bf1c771966f04f96319bb3605d1aff90827843153ca8e89ba8965c5a79d2b"
)
COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS = (42, 43, 44)
COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SIZES = (64,)
COVERAGE_STATE_DATASET_FREE_CASES = (
    "one_pixel_target",
    "three_pixel_compact_target",
    "two_disconnected_targets",
    "target_at_image_edge",
    "target_near_invalid_barrier",
    "empty_state",
    "single_false_negative_island",
    "multiple_false_islands",
    "component_null_deletion",
    "identity_null",
    "same_feature_cell_multiple_occupancy_phases",
    "full_grid_deletion_hidden_by_scalar_projection",
    "clean_pair_with_natural_miss_already_present",
    "clutter_feature_peak_without_target",
    "low_rms_feature",
    "high_dynamic_range_feature",
)
COVERAGE_STATE_DATASET_FREE_SIZES = (64, 256)
COVERAGE_STATE_DATASET_FREE_SEEDS = (42, 43, 44)
COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES = 3
COVERAGE_STATE_DATASET_FREE_STRIDE = 4
COVERAGE_STATE_DATASET_FREE_FEATURE_CHANNELS = 2
COVERAGE_STATE_DATASET_FREE_WIDTH = 4
COVERAGE_STATE_COMPLETION_ROOT_OFFSET = (
    0.001
    + CSLF_FIELD_AMPLITUDE
    / float(COVERAGE_STATE_DATASET_FREE_STRIDE)
)
COVERAGE_STATE_COMPLETION_ROOT_RESPONSE_TOLERANCE = 1.0e-6
COVERAGE_STATE_COMPLETION_ROOT_LEGACY_GRADIENT_TOLERANCE = 1.0e-6
COVERAGE_STATE_COMPLETION_ROOT_GRADIENT_MINIMUM = 1.0e-4


@dataclass(frozen=True)
class CoverageStateDatasetFreeCaseResult:
    case_name: str
    size: int
    seed: int
    state_type: str
    pair_kind: str | None
    optimizer_eligible_pair: bool
    expected_scalar_hidden: bool
    target_pixels_plus: int
    target_pixels_minus: int
    target_zero_level_exact: bool
    target_fields_finite: bool
    model_fields_finite: bool
    hard_union_exact: bool
    scalar_visible: bool | None
    phase_visible: bool | None
    phase_roundtrip_exact: bool | None
    target_response_pixels: int
    target_response_outside_scalar_rf_pixels: int
    target_response_outside_phase_rf_pixels: int
    predicted_negative_pixels: int
    predicted_negative_components: int
    identity_field_exact: bool | None
    hidden_component_field_exact: bool | None
    component_new_negative_pixels: int | None
    component_new_negative_components: int | None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "size": self.size,
            "seed": self.seed,
            "state_type": self.state_type,
            "pair_kind": self.pair_kind,
            "optimizer_eligible_pair": self.optimizer_eligible_pair,
            "expected_scalar_hidden": self.expected_scalar_hidden,
            "target_pixels_plus": self.target_pixels_plus,
            "target_pixels_minus": self.target_pixels_minus,
            "target_zero_level_exact": self.target_zero_level_exact,
            "target_fields_finite": self.target_fields_finite,
            "model_fields_finite": self.model_fields_finite,
            "hard_union_exact": self.hard_union_exact,
            "scalar_visible": self.scalar_visible,
            "phase_visible": self.phase_visible,
            "phase_roundtrip_exact": self.phase_roundtrip_exact,
            "target_response_pixels": self.target_response_pixels,
            "target_response_outside_scalar_rf_pixels": (
                self.target_response_outside_scalar_rf_pixels
            ),
            "target_response_outside_phase_rf_pixels": (
                self.target_response_outside_phase_rf_pixels
            ),
            "predicted_negative_pixels": self.predicted_negative_pixels,
            "predicted_negative_components": (
                self.predicted_negative_components
            ),
            "identity_field_exact": self.identity_field_exact,
            "hidden_component_field_exact": (
                self.hidden_component_field_exact
            ),
            "component_new_negative_pixels": (
                self.component_new_negative_pixels
            ),
            "component_new_negative_components": (
                self.component_new_negative_components
            ),
        }


@dataclass(frozen=True)
class CoverageStateDatasetFreeTrainingResult:
    seed: int
    objective: str
    updates: int
    forward_calls: int
    backward_calls: int
    optimizer_steps: int
    logical_state_evaluations: int
    initial_model_fingerprint: str
    final_model_fingerprint: str
    selection_fingerprint: str
    first_nonzero_gradient_update: tuple[tuple[str, int], ...]
    factual_miss_target_pixels: int
    factual_no_miss_target_pixels: int
    losses_finite: bool
    parameters_changed: bool
    diagnostic_fields_finite: bool
    identity_field_exact: bool
    hidden_component_field_exact: bool
    component_new_negative_pixels: int
    component_new_negative_components: int
    empty_negative_pixels: int
    empty_negative_components: int
    hard_union_exact: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "objective": self.objective,
            "updates": self.updates,
            "forward_calls": self.forward_calls,
            "backward_calls": self.backward_calls,
            "optimizer_steps": self.optimizer_steps,
            "logical_state_evaluations": self.logical_state_evaluations,
            "initial_model_fingerprint": self.initial_model_fingerprint,
            "final_model_fingerprint": self.final_model_fingerprint,
            "selection_fingerprint": self.selection_fingerprint,
            "first_nonzero_gradient_update": {
                name: update
                for name, update in self.first_nonzero_gradient_update
            },
            "factual_miss_target_pixels": self.factual_miss_target_pixels,
            "factual_no_miss_target_pixels": (
                self.factual_no_miss_target_pixels
            ),
            "losses_finite": self.losses_finite,
            "parameters_changed": self.parameters_changed,
            "diagnostic_fields_finite": self.diagnostic_fields_finite,
            "identity_field_exact": self.identity_field_exact,
            "hidden_component_field_exact": (
                self.hidden_component_field_exact
            ),
            "component_new_negative_pixels": (
                self.component_new_negative_pixels
            ),
            "component_new_negative_components": (
                self.component_new_negative_components
            ),
            "empty_negative_pixels": self.empty_negative_pixels,
            "empty_negative_components": self.empty_negative_components,
            "hard_union_exact": self.hard_union_exact,
        }


@dataclass(frozen=True)
class CoverageStateCompletionRootProbeResult:
    """Raw field/gradient evidence for the completion-root correction."""

    size: int
    seed: int
    target_pixels: int
    target_negative_pixels_before_update: int
    response_sign_correct_pixels: int
    response_sign_pixels: int
    response_error_max_hex: str
    legacy_minus_gradient_max_hex: str
    rooted_minus_target_gradient_min_hex: str
    exact_rooted_loss_hex: str
    component_null_rooted_loss_hex: str
    objective_policy: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "size": self.size,
            "seed": self.seed,
            "target_pixels": self.target_pixels,
            "target_negative_pixels_before_update": (
                self.target_negative_pixels_before_update
            ),
            "response_sign_correct_pixels": (
                self.response_sign_correct_pixels
            ),
            "response_sign_pixels": self.response_sign_pixels,
            "response_error_max_hex": self.response_error_max_hex,
            "legacy_minus_gradient_max_hex": (
                self.legacy_minus_gradient_max_hex
            ),
            "rooted_minus_target_gradient_min_hex": (
                self.rooted_minus_target_gradient_min_hex
            ),
            "exact_rooted_loss_hex": self.exact_rooted_loss_hex,
            "component_null_rooted_loss_hex": (
                self.component_null_rooted_loss_hex
            ),
            "objective_policy": self.objective_policy,
        }


@dataclass(frozen=True)
class CoverageStateDatasetFreeReceipt:
    case_results: tuple[CoverageStateDatasetFreeCaseResult, ...]
    training_results: tuple[CoverageStateDatasetFreeTrainingResult, ...]
    completion_root_probes: tuple[
        CoverageStateCompletionRootProbeResult,
        ...,
    ]
    checks: tuple[tuple[str, bool], ...]
    D_R_accessed: bool = False
    D_V_accessed: bool = False
    D_T_accessed: bool = False
    performance_claim_supported: bool = False

    def __post_init__(self) -> None:
        self.verify()

    def verify(self) -> None:
        """Fail closed on incomplete evidence, scope drift, or check drift."""

        if (
            not isinstance(self.case_results, tuple)
            or not isinstance(self.training_results, tuple)
            or not isinstance(self.completion_root_probes, tuple)
            or not isinstance(self.checks, tuple)
            or not all(
                isinstance(value, CoverageStateDatasetFreeCaseResult)
                for value in self.case_results
            )
            or not all(
                isinstance(value, CoverageStateDatasetFreeTrainingResult)
                for value in self.training_results
            )
            or not all(
                isinstance(value, CoverageStateCompletionRootProbeResult)
                for value in self.completion_root_probes
            )
            or not all(
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], str)
                and type(item[1]) is bool
                for item in self.checks
            )
        ):
            raise ValueError("dataset-free receipt matrix is incomplete")
        expected_cases = (
            len(COVERAGE_STATE_DATASET_FREE_CASES)
            * len(COVERAGE_STATE_DATASET_FREE_SIZES)
            * len(COVERAGE_STATE_DATASET_FREE_SEEDS)
        )
        expected_training = (
            len(COVERAGE_STATE_DATASET_FREE_SEEDS)
            * len(COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES)
        )
        expected_probes = (
            len(COVERAGE_STATE_DATASET_FREE_SIZES)
            * len(COVERAGE_STATE_DATASET_FREE_SEEDS)
        )
        expected_case_keys = {
            (name, size, seed)
            for name in COVERAGE_STATE_DATASET_FREE_CASES
            for size in COVERAGE_STATE_DATASET_FREE_SIZES
            for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
        }
        actual_case_keys = {
            (value.case_name, value.size, value.seed)
            for value in self.case_results
        }
        expected_training_keys = {
            (seed, objective.value)
            for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
            for objective in (
                COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES
            )
        }
        actual_training_keys = {
            (value.seed, value.objective)
            for value in self.training_results
        }
        expected_probe_keys = {
            (size, seed)
            for size in COVERAGE_STATE_DATASET_FREE_SIZES
            for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
        }
        actual_probe_keys = {
            (value.size, value.seed)
            for value in self.completion_root_probes
        }
        if (
            len(self.case_results) != expected_cases
            or len(self.training_results) != expected_training
            or actual_case_keys != expected_case_keys
            or actual_training_keys != expected_training_keys
            or len(self.completion_root_probes) != expected_probes
            or actual_probe_keys != expected_probe_keys
            or not self.checks
            or len({name for name, _ in self.checks}) != len(self.checks)
        ):
            raise ValueError("dataset-free receipt matrix is incomplete")
        if (
            self.D_R_accessed
            or self.D_V_accessed
            or self.D_T_accessed
            or self.performance_claim_supported
        ):
            raise ValueError("dataset-free receipt exceeds its evidence scope")
        recomputed = recompute_coverage_state_dataset_free_checks(
            self.case_results,
            self.training_results,
            self.completion_root_probes,
        )
        if self.checks != recomputed:
            raise ValueError(
                "dataset-free receipt checks do not match recomputed results"
            )

    @property
    def all_pass(self) -> bool:
        self.verify()
        return all(passed for _, passed in self.checks)

    @property
    def status(self) -> str:
        return (
            "DATASET_FREE_GATE_PASS"
            if self.all_pass
            else "DATASET_FREE_GATE_FAIL"
        )

    def canonical_payload(self) -> dict[str, object]:
        self.verify()
        return {
            "schema_version": COVERAGE_STATE_DATASET_FREE_SCHEMA,
            "scope": {
                "case_names": list(COVERAGE_STATE_DATASET_FREE_CASES),
                "sizes": list(COVERAGE_STATE_DATASET_FREE_SIZES),
                "seeds": list(COVERAGE_STATE_DATASET_FREE_SEEDS),
                "geometry_case_count": len(self.case_results),
                "training_updates_per_objective": (
                    COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                ),
                "training_result_count": len(self.training_results),
                "completion_root_probe_count": len(
                    self.completion_root_probes
                ),
                "objective_suite": [
                    value.value
                    for value in (
                        COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES
                    )
                ],
                "training_role": (
                    "computational_learnability_and_early_gradient_only"
                ),
                "fixture_source": "generated_in_memory_only",
                "selected_representation": "scalar_max",
            },
            "case_results": [
                value.canonical_payload() for value in self.case_results
            ],
            "training_results": [
                value.canonical_payload() for value in self.training_results
            ],
            "completion_root_probes": [
                value.canonical_payload()
                for value in self.completion_root_probes
            ],
            "checks": {name: passed for name, passed in self.checks},
            "status": self.status,
            "all_pass": self.all_pass,
            "data_access": {
                "D_R_accessed": self.D_R_accessed,
                "D_V_accessed": self.D_V_accessed,
                "D_T_accessed": self.D_T_accessed,
            },
            "performance_claim_supported": self.performance_claim_supported,
            "formal_training_authorized": False,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


@dataclass(frozen=True)
class _DatasetFreeCase:
    name: str
    feature: Tensor
    occupancy_plus: Tensor
    occupancy_minus: Tensor
    target_plus: Tensor
    target_minus: Tensor
    valid_mask: Tensor
    pair_kind: str | None = None
    optimizer_eligible_pair: bool = False
    expected_scalar_hidden: bool = False

    @property
    def is_pair(self) -> bool:
        return self.pair_kind is not None


def _empty_mask(size: int) -> Tensor:
    return torch.zeros((1, 1, size, size), dtype=torch.bool)


def _put_pixels(mask: Tensor, coordinates: tuple[tuple[int, int], ...]) -> None:
    for row, column in coordinates:
        mask[0, 0, row, column] = True


def _put_box(
    mask: Tensor,
    *,
    row: int,
    column: int,
    height: int,
    width: int,
) -> None:
    mask[0, 0, row : row + height, column : column + width] = True


def _feature(case_name: str, *, size: int, seed: int) -> Tensor:
    feature_size = size // COVERAGE_STATE_DATASET_FREE_STRIDE
    grid = torch.arange(
        feature_size * feature_size,
        dtype=torch.float32,
    ).reshape(1, 1, feature_size, feature_size)
    offset = float(
        (
            seed
            + 17 * COVERAGE_STATE_DATASET_FREE_CASES.index(case_name)
        )
        % 29
    )
    first = ((grid + offset) % 23.0) / 23.0 + 0.05
    second = ((grid * 3.0 + offset + 1.0) % 31.0) / 31.0 + 0.05
    result = torch.cat((first, second), dim=1).contiguous()
    if case_name == "clutter_feature_peak_without_target":
        result = result.clone()
        result[0, 0, feature_size // 2, feature_size // 2] = 100.0
    elif case_name == "low_rms_feature":
        result = result * 1.0e-8
    elif case_name == "high_dynamic_range_feature":
        result = torch.full_like(result, 1.0e-4)
        result[0, 0, feature_size // 2, feature_size // 2] = 1.0e4
        result[0, 1, feature_size // 3, feature_size // 3] = -1.0e4
    return result.to(dtype=torch.float32).contiguous()


def _make_case(case_name: str, *, size: int, seed: int) -> _DatasetFreeCase:
    if case_name not in COVERAGE_STATE_DATASET_FREE_CASES:
        raise ValueError("unknown dataset-free case")
    feature = _feature(case_name, size=size, seed=seed)
    plus = _empty_mask(size)
    minus = _empty_mask(size)
    target_plus = _empty_mask(size)
    target_minus = _empty_mask(size)
    valid = torch.ones_like(plus)
    center = size // 2
    cell_row = (center // COVERAGE_STATE_DATASET_FREE_STRIDE) * (
        COVERAGE_STATE_DATASET_FREE_STRIDE
    )
    cell_column = cell_row
    pair_kind: str | None = None
    optimizer_eligible = False
    expected_hidden = False

    if case_name == "one_pixel_target":
        _put_pixels(target_plus, ((center, center),))
    elif case_name == "three_pixel_compact_target":
        _put_pixels(
            target_plus,
            ((center, center), (center, center + 1), (center + 1, center)),
        )
    elif case_name == "two_disconnected_targets":
        _put_pixels(
            target_plus,
            ((size // 3, size // 3), (2 * size // 3, 2 * size // 3)),
        )
    elif case_name == "target_at_image_edge":
        _put_pixels(target_plus, ((0, 0), (0, 1)))
    elif case_name == "target_near_invalid_barrier":
        valid[:, :, :, center] = False
        _put_pixels(target_plus, ((center, center - 1),))
    elif case_name == "single_false_negative_island":
        _put_box(
            target_plus,
            row=center - 1,
            column=center - 1,
            height=2,
            width=2,
        )
    elif case_name == "multiple_false_islands":
        _put_box(
            target_plus,
            row=size // 4,
            column=size // 4,
            height=2,
            width=2,
        )
        _put_box(
            target_plus,
            row=3 * size // 4,
            column=3 * size // 4,
            height=2,
            width=2,
        )
    elif case_name == "component_null_deletion":
        pair_kind = "component_null"
        optimizer_eligible = True
        _put_box(
            plus,
            row=cell_row,
            column=cell_column,
            height=COVERAGE_STATE_DATASET_FREE_STRIDE,
            width=COVERAGE_STATE_DATASET_FREE_STRIDE,
        )
    elif case_name == "identity_null":
        pair_kind = "identity_null"
        _put_box(
            plus,
            row=cell_row,
            column=cell_column,
            height=2,
            width=2,
        )
        minus = plus.clone()
    elif case_name == "same_feature_cell_multiple_occupancy_phases":
        pair_kind = "component_null"
        expected_hidden = True
        _put_pixels(
            plus,
            (
                (cell_row, cell_column),
                (cell_row, cell_column + 1),
            ),
        )
        _put_pixels(minus, ((cell_row, cell_column),))
    elif case_name == "full_grid_deletion_hidden_by_scalar_projection":
        pair_kind = "component_null"
        expected_hidden = True
        _put_box(
            plus,
            row=cell_row,
            column=cell_column,
            height=COVERAGE_STATE_DATASET_FREE_STRIDE,
            width=COVERAGE_STATE_DATASET_FREE_STRIDE,
        )
        _put_pixels(minus, ((cell_row, cell_column),))
    elif case_name == "clean_pair_with_natural_miss_already_present":
        pair_kind = "clean_positive"
        optimizer_eligible = True
        _put_box(
            plus,
            row=cell_row,
            column=cell_column,
            height=COVERAGE_STATE_DATASET_FREE_STRIDE,
            width=COVERAGE_STATE_DATASET_FREE_STRIDE,
        )
        _put_pixels(target_plus, ((size // 4, size // 4),))
        target_minus = target_plus.clone()
        _put_box(
            target_minus,
            row=cell_row,
            column=cell_column,
            height=COVERAGE_STATE_DATASET_FREE_STRIDE,
            width=COVERAGE_STATE_DATASET_FREE_STRIDE,
        )
    elif case_name in {
        "low_rms_feature",
        "high_dynamic_range_feature",
    }:
        _put_pixels(target_plus, ((center, center),))
    elif case_name in {
        "empty_state",
        "clutter_feature_peak_without_target",
    }:
        pass
    else:
        raise AssertionError("dataset-free case construction is incomplete")

    if pair_kind is None:
        target_minus = target_plus.clone()
        minus = plus.clone()
    return _DatasetFreeCase(
        name=case_name,
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        target_plus=target_plus,
        target_minus=target_minus,
        valid_mask=valid,
        pair_kind=pair_kind,
        optimizer_eligible_pair=optimizer_eligible,
        expected_scalar_hidden=expected_hidden,
    )


def _target_zero_exact(field: Tensor, target: Tensor, valid: Tensor) -> bool:
    return torch.equal((field < 0.0) & valid, target)


def _component_count(value: Tensor) -> int:
    if (
        value.dtype != torch.bool
        or value.device.type != "cpu"
        or tuple(value.shape[:2]) != (1, 1)
    ):
        raise ValueError("component counter requires CPU bool [1,1,H,W]")
    mask = value[0, 0]
    height, width = (int(item) for item in mask.shape)
    visited = torch.zeros_like(mask)
    count = 0
    for row, column in torch.nonzero(mask, as_tuple=False).tolist():
        if bool(visited[row, column]):
            continue
        count += 1
        queue: deque[tuple[int, int]] = deque([(row, column)])
        visited[row, column] = True
        while queue:
            current_row, current_column = queue.popleft()
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    next_row = current_row + row_offset
                    next_column = current_column + column_offset
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and bool(mask[next_row, next_column])
                        and not bool(visited[next_row, next_column])
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
    return count


def _new_component_count(plus: Tensor, minus: Tensor) -> int:
    """Count minus components with no overlap with a plus component."""

    if plus.shape != minus.shape or plus.dtype != torch.bool:
        raise ValueError("completion endpoints must be aligned bool masks")
    mask = minus[0, 0]
    plus_mask = plus[0, 0]
    height, width = (int(item) for item in mask.shape)
    visited = torch.zeros_like(mask)
    new_count = 0
    for row, column in torch.nonzero(mask, as_tuple=False).tolist():
        if bool(visited[row, column]):
            continue
        overlaps_plus = False
        queue: deque[tuple[int, int]] = deque([(row, column)])
        visited[row, column] = True
        while queue:
            current_row, current_column = queue.popleft()
            overlaps_plus = overlaps_plus or bool(
                plus_mask[current_row, current_column]
            )
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    next_row = current_row + row_offset
                    next_column = current_column + column_offset
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and bool(mask[next_row, next_column])
                        and not bool(visited[next_row, next_column])
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
        if not overlaps_plus:
            new_count += 1
    return new_count


def _hard_union_exact(
    model: CURELiteCoverageStateLevelSet,
    feature: Tensor,
    occupancy: Tensor,
) -> bool:
    completion = model.predict_completion(feature, occupancy)
    union = model.predict_union(feature, occupancy)
    return (
        torch.equal(union, occupancy | completion)
        and not bool(torch.any(completion & occupancy))
    )


def _evaluate_case(
    case: _DatasetFreeCase,
    *,
    model: CURELiteCoverageStateLevelSet,
    config: CoverageStateSobolevConfig,
    size: int,
    seed: int,
) -> CoverageStateDatasetFreeCaseResult:
    field_plus_target = truncated_signed_distance_field(
        case.target_plus,
        case.valid_mask,
        radius=config.truncation_radius,
    )
    field_minus_target = truncated_signed_distance_field(
        case.target_minus,
        case.valid_mask,
        radius=config.truncation_radius,
    )
    target_finite = bool(
        torch.stack(
            (
                torch.isfinite(field_plus_target).all(),
                torch.isfinite(field_minus_target).all(),
                torch.isfinite(normalize_cslf_feature(case.feature)).all(),
            )
        ).all()
    )
    zero_exact = _target_zero_exact(
        field_plus_target,
        case.target_plus,
        case.valid_mask,
    ) and _target_zero_exact(
        field_minus_target,
        case.target_minus,
        case.valid_mask,
    )
    with torch.no_grad():
        feature = (
            torch.cat((case.feature, case.feature), dim=0)
            if case.is_pair
            else case.feature
        )
        occupancy = (
            torch.cat(
                (case.occupancy_plus, case.occupancy_minus),
                dim=0,
            )
            if case.is_pair
            else case.occupancy_plus
        )
        predicted = model(feature, occupancy)
        predicted_plus = predicted[:1]
        predicted_minus = predicted[1:] if case.is_pair else predicted[:1]
        model_finite = bool(torch.isfinite(predicted).all())
        predicted_completion = (
            (predicted_plus < 0.0)
            & ~case.occupancy_plus
            & case.valid_mask
        )
        hard_union = _hard_union_exact(
            model,
            case.feature,
            case.occupancy_plus,
        )

    scalar_visible: bool | None = None
    phase_visible: bool | None = None
    phase_roundtrip: bool | None = None
    response_pixels = 0
    outside_scalar_pixels = 0
    outside_phase_pixels = 0
    identity_exact: bool | None = None
    hidden_exact: bool | None = None
    component_new_pixels: int | None = None
    component_new_components: int | None = None
    if case.is_pair:
        feature_size = tuple(int(value) for value in case.feature.shape[-2:])
        scalar_plus = occupancy_to_scalar_grid(
            case.occupancy_plus,
            feature_size=feature_size,
        )
        scalar_minus = occupancy_to_scalar_grid(
            case.occupancy_minus,
            feature_size=feature_size,
        )
        phase_plus = occupancy_to_phase_grid(
            case.occupancy_plus,
            stride=COVERAGE_STATE_DATASET_FREE_STRIDE,
        )
        phase_minus = occupancy_to_phase_grid(
            case.occupancy_minus,
            stride=COVERAGE_STATE_DATASET_FREE_STRIDE,
        )
        scalar_changes = changed_feature_cells(
            scalar_plus,
            scalar_minus,
        )
        phase_changes = changed_feature_cells(phase_plus, phase_minus)
        scalar_visible = bool(torch.any(scalar_changes))
        phase_visible = bool(torch.any(phase_changes))
        phase_roundtrip = (
            torch.equal(
                torch.nn.functional.pixel_shuffle(
                    phase_plus.to(torch.float32),
                    COVERAGE_STATE_DATASET_FREE_STRIDE,
                ).to(torch.bool),
                case.occupancy_plus,
            )
            and torch.equal(
                torch.nn.functional.pixel_shuffle(
                    phase_minus.to(torch.float32),
                    COVERAGE_STATE_DATASET_FREE_STRIDE,
                ).to(torch.bool),
                case.occupancy_minus,
            )
        )
        response = target_response_support(
            field_plus_target,
            field_minus_target,
            case.valid_mask,
        )
        response_pixels = int(torch.count_nonzero(response))
        receptive = structural_output_support(
            scalar_changes,
            stride=COVERAGE_STATE_DATASET_FREE_STRIDE,
        )
        phase_receptive = structural_output_support(
            phase_changes,
            stride=COVERAGE_STATE_DATASET_FREE_STRIDE,
        )
        outside_scalar_pixels = int(
            torch.count_nonzero(response & ~receptive)
        )
        outside_phase_pixels = int(
            torch.count_nonzero(response & ~phase_receptive)
        )
        if case.pair_kind == "identity_null":
            identity_exact = torch.equal(predicted_plus, predicted_minus)
        if case.expected_scalar_hidden:
            hidden_exact = torch.equal(predicted_plus, predicted_minus)
        if case.pair_kind == "component_null":
            completion_plus = (
                (predicted_plus < 0.0)
                & ~case.occupancy_plus
                & case.valid_mask
            )
            completion_minus = (
                (predicted_minus < 0.0)
                & ~case.occupancy_minus
                & case.valid_mask
            )
            removed = case.occupancy_plus & ~case.occupancy_minus
            component_new_pixels = int(
                torch.count_nonzero(
                    completion_minus & ~completion_plus & removed
                )
            )
            component_new_components = _new_component_count(
                completion_plus,
                completion_minus,
            )

    return CoverageStateDatasetFreeCaseResult(
        case_name=case.name,
        size=size,
        seed=seed,
        state_type="pair" if case.is_pair else "absolute",
        pair_kind=case.pair_kind,
        optimizer_eligible_pair=case.optimizer_eligible_pair,
        expected_scalar_hidden=case.expected_scalar_hidden,
        target_pixels_plus=int(torch.count_nonzero(case.target_plus)),
        target_pixels_minus=int(torch.count_nonzero(case.target_minus)),
        target_zero_level_exact=zero_exact,
        target_fields_finite=target_finite,
        model_fields_finite=model_finite,
        hard_union_exact=hard_union,
        scalar_visible=scalar_visible,
        phase_visible=phase_visible,
        phase_roundtrip_exact=phase_roundtrip,
        target_response_pixels=response_pixels,
        target_response_outside_scalar_rf_pixels=outside_scalar_pixels,
        target_response_outside_phase_rf_pixels=outside_phase_pixels,
        predicted_negative_pixels=int(
            torch.count_nonzero(predicted_completion)
        ),
        predicted_negative_components=_component_count(
            predicted_completion
        ),
        identity_field_exact=identity_exact,
        hidden_component_field_exact=hidden_exact,
        component_new_negative_pixels=component_new_pixels,
        component_new_negative_components=component_new_components,
    )


def _stack_absolute(
    values: tuple[CoverageStateAbsoluteTargets, ...],
) -> CoverageStateAbsoluteTargets:
    result = CoverageStateAbsoluteTargets(
        target_field=torch.cat(
            tuple(value.target_field for value in values),
            dim=0,
        ).contiguous(),
        integration_measure=torch.cat(
            tuple(value.integration_measure for value in values),
            dim=0,
        ).contiguous(),
        field_valid_mask=torch.cat(
            tuple(value.field_valid_mask for value in values),
            dim=0,
        ).contiguous(),
        loss_valid_mask=torch.cat(
            tuple(value.loss_valid_mask for value in values),
            dim=0,
        ).contiguous(),
        focus_support=torch.cat(
            tuple(value.focus_support for value in values),
            dim=0,
        ).contiguous(),
        focus_support_field=torch.cat(
            tuple(value.focus_support_field for value in values),
            dim=0,
        ).contiguous(),
    )
    result.validate()
    return result


def _stack_pair(
    values: tuple[CoverageStatePairTargets, ...],
) -> CoverageStatePairTargets:
    result = CoverageStatePairTargets(
        target_field_plus=torch.cat(
            tuple(value.target_field_plus for value in values),
            dim=0,
        ).contiguous(),
        target_field_minus=torch.cat(
            tuple(value.target_field_minus for value in values),
            dim=0,
        ).contiguous(),
        focus_support=torch.cat(
            tuple(value.focus_support for value in values),
            dim=0,
        ).contiguous(),
        focus_support_field=torch.cat(
            tuple(value.focus_support_field for value in values),
            dim=0,
        ).contiguous(),
        integration_measure=torch.cat(
            tuple(value.integration_measure for value in values),
            dim=0,
        ).contiguous(),
        valid_mask=torch.cat(
            tuple(value.valid_mask for value in values),
            dim=0,
        ).contiguous(),
    )
    result.validate()
    return result


def _training_batch(seed: int) -> CoverageStateFusedBatch:
    size = 64
    config = CoverageStateSobolevConfig(
        truncation_radius=COVERAGE_STATE_DATASET_FREE_STRIDE
    )
    miss_cases = (
        "one_pixel_target",
        "three_pixel_compact_target",
        "two_disconnected_targets",
        "target_at_image_edge",
    )
    no_cases = (
        "empty_state",
        "clutter_feature_peak_without_target",
        "low_rms_feature",
        "high_dynamic_range_feature",
    )
    miss = tuple(
        _make_case(name, size=size, seed=seed) for name in miss_cases
    )
    no_miss = tuple(
        replace(
            _make_case(name, size=size, seed=seed),
            target_plus=_empty_mask(size),
            target_minus=_empty_mask(size),
        )
        for name in no_cases
    )
    if (
        any(not bool(torch.any(value.target_plus)) for value in miss)
        or any(bool(torch.any(value.target_plus)) for value in no_miss)
    ):
        raise AssertionError("dataset-free natural roles are not disjoint")
    miss_targets = tuple(
        prepare_coverage_state_focused_absolute_targets(
            value.target_plus,
            value.valid_mask,
            value.valid_mask & ~value.occupancy_plus,
            config=config,
        )
        for value in miss
    )
    no_targets = tuple(
        prepare_coverage_state_focused_absolute_targets(
            value.target_plus,
            value.valid_mask,
            value.valid_mask & ~value.occupancy_plus,
            config=config,
        )
        for value in no_miss
    )
    clean = _make_case(
        "clean_pair_with_natural_miss_already_present",
        size=size,
        seed=seed,
    )
    component = _make_case(
        "component_null_deletion",
        size=size,
        seed=seed,
    )
    pair_values = (clean, component)
    joint = tuple(
        prepare_coverage_state_pair_targets(
            value.occupancy_plus,
            value.occupancy_minus,
            value.target_plus,
            value.target_minus,
            value.valid_mask,
            config=config,
        )
        for value in pair_values
    )
    absolute_plus = tuple(
        prepare_coverage_state_focused_absolute_targets(
            value.target_plus,
            value.valid_mask,
            value.valid_mask & ~value.occupancy_plus,
            config=config,
        )
        for value in pair_values
    )
    absolute_minus = tuple(
        prepare_coverage_state_focused_absolute_targets(
            value.target_minus,
            value.valid_mask,
            value.valid_mask & ~value.occupancy_minus,
            config=config,
        )
        for value in pair_values
    )
    result = CoverageStateFusedBatch(
        factual_miss=CoverageStateNaturalTrainBatch(
            feature=torch.cat(tuple(value.feature for value in miss), dim=0),
            occupancy=torch.cat(
                tuple(value.occupancy_plus for value in miss),
                dim=0,
            ),
            targets=_stack_absolute(miss_targets),
            record_ids=tuple(f"df-miss-{seed}-{index}" for index in range(4)),
            sample_ids=tuple(
                f"df-miss-source-{seed}-{index}" for index in range(4)
            ),
            actual_input_fingerprints=tuple(
                stable_fingerprint(
                    {"kind": "miss", "seed": seed, "index": index}
                )
                for index in range(4)
            ),
            state_kind="factual_miss",
        ),
        factual_no_miss=CoverageStateNaturalTrainBatch(
            feature=torch.cat(
                tuple(value.feature for value in no_miss),
                dim=0,
            ),
            occupancy=torch.cat(
                tuple(value.occupancy_plus for value in no_miss),
                dim=0,
            ),
            targets=_stack_absolute(no_targets),
            record_ids=tuple(f"df-no-{seed}-{index}" for index in range(4)),
            sample_ids=tuple(
                f"df-no-source-{seed}-{index}" for index in range(4)
            ),
            actual_input_fingerprints=tuple(
                stable_fingerprint(
                    {"kind": "no_miss", "seed": seed, "index": index}
                )
                for index in range(4)
            ),
            state_kind="factual_no_miss",
        ),
        pairs=CoverageStatePairTrainBatch(
            feature=torch.cat(
                tuple(value.feature for value in pair_values),
                dim=0,
            ),
            occupancy_plus=torch.cat(
                tuple(value.occupancy_plus for value in pair_values),
                dim=0,
            ),
            occupancy_minus=torch.cat(
                tuple(value.occupancy_minus for value in pair_values),
                dim=0,
            ),
            joint_targets=_stack_pair(joint),
            absolute_targets_plus=_stack_absolute(absolute_plus),
            absolute_targets_minus=_stack_absolute(absolute_minus),
            pair_ids=(
                f"df-clean-{seed}",
                f"df-component-{seed}",
            ),
            pair_kinds=("clean_positive", "component_null"),
            sample_ids=(
                f"df-clean-source-{seed}",
                f"df-component-source-{seed}",
            ),
            actual_input_plus_fingerprints=(
                stable_fingerprint(
                    {"kind": "clean_plus", "seed": seed}
                ),
                stable_fingerprint(
                    {"kind": "component_plus", "seed": seed}
                ),
            ),
            actual_input_minus_fingerprints=(
                stable_fingerprint(
                    {"kind": "clean_minus", "seed": seed}
                ),
                stable_fingerprint(
                    {"kind": "component_minus", "seed": seed}
                ),
            ),
        ),
    )
    result.validate()
    return result


def _post_training_diagnostics(
    model: CURELiteCoverageStateLevelSet,
    *,
    seed: int,
) -> dict[str, object]:
    size = 64
    identity = _make_case("identity_null", size=size, seed=seed)
    hidden = _make_case(
        "full_grid_deletion_hidden_by_scalar_projection",
        size=size,
        seed=seed,
    )
    component = _make_case(
        "component_null_deletion",
        size=size,
        seed=seed,
    )
    empty = _make_case("empty_state", size=size, seed=seed)
    with torch.no_grad():
        identity_field = model(
            torch.cat((identity.feature, identity.feature), dim=0),
            torch.cat(
                (identity.occupancy_plus, identity.occupancy_minus),
                dim=0,
            ),
        )
        hidden_field = model(
            torch.cat((hidden.feature, hidden.feature), dim=0),
            torch.cat(
                (hidden.occupancy_plus, hidden.occupancy_minus),
                dim=0,
            ),
        )
        component_field = model(
            torch.cat((component.feature, component.feature), dim=0),
            torch.cat(
                (
                    component.occupancy_plus,
                    component.occupancy_minus,
                ),
                dim=0,
            ),
        )
        component_plus = (
            (component_field[:1] < 0.0)
            & ~component.occupancy_plus
            & component.valid_mask
        )
        component_minus = (
            (component_field[1:] < 0.0)
            & ~component.occupancy_minus
            & component.valid_mask
        )
        removed = component.occupancy_plus & ~component.occupancy_minus
        empty_field = model(empty.feature, empty.occupancy_plus)
        empty_completion = (
            (empty_field < 0.0)
            & ~empty.occupancy_plus
            & empty.valid_mask
        )
        hard_union = _hard_union_exact(
            model,
            empty.feature,
            empty.occupancy_plus,
        )
    return {
        "diagnostic_fields_finite": bool(
            torch.isfinite(identity_field).all()
            and torch.isfinite(hidden_field).all()
            and torch.isfinite(component_field).all()
            and torch.isfinite(empty_field).all()
        ),
        "identity_field_exact": torch.equal(
            identity_field[:1],
            identity_field[1:],
        ),
        "hidden_component_field_exact": torch.equal(
            hidden_field[:1],
            hidden_field[1:],
        ),
        "component_new_negative_pixels": int(
            torch.count_nonzero(
                component_minus & ~component_plus & removed
            )
        ),
        "component_new_negative_components": _new_component_count(
            component_plus,
            component_minus,
        ),
        "empty_negative_pixels": int(
            torch.count_nonzero(empty_completion)
        ),
        "empty_negative_components": _component_count(empty_completion),
        "hard_union_exact": hard_union,
    }


def _run_completion_root_probe(
    *,
    size: int,
    seed: int,
) -> CoverageStateCompletionRootProbeResult:
    """Recompute the response-correct/no-crossing counterexample."""

    if size not in COVERAGE_STATE_DATASET_FREE_SIZES:
        raise ValueError("completion-root probe size is not frozen")
    if seed not in COVERAGE_STATE_DATASET_FREE_SEEDS:
        raise ValueError("completion-root probe seed is not frozen")
    valid = torch.ones(1, 1, size, size, dtype=torch.bool)
    target_plus = torch.zeros_like(valid)
    target_minus = torch.zeros_like(valid)
    offset_y = (seed - COVERAGE_STATE_DATASET_FREE_SEEDS[0]) % 3
    offset_x = (2 * offset_y) % 3
    target_minus[
        ...,
        size // 2 - 1 + offset_y,
        size // 2 - 1 + offset_x,
    ] = True
    occupancy_plus = target_minus.clone()
    occupancy_minus = torch.zeros_like(valid)
    config = CoverageStateSobolevConfig(
        truncation_radius=COVERAGE_STATE_DATASET_FREE_STRIDE,
    )
    targets = prepare_coverage_state_pair_targets(
        occupancy_plus,
        occupancy_minus,
        target_plus,
        target_minus,
        valid,
        config=config,
    )
    legacy_plus = (
        targets.target_field_plus
        + COVERAGE_STATE_COMPLETION_ROOT_OFFSET
    ).detach().requires_grad_()
    legacy_minus = (
        targets.target_field_minus
        + COVERAGE_STATE_COMPLETION_ROOT_OFFSET
    ).detach().requires_grad_()
    rooted_plus = legacy_plus.detach().clone().requires_grad_()
    rooted_minus = legacy_minus.detach().clone().requires_grad_()
    legacy = coverage_state_pair_sobolev_loss_from_targets(
        legacy_plus,
        legacy_minus,
        targets,
        config=config,
    )
    rooted = (
        coverage_state_completion_rooted_pair_sobolev_loss_from_targets(
            rooted_plus,
            rooted_minus,
            targets,
            config=config,
        )
    )
    legacy.loss.backward()
    rooted.loss.backward()
    if (
        legacy_minus.grad is None
        or rooted_minus.grad is None
        or not bool(torch.isfinite(legacy_minus.grad).all())
        or not bool(torch.isfinite(rooted_minus.grad).all())
    ):
        raise FloatingPointError(
            "completion-root probe produced an invalid gradient"
        )
    target_response = (
        targets.target_field_minus - targets.target_field_plus
    )
    response_support = target_response.ne(0.0)
    predicted_response = rooted_minus - rooted_plus
    response_correct = (
        predicted_response[response_support]
        * target_response[response_support]
    ) > 0.0
    exact = (
        coverage_state_completion_rooted_pair_sobolev_loss_from_targets(
            targets.target_field_plus,
            targets.target_field_minus,
            targets,
            config=config,
        )
    )
    null_targets = prepare_coverage_state_pair_targets(
        occupancy_plus,
        occupancy_minus,
        target_plus,
        target_plus,
        valid,
        config=config,
    )
    component_null = (
        coverage_state_completion_rooted_pair_sobolev_loss_from_targets(
            null_targets.target_field_plus,
            null_targets.target_field_minus,
            null_targets,
            config=config,
        )
    )
    return CoverageStateCompletionRootProbeResult(
        size=size,
        seed=seed,
        target_pixels=int(torch.count_nonzero(target_minus)),
        target_negative_pixels_before_update=int(
            torch.count_nonzero(rooted_minus[target_minus] < 0.0)
        ),
        response_sign_correct_pixels=int(
            torch.count_nonzero(response_correct)
        ),
        response_sign_pixels=int(torch.count_nonzero(response_support)),
        response_error_max_hex=float(
            rooted.response_error.abs().max().item()
        ).hex(),
        legacy_minus_gradient_max_hex=float(
            legacy_minus.grad.abs().max().item()
        ).hex(),
        rooted_minus_target_gradient_min_hex=float(
            rooted_minus.grad[target_minus].min().item()
        ).hex(),
        exact_rooted_loss_hex=float(exact.loss.item()).hex(),
        component_null_rooted_loss_hex=float(
            component_null.loss.item()
        ).hex(),
        objective_policy=CSLF_COMPLETION_ROOTED_RESPONSE_POLICY,
    )


def _run_training_matrix(
    seed: int,
    *,
    objectives: tuple[
        CoverageStatePairObjective,
        CoverageStatePairObjective,
        CoverageStatePairObjective,
    ] = COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES,
) -> tuple[CoverageStateDatasetFreeTrainingResult, ...]:
    if objectives not in (
        COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES,
        COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
    ):
        raise ValueError("dataset-free objective suite is not frozen")
    model_config = CoverageStateLevelSetConfig(
        feature_channels=COVERAGE_STATE_DATASET_FREE_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_DATASET_FREE_STRIDE,
        width=COVERAGE_STATE_DATASET_FREE_WIDTH,
    )
    sobolev = CoverageStateSobolevConfig(
        truncation_radius=COVERAGE_STATE_DATASET_FREE_STRIDE
    )
    torch.manual_seed(seed)
    initial_model = CURELiteCoverageStateLevelSet(model_config)
    initial_fingerprint = coverage_state_model_fingerprint(initial_model)
    initial_state = deepcopy(initial_model.state_dict())
    batch = _training_batch(seed)
    results: list[CoverageStateDatasetFreeTrainingResult] = []
    for objective in objectives:
        model = CURELiteCoverageStateLevelSet(model_config)
        model.load_state_dict(initial_state, strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        first_nonzero: dict[str, int] = {}
        losses_finite = True
        forward_calls = 0
        backward_calls = 0
        optimizer_steps = 0
        logical_states = 0
        selection_fingerprints: set[str] = set()
        for update in range(COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES):
            logs = coverage_state_fused_train_step(
                model,
                optimizer,
                batch,
                config=sobolev,
                pair_objective=objective,
                audit=False,
                track_nonzero_gradients=True,
            )
            losses_finite = losses_finite and all(
                isfinite(float(logs[name]))
                for name in (
                    "factual_miss/loss",
                    "factual_no_miss/loss",
                    "pair/loss",
                    "total",
                    "gradient_l2_norm",
                )
            )
            for name in filter(
                None,
                str(logs["nonzero_gradient_parameters"]).split(","),
            ):
                first_nonzero.setdefault(name, update)
            forward_calls += int(logs["model_forward_calls"])
            backward_calls += int(logs["backward_calls"])
            optimizer_steps += int(logs["optimizer_steps"])
            logical_states += int(logs["logical_states"])
            selection_fingerprints.add(
                str(logs["selection_fingerprint"])
            )
        final_fingerprint = coverage_state_model_fingerprint(model)
        diagnostics = _post_training_diagnostics(model, seed=seed)
        if len(selection_fingerprints) != 1:
            raise AssertionError("dataset-free selection changed across updates")
        results.append(
            CoverageStateDatasetFreeTrainingResult(
                seed=seed,
                objective=objective.value,
                updates=COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES,
                forward_calls=forward_calls,
                backward_calls=backward_calls,
                optimizer_steps=optimizer_steps,
                logical_state_evaluations=logical_states,
                initial_model_fingerprint=initial_fingerprint,
                final_model_fingerprint=final_fingerprint,
                selection_fingerprint=next(iter(selection_fingerprints)),
                first_nonzero_gradient_update=tuple(
                    sorted(first_nonzero.items())
                ),
                factual_miss_target_pixels=int(
                    torch.count_nonzero(
                        batch.factual_miss.targets.target_field < 0.0
                    )
                ),
                factual_no_miss_target_pixels=int(
                    torch.count_nonzero(
                        batch.factual_no_miss.targets.target_field < 0.0
                    )
                ),
                losses_finite=losses_finite,
                parameters_changed=(
                    final_fingerprint != initial_fingerprint
                ),
                diagnostic_fields_finite=bool(
                    diagnostics["diagnostic_fields_finite"]
                ),
                identity_field_exact=bool(
                    diagnostics["identity_field_exact"]
                ),
                hidden_component_field_exact=bool(
                    diagnostics["hidden_component_field_exact"]
                ),
                component_new_negative_pixels=int(
                    diagnostics["component_new_negative_pixels"]
                ),
                component_new_negative_components=int(
                    diagnostics["component_new_negative_components"]
                ),
                empty_negative_pixels=int(
                    diagnostics["empty_negative_pixels"]
                ),
                empty_negative_components=int(
                    diagnostics["empty_negative_components"]
                ),
                hard_union_exact=bool(diagnostics["hard_union_exact"]),
            )
        )
    return tuple(results)


def recompute_coverage_state_dataset_free_checks(
    case_results: tuple[CoverageStateDatasetFreeCaseResult, ...],
    training_results: tuple[CoverageStateDatasetFreeTrainingResult, ...],
    completion_root_probes: tuple[
        CoverageStateCompletionRootProbeResult,
        ...,
    ],
) -> tuple[tuple[str, bool], ...]:
    """Recompute the complete frozen gate from its evidence rows.

    This function is the sole implementation of the dataset-free decision
    rule.  Caller-supplied check booleans are never treated as authoritative.
    """

    cases = tuple(case_results)
    training_results = tuple(training_results)
    completion_root_probes = tuple(completion_root_probes)
    optimizer_pairs = tuple(
        value for value in cases if value.optimizer_eligible_pair
    )
    hidden_pairs = tuple(
        value for value in cases if value.expected_scalar_hidden
    )
    identity_pairs = tuple(
        value for value in cases if value.pair_kind == "identity_null"
    )
    component_pairs = tuple(
        value for value in cases if value.pair_kind == "component_null"
    )
    empty_states = tuple(
        value for value in cases if value.case_name == "empty_state"
    )
    expected_case_keys = {
        (name, size, seed)
        for name in COVERAGE_STATE_DATASET_FREE_CASES
        for size in COVERAGE_STATE_DATASET_FREE_SIZES
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
    }
    actual_case_keys = {
        (value.case_name, value.size, value.seed) for value in cases
    }
    expected_probe_keys = {
        (size, seed)
        for size in COVERAGE_STATE_DATASET_FREE_SIZES
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
    }
    actual_probe_keys = {
        (value.size, value.seed) for value in completion_root_probes
    }
    with torch.random.fork_rng(devices=[]):
        parameter_names = {
            name
            for name, _ in CURELiteCoverageStateLevelSet(
                CoverageStateLevelSetConfig(
                    feature_channels=(
                        COVERAGE_STATE_DATASET_FREE_FEATURE_CHANNELS
                    ),
                    feature_stride=COVERAGE_STATE_DATASET_FREE_STRIDE,
                    width=COVERAGE_STATE_DATASET_FREE_WIDTH,
                )
            ).named_parameters()
        }
    return (
        ("geometry_matrix_complete", actual_case_keys == expected_case_keys),
        (
            "geometry_and_model_fields_finite",
            all(
                value.target_fields_finite and value.model_fields_finite
                for value in cases
            )
            and all(
                value.diagnostic_fields_finite
                for value in training_results
            ),
        ),
        (
            "target_zero_level_exact",
            all(value.target_zero_level_exact for value in cases),
        ),
        (
            "hard_union_exact",
            all(value.hard_union_exact for value in cases)
            and all(value.hard_union_exact for value in training_results),
        ),
        (
            "identity_null_exact",
            all(value.identity_field_exact is True for value in identity_pairs)
            and all(value.identity_field_exact for value in training_results),
        ),
        (
            "scalar_hidden_component_exact",
            bool(hidden_pairs)
            and all(
                value.scalar_visible is False
                and value.phase_visible is True
                and value.hidden_component_field_exact is True
                for value in hidden_pairs
            )
            and all(
                value.hidden_component_field_exact
                for value in training_results
            ),
        ),
        (
            "component_null_no_new_negative_island",
            all(
                value.component_new_negative_pixels == 0
                and value.component_new_negative_components == 0
                for value in component_pairs
            )
            and all(
                value.component_new_negative_pixels == 0
                and value.component_new_negative_components == 0
                for value in training_results
            ),
        ),
        (
            "empty_state_negative_component_count_zero",
            bool(empty_states)
            and all(
                value.predicted_negative_pixels == 0
                and value.predicted_negative_components == 0
                for value in empty_states
            )
            and all(
                value.empty_negative_pixels == 0
                and value.empty_negative_components == 0
                for value in training_results
            ),
        ),
        (
            "selected_scalar_representation_has_no_hidden_optimizer_pair",
            bool(optimizer_pairs)
            and all(value.scalar_visible is True for value in optimizer_pairs),
        ),
        (
            "phase_roundtrip_exact",
            all(
                value.phase_roundtrip_exact is True
                for value in cases
                if value.state_type == "pair"
            ),
        ),
        (
            "target_response_inside_selected_scalar_rf",
            all(
                value.target_response_outside_scalar_rf_pixels == 0
                for value in cases
            ),
        ),
        (
            "target_response_inside_phase_rf",
            all(
                value.target_response_outside_phase_rf_pixels == 0
                for value in cases
            ),
        ),
        (
            "completion_root_probe_complete",
            actual_probe_keys == expected_probe_keys
            and len(completion_root_probes) == len(expected_probe_keys)
            and all(
                value.objective_policy
                == CSLF_COMPLETION_ROOTED_RESPONSE_POLICY
                and value.target_pixels > 0
                for value in completion_root_probes
            ),
        ),
        (
            "completion_root_probe_response_correct_without_crossing",
            bool(completion_root_probes)
            and all(
                value.response_sign_pixels > 0
                and value.response_sign_correct_pixels
                == value.response_sign_pixels
                and value.target_negative_pixels_before_update == 0
                and float.fromhex(value.response_error_max_hex)
                <= COVERAGE_STATE_COMPLETION_ROOT_RESPONSE_TOLERANCE
                for value in completion_root_probes
            ),
        ),
        (
            "completion_root_probe_direct_gradient",
            bool(completion_root_probes)
            and all(
                float.fromhex(value.legacy_minus_gradient_max_hex)
                <= (
                    COVERAGE_STATE_COMPLETION_ROOT_LEGACY_GRADIENT_TOLERANCE
                )
                and float.fromhex(
                    value.rooted_minus_target_gradient_min_hex
                )
                >= COVERAGE_STATE_COMPLETION_ROOT_GRADIENT_MINIMUM
                for value in completion_root_probes
            ),
        ),
        (
            "completion_root_probe_fixed_points",
            bool(completion_root_probes)
            and all(
                float.fromhex(value.exact_rooted_loss_hex) == 0.0
                and float.fromhex(
                    value.component_null_rooted_loss_hex
                )
                == 0.0
                for value in completion_root_probes
            ),
        ),
        (
            "three_objectives_computationally_learnable",
            all(
                value.losses_finite
                and value.parameters_changed
                and value.factual_miss_target_pixels > 0
                and value.factual_no_miss_target_pixels == 0
                for value in training_results
            ),
        ),
        (
            "early_gradient_latency",
            all(
                set(dict(value.first_nonzero_gradient_update))
                == parameter_names
                and dict(value.first_nonzero_gradient_update)[
                    "phase_projection.weight"
                ]
                == 0
                and dict(value.first_nonzero_gradient_update)[
                    "phase_projection.bias"
                ]
                == 0
                and dict(value.first_nonzero_gradient_update)[
                    "input_projection.weight"
                ]
                <= 2
                and dict(value.first_nonzero_gradient_update)[
                    "spatial_mixing.weight"
                ]
                <= 2
                for value in training_results
            ),
        ),
        (
            "training_compute_ledger_exact",
            all(
                value.updates
                == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                and value.forward_calls
                == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                and value.backward_calls
                == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                and value.optimizer_steps
                == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                and value.logical_state_evaluations
                == 12 * COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                for value in training_results
            ),
        ),
        (
            "matched_objective_fairness",
            all(
                len(
                    {
                        value.initial_model_fingerprint
                        for value in training_results
                        if value.seed == seed
                    }
                )
                == 1
                and len(
                    {
                        value.selection_fingerprint
                        for value in training_results
                        if value.seed == seed
                    }
                )
                == 1
                for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
            ),
        ),
        (
            "no_dataset_split_access",
            True,
        ),
    )


def run_coverage_state_dataset_free_gate() -> CoverageStateDatasetFreeReceipt:
    """Run the frozen expanded dataset-free gate entirely on CPU."""

    model_config = CoverageStateLevelSetConfig(
        feature_channels=COVERAGE_STATE_DATASET_FREE_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_DATASET_FREE_STRIDE,
        width=COVERAGE_STATE_DATASET_FREE_WIDTH,
    )
    sobolev = CoverageStateSobolevConfig(
        truncation_radius=COVERAGE_STATE_DATASET_FREE_STRIDE
    )
    case_results: list[CoverageStateDatasetFreeCaseResult] = []
    for seed in COVERAGE_STATE_DATASET_FREE_SEEDS:
        torch.manual_seed(seed)
        model = CURELiteCoverageStateLevelSet(model_config)
        model.eval()
        for size in COVERAGE_STATE_DATASET_FREE_SIZES:
            for case_name in COVERAGE_STATE_DATASET_FREE_CASES:
                case_results.append(
                    _evaluate_case(
                        _make_case(case_name, size=size, seed=seed),
                        model=model,
                        config=sobolev,
                        size=size,
                        seed=seed,
                    )
                )
    training_results = tuple(
        result
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
        for result in _run_training_matrix(seed)
    )
    completion_root_probes = tuple(
        _run_completion_root_probe(size=size, seed=seed)
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
        for size in COVERAGE_STATE_DATASET_FREE_SIZES
    )
    cases = tuple(case_results)
    checks = recompute_coverage_state_dataset_free_checks(
        cases,
        training_results,
        completion_root_probes,
    )
    return CoverageStateDatasetFreeReceipt(
        case_results=cases,
        training_results=training_results,
        completion_root_probes=completion_root_probes,
        checks=checks,
    )


@dataclass(frozen=True)
class CoverageStateSupportOrientedProbeResult:
    """Parameter-free selector/root evidence for one canonical pair."""

    size: int
    seed: int
    selector_pixels: int
    expected_added_target_pixels: int
    selector_exact: bool
    root_inside_exact: bool
    root_outside_exact: bool
    response_exact: bool
    direct_minus_gradient_nonzero: bool
    identity_null_selector_empty: bool
    identity_null_exact: bool
    component_null_selector_empty: bool
    component_null_exact: bool
    fixed_point_zero: bool
    fixed_point_gradients_zero: bool
    gradients_finite: bool
    boundary_gradients_finite: bool
    objective_policy: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "size": self.size,
            "seed": self.seed,
            "selector_pixels": self.selector_pixels,
            "expected_added_target_pixels": (
                self.expected_added_target_pixels
            ),
            "selector_exact": self.selector_exact,
            "root_inside_exact": self.root_inside_exact,
            "root_outside_exact": self.root_outside_exact,
            "response_exact": self.response_exact,
            "direct_minus_gradient_nonzero": (
                self.direct_minus_gradient_nonzero
            ),
            "identity_null_selector_empty": (
                self.identity_null_selector_empty
            ),
            "identity_null_exact": self.identity_null_exact,
            "component_null_selector_empty": (
                self.component_null_selector_empty
            ),
            "component_null_exact": self.component_null_exact,
            "fixed_point_zero": self.fixed_point_zero,
            "fixed_point_gradients_zero": (
                self.fixed_point_gradients_zero
            ),
            "gradients_finite": self.gradients_finite,
            "boundary_gradients_finite": (
                self.boundary_gradients_finite
            ),
            "objective_policy": self.objective_policy,
        }


def _pair_loss_fields_exact(
    actual: CoverageStatePairLossFields,
    expected: CoverageStatePairLossFields,
) -> bool:
    names = (
        "loss",
        "value_power",
        "spatial_power",
        "per_state_loss",
        "per_state_value_power",
        "per_state_spatial_power",
        "target_field_plus",
        "target_field_minus",
        "predicted_coverage_response",
        "target_coverage_response",
        "anchor_error",
        "response_error",
        "focus_support",
        "focus_support_field",
        "integration_measure",
    )
    return all(
        torch.equal(getattr(actual, name), getattr(expected, name))
        for name in names
    )


def _run_support_oriented_probe(
    *,
    size: int,
    seed: int,
) -> CoverageStateSupportOrientedProbeResult:
    """Exercise SORR only on generated fields and frozen core functions."""

    if size not in COVERAGE_STATE_DATASET_FREE_SIZES:
        raise ValueError("support-oriented probe size is not frozen")
    if seed not in COVERAGE_STATE_DATASET_FREE_SEEDS:
        raise ValueError("support-oriented probe seed is not frozen")
    config = CoverageStateSobolevConfig(
        truncation_radius=COVERAGE_STATE_DATASET_FREE_STRIDE,
    )
    case = _make_case(
        "clean_pair_with_natural_miss_already_present",
        size=size,
        seed=seed,
    )
    targets = prepare_coverage_state_pair_targets(
        case.occupancy_plus,
        case.occupancy_minus,
        case.target_plus,
        case.target_minus,
        case.valid_mask,
        config=config,
    )
    selector = coverage_state_added_target_support_from_targets(targets)
    expected_selector = case.target_minus & ~case.target_plus

    count = targets.target_field_plus.numel()
    ramp = torch.arange(
        count,
        dtype=torch.float32,
    ).reshape_as(targets.target_field_plus) / float(count)
    seed_scale = float(
        seed - COVERAGE_STATE_DATASET_FREE_SEEDS[0] + 1
    )
    error_plus = (0.03125 * seed_scale + 0.125 * ramp).contiguous()
    error_minus = (
        -0.015625 * seed_scale
        - 0.0625 * torch.flip(ramp, dims=(-1,))
    ).contiguous()
    field_plus = (
        targets.target_field_plus + error_plus
    ).detach().requires_grad_()
    field_minus = (
        targets.target_field_minus + error_minus
    ).detach().requires_grad_()
    result = coverage_state_support_oriented_pair_sobolev_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )
    boundary_gradients = torch.autograd.grad(
        result.loss,
        (field_plus, field_minus),
    )
    valid_outside = targets.valid_mask & ~selector
    response_error = (
        (field_minus.detach() - field_plus.detach())
        - (
            targets.target_field_minus
            - targets.target_field_plus
        )
    )

    direct_plus = (
        targets.target_field_plus + COVERAGE_STATE_COMPLETION_ROOT_OFFSET
    ).detach().requires_grad_()
    direct_minus = (
        targets.target_field_minus + COVERAGE_STATE_COMPLETION_ROOT_OFFSET
    ).detach().requires_grad_()
    direct = coverage_state_support_oriented_pair_sobolev_loss_from_targets(
        direct_plus,
        direct_minus,
        targets,
        config=config,
    )
    direct_gradients = torch.autograd.grad(
        direct.loss,
        (direct_plus, direct_minus),
    )

    fixed_plus = (
        targets.target_field_plus.detach().clone().requires_grad_()
    )
    fixed_minus = (
        targets.target_field_minus.detach().clone().requires_grad_()
    )
    fixed = coverage_state_support_oriented_pair_sobolev_loss_from_targets(
        fixed_plus,
        fixed_minus,
        targets,
        config=config,
    )
    fixed_gradients = torch.autograd.grad(
        fixed.loss,
        (fixed_plus, fixed_minus),
    )

    null_results: dict[str, tuple[bool, bool]] = {}
    null_gradients_finite = True
    for case_name in ("identity_null", "component_null_deletion"):
        null_case = _make_case(case_name, size=size, seed=seed)
        null_targets = prepare_coverage_state_pair_targets(
            null_case.occupancy_plus,
            null_case.occupancy_minus,
            null_case.target_plus,
            null_case.target_minus,
            null_case.valid_mask,
            config=config,
        )
        null_selector = (
            coverage_state_added_target_support_from_targets(null_targets)
        )
        null_count = null_targets.target_field_plus.numel()
        null_ramp = torch.arange(
            null_count,
            dtype=torch.float32,
        ).reshape_as(null_targets.target_field_plus) / float(null_count)
        null_plus = (
            null_targets.target_field_plus + 0.125 * null_ramp
        ).detach().requires_grad_()
        null_minus = (
            null_targets.target_field_minus
            - 0.0625 * torch.flip(null_ramp, dims=(-1,))
        ).detach().requires_grad_()
        support_oriented = (
            coverage_state_support_oriented_pair_sobolev_loss_from_targets(
                null_plus,
                null_minus,
                null_targets,
                config=config,
            )
        )
        support_gradients = torch.autograd.grad(
            support_oriented.loss,
            (null_plus, null_minus),
        )
        legacy_plus = null_plus.detach().clone().requires_grad_()
        legacy_minus = null_minus.detach().clone().requires_grad_()
        legacy = coverage_state_pair_sobolev_loss_from_targets(
            legacy_plus,
            legacy_minus,
            null_targets,
            config=config,
        )
        legacy_gradients = torch.autograd.grad(
            legacy.loss,
            (legacy_plus, legacy_minus),
        )
        null_results[case_name] = (
            not bool(null_selector.any()),
            _pair_loss_fields_exact(support_oriented, legacy)
            and torch.equal(support_gradients[0], legacy_gradients[0])
            and torch.equal(support_gradients[1], legacy_gradients[1]),
        )
        null_gradients_finite = null_gradients_finite and all(
            bool(torch.isfinite(value).all())
            for value in (*support_gradients, *legacy_gradients)
        )

    return CoverageStateSupportOrientedProbeResult(
        size=size,
        seed=seed,
        selector_pixels=int(torch.count_nonzero(selector)),
        expected_added_target_pixels=int(
            torch.count_nonzero(expected_selector)
        ),
        selector_exact=torch.equal(selector, expected_selector),
        root_inside_exact=torch.equal(
            result.anchor_error[selector],
            error_minus[selector],
        ),
        root_outside_exact=torch.equal(
            result.anchor_error[valid_outside],
            error_plus[valid_outside],
        ),
        response_exact=torch.equal(
            result.response_error,
            response_error,
        ),
        direct_minus_gradient_nonzero=(
            int(torch.count_nonzero(direct_gradients[1][selector]))
            == int(torch.count_nonzero(selector))
        ),
        identity_null_selector_empty=null_results["identity_null"][0],
        identity_null_exact=null_results["identity_null"][1],
        component_null_selector_empty=(
            null_results["component_null_deletion"][0]
        ),
        component_null_exact=(
            null_results["component_null_deletion"][1]
        ),
        fixed_point_zero=(
            fixed.loss.detach().item() == 0.0
            and int(torch.count_nonzero(fixed.anchor_error)) == 0
            and int(torch.count_nonzero(fixed.response_error)) == 0
        ),
        fixed_point_gradients_zero=all(
            int(torch.count_nonzero(value)) == 0
            for value in fixed_gradients
        ),
        gradients_finite=(
            null_gradients_finite
            and all(
                bool(torch.isfinite(value).all())
                for value in (*direct_gradients, *fixed_gradients)
            )
        ),
        boundary_gradients_finite=(
            bool(torch.isfinite(result.loss))
            and bool(torch.isfinite(result.value_power))
            and bool(torch.isfinite(result.spatial_power))
            and all(
                bool(torch.isfinite(value).all())
                for value in boundary_gradients
            )
        ),
        objective_policy=CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
    )


def recompute_coverage_state_support_oriented_dataset_free_checks(
    legacy_receipt: CoverageStateDatasetFreeReceipt,
    training_results: tuple[CoverageStateDatasetFreeTrainingResult, ...],
    support_oriented_probes: tuple[
        CoverageStateSupportOrientedProbeResult,
        ...,
    ],
) -> tuple[tuple[str, bool], ...]:
    """Recompute every SORR gate from canonical evidence rows."""

    if not isinstance(legacy_receipt, CoverageStateDatasetFreeReceipt):
        raise TypeError("legacy_receipt must be CoverageStateDatasetFreeReceipt")
    training_results = tuple(training_results)
    support_oriented_probes = tuple(support_oriented_probes)
    expected_probe_keys = {
        (size, seed)
        for size in COVERAGE_STATE_DATASET_FREE_SIZES
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
    }
    actual_probe_keys = {
        (value.size, value.seed) for value in support_oriented_probes
    }
    expected_training_keys = {
        (seed, objective.value)
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
        for objective in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
    }
    actual_training_keys = {
        (value.seed, value.objective) for value in training_results
    }
    inherited_checks = dict(
        recompute_coverage_state_dataset_free_checks(
            legacy_receipt.case_results,
            training_results,
            legacy_receipt.completion_root_probes,
        )
    )
    return (
        (
            "legacy_dataset_free_gate_bound",
            legacy_receipt.receipt_fingerprint
            == COVERAGE_STATE_LEGACY_DATASET_FREE_FINGERPRINT
            and legacy_receipt.all_pass,
        ),
        (
            "support_oriented_probe_matrix_complete",
            actual_probe_keys == expected_probe_keys
            and len(support_oriented_probes) == len(expected_probe_keys)
            and all(
                value.objective_policy
                == CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
                for value in support_oriented_probes
            ),
        ),
        (
            "support_selector_exact",
            bool(support_oriented_probes)
            and all(
                value.selector_exact
                and value.selector_pixels
                == value.expected_added_target_pixels
                and value.selector_pixels > 0
                for value in support_oriented_probes
            ),
        ),
        (
            "support_oriented_root_partition_exact",
            bool(support_oriented_probes)
            and all(
                value.root_inside_exact and value.root_outside_exact
                for value in support_oriented_probes
            ),
        ),
        (
            "finite_response_unchanged",
            bool(support_oriented_probes)
            and all(
                value.response_exact for value in support_oriented_probes
            ),
        ),
        (
            "added_target_minus_root_direct_gradient",
            bool(support_oriented_probes)
            and all(
                value.direct_minus_gradient_nonzero
                for value in support_oriented_probes
            ),
        ),
        (
            "null_pairs_reduce_exactly_to_response_joint",
            bool(support_oriented_probes)
            and all(
                value.identity_null_selector_empty
                and value.identity_null_exact
                and value.component_null_selector_empty
                and value.component_null_exact
                for value in support_oriented_probes
            ),
        ),
        (
            "exact_endpoint_fixed_point",
            bool(support_oriented_probes)
            and all(
                value.fixed_point_zero
                and value.fixed_point_gradients_zero
                for value in support_oriented_probes
            ),
        ),
        (
            "support_boundary_gradients_finite",
            bool(support_oriented_probes)
            and all(
                value.gradients_finite
                and value.boundary_gradients_finite
                for value in support_oriented_probes
            ),
        ),
        (
            "support_oriented_matched_suite_complete",
            actual_training_keys == expected_training_keys
            and len(training_results) == len(expected_training_keys),
        ),
        (
            "support_oriented_short_training_computationally_valid",
            all(
                inherited_checks[name]
                for name in (
                    "three_objectives_computationally_learnable",
                    "early_gradient_latency",
                    "training_compute_ledger_exact",
                    "matched_objective_fairness",
                )
            ),
        ),
        ("no_dataset_split_access", True),
    )


@dataclass(frozen=True)
class CoverageStateSupportOrientedDatasetFreeReceipt:
    """SORR probes and short training bound to the unchanged legacy gate."""

    legacy_receipt: CoverageStateDatasetFreeReceipt
    legacy_receipt_fingerprint: str
    training_results: tuple[CoverageStateDatasetFreeTrainingResult, ...]
    support_oriented_probes: tuple[
        CoverageStateSupportOrientedProbeResult,
        ...,
    ]
    checks: tuple[tuple[str, bool], ...]
    D_R_accessed: bool = False
    D_V_accessed: bool = False
    D_T_accessed: bool = False
    performance_claim_supported: bool = False

    def __post_init__(self) -> None:
        self.verify()

    def verify(self) -> None:
        if (
            not isinstance(self.legacy_receipt, CoverageStateDatasetFreeReceipt)
            or self.legacy_receipt.receipt_fingerprint
            != self.legacy_receipt_fingerprint
            or self.legacy_receipt_fingerprint
            != COVERAGE_STATE_LEGACY_DATASET_FREE_FINGERPRINT
            or not isinstance(self.training_results, tuple)
            or not isinstance(self.support_oriented_probes, tuple)
            or not isinstance(self.checks, tuple)
            or not all(
                isinstance(value, CoverageStateDatasetFreeTrainingResult)
                for value in self.training_results
            )
            or not all(
                isinstance(value, CoverageStateSupportOrientedProbeResult)
                for value in self.support_oriented_probes
            )
            or not all(
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], str)
                and type(item[1]) is bool
                for item in self.checks
            )
            or not self.checks
            or len({name for name, _ in self.checks}) != len(self.checks)
        ):
            raise ValueError("support-oriented dataset-free receipt is incomplete")
        if (
            self.D_R_accessed
            or self.D_V_accessed
            or self.D_T_accessed
            or self.performance_claim_supported
        ):
            raise ValueError(
                "support-oriented dataset-free receipt exceeds its scope"
            )
        recomputed = (
            recompute_coverage_state_support_oriented_dataset_free_checks(
                self.legacy_receipt,
                self.training_results,
                self.support_oriented_probes,
            )
        )
        if self.checks != recomputed:
            raise ValueError(
                "support-oriented dataset-free checks changed"
            )

    @property
    def all_pass(self) -> bool:
        self.verify()
        return all(passed for _, passed in self.checks)

    @property
    def status(self) -> str:
        return (
            "SUPPORT_ORIENTED_DATASET_FREE_GATE_PASS"
            if self.all_pass
            else "SUPPORT_ORIENTED_DATASET_FREE_GATE_FAIL"
        )

    def canonical_payload(self) -> dict[str, object]:
        self.verify()
        return {
            "schema_version": (
                COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_SCHEMA
            ),
            "scope": {
                "sizes": list(COVERAGE_STATE_DATASET_FREE_SIZES),
                "seeds": list(COVERAGE_STATE_DATASET_FREE_SEEDS),
                "probe_count": len(self.support_oriented_probes),
                "training_updates_per_objective": (
                    COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                ),
                "training_result_count": len(self.training_results),
                "objective_suite": [
                    value.value
                    for value in (
                        COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
                    )
                ],
                "fixture_source": "generated_in_memory_only",
                "training_role": (
                    "computational_learnability_and_early_gradient_only"
                ),
                "selector_source": (
                    "frozen_target_fields_strict_signs_only"
                ),
                "no_gate_hyperparameter_search": True,
            },
            "legacy_dataset_free_receipt_fingerprint": (
                self.legacy_receipt_fingerprint
            ),
            "training_results": [
                value.canonical_payload() for value in self.training_results
            ],
            "support_oriented_probes": [
                value.canonical_payload()
                for value in self.support_oriented_probes
            ],
            "checks": {name: passed for name, passed in self.checks},
            "status": self.status,
            "all_pass": self.all_pass,
            "data_access": {
                "D_R_accessed": self.D_R_accessed,
                "D_V_accessed": self.D_V_accessed,
                "D_T_accessed": self.D_T_accessed,
            },
            "performance_claim_supported": self.performance_claim_supported,
            "formal_training_authorized": False,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_coverage_state_support_oriented_dataset_free_gate(
) -> CoverageStateSupportOrientedDatasetFreeReceipt:
    """Run the generated-only SORR gate with no dataset access."""

    legacy_receipt = run_coverage_state_dataset_free_gate()
    if (
        legacy_receipt.receipt_fingerprint
        != COVERAGE_STATE_LEGACY_DATASET_FREE_FINGERPRINT
    ):
        raise RuntimeError("legacy dataset-free receipt changed")
    training_results = tuple(
        result
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
        for result in _run_training_matrix(
            seed,
            objectives=COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
        )
    )
    probes = tuple(
        _run_support_oriented_probe(size=size, seed=seed)
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
        for size in COVERAGE_STATE_DATASET_FREE_SIZES
    )
    checks = (
        recompute_coverage_state_support_oriented_dataset_free_checks(
            legacy_receipt,
            training_results,
            probes,
        )
    )
    return CoverageStateSupportOrientedDatasetFreeReceipt(
        legacy_receipt=legacy_receipt,
        legacy_receipt_fingerprint=legacy_receipt.receipt_fingerprint,
        training_results=training_results,
        support_oriented_probes=probes,
        checks=checks,
    )


@dataclass(frozen=True)
class CoverageStatePhasePreservingArchitectureProbeResult:
    """Generated-only evidence for the PPCE structural coordinate."""

    stride: int
    phase_channel_count: int
    roundtrip_case_count: int
    roundtrip_exact: bool
    phase_index_checks: int
    phase_index_exact: bool
    diagonal_alignment_checks: int
    diagonal_alignment_exact: bool
    scalar_projection_collision_exact: bool
    phase_encoding_separates_collision: bool
    ppce_state_separates_collision: bool
    module_names: tuple[str, ...]
    single_path_exact: bool
    legacy_parameter_count: int
    ppce_parameter_count: int
    expected_ppce_parameter_count: int
    parameter_formula_exact: bool
    parameter_delta: int
    expected_parameter_delta: int
    initial_field_value_hex: str
    initial_positive_field_exact: bool
    initial_completion_empty: bool
    phase_occupancy_bool: bool
    phase_occupancy_contiguous: bool
    coverage_policy: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "stride": self.stride,
            "phase_channel_count": self.phase_channel_count,
            "roundtrip_case_count": self.roundtrip_case_count,
            "roundtrip_exact": self.roundtrip_exact,
            "phase_index_checks": self.phase_index_checks,
            "phase_index_exact": self.phase_index_exact,
            "diagonal_alignment_checks": self.diagonal_alignment_checks,
            "diagonal_alignment_exact": self.diagonal_alignment_exact,
            "scalar_projection_collision_exact": (
                self.scalar_projection_collision_exact
            ),
            "phase_encoding_separates_collision": (
                self.phase_encoding_separates_collision
            ),
            "ppce_state_separates_collision": (
                self.ppce_state_separates_collision
            ),
            "module_names": list(self.module_names),
            "single_path_exact": self.single_path_exact,
            "legacy_parameter_count": self.legacy_parameter_count,
            "ppce_parameter_count": self.ppce_parameter_count,
            "expected_ppce_parameter_count": (
                self.expected_ppce_parameter_count
            ),
            "parameter_formula_exact": self.parameter_formula_exact,
            "parameter_delta": self.parameter_delta,
            "expected_parameter_delta": self.expected_parameter_delta,
            "initial_field_value_hex": self.initial_field_value_hex,
            "initial_positive_field_exact": (
                self.initial_positive_field_exact
            ),
            "initial_completion_empty": self.initial_completion_empty,
            "phase_occupancy_bool": self.phase_occupancy_bool,
            "phase_occupancy_contiguous": (
                self.phase_occupancy_contiguous
            ),
            "coverage_policy": self.coverage_policy,
        }


@dataclass(frozen=True)
class CoverageStatePhasePreservingGeometryProbeResult:
    """Frozen target/SDF/measure evidence around a PPCE forward."""

    seed: int
    target_field_fingerprint: str
    recomputed_target_field_fingerprint: str
    post_forward_target_field_fingerprint: str
    sdf_fingerprint: str
    recomputed_sdf_fingerprint: str
    post_forward_sdf_fingerprint: str
    integration_measure_fingerprint: str
    recomputed_integration_measure_fingerprint: str
    post_forward_integration_measure_fingerprint: str
    geometry_fingerprint: str
    recomputed_geometry_fingerprint: str
    post_forward_geometry_fingerprint: str
    target_fields_exact: bool
    sdf_fields_exact: bool
    integration_measures_exact: bool
    geometry_exact: bool
    geometry_unchanged_after_ppce_forward: bool
    support_oriented_inverse_exact: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "target_field_fingerprint": self.target_field_fingerprint,
            "recomputed_target_field_fingerprint": (
                self.recomputed_target_field_fingerprint
            ),
            "post_forward_target_field_fingerprint": (
                self.post_forward_target_field_fingerprint
            ),
            "sdf_fingerprint": self.sdf_fingerprint,
            "recomputed_sdf_fingerprint": (
                self.recomputed_sdf_fingerprint
            ),
            "post_forward_sdf_fingerprint": (
                self.post_forward_sdf_fingerprint
            ),
            "integration_measure_fingerprint": (
                self.integration_measure_fingerprint
            ),
            "recomputed_integration_measure_fingerprint": (
                self.recomputed_integration_measure_fingerprint
            ),
            "post_forward_integration_measure_fingerprint": (
                self.post_forward_integration_measure_fingerprint
            ),
            "geometry_fingerprint": self.geometry_fingerprint,
            "recomputed_geometry_fingerprint": (
                self.recomputed_geometry_fingerprint
            ),
            "post_forward_geometry_fingerprint": (
                self.post_forward_geometry_fingerprint
            ),
            "target_fields_exact": self.target_fields_exact,
            "sdf_fields_exact": self.sdf_fields_exact,
            "integration_measures_exact": (
                self.integration_measures_exact
            ),
            "geometry_exact": self.geometry_exact,
            "geometry_unchanged_after_ppce_forward": (
                self.geometry_unchanged_after_ppce_forward
            ),
            "support_oriented_inverse_exact": (
                self.support_oriented_inverse_exact
            ),
        }


def _fingerprint_named_tensors(
    values: dict[str, Tensor],
) -> str:
    return stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(values.items())
        }
    )


def _coverage_state_geometry_fingerprints(
    batch: CoverageStateFusedBatch,
) -> tuple[str, str, str, str]:
    """Fingerprint every frozen target tensor by semantic coordinate."""

    if not isinstance(batch, CoverageStateFusedBatch):
        raise TypeError("batch must be CoverageStateFusedBatch")
    batch.validate()
    target_fields: dict[str, Tensor] = {}
    sdf_fields: dict[str, Tensor] = {}
    measures: dict[str, Tensor] = {}
    geometry: dict[str, Tensor] = {}

    def add_absolute(
        prefix: str,
        targets: CoverageStateAbsoluteTargets,
    ) -> None:
        target_fields[f"{prefix}.target_field"] = targets.target_field
        sdf_fields[f"{prefix}.target_field"] = targets.target_field
        sdf_fields[f"{prefix}.focus_support_field"] = (
            targets.focus_support_field
        )
        measures[f"{prefix}.integration_measure"] = (
            targets.integration_measure
        )
        geometry.update(
            {
                f"{prefix}.target_field": targets.target_field,
                f"{prefix}.focus_support_field": (
                    targets.focus_support_field
                ),
                f"{prefix}.integration_measure": (
                    targets.integration_measure
                ),
                f"{prefix}.field_valid_mask": targets.field_valid_mask,
                f"{prefix}.loss_valid_mask": targets.loss_valid_mask,
                f"{prefix}.focus_support": targets.focus_support,
            }
        )

    def add_pair(
        prefix: str,
        targets: CoverageStatePairTargets,
    ) -> None:
        target_fields[f"{prefix}.target_field_plus"] = (
            targets.target_field_plus
        )
        target_fields[f"{prefix}.target_field_minus"] = (
            targets.target_field_minus
        )
        sdf_fields[f"{prefix}.target_field_plus"] = (
            targets.target_field_plus
        )
        sdf_fields[f"{prefix}.target_field_minus"] = (
            targets.target_field_minus
        )
        sdf_fields[f"{prefix}.focus_support_field"] = (
            targets.focus_support_field
        )
        measures[f"{prefix}.integration_measure"] = (
            targets.integration_measure
        )
        geometry.update(
            {
                f"{prefix}.target_field_plus": (
                    targets.target_field_plus
                ),
                f"{prefix}.target_field_minus": (
                    targets.target_field_minus
                ),
                f"{prefix}.focus_support_field": (
                    targets.focus_support_field
                ),
                f"{prefix}.integration_measure": (
                    targets.integration_measure
                ),
                f"{prefix}.valid_mask": targets.valid_mask,
                f"{prefix}.focus_support": targets.focus_support,
            }
        )

    add_absolute("factual_miss", batch.factual_miss.targets)
    add_absolute("factual_no_miss", batch.factual_no_miss.targets)
    add_pair("pair_joint", batch.pairs.joint_targets)
    add_absolute("pair_absolute_plus", batch.pairs.absolute_targets_plus)
    add_absolute("pair_absolute_minus", batch.pairs.absolute_targets_minus)
    return (
        _fingerprint_named_tensors(target_fields),
        _fingerprint_named_tensors(sdf_fields),
        _fingerprint_named_tensors(measures),
        _fingerprint_named_tensors(geometry),
    )


def _run_phase_preserving_architecture_probe(
) -> CoverageStatePhasePreservingArchitectureProbeResult:
    stride = COVERAGE_STATE_DATASET_FREE_STRIDE
    phase_channels = stride**2
    roundtrip_cases: list[Tensor] = []
    empty = torch.zeros(1, 1, 3 * stride, 5 * stride, dtype=torch.bool)
    roundtrip_cases.append(empty)
    roundtrip_cases.append(torch.ones_like(empty))
    one_per_phase = torch.zeros_like(empty)
    for row in range(stride):
        for column in range(stride):
            one_per_phase[0, 0, stride + row, 2 * stride + column] = True
    roundtrip_cases.append(one_per_phase)
    cross_cell = torch.zeros_like(empty)
    cross_cell[0, 0, stride - 1 : stride + 2, stride - 1 : stride + 2] = True
    roundtrip_cases.append(cross_cell)
    multi_component = torch.zeros_like(empty)
    multi_component[0, 0, 1, 1] = True
    multi_component[0, 0, 2 * stride + 2, 4 * stride + 1] = True
    multi_component[0, 0, stride : stride + 2, 3 * stride : 3 * stride + 2] = True
    roundtrip_cases.append(multi_component)
    phase_values = tuple(
        pixel_unshuffle_bool_occupancy(value, stride=stride)
        for value in roundtrip_cases
    )
    roundtrip_exact = all(
        torch.equal(
            torch.nn.functional.pixel_shuffle(
                phase.to(dtype=torch.float32),
                stride,
            ).to(dtype=torch.bool),
            occupancy,
        )
        for occupancy, phase in zip(roundtrip_cases, phase_values)
    )

    phase_index_exact = True
    phase_index_checks = 0
    coarse_y, coarse_x = 1, 2
    for row in range(stride):
        for column in range(stride):
            occupancy = torch.zeros(
                1,
                1,
                3 * stride,
                4 * stride,
                dtype=torch.bool,
            )
            occupancy[
                0,
                0,
                coarse_y * stride + row,
                coarse_x * stride + column,
            ] = True
            phase = pixel_unshuffle_bool_occupancy(
                occupancy,
                stride=stride,
            )
            expected = [[0, row * stride + column, coarse_y, coarse_x]]
            phase_index_exact = (
                phase_index_exact
                and torch.nonzero(phase, as_tuple=False).tolist() == expected
            )
            phase_index_checks += 1

    diagonal_config = CoverageStatePhasePreservingConfig(
        feature_channels=1,
        feature_stride=stride,
        width=phase_channels,
    )
    diagonal_model = CURELitePhasePreservingCoverageStateLevelSet(
        diagonal_config
    )
    with torch.no_grad():
        diagonal_model.input_projection.weight.zero_()
        diagonal_model.spatial_mixing.weight.zero_()
        diagonal_model.phase_projection.weight.zero_()
        diagonal_model.phase_projection.bias.zero_()
        for phase_index in range(phase_channels):
            diagonal_model.input_projection.weight[
                phase_index,
                1 + phase_index,
                1,
                1,
            ] = 1.0
            diagonal_model.phase_projection.weight[
                phase_index,
                phase_index,
                0,
                0,
            ] = 1.0
    diagonal_feature = torch.zeros(
        phase_channels,
        1,
        2,
        3,
        dtype=torch.float32,
    )
    diagonal_occupancy = torch.zeros(
        phase_channels,
        1,
        2 * stride,
        3 * stride,
        dtype=torch.bool,
    )
    for phase_index in range(phase_channels):
        row, column = divmod(phase_index, stride)
        diagonal_occupancy[
            phase_index,
            0,
            stride + row,
            stride + column,
        ] = True
    with torch.no_grad():
        diagonal_field = diagonal_model(
            diagonal_feature,
            diagonal_occupancy,
        )
    expected_diagonal = torch.zeros_like(diagonal_field)
    expected_value = float(torch.nn.functional.silu(torch.tensor(1.0)))
    for phase_index in range(phase_channels):
        row, column = divmod(phase_index, stride)
        expected_diagonal[
            phase_index,
            0,
            stride + row,
            stride + column,
        ] = expected_value
    diagonal_alignment_exact = torch.equal(
        diagonal_field,
        expected_diagonal,
    )

    collision_feature = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
    collision_first = torch.zeros(
        1,
        1,
        2 * stride,
        2 * stride,
        dtype=torch.bool,
    )
    collision_second = torch.zeros_like(collision_first)
    collision_first[0, 0, 0, 0] = True
    collision_second[0, 0, stride - 1, stride - 1] = True
    scalar_projection_collision_exact = torch.equal(
        occupancy_to_scalar_grid(
            collision_first,
            feature_size=(2, 2),
        ),
        occupancy_to_scalar_grid(
            collision_second,
            feature_size=(2, 2),
        ),
    )
    first_fields = diagonal_model.forward_fields(
        collision_feature,
        collision_first,
    )
    second_fields = diagonal_model.forward_fields(
        collision_feature,
        collision_second,
    )
    phase_encoding_separates_collision = not torch.equal(
        first_fields.phase_occupancy,
        second_fields.phase_occupancy,
    )
    ppce_state_separates_collision = not torch.equal(
        first_fields.field,
        second_fields.field,
    )

    formal_config = CoverageStatePhasePreservingConfig(
        feature_channels=64,
        feature_stride=stride,
        width=32,
    )
    legacy_config = CoverageStateLevelSetConfig(
        feature_channels=64,
        feature_stride=stride,
        width=32,
    )
    formal_model = CURELitePhasePreservingCoverageStateLevelSet(
        formal_config
    )
    formal_feature = torch.zeros(1, 64, 2, 3, dtype=torch.float32)
    formal_occupancy = torch.zeros(
        1,
        1,
        2 * stride,
        3 * stride,
        dtype=torch.bool,
    )
    with torch.no_grad():
        formal_fields = formal_model.forward_fields(
            formal_feature,
            formal_occupancy,
        )
        formal_completion = formal_model.predict_completion(
            formal_feature,
            formal_occupancy,
        )
    module_names = tuple(dict(formal_model.named_children()))
    ppce_parameter_count = sum(
        parameter.numel() for parameter in formal_model.parameters()
    )
    legacy_parameter_count = legacy_config.expected_parameter_count
    expected_delta = (
        (formal_config.phase_occupancy_channels - 1)
        * formal_config.width
        * 3
        * 3
    )
    return CoverageStatePhasePreservingArchitectureProbeResult(
        stride=stride,
        phase_channel_count=phase_channels,
        roundtrip_case_count=len(roundtrip_cases),
        roundtrip_exact=roundtrip_exact,
        phase_index_checks=phase_index_checks,
        phase_index_exact=phase_index_exact,
        diagonal_alignment_checks=phase_channels,
        diagonal_alignment_exact=diagonal_alignment_exact,
        scalar_projection_collision_exact=(
            scalar_projection_collision_exact
        ),
        phase_encoding_separates_collision=(
            phase_encoding_separates_collision
        ),
        ppce_state_separates_collision=ppce_state_separates_collision,
        module_names=module_names,
        single_path_exact=module_names
        == (
            "input_projection",
            "spatial_mixing",
            "phase_projection",
            "pixel_shuffle",
        ),
        legacy_parameter_count=legacy_parameter_count,
        ppce_parameter_count=ppce_parameter_count,
        expected_ppce_parameter_count=formal_config.expected_parameter_count,
        parameter_formula_exact=(
            ppce_parameter_count == formal_config.expected_parameter_count
            and ppce_parameter_count == 23856
        ),
        parameter_delta=ppce_parameter_count - legacy_parameter_count,
        expected_parameter_delta=expected_delta,
        initial_field_value_hex=float(CSLF_FIELD_AMPLITUDE).hex(),
        initial_positive_field_exact=torch.equal(
            formal_fields.field,
            torch.full_like(formal_fields.field, CSLF_FIELD_AMPLITUDE),
        ),
        initial_completion_empty=not bool(formal_completion.any()),
        phase_occupancy_bool=all(
            value.dtype == torch.bool for value in phase_values
        ),
        phase_occupancy_contiguous=all(
            value.is_contiguous() for value in phase_values
        ),
        coverage_policy=formal_config.coverage_policy,
    )


def _run_phase_preserving_geometry_probe(
    *,
    seed: int,
) -> CoverageStatePhasePreservingGeometryProbeResult:
    if seed not in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS:
        raise ValueError("PPCE geometry seed is not frozen")
    batch = _training_batch(seed)
    recomputed = _training_batch(seed)
    before = _coverage_state_geometry_fingerprints(batch)
    repeated = _coverage_state_geometry_fingerprints(recomputed)
    model = CURELitePhasePreservingCoverageStateLevelSet(
        CoverageStatePhasePreservingConfig(
            feature_channels=(
                COVERAGE_STATE_DATASET_FREE_FEATURE_CHANNELS
            ),
            feature_stride=COVERAGE_STATE_DATASET_FREE_STRIDE,
            width=COVERAGE_STATE_DATASET_FREE_WIDTH,
        )
    )
    feature, occupancy = batch.model_inputs()
    with torch.no_grad():
        model(feature, occupancy)
    after = _coverage_state_geometry_fingerprints(batch)

    targets = batch.pairs.joint_targets
    selector = coverage_state_added_target_support_from_targets(targets)
    error_plus = torch.full_like(targets.target_field_plus, 0.125)
    error_minus = torch.full_like(targets.target_field_minus, -0.25)
    response_error = error_minus - error_plus
    anchor_error = torch.where(
        selector,
        error_minus,
        error_plus,
    )
    selector_float = selector.to(dtype=anchor_error.dtype)
    reconstructed_plus = (
        anchor_error - selector_float * response_error
    )
    reconstructed_minus = (
        anchor_error
        + (1.0 - selector_float) * response_error
    )
    inverse_exact = (
        torch.equal(reconstructed_plus, error_plus)
        and torch.equal(reconstructed_minus, error_minus)
    )
    return CoverageStatePhasePreservingGeometryProbeResult(
        seed=seed,
        target_field_fingerprint=before[0],
        recomputed_target_field_fingerprint=repeated[0],
        post_forward_target_field_fingerprint=after[0],
        sdf_fingerprint=before[1],
        recomputed_sdf_fingerprint=repeated[1],
        post_forward_sdf_fingerprint=after[1],
        integration_measure_fingerprint=before[2],
        recomputed_integration_measure_fingerprint=repeated[2],
        post_forward_integration_measure_fingerprint=after[2],
        geometry_fingerprint=before[3],
        recomputed_geometry_fingerprint=repeated[3],
        post_forward_geometry_fingerprint=after[3],
        target_fields_exact=before[0] == repeated[0],
        sdf_fields_exact=before[1] == repeated[1],
        integration_measures_exact=before[2] == repeated[2],
        geometry_exact=before[3] == repeated[3],
        geometry_unchanged_after_ppce_forward=before == after,
        support_oriented_inverse_exact=inverse_exact,
    )


def _run_phase_preserving_training_matrix(
    seed: int,
) -> tuple[CoverageStateDatasetFreeTrainingResult, ...]:
    """Run the three matched SORR objectives with one shared PPCE state."""

    if seed not in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS:
        raise ValueError("PPCE training seed is not frozen")
    model_config = CoverageStatePhasePreservingConfig(
        feature_channels=COVERAGE_STATE_DATASET_FREE_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_DATASET_FREE_STRIDE,
        width=COVERAGE_STATE_DATASET_FREE_WIDTH,
    )
    sobolev = CoverageStateSobolevConfig(
        truncation_radius=COVERAGE_STATE_DATASET_FREE_STRIDE
    )
    torch.manual_seed(seed)
    initial_model = CURELitePhasePreservingCoverageStateLevelSet(
        model_config
    )
    initial_fingerprint = coverage_state_model_fingerprint(initial_model)
    initial_state = deepcopy(initial_model.state_dict())
    batch = _training_batch(seed)
    results: list[CoverageStateDatasetFreeTrainingResult] = []
    for objective in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES:
        model = CURELitePhasePreservingCoverageStateLevelSet(model_config)
        model.load_state_dict(initial_state, strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        first_nonzero: dict[str, int] = {}
        losses_finite = True
        forward_calls = 0
        backward_calls = 0
        optimizer_steps = 0
        logical_states = 0
        selection_fingerprints: set[str] = set()
        for update in range(COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES):
            logs = coverage_state_fused_train_step(
                model,
                optimizer,
                batch,
                config=sobolev,
                pair_objective=objective,
                audit=False,
                track_nonzero_gradients=True,
            )
            losses_finite = losses_finite and all(
                isfinite(float(logs[name]))
                for name in (
                    "factual_miss/loss",
                    "factual_no_miss/loss",
                    "pair/loss",
                    "total",
                    "gradient_l2_norm",
                )
            )
            for name in filter(
                None,
                str(logs["nonzero_gradient_parameters"]).split(","),
            ):
                first_nonzero.setdefault(name, update)
            forward_calls += int(logs["model_forward_calls"])
            backward_calls += int(logs["backward_calls"])
            optimizer_steps += int(logs["optimizer_steps"])
            logical_states += int(logs["logical_states"])
            selection_fingerprints.add(
                str(logs["selection_fingerprint"])
            )
        final_fingerprint = coverage_state_model_fingerprint(model)
        diagnostics = _post_training_diagnostics(model, seed=seed)
        if len(selection_fingerprints) != 1:
            raise AssertionError("PPCE selection changed across updates")
        results.append(
            CoverageStateDatasetFreeTrainingResult(
                seed=seed,
                objective=objective.value,
                updates=COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES,
                forward_calls=forward_calls,
                backward_calls=backward_calls,
                optimizer_steps=optimizer_steps,
                logical_state_evaluations=logical_states,
                initial_model_fingerprint=initial_fingerprint,
                final_model_fingerprint=final_fingerprint,
                selection_fingerprint=next(iter(selection_fingerprints)),
                first_nonzero_gradient_update=tuple(
                    sorted(first_nonzero.items())
                ),
                factual_miss_target_pixels=int(
                    torch.count_nonzero(
                        batch.factual_miss.targets.target_field < 0.0
                    )
                ),
                factual_no_miss_target_pixels=int(
                    torch.count_nonzero(
                        batch.factual_no_miss.targets.target_field < 0.0
                    )
                ),
                losses_finite=losses_finite,
                parameters_changed=(
                    final_fingerprint != initial_fingerprint
                ),
                diagnostic_fields_finite=bool(
                    diagnostics["diagnostic_fields_finite"]
                ),
                identity_field_exact=bool(
                    diagnostics["identity_field_exact"]
                ),
                hidden_component_field_exact=bool(
                    diagnostics["hidden_component_field_exact"]
                ),
                component_new_negative_pixels=int(
                    diagnostics["component_new_negative_pixels"]
                ),
                component_new_negative_components=int(
                    diagnostics["component_new_negative_components"]
                ),
                empty_negative_pixels=int(
                    diagnostics["empty_negative_pixels"]
                ),
                empty_negative_components=int(
                    diagnostics["empty_negative_components"]
                ),
                hard_union_exact=bool(diagnostics["hard_union_exact"]),
            )
        )
    return tuple(results)


def recompute_coverage_state_phase_preserving_dataset_free_checks(
    support_oriented_receipt: CoverageStateSupportOrientedDatasetFreeReceipt,
    architecture_probe: CoverageStatePhasePreservingArchitectureProbeResult,
    geometry_probes: tuple[
        CoverageStatePhasePreservingGeometryProbeResult,
        ...,
    ],
    training_results: tuple[CoverageStateDatasetFreeTrainingResult, ...],
    support_oriented_probes: tuple[
        CoverageStateSupportOrientedProbeResult,
        ...,
    ],
) -> tuple[tuple[str, bool], ...]:
    """Recompute PPCE gates from canonical evidence without dataset access."""

    if not isinstance(
        support_oriented_receipt,
        CoverageStateSupportOrientedDatasetFreeReceipt,
    ):
        raise TypeError(
            "support_oriented_receipt must be the frozen SORR receipt"
        )
    if not isinstance(
        architecture_probe,
        CoverageStatePhasePreservingArchitectureProbeResult,
    ):
        raise TypeError("architecture_probe has an invalid type")
    geometry_probes = tuple(geometry_probes)
    training_results = tuple(training_results)
    support_oriented_probes = tuple(support_oriented_probes)
    expected_geometry_keys = set(
        COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
    )
    actual_geometry_keys = {value.seed for value in geometry_probes}
    expected_probe_keys = {
        (size, seed)
        for size in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SIZES
        for seed in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
    }
    actual_probe_keys = {
        (value.size, value.seed) for value in support_oriented_probes
    }
    expected_training_keys = {
        (seed, objective.value)
        for seed in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
        for objective in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
    }
    actual_training_keys = {
        (value.seed, value.objective) for value in training_results
    }
    expected_parameter_names = {
        "input_projection.weight",
        "spatial_mixing.weight",
        "phase_projection.weight",
        "phase_projection.bias",
    }
    return (
        (
            "support_oriented_dataset_free_gate_bound",
            support_oriented_receipt.receipt_fingerprint
            == COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_FINGERPRINT
            and support_oriented_receipt.all_pass,
        ),
        (
            "phase_roundtrip_and_index_exact",
            architecture_probe.roundtrip_case_count == 5
            and architecture_probe.roundtrip_exact
            and architecture_probe.phase_index_checks
            == architecture_probe.phase_channel_count
            and architecture_probe.phase_index_exact
            and architecture_probe.phase_occupancy_bool
            and architecture_probe.phase_occupancy_contiguous,
        ),
        (
            "input_hidden_output_phase_diagonal_exact",
            architecture_probe.diagonal_alignment_checks
            == architecture_probe.phase_channel_count
            and architecture_probe.diagonal_alignment_exact,
        ),
        (
            "scalar_collision_released_by_ppce",
            architecture_probe.scalar_projection_collision_exact
            and architecture_probe.phase_encoding_separates_collision
            and architecture_probe.ppce_state_separates_collision,
        ),
        (
            "single_path_and_parameter_formula_exact",
            architecture_probe.single_path_exact
            and architecture_probe.parameter_formula_exact
            and architecture_probe.ppce_parameter_count
            == architecture_probe.expected_ppce_parameter_count
            and architecture_probe.parameter_delta
            == architecture_probe.expected_parameter_delta
            and architecture_probe.parameter_delta == 4320
            and architecture_probe.coverage_policy
            == CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
        ),
        (
            "initial_positive_field_and_empty_completion",
            architecture_probe.initial_positive_field_exact
            and architecture_probe.initial_completion_empty
            and architecture_probe.initial_field_value_hex
            == float(CSLF_FIELD_AMPLITUDE).hex(),
        ),
        (
            "frozen_geometry_probe_complete",
            actual_geometry_keys == expected_geometry_keys
            and len(geometry_probes) == len(expected_geometry_keys),
        ),
        (
            "targets_sdf_and_measure_recomputed_exact",
            bool(geometry_probes)
            and all(
                value.target_fields_exact
                and value.sdf_fields_exact
                and value.integration_measures_exact
                and value.geometry_exact
                for value in geometry_probes
            ),
        ),
        (
            "ppce_forward_does_not_mutate_frozen_geometry",
            bool(geometry_probes)
            and all(
                value.geometry_unchanged_after_ppce_forward
                and value.target_field_fingerprint
                == value.post_forward_target_field_fingerprint
                and value.sdf_fingerprint
                == value.post_forward_sdf_fingerprint
                and value.integration_measure_fingerprint
                == value.post_forward_integration_measure_fingerprint
                for value in geometry_probes
            ),
        ),
        (
            "support_oriented_selector_inverse_exact",
            bool(geometry_probes)
            and all(
                value.support_oriented_inverse_exact
                for value in geometry_probes
            ),
        ),
        (
            "support_oriented_probe_matrix_complete",
            actual_probe_keys == expected_probe_keys
            and len(support_oriented_probes) == len(expected_probe_keys),
        ),
        (
            "support_oriented_selector_null_fixedpoint_and_gradient_exact",
            bool(support_oriented_probes)
            and all(
                value.objective_policy
                == CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
                and value.selector_exact
                and value.selector_pixels
                == value.expected_added_target_pixels
                and value.root_inside_exact
                and value.root_outside_exact
                and value.response_exact
                and value.direct_minus_gradient_nonzero
                and value.identity_null_selector_empty
                and value.identity_null_exact
                and value.component_null_selector_empty
                and value.component_null_exact
                and value.fixed_point_zero
                and value.fixed_point_gradients_zero
                and value.gradients_finite
                and value.boundary_gradients_finite
                for value in support_oriented_probes
            ),
        ),
        (
            "ppce_matched_short_training_complete",
            actual_training_keys == expected_training_keys
            and len(training_results) == len(expected_training_keys),
        ),
        (
            "ppce_matched_short_training_computationally_valid",
            bool(training_results)
            and all(
                value.updates
                == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                and value.losses_finite
                and value.parameters_changed
                and value.diagnostic_fields_finite
                and value.identity_field_exact
                and value.empty_negative_pixels == 0
                and value.empty_negative_components == 0
                and value.hard_union_exact
                and value.forward_calls
                == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                and value.backward_calls
                == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                and value.optimizer_steps
                == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                and value.logical_state_evaluations
                == 12 * COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                and {
                    name
                    for name, update in value.first_nonzero_gradient_update
                    if 0 <= update
                    < COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                }
                == expected_parameter_names
                for value in training_results
            ),
        ),
        (
            "ppce_matched_objective_fairness",
            bool(training_results)
            and all(
                len(
                    {
                        value.initial_model_fingerprint
                        for value in training_results
                        if value.seed == seed
                    }
                )
                == 1
                and len(
                    {
                        value.selection_fingerprint
                        for value in training_results
                        if value.seed == seed
                    }
                )
                == 1
                and len(
                    {
                        (
                            value.factual_miss_target_pixels,
                            value.factual_no_miss_target_pixels,
                        )
                        for value in training_results
                        if value.seed == seed
                    }
                )
                == 1
                for seed in (
                    COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
                )
            ),
        ),
        ("no_dataset_split_access", True),
    )


@dataclass(frozen=True)
class CoverageStatePhasePreservingDatasetFreeReceipt:
    """PPCE evidence bound to, but independent from, the SORR receipt."""

    support_oriented_receipt: CoverageStateSupportOrientedDatasetFreeReceipt
    support_oriented_receipt_fingerprint: str
    architecture_probe: CoverageStatePhasePreservingArchitectureProbeResult
    geometry_probes: tuple[
        CoverageStatePhasePreservingGeometryProbeResult,
        ...,
    ]
    training_results: tuple[CoverageStateDatasetFreeTrainingResult, ...]
    support_oriented_probes: tuple[
        CoverageStateSupportOrientedProbeResult,
        ...,
    ]
    checks: tuple[tuple[str, bool], ...]
    D_R_accessed: bool = False
    D_V_accessed: bool = False
    D_T_accessed: bool = False
    performance_claim_supported: bool = False

    def __post_init__(self) -> None:
        self.verify()

    def verify(self) -> None:
        if (
            not isinstance(
                self.support_oriented_receipt,
                CoverageStateSupportOrientedDatasetFreeReceipt,
            )
            or self.support_oriented_receipt.receipt_fingerprint
            != self.support_oriented_receipt_fingerprint
            or self.support_oriented_receipt_fingerprint
            != COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_FINGERPRINT
            or not isinstance(
                self.architecture_probe,
                CoverageStatePhasePreservingArchitectureProbeResult,
            )
            or not isinstance(self.geometry_probes, tuple)
            or not isinstance(self.training_results, tuple)
            or not isinstance(self.support_oriented_probes, tuple)
            or not isinstance(self.checks, tuple)
            or not all(
                isinstance(
                    value,
                    CoverageStatePhasePreservingGeometryProbeResult,
                )
                for value in self.geometry_probes
            )
            or not all(
                isinstance(value, CoverageStateDatasetFreeTrainingResult)
                for value in self.training_results
            )
            or not all(
                isinstance(value, CoverageStateSupportOrientedProbeResult)
                for value in self.support_oriented_probes
            )
            or not all(
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], str)
                and type(item[1]) is bool
                for item in self.checks
            )
            or not self.checks
            or len({name for name, _ in self.checks}) != len(self.checks)
        ):
            raise ValueError(
                "phase-preserving dataset-free receipt is incomplete"
            )
        if (
            self.D_R_accessed
            or self.D_V_accessed
            or self.D_T_accessed
            or self.performance_claim_supported
        ):
            raise ValueError(
                "phase-preserving dataset-free receipt exceeds its scope"
            )
        recomputed = (
            recompute_coverage_state_phase_preserving_dataset_free_checks(
                self.support_oriented_receipt,
                self.architecture_probe,
                self.geometry_probes,
                self.training_results,
                self.support_oriented_probes,
            )
        )
        if self.checks != recomputed:
            raise ValueError(
                "phase-preserving dataset-free checks changed"
            )

    @property
    def all_pass(self) -> bool:
        self.verify()
        return all(passed for _, passed in self.checks)

    @property
    def status(self) -> str:
        return (
            "PHASE_PRESERVING_DATASET_FREE_GATE_PASS"
            if self.all_pass
            else "PHASE_PRESERVING_DATASET_FREE_GATE_FAIL"
        )

    def canonical_payload(self) -> dict[str, object]:
        self.verify()
        return {
            "schema_version": (
                COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SCHEMA
            ),
            "scope": {
                "sizes": list(
                    COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SIZES
                ),
                "seeds": list(
                    COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
                ),
                "architecture_probe_count": 1,
                "geometry_probe_count": len(self.geometry_probes),
                "support_oriented_probe_count": len(
                    self.support_oriented_probes
                ),
                "training_updates_per_objective": (
                    COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
                ),
                "training_result_count": len(self.training_results),
                "objective_suite": [
                    value.value
                    for value in (
                        COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
                    )
                ],
                "fixture_source": "generated_in_memory_only",
                "training_role": (
                    "computational_learnability_and_early_gradient_only"
                ),
                "changed_model_coordinate": (
                    "lossless_phase_preserving_occupancy_encoding_only"
                ),
                "target_sdf_measure_policy": "unchanged_and_fingerprinted",
                "no_gate_hyperparameter_search": True,
            },
            "support_oriented_dataset_free_receipt_fingerprint": (
                self.support_oriented_receipt_fingerprint
            ),
            "architecture_probe": self.architecture_probe.canonical_payload(),
            "geometry_probes": [
                value.canonical_payload() for value in self.geometry_probes
            ],
            "training_results": [
                value.canonical_payload() for value in self.training_results
            ],
            "support_oriented_probes": [
                value.canonical_payload()
                for value in self.support_oriented_probes
            ],
            "checks": {name: passed for name, passed in self.checks},
            "status": self.status,
            "all_pass": self.all_pass,
            "data_access": {
                "D_R_accessed": self.D_R_accessed,
                "D_V_accessed": self.D_V_accessed,
                "D_T_accessed": self.D_T_accessed,
            },
            "performance_claim_supported": self.performance_claim_supported,
            "formal_training_authorized": False,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_coverage_state_phase_preserving_dataset_free_gate(
) -> CoverageStatePhasePreservingDatasetFreeReceipt:
    """Run PPCE structural and short-compute gates without dataset access."""

    support_oriented_receipt = (
        run_coverage_state_support_oriented_dataset_free_gate()
    )
    if (
        support_oriented_receipt.receipt_fingerprint
        != COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_FINGERPRINT
    ):
        raise RuntimeError("support-oriented dataset-free receipt changed")
    architecture_probe = _run_phase_preserving_architecture_probe()
    geometry_probes = tuple(
        _run_phase_preserving_geometry_probe(seed=seed)
        for seed in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
    )
    training_results = tuple(
        result
        for seed in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
        for result in _run_phase_preserving_training_matrix(seed)
    )
    support_oriented_probes = tuple(
        _run_support_oriented_probe(size=size, seed=seed)
        for seed in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
        for size in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SIZES
    )
    checks = recompute_coverage_state_phase_preserving_dataset_free_checks(
        support_oriented_receipt,
        architecture_probe,
        geometry_probes,
        training_results,
        support_oriented_probes,
    )
    return CoverageStatePhasePreservingDatasetFreeReceipt(
        support_oriented_receipt=support_oriented_receipt,
        support_oriented_receipt_fingerprint=(
            support_oriented_receipt.receipt_fingerprint
        ),
        architecture_probe=architecture_probe,
        geometry_probes=geometry_probes,
        training_results=training_results,
        support_oriented_probes=support_oriented_probes,
        checks=checks,
    )


__all__ = [
    "COVERAGE_STATE_DATASET_FREE_CASES",
    "COVERAGE_STATE_DATASET_FREE_SCHEMA",
    "COVERAGE_STATE_DATASET_FREE_SEEDS",
    "COVERAGE_STATE_DATASET_FREE_SIZES",
    "COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES",
    "COVERAGE_STATE_LEGACY_DATASET_FREE_FINGERPRINT",
    "COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SCHEMA",
    "COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS",
    "COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SIZES",
    "COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT",
    "COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_FINGERPRINT",
    "COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_SCHEMA",
    "CoverageStateDatasetFreeCaseResult",
    "CoverageStateCompletionRootProbeResult",
    "CoverageStateDatasetFreeReceipt",
    "CoverageStateDatasetFreeTrainingResult",
    "CoverageStatePhasePreservingArchitectureProbeResult",
    "CoverageStatePhasePreservingDatasetFreeReceipt",
    "CoverageStatePhasePreservingGeometryProbeResult",
    "CoverageStateSupportOrientedDatasetFreeReceipt",
    "CoverageStateSupportOrientedProbeResult",
    "recompute_coverage_state_dataset_free_checks",
    "recompute_coverage_state_phase_preserving_dataset_free_checks",
    "recompute_coverage_state_support_oriented_dataset_free_checks",
    "run_coverage_state_dataset_free_gate",
    "run_coverage_state_phase_preserving_dataset_free_gate",
    "run_coverage_state_support_oriented_dataset_free_gate",
]
