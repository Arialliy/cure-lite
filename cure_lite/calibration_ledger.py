"""Exact, reusable candidate ledgers for validation-threshold calibration.

The original scalar calibration functions are intentionally simple, but a
complete threshold grid repeatedly rebuilds the same fixed occupancy anchor,
GT components, anchor miss sets, and reachability diagnostics.  This module
prepares that immutable per-image context once and then evaluates independent
Base@B and residual candidates either serially or in isolated worker processes.

This is an acceleration layer, not a second metric implementation.  Both the
legacy evaluators and this ledger delegate to the same metric core.  Candidate
results therefore retain exact :class:`AggregateEvaluation` dataclass equality
rather than merely agreeing within a floating-point tolerance.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import isfinite
import multiprocessing as mp
from multiprocessing.context import BaseContext
import os
from typing import Callable, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor

from .calibration import (
    CalibrationSample,
    FalseAlarmBudget,
    ThresholdSelection,
    threshold_selection_key,
)
from .config import MatchConfig, OccupancyConfig
from .instances import instances_from_binary_mask
from .matching import match_components
from .metrics import (
    AggregateEvaluation,
    ImageEvaluation,
    aggregate_evaluations,
    evaluate_binary_prediction_from_instances,
    full_pipeline_reachable_anchor_miss_ids_from_instances,
)
from .occupancy import build_occupancy
from .types import InstanceMap


CandidateMode = Literal["base", "residual"]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True, eq=False)
class PreparedCalibrationRow:
    """One normalized D_V row with its fixed anchor/GT context."""

    sample_id: str
    base_probability: Tensor
    occupancy: Tensor
    anchor_instances: InstanceMap
    gt_mask: Tensor
    gt_instances: InstanceMap
    anchor_miss_ids: frozenset[int]
    reachable_anchor_miss_ids: frozenset[int]
    anchor_evaluation: ImageEvaluation


@dataclass(frozen=True, slots=True, eq=False)
class PreparedCalibrationContext:
    """Canonical fixed-anchor context shared by every candidate threshold."""

    rows: tuple[PreparedCalibrationRow, ...]
    occupancy_config: OccupancyConfig
    match_config: MatchConfig
    anchor_metrics: AggregateEvaluation

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row.sample_id for row in self.rows)

    def row_by_sample_id(self, sample_id: str) -> PreparedCalibrationRow:
        """Return one prepared row without exposing an unsealed mutable map."""

        matches = tuple(row for row in self.rows if row.sample_id == sample_id)
        if len(matches) != 1:
            raise KeyError(f"unknown prepared sample {sample_id!r}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """One method/threshold row in a deterministic candidate ledger."""

    method: str
    mode: CandidateMode
    threshold: float | None
    metrics: AggregateEvaluation

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("candidate method must be a non-empty string")
        if self.mode not in {"base", "residual"}:
            raise ValueError("candidate mode must be 'base' or 'residual'")
        if self.mode == "base" and self.threshold is None:
            raise ValueError("a base candidate cannot use a null threshold")
        if self.threshold is not None:
            if isinstance(self.threshold, bool):
                raise TypeError("candidate threshold may not be bool")
            value = float(self.threshold)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("candidate threshold must be finite and in [0,1]")
            object.__setattr__(self, "threshold", value)
        if not isinstance(self.metrics, AggregateEvaluation):
            raise TypeError("candidate metrics must be AggregateEvaluation")


@dataclass(frozen=True, slots=True)
class CalibrationCandidateLedger:
    """Ordered exact results for one Base@B grid and any residual methods."""

    base_method: str
    anchor_threshold: float
    anchor_metrics: AggregateEvaluation
    entries: tuple[CandidateEvaluation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base_method, str) or not self.base_method:
            raise ValueError("base_method must be a non-empty string")
        if isinstance(self.anchor_threshold, bool):
            raise TypeError("anchor_threshold may not be bool")
        threshold = float(self.anchor_threshold)
        if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("anchor_threshold must be finite and in [0,1]")
        object.__setattr__(self, "anchor_threshold", threshold)
        if not isinstance(self.anchor_metrics, AggregateEvaluation):
            raise TypeError("anchor_metrics must be AggregateEvaluation")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("candidate ledger must contain entries")
        keys: set[tuple[str, float | None]] = set()
        for entry in self.entries:
            if not isinstance(entry, CandidateEvaluation):
                raise TypeError("ledger entries must be CandidateEvaluation")
            key = (entry.method, entry.threshold)
            if key in keys:
                raise ValueError("candidate ledger contains a duplicate method/threshold")
            keys.add(key)
        anchor = self.get(self.base_method, self.anchor_threshold)
        if anchor.mode != "base" or anchor.metrics != self.anchor_metrics:
            raise ValueError("base anchor entry differs from anchor_metrics")
        for method in self.methods:
            rows = self.for_method(method)
            modes = {row.mode for row in rows}
            if len(modes) != 1:
                raise ValueError("one method cannot mix base and residual candidates")
            if method != self.base_method:
                null_rows = [row for row in rows if row.threshold is None]
                if len(null_rows) != 1 or null_rows[0].metrics != self.anchor_metrics:
                    raise ValueError(
                        "each residual method needs one null anchor candidate"
                    )

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(entry.method for entry in self.entries))

    def for_method(self, method: str) -> tuple[CandidateEvaluation, ...]:
        rows = tuple(entry for entry in self.entries if entry.method == method)
        if not rows:
            raise KeyError(f"unknown candidate method {method!r}")
        return rows

    def get(self, method: str, threshold: float | None) -> CandidateEvaluation:
        resolved = None if threshold is None else float(threshold)
        matches = tuple(
            entry
            for entry in self.entries
            if entry.method == method and entry.threshold == resolved
        )
        if len(matches) != 1:
            raise KeyError(f"unknown candidate {(method, resolved)!r}")
        return matches[0]

    def select(
        self,
        method: str,
        budget: FalseAlarmBudget,
    ) -> ThresholdSelection:
        """Apply the deterministic standard-metric budget rule to one ledger."""

        if not isinstance(budget, FalseAlarmBudget):
            raise TypeError("budget must be a FalseAlarmBudget")
        if not budget.accepts(self.anchor_metrics):
            raise ValueError(
                "the preregistered FA budget cannot be below fixed anchor occupancy FA"
            )
        rows = self.for_method(method)
        feasible = [row for row in rows if budget.accepts(row.metrics)]
        if not feasible:  # defended by the required anchor/null candidates
            mode = rows[0].mode
            return ThresholdSelection(
                threshold=None,
                metrics=None,
                feasible=False,
                reason=f"no {mode} threshold satisfies the preregistered FA budget",
            )
        if rows[0].mode == "base":
            selected = max(
                feasible,
                key=lambda row: threshold_selection_key(
                    row.threshold,
                    row.metrics,
                ),
            )
        else:
            selected = max(
                feasible,
                key=lambda row: threshold_selection_key(
                    row.threshold,
                    row.metrics,
                ),
            )
        return ThresholdSelection(selected.threshold, selected.metrics, True)


@dataclass(frozen=True, slots=True, eq=False)
class _PreparedResidualMethod:
    method: str
    probabilities: tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class _CandidateTask:
    method: str
    mode: CandidateMode
    threshold: float


_WORKER_CONTEXT: PreparedCalibrationContext | None = None
_WORKER_RESIDUALS: dict[str, _PreparedResidualMethod] = {}


def _canonical_thresholds(
    values: Iterable[float],
    *,
    allow_empty: bool,
) -> tuple[float, ...]:
    result = tuple(sorted(set(float(value) for value in values)))
    if not result and not allow_empty:
        raise ValueError("threshold candidates are empty")
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in result):
        raise ValueError("threshold candidates must be finite and in [0,1]")
    return result


def prepare_calibration_context(
    base_samples: Sequence[CalibrationSample],
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
) -> PreparedCalibrationContext:
    """Normalize D_V once and construct its fixed anchor/GT state once."""

    if not isinstance(occupancy_config, OccupancyConfig):
        raise TypeError("occupancy_config must be an OccupancyConfig")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be a MatchConfig")
    if not base_samples:
        raise ValueError("at least one calibration sample is required")

    rows: list[PreparedCalibrationRow] = []
    seen: set[str] = set()
    for sample in base_samples:
        if not isinstance(sample, CalibrationSample):
            raise TypeError("base_samples must contain CalibrationSample values")
        base, _, gt = sample.normalized()
        if sample.sample_id in seen:
            raise ValueError(f"duplicate sample_id {sample.sample_id!r}")
        seen.add(sample.sample_id)

        occupancy, anchor_instances = build_occupancy(base, occupancy_config)
        gt_instances = instances_from_binary_mask(gt, connectivity=8, min_area=1)
        anchor_match = match_components(anchor_instances, gt_instances, match_config)
        anchor_misses = frozenset(anchor_match.unmatched_gt_ids)
        reachable = full_pipeline_reachable_anchor_miss_ids_from_instances(
            occupancy,
            gt,
            anchor_instances,
            gt_instances,
            match_config,
        )
        anchor_evaluation = evaluate_binary_prediction_from_instances(
            occupancy,
            gt,
            anchor_instances,
            gt_instances,
            match_config,
            anchor_miss_ids=anchor_misses,
            reachable_anchor_miss_ids=reachable,
        )
        rows.append(
            PreparedCalibrationRow(
                sample_id=sample.sample_id,
                base_probability=base,
                occupancy=occupancy,
                anchor_instances=anchor_instances,
                gt_mask=gt,
                gt_instances=gt_instances,
                anchor_miss_ids=anchor_misses,
                reachable_anchor_miss_ids=reachable,
                anchor_evaluation=anchor_evaluation,
            )
        )

    prepared_rows = tuple(rows)
    return PreparedCalibrationContext(
        rows=prepared_rows,
        occupancy_config=occupancy_config,
        match_config=match_config,
        anchor_metrics=aggregate_evaluations(
            row.anchor_evaluation for row in prepared_rows
        ),
    )


def _prepare_residual_method(
    context: PreparedCalibrationContext,
    method: str,
    samples: Sequence[CalibrationSample],
) -> _PreparedResidualMethod:
    if not isinstance(method, str) or not method:
        raise ValueError("residual method names must be non-empty strings")
    if len(samples) != len(context.rows):
        raise ValueError(f"{method} sample count differs from the base context")
    probabilities: list[Tensor] = []
    for expected, sample in zip(context.rows, samples, strict=True):
        if not isinstance(sample, CalibrationSample):
            raise TypeError("residual methods must contain CalibrationSample values")
        base, residual, gt = sample.normalized()
        if sample.sample_id != expected.sample_id:
            raise ValueError(f"{method} sample order differs from the base context")
        if not torch.equal(base, expected.base_probability):
            raise ValueError(f"{method} base probability differs from the base context")
        if not torch.equal(gt, expected.gt_mask):
            raise ValueError(f"{method} GT mask differs from the base context")
        probabilities.append(residual)
    return _PreparedResidualMethod(method, tuple(probabilities))


def _evaluate_task_with_context(
    task: _CandidateTask,
    context: PreparedCalibrationContext,
    residuals: Mapping[str, _PreparedResidualMethod],
) -> CandidateEvaluation:
    records: list[ImageEvaluation] = []
    prepared_residual = residuals.get(task.method)
    if task.mode == "residual" and prepared_residual is None:
        raise RuntimeError(f"residual method {task.method!r} was not prepared")

    for index, row in enumerate(context.rows):
        residual_mask: Tensor | None
        if task.mode == "base":
            prediction = row.base_probability >= task.threshold
            residual_mask = None
        else:
            assert prepared_residual is not None
            residual_mask = (
                prepared_residual.probabilities[index] >= task.threshold
            ) & ~row.occupancy
            prediction = row.occupancy | residual_mask
        pred_instances = instances_from_binary_mask(
            prediction,
            connectivity=8,
            min_area=1,
        )
        records.append(
            evaluate_binary_prediction_from_instances(
                prediction,
                row.gt_mask,
                pred_instances,
                row.gt_instances,
                context.match_config,
                anchor_miss_ids=row.anchor_miss_ids,
                reachable_anchor_miss_ids=row.reachable_anchor_miss_ids,
                residual_mask=residual_mask,
            )
        )
    return CandidateEvaluation(
        method=task.method,
        mode=task.mode,
        threshold=task.threshold,
        metrics=aggregate_evaluations(records),
    )


def _install_worker_context(
    context: PreparedCalibrationContext,
    residuals: dict[str, _PreparedResidualMethod],
) -> None:
    global _WORKER_CONTEXT, _WORKER_RESIDUALS
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_CONTEXT = context
    _WORKER_RESIDUALS = residuals


def _evaluate_worker_task(task: _CandidateTask) -> CandidateEvaluation:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("candidate-ledger worker context was not initialized")
    return _evaluate_task_with_context(task, _WORKER_CONTEXT, _WORKER_RESIDUALS)


def evaluate_candidate_ledger(
    context: PreparedCalibrationContext,
    residual_samples_by_method: Mapping[str, Sequence[CalibrationSample]],
    *,
    base_thresholds: Iterable[float],
    residual_thresholds_by_method: Mapping[str, Iterable[float]],
    base_method: str = "Base@B",
    max_workers: int = 1,
    mp_context: BaseContext | str | None = None,
    progress: ProgressCallback | None = None,
) -> CalibrationCandidateLedger:
    """Evaluate all unique numeric candidates and reuse the exact anchor result.

    ``residual_samples_by_method`` is intentionally open-ended: callers may
    supply F, exposure-matched F, U, or later controls without changing this
    evaluator.  Each residual method receives an explicit null candidate whose
    metrics are the fixed anchor metrics.  Numeric candidate tasks are
    independent and reconstructed in deterministic input order after parallel
    execution.
    """

    if not isinstance(context, PreparedCalibrationContext):
        raise TypeError("context must be a PreparedCalibrationContext")
    if not isinstance(base_method, str) or not base_method:
        raise ValueError("base_method must be a non-empty string")
    if base_method in residual_samples_by_method:
        raise ValueError("base_method cannot also be a residual method")
    if set(residual_samples_by_method) != set(residual_thresholds_by_method):
        raise ValueError("residual methods and residual threshold grids differ")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError("max_workers must be an integer")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    base_grid = tuple(
        threshold
        for threshold in sorted(
            set(
                (
                    *_canonical_thresholds(base_thresholds, allow_empty=False),
                    context.occupancy_config.threshold,
                )
            )
        )
        if threshold <= context.occupancy_config.threshold
    )
    method_order = tuple(residual_samples_by_method)
    if any(not isinstance(method, str) or not method for method in method_order):
        raise ValueError("residual method names must be non-empty strings")
    residual_grids = {
        method: _canonical_thresholds(
            residual_thresholds_by_method[method],
            allow_empty=True,
        )
        for method in method_order
    }
    prepared_residuals = {
        method: _prepare_residual_method(
            context,
            method,
            residual_samples_by_method[method],
        )
        for method in method_order
    }

    anchor_threshold = context.occupancy_config.threshold
    tasks: list[_CandidateTask] = []
    for threshold in base_grid:
        if threshold != anchor_threshold:
            tasks.append(_CandidateTask(base_method, "base", threshold))
    for method in method_order:
        tasks.extend(
            _CandidateTask(method, "residual", threshold)
            for threshold in residual_grids[method]
        )

    completed: dict[tuple[str, float], CandidateEvaluation] = {}
    total = len(tasks)
    if max_workers == 1 or total <= 1:
        for done, task in enumerate(tasks, start=1):
            result = _evaluate_task_with_context(task, context, prepared_residuals)
            completed[(task.method, task.threshold)] = result
            if progress is not None:
                progress(done, total)
    elif tasks:
        worker_count = min(max_workers, total, os.cpu_count() or 1)
        if mp_context is None:
            # Calibration normally follows CUDA decoder training.  ``spawn``
            # avoids inheriting a live CUDA runtime into CPU metric workers.
            process_context = mp.get_context("spawn")
        elif isinstance(mp_context, str):
            try:
                process_context = mp.get_context(mp_context)
            except ValueError as error:
                raise ValueError(
                    f"unsupported multiprocessing context {mp_context!r}"
                ) from error
        elif isinstance(mp_context, BaseContext):
            process_context = mp_context
        else:
            raise TypeError("mp_context must be a multiprocessing context or name")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=process_context,
            initializer=_install_worker_context,
            initargs=(context, prepared_residuals),
        ) as executor:
            futures = {
                executor.submit(_evaluate_worker_task, task): task for task in tasks
            }
            for done, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                result = future.result()
                completed[(task.method, task.threshold)] = result
                if progress is not None:
                    progress(done, total)
    if len(completed) != total:
        raise RuntimeError("parallel candidate ledger is incomplete")

    entries: list[CandidateEvaluation] = []
    for threshold in base_grid:
        if threshold == anchor_threshold:
            entries.append(
                CandidateEvaluation(
                    base_method,
                    "base",
                    threshold,
                    context.anchor_metrics,
                )
            )
        else:
            entries.append(completed[(base_method, threshold)])
    for method in method_order:
        entries.append(
            CandidateEvaluation(
                method,
                "residual",
                None,
                context.anchor_metrics,
            )
        )
        entries.extend(
            completed[(method, threshold)] for threshold in residual_grids[method]
        )

    return CalibrationCandidateLedger(
        base_method=base_method,
        anchor_threshold=anchor_threshold,
        anchor_metrics=context.anchor_metrics,
        entries=tuple(entries),
    )


__all__ = [
    "CalibrationCandidateLedger",
    "CandidateEvaluation",
    "PreparedCalibrationContext",
    "PreparedCalibrationRow",
    "evaluate_candidate_ledger",
    "prepare_calibration_context",
]
