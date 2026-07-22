from __future__ import annotations

import multiprocessing as mp

import pytest
import torch

import cure_lite.calibration as legacy
import cure_lite.calibration_ledger as ledger_module
from cure_lite.calibration import CalibrationSample, FalseAlarmBudget
from cure_lite.calibration_ledger import (
    CalibrationCandidateLedger,
    evaluate_candidate_ledger,
    prepare_calibration_context,
)
from cure_lite.config import MatchConfig, OccupancyConfig


def _samples() -> tuple[
    tuple[CalibrationSample, ...],
    dict[str, tuple[CalibrationSample, ...]],
]:
    base_rows: list[CalibrationSample] = []
    residual_rows: dict[str, list[CalibrationSample]] = {
        "F": [],
        "F×": [],
        "U": [],
    }

    base_1 = torch.zeros(8, 8, dtype=torch.float32)
    base_1[1:3, 1:3] = 0.9
    base_1[5:7, 5:7] = 0.4
    base_1[0, 7] = 0.3
    gt_1 = torch.zeros(8, 8, dtype=torch.bool)
    gt_1[1:3, 1:3] = True
    gt_1[5:7, 5:7] = True

    base_2 = torch.zeros(8, 8, dtype=torch.float32)
    base_2[2:4, 4:6] = 0.45
    base_2[7, 0] = 0.2
    gt_2 = torch.zeros(8, 8, dtype=torch.bool)
    gt_2[2:4, 4:6] = True

    for sample_id, base, gt in (
        ("s1", base_1, gt_1),
        ("s2", base_2, gt_2),
    ):
        base_rows.append(
            CalibrationSample(sample_id, base, torch.zeros_like(base), gt)
        )

        factual = torch.zeros_like(base)
        exposure_matched = torch.zeros_like(base)
        uniform = torch.zeros_like(base)
        if sample_id == "s1":
            factual[5:7, 5:7] = 0.65
            factual[0, 0] = 0.35
            exposure_matched[5:7, 5:7] = 0.8
            exposure_matched[4, 4] = 0.55
            uniform[5:7, 5:7] = 0.9
            uniform[0, 0:2] = 0.45
        else:
            factual[2:4, 4:6] = 0.55
            exposure_matched[2:4, 4:6] = 0.75
            exposure_matched[6, 6] = 0.4
            uniform[2:4, 4:6] = 0.85
            uniform[0:2, 0] = 0.5
        for method, probability in (
            ("F", factual),
            ("F×", exposure_matched),
            ("U", uniform),
        ):
            residual_rows[method].append(
                CalibrationSample(sample_id, base, probability, gt)
            )

    return tuple(base_rows), {
        method: tuple(rows) for method, rows in residual_rows.items()
    }


def _grids() -> tuple[tuple[float, ...], dict[str, tuple[float, ...]]]:
    return (
        (0.2, 0.4, 0.5, 0.7),
        {
            "F": (0.3, 0.5, 0.7, 1.0),
            "F×": (0.25, 0.55, 0.75),
            "U": (0.4, 0.6, 0.8),
        },
    )


def _legacy_candidate_metrics(
    entry: object,
    base_samples: tuple[CalibrationSample, ...],
    residual_samples: dict[str, tuple[CalibrationSample, ...]],
    occupancy: OccupancyConfig,
    matching: MatchConfig,
):
    anchors, reachable = legacy._fixed_anchor_state(
        base_samples,
        occupancy,
        matching,
    )
    assert hasattr(entry, "method")
    if entry.mode == "base":
        return legacy.evaluate_base_threshold(
            base_samples,
            entry.threshold,
            occupancy,
            matching,
            anchor_miss_ids_by_sample=anchors,
            reachable_anchor_miss_ids_by_sample=reachable,
        )
    return legacy.evaluate_residual_threshold(
        residual_samples[entry.method],
        entry.threshold,
        occupancy,
        matching,
        anchor_miss_ids_by_sample=anchors,
        reachable_anchor_miss_ids_by_sample=reachable,
    )


def test_every_candidate_and_selection_have_exact_legacy_dataclass_equality() -> None:
    base_samples, residual_samples = _samples()
    base_grid, residual_grids = _grids()
    occupancy = OccupancyConfig(threshold=0.5)
    matching = MatchConfig(max_distance=3.0)
    context = prepare_calibration_context(base_samples, occupancy, matching)

    ledger = evaluate_candidate_ledger(
        context,
        residual_samples,
        base_thresholds=base_grid,
        residual_thresholds_by_method=residual_grids,
    )

    assert isinstance(ledger, CalibrationCandidateLedger)
    assert ledger.methods == ("Base@B", "F", "F×", "U")
    assert ledger.for_method("F×")[0].threshold is None
    for entry in ledger.entries:
        exact = _legacy_candidate_metrics(
            entry,
            base_samples,
            residual_samples,
            occupancy,
            matching,
        )
        assert entry.metrics == exact

    budgets = (
        FalseAlarmBudget(
            pixel_fa_budget=1.0,
            component_fa_per_mp_budget=float("inf"),
            raw_background_fa_budget=1.0,
            minimum_retention=0.0,
        ),
        FalseAlarmBudget(
            pixel_fa_budget=0.0,
            component_fa_per_mp_budget=0.0,
            raw_background_fa_budget=0.0,
            minimum_retention=1.0,
        ),
    )
    for budget in budgets:
        expected_base = legacy.select_base_threshold_at_budget(
            base_samples,
            base_grid,
            occupancy,
            matching,
            budget,
        )
        assert ledger.select("Base@B", budget) == expected_base
        for method, samples in residual_samples.items():
            expected = legacy.select_residual_threshold(
                samples,
                residual_grids[method],
                occupancy,
                matching,
                budget,
            )
            assert ledger.select(method, budget) == expected


@pytest.mark.skipif("spawn" not in mp.get_all_start_methods(), reason="no spawn")
@pytest.mark.parametrize("context_argument", [None, "spawn"], ids=["default", "explicit"])
def test_spawn_parallel_candidate_ledger_is_exactly_equal_to_sequential(
    context_argument: str | None,
) -> None:
    base_samples, residual_samples = _samples()
    base_grid, residual_grids = _grids()
    context = prepare_calibration_context(
        base_samples,
        OccupancyConfig(threshold=0.5),
        MatchConfig(max_distance=3.0),
    )

    sequential = evaluate_candidate_ledger(
        context,
        residual_samples,
        base_thresholds=base_grid,
        residual_thresholds_by_method=residual_grids,
        max_workers=1,
    )
    progress: list[tuple[int, int]] = []
    parallel = evaluate_candidate_ledger(
        context,
        residual_samples,
        base_thresholds=base_grid,
        residual_thresholds_by_method=residual_grids,
        max_workers=2,
        mp_context=context_argument,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert parallel == sequential
    numeric_candidates = sum(
        entry.threshold is not None
        and not (
            entry.method == "Base@B"
            and entry.threshold == context.occupancy_config.threshold
        )
        for entry in parallel.entries
    )
    assert progress == [
        (done, numeric_candidates) for done in range(1, numeric_candidates + 1)
    ]


def test_gt_components_are_built_once_during_context_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_samples, _ = _samples()
    original = ledger_module.instances_from_binary_mask
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ledger_module, "instances_from_binary_mask", counted)
    context = prepare_calibration_context(
        base_samples,
        OccupancyConfig(threshold=0.5),
        MatchConfig(),
    )

    # Anchor components are returned by build_occupancy.  This module invokes
    # the CC8 constructor exactly once per GT and stores that map in the row.
    assert calls == len(base_samples)
    assert context.sample_ids == ("s1", "s2")
    assert context.row_by_sample_id("s1").gt_instances.ids == (1, 2)


def test_residual_binding_rejects_base_or_gt_drift() -> None:
    base_samples, residual_samples = _samples()
    base_grid, residual_grids = _grids()
    context = prepare_calibration_context(
        base_samples,
        OccupancyConfig(threshold=0.5),
        MatchConfig(),
    )
    changed = list(residual_samples["F"])
    first = changed[0]
    changed_base = first.base_probability.clone()
    changed_base[0, 0] = 0.99
    changed[0] = CalibrationSample(
        first.sample_id,
        changed_base,
        first.residual_probability,
        first.gt_mask,
    )
    drifted = {**residual_samples, "F": tuple(changed)}

    with pytest.raises(ValueError, match="base probability differs"):
        evaluate_candidate_ledger(
            context,
            drifted,
            base_thresholds=base_grid,
            residual_thresholds_by_method=residual_grids,
        )

    with pytest.raises(ValueError, match="threshold grids differ"):
        evaluate_candidate_ledger(
            context,
            residual_samples,
            base_thresholds=base_grid,
            residual_thresholds_by_method={"F": residual_grids["F"]},
        )
