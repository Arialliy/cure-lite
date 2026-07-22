from __future__ import annotations

import inspect

import pytest
import torch

from cure_lite.calibration import CalibrationSample, FalseAlarmBudget
from cure_lite.config import DecoderConfig, MatchConfig, OccupancyConfig
from cure_lite.decoder import CURELiteDecoder
from cure_lite.experiment import evaluation_pipeline as pipeline
from cure_lite.experiment.evaluation_pipeline import (
    BoundDVThresholdProtocol,
    DevelopmentSplitAccess,
    Gate2SplitAccessError,
    calibration_sample_from_cached_base,
    evaluate_frozen_base_threshold,
    evaluate_frozen_residual_threshold,
    select_base_threshold_on_d_v,
    select_residual_threshold_on_d_v,
)
from cure_lite.splits import SplitManifest, SplitRecord


def _manifest() -> SplitManifest:
    return SplitManifest(
        dataset="toy",
        records=(
            SplitRecord("db", "D_B", "g-db", "db.png"),
            SplitRecord("dr", "D_R", "g-dr", "dr.png"),
            SplitRecord("dv", "D_V", "g-dv", "dv.png"),
            SplitRecord("dt", "D_T", "g-dt", "dt.png"),
        ),
    )


def test_gate_2_access_returns_only_d_r_or_d_v() -> None:
    manifest = _manifest()
    access = DevelopmentSplitAccess(manifest)
    assert [record.sample_id for record in access.records_for("D_R")] == ["dr"]
    assert [record.sample_id for record in access.records_for("D_V")] == ["dv"]
    with pytest.raises(Gate2SplitAccessError, match="only D_R or D_V"):
        access.records_for("D_B")
    with pytest.raises(Gate2SplitAccessError, match="D_T is sealed"):
        access.records_for("D_T")
    with pytest.raises(Gate2SplitAccessError, match="D_T is sealed"):
        pipeline.development_records(manifest, "D_T")


def test_cached_base_and_decoder_build_a_canonical_calibration_sample() -> None:
    decoder = CURELiteDecoder(DecoderConfig(feature_channels=2))
    for parameter in decoder.parameters():
        torch.nn.init.zeros_(parameter)
    decoder.train()
    base = torch.tensor(
        [[[[0.8, 0.1], [0.2, 0.3]]]], dtype=torch.float32
    )
    feature = torch.zeros(1, 2, 1, 1, dtype=torch.float32)
    gt = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

    sample = calibration_sample_from_cached_base(
        SplitRecord("sample", "D_V", "g-sample", "sample.png"),
        base,
        feature,
        decoder,
        gt,
        OccupancyConfig(threshold=0.5),
    )

    assert decoder.training  # caller mode is restored
    assert sample.base_probability.shape == (2, 2)
    assert sample.residual_probability.shape == (2, 2)
    assert sample.gt_mask.dtype == torch.bool
    assert sample.residual_probability[0, 0] == 0.0
    torch.testing.assert_close(
        sample.residual_probability[torch.tensor([[False, True], [True, True]])],
        torch.full((3,), 0.5),
    )
    assert all(parameter.grad is None for parameter in decoder.parameters())

    for sealed_split in ("D_B", "D_R", "D_T"):
        with pytest.raises(Gate2SplitAccessError, match="only for D_V"):
            calibration_sample_from_cached_base(
                SplitRecord(
                    f"sealed-{sealed_split}",
                    sealed_split,
                    f"g-{sealed_split}",
                    f"{sealed_split}.png",
                ),
                base,
                feature,
                decoder,
                gt,
                OccupancyConfig(threshold=0.5),
            )


def _recoverable_sample(sample_id: str) -> CalibrationSample:
    base = torch.zeros(3, 3, dtype=torch.float32)
    residual = torch.zeros(3, 3, dtype=torch.float32)
    residual[1, 1] = 0.9
    gt = torch.zeros(3, 3, dtype=torch.bool)
    gt[1, 1] = True
    return CalibrationSample(sample_id, base, residual, gt)


def _base_relaxation_sample(sample_id: str) -> CalibrationSample:
    base = torch.zeros(3, 3, dtype=torch.float32)
    base[1, 1] = 0.4
    residual = torch.zeros(3, 3, dtype=torch.float32)
    gt = torch.zeros(3, 3, dtype=torch.bool)
    gt[1, 1] = True
    return CalibrationSample(sample_id, base, residual, gt)


def test_d_v_selection_and_frozen_evaluation_reproves_grid_optimum() -> None:
    manifest = _manifest()
    access = DevelopmentSplitAccess(manifest)
    occupancy = OccupancyConfig(threshold=0.5)
    matching = MatchConfig()
    budget = FalseAlarmBudget(pixel_fa_budget=0.0)
    protocol = select_residual_threshold_on_d_v(
        access,
        [_recoverable_sample("dv")],
        [0.5, 0.8],
        occupancy,
        matching,
        budget,
    )
    assert isinstance(protocol, BoundDVThresholdProtocol)
    assert protocol.variant == "residual"
    assert protocol.selected_threshold == 0.8  # conservative tie-break
    assert protocol.manifest_fingerprint == manifest.fingerprint
    assert protocol.ordered_d_v_sample_ids == ("dv",)
    assert protocol.candidate_threshold_grid == (0.5, 0.8)
    assert protocol.occupancy_config == occupancy
    assert protocol.match_config == matching
    assert protocol.budget == budget
    assert len(protocol.sample_tensor_fingerprint) == 64
    assert len(protocol.receipt_fingerprint) == 64

    metrics = evaluate_frozen_residual_threshold(
        access,
        [_recoverable_sample("dv")],
        protocol,
    )
    assert metrics.pd == 1.0
    assert metrics.net_rmr == 1.0
    assert not metrics.budget_violation

    # A hand-built receipt may reproduce the metrics at one threshold while
    # lying about the deterministic conservative tie-break over the full grid.
    # Fixed evaluation must rerun the selector and reject that receipt.
    fabricated = BoundDVThresholdProtocol(
        variant=protocol.variant,
        manifest_fingerprint=protocol.manifest_fingerprint,
        ordered_d_v_sample_ids=protocol.ordered_d_v_sample_ids,
        sample_tensor_fingerprint=protocol.sample_tensor_fingerprint,
        candidate_threshold_grid=protocol.candidate_threshold_grid,
        occupancy_config=protocol.occupancy_config,
        match_config=protocol.match_config,
        budget=protocol.budget,
        selected_threshold=0.5,
        selected_metrics=protocol.selected_metrics,
    )
    with pytest.raises(RuntimeError, match="deterministic frozen-grid selection"):
        evaluate_frozen_residual_threshold(
            access,
            [_recoverable_sample("dv")],
            fabricated,
        )

    changed = _recoverable_sample("dv")
    changed_residual = changed.residual_probability.clone()
    changed_residual[0, 0] = 0.1
    changed = CalibrationSample(
        "dv", changed.base_probability, changed_residual, changed.gt_mask
    )
    with pytest.raises(RuntimeError, match="tensor content differs"):
        evaluate_frozen_residual_threshold(
            access,
            [changed],
            protocol,
        )

    swapped_manifest = SplitManifest(
        dataset="toy-changed",
        records=manifest.records,
    )
    with pytest.raises(RuntimeError, match="manifest fingerprint differs"):
        evaluate_frozen_residual_threshold(
            DevelopmentSplitAccess(swapped_manifest),
            [_recoverable_sample("dv")],
            protocol,
        )

    assert tuple(inspect.signature(evaluate_frozen_residual_threshold).parameters) == (
        "access",
        "samples",
        "protocol",
    )
    with pytest.raises(TypeError):
        evaluate_frozen_residual_threshold(
            access,
            [_recoverable_sample("dv")],
            protocol,
            MatchConfig(),  # type: ignore[call-arg]
        )

    with pytest.raises(ValueError, match="exactly match"):
        select_residual_threshold_on_d_v(
            access,
            [_recoverable_sample("dt")],
            [0.5],
            occupancy,
            matching,
            budget,
        )


def test_base_at_budget_selection_and_frozen_evaluation_are_d_v_only() -> None:
    manifest = _manifest()
    access = DevelopmentSplitAccess(manifest)
    occupancy = OccupancyConfig(threshold=0.5)
    matching = MatchConfig()
    budget = FalseAlarmBudget(pixel_fa_budget=0.0)
    protocol = select_base_threshold_on_d_v(
        access,
        [_base_relaxation_sample("dv")],
        [0.3, 0.5],
        occupancy,
        matching,
        budget,
    )
    assert protocol.variant == "base_at_budget"
    assert protocol.selected_threshold == 0.3
    assert protocol.candidate_threshold_grid == (0.3, 0.5)
    assert protocol.match_config == matching

    metrics = evaluate_frozen_base_threshold(
        access,
        [_base_relaxation_sample("dv")],
        protocol,
    )
    assert metrics.pd == 1.0
    assert not metrics.budget_violation

    assert tuple(inspect.signature(evaluate_frozen_base_threshold).parameters) == (
        "access",
        "samples",
        "protocol",
    )
    with pytest.raises(TypeError):
        evaluate_frozen_base_threshold(
            access,
            [_base_relaxation_sample("dv")],
            protocol,
            MatchConfig(),  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError, match="exactly match"):
        select_base_threshold_on_d_v(
            access,
            [_base_relaxation_sample("dt")],
            [0.3],
            occupancy,
            matching,
            budget,
        )
