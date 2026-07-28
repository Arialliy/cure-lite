from __future__ import annotations

import inspect

import pytest
import torch

from cure_lite.calibration import CalibrationSample, FalseAlarmBudget
from cure_lite.config import MatchConfig, OccupancyConfig
from cure_lite.experiment import coverage_state_paet_formal_evaluation as module
from cure_lite.experiment.coverage_state_paet_formal_evaluation import (
    PAET_FORMAL_BASE_THRESHOLD_GRID,
    PAETFormalArtifactBinding,
    PAETFormalDVEvaluationResult,
    fixed_paet_completion,
)


def test_fixed_paet_completion_uses_strict_negative_and_masks_occupancy() -> None:
    field = torch.tensor([[[[-1.0, -0.0, 0.0, 1.0, -2.0]]]])
    occupancy = torch.tensor([[[[False, False, False, False, True]]]])
    assert fixed_paet_completion(field, occupancy).tolist() == [
        [[[True, False, False, False, False]]]
    ]


def test_fixed_paet_completion_rejects_nonfinite_and_shape_drift() -> None:
    occupancy = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="finite"):
        fixed_paet_completion(torch.full((1, 1, 2, 2), float("nan")), occupancy)
    with pytest.raises(TypeError, match="shape"):
        fixed_paet_completion(
            torch.zeros((1, 1, 2, 2)),
            torch.zeros((1, 1, 2, 1), dtype=torch.bool),
        )


def test_operating_points_search_only_base_at_b_on_existing_grid() -> None:
    base = torch.zeros((1, 1, 8, 8))
    base[0, 0, 1, 1], base[0, 0, 6, 6] = 0.9, 0.6
    gt = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
    gt[0, 0, 1, 1] = gt[0, 0, 6, 6] = True
    completion = torch.zeros_like(base)
    completion[0, 0, 6, 6] = 1.0
    result = module._evaluate_fixed_operating_points(
        (CalibrationSample("synthetic", base, torch.zeros_like(base), gt),),
        (CalibrationSample("synthetic", base, completion, gt),),
        occupancy_config=OccupancyConfig(threshold=0.72),
        match_config=MatchConfig(),
        base_threshold_grid=PAET_FORMAL_BASE_THRESHOLD_GRID,
        budget=FalseAlarmBudget(1e-4, 100.0, 1e-4, 0.99),
    )
    assert result.base_at_b_selected_threshold == 0.6
    assert result.base_candidate_ledger.methods == ("Base@B",)


def test_stage_a_grid_is_rehashed_and_not_just_a_constant() -> None:
    assert module._verified_stage_a_base_grid() == PAET_FORMAL_BASE_THRESHOLD_GRID


def test_public_constructors_cannot_forge_dv_authority() -> None:
    with pytest.raises(PermissionError, match="strict factory"):
        PAETFormalArtifactBinding(
            seed=42,
            epochs=800,
            steps_per_epoch=40,
            completed_updates=32_000,
            trained_from_scratch=True,
            resumed=False,
            runtime_splits=("D_R",),
            artifact_fingerprint="a" * 64,
            artifact_receipt_sha256="a" * 64,
            model_state_fingerprint="a" * 64,
            model_config_fingerprint="a" * 64,
            formal_training_protocol_fingerprint="a" * 64,
            formal_schedule_fingerprint="a" * 64,
            formal_training_result_fingerprint="a" * 64,
            source_closure_fingerprint="a" * 64,
            source_closure_manifest_sha256="a" * 64,
            source_closure_archive_sha256="a" * 64,
            source_closure_file_count=1,
            structural_source_receipt_fingerprint="a" * 64,
            formal_attempt_complete_fingerprint="a" * 64,
            manifest_fingerprint="a" * 64,
            manifest_file_sha256="a" * 64,
            preprocessing_fingerprint="a" * 64,
            base_fingerprint="a" * 64,
            base_state_fingerprint="a" * 64,
            stage_a_config_sha256="a" * 64,
            comparison_protocol_fingerprint="a" * 64,
        )
    with pytest.raises(TypeError):
        PAETFormalDVEvaluationResult()  # type: ignore[call-arg]


def test_builder_and_evaluator_accept_only_sealed_objects() -> None:
    with pytest.raises(TypeError, match="PAETFormalArtifactBinding"):
        module.build_paet_fixed_d_v_samples(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PAETFixedDVSamples"):
        module.evaluate_paet_formal_d_v(object(), object())  # type: ignore[arg-type]


def test_binding_factory_rejects_unloaded_or_unsealed_sources() -> None:
    with pytest.raises(TypeError, match="strictly Loaded"):
        module.bind_paet_formal_artifact(
            object(), object(), object()  # type: ignore[arg-type]
        )


def test_public_api_has_no_threshold_or_unbound_dv_inputs() -> None:
    build = inspect.signature(module.build_paet_fixed_d_v_samples)
    evaluate = inspect.signature(module.evaluate_paet_formal_d_v)
    assert tuple(build.parameters) == ("artifact_binding", "batch_size")
    assert tuple(evaluate.parameters) == ("samples", "artifact_binding")
    assert "field_threshold" not in str(build)
    assert "residual_threshold" not in str(evaluate)
