from __future__ import annotations

import pytest
import torch

from cure_lite_v23.factory import build_pacre_vc_training_model
from cure_lite_v23.pacre_vc import (
    CoverageStatePACREVerifierCorrectedConfig,
)
from cure_lite_v24.factory import build_gcr_pacre_training_model
from cure_lite_v24.gcr_pacre import (
    CURELiteGatedCommonResidualPACRELevelSet,
    CoverageStateGCRPACREConfig,
)
from cure_lite_v24.oof_evaluation import (
    OOFConcreteEvaluator,
    OOF_BASE_A_ARM,
    OOF_G1_ARM,
    OOF_V23_ARM,
    OOF_V24_ARM,
    seal_oof_evaluation_dataset,
    seal_oof_evaluation_sample,
)


def _dataset(partition: str = "holdout"):
    gt = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    gt[:, :, 1, 1] = True
    row = seal_oof_evaluation_sample(
        sample_id="generated-sample",
        root_source_id="generated-root",
        base_probability=torch.zeros(1, 1, 4, 4, dtype=torch.float32),
        feature=torch.linspace(
            -1.0,
            1.0,
            8,
            dtype=torch.float32,
        ).reshape(1, 2, 2, 2),
        gt_mask=gt,
        valid_mask=torch.ones_like(gt),
        anchor_miss_ids=[1],
        reachable_anchor_miss_ids=[1],
    )
    return seal_oof_evaluation_dataset(
        fold_id=0,
        partition=partition,
        closure_fingerprint="a" * 64,
        rows=[row],
    )


def _padded_dataset(*, invalid_probability: float = 0.0):
    gt = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    gt[:, :, 0, 0] = True
    valid = torch.zeros_like(gt)
    valid[:, :, :2, :2] = True
    probability = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    probability[:, :, 2:, :] = invalid_probability
    row = seal_oof_evaluation_sample(
        sample_id="generated-padded",
        root_source_id="generated-root",
        base_probability=probability,
        feature=torch.linspace(
            -1.0,
            1.0,
            8,
            dtype=torch.float32,
        ).reshape(1, 2, 2, 2),
        gt_mask=gt,
        valid_mask=valid,
        anchor_miss_ids=[1],
        reachable_anchor_miss_ids=[1],
    )
    return seal_oof_evaluation_dataset(
        fold_id=0,
        partition="holdout",
        closure_fingerprint="b" * 64,
        rows=[row],
    )


def test_fixed_evaluator_emits_factual_statistics_and_complete_base_grid() -> None:
    evaluator = OOFConcreteEvaluator.fixed()
    holdout = _dataset()
    base = evaluator.evaluate_base(
        holdout,
        threshold=0.72,
        arm=OOF_BASE_A_ARM,
    )
    assert base.pooled_statistics.images == 1
    assert base.pooled_statistics.total_gt == 1
    threshold, rows = evaluator.select_base_b_train_only(
        _dataset("train")
    )
    assert len(rows) == 51
    assert threshold in tuple(index / 50 for index in range(51))


def test_fixed_model_evaluator_uses_same_v24_terminal_for_forced_g1() -> None:
    evaluator = OOFConcreteEvaluator.fixed()
    dataset = _dataset()
    v23 = build_pacre_vc_training_model(
        CoverageStatePACREVerifierCorrectedConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    v24 = build_gcr_pacre_training_model(
        CoverageStateGCRPACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    control = evaluator.evaluate_model(
        dataset,
        v23,
        arm=OOF_V23_ARM,
    )
    candidate = evaluator.evaluate_model(
        dataset,
        v24,
        arm=OOF_V24_ARM,
    )
    forced = evaluator.evaluate_model(
        dataset,
        v24,
        arm=OOF_G1_ARM,
        forced_unit_gate=True,
    )
    assert control.pooled_statistics.images == 1
    assert candidate.model_fingerprint == forced.model_fingerprint
    assert candidate.per_sample_rows[0]["role_statistics"]
    assert forced.per_sample_rows[0]["role_statistics"]


def test_valid_mask_controls_fa_denominator_and_field_identity() -> None:
    evaluator = OOFConcreteEvaluator.fixed()
    first = evaluator.evaluate_base(
        _padded_dataset(invalid_probability=0.0),
        threshold=0.72,
        arm=OOF_BASE_A_ARM,
    )
    second = evaluator.evaluate_base(
        _padded_dataset(invalid_probability=1.0),
        threshold=0.72,
        arm=OOF_BASE_A_ARM,
    )
    assert first.pooled_statistics.total_pixels == 4
    assert second.pooled_statistics.total_pixels == 4
    assert first.pooled_statistics.unmatched_pred_pixels == 0
    assert second.pooled_statistics.unmatched_pred_pixels == 0
    assert (
        first.per_sample_rows[0]["field_fingerprint"]
        == second.per_sample_rows[0]["field_fingerprint"]
    )


def test_invalid_only_g1_field_change_is_not_a_difference_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = OOFConcreteEvaluator.fixed()
    dataset = _padded_dataset()
    model = build_gcr_pacre_training_model(
        CoverageStateGCRPACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    natural = evaluator.evaluate_model(
        dataset,
        model,
        arm=OOF_V24_ARM,
    )

    def invalid_only(
        self,
        feature: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        field = self.forward_fields(feature, occupancy).field.clone()
        field[:, :, 2:, :] += 1.0
        return field

    monkeypatch.setattr(
        CURELiteGatedCommonResidualPACRELevelSet,
        "forward_forced_unit_gate",
        invalid_only,
    )
    forced = evaluator.evaluate_model(
        dataset,
        model,
        arm=OOF_G1_ARM,
        forced_unit_gate=True,
    )
    assert (
        natural.per_sample_rows[0]["field_fingerprint"]
        == forced.per_sample_rows[0]["field_fingerprint"]
    )
    assert (
        natural.per_sample_rows[0]["prediction_fingerprint"]
        == forced.per_sample_rows[0]["prediction_fingerprint"]
    )


def test_model_mode_is_restored_when_evaluation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = OOFConcreteEvaluator.fixed()
    model = build_gcr_pacre_training_model(
        CoverageStateGCRPACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    model.train(True)

    def fail(*_args, **_kwargs):
        raise RuntimeError("generated evaluator failure")

    monkeypatch.setattr(
        CURELiteGatedCommonResidualPACRELevelSet,
        "forward_fields",
        fail,
    )
    with pytest.raises(RuntimeError, match="generated evaluator failure"):
        evaluator.evaluate_model(
            _dataset(),
            model,
            arm=OOF_V24_ARM,
        )
    assert model.training is True
