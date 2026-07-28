from __future__ import annotations

import pytest
import torch

from cure_lite_v23.factory import build_pacre_vc_training_model
from cure_lite_v23.numerical_diagnostics import (
    PACRE_VC_LEGACY_ATOL,
    PACRE_VC_LEGACY_RTOL,
    PACRE_VC_NORMALIZED_ERROR_POLICY,
    PACRE_VC_RELATIVE_ERROR_POLICY,
    PACRE_VC_ULP_POLICY,
    fp32_ulp_distance,
    legacy_residual_lane,
    legacy_subtraction_diagnostics,
    normalized_absolute_error,
    reference_relative_error,
)
from cure_lite_v23.pacre_vc import (
    CoverageStatePACREVerifierCorrectedConfig,
)


def _model_and_fields():
    torch.manual_seed(230301)
    model = build_pacre_vc_training_model(
        CoverageStatePACREVerifierCorrectedConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    generator = torch.Generator().manual_seed(230302)
    feature = torch.randn(
        (1, 2, 3, 4),
        generator=generator,
        dtype=torch.float32,
    )
    occupancy = (
        torch.rand(
            (1, 1, 6, 8),
            generator=generator,
            dtype=torch.float32,
        )
        > 0.65
    )
    return model, model.forward_fields(feature, occupancy)


def test_unique_normalized_and_reference_relative_formulas() -> None:
    actual = torch.tensor([2.0, 1.0], dtype=torch.float32)
    reference = torch.tensor([1.0, 0.0], dtype=torch.float64)

    normalized = normalized_absolute_error(actual, reference)
    relative = reference_relative_error(actual, reference)

    assert normalized.dtype == torch.float64
    assert normalized.device.type == "cpu"
    assert normalized.tolist() == pytest.approx([0.25, 0.5])
    assert relative[0].item() == pytest.approx(1.0)
    assert relative[1].item() == pytest.approx(
        1.0 / torch.finfo(torch.float32).tiny
    )
    assert "one_plus_abs_actual" in PACRE_VC_NORMALIZED_ERROR_POLICY
    assert "max_abs_reference" in PACRE_VC_RELATIVE_ERROR_POLICY


def test_frozen_ulp_metric_handles_adjacency_sign_and_signed_zero() -> None:
    one = torch.tensor([1.0], dtype=torch.float32)
    next_one = torch.nextafter(
        one,
        torch.tensor([float("inf")], dtype=torch.float32),
    )
    negative_one = torch.tensor([-1.0], dtype=torch.float32)
    next_negative = torch.nextafter(
        negative_one,
        torch.tensor([float("-inf")], dtype=torch.float32),
    )
    positive_zero = torch.tensor([0.0], dtype=torch.float32)
    negative_zero = torch.tensor([-0.0], dtype=torch.float32)

    assert fp32_ulp_distance(one, next_one).item() == 1
    assert fp32_ulp_distance(negative_one, next_negative).item() == 1
    assert fp32_ulp_distance(positive_zero, negative_zero).item() == 0
    assert "signed_zero_coalesced" in PACRE_VC_ULP_POLICY


def test_known_legacy_cancellation_counterexample_is_diagnostic_only() -> None:
    common = torch.tensor([100.0], dtype=torch.float32)
    residual = torch.tensor([0.1], dtype=torch.float32)
    specific = (common + residual).contiguous()

    lane = legacy_residual_lane(
        "known_counterexample",
        specific,
        common,
        residual,
    )

    assert lane.gate_eligible is False
    assert lane.decision_weight == 0
    assert lane.failed_under_v22_allclose_count == 1
    assert lane.error.maximum_absolute_error > PACRE_VC_LEGACY_ATOL
    assert lane.maximum_ulp_distance > 0
    payload = lane.canonical_payload()
    assert payload["legacy_rtol_hex"] == PACRE_VC_LEGACY_RTOL.hex()
    assert payload["legacy_atol_hex"] == PACRE_VC_LEGACY_ATOL.hex()


def test_complete_field_legacy_diagnostics_are_read_only() -> None:
    model, fields = _model_and_fields()
    parameters_before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    actual_before = fields.actual_specific_joint_affine.clone()
    flipped_before = fields.flipped_specific_joint_affine.clone()

    diagnostic = legacy_subtraction_diagnostics(fields)

    assert diagnostic.actual.gate_eligible is False
    assert diagnostic.flipped.gate_eligible is False
    assert diagnostic.canonical_payload()["decision_weight"] == 0
    assert torch.equal(
        actual_before,
        fields.actual_specific_joint_affine,
    )
    assert torch.equal(
        flipped_before,
        fields.flipped_specific_joint_affine,
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            model.parameters(),
            strict=True,
        )
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_diagnostic_formulas_reject_nonfinite_or_wrong_dtype_inputs() -> None:
    good = torch.ones(2, dtype=torch.float32)
    nonfinite = torch.tensor(
        [1.0, float("nan")],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="finite"):
        normalized_absolute_error(good, nonfinite)
    with pytest.raises(ValueError, match="aligned FP32"):
        fp32_ulp_distance(good, good.to(torch.float64))
    with pytest.raises(TypeError, match="FP32"):
        legacy_residual_lane(
            "wrong_dtype",
            good.to(torch.float64),
            good.to(torch.float64),
            good.to(torch.float64),
        )
