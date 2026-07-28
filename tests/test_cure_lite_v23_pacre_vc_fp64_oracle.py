from __future__ import annotations

import torch

from cure_lite_v23.factory import build_pacre_vc_training_model
from cure_lite_v23.numerical_diagnostics import (
    PACRE_VC_FIXED_READOUT_POLICY,
    PACRE_VC_FP64_ORACLE_POLICY,
    complete_swallow_observation,
    run_pacre_fp64_oracle,
)
from cure_lite_v23.pacre_vc import (
    CoverageStatePACREVerifierCorrectedConfig,
)


def _model_and_fields():
    torch.manual_seed(230401)
    model = build_pacre_vc_training_model(
        CoverageStatePACREVerifierCorrectedConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    generator = torch.Generator().manual_seed(230402)
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
        > 0.7
    )
    return model, model.forward_fields(feature, occupancy)


def test_oracle_integrity_is_separate_from_numerical_diagnostics() -> None:
    model, fields = _model_and_fields()

    report = run_pacre_fp64_oracle(model, fields)
    payload = report.canonical_payload()

    assert report.integrity.passed
    assert report.integrity.reference_finite
    assert report.integrity.zero_readout_lane_exact
    assert report.integrity.fixed_readout_policy_valid
    assert payload["policy"] == PACRE_VC_FP64_ORACLE_POLICY
    assert payload["integrity"]["passed"] is True
    assert (
        payload["integrity"]["error_threshold_gate_eligible"]
        is False
    )
    assert payload["numerical_diagnostics"]["gate_eligible"] is False


def test_zero_lane_is_preserved_and_fixed_lane_is_nonzero_functional_probe() -> None:
    model, fields = _model_and_fields()
    assert torch.count_nonzero(model.scalar_energy_weight) == 0

    report = run_pacre_fp64_oracle(model, fields)
    zero = report.numerical.zero_readout
    fixed = report.numerical.fixed_readout

    assert zero.readout_exact_zero
    assert zero.actual_energy_error.maximum_absolute_error == 0.0
    assert zero.flipped_energy_error.maximum_absolute_error == 0.0
    assert zero.interaction_error.maximum_absolute_error == 0.0
    assert zero.field_error.maximum_absolute_error == 0.0
    assert not fixed.readout_exact_zero
    expected_readout = tuple(
        float(value)
        for value in torch.linspace(
            0.5,
            1.5,
            4,
            dtype=torch.float32,
        )
    )
    assert tuple(
        float.fromhex(value) for value in fixed.readout_hex
    ) == expected_readout
    assert (
        fixed.actual_energy_swallow.eligible_element_count
        + fixed.flipped_energy_swallow.eligible_element_count
        > 0
    )
    assert "linspace_0.5_to_1.5" in PACRE_VC_FIXED_READOUT_POLICY


def test_oracle_and_fixed_readout_do_not_mutate_parameters_fields_or_grads() -> None:
    model, fields = _model_and_fields()
    parameters_before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    hidden_before = fields.actual_compatibility_hidden.detach().clone()
    field_before = fields.field.detach().clone()

    first = run_pacre_fp64_oracle(model, fields)
    second = run_pacre_fp64_oracle(model, fields)

    assert first.canonical_payload() == second.canonical_payload()
    assert first.integrity.model_state_unchanged
    assert first.integrity.gradient_buffers_unchanged
    assert torch.equal(hidden_before, fields.actual_compatibility_hidden)
    assert torch.equal(field_before, fields.field)
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            model.parameters(),
            strict=True,
        )
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_complete_swallow_uses_exact_zero_and_prerequisite_mask() -> None:
    actual = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32)
    reference = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    prerequisite = torch.tensor([True, True, False], dtype=torch.bool)

    observation = complete_swallow_observation(
        "synthetic_complete_swallow",
        actual,
        reference,
        prerequisite=prerequisite,
    )

    assert observation.gate_eligible is False
    assert observation.eligible_element_count == 1
    assert observation.swallowed_element_count == 1
    assert observation.maximum_swallowed_reference_magnitude == 1.0
    assert observation.maximum_swallowed_coordinate == (0,)


def test_joint_and_hidden_swallow_ledgers_are_present_but_non_gating() -> None:
    model, fields = _model_and_fields()

    numerical = run_pacre_fp64_oracle(model, fields).numerical
    observations = (
        numerical.actual_joint_swallow,
        numerical.flipped_joint_swallow,
        numerical.actual_hidden_swallow,
        numerical.flipped_hidden_swallow,
    )

    assert all(value.gate_eligible is False for value in observations)
    assert all(value.eligible_element_count >= 0 for value in observations)
    assert all(value.swallowed_element_count >= 0 for value in observations)
