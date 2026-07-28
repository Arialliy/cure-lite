from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite_v23.algebra_verifier import (
    PACRE_VC_SILU_REPLAY_FACTOR,
    validate_pacre_fields_contract,
    verify_exact_forward_replay,
    verify_pacre_v22_forward_fields,
)
from cure_lite_v23.factory import build_pacre_vc_training_model
from cure_lite_v23.pacre_vc import (
    CoverageStatePACREVerifierCorrectedConfig,
)


def _model_and_fields():
    torch.manual_seed(230101)
    model = build_pacre_vc_training_model(
        CoverageStatePACREVerifierCorrectedConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    generator = torch.Generator().manual_seed(230102)
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


def test_exact_replay_covers_flip_and_complete_downstream_chain() -> None:
    model, fields = _model_and_fields()

    report = verify_pacre_v22_forward_fields(model, fields)

    assert report.passed
    names = tuple(check.name for check in report.exact_replay.checks)
    assert names == (
        "phase_mean_forward_exact",
        "phase_residual_forward_exact",
        "flip_delta_forward_exact",
        "flipped_occupancy_affine_forward_exact",
        "actual_common_forward_exact",
        "actual_specific_forward_exact",
        "flipped_common_forward_exact",
        "flipped_specific_forward_exact",
        "actual_hidden_forward_bounded",
        "flipped_hidden_forward_bounded",
        "actual_energy_forward_exact",
        "flipped_energy_forward_exact",
        "native_interaction_forward_exact",
        "native_field_forward_exact",
        "output_field_forward_exact",
    )
    by_name = {
        check.name: check for check in report.exact_replay.checks
    }
    assert all(
        check.maximum_bound == 0.0
        for name, check in by_name.items()
        if not name.endswith("_hidden_forward_bounded")
    )
    assert (
        by_name["actual_hidden_forward_bounded"].maximum_bound > 0.0
    )
    assert (
        by_name["flipped_hidden_forward_bounded"].maximum_bound > 0.0
    )
    payload = report.canonical_payload()
    assert payload["passed"] is True
    assert payload["exact_replay"]["passed"] is True
    assert payload["exact_replay"]["algebraic_replay_exact"] is True
    assert (
        payload["exact_replay"]["transcendental_replay_bounded"]
        is True
    )
    assert payload["exact_replay"]["silu_replay_factor_hex"] == (
        PACRE_VC_SILU_REPLAY_FACTOR.hex()
    )


def test_flip_delta_tamper_fails_closed_with_exact_coordinate() -> None:
    model, fields = _model_and_fields()
    changed = fields.flip_delta.clone()
    coordinate = (0, 1, 2, 1, 2)
    changed[coordinate] += 0.125
    tampered = replace(fields, flip_delta=changed)

    report = verify_exact_forward_replay(model, tampered)
    by_name = {check.name: check for check in report.checks}

    assert not report.passed
    assert not by_name["flip_delta_forward_exact"].passed
    assert (
        by_name["flip_delta_forward_exact"].argmax_coordinate
        == coordinate
    )
    assert by_name["flip_delta_forward_exact"].failed_element_count == 1
    assert not by_name["flipped_occupancy_affine_forward_exact"].passed


def test_flipped_occupancy_tamper_cannot_self_consistently_pass() -> None:
    model, fields = _model_and_fields()
    changed = fields.flipped_occupancy_affine.clone()
    coordinate = (0, 2, 1, 0, 3)
    changed[coordinate] -= 0.25
    tampered = replace(fields, flipped_occupancy_affine=changed)

    report = verify_exact_forward_replay(model, tampered)
    by_name = {check.name: check for check in report.checks}

    assert by_name["flip_delta_forward_exact"].passed
    assert not by_name["flipped_occupancy_affine_forward_exact"].passed
    assert (
        by_name[
            "flipped_occupancy_affine_forward_exact"
        ].argmax_coordinate
        == coordinate
    )
    assert not by_name["flipped_common_forward_exact"].passed
    assert not by_name["flipped_specific_forward_exact"].passed


def test_material_hidden_tamper_exceeds_frozen_silu_replay_bound() -> None:
    model, fields = _model_and_fields()
    changed = fields.flipped_compatibility_hidden.clone()
    coordinate = (0, 2, 1, 0, 3)
    changed[coordinate] += 1.0e-3
    tampered = replace(
        fields,
        flipped_compatibility_hidden=changed,
    )

    report = verify_exact_forward_replay(model, tampered)
    by_name = {check.name: check for check in report.checks}
    hidden = by_name["flipped_hidden_forward_bounded"]

    assert not report.passed
    assert not hidden.passed
    assert hidden.argmax_coordinate == coordinate
    assert hidden.maximum_error > hidden.bound_at_maximum_error
    assert hidden.maximum_error > 9.0e-4


def test_replay_is_no_grad_and_does_not_mutate_model_or_fields() -> None:
    model, fields = _model_and_fields()
    parameters_before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    residual_before = fields.phase_feature_residual.detach().clone()
    flip_before = fields.flip_delta.detach().clone()
    gradients_before = tuple(
        parameter.grad for parameter in model.parameters()
    )

    report = verify_exact_forward_replay(model, fields)

    assert report.passed
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            model.parameters(),
            strict=True,
        )
    )
    assert torch.equal(residual_before, fields.phase_feature_residual)
    assert torch.equal(flip_before, fields.flip_delta)
    assert gradients_before == tuple(
        parameter.grad for parameter in model.parameters()
    )


def test_contract_rejects_noncontiguous_dtype_and_nonfinite_fields() -> None:
    model, fields = _model_and_fields()
    residual = fields.phase_feature_residual
    noncontiguous = (
        residual.transpose(-1, -2)
        .contiguous()
        .transpose(-1, -2)
    )
    assert noncontiguous.shape == residual.shape
    assert not noncontiguous.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        validate_pacre_fields_contract(
            model,
            replace(fields, phase_feature_residual=noncontiguous),
        )
    with pytest.raises(TypeError, match="dtype"):
        validate_pacre_fields_contract(
            model,
            replace(
                fields,
                phase_feature_residual=residual.to(torch.float64),
            ),
        )
    nonfinite = residual.clone()
    nonfinite.reshape(-1)[0] = float("nan")
    with pytest.raises(FloatingPointError, match="finite"):
        validate_pacre_fields_contract(
            model,
            replace(fields, phase_feature_residual=nonfinite),
        )


def test_contract_rejects_shape_drift_before_replay() -> None:
    model, fields = _model_and_fields()
    changed = fields.phase_feature_residual[..., :-1].contiguous()

    with pytest.raises(ValueError, match="shape"):
        validate_pacre_fields_contract(
            model,
            replace(fields, phase_feature_residual=changed),
        )
