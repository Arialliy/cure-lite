from __future__ import annotations

from dataclasses import fields as dataclass_fields

import pytest
import torch

from cure_lite.coverage_state_level_set import CSLF_FIELD_AMPLITUDE
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
)
from cure_lite.coverage_state_phase_preserving import (
    pixel_unshuffle_bool_occupancy,
)
from cure_lite_v22.pacre import (
    CSLF_PACRE_CENTERING_POLICY,
    CSLF_PACRE_EQUATION_POLICY,
    CSLF_PACRE_FIELD_POLICY,
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
    phase_centered_feature_affine,
)


def _model() -> (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
):
    model = (
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
            CoverageStatePACREConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(220022)
    with torch.no_grad():
        model.joint_state_weight.copy_(
            0.12
            * torch.randn(
                model.joint_state_weight.shape,
                generator=generator,
            )
        )
        model.joint_hidden_bias.copy_(
            0.08
            * torch.randn(
                model.joint_hidden_bias.shape,
                generator=generator,
            )
        )
        model.scalar_energy_weight.copy_(
            0.20
            * torch.randn(
                model.scalar_energy_weight.shape,
                generator=generator,
            )
        )
    return model


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(220023)
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
        > 0.72
    )
    return feature, occupancy


def test_config_and_parameter_contract_match_v21_topology() -> None:
    config = CoverageStatePACREConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    model = (
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
            config
        )
    )

    assert config.field_policy == CSLF_PACRE_FIELD_POLICY
    assert config.equation_policy == CSLF_PACRE_EQUATION_POLICY
    assert config.centering_policy == CSLF_PACRE_CENTERING_POLICY
    assert config.expected_parameter_count == 64064
    assert sum(value.numel() for value in model.parameters()) == 64064
    assert tuple(name for name, _ in model.named_parameters()) == (
        "joint_state_weight",
        "joint_hidden_bias",
        "scalar_energy_weight",
    )
    assert isinstance(
        model,
        CURELitePhaseAlignedEvidenceTransportLevelSet,
    )
    assert {
        field.name for field in dataclass_fields(config)
    } >= {"centering_policy"}


def test_config_rejects_policy_drift() -> None:
    with pytest.raises(ValueError, match="PACRE fixes centering_policy"):
        CoverageStatePACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
            centering_policy="changed",
        )


def test_config_rejects_single_phase_degeneracy() -> None:
    with pytest.raises(
        ValueError,
        match="feature_stride >= 2",
    ):
        CoverageStatePACREConfig(
            feature_channels=2,
            feature_stride=1,
            width=4,
        )


def test_phase_centering_is_exact_up_to_fp32_reduction() -> None:
    generator = torch.Generator(device="cpu").manual_seed(220024)
    phase = torch.randn(
        (2, 16, 5, 3, 4),
        generator=generator,
        dtype=torch.float32,
    )
    mean, residual = phase_centered_feature_affine(phase)

    assert mean.shape == (2, 1, 5, 3, 4)
    assert residual.shape == phase.shape
    torch.testing.assert_close(
        mean + residual,
        phase,
        rtol=0.0,
        atol=2.5e-7,
    )
    torch.testing.assert_close(
        residual.mean(dim=1),
        torch.zeros_like(mean[:, 0]),
        rtol=0.0,
        atol=2.0e-7,
    )


def test_fast_field_matches_literal_local_equation() -> None:
    model = _model()
    feature, occupancy = _inputs()

    actual = model(feature, occupancy)
    reference = model.forward_reference(feature, occupancy)

    torch.testing.assert_close(
        actual,
        reference,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_zero_feature_and_phase_common_feature_are_exact_anchors() -> None:
    model = _model()
    _, occupancy = _inputs()
    zero = torch.zeros((1, 2, 3, 4), dtype=torch.float32)

    zero_field = model(zero, occupancy)
    torch.testing.assert_close(
        zero_field,
        torch.full_like(zero_field, CSLF_FIELD_AMPLITUDE),
        rtol=0.0,
        atol=0.0,
    )

    encoded = torch.randn(
        (1, 2, 3, 4),
        generator=torch.Generator().manual_seed(220025),
    )
    phase = pixel_unshuffle_bool_occupancy(
        occupancy,
        stride=model.config.feature_stride,
    )
    occupancy_affine, coarse, upsampled, phase_feature = (
        model._affine_states(encoded, phase)
    )
    del occupancy_affine, coarse, upsampled
    phase_common = phase_feature.mean(dim=1, keepdim=True).expand_as(
        phase_feature
    )
    _, residual = phase_centered_feature_affine(phase_common)
    assert torch.count_nonzero(residual) == 0


def test_single_phase_flip_is_exactly_odd_at_affected_coordinate() -> None:
    model = _model()
    feature, occupancy = _inputs()
    row = 1
    column = 2
    phase_row = 0
    phase_column = 1
    out_row = row * model.config.feature_stride + phase_row
    out_column = (
        column * model.config.feature_stride + phase_column
    )
    flipped = occupancy.clone()
    flipped[:, :, out_row, out_column] = ~flipped[
        :,
        :,
        out_row,
        out_column,
    ]

    first = model.forward_fields(feature, occupancy)
    second = model.forward_fields(feature, flipped)
    first_delta = (
        first.field[:, :, out_row, out_column]
        - model.config.field_amplitude
    )
    second_delta = (
        second.field[:, :, out_row, out_column]
        - model.config.field_amplitude
    )
    torch.testing.assert_close(
        first_delta,
        -second_delta,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_one_field_and_all_parameters_receive_finite_gradients() -> None:
    model = _model()
    feature, occupancy = _inputs()
    field = model(feature, occupancy)
    assert field.shape == occupancy.shape
    assert field.dtype == torch.float32
    assert len(tuple(model.named_children())) == 1

    loss = field.square().mean()
    parameters = tuple(model.parameters())
    first = torch.autograd.grad(
        loss,
        parameters,
        create_graph=True,
        allow_unused=False,
    )
    assert all(
        bool(torch.isfinite(value).all())
        and bool(torch.any(value != 0.0))
        for value in first
    )
    second = torch.autograd.grad(
        torch.stack([value.sum() for value in first]).sum(),
        parameters,
        allow_unused=False,
    )
    assert all(bool(torch.isfinite(value).all()) for value in second)


def test_frozen_zero_readout_initialization_starts_all_gradients() -> None:
    torch.manual_seed(220026)
    model = (
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
            CoverageStatePACREConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )
    )
    feature, occupancy = _inputs()
    parameters = dict(model.named_parameters())

    initial_loss = model(feature, occupancy).square().mean()
    initial_gradients = dict(
        zip(
            parameters,
            torch.autograd.grad(
                initial_loss,
                tuple(parameters.values()),
                allow_unused=False,
            ),
            strict=True,
        )
    )
    assert bool(
        torch.any(initial_gradients["scalar_energy_weight"] != 0.0)
    )
    assert torch.count_nonzero(
        initial_gradients["joint_state_weight"]
    ) == 0
    assert torch.count_nonzero(
        initial_gradients["joint_hidden_bias"]
    ) == 0

    with torch.no_grad():
        parameters["scalar_energy_weight"].sub_(
            1.0e-2
            * initial_gradients["scalar_energy_weight"]
        )
    started_loss = model(feature, occupancy).square().mean()
    started_gradients = torch.autograd.grad(
        started_loss,
        tuple(parameters.values()),
        allow_unused=False,
    )
    assert all(
        bool(torch.isfinite(value).all())
        and bool(torch.any(value != 0.0))
        for value in started_gradients
    )
