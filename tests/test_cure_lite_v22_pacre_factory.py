from __future__ import annotations

import pytest
import torch

from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
)
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.experiment.coverage_state_training import (
    audit_coverage_state_training_state,
)
from cure_lite_v22.factory import (
    PACRE_PARAMETER_NAMES,
    PACRE_TRAINING_MODEL_FACTORY,
    PACRETrainingModelFactory,
    build_pacre_training_model,
)
from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
)


def _config(*, stride: int = 2) -> CoverageStatePACREConfig:
    return CoverageStatePACREConfig(
        feature_channels=2,
        feature_stride=stride,
        width=4,
    )


def test_factory_constructs_exact_pacre_with_frozen_topology() -> None:
    config = _config()
    injectable_factory: PACRETrainingModelFactory = (
        PACRE_TRAINING_MODEL_FACTORY
    )
    model = injectable_factory(config)

    assert (
        type(model)
        is
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
    )
    assert isinstance(model, CURELiteCoverageStateLevelSet)
    assert model.config is config
    assert PACRE_TRAINING_MODEL_FACTORY is build_pacre_training_model
    named_parameters = tuple(model.named_parameters())
    assert tuple(name for name, _ in named_parameters) == (
        PACRE_PARAMETER_NAMES
    )
    assert tuple(parameter.shape for _, parameter in named_parameters) == (
        (4, 6, 5, 5),
        (4,),
        (4,),
    )
    assert sum(
        parameter.numel() for _, parameter in named_parameters
    ) == config.expected_parameter_count
    assert all(
        parameter.dtype == torch.float32 and parameter.requires_grad
        for _, parameter in named_parameters
    )


def test_factory_rejects_v21_and_nonexact_pacre_configs() -> None:
    v21 = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )

    class DerivedPACREConfig(CoverageStatePACREConfig):
        pass

    derived = DerivedPACREConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    for invalid in (v21, derived, None):
        with pytest.raises(
            TypeError,
            match="exact type CoverageStatePACREConfig",
        ):
            build_pacre_training_model(invalid)


def test_factory_rejects_stride_one_degenerate_training_config() -> None:
    with pytest.raises(
        ValueError,
        match="feature_stride >= 2",
    ):
        build_pacre_training_model(_config(stride=1))


def test_factory_model_is_forward_backward_and_optimizer_compatible() -> None:
    torch.manual_seed(220026)
    model = build_pacre_training_model(_config())
    with torch.no_grad():
        model.scalar_energy_weight.copy_(
            torch.tensor([0.15, -0.20, 0.25, -0.30])
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    audit_coverage_state_training_state(model, optimizer)

    generator = torch.Generator(device="cpu").manual_seed(220027)
    feature = torch.randn(
        (2, 2, 3, 4),
        generator=generator,
        dtype=torch.float32,
    )
    occupancy = (
        torch.rand(
            (2, 1, 6, 8),
            generator=generator,
            dtype=torch.float32,
        )
        > 0.7
    )
    field = model(feature, occupancy)
    assert field.shape == occupancy.shape
    assert field.dtype == torch.float32
    assert bool(torch.isfinite(field).all())

    loss = (field - 0.25).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = tuple(
        parameter.grad for parameter in model.parameters()
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in gradients
        if gradient is not None
    )
    assert all(
        bool(torch.count_nonzero(gradient))
        for gradient in gradients
        if gradient is not None
    )

    optimizer.step()
    audit_coverage_state_training_state(model, optimizer)
