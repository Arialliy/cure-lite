from __future__ import annotations

from dataclasses import fields as dataclass_fields

import pytest
import torch

from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
    CoverageStatePACREFields,
)
from cure_lite_v23.pacre_vc import (
    PACRE_VC_CANDIDATE,
    PACRE_VC_FIELDS_FQCN,
    PACRE_VC_VERIFIER_POLICY,
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)


def _config() -> CoverageStatePACREVerifierCorrectedConfig:
    return CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(230001)
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
    return feature.contiguous(), occupancy.contiguous()


def test_v23_config_adds_only_the_frozen_verifier_policy() -> None:
    config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )

    assert isinstance(config, CoverageStatePACREConfig)
    assert config.verifier_policy == PACRE_VC_VERIFIER_POLICY
    assert config.expected_parameter_count == 64064
    assert PACRE_VC_CANDIDATE == "PACRE-VC-v23"
    assert {
        field.name for field in dataclass_fields(config)
    } - {
        field.name
        for field in dataclass_fields(CoverageStatePACREConfig)
    } == {"verifier_policy"}


def test_v23_config_rejects_verifier_policy_drift() -> None:
    with pytest.raises(
        ValueError,
        match="PACRE-VC fixes verifier_policy",
    ):
        CoverageStatePACREVerifierCorrectedConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
            verifier_policy="changed",
        )


def test_v23_model_inherits_every_numerical_forward_method() -> None:
    model_class = CURELitePACREVerifierCorrectedLevelSet
    parent = (
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
    )

    assert issubclass(model_class, parent)
    assert "forward" not in model_class.__dict__
    assert "forward_fields" not in model_class.__dict__
    assert "_compatibility_energy" not in model_class.__dict__
    assert model_class.forward is parent.forward
    assert model_class.forward_fields is parent.forward_fields
    assert (
        model_class._compatibility_energy
        is parent._compatibility_energy
    )


def test_v23_forward_keeps_the_exact_v22_fields_fqcn() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(230002)
        model = CURELitePACREVerifierCorrectedLevelSet(_config())
    feature, occupancy = _inputs()

    fields = model.forward_fields(feature, occupancy)

    assert model.config is not None
    assert type(model.config) is (
        CoverageStatePACREVerifierCorrectedConfig
    )
    assert type(fields) is CoverageStatePACREFields
    fields_fqcn = (
        f"{type(fields).__module__}.{type(fields).__qualname__}"
    )
    assert fields_fqcn == PACRE_VC_FIELDS_FQCN
    assert PACRE_VC_FIELDS_FQCN == (
        "cure_lite_v22.pacre.CoverageStatePACREFields"
    )
    assert fields.field.shape == occupancy.shape
    assert model(feature, occupancy).shape == occupancy.shape


def test_v23_model_preserves_parameter_and_module_topology() -> None:
    config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(230003)
        model = CURELitePACREVerifierCorrectedLevelSet(config)

    assert model.config is config
    assert tuple(name for name, _ in model.named_parameters()) == (
        "joint_state_weight",
        "joint_hidden_bias",
        "scalar_energy_weight",
    )
    assert sum(value.numel() for value in model.parameters()) == 64064
    assert tuple(model.named_children())[0][0] == "pixel_shuffle"
    assert len(tuple(model.named_children())) == 1
