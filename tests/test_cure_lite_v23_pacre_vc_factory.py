from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
)
from cure_lite_v22.factory import build_pacre_training_model
from cure_lite_v22.pacre import CoverageStatePACREConfig
from cure_lite_v23.factory import (
    PACRE_VC_PARAMETER_NAMES,
    PACRE_VC_TRAINING_MODEL_FACTORY,
    build_pacre_vc_training_model,
)
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)


def _config() -> CoverageStatePACREVerifierCorrectedConfig:
    return CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )


def test_factory_constructs_only_the_exact_v23_model() -> None:
    config = _config()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(230010)
        model = build_pacre_vc_training_model(config)

    assert type(model) is CURELitePACREVerifierCorrectedLevelSet
    assert isinstance(model, CURELiteCoverageStateLevelSet)
    assert model.config is config
    assert PACRE_VC_TRAINING_MODEL_FACTORY is (
        build_pacre_vc_training_model
    )
    assert tuple(name for name, _ in model.named_parameters()) == (
        PACRE_VC_PARAMETER_NAMES
    )
    assert sum(value.numel() for value in model.parameters()) == 64064
    assert all(
        value.dtype == torch.float32 and value.requires_grad
        for value in model.parameters()
    )


def test_factory_rejects_v22_config_and_arbitrary_objects() -> None:
    with pytest.raises(TypeError, match="exact type"):
        build_pacre_vc_training_model(
            CoverageStatePACREConfig(
                feature_channels=64,
                feature_stride=4,
                width=32,
            )
        )
    with pytest.raises(TypeError, match="exact type"):
        build_pacre_vc_training_model(object())


def test_factory_rejects_a_v23_config_subclass() -> None:
    @dataclass(frozen=True)
    class DerivedConfig(CoverageStatePACREVerifierCorrectedConfig):
        pass

    with pytest.raises(TypeError, match="exact type"):
        build_pacre_vc_training_model(
            DerivedConfig(
                feature_channels=64,
                feature_stride=4,
                width=32,
            )
        )


def test_legacy_v22_factory_does_not_silently_route_v23_config() -> None:
    with pytest.raises(TypeError, match="CoverageStatePACREConfig"):
        build_pacre_training_model(_config())
