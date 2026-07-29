"""Strict construction boundary for the GCR-PACRE v24 model."""

from __future__ import annotations

from typing import Callable, Final, TypeAlias

import torch

from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
)

from .gcr_pacre import (
    CURELiteGatedCommonResidualPACRELevelSet,
    CoverageStateGCRPACREConfig,
)


GCR_PACRE_FORMAL_FEATURE_CHANNELS: Final = 64
GCR_PACRE_FORMAL_FEATURE_STRIDE: Final = 4
GCR_PACRE_FORMAL_WIDTH: Final = 32
GCR_PACRE_FORMAL_PARAMETER_COUNT: Final = 64064
GCR_PACRE_MINIMUM_FEATURE_STRIDE: Final = 2
GCR_PACRE_PARAMETER_NAMES: Final = (
    "joint_state_weight",
    "joint_hidden_bias",
    "scalar_energy_weight",
)

GCRPACRETrainingModel: TypeAlias = (
    CURELiteGatedCommonResidualPACRELevelSet
)
GCRPACRETrainingModelFactory: TypeAlias = Callable[
    [object],
    GCRPACRETrainingModel,
]


def build_gcr_pacre_training_model(
    config: object,
) -> GCRPACRETrainingModel:
    """Construct exactly the v24 model and verify its parameter contract."""

    if type(config) is not CoverageStateGCRPACREConfig:
        raise TypeError(
            "config must have exact type CoverageStateGCRPACREConfig"
        )
    if config.feature_stride < GCR_PACRE_MINIMUM_FEATURE_STRIDE:
        raise ValueError(
            "GCR-PACRE requires feature_stride >= 2"
        )
    model = CURELiteGatedCommonResidualPACRELevelSet(config)
    if (
        type(model) is not CURELiteGatedCommonResidualPACRELevelSet
        or not isinstance(model, CURELiteCoverageStateLevelSet)
        or model.config is not config
    ):
        raise AssertionError("factory constructed the wrong v24 model")
    named_parameters = tuple(model.named_parameters())
    if tuple(name for name, _ in named_parameters) != (
        GCR_PACRE_PARAMETER_NAMES
    ):
        raise AssertionError("GCR-PACRE parameter names changed")
    if (
        sum(parameter.numel() for _, parameter in named_parameters)
        != config.expected_parameter_count
    ):
        raise AssertionError("GCR-PACRE parameter count changed")
    if any(
        parameter.dtype != torch.float32
        or not parameter.requires_grad
        for _, parameter in named_parameters
    ):
        raise AssertionError(
            "GCR-PACRE parameters must be trainable float32 tensors"
        )
    return model


def build_formal_gcr_pacre_training_model() -> GCRPACRETrainingModel:
    """Build the frozen formal ``64/4/32`` v24 configuration."""

    config = CoverageStateGCRPACREConfig(
        feature_channels=GCR_PACRE_FORMAL_FEATURE_CHANNELS,
        feature_stride=GCR_PACRE_FORMAL_FEATURE_STRIDE,
        width=GCR_PACRE_FORMAL_WIDTH,
    )
    model = build_gcr_pacre_training_model(config)
    if (
        config.expected_parameter_count
        != GCR_PACRE_FORMAL_PARAMETER_COUNT
    ):
        raise AssertionError("formal GCR-PACRE parameter count changed")
    return model


GCR_PACRE_TRAINING_MODEL_FACTORY: Final[
    GCRPACRETrainingModelFactory
] = build_gcr_pacre_training_model


__all__ = [
    "GCR_PACRE_FORMAL_FEATURE_CHANNELS",
    "GCR_PACRE_FORMAL_FEATURE_STRIDE",
    "GCR_PACRE_FORMAL_PARAMETER_COUNT",
    "GCR_PACRE_FORMAL_WIDTH",
    "GCR_PACRE_MINIMUM_FEATURE_STRIDE",
    "GCR_PACRE_PARAMETER_NAMES",
    "GCR_PACRE_TRAINING_MODEL_FACTORY",
    "GCRPACRETrainingModel",
    "GCRPACRETrainingModelFactory",
    "build_formal_gcr_pacre_training_model",
    "build_gcr_pacre_training_model",
]
