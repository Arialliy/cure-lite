"""Strict construction boundary for the PACRE-VC v23 training model."""

from __future__ import annotations

from typing import Callable, Final, TypeAlias

import torch

from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
)

from .pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)


PACRE_VC_MINIMUM_TRAINING_FEATURE_STRIDE: Final = 2
PACRE_VC_PARAMETER_NAMES: Final = (
    "joint_state_weight",
    "joint_hidden_bias",
    "scalar_energy_weight",
)

PACREVCTrainingModel: TypeAlias = (
    CURELitePACREVerifierCorrectedLevelSet
)
PACREVCTrainingModelFactory: TypeAlias = Callable[
    [object],
    PACREVCTrainingModel,
]


def build_pacre_vc_training_model(
    config: object,
) -> PACREVCTrainingModel:
    """Construct exactly the versioned v23 wrapper.

    Exact type equality prevents the inheritance-based legacy registry from
    silently routing the v23 config to a v21/v22 model FQCN.
    """

    if type(config) is not CoverageStatePACREVerifierCorrectedConfig:
        raise TypeError(
            "config must have exact type "
            "CoverageStatePACREVerifierCorrectedConfig"
        )
    if (
        config.feature_stride
        < PACRE_VC_MINIMUM_TRAINING_FEATURE_STRIDE
    ):
        raise ValueError(
            "PACRE-VC training requires feature_stride >= 2 because "
            "stride 1 has no phase-specific residual"
        )

    model = CURELitePACREVerifierCorrectedLevelSet(config)
    if (
        type(model) is not CURELitePACREVerifierCorrectedLevelSet
        or not isinstance(model, CURELiteCoverageStateLevelSet)
        or model.config is not config
    ):
        raise AssertionError(
            "PACRE-VC factory constructed the wrong model"
        )

    named_parameters = tuple(model.named_parameters())
    if (
        tuple(name for name, _ in named_parameters)
        != PACRE_VC_PARAMETER_NAMES
    ):
        raise AssertionError(
            "PACRE-VC parameter names differ from the contract"
        )
    if (
        sum(parameter.numel() for _, parameter in named_parameters)
        != config.expected_parameter_count
    ):
        raise AssertionError(
            "PACRE-VC parameter count differs from the contract"
        )
    if any(
        parameter.dtype != torch.float32
        or not parameter.requires_grad
        for _, parameter in named_parameters
    ):
        raise AssertionError(
            "PACRE-VC parameters must be trainable float32 tensors"
        )
    return model


PACRE_VC_TRAINING_MODEL_FACTORY: Final[
    PACREVCTrainingModelFactory
] = build_pacre_vc_training_model


__all__ = [
    "PACRE_VC_MINIMUM_TRAINING_FEATURE_STRIDE",
    "PACRE_VC_PARAMETER_NAMES",
    "PACRE_VC_TRAINING_MODEL_FACTORY",
    "PACREVCTrainingModel",
    "PACREVCTrainingModelFactory",
    "build_pacre_vc_training_model",
]
