"""Strict construction boundary for a trainable CURE-Lite v22 PACRE.

The legacy coverage-state registry dispatches subclasses with ``isinstance``.
Because :class:`CoverageStatePACREConfig` derives from the sealed v21 PAET
configuration, passing it to that registry constructs a v21 PAET model.  This
module deliberately bypasses that registry and exposes one exact-type callable
for code that accepts a model factory, or for callers that pass the returned
model to ``train_coverage_state_objective``.

The factory also rejects stride one.  PACRE centers over the phase axis, so a
single phase has an identically zero residual and cannot provide a trainable
completion field.
"""

from __future__ import annotations

from typing import Callable, Final, TypeAlias

import torch

from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
)
from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
)


PACRE_MINIMUM_TRAINING_FEATURE_STRIDE: Final = 2
PACRE_PARAMETER_NAMES: Final = (
    "joint_state_weight",
    "joint_hidden_bias",
    "scalar_energy_weight",
)

PACRETrainingModel: TypeAlias = (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
)
PACRETrainingModelFactory: TypeAlias = Callable[
    [object],
    PACRETrainingModel,
]


def build_pacre_training_model(config: object) -> PACRETrainingModel:
    """Construct the exact PACRE model accepted by existing training code.

    ``type`` equality is intentional.  In particular, the v21 PAET config and
    subclasses of the PACRE config are rejected rather than being routed by
    inheritance.  The returned object is a
    :class:`CURELiteCoverageStateLevelSet`, so it can be passed directly to the
    existing public ``train_coverage_state_objective`` entry.
    """

    if type(config) is not CoverageStatePACREConfig:
        raise TypeError(
            "config must have exact type CoverageStatePACREConfig"
        )
    if config.feature_stride < PACRE_MINIMUM_TRAINING_FEATURE_STRIDE:
        raise ValueError(
            "PACRE training requires feature_stride >= 2 because stride 1 "
            "has no phase-specific residual"
        )

    model = (
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
            config
        )
    )
    if (
        type(model)
        is not
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
        or not isinstance(model, CURELiteCoverageStateLevelSet)
        or model.config is not config
    ):
        raise AssertionError("PACRE factory constructed the wrong model")

    named_parameters = tuple(model.named_parameters())
    if tuple(name for name, _ in named_parameters) != PACRE_PARAMETER_NAMES:
        raise AssertionError("PACRE parameter names differ from the contract")
    if (
        sum(parameter.numel() for _, parameter in named_parameters)
        != config.expected_parameter_count
    ):
        raise AssertionError("PACRE parameter count differs from the contract")
    if any(
        parameter.dtype != torch.float32 or not parameter.requires_grad
        for _, parameter in named_parameters
    ):
        raise AssertionError(
            "PACRE training parameters must be trainable float32 tensors"
        )
    return model


# Stable callable for dependency injection without consulting the legacy
# inheritance-based registry.
PACRE_TRAINING_MODEL_FACTORY: Final[PACRETrainingModelFactory] = (
    build_pacre_training_model
)


__all__ = [
    "PACRE_MINIMUM_TRAINING_FEATURE_STRIDE",
    "PACRE_PARAMETER_NAMES",
    "PACRE_TRAINING_MODEL_FACTORY",
    "PACRETrainingModel",
    "PACRETrainingModelFactory",
    "build_pacre_training_model",
]
