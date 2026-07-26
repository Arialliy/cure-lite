"""Frozen configuration for the additive CURE-Lite PR-SVEF v6 decoder."""

from __future__ import annotations

from dataclasses import dataclass

from .factorized_config import (
    SVEF_RESIZE_POLICY,
    FactorizedDecoderConfig,
)


RECOVERABLE_FORWARD_EVIDENCE_TRANSFORM = (
    "one_sided_zero_anchored_squared_softplus_v2"
)
RECOVERABLE_BACKWARD_SURROGATE_POLICY = (
    "negative_half_axis_softplus_recovery_v1"
)
RECOVERABLE_ZERO_BOUNDARY_POLICY = (
    "recovery_branch_gradient_half_v1"
)


@dataclass(frozen=True)
class RecoverableFactorizedDecoderConfig:
    """The v4 topology with one frozen forward/backward evidence operator."""

    feature_channels: int
    feature_stride: int
    width: int = 32
    groups: int = 8
    trunk_residual_scale: float = 0.5
    baseline_probability: float = 0.1
    vacancy_kernel_size: int = 3
    forward_evidence_transform: str = (
        RECOVERABLE_FORWARD_EVIDENCE_TRANSFORM
    )
    backward_surrogate_policy: str = (
        RECOVERABLE_BACKWARD_SURROGATE_POLICY
    )
    zero_boundary_policy: str = RECOVERABLE_ZERO_BOUNDARY_POLICY
    resize_policy: str = SVEF_RESIZE_POLICY

    def __post_init__(self) -> None:
        if (
            self.forward_evidence_transform
            != RECOVERABLE_FORWARD_EVIDENCE_TRANSFORM
        ):
            raise ValueError(
                "PR-SVEF v6 fixes the forward evidence transform"
            )
        if (
            self.backward_surrogate_policy
            != RECOVERABLE_BACKWARD_SURROGATE_POLICY
        ):
            raise ValueError(
                "PR-SVEF v6 fixes the backward surrogate policy"
            )
        if self.zero_boundary_policy != RECOVERABLE_ZERO_BOUNDARY_POLICY:
            raise ValueError(
                "PR-SVEF v6 fixes the zero-boundary policy"
            )
        topology = FactorizedDecoderConfig(
            feature_channels=self.feature_channels,
            feature_stride=self.feature_stride,
            width=self.width,
            groups=self.groups,
            trunk_residual_scale=self.trunk_residual_scale,
            baseline_probability=self.baseline_probability,
            vacancy_kernel_size=self.vacancy_kernel_size,
            resize_policy=self.resize_policy,
        )
        for name in (
            "feature_channels",
            "feature_stride",
            "width",
            "groups",
            "trunk_residual_scale",
            "baseline_probability",
            "vacancy_kernel_size",
            "resize_policy",
        ):
            object.__setattr__(self, name, getattr(topology, name))

    @property
    def phase_channels(self) -> int:
        return self.feature_stride**2

    @property
    def expected_parameter_count(self) -> int:
        return self.to_v4_topology_config().expected_parameter_count

    def to_v4_topology_config(self) -> FactorizedDecoderConfig:
        """Return the unchanged v4 module topology and initialization."""

        return FactorizedDecoderConfig(
            feature_channels=self.feature_channels,
            feature_stride=self.feature_stride,
            width=self.width,
            groups=self.groups,
            trunk_residual_scale=self.trunk_residual_scale,
            baseline_probability=self.baseline_probability,
            vacancy_kernel_size=self.vacancy_kernel_size,
            resize_policy=self.resize_policy,
        )


__all__ = [
    "RECOVERABLE_BACKWARD_SURROGATE_POLICY",
    "RECOVERABLE_FORWARD_EVIDENCE_TRANSFORM",
    "RECOVERABLE_ZERO_BOUNDARY_POLICY",
    "RecoverableFactorizedDecoderConfig",
]
