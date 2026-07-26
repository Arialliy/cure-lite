"""Frozen configuration for the additive CURE-Lite CR-LVEC v7 decoder."""

from __future__ import annotations

from dataclasses import dataclass

from .factorized_config import (
    SVEF_RESIZE_POLICY,
    FactorizedDecoderConfig,
)


CROSSING_OCCUPANCY_BURDEN_POLICY = (
    "nearest_log1p_local_occupancy_count_v1"
)
CROSSING_FORWARD_EVIDENCE_TRANSFORM = (
    "positive_exponential_ratio_crossing_v1"
)
CROSSING_BACKWARD_SURROGATE_POLICY = (
    "full_axis_exponential_continuation_v1"
)
CROSSING_ZERO_BOUNDARY_POLICY = (
    "full_axis_unit_gradient_v1"
)
CROSSING_LOGIT_COMPOSITION_POLICY = (
    "baseline_plus_crossing_evidence_v1"
)
CROSSING_RESIZE_POLICY = (
    "bilinear_raw_nearest_burden_then_crossing_v1"
)


@dataclass(frozen=True)
class CrossingFactorizedDecoderConfig:
    """The v4 topology with one frozen local-burden crossing equation."""

    feature_channels: int
    feature_stride: int
    width: int = 32
    groups: int = 8
    trunk_residual_scale: float = 0.5
    baseline_probability: float = 0.1
    vacancy_kernel_size: int = 3
    occupancy_burden_policy: str = CROSSING_OCCUPANCY_BURDEN_POLICY
    forward_evidence_transform: str = (
        CROSSING_FORWARD_EVIDENCE_TRANSFORM
    )
    backward_surrogate_policy: str = (
        CROSSING_BACKWARD_SURROGATE_POLICY
    )
    zero_boundary_policy: str = CROSSING_ZERO_BOUNDARY_POLICY
    logit_composition_policy: str = CROSSING_LOGIT_COMPOSITION_POLICY
    resize_policy: str = CROSSING_RESIZE_POLICY

    def __post_init__(self) -> None:
        if (
            self.occupancy_burden_policy
            != CROSSING_OCCUPANCY_BURDEN_POLICY
        ):
            raise ValueError(
                "CR-LVEC v7 fixes the occupancy-burden policy"
            )
        if (
            self.forward_evidence_transform
            != CROSSING_FORWARD_EVIDENCE_TRANSFORM
        ):
            raise ValueError(
                "CR-LVEC v7 fixes the forward evidence transform"
            )
        if (
            self.backward_surrogate_policy
            != CROSSING_BACKWARD_SURROGATE_POLICY
        ):
            raise ValueError(
                "CR-LVEC v7 fixes the backward surrogate policy"
            )
        if self.zero_boundary_policy != CROSSING_ZERO_BOUNDARY_POLICY:
            raise ValueError(
                "CR-LVEC v7 fixes the zero-boundary policy"
            )
        if (
            self.logit_composition_policy
            != CROSSING_LOGIT_COMPOSITION_POLICY
        ):
            raise ValueError(
                "CR-LVEC v7 fixes the logit-composition policy"
            )
        if self.resize_policy != CROSSING_RESIZE_POLICY:
            raise ValueError("CR-LVEC v7 fixes the field resize policy")

        topology = FactorizedDecoderConfig(
            feature_channels=self.feature_channels,
            feature_stride=self.feature_stride,
            width=self.width,
            groups=self.groups,
            trunk_residual_scale=self.trunk_residual_scale,
            baseline_probability=self.baseline_probability,
            vacancy_kernel_size=self.vacancy_kernel_size,
            resize_policy=SVEF_RESIZE_POLICY,
        )
        for name in (
            "feature_channels",
            "feature_stride",
            "width",
            "groups",
            "trunk_residual_scale",
            "baseline_probability",
            "vacancy_kernel_size",
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
            resize_policy=SVEF_RESIZE_POLICY,
        )


__all__ = [
    "CROSSING_BACKWARD_SURROGATE_POLICY",
    "CROSSING_FORWARD_EVIDENCE_TRANSFORM",
    "CROSSING_LOGIT_COMPOSITION_POLICY",
    "CROSSING_OCCUPANCY_BURDEN_POLICY",
    "CROSSING_RESIZE_POLICY",
    "CROSSING_ZERO_BOUNDARY_POLICY",
    "CrossingFactorizedDecoderConfig",
]
