"""Frozen configuration for CURE-Lite PB-NAES v9."""

from __future__ import annotations

from dataclasses import dataclass

from .factorized_config import (
    SVEF_RESIZE_POLICY,
    FactorizedDecoderConfig,
)


PB_NAES_METHOD_ID = "pb_naes_v9"
PB_NAES_PHASE_INTENSITY_POLICY = "elementwise_exponential_intensity_v1"
PB_NAES_NULL_MASS_POLICY = "one_unit_per_phase_v1"
PB_NAES_BALANCE_POLICY = (
    "phase_intensity_and_equal_total_null_mass_midpoint_v1"
)
PB_NAES_OCCUPANCY_POLICY = (
    "native_project_max_then_3x3_count_divisor_v1"
)
PB_NAES_FORWARD_POLICY = "positive_signed_surplus_v1"
PB_NAES_BACKWARD_POLICY = "full_signed_surplus_straight_through_v1"
PB_NAES_LOGIT_COMPOSITION_POLICY = (
    "baseline_plus_phase_balanced_null_surplus_v1"
)
PB_NAES_RESIZE_POLICY = (
    "native_surplus_then_pixelshuffle_then_bilinear_v1"
)


@dataclass(frozen=True)
class PhaseBalancedNullSurplusFactorizedDecoderConfig:
    """The v4 topology with the parameter-free PB-NAES v9 operator.

    ``feature_channels`` and ``feature_stride`` remain adapter-bound.  Every
    other field is frozen, so PB-NAES introduces neither a new trainable
    component nor a new tuning surface.
    """

    feature_channels: int
    feature_stride: int
    width: int = 32
    groups: int = 8
    trunk_residual_scale: float = 0.5
    baseline_probability: float = 0.1
    vacancy_kernel_size: int = 3
    method_id: str = PB_NAES_METHOD_ID
    phase_intensity_policy: str = PB_NAES_PHASE_INTENSITY_POLICY
    null_mass_policy: str = PB_NAES_NULL_MASS_POLICY
    balance_policy: str = PB_NAES_BALANCE_POLICY
    occupancy_policy: str = PB_NAES_OCCUPANCY_POLICY
    forward_policy: str = PB_NAES_FORWARD_POLICY
    backward_policy: str = PB_NAES_BACKWARD_POLICY
    logit_composition_policy: str = (
        PB_NAES_LOGIT_COMPOSITION_POLICY
    )
    resize_policy: str = PB_NAES_RESIZE_POLICY

    def __post_init__(self) -> None:
        frozen = {
            "method_id": PB_NAES_METHOD_ID,
            "phase_intensity_policy": PB_NAES_PHASE_INTENSITY_POLICY,
            "null_mass_policy": PB_NAES_NULL_MASS_POLICY,
            "balance_policy": PB_NAES_BALANCE_POLICY,
            "occupancy_policy": PB_NAES_OCCUPANCY_POLICY,
            "forward_policy": PB_NAES_FORWARD_POLICY,
            "backward_policy": PB_NAES_BACKWARD_POLICY,
            "logit_composition_policy": (
                PB_NAES_LOGIT_COMPOSITION_POLICY
            ),
            "resize_policy": PB_NAES_RESIZE_POLICY,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"PB-NAES v9 fixes {name}")

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
        """Number of subpixel phases represented by one feature cell."""

        return self.feature_stride**2

    @property
    def expected_parameter_count(self) -> int:
        """Exact trainable parameter count of the unchanged v4 topology."""

        return self.to_v4_topology_config().expected_parameter_count

    def to_v4_topology_config(self) -> FactorizedDecoderConfig:
        """Return the unchanged v4 topology and initialization contract."""

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
    "PB_NAES_BACKWARD_POLICY",
    "PB_NAES_BALANCE_POLICY",
    "PB_NAES_FORWARD_POLICY",
    "PB_NAES_LOGIT_COMPOSITION_POLICY",
    "PB_NAES_METHOD_ID",
    "PB_NAES_NULL_MASS_POLICY",
    "PB_NAES_OCCUPANCY_POLICY",
    "PB_NAES_PHASE_INTENSITY_POLICY",
    "PB_NAES_RESIZE_POLICY",
    "PhaseBalancedNullSurplusFactorizedDecoderConfig",
]
