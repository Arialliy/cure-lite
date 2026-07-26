"""Frozen configuration for CURE-Lite NLCC v12."""

from __future__ import annotations

from dataclasses import dataclass

from .factorized_config import (
    SVEF_RESIZE_POLICY,
    FactorizedDecoderConfig,
)


NLCC_OCCUPANCY_PROJECTION_POLICY = (
    "adaptive_max_then_native_3x3_count_v1"
)
NLCC_PHASE_REFERENCE_POLICY = (
    "single_fixed_zero_null_arithmetic_reference_v1"
)
NLCC_COUNT_BOUNDARY_POLICY = "unit_local_occupancy_count_boundary_v1"
NLCC_CROSSING_POLICY = (
    "positive_expm1_forward_full_axis_exponential_recovery_v1"
)
NLCC_BASELINE_POLICY = "occupancy_invariant_v4_phase_baseline_v1"
NLCC_LOGIT_COMPOSITION_POLICY = (
    "negative_baseline_plus_local_phase_crossing_evidence_v1"
)


@dataclass(frozen=True)
class NullAnchoredLocalCountCrossingDecoderConfig:
    """The v4 topology with one null-anchored local-count equation.

    Only ``feature_channels`` and ``feature_stride`` are adapter-bound.
    Every other field is a frozen method constant, not a search surface.
    """

    feature_channels: int
    feature_stride: int
    width: int = 32
    groups: int = 8
    trunk_residual_scale: float = 0.5
    baseline_probability: float = 0.1
    vacancy_kernel_size: int = 3
    occupancy_projection_policy: str = (
        NLCC_OCCUPANCY_PROJECTION_POLICY
    )
    phase_reference_policy: str = NLCC_PHASE_REFERENCE_POLICY
    count_boundary_policy: str = NLCC_COUNT_BOUNDARY_POLICY
    crossing_policy: str = NLCC_CROSSING_POLICY
    baseline_policy: str = NLCC_BASELINE_POLICY
    logit_composition_policy: str = (
        NLCC_LOGIT_COMPOSITION_POLICY
    )
    resize_policy: str = SVEF_RESIZE_POLICY

    def __post_init__(self) -> None:
        frozen = {
            "occupancy_projection_policy": (
                NLCC_OCCUPANCY_PROJECTION_POLICY
            ),
            "phase_reference_policy": NLCC_PHASE_REFERENCE_POLICY,
            "count_boundary_policy": NLCC_COUNT_BOUNDARY_POLICY,
            "crossing_policy": NLCC_CROSSING_POLICY,
            "baseline_policy": NLCC_BASELINE_POLICY,
            "logit_composition_policy": (
                NLCC_LOGIT_COMPOSITION_POLICY
            ),
            "resize_policy": SVEF_RESIZE_POLICY,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"NLCC v12 fixes {name}")

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
        """Number of output subpixels represented by one feature cell."""

        return self.feature_stride**2

    @property
    def expected_parameter_count(self) -> int:
        """Exact parameter count of the unchanged active-v4 topology."""

        return self.to_v4_topology_config().expected_parameter_count

    def to_v4_topology_config(self) -> FactorizedDecoderConfig:
        """Return the unchanged module topology and initialization."""

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
    "NLCC_BASELINE_POLICY",
    "NLCC_COUNT_BOUNDARY_POLICY",
    "NLCC_CROSSING_POLICY",
    "NLCC_LOGIT_COMPOSITION_POLICY",
    "NLCC_OCCUPANCY_PROJECTION_POLICY",
    "NLCC_PHASE_REFERENCE_POLICY",
    "NullAnchoredLocalCountCrossingDecoderConfig",
]
