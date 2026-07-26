"""Frozen configuration for CURE-Lite CCFR v11."""

from __future__ import annotations

from dataclasses import dataclass

from .conservative_factorized_config import (
    CONSERVATIVE_ALLOCATION_POLICY,
    CONSERVATIVE_MASS_POLICY,
    CONSERVATIVE_PHASE_AGGREGATION_POLICY,
    CONSERVATIVE_RESIZE_POLICY,
)
from .factorized_config import (
    SVEF_RESIZE_POLICY,
    FactorizedDecoderConfig,
)


CCFR_OCCUPANCY_PROJECTION_POLICY = (
    "adaptive_max_then_native_3x3_count_v1"
)
CCFR_FEATURE_RELEASE_POLICY = (
    "post_stem_norm_inverse_local_count_release_v1"
)
CCFR_JOINT_TRUNK_POLICY = (
    "release_before_depthwise_residual_transform_v1"
)
CCFR_OUTPUT_BUDGET_POLICY = (
    "feature_conditioned_common_mode_crossing_without_posthead_occupancy_v1"
)
CCFR_LOGIT_COMPOSITION_POLICY = (
    "v8_baseline_plus_conserved_phase_evidence_without_output_gate_v1"
)


@dataclass(frozen=True)
class CoverageFeatureReleaseDecoderConfig:
    """The v4 topology with one in-trunk coverage intervention.

    Only ``feature_channels`` and ``feature_stride`` are adapter-bound.
    Every other value is a frozen method constant rather than a search
    surface.
    """

    feature_channels: int
    feature_stride: int
    width: int = 32
    groups: int = 8
    trunk_residual_scale: float = 0.5
    baseline_probability: float = 0.1
    vacancy_kernel_size: int = 3
    occupancy_projection_policy: str = (
        CCFR_OCCUPANCY_PROJECTION_POLICY
    )
    feature_release_policy: str = CCFR_FEATURE_RELEASE_POLICY
    joint_trunk_policy: str = CCFR_JOINT_TRUNK_POLICY
    phase_aggregation_policy: str = (
        CONSERVATIVE_PHASE_AGGREGATION_POLICY
    )
    output_budget_policy: str = CCFR_OUTPUT_BUDGET_POLICY
    allocation_policy: str = CONSERVATIVE_ALLOCATION_POLICY
    mass_policy: str = CONSERVATIVE_MASS_POLICY
    logit_composition_policy: str = (
        CCFR_LOGIT_COMPOSITION_POLICY
    )
    resize_policy: str = CONSERVATIVE_RESIZE_POLICY

    def __post_init__(self) -> None:
        frozen = {
            "occupancy_projection_policy": (
                CCFR_OCCUPANCY_PROJECTION_POLICY
            ),
            "feature_release_policy": CCFR_FEATURE_RELEASE_POLICY,
            "joint_trunk_policy": CCFR_JOINT_TRUNK_POLICY,
            "phase_aggregation_policy": (
                CONSERVATIVE_PHASE_AGGREGATION_POLICY
            ),
            "output_budget_policy": CCFR_OUTPUT_BUDGET_POLICY,
            "allocation_policy": CONSERVATIVE_ALLOCATION_POLICY,
            "mass_policy": CONSERVATIVE_MASS_POLICY,
            "logit_composition_policy": (
                CCFR_LOGIT_COMPOSITION_POLICY
            ),
            "resize_policy": CONSERVATIVE_RESIZE_POLICY,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"CCFR v11 fixes {name}")

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
        """Exact parameter count of the unchanged v4 topology."""

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
    "CCFR_FEATURE_RELEASE_POLICY",
    "CCFR_JOINT_TRUNK_POLICY",
    "CCFR_LOGIT_COMPOSITION_POLICY",
    "CCFR_OCCUPANCY_PROJECTION_POLICY",
    "CCFR_OUTPUT_BUDGET_POLICY",
    "CoverageFeatureReleaseDecoderConfig",
]
