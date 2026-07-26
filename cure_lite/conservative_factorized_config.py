"""Frozen configuration for CURE-Lite CC-SEA v8."""

from __future__ import annotations

from dataclasses import dataclass

from .factorized_config import (
    SVEF_RESIZE_POLICY,
    FactorizedDecoderConfig,
)


CONSERVATIVE_OCCUPANCY_BURDEN_POLICY = (
    "native_log1p_local_occupancy_count_v1"
)
CONSERVATIVE_PHASE_AGGREGATION_POLICY = (
    "phase_common_mode_arithmetic_mean_v1"
)
CONSERVATIVE_BUDGET_POLICY = (
    "continuously_recoverable_log_coverage_crossing_v1"
)
CONSERVATIVE_ALLOCATION_POLICY = "phase_softmax_v1"
CONSERVATIVE_MASS_POLICY = "native_phase_sum_equals_budget_v1"
CONSERVATIVE_LOGIT_COMPOSITION_POLICY = (
    "baseline_plus_conserved_subpixel_evidence_v1"
)
CONSERVATIVE_RESIZE_POLICY = (
    "allocate_native_phases_then_pixelshuffle_then_bilinear_v1"
)


@dataclass(frozen=True)
class ConservativeFactorizedDecoderConfig:
    """The v4 topology with one coverage-conserving evidence operator.

    Every field except the adapter-bound feature shape is a method constant.
    In particular, CC-SEA has no temperature, allocation width, learned
    budget head, or tunable conservation coefficient.
    """

    feature_channels: int
    feature_stride: int
    width: int = 32
    groups: int = 8
    trunk_residual_scale: float = 0.5
    baseline_probability: float = 0.1
    vacancy_kernel_size: int = 3
    occupancy_burden_policy: str = (
        CONSERVATIVE_OCCUPANCY_BURDEN_POLICY
    )
    phase_aggregation_policy: str = (
        CONSERVATIVE_PHASE_AGGREGATION_POLICY
    )
    budget_policy: str = CONSERVATIVE_BUDGET_POLICY
    allocation_policy: str = CONSERVATIVE_ALLOCATION_POLICY
    mass_policy: str = CONSERVATIVE_MASS_POLICY
    logit_composition_policy: str = (
        CONSERVATIVE_LOGIT_COMPOSITION_POLICY
    )
    resize_policy: str = CONSERVATIVE_RESIZE_POLICY

    def __post_init__(self) -> None:
        frozen = {
            "occupancy_burden_policy": (
                CONSERVATIVE_OCCUPANCY_BURDEN_POLICY
            ),
            "phase_aggregation_policy": (
                CONSERVATIVE_PHASE_AGGREGATION_POLICY
            ),
            "budget_policy": CONSERVATIVE_BUDGET_POLICY,
            "allocation_policy": CONSERVATIVE_ALLOCATION_POLICY,
            "mass_policy": CONSERVATIVE_MASS_POLICY,
            "logit_composition_policy": (
                CONSERVATIVE_LOGIT_COMPOSITION_POLICY
            ),
            "resize_policy": CONSERVATIVE_RESIZE_POLICY,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"CC-SEA v8 fixes {name}")

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
    "CONSERVATIVE_ALLOCATION_POLICY",
    "CONSERVATIVE_BUDGET_POLICY",
    "CONSERVATIVE_LOGIT_COMPOSITION_POLICY",
    "CONSERVATIVE_MASS_POLICY",
    "CONSERVATIVE_OCCUPANCY_BURDEN_POLICY",
    "CONSERVATIVE_PHASE_AGGREGATION_POLICY",
    "CONSERVATIVE_RESIZE_POLICY",
    "ConservativeFactorizedDecoderConfig",
]
