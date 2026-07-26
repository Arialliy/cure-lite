"""Frozen configuration for the additive CURE-Lite D-SVEF v5 decoder."""

from __future__ import annotations

from dataclasses import dataclass

from .factorized_config import (
    SVEF_RESIZE_POLICY,
    FactorizedDecoderConfig,
)


DIRECTED_EVIDENCE_TRANSFORM = (
    "one_sided_zero_anchored_softplus_relu_v1"
)


@dataclass(frozen=True)
class DirectedFactorizedDecoderConfig:
    """Truthful v5 config with the v4 topology and one directed activation."""

    feature_channels: int
    feature_stride: int
    width: int = 32
    groups: int = 8
    trunk_residual_scale: float = 0.5
    baseline_probability: float = 0.1
    vacancy_kernel_size: int = 3
    evidence_transform: str = DIRECTED_EVIDENCE_TRANSFORM
    resize_policy: str = SVEF_RESIZE_POLICY

    def __post_init__(self) -> None:
        if self.evidence_transform != DIRECTED_EVIDENCE_TRANSFORM:
            raise ValueError("D-SVEF v5 fixes the directed evidence transform")
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
        """Return the byte-identical v4 module topology/initialization config."""

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
    "DIRECTED_EVIDENCE_TRANSFORM",
    "DirectedFactorizedDecoderConfig",
]
