"""Frozen configuration for the additive CURE-Lite SVEF v4 decoder."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


SVEF_EVIDENCE_TRANSFORM = "zero_anchored_squared_softplus_v1"
SVEF_RESIZE_POLICY = "separate_fields_then_final_gate_v1"


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class FactorizedDecoderConfig:
    """Adapter-bound topology for Subpixel Vacancy-Evidence Factorization.

    ``feature_channels`` and ``feature_stride`` come from a frozen detector
    adapter/cache receipt.  Every remaining field is a method constant rather
    than a search surface.
    """

    feature_channels: int
    feature_stride: int
    width: int = 32
    groups: int = 8
    trunk_residual_scale: float = 0.5
    baseline_probability: float = 0.1
    vacancy_kernel_size: int = 3
    evidence_transform: str = SVEF_EVIDENCE_TRANSFORM
    resize_policy: str = SVEF_RESIZE_POLICY

    def __post_init__(self) -> None:
        channels = _positive_integer(
            self.feature_channels,
            name="feature_channels",
        )
        stride = _positive_integer(
            self.feature_stride,
            name="feature_stride",
        )
        width = _positive_integer(self.width, name="width")
        groups = _positive_integer(self.groups, name="groups")
        kernel = _positive_integer(
            self.vacancy_kernel_size,
            name="vacancy_kernel_size",
        )
        residual_scale = _finite_float(
            self.trunk_residual_scale,
            name="trunk_residual_scale",
        )
        baseline_probability = _finite_float(
            self.baseline_probability,
            name="baseline_probability",
        )
        object.__setattr__(self, "feature_channels", channels)
        object.__setattr__(self, "feature_stride", stride)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "vacancy_kernel_size", kernel)
        object.__setattr__(
            self,
            "trunk_residual_scale",
            residual_scale,
        )
        object.__setattr__(
            self,
            "baseline_probability",
            baseline_probability,
        )

        if width != 32 or groups != 8:
            raise ValueError("SVEF v4 fixes width/groups at 32/8")
        if width % groups:
            raise ValueError("groups must divide width")
        if residual_scale != 0.5:
            raise ValueError("SVEF v4 fixes trunk_residual_scale at 0.5")
        if baseline_probability != 0.1:
            raise ValueError("SVEF v4 fixes baseline_probability at 0.1")
        if kernel != 3:
            raise ValueError("SVEF v4 fixes vacancy_kernel_size at 3")
        if self.evidence_transform != SVEF_EVIDENCE_TRANSFORM:
            raise ValueError("SVEF v4 fixes the evidence transform")
        if self.resize_policy != SVEF_RESIZE_POLICY:
            raise ValueError("SVEF v4 fixes the field resize policy")

    @property
    def phase_channels(self) -> int:
        """Channels decoded into one subpixel field by PixelShuffle."""

        return self.feature_stride**2

    @property
    def expected_parameter_count(self) -> int:
        """Exact trainable parameter count for the frozen topology."""

        return (
            self.width * self.feature_channels
            + self.width * 3 * 3
            + self.width * self.width
            + 2 * self.width * self.phase_channels
            + 1
        )


__all__ = [
    "FactorizedDecoderConfig",
    "SVEF_EVIDENCE_TRANSFORM",
    "SVEF_RESIZE_POLICY",
]
