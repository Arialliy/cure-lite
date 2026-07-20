"""Configuration contracts for full CURE counterfactual uncensoring.

The CURE-Lite configuration remains frozen in :mod:`cure_lite.config`.  The
objects in this module describe the full method without changing the Lite
baseline or making any backbone-specific assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(name: str, value: Integral, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class CUREResidualConfig:
    """Backbone-independent residual decoder and fusion configuration."""

    feature_channels: int
    width: int = 32
    groups: int = 8
    occupancy_threshold: float = 0.5
    suppression_radius: int = 0
    condition_on_probability: bool = False
    initial_residual_probability: float = 1e-4

    def __post_init__(self) -> None:
        channels = _integer("feature_channels", self.feature_channels, minimum=1)
        width = _integer("width", self.width, minimum=1)
        groups = _integer("groups", self.groups, minimum=1)
        radius = _integer("suppression_radius", self.suppression_radius)
        occupancy = _real("occupancy_threshold", self.occupancy_threshold)
        initial = _real(
            "initial_residual_probability", self.initial_residual_probability
        )
        if not isinstance(self.condition_on_probability, bool):
            raise TypeError("condition_on_probability must be bool")
        if width % groups:
            raise ValueError("groups must divide width")
        if not 0.0 <= occupancy <= 1.0:
            raise ValueError("occupancy_threshold must lie in [0,1]")
        if not 0.0 < initial < 0.5:
            raise ValueError("initial_residual_probability must lie in (0,0.5)")
        object.__setattr__(self, "feature_channels", channels)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "suppression_radius", radius)
        object.__setattr__(self, "occupancy_threshold", occupancy)
        object.__setattr__(self, "initial_residual_probability", initial)


@dataclass(frozen=True)
class CURELossConfig:
    """Fixed carrier loss for the single CURE intervention mechanism."""

    background_bce_weight: float = 1.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        background = _real("background_bce_weight", self.background_bce_weight)
        epsilon = _real("epsilon", self.epsilon)
        if background < 0.0:
            raise ValueError("background_bce_weight must be non-negative")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "background_bce_weight", background)
        object.__setattr__(self, "epsilon", epsilon)


@dataclass(frozen=True)
class DescriptorConfig:
    """Frozen geometry and numerical choices for propensity descriptors."""

    ring_inner_radius: int = 1
    ring_outer_radius: int = 4
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        inner = _integer("ring_inner_radius", self.ring_inner_radius)
        outer = _integer("ring_outer_radius", self.ring_outer_radius, minimum=1)
        epsilon = _real("epsilon", self.epsilon)
        if outer <= inner:
            raise ValueError("ring_outer_radius must exceed ring_inner_radius")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "ring_inner_radius", inner)
        object.__setattr__(self, "ring_outer_radius", outer)
        object.__setattr__(self, "epsilon", epsilon)


@dataclass(frozen=True)
class PropensityConfig:
    """Source-only grouped cross-fitting for miss-propensity estimation."""

    folds: int = 5
    l2: float = 1e-2
    max_iterations: int = 100
    tolerance: float = 1e-8
    clip_epsilon: float = 0.02
    max_odds: float = 10.0
    seed: int = 0

    def __post_init__(self) -> None:
        folds = _integer("folds", self.folds, minimum=2)
        iterations = _integer("max_iterations", self.max_iterations, minimum=1)
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        l2 = _real("l2", self.l2)
        tolerance = _real("tolerance", self.tolerance)
        epsilon = _real("clip_epsilon", self.clip_epsilon)
        max_odds = _real("max_odds", self.max_odds)
        if l2 <= 0.0:
            raise ValueError("l2 must be positive to keep cross-fit models well posed")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if not 0.0 < epsilon < 0.5:
            raise ValueError("clip_epsilon must lie in (0,0.5)")
        if max_odds <= 0.0:
            raise ValueError("max_odds must be positive")
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "l2", l2)
        object.__setattr__(self, "max_iterations", iterations)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "clip_epsilon", epsilon)
        object.__setattr__(self, "max_odds", max_odds)
        object.__setattr__(self, "seed", int(self.seed))


__all__ = [
    "CURELossConfig",
    "CUREResidualConfig",
    "DescriptorConfig",
    "PropensityConfig",
]
