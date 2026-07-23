"""Validated configuration objects for the CURE-Lite v0.1 core.

The method specification intentionally fixes several choices.  In particular,
occupancy uses inclusive thresholding, eight-connected components, and no
component filtering.  The validation below makes those frozen choices explicit
instead of accepting a configuration that would silently define another method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True)
class OccupancyConfig:
    """Frozen occupancy definition for CURE-Lite v0.1."""

    threshold: float = 0.5
    connectivity: int = 8
    min_component_area: int = 1

    def __post_init__(self) -> None:
        threshold = _finite_number("threshold", self.threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        object.__setattr__(self, "threshold", threshold)
        _positive_integer("connectivity", self.connectivity)
        _positive_integer("min_component_area", self.min_component_area)
        if self.connectivity != 8:
            raise ValueError("CURE-Lite v0.1 fixes occupancy connectivity at 8")
        if self.min_component_area != 1:
            raise ValueError("CURE-Lite v0.1 fixes min_component_area at 1")


@dataclass(frozen=True)
class MatchConfig:
    """Configuration for deterministic lexicographic component matching."""

    max_distance: float = 3.0
    distance_quantization: int = 1_000_000
    iou_quantization: int = 1_000_000

    def __post_init__(self) -> None:
        max_distance = _finite_number("max_distance", self.max_distance)
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive")
        object.__setattr__(self, "max_distance", max_distance)
        _positive_integer("distance_quantization", self.distance_quantization)
        _positive_integer("iou_quantization", self.iou_quantization)


@dataclass(frozen=True)
class InterventionConfig:
    """Configuration for strict legal single-component deletion."""

    min_writable_pixels: int = 1

    def __post_init__(self) -> None:
        _positive_integer("min_writable_pixels", self.min_writable_pixels)
        if self.min_writable_pixels != 1:
            raise ValueError("CURE-Lite v0.1 fixes min_writable_pixels at 1")


MISS_ALIGNMENT_POLICY = (
    "global-decoder-visible-positive-region-log1p-feature-rms-nearest-v1"
)


@dataclass(frozen=True)
class MissAlignmentConfig:
    """Frozen target-state alignment rule for CURE-Lite v0.2.

    The policy deliberately exposes no tunable neighbourhood size, temperature,
    or fallback.  It maps each reachable factual miss to the globally nearest
    decoder-visible legal intervention using only the frozen feature RMS over
    the positive supervision region.
    """

    policy: str = MISS_ALIGNMENT_POLICY
    distance_quantization: int = 1_000_000

    def __post_init__(self) -> None:
        if self.policy != MISS_ALIGNMENT_POLICY:
            raise ValueError(
                "CURE-Lite v0.2 fixes the miss-alignment policy"
            )
        _positive_integer(
            "distance_quantization",
            self.distance_quantization,
        )
        if self.distance_quantization != 1_000_000:
            raise ValueError(
                "CURE-Lite v0.2 fixes distance_quantization at 1000000"
            )


@dataclass(frozen=True)
class DecoderConfig:
    """Frozen residual-decoder topology, parameterized only by input channels."""

    feature_channels: int
    width: int = 32
    groups: int = 8

    def __post_init__(self) -> None:
        _positive_integer("feature_channels", self.feature_channels)
        _positive_integer("width", self.width)
        _positive_integer("groups", self.groups)
        if self.width != 32 or self.groups != 8:
            raise ValueError("CURE-Lite v0.1 fixes decoder width/groups at 32/8")
        if self.width % self.groups:
            raise ValueError("groups must divide width")


@dataclass(frozen=True)
class LossConfig:
    """Masked balanced-logistic plus positive-only Dice configuration."""

    dice_weight: float = 1.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        dice_weight = _finite_number("dice_weight", self.dice_weight)
        epsilon = _finite_number("epsilon", self.epsilon)
        if dice_weight < 0.0:
            raise ValueError("dice_weight must be non-negative")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "dice_weight", dice_weight)
        object.__setattr__(self, "epsilon", epsilon)


@dataclass(frozen=True)
class TrainingConfig:
    """Weights used when independently averaged branch losses are combined."""

    lambda_no_miss: float = 1.0
    lambda_synthetic: float = 1.0

    def __post_init__(self) -> None:
        no_miss = _finite_number("lambda_no_miss", self.lambda_no_miss)
        synthetic = _finite_number("lambda_synthetic", self.lambda_synthetic)
        if no_miss < 0.0 or synthetic < 0.0:
            raise ValueError("branch weights must be non-negative")
        object.__setattr__(self, "lambda_no_miss", no_miss)
        object.__setattr__(self, "lambda_synthetic", synthetic)


@dataclass(frozen=True)
class CalibrationConfig:
    """Pre-registered total budgets for threshold selection."""

    pixel_fa_budget: float
    component_fa_per_mp_budget: float = float("inf")
    raw_background_fa_budget: float = float("inf")
    minimum_retention: float = 0.0

    def __post_init__(self) -> None:
        pixel_budget = _finite_number("pixel_fa_budget", self.pixel_fa_budget)
        if pixel_budget < 0.0:
            raise ValueError("pixel_fa_budget must be non-negative")
        component_budget = self.component_fa_per_mp_budget
        if isinstance(component_budget, bool) or not isinstance(component_budget, Real):
            raise TypeError("component_fa_per_mp_budget must be a real number")
        component_budget = float(component_budget)
        if component_budget < 0.0 or component_budget != component_budget:
            raise ValueError("component_fa_per_mp_budget must be non-negative")
        raw_background_budget = self.raw_background_fa_budget
        if (
            isinstance(raw_background_budget, bool)
            or not isinstance(raw_background_budget, Real)
        ):
            raise TypeError("raw_background_fa_budget must be a real number")
        raw_background_budget = float(raw_background_budget)
        if raw_background_budget < 0.0 or raw_background_budget != raw_background_budget:
            raise ValueError("raw_background_fa_budget must be non-negative")
        minimum_retention = _finite_number(
            "minimum_retention", self.minimum_retention
        )
        if not 0.0 <= minimum_retention <= 1.0:
            raise ValueError("minimum_retention must lie in [0, 1]")
        object.__setattr__(self, "pixel_fa_budget", pixel_budget)
        object.__setattr__(self, "component_fa_per_mp_budget", component_budget)
        object.__setattr__(self, "raw_background_fa_budget", raw_background_budget)
        object.__setattr__(self, "minimum_retention", minimum_retention)


def config_to_dict(config: Any) -> dict[str, Any]:
    """Return a dataclass configuration as a fingerprint-friendly dictionary."""

    try:
        return asdict(config)
    except TypeError as exc:
        raise TypeError("config must be a dataclass instance") from exc
