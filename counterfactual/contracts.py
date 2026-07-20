"""Strict contracts for CURE input-level counterfactual transforms.

The counterfactual package is deliberately independent of any detector.  It
only constructs deterministic image candidates; a later search layer is
responsible for running a frozen detector and deciding whether a candidate is
an admissible failure state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


LOCAL_CONTRAST_ATTENUATION = (
    "local_target_to_ring_background_contrast_attenuation_v1"
)


def _integer(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(name: str, value: float, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class TransformConfig:
    """Frozen candidate-family configuration for local contrast attenuation.

    Distances use a Chebyshev neighbourhood, matching square max-pooling
    morphology exactly.  ``roi_radius`` is positive so every candidate has a
    genuinely soft boundary.  The background ring starts strictly outside the
    editable ROI and is therefore never modified by the transform.
    """

    roi_radius: int = 2
    ring_inner_radius: int = 2
    ring_outer_radius: int = 5
    strength_grid: tuple[float, ...] = (0.25, 0.5, 0.75)
    minimum_ring_pixels: int = 1

    def __post_init__(self) -> None:
        roi_radius = _integer("roi_radius", self.roi_radius, minimum=1)
        ring_inner = _integer(
            "ring_inner_radius", self.ring_inner_radius, minimum=1
        )
        ring_outer = _integer(
            "ring_outer_radius", self.ring_outer_radius, minimum=1
        )
        minimum_ring_pixels = _integer(
            "minimum_ring_pixels", self.minimum_ring_pixels, minimum=1
        )
        if ring_inner < roi_radius:
            raise ValueError("ring_inner_radius must be at least roi_radius")
        if ring_outer <= ring_inner:
            raise ValueError("ring_outer_radius must exceed ring_inner_radius")
        if not isinstance(self.strength_grid, tuple):
            raise TypeError("strength_grid must be a tuple")
        if not self.strength_grid:
            raise ValueError("strength_grid must not be empty")
        strengths = tuple(
            _finite(f"strength_grid[{index}]", value)
            for index, value in enumerate(self.strength_grid)
        )
        if any(not 0.0 < value < 1.0 for value in strengths):
            raise ValueError("every attenuation strength must lie strictly in (0,1)")
        if strengths != tuple(sorted(set(strengths))):
            raise ValueError("strength_grid must be strictly increasing and unique")

        object.__setattr__(self, "roi_radius", roi_radius)
        object.__setattr__(self, "ring_inner_radius", ring_inner)
        object.__setattr__(self, "ring_outer_radius", ring_outer)
        object.__setattr__(self, "strength_grid", strengths)
        object.__setattr__(self, "minimum_ring_pixels", minimum_ring_pixels)


@dataclass(frozen=True)
class TransformSpec:
    """One concrete member of a configured finite transform family."""

    config: TransformConfig
    strength: float
    transform_name: str = LOCAL_CONTRAST_ATTENUATION

    def __post_init__(self) -> None:
        if not isinstance(self.config, TransformConfig):
            raise TypeError("config must be TransformConfig")
        if self.transform_name != LOCAL_CONTRAST_ATTENUATION:
            raise ValueError(
                f"transform_name must be exactly {LOCAL_CONTRAST_ATTENUATION!r}"
            )
        strength = _finite("strength", self.strength)
        if strength not in self.config.strength_grid:
            raise ValueError("strength must be one of config.strength_grid")
        object.__setattr__(self, "strength", strength)


@dataclass(frozen=True)
class TransformDiagnostics:
    """Auditable sufficient diagnostics for one transformed image.

    Delta summaries are computed over all image elements.  ``ring_pixels`` is
    a spatial count and is not multiplied by the number of image channels.
    """

    max_abs_delta: float
    mean_abs_delta: float
    outside_roi_max_delta: float
    ring_pixels: int

    def __post_init__(self) -> None:
        for name in (
            "max_abs_delta",
            "mean_abs_delta",
            "outside_roi_max_delta",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        ring_pixels = _integer("ring_pixels", self.ring_pixels, minimum=1)
        if self.mean_abs_delta > self.max_abs_delta:
            raise ValueError("mean_abs_delta cannot exceed max_abs_delta")
        if self.outside_roi_max_delta > self.max_abs_delta:
            raise ValueError("outside_roi_max_delta cannot exceed max_abs_delta")
        object.__setattr__(self, "ring_pixels", ring_pixels)


__all__ = [
    "LOCAL_CONTRAST_ATTENUATION",
    "TransformConfig",
    "TransformDiagnostics",
    "TransformSpec",
]
