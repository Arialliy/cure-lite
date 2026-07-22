"""Standalone deterministic toy base and scenes for CURE contract tests."""

from .provider import ToyFrozenBaseAdapter
from .scenes import (
    TOY_GRID_SIZE,
    TOY_MISSED_TARGET_CONTRAST,
    TOY_OCCUPANCY_THRESHOLD,
    TOY_TARGET_CONTRAST,
    ToyScene,
    attenuate_target,
    make_custom_two_target_scene,
    make_factual_miss_scene,
    make_two_target_scene,
)

__all__ = [
    "TOY_GRID_SIZE",
    "TOY_MISSED_TARGET_CONTRAST",
    "TOY_OCCUPANCY_THRESHOLD",
    "TOY_TARGET_CONTRAST",
    "ToyFrozenBaseAdapter",
    "ToyScene",
    "attenuate_target",
    "make_custom_two_target_scene",
    "make_factual_miss_scene",
    "make_two_target_scene",
]
