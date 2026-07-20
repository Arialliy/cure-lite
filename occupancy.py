"""Fixed, GT-free occupancy construction for CURE-Lite v0.1."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import OccupancyConfig
from .instances import instances_from_binary_mask
from .types import InstanceMap


def _validated_probability(probability: Tensor) -> Tensor:
    if not isinstance(probability, Tensor):
        raise TypeError("probability must be a torch.Tensor")
    if probability.dtype != torch.float32:
        raise TypeError("probability must be float32")
    detached = probability.detach()
    if not torch.isfinite(detached).all():
        raise ValueError("probability contains non-finite values")
    if torch.any((detached < 0.0) | (detached > 1.0)):
        raise ValueError("probability must lie in [0,1]")
    return detached


def threshold_occupancy(probability: Tensor, threshold: float) -> Tensor:
    """Apply the inclusive occupancy threshold without GT or morphology."""

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a real number")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0,1]")
    return _validated_probability(probability) >= threshold


def _single_probability_map(probability: Tensor) -> Tensor:
    p = _validated_probability(probability).to(device="cpu", dtype=torch.float32)
    if p.ndim == 4 and p.shape[:2] == (1, 1):
        p = p[0, 0]
    elif p.ndim == 3 and p.shape[0] == 1:
        p = p[0]
    if p.ndim != 2:
        raise ValueError(f"expected one probability map, got shape {tuple(p.shape)}")
    if p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("probability spatial dimensions must be non-empty")
    return p.contiguous()


def decompose_occupancy(occupancy: Tensor) -> InstanceMap:
    """Decompose a v0.1 occupancy mask using fixed eight-connectivity."""

    return instances_from_binary_mask(occupancy, connectivity=8, min_area=1)


def build_occupancy(
    probability: Tensor,
    config: OccupancyConfig = OccupancyConfig(),
) -> tuple[Tensor, InstanceMap]:
    """Build one CPU occupancy mask and its deterministic components."""

    if not isinstance(config, OccupancyConfig):
        raise TypeError("config must be OccupancyConfig")
    p = _single_probability_map(probability)
    occupancy = p >= config.threshold
    instance_map = instances_from_binary_mask(
        occupancy,
        connectivity=config.connectivity,
        min_area=config.min_component_area,
    )
    if not torch.equal(occupancy, instance_map.occupancy):
        raise AssertionError("occupancy must equal (component labels > 0)")
    return occupancy, instance_map


def build_occupancy_batch(
    probability: Tensor,
    config: OccupancyConfig = OccupancyConfig(),
) -> Tensor:
    """Threshold a ``[B,1,H,W]`` batch while preserving its device."""

    if not isinstance(config, OccupancyConfig):
        raise TypeError("config must be OccupancyConfig")
    p = _validated_probability(probability)
    if p.ndim != 4 or p.shape[1] != 1:
        raise ValueError("probability must have shape [B,1,H,W]")
    if p.shape[0] == 0 or p.shape[2] == 0 or p.shape[3] == 0:
        raise ValueError("probability batch and spatial dimensions must be non-empty")
    return p >= config.threshold
