"""Deterministic connected components and instance geometry helpers."""

from __future__ import annotations

from collections import deque
from math import hypot
from typing import Iterable

import torch
from torch import Tensor

from .types import Instance, InstanceMap


def _as_cpu_bool_2d(mask: Tensor) -> Tensor:
    if not isinstance(mask, Tensor):
        mask = torch.as_tensor(mask)
    if mask.is_complex():
        raise TypeError("binary mask may not have a complex dtype")
    if mask.is_floating_point() and not torch.isfinite(mask).all():
        raise ValueError("binary mask contains non-finite values")
    tensor = mask.detach().to(device="cpu", dtype=torch.bool)
    if tensor.ndim == 4 and tensor.shape[:2] == (1, 1):
        tensor = tensor[0, 0]
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"expected [H,W], [1,H,W], or [1,1,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError("binary mask spatial dimensions must be non-empty")
    return tensor.contiguous()


def _neighbours(y: int, x: int, height: int, width: int, connectivity: int):
    if connectivity == 4:
        offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    elif connectivity == 8:
        offsets = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )
    else:
        raise ValueError("connectivity must be 4 or 8")
    for dy, dx in offsets:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            yield ny, nx


def instances_from_binary_mask(
    mask: Tensor,
    *,
    connectivity: int = 8,
    min_area: int = 1,
) -> InstanceMap:
    """Decompose a binary mask into reproducibly numbered components.

    Components are sorted by ``(bbox_ymin, bbox_xmin, centroid_y,
    centroid_x)`` and assigned one-based IDs.  The implementation is CPU-only
    by design because matching/state construction is an offline operation.
    """

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if isinstance(min_area, bool) or not isinstance(min_area, int):
        raise TypeError("min_area must be an integer")
    if min_area < 1:
        raise ValueError("min_area must be at least 1")

    binary = _as_cpu_bool_2d(mask)
    height, width = binary.shape
    visited = torch.zeros_like(binary)
    components: list[tuple[tuple[float, float, float, float], tuple[tuple[int, int], ...]]] = []

    # Row-major discovery is deterministic; the explicit final sort below is
    # still retained because it is part of the public ID contract.
    for y0 in range(height):
        for x0 in range(width):
            if not bool(binary[y0, x0]) or bool(visited[y0, x0]):
                continue
            visited[y0, x0] = True
            queue: deque[tuple[int, int]] = deque([(y0, x0)])
            pixels: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))
                for ny, nx in _neighbours(y, x, height, width, connectivity):
                    if bool(binary[ny, nx]) and not bool(visited[ny, nx]):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(pixels) < min_area:
                continue
            ys = [pixel[0] for pixel in pixels]
            xs = [pixel[1] for pixel in pixels]
            ymin, xmin = min(ys), min(xs)
            cy = sum(ys) / len(ys)
            cx = sum(xs) / len(xs)
            key = (float(ymin), float(xmin), float(cy), float(cx))
            components.append((key, tuple(pixels)))

    components.sort(key=lambda item: item[0])
    labels = torch.zeros((height, width), dtype=torch.int64)
    records: list[Instance] = []
    for instance_id, (_, pixels) in enumerate(components, start=1):
        component_mask = torch.zeros((height, width), dtype=torch.bool)
        ys: list[int] = []
        xs: list[int] = []
        for y, x in pixels:
            component_mask[y, x] = True
            labels[y, x] = instance_id
            ys.append(y)
            xs.append(x)
        area = len(pixels)
        bbox = (min(ys), min(xs), max(ys) + 1, max(xs) + 1)
        centroid = (sum(ys) / area, sum(xs) / area)
        records.append(
            Instance(
                instance_id=instance_id,
                mask=component_mask,
                area=area,
                bbox=bbox,
                centroid=centroid,
            )
        )
    return InstanceMap(labels=labels, instances=tuple(records))


def centroid_distance(first: Instance, second: Instance) -> float:
    """Return Euclidean centroid distance in evaluation-grid pixels."""

    return float(hypot(first.centroid[0] - second.centroid[0], first.centroid[1] - second.centroid[1]))


def mask_iou(first: Tensor, second: Tensor) -> float:
    """Return binary mask intersection-over-union."""

    a = _as_cpu_bool_2d(first)
    b = _as_cpu_bool_2d(second)
    if a.shape != b.shape:
        raise ValueError("mask shapes differ")
    union = int(torch.count_nonzero(a | b))
    if union == 0:
        return 0.0
    intersection = int(torch.count_nonzero(a & b))
    return intersection / union


def union_instance_masks(instance_map: InstanceMap, ids: Iterable[int]) -> Tensor:
    """Return the union of selected instance masks, validating every ID."""

    result = torch.zeros(instance_map.shape, dtype=torch.bool)
    for instance_id in sorted(set(ids)):
        result |= instance_map.by_id(instance_id).mask
    return result
