from __future__ import annotations

from collections import deque

import pytest
import torch

from cure_lite.instances import instances_from_binary_mask


def _legacy_labels(
    mask: torch.Tensor,
    *,
    connectivity: int,
    min_area: int,
) -> tuple[torch.Tensor, tuple[tuple[int, tuple[int, int, int, int], tuple[float, float]], ...]]:
    binary = torch.as_tensor(mask, device="cpu", dtype=torch.bool).contiguous()
    height, width = binary.shape
    visited = torch.zeros_like(binary)
    offsets = (
        ((-1, 0), (0, -1), (0, 1), (1, 0))
        if connectivity == 4
        else (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    )
    components: list[
        tuple[
            tuple[float, float, float, float],
            tuple[tuple[int, int], ...],
        ]
    ] = []
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
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and bool(binary[ny, nx])
                        and not bool(visited[ny, nx])
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(pixels) < min_area:
                continue
            ys = [pixel[0] for pixel in pixels]
            xs = [pixel[1] for pixel in pixels]
            components.append(
                (
                    (
                        float(min(ys)),
                        float(min(xs)),
                        float(sum(ys) / len(ys)),
                        float(sum(xs) / len(xs)),
                    ),
                    tuple(pixels),
                )
            )
    components.sort(key=lambda item: item[0])
    labels = torch.zeros((height, width), dtype=torch.int64)
    metadata: list[
        tuple[int, tuple[int, int, int, int], tuple[float, float]]
    ] = []
    for instance_id, (_, pixels) in enumerate(components, start=1):
        ys = [pixel[0] for pixel in pixels]
        xs = [pixel[1] for pixel in pixels]
        for y, x in pixels:
            labels[y, x] = instance_id
        area = len(pixels)
        metadata.append(
            (
                area,
                (min(ys), min(xs), max(ys) + 1, max(xs) + 1),
                (sum(ys) / area, sum(xs) / area),
            )
        )
    return labels, tuple(metadata)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("min_area", [1, 2, 5])
def test_foreground_only_cc_is_exactly_legacy_equivalent(
    connectivity: int,
    min_area: int,
) -> None:
    generator = torch.Generator().manual_seed(1701 + connectivity + min_area)
    masks = [
        torch.zeros((17, 19), dtype=torch.bool),
        torch.ones((17, 19), dtype=torch.bool),
        torch.rand((17, 19), generator=generator) < 0.08,
        torch.rand((17, 19), generator=generator) < 0.55,
    ]
    diagonal = torch.zeros((17, 19), dtype=torch.bool)
    diagonal[torch.arange(17), torch.arange(17)] = True
    masks.append(diagonal)

    for mask in masks:
        expected_labels, expected_metadata = _legacy_labels(
            mask,
            connectivity=connectivity,
            min_area=min_area,
        )
        actual = instances_from_binary_mask(
            mask,
            connectivity=connectivity,
            min_area=min_area,
        )
        assert torch.equal(actual.labels, expected_labels)
        assert tuple(
            (item.area, item.bbox, item.centroid) for item in actual.instances
        ) == expected_metadata
        for item in actual.instances:
            assert torch.equal(item.mask, actual.labels == item.instance_id)
