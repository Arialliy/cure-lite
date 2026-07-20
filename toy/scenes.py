"""Deterministic toy scenes for exercising the frozen-base CURE contracts.

The scenes deliberately contain no dependency on a real detector or dataset.
They model two separated, three-by-three targets on a fixed single-channel
grid.  Reducing one target's local contrast below the toy base's operating
point removes exactly that prediction while leaving the other target intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


TOY_GRID_SIZE = 32
TOY_OCCUPANCY_THRESHOLD = 0.5
TOY_TARGET_CONTRAST = 1.0
TOY_MISSED_TARGET_CONTRAST = 0.35


@dataclass(frozen=True, eq=False)
class ToyScene:
    """One immutable-by-contract CPU toy scene and its instance masks."""

    sample_id: str
    image: Tensor
    gt_mask: Tensor
    target_masks: tuple[Tensor, ...]
    attenuated_gt_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.image, Tensor) or self.image.ndim != 3:
            raise ValueError("image must have shape [1,H,W]")
        if self.image.shape[0] != 1 or self.image.dtype != torch.float32:
            raise TypeError("image must be a float32 single-channel tensor")
        if self.image.device.type != "cpu":
            raise ValueError("toy scenes must be constructed on CPU")
        if not torch.isfinite(self.image).all() or torch.any(
            (self.image < 0.0) | (self.image > 1.0)
        ):
            raise ValueError("image values must be finite and lie in [0,1]")
        if (
            not isinstance(self.gt_mask, Tensor)
            or self.gt_mask.shape != self.image.shape
            or self.gt_mask.dtype != torch.bool
            or self.gt_mask.device.type != "cpu"
        ):
            raise TypeError("gt_mask must be a CPU bool tensor matching image")
        if not isinstance(self.target_masks, tuple) or len(self.target_masks) < 2:
            raise ValueError("a toy scene requires at least two target masks")

        represented = torch.zeros_like(self.gt_mask[0])
        for target_mask in self.target_masks:
            if (
                not isinstance(target_mask, Tensor)
                or target_mask.shape != self.image.shape[-2:]
                or target_mask.dtype != torch.bool
                or target_mask.device.type != "cpu"
            ):
                raise TypeError("target masks must be CPU bool [H,W] tensors")
            if not torch.any(target_mask):
                raise ValueError("target masks cannot be empty")
            if torch.any(represented & target_mask):
                raise ValueError("target masks must be disjoint")
            represented |= target_mask
        if not torch.equal(represented, self.gt_mask[0]):
            raise ValueError("the target-mask union must exactly equal gt_mask")

        if self.attenuated_gt_id is not None:
            if (
                isinstance(self.attenuated_gt_id, bool)
                or not isinstance(self.attenuated_gt_id, int)
                or not 1 <= self.attenuated_gt_id <= len(self.target_masks)
            ):
                raise ValueError("attenuated_gt_id must identify a toy target")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.image.shape[-2]), int(self.image.shape[-1])

    def image_batch(self) -> Tensor:
        """Return the NCHW batch consumed by :class:`ToyFrozenBaseAdapter`."""

        return self.image.unsqueeze(0)


def _square_mask(size: int, top: int, left: int, width: int = 3) -> Tensor:
    mask = torch.zeros((size, size), dtype=torch.bool)
    mask[top : top + width, left : left + width] = True
    return mask


def make_custom_two_target_scene(
    *,
    sample_id: str,
    target_top_lefts: tuple[tuple[int, int], tuple[int, int]],
    distractor_points: tuple[tuple[int, int], ...] = (),
    size: int = TOY_GRID_SIZE,
) -> ToyScene:
    """Build a translated scene with optional subthreshold point distractors.

    A distractor has the same intensity as a factual missed-target pixel but a
    different local shape.  Lowering the toy base threshold far enough to
    recover the 3x3 missed target therefore also creates distractor false
    positives; a convolutional repair head can instead use local structure.
    """

    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if isinstance(size, bool) or not isinstance(size, int) or size < 24:
        raise ValueError("size must be an integer of at least 24")
    if (
        not isinstance(target_top_lefts, tuple)
        or len(target_top_lefts) != 2
    ):
        raise ValueError("target_top_lefts must contain exactly two positions")
    targets: list[Tensor] = []
    for position in target_top_lefts:
        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in position)
        ):
            raise TypeError("every target position must contain two integers")
        top, left = position
        if not (0 <= top <= size - 3 and 0 <= left <= size - 3):
            raise ValueError("target position lies outside the scene")
        targets.append(_square_mask(size, top, left))
    if torch.any(targets[0] & targets[1]):
        raise ValueError("target squares must be disjoint")

    image = torch.zeros((1, size, size), dtype=torch.float32)
    gt_union = targets[0] | targets[1]
    image[0][gt_union] = TOY_TARGET_CONTRAST
    if not isinstance(distractor_points, tuple):
        raise TypeError("distractor_points must be a tuple")
    seen: set[tuple[int, int]] = set()
    for point in distractor_points:
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in point)
        ):
            raise TypeError("every distractor point must contain two integers")
        y, x = point
        if not (0 <= y < size and 0 <= x < size):
            raise ValueError("distractor point lies outside the scene")
        if point in seen:
            raise ValueError("distractor points must be unique")
        if gt_union[y, x]:
            raise ValueError("a distractor cannot overlap a target")
        seen.add(point)
        image[0, y, x] = TOY_MISSED_TARGET_CONTRAST

    return ToyScene(
        sample_id=sample_id,
        image=image,
        gt_mask=gt_union.unsqueeze(0),
        target_masks=tuple(targets),
    )


def make_two_target_scene(*, size: int = TOY_GRID_SIZE) -> ToyScene:
    """Return a clean scene whose two targets are both base-detectable."""

    return make_custom_two_target_scene(
        sample_id=f"toy-two-targets-{size}",
        target_top_lefts=((6, 6), (size - 9, size - 9)),
        size=size,
    )


def attenuate_target(
    scene: ToyScene,
    gt_id: int,
    *,
    contrast: float = TOY_MISSED_TARGET_CONTRAST,
    sample_id: str | None = None,
) -> ToyScene:
    """Locally attenuate exactly one target below the toy operating point."""

    if not isinstance(scene, ToyScene):
        raise TypeError("scene must be a ToyScene")
    if isinstance(gt_id, bool) or not isinstance(gt_id, int):
        raise TypeError("gt_id must be an integer")
    if not 1 <= gt_id <= len(scene.target_masks):
        raise ValueError("gt_id does not identify a target in this scene")
    if isinstance(contrast, bool) or not isinstance(contrast, (int, float)):
        raise TypeError("contrast must be numeric")
    contrast = float(contrast)
    # The toy base maps p = 0.1 + 0.8*x, so p < 0.5 exactly when x < 0.5.
    if not 0.0 < contrast < 0.5:
        raise ValueError("attenuated contrast must lie strictly in (0,0.5)")

    image = scene.image.clone()
    selected = scene.target_masks[gt_id - 1]
    image[0][selected] = contrast
    return ToyScene(
        sample_id=(
            sample_id
            if sample_id is not None
            else f"{scene.sample_id}-attenuated-gt{gt_id}"
        ),
        image=image,
        gt_mask=scene.gt_mask.clone(),
        target_masks=tuple(mask.clone() for mask in scene.target_masks),
        attenuated_gt_id=gt_id,
    )


def make_factual_miss_scene(
    *,
    size: int = TOY_GRID_SIZE,
    missed_gt_id: int = 1,
) -> ToyScene:
    """Return a factual-miss scene with subthreshold but nonzero target evidence."""

    return attenuate_target(
        make_two_target_scene(size=size),
        missed_gt_id,
        sample_id=f"toy-factual-miss-gt{missed_gt_id}-{size}",
    )


__all__ = [
    "TOY_GRID_SIZE",
    "TOY_MISSED_TARGET_CONTRAST",
    "TOY_OCCUPANCY_THRESHOLD",
    "TOY_TARGET_CONTRAST",
    "ToyScene",
    "attenuate_target",
    "make_custom_two_target_scene",
    "make_factual_miss_scene",
    "make_two_target_scene",
]
