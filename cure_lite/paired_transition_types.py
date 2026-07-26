"""Additive pair-local anchoring types for transition-supervised CURE-Lite.

The frozen paired-v1 objects remain unchanged.  This module wraps an existing
``PairBatch`` and adds only the information needed to identify the absolute
plus endpoint of a clean counterfactual transition.  The minus completion is
not stored: for a valid clean pair it is uniquely reconstructed as

``completion_plus | label_increment``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .paired_types import PairBatch, PairExample, stack_pair_examples


def _validated_bool_batch(
    value: Tensor,
    *,
    name: str,
    reference: Tensor,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must be bool")
    if value.device != reference.device:
        raise ValueError(f"{name} must share the PairBatch device")
    if value.shape != reference.shape:
        raise ValueError(f"{name} must match the PairBatch evaluation shape")
    return value.detach().clone().contiguous()


@dataclass(frozen=True)
class AnchoredPairBatch:
    """One clean ``PairBatch`` with its pair-local absolute plus-state anchor.

    ``completion_plus`` is the exact completion field before the occupancy
    intervention.  ``gt_union`` is used only to distinguish target support
    from writable background; it is supervision, not a decoder input.
    """

    pair_batch: PairBatch
    completion_plus: Tensor
    gt_union: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.pair_batch, PairBatch):
            raise TypeError("pair_batch must be a PairBatch")
        completion_plus = _validated_bool_batch(
            self.completion_plus,
            name="completion_plus",
            reference=self.pair_batch.image_valid_mask,
        )
        gt_union = _validated_bool_batch(
            self.gt_union,
            name="gt_union",
            reference=self.pair_batch.image_valid_mask,
        )
        object.__setattr__(self, "completion_plus", completion_plus)
        object.__setattr__(self, "gt_union", gt_union)
        self.validate()

    def validate(self) -> None:
        """Revalidate structure and semantics without mutating this batch.

        Tensors remain mutable even when their containing dataclass is frozen.
        The training preflight therefore calls this method on every update.
        Shape/device checks deliberately precede all boolean operations so
        PyTorch broadcasting cannot turn malformed supervision into a
        different, apparently valid batch.
        """

        if not isinstance(self.pair_batch, PairBatch):
            raise TypeError("pair_batch must be a PairBatch")
        self.pair_batch.validate()
        if any(
            kind != "clean_positive"
            for kind in self.pair_batch.pair_kinds
        ):
            raise ValueError("AnchoredPairBatch accepts only clean_positive pairs")
        reference = self.pair_batch.image_valid_mask
        for name in ("completion_plus", "gt_union"):
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must be bool")
            if value.device != reference.device:
                raise ValueError(f"{name} must share the PairBatch device")
            if value.shape != reference.shape:
                raise ValueError(
                    f"{name} must match the PairBatch evaluation shape"
                )

        valid = self.pair_batch.image_valid_mask
        occupancy_plus = self.pair_batch.occupancy_plus
        occupancy_minus = self.pair_batch.occupancy_minus
        increment = self.pair_batch.label_increment.to(dtype=torch.bool)
        completion_plus = self.completion_plus
        gt_union = self.gt_union

        if torch.any(completion_plus & increment):
            raise ValueError(
                "completion_plus and label_increment must be disjoint"
            )
        if torch.any(completion_plus & (~valid | occupancy_plus)):
            raise ValueError(
                "completion_plus must be valid and writable under occupancy_plus"
            )
        if torch.any(gt_union & ~valid):
            raise ValueError("gt_union must remain inside image_valid_mask")
        if torch.any(completion_plus & ~gt_union):
            raise ValueError("completion_plus must contain only gt_union pixels")
        if torch.any(increment & ~gt_union):
            raise ValueError("label_increment must contain only gt_union pixels")

        completion_minus = completion_plus | increment
        if torch.any(completion_minus & (~valid | occupancy_minus)):
            raise ValueError(
                "reconstructed completion_minus must be valid and writable "
                "under occupancy_minus"
            )
        if not torch.equal(
            completion_minus & ~completion_plus,
            increment,
        ):
            raise ValueError(
                "label_increment is inconsistent with the anchored transition"
            )

    @property
    def completion_minus(self) -> Tensor:
        """Reconstruct the unique minus completion without storing a copy."""

        return (
            self.completion_plus
            | self.pair_batch.label_increment.to(dtype=torch.bool)
        )

    @property
    def feature(self) -> Tensor:
        return self.pair_batch.feature

    @property
    def occupancy_plus(self) -> Tensor:
        return self.pair_batch.occupancy_plus

    @property
    def occupancy_minus(self) -> Tensor:
        return self.pair_batch.occupancy_minus

    @property
    def label_increment(self) -> Tensor:
        return self.pair_batch.label_increment

    @property
    def image_valid_mask(self) -> Tensor:
        return self.pair_batch.image_valid_mask

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return self.pair_batch.pair_ids

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.pair_batch.sample_ids

    @property
    def group_ids(self) -> tuple[str, ...]:
        return self.pair_batch.group_ids

    @property
    def pair_kinds(self) -> tuple[str, ...]:
        return self.pair_batch.pair_kinds


def stack_anchored_pair_examples(
    examples: Iterable[PairExample],
    *,
    gt_union_by_sample: Mapping[str, Tensor],
    device: torch.device | str,
) -> AnchoredPairBatch:
    """Stack clean examples and their exact per-source GT unions.

    The function delegates feature/occupancy/identity stacking to the frozen
    ``stack_pair_examples`` implementation and independently verifies that the
    stored plus completion and increment reconstruct each example's exact
    minus completion before moving supervision to ``device``.
    """

    values = tuple(examples)
    if not values:
        raise ValueError("cannot stack an empty anchored pair selection")
    if any(not isinstance(value, PairExample) for value in values):
        raise TypeError("examples must contain only PairExample values")
    if any(value.pair_kind != "clean_positive" for value in values):
        raise ValueError("anchored stacking accepts only clean_positive pairs")
    if not isinstance(gt_union_by_sample, Mapping):
        raise TypeError("gt_union_by_sample must be a mapping")

    evaluation_shape = tuple(values[0].completion_plus.shape)
    unions: list[Tensor] = []
    for example in values:
        if tuple(example.completion_plus.shape) != evaluation_shape:
            raise ValueError("selected examples must share an evaluation shape")
        expected_minus = (
            example.completion_plus
            | example.label_increment.to(dtype=torch.bool)
        )
        if not torch.equal(expected_minus, example.completion_minus):
            raise ValueError(
                "PairExample completion fields do not match label_increment"
            )
        if example.sample_id not in gt_union_by_sample:
            raise KeyError(
                f"gt_union_by_sample is missing {example.sample_id!r}"
            )
        gt_union = gt_union_by_sample[example.sample_id]
        if not isinstance(gt_union, Tensor):
            raise TypeError("every gt_union_by_sample value must be a tensor")
        if gt_union.device.type != "cpu" or gt_union.dtype != torch.bool:
            raise TypeError(
                "every gt_union_by_sample value must be a CPU bool tensor"
            )
        if tuple(gt_union.shape) != evaluation_shape:
            raise ValueError(
                "every gt_union_by_sample value must match PairExample "
                "evaluation shape"
            )
        unions.append(gt_union.detach().clone().contiguous())

    pair_batch = stack_pair_examples(values, device=device)
    return AnchoredPairBatch(
        pair_batch=pair_batch,
        completion_plus=torch.stack(
            [value.completion_plus for value in values],
            dim=0,
        ).to(device=device),
        gt_union=torch.stack(unions, dim=0).to(device=device),
    )


__all__ = [
    "AnchoredPairBatch",
    "stack_anchored_pair_examples",
]
