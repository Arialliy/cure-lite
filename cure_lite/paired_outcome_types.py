"""Outcome-complete paired value objects for OC-APTO v3.

This module is additive: it leaves the frozen v1/v2 ``PairExample`` and
``PairBatch`` contracts untouched.  It binds exact completion endpoints and
the direct projected conditioning footprint to one already-stacked
``PairBatch`` containing clean-positive and/or component-null interventions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .decoder import project_occupancy_to_feature_grid
from .paired_types import PairBatch, PairExample, stack_pair_examples


OUTCOME_PAIR_KINDS = ("clean_positive", "component_null")


def _validate_bool_batch_tensor(
    value: Tensor,
    *,
    name: str,
    reference: Tensor,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must be bool")
    if value.shape != reference.shape:
        raise ValueError(f"{name} must match the paired evaluation shape")
    if value.device != reference.device:
        raise ValueError(f"{name} must share the PairBatch device")


def direct_projected_intervention_footprint(batch: PairBatch) -> Tensor:
    """Return the frozen direct projected conditioning footprint ``J``.

    ``C = O_plus \\ O_minus`` is unioned with a nearest-neighbor lift of the
    feature-grid occupancy XOR.  The result is cropped to ``image_valid_mask``.
    No decoder output, learned value, dilation, or tunable radius participates.
    """

    if not isinstance(batch, PairBatch):
        raise TypeError("batch must be a PairBatch")
    batch.validate()
    feature_grid = tuple(int(value) for value in batch.feature.shape[-2:])
    evaluation_grid = tuple(
        int(value) for value in batch.occupancy_plus.shape[-2:]
    )
    projected_plus = project_occupancy_to_feature_grid(
        batch.occupancy_plus,
        feature_grid,
    )
    projected_minus = project_occupancy_to_feature_grid(
        batch.occupancy_minus,
        feature_grid,
    )
    changed_cells = projected_plus ^ projected_minus
    changed_by_pair = changed_cells.flatten(1).any(dim=1)
    if not torch.all(changed_by_pair):
        raise ValueError(
            "every outcome pair requires a directly visible projected change"
        )
    lifted = F.interpolate(
        changed_cells.to(dtype=torch.float32),
        size=evaluation_grid,
        mode="nearest",
    ).to(dtype=torch.bool)
    component = batch.occupancy_plus & ~batch.occupancy_minus
    component_by_pair = component.flatten(1).any(dim=1)
    if not torch.all(component_by_pair):
        raise ValueError("every outcome pair requires a non-empty deletion component")
    return ((component | lifted) & batch.image_valid_mask).contiguous()


@dataclass(frozen=True)
class OutcomePairBatch:
    """Device-ready complete-outcome batch for one OC-APTO optimizer branch."""

    pair_batch: PairBatch
    completion_plus: Tensor
    completion_minus: Tensor
    gt_union: Tensor
    intervention_footprint: Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def response_stratum(self) -> Tensor:
        """Return ``D = (R_minus \\ R_plus) intersect V``."""

        return (
            self.completion_minus
            & ~self.completion_plus
            & self.pair_batch.image_valid_mask
        )

    @property
    def local_zero_stratum(self) -> Tensor:
        """Return ``H = J \\ D`` inside the valid image domain."""

        return (
            self.intervention_footprint
            & ~self.response_stratum
            & self.pair_batch.image_valid_mask
        )

    @property
    def global_zero_stratum(self) -> Tensor:
        """Return ``G = V \\ (D union J)``."""

        return self.pair_batch.image_valid_mask & ~(
            self.response_stratum | self.intervention_footprint
        )

    @property
    def removed_component(self) -> Tensor:
        """Return the exact evaluation-grid component ``C``."""

        return (
            self.pair_batch.occupancy_plus
            & ~self.pair_batch.occupancy_minus
        )

    def validate(self) -> None:
        """Validate exact outcomes, footprint construction, and D/H/G partition."""

        if not isinstance(self.pair_batch, PairBatch):
            raise TypeError("pair_batch must be a PairBatch")
        self.pair_batch.validate()
        if any(
            kind not in OUTCOME_PAIR_KINDS
            for kind in self.pair_batch.pair_kinds
        ):
            raise ValueError(
                "OutcomePairBatch accepts only clean_positive/component_null pairs"
            )

        reference = self.pair_batch.image_valid_mask
        for name in (
            "completion_plus",
            "completion_minus",
            "gt_union",
            "intervention_footprint",
        ):
            _validate_bool_batch_tensor(
                getattr(self, name),
                name=name,
                reference=reference,
            )

        valid = reference
        occupancy_plus = self.pair_batch.occupancy_plus
        occupancy_minus = self.pair_batch.occupancy_minus
        r_plus = self.completion_plus
        r_minus = self.completion_minus
        gt_union = self.gt_union
        if torch.any(gt_union & ~valid):
            raise ValueError("gt_union extends outside image_valid_mask")
        if torch.any(r_plus & (~valid | occupancy_plus)):
            raise ValueError(
                "completion_plus must be valid and writable under occupancy_plus"
            )
        if torch.any(r_minus & (~valid | occupancy_minus)):
            raise ValueError(
                "completion_minus must be valid and writable under occupancy_minus"
            )
        if torch.any(r_plus & ~gt_union) or torch.any(r_minus & ~gt_union):
            raise ValueError("completion endpoints must be subsets of gt_union")

        for index, kind in enumerate(self.pair_batch.pair_kinds):
            if kind == "component_null" and not torch.equal(
                r_plus[index],
                r_minus[index],
            ):
                raise ValueError(
                    "component_null requires completion_plus equal completion_minus"
                )
        if torch.any(r_plus & ~r_minus):
            raise ValueError("completion_plus must be a subset of completion_minus")

        increment = r_minus & ~r_plus
        label_increment = self.pair_batch.label_increment.to(dtype=torch.bool)
        if not torch.equal(increment, label_increment):
            raise ValueError(
                "label_increment must equal completion_minus minus completion_plus"
            )
        response_present = increment.flatten(1).any(dim=1)
        for index, kind in enumerate(self.pair_batch.pair_kinds):
            if kind == "clean_positive" and not bool(response_present[index]):
                raise ValueError("clean_positive requires non-empty D")
            if kind == "component_null" and bool(response_present[index]):
                raise ValueError("component_null requires empty D")

        expected_footprint = direct_projected_intervention_footprint(
            self.pair_batch
        )
        if not torch.equal(self.intervention_footprint, expected_footprint):
            raise ValueError(
                "intervention_footprint must equal "
                "(C union nearest_lift(projected XOR)) intersect V"
            )

        response = self.response_stratum
        local_zero = self.local_zero_stratum
        global_zero = self.global_zero_stratum
        if (
            torch.any(response & local_zero)
            or torch.any(response & global_zero)
            or torch.any(local_zero & global_zero)
        ):
            raise AssertionError("D/H/G strata must be pairwise disjoint")
        if not torch.equal(response | local_zero | global_zero, valid):
            raise AssertionError("D/H/G strata must partition image_valid_mask")
        if not torch.all(global_zero.flatten(1).any(dim=1)):
            raise ValueError("every outcome pair requires non-empty global stratum G")
        for index, kind in enumerate(self.pair_batch.pair_kinds):
            if kind == "component_null" and not bool(
                local_zero[index].any()
            ):
                raise ValueError(
                    "component_null requires non-empty local zero stratum H"
                )


def _validate_source_gt_union(
    value: Tensor,
    *,
    sample_id: str,
    reference: Tensor,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"gt_union_by_sample[{sample_id!r}] must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.bool:
        raise TypeError(
            f"gt_union_by_sample[{sample_id!r}] must be a CPU bool tensor"
        )
    if value.shape != reference.shape:
        raise ValueError(
            f"gt_union_by_sample[{sample_id!r}] shape differs from its pair"
        )
    if torch.any(value & ~reference):
        raise ValueError(
            f"gt_union_by_sample[{sample_id!r}] extends outside image_valid_mask"
        )
    return value.detach().clone().contiguous()


def stack_outcome_pair_examples(
    examples: Iterable[PairExample],
    *,
    gt_union_by_sample: Mapping[str, Tensor],
    device: torch.device | str,
) -> OutcomePairBatch:
    """Stack mixed clean/component interventions with exact outcome truth."""

    values = tuple(examples)
    if not values:
        raise ValueError("cannot stack an empty outcome pair selection")
    if any(not isinstance(value, PairExample) for value in values):
        raise TypeError("examples must contain only PairExample values")
    if any(value.pair_kind not in OUTCOME_PAIR_KINDS for value in values):
        raise ValueError(
            "outcome optimizer accepts only clean_positive/component_null examples"
        )
    if not isinstance(gt_union_by_sample, Mapping):
        raise TypeError("gt_union_by_sample must be a mapping")

    pair_batch = stack_pair_examples(values, device=device)
    gt_values: list[Tensor] = []
    for value in values:
        if value.sample_id not in gt_union_by_sample:
            raise KeyError(
                f"gt_union_by_sample is missing source {value.sample_id!r}"
            )
        gt_values.append(
            _validate_source_gt_union(
                gt_union_by_sample[value.sample_id],
                sample_id=value.sample_id,
                reference=value.image_valid_mask,
            )
        )

    completion_plus = torch.stack(
        [value.completion_plus for value in values],
        dim=0,
    ).to(device=device)
    completion_minus = torch.stack(
        [value.completion_minus for value in values],
        dim=0,
    ).to(device=device)
    gt_union = torch.stack(gt_values, dim=0).to(device=device)
    footprint = direct_projected_intervention_footprint(pair_batch)
    return OutcomePairBatch(
        pair_batch=pair_batch,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        gt_union=gt_union,
        intervention_footprint=footprint,
    )


__all__ = [
    "OUTCOME_PAIR_KINDS",
    "OutcomePairBatch",
    "direct_projected_intervention_footprint",
    "stack_outcome_pair_examples",
]
