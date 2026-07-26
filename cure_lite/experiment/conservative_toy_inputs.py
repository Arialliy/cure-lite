"""Dataset-free toy inputs for the CURE-Lite CC-SEA v8 gate."""

from __future__ import annotations

import hashlib

import torch

from ..paired_outcome_types import (
    OutcomePairBatch,
    direct_projected_intervention_footprint,
)
from ..paired_types import PairBatch
from ..train.step import BranchBatch


LEGACY_FAMILY = "component_contains_response"
SUPPORT_FAMILY = "response_outside_component_inside_count_support"
CONSERVATIVE_TOY_CASES = (
    (LEGACY_FAMILY, "legacy_one_pixel", ((1, 2),)),
    (LEGACY_FAMILY, "legacy_two_pixels", ((1, 2), (2, 1))),
    (
        LEGACY_FAMILY,
        "legacy_three_pixels",
        ((1, 2), (2, 1), (2, 2)),
    ),
    (SUPPORT_FAMILY, "support_one_pixel", ((1, 6),)),
    (SUPPORT_FAMILY, "support_two_pixels", ((1, 6), (2, 5))),
    (
        SUPPORT_FAMILY,
        "support_three_pixels",
        ((1, 6), (2, 5), (2, 6)),
    ),
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _base_toy(
    clean_pixels: tuple[tuple[int, int], ...],
) -> tuple[OutcomePairBatch, dict[str, BranchBatch]]:
    """Return one clean and one component-null stride-four problem."""

    feature = torch.zeros(2, 8, 2, 2)
    feature[0, 0, 0, 0] = 5.0
    feature[0, 1, 1, 0] = 4.0
    feature[0, 6] = 0.5
    feature[1, 2, 1, 1] = 5.0
    feature[1, 3, 0, 1] = 4.0
    feature[1, 7] = 0.5

    occupancy_plus = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    occupancy_plus[0, 0, 0:4, 0:4] = True
    occupancy_plus[1, 0, 4:8, 4:8] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)

    completion_plus = torch.zeros_like(occupancy_plus)
    completion_plus[0, 0, 5, 1] = True
    completion_plus[1, 0, 1, 6] = True
    completion_minus = completion_plus.clone()
    for row, column in clean_pixels:
        completion_minus[0, 0, row, column] = True
    increment = (completion_minus & ~completion_plus).to(torch.float32)
    valid = torch.ones_like(occupancy_plus)

    pair_batch = PairBatch(
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        label_increment=increment,
        image_valid_mask=valid,
        pair_ids=(_sha("cc-sea-clean"), _sha("cc-sea-component")),
        sample_ids=("cc-sea-clean-source", "cc-sea-component-source"),
        group_ids=("cc-sea-clean-group", "cc-sea-component-group"),
        pair_kinds=("clean_positive", "component_null"),
        projection_visible=(True, True),
    )
    outcome = OutcomePairBatch(
        pair_batch=pair_batch,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        gt_union=completion_minus.clone(),
        intervention_footprint=direct_projected_intervention_footprint(
            pair_batch
        ),
    )

    factual_occupancy = torch.zeros(4, 1, 8, 8, dtype=torch.bool)
    factual_valid = torch.ones_like(factual_occupancy)
    no_miss_feature = torch.zeros(4, 8, 2, 2)
    no_miss_feature[:, 4, 0, 1] = 3.0
    no_miss_feature[:, 5] = -0.5
    factual = {
        "factual_miss": BranchBatch(
            feature=feature[0:1].repeat(4, 1, 1, 1),
            occupancy=factual_occupancy,
            target=(
                completion_minus[0:1]
                .to(torch.float32)
                .repeat(4, 1, 1, 1)
            ),
            valid_mask=factual_valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=no_miss_feature,
            occupancy=factual_occupancy.clone(),
            target=torch.zeros(4, 1, 8, 8),
            valid_mask=factual_valid.clone(),
        ),
    }
    return outcome, factual


def build_conservative_toy_case(
    family_id: str,
    clean_pixels: tuple[tuple[int, int], ...],
) -> tuple[OutcomePairBatch, dict[str, BranchBatch]]:
    """Build one frozen case without reading a dataset or cache."""

    if family_id not in {LEGACY_FAMILY, SUPPORT_FAMILY}:
        raise ValueError("unknown CC-SEA toy family")
    outcome, factual = _base_toy(clean_pixels)
    if family_id == LEGACY_FAMILY:
        return outcome, factual

    feature = outcome.pair_batch.feature.clone()
    feature[0, 0, 0, 0] = 0.0
    feature[0, 0, 0, 1] = 5.0
    occupancy_plus = torch.zeros_like(outcome.pair_batch.occupancy_plus)
    occupancy_plus[0, 0, 0, 0] = True
    occupancy_plus[1, 0, 7, 7] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)
    pair_batch = PairBatch(
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        label_increment=outcome.pair_batch.label_increment.clone(),
        image_valid_mask=outcome.pair_batch.image_valid_mask.clone(),
        pair_ids=(
            _sha("cc-sea-support-clean"),
            _sha("cc-sea-support-component"),
        ),
        sample_ids=(
            "cc-sea-support-clean-source",
            "cc-sea-support-component-source",
        ),
        group_ids=(
            "cc-sea-support-clean-group",
            "cc-sea-support-component-group",
        ),
        pair_kinds=("clean_positive", "component_null"),
        projection_visible=(True, True),
    )
    adjusted_outcome = OutcomePairBatch(
        pair_batch=pair_batch,
        completion_plus=outcome.completion_plus.clone(),
        completion_minus=outcome.completion_minus.clone(),
        gt_union=outcome.gt_union.clone(),
        intervention_footprint=direct_projected_intervention_footprint(
            pair_batch
        ),
    )
    miss = factual["factual_miss"]
    adjusted_factual = dict(factual)
    adjusted_factual["factual_miss"] = BranchBatch(
        feature=feature[0:1].repeat(4, 1, 1, 1),
        occupancy=miss.occupancy.clone(),
        target=miss.target.clone(),
        valid_mask=miss.valid_mask.clone(),
    )
    return adjusted_outcome, adjusted_factual


__all__ = [
    "CONSERVATIVE_TOY_CASES",
    "LEGACY_FAMILY",
    "SUPPORT_FAMILY",
    "build_conservative_toy_case",
]
