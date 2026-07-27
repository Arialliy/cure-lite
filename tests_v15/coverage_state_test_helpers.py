"""Pure toy inputs for coverage-state contract tests."""

from __future__ import annotations

import torch
from torch import Tensor

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_raw_catalog import (
    CoverageStateNaturalRecord,
    CoverageStatePairRecord,
    CoverageStateRawCatalog,
    make_coverage_state_raw_catalog,
)


TOY_STRIDE = 2
TOY_FEATURE_SIZE = 4
TOY_OUTPUT_SIZE = TOY_FEATURE_SIZE * TOY_STRIDE


def make_feature(variant: int = 0) -> Tensor:
    """Return a finite feature whose normalized direction varies by variant."""

    base = torch.arange(
        1,
        1 + 2 * TOY_FEATURE_SIZE * TOY_FEATURE_SIZE,
        dtype=torch.float32,
    ).reshape(1, 2, TOY_FEATURE_SIZE, TOY_FEATURE_SIZE)
    if variant:
        base = base.clone()
        row = variant % TOY_FEATURE_SIZE
        column = (2 * variant + 1) % TOY_FEATURE_SIZE
        base[0, variant % 2, row, column] += float(variant + 3)
    return base.contiguous()


def make_mask(*pixels: tuple[int, int]) -> Tensor:
    value = torch.zeros(
        1,
        1,
        TOY_OUTPUT_SIZE,
        TOY_OUTPUT_SIZE,
        dtype=torch.bool,
    )
    for row, column in pixels:
        value[0, 0, row, column] = True
    return value


def make_natural_no_miss(*, variant: int = 9) -> CoverageStateNaturalRecord:
    return CoverageStateNaturalRecord(
        record_id=f"natural-no-miss-{variant:03d}",
        sample_id=f"sample-natural-{variant:03d}",
        group_id=f"group-natural-{variant:03d}",
        state_kind="factual_no_miss",
        feature=make_feature(variant),
        occupancy=make_mask((6, 6)),
        target=make_mask(),
        valid_mask=torch.ones(
            1,
            1,
            TOY_OUTPUT_SIZE,
            TOY_OUTPUT_SIZE,
            dtype=torch.bool,
        ),
        loss_valid_mask=(
            torch.ones(
                1,
                1,
                TOY_OUTPUT_SIZE,
                TOY_OUTPUT_SIZE,
                dtype=torch.bool,
            )
            & ~make_mask((6, 6))
        ),
        target_ids=(),
        focus_target_ids=(),
        source_row_fingerprint=stable_fingerprint(
            {"toy_source_row": variant}
        ),
        evaluation_gt_ids=(),
        native_gt_ids=(),
        lineage_record_fingerprint=None,
    )


def make_scalar_hidden_pair(
    kind: str,
    *,
    variant: int,
    pair_id: str | None = None,
) -> CoverageStatePairRecord:
    """Delete one of two occupied phases inside a single feature cell."""

    occupancy_plus = make_mask((2, 2), (2, 3))
    occupancy_minus = make_mask((2, 2))
    removed = make_mask((2, 3))
    empty = make_mask()
    target_minus = removed.clone() if kind == "clean_positive" else empty.clone()
    before_match = stable_fingerprint(
        {"toy_match": "before", "variant": variant}
    )
    after_match = stable_fingerprint(
        {"toy_match": "after", "variant": variant}
    )
    return CoverageStatePairRecord(
        pair_id=pair_id or f"pair-{kind}-{variant:03d}",
        sample_id=f"sample-pair-{variant:03d}",
        group_id=f"group-pair-{variant:03d}",
        pair_kind=kind,
        feature=make_feature(variant),
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        target_plus=empty,
        target_minus=target_minus,
        valid_mask=torch.ones_like(empty),
        removed_component=removed,
        removed_component_ids=(f"pred-{variant:03d}",),
        target_ids_added=(
            (f"gt-{variant:03d}",)
            if kind == "clean_positive"
            else ()
        ),
        source_row_fingerprint=stable_fingerprint(
            {"toy_source_row": variant}
        ),
        evaluation_gt_id=variant if kind == "clean_positive" else None,
        native_gt_id=variant if kind == "clean_positive" else None,
        pred_id=variant,
        before_match_fingerprint=before_match,
        after_match_fingerprint=after_match,
        lineage_record_fingerprint=(
            stable_fingerprint(
                {"toy_lineage": variant, "kind": kind}
            )
            if kind == "clean_positive"
            else None
        ),
    )


def make_identity_pair(
    *,
    variant: int = 3,
    pair_id: str | None = None,
) -> CoverageStatePairRecord:
    occupancy = make_mask((5, 5))
    empty = make_mask()
    match = stable_fingerprint(
        {"toy_match": "identity", "variant": variant}
    )
    return CoverageStatePairRecord(
        pair_id=pair_id or f"pair-identity-{variant:03d}",
        sample_id=f"sample-identity-{variant:03d}",
        group_id=f"group-identity-{variant:03d}",
        pair_kind="identity_null",
        feature=make_feature(variant),
        occupancy_plus=occupancy,
        occupancy_minus=occupancy,
        target_plus=empty,
        target_minus=empty,
        valid_mask=torch.ones_like(empty),
        removed_component=empty,
        removed_component_ids=(),
        target_ids_added=(),
        source_row_fingerprint=stable_fingerprint(
            {"toy_source_row": variant}
        ),
        evaluation_gt_id=None,
        native_gt_id=None,
        pred_id=None,
        before_match_fingerprint=match,
        after_match_fingerprint=match,
        lineage_record_fingerprint=None,
    )


def make_toy_raw_catalog(
    *,
    pairs: tuple[CoverageStatePairRecord, ...] | None = None,
    naturals: tuple[CoverageStateNaturalRecord, ...] | None = None,
) -> CoverageStateRawCatalog:
    pair_values = pairs or (
        make_scalar_hidden_pair(
            "clean_positive",
            variant=1,
            pair_id="pair-clean",
        ),
        make_scalar_hidden_pair(
            "component_null",
            variant=2,
            pair_id="pair-component",
        ),
        make_identity_pair(variant=3, pair_id="pair-identity"),
    )
    natural_values = naturals or (make_natural_no_miss(),)
    source_fingerprint = stable_fingerprint(
        {
            "schema_version": "cure-lite-toy-raw-source-v1",
            "natural_ids": sorted(value.record_id for value in natural_values),
            "pair_ids": sorted(value.pair_id for value in pair_values),
        }
    )
    return make_coverage_state_raw_catalog(
        dataset="toy",
        feature_stride=TOY_STRIDE,
        source_fingerprint=source_fingerprint,
        natural_records=natural_values,
        pair_records=pair_values,
    )
