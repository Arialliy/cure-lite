"""Toy scalar-authorized cache and batches for CSLF training tests."""

from __future__ import annotations

import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_observability import (
    CoverageStateObservabilityDecision,
    audit_population_observability,
)
from cure_lite.coverage_state_batches import (
    CoverageStateFusedBatch,
    make_coverage_state_natural_train_batch,
    make_coverage_state_pair_train_batch,
)
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
    prepare_scalar_coverage_state_cache,
)
from cure_lite.coverage_state_raw_catalog import (
    CoverageStateNaturalRecord,
    CoverageStatePairRecord,
    make_coverage_state_raw_catalog,
)
from cure_lite.coverage_state_sobolev import CoverageStateSobolevConfig
from tests_v15.coverage_state_test_helpers import (
    TOY_STRIDE,
    make_feature,
    make_identity_pair,
    make_mask,
    make_natural_no_miss,
    make_scalar_hidden_pair,
)


def make_visible_pair(
    kind: str,
    *,
    variant: int,
    pair_id: str,
) -> CoverageStatePairRecord:
    plus = make_mask((2, 2), (4, 4))
    minus = make_mask((2, 2))
    removed = make_mask((4, 4))
    empty = make_mask()
    target_minus = removed if kind == "clean_positive" else empty
    before = stable_fingerprint(
        {"training_pair": pair_id, "endpoint": "before"}
    )
    after = stable_fingerprint(
        {"training_pair": pair_id, "endpoint": "after"}
    )
    return CoverageStatePairRecord(
        pair_id=pair_id,
        sample_id=f"sample-{pair_id}",
        group_id=f"group-{pair_id}",
        pair_kind=kind,
        feature=make_feature(variant),
        occupancy_plus=plus,
        occupancy_minus=minus,
        target_plus=empty,
        target_minus=target_minus,
        valid_mask=torch.ones_like(empty),
        removed_component=removed,
        removed_component_ids=(f"pred-{variant}",),
        target_ids_added=(
            (f"gt-{variant}",)
            if kind == "clean_positive"
            else ()
        ),
        source_row_fingerprint=stable_fingerprint(
            {"training_source": variant}
        ),
        evaluation_gt_id=variant if kind == "clean_positive" else None,
        native_gt_id=variant if kind == "clean_positive" else None,
        pred_id=variant,
        before_match_fingerprint=before,
        after_match_fingerprint=after,
        lineage_record_fingerprint=(
            stable_fingerprint({"training_lineage": variant})
            if kind == "clean_positive"
            else None
        ),
    )


def make_factual_miss(*, variant: int) -> CoverageStateNaturalRecord:
    target = make_mask((1 + variant % 3, 4 + variant % 2))
    occupancy = make_mask((7, 7))
    valid = torch.ones_like(target)
    evaluation_id = variant
    return CoverageStateNaturalRecord(
        record_id=f"natural-miss-{variant:03d}",
        sample_id=f"sample-miss-{variant:03d}",
        group_id=f"group-miss-{variant:03d}",
        state_kind="factual_miss",
        feature=make_feature(variant),
        occupancy=occupancy,
        target=target,
        valid_mask=valid,
        loss_valid_mask=valid & ~occupancy,
        target_ids=(f"evaluation_gt:{evaluation_id}",),
        focus_target_ids=(f"evaluation_gt:{evaluation_id}",),
        source_row_fingerprint=stable_fingerprint(
            {"training_natural_source": variant}
        ),
        evaluation_gt_ids=(evaluation_id,),
        native_gt_ids=(evaluation_id,),
        lineage_record_fingerprint=stable_fingerprint(
            {"training_natural_lineage": variant}
        ),
    )


def make_training_scalar_cache() -> CoverageStateScalarCache:
    naturals = (
        *(make_factual_miss(variant=value) for value in range(10, 14)),
        *(make_natural_no_miss(variant=value) for value in range(20, 24)),
    )
    pairs = (
        make_visible_pair(
            "clean_positive",
            variant=30,
            pair_id="pair-clean-training",
        ),
        make_visible_pair(
            "component_null",
            variant=31,
            pair_id="pair-component-training",
        ),
        make_scalar_hidden_pair(
            "component_null",
            variant=32,
            pair_id="pair-component-diagnostic",
        ),
        make_identity_pair(
            variant=33,
            pair_id="pair-identity-diagnostic",
        ),
    )
    catalog = make_coverage_state_raw_catalog(
        dataset="toy",
        feature_stride=TOY_STRIDE,
        source_fingerprint=stable_fingerprint(
            {
                "training_cache": "v1",
                "naturals": sorted(value.record_id for value in naturals),
                "pairs": sorted(value.pair_id for value in pairs),
            }
        ),
        natural_records=tuple(naturals),
        pair_records=pairs,
    )
    receipt = audit_population_observability(catalog)
    if (
        receipt.decision
        is not CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
    ):
        raise AssertionError("training toy catalog must authorize scalar CSLF")
    return prepare_scalar_coverage_state_cache(
        catalog,
        receipt,
        CoverageStateSobolevConfig(truncation_radius=TOY_STRIDE),
    )


def make_bounded_training_scalar_cache() -> CoverageStateScalarCache:
    """Return a toy population large enough for the fixed 16-role protocol."""

    naturals = (
        *(make_factual_miss(variant=value) for value in range(100, 116)),
        *(make_natural_no_miss(variant=value) for value in range(200, 216)),
    )
    pairs = (
        *(
            make_visible_pair(
                "clean_positive",
                variant=value,
                pair_id=f"pair-clean-bounded-{value:03d}",
            )
            for value in range(300, 316)
        ),
        *(
            make_visible_pair(
                "component_null",
                variant=value,
                pair_id=f"pair-component-bounded-{value:03d}",
            )
            for value in range(400, 416)
        ),
        *(
            make_identity_pair(
                variant=value,
                pair_id=f"pair-identity-bounded-{value:03d}",
            )
            for value in range(500, 516)
        ),
        make_scalar_hidden_pair(
            "component_null",
            variant=600,
            pair_id="pair-component-bounded-diagnostic",
        ),
    )
    catalog = make_coverage_state_raw_catalog(
        dataset="toy-bounded",
        feature_stride=TOY_STRIDE,
        source_fingerprint=stable_fingerprint(
            {
                "training_cache": "bounded-v1",
                "naturals": sorted(value.record_id for value in naturals),
                "pairs": sorted(value.pair_id for value in pairs),
            }
        ),
        natural_records=tuple(naturals),
        pair_records=tuple(pairs),
    )
    receipt = audit_population_observability(catalog)
    if (
        receipt.decision
        is not CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
    ):
        raise AssertionError("bounded toy catalog must authorize scalar CSLF")
    return prepare_scalar_coverage_state_cache(
        catalog,
        receipt,
        CoverageStateSobolevConfig(truncation_radius=TOY_STRIDE),
    )


def make_training_fused_batch(
    *,
    device: torch.device | str = "cpu",
) -> CoverageStateFusedBatch:
    cache = make_training_scalar_cache()
    misses = tuple(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_miss"
    )
    no_misses = tuple(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_no_miss"
    )
    result = CoverageStateFusedBatch(
        factual_miss=make_coverage_state_natural_train_batch(
            misses,
            state_kind="factual_miss",
            device=device,
        ),
        factual_no_miss=make_coverage_state_natural_train_batch(
            no_misses,
            state_kind="factual_no_miss",
            device=device,
        ),
        pairs=make_coverage_state_pair_train_batch(
            cache.clean_positive_records[0],
            cache.component_null_records[0],
            device=device,
        ),
    )
    result.validate()
    return result
