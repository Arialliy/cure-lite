from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_observability import (
    CoverageStateObservabilityDecision,
    audit_population_observability,
)
from cure_lite.coverage_state_precomputed_cache import (
    prepare_scalar_coverage_state_cache,
)
from cure_lite.coverage_state_raw_catalog import (
    CoverageStateNaturalRecord,
    CoverageStatePairRecord,
    make_coverage_state_raw_catalog,
)
from cure_lite.coverage_state_sobolev import CoverageStateSobolevConfig
from tests_v15.coverage_state_test_helpers import (
    TOY_OUTPUT_SIZE,
    TOY_STRIDE,
    make_feature,
    make_identity_pair,
    make_mask,
    make_natural_no_miss,
    make_scalar_hidden_pair,
)


def _visible_pair(
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
        {"visible_pair": pair_id, "endpoint": "before"}
    )
    after = stable_fingerprint(
        {"visible_pair": pair_id, "endpoint": "after"}
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
            {"visible_source": variant}
        ),
        evaluation_gt_id=variant if kind == "clean_positive" else None,
        native_gt_id=variant if kind == "clean_positive" else None,
        pred_id=variant,
        before_match_fingerprint=before,
        after_match_fingerprint=after,
        lineage_record_fingerprint=(
            stable_fingerprint({"visible_lineage": variant})
            if kind == "clean_positive"
            else None
        ),
    )


def _focused_naturals() -> tuple[CoverageStateNaturalRecord, ...]:
    scene = make_mask((1, 1), (4, 4))
    occupancy = make_mask((6, 6))
    valid = torch.ones_like(scene)
    common = {
        "sample_id": "sample-multi-miss",
        "group_id": "group-multi-miss",
        "state_kind": "factual_miss",
        "feature": make_feature(11),
        "occupancy": occupancy,
        "target": scene,
        "valid_mask": valid,
        "target_ids": ("evaluation_gt:1", "evaluation_gt:2"),
        "source_row_fingerprint": stable_fingerprint(
            {"source": "multi-miss"}
        ),
        "evaluation_gt_ids": (1, 2),
        "native_gt_ids": (1, 2),
        "lineage_record_fingerprint": stable_fingerprint(
            {"lineage": "multi-miss"}
        ),
    }
    loss_one = valid & ~occupancy & ~make_mask((4, 4))
    loss_two = valid & ~occupancy & ~make_mask((1, 1))
    return (
        CoverageStateNaturalRecord(
            record_id="natural-focus-001",
            loss_valid_mask=loss_one,
            focus_target_ids=("evaluation_gt:1",),
            **common,
        ),
        CoverageStateNaturalRecord(
            record_id="natural-focus-002",
            loss_valid_mask=loss_two,
            focus_target_ids=("evaluation_gt:2",),
            **common,
        ),
    )


def _scalar_catalog(*, include_hidden_component: bool = True):
    pairs = [
        _visible_pair(
            "clean_positive",
            variant=1,
            pair_id="pair-clean-visible",
        ),
        _visible_pair(
            "component_null",
            variant=2,
            pair_id="pair-component-visible",
        ),
        make_identity_pair(
            variant=3,
            pair_id="pair-identity",
        ),
    ]
    if include_hidden_component:
        pairs.append(
            make_scalar_hidden_pair(
                "component_null",
                variant=4,
                pair_id="pair-component-hidden",
            )
        )
    naturals = (
        *_focused_naturals(),
        make_natural_no_miss(variant=12),
    )
    source = stable_fingerprint(
        {
            "cache_test": "scalar",
            "pairs": sorted(value.pair_id for value in pairs),
            "naturals": sorted(value.record_id for value in naturals),
        }
    )
    return make_coverage_state_raw_catalog(
        dataset="toy",
        feature_stride=TOY_STRIDE,
        source_fingerprint=source,
        natural_records=tuple(naturals),
        pair_records=tuple(pairs),
    )


def _scalar_cache():
    catalog = _scalar_catalog()
    receipt = audit_population_observability(catalog)
    assert (
        receipt.decision
        is CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
    )
    return prepare_scalar_coverage_state_cache(
        catalog,
        receipt,
        CoverageStateSobolevConfig(truncation_radius=TOY_STRIDE),
    )


def test_scene_complete_field_is_shared_while_focus_measure_changes() -> None:
    cache = _scalar_cache()
    focused = tuple(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_miss"
    )
    assert len(focused) == 2
    first, second = focused
    assert first.targets.target_field is second.targets.target_field
    assert torch.equal(
        first.targets.target_field,
        second.targets.target_field,
    )
    assert not torch.equal(
        first.targets.integration_measure,
        second.targets.integration_measure,
    )
    assert torch.equal(
        first.targets.focus_support,
        make_mask((1, 1)),
    )
    assert torch.equal(
        second.targets.focus_support,
        make_mask((4, 4)),
    )
    assert bool(
        torch.all(first.targets.target_field[make_mask((4, 4))] < 0.0)
    )
    assert bool(
        torch.all(second.targets.target_field[make_mask((1, 1))] < 0.0)
    )
    assert not bool(
        torch.any(
            first.targets.integration_measure[
                ~first.record.loss_valid_mask
            ]
        )
    )


def test_scalar_cache_preserves_hidden_component_as_diagnostic_only() -> None:
    cache = _scalar_cache()
    assert len(cache.clean_positive_records) == 1
    assert len(cache.component_null_records) == 1
    hidden = tuple(
        value
        for value in cache.pair_records
        if value.record.pair_id == "pair-component-hidden"
    )
    assert len(hidden) == 1
    assert hidden[0].optimizer_role == "diagnostic_only"
    assert not hidden[0].optimization_eligible
    counts = cache.canonical_payload()["counts"]
    assert counts["component_null_total"] == 2
    assert counts["component_null_optimization_eligible"] == 1
    assert counts["component_null_diagnostic_only"] == 1


def test_separable_endpoint_domains_are_writable_and_pair_independent() -> None:
    cache = _scalar_cache()
    pair = cache.clean_positive_records[0]
    assert torch.equal(
        pair.absolute_targets_plus.loss_valid_mask,
        pair.record.valid_mask & ~pair.record.occupancy_plus,
    )
    assert torch.equal(
        pair.absolute_targets_minus.loss_valid_mask,
        pair.record.valid_mask & ~pair.record.occupancy_minus,
    )
    assert not torch.equal(
        pair.absolute_targets_plus.integration_measure,
        pair.joint_targets.integration_measure,
    )


def test_cache_is_deterministic_and_detects_tensor_mutation() -> None:
    first = _scalar_cache()
    second = _scalar_cache()
    assert first.canonical_payload() == second.canonical_payload()
    assert first.cache_fingerprint == second.cache_fingerprint
    first.verify_unchanged()
    first.raw_catalog.natural_records[0].feature[0, 0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="raw catalog changed"):
        first.verify_unchanged()


def test_scalar_cache_rejects_pp_observability_receipt() -> None:
    catalog = make_coverage_state_raw_catalog(
        dataset="toy",
        feature_stride=TOY_STRIDE,
        source_fingerprint=stable_fingerprint({"cache_test": "pp"}),
        natural_records=(make_natural_no_miss(),),
        pair_records=(
            make_scalar_hidden_pair(
                "clean_positive",
                variant=1,
                pair_id="pair-clean-hidden",
            ),
            make_scalar_hidden_pair(
                "component_null",
                variant=2,
                pair_id="pair-component-hidden",
            ),
            make_identity_pair(),
        ),
    )
    receipt = audit_population_observability(catalog)
    assert (
        receipt.decision
        is CoverageStateObservabilityDecision.AUTHORIZE_PP_CSLF
    )
    with pytest.raises(PermissionError, match="AUTHORIZE_SCALAR_CSLF"):
        prepare_scalar_coverage_state_cache(
            catalog,
            receipt,
            CoverageStateSobolevConfig(truncation_radius=TOY_STRIDE),
        )


def test_scalar_cache_rejects_wrong_radius_without_mutation() -> None:
    catalog = _scalar_catalog(include_hidden_component=False)
    receipt = audit_population_observability(catalog)
    bad = replace(
        CoverageStateSobolevConfig(truncation_radius=TOY_STRIDE),
        truncation_radius=TOY_STRIDE * 2,
    )
    with pytest.raises(ValueError, match="authorization conditions"):
        prepare_scalar_coverage_state_cache(catalog, receipt, bad)


def test_toy_grid_contract_is_not_silently_changed() -> None:
    assert TOY_OUTPUT_SIZE == 8
