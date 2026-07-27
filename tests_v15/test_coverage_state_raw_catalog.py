from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.coverage_state_observability import (
    occupancy_to_phase_grid,
    occupancy_to_scalar_grid,
)
from cure_lite.coverage_state_raw_catalog import (
    CoverageStateNaturalRecord,
    CoverageStatePairRecord,
    make_coverage_state_raw_catalog,
)
from tests_v15.coverage_state_test_helpers import (
    TOY_FEATURE_SIZE,
    TOY_STRIDE,
    make_feature,
    make_identity_pair,
    make_mask,
    make_natural_no_miss,
    make_scalar_hidden_pair,
    make_toy_raw_catalog,
)


@pytest.mark.parametrize(
    ("kind", "variant"),
    (("clean_positive", 1), ("component_null", 2)),
)
def test_raw_catalog_retains_scalar_hidden_phase_visible_pair(
    kind: str,
    variant: int,
) -> None:
    pair = make_scalar_hidden_pair(kind, variant=variant)
    catalog = make_toy_raw_catalog(pairs=(pair,))
    assert catalog.pair_records == (pair,)
    scalar_plus = occupancy_to_scalar_grid(
        pair.occupancy_plus,
        feature_size=(TOY_FEATURE_SIZE, TOY_FEATURE_SIZE),
    )
    scalar_minus = occupancy_to_scalar_grid(
        pair.occupancy_minus,
        feature_size=(TOY_FEATURE_SIZE, TOY_FEATURE_SIZE),
    )
    phase_plus = occupancy_to_phase_grid(
        pair.occupancy_plus,
        stride=TOY_STRIDE,
    )
    phase_minus = occupancy_to_phase_grid(
        pair.occupancy_minus,
        stride=TOY_STRIDE,
    )
    assert torch.equal(scalar_plus, scalar_minus)
    assert not torch.equal(phase_plus, phase_minus)


def test_raw_catalog_enforces_clean_and_component_target_semantics() -> None:
    clean = make_scalar_hidden_pair("clean_positive", variant=1)
    component = make_scalar_hidden_pair("component_null", variant=2)
    empty = make_mask()
    with pytest.raises(ValueError, match="clean_positive"):
        replace(clean, target_minus=empty, target_ids_added=())
    with pytest.raises(ValueError, match="unchanged target"):
        replace(
            component,
            target_minus=make_mask((2, 3)),
        )
    with pytest.raises(ValueError, match="subset"):
        replace(
            clean,
            occupancy_minus=clean.occupancy_minus | make_mask((7, 7)),
            removed_component=clean.removed_component,
        )
    with pytest.raises(ValueError, match="removed_component"):
        replace(clean, removed_component=empty)


def test_identity_null_requires_exact_endpoint_identity() -> None:
    identity = make_identity_pair()
    with pytest.raises(ValueError, match="identity_null"):
        replace(
            identity,
            occupancy_minus=make_mask(),
            removed_component=identity.occupancy_plus,
        )
    with pytest.raises(ValueError, match="identity_null"):
        replace(
            identity,
            target_minus=make_mask((0, 0)),
        )
    with pytest.raises(ValueError, match="identity_null"):
        replace(identity, removed_component_ids=("pred-1",))


def test_raw_tensor_contract_and_grid_integrity_fail_closed() -> None:
    valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    common = {
        "record_id": "natural",
        "sample_id": "sample",
        "group_id": "group",
        "state_kind": "factual_no_miss",
        "feature": make_feature(),
        "occupancy": make_mask(),
        "target": make_mask(),
        "valid_mask": valid,
        "loss_valid_mask": valid,
        "target_ids": (),
        "focus_target_ids": (),
        "source_row_fingerprint": stable_fingerprint({"source": "natural"}),
        "evaluation_gt_ids": (),
        "native_gt_ids": (),
        "lineage_record_fingerprint": None,
    }
    with pytest.raises(TypeError, match="float32"):
        CoverageStateNaturalRecord(
            **{**common, "feature": make_feature().to(torch.float64)}
        )
    nonfinite = make_feature()
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        CoverageStateNaturalRecord(
            **{**common, "feature": nonfinite}
        )
    with pytest.raises(TypeError, match="CPU bool"):
        CoverageStateNaturalRecord(
            **{**common, "occupancy": make_mask().to(torch.float32)}
        )
    natural = CoverageStateNaturalRecord(**common)
    pair = make_identity_pair()
    with pytest.raises(ValueError, match="feature grid times"):
        make_coverage_state_raw_catalog(
            dataset="toy",
            feature_stride=4,
            source_fingerprint="a" * 64,
            natural_records=(natural,),
            pair_records=(pair,),
        )


def test_target_and_occupancy_must_remain_in_valid_domain() -> None:
    valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    valid[..., 2, 3] = False
    with pytest.raises(ValueError, match="inside valid_mask"):
        CoverageStatePairRecord(
            pair_id="pair",
            sample_id="sample",
            group_id="group",
            pair_kind="clean_positive",
            feature=make_feature(),
            occupancy_plus=make_mask((2, 2), (2, 3)),
            occupancy_minus=make_mask((2, 2)),
            target_plus=make_mask(),
            target_minus=make_mask((2, 3)),
            valid_mask=valid,
            removed_component=make_mask((2, 3)),
            removed_component_ids=("pred-1",),
            target_ids_added=("gt-1",),
            source_row_fingerprint=stable_fingerprint({"source": "pair"}),
            evaluation_gt_id=1,
            native_gt_id=1,
            pred_id=1,
            before_match_fingerprint=stable_fingerprint({"match": "before"}),
            after_match_fingerprint=stable_fingerprint({"match": "after"}),
            lineage_record_fingerprint=stable_fingerprint(
                {"lineage": "pair"}
            ),
        )


def test_raw_catalog_is_order_invariant_and_byte_deterministic() -> None:
    natural_a = make_natural_no_miss(variant=8)
    natural_b = make_natural_no_miss(variant=9)
    pair_a = make_identity_pair(variant=3, pair_id="pair-a")
    pair_b = make_scalar_hidden_pair(
        "clean_positive",
        variant=1,
        pair_id="pair-b",
    )
    kwargs = {
        "dataset": "toy",
        "feature_stride": TOY_STRIDE,
        "source_fingerprint": "f" * 64,
    }
    first = make_coverage_state_raw_catalog(
        **kwargs,
        natural_records=(natural_a, natural_b),
        pair_records=(pair_a, pair_b),
    )
    second = make_coverage_state_raw_catalog(
        **kwargs,
        natural_records=(natural_b, natural_a),
        pair_records=(pair_b, pair_a),
    )
    assert first.canonical_payload() == second.canonical_payload()
    assert first.catalog_fingerprint == second.catalog_fingerprint
    assert canonical_json(first.canonical_payload()).encode() == canonical_json(
        second.canonical_payload()
    ).encode()
