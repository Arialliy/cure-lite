from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
    normalize_cslf_feature,
)
from cure_lite.coverage_state_observability import (
    CoverageStateObservabilityDecision,
    audit_population_observability,
    changed_feature_cells,
    decide_observability,
    occupancy_to_phase_grid,
    occupancy_to_scalar_grid,
    structural_output_support,
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


def _clean_binding(tag: str) -> dict[str, object]:
    return {
        "source_row_fingerprint": stable_fingerprint(
            {"source_row": tag}
        ),
        "evaluation_gt_id": 1,
        "native_gt_id": 1,
        "pred_id": 1,
        "before_match_fingerprint": stable_fingerprint(
            {"match": tag, "endpoint": "before"}
        ),
        "after_match_fingerprint": stable_fingerprint(
            {"match": tag, "endpoint": "after"}
        ),
        "lineage_record_fingerprint": stable_fingerprint(
            {"lineage": tag}
        ),
    }


def _natural_binding(
    tag: str,
    *,
    miss: bool,
) -> dict[str, object]:
    return {
        "source_row_fingerprint": stable_fingerprint(
            {"source_row": tag}
        ),
        "evaluation_gt_ids": (1,) if miss else (),
        "native_gt_ids": (1,) if miss else (),
        "lineage_record_fingerprint": (
            stable_fingerprint({"lineage": tag})
            if miss
            else None
        ),
    }


def test_normalization_is_the_exact_model_input_and_batch_local() -> None:
    model = CURELiteCoverageStateLevelSet(
        CoverageStateLevelSetConfig(
            feature_channels=2,
            feature_stride=TOY_STRIDE,
            width=4,
        )
    )
    first = make_feature(1)
    second = make_feature(2)
    batch = torch.cat((first, second), dim=0).requires_grad_()
    occupancy = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    actual = normalize_cslf_feature(batch)
    fields = model.forward_fields(batch, occupancy)
    assert torch.equal(actual, fields.encoded_feature)
    assert torch.equal(actual[0:1], normalize_cslf_feature(first))
    assert torch.equal(actual[1:2], normalize_cslf_feature(second))
    assert not actual.requires_grad
    zeros = torch.zeros(1, 2, 4, 4, dtype=torch.float32)
    assert torch.count_nonzero(normalize_cslf_feature(zeros)) == 0
    with pytest.raises(TypeError, match="float32"):
        normalize_cslf_feature(first.to(torch.float64))
    nonfinite = first.clone()
    nonfinite[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        normalize_cslf_feature(nonfinite)


@pytest.mark.parametrize("stride", (2, 4))
def test_phase_grid_roundtrip_preserves_subcell_identity(
    stride: int,
) -> None:
    occupancy = torch.zeros(1, 1, 3 * stride, 4 * stride, dtype=torch.bool)
    occupancy[..., stride - 1, 2 * stride - 1] = True
    occupancy[..., 2 * stride, 3 * stride] = True
    phase = occupancy_to_phase_grid(occupancy, stride=stride)
    assert phase.dtype == torch.bool
    assert phase.is_contiguous()
    assert phase.shape == (1, stride**2, 3, 4)
    assert torch.equal(
        F.pixel_shuffle(phase.to(torch.float32), stride).to(torch.bool),
        occupancy,
    )


def test_phase_grid_and_representation_validation_fail_closed() -> None:
    occupancy = make_mask((1, 1))
    with pytest.raises(TypeError, match="bool"):
        occupancy_to_phase_grid(
            occupancy.to(torch.float32),
            stride=TOY_STRIDE,
        )
    with pytest.raises(ValueError, match="divisible"):
        occupancy_to_phase_grid(
            torch.zeros(1, 1, 7, 8, dtype=torch.bool),
            stride=TOY_STRIDE,
        )
    with pytest.raises(ValueError, match="positive"):
        occupancy_to_phase_grid(occupancy, stride=0)
    with pytest.raises(TypeError, match="aligned bool"):
        changed_feature_cells(
            torch.zeros(1, 1, 4, 4, dtype=torch.bool),
            torch.zeros(1, 2, 4, 4, dtype=torch.bool),
        )


def test_structural_rf_is_exact_radius_two_and_all_output_phases() -> None:
    center = torch.zeros(1, 1, 7, 7, dtype=torch.bool)
    center[..., 3, 3] = True
    support = structural_output_support(center, stride=2)
    assert support.shape == (1, 1, 14, 14)
    assert torch.count_nonzero(support) == 25 * 4
    boundary = torch.zeros_like(center)
    boundary[..., 0, 0] = True
    boundary_support = structural_output_support(boundary, stride=2)
    assert torch.count_nonzero(boundary_support) == 9 * 4
    assert torch.count_nonzero(
        structural_output_support(torch.zeros_like(center), stride=2)
    ) == 0


def test_scalar_hidden_response_selects_phase_preserving_model() -> None:
    catalog = make_toy_raw_catalog()
    receipt = audit_population_observability(catalog)
    assert receipt.unique_encoded_feature_tensors <= (
        receipt.natural_record_count + receipt.pair_record_count
    )
    assert receipt.unique_occupancy_states < (
        receipt.natural_record_count + 2 * receipt.pair_record_count
    )
    assert receipt.unique_target_fields < (
        receipt.natural_record_count + 2 * receipt.pair_record_count
    )
    assert receipt.full_grid_changed_pairs == 2
    assert receipt.phase_changed_pairs == 2
    assert receipt.scalar_projected_changed_pairs == 0
    assert receipt.hidden_by_scalar_projection_pairs == 2
    assert receipt.clean_positive_hidden_pairs == 1
    assert receipt.component_null_hidden_pairs == 1
    assert receipt.target_response_pixels > 0
    assert (
        receipt.target_response_outside_scalar_rf_pixels
        == receipt.target_response_pixels
    )
    assert receipt.target_response_outside_phase_rf_pixels == 0
    assert (
        receipt.target_response_hidden_only_by_scalar_pixels
        == receipt.target_response_pixels
    )
    assert receipt.identity_null_nonidentical_count == 0
    assert receipt.phase_duplicate_input_target_conflicts == 0
    assert receipt.decision is (
        CoverageStateObservabilityDecision.AUTHORIZE_PP_CSLF
    )
    assert not receipt.scalar_authorized
    assert receipt.pp_authorized


def test_zero_response_scalar_hidden_component_does_not_alone_select_pp() -> None:
    visible_clean = CoverageStatePairRecord(
        pair_id="pair-clean-visible",
        sample_id="sample-clean-visible",
        group_id="group-clean-visible",
        pair_kind="clean_positive",
        feature=make_feature(5),
        occupancy_plus=make_mask((0, 0)),
        occupancy_minus=make_mask(),
        target_plus=make_mask(),
        target_minus=make_mask((0, 0)),
        valid_mask=torch.ones_like(make_mask()),
        removed_component=make_mask((0, 0)),
        removed_component_ids=("pred-visible",),
        target_ids_added=("gt-visible",),
        **_clean_binding("clean-visible"),
    )
    hidden_component = make_scalar_hidden_pair(
        "component_null",
        variant=2,
        pair_id="pair-component-hidden",
    )
    catalog = make_toy_raw_catalog(
        pairs=(
            visible_clean,
            hidden_component,
            make_identity_pair(),
        )
    )
    receipt = audit_population_observability(catalog)
    assert receipt.component_null_hidden_pairs == 1
    assert receipt.target_response_outside_scalar_rf_pixels == 0
    assert receipt.scalar_duplicate_input_target_conflicts == 0
    assert receipt.decision is (
        CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
    )


def test_target_response_outside_phase_rf_blocks_both_models() -> None:
    far_response = CoverageStatePairRecord(
        pair_id="pair-far-response",
        sample_id="sample-far-response",
        group_id="group-far-response",
        pair_kind="clean_positive",
        feature=make_feature(6),
        occupancy_plus=make_mask((0, 0)),
        occupancy_minus=make_mask(),
        target_plus=make_mask(),
        target_minus=make_mask((7, 7)),
        valid_mask=torch.ones_like(make_mask()),
        removed_component=make_mask((0, 0)),
        removed_component_ids=("pred-far",),
        target_ids_added=("gt-far",),
        **_clean_binding("far-response"),
    )
    catalog = make_toy_raw_catalog(
        pairs=(far_response, make_identity_pair())
    )
    receipt = audit_population_observability(catalog)
    assert receipt.target_response_outside_phase_rf_pixels > 0
    assert receipt.decision is (
        CoverageStateObservabilityDecision.PHASE_RF_UNREACHABLE
    )
    assert not receipt.scalar_authorized
    assert not receipt.pp_authorized


def test_normalized_duplicate_with_opposing_targets_is_phase_conflict() -> None:
    feature = make_feature(7)
    occupancy = make_mask((6, 6))
    valid = torch.ones_like(occupancy)
    empty = make_mask()
    natural_empty = CoverageStateNaturalRecord(
        record_id="natural-duplicate-empty",
        sample_id="sample-duplicate-empty",
        group_id="group-duplicate-empty",
        state_kind="factual_no_miss",
        feature=feature,
        occupancy=occupancy,
        target=empty,
        valid_mask=valid,
        loss_valid_mask=valid & ~occupancy,
        target_ids=(),
        focus_target_ids=(),
        **_natural_binding("duplicate-empty", miss=False),
    )
    natural_target = CoverageStateNaturalRecord(
        record_id="natural-duplicate-target",
        sample_id="sample-duplicate-target",
        group_id="group-duplicate-target",
        state_kind="factual_miss",
        feature=feature * 2.0,
        occupancy=occupancy,
        target=make_mask((2, 3)),
        valid_mask=valid,
        loss_valid_mask=valid & ~occupancy,
        target_ids=("gt-duplicate",),
        focus_target_ids=("gt-duplicate",),
        **_natural_binding("duplicate-target", miss=True),
    )
    assert not torch.equal(natural_empty.feature, natural_target.feature)
    assert torch.equal(
        normalize_cslf_feature(natural_empty.feature),
        normalize_cslf_feature(natural_target.feature),
    )
    clean = CoverageStatePairRecord(
        pair_id="pair-clean-visible",
        sample_id="sample-clean-visible",
        group_id="group-clean-visible",
        pair_kind="clean_positive",
        feature=make_feature(5),
        occupancy_plus=make_mask((0, 0)),
        occupancy_minus=make_mask(),
        target_plus=empty,
        target_minus=make_mask((0, 0)),
        valid_mask=valid,
        removed_component=make_mask((0, 0)),
        removed_component_ids=("pred-visible",),
        target_ids_added=("gt-visible",),
        **_clean_binding("duplicate-clean-visible"),
    )
    catalog = make_coverage_state_raw_catalog(
        dataset="toy",
        feature_stride=TOY_STRIDE,
        source_fingerprint=stable_fingerprint({"duplicate": True}),
        natural_records=(natural_empty, natural_target),
        pair_records=(clean, make_identity_pair()),
    )
    receipt = audit_population_observability(catalog)
    assert receipt.scalar_duplicate_input_target_conflicts >= 1
    assert receipt.phase_duplicate_input_target_conflicts >= 1
    assert receipt.decision is (
        CoverageStateObservabilityDecision.STATE_TARGET_CONTRACT_UNREALIZABLE
    )


def test_identity_null_is_exact_without_conflict_or_pp_trigger() -> None:
    identity = make_identity_pair()
    catalog = make_toy_raw_catalog(
        pairs=(
            CoverageStatePairRecord(
                pair_id="pair-clean-visible",
                sample_id="sample-clean-visible",
                group_id="group-clean-visible",
                pair_kind="clean_positive",
                feature=make_feature(5),
                occupancy_plus=make_mask((0, 0)),
                occupancy_minus=make_mask(),
                target_plus=make_mask(),
                target_minus=make_mask((0, 0)),
                valid_mask=torch.ones_like(make_mask()),
                removed_component=make_mask((0, 0)),
                removed_component_ids=("pred-visible",),
                target_ids_added=("gt-visible",),
                **_clean_binding("identity-control-clean-visible"),
            ),
            identity,
        )
    )
    receipt = audit_population_observability(catalog)
    audit = next(
        value
        for value in receipt.pair_audits
        if value.pair_kind == "identity_null"
    )
    for representation in (audit.scalar, audit.phase):
        assert representation.input_plus_sha256 == (
            representation.input_minus_sha256
        )
        assert representation.target_plus_sha256 == (
            representation.target_minus_sha256
        )
        assert not representation.changed_feature_cells
        assert representation.target_response_pixels == 0
        assert not representation.duplicate_input_target_conflict
    assert receipt.decision is (
        CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
    )


@pytest.mark.parametrize(
    (
        "values",
        "expected",
    ),
    (
        (
            dict(
                informative_clean_positive_count=0,
                phase_duplicate_input_target_conflicts=0,
                target_response_outside_phase_rf_pixels=0,
                scalar_duplicate_input_target_conflicts=0,
                target_response_outside_scalar_rf_pixels=0,
            ),
            CoverageStateObservabilityDecision.INSUFFICIENT_INFORMATIVE_POPULATION,
        ),
        (
            dict(
                informative_clean_positive_count=1,
                phase_duplicate_input_target_conflicts=1,
                target_response_outside_phase_rf_pixels=1,
                scalar_duplicate_input_target_conflicts=1,
                target_response_outside_scalar_rf_pixels=1,
            ),
            CoverageStateObservabilityDecision.STATE_TARGET_CONTRACT_UNREALIZABLE,
        ),
        (
            dict(
                informative_clean_positive_count=1,
                phase_duplicate_input_target_conflicts=0,
                target_response_outside_phase_rf_pixels=1,
                scalar_duplicate_input_target_conflicts=1,
                target_response_outside_scalar_rf_pixels=1,
            ),
            CoverageStateObservabilityDecision.PHASE_RF_UNREACHABLE,
        ),
        (
            dict(
                informative_clean_positive_count=1,
                phase_duplicate_input_target_conflicts=0,
                target_response_outside_phase_rf_pixels=0,
                scalar_duplicate_input_target_conflicts=1,
                target_response_outside_scalar_rf_pixels=0,
            ),
            CoverageStateObservabilityDecision.AUTHORIZE_PP_CSLF,
        ),
        (
            dict(
                informative_clean_positive_count=1,
                phase_duplicate_input_target_conflicts=0,
                target_response_outside_phase_rf_pixels=0,
                scalar_duplicate_input_target_conflicts=0,
                target_response_outside_scalar_rf_pixels=0,
            ),
            CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF,
        ),
    ),
)
def test_gate_precedence(
    values: dict[str, int],
    expected: CoverageStateObservabilityDecision,
) -> None:
    assert decide_observability(**values) is expected


def test_receipt_is_order_invariant_and_byte_deterministic() -> None:
    first_catalog = make_toy_raw_catalog()
    second_catalog = make_toy_raw_catalog(
        pairs=tuple(reversed(first_catalog.pair_records)),
        naturals=tuple(reversed(first_catalog.natural_records)),
    )
    first = audit_population_observability(first_catalog)
    second = audit_population_observability(second_catalog)
    assert first.canonical_payload() == second.canonical_payload()
    assert first.receipt_fingerprint == second.receipt_fingerprint
    assert canonical_json(first.canonical_payload()).encode() == canonical_json(
        second.canonical_payload()
    ).encode()
