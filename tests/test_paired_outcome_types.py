from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.decoder import project_occupancy_to_feature_grid
from cure_lite.paired_outcome_types import (
    OutcomePairBatch,
    direct_projected_intervention_footprint,
    stack_outcome_pair_examples,
)
from cure_lite.paired_types import (
    PairExample,
    stack_pair_examples,
    tensor_content_fingerprint,
)


def _pair(
    *,
    kind: str,
    sample_id: str,
    component: tuple[int, int],
) -> tuple[PairExample, torch.Tensor]:
    feature = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    valid = torch.ones((1, 8, 8), dtype=torch.bool)
    plus = torch.zeros_like(valid)
    plus[0, component[0], component[1]] = True
    minus = plus.clone() if kind == "identity_null" else torch.zeros_like(plus)
    removed = plus & ~minus
    completion_plus = torch.zeros_like(valid)
    completion_minus = torch.zeros_like(valid)
    clean = torch.zeros_like(valid)
    evaluation_gt_id: int | None
    native_gt_id: int | None
    pred_id: int | None
    clean_checks: tuple[bool | None, bool | None, bool | None, bool | None]
    if kind == "clean_positive":
        completion_minus[0, component[0], component[1]] = True
        completion_minus[0, component[0], min(component[1] + 1, 7)] = True
        clean = completion_minus.clone()
        evaluation_gt_id = 1
        native_gt_id = 11
        pred_id = 21
        clean_checks = (True, True, True, True)
    elif kind == "component_null":
        completion_plus[0, 0, 7] = True
        completion_minus.copy_(completion_plus)
        evaluation_gt_id = None
        native_gt_id = None
        pred_id = 22
        clean_checks = (None, None, None, None)
    elif kind == "identity_null":
        completion_plus[0, 0, 7] = True
        completion_minus.copy_(completion_plus)
        evaluation_gt_id = None
        native_gt_id = None
        pred_id = None
        clean_checks = (None, None, None, None)
    else:
        raise AssertionError(kind)

    projected_plus = project_occupancy_to_feature_grid(
        plus.unsqueeze(0),
        (2, 2),
    )
    projected_minus = project_occupancy_to_feature_grid(
        minus.unsqueeze(0),
        (2, 2),
    )
    pair = PairExample(
        pair_id=stable_fingerprint(
            {"kind": kind, "sample_id": sample_id, "component": component}
        ),
        pair_kind=kind,
        sample_id=sample_id,
        group_id=f"group-{sample_id}",
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        removed_component=removed,
        image_valid_mask=valid,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        label_increment=(completion_minus & ~completion_plus).to(torch.float32),
        clean_increment=clean,
        evaluation_gt_id=evaluation_gt_id,
        native_gt_id=native_gt_id,
        pred_id=pred_id,
        feature_fingerprint=tensor_content_fingerprint(feature),
        before_match_fingerprint=stable_fingerprint(
            {"sample_id": sample_id, "state": "before"}
        ),
        after_match_fingerprint=stable_fingerprint(
            {"sample_id": sample_id, "state": "after"}
        ),
        projected_occupancy_plus_fingerprint=tensor_content_fingerprint(
            projected_plus
        ),
        projected_occupancy_minus_fingerprint=tensor_content_fingerprint(
            projected_minus
        ),
        projection_visible=kind != "identity_null",
        geometry_safe_bijective_lineage=clean_checks[0],
        selected_gt_is_only_new_unmatched=clean_checks[1],
        other_match_identities_unchanged=clean_checks[2],
        preexisting_unmatched_gt_noninterference=clean_checks[3],
    )
    gt_union = completion_plus | completion_minus
    return pair, gt_union


def _mixed_outcome() -> OutcomePairBatch:
    clean, clean_gt = _pair(
        kind="clean_positive",
        sample_id="clean-source",
        component=(1, 1),
    )
    null, null_gt = _pair(
        kind="component_null",
        sample_id="null-source",
        component=(6, 6),
    )
    return stack_outcome_pair_examples(
        (clean, null),
        gt_union_by_sample={
            clean.sample_id: clean_gt,
            null.sample_id: null_gt,
        },
        device="cpu",
    )


def test_stack_accepts_mixed_clean_and_component_outcomes() -> None:
    outcome = _mixed_outcome()

    assert outcome.pair_batch.pair_kinds == (
        "clean_positive",
        "component_null",
    )
    assert torch.equal(
        outcome.response_stratum,
        outcome.completion_minus & ~outcome.completion_plus,
    )
    assert bool(outcome.response_stratum[0].any())
    assert not bool(outcome.response_stratum[1].any())
    assert torch.equal(
        outcome.completion_plus[1],
        outcome.completion_minus[1],
    )
    outcome.validate()


def test_direct_footprint_is_component_union_nearest_lifted_projection_xor() -> None:
    outcome = _mixed_outcome()
    pair = outcome.pair_batch
    projected_plus = project_occupancy_to_feature_grid(
        pair.occupancy_plus,
        (2, 2),
    )
    projected_minus = project_occupancy_to_feature_grid(
        pair.occupancy_minus,
        (2, 2),
    )
    lifted = F.interpolate(
        (projected_plus ^ projected_minus).to(torch.float32),
        size=(8, 8),
        mode="nearest",
    ).to(torch.bool)
    expected = (
        (pair.occupancy_plus & ~pair.occupancy_minus) | lifted
    ) & pair.image_valid_mask

    assert torch.equal(outcome.intervention_footprint, expected)
    assert torch.equal(
        direct_projected_intervention_footprint(pair),
        expected,
    )
    assert torch.all(outcome.removed_component <= outcome.intervention_footprint)


def test_response_local_and_global_strata_exactly_partition_valid_domain() -> None:
    outcome = _mixed_outcome()
    response = outcome.response_stratum
    local = outcome.local_zero_stratum
    global_zero = outcome.global_zero_stratum

    assert not torch.any(response & local)
    assert not torch.any(response & global_zero)
    assert not torch.any(local & global_zero)
    assert torch.equal(
        response | local | global_zero,
        outcome.pair_batch.image_valid_mask,
    )
    assert torch.all(global_zero.flatten(1).any(dim=1))
    assert bool(local[1].any())


def test_outcome_rejects_identity_null() -> None:
    identity, identity_gt = _pair(
        kind="identity_null",
        sample_id="identity-source",
        component=(1, 1),
    )

    with pytest.raises(ValueError, match="clean_positive/component_null"):
        stack_outcome_pair_examples(
            (identity,),
            gt_union_by_sample={identity.sample_id: identity_gt},
            device="cpu",
        )


def test_outcome_rejects_nonmonotone_completion_truth() -> None:
    outcome = _mixed_outcome()
    invalid_plus = outcome.completion_plus.clone()
    invalid_plus[0, 0, 7, 7] = True
    invalid_gt = outcome.gt_union.clone()
    invalid_gt[0, 0, 7, 7] = True

    with pytest.raises(ValueError, match="subset of completion_minus"):
        OutcomePairBatch(
            pair_batch=outcome.pair_batch,
            completion_plus=invalid_plus,
            completion_minus=outcome.completion_minus,
            gt_union=invalid_gt,
            intervention_footprint=outcome.intervention_footprint,
        )


def test_outcome_rejects_component_completion_change() -> None:
    outcome = _mixed_outcome()
    invalid_minus = outcome.completion_minus.clone()
    invalid_minus[1, 0, 1, 7] = True
    invalid_gt = outcome.gt_union.clone()
    invalid_gt[1, 0, 1, 7] = True

    with pytest.raises(
        ValueError,
        match="component_null requires completion_plus equal completion_minus",
    ):
        OutcomePairBatch(
            pair_batch=outcome.pair_batch,
            completion_plus=outcome.completion_plus,
            completion_minus=invalid_minus,
            gt_union=invalid_gt,
            intervention_footprint=outcome.intervention_footprint,
        )


def test_outcome_rejects_label_increment_that_is_not_exact_D() -> None:
    outcome = _mixed_outcome()
    invalid_label = outcome.pair_batch.label_increment.clone()
    invalid_label[0, 0, 1, 2] = 0.0
    invalid_pair = replace(outcome.pair_batch, label_increment=invalid_label)

    with pytest.raises(
        ValueError,
        match="label_increment must equal completion_minus minus completion_plus",
    ):
        OutcomePairBatch(
            pair_batch=invalid_pair,
            completion_plus=outcome.completion_plus,
            completion_minus=outcome.completion_minus,
            gt_union=outcome.gt_union,
            intervention_footprint=outcome.intervention_footprint,
        )


def test_outcome_rejects_nonwritable_or_non_gt_completion() -> None:
    outcome = _mixed_outcome()
    invalid_plus = outcome.completion_plus.clone()
    invalid_plus[0] |= outcome.pair_batch.occupancy_plus[0]
    invalid_gt = outcome.gt_union | invalid_plus
    with pytest.raises(ValueError, match="writable under occupancy_plus"):
        OutcomePairBatch(
            pair_batch=outcome.pair_batch,
            completion_plus=invalid_plus,
            completion_minus=outcome.completion_minus | invalid_plus,
            gt_union=invalid_gt,
            intervention_footprint=outcome.intervention_footprint,
        )

    invalid_minus = outcome.completion_minus.clone()
    invalid_minus[0, 0, 7, 7] = True
    with pytest.raises(ValueError, match="subsets of gt_union"):
        OutcomePairBatch(
            pair_batch=outcome.pair_batch,
            completion_plus=outcome.completion_plus,
            completion_minus=invalid_minus,
            gt_union=outcome.gt_union,
            intervention_footprint=outcome.intervention_footprint,
        )


def test_outcome_rejects_tampered_direct_footprint() -> None:
    outcome = _mixed_outcome()
    invalid = outcome.intervention_footprint.clone()
    invalid[0, 0, 0, 0] = ~invalid[0, 0, 0, 0]

    with pytest.raises(ValueError, match="nearest_lift"):
        OutcomePairBatch(
            pair_batch=outcome.pair_batch,
            completion_plus=outcome.completion_plus,
            completion_minus=outcome.completion_minus,
            gt_union=outcome.gt_union,
            intervention_footprint=invalid,
        )


def test_outcome_rejects_empty_global_stratum() -> None:
    clean, clean_gt = _pair(
        kind="clean_positive",
        sample_id="single-cell-grid",
        component=(1, 1),
    )
    feature = clean.feature[:, :, :1, :1].contiguous()
    projected_plus = project_occupancy_to_feature_grid(
        clean.occupancy_plus.unsqueeze(0),
        (1, 1),
    )
    projected_minus = project_occupancy_to_feature_grid(
        clean.occupancy_minus.unsqueeze(0),
        (1, 1),
    )
    clean = replace(
        clean,
        feature=feature,
        feature_fingerprint=tensor_content_fingerprint(feature),
        projected_occupancy_plus_fingerprint=tensor_content_fingerprint(
            projected_plus
        ),
        projected_occupancy_minus_fingerprint=tensor_content_fingerprint(
            projected_minus
        ),
    )

    with pytest.raises(ValueError, match="non-empty global stratum G"):
        stack_outcome_pair_examples(
            (clean,),
            gt_union_by_sample={clean.sample_id: clean_gt},
            device="cpu",
        )


def test_stack_rejects_missing_or_malformed_gt_union() -> None:
    clean, clean_gt = _pair(
        kind="clean_positive",
        sample_id="missing-gt",
        component=(1, 1),
    )
    with pytest.raises(KeyError, match="missing source"):
        stack_outcome_pair_examples(
            (clean,),
            gt_union_by_sample={},
            device="cpu",
        )
    with pytest.raises(TypeError, match="CPU bool"):
        stack_outcome_pair_examples(
            (clean,),
            gt_union_by_sample={clean.sample_id: clean_gt.to(torch.float32)},
            device="cpu",
        )


def test_post_construction_tensor_tampering_is_detected_by_validate() -> None:
    outcome = _mixed_outcome()
    outcome.intervention_footprint[0, 0, 0, 0] = (
        ~outcome.intervention_footprint[0, 0, 0, 0]
    )

    with pytest.raises(ValueError, match="nearest_lift"):
        outcome.validate()
