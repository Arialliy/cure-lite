from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest
import torch

from cure_lite.paired_transition_types import (
    AnchoredPairBatch,
    stack_anchored_pair_examples,
)
from cure_lite.paired_types import (
    PairBatch,
    PairExample,
    tensor_content_fingerprint,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean_example(
    *,
    sample_id: str = "source-a",
    row: int = 1,
    column: int = 1,
) -> tuple[PairExample, torch.Tensor]:
    height = width = 4
    feature = torch.zeros(1, 2, height, width)
    feature[0, 0, row, column] = 2.0
    occupancy_plus = torch.zeros(1, height, width, dtype=torch.bool)
    occupancy_plus[0, row, column] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)
    removed = occupancy_plus.clone()
    valid = torch.ones_like(occupancy_plus)
    completion_plus = torch.zeros_like(occupancy_plus)
    completion_plus[0, 3, 0] = True
    increment = torch.zeros_like(occupancy_plus)
    increment[0, row, column] = True
    completion_minus = completion_plus | increment
    gt_union = completion_minus.clone()
    projected_plus = occupancy_plus.unsqueeze(0)
    projected_minus = occupancy_minus.unsqueeze(0)
    example = PairExample(
        pair_id=_sha(f"pair:{sample_id}:{row}:{column}"),
        pair_kind="clean_positive",
        sample_id=sample_id,
        group_id=f"group:{sample_id}",
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        removed_component=removed,
        image_valid_mask=valid,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        label_increment=increment.to(torch.float32),
        clean_increment=increment,
        evaluation_gt_id=1,
        native_gt_id=1,
        pred_id=1,
        feature_fingerprint=tensor_content_fingerprint(feature),
        before_match_fingerprint=_sha(f"before:{sample_id}"),
        after_match_fingerprint=_sha(f"after:{sample_id}"),
        projected_occupancy_plus_fingerprint=(
            tensor_content_fingerprint(projected_plus)
        ),
        projected_occupancy_minus_fingerprint=(
            tensor_content_fingerprint(projected_minus)
        ),
        projection_visible=True,
        geometry_safe_bijective_lineage=True,
        selected_gt_is_only_new_unmatched=True,
        other_match_identities_unchanged=True,
        preexisting_unmatched_gt_noninterference=True,
    )
    return example, gt_union


def _anchored_single() -> AnchoredPairBatch:
    example, gt_union = _clean_example()
    return stack_anchored_pair_examples(
        (example,),
        gt_union_by_sample={example.sample_id: gt_union},
        device="cpu",
    )


def test_valid_anchor_reconstructs_minus_and_preserves_pair_identity() -> None:
    anchored = _anchored_single()

    assert anchored.pair_batch.feature.shape[0] == 1
    assert anchored.feature is anchored.pair_batch.feature
    assert anchored.pair_ids is anchored.pair_batch.pair_ids
    assert anchored.sample_ids == ("source-a",)
    assert anchored.pair_kinds == ("clean_positive",)
    assert torch.equal(
        anchored.completion_minus,
        anchored.completion_plus
        | anchored.label_increment.to(torch.bool),
    )
    assert anchored.completion_plus.dtype == torch.bool
    assert anchored.gt_union.dtype == torch.bool
    assert anchored.completion_plus.device == anchored.feature.device


def test_anchor_is_a_frozen_value_object_and_clones_supervision() -> None:
    anchored = _anchored_single()
    with pytest.raises(FrozenInstanceError):
        anchored.gt_union = torch.zeros_like(anchored.gt_union)  # type: ignore[misc]

    completion = anchored.completion_plus.clone()
    union = anchored.gt_union.clone()
    rebuilt = AnchoredPairBatch(
        pair_batch=anchored.pair_batch,
        completion_plus=completion,
        gt_union=union,
    )
    completion.zero_()
    union.zero_()
    assert torch.any(rebuilt.completion_plus)
    assert torch.any(rebuilt.gt_union)


def test_stack_preserves_order_device_dtype_and_exact_example_truth() -> None:
    first, first_union = _clean_example(sample_id="source-a", row=1, column=1)
    second, second_union = _clean_example(sample_id="source-b", row=2, column=2)

    anchored = stack_anchored_pair_examples(
        (first, second),
        gt_union_by_sample={
            "source-a": first_union,
            "source-b": second_union,
        },
        device=torch.device("cpu"),
    )

    assert anchored.sample_ids == ("source-a", "source-b")
    assert anchored.pair_ids == (first.pair_id, second.pair_id)
    assert anchored.feature.dtype == torch.float32
    assert anchored.completion_plus.dtype == torch.bool
    assert torch.equal(anchored.completion_minus[0], first.completion_minus)
    assert torch.equal(anchored.completion_minus[1], second.completion_minus)


def test_stack_requires_a_gt_union_for_every_selected_source() -> None:
    example, _ = _clean_example()
    with pytest.raises(KeyError, match="source-a"):
        stack_anchored_pair_examples(
            (example,),
            gt_union_by_sample={},
            device="cpu",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("completion_outside_valid", "valid and writable"),
        ("completion_overlaps_plus", "valid and writable"),
        ("completion_outside_gt", "only gt_union"),
        ("increment_outside_gt", "label_increment.*gt_union"),
        ("completion_overlaps_increment", "must be disjoint"),
        ("gt_outside_valid", "gt_union.*image_valid_mask"),
    ],
)
def test_anchor_rejects_invalid_absolute_transition_semantics(
    mutation: str,
    message: str,
) -> None:
    valid_anchor = _anchored_single()
    batch = valid_anchor.pair_batch
    completion = valid_anchor.completion_plus.clone()
    union = valid_anchor.gt_union.clone()

    if mutation == "completion_outside_valid":
        valid = batch.image_valid_mask.clone()
        valid[0, 0, 3, 0] = False
        batch = replace(batch, image_valid_mask=valid)
    elif mutation == "completion_overlaps_plus":
        occupancy_plus = batch.occupancy_plus.clone()
        occupancy_plus[completion] = True
        batch = replace(batch, occupancy_plus=occupancy_plus)
    elif mutation == "completion_outside_gt":
        union[completion] = False
    elif mutation == "increment_outside_gt":
        union[batch.label_increment.to(torch.bool)] = False
    elif mutation == "completion_overlaps_increment":
        completion[batch.label_increment.to(torch.bool)] = True
    elif mutation == "gt_outside_valid":
        union[0, 0, 0, 3] = True
        valid = batch.image_valid_mask.clone()
        valid[0, 0, 0, 3] = False
        batch = replace(batch, image_valid_mask=valid)
    else:
        raise AssertionError("unknown test mutation")

    with pytest.raises(ValueError, match=message):
        AnchoredPairBatch(
            pair_batch=batch,
            completion_plus=completion,
            gt_union=union,
        )


def test_anchor_rejects_non_clean_pair_roles() -> None:
    anchored = _anchored_single()
    clean = anchored.pair_batch
    component_null = PairBatch(
        feature=clean.feature,
        occupancy_plus=clean.occupancy_plus,
        occupancy_minus=clean.occupancy_minus,
        label_increment=torch.zeros_like(clean.label_increment),
        image_valid_mask=clean.image_valid_mask,
        pair_ids=clean.pair_ids,
        sample_ids=clean.sample_ids,
        group_ids=clean.group_ids,
        pair_kinds=("component_null",),
        projection_visible=(True,),
    )

    with pytest.raises(ValueError, match="only clean_positive"):
        AnchoredPairBatch(
            pair_batch=component_null,
            completion_plus=torch.zeros_like(clean.image_valid_mask),
            gt_union=anchored.gt_union,
        )
