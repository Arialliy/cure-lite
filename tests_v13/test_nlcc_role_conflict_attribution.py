from __future__ import annotations

import json

import torch

from cure_lite.nlcc_development_inputs import (
    build_nlcc_development_pair_specs,
    nlcc_development_fingerprint,
)
from cure_lite.nlcc_role_conflict_attribution import (
    ROLE_CONFLICT_ATTRIBUTION_ALGORITHM_VERSION,
    audit_nlcc_role_conflict_attribution,
    build_nlcc_development_role_conflict_attribution_receipt,
    factorial_input_key_fingerprint,
)
from cure_lite.nlcc_role_quotient_audit import NLCCSupervisedState


def _state(
    *,
    state_id: str,
    role: str,
    feature: torch.Tensor,
    occupancy: torch.Tensor,
    position: tuple[int, int],
    label: bool,
) -> NLCCSupervisedState:
    target = torch.zeros_like(occupancy)
    target[0, 0, position[0], position[1]] = label
    valid = torch.zeros_like(occupancy)
    valid[0, 0, position[0], position[1]] = True
    return NLCCSupervisedState(
        state_id=state_id,
        supervision_role=role,
        feature=feature,
        occupancy=occupancy,
        target=target,
        valid_mask=valid,
    )


def _base_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    feature = torch.zeros(1, 2, 3, 3, dtype=torch.float32)
    feature[0, 0, 1, 1] = 2.0
    feature[0, 1, 0, 2] = -3.0
    occupancy = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    occupancy[0, 0, 4, 0] = True
    return feature, occupancy


def test_factorial_axes_separate_value_and_sign_shortcuts() -> None:
    feature, occupancy = _base_tensors()
    value_changed = feature.clone()
    value_changed[feature != 0.0] *= 2.0
    sign_changed = feature.clone()
    sign_changed[0, 1, 0, 2] *= -1.0

    value_states = (
        _state(
            state_id="negative-value",
            role="metadata-a",
            feature=feature,
            occupancy=occupancy,
            position=(1, 4),
            label=False,
        ),
        _state(
            state_id="positive-value",
            role="metadata-b",
            feature=value_changed,
            occupancy=occupancy,
            position=(1, 4),
            label=True,
        ),
    )
    value_receipt = audit_nlcc_role_conflict_attribution(
        value_states,
        input_fingerprint="toy-value-factor",
        feature_stride=2,
    )
    value_factors = value_receipt["factors"]
    assert value_factors["exact__absolute"]["conflict_key_count"] == 0
    assert value_factors["absolute_value__absolute"][
        "conflict_key_count"
    ] == 0
    assert value_factors["signed_support__absolute"][
        "conflict_key_count"
    ] == 1
    assert value_factors["unsigned_support__absolute"][
        "conflict_key_count"
    ] == 1
    assert value_receipt["coarsening_comparisons"][
        "remove_numerical_values_at_absolute_origin"
    ]["new_opposing_pair_count"] == 1

    sign_states = (
        _state(
            state_id="negative-sign",
            role="metadata-a",
            feature=feature,
            occupancy=occupancy,
            position=(1, 4),
            label=False,
        ),
        _state(
            state_id="positive-sign",
            role="metadata-c",
            feature=sign_changed,
            occupancy=occupancy,
            position=(1, 4),
            label=True,
        ),
    )
    sign_receipt = audit_nlcc_role_conflict_attribution(
        sign_states,
        input_fingerprint="toy-sign-factor",
        feature_stride=2,
    )
    sign_factors = sign_receipt["factors"]
    assert sign_factors["exact__absolute"]["conflict_key_count"] == 0
    assert sign_factors["signed_support__absolute"][
        "conflict_key_count"
    ] == 0
    assert sign_factors["absolute_value__absolute"][
        "conflict_key_count"
    ] == 1
    assert sign_factors["unsigned_support__absolute"][
        "conflict_key_count"
    ] == 1
    assert sign_receipt["coarsening_comparisons"][
        "remove_sign_at_absolute_origin"
    ]["new_opposing_pair_count"] == 1
    assert sign_receipt["decision"]["training_authorized"] is False


def test_relative_axis_removes_only_absolute_translation_identity() -> None:
    feature_a = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    feature_a[0, 0, 1, 1] = 2.0
    occupancy_a = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    occupancy_a[0, 0, 2, 2] = True

    feature_b = torch.zeros_like(feature_a)
    feature_b[0, 0, 2, 2] = 2.0
    occupancy_b = torch.zeros_like(occupancy_a)
    occupancy_b[0, 0, 4, 4] = True
    states = (
        _state(
            state_id="negative-origin",
            role="metadata-a",
            feature=feature_a,
            occupancy=occupancy_a,
            position=(3, 3),
            label=False,
        ),
        _state(
            state_id="positive-origin",
            role="metadata-b",
            feature=feature_b,
            occupancy=occupancy_b,
            position=(5, 5),
            label=True,
        ),
    )

    receipt = audit_nlcc_role_conflict_attribution(
        states,
        input_fingerprint="toy-origin",
        feature_stride=2,
    )

    assert receipt["factors"]["exact__absolute"][
        "conflict_key_count"
    ] == 0
    assert receipt["factors"]["exact__relative"][
        "conflict_key_count"
    ] == 1
    comparison = receipt["coarsening_comparisons"][
        "remove_absolute_origin_with_exact_values"
    ]
    assert comparison["new_opposing_pair_count"] == 1


def test_reporting_metadata_never_changes_factorial_keys() -> None:
    feature, occupancy = _base_tensors()
    first = _state(
        state_id="first-id",
        role="first-role",
        feature=feature,
        occupancy=occupancy,
        position=(1, 4),
        label=False,
    )
    second = _state(
        state_id="second-id",
        role="second-role",
        feature=feature,
        occupancy=occupancy,
        position=(1, 4),
        label=False,
    )
    for feature_mode in (
        "exact",
        "absolute_value",
        "signed_support",
        "unsigned_support",
    ):
        for origin_mode in ("absolute", "relative"):
            assert factorial_input_key_fingerprint(
                first,
                (1, 4),
                feature_value_mode=feature_mode,
                origin_mode=origin_mode,
                feature_stride=2,
            ) == factorial_input_key_fingerprint(
                second,
                (1, 4),
                feature_value_mode=feature_mode,
                origin_mode=origin_mode,
                feature_stride=2,
            )


def test_factorial_receipt_is_byte_deterministic() -> None:
    feature, occupancy = _base_tensors()
    states = (
        _state(
            state_id="negative",
            role="negative-role",
            feature=feature,
            occupancy=occupancy,
            position=(1, 4),
            label=False,
        ),
        _state(
            state_id="positive",
            role="positive-role",
            feature=2.0 * feature,
            occupancy=occupancy,
            position=(1, 4),
            label=True,
        ),
    )
    first = audit_nlcc_role_conflict_attribution(
        states,
        input_fingerprint="toy-repeat",
        feature_stride=2,
    )
    second = audit_nlcc_role_conflict_attribution(
        states,
        input_fingerprint="toy-repeat",
        feature_stride=2,
    )

    first_bytes = json.dumps(
        first,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    second_bytes = json.dumps(
        second,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert first_bytes == second_bytes
    assert first["schema_version"] == (
        ROLE_CONFLICT_ATTRIBUTION_ALGORITHM_VERSION
    )
    assert first["receipt_fingerprint"] == second["receipt_fingerprint"]


def test_frozen_development_factorial_attribution_matches_r13_1() -> None:
    receipt = (
        build_nlcc_development_role_conflict_attribution_receipt(
            max_examples=2,
            max_records_per_label=1,
        )
    )
    specs = build_nlcc_development_pair_specs()

    assert receipt["input_fingerprint"] == (
        nlcc_development_fingerprint(specs)
    )
    assert receipt["population"]["state_count"] == 96
    assert receipt["population"]["supervised_record_count"] == 75192
    factors = receipt["factors"]
    assert factors["exact__absolute"]["conflict_key_count"] == 0
    assert factors["exact__relative"]["conflict_key_count"] == 0
    assert factors["absolute_value__absolute"][
        "conflict_key_count"
    ] == 0
    assert factors["absolute_value__relative"][
        "conflict_key_count"
    ] == 0
    assert factors["signed_support__relative"][
        "conflict_key_count"
    ] == 8
    assert factors["unsigned_support__relative"][
        "conflict_key_count"
    ] == 7
    assert receipt["decision"]["role_quotient_conflict_free"] is False
    assert receipt["decision"]["training_authorized"] is False
