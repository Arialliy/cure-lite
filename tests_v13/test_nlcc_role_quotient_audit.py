from __future__ import annotations

import json

import torch

from cure_lite.nlcc_development_inputs import (
    build_nlcc_development_pair_specs,
    nlcc_development_fingerprint,
)
from cure_lite.nlcc_role_quotient_audit import (
    NLCCSupervisedState,
    ROLE_QUOTIENT_ALGORITHM_VERSION,
    audit_nlcc_role_quotient,
    build_nlcc_development_role_quotient_receipt,
    canonical_input_key_fingerprint,
)


def _state(
    *,
    state_id: str,
    role: str,
    feature_value: float,
    position: tuple[int, int],
    label: bool,
) -> NLCCSupervisedState:
    feature = torch.zeros(1, 2, 3, 3, dtype=torch.float32)
    feature[0, 0, 1, 1] = feature_value
    feature[0, 1, 0, 2] = 1.5 * feature_value
    occupancy = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    occupancy[0, 0, 4, 0] = True
    target = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    target[0, 0, position[0], position[1]] = label
    valid = torch.zeros_like(target)
    valid[0, 0, position[0], position[1]] = True
    return NLCCSupervisedState(
        state_id=state_id,
        supervision_role=role,
        feature=feature,
        occupancy=occupancy,
        target=target,
        valid_mask=valid,
    )


def test_exact_collision_is_a_hard_input_identifiability_failure() -> None:
    negative = _state(
        state_id="negative",
        role="role_a",
        feature_value=2.0,
        position=(1, 4),
        label=False,
    )
    positive = _state(
        state_id="positive",
        role="role_b",
        feature_value=2.0,
        position=(1, 4),
        label=True,
    )

    receipt = audit_nlcc_role_quotient(
        (negative, positive),
        input_fingerprint="toy-identical-input",
        feature_stride=2,
    )

    assert receipt["exact_tensor"]["conflict_key_count"] == 1
    assert receipt["decision"]["hard_gate_pass"] is False
    assert receipt["decision"]["development_authorized"] is False
    example = receipt["exact_tensor"]["examples"][0]
    assert example["negative"]["supervision_role_counts"] == {"role_a": 1}
    assert example["positive"]["supervision_role_counts"] == {"role_b": 1}


def test_hard_key_uses_nlcc_effective_count_not_raw_occupancy_pixels() -> None:
    negative = _state(
        state_id="negative-raw-occupancy",
        role="role_a",
        feature_value=2.0,
        position=(1, 4),
        label=False,
    )
    moved_occupancy = negative.occupancy.clone()
    moved_occupancy.zero_()
    moved_occupancy[0, 0, 5, 1] = True
    assert not torch.equal(negative.occupancy, moved_occupancy)
    positive_target = negative.target.clone()
    positive_target[0, 0, 1, 4] = True
    positive = NLCCSupervisedState(
        state_id="positive-raw-occupancy",
        supervision_role="role_b",
        feature=negative.feature,
        occupancy=moved_occupancy,
        target=positive_target,
        valid_mask=negative.valid_mask,
    )

    receipt = audit_nlcc_role_quotient(
        (negative, positive),
        input_fingerprint="toy-effective-count",
        feature_stride=2,
    )

    assert receipt["exact_tensor"]["conflict_key_count"] == 1
    assert receipt["decision"]["hard_gate_pass"] is False


def test_unsigned_role_quotient_removes_amplitude_and_sign_identity() -> None:
    negative = _state(
        state_id="negative-amplitude",
        role="role_a",
        feature_value=2.0,
        position=(1, 4),
        label=False,
    )
    positive = _state(
        state_id="positive-amplitude",
        role="role_b",
        feature_value=-7.0,
        position=(1, 4),
        label=True,
    )

    receipt = audit_nlcc_role_quotient(
        (negative, positive),
        input_fingerprint="toy-amplitude-identity",
        feature_stride=2,
    )

    assert receipt["exact_tensor"]["conflict_key_count"] == 0
    assert receipt["signed_amplitude_quotient"]["conflict_key_count"] == 0
    assert receipt["role_quotient"]["conflict_key_count"] == 1
    assert receipt["decision"]["hard_gate_pass"] is True
    assert receipt["decision"]["role_gate_pass"] is False
    assert receipt["decision"]["development_authorized"] is False


def test_reporting_metadata_never_changes_an_input_key() -> None:
    first = _state(
        state_id="opaque-a",
        role="explanation-a",
        feature_value=2.0,
        position=(1, 4),
        label=False,
    )
    second = NLCCSupervisedState(
        state_id="opaque-b",
        supervision_role="explanation-b",
        feature=first.feature,
        occupancy=first.occupancy,
        target=first.target,
        valid_mask=first.valid_mask,
    )

    for quotient in ("exact", "signed_role", "role", "d4_diagnostic"):
        assert canonical_input_key_fingerprint(
            first,
            (1, 4),
            feature_stride=2,
            quotient=quotient,
        ) == canonical_input_key_fingerprint(
            second,
            (1, 4),
            feature_stride=2,
            quotient=quotient,
        )


def test_d4_diagnostic_jointly_transforms_inputs_and_output_phase() -> None:
    original = _state(
        state_id="original",
        role="original_role",
        feature_value=2.0,
        position=(1, 4),
        label=True,
    )
    rotated_position = (6 - 1 - 4, 1)
    rotated = NLCCSupervisedState(
        state_id="rotated",
        supervision_role="rotated_role",
        feature=torch.rot90(original.feature, 1, dims=(-2, -1)),
        occupancy=torch.rot90(original.occupancy, 1, dims=(-2, -1)),
        target=torch.rot90(original.target, 1, dims=(-2, -1)),
        valid_mask=torch.rot90(original.valid_mask, 1, dims=(-2, -1)),
    )

    # Raw phase is retained by the development-gate quotient, so the two
    # orientations need not collide.
    assert canonical_input_key_fingerprint(
        original,
        (1, 4),
        feature_stride=2,
        quotient="role",
    ) != canonical_input_key_fingerprint(
        rotated,
        rotated_position,
        feature_stride=2,
        quotient="role",
    )
    # The separate D4 diagnostic rotates feature, occupancy, and the output
    # coordinate together and therefore canonicalizes the pair consistently.
    assert canonical_input_key_fingerprint(
        original,
        (1, 4),
        feature_stride=2,
        quotient="d4_diagnostic",
    ) == canonical_input_key_fingerprint(
        rotated,
        rotated_position,
        feature_stride=2,
        quotient="d4_diagnostic",
    )

    misaligned = NLCCSupervisedState(
        state_id="misaligned",
        supervision_role="misaligned_role",
        feature=rotated.feature,
        occupancy=rotated.occupancy,
        target=original.target,
        valid_mask=original.valid_mask,
    )
    assert canonical_input_key_fingerprint(
        original,
        (1, 4),
        feature_stride=2,
        quotient="d4_diagnostic",
    ) != canonical_input_key_fingerprint(
        misaligned,
        (1, 4),
        feature_stride=2,
        quotient="d4_diagnostic",
    )


def test_receipt_is_json_compatible_and_byte_deterministic() -> None:
    states = (
        _state(
            state_id="a",
            role="role_a",
            feature_value=2.0,
            position=(1, 4),
            label=False,
        ),
        _state(
            state_id="b",
            role="role_b",
            feature_value=-7.0,
            position=(1, 4),
            label=True,
        ),
    )
    first = audit_nlcc_role_quotient(
        states,
        input_fingerprint="toy-determinism",
        feature_stride=2,
    )
    second = audit_nlcc_role_quotient(
        states,
        input_fingerprint="toy-determinism",
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
    assert first["receipt_fingerprint"] == second["receipt_fingerprint"]
    assert first["schema_version"] == ROLE_QUOTIENT_ALGORITHM_VERSION
    assert first["algorithm"]["d4_is_development_gate"] is False
    assert first["algorithm"]["d4_equivariance_claimed"] is False
    assert first["d4_joint_transform_diagnostic"]["status"] == (
        "not_evaluated"
    )


def test_frozen_v12_development_receipt_binds_existing_input() -> None:
    first = build_nlcc_development_role_quotient_receipt(max_examples=3)

    assert first["input_fingerprint"] == nlcc_development_fingerprint(
        build_nlcc_development_pair_specs()
    )
    assert first["population"]["state_count"] == 96
    assert first["decision"]["hard_gate_pass"] is True
    assert first["exact_tensor"]["conflict_key_count"] == 0
    # The frozen v12 input is not authorized merely because its amplitude
    # hashes make the exact tensors distinct.
    assert first["role_quotient"]["conflict_key_count"] > 0
    assert first["decision"]["role_gate_pass"] is False
    assert first["decision"]["development_authorized"] is False
    assert first["d4_joint_transform_diagnostic"]["status"] == (
        "not_evaluated"
    )
