"""Input-identifiability audit for the frozen NLCC-v12 populations.

The audit has two deliberately different meanings:

``exact_tensor``
    A hard identifiability check.  Two supervised records collide only when
    the complete NLCC-effective inputs ``(feature, local 3x3 count field)``
    and the output coordinate are byte-identical but their binary supervision
    differs.  Raw occupancy differences which collapse to the same NLCC count
    field are intentionally not treated as distinguishable.

``role_quotient``
    A development check which removes the deterministic feature-amplitude
    identity shortcut.  Non-zero feature values become unsigned channel-role
    markers, absolute spatial origin is removed, the complete local 3x3 count
    field is expressed relative to the supervised output cell, and output
    phase is retained.  Metadata such as group, sample, match, anchor role,
    and endpoint role never participates in either key.

A signed-amplitude quotient and a jointly transformed dihedral (D4) quotient
are reported as diagnostics.  D4 is intentionally *not* a gate: PixelShuffle
phase channels are not assumed to be rotation/flip equivariant.  Every D4
candidate transforms the full feature tensor, occupancy tensor, and supervised
output position together before the phase is recomputed.

This module is additive and does not modify the frozen NLCC-v12 input
generators or formal artifacts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .cache.schema import stable_fingerprint
from .decoder import project_occupancy_to_feature_grid
from .nlcc_dataset_free_inputs import (
    FEATURE_STRIDE,
    NLCCInputProfile,
    NLCCPairSpec,
    build_factual_population,
    build_outcome_batch,
    build_pair_specs,
)
from .nlcc_development_inputs import (
    DEVELOPMENT_PROFILE,
    nlcc_development_fingerprint,
)
from .paired_types import tensor_content_fingerprint


ROLE_QUOTIENT_ALGORITHM_VERSION = (
    "cure-lite.nlcc-role-quotient-audit.v1"
)
NLCC_LOCAL_COUNT_KERNEL_SIZE = 3
_D4_OPERATIONS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "mirror",
    "mirror_rot90",
    "mirror_rot180",
    "mirror_rot270",
)


def _single_feature(value: Tensor) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.float32
        or value.ndim != 4
        or value.shape[0] != 1
        or value.shape[1] < 1
        or min(value.shape[-2:]) < 1
    ):
        raise TypeError("feature must be float32 with shape [1,C,h,w]")
    if value.requires_grad or not torch.isfinite(value).all():
        raise ValueError("feature must be finite and detached")
    return value.detach().cpu().clone().contiguous()


def _single_bool(value: Tensor, *, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.bool
        or value.ndim != 4
        or value.shape[0] != 1
        or value.shape[1] != 1
        or min(value.shape[-2:]) < 1
    ):
        raise TypeError(f"{name} must be bool with shape [1,1,H,W]")
    return value.detach().cpu().clone().contiguous()


def _single_binary_target(value: Tensor) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 4
        or value.shape[0] != 1
        or value.shape[1] != 1
    ):
        raise TypeError("target must have shape [1,1,H,W]")
    if value.dtype == torch.bool:
        result = value.detach().cpu().clone()
    elif value.dtype.is_floating_point:
        if not torch.isfinite(value).all():
            raise ValueError("target must be finite")
        if torch.any((value != 0.0) & (value != 1.0)):
            raise ValueError("target must be binary")
        result = value.detach().cpu().to(dtype=torch.bool)
    else:
        raise TypeError("target must be bool or a floating binary tensor")
    return result.contiguous()


@dataclass(frozen=True, eq=False)
class NLCCSupervisedState:
    """One decoder input state plus binary output supervision.

    ``state_id`` and ``supervision_role`` exist only for receipt explanations.
    The key builders below never read either field.
    """

    state_id: str
    supervision_role: str
    feature: Tensor
    occupancy: Tensor
    target: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError("state_id must be a non-empty string")
        if (
            not isinstance(self.supervision_role, str)
            or not self.supervision_role
        ):
            raise ValueError("supervision_role must be a non-empty string")
        feature = _single_feature(self.feature)
        occupancy = _single_bool(self.occupancy, name="occupancy")
        target = _single_binary_target(self.target)
        valid = _single_bool(self.valid_mask, name="valid_mask")
        if target.shape != occupancy.shape or valid.shape != occupancy.shape:
            raise ValueError("occupancy, target, and valid_mask must align")
        if torch.any(target & ~valid):
            raise ValueError("positive supervision extends outside valid_mask")
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "valid_mask", valid)


def _validated_stride(
    state: NLCCSupervisedState,
    feature_stride: int,
) -> int:
    if (
        isinstance(feature_stride, bool)
        or not isinstance(feature_stride, int)
        or feature_stride < 1
    ):
        raise ValueError("feature_stride must be a positive integer")
    expected = (
        int(state.feature.shape[-2]) * feature_stride,
        int(state.feature.shape[-1]) * feature_stride,
    )
    actual = tuple(int(value) for value in state.occupancy.shape[-2:])
    if actual != expected:
        raise ValueError(
            "occupancy grid must equal feature grid times feature_stride"
        )
    return feature_stride


def _validated_position(
    state: NLCCSupervisedState,
    output_position: tuple[int, int],
) -> tuple[int, int]:
    if (
        not isinstance(output_position, tuple)
        or len(output_position) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in output_position
        )
    ):
        raise TypeError("output_position must be a pair of integers")
    row, column = output_position
    height, width = state.occupancy.shape[-2:]
    if not (0 <= row < height and 0 <= column < width):
        raise ValueError("output_position lies outside the output grid")
    if not bool(state.valid_mask[0, 0, row, column]):
        raise ValueError("output_position is not supervised")
    return row, column


def _exact_key(
    state: NLCCSupervisedState,
    output_position: tuple[int, int],
    *,
    local_count_field: Tensor | None = None,
) -> tuple[object, ...]:
    row, column = output_position
    count = (
        _nlcc_local_count_field(
            state.occupancy,
            feature_size=tuple(
                int(value) for value in state.feature.shape[-2:]
            ),
        )
        if local_count_field is None
        else local_count_field
    )
    return (
        tensor_content_fingerprint(state.feature),
        tensor_content_fingerprint(count),
        row,
        column,
    )


def _feature_role_tensor(feature: Tensor, *, include_sign: bool) -> Tensor:
    if include_sign:
        return torch.sign(feature).to(dtype=torch.int8).contiguous()
    return (feature != 0.0).to(dtype=torch.int8).contiguous()


def _relative_role_key(
    feature_role: Tensor,
    local_count_field: Tensor,
    *,
    output_position: tuple[int, int],
    feature_stride: int,
) -> tuple[object, ...]:
    feature_entries, count_entries = _absolute_role_entries(
        feature_role,
        local_count_field,
    )
    return _relative_role_key_from_entries(
        feature_entries,
        count_entries,
        output_position=output_position,
        feature_stride=feature_stride,
    )


def _nlcc_local_count_field(
    occupancy: Tensor,
    *,
    feature_size: tuple[int, int],
) -> Tensor:
    projected = project_occupancy_to_feature_grid(
        occupancy,
        feature_size,
    )
    kernel = torch.ones(
        1,
        1,
        NLCC_LOCAL_COUNT_KERNEL_SIZE,
        NLCC_LOCAL_COUNT_KERNEL_SIZE,
        dtype=torch.float32,
    )
    return F.conv2d(
        projected.to(dtype=torch.float32),
        kernel,
        padding=NLCC_LOCAL_COUNT_KERNEL_SIZE // 2,
    ).to(dtype=torch.int8).contiguous()


def _absolute_role_entries(
    feature_role: Tensor,
    local_count_field: Tensor,
) -> tuple[
    tuple[tuple[int, int, int, int], ...],
    tuple[tuple[int, int, int], ...],
]:
    feature_entries = tuple(
        (
            int(channel),
            int(row),
            int(column),
            int(feature_role[0, channel, row, column].item()),
        )
        for _, channel, row, column in torch.nonzero(
            feature_role,
            as_tuple=False,
        ).tolist()
    )
    count_entries = tuple(
        (
            int(row),
            int(column),
            int(local_count_field[0, 0, row, column].item()),
        )
        for _, _, row, column in torch.nonzero(
            local_count_field,
            as_tuple=False,
        ).tolist()
    )
    return feature_entries, count_entries


def _relative_role_key_from_entries(
    feature_entries: tuple[tuple[int, int, int, int], ...],
    count_entries: tuple[tuple[int, int, int], ...],
    *,
    output_position: tuple[int, int],
    feature_stride: int,
) -> tuple[object, ...]:
    output_row, output_column = output_position
    origin_row = output_row // feature_stride
    origin_column = output_column // feature_stride
    phase = (
        output_row % feature_stride,
        output_column % feature_stride,
    )
    relative_feature_entries = tuple(
        (
            int(channel),
            int(row) - origin_row,
            int(column) - origin_column,
            int(value),
        )
        for channel, row, column, value in feature_entries
    )
    relative_count_entries = tuple(
        (
            int(row) - origin_row,
            int(column) - origin_column,
            int(value),
        )
        for row, column, value in count_entries
    )
    return (
        phase,
        relative_feature_entries,
        relative_count_entries,
    )


def _spatial_transform(value: Tensor, operation: str) -> Tensor:
    if operation not in _D4_OPERATIONS:
        raise ValueError(f"unknown D4 operation {operation!r}")
    result = value
    if operation.startswith("mirror"):
        result = torch.flip(result, dims=(-1,))
    rotations = {
        "identity": 0,
        "rot90": 1,
        "rot180": 2,
        "rot270": 3,
        "mirror": 0,
        "mirror_rot90": 1,
        "mirror_rot180": 2,
        "mirror_rot270": 3,
    }[operation]
    if rotations:
        result = torch.rot90(result, rotations, dims=(-2, -1))
    return result.contiguous()


def _position_transform(
    position: tuple[int, int],
    *,
    size: tuple[int, int],
    operation: str,
) -> tuple[int, int]:
    height, width = size
    if height != width:
        raise ValueError("D4 diagnostic requires a square spatial grid")
    row, column = position
    if operation.startswith("mirror"):
        column = width - 1 - column
    rotations = {
        "identity": 0,
        "rot90": 1,
        "rot180": 2,
        "rot270": 3,
        "mirror": 0,
        "mirror_rot90": 1,
        "mirror_rot180": 2,
        "mirror_rot270": 3,
    }[operation]
    for _ in range(rotations):
        row, column = width - 1 - column, row
        height, width = width, height
    return row, column


def _role_key(
    state: NLCCSupervisedState,
    output_position: tuple[int, int],
    *,
    feature_stride: int,
    include_sign: bool,
) -> tuple[object, ...]:
    feature_role = _feature_role_tensor(
        state.feature,
        include_sign=include_sign,
    )
    local_count = _nlcc_local_count_field(
        state.occupancy,
        feature_size=tuple(
            int(value) for value in state.feature.shape[-2:]
        ),
    )
    return _relative_role_key(
        feature_role,
        local_count,
        output_position=output_position,
        feature_stride=feature_stride,
    )


def _d4_role_key(
    state: NLCCSupervisedState,
    output_position: tuple[int, int],
    *,
    feature_stride: int,
) -> tuple[object, ...]:
    candidates: list[tuple[object, ...]] = []
    output_size = tuple(int(value) for value in state.occupancy.shape[-2:])
    for operation in _D4_OPERATIONS:
        transformed_feature = _spatial_transform(state.feature, operation)
        transformed_occupancy = _spatial_transform(
            state.occupancy,
            operation,
        )
        transformed_position = _position_transform(
            output_position,
            size=output_size,
            operation=operation,
        )
        feature_role = _feature_role_tensor(
            transformed_feature,
            include_sign=False,
        )
        local_count = _nlcc_local_count_field(
            transformed_occupancy,
            feature_size=tuple(
                int(value) for value in transformed_feature.shape[-2:]
            ),
        )
        candidates.append(
            _relative_role_key(
                feature_role,
                local_count,
                output_position=transformed_position,
                feature_stride=feature_stride,
            )
        )
    return min(candidates)


def canonical_input_key_fingerprint(
    state: NLCCSupervisedState,
    output_position: tuple[int, int],
    *,
    feature_stride: int = FEATURE_STRIDE,
    quotient: str = "role",
) -> str:
    """Fingerprint one input/position key without reading supervision metadata.

    ``quotient`` may be ``exact``, ``signed_role``, ``role``, or
    ``d4_diagnostic``.
    """

    if not isinstance(state, NLCCSupervisedState):
        raise TypeError("state must be NLCCSupervisedState")
    stride = _validated_stride(state, feature_stride)
    position = _validated_position(state, output_position)
    if quotient == "exact":
        key = _exact_key(state, position)
    elif quotient == "signed_role":
        key = _role_key(
            state,
            position,
            feature_stride=stride,
            include_sign=True,
        )
    elif quotient == "role":
        key = _role_key(
            state,
            position,
            feature_stride=stride,
            include_sign=False,
        )
    elif quotient == "d4_diagnostic":
        key = _d4_role_key(
            state,
            position,
            feature_stride=stride,
        )
    else:
        raise ValueError(f"unknown quotient {quotient!r}")
    return stable_fingerprint(key)


@dataclass
class _CollisionBucket:
    label_counts: Counter[int] = field(default_factory=Counter)
    role_counts: dict[int, Counter[str]] = field(
        default_factory=lambda: {0: Counter(), 1: Counter()}
    )
    examples: dict[int, list[dict[str, object]]] = field(
        default_factory=lambda: {0: [], 1: []}
    )
    exact_key_fingerprints: set[str] = field(default_factory=set)


def _record(
    table: dict[tuple[object, ...], _CollisionBucket],
    *,
    key: tuple[object, ...],
    exact_key_fingerprint: str,
    state: NLCCSupervisedState,
    output_position: tuple[int, int],
    label: int,
    max_records_per_label: int,
) -> None:
    bucket = table.setdefault(key, _CollisionBucket())
    bucket.label_counts[label] += 1
    bucket.role_counts[label][state.supervision_role] += 1
    bucket.exact_key_fingerprints.add(exact_key_fingerprint)
    examples = bucket.examples[label]
    if len(examples) < max_records_per_label:
        examples.append(
            {
                "state_id": state.state_id,
                "supervision_role": state.supervision_role,
                "output_position": [
                    int(output_position[0]),
                    int(output_position[1]),
                ],
            }
        )


def _table_receipt(
    table: Mapping[tuple[object, ...], _CollisionBucket],
    *,
    max_examples: int,
) -> dict[str, object]:
    conflicts = tuple(
        (key, bucket)
        for key, bucket in table.items()
        if bucket.label_counts[0] and bucket.label_counts[1]
    )
    ranked = sorted(
        conflicts,
        key=lambda item: stable_fingerprint(item[0]),
    )
    examples: list[dict[str, object]] = []
    for key, bucket in ranked[:max_examples]:
        examples.append(
            {
                "key_fingerprint": stable_fingerprint(key),
                "negative": {
                    "record_count": int(bucket.label_counts[0]),
                    "supervision_role_counts": dict(
                        sorted(bucket.role_counts[0].items())
                    ),
                    "examples": bucket.examples[0],
                },
                "positive": {
                    "record_count": int(bucket.label_counts[1]),
                    "supervision_role_counts": dict(
                        sorted(bucket.role_counts[1].items())
                    ),
                    "examples": bucket.examples[1],
                },
            }
        )
    return {
        "key_count": len(table),
        "conflict_key_count": len(conflicts),
        "conflicting_record_count": sum(
            sum(bucket.label_counts.values()) for _, bucket in conflicts
        ),
        "negative_record_count_in_conflicts": sum(
            bucket.label_counts[0] for _, bucket in conflicts
        ),
        "positive_record_count_in_conflicts": sum(
            bucket.label_counts[1] for _, bucket in conflicts
        ),
        "collapsed_exact_key_class_count": sum(
            len(bucket.exact_key_fingerprints) > 1
            for bucket in table.values()
        ),
        "collapsed_exact_key_excess": sum(
            max(0, len(bucket.exact_key_fingerprints) - 1)
            for bucket in table.values()
        ),
        "examples": examples,
    }


def _audited_tensor_fingerprint(
    states: tuple[NLCCSupervisedState, ...],
) -> str:
    return stable_fingerprint(
        {
            "states": [
                {
                    "state_id": state.state_id,
                    "supervision_role": state.supervision_role,
                    "feature": tensor_content_fingerprint(state.feature),
                    "occupancy": tensor_content_fingerprint(
                        state.occupancy
                    ),
                    "target": tensor_content_fingerprint(state.target),
                    "valid_mask": tensor_content_fingerprint(
                        state.valid_mask
                    ),
                }
                for state in states
            ]
        }
    )


def audit_nlcc_role_quotient(
    states: Iterable[NLCCSupervisedState],
    *,
    input_fingerprint: str,
    feature_stride: int = FEATURE_STRIDE,
    max_examples: int = 8,
    max_records_per_label: int = 3,
    include_d4_diagnostic: bool = False,
) -> dict[str, object]:
    """Return a deterministic, JSON-compatible identifiability receipt."""

    values = tuple(states)
    if not values or any(
        not isinstance(value, NLCCSupervisedState) for value in values
    ):
        raise ValueError(
            "states must be a nonempty NLCCSupervisedState collection"
        )
    if len({value.state_id for value in values}) != len(values):
        raise ValueError("state_id values must be unique")
    if not isinstance(input_fingerprint, str) or not input_fingerprint:
        raise ValueError("input_fingerprint must be non-empty")
    if not isinstance(include_d4_diagnostic, bool):
        raise TypeError("include_d4_diagnostic must be bool")
    for name, value in (
        ("max_examples", max_examples),
        ("max_records_per_label", max_records_per_label),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for state in values:
        _validated_stride(state, feature_stride)

    exact: dict[tuple[object, ...], _CollisionBucket] = {}
    signed: dict[tuple[object, ...], _CollisionBucket] = {}
    role: dict[tuple[object, ...], _CollisionBucket] = {}
    d4: dict[tuple[object, ...], _CollisionBucket] | None = (
        {} if include_d4_diagnostic else None
    )
    supervised_record_count = 0
    positive_record_count = 0
    supervision_role_counts: Counter[str] = Counter()

    for state in values:
        feature_role_signed = _feature_role_tensor(
            state.feature,
            include_sign=True,
        )
        feature_role_unsigned = _feature_role_tensor(
            state.feature,
            include_sign=False,
        )
        local_count = _nlcc_local_count_field(
            state.occupancy,
            feature_size=tuple(
                int(value) for value in state.feature.shape[-2:]
            ),
        )
        signed_entries = _absolute_role_entries(
            feature_role_signed,
            local_count,
        )
        unsigned_entries = _absolute_role_entries(
            feature_role_unsigned,
            local_count,
        )
        d4_values = tuple(
            (
                _absolute_role_entries(
                    _feature_role_tensor(
                        _spatial_transform(state.feature, operation),
                        include_sign=False,
                    ),
                    _nlcc_local_count_field(
                        _spatial_transform(state.occupancy, operation),
                        feature_size=tuple(
                            int(value)
                            for value in _spatial_transform(
                                state.feature,
                                operation,
                            ).shape[-2:]
                        ),
                    ),
                ),
                operation,
            )
            for operation in _D4_OPERATIONS
        ) if include_d4_diagnostic else ()
        for row, column in torch.nonzero(
            state.valid_mask[0, 0],
            as_tuple=False,
        ).tolist():
            position = (int(row), int(column))
            label = int(state.target[0, 0, row, column].item())
            exact_key = _exact_key(
                state,
                position,
                local_count_field=local_count,
            )
            exact_fingerprint = stable_fingerprint(exact_key)
            signed_key = _relative_role_key_from_entries(
                *signed_entries,
                output_position=position,
                feature_stride=feature_stride,
            )
            role_key = _relative_role_key_from_entries(
                *unsigned_entries,
                output_position=position,
                feature_stride=feature_stride,
            )
            tables_and_keys: list[
                tuple[
                    dict[tuple[object, ...], _CollisionBucket],
                    tuple[object, ...],
                ]
            ] = [
                (exact, exact_key),
                (signed, signed_key),
                (role, role_key),
            ]
            if d4 is not None:
                output_size = tuple(
                    int(value) for value in state.occupancy.shape[-2:]
                )
                d4_key = min(
                    _relative_role_key_from_entries(
                        *transformed_entries,
                        output_position=_position_transform(
                            position,
                            size=output_size,
                            operation=operation,
                        ),
                        feature_stride=feature_stride,
                    )
                    for (
                        transformed_entries,
                        operation,
                    ) in d4_values
                )
                tables_and_keys.append((d4, d4_key))
            for table, key in tables_and_keys:
                _record(
                    table,
                    key=key,
                    exact_key_fingerprint=exact_fingerprint,
                    state=state,
                    output_position=position,
                    label=label,
                    max_records_per_label=max_records_per_label,
                )
            supervised_record_count += 1
            positive_record_count += label
            supervision_role_counts[state.supervision_role] += 1

    exact_receipt = _table_receipt(exact, max_examples=max_examples)
    signed_receipt = _table_receipt(signed, max_examples=max_examples)
    role_receipt = _table_receipt(role, max_examples=max_examples)
    d4_receipt: dict[str, object]
    if d4 is None:
        d4_receipt = {
            "status": "not_evaluated",
            "reason": (
                "D4 is an optional non-gating diagnostic because NLCC "
                "PixelShuffle phase channels are not rotation/flip tied"
            ),
        }
    else:
        d4_receipt = {
            "status": "evaluated",
            **_table_receipt(d4, max_examples=max_examples),
        }
    hard_gate_pass = exact_receipt["conflict_key_count"] == 0
    role_gate_pass = role_receipt["conflict_key_count"] == 0
    payload: dict[str, object] = {
        "schema_version": ROLE_QUOTIENT_ALGORITHM_VERSION,
        "algorithm": {
            "hard_key": (
                "sha256(feature tensor), sha256(full NLCC local 3x3 count "
                "field), "
                "raw output coordinate"
            ),
            "signed_amplitude_quotient_key": (
                "signed nonzero channel support, output-relative full "
                "support, output-relative full local count field, phase"
            ),
            "role_quotient_key": (
                "unsigned nonzero channel-role support, output-relative "
                "full support, output-relative full local count field, phase"
            ),
            "d4_diagnostic_key": (
                "lexicographic minimum after jointly transforming feature, "
                "occupancy, and output coordinate over D4"
            ),
            "model_input_fields": ["feature", "occupancy"],
            "supervision_fields": ["target", "valid_mask"],
            "metadata_excluded_from_keys": [
                "state_id",
                "supervision_role",
                "group_id",
                "sample_id",
                "match_id",
                "anchor_role",
                "endpoint_role",
            ],
            "feature_stride": feature_stride,
            "local_count_kernel_size": NLCC_LOCAL_COUNT_KERNEL_SIZE,
            "d4_is_development_gate": False,
            "d4_equivariance_claimed": False,
            "d4_requested": include_d4_diagnostic,
        },
        "input_fingerprint": input_fingerprint,
        "audited_tensor_fingerprint": _audited_tensor_fingerprint(values),
        "population": {
            "state_count": len(values),
            "supervised_record_count": supervised_record_count,
            "positive_record_count": positive_record_count,
            "negative_record_count": (
                supervised_record_count - positive_record_count
            ),
            "supervision_role_record_counts": dict(
                sorted(supervision_role_counts.items())
            ),
        },
        "exact_tensor": exact_receipt,
        "signed_amplitude_quotient": signed_receipt,
        "role_quotient": role_receipt,
        "d4_joint_transform_diagnostic": d4_receipt,
        "decision": {
            "hard_gate_pass": hard_gate_pass,
            "role_gate_pass": role_gate_pass,
            "development_authorized": (
                hard_gate_pass and role_gate_pass
            ),
            "d4_diagnostic_affects_authorization": False,
            "hard_gate_meaning": (
                "false means identical feature/local-count fields at one "
                "output coordinate require opposite supervision"
            ),
            "role_gate_meaning": (
                "false means identifiability relies on deterministic "
                "feature amplitude/sign or absolute spatial identity"
            ),
        },
    }
    payload["receipt_fingerprint"] = stable_fingerprint(payload)
    return payload


def materialize_nlcc_supervised_states(
    profile: NLCCInputProfile,
    specs: Iterable[NLCCPairSpec] | None = None,
) -> tuple[NLCCSupervisedState, ...]:
    """Reuse a frozen input profile and expose only decoder states + truth."""

    if not isinstance(profile, NLCCInputProfile):
        raise TypeError("profile must be NLCCInputProfile")
    values = build_pair_specs(profile) if specs is None else tuple(specs)
    if not values:
        raise ValueError("specs cannot be empty")
    outcome = build_outcome_batch(profile, values, device="cpu")
    pair = outcome.pair_batch
    states: list[NLCCSupervisedState] = []
    state_index = 0
    for row_index in range(len(values)):
        feature = pair.feature[row_index : row_index + 1]
        for occupancy, target, role in (
            (
                pair.occupancy_plus[row_index : row_index + 1],
                outcome.completion_plus[row_index : row_index + 1],
                "paired_absolute_endpoint_0",
            ),
            (
                pair.occupancy_minus[row_index : row_index + 1],
                outcome.completion_minus[row_index : row_index + 1],
                "paired_absolute_endpoint_1",
            ),
        ):
            valid = (
                pair.image_valid_mask[row_index : row_index + 1]
                & ~occupancy
            )
            states.append(
                NLCCSupervisedState(
                    state_id=f"state-{state_index:04d}",
                    supervision_role=role,
                    feature=feature,
                    occupancy=occupancy,
                    target=target,
                    valid_mask=valid,
                )
            )
            state_index += 1

    factual = build_factual_population(profile, device="cpu")
    for branch, role in (
        ("factual_miss", "factual_absolute_positive"),
        ("factual_no_miss", "factual_absolute_null"),
    ):
        batch = factual[branch]
        for row_index in range(int(batch.feature.shape[0])):
            states.append(
                NLCCSupervisedState(
                    state_id=f"state-{state_index:04d}",
                    supervision_role=role,
                    feature=batch.feature[row_index : row_index + 1],
                    occupancy=batch.occupancy[row_index : row_index + 1],
                    target=batch.target[row_index : row_index + 1],
                    valid_mask=(
                        batch.valid_mask[row_index : row_index + 1]
                        & ~batch.occupancy[row_index : row_index + 1]
                    ),
                )
            )
            state_index += 1
    return tuple(states)


def build_nlcc_development_role_quotient_receipt(
    *,
    max_examples: int = 8,
    max_records_per_label: int = 3,
    include_d4_diagnostic: bool = False,
) -> dict[str, object]:
    """Audit the byte-frozen v12 Development inputs without regenerating v13."""

    specs = build_pair_specs(DEVELOPMENT_PROFILE)
    states = materialize_nlcc_supervised_states(
        DEVELOPMENT_PROFILE,
        specs,
    )
    return audit_nlcc_role_quotient(
        states,
        input_fingerprint=nlcc_development_fingerprint(specs),
        feature_stride=FEATURE_STRIDE,
        max_examples=max_examples,
        max_records_per_label=max_records_per_label,
        include_d4_diagnostic=include_d4_diagnostic,
    )


__all__ = [
    "NLCCSupervisedState",
    "ROLE_QUOTIENT_ALGORITHM_VERSION",
    "audit_nlcc_role_quotient",
    "build_nlcc_development_role_quotient_receipt",
    "canonical_input_key_fingerprint",
    "materialize_nlcc_supervised_states",
]
