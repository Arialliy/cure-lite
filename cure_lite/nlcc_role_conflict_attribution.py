"""Factorial attribution of NLCC-v12 role-equivalence conflicts.

The frozen R13-1 audit establishes that exact effective tensors do not require
opposite labels, while a role quotient does.  That quotient removes feature
values, feature signs, and absolute spatial origin together, so its result
cannot identify which discarded factor carries the distinction.

This module performs the complete 2 x 2 x 2 quotient lattice:

``magnitude removed``
    preserve float32 magnitudes or reduce each non-zero value to unit
    magnitude.

``sign removed``
    preserve or remove feature sign.

``spatial origin mode``
    absolute coordinates or coordinates relative to the supervised output
    cell (while retaining PixelShuffle phase).

Only decoder-accessible feature and occupancy-derived count tensors enter the
keys.  Frozen population metadata is attached after grouping solely to
explain which generated states participate in a conflict.  It never affects
key construction or any gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor

from .cache.schema import stable_fingerprint
from .nlcc_dataset_free_inputs import (
    FEATURE_STRIDE,
    FACTUAL_POPULATION_SIZE,
    NLCCInputProfile,
    NLCCPairSpec,
    build_pair_specs,
)
from .nlcc_development_inputs import (
    DEVELOPMENT_PROFILE,
    nlcc_development_fingerprint,
)
from .nlcc_role_quotient_audit import (
    NLCCSupervisedState,
    _nlcc_local_count_field,
    materialize_nlcc_supervised_states,
)


ROLE_CONFLICT_ATTRIBUTION_ALGORITHM_VERSION = (
    "cure-lite.nlcc-role-conflict-attribution.v1"
)

_FEATURE_VALUE_MODES = (
    "exact",
    "absolute_value",
    "signed_support",
    "unsigned_support",
)
_ORIGIN_MODES = ("absolute", "relative")
_FACTOR_IDS = tuple(
    f"{feature_mode}__{origin_mode}"
    for feature_mode in _FEATURE_VALUE_MODES
    for origin_mode in _ORIGIN_MODES
)


@dataclass(frozen=True)
class _Record:
    state_index: int
    state_id: str
    supervision_role: str
    output_position: tuple[int, int]
    label: int

    @property
    def record_id(self) -> str:
        return (
            f"{self.state_id}@{self.output_position[0]:04d},"
            f"{self.output_position[1]:04d}"
        )


@dataclass(frozen=True)
class _StateContext:
    state: NLCCSupervisedState
    feature_shape: tuple[int, ...]
    count_shape: tuple[int, ...]
    feature_entries: Mapping[
        str,
        tuple[tuple[int, int, int, int | float], ...],
    ]
    count_entries: tuple[tuple[int, int, int], ...]


def _validated_feature_stride(feature_stride: int) -> int:
    if (
        isinstance(feature_stride, bool)
        or not isinstance(feature_stride, int)
        or feature_stride < 1
    ):
        raise ValueError("feature_stride must be a positive integer")
    return feature_stride


def _feature_entries(
    feature: Tensor,
    *,
    value_mode: str,
) -> tuple[tuple[int, int, int, int | float], ...]:
    if value_mode not in _FEATURE_VALUE_MODES:
        raise ValueError(f"unknown feature value mode {value_mode!r}")
    entries: list[tuple[int, int, int, int | float]] = []
    for _, channel, row, column in torch.nonzero(
        feature,
        as_tuple=False,
    ).tolist():
        exact_value = float(feature[0, channel, row, column].item())
        if value_mode == "exact":
            value: int | float = exact_value
        elif value_mode == "absolute_value":
            value = abs(exact_value)
        elif value_mode == "signed_support":
            value = 1 if exact_value > 0.0 else -1
        else:
            value = 1
        entries.append(
            (
                int(channel),
                int(row),
                int(column),
                value,
            )
        )
    return tuple(entries)


def _count_entries(
    count: Tensor,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            int(row),
            int(column),
            int(count[0, 0, row, column].item()),
        )
        for _, _, row, column in torch.nonzero(
            count,
            as_tuple=False,
        ).tolist()
    )


def _state_context(state: NLCCSupervisedState) -> _StateContext:
    count = _nlcc_local_count_field(
        state.occupancy,
        feature_size=tuple(
            int(value) for value in state.feature.shape[-2:]
        ),
    )
    return _StateContext(
        state=state,
        feature_shape=tuple(int(value) for value in state.feature.shape),
        count_shape=tuple(int(value) for value in count.shape),
        feature_entries={
            value_mode: _feature_entries(
                state.feature,
                value_mode=value_mode,
            )
            for value_mode in _FEATURE_VALUE_MODES
        },
        count_entries=_count_entries(count),
    )


def _factor_key(
    context: _StateContext,
    output_position: tuple[int, int],
    *,
    feature_value_mode: str,
    origin_mode: str,
    feature_stride: int,
) -> tuple[object, ...]:
    if feature_value_mode not in _FEATURE_VALUE_MODES:
        raise ValueError(
            f"unknown feature value mode {feature_value_mode!r}"
        )
    if origin_mode not in _ORIGIN_MODES:
        raise ValueError(f"unknown origin mode {origin_mode!r}")
    row, column = output_position
    feature_entries = context.feature_entries[feature_value_mode]
    if origin_mode == "absolute":
        return (
            context.feature_shape,
            context.count_shape,
            (row, column),
            feature_entries,
            context.count_entries,
        )

    origin_row = row // feature_stride
    origin_column = column // feature_stride
    phase = (row % feature_stride, column % feature_stride)
    relative_feature = tuple(
        (
            channel,
            feature_row - origin_row,
            feature_column - origin_column,
            value,
        )
        for channel, feature_row, feature_column, value in feature_entries
    )
    relative_count = tuple(
        (
            count_row - origin_row,
            count_column - origin_column,
            value,
        )
        for count_row, count_column, value in context.count_entries
    )
    return (
        context.feature_shape,
        context.count_shape,
        phase,
        relative_feature,
        relative_count,
    )


def factorial_input_key_fingerprint(
    state: NLCCSupervisedState,
    output_position: tuple[int, int],
    *,
    feature_value_mode: str,
    origin_mode: str,
    feature_stride: int = FEATURE_STRIDE,
) -> str:
    """Fingerprint one factorial key without consulting population metadata."""

    if not isinstance(state, NLCCSupervisedState):
        raise TypeError("state must be NLCCSupervisedState")
    stride = _validated_feature_stride(feature_stride)
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
    output_height, output_width = state.occupancy.shape[-2:]
    if not (0 <= row < output_height and 0 <= column < output_width):
        raise ValueError("output_position lies outside the output grid")
    if not bool(state.valid_mask[0, 0, row, column]):
        raise ValueError("output_position is not supervised")
    expected_output = (
        int(state.feature.shape[-2]) * stride,
        int(state.feature.shape[-1]) * stride,
    )
    if tuple(int(value) for value in state.occupancy.shape[-2:]) != (
        expected_output
    ):
        raise ValueError(
            "occupancy grid must equal feature grid times feature_stride"
        )
    return stable_fingerprint(
        _factor_key(
            _state_context(state),
            (row, column),
            feature_value_mode=feature_value_mode,
            origin_mode=origin_mode,
            feature_stride=stride,
        )
    )


def _factor_id(
    feature_value_mode: str,
    origin_mode: str,
) -> str:
    return f"{feature_value_mode}__{origin_mode}"


def _records(
    states: tuple[NLCCSupervisedState, ...],
) -> tuple[_Record, ...]:
    result: list[_Record] = []
    for state_index, state in enumerate(states):
        for row, column in torch.nonzero(
            state.valid_mask[0, 0],
            as_tuple=False,
        ).tolist():
            result.append(
                _Record(
                    state_index=state_index,
                    state_id=state.state_id,
                    supervision_role=state.supervision_role,
                    output_position=(int(row), int(column)),
                    label=int(state.target[0, 0, row, column].item()),
                )
            )
    return tuple(result)


def _record_provenance(
    record: _Record,
    *,
    specs: tuple[NLCCPairSpec, ...] | None,
) -> dict[str, object]:
    base: dict[str, object] = {
        "record_id": record.record_id,
        "state_id": record.state_id,
        "supervision_role": record.supervision_role,
        "output_position": list(record.output_position),
        "label": record.label,
    }
    if specs is None:
        base["population_metadata"] = "not_available"
        return base

    paired_state_count = 2 * len(specs)
    if record.state_index < paired_state_count:
        spec_index, endpoint_index = divmod(record.state_index, 2)
        spec = specs[spec_index]
        base["population_metadata"] = {
            "population_kind": "paired",
            "population_index": spec.population_index,
            "group_id": spec.group_id,
            "pair_kind": spec.pair_kind,
            "geometry_family": spec.geometry_family,
            "response_pixel_count": spec.response_pixel_count,
            "anchor_role": spec.anchor_role,
            "endpoint_role": (
                "occupancy_plus" if endpoint_index == 0
                else "occupancy_minus"
            ),
            "component_cell": list(spec.component_cell),
            "response_cell": list(spec.response_cell),
            "fixed_occupancy_cells": [
                list(cell) for cell in spec.fixed_occupancy_cells
            ],
        }
        return base

    factual_offset = record.state_index - paired_state_count
    if factual_offset < FACTUAL_POPULATION_SIZE:
        branch = "factual_miss"
        factual_index = factual_offset
        response_pixel_count = 1 if factual_index % 2 == 0 else 3
    else:
        branch = "factual_no_miss"
        factual_index = factual_offset - FACTUAL_POPULATION_SIZE
        response_pixel_count = 0
    base["population_metadata"] = {
        "population_kind": "factual",
        "branch": branch,
        "factual_index": factual_index,
        "response_pixel_count": response_pixel_count,
    }
    return base


def _semantic_pair_category(
    negative: _Record,
    positive: _Record,
    *,
    specs: tuple[NLCCPairSpec, ...] | None,
) -> str:
    if specs is None:
        return "metadata_not_available"
    paired_state_count = 2 * len(specs)
    negative_paired = negative.state_index < paired_state_count
    positive_paired = positive.state_index < paired_state_count
    if negative_paired != positive_paired:
        return "paired_factual_role_alias"
    if not negative_paired:
        negative_branch_offset = negative.state_index - paired_state_count
        positive_branch_offset = positive.state_index - paired_state_count
        negative_branch = (
            "factual_miss"
            if negative_branch_offset < FACTUAL_POPULATION_SIZE
            else "factual_no_miss"
        )
        positive_branch = (
            "factual_miss"
            if positive_branch_offset < FACTUAL_POPULATION_SIZE
            else "factual_no_miss"
        )
        if negative_branch != positive_branch:
            return "factual_branch_alias"
        if negative_branch == "factual_miss":
            negative_extent = (
                1 if negative_branch_offset % 2 == 0 else 3
            )
            positive_extent = (
                1 if positive_branch_offset % 2 == 0 else 3
            )
            if negative_extent != positive_extent:
                return "factual_response_extent_alias"
        return "factual_within_branch_alias"

    negative_spec = specs[negative.state_index // 2]
    positive_spec = specs[positive.state_index // 2]
    if (
        negative_spec.response_pixel_count
        != positive_spec.response_pixel_count
    ):
        return "paired_response_extent_alias"
    if negative_spec.geometry_family != positive_spec.geometry_family:
        return "paired_coverage_geometry_alias"
    if negative.state_index % 2 != positive.state_index % 2:
        return "paired_endpoint_alias"
    if negative_spec.anchor_role != positive_spec.anchor_role:
        return "paired_anchor_witness_alias"
    return "paired_within_family_alias"


def _factor_receipt(
    counts: Mapping[tuple[object, ...], tuple[int, int]],
    conflict_records: Mapping[
        tuple[object, ...],
        tuple[_Record, ...],
    ],
    *,
    specs: tuple[NLCCPairSpec, ...] | None,
    max_examples: int,
    max_records_per_label: int,
) -> tuple[dict[str, object], frozenset[tuple[str, str]]]:
    conflict_keys = tuple(
        key
        for key, label_counts in counts.items()
        if label_counts[0] and label_counts[1]
    )
    ranked_keys = sorted(conflict_keys, key=stable_fingerprint)
    examples: list[dict[str, object]] = []
    opposing_pairs: set[tuple[str, str]] = set()
    semantic_counts: Counter[str] = Counter()

    for key in ranked_keys:
        records = conflict_records[key]
        negative_records = tuple(
            record for record in records if record.label == 0
        )
        positive_records = tuple(
            record for record in records if record.label == 1
        )
        for negative in negative_records:
            for positive in positive_records:
                pair = (negative.record_id, positive.record_id)
                opposing_pairs.add(pair)
                semantic_counts[
                    _semantic_pair_category(
                        negative,
                        positive,
                        specs=specs,
                    )
                ] += 1
        if len(examples) < max_examples:
            examples.append(
                {
                    "key_fingerprint": stable_fingerprint(key),
                    "negative_record_count": len(negative_records),
                    "positive_record_count": len(positive_records),
                    "negative_examples": [
                        _record_provenance(record, specs=specs)
                        for record in negative_records[
                            :max_records_per_label
                        ]
                    ],
                    "positive_examples": [
                        _record_provenance(record, specs=specs)
                        for record in positive_records[
                            :max_records_per_label
                        ]
                    ],
                }
            )

    conflicting_record_ids = {
        record.record_id
        for key in conflict_keys
        for record in conflict_records[key]
    }
    negative_count = sum(counts[key][0] for key in conflict_keys)
    positive_count = sum(counts[key][1] for key in conflict_keys)
    receipt = {
        "key_count": len(counts),
        "conflict_key_count": len(conflict_keys),
        "conflicting_record_count": len(conflicting_record_ids),
        "negative_record_count_in_conflicts": negative_count,
        "positive_record_count_in_conflicts": positive_count,
        "opposing_record_pair_count": len(opposing_pairs),
        "opposing_record_pair_fingerprint": stable_fingerprint(
            sorted(opposing_pairs)
        ),
        "semantic_pair_category_counts": dict(
            sorted(semantic_counts.items())
        ),
        "examples": examples,
    }
    return receipt, frozenset(opposing_pairs)


def _coarsening_delta(
    finer: frozenset[tuple[str, str]],
    coarser: frozenset[tuple[str, str]],
    *,
    comparison_id: str,
) -> dict[str, object]:
    if not finer.issubset(coarser):
        raise AssertionError(
            f"{comparison_id} is not a valid quotient coarsening"
        )
    introduced = sorted(coarser - finer)
    return {
        "comparison_id": comparison_id,
        "finer_opposing_pair_count": len(finer),
        "coarser_opposing_pair_count": len(coarser),
        "new_opposing_pair_count": len(introduced),
        "new_opposing_pair_fingerprint": stable_fingerprint(introduced),
    }


def audit_nlcc_role_conflict_attribution(
    states: Iterable[NLCCSupervisedState],
    *,
    input_fingerprint: str,
    specs: Iterable[NLCCPairSpec] | None = None,
    feature_stride: int = FEATURE_STRIDE,
    max_examples: int = 8,
    max_records_per_label: int = 3,
) -> dict[str, object]:
    """Return a deterministic factorial attribution receipt."""

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
    stride = _validated_feature_stride(feature_stride)
    for name, value in (
        ("max_examples", max_examples),
        ("max_records_per_label", max_records_per_label),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    spec_values = None if specs is None else tuple(specs)
    if spec_values is not None:
        if not spec_values or any(
            not isinstance(value, NLCCPairSpec) for value in spec_values
        ):
            raise ValueError("specs must contain NLCCPairSpec values")
        expected_state_count = (
            2 * len(spec_values) + 2 * FACTUAL_POPULATION_SIZE
        )
        if len(values) != expected_state_count:
            raise ValueError(
                "state count does not match paired and factual populations"
            )

    contexts = tuple(_state_context(state) for state in values)
    for context in contexts:
        expected_output = (
            int(context.state.feature.shape[-2]) * stride,
            int(context.state.feature.shape[-1]) * stride,
        )
        if tuple(
            int(value)
            for value in context.state.occupancy.shape[-2:]
        ) != expected_output:
            raise ValueError(
                "occupancy grid must equal feature grid times "
                "feature_stride"
            )

    records = _records(values)
    factor_counts: dict[
        str,
        dict[tuple[object, ...], list[int]],
    ] = {factor_id: {} for factor_id in _FACTOR_IDS}
    for record in records:
        context = contexts[record.state_index]
        for feature_mode in _FEATURE_VALUE_MODES:
            for origin_mode in _ORIGIN_MODES:
                factor_id = _factor_id(feature_mode, origin_mode)
                key = _factor_key(
                    context,
                    record.output_position,
                    feature_value_mode=feature_mode,
                    origin_mode=origin_mode,
                    feature_stride=stride,
                )
                counts = factor_counts[factor_id].setdefault(key, [0, 0])
                counts[record.label] += 1

    frozen_counts: dict[
        str,
        dict[tuple[object, ...], tuple[int, int]],
    ] = {
        factor_id: {
            key: (counts[0], counts[1])
            for key, counts in table.items()
        }
        for factor_id, table in factor_counts.items()
    }
    conflict_key_sets = {
        factor_id: {
            key
            for key, counts in table.items()
            if counts[0] and counts[1]
        }
        for factor_id, table in frozen_counts.items()
    }
    conflict_records: dict[
        str,
        dict[tuple[object, ...], list[_Record]],
    ] = {
        factor_id: {key: [] for key in conflict_keys}
        for factor_id, conflict_keys in conflict_key_sets.items()
    }
    for record in records:
        context = contexts[record.state_index]
        for feature_mode in _FEATURE_VALUE_MODES:
            for origin_mode in _ORIGIN_MODES:
                factor_id = _factor_id(feature_mode, origin_mode)
                key = _factor_key(
                    context,
                    record.output_position,
                    feature_value_mode=feature_mode,
                    origin_mode=origin_mode,
                    feature_stride=stride,
                )
                if key in conflict_key_sets[factor_id]:
                    conflict_records[factor_id][key].append(record)

    factor_receipts: dict[str, dict[str, object]] = {}
    factor_pairs: dict[str, frozenset[tuple[str, str]]] = {}
    for factor_id in _FACTOR_IDS:
        receipt, pairs = _factor_receipt(
            frozen_counts[factor_id],
            {
                key: tuple(records_for_key)
                for key, records_for_key in conflict_records[
                    factor_id
                ].items()
            },
            specs=spec_values,
            max_examples=max_examples,
            max_records_per_label=max_records_per_label,
        )
        feature_mode, origin_mode = factor_id.split("__", maxsplit=1)
        receipt["feature_value_mode"] = feature_mode
        receipt["origin_mode"] = origin_mode
        factor_receipts[factor_id] = receipt
        factor_pairs[factor_id] = pairs

    comparisons = {
        "remove_numerical_values_at_absolute_origin": _coarsening_delta(
            factor_pairs["exact__absolute"],
            factor_pairs["signed_support__absolute"],
            comparison_id=(
                "exact__absolute -> signed_support__absolute"
            ),
        ),
        "remove_sign_at_absolute_origin": _coarsening_delta(
            factor_pairs["exact__absolute"],
            factor_pairs["absolute_value__absolute"],
            comparison_id=(
                "exact__absolute -> absolute_value__absolute"
            ),
        ),
        "remove_numerical_values_after_sign_quotient_at_absolute_origin": (
            _coarsening_delta(
                factor_pairs["absolute_value__absolute"],
                factor_pairs["unsigned_support__absolute"],
                comparison_id=(
                    "absolute_value__absolute -> "
                    "unsigned_support__absolute"
                ),
            )
        ),
        "remove_sign_after_value_quotient_at_absolute_origin": (
            _coarsening_delta(
            factor_pairs["signed_support__absolute"],
            factor_pairs["unsigned_support__absolute"],
            comparison_id=(
                "signed_support__absolute -> "
                "unsigned_support__absolute"
            ),
            )
        ),
        "remove_absolute_origin_with_exact_values": _coarsening_delta(
            factor_pairs["exact__absolute"],
            factor_pairs["exact__relative"],
            comparison_id="exact__absolute -> exact__relative",
        ),
        "remove_absolute_origin_with_absolute_values": _coarsening_delta(
            factor_pairs["absolute_value__absolute"],
            factor_pairs["absolute_value__relative"],
            comparison_id=(
                "absolute_value__absolute -> absolute_value__relative"
            ),
        ),
        "remove_absolute_origin_with_signed_support": _coarsening_delta(
            factor_pairs["signed_support__absolute"],
            factor_pairs["signed_support__relative"],
            comparison_id=(
                "signed_support__absolute -> "
                "signed_support__relative"
            ),
        ),
        "remove_absolute_origin_with_unsigned_support": _coarsening_delta(
            factor_pairs["unsigned_support__absolute"],
            factor_pairs["unsigned_support__relative"],
            comparison_id=(
                "unsigned_support__absolute -> "
                "unsigned_support__relative"
            ),
        ),
        "remove_numerical_values_after_origin_quotient": (
            _coarsening_delta(
                factor_pairs["exact__relative"],
                factor_pairs["signed_support__relative"],
                comparison_id=(
                    "exact__relative -> signed_support__relative"
                ),
            )
        ),
        "remove_sign_after_origin_quotient": _coarsening_delta(
            factor_pairs["exact__relative"],
            factor_pairs["absolute_value__relative"],
            comparison_id=(
                "exact__relative -> absolute_value__relative"
            ),
        ),
        "remove_numerical_values_after_sign_and_origin_quotients": (
            _coarsening_delta(
                factor_pairs["absolute_value__relative"],
                factor_pairs["unsigned_support__relative"],
                comparison_id=(
                    "absolute_value__relative -> "
                    "unsigned_support__relative"
                ),
            )
        ),
        "remove_sign_after_value_and_origin_quotients": (
            _coarsening_delta(
                factor_pairs["signed_support__relative"],
                factor_pairs["unsigned_support__relative"],
                comparison_id=(
                    "signed_support__relative -> "
                    "unsigned_support__relative"
                ),
            )
        ),
    }

    payload: dict[str, object] = {
        "schema_version": ROLE_CONFLICT_ATTRIBUTION_ALGORITHM_VERSION,
        "algorithm": {
            "factorial_axes": {
                "magnitude_removed": [False, True],
                "sign_removed": [False, True],
                "origin_mode": list(_ORIGIN_MODES),
            },
            "feature_value_modes": {
                "exact": {
                    "magnitude_removed": False,
                    "sign_removed": False,
                },
                "absolute_value": {
                    "magnitude_removed": False,
                    "sign_removed": True,
                },
                "signed_support": {
                    "magnitude_removed": True,
                    "sign_removed": False,
                },
                "unsigned_support": {
                    "magnitude_removed": True,
                    "sign_removed": True,
                },
            },
            "key_inputs": [
                "feature tensor",
                "occupancy-derived full NLCC local count field",
                "supervised output coordinate or output-relative phase",
            ],
            "metadata_excluded_from_keys": [
                "state_id",
                "supervision_role",
                "group_id",
                "sample_id",
                "match_id",
                "anchor_role",
                "endpoint_role",
                "geometry_family",
                "response_pixel_count",
            ],
            "metadata_use": (
                "post-grouping explanation only; never an input key or gate"
            ),
            "feature_stride": stride,
            "comparison_unit": (
                "unordered opposite-label supervised-record pair"
            ),
        },
        "input_fingerprint": input_fingerprint,
        "population": {
            "state_count": len(values),
            "supervised_record_count": len(records),
            "positive_record_count": sum(
                record.label for record in records
            ),
            "negative_record_count": sum(
                1 - record.label for record in records
            ),
        },
        "factors": factor_receipts,
        "coarsening_comparisons": comparisons,
        "decision": {
            "factorial_attribution_complete": True,
            "exact_effective_input_conflict_free": (
                factor_receipts["exact__absolute"][
                    "conflict_key_count"
                ]
                == 0
            ),
            "exact_values_translation_quotient_conflict_free": (
                factor_receipts["exact__relative"][
                    "conflict_key_count"
                ]
                == 0
            ),
            "absolute_values_absolute_conflict_free": (
                factor_receipts["absolute_value__absolute"][
                    "conflict_key_count"
                ]
                == 0
            ),
            "absolute_values_translation_quotient_conflict_free": (
                factor_receipts["absolute_value__relative"][
                    "conflict_key_count"
                ]
                == 0
            ),
            "signed_support_absolute_conflict_free": (
                factor_receipts["signed_support__absolute"][
                    "conflict_key_count"
                ]
                == 0
            ),
            "unsigned_support_absolute_conflict_free": (
                factor_receipts["unsigned_support__absolute"][
                    "conflict_key_count"
                ]
                == 0
            ),
            "role_quotient_conflict_free": (
                factor_receipts["unsigned_support__relative"][
                    "conflict_key_count"
                ]
                == 0
            ),
            "training_authorized": False,
            "reason": (
                "this receipt attributes the frozen R13-1 input failure; "
                "it does not define or authorize a replacement model"
            ),
        },
    }
    payload["receipt_fingerprint"] = stable_fingerprint(payload)
    return payload


def build_nlcc_development_role_conflict_attribution_receipt(
    *,
    max_examples: int = 8,
    max_records_per_label: int = 3,
) -> dict[str, object]:
    """Attribute the frozen NLCC-v12 Development role conflicts."""

    specs = build_pair_specs(DEVELOPMENT_PROFILE)
    states = materialize_nlcc_supervised_states(
        DEVELOPMENT_PROFILE,
        specs,
    )
    return audit_nlcc_role_conflict_attribution(
        states,
        input_fingerprint=nlcc_development_fingerprint(specs),
        specs=specs,
        feature_stride=FEATURE_STRIDE,
        max_examples=max_examples,
        max_records_per_label=max_records_per_label,
    )


__all__ = [
    "ROLE_CONFLICT_ATTRIBUTION_ALGORITHM_VERSION",
    "audit_nlcc_role_conflict_attribution",
    "build_nlcc_development_role_conflict_attribution_receipt",
    "factorial_input_key_fingerprint",
]
