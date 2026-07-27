"""Formal input-contract preflight for CURE-Lite relation state v2."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import torch

from .cache.schema import stable_fingerprint
from .phase_resolved_feature_coverage_relation import (
    PhaseResolvedFeatureCoverageRelation,
)
from .phase_resolved_relation_population import (
    PFCR_FEATURE_STRIDE,
    PFCR_ONE_PIXEL_PATTERN,
    PFCR_THREE_PIXEL_PATTERN,
    PhaseResolvedRelationPairSpec,
    PhaseResolvedRelationState,
    analytic_reference_relation_config,
    analytic_relation_completion,
    build_phase_resolved_relation_pair_specs,
    materialize_phase_resolved_relation_population,
    phase_resolved_relation_population_manifest,
    set_analytic_reference_projection,
)


PFCR_PREFLIGHT_ALGORITHM_VERSION = (
    "cure-lite.phase-resolved-relation-preflight.v1"
)


def _relation_role_key(
    state: PhaseResolvedRelationState,
    *,
    row: int,
    column: int,
    fields: object,
) -> tuple[object, ...]:
    cell_row = row // PFCR_FEATURE_STRIDE
    cell_column = column // PFCR_FEATURE_STRIDE
    phase_row = row % PFCR_FEATURE_STRIDE
    phase_column = column % PFCR_FEATURE_STRIDE
    phase_index = phase_row * PFCR_FEATURE_STRIDE + phase_column
    evidence = float(
        fields.phase_evidence_strength[
            0,
            phase_index,
            cell_row,
            cell_column,
        ].item()
    )
    directional_relation = tuple(
        float(value)
        for value in fields.relevant_coverage[
            0,
            phase_index,
            :,
            cell_row,
            cell_column,
        ].tolist()
    )
    return (
        (phase_row, phase_column),
        evidence,
        directional_relation,
    )


def _role_conflict_receipt(
    states: tuple[PhaseResolvedRelationState, ...],
    relation_fields: Mapping[str, object],
    *,
    max_examples: int,
) -> dict[str, object]:
    table: dict[
        tuple[object, ...],
        dict[int, list[dict[str, object]]],
    ] = {}
    record_count = 0
    positive_count = 0
    for state in states:
        fields = relation_fields[state.state_id]
        for row, column in torch.nonzero(
            state.valid_mask[0, 0],
            as_tuple=False,
        ).tolist():
            label = int(
                state.completion_target[0, 0, row, column].item()
            )
            key = _relation_role_key(
                state,
                row=int(row),
                column=int(column),
                fields=fields,
            )
            buckets = table.setdefault(key, {0: [], 1: []})
            if len(buckets[label]) < 3:
                buckets[label].append(
                    {
                        "state_id": state.state_id,
                        "output_position": [int(row), int(column)],
                        "endpoint_role": state.endpoint_role,
                    }
                )
            record_count += 1
            positive_count += label
    conflicts = tuple(
        (key, buckets)
        for key, buckets in table.items()
        if buckets[0] and buckets[1]
    )
    ranked = sorted(conflicts, key=lambda item: stable_fingerprint(item[0]))
    return {
        "key_definition": (
            "output phase, phase-query evidence norm, complete directional "
            "feature-conditioned relevant-coverage vector"
        ),
        "absolute_origin_used": False,
        "raw_feature_value_used": False,
        "prototype_id_used": False,
        "metadata_used": False,
        "record_count": record_count,
        "positive_record_count": positive_count,
        "negative_record_count": record_count - positive_count,
        "key_count": len(table),
        "conflict_key_count": len(conflicts),
        "examples": [
            {
                "key_fingerprint": stable_fingerprint(key),
                "negative_examples": buckets[0],
                "positive_examples": buckets[1],
            }
            for key, buckets in ranked[:max_examples]
        ],
    }


def _matched_relation_receipt(
    specs: tuple[PhaseResolvedRelationPairSpec, ...],
    states: tuple[PhaseResolvedRelationState, ...],
    relation_fields: Mapping[str, object],
) -> dict[str, object]:
    states_by_pair = {
        state.pair_index: state
        for state in states
        if state.endpoint_role == "occupancy_plus"
    }
    grouped: dict[
        tuple[object, ...],
        dict[bool, PhaseResolvedRelationPairSpec],
    ] = {}
    for spec in specs:
        key = (
            spec.target_prototype,
            spec.target_cell,
            spec.coverage_cell,
            spec.target_phases,
        )
        grouped.setdefault(key, {})[
            spec.same_object_relation
        ] = spec
    rows: list[dict[str, object]] = []
    for group_key in sorted(grouped, key=stable_fingerprint):
        variants = grouped[group_key]
        if set(variants) != {False, True}:
            raise AssertionError(
                "each matched geometry requires same/different variants"
            )
        same_spec = variants[True]
        different_spec = variants[False]
        same_state = states_by_pair[same_spec.pair_index]
        different_state = states_by_pair[different_spec.pair_index]
        same_fields = relation_fields[same_state.state_id]
        different_fields = relation_fields[different_state.state_id]
        occupancy_equal = torch.equal(
            same_state.occupancy,
            different_state.occupancy,
        )
        spatial_support_equal = torch.equal(
            torch.any(same_state.feature != 0.0, dim=1),
            torch.any(different_state.feature != 0.0, dim=1),
        )
        target_differs = not torch.equal(
            same_state.completion_target,
            different_state.completion_target,
        )
        relation_differs = not torch.equal(
            same_fields.relevant_coverage,
            different_fields.relevant_coverage,
        )
        passed = (
            occupancy_equal
            and spatial_support_equal
            and target_differs
            and relation_differs
        )
        rows.append(
            {
                "group_fingerprint": stable_fingerprint(group_key),
                "same_pair_index": same_spec.pair_index,
                "different_pair_index": different_spec.pair_index,
                "occupancy_byte_equal": occupancy_equal,
                "spatial_feature_support_equal": spatial_support_equal,
                "completion_target_differs": target_differs,
                "relation_tensor_differs": relation_differs,
                "passed": passed,
            }
        )
    return {
        "matched_group_count": len(rows),
        "passed_group_count": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "rows": rows,
    }


def _phase_sufficiency_receipt(
    specs: tuple[PhaseResolvedRelationPairSpec, ...],
    states: tuple[PhaseResolvedRelationState, ...],
) -> dict[str, object]:
    plus_by_pair = {
        state.pair_index: state
        for state in states
        if state.endpoint_role == "occupancy_plus"
    }
    counts: Counter[str] = Counter()
    for spec in specs:
        key = (
            "one_pixel"
            if spec.target_phases == PFCR_ONE_PIXEL_PATTERN
            else "three_pixel"
        )
        state = plus_by_pair[spec.pair_index]
        counts[f"{key}_specs"] += 1
        counts[f"{key}_feature_nonzero"] += int(
            torch.count_nonzero(state.feature).item()
        )
    return {
        "one_pixel_spec_count": counts["one_pixel_specs"],
        "three_pixel_spec_count": counts["three_pixel_specs"],
        "one_pixel_total_feature_nonzero": counts[
            "one_pixel_feature_nonzero"
        ],
        "three_pixel_total_feature_nonzero": counts[
            "three_pixel_feature_nonzero"
        ],
        "phase_patterns_have_distinct_global_encoder_outputs": (
            counts["one_pixel_feature_nonzero"]
            != counts["three_pixel_feature_nonzero"]
        ),
    }


def build_phase_resolved_relation_preflight_receipt(
    *,
    max_examples: int = 8,
) -> dict[str, object]:
    """Build the deterministic CURE-Lite relation-state v2 preflight."""

    if (
        isinstance(max_examples, bool)
        or not isinstance(max_examples, int)
        or max_examples < 1
    ):
        raise ValueError("max_examples must be a positive integer")
    specs = build_phase_resolved_relation_pair_specs()
    states = materialize_phase_resolved_relation_population(specs)
    population_manifest = phase_resolved_relation_population_manifest(
        specs
    )
    module = PhaseResolvedFeatureCoverageRelation(
        analytic_reference_relation_config()
    )
    set_analytic_reference_projection(module)
    module.eval()

    relation_fields: dict[str, object] = {}
    state_rows: list[dict[str, object]] = []
    maximum_absolute_error = 0.0
    mismatch_pixel_count = 0
    with torch.no_grad():
        for state in states:
            completion = analytic_relation_completion(module, state)
            relation_fields[state.state_id] = completion.relation
            expected = state.completion_target.to(dtype=torch.float32)
            absolute_error = torch.abs(
                completion.completion_score - expected
            )
            state_maximum = float(absolute_error.max().item())
            state_mismatches = int(
                torch.count_nonzero(absolute_error).item()
            )
            maximum_absolute_error = max(
                maximum_absolute_error,
                state_maximum,
            )
            mismatch_pixel_count += state_mismatches
            state_rows.append(
                {
                    "state_id": state.state_id,
                    "pair_index": state.pair_index,
                    "endpoint_role": state.endpoint_role,
                    "positive_target_pixels": int(
                        state.completion_target.sum().item()
                    ),
                    "maximum_absolute_error": state_maximum,
                    "mismatch_pixel_count": state_mismatches,
                }
            )

    role = _role_conflict_receipt(
        states,
        relation_fields,
        max_examples=max_examples,
    )
    matched = _matched_relation_receipt(
        specs,
        states,
        relation_fields,
    )
    phase = _phase_sufficiency_receipt(specs, states)
    analytic_exact = (
        maximum_absolute_error == 0.0
        and mismatch_pixel_count == 0
    )
    input_contract_pass = (
        analytic_exact
        and role["conflict_key_count"] == 0
        and matched["all_passed"]
        and phase[
            "phase_patterns_have_distinct_global_encoder_outputs"
        ]
    )
    payload: dict[str, object] = {
        "schema_version": PFCR_PREFLIGHT_ALGORITHM_VERSION,
        "scope": {
            "model": "CURE-Lite",
            "stage": "independent relation-state input contract v2",
            "external_inputs": ["feature", "occupancy"],
            "full_decoder_training_performed": False,
            "dataset_metrics_read": False,
            "full_CURE_in_scope": False,
        },
        "population_fingerprint": population_manifest[
            "population_fingerprint"
        ],
        "population_manifest": population_manifest,
        "analytic_reference": {
            "meaning": (
                "representational sufficiency of the fixed global scene "
                "encoder; not a trained result or production initialization"
            ),
            "state_count": len(states),
            "maximum_absolute_error": maximum_absolute_error,
            "mismatch_pixel_count": mismatch_pixel_count,
            "exact_completion_match": analytic_exact,
            "state_rows": state_rows,
        },
        "relation_role_identifiability": role,
        "matched_same_geometry_relevance": matched,
        "phase_sufficiency": phase,
        "decision": {
            "input_contract_v2_pass": input_contract_pass,
            "relation_state_implementation_authorized": input_contract_pass,
            "full_decoder_training_authorized": False,
            "training_blocker": (
                "the learned evidence-release state equation and its "
                "training objective are not implemented or frozen"
            ),
            "old_nlcc_training_authorized": False,
        },
    }
    payload["receipt_fingerprint"] = stable_fingerprint(payload)
    return payload


__all__ = [
    "PFCR_PREFLIGHT_ALGORITHM_VERSION",
    "build_phase_resolved_relation_preflight_receipt",
]
