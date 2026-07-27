from __future__ import annotations

import json

import torch

from cure_lite.phase_resolved_feature_coverage_relation import (
    PhaseResolvedFeatureCoverageRelation,
)
from cure_lite.phase_resolved_relation_population import (
    PFCR_FEATURE_CHANNELS,
    PFCR_ONE_PIXEL_PATTERN,
    PFCR_THREE_PIXEL_PATTERN,
    analytic_reference_relation_config,
    analytic_relation_completion,
    build_phase_resolved_relation_pair_specs,
    materialize_phase_resolved_relation_pair,
    materialize_phase_resolved_relation_population,
    phase_resolved_relation_population_manifest,
    set_analytic_reference_projection,
)


def _reference_module() -> PhaseResolvedFeatureCoverageRelation:
    module = PhaseResolvedFeatureCoverageRelation(
        analytic_reference_relation_config()
    )
    set_analytic_reference_projection(module)
    return module


def test_population_is_balanced_and_pair_features_are_byte_identical() -> None:
    specs = build_phase_resolved_relation_pair_specs()
    states = materialize_phase_resolved_relation_population(specs)

    assert len(specs) == 16
    assert len(states) == 32
    assert {spec.same_object_relation for spec in specs} == {False, True}
    assert {spec.target_phases for spec in specs} == {
        PFCR_ONE_PIXEL_PATTERN,
        PFCR_THREE_PIXEL_PATTERN,
    }
    assert {spec.target_prototype for spec in specs} == {0, 1}
    for spec in specs:
        plus, minus = materialize_phase_resolved_relation_pair(spec)
        assert plus.feature.shape[1] == PFCR_FEATURE_CHANNELS
        assert torch.equal(plus.feature, minus.feature)
        assert not torch.equal(plus.occupancy, minus.occupancy)


def test_scene_process_encodes_phase_structure_without_endpoint_codes() -> None:
    specs = build_phase_resolved_relation_pair_specs()
    one = next(
        spec
        for spec in specs
        if spec.target_prototype == 0
        and spec.target_phases == PFCR_ONE_PIXEL_PATTERN
        and spec.same_object_relation
    )
    three = next(
        spec
        for spec in specs
        if spec.target_prototype == 0
        and spec.target_cell == one.target_cell
        and spec.target_phases == PFCR_THREE_PIXEL_PATTERN
        and spec.same_object_relation
    )
    one_plus, one_minus = materialize_phase_resolved_relation_pair(one)
    three_plus, _ = materialize_phase_resolved_relation_pair(three)

    assert torch.equal(one_plus.feature, one_minus.feature)
    assert not torch.equal(one_plus.feature, three_plus.feature)
    assert int(torch.count_nonzero(one_plus.feature).item()) == 2
    assert int(torch.count_nonzero(three_plus.feature).item()) == 4


def test_analytic_relation_matches_every_completion_pixel_exactly() -> None:
    module = _reference_module()
    states = materialize_phase_resolved_relation_population()

    for state in states:
        fields = analytic_relation_completion(module, state)
        expected = state.completion_target.to(dtype=torch.float32)
        assert torch.equal(fields.completion_score, expected), state.state_id


def test_same_occupancy_geometry_is_resolved_by_feature_relation() -> None:
    specs = build_phase_resolved_relation_pair_specs()
    same = next(
        spec
        for spec in specs
        if spec.target_prototype == 0
        and spec.target_phases == PFCR_ONE_PIXEL_PATTERN
        and spec.same_object_relation
    )
    different = next(
        spec
        for spec in specs
        if spec.target_prototype == same.target_prototype
        and spec.target_cell == same.target_cell
        and spec.target_phases == same.target_phases
        and not spec.same_object_relation
    )
    same_plus, _ = materialize_phase_resolved_relation_pair(same)
    different_plus, _ = materialize_phase_resolved_relation_pair(
        different
    )
    assert torch.equal(same_plus.occupancy, different_plus.occupancy)

    module = _reference_module()
    same_fields = analytic_relation_completion(module, same_plus)
    different_fields = analytic_relation_completion(
        module,
        different_plus,
    )
    assert not torch.equal(
        same_fields.relation.relevant_coverage,
        different_fields.relation.relevant_coverage,
    )
    assert int(same_plus.completion_target.sum().item()) == 0
    assert int(different_plus.completion_target.sum().item()) == 1
    assert torch.equal(
        same_fields.completion_score,
        same_plus.completion_target.to(torch.float32),
    )
    assert torch.equal(
        different_fields.completion_score,
        different_plus.completion_target.to(torch.float32),
    )


def test_prototype_swap_preserves_relation_decision() -> None:
    specs = build_phase_resolved_relation_pair_specs()
    module = _reference_module()
    for relation in (False, True):
        first = next(
            spec
            for spec in specs
            if spec.target_prototype == 0
            and spec.target_phases == PFCR_THREE_PIXEL_PATTERN
            and spec.same_object_relation is relation
        )
        second = next(
            spec
            for spec in specs
            if spec.target_prototype == 1
            and spec.target_cell == first.target_cell
            and spec.target_phases == first.target_phases
            and spec.same_object_relation is relation
        )
        first_plus, _ = materialize_phase_resolved_relation_pair(first)
        second_plus, _ = materialize_phase_resolved_relation_pair(second)
        first_fields = analytic_relation_completion(module, first_plus)
        second_fields = analytic_relation_completion(module, second_plus)
        assert torch.equal(
            first_fields.completion_score,
            second_fields.completion_score,
        )
        assert torch.equal(
            first_plus.completion_target,
            second_plus.completion_target,
        )


def test_population_manifest_is_byte_deterministic() -> None:
    first = phase_resolved_relation_population_manifest()
    second = phase_resolved_relation_population_manifest()
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
    assert first["population_fingerprint"] == (
        second["population_fingerprint"]
    )
    assert first["scene_process"][
        "feature_created_before_endpoint_occupancy"
    ] is True
    assert first["scene_process"]["metadata_is_model_input"] is False
    assert first["scene_process"][
        "sample_specific_float_identity_used"
    ] is False
