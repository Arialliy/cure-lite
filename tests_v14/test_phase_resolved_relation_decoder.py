from __future__ import annotations

import inspect

import pytest
import torch

from cure_lite.phase_resolved_relation_decoder import (
    CURELitePhaseResolvedRelationDecoder,
    PFCR_EVIDENCE_CEILING,
    PFCR_INITIAL_BASELINE_PROBABILITY,
    PhaseResolvedRelationDecoderConfig,
)
from cure_lite.phase_resolved_relation_population import (
    PFCR_FEATURE_CHANNELS,
    PFCR_FEATURE_STRIDE,
    PFCR_LATENT_DIM,
    analytic_reference_relation_config,
    build_phase_resolved_relation_pair_specs,
    materialize_phase_resolved_relation_pair,
    materialize_phase_resolved_relation_population,
    set_analytic_reference_projection,
)


def _config() -> PhaseResolvedRelationDecoderConfig:
    return PhaseResolvedRelationDecoderConfig(
        feature_channels=PFCR_FEATURE_CHANNELS,
        feature_stride=PFCR_FEATURE_STRIDE,
        relation_dim=PFCR_LATENT_DIM,
    )


def _reference_decoder() -> CURELitePhaseResolvedRelationDecoder:
    decoder = CURELitePhaseResolvedRelationDecoder(_config())
    set_analytic_reference_projection(decoder.relation)
    with torch.no_grad():
        phase_output_count = (
            decoder.config.phase_channels * decoder.config.relation_dim
        )
        decoder.relation.projection.weight[
            :phase_output_count
        ].mul_(8.0)
    return decoder


def test_decoder_is_one_relation_equation_with_exact_parameter_count() -> None:
    config = _config()
    decoder = CURELitePhaseResolvedRelationDecoder(config)

    reference = analytic_reference_relation_config()
    assert decoder.relation.config.feature_channels == (
        reference.feature_channels
    )
    assert decoder.relation.config.feature_stride == (
        reference.feature_stride
    )
    assert decoder.relation.config.relation_dim == reference.relation_dim
    assert sum(parameter.numel() for parameter in decoder.parameters()) == (
        config.expected_parameter_count
    )
    assert config.expected_parameter_count == 1089
    assert tuple(inspect.signature(decoder.forward).parameters) == (
        "feature",
        "occupancy",
    )
    key_offset = config.phase_channels * config.relation_dim
    key_weight = decoder.relation.projection.weight[
        key_offset : key_offset + config.relation_dim
    ]
    for phase_index in range(config.phase_channels):
        start = phase_index * config.relation_dim
        assert torch.equal(
            decoder.relation.projection.weight[
                start : start + config.relation_dim
            ],
            key_weight,
        )


def test_zero_feature_produces_the_frozen_negative_baseline() -> None:
    decoder = CURELitePhaseResolvedRelationDecoder(_config())
    feature = torch.zeros(1, PFCR_FEATURE_CHANNELS, 7, 7)
    occupancy = torch.zeros(1, 1, 28, 28, dtype=torch.bool)
    fields = decoder.forward_fields(feature, occupancy)

    assert torch.equal(
        fields.phase_evidence,
        torch.zeros_like(fields.phase_evidence),
    )
    assert torch.equal(
        fields.native_phase_evidence,
        torch.zeros_like(fields.native_phase_evidence),
    )
    expected = torch.full_like(
        fields.completion_probability,
        PFCR_INITIAL_BASELINE_PROBABILITY,
    )
    assert torch.allclose(
        fields.completion_probability,
        expected,
        atol=1.0e-7,
        rtol=0.0,
    )


def test_reference_parameterization_closes_all_population_states() -> None:
    decoder = _reference_decoder()
    states = materialize_phase_resolved_relation_population()

    for state in states:
        completion = decoder.predict_completion(
            state.feature,
            state.occupancy,
            threshold=0.5,
        )
        assert torch.equal(completion, state.completion_target), state.state_id
        expected_union = state.occupancy | state.completion_target
        prediction = decoder.predict_union(
            state.feature,
            state.occupancy,
            threshold=0.5,
        )
        assert torch.equal(prediction, expected_union), state.state_id


def test_same_geometry_is_controlled_by_feature_coverage_relation() -> None:
    specs = build_phase_resolved_relation_pair_specs()
    same = next(
        spec
        for spec in specs
        if spec.target_prototype == 0
        and len(spec.target_phases) == 1
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

    decoder = _reference_decoder()
    same_fields = decoder.forward_fields(
        same_plus.feature,
        same_plus.occupancy,
    )
    different_fields = decoder.forward_fields(
        different_plus.feature,
        different_plus.occupancy,
    )
    target_position = torch.nonzero(
        different_plus.completion_target[0, 0],
        as_tuple=False,
    )[0]
    row, column = (int(value) for value in target_position.tolist())
    assert same_fields.completion_probability[
        0, 0, row, column
    ].item() < 0.05
    assert different_fields.completion_probability[
        0, 0, row, column
    ].item() > 0.95


def test_decoder_does_not_backpropagate_into_detector_feature() -> None:
    decoder = _reference_decoder()
    state = materialize_phase_resolved_relation_population()[1]
    feature = state.feature.clone().requires_grad_(True)

    logits = decoder(feature, state.occupancy)
    logits.sum().backward()

    assert feature.grad is None
    assert decoder.relation.projection.weight.grad is not None
    assert decoder.baseline_raw.grad is not None
    assert torch.isfinite(decoder.relation.projection.weight.grad).all()
    assert torch.isfinite(decoder.baseline_raw.grad)


def test_native_subpixel_contract_refuses_interpolation() -> None:
    decoder = CURELitePhaseResolvedRelationDecoder(_config())
    feature = torch.zeros(1, PFCR_FEATURE_CHANNELS, 7, 7)
    wrong_size = torch.zeros(1, 1, 27, 28, dtype=torch.bool)

    with pytest.raises(
        ValueError,
        match="does not interpolate tiny-target logits",
    ):
        decoder(feature, wrong_size)


def test_threshold_and_field_contracts_are_strict() -> None:
    decoder = _reference_decoder()
    state = materialize_phase_resolved_relation_population()[0]
    fields = decoder.forward_fields(state.feature, state.occupancy)

    assert fields.phase_evidence.shape == (1, 16, 7, 7)
    assert fields.phase_evidence.max().item() <= PFCR_EVIDENCE_CEILING
    assert fields.release_gate.shape == (1, 16, 7, 7)
    assert fields.native_phase_evidence.shape == (1, 16, 7, 7)
    assert fields.native_phase_logits.shape == (1, 16, 7, 7)
    assert fields.logits.shape == (1, 1, 28, 28)
    assert fields.completion_probability.shape == (1, 1, 28, 28)
    assert fields.output_size == (28, 28)
    with pytest.raises(ValueError, match="threshold"):
        decoder.predict_completion(
            state.feature,
            state.occupancy,
            threshold=1.0,
        )
