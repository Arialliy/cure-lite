from __future__ import annotations

import inspect

import pytest
import torch

from cure_lite.phase_resolved_feature_coverage_relation import (
    PFCR_NEIGHBORHOOD_OFFSETS,
    PhaseResolvedFeatureCoverageRelation,
    PhaseResolvedFeatureCoverageRelationConfig,
    directional_occupancy_basis,
    zero_preserving_l2_normalize,
)


def _config() -> PhaseResolvedFeatureCoverageRelationConfig:
    return PhaseResolvedFeatureCoverageRelationConfig(
        feature_channels=2,
        feature_stride=2,
        relation_dim=1,
    )


def _operator() -> PhaseResolvedFeatureCoverageRelation:
    module = PhaseResolvedFeatureCoverageRelation(_config())
    with torch.no_grad():
        module.projection.weight.zero_()
        # Phase 0 reads channel 0; phase 1 reads channel 1.  The remaining
        # two queries read channel 0.  The final projection is the key and
        # reads channel 0.
        module.projection.weight[0, 0, 0, 0] = 1.0
        module.projection.weight[1, 1, 0, 0] = 1.0
        module.projection.weight[2, 0, 0, 0] = 1.0
        module.projection.weight[3, 0, 0, 0] = 1.0
        module.projection.weight[4, 0, 0, 0] = 1.0
    return module


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    feature = torch.zeros(1, 2, 3, 3, dtype=torch.float32)
    feature[0, 0, 1, 1] = 1.0
    feature[0, 0, 1, 2] = 1.0
    occupancy = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    occupancy[0, 0, 2, 4] = True
    return feature, occupancy


def test_config_fixes_one_lightweight_shared_relation_operator() -> None:
    config = _config()
    module = PhaseResolvedFeatureCoverageRelation(config)

    assert config.phase_channels == 4
    assert config.neighborhood_count == 9
    assert config.projection_channels == 5
    assert config.expected_parameter_count == 10
    assert sum(parameter.numel() for parameter in module.parameters()) == 10
    assert module.projection.bias is None
    assert tuple(inspect.signature(module.forward).parameters) == (
        "feature",
        "occupancy",
    )


def test_zero_preserving_normalization_has_exact_zero_fixed_point() -> None:
    value = torch.tensor(
        [[[[0.0]], [[0.0]]], [[[3.0]], [[4.0]]]],
        dtype=torch.float32,
    )
    normalized = zero_preserving_l2_normalize(
        value,
        dim=1,
        epsilon=1.0e-6,
    )

    assert torch.equal(normalized[0], torch.zeros_like(normalized[0]))
    assert normalized[1, 0, 0, 0].item() == pytest.approx(0.6)
    assert normalized[1, 1, 0, 0].item() == pytest.approx(0.8)


def test_directional_basis_preserves_offset_identity() -> None:
    projected = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    projected[0, 0, 1, 2] = True
    basis = directional_occupancy_basis(projected)

    assert PFCR_NEIGHBORHOOD_OFFSETS[5] == (0, 1)
    assert bool(basis[0, 5, 1, 1])
    assert int(basis[:, :, 1, 1].sum().item()) == 1


def test_relation_distinguishes_relevant_and_orthogonal_coverage() -> None:
    module = _operator()
    feature, occupancy = _inputs()
    fields = module.forward_fields(feature, occupancy)

    # Candidate (1,1), right-neighbor offset 5.  Query phase 0 and the
    # occupied neighbor key both use channel 0.
    assert fields.affinity[0, 0, 5, 1, 1].item() == pytest.approx(1.0)
    assert fields.relevant_coverage[
        0, 0, 5, 1, 1
    ].item() == pytest.approx(1.0)
    assert fields.coverage_burden[0, 0, 1, 1].item() == pytest.approx(
        1.0
    )

    # Phase 1 reads channel 1, which is absent at the candidate.
    assert fields.affinity[0, 1, 5, 1, 1].item() == pytest.approx(0.0)
    assert fields.coverage_burden[0, 1, 1, 1].item() == pytest.approx(
        0.0
    )

    orthogonal_feature = feature.clone()
    orthogonal_feature[0, 0, 1, 2] = 0.0
    orthogonal_feature[0, 1, 1, 2] = 1.0
    orthogonal = module.forward_fields(
        orthogonal_feature,
        occupancy,
    )
    assert orthogonal.affinity[0, 0, 5, 1, 1].item() == pytest.approx(
        0.0
    )
    assert orthogonal.relevant_coverage[
        0, 0, 5, 1, 1
    ].item() == pytest.approx(0.0)


def test_relation_affinity_does_not_use_global_scale_or_sign_identity() -> None:
    module = _operator()
    feature, occupancy = _inputs()
    reference = module.forward_fields(feature, occupancy)
    scaled = module.forward_fields(7.0 * feature, occupancy)
    sign_flipped = module.forward_fields(-feature, occupancy)

    assert torch.equal(reference.affinity, scaled.affinity)
    assert torch.equal(reference.affinity, sign_flipped.affinity)
    assert torch.equal(
        reference.relevant_coverage,
        scaled.relevant_coverage,
    )
    assert torch.equal(
        reference.relevant_coverage,
        sign_flipped.relevant_coverage,
    )
    assert torch.allclose(
        reference.normalized_feature,
        scaled.normalized_feature,
        atol=1.0e-7,
        rtol=0.0,
    )


def test_zero_feature_or_zero_occupancy_forces_zero_relation() -> None:
    module = _operator()
    feature, occupancy = _inputs()

    zero_feature = module.forward_fields(
        torch.zeros_like(feature),
        occupancy,
    )
    assert torch.equal(
        zero_feature.affinity,
        torch.zeros_like(zero_feature.affinity),
    )
    assert torch.equal(
        zero_feature.phase_evidence_strength,
        torch.zeros_like(zero_feature.phase_evidence_strength),
    )
    assert torch.equal(
        zero_feature.relevant_coverage,
        torch.zeros_like(zero_feature.relevant_coverage),
    )
    assert torch.equal(
        zero_feature.coverage_burden,
        torch.zeros_like(zero_feature.coverage_burden),
    )

    zero_occupancy = module.forward_fields(
        feature,
        torch.zeros_like(occupancy),
    )
    assert torch.equal(
        zero_occupancy.relevant_coverage,
        torch.zeros_like(zero_occupancy.relevant_coverage),
    )
    assert torch.equal(
        zero_occupancy.coverage_burden,
        torch.zeros_like(zero_occupancy.coverage_burden),
    )


def test_feature_is_frozen_while_relation_parameters_receive_gradient() -> None:
    module = _operator()
    feature, occupancy = _inputs()
    feature.requires_grad_(True)

    burden = module(feature, occupancy)
    burden.sum().backward()

    assert feature.grad is None
    assert module.projection.weight.grad is not None
    assert torch.isfinite(module.projection.weight.grad).all()


def test_fields_have_frozen_shapes_types_and_bounds() -> None:
    module = _operator()
    feature, occupancy = _inputs()
    fields = module.forward_fields(feature, occupancy)

    assert fields.phase_query.shape == (1, 4, 1, 3, 3)
    assert fields.coverage_key.shape == (1, 1, 3, 3)
    assert fields.phase_evidence_strength.shape == (1, 4, 3, 3)
    assert fields.projected_occupancy.shape == (1, 1, 3, 3)
    assert fields.projected_occupancy.dtype == torch.bool
    assert fields.occupancy_basis.shape == (1, 9, 3, 3)
    assert fields.occupancy_basis.dtype == torch.bool
    assert fields.affinity.shape == (1, 4, 9, 3, 3)
    assert fields.relevant_coverage.shape == (1, 4, 9, 3, 3)
    assert fields.coverage_burden.shape == (1, 4, 3, 3)
    assert fields.neighborhood_offsets == PFCR_NEIGHBORHOOD_OFFSETS
    for value in (
        fields.affinity,
        fields.relevant_coverage,
        fields.coverage_burden,
    ):
        assert torch.isfinite(value).all()
        assert bool(((value >= 0.0) & (value <= 1.0)).all())


def test_invalid_inputs_fail_closed() -> None:
    module = _operator()
    feature, occupancy = _inputs()

    with pytest.raises(TypeError, match="occupancy must be bool"):
        module(feature, occupancy.to(dtype=torch.float32))
    malformed = feature.clone()
    malformed[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="feature must be finite"):
        module(malformed, occupancy)
    with pytest.raises(ValueError, match="configured C"):
        module(feature[:, :1], occupancy)
