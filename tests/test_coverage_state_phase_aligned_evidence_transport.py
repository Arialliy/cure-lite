from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from cure_lite.coverage_state_level_set import (
    CSLF_FIELD_AMPLITUDE,
    CURELiteCoverageStateLevelSet,
)
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
    align_corners_false_axis_offsets,
    align_corners_false_phase_offsets,
    bilinear_phase_aligned_feature_affine,
    row_major_phase_pack,
    row_major_phase_unpack,
)


def _config(
    *,
    channels: int = 2,
    stride: int = 2,
    width: int = 4,
) -> CoverageStatePhaseAlignedEvidenceTransportConfig:
    return CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=channels,
        feature_stride=stride,
        width=width,
    )


def _randomize_output_path(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
    *,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.joint_state_weight.copy_(
            0.12
            * torch.randn(
                model.joint_state_weight.shape,
                generator=generator,
            )
        )
        model.joint_hidden_bias.copy_(
            0.08
            * torch.randn(
                model.joint_hidden_bias.shape,
                generator=generator,
            )
        )
        model.scalar_energy_weight.copy_(
            0.2
            * torch.randn(
                model.scalar_energy_weight.shape,
                generator=generator,
            )
        )


def test_paet_contract_has_exact_bfa_state_shapes_init_and_count() -> None:
    config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    assert config.expected_parameter_count == 64064
    assert config.phase_occupancy_channels == 16
    assert config.coarse_radius == 2
    assert config.kernel_size == 5
    assert not hasattr(config, "curvature")
    assert not hasattr(config, "learned_scale")

    torch.manual_seed(3101)
    bfa = CURELiteBinaryFlipAntisymmetricLevelSet(
        CoverageStateBinaryFlipAntisymmetricConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    torch.manual_seed(3101)
    paet = CURELitePhaseAlignedEvidenceTransportLevelSet(_config())
    assert isinstance(paet, CURELiteCoverageStateLevelSet)
    assert tuple(paet.state_dict()) == tuple(bfa.state_dict()) == (
        "joint_state_weight",
        "joint_hidden_bias",
        "scalar_energy_weight",
    )
    for name, value in bfa.state_dict().items():
        assert torch.equal(value, paet.state_dict()[name])
    assert tuple(paet.named_buffers()) == ()
    assert sum(value.numel() for value in paet.parameters()) == (
        paet.config.expected_parameter_count
    )


def test_stride_four_offsets_are_exact_and_row_major() -> None:
    axis = align_corners_false_axis_offsets(4)
    assert axis == (-3 / 8, -1 / 8, 1 / 8, 3 / 8)
    expected = tuple(
        (row, column)
        for row in axis
        for column in axis
    )
    assert align_corners_false_phase_offsets(4) == expected
    assert expected[0] == (-3 / 8, -3 / 8)
    assert expected[3] == (-3 / 8, 3 / 8)
    assert expected[4] == (-1 / 8, -3 / 8)
    assert expected[-1] == (3 / 8, 3 / 8)


@pytest.mark.parametrize("stride", (1, 2, 4))
def test_phase_pack_unpack_is_exact_pixelshuffle_inverse(stride: int) -> None:
    generator = torch.Generator().manual_seed(3102 + stride)
    fine = torch.randn(
        2,
        3,
        3 * stride,
        5 * stride,
        generator=generator,
    )
    phase = row_major_phase_pack(fine, stride=stride)
    assert phase.shape == (2, stride * stride, 3, 3, 5)
    assert torch.equal(
        row_major_phase_unpack(phase, stride=stride),
        fine,
    )
    canonical = (
        F.pixel_unshuffle(fine, stride)
        .reshape(2, 3, stride * stride, 3, 5)
        .permute(0, 2, 1, 3, 4)
    )
    assert torch.equal(phase, canonical)


def test_transport_pack_unpacks_to_exact_bilinear_grid() -> None:
    coarse = torch.randn(2, 3, 4, 5)
    upsampled, phase = bilinear_phase_aligned_feature_affine(
        coarse,
        stride=4,
    )
    expected = F.interpolate(
        coarse,
        scale_factor=4,
        mode="bilinear",
        align_corners=False,
    )
    assert torch.equal(upsampled, expected)
    assert torch.equal(
        row_major_phase_unpack(phase, stride=4),
        expected,
    )


def test_constant_and_ramp_bind_phase_alignment_and_nondegeneracy() -> None:
    constant = torch.full((1, 2, 4, 4), 2.75)
    _, constant_phase = bilinear_phase_aligned_feature_affine(
        constant,
        stride=4,
    )
    assert torch.equal(
        constant_phase,
        torch.full_like(constant_phase, 2.75),
    )

    rows = torch.arange(4, dtype=torch.float32)[:, None]
    columns = torch.arange(4, dtype=torch.float32)[None, :]
    ramp = (10.0 * rows + columns)[None, None]
    _, phase = bilinear_phase_aligned_feature_affine(ramp, stride=4)
    offsets = align_corners_false_phase_offsets(4)
    observed = phase[0, :, 0, 1, 1]
    expected = torch.tensor(
        [
            10.0 * (1.0 + row_offset)
            + (1.0 + column_offset)
            for row_offset, column_offset in offsets
        ]
    )
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=1.0e-6)
    assert torch.unique(observed).numel() == 16


@pytest.mark.parametrize(
    ("seed", "shape"),
    (
        (3110, (1, 2, 3, 4)),
        (3111, (2, 2, 2, 3)),
    ),
)
def test_efficient_paet_matches_literal_local_reference(
    seed: int,
    shape: tuple[int, int, int, int],
) -> None:
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(
        _config(channels=2, stride=2, width=3)
    )
    _randomize_output_path(model, seed=seed)
    generator = torch.Generator().manual_seed(seed + 50)
    feature = torch.randn(shape, generator=generator)
    occupancy = (
        torch.rand(
            shape[0],
            1,
            shape[2] * 2,
            shape[3] * 2,
            generator=generator,
        )
        > 0.57
    )
    efficient = model.forward_fields(feature, occupancy).field
    reference = model.forward_reference(feature, occupancy)
    torch.testing.assert_close(
        efficient,
        reference,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_zero_feature_anchor_is_exact_for_arbitrary_occupancy() -> None:
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(_config())
    _randomize_output_path(model, seed=3112)
    feature = torch.zeros(2, 2, 4, 5)
    occupancy = torch.rand(2, 1, 8, 10) > 0.45
    fields = model.forward_fields(feature, occupancy)
    assert torch.equal(
        fields.field,
        torch.full_like(fields.field, CSLF_FIELD_AMPLITUDE),
    )
    assert torch.count_nonzero(fields.phase_feature_affine) == 0
    assert not bool(torch.any(model.predict_completion(feature, occupancy)))


def test_local_binary_flip_is_odd_and_paired_fields_sum_to_1_8() -> None:
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(_config())
    _randomize_output_path(model, seed=3113)
    generator = torch.Generator().manual_seed(3114)
    feature = torch.randn(1, 2, 4, 5, generator=generator)
    occupancy = torch.rand(1, 1, 8, 10, generator=generator) > 0.55
    output_row, output_column = 5, 6
    flipped = occupancy.clone()
    flipped[0, 0, output_row, output_column] = ~flipped[
        0, 0, output_row, output_column
    ]

    first = model.forward_fields(feature, occupancy)
    second = model.forward_fields(feature, flipped)
    stride = model.feature_stride
    phase = (
        (output_row % stride) * stride + output_column % stride
    )
    row = output_row // stride
    column = output_column // stride
    torch.testing.assert_close(
        first.native_phase_interaction[0, phase, row, column],
        -second.native_phase_interaction[0, phase, row, column],
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    torch.testing.assert_close(
        first.field[0, 0, output_row, output_column]
        + second.field[0, 0, output_row, output_column],
        torch.tensor(1.8),
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_paet_transport_produces_nondegenerate_phase_evidence() -> None:
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(
        _config(channels=1, stride=4, width=1)
    )
    center = model.config.coarse_radius
    with torch.no_grad():
        model.joint_state_weight.zero_()
        model.joint_state_weight[0, 0, center, center] = 1.0
        model.joint_state_weight[
            0, 1:, center, center
        ] = torch.linspace(-0.5, 0.5, 16)
        model.scalar_energy_weight.fill_(1.0)
    rows = torch.arange(4, dtype=torch.float32)[:, None]
    columns = torch.arange(4, dtype=torch.float32)[None, :]
    feature = (10.0 * rows + columns)[None, None]
    occupancy = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    fields = model.forward_fields(feature, occupancy)
    assert torch.unique(
        fields.phase_feature_affine[0, :, 0, 1, 1]
    ).numel() == 16
    assert torch.unique(
        fields.native_phase_interaction[0, :, 1, 1]
    ).numel() > 1


def test_gradient_latency_and_second_step_reach_exact_three_parameters() -> None:
    torch.manual_seed(3115)
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(_config())
    feature = torch.randn(2, 2, 4, 4)
    occupancy = torch.rand(2, 1, 8, 8) > 0.6
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    optimizer.zero_grad(set_to_none=True)
    model(feature, occupancy).square().mean().backward()
    assert bool(torch.any(model.scalar_energy_weight.grad != 0.0))
    assert torch.count_nonzero(model.joint_state_weight.grad) == 0
    assert torch.count_nonzero(model.joint_hidden_bias.grad) == 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    model(feature, occupancy).square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(torch.any(parameter.grad != 0.0))


def test_paet_rejects_wrong_dtype_grid_and_config() -> None:
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(_config())
    feature = torch.randn(1, 2, 3, 3)
    occupancy = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="float32"):
        model(feature.to(torch.float64), occupancy)
    with pytest.raises(ValueError, match="feature_stride"):
        model(feature, occupancy[..., :-1, :])
    with pytest.raises(TypeError, match="config"):
        CURELitePhaseAlignedEvidenceTransportLevelSet(  # type: ignore[arg-type]
            CoverageStateBinaryFlipAntisymmetricConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )
