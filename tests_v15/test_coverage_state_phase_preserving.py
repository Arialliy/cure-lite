from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from cure_lite.coverage_state_level_set import (
    CSLF_FIELD_AMPLITUDE,
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
)
from cure_lite.coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
    CURELitePhasePreservingCoverageStateLevelSet,
    CoverageStatePhasePreservingConfig,
    pixel_unshuffle_bool_occupancy,
)
from cure_lite.decoder import project_occupancy_to_feature_grid


@pytest.mark.parametrize("stride", (1, 2, 4))
def test_bool_pixel_unshuffle_is_exactly_invertible(
    stride: int,
) -> None:
    generator = torch.Generator().manual_seed(1103 + stride)
    occupancy = torch.rand(
        2,
        1,
        3 * stride,
        5 * stride,
        generator=generator,
    ) > 0.72
    phase = pixel_unshuffle_bool_occupancy(
        occupancy,
        stride=stride,
    )
    assert phase.dtype == torch.bool
    assert phase.is_contiguous()
    assert phase.shape == (2, stride**2, 3, 5)
    reconstructed = F.pixel_shuffle(
        phase.to(torch.float32),
        stride,
    ).to(torch.bool)
    assert torch.equal(reconstructed, occupancy)


@pytest.mark.parametrize("stride", (2, 4))
def test_single_pixel_phase_index_is_row_major_and_pixelshuffle_aligned(
    stride: int,
) -> None:
    for phase_y in range(stride):
        for phase_x in range(stride):
            coarse_y, coarse_x = 1, 2
            occupancy = torch.zeros(
                1,
                1,
                3 * stride,
                4 * stride,
                dtype=torch.bool,
            )
            occupancy[
                0,
                0,
                coarse_y * stride + phase_y,
                coarse_x * stride + phase_x,
            ] = True
            phase = pixel_unshuffle_bool_occupancy(
                occupancy,
                stride=stride,
            )
            expected_channel = phase_y * stride + phase_x
            indices = torch.nonzero(phase, as_tuple=False)
            assert indices.tolist() == [
                [0, expected_channel, coarse_y, coarse_x]
            ]
            assert torch.equal(
                F.pixel_shuffle(
                    phase.to(torch.float32),
                    stride,
                ).to(torch.bool),
                occupancy,
            )


def test_cross_cell_and_multi_component_occupancy_roundtrip() -> None:
    stride = 4
    occupancy = torch.zeros(1, 1, 12, 20, dtype=torch.bool)
    occupancy[..., 3:6, 3:7] = True
    occupancy[..., 8, 16] = True
    occupancy[..., 1, 18] = True
    phase = pixel_unshuffle_bool_occupancy(
        occupancy,
        stride=stride,
    )
    assert torch.count_nonzero(phase) == torch.count_nonzero(occupancy)
    assert torch.equal(
        F.pixel_shuffle(
            phase.to(torch.float32),
            stride,
        ).to(torch.bool),
        occupancy,
    )


def test_ppce_config_policy_and_parameter_formula_are_fixed() -> None:
    config = CoverageStatePhasePreservingConfig(
        feature_channels=5,
        feature_stride=4,
        width=8,
    )
    assert isinstance(config, CoverageStateLevelSetConfig)
    assert config.coverage_policy == (
        CSLF_PHASE_PRESERVING_COVERAGE_POLICY
    )
    assert config.phase_occupancy_channels == 16
    expected = (
        (5 + 16) * 8 * 3 * 3
        + 8 * 3 * 3
        + 8 * 16
        + 16
    )
    assert config.expected_parameter_count == expected
    formal = CoverageStatePhasePreservingConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    legacy = CoverageStateLevelSetConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    assert formal.expected_parameter_count == 23856
    assert (
        formal.expected_parameter_count
        - legacy.expected_parameter_count
    ) == 4320
    with pytest.raises(ValueError, match="coverage_policy"):
        CoverageStatePhasePreservingConfig(
            feature_channels=5,
            feature_stride=4,
            width=8,
            coverage_policy="changed",
        )


def test_ppce_model_is_legacy_subclass_single_path_and_auditable() -> None:
    config = CoverageStatePhasePreservingConfig(
        feature_channels=5,
        feature_stride=2,
        width=8,
    )
    model = CURELitePhasePreservingCoverageStateLevelSet(config)
    assert isinstance(model, CURELiteCoverageStateLevelSet)
    assert model.input_projection.in_channels == 5 + 4
    assert model.input_projection.kernel_size == (3, 3)
    assert model.input_projection.bias is None
    assert model.spatial_mixing.groups == 8
    assert model.spatial_mixing.kernel_size == (3, 3)
    assert model.phase_projection.kernel_size == (1, 1)
    assert model.phase_projection.out_channels == 4
    assert isinstance(model.pixel_shuffle, torch.nn.PixelShuffle)
    assert set(dict(model.named_parameters())) == {
        "input_projection.weight",
        "spatial_mixing.weight",
        "phase_projection.weight",
        "phase_projection.bias",
    }
    assert torch.count_nonzero(model.phase_projection.weight) == 0
    assert torch.all(
        model.phase_projection.bias == CSLF_FIELD_AMPLITUDE
    )
    assert sum(
        parameter.numel() for parameter in model.parameters()
    ) == config.expected_parameter_count


def test_ppce_forward_shapes_dtypes_finiteness_and_initial_field() -> None:
    config = CoverageStatePhasePreservingConfig(
        feature_channels=3,
        feature_stride=2,
        width=6,
    )
    model = CURELitePhasePreservingCoverageStateLevelSet(config)
    feature = torch.randn(2, 3, 4, 5)
    occupancy = torch.zeros(2, 1, 8, 10, dtype=torch.bool)
    occupancy[0, 0, 1, 3] = True
    occupancy[1, 0, 6, 8] = True
    fields = model.forward_fields(feature, occupancy)
    assert fields.encoded_feature.shape == feature.shape
    assert fields.encoded_feature.dtype == torch.float32
    assert fields.phase_occupancy.shape == (2, 4, 4, 5)
    assert fields.phase_occupancy.dtype == torch.bool
    assert fields.hidden.shape == (2, 6, 4, 5)
    assert fields.native_phase_field.shape == (2, 4, 4, 5)
    assert fields.field.shape == occupancy.shape
    assert fields.field.dtype == torch.float32
    assert fields.output_size == (8, 10)
    assert all(
        torch.isfinite(value).all()
        for value in (
            fields.encoded_feature,
            fields.hidden,
            fields.native_phase_field,
            fields.field,
        )
    )
    assert torch.allclose(
        fields.field,
        torch.full_like(fields.field, CSLF_FIELD_AMPLITUDE),
    )


def test_model_output_phases_use_the_same_row_major_alignment() -> None:
    stride = 2
    config = CoverageStatePhasePreservingConfig(
        feature_channels=1,
        feature_stride=stride,
        width=2,
    )
    model = CURELitePhasePreservingCoverageStateLevelSet(config)
    with torch.no_grad():
        model.phase_projection.weight.zero_()
        model.phase_projection.bias.copy_(
            torch.arange(stride**2, dtype=torch.float32)
        )
    feature = torch.ones(1, 1, 2, 3)
    occupancy = torch.zeros(1, 1, 4, 6, dtype=torch.bool)
    field = model(feature, occupancy)
    for y in range(field.shape[-2]):
        for x in range(field.shape[-1]):
            expected_phase = (y % stride) * stride + (x % stride)
            assert field[0, 0, y, x].item() == expected_phase


def _wire_phase_input_to_matching_output(
    model: CURELitePhasePreservingCoverageStateLevelSet,
) -> None:
    stride = model.config.feature_stride
    phase_channels = stride**2
    assert model.config.width >= phase_channels
    with torch.no_grad():
        model.input_projection.weight.zero_()
        model.spatial_mixing.weight.zero_()
        model.phase_projection.weight.zero_()
        model.phase_projection.bias.zero_()
        for phase_index in range(phase_channels):
            occupancy_input_channel = (
                model.config.feature_channels + phase_index
            )
            model.input_projection.weight[
                phase_index,
                occupancy_input_channel,
                1,
                1,
            ] = 1.0
            model.phase_projection.weight[
                phase_index,
                phase_index,
                0,
                0,
            ] = 1.0


def test_ppce_input_phase_is_diagonally_aligned_to_output_phase() -> None:
    stride = 4
    model = CURELitePhasePreservingCoverageStateLevelSet(
        CoverageStatePhasePreservingConfig(
            feature_channels=1,
            feature_stride=stride,
            width=stride**2,
        )
    )
    _wire_phase_input_to_matching_output(model)
    feature = torch.zeros(1, 1, 2, 3)
    coarse_y, coarse_x = 1, 1
    for phase_y in range(stride):
        for phase_x in range(stride):
            occupancy = torch.zeros(
                1,
                1,
                2 * stride,
                3 * stride,
                dtype=torch.bool,
            )
            output_y = coarse_y * stride + phase_y
            output_x = coarse_x * stride + phase_x
            occupancy[0, 0, output_y, output_x] = True
            field = model(feature, occupancy)
            nonzero = torch.nonzero(field > 0.0, as_tuple=False)
            assert nonzero.tolist() == [[0, 0, output_y, output_x]]


def test_ppce_separates_states_that_collide_under_scalar_max() -> None:
    stride = 2
    model = CURELitePhasePreservingCoverageStateLevelSet(
        CoverageStatePhasePreservingConfig(
            feature_channels=1,
            feature_stride=stride,
            width=stride**2,
        )
    )
    _wire_phase_input_to_matching_output(model)
    feature = torch.zeros(1, 1, 2, 2)
    first = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    second = torch.zeros_like(first)
    first[0, 0, 0, 0] = True
    second[0, 0, 1, 1] = True
    assert torch.equal(
        project_occupancy_to_feature_grid(first, (2, 2)),
        project_occupancy_to_feature_grid(second, (2, 2)),
    )
    first_phase = pixel_unshuffle_bool_occupancy(
        first,
        stride=stride,
    )
    second_phase = pixel_unshuffle_bool_occupancy(
        second,
        stride=stride,
    )
    assert not torch.equal(first_phase, second_phase)
    assert not torch.equal(
        model(feature, first),
        model(feature, second),
    )


def test_every_phase_channel_has_a_finite_trainable_input_path() -> None:
    stride = 2
    model = CURELitePhasePreservingCoverageStateLevelSet(
        CoverageStatePhasePreservingConfig(
            feature_channels=1,
            feature_stride=stride,
            width=stride**2,
        )
    )
    _wire_phase_input_to_matching_output(model)
    feature = torch.zeros(1, 1, 2, 2)
    occupancy = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    occupancy[0, 0, :stride, :stride] = True
    loss = model(feature, occupancy).sum()
    loss.backward()
    gradient = model.input_projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    for phase_index in range(stride**2):
        input_channel = model.config.feature_channels + phase_index
        assert torch.count_nonzero(
            gradient[phase_index, input_channel]
        ) > 0


def test_inherited_predict_interfaces_preserve_base_occupancy() -> None:
    model = CURELitePhasePreservingCoverageStateLevelSet(
        CoverageStatePhasePreservingConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    feature = torch.randn(1, 2, 3, 3)
    occupancy = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    occupancy[..., 1, 4] = True
    with torch.no_grad():
        model.phase_projection.bias.fill_(-1.0)
    completion = model.predict_completion(feature, occupancy)
    union = model.predict_union(feature, occupancy)
    assert completion.dtype == torch.bool
    assert union.dtype == torch.bool
    assert completion.shape == occupancy.shape
    assert union.shape == occupancy.shape
    assert not torch.any(completion & occupancy)
    assert torch.all(union[occupancy])
    assert torch.equal(union, occupancy | completion)


def test_ppce_input_validation_fails_closed() -> None:
    occupancy = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    with pytest.raises(TypeError, match="bool"):
        pixel_unshuffle_bool_occupancy(
            occupancy.to(torch.float32),
            stride=2,
        )
    with pytest.raises(ValueError, match="divisible"):
        pixel_unshuffle_bool_occupancy(
            torch.zeros(1, 1, 7, 8, dtype=torch.bool),
            stride=2,
        )
    with pytest.raises(ValueError, match="positive"):
        pixel_unshuffle_bool_occupancy(occupancy, stride=0)
    with pytest.raises(TypeError, match="Config"):
        CURELitePhasePreservingCoverageStateLevelSet(
            CoverageStateLevelSetConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )


def test_legacy_level_set_contract_remains_scalar_max_projected() -> None:
    config = CoverageStateLevelSetConfig(
        feature_channels=3,
        feature_stride=2,
        width=8,
    )
    model = CURELiteCoverageStateLevelSet(config)
    feature = torch.randn(1, 3, 4, 4)
    occupancy = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    occupancy[..., 1, 3] = True
    fields = model.forward_fields(feature, occupancy)
    assert model.input_projection.in_channels == 3 + 1
    assert fields.projected_occupancy.shape == (1, 1, 4, 4)
    assert torch.equal(
        fields.projected_occupancy,
        project_occupancy_to_feature_grid(
            occupancy,
            (4, 4),
        ),
    )
    assert sum(
        parameter.numel() for parameter in model.parameters()
    ) == config.expected_parameter_count
