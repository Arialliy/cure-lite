from __future__ import annotations

import pytest
import torch

from cure_lite.coverage_state_level_set import (
    CSLF_FIELD_AMPLITUDE,
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
    truncated_signed_distance_field,
)
from cure_lite.coverage_state_sobolev import (
    CoverageStatePairBatch,
    CoverageStateSobolevConfig,
    coverage_state_absolute_sobolev_loss,
    coverage_state_absolute_sobolev_loss_from_targets,
    coverage_state_independent_endpoint_loss_from_targets,
    coverage_state_pair_sobolev_loss,
    coverage_state_pair_sobolev_loss_from_targets,
    prepare_coverage_state_absolute_targets,
    prepare_coverage_state_pair_targets,
)


def _single_pixel(
    *,
    size: int = 7,
    y: int = 3,
    x: int = 3,
) -> torch.Tensor:
    value = torch.zeros(1, 1, size, size, dtype=torch.bool)
    value[0, 0, y, x] = True
    return value


def test_truncated_field_has_exact_empty_and_zero_level_semantics() -> None:
    valid = torch.ones(1, 1, 7, 7, dtype=torch.bool)
    empty = torch.zeros_like(valid)
    empty_field = truncated_signed_distance_field(
        empty,
        valid,
        radius=2,
    )
    assert torch.equal(
        empty_field,
        torch.full_like(empty_field, CSLF_FIELD_AMPLITUDE),
    )

    target = _single_pixel()
    field = truncated_signed_distance_field(target, valid, radius=2)
    assert field[0, 0, 3, 3].item() == pytest.approx(-0.45)
    assert field[0, 0, 3, 4].item() == pytest.approx(0.45)
    assert field[0, 0, 0, 0].item() == pytest.approx(0.9)
    assert torch.equal(field < 0.0, target)


def test_truncated_field_encodes_interior_depth_and_valid_domain() -> None:
    target = torch.zeros(1, 1, 9, 9, dtype=torch.bool)
    target[..., 2:7, 2:7] = True
    valid = torch.ones_like(target)
    valid[..., 0, :] = False
    field = truncated_signed_distance_field(target, valid, radius=3)
    assert field[0, 0, 4, 4].item() == pytest.approx(-0.9)
    assert field[0, 0, 2, 2].item() == pytest.approx(-0.3)
    assert field[0, 0, 1, 2].item() == pytest.approx(0.3)
    assert torch.all(field[..., 0, :] == CSLF_FIELD_AMPLITUDE)


def test_truncated_field_does_not_cross_an_invalid_grid_barrier() -> None:
    target = _single_pixel(size=7, y=3, x=1)
    valid = torch.ones_like(target)
    valid[..., :, 3] = False
    field = truncated_signed_distance_field(target, valid, radius=3)
    assert field[0, 0, 3, 2].item() == pytest.approx(0.3)
    assert field[0, 0, 3, 4].item() == pytest.approx(
        CSLF_FIELD_AMPLITUDE
    )


def test_truncated_field_rejects_target_outside_valid_domain() -> None:
    target = _single_pixel()
    valid = torch.ones_like(target)
    valid[0, 0, 3, 3] = False
    with pytest.raises(ValueError, match="outside valid_mask"):
        truncated_signed_distance_field(target, valid, radius=2)


def test_level_set_decoder_is_native_resolution_non_saturating_and_lightweight() -> None:
    config = CoverageStateLevelSetConfig(
        feature_channels=5,
        feature_stride=2,
        width=8,
    )
    model = CURELiteCoverageStateLevelSet(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        config.expected_parameter_count
    )
    feature = torch.randn(2, 5, 4, 6)
    occupancy = torch.zeros(2, 1, 8, 12, dtype=torch.bool)
    fields = model.forward_fields(feature, occupancy)
    assert fields.field.shape == occupancy.shape
    assert fields.native_phase_field.shape == (2, 4, 4, 6)
    assert fields.projected_occupancy.shape == (2, 1, 4, 6)
    assert torch.allclose(
        fields.field,
        torch.full_like(fields.field, CSLF_FIELD_AMPLITUDE),
    )
    assert not torch.any(model.predict_completion(feature, occupancy))
    model.phase_projection.bias.data.fill_(2.5)
    unbounded = model(feature, occupancy)
    assert torch.allclose(unbounded, torch.full_like(unbounded, 2.5))


def test_feature_encoding_preserves_within_image_peak_magnitude() -> None:
    model = CURELiteCoverageStateLevelSet(
        CoverageStateLevelSetConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    feature = torch.zeros(1, 2, 2, 2)
    feature[0, :, 0, 0] = torch.tensor([1.0, 2.0])
    feature[0, :, 0, 1] = torch.tensor([10.0, 20.0])
    occupancy = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    encoded = model.forward_fields(feature, occupancy).encoded_feature
    assert torch.allclose(
        encoded[0, :, 0, 1],
        10.0 * encoded[0, :, 0, 0],
    )


def test_coverage_deletion_has_a_finite_spatial_response_domain() -> None:
    torch.manual_seed(13)
    stride = 2
    model = CURELiteCoverageStateLevelSet(
        CoverageStateLevelSetConfig(
            feature_channels=3,
            feature_stride=stride,
            width=8,
        )
    )
    torch.nn.init.normal_(model.phase_projection.weight, std=0.1)
    feature = torch.randn(1, 3, 9, 9)
    occupancy_plus = torch.zeros(1, 1, 18, 18, dtype=torch.bool)
    occupancy_plus[..., 8, 8] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)
    plus = model(feature, occupancy_plus)
    minus = model(feature, occupancy_minus)
    difference = (plus - minus).abs()
    yy, xx = torch.meshgrid(
        torch.arange(18),
        torch.arange(18),
        indexing="ij",
    )
    feature_y = yy // stride
    feature_x = xx // stride
    outside_receptive_field = (
        torch.maximum((feature_y - 4).abs(), (feature_x - 4).abs())
        > 2
    )
    assert torch.count_nonzero(
        difference[0, 0][outside_receptive_field]
    ) == 0
    assert torch.count_nonzero(
        difference[0, 0][~outside_receptive_field]
    ) > 0


def test_level_set_decoder_detaches_frozen_feature_and_preserves_base() -> None:
    model = CURELiteCoverageStateLevelSet(
        CoverageStateLevelSetConfig(
            feature_channels=3,
            feature_stride=2,
            width=8,
        )
    )
    feature = torch.randn(1, 3, 4, 4, requires_grad=True)
    occupancy = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    occupancy[..., 2, 5] = True
    model.phase_projection.bias.data.fill_(-2.0)
    field = model(feature, occupancy)
    field.sum().backward()
    assert feature.grad is None
    union = model.predict_union(feature.detach(), occupancy)
    assert torch.all(union[occupancy])
    assert torch.all(union >= occupancy)


def test_exact_target_fields_zero_the_absolute_and_pair_objectives() -> None:
    valid = torch.ones(1, 1, 7, 7, dtype=torch.bool)
    plus_target = torch.zeros_like(valid)
    minus_target = _single_pixel()
    config = CoverageStateSobolevConfig(truncation_radius=2)
    plus_field = truncated_signed_distance_field(
        plus_target,
        valid,
        radius=2,
    ).requires_grad_()
    minus_field = truncated_signed_distance_field(
        minus_target,
        valid,
        radius=2,
    ).requires_grad_()

    absolute = coverage_state_absolute_sobolev_loss(
        minus_field,
        minus_target,
        valid,
        config=config,
    )
    pair = coverage_state_pair_sobolev_loss(
        plus_field,
        minus_field,
        minus_target,
        plus_target,
        plus_target,
        minus_target,
        valid,
        config=config,
    )
    assert absolute.loss.item() == pytest.approx(0.0)
    assert pair.loss.item() == pytest.approx(0.0)
    assert pair.value_power.item() == pytest.approx(0.0)
    assert pair.spatial_power.item() == pytest.approx(0.0)
    assert torch.count_nonzero(pair.response_error) == 0


def test_precomputed_geometry_matches_direct_losses_and_control() -> None:
    valid = torch.ones(1, 1, 9, 9, dtype=torch.bool)
    target_plus = torch.zeros_like(valid)
    target_minus = _single_pixel(size=9, y=4, x=4)
    occupancy_plus = target_minus.clone()
    occupancy_minus = torch.zeros_like(valid)
    config = CoverageStateSobolevConfig(truncation_radius=2)
    field_plus = truncated_signed_distance_field(
        target_plus,
        valid,
        radius=2,
    )
    field_minus = truncated_signed_distance_field(
        target_minus,
        valid,
        radius=2,
    )
    absolute_targets = prepare_coverage_state_absolute_targets(
        target_minus,
        valid,
        config=config,
    )
    direct_absolute = coverage_state_absolute_sobolev_loss(
        field_minus,
        target_minus,
        valid,
        config=config,
    )
    cached_absolute = coverage_state_absolute_sobolev_loss_from_targets(
        field_minus,
        absolute_targets,
        config=config,
    )
    assert torch.equal(direct_absolute.loss, cached_absolute.loss)

    pair_targets = prepare_coverage_state_pair_targets(
        occupancy_plus,
        occupancy_minus,
        target_plus,
        target_minus,
        valid,
        config=config,
    )
    direct_pair = coverage_state_pair_sobolev_loss(
        field_plus,
        field_minus,
        occupancy_plus,
        occupancy_minus,
        target_plus,
        target_minus,
        valid,
        config=config,
    )
    cached_pair = coverage_state_pair_sobolev_loss_from_targets(
        field_plus,
        field_minus,
        pair_targets,
        config=config,
    )
    independent = coverage_state_independent_endpoint_loss_from_targets(
        field_plus,
        field_minus,
        pair_targets,
        config=config,
    )
    assert torch.equal(direct_pair.loss, cached_pair.loss)
    assert cached_pair.loss.item() == pytest.approx(0.0)
    assert independent.loss.item() == pytest.approx(0.0)

    perturbed_plus = field_plus.clone()
    perturbed_plus[..., 4, 4] = 0.0
    coupled_perturbed = coverage_state_pair_sobolev_loss_from_targets(
        perturbed_plus,
        field_minus,
        pair_targets,
        config=config,
    )
    independent_perturbed = (
        coverage_state_independent_endpoint_loss_from_targets(
            perturbed_plus,
            field_minus,
            pair_targets,
            config=config,
        )
    )
    assert not torch.isclose(
        coupled_perturbed.loss,
        independent_perturbed.loss,
    )


def test_pair_objective_penalizes_missing_coverage_response_and_backpropagates() -> None:
    valid = torch.ones(1, 1, 7, 7, dtype=torch.bool)
    plus_target = torch.zeros_like(valid)
    minus_target = _single_pixel()
    config = CoverageStateSobolevConfig(truncation_radius=2)
    plus_field = truncated_signed_distance_field(
        plus_target,
        valid,
        radius=2,
    )
    minus_field = plus_field.detach().clone().requires_grad_()
    plus_field = plus_field.detach().clone().requires_grad_()
    result = coverage_state_pair_sobolev_loss(
        plus_field,
        minus_field,
        minus_target,
        plus_target,
        plus_target,
        minus_target,
        valid,
        config=config,
    )
    assert result.loss.item() > 0.0
    assert result.value_power.item() > 0.0
    assert result.spatial_power.item() > 0.0
    result.loss.backward()
    assert plus_field.grad is not None
    assert minus_field.grad is not None
    assert torch.isfinite(plus_field.grad).all()
    assert torch.isfinite(minus_field.grad).all()
    assert torch.count_nonzero(plus_field.grad) > 0
    assert torch.count_nonzero(minus_field.grad) > 0


def test_component_null_measure_does_not_dilute_local_false_response() -> None:
    size = 256
    valid = torch.ones(1, 1, size, size, dtype=torch.bool)
    empty_target = torch.zeros_like(valid)
    occupancy_plus = torch.zeros_like(valid)
    occupancy_plus[..., size // 2, size // 2] = True
    occupancy_minus = torch.zeros_like(valid)
    plus_field = torch.full(
        valid.shape,
        CSLF_FIELD_AMPLITUDE,
        dtype=torch.float32,
        requires_grad=True,
    )
    minus_field = plus_field.detach().clone()
    minus_field[..., size // 2, size // 2] = -CSLF_FIELD_AMPLITUDE
    minus_field.requires_grad_()
    result = coverage_state_pair_sobolev_loss(
        plus_field,
        minus_field,
        occupancy_plus,
        occupancy_minus,
        empty_target,
        empty_target,
        valid,
        config=CoverageStateSobolevConfig(truncation_radius=4),
    )
    result.loss.backward()
    local_gradient = minus_field.grad[
        0,
        0,
        size // 2,
        size // 2,
    ].abs()
    far_gradient = minus_field.grad[0, 0, 0, 0].abs()
    assert result.loss.item() > 0.0
    assert torch.equal(result.focus_support, occupancy_plus)
    assert local_gradient > 0.1
    assert far_gradient == 0.0


def test_component_null_focus_includes_existing_completion_and_deletion() -> None:
    size = 64
    valid = torch.ones(1, 1, size, size, dtype=torch.bool)
    target = torch.zeros_like(valid)
    target[..., 16, 16] = True
    occupancy_plus = torch.zeros_like(valid)
    occupancy_plus[..., 48, 48] = True
    occupancy_minus = torch.zeros_like(valid)
    target_field = truncated_signed_distance_field(
        target,
        valid,
        radius=4,
    )
    plus_field = target_field.detach().clone()
    plus_field[..., 16, 16] = CSLF_FIELD_AMPLITUDE
    plus_field.requires_grad_()
    minus_field = plus_field.detach().clone()
    minus_field[..., 48, 48] = -CSLF_FIELD_AMPLITUDE
    minus_field.requires_grad_()
    result = coverage_state_pair_sobolev_loss(
        plus_field,
        minus_field,
        occupancy_plus,
        occupancy_minus,
        target,
        target,
        valid,
        config=CoverageStateSobolevConfig(truncation_radius=4),
    )
    result.loss.backward()
    expected_focus = occupancy_plus | target
    assert torch.equal(result.focus_support, expected_focus)
    assert minus_field.grad[..., 48, 48].abs().item() > 0.0
    assert plus_field.grad[..., 16, 16].abs().item() > 0.0


def _empty_state_island_loss(
    *,
    size: int,
    points: tuple[tuple[int, int], ...],
) -> float:
    valid = torch.ones(1, 1, size, size, dtype=torch.bool)
    target = torch.zeros_like(valid)
    field = torch.full(
        valid.shape,
        CSLF_FIELD_AMPLITUDE,
        dtype=torch.float32,
    )
    for y, x in points:
        field[..., y, x] = -CSLF_FIELD_AMPLITUDE
    return float(
        coverage_state_absolute_sobolev_loss(
            field,
            target,
            valid,
            config=CoverageStateSobolevConfig(truncation_radius=4),
        ).loss
    )


def test_rooted_w1p4_energy_obeys_island_count_and_resolution_laws() -> None:
    grid_points = tuple(
        (8 + 12 * row, 8 + 12 * column)
        for row in range(4)
        for column in range(4)
    )
    one = _empty_state_island_loss(size=64, points=grid_points[:1])
    four = _empty_state_island_loss(size=64, points=grid_points[:4])
    sixteen = _empty_state_island_loss(size=64, points=grid_points)
    assert four / one == pytest.approx(2.0**0.5, rel=0.08)
    assert sixteen / one == pytest.approx(2.0, rel=0.08)

    one_256 = _empty_state_island_loss(
        size=256,
        points=((128, 128),),
    )
    assert one / one_256 == pytest.approx(2.0, rel=0.12)


def test_rooted_w1p4_energy_does_not_starve_a_smaller_false_island() -> None:
    size = 64
    valid = torch.ones(1, 1, size, size, dtype=torch.bool)
    target = torch.zeros_like(valid)
    field = torch.full(
        valid.shape,
        CSLF_FIELD_AMPLITUDE,
        dtype=torch.float32,
    )
    field[..., 16, 16] = -CSLF_FIELD_AMPLITUDE
    field[..., 48, 48] = 0.0
    field.requires_grad_()
    result = coverage_state_absolute_sobolev_loss(
        field,
        target,
        valid,
        config=CoverageStateSobolevConfig(truncation_radius=4),
    )
    result.loss.backward()
    large = field.grad[..., 16, 16].abs().item()
    smaller = field.grad[..., 48, 48].abs().item()
    assert large > smaller > 0.0


def test_coverage_state_pair_batch_distinguishes_edge_kinds() -> None:
    feature = torch.zeros(3, 2, 3, 3)
    valid = torch.ones(3, 1, 6, 6, dtype=torch.bool)
    plus = torch.zeros_like(valid)
    minus = torch.zeros_like(valid)
    target_plus = torch.zeros_like(valid)
    target_minus = torch.zeros_like(valid)

    plus[0, 0, 2, 2] = True
    target_minus[0, 0, 2, 2] = True
    plus[1, 0, 1, 1] = True
    batch = CoverageStatePairBatch(
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        target_plus=target_plus,
        target_minus=target_minus,
        valid_mask=valid,
        pair_ids=("a" * 64, "b" * 64, "c" * 64),
        pair_kinds=(
            "clean_positive",
            "component_null",
            "identity_null",
        ),
        sample_ids=("one", "two", "three"),
    )
    batch.validate()
    invalid = CoverageStatePairBatch(
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        target_plus=target_plus,
        target_minus=target_plus,
        valid_mask=valid,
        pair_ids=batch.pair_ids,
        pair_kinds=batch.pair_kinds,
        sample_ids=batch.sample_ids,
    )
    with pytest.raises(ValueError, match="clean_positive"):
        invalid.validate()


def test_pair_batch_rejects_a_full_grid_change_hidden_by_projection() -> None:
    feature = torch.zeros(1, 2, 3, 3)
    valid = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    plus = torch.zeros_like(valid)
    plus[..., 2, 2] = True
    plus[..., 2, 3] = True
    minus = plus.clone()
    minus[..., 2, 3] = False
    target_plus = torch.zeros_like(valid)
    target_minus = torch.zeros_like(valid)
    target_minus[..., 2, 3] = True
    batch = CoverageStatePairBatch(
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        target_plus=target_plus,
        target_minus=target_minus,
        valid_mask=valid,
        pair_ids=("d" * 64,),
        pair_kinds=("clean_positive",),
        sample_ids=("hidden",),
    )
    with pytest.raises(ValueError, match="projected coverage"):
        batch.validate()


def test_balanced_field_measure_prevents_one_pixel_gradient_dilution() -> None:
    size = 64
    valid = torch.ones(1, 1, size, size, dtype=torch.bool)
    target = _single_pixel(size=size, y=size // 2, x=size // 2)
    field = torch.full(
        target.shape,
        CSLF_FIELD_AMPLITUDE,
        dtype=torch.float32,
        requires_grad=True,
    )
    result = coverage_state_absolute_sobolev_loss(
        field,
        target,
        valid,
        config=CoverageStateSobolevConfig(truncation_radius=4),
    )
    result.loss.backward()
    target_gradient = field.grad[0, 0, size // 2, size // 2].abs()
    far_gradient = field.grad[0, 0, 0, 0].abs()
    assert target_gradient > 100.0 * far_gradient


def test_small_pair_can_learn_a_nonzero_coverage_state_finite_response() -> None:
    torch.manual_seed(7)
    model = CURELiteCoverageStateLevelSet(
        CoverageStateLevelSetConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    feature = torch.randn(1, 2, 4, 4)
    occupancy_plus = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    occupancy_plus[..., 3, 3] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)
    target_plus = torch.zeros_like(occupancy_plus)
    target_minus = torch.zeros_like(occupancy_plus)
    target_minus[..., 3, 3] = True
    valid = torch.ones_like(occupancy_plus)
    config = CoverageStateSobolevConfig(truncation_radius=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    def objective() -> torch.Tensor:
        fields = model(
            torch.cat((feature, feature), dim=0),
            torch.cat((occupancy_plus, occupancy_minus), dim=0),
        )
        return coverage_state_pair_sobolev_loss(
            fields[:1],
            fields[1:],
            occupancy_plus,
            occupancy_minus,
            target_plus,
            target_minus,
            valid,
            config=config,
        ).loss

    initial = float(objective().detach())
    for _ in range(80):
        loss = objective()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = float(objective().detach())
    assert final < 0.45 * initial
