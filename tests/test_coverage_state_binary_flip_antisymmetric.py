from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from cure_lite.coverage_state_binary_flip_antisymmetric import (
    BFA_COARSE_RADIUS,
    BFA_ENERGY_POLICY,
    BFA_FLIP_POLICY,
    BFA_INPUT_REPRESENTATION,
    BFA_INTERACTION_POLICY,
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
    binary_flip_odd_projection,
    flip_binary_center_phase,
)
from cure_lite.coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
    centered_mixed_energy_difference,
)
from cure_lite.coverage_state_level_set import (
    CSLF_FIELD_AMPLITUDE,
    CURELiteCoverageStateLevelSet,
)
from cure_lite.coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
    pixel_unshuffle_bool_occupancy,
)


def _config(
    *,
    channels: int = 2,
    stride: int = 2,
    width: int = 4,
) -> CoverageStateBinaryFlipAntisymmetricConfig:
    return CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=channels,
        feature_stride=stride,
        width=width,
    )


def _randomize_output_path(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
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


def test_bfa_config_freezes_one_independent_mechanism_contract() -> None:
    config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    assert config.coverage_policy == (
        CSLF_PHASE_PRESERVING_COVERAGE_POLICY
    )
    assert config.input_representation == BFA_INPUT_REPRESENTATION
    assert config.interaction_policy == BFA_INTERACTION_POLICY
    assert config.energy_policy == BFA_ENERGY_POLICY
    assert config.flip_policy == BFA_FLIP_POLICY
    assert config.coarse_radius == BFA_COARSE_RADIUS == 2
    assert config.kernel_size == 5
    assert config.phase_occupancy_channels == 16
    assert config.expected_parameter_count == 64064
    assert not hasattr(config, "neutral_phase")

    for kwargs, message in (
        ({"coarse_radius": 1}, "coarse_radius"),
        ({"coverage_policy": "changed"}, "coverage_policy"),
        ({"input_representation": "scalar_max"}, "input_representation"),
        ({"interaction_policy": "changed"}, "interaction_policy"),
        ({"energy_policy": "changed"}, "energy_policy"),
        ({"flip_policy": "changed"}, "flip_policy"),
    ):
        with pytest.raises(ValueError, match=message):
            CoverageStateBinaryFlipAntisymmetricConfig(
                feature_channels=4,
                feature_stride=2,
                width=3,
                **kwargs,
            )


def test_bfa_has_exactly_the_three_cmif_parameters_and_same_initialization() -> None:
    torch.manual_seed(2001)
    old = CURELiteCenteredMixedInteractionLevelSet(
        CoverageStateCenteredMixedInteractionConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    torch.manual_seed(2001)
    new = CURELiteBinaryFlipAntisymmetricLevelSet(_config())

    assert isinstance(new, CURELiteCoverageStateLevelSet)
    assert set(dict(new.named_parameters())) == {
        "joint_state_weight",
        "joint_hidden_bias",
        "scalar_energy_weight",
    }
    assert tuple(new.joint_state_weight.shape) == (4, 6, 5, 5)
    assert tuple(new.joint_hidden_bias.shape) == (4,)
    assert tuple(new.scalar_energy_weight.shape) == (4,)
    assert not tuple(new.named_buffers())
    assert sum(
        parameter.numel() for parameter in new.parameters()
    ) == new.config.expected_parameter_count
    assert torch.count_nonzero(new.joint_state_weight) > 0
    assert torch.count_nonzero(new.joint_hidden_bias) == 0
    assert torch.count_nonzero(new.scalar_energy_weight) == 0
    assert tuple(old.state_dict()) == tuple(new.state_dict())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])


def test_binary_center_flip_is_an_exact_single_bit_involution() -> None:
    generator = torch.Generator().manual_seed(2002)
    patch = torch.rand(4, 5, 5, generator=generator) > 0.5
    flipped = flip_binary_center_phase(
        patch,
        phase_index=2,
        center=2,
    )
    restored = flip_binary_center_phase(
        flipped,
        phase_index=2,
        center=2,
    )
    changed = torch.nonzero(patch != flipped, as_tuple=False)

    assert changed.tolist() == [[2, 2, 2]]
    assert torch.equal(restored, patch)
    assert not torch.equal(flipped, patch)


def test_binary_flip_projection_is_antisymmetric_and_rejects_broadcast() -> None:
    actual = torch.tensor(
        [[[-1.2, 0.3], [2.0, -0.7]]],
        dtype=torch.float32,
    )
    flipped = torch.tensor(
        [[[0.8, -0.5], [1.0, 0.9]]],
        dtype=torch.float32,
    )
    forward = binary_flip_odd_projection(actual, flipped)
    reverse = binary_flip_odd_projection(flipped, actual)
    assert torch.equal(forward, -reverse)
    with pytest.raises(TypeError, match="aligned"):
        binary_flip_odd_projection(actual, flipped[..., :1])


def test_initial_field_is_exact_anchor_without_double_addition() -> None:
    model = CURELiteBinaryFlipAntisymmetricLevelSet(_config())
    feature = torch.randn(2, 2, 4, 5)
    occupancy = torch.rand(2, 1, 8, 10) > 0.72
    fields = model.forward_fields(feature, occupancy)

    assert fields.phase_occupancy.shape == (2, 4, 4, 5)
    assert fields.flip_delta.shape == (2, 4, 4, 4, 5)
    assert fields.odd_feature_presence_hidden.shape == (2, 4, 4, 4, 5)
    assert fields.native_phase_field.shape == (2, 4, 4, 5)
    assert fields.field.shape == occupancy.shape
    assert torch.equal(
        fields.native_phase_field,
        torch.full_like(
            fields.native_phase_field,
            CSLF_FIELD_AMPLITUDE,
        ),
    )
    assert torch.equal(
        fields.field,
        torch.full_like(fields.field, CSLF_FIELD_AMPLITUDE),
    )
    assert not bool(torch.any(model.predict_completion(feature, occupancy)))
    assert torch.equal(model.predict_union(feature, occupancy), occupancy)


def test_local_flip_pair_has_opposite_interaction_and_field_sum_two_a() -> None:
    model = CURELiteBinaryFlipAntisymmetricLevelSet(_config())
    _randomize_output_path(model, seed=2003)
    generator = torch.Generator().manual_seed(2004)
    feature = torch.randn(1, 2, 3, 4, generator=generator)
    occupancy = torch.rand(
        1,
        1,
        6,
        8,
        generator=generator,
    ) > 0.55
    output_row, output_column = 3, 6
    flipped_occupancy = occupancy.clone()
    flipped_occupancy[
        0,
        0,
        output_row,
        output_column,
    ] = ~flipped_occupancy[0, 0, output_row, output_column]

    actual = model.forward_fields(feature, occupancy)
    flipped = model.forward_fields(feature, flipped_occupancy)
    stride = model.config.feature_stride
    phase_index = (
        output_row % stride
    ) * stride + output_column % stride
    coarse_row = output_row // stride
    coarse_column = output_column // stride
    actual_interaction = actual.native_phase_interaction[
        0,
        phase_index,
        coarse_row,
        coarse_column,
    ]
    flipped_interaction = flipped.native_phase_interaction[
        0,
        phase_index,
        coarse_row,
        coarse_column,
    ]
    actual_field = actual.field[0, 0, output_row, output_column]
    flipped_field = flipped.field[0, 0, output_row, output_column]

    torch.testing.assert_close(
        actual_interaction,
        -flipped_interaction,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    torch.testing.assert_close(
        actual_field + flipped_field,
        torch.tensor(2.0 * CSLF_FIELD_AMPLITUDE),
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    torch.testing.assert_close(
        actual.native_phase_field
        - actual.native_phase_interaction,
        torch.full_like(
            actual.native_phase_field,
            CSLF_FIELD_AMPLITUDE,
        ),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert not torch.isclose(
        actual_field + flipped_field,
        torch.tensor(4.0 * CSLF_FIELD_AMPLITUDE),
    )


def test_hidden_and_scalar_energy_odd_projections_are_identical() -> None:
    model = CURELiteBinaryFlipAntisymmetricLevelSet(_config())
    _randomize_output_path(model, seed=2020)
    generator = torch.Generator().manual_seed(2021)
    feature = torch.randn(2, 2, 3, 4, generator=generator)
    occupancy = torch.rand(
        2,
        1,
        6,
        8,
        generator=generator,
    ) > 0.57

    fields = model.forward_fields(feature, occupancy)
    projected_hidden = (
        fields.odd_feature_presence_hidden
        * model.scalar_energy_weight[None, None, :, None, None]
    ).sum(dim=2)
    projected_energy = binary_flip_odd_projection(
        fields.actual_feature_presence_energy.expand_as(
            fields.flipped_feature_presence_energy
        ),
        fields.flipped_feature_presence_energy,
    )

    torch.testing.assert_close(
        projected_hidden,
        fields.native_phase_interaction,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    torch.testing.assert_close(
        projected_energy,
        fields.native_phase_interaction,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("seed", "shape"),
    (
        (2005, (1, 2, 3, 4)),
        (2006, (2, 2, 2, 3)),
    ),
)
def test_efficient_binary_flip_matches_literal_local_reference(
    seed: int,
    shape: tuple[int, int, int, int],
) -> None:
    model = CURELiteBinaryFlipAntisymmetricLevelSet(
        _config(channels=2, stride=2, width=3)
    )
    _randomize_output_path(model, seed=seed)
    generator = torch.Generator().manual_seed(seed + 100)
    feature = torch.randn(shape, generator=generator)
    occupancy = torch.rand(
        shape[0],
        1,
        shape[2] * 2,
        shape[3] * 2,
        generator=generator,
    ) > 0.58

    efficient = model.forward_fields(feature, occupancy).field
    reference = model.forward_reference(feature, occupancy)
    torch.testing.assert_close(
        efficient,
        reference,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


@pytest.mark.parametrize(
    "occupancy_kind",
    ("empty", "single", "dense", "random"),
)
def test_zero_feature_is_exactly_silent_for_every_occupancy(
    occupancy_kind: str,
) -> None:
    model = CURELiteBinaryFlipAntisymmetricLevelSet(_config())
    _randomize_output_path(model, seed=2007)
    feature = torch.zeros(2, 2, 4, 4)
    if occupancy_kind == "empty":
        occupancy = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    elif occupancy_kind == "single":
        occupancy = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
        occupancy[0, 0, 3, 5] = True
        occupancy[1, 0, 6, 1] = True
    elif occupancy_kind == "dense":
        occupancy = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    else:
        occupancy = torch.rand(2, 1, 8, 8) > 0.43
    field = model(feature, occupancy)
    assert torch.equal(
        field,
        torch.full_like(field, CSLF_FIELD_AMPLITUDE),
    )


@pytest.mark.parametrize("zero_path", ("feature", "occupancy"))
def test_either_pure_path_cancels_with_nonzero_scalar_energy(
    zero_path: str,
) -> None:
    model = CURELiteBinaryFlipAntisymmetricLevelSet(_config())
    _randomize_output_path(model, seed=2008)
    with torch.no_grad():
        split = model.config.feature_channels
        if zero_path == "feature":
            model.joint_state_weight[:, :split].zero_()
        else:
            model.joint_state_weight[:, split:].zero_()
        assert bool(torch.any(model.scalar_energy_weight != 0.0))
    feature = torch.randn(2, 2, 4, 4)
    occupancy = torch.rand(2, 1, 8, 8) > 0.5
    assert torch.equal(
        model(feature, occupancy),
        torch.full_like(occupancy, CSLF_FIELD_AMPLITUDE, dtype=torch.float32),
    )


def test_affine_feature_presence_energy_matches_old_midpoint_exactly() -> None:
    occupancy = torch.tensor([0.0, 1.0], dtype=torch.float64)
    midpoint = torch.full_like(occupancy, 0.5)

    def energy(feature_present: float, phase: Tensor) -> Tensor:
        return (
            3.0
            + 2.0 * feature_present
            - 4.0 * phase
            + 5.0 * feature_present * phase
        )

    old_midpoint = centered_mixed_energy_difference(
        energy(1.0, occupancy),
        energy(0.0, occupancy),
        energy(1.0, midpoint),
        energy(0.0, midpoint),
    )
    actual_presence = (
        energy(1.0, occupancy) - energy(0.0, occupancy)
    )
    flipped_presence = (
        energy(1.0, 1.0 - occupancy)
        - energy(0.0, 1.0 - occupancy)
    )
    new_binary_flip = binary_flip_odd_projection(
        actual_presence,
        flipped_presence,
    )

    assert torch.equal(old_midpoint, new_binary_flip)
    assert torch.equal(
        new_binary_flip,
        5.0 * (occupancy - 0.5),
    )


def test_nonlinear_shared_energy_differs_from_old_and_changes_zero_level() -> None:
    new = CURELiteBinaryFlipAntisymmetricLevelSet(
        _config(channels=1, stride=1, width=1)
    )
    old = CURELiteCenteredMixedInteractionLevelSet(
        CoverageStateCenteredMixedInteractionConfig(
            feature_channels=1,
            feature_stride=1,
            width=1,
        )
    )
    center = new.config.coarse_radius
    with torch.no_grad():
        new.joint_state_weight.zero_()
        new.joint_hidden_bias.fill_(-3.0)
        new.joint_state_weight[0, 0, center, center] = 1.0
        new.joint_state_weight[0, 1, center, center] = 3.0
        new.scalar_energy_weight.fill_(3.0)
        old.load_state_dict(new.state_dict())
    feature = torch.ones(1, 1, 1, 1)
    vacant = torch.zeros(1, 1, 1, 1, dtype=torch.bool)

    old_field = old(feature, vacant).reshape(())
    new_field = new(feature, vacant).reshape(())
    assert old_field.item() == pytest.approx(0.3570115, abs=2.0e-6)
    assert new_field.item() == pytest.approx(-0.3407803, abs=2.0e-6)
    assert old_field > 0.0
    assert new_field < 0.0
    assert not bool(old.predict_completion(feature, vacant).item())
    assert bool(new.predict_completion(feature, vacant).item())


def test_phase_roundtrip_and_native_field_alignment_are_exact() -> None:
    model = CURELiteBinaryFlipAntisymmetricLevelSet(
        _config(channels=1, stride=2, width=2)
    )
    _randomize_output_path(model, seed=2009)
    feature = torch.randn(1, 1, 3, 4)
    occupancy = torch.rand(1, 1, 6, 8) > 0.5
    fields = model.forward_fields(feature, occupancy)
    phase = pixel_unshuffle_bool_occupancy(occupancy, stride=2)

    assert torch.equal(fields.phase_occupancy, phase)
    assert torch.equal(
        F.pixel_shuffle(phase.to(dtype=torch.float32), 2).to(
            dtype=torch.bool
        ),
        occupancy,
    )
    assert torch.equal(
        fields.field,
        F.pixel_shuffle(fields.native_phase_field, 2),
    )


def test_gradient_latency_and_post_step_reach_all_three_parameters() -> None:
    torch.manual_seed(2010)
    model = CURELiteBinaryFlipAntisymmetricLevelSet(_config())
    generator = torch.Generator().manual_seed(2011)
    feature = torch.randn(2, 2, 4, 4, generator=generator)
    occupancy = torch.rand(
        2,
        1,
        8,
        8,
        generator=generator,
    ) > 0.61
    first = model(feature, occupancy)
    assert torch.equal(first, model(feature, occupancy))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    first.square().mean().backward()
    assert model.scalar_energy_weight.grad is not None
    assert bool(torch.isfinite(model.scalar_energy_weight.grad).all())
    assert bool(torch.any(model.scalar_energy_weight.grad != 0.0))
    assert model.joint_state_weight.grad is not None
    assert model.joint_hidden_bias.grad is not None
    assert torch.count_nonzero(model.joint_state_weight.grad) == 0
    assert torch.count_nonzero(model.joint_hidden_bias.grad) == 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    loss = model(feature, occupancy).square().mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(torch.any(parameter.grad != 0.0))


def test_model_rejects_wrong_grid_dtype_and_config_class() -> None:
    model = CURELiteBinaryFlipAntisymmetricLevelSet(_config())
    feature = torch.randn(1, 2, 3, 3)
    occupancy = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="float32"):
        model(feature.to(torch.float64), occupancy)
    with pytest.raises(ValueError, match="feature_stride"):
        model(feature, occupancy[..., :-1, :])
    with pytest.raises(TypeError, match="config"):
        CURELiteBinaryFlipAntisymmetricLevelSet(  # type: ignore[arg-type]
            CoverageStateCenteredMixedInteractionConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )
