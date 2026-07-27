from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from cure_lite.coverage_state_centered_mixed_interaction import (
    CMIF_COARSE_RADIUS,
    CMIF_ENERGY_POLICY,
    CMIF_INPUT_REPRESENTATION,
    CMIF_INTERACTION_POLICY,
    CMIF_NEUTRAL_PHASE,
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from cure_lite.coverage_state_level_set import (
    CSLF_FIELD_AMPLITUDE,
    CURELiteCoverageStateLevelSet,
)
from cure_lite.coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
    build_coverage_state_level_set,
)


def _config(
    *,
    channels: int = 2,
    stride: int = 2,
    width: int = 4,
) -> CoverageStateCenteredMixedInteractionConfig:
    return CoverageStateCenteredMixedInteractionConfig(
        feature_channels=channels,
        feature_stride=stride,
        width=width,
    )


def _randomize_output_path(
    model: CURELiteCenteredMixedInteractionLevelSet,
    *,
    seed: int = 1701,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.joint_state_weight.copy_(
            torch.randn(
                model.joint_state_weight.shape,
                generator=generator,
            )
            * 0.12
        )
        model.joint_hidden_bias.copy_(
            torch.randn(
                model.joint_hidden_bias.shape,
                generator=generator,
            )
            * 0.08
        )
        model.scalar_energy_weight.copy_(
            torch.randn(
                model.scalar_energy_weight.shape,
                generator=generator,
            )
            * 0.2
        )


def test_cmif_config_freezes_the_single_mechanism_contract() -> None:
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    assert config.coverage_policy == (
        CSLF_PHASE_PRESERVING_COVERAGE_POLICY
    )
    assert config.input_representation == CMIF_INPUT_REPRESENTATION
    assert config.interaction_policy == CMIF_INTERACTION_POLICY
    assert config.energy_policy == CMIF_ENERGY_POLICY
    assert config.coarse_radius == CMIF_COARSE_RADIUS == 2
    assert config.neutral_phase == CMIF_NEUTRAL_PHASE == 0.5
    assert config.kernel_size == 5
    assert config.phase_occupancy_channels == 16
    assert config.expected_parameter_count == 64064
    for kwargs, message in (
        ({"coarse_radius": 1}, "coarse_radius"),
        ({"neutral_phase": 0.0}, "neutral_phase"),
        ({"coverage_policy": "changed"}, "coverage_policy"),
        ({"input_representation": "scalar_max"}, "input_representation"),
        ({"interaction_policy": "changed"}, "interaction_policy"),
        ({"energy_policy": "changed"}, "energy_policy"),
    ):
        with pytest.raises(ValueError, match=message):
            CoverageStateCenteredMixedInteractionConfig(
                feature_channels=4,
                feature_stride=2,
                width=3,
                **kwargs,
            )


def test_cmif_is_one_registered_level_set_with_three_parameters() -> None:
    config = _config()
    model = build_coverage_state_level_set(config)
    assert type(model) is CURELiteCenteredMixedInteractionLevelSet
    assert isinstance(model, CURELiteCoverageStateLevelSet)
    assert set(dict(model.named_parameters())) == {
        "joint_state_weight",
        "joint_hidden_bias",
        "scalar_energy_weight",
    }
    assert tuple(model.joint_state_weight.shape) == (4, 6, 5, 5)
    assert tuple(model.joint_hidden_bias.shape) == (4,)
    assert tuple(model.scalar_energy_weight.shape) == (4,)
    assert torch.count_nonzero(model.scalar_energy_weight) == 0
    assert sum(
        parameter.numel() for parameter in model.parameters()
    ) == config.expected_parameter_count
    assert not tuple(model.named_buffers())


def test_initial_field_is_exact_positive_anchor_and_completion_empty() -> None:
    model = CURELiteCenteredMixedInteractionLevelSet(_config())
    feature = torch.randn(2, 2, 4, 5)
    occupancy = torch.rand(2, 1, 8, 10) > 0.72
    fields = model.forward_fields(feature, occupancy)
    assert fields.phase_occupancy.shape == (2, 4, 4, 5)
    assert fields.neutral_delta.shape == (2, 4, 4, 4, 5)
    assert fields.mixed_hidden.shape == (2, 4, 4, 4, 5)
    assert fields.native_phase_field.shape == (2, 4, 4, 5)
    assert fields.field.shape == occupancy.shape
    assert torch.equal(
        fields.field,
        torch.full_like(fields.field, CSLF_FIELD_AMPLITUDE),
    )
    assert not bool(torch.any(model.predict_completion(feature, occupancy)))
    assert torch.equal(
        model.predict_union(feature, occupancy),
        occupancy,
    )


def test_efficient_center_column_equation_matches_literal_reference() -> None:
    model = CURELiteCenteredMixedInteractionLevelSet(
        _config(channels=2, stride=2, width=3)
    )
    _randomize_output_path(model)
    generator = torch.Generator().manual_seed(1702)
    feature = torch.randn(1, 2, 3, 4, generator=generator)
    occupancy = torch.rand(
        1,
        1,
        6,
        8,
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
    model = CURELiteCenteredMixedInteractionLevelSet(_config())
    _randomize_output_path(model, seed=1703)
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
def test_either_pure_path_is_annihilated_with_nonzero_energy(
    zero_path: str,
) -> None:
    model = CURELiteCenteredMixedInteractionLevelSet(_config())
    _randomize_output_path(model, seed=1704)
    with torch.no_grad():
        split = model.config.feature_channels
        if zero_path == "feature":
            model.joint_state_weight[:, :split].zero_()
        else:
            model.joint_state_weight[:, split:].zero_()
        assert bool(torch.any(model.scalar_energy_weight != 0.0))
    feature = torch.randn(2, 2, 4, 4)
    occupancy = torch.rand(2, 1, 8, 8) > 0.5
    field = model(feature, occupancy)
    assert torch.equal(
        field,
        torch.full_like(field, CSLF_FIELD_AMPLITUDE),
    )


def _endpoint_basis(
    model: CURELiteCenteredMixedInteractionLevelSet,
) -> Tensor:
    feature = torch.ones(1, 1, 1, 1)
    fields: list[Tensor] = []
    for hidden in range(2):
        with torch.no_grad():
            model.scalar_energy_weight.zero_()
            model.scalar_energy_weight[hidden] = 1.0
        endpoints: list[Tensor] = []
        for occupied in (False, True):
            occupancy = torch.full(
                (1, 1, 1, 1),
                occupied,
                dtype=torch.bool,
            )
            endpoints.append(
                model.forward_fields(
                    feature,
                    occupancy,
                ).native_phase_interaction.reshape(())
            )
        fields.append(torch.stack(endpoints))
    return torch.stack(fields)


def test_cmif_has_asymmetric_rank_two_endpoint_basis() -> None:
    model = CURELiteCenteredMixedInteractionLevelSet(
        _config(channels=1, stride=1, width=2)
    )
    with torch.no_grad():
        model.joint_state_weight.zero_()
        model.joint_hidden_bias.copy_(torch.tensor([0.0, -0.5]))
        center = model.config.coarse_radius
        model.joint_state_weight[0, 0, center, center] = 1.0
        model.joint_state_weight[0, 1, center, center] = 1.0
        model.joint_state_weight[1, 0, center, center] = 1.0
        model.joint_state_weight[1, 1, center, center] = 2.0
    basis = _endpoint_basis(model)
    assert not torch.isclose(basis[0].sum(), torch.zeros(()))
    determinant = torch.linalg.det(basis)
    assert abs(float(determinant.detach())) > 1.0e-3
    assert torch.linalg.matrix_rank(basis).item() == 2


@pytest.mark.parametrize(
    ("target_interaction", "readout"),
    (
        (-1.125, (0.631087, 0.082765)),
        (-1.350, (0.757304, 0.099317)),
        (-1.575, (0.883522, 0.115870)),
    ),
)
def test_rank_two_basis_constructs_the_correct_deletion_endpoint_direction(
    target_interaction: float,
    readout: tuple[float, float],
) -> None:
    model = CURELiteCenteredMixedInteractionLevelSet(
        _config(channels=1, stride=1, width=2)
    )
    center = model.config.coarse_radius
    with torch.no_grad():
        model.joint_state_weight.zero_()
        model.joint_hidden_bias.copy_(torch.tensor([2.0, -1.0]))
        model.joint_state_weight[0, 0, center, center] = -4.0
        model.joint_state_weight[0, 1, center, center] = -4.0
        model.joint_state_weight[1, 0, center, center] = -4.0
        model.joint_state_weight[1, 1, center, center] = 4.0
        model.scalar_energy_weight.copy_(torch.tensor(readout))
    feature = torch.ones(1, 1, 1, 1)
    vacant = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
    covered = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    vacant_interaction = model.forward_fields(
        feature,
        vacant,
    ).native_phase_interaction.item()
    covered_interaction = model.forward_fields(
        feature,
        covered,
    ).native_phase_interaction.item()
    assert vacant_interaction == pytest.approx(
        target_interaction,
        abs=2.0e-5,
    )
    assert covered_interaction == pytest.approx(0.0, abs=2.0e-5)
    assert model(feature, vacant).item() < 0.0
    assert model(feature, covered).item() == pytest.approx(
        CSLF_FIELD_AMPLITUDE,
        abs=2.0e-5,
    )


def test_cmif_radius_two_is_exact_and_not_accidentally_smaller() -> None:
    config = _config(channels=1, stride=1, width=1)
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    radius = config.coarse_radius
    with torch.no_grad():
        model.joint_state_weight.zero_()
        model.joint_hidden_bias.zero_()
        model.scalar_energy_weight.fill_(1.0)
        model.joint_state_weight[0, 0, 0, 0] = 0.7
        model.joint_state_weight[0, 1, radius, radius] = 1.1
    occupancy = torch.zeros(1, 1, 9, 9, dtype=torch.bool)
    feature_zero = torch.zeros(1, 1, 9, 9)
    feature_source = feature_zero.clone()
    feature_source[0, 0, 4, 4] = 1.0
    baseline = model.forward_fields(
        feature_zero,
        occupancy,
    ).native_phase_interaction
    changed = model.forward_fields(
        feature_source,
        occupancy,
    ).native_phase_interaction
    delta = changed - baseline
    nonzero = torch.nonzero(delta[0, 0] != 0.0, as_tuple=False)
    assert nonzero.tolist() == [[6, 6]]
    assert abs(6 - 4) == radius


def test_fixed_inputs_replay_bitwise_and_backward_reaches_all_parameters() -> None:
    model = CURELiteCenteredMixedInteractionLevelSet(_config())
    generator = torch.Generator().manual_seed(1705)
    feature = torch.randn(2, 2, 4, 4, generator=generator)
    occupancy = torch.rand(
        2,
        1,
        8,
        8,
        generator=generator,
    ) > 0.61
    first = model(feature, occupancy)
    second = model(feature, occupancy)
    assert torch.equal(first, second)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    first.square().mean().backward()
    assert model.scalar_energy_weight.grad is not None
    assert bool(torch.isfinite(model.scalar_energy_weight.grad).all())
    assert bool(torch.any(model.scalar_energy_weight.grad != 0.0))
    assert model.joint_state_weight.grad is not None
    assert model.joint_hidden_bias.grad is not None
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    loss = model(feature, occupancy).square().mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(torch.any(parameter.grad != 0.0))


def test_model_rejects_wrong_grid_dtype_and_unregistered_config() -> None:
    model = CURELiteCenteredMixedInteractionLevelSet(_config())
    feature = torch.randn(1, 2, 3, 3)
    occupancy = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="float32"):
        model(feature.to(torch.float64), occupancy)
    with pytest.raises(ValueError, match="feature_stride"):
        model(feature, occupancy[..., :-1, :])
    with pytest.raises(TypeError, match="registered"):
        build_coverage_state_level_set(object())  # type: ignore[arg-type]


def test_pixelshuffle_field_alignment_remains_exact() -> None:
    model = CURELiteCenteredMixedInteractionLevelSet(
        _config(channels=1, stride=2, width=2)
    )
    _randomize_output_path(model, seed=1706)
    feature = torch.randn(1, 1, 3, 4)
    occupancy = torch.rand(1, 1, 6, 8) > 0.5
    fields = model.forward_fields(feature, occupancy)
    assert torch.equal(
        fields.field,
        F.pixel_shuffle(fields.native_phase_field, 2),
    )
