"""Dataset-free structural gate for the v17 CMIF completion field.

The gate exercises only generated tensors.  It has no dataset, cache,
training, calibration, ``D_V``, or ``D_T`` entry point.  Its receipt binds
the exact equation, phase convention, parameter contract, algebraic nulls,
endpoint capacity, locality, gradient latency, and deterministic replay that
must hold before a frozen real-``D_R`` representability audit is permitted.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_centered_mixed_interaction import (
    CMIF_COARSE_RADIUS,
    CMIF_ENERGY_POLICY,
    CMIF_INPUT_REPRESENTATION,
    CMIF_INTERACTION_POLICY,
    CMIF_NEUTRAL_PHASE,
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
    centered_mixed_energy_difference,
)
from ..coverage_state_level_set import (
    CSLF_FEATURE_POLICY,
    CSLF_FIELD_AMPLITUDE,
    CSLF_NORMALIZATION_EPSILON,
    CSLF_NUMERICAL_POLICY,
    CSLF_OUTPUT_POLICY,
    CSLF_TARGET_POLICY,
)
from ..coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
    pixel_unshuffle_bool_occupancy,
)
from ..paired_types import tensor_content_fingerprint


COVERAGE_STATE_CMIF_DATASET_FREE_SCHEMA = (
    "cure-lite-cmif-v17-dataset-free-receipt-v1"
)
COVERAGE_STATE_CMIF_DATASET_FREE_SEEDS = (42, 43, 44)
COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS = 64
COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE = 4
COVERAGE_STATE_CMIF_FORMAL_WIDTH = 32
COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT = 64064
COVERAGE_STATE_CMIF_DATASET_FREE_EXECUTION_SEED = 170017
COVERAGE_STATE_CMIF_REFERENCE_MATRIX = (
    (42, (2, 3)),
    (43, (3, 2)),
    (44, (3, 3)),
)
COVERAGE_STATE_CMIF_REFERENCE_ATOL = 2.0e-6
COVERAGE_STATE_CMIF_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_phase_preserving.py",
    "cure_lite/coverage_state_centered_mixed_interaction.py",
    "cure_lite/experiment/coverage_state_cmif_dataset_free.py",
)


def _current_implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_CMIF_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"CMIF implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _formal_config() -> CoverageStateCenteredMixedInteractionConfig:
    return CoverageStateCenteredMixedInteractionConfig(
        feature_channels=COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_CMIF_FORMAL_WIDTH,
    )


def _toy_config(
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


def _randomize_model(
    model: CURELiteCenteredMixedInteractionLevelSet,
    *,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.joint_state_weight.copy_(
            torch.randn(
                model.joint_state_weight.shape,
                generator=generator,
            )
            * 0.1
        )
        model.joint_hidden_bias.copy_(
            torch.randn(
                model.joint_hidden_bias.shape,
                generator=generator,
            )
            * 0.05
        )
        model.scalar_energy_weight.copy_(
            torch.randn(
                model.scalar_energy_weight.shape,
                generator=generator,
            )
            * 0.2
        )


def _reference_probes() -> tuple[dict[str, object], ...]:
    results: list[dict[str, object]] = []
    for seed, shape in COVERAGE_STATE_CMIF_REFERENCE_MATRIX:
        config = _toy_config(stride=4, width=3)
        model = CURELiteCenteredMixedInteractionLevelSet(config)
        _randomize_model(model, seed=seed)
        generator = torch.Generator().manual_seed(seed + 1700)
        feature = torch.randn(
            1,
            config.feature_channels,
            shape[0],
            shape[1],
            generator=generator,
        )
        occupancy = torch.rand(
            1,
            1,
            shape[0] * config.feature_stride,
            shape[1] * config.feature_stride,
            generator=generator,
        ) > 0.57
        efficient = model(feature, occupancy)
        reference = model.forward_reference(feature, occupancy)
        maximum_error = float(
            (efficient - reference).abs().max().detach()
        )
        zero_field = model(torch.zeros_like(feature), occupancy)
        results.append(
            {
                "seed": seed,
                "feature_size": list(shape),
                "coarse_cell_count": shape[0] * shape[1],
                "phase_output_count": (
                    shape[0]
                    * shape[1]
                    * config.phase_occupancy_channels
                ),
                "reference_energy_evaluation_count": (
                    shape[0]
                    * shape[1]
                    * config.phase_occupancy_channels
                    * 4
                ),
                "efficient_full_grid_convolution_count": 2,
                "model_state_fingerprint": stable_fingerprint(
                    {
                        name: tensor_content_fingerprint(value)
                        for name, value in sorted(
                            model.state_dict().items()
                        )
                    }
                ),
                "feature_sha256": tensor_content_fingerprint(feature),
                "occupancy_sha256": tensor_content_fingerprint(
                    occupancy
                ),
                "efficient_sha256": tensor_content_fingerprint(efficient),
                "reference_sha256": tensor_content_fingerprint(reference),
                "zero_field_sha256": tensor_content_fingerprint(
                    zero_field
                ),
                "maximum_absolute_error_hex": maximum_error.hex(),
                "reference_close": bool(
                    torch.allclose(
                        efficient,
                        reference,
                        rtol=2.0e-5,
                        atol=COVERAGE_STATE_CMIF_REFERENCE_ATOL,
                    )
                ),
                "finite": bool(
                    torch.isfinite(efficient).all()
                    and torch.isfinite(reference).all()
                ),
                "zero_feature_exact": torch.equal(
                    zero_field,
                    torch.full_like(
                        zero_field,
                        CSLF_FIELD_AMPLITUDE,
                    ),
                ),
            }
        )
    return tuple(results)


def _phase_contract_probe() -> dict[str, object]:
    stride = COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE
    occupancy = torch.zeros(
        1,
        1,
        3 * stride,
        4 * stride,
        dtype=torch.bool,
    )
    expected_indices: list[list[int]] = []
    observed_indices: list[list[int]] = []
    for phase_y in range(stride):
        for phase_x in range(stride):
            value = occupancy.clone()
            row = stride + phase_y
            column = 2 * stride + phase_x
            value[0, 0, row, column] = True
            phase = pixel_unshuffle_bool_occupancy(
                value,
                stride=stride,
            )
            expected_indices.append(
                [0, phase_y * stride + phase_x, 1, 2]
            )
            observed_indices.append(
                torch.nonzero(phase, as_tuple=False)[0].tolist()
            )
            if not torch.equal(
                F.pixel_shuffle(
                    phase.to(dtype=torch.float32),
                    stride,
                ).to(dtype=torch.bool),
                value,
            ):
                raise AssertionError("CMIF phase roundtrip changed occupancy")
    return {
        "phase_count": stride**2,
        "expected_indices": expected_indices,
        "observed_indices": observed_indices,
        "row_major_exact": observed_indices == expected_indices,
        "roundtrip_exact": True,
    }


def _center_only_probe() -> dict[str, object]:
    config = _toy_config(channels=1, stride=2, width=2)
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    _randomize_model(model, seed=1801)
    feature = torch.randn(1, 1, 4, 5)
    occupancy = torch.zeros(1, 1, 8, 10, dtype=torch.bool)
    occupancy[0, 0, 2, 3] = True
    occupancy[0, 0, 6, 8] = True
    original = model.forward_fields(feature, occupancy).neutral_delta
    phase = pixel_unshuffle_bool_occupancy(
        occupancy,
        stride=config.feature_stride,
    )
    center = config.coarse_radius
    center_columns = model.occupancy_weight[
        :,
        :,
        center,
        center,
    ].transpose(0, 1)
    expected_delta = (
        (
            config.neutral_phase
            - phase.to(dtype=torch.float32)
        ).unsqueeze(2)
        * center_columns[None, :, :, None, None]
    )
    midpoint = phase.to(dtype=torch.float32).clone()
    chosen_phase, chosen_row, chosen_column = 2, 1, 3
    midpoint[
        0,
        chosen_phase,
        chosen_row,
        chosen_column,
    ] = config.neutral_phase
    midpoint_difference = (
        midpoint - phase.to(dtype=torch.float32)
    )
    changed_phase = phase.clone()
    changed_phase[0, 1, 0, 0] = ~changed_phase[0, 1, 0, 0]
    changed_occupancy = F.pixel_shuffle(
        changed_phase.to(dtype=torch.float32),
        config.feature_stride,
    ).to(dtype=torch.bool)
    changed = model.forward_fields(
        feature,
        changed_occupancy,
    ).neutral_delta
    unaffected = torch.ones_like(original, dtype=torch.bool)
    unaffected[:, 1, :, 0, 0] = False

    encoded = model.forward_fields(feature, occupancy).encoded_feature
    phase_float = phase.to(dtype=torch.float32)
    feature_affine = F.conv2d(
        encoded,
        model.feature_weight,
        padding=config.coarse_radius,
    )
    occupancy_affine = F.conv2d(
        phase_float,
        model.occupancy_weight,
        bias=model.joint_hidden_bias,
        padding=config.coarse_radius,
    )
    base_contrast = (
        F.silu(feature_affine + occupancy_affine)
        - F.silu(occupancy_affine)
    )
    base_energy = (
        base_contrast
        * model.scalar_energy_weight[None, :, None, None]
    ).sum(dim=1)
    wrong_native = torch.empty(
        1,
        config.phase_occupancy_channels,
        phase.shape[-2],
        phase.shape[-1],
    )
    for phase_index in range(config.phase_occupancy_channels):
        globally_neutral = phase_float.clone()
        globally_neutral[:, phase_index] = config.neutral_phase
        globally_neutral_affine = F.conv2d(
            globally_neutral,
            model.occupancy_weight,
            bias=model.joint_hidden_bias,
            padding=config.coarse_radius,
        )
        globally_neutral_contrast = (
            F.silu(feature_affine + globally_neutral_affine)
            - F.silu(globally_neutral_affine)
        )
        globally_neutral_energy = (
            globally_neutral_contrast
            * model.scalar_energy_weight[None, :, None, None]
        ).sum(dim=1)
        wrong_native[:, phase_index] = (
            config.field_amplitude
            + base_energy
            - globally_neutral_energy
        )
    wrong_field = F.pixel_shuffle(
        wrong_native,
        config.feature_stride,
    )
    correct_field = model(feature, occupancy)
    return {
        "changed_coordinate": [0, 1, 0, 0],
        "neutral_phase_hex": config.neutral_phase.hex(),
        "delta_formula_exact": torch.equal(original, expected_delta),
        "midpoint_changed_element_count": int(
            torch.count_nonzero(midpoint_difference).item()
        ),
        "midpoint_selected_value_hex": float(
            midpoint[
                0,
                chosen_phase,
                chosen_row,
                chosen_column,
            ]
        ).hex(),
        "changed_coordinate_differs": bool(
            torch.any(original[:, 1, :, 0, 0] != changed[:, 1, :, 0, 0])
        ),
        "all_other_coordinates_exact": torch.equal(
            original[unaffected],
            changed[unaffected],
        ),
        "global_phase_replacement_differs": bool(
            torch.any(correct_field != wrong_field)
        ),
        "global_phase_replacement_max_difference_hex": float(
            (correct_field - wrong_field).abs().max().detach()
        ).hex(),
    }


def _zero_and_pure_path_probe() -> dict[str, object]:
    config = _toy_config()
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    _randomize_model(model, seed=1802)
    patterns: dict[str, Tensor] = {
        "empty": torch.zeros(1, 1, 8, 8, dtype=torch.bool),
        "dense": torch.ones(1, 1, 8, 8, dtype=torch.bool),
        "single": torch.zeros(1, 1, 8, 8, dtype=torch.bool),
        "multi_component": torch.zeros(
            1,
            1,
            8,
            8,
            dtype=torch.bool,
        ),
        "random": torch.rand(1, 1, 8, 8) > 0.53,
    }
    patterns["single"][0, 0, 3, 5] = True
    patterns["multi_component"][0, 0, 1:3, 1:3] = True
    patterns["multi_component"][0, 0, 6, 6] = True
    zero_feature = torch.zeros(1, 2, 4, 4)
    zero_results = {
        name: torch.equal(
            model(zero_feature, occupancy),
            torch.full(
                occupancy.shape,
                CSLF_FIELD_AMPLITUDE,
                dtype=torch.float32,
            ),
        )
        for name, occupancy in patterns.items()
    }
    feature = torch.randn(1, 2, 4, 4)
    occupancy = patterns["random"]
    feature_null = CURELiteCenteredMixedInteractionLevelSet(config)
    occupancy_null = CURELiteCenteredMixedInteractionLevelSet(config)
    _randomize_model(feature_null, seed=1803)
    occupancy_null.load_state_dict(feature_null.state_dict(), strict=True)
    with torch.no_grad():
        split = config.feature_channels
        feature_null.joint_state_weight[:, :split].zero_()
        occupancy_null.joint_state_weight[:, split:].zero_()
    anchor = torch.full(
        occupancy.shape,
        CSLF_FIELD_AMPLITUDE,
        dtype=torch.float32,
    )
    return {
        "zero_feature_patterns": zero_results,
        "zero_feature_all_exact": all(zero_results.values()),
        "feature_weight_zero_exact": torch.equal(
            feature_null(feature, occupancy),
            anchor,
        ),
        "occupancy_weight_zero_exact": torch.equal(
            occupancy_null(feature, occupancy),
            anchor,
        ),
        "nonzero_energy_readout": bool(
            torch.any(feature_null.scalar_energy_weight != 0.0)
            and torch.any(occupancy_null.scalar_energy_weight != 0.0)
        ),
    }


def _gauge_probe() -> dict[str, object]:
    generator = torch.Generator().manual_seed(1804)
    values = tuple(
        torch.randn(2, 3, generator=generator) for _ in range(4)
    )
    baseline = centered_mixed_energy_difference(*values)
    feature_b = torch.randn(2, 3, generator=generator)
    feature_zero = torch.randn(2, 3, generator=generator)
    occupancy_u = torch.randn(2, 3, generator=generator)
    occupancy_mid = torch.randn(2, 3, generator=generator)
    constant = torch.full((2, 3), 0.375)
    feature_gauge = centered_mixed_energy_difference(
        values[0] + feature_b,
        values[1] + feature_zero,
        values[2] + feature_b,
        values[3] + feature_zero,
    )
    occupancy_gauge = centered_mixed_energy_difference(
        values[0] + occupancy_u,
        values[1] + occupancy_u,
        values[2] + occupancy_mid,
        values[3] + occupancy_mid,
    )
    constant_gauge = centered_mixed_energy_difference(
        *(value + constant for value in values)
    )
    return {
        "feature_additive_invariant": bool(
            torch.allclose(baseline, feature_gauge, atol=5.0e-7, rtol=0.0)
        ),
        "occupancy_additive_invariant": bool(
            torch.allclose(
                baseline,
                occupancy_gauge,
                atol=5.0e-7,
                rtol=0.0,
            )
        ),
        "constant_invariant": bool(
            torch.allclose(
                baseline,
                constant_gauge,
                atol=5.0e-7,
                rtol=0.0,
            )
        ),
    }


def _endpoint_probe() -> dict[str, object]:
    config = _toy_config(channels=1, stride=1, width=2)
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    center = config.coarse_radius
    with torch.no_grad():
        model.joint_state_weight.zero_()
        model.joint_hidden_bias.copy_(torch.tensor([2.0, -1.0]))
        model.joint_state_weight[0, 0, center, center] = -4.0
        model.joint_state_weight[0, 1, center, center] = -4.0
        model.joint_state_weight[1, 0, center, center] = -4.0
        model.joint_state_weight[1, 1, center, center] = 4.0
    feature = torch.ones(1, 1, 1, 1)
    vacant = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
    covered = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    basis_rows: list[Tensor] = []
    for hidden in range(2):
        with torch.no_grad():
            model.scalar_energy_weight.zero_()
            model.scalar_energy_weight[hidden] = 1.0
        basis_rows.append(
            torch.stack(
                (
                    model.forward_fields(
                        feature,
                        vacant,
                    ).native_phase_interaction.reshape(()),
                    model.forward_fields(
                        feature,
                        covered,
                    ).native_phase_interaction.reshape(()),
                )
            )
        )
    basis = torch.stack(basis_rows)
    determinant = float(torch.linalg.det(basis).detach())
    with torch.no_grad():
        model.scalar_energy_weight.copy_(
            torch.tensor([0.631087, 0.082765])
        )
    vacant_value = float(
        model.forward_fields(
            feature,
            vacant,
        ).native_phase_interaction.detach().reshape(())
    )
    covered_value = float(
        model.forward_fields(
            feature,
            covered,
        ).native_phase_interaction.detach().reshape(())
    )
    return {
        "basis": [
            [float(value.detach()).hex() for value in row]
            for row in basis
        ],
        "determinant_hex": determinant.hex(),
        "rank": int(torch.linalg.matrix_rank(basis).item()),
        "non_antisymmetric": bool(
            not torch.isclose(basis[0].sum(), torch.zeros(()))
        ),
        "vacant_interaction_hex": vacant_value.hex(),
        "covered_interaction_hex": covered_value.hex(),
        "correct_deletion_direction": (
            abs(vacant_value + 1.125) <= 2.0e-5
            and abs(covered_value) <= 2.0e-5
            and CSLF_FIELD_AMPLITUDE + vacant_value < 0.0
        ),
    }


def _locality_probe() -> dict[str, object]:
    config = _toy_config(channels=1, stride=1, width=1)
    radius = config.coarse_radius
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    with torch.no_grad():
        model.joint_state_weight.zero_()
        model.joint_hidden_bias.fill_(0.1)
        model.scalar_energy_weight.fill_(1.0)
        model.joint_state_weight[0, 0, radius, radius] = 1.0
        model.joint_state_weight[0, 1, radius, radius] = 1.1
        model.joint_state_weight[0, 1, 1, 1] = 0.5
        model.joint_state_weight[0, 1, 0, 0] = 0.7
    feature = torch.ones(1, 1, 9, 9)
    empty = torch.zeros(1, 1, 9, 9, dtype=torch.bool)
    changed = empty.clone()
    changed[0, 0, 4, 4] = True
    before = model.forward_fields(
        feature,
        empty,
    ).native_phase_interaction
    after = model.forward_fields(
        feature,
        changed,
    ).native_phase_interaction
    delta = after - before
    rows = torch.arange(9).view(9, 1)
    columns = torch.arange(9).view(1, 9)
    outside = torch.maximum(
        (rows - 4).abs(),
        (columns - 4).abs(),
    ) > radius
    occupancy_radius_two_witness = bool(delta[0, 0, 6, 6] != 0.0)

    feature_model = CURELiteCenteredMixedInteractionLevelSet(config)
    with torch.no_grad():
        feature_model.joint_state_weight.zero_()
        feature_model.joint_hidden_bias.zero_()
        feature_model.scalar_energy_weight.fill_(1.0)
        feature_model.joint_state_weight[0, 0, 0, 0] = 0.7
        feature_model.joint_state_weight[0, 0, 1, 1] = 0.5
        feature_model.joint_state_weight[
            0,
            1,
            radius,
            radius,
        ] = 1.1
    zero_feature = torch.zeros(1, 1, 9, 9)
    source_feature = zero_feature.clone()
    source_feature[0, 0, 4, 4] = 1.0
    feature_delta = (
        feature_model.forward_fields(
            source_feature,
            empty,
        ).native_phase_interaction
        - feature_model.forward_fields(
            zero_feature,
            empty,
        ).native_phase_interaction
    )

    cross_config = _toy_config(channels=1, stride=2, width=1)
    cross_model = CURELiteCenteredMixedInteractionLevelSet(cross_config)
    cross_radius = cross_config.coarse_radius
    output_phase = 0
    neighbour_phase = 3
    with torch.no_grad():
        cross_model.joint_state_weight.zero_()
        cross_model.joint_hidden_bias.fill_(0.1)
        cross_model.scalar_energy_weight.fill_(1.0)
        cross_model.joint_state_weight[
            0,
            0,
            cross_radius,
            cross_radius,
        ] = 1.0
        occupancy_offset = cross_config.feature_channels
        cross_model.joint_state_weight[
            0,
            occupancy_offset + output_phase,
            cross_radius,
            cross_radius,
        ] = 1.1
        cross_model.joint_state_weight[
            0,
            occupancy_offset + neighbour_phase,
            1,
            1,
        ] = 0.8
    cross_feature = torch.ones(1, 1, 7, 7)
    cross_empty = torch.zeros(1, 1, 14, 14, dtype=torch.bool)
    cross_changed = cross_empty.clone()
    source_coarse = (3, 3)
    source_phase_offset = (1, 1)
    cross_changed[
        0,
        0,
        source_coarse[0] * 2 + source_phase_offset[0],
        source_coarse[1] * 2 + source_phase_offset[1],
    ] = True
    cross_before = cross_model.forward_fields(
        cross_feature,
        cross_empty,
    ).native_phase_interaction
    cross_after = cross_model.forward_fields(
        cross_feature,
        cross_changed,
    ).native_phase_interaction
    cross_output_coarse = (4, 4)
    cross_phase_delta = (
        cross_after - cross_before
    )[0, output_phase, cross_output_coarse[0], cross_output_coarse[1]]
    return {
        "occupancy_outside_radius_two_exact": bool(
            torch.all(delta[0, 0][outside] == 0.0)
        ),
        "occupancy_radius_two_witness": occupancy_radius_two_witness,
        "occupancy_radius_one_witness": bool(
            delta[0, 0, 5, 5] != 0.0
        ),
        "feature_outside_radius_two_exact": bool(
            torch.all(feature_delta[0, 0][outside] == 0.0)
        ),
        "feature_radius_two_witness": bool(
            feature_delta[0, 0, 6, 6] != 0.0
        ),
        "feature_radius_one_witness": bool(
            feature_delta[0, 0, 5, 5] != 0.0
        ),
        "occupancy_nonzero_count": int(
            torch.count_nonzero(delta).item()
        ),
        "feature_nonzero_count": int(
            torch.count_nonzero(feature_delta).item()
        ),
        "cross_phase_output_phase": output_phase,
        "cross_phase_neighbour_phase": neighbour_phase,
        "cross_phase_neighbour_cell_differs": (
            source_coarse != cross_output_coarse
        ),
        "cross_phase_neighbour_witness": bool(
            cross_phase_delta != 0.0
        ),
    }


def _gradient_and_identity_probe() -> dict[str, object]:
    config = _toy_config()
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    generator = torch.Generator().manual_seed(1805)
    feature = torch.randn(2, 2, 4, 4, generator=generator)
    occupancy = torch.rand(
        2,
        1,
        8,
        8,
        generator=generator,
    ) > 0.49
    initial = model(feature, occupancy)
    replay = model(feature, occupancy)
    identity_response = initial - replay
    initial_anchor = torch.full_like(initial, CSLF_FIELD_AMPLITUDE)
    initial_completion = model.predict_completion(feature, occupancy)
    initial_union = model.predict_union(feature, occupancy)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    initial.square().mean().backward()
    update_zero = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        update_zero[name] = {
            "present": gradient is not None,
            "finite": bool(
                gradient is not None
                and torch.isfinite(gradient).all()
            ),
            "nonzero": bool(
                gradient is not None
                and torch.any(gradient != 0.0)
            ),
            "exact_zero": bool(
                gradient is not None
                and torch.all(gradient == 0.0)
            ),
        }
    update_zero_gradient_fingerprints = {
        name: tensor_content_fingerprint(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(feature, occupancy).square().mean().backward()
    update_one = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        update_one[name] = {
            "present": gradient is not None,
            "finite": bool(
                gradient is not None
                and torch.isfinite(gradient).all()
            ),
            "nonzero": bool(
                gradient is not None
                and torch.any(gradient != 0.0)
            ),
        }
    update_one_gradient_fingerprints = {
        name: tensor_content_fingerprint(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    nondegenerate = CURELiteCenteredMixedInteractionLevelSet(config)
    _randomize_model(nondegenerate, seed=1806)
    nondegenerate.zero_grad(set_to_none=True)
    nondegenerate(feature, occupancy).square().mean().backward()
    artificial_nonzero = {
        name: bool(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.any(parameter.grad != 0.0)
        )
        for name, parameter in nondegenerate.named_parameters()
    }
    return {
        "fixed_input_replay_exact": torch.equal(initial, replay),
        "identity_response_exact": bool(
            torch.all(identity_response == 0.0)
        ),
        "initial_field_exact": torch.equal(initial, initial_anchor),
        "initial_completion_empty": not bool(
            torch.any(initial_completion)
        ),
        "initial_union_preserves_occupancy": torch.equal(
            initial_union,
            occupancy,
        ),
        "update_zero_nonzero_gradient": update_zero,
        "update_one_nonzero_gradient": update_one,
        "feature_sha256": tensor_content_fingerprint(feature),
        "occupancy_sha256": tensor_content_fingerprint(occupancy),
        "initial_field_sha256": tensor_content_fingerprint(initial),
        "update_zero_gradient_fingerprints": (
            update_zero_gradient_fingerprints
        ),
        "update_one_gradient_fingerprints": (
            update_one_gradient_fingerprints
        ),
        "artificial_nonzero_readout_gradients": artificial_nonzero,
        "gradient_latency_passed": (
            update_zero["scalar_energy_weight"]["present"]
            and update_zero["scalar_energy_weight"]["finite"]
            and update_zero["scalar_energy_weight"]["nonzero"]
            and update_zero["joint_state_weight"]["present"]
            and update_zero["joint_state_weight"]["finite"]
            and update_zero["joint_state_weight"]["exact_zero"]
            and update_zero["joint_hidden_bias"]["present"]
            and update_zero["joint_hidden_bias"]["finite"]
            and update_zero["joint_hidden_bias"]["exact_zero"]
            and update_one["joint_state_weight"]["present"]
            and update_one["joint_state_weight"]["finite"]
            and update_one["joint_state_weight"]["nonzero"]
            and update_one["joint_hidden_bias"]["present"]
            and update_one["joint_hidden_bias"]["finite"]
            and update_one["joint_hidden_bias"]["nonzero"]
            and update_one["scalar_energy_weight"]["present"]
            and update_one["scalar_energy_weight"]["finite"]
            and update_one["scalar_energy_weight"]["nonzero"]
            and all(artificial_nonzero.values())
        ),
    }


def recompute_coverage_state_cmif_dataset_free_checks(
    *,
    formal_config_payload: dict[str, object],
    parameter_names: tuple[str, ...],
    parameter_contract: tuple[
        tuple[str, tuple[int, ...], str, bool],
        ...,
    ],
    implementation_binding: tuple[tuple[str, str], ...],
    reference_probes: tuple[dict[str, object], ...],
    phase_probe: dict[str, object],
    center_probe: dict[str, object],
    null_probe: dict[str, object],
    gauge_probe: dict[str, object],
    endpoint_probe: dict[str, object],
    locality_probe: dict[str, object],
    gradient_probe: dict[str, object],
) -> tuple[tuple[str, bool], ...]:
    """Recompute every authorization bit from canonical generated evidence."""

    expected_names = (
        "joint_hidden_bias",
        "joint_state_weight",
        "scalar_energy_weight",
    )
    expected_parameter_contract = (
        ("joint_hidden_bias", (32,), "torch.float32", True),
        (
            "joint_state_weight",
            (32, 80, 5, 5),
            "torch.float32",
            True,
        ),
        ("scalar_energy_weight", (32,), "torch.float32", True),
    )
    expected_reference_matrix = tuple(
        (seed, list(shape))
        for seed, shape in COVERAGE_STATE_CMIF_REFERENCE_MATRIX
    )
    actual_reference_matrix = tuple(
        (value["seed"], value["feature_size"])
        for value in reference_probes
    )
    checks = {
        "formal_config_exact": (
            formal_config_payload["model_class"]
            == CURELiteCenteredMixedInteractionLevelSet.__name__
            and formal_config_payload["feature_channels"]
            == COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS
            and formal_config_payload["feature_stride"]
            == COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE
            and formal_config_payload["width"]
            == COVERAGE_STATE_CMIF_FORMAL_WIDTH
            and formal_config_payload["coarse_radius"]
            == CMIF_COARSE_RADIUS
            and formal_config_payload["neutral_phase_hex"]
            == float(CMIF_NEUTRAL_PHASE).hex()
            and formal_config_payload["expected_parameter_count"]
            == COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
            and formal_config_payload["actual_parameter_count"]
            == COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
            and formal_config_payload["phase_channels"] == 16
            and formal_config_payload["kernel_size"] == 5
            and formal_config_payload["field_amplitude_hex"]
            == float(CSLF_FIELD_AMPLITUDE).hex()
            and formal_config_payload["normalization_epsilon_hex"]
            == float(CSLF_NORMALIZATION_EPSILON).hex()
            and formal_config_payload["feature_policy"]
            == CSLF_FEATURE_POLICY
            and formal_config_payload["target_policy"]
            == CSLF_TARGET_POLICY
            and formal_config_payload["output_policy"]
            == CSLF_OUTPUT_POLICY
            and formal_config_payload["numerical_policy"]
            == CSLF_NUMERICAL_POLICY
            and formal_config_payload["coverage_policy"]
            == CSLF_PHASE_PRESERVING_COVERAGE_POLICY
            and formal_config_payload["input_representation"]
            == CMIF_INPUT_REPRESENTATION
            and formal_config_payload["field_policy"]
            == CMIF_INTERACTION_POLICY
            and formal_config_payload["interaction_policy"]
            == CMIF_INTERACTION_POLICY
            and formal_config_payload["energy_policy"]
            == CMIF_ENERGY_POLICY
            and formal_config_payload["buffer_names"] == []
            and formal_config_payload["state_equation_energy_terms"] == 4
            and formal_config_payload["efficient_feature_convolutions"]
            == 1
            and formal_config_payload[
                "efficient_occupancy_convolutions"
            ]
            == 1
            and formal_config_payload[
                "joint_kernel_macs_per_coarse_cell"
            ]
            == 64000
            and formal_config_payload[
                "phase_hidden_elements_per_coarse_cell"
            ]
            == 512
            and formal_config_payload["scalar_output_fields"] == 1
            and formal_config_payload["auxiliary_outputs"] == 0
            and formal_config_payload["learned_threshold"] is False
            and formal_config_payload["attention_or_softmax"] is False
            and formal_config_payload["recurrent_spatial_mixing"] is False
        ),
        "single_joint_parameterization": (
            parameter_names == expected_names
            and parameter_contract == expected_parameter_contract
            and "energy_output_bias" not in parameter_names
        ),
        "implementation_closure_bound": (
            tuple(path for path, _ in implementation_binding)
            == COVERAGE_STATE_CMIF_IMPLEMENTATION_PATHS
            and len({path for path, _ in implementation_binding})
            == len(implementation_binding)
            and all(
                len(digest) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in digest
                )
                for _, digest in implementation_binding
            )
        ),
        "reference_matrix_complete": (
            actual_reference_matrix == expected_reference_matrix
        ),
        "reference_equivalence": (
            bool(reference_probes)
            and all(
                float.fromhex(
                    str(value["maximum_absolute_error_hex"])
                )
                <= COVERAGE_STATE_CMIF_REFERENCE_ATOL
                and bool(value["reference_close"])
                and bool(value["finite"])
                and value["coarse_cell_count"]
                == value["feature_size"][0] * value["feature_size"][1]
                and value["phase_output_count"]
                == value["coarse_cell_count"] * 16
                and value["reference_energy_evaluation_count"]
                == value["phase_output_count"] * 4
                and value["efficient_full_grid_convolution_count"] == 2
                for value in reference_probes
            )
        ),
        "zero_feature_exact": (
            all(
                bool(value["zero_feature_exact"])
                for value in reference_probes
            )
            and bool(null_probe["zero_feature_all_exact"])
        ),
        "phase_roundtrip_and_order": (
            bool(phase_probe["roundtrip_exact"])
            and bool(phase_probe["row_major_exact"])
            and phase_probe["phase_count"] == 16
        ),
        "center_only_neutralization": (
            center_probe["neutral_phase_hex"]
            == float(CMIF_NEUTRAL_PHASE).hex()
            and bool(center_probe["delta_formula_exact"])
            and center_probe["midpoint_changed_element_count"] == 1
            and center_probe["midpoint_selected_value_hex"]
            == float(CMIF_NEUTRAL_PHASE).hex()
            and
            bool(center_probe["changed_coordinate_differs"])
            and bool(center_probe["all_other_coordinates_exact"])
            and bool(center_probe["global_phase_replacement_differs"])
            and float.fromhex(
                str(
                    center_probe[
                        "global_phase_replacement_max_difference_hex"
                    ]
                )
            )
            > 0.0
        ),
        "pure_paths_annihilated": (
            bool(null_probe["feature_weight_zero_exact"])
            and bool(null_probe["occupancy_weight_zero_exact"])
            and bool(null_probe["nonzero_energy_readout"])
        ),
        "energy_gauges_invariant": all(
            bool(value) for value in gauge_probe.values()
        ),
        "endpoint_asymmetry_and_rank": (
            endpoint_probe["rank"] == 2
            and bool(endpoint_probe["non_antisymmetric"])
            and bool(endpoint_probe["correct_deletion_direction"])
        ),
        "radius_two_exact_and_active": (
            bool(locality_probe["occupancy_outside_radius_two_exact"])
            and bool(locality_probe["occupancy_radius_two_witness"])
            and bool(locality_probe["occupancy_radius_one_witness"])
            and bool(locality_probe["feature_outside_radius_two_exact"])
            and bool(locality_probe["feature_radius_two_witness"])
            and bool(locality_probe["feature_radius_one_witness"])
            and locality_probe["cross_phase_output_phase"]
            != locality_probe["cross_phase_neighbour_phase"]
            and bool(
                locality_probe[
                    "cross_phase_neighbour_cell_differs"
                ]
            )
            and bool(locality_probe["cross_phase_neighbour_witness"])
        ),
        "identity_and_gradient_latency": (
            bool(gradient_probe["fixed_input_replay_exact"])
            and bool(gradient_probe["identity_response_exact"])
            and bool(gradient_probe["initial_field_exact"])
            and bool(gradient_probe["initial_completion_empty"])
            and bool(
                gradient_probe["initial_union_preserves_occupancy"]
            )
            and bool(gradient_probe["gradient_latency_passed"])
        ),
        "D_V_not_accessed": True,
        "D_T_not_accessed": True,
        "D_R_not_accessed": True,
        "dataset_training_not_performed": True,
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True)
class CoverageStateCMIFDatasetFreeReceipt:
    """Fingerprintable generated-only evidence for CMIF-v17."""

    formal_config_payload: dict[str, object]
    parameter_names: tuple[str, ...]
    parameter_contract: tuple[
        tuple[str, tuple[int, ...], str, bool],
        ...,
    ]
    implementation_binding: tuple[tuple[str, str], ...]
    reference_probes: tuple[dict[str, object], ...]
    phase_probe: dict[str, object]
    center_probe: dict[str, object]
    null_probe: dict[str, object]
    gauge_probe: dict[str, object]
    endpoint_probe: dict[str, object]
    locality_probe: dict[str, object]
    gradient_probe: dict[str, object]
    checks: tuple[tuple[str, bool], ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def _evidence_payload(self) -> dict[str, object]:
        return {
            "formal_config": deepcopy(self.formal_config_payload),
            "parameter_names": list(self.parameter_names),
            "parameter_contract": [
                [
                    name,
                    list(shape),
                    dtype,
                    requires_grad,
                ]
                for name, shape, dtype, requires_grad
                in self.parameter_contract
            ],
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "reference_probes": deepcopy(list(self.reference_probes)),
            "phase_probe": deepcopy(self.phase_probe),
            "center_probe": deepcopy(self.center_probe),
            "null_probe": deepcopy(self.null_probe),
            "gauge_probe": deepcopy(self.gauge_probe),
            "endpoint_probe": deepcopy(self.endpoint_probe),
            "locality_probe": deepcopy(self.locality_probe),
            "gradient_probe": deepcopy(self.gradient_probe),
        }

    def verify_unchanged(self) -> None:
        expected = recompute_coverage_state_cmif_dataset_free_checks(
            formal_config_payload=self.formal_config_payload,
            parameter_names=self.parameter_names,
            parameter_contract=self.parameter_contract,
            implementation_binding=self.implementation_binding,
            reference_probes=self.reference_probes,
            phase_probe=self.phase_probe,
            center_probe=self.center_probe,
            null_probe=self.null_probe,
            gauge_probe=self.gauge_probe,
            endpoint_probe=self.endpoint_probe,
            locality_probe=self.locality_probe,
            gradient_probe=self.gradient_probe,
        )
        if (
            self.checks != expected
            or stable_fingerprint(self._evidence_payload())
            != self.evidence_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
        ):
            raise RuntimeError(
                "CMIF dataset-free evidence changed after creation"
            )

    @property
    def all_pass(self) -> bool:
        self.verify_unchanged()
        return bool(self.checks) and all(value for _, value in self.checks)

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_CMIF_DATASET_FREE_SCHEMA,
            "model_class": (
                self.formal_config_payload["model_class"]
            ),
            "formal_config": deepcopy(self.formal_config_payload),
            "parameter_names": list(self.parameter_names),
            "parameter_contract": [
                {
                    "name": name,
                    "shape": list(shape),
                    "dtype": dtype,
                    "requires_grad": requires_grad,
                }
                for name, shape, dtype, requires_grad
                in self.parameter_contract
            ],
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "reference_probes": deepcopy(list(self.reference_probes)),
            "phase_probe": deepcopy(self.phase_probe),
            "center_probe": deepcopy(self.center_probe),
            "null_probe": deepcopy(self.null_probe),
            "gauge_probe": deepcopy(self.gauge_probe),
            "endpoint_probe": deepcopy(self.endpoint_probe),
            "locality_probe": deepcopy(self.locality_probe),
            "gradient_probe": deepcopy(self.gradient_probe),
            "evidence_fingerprint": self.evidence_fingerprint,
            "checks": dict(self.checks),
            "all_pass": self.all_pass,
            "runtime_splits": [],
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "dataset_training_performed": False,
            "synthetic_gradient_probe_optimizer_steps": 1,
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_coverage_state_cmif_dataset_free_gate(
) -> CoverageStateCMIFDatasetFreeReceipt:
    """Run the complete generated-only v17 CMIF structural gate."""

    before_rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(
            COVERAGE_STATE_CMIF_DATASET_FREE_EXECUTION_SEED
        )
        result = _run_coverage_state_cmif_dataset_free_gate_inner()
    if not torch.equal(before_rng, torch.random.get_rng_state()):
        raise RuntimeError("CMIF dataset-free gate changed global RNG state")
    return result


def _run_coverage_state_cmif_dataset_free_gate_inner(
) -> CoverageStateCMIFDatasetFreeReceipt:
    formal_config = _formal_config()
    formal_model = CURELiteCenteredMixedInteractionLevelSet(formal_config)
    formal_payload = {
        "feature_channels": formal_config.feature_channels,
        "model_class": formal_model.__class__.__name__,
        "feature_stride": formal_config.feature_stride,
        "phase_channels": formal_config.phase_occupancy_channels,
        "width": formal_config.width,
        "coarse_radius": formal_config.coarse_radius,
        "kernel_size": formal_config.kernel_size,
        "neutral_phase_hex": formal_config.neutral_phase.hex(),
        "field_amplitude_hex": formal_config.field_amplitude.hex(),
        "normalization_epsilon_hex": (
            formal_config.normalization_epsilon.hex()
        ),
        "feature_policy": formal_config.feature_policy,
        "target_policy": formal_config.target_policy,
        "output_policy": formal_config.output_policy,
        "numerical_policy": formal_config.numerical_policy,
        "coverage_policy": formal_config.coverage_policy,
        "input_representation": formal_config.input_representation,
        "field_policy": formal_config.field_policy,
        "interaction_policy": formal_config.interaction_policy,
        "energy_policy": formal_config.energy_policy,
        "expected_parameter_count": formal_config.expected_parameter_count,
        "actual_parameter_count": sum(
            parameter.numel() for parameter in formal_model.parameters()
        ),
        "buffer_names": [
            name for name, _ in formal_model.named_buffers()
        ],
        "state_equation_energy_terms": 4,
        "efficient_feature_convolutions": 1,
        "efficient_occupancy_convolutions": 1,
        "joint_kernel_macs_per_coarse_cell": (
            formal_config.width
            * (
                formal_config.feature_channels
                + formal_config.phase_occupancy_channels
            )
            * formal_config.kernel_size
            * formal_config.kernel_size
        ),
        "phase_hidden_elements_per_coarse_cell": (
            formal_config.phase_occupancy_channels * formal_config.width
        ),
        "scalar_output_fields": 1,
        "auxiliary_outputs": 0,
        "learned_threshold": False,
        "attention_or_softmax": False,
        "recurrent_spatial_mixing": False,
    }
    parameter_names = tuple(
        sorted(name for name, _ in formal_model.named_parameters())
    )
    parameter_contract = tuple(
        (
            name,
            tuple(int(value) for value in parameter.shape),
            str(parameter.dtype),
            bool(parameter.requires_grad),
        )
        for name, parameter in sorted(formal_model.named_parameters())
    )
    implementation_binding = _current_implementation_binding()
    reference_probes = _reference_probes()
    phase_probe = _phase_contract_probe()
    center_probe = _center_only_probe()
    null_probe = _zero_and_pure_path_probe()
    gauge_probe = _gauge_probe()
    endpoint_probe = _endpoint_probe()
    locality_probe = _locality_probe()
    gradient_probe = _gradient_and_identity_probe()
    checks = recompute_coverage_state_cmif_dataset_free_checks(
        formal_config_payload=formal_payload,
        parameter_names=parameter_names,
        parameter_contract=parameter_contract,
        implementation_binding=implementation_binding,
        reference_probes=reference_probes,
        phase_probe=phase_probe,
        center_probe=center_probe,
        null_probe=null_probe,
        gauge_probe=gauge_probe,
        endpoint_probe=endpoint_probe,
        locality_probe=locality_probe,
        gradient_probe=gradient_probe,
    )
    evidence_payload = {
        "formal_config": deepcopy(formal_payload),
        "parameter_names": list(parameter_names),
        "parameter_contract": [
            [name, list(shape), dtype, requires_grad]
            for name, shape, dtype, requires_grad
            in parameter_contract
        ],
        "implementation_binding": dict(implementation_binding),
        "reference_probes": deepcopy(list(reference_probes)),
        "phase_probe": deepcopy(phase_probe),
        "center_probe": deepcopy(center_probe),
        "null_probe": deepcopy(null_probe),
        "gauge_probe": deepcopy(gauge_probe),
        "endpoint_probe": deepcopy(endpoint_probe),
        "locality_probe": deepcopy(locality_probe),
        "gradient_probe": deepcopy(gradient_probe),
    }
    return CoverageStateCMIFDatasetFreeReceipt(
        formal_config_payload=formal_payload,
        parameter_names=parameter_names,
        parameter_contract=parameter_contract,
        implementation_binding=implementation_binding,
        reference_probes=reference_probes,
        phase_probe=phase_probe,
        center_probe=center_probe,
        null_probe=null_probe,
        gauge_probe=gauge_probe,
        endpoint_probe=endpoint_probe,
        locality_probe=locality_probe,
        gradient_probe=gradient_probe,
        checks=checks,
        evidence_fingerprint=stable_fingerprint(evidence_payload),
    )


__all__ = [
    "COVERAGE_STATE_CMIF_DATASET_FREE_SCHEMA",
    "COVERAGE_STATE_CMIF_DATASET_FREE_SEEDS",
    "COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS",
    "COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE",
    "COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT",
    "COVERAGE_STATE_CMIF_FORMAL_WIDTH",
    "CoverageStateCMIFDatasetFreeReceipt",
    "recompute_coverage_state_cmif_dataset_free_checks",
    "run_coverage_state_cmif_dataset_free_gate",
]
