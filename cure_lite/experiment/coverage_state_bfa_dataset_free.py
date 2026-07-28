"""Generated-only structural gate for v20 BFA-CMIF.

This module checks the fifteen frozen, dataset-free requirements of the
binary-flip antisymmetrized completion field.  Every tensor is generated in
memory.  The gate does not construct an optimizer, perform a parameter
update, inspect a split, or read a model/cache artifact.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import inspect
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_binary_flip_antisymmetric import (
    BFA_ENERGY_POLICY,
    BFA_FLIP_POLICY,
    BFA_INPUT_REPRESENTATION,
    BFA_INTERACTION_POLICY,
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
    binary_flip_odd_projection,
    flip_binary_center_phase,
)
from ..coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
    centered_mixed_energy_difference,
)
from ..coverage_state_level_set import CSLF_FIELD_AMPLITUDE
from ..coverage_state_phase_preserving import (
    pixel_unshuffle_bool_occupancy,
)
from ..coverage_state_sobolev import (
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    coverage_state_pmope_pair_loss_from_targets,
    prepare_coverage_state_pair_targets,
)
from ..paired_types import tensor_content_fingerprint


COVERAGE_STATE_BFA_DATASET_FREE_SCHEMA = (
    "cure-lite-bfa-cmif-v20-dataset-free-receipt-v1"
)
COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED = 200020
COVERAGE_STATE_BFA_FORMAL_FEATURE_CHANNELS = 64
COVERAGE_STATE_BFA_FORMAL_FEATURE_STRIDE = 4
COVERAGE_STATE_BFA_FORMAL_WIDTH = 32
COVERAGE_STATE_BFA_FORMAL_PARAMETER_COUNT = 64064
COVERAGE_STATE_BFA_TRUNCATION_RADIUS = 4
COVERAGE_STATE_BFA_MARGIN = (
    CSLF_FIELD_AMPLITUDE / COVERAGE_STATE_BFA_TRUNCATION_RADIUS
)
COVERAGE_STATE_BFA_REFERENCE_RTOL = 2.0e-5
COVERAGE_STATE_BFA_REFERENCE_ATOL = 2.0e-6
COVERAGE_STATE_BFA_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_phase_preserving.py",
    "cure_lite/coverage_state_centered_mixed_interaction.py",
    "cure_lite/coverage_state_binary_flip_antisymmetric.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/experiment/coverage_state_bfa_dataset_free.py",
)
COVERAGE_STATE_BFA_DATASET_FREE_CHECK_NAMES = (
    "01_exact_boolean_flip_involution",
    "02_efficient_reference_elementwise",
    "03_local_interaction_antisymmetric",
    "04_local_field_sum_two_anchor",
    "05_zero_feature_anchor",
    "06_pure_additive_paths_silent",
    "07_affine_energy_equals_midpoint_cmif",
    "08_nonlinear_difference_zero_level_witness",
    "09_target_background_component_intervals_feasible",
    "10_pmope_gradient_directions_finite",
    "11_phase_roundtrip_exact",
    "12_parameter_contract_and_initialization_exact",
    "13_staged_gradient_path_finite",
    "14_forward_interface_has_no_role_metadata",
    "15_no_runtime_data_or_optimizer_path",
)


def _current_implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_BFA_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"BFA-CMIF implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _hex(value: Tensor | float) -> str:
    scalar = float(value.detach().reshape(()).item()) if isinstance(
        value, Tensor
    ) else float(value)
    return scalar.hex()


def _state_fingerprint(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
) -> str:
    return stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(model.state_dict().items())
        }
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
            0.20
            * torch.randn(
                model.scalar_energy_weight.shape,
                generator=generator,
            )
        )


def _mask(
    size: int,
    coordinates: tuple[tuple[int, int], ...] = (),
) -> Tensor:
    result = torch.zeros(1, 1, size, size, dtype=torch.bool)
    for row, column in coordinates:
        result[..., row, column] = True
    return result


def _loss_config() -> CoverageStateSobolevConfig:
    return CoverageStateSobolevConfig(
        truncation_radius=COVERAGE_STATE_BFA_TRUNCATION_RADIUS
    )


def _pair_targets(
    *,
    size: int,
    occupancy_plus: Tensor | None = None,
    occupancy_minus: Tensor | None = None,
    target_plus: Tensor | None = None,
    target_minus: Tensor | None = None,
) -> CoverageStatePairTargets:
    empty = _mask(size)
    return prepare_coverage_state_pair_targets(
        empty if occupancy_plus is None else occupancy_plus,
        empty if occupancy_minus is None else occupancy_minus,
        empty if target_plus is None else target_plus,
        empty if target_minus is None else target_minus,
        torch.ones(1, 1, size, size, dtype=torch.bool),
        config=_loss_config(),
    )


def _deep_feasible_fields(
    targets: CoverageStatePairTargets,
) -> tuple[Tensor, Tensor]:
    magnitude = COVERAGE_STATE_BFA_MARGIN + 0.35
    return (
        torch.sign(targets.target_field_plus) * magnitude,
        torch.sign(targets.target_field_minus) * magnitude,
    )


def _flip_involution_probe() -> dict[str, object]:
    generator = torch.Generator().manual_seed(
        COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 1
    )
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
    return {
        "input_dtype": str(patch.dtype),
        "flipped_dtype": str(flipped.dtype),
        "changed_coordinates": changed.tolist(),
        "exactly_one_bit_changed": changed.tolist() == [[2, 2, 2]],
        "selected_bit_complemented": bool(
            flipped[2, 2, 2] == torch.logical_not(patch[2, 2, 2])
        ),
        "all_other_bits_unchanged": int(changed.shape[0]) == 1,
        "involution_exact": torch.equal(restored, patch),
        "input_not_mutated": not torch.equal(patch, flipped),
        "input_sha256": tensor_content_fingerprint(patch),
        "flipped_sha256": tensor_content_fingerprint(flipped),
        "restored_sha256": tensor_content_fingerprint(restored),
    }


def _reference_equivalence_probe() -> dict[str, object]:
    config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=2,
        feature_stride=2,
        width=3,
    )
    model = CURELiteBinaryFlipAntisymmetricLevelSet(config)
    _randomize_output_path(
        model,
        seed=COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 2,
    )
    generator = torch.Generator().manual_seed(
        COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 3
    )
    feature = torch.randn(1, 2, 3, 4, generator=generator)
    occupancy = (
        torch.rand(1, 1, 6, 8, generator=generator) > 0.57
    )
    state_before = _state_fingerprint(model)
    efficient = model.forward_fields(feature, occupancy).field
    reference = model.forward_reference(feature, occupancy)
    state_after = _state_fingerprint(model)
    absolute_error = (efficient - reference).abs()
    tolerance = (
        COVERAGE_STATE_BFA_REFERENCE_ATOL
        + COVERAGE_STATE_BFA_REFERENCE_RTOL * reference.abs()
    )
    return {
        "efficient_shape": list(efficient.shape),
        "reference_shape": list(reference.shape),
        "element_count": efficient.numel(),
        "all_elements_within_frozen_tolerance": bool(
            torch.all(absolute_error <= tolerance)
        ),
        "max_absolute_error_hex": _hex(absolute_error.max()),
        "rtol_hex": COVERAGE_STATE_BFA_REFERENCE_RTOL.hex(),
        "atol_hex": COVERAGE_STATE_BFA_REFERENCE_ATOL.hex(),
        "efficient_finite": bool(torch.isfinite(efficient).all()),
        "reference_finite": bool(torch.isfinite(reference).all()),
        "model_state_unchanged": state_before == state_after,
        "feature_sha256": tensor_content_fingerprint(feature),
        "occupancy_sha256": tensor_content_fingerprint(occupancy),
        "efficient_sha256": tensor_content_fingerprint(efficient),
        "reference_sha256": tensor_content_fingerprint(reference),
    }


def _local_antisymmetry_probe() -> dict[str, object]:
    config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    model = CURELiteBinaryFlipAntisymmetricLevelSet(config)
    _randomize_output_path(
        model,
        seed=COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 4,
    )
    generator = torch.Generator().manual_seed(
        COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 5
    )
    feature = torch.randn(1, 2, 3, 4, generator=generator)
    occupancy = (
        torch.rand(1, 1, 6, 8, generator=generator) > 0.52
    )
    actual = model.forward_fields(feature, occupancy)
    interaction_residuals: list[Tensor] = []
    field_residuals: list[Tensor] = []
    interaction_checks: list[bool] = []
    field_checks: list[bool] = []
    bit_checks: list[bool] = []
    flipped_interactions: list[Tensor] = []
    flipped_fields: list[Tensor] = []
    for output_row in range(occupancy.shape[-2]):
        for output_column in range(occupancy.shape[-1]):
            flipped_occupancy = occupancy.clone()
            flipped_occupancy[
                0, 0, output_row, output_column
            ] = torch.logical_not(
                flipped_occupancy[
                    0, 0, output_row, output_column
                ]
            )
            flipped = model.forward_fields(
                feature,
                flipped_occupancy,
            )
            phase_index = (
                output_row % config.feature_stride
            ) * config.feature_stride + (
                output_column % config.feature_stride
            )
            coarse_row = output_row // config.feature_stride
            coarse_column = output_column // config.feature_stride
            index = (
                0,
                phase_index,
                coarse_row,
                coarse_column,
            )
            actual_interaction = actual.native_phase_interaction[
                index
            ]
            flipped_interaction = flipped.native_phase_interaction[
                index
            ]
            actual_field = actual.field[
                0, 0, output_row, output_column
            ]
            flipped_field = flipped.field[
                0, 0, output_row, output_column
            ]
            interaction_residuals.append(
                (actual_interaction + flipped_interaction).abs()
            )
            field_residuals.append(
                (
                    actual_field
                    + flipped_field
                    - 2.0 * CSLF_FIELD_AMPLITUDE
                ).abs()
            )
            interaction_checks.append(
                bool(
                    torch.allclose(
                        actual_interaction,
                        -flipped_interaction,
                        rtol=COVERAGE_STATE_BFA_REFERENCE_RTOL,
                        atol=COVERAGE_STATE_BFA_REFERENCE_ATOL,
                    )
                )
            )
            field_checks.append(
                bool(
                    torch.allclose(
                        actual_field + flipped_field,
                        torch.tensor(
                            2.0 * CSLF_FIELD_AMPLITUDE
                        ),
                        rtol=COVERAGE_STATE_BFA_REFERENCE_RTOL,
                        atol=COVERAGE_STATE_BFA_REFERENCE_ATOL,
                    )
                )
            )
            bit_checks.append(
                bool(
                    occupancy[
                        0, 0, output_row, output_column
                    ]
                    != flipped_occupancy[
                        0, 0, output_row, output_column
                    ]
                )
            )
            flipped_interactions.append(
                flipped_interaction.detach()
            )
            flipped_fields.append(flipped_field.detach())
    interaction_residual = torch.stack(interaction_residuals)
    field_residual = torch.stack(field_residuals)
    return {
        "tested_coordinate_count": len(interaction_checks),
        "expected_coordinate_count": occupancy.shape[-2]
        * occupancy.shape[-1],
        "every_selected_occupancy_bit_changed": all(bit_checks),
        "interaction_antisymmetric": all(interaction_checks),
        "max_interaction_residual_hex": _hex(
            interaction_residual.max()
        ),
        "field_sum_equals_two_anchor": all(field_checks),
        "max_field_sum_residual_hex": _hex(
            field_residual.max()
        ),
        "actual_interaction_sha256": tensor_content_fingerprint(
            actual.native_phase_interaction
        ),
        "flipped_selected_interactions_sha256": (
            tensor_content_fingerprint(
                torch.stack(flipped_interactions)
            )
        ),
        "actual_field_sha256": tensor_content_fingerprint(
            actual.field
        ),
        "flipped_selected_fields_sha256": (
            tensor_content_fingerprint(torch.stack(flipped_fields))
        ),
    }


def _zero_feature_probe() -> dict[str, object]:
    config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    model = CURELiteBinaryFlipAntisymmetricLevelSet(config)
    _randomize_output_path(
        model,
        seed=COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 6,
    )
    generator = torch.Generator().manual_seed(
        COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 7
    )
    occupancy = torch.stack(
        (
            torch.zeros(1, 8, 8, dtype=torch.bool),
            torch.ones(1, 8, 8, dtype=torch.bool),
            torch.rand(1, 8, 8, generator=generator) > 0.5,
            F.pad(
                torch.ones(1, 1, 1, dtype=torch.bool),
                (3, 4, 2, 5),
            ),
        ),
        dim=0,
    )
    feature = torch.zeros(4, 2, 4, 4)
    fields = model.forward_fields(feature, occupancy)
    anchor = torch.full_like(fields.field, CSLF_FIELD_AMPLITUDE)
    return {
        "occupancy_kinds": ["empty", "dense", "random", "single"],
        "feature_exact_zero": not bool(torch.any(feature)),
        "feature_presence_hidden_exact_zero": not bool(
            torch.any(fields.actual_feature_presence_hidden)
        ),
        "interaction_exact_zero": not bool(
            torch.any(fields.native_phase_interaction)
        ),
        "field_exact_anchor": torch.equal(fields.field, anchor),
        "completion_exact_empty": not bool(
            torch.any(model.predict_completion(feature, occupancy))
        ),
        "field_sha256": tensor_content_fingerprint(fields.field),
    }


def _pure_path_probe() -> dict[str, object]:
    generator = torch.Generator().manual_seed(
        COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 8
    )
    feature = torch.randn(2, 2, 4, 4, generator=generator)
    occupancy = (
        torch.rand(2, 1, 8, 8, generator=generator) > 0.5
    )
    results: dict[str, dict[str, object]] = {}
    for path_name in ("feature_weight_zero", "occupancy_weight_zero"):
        model = CURELiteBinaryFlipAntisymmetricLevelSet(
            CoverageStateBinaryFlipAntisymmetricConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )
        _randomize_output_path(
            model,
            seed=COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 9,
        )
        with torch.no_grad():
            if path_name == "feature_weight_zero":
                model.joint_state_weight[
                    :, : model.config.feature_channels
                ].zero_()
            else:
                model.joint_state_weight[
                    :, model.config.feature_channels :
                ].zero_()
        fields = model.forward_fields(feature, occupancy)
        anchor = torch.full_like(fields.field, CSLF_FIELD_AMPLITUDE)
        results[path_name] = {
            "scalar_energy_nonzero": bool(
                torch.any(model.scalar_energy_weight != 0.0)
            ),
            "interaction_exact_zero": not bool(
                torch.any(fields.native_phase_interaction)
            ),
            "field_exact_anchor": torch.equal(fields.field, anchor),
            "completion_exact_empty": not bool(
                torch.any(model.predict_completion(feature, occupancy))
            ),
            "field_sha256": tensor_content_fingerprint(fields.field),
        }
    return {
        "feature_weight_zero": results["feature_weight_zero"],
        "occupancy_weight_zero": results["occupancy_weight_zero"],
        "both_pure_paths_silent": all(
            bool(result["field_exact_anchor"])
            and bool(result["completion_exact_empty"])
            for result in results.values()
        ),
    }


def _affine_equivalence_probe() -> dict[str, object]:
    occupancy = torch.tensor((0.0, 1.0), dtype=torch.float64)
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
    expected = 5.0 * (occupancy - 0.5)
    return {
        "energy_kind": "affine_feature_presence_in_binary_phase",
        "old_midpoint_equals_new_binary_flip_exact": torch.equal(
            old_midpoint,
            new_binary_flip,
        ),
        "new_equals_analytic_exact": torch.equal(
            new_binary_flip,
            expected,
        ),
        "endpoint_antisymmetry_exact": torch.equal(
            new_binary_flip,
            -torch.flip(new_binary_flip, dims=(0,)),
        ),
        "old_midpoint_hex": [_hex(value) for value in old_midpoint],
        "new_binary_flip_hex": [
            _hex(value) for value in new_binary_flip
        ],
    }


def _nonlinear_witness_probe() -> dict[str, object]:
    new = CURELiteBinaryFlipAntisymmetricLevelSet(
        CoverageStateBinaryFlipAntisymmetricConfig(
            feature_channels=1,
            feature_stride=1,
            width=1,
        )
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

    def feature_presence(phase: float) -> Tensor:
        phase_tensor = torch.tensor(phase, dtype=torch.float32)
        return 3.0 * (
            F.silu(-3.0 + 1.0 + 3.0 * phase_tensor)
            - F.silu(-3.0 + 3.0 * phase_tensor)
        )

    h0 = feature_presence(0.0)
    h1 = feature_presence(1.0)
    hm = feature_presence(0.5)
    odd = 0.5 * (h0 - h1)
    curvature = 0.5 * (h0 + h1) - hm
    old_interaction = old_field - CSLF_FIELD_AMPLITUDE
    new_interaction = new_field - CSLF_FIELD_AMPLITUDE
    return {
        "energy_kind": "shared_silu_feature_presence",
        "h0_hex": _hex(h0),
        "h1_hex": _hex(h1),
        "hm_hex": _hex(hm),
        "odd_hex": _hex(odd),
        "curvature_hex": _hex(curvature),
        "curvature_nonzero": bool(curvature.abs() > 1.0e-4),
        "old_interaction_equals_odd_plus_curvature": bool(
            torch.allclose(
                old_interaction,
                odd + curvature,
                rtol=2.0e-6,
                atol=2.0e-7,
            )
        ),
        "new_interaction_equals_odd": bool(
            torch.allclose(
                new_interaction,
                odd,
                rtol=2.0e-6,
                atol=2.0e-7,
            )
        ),
        "old_field_hex": _hex(old_field),
        "new_field_hex": _hex(new_field),
        "old_new_field_different": not torch.equal(
            old_field,
            new_field,
        ),
        "old_field_positive": bool(old_field > 0.0),
        "new_field_negative": bool(new_field < 0.0),
        "old_completion_empty": not bool(
            old.predict_completion(feature, vacant).item()
        ),
        "new_completion_present": bool(
            new.predict_completion(feature, vacant).item()
        ),
        "zero_level_output_differs": bool(
            old.predict_completion(feature, vacant).item()
            != new.predict_completion(feature, vacant).item()
        ),
    }


def _interval_feasibility_probe() -> dict[str, object]:
    anchor = CSLF_FIELD_AMPLITUDE
    margin = COVERAGE_STATE_BFA_MARGIN
    target_contrast = -1.2
    background_contrast = -0.5
    component_contrast = 0.5
    target_field = anchor + target_contrast
    background_field = anchor + background_contrast
    component_zero_endpoint = anchor + component_contrast
    component_one_endpoint = anchor - component_contrast
    target_upper = -(anchor + margin)
    background_lower = margin - anchor
    component_abs_upper = anchor - margin
    return {
        "anchor_hex": anchor.hex(),
        "margin_hex": margin.hex(),
        "target_contrast_hex": target_contrast.hex(),
        "background_contrast_hex": background_contrast.hex(),
        "component_contrast_hex": component_contrast.hex(),
        "target_upper_bound_hex": target_upper.hex(),
        "background_lower_bound_hex": background_lower.hex(),
        "component_abs_upper_bound_hex": component_abs_upper.hex(),
        "target_field_hex": target_field.hex(),
        "background_field_hex": background_field.hex(),
        "component_zero_endpoint_field_hex": (
            component_zero_endpoint.hex()
        ),
        "component_one_endpoint_field_hex": (
            component_one_endpoint.hex()
        ),
        "target_feasible": (
            target_contrast <= target_upper
            and target_field <= -margin
        ),
        "background_feasible": (
            background_contrast >= background_lower
            and background_field >= margin
        ),
        "component_feasible": (
            abs(component_contrast) <= component_abs_upper
            and component_zero_endpoint >= margin
            and component_one_endpoint >= margin
        ),
        "target_background_interval_gap_positive": (
            background_lower - target_upper > 0.0
        ),
        "all_three_simultaneously_feasible": (
            target_field <= -margin
            and background_field >= margin
            and component_zero_endpoint >= margin
            and component_one_endpoint >= margin
        ),
    }


def _one_pmope_direction(
    *,
    kind: str,
) -> dict[str, object]:
    size = 9
    center = size // 2
    if kind == "target":
        target = _mask(size, ((center, center),))
        targets = _pair_targets(
            size=size,
            target_plus=target,
            target_minus=target.clone(),
        )
    elif kind == "background":
        targets = _pair_targets(size=size)
    elif kind == "component":
        component = _mask(size, ((center, center),))
        targets = _pair_targets(
            size=size,
            occupancy_plus=component,
            occupancy_minus=_mask(size),
        )
    else:
        raise ValueError("unknown PMOPE direction probe kind")
    field_plus, field_minus = _deep_feasible_fields(targets)
    field_minus = field_minus.clone()
    field_minus[..., center, center] = (
        0.1 if kind == "target" else -0.1
    )
    field_minus.requires_grad_(True)
    fields = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_loss_config(),
    )
    (gradient,) = torch.autograd.grad(fields.loss, (field_minus,))
    value = gradient[0, 0, center, center]
    expects_negative_field = kind == "target"
    expected_gradient = (
        bool(value > 0.0)
        if expects_negative_field
        else bool(value < 0.0)
    )
    expected_descent = (
        bool(-value < 0.0)
        if expects_negative_field
        else bool(-value > 0.0)
    )
    return {
        "kind": kind,
        "loss_positive": bool(fields.loss > 0.0),
        "loss_finite": bool(torch.isfinite(fields.loss)),
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "probe_gradient_nonzero": bool(value != 0.0),
        "gradient_has_expected_sign": expected_gradient,
        "descent_has_expected_direction": expected_descent,
        "expected_descent": (
            "field_negative" if expects_negative_field
            else "field_positive"
        ),
        "probe_gradient_hex": _hex(value),
        "violation_at_probe_positive": bool(
            fields.violation_minus[0, 0, center, center] > 0.0
        ),
        "component_occupancy_removed": (
            kind != "component"
            or bool(
                targets.focus_support[0, 0, center, center]
            )
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }


def _pmope_gradient_probe() -> dict[str, object]:
    target = _one_pmope_direction(kind="target")
    background = _one_pmope_direction(kind="background")
    component = _one_pmope_direction(kind="component")
    required = (
        "loss_positive",
        "loss_finite",
        "gradient_finite",
        "probe_gradient_nonzero",
        "gradient_has_expected_sign",
        "descent_has_expected_direction",
        "violation_at_probe_positive",
        "component_occupancy_removed",
    )
    return {
        "objective": (
            "paired_minimum_sdf_margin_target_orthant_projection_"
            "joint_w1p4_energy_v1"
        ),
        "margin_hex": COVERAGE_STATE_BFA_MARGIN.hex(),
        "target": target,
        "background": background,
        "component": component,
        "all_three_directions_correct_and_finite": all(
            all(bool(probe[name]) for name in required)
            for probe in (target, background, component)
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }


def _phase_roundtrip_probe() -> dict[str, object]:
    generator = torch.Generator().manual_seed(
        COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 10
    )
    occupancy = (
        torch.rand(2, 1, 12, 16, generator=generator) > 0.5
    )
    phase = pixel_unshuffle_bool_occupancy(occupancy, stride=4)
    restored = F.pixel_shuffle(
        phase.to(dtype=torch.float32),
        4,
    ).to(dtype=torch.bool)
    return {
        "input_shape": list(occupancy.shape),
        "phase_shape": list(phase.shape),
        "restored_shape": list(restored.shape),
        "phase_dtype": str(phase.dtype),
        "roundtrip_exact": torch.equal(restored, occupancy),
        "input_sha256": tensor_content_fingerprint(occupancy),
        "phase_sha256": tensor_content_fingerprint(phase),
        "restored_sha256": tensor_content_fingerprint(restored),
    }


def _parameter_contract_probe() -> dict[str, object]:
    bfa_config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=COVERAGE_STATE_BFA_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_BFA_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_BFA_FORMAL_WIDTH,
    )
    cmif_config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=COVERAGE_STATE_BFA_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_BFA_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_BFA_FORMAL_WIDTH,
    )
    seed = COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 11
    torch.manual_seed(seed)
    old = CURELiteCenteredMixedInteractionLevelSet(cmif_config)
    torch.manual_seed(seed)
    new = CURELiteBinaryFlipAntisymmetricLevelSet(bfa_config)
    old_state = old.state_dict()
    new_state = new.state_dict()
    names = tuple(new_state)
    contract = tuple(
        (
            name,
            tuple(parameter.shape),
            str(parameter.dtype),
            bool(parameter.requires_grad),
        )
        for name, parameter in new.named_parameters()
    )
    generator = torch.Generator().manual_seed(seed + 1)
    feature = torch.randn(1, 64, 2, 3, generator=generator)
    occupancy = (
        torch.rand(1, 1, 8, 12, generator=generator) > 0.5
    )
    field = new(feature, occupancy)
    return {
        "model_class": new.__class__.__name__,
        "parameter_names": list(names),
        "parameter_contract": [
            [name, list(shape), dtype, requires_grad]
            for name, shape, dtype, requires_grad in contract
        ],
        "parameter_tensor_count": len(tuple(new.parameters())),
        "parameter_count": sum(
            parameter.numel() for parameter in new.parameters()
        ),
        "expected_parameter_count": bfa_config.expected_parameter_count,
        "buffer_names": [name for name, _ in new.named_buffers()],
        "joint_weight_nonzero": bool(
            torch.any(new.joint_state_weight != 0.0)
        ),
        "hidden_bias_exact_zero": not bool(
            torch.any(new.joint_hidden_bias)
        ),
        "scalar_energy_weight_exact_zero": not bool(
            torch.any(new.scalar_energy_weight)
        ),
        "state_keys_same_as_cmif": tuple(old_state) == tuple(new_state),
        "initial_state_byte_equal_to_cmif": all(
            torch.equal(value, new_state[name])
            for name, value in old_state.items()
        ),
        "initial_field_exact_anchor": torch.equal(
            field,
            torch.full_like(field, CSLF_FIELD_AMPLITUDE),
        ),
        "initial_completion_exact_empty": not bool(
            torch.any(new.predict_completion(feature, occupancy))
        ),
        "formal_feature_channels": bfa_config.feature_channels,
        "formal_feature_stride": bfa_config.feature_stride,
        "formal_width": bfa_config.width,
        "kernel_size": bfa_config.kernel_size,
        "phase_channels": bfa_config.phase_occupancy_channels,
        "state_sha256": stable_fingerprint(
            {
                name: tensor_content_fingerprint(value)
                for name, value in new_state.items()
            }
        ),
    }


def _staged_gradient_probe() -> dict[str, object]:
    config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    model = CURELiteBinaryFlipAntisymmetricLevelSet(config)
    generator = torch.Generator().manual_seed(
        COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED + 12
    )
    feature = torch.randn(1, 2, 4, 4, generator=generator)
    added = _mask(8, ((3, 3), (3, 4), (4, 3), (4, 4)))
    occupancy_plus = added.clone()
    occupancy_minus = _mask(8)
    targets = _pair_targets(
        size=8,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        target_plus=_mask(8),
        target_minus=added,
    )
    parameter_items = tuple(model.named_parameters())
    state_at_initial = _state_fingerprint(model)
    first_plus = model(feature, occupancy_plus)
    first_minus = model(feature, occupancy_minus)
    first_loss = coverage_state_pmope_pair_loss_from_targets(
        first_plus,
        first_minus,
        targets,
        config=_loss_config(),
    ).loss
    first_gradients = torch.autograd.grad(
        first_loss,
        tuple(parameter for _, parameter in parameter_items),
    )
    state_after_first = _state_fingerprint(model)
    first_contract = {
        name: {
            "finite": bool(torch.isfinite(gradient).all()),
            "nonzero": bool(torch.any(gradient != 0.0)),
            "sha256": tensor_content_fingerprint(gradient),
        }
        for (name, _), gradient in zip(
            parameter_items,
            first_gradients,
            strict=True,
        )
    }

    with torch.no_grad():
        model.scalar_energy_weight.copy_(
            torch.tensor((0.20, -0.15, 0.10, -0.05))
        )
    state_before_fixed_readout = _state_fingerprint(model)
    second_plus = model(feature, occupancy_plus)
    second_minus = model(feature, occupancy_minus)
    second_loss = coverage_state_pmope_pair_loss_from_targets(
        second_plus,
        second_minus,
        targets,
        config=_loss_config(),
    ).loss
    second_gradients = torch.autograd.grad(
        second_loss,
        tuple(parameter for _, parameter in parameter_items),
    )
    state_after_second = _state_fingerprint(model)
    second_contract = {
        name: {
            "finite": bool(torch.isfinite(gradient).all()),
            "nonzero": bool(torch.any(gradient != 0.0)),
            "sha256": tensor_content_fingerprint(gradient),
        }
        for (name, _), gradient in zip(
            parameter_items,
            second_gradients,
            strict=True,
        )
    }
    return {
        "probe_kind": "generated_pair_two_autograd_stages_no_update",
        "parameter_names": [name for name, _ in parameter_items],
        "first_loss_positive": bool(first_loss > 0.0),
        "first_loss_finite": bool(torch.isfinite(first_loss)),
        "first_gradient_contract": first_contract,
        "first_scalar_gradient_nonzero": bool(
            first_contract["scalar_energy_weight"]["nonzero"]
        ),
        "first_joint_weight_gradient_exact_zero": not bool(
            first_contract["joint_state_weight"]["nonzero"]
        ),
        "first_hidden_bias_gradient_exact_zero": not bool(
            first_contract["joint_hidden_bias"]["nonzero"]
        ),
        "known_initial_multiplicative_latency_observed": True,
        "initial_state_unchanged_by_autograd": (
            state_at_initial == state_after_first
        ),
        "fixed_nonzero_readout_witness": bool(
            torch.any(model.scalar_energy_weight != 0.0)
        ),
        "second_loss_positive": bool(second_loss > 0.0),
        "second_loss_finite": bool(torch.isfinite(second_loss)),
        "second_gradient_contract": second_contract,
        "all_second_gradients_finite": all(
            bool(value["finite"])
            for value in second_contract.values()
        ),
        "all_second_gradients_nonzero": all(
            bool(value["nonzero"])
            for value in second_contract.values()
        ),
        "fixed_readout_state_unchanged_by_autograd": (
            state_before_fixed_readout == state_after_second
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "feature_sha256": tensor_content_fingerprint(feature),
        "occupancy_plus_sha256": tensor_content_fingerprint(
            occupancy_plus
        ),
        "occupancy_minus_sha256": tensor_content_fingerprint(
            occupancy_minus
        ),
    }


def _forward_interface_probe() -> dict[str, object]:
    model = CURELiteBinaryFlipAntisymmetricLevelSet(
        CoverageStateBinaryFlipAntisymmetricConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    method_names = ("forward", "forward_fields", "predict_completion")
    signatures = {
        name: list(
            inspect.signature(getattr(model, name)).parameters
        )
        for name in method_names
    }
    forbidden = {
        "role",
        "roles",
        "pair_kind",
        "pair_kinds",
        "sample_id",
        "sample_ids",
        "target",
        "targets",
        "ground_truth",
        "gt",
        "metadata",
    }
    exposed = sorted(
        {
            parameter
            for parameters in signatures.values()
            for parameter in parameters
            if parameter in forbidden
        }
    )
    feature = torch.randn(1, 2, 2, 2)
    occupancy = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    unexpected_keyword_rejected = False
    try:
        model(feature, occupancy, role="factual")  # type: ignore[call-arg]
    except TypeError:
        unexpected_keyword_rejected = True
    output = model(feature, occupancy)
    return {
        "method_signatures": signatures,
        "forward_parameters_exact": signatures["forward"]
        == ["feature", "occupancy"],
        "forward_fields_parameters_exact": (
            signatures["forward_fields"] == ["feature", "occupancy"]
        ),
        "predict_completion_parameters_exact": (
            signatures["predict_completion"] == ["feature", "occupancy"]
        ),
        "forbidden_metadata_parameters": exposed,
        "forbidden_metadata_parameters_absent": not exposed,
        "unexpected_role_keyword_rejected": unexpected_keyword_rejected,
        "single_tensor_output": isinstance(output, Tensor),
        "single_scalar_field": tuple(output.shape) == (1, 1, 4, 4),
        "output_finite": bool(torch.isfinite(output).all()),
    }


def _qualified_call_name(node: ast.Call) -> str:
    value: ast.AST = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _static_boundary_probe() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    checked_paths = (
        "cure_lite/coverage_state_binary_flip_antisymmetric.py",
        "cure_lite/experiment/coverage_state_bfa_dataset_free.py",
    )
    imports: set[str] = set()
    calls: set[str] = set()
    for relative in checked_paths:
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
            elif isinstance(node, ast.Call):
                name = _qualified_call_name(node)
                if name:
                    calls.add(name)
    forbidden_import_fragments = (
        "torch.optim",
        "torch.utils.data",
        "dataloader",
        "datasets",
    )
    forbidden_call_suffixes = (
        ".SGD",
        ".Adam",
        ".AdamW",
        ".Optimizer",
        ".DataLoader",
        ".load_dataset",
        ".build_coverage_state_real_dr_inputs",
        ".torch.load",
    )
    forbidden_imports = sorted(
        name
        for name in imports
        if any(fragment in name.lower() for fragment in forbidden_import_fragments)
    )
    forbidden_calls = sorted(
        name
        for name in calls
        if any(
            name == suffix.removeprefix(".")
            or name.endswith(suffix)
            for suffix in forbidden_call_suffixes
        )
    )
    return {
        "checked_paths": list(checked_paths),
        "parsed_python_sources": len(checked_paths),
        "forbidden_imports": forbidden_imports,
        "forbidden_calls": forbidden_calls,
        "no_data_loader_or_split_import": not forbidden_imports,
        "no_optimizer_or_dataset_call": not forbidden_calls,
        "runtime_splits": [],
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "cache_artifact_accessed": False,
        "model_artifact_accessed": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "implementation_source_hashing_only": True,
    }


def _collect_generated_evidence() -> dict[str, dict[str, object]]:
    torch.manual_seed(COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED)
    return {
        "flip_involution": _flip_involution_probe(),
        "reference_equivalence": _reference_equivalence_probe(),
        "local_antisymmetry": _local_antisymmetry_probe(),
        "zero_feature": _zero_feature_probe(),
        "pure_paths": _pure_path_probe(),
        "affine_equivalence": _affine_equivalence_probe(),
        "nonlinear_witness": _nonlinear_witness_probe(),
        "interval_feasibility": _interval_feasibility_probe(),
        "pmope_gradient": _pmope_gradient_probe(),
        "phase_roundtrip": _phase_roundtrip_probe(),
        "parameter_contract": _parameter_contract_probe(),
        "staged_gradient": _staged_gradient_probe(),
        "forward_interface": _forward_interface_probe(),
        "static_boundary": _static_boundary_probe(),
    }


def recompute_coverage_state_bfa_dataset_free_checks(
    *,
    probes: dict[str, dict[str, object]],
    implementation_binding: tuple[tuple[str, str], ...],
    generated_replay_fingerprint: str,
) -> tuple[tuple[str, bool], ...]:
    """Recompute the fifteen frozen BFA-CMIF generated-only gate bits."""

    expected_probe_names = {
        "flip_involution",
        "reference_equivalence",
        "local_antisymmetry",
        "zero_feature",
        "pure_paths",
        "affine_equivalence",
        "nonlinear_witness",
        "interval_feasibility",
        "pmope_gradient",
        "phase_roundtrip",
        "parameter_contract",
        "staged_gradient",
        "forward_interface",
        "static_boundary",
    }
    if set(probes) != expected_probe_names:
        return tuple(
            (name, False)
            for name in COVERAGE_STATE_BFA_DATASET_FREE_CHECK_NAMES
        )
    flip = probes["flip_involution"]
    reference = probes["reference_equivalence"]
    local = probes["local_antisymmetry"]
    zero_feature = probes["zero_feature"]
    pure = probes["pure_paths"]
    affine = probes["affine_equivalence"]
    nonlinear = probes["nonlinear_witness"]
    intervals = probes["interval_feasibility"]
    pmope = probes["pmope_gradient"]
    phase = probes["phase_roundtrip"]
    parameters = probes["parameter_contract"]
    staged = probes["staged_gradient"]
    interface = probes["forward_interface"]
    boundary = probes["static_boundary"]
    expected_parameter_names = [
        "joint_state_weight",
        "joint_hidden_bias",
        "scalar_energy_weight",
    ]
    expected_parameter_contract = [
        [
            "joint_state_weight",
            [32, 80, 5, 5],
            "torch.float32",
            True,
        ],
        ["joint_hidden_bias", [32], "torch.float32", True],
        ["scalar_energy_weight", [32], "torch.float32", True],
    ]
    binding_paths = tuple(path for path, _ in implementation_binding)
    binding_valid = (
        binding_paths == COVERAGE_STATE_BFA_IMPLEMENTATION_PATHS
        and all(
            len(digest) == 64
            and all(
                character in "0123456789abcdef"
                for character in digest
            )
            for _, digest in implementation_binding
        )
    )
    replay_valid = (
        generated_replay_fingerprint == stable_fingerprint(probes)
    )
    checks = {
        "01_exact_boolean_flip_involution": all(
            bool(flip[name])
            for name in (
                "exactly_one_bit_changed",
                "selected_bit_complemented",
                "all_other_bits_unchanged",
                "involution_exact",
                "input_not_mutated",
            )
        )
        and flip["input_dtype"] == "torch.bool"
        and flip["flipped_dtype"] == "torch.bool",
        "02_efficient_reference_elementwise": all(
            bool(reference[name])
            for name in (
                "all_elements_within_frozen_tolerance",
                "efficient_finite",
                "reference_finite",
                "model_state_unchanged",
            )
        )
        and reference["efficient_shape"] == reference["reference_shape"]
        and int(reference["element_count"]) > 0,
        "03_local_interaction_antisymmetric": bool(
            local["every_selected_occupancy_bit_changed"]
            and local["interaction_antisymmetric"]
            and local["tested_coordinate_count"]
            == local["expected_coordinate_count"]
        ),
        "04_local_field_sum_two_anchor": bool(
            local["every_selected_occupancy_bit_changed"]
            and local["field_sum_equals_two_anchor"]
            and local["tested_coordinate_count"]
            == local["expected_coordinate_count"]
        ),
        "05_zero_feature_anchor": all(
            bool(zero_feature[name])
            for name in (
                "feature_exact_zero",
                "feature_presence_hidden_exact_zero",
                "interaction_exact_zero",
                "field_exact_anchor",
                "completion_exact_empty",
            )
        ),
        "06_pure_additive_paths_silent": bool(
            pure["both_pure_paths_silent"]
            and pure["feature_weight_zero"]["scalar_energy_nonzero"]
            and pure["feature_weight_zero"]["interaction_exact_zero"]
            and pure["occupancy_weight_zero"]["scalar_energy_nonzero"]
            and pure["occupancy_weight_zero"]["interaction_exact_zero"]
        ),
        "07_affine_energy_equals_midpoint_cmif": all(
            bool(affine[name])
            for name in (
                "old_midpoint_equals_new_binary_flip_exact",
                "new_equals_analytic_exact",
                "endpoint_antisymmetry_exact",
            )
        ),
        "08_nonlinear_difference_zero_level_witness": all(
            bool(nonlinear[name])
            for name in (
                "curvature_nonzero",
                "old_interaction_equals_odd_plus_curvature",
                "new_interaction_equals_odd",
                "old_new_field_different",
                "old_field_positive",
                "new_field_negative",
                "old_completion_empty",
                "new_completion_present",
                "zero_level_output_differs",
            )
        ),
        "09_target_background_component_intervals_feasible": all(
            bool(intervals[name])
            for name in (
                "target_feasible",
                "background_feasible",
                "component_feasible",
                "target_background_interval_gap_positive",
                "all_three_simultaneously_feasible",
            )
        )
        and intervals["anchor_hex"] == CSLF_FIELD_AMPLITUDE.hex()
        and intervals["margin_hex"] == COVERAGE_STATE_BFA_MARGIN.hex(),
        "10_pmope_gradient_directions_finite": bool(
            pmope["all_three_directions_correct_and_finite"]
            and not pmope["optimizer_constructed"]
            and pmope["optimizer_steps"] == 0
        ),
        "11_phase_roundtrip_exact": bool(
            phase["roundtrip_exact"]
            and phase["phase_dtype"] == "torch.bool"
            and phase["input_shape"] == phase["restored_shape"]
        ),
        "12_parameter_contract_and_initialization_exact": (
            parameters["parameter_names"] == expected_parameter_names
            and parameters["parameter_contract"]
            == expected_parameter_contract
            and parameters["parameter_tensor_count"] == 3
            and parameters["parameter_count"]
            == COVERAGE_STATE_BFA_FORMAL_PARAMETER_COUNT
            and parameters["expected_parameter_count"]
            == COVERAGE_STATE_BFA_FORMAL_PARAMETER_COUNT
            and parameters["buffer_names"] == []
            and parameters["joint_weight_nonzero"]
            and parameters["hidden_bias_exact_zero"]
            and parameters["scalar_energy_weight_exact_zero"]
            and parameters["state_keys_same_as_cmif"]
            and parameters["initial_state_byte_equal_to_cmif"]
            and parameters["initial_field_exact_anchor"]
            and parameters["initial_completion_exact_empty"]
            and parameters["formal_feature_channels"]
            == COVERAGE_STATE_BFA_FORMAL_FEATURE_CHANNELS
            and parameters["formal_feature_stride"]
            == COVERAGE_STATE_BFA_FORMAL_FEATURE_STRIDE
            and parameters["formal_width"]
            == COVERAGE_STATE_BFA_FORMAL_WIDTH
            and parameters["kernel_size"] == 5
            and parameters["phase_channels"] == 16
        ),
        "13_staged_gradient_path_finite": (
            staged["probe_kind"]
            == "generated_pair_two_autograd_stages_no_update"
            and staged["parameter_names"] == expected_parameter_names
            and staged["first_loss_positive"]
            and staged["first_loss_finite"]
            and staged["first_scalar_gradient_nonzero"]
            and staged["first_joint_weight_gradient_exact_zero"]
            and staged["first_hidden_bias_gradient_exact_zero"]
            and staged["known_initial_multiplicative_latency_observed"]
            and staged["initial_state_unchanged_by_autograd"]
            and staged["fixed_nonzero_readout_witness"]
            and staged["second_loss_positive"]
            and staged["second_loss_finite"]
            and staged["all_second_gradients_finite"]
            and staged["all_second_gradients_nonzero"]
            and staged["fixed_readout_state_unchanged_by_autograd"]
            and not staged["optimizer_constructed"]
            and staged["optimizer_steps"] == 0
            and staged["parameter_updates"] == 0
        ),
        "14_forward_interface_has_no_role_metadata": all(
            bool(interface[name])
            for name in (
                "forward_parameters_exact",
                "forward_fields_parameters_exact",
                "predict_completion_parameters_exact",
                "forbidden_metadata_parameters_absent",
                "unexpected_role_keyword_rejected",
                "single_tensor_output",
                "single_scalar_field",
                "output_finite",
            )
        )
        and interface["forbidden_metadata_parameters"] == [],
        "15_no_runtime_data_or_optimizer_path": (
            boundary["parsed_python_sources"]
            == len(boundary["checked_paths"])
            and boundary["forbidden_imports"] == []
            and boundary["forbidden_calls"] == []
            and boundary["no_data_loader_or_split_import"]
            and boundary["no_optimizer_or_dataset_call"]
            and boundary["runtime_splits"] == []
            and not boundary["D_R_accessed"]
            and not boundary["D_V_accessed"]
            and not boundary["D_T_accessed"]
            and not boundary["cache_artifact_accessed"]
            and not boundary["model_artifact_accessed"]
            and not boundary["optimizer_constructed"]
            and boundary["optimizer_steps"] == 0
            and boundary["parameter_updates"] == 0
            and not boundary["training_performed"]
            and boundary["implementation_source_hashing_only"]
            and binding_valid
            and replay_valid
        ),
    }
    return tuple(
        (name, bool(checks[name]))
        for name in COVERAGE_STATE_BFA_DATASET_FREE_CHECK_NAMES
    )


@dataclass(frozen=True)
class CoverageStateBFADatasetFreeReceipt:
    """Fingerprint-bound evidence for the v20 BFA-CMIF structural gate."""

    probes: dict[str, dict[str, object]]
    implementation_binding: tuple[tuple[str, str], ...]
    generated_replay_fingerprint: str
    checks: tuple[tuple[str, bool], ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def _evidence_payload(self) -> dict[str, object]:
        return {
            "probes": deepcopy(self.probes),
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "generated_replay_fingerprint": (
                self.generated_replay_fingerprint
            ),
        }

    def verify_unchanged(self) -> None:
        expected_checks = (
            recompute_coverage_state_bfa_dataset_free_checks(
                probes=self.probes,
                implementation_binding=self.implementation_binding,
                generated_replay_fingerprint=(
                    self.generated_replay_fingerprint
                ),
            )
        )
        if (
            self.checks != expected_checks
            or stable_fingerprint(self._evidence_payload())
            != self.evidence_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
        ):
            raise RuntimeError(
                "BFA-CMIF dataset-free evidence changed after creation"
            )

    @property
    def all_pass(self) -> bool:
        self.verify_unchanged()
        return (
            len(self.checks)
            == len(COVERAGE_STATE_BFA_DATASET_FREE_CHECK_NAMES)
            and all(value for _, value in self.checks)
        )

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_BFA_DATASET_FREE_SCHEMA,
            "model": "BFA-CMIF",
            "version": "v20",
            "input_interface": ["F_b", "O"],
            "interaction_policy": BFA_INTERACTION_POLICY,
            "energy_policy": BFA_ENERGY_POLICY,
            "flip_policy": BFA_FLIP_POLICY,
            "input_representation": BFA_INPUT_REPRESENTATION,
            "field_anchor_hex": CSLF_FIELD_AMPLITUDE.hex(),
            "fixed_margin_hex": COVERAGE_STATE_BFA_MARGIN.hex(),
            "execution_seed": (
                COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED
            ),
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "probes": deepcopy(self.probes),
            "generated_replay_fingerprint": (
                self.generated_replay_fingerprint
            ),
            "evidence_fingerprint": self.evidence_fingerprint,
            "checks": dict(self.checks),
            "check_count": len(self.checks),
            "all_pass": self.all_pass,
            "runtime_splits": [],
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "cache_artifact_accessed": False,
            "model_artifact_accessed": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "training_performed": False,
            "D_R_gate_authorized": self.all_pass,
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _run_coverage_state_bfa_dataset_free_gate_inner(
) -> CoverageStateBFADatasetFreeReceipt:
    first = _collect_generated_evidence()
    second = _collect_generated_evidence()
    first_fingerprint = stable_fingerprint(first)
    second_fingerprint = stable_fingerprint(second)
    if first_fingerprint != second_fingerprint:
        raise RuntimeError(
            "BFA-CMIF generated replay is not deterministic"
        )
    implementation_binding = _current_implementation_binding()
    checks = recompute_coverage_state_bfa_dataset_free_checks(
        probes=first,
        implementation_binding=implementation_binding,
        generated_replay_fingerprint=second_fingerprint,
    )
    evidence_payload = {
        "probes": deepcopy(first),
        "implementation_binding": dict(implementation_binding),
        "generated_replay_fingerprint": second_fingerprint,
    }
    return CoverageStateBFADatasetFreeReceipt(
        probes=first,
        implementation_binding=implementation_binding,
        generated_replay_fingerprint=second_fingerprint,
        checks=checks,
        evidence_fingerprint=stable_fingerprint(evidence_payload),
    )


def run_coverage_state_bfa_dataset_free_gate(
) -> CoverageStateBFADatasetFreeReceipt:
    """Run all fifteen generated-only BFA-CMIF structural checks."""

    before_rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        result = _run_coverage_state_bfa_dataset_free_gate_inner()
    if not torch.equal(before_rng, torch.random.get_rng_state()):
        raise RuntimeError(
            "BFA-CMIF dataset-free gate changed global RNG state"
        )
    return result


__all__ = [
    "COVERAGE_STATE_BFA_DATASET_FREE_SCHEMA",
    "COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED",
    "COVERAGE_STATE_BFA_FORMAL_FEATURE_CHANNELS",
    "COVERAGE_STATE_BFA_FORMAL_FEATURE_STRIDE",
    "COVERAGE_STATE_BFA_FORMAL_WIDTH",
    "COVERAGE_STATE_BFA_FORMAL_PARAMETER_COUNT",
    "COVERAGE_STATE_BFA_TRUNCATION_RADIUS",
    "COVERAGE_STATE_BFA_MARGIN",
    "COVERAGE_STATE_BFA_REFERENCE_RTOL",
    "COVERAGE_STATE_BFA_REFERENCE_ATOL",
    "COVERAGE_STATE_BFA_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_BFA_DATASET_FREE_CHECK_NAMES",
    "CoverageStateBFADatasetFreeReceipt",
    "recompute_coverage_state_bfa_dataset_free_checks",
    "run_coverage_state_bfa_dataset_free_gate",
]
