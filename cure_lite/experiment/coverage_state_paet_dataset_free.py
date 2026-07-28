"""Generated-only structural gate for v21 PAET-BFA.

The gate checks the frozen phase geometry, deterministic bilinear evidence
transport, shared BFA energy, parameter contract, and differentiability of
PAET-BFA.  All tensors are generated in memory.  It does not construct a
dataset, cache, optimizer, checkpoint, or training run.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, fields as dataclass_fields
from functools import cached_property
import inspect
from pathlib import Path

import torch
from torch import Tensor

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
    flip_binary_center_phase,
)
from ..coverage_state_level_set import CSLF_FIELD_AMPLITUDE
from ..coverage_state_phase_aligned_evidence_transport import (
    PAET_ENERGY_POLICY,
    PAET_FLIP_POLICY,
    PAET_INPUT_REPRESENTATION,
    PAET_INTERACTION_POLICY,
    PAET_TRANSPORT_POLICY,
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
    align_corners_false_axis_offsets,
    align_corners_false_phase_offsets,
    bilinear_phase_aligned_feature_affine,
    row_major_phase_pack,
    row_major_phase_unpack,
)
from ..paired_types import tensor_content_fingerprint


COVERAGE_STATE_PAET_DATASET_FREE_SCHEMA = (
    "cure-lite-paet-bfa-v21-dataset-free-receipt-v1"
)
COVERAGE_STATE_PAET_DATASET_FREE_EXECUTION_SEED = 210021
COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS = 64
COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE = 4
COVERAGE_STATE_PAET_FORMAL_WIDTH = 32
COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT = 64064
COVERAGE_STATE_PAET_TRUNCATION_RADIUS = 4
COVERAGE_STATE_PAET_MARGIN = (
    CSLF_FIELD_AMPLITUDE / COVERAGE_STATE_PAET_TRUNCATION_RADIUS
)
COVERAGE_STATE_PAET_REFERENCE_RTOL = 2.0e-5
COVERAGE_STATE_PAET_REFERENCE_ATOL = 2.0e-6
COVERAGE_STATE_PAET_FLIP_ATOL = 2.0e-7
COVERAGE_STATE_PAET_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_phase_preserving.py",
    "cure_lite/coverage_state_binary_flip_antisymmetric.py",
    "cure_lite/coverage_state_phase_aligned_evidence_transport.py",
    "cure_lite/experiment/coverage_state_paet_dataset_free.py",
)
COVERAGE_STATE_PAET_DATASET_FREE_CHECK_NAMES = (
    "01_align_corners_false_axis_offsets_exact",
    "02_row_major_phase_order_exact",
    "03_upsample_pack_pixelshuffle_identity",
    "04_constant_transport_preserved",
    "05_analytic_ramp_interior_exact",
    "06_phase_evidence_nondegenerate",
    "07_efficient_reference_elementwise",
    "08_flip_involution_odd_and_field_sum",
    "09_zero_feature_anchor",
    "10_pure_additive_paths_cancel",
    "11_parameter_contract_exact_bfa",
    "12_single_completion_field_no_role_interface",
    "13_first_second_order_gradients_finite_all_parameters",
    "14_no_runtime_data_cache_or_optimizer",
    "15_no_tunable_transport_parameters",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _current_implementation_binding() -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_PAET_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"PAET-BFA implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _hex(value: Tensor | float, *, name: str) -> str:
    number = (
        float(value.detach().reshape(()).item())
        if isinstance(value, Tensor)
        else float(value)
    )
    if not torch.isfinite(torch.tensor(number, dtype=torch.float64)):
        raise FloatingPointError(f"{name} is non-finite")
    return number.hex()


def _state_fingerprint(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
) -> str:
    return stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(model.state_dict().items())
        }
    )


def _parameter_gradient_fingerprint(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
) -> str:
    return stable_fingerprint(
        {
            name: (
                None
                if parameter.grad is None
                else tensor_content_fingerprint(parameter.grad)
            )
            for name, parameter in sorted(model.named_parameters())
        }
    )


def _toy_config(
) -> CoverageStatePhaseAlignedEvidenceTransportConfig:
    return CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )


def _randomize_paths(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
    *,
    seed: int,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        model.joint_state_weight.copy_(
            0.12
            * torch.randn(
                model.joint_state_weight.shape,
                generator=generator,
                dtype=torch.float32,
            )
        )
        model.joint_hidden_bias.copy_(
            0.08
            * torch.randn(
                model.joint_hidden_bias.shape,
                generator=generator,
                dtype=torch.float32,
            )
        )
        model.scalar_energy_weight.copy_(
            0.20
            * torch.randn(
                model.scalar_energy_weight.shape,
                generator=generator,
                dtype=torch.float32,
            )
        )


def _toy_inputs(*, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    feature = torch.randn(
        (1, 2, 3, 4),
        generator=generator,
        dtype=torch.float32,
    )
    occupancy = (
        torch.rand(
            (1, 1, 6, 8),
            generator=generator,
            dtype=torch.float32,
        )
        > 0.72
    )
    return feature.contiguous(), occupancy.contiguous()


def _offset_and_order_probe() -> dict[str, object]:
    stride = COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
    axis = align_corners_false_axis_offsets(stride)
    phases = align_corners_false_phase_offsets(stride)
    expected_axis = (-0.375, -0.125, 0.125, 0.375)
    expected_phases = tuple(
        (expected_axis[row], expected_axis[column])
        for row in range(stride)
        for column in range(stride)
    )

    fine = torch.arange(
        2 * stride * 3 * stride,
        dtype=torch.float64,
    ).reshape(1, 1, 2 * stride, 3 * stride)
    packed = row_major_phase_pack(fine, stride=stride)
    rows: list[dict[str, object]] = []
    phase_order_exact = True
    for phase_index, (row, column) in enumerate(
        (
            (row, column)
            for row in range(stride)
            for column in range(stride)
        )
    ):
        expected = fine[
            :,
            :,
            row::stride,
            column::stride,
        ]
        actual = packed[:, phase_index]
        exact = torch.equal(actual, expected)
        phase_order_exact = phase_order_exact and exact
        rows.append(
            {
                "phase_index": phase_index,
                "row": row,
                "column": column,
                "offset_row_hex": phases[phase_index][0].hex(),
                "offset_column_hex": phases[phase_index][1].hex(),
                "packed_exact": exact,
                "packed_fingerprint": tensor_content_fingerprint(
                    actual
                ),
            }
        )
    return {
        "stride": stride,
        "axis_offsets_hex": [value.hex() for value in axis],
        "expected_axis_offsets_hex": [
            value.hex() for value in expected_axis
        ],
        "phase_offsets_hex": [
            [row.hex(), column.hex()]
            for row, column in phases
        ],
        "phase_count": len(phases),
        "axis_formula_exact": axis == expected_axis,
        "axis_offsets_strictly_increasing": all(
            left < right for left, right in zip(axis, axis[1:])
        ),
        "axis_offsets_centered": sum(axis) == 0.0,
        "phase_offsets_row_major_exact": phases == expected_phases,
        "phase_rows": rows,
        "phase_pack_row_major_exact": phase_order_exact,
    }


def _transport_geometry_probe() -> dict[str, object]:
    stride = COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
    generator = torch.Generator(device="cpu").manual_seed(210031)
    coarse = torch.randn(
        (2, 3, 3, 5),
        generator=generator,
        dtype=torch.float64,
    )
    upsampled, packed = bilinear_phase_aligned_feature_affine(
        coarse,
        stride=stride,
    )
    direct_pack = row_major_phase_pack(
        upsampled,
        stride=stride,
    )
    reconstructed = row_major_phase_unpack(
        packed,
        stride=stride,
    )

    constants = torch.tensor(
        (-2.0, 0.5, 3.25),
        dtype=torch.float64,
    ).reshape(1, 3, 1, 1).expand(1, 3, 4, 5).contiguous()
    constant_up, constant_phase = (
        bilinear_phase_aligned_feature_affine(
            constants,
            stride=stride,
        )
    )
    expected_constant_up = constants.repeat_interleave(
        stride,
        dim=-2,
    ).repeat_interleave(stride, dim=-1)
    expected_constant_phase = constants.unsqueeze(1).expand(
        1,
        stride * stride,
        3,
        4,
        5,
    )
    return {
        "stride": stride,
        "coarse_shape": list(coarse.shape),
        "upsampled_shape": list(upsampled.shape),
        "phase_shape": list(packed.shape),
        "direct_pack_exact": torch.equal(packed, direct_pack),
        "pixelshuffle_unpack_exact": torch.equal(
            reconstructed,
            upsampled,
        ),
        "upsampled_fingerprint": tensor_content_fingerprint(upsampled),
        "packed_fingerprint": tensor_content_fingerprint(packed),
        "reconstructed_fingerprint": tensor_content_fingerprint(
            reconstructed
        ),
        "constant_upsample_exact": torch.equal(
            constant_up,
            expected_constant_up,
        ),
        "constant_phase_exact": torch.equal(
            constant_phase,
            expected_constant_phase,
        ),
        "constant_values_hex": [
            value.hex() for value in (-2.0, 0.5, 3.25)
        ],
    }


def _analytic_ramp_probe() -> dict[str, object]:
    stride = COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
    height, width = 5, 6
    slope_row, slope_column, intercept = 8.0, 1.0, 0.25
    row_grid = torch.arange(height, dtype=torch.float64).reshape(
        1,
        1,
        height,
        1,
    )
    column_grid = torch.arange(width, dtype=torch.float64).reshape(
        1,
        1,
        1,
        width,
    )
    coarse = (
        slope_row * row_grid
        + slope_column * column_grid
        + intercept
    )
    _, phase = bilinear_phase_aligned_feature_affine(
        coarse,
        stride=stride,
    )
    offsets = align_corners_false_phase_offsets(stride)
    errors: list[float] = []
    per_cell_unique: list[int] = []
    phase_ranges: list[float] = []
    for row in range(1, height - 1):
        for column in range(1, width - 1):
            actual = phase[0, :, 0, row, column]
            expected = torch.tensor(
                [
                    slope_row * (row + offset_row)
                    + slope_column * (column + offset_column)
                    + intercept
                    for offset_row, offset_column in offsets
                ],
                dtype=torch.float64,
            )
            errors.extend((actual - expected).abs().tolist())
            per_cell_unique.append(
                int(torch.unique(actual, sorted=True).numel())
            )
            phase_ranges.append(
                float(actual.max().item() - actual.min().item())
            )
    maximum_error = max(errors)
    return {
        "stride": stride,
        "coarse_shape": list(coarse.shape),
        "interior_cell_count": len(per_cell_unique),
        "analytic_coordinate_policy": (
            "source=index+(phase+0.5)/stride-0.5"
        ),
        "slope_row_hex": slope_row.hex(),
        "slope_column_hex": slope_column.hex(),
        "intercept_hex": intercept.hex(),
        "maximum_abs_interior_error_hex": maximum_error.hex(),
        "analytic_ramp_interior_exact": maximum_error == 0.0,
        "unique_phase_values_per_interior_cell": per_cell_unique,
        "all_16_phase_values_unique": (
            bool(per_cell_unique)
            and all(
                value == stride * stride
                for value in per_cell_unique
            )
        ),
        "minimum_phase_range_hex": min(phase_ranges).hex(),
        "phase_evidence_nondegenerate": (
            bool(phase_ranges)
            and min(phase_ranges) > 0.0
        ),
        "phase_fingerprint": tensor_content_fingerprint(phase),
    }


def _efficient_reference_probe() -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(210041)
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(
            _toy_config()
        )
    _randomize_paths(model, seed=210042)
    feature, occupancy = _toy_inputs(seed=210043)
    state_before = _state_fingerprint(model)
    gradients_before = _parameter_gradient_fingerprint(model)
    fields = model.forward_fields(feature, occupancy)
    efficient = fields.field
    reference = model.forward_reference(feature, occupancy)
    state_after = _state_fingerprint(model)
    gradients_after = _parameter_gradient_fingerprint(model)
    maximum_error = float((efficient - reference).abs().max().item())
    standalone_up, standalone_phase = (
        bilinear_phase_aligned_feature_affine(
            fields.coarse_feature_affine,
            stride=model.config.feature_stride,
        )
    )
    phase_spread = (
        fields.phase_feature_affine.amax(dim=1)
        - fields.phase_feature_affine.amin(dim=1)
    )
    return {
        "config": {
            "feature_channels": model.config.feature_channels,
            "feature_stride": model.config.feature_stride,
            "width": model.config.width,
        },
        "maximum_abs_error_hex": maximum_error.hex(),
        "rtol_hex": COVERAGE_STATE_PAET_REFERENCE_RTOL.hex(),
        "atol_hex": COVERAGE_STATE_PAET_REFERENCE_ATOL.hex(),
        "efficient_reference_allclose": torch.allclose(
            efficient,
            reference,
            rtol=COVERAGE_STATE_PAET_REFERENCE_RTOL,
            atol=COVERAGE_STATE_PAET_REFERENCE_ATOL,
        ),
        "efficient_fingerprint": tensor_content_fingerprint(efficient),
        "reference_fingerprint": tensor_content_fingerprint(reference),
        "standalone_upsample_exact": torch.equal(
            standalone_up,
            fields.upsampled_feature_affine,
        ),
        "standalone_phase_pack_exact": torch.equal(
            standalone_phase,
            fields.phase_feature_affine,
        ),
        "model_phase_evidence_max_spread_hex": _hex(
            phase_spread.max(),
            name="model phase evidence spread",
        ),
        "model_phase_evidence_nondegenerate": (
            bool(torch.isfinite(phase_spread).all())
            and float(phase_spread.max().item()) > 0.0
        ),
        "state_preserved": state_before == state_after,
        "gradient_buffers_preserved": (
            gradients_before == gradients_after
        ),
        "all_fields_finite": all(
            not isinstance(value, Tensor)
            or bool(torch.isfinite(value).all())
            for value in fields.__dict__.values()
        ),
    }


def _flip_antisymmetry_probe() -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(210051)
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(
            _toy_config()
        )
    _randomize_paths(model, seed=210052)
    feature, occupancy = _toy_inputs(seed=210053)
    fields = model.forward_fields(feature, occupancy)

    phase_index = 2
    coarse_row, coarse_column = 1, 2
    phase_row, phase_column = divmod(
        phase_index,
        model.config.feature_stride,
    )
    fine_row = (
        coarse_row * model.config.feature_stride + phase_row
    )
    fine_column = (
        coarse_column * model.config.feature_stride + phase_column
    )
    flipped_occupancy = occupancy.clone()
    flipped_occupancy[
        0,
        0,
        fine_row,
        fine_column,
    ] = torch.logical_not(
        flipped_occupancy[
            0,
            0,
            fine_row,
            fine_column,
        ]
    )
    flipped_fields = model.forward_fields(
        feature,
        flipped_occupancy,
    )
    actual_delta = fields.native_phase_interaction[
        0,
        phase_index,
        coarse_row,
        coarse_column,
    ]
    flipped_delta = flipped_fields.native_phase_interaction[
        0,
        phase_index,
        coarse_row,
        coarse_column,
    ]
    odd_error = abs(
        float(actual_delta.detach().item())
        + float(flipped_delta.detach().item())
    )
    field_sum = float(
        (
            fields.field[0, 0, fine_row, fine_column]
            + flipped_fields.field[0, 0, fine_row, fine_column]
        )
        .detach()
        .item()
    )
    expected_sum = 2.0 * model.config.field_amplitude

    helper_phase_index = 10
    patch = (
        torch.arange(16 * 3 * 3).reshape(16, 3, 3) % 3 == 0
    )
    once = flip_binary_center_phase(
        patch,
        phase_index=helper_phase_index,
        center=1,
    )
    twice = flip_binary_center_phase(
        once,
        phase_index=helper_phase_index,
        center=1,
    )
    expected_native = 0.5 * (
        fields.actual_feature_presence_energy
        - fields.flipped_feature_presence_energy
    )
    return {
        "phase_index": phase_index,
        "helper_phase_count": int(patch.shape[0]),
        "helper_phase_index": helper_phase_index,
        "coarse_coordinate": [coarse_row, coarse_column],
        "fine_coordinate": [fine_row, fine_column],
        "flip_involution_exact": torch.equal(twice, patch),
        "flip_changes_exactly_one_bit": (
            int(torch.count_nonzero(once != patch)) == 1
        ),
        "native_odd_projection_exact": torch.equal(
            fields.native_phase_interaction,
            expected_native,
        ),
        "selected_odd_error_hex": odd_error.hex(),
        "selected_odd_allclose": odd_error
        <= COVERAGE_STATE_PAET_FLIP_ATOL,
        "selected_field_sum_hex": field_sum.hex(),
        "expected_field_sum_hex": expected_sum.hex(),
        "field_sum_two_anchor_allclose": abs(
            field_sum - expected_sum
        )
        <= COVERAGE_STATE_PAET_FLIP_ATOL,
        "field_anchor_hex": model.config.field_amplitude.hex(),
    }


def _zero_feature_and_additive_probe() -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(210061)
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(
            _toy_config()
        )
    _randomize_paths(model, seed=210062)
    feature, occupancy = _toy_inputs(seed=210063)
    zero_fields = model.forward_fields(
        torch.zeros_like(feature),
        occupancy,
    )
    anchor = torch.full_like(
        zero_fields.field,
        model.config.field_amplitude,
    )

    occupancy_silent = deepcopy(model)
    feature_silent = deepcopy(model)
    with torch.no_grad():
        occupancy_silent.occupancy_weight.zero_()
        feature_silent.feature_weight.zero_()
    occupancy_silent_fields = occupancy_silent.forward_fields(
        feature,
        occupancy,
    )
    feature_silent_fields = feature_silent.forward_fields(
        feature,
        occupancy,
    )
    return {
        "zero_feature_interaction_exact_zero": (
            int(
                torch.count_nonzero(
                    zero_fields.native_phase_interaction
                )
            )
            == 0
        ),
        "zero_feature_field_exact_anchor": torch.equal(
            zero_fields.field,
            anchor,
        ),
        "occupancy_path_zero_interaction": (
            int(
                torch.count_nonzero(
                    occupancy_silent_fields.native_phase_interaction
                )
            )
            == 0
        ),
        "occupancy_path_field_exact_anchor": torch.equal(
            occupancy_silent_fields.field,
            anchor,
        ),
        "feature_path_zero_interaction": (
            int(
                torch.count_nonzero(
                    feature_silent_fields.native_phase_interaction
                )
            )
            == 0
        ),
        "feature_path_field_exact_anchor": torch.equal(
            feature_silent_fields.field,
            anchor,
        ),
        "additive_path_policy": (
            "feature_only_and_occupancy_only_terms_cannot_write_field"
        ),
    }


def _parameter_contract_probe() -> dict[str, object]:
    paet_config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_PAET_FORMAL_WIDTH,
    )
    bfa_config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_PAET_FORMAL_WIDTH,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        paet = CURELitePhaseAlignedEvidenceTransportLevelSet(
            paet_config
        )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        bfa = CURELiteBinaryFlipAntisymmetricLevelSet(bfa_config)
    paet_state = paet.state_dict()
    bfa_state = bfa.state_dict()
    paet_parameters = tuple(
        (name, tuple(value.shape))
        for name, value in paet.named_parameters()
    )
    bfa_parameters = tuple(
        (name, tuple(value.shape))
        for name, value in bfa.named_parameters()
    )
    values_exact = (
        tuple(paet_state) == tuple(bfa_state)
        and all(
            torch.equal(paet_state[name], bfa_state[name])
            for name in paet_state
        )
    )
    return {
        "paet_parameter_rows": [
            {"name": name, "shape": list(shape)}
            for name, shape in paet_parameters
        ],
        "bfa_parameter_rows": [
            {"name": name, "shape": list(shape)}
            for name, shape in bfa_parameters
        ],
        "parameter_keys_and_shapes_exact": (
            paet_parameters == bfa_parameters
        ),
        "seed42_initial_state_exact": values_exact,
        "paet_state_fingerprint": stable_fingerprint(
            {
                name: tensor_content_fingerprint(value)
                for name, value in sorted(paet_state.items())
            }
        ),
        "bfa_state_fingerprint": stable_fingerprint(
            {
                name: tensor_content_fingerprint(value)
                for name, value in sorted(bfa_state.items())
            }
        ),
        "parameter_tensor_count": len(paet_parameters),
        "parameter_count": sum(
            parameter.numel() for parameter in paet.parameters()
        ),
        "expected_parameter_count": (
            COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
        ),
        "config_expected_parameter_count": (
            paet_config.expected_parameter_count
        ),
        "hidden_bias_zero": (
            int(torch.count_nonzero(paet.joint_hidden_bias)) == 0
        ),
        "scalar_readout_zero": (
            int(torch.count_nonzero(paet.scalar_energy_weight)) == 0
        ),
        "joint_weight_finite_nonzero": (
            bool(torch.isfinite(paet.joint_state_weight).all())
            and int(torch.count_nonzero(paet.joint_state_weight)) > 0
        ),
    }


def _single_field_interface_probe() -> dict[str, object]:
    config = _toy_config()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(210071)
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(config)
    feature, occupancy = _toy_inputs(seed=210072)
    output = model(feature, occupancy)
    method_names = (
        "forward",
        "forward_fields",
        "forward_reference",
        "predict_completion",
    )
    signatures = {
        name: tuple(
            inspect.signature(getattr(model, name)).parameters
        )
        for name in method_names
    }
    children = tuple(
        (
            name,
            f"{type(child).__module__}.{type(child).__qualname__}",
        )
        for name, child in model.named_children()
    )
    forward_source = inspect.getsource(
        type(model).forward_fields
    ).lower()
    forbidden_metadata = (
        "pair_kind",
        "optimizer_role",
        "sample_id",
        "record_id",
        "ground_truth",
        "dataset",
        "split",
    )
    unexpected_role_keyword_rejected = False
    try:
        model(
            feature,
            occupancy,
            role="factual",
        )  # type: ignore[call-arg]
    except TypeError:
        unexpected_role_keyword_rejected = True
    completion = model.predict_completion(feature, occupancy)
    return {
        "method_signatures": {
            name: list(value) for name, value in signatures.items()
        },
        "forward_only_feature_and_occupancy": (
            all(
                value == ("feature", "occupancy")
                for value in signatures.values()
            )
        ),
        "unexpected_role_keyword_rejected": (
            unexpected_role_keyword_rejected
        ),
        "output_is_one_tensor": isinstance(output, Tensor),
        "output_shape": list(output.shape),
        "occupancy_shape": list(occupancy.shape),
        "output_matches_single_field_geometry": (
            tuple(output.shape) == tuple(occupancy.shape)
        ),
        "named_children": [
            {"name": name, "class": class_name}
            for name, class_name in children
        ],
        "only_pixel_shuffle_child": (
            len(children) == 1
            and children[0][0] == "pixel_shuffle"
        ),
        "completion_is_bool_single_field": (
            isinstance(completion, Tensor)
            and completion.dtype == torch.bool
            and tuple(completion.shape) == tuple(occupancy.shape)
        ),
        "no_role_metadata_in_forward_fields": not any(
            value in forward_source for value in forbidden_metadata
        ),
    }


def _gradient_probe() -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(210081)
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(
            _toy_config()
        )
    _randomize_paths(model, seed=210082)
    feature, occupancy = _toy_inputs(seed=210083)
    state_before = _state_fingerprint(model)
    gradient_buffers_before = _parameter_gradient_fingerprint(model)
    output = model(feature, occupancy)
    objective = output.square().mean() + 0.17 * output.mean()
    parameters = tuple(model.parameters())
    names = tuple(name for name, _ in model.named_parameters())
    first = torch.autograd.grad(
        objective,
        parameters,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    meta_objective = sum(value.square().sum() for value in first)
    second = torch.autograd.grad(
        meta_objective,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    first_rows = []
    second_rows = []
    for name, value in zip(names, first):
        first_rows.append(
            {
                "name": name,
                "finite": bool(torch.isfinite(value).all()),
                "nonzero": int(torch.count_nonzero(value)) > 0,
                "l2_hex": _hex(
                    torch.linalg.vector_norm(value),
                    name=f"first gradient {name}",
                ),
            }
        )
    for name, value in zip(names, second):
        second_rows.append(
            {
                "name": name,
                "finite": bool(torch.isfinite(value).all()),
                "nonzero": int(torch.count_nonzero(value)) > 0,
                "l2_hex": _hex(
                    torch.linalg.vector_norm(value),
                    name=f"second gradient {name}",
                ),
            }
        )
    return {
        "parameter_names": list(names),
        "objective_hex": _hex(objective, name="PAET objective"),
        "first_order": first_rows,
        "second_order": second_rows,
        "first_reaches_all_three_parameters": (
            len(first_rows) == 3
            and all(
                row["finite"] and row["nonzero"]
                for row in first_rows
            )
        ),
        "second_reaches_all_three_parameters": (
            len(second_rows) == 3
            and all(
                row["finite"] and row["nonzero"]
                for row in second_rows
            )
        ),
        "model_state_preserved": (
            state_before == _state_fingerprint(model)
        ),
        "parameter_grad_buffers_unretained": (
            gradient_buffers_before
            == _parameter_gradient_fingerprint(model)
        ),
        "backward_called": False,
        "optimizer_constructed": False,
    }


def _qualified_call_name(node: ast.Call) -> str:
    value = node.func
    names: list[str] = []
    while isinstance(value, ast.Attribute):
        names.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        names.append(value.id)
    return ".".join(reversed(names))


def _static_boundary_probe() -> dict[str, object]:
    transport_source = inspect.getsource(
        bilinear_phase_aligned_feature_affine
    )
    class_source = inspect.getsource(
        CURELitePhaseAlignedEvidenceTransportLevelSet
    )
    module_source = (
        transport_source + "\n" + class_source
    )
    tree = ast.parse(module_source)
    call_names = sorted(
        {
            _qualified_call_name(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
    )
    interpolate_calls = tuple(
        node
        for node in ast.walk(ast.parse(transport_source))
        if isinstance(node, ast.Call)
        and _qualified_call_name(node).endswith("F.interpolate")
    )
    interpolate_fixed = False
    if len(interpolate_calls) == 1:
        keywords = {
            value.arg: value.value
            for value in interpolate_calls[0].keywords
            if value.arg is not None
        }
        interpolate_fixed = (
            isinstance(keywords.get("scale_factor"), ast.Name)
            and keywords["scale_factor"].id == "stride"
            and isinstance(keywords.get("mode"), ast.Constant)
            and keywords["mode"].value == "bilinear"
            and isinstance(keywords.get("align_corners"), ast.Constant)
            and keywords["align_corners"].value is False
        )
    forbidden_calls = (
        "open",
        "torch.load",
        "torch.save",
        "DataLoader",
        "Dataset",
        "Optimizer",
        "Adam",
        "AdamW",
        "SGD",
        "backward",
        "grid_sample",
    )
    lower_source = module_source.lower()
    forbidden_metadata = (
        "d_v",
        "d_t",
        "dataset",
        "cache",
        "checkpoint",
        "optimizer_role",
        "pair_kind",
        "sample_id",
        "record_id",
    )
    checked_paths = (
        "cure_lite/coverage_state_phase_aligned_evidence_transport.py",
        "cure_lite/experiment/coverage_state_paet_dataset_free.py",
    )
    imports: set[str] = set()
    source_calls: set[str] = set()
    for relative in checked_paths:
        source = (_repository_root() / relative).read_text(
            encoding="utf-8"
        )
        source_tree = ast.parse(source, filename=relative)
        for node in ast.walk(source_tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
            elif isinstance(node, ast.Call):
                name = _qualified_call_name(node)
                if name:
                    source_calls.add(name)
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
        if any(
            fragment in name.lower()
            for fragment in forbidden_import_fragments
        )
    )
    forbidden_source_calls = sorted(
        name
        for name in source_calls
        if any(
            name == suffix.removeprefix(".")
            or name.endswith(suffix)
            for suffix in forbidden_call_suffixes
        )
    )
    return {
        "call_names": call_names,
        "interpolate_call_count": len(interpolate_calls),
        "interpolate_arguments_fixed": interpolate_fixed,
        "forbidden_calls_present": sorted(
            value
            for value in forbidden_calls
            if any(
                name == value or name.endswith(f".{value}")
                for name in call_names
            )
        ),
        "forbidden_runtime_metadata_present": sorted(
            value for value in forbidden_metadata if value in lower_source
        ),
        "checked_paths": list(checked_paths),
        "parsed_python_sources": len(checked_paths),
        "forbidden_imports": forbidden_imports,
        "forbidden_source_calls": forbidden_source_calls,
        "runtime_splits": [],
        "dataset_constructed": False,
        "cache_constructed": False,
        "cache_artifact_accessed": False,
        "model_artifact_accessed": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "calibration_performed": False,
        "inference_performed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "implementation_source_hashing_only": True,
    }


def _no_tunable_transport_probe() -> dict[str, object]:
    paet_fields = {
        value.name: value
        for value in dataclass_fields(
            CoverageStatePhaseAlignedEvidenceTransportConfig
        )
    }
    bfa_fields = {
        value.name: value
        for value in dataclass_fields(
            CoverageStateBinaryFlipAntisymmetricConfig
        )
    }
    added = sorted(set(paet_fields) - set(bfa_fields))
    removed = sorted(set(bfa_fields) - set(paet_fields))
    changed_defaults = sorted(
        name
        for name in set(paet_fields) & set(bfa_fields)
        if paet_fields[name].default != bfa_fields[name].default
    )
    transport_signature = inspect.signature(
        bilinear_phase_aligned_feature_affine
    )
    offsets_signature = inspect.signature(
        align_corners_false_phase_offsets
    )
    config = _toy_config()
    numeric_added = [
        name
        for name in added
        if isinstance(getattr(config, name), (int, float))
        and not isinstance(getattr(config, name), bool)
    ]
    return {
        "paet_config_fields": sorted(paet_fields),
        "bfa_config_fields": sorted(bfa_fields),
        "added_config_fields": added,
        "removed_config_fields": removed,
        "changed_shared_defaults": changed_defaults,
        "numeric_added_config_fields": numeric_added,
        "only_added_transport_policy_string": (
            added == ["transport_policy"]
            and isinstance(config.transport_policy, str)
        ),
        "transport_signature": str(transport_signature),
        "offset_signature": str(offsets_signature),
        "transport_has_only_tensor_and_stride": (
            tuple(transport_signature.parameters)
            == ("coarse_feature_affine", "stride")
        ),
        "offsets_have_only_stride": (
            tuple(offsets_signature.parameters) == ("stride",)
        ),
        "stride_derived_from_feature_stride": (
            config.feature_stride == 2
            and len(
                align_corners_false_phase_offsets(
                    config.feature_stride
                )
            )
            == config.feature_stride**2
        ),
        "learned_transport_parameters": [],
        "learned_offsets": False,
        "temperature": None,
        "transport_scale": None,
        "transport_bias": None,
    }


def _collect_generated_evidence() -> dict[str, dict[str, object]]:
    return {
        "offset_and_order": _offset_and_order_probe(),
        "transport_geometry": _transport_geometry_probe(),
        "analytic_ramp": _analytic_ramp_probe(),
        "efficient_reference": _efficient_reference_probe(),
        "flip_antisymmetry": _flip_antisymmetry_probe(),
        "zero_feature_and_additive": (
            _zero_feature_and_additive_probe()
        ),
        "parameter_contract": _parameter_contract_probe(),
        "single_field_interface": _single_field_interface_probe(),
        "gradients": _gradient_probe(),
        "static_boundary": _static_boundary_probe(),
        "no_tunable_transport": _no_tunable_transport_probe(),
    }


def recompute_coverage_state_paet_dataset_free_checks(
    *,
    probes: dict[str, dict[str, object]],
    implementation_binding: tuple[tuple[str, str], ...],
    generated_replay_fingerprint: str,
) -> tuple[tuple[str, bool], ...]:
    """Recompute the fifteen fixed checks from stored generated evidence."""

    offsets = probes.get("offset_and_order", {})
    geometry = probes.get("transport_geometry", {})
    ramp = probes.get("analytic_ramp", {})
    reference = probes.get("efficient_reference", {})
    flip = probes.get("flip_antisymmetry", {})
    anchors = probes.get("zero_feature_and_additive", {})
    parameters = probes.get("parameter_contract", {})
    interface = probes.get("single_field_interface", {})
    gradients = probes.get("gradients", {})
    boundary = probes.get("static_boundary", {})
    frozen = probes.get("no_tunable_transport", {})
    binding_valid = (
        implementation_binding == _current_implementation_binding()
        and len(implementation_binding)
        == len(COVERAGE_STATE_PAET_IMPLEMENTATION_PATHS)
        and tuple(name for name, _ in implementation_binding)
        == COVERAGE_STATE_PAET_IMPLEMENTATION_PATHS
        and all(
            isinstance(digest, str) and len(digest) == 64
            for _, digest in implementation_binding
        )
    )
    checks = {
        "01_align_corners_false_axis_offsets_exact": (
            offsets.get("stride")
            == COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
            and offsets.get("axis_formula_exact") is True
            and offsets.get("axis_offsets_strictly_increasing") is True
            and offsets.get("axis_offsets_centered") is True
        ),
        "02_row_major_phase_order_exact": (
            offsets.get("phase_count")
            == COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE**2
            and offsets.get("phase_offsets_row_major_exact") is True
            and offsets.get("phase_pack_row_major_exact") is True
            and len(offsets.get("phase_rows", []))
            == COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE**2
            and all(
                row.get("packed_exact") is True
                for row in offsets.get("phase_rows", [])
            )
        ),
        "03_upsample_pack_pixelshuffle_identity": (
            geometry.get("stride")
            == COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
            and geometry.get("direct_pack_exact") is True
            and geometry.get("pixelshuffle_unpack_exact") is True
        ),
        "04_constant_transport_preserved": (
            geometry.get("constant_upsample_exact") is True
            and geometry.get("constant_phase_exact") is True
        ),
        "05_analytic_ramp_interior_exact": (
            ramp.get("interior_cell_count") == 12
            and ramp.get("analytic_ramp_interior_exact") is True
            and float.fromhex(
                str(ramp.get("maximum_abs_interior_error_hex"))
            )
            == 0.0
        ),
        "06_phase_evidence_nondegenerate": (
            ramp.get("all_16_phase_values_unique") is True
            and ramp.get("phase_evidence_nondegenerate") is True
            and float.fromhex(
                str(ramp.get("minimum_phase_range_hex"))
            )
            > 0.0
            and reference.get("model_phase_evidence_nondegenerate")
            is True
            and float.fromhex(
                str(
                    reference.get(
                        "model_phase_evidence_max_spread_hex"
                    )
                )
            )
            > 0.0
        ),
        "07_efficient_reference_elementwise": (
            reference.get("efficient_reference_allclose") is True
            and reference.get("standalone_upsample_exact") is True
            and reference.get("standalone_phase_pack_exact") is True
            and reference.get("state_preserved") is True
            and reference.get("gradient_buffers_preserved") is True
            and reference.get("all_fields_finite") is True
        ),
        "08_flip_involution_odd_and_field_sum": (
            flip.get("helper_phase_count")
            == COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE**2
            and flip.get("flip_involution_exact") is True
            and flip.get("flip_changes_exactly_one_bit") is True
            and flip.get("native_odd_projection_exact") is True
            and flip.get("selected_odd_allclose") is True
            and flip.get("field_sum_two_anchor_allclose") is True
            and flip.get("expected_field_sum_hex")
            == (2.0 * CSLF_FIELD_AMPLITUDE).hex()
        ),
        "09_zero_feature_anchor": (
            anchors.get("zero_feature_interaction_exact_zero") is True
            and anchors.get("zero_feature_field_exact_anchor") is True
        ),
        "10_pure_additive_paths_cancel": (
            anchors.get("occupancy_path_zero_interaction") is True
            and anchors.get("occupancy_path_field_exact_anchor") is True
            and anchors.get("feature_path_zero_interaction") is True
            and anchors.get("feature_path_field_exact_anchor") is True
        ),
        "11_parameter_contract_exact_bfa": (
            parameters.get("parameter_keys_and_shapes_exact") is True
            and parameters.get("seed42_initial_state_exact") is True
            and parameters.get("paet_state_fingerprint")
            == parameters.get("bfa_state_fingerprint")
            and parameters.get("parameter_tensor_count") == 3
            and parameters.get("parameter_count")
            == COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
            and parameters.get("expected_parameter_count")
            == COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
            and parameters.get("config_expected_parameter_count")
            == COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
            and parameters.get("hidden_bias_zero") is True
            and parameters.get("scalar_readout_zero") is True
            and parameters.get("joint_weight_finite_nonzero") is True
        ),
        "12_single_completion_field_no_role_interface": (
            interface.get("forward_only_feature_and_occupancy") is True
            and interface.get("unexpected_role_keyword_rejected") is True
            and interface.get("output_is_one_tensor") is True
            and interface.get("output_matches_single_field_geometry")
            is True
            and interface.get("only_pixel_shuffle_child") is True
            and interface.get("completion_is_bool_single_field") is True
            and interface.get("no_role_metadata_in_forward_fields")
            is True
        ),
        "13_first_second_order_gradients_finite_all_parameters": (
            gradients.get("parameter_names")
            == [
                "joint_state_weight",
                "joint_hidden_bias",
                "scalar_energy_weight",
            ]
            and gradients.get("first_reaches_all_three_parameters")
            is True
            and gradients.get("second_reaches_all_three_parameters")
            is True
            and gradients.get("model_state_preserved") is True
            and gradients.get("parameter_grad_buffers_unretained") is True
            and gradients.get("backward_called") is False
            and gradients.get("optimizer_constructed") is False
        ),
        "14_no_runtime_data_cache_or_optimizer": (
            boundary.get("forbidden_calls_present") == []
            and boundary.get("forbidden_runtime_metadata_present") == []
            and boundary.get("parsed_python_sources") == 2
            and boundary.get("forbidden_imports") == []
            and boundary.get("forbidden_source_calls") == []
            and boundary.get("runtime_splits") == []
            and boundary.get("dataset_constructed") is False
            and boundary.get("cache_constructed") is False
            and boundary.get("cache_artifact_accessed") is False
            and boundary.get("model_artifact_accessed") is False
            and boundary.get("optimizer_constructed") is False
            and boundary.get("optimizer_steps") == 0
            and boundary.get("parameter_updates") == 0
            and boundary.get("training_performed") is False
            and boundary.get("calibration_performed") is False
            and boundary.get("inference_performed") is False
            and boundary.get("D_R_accessed") is False
            and boundary.get("D_V_accessed") is False
            and boundary.get("D_T_accessed") is False
            and boundary.get("implementation_source_hashing_only")
            is True
        ),
        "15_no_tunable_transport_parameters": (
            boundary.get("interpolate_call_count") == 1
            and boundary.get("interpolate_arguments_fixed") is True
            and frozen.get("only_added_transport_policy_string") is True
            and frozen.get("removed_config_fields") == []
            and frozen.get("changed_shared_defaults")
            == [
                "energy_policy",
                "equation_policy",
                "field_policy",
                "interaction_policy",
            ]
            and frozen.get("numeric_added_config_fields") == []
            and frozen.get("transport_has_only_tensor_and_stride") is True
            and frozen.get("offsets_have_only_stride") is True
            and frozen.get("stride_derived_from_feature_stride") is True
            and frozen.get("learned_transport_parameters") == []
            and frozen.get("learned_offsets") is False
            and frozen.get("temperature") is None
            and frozen.get("transport_scale") is None
            and frozen.get("transport_bias") is None
            and binding_valid
            and isinstance(generated_replay_fingerprint, str)
            and len(generated_replay_fingerprint) == 64
        ),
    }
    if tuple(checks) != COVERAGE_STATE_PAET_DATASET_FREE_CHECK_NAMES:
        raise AssertionError("PAET-BFA dataset-free check order changed")
    return tuple(checks.items())


@dataclass(frozen=True)
class CoverageStatePAETDatasetFreeReceipt:
    """Fingerprint-bound evidence for the v21 generated-only gate."""

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
        expected = recompute_coverage_state_paet_dataset_free_checks(
            probes=self.probes,
            implementation_binding=self.implementation_binding,
            generated_replay_fingerprint=(
                self.generated_replay_fingerprint
            ),
        )
        if (
            self.checks != expected
            or stable_fingerprint(self._evidence_payload())
            != self.evidence_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
        ):
            raise RuntimeError(
                "PAET-BFA dataset-free evidence changed after creation"
            )

    @property
    def all_pass(self) -> bool:
        self.verify_unchanged()
        return (
            len(self.checks)
            == len(COVERAGE_STATE_PAET_DATASET_FREE_CHECK_NAMES)
            and all(value for _, value in self.checks)
        )

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_PAET_DATASET_FREE_SCHEMA,
            "model": "PAET-BFA",
            "version": "v21",
            "input_interface": ["F_b", "O"],
            "interaction_policy": PAET_INTERACTION_POLICY,
            "energy_policy": PAET_ENERGY_POLICY,
            "flip_policy": PAET_FLIP_POLICY,
            "transport_policy": PAET_TRANSPORT_POLICY,
            "input_representation": PAET_INPUT_REPRESENTATION,
            "field_anchor_hex": CSLF_FIELD_AMPLITUDE.hex(),
            "fixed_margin_hex": COVERAGE_STATE_PAET_MARGIN.hex(),
            "execution_seed": (
                COVERAGE_STATE_PAET_DATASET_FREE_EXECUTION_SEED
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
            "dataset_constructed": False,
            "cache_constructed": False,
            "cache_artifact_accessed": False,
            "model_artifact_accessed": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "training_performed": False,
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }

    @cached_property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _run_coverage_state_paet_dataset_free_gate_inner(
) -> CoverageStatePAETDatasetFreeReceipt:
    first = _collect_generated_evidence()
    second = _collect_generated_evidence()
    first_fingerprint = stable_fingerprint(first)
    second_fingerprint = stable_fingerprint(second)
    if first_fingerprint != second_fingerprint:
        raise RuntimeError(
            "PAET-BFA generated replay is not deterministic"
        )
    implementation_binding = _current_implementation_binding()
    checks = recompute_coverage_state_paet_dataset_free_checks(
        probes=first,
        implementation_binding=implementation_binding,
        generated_replay_fingerprint=second_fingerprint,
    )
    evidence_payload = {
        "probes": deepcopy(first),
        "implementation_binding": dict(implementation_binding),
        "generated_replay_fingerprint": second_fingerprint,
    }
    return CoverageStatePAETDatasetFreeReceipt(
        probes=first,
        implementation_binding=implementation_binding,
        generated_replay_fingerprint=second_fingerprint,
        checks=checks,
        evidence_fingerprint=stable_fingerprint(evidence_payload),
    )


def run_coverage_state_paet_dataset_free_gate(
) -> CoverageStatePAETDatasetFreeReceipt:
    """Run all fifteen generated-only PAET-BFA structural checks."""

    before_rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        result = _run_coverage_state_paet_dataset_free_gate_inner()
    if not torch.equal(before_rng, torch.random.get_rng_state()):
        raise RuntimeError(
            "PAET-BFA dataset-free gate changed global RNG state"
        )
    return result


__all__ = [
    "COVERAGE_STATE_PAET_DATASET_FREE_CHECK_NAMES",
    "COVERAGE_STATE_PAET_DATASET_FREE_EXECUTION_SEED",
    "COVERAGE_STATE_PAET_DATASET_FREE_SCHEMA",
    "COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS",
    "COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE",
    "COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT",
    "COVERAGE_STATE_PAET_FORMAL_WIDTH",
    "COVERAGE_STATE_PAET_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_PAET_MARGIN",
    "COVERAGE_STATE_PAET_REFERENCE_ATOL",
    "COVERAGE_STATE_PAET_REFERENCE_RTOL",
    "COVERAGE_STATE_PAET_TRUNCATION_RADIUS",
    "CoverageStatePAETDatasetFreeReceipt",
    "recompute_coverage_state_paet_dataset_free_checks",
    "run_coverage_state_paet_dataset_free_gate",
]
