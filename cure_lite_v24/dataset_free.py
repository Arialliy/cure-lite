"""Generated-only pre-``D_R`` evidence gate for v24 GCR-PACRE.

This module has no dataset, cache, checkpoint, calibration, ``D_R``, ``D_V``,
or ``D_T`` entry point.  It builds fixed tensors in memory, exercises only
public model/training APIs, performs exactly one real PMOPE/Adam warm-up
update, and returns a self-verifying JSON-compatible receipt.
"""

from __future__ import annotations

from dataclasses import asdict, fields as dataclass_fields, is_dataclass
import io
from math import ceil, isfinite
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from types import MethodType
from typing import Final, Mapping

import torch
from torch import Tensor

from cure_lite.cache.schema import (
    file_sha256,
    stable_fingerprint,
)
from cure_lite.coverage_state_batches import (
    CoverageStateFusedBatch,
    CoverageStateNaturalTrainBatch,
    CoverageStatePairTrainBatch,
)
from cure_lite.coverage_state_level_set import CSLF_FIELD_AMPLITUDE
from cure_lite.coverage_state_sobolev import (
    CSLF_PMOPE_POLICY,
    CoverageStateAbsoluteTargets,
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    prepare_coverage_state_focused_absolute_targets,
    prepare_coverage_state_pair_targets,
)
from cure_lite.experiment.coverage_state_training import (
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
    coverage_state_fused_train_step,
)
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)

from .factory import (
    GCR_PACRE_FORMAL_FEATURE_CHANNELS,
    GCR_PACRE_FORMAL_FEATURE_STRIDE,
    GCR_PACRE_FORMAL_PARAMETER_COUNT,
    GCR_PACRE_FORMAL_WIDTH,
    GCR_PACRE_PARAMETER_NAMES,
    build_formal_gcr_pacre_training_model,
    build_gcr_pacre_training_model,
)
from .gcr_pacre import (
    CSLF_GCR_PACRE_EQUATION_POLICY,
    CSLF_GCR_PACRE_FIELD_POLICY,
    GCR_PACRE_CANDIDATE,
    GCR_PACRE_CENTERING_POLICY,
    GCR_PACRE_ENERGY_POLICY,
    GCR_PACRE_FP64_ORACLE_ABS_TOL,
    GCR_PACRE_FP64_ORACLE_MAX_ULP,
    GCR_PACRE_INTERACTION_POLICY,
    GCR_PACRE_METHOD_ID,
    GCR_PACRE_NUMERICAL_POLICY,
    CURELiteGatedCommonResidualPACRELevelSet,
    CoverageStateGCRPACREConfig,
    compare_gcr_pacre_fp32_to_fp64_oracle,
    summarize_gcr_pacre_gate_saturation,
    validate_gcr_pacre_fields,
)


GCR_PACRE_DATASET_FREE_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-dataset-free-receipt-v1"
)
GCR_PACRE_DATASET_FREE_DECISION_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-dataset-free-decision-v1"
)
GCR_PACRE_DATASET_FREE_EXECUTION_SEED: Final = 240_100
GCR_PACRE_DATASET_FREE_MODEL_SEED: Final = 240_101
GCR_PACRE_DATASET_FREE_INPUT_SEED: Final = 240_102
GCR_PACRE_DATASET_FREE_WARMUP_SEED: Final = 240_103
GCR_PACRE_DATASET_FREE_FORMAL_SEED: Final = 42
GCR_PACRE_DATASET_FREE_STEP0_GRADIENT_ATOL: Final = 2.0e-6
GCR_PACRE_DATASET_FREE_SELECTIVITY_MARGIN: Final = 1.0e-5
GCR_PACRE_DATASET_FREE_WARMUP_UPDATES: Final = 1
GCR_PACRE_EFFICIENCY_FORWARD_WARMUPS: Final = 2
GCR_PACRE_EFFICIENCY_FORWARD_REPEATS: Final = 5
GCR_PACRE_EFFICIENCY_TRAIN_WARMUPS: Final = 1
GCR_PACRE_EFFICIENCY_TRAIN_REPEATS: Final = 3
GCR_PACRE_EFFICIENCY_FEATURE_HEIGHT: Final = 1
GCR_PACRE_EFFICIENCY_FEATURE_WIDTH: Final = 1

GCR_PACRE_DATASET_FREE_SOURCE_PATHS: Final = (
    "cure_lite/cache/schema.py",
    "cure_lite/coverage_state_batches.py",
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_phase_aligned_evidence_transport.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/experiment/coverage_state_training.py",
    "cure_lite/paired_types.py",
    "cure_lite/train/coverage_state_fused_step.py",
    "cure_lite_v23/__init__.py",
    "cure_lite_v23/pacre_vc.py",
    "cure_lite_v24/__init__.py",
    "cure_lite_v24/gcr_pacre.py",
    "cure_lite_v24/factory.py",
    "cure_lite_v24/dataset_free.py",
    "tools/audit_cure_lite_v24_gcr_pacre_dataset_free.py",
)

GCR_PACRE_DATASET_FREE_CHECK_NAMES: Final = (
    "01_generated_boundary_closed",
    "02_source_binding_current",
    "03_canonical_v24_identity_exact",
    "04_formal_parameter_count_64064",
    "05_parameter_names_shapes_and_initial_bytes_match_v23",
    "06_single_scalar_field_no_additional_head",
    "07_lightweight_validator_called_by_forward",
    "08_full_fields_replay_pass",
    "09_fp64_oracle_envelope_pass",
    "10_non_unit_gate_proves_gcr_reference",
    "11_gate_finite_closed_and_statistics_complete",
    "12_gate_endpoint_and_interior_witnesses_present",
    "13_reference_common_even_flip_symmetric",
    "14_reference_residual_odd_flip_antisymmetric",
    "15_reference_gate_flip_symmetric",
    "16_reference_gated_interaction_flip_antisymmetric",
    "17_zero_common_energy_maps_to_unit_gate",
    "18_zero_residual_maps_to_zero_interaction",
    "19_zero_feature_field_exact_positive_point_nine",
    "20_target_like_gate_boosts_negative_residual",
    "21_background_like_gate_suppresses_negative_residual",
    "22_common_only_evidence_creates_no_completion",
    "23_forced_unit_gate_is_read_only",
    "24_fixed_zero_threshold_and_hard_union_exact",
    "25_step0_v23_output_and_gradient_equivalence",
    "26_one_real_frozen_pmope_warmup_update",
    "27_post_warmup_residual_path_gradient_nonzero",
    "28_post_warmup_gate_path_gradient_nonzero",
    "29_warmup_total_pmope_gradient_reaches_shared_readout",
    "30_efficiency_audit_complete_finite_and_threshold_free",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _current_source_hashes() -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    result: list[tuple[str, str]] = []
    for relative in GCR_PACRE_DATASET_FREE_SOURCE_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"GCR-PACRE dataset-free source is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _fingerprinted(
    body: Mapping[str, object],
    *,
    field: str,
) -> dict[str, object]:
    payload = dict(body)
    return {**payload, field: stable_fingerprint(payload)}


def _verify_section(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    payload = dict(value)
    fingerprint = payload.pop("section_fingerprint", None)
    if (
        not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(payload)
    ):
        raise ValueError(f"{name} section_fingerprint is invalid")
    return dict(value)


def _tensor_map_fingerprint(
    values: Mapping[str, Tensor],
) -> str:
    return stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(values.items())
        }
    )


def _model_state_fingerprint(model: torch.nn.Module) -> str:
    return _tensor_map_fingerprint(
        {
            name: value
            for name, value in model.state_dict().items()
        }
    )


def _toy_config(
    *,
    channels: int = 2,
    stride: int = 2,
    width: int = 4,
) -> CoverageStateGCRPACREConfig:
    return CoverageStateGCRPACREConfig(
        feature_channels=channels,
        feature_stride=stride,
        width=width,
    )


def _generated_inputs(
    *,
    seed: int = GCR_PACRE_DATASET_FREE_INPUT_SEED,
    channels: int = 2,
    stride: int = 2,
    height: int = 3,
    width: int = 4,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    feature = torch.randn(
        (1, channels, height, width),
        generator=generator,
        dtype=torch.float32,
    )
    occupancy = (
        torch.rand(
            (1, 1, height * stride, width * stride),
            generator=generator,
            dtype=torch.float32,
        )
        > 0.61
    )
    return feature.contiguous(), occupancy.contiguous()


def _randomized_toy_model(
    *,
    seed: int = GCR_PACRE_DATASET_FREE_MODEL_SEED,
) -> CURELiteGatedCommonResidualPACRELevelSet:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = build_gcr_pacre_training_model(_toy_config())
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    with torch.no_grad():
        model.joint_state_weight.copy_(
            0.3
            * torch.randn(
                model.joint_state_weight.shape,
                generator=generator,
                dtype=torch.float32,
            )
        )
        model.joint_hidden_bias.copy_(
            0.1
            * torch.randn(
                model.joint_hidden_bias.shape,
                generator=generator,
                dtype=torch.float32,
            )
        )
        model.scalar_energy_weight.copy_(
            0.5
            * torch.randn(
                model.scalar_energy_weight.shape,
                generator=generator,
                dtype=torch.float32,
            )
        )
    return model


def _parameter_rows(
    model: torch.nn.Module,
) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "byte_count": parameter.numel() * parameter.element_size(),
            "content_fingerprint": tensor_content_fingerprint(parameter),
        }
        for name, parameter in model.named_parameters()
    ]


def _generated_feature(
    index: int,
    *,
    channels: int = 2,
    height: int = 4,
    width: int = 4,
) -> Tensor:
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in (channels, height, width)
        )
    ):
        raise ValueError("generated feature dimensions/index are invalid")
    if (channels, height, width) == (2, 4, 4):
        grid = torch.arange(
            16,
            dtype=torch.float32,
        ).reshape(1, 1, 4, 4)
        first = ((grid + float(index + 1)) % 11.0) / 7.0 + 0.1
        second = (
            (3.0 * grid + float(2 * index + 1)) % 13.0
        ) / 9.0
        return torch.cat((first, second), dim=1).contiguous()
    grid = torch.arange(
        channels * height * width,
        dtype=torch.float32,
    ).reshape(1, channels, height, width)
    return (
        (
            (float(index + 3) * grid + float(2 * index + 1))
            % 31.0
        )
        / 17.0
        + 0.05
    ).contiguous()


def _mask(
    *coordinates: tuple[int, int],
    height: int = 8,
    width: int = 8,
) -> Tensor:
    result = torch.zeros((1, 1, height, width), dtype=torch.bool)
    for row, column in coordinates:
        result[0, 0, row, column] = True
    return result


def _stack_absolute_targets(
    values: tuple[CoverageStateAbsoluteTargets, ...],
) -> CoverageStateAbsoluteTargets:
    result = CoverageStateAbsoluteTargets(
        target_field=torch.cat(
            tuple(value.target_field for value in values),
            dim=0,
        ).contiguous(),
        integration_measure=torch.cat(
            tuple(value.integration_measure for value in values),
            dim=0,
        ).contiguous(),
        field_valid_mask=torch.cat(
            tuple(value.field_valid_mask for value in values),
            dim=0,
        ).contiguous(),
        loss_valid_mask=torch.cat(
            tuple(value.loss_valid_mask for value in values),
            dim=0,
        ).contiguous(),
        focus_support=torch.cat(
            tuple(value.focus_support for value in values),
            dim=0,
        ).contiguous(),
        focus_support_field=torch.cat(
            tuple(value.focus_support_field for value in values),
            dim=0,
        ).contiguous(),
    )
    result.validate()
    return result


def _stack_pair_targets(
    values: tuple[CoverageStatePairTargets, ...],
) -> CoverageStatePairTargets:
    result = CoverageStatePairTargets(
        target_field_plus=torch.cat(
            tuple(value.target_field_plus for value in values),
            dim=0,
        ).contiguous(),
        target_field_minus=torch.cat(
            tuple(value.target_field_minus for value in values),
            dim=0,
        ).contiguous(),
        focus_support=torch.cat(
            tuple(value.focus_support for value in values),
            dim=0,
        ).contiguous(),
        focus_support_field=torch.cat(
            tuple(value.focus_support_field for value in values),
            dim=0,
        ).contiguous(),
        integration_measure=torch.cat(
            tuple(value.integration_measure for value in values),
            dim=0,
        ).contiguous(),
        valid_mask=torch.cat(
            tuple(value.valid_mask for value in values),
            dim=0,
        ).contiguous(),
    )
    result.validate()
    return result


def _actual_input_fingerprint(
    feature: Tensor,
    occupancy: Tensor,
) -> str:
    return stable_fingerprint(
        {
            "feature": tensor_content_fingerprint(feature),
            "occupancy": tensor_content_fingerprint(occupancy),
        }
    )


def _generated_pmope_batch(
    *,
    feature_channels: int = 2,
    feature_stride: int = 2,
    feature_height: int = 4,
    feature_width: int = 4,
) -> tuple[
    CoverageStateFusedBatch,
    CoverageStateSobolevConfig,
]:
    """Build the complete public 4+4+2-pair generated warm-up batch."""

    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for value in (
            feature_channels,
            feature_stride,
            feature_height,
            feature_width,
        )
    ):
        raise ValueError("generated PMOPE dimensions must be positive")
    output_height = feature_height * feature_stride
    output_width = feature_width * feature_stride
    if min(output_height, output_width) < 4:
        raise ValueError("generated PMOPE output must be at least 4x4")

    def feature(index: int) -> Tensor:
        return _generated_feature(
            index,
            channels=feature_channels,
            height=feature_height,
            width=feature_width,
        )

    def mask(*coordinates: tuple[int, int]) -> Tensor:
        return _mask(
            *coordinates,
            height=output_height,
            width=output_width,
        )

    last_row = output_height - 1
    last_column = output_width - 1
    middle_row = output_height // 2
    middle_column = output_width // 2
    upper_middle_row = middle_row - 1
    upper_middle_column = middle_column - 1

    config = CoverageStateSobolevConfig(
        truncation_radius=feature_stride
    )
    valid = torch.ones(
        (1, 1, output_height, output_width),
        dtype=torch.bool,
    )
    miss_features = tuple(feature(index) for index in range(4))
    miss_occupancies = (
        mask((0, last_column)),
        mask((last_row, 0)),
        mask((0, 0)),
        mask((last_row, last_column)),
    )
    miss_targets = (
        mask((upper_middle_row, upper_middle_column)),
        mask(
            (upper_middle_row, middle_column),
            (middle_row, upper_middle_column),
        ),
        mask(
            (upper_middle_row, upper_middle_column),
            (middle_row, middle_column),
        ),
        mask((middle_row, middle_column)),
    )
    miss_prepared = tuple(
        prepare_coverage_state_focused_absolute_targets(
            target,
            valid,
            valid & ~occupancy,
            config=config,
        )
        for target, occupancy in zip(
            miss_targets,
            miss_occupancies,
            strict=True,
        )
    )

    no_features = tuple(
        feature(index + 4) for index in range(4)
    )
    no_occupancies = (
        mask(),
        mask((1, 1)),
        mask((1, max(1, last_column - 1))),
        mask((max(1, last_row - 1), 1)),
    )
    empty = mask()
    no_prepared = tuple(
        prepare_coverage_state_focused_absolute_targets(
            empty,
            valid,
            valid & ~occupancy,
            config=config,
        )
        for occupancy in no_occupancies
    )

    pair_features = (
        feature(8),
        feature(9),
    )
    clean_plus = mask(
        (upper_middle_row, upper_middle_column),
        (upper_middle_row, middle_column),
        (middle_row, upper_middle_column),
        (middle_row, middle_column),
    )
    clean_minus = mask()
    clean_target_plus = mask()
    clean_target_minus = clean_plus.clone()
    null_first_column = max(0, last_column - 1)
    null_plus = mask(
        (0, null_first_column),
        (0, last_column),
        (1, null_first_column),
        (1, last_column),
    )
    null_minus = mask()
    pair_occupancies_plus = (clean_plus, null_plus)
    pair_occupancies_minus = (clean_minus, null_minus)
    pair_targets_plus = (clean_target_plus, empty)
    pair_targets_minus = (clean_target_minus, empty)
    pair_prepared = tuple(
        prepare_coverage_state_pair_targets(
            plus,
            minus,
            target_plus,
            target_minus,
            valid,
            config=config,
        )
        for plus, minus, target_plus, target_minus in zip(
            pair_occupancies_plus,
            pair_occupancies_minus,
            pair_targets_plus,
            pair_targets_minus,
            strict=True,
        )
    )
    pair_absolute_plus = tuple(
        prepare_coverage_state_focused_absolute_targets(
            target,
            valid,
            valid & ~occupancy,
            config=config,
        )
        for target, occupancy in zip(
            pair_targets_plus,
            pair_occupancies_plus,
            strict=True,
        )
    )
    pair_absolute_minus = tuple(
        prepare_coverage_state_focused_absolute_targets(
            target,
            valid,
            valid & ~occupancy,
            config=config,
        )
        for target, occupancy in zip(
            pair_targets_minus,
            pair_occupancies_minus,
            strict=True,
        )
    )

    result = CoverageStateFusedBatch(
        factual_miss=CoverageStateNaturalTrainBatch(
            feature=torch.cat(miss_features, dim=0).contiguous(),
            occupancy=torch.cat(
                miss_occupancies,
                dim=0,
            ).contiguous(),
            targets=_stack_absolute_targets(miss_prepared),
            record_ids=tuple(
                f"v24-generated-miss-{index}" for index in range(4)
            ),
            sample_ids=tuple(
                f"v24-generated-miss-source-{index}"
                for index in range(4)
            ),
            actual_input_fingerprints=tuple(
                _actual_input_fingerprint(feature, occupancy)
                for feature, occupancy in zip(
                    miss_features,
                    miss_occupancies,
                    strict=True,
                )
            ),
            state_kind="factual_miss",
        ),
        factual_no_miss=CoverageStateNaturalTrainBatch(
            feature=torch.cat(no_features, dim=0).contiguous(),
            occupancy=torch.cat(
                no_occupancies,
                dim=0,
            ).contiguous(),
            targets=_stack_absolute_targets(no_prepared),
            record_ids=tuple(
                f"v24-generated-no-miss-{index}" for index in range(4)
            ),
            sample_ids=tuple(
                f"v24-generated-no-miss-source-{index}"
                for index in range(4)
            ),
            actual_input_fingerprints=tuple(
                _actual_input_fingerprint(feature, occupancy)
                for feature, occupancy in zip(
                    no_features,
                    no_occupancies,
                    strict=True,
                )
            ),
            state_kind="factual_no_miss",
        ),
        pairs=CoverageStatePairTrainBatch(
            feature=torch.cat(pair_features, dim=0).contiguous(),
            occupancy_plus=torch.cat(
                pair_occupancies_plus,
                dim=0,
            ).contiguous(),
            occupancy_minus=torch.cat(
                pair_occupancies_minus,
                dim=0,
            ).contiguous(),
            joint_targets=_stack_pair_targets(pair_prepared),
            absolute_targets_plus=(
                _stack_absolute_targets(pair_absolute_plus)
            ),
            absolute_targets_minus=(
                _stack_absolute_targets(pair_absolute_minus)
            ),
            pair_ids=(
                "v24-generated-clean-positive",
                "v24-generated-component-null",
            ),
            pair_kinds=("clean_positive", "component_null"),
            sample_ids=(
                "v24-generated-clean-source",
                "v24-generated-null-source",
            ),
            actual_input_plus_fingerprints=tuple(
                _actual_input_fingerprint(feature, occupancy)
                for feature, occupancy in zip(
                    pair_features,
                    pair_occupancies_plus,
                    strict=True,
                )
            ),
            actual_input_minus_fingerprints=tuple(
                _actual_input_fingerprint(feature, occupancy)
                for feature, occupancy in zip(
                    pair_features,
                    pair_occupancies_minus,
                    strict=True,
                )
            ),
        ),
    )
    result.validate()
    return result, config


def _fused_batch_fingerprint(batch: CoverageStateFusedBatch) -> str:
    feature, occupancy = batch.model_inputs()
    targets = {
        "miss_target_field": batch.factual_miss.targets.target_field,
        "miss_measure": batch.factual_miss.targets.integration_measure,
        "no_target_field": batch.factual_no_miss.targets.target_field,
        "no_measure": batch.factual_no_miss.targets.integration_measure,
        "pair_target_plus": batch.pairs.joint_targets.target_field_plus,
        "pair_target_minus": batch.pairs.joint_targets.target_field_minus,
        "pair_measure": batch.pairs.joint_targets.integration_measure,
    }
    return stable_fingerprint(
        {
            "schema": "cure-lite-v24-generated-pmope-fixture-v1",
            "selection_fingerprint": batch.selection_fingerprint,
            "feature_fingerprint": tensor_content_fingerprint(feature),
            "occupancy_fingerprint": tensor_content_fingerprint(occupancy),
            "target_tensors": {
                name: tensor_content_fingerprint(value)
                for name, value in sorted(targets.items())
            },
        }
    )


def _move_generated_value(value: object, device: torch.device) -> object:
    if isinstance(value, Tensor):
        return value.to(device=device)
    if is_dataclass(value):
        return type(value)(
            **{
                field.name: _move_generated_value(
                    getattr(value, field.name),
                    device,
                )
                for field in dataclass_fields(value)
            }
        )
    if isinstance(value, tuple):
        return tuple(_move_generated_value(item, device) for item in value)
    return value


def _batch_to_device(
    batch: CoverageStateFusedBatch,
    *,
    device: torch.device,
) -> CoverageStateFusedBatch:
    moved = _move_generated_value(batch, device)
    if type(moved) is not CoverageStateFusedBatch:
        raise TypeError("generated batch move returned the wrong type")
    moved.validate()
    return moved


def _identity_probe() -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(
            GCR_PACRE_DATASET_FREE_FORMAL_SEED
        )
        v24 = build_formal_gcr_pacre_training_model()
        torch.random.default_generator.manual_seed(
            GCR_PACRE_DATASET_FREE_FORMAL_SEED
        )
        v23 = CURELitePACREVerifierCorrectedLevelSet(
            CoverageStatePACREVerifierCorrectedConfig(
                feature_channels=GCR_PACRE_FORMAL_FEATURE_CHANNELS,
                feature_stride=GCR_PACRE_FORMAL_FEATURE_STRIDE,
                width=GCR_PACRE_FORMAL_WIDTH,
            )
        )
    v24_rows = _parameter_rows(v24)
    v23_rows = _parameter_rows(v23)
    config = v24.config
    identity = {
        "method_id": config.method_id,
        "field_policy": config.field_policy,
        "equation_policy": config.equation_policy,
        "interaction_policy": config.interaction_policy,
        "energy_policy": config.energy_policy,
        "numerical_policy": config.numerical_policy,
        "centering_policy": config.centering_policy,
    }
    expected_identity = {
        "method_id": GCR_PACRE_METHOD_ID,
        "field_policy": CSLF_GCR_PACRE_FIELD_POLICY,
        "equation_policy": CSLF_GCR_PACRE_EQUATION_POLICY,
        "interaction_policy": GCR_PACRE_INTERACTION_POLICY,
        "energy_policy": GCR_PACRE_ENERGY_POLICY,
        "numerical_policy": GCR_PACRE_NUMERICAL_POLICY,
        "centering_policy": GCR_PACRE_CENTERING_POLICY,
    }
    return _fingerprinted(
        {
            "candidate": GCR_PACRE_CANDIDATE,
            "model_fqcn": (
                f"{type(v24).__module__}.{type(v24).__qualname__}"
            ),
            "config_fqcn": (
                f"{type(config).__module__}.{type(config).__qualname__}"
            ),
            "model_contract": coverage_state_model_contract_payload(v24),
            "formal_config": {
                "feature_channels": config.feature_channels,
                "feature_stride": config.feature_stride,
                "width": config.width,
            },
            "canonical_identity": identity,
            "expected_identity": expected_identity,
            "canonical_identity_exact": identity == expected_identity,
            "parameter_count": sum(
                parameter.numel() for parameter in v24.parameters()
            ),
            "expected_parameter_count": (
                GCR_PACRE_FORMAL_PARAMETER_COUNT
            ),
            "parameter_tensor_count": len(v24_rows),
            "parameter_names": [row["name"] for row in v24_rows],
            "expected_parameter_names": list(
                GCR_PACRE_PARAMETER_NAMES
            ),
            "v24_initial_parameters": v24_rows,
            "v23_initial_parameters": v23_rows,
            "initial_parameter_names_shapes_bytes_match_v23": (
                v24_rows == v23_rows
            ),
            "named_children": [
                {
                    "name": name,
                    "fqcn": f"{type(module).__module__}."
                    f"{type(module).__qualname__}",
                }
                for name, module in v24.named_children()
            ],
            "additional_head_count": 0,
            "input_signature": ["feature", "occupancy"],
            "output_tensor_count": 1,
            "field_threshold": 0.0,
        },
        field="section_fingerprint",
    )


def _forward_with_validator_count(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    feature: Tensor,
    occupancy: Tensor,
):
    call_count = 0
    original = type(model)._validate_gcr_fields

    def counted(
        self: CURELiteGatedCommonResidualPACRELevelSet,
        fields,
        *,
        feature: Tensor,
        occupancy: Tensor,
    ) -> None:
        nonlocal call_count
        call_count += 1
        original(
            self,
            fields,
            feature=feature,
            occupancy=occupancy,
        )

    model._validate_gcr_fields = MethodType(  # type: ignore[method-assign]
        counted,
        model,
    )
    try:
        fields = model.forward_fields(feature, occupancy)
    finally:
        delattr(model, "_validate_gcr_fields")
    return fields, call_count


def _comparison_payload(comparison: object) -> dict[str, object]:
    payload = asdict(comparison)
    if not all(
        isfinite(float(payload[name]))
        for name in (
            "maximum_absolute_error",
            "absolute_tolerance",
        )
    ):
        raise FloatingPointError("oracle comparison is non-finite")
    return payload


def _flip_coordinate(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    occupancy: Tensor,
) -> tuple[Tensor, tuple[int, int, int, int]]:
    native_row, native_column, phase_index = 1, 2, 1
    phase_row, phase_column = divmod(
        phase_index,
        model.config.feature_stride,
    )
    output_row = (
        native_row * model.config.feature_stride + phase_row
    )
    output_column = (
        native_column * model.config.feature_stride + phase_column
    )
    flipped = occupancy.clone()
    flipped[0, 0, output_row, output_column] = ~flipped[
        0,
        0,
        output_row,
        output_column,
    ]
    return (
        flipped.contiguous(),
        (0, phase_index, native_row, native_column),
    )


def _gate_endpoint_probe() -> dict[str, object]:
    config = _toy_config(channels=1, stride=2, width=1)
    feature = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    occupancy = torch.zeros((1, 1, 4, 4), dtype=torch.bool)

    def one(readout: float) -> tuple[object, object]:
        model = build_gcr_pacre_training_model(config)
        center = model.config.coarse_radius
        with torch.no_grad():
            model.joint_state_weight.zero_()
            model.joint_hidden_bias.zero_()
            model.joint_state_weight[0, 0, center, center] = 20.0
            model.scalar_energy_weight.fill_(readout)
        fields = model.forward_fields(feature, occupancy)
        return fields, summarize_gcr_pacre_gate_saturation(fields)

    upper_fields, upper = one(20.0)
    lower_fields, lower = one(-20.0)
    return {
        "upper_statistics": asdict(upper),
        "lower_statistics": asdict(lower),
        "upper_residual_exact_zero": bool(
            torch.count_nonzero(
                upper_fields.residual_odd_interaction
            )
            == 0
        ),
        "lower_residual_exact_zero": bool(
            torch.count_nonzero(
                lower_fields.residual_odd_interaction
            )
            == 0
        ),
        "upper_common_nonzero": bool(
            torch.count_nonzero(upper_fields.common_even_energy) > 0
        ),
        "lower_common_nonzero": bool(
            torch.count_nonzero(lower_fields.common_even_energy) > 0
        ),
        "upper_field_exact_anchor": torch.equal(
            upper_fields.field,
            torch.full_like(
                upper_fields.field,
                CSLF_FIELD_AMPLITUDE,
            ),
        ),
        "lower_field_exact_anchor": torch.equal(
            lower_fields.field,
            torch.full_like(
                lower_fields.field,
                CSLF_FIELD_AMPLITUDE,
            ),
        ),
        "upper_completion_count": int(
            torch.count_nonzero(upper_fields.field < 0.0)
        ),
        "lower_completion_count": int(
            torch.count_nonzero(lower_fields.field < 0.0)
        ),
    }


def _forward_algebra_probe() -> dict[str, object]:
    model = _randomized_toy_model()
    feature, occupancy = _generated_inputs()
    model_before = _model_state_fingerprint(model)
    fields, validator_calls = _forward_with_validator_count(
        model,
        feature,
        occupancy,
    )
    validate_gcr_pacre_fields(
        model,
        fields,
        feature=feature,
        occupancy=occupancy,
    )
    oracle = model.forward_reference_fields_fp64(feature, occupancy)
    component_comparisons = {
        name: _comparison_payload(
            compare_gcr_pacre_fp32_to_fp64_oracle(
                getattr(fields, name),
                getattr(oracle, name),
            )
        )
        for name in (
            "residual_odd_interaction",
            "common_even_energy",
            "common_gate",
            "gated_interaction",
            "native_phase_field",
            "field",
        )
    }
    forced_before = _model_state_fingerprint(model)
    gradients_before = tuple(
        parameter.grad for parameter in model.parameters()
    )
    forced = model.forward_forced_unit_gate(feature, occupancy)
    forced_after = _model_state_fingerprint(model)
    gradients_after = tuple(
        parameter.grad for parameter in model.parameters()
    )
    completion = model.predict_completion(feature, occupancy)
    union = model.predict_union(feature, occupancy)

    flipped_occupancy, coordinate = _flip_coordinate(model, occupancy)
    flipped_fields = model.forward_fields(feature, flipped_occupancy)
    first_oracle = oracle
    second_oracle = model.forward_reference_fields_fp64(
        feature,
        flipped_occupancy,
    )
    reference_parity = {
        "residual_odd_antisymmetric": torch.equal(
            first_oracle.residual_odd_interaction[coordinate],
            -second_oracle.residual_odd_interaction[coordinate],
        ),
        "common_even_symmetric": torch.equal(
            first_oracle.common_even_energy[coordinate],
            second_oracle.common_even_energy[coordinate],
        ),
        "gate_symmetric": torch.equal(
            first_oracle.common_gate[coordinate],
            second_oracle.common_gate[coordinate],
        ),
        "gated_interaction_antisymmetric": torch.equal(
            first_oracle.gated_interaction[coordinate],
            -second_oracle.gated_interaction[coordinate],
        ),
    }
    fast_flip_comparisons = {
        "residual_odd": _comparison_payload(
            compare_gcr_pacre_fp32_to_fp64_oracle(
                fields.residual_odd_interaction[coordinate][None],
                first_oracle.residual_odd_interaction[
                    coordinate
                ][None],
            )
        ),
        "common_even": _comparison_payload(
            compare_gcr_pacre_fp32_to_fp64_oracle(
                flipped_fields.common_even_energy[coordinate][None],
                second_oracle.common_even_energy[coordinate][None],
            )
        ),
        "gate": _comparison_payload(
            compare_gcr_pacre_fp32_to_fp64_oracle(
                flipped_fields.common_gate[coordinate][None],
                second_oracle.common_gate[coordinate][None],
            )
        ),
        "gated_interaction": _comparison_payload(
            compare_gcr_pacre_fp32_to_fp64_oracle(
                flipped_fields.gated_interaction[coordinate][None],
                second_oracle.gated_interaction[coordinate][None],
            )
        ),
    }
    zero_feature = torch.zeros_like(feature)
    zero_fields = model.forward_fields(zero_feature, occupancy)
    statistics = summarize_gcr_pacre_gate_saturation(fields)
    endpoint = _gate_endpoint_probe()
    legacy_ungated = model.pixel_shuffle(
        model.config.field_amplitude
        + fields.residual_odd_interaction
    ).contiguous()
    return _fingerprinted(
        {
            "fixture": {
                "feature_fingerprint": tensor_content_fingerprint(feature),
                "occupancy_fingerprint": tensor_content_fingerprint(
                    occupancy
                ),
                "model_state_fingerprint": model_before,
                "flip_coordinate": list(coordinate),
                "flipped_occupancy_fingerprint": (
                    tensor_content_fingerprint(flipped_occupancy)
                ),
            },
            "lightweight_validator_forward_count": 1,
            "lightweight_validator_call_count": validator_calls,
            "full_replay_passed": True,
            "fp64_envelope": {
                "absolute_tolerance": (
                    GCR_PACRE_FP64_ORACLE_ABS_TOL
                ),
                "maximum_allowed_ulp": (
                    GCR_PACRE_FP64_ORACLE_MAX_ULP
                ),
                # The frozen production envelope is an output-field
                # contract.  Intermediate comparisons remain diagnostics:
                # near-zero latent values may exceed an ULP budget without
                # violating the independently recomputed output contract.
                "required_components": ["field"],
                "components": component_comparisons,
            },
            "non_unit_gate_count": int(
                torch.count_nonzero(fields.common_gate != 1.0)
            ),
            "fast_differs_from_legacy_ungated": not torch.equal(
                fields.field,
                legacy_ungated,
            ),
            "gate_statistics": asdict(statistics),
            "endpoint_witnesses": endpoint,
            "reference_parity": reference_parity,
            "fast_flip_oracle_comparisons": (
                fast_flip_comparisons
            ),
            "zero_common_maps_to_unit_gate": float(
                2.0
                * torch.sigmoid(
                    torch.tensor(0.0, dtype=torch.float64)
                )
            )
            == 1.0,
            "zero_residual_maps_to_zero_interaction": float(
                torch.tensor(0.0, dtype=torch.float64)
                * torch.tensor(1.75, dtype=torch.float64)
            )
            == 0.0,
            "zero_feature": {
                "residual_exact_zero": bool(
                    torch.count_nonzero(
                        zero_fields.residual_odd_interaction
                    )
                    == 0
                ),
                "common_exact_zero": bool(
                    torch.count_nonzero(
                        zero_fields.common_even_energy
                    )
                    == 0
                ),
                "gate_exact_one": torch.equal(
                    zero_fields.common_gate,
                    torch.ones_like(zero_fields.common_gate),
                ),
                "field_exact_anchor": torch.equal(
                    zero_fields.field,
                    torch.full_like(
                        zero_fields.field,
                        CSLF_FIELD_AMPLITUDE,
                    ),
                ),
            },
            "forced_unit_gate": {
                "state_unchanged": forced_before == forced_after,
                "gradients_unchanged": (
                    gradients_before == gradients_after
                ),
                "matches_ungated_equation": torch.equal(
                    forced,
                    legacy_ungated,
                ),
                "differs_from_gcr": not torch.equal(
                    forced,
                    fields.field,
                ),
            },
            "hard_union": {
                "completion_exact": torch.equal(
                    completion,
                    (fields.field < 0.0) & ~occupancy,
                ),
                "union_exact": torch.equal(
                    union,
                    occupancy | completion,
                ),
                "retains_occupancy": torch.equal(
                    union & occupancy,
                    occupancy,
                ),
                "threshold": 0.0,
            },
            "model_state_unchanged": (
                model_before == _model_state_fingerprint(model)
            ),
        },
        field="section_fingerprint",
    )


def _selectivity_probe() -> dict[str, object]:
    model = _randomized_toy_model()
    feature, occupancy = _generated_inputs()
    fields = model.forward_fields(feature, occupancy)
    residual = fields.residual_odd_interaction
    common = fields.common_even_energy
    margin = GCR_PACRE_DATASET_FREE_SELECTIVITY_MARGIN
    target_candidates = torch.nonzero(
        (residual < -margin) & (common > margin),
        as_tuple=False,
    )
    background_candidates = torch.nonzero(
        (residual < -margin) & (common < -margin),
        as_tuple=False,
    )

    def row(candidates: Tensor, *, role: str) -> dict[str, object]:
        if candidates.numel() == 0:
            return {
                "role": role,
                "witness_present": False,
            }
        coordinate = tuple(int(value) for value in candidates[0].tolist())
        d_value = float(residual[coordinate].detach())
        e_value = float(common[coordinate].detach())
        gate_value = float(fields.common_gate[coordinate].detach())
        gcr_field = float(fields.native_phase_field[coordinate].detach())
        pacre_field = float(CSLF_FIELD_AMPLITUDE + d_value)
        return {
            "role": role,
            "witness_present": True,
            "coordinate": list(coordinate),
            "residual_odd": d_value,
            "common_even": e_value,
            "gate": gate_value,
            "gcr_native_field": gcr_field,
            "pacre_native_field": pacre_field,
            "gcr_minus_pacre": gcr_field - pacre_field,
        }

    target = row(target_candidates, role="target_like")
    background = row(
        background_candidates,
        role="background_like",
    )
    common_only = _gate_endpoint_probe()
    return _fingerprinted(
        {
            "fixture": {
                "feature_fingerprint": tensor_content_fingerprint(feature),
                "occupancy_fingerprint": tensor_content_fingerprint(
                    occupancy
                ),
                "model_state_fingerprint": (
                    _model_state_fingerprint(model)
                ),
                "selection_rule": (
                    "lexicographically_first_cell_satisfying_frozen_sign_margin"
                ),
                "sign_margin": margin,
            },
            "target_like": target,
            "background_like": background,
            "common_only": {
                "residual_exact_zero": (
                    common_only["upper_residual_exact_zero"]
                ),
                "common_nonzero": common_only["upper_common_nonzero"],
                "field_exact_anchor": (
                    common_only["upper_field_exact_anchor"]
                ),
                "completion_count": (
                    common_only["upper_completion_count"]
                ),
            },
        },
        field="section_fingerprint",
    )


def _gradient_rows(
    model: torch.nn.Module,
    values: tuple[Tensor, ...],
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for (name, _), value in zip(
        model.named_parameters(),
        values,
        strict=True,
    ):
        rows[name] = {
            "finite": bool(torch.isfinite(value).all()),
            "nonzero_count": int(torch.count_nonzero(value)),
            "l2_norm": float(value.detach().square().sum().sqrt()),
            "fingerprint": tensor_content_fingerprint(value),
        }
    return rows


def _gradient_probe() -> dict[str, object]:
    seed = GCR_PACRE_DATASET_FREE_WARMUP_SEED
    v24_config = _toy_config()
    v23_config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        v24_step0 = build_gcr_pacre_training_model(v24_config)
        torch.random.default_generator.manual_seed(seed)
        v23_step0 = CURELitePACREVerifierCorrectedLevelSet(v23_config)
    feature, occupancy = _generated_inputs(seed=seed + 1)
    fields_v24 = v24_step0.forward_fields(feature, occupancy)
    fields_v23 = v23_step0.forward_fields(feature, occupancy)
    weights = torch.linspace(
        0.3,
        1.7,
        fields_v24.field.numel(),
        dtype=torch.float32,
    ).reshape_as(fields_v24.field)
    gradients_v24 = torch.autograd.grad(
        (fields_v24.field * weights).sum(),
        tuple(v24_step0.parameters()),
        allow_unused=False,
    )
    gradients_v23 = torch.autograd.grad(
        (fields_v23.field * weights).sum(),
        tuple(v23_step0.parameters()),
        allow_unused=False,
    )
    gradient_errors = [
        float((first - second).detach().abs().amax())
        for first, second in zip(
            gradients_v24,
            gradients_v23,
            strict=True,
        )
    ]

    batch, sobolev_config = _generated_pmope_batch()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        warmed = build_gcr_pacre_training_model(v24_config)
    optimizer = torch.optim.Adam(
        warmed.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    optimizer_fingerprint = coverage_state_optimizer_config_fingerprint(
        warmed,
        optimizer,
    )
    before = coverage_state_model_fingerprint(warmed)
    logs = coverage_state_fused_train_step(
        warmed,
        optimizer,
        batch,
        config=sobolev_config,
        pair_objective=CoverageStatePairObjective.PMOPE_JOINT,
        audit=True,
        track_nonzero_gradients=True,
    )
    after = coverage_state_model_fingerprint(warmed)
    pmope_gradients = tuple(
        parameter.grad.detach().clone()
        if parameter.grad is not None
        else torch.full_like(parameter, float("nan"))
        for parameter in warmed.parameters()
    )
    pmope_rows = _gradient_rows(warmed, pmope_gradients)
    optimizer.zero_grad(set_to_none=True)

    post = warmed.forward_fields(feature, occupancy)
    path_weights = torch.linspace(
        0.2,
        1.4,
        post.common_gate.numel(),
        dtype=torch.float32,
    ).reshape_as(post.common_gate)
    residual_probe = (
        post.residual_odd_interaction * path_weights
    ).sum()
    gate_probe = (
        post.common_gate
        * post.residual_odd_interaction.detach()
        * path_weights
    ).sum()
    residual_gradients = torch.autograd.grad(
        residual_probe,
        tuple(warmed.parameters()),
        retain_graph=True,
        allow_unused=False,
    )
    gate_gradients = torch.autograd.grad(
        gate_probe,
        tuple(warmed.parameters()),
        allow_unused=False,
    )
    residual_rows = _gradient_rows(warmed, residual_gradients)
    gate_rows = _gradient_rows(warmed, gate_gradients)

    return _fingerprinted(
        {
            "step0": {
                "fixture": {
                    "feature_fingerprint": (
                        tensor_content_fingerprint(feature)
                    ),
                    "occupancy_fingerprint": (
                        tensor_content_fingerprint(occupancy)
                    ),
                },
                "v23_v24_initial_parameter_bytes_equal": (
                    _parameter_rows(v23_step0)
                    == _parameter_rows(v24_step0)
                ),
                "output_raw_equal": torch.equal(
                    fields_v23.field,
                    fields_v24.field,
                ),
                "residual_exact_zero": bool(
                    torch.count_nonzero(
                        fields_v24.residual_odd_interaction
                    )
                    == 0
                ),
                "common_exact_zero": bool(
                    torch.count_nonzero(
                        fields_v24.common_even_energy
                    )
                    == 0
                ),
                "gate_exact_one": torch.equal(
                    fields_v24.common_gate,
                    torch.ones_like(fields_v24.common_gate),
                ),
                "gradient_maximum_absolute_errors": gradient_errors,
                "gradient_absolute_tolerance": (
                    GCR_PACRE_DATASET_FREE_STEP0_GRADIENT_ATOL
                ),
                "all_gradients_finite": all(
                    bool(torch.isfinite(value).all())
                    for value in (*gradients_v23, *gradients_v24)
                ),
            },
            "warmup": {
                "fixture_fingerprint": _fused_batch_fingerprint(batch),
                "selection_fingerprint": batch.selection_fingerprint,
                "objective": logs["pair_objective"],
                "objective_policy": logs["pair_objective_policy"],
                "expected_objective": (
                    CoverageStatePairObjective.PMOPE_JOINT.value
                ),
                "expected_objective_policy": CSLF_PMOPE_POLICY,
                "function_fqcn": (
                    "cure_lite.train.coverage_state_fused_step."
                    "coverage_state_fused_train_step"
                ),
                "optimizer_fqcn": (
                    f"{type(optimizer).__module__}."
                    f"{type(optimizer).__qualname__}"
                ),
                "optimizer_config_fingerprint": optimizer_fingerprint,
                "optimizer_hyperparameters": {
                    "learning_rate_hex": float(
                        optimizer.param_groups[0]["lr"]
                    ).hex(),
                    "beta1_hex": float(
                        optimizer.param_groups[0]["betas"][0]
                    ).hex(),
                    "beta2_hex": float(
                        optimizer.param_groups[0]["betas"][1]
                    ).hex(),
                    "epsilon_hex": float(
                        optimizer.param_groups[0]["eps"]
                    ).hex(),
                    "weight_decay_hex": float(
                        optimizer.param_groups[0]["weight_decay"]
                    ).hex(),
                },
                "model_forward_calls": logs["model_forward_calls"],
                "backward_calls": logs["backward_calls"],
                "optimizer_steps": logs["optimizer_steps"],
                "logical_states": logs["logical_states"],
                "initial_model_fingerprint": before,
                "final_model_fingerprint": after,
                "parameter_state_changed": before != after,
                "all_logged_losses_finite": all(
                    isfinite(float(logs[name]))
                    for name in (
                        "factual_miss/loss",
                        "factual_no_miss/loss",
                        "pair/loss",
                        "total",
                        "gradient_l2_norm",
                    )
                ),
                "pmope_parameter_gradients": pmope_rows,
            },
            "post_warmup": {
                "residual_nonzero_count": int(
                    torch.count_nonzero(
                        post.residual_odd_interaction
                    )
                ),
                "common_nonzero_count": int(
                    torch.count_nonzero(post.common_even_energy)
                ),
                "residual_path_parameter_gradients": residual_rows,
                "gate_path_parameter_gradients": gate_rows,
            },
        },
        field="section_fingerprint",
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _runtime_device_payload(device: torch.device) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "cure-lite-v24-runtime-device-v1",
        "selected_device": str(device),
        "device_type": device.type,
        "device_index": device.index,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        payload["accelerator"] = {
            "name": properties.name,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        }
    else:
        payload["accelerator"] = None
    return payload


def _latency_summary(values: list[int]) -> dict[str, object]:
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for value in values
    ):
        raise ValueError("latency samples must be positive integer nanoseconds")
    ordered = sorted(values)
    p95_index = max(0, ceil(0.95 * len(ordered)) - 1)
    return {
        "sample_count": len(values),
        "samples_ns": values,
        "median_ns": float(median(values)),
        "p95_ns": float(ordered[p95_index]),
    }


def _checkpoint_bytes(model: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue())


def _profile_forward_flops(
    model: torch.nn.Module,
    feature: Tensor,
    occupancy: Tensor,
    *,
    device: torch.device,
) -> tuple[int, bool]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        with_flops=True,
        record_shapes=False,
        profile_memory=False,
    ) as profile:
        with torch.no_grad():
            model(feature, occupancy)
        _synchronize(device)
    values = [
        int(event.flops)
        for event in profile.key_averages()
        if event.flops is not None
    ]
    total = sum(values)
    return total, total > 0


def _efficiency_model(
    arm: str,
    *,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if arm == "PACRE_VC_v23":
            model = CURELitePACREVerifierCorrectedLevelSet(
                CoverageStatePACREVerifierCorrectedConfig(
                    feature_channels=(
                        GCR_PACRE_FORMAL_FEATURE_CHANNELS
                    ),
                    feature_stride=GCR_PACRE_FORMAL_FEATURE_STRIDE,
                    width=GCR_PACRE_FORMAL_WIDTH,
                )
            )
        elif arm == "GCR_PACRE_v24":
            model = build_formal_gcr_pacre_training_model()
        else:
            raise ValueError("unknown efficiency arm")
    return model.to(device=device, dtype=torch.float32)


def _efficiency_arm(
    arm: str,
    *,
    seed: int,
    device: torch.device,
    batch: CoverageStateFusedBatch,
    sobolev_config: CoverageStateSobolevConfig,
) -> dict[str, object]:
    model = _efficiency_model(arm, seed=seed, device=device)
    feature, occupancy = batch.model_inputs()
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    checkpoint_bytes = _checkpoint_bytes(model)
    parameter_tensors = _parameter_rows(model)
    initial_parameter_fingerprint = stable_fingerprint(parameter_tensors)
    model_config = {
        "feature_channels": model.config.feature_channels,
        "feature_stride": model.config.feature_stride,
        "width": model.config.width,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        forward_flops, flops_supported = _profile_forward_flops(
            model,
            feature,
            occupancy,
            device=device,
        )
        for _ in range(GCR_PACRE_EFFICIENCY_FORWARD_WARMUPS):
            with torch.no_grad():
                field = model(feature, occupancy)
            _synchronize(device)
        forward_samples: list[int] = []
        for _ in range(GCR_PACRE_EFFICIENCY_FORWARD_REPEATS):
            _synchronize(device)
            start = perf_counter_ns()
            with torch.no_grad():
                field = model(feature, occupancy)
            _synchronize(device)
            forward_samples.append(perf_counter_ns() - start)
        forward_finite = bool(torch.isfinite(field).all())
        field_bytes = field.numel() * field.element_size()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
        )
        training_logs: list[dict[str, float | int | str]] = []
        for _ in range(GCR_PACRE_EFFICIENCY_TRAIN_WARMUPS):
            training_logs.append(
                coverage_state_fused_train_step(
                    model,
                    optimizer,
                    batch,
                    config=sobolev_config,
                    pair_objective=(
                        CoverageStatePairObjective.PMOPE_JOINT
                    ),
                    audit=True,
                    track_nonzero_gradients=False,
                )
            )
            _synchronize(device)
        train_samples: list[int] = []
        for _ in range(GCR_PACRE_EFFICIENCY_TRAIN_REPEATS):
            _synchronize(device)
            start = perf_counter_ns()
            training_logs.append(
                coverage_state_fused_train_step(
                    model,
                    optimizer,
                    batch,
                    config=sobolev_config,
                    pair_objective=(
                        CoverageStatePairObjective.PMOPE_JOINT
                    ),
                    audit=True,
                    track_nonzero_gradients=False,
                )
            )
            _synchronize(device)
            train_samples.append(perf_counter_ns() - start)
        logs_finite = all(
            isfinite(float(row[name]))
            for row in training_logs
            for name in (
                "factual_miss/loss",
                "factual_no_miss/loss",
                "pair/loss",
                "total",
                "gradient_l2_norm",
            )
        )
        memory = (
            {
                "supported": True,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
            if device.type == "cuda"
            else {
                "supported": False,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            }
        )
        return {
            "arm": arm,
            "device": str(device),
            "dtype": "torch.float32",
            "parameter_tensor_count": len(tuple(model.parameters())),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "model_config": model_config,
            "parameter_tensors": parameter_tensors,
            "parameter_bytes": parameter_bytes,
            "checkpoint_counter": "torch.save_state_dict_zip_bytes_v1",
            "checkpoint_bytes": checkpoint_bytes,
            "initial_parameter_fingerprint": (
                initial_parameter_fingerprint
            ),
            "forward_flop_counter": (
                "torch.profiler.key_averages_sum_flops_v1"
            ),
            "forward_flops": forward_flops,
            "forward_flop_counter_supported": flops_supported,
            "forward_latency": _latency_summary(forward_samples),
            "train_step_latency": _latency_summary(train_samples),
            "forward_warmups": (
                GCR_PACRE_EFFICIENCY_FORWARD_WARMUPS
            ),
            "forward_repeats": (
                GCR_PACRE_EFFICIENCY_FORWARD_REPEATS
            ),
            "train_step_warmups": (
                GCR_PACRE_EFFICIENCY_TRAIN_WARMUPS
            ),
            "train_step_repeats": (
                GCR_PACRE_EFFICIENCY_TRAIN_REPEATS
            ),
            "train_optimizer_steps": (
                GCR_PACRE_EFFICIENCY_TRAIN_WARMUPS
                + GCR_PACRE_EFFICIENCY_TRAIN_REPEATS
            ),
            "field_tensor_bytes": field_bytes,
            "output_shape": list(field.shape),
            "memory": memory,
            "oom": False,
            "nonfinite": not (forward_finite and logs_finite),
        }
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        message = str(error)
        is_oom = (
            isinstance(error, torch.cuda.OutOfMemoryError)
            or "out of memory" in message.lower()
        )
        if not is_oom:
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "arm": arm,
            "device": str(device),
            "dtype": "torch.float32",
            "oom": True,
            "nonfinite": False,
            "error_type": type(error).__name__,
        }


def _efficiency_probe(device: torch.device) -> dict[str, object]:
    cpu_batch, sobolev_config = _generated_pmope_batch(
        feature_channels=GCR_PACRE_FORMAL_FEATURE_CHANNELS,
        feature_stride=GCR_PACRE_FORMAL_FEATURE_STRIDE,
        feature_height=GCR_PACRE_EFFICIENCY_FEATURE_HEIGHT,
        feature_width=GCR_PACRE_EFFICIENCY_FEATURE_WIDTH,
    )
    batch = _batch_to_device(cpu_batch, device=device)
    seed = GCR_PACRE_DATASET_FREE_WARMUP_SEED
    arms = {
        arm: _efficiency_arm(
            arm,
            seed=seed,
            device=device,
            batch=batch,
            sobolev_config=sobolev_config,
        )
        for arm in ("PACRE_VC_v23", "GCR_PACRE_v24")
    }
    additional_ops = {
        "PACRE_VC_v23": [
            "actual_and_flipped_residual_silu_difference",
            "shared_scalar_readout",
            "odd_residual_average",
            "field_add",
            "pixel_shuffle",
        ],
        "GCR_PACRE_v24": [
            "actual_and_flipped_residual_silu_difference",
            "actual_and_flipped_common_hidden_difference",
            "shared_residual_and_common_scalar_readout",
            "odd_residual_average",
            "even_common_average",
            "two_times_sigmoid",
            "gate_times_residual",
            "endpoint_saturation_masks",
            "field_add",
            "pixel_shuffle",
        ],
        "v24_additions_over_v23": [
            "two_occupancy_only_silu",
            "two_common_hidden_difference",
            "two_common_scalar_readout",
            "one_even_average",
            "one_sigmoid_and_scale",
            "one_gate_multiply",
            "two_endpoint_comparisons",
        ],
    }
    common_conditions = {
        "device": str(device),
        "dtype": "torch.float32",
        "runtime_device": _runtime_device_payload(device),
        "formal_model_config": {
            "feature_channels": GCR_PACRE_FORMAL_FEATURE_CHANNELS,
            "feature_stride": GCR_PACRE_FORMAL_FEATURE_STRIDE,
            "width": GCR_PACRE_FORMAL_WIDTH,
            "parameter_count": GCR_PACRE_FORMAL_PARAMETER_COUNT,
        },
        "batch_fixture_fingerprint": _fused_batch_fingerprint(
            cpu_batch
        ),
        "input_shape": list(batch.model_inputs()[0].shape),
        "occupancy_shape": list(batch.model_inputs()[1].shape),
        "spatial_fixture_role": (
            "minimum_legal_generated_shape_not_deployment_workload"
        ),
        "forward_warmups": GCR_PACRE_EFFICIENCY_FORWARD_WARMUPS,
        "forward_repeats": GCR_PACRE_EFFICIENCY_FORWARD_REPEATS,
        "train_step_warmups": GCR_PACRE_EFFICIENCY_TRAIN_WARMUPS,
        "train_step_repeats": GCR_PACRE_EFFICIENCY_TRAIN_REPEATS,
        "pair_objective": CoverageStatePairObjective.PMOPE_JOINT.value,
        "pair_objective_policy": CSLF_PMOPE_POLICY,
        "threshold_or_ratio_gate": None,
    }
    return _fingerprinted(
        {
            "common_conditions": common_conditions,
            "arms": arms,
            "additional_op_inventory": additional_ops,
            "interpretation": (
                "measurement_only_no_post_hoc_lite_overhead_threshold"
            ),
        },
        field="section_fingerprint",
    )


def _all_gradient_rows_pass(
    value: object,
    *,
    require_each_nonzero: bool,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(
        GCR_PACRE_PARAMETER_NAMES
    ):
        return False
    for raw in value.values():
        if not isinstance(raw, Mapping):
            return False
        norm = raw.get("l2_norm")
        if (
            raw.get("finite") is not True
            or isinstance(norm, bool)
            or not isinstance(norm, (int, float))
            or not isfinite(float(norm))
            or (
                require_each_nonzero
                and (
                    not isinstance(raw.get("nonzero_count"), int)
                    or isinstance(raw.get("nonzero_count"), bool)
                    or int(raw["nonzero_count"]) < 1
                    or float(norm) <= 0.0
                )
            )
            or not _is_sha256(raw.get("fingerprint"))
        ):
            return False
    return True


def _efficiency_complete(
    efficiency: Mapping[str, object],
) -> bool:
    conditions = efficiency.get("common_conditions")
    arms = efficiency.get("arms")
    inventory = efficiency.get("additional_op_inventory")
    runtime_device = (
        conditions.get("runtime_device")
        if isinstance(conditions, Mapping)
        else None
    )
    expected_formal_config = {
        "feature_channels": GCR_PACRE_FORMAL_FEATURE_CHANNELS,
        "feature_stride": GCR_PACRE_FORMAL_FEATURE_STRIDE,
        "width": GCR_PACRE_FORMAL_WIDTH,
        "parameter_count": GCR_PACRE_FORMAL_PARAMETER_COUNT,
    }
    if (
        not isinstance(conditions, Mapping)
        or not isinstance(arms, Mapping)
        or set(arms) != {"PACRE_VC_v23", "GCR_PACRE_v24"}
        or not isinstance(inventory, Mapping)
        or conditions.get("formal_model_config")
        != expected_formal_config
        or conditions.get("input_shape")
        != [
            12,
            GCR_PACRE_FORMAL_FEATURE_CHANNELS,
            GCR_PACRE_EFFICIENCY_FEATURE_HEIGHT,
            GCR_PACRE_EFFICIENCY_FEATURE_WIDTH,
        ]
        or conditions.get("occupancy_shape")
        != [
            12,
            1,
            (
                GCR_PACRE_EFFICIENCY_FEATURE_HEIGHT
                * GCR_PACRE_FORMAL_FEATURE_STRIDE
            ),
            (
                GCR_PACRE_EFFICIENCY_FEATURE_WIDTH
                * GCR_PACRE_FORMAL_FEATURE_STRIDE
            ),
        ]
        or conditions.get("spatial_fixture_role")
        != "minimum_legal_generated_shape_not_deployment_workload"
        or not _is_sha256(
            conditions.get("batch_fixture_fingerprint")
        )
        or conditions.get("pair_objective")
        != CoverageStatePairObjective.PMOPE_JOINT.value
        or conditions.get("pair_objective_policy") != CSLF_PMOPE_POLICY
        or conditions.get("threshold_or_ratio_gate") is not None
        or efficiency.get("interpretation")
        != "measurement_only_no_post_hoc_lite_overhead_threshold"
    ):
        return False
    expected_forward_samples = conditions.get("forward_repeats")
    expected_train_samples = conditions.get("train_step_repeats")
    if (
        isinstance(expected_forward_samples, bool)
        or not isinstance(expected_forward_samples, int)
        or isinstance(expected_train_samples, bool)
        or not isinstance(expected_train_samples, int)
    ):
        return False
    common_device = conditions.get("device")
    common_dtype = conditions.get("dtype")
    if (
        not isinstance(runtime_device, Mapping)
        or runtime_device.get("schema")
        != "cure-lite-v24-runtime-device-v1"
        or runtime_device.get("selected_device") != common_device
        or runtime_device.get("device_type")
        != ("cuda" if common_device == "cuda:0" else "cpu")
        or runtime_device.get("device_index")
        != (0 if common_device == "cuda:0" else None)
        or not isinstance(runtime_device.get("torch_version"), str)
        or not runtime_device["torch_version"]
    ):
        return False
    accelerator = runtime_device.get("accelerator")
    if common_device == "cuda:0":
        if (
            not isinstance(runtime_device.get("cuda_runtime_version"), str)
            or not runtime_device["cuda_runtime_version"]
            or not isinstance(accelerator, Mapping)
            or not isinstance(accelerator.get("name"), str)
            or not accelerator["name"]
            or not isinstance(
                accelerator.get("compute_capability"),
                list,
            )
            or len(accelerator["compute_capability"]) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in accelerator["compute_capability"]
            )
            or not isinstance(accelerator.get("total_memory_bytes"), int)
            or accelerator["total_memory_bytes"] < 1
            or not isinstance(
                accelerator.get("multiprocessor_count"),
                int,
            )
            or accelerator["multiprocessor_count"] < 1
        ):
            return False
    elif common_device != "cpu" or accelerator is not None:
        return False
    output_shapes: set[tuple[object, ...]] = set()
    initial_fingerprints: set[str] = set()
    for arm_name, raw in arms.items():
        parameter_tensors = (
            raw.get("parameter_tensors")
            if isinstance(raw, Mapping)
            else None
        )
        if (
            not isinstance(raw, Mapping)
            or raw.get("arm") != arm_name
            or raw.get("device") != common_device
            or raw.get("dtype") != common_dtype
            or raw.get("model_config")
            != {
                "feature_channels": (
                    GCR_PACRE_FORMAL_FEATURE_CHANNELS
                ),
                "feature_stride": GCR_PACRE_FORMAL_FEATURE_STRIDE,
                "width": GCR_PACRE_FORMAL_WIDTH,
            }
            or raw.get("oom") is not False
            or raw.get("nonfinite") is not False
            or raw.get("forward_flop_counter")
            != "torch.profiler.key_averages_sum_flops_v1"
            or raw.get("forward_flop_counter_supported") is not True
            or not isinstance(raw.get("forward_flops"), int)
            or isinstance(raw.get("forward_flops"), bool)
            or int(raw["forward_flops"]) < 1
            or not isinstance(raw.get("parameter_tensor_count"), int)
            or int(raw["parameter_tensor_count"]) != 3
            or not isinstance(raw.get("parameter_count"), int)
            or isinstance(raw.get("parameter_count"), bool)
            or int(raw["parameter_count"])
            != GCR_PACRE_FORMAL_PARAMETER_COUNT
            or not isinstance(parameter_tensors, list)
            or len(parameter_tensors) != 3
            or [
                row.get("name")
                for row in parameter_tensors
                if isinstance(row, Mapping)
            ]
            != list(GCR_PACRE_PARAMETER_NAMES)
            or any(
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "name",
                    "shape",
                    "dtype",
                    "numel",
                    "byte_count",
                    "content_fingerprint",
                }
                or row.get("dtype") != "torch.float32"
                or not isinstance(row.get("shape"), list)
                or not isinstance(row.get("numel"), int)
                or isinstance(row.get("numel"), bool)
                or row["numel"] < 1
                or row.get("byte_count") != row["numel"] * 4
                or not _is_sha256(row.get("content_fingerprint"))
                for row in parameter_tensors
            )
            or sum(
                int(row["numel"])
                for row in parameter_tensors
                if isinstance(row, Mapping)
            )
            != GCR_PACRE_FORMAL_PARAMETER_COUNT
            or not isinstance(raw.get("parameter_bytes"), int)
            or isinstance(raw.get("parameter_bytes"), bool)
            or int(raw["parameter_bytes"])
            != GCR_PACRE_FORMAL_PARAMETER_COUNT * 4
            or raw.get("checkpoint_counter")
            != "torch.save_state_dict_zip_bytes_v1"
            or not isinstance(raw.get("checkpoint_bytes"), int)
            or int(raw["checkpoint_bytes"]) < 1
            or not isinstance(raw.get("field_tensor_bytes"), int)
            or int(raw["field_tensor_bytes"]) < 1
            or not _is_sha256(
                raw.get("initial_parameter_fingerprint")
            )
            or raw.get("initial_parameter_fingerprint")
            != stable_fingerprint(parameter_tensors)
        ):
                return False
        initial_fingerprints.add(
            str(raw["initial_parameter_fingerprint"])
        )
        for name, expected_count in (
            ("forward_latency", expected_forward_samples),
            ("train_step_latency", expected_train_samples),
        ):
            latency = raw.get(name)
            if (
                not isinstance(latency, Mapping)
                or latency.get("sample_count") != expected_count
                or not isinstance(latency.get("samples_ns"), list)
                or len(latency["samples_ns"]) != expected_count
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    for value in latency["samples_ns"]
                )
                or dict(latency)
                != _latency_summary(latency["samples_ns"])
                or isinstance(latency.get("median_ns"), bool)
                or not isinstance(
                    latency.get("median_ns"),
                    (int, float),
                )
                or not isfinite(float(latency["median_ns"]))
                or float(latency["median_ns"]) <= 0.0
                or isinstance(latency.get("p95_ns"), bool)
                or not isinstance(
                    latency.get("p95_ns"),
                    (int, float),
                )
                or not isfinite(float(latency["p95_ns"]))
                or float(latency["p95_ns"]) <= 0.0
            ):
                return False
        shape = raw.get("output_shape")
        if shape != [
            12,
            1,
            (
                GCR_PACRE_EFFICIENCY_FEATURE_HEIGHT
                * GCR_PACRE_FORMAL_FEATURE_STRIDE
            ),
            (
                GCR_PACRE_EFFICIENCY_FEATURE_WIDTH
                * GCR_PACRE_FORMAL_FEATURE_STRIDE
            ),
        ]:
            return False
        output_shapes.add(tuple(shape))
        if (
            raw.get("field_tensor_bytes")
            != 12
            * GCR_PACRE_EFFICIENCY_FEATURE_HEIGHT
            * GCR_PACRE_FORMAL_FEATURE_STRIDE
            * GCR_PACRE_EFFICIENCY_FEATURE_WIDTH
            * GCR_PACRE_FORMAL_FEATURE_STRIDE
            * 4
            or raw.get("forward_warmups")
            != conditions.get("forward_warmups")
            or raw.get("forward_repeats")
            != expected_forward_samples
            or raw.get("train_step_warmups")
            != conditions.get("train_step_warmups")
            or raw.get("train_step_repeats")
            != expected_train_samples
            or raw.get("train_optimizer_steps")
            != (
                GCR_PACRE_EFFICIENCY_TRAIN_WARMUPS
                + GCR_PACRE_EFFICIENCY_TRAIN_REPEATS
            )
        ):
            return False
        memory = raw.get("memory")
        if not isinstance(memory, Mapping):
            return False
        if common_device == "cuda:0":
            if (
                memory.get("supported") is not True
                or not isinstance(
                    memory.get("peak_allocated_bytes"),
                    int,
                )
                or not isinstance(
                    memory.get("peak_reserved_bytes"),
                    int,
                )
                or int(memory["peak_allocated_bytes"]) < 1
                or int(memory["peak_reserved_bytes"]) < 1
            ):
                return False
        elif (
            common_device != "cpu"
            or memory
            != {
                "supported": False,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            }
        ):
            return False
    return (
        len(output_shapes) == 1
        and len(initial_fingerprints) == 1
        and conditions.get("forward_warmups")
        == GCR_PACRE_EFFICIENCY_FORWARD_WARMUPS
        and expected_forward_samples
        == GCR_PACRE_EFFICIENCY_FORWARD_REPEATS
        and conditions.get("train_step_warmups")
        == GCR_PACRE_EFFICIENCY_TRAIN_WARMUPS
        and expected_train_samples
        == GCR_PACRE_EFFICIENCY_TRAIN_REPEATS
        and isinstance(
            inventory.get("v24_additions_over_v23"),
            list,
        )
        and bool(inventory["v24_additions_over_v23"])
    )


def _derive_checks(
    *,
    identity: Mapping[str, object],
    algebra: Mapping[str, object],
    selectivity: Mapping[str, object],
    gradients: Mapping[str, object],
    efficiency: Mapping[str, object],
    boundary: Mapping[str, object],
    source_hashes: Mapping[str, object],
) -> dict[str, bool]:
    zero_feature = algebra.get("zero_feature")
    hard_union = algebra.get("hard_union")
    forced = algebra.get("forced_unit_gate")
    fp64 = algebra.get("fp64_envelope")
    reference_parity = algebra.get("reference_parity")
    gate_statistics = algebra.get("gate_statistics")
    endpoints = algebra.get("endpoint_witnesses")
    target = selectivity.get("target_like")
    background = selectivity.get("background_like")
    common_only = selectivity.get("common_only")
    step0 = gradients.get("step0")
    warmup = gradients.get("warmup")
    post = gradients.get("post_warmup")
    components = (
        fp64.get("components")
        if isinstance(fp64, Mapping)
        else None
    )
    upper = (
        endpoints.get("upper_statistics")
        if isinstance(endpoints, Mapping)
        else None
    )
    lower = (
        endpoints.get("lower_statistics")
        if isinstance(endpoints, Mapping)
        else None
    )
    gradient_errors = (
        step0.get("gradient_maximum_absolute_errors")
        if isinstance(step0, Mapping)
        else None
    )
    gradient_tolerance = (
        step0.get("gradient_absolute_tolerance")
        if isinstance(step0, Mapping)
        else None
    )
    expected_sources = dict(_current_source_hashes())
    checks = {
        "01_generated_boundary_closed": boundary
        == {
            "generated_only": True,
            "real_dataset_accessed": False,
            "cache_accessed": False,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "checkpoint_loaded": False,
            "threshold_search_performed": False,
            "mechanism_warmup_optimizer_constructed": True,
            "mechanism_warmup_updates": 1,
            "efficiency_training_updates_per_arm": (
                GCR_PACRE_EFFICIENCY_TRAIN_WARMUPS
                + GCR_PACRE_EFFICIENCY_TRAIN_REPEATS
            ),
            "performance_claim_supported": False,
            "evidence_role": (
                "generated_mechanism_and_efficiency_gate_only"
            ),
        },
        "02_source_binding_current": (
            dict(source_hashes) == expected_sources
            and set(source_hashes)
            == set(GCR_PACRE_DATASET_FREE_SOURCE_PATHS)
            and all(_is_sha256(value) for value in source_hashes.values())
        ),
        "03_canonical_v24_identity_exact": (
            identity.get("candidate") == GCR_PACRE_CANDIDATE
            and identity.get("canonical_identity_exact") is True
        ),
        "04_formal_parameter_count_64064": (
            identity.get("parameter_count")
            == GCR_PACRE_FORMAL_PARAMETER_COUNT
            and identity.get("expected_parameter_count")
            == GCR_PACRE_FORMAL_PARAMETER_COUNT
        ),
        "05_parameter_names_shapes_and_initial_bytes_match_v23": (
            identity.get(
                "initial_parameter_names_shapes_bytes_match_v23"
            )
            is True
            and identity.get("parameter_names")
            == list(GCR_PACRE_PARAMETER_NAMES)
            and identity.get("expected_parameter_names")
            == list(GCR_PACRE_PARAMETER_NAMES)
        ),
        "06_single_scalar_field_no_additional_head": (
            identity.get("parameter_tensor_count") == 3
            and identity.get("additional_head_count") == 0
            and identity.get("input_signature")
            == ["feature", "occupancy"]
            and identity.get("output_tensor_count") == 1
            and identity.get("field_threshold") == 0.0
        ),
        "07_lightweight_validator_called_by_forward": (
            algebra.get("lightweight_validator_forward_count") == 1
            and algebra.get("lightweight_validator_call_count") == 1
        ),
        "08_full_fields_replay_pass": (
            algebra.get("full_replay_passed") is True
            and algebra.get("model_state_unchanged") is True
        ),
        "09_fp64_oracle_envelope_pass": (
            isinstance(components, Mapping)
            and set(components)
            == {
                "residual_odd_interaction",
                "common_even_energy",
                "common_gate",
                "gated_interaction",
                "native_phase_field",
                "field",
            }
            and fp64.get("required_components") == ["field"]
            and all(
                isinstance(value, Mapping)
                and isinstance(
                    value.get("maximum_absolute_error"),
                    (int, float),
                )
                and not isinstance(
                    value.get("maximum_absolute_error"),
                    bool,
                )
                and isfinite(
                    float(value["maximum_absolute_error"])
                )
                and isinstance(value.get("maximum_ulp_distance"), int)
                and not isinstance(
                    value.get("maximum_ulp_distance"),
                    bool,
                )
                and int(value["maximum_ulp_distance"]) >= 0
                for value in components.values()
            )
            and isinstance(components.get("field"), Mapping)
            and components["field"].get("passed") is True
            and fp64.get("absolute_tolerance")
            == GCR_PACRE_FP64_ORACLE_ABS_TOL
            and fp64.get("maximum_allowed_ulp")
            == GCR_PACRE_FP64_ORACLE_MAX_ULP
        ),
        "10_non_unit_gate_proves_gcr_reference": (
            isinstance(algebra.get("non_unit_gate_count"), int)
            and int(algebra["non_unit_gate_count"]) > 0
            and algebra.get("fast_differs_from_legacy_ungated") is True
        ),
        "11_gate_finite_closed_and_statistics_complete": (
            isinstance(gate_statistics, Mapping)
            and gate_statistics.get("element_count", 0) > 0
            and gate_statistics.get("minimum", -1.0) >= 0.0
            and gate_statistics.get("maximum", 3.0) <= 2.0
            and gate_statistics.get("saturated_count")
            == gate_statistics.get("zero_count", 0)
            + gate_statistics.get("two_count", 0)
            and gate_statistics.get("element_count")
            == gate_statistics.get("saturated_count", 0)
            + gate_statistics.get("interior_count", 0)
        ),
        "12_gate_endpoint_and_interior_witnesses_present": (
            isinstance(gate_statistics, Mapping)
            and gate_statistics.get("interior_count", 0) > 0
            and isinstance(upper, Mapping)
            and upper.get("two_count", 0) > 0
            and isinstance(lower, Mapping)
            and lower.get("zero_count", 0) > 0
        ),
        "13_reference_common_even_flip_symmetric": (
            isinstance(reference_parity, Mapping)
            and reference_parity.get("common_even_symmetric") is True
        ),
        "14_reference_residual_odd_flip_antisymmetric": (
            isinstance(reference_parity, Mapping)
            and reference_parity.get("residual_odd_antisymmetric")
            is True
        ),
        "15_reference_gate_flip_symmetric": (
            isinstance(reference_parity, Mapping)
            and reference_parity.get("gate_symmetric") is True
        ),
        "16_reference_gated_interaction_flip_antisymmetric": (
            isinstance(reference_parity, Mapping)
            and reference_parity.get(
                "gated_interaction_antisymmetric"
            )
            is True
        ),
        "17_zero_common_energy_maps_to_unit_gate": (
            algebra.get("zero_common_maps_to_unit_gate") is True
        ),
        "18_zero_residual_maps_to_zero_interaction": (
            algebra.get("zero_residual_maps_to_zero_interaction") is True
        ),
        "19_zero_feature_field_exact_positive_point_nine": (
            isinstance(zero_feature, Mapping)
            and all(
                zero_feature.get(name) is True
                for name in (
                    "residual_exact_zero",
                    "common_exact_zero",
                    "gate_exact_one",
                    "field_exact_anchor",
                )
            )
        ),
        "20_target_like_gate_boosts_negative_residual": (
            isinstance(target, Mapping)
            and target.get("witness_present") is True
            and float(target.get("residual_odd", 1.0)) < 0.0
            and float(target.get("common_even", -1.0)) > 0.0
            and float(target.get("gate", 0.0)) > 1.0
            and float(target.get("gcr_minus_pacre", 1.0)) < 0.0
        ),
        "21_background_like_gate_suppresses_negative_residual": (
            isinstance(background, Mapping)
            and background.get("witness_present") is True
            and float(background.get("residual_odd", 1.0)) < 0.0
            and float(background.get("common_even", 1.0)) < 0.0
            and float(background.get("gate", 2.0)) < 1.0
            and float(background.get("gcr_minus_pacre", -1.0)) > 0.0
        ),
        "22_common_only_evidence_creates_no_completion": (
            isinstance(common_only, Mapping)
            and common_only.get("residual_exact_zero") is True
            and common_only.get("common_nonzero") is True
            and common_only.get("field_exact_anchor") is True
            and common_only.get("completion_count") == 0
        ),
        "23_forced_unit_gate_is_read_only": (
            isinstance(forced, Mapping)
            and all(
                forced.get(name) is True
                for name in (
                    "state_unchanged",
                    "gradients_unchanged",
                    "matches_ungated_equation",
                    "differs_from_gcr",
                )
            )
        ),
        "24_fixed_zero_threshold_and_hard_union_exact": (
            isinstance(hard_union, Mapping)
            and hard_union.get("threshold") == 0.0
            and all(
                hard_union.get(name) is True
                for name in (
                    "completion_exact",
                    "union_exact",
                    "retains_occupancy",
                )
            )
        ),
        "25_step0_v23_output_and_gradient_equivalence": (
            isinstance(step0, Mapping)
            and step0.get(
                "v23_v24_initial_parameter_bytes_equal"
            )
            is True
            and step0.get("output_raw_equal") is True
            and step0.get("residual_exact_zero") is True
            and step0.get("common_exact_zero") is True
            and step0.get("gate_exact_one") is True
            and step0.get("all_gradients_finite") is True
            and isinstance(gradient_errors, list)
            and len(gradient_errors) == 3
            and isinstance(gradient_tolerance, (int, float))
            and not isinstance(gradient_tolerance, bool)
            and float(gradient_tolerance)
            == GCR_PACRE_DATASET_FREE_STEP0_GRADIENT_ATOL
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(float(value))
                and float(value) <= float(gradient_tolerance)
                for value in gradient_errors
            )
        ),
        "26_one_real_frozen_pmope_warmup_update": (
            isinstance(warmup, Mapping)
            and warmup.get("objective")
            == CoverageStatePairObjective.PMOPE_JOINT.value
            and warmup.get("objective_policy") == CSLF_PMOPE_POLICY
            and warmup.get("expected_objective")
            == CoverageStatePairObjective.PMOPE_JOINT.value
            and warmup.get("expected_objective_policy")
            == CSLF_PMOPE_POLICY
            and warmup.get("function_fqcn")
            == (
                "cure_lite.train.coverage_state_fused_step."
                "coverage_state_fused_train_step"
            )
            and warmup.get("optimizer_fqcn")
            == "torch.optim.adam.Adam"
            and warmup.get("model_forward_calls") == 1
            and warmup.get("backward_calls") == 1
            and warmup.get("optimizer_steps") == 1
            and warmup.get("logical_states") == 12
            and warmup.get("parameter_state_changed") is True
            and warmup.get("all_logged_losses_finite") is True
            and _is_sha256(warmup.get("fixture_fingerprint"))
            and _is_sha256(warmup.get("selection_fingerprint"))
            and _is_sha256(
                warmup.get("optimizer_config_fingerprint")
            )
        ),
        "27_post_warmup_residual_path_gradient_nonzero": (
            isinstance(post, Mapping)
            and post.get("residual_nonzero_count", 0) > 0
            and _all_gradient_rows_pass(
                post.get("residual_path_parameter_gradients"),
                require_each_nonzero=True,
            )
        ),
        "28_post_warmup_gate_path_gradient_nonzero": (
            isinstance(post, Mapping)
            and post.get("common_nonzero_count", 0) > 0
            and _all_gradient_rows_pass(
                post.get("gate_path_parameter_gradients"),
                require_each_nonzero=True,
            )
        ),
        "29_warmup_total_pmope_gradient_reaches_shared_readout": (
            isinstance(warmup, Mapping)
            and _all_gradient_rows_pass(
                warmup.get("pmope_parameter_gradients"),
                require_each_nonzero=False,
            )
            and isinstance(
                warmup.get("pmope_parameter_gradients"),
                Mapping,
            )
            and warmup["pmope_parameter_gradients"].get(
                "scalar_energy_weight",
                {},
            ).get("nonzero_count", 0)
            > 0
        ),
        "30_efficiency_audit_complete_finite_and_threshold_free": (
            _efficiency_complete(efficiency)
        ),
    }
    if tuple(checks) != GCR_PACRE_DATASET_FREE_CHECK_NAMES:
        raise AssertionError("dataset-free check order changed")
    return checks


def _resolve_audit_device(device: torch.device | str) -> torch.device:
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise TypeError("device must be cpu or cuda:0") from error
    if resolved not in {torch.device("cpu"), torch.device("cuda:0")}:
        raise ValueError("dataset-free audit supports only cpu or cuda:0")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda:0 was requested but CUDA is unavailable")
    return resolved


def run_gcr_pacre_dataset_free_audit(
    *,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Run the complete generated-only mechanism and efficiency gate."""

    resolved_device = _resolve_audit_device(device)
    sources_before = _current_source_hashes()
    identity = _identity_probe()
    algebra = _forward_algebra_probe()
    selectivity = _selectivity_probe()
    gradients = _gradient_probe()
    efficiency = _efficiency_probe(resolved_device)
    sources_after = _current_source_hashes()
    if sources_before != sources_after:
        raise RuntimeError(
            "GCR-PACRE source changed during dataset-free audit"
        )
    source_hashes = dict(sources_before)
    boundary: dict[str, object] = {
        "generated_only": True,
        "real_dataset_accessed": False,
        "cache_accessed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "checkpoint_loaded": False,
        "threshold_search_performed": False,
        "mechanism_warmup_optimizer_constructed": True,
        "mechanism_warmup_updates": 1,
        "efficiency_training_updates_per_arm": (
            GCR_PACRE_EFFICIENCY_TRAIN_WARMUPS
            + GCR_PACRE_EFFICIENCY_TRAIN_REPEATS
        ),
        "performance_claim_supported": False,
        "evidence_role": (
            "generated_mechanism_and_efficiency_gate_only"
        ),
    }
    checks = _derive_checks(
        identity=identity,
        algebra=algebra,
        selectivity=selectivity,
        gradients=gradients,
        efficiency=efficiency,
        boundary=boundary,
        source_hashes=source_hashes,
    )
    failed = [
        name for name, passed in checks.items() if passed is not True
    ]
    decision_body: dict[str, object] = {
        "schema_version": GCR_PACRE_DATASET_FREE_DECISION_SCHEMA,
        "candidate": GCR_PACRE_CANDIDATE,
        "check_names": list(GCR_PACRE_DATASET_FREE_CHECK_NAMES),
        "checks_fingerprint": stable_fingerprint(checks),
        "gate_passed": not failed,
        "failed_checks": failed,
        "next_action": (
            "ELIGIBLE_FOR_EXTERNAL_D_R_STRUCTURAL_AUTHORIZATION_CHECKS"
            if not failed
            else "STOP_AND_REVISE_GCR_PACRE"
        ),
        "authorizes_D_R_execution": False,
        "authorizes_D_V_execution": False,
        "authorizes_D_T_execution": False,
    }
    decision = {
        **decision_body,
        "decision_fingerprint": stable_fingerprint(decision_body),
    }
    body: dict[str, object] = {
        "schema_version": GCR_PACRE_DATASET_FREE_SCHEMA,
        "candidate": GCR_PACRE_CANDIDATE,
        "execution_seed": GCR_PACRE_DATASET_FREE_EXECUTION_SEED,
        "efficiency_device": str(resolved_device),
        "source_hashes": source_hashes,
        "source_binding_fingerprint": stable_fingerprint(source_hashes),
        "evidence": {
            "identity": identity,
            "algebra": algebra,
            "selectivity": selectivity,
            "gradients": gradients,
            "efficiency": efficiency,
        },
        "boundary": boundary,
        "checks": checks,
        "decision": decision,
    }
    receipt = {
        **body,
        "receipt_fingerprint": stable_fingerprint(body),
    }
    verify_gcr_pacre_dataset_free_receipt(receipt)
    return receipt


def verify_gcr_pacre_dataset_free_receipt(
    receipt: Mapping[str, object],
) -> str:
    """Validate a sealed receipt without rerunning generated computation."""

    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    payload = dict(receipt)
    fingerprint = payload.pop("receipt_fingerprint", None)
    if (
        not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(payload)
    ):
        raise ValueError("dataset-free receipt_fingerprint is invalid")
    if (
        payload.get("schema_version") != GCR_PACRE_DATASET_FREE_SCHEMA
        or payload.get("candidate") != GCR_PACRE_CANDIDATE
        or payload.get("execution_seed")
        != GCR_PACRE_DATASET_FREE_EXECUTION_SEED
        or payload.get("efficiency_device") not in {"cpu", "cuda:0"}
    ):
        raise ValueError("dataset-free receipt identity changed")
    source_hashes = payload.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes)
        != set(GCR_PACRE_DATASET_FREE_SOURCE_PATHS)
        or dict(source_hashes) != dict(_current_source_hashes())
        or payload.get("source_binding_fingerprint")
        != stable_fingerprint(dict(source_hashes))
    ):
        raise ValueError("dataset-free source binding changed")
    evidence = payload.get("evidence")
    if (
        not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "identity",
            "algebra",
            "selectivity",
            "gradients",
            "efficiency",
        }
    ):
        raise ValueError("dataset-free evidence inventory changed")
    verified = {
        name: _verify_section(value, name=name)
        for name, value in evidence.items()
    }
    boundary = payload.get("boundary")
    if not isinstance(boundary, Mapping):
        raise TypeError("dataset-free boundary must be a mapping")
    expected_checks = _derive_checks(
        identity=verified["identity"],
        algebra=verified["algebra"],
        selectivity=verified["selectivity"],
        gradients=verified["gradients"],
        efficiency=verified["efficiency"],
        boundary=boundary,
        source_hashes=source_hashes,
    )
    if (
        payload.get("checks") != expected_checks
        or tuple(expected_checks)
        != GCR_PACRE_DATASET_FREE_CHECK_NAMES
        or not all(type(value) is bool for value in expected_checks.values())
    ):
        raise ValueError("dataset-free checks differ from evidence")
    failed = [
        name
        for name, passed in expected_checks.items()
        if passed is not True
    ]
    decision = payload.get("decision")
    if not isinstance(decision, Mapping):
        raise TypeError("dataset-free decision must be a mapping")
    decision_payload = dict(decision)
    decision_fingerprint = decision_payload.pop(
        "decision_fingerprint",
        None,
    )
    if (
        not _is_sha256(decision_fingerprint)
        or decision_fingerprint
        != stable_fingerprint(decision_payload)
        or decision_payload.get("schema_version")
        != GCR_PACRE_DATASET_FREE_DECISION_SCHEMA
        or decision_payload.get("candidate") != GCR_PACRE_CANDIDATE
        or decision_payload.get("check_names")
        != list(GCR_PACRE_DATASET_FREE_CHECK_NAMES)
        or decision_payload.get("checks_fingerprint")
        != stable_fingerprint(expected_checks)
        or decision_payload.get("gate_passed") is not (not failed)
        or decision_payload.get("failed_checks") != failed
        or decision_payload.get("next_action")
        != (
            "ELIGIBLE_FOR_EXTERNAL_D_R_STRUCTURAL_AUTHORIZATION_CHECKS"
            if not failed
            else "STOP_AND_REVISE_GCR_PACRE"
        )
        or decision_payload.get("authorizes_D_R_execution") is not False
        or decision_payload.get("authorizes_D_V_execution") is not False
        or decision_payload.get("authorizes_D_T_execution") is not False
    ):
        raise ValueError("dataset-free decision binding changed")
    if failed:
        raise PermissionError(
            "GCR-PACRE dataset-free gate failed: " + ", ".join(failed)
        )
    return str(fingerprint)


__all__ = [
    "GCR_PACRE_DATASET_FREE_CHECK_NAMES",
    "GCR_PACRE_DATASET_FREE_DECISION_SCHEMA",
    "GCR_PACRE_DATASET_FREE_EXECUTION_SEED",
    "GCR_PACRE_DATASET_FREE_SCHEMA",
    "GCR_PACRE_DATASET_FREE_SOURCE_PATHS",
    "GCR_PACRE_EFFICIENCY_FORWARD_REPEATS",
    "GCR_PACRE_EFFICIENCY_FORWARD_WARMUPS",
    "GCR_PACRE_EFFICIENCY_TRAIN_REPEATS",
    "GCR_PACRE_EFFICIENCY_TRAIN_WARMUPS",
    "run_gcr_pacre_dataset_free_audit",
    "verify_gcr_pacre_dataset_free_receipt",
]
