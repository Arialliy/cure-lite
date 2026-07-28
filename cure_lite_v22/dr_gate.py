"""Read-only real-``D_R`` structural gate for CURE-Lite v22 PACRE.

The gate checks whether the exact PACRE representation and its frozen
PMOPE objective expose a non-degenerate training path on the fixed seed-42
bounded population.  It neither estimates performance nor trains a model.
Only ``D_R`` objects already built by the common CURE-Lite data pipeline are
accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from math import isfinite
from pathlib import Path
from typing import Final, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.experiment.coverage_state_bfa_dr_gate import (
    _direction_probe,
    _pair_targets_to_device,
    _phase_hidden_to_output,
    _state_specs,
    _vectors_at,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    COVERAGE_STATE_BOUNDED_SEED,
    CoverageStateBoundedPopulation,
)
from cure_lite.experiment.coverage_state_paet_dr_gate import (
    _cuda_rng_devices,
    _deterministic_execution_scope,
    _row_bit_hash,
    _scan_exact_collisions,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRInputs,
)
from cure_lite.experiment.coverage_state_training import (
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
)
from cure_lite.coverage_state_sobolev import (
    coverage_state_pmope_pair_loss_from_targets,
)
from cure_lite.paired_types import tensor_content_fingerprint

from .dataset_free import (
    PACRE_FORMAL_FEATURE_CHANNELS,
    PACRE_FORMAL_FEATURE_STRIDE,
    PACRE_FORMAL_PARAMETER_COUNT,
    PACRE_FORMAL_WIDTH,
    _implementation_binding as _dataset_free_implementation_binding,
)
from .factory import PACRE_TRAINING_MODEL_FACTORY
from .pacre import (
    CSLF_PACRE_CENTERING_POLICY,
    CSLF_PACRE_EQUATION_POLICY,
    CSLF_PACRE_FIELD_POLICY,
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
    CoverageStatePACREFields,
)


PACRE_DR_GATE_SCHEMA: Final = (
    "cure-lite-pacre-v22-real-dr-structural-gate-v1"
)
PACRE_DR_EXECUTION_SEED: Final = 42
PACRE_DR_SEPARATION_THRESHOLD: Final = (
    128.0 * torch.finfo(torch.float32).eps
)
PACRE_DR_PASS_DECISION: Final = "PACRE_V22_D_R_STRUCTURAL_PASS"
PACRE_DR_FAIL_DECISION: Final = "PACRE_V22_D_R_STRUCTURAL_FAIL"
PACRE_DR_CHECK_NAMES: Final = (
    "01_dataset_free_prerequisite_exact_and_passed",
    "02_real_D_R_seed42_population_bound",
    "03_exact_pacre_model_config_factory_and_parameter_contract",
    "04_complete_state_forward_ledger_and_exact_fields_type",
    "05_phase_residual_and_compatibility_algebra_valid",
    "06_each_target_group_has_one_bound_residual_flip_latent_witness",
    "07_no_exact_target_positive_latent_collision",
    "08_zero_readout_anchor_and_fixed_readout_witness",
    "09_real_pmope_initialization_gradient_path",
    "10_field_loss_direction_correct_for_all_roles",
    "11_model_population_cache_rng_and_grad_buffers_preserved",
    "12_read_only_zero_update_D_R_scope",
)
PACRE_DR_IMPLEMENTATION_PATHS: Final = (
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/experiment/coverage_state_bfa_dr_gate.py",
    "cure_lite/experiment/coverage_state_bounded_protocol.py",
    "cure_lite/experiment/coverage_state_paet_dr_gate.py",
    "cure_lite/experiment/coverage_state_real_dr_inputs.py",
    "cure_lite_v22/pacre.py",
    "cure_lite_v22/factory.py",
    "cure_lite_v22/dataset_free.py",
    "cure_lite_v22/dr_gate.py",
)


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[1]
    rows: list[tuple[str, str]] = []
    for relative in PACRE_DR_IMPLEMENTATION_PATHS:
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
        ):
            raise RuntimeError(
                f"invalid PACRE D_R implementation path: {relative}"
            )
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


def _finite_hex(value: float, *, name: str) -> str:
    result = float(value)
    if not isfinite(result):
        raise FloatingPointError(f"{name} must be finite")
    return result.hex()


def _validate_dataset_free_receipt(
    receipt: Mapping[str, object],
) -> str:
    if not isinstance(receipt, Mapping):
        raise TypeError("dataset_free_receipt must be a mapping")
    body = dict(receipt)
    fingerprint = body.pop("receipt_fingerprint", None)
    expected_implementation = [
        {"repo_path": name, "sha256": digest}
        for name, digest in _dataset_free_implementation_binding()
    ]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or stable_fingerprint(body) != fingerprint
        or body.get("candidate") != "PACRE-v22"
        or body.get("gate_passed") is not True
        or body.get("D_R_accessed") is not False
        or body.get("D_V_accessed") is not False
        or body.get("D_T_accessed") is not False
        or body.get("training_performed") is not False
        or body.get("implementation_binding")
        != expected_implementation
    ):
        raise PermissionError(
            "PACRE dataset-free prerequisite is invalid"
        )
    return fingerprint


def _normalized_separation(first: Tensor, second: Tensor) -> float:
    if (
        first.ndim != 1
        or second.ndim != 1
        or first.shape != second.shape
        or first.dtype != torch.float32
        or second.dtype != torch.float32
    ):
        raise ValueError("separation vectors must be aligned FP32 rows")
    numerator = torch.linalg.vector_norm(first - second)
    denominator = torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(first),
            torch.linalg.vector_norm(second),
        ),
        torch.ones((), dtype=torch.float32, device=first.device),
    )
    return float((numerator / denominator).detach().cpu())


def _separation_summary(values: list[float]) -> dict[str, object]:
    if not values or any(not isfinite(value) or value < 0.0 for value in values):
        raise ValueError("separation summary requires finite values")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum_hex": ordered[0].hex(),
        "median_hex": ordered[(len(ordered) - 1) // 2].hex(),
        "maximum_hex": ordered[-1].hex(),
        "above_threshold_count": sum(
            value > PACRE_DR_SEPARATION_THRESHOLD for value in ordered
        ),
    }


def _bound_pacre_pair_witness(
    phase_residual: Tensor,
    actual_hidden: Tensor,
    flipped_hidden: Tensor,
    latent: Tensor,
    *,
    target_mask: Tensor,
    background_mask: Tensor,
    stride: int,
) -> dict[str, object]:
    """Find same-cell target/background pairs using the exact PACRE latent."""

    tensors = (phase_residual, actual_hidden, flipped_hidden, latent)
    if (
        any(
            not isinstance(value, Tensor)
            or value.ndim != 4
            or value.dtype != torch.float32
            for value in tensors
        )
        or len({tuple(value.shape) for value in tensors}) != 1
        or target_mask.dtype != torch.bool
        or background_mask.dtype != torch.bool
        or target_mask.shape != background_mask.shape
        or target_mask.shape[0] != phase_residual.shape[0]
        or tuple(target_mask.shape[-2:])
        != tuple(phase_residual.shape[-2:])
        or isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride < 2
    ):
        raise ValueError("PACRE bound-pair tensors do not align")
    coordinates = torch.nonzero(target_mask[:, 0], as_tuple=False)
    height, width = target_mask.shape[-2:]
    residual_values: list[float] = []
    flip_values: list[float] = []
    latent_values: list[float] = []
    legal_pairs = 0
    jointly_separated = 0
    covered_targets: set[tuple[int, int, int]] = set()
    selected: dict[str, object] | None = None
    binding_rows: list[dict[str, object]] = []
    for coordinate in coordinates:
        batch, row, column = (
            int(value) for value in coordinate.tolist()
        )
        coarse = (batch, row // stride, column // stride)
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                neighbour_row = row + row_delta
                neighbour_column = column + column_delta
                if (
                    not 0 <= neighbour_row < height
                    or not 0 <= neighbour_column < width
                    or not bool(
                        background_mask[
                            batch,
                            0,
                            neighbour_row,
                            neighbour_column,
                        ]
                    )
                    or neighbour_row // stride != coarse[1]
                    or neighbour_column // stride != coarse[2]
                ):
                    continue
                residual_separation = _normalized_separation(
                    phase_residual[batch, :, row, column],
                    phase_residual[
                        batch, :, neighbour_row, neighbour_column
                    ],
                )
                flip_separation = _normalized_separation(
                    actual_hidden[batch, :, row, column],
                    flipped_hidden[batch, :, row, column],
                )
                latent_separation = _normalized_separation(
                    latent[batch, :, row, column],
                    latent[
                        batch, :, neighbour_row, neighbour_column
                    ],
                )
                jointly = all(
                    value > PACRE_DR_SEPARATION_THRESHOLD
                    for value in (
                        residual_separation,
                        flip_separation,
                        latent_separation,
                    )
                )
                legal_pairs += 1
                jointly_separated += int(jointly)
                covered_targets.add((batch, row, column))
                residual_values.append(residual_separation)
                flip_values.append(flip_separation)
                latent_values.append(latent_separation)
                row_payload = {
                    "target": [batch, row, column],
                    "background": [
                        batch,
                        neighbour_row,
                        neighbour_column,
                    ],
                    "coarse_cell": list(coarse),
                    "target_phase": (
                        (row % stride) * stride + column % stride
                    ),
                    "background_phase": (
                        (neighbour_row % stride) * stride
                        + neighbour_column % stride
                    ),
                    "residual_separation_hex": (
                        residual_separation.hex()
                    ),
                    "target_flip_separation_hex": (
                        flip_separation.hex()
                    ),
                    "latent_separation_hex": latent_separation.hex(),
                    "jointly_above_threshold": jointly,
                }
                binding_rows.append(row_payload)
                if jointly and selected is None:
                    selected = row_payload
    return {
        "target_coordinate_count": int(coordinates.shape[0]),
        "target_coordinates_with_legal_background": len(covered_targets),
        "target_coordinates_without_legal_background": (
            int(coordinates.shape[0]) - len(covered_targets)
        ),
        "legal_pair_count": legal_pairs,
        "jointly_separated_pair_count": jointly_separated,
        "at_least_one_jointly_separated_pair": selected is not None,
        "selected_first_joint_witness": selected,
        "binding_fingerprint": stable_fingerprint(binding_rows),
        "separation_threshold_hex": (
            PACRE_DR_SEPARATION_THRESHOLD.hex()
        ),
        "residual_separation": (
            _separation_summary(residual_values)
            if residual_values
            else None
        ),
        "target_flip_separation": (
            _separation_summary(flip_values) if flip_values else None
        ),
        "latent_separation": (
            _separation_summary(latent_values)
            if latent_values
            else None
        ),
        "same_pair_for_all_three_checks": True,
        "legal_background_policy": (
            "state_background_mask_same_coarse_cell_chebyshev1"
        ),
    }


def _algebra_checks(fields: CoverageStatePACREFields) -> bool:
    residual = fields.phase_feature_residual
    phases = int(residual.shape[1])
    tolerance = (
        4.0
        * float(phases)
        * torch.finfo(torch.float32).eps
        * (
            1.0
            + float(
                fields.phase_feature_affine.detach().abs().amax()
            )
        )
    )
    checks = (
        torch.allclose(
            fields.phase_feature_mean + residual,
            fields.phase_feature_affine,
            rtol=0.0,
            atol=tolerance,
        ),
        bool(
            torch.all(
                residual.sum(dim=1).abs() <= tolerance
            )
        ),
        torch.allclose(
            fields.actual_specific_joint_affine
            - fields.actual_common_joint_affine,
            residual,
            rtol=2.0e-6,
            atol=2.0e-7,
        ),
        torch.allclose(
            fields.flipped_specific_joint_affine
            - fields.flipped_common_joint_affine,
            residual,
            rtol=2.0e-6,
            atol=2.0e-7,
        ),
        torch.allclose(
            fields.actual_compatibility_hidden,
            F.silu(fields.actual_specific_joint_affine)
            - F.silu(fields.actual_common_joint_affine),
            rtol=2.0e-6,
            atol=2.0e-7,
        ),
        torch.allclose(
            fields.flipped_compatibility_hidden,
            F.silu(fields.flipped_specific_joint_affine)
            - F.silu(fields.flipped_common_joint_affine),
            rtol=2.0e-6,
            atol=2.0e-7,
        ),
    )
    return all(bool(value) for value in checks)


def _representation_probe(
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    target_states, context_states = _state_specs(population)
    target_hash_rows: dict[int, list[dict[str, object]]] = {}
    target_rows: list[dict[str, object]] = []
    target_coordinates_total = 0
    target_forwards = 0
    all_fields_exact = True
    all_algebra = True
    all_zero_anchor = True
    fixed_readout_nonzero = False
    fixed_readout = torch.linspace(
        0.5,
        1.5,
        model.config.width,
        device=device,
        dtype=torch.float32,
    )
    for state in target_states:
        feature = state.feature.to(device=device, dtype=torch.float32)
        occupancy = state.occupancy.to(device=device)
        with torch.no_grad():
            fields = model.forward_fields(feature, occupancy)
        target_forwards += 1
        all_fields_exact &= type(fields) is CoverageStatePACREFields
        all_algebra &= _algebra_checks(fields)
        all_zero_anchor &= bool(
            torch.count_nonzero(fields.native_phase_interaction) == 0
            and torch.equal(
                fields.field,
                torch.full_like(
                    fields.field,
                    model.config.field_amplitude,
                ),
            )
        )
        latent_native = 0.5 * (
            fields.actual_compatibility_hidden
            - fields.flipped_compatibility_hidden
        )
        stride = model.config.feature_stride
        residual = _phase_hidden_to_output(
            fields.phase_feature_residual,
            stride=stride,
        )
        actual = _phase_hidden_to_output(
            fields.actual_compatibility_hidden,
            stride=stride,
        )
        flipped = _phase_hidden_to_output(
            fields.flipped_compatibility_hidden,
            stride=stride,
        )
        latent = _phase_hidden_to_output(
            latent_native,
            stride=stride,
        )
        target_mask = state.target_mask.to(device=device)
        witness = _bound_pacre_pair_witness(
            residual,
            actual,
            flipped,
            latent,
            target_mask=target_mask,
            background_mask=state.background_mask.to(device=device),
            stride=stride,
        )
        target_vectors = _vectors_at(latent, target_mask)
        target_coordinates = torch.nonzero(
            state.target_mask[:, 0],
            as_tuple=False,
        ).to("cpu")
        if target_vectors.shape[0] != target_coordinates.shape[0]:
            raise AssertionError("PACRE target vectors do not align")
        hashes = _row_bit_hash(target_vectors.detach().to("cpu"))
        for index in range(target_vectors.shape[0]):
            hash_value = int(hashes[index].item())
            target_hash_rows.setdefault(hash_value, []).append(
                {
                    "target_group_id": state.target_group_id,
                    "state_id": state.state_id,
                    "sample_id": state.sample_id,
                    "coordinate": target_coordinates[index].tolist(),
                    "vector": target_vectors[index]
                    .detach()
                    .to("cpu")
                    .contiguous(),
                }
            )
        functional_interaction = (
            latent
            * fixed_readout[None, :, None, None]
        ).sum(dim=1, keepdim=True)
        fixed_readout_nonzero |= bool(
            torch.any(functional_interaction != 0.0)
        )
        target_coordinates_total += int(target_vectors.shape[0])
        target_rows.append(
            {
                "target_group_id": state.target_group_id,
                "state_id": state.state_id,
                "sample_id": state.sample_id,
                "state_kind": state.state_kind,
                "target_coordinate_count": int(
                    target_vectors.shape[0]
                ),
                "target_mask_fingerprint": (
                    tensor_content_fingerprint(state.target_mask)
                ),
                "latent_target_fingerprint": (
                    tensor_content_fingerprint(
                        target_vectors.detach().to("cpu")
                    )
                ),
                "bound_pair_witness": witness,
            }
        )
    if not target_hash_rows:
        raise RuntimeError("PACRE D_R has no target latent vectors")

    target_hashes = torch.tensor(
        sorted(target_hash_rows),
        dtype=torch.int64,
    )
    collisions = 0
    examples: list[dict[str, object]] = []
    context_forwards = 0
    role_counts = {"background": 0, "component": 0}
    for state in context_states:
        feature = state.feature.to(device=device, dtype=torch.float32)
        occupancy = state.occupancy.to(device=device)
        with torch.no_grad():
            fields = model.forward_fields(feature, occupancy)
        context_forwards += 1
        all_fields_exact &= type(fields) is CoverageStatePACREFields
        latent = _phase_hidden_to_output(
            0.5
            * (
                fields.actual_compatibility_hidden
                - fields.flipped_compatibility_hidden
            ),
            stride=model.config.feature_stride,
        )
        for role, mask_cpu in (
            ("background", state.background_mask),
            ("component", state.component_mask),
        ):
            if not bool(torch.any(mask_cpu)):
                continue
            vectors = _vectors_at(
                latent,
                mask_cpu.to(device=device),
            ).detach().to("cpu").contiguous()
            coordinates = torch.nonzero(
                mask_cpu[:, 0],
                as_tuple=False,
            ).to("cpu")
            role_counts[role] += int(vectors.shape[0])
            collisions += _scan_exact_collisions(
                vectors,
                coordinates=coordinates,
                state=state,
                role=role,
                target_hashes=target_hashes,
                targets_by_hash=target_hash_rows,
                examples=examples,
            )
    return {
        "target_group_count": len(target_rows),
        "target_coordinate_count": target_coordinates_total,
        "target_forward_calls": target_forwards,
        "context_state_count": len(context_states),
        "context_forward_calls": context_forwards,
        "all_fields_exact_pacre": all_fields_exact,
        "all_algebra_checks_passed": all_algebra,
        "all_target_groups_have_joint_witness": all(
            row["bound_pair_witness"][
                "at_least_one_jointly_separated_pair"
            ]
            is True
            for row in target_rows
        ),
        "target_rows": target_rows,
        "exact_latent_collision_count": collisions,
        "exact_collision_examples": examples,
        "positive_role_coordinate_counts": role_counts,
        "zero_readout_anchor_all_target_states": all_zero_anchor,
        "fixed_readout_interaction_nonzero": (
            fixed_readout_nonzero
        ),
        "fixed_readout_policy": (
            "linspace_0.5_to_1.5_width_stateless_witness_v1"
        ),
        "two_pass_streaming": True,
    }


def _gradient_probe(
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    clean = sorted(
        population.cache.clean_positive_records,
        key=lambda value: value.record.pair_id,
    )
    if not clean:
        raise ValueError("PACRE gradient probe requires a clean pair")
    value = clean[0]
    record = value.record
    feature = torch.cat(
        (record.feature, record.feature),
        dim=0,
    ).to(device=device, dtype=torch.float32)
    occupancy = torch.cat(
        (record.occupancy_plus, record.occupancy_minus),
        dim=0,
    ).to(device=device)
    targets = _pair_targets_to_device(
        value.joint_targets,
        device=device,
    )
    field_plus, field_minus = model(feature, occupancy).split(
        1,
        dim=0,
    )
    result = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=population.cache.sobolev_config,
        validate=True,
    )
    parameters = dict(model.named_parameters())
    gradients = dict(
        zip(
            parameters,
            torch.autograd.grad(
                result.loss,
                tuple(parameters.values()),
                create_graph=True,
                allow_unused=False,
            ),
            strict=True,
        )
    )
    cross = torch.autograd.grad(
        gradients["scalar_energy_weight"].square().sum(),
        (
            parameters["joint_state_weight"],
            parameters["joint_hidden_bias"],
        ),
        allow_unused=False,
    )
    return {
        "pair_id": record.pair_id,
        "sample_id": record.sample_id,
        "selection_policy": (
            "lexicographically_first_clean_positive_pair_v1"
        ),
        "loss_hex": _finite_hex(
            float(result.loss.detach().cpu()),
            name="PACRE PMOPE initialization loss",
        ),
        "initial_gradient_nonzero": {
            name: bool(torch.any(gradient != 0.0))
            for name, gradient in gradients.items()
        },
        "initial_gradient_finite": {
            name: bool(torch.isfinite(gradient).all())
            for name, gradient in gradients.items()
        },
        "readout_visible_upstream_dormant": (
            bool(torch.any(gradients["scalar_energy_weight"] != 0.0))
            and int(
                torch.count_nonzero(
                    gradients["joint_state_weight"]
                ).detach().cpu()
            )
            == 0
            and int(
                torch.count_nonzero(
                    gradients["joint_hidden_bias"]
                ).detach().cpu()
            )
            == 0
        ),
        "readout_to_upstream_cross_gradient_finite_nonzero": [
            bool(torch.isfinite(gradient).all())
            and bool(torch.any(gradient != 0.0))
            for gradient in cross
        ],
        "parameter_grad_buffers_unretained": all(
            parameter.grad is None
            for parameter in model.parameters()
        ),
    }


def _probe(
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    before_cpu_rng = torch.get_rng_state().clone()
    before_cuda_rng = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda"
        else None
    )
    before_population = population.population_fingerprint
    before_cache = population.cache.cache_fingerprint
    with _deterministic_execution_scope() as deterministic_execution:
        with torch.random.fork_rng(
            devices=_cuda_rng_devices(device),
            device_type=(
                "cuda" if device.type == "cuda" else None
            ),
        ):
            torch.random.default_generator.manual_seed(
                PACRE_DR_EXECUTION_SEED
            )
            config = CoverageStatePACREConfig(
                feature_channels=PACRE_FORMAL_FEATURE_CHANNELS,
                feature_stride=PACRE_FORMAL_FEATURE_STRIDE,
                width=PACRE_FORMAL_WIDTH,
            )
            model = PACRE_TRAINING_MODEL_FACTORY(config).to(
                device=device,
                dtype=torch.float32,
            )
            model.eval()
            initial_model = coverage_state_model_fingerprint(model)
            initial_ids = {
                name: id(parameter)
                for name, parameter in model.named_parameters()
            }
            representation = _representation_probe(
                model,
                population,
                device=device,
            )
            gradient = _gradient_probe(
                model,
                population,
                device=device,
            )
            direction = _direction_probe(
                population,
                device=device,
            )
            final_model = coverage_state_model_fingerprint(model)
            final_ids = {
                name: id(parameter)
                for name, parameter in model.named_parameters()
            }
            model_contract = coverage_state_model_contract_payload(model)
            parameter_grad_buffers_unretained = all(
                parameter.grad is None
                for parameter in model.parameters()
            )
    population.verify_unchanged()
    after_cuda_rng = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda"
        else None
    )
    return {
        "device": str(device),
        "execution_seed": PACRE_DR_EXECUTION_SEED,
        "model_fqcn": (
            f"{type(model).__module__}.{type(model).__qualname__}"
        ),
        "config_fqcn": (
            f"{type(config).__module__}.{type(config).__qualname__}"
        ),
        "model_contract": model_contract,
        "model_contract_fingerprint": stable_fingerprint(
            model_contract
        ),
        "initial_model_fingerprint": initial_model,
        "final_model_fingerprint": final_model,
        "parameter_ids_preserved": initial_ids == final_ids,
        "representation": representation,
        "gradient_path": gradient,
        "field_direction": direction,
        "deterministic_execution": deterministic_execution,
        "population_fingerprint_before": before_population,
        "population_fingerprint_after": (
            population.population_fingerprint
        ),
        "cache_fingerprint_before": before_cache,
        "cache_fingerprint_after": (
            population.cache.cache_fingerprint
        ),
        "global_cpu_rng_preserved": torch.equal(
            before_cpu_rng,
            torch.get_rng_state(),
        ),
        "selected_device_rng_preserved": (
            before_cuda_rng is None
            or torch.equal(before_cuda_rng, after_cuda_rng)
        ),
        "parameter_grad_buffers_unretained": (
            parameter_grad_buffers_unretained
            and gradient["parameter_grad_buffers_unretained"]
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "D_R_accessed": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def recompute_pacre_dr_checks(
    *,
    dataset_free_receipt_fingerprint: str,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
    probe: Mapping[str, object],
) -> tuple[tuple[str, bool], ...]:
    representation = probe.get("representation")
    gradient = probe.get("gradient_path")
    direction = probe.get("field_direction")
    model_contract = probe.get("model_contract")
    contract_config = (
        model_contract.get("config")
        if isinstance(model_contract, Mapping)
        else None
    )
    parameter_shapes = (
        model_contract.get("parameter_shapes")
        if isinstance(model_contract, Mapping)
        else None
    )
    if (
        not isinstance(representation, Mapping)
        or not isinstance(gradient, Mapping)
        or not isinstance(direction, Mapping)
        or not isinstance(model_contract, Mapping)
        or not isinstance(contract_config, Mapping)
        or not isinstance(parameter_shapes, Mapping)
    ):
        raise TypeError("PACRE D_R probe is incomplete")
    checks = {
        "01_dataset_free_prerequisite_exact_and_passed": (
            isinstance(dataset_free_receipt_fingerprint, str)
            and len(dataset_free_receipt_fingerprint) == 64
        ),
        "02_real_D_R_seed42_population_bound": (
            real_inputs.source_binding.split == "D_R"
            and real_inputs.scalar_cache.raw_catalog.split == "D_R"
            and bounded_population.seed == COVERAGE_STATE_BOUNDED_SEED
            and bounded_population.source_cache
            is real_inputs.scalar_cache
            and bounded_population.source_cache_fingerprint
            == real_inputs.scalar_cache.cache_fingerprint
        ),
        "03_exact_pacre_model_config_factory_and_parameter_contract": (
            probe.get("model_fqcn")
            == (
                "cure_lite_v22.pacre."
                "CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet"
            )
            and probe.get("config_fqcn")
            == "cure_lite_v22.pacre.CoverageStatePACREConfig"
            and model_contract.get("parameter_count")
            == PACRE_FORMAL_PARAMETER_COUNT
            and model_contract.get("model_class")
            == probe.get("model_fqcn")
            and model_contract.get("config_class")
            == probe.get("config_fqcn")
            and contract_config.get("feature_channels")
            == PACRE_FORMAL_FEATURE_CHANNELS
            and contract_config.get("feature_stride")
            == PACRE_FORMAL_FEATURE_STRIDE
            and contract_config.get("width") == PACRE_FORMAL_WIDTH
            and contract_config.get("field_policy")
            == CSLF_PACRE_FIELD_POLICY
            and contract_config.get("equation_policy")
            == CSLF_PACRE_EQUATION_POLICY
            and contract_config.get("centering_policy")
            == CSLF_PACRE_CENTERING_POLICY
            and parameter_shapes
            == {
                "joint_hidden_bias": [PACRE_FORMAL_WIDTH],
                "joint_state_weight": [
                    PACRE_FORMAL_WIDTH,
                    PACRE_FORMAL_FEATURE_CHANNELS
                    + PACRE_FORMAL_FEATURE_STRIDE**2,
                    5,
                    5,
                ],
                "scalar_energy_weight": [PACRE_FORMAL_WIDTH],
            }
            and probe.get("model_contract_fingerprint")
            == stable_fingerprint(model_contract)
        ),
        "04_complete_state_forward_ledger_and_exact_fields_type": (
            representation.get("target_group_count")
            == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and representation.get("target_forward_calls")
            == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and representation.get("context_state_count")
            == 6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and representation.get("context_state_count")
            == representation.get("context_forward_calls")
            and representation.get("all_fields_exact_pacre") is True
        ),
        "05_phase_residual_and_compatibility_algebra_valid": (
            representation.get("all_algebra_checks_passed") is True
        ),
        "06_each_target_group_has_one_bound_residual_flip_latent_witness": (
            representation.get(
                "all_target_groups_have_joint_witness"
            )
            is True
        ),
        "07_no_exact_target_positive_latent_collision": (
            representation.get("exact_latent_collision_count") == 0
        ),
        "08_zero_readout_anchor_and_fixed_readout_witness": (
            representation.get(
                "zero_readout_anchor_all_target_states"
            )
            is True
            and representation.get(
                "fixed_readout_interaction_nonzero"
            )
            is True
        ),
        "09_real_pmope_initialization_gradient_path": (
            gradient.get("initial_gradient_finite")
            == {
                "joint_hidden_bias": True,
                "joint_state_weight": True,
                "scalar_energy_weight": True,
            }
            and gradient.get("initial_gradient_nonzero")
            == {
                "joint_hidden_bias": False,
                "joint_state_weight": False,
                "scalar_energy_weight": True,
            }
            and gradient.get("readout_visible_upstream_dormant") is True
            and gradient.get(
                "readout_to_upstream_cross_gradient_finite_nonzero"
            )
            == [True, True]
            and gradient.get("parameter_grad_buffers_unretained")
            is True
        ),
        "10_field_loss_direction_correct_for_all_roles": (
            direction.get("all_roles_finite_nonzero_correct")
            is True
        ),
        "11_model_population_cache_rng_and_grad_buffers_preserved": (
            probe.get("initial_model_fingerprint")
            == probe.get("final_model_fingerprint")
            and probe.get("parameter_ids_preserved") is True
            and probe.get("population_fingerprint_before")
            == probe.get("population_fingerprint_after")
            and probe.get("cache_fingerprint_before")
            == probe.get("cache_fingerprint_after")
            and probe.get("global_cpu_rng_preserved") is True
            and probe.get("selected_device_rng_preserved") is True
            and probe.get("parameter_grad_buffers_unretained") is True
            and isinstance(
                probe.get("deterministic_execution"),
                Mapping,
            )
            and probe["deterministic_execution"].get(
                "restored_exactly"
            )
            is True
        ),
        "12_read_only_zero_update_D_R_scope": (
            probe.get("D_R_accessed") is True
            and probe.get("D_V_accessed") is False
            and probe.get("D_T_accessed") is False
            and probe.get("optimizer_constructed") is False
            and probe.get("optimizer_steps") == 0
            and probe.get("parameter_updates") == 0
            and probe.get("training_performed") is False
        ),
    }
    if tuple(checks) != PACRE_DR_CHECK_NAMES:
        raise AssertionError("PACRE D_R check order changed")
    return tuple(checks.items())


@dataclass(frozen=True)
class CoverageStatePACREDRGateReceipt:
    """Immutable binding of one read-only PACRE ``D_R`` probe."""

    dataset_free_receipt_fingerprint: str
    real_inputs_fingerprint: str
    population_fingerprint: str
    cache_fingerprint: str
    implementation_binding: tuple[tuple[str, str], ...]
    probe_json: str
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        digests = (
            self.dataset_free_receipt_fingerprint,
            self.real_inputs_fingerprint,
            self.population_fingerprint,
            self.cache_fingerprint,
            *(digest for _, digest in self.implementation_binding),
        )
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in digests
        ):
            raise ValueError("PACRE D_R receipt digest is malformed")
        if (
            tuple(name for name, _ in self.implementation_binding)
            != PACRE_DR_IMPLEMENTATION_PATHS
            or tuple(name for name, _ in self.checks)
            != PACRE_DR_CHECK_NAMES
            or any(not isinstance(value, bool) for _, value in self.checks)
        ):
            raise ValueError("PACRE D_R receipt contract changed")
        probe = json.loads(self.probe_json)
        if not isinstance(probe, dict):
            raise ValueError("PACRE D_R probe JSON must be an object")

    @property
    def gate_passed(self) -> bool:
        return bool(self.checks) and all(
            value for _, value in self.checks
        )

    @property
    def decision(self) -> str:
        return (
            PACRE_DR_PASS_DECISION
            if self.gate_passed
            else PACRE_DR_FAIL_DECISION
        )

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(
            name for name, passed in self.checks if not passed
        )

    @property
    def probe(self) -> dict[str, object]:
        value = json.loads(self.probe_json)
        if not isinstance(value, dict):
            raise AssertionError("PACRE D_R probe changed")
        return value

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PACRE_DR_GATE_SCHEMA,
            "candidate": "PACRE-v22",
            "field_policy": CSLF_PACRE_FIELD_POLICY,
            "equation_policy": CSLF_PACRE_EQUATION_POLICY,
            "centering_policy": CSLF_PACRE_CENTERING_POLICY,
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "real_inputs_fingerprint": self.real_inputs_fingerprint,
            "population_fingerprint": self.population_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "probe": self.probe,
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "gate_passed": self.gate_passed,
            "decision": self.decision,
            "identifiability_only": True,
            "performance_gate_present": False,
            "AUC_gate_present": False,
            "bounded_400_authorized_by_this_receipt_alone": False,
            "training_performed": False,
            "D_R_accessed": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }

    @cached_property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_sources_unchanged(self) -> None:
        if self.implementation_binding != _implementation_binding():
            raise RuntimeError("PACRE D_R implementation changed")

    def verify_unchanged(
        self,
        *,
        dataset_free_receipt: Mapping[str, object],
        real_inputs: CoverageStateRealDRInputs,
        bounded_population: CoverageStateBoundedPopulation,
    ) -> None:
        """Revalidate the receipt against its complete live input graph."""

        self.verify_sources_unchanged()
        dataset_fingerprint = _validate_dataset_free_receipt(
            dataset_free_receipt
        )
        real_inputs.verify_unchanged()
        bounded_population.verify_unchanged()
        if (
            dataset_fingerprint
            != self.dataset_free_receipt_fingerprint
            or real_inputs.build_fingerprint
            != self.real_inputs_fingerprint
            or bounded_population.population_fingerprint
            != self.population_fingerprint
            or bounded_population.cache.cache_fingerprint
            != self.cache_fingerprint
            or bounded_population.source_cache
            is not real_inputs.scalar_cache
        ):
            raise RuntimeError("PACRE D_R receipt input binding changed")
        recomputed = recompute_pacre_dr_checks(
            dataset_free_receipt_fingerprint=dataset_fingerprint,
            real_inputs=real_inputs,
            bounded_population=bounded_population,
            probe=self.probe,
        )
        if recomputed != self.checks:
            raise RuntimeError("PACRE D_R receipt checks changed")


def run_pacre_dr_gate(
    *,
    dataset_free_receipt: Mapping[str, object],
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
    device: torch.device | str = "cpu",
) -> CoverageStatePACREDRGateReceipt:
    """Run the exact PACRE structural checks on the fixed bounded ``D_R``."""

    dataset_fingerprint = _validate_dataset_free_receipt(
        dataset_free_receipt
    )
    if not isinstance(real_inputs, CoverageStateRealDRInputs):
        raise TypeError("real_inputs must be CoverageStateRealDRInputs")
    if not isinstance(
        bounded_population,
        CoverageStateBoundedPopulation,
    ):
        raise TypeError(
            "bounded_population must be CoverageStateBoundedPopulation"
        )
    real_inputs.verify_unchanged()
    bounded_population.verify_unchanged()
    if (
        bounded_population.source_cache is not real_inputs.scalar_cache
        or bounded_population.seed != COVERAGE_STATE_BOUNDED_SEED
    ):
        raise PermissionError("PACRE D_R input bindings differ")
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("PACRE D_R gate supports CPU or CUDA")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("requested CUDA device is unavailable")
        if resolved.index is None:
            resolved = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )
        if (
            resolved.index is None
            or resolved.index < 0
            or resolved.index >= torch.cuda.device_count()
        ):
            raise ValueError("requested CUDA device is unavailable")

    probe = _probe(bounded_population, device=resolved)
    checks = recompute_pacre_dr_checks(
        dataset_free_receipt_fingerprint=dataset_fingerprint,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
        probe=probe,
    )
    receipt = CoverageStatePACREDRGateReceipt(
        dataset_free_receipt_fingerprint=dataset_fingerprint,
        real_inputs_fingerprint=real_inputs.build_fingerprint,
        population_fingerprint=(
            bounded_population.population_fingerprint
        ),
        cache_fingerprint=(
            bounded_population.cache.cache_fingerprint
        ),
        implementation_binding=_implementation_binding(),
        probe_json=canonical_json(probe),
        checks=checks,
    )
    receipt.verify_unchanged(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
    )
    return receipt


__all__ = [
    "PACRE_DR_CHECK_NAMES",
    "PACRE_DR_EXECUTION_SEED",
    "PACRE_DR_FAIL_DECISION",
    "PACRE_DR_GATE_SCHEMA",
    "PACRE_DR_PASS_DECISION",
    "PACRE_DR_SEPARATION_THRESHOLD",
    "CoverageStatePACREDRGateReceipt",
    "recompute_pacre_dr_checks",
    "run_pacre_dr_gate",
]
