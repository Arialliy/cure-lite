"""Generated-only gate for the v18 PMOPE objective.

The gate binds the Paired Minimum-Margin Orthant Projection Energy (PMOPE)
to the unchanged v17 CMIF completion field.  It uses generated tensors only:
no split, cache, detector output, optimizer step, calibration, or training
entry point is reachable from this module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
from math import sqrt
from pathlib import Path

import torch
from torch import Tensor

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from ..coverage_state_level_set import CSLF_FIELD_AMPLITUDE
from ..coverage_state_sobolev import (
    CSLF_PMOPE_POLICY,
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    _pair_energy,
    coverage_state_absolute_sobolev_loss_from_targets,
    coverage_state_identity_joint_loss_from_targets,
    coverage_state_pmope_pair_loss_from_targets,
    coverage_state_support_oriented_pair_sobolev_loss_from_targets,
    prepare_coverage_state_focused_absolute_targets,
    prepare_coverage_state_pair_targets,
)
from ..paired_types import tensor_content_fingerprint


COVERAGE_STATE_PMOPE_DATASET_FREE_SCHEMA = (
    "cure-lite-pmope-v18-dataset-free-receipt-v1"
)
COVERAGE_STATE_PMOPE_DATASET_FREE_EXECUTION_SEED = 180018
COVERAGE_STATE_PMOPE_POLICY = (
    "paired_minimum_sdf_margin_target_orthant_projection_"
    "joint_w1p4_energy_v1"
)
COVERAGE_STATE_PMOPE_TRUNCATION_RADIUS = 4
COVERAGE_STATE_PMOPE_MARGIN = (
    CSLF_FIELD_AMPLITUDE / COVERAGE_STATE_PMOPE_TRUNCATION_RADIUS
)
COVERAGE_STATE_PMOPE_FORMAL_FEATURE_CHANNELS = 64
COVERAGE_STATE_PMOPE_FORMAL_FEATURE_STRIDE = 4
COVERAGE_STATE_PMOPE_FORMAL_WIDTH = 32
COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT = 64064
COVERAGE_STATE_PMOPE_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/coverage_state_centered_mixed_interaction.py",
    "cure_lite/experiment/coverage_state_pmope_dataset_free.py",
)


def _current_implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_PMOPE_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"PMOPE implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


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
        truncation_radius=COVERAGE_STATE_PMOPE_TRUNCATION_RADIUS
    )


def _prepare_pair(
    *,
    size: int,
    occupancy_plus: Tensor,
    occupancy_minus: Tensor,
    target_plus: Tensor,
    target_minus: Tensor,
) -> CoverageStatePairTargets:
    return prepare_coverage_state_pair_targets(
        occupancy_plus,
        occupancy_minus,
        target_plus,
        target_minus,
        torch.ones(1, 1, size, size, dtype=torch.bool),
        config=_loss_config(),
    )


def _model_contract_probe() -> tuple[
    dict[str, object],
    tuple[str, ...],
    tuple[tuple[str, tuple[int, ...], str, bool], ...],
]:
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=COVERAGE_STATE_PMOPE_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_PMOPE_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_PMOPE_FORMAL_WIDTH,
    )
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    parameter_names = tuple(sorted(dict(model.named_parameters())))
    parameter_contract = tuple(
        (
            name,
            tuple(parameter.shape),
            str(parameter.dtype),
            bool(parameter.requires_grad),
        )
        for name, parameter in sorted(model.named_parameters())
    )
    state_before = stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(model.state_dict().items())
        }
    )
    generator = torch.Generator().manual_seed(
        COVERAGE_STATE_PMOPE_DATASET_FREE_EXECUTION_SEED + 1
    )
    feature = torch.randn(
        1,
        config.feature_channels,
        2,
        3,
        generator=generator,
    )
    occupancy = (
        torch.rand(
            1,
            1,
            2 * config.feature_stride,
            3 * config.feature_stride,
            generator=generator,
        )
        > 0.6
    )
    output = model(feature, occupancy)
    state_after = stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(model.state_dict().items())
        }
    )
    forward_parameters = tuple(
        inspect.signature(model.forward).parameters
    )
    payload = {
        "model_class": model.__class__.__name__,
        "feature_channels": config.feature_channels,
        "feature_stride": config.feature_stride,
        "width": config.width,
        "expected_parameter_count": config.expected_parameter_count,
        "actual_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "parameter_tensor_count": len(parameter_names),
        "buffer_names": [
            name for name, _ in model.named_buffers()
        ],
        "forward_parameters": list(forward_parameters),
        "input_feature_shape": list(feature.shape),
        "input_occupancy_shape": list(occupancy.shape),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_is_single_tensor": isinstance(output, Tensor),
        "output_finite": bool(torch.isfinite(output).all()),
        "scalar_output_fields": 1,
        "auxiliary_outputs": 0,
        "state_before": state_before,
        "state_after": state_after,
        "state_unchanged": state_before == state_after,
    }
    return payload, parameter_names, parameter_contract


def _clean_gradient_probe() -> dict[str, object]:
    size = 11
    added = ((5, 5), (5, 6), (6, 5), (6, 6))
    occupancy_plus = _mask(size, added)
    occupancy_minus = _mask(size)
    target_plus = _mask(size)
    target_minus = _mask(size, added)
    targets = _prepare_pair(
        size=size,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        target_plus=target_plus,
        target_minus=target_minus,
    )
    field_plus = torch.full_like(
        targets.target_field_plus,
        CSLF_FIELD_AMPLITUDE,
        requires_grad=True,
    )
    field_minus = torch.full_like(
        targets.target_field_minus,
        CSLF_FIELD_AMPLITUDE,
        requires_grad=True,
    )
    fields = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_loss_config(),
    )
    gradient_plus, gradient_minus = torch.autograd.grad(
        fields.loss,
        (field_plus, field_minus),
    )
    added_mask = target_minus
    outside = targets.valid_mask & ~added_mask
    return {
        "policy": CSLF_PMOPE_POLICY,
        "configured_margin_hex": COVERAGE_STATE_PMOPE_MARGIN.hex(),
        "tensor_margin_hex": float(fields.margin.item()).hex(),
        "valid_pixel_count": int(targets.valid_mask.sum().item()),
        "added_pixel_count": int(added_mask.sum().item()),
        "loss_positive": bool(fields.loss > 0.0),
        "loss_finite": bool(torch.isfinite(fields.loss)),
        "plus_violation_exact_zero": not bool(
            torch.any(fields.violation_plus)
        ),
        "minus_violation_positive_on_added": bool(
            torch.all(fields.violation_minus[added_mask] > 0.0)
        ),
        "minus_violation_exact_zero_outside": not bool(
            torch.any(fields.violation_minus[outside])
        ),
        "plus_gradient_exact_zero": not bool(
            torch.any(gradient_plus)
        ),
        "minus_gradient_positive_on_added": bool(
            torch.all(gradient_minus[added_mask] > 0.0)
        ),
        "minus_descent_direction_negative_on_added": bool(
            torch.all(-gradient_minus[added_mask] < 0.0)
        ),
        "minus_gradient_exact_zero_outside": not bool(
            torch.any(gradient_minus[outside])
        ),
        "gradient_finite": bool(
            torch.isfinite(gradient_plus).all()
            and torch.isfinite(gradient_minus).all()
        ),
        "violation_plus_sha256": tensor_content_fingerprint(
            fields.violation_plus
        ),
        "violation_minus_sha256": tensor_content_fingerprint(
            fields.violation_minus
        ),
        "gradient_plus_sha256": tensor_content_fingerprint(
            gradient_plus
        ),
        "gradient_minus_sha256": tensor_content_fingerprint(
            gradient_minus
        ),
    }


def _component_null_gradient_probe() -> dict[str, object]:
    size = 11
    removed = ((4, 5), (5, 5), (6, 5))
    occupancy_plus = _mask(size, removed)
    occupancy_minus = _mask(size)
    empty_target = _mask(size)
    targets = _prepare_pair(
        size=size,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        target_plus=empty_target,
        target_minus=empty_target,
    )
    field_plus = torch.full_like(
        targets.target_field_plus,
        CSLF_FIELD_AMPLITUDE,
        requires_grad=True,
    )
    initial_minus = torch.full_like(
        targets.target_field_minus,
        CSLF_FIELD_AMPLITUDE,
    )
    initial_minus[occupancy_plus] = -CSLF_FIELD_AMPLITUDE
    field_minus = initial_minus.detach().requires_grad_(True)
    fields = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_loss_config(),
    )
    gradient_plus, gradient_minus = torch.autograd.grad(
        fields.loss,
        (field_plus, field_minus),
    )
    outside = targets.valid_mask & ~occupancy_plus
    return {
        "target_plus_is_all_exterior": bool(
            torch.all(targets.target_field_plus > 0.0)
        ),
        "target_minus_is_all_exterior": bool(
            torch.all(targets.target_field_minus > 0.0)
        ),
        "removed_pixel_count": int(occupancy_plus.sum().item()),
        "loss_positive": bool(fields.loss > 0.0),
        "plus_violation_exact_zero": not bool(
            torch.any(fields.violation_plus)
        ),
        "minus_violation_positive_on_removed": bool(
            torch.all(fields.violation_minus[occupancy_plus] > 0.0)
        ),
        "minus_violation_exact_zero_outside": not bool(
            torch.any(fields.violation_minus[outside])
        ),
        "plus_gradient_exact_zero": not bool(
            torch.any(gradient_plus)
        ),
        "minus_gradient_negative_on_removed": bool(
            torch.all(gradient_minus[occupancy_plus] < 0.0)
        ),
        "minus_descent_direction_positive_on_removed": bool(
            torch.all(-gradient_minus[occupancy_plus] > 0.0)
        ),
        "minus_gradient_exact_zero_outside": not bool(
            torch.any(gradient_minus[outside])
        ),
        "gradient_finite": bool(
            torch.isfinite(gradient_plus).all()
            and torch.isfinite(gradient_minus).all()
        ),
        "gradient_minus_sha256": tensor_content_fingerprint(
            gradient_minus
        ),
    }


def _zero_semantics_probe() -> dict[str, object]:
    size = 11
    retained = ((2, 2), (2, 3))
    added = ((6, 5), (6, 6), (7, 5), (7, 6))
    occupancy_plus = _mask(size, added)
    occupancy_minus = _mask(size)
    target_plus = _mask(size, retained)
    target_minus = target_plus | _mask(size, added)
    targets = _prepare_pair(
        size=size,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        target_plus=target_plus,
        target_minus=target_minus,
    )
    margin = COVERAGE_STATE_PMOPE_MARGIN
    field_plus = torch.sign(targets.target_field_plus) * margin
    field_minus = torch.sign(targets.target_field_minus) * margin
    fields = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_loss_config(),
    )
    valid = targets.valid_mask
    raw_plus = (field_plus < 0.0) & valid
    raw_minus = (field_minus < 0.0) & valid
    target_raw_plus = (targets.target_field_plus < 0.0) & valid
    target_raw_minus = (targets.target_field_minus < 0.0) & valid
    completion_plus = raw_plus & ~occupancy_plus
    completion_minus = raw_minus & ~occupancy_minus
    violating = field_plus.clone()
    witness = valid & ~target_raw_plus
    witness_index = torch.nonzero(witness, as_tuple=False)[0]
    violating[tuple(witness_index)] = 0.0
    violated = coverage_state_pmope_pair_loss_from_targets(
        violating,
        field_minus,
        targets,
        config=_loss_config(),
    )
    return {
        "valid_pixel_count": int(valid.sum().item()),
        "zero_loss_exact": float(fields.loss.item()) == 0.0,
        "violation_plus_exact_zero": not bool(
            torch.any(fields.violation_plus)
        ),
        "violation_minus_exact_zero": not bool(
            torch.any(fields.violation_minus)
        ),
        "raw_plus_sign_set_exact": torch.equal(
            raw_plus,
            target_raw_plus,
        ),
        "raw_minus_sign_set_exact": torch.equal(
            raw_minus,
            target_raw_minus,
        ),
        "completion_plus_exact": torch.equal(
            completion_plus,
            target_plus,
        ),
        "completion_minus_exact": torch.equal(
            completion_minus,
            target_minus,
        ),
        "targets_disjoint_from_endpoint_occupancy": bool(
            not torch.any(target_plus & occupancy_plus)
            and not torch.any(target_minus & occupancy_minus)
        ),
        "one_valid_zero_margin_violation_positive": bool(
            violated.loss > 0.0
        ),
        "one_valid_zero_margin_violation_count": int(
            torch.count_nonzero(violated.violation_plus).item()
        ),
        "full_valid_domain_used": torch.equal(
            fields.valid_mask,
            valid,
        ),
        "raw_plus_sha256": tensor_content_fingerprint(raw_plus),
        "raw_minus_sha256": tensor_content_fingerprint(raw_minus),
        "completion_plus_sha256": tensor_content_fingerprint(
            completion_plus
        ),
        "completion_minus_sha256": tensor_content_fingerprint(
            completion_minus
        ),
    }


def _non_equivalence_probe() -> dict[str, object]:
    size = 11
    retained = ((2, 2),)
    added = ((6, 5), (6, 6))
    occupancy_plus = _mask(size, added)
    occupancy_minus = _mask(size)
    target_plus = _mask(size, retained)
    target_minus = target_plus | _mask(size, added)
    targets = _prepare_pair(
        size=size,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        target_plus=target_plus,
        target_minus=target_minus,
    )
    deep_plus = (
        torch.sign(targets.target_field_plus) * CSLF_FIELD_AMPLITUDE
    )
    deep_minus = (
        torch.sign(targets.target_field_minus) * CSLF_FIELD_AMPLITUDE
    )
    pmope = coverage_state_pmope_pair_loss_from_targets(
        deep_plus,
        deep_minus,
        targets,
        config=_loss_config(),
    )
    identity = coverage_state_identity_joint_loss_from_targets(
        deep_plus,
        deep_minus,
        targets,
        config=_loss_config(),
    )
    sorr = coverage_state_support_oriented_pair_sobolev_loss_from_targets(
        deep_plus,
        deep_minus,
        targets,
        config=_loss_config(),
    )
    absolute_plus = prepare_coverage_state_focused_absolute_targets(
        target_plus,
        targets.valid_mask,
        targets.valid_mask & ~occupancy_plus,
        config=_loss_config(),
    )
    absolute_minus = prepare_coverage_state_focused_absolute_targets(
        target_minus,
        targets.valid_mask,
        targets.valid_mask & ~occupancy_minus,
        config=_loss_config(),
    )
    separable_plus = coverage_state_absolute_sobolev_loss_from_targets(
        deep_plus,
        absolute_plus,
        config=_loss_config(),
    )
    separable_minus = coverage_state_absolute_sobolev_loss_from_targets(
        deep_minus,
        absolute_minus,
        config=_loss_config(),
    )
    separable_loss = 0.5 * (
        separable_plus.loss + separable_minus.loss
    )
    error_plus = deep_plus - targets.target_field_plus
    error_minus = deep_minus - targets.target_field_minus
    orthogonal_scale = sqrt(2.0)
    omco_first = (error_plus + error_minus) / orthogonal_scale
    omco_second = (error_minus - error_plus) / orthogonal_scale
    omco_loss = _pair_energy(
        (omco_first, omco_second),
        targets,
        config=_loss_config(),
    )[0]
    boundary_magnitude_differs = bool(
        torch.any(
            targets.target_field_plus.abs()
            < CSLF_FIELD_AMPLITUDE
        )
        or torch.any(
            targets.target_field_minus.abs()
            < CSLF_FIELD_AMPLITUDE
        )
    )
    return {
        "deep_correct_sign_field": bool(
            torch.equal(
                deep_plus < 0.0,
                targets.target_field_plus < 0.0,
            )
            and torch.equal(
                deep_minus < 0.0,
                targets.target_field_minus < 0.0,
            )
        ),
        "pmope_loss_hex": float(pmope.loss.item()).hex(),
        "pmope_loss_exact_zero": float(pmope.loss.item()) == 0.0,
        "identity_loss_hex": float(identity.loss.item()).hex(),
        "identity_loss_positive": bool(identity.loss > 0.0),
        "sorr_loss_hex": float(sorr.loss.item()).hex(),
        "sorr_loss_positive": bool(sorr.loss > 0.0),
        "separable_plus_loss_hex": float(
            separable_plus.loss.item()
        ).hex(),
        "separable_minus_loss_hex": float(
            separable_minus.loss.item()
        ).hex(),
        "separable_loss_hex": float(separable_loss.item()).hex(),
        "separable_loss_positive": bool(separable_loss > 0.0),
        "omco_coordinate_policy": (
            "fixed_orthogonal_sum_difference_of_endpoint_errors_v1"
        ),
        "omco_loss_hex": float(omco_loss.item()).hex(),
        "omco_loss_positive": bool(omco_loss > 0.0),
        "all_four_old_objective_losses_positive": all(
            bool(value > 0.0)
            for value in (
                identity.loss,
                sorr.loss,
                separable_loss,
                omco_loss,
            )
        ),
        "target_has_nonendpoint_sdf_magnitudes": (
            boundary_magnitude_differs
        ),
        "one_sided_feasible_cone_witness": bool(
            pmope.loss == 0.0 and identity.loss > 0.0
        ),
        "deep_plus_sha256": tensor_content_fingerprint(deep_plus),
        "deep_minus_sha256": tensor_content_fingerprint(deep_minus),
    }


def _cmif_parameter_gradient_probe() -> dict[str, object]:
    """Prove PMOPE reaches every CMIF parameter without making an update."""

    model_config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    model = CURELiteCenteredMixedInteractionLevelSet(model_config)
    with torch.no_grad():
        model.scalar_energy_weight.copy_(
            torch.tensor((0.20, -0.15, 0.10, -0.05))
        )
        model.joint_hidden_bias.copy_(
            torch.tensor((0.03, -0.02, 0.01, -0.04))
        )
    state_before = stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(model.state_dict().items())
        }
    )
    generator = torch.Generator().manual_seed(
        COVERAGE_STATE_PMOPE_DATASET_FREE_EXECUTION_SEED + 2
    )
    feature = torch.randn(
        1,
        model_config.feature_channels,
        6,
        6,
        generator=generator,
    )
    occupancy_plus = torch.zeros(
        1,
        1,
        12,
        12,
        dtype=torch.bool,
    )
    occupancy_plus[..., 5:7, 5:7] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)
    target_plus = torch.zeros_like(occupancy_plus)
    target_minus = occupancy_plus.clone()
    targets = _prepare_pair(
        size=12,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        target_plus=target_plus,
        target_minus=target_minus,
    )
    field_plus = model(feature, occupancy_plus)
    field_minus = model(feature, occupancy_minus)
    fields = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_loss_config(),
    )
    named_parameters = tuple(model.named_parameters())
    gradients = torch.autograd.grad(
        fields.loss,
        tuple(parameter for _, parameter in named_parameters),
    )
    gradient_contract = {
        name: {
            "shape": list(gradient.shape),
            "dtype": str(gradient.dtype),
            "finite": bool(torch.isfinite(gradient).all()),
            "nonzero": bool(torch.any(gradient != 0.0)),
            "l2_norm_hex": float(
                torch.linalg.vector_norm(gradient).item()
            ).hex(),
            "maximum_absolute_hex": float(
                gradient.abs().max().item()
            ).hex(),
            "sha256": tensor_content_fingerprint(gradient),
        }
        for (name, _), gradient in zip(
            named_parameters,
            gradients,
            strict=True,
        )
    }
    state_after = stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(model.state_dict().items())
        }
    )
    return {
        "probe_kind": "generated_clean_pair_autograd_only",
        "known_initial_multiplicative_latency_avoided": True,
        "scalar_energy_weight_fixed_nonzero": bool(
            torch.all(model.scalar_energy_weight != 0.0)
        ),
        "pair_loss_positive": bool(fields.loss > 0.0),
        "pair_loss_finite": bool(torch.isfinite(fields.loss)),
        "plus_field_shape": list(field_plus.shape),
        "minus_field_shape": list(field_minus.shape),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "expected_parameter_count": model_config.expected_parameter_count,
        "parameter_names": [
            name for name, _ in named_parameters
        ],
        "gradient_contract": gradient_contract,
        "all_parameter_gradients_finite": all(
            bool(value["finite"])
            for value in gradient_contract.values()
        ),
        "all_parameter_gradients_nonzero": all(
            bool(value["nonzero"])
            for value in gradient_contract.values()
        ),
        "state_before": state_before,
        "state_after": state_after,
        "model_state_unchanged": state_before == state_after,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "feature_sha256": tensor_content_fingerprint(feature),
        "occupancy_plus_sha256": tensor_content_fingerprint(
            occupancy_plus
        ),
        "occupancy_minus_sha256": tensor_content_fingerprint(
            occupancy_minus
        ),
        "field_plus_sha256": tensor_content_fingerprint(field_plus),
        "field_minus_sha256": tensor_content_fingerprint(field_minus),
    }


def _generated_payload(
    *,
    formal_config_payload: dict[str, object],
    parameter_names: tuple[str, ...],
    parameter_contract: tuple[
        tuple[str, tuple[int, ...], str, bool],
        ...,
    ],
    clean_probe: dict[str, object],
    component_null_probe: dict[str, object],
    zero_semantics_probe: dict[str, object],
    non_equivalence_probe: dict[str, object],
    cmif_parameter_gradient_probe: dict[str, object],
) -> dict[str, object]:
    return {
        "formal_config": deepcopy(formal_config_payload),
        "parameter_names": list(parameter_names),
        "parameter_contract": [
            [name, list(shape), dtype, requires_grad]
            for name, shape, dtype, requires_grad in parameter_contract
        ],
        "clean_probe": deepcopy(clean_probe),
        "component_null_probe": deepcopy(component_null_probe),
        "zero_semantics_probe": deepcopy(zero_semantics_probe),
        "non_equivalence_probe": deepcopy(non_equivalence_probe),
        "cmif_parameter_gradient_probe": deepcopy(
            cmif_parameter_gradient_probe
        ),
    }


def _collect_generated_evidence() -> dict[str, object]:
    torch.manual_seed(COVERAGE_STATE_PMOPE_DATASET_FREE_EXECUTION_SEED)
    (
        formal_config_payload,
        parameter_names,
        parameter_contract,
    ) = _model_contract_probe()
    clean_probe = _clean_gradient_probe()
    component_null_probe = _component_null_gradient_probe()
    zero_semantics_probe = _zero_semantics_probe()
    non_equivalence_probe = _non_equivalence_probe()
    cmif_parameter_gradient_probe = _cmif_parameter_gradient_probe()
    return {
        "formal_config_payload": formal_config_payload,
        "parameter_names": parameter_names,
        "parameter_contract": parameter_contract,
        "clean_probe": clean_probe,
        "component_null_probe": component_null_probe,
        "zero_semantics_probe": zero_semantics_probe,
        "non_equivalence_probe": non_equivalence_probe,
        "cmif_parameter_gradient_probe": (
            cmif_parameter_gradient_probe
        ),
    }


def recompute_coverage_state_pmope_dataset_free_checks(
    *,
    formal_config_payload: dict[str, object],
    parameter_names: tuple[str, ...],
    parameter_contract: tuple[
        tuple[str, tuple[int, ...], str, bool],
        ...,
    ],
    implementation_binding: tuple[tuple[str, str], ...],
    clean_probe: dict[str, object],
    component_null_probe: dict[str, object],
    zero_semantics_probe: dict[str, object],
    non_equivalence_probe: dict[str, object],
    cmif_parameter_gradient_probe: dict[str, object],
    generated_replay_fingerprint: str,
) -> tuple[tuple[str, bool], ...]:
    """Recompute every v18 generated-only gate bit."""

    expected_names = (
        "joint_hidden_bias",
        "joint_state_weight",
        "scalar_energy_weight",
    )
    expected_contract = (
        ("joint_hidden_bias", (32,), "torch.float32", True),
        (
            "joint_state_weight",
            (32, 80, 5, 5),
            "torch.float32",
            True,
        ),
        ("scalar_energy_weight", (32,), "torch.float32", True),
    )
    expected_tensor_margin = float(
        torch.tensor(COVERAGE_STATE_PMOPE_MARGIN, dtype=torch.float32)
        .item()
    ).hex()
    expected_generated_fingerprint = stable_fingerprint(
        _generated_payload(
            formal_config_payload=formal_config_payload,
            parameter_names=parameter_names,
            parameter_contract=parameter_contract,
            clean_probe=clean_probe,
            component_null_probe=component_null_probe,
            zero_semantics_probe=zero_semantics_probe,
            non_equivalence_probe=non_equivalence_probe,
            cmif_parameter_gradient_probe=(
                cmif_parameter_gradient_probe
            ),
        )
    )
    binding_paths = tuple(path for path, _ in implementation_binding)
    checks = {
        "policy_exact": (
            CSLF_PMOPE_POLICY == COVERAGE_STATE_PMOPE_POLICY
            and clean_probe["policy"] == COVERAGE_STATE_PMOPE_POLICY
        ),
        "fixed_margin_exact": (
            COVERAGE_STATE_PMOPE_MARGIN == 0.225
            and clean_probe["configured_margin_hex"]
            == float(0.225).hex()
            and clean_probe["tensor_margin_hex"]
            == expected_tensor_margin
        ),
        "cmif_parameter_interface_unchanged": (
            formal_config_payload["model_class"]
            == CURELiteCenteredMixedInteractionLevelSet.__name__
            and formal_config_payload["feature_channels"]
            == COVERAGE_STATE_PMOPE_FORMAL_FEATURE_CHANNELS
            and formal_config_payload["feature_stride"]
            == COVERAGE_STATE_PMOPE_FORMAL_FEATURE_STRIDE
            and formal_config_payload["width"]
            == COVERAGE_STATE_PMOPE_FORMAL_WIDTH
            and formal_config_payload["expected_parameter_count"]
            == COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT
            and formal_config_payload["actual_parameter_count"]
            == COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT
            and formal_config_payload["parameter_tensor_count"] == 3
            and formal_config_payload["buffer_names"] == []
            and formal_config_payload["forward_parameters"]
            == ["feature", "occupancy"]
            and formal_config_payload["output_shape"]
            == [1, 1, 8, 12]
            and formal_config_payload["output_dtype"]
            == "torch.float32"
            and formal_config_payload["output_is_single_tensor"]
            and formal_config_payload["output_finite"]
            and formal_config_payload["scalar_output_fields"] == 1
            and formal_config_payload["auxiliary_outputs"] == 0
            and formal_config_payload["state_unchanged"]
            and parameter_names == expected_names
            and parameter_contract == expected_contract
        ),
        "clean_plus_zero_minus_negative_descent": all(
            bool(clean_probe[name])
            for name in (
                "loss_positive",
                "loss_finite",
                "plus_violation_exact_zero",
                "minus_violation_positive_on_added",
                "minus_violation_exact_zero_outside",
                "plus_gradient_exact_zero",
                "minus_gradient_positive_on_added",
                "minus_descent_direction_negative_on_added",
                "minus_gradient_exact_zero_outside",
                "gradient_finite",
            )
        ),
        "component_null_positive_descent": all(
            bool(component_null_probe[name])
            for name in (
                "target_plus_is_all_exterior",
                "target_minus_is_all_exterior",
                "loss_positive",
                "plus_violation_exact_zero",
                "minus_violation_positive_on_removed",
                "minus_violation_exact_zero_outside",
                "plus_gradient_exact_zero",
                "minus_gradient_negative_on_removed",
                "minus_descent_direction_positive_on_removed",
                "minus_gradient_exact_zero_outside",
                "gradient_finite",
            )
        ),
        "zero_loss_raw_sign_set_full_valid": all(
            bool(zero_semantics_probe[name])
            for name in (
                "zero_loss_exact",
                "violation_plus_exact_zero",
                "violation_minus_exact_zero",
                "raw_plus_sign_set_exact",
                "raw_minus_sign_set_exact",
                "full_valid_domain_used",
                "one_valid_zero_margin_violation_positive",
            )
        ),
        "zero_loss_completion_equivalence": all(
            bool(zero_semantics_probe[name])
            for name in (
                "completion_plus_exact",
                "completion_minus_exact",
                "targets_disjoint_from_endpoint_occupancy",
            )
        ),
        "old_identity_objective_not_equivalent": all(
            bool(non_equivalence_probe[name])
            for name in (
                "deep_correct_sign_field",
                "pmope_loss_exact_zero",
                "identity_loss_positive",
                "target_has_nonendpoint_sdf_magnitudes",
                "one_sided_feasible_cone_witness",
            )
        ),
        "identity_sorr_separable_omco_not_equivalent": (
            non_equivalence_probe["deep_correct_sign_field"]
            and non_equivalence_probe["pmope_loss_exact_zero"]
            and non_equivalence_probe["identity_loss_positive"]
            and non_equivalence_probe["sorr_loss_positive"]
            and non_equivalence_probe["separable_loss_positive"]
            and non_equivalence_probe["omco_loss_positive"]
            and non_equivalence_probe[
                "all_four_old_objective_losses_positive"
            ]
            and non_equivalence_probe[
                "omco_coordinate_policy"
            ]
            == "fixed_orthogonal_sum_difference_of_endpoint_errors_v1"
            and non_equivalence_probe[
                "target_has_nonendpoint_sdf_magnitudes"
            ]
            and non_equivalence_probe[
                "one_sided_feasible_cone_witness"
            ]
        ),
        "pmope_reaches_all_cmif_parameters": (
            cmif_parameter_gradient_probe["probe_kind"]
            == "generated_clean_pair_autograd_only"
            and cmif_parameter_gradient_probe[
                "known_initial_multiplicative_latency_avoided"
            ]
            and cmif_parameter_gradient_probe[
                "scalar_energy_weight_fixed_nonzero"
            ]
            and cmif_parameter_gradient_probe["pair_loss_positive"]
            and cmif_parameter_gradient_probe["pair_loss_finite"]
            and cmif_parameter_gradient_probe["plus_field_shape"]
            == [1, 1, 12, 12]
            and cmif_parameter_gradient_probe["minus_field_shape"]
            == [1, 1, 12, 12]
            and cmif_parameter_gradient_probe["parameter_count"]
            == cmif_parameter_gradient_probe[
                "expected_parameter_count"
            ]
            and cmif_parameter_gradient_probe["parameter_names"]
            == [
                "joint_state_weight",
                "joint_hidden_bias",
                "scalar_energy_weight",
            ]
            and set(
                cmif_parameter_gradient_probe[
                    "gradient_contract"
                ]
            )
            == {
                "joint_state_weight",
                "joint_hidden_bias",
                "scalar_energy_weight",
            }
            and cmif_parameter_gradient_probe[
                "all_parameter_gradients_finite"
            ]
            and cmif_parameter_gradient_probe[
                "all_parameter_gradients_nonzero"
            ]
            and cmif_parameter_gradient_probe[
                "model_state_unchanged"
            ]
            and not cmif_parameter_gradient_probe[
                "optimizer_constructed"
            ]
            and cmif_parameter_gradient_probe["optimizer_steps"] == 0
        ),
        "implementation_binding_complete": (
            binding_paths == COVERAGE_STATE_PMOPE_IMPLEMENTATION_PATHS
            and all(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
                for _, digest in implementation_binding
            )
        ),
        "generated_replay_exact": (
            generated_replay_fingerprint
            == expected_generated_fingerprint
        ),
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True)
class CoverageStatePMOPEDatasetFreeReceipt:
    """Fingerprint-bound generated evidence for the v18 objective."""

    formal_config_payload: dict[str, object]
    parameter_names: tuple[str, ...]
    parameter_contract: tuple[
        tuple[str, tuple[int, ...], str, bool],
        ...,
    ]
    implementation_binding: tuple[tuple[str, str], ...]
    clean_probe: dict[str, object]
    component_null_probe: dict[str, object]
    zero_semantics_probe: dict[str, object]
    non_equivalence_probe: dict[str, object]
    cmif_parameter_gradient_probe: dict[str, object]
    generated_replay_fingerprint: str
    checks: tuple[tuple[str, bool], ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def _evidence_payload(self) -> dict[str, object]:
        payload = _generated_payload(
            formal_config_payload=self.formal_config_payload,
            parameter_names=self.parameter_names,
            parameter_contract=self.parameter_contract,
            clean_probe=self.clean_probe,
            component_null_probe=self.component_null_probe,
            zero_semantics_probe=self.zero_semantics_probe,
            non_equivalence_probe=self.non_equivalence_probe,
            cmif_parameter_gradient_probe=(
                self.cmif_parameter_gradient_probe
            ),
        )
        payload["implementation_binding"] = dict(
            self.implementation_binding
        )
        payload["generated_replay_fingerprint"] = (
            self.generated_replay_fingerprint
        )
        return payload

    def verify_unchanged(self) -> None:
        expected = recompute_coverage_state_pmope_dataset_free_checks(
            formal_config_payload=self.formal_config_payload,
            parameter_names=self.parameter_names,
            parameter_contract=self.parameter_contract,
            implementation_binding=self.implementation_binding,
            clean_probe=self.clean_probe,
            component_null_probe=self.component_null_probe,
            zero_semantics_probe=self.zero_semantics_probe,
            non_equivalence_probe=self.non_equivalence_probe,
            cmif_parameter_gradient_probe=(
                self.cmif_parameter_gradient_probe
            ),
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
                "PMOPE dataset-free evidence changed after creation"
            )

    @property
    def all_pass(self) -> bool:
        self.verify_unchanged()
        return bool(self.checks) and all(value for _, value in self.checks)

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_PMOPE_DATASET_FREE_SCHEMA,
            "objective_policy": COVERAGE_STATE_PMOPE_POLICY,
            "fixed_margin_hex": COVERAGE_STATE_PMOPE_MARGIN.hex(),
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
            "clean_probe": deepcopy(self.clean_probe),
            "component_null_probe": deepcopy(
                self.component_null_probe
            ),
            "zero_semantics_probe": deepcopy(
                self.zero_semantics_probe
            ),
            "non_equivalence_probe": deepcopy(
                self.non_equivalence_probe
            ),
            "cmif_parameter_gradient_probe": deepcopy(
                self.cmif_parameter_gradient_probe
            ),
            "generated_replay_fingerprint": (
                self.generated_replay_fingerprint
            ),
            "evidence_fingerprint": self.evidence_fingerprint,
            "checks": dict(self.checks),
            "all_pass": self.all_pass,
            "runtime_splits": [],
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "dataset_training_performed": False,
            "synthetic_gradient_probe_optimizer_steps": 0,
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _run_coverage_state_pmope_dataset_free_gate_inner(
) -> CoverageStatePMOPEDatasetFreeReceipt:
    first = _collect_generated_evidence()
    second = _collect_generated_evidence()
    first_generated_payload = _generated_payload(**first)
    second_generated_payload = _generated_payload(**second)
    first_fingerprint = stable_fingerprint(first_generated_payload)
    second_fingerprint = stable_fingerprint(second_generated_payload)
    if first_fingerprint != second_fingerprint:
        raise RuntimeError("PMOPE generated replay is not deterministic")

    implementation_binding = _current_implementation_binding()
    checks = recompute_coverage_state_pmope_dataset_free_checks(
        implementation_binding=implementation_binding,
        generated_replay_fingerprint=second_fingerprint,
        **first,
    )
    evidence_payload = deepcopy(first_generated_payload)
    evidence_payload["implementation_binding"] = dict(
        implementation_binding
    )
    evidence_payload["generated_replay_fingerprint"] = second_fingerprint
    return CoverageStatePMOPEDatasetFreeReceipt(
        implementation_binding=implementation_binding,
        generated_replay_fingerprint=second_fingerprint,
        checks=checks,
        evidence_fingerprint=stable_fingerprint(evidence_payload),
        **first,
    )


def run_coverage_state_pmope_dataset_free_gate(
) -> CoverageStatePMOPEDatasetFreeReceipt:
    """Run the complete generated-only PMOPE structural gate."""

    before_rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        result = _run_coverage_state_pmope_dataset_free_gate_inner()
    if not torch.equal(before_rng, torch.random.get_rng_state()):
        raise RuntimeError("PMOPE dataset-free gate changed global RNG state")
    return result


__all__ = [
    "COVERAGE_STATE_PMOPE_DATASET_FREE_SCHEMA",
    "COVERAGE_STATE_PMOPE_DATASET_FREE_EXECUTION_SEED",
    "COVERAGE_STATE_PMOPE_POLICY",
    "COVERAGE_STATE_PMOPE_TRUNCATION_RADIUS",
    "COVERAGE_STATE_PMOPE_MARGIN",
    "COVERAGE_STATE_PMOPE_FORMAL_FEATURE_CHANNELS",
    "COVERAGE_STATE_PMOPE_FORMAL_FEATURE_STRIDE",
    "COVERAGE_STATE_PMOPE_FORMAL_WIDTH",
    "COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT",
    "COVERAGE_STATE_PMOPE_IMPLEMENTATION_PATHS",
    "CoverageStatePMOPEDatasetFreeReceipt",
    "recompute_coverage_state_pmope_dataset_free_checks",
    "run_coverage_state_pmope_dataset_free_gate",
]
