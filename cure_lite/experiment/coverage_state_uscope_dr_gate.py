"""One-pass, read-only real-``D_R`` gate for USCOPE.

The gate evaluates every bounded clean and component-null pair in one CMIF
forward.  It audits the complete declared valid domain ``V``, the statewise
supremal certificate, and the differentiable path without constructing an
optimizer or mutating model/data state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import torch
from torch import Tensor

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from ..coverage_state_sobolev import CoverageStatePairTargets
from ..coverage_state_supremal_projection import (
    CSLF_USCOPE_POLICY,
    coverage_state_uscope_pair_loss_from_targets,
)
from ..paired_types import tensor_content_fingerprint
from .coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    COVERAGE_STATE_BOUNDED_SEED,
    CoverageStateBoundedPopulation,
)
from .coverage_state_cmif_dataset_free import (
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_CMIF_FORMAL_WIDTH,
)
from .coverage_state_real_dr_inputs import CoverageStateRealDRInputs
from .coverage_state_uscope_dataset_free import (
    COVERAGE_STATE_USCOPE_MARGIN,
    CoverageStateUSCOPEDatasetFreeReceipt,
)


COVERAGE_STATE_USCOPE_DR_GATE_SCHEMA = (
    "cure-lite-cmif-v19-uscope-real-dr-gate-v1"
)
COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED = 42
COVERAGE_STATE_USCOPE_DR_MASS_RTOL = 1.0e-5
COVERAGE_STATE_USCOPE_DR_MASS_ATOL = 1.0e-8
COVERAGE_STATE_USCOPE_DR_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_centered_mixed_interaction.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/coverage_state_supremal_projection.py",
    "cure_lite/experiment/coverage_state_uscope_dataset_free.py",
    "cure_lite/experiment/coverage_state_uscope_dr_gate.py",
)


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    rows: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_USCOPE_DR_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"USCOPE D_R implementation path is invalid: {relative}"
            )
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


def _model_fingerprint(
    model: CURELiteCenteredMixedInteractionLevelSet,
) -> str:
    return stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(model.state_dict().items())
        }
    )


def _cuda_rng_devices(device: torch.device) -> list[int]:
    """Return the integer CUDA IDs accepted by ``torch.random.fork_rng``."""

    if device.type != "cuda":
        return []
    if device.index is None:
        raise ValueError("CUDA device must have an explicit index")
    return [device.index]


def _stack_targets(
    values: tuple[CoverageStatePairTargets, ...],
    *,
    device: torch.device,
) -> CoverageStatePairTargets:
    if not values:
        raise ValueError("USCOPE D_R requires nonempty pair targets")

    def cat(name: str) -> Tensor:
        return torch.cat(
            tuple(getattr(value, name) for value in values),
            dim=0,
        ).to(device=device, non_blocking=False).contiguous()

    result = CoverageStatePairTargets(
        target_field_plus=cat("target_field_plus"),
        target_field_minus=cat("target_field_minus"),
        focus_support=cat("focus_support"),
        focus_support_field=cat("focus_support_field"),
        integration_measure=cat("integration_measure"),
        valid_mask=cat("valid_mask"),
    )
    result.validate()
    return result


def _geometry(
    population: CoverageStateBoundedPopulation,
) -> dict[str, object]:
    cache = population.cache
    cache.verify_unchanged()
    optimized = tuple(
        sorted(
            (
                *cache.clean_positive_records,
                *cache.component_null_records,
            ),
            key=lambda value: (
                value.optimizer_role,
                value.record.pair_id,
            ),
        )
    )
    valid_masks = tuple(
        value.joint_targets.valid_mask for value in optimized
    )
    clean = tuple(
        value
        for value in optimized
        if value.optimizer_role == "clean_positive"
    )
    component = tuple(
        value
        for value in optimized
        if value.optimizer_role == "component_null"
    )
    strict_target_sign = all(
        not bool(
            torch.any(
                target[target_geometry.valid_mask] == 0.0
            )
        )
        for value in optimized
        for target_geometry in (value.joint_targets,)
        for target in (
            target_geometry.target_field_plus,
            target_geometry.target_field_minus,
        )
    )
    clean_geometry_contract = all(
        value.record.pair_kind == "clean_positive"
        and not torch.equal(
            value.record.occupancy_plus,
            value.record.occupancy_minus,
        )
        and not torch.equal(
            value.joint_targets.target_field_plus,
            value.joint_targets.target_field_minus,
        )
        and bool(torch.any(value.joint_targets.focus_support))
        for value in clean
    )
    component_geometry_contract = all(
        value.record.pair_kind == "component_null"
        and not torch.equal(
            value.record.occupancy_plus,
            value.record.occupancy_minus,
        )
        and torch.equal(
            value.joint_targets.target_field_plus,
            value.joint_targets.target_field_minus,
        )
        for value in component
    )
    geometry_rows: list[dict[str, object]] = []
    target_and_occupancy_inside_valid = True
    joint_valid_matches_record = True
    integration_mass_one = True
    for value in optimized:
        targets = value.joint_targets
        targets.validate()
        record = value.record
        mass = targets.integration_measure.flatten(1).sum(dim=1)
        mass_one = bool(
            torch.allclose(
                mass,
                torch.ones_like(mass),
                rtol=COVERAGE_STATE_USCOPE_DR_MASS_RTOL,
                atol=COVERAGE_STATE_USCOPE_DR_MASS_ATOL,
            )
        )
        record_masks_inside = not bool(
            torch.any(
                (
                    record.target_plus
                    | record.target_minus
                    | record.occupancy_plus
                    | record.occupancy_minus
                )
                & ~record.valid_mask
            )
        )
        valid_matches = torch.equal(
            targets.valid_mask,
            record.valid_mask,
        )
        integration_mass_one = integration_mass_one and mass_one
        target_and_occupancy_inside_valid = (
            target_and_occupancy_inside_valid
            and record_masks_inside
        )
        joint_valid_matches_record = (
            joint_valid_matches_record and valid_matches
        )
        geometry_rows.append(
            {
                "pair_id": record.pair_id,
                "sample_id": record.sample_id,
                "optimizer_role": value.optimizer_role,
                "integration_mass_hex": [
                    _hex(
                        float(item),
                        name="USCOPE integration mass",
                    )
                    for item in mass.tolist()
                ],
                "integration_mass_one": mass_one,
                "pair_targets_validate_passed": True,
                "target_and_occupancy_inside_valid": (
                    record_masks_inside
                ),
                "joint_valid_matches_record": valid_matches,
            }
        )
    endpoint_count = 2 * len(optimized)
    aligned_feature_shapes = {
        tuple(value.record.feature.shape) for value in optimized
    }
    aligned_output_shapes = {
        tuple(value.record.occupancy_plus.shape) for value in optimized
    }
    if len(aligned_feature_shapes) != 1 or len(aligned_output_shapes) != 1:
        raise ValueError("USCOPE D_R pairs do not share one tensor shape")
    feature_shape = next(iter(aligned_feature_shapes))
    output_shape = next(iter(aligned_output_shapes))
    coarse_height, coarse_width = feature_shape[-2:]
    phase_channels = COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE**2
    dominant_elements = (
        endpoint_count
        * phase_channels
        * COVERAGE_STATE_CMIF_FORMAL_WIDTH
        * coarse_height
        * coarse_width
    )
    dominant_bytes = dominant_elements * torch.finfo(torch.float32).bits // 8
    output_matches_feature_stride = (
        output_shape[-2]
        == coarse_height * COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE
        and output_shape[-1]
        == coarse_width * COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE
    )
    return {
        "optimized_pair_count": len(optimized),
        "clean_positive_count": len(cache.clean_positive_records),
        "component_null_count": len(cache.component_null_records),
        "valid_domain_is_full_output": bool(valid_masks)
        and all(bool(mask.all()) for mask in valid_masks),
        "valid_domain_nonempty_per_state": bool(valid_masks)
        and all(bool(torch.any(mask)) for mask in valid_masks),
        "target_fields_strictly_nonzero_on_valid": strict_target_sign,
        "clean_geometry_contract": clean_geometry_contract,
        "component_geometry_contract": component_geometry_contract,
        "geometry_rows": geometry_rows,
        "integration_measure_mass_one": integration_mass_one,
        "integration_mass_rtol_hex": (
            COVERAGE_STATE_USCOPE_DR_MASS_RTOL.hex()
        ),
        "integration_mass_atol_hex": (
            COVERAGE_STATE_USCOPE_DR_MASS_ATOL.hex()
        ),
        "target_and_occupancy_inside_valid": (
            target_and_occupancy_inside_valid
        ),
        "joint_valid_matches_record": joint_valid_matches_record,
        "output_matches_feature_stride": output_matches_feature_stride,
        "optimized_pair_ids": [
            value.record.pair_id for value in optimized
        ],
        "endpoint_valid_pixel_count": sum(
            2 * int(torch.count_nonzero(mask)) for mask in valid_masks
        ),
        "memory_plan": {
            "pair_batch_count": len(optimized),
            "endpoint_batch_count": endpoint_count,
            "feature_shape_per_pair": list(feature_shape),
            "output_shape_per_endpoint": list(output_shape),
            "phase_channels": phase_channels,
            "hidden_width": COVERAGE_STATE_CMIF_FORMAL_WIDTH,
            "dominant_phase_width_tensor_bytes": dominant_bytes,
            # phase_delta, two midpoint affines, the neutralized hidden,
            # and mixed_hidden all have this dominant shape.  This is an
            # exact named-forward-tensor subtotal, not a peak-memory claim.
            "five_named_forward_tensors_bytes": 5 * dominant_bytes,
            "estimate_is_not_measured_peak": True,
            "execution_policy": (
                "one_batch_32_pairs_64_endpoints_one_model_forward"
            ),
        },
        "input_policy": "declared_valid_domain_two_endpoint_product",
        "same_sign_response_policy": "diagnostic_only",
    }


def _hex(value: float, *, name: str) -> str:
    number = float(value)
    if not isfinite(number):
        raise FloatingPointError(f"{name} is non-finite")
    return number.hex()


def _probe(
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    before_rng = torch.random.get_rng_state().clone()
    before_device_rng = (
        None
        if device.type != "cuda"
        else torch.cuda.get_rng_state(device).clone()
    )
    before_population = population.population_fingerprint
    runtime_splits = sorted(
        {
            population.source_cache.raw_catalog.split,
            population.cache.raw_catalog.split,
        }
    )
    optimized = tuple(
        sorted(
            (
                *population.cache.clean_positive_records,
                *population.cache.component_null_records,
            ),
            key=lambda value: (
                value.optimizer_role,
                value.record.pair_id,
            ),
        )
    )
    with torch.random.fork_rng(
        devices=_cuda_rng_devices(device),
        device_type=("cuda" if device.type == "cuda" else None),
    ):
        # The model is initialized on CPU before it is moved to ``device``.
        # Seed only the CPU generator: torch.manual_seed() would also alter
        # every visible CUDA generator, including devices outside this gate.
        torch.random.default_generator.manual_seed(
            COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED
        )
        config = CoverageStateCenteredMixedInteractionConfig(
            feature_channels=COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
            feature_stride=COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
            width=COVERAGE_STATE_CMIF_FORMAL_WIDTH,
        )
        model = CURELiteCenteredMixedInteractionLevelSet(config).to(
            device=device,
            dtype=torch.float32,
        )
        model.eval()
        initial_model = _model_fingerprint(model)
        feature_once = torch.cat(
            tuple(value.record.feature for value in optimized),
            dim=0,
        ).to(device=device, dtype=torch.float32)
        occupancy_plus = torch.cat(
            tuple(value.record.occupancy_plus for value in optimized),
            dim=0,
        ).to(device=device)
        occupancy_minus = torch.cat(
            tuple(value.record.occupancy_minus for value in optimized),
            dim=0,
        ).to(device=device)
        feature = torch.cat((feature_once, feature_once), dim=0)
        occupancy = torch.cat(
            (occupancy_plus, occupancy_minus),
            dim=0,
        )
        targets = _stack_targets(
            tuple(value.joint_targets for value in optimized),
            device=device,
        )
        field = model(feature, occupancy)
        pair_count = len(optimized)
        field_plus, field_minus = field.split(pair_count, dim=0)
        fields = coverage_state_uscope_pair_loss_from_targets(
            field_plus,
            field_minus,
            targets,
            config=population.cache.sobolev_config,
            validate=True,
        )
        (
            gradient_plus,
            gradient_minus,
            scalar_gradient,
        ) = torch.autograd.grad(
            fields.loss,
            (
                field_plus,
                field_minus,
                model.scalar_energy_weight,
            ),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        gamma = fields.per_state_chebyshev_violation.detach().cpu()
        losses = fields.per_state_loss.detach().cpu()
        q_plus_grid = fields.violation_plus.detach().cpu()
        q_minus_grid = fields.violation_minus.detach().cpu()
        q_plus = q_plus_grid.flatten(1)
        q_minus = q_minus_grid.flatten(1)
        valid_mask = targets.valid_mask.detach().cpu()
        signed_descent_plus = (
            -gradient_plus
            * torch.sign(targets.target_field_plus)
            * targets.valid_mask
        ).detach().cpu()
        signed_descent_minus = (
            -gradient_minus
            * torch.sign(targets.target_field_minus)
            * targets.valid_mask
        ).detach().cpu()
        target_field_plus = targets.target_field_plus.detach().cpu()
        target_field_minus = targets.target_field_minus.detach().cpu()
        rows: list[dict[str, object]] = []
        for index, value in enumerate(optimized):
            plus_max = float(q_plus[index].amax().item())
            minus_max = float(q_minus[index].amax().item())
            state_descent = torch.cat(
                (
                    signed_descent_plus[index].flatten(),
                    signed_descent_minus[index].flatten(),
                )
            )
            state_target = torch.cat(
                (
                    target_field_plus[index].flatten(),
                    target_field_minus[index].flatten(),
                )
            )
            target_descent = float(
                state_descent[state_target < 0.0].sum().item()
            )
            background_descent = float(
                state_descent[state_target > 0.0].sum().item()
            )
            total_descent = float(state_descent.sum().item())
            violation_zero_outside_valid = bool(
                torch.all(
                    q_plus_grid[index][~valid_mask[index]] == 0.0
                )
                and torch.all(
                    q_minus_grid[index][~valid_mask[index]] == 0.0
                )
            )
            rows.append(
                {
                    "pair_id": value.record.pair_id,
                    "sample_id": value.record.sample_id,
                    "optimizer_role": value.optimizer_role,
                    "loss_hex": _hex(
                        float(losses[index].item()),
                        name="USCOPE per-pair loss",
                    ),
                    "gamma_hex": _hex(
                        float(gamma[index].item()),
                        name="USCOPE gamma",
                    ),
                    "gamma_plus_hex": _hex(
                        plus_max,
                        name="USCOPE plus gamma",
                    ),
                    "gamma_minus_hex": _hex(
                        minus_max,
                        name="USCOPE minus gamma",
                    ),
                    "gamma_equals_global_endpoint_amax": (
                        float(gamma[index].item())
                        == max(plus_max, minus_max)
                    ),
                    "violation_zero_outside_declared_V": (
                        violation_zero_outside_valid
                    ),
                    "target_orthant_descent_alignment_hex": _hex(
                        target_descent,
                        name="USCOPE target descent alignment",
                    ),
                    "background_orthant_descent_alignment_hex": _hex(
                        background_descent,
                        name="USCOPE background descent alignment",
                    ),
                    "full_tensor_orthant_descent_alignment_hex": _hex(
                        total_descent,
                        name="USCOPE full-tensor descent alignment",
                    ),
                    "certificate_gamma_below_margin": bool(
                        gamma[index] < COVERAGE_STATE_USCOPE_MARGIN
                    ),
                    "worst_endpoint": (
                        "plus" if plus_max >= minus_max else "minus"
                    ),
                }
            )
        parameter_contract = tuple(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in model.named_parameters()
        )
        final_model = _model_fingerprint(model)
        fields_finite = bool(
            torch.isfinite(field).all()
            and torch.isfinite(fields.loss)
            and torch.isfinite(gradient_plus).all()
            and torch.isfinite(gradient_minus).all()
        )
        field_gradient_l2 = float(
            torch.sqrt(
                gradient_plus.square().sum()
                + gradient_minus.square().sum()
            ).detach().cpu()
        )
        scalar_gradient_l2 = float(
            scalar_gradient.detach().norm().cpu()
        )
        model_config_payload = {
            "feature_channels": config.feature_channels,
            "feature_stride": config.feature_stride,
            "width": config.width,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "objective_policy": CSLF_USCOPE_POLICY,
            "fixed_margin_hex": COVERAGE_STATE_USCOPE_MARGIN.hex(),
        }
        parameter_grad_buffers_unretained = all(
            parameter.grad is None for parameter in model.parameters()
        )
    population.verify_unchanged()
    device_rng_preserved = (
        True
        if before_device_rng is None
        else torch.equal(
            before_device_rng,
            torch.cuda.get_rng_state(device),
        )
    )
    return {
        "device": str(device),
        "execution_seed": COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED,
        "runtime_splits": runtime_splits,
        "split_access_evidence_policy": (
            "single_verified_real_input_graph_and_bounded_population_only"
        ),
        "single_batched_model_forward": True,
        "model_forward_invocations": 1,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "model_config": model_config_payload,
        "parameter_contract": list(parameter_contract),
        "initial_model_fingerprint": initial_model,
        "final_model_fingerprint": final_model,
        "population_fingerprint_before": before_population,
        "population_fingerprint_after": (
            population.population_fingerprint
        ),
        "rows": rows,
        "field_gradient_l2_hex": _hex(
            field_gradient_l2,
            name="USCOPE field gradient norm",
        ),
        "scalar_energy_gradient_l2_hex": _hex(
            scalar_gradient_l2,
            name="USCOPE scalar gradient norm",
        ),
        "field_and_gradients_finite": fields_finite,
        "field_gradient_nonzero": field_gradient_l2 > 0.0,
        "scalar_energy_gradient_finite": bool(
            torch.isfinite(scalar_gradient).all()
        ),
        "scalar_energy_gradient_nonzero": scalar_gradient_l2 > 0.0,
        "parameter_grad_buffers_unretained": (
            parameter_grad_buffers_unretained
        ),
        "global_cpu_rng_preserved": torch.equal(
            before_rng,
            torch.random.get_rng_state(),
        ),
        "selected_device_rng_preserved": device_rng_preserved,
    }


def recompute_coverage_state_uscope_dr_checks(
    *,
    dataset_free_receipt: CoverageStateUSCOPEDatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    population: CoverageStateBoundedPopulation,
    geometry: dict[str, object],
    probe: dict[str, object],
) -> tuple[tuple[str, bool], ...]:
    """Recompute every authorization bit without rerunning the forward."""

    rows = probe.get("rows", [])
    clean_rows = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("optimizer_role") == "clean_positive"
    ]
    component_rows = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("optimizer_role") == "component_null"
    ]
    geometry_rows = geometry.get("geometry_rows", [])
    optimized_pair_ids = geometry.get("optimized_pair_ids", [])
    memory_plan = geometry.get("memory_plan", {})
    checks = {
        "dataset_free_gate_passed": dataset_free_receipt.all_pass,
        "real_D_R_binding": (
            real_inputs.bundle.split == "D_R"
            and real_inputs.source_binding.split == "D_R"
            and real_inputs.source_binding.dataset == "IRSTD-1K"
            and population.source_cache is real_inputs.scalar_cache
        ),
        "bounded_seed42_population": (
            population.seed
            == COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED
            == COVERAGE_STATE_BOUNDED_SEED
        ),
        "bounded_pair_counts": (
            geometry.get("clean_positive_count")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and geometry.get("component_null_count")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and geometry.get("optimized_pair_count")
            == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
        ),
        "declared_V_nonempty_with_strict_target_sign": (
            geometry.get("valid_domain_nonempty_per_state") is True
            and geometry.get(
                "target_fields_strictly_nonzero_on_valid"
            )
            is True
            and int(geometry.get("endpoint_valid_pixel_count", 0)) > 0
        ),
        "pair_target_contracts_valid_on_declared_V": (
            geometry.get("integration_measure_mass_one") is True
            and geometry.get("target_and_occupancy_inside_valid") is True
            and geometry.get("joint_valid_matches_record") is True
            and geometry.get("integration_mass_rtol_hex")
            == COVERAGE_STATE_USCOPE_DR_MASS_RTOL.hex()
            and geometry.get("integration_mass_atol_hex")
            == COVERAGE_STATE_USCOPE_DR_MASS_ATOL.hex()
            and isinstance(geometry_rows, list)
            and len(geometry_rows)
            == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and all(
                isinstance(row, dict)
                and row.get("integration_mass_one") is True
                and row.get("pair_targets_validate_passed") is True
                and row.get("target_and_occupancy_inside_valid") is True
                and row.get("joint_valid_matches_record") is True
                and len(row.get("integration_mass_hex", [])) == 1
                and all(
                    abs(float.fromhex(str(value)) - 1.0)
                    <= (
                        COVERAGE_STATE_USCOPE_DR_MASS_ATOL
                        + COVERAGE_STATE_USCOPE_DR_MASS_RTOL
                    )
                    for value in row.get("integration_mass_hex", [])
                )
                for row in geometry_rows
            )
        ),
        "optimized_pair_geometry_contracts": (
            geometry.get("clean_geometry_contract") is True
            and geometry.get("component_geometry_contract") is True
            and geometry.get("output_matches_feature_stride") is True
        ),
        "single_batched_forward": (
            probe.get("single_batched_model_forward") is True
            and probe.get("model_forward_invocations") == 1
        ),
        "all_pair_gammas_finite_nonnegative": (
            isinstance(rows, list)
            and len(rows) == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and all(
                float.fromhex(str(row["gamma_hex"])) >= 0.0
                and float.fromhex(str(row["loss_hex"])) >= 0.0
                for row in rows
            )
        ),
        "gamma_is_exact_global_endpoint_amax": (
            isinstance(rows, list)
            and len(rows) == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and all(
                row.get("gamma_equals_global_endpoint_amax") is True
                and float.fromhex(str(row["gamma_hex"]))
                == max(
                    float.fromhex(str(row["gamma_plus_hex"])),
                    float.fromhex(str(row["gamma_minus_hex"])),
                )
                for row in rows
            )
        ),
        "clean_pair_signal_present": (
            len(clean_rows) == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and all(
                float.fromhex(str(row["gamma_hex"])) > 0.0
                and float.fromhex(str(row["loss_hex"])) > 0.0
                for row in clean_rows
            )
        ),
        "orthant_descent_direction_audited": (
            len(clean_rows) == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and all(
                float.fromhex(
                    str(
                        row[
                            "target_orthant_descent_alignment_hex"
                        ]
                    )
                )
                > 0.0
                and float.fromhex(
                    str(
                        row[
                            "full_tensor_orthant_descent_alignment_hex"
                        ]
                    )
                )
                > 0.0
                and float.fromhex(
                    str(
                        row[
                            "background_orthant_descent_alignment_hex"
                        ]
                    )
                )
                >= 0.0
                for row in clean_rows
            )
            and all(
                float.fromhex(
                    str(
                        row[
                            "full_tensor_orthant_descent_alignment_hex"
                        ]
                    )
                )
                >= 0.0
                and (
                    float.fromhex(str(row["gamma_hex"])) == 0.0
                    or float.fromhex(
                        str(
                            row[
                                "full_tensor_orthant_descent_alignment_hex"
                            ]
                        )
                    )
                    > 0.0
                )
                for row in component_rows
            )
        ),
        "component_rows_present": (
            len(component_rows) == COVERAGE_STATE_BOUNDED_ROLE_COUNT
        ),
        "clean_and_component_pairs_covered_exactly_once": (
            isinstance(optimized_pair_ids, list)
            and len(optimized_pair_ids)
            == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and len(set(optimized_pair_ids)) == len(optimized_pair_ids)
            and optimized_pair_ids
            == [row.get("pair_id") for row in geometry_rows]
            == [row.get("pair_id") for row in rows]
        ),
        "single_batch_memory_plan_sealed": (
            isinstance(memory_plan, dict)
            and memory_plan.get("pair_batch_count")
            == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and memory_plan.get("endpoint_batch_count")
            == 4 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and int(
                memory_plan.get(
                    "dominant_phase_width_tensor_bytes",
                    0,
                )
            )
            > 0
            and int(
                memory_plan.get(
                    "five_named_forward_tensors_bytes",
                    0,
                )
            )
            == 5
            * int(
                memory_plan.get(
                    "dominant_phase_width_tensor_bytes",
                    0,
                )
            )
            and memory_plan.get("estimate_is_not_measured_peak") is True
            and memory_plan.get("execution_policy")
            == "one_batch_32_pairs_64_endpoints_one_model_forward"
        ),
        "declared_V_hidden_negative_precondition": (
            geometry.get("valid_domain_nonempty_per_state") is True
            and geometry.get("target_and_occupancy_inside_valid") is True
            and geometry.get("joint_valid_matches_record") is True
            and all(
                row.get("gamma_equals_global_endpoint_amax") is True
                and row.get("violation_zero_outside_declared_V") is True
                for row in rows
            )
        ),
        "field_and_parameter_gradient_path": (
            probe.get("field_and_gradients_finite") is True
            and probe.get("field_gradient_nonzero") is True
            and probe.get("scalar_energy_gradient_finite") is True
            and probe.get("scalar_energy_gradient_nonzero") is True
        ),
        "model_and_population_unchanged": (
            probe.get("initial_model_fingerprint")
            == probe.get("final_model_fingerprint")
            and probe.get("population_fingerprint_before")
            == probe.get("population_fingerprint_after")
            == population.population_fingerprint
        ),
        "read_only_execution": (
            probe.get("execution_seed")
            == COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED
            and probe.get("runtime_splits") == ["D_R"]
            and probe.get("split_access_evidence_policy")
            == (
                "single_verified_real_input_graph_and_"
                "bounded_population_only"
            )
            and probe.get("optimizer_constructed") is False
            and probe.get("optimizer_steps") == 0
            and probe.get("parameter_grad_buffers_unretained") is True
            and probe.get("global_cpu_rng_preserved") is True
            and probe.get("selected_device_rng_preserved") is True
        ),
        "fixed_cmif_contract": (
            isinstance(probe.get("model_config"), dict)
            and probe["model_config"].get("feature_channels")
            == COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS
            and probe["model_config"].get("feature_stride")
            == COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE
            and probe["model_config"].get("width")
            == COVERAGE_STATE_CMIF_FORMAL_WIDTH
            and probe["model_config"].get("parameter_count")
            == COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
            and probe["model_config"].get("objective_policy")
            == CSLF_USCOPE_POLICY
        ),
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True, eq=False)
class CoverageStateUSCOPEDRGateReceipt:
    """Bound real-D_R geometry and one non-mutating USCOPE probe."""

    dataset_free_receipt: CoverageStateUSCOPEDatasetFreeReceipt
    real_inputs: CoverageStateRealDRInputs
    population: CoverageStateBoundedPopulation
    implementation_binding: tuple[tuple[str, str], ...]
    geometry: dict[str, object]
    probe: dict[str, object]
    checks: tuple[tuple[str, bool], ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        self.verify_unchanged()

    @property
    def all_pass(self) -> bool:
        self.verify_unchanged()
        return bool(self.checks) and all(value for _, value in self.checks)

    def _evidence_payload(self) -> dict[str, object]:
        return {
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt.receipt_fingerprint
            ),
            "real_inputs_build_fingerprint": (
                self.real_inputs.build_fingerprint
            ),
            "source_binding_fingerprint": (
                self.real_inputs.source_binding.binding_fingerprint
            ),
            "population_fingerprint": (
                self.population.population_fingerprint
            ),
            "cache_fingerprint": self.population.cache.cache_fingerprint,
            "implementation_binding": dict(self.implementation_binding),
            "geometry": deepcopy(self.geometry),
            "probe": deepcopy(self.probe),
        }

    def verify_unchanged(self) -> None:
        self.dataset_free_receipt.verify_unchanged()
        self.real_inputs.verify_unchanged()
        self.population.verify_unchanged()
        expected = recompute_coverage_state_uscope_dr_checks(
            dataset_free_receipt=self.dataset_free_receipt,
            real_inputs=self.real_inputs,
            population=self.population,
            geometry=self.geometry,
            probe=self.probe,
        )
        if (
            self.checks != expected
            or self.implementation_binding != _implementation_binding()
            or stable_fingerprint(self._evidence_payload())
            != self.evidence_fingerprint
        ):
            raise RuntimeError("USCOPE D_R evidence changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_USCOPE_DR_GATE_SCHEMA,
            **self._evidence_payload(),
            "checks": dict(self.checks),
            "all_pass": (
                bool(self.checks)
                and all(value for _, value in self.checks)
            ),
            "evidence_fingerprint": self.evidence_fingerprint,
            "execution": {
                "seed": COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED,
                "runtime_splits": list(
                    self.probe.get("runtime_splits", [])
                ),
                "training_performed": False,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "D_V_accessed": (
                    "D_V"
                    in self.probe.get("runtime_splits", [])
                ),
                "D_T_accessed": (
                    "D_T"
                    in self.probe.get("runtime_splits", [])
                ),
            },
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_coverage_state_uscope_dr_gate(
    *,
    dataset_free_receipt: CoverageStateUSCOPEDatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    population: CoverageStateBoundedPopulation,
    device: torch.device | str,
) -> CoverageStateUSCOPEDRGateReceipt:
    """Run the sole fixed-seed read-only USCOPE real-D_R probe."""

    if not isinstance(
        dataset_free_receipt,
        CoverageStateUSCOPEDatasetFreeReceipt,
    ):
        raise TypeError("dataset_free_receipt must be USCOPE receipt")
    if not isinstance(real_inputs, CoverageStateRealDRInputs):
        raise TypeError("real_inputs must be CoverageStateRealDRInputs")
    if not isinstance(population, CoverageStateBoundedPopulation):
        raise TypeError("population must be bounded population")
    real_inputs.verify_unchanged()
    population.verify_unchanged()
    if (
        not dataset_free_receipt.all_pass
        or population.source_cache is not real_inputs.scalar_cache
        or population.seed != COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED
    ):
        raise PermissionError("USCOPE D_R prerequisites did not pass")
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("USCOPE D_R gate supports only CPU or CUDA")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    if resolved_device.type == "cuda" and resolved_device.index is None:
        resolved_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
    if resolved_device.type == "cuda" and (
        resolved_device.index is None
        or resolved_device.index < 0
        or resolved_device.index >= torch.cuda.device_count()
    ):
        raise ValueError("requested CUDA device is unavailable")
    geometry = _geometry(population)
    probe = _probe(population, device=resolved_device)
    checks = recompute_coverage_state_uscope_dr_checks(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        population=population,
        geometry=geometry,
        probe=probe,
    )
    implementation = _implementation_binding()
    evidence = {
        "dataset_free_receipt_fingerprint": (
            dataset_free_receipt.receipt_fingerprint
        ),
        "real_inputs_build_fingerprint": real_inputs.build_fingerprint,
        "source_binding_fingerprint": (
            real_inputs.source_binding.binding_fingerprint
        ),
        "population_fingerprint": population.population_fingerprint,
        "cache_fingerprint": population.cache.cache_fingerprint,
        "implementation_binding": dict(implementation),
        "geometry": deepcopy(geometry),
        "probe": deepcopy(probe),
    }
    return CoverageStateUSCOPEDRGateReceipt(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        population=population,
        implementation_binding=implementation,
        geometry=geometry,
        probe=probe,
        checks=checks,
        evidence_fingerprint=stable_fingerprint(evidence),
    )


__all__ = [
    "COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED",
    "COVERAGE_STATE_USCOPE_DR_GATE_SCHEMA",
    "COVERAGE_STATE_USCOPE_DR_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_USCOPE_DR_MASS_ATOL",
    "COVERAGE_STATE_USCOPE_DR_MASS_RTOL",
    "CoverageStateUSCOPEDRGateReceipt",
    "recompute_coverage_state_uscope_dr_checks",
    "run_coverage_state_uscope_dr_gate",
]
