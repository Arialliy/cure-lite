"""Frozen spatial-tail companion diagnostic for the bounded paired replay.

This module is additive and diagnostic-only.  It reconstructs the exact
bounded 400-update proposed run, verifies the replay against its sealed
authority result, and measures spatial response tails on the fixed clean,
component-null, and identity-null micro-populations.  It does not change any
training objective, add a retrospective gate, access D_V/D_T, or authorize
formal training.
"""

from __future__ import annotations

from math import isfinite, sqrt
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from ..cache.schema import stable_fingerprint
from ..config import DecoderConfig, LossConfig
from ..decoder import CURELiteDecoder, project_occupancy_to_feature_grid
from ..losses import CURELiteLoss
from ..paired_losses import PairedDifferenceLoss
from ..paired_types import PairBatch, PairExample, stack_pair_examples
from ..train.paired_step import _paired_endpoint_logits, paired_train_step
from .artifacts import decoder_state_fingerprint
from .paired_bounded_learnability import (
    BoundedMicroPopulation,
    BoundedMicroSchedule,
    _ForwardLedger,
    _deterministic_torch_runtime,
    _factual_batches,
    _pair_batch,
    evaluate_bounded_micro_population,
)


SPATIAL_TAIL_EXECUTION_SCHEMA = (
    "cure-lite-paired-spatial-tail-diagnostic-execution-v1"
)
SPATIAL_TAIL_POPULATION_SCHEMA = (
    "cure-lite-paired-spatial-tail-population-v1"
)


def _finite_fraction_sequence(
    values: object,
    *,
    name: str,
    lower_exclusive: float,
    upper_exclusive: float,
) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(float(value) for value in values)
    if (
        not result
        or any(not isfinite(value) for value in result)
        or any(
            value <= lower_exclusive or value >= upper_exclusive
            for value in result
        )
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(
            f"{name} must be sorted, unique, finite, and lie strictly "
            f"inside ({lower_exclusive}, {upper_exclusive})"
        )
    return result


def _positive_int_sequence(
    values: object,
    *,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must contain positive integers")
        result.append(value)
    output = tuple(result)
    if not output or output != tuple(sorted(set(output))):
        raise ValueError(f"{name} must be sorted, unique, and non-empty")
    return output


def validate_spatial_tail_specification(
    specification: Mapping[str, object],
) -> dict[str, object]:
    """Validate and normalize the frozen descriptive diagnostic settings."""

    if not isinstance(specification, Mapping):
        raise TypeError("spatial-tail specification must be a mapping")
    thresholds = _finite_fraction_sequence(
        specification.get("absolute_delta_thresholds"),
        name="absolute_delta_thresholds",
        lower_exclusive=0.0,
        upper_exclusive=1.0,
    )
    quantiles = _finite_fraction_sequence(
        specification.get("quantiles"),
        name="quantiles",
        lower_exclusive=0.0,
        upper_exclusive=1.0,
    )
    radii = _positive_int_sequence(
        specification.get("deleted_component_neighborhood_radii_px"),
        name="deleted_component_neighborhood_radii_px",
    )
    connectivity = specification.get("connected_component_connectivity")
    if connectivity != 8:
        raise ValueError(
            "spatial-tail connected-component connectivity must remain 8"
        )
    if specification.get("thresholds_are_descriptive_not_gates") is not True:
        raise ValueError("spatial-tail thresholds may not be decision gates")
    if (
        specification.get("projected_cell_output_mapping")
        != "nearest-output-support-of-xor-projected-endpoints-v1"
    ):
        raise ValueError("projected feature-cell mapping differs from freeze")
    return {
        "absolute_delta_thresholds": thresholds,
        "quantiles": quantiles,
        "deleted_component_neighborhood_radii_px": radii,
        "connected_component_connectivity": connectivity,
        "thresholds_are_descriptive_not_gates": True,
        "projected_cell_output_mapping": (
            "nearest-output-support-of-xor-projected-endpoints-v1"
        ),
    }


def _mask_coordinates(mask: Tensor) -> Tensor:
    if mask.ndim != 2 or mask.dtype != torch.bool or mask.device.type != "cpu":
        raise TypeError("distance masks must be CPU bool [H,W] tensors")
    return torch.nonzero(mask, as_tuple=False).to(torch.float64)


def _minimum_euclidean_distance(
    left: Tensor,
    right: Tensor,
) -> float | None:
    left_coordinates = _mask_coordinates(left)
    right_coordinates = _mask_coordinates(right)
    if left_coordinates.numel() == 0 or right_coordinates.numel() == 0:
        return None
    minimum = float("inf")
    for start in range(0, left_coordinates.shape[0], 2048):
        distances = torch.cdist(
            left_coordinates[start : start + 2048],
            right_coordinates,
            p=2.0,
        )
        minimum = min(minimum, float(distances.min().item()))
    return minimum


def _point_distances(
    coordinate: tuple[int, int],
    support: Tensor,
) -> dict[str, float] | None:
    support_coordinates = _mask_coordinates(support)
    if support_coordinates.numel() == 0:
        return None
    point = torch.tensor(coordinate, dtype=torch.float64).reshape(1, 2)
    offsets = (support_coordinates - point).abs()
    return {
        "euclidean_px": float(
            torch.sqrt((offsets.square()).sum(dim=1)).min().item()
        ),
        "chebyshev_px": float(offsets.max(dim=1).values.min().item()),
    }


def _dilate(mask: Tensor, radius: int) -> Tensor:
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise TypeError("dilation mask must be bool [H,W]")
    value = F.max_pool2d(
        mask.to(torch.float32).reshape(1, 1, *mask.shape),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return value[0, 0].to(torch.bool)


def _connected_components(
    mask: Tensor,
    *,
    connectivity: int,
) -> dict[str, object]:
    if mask.ndim != 2 or mask.dtype != torch.bool or mask.device.type != "cpu":
        raise TypeError("component mask must be CPU bool [H,W]")
    if connectivity != 8:
        raise ValueError("only frozen 8-connectivity is supported")
    remaining = {
        (int(row), int(column))
        for row, column in torch.nonzero(mask, as_tuple=False).tolist()
    }
    areas: list[int] = []
    neighbors = tuple(
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        area = 0
        while stack:
            row, column = stack.pop()
            area += 1
            for dy, dx in neighbors:
                candidate = (row + dy, column + dx)
                if candidate in remaining:
                    remaining.remove(candidate)
                    stack.append(candidate)
        areas.append(area)
    areas.sort(reverse=True)
    return {
        "count": len(areas),
        "largest_area_px": 0 if not areas else areas[0],
        "areas_px_descending": areas,
    }


def _projected_change_support(
    batch: PairBatch,
    index: int,
) -> tuple[Tensor, Tensor]:
    feature_size = tuple(int(value) for value in batch.feature.shape[-2:])
    plus = project_occupancy_to_feature_grid(
        batch.occupancy_plus[index : index + 1].cpu(),
        feature_size,
    )
    minus = project_occupancy_to_feature_grid(
        batch.occupancy_minus[index : index + 1].cpu(),
        feature_size,
    )
    changed = (plus ^ minus)[0, 0]
    output = F.interpolate(
        changed.to(torch.float32).reshape(1, 1, *changed.shape),
        size=tuple(int(value) for value in batch.occupancy_plus.shape[-2:]),
        mode="nearest",
    )[0, 0].to(torch.bool)
    return changed, output


def _overlap_summary(
    response: Tensor,
    support: Tensor,
) -> dict[str, object]:
    response_count = int(torch.count_nonzero(response))
    support_count = int(torch.count_nonzero(support))
    intersection = int(torch.count_nonzero(response & support))
    return {
        "support_pixel_count": support_count,
        "intersection_pixel_count": intersection,
        "response_fraction": (
            None if response_count == 0 else intersection / response_count
        ),
        "support_fraction": (
            None if support_count == 0 else intersection / support_count
        ),
        "minimum_euclidean_distance_px": _minimum_euclidean_distance(
            response,
            support,
        ),
    }


def summarize_pair_spatial_tail(
    delta: Tensor,
    batch: PairBatch,
    examples: Sequence[PairExample],
    specification: Mapping[str, object],
) -> dict[str, object]:
    """Summarize every pair without reducing sparse peaks into a map mean."""

    normalized = validate_spatial_tail_specification(specification)
    if not isinstance(delta, Tensor) or delta.ndim != 4 or delta.shape[1] != 1:
        raise ValueError("delta must have shape [B,1,H,W]")
    if not delta.is_floating_point() or not torch.isfinite(delta).all():
        raise ValueError("delta must be finite floating point")
    batch.validate()
    values = tuple(examples)
    if any(not isinstance(example, PairExample) for example in values):
        raise TypeError("examples must contain only PairExample values")
    if len(values) != delta.shape[0] or len(values) != len(batch.pair_ids):
        raise ValueError("examples, delta, and batch must align")
    if any(
        example.pair_id != pair_id
        for example, pair_id in zip(values, batch.pair_ids, strict=True)
    ):
        raise ValueError("example identities do not align with PairBatch")

    thresholds = normalized["absolute_delta_thresholds"]
    quantiles = normalized["quantiles"]
    radii = normalized["deleted_component_neighborhood_radii_px"]
    connectivity = int(normalized["connected_component_connectivity"])
    if not isinstance(thresholds, tuple) or not isinstance(
        quantiles,
        tuple,
    ) or not isinstance(radii, tuple):
        raise AssertionError("normalized spatial-tail sequences changed type")

    rows: list[dict[str, object]] = []
    for index, example in enumerate(values):
        signed = delta[index, 0].detach().to(torch.float64).cpu()
        valid = batch.image_valid_mask[index, 0].detach().cpu()
        selected = signed[valid]
        if selected.numel() == 0:
            raise ValueError("every spatial-tail pair requires valid pixels")
        absolute = signed.abs()
        absolute_valid = absolute[valid]
        masked_absolute = absolute.clone()
        masked_absolute[~valid] = -1.0
        flat_index = int(masked_absolute.reshape(-1).argmax().item())
        width = int(signed.shape[1])
        argmax = (flat_index // width, flat_index % width)
        masked_signed_max = signed.clone()
        masked_signed_max[~valid] = -float("inf")
        signed_max_flat = int(masked_signed_max.reshape(-1).argmax().item())
        signed_max_argmax = (
            signed_max_flat // width,
            signed_max_flat % width,
        )
        masked_signed_min = signed.clone()
        masked_signed_min[~valid] = float("inf")
        signed_min_flat = int(masked_signed_min.reshape(-1).argmin().item())
        signed_min_argmin = (
            signed_min_flat // width,
            signed_min_flat % width,
        )

        deleted = (
            batch.occupancy_plus[index, 0]
            & ~batch.occupancy_minus[index, 0]
        ).detach().cpu()
        label_increment = (
            batch.label_increment[index, 0].to(torch.bool).detach().cpu()
        )
        projected_grid, projected_output = _projected_change_support(
            batch,
            index,
        )
        neighborhoods = {
            radius: (_dilate(deleted, radius) & valid)
            for radius in radii
        }
        support_masks: dict[str, Tensor] = {
            "deleted_component": deleted & valid,
            "label_increment": label_increment & valid,
            "projected_changed_cell_output_support": (
                projected_output & valid
            ),
            **{
                f"deleted_component_neighborhood_r{radius}": mask
                for radius, mask in neighborhoods.items()
            },
        }

        threshold_rows: dict[str, object] = {}
        for threshold in thresholds:
            response = (absolute >= threshold) & valid
            positive = (signed >= threshold) & valid
            negative = (signed <= -threshold) & valid
            key = f"{threshold:.3f}"
            threshold_rows[key] = {
                "absolute_pixel_count": int(torch.count_nonzero(response)),
                "positive_pixel_count": int(torch.count_nonzero(positive)),
                "negative_pixel_count": int(torch.count_nonzero(negative)),
                "absolute_fraction_of_valid": (
                    float(torch.count_nonzero(response)) / selected.numel()
                ),
                "connected_components_8": _connected_components(
                    response,
                    connectivity=connectivity,
                ),
                "support_overlap": {
                    name: _overlap_summary(response, support)
                    for name, support in support_masks.items()
                },
            }

        absolute_at_argmax = float(absolute[argmax].item())
        rows.append(
            {
                "pair_id": example.pair_id,
                "pair_kind": example.pair_kind,
                "sample_id": example.sample_id,
                "valid_pixel_count": int(selected.numel()),
                "removed_component_pixel_count": int(
                    torch.count_nonzero(deleted)
                ),
                "label_increment_pixel_count": int(
                    torch.count_nonzero(label_increment)
                ),
                "projected_changed_feature_cell_count": int(
                    torch.count_nonzero(projected_grid)
                ),
                "projected_changed_output_support_pixel_count": int(
                    torch.count_nonzero(projected_output & valid)
                ),
                "signed_min_delta": float(selected.min().item()),
                "signed_max_delta": float(selected.max().item()),
                "signed_min_argmin_yx": [
                    signed_min_argmin[0],
                    signed_min_argmin[1],
                ],
                "signed_max_argmax_yx": [
                    signed_max_argmax[0],
                    signed_max_argmax[1],
                ],
                "absolute_max_delta": absolute_at_argmax,
                "signed_delta_at_absolute_argmax": float(
                    signed[argmax].item()
                ),
                "absolute_argmax_yx": [argmax[0], argmax[1]],
                "mean_abs_delta": float(absolute_valid.mean().item()),
                "rms_delta": float(
                    torch.sqrt(selected.square().mean()).item()
                ),
                "signed_quantiles": {
                    f"{quantile:.3f}": float(
                        torch.quantile(selected, quantile).item()
                    )
                    for quantile in quantiles
                },
                "absolute_quantiles": {
                    f"{quantile:.3f}": float(
                        torch.quantile(absolute_valid, quantile).item()
                    )
                    for quantile in quantiles
                },
                "absolute_argmax_distance_to_support": {
                    name: _point_distances(argmax, support)
                    for name, support in support_masks.items()
                },
                "thresholds": threshold_rows,
            }
        )
    return {
        "schema_version": SPATIAL_TAIL_POPULATION_SCHEMA,
        "pair_count": len(rows),
        "pair_kind": (
            rows[0]["pair_kind"]
            if len({row["pair_kind"] for row in rows}) == 1
            else "mixed"
        ),
        "maximum_abs_delta": max(
            float(row["absolute_max_delta"]) for row in rows
        ),
        "macro_mean_abs_delta": sum(
            float(row["mean_abs_delta"]) for row in rows
        )
        / len(rows),
        "rows": rows,
    }


def evaluate_spatial_tail_populations(
    decoder: CURELiteDecoder,
    population: BoundedMicroPopulation,
    specification: Mapping[str, object],
    *,
    device: torch.device | str,
) -> dict[str, object]:
    """Evaluate all three fixed 16-pair comparator populations once."""

    decoder.eval()
    result: dict[str, object] = {}
    with torch.no_grad():
        for name, examples in (
            ("clean_positive", population.clean_pairs),
            ("component_null", population.component_null),
            ("identity_null", population.identity_null),
        ):
            batch = stack_pair_examples(examples, device=device)
            # ``paired_endpoint_logits`` is intentionally restricted to the
            # optimizer's clean-positive path.  This companion is read-only
            # and must also evaluate component/identity nulls, so reuse the
            # same validated 2B tensor forward without entering that training
            # preflight.
            logits_plus, logits_minus = _paired_endpoint_logits(
                decoder,
                feature=batch.feature,
                occupancy_plus=batch.occupancy_plus,
                occupancy_minus=batch.occupancy_minus,
            )
            delta = torch.sigmoid(logits_minus) - torch.sigmoid(logits_plus)
            result[name] = summarize_pair_spatial_tail(
                delta,
                batch,
                examples,
                specification,
            )
    return result


def execute_spatial_tail_replay(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    bounded_config: Mapping[str, object],
    diagnostic_config: Mapping[str, object],
    authority_result: Mapping[str, object],
    *,
    device: torch.device | str,
) -> dict[str, object]:
    """Replay the exact bounded optimizer path and add descriptive snapshots."""

    for name, value in (
        ("bounded_config", bounded_config),
        ("diagnostic_config", diagnostic_config),
        ("authority_result", authority_result),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
    diagnostic = validate_spatial_tail_specification(
        diagnostic_config["diagnostic"]
    )
    optimization = bounded_config["optimization"]
    budget = bounded_config["budget"]
    determinism = bounded_config["determinism"]
    if not all(
        isinstance(value, Mapping)
        for value in (optimization, budget, determinism)
    ):
        raise TypeError("bounded replay configuration is malformed")
    if (
        int(budget["optimizer_updates"]) != 400
        or int(budget["steps_per_epoch"]) != 40
        or schedule.optimizer_updates != 400
        or schedule.steps_per_epoch != 40
    ):
        raise RuntimeError("spatial-tail replay budget differs from bounded run")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")

    seed = int(optimization["seed"])
    decoder_config = DecoderConfig(**dict(optimization["decoder"]))
    loss_config = LossConfig(**dict(optimization["loss"]))
    cuda_devices: list[int] = []
    if target_device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device()
            if target_device.index is None
            else target_device.index
        ]

    with _deterministic_torch_runtime(
        target_device,
        determinism,
    ) as deterministic_runtime, torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        decoder = CURELiteDecoder(decoder_config).to(target_device)
        absolute = CURELiteLoss(loss_config)
        paired = PairedDifferenceLoss()
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        initial_decoder_fingerprint = decoder_state_fingerprint(decoder)

        initial_standard = evaluate_bounded_micro_population(
            decoder,
            population,
            absolute,
            paired,
            device=target_device,
        )
        initial_spatial_ledger = _ForwardLedger(decoder)
        try:
            initial_spatial = evaluate_spatial_tail_populations(
                decoder,
                population,
                diagnostic,
                device=target_device,
            )
            initial_spatial_forward = {
                "calls": initial_spatial_ledger.calls,
                "state_evaluations": initial_spatial_ledger.states,
            }
        finally:
            initial_spatial_ledger.close()

        ledger = _ForwardLedger(decoder)
        trace: list[dict[str, object]] = []
        minimum_gradient_norm = float("inf")
        maximum_gradient_norm = 0.0
        try:
            for update in range(schedule.optimizer_updates):
                before_update = ledger.snapshot()
                logs = paired_train_step(
                    decoder,
                    absolute,
                    paired,
                    optimizer,
                    _factual_batches(
                        population,
                        schedule,
                        update,
                        device=target_device,
                    ),
                    _pair_batch(
                        population,
                        schedule,
                        update,
                        device=target_device,
                    ),
                )
                squared_norm = sum(
                    float(
                        parameter.grad.detach().double().square().sum().cpu()
                    )
                    for parameter in decoder.parameters()
                    if parameter.grad is not None
                )
                gradient_norm = sqrt(squared_norm)
                minimum_gradient_norm = min(
                    minimum_gradient_norm,
                    gradient_norm,
                )
                maximum_gradient_norm = max(
                    maximum_gradient_norm,
                    gradient_norm,
                )
                after_update = ledger.snapshot()
                trace.append(
                    {
                        "update": update,
                        "epoch": update // schedule.steps_per_epoch,
                        "step": update % schedule.steps_per_epoch,
                        "losses": logs,
                        "gradient_l2_norm": gradient_norm,
                        "decoder_forward_calls": (
                            after_update[0] - before_update[0]
                        ),
                        "decoder_state_evaluations": (
                            after_update[1] - before_update[1]
                        ),
                    }
                )
            training_forward = {
                "calls": ledger.calls,
                "state_evaluations": ledger.states,
            }
        finally:
            ledger.close()

        final_standard = evaluate_bounded_micro_population(
            decoder,
            population,
            absolute,
            paired,
            device=target_device,
        )
        final_spatial_ledger = _ForwardLedger(decoder)
        try:
            final_spatial = evaluate_spatial_tail_populations(
                decoder,
                population,
                diagnostic,
                device=target_device,
            )
            final_spatial_forward = {
                "calls": final_spatial_ledger.calls,
                "state_evaluations": final_spatial_ledger.states,
            }
        finally:
            final_spatial_ledger.close()
        final_decoder_fingerprint = decoder_state_fingerprint(decoder)

    authority_parameters = authority_result.get("parameters")
    authority_gradients = authority_result.get("gradients")
    if not isinstance(authority_parameters, Mapping) or not isinstance(
        authority_gradients,
        Mapping,
    ):
        raise RuntimeError("bounded authority result is malformed")
    diagnostic_contract = diagnostic_config.get("diagnostic")
    if not isinstance(diagnostic_contract, Mapping):
        raise RuntimeError("spatial-tail diagnostic contract is malformed")
    diagnostic_forward = {
        "snapshots": 2,
        "populations_per_snapshot": 3,
        "decoder_forward_calls": (
            initial_spatial_forward["calls"] + final_spatial_forward["calls"]
        ),
        "decoder_state_evaluations": (
            initial_spatial_forward["state_evaluations"]
            + final_spatial_forward["state_evaluations"]
        ),
        "initial_snapshot": initial_spatial_forward,
        "final_snapshot": final_spatial_forward,
    }
    replay_checks = {
        "deterministic_runtime_contract_satisfied": (
            deterministic_runtime["contract_satisfied"] is True
            and deterministic_runtime["flags_restored_after_execution"] is True
        ),
        "population_fingerprint_exact": (
            authority_result.get("population_fingerprint")
            == population.population_fingerprint
        ),
        "schedule_fingerprint_exact": (
            authority_result.get("schedule_fingerprint")
            == schedule.schedule_fingerprint
        ),
        "optimizer_updates_exact": (
            len(trace)
            == authority_result.get("optimizer_updates_completed")
            == 400
        ),
        "initial_decoder_fingerprint_exact": (
            initial_decoder_fingerprint
            == authority_parameters.get("initial_decoder_fingerprint")
        ),
        "final_decoder_fingerprint_exact": (
            final_decoder_fingerprint
            == authority_parameters.get("final_decoder_fingerprint")
        ),
        "initial_standard_metrics_exact": (
            initial_standard == authority_result.get("initial")
        ),
        "final_standard_metrics_exact": (
            final_standard == authority_result.get("final")
        ),
        "training_trace_exact": trace == authority_result.get("trace"),
        "minimum_gradient_norm_exact": (
            minimum_gradient_norm
            == authority_gradients.get("minimum_update_l2_norm")
        ),
        "maximum_gradient_norm_exact": (
            maximum_gradient_norm
            == authority_gradients.get("maximum_update_l2_norm")
        ),
        "training_forward_budget_exact": (
            training_forward
            == authority_result.get("forward_budget", {}).get("training")
            == {
                "calls": 1200,
                "state_evaluations": 4800,
            }
        ),
        "all_three_populations_complete": all(
            initial_spatial[name]["pair_count"]
            == final_spatial[name]["pair_count"]
            == 16
            for name in (
                "clean_positive",
                "component_null",
                "identity_null",
            )
        ),
        "identity_null_remains_exact": (
            final_spatial["identity_null"]["maximum_abs_delta"] == 0.0
        ),
        "diagnostic_forward_budget_exact": (
            diagnostic_forward["decoder_forward_calls"]
            == diagnostic_contract.get("diagnostic_decoder_forward_calls")
            == 6
            and diagnostic_forward["decoder_state_evaluations"]
            == diagnostic_contract.get(
                "diagnostic_decoder_state_evaluations"
            )
            == 192
            and initial_spatial_forward
            == final_spatial_forward
            == {"calls": 3, "state_evaluations": 96}
        ),
    }
    replay_exact = all(replay_checks.values())
    if not replay_exact:
        failed = sorted(
            name for name, passed in replay_checks.items() if not passed
        )
        raise RuntimeError(
            "spatial-tail replay differs from bounded authority: "
            + ", ".join(failed)
        )
    return {
        "schema_version": SPATIAL_TAIL_EXECUTION_SCHEMA,
        "execution_status": "completed",
        "split": "D_R",
        "device": str(target_device),
        "population_fingerprint": population.population_fingerprint,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "authority_result_receipt_fingerprint": authority_result[
            "receipt_fingerprint"
        ],
        "optimizer_updates_completed": len(trace),
        "replay_checks": replay_checks,
        "exact_bounded_replay_verified": True,
        "deterministic_runtime": deterministic_runtime,
        "training_forward_budget": training_forward,
        "diagnostic_forward_budget": diagnostic_forward,
        "parameters": {
            "initial_decoder_fingerprint": initial_decoder_fingerprint,
            "final_decoder_fingerprint": final_decoder_fingerprint,
        },
        "gradients": {
            "minimum_update_l2_norm": minimum_gradient_norm,
            "maximum_update_l2_norm": maximum_gradient_norm,
        },
        "trace_fingerprint": stable_fingerprint(trace),
        "diagnostic_specification": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in diagnostic.items()
        },
        "initial_spatial_tail": initial_spatial,
        "final_spatial_tail": final_spatial,
        "interpretation": {
            "descriptive_companion_only": True,
            "retroactive_gate_added": False,
            "bounded_decision_changed": False,
            "not_performance_evidence": True,
            "authorizes_formal_800": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "checkpoint_recovery_performed": False,
            "checkpoint_persisted": False,
        },
    }


__all__ = [
    "SPATIAL_TAIL_EXECUTION_SCHEMA",
    "SPATIAL_TAIL_POPULATION_SCHEMA",
    "evaluate_spatial_tail_populations",
    "execute_spatial_tail_replay",
    "summarize_pair_spatial_tail",
    "validate_spatial_tail_specification",
]
