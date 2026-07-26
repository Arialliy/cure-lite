"""Bounded D_R-only computation for the anchored paired-transition objective.

This is an in-memory model-design gate, not a detector evaluation.  It trains
one fresh CURE-Lite decoder with the frozen 4/4/2 micro schedule, evaluates all
16 selected clean pairs before and after training, and keeps both null
populations strictly outside the optimizer path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from math import isfinite, sqrt
from typing import Mapping

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..config import DecoderConfig, LossConfig
from ..decoder import CURELiteDecoder
from ..losses import CURELiteLoss
from ..paired_transition_losses import AnchoredTransitionLoss
from ..paired_transition_types import (
    AnchoredPairBatch,
    stack_anchored_pair_examples,
)
from ..paired_types import stack_pair_examples
from ..train.paired_step import diagnose_null_pairs, paired_endpoint_logits
from ..train.paired_transition_step import anchored_transition_train_step
from ..train.pools import stack_state_examples
from .artifacts import decoder_state_fingerprint
from .paired_bounded_learnability import (
    BoundedMicroPopulation,
    BoundedMicroSchedule,
    _deterministic_torch_runtime,
)
from .paired_transition_inputs import PairedTransitionInputMaterializer


PAIRED_TRANSITION_BOUNDED_SCHEMA = (
    "cure-lite-paired-transition-bounded-execution-v1"
)

DETERMINISM_SPECIFICATION: Mapping[str, object] = {
    "torch_use_deterministic_algorithms": True,
    "torch_deterministic_warn_only": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cublas_workspace_config": ":4096:8",
    "restore_process_torch_flags_after_execution": True,
    "exact_replay_required_under_same_frozen_environment": True,
}

# These gates establish only bounded computational learnability on the sealed
# D_R micro-population.  They are deliberately not Pd/FA, calibration, or
# cross-dataset performance thresholds.
COMPUTATIONAL_THRESHOLDS: Mapping[str, float] = {
    "apto_total_loss_final_over_initial_max": 0.50,
    "plus_anchor_loss_final_over_initial_max": 0.75,
    "transition_loss_final_over_initial_max": 0.50,
    "d_positive_macro_mean_delta_min": 0.50,
    "d_positive_pair_fraction_ge_0_25_min": 0.75,
    "plus_anchor_target_probability_mean_min": 0.75,
    "plus_anchor_background_probability_mean_max": 0.05,
    "zero_response_macro_mean_abs_delta_max": 0.05,
    "component_null_macro_mean_abs_delta_max": 0.05,
    "identity_null_max_abs_delta_max": 1.0e-7,
}


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_execution_inputs(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    materializer: PairedTransitionInputMaterializer,
    decoder_config: DecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    device: torch.device | str,
) -> tuple[torch.device, int, float, float]:
    if not isinstance(population, BoundedMicroPopulation):
        raise TypeError("population must be BoundedMicroPopulation")
    if not isinstance(schedule, BoundedMicroSchedule):
        raise TypeError("schedule must be BoundedMicroSchedule")
    if not isinstance(materializer, PairedTransitionInputMaterializer):
        raise TypeError(
            "materializer must be PairedTransitionInputMaterializer"
        )
    if not isinstance(decoder_config, DecoderConfig):
        raise TypeError("decoder_config must be DecoderConfig")
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss_config must be LossConfig")
    if not isinstance(optimization_budget, Mapping):
        raise TypeError("optimization_budget must be a mapping")

    required_budget = {
        "seed",
        "optimizer_updates",
        "steps_per_epoch",
        "factual_miss_states_per_update",
        "factual_no_miss_states_per_update",
        "clean_pairs_per_update",
        "learning_rate",
        "weight_decay",
    }
    if set(optimization_budget) != required_budget:
        raise ValueError(
            "optimization_budget must contain exactly "
            f"{sorted(required_budget)}"
        )
    seed = optimization_budget["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("optimization_budget.seed must be an integer")
    updates = _positive_int(
        optimization_budget["optimizer_updates"],
        name="optimization_budget.optimizer_updates",
    )
    steps_per_epoch = _positive_int(
        optimization_budget["steps_per_epoch"],
        name="optimization_budget.steps_per_epoch",
    )
    if (
        updates != schedule.optimizer_updates
        or steps_per_epoch != schedule.steps_per_epoch
    ):
        raise ValueError("optimization budget and bounded schedule disagree")
    if optimization_budget["factual_miss_states_per_update"] != 4:
        raise ValueError("APTO requires four factual-miss states per update")
    if optimization_budget["factual_no_miss_states_per_update"] != 4:
        raise ValueError("APTO requires four factual-no-miss states per update")
    if optimization_budget["clean_pairs_per_update"] != 2:
        raise ValueError("APTO requires two clean pairs per update")
    learning_rate = _finite_float(
        optimization_budget["learning_rate"],
        name="optimization_budget.learning_rate",
        positive=True,
    )
    weight_decay = _finite_float(
        optimization_budget["weight_decay"],
        name="optimization_budget.weight_decay",
        nonnegative=True,
    )

    if any(
        len(getattr(population, name)) != 16
        for name in (
            "clean_pairs",
            "factual_miss",
            "factual_no_miss",
            "component_null",
            "identity_null",
        )
    ):
        raise ValueError("APTO bounded execution requires five 16-unit populations")
    if (
        population.pair_catalog_fingerprint
        != materializer.pair_catalog_fingerprint
    ):
        raise ValueError("population and materializer pair catalogs disagree")
    materializer.verify_unchanged()
    clean_ids = tuple(pair.pair_id for pair in population.clean_pairs)
    if len(set(clean_ids)) != 16:
        raise ValueError("the bounded clean-pair identities must be unique")
    if any(pair_id not in materializer.pair_by_id for pair_id in clean_ids):
        raise ValueError("materializer does not bind every bounded clean pair")
    for pair in population.clean_pairs:
        bound = materializer.pair_by_id[pair.pair_id]
        if pair.canonical_payload() != bound.canonical_payload():
            raise ValueError("bounded clean pair differs from materialized binding")

    if decoder_config.feature_channels != materializer.feature_shape[1]:
        raise ValueError("decoder feature channels differ from materialized inputs")
    if len(schedule.pair_counts) != 16:
        raise ValueError("pair exposure ledger must contain 16 entries")
    if len(schedule.factual_miss_counts) != 16:
        raise ValueError("factual-miss exposure ledger must contain 16 entries")
    if len(schedule.factual_no_miss_counts) != 16:
        raise ValueError("factual-no-miss exposure ledger must contain 16 entries")
    if any(
        not 0 <= index < 16
        for rows in (
            schedule.pair_indices,
            schedule.factual_miss_indices,
            schedule.factual_no_miss_indices,
        )
        for row in rows
        for index in row
    ):
        raise ValueError("bounded schedule contains an out-of-range index")

    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("APTO bounded execution supports only CPU or CUDA")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return target_device, seed, learning_rate, weight_decay


class _ForwardLedger:
    def __init__(self, decoder: CURELiteDecoder) -> None:
        self.calls = 0
        self.states = 0
        self._handle = decoder.register_forward_hook(self._record)

    def _record(
        self,
        module: torch.nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise RuntimeError("decoder forward ledger received invalid output")
        self.calls += 1
        self.states += int(output.shape[0])

    def snapshot(self) -> tuple[int, int]:
        return self.calls, self.states

    def close(self) -> None:
        self._handle.remove()


def _masked_scalar(values: Tensor, mask: Tensor, *, name: str) -> Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        raise RuntimeError(f"{name} population is empty")
    if not torch.isfinite(selected).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return selected


def _clean_metrics(
    decoder: CURELiteDecoder,
    materializer: PairedTransitionInputMaterializer,
    pair_ids: tuple[str, ...],
    criterion: AnchoredTransitionLoss,
    *,
    device: torch.device,
) -> dict[str, object]:
    batch = _materialize_bound_pairs(
        materializer,
        pair_ids,
        device=device,
    )
    logits_plus, logits_minus = paired_endpoint_logits(
        decoder,
        batch.pair_batch,
    )
    result = criterion(
        logits_plus,
        logits_minus,
        batch.completion_plus,
        batch.occupancy_plus,
        batch.gt_union,
        batch.label_increment,
        batch.image_valid_mask,
    )
    score_plus = torch.sigmoid(logits_plus)
    score_minus = torch.sigmoid(logits_minus)
    delta = score_minus - score_plus
    positive = batch.label_increment.to(dtype=torch.bool)
    zero = batch.image_valid_mask & ~positive
    anchor_target = batch.completion_plus
    anchor_background = (
        batch.image_valid_mask
        & ~batch.occupancy_plus
        & ~batch.gt_union
    )

    d_values = _masked_scalar(delta, positive, name="D delta")
    zero_values = _masked_scalar(delta.abs(), zero, name="zero-response delta")
    anchor_target_values = score_plus[anchor_target]
    anchor_background_values = score_plus[anchor_background]
    per_pair_d = torch.stack(
        [delta[index][positive[index]].mean() for index in range(len(pair_ids))]
    )
    per_pair_zero = torch.stack(
        [delta[index][zero[index]].abs().mean() for index in range(len(pair_ids))]
    )
    per_pair_target: list[float | None] = []
    per_pair_background: list[float | None] = []
    for index in range(len(pair_ids)):
        target_values = score_plus[index][anchor_target[index]]
        background_values = score_plus[index][anchor_background[index]]
        per_pair_target.append(
            None if target_values.numel() == 0 else float(target_values.mean().cpu())
        )
        per_pair_background.append(
            None
            if background_values.numel() == 0
            else float(background_values.mean().cpu())
        )

    return {
        "clean_pair_count": len(pair_ids),
        "clean_pair_ids": list(pair_ids),
        "apto": {
            "total_loss": float(result["total"].cpu()),
            "plus_anchor_loss": float(result["plus_anchor_loss"].cpu()),
            "transition_loss": float(result["transition_loss"].cpu()),
        },
        "d_positive_delta": {
            "pixel_mean": float(d_values.mean().cpu()),
            "pixel_minimum": float(d_values.min().cpu()),
            "pair_macro_mean": float(per_pair_d.mean().cpu()),
            "pair_minimum_mean": float(per_pair_d.min().cpu()),
            "pair_fraction_mean_ge_0_25": float(
                (per_pair_d >= 0.25).to(torch.float32).mean().cpu()
            ),
        },
        "plus_anchor_target_probability": {
            "pixel_count": int(anchor_target_values.numel()),
            "status": (
                "evaluated"
                if anchor_target_values.numel()
                else "not_applicable_empty_R_plus"
            ),
            "pixel_mean": (
                None
                if anchor_target_values.numel() == 0
                else float(anchor_target_values.mean().cpu())
            ),
            "pixel_minimum": (
                None
                if anchor_target_values.numel() == 0
                else float(anchor_target_values.min().cpu())
            ),
        },
        "plus_anchor_background_probability": {
            "pixel_count": int(anchor_background_values.numel()),
            "status": (
                "evaluated"
                if anchor_background_values.numel()
                else "not_applicable_empty_B_plus"
            ),
            "pixel_mean": (
                None
                if anchor_background_values.numel() == 0
                else float(anchor_background_values.mean().cpu())
            ),
            "pixel_maximum": (
                None
                if anchor_background_values.numel() == 0
                else float(anchor_background_values.max().cpu())
            ),
        },
        "zero_response_delta": {
            "pixel_count": int(zero_values.numel()),
            "pixel_mean_abs": float(zero_values.mean().cpu()),
            "pixel_maximum_abs": float(zero_values.max().cpu()),
            "pair_macro_mean_abs": float(per_pair_zero.mean().cpu()),
        },
        "per_pair": [
            {
                "pair_id": pair_id,
                "d_positive_mean_delta": float(per_pair_d[index].cpu()),
                "zero_response_mean_abs_delta": float(
                    per_pair_zero[index].cpu()
                ),
                "plus_anchor_target_probability_mean": per_pair_target[index],
                "plus_anchor_background_probability_mean": (
                    per_pair_background[index]
                ),
                "apto_total_loss": float(result["per_pair_total"][index].cpu()),
                "plus_anchor_loss": float(
                    result["per_pair_plus_anchor"][index].cpu()
                ),
                "transition_loss": float(
                    result["per_pair_transition"][index].cpu()
                ),
            }
            for index, pair_id in enumerate(pair_ids)
        ],
    }


def _materialize_bound_pairs(
    materializer: PairedTransitionInputMaterializer,
    pair_ids: tuple[str, ...],
    *,
    device: torch.device,
) -> AnchoredPairBatch:
    """Stack already verified bindings without re-hashing the population.

    The executor verifies the complete materializer immediately before and
    after the optimizer loop. Repeating the full population hash for every
    two-pair update would be redundant and dominate bounded computation.
    """

    if (
        not isinstance(pair_ids, tuple)
        or not pair_ids
        or len(set(pair_ids)) != len(pair_ids)
    ):
        raise ValueError("pair_ids must be a non-empty unique tuple")
    missing = tuple(
        pair_id for pair_id in pair_ids if pair_id not in materializer.pair_by_id
    )
    if missing:
        raise KeyError(f"unknown clean pair IDs: {missing}")
    return stack_anchored_pair_examples(
        tuple(materializer.pair_by_id[pair_id] for pair_id in pair_ids),
        gt_union_by_sample=materializer.gt_union_by_sample,
        device=device,
    )


def _null_metrics(
    decoder: CURELiteDecoder,
    population: BoundedMicroPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    if torch.is_grad_enabled():
        raise RuntimeError("null diagnostics must execute under torch.no_grad")
    output: dict[str, object] = {}
    for name, pairs in (
        ("component_null", population.component_null),
        ("identity_null", population.identity_null),
    ):
        batch = stack_pair_examples(pairs, device=device)
        result = diagnose_null_pairs(decoder, batch)
        mean_abs = result["per_pair_mean_abs_delta"]
        max_abs = result["per_pair_max_abs_delta"]
        rms = result["per_pair_rms_delta"]
        output[name] = {
            "pair_count": len(pairs),
            "macro_mean_abs_delta": float(mean_abs.mean().cpu()),
            "maximum_abs_delta": float(max_abs.max().cpu()),
            "macro_rms_delta": float(rms.mean().cpu()),
            "per_pair": [
                {
                    "pair_id": pair_id,
                    "mean_abs_delta": float(mean_abs[index].cpu()),
                    "maximum_abs_delta": float(max_abs[index].cpu()),
                    "rms_delta": float(rms[index].cpu()),
                }
                for index, pair_id in enumerate(batch.pair_ids)
            ],
            "diagnostic_only": True,
            "autograd_enabled": False,
            "optimizer_exposure_count": 0,
        }
    return output


def _evaluate_snapshot(
    decoder: CURELiteDecoder,
    population: BoundedMicroPopulation,
    materializer: PairedTransitionInputMaterializer,
    criterion: AnchoredTransitionLoss,
    *,
    device: torch.device,
) -> dict[str, object]:
    decoder.eval()
    pair_ids = tuple(pair.pair_id for pair in population.clean_pairs)
    with torch.no_grad():
        clean = _clean_metrics(
            decoder,
            materializer,
            pair_ids,
            criterion,
            device=device,
        )
        nulls = _null_metrics(decoder, population, device=device)
    return {"clean": clean, "nulls": nulls}


def _factual_batches(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    update: int,
    *,
    device: torch.device,
) -> dict[str, object]:
    return {
        "factual_miss": stack_state_examples(
            tuple(
                population.factual_miss[index]
                for index in schedule.factual_miss_indices[update]
            ),
            device=device,
        ),
        "factual_no_miss": stack_state_examples(
            tuple(
                population.factual_no_miss[index]
                for index in schedule.factual_no_miss_indices[update]
            ),
            device=device,
        ),
    }


def _ratio(final: float, initial: float, *, name: str) -> float:
    if not isfinite(initial) or not isfinite(final) or initial <= 0.0:
        raise ValueError(f"{name} requires finite values and a positive denominator")
    return final / initial


def _computational_gates(
    initial: Mapping[str, object],
    final: Mapping[str, object],
) -> dict[str, object]:
    initial_clean = initial["clean"]
    final_clean = final["clean"]
    final_nulls = final["nulls"]
    if not all(
        isinstance(value, Mapping)
        for value in (initial_clean, final_clean, final_nulls)
    ):
        raise TypeError("APTO snapshot metrics are malformed")
    initial_apto = initial_clean["apto"]
    final_apto = final_clean["apto"]
    final_d = final_clean["d_positive_delta"]
    final_target = final_clean["plus_anchor_target_probability"]
    final_background = final_clean["plus_anchor_background_probability"]
    final_zero = final_clean["zero_response_delta"]
    component_null = final_nulls["component_null"]
    identity_null = final_nulls["identity_null"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            initial_apto,
            final_apto,
            final_d,
            final_target,
            final_background,
            final_zero,
            component_null,
            identity_null,
        )
    ):
        raise TypeError("APTO metric components are malformed")

    observed = {
        "apto_total_loss_final_over_initial": _ratio(
            float(final_apto["total_loss"]),
            float(initial_apto["total_loss"]),
            name="APTO total loss ratio",
        ),
        "plus_anchor_loss_final_over_initial": _ratio(
            float(final_apto["plus_anchor_loss"]),
            float(initial_apto["plus_anchor_loss"]),
            name="plus-anchor loss ratio",
        ),
        "transition_loss_final_over_initial": _ratio(
            float(final_apto["transition_loss"]),
            float(initial_apto["transition_loss"]),
            name="transition loss ratio",
        ),
        "d_positive_macro_mean_delta": float(final_d["pair_macro_mean"]),
        "d_positive_pair_fraction_ge_0_25": float(
            final_d["pair_fraction_mean_ge_0_25"]
        ),
        "plus_anchor_target_probability_mean": final_target["pixel_mean"],
        "plus_anchor_background_probability_mean": final_background["pixel_mean"],
        "zero_response_macro_mean_abs_delta": float(
            final_zero["pair_macro_mean_abs"]
        ),
        "component_null_macro_mean_abs_delta": float(
            component_null["macro_mean_abs_delta"]
        ),
        "identity_null_max_abs_delta": float(
            identity_null["maximum_abs_delta"]
        ),
    }
    rules = {
        "apto_total_loss_final_over_initial": (
            "max",
            "apto_total_loss_final_over_initial_max",
        ),
        "plus_anchor_loss_final_over_initial": (
            "max",
            "plus_anchor_loss_final_over_initial_max",
        ),
        "transition_loss_final_over_initial": (
            "max",
            "transition_loss_final_over_initial_max",
        ),
        "d_positive_macro_mean_delta": (
            "min",
            "d_positive_macro_mean_delta_min",
        ),
        "d_positive_pair_fraction_ge_0_25": (
            "min",
            "d_positive_pair_fraction_ge_0_25_min",
        ),
        "plus_anchor_target_probability_mean": (
            "min",
            "plus_anchor_target_probability_mean_min",
        ),
        "plus_anchor_background_probability_mean": (
            "max",
            "plus_anchor_background_probability_mean_max",
        ),
        "zero_response_macro_mean_abs_delta": (
            "max",
            "zero_response_macro_mean_abs_delta_max",
        ),
        "component_null_macro_mean_abs_delta": (
            "max",
            "component_null_macro_mean_abs_delta_max",
        ),
        "identity_null_max_abs_delta": (
            "max",
            "identity_null_max_abs_delta_max",
        ),
    }
    checks: dict[str, object] = {}
    for name, (direction, threshold_name) in rules.items():
        value = observed[name]
        threshold = COMPUTATIONAL_THRESHOLDS[threshold_name]
        if value is None:
            checks[name] = {
                "value": None,
                "direction": direction,
                "threshold": threshold,
                "applicable": False,
                "status": (
                    "NOT_APPLICABLE_EMPTY_MASKED_POPULATION"
                ),
                "pass": True,
            }
            continue
        value = float(value)
        if not isfinite(value):
            raise FloatingPointError(f"non-finite APTO gate value for {name}")
        checks[name] = {
            "value": value,
            "direction": direction,
            "threshold": threshold,
            "applicable": True,
            "status": "EVALUATED",
            "pass": value >= threshold if direction == "min" else value <= threshold,
        }
    return {
        "scope": "bounded_D_R_micro_population_computational_learnability",
        "not_detection_performance": True,
        "thresholds": dict(COMPUTATIONAL_THRESHOLDS),
        "observed": observed,
        "checks": checks,
        "all_pass": all(bool(check["pass"]) for check in checks.values()),
    }


def execute_paired_transition_bounded(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    materializer: PairedTransitionInputMaterializer,
    decoder_config: DecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    *,
    device: torch.device | str,
) -> dict[str, object]:
    """Train and audit a fresh APTO decoder on the sealed D_R micro-run."""

    target_device, seed, learning_rate, weight_decay = (
        _validate_execution_inputs(
            population,
            schedule,
            materializer,
            decoder_config,
            loss_config,
            optimization_budget,
            device,
        )
    )
    clean_pair_ids = tuple(pair.pair_id for pair in population.clean_pairs)
    cuda_devices: list[int] = []
    if target_device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device()
            if target_device.index is None
            else target_device.index
        ]

    with _deterministic_torch_runtime(
        target_device,
        DETERMINISM_SPECIFICATION,
    ) as deterministic_runtime, torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        decoder = CURELiteDecoder(decoder_config).to(target_device)
        absolute_criterion = CURELiteLoss(loss_config)
        transition_criterion = AnchoredTransitionLoss()
        transition_criterion.plus_anchor_criterion = CURELiteLoss(loss_config)
        transition_criterion.to(target_device)
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        parameter_count = sum(parameter.numel() for parameter in decoder.parameters())
        initial_decoder_fingerprint = decoder_state_fingerprint(decoder)
        initial_parameter_norm = sqrt(
            sum(
                float(parameter.detach().double().square().sum().cpu())
                for parameter in decoder.parameters()
            )
        )

        ledger = _ForwardLedger(decoder)
        pair_exposure: Counter[str] = Counter()
        miss_exposure: Counter[str] = Counter()
        no_miss_exposure: Counter[str] = Counter()
        trace: list[dict[str, object]] = []
        minimum_gradient_norm = float("inf")
        maximum_gradient_norm = 0.0
        nonfinite_gradient_updates = 0
        zero_gradient_updates = 0
        optimizer_steps = 0
        backward_calls = 0
        try:
            before_initial = ledger.snapshot()
            initial = _evaluate_snapshot(
                decoder,
                population,
                materializer,
                transition_criterion,
                device=target_device,
            )
            after_initial = ledger.snapshot()
            initial_forward = {
                "calls": after_initial[0] - before_initial[0],
                "state_evaluations": after_initial[1] - before_initial[1],
            }

            training_start = ledger.snapshot()
            for update in range(schedule.optimizer_updates):
                pair_indices = schedule.pair_indices[update]
                pair_ids = tuple(
                    population.clean_pairs[index].pair_id
                    for index in pair_indices
                )
                miss_indices = schedule.factual_miss_indices[update]
                no_miss_indices = schedule.factual_no_miss_indices[update]
                before_update = ledger.snapshot()
                logs = anchored_transition_train_step(
                    decoder,
                    absolute_criterion,
                    transition_criterion,
                    optimizer,
                    _factual_batches(
                        population,
                        schedule,
                        update,
                        device=target_device,
                    ),
                    _materialize_bound_pairs(
                        materializer,
                        pair_ids,
                        device=target_device,
                    ),
                )
                squared_gradient_norm = sum(
                    float(parameter.grad.detach().double().square().sum().cpu())
                    for parameter in decoder.parameters()
                    if parameter.grad is not None
                )
                gradient_norm = sqrt(squared_gradient_norm)
                if not isfinite(gradient_norm):
                    nonfinite_gradient_updates += 1
                if gradient_norm <= 0.0:
                    zero_gradient_updates += 1
                minimum_gradient_norm = min(
                    minimum_gradient_norm,
                    gradient_norm,
                )
                maximum_gradient_norm = max(
                    maximum_gradient_norm,
                    gradient_norm,
                )
                optimizer_steps += int(logs["optimizer_steps"])
                backward_calls += 1
                pair_exposure.update(pair_ids)
                miss_exposure.update(
                    population.factual_miss_ids[index]
                    for index in miss_indices
                )
                no_miss_exposure.update(
                    population.factual_no_miss_ids[index]
                    for index in no_miss_indices
                )
                after_update = ledger.snapshot()
                trace.append(
                    {
                        "update": update,
                        "epoch": update // schedule.steps_per_epoch,
                        "step": update % schedule.steps_per_epoch,
                        "clean_pair_ids": list(pair_ids),
                        "factual_miss_ids": [
                            population.factual_miss_ids[index]
                            for index in miss_indices
                        ],
                        "factual_no_miss_ids": [
                            population.factual_no_miss_ids[index]
                            for index in no_miss_indices
                        ],
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
            training_end = ledger.snapshot()
            training_forward = {
                "calls": training_end[0] - training_start[0],
                "state_evaluations": training_end[1] - training_start[1],
            }

            # Second and final full-population verification. The training hot
            # path above consumes only the already sealed bindings.
            materializer.verify_unchanged()
            before_final = ledger.snapshot()
            final = _evaluate_snapshot(
                decoder,
                population,
                materializer,
                transition_criterion,
                device=target_device,
            )
            after_final = ledger.snapshot()
            final_forward = {
                "calls": after_final[0] - before_final[0],
                "state_evaluations": after_final[1] - before_final[1],
            }
            total_forward = {
                "calls": after_final[0],
                "state_evaluations": after_final[1],
            }
        finally:
            ledger.close()

        final_decoder_fingerprint = decoder_state_fingerprint(decoder)
        final_parameter_norm = sqrt(
            sum(
                float(parameter.detach().double().square().sum().cpu())
                for parameter in decoder.parameters()
            )
        )

    exposure = {
        "clean_pairs": [
            {
                "pair_id": pair.pair_id,
                "sample_id": pair.sample_id,
                "count": pair_exposure[pair.pair_id],
            }
            for pair in population.clean_pairs
        ],
        "factual_miss": [
            {
                "anchor_id": anchor_id,
                "sample_id": example.sample_id,
                "count": miss_exposure[anchor_id],
            }
            for anchor_id, example in zip(
                population.factual_miss_ids,
                population.factual_miss,
                strict=True,
            )
        ],
        "factual_no_miss": [
            {
                "anchor_id": anchor_id,
                "sample_id": example.sample_id,
                "count": no_miss_exposure[anchor_id],
            }
            for anchor_id, example in zip(
                population.factual_no_miss_ids,
                population.factual_no_miss,
                strict=True,
            )
        ],
        "component_null_optimizer_exposure": 0,
        "identity_null_optimizer_exposure": 0,
    }
    per_update_forward_exact = all(
        row["decoder_forward_calls"] == 3
        and row["decoder_state_evaluations"] == 12
        for row in trace
    )
    actual_pair_counts = tuple(
        pair_exposure[pair_id] for pair_id in clean_pair_ids
    )
    actual_miss_counts = tuple(
        miss_exposure[anchor_id] for anchor_id in population.factual_miss_ids
    )
    actual_no_miss_counts = tuple(
        no_miss_exposure[anchor_id]
        for anchor_id in population.factual_no_miss_ids
    )
    expected_snapshot_forward = {"calls": 3, "state_evaluations": 96}
    expected_training_forward = {
        "calls": 3 * schedule.optimizer_updates,
        "state_evaluations": 12 * schedule.optimizer_updates,
    }
    expected_total_forward = {
        "calls": expected_training_forward["calls"] + 6,
        "state_evaluations": (
            expected_training_forward["state_evaluations"] + 192
        ),
    }
    structural_checks = {
        "deterministic_runtime_contract_satisfied": (
            deterministic_runtime["contract_satisfied"] is True
            and deterministic_runtime["flags_restored_after_execution"] is True
        ),
        "micro_population_counts_exact": all(
            len(getattr(population, name)) == 16
            for name in (
                "clean_pairs",
                "factual_miss",
                "factual_no_miss",
                "component_null",
                "identity_null",
            )
        ),
        "materializer_binding_verified": (
            set(clean_pair_ids) <= set(materializer.canonical_pair_ids)
        ),
        "all_clean_pairs_evaluated_initial": (
            initial["clean"]["clean_pair_ids"] == list(clean_pair_ids)
        ),
        "all_clean_pairs_evaluated_final": (
            final["clean"]["clean_pair_ids"] == list(clean_pair_ids)
        ),
        "all_optimizer_updates_completed": (
            len(trace) == schedule.optimizer_updates
        ),
        "one_backward_per_update": backward_calls == schedule.optimizer_updates,
        "one_optimizer_step_per_update": (
            optimizer_steps == schedule.optimizer_updates
        ),
        "all_gradients_finite": nonfinite_gradient_updates == 0,
        "every_update_total_gradient_norm_positive": zero_gradient_updates == 0,
        "decoder_parameters_changed": (
            final_decoder_fingerprint != initial_decoder_fingerprint
        ),
        "training_forward_budget_exact": (
            training_forward == expected_training_forward
            and per_update_forward_exact
        ),
        "evaluation_forward_budget_exact": (
            initial_forward == expected_snapshot_forward
            and final_forward == expected_snapshot_forward
        ),
        "total_forward_budget_exact": total_forward == expected_total_forward,
        "schedule_exposure_ledgers_exact": (
            actual_pair_counts == schedule.pair_counts
            and actual_miss_counts == schedule.factual_miss_counts
            and actual_no_miss_counts == schedule.factual_no_miss_counts
        ),
        "every_clean_pair_exposed_to_optimizer": min(actual_pair_counts) > 0,
        "nulls_excluded_from_optimizer": (
            exposure["component_null_optimizer_exposure"] == 0
            and exposure["identity_null_optimizer_exposure"] == 0
        ),
        "nulls_diagnosed_without_autograd": all(
            snapshot["nulls"][name]["autograd_enabled"] is False
            for snapshot in (initial, final)
            for name in ("component_null", "identity_null")
        ),
    }
    structural_execution_pass = all(structural_checks.values())
    computational = _computational_gates(initial, final)
    computational_pass = (
        structural_execution_pass and computational["all_pass"] is True
    )
    decision = (
        "BOUNDED_COMPUTATIONAL_LEARNABILITY_PASS"
        if computational_pass
        else (
            "STRUCTURAL_EXECUTION_FAIL"
            if not structural_execution_pass
            else "BOUNDED_COMPUTATIONAL_LEARNABILITY_FAIL"
        )
    )
    result: dict[str, object] = {
        "schema_version": PAIRED_TRANSITION_BOUNDED_SCHEMA,
        "execution_status": "completed",
        "decision": decision,
        "device": str(target_device),
        "population_fingerprint": population.population_fingerprint,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "materializer_fingerprint": materializer.materializer_fingerprint,
        "decoder_config": asdict(decoder_config),
        "loss_config": asdict(loss_config),
        "optimization_budget": dict(optimization_budget),
        "optimizer_updates_completed": len(trace),
        "initial": initial,
        "final": final,
        "computational_gates": computational,
        "structural_checks": structural_checks,
        "structural_execution_pass": structural_execution_pass,
        "computational_learnability_pass": computational_pass,
        "parameters": {
            "trainable_parameter_count": parameter_count,
            "initial_decoder_fingerprint": initial_decoder_fingerprint,
            "final_decoder_fingerprint": final_decoder_fingerprint,
            "initial_l2_norm": initial_parameter_norm,
            "final_l2_norm": final_parameter_norm,
        },
        "gradients": {
            "minimum_update_l2_norm": minimum_gradient_norm,
            "maximum_update_l2_norm": maximum_gradient_norm,
            "nonfinite_updates": nonfinite_gradient_updates,
            "zero_norm_updates": zero_gradient_updates,
        },
        "execution_ledger": {
            "backward_calls": backward_calls,
            "optimizer_steps": optimizer_steps,
            "expected_backward_calls": schedule.optimizer_updates,
            "expected_optimizer_steps": schedule.optimizer_updates,
        },
        "forward_budget": {
            "initial_evaluation": initial_forward,
            "training": training_forward,
            "final_evaluation": final_forward,
            "total": total_forward,
            "expected_initial_evaluation": expected_snapshot_forward,
            "expected_training": expected_training_forward,
            "expected_final_evaluation": expected_snapshot_forward,
            "expected_total": expected_total_forward,
        },
        "deterministic_runtime": deterministic_runtime,
        "exposure": exposure,
        "trace": trace,
        "interpretation": {
            "evidence_scope": (
                "fresh_decoder_bounded_D_R_micro_population_learnability"
            ),
            "not_detection_performance_evidence": True,
            "does_not_establish_Pd_or_FA": True,
            "does_not_authorize_formal_training": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "base_or_backbone_updated": False,
            "component_null_optimizer_exposure": 0,
            "identity_null_optimizer_exposure": 0,
        },
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


__all__ = [
    "COMPUTATIONAL_THRESHOLDS",
    "DETERMINISM_SPECIFICATION",
    "PAIRED_TRANSITION_BOUNDED_SCHEMA",
    "execute_paired_transition_bounded",
]
