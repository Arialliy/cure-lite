"""Formal 800 x 40 training engine for paired CURE-Lite methods.

This module is deliberately narrower than an experiment runner.  It consumes
one already-frozen :class:`PairedFormalSchedule`, one externally constructed
fresh decoder/Adam state, and the existing paired-difference or
matched-control train step.  It has no checkpoint, resume, horizon,
batch-size, coefficient, or evaluation-split interface.

The primary method is serialized as ``paired_difference`` to match the formal
paired artifact schema; the remaining eight names are the frozen matched
controls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isclose, isfinite, sqrt
from typing import Protocol

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..decoder import CURELiteDecoder
from ..losses import CURELiteLoss
from ..paired_losses import PairedDifferenceLoss
from ..paired_types import PairBatch, PairExample
from ..train.paired_control_step import (
    CONTROL_KINDS,
    paired_control_train_step,
)
from ..train.paired_pools import (
    PAIRED_EPOCHS,
    PAIRED_OPTIMIZER_UPDATES,
    PAIRED_STEPS_PER_EPOCH,
)
from ..train.paired_step import paired_train_step
from .artifacts import decoder_state_fingerprint
from .paired_artifacts import (
    PAIRED_METHODS,
    PairedDecoderRunConfig,
    PairedExecutionLedger,
)
from .paired_formal_controls import PairedFormalControlInputProvider
from .paired_formal_schedule import (
    DECODER_FORWARDS_PER_UPDATE,
    PairedFormalSchedule,
    formal_batches_for_update,
)


PAIRED_FORMAL_EXPOSURE_SCHEMA = (
    "cure-lite-paired-formal-exposure-fingerprint-v1"
)
PAIRED_DIFFERENCE_METHOD = "paired_difference"
FORMAL_TRAINING_METHODS = (PAIRED_DIFFERENCE_METHOD, *CONTROL_KINDS)

_CONTROL_KWARG_KEYS: dict[str, frozenset[str]] = {
    "independent_endpoint": frozenset(
        {"gt_union", "completion_plus", "completion_minus"}
    ),
    "after_only": frozenset({"gt_union"}),
    "zero_feature": frozenset(),
    "coordinate_basis": frozenset({"coordinate_basis"}),
    "feature_only": frozenset(),
    "target_permutation": frozenset({"permuted_label_increment"}),
    "plus_detach": frozenset(),
    "minus_detach": frozenset(),
}


class FormalControlKwargsProvider(Protocol):
    """Materialize only the pre-frozen inputs required by one control."""

    def __call__(
        self,
        *,
        control_kind: str,
        pairs: tuple[PairExample, PairExample],
        pair_batch: PairBatch,
        epoch: int,
        step: int,
        device: torch.device,
    ) -> Mapping[str, object]:
        ...


@dataclass(frozen=True)
class PairedFormalTrainingResult:
    """In-memory result that can be passed to the create-only artifact writer."""

    epoch_logs: tuple[dict[str, object], ...]
    execution_ledger: PairedExecutionLedger

    def __post_init__(self) -> None:
        if len(self.epoch_logs) != PAIRED_EPOCHS:
            raise ValueError("formal training result requires 800 epoch logs")
        if not isinstance(self.execution_ledger, PairedExecutionLedger):
            raise TypeError("execution_ledger must be PairedExecutionLedger")


@dataclass(frozen=True)
class _UpdateOutcome:
    logs: Mapping[str, float | int | str]
    gradient_l2_norm: float


class _DecoderForwardLedger:
    """Count actual decoder calls and state evaluations during training only."""

    def __init__(self, decoder: CURELiteDecoder) -> None:
        self.calls = 0
        self.states = 0
        self._handle = decoder.register_forward_hook(self._hook)

    def _hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise RuntimeError("formal decoder forward returned an invalid output")
        self.calls += 1
        self.states += int(output.shape[0])

    def snapshot(self) -> tuple[int, int]:
        return self.calls, self.states

    def close(self) -> None:
        self._handle.remove()


def _exposure_payloads(
    schedule: PairedFormalSchedule,
) -> dict[str, dict[str, object]]:
    common = {
        "schema_version": PAIRED_FORMAL_EXPOSURE_SCHEMA,
        "formal_schedule_fingerprint": schedule.schedule_fingerprint,
        "combined_sequence_fingerprint": (
            schedule.combined_sequence_fingerprint
        ),
    }
    return {
        "pair": {
            **common,
            "branch": "clean_pair",
            "paired_schedule_fingerprint": (
                schedule.paired_schedule.schedule_fingerprint
            ),
            "sequence_fingerprint": schedule.pair_sequence_fingerprint,
            "identities": [
                {
                    "pair_id": pair.pair_id,
                    "sample_id": pair.sample_id,
                    "count": schedule.pair_exposure_counts[index],
                }
                for index, pair in enumerate(
                    schedule.paired_schedule.pairs
                )
            ],
        },
        "factual_miss": {
            **common,
            "branch": "factual_miss",
            "sequence_fingerprint": (
                schedule.factual_miss_sequence_fingerprint
            ),
            "identities": [
                {
                    **anchor.canonical_payload(),
                    "count": schedule.factual_miss_exposure_counts[index],
                }
                for index, anchor in enumerate(
                    schedule.factual_miss_anchors
                )
            ],
        },
        "factual_no_miss": {
            **common,
            "branch": "factual_no_miss",
            "sequence_fingerprint": (
                schedule.factual_no_miss_sequence_fingerprint
            ),
            "identities": [
                {
                    **anchor.canonical_payload(),
                    "count": (
                        schedule.factual_no_miss_exposure_counts[index]
                    ),
                }
                for index, anchor in enumerate(
                    schedule.factual_no_miss_anchors
                )
            ],
        },
    }


def formal_exposure_fingerprints(
    schedule: PairedFormalSchedule,
) -> dict[str, str]:
    """Bind each ledger fingerprint to identities, counts, and sequence."""

    if not isinstance(schedule, PairedFormalSchedule):
        raise TypeError("schedule must be a PairedFormalSchedule")
    return {
        branch: stable_fingerprint(payload)
        for branch, payload in _exposure_payloads(schedule).items()
    }


def _decoder_device(decoder: CURELiteDecoder) -> torch.device:
    parameters = tuple(decoder.parameters())
    if not parameters:
        raise ValueError("formal decoder has no trainable parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("formal decoder parameters must share device and dtype")
    if not next(iter(dtypes)).is_floating_point:
        raise TypeError("formal decoder parameters must be floating point")
    return next(iter(devices))


def _validate_fresh_adam(
    decoder: CURELiteDecoder,
    optimizer: torch.optim.Optimizer,
    config: PairedDecoderRunConfig,
) -> None:
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError("formal paired optimizer must be a fresh torch.optim.Adam")
    if optimizer.state:
        raise ValueError("formal paired optimizer state is not fresh; resume is forbidden")
    parameters = tuple(decoder.parameters())
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("every formal decoder parameter must remain trainable")
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("fresh formal decoder parameters cannot carry gradients")
    if len(optimizer.param_groups) != 1:
        raise ValueError("formal paired Adam must contain exactly one parameter group")
    group = optimizer.param_groups[0]
    group_parameters = tuple(group["params"])
    if (
        len(group_parameters) != len(parameters)
        or len({id(parameter) for parameter in group_parameters})
        != len(parameters)
        or {id(parameter) for parameter in group_parameters}
        != {id(parameter) for parameter in parameters}
    ):
        raise ValueError("formal paired Adam must own every decoder parameter once")

    expected = {
        "lr": config.learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": config.weight_decay,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }
    if any(group.get(name) != value for name, value in expected.items()):
        raise ValueError("formal paired Adam options differ from the frozen defaults")


def _validate_formal_inputs(
    decoder: CURELiteDecoder,
    absolute_criterion: CURELiteLoss,
    paired_criterion: PairedDifferenceLoss,
    optimizer: torch.optim.Optimizer,
    schedule: PairedFormalSchedule,
    config: PairedDecoderRunConfig,
    control_kwargs_provider: FormalControlKwargsProvider | None,
) -> torch.device:
    if not isinstance(config, PairedDecoderRunConfig):
        raise TypeError("config must be PairedDecoderRunConfig")
    if type(decoder) is not CURELiteDecoder:
        raise TypeError("formal decoder must be exactly CURELiteDecoder")
    if not isinstance(absolute_criterion, CURELiteLoss):
        raise TypeError("absolute_criterion must be CURELiteLoss")
    if absolute_criterion.config != config.absolute_loss_config:
        raise ValueError("absolute criterion differs from the frozen run config")
    if type(paired_criterion) is not PairedDifferenceLoss:
        raise TypeError("paired_criterion must be exactly PairedDifferenceLoss")
    if not isinstance(schedule, PairedFormalSchedule):
        raise TypeError("schedule must be a PairedFormalSchedule")
    if config.method not in FORMAL_TRAINING_METHODS or (
        config.method not in PAIRED_METHODS
    ):
        raise ValueError("formal paired method is invalid")
    if (
        config.seed != schedule.seed
        or config.formal_schedule_fingerprint
        != schedule.schedule_fingerprint
        or config.paired_schedule_fingerprint
        != schedule.paired_schedule.schedule_fingerprint
        or config.pair_catalog_fingerprint
        != schedule.paired_schedule.catalog_fingerprint
    ):
        raise ValueError("formal run config does not bind the supplied schedule")
    if (
        schedule.optimizer_updates != PAIRED_OPTIMIZER_UPDATES
        or schedule.decoder_forward_calls
        != PAIRED_OPTIMIZER_UPDATES * DECODER_FORWARDS_PER_UPDATE
        or schedule.decoder_state_evaluations != 384_000
    ):
        raise ValueError("formal schedule compute budget changed")
    if decoder.config != config.decoder_config:
        raise ValueError("decoder architecture differs from the frozen run config")
    if (
        decoder_state_fingerprint(decoder)
        != config.initial_decoder_fingerprint
    ):
        raise ValueError("decoder does not match the externally frozen initial state")
    if config.method == PAIRED_DIFFERENCE_METHOD:
        if control_kwargs_provider is not None:
            raise ValueError(
                "paired_difference training cannot receive control inputs"
            )
    else:
        if not isinstance(
            control_kwargs_provider,
            PairedFormalControlInputProvider,
        ):
            raise TypeError(
                "every formal matched control requires the frozen "
                "PairedFormalControlInputProvider"
            )
        control_kwargs_provider.verify_unchanged()
        if (
            control_kwargs_provider.provider_fingerprint
            != config.control_provider_fingerprint
        ):
            raise ValueError(
                "formal control provider fingerprint differs from run config"
            )
    _validate_fresh_adam(decoder, optimizer, config)
    return _decoder_device(decoder)


def _validate_completed_adam(
    decoder: CURELiteDecoder,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Prove that Adam advanced every decoder parameter exactly 32,000 times."""

    parameters = tuple(decoder.parameters())
    if len(optimizer.state) != len(parameters):
        raise RuntimeError("formal Adam state does not cover every decoder parameter")
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping) or set(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RuntimeError(
                "formal Adam state must contain exactly step/exp_avg/exp_avg_sq"
            )
        step = state["step"]
        if isinstance(step, Tensor):
            if step.numel() != 1 or not torch.isfinite(step).all():
                raise RuntimeError("formal Adam step counter is invalid")
            step_value = float(step.detach().cpu())
        elif isinstance(step, (int, float)) and not isinstance(step, bool):
            step_value = float(step)
        else:
            raise TypeError("formal Adam step counter has an invalid type")
        if step_value != float(PAIRED_OPTIMIZER_UPDATES):
            raise RuntimeError("formal Adam did not complete exactly 32,000 steps")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            if (
                not isinstance(value, Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.device != parameter.device
            ):
                raise RuntimeError(
                    f"formal Adam state {name!r} does not match its parameter"
                )
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"formal Adam state {name!r} became non-finite"
                )


def _selected_pairs(
    schedule: PairedFormalSchedule,
    *,
    epoch: int,
    step: int,
) -> tuple[PairExample, PairExample]:
    update = epoch * PAIRED_STEPS_PER_EPOCH + step
    first, second = schedule.paired_schedule.batch_pair_indices[update]
    return (
        schedule.paired_schedule.pairs[first],
        schedule.paired_schedule.pairs[second],
    )


def _control_kwargs_for_update(
    provider: FormalControlKwargsProvider,
    *,
    control_kind: str,
    pairs: tuple[PairExample, PairExample],
    pair_batch: PairBatch,
    epoch: int,
    step: int,
    device: torch.device,
) -> dict[str, object]:
    result = provider(
        control_kind=control_kind,
        pairs=pairs,
        pair_batch=pair_batch,
        epoch=epoch,
        step=step,
        device=device,
    )
    if not isinstance(result, Mapping):
        raise TypeError("formal control input provider must return a mapping")
    normalized = dict(result)
    if set(normalized) != _CONTROL_KWARG_KEYS[control_kind]:
        raise ValueError(
            f"{control_kind} control inputs differ from the frozen key set"
        )
    return normalized


def _gradient_l2_norm(decoder: CURELiteDecoder) -> float:
    parameters = tuple(decoder.parameters())
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("every formal decoder parameter must receive a gradient")
    squared = fsum(
        float(parameter.grad.detach().double().square().sum().cpu())
        for parameter in parameters
    )
    result = sqrt(squared)
    if not isfinite(result):
        raise FloatingPointError("formal decoder gradient norm is non-finite")
    if result <= 0.0:
        raise RuntimeError(
            "formal decoder gradient norm must be positive on every update"
        )
    return result


def _validate_step_logs(
    logs: Mapping[str, float | int | str],
    *,
    method: str,
) -> None:
    paired_key = (
        "paired/loss"
        if method == PAIRED_DIFFERENCE_METHOD
        else "control/loss"
    )
    required = (
        "total",
        "factual_miss/loss",
        "factual_no_miss/loss",
        paired_key,
        "optimizer_steps",
    )
    if any(name not in logs for name in required):
        raise RuntimeError("formal train step omitted a required ledger value")
    numeric = tuple(float(logs[name]) for name in required[:-1])
    if not all(isfinite(value) for value in numeric):
        raise FloatingPointError("formal train step emitted a non-finite loss")
    if logs["optimizer_steps"] != 1:
        raise RuntimeError("formal train step optimizer budget changed")
    if not isclose(
        numeric[0],
        numeric[1] + numeric[2] + numeric[3],
        rel_tol=2e-6,
        abs_tol=2e-7,
    ):
        raise RuntimeError("formal objective no longer uses frozen 1:1:1 terms")


def _execute_one_update(
    decoder: CURELiteDecoder,
    absolute_criterion: CURELiteLoss,
    paired_criterion: PairedDifferenceLoss,
    optimizer: torch.optim.Optimizer,
    schedule: PairedFormalSchedule,
    *,
    method: str,
    epoch: int,
    step: int,
    device: torch.device,
    control_kwargs_provider: FormalControlKwargsProvider | None,
    forward_ledger: _DecoderForwardLedger,
) -> _UpdateOutcome:
    factual_batches, pair_batch = formal_batches_for_update(
        schedule,
        epoch=epoch,
        step=step,
        device=device,
    )
    before = forward_ledger.snapshot()
    if method == PAIRED_DIFFERENCE_METHOD:
        logs = paired_train_step(
            decoder,
            absolute_criterion,
            paired_criterion,
            optimizer,
            factual_batches,
            pair_batch,
        )
    else:
        if control_kwargs_provider is None:
            raise AssertionError("validated formal control provider disappeared")
        pairs = _selected_pairs(schedule, epoch=epoch, step=step)
        control_kwargs = _control_kwargs_for_update(
            control_kwargs_provider,
            control_kind=method,
            pairs=pairs,
            pair_batch=pair_batch,
            epoch=epoch,
            step=step,
            device=device,
        )
        logs = paired_control_train_step(
            decoder,
            absolute_criterion,
            paired_criterion,
            optimizer,
            factual_batches,
            pair_batch,
            control_kind=method,
            **control_kwargs,
        )
    after = forward_ledger.snapshot()
    if after[0] - before[0] != 3 or after[1] - before[1] != 12:
        raise RuntimeError("formal per-update decoder compute budget changed")
    _validate_step_logs(logs, method=method)
    gradient_norm = _gradient_l2_norm(decoder)
    if any(
        not torch.isfinite(parameter.detach()).all()
        for parameter in decoder.parameters()
    ):
        raise FloatingPointError("formal decoder parameter became non-finite")
    return _UpdateOutcome(
        logs=logs,
        gradient_l2_norm=gradient_norm,
    )


def _epoch_log(
    epoch: int,
    outcomes: list[_UpdateOutcome],
    *,
    method: str,
) -> dict[str, object]:
    if len(outcomes) != PAIRED_STEPS_PER_EPOCH:
        raise RuntimeError("formal epoch did not contain exactly 40 updates")
    paired_key = (
        "paired/loss"
        if method == PAIRED_DIFFERENCE_METHOD
        else "control/loss"
    )
    total = [float(outcome.logs["total"]) for outcome in outcomes]
    miss = [
        float(outcome.logs["factual_miss/loss"])
        for outcome in outcomes
    ]
    no_miss = [
        float(outcome.logs["factual_no_miss/loss"])
        for outcome in outcomes
    ]
    paired_or_control = [
        float(outcome.logs[paired_key]) for outcome in outcomes
    ]
    divisor = float(PAIRED_STEPS_PER_EPOCH)
    return {
        "epoch": epoch,
        "steps": PAIRED_STEPS_PER_EPOCH,
        "metrics": {
            "mean_total_loss": fsum(total) / divisor,
            "mean_factual_miss_loss": fsum(miss) / divisor,
            "mean_factual_no_miss_loss": fsum(no_miss) / divisor,
            "mean_paired_or_control_loss": (
                fsum(paired_or_control) / divisor
            ),
            "minimum_total_loss": min(total),
            "maximum_total_loss": max(total),
        },
    }


def execute_paired_formal_training(
    decoder: CURELiteDecoder,
    absolute_criterion: CURELiteLoss,
    paired_criterion: PairedDifferenceLoss,
    optimizer: torch.optim.Optimizer,
    schedule: PairedFormalSchedule,
    config: PairedDecoderRunConfig,
    *,
    control_kwargs_provider: FormalControlKwargsProvider | None = None,
) -> PairedFormalTrainingResult:
    """Execute one complete create-only paired-difference/control formal run.

    The caller must construct a fresh decoder and fresh Adam from the same
    externally frozen initial state for every method at one seed.  This
    function never loads or writes a checkpoint and cannot execute a partial
    horizon.
    """

    device = _validate_formal_inputs(
        decoder,
        absolute_criterion,
        paired_criterion,
        optimizer,
        schedule,
        config,
        control_kwargs_provider,
    )
    initial_fingerprint = decoder_state_fingerprint(decoder)
    parameter_count = sum(
        parameter.numel() for parameter in decoder.parameters()
    )
    epoch_logs: list[dict[str, object]] = []
    minimum_gradient = float("inf")
    maximum_gradient = 0.0
    completed_updates = 0
    optimizer_steps = 0
    forward_ledger = _DecoderForwardLedger(decoder)
    try:
        for epoch in range(PAIRED_EPOCHS):
            outcomes: list[_UpdateOutcome] = []
            for step in range(PAIRED_STEPS_PER_EPOCH):
                outcome = _execute_one_update(
                    decoder,
                    absolute_criterion,
                    paired_criterion,
                    optimizer,
                    schedule,
                    method=config.method,
                    epoch=epoch,
                    step=step,
                    device=device,
                    control_kwargs_provider=control_kwargs_provider,
                    forward_ledger=forward_ledger,
                )
                outcomes.append(outcome)
                minimum_gradient = min(
                    minimum_gradient,
                    outcome.gradient_l2_norm,
                )
                maximum_gradient = max(
                    maximum_gradient,
                    outcome.gradient_l2_norm,
                )
                completed_updates += 1
                optimizer_steps += int(outcome.logs["optimizer_steps"])
            epoch_logs.append(
                _epoch_log(epoch, outcomes, method=config.method)
            )
    finally:
        forward_ledger.close()

    if (
        completed_updates != PAIRED_OPTIMIZER_UPDATES
        or optimizer_steps != PAIRED_OPTIMIZER_UPDATES
        or len(epoch_logs) != PAIRED_EPOCHS
        or forward_ledger.calls != schedule.decoder_forward_calls
        or forward_ledger.states != schedule.decoder_state_evaluations
    ):
        raise RuntimeError("formal paired execution budget is incomplete")
    _validate_completed_adam(decoder, optimizer)
    if isinstance(
        control_kwargs_provider,
        PairedFormalControlInputProvider,
    ):
        control_kwargs_provider.verify_unchanged()
    final_fingerprint = decoder_state_fingerprint(decoder)
    if final_fingerprint == initial_fingerprint:
        raise RuntimeError("formal paired training did not change decoder parameters")
    if not isfinite(minimum_gradient) or not isfinite(maximum_gradient):
        raise FloatingPointError("formal paired gradient ledger is non-finite")

    exposure = formal_exposure_fingerprints(schedule)
    ledger = PairedExecutionLedger(
        method=config.method,
        seed=config.seed,
        formal_schedule_fingerprint=schedule.schedule_fingerprint,
        runtime_input_fingerprint=config.runtime_input_fingerprint,
        control_provider_fingerprint=(
            config.control_provider_fingerprint
        ),
        pair_exposure_fingerprint=exposure["pair"],
        factual_miss_exposure_fingerprint=exposure["factual_miss"],
        factual_no_miss_exposure_fingerprint=exposure[
            "factual_no_miss"
        ],
        initial_decoder_fingerprint=initial_fingerprint,
        final_decoder_fingerprint=final_fingerprint,
        trainable_parameter_count=parameter_count,
        minimum_gradient_l2_norm=minimum_gradient,
        maximum_gradient_l2_norm=maximum_gradient,
    )
    if ledger.formal_schedule_fingerprint != schedule.schedule_fingerprint:
        raise AssertionError("execution ledger lost the formal schedule binding")
    return PairedFormalTrainingResult(
        epoch_logs=tuple(epoch_logs),
        execution_ledger=ledger,
    )


__all__ = [
    "FORMAL_TRAINING_METHODS",
    "PAIRED_FORMAL_EXPOSURE_SCHEMA",
    "PAIRED_DIFFERENCE_METHOD",
    "FormalControlKwargsProvider",
    "PairedFormalTrainingResult",
    "execute_paired_formal_training",
    "formal_exposure_fingerprints",
]
