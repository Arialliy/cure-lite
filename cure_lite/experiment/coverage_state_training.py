"""Objective-specific scalar CSLF training over one frozen schedule."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields as dataclass_fields
from math import isfinite
from typing import Any, Callable, Mapping

import torch

from ..cache.schema import stable_fingerprint
from ..coverage_state_device_cache import (
    CoverageStateDeviceCache,
    prepare_coverage_state_device_cache,
)
from ..coverage_state_level_set import CURELiteCoverageStateLevelSet
from ..coverage_state_level_set import CoverageStateLevelSetConfig
from ..coverage_state_phase_preserving import (
    CoverageStatePhasePreservingConfig,
    build_coverage_state_level_set,
)
from ..coverage_state_precomputed_cache import CoverageStateScalarCache
from ..coverage_state_schedule import (
    CoverageStateTrainingSchedule,
    coverage_state_schedule_exposure_report,
)
from ..paired_types import tensor_content_fingerprint
from ..train.coverage_state_fused_step import (
    COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES,
    COVERAGE_STATE_LEGACY_MATCHED_OBJECTIVES,
    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
    CoverageStatePairObjective,
    audit_coverage_state_training_state,
    coverage_state_fused_train_step,
    coverage_state_pair_objective_policy,
)


COVERAGE_STATE_TRAINING_RESULT_SCHEMA = (
    "cure-lite-cslf-training-result-v2"
)
COVERAGE_STATE_MATCHED_RESULT_SCHEMA = (
    "cure-lite-cslf-matched-training-result-v2"
)
COVERAGE_STATE_BOUNDED_SCOPE = "D_R_bounded_400"
COVERAGE_STATE_FORMAL_SCOPE = "D_R_formal_800"
COVERAGE_STATE_REGISTERED_MATCHED_OBJECTIVE_SUITES = (
    COVERAGE_STATE_LEGACY_MATCHED_OBJECTIVES,
    COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES,
    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
)


class CoverageStateRunAuthorization(ABC):
    """Nominal base type for prerequisite-bound protected-run approvals."""

    @abstractmethod
    def verify_for_run(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> None:
        """Raise unless this exact cache, schedule, and scope are approved."""


def coverage_state_model_fingerprint(
    model: CURELiteCoverageStateLevelSet,
) -> str:
    if not isinstance(model, CURELiteCoverageStateLevelSet):
        raise TypeError("model must be CURELiteCoverageStateLevelSet")
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-cslf-model-state-v1",
            "tensors": {
                name: tensor_content_fingerprint(value)
                for name, value in sorted(model.state_dict().items())
            },
        }
    )


def _canonical_optimizer_value(value: Any) -> object:
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical_optimizer_value(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(
        "optimizer configuration contains an unsupported value: "
        f"{type(value).__name__}"
    )


def coverage_state_optimizer_config_fingerprint(
    model: CURELiteCoverageStateLevelSet,
    optimizer: torch.optim.Optimizer,
) -> str:
    """Bind an optimizer policy without incorporating its mutable moments."""

    if not isinstance(model, CURELiteCoverageStateLevelSet):
        raise TypeError("model must be CURELiteCoverageStateLevelSet")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    parameter_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    groups: list[dict[str, object]] = []
    for group in optimizer.param_groups:
        identities = [id(value) for value in group.get("params", ())]
        if any(identity not in parameter_names for identity in identities):
            raise ValueError("optimizer contains a parameter outside the model")
        groups.append(
            {
                "parameters": [
                    parameter_names[identity] for identity in identities
                ],
                "options": {
                    name: _canonical_optimizer_value(value)
                    for name, value in sorted(group.items())
                    if name != "params"
                },
            }
        )
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-cslf-optimizer-config-v1",
            "class": (
                f"{optimizer.__class__.__module__}."
                f"{optimizer.__class__.__qualname__}"
            ),
            "groups": groups,
        }
    )


def coverage_state_model_contract_payload(
    model: CURELiteCoverageStateLevelSet,
) -> dict[str, object]:
    """Bind the exact structural class/configuration used by one objective."""

    if not isinstance(model, CURELiteCoverageStateLevelSet):
        raise TypeError("model must be CURELiteCoverageStateLevelSet")
    config = model.config
    if not isinstance(config, CoverageStateLevelSetConfig):
        raise TypeError("model config must be CoverageStateLevelSetConfig")
    return {
        "model_class": (
            f"{model.__class__.__module__}."
            f"{model.__class__.__qualname__}"
        ),
        "config_class": (
            f"{config.__class__.__module__}."
            f"{config.__class__.__qualname__}"
        ),
        "config": {
            field.name: _canonical_optimizer_value(
                getattr(config, field.name)
            )
            for field in dataclass_fields(config)
        },
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "parameter_shapes": {
            name: list(parameter.shape)
            for name, parameter in sorted(model.named_parameters())
        },
    }


@dataclass(frozen=True)
class CoverageStateTrainingResult:
    objective: str
    objective_policy: str
    seed: int
    epochs: int
    steps_per_epoch: int
    completed_updates: int
    schedule_fingerprint: str
    cache_fingerprint: str
    execution_device: str
    device_cache_fingerprint: str
    device_cache_resident_bytes: int
    optimizer_config_fingerprint: str
    initial_model_fingerprint: str
    final_model_fingerprint: str
    epoch_logs: tuple[dict[str, object], ...]
    first_nonzero_gradient_update: tuple[tuple[str, int], ...]
    forward_calls: int
    backward_calls: int
    optimizer_steps: int
    logical_state_evaluations: int
    finite_state_audits: int

    def __post_init__(self) -> None:
        expected = self.epochs * self.steps_per_epoch
        if (
            self.completed_updates != expected
            or len(self.epoch_logs) != self.epochs
            or self.forward_calls != expected
            or self.backward_calls != expected
            or self.optimizer_steps != expected
            or self.logical_state_evaluations != expected * 12
            or self.finite_state_audits != expected + 1
            or len(self.device_cache_fingerprint) != 64
            or self.device_cache_resident_bytes < 1
            or not self.execution_device
        ):
            raise ValueError("training result compute ledger is incomplete")
        if self.objective not in {
            value.value for value in CoverageStatePairObjective
        }:
            raise ValueError("training result has an unknown objective")
        if self.objective_policy != coverage_state_pair_objective_policy(
            self.objective
        ):
            raise ValueError("training result objective policy changed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_TRAINING_RESULT_SCHEMA,
            "objective": self.objective,
            "objective_policy": self.objective_policy,
            "seed": self.seed,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "completed_updates": self.completed_updates,
            "schedule_fingerprint": self.schedule_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "execution_device": self.execution_device,
            "device_cache_fingerprint": self.device_cache_fingerprint,
            "device_cache_resident_bytes": (
                self.device_cache_resident_bytes
            ),
            "optimizer_config_fingerprint": (
                self.optimizer_config_fingerprint
            ),
            "initial_model_fingerprint": (
                self.initial_model_fingerprint
            ),
            "final_model_fingerprint": self.final_model_fingerprint,
            "epoch_logs": list(self.epoch_logs),
            "first_nonzero_gradient_update": {
                name: update
                for name, update in self.first_nonzero_gradient_update
            },
            "compute": {
                "forward_calls": self.forward_calls,
                "backward_calls": self.backward_calls,
                "optimizer_steps": self.optimizer_steps,
                "logical_state_evaluations": (
                    self.logical_state_evaluations
                ),
                "finite_state_audits": self.finite_state_audits,
            },
        }

    @property
    def result_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


@dataclass(frozen=True)
class CoverageStateMatchedTrainingConfig:
    """One optimizer policy shared by all three objective coordinates."""

    seed: int
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        frozen = {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1.0e-8,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"matched CSLF training fixes {name}")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "optimizer": "adam",
            "learning_rate_hex": self.learning_rate.hex(),
            "weight_decay_hex": self.weight_decay.hex(),
            "betas_hex": [
                self.adam_beta1.hex(),
                self.adam_beta2.hex(),
            ],
            "epsilon_hex": self.adam_epsilon.hex(),
        }


@dataclass(frozen=True, eq=False)
class CoverageStateMatchedTrainingResult:
    """Three trained models sharing initialization, schedule, and endpoints."""

    config: CoverageStateMatchedTrainingConfig
    common_initial_model_fingerprint: str
    schedule_fingerprint: str
    cache_fingerprint: str
    results: tuple[CoverageStateTrainingResult, ...]
    models: tuple[
        tuple[str, CURELiteCoverageStateLevelSet],
        ...,
    ]

    def __post_init__(self) -> None:
        actual = tuple(value.objective for value in self.results)
        registered = tuple(
            tuple(value.value for value in suite)
            for suite in COVERAGE_STATE_REGISTERED_MATCHED_OBJECTIVE_SUITES
        )
        if actual not in registered:
            raise ValueError(
                "matched results must contain one registered objective suite"
            )
        if tuple(name for name, _ in self.models) != actual:
            raise ValueError(
                "matched models must contain the result objectives in order"
            )
        if any(
            value.initial_model_fingerprint
            != self.common_initial_model_fingerprint
            or value.schedule_fingerprint != self.schedule_fingerprint
            or value.cache_fingerprint != self.cache_fingerprint
            for value in self.results
        ):
            raise ValueError("matched objective fairness binding changed")
        if len(
            {
                value.optimizer_config_fingerprint
                for value in self.results
            }
        ) != 1:
            raise ValueError("matched objectives use different optimizers")
        if len(
            {
                (
                    value.execution_device,
                    value.device_cache_fingerprint,
                    value.device_cache_resident_bytes,
                )
                for value in self.results
            }
        ) != 1:
            raise ValueError(
                "matched objectives use different device caches"
            )
        model_contracts = {
            stable_fingerprint(
                coverage_state_model_contract_payload(model)
            )
            for _, model in self.models
        }
        if len(model_contracts) != 1:
            raise ValueError(
                "matched objectives use different model contracts"
            )
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        for result, (name, model) in zip(
            self.results,
            self.models,
            strict=True,
        ):
            if (
                name != result.objective
                or coverage_state_model_fingerprint(model)
                != result.final_model_fingerprint
            ):
                raise ValueError("matched trained model/result binding changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        objective_suite = tuple(
            value.objective for value in self.results
        )
        payload: dict[str, object] = {
            "schema_version": COVERAGE_STATE_MATCHED_RESULT_SCHEMA,
            "config": self.config.canonical_payload(),
            "common_initial_model_fingerprint": (
                self.common_initial_model_fingerprint
            ),
            "schedule_fingerprint": self.schedule_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "objectives": [
                value.canonical_payload() for value in self.results
            ],
            "objective_suite": list(objective_suite),
            "fairness": {
                "same_initial_state": True,
                "same_schedule": True,
                "same_endpoints": True,
                "same_model": True,
                "same_optimizer": True,
                "same_device_cache": True,
                "same_compute_budget": True,
                "same_natural_branches": True,
                "response_identity_share_joint_measure": True,
                "separable_uses_endpoint_absolute_measures": True,
                "allowed_difference": (
                    "pair_coordinate_and_predeclared_pair_measure"
                ),
            },
        }
        if all(
            isinstance(
                model.config,
                CoverageStatePhasePreservingConfig,
            )
            for _, model in self.models
        ):
            payload["model_contract"] = (
                coverage_state_model_contract_payload(self.models[0][1])
            )
            fairness = payload["fairness"]
            if not isinstance(fairness, dict):
                raise AssertionError("fairness payload must be a mapping")
            fairness.update(
                {
                    "same_model_class": True,
                    "same_model_config": True,
                    "same_parameter_count": True,
                    "same_parameter_shapes": True,
                }
            )
        return payload

    @property
    def result_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _finite_metric(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise FloatingPointError(f"{name} is non-finite")
    return result


def _seed_matched_rng(
    seed: int,
    device: torch.device,
) -> None:
    """Seed only the CPU generator and the selected execution device."""

    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _capture_matched_rng(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Capture CPU state plus, at most, the selected CUDA generator."""

    cpu_state = torch.get_rng_state().clone()
    cuda_state = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda"
        else None
    )
    return cpu_state, cuda_state


def _restore_matched_rng(
    cpu_state: torch.Tensor,
    cuda_state: torch.Tensor | None,
    *,
    device: torch.device,
) -> None:
    """Restore the matched coordinates without touching another GPU."""

    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        if device.type != "cuda":
            raise ValueError("CUDA RNG state requires a CUDA execution device")
        torch.cuda.set_rng_state(cuda_state, device=device)


def _protected_training_scope(
    schedule: CoverageStateTrainingSchedule,
) -> str | None:
    config = schedule.config
    if (
        config.epochs == 10
        and config.steps_per_epoch == 40
    ):
        return COVERAGE_STATE_BOUNDED_SCOPE
    if config.epochs == 800 and config.steps_per_epoch == 40:
        return COVERAGE_STATE_FORMAL_SCOPE
    return None


def _verify_run_authorization(
    authorization: object | None,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
) -> None:
    scope = _protected_training_scope(schedule)
    if scope is None:
        if authorization is not None:
            raise ValueError(
                "development training does not consume a bounded/formal authorization"
            )
        return
    if authorization is None:
        raise PermissionError(
            f"{scope} requires an explicit prerequisite-bound authorization"
        )
    if not isinstance(authorization, CoverageStateRunAuthorization):
        raise TypeError(
            "authorization must be a CoverageStateRunAuthorization"
        )
    authorization.verify_for_run(
        cache=cache,
        schedule=schedule,
        scope=scope,
    )


def _train_coverage_state_objective(
    model: CURELiteCoverageStateLevelSet,
    optimizer: torch.optim.Optimizer,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    objective: CoverageStatePairObjective | str,
    device: torch.device | str,
    expected_initial_model_fingerprint: str,
    authorization: object | None = None,
    epoch_callback: Callable[[Mapping[str, object]], None] | None = None,
    _cache_already_verified: bool = False,
    _schedule_already_verified: bool = False,
    _authorization_already_verified: bool = False,
    _device_cache: CoverageStateDeviceCache | None = None,
    _device_cache_already_verified: bool = False,
    _defer_device_cache_content_verification: bool = False,
) -> CoverageStateTrainingResult:
    """Internal implementation supporting the matched-run verified fast path."""

    if not isinstance(model, CURELiteCoverageStateLevelSet):
        raise TypeError("model must be CURELiteCoverageStateLevelSet")
    if not isinstance(cache, CoverageStateScalarCache):
        raise TypeError("cache must be CoverageStateScalarCache")
    if not isinstance(schedule, CoverageStateTrainingSchedule):
        raise TypeError("schedule must be CoverageStateTrainingSchedule")
    try:
        normalized_objective = (
            objective
            if isinstance(objective, CoverageStatePairObjective)
            else CoverageStatePairObjective(objective)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("unknown coverage-state objective") from error
    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError("epoch_callback must be callable or None")
    if not isinstance(_cache_already_verified, bool):
        raise TypeError("_cache_already_verified must be bool")
    if not isinstance(_schedule_already_verified, bool):
        raise TypeError("_schedule_already_verified must be bool")
    if not isinstance(_authorization_already_verified, bool):
        raise TypeError("_authorization_already_verified must be bool")
    if (
        not isinstance(_device_cache_already_verified, bool)
        or not isinstance(
            _defer_device_cache_content_verification,
            bool,
        )
    ):
        raise TypeError("device-cache verification flags must be bool")
    if _device_cache is not None and not isinstance(
        _device_cache,
        CoverageStateDeviceCache,
    ):
        raise TypeError("_device_cache must be CoverageStateDeviceCache")
    if _device_cache is None and (
        _device_cache_already_verified
        or _defer_device_cache_content_verification
    ):
        raise ValueError(
            "device-cache fast-path flags require an explicit device cache"
        )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if optimizer.state:
        raise RuntimeError(
            "coverage-state training requires a fresh empty optimizer"
        )
    if not _authorization_already_verified:
        _verify_run_authorization(authorization, cache, schedule)
    if not _schedule_already_verified:
        coverage_state_schedule_exposure_report(cache, schedule)
    elif not _cache_already_verified:
        cache.verify_unchanged()
    if schedule.cache_fingerprint != cache.cache_fingerprint:
        raise ValueError("training schedule and scalar cache differ")
    initial = coverage_state_model_fingerprint(model)
    if initial != expected_initial_model_fingerprint:
        raise ValueError("model does not match the frozen initial state")
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and requested_device.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        requested_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
    model_device = next(model.parameters()).device
    if model_device != requested_device:
        raise ValueError("model and requested training device differ")
    if (
        model.config.feature_stride
        != cache.sobolev_config.truncation_radius
        or model.config.feature_channels
        != cache.raw_catalog.natural_records[0].feature.shape[1]
    ):
        raise ValueError("model and scalar cache contracts differ")
    optimizer_config_fingerprint = (
        coverage_state_optimizer_config_fingerprint(model, optimizer)
    )
    device_cache = (
        prepare_coverage_state_device_cache(
            cache,
            device=requested_device,
        )
        if _device_cache is None
        else _device_cache
    )
    if (
        device_cache.source_cache is not cache
        or device_cache.source_cache_fingerprint
        != cache.cache_fingerprint
        or device_cache.device != requested_device
        or device_cache.dtype != torch.float32
    ):
        raise ValueError(
            "device cache, scalar cache, and requested device differ"
        )
    if not _device_cache_already_verified:
        device_cache.verify_unchanged(verify_content=True)
    audit_coverage_state_training_state(model, optimizer)
    finite_state_audits = 1

    first_nonzero: dict[str, int] = {}
    parameter_count = len(tuple(model.named_parameters()))
    epoch_logs: list[dict[str, object]] = []
    forward_calls = 0
    backward_calls = 0
    optimizer_steps = 0
    logical_states = 0
    for epoch in range(schedule.config.epochs):
        sums = {
            "factual_miss/loss": 0.0,
            "factual_no_miss/loss": 0.0,
            "pair/loss": 0.0,
            "total": 0.0,
            "gradient_l2_norm": 0.0,
        }
        selection_fingerprints: list[str] = []
        for step in range(schedule.config.steps_per_epoch):
            update = epoch * schedule.config.steps_per_epoch + step
            batch = device_cache.materialize(
                schedule.selections[update],
                verify=False,
                validate=False,
            )
            logs = coverage_state_fused_train_step(
                model,
                optimizer,
                batch,
                config=cache.sobolev_config,
                pair_objective=normalized_objective,
                audit=False,
                track_nonzero_gradients=(
                    len(first_nonzero) < parameter_count
                ),
            )
            for name in sums:
                sums[name] += _finite_metric(logs[name], name=name)
            names = str(logs["nonzero_gradient_parameters"])
            for name in filter(None, names.split(",")):
                first_nonzero.setdefault(name, update)
            selection_fingerprints.append(
                str(logs["selection_fingerprint"])
            )
            forward_calls += int(logs["model_forward_calls"])
            backward_calls += int(logs["backward_calls"])
            optimizer_steps += int(logs["optimizer_steps"])
            logical_states += int(logs["logical_states"])
            finite_state_audits += int(
                logs["post_step_finite_audits"]
            )
        denominator = float(schedule.config.steps_per_epoch)
        epoch_row: dict[str, object] = {
            "epoch": epoch,
            "completed_updates": (
                (epoch + 1) * schedule.config.steps_per_epoch
            ),
            "objective": normalized_objective.value,
            "selection_sequence_fingerprint": stable_fingerprint(
                selection_fingerprints
            ),
            **{
                f"mean_{name}": value / denominator
                for name, value in sums.items()
            },
        }
        if epoch_callback is not None:
            epoch_callback(dict(epoch_row))
        epoch_logs.append(epoch_row)
    if set(first_nonzero) != {
        name for name, _ in model.named_parameters()
    }:
        missing = sorted(
            {name for name, _ in model.named_parameters()}
            - set(first_nonzero)
        )
        raise RuntimeError(
            "CSLF parameters never received a nonzero gradient: "
            + ", ".join(missing)
        )
    latency = dict(first_nonzero)
    if (
        latency.get("phase_projection.weight") != 0
        or latency.get("phase_projection.bias") != 0
        or latency.get("input_projection.weight", 3) > 2
        or latency.get("spatial_mixing.weight", 3) > 2
    ):
        raise RuntimeError(
            "CSLF upstream-gradient latency gate did not pass"
        )
    if not _schedule_already_verified:
        coverage_state_schedule_exposure_report(cache, schedule)
    device_cache.verify_unchanged(
        verify_content=not _defer_device_cache_content_verification,
        verify_source=not _defer_device_cache_content_verification,
    )
    return CoverageStateTrainingResult(
        objective=normalized_objective.value,
        objective_policy=coverage_state_pair_objective_policy(
            normalized_objective
        ),
        seed=schedule.config.seed,
        epochs=schedule.config.epochs,
        steps_per_epoch=schedule.config.steps_per_epoch,
        completed_updates=schedule.config.updates,
        schedule_fingerprint=schedule.schedule_fingerprint,
        cache_fingerprint=cache.cache_fingerprint,
        execution_device=str(requested_device),
        device_cache_fingerprint=(
            device_cache.device_cache_fingerprint
        ),
        device_cache_resident_bytes=(
            device_cache.resident_tensor_bytes
        ),
        optimizer_config_fingerprint=optimizer_config_fingerprint,
        initial_model_fingerprint=initial,
        final_model_fingerprint=coverage_state_model_fingerprint(model),
        epoch_logs=tuple(epoch_logs),
        first_nonzero_gradient_update=tuple(
            sorted(first_nonzero.items())
        ),
        forward_calls=forward_calls,
        backward_calls=backward_calls,
        optimizer_steps=optimizer_steps,
        logical_state_evaluations=logical_states,
        finite_state_audits=finite_state_audits,
    )


def train_coverage_state_objective(
    model: CURELiteCoverageStateLevelSet,
    optimizer: torch.optim.Optimizer,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    objective: CoverageStatePairObjective | str,
    device: torch.device | str,
    expected_initial_model_fingerprint: str,
    authorization: object | None = None,
    epoch_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> CoverageStateTrainingResult:
    """Train one objective after all public prerequisite checks.

    The cache- and authorization-skipping controls belong exclusively to the
    private matched-run implementation.  Keeping them out of this public
    signature prevents a standalone caller from bypassing a bounded/formal
    scope's prerequisite-bound authorization.
    """

    return _train_coverage_state_objective(
        model,
        optimizer,
        cache,
        schedule,
        objective=objective,
        device=device,
        expected_initial_model_fingerprint=(
            expected_initial_model_fingerprint
        ),
        authorization=authorization,
        epoch_callback=epoch_callback,
    )


def _train_matched_coverage_state_objective_suite(
    model_config: CoverageStateLevelSetConfig,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    config: CoverageStateMatchedTrainingConfig,
    device: torch.device | str,
    objectives: tuple[CoverageStatePairObjective, ...],
    authorization: object | None = None,
    epoch_callback: (
        Callable[[str, Mapping[str, object]], None] | None
    ) = None,
) -> CoverageStateMatchedTrainingResult:
    """Train one registered objective suite from an exact shared state."""

    if not isinstance(model_config, CoverageStateLevelSetConfig):
        raise TypeError("model_config must be CoverageStateLevelSetConfig")
    if not isinstance(config, CoverageStateMatchedTrainingConfig):
        raise TypeError(
            "config must be CoverageStateMatchedTrainingConfig"
        )
    if objectives not in COVERAGE_STATE_REGISTERED_MATCHED_OBJECTIVE_SUITES:
        raise ValueError("objectives must be one registered matched suite")
    if schedule.config.seed != config.seed:
        raise ValueError("matched training seed and schedule seed differ")
    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError("epoch_callback must be callable or None")
    _verify_run_authorization(authorization, cache, schedule)
    coverage_state_schedule_exposure_report(cache, schedule)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and resolved_device.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        resolved_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
    device_cache = prepare_coverage_state_device_cache(
        cache,
        device=resolved_device,
    )
    device_cache.verify_unchanged(
        verify_content=True,
        verify_source=False,
    )
    _seed_matched_rng(config.seed, resolved_device)
    initial_model = build_coverage_state_level_set(model_config)
    initial_fingerprint = coverage_state_model_fingerprint(initial_model)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in initial_model.state_dict().items()
    }
    training_cpu_rng_state, training_cuda_rng_state = (
        _capture_matched_rng(resolved_device)
    )
    del initial_model

    results: list[CoverageStateTrainingResult] = []
    models: list[tuple[str, CURELiteCoverageStateLevelSet]] = []
    for objective in objectives:
        model = build_coverage_state_level_set(model_config)
        model.load_state_dict(initial_state, strict=True)
        if coverage_state_model_fingerprint(model) != initial_fingerprint:
            raise AssertionError("matched model failed exact initial reload")
        model = model.to(device=resolved_device, dtype=torch.float32)
        _restore_matched_rng(
            training_cpu_rng_state,
            training_cuda_rng_state,
            device=resolved_device,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        if optimizer.state:
            raise RuntimeError("fresh matched optimizer unexpectedly has state")

        def callback(row: Mapping[str, object]) -> None:
            if epoch_callback is not None:
                epoch_callback(objective.value, row)

        result = _train_coverage_state_objective(
            model,
            optimizer,
            cache,
            schedule,
            objective=objective,
            device=resolved_device,
            expected_initial_model_fingerprint=initial_fingerprint,
            authorization=authorization,
            epoch_callback=callback,
            _cache_already_verified=True,
            _schedule_already_verified=True,
            _authorization_already_verified=True,
            _device_cache=device_cache,
            _device_cache_already_verified=True,
            _defer_device_cache_content_verification=True,
        )
        results.append(result)
        models.append((objective.value, model))
    coverage_state_schedule_exposure_report(cache, schedule)
    device_cache.verify_unchanged(
        verify_content=True,
        verify_source=False,
    )
    return CoverageStateMatchedTrainingResult(
        config=config,
        common_initial_model_fingerprint=initial_fingerprint,
        schedule_fingerprint=schedule.schedule_fingerprint,
        cache_fingerprint=cache.cache_fingerprint,
        results=tuple(results),
        models=tuple(models),
    )


def train_matched_coverage_state_objectives(
    model_config: CoverageStateLevelSetConfig,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    config: CoverageStateMatchedTrainingConfig,
    device: torch.device | str,
    authorization: object | None = None,
    epoch_callback: (
        Callable[[str, Mapping[str, object]], None] | None
    ) = None,
) -> CoverageStateMatchedTrainingResult:
    """Train the frozen v15 response/identity/separable suite."""

    return _train_matched_coverage_state_objective_suite(
        model_config,
        cache,
        schedule,
        config=config,
        device=device,
        objectives=COVERAGE_STATE_LEGACY_MATCHED_OBJECTIVES,
        authorization=authorization,
        epoch_callback=epoch_callback,
    )


def train_matched_coverage_state_completion_rooted_objectives(
    model_config: CoverageStateLevelSetConfig,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    config: CoverageStateMatchedTrainingConfig,
    device: torch.device | str,
    authorization: object | None = None,
    epoch_callback: (
        Callable[[str, Mapping[str, object]], None] | None
    ) = None,
) -> CoverageStateMatchedTrainingResult:
    """Train completion-rooted response and both frozen controls."""

    return _train_matched_coverage_state_objective_suite(
        model_config,
        cache,
        schedule,
        config=config,
        device=device,
        objectives=COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES,
        authorization=authorization,
        epoch_callback=epoch_callback,
    )


def train_matched_coverage_state_support_oriented_objectives(
    model_config: CoverageStateLevelSetConfig,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    config: CoverageStateMatchedTrainingConfig,
    device: torch.device | str,
    authorization: object | None = None,
    epoch_callback: (
        Callable[[str, Mapping[str, object]], None] | None
    ) = None,
) -> CoverageStateMatchedTrainingResult:
    """Train support-oriented response and both frozen controls."""

    return _train_matched_coverage_state_objective_suite(
        model_config,
        cache,
        schedule,
        config=config,
        device=device,
        objectives=COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
        authorization=authorization,
        epoch_callback=epoch_callback,
    )


def train_matched_coverage_state_phase_preserving_support_oriented_objectives(
    model_config: CoverageStatePhasePreservingConfig,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    config: CoverageStateMatchedTrainingConfig,
    device: torch.device | str,
    authorization: object | None = None,
    epoch_callback: (
        Callable[[str, Mapping[str, object]], None] | None
    ) = None,
) -> CoverageStateMatchedTrainingResult:
    """Train the SORR suite with one shared PPCE model architecture."""

    if not isinstance(model_config, CoverageStatePhasePreservingConfig):
        raise TypeError(
            "model_config must be CoverageStatePhasePreservingConfig"
        )
    return _train_matched_coverage_state_objective_suite(
        model_config,
        cache,
        schedule,
        config=config,
        device=device,
        objectives=COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
        authorization=authorization,
        epoch_callback=epoch_callback,
    )


__all__ = [
    "COVERAGE_STATE_TRAINING_RESULT_SCHEMA",
    "COVERAGE_STATE_MATCHED_RESULT_SCHEMA",
    "COVERAGE_STATE_BOUNDED_SCOPE",
    "COVERAGE_STATE_FORMAL_SCOPE",
    "COVERAGE_STATE_REGISTERED_MATCHED_OBJECTIVE_SUITES",
    "CoverageStateMatchedTrainingConfig",
    "CoverageStateMatchedTrainingResult",
    "CoverageStateRunAuthorization",
    "CoverageStateTrainingResult",
    "coverage_state_model_fingerprint",
    "coverage_state_model_contract_payload",
    "coverage_state_optimizer_config_fingerprint",
    "train_matched_coverage_state_objectives",
    "train_matched_coverage_state_completion_rooted_objectives",
    (
        "train_matched_coverage_state_"
        "phase_preserving_support_oriented_objectives"
    ),
    "train_matched_coverage_state_support_oriented_objectives",
    "train_coverage_state_objective",
]
