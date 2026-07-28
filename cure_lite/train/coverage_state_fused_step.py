"""One-update scalar CSLF training with one fixed 12-state forward."""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
from torch import Tensor

from ..coverage_state_batches import (
    COVERAGE_STATE_FUSED_LOGICAL_STATES,
    COVERAGE_STATE_FUSED_NATURAL_COUNT,
    COVERAGE_STATE_FUSED_PAIR_COUNT,
    CoverageStateFusedBatch,
)
from ..coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
)
from ..coverage_state_supremal_projection import (
    CSLF_USCOPE_POLICY,
    coverage_state_uscope_pair_loss_from_targets,
)
from ..coverage_state_sobolev import (
    CSLF_COMPLETION_ROOTED_RESPONSE_POLICY,
    CSLF_PMOPE_POLICY,
    CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
    CoverageStateSobolevConfig,
    coverage_state_absolute_sobolev_loss_from_targets,
    coverage_state_completion_rooted_pair_sobolev_loss_from_targets,
    coverage_state_identity_joint_loss_from_targets,
    coverage_state_pair_sobolev_loss_from_targets,
    coverage_state_pmope_pair_loss_from_targets,
    coverage_state_support_oriented_pair_sobolev_loss_from_targets,
)


class CoverageStatePairObjective(str, Enum):
    RESPONSE_JOINT = "response_joint"
    COMPLETION_ROOTED_RESPONSE_JOINT = (
        "completion_rooted_response_joint"
    )
    SUPPORT_ORIENTED_RESPONSE_JOINT = (
        "support_oriented_response_joint"
    )
    PMOPE_JOINT = "pmope_joint"
    USCOPE_JOINT = "uscope_joint"
    IDENTITY_JOINT = "identity_joint"
    SEPARABLE_ENDPOINT = "separable_endpoint"


COVERAGE_STATE_LEGACY_MATCHED_OBJECTIVES = (
    CoverageStatePairObjective.RESPONSE_JOINT,
    CoverageStatePairObjective.IDENTITY_JOINT,
    CoverageStatePairObjective.SEPARABLE_ENDPOINT,
)
COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES = (
    CoverageStatePairObjective.COMPLETION_ROOTED_RESPONSE_JOINT,
    CoverageStatePairObjective.IDENTITY_JOINT,
    CoverageStatePairObjective.SEPARABLE_ENDPOINT,
)
COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES = (
    CoverageStatePairObjective.SUPPORT_ORIENTED_RESPONSE_JOINT,
    CoverageStatePairObjective.IDENTITY_JOINT,
    CoverageStatePairObjective.SEPARABLE_ENDPOINT,
)
COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES = (
    CoverageStatePairObjective.PMOPE_JOINT,
)
COVERAGE_STATE_USCOPE_MATCHED_OBJECTIVES = (
    CoverageStatePairObjective.USCOPE_JOINT,
)


def _normalize_objective(
    value: CoverageStatePairObjective | str,
) -> CoverageStatePairObjective:
    if isinstance(value, CoverageStatePairObjective):
        return value
    if not isinstance(value, str):
        raise TypeError("pair_objective must be a string or enum")
    try:
        return CoverageStatePairObjective(value)
    except ValueError as error:
        raise ValueError("unknown coverage-state pair objective") from error


def coverage_state_pair_objective_policy(
    value: CoverageStatePairObjective | str,
) -> str:
    """Return the immutable mathematical policy for one pair coordinate."""

    objective = _normalize_objective(value)
    if (
        objective
        is CoverageStatePairObjective.COMPLETION_ROOTED_RESPONSE_JOINT
    ):
        return CSLF_COMPLETION_ROOTED_RESPONSE_POLICY
    if (
        objective
        is CoverageStatePairObjective.SUPPORT_ORIENTED_RESPONSE_JOINT
    ):
        return CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
    if objective is CoverageStatePairObjective.PMOPE_JOINT:
        return CSLF_PMOPE_POLICY
    if objective is CoverageStatePairObjective.USCOPE_JOINT:
        return CSLF_USCOPE_POLICY
    return objective.value


def _validate_optimizer(
    model: CURELiteCoverageStateLevelSet,
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.nn.Parameter, ...]:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    parameters = tuple(model.parameters())
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )
    parameter_ids = tuple(id(value) for value in parameters)
    optimizer_ids = tuple(id(value) for value in optimizer_parameters)
    if (
        not parameters
        or any(
            not value.requires_grad or value.dtype != torch.float32
            for value in parameters
        )
        or len(set(parameter_ids)) != len(parameter_ids)
        or len(set(optimizer_ids)) != len(optimizer_ids)
        or set(optimizer_ids) != set(parameter_ids)
    ):
        raise ValueError(
            "optimizer must contain every trainable FP32 CSLF parameter once"
        )
    return parameters


def _state_tensors(value: Any) -> tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        return (value,)
    if isinstance(value, dict):
        return tuple(
            tensor
            for item in value.values()
            for tensor in _state_tensors(item)
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            tensor
            for item in value
            for tensor in _state_tensors(item)
        )
    return ()


def audit_coverage_state_training_state(
    model: CURELiteCoverageStateLevelSet,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Fail closed on any non-finite model or optimizer tensor.

    Finite flags are reduced once per device instead of synchronizing once per
    tensor.  The function is used both before update zero and after every
    optimizer step, so a complete run has ``updates + 1`` audits.
    """

    if not isinstance(model, CURELiteCoverageStateLevelSet):
        raise TypeError("model must be CURELiteCoverageStateLevelSet")
    _validate_optimizer(model, optimizer)
    named_tensors = [
        (f"model parameter {name}", value)
        for name, value in model.named_parameters()
    ]
    named_tensors.extend(
        (f"model buffer {name}", value)
        for name, value in model.named_buffers()
    )
    named_tensors.extend(
        ("optimizer state", value)
        for value in _state_tensors(optimizer.state)
    )
    by_device: dict[torch.device, list[Tensor]] = {}
    for _, value in named_tensors:
        by_device.setdefault(value.device, []).append(
            torch.isfinite(value).all()
        )
    for device, flags in by_device.items():
        if flags and not bool(torch.stack(flags).all().detach().cpu()):
            raise FloatingPointError(
                f"training state contains a non-finite tensor on {device}"
            )


def _preflight(
    model: CURELiteCoverageStateLevelSet,
    optimizer: torch.optim.Optimizer,
    batch: CoverageStateFusedBatch,
    config: CoverageStateSobolevConfig,
) -> tuple[tuple[torch.nn.Parameter, ...], Tensor, Tensor]:
    if not isinstance(model, CURELiteCoverageStateLevelSet):
        raise TypeError("model must be CURELiteCoverageStateLevelSet")
    if not isinstance(batch, CoverageStateFusedBatch):
        raise TypeError("batch must be CoverageStateFusedBatch")
    if not isinstance(config, CoverageStateSobolevConfig):
        raise TypeError("config must be CoverageStateSobolevConfig")
    feature, occupancy = batch.model_inputs()
    if (
        model.config.feature_channels != feature.shape[1]
        or model.config.feature_stride != config.truncation_radius
        or tuple(occupancy.shape[-2:])
        != (
            int(feature.shape[-2]) * model.config.feature_stride,
            int(feature.shape[-1]) * model.config.feature_stride,
        )
    ):
        raise ValueError("model, Sobolev config, and fused grids differ")
    if feature.device != next(model.parameters()).device:
        raise ValueError("model and fused batch must share a device")
    return _validate_optimizer(model, optimizer), feature, occupancy


def _pair_loss(
    objective: CoverageStatePairObjective,
    field_plus: Tensor,
    field_minus: Tensor,
    batch: CoverageStateFusedBatch,
    *,
    config: CoverageStateSobolevConfig,
) -> Tensor:
    if objective is CoverageStatePairObjective.RESPONSE_JOINT:
        return coverage_state_pair_sobolev_loss_from_targets(
            field_plus,
            field_minus,
            batch.pairs.joint_targets,
            config=config,
            validate=False,
        ).loss
    if (
        objective
        is CoverageStatePairObjective.COMPLETION_ROOTED_RESPONSE_JOINT
    ):
        return (
            coverage_state_completion_rooted_pair_sobolev_loss_from_targets(
                field_plus,
                field_minus,
                batch.pairs.joint_targets,
                config=config,
                validate=False,
            ).loss
        )
    if (
        objective
        is CoverageStatePairObjective.SUPPORT_ORIENTED_RESPONSE_JOINT
    ):
        return coverage_state_support_oriented_pair_sobolev_loss_from_targets(
            field_plus,
            field_minus,
            batch.pairs.joint_targets,
            config=config,
            validate=False,
        ).loss
    if objective is CoverageStatePairObjective.PMOPE_JOINT:
        return coverage_state_pmope_pair_loss_from_targets(
            field_plus,
            field_minus,
            batch.pairs.joint_targets,
            config=config,
            validate=False,
        ).loss
    if objective is CoverageStatePairObjective.USCOPE_JOINT:
        return coverage_state_uscope_pair_loss_from_targets(
            field_plus,
            field_minus,
            batch.pairs.joint_targets,
            config=config,
            validate=False,
        ).loss
    if objective is CoverageStatePairObjective.IDENTITY_JOINT:
        return coverage_state_identity_joint_loss_from_targets(
            field_plus,
            field_minus,
            batch.pairs.joint_targets,
            config=config,
            validate=False,
        ).loss
    plus = coverage_state_absolute_sobolev_loss_from_targets(
        field_plus,
        batch.pairs.absolute_targets_plus,
        config=config,
        validate=False,
    ).loss
    minus = coverage_state_absolute_sobolev_loss_from_targets(
        field_minus,
        batch.pairs.absolute_targets_minus,
        config=config,
        validate=False,
    ).loss
    return 0.5 * (plus + minus)


def coverage_state_fused_train_step(
    model: CURELiteCoverageStateLevelSet,
    optimizer: torch.optim.Optimizer,
    batch: CoverageStateFusedBatch,
    *,
    config: CoverageStateSobolevConfig,
    pair_objective: CoverageStatePairObjective | str,
    audit: bool = True,
    track_nonzero_gradients: bool = True,
) -> dict[str, float | int | str]:
    """Run the fixed three branches through CSLF once and make one update.

    Branches are averaged internally and then combined as
    ``miss + no_miss + pair``.  Registered matched objectives differ only
    in the pair criterion; model input, endpoint order, forward count, and
    optimizer update are identical.
    """

    objective = _normalize_objective(pair_objective)
    if not isinstance(audit, bool):
        raise TypeError("audit must be bool")
    if not isinstance(track_nonzero_gradients, bool):
        raise TypeError("track_nonzero_gradients must be bool")
    parameters, feature, occupancy = _preflight(
        model,
        optimizer,
        batch,
        config,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        field = model(feature, occupancy)
        if (
            field.dtype != torch.float32
            or tuple(field.shape) != tuple(occupancy.shape)
            or not bool(torch.isfinite(field).all())
        ):
            raise FloatingPointError(
                "CSLF fused field must be finite FP32 and grid-aligned"
            )
        field_miss, field_no_miss, field_plus, field_minus = torch.split(
            field,
            (
                COVERAGE_STATE_FUSED_NATURAL_COUNT,
                COVERAGE_STATE_FUSED_NATURAL_COUNT,
                COVERAGE_STATE_FUSED_PAIR_COUNT,
                COVERAGE_STATE_FUSED_PAIR_COUNT,
            ),
            dim=0,
        )
        miss_loss = coverage_state_absolute_sobolev_loss_from_targets(
            field_miss,
            batch.factual_miss.targets,
            config=config,
            validate=False,
        ).loss
        no_miss_loss = coverage_state_absolute_sobolev_loss_from_targets(
            field_no_miss,
            batch.factual_no_miss.targets,
            config=config,
            validate=False,
        ).loss
        pair_loss = _pair_loss(
            objective,
            field_plus,
            field_minus,
            batch,
            config=config,
        )
        total = miss_loss + no_miss_loss + pair_loss
    if (
        total.dtype != torch.float32
        or not bool(torch.isfinite(total))
    ):
        raise FloatingPointError("CSLF fused loss is non-finite")

    total.backward()
    squared_gradient_norm = torch.zeros(
        (),
        device=total.device,
        dtype=torch.float32,
    )
    finite_gradient_flags: list[Tensor] = []
    nonzero_gradient_flags: list[Tensor] = []
    ordered_parameter_names: list[str] = []
    parameter_names = {
        id(parameter): name
        for name, parameter in model.named_parameters()
    }
    for parameter in parameters:
        name = parameter_names[id(parameter)]
        if parameter.grad is None:
            raise RuntimeError(f"CSLF parameter {name} received no gradient")
        if parameter.grad.dtype != torch.float32:
            raise FloatingPointError(f"CSLF parameter {name} has an invalid gradient")
        finite_gradient_flags.append(torch.isfinite(parameter.grad).all())
        squared_gradient_norm = (
            squared_gradient_norm
            + parameter.grad.detach().square().sum()
        )
        if track_nonzero_gradients:
            ordered_parameter_names.append(name)
            nonzero_gradient_flags.append(torch.any(parameter.grad != 0.0))
    gradient_norm = torch.sqrt(squared_gradient_norm)
    gradient_status = torch.stack(
        (
            torch.isfinite(gradient_norm),
            torch.stack(finite_gradient_flags).all(),
        )
    ).all()
    if not bool(gradient_status.detach().cpu()):
        raise FloatingPointError("CSLF fused update has a non-finite gradient")
    nonzero_gradient_names: list[str] = []
    if nonzero_gradient_flags:
        nonzero_values = (
            torch.stack(nonzero_gradient_flags).detach().cpu().tolist()
        )
        nonzero_gradient_names = [
            name
            for name, nonzero in zip(
                ordered_parameter_names,
                nonzero_values,
                strict=True,
            )
            if bool(nonzero)
        ]

    optimizer.step()
    audit_coverage_state_training_state(model, optimizer)

    logs: dict[str, float | int | str] = {
        "pair_objective": objective.value,
        "pair_objective_policy": coverage_state_pair_objective_policy(
            objective
        ),
        "selection_fingerprint": batch.selection_fingerprint,
        "model_forward_calls": 1,
        "backward_calls": 1,
        "optimizer_steps": 1,
        "logical_states": COVERAGE_STATE_FUSED_LOGICAL_STATES,
        "factual_miss_states": COVERAGE_STATE_FUSED_NATURAL_COUNT,
        "factual_no_miss_states": COVERAGE_STATE_FUSED_NATURAL_COUNT,
        "pair_count": COVERAGE_STATE_FUSED_PAIR_COUNT,
        "pair_endpoint_states": 2 * COVERAGE_STATE_FUSED_PAIR_COUNT,
        "clean_positive_pairs": 1,
        "component_null_pairs": 1,
        "identity_null_optimizer_exposure": 0,
        "diagnostic_only_optimizer_exposure": 0,
        "autocast_enabled": 0,
        "post_step_finite_audits": 1,
        "gradient_latency_tracked": int(track_nonzero_gradients),
        "nonzero_gradient_parameter_count": len(
            nonzero_gradient_names
        ),
        "nonzero_gradient_parameters": ",".join(
            sorted(nonzero_gradient_names)
        ),
    }
    if audit:
        logs.update(
            {
                "factual_miss/loss": float(miss_loss.detach().cpu()),
                "factual_no_miss/loss": float(
                    no_miss_loss.detach().cpu()
                ),
                "pair/loss": float(pair_loss.detach().cpu()),
                "total": float(total.detach().cpu()),
                "gradient_l2_norm": float(gradient_norm.detach().cpu()),
            }
        )
    else:
        values = torch.stack(
            (
                miss_loss,
                no_miss_loss,
                pair_loss,
                total,
                gradient_norm,
            )
        ).detach().cpu().tolist()
        logs.update(
            {
                "factual_miss/loss": float(values[0]),
                "factual_no_miss/loss": float(values[1]),
                "pair/loss": float(values[2]),
                "total": float(values[3]),
                "gradient_l2_norm": float(values[4]),
            }
        )
    return logs


__all__ = [
    "COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES",
    "COVERAGE_STATE_LEGACY_MATCHED_OBJECTIVES",
    "COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES",
    "COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES",
    "COVERAGE_STATE_USCOPE_MATCHED_OBJECTIVES",
    "CoverageStatePairObjective",
    "audit_coverage_state_training_state",
    "coverage_state_pair_objective_policy",
    "coverage_state_fused_train_step",
]
