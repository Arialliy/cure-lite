"""Generated-only raw parity receipts for PACRE v22 and PACRE-VC v23.

The probes in this module construct every tensor in memory.  They compare the
unchanged v22 forward with the versioned v23 wrapper, then run three matched
Adam updates using the real PMOPE target preparation and loss functions.
No dataset, cache, checkpoint, evaluation split, or external model is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
import hashlib
from typing import Final, Mapping

import torch
from torch import Tensor

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_sobolev import (
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    coverage_state_pmope_pair_loss_from_targets,
    prepare_coverage_state_pair_targets,
)
from cure_lite.experiment.coverage_state_paet_dr_gate import (
    _deterministic_execution_scope,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
    CoverageStatePACREFields,
)

from .pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)


PACRE_VC_PARITY_SCHEMA: Final = (
    "cure-lite-pacre-v23-generated-forward-optimizer-parity-v1"
)
PACRE_VC_PARITY_RUN_SCHEMA: Final = (
    "cure-lite-pacre-v23-generated-forward-optimizer-parity-run-v1"
)
PACRE_VC_PARITY_SEEDS: Final = (42, 43, 44)
PACRE_VC_PARITY_ADAM_STEPS: Final = 3
PACRE_VC_PARITY_FEATURE_CHANNELS: Final = 2
PACRE_VC_PARITY_FEATURE_STRIDE: Final = 2
PACRE_VC_PARITY_WIDTH: Final = 4
PACRE_VC_PARITY_BATCH_SIZE: Final = 2
PACRE_VC_PARITY_ADAM_CONFIG: Final = {
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1.0e-8,
    "weight_decay": 0.0,
}


@dataclass(frozen=True)
class _GeneratedPairBatch:
    feature: Tensor
    occupancy_plus: Tensor
    occupancy_minus: Tensor
    target_plus: Tensor
    target_minus: Tensor
    valid_mask: Tensor
    targets: CoverageStatePairTargets


def _resolve_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        if resolved.index is not None:
            raise ValueError("CPU parity device must not have an index")
        return resolved
    if resolved.type != "cuda":
        raise ValueError("parity supports only CPU or CUDA")
    if not torch.cuda.is_available():
        raise ValueError("requested CUDA parity device is unavailable")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    if (
        resolved.index is None
        or resolved.index < 0
        or resolved.index >= torch.cuda.device_count()
    ):
        raise ValueError("requested CUDA parity device is unavailable")
    return resolved


def _tensor_observation(value: Tensor) -> dict[str, object]:
    if not isinstance(value, Tensor):
        raise TypeError("parity observation requires a tensor")
    if value.dim() == 0:
        cpu = value.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(b"[]")
        if cpu.numel():
            digest.update(
                cpu.reshape(1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        raw_fingerprint = digest.hexdigest()
    else:
        raw_fingerprint = tensor_content_fingerprint(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "raw_fingerprint": raw_fingerprint,
    }


def _tensor_mapping_observation(
    values: Mapping[str, Tensor],
) -> dict[str, object]:
    rows = {
        name: _tensor_observation(value)
        for name, value in sorted(values.items())
    }
    return {
        "tensors": rows,
        "raw_fingerprint": stable_fingerprint(rows),
    }


def _model_state_observation(
    model: torch.nn.Module,
) -> dict[str, object]:
    return _tensor_mapping_observation(model.state_dict())


def _fields_observation(
    fields: CoverageStatePACREFields,
) -> dict[str, object]:
    if type(fields) is not CoverageStatePACREFields:
        raise TypeError("parity requires exact v22 PACRE fields")
    tensors: dict[str, object] = {}
    output_size: list[int] | None = None
    for definition in dataclass_fields(CoverageStatePACREFields):
        value = getattr(fields, definition.name)
        if isinstance(value, Tensor):
            tensors[definition.name] = _tensor_observation(value)
        elif definition.name == "output_size":
            output_size = [int(item) for item in value]
        else:
            raise TypeError("PACRE fields contain an unsupported value")
    body = {
        "fields_fqcn": (
            f"{type(fields).__module__}.{type(fields).__qualname__}"
        ),
        "tensors": tensors,
        "output_size": output_size,
    }
    return {**body, "raw_fingerprint": stable_fingerprint(body)}


def _targets_observation(
    targets: CoverageStatePairTargets,
) -> dict[str, object]:
    rows = {
        definition.name: _tensor_observation(
            getattr(targets, definition.name)
        )
        for definition in dataclass_fields(CoverageStatePairTargets)
    }
    return {
        "tensors": rows,
        "raw_fingerprint": stable_fingerprint(rows),
    }


def _batch_observation(
    batch: _GeneratedPairBatch,
) -> dict[str, object]:
    body = {
        "feature": _tensor_observation(batch.feature),
        "occupancy_plus": _tensor_observation(batch.occupancy_plus),
        "occupancy_minus": _tensor_observation(batch.occupancy_minus),
        "target_plus": _tensor_observation(batch.target_plus),
        "target_minus": _tensor_observation(batch.target_minus),
        "valid_mask": _tensor_observation(batch.valid_mask),
        "prepared_targets": _targets_observation(batch.targets),
    }
    return {**body, "raw_fingerprint": stable_fingerprint(body)}


def _generated_pair_batches(
    *,
    seed: int,
    device: torch.device,
    loss_config: CoverageStateSobolevConfig,
) -> tuple[_GeneratedPairBatch, ...]:
    batches: list[_GeneratedPairBatch] = []
    coordinate_rows = (
        (((0, 0), (2, 2), (4, 6)), ((0, 1), (3, 3), (5, 6))),
        (((1, 0), (2, 3), (4, 5)), ((1, 1), (3, 4), (5, 5))),
        (((0, 2), (2, 4), (4, 6)), ((1, 2), (3, 5), (5, 6))),
    )
    for step, sample_rows in enumerate(coordinate_rows):
        generator = torch.Generator(device="cpu").manual_seed(
            seed * 1000 + 230 + step
        )
        feature = torch.randn(
            (
                PACRE_VC_PARITY_BATCH_SIZE,
                PACRE_VC_PARITY_FEATURE_CHANNELS,
                3,
                4,
            ),
            generator=generator,
            dtype=torch.float32,
        )
        shape = (
            PACRE_VC_PARITY_BATCH_SIZE,
            1,
            6,
            8,
        )
        occupancy_plus = torch.zeros(shape, dtype=torch.bool)
        occupancy_minus = torch.zeros_like(occupancy_plus)
        target_plus = torch.zeros_like(occupancy_plus)
        target_minus = torch.zeros_like(occupancy_plus)
        valid_mask = torch.ones_like(occupancy_plus)
        for batch_index, (
            shared_coverage,
            removed_coverage,
            persistent_target,
        ) in enumerate(sample_rows):
            shared_row, shared_column = shared_coverage
            removed_row, removed_column = removed_coverage
            target_row, target_column = persistent_target
            occupancy_plus[
                batch_index, 0, shared_row, shared_column
            ] = True
            occupancy_minus[
                batch_index, 0, shared_row, shared_column
            ] = True
            occupancy_plus[
                batch_index, 0, removed_row, removed_column
            ] = True
            target_plus[
                batch_index, 0, target_row, target_column
            ] = True
            target_minus[
                batch_index, 0, target_row, target_column
            ] = True
            target_minus[
                batch_index, 0, removed_row, removed_column
            ] = True

        feature = feature.contiguous().to(device=device)
        occupancy_plus = occupancy_plus.contiguous().to(device=device)
        occupancy_minus = occupancy_minus.contiguous().to(device=device)
        target_plus = target_plus.contiguous().to(device=device)
        target_minus = target_minus.contiguous().to(device=device)
        valid_mask = valid_mask.contiguous().to(device=device)
        targets = prepare_coverage_state_pair_targets(
            occupancy_plus,
            occupancy_minus,
            target_plus,
            target_minus,
            valid_mask,
            config=loss_config,
        )
        batches.append(
            _GeneratedPairBatch(
                feature=feature,
                occupancy_plus=occupancy_plus,
                occupancy_minus=occupancy_minus,
                target_plus=target_plus,
                target_minus=target_minus,
                valid_mask=valid_mask,
                targets=targets,
            )
        )
    if len(batches) != PACRE_VC_PARITY_ADAM_STEPS:
        raise AssertionError("generated parity schedule changed")
    return tuple(batches)


def _build_models(
    *,
    seed: int,
    device: torch.device,
) -> tuple[
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CURELitePACREVerifierCorrectedLevelSet,
]:
    torch.random.default_generator.manual_seed(seed)
    v22 = CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
        CoverageStatePACREConfig(
            feature_channels=PACRE_VC_PARITY_FEATURE_CHANNELS,
            feature_stride=PACRE_VC_PARITY_FEATURE_STRIDE,
            width=PACRE_VC_PARITY_WIDTH,
        )
    )
    torch.random.default_generator.manual_seed(seed)
    v23 = CURELitePACREVerifierCorrectedLevelSet(
        CoverageStatePACREVerifierCorrectedConfig(
            feature_channels=PACRE_VC_PARITY_FEATURE_CHANNELS,
            feature_stride=PACRE_VC_PARITY_FEATURE_STRIDE,
            width=PACRE_VC_PARITY_WIDTH,
        )
    )
    return (
        v22.to(device=device, dtype=torch.float32),
        v23.to(device=device, dtype=torch.float32),
    )


def _probe_loss(fields: CoverageStatePACREFields) -> Tensor:
    return (
        fields.field.square().mean()
        + fields.actual_compatibility_hidden.square().mean()
        + fields.flipped_compatibility_hidden.square().mean()
    )


def _gradient_observation(
    model: torch.nn.Module,
    fields: CoverageStatePACREFields,
) -> dict[str, object]:
    parameters = dict(model.named_parameters())
    gradients = dict(
        zip(
            parameters,
            torch.autograd.grad(
                _probe_loss(fields),
                tuple(parameters.values()),
                allow_unused=False,
            ),
            strict=True,
        )
    )
    if any(parameter.grad is not None for parameter in parameters.values()):
        raise RuntimeError("probe gradients leaked into model buffers")
    return _tensor_mapping_observation(gradients)


def _adam() -> dict[str, object]:
    return {
        "lr": PACRE_VC_PARITY_ADAM_CONFIG["learning_rate"],
        "betas": (
            PACRE_VC_PARITY_ADAM_CONFIG["beta1"],
            PACRE_VC_PARITY_ADAM_CONFIG["beta2"],
        ),
        "eps": PACRE_VC_PARITY_ADAM_CONFIG["epsilon"],
        "weight_decay": PACRE_VC_PARITY_ADAM_CONFIG["weight_decay"],
    }


def _run_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    batch: _GeneratedPairBatch,
    *,
    loss_config: CoverageStateSobolevConfig,
) -> Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    feature = torch.cat((batch.feature, batch.feature), dim=0)
    occupancy = torch.cat(
        (batch.occupancy_plus, batch.occupancy_minus),
        dim=0,
    )
    with torch.autocast(device_type=feature.device.type, enabled=False):
        field_plus, field_minus = model(feature, occupancy).split(
            PACRE_VC_PARITY_BATCH_SIZE,
            dim=0,
        )
        loss = coverage_state_pmope_pair_loss_from_targets(
            field_plus,
            field_minus,
            batch.targets,
            config=loss_config,
            validate=True,
        ).loss
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("generated PMOPE parity loss is invalid")
    loss.backward()
    if any(
        parameter.grad is None
        or parameter.grad.dtype != torch.float32
        or not bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise FloatingPointError("generated PMOPE parity gradient is invalid")
    optimizer.step()
    if any(
        not bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    ):
        raise FloatingPointError("generated PMOPE parity update is invalid")
    return loss.detach().clone()


def _optimizer_component_observations(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {
        "step": {},
        "exp_avg": {},
        "exp_avg_sq": {},
    }
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict) or frozenset(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RuntimeError("Adam parity state contract changed")
        for component in result:
            value = state.get(component)
            if not isinstance(value, Tensor):
                raise TypeError("Adam parity state must be tensor-valued")
            result[component][name] = _tensor_observation(value)
    return result


def _loss_observation(loss: Tensor) -> dict[str, object]:
    return {
        "value_hex": float(loss.detach().cpu()).hex(),
        "tensor": _tensor_observation(loss),
    }


def _matched_observation(
    v22: object,
    v23: object,
) -> dict[str, object]:
    return {
        "passed": v22 == v23,
        "v22": v22,
        "v23": v23,
    }


def _run_generated_parity(
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    loss_config = CoverageStateSobolevConfig(
        truncation_radius=PACRE_VC_PARITY_FEATURE_STRIDE
    )
    batches = _generated_pair_batches(
        seed=seed,
        device=device,
        loss_config=loss_config,
    )

    probe_v22, probe_v23 = _build_models(seed=seed, device=device)
    initial_v22 = _model_state_observation(probe_v22)
    initial_v23 = _model_state_observation(probe_v23)
    fields_v22 = probe_v22.forward_fields(
        batches[0].feature,
        batches[0].occupancy_plus,
    )
    fields_v23 = probe_v23.forward_fields(
        batches[0].feature,
        batches[0].occupancy_plus,
    )
    fields_observation_v22 = _fields_observation(fields_v22)
    fields_observation_v23 = _fields_observation(fields_v23)
    gradient_v22 = _gradient_observation(probe_v22, fields_v22)
    gradient_v23 = _gradient_observation(probe_v23, fields_v23)
    probe_final_v22 = _model_state_observation(probe_v22)
    probe_final_v23 = _model_state_observation(probe_v23)
    probe_models_preserved = (
        initial_v22 == probe_final_v22
        and initial_v23 == probe_final_v23
        and all(
            parameter.grad is None
            for model in (probe_v22, probe_v23)
            for parameter in model.parameters()
        )
    )

    train_v22, train_v23 = _build_models(seed=seed, device=device)
    train_initial_v22 = _model_state_observation(train_v22)
    train_initial_v23 = _model_state_observation(train_v23)
    optimizer_v22 = torch.optim.Adam(
        train_v22.parameters(),
        **_adam(),
    )
    optimizer_v23 = torch.optim.Adam(
        train_v23.parameters(),
        **_adam(),
    )
    fresh_optimizers_empty = (
        not optimizer_v22.state and not optimizer_v23.state
    )
    steps: list[dict[str, object]] = []
    for index, batch in enumerate(batches, start=1):
        loss_v22 = _run_optimizer_step(
            train_v22,
            optimizer_v22,
            batch,
            loss_config=loss_config,
        )
        loss_v23 = _run_optimizer_step(
            train_v23,
            optimizer_v23,
            batch,
            loss_config=loss_config,
        )
        model_state_v22 = _model_state_observation(train_v22)
        model_state_v23 = _model_state_observation(train_v23)
        optimizer_state_v22 = _optimizer_component_observations(
            train_v22,
            optimizer_v22,
        )
        optimizer_state_v23 = _optimizer_component_observations(
            train_v23,
            optimizer_v23,
        )
        component_rows = {
            component: _matched_observation(
                optimizer_state_v22[component],
                optimizer_state_v23[component],
            )
            for component in ("step", "exp_avg", "exp_avg_sq")
        }
        loss_row = _matched_observation(
            _loss_observation(loss_v22),
            _loss_observation(loss_v23),
        )
        model_row = _matched_observation(
            model_state_v22,
            model_state_v23,
        )
        step_passed = (
            loss_row["passed"] is True
            and model_row["passed"] is True
            and all(
                row["passed"] is True
                for row in component_rows.values()
            )
        )
        steps.append(
            {
                "step": index,
                "batch_fingerprint": _batch_observation(batch)[
                    "raw_fingerprint"
                ],
                "loss": loss_row,
                "model_state": model_row,
                "optimizer_state": component_rows,
                "passed": step_passed,
            }
        )

    state_dict_row = _matched_observation(initial_v22, initial_v23)
    fields_row = _matched_observation(
        fields_observation_v22,
        fields_observation_v23,
    )
    gradient_row = _matched_observation(gradient_v22, gradient_v23)
    train_initial_row = _matched_observation(
        train_initial_v22,
        train_initial_v23,
    )
    optimizer_passed = (
        fresh_optimizers_empty
        and train_initial_row["passed"] is True
        and len(steps) == PACRE_VC_PARITY_ADAM_STEPS
        and all(step["passed"] is True for step in steps)
    )
    return {
        "device": str(device),
        "seed": seed,
        "model_config": {
            "feature_channels": PACRE_VC_PARITY_FEATURE_CHANNELS,
            "feature_stride": PACRE_VC_PARITY_FEATURE_STRIDE,
            "width": PACRE_VC_PARITY_WIDTH,
        },
        "generated_batches": [
            _batch_observation(batch) for batch in batches
        ],
        "state_dict_parity": state_dict_row,
        "all_fields_raw_parity": fields_row,
        "probe_gradient_parity": gradient_row,
        "probe_models_preserved": probe_models_preserved,
        "optimizer_parity": {
            "adam_config": {
                "learning_rate_hex": (
                    PACRE_VC_PARITY_ADAM_CONFIG[
                        "learning_rate"
                    ].hex()
                ),
                "betas_hex": [
                    PACRE_VC_PARITY_ADAM_CONFIG["beta1"].hex(),
                    PACRE_VC_PARITY_ADAM_CONFIG["beta2"].hex(),
                ],
                "epsilon_hex": (
                    PACRE_VC_PARITY_ADAM_CONFIG["epsilon"].hex()
                ),
                "weight_decay_hex": (
                    PACRE_VC_PARITY_ADAM_CONFIG[
                        "weight_decay"
                    ].hex()
                ),
            },
            "fresh_optimizer_state_empty": fresh_optimizers_empty,
            "initial_model_state": train_initial_row,
            "steps": steps,
            "passed": optimizer_passed,
        },
        "optimizer_steps_per_model": PACRE_VC_PARITY_ADAM_STEPS,
        "dataset_accessed": False,
        "cache_accessed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "external_models_received": 0,
    }


def run_pacre_vc_generated_parity_run(
    *,
    device: torch.device | str,
    seed: int,
) -> dict[str, object]:
    """Run one generated-only device/seed parity probe."""

    resolved = _resolve_device(device)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in PACRE_VC_PARITY_SEEDS
    ):
        raise ValueError("seed must be one of the frozen parity seeds")

    before_cpu_rng = torch.random.get_rng_state().clone()
    before_cuda_rng = (
        torch.cuda.get_rng_state(resolved).clone()
        if resolved.type == "cuda"
        else None
    )
    with _deterministic_execution_scope() as deterministic_execution:
        with torch.random.fork_rng(
            devices=(
                [int(resolved.index)]
                if resolved.type == "cuda"
                and resolved.index is not None
                else []
            ),
            device_type=(
                "cuda" if resolved.type == "cuda" else None
            ),
        ):
            payload = _run_generated_parity(
                device=resolved,
                seed=seed,
            )
    after_cuda_rng = (
        torch.cuda.get_rng_state(resolved).clone()
        if resolved.type == "cuda"
        else None
    )
    cpu_rng_preserved = torch.equal(
        before_cpu_rng,
        torch.random.get_rng_state(),
    )
    selected_rng_preserved = (
        before_cuda_rng is None
        or torch.equal(before_cuda_rng, after_cuda_rng)
    )
    determinism = {
        "policy": deterministic_execution["policy"],
        "active": deterministic_execution["active"],
        "restored_exactly": deterministic_execution[
            "restored_exactly"
        ],
    }
    gate_passed = (
        payload["state_dict_parity"]["passed"] is True
        and payload["all_fields_raw_parity"]["passed"] is True
        and payload["probe_gradient_parity"]["passed"] is True
        and payload["probe_models_preserved"] is True
        and payload["optimizer_parity"]["passed"] is True
        and cpu_rng_preserved
        and selected_rng_preserved
        and determinism["restored_exactly"] is True
    )
    body = {
        "schema_version": PACRE_VC_PARITY_RUN_SCHEMA,
        **payload,
        "global_cpu_rng_preserved": cpu_rng_preserved,
        "selected_device_rng_preserved": selected_rng_preserved,
        "deterministic_execution": determinism,
        "gate_passed": gate_passed,
    }
    return {**body, "receipt_fingerprint": stable_fingerprint(body)}


def run_pacre_vc_generated_parity_receipt(
    *,
    include_cuda: bool = True,
) -> dict[str, object]:
    """Run the frozen seed matrix and return one canonical receipt mapping."""

    if not isinstance(include_cuda, bool):
        raise TypeError("include_cuda must be bool")
    devices = ["cpu"]
    if include_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA parity matrix was requested but unavailable")
        devices.append("cuda:0")
    runs = [
        run_pacre_vc_generated_parity_run(
            device=device,
            seed=seed,
        )
        for device in devices
        for seed in PACRE_VC_PARITY_SEEDS
    ]
    expected_run_count = len(devices) * len(PACRE_VC_PARITY_SEEDS)
    body = {
        "schema_version": PACRE_VC_PARITY_SCHEMA,
        "candidate_pair": ["PACRE-v22", "PACRE-VC-v23"],
        "required_devices": devices,
        "required_seeds": list(PACRE_VC_PARITY_SEEDS),
        "runs": runs,
        "expected_run_count": expected_run_count,
        "observed_run_count": len(runs),
        "all_required_runs_present": len(runs) == expected_run_count,
        "gate_passed": (
            len(runs) == expected_run_count
            and all(run["gate_passed"] is True for run in runs)
        ),
        "generated_only": True,
        "dataset_accessed": False,
        "cache_accessed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    return {**body, "receipt_fingerprint": stable_fingerprint(body)}


__all__ = [
    "PACRE_VC_PARITY_ADAM_CONFIG",
    "PACRE_VC_PARITY_ADAM_STEPS",
    "PACRE_VC_PARITY_RUN_SCHEMA",
    "PACRE_VC_PARITY_SCHEMA",
    "PACRE_VC_PARITY_SEEDS",
    "run_pacre_vc_generated_parity_receipt",
    "run_pacre_vc_generated_parity_run",
]
