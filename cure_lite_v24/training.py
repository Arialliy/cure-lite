"""Strict, protocol-neutral PMOPE training core for v24 GCR-PACRE.

The core owns only one from-scratch model construction, one fresh Adam
optimizer, and one call to the existing public PMOPE trainer.  Evidence-stage
authorization, OOF root/fold closure, and evaluation are deliberately left to
the outer protocol layer.

The caller must state the role and seed explicitly.  The only accepted
bindings are seed 42 for OOF, bounded, and Formal primary training, and seed
43 for Formal training-integrity training.  Neither returned model is
authorized to execute on ``D_V`` or ``D_T``; only the seed-42 Formal primary
model can become eligible for a future external authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
from math import prod
from pathlib import Path
from threading import Lock
from typing import Callable, Final, Mapping

import torch

import cure_lite.experiment.coverage_state_training as _coverage_state_training_runtime
from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateTrainingSchedule,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    COVERAGE_STATE_FORMAL_SCOPE,
    CoverageStateRunAuthorization,
    CoverageStateTrainingResult,
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
    train_coverage_state_objective,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
)

from .factory import (
    GCR_PACRE_PARAMETER_NAMES,
    build_gcr_pacre_training_model,
)
from .gcr_pacre import (
    CURELiteGatedCommonResidualPACRELevelSet,
    CoverageStateGCRPACREConfig,
)


GCR_PACRE_TRAINING_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-pmope-training-receipt-v1"
)
GCR_PACRE_TRAINING_BUNDLE_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-pmope-training-bundle-v1"
)
GCR_PACRE_OBJECTIVE: Final = CoverageStatePairObjective.PMOPE_JOINT.value
GCR_PACRE_OPTIMIZER_FQCN: Final = "torch.optim.adam.Adam"
GCR_PACRE_MODEL_FQCN: Final = (
    "cure_lite_v24.gcr_pacre."
    "CURELiteGatedCommonResidualPACRELevelSet"
)
GCR_PACRE_CONFIG_FQCN: Final = (
    "cure_lite_v24.gcr_pacre.CoverageStateGCRPACREConfig"
)

GCR_PACRE_ROLE_OOF: Final = "oof"
GCR_PACRE_ROLE_BOUNDED: Final = "bounded"
GCR_PACRE_ROLE_PRIMARY: Final = "primary"
GCR_PACRE_ROLE_TRAINING_INTEGRITY: Final = "training_integrity_only"
GCR_PACRE_TRAINING_ROLES: Final = (
    GCR_PACRE_ROLE_OOF,
    GCR_PACRE_ROLE_BOUNDED,
    GCR_PACRE_ROLE_PRIMARY,
    GCR_PACRE_ROLE_TRAINING_INTEGRITY,
)

GCR_PACRE_SEED_ROLE_BINDINGS: Final = (
    (42, GCR_PACRE_ROLE_OOF),
    (42, GCR_PACRE_ROLE_BOUNDED),
    (42, GCR_PACRE_ROLE_PRIMARY),
    (43, GCR_PACRE_ROLE_TRAINING_INTEGRITY),
)
GCR_PACRE_ROLE_BUDGETS: Final = {
    GCR_PACRE_ROLE_OOF: (10, 40),
    GCR_PACRE_ROLE_BOUNDED: (10, 40),
    GCR_PACRE_ROLE_PRIMARY: (800, 40),
    GCR_PACRE_ROLE_TRAINING_INTEGRITY: (800, 40),
}
GCR_PACRE_ROLE_SCOPES: Final = {
    GCR_PACRE_ROLE_OOF: COVERAGE_STATE_BOUNDED_SCOPE,
    GCR_PACRE_ROLE_BOUNDED: COVERAGE_STATE_BOUNDED_SCOPE,
    GCR_PACRE_ROLE_PRIMARY: COVERAGE_STATE_FORMAL_SCOPE,
    GCR_PACRE_ROLE_TRAINING_INTEGRITY: COVERAGE_STATE_FORMAL_SCOPE,
}

GCR_PACRE_SOURCE_PATHS: Final = (
    "cure_lite_v24/gcr_pacre.py",
    "cure_lite_v24/factory.py",
    "cure_lite_v24/training.py",
)
_GCR_PACRE_STEP_TRACE_LOCK: Final = Lock()


def _fqcn(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_hashes() -> tuple[tuple[str, str], ...]:
    package_root = Path(__file__).resolve().parent
    paths = {
        "cure_lite_v24/gcr_pacre.py": package_root / "gcr_pacre.py",
        "cure_lite_v24/factory.py": package_root / "factory.py",
        "cure_lite_v24/training.py": package_root / "training.py",
    }
    if tuple(paths) != GCR_PACRE_SOURCE_PATHS:
        raise AssertionError("GCR-PACRE training source inventory changed")
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise RuntimeError(
            "GCR-PACRE training source is missing or not regular"
        )
    return tuple(
        (name, file_sha256(path)) for name, path in paths.items()
    )


@dataclass(frozen=True)
class GCRPACRETrainingPolicy:
    """One exact stage/seed binding with the unchanged PMOPE/Adam policy."""

    role: str
    seed: int
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if (
            type(self.role) is not str
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or (self.seed, self.role) not in GCR_PACRE_SEED_ROLE_BINDINGS
        ):
            raise ValueError("GCR-PACRE seed/role binding changed")
        frozen = {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1.0e-8,
        }
        for name, expected in frozen.items():
            value = getattr(self, name)
            if isinstance(value, bool) or value != expected:
                raise ValueError(f"GCR-PACRE training fixes {name}")

    @property
    def epochs(self) -> int:
        return GCR_PACRE_ROLE_BUDGETS[self.role][0]

    @property
    def steps_per_epoch(self) -> int:
        return GCR_PACRE_ROLE_BUDGETS[self.role][1]

    @property
    def completed_updates(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def scope(self) -> str:
        return GCR_PACRE_ROLE_SCOPES[self.role]

    @property
    def is_formal_primary(self) -> bool:
        return self.seed == 42 and self.role == GCR_PACRE_ROLE_PRIMARY

    def canonical_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "seed": self.seed,
            "scope": self.scope,
            "budget": {
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "updates": self.completed_updates,
            },
            "objective": GCR_PACRE_OBJECTIVE,
            "optimizer_fqcn": GCR_PACRE_OPTIMIZER_FQCN,
            "learning_rate_hex": self.learning_rate.hex(),
            "weight_decay_hex": self.weight_decay.hex(),
            "betas_hex": [
                self.adam_beta1.hex(),
                self.adam_beta2.hex(),
            ],
            "epsilon_hex": self.adam_epsilon.hex(),
            "training_invocations": 1,
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "checkpoint_policy": "final_only",
            "D_V_execution_authorized": False,
            "D_T_execution_authorized": False,
            "eligible_for_future_D_V_authorization_after_all_external_prerequisites": (
                self.is_formal_primary
            ),
            "eligible_for_future_D_T_authorization_after_all_external_prerequisites": (
                self.is_formal_primary
            ),
        }

    @property
    def policy_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


@dataclass(frozen=True)
class GCRPACREInitialParameterBinding:
    """Name, shape, and byte-exact content of one initial parameter."""

    name: str
    shape: tuple[int, ...]
    numel: int
    dtype: str
    byte_count: int
    content_fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.name not in GCR_PACRE_PARAMETER_NAMES
            or not self.shape
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in self.shape
            )
            or isinstance(self.numel, bool)
            or not isinstance(self.numel, int)
            or self.numel != prod(self.shape)
            or self.numel < 1
            or self.dtype != "torch.float32"
            or isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count != self.numel * 4
            or not _is_sha256(self.content_fingerprint)
        ):
            raise ValueError("invalid initial GCR-PACRE parameter binding")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "dtype": self.dtype,
            "byte_count": self.byte_count,
            "content_fingerprint": self.content_fingerprint,
        }


def _initial_parameter_bindings(
    model: CURELiteGatedCommonResidualPACRELevelSet,
) -> tuple[GCRPACREInitialParameterBinding, ...]:
    rows = tuple(
        GCRPACREInitialParameterBinding(
            name=name,
            shape=tuple(parameter.shape),
            numel=parameter.numel(),
            dtype=str(parameter.dtype),
            byte_count=parameter.numel() * parameter.element_size(),
            content_fingerprint=tensor_content_fingerprint(parameter),
        )
        for name, parameter in model.named_parameters()
    )
    if tuple(row.name for row in rows) != GCR_PACRE_PARAMETER_NAMES:
        raise AssertionError("GCR-PACRE parameter order changed")
    return rows


def _parameter_binding_fingerprint(
    rows: tuple[GCRPACREInitialParameterBinding, ...],
) -> str:
    return stable_fingerprint(
        [row.canonical_payload() for row in rows]
    )


def _construct_fresh_model(
    model_config: CoverageStateGCRPACREConfig,
    *,
    seed: int,
) -> tuple[
    CURELiteGatedCommonResidualPACRELevelSet,
    tuple[GCRPACREInitialParameterBinding, ...],
    str,
]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = build_gcr_pacre_training_model(model_config)
    if type(model) is not CURELiteGatedCommonResidualPACRELevelSet:
        raise AssertionError("GCR-PACRE factory returned the wrong model")
    rows = _initial_parameter_bindings(model)
    return model, rows, coverage_state_model_fingerprint(model)


@dataclass(frozen=True)
class GCRPACRETrainingReceipt:
    """Immutable role, source, initial-state, data, and compute binding."""

    schema_version: str
    role: str
    seed: int
    scope: str
    objective: str
    optimizer_fqcn: str
    policy_json: str
    policy_fingerprint: str
    model_fqcn: str
    config_fqcn: str
    model_contract_json: str
    model_contract_fingerprint: str
    source_hashes: tuple[tuple[str, str], ...]
    initial_parameters: tuple[GCRPACREInitialParameterBinding, ...]
    initial_parameter_state_fingerprint: str
    parameter_count: int
    cache_fingerprint: str
    schedule_fingerprint: str
    optimizer_config_fingerprint: str
    initial_model_fingerprint: str
    final_model_fingerprint: str
    training_result_fingerprint: str
    epochs: int
    steps_per_epoch: int
    completed_updates: int
    forward_calls: int
    backward_calls: int
    optimizer_steps: int
    training_invocations: int
    from_scratch: bool
    resume_allowed: bool
    automatic_retry_allowed: bool
    checkpoint_policy: str
    eligible_for_future_D_V_authorization_after_all_external_prerequisites: bool
    eligible_for_future_D_T_authorization_after_all_external_prerequisites: bool
    D_V_execution_authorized: bool
    D_T_execution_authorized: bool
    D_V_payload_accessed: bool
    D_T_payload_accessed: bool
    selection_effect: str
    may_replace_seed42_primary: bool

    def __post_init__(self) -> None:
        policy = GCRPACRETrainingPolicy(role=self.role, seed=self.seed)
        if (
            self.schema_version != GCR_PACRE_TRAINING_SCHEMA
            or self.scope != policy.scope
            or self.objective != GCR_PACRE_OBJECTIVE
            or self.optimizer_fqcn != GCR_PACRE_OPTIMIZER_FQCN
            or self.model_fqcn != GCR_PACRE_MODEL_FQCN
            or self.config_fqcn != GCR_PACRE_CONFIG_FQCN
            or self.policy_json != canonical_json(policy.canonical_payload())
            or self.policy_fingerprint != policy.policy_fingerprint
            or sha256(self.policy_json.encode("utf-8")).hexdigest()
            != self.policy_fingerprint
        ):
            raise ValueError("GCR-PACRE training identity changed")
        digests = (
            self.policy_fingerprint,
            self.model_contract_fingerprint,
            *(digest for _, digest in self.source_hashes),
            self.initial_parameter_state_fingerprint,
            self.cache_fingerprint,
            self.schedule_fingerprint,
            self.optimizer_config_fingerprint,
            self.initial_model_fingerprint,
            self.final_model_fingerprint,
            self.training_result_fingerprint,
        )
        if not all(_is_sha256(value) for value in digests):
            raise ValueError("GCR-PACRE training receipt has an invalid digest")
        if (
            tuple(name for name, _ in self.source_hashes)
            != GCR_PACRE_SOURCE_PATHS
            or sha256(self.model_contract_json.encode("utf-8")).hexdigest()
            != self.model_contract_fingerprint
            or tuple(row.name for row in self.initial_parameters)
            != GCR_PACRE_PARAMETER_NAMES
            or self.initial_parameter_state_fingerprint
            != _parameter_binding_fingerprint(self.initial_parameters)
            or self.parameter_count
            != sum(row.numel for row in self.initial_parameters)
        ):
            raise ValueError("GCR-PACRE model/source binding changed")
        expected_updates = self.epochs * self.steps_per_epoch
        if (
            (self.epochs, self.steps_per_epoch)
            != (policy.epochs, policy.steps_per_epoch)
            or self.completed_updates != expected_updates
            or self.forward_calls != expected_updates
            or self.backward_calls != expected_updates
            or self.optimizer_steps != expected_updates
            or self.training_invocations != 1
            or self.from_scratch is not True
            or self.resume_allowed is not False
            or self.automatic_retry_allowed is not False
            or self.checkpoint_policy != "final_only"
            or self.initial_model_fingerprint
            == self.final_model_fingerprint
        ):
            raise ValueError("GCR-PACRE training compute ledger is incomplete")
        future_primary = policy.is_formal_primary
        if (
            self.eligible_for_future_D_V_authorization_after_all_external_prerequisites
            is not future_primary
            or self.eligible_for_future_D_T_authorization_after_all_external_prerequisites
            is not future_primary
            or self.D_V_execution_authorized is not False
            or self.D_T_execution_authorized is not False
            or self.D_V_payload_accessed is not False
            or self.D_T_payload_accessed is not False
            or self.selection_effect
            != ("predeclared_primary" if future_primary else "none")
            or self.may_replace_seed42_primary is not False
        ):
            raise ValueError("GCR-PACRE evaluation-role firewall changed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "evaluation_role": self.role,
            "seed": self.seed,
            "scope": self.scope,
            "objective": self.objective,
            "optimizer_fqcn": self.optimizer_fqcn,
            "policy_json": self.policy_json,
            "policy_fingerprint": self.policy_fingerprint,
            "model": {
                "model_fqcn": self.model_fqcn,
                "config_fqcn": self.config_fqcn,
                "contract_json": self.model_contract_json,
                "contract_fingerprint": self.model_contract_fingerprint,
                "parameter_count": self.parameter_count,
                "initial_parameters": [
                    row.canonical_payload()
                    for row in self.initial_parameters
                ],
                "initial_parameter_state_fingerprint": (
                    self.initial_parameter_state_fingerprint
                ),
                "initial_fingerprint": self.initial_model_fingerprint,
                "final_fingerprint": self.final_model_fingerprint,
            },
            "source_hashes": dict(self.source_hashes),
            "cache_fingerprint": self.cache_fingerprint,
            "schedule_fingerprint": self.schedule_fingerprint,
            "optimizer_config_fingerprint": (
                self.optimizer_config_fingerprint
            ),
            "training_result_fingerprint": (
                self.training_result_fingerprint
            ),
            "budget": {
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "updates": self.completed_updates,
                "training_invocations": self.training_invocations,
            },
            "compute": {
                "completed_updates": self.completed_updates,
                "forward_calls": self.forward_calls,
                "backward_calls": self.backward_calls,
                "optimizer_steps": self.optimizer_steps,
            },
            "from_scratch": self.from_scratch,
            "resume_allowed": self.resume_allowed,
            "automatic_retry_allowed": self.automatic_retry_allowed,
            "checkpoint_policy": self.checkpoint_policy,
            "eligible_for_future_D_V_authorization_after_all_external_prerequisites": (
                self.eligible_for_future_D_V_authorization_after_all_external_prerequisites
            ),
            "eligible_for_future_D_T_authorization_after_all_external_prerequisites": (
                self.eligible_for_future_D_T_authorization_after_all_external_prerequisites
            ),
            "D_V_execution_authorized": self.D_V_execution_authorized,
            "D_T_execution_authorized": self.D_T_execution_authorized,
            "D_V_payload_accessed": self.D_V_payload_accessed,
            "D_T_payload_accessed": self.D_T_payload_accessed,
            "selection_effect": self.selection_effect,
            "may_replace_seed42_primary": self.may_replace_seed42_primary,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


@dataclass(frozen=True, eq=False)
class GCRPACRETrainingBundle:
    """One terminal model bound to its exact training receipt."""

    model_config: CoverageStateGCRPACREConfig
    policy: GCRPACRETrainingPolicy
    model: CURELiteGatedCommonResidualPACRELevelSet
    training_result: CoverageStateTrainingResult
    receipt: GCRPACRETrainingReceipt

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        if (
            type(self.model_config) is not CoverageStateGCRPACREConfig
            or type(self.policy) is not GCRPACRETrainingPolicy
            or type(self.model)
            is not CURELiteGatedCommonResidualPACRELevelSet
            or type(self.training_result) is not CoverageStateTrainingResult
            or type(self.receipt) is not GCRPACRETrainingReceipt
            or self.model.config is not self.model_config
        ):
            raise TypeError("GCR-PACRE training bundle contains a wrong type")
        _, initial_rows, initial_model_fingerprint = _construct_fresh_model(
            self.model_config,
            seed=self.policy.seed,
        )
        contract = coverage_state_model_contract_payload(self.model)
        contract_json = canonical_json(contract)
        result = self.training_result
        receipt = self.receipt
        if (
            receipt.role != self.policy.role
            or receipt.seed != self.policy.seed
            or receipt.policy_fingerprint != self.policy.policy_fingerprint
            or receipt.model_fqcn != _fqcn(self.model)
            or receipt.config_fqcn != _fqcn(self.model_config)
            or receipt.model_contract_json != contract_json
            or receipt.model_contract_fingerprint
            != stable_fingerprint(contract)
            or receipt.source_hashes != _source_hashes()
            or receipt.initial_parameters != initial_rows
            or receipt.initial_parameter_state_fingerprint
            != _parameter_binding_fingerprint(initial_rows)
            or receipt.initial_model_fingerprint
            != initial_model_fingerprint
            or receipt.initial_model_fingerprint
            != result.initial_model_fingerprint
            or receipt.final_model_fingerprint
            != result.final_model_fingerprint
            or receipt.final_model_fingerprint
            != coverage_state_model_fingerprint(self.model)
            or receipt.training_result_fingerprint
            != result.result_fingerprint
            or receipt.cache_fingerprint != result.cache_fingerprint
            or receipt.schedule_fingerprint
            != result.schedule_fingerprint
            or receipt.optimizer_config_fingerprint
            != result.optimizer_config_fingerprint
            or receipt.completed_updates != result.completed_updates
            or receipt.forward_calls != result.forward_calls
            or receipt.backward_calls != result.backward_calls
            or receipt.optimizer_steps != result.optimizer_steps
            or result.seed != self.policy.seed
            or result.objective != GCR_PACRE_OBJECTIVE
        ):
            raise ValueError(
                "GCR-PACRE trained model/receipt binding changed"
            )

    @property
    def bundle_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(
            {
                "schema_version": GCR_PACRE_TRAINING_BUNDLE_SCHEMA,
                "receipt_fingerprint": self.receipt.receipt_fingerprint,
                "training_result_fingerprint": (
                    self.training_result.result_fingerprint
                ),
            }
        )


def _resolve_device(device: torch.device | str) -> torch.device:
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise TypeError("device must identify a torch device") from error
    if resolved.type == "cuda" and resolved.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _trace_tensor_content_fingerprint(tensor: torch.Tensor) -> str:
    """Hash dtype, original shape, and exact bytes, including 0-D tensors."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("training trace state must be a tensor")
    value = tensor.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            list(value.shape),
            separators=(",", ":"),
        ).encode("ascii")
    )
    if value.numel():
        digest.update(
            value.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def gcr_pacre_training_state_summary_fingerprint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[str, str, int]:
    """Return byte-exact parameter/optimizer digests and the real Adam step."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("trace model must be a torch module")
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError("trace optimizer must be exact Adam")

    def row(name: str, tensor: torch.Tensor) -> dict[str, object]:
        value = tensor.detach()
        if not bool(torch.isfinite(value).all().detach().cpu()):
            raise FloatingPointError(
                f"training trace encountered non-finite state in {name}"
            )
        return {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "content_fingerprint": (
                _trace_tensor_content_fingerprint(value)
            ),
        }

    parameter_rows = [
        row(f"parameter:{name}", parameter)
        for name, parameter in model.named_parameters()
    ]
    optimizer_rows: list[dict[str, object]] = []
    step_values: set[int] = set()
    for parameter_index, parameter in enumerate(model.parameters()):
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict):
            raise RuntimeError("training trace lacks optimizer state")
        step = state.get("step")
        if not isinstance(step, torch.Tensor) or step.numel() != 1:
            raise RuntimeError("training trace lacks an exact Adam step")
        raw_step = float(step.detach().cpu().item())
        integer_step = int(raw_step)
        if raw_step != float(integer_step) or integer_step < 1:
            raise RuntimeError("training trace Adam step is not integral")
        step_values.add(integer_step)
        for state_name in sorted(state):
            state_value = state[state_name]
            if not isinstance(state_value, torch.Tensor):
                raise TypeError("optimizer state must remain tensor-only")
            optimizer_rows.append(
                row(
                    f"optimizer:{parameter_index}:{state_name}",
                    state_value,
                )
            )
    if len(step_values) != 1:
        raise RuntimeError("Adam parameter step counters diverged")
    return (
        stable_fingerprint(parameter_rows),
        stable_fingerprint(optimizer_rows),
        next(iter(step_values)),
    )


def _trace_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not torch.isfinite(torch.tensor(result, dtype=torch.float64)):
        raise FloatingPointError(f"{name} is non-finite")
    return result


def _verify_step_gradients_finite(
    model: CURELiteGatedCommonResidualPACRELevelSet,
) -> None:
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None or not bool(
            torch.isfinite(gradient).all().detach().cpu()
        ):
            raise FloatingPointError(
                f"training trace gradient is absent/non-finite for {name}"
            )


def _train_public_objective_with_step_trace(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    optimizer: torch.optim.Adam,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    resolved_device: torch.device,
    initial_fingerprint: str,
    authorization: CoverageStateRunAuthorization,
    epoch_callback: Callable[[Mapping[str, object]], None] | None,
    update_callback: Callable[[Mapping[str, object]], None] | None,
) -> CoverageStateTrainingResult:
    """Trace protected steps without changing the r2-bound public trainer."""

    call_kwargs = {
        "objective": CoverageStatePairObjective.PMOPE_JOINT,
        "device": resolved_device,
        "expected_initial_model_fingerprint": initial_fingerprint,
        "authorization": authorization,
        "epoch_callback": epoch_callback,
    }
    if update_callback is None:
        return train_coverage_state_objective(
            model,
            optimizer,
            cache,
            schedule,
            **call_kwargs,
        )

    # Test doubles may expose an explicit callback.  Production reaches the
    # fixed r2-bound trainer, whose bytes/signature must remain untouched.
    if "update_callback" in inspect.signature(
        train_coverage_state_objective
    ).parameters:
        return train_coverage_state_objective(
            model,
            optimizer,
            cache,
            schedule,
            **call_kwargs,
            update_callback=update_callback,
        )

    original_step = (
        _coverage_state_training_runtime.coverage_state_fused_train_step
    )
    if (
        getattr(original_step, "__module__", None)
        != "cure_lite.train.coverage_state_fused_step"
        or getattr(original_step, "__name__", None)
        != "coverage_state_fused_train_step"
    ):
        raise RuntimeError("public PMOPE fused-step binding changed")
    completed = 0

    def traced_step(*args, **kwargs):
        nonlocal completed
        if (
            len(args) < 3
            or args[0] is not model
            or args[1] is not optimizer
            or completed >= schedule.config.updates
        ):
            raise RuntimeError("protected PMOPE step identity changed")
        logs = original_step(*args, **kwargs)
        if not isinstance(logs, Mapping):
            raise TypeError("public PMOPE step did not return logs")
        selection = schedule.selections[completed]
        batch = args[2]
        if (
            logs.get("selection_fingerprint")
            != getattr(batch, "selection_fingerprint", None)
            or getattr(
                getattr(batch, "factual_miss", None),
                "record_ids",
                None,
            )
            != selection.factual_miss_record_ids
            or getattr(
                getattr(batch, "factual_no_miss", None),
                "record_ids",
                None,
            )
            != selection.factual_no_miss_record_ids
            or getattr(
                getattr(batch, "pairs", None),
                "pair_ids",
                None,
            )
            != (
                selection.clean_positive_pair_id,
                selection.component_null_pair_id,
            )
        ):
            raise RuntimeError("protected PMOPE schedule order changed")
        _verify_step_gradients_finite(model)
        (
            parameter_state_digest,
            optimizer_state_digest,
            optimizer_step_counter,
        ) = gcr_pacre_training_state_summary_fingerprint(
            model,
            optimizer,
        )
        if optimizer_step_counter != completed + 1:
            raise RuntimeError("protected PMOPE Adam counter changed")
        update_callback(
            {
                "update": completed,
                "epoch": selection.epoch,
                "step": selection.step,
                "selection_fingerprint": (
                    selection.selection_fingerprint
                ),
                "loss": _trace_finite_real(
                    logs.get("total"),
                    name="trace.loss",
                ),
                "gradient_l2_norm": _trace_finite_real(
                    logs.get("gradient_l2_norm"),
                    name="trace.gradient_l2_norm",
                ),
                "optimizer_step_counter": optimizer_step_counter,
                "parameter_state_digest": parameter_state_digest,
                "optimizer_state_digest": optimizer_state_digest,
                "loss_finite": True,
                "gradients_finite": True,
                "parameters_finite": True,
                "optimizer_state_finite": True,
            }
        )
        completed += 1
        return logs

    with _GCR_PACRE_STEP_TRACE_LOCK:
        if (
            _coverage_state_training_runtime.coverage_state_fused_train_step
            is not original_step
        ):
            raise RuntimeError("public PMOPE fused-step binding raced")
        _coverage_state_training_runtime.coverage_state_fused_train_step = (
            traced_step
        )
        try:
            result = train_coverage_state_objective(
                model,
                optimizer,
                cache,
                schedule,
                **call_kwargs,
            )
        finally:
            observed_step = (
                _coverage_state_training_runtime.coverage_state_fused_train_step
            )
            _coverage_state_training_runtime.coverage_state_fused_train_step = (
                original_step
            )
            if observed_step is not traced_step:
                raise RuntimeError(
                    "public PMOPE fused-step binding changed during trace"
                )
    if completed != schedule.config.updates:
        raise RuntimeError("protected PMOPE trace update count changed")
    return result


def train_gcr_pacre_pmope_candidate(
    model_config: object,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    role: str,
    seed: int,
    authorization: CoverageStateRunAuthorization,
    device: torch.device | str = "cpu",
    epoch_callback: Callable[[Mapping[str, object]], None] | None = None,
    update_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> GCRPACRETrainingBundle:
    """Train one exact GCR-PACRE model through the public PMOPE path."""

    if type(model_config) is not CoverageStateGCRPACREConfig:
        raise TypeError(
            "model_config must have exact type CoverageStateGCRPACREConfig"
        )
    if not isinstance(cache, CoverageStateScalarCache):
        raise TypeError("cache must be CoverageStateScalarCache")
    if not isinstance(schedule, CoverageStateTrainingSchedule):
        raise TypeError("schedule must be CoverageStateTrainingSchedule")
    if not isinstance(authorization, CoverageStateRunAuthorization):
        raise TypeError(
            "authorization must be a CoverageStateRunAuthorization"
        )
    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError("epoch_callback must be callable or None")
    if update_callback is not None and not callable(update_callback):
        raise TypeError("update_callback must be callable or None")
    policy = GCRPACRETrainingPolicy(role=role, seed=seed)
    if schedule.config.seed != policy.seed:
        raise ValueError("GCR-PACRE training and schedule seed differ")
    if (
        schedule.config.epochs,
        schedule.config.steps_per_epoch,
    ) != (policy.epochs, policy.steps_per_epoch):
        raise PermissionError(
            "GCR-PACRE schedule differs from the explicit role budget"
        )
    if schedule.cache_fingerprint != cache.cache_fingerprint:
        raise ValueError("GCR-PACRE schedule and cache differ")

    resolved_device = _resolve_device(device)
    source_hashes = _source_hashes()
    model, initial_rows, initial_fingerprint = _construct_fresh_model(
        model_config,
        seed=policy.seed,
    )
    model_contract = coverage_state_model_contract_payload(model)
    model_contract_json = canonical_json(model_contract)
    model = model.to(device=resolved_device, dtype=torch.float32)
    if coverage_state_model_fingerprint(model) != initial_fingerprint:
        raise RuntimeError("GCR-PACRE device move changed initial bytes")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=policy.learning_rate,
        betas=(policy.adam_beta1, policy.adam_beta2),
        eps=policy.adam_epsilon,
        weight_decay=policy.weight_decay,
    )
    if type(optimizer) is not torch.optim.Adam or optimizer.state:
        raise RuntimeError("GCR-PACRE requires one fresh exact Adam")
    result = _train_public_objective_with_step_trace(
        model,
        optimizer,
        cache,
        schedule,
        resolved_device=resolved_device,
        initial_fingerprint=initial_fingerprint,
        authorization=authorization,
        epoch_callback=epoch_callback,
        update_callback=update_callback,
    )
    if type(result) is not CoverageStateTrainingResult:
        raise TypeError(
            "public training must return CoverageStateTrainingResult"
        )
    if (
        source_hashes != _source_hashes()
        or result.seed != policy.seed
        or result.epochs != policy.epochs
        or result.steps_per_epoch != policy.steps_per_epoch
        or result.completed_updates != policy.completed_updates
        or result.cache_fingerprint != cache.cache_fingerprint
        or result.schedule_fingerprint != schedule.schedule_fingerprint
        or result.objective != GCR_PACRE_OBJECTIVE
        or result.initial_model_fingerprint != initial_fingerprint
        or result.final_model_fingerprint
        != coverage_state_model_fingerprint(model)
    ):
        raise RuntimeError("GCR-PACRE terminal training binding changed")

    future_primary = policy.is_formal_primary
    receipt = GCRPACRETrainingReceipt(
        schema_version=GCR_PACRE_TRAINING_SCHEMA,
        role=policy.role,
        seed=policy.seed,
        scope=policy.scope,
        objective=GCR_PACRE_OBJECTIVE,
        optimizer_fqcn=_fqcn(optimizer),
        policy_json=canonical_json(policy.canonical_payload()),
        policy_fingerprint=policy.policy_fingerprint,
        model_fqcn=_fqcn(model),
        config_fqcn=_fqcn(model_config),
        model_contract_json=model_contract_json,
        model_contract_fingerprint=stable_fingerprint(model_contract),
        source_hashes=source_hashes,
        initial_parameters=initial_rows,
        initial_parameter_state_fingerprint=(
            _parameter_binding_fingerprint(initial_rows)
        ),
        parameter_count=sum(row.numel for row in initial_rows),
        cache_fingerprint=result.cache_fingerprint,
        schedule_fingerprint=result.schedule_fingerprint,
        optimizer_config_fingerprint=(
            result.optimizer_config_fingerprint
        ),
        initial_model_fingerprint=result.initial_model_fingerprint,
        final_model_fingerprint=result.final_model_fingerprint,
        training_result_fingerprint=result.result_fingerprint,
        epochs=result.epochs,
        steps_per_epoch=result.steps_per_epoch,
        completed_updates=result.completed_updates,
        forward_calls=result.forward_calls,
        backward_calls=result.backward_calls,
        optimizer_steps=result.optimizer_steps,
        training_invocations=1,
        from_scratch=True,
        resume_allowed=False,
        automatic_retry_allowed=False,
        checkpoint_policy="final_only",
        eligible_for_future_D_V_authorization_after_all_external_prerequisites=(
            future_primary
        ),
        eligible_for_future_D_T_authorization_after_all_external_prerequisites=(
            future_primary
        ),
        D_V_execution_authorized=False,
        D_T_execution_authorized=False,
        D_V_payload_accessed=False,
        D_T_payload_accessed=False,
        selection_effect=(
            "predeclared_primary" if future_primary else "none"
        ),
        may_replace_seed42_primary=False,
    )
    return GCRPACRETrainingBundle(
        model_config=model_config,
        policy=policy,
        model=model,
        training_result=result,
        receipt=receipt,
    )


__all__ = [
    "GCR_PACRE_CONFIG_FQCN",
    "GCR_PACRE_MODEL_FQCN",
    "GCR_PACRE_OBJECTIVE",
    "GCR_PACRE_OPTIMIZER_FQCN",
    "GCR_PACRE_ROLE_BOUNDED",
    "GCR_PACRE_ROLE_BUDGETS",
    "GCR_PACRE_ROLE_OOF",
    "GCR_PACRE_ROLE_PRIMARY",
    "GCR_PACRE_ROLE_SCOPES",
    "GCR_PACRE_ROLE_TRAINING_INTEGRITY",
    "GCR_PACRE_SEED_ROLE_BINDINGS",
    "GCR_PACRE_SOURCE_PATHS",
    "GCR_PACRE_TRAINING_BUNDLE_SCHEMA",
    "GCR_PACRE_TRAINING_ROLES",
    "GCR_PACRE_TRAINING_SCHEMA",
    "GCRPACREInitialParameterBinding",
    "GCRPACRETrainingBundle",
    "GCRPACRETrainingPolicy",
    "GCRPACRETrainingReceipt",
    "gcr_pacre_training_state_summary_fingerprint",
    "train_gcr_pacre_pmope_candidate",
]
