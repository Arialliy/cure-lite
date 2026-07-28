"""Independent single-candidate PMOPE training entry for v22 PACRE.

This module does not use the inheritance-based legacy model registry or any
matched-suite result.  It constructs PACRE only through the strict v22
factory, creates the frozen Adam optimizer, and calls the existing public
single-objective training function with the fixed PMOPE objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Callable, Final, Mapping

import torch

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
    CoverageStateTrainingResult,
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
    train_coverage_state_objective,
)
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
)
from cure_lite_v22.factory import (
    PACRE_PARAMETER_NAMES,
    build_pacre_training_model,
)
from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
)


PACRE_PMOPE_TRAINING_SCHEMA: Final = (
    "cure-lite-v22-pacre-pmope-training-receipt-v1"
)
PACRE_PMOPE_TRAINING_SEED: Final = 42
PACRE_PMOPE_OBJECTIVE: Final = (
    CoverageStatePairObjective.PMOPE_JOINT.value
)
PACRE_MODEL_FQCN: Final = (
    "cure_lite_v22.pacre."
    "CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet"
)
PACRE_CONFIG_FQCN: Final = (
    "cure_lite_v22.pacre.CoverageStatePACREConfig"
)
PACRE_OPTIMIZER_FQCN: Final = "torch.optim.adam.Adam"
PACRE_SOURCE_PATHS: Final = (
    "cure_lite_v22/pacre.py",
    "cure_lite_v22/factory.py",
    "cure_lite_v22/training.py",
)


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
        "cure_lite_v22/pacre.py": package_root / "pacre.py",
        "cure_lite_v22/factory.py": package_root / "factory.py",
        "cure_lite_v22/training.py": package_root / "training.py",
    }
    if tuple(paths) != PACRE_SOURCE_PATHS:
        raise AssertionError("PACRE source inventory changed")
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise RuntimeError("PACRE training source is missing or not regular")
    return tuple(
        (name, file_sha256(path)) for name, path in paths.items()
    )


@dataclass(frozen=True)
class PACREPMOPETrainingConfig:
    """Frozen single-seed Adam policy for the v22 candidate."""

    seed: int = PACRE_PMOPE_TRAINING_SEED
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        frozen = {
            "seed": PACRE_PMOPE_TRAINING_SEED,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1.0e-8,
        }
        for name, expected in frozen.items():
            value = getattr(self, name)
            if isinstance(value, bool) or value != expected:
                raise ValueError(f"PACRE PMOPE training fixes {name}")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "objective": PACRE_PMOPE_OBJECTIVE,
            "optimizer_fqcn": PACRE_OPTIMIZER_FQCN,
            "learning_rate_hex": self.learning_rate.hex(),
            "weight_decay_hex": self.weight_decay.hex(),
            "betas_hex": [
                self.adam_beta1.hex(),
                self.adam_beta2.hex(),
            ],
            "epsilon_hex": self.adam_epsilon.hex(),
        }

    @property
    def config_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


PACRE_PMOPE_TRAINING_CONFIG: Final = PACREPMOPETrainingConfig()


@dataclass(frozen=True)
class PACREParameterTopology:
    """One immutable trainable-parameter contract row."""

    name: str
    shape: tuple[int, ...]
    numel: int
    dtype: str
    requires_grad: bool

    def __post_init__(self) -> None:
        if (
            self.name not in PACRE_PARAMETER_NAMES
            or not self.shape
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in self.shape
            )
            or self.numel < 1
            or self.numel != prod(self.shape)
            or self.dtype != "torch.float32"
            or self.requires_grad is not True
        ):
            raise ValueError("invalid PACRE parameter topology")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
        }


def _parameter_topology(
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
) -> tuple[PACREParameterTopology, ...]:
    rows = tuple(
        PACREParameterTopology(
            name=name,
            shape=tuple(parameter.shape),
            numel=parameter.numel(),
            dtype=str(parameter.dtype),
            requires_grad=parameter.requires_grad,
        )
        for name, parameter in model.named_parameters()
    )
    if tuple(row.name for row in rows) != PACRE_PARAMETER_NAMES:
        raise AssertionError("PACRE parameter topology changed")
    return rows


@dataclass(frozen=True)
class PACREPMOPETrainingReceipt:
    """Immutable source, structure, state, and compute binding."""

    schema_version: str
    seed: int
    objective: str
    optimizer_fqcn: str
    training_config_json: str
    training_config_fingerprint: str
    model_fqcn: str
    config_fqcn: str
    model_contract_json: str
    model_contract_fingerprint: str
    source_hashes: tuple[tuple[str, str], ...]
    parameter_topology: tuple[PACREParameterTopology, ...]
    parameter_count: int
    cache_fingerprint: str
    schedule_fingerprint: str
    optimizer_config_fingerprint: str
    initial_model_fingerprint: str
    final_model_fingerprint: str
    training_result_fingerprint: str
    completed_updates: int
    forward_calls: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != PACRE_PMOPE_TRAINING_SCHEMA
            or self.seed != PACRE_PMOPE_TRAINING_SEED
            or self.objective != PACRE_PMOPE_OBJECTIVE
            or self.optimizer_fqcn != PACRE_OPTIMIZER_FQCN
            or self.model_fqcn != PACRE_MODEL_FQCN
            or self.config_fqcn != PACRE_CONFIG_FQCN
        ):
            raise ValueError("PACRE training receipt policy changed")
        digests = (
            self.training_config_fingerprint,
            self.model_contract_fingerprint,
            *(digest for _, digest in self.source_hashes),
            self.cache_fingerprint,
            self.schedule_fingerprint,
            self.optimizer_config_fingerprint,
            self.initial_model_fingerprint,
            self.final_model_fingerprint,
            self.training_result_fingerprint,
        )
        if not all(_is_sha256(value) for value in digests):
            raise ValueError("PACRE training receipt has an invalid digest")
        if tuple(name for name, _ in self.source_hashes) != (
            PACRE_SOURCE_PATHS
        ):
            raise ValueError("PACRE source hash inventory changed")
        if sha256(self.model_contract_json.encode("utf-8")).hexdigest() != (
            self.model_contract_fingerprint
        ):
            raise ValueError("PACRE model contract fingerprint differs")
        if sha256(
            self.training_config_json.encode("utf-8")
        ).hexdigest() != self.training_config_fingerprint:
            raise ValueError("PACRE training config fingerprint differs")
        if (
            tuple(row.name for row in self.parameter_topology)
            != PACRE_PARAMETER_NAMES
            or self.parameter_count
            != sum(row.numel for row in self.parameter_topology)
            or self.completed_updates < 1
            or self.forward_calls != self.completed_updates
            or self.initial_model_fingerprint
            == self.final_model_fingerprint
        ):
            raise ValueError("PACRE training receipt is incomplete")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "objective": self.objective,
            "optimizer_fqcn": self.optimizer_fqcn,
            "training_config_json": self.training_config_json,
            "training_config_fingerprint": (
                self.training_config_fingerprint
            ),
            "model": {
                "model_fqcn": self.model_fqcn,
                "config_fqcn": self.config_fqcn,
                "contract_json": self.model_contract_json,
                "contract_fingerprint": (
                    self.model_contract_fingerprint
                ),
                "parameter_count": self.parameter_count,
                "parameter_topology": [
                    row.canonical_payload()
                    for row in self.parameter_topology
                ],
                "initial_fingerprint": (
                    self.initial_model_fingerprint
                ),
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
            "compute": {
                "completed_updates": self.completed_updates,
                "forward_calls": self.forward_calls,
            },
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


@dataclass(frozen=True, eq=False)
class PACREPMOPETrainingBundle:
    """Frozen binding of one trained PACRE model to its receipt."""

    model_config: CoverageStatePACREConfig
    training_config: PACREPMOPETrainingConfig
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
    training_result: CoverageStateTrainingResult
    receipt: PACREPMOPETrainingReceipt

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        if (
            type(self.model_config) is not CoverageStatePACREConfig
            or type(self.training_config) is not PACREPMOPETrainingConfig
            or type(self.model)
            is not
            CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
            or type(self.training_result) is not CoverageStateTrainingResult
            or type(self.receipt) is not PACREPMOPETrainingReceipt
            or self.model.config is not self.model_config
        ):
            raise TypeError("PACRE training bundle contains a wrong type")
        model_contract = coverage_state_model_contract_payload(self.model)
        model_contract_json = canonical_json(model_contract)
        if (
            self.receipt.training_config_fingerprint
            != self.training_config.config_fingerprint
            or self.receipt.training_config_json
            != canonical_json(self.training_config.canonical_payload())
            or self.receipt.model_fqcn != _fqcn(self.model)
            or self.receipt.config_fqcn != _fqcn(self.model_config)
            or self.receipt.model_contract_json != model_contract_json
            or self.receipt.model_contract_fingerprint
            != stable_fingerprint(model_contract)
            or self.receipt.source_hashes != _source_hashes()
            or self.receipt.parameter_topology
            != _parameter_topology(self.model)
            or self.receipt.parameter_count
            != sum(parameter.numel() for parameter in self.model.parameters())
            or self.receipt.initial_model_fingerprint
            != self.training_result.initial_model_fingerprint
            or self.receipt.final_model_fingerprint
            != self.training_result.final_model_fingerprint
            or self.receipt.final_model_fingerprint
            != coverage_state_model_fingerprint(self.model)
            or self.receipt.training_result_fingerprint
            != self.training_result.result_fingerprint
            or self.receipt.cache_fingerprint
            != self.training_result.cache_fingerprint
            or self.receipt.schedule_fingerprint
            != self.training_result.schedule_fingerprint
            or self.receipt.optimizer_config_fingerprint
            != self.training_result.optimizer_config_fingerprint
            or self.receipt.completed_updates
            != self.training_result.completed_updates
            or self.receipt.forward_calls
            != self.training_result.forward_calls
            or self.training_result.seed != self.training_config.seed
            or self.training_result.objective != PACRE_PMOPE_OBJECTIVE
        ):
            raise ValueError("PACRE trained model/receipt binding changed")

    @property
    def bundle_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(
            {
                "schema_version": (
                    "cure-lite-v22-pacre-pmope-training-bundle-v1"
                ),
                "receipt_fingerprint": (
                    self.receipt.receipt_fingerprint
                ),
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


def _reserve_protected_training_before_allocation(
    authorization: object | None,
    model_config: CoverageStatePACREConfig,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    device: torch.device | str,
) -> object | None:
    schedule_config = schedule.config
    bounded = (
        schedule_config.epochs == 10
        and schedule_config.steps_per_epoch == 40
    )
    formal = (
        schedule_config.epochs == 800
        and schedule_config.steps_per_epoch == 40
    )
    if not bounded and not formal:
        if authorization is not None:
            raise ValueError(
                "development PACRE training rejects an authorization"
            )
        return None
    if formal:
        raise PermissionError(
            "PACRE Formal800 requires its own unimplemented authorization"
        )
    from .bounded_runner import (
        CoverageStatePACREBoundedRunAuthorization,
    )

    if type(authorization) is not (
        CoverageStatePACREBoundedRunAuthorization
    ):
        raise TypeError(
            "bounded PACRE training requires its exact authorization"
        )
    if authorization.reserved:
        authorization.verify_reserved_for_training(
            model_config=model_config,
            cache=cache,
            schedule=schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
            device=device,
        )
    else:
        authorization.claim_for_training(
            model_config=model_config,
            cache=cache,
            schedule=schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
            device=device,
        )
    return authorization


def train_pacre_pmope_candidate(
    model_config: object,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    *,
    config: PACREPMOPETrainingConfig = PACRE_PMOPE_TRAINING_CONFIG,
    device: torch.device | str = "cpu",
    authorization: object | None = None,
    epoch_callback: (
        Callable[[Mapping[str, object]], None] | None
    ) = None,
) -> PACREPMOPETrainingBundle:
    """Train one exact PACRE candidate through the public PMOPE path."""

    if type(model_config) is not CoverageStatePACREConfig:
        raise TypeError(
            "model_config must have exact type CoverageStatePACREConfig"
        )
    if type(config) is not PACREPMOPETrainingConfig:
        raise TypeError(
            "config must have exact type PACREPMOPETrainingConfig"
        )
    if not isinstance(cache, CoverageStateScalarCache):
        raise TypeError("cache must be CoverageStateScalarCache")
    if not isinstance(schedule, CoverageStateTrainingSchedule):
        raise TypeError("schedule must be CoverageStateTrainingSchedule")
    if schedule.config.seed != config.seed:
        raise ValueError("PACRE training and schedule seed differ")
    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError("epoch_callback must be callable or None")

    protected_authorization = _reserve_protected_training_before_allocation(
        authorization,
        model_config,
        cache,
        schedule,
        device,
    )
    try:
        resolved_device = _resolve_device(device)
        source_hashes = _source_hashes()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            model = build_pacre_training_model(model_config)
        model = model.to(device=resolved_device, dtype=torch.float32)
        if type(model) is not (
            CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
        ):
            raise AssertionError("PACRE factory returned the wrong model")

        model_contract = coverage_state_model_contract_payload(model)
        model_contract_json = canonical_json(model_contract)
        topology = _parameter_topology(model)
        initial_fingerprint = coverage_state_model_fingerprint(model)
        if protected_authorization is not None:
            protected_authorization.consume_for_training(
                model=model,
                model_config=model_config,
                cache=cache,
                schedule=schedule,
                scope=COVERAGE_STATE_BOUNDED_SCOPE,
                device=resolved_device,
                objective=PACRE_PMOPE_OBJECTIVE,
                initial_model_fingerprint=initial_fingerprint,
            )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        result = train_coverage_state_objective(
            model,
            optimizer,
            cache,
            schedule,
            objective=CoverageStatePairObjective.PMOPE_JOINT,
            device=resolved_device,
            expected_initial_model_fingerprint=initial_fingerprint,
            authorization=authorization,
            epoch_callback=epoch_callback,
        )
        if type(result) is not CoverageStateTrainingResult:
            raise TypeError(
                "public training must return CoverageStateTrainingResult"
            )
        if source_hashes != _source_hashes():
            raise RuntimeError(
                "PACRE training source changed during execution"
            )

        receipt = PACREPMOPETrainingReceipt(
            schema_version=PACRE_PMOPE_TRAINING_SCHEMA,
            seed=config.seed,
            objective=PACRE_PMOPE_OBJECTIVE,
            optimizer_fqcn=_fqcn(optimizer),
            training_config_json=canonical_json(
                config.canonical_payload()
            ),
            training_config_fingerprint=config.config_fingerprint,
            model_fqcn=_fqcn(model),
            config_fqcn=_fqcn(model_config),
            model_contract_json=model_contract_json,
            model_contract_fingerprint=stable_fingerprint(model_contract),
            source_hashes=source_hashes,
            parameter_topology=topology,
            parameter_count=sum(
                parameter.numel() for parameter in model.parameters()
            ),
            cache_fingerprint=result.cache_fingerprint,
            schedule_fingerprint=result.schedule_fingerprint,
            optimizer_config_fingerprint=(
                result.optimizer_config_fingerprint
            ),
            initial_model_fingerprint=result.initial_model_fingerprint,
            final_model_fingerprint=result.final_model_fingerprint,
            training_result_fingerprint=result.result_fingerprint,
            completed_updates=result.completed_updates,
            forward_calls=result.forward_calls,
        )
        return PACREPMOPETrainingBundle(
            model_config=model_config,
            training_config=config,
            model=model,
            training_result=result,
            receipt=receipt,
        )
    except BaseException:
        if protected_authorization is not None:
            protected_authorization.mark_failed()
        raise


__all__ = [
    "PACRE_CONFIG_FQCN",
    "PACRE_MODEL_FQCN",
    "PACRE_OPTIMIZER_FQCN",
    "PACRE_PARAMETER_NAMES",
    "PACRE_PMOPE_OBJECTIVE",
    "PACRE_PMOPE_TRAINING_CONFIG",
    "PACRE_PMOPE_TRAINING_SCHEMA",
    "PACRE_PMOPE_TRAINING_SEED",
    "PACRE_SOURCE_PATHS",
    "PACREPMOPETrainingBundle",
    "PACREPMOPETrainingConfig",
    "PACREPMOPETrainingReceipt",
    "PACREParameterTopology",
    "train_pacre_pmope_candidate",
]
