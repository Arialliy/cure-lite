"""Real D_R execution contracts for the PFCR CURE-Lite decoder."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite

import torch
from torch import Tensor

from ..cache.schema import canonical_json, stable_fingerprint
from ..paired_types import tensor_content_fingerprint
from ..phase_resolved_real_cache import PFCRRealCacheAdapter
from ..phase_resolved_real_states import (
    PFCRRealStateCatalog,
    build_pfcr_epoch_pools,
)
from ..phase_resolved_relation_decoder import (
    CURELitePhaseResolvedRelationDecoder,
    PFCR_EVIDENCE_CEILING,
)
from ..phase_resolved_relation_training import (
    PFCR_TRAIN_RELATION_DIM,
    PhaseResolvedRelationTrainingConfig,
)
from ..train.phase_resolved_relation_step import (
    phase_resolved_real_train_step,
)
from ..train.pools import iter_fixed_branch_batches
from ..train.pools import fixed_branch_selection_indices


PFCR_REAL_PREFLIGHT_SCHEMA = (
    "cure-lite-pfcr-real-training-preflight-v1"
)
PFCR_REAL_FORMAL_TRAINING_SCHEMA = (
    "cure-lite-pfcr-real-formal-training-config-v1"
)
PFCR_REAL_FORMAL_SCHEDULE_SCHEMA = (
    "cure-lite-pfcr-real-formal-schedule-v1"
)
PFCR_REAL_FORMAL_EXECUTION_SCHEMA = (
    "cure-lite-pfcr-real-formal-execution-ledger-v1"
)
PFCR_REAL_FORMAL_EXPOSURE_SCHEMA = (
    "cure-lite-pfcr-real-formal-exposure-v1"
)


@dataclass(frozen=True)
class PFCRRealPreflightConfig:
    """Frozen engineering-only D_R execution check."""

    seed: int
    update_count: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    factual_miss_batch: int = 4
    factual_no_miss_batch: int = 4
    synthetic_batch: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 0xFFFFFFFF
        ):
            raise ValueError("seed must be a uint32")
        frozen = {
            "update_count": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "factual_miss_batch": 4,
            "factual_no_miss_batch": 4,
            "synthetic_batch": 4,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"real preflight fixes {name}")

    @property
    def branch_batch_sizes(self) -> dict[str, int]:
        return {
            "factual_miss": self.factual_miss_batch,
            "factual_no_miss": self.factual_no_miss_batch,
            "synthetic": self.synthetic_batch,
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "update_count": self.update_count,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": "adam",
            "branch_batch_sizes": self.branch_batch_sizes,
            "relation_dim": PFCR_TRAIN_RELATION_DIM,
            "fp32": True,
            "reads_only": ["D_R"],
            "D_V_read": False,
            "performance_claim_authorized": False,
        }


@dataclass(frozen=True)
class PFCRRealFormalTrainingConfig:
    """Production CURE-Lite schedule; deliberately has no continuation mode."""

    seed: int
    epochs: int = 800
    steps_per_epoch: int = 40
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    factual_miss_batch: int = 4
    factual_no_miss_batch: int = 4
    synthetic_batch: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed not in {42, 43}
        ):
            raise ValueError("formal PFCR seed must be 42 or 43")
        frozen = {
            "epochs": 800,
            "steps_per_epoch": 40,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "factual_miss_batch": 4,
            "factual_no_miss_batch": 4,
            "synthetic_batch": 4,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"formal PFCR training fixes {name}")

    @property
    def optimizer_updates(self) -> int:
        return self.epochs * self.steps_per_epoch

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PFCR_REAL_FORMAL_TRAINING_SCHEMA,
            "seed": self.seed,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "optimizer_updates": self.optimizer_updates,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": "adam",
            "branch_batch_sizes": {
                "factual_miss": self.factual_miss_batch,
                "factual_no_miss": self.factual_no_miss_batch,
                "synthetic": self.synthetic_batch,
            },
            "relation_dim": PFCR_TRAIN_RELATION_DIM,
            "fp32": True,
            "continuation_supported": False,
            "intermediate_optimizer_state_saved": False,
            "incomplete_attempt_may_be_evaluated": False,
        }


def pfcr_model_state_fingerprint(
    model: CURELitePhaseResolvedRelationDecoder,
) -> str:
    """Fingerprint the exact finite floating-point PFCR state."""

    if not isinstance(
        model,
        CURELitePhaseResolvedRelationDecoder,
    ):
        raise TypeError("model must be the PFCR decoder")
    state = model.state_dict()
    if not state:
        raise ValueError("PFCR decoder state is empty")
    for name, value in state.items():
        if (
            not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                f"PFCR decoder tensor {name!r} must be finite floating point"
            )
    return stable_fingerprint(
        {
            name: {
                "shape": list(value.shape),
                "content": tensor_content_fingerprint(
                    value.detach().cpu().reshape(-1)
                ),
            }
            for name, value in sorted(model.state_dict().items())
        }
    )


_model_state_fingerprint = pfcr_model_state_fingerprint


def pfcr_real_formal_schedule_payload(
    catalog: PFCRRealStateCatalog,
    config: PFCRRealFormalTrainingConfig,
) -> dict[str, object]:
    """Bind the deterministic 800 x 40 sampling law to one real population."""

    if not isinstance(catalog, PFCRRealStateCatalog):
        raise TypeError("catalog must be PFCRRealStateCatalog")
    if not isinstance(config, PFCRRealFormalTrainingConfig):
        raise TypeError("config must be PFCRRealFormalTrainingConfig")
    exposure = pfcr_real_formal_exposure_payload(catalog, config)
    return {
        "schema_version": PFCR_REAL_FORMAL_SCHEDULE_SCHEMA,
        "state_catalog_fingerprint": catalog.catalog_fingerprint,
        "lineage_allowlist_fingerprint": (
            catalog.allowlist.allowlist_fingerprint
        ),
        "seed": config.seed,
        "epochs": config.epochs,
        "steps_per_epoch": config.steps_per_epoch,
        "optimizer_updates": config.optimizer_updates,
        "branch_batch_sizes": {
            "factual_miss": config.factual_miss_batch,
            "factual_no_miss": config.factual_no_miss_batch,
            "synthetic": config.synthetic_batch,
        },
        "epoch_population_policy": (
            "one_uniform_target_per_eligible_source_per_epoch_v1"
        ),
        "step_sampling_policy": (
            "stable_hash_with_replacement_per_branch_epoch_step_draw_v1"
        ),
        "all_three_branches_active_per_update": True,
        "decoder_forwards_per_update": 1,
        "decoder_states_per_update": 12,
        "exposure": exposure,
    }


def _example_identity_lookup(
    catalog: PFCRRealStateCatalog,
) -> tuple[
    dict[int, tuple[str, int | None, int | None]],
    dict[str, tuple[tuple[str, int | None, int | None], ...]],
]:
    lookup: dict[int, tuple[str, int | None, int | None]] = {}
    population: dict[
        str,
        list[tuple[str, int | None, int | None]],
    ] = {
        "factual_miss": [],
        "factual_no_miss": [],
        "synthetic": [],
    }
    for entry, selected_indices in zip(
        catalog.prepared.entries,
        catalog.selected_legal_indices,
        strict=True,
    ):
        for gt_id, example in zip(
            entry.reachable_gt_ids,
            entry.factual_examples,
            strict=True,
        ):
            identity = (entry.sample_id, gt_id, None)
            lookup[id(example)] = identity
            population["factual_miss"].append(identity)
        if entry.factual_no_miss_example is not None:
            identity = (entry.sample_id, None, None)
            lookup[id(entry.factual_no_miss_example)] = identity
            population["factual_no_miss"].append(identity)
        for index in selected_indices:
            candidate = entry.decoder_visible_legal_candidates[index]
            example = entry.synthetic_examples[index]
            identity = (
                entry.sample_id,
                candidate.gt_id,
                candidate.pred_id,
            )
            lookup[id(example)] = identity
            population["synthetic"].append(identity)
    expected = (
        catalog.factual_target_count
        + catalog.factual_no_miss_source_count
        + catalog.legal_target_count
    )
    if len(lookup) != expected:
        raise RuntimeError("PFCR state identity lookup is not one-to-one")
    return lookup, {
        branch: tuple(values)
        for branch, values in population.items()
    }


def _identity_rows(
    counts: Counter[tuple[str, int | None, int | None]],
    population: Sequence[tuple[str, int | None, int | None]],
) -> list[dict[str, object]]:
    return [
        {
            "identity": list(identity),
            "count": counts[identity],
        }
        for identity in sorted(
            population,
            key=lambda value: (
                value[0],
                -1 if value[1] is None else value[1],
                -1 if value[2] is None else value[2],
            ),
        )
    ]


def pfcr_real_formal_exposure_payload(
    catalog: PFCRRealStateCatalog,
    config: PFCRRealFormalTrainingConfig,
) -> dict[str, object]:
    """Replay all 32,000 deterministic draws without materializing tensors."""

    if not isinstance(catalog, PFCRRealStateCatalog):
        raise TypeError("catalog must be PFCRRealStateCatalog")
    if not isinstance(config, PFCRRealFormalTrainingConfig):
        raise TypeError("config must be PFCRRealFormalTrainingConfig")
    lookup, population = _example_identity_lookup(catalog)
    batch_sizes = {
        "factual_miss": config.factual_miss_batch,
        "factual_no_miss": config.factual_no_miss_batch,
        "synthetic": config.synthetic_batch,
    }
    hashers = {
        branch: sha256()
        for branch in batch_sizes
    }
    combined_hasher = sha256()
    state_counts = {
        branch: Counter()
        for branch in batch_sizes
    }
    source_counts = {
        branch: Counter()
        for branch in batch_sizes
    }
    for epoch in range(config.epochs):
        pools = build_pfcr_epoch_pools(
            catalog,
            epoch=epoch,
            global_seed=config.seed,
        )
        for branch, batch_size in batch_sizes.items():
            pool = pools.get(branch)
            identities = tuple(
                lookup[id(example)] for example in pool
            )
            for step in range(config.steps_per_epoch):
                indices = fixed_branch_selection_indices(
                    len(identities),
                    batch_size,
                    branch=branch,
                    epoch=epoch,
                    step=step,
                    global_seed=config.seed,
                )
                selected = tuple(identities[index] for index in indices)
                row = {
                    "epoch": epoch,
                    "step": step,
                    "branch": branch,
                    "identities": [
                        list(identity) for identity in selected
                    ],
                }
                encoded = canonical_json(row).encode("utf-8") + b"\n"
                hashers[branch].update(encoded)
                combined_hasher.update(encoded)
                state_counts[branch].update(selected)
                source_counts[branch].update(
                    identity[0] for identity in selected
                )
    branches: dict[str, object] = {}
    for branch in batch_sizes:
        expected_per_branch = (
            config.optimizer_updates * batch_sizes[branch]
        )
        target_rows = _identity_rows(
            state_counts[branch],
            population[branch],
        )
        source_rows = [
            {
                "sample_id": sample_id,
                "count": count,
            }
            for sample_id, count in sorted(
                source_counts[branch].items()
            )
        ]
        total = sum(row["count"] for row in target_rows)
        if total != expected_per_branch:
            raise RuntimeError("PFCR formal exposure total changed")
        branches[branch] = {
            "sequence_fingerprint": hashers[branch].hexdigest(),
            "state_population": len(population[branch]),
            "source_population": len(
                {identity[0] for identity in population[branch]}
            ),
            "state_exposure_total": total,
            "source_exposure_total": sum(
                row["count"] for row in source_rows
            ),
            "zero_exposure_states": sum(
                row["count"] == 0 for row in target_rows
            ),
            "state_exposures": target_rows,
            "source_exposures": source_rows,
        }
    payload = {
        "schema_version": PFCR_REAL_FORMAL_EXPOSURE_SCHEMA,
        "seed": config.seed,
        "state_catalog_fingerprint": catalog.catalog_fingerprint,
        "epochs": config.epochs,
        "steps_per_epoch": config.steps_per_epoch,
        "optimizer_updates": config.optimizer_updates,
        "combined_sequence_fingerprint": (
            combined_hasher.hexdigest()
        ),
        "branches": branches,
    }
    return payload


@dataclass(frozen=True)
class PFCRRealFormalExecutionLedger:
    """Exact completed-compute proof for one PFCR 800 x 40 run."""

    seed: int
    cache_contract_fingerprint: str
    state_catalog_fingerprint: str
    lineage_allowlist_fingerprint: str
    formal_schedule_fingerprint: str
    initial_model_fingerprint: str
    final_model_fingerprint: str
    optimizer_state_fingerprint: str
    trainable_parameter_count: int
    minimum_gradient_l2_norm: float
    maximum_gradient_l2_norm: float
    minimum_adam_step: int = 32_000
    maximum_adam_step: int = 32_000
    optimizer_updates: int = 32_000
    completed_epochs: int = 800
    steps_per_epoch: int = 40
    decoder_forward_calls: int = 32_000
    decoder_state_evaluations: int = 384_000
    backward_calls: int = 32_000
    optimizer_steps: int = 32_000
    factual_miss_state_evaluations: int = 128_000
    factual_no_miss_state_evaluations: int = 128_000
    synthetic_state_evaluations: int = 128_000
    all_trace_values_finite: bool = True
    all_optimizer_moments_finite: bool = True
    parameters_changed: bool = True
    cache_unchanged_after_execution: bool = True
    schema_version: str = PFCR_REAL_FORMAL_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PFCR_REAL_FORMAL_EXECUTION_SCHEMA:
            raise ValueError("unsupported PFCR formal execution schema")
        if self.seed not in {42, 43}:
            raise ValueError("PFCR formal execution seed must be 42 or 43")
        for name in (
            "cache_contract_fingerprint",
            "state_catalog_fingerprint",
            "lineage_allowlist_fingerprint",
            "formal_schedule_fingerprint",
            "initial_model_fingerprint",
            "final_model_fingerprint",
            "optimizer_state_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest")
        exact = {
            "optimizer_updates": 32_000,
            "completed_epochs": 800,
            "steps_per_epoch": 40,
            "decoder_forward_calls": 32_000,
            "decoder_state_evaluations": 384_000,
            "backward_calls": 32_000,
            "optimizer_steps": 32_000,
            "minimum_adam_step": 32_000,
            "maximum_adam_step": 32_000,
            "factual_miss_state_evaluations": 128_000,
            "factual_no_miss_state_evaluations": 128_000,
            "synthetic_state_evaluations": 128_000,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if isinstance(value, bool) or value != expected:
                raise ValueError(f"{name} must equal {expected}")
        if (
            isinstance(self.trainable_parameter_count, bool)
            or not isinstance(self.trainable_parameter_count, int)
            or self.trainable_parameter_count < 1
        ):
            raise ValueError("trainable_parameter_count must be positive")
        for name in (
            "minimum_gradient_l2_norm",
            "maximum_gradient_l2_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_gradient_l2_norm < self.minimum_gradient_l2_norm:
            raise ValueError("PFCR gradient norm bounds are inconsistent")
        if self.all_trace_values_finite is not True:
            raise ValueError("PFCR formal execution requires finite trace values")
        if self.all_optimizer_moments_finite is not True:
            raise ValueError(
                "PFCR formal execution requires finite Adam moments"
            )
        if self.parameters_changed is not True:
            raise ValueError("PFCR formal execution requires changed parameters")
        if self.cache_unchanged_after_execution is not True:
            raise ValueError("PFCR formal execution requires an unchanged cache")

    def canonical_payload(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "PFCRRealFormalExecutionLedger":
        if not isinstance(value, Mapping):
            raise TypeError("PFCR formal execution ledger must be a mapping")
        fields = tuple(cls.__dataclass_fields__)
        if set(value) != set(fields):
            raise ValueError("PFCR formal execution ledger fields are not canonical")
        result = cls(**{field: value[field] for field in fields})
        if result.canonical_payload() != dict(value):
            raise ValueError("PFCR formal execution ledger is not canonical")
        return result


@dataclass(frozen=True)
class PFCRRealFormalTrainingResult:
    """A finished in-memory model and its complete 800-epoch evidence."""

    decoder: CURELitePhaseResolvedRelationDecoder
    epoch_logs: tuple[Mapping[str, object], ...]
    execution_ledger: PFCRRealFormalExecutionLedger

    def __post_init__(self) -> None:
        if not isinstance(
            self.decoder,
            CURELitePhaseResolvedRelationDecoder,
        ):
            raise TypeError("decoder must be the PFCR decoder")
        if len(self.epoch_logs) != 800:
            raise ValueError("PFCR formal result requires 800 epoch logs")
        if not isinstance(
            self.execution_ledger,
            PFCRRealFormalExecutionLedger,
        ):
            raise TypeError("execution_ledger has the wrong type")
        if (
            pfcr_model_state_fingerprint(self.decoder)
            != self.execution_ledger.final_model_fingerprint
        ):
            raise ValueError("PFCR final decoder and execution ledger differ")


class _PFCRForwardLedger:
    """Count calls through PFCR's unique shared projection."""

    def __init__(
        self,
        decoder: CURELitePhaseResolvedRelationDecoder,
    ) -> None:
        self.calls = 0
        self.states = 0
        self._handle = decoder.relation.projection.register_forward_hook(
            self._hook
        )

    def _hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise RuntimeError(
                "PFCR formal shared projection returned an invalid output"
            )
        self.calls += 1
        self.states += int(output.shape[0])

    def close(self) -> None:
        self._handle.remove()


def _mean(rows: Sequence[Mapping[str, float | int]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values or any(not isfinite(value) for value in values):
        raise FloatingPointError(f"PFCR epoch metric {key} is not finite")
    return fsum(values) / len(values)


def _validate_final_adam(
    decoder: CURELitePhaseResolvedRelationDecoder,
    optimizer: torch.optim.Optimizer,
    *,
    expected_steps: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[str, int, int]:
    """Validate every final Adam state without serializing optimizer state."""

    if type(optimizer) is not torch.optim.Adam:
        raise TypeError("PFCR formal optimizer must be torch.optim.Adam")
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("PFCR formal Adam must have one parameter group")
    group = optimizer.param_groups[0]
    expected_options = {
        "lr": learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": weight_decay,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }
    if any(
        group.get(name) != value
        for name, value in expected_options.items()
    ):
        raise RuntimeError("PFCR formal Adam options changed")
    named_parameters = tuple(decoder.named_parameters())
    if (
        tuple(group["params"])
        != tuple(parameter for _, parameter in named_parameters)
    ):
        raise RuntimeError("PFCR formal Adam parameter order changed")
    rows: list[dict[str, object]] = []
    steps: list[int] = []
    for name, parameter in named_parameters:
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict) or set(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RuntimeError(
                f"PFCR final Adam state for {name!r} is incomplete"
            )
        raw_step = state["step"]
        if not isinstance(raw_step, Tensor) or raw_step.numel() != 1:
            raise RuntimeError("PFCR Adam step must be a scalar tensor")
        step = int(raw_step.detach().cpu().item())
        if step != expected_steps:
            raise RuntimeError(
                f"PFCR Adam step for {name!r} must equal {expected_steps}"
            )
        moments: dict[str, object] = {}
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state[moment_name]
            if (
                not isinstance(moment, Tensor)
                or moment.shape != parameter.shape
                or moment.dtype != parameter.dtype
                or moment.device != parameter.device
                or not bool(torch.isfinite(moment).all())
            ):
                raise RuntimeError(
                    f"PFCR Adam moment {moment_name!r} for {name!r} "
                    "violates its parameter contract"
                )
            cpu = moment.detach().cpu().reshape(-1)
            moments[moment_name] = {
                "shape": list(moment.shape),
                "dtype": str(moment.dtype),
                "content_fingerprint": tensor_content_fingerprint(cpu),
            }
        steps.append(step)
        rows.append(
            {
                "parameter": name,
                "step": step,
                "moments": moments,
            }
        )
    if not rows:
        raise RuntimeError("PFCR formal Adam has no parameter state")
    payload = {
        "optimizer": "adam",
        "expected_steps": expected_steps,
        "parameters": rows,
    }
    return stable_fingerprint(payload), min(steps), max(steps)


def execute_pfcr_real_formal_training(
    cache: PFCRRealCacheAdapter,
    catalog: PFCRRealStateCatalog,
    config: PFCRRealFormalTrainingConfig,
    *,
    device: torch.device | str,
    epoch_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> PFCRRealFormalTrainingResult:
    """Run exactly one fresh, no-resume, D_R-only PFCR formal attempt."""

    if not isinstance(cache, PFCRRealCacheAdapter):
        raise TypeError("cache must be PFCRRealCacheAdapter")
    if not isinstance(catalog, PFCRRealStateCatalog):
        raise TypeError("catalog must be PFCRRealStateCatalog")
    if not isinstance(config, PFCRRealFormalTrainingConfig):
        raise TypeError("config must be PFCRRealFormalTrainingConfig")
    if (
        catalog.prepared is not cache.prepared_catalog
        or catalog.cache_contract_fingerprint
        != cache.contract.contract_fingerprint
    ):
        raise RuntimeError("PFCR real catalog is bound to another cache")
    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError("epoch_callback must be callable or None")
    resolved_device = torch.device(device)
    if (
        resolved_device.type == "cuda"
        and (
            not torch.cuda.is_available()
            or resolved_device.index is None
            or resolved_device.index >= torch.cuda.device_count()
        )
    ):
        raise RuntimeError("requested explicit CUDA device is unavailable")
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("PFCR formal device must be CPU or explicit CUDA")

    cache.verify_unchanged()
    catalog.verify_unchanged()
    schedule_payload = pfcr_real_formal_schedule_payload(catalog, config)
    schedule_fingerprint = stable_fingerprint(schedule_payload)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    epoch_logs: list[dict[str, object]] = []
    minimum_gradient = float("inf")
    maximum_gradient = 0.0
    trace_is_finite = True
    forward_ledger: _PFCRForwardLedger | None = None
    try:
        torch.use_deterministic_algorithms(True)
        rng_devices = (
            [resolved_device]
            if resolved_device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(config.seed)
            if resolved_device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)
            decoder = CURELitePhaseResolvedRelationDecoder(
                cache.contract.decoder_config(
                    relation_dim=PFCR_TRAIN_RELATION_DIM
                )
            )
            initial_fingerprint = pfcr_model_state_fingerprint(decoder)
            decoder = decoder.to(
                device=resolved_device,
                dtype=torch.float32,
            )
            optimizer = torch.optim.Adam(
                decoder.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            if optimizer.state:
                raise RuntimeError(
                    "fresh PFCR formal optimizer unexpectedly has state"
                )
            parameter_count = sum(
                parameter.numel()
                for parameter in decoder.parameters()
                if parameter.requires_grad
            )
            logit_margin = PhaseResolvedRelationTrainingConfig(
                seed=config.seed
            ).logit_margin
            forward_ledger = _PFCRForwardLedger(decoder)
            for epoch in range(config.epochs):
                pools = build_pfcr_epoch_pools(
                    catalog,
                    epoch=epoch,
                    global_seed=config.seed,
                )
                batch_iterator = iter_fixed_branch_batches(
                    pools,
                    {
                        "factual_miss": config.factual_miss_batch,
                        "factual_no_miss": (
                            config.factual_no_miss_batch
                        ),
                        "synthetic": config.synthetic_batch,
                    },
                    epoch=epoch,
                    global_seed=config.seed,
                    device=resolved_device,
                    steps=config.steps_per_epoch,
                )
                step_logs: list[dict[str, float | int]] = []
                calls_before = forward_ledger.calls
                states_before = forward_ledger.states
                for batches in batch_iterator:
                    row = phase_resolved_real_train_step(
                        decoder,
                        optimizer,
                        batches,
                        logit_margin=logit_margin,
                        audit=False,
                    )
                    if any(
                        not isfinite(float(value))
                        for value in row.values()
                    ):
                        trace_is_finite = False
                        raise FloatingPointError(
                            "PFCR formal trace contains a non-finite value"
                        )
                    gradient = float(row["gradient_l2_norm"])
                    minimum_gradient = min(minimum_gradient, gradient)
                    maximum_gradient = max(maximum_gradient, gradient)
                    step_logs.append(row)
                if len(step_logs) != config.steps_per_epoch:
                    raise RuntimeError(
                        "PFCR formal epoch did not complete 40 updates"
                    )
                if any(
                    not bool(torch.isfinite(parameter).all())
                    for parameter in decoder.parameters()
                ):
                    raise FloatingPointError(
                        "PFCR formal parameter is non-finite at epoch end"
                    )
                epoch_row: dict[str, object] = {
                    "epoch": epoch,
                    "steps": len(step_logs),
                    "optimizer_updates_completed": (
                        (epoch + 1) * config.steps_per_epoch
                    ),
                    "decoder_forward_calls": (
                        forward_ledger.calls - calls_before
                    ),
                    "decoder_state_evaluations": (
                        forward_ledger.states - states_before
                    ),
                    "metrics": {
                        "mean_total_loss": _mean(step_logs, "total"),
                        "mean_factual_miss_loss": _mean(
                            step_logs,
                            "factual_miss/loss",
                        ),
                        "mean_factual_no_miss_loss": _mean(
                            step_logs,
                            "factual_no_miss/loss",
                        ),
                        "mean_synthetic_loss": _mean(
                            step_logs,
                            "synthetic/loss",
                        ),
                        "minimum_total_loss": min(
                            float(row["total"]) for row in step_logs
                        ),
                        "maximum_total_loss": max(
                            float(row["total"]) for row in step_logs
                        ),
                        "minimum_gradient_l2_norm": min(
                            float(row["gradient_l2_norm"])
                            for row in step_logs
                        ),
                        "maximum_gradient_l2_norm": max(
                            float(row["gradient_l2_norm"])
                            for row in step_logs
                        ),
                    },
                }
                epoch_logs.append(epoch_row)
                if epoch_callback is not None:
                    epoch_callback(deepcopy(epoch_row))
            (
                optimizer_state_fingerprint,
                minimum_adam_step,
                maximum_adam_step,
            ) = _validate_final_adam(
                decoder,
                optimizer,
                expected_steps=config.optimizer_updates,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            final_fingerprint = pfcr_model_state_fingerprint(decoder)
            actual_calls = forward_ledger.calls
            actual_states = forward_ledger.states
    finally:
        if forward_ledger is not None:
            forward_ledger.close()
        torch.use_deterministic_algorithms(previous_deterministic)

    cache.verify_unchanged()
    catalog.verify_unchanged()
    expected_calls = config.optimizer_updates
    expected_states = expected_calls * sum(
        (
            config.factual_miss_batch,
            config.factual_no_miss_batch,
            config.synthetic_batch,
        )
    )
    if (
        len(epoch_logs) != config.epochs
        or actual_calls != expected_calls
        or actual_states != expected_states
        or minimum_gradient == float("inf")
        or initial_fingerprint == final_fingerprint
    ):
        raise RuntimeError("PFCR formal execution accounting did not close")
    ledger = PFCRRealFormalExecutionLedger(
        seed=config.seed,
        cache_contract_fingerprint=(
            cache.contract.contract_fingerprint
        ),
        state_catalog_fingerprint=catalog.catalog_fingerprint,
        lineage_allowlist_fingerprint=(
            catalog.allowlist.allowlist_fingerprint
        ),
        formal_schedule_fingerprint=schedule_fingerprint,
        initial_model_fingerprint=initial_fingerprint,
        final_model_fingerprint=final_fingerprint,
        optimizer_state_fingerprint=optimizer_state_fingerprint,
        trainable_parameter_count=parameter_count,
        minimum_gradient_l2_norm=minimum_gradient,
        maximum_gradient_l2_norm=maximum_gradient,
        minimum_adam_step=minimum_adam_step,
        maximum_adam_step=maximum_adam_step,
        all_trace_values_finite=trace_is_finite,
    )
    return PFCRRealFormalTrainingResult(
        decoder=decoder,
        epoch_logs=tuple(epoch_logs),
        execution_ledger=ledger,
    )


def _batch_metrics(
    model: CURELitePhaseResolvedRelationDecoder,
    batches,
) -> dict[str, float | int | bool]:
    order = ("factual_miss", "factual_no_miss", "synthetic")
    feature = torch.cat(
        [batches[name].feature.detach() for name in order],
        dim=0,
    )
    occupancy = torch.cat(
        [batches[name].occupancy for name in order],
        dim=0,
    )
    target = torch.cat(
        [batches[name].target for name in order],
        dim=0,
    ).to(torch.bool)
    valid = torch.cat(
        [batches[name].valid_mask for name in order],
        dim=0,
    )
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            fields = model.forward_fields(feature, occupancy)
    finally:
        model.train(was_training)
    logits = fields.logits
    probability = fields.completion_probability
    writable = valid & ~occupancy
    positive = target & writable
    negative = ~target & writable
    return {
        "state_count": int(logits.shape[0]),
        "all_fields_finite": bool(
            torch.isfinite(logits).all()
            and torch.isfinite(fields.phase_evidence).all()
            and torch.isfinite(
                fields.relation.coverage_burden
            ).all()
        ),
        "logit_min": float(logits.min().item()),
        "logit_max": float(logits.max().item()),
        "positive_probability_min": float(
            probability[positive].min().item()
        ),
        "negative_probability_max": float(
            probability[negative].max().item()
        ),
        "phase_evidence_min": float(
            fields.phase_evidence.min().item()
        ),
        "phase_evidence_max": float(
            fields.phase_evidence.max().item()
        ),
    }


def run_pfcr_real_preflight(
    cache: PFCRRealCacheAdapter,
    catalog: PFCRRealStateCatalog,
    config: PFCRRealPreflightConfig,
    *,
    device: torch.device | str,
) -> dict[str, object]:
    """Execute ten D_R-only updates and return an immutable evidence payload."""

    if not isinstance(cache, PFCRRealCacheAdapter):
        raise TypeError("cache must be PFCRRealCacheAdapter")
    if not isinstance(catalog, PFCRRealStateCatalog):
        raise TypeError("catalog must be PFCRRealStateCatalog")
    if not isinstance(config, PFCRRealPreflightConfig):
        raise TypeError("config must be PFCRRealPreflightConfig")
    if (
        catalog.prepared is not cache.prepared_catalog
        or catalog.cache_contract_fingerprint
        != cache.contract.contract_fingerprint
    ):
        raise RuntimeError("PFCR real catalog is bound to another cache")
    resolved_device = torch.device(device)
    if (
        resolved_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("requested CUDA device is unavailable")

    cache.verify_unchanged()
    catalog.verify_unchanged()
    previous_deterministic = (
        torch.are_deterministic_algorithms_enabled()
    )
    trace: list[dict[str, float | int]] = []
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(
            devices=(
                [resolved_device]
                if resolved_device.type == "cuda"
                else []
            )
        ):
            torch.manual_seed(config.seed)
            if resolved_device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)
            decoder = CURELitePhaseResolvedRelationDecoder(
                cache.contract.decoder_config(
                    relation_dim=PFCR_TRAIN_RELATION_DIM
                )
            ).to(device=resolved_device, dtype=torch.float32)
            optimizer = torch.optim.Adam(
                decoder.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            initial_model_fingerprint = _model_state_fingerprint(
                decoder
            )
            pools = build_pfcr_epoch_pools(
                catalog,
                epoch=0,
                global_seed=config.seed,
            )
            batch_iterator = iter_fixed_branch_batches(
                pools,
                config.branch_batch_sizes,
                epoch=0,
                global_seed=config.seed,
                device=resolved_device,
                steps=config.update_count,
            )
            first_batches = None
            for update_index, batches in enumerate(batch_iterator):
                if first_batches is None:
                    first_batches = batches
                    initial_metrics = _batch_metrics(
                        decoder,
                        first_batches,
                    )
                logs = phase_resolved_real_train_step(
                    decoder,
                    optimizer,
                    batches,
                    logit_margin=(
                        PhaseResolvedRelationTrainingConfig(
                            seed=config.seed
                        ).logit_margin
                    ),
                )
                row: dict[str, float | int] = {
                    "update_index": update_index,
                }
                row.update(logs)
                trace.append(row)
            if first_batches is None:
                raise AssertionError("PFCR preflight produced no batch")
            final_metrics = _batch_metrics(decoder, first_batches)
            final_model_fingerprint = _model_state_fingerprint(
                decoder
            )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    cache.verify_unchanged()
    catalog.verify_unchanged()
    finite_trace = all(
        all(
            isfinite(float(value))
            for key, value in row.items()
            if key != "update_index"
        )
        for row in trace
    )
    one_forward_per_update = all(
        row["decoder_forward_calls"] == 1 for row in trace
    )
    gates = {
        "all_updates_completed": len(trace) == config.update_count,
        "all_trace_values_finite": finite_trace,
        "one_decoder_forward_per_update": one_forward_per_update,
        "decoder_parameters_changed": (
            initial_model_fingerprint != final_model_fingerprint
        ),
        "initial_fields_finite": initial_metrics["all_fields_finite"],
        "final_fields_finite": final_metrics["all_fields_finite"],
        "initial_evidence_bounded": (
            0.0 <= initial_metrics["phase_evidence_min"]
            and initial_metrics["phase_evidence_max"]
            <= PFCR_EVIDENCE_CEILING
        ),
        "final_evidence_bounded": (
            0.0 <= final_metrics["phase_evidence_min"]
            and final_metrics["phase_evidence_max"]
            <= PFCR_EVIDENCE_CEILING
        ),
        "cache_unchanged_after_execution": True,
    }
    passed = all(gates.values())
    payload: dict[str, object] = {
        "schema_version": PFCR_REAL_PREFLIGHT_SCHEMA,
        "scope": {
            "model": "CURE-Lite",
            "split_read": ["D_R"],
            "D_V_read": False,
            "D_T_read": False,
            "full_CURE_in_scope": False,
            "performance_evaluation": False,
        },
        "config": config.canonical_payload(),
        "device": {
            "type": resolved_device.type,
            "index": resolved_device.index,
            "name": (
                torch.cuda.get_device_name(resolved_device)
                if resolved_device.type == "cuda"
                else "cpu"
            ),
        },
        "cache_contract_fingerprint": (
            cache.contract.contract_fingerprint
        ),
        "state_catalog_fingerprint": catalog.catalog_fingerprint,
        "lineage_allowlist_fingerprint": (
            catalog.allowlist.allowlist_fingerprint
        ),
        "population": {
            "D_R_samples": cache.contract.sample_count,
            "factual_targets": catalog.factual_target_count,
            "factual_sources": catalog.factual_source_count,
            "factual_no_miss_sources": (
                catalog.factual_no_miss_source_count
            ),
            "lineage_safe_legal_targets": (
                catalog.legal_target_count
            ),
            "lineage_safe_legal_sources": (
                catalog.legal_source_count
            ),
            "excluded_legal_identities": [
                list(value)
                for value in (
                    catalog.allowlist.excluded_legal_identities
                )
            ],
        },
        "decoder": {
            "feature_channels": cache.contract.feature_channels,
            "feature_stride": cache.contract.feature_stride,
            "relation_dim": PFCR_TRAIN_RELATION_DIM,
            "evidence_ceiling": PFCR_EVIDENCE_CEILING,
            "initial_model_fingerprint": initial_model_fingerprint,
            "final_model_fingerprint": final_model_fingerprint,
        },
        "initial_fixed_batch_metrics": initial_metrics,
        "trace": trace,
        "final_fixed_batch_metrics": final_metrics,
        "gates": gates,
        "decision": {
            "real_training_execution_preflight_pass": passed,
            "formal_800_epoch_training_implementation_authorized": (
                passed
            ),
            "real_dataset_model_success_claimed": False,
            "D_V_evaluation_authorized_by_this_receipt": False,
            "full_CURE_authorized": False,
        },
    }
    payload["result_fingerprint"] = stable_fingerprint(payload)
    return payload


__all__ = [
    "PFCR_REAL_FORMAL_EXECUTION_SCHEMA",
    "PFCR_REAL_FORMAL_SCHEDULE_SCHEMA",
    "PFCR_REAL_FORMAL_TRAINING_SCHEMA",
    "PFCR_REAL_PREFLIGHT_SCHEMA",
    "PFCRRealFormalExecutionLedger",
    "PFCRRealFormalTrainingConfig",
    "PFCRRealFormalTrainingResult",
    "PFCRRealPreflightConfig",
    "execute_pfcr_real_formal_training",
    "pfcr_model_state_fingerprint",
    "pfcr_real_formal_schedule_payload",
    "run_pfcr_real_preflight",
]
