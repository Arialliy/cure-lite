"""Paired, one-shot, D_R-only bounded-400 runner for GCR-PACRE v24.

This module implements an optimization/execution smoke test, not a
generalization gate.  PACRE-VC v23 and GCR-PACRE v24 are trained from
byte-identical seed-42 parameters with independent modules, caches, packed
device storage, and Adam optimizers.  They consume the same frozen schedule
in lockstep for exactly ``10 * 40`` updates.

Relative candidate/control and candidate/same-weight-G1 quantities are
terminal diagnostics only.  There is intentionally no fixed relative uplift
threshold in this runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
import json
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Final, Mapping

import torch

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.coverage_state_device_cache import (
    CoverageStateDeviceCache,
    prepare_coverage_state_device_cache,
)
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateTrainingSchedule,
    coverage_state_schedule_exposure_report,
)
from cure_lite.coverage_state_sobolev import CSLF_PMOPE_POLICY
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    CoverageStateRunAuthorization,
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
    audit_coverage_state_training_state,
    coverage_state_fused_train_step,
    coverage_state_pair_objective_policy,
)
from cure_lite_v23.factory import build_pacre_vc_training_model
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)
from .training import gcr_pacre_training_state_summary_fingerprint
from tools.gcr_pacre_v24_protocol import (
    VerifiedAccessAudit,
    VerifiedOOFDecision,
    require_verified_access_audit,
    require_verified_oof_decision,
)

from .artifact_io import (
    atomic_write_new_canonical_json,
    read_canonical_json,
    save_terminal_safetensors_new,
)
from .bounded_run_start import (
    GCRPACREBoundedRunStartToken,
    VerifiedGCRPACREBoundedChainConfig,
    require_verified_gcr_pacre_bounded_chain_config,
    verify_gcr_pacre_bounded_chain_authorization_binding,
    verify_gcr_pacre_bounded_run_start_token,
)
from .factory import build_gcr_pacre_training_model
from .formal_cache_artifacts import (
    VerifiedFormalCacheArtifact,
    require_verified_formal_cache_artifact,
    verify_formal_cache_artifact,
)
from .gcr_pacre import (
    CURELiteGatedCommonResidualPACRELevelSet,
    CoverageStateGCRPACREConfig,
)
from .source_closure import gcr_pacre_v24_source_closure_hashes
from .training_trace import (
    build_training_trace_payload,
    save_training_trace_new,
    trace_finite_audit,
)


GCR_PACRE_BOUNDED_AUTHORIZATION_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-paired-bounded400-authorization-v2"
)
GCR_PACRE_BOUNDED_RESULT_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-paired-bounded400-result-v2"
)
GCR_PACRE_BOUNDED_RECEIPT_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-paired-bounded400-receipt-v6"
)
GCR_PACRE_BOUNDED_SEED: Final = 42
GCR_PACRE_BOUNDED_EPOCHS: Final = 10
GCR_PACRE_BOUNDED_STEPS_PER_EPOCH: Final = 40
GCR_PACRE_BOUNDED_UPDATES: Final = 400
GCR_PACRE_CONTROL_ARM: Final = "PACRE_VC_v23_control"
GCR_PACRE_CANDIDATE_ARM: Final = "GCR_PACRE_v24"
GCR_PACRE_NATIVE_MODE: Final = "native"
GCR_PACRE_FORCED_G1_MODE: Final = "forced_G1_same_weights"
GCR_PACRE_BOUNDED_EVALUATOR_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-bounded-evaluator-v1"
)
_PARAMETER_NAMES: Final = (
    "joint_state_weight",
    "joint_hidden_bias",
    "scalar_energy_weight",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_gate_role_distribution(
    value: Mapping[str, object],
) -> None:
    expected = {
        "schema_version",
        "endpoint_counts",
        "target_G",
        "background_G",
        "target_E",
        "background_E",
    }
    if (
        set(value) != expected
        or value.get("schema_version")
        != "cure-lite-v24-gcr-pacre-gate-role-summary-v1"
    ):
        raise ValueError("gate role distribution schema changed")
    endpoints = value.get("endpoint_counts")
    if not isinstance(endpoints, Mapping) or set(endpoints) != {
        "G_equal_0",
        "G_equal_2",
        "G_strict_interior",
    }:
        raise ValueError("gate endpoint distribution fields changed")
    endpoint_total = sum(
        _count(item, name=f"endpoint_counts.{name}")
        for name, item in endpoints.items()
    )
    summaries: dict[str, dict[str, float | int]] = {}
    for name in ("target_G", "background_G", "target_E", "background_E"):
        raw = value.get(name)
        if not isinstance(raw, Mapping) or set(raw) != {
            "count",
            "minimum",
            "maximum",
            "mean",
        }:
            raise ValueError(f"{name} distribution fields changed")
        count = _count(raw.get("count"), name=f"{name}.count")
        if count < 1:
            raise ValueError(f"{name}.count must be positive")
        minimum = _finite_real(raw.get("minimum"), name=f"{name}.minimum")
        maximum = _finite_real(raw.get("maximum"), name=f"{name}.maximum")
        mean = _finite_real(raw.get("mean"), name=f"{name}.mean")
        if not minimum <= mean <= maximum:
            raise ValueError(f"{name} summary order is invalid")
        if name.endswith("_G") and (minimum < 0.0 or maximum > 2.0):
            raise ValueError(f"{name} left the closed gate interval")
        summaries[name] = {
            "count": count,
            "minimum": minimum,
            "maximum": maximum,
            "mean": mean,
        }
    if (
        endpoint_total
        != summaries["target_G"]["count"]
        + summaries["background_G"]["count"]
        or summaries["target_G"]["count"]
        != summaries["target_E"]["count"]
        or summaries["background_G"]["count"]
        != summaries["background_E"]["count"]
    ):
        raise ValueError("gate endpoint/role distribution counts differ")


def _source_hashes() -> tuple[tuple[str, str], ...]:
    return gcr_pacre_v24_source_closure_hashes()


def _named_parameter_rows(
    model: torch.nn.Module,
) -> tuple[dict[str, object], ...]:
    rows = tuple(
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "content_fingerprint": tensor_content_fingerprint(parameter),
        }
        for name, parameter in model.named_parameters()
    )
    if tuple(str(row["name"]) for row in rows) != _PARAMETER_NAMES:
        raise RuntimeError("bounded parameter inventory changed")
    return rows


def _logical_storage_ids(
    *,
    arm: str,
    initial_rows: tuple[dict[str, object], ...],
) -> tuple[str, ...]:
    return tuple(
        stable_fingerprint(
            {
                "schema_version": (
                    "cure-lite-v24-bounded-logical-parameter-storage-v1"
                ),
                "arm": arm,
                **row,
            }
        )
        for row in initial_rows
    )


def _module_storage_addresses(
    model: torch.nn.Module,
) -> set[tuple[str, int, int]]:
    return {
        (
            str(parameter.device),
            int(parameter.untyped_storage().data_ptr()),
            int(parameter.untyped_storage().nbytes()),
        )
        for parameter in model.parameters()
    }


def _device_cache_storage_addresses(
    cache: CoverageStateDeviceCache,
) -> set[tuple[str, int, int]]:
    return {
        (
            str(tensor.device),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        )
        for _, tensor in cache.named_tensors()
    }


def _cache_storage_addresses(
    cache: CoverageStateScalarCache,
) -> set[tuple[str, int, int]]:
    """Walk only neutral containers/dataclasses and collect tensor storage."""

    addresses: set[tuple[str, int, int]] = set()
    visited: set[int] = set()

    def visit(value: object) -> None:
        identity = id(value)
        if identity in visited:
            return
        if isinstance(value, torch.Tensor):
            visited.add(identity)
            addresses.add(
                (
                    str(value.device),
                    int(value.untyped_storage().data_ptr()),
                    int(value.untyped_storage().nbytes()),
                )
            )
            return
        if is_dataclass(value) and not isinstance(value, type):
            visited.add(identity)
            for row in dataclass_fields(value):
                visit(getattr(value, row.name))
            return
        if isinstance(value, Mapping):
            visited.add(identity)
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (tuple, list)):
            visited.add(identity)
            for item in value:
                visit(item)

    visit(cache)
    if not addresses:
        raise RuntimeError("bounded scalar cache has no tensor storage")
    return addresses


def _seeded_model(
    factory,
    config: object,
    *,
    seed: int,
) -> torch.nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return factory(config)


def _resolved_device(device: torch.device | str) -> torch.device:
    try:
        result = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise TypeError("device must identify a torch device") from error
    if result.type == "cuda" and result.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        result = torch.device("cuda", torch.cuda.current_device())
    return result


@contextmanager
def _deterministic_execution(device: torch.device):
    old_algorithms = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        yield
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        torch.use_deterministic_algorithms(
            old_algorithms,
            warn_only=old_warn_only,
        )
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
        torch.backends.cudnn.allow_tf32 = old_tf32_cudnn


@dataclass(frozen=True)
class GCRPACREBoundedEvaluation:
    """One frozen evaluator output at an initial/terminal model state."""

    true_targets: int
    recovered_anchor_misses: int
    mIoU: float
    nIoU: float
    pd: float
    retention: float
    pixel_fa: float
    raw_background_fa: float
    fp_components_per_mp: float
    budget_violation: bool
    PMOPE: float
    target_role_violation: float
    background_role_violation: float
    zero_crossed_target_states: int
    false_completion_states: int
    gate_role_distributions_present: bool
    gate_role_distribution_json: str | None
    field_fingerprint: str
    role_prediction_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "true_targets",
            "recovered_anchor_misses",
            "zero_crossed_target_states",
            "false_completion_states",
        ):
            _count(getattr(self, name), name=name)
        for name in (
            "mIoU",
            "nIoU",
            "pd",
            "retention",
            "pixel_fa",
            "raw_background_fa",
            "fp_components_per_mp",
            "PMOPE",
            "target_role_violation",
            "background_role_violation",
        ):
            _finite_real(getattr(self, name), name=name)
        if (
            not isinstance(self.budget_violation, bool)
            or not isinstance(self.gate_role_distributions_present, bool)
            or not _is_sha256(self.field_fingerprint)
            or not _is_sha256(self.role_prediction_fingerprint)
        ):
            raise ValueError("bounded evaluator output is malformed")
        if self.gate_role_distribution_json is None:
            if self.gate_role_distributions_present:
                raise ValueError("gate distribution presence flag is inconsistent")
        else:
            try:
                distribution = json.loads(
                    self.gate_role_distribution_json
                )
            except json.JSONDecodeError as error:
                raise ValueError("gate role distribution JSON is invalid") from error
            if (
                not self.gate_role_distributions_present
                or not isinstance(distribution, dict)
                or canonical_json(distribution)
                != self.gate_role_distribution_json
            ):
                raise ValueError("gate role distribution binding is invalid")
            _validate_gate_role_distribution(distribution)

    @property
    def gate_role_distribution(self) -> dict[str, object] | None:
        if self.gate_role_distribution_json is None:
            return None
        value = json.loads(self.gate_role_distribution_json)
        if not isinstance(value, dict):
            raise AssertionError("validated gate distribution changed")
        return value

    def canonical_payload(self) -> dict[str, object]:
        return {
            "true_targets": self.true_targets,
            "recovered_anchor_misses": self.recovered_anchor_misses,
            "mIoU": self.mIoU,
            "nIoU": self.nIoU,
            "pd": self.pd,
            "retention": self.retention,
            "pixel_fa": self.pixel_fa,
            "raw_background_fa": self.raw_background_fa,
            "fp_components_per_mp": self.fp_components_per_mp,
            "budget_violation": self.budget_violation,
            "PMOPE": self.PMOPE,
            "target_role_violation": self.target_role_violation,
            "background_role_violation": self.background_role_violation,
            "zero_crossed_target_states": (
                self.zero_crossed_target_states
            ),
            "false_completion_states": self.false_completion_states,
            "gate_role_distributions_present": (
                self.gate_role_distributions_present
            ),
            "gate_role_distribution": self.gate_role_distribution,
            "field_fingerprint": self.field_fingerprint,
            "role_prediction_fingerprint": (
                self.role_prediction_fingerprint
            ),
        }


class GCRPACREBoundedEvaluator(ABC):
    """Frozen D_R evaluator interface bound by an authorization token."""

    @property
    @abstractmethod
    def evaluator_fingerprint(self) -> str:
        """Return the frozen evaluator/source/aggregation fingerprint."""

    @abstractmethod
    def evaluate(
        self,
        model: torch.nn.Module,
        cache: CoverageStateScalarCache,
        *,
        arm: str,
        checkpoint: str,
        forward_mode: str,
    ) -> GCRPACREBoundedEvaluation:
        """Evaluate one exact model without mutating it."""


class _OneShotAttempt:
    def __init__(self) -> None:
        self.lock = Lock()
        self.state = "available"

    def claim(self) -> None:
        with self.lock:
            if self.state != "available":
                raise PermissionError("bounded authorization is no longer available")
            self.state = "running"

    def consume(self) -> None:
        with self.lock:
            if self.state != "running":
                raise PermissionError("bounded authorization was not claimed")
            self.state = "consumed"

    def fail(self) -> None:
        with self.lock:
            if self.state in {"running", "consumed"}:
                self.state = "failed"


@dataclass(frozen=True, eq=False)
class GCRPACREBoundedAuthorization(CoverageStateRunAuthorization):
    """Protocol-token-bound approval for exactly one paired bounded run."""

    oof_decision: VerifiedOOFDecision
    access_audit: VerifiedAccessAudit
    full_d_r_cache_artifact: VerifiedFormalCacheArtifact
    chain_config: VerifiedGCRPACREBoundedChainConfig
    dataset_free_receipt_fingerprint: str
    d_r_structural_receipt_fingerprint: str
    control_cache: CoverageStateScalarCache
    candidate_cache: CoverageStateScalarCache
    schedule: CoverageStateTrainingSchedule
    control_config: CoverageStatePACREVerifierCorrectedConfig
    candidate_config: CoverageStateGCRPACREConfig
    evaluator: GCRPACREBoundedEvaluator
    evaluator_fingerprint: str
    source_hashes: tuple[tuple[str, str], ...]
    _attempt: _OneShotAttempt = field(
        default_factory=_OneShotAttempt,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.verify_unchanged()

    @property
    def requested_device(self) -> str:
        value = self.chain_config.payload["requested_device"]
        if not isinstance(value, str):
            raise AssertionError("bounded chain device changed")
        return value

    @property
    def output_directory(self) -> str:
        value = self.chain_config.payload["output_directory"]
        if not isinstance(value, str):
            raise AssertionError("bounded chain output changed")
        return value

    def verify_unchanged(self) -> None:
        oof = require_verified_oof_decision(self.oof_decision)
        access = require_verified_access_audit(self.access_audit)
        assert isinstance(oof, VerifiedOOFDecision)
        assert isinstance(access, VerifiedAccessAudit)
        full_cache = require_verified_formal_cache_artifact(
            self.full_d_r_cache_artifact
        )
        chain = require_verified_gcr_pacre_bounded_chain_config(
            self.chain_config
        )
        reverified_full_cache = verify_formal_cache_artifact(
            full_cache.path,
            cache_id=full_cache.cache_id,
            expected_semantic_cache_fingerprint=(
                full_cache.semantic_cache_fingerprint
            ),
            expected_neutral_payload_fingerprint=(
                full_cache.neutral_payload_fingerprint
            ),
        )
        if (
            oof.payload.get("gate_passed") is not True
            or access.stage_id != "paired_bounded400"
            or access.allowed_splits != ("D_R",)
            or not _is_sha256(self.dataset_free_receipt_fingerprint)
            or not _is_sha256(self.d_r_structural_receipt_fingerprint)
            or full_cache.cache_id
            != "paired-bounded400-full-D_R-materialization"
            or reverified_full_cache.receipt_fingerprint
            != full_cache.receipt_fingerprint
        ):
            raise PermissionError("bounded predecessor tokens are not eligible")
        verify_gcr_pacre_bounded_chain_authorization_binding(
            chain,
            oof_decision=oof,
            access_audit=access,
            full_d_r_cache_artifact=full_cache,
            dataset_free_receipt_fingerprint=(
                self.dataset_free_receipt_fingerprint
            ),
            d_r_structural_receipt_fingerprint=(
                self.d_r_structural_receipt_fingerprint
            ),
        )
        if (
            type(self.control_cache) is not CoverageStateScalarCache
            or type(self.candidate_cache) is not CoverageStateScalarCache
            or self.control_cache is self.candidate_cache
        ):
            raise TypeError("bounded arms require two exact independent caches")
        self.control_cache.verify_unchanged()
        self.candidate_cache.verify_unchanged()
        if (
            self.control_cache.cache_fingerprint
            != self.candidate_cache.cache_fingerprint
            or self.control_cache.cache_fingerprint
            != full_cache.semantic_cache_fingerprint
        ):
            raise ValueError("bounded caches differ semantically")
        if _cache_storage_addresses(
            self.control_cache
        ) & _cache_storage_addresses(self.candidate_cache):
            raise PermissionError("bounded scalar caches share tensor storage")
        if (
            type(self.schedule) is not CoverageStateTrainingSchedule
            or self.schedule.cache_fingerprint
            != self.control_cache.cache_fingerprint
            or (
                self.schedule.config.seed,
                self.schedule.config.epochs,
                self.schedule.config.steps_per_epoch,
            )
            != (
                GCR_PACRE_BOUNDED_SEED,
                GCR_PACRE_BOUNDED_EPOCHS,
                GCR_PACRE_BOUNDED_STEPS_PER_EPOCH,
            )
        ):
            raise ValueError("bounded schedule differs from frozen 10x40 seed42")
        coverage_state_schedule_exposure_report(
            self.control_cache,
            self.schedule,
        )
        coverage_state_schedule_exposure_report(
            self.candidate_cache,
            self.schedule,
        )
        if (
            type(self.control_config)
            is not CoverageStatePACREVerifierCorrectedConfig
            or type(self.candidate_config) is not CoverageStateGCRPACREConfig
            or (
                self.control_config.feature_channels,
                self.control_config.feature_stride,
                self.control_config.width,
            )
            != (
                self.candidate_config.feature_channels,
                self.candidate_config.feature_stride,
                self.candidate_config.width,
            )
            or self.control_config.expected_parameter_count
            != self.candidate_config.expected_parameter_count
        ):
            raise ValueError("bounded arm model coordinates differ")
        if not isinstance(self.evaluator, GCRPACREBoundedEvaluator):
            raise TypeError("evaluator must implement GCRPACREBoundedEvaluator")
        if (
            not _is_sha256(self.evaluator_fingerprint)
            or self.evaluator.evaluator_fingerprint
            != self.evaluator_fingerprint
        ):
            raise PermissionError("bounded evaluator binding changed")
        if (
            self.source_hashes != _source_hashes()
            or dict(self.source_hashes)
            != chain.payload.get("source_hashes")
        ):
            raise RuntimeError("bounded unified source closure changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": GCR_PACRE_BOUNDED_AUTHORIZATION_SCHEMA,
            "predecessor_tokens": {
                "OOF4_decision_fingerprint": (
                    self.oof_decision.decision_fingerprint
                ),
                "access_audit_receipt_fingerprint": (
                    self.access_audit.receipt_fingerprint
                ),
                "full_D_R_cache_materialization_receipt_fingerprint": (
                    self.full_d_r_cache_artifact.receipt_fingerprint
                ),
                "dataset_free_receipt_fingerprint": (
                    self.dataset_free_receipt_fingerprint
                ),
                "D_R_structural_receipt_fingerprint": (
                    self.d_r_structural_receipt_fingerprint
                ),
            },
            "chain_config": {
                "path": self.chain_config.path,
                "file_sha256": self.chain_config.file_sha256,
                "config_fingerprint": (
                    self.chain_config.config_fingerprint
                ),
                "source_closure_fingerprint": (
                    self.chain_config.source_closure_fingerprint
                ),
            },
            "execution_binding": {
                name: self.chain_config.payload[name]
                for name in (
                    "requested_device",
                    "output_directory",
                    "run_start_marker_path",
                    "authorization_artifact_path",
                    "schedule_artifact_path",
                    "control_terminal_artifact_path",
                    "candidate_terminal_artifact_path",
                    "result_artifact_path",
                    "diagnostics_artifact_path",
                    "decision_artifact_path",
                )
            },
            "budget": {
                "seed": GCR_PACRE_BOUNDED_SEED,
                "epochs": GCR_PACRE_BOUNDED_EPOCHS,
                "steps_per_epoch": GCR_PACRE_BOUNDED_STEPS_PER_EPOCH,
                "updates": GCR_PACRE_BOUNDED_UPDATES,
                "training_invocations_per_arm": 1,
            },
            "population_fingerprint": (
                self.control_cache.cache_fingerprint
            ),
            "schedule_fingerprint": self.schedule.schedule_fingerprint,
            "evaluator_fingerprint": self.evaluator_fingerprint,
            "source_hashes": dict(self.source_hashes),
            "objective": CoverageStatePairObjective.PMOPE_JOINT.value,
            "objective_policy": CSLF_PMOPE_POLICY,
            "optimizer": {
                "class": "torch.optim.adam.Adam",
                "learning_rate_hex": (0.001).hex(),
                "betas_hex": [(0.9).hex(), (0.999).hex()],
                "epsilon_hex": (1.0e-8).hex(),
                "weight_decay_hex": (0.0).hex(),
            },
            "arms": [GCR_PACRE_CONTROL_ARM, GCR_PACRE_CANDIDATE_ARM],
            "relative_diagnostics_authorize": False,
            "fixed_relative_promotion_threshold": None,
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "checkpoint_policy": "final_only",
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }

    @property
    def authorization_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_for_run(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> None:
        del cache, schedule, scope
        raise PermissionError(
            "paired bounded authorization is consumable only by the paired runner"
        )


def prepare_gcr_pacre_paired_bounded_authorization(
    *,
    oof_decision: VerifiedOOFDecision,
    access_audit: VerifiedAccessAudit,
    full_d_r_cache_artifact: VerifiedFormalCacheArtifact,
    chain_config: VerifiedGCRPACREBoundedChainConfig,
    dataset_free_receipt_fingerprint: str,
    d_r_structural_receipt_fingerprint: str,
    control_cache: CoverageStateScalarCache,
    candidate_cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    candidate_config: CoverageStateGCRPACREConfig,
    evaluator: GCRPACREBoundedEvaluator,
) -> GCRPACREBoundedAuthorization:
    """Bind verified predecessors and both physical arms before allocation."""

    if type(candidate_config) is not CoverageStateGCRPACREConfig:
        raise TypeError("candidate_config must be the exact v24 config")
    control_config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=candidate_config.feature_channels,
        feature_stride=candidate_config.feature_stride,
        width=candidate_config.width,
    )
    return GCRPACREBoundedAuthorization(
        oof_decision=oof_decision,
        access_audit=access_audit,
        full_d_r_cache_artifact=full_d_r_cache_artifact,
        chain_config=chain_config,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt_fingerprint
        ),
        d_r_structural_receipt_fingerprint=(
            d_r_structural_receipt_fingerprint
        ),
        control_cache=control_cache,
        candidate_cache=candidate_cache,
        schedule=schedule,
        control_config=control_config,
        candidate_config=candidate_config,
        evaluator=evaluator,
        evaluator_fingerprint=evaluator.evaluator_fingerprint,
        source_hashes=_source_hashes(),
    )


@dataclass(frozen=True)
class GCRPACREPairedUpdateDiagnostic:
    update: int
    selection_fingerprint: str
    control_loss: float
    candidate_loss: float
    candidate_minus_control_loss: float
    control_gradient_l2_norm: float
    candidate_gradient_l2_norm: float
    candidate_minus_control_gradient_l2_norm: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.update, bool)
            or not isinstance(self.update, int)
            or not 0 <= self.update < GCR_PACRE_BOUNDED_UPDATES
            or not _is_sha256(self.selection_fingerprint)
        ):
            raise ValueError("paired update identity is invalid")
        for name in (
            "control_loss",
            "candidate_loss",
            "candidate_minus_control_loss",
            "control_gradient_l2_norm",
            "candidate_gradient_l2_norm",
            "candidate_minus_control_gradient_l2_norm",
        ):
            _finite_real(getattr(self, name), name=name)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "update": self.update,
            "selection_fingerprint": self.selection_fingerprint,
            "control_loss": self.control_loss,
            "candidate_loss": self.candidate_loss,
            "candidate_minus_control_loss": (
                self.candidate_minus_control_loss
            ),
            "control_gradient_l2_norm": self.control_gradient_l2_norm,
            "candidate_gradient_l2_norm": (
                self.candidate_gradient_l2_norm
            ),
            "candidate_minus_control_gradient_l2_norm": (
                self.candidate_minus_control_gradient_l2_norm
            ),
        }


@dataclass(frozen=True, eq=False)
class GCRPACREBoundedArmResult:
    arm: str
    role: str
    model: torch.nn.Module
    training_result: CoverageStateTrainingResult
    initial_parameters: tuple[dict[str, object], ...]
    initial_evaluation: GCRPACREBoundedEvaluation
    terminal_evaluation: GCRPACREBoundedEvaluation
    forced_g1_evaluation: GCRPACREBoundedEvaluation
    terminal_artifact: Mapping[str, object]
    source_hashes: tuple[tuple[str, str], ...]
    cache_instance_id: str
    rng_instance_id: str
    module_instance_id: str
    optimizer_instance_id: str
    parameter_storage_ids: tuple[str, ...]

    def metrics_payload(self) -> dict[str, object]:
        terminal = self.terminal_evaluation
        return {
            "true_targets": terminal.true_targets,
            "recovered_anchor_misses": (
                terminal.recovered_anchor_misses
            ),
            "mIoU": terminal.mIoU,
            "nIoU": terminal.nIoU,
            "pd": terminal.pd,
            "retention": terminal.retention,
            "pixel_fa": terminal.pixel_fa,
            "raw_background_fa": terminal.raw_background_fa,
            "fp_components_per_mp": terminal.fp_components_per_mp,
            "budget_violation": terminal.budget_violation,
            "initial_PMOPE": self.initial_evaluation.PMOPE,
            "terminal_PMOPE": terminal.PMOPE,
            "terminal_target_role_violation": (
                terminal.target_role_violation
            ),
            "terminal_background_role_violation": (
                terminal.background_role_violation
            ),
            "terminal_zero_crossed_target_states": (
                terminal.zero_crossed_target_states
            ),
            "terminal_false_completion_states": (
                terminal.false_completion_states
            ),
            "G1_zero_crossed_target_states": (
                self.forced_g1_evaluation.zero_crossed_target_states
            ),
            "terminal_field_fingerprint": terminal.field_fingerprint,
            "terminal_role_prediction_fingerprint": (
                terminal.role_prediction_fingerprint
            ),
            "G1_PMOPE": self.forced_g1_evaluation.PMOPE,
            "G1_target_role_violation": (
                self.forced_g1_evaluation.target_role_violation
            ),
            "G1_background_role_violation": (
                self.forced_g1_evaluation.background_role_violation
            ),
            "G1_false_completion_states": (
                self.forced_g1_evaluation.false_completion_states
            ),
            "G1_field_fingerprint": (
                self.forced_g1_evaluation.field_fingerprint
            ),
            "G1_role_prediction_fingerprint": (
                self.forced_g1_evaluation.role_prediction_fingerprint
            ),
            "terminal_gate_distribution": (
                terminal.gate_role_distribution
            ),
            "G1_gate_distribution": (
                self.forced_g1_evaluation.gate_role_distribution
            ),
            "gate_role_distributions_present": (
                terminal.gate_role_distributions_present
            ),
        }


@dataclass(frozen=True, eq=False)
class GCRPACREPairedBoundedResult:
    authorization: GCRPACREBoundedAuthorization
    run_start_token: GCRPACREBoundedRunStartToken
    control: GCRPACREBoundedArmResult
    candidate: GCRPACREBoundedArmResult
    updates: tuple[GCRPACREPairedUpdateDiagnostic, ...]
    schedule_artifact: Mapping[str, object]
    training_trace_artifact: Mapping[str, object]

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        verify_gcr_pacre_bounded_run_start_token(
            self.authorization,
            self.run_start_token,
        )
        if (
            self.authorization._attempt.state != "consumed"
            or self.control.arm != GCR_PACRE_CONTROL_ARM
            or self.candidate.arm != GCR_PACRE_CANDIDATE_ARM
            or len(self.updates) != GCR_PACRE_BOUNDED_UPDATES
            or tuple(row.update for row in self.updates)
            != tuple(range(GCR_PACRE_BOUNDED_UPDATES))
        ):
            raise PermissionError("bounded paired result is incomplete")
        for arm in (self.control, self.candidate):
            result = arm.training_result
            if (
                result.completed_updates != GCR_PACRE_BOUNDED_UPDATES
                or result.final_model_fingerprint
                != coverage_state_model_fingerprint(arm.model)
                or result.initial_model_fingerprint
                == result.final_model_fingerprint
                or arm.terminal_evaluation.PMOPE
                > arm.initial_evaluation.PMOPE
            ):
                raise ValueError("bounded arm optimization sanity failed")
            if arm.source_hashes != _source_hashes():
                raise RuntimeError("bounded source bytes changed")
        candidate_before = coverage_state_model_fingerprint(
            self.candidate.model
        )
        trace_path = Path(str(self.training_trace_artifact.get("path")))
        if (
            candidate_before
            != self.candidate.training_result.final_model_fingerprint
            or trace_path
            != Path(self.authorization.output_directory)
            / "training_trace.json"
            or not trace_path.is_file()
            or trace_path.is_symlink()
            or trace_path.resolve(strict=True) != trace_path
            or trace_path.stat().st_nlink != 1
            or trace_path.stat().st_mode & 0o222
            or trace_path.stat().st_size
            != self.training_trace_artifact.get("size_bytes")
            or file_sha256(trace_path)
            != self.training_trace_artifact.get("file_sha256")
        ):
            raise RuntimeError("candidate terminal model changed")
        if (
            self.candidate.terminal_evaluation.field_fingerprint
            == self.candidate.forced_g1_evaluation.field_fingerprint
            or self.candidate.terminal_evaluation.role_prediction_fingerprint
            == self.candidate.forced_g1_evaluation.role_prediction_fingerprint
        ):
            raise ValueError(
                "candidate/G1 nonidentity witnesses are absent"
            )
        if (
            self.control.terminal_evaluation.gate_role_distribution
            is not None
            or self.candidate.terminal_evaluation.gate_role_distribution
            is None
            or self.candidate.forced_g1_evaluation.gate_role_distribution
            is None
        ):
            raise ValueError(
                "bounded v23/v24 gate distribution applicability changed"
            )

    @property
    def diagnostic_payload(self) -> dict[str, object]:
        candidate = self.candidate.terminal_evaluation
        control = self.control.terminal_evaluation
        g1 = self.candidate.forced_g1_evaluation
        return {
            "interpretation": (
                "paired_deltas_are_diagnostic_only_without_a_fixed_threshold"
            ),
            "candidate_minus_control": {
                "PMOPE": candidate.PMOPE - control.PMOPE,
                "target_role_violation": (
                    candidate.target_role_violation
                    - control.target_role_violation
                ),
                "background_role_violation": (
                    candidate.background_role_violation
                    - control.background_role_violation
                ),
                "zero_crossed_target_states": (
                    candidate.zero_crossed_target_states
                    - control.zero_crossed_target_states
                ),
                "false_completion_states": (
                    candidate.false_completion_states
                    - control.false_completion_states
                ),
            },
            "candidate_minus_same_weight_G1": {
                "PMOPE": candidate.PMOPE - g1.PMOPE,
                "target_role_violation": (
                    candidate.target_role_violation
                    - g1.target_role_violation
                ),
                "background_role_violation": (
                    candidate.background_role_violation
                    - g1.background_role_violation
                ),
                "zero_crossed_target_states": (
                    candidate.zero_crossed_target_states
                    - g1.zero_crossed_target_states
                ),
                "false_completion_states": (
                    candidate.false_completion_states
                    - g1.false_completion_states
                ),
                "field_nonidentity_witness": (
                    candidate.field_fingerprint != g1.field_fingerprint
                ),
                "role_prediction_nonidentity_witness": (
                    candidate.role_prediction_fingerprint
                    != g1.role_prediction_fingerprint
                ),
            },
            "per_update_fingerprint": stable_fingerprint(
                [row.canonical_payload() for row in self.updates]
            ),
            "per_update": [
                row.canonical_payload() for row in self.updates
            ],
        }

    @property
    def result_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(
            {
                "schema_version": GCR_PACRE_BOUNDED_RESULT_SCHEMA,
                "authorization_fingerprint": (
                    self.authorization.authorization_fingerprint
                ),
                "run_start_marker_fingerprint": (
                    self.run_start_token.marker_fingerprint
                ),
                "control_training_result_fingerprint": (
                    self.control.training_result.result_fingerprint
                ),
                "candidate_training_result_fingerprint": (
                    self.candidate.training_result.result_fingerprint
                ),
                "diagnostic_payload": self.diagnostic_payload,
                "training_trace_fingerprint": (
                    self.training_trace_artifact["trace_fingerprint"]
                ),
            }
        )


def _evaluate_unchanged(
    authorization: GCRPACREBoundedAuthorization,
    model: torch.nn.Module,
    cache: CoverageStateScalarCache,
    *,
    arm: str,
    checkpoint: str,
    forward_mode: str,
) -> GCRPACREBoundedEvaluation:
    before = coverage_state_model_fingerprint(model)
    value = authorization.evaluator.evaluate(
        model,
        cache,
        arm=arm,
        checkpoint=checkpoint,
        forward_mode=forward_mode,
    )
    after = coverage_state_model_fingerprint(model)
    if (
        type(value) is not GCRPACREBoundedEvaluation
        or before != after
        or authorization.evaluator.evaluator_fingerprint
        != authorization.evaluator_fingerprint
    ):
        raise RuntimeError("bounded evaluator mutated or changed its binding")
    return value


def _training_result(
    *,
    model: torch.nn.Module,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    device_cache: CoverageStateDeviceCache,
    optimizer_fingerprint: str,
    initial_fingerprint: str,
    epoch_logs: list[dict[str, object]],
    first_nonzero: Mapping[str, int],
    device: torch.device,
) -> CoverageStateTrainingResult:
    if set(first_nonzero) != set(_PARAMETER_NAMES):
        raise RuntimeError("bounded model has a missing gradient path")
    return CoverageStateTrainingResult(
        objective=CoverageStatePairObjective.PMOPE_JOINT.value,
        objective_policy=coverage_state_pair_objective_policy(
            CoverageStatePairObjective.PMOPE_JOINT
        ),
        seed=GCR_PACRE_BOUNDED_SEED,
        epochs=GCR_PACRE_BOUNDED_EPOCHS,
        steps_per_epoch=GCR_PACRE_BOUNDED_STEPS_PER_EPOCH,
        completed_updates=GCR_PACRE_BOUNDED_UPDATES,
        schedule_fingerprint=schedule.schedule_fingerprint,
        cache_fingerprint=cache.cache_fingerprint,
        execution_device=str(device),
        device_cache_fingerprint=device_cache.device_cache_fingerprint,
        device_cache_resident_bytes=device_cache.resident_tensor_bytes,
        optimizer_config_fingerprint=optimizer_fingerprint,
        initial_model_fingerprint=initial_fingerprint,
        final_model_fingerprint=coverage_state_model_fingerprint(model),
        epoch_logs=tuple(epoch_logs),
        first_nonzero_gradient_update=tuple(sorted(first_nonzero.items())),
        forward_calls=GCR_PACRE_BOUNDED_UPDATES,
        backward_calls=GCR_PACRE_BOUNDED_UPDATES,
        optimizer_steps=GCR_PACRE_BOUNDED_UPDATES,
        logical_state_evaluations=GCR_PACRE_BOUNDED_UPDATES * 12,
        finite_state_audits=GCR_PACRE_BOUNDED_UPDATES + 1,
    )


def run_gcr_pacre_paired_bounded_400(
    authorization: GCRPACREBoundedAuthorization,
    *,
    run_start_token: GCRPACREBoundedRunStartToken,
    output_directory: str | Path,
    device: torch.device | str = "cpu",
) -> GCRPACREPairedBoundedResult:
    """Execute the exact synchronized two-arm 10x40 run once."""

    if type(authorization) is not GCRPACREBoundedAuthorization:
        raise TypeError("authorization must be exact bounded authorization")
    authorization._attempt.claim()
    try:
        authorization.verify_unchanged()
        resolved = _resolved_device(device)
        verify_gcr_pacre_bounded_run_start_token(
            authorization,
            run_start_token,
        )
        output = Path(output_directory)
        if (
            str(resolved) != authorization.requested_device
            or output != Path(authorization.output_directory)
        ):
            raise PermissionError(
                "bounded device/output differ from frozen chain config"
            )
        with _deterministic_execution(resolved):
            control = _seeded_model(
                build_pacre_vc_training_model,
                authorization.control_config,
                seed=GCR_PACRE_BOUNDED_SEED,
            )
            candidate = _seeded_model(
                build_gcr_pacre_training_model,
                authorization.candidate_config,
                seed=GCR_PACRE_BOUNDED_SEED,
            )
            if (
                type(control) is not CURELitePACREVerifierCorrectedLevelSet
                or type(candidate)
                is not CURELiteGatedCommonResidualPACRELevelSet
            ):
                raise AssertionError("bounded factory returned a wrong arm")
            control_rows = _named_parameter_rows(control)
            candidate_rows = _named_parameter_rows(candidate)
            if control_rows != candidate_rows:
                raise RuntimeError(
                    "bounded arm initial names/shapes/bytes are not exact"
                )
            shared_parameter_fingerprint = stable_fingerprint(
                list(control_rows)
            )
            control_initial = coverage_state_model_fingerprint(control)
            candidate_initial = coverage_state_model_fingerprint(candidate)
            if control_initial != candidate_initial:
                raise RuntimeError("bounded initial state bytes differ")
            control = control.to(device=resolved, dtype=torch.float32)
            candidate = candidate.to(device=resolved, dtype=torch.float32)
            if (
                coverage_state_model_fingerprint(control) != control_initial
                or coverage_state_model_fingerprint(candidate)
                != candidate_initial
                or _module_storage_addresses(control)
                & _module_storage_addresses(candidate)
            ):
                raise RuntimeError("bounded model storage is shared or changed")

            control_optimizer = torch.optim.Adam(
                control.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1.0e-8,
                weight_decay=0.0,
            )
            candidate_optimizer = torch.optim.Adam(
                candidate.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1.0e-8,
                weight_decay=0.0,
            )
            if (
                type(control_optimizer) is not torch.optim.Adam
                or type(candidate_optimizer) is not torch.optim.Adam
                or control_optimizer.state
                or candidate_optimizer.state
                or control_optimizer is candidate_optimizer
            ):
                raise RuntimeError("bounded arms require independent empty Adam")
            control_optimizer_fp = (
                coverage_state_optimizer_config_fingerprint(
                    control,
                    control_optimizer,
                )
            )
            candidate_optimizer_fp = (
                coverage_state_optimizer_config_fingerprint(
                    candidate,
                    candidate_optimizer,
                )
            )
            if control_optimizer_fp != candidate_optimizer_fp:
                raise RuntimeError("bounded Adam policies differ")

            control_device_cache = prepare_coverage_state_device_cache(
                authorization.control_cache,
                device=resolved,
            )
            candidate_device_cache = prepare_coverage_state_device_cache(
                authorization.candidate_cache,
                device=resolved,
            )
            if (
                control_device_cache is candidate_device_cache
                or _device_cache_storage_addresses(control_device_cache)
                & _device_cache_storage_addresses(candidate_device_cache)
            ):
                raise RuntimeError("bounded packed cache storage is shared")
            control_device_cache.verify_unchanged()
            candidate_device_cache.verify_unchanged()
            audit_coverage_state_training_state(control, control_optimizer)
            audit_coverage_state_training_state(
                candidate,
                candidate_optimizer,
            )

            control_initial_eval = _evaluate_unchanged(
                authorization,
                control,
                authorization.control_cache,
                arm=GCR_PACRE_CONTROL_ARM,
                checkpoint="initial",
                forward_mode=GCR_PACRE_NATIVE_MODE,
            )
            candidate_initial_eval = _evaluate_unchanged(
                authorization,
                candidate,
                authorization.candidate_cache,
                arm=GCR_PACRE_CANDIDATE_ARM,
                checkpoint="initial",
                forward_mode=GCR_PACRE_NATIVE_MODE,
            )

            update_rows: list[GCRPACREPairedUpdateDiagnostic] = []
            raw_trace_rows: list[dict[str, object]] = []
            first_nonzero = {
                GCR_PACRE_CONTROL_ARM: {},
                GCR_PACRE_CANDIDATE_ARM: {},
            }
            epoch_logs = {
                GCR_PACRE_CONTROL_ARM: [],
                GCR_PACRE_CANDIDATE_ARM: [],
            }
            epoch_sums = {
                GCR_PACRE_CONTROL_ARM: {
                    "factual_miss/loss": 0.0,
                    "factual_no_miss/loss": 0.0,
                    "pair/loss": 0.0,
                    "total": 0.0,
                    "gradient_l2_norm": 0.0,
                },
                GCR_PACRE_CANDIDATE_ARM: {
                    "factual_miss/loss": 0.0,
                    "factual_no_miss/loss": 0.0,
                    "pair/loss": 0.0,
                    "total": 0.0,
                    "gradient_l2_norm": 0.0,
                },
            }
            epoch_selections: list[str] = []
            for update, selection in enumerate(
                authorization.schedule.selections
            ):
                control_batch = control_device_cache.materialize(
                    selection,
                    verify=False,
                    validate=False,
                )
                candidate_batch = candidate_device_cache.materialize(
                    selection,
                    verify=False,
                    validate=False,
                )
                if (
                    control_batch.selection_fingerprint
                    != candidate_batch.selection_fingerprint
                    or control_batch.factual_miss.record_ids
                    != selection.factual_miss_record_ids
                    or control_batch.factual_no_miss.record_ids
                    != selection.factual_no_miss_record_ids
                    or control_batch.pairs.pair_ids
                    != (
                        selection.clean_positive_pair_id,
                        selection.component_null_pair_id,
                    )
                    or candidate_batch.factual_miss.record_ids
                    != selection.factual_miss_record_ids
                    or candidate_batch.factual_no_miss.record_ids
                    != selection.factual_no_miss_record_ids
                    or candidate_batch.pairs.pair_ids
                    != (
                        selection.clean_positive_pair_id,
                        selection.component_null_pair_id,
                    )
                ):
                    raise RuntimeError("bounded paired batch order changed")
                control_log = coverage_state_fused_train_step(
                    control,
                    control_optimizer,
                    control_batch,
                    config=authorization.control_cache.sobolev_config,
                    pair_objective=CoverageStatePairObjective.PMOPE_JOINT,
                    audit=False,
                    track_nonzero_gradients=(
                        len(first_nonzero[GCR_PACRE_CONTROL_ARM])
                        < len(_PARAMETER_NAMES)
                    ),
                )
                candidate_log = coverage_state_fused_train_step(
                    candidate,
                    candidate_optimizer,
                    candidate_batch,
                    config=authorization.candidate_cache.sobolev_config,
                    pair_objective=CoverageStatePairObjective.PMOPE_JOINT,
                    audit=False,
                    track_nonzero_gradients=(
                        len(first_nonzero[GCR_PACRE_CANDIDATE_ARM])
                        < len(_PARAMETER_NAMES)
                    ),
                )
                for arm, log in (
                    (GCR_PACRE_CONTROL_ARM, control_log),
                    (GCR_PACRE_CANDIDATE_ARM, candidate_log),
                ):
                    for metric_name in epoch_sums[arm]:
                        epoch_sums[arm][metric_name] += _finite_real(
                            log[metric_name],
                            name=f"{arm}.{metric_name}",
                        )
                    for name in filter(
                        None,
                        str(log["nonzero_gradient_parameters"]).split(","),
                    ):
                        first_nonzero[arm].setdefault(name, update)
                control_loss = _finite_real(
                    control_log["total"],
                    name="control total",
                )
                candidate_loss = _finite_real(
                    candidate_log["total"],
                    name="candidate total",
                )
                control_gradient = _finite_real(
                    control_log["gradient_l2_norm"],
                    name="control gradient",
                )
                candidate_gradient = _finite_real(
                    candidate_log["gradient_l2_norm"],
                    name="candidate gradient",
                )
                (
                    control_parameter_digest,
                    control_optimizer_digest,
                    control_optimizer_step,
                ) = gcr_pacre_training_state_summary_fingerprint(
                    control,
                    control_optimizer,
                )
                (
                    candidate_parameter_digest,
                    candidate_optimizer_digest,
                    candidate_optimizer_step,
                ) = gcr_pacre_training_state_summary_fingerprint(
                    candidate,
                    candidate_optimizer,
                )
                raw_trace_rows.append(
                    {
                        "update": update,
                        "epoch": selection.epoch,
                        "step": selection.step,
                        "selection_fingerprint": (
                            selection.selection_fingerprint
                        ),
                        "arms": {
                            GCR_PACRE_CONTROL_ARM: {
                                "loss": control_loss,
                                "gradient_l2_norm": control_gradient,
                                "optimizer_step_counter": (
                                    control_optimizer_step
                                ),
                                "parameter_state_digest": (
                                    control_parameter_digest
                                ),
                                "optimizer_state_digest": (
                                    control_optimizer_digest
                                ),
                                "loss_finite": True,
                                "gradients_finite": True,
                                "parameters_finite": True,
                                "optimizer_state_finite": True,
                            },
                            GCR_PACRE_CANDIDATE_ARM: {
                                "loss": candidate_loss,
                                "gradient_l2_norm": candidate_gradient,
                                "optimizer_step_counter": (
                                    candidate_optimizer_step
                                ),
                                "parameter_state_digest": (
                                    candidate_parameter_digest
                                ),
                                "optimizer_state_digest": (
                                    candidate_optimizer_digest
                                ),
                                "loss_finite": True,
                                "gradients_finite": True,
                                "parameters_finite": True,
                                "optimizer_state_finite": True,
                            },
                        },
                    }
                )
                update_rows.append(
                    GCRPACREPairedUpdateDiagnostic(
                        update=update,
                        selection_fingerprint=(
                            selection.selection_fingerprint
                        ),
                        control_loss=control_loss,
                        candidate_loss=candidate_loss,
                        candidate_minus_control_loss=(
                            candidate_loss - control_loss
                        ),
                        control_gradient_l2_norm=control_gradient,
                        candidate_gradient_l2_norm=candidate_gradient,
                        candidate_minus_control_gradient_l2_norm=(
                            candidate_gradient - control_gradient
                        ),
                    )
                )
                epoch_selections.append(selection.selection_fingerprint)
                if (
                    (update + 1)
                    % GCR_PACRE_BOUNDED_STEPS_PER_EPOCH
                    == 0
                ):
                    epoch = (
                        update // GCR_PACRE_BOUNDED_STEPS_PER_EPOCH
                    )
                    selection_fp = stable_fingerprint(epoch_selections)
                    for arm in (
                        GCR_PACRE_CONTROL_ARM,
                        GCR_PACRE_CANDIDATE_ARM,
                    ):
                        epoch_logs[arm].append(
                            {
                                "epoch": epoch,
                                "completed_updates": update + 1,
                                "objective": (
                                    CoverageStatePairObjective.PMOPE_JOINT.value
                                ),
                                "selection_sequence_fingerprint": (
                                    selection_fp
                                ),
                                **{
                                    f"mean_{name}": value
                                    / GCR_PACRE_BOUNDED_STEPS_PER_EPOCH
                                    for name, value in epoch_sums[arm].items()
                                },
                            }
                        )
                        epoch_sums[arm] = {
                            name: 0.0 for name in epoch_sums[arm]
                        }
                    epoch_selections = []

            control_device_cache.verify_unchanged()
            candidate_device_cache.verify_unchanged()
            control_training = _training_result(
                model=control,
                cache=authorization.control_cache,
                schedule=authorization.schedule,
                device_cache=control_device_cache,
                optimizer_fingerprint=control_optimizer_fp,
                initial_fingerprint=control_initial,
                epoch_logs=epoch_logs[GCR_PACRE_CONTROL_ARM],
                first_nonzero=first_nonzero[GCR_PACRE_CONTROL_ARM],
                device=resolved,
            )
            candidate_training = _training_result(
                model=candidate,
                cache=authorization.candidate_cache,
                schedule=authorization.schedule,
                device_cache=candidate_device_cache,
                optimizer_fingerprint=candidate_optimizer_fp,
                initial_fingerprint=candidate_initial,
                epoch_logs=epoch_logs[GCR_PACRE_CANDIDATE_ARM],
                first_nonzero=first_nonzero[GCR_PACRE_CANDIDATE_ARM],
                device=resolved,
            )
            control_terminal_eval = _evaluate_unchanged(
                authorization,
                control,
                authorization.control_cache,
                arm=GCR_PACRE_CONTROL_ARM,
                checkpoint="terminal",
                forward_mode=GCR_PACRE_NATIVE_MODE,
            )
            candidate_terminal_eval = _evaluate_unchanged(
                authorization,
                candidate,
                authorization.candidate_cache,
                arm=GCR_PACRE_CANDIDATE_ARM,
                checkpoint="terminal",
                forward_mode=GCR_PACRE_NATIVE_MODE,
            )
            candidate_g1_eval = _evaluate_unchanged(
                authorization,
                candidate,
                authorization.candidate_cache,
                arm=GCR_PACRE_CANDIDATE_ARM,
                checkpoint="terminal",
                forward_mode=GCR_PACRE_FORCED_G1_MODE,
            )

            schedule_path = atomic_write_new_canonical_json(
                Path(
                    str(
                        authorization.chain_config.payload[
                            "schedule_artifact_path"
                        ]
                    )
                ),
                authorization.schedule.canonical_payload(),
            )
            schedule_stat = schedule_path.stat()
            schedule_artifact = {
                "path": str(schedule_path.resolve(strict=True)),
                "size_bytes": schedule_stat.st_size,
                "file_sha256": file_sha256(schedule_path),
                "schedule_fingerprint": (
                    authorization.schedule.schedule_fingerprint
                ),
            }
            control_artifact = save_terminal_safetensors_new(
                Path(
                    str(
                        authorization.chain_config.payload[
                            "control_terminal_artifact_path"
                        ]
                    )
                ),
                control,
                metadata={
                    "schema": (
                        "cure-lite-v24-bounded-terminal-safetensors-v1"
                    ),
                    "run": "paired_bounded400",
                    "role": "control",
                    "model_fingerprint": (
                        control_training.final_model_fingerprint
                    ),
                    "arm": GCR_PACRE_CONTROL_ARM,
                    "seed": "42",
                    "epochs": "10",
                    "updates": "400",
                    "checkpoint_policy": "final_only",
                },
            )
            candidate_artifact = save_terminal_safetensors_new(
                Path(
                    str(
                        authorization.chain_config.payload[
                            "candidate_terminal_artifact_path"
                        ]
                    )
                ),
                candidate,
                metadata={
                    "schema": (
                        "cure-lite-v24-bounded-terminal-safetensors-v1"
                    ),
                    "run": "paired_bounded400",
                    "role": "candidate",
                    "model_fingerprint": (
                        candidate_training.final_model_fingerprint
                    ),
                    "arm": GCR_PACRE_CANDIDATE_ARM,
                    "seed": "42",
                    "epochs": "10",
                    "updates": "400",
                    "checkpoint_policy": "final_only",
                },
            )
            control_artifact["model_fingerprint"] = (
                control_training.final_model_fingerprint
            )
            candidate_artifact["model_fingerprint"] = (
                candidate_training.final_model_fingerprint
            )
            trace_payload = build_training_trace_payload(
                stage_id="paired_bounded400",
                authorization_fingerprint=(
                    authorization.authorization_fingerprint
                ),
                schedule=authorization.schedule,
                arm_names=(
                    GCR_PACRE_CONTROL_ARM,
                    GCR_PACRE_CANDIDATE_ARM,
                ),
                terminal_model_fingerprints={
                    GCR_PACRE_CONTROL_ARM: (
                        control_training.final_model_fingerprint
                    ),
                    GCR_PACRE_CANDIDATE_ARM: (
                        candidate_training.final_model_fingerprint
                    ),
                },
                raw_rows=raw_trace_rows,
            )
            trace_artifact = save_training_trace_new(
                Path(authorization.output_directory)
                / "training_trace.json",
                trace_payload,
            )

            control_arm = GCRPACREBoundedArmResult(
                arm=GCR_PACRE_CONTROL_ARM,
                role="control",
                model=control,
                training_result=control_training,
                initial_parameters=control_rows,
                initial_evaluation=control_initial_eval,
                terminal_evaluation=control_terminal_eval,
                forced_g1_evaluation=control_terminal_eval,
                terminal_artifact=control_artifact,
                source_hashes=_source_hashes(),
                cache_instance_id=stable_fingerprint(
                    {
                        "arm": GCR_PACRE_CONTROL_ARM,
                        "cache": authorization.control_cache.cache_fingerprint,
                    }
                ),
                rng_instance_id=stable_fingerprint(
                    {"arm": GCR_PACRE_CONTROL_ARM, "seed": 42}
                ),
                module_instance_id=stable_fingerprint(
                    {
                        "arm": GCR_PACRE_CONTROL_ARM,
                        "initial": control_initial,
                    }
                ),
                optimizer_instance_id=stable_fingerprint(
                    {
                        "arm": GCR_PACRE_CONTROL_ARM,
                        "policy": control_optimizer_fp,
                    }
                ),
                parameter_storage_ids=_logical_storage_ids(
                    arm=GCR_PACRE_CONTROL_ARM,
                    initial_rows=control_rows,
                ),
            )
            candidate_arm = GCRPACREBoundedArmResult(
                arm=GCR_PACRE_CANDIDATE_ARM,
                role="candidate",
                model=candidate,
                training_result=candidate_training,
                initial_parameters=candidate_rows,
                initial_evaluation=candidate_initial_eval,
                terminal_evaluation=candidate_terminal_eval,
                forced_g1_evaluation=candidate_g1_eval,
                terminal_artifact=candidate_artifact,
                source_hashes=_source_hashes(),
                cache_instance_id=stable_fingerprint(
                    {
                        "arm": GCR_PACRE_CANDIDATE_ARM,
                        "cache": (
                            authorization.candidate_cache.cache_fingerprint
                        ),
                    }
                ),
                rng_instance_id=stable_fingerprint(
                    {"arm": GCR_PACRE_CANDIDATE_ARM, "seed": 42}
                ),
                module_instance_id=stable_fingerprint(
                    {
                        "arm": GCR_PACRE_CANDIDATE_ARM,
                        "initial": candidate_initial,
                    }
                ),
                optimizer_instance_id=stable_fingerprint(
                    {
                        "arm": GCR_PACRE_CANDIDATE_ARM,
                        "policy": candidate_optimizer_fp,
                    }
                ),
                parameter_storage_ids=_logical_storage_ids(
                    arm=GCR_PACRE_CANDIDATE_ARM,
                    initial_rows=candidate_rows,
                ),
            )
            del shared_parameter_fingerprint
        authorization._attempt.consume()
        return GCRPACREPairedBoundedResult(
            authorization=authorization,
            run_start_token=run_start_token,
            control=control_arm,
            candidate=candidate_arm,
            updates=tuple(update_rows),
            schedule_artifact=schedule_artifact,
            training_trace_artifact=trace_artifact,
        )
    except BaseException:
        authorization._attempt.fail()
        raise


def _arm_receipt(
    arm: GCRPACREBoundedArmResult,
    *,
    schedule: CoverageStateTrainingSchedule,
    population_fingerprint: str,
    initial_shared_parameter_fingerprint: str,
    neutral_payload_fingerprint: str,
    device: str,
    finite_audit: Mapping[str, object],
) -> dict[str, object]:
    result = arm.training_result
    artifact = arm.terminal_artifact
    terminal_artifact = {
        "path": artifact["path"],
        "size_bytes": artifact["size_bytes"],
        "file_sha256": artifact["file_sha256"],
        "model_fingerprint": artifact["model_fingerprint"],
    }
    return {
        "role": arm.role,
        "seed": GCR_PACRE_BOUNDED_SEED,
        "epochs": GCR_PACRE_BOUNDED_EPOCHS,
        "steps_per_epoch": GCR_PACRE_BOUNDED_STEPS_PER_EPOCH,
        "completed_updates": GCR_PACRE_BOUNDED_UPDATES,
        "training_invocations": 1,
        "from_scratch": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "checkpoint_policy": "final_only",
        "optimizer_state_initial_empty": True,
        "population_fingerprint": population_fingerprint,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "batch_sequence_fingerprint": stable_fingerprint(
            [
                selection.canonical_payload()
                for selection in schedule.selections
            ]
        ),
        "initial_shared_parameter_fingerprint": (
            initial_shared_parameter_fingerprint
        ),
        "PMOPE_fingerprint": stable_fingerprint(
            {
                "objective": CoverageStatePairObjective.PMOPE_JOINT.value,
                "policy": CSLF_PMOPE_POLICY,
            }
        ),
        "Adam_policy_fingerprint": (
            result.optimizer_config_fingerprint
        ),
        "dtype_device_policy_fingerprint": stable_fingerprint(
            {
                "dtype": "torch.float32",
                "device": device,
                "autocast": False,
                "TF32": False,
                "deterministic_algorithms": True,
            }
        ),
        "source_hashes": dict(arm.source_hashes),
        "cache_fingerprint": result.cache_fingerprint,
        "neutral_payload_fingerprint": neutral_payload_fingerprint,
        "cache_instance_id": arm.cache_instance_id,
        "rng_instance_id": arm.rng_instance_id,
        "module_instance_id": arm.module_instance_id,
        "optimizer_instance_id": arm.optimizer_instance_id,
        "parameter_storage_ids": list(arm.parameter_storage_ids),
        "initial_model_fingerprint": result.initial_model_fingerprint,
        "final_model_fingerprint": result.final_model_fingerprint,
        "terminal_artifact": terminal_artifact,
        "finite_audit": dict(finite_audit),
        "metrics": arm.metrics_payload(),
    }


def build_paired_bounded_receipt(
    result: GCRPACREPairedBoundedResult,
) -> dict[str, object]:
    """Build the protocol-verifiable paired bounded evidence receipt."""

    if type(result) is not GCRPACREPairedBoundedResult:
        raise TypeError("result must be exact paired bounded result")
    result.verify_unchanged()
    authorization = result.authorization
    initial_shared_fp = stable_fingerprint(
        list(result.control.initial_parameters)
    )
    if initial_shared_fp != stable_fingerprint(
        list(result.candidate.initial_parameters)
    ):
        raise RuntimeError("paired initial parameter binding changed")
    device = result.control.training_result.execution_device
    if device != result.candidate.training_result.execution_device:
        raise RuntimeError("paired execution devices differ")
    population_fp = authorization.control_cache.cache_fingerprint
    trace_payload = read_canonical_json(
        str(result.training_trace_artifact["path"])
    )
    control_finite = trace_finite_audit(
        trace_payload,
        arm=GCR_PACRE_CONTROL_ARM,
    )
    candidate_finite = trace_finite_audit(
        trace_payload,
        arm=GCR_PACRE_CANDIDATE_ARM,
    )
    body = {
        "schema_version": GCR_PACRE_BOUNDED_RECEIPT_SCHEMA,
        "budget": {
            "epochs": GCR_PACRE_BOUNDED_EPOCHS,
            "steps_per_epoch": GCR_PACRE_BOUNDED_STEPS_PER_EPOCH,
            "updates": GCR_PACRE_BOUNDED_UPDATES,
            "training_invocations_per_arm": 1,
        },
        "prerequisites": {
            "dataset_free_receipt_fingerprint": (
                authorization.dataset_free_receipt_fingerprint
            ),
            "D_R_structural_receipt_fingerprint": (
                authorization.d_r_structural_receipt_fingerprint
            ),
            "OOF4_decision_fingerprint": (
                authorization.oof_decision.decision_fingerprint
            ),
        },
        "access_audit_receipt_fingerprint": (
            authorization.access_audit.receipt_fingerprint
        ),
        "paired_population_fingerprint": population_fp,
        "full_D_R_cache_materialization": (
            authorization.full_d_r_cache_artifact.payload
        ),
        "run_start_artifact": (
            verify_gcr_pacre_bounded_run_start_token(
                authorization,
                result.run_start_token,
            )
        ),
        "schedule_artifact": dict(result.schedule_artifact),
        "training_trace_artifact": dict(
            result.training_trace_artifact
        ),
        "arms": {
            GCR_PACRE_CONTROL_ARM: _arm_receipt(
                result.control,
                schedule=authorization.schedule,
                population_fingerprint=population_fp,
                initial_shared_parameter_fingerprint=initial_shared_fp,
                neutral_payload_fingerprint=(
                    authorization.full_d_r_cache_artifact.neutral_payload_fingerprint
                ),
                device=device,
                finite_audit=control_finite,
            ),
            GCR_PACRE_CANDIDATE_ARM: _arm_receipt(
                result.candidate,
                schedule=authorization.schedule,
                population_fingerprint=population_fp,
                initial_shared_parameter_fingerprint=initial_shared_fp,
                neutral_payload_fingerprint=(
                    authorization.full_d_r_cache_artifact.neutral_payload_fingerprint
                ),
                device=device,
                finite_audit=candidate_finite,
            ),
        },
        "paired_diagnostics": result.diagnostic_payload,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {**body, "receipt_fingerprint": stable_fingerprint(body)}


__all__ = [
    "GCR_PACRE_BOUNDED_AUTHORIZATION_SCHEMA",
    "GCR_PACRE_BOUNDED_EPOCHS",
    "GCR_PACRE_BOUNDED_RECEIPT_SCHEMA",
    "GCR_PACRE_BOUNDED_RESULT_SCHEMA",
    "GCR_PACRE_BOUNDED_SEED",
    "GCR_PACRE_BOUNDED_STEPS_PER_EPOCH",
    "GCR_PACRE_BOUNDED_UPDATES",
    "GCR_PACRE_CANDIDATE_ARM",
    "GCR_PACRE_CONTROL_ARM",
    "GCR_PACRE_FORCED_G1_MODE",
    "GCR_PACRE_NATIVE_MODE",
    "GCRPACREBoundedAuthorization",
    "GCRPACREBoundedArmResult",
    "GCRPACREBoundedEvaluation",
    "GCRPACREBoundedEvaluator",
    "GCRPACREPairedBoundedResult",
    "GCRPACREPairedUpdateDiagnostic",
    "build_paired_bounded_receipt",
    "prepare_gcr_pacre_paired_bounded_authorization",
    "run_gcr_pacre_paired_bounded_400",
]
