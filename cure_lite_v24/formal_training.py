"""One-shot, token-authorized Formal800 training for GCR-PACRE v24.

Only two executions exist:

* seed 42, role ``primary``;
* seed 43, role ``training_integrity_only``.

Each execution is exactly 800 epochs by 40 steps, starts from a fresh model
and empty Adam state through :mod:`cure_lite_v24.training`, and is D_R-only.
The seed-43 result has no selection effect and can never replace or authorize
evaluation of the seed-42 primary model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateTrainingSchedule,
    coverage_state_schedule_exposure_report,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_FORMAL_SCOPE,
    CoverageStateRunAuthorization,
    coverage_state_model_fingerprint,
)
from tools.gcr_pacre_v24_protocol import (
    VerifiedAccessAudit,
    VerifiedBoundedDecision,
    VerifiedOOFDecision,
    require_verified_access_audit,
    require_verified_bounded_decision,
    require_verified_oof_decision,
)

from .factory import (
    GCR_PACRE_FORMAL_FEATURE_CHANNELS,
    GCR_PACRE_FORMAL_FEATURE_STRIDE,
    GCR_PACRE_FORMAL_PARAMETER_COUNT,
    GCR_PACRE_FORMAL_WIDTH,
)
from .formal_cache_artifacts import (
    VerifiedFormalCacheArtifact,
    require_verified_formal_cache_artifact,
    verify_formal_cache_artifact,
)
from .formal_run_start import (
    GCRPACREFormalRunStartToken,
    VerifiedGCRPACREFormalChainConfig,
    require_verified_gcr_pacre_formal_chain_config,
    verify_gcr_pacre_formal_chain_authorization_binding,
    verify_gcr_pacre_formal_run_start_token,
)
from .gcr_pacre import CoverageStateGCRPACREConfig
from .source_closure import gcr_pacre_v24_source_closure_hashes
from .training import (
    GCR_PACRE_ROLE_PRIMARY,
    GCR_PACRE_ROLE_TRAINING_INTEGRITY,
    GCRPACRETrainingBundle,
    train_gcr_pacre_pmope_candidate,
)
from .training_trace import (
    build_training_trace_payload,
    save_training_trace_new,
)


GCR_PACRE_FORMAL_AUTHORIZATION_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-formal800-authorization-v2"
)
GCR_PACRE_FORMAL_RESULT_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-formal800-result-v2"
)
GCR_PACRE_FORMAL_TERMINAL_EVALUATION_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-formal800-D_R-terminal-evaluation-v1"
)
GCR_PACRE_FORMAL_EPOCHS: Final = 800
GCR_PACRE_FORMAL_STEPS_PER_EPOCH: Final = 40
GCR_PACRE_FORMAL_UPDATES: Final = 32_000
GCR_PACRE_FORMAL_SEED_ROLES: Final = (
    (42, GCR_PACRE_ROLE_PRIMARY),
    (43, GCR_PACRE_ROLE_TRAINING_INTEGRITY),
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_hashes() -> tuple[tuple[str, str], ...]:
    return gcr_pacre_v24_source_closure_hashes()


def _formal_config() -> CoverageStateGCRPACREConfig:
    config = CoverageStateGCRPACREConfig(
        feature_channels=GCR_PACRE_FORMAL_FEATURE_CHANNELS,
        feature_stride=GCR_PACRE_FORMAL_FEATURE_STRIDE,
        width=GCR_PACRE_FORMAL_WIDTH,
    )
    if config.expected_parameter_count != GCR_PACRE_FORMAL_PARAMETER_COUNT:
        raise AssertionError("Formal configuration parameter count changed")
    return config


def _finite_tree(value: object, *, path: str = "metrics") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite value")
        return
    if isinstance(value, Mapping):
        if not value or any(
            not isinstance(key, str) or not key for key in value
        ):
            raise ValueError(f"{path} must be a non-empty text-key mapping")
        for key, item in value.items():
            _finite_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_tree(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


def _full_d_r_materialization(
    bounded: VerifiedBoundedDecision,
) -> tuple[str, str, str]:
    """Read the content-bound materialization identity from a verified token."""

    raw = bounded.payload.get("full_D_R_cache_binding")
    if not isinstance(raw, Mapping) or set(raw) != {
        "semantic_cache_fingerprint",
        "neutral_payload_fingerprint",
        "materialization_receipt_fingerprint",
    }:
        raise PermissionError(
            "bounded token lacks the verified full-D_R materialization"
        )
    semantic = raw.get("semantic_cache_fingerprint")
    neutral = raw.get("neutral_payload_fingerprint")
    receipt = raw.get("materialization_receipt_fingerprint")
    if not all(_is_sha256(value) for value in (semantic, neutral, receipt)):
        raise ValueError("full-D_R materialization digest is malformed")
    if (
        semantic != bounded.full_d_r_semantic_cache_fingerprint
        or neutral != bounded.full_d_r_neutral_payload_fingerprint
        or receipt
        != bounded.full_d_r_materialization_receipt_fingerprint
    ):
        raise PermissionError("bounded token/cache payload binding changed")
    return str(semantic), str(neutral), str(receipt)


class GCRPACREFormalTerminalEvaluator(ABC):
    """Frozen evaluator for the terminal D_R-only integrity receipt."""

    @property
    @abstractmethod
    def evaluator_fingerprint(self) -> str:
        """Return the evaluator/source/aggregation fingerprint."""

    @abstractmethod
    def evaluate_terminal_d_r(
        self,
        model: torch.nn.Module,
        cache: CoverageStateScalarCache,
        *,
        seed: int,
        role: str,
    ) -> Mapping[str, object]:
        """Return finite D_R-only terminal metrics without mutation."""


@dataclass(frozen=True)
class GCRPACREFormalTerminalEvaluation:
    seed: int
    role: str
    evaluator_fingerprint: str
    model_fingerprint: str
    metrics_json: str

    def __post_init__(self) -> None:
        if (
            (self.seed, self.role) not in GCR_PACRE_FORMAL_SEED_ROLES
            or not _is_sha256(self.evaluator_fingerprint)
            or not _is_sha256(self.model_fingerprint)
        ):
            raise ValueError("Formal terminal evaluation identity changed")
        try:
            metrics = json.loads(self.metrics_json)
        except json.JSONDecodeError as error:
            raise ValueError("Formal terminal metrics JSON is invalid") from error
        if (
            not isinstance(metrics, dict)
            or not metrics
            or canonical_json(metrics) != self.metrics_json
        ):
            raise ValueError("Formal terminal metrics are not canonical")
        _finite_tree(metrics)

    @property
    def metrics(self) -> dict[str, object]:
        value = json.loads(self.metrics_json)
        if not isinstance(value, dict):
            raise AssertionError("validated metrics changed")
        return value

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                GCR_PACRE_FORMAL_TERMINAL_EVALUATION_SCHEMA
            ),
            "seed": self.seed,
            "role": self.role,
            "split": "D_R",
            "evaluator_fingerprint": self.evaluator_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "metrics": self.metrics,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }

    @property
    def evaluation_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


class _FormalAttempt:
    def __init__(self) -> None:
        self.lock = Lock()
        self.state = "available"
        self.verify_calls = 0

    def claim(self) -> None:
        with self.lock:
            if self.state != "available":
                raise PermissionError("Formal authorization is no longer available")
            self.state = "running"

    def verified(self) -> None:
        with self.lock:
            if self.state != "running" or self.verify_calls != 0:
                raise PermissionError("Formal training verification is not one-shot")
            self.verify_calls = 1

    def consume(self) -> None:
        with self.lock:
            if self.state != "running" or self.verify_calls != 1:
                raise PermissionError("Formal authorization was not verified once")
            self.state = "consumed"

    def fail(self) -> None:
        with self.lock:
            if self.state in {"running", "consumed"}:
                self.state = "failed"


@dataclass(frozen=True, eq=False)
class GCRPACREFormalAuthorization(CoverageStateRunAuthorization):
    """Verifier-token-bound authorization for one exact Formal800 run."""

    seed: int
    role: str
    oof_decision: VerifiedOOFDecision
    bounded_decision: VerifiedBoundedDecision
    access_audit: VerifiedAccessAudit
    cache_artifact: VerifiedFormalCacheArtifact
    chain_config: VerifiedGCRPACREFormalChainConfig
    dataset_free_receipt_fingerprint: str
    d_r_structural_receipt_fingerprint: str
    cache: CoverageStateScalarCache
    schedule: CoverageStateTrainingSchedule
    model_config: CoverageStateGCRPACREConfig
    evaluator: GCRPACREFormalTerminalEvaluator
    evaluator_fingerprint: str
    source_hashes: tuple[tuple[str, str], ...]
    _attempt: _FormalAttempt = field(
        default_factory=_FormalAttempt,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.verify_unchanged()

    @property
    def stage_id(self) -> str:
        return f"formal800_seed{self.seed}_{self.role}"

    @property
    def chain_run_binding(self) -> dict[str, object]:
        return self.chain_config.run(self.seed)

    @property
    def requested_device(self) -> str:
        value = self.chain_run_binding["requested_device"]
        if not isinstance(value, str):
            raise AssertionError("verified Formal device binding changed")
        return value

    @property
    def output_directory(self) -> str:
        value = self.chain_run_binding["output_directory"]
        if not isinstance(value, str):
            raise AssertionError("verified Formal output binding changed")
        return value

    def verify_unchanged(self) -> None:
        if (self.seed, self.role) not in GCR_PACRE_FORMAL_SEED_ROLES:
            raise ValueError("Formal seed/role binding changed")
        oof = require_verified_oof_decision(self.oof_decision)
        bounded = require_verified_bounded_decision(
            self.bounded_decision
        )
        access = require_verified_access_audit(self.access_audit)
        assert isinstance(oof, VerifiedOOFDecision)
        assert isinstance(bounded, VerifiedBoundedDecision)
        assert isinstance(access, VerifiedAccessAudit)
        cache_artifact = require_verified_formal_cache_artifact(
            self.cache_artifact
        )
        chain = require_verified_gcr_pacre_formal_chain_config(
            self.chain_config
        )
        if (
            oof.payload.get("gate_passed") is not True
            or bounded.payload.get("gate_passed") is not True
            or bounded.oof_decision_fingerprint
            != oof.decision_fingerprint
            or access.stage_id != self.stage_id
            or access.allowed_splits != ("D_R",)
            or not _is_sha256(self.dataset_free_receipt_fingerprint)
            or not _is_sha256(self.d_r_structural_receipt_fingerprint)
        ):
            raise PermissionError("Formal predecessor tokens are incoherent")
        semantic_fp, neutral_fp, _ = _full_d_r_materialization(bounded)
        reverified_cache = verify_formal_cache_artifact(
            cache_artifact.path,
            cache_id=cache_artifact.cache_id,
            expected_semantic_cache_fingerprint=(
                cache_artifact.semantic_cache_fingerprint
            ),
            expected_neutral_payload_fingerprint=(
                cache_artifact.neutral_payload_fingerprint
            ),
        )
        if (
            reverified_cache.receipt_fingerprint
            != cache_artifact.receipt_fingerprint
            or
            cache_artifact.semantic_cache_fingerprint != semantic_fp
            or cache_artifact.neutral_payload_fingerprint != neutral_fp
        ):
            raise PermissionError(
                "Formal cache differs from verified bounded full-D_R materialization"
            )
        run_binding = verify_gcr_pacre_formal_chain_authorization_binding(
            chain,
            seed=self.seed,
            role=self.role,
            oof_decision=oof,
            bounded_decision=bounded,
            access_audit=access,
            cache_artifact=cache_artifact,
            dataset_free_receipt_fingerprint=(
                self.dataset_free_receipt_fingerprint
            ),
            d_r_structural_receipt_fingerprint=(
                self.d_r_structural_receipt_fingerprint
            ),
        )
        if (
            run_binding != self.chain_run_binding
            or not isinstance(self.requested_device, str)
            or not self.requested_device
            or not Path(self.output_directory).is_absolute()
        ):
            raise PermissionError("Formal frozen execution binding changed")
        if (
            type(self.cache) is not CoverageStateScalarCache
            or self.cache.cache_fingerprint != semantic_fp
        ):
            raise ValueError("Formal in-memory cache semantic identity changed")
        self.cache.verify_unchanged()
        if (
            type(self.schedule) is not CoverageStateTrainingSchedule
            or self.schedule.cache_fingerprint != semantic_fp
            or (
                self.schedule.config.seed,
                self.schedule.config.epochs,
                self.schedule.config.steps_per_epoch,
            )
            != (
                self.seed,
                GCR_PACRE_FORMAL_EPOCHS,
                GCR_PACRE_FORMAL_STEPS_PER_EPOCH,
            )
        ):
            raise PermissionError("Formal schedule is not exact seed/800x40")
        coverage_state_schedule_exposure_report(self.cache, self.schedule)
        if (
            type(self.model_config) is not CoverageStateGCRPACREConfig
            or self.model_config != _formal_config()
            or self.model_config.expected_parameter_count
            != GCR_PACRE_FORMAL_PARAMETER_COUNT
        ):
            raise ValueError("Formal model is not fixed 64/4/32/64064")
        if (
            not isinstance(
                self.evaluator,
                GCRPACREFormalTerminalEvaluator,
            )
            or not _is_sha256(self.evaluator_fingerprint)
            or self.evaluator.evaluator_fingerprint
            != self.evaluator_fingerprint
        ):
            raise PermissionError("Formal terminal evaluator binding changed")
        if (
            self.source_hashes != _source_hashes()
            or dict(self.source_hashes)
            != chain.payload.get("source_hashes")
        ):
            raise RuntimeError("Formal source closure changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        semantic_fp, neutral_fp, materialization_fp = (
            _full_d_r_materialization(self.bounded_decision)
        )
        return {
            "schema_version": GCR_PACRE_FORMAL_AUTHORIZATION_SCHEMA,
            "seed": self.seed,
            "role": self.role,
            "evaluation_role": self.role,
            "predecessor_tokens": {
                "OOF4_decision_fingerprint": (
                    self.oof_decision.decision_fingerprint
                ),
                "paired_bounded400_decision_fingerprint": (
                    self.bounded_decision.decision_fingerprint
                ),
                "access_audit_receipt_fingerprint": (
                    self.access_audit.receipt_fingerprint
                ),
                "dataset_free_receipt_fingerprint": (
                    self.dataset_free_receipt_fingerprint
                ),
                "D_R_structural_receipt_fingerprint": (
                    self.d_r_structural_receipt_fingerprint
                ),
            },
            "full_D_R_materialization": {
                "semantic_cache_fingerprint": semantic_fp,
                "neutral_payload_fingerprint": neutral_fp,
                "materialization_receipt_fingerprint": materialization_fp,
                "cache_artifact_receipt_fingerprint": (
                    self.cache_artifact.receipt_fingerprint
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
            "execution_binding": self.chain_run_binding,
            "schedule_fingerprint": self.schedule.schedule_fingerprint,
            "budget": {
                "epochs": GCR_PACRE_FORMAL_EPOCHS,
                "steps_per_epoch": GCR_PACRE_FORMAL_STEPS_PER_EPOCH,
                "updates": GCR_PACRE_FORMAL_UPDATES,
                "training_invocations": 1,
            },
            "model": {
                "feature_channels": GCR_PACRE_FORMAL_FEATURE_CHANNELS,
                "feature_stride": GCR_PACRE_FORMAL_FEATURE_STRIDE,
                "width": GCR_PACRE_FORMAL_WIDTH,
                "parameter_count": GCR_PACRE_FORMAL_PARAMETER_COUNT,
            },
            "evaluator_fingerprint": self.evaluator_fingerprint,
            "source_hashes": dict(self.source_hashes),
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "checkpoint_policy": "final_only",
            "selection_effect": (
                "predeclared_primary"
                if self.role == GCR_PACRE_ROLE_PRIMARY
                else "none"
            ),
            "may_replace_seed42_primary": False,
            "D_V_execution_authorized": False,
            "D_T_execution_authorized": False,
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
        self.verify_unchanged()
        if (
            self._attempt.state != "running"
            or cache is not self.cache
            or schedule is not self.schedule
            or scope != COVERAGE_STATE_FORMAL_SCOPE
        ):
            raise PermissionError("Formal training binding changed")
        self._attempt.verified()


def prepare_gcr_pacre_formal_authorization(
    *,
    seed: int,
    role: str,
    oof_decision: VerifiedOOFDecision,
    bounded_decision: VerifiedBoundedDecision,
    access_audit: VerifiedAccessAudit,
    cache_artifact: VerifiedFormalCacheArtifact,
    chain_config: VerifiedGCRPACREFormalChainConfig,
    dataset_free_receipt_fingerprint: str,
    d_r_structural_receipt_fingerprint: str,
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
    evaluator: GCRPACREFormalTerminalEvaluator,
) -> GCRPACREFormalAuthorization:
    """Prepare one exact token-only seed42 or seed43 authorization."""

    return GCRPACREFormalAuthorization(
        seed=seed,
        role=role,
        oof_decision=oof_decision,
        bounded_decision=bounded_decision,
        access_audit=access_audit,
        cache_artifact=cache_artifact,
        chain_config=chain_config,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt_fingerprint
        ),
        d_r_structural_receipt_fingerprint=(
            d_r_structural_receipt_fingerprint
        ),
        cache=cache,
        schedule=schedule,
        model_config=_formal_config(),
        evaluator=evaluator,
        evaluator_fingerprint=evaluator.evaluator_fingerprint,
        source_hashes=_source_hashes(),
    )


@dataclass(frozen=True, eq=False)
class GCRPACREFormalRunResult:
    """One terminal Formal model plus immutable training/evaluation ledgers."""

    authorization: GCRPACREFormalAuthorization
    run_start_token: GCRPACREFormalRunStartToken
    training_bundle: GCRPACRETrainingBundle
    terminal_evaluation: GCRPACREFormalTerminalEvaluation
    training_trace_artifact: Mapping[str, object]
    source_hashes_after: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        self.verify_unchanged()

    @property
    def seed(self) -> int:
        return self.authorization.seed

    @property
    def role(self) -> str:
        return self.authorization.role

    @property
    def model(self):
        return self.training_bundle.model

    @property
    def training_result(self):
        return self.training_bundle.training_result

    @property
    def training_receipt(self):
        return self.training_bundle.receipt

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        verify_gcr_pacre_formal_run_start_token(
            self.authorization,
            self.run_start_token,
        )
        self.training_bundle.verify_unchanged()
        receipt = self.training_receipt
        trace_path = Path(str(self.training_trace_artifact.get("path")))
        if (
            self.authorization._attempt.state != "consumed"
            or self.source_hashes_after != self.authorization.source_hashes
            or self.source_hashes_after != _source_hashes()
            or (receipt.seed, receipt.role) != (self.seed, self.role)
            or receipt.epochs != GCR_PACRE_FORMAL_EPOCHS
            or receipt.steps_per_epoch
            != GCR_PACRE_FORMAL_STEPS_PER_EPOCH
            or receipt.completed_updates != GCR_PACRE_FORMAL_UPDATES
            or receipt.training_invocations != 1
            or receipt.from_scratch is not True
            or receipt.resume_allowed is not False
            or receipt.automatic_retry_allowed is not False
            or receipt.checkpoint_policy != "final_only"
            or receipt.D_V_payload_accessed is not False
            or receipt.D_T_payload_accessed is not False
            or self.terminal_evaluation.seed != self.seed
            or self.terminal_evaluation.role != self.role
            or self.terminal_evaluation.model_fingerprint
            != receipt.final_model_fingerprint
            or receipt.final_model_fingerprint
            != coverage_state_model_fingerprint(self.model)
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
            raise RuntimeError("Formal terminal result binding changed")
        if self.seed == 43 and (
            receipt.selection_effect != "none"
            or receipt.may_replace_seed42_primary is not False
            or receipt.eligible_for_future_D_V_authorization_after_all_external_prerequisites
            is not False
            or receipt.eligible_for_future_D_T_authorization_after_all_external_prerequisites
            is not False
        ):
            raise PermissionError("seed43 evaluation/selection firewall changed")

    @property
    def result_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(
            {
                "schema_version": GCR_PACRE_FORMAL_RESULT_SCHEMA,
                "authorization_fingerprint": (
                    self.authorization.authorization_fingerprint
                ),
                "run_start_marker_fingerprint": (
                    self.run_start_token.marker_fingerprint
                ),
                "training_bundle_fingerprint": (
                    self.training_bundle.bundle_fingerprint
                ),
                "terminal_evaluation_fingerprint": (
                    self.terminal_evaluation.evaluation_fingerprint
                ),
                "training_trace_fingerprint": (
                    self.training_trace_artifact["trace_fingerprint"]
                ),
                "source_hashes": dict(self.source_hashes_after),
            }
        )


def run_gcr_pacre_formal_800(
    authorization: GCRPACREFormalAuthorization,
    *,
    run_start_token: GCRPACREFormalRunStartToken,
    device: torch.device | str = "cpu",
) -> GCRPACREFormalRunResult:
    """Consume one authorization and execute one exact Formal800 training."""

    if type(authorization) is not GCRPACREFormalAuthorization:
        raise TypeError("authorization must be exact Formal authorization")
    authorization._attempt.claim()
    try:
        authorization.verify_unchanged()
        verify_gcr_pacre_formal_run_start_token(
            authorization,
            run_start_token,
        )
        if str(torch.device(device)) != authorization.requested_device:
            raise PermissionError("Formal device differs from chain config")
        raw_trace_rows: list[dict[str, object]] = []

        def capture_trace(raw: Mapping[str, object]) -> None:
            identity_fields = {
                name: raw[name]
                for name in (
                    "update",
                    "epoch",
                    "step",
                    "selection_fingerprint",
                )
            }
            arm_fields = {
                name: raw[name]
                for name in (
                    "loss",
                    "gradient_l2_norm",
                    "optimizer_step_counter",
                    "parameter_state_digest",
                    "optimizer_state_digest",
                    "loss_finite",
                    "gradients_finite",
                    "parameters_finite",
                    "optimizer_state_finite",
                )
            }
            raw_trace_rows.append(
                {
                    **identity_fields,
                    "arms": {authorization.role: arm_fields},
                }
            )

        bundle = train_gcr_pacre_pmope_candidate(
            authorization.model_config,
            authorization.cache,
            authorization.schedule,
            role=authorization.role,
            seed=authorization.seed,
            authorization=authorization,
            device=device,
            update_callback=capture_trace,
        )
        bundle.verify_unchanged()
        before = coverage_state_model_fingerprint(bundle.model)
        raw_metrics = authorization.evaluator.evaluate_terminal_d_r(
            bundle.model,
            authorization.cache,
            seed=authorization.seed,
            role=authorization.role,
        )
        after = coverage_state_model_fingerprint(bundle.model)
        if (
            before != after
            or authorization.evaluator.evaluator_fingerprint
            != authorization.evaluator_fingerprint
            or not isinstance(raw_metrics, Mapping)
        ):
            raise RuntimeError("Formal terminal evaluator mutated its binding")
        metrics = dict(raw_metrics)
        _finite_tree(metrics)
        terminal = GCRPACREFormalTerminalEvaluation(
            seed=authorization.seed,
            role=authorization.role,
            evaluator_fingerprint=authorization.evaluator_fingerprint,
            model_fingerprint=after,
            metrics_json=canonical_json(metrics),
        )
        trace_payload = build_training_trace_payload(
            stage_id=authorization.stage_id,
            authorization_fingerprint=(
                authorization.authorization_fingerprint
            ),
            schedule=authorization.schedule,
            arm_names=(authorization.role,),
            terminal_model_fingerprints={authorization.role: after},
            raw_rows=raw_trace_rows,
        )
        trace_artifact = save_training_trace_new(
            Path(authorization.output_directory) / "training_trace.json",
            trace_payload,
        )
        source_after = _source_hashes()
        if source_after != authorization.source_hashes:
            raise RuntimeError("Formal source bytes changed during training")
        authorization._attempt.consume()
        return GCRPACREFormalRunResult(
            authorization=authorization,
            run_start_token=run_start_token,
            training_bundle=bundle,
            terminal_evaluation=terminal,
            training_trace_artifact=trace_artifact,
            source_hashes_after=source_after,
        )
    except BaseException:
        authorization._attempt.fail()
        raise


__all__ = [
    "GCR_PACRE_FORMAL_AUTHORIZATION_SCHEMA",
    "GCR_PACRE_FORMAL_EPOCHS",
    "GCR_PACRE_FORMAL_RESULT_SCHEMA",
    "GCR_PACRE_FORMAL_SEED_ROLES",
    "GCR_PACRE_FORMAL_STEPS_PER_EPOCH",
    "GCR_PACRE_FORMAL_TERMINAL_EVALUATION_SCHEMA",
    "GCR_PACRE_FORMAL_UPDATES",
    "GCRPACREFormalAuthorization",
    "GCRPACREFormalRunResult",
    "GCRPACREFormalTerminalEvaluation",
    "GCRPACREFormalTerminalEvaluator",
    "prepare_gcr_pacre_formal_authorization",
    "run_gcr_pacre_formal_800",
]
