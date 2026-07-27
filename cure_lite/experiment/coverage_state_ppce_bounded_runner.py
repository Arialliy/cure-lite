"""Prerequisite-bound PPCE/SORR bounded-400 protocol.

This protocol changes exactly one model coordinate relative to v15B: the
lossy scalar-max occupancy input is replaced by the phase-preserving PPCE
representation.  SORR, both controls, the D_R schedule, zero-level evaluator,
and candidate-only decision rule remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from pathlib import Path

import torch

from ..cache.schema import stable_fingerprint
from ..coverage_state_level_set import CURELiteCoverageStateLevelSet
from ..coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
    CURELitePhasePreservingCoverageStateLevelSet,
    CoverageStatePhasePreservingConfig,
)
from ..coverage_state_precomputed_cache import CoverageStateScalarCache
from ..coverage_state_schedule import CoverageStateTrainingSchedule
from ..coverage_state_sobolev import (
    CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
)
from .coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_SEED,
    CoverageStateBoundedPreflight,
)
from .coverage_state_bounded_runner import (
    COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
    COVERAGE_STATE_BOUNDED_MODEL_WIDTH,
    CoverageStateBoundedRunResult,
    _bounded_result_checks,
    _deterministic_execution,
)
from .coverage_state_dataset_free import (
    COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT,
    CoverageStatePhasePreservingDatasetFreeReceipt,
)
from .coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    CoverageStateMatchedTrainingConfig,
    CoverageStateMatchedTrainingResult,
    CoverageStateRunAuthorization,
    train_matched_coverage_state_phase_preserving_support_oriented_objectives,
)
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
    evaluate_coverage_state_zero_level_checkpoint,
)
from ..train.coverage_state_fused_step import (
    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
    CoverageStatePairObjective,
)


COVERAGE_STATE_PPCE_BOUNDED_AUTHORIZATION_SCHEMA = (
    "cure-lite-cslf-ppce-support-oriented-bounded-run-authorization-v1"
)
COVERAGE_STATE_PPCE_BOUNDED_RESULT_SCHEMA = (
    "cure-lite-cslf-ppce-support-oriented-bounded-run-result-v1"
)
COVERAGE_STATE_V15B_PARENT_COMPLETE_FINGERPRINT = (
    "13cc94f4f5140031fc050ac8d1726e13f9e5e1bbfa8a433bda28783088121f95"
)
COVERAGE_STATE_V15B_PARENT_COMPLETE_SHA256 = (
    "58460fde25d08123231e2ab1ae5767f46ae3e40896b605b9e77c144413f6a896"
)
COVERAGE_STATE_V15B_PARENT_SOURCE_MANIFEST_SHA256 = (
    "d5d5df197eab3bf4423777a4192f7d1bc0781518a54d9e56d37a8dbb48d9da8f"
)
COVERAGE_STATE_V15B_PARENT_SOURCE_ARCHIVE_SHA256 = (
    "e6ced21bef5926cb4fd6b9c79181980614eef3bf0fd7c14ac1cead63815cc069"
)
COVERAGE_STATE_PPCE_BOUNDED_IMPLEMENTATION_PATHS = tuple(
    dict.fromkeys(
        (
            *COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
            "cure_lite/coverage_state_phase_preserving.py",
            "cure_lite/experiment/coverage_state_ppce_bounded_runner.py",
        )
    )
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _current_ppce_implementation_binding(
) -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_PPCE_BOUNDED_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"PPCE bounded implementation file is missing: {relative}"
            )
        result.append((relative, sha256(path.read_bytes()).hexdigest()))
    return tuple(result)


def _ppce_model_config_payload(
    config: CoverageStatePhasePreservingConfig,
) -> dict[str, object]:
    if not isinstance(config, CoverageStatePhasePreservingConfig):
        raise TypeError(
            "config must be CoverageStatePhasePreservingConfig"
        )
    return {
        "feature_channels": config.feature_channels,
        "feature_stride": config.feature_stride,
        "width": config.width,
        "normalization_epsilon_hex": (
            config.normalization_epsilon.hex()
        ),
        "field_amplitude_hex": config.field_amplitude.hex(),
        "initial_field_value_hex": config.initial_field_value.hex(),
        "field_policy": config.field_policy,
        "target_policy": config.target_policy,
        "output_policy": config.output_policy,
        "feature_policy": config.feature_policy,
        "numerical_policy": config.numerical_policy,
        "coverage_policy": config.coverage_policy,
        "phase_occupancy_channels": (
            config.phase_occupancy_channels
        ),
        "expected_parameter_count": config.expected_parameter_count,
        "model_class": (
            "CURELitePhasePreservingCoverageStateLevelSet"
        ),
    }


def expected_coverage_state_ppce_config(
    preflight: CoverageStateBoundedPreflight,
) -> CoverageStatePhasePreservingConfig:
    """Return the only PPCE config permitted by one bounded preflight."""

    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    first = preflight.population.cache.raw_catalog.natural_records[0]
    return CoverageStatePhasePreservingConfig(
        feature_channels=int(first.feature.shape[1]),
        feature_stride=(
            preflight.population.cache.raw_catalog.feature_stride
        ),
        width=COVERAGE_STATE_BOUNDED_MODEL_WIDTH,
    )


@dataclass(frozen=True, eq=False)
class CoverageStatePPCEBoundedRunAuthorization(
    CoverageStateRunAuthorization,
):
    """Bind PPCE/SORR to v15B and one actual dataset-free receipt."""

    preflight: CoverageStateBoundedPreflight
    dataset_free_receipt: CoverageStatePhasePreservingDatasetFreeReceipt
    dataset_free_receipt_fingerprint: str
    implementation_binding: tuple[tuple[str, str], ...]
    implementation_fingerprint: str
    model_config_fingerprint: str
    expected_parameter_count: int
    objective_suite: tuple[str, ...]
    candidate_objective: str
    candidate_objective_policy: str
    coverage_policy: str
    parent_v15b_complete_fingerprint: str
    parent_v15b_complete_sha256: str
    parent_v15b_source_manifest_sha256: str
    parent_v15b_source_archive_sha256: str

    def __post_init__(self) -> None:
        expected_config = expected_coverage_state_ppce_config(
            self.preflight
        )
        expected_suite = tuple(
            value.value
            for value in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
        )
        if (
            not isinstance(self.preflight, CoverageStateBoundedPreflight)
            or not isinstance(
                self.dataset_free_receipt,
                CoverageStatePhasePreservingDatasetFreeReceipt,
            )
            or self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or self.dataset_free_receipt_fingerprint
            != COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT
            or self.implementation_binding
            != _current_ppce_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != stable_fingerprint(
                _ppce_model_config_payload(expected_config)
            )
            or self.expected_parameter_count
            != expected_config.expected_parameter_count
            or self.objective_suite != expected_suite
            or self.candidate_objective
            != CoverageStatePairObjective.SUPPORT_ORIENTED_RESPONSE_JOINT.value
            or self.objective_suite[0] != self.candidate_objective
            or self.candidate_objective_policy
            != CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
            or self.coverage_policy
            != CSLF_PHASE_PRESERVING_COVERAGE_POLICY
            or self.parent_v15b_complete_fingerprint
            != COVERAGE_STATE_V15B_PARENT_COMPLETE_FINGERPRINT
            or self.parent_v15b_complete_sha256
            != COVERAGE_STATE_V15B_PARENT_COMPLETE_SHA256
            or self.parent_v15b_source_manifest_sha256
            != COVERAGE_STATE_V15B_PARENT_SOURCE_MANIFEST_SHA256
            or self.parent_v15b_source_archive_sha256
            != COVERAGE_STATE_V15B_PARENT_SOURCE_ARCHIVE_SHA256
        ):
            raise ValueError("PPCE bounded authorization binding changed")

    @property
    def training_authorized(self) -> bool:
        return (
            self.preflight.training_authorized
            and self.dataset_free_receipt.all_pass
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                COVERAGE_STATE_PPCE_BOUNDED_AUTHORIZATION_SCHEMA
            ),
            "scope": COVERAGE_STATE_BOUNDED_SCOPE,
            "runtime_splits": ["D_R"],
            "preflight_fingerprint": self.preflight.preflight_fingerprint,
            "population_fingerprint": (
                self.preflight.population.population_fingerprint
            ),
            "cache_fingerprint": (
                self.preflight.population.bounded_cache_fingerprint
            ),
            "schedule_fingerprint": (
                self.preflight.schedule.schedule_fingerprint
            ),
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "implementation_binding": dict(self.implementation_binding),
            "implementation_fingerprint": (
                self.implementation_fingerprint
            ),
            "model_config_fingerprint": self.model_config_fingerprint,
            "model_class": (
                "CURELitePhasePreservingCoverageStateLevelSet"
            ),
            "expected_parameter_count": self.expected_parameter_count,
            "coverage_policy": self.coverage_policy,
            "objective_suite": list(self.objective_suite),
            "candidate_objective": self.candidate_objective,
            "candidate_objective_policy": (
                self.candidate_objective_policy
            ),
            "parent_v15b": {
                "complete_fingerprint": (
                    self.parent_v15b_complete_fingerprint
                ),
                "complete_sha256": self.parent_v15b_complete_sha256,
                "source_manifest_sha256": (
                    self.parent_v15b_source_manifest_sha256
                ),
                "source_archive_sha256": (
                    self.parent_v15b_source_archive_sha256
                ),
            },
            "checks": {
                "preflight_passed": self.preflight.training_authorized,
                "dataset_free_gate_passed": (
                    self.dataset_free_receipt.all_pass
                ),
            },
            "training_authorized": self.training_authorized,
            "formal_training_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_claim_supported": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        }

    @cached_property
    def authorization_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        self.preflight.verify_unchanged()
        expected_config = expected_coverage_state_ppce_config(
            self.preflight
        )
        if (
            self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or self.dataset_free_receipt_fingerprint
            != COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT
            or self.implementation_binding
            != _current_ppce_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != stable_fingerprint(
                _ppce_model_config_payload(expected_config)
            )
            or self.expected_parameter_count
            != expected_config.expected_parameter_count
            or stable_fingerprint(self.canonical_payload())
            != self.authorization_fingerprint
        ):
            raise RuntimeError(
                "PPCE bounded authorization changed after creation"
            )

    def verify_model_config(
        self,
        model_config: CoverageStatePhasePreservingConfig,
    ) -> None:
        if (
            not isinstance(
                model_config,
                CoverageStatePhasePreservingConfig,
            )
            or stable_fingerprint(
                _ppce_model_config_payload(model_config)
            )
            != self.model_config_fingerprint
            or model_config.expected_parameter_count
            != self.expected_parameter_count
        ):
            raise PermissionError(
                "PPCE authorization does not permit this model config"
            )

    def verify_for_run(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> None:
        self.verify_unchanged()
        if (
            scope != COVERAGE_STATE_BOUNDED_SCOPE
            or cache.cache_fingerprint
            != self.preflight.population.bounded_cache_fingerprint
            or schedule.schedule_fingerprint
            != self.preflight.schedule.schedule_fingerprint
            or not self.training_authorized
        ):
            raise PermissionError(
                "PPCE authorization does not permit this training run"
            )


def prepare_coverage_state_ppce_bounded_run_authorization(
    preflight: CoverageStateBoundedPreflight,
    dataset_free_receipt: CoverageStatePhasePreservingDatasetFreeReceipt,
) -> CoverageStatePPCEBoundedRunAuthorization:
    """Bind an actual receipt without inventing a dataset-free fingerprint."""

    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    if not isinstance(
        dataset_free_receipt,
        CoverageStatePhasePreservingDatasetFreeReceipt,
    ):
        raise TypeError(
            "dataset_free_receipt must be "
            "CoverageStatePhasePreservingDatasetFreeReceipt"
        )
    preflight.verify_unchanged()
    implementation_binding = _current_ppce_implementation_binding()
    model_config = expected_coverage_state_ppce_config(preflight)
    objective_suite = tuple(
        value.value
        for value in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
    )
    result = CoverageStatePPCEBoundedRunAuthorization(
        preflight=preflight,
        dataset_free_receipt=dataset_free_receipt,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt.receipt_fingerprint
        ),
        implementation_binding=implementation_binding,
        implementation_fingerprint=stable_fingerprint(
            dict(implementation_binding)
        ),
        model_config_fingerprint=stable_fingerprint(
            _ppce_model_config_payload(model_config)
        ),
        expected_parameter_count=model_config.expected_parameter_count,
        objective_suite=objective_suite,
        candidate_objective=objective_suite[0],
        candidate_objective_policy=CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
        coverage_policy=CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
        parent_v15b_complete_fingerprint=(
            COVERAGE_STATE_V15B_PARENT_COMPLETE_FINGERPRINT
        ),
        parent_v15b_complete_sha256=(
            COVERAGE_STATE_V15B_PARENT_COMPLETE_SHA256
        ),
        parent_v15b_source_manifest_sha256=(
            COVERAGE_STATE_V15B_PARENT_SOURCE_MANIFEST_SHA256
        ),
        parent_v15b_source_archive_sha256=(
            COVERAGE_STATE_V15B_PARENT_SOURCE_ARCHIVE_SHA256
        ),
    )
    result.verify_unchanged()
    return result


def _ppce_bounded_result_checks(
    authorization: CoverageStatePPCEBoundedRunAuthorization,
    training: CoverageStateMatchedTrainingResult,
    diagnostics: tuple[
        tuple[str, CoverageStateZeroLevelEvaluationResult],
        ...,
    ],
) -> tuple[tuple[str, bool], ...]:
    """Reuse execution checks, then enforce the one PPCE structure."""

    generic = dict(
        _bounded_result_checks(
            authorization,
            training,
            diagnostics,
        )
    )
    generic.pop("zero_level_gates")
    models = training.models
    expected = authorization.objective_suite
    names = tuple(value.objective for value in training.results)
    diagnostic_by_name = dict(diagnostics)
    candidate = diagnostic_by_name.get(
        authorization.candidate_objective
    )
    controls = expected[1:]
    exact_ppce_models = (
        len(models) == 3
        and all(
            type(model)
            is CURELitePhasePreservingCoverageStateLevelSet
            for _, model in models
        )
    )
    generic["authorized_model_config"] = all(
        type(model)
        is CURELitePhasePreservingCoverageStateLevelSet
        and stable_fingerprint(
            _ppce_model_config_payload(model.config)
        )
        == authorization.model_config_fingerprint
        for _, model in models
    )
    generic.update(
        {
            "ppce_objective_suite": names == expected,
            "candidate_original_zero_level_gates": (
                candidate is not None
                and candidate.bounded_gate_passed
            ),
            "control_diagnostics_complete": (
                tuple(name for name, _ in diagnostics) == expected
                and all(name in diagnostic_by_name for name in controls)
                and len(controls) == 2
            ),
            "all_models_exact_ppce_class": exact_ppce_models,
            "all_models_same_ppce_config": (
                exact_ppce_models
                and len(
                    {
                        stable_fingerprint(
                            _ppce_model_config_payload(model.config)
                        )
                        for _, model in models
                    }
                )
                == 1
            ),
            "all_models_expected_parameter_count": (
                len(models) == 3
                and all(
                    sum(
                        parameter.numel()
                        for parameter in model.parameters()
                    )
                    == authorization.expected_parameter_count
                    for _, model in models
                )
            ),
            "candidate_policy_bound": (
                authorization.candidate_objective_policy
                == CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
            ),
            "coverage_policy_bound": (
                authorization.coverage_policy
                == CSLF_PHASE_PRESERVING_COVERAGE_POLICY
            ),
            "phase_preserving_zero_evaluation": all(
                getattr(
                    getattr(diagnostic, "config", None),
                    "input_representation",
                    None,
                )
                == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                for _, diagnostic in diagnostics
            ),
            "parent_v15b_evidence_bound": (
                authorization.parent_v15b_complete_fingerprint
                == COVERAGE_STATE_V15B_PARENT_COMPLETE_FINGERPRINT
                and authorization.parent_v15b_complete_sha256
                == COVERAGE_STATE_V15B_PARENT_COMPLETE_SHA256
                and authorization.parent_v15b_source_manifest_sha256
                == COVERAGE_STATE_V15B_PARENT_SOURCE_MANIFEST_SHA256
                and authorization.parent_v15b_source_archive_sha256
                == COVERAGE_STATE_V15B_PARENT_SOURCE_ARCHIVE_SHA256
            ),
        }
    )
    return tuple(sorted(generic.items()))


@dataclass(frozen=True, eq=False)
class CoverageStatePPCEBoundedRunResult(CoverageStateBoundedRunResult):
    """PPCE candidate result with complete, non-gating controls."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(
            self.authorization,
            CoverageStatePPCEBoundedRunAuthorization,
        ):
            raise ValueError("PPCE result requires PPCE authorization")
        if tuple(
            value.objective for value in self.training.results
        ) != self.authorization.objective_suite:
            raise ValueError("PPCE result objective suite changed")

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        self.training.verify_unchanged()
        if self.checks != _ppce_bounded_result_checks(
            self.authorization,
            self.training,
            self.diagnostics,
        ):
            raise RuntimeError("PPCE bounded result checks changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        diagnostic_by_name = dict(self.diagnostics)
        controls = self.authorization.objective_suite[1:]
        return {
            "schema_version": COVERAGE_STATE_PPCE_BOUNDED_RESULT_SCHEMA,
            "runtime_splits": ["D_R"],
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "parent_v15b": {
                "complete_fingerprint": (
                    self.authorization.parent_v15b_complete_fingerprint
                ),
                "complete_sha256": (
                    self.authorization.parent_v15b_complete_sha256
                ),
                "source_manifest_sha256": (
                    self.authorization
                    .parent_v15b_source_manifest_sha256
                ),
                "source_archive_sha256": (
                    self.authorization
                    .parent_v15b_source_archive_sha256
                ),
            },
            "model_class": (
                "CURELitePhasePreservingCoverageStateLevelSet"
            ),
            "model_config_fingerprint": (
                self.authorization.model_config_fingerprint
            ),
            "expected_parameter_count": (
                self.authorization.expected_parameter_count
            ),
            "candidate_objective": (
                self.authorization.candidate_objective
            ),
            "candidate_diagnostic": diagnostic_by_name[
                self.authorization.candidate_objective
            ].canonical_payload(),
            "control_diagnostics": {
                name: diagnostic_by_name[name].canonical_payload()
                for name in controls
            },
            "training": self.training.canonical_payload(),
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "bounded_gate_passed": self.bounded_gate_passed,
            "candidate_qualification_uses_original_gates_only": True,
            "control_outcomes_are_not_candidate_gates": True,
            "formal_training_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_claim_supported": False,
        }


def run_coverage_state_ppce_support_oriented_bounded_400(
    authorization: CoverageStatePPCEBoundedRunAuthorization,
    model_config: CoverageStatePhasePreservingConfig,
    *,
    device: torch.device | str,
) -> CoverageStatePPCEBoundedRunResult:
    """Run the fixed PPCE/SORR suite; caller owns external artifact policy."""

    if not isinstance(
        authorization,
        CoverageStatePPCEBoundedRunAuthorization,
    ):
        raise TypeError("authorization must be PPCE bounded authorization")
    authorization.verify_model_config(model_config)
    preflight = authorization.preflight
    authorization.verify_for_run(
        cache=preflight.population.cache,
        schedule=preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
    )
    with _deterministic_execution(device):
        training = (
            train_matched_coverage_state_phase_preserving_support_oriented_objectives(
                model_config,
                preflight.population.cache,
                preflight.schedule,
                config=CoverageStateMatchedTrainingConfig(
                    seed=COVERAGE_STATE_BOUNDED_SEED
                ),
                device=device,
                authorization=authorization,
            )
        )
        if any(
            type(model)
            is not CURELitePhasePreservingCoverageStateLevelSet
            for _, model in training.models
        ):
            raise RuntimeError(
                "matched PPCE training returned a different model class"
            )
        diagnostics = tuple(
            (
                name,
                evaluate_coverage_state_zero_level_checkpoint(
                    model.eval(),
                    preflight.population.cache,
                    device=device,
                    config=CoverageStateZeroLevelEvaluationConfig(
                        input_representation=(
                            COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                        )
                    ),
                ),
            )
            for name, model in training.models
        )
    result = CoverageStatePPCEBoundedRunResult(
        authorization=authorization,
        training=training,
        diagnostics=diagnostics,
        checks=_ppce_bounded_result_checks(
            authorization,
            training,
            diagnostics,
        ),
    )
    result.verify_unchanged()
    return result


__all__ = [
    "COVERAGE_STATE_PPCE_BOUNDED_AUTHORIZATION_SCHEMA",
    "COVERAGE_STATE_PPCE_BOUNDED_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_PPCE_BOUNDED_RESULT_SCHEMA",
    "COVERAGE_STATE_V15B_PARENT_COMPLETE_FINGERPRINT",
    "COVERAGE_STATE_V15B_PARENT_COMPLETE_SHA256",
    "COVERAGE_STATE_V15B_PARENT_SOURCE_ARCHIVE_SHA256",
    "COVERAGE_STATE_V15B_PARENT_SOURCE_MANIFEST_SHA256",
    "CoverageStatePPCEBoundedRunAuthorization",
    "CoverageStatePPCEBoundedRunResult",
    "expected_coverage_state_ppce_config",
    "prepare_coverage_state_ppce_bounded_run_authorization",
    "run_coverage_state_ppce_support_oriented_bounded_400",
]
