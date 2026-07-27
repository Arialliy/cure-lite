"""One prerequisite-bound D_R bounded-400 run for scalar CSLF."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from typing import Iterator

import torch

from ..cache.schema import stable_fingerprint
from ..coverage_state_level_set import CoverageStateLevelSetConfig
from ..coverage_state_sobolev import (
    CSLF_COMPLETION_ROOTED_RESPONSE_POLICY,
    CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
)
from ..coverage_state_precomputed_cache import CoverageStateScalarCache
from ..coverage_state_schedule import CoverageStateTrainingSchedule
from ..frozen_base import module_state_fingerprint
from .coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_SEED,
    COVERAGE_STATE_BOUNDED_UPDATES,
    CoverageStateBoundedPreflight,
)
from .coverage_state_dataset_free import (
    CoverageStateDatasetFreeReceipt,
    CoverageStateSupportOrientedDatasetFreeReceipt,
)
from .coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    CoverageStateMatchedTrainingConfig,
    CoverageStateMatchedTrainingResult,
    CoverageStateRunAuthorization,
    train_matched_coverage_state_completion_rooted_objectives,
    train_matched_coverage_state_objectives,
    train_matched_coverage_state_support_oriented_objectives,
)
from ..train.coverage_state_fused_step import (
    COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES,
    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
    CoverageStatePairObjective,
)
from .coverage_state_zero_level_evaluation import (
    CoverageStateZeroLevelEvaluationResult,
    evaluate_coverage_state_zero_level_checkpoint,
)


COVERAGE_STATE_BOUNDED_AUTHORIZATION_SCHEMA = (
    "cure-lite-cslf-bounded-run-authorization-v2"
)
COVERAGE_STATE_BOUNDED_RESULT_SCHEMA = (
    "cure-lite-cslf-bounded-run-result-v2"
)
COVERAGE_STATE_COMPLETION_ROOTED_BOUNDED_AUTHORIZATION_SCHEMA = (
    "cure-lite-cslf-completion-rooted-bounded-run-authorization-v1"
)
COVERAGE_STATE_COMPLETION_ROOTED_BOUNDED_RESULT_SCHEMA = (
    "cure-lite-cslf-completion-rooted-bounded-run-result-v1"
)
COVERAGE_STATE_SUPPORT_ORIENTED_BOUNDED_AUTHORIZATION_SCHEMA = (
    "cure-lite-cslf-support-oriented-bounded-run-authorization-v1"
)
COVERAGE_STATE_SUPPORT_ORIENTED_BOUNDED_RESULT_SCHEMA = (
    "cure-lite-cslf-support-oriented-bounded-run-result-v1"
)
COVERAGE_STATE_V15_PARENT_COMPLETE_FINGERPRINT = (
    "faaa2395623f5edfa0e56ab849d20305b73df1e7b3446b22b834279a2637d14b"
)
COVERAGE_STATE_V15A_PARENT_COMPLETE_FINGERPRINT = (
    "f925ece389a96cd6e8ef5487d91428d7981764b12601133cc3eaf9d11b782d35"
)
COVERAGE_STATE_BOUNDED_MODEL_WIDTH = 32
COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS = (
    "cure_lite/cache/schema.py",
    "cure_lite/coverage_state_batches.py",
    "cure_lite/coverage_state_device_cache.py",
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_observability.py",
    "cure_lite/coverage_state_precomputed_cache.py",
    "cure_lite/coverage_state_raw_catalog.py",
    "cure_lite/coverage_state_schedule.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/data.py",
    "cure_lite/frozen_base.py",
    "cure_lite/intervention.py",
    "cure_lite/instances.py",
    "cure_lite/matching.py",
    "cure_lite/paired_types.py",
    "cure_lite/splits.py",
    "cure_lite/types.py",
    "cure_lite/train/coverage_state_fused_step.py",
    "cure_lite/experiment/cache_pipeline.py",
    "cure_lite/experiment/coverage_state_bounded_protocol.py",
    "cure_lite/experiment/coverage_state_bounded_runner.py",
    "cure_lite/experiment/coverage_state_dataset_free.py",
    "cure_lite/experiment/coverage_state_observability_protocol.py",
    "cure_lite/experiment/coverage_state_raw_catalog.py",
    "cure_lite/experiment/coverage_state_real_dr_inputs.py",
    "cure_lite/experiment/coverage_state_training.py",
    "cure_lite/experiment/coverage_state_zero_level_evaluation.py",
    "cure_lite/experiment/geometry_catalog_protocol.py",
    "cure_lite/experiment/geometry_safe_catalog.py",
    "cure_lite/experiment/training_pipeline.py",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _current_implementation_binding() -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"bounded implementation file is missing: {relative}"
            )
        result.append((relative, sha256(path.read_bytes()).hexdigest()))
    return tuple(result)


def _model_config_payload(
    config: CoverageStateLevelSetConfig,
) -> dict[str, object]:
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
    }


def _expected_model_config(
    preflight: CoverageStateBoundedPreflight,
) -> CoverageStateLevelSetConfig:
    first = preflight.population.cache.raw_catalog.natural_records[0]
    return CoverageStateLevelSetConfig(
        feature_channels=int(first.feature.shape[1]),
        feature_stride=preflight.population.cache.raw_catalog.feature_stride,
        width=COVERAGE_STATE_BOUNDED_MODEL_WIDTH,
    )


@contextmanager
def _deterministic_execution(
    device: torch.device | str,
) -> Iterator[None]:
    """Apply and restore the fixed deterministic FP32 execution policy."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and resolved_device.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        resolved_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
    previous_cpu_rng_state = torch.get_rng_state().clone()
    previous_cuda_rng_state = (
        torch.cuda.get_rng_state(resolved_device).clone()
        if resolved_device.type == "cuda"
        else None
    )
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_matmul_precision = torch.get_float32_matmul_precision()
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms,
            warn_only=previous_warn_only,
        )
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.set_rng_state(previous_cpu_rng_state)
        if previous_cuda_rng_state is not None:
            torch.cuda.set_rng_state(
                previous_cuda_rng_state,
                resolved_device,
            )


@dataclass(frozen=True, eq=False)
class CoverageStateBoundedRunAuthorization(
    CoverageStateRunAuthorization,
):
    """Bind expanded dataset-free evidence to one D_R bounded preflight."""

    preflight: CoverageStateBoundedPreflight
    dataset_free_receipt: CoverageStateDatasetFreeReceipt
    dataset_free_receipt_fingerprint: str
    implementation_binding: tuple[tuple[str, str], ...]
    implementation_fingerprint: str
    model_config_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.preflight, CoverageStateBoundedPreflight):
            raise ValueError("bounded authorization binding changed")
        expected_model_config_fingerprint = stable_fingerprint(
            _model_config_payload(_expected_model_config(self.preflight))
        )
        if (
            not isinstance(
                self.dataset_free_receipt,
                CoverageStateDatasetFreeReceipt,
            )
            or self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != expected_model_config_fingerprint
        ):
            raise ValueError("bounded authorization binding changed")

    @property
    def training_authorized(self) -> bool:
        return (
            self.preflight.training_authorized
            and self.dataset_free_receipt.all_pass
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_BOUNDED_AUTHORIZATION_SCHEMA,
            "scope": COVERAGE_STATE_BOUNDED_SCOPE,
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
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "implementation_fingerprint": (
                self.implementation_fingerprint
            ),
            "model_config_fingerprint": self.model_config_fingerprint,
            "checks": {
                "preflight_passed": self.preflight.training_authorized,
                "dataset_free_gate_passed": (
                    self.dataset_free_receipt.all_pass
                ),
            },
            "training_authorized": self.training_authorized,
            "formal_training_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "detection_performance_evaluation_authorized": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        }

    @cached_property
    def authorization_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        self.preflight.verify_unchanged()
        if (
            self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != stable_fingerprint(
                _model_config_payload(
                    _expected_model_config(self.preflight)
                )
            )
            or stable_fingerprint(self.canonical_payload())
            != self.authorization_fingerprint
        ):
            raise RuntimeError("bounded authorization changed after creation")

    def verify_model_config(
        self,
        model_config: CoverageStateLevelSetConfig,
    ) -> None:
        if (
            not isinstance(model_config, CoverageStateLevelSetConfig)
            or stable_fingerprint(_model_config_payload(model_config))
            != self.model_config_fingerprint
        ):
            raise PermissionError(
                "bounded authorization does not permit this model config"
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
                "bounded authorization does not permit this training run"
            )


def prepare_coverage_state_bounded_run_authorization(
    preflight: CoverageStateBoundedPreflight,
    dataset_free_receipt: CoverageStateDatasetFreeReceipt,
) -> CoverageStateBoundedRunAuthorization:
    """Create a fail-closed bounded authorization from recomputable inputs."""

    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    if not isinstance(
        dataset_free_receipt,
        CoverageStateDatasetFreeReceipt,
    ):
        raise TypeError(
            "dataset_free_receipt must be CoverageStateDatasetFreeReceipt"
        )
    preflight.verify_unchanged()
    implementation_binding = _current_implementation_binding()
    model_config_fingerprint = stable_fingerprint(
        _model_config_payload(_expected_model_config(preflight))
    )
    result = CoverageStateBoundedRunAuthorization(
        preflight=preflight,
        dataset_free_receipt=dataset_free_receipt,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt.receipt_fingerprint
        ),
        implementation_binding=implementation_binding,
        implementation_fingerprint=stable_fingerprint(
            dict(implementation_binding)
        ),
        model_config_fingerprint=model_config_fingerprint,
    )
    result.verify_unchanged()
    return result


@dataclass(frozen=True, eq=False)
class CoverageStateCompletionRootedBoundedRunAuthorization(
    CoverageStateBoundedRunAuthorization,
):
    """Bind the completion-rooted candidate and its frozen controls."""

    objective_suite: tuple[str, ...]
    candidate_objective: str
    candidate_objective_policy: str
    parent_v15_complete_fingerprint: str

    def __post_init__(self) -> None:
        super().__post_init__()
        expected_suite = tuple(
            value.value
            for value in COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES
        )
        if (
            self.objective_suite != expected_suite
            or self.candidate_objective
            != (
                CoverageStatePairObjective
                .COMPLETION_ROOTED_RESPONSE_JOINT.value
            )
            or self.objective_suite[0] != self.candidate_objective
            or self.candidate_objective_policy
            != CSLF_COMPLETION_ROOTED_RESPONSE_POLICY
            or self.parent_v15_complete_fingerprint
            != COVERAGE_STATE_V15_PARENT_COMPLETE_FINGERPRINT
        ):
            raise ValueError(
                "completion-rooted bounded authorization binding changed"
            )

    def canonical_payload(self) -> dict[str, object]:
        payload = super().canonical_payload()
        payload["schema_version"] = (
            COVERAGE_STATE_COMPLETION_ROOTED_BOUNDED_AUTHORIZATION_SCHEMA
        )
        payload["objective_suite"] = list(self.objective_suite)
        payload["candidate_objective"] = self.candidate_objective
        payload["candidate_objective_policy"] = (
            self.candidate_objective_policy
        )
        payload["parent_v15_complete_fingerprint"] = (
            self.parent_v15_complete_fingerprint
        )
        payload["decision_policy"] = (
            "candidate_all_frozen_zero_level_gates_and_"
            "complete_matched_control_diagnostics_v1"
        )
        return payload

    def verify_unchanged(self) -> None:
        super().verify_unchanged()
        expected_suite = tuple(
            value.value
            for value in COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES
        )
        if (
            self.objective_suite != expected_suite
            or self.objective_suite[0] != self.candidate_objective
            or self.candidate_objective_policy
            != CSLF_COMPLETION_ROOTED_RESPONSE_POLICY
            or self.parent_v15_complete_fingerprint
            != COVERAGE_STATE_V15_PARENT_COMPLETE_FINGERPRINT
        ):
            raise RuntimeError(
                "completion-rooted bounded authorization changed"
            )


def prepare_coverage_state_completion_rooted_bounded_run_authorization(
    preflight: CoverageStateBoundedPreflight,
    dataset_free_receipt: CoverageStateDatasetFreeReceipt,
) -> CoverageStateCompletionRootedBoundedRunAuthorization:
    """Authorize one completion-rooted candidate/control bounded run."""

    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    if not isinstance(
        dataset_free_receipt,
        CoverageStateDatasetFreeReceipt,
    ):
        raise TypeError(
            "dataset_free_receipt must be CoverageStateDatasetFreeReceipt"
        )
    preflight.verify_unchanged()
    implementation_binding = _current_implementation_binding()
    model_config_fingerprint = stable_fingerprint(
        _model_config_payload(_expected_model_config(preflight))
    )
    objective_suite = tuple(
        value.value
        for value in COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES
    )
    result = CoverageStateCompletionRootedBoundedRunAuthorization(
        preflight=preflight,
        dataset_free_receipt=dataset_free_receipt,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt.receipt_fingerprint
        ),
        implementation_binding=implementation_binding,
        implementation_fingerprint=stable_fingerprint(
            dict(implementation_binding)
        ),
        model_config_fingerprint=model_config_fingerprint,
        objective_suite=objective_suite,
        candidate_objective=objective_suite[0],
        candidate_objective_policy=(
            CSLF_COMPLETION_ROOTED_RESPONSE_POLICY
        ),
        parent_v15_complete_fingerprint=(
            COVERAGE_STATE_V15_PARENT_COMPLETE_FINGERPRINT
        ),
    )
    result.verify_unchanged()
    return result


@dataclass(frozen=True, eq=False)
class CoverageStateSupportOrientedBoundedRunAuthorization(
    CoverageStateRunAuthorization,
):
    """Bind the SORR candidate to v15A and its own dataset-free gate."""

    preflight: CoverageStateBoundedPreflight
    dataset_free_receipt: CoverageStateSupportOrientedDatasetFreeReceipt
    dataset_free_receipt_fingerprint: str
    implementation_binding: tuple[tuple[str, str], ...]
    implementation_fingerprint: str
    model_config_fingerprint: str
    objective_suite: tuple[str, ...]
    candidate_objective: str
    candidate_objective_policy: str
    parent_v15a_complete_fingerprint: str

    def __post_init__(self) -> None:
        expected_suite = tuple(
            value.value
            for value in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
        )
        expected_model_config_fingerprint = stable_fingerprint(
            _model_config_payload(_expected_model_config(self.preflight))
        )
        if (
            not isinstance(self.preflight, CoverageStateBoundedPreflight)
            or not isinstance(
                self.dataset_free_receipt,
                CoverageStateSupportOrientedDatasetFreeReceipt,
            )
            or self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != expected_model_config_fingerprint
            or self.objective_suite != expected_suite
            or self.candidate_objective
            != CoverageStatePairObjective.SUPPORT_ORIENTED_RESPONSE_JOINT.value
            or self.objective_suite[0] != self.candidate_objective
            or self.candidate_objective_policy
            != CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
            or self.parent_v15a_complete_fingerprint
            != COVERAGE_STATE_V15A_PARENT_COMPLETE_FINGERPRINT
        ):
            raise ValueError(
                "support-oriented bounded authorization binding changed"
            )

    @property
    def training_authorized(self) -> bool:
        return (
            self.preflight.training_authorized
            and self.dataset_free_receipt.all_pass
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                COVERAGE_STATE_SUPPORT_ORIENTED_BOUNDED_AUTHORIZATION_SCHEMA
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
            "objective_suite": list(self.objective_suite),
            "candidate_objective": self.candidate_objective,
            "candidate_objective_policy": (
                self.candidate_objective_policy
            ),
            "parent_v15a_complete_fingerprint": (
                self.parent_v15a_complete_fingerprint
            ),
            "decision_policy": (
                "candidate_original_zero_level_gates_and_"
                "complete_matched_control_diagnostics_v1"
            ),
            "checks": {
                "preflight_passed": self.preflight.training_authorized,
                "support_oriented_dataset_free_gate_passed": (
                    self.dataset_free_receipt.all_pass
                ),
            },
            "training_authorized": self.training_authorized,
            "formal_training_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "detection_performance_evaluation_authorized": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        }

    @cached_property
    def authorization_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        self.preflight.verify_unchanged()
        expected_suite = tuple(
            value.value
            for value in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
        )
        if (
            self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != stable_fingerprint(
                _model_config_payload(
                    _expected_model_config(self.preflight)
                )
            )
            or self.objective_suite != expected_suite
            or self.objective_suite[0] != self.candidate_objective
            or self.candidate_objective_policy
            != CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
            or self.parent_v15a_complete_fingerprint
            != COVERAGE_STATE_V15A_PARENT_COMPLETE_FINGERPRINT
            or stable_fingerprint(self.canonical_payload())
            != self.authorization_fingerprint
        ):
            raise RuntimeError(
                "support-oriented bounded authorization changed"
            )

    def verify_model_config(
        self,
        model_config: CoverageStateLevelSetConfig,
    ) -> None:
        if (
            not isinstance(model_config, CoverageStateLevelSetConfig)
            or stable_fingerprint(_model_config_payload(model_config))
            != self.model_config_fingerprint
        ):
            raise PermissionError(
                "support-oriented bounded authorization does not permit "
                "this model config"
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
                "support-oriented bounded authorization does not permit "
                "this training run"
            )


def prepare_coverage_state_support_oriented_bounded_run_authorization(
    preflight: CoverageStateBoundedPreflight,
    dataset_free_receipt: CoverageStateSupportOrientedDatasetFreeReceipt,
) -> CoverageStateSupportOrientedBoundedRunAuthorization:
    """Authorize the fixed SORR candidate and two frozen controls."""

    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    if not isinstance(
        dataset_free_receipt,
        CoverageStateSupportOrientedDatasetFreeReceipt,
    ):
        raise TypeError(
            "dataset_free_receipt must be "
            "CoverageStateSupportOrientedDatasetFreeReceipt"
        )
    preflight.verify_unchanged()
    implementation_binding = _current_implementation_binding()
    model_config_fingerprint = stable_fingerprint(
        _model_config_payload(_expected_model_config(preflight))
    )
    objective_suite = tuple(
        value.value
        for value in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
    )
    result = CoverageStateSupportOrientedBoundedRunAuthorization(
        preflight=preflight,
        dataset_free_receipt=dataset_free_receipt,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt.receipt_fingerprint
        ),
        implementation_binding=implementation_binding,
        implementation_fingerprint=stable_fingerprint(
            dict(implementation_binding)
        ),
        model_config_fingerprint=model_config_fingerprint,
        objective_suite=objective_suite,
        candidate_objective=objective_suite[0],
        candidate_objective_policy=CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
        parent_v15a_complete_fingerprint=(
            COVERAGE_STATE_V15A_PARENT_COMPLETE_FINGERPRINT
        ),
    )
    result.verify_unchanged()
    return result


def _bounded_result_checks(
    authorization: CoverageStateBoundedRunAuthorization,
    training: CoverageStateMatchedTrainingResult,
    diagnostics: tuple[
        tuple[str, CoverageStateZeroLevelEvaluationResult],
        ...,
    ],
) -> tuple[tuple[str, bool], ...]:
    results = training.results
    checks = {
        "authorization": authorization.training_authorized,
        "three_objectives": len(results) == 3,
        "updates": all(
            value.completed_updates == COVERAGE_STATE_BOUNDED_UPDATES
            for value in results
        ),
        "forward_calls": all(
            value.forward_calls == COVERAGE_STATE_BOUNDED_UPDATES
            for value in results
        ),
        "backward_calls": all(
            value.backward_calls == COVERAGE_STATE_BOUNDED_UPDATES
            for value in results
        ),
        "optimizer_steps": all(
            value.optimizer_steps == COVERAGE_STATE_BOUNDED_UPDATES
            for value in results
        ),
        "logical_states": all(
            value.logical_state_evaluations
            == COVERAGE_STATE_BOUNDED_UPDATES * 12
            for value in results
        ),
        "finite_state_audits": all(
            value.finite_state_audits
            == COVERAGE_STATE_BOUNDED_UPDATES + 1
            for value in results
        ),
        "same_optimizer": len(
            {value.optimizer_config_fingerprint for value in results}
        )
        == 1,
        "same_device_cache": len(
            {
                (
                    value.execution_device,
                    value.device_cache_fingerprint,
                    value.device_cache_resident_bytes,
                )
                for value in results
            }
        )
        == 1,
        "matched_training_binding": (
            training.schedule_fingerprint
            == authorization.preflight.schedule.schedule_fingerprint
            and training.cache_fingerprint
            == authorization.preflight.population.bounded_cache_fingerprint
            and training.config.seed == COVERAGE_STATE_BOUNDED_SEED
            and all(
                value.seed == COVERAGE_STATE_BOUNDED_SEED
                for value in results
            )
        ),
        "same_selection_sequence": len(
            {
                tuple(
                    str(row["selection_sequence_fingerprint"])
                    for row in value.epoch_logs
                )
                for value in results
            }
        )
        == 1,
        "authorized_model_config": all(
            stable_fingerprint(_model_config_payload(model.config))
            == authorization.model_config_fingerprint
            for _, model in training.models
        ),
        "diagnostic_objectives": (
            tuple(name for name, _ in diagnostics)
            == tuple(value.objective for value in results)
        ),
        "diagnostic_checkpoint_binding": all(
            diagnostic.checkpoint_fingerprint
            == module_state_fingerprint(model)
            and result.objective == model_name == diagnostic_name
            for result, (model_name, model), (
                diagnostic_name,
                diagnostic,
            ) in zip(
                results,
                training.models,
                diagnostics,
                strict=True,
            )
        ),
        "diagnostic_cache_binding": all(
            diagnostic.cache_fingerprint
            == authorization.preflight.population.bounded_cache_fingerprint
            and diagnostic.dataset
            == authorization.preflight.population.cache.raw_catalog.dataset
            and diagnostic.split == "D_R"
            for _, diagnostic in diagnostics
        ),
        "zero_level_gates": all(
            diagnostic.bounded_gate_passed
            for _, diagnostic in diagnostics
        ),
        "diagnostic_no_backward": all(
            diagnostic.backward_calls == 0
            for _, diagnostic in diagnostics
        ),
        "diagnostic_no_optimizer": all(
            diagnostic.optimizer_steps == 0
            for _, diagnostic in diagnostics
        ),
        "threshold_zero_no_search": all(
            diagnostic.config.residual_threshold == 0.0
            and not diagnostic.config.threshold_search_performed
            for _, diagnostic in diagnostics
        ),
        "D_V_not_accessed": all(
            not diagnostic.config.d_v_accessed
            for _, diagnostic in diagnostics
        ),
        "D_T_not_accessed": all(
            not diagnostic.config.d_t_accessed
            for _, diagnostic in diagnostics
        ),
        "no_resume": True,
        "no_automatic_retry": True,
    }
    return tuple(sorted(checks.items()))


def _completion_rooted_bounded_result_checks(
    authorization: CoverageStateCompletionRootedBoundedRunAuthorization,
    training: CoverageStateMatchedTrainingResult,
    diagnostics: tuple[
        tuple[str, CoverageStateZeroLevelEvaluationResult],
        ...,
    ],
) -> tuple[tuple[str, bool], ...]:
    """Keep candidate qualification separate from control outcomes."""

    generic = dict(
        _bounded_result_checks(
            authorization,
            training,
            diagnostics,
        )
    )
    generic.pop("zero_level_gates")
    names = tuple(value.objective for value in training.results)
    expected = authorization.objective_suite
    diagnostic_by_name = dict(diagnostics)
    candidate = diagnostic_by_name.get(
        authorization.candidate_objective
    )
    controls = expected[1:]
    generic.update(
        {
            "completion_rooted_objective_suite": names == expected,
            "candidate_zero_level_gates": (
                candidate is not None
                and candidate.bounded_gate_passed
            ),
            "control_diagnostics_complete": (
                tuple(name for name, _ in diagnostics) == expected
                and all(name in diagnostic_by_name for name in controls)
                and len(controls) == 2
            ),
            "candidate_policy_bound": (
                authorization.candidate_objective_policy
                == CSLF_COMPLETION_ROOTED_RESPONSE_POLICY
            ),
            "parent_v15_result_bound": (
                authorization.parent_v15_complete_fingerprint
                == COVERAGE_STATE_V15_PARENT_COMPLETE_FINGERPRINT
            ),
        }
    )
    return tuple(sorted(generic.items()))


def _support_oriented_bounded_result_checks(
    authorization: CoverageStateSupportOrientedBoundedRunAuthorization,
    training: CoverageStateMatchedTrainingResult,
    diagnostics: tuple[
        tuple[str, CoverageStateZeroLevelEvaluationResult],
        ...,
    ],
) -> tuple[tuple[str, bool], ...]:
    """Qualify only SORR while requiring complete matched controls."""

    generic = dict(
        _bounded_result_checks(
            authorization,
            training,
            diagnostics,
        )
    )
    generic.pop("zero_level_gates")
    names = tuple(value.objective for value in training.results)
    expected = authorization.objective_suite
    diagnostic_by_name = dict(diagnostics)
    candidate = diagnostic_by_name.get(
        authorization.candidate_objective
    )
    controls = expected[1:]
    generic.update(
        {
            "support_oriented_objective_suite": names == expected,
            "candidate_original_zero_level_gates": (
                candidate is not None
                and candidate.bounded_gate_passed
            ),
            "control_diagnostics_complete": (
                tuple(name for name, _ in diagnostics) == expected
                and all(name in diagnostic_by_name for name in controls)
                and len(controls) == 2
            ),
            "candidate_policy_bound": (
                authorization.candidate_objective_policy
                == CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
            ),
            "parent_v15a_result_bound": (
                authorization.parent_v15a_complete_fingerprint
                == COVERAGE_STATE_V15A_PARENT_COMPLETE_FINGERPRINT
            ),
        }
    )
    return tuple(sorted(generic.items()))


@dataclass(frozen=True, eq=False)
class CoverageStateBoundedRunResult:
    """Matched training plus read-only zero-level diagnostics."""

    authorization: CoverageStateBoundedRunAuthorization
    training: CoverageStateMatchedTrainingResult
    diagnostics: tuple[
        tuple[str, CoverageStateZeroLevelEvaluationResult],
        ...,
    ]
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        expected = tuple(value.objective for value in self.training.results)
        if (
            tuple(name for name, _ in self.diagnostics) != expected
            or self.checks != tuple(sorted(self.checks))
            or len({name for name, _ in self.checks})
            != len(self.checks)
        ):
            raise ValueError("bounded result is incomplete")

    @property
    def bounded_gate_passed(self) -> bool:
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks if not passed)

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        self.training.verify_unchanged()
        expected = _bounded_result_checks(
            self.authorization,
            self.training,
            self.diagnostics,
        )
        if expected != self.checks:
            raise RuntimeError("bounded result checks changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_BOUNDED_RESULT_SCHEMA,
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "training": self.training.canonical_payload(),
            "diagnostics": {
                name: value.canonical_payload()
                for name, value in self.diagnostics
            },
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "bounded_gate_passed": self.bounded_gate_passed,
            "formal_training_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_claim_supported": False,
        }

    @property
    def result_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


@dataclass(frozen=True, eq=False)
class CoverageStateCompletionRootedBoundedRunResult(
    CoverageStateBoundedRunResult,
):
    """Candidate-qualified result with fully reported matched controls."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(
            self.authorization,
            CoverageStateCompletionRootedBoundedRunAuthorization,
        ):
            raise ValueError(
                "completion-rooted result requires its own authorization"
            )
        expected = self.authorization.objective_suite
        if tuple(
            value.objective for value in self.training.results
        ) != expected:
            raise ValueError(
                "completion-rooted result objective suite changed"
            )

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        self.training.verify_unchanged()
        expected = _completion_rooted_bounded_result_checks(
            self.authorization,
            self.training,
            self.diagnostics,
        )
        if expected != self.checks:
            raise RuntimeError(
                "completion-rooted bounded result checks changed"
            )

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        diagnostic_by_name = dict(self.diagnostics)
        controls = self.authorization.objective_suite[1:]
        return {
            "schema_version": (
                COVERAGE_STATE_COMPLETION_ROOTED_BOUNDED_RESULT_SCHEMA
            ),
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "parent_v15_complete_fingerprint": (
                self.authorization.parent_v15_complete_fingerprint
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
            "control_outcomes_are_not_candidate_gates": True,
            "formal_training_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_claim_supported": False,
        }


@dataclass(frozen=True, eq=False)
class CoverageStateSupportOrientedBoundedRunResult(
    CoverageStateBoundedRunResult,
):
    """SORR-qualified bounded result with non-gating controls."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(
            self.authorization,
            CoverageStateSupportOrientedBoundedRunAuthorization,
        ):
            raise ValueError(
                "support-oriented result requires its own authorization"
            )
        expected = self.authorization.objective_suite
        if tuple(
            value.objective for value in self.training.results
        ) != expected:
            raise ValueError(
                "support-oriented result objective suite changed"
            )

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        self.training.verify_unchanged()
        expected = _support_oriented_bounded_result_checks(
            self.authorization,
            self.training,
            self.diagnostics,
        )
        if expected != self.checks:
            raise RuntimeError(
                "support-oriented bounded result checks changed"
            )

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        diagnostic_by_name = dict(self.diagnostics)
        controls = self.authorization.objective_suite[1:]
        return {
            "schema_version": (
                COVERAGE_STATE_SUPPORT_ORIENTED_BOUNDED_RESULT_SCHEMA
            ),
            "runtime_splits": ["D_R"],
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "parent_v15a_complete_fingerprint": (
                self.authorization.parent_v15a_complete_fingerprint
            ),
            "candidate_objective": (
                self.authorization.candidate_objective
            ),
            "candidate_objective_policy": (
                self.authorization.candidate_objective_policy
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


def run_coverage_state_bounded_400(
    authorization: CoverageStateBoundedRunAuthorization,
    model_config: CoverageStateLevelSetConfig,
    *,
    device: torch.device | str,
) -> CoverageStateBoundedRunResult:
    """Run exactly three matched 400-update objectives, then diagnose them."""

    if not isinstance(
        authorization,
        CoverageStateBoundedRunAuthorization,
    ):
        raise TypeError(
            "authorization must be CoverageStateBoundedRunAuthorization"
        )
    if not isinstance(model_config, CoverageStateLevelSetConfig):
        raise TypeError("model_config must be CoverageStateLevelSetConfig")
    preflight = authorization.preflight
    authorization.verify_model_config(model_config)
    authorization.verify_for_run(
        cache=preflight.population.cache,
        schedule=preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
    )
    with _deterministic_execution(device):
        training = train_matched_coverage_state_objectives(
            model_config,
            preflight.population.cache,
            preflight.schedule,
            config=CoverageStateMatchedTrainingConfig(seed=42),
            device=device,
            authorization=authorization,
        )
        diagnostic_values: list[
            tuple[str, CoverageStateZeroLevelEvaluationResult]
        ] = []
        for name, model in training.models:
            model.eval()
            diagnostic_values.append(
                (
                    name,
                    evaluate_coverage_state_zero_level_checkpoint(
                        model,
                        preflight.population.cache,
                        device=device,
                    ),
                )
            )
        diagnostics = tuple(diagnostic_values)
    result = CoverageStateBoundedRunResult(
        authorization=authorization,
        training=training,
        diagnostics=diagnostics,
        checks=_bounded_result_checks(
            authorization,
            training,
            diagnostics,
        ),
    )
    result.verify_unchanged()
    return result


def run_coverage_state_completion_rooted_bounded_400(
    authorization: CoverageStateCompletionRootedBoundedRunAuthorization,
    model_config: CoverageStateLevelSetConfig,
    *,
    device: torch.device | str,
) -> CoverageStateCompletionRootedBoundedRunResult:
    """Run one completion-rooted candidate and two frozen controls."""

    if not isinstance(
        authorization,
        CoverageStateCompletionRootedBoundedRunAuthorization,
    ):
        raise TypeError(
            "authorization must be completion-rooted bounded authorization"
        )
    if not isinstance(model_config, CoverageStateLevelSetConfig):
        raise TypeError("model_config must be CoverageStateLevelSetConfig")
    preflight = authorization.preflight
    authorization.verify_model_config(model_config)
    authorization.verify_for_run(
        cache=preflight.population.cache,
        schedule=preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
    )
    with _deterministic_execution(device):
        training = (
            train_matched_coverage_state_completion_rooted_objectives(
                model_config,
                preflight.population.cache,
                preflight.schedule,
                config=CoverageStateMatchedTrainingConfig(seed=42),
                device=device,
                authorization=authorization,
            )
        )
        diagnostic_values: list[
            tuple[str, CoverageStateZeroLevelEvaluationResult]
        ] = []
        for name, model in training.models:
            model.eval()
            diagnostic_values.append(
                (
                    name,
                    evaluate_coverage_state_zero_level_checkpoint(
                        model,
                        preflight.population.cache,
                        device=device,
                    ),
                )
            )
        diagnostics = tuple(diagnostic_values)
    result = CoverageStateCompletionRootedBoundedRunResult(
        authorization=authorization,
        training=training,
        diagnostics=diagnostics,
        checks=_completion_rooted_bounded_result_checks(
            authorization,
            training,
            diagnostics,
        ),
    )
    result.verify_unchanged()
    return result


def run_coverage_state_support_oriented_bounded_400(
    authorization: CoverageStateSupportOrientedBoundedRunAuthorization,
    model_config: CoverageStateLevelSetConfig,
    *,
    device: torch.device | str,
) -> CoverageStateSupportOrientedBoundedRunResult:
    """Run SORR and two frozen controls for exactly 400 D_R updates."""

    if not isinstance(
        authorization,
        CoverageStateSupportOrientedBoundedRunAuthorization,
    ):
        raise TypeError(
            "authorization must be support-oriented bounded authorization"
        )
    if not isinstance(model_config, CoverageStateLevelSetConfig):
        raise TypeError("model_config must be CoverageStateLevelSetConfig")
    preflight = authorization.preflight
    authorization.verify_model_config(model_config)
    authorization.verify_for_run(
        cache=preflight.population.cache,
        schedule=preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
    )
    with _deterministic_execution(device):
        training = train_matched_coverage_state_support_oriented_objectives(
            model_config,
            preflight.population.cache,
            preflight.schedule,
            config=CoverageStateMatchedTrainingConfig(
                seed=COVERAGE_STATE_BOUNDED_SEED
            ),
            device=device,
            authorization=authorization,
        )
        diagnostic_values: list[
            tuple[str, CoverageStateZeroLevelEvaluationResult]
        ] = []
        for name, model in training.models:
            model.eval()
            diagnostic_values.append(
                (
                    name,
                    evaluate_coverage_state_zero_level_checkpoint(
                        model,
                        preflight.population.cache,
                        device=device,
                    ),
                )
            )
        diagnostics = tuple(diagnostic_values)
    result = CoverageStateSupportOrientedBoundedRunResult(
        authorization=authorization,
        training=training,
        diagnostics=diagnostics,
        checks=_support_oriented_bounded_result_checks(
            authorization,
            training,
            diagnostics,
        ),
    )
    result.verify_unchanged()
    return result


__all__ = [
    "COVERAGE_STATE_BOUNDED_AUTHORIZATION_SCHEMA",
    "COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_BOUNDED_MODEL_WIDTH",
    "COVERAGE_STATE_BOUNDED_RESULT_SCHEMA",
    "COVERAGE_STATE_COMPLETION_ROOTED_BOUNDED_AUTHORIZATION_SCHEMA",
    "COVERAGE_STATE_COMPLETION_ROOTED_BOUNDED_RESULT_SCHEMA",
    "COVERAGE_STATE_SUPPORT_ORIENTED_BOUNDED_AUTHORIZATION_SCHEMA",
    "COVERAGE_STATE_SUPPORT_ORIENTED_BOUNDED_RESULT_SCHEMA",
    "COVERAGE_STATE_V15A_PARENT_COMPLETE_FINGERPRINT",
    "COVERAGE_STATE_V15_PARENT_COMPLETE_FINGERPRINT",
    "CoverageStateBoundedRunAuthorization",
    "CoverageStateBoundedRunResult",
    "CoverageStateCompletionRootedBoundedRunAuthorization",
    "CoverageStateCompletionRootedBoundedRunResult",
    "CoverageStateSupportOrientedBoundedRunAuthorization",
    "CoverageStateSupportOrientedBoundedRunResult",
    "prepare_coverage_state_completion_rooted_bounded_run_authorization",
    "prepare_coverage_state_bounded_run_authorization",
    "prepare_coverage_state_support_oriented_bounded_run_authorization",
    "run_coverage_state_bounded_400",
    "run_coverage_state_completion_rooted_bounded_400",
    "run_coverage_state_support_oriented_bounded_400",
]
