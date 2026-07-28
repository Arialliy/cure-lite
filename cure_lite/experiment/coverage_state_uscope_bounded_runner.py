"""Single-candidate bounded-400 runner for CMIF + USCOPE.

The runner binds the generated and real-``D_R`` prerequisites, the sealed
v18 PMOPE negative result, and the unchanged seed-42 bounded coordinates.
One call performs exactly one USCOPE training run, one independent pair
certificate, and one shared zero-level evaluation, in that order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Mapping

import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_centered_mixed_interaction import (
    CMIF_ENERGY_POLICY,
    CMIF_INPUT_REPRESENTATION,
    CMIF_INTERACTION_POLICY,
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from ..coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
)
from ..coverage_state_precomputed_cache import CoverageStateScalarCache
from ..coverage_state_schedule import CoverageStateTrainingSchedule
from ..coverage_state_supremal_projection import CSLF_USCOPE_POLICY
from ..frozen_base import module_state_fingerprint
from ..paired_types import tensor_content_fingerprint
from ..train.coverage_state_fused_step import (
    COVERAGE_STATE_USCOPE_MATCHED_OBJECTIVES,
    CoverageStatePairObjective,
)
from .coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_EPOCHS,
    COVERAGE_STATE_BOUNDED_SEED,
    COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH,
    COVERAGE_STATE_BOUNDED_UPDATES,
    CoverageStateBoundedPreflight,
)
from .coverage_state_bounded_runner import (
    COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
    _deterministic_execution,
)
from .coverage_state_cmif_dataset_free import (
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_CMIF_FORMAL_WIDTH,
)
from .coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    CoverageStateMatchedTrainingConfig,
    CoverageStateMatchedTrainingResult,
    CoverageStateRunAuthorization,
    coverage_state_model_fingerprint,
    train_matched_coverage_state_cmif_uscope_objectives,
)
from .coverage_state_uscope_certificate import (
    COVERAGE_STATE_USCOPE_CERTIFICATE_PAIR_COUNT,
    CoverageStateUSCOPECertificateReceipt,
    audit_coverage_state_uscope_pair_certificate,
)
from .coverage_state_uscope_dataset_free import (
    COVERAGE_STATE_USCOPE_MARGIN,
    CoverageStateUSCOPEDatasetFreeReceipt,
)
from .coverage_state_uscope_decision import (
    CoverageStateUSCOPEZeroLevelDecision,
    decide_coverage_state_uscope_zero_level,
)
from .coverage_state_uscope_dr_gate import (
    COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED,
    COVERAGE_STATE_USCOPE_DR_GATE_SCHEMA,
    CoverageStateUSCOPEDRGateReceipt,
)
from .coverage_state_uscope_sealed_v18 import (
    COVERAGE_STATE_USCOPE_V18_SEALED_RECEIPT_FINGERPRINT,
    CoverageStateUSCOPESealedV18Receipt,
    verify_repository_coverage_state_uscope_sealed_v18,
)
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
    evaluate_coverage_state_zero_level_checkpoint,
)


COVERAGE_STATE_USCOPE_BOUNDED_AUTHORIZATION_SCHEMA = (
    "cure-lite-cmif-v19-uscope-bounded-run-authorization-v1"
)
COVERAGE_STATE_USCOPE_BOUNDED_RESULT_SCHEMA = (
    "cure-lite-cmif-v19-uscope-bounded-run-result-v1"
)
COVERAGE_STATE_USCOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT = (
    "a4086bcffba4035984a8c334b3fa194910bcb7376a573f7f96ef8d36e097240d"
)
COVERAGE_STATE_USCOPE_HISTORICAL_SCHEDULE_FINGERPRINT = (
    "641699803ee6d0472e447c3d4150b7fbe02bc20160d2707658b0d90a175c3c9c"
)
COVERAGE_STATE_USCOPE_HISTORICAL_CACHE_FINGERPRINT = (
    "c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca00754c206092aaf66d8"
)
COVERAGE_STATE_USCOPE_HISTORICAL_OPTIMIZER_FINGERPRINT = (
    "2d058b1cad606e3c1b723aab05925efb2e873c2b3bf021aeaf0f7df40e0690f0"
)
COVERAGE_STATE_USCOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT = (
    "76ed2f94b4187154bad62896b93d637f131865f2e4e8dad38becb4bebc71119f"
)
COVERAGE_STATE_USCOPE_BOUNDED_IMPLEMENTATION_PATHS = tuple(
    dict.fromkeys(
        (
            *COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
            "cure_lite/coverage_state_phase_preserving.py",
            "cure_lite/coverage_state_centered_mixed_interaction.py",
            "cure_lite/coverage_state_supremal_projection.py",
            "cure_lite/train/coverage_state_fused_step.py",
            "cure_lite/experiment/coverage_state_training.py",
            "cure_lite/experiment/coverage_state_uscope_dataset_free.py",
            "cure_lite/experiment/coverage_state_uscope_dr_gate.py",
            "cure_lite/experiment/coverage_state_uscope_sealed_v18.py",
            "cure_lite/experiment/coverage_state_uscope_certificate.py",
            "cure_lite/experiment/coverage_state_uscope_decision.py",
            "cure_lite/experiment/coverage_state_uscope_bounded_runner.py",
        )
    )
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _current_implementation_binding() -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_USCOPE_BOUNDED_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"USCOPE bounded implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _model_config_payload(
    config: CoverageStateCenteredMixedInteractionConfig,
) -> dict[str, object]:
    if type(config) is not CoverageStateCenteredMixedInteractionConfig:
        raise TypeError("USCOPE requires the exact CMIF config")
    return {
        "config": asdict(config),
        "model_class": "CURELiteCenteredMixedInteractionLevelSet",
        "expected_parameter_count": config.expected_parameter_count,
    }


def _certificate_model_fingerprint(
    model: CURELiteCenteredMixedInteractionLevelSet,
) -> str:
    return stable_fingerprint(
        {
            "class": (
                f"{type(model).__module__}.{type(model).__qualname__}"
            ),
            "config": asdict(model.config),
            "state": {
                name: tensor_content_fingerprint(value)
                for name, value in sorted(model.state_dict().items())
            },
        }
    )


def _dr_model_config_payload(
    config: CoverageStateCenteredMixedInteractionConfig,
) -> dict[str, object]:
    return {
        "feature_channels": config.feature_channels,
        "feature_stride": config.feature_stride,
        "width": config.width,
        "parameter_count": config.expected_parameter_count,
        "objective_policy": CSLF_USCOPE_POLICY,
        "fixed_margin_hex": COVERAGE_STATE_USCOPE_MARGIN.hex(),
    }


def _common_initial_model_fingerprints(
    config: CoverageStateCenteredMixedInteractionConfig,
) -> tuple[str, str]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(
            COVERAGE_STATE_BOUNDED_SEED
        )
        model = CURELiteCenteredMixedInteractionLevelSet(config)
    return (
        coverage_state_model_fingerprint(model),
        stable_fingerprint(
            {
                name: tensor_content_fingerprint(value)
                for name, value in sorted(model.state_dict().items())
            }
        ),
    )


def expected_coverage_state_uscope_config(
    preflight: CoverageStateBoundedPreflight,
) -> CoverageStateCenteredMixedInteractionConfig:
    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    return CoverageStateCenteredMixedInteractionConfig(
        feature_channels=COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_CMIF_FORMAL_WIDTH,
    )


def verify_current_sealed_v18_negative(
) -> CoverageStateUSCOPESealedV18Receipt:
    return verify_repository_coverage_state_uscope_sealed_v18(
        _repository_root()
    )


def _dr_gate_binding_payload(
    receipt: CoverageStateUSCOPEDRGateReceipt,
    *,
    dataset_free_receipt_fingerprint: str,
    canonical_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(receipt, CoverageStateUSCOPEDRGateReceipt):
        raise TypeError("dr_gate_receipt must be a USCOPE D_R receipt")
    if canonical_payload is None:
        payload = receipt.canonical_payload()
    else:
        receipt.verify_unchanged()
        payload = dict(canonical_payload)
    checks = dict(receipt.checks)
    values = {
        "schema_version": payload.get("schema_version"),
        "receipt_fingerprint": stable_fingerprint(payload),
        "dataset_free_receipt_fingerprint": (
            receipt.dataset_free_receipt.receipt_fingerprint
        ),
        "real_inputs_build_fingerprint": (
            receipt.real_inputs.build_fingerprint
        ),
        "source_binding_fingerprint": (
            receipt.real_inputs.source_binding.binding_fingerprint
        ),
        "bounded_population_fingerprint": (
            receipt.population.population_fingerprint
        ),
        "bounded_cache_fingerprint": (
            receipt.population.cache.cache_fingerprint
        ),
        "implementation_fingerprint": stable_fingerprint(
            dict(receipt.implementation_binding)
        ),
        "geometry_fingerprint": stable_fingerprint(receipt.geometry),
        "probe_fingerprint": stable_fingerprint(receipt.probe),
        "checks_fingerprint": stable_fingerprint(checks),
        "execution_seed": receipt.probe.get("execution_seed"),
        "runtime_splits": receipt.probe.get("runtime_splits"),
        "model_config_fingerprint": stable_fingerprint(
            receipt.probe.get("model_config")
        ),
        "initial_model_fingerprint": receipt.probe.get(
            "initial_model_fingerprint"
        ),
        "all_pass": payload.get("all_pass"),
    }
    if (
        values["schema_version"] != COVERAGE_STATE_USCOPE_DR_GATE_SCHEMA
        or values["dataset_free_receipt_fingerprint"]
        != dataset_free_receipt_fingerprint
        or values["execution_seed"]
        != COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED
        or values["runtime_splits"] != ["D_R"]
        or values["all_pass"] is not True
        or payload.get("all_pass") is not True
        or payload.get("evidence_fingerprint")
        != receipt.evidence_fingerprint
        or not checks
        or not all(checks.values())
    ):
        raise PermissionError("USCOPE D_R gate is not a bound pass")
    return values


@dataclass(frozen=True, eq=False)
class CoverageStateUSCOPEBoundedRunAuthorization(
    CoverageStateRunAuthorization,
):
    preflight: CoverageStateBoundedPreflight
    dataset_free_receipt: CoverageStateUSCOPEDatasetFreeReceipt
    dataset_free_receipt_fingerprint: str
    dr_gate_receipt: CoverageStateUSCOPEDRGateReceipt
    dr_gate_binding: tuple[tuple[str, object], ...]
    sealed_v18_receipt: CoverageStateUSCOPESealedV18Receipt
    sealed_v18_receipt_fingerprint: str
    implementation_binding: tuple[tuple[str, str], ...]
    implementation_fingerprint: str
    model_config_fingerprint: str
    expected_parameter_count: int
    objective_suite: tuple[str, ...]
    candidate_objective: str
    candidate_objective_policy: str
    fixed_margin: float
    coverage_policy: str
    interaction_policy: str
    energy_policy: str

    def __post_init__(self) -> None:
        self._validate_lightweight_bindings()
        _ = self.authorization_fingerprint

    def _validate_lightweight_bindings(self) -> None:
        """Validate the sealed snapshot without replaying upstream work."""

        if (
            not isinstance(self.preflight, CoverageStateBoundedPreflight)
            or not isinstance(
                self.dataset_free_receipt,
                CoverageStateUSCOPEDatasetFreeReceipt,
            )
            or not isinstance(
                self.dr_gate_receipt,
                CoverageStateUSCOPEDRGateReceipt,
            )
            or not isinstance(
                self.sealed_v18_receipt,
                CoverageStateUSCOPESealedV18Receipt,
            )
        ):
            raise TypeError("USCOPE authorization prerequisite type changed")
        expected_config = expected_coverage_state_uscope_config(
            self.preflight
        )
        runtime_initial, dr_initial = _common_initial_model_fingerprints(
            expected_config
        )
        dr = dict(self.dr_gate_binding)
        if (
            self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or not all(
                value for _, value in self.dataset_free_receipt.checks
            )
            or self.sealed_v18_receipt.receipt_fingerprint
            != self.sealed_v18_receipt_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != stable_fingerprint(_model_config_payload(expected_config))
            or self.expected_parameter_count
            != COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
            or self.expected_parameter_count
            != expected_config.expected_parameter_count
            or self.objective_suite != ("uscope_joint",)
            or self.candidate_objective
            != CoverageStatePairObjective.USCOPE_JOINT.value
            or self.candidate_objective_policy != CSLF_USCOPE_POLICY
            or self.fixed_margin != COVERAGE_STATE_USCOPE_MARGIN
            or self.coverage_policy
            != CSLF_PHASE_PRESERVING_COVERAGE_POLICY
            or self.interaction_policy != CMIF_INTERACTION_POLICY
            or self.energy_policy != CMIF_ENERGY_POLICY
            or runtime_initial
            != COVERAGE_STATE_USCOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            or self.preflight.schedule.config.seed
            != COVERAGE_STATE_BOUNDED_SEED
            or self.preflight.schedule.config.epochs
            != COVERAGE_STATE_BOUNDED_EPOCHS
            or self.preflight.schedule.config.steps_per_epoch
            != COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
            or self.preflight.schedule.config.updates
            != COVERAGE_STATE_BOUNDED_UPDATES
            or self.preflight.schedule.schedule_fingerprint
            != COVERAGE_STATE_USCOPE_HISTORICAL_SCHEDULE_FINGERPRINT
            or self.preflight.population.bounded_cache_fingerprint
            != COVERAGE_STATE_USCOPE_HISTORICAL_CACHE_FINGERPRINT
            or dr.get("bounded_population_fingerprint")
            != self.preflight.population.population_fingerprint
            or dr.get("bounded_cache_fingerprint")
            != self.preflight.population.bounded_cache_fingerprint
            or dr.get("execution_seed") != COVERAGE_STATE_BOUNDED_SEED
            or dr.get("initial_model_fingerprint") != dr_initial
            or dr.get("model_config_fingerprint")
            != stable_fingerprint(
                _dr_model_config_payload(expected_config)
            )
            or dr.get("all_pass") is not True
        ):
            raise ValueError("USCOPE bounded authorization binding changed")

    @property
    def training_authorized(self) -> bool:
        return (
            self.preflight.training_authorized
            and all(
                value for _, value in self.dataset_free_receipt.checks
            )
            and dict(self.dr_gate_binding).get("all_pass") is True
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                COVERAGE_STATE_USCOPE_BOUNDED_AUTHORIZATION_SCHEMA
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
            "budget": {
                "seed": COVERAGE_STATE_BOUNDED_SEED,
                "epochs": COVERAGE_STATE_BOUNDED_EPOCHS,
                "steps_per_epoch": (
                    COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
                ),
                "updates": COVERAGE_STATE_BOUNDED_UPDATES,
                "objectives": 1,
            },
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "dr_gate": dict(self.dr_gate_binding),
            "sealed_v18_receipt": (
                self.sealed_v18_receipt.canonical_payload()
            ),
            "sealed_v18_receipt_fingerprint": (
                self.sealed_v18_receipt_fingerprint
            ),
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "implementation_fingerprint": (
                self.implementation_fingerprint
            ),
            "model_config_fingerprint": self.model_config_fingerprint,
            "model_class": "CURELiteCenteredMixedInteractionLevelSet",
            "expected_parameter_count": self.expected_parameter_count,
            "input_representation": CMIF_INPUT_REPRESENTATION,
            "coverage_policy": self.coverage_policy,
            "interaction_policy": self.interaction_policy,
            "energy_policy": self.energy_policy,
            "objective_suite": list(self.objective_suite),
            "candidate_objective": self.candidate_objective,
            "candidate_objective_policy": (
                self.candidate_objective_policy
            ),
            "fixed_margin_hex": self.fixed_margin.hex(),
            "historical_comparison_coordinates": {
                "common_initial_model_fingerprint": (
                    COVERAGE_STATE_USCOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT
                ),
                "optimizer_fingerprint": (
                    COVERAGE_STATE_USCOPE_HISTORICAL_OPTIMIZER_FINGERPRINT
                ),
                "device_cache_fingerprint": (
                    COVERAGE_STATE_USCOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT
                ),
            },
            "checks": {
                "preflight_passed": self.preflight.training_authorized,
                "dataset_free_gate_passed": (
                    all(
                        value
                        for _, value
                        in self.dataset_free_receipt.checks
                    )
                ),
                "D_R_gate_passed": (
                    dict(self.dr_gate_binding).get("all_pass") is True
                ),
                "sealed_v18_negative_bound_read_only": True,
                "singleton_uscope_candidate": (
                    self.objective_suite == ("uscope_joint",)
                ),
            },
            "training_authorized": self.training_authorized,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "performance_claim_supported": False,
        }

    @cached_property
    def authorization_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        self._validate_lightweight_bindings()
        if stable_fingerprint(
            self.canonical_payload()
        ) != self.authorization_fingerprint:
            raise RuntimeError("USCOPE bounded authorization changed")

    def verify_model_config(
        self,
        model_config: CoverageStateCenteredMixedInteractionConfig,
    ) -> None:
        if (
            type(model_config)
            is not CoverageStateCenteredMixedInteractionConfig
            or stable_fingerprint(_model_config_payload(model_config))
            != self.model_config_fingerprint
            or model_config.expected_parameter_count
            != self.expected_parameter_count
        ):
            raise PermissionError(
                "USCOPE authorization rejects this model config"
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
                "USCOPE authorization rejects this bounded run"
            )


def prepare_coverage_state_uscope_bounded_run_authorization(
    preflight: CoverageStateBoundedPreflight,
    dataset_free_receipt: CoverageStateUSCOPEDatasetFreeReceipt,
    dr_gate_receipt: CoverageStateUSCOPEDRGateReceipt,
    *,
    sealed_v18_receipt: (
        CoverageStateUSCOPESealedV18Receipt | None
    ) = None,
    dr_gate_canonical_payload: Mapping[str, object] | None = None,
) -> CoverageStateUSCOPEBoundedRunAuthorization:
    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    if not isinstance(
        dataset_free_receipt,
        CoverageStateUSCOPEDatasetFreeReceipt,
    ):
        raise TypeError("dataset_free_receipt must be USCOPE dataset-free")
    if not isinstance(
        dr_gate_receipt,
        CoverageStateUSCOPEDRGateReceipt,
    ):
        raise TypeError("dr_gate_receipt must be USCOPE D_R")
    preflight.verify_unchanged()
    dataset_free_receipt.verify_unchanged()
    dr_binding = tuple(
        sorted(
            _dr_gate_binding_payload(
                dr_gate_receipt,
                dataset_free_receipt_fingerprint=(
                    dataset_free_receipt.receipt_fingerprint
                ),
                canonical_payload=dr_gate_canonical_payload,
            ).items()
        )
    )
    if sealed_v18_receipt is None:
        sealed = verify_current_sealed_v18_negative()
    else:
        if not isinstance(
            sealed_v18_receipt,
            CoverageStateUSCOPESealedV18Receipt,
        ):
            raise TypeError(
                "sealed_v18_receipt must be the sealed v18 receipt"
            )
        sealed = sealed_v18_receipt
    if sealed.receipt_fingerprint != (
        COVERAGE_STATE_USCOPE_V18_SEALED_RECEIPT_FINGERPRINT
    ):
        raise RuntimeError("provided sealed v18 receipt changed")
    implementation = _current_implementation_binding()
    model_config = expected_coverage_state_uscope_config(preflight)
    result = CoverageStateUSCOPEBoundedRunAuthorization(
        preflight=preflight,
        dataset_free_receipt=dataset_free_receipt,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt.receipt_fingerprint
        ),
        dr_gate_receipt=dr_gate_receipt,
        dr_gate_binding=dr_binding,
        sealed_v18_receipt=sealed,
        sealed_v18_receipt_fingerprint=sealed.receipt_fingerprint,
        implementation_binding=implementation,
        implementation_fingerprint=stable_fingerprint(
            dict(implementation)
        ),
        model_config_fingerprint=stable_fingerprint(
            _model_config_payload(model_config)
        ),
        expected_parameter_count=model_config.expected_parameter_count,
        objective_suite=tuple(
            value.value
            for value in COVERAGE_STATE_USCOPE_MATCHED_OBJECTIVES
        ),
        candidate_objective=CoverageStatePairObjective.USCOPE_JOINT.value,
        candidate_objective_policy=CSLF_USCOPE_POLICY,
        fixed_margin=COVERAGE_STATE_USCOPE_MARGIN,
        coverage_policy=CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
        interaction_policy=CMIF_INTERACTION_POLICY,
        energy_policy=CMIF_ENERGY_POLICY,
    )
    result.verify_unchanged()
    return result


def _uscope_bounded_result_checks(
    authorization: CoverageStateUSCOPEBoundedRunAuthorization,
    training: CoverageStateMatchedTrainingResult,
    certificate: CoverageStateUSCOPECertificateReceipt,
    diagnostic: CoverageStateZeroLevelEvaluationResult,
    decision: CoverageStateUSCOPEZeroLevelDecision,
    *,
    training_invocations: int,
    certificate_invocations: int,
    zero_level_evaluation_invocations: int,
) -> tuple[tuple[str, bool], ...]:
    results = training.results
    models = training.models
    row = results[0] if len(results) == 1 else None
    model_entry = models[0] if len(models) == 1 else None
    model = model_entry[1] if model_entry is not None else None
    pair_certificates = certificate.pair_certificates
    latency = (
        dict(row.first_nonzero_gradient_update)
        if row is not None
        else {}
    )
    checks = {
        "authorization_and_provenance": (
            authorization.training_authorized
            and authorization.sealed_v18_receipt.canonical_payload()[
                "historical_negative_result"
            ]
            is True
        ),
        "singleton_uscope_training": (
            row is not None
            and model_entry is not None
            and row.objective == "uscope_joint"
            and row.objective_policy == CSLF_USCOPE_POLICY
            and model_entry[0] == "uscope_joint"
        ),
        "fixed_seed_budget_and_compute": (
            row is not None
            and row.seed == COVERAGE_STATE_BOUNDED_SEED
            and row.epochs == COVERAGE_STATE_BOUNDED_EPOCHS
            and row.steps_per_epoch
            == COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
            and row.completed_updates == COVERAGE_STATE_BOUNDED_UPDATES
            and row.forward_calls == COVERAGE_STATE_BOUNDED_UPDATES
            and row.backward_calls == COVERAGE_STATE_BOUNDED_UPDATES
            and row.optimizer_steps == COVERAGE_STATE_BOUNDED_UPDATES
            and row.logical_state_evaluations
            == 12 * COVERAGE_STATE_BOUNDED_UPDATES
            and row.finite_state_audits
            == COVERAGE_STATE_BOUNDED_UPDATES + 1
        ),
        "historical_training_coordinates": (
            row is not None
            and training.common_initial_model_fingerprint
            == COVERAGE_STATE_USCOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            and row.initial_model_fingerprint
            == COVERAGE_STATE_USCOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            and row.schedule_fingerprint
            == COVERAGE_STATE_USCOPE_HISTORICAL_SCHEDULE_FINGERPRINT
            and row.cache_fingerprint
            == COVERAGE_STATE_USCOPE_HISTORICAL_CACHE_FINGERPRINT
            and row.optimizer_config_fingerprint
            == COVERAGE_STATE_USCOPE_HISTORICAL_OPTIMIZER_FINGERPRINT
            and row.device_cache_fingerprint
            == COVERAGE_STATE_USCOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT
            and row.execution_device == "cuda:0"
            and training.schedule_fingerprint
            == COVERAGE_STATE_USCOPE_HISTORICAL_SCHEDULE_FINGERPRINT
            and training.cache_fingerprint
            == COVERAGE_STATE_USCOPE_HISTORICAL_CACHE_FINGERPRINT
        ),
        "model_learned_with_cmif_gradient_path": (
            row is not None
            and row.final_model_fingerprint
            != row.initial_model_fingerprint
            and set(latency)
            == {
                "joint_hidden_bias",
                "joint_state_weight",
                "scalar_energy_weight",
            }
            and latency["scalar_energy_weight"] == 0
            and 0 <= latency["joint_state_weight"] <= 2
            and 0 <= latency["joint_hidden_bias"] <= 2
        ),
        "exact_cmif_model": (
            model is not None
            and type(model)
            is CURELiteCenteredMixedInteractionLevelSet
            and stable_fingerprint(_model_config_payload(model.config))
            == authorization.model_config_fingerprint
            and sum(
                parameter.numel() for parameter in model.parameters()
            )
            == COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
        ),
        "exact_once_execution_order_ledger": (
            training_invocations
            == certificate_invocations
            == zero_level_evaluation_invocations
            == 1
        ),
        "certificate_checkpoint_and_cache_binding": (
            model is not None
            and certificate.cache_fingerprint
            == authorization.preflight.population.bounded_cache_fingerprint
            and certificate.model_fingerprint_before
            == _certificate_model_fingerprint(model)
            and certificate.model_fingerprint_after
            == certificate.model_fingerprint_before
        ),
        "every_pair_uniform_certificate": (
            certificate.all_pass
            and certificate.pair_batch_size
            == COVERAGE_STATE_USCOPE_CERTIFICATE_PAIR_COUNT
            and certificate.model_forward_invocations == 1
            and len(pair_certificates)
            == COVERAGE_STATE_USCOPE_CERTIFICATE_PAIR_COUNT
            and all(
                value.gamma_strictly_below_margin
                and value.raw_sign_error_pixels == 0
                and value.pair_certificate_passed
                for value in pair_certificates
            )
        ),
        "zero_level_checkpoint_and_D_R_binding": (
            model is not None
            and diagnostic.checkpoint_fingerprint
            == module_state_fingerprint(model)
            and diagnostic.cache_fingerprint
            == authorization.preflight.population.bounded_cache_fingerprint
            and diagnostic.split == "D_R"
        ),
        "zero_level_detection_gates": (
            decision.diagnostic is diagnostic
            and decision.zero_level_gate_passed
        ),
        "response_is_reported_but_not_gating": (
            certificate.same_sign_response_evaluated is False
            and certificate.same_sign_response_is_gate is False
            and decision.canonical_payload()["same_sign_response_policy"]
            == (
                "legacy_all_response_ordering_is_"
                "diagnostic_not_binary_gate"
            )
        ),
        "read_only_post_training_checks": (
            certificate.optimizer_constructed is False
            and certificate.backward_performed is False
            and certificate.training_performed is False
            and diagnostic.backward_calls == 0
            and diagnostic.optimizer_steps == 0
            and not diagnostic.config.d_v_accessed
            and not diagnostic.config.d_t_accessed
        ),
        "no_resume_retry_or_formal800": True,
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True, eq=False)
class CoverageStateUSCOPEBoundedRunResult:
    authorization: CoverageStateUSCOPEBoundedRunAuthorization
    training: CoverageStateMatchedTrainingResult
    certificate: CoverageStateUSCOPECertificateReceipt
    diagnostic: CoverageStateZeroLevelEvaluationResult
    decision: CoverageStateUSCOPEZeroLevelDecision
    training_invocations: int
    certificate_invocations: int
    zero_level_evaluation_invocations: int
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if (
            tuple(value.objective for value in self.training.results)
            != ("uscope_joint",)
            or tuple(name for name, _ in self.training.models)
            != ("uscope_joint",)
            or self.decision.diagnostic is not self.diagnostic
            or self.checks != tuple(sorted(self.checks))
        ):
            raise ValueError("USCOPE bounded result is incomplete")

    @property
    def bounded_gate_passed(self) -> bool:
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(
            name for name, passed in self.checks if not passed
        )

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        self.training.verify_unchanged()
        self.certificate.verify()
        expected = _uscope_bounded_result_checks(
            self.authorization,
            self.training,
            self.certificate,
            self.diagnostic,
            self.decision,
            training_invocations=self.training_invocations,
            certificate_invocations=self.certificate_invocations,
            zero_level_evaluation_invocations=(
                self.zero_level_evaluation_invocations
            ),
        )
        if self.checks != expected:
            raise RuntimeError("USCOPE bounded result checks changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_USCOPE_BOUNDED_RESULT_SCHEMA,
            "runtime_splits": ["D_R"],
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "candidate_objective": "uscope_joint",
            "candidate_objective_policy": CSLF_USCOPE_POLICY,
            "fixed_margin_hex": COVERAGE_STATE_USCOPE_MARGIN.hex(),
            "training": self.training.canonical_payload(),
            "certificate": self.certificate.canonical_payload(),
            "candidate_diagnostic": self.diagnostic.canonical_payload(),
            "decision": self.decision.canonical_payload(),
            "execution_invocations": {
                "training": self.training_invocations,
                "certificate": self.certificate_invocations,
                "zero_level_evaluation": (
                    self.zero_level_evaluation_invocations
                ),
            },
            "sealed_v18_negative": (
                self.authorization
                .sealed_v18_receipt.canonical_payload()
            ),
            "historical_candidate_retrained": False,
            "historical_candidate_reevaluated": False,
            "historical_outcome_is_candidate_gate": False,
            "same_sign_response_is_gate": False,
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "bounded_gate_passed": self.bounded_gate_passed,
            "formal800_eligible": self.bounded_gate_passed,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "performance_claim_supported": False,
        }

    @property
    def result_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_coverage_state_cmif_uscope_bounded_400(
    authorization: CoverageStateUSCOPEBoundedRunAuthorization,
    model_config: CoverageStateCenteredMixedInteractionConfig,
    *,
    device: torch.device | str,
) -> CoverageStateUSCOPEBoundedRunResult:
    """Run training -> certificate -> zero-level evaluation exactly once."""

    if not isinstance(
        authorization,
        CoverageStateUSCOPEBoundedRunAuthorization,
    ):
        raise TypeError("authorization must be a USCOPE authorization")
    if torch.device(device) != torch.device("cuda:0"):
        raise PermissionError(
            "USCOPE bounded-400 is frozen to visible cuda:0"
        )
    authorization.verify_model_config(model_config)
    preflight = authorization.preflight
    authorization.verify_for_run(
        cache=preflight.population.cache,
        schedule=preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
    )
    training_invocations = 0
    certificate_invocations = 0
    evaluation_invocations = 0
    with _deterministic_execution(device):
        training_invocations += 1
        training = train_matched_coverage_state_cmif_uscope_objectives(
            model_config,
            preflight.population.cache,
            preflight.schedule,
            config=CoverageStateMatchedTrainingConfig(
                seed=COVERAGE_STATE_BOUNDED_SEED
            ),
            device=device,
            authorization=authorization,
        )
        if (
            tuple(value.objective for value in training.results)
            != ("uscope_joint",)
            or tuple(name for name, _ in training.models)
            != ("uscope_joint",)
            or type(training.models[0][1])
            is not CURELiteCenteredMixedInteractionLevelSet
        ):
            raise RuntimeError(
                "USCOPE training returned a non-singleton model"
            )
        model = training.models[0][1].eval()
        certificate_invocations += 1
        certificate = audit_coverage_state_uscope_pair_certificate(
            model,
            preflight.population.cache,
            device=device,
            pair_batch_size=COVERAGE_STATE_USCOPE_CERTIFICATE_PAIR_COUNT,
        )
        evaluation_invocations += 1
        diagnostic = evaluate_coverage_state_zero_level_checkpoint(
            model,
            preflight.population.cache,
            device=device,
            config=CoverageStateZeroLevelEvaluationConfig(
                input_representation=(
                    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                )
            ),
        )
        decision = decide_coverage_state_uscope_zero_level(diagnostic)
    result = CoverageStateUSCOPEBoundedRunResult(
        authorization=authorization,
        training=training,
        certificate=certificate,
        diagnostic=diagnostic,
        decision=decision,
        training_invocations=training_invocations,
        certificate_invocations=certificate_invocations,
        zero_level_evaluation_invocations=evaluation_invocations,
        checks=_uscope_bounded_result_checks(
            authorization,
            training,
            certificate,
            diagnostic,
            decision,
            training_invocations=training_invocations,
            certificate_invocations=certificate_invocations,
            zero_level_evaluation_invocations=evaluation_invocations,
        ),
    )
    result.verify_unchanged()
    return result


__all__ = [
    "COVERAGE_STATE_USCOPE_BOUNDED_AUTHORIZATION_SCHEMA",
    "COVERAGE_STATE_USCOPE_BOUNDED_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_USCOPE_BOUNDED_RESULT_SCHEMA",
    "COVERAGE_STATE_USCOPE_HISTORICAL_CACHE_FINGERPRINT",
    "COVERAGE_STATE_USCOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT",
    "COVERAGE_STATE_USCOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT",
    "COVERAGE_STATE_USCOPE_HISTORICAL_OPTIMIZER_FINGERPRINT",
    "COVERAGE_STATE_USCOPE_HISTORICAL_SCHEDULE_FINGERPRINT",
    "CoverageStateUSCOPEBoundedRunAuthorization",
    "CoverageStateUSCOPEBoundedRunResult",
    "expected_coverage_state_uscope_config",
    "prepare_coverage_state_uscope_bounded_run_authorization",
    "run_coverage_state_cmif_uscope_bounded_400",
    "verify_current_sealed_v18_negative",
]
