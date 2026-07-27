"""Singleton PMOPE bounded-400 protocol for the frozen CMIF field.

The v18 run trains and evaluates exactly one ``pmope_joint`` candidate.
The three completed v17 runs are consumed only through their read-only sealed
receipt.  They are not loaded, retrained, reevaluated, or used as candidate
gates.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from ..coverage_state_sobolev import CSLF_PMOPE_POLICY
from ..frozen_base import module_state_fingerprint
from ..paired_types import tensor_content_fingerprint
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
    COVERAGE_STATE_CMIF_FORMAL_WIDTH,
)
from .coverage_state_pmope_dataset_free import (
    COVERAGE_STATE_PMOPE_MARGIN,
    CoverageStatePMOPEDatasetFreeReceipt,
)
from .coverage_state_pmope_dr_gate import (
    COVERAGE_STATE_PMOPE_DR_GATE_SCHEMA,
    CoverageStatePMOPEDRGateReceipt,
)
from .coverage_state_pmope_sealed_v17 import (
    COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES,
    COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH,
    COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_REPO_PATH,
    COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_REPO_PATH,
    CoverageStatePMOPESealedV17Receipt,
    verify_coverage_state_pmope_sealed_v17_controls,
)
from .coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    CoverageStateMatchedTrainingConfig,
    CoverageStateMatchedTrainingResult,
    CoverageStateRunAuthorization,
    coverage_state_model_fingerprint,
    train_matched_coverage_state_cmif_pmope_objectives,
)
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
    evaluate_coverage_state_zero_level_checkpoint,
)
from ..train.coverage_state_fused_step import (
    COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES,
    CoverageStatePairObjective,
)


COVERAGE_STATE_PMOPE_BOUNDED_AUTHORIZATION_SCHEMA = (
    "cure-lite-cmif-v18-pmope-bounded-run-authorization-v1"
)
COVERAGE_STATE_PMOPE_BOUNDED_RESULT_SCHEMA = (
    "cure-lite-cmif-v18-pmope-bounded-run-result-v1"
)
COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT = 64064
COVERAGE_STATE_PMOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT = (
    "a4086bcffba4035984a8c334b3fa194910bcb7376a573f7f96ef8d36e097240d"
)
COVERAGE_STATE_PMOPE_HISTORICAL_SCHEDULE_FINGERPRINT = (
    "641699803ee6d0472e447c3d4150b7fbe02bc20160d2707658b0d90a175c3c9c"
)
COVERAGE_STATE_PMOPE_HISTORICAL_CACHE_FINGERPRINT = (
    "c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca00754c206092aaf66d8"
)
COVERAGE_STATE_PMOPE_HISTORICAL_OPTIMIZER_FINGERPRINT = (
    "2d058b1cad606e3c1b723aab05925efb2e873c2b3bf021aeaf0f7df40e0690f0"
)
COVERAGE_STATE_PMOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT = (
    "76ed2f94b4187154bad62896b93d637f131865f2e4e8dad38becb4bebc71119f"
)
COVERAGE_STATE_PMOPE_BOUNDED_IMPLEMENTATION_PATHS = tuple(
    dict.fromkeys(
        (
            *COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
            "cure_lite/coverage_state_phase_preserving.py",
            "cure_lite/coverage_state_centered_mixed_interaction.py",
            "cure_lite/experiment/coverage_state_pmope_dataset_free.py",
            "cure_lite/experiment/coverage_state_pmope_dr_gate.py",
            "cure_lite/experiment/coverage_state_pmope_sealed_v17.py",
            "cure_lite/experiment/coverage_state_pmope_bounded_runner.py",
            "tools/audit_coverage_state_cmif_pmope_v18.py",
        )
    )
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _current_implementation_binding() -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_PMOPE_BOUNDED_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"PMOPE bounded implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _cmif_model_config_payload(
    config: CoverageStateCenteredMixedInteractionConfig,
) -> dict[str, object]:
    if type(config) is not CoverageStateCenteredMixedInteractionConfig:
        raise TypeError("PMOPE requires the exact CMIF config class")
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
        "input_representation": config.input_representation,
        "interaction_policy": config.interaction_policy,
        "energy_policy": config.energy_policy,
        "coarse_radius": config.coarse_radius,
        "neutral_phase_hex": config.neutral_phase.hex(),
        "phase_occupancy_channels": (
            config.phase_occupancy_channels
        ),
        "expected_parameter_count": config.expected_parameter_count,
        "model_class": "CURELiteCenteredMixedInteractionLevelSet",
    }


def _dr_gate_model_config_payload(
    config: CoverageStateCenteredMixedInteractionConfig,
) -> dict[str, object]:
    """Project the CMIF config onto the frozen real-D_R receipt contract."""

    return {
        "feature_channels": config.feature_channels,
        "feature_stride": config.feature_stride,
        "width": config.width,
        "parameter_count": config.expected_parameter_count,
        "field_amplitude_hex": config.field_amplitude.hex(),
        "fixed_margin_hex": COVERAGE_STATE_PMOPE_MARGIN.hex(),
        "objective_policy": CSLF_PMOPE_POLICY,
    }


def _common_initial_model_fingerprints(
    config: CoverageStateCenteredMixedInteractionConfig,
) -> tuple[str, str]:
    """Return training-side and D_R-side hashes of one seed-42 CMIF state."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(COVERAGE_STATE_BOUNDED_SEED)
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


def expected_coverage_state_pmope_config(
    preflight: CoverageStateBoundedPreflight,
) -> CoverageStateCenteredMixedInteractionConfig:
    """Return the only CMIF configuration allowed for v18."""

    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    return CoverageStateCenteredMixedInteractionConfig(
        feature_channels=COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_CMIF_FORMAL_WIDTH,
    )


def verify_current_sealed_v17_controls(
) -> CoverageStatePMOPESealedV17Receipt:
    """Verify the in-repository v17 evidence without executing a model."""

    root = _repository_root()
    return verify_coverage_state_pmope_sealed_v17_controls(
        root / COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH,
        source_manifest_path=(
            root / COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_REPO_PATH
        ),
        source_archive_path=(
            root / COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_REPO_PATH
        ),
    )


def _dr_gate_binding_payload(
    receipt: CoverageStatePMOPEDRGateReceipt,
    *,
    dataset_free_receipt_fingerprint: str,
    canonical_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(receipt, CoverageStatePMOPEDRGateReceipt):
        raise TypeError(
            "dr_gate_receipt must be CoverageStatePMOPEDRGateReceipt"
        )
    payload = (
        receipt.canonical_payload()
        if canonical_payload is None
        else dict(canonical_payload)
    )
    receipt_fingerprint = stable_fingerprint(payload)
    checks = dict(receipt.checks)
    mass_rows = receipt.geometry.get("mass_rows")
    if (
        payload.get("schema_version")
        != COVERAGE_STATE_PMOPE_DR_GATE_SCHEMA
        or payload.get("evidence_fingerprint")
        != receipt.evidence_fingerprint
        or payload.get("checks") != checks
        or payload.get("all_pass") is not True
        or receipt.dataset_free_receipt_fingerprint
        != dataset_free_receipt_fingerprint
        or not checks
        or not all(checks.values())
        or len(receipt_fingerprint) != 64
        or receipt.initial_model_fingerprint
        != receipt.final_model_fingerprint
        or not receipt.gradient_rows
        or not isinstance(mass_rows, list)
        or not mass_rows
    ):
        raise PermissionError("PMOPE D_R gate did not pass")
    return {
        "receipt_fingerprint": receipt_fingerprint,
        "evidence_fingerprint": receipt.evidence_fingerprint,
        "all_pass": True,
        "dataset_free_receipt_fingerprint": (
            receipt.dataset_free_receipt_fingerprint
        ),
        "real_inputs_build_fingerprint": (
            receipt.real_inputs_build_fingerprint
        ),
        "source_binding_fingerprint": (
            receipt.source_binding_fingerprint
        ),
        "bounded_population_fingerprint": (
            receipt.bounded_population_fingerprint
        ),
        "bounded_cache_fingerprint": receipt.bounded_cache_fingerprint,
        "sealed_v17_receipt_fingerprint": (
            receipt.v17_binding_fingerprint
        ),
        "execution_seed": receipt.execution_seed,
        "model_config_fingerprint": stable_fingerprint(
            receipt.model_config_payload
        ),
        "initial_model_fingerprint": (
            receipt.initial_model_fingerprint
        ),
        "gradient_rows_fingerprint": stable_fingerprint(
            list(receipt.gradient_rows)
        ),
        "mass_rows_fingerprint": stable_fingerprint(
            mass_rows
        ),
        "checks_fingerprint": stable_fingerprint(checks),
    }


def _verify_bound_dr_gate_lightweight(
    receipt: CoverageStatePMOPEDRGateReceipt,
    binding: tuple[tuple[str, object], ...],
) -> None:
    """Detect in-process receipt drift without rerunning the real-D_R audit."""

    values = dict(binding)
    mass_rows = receipt.geometry.get("mass_rows")
    checks = dict(receipt.checks)
    if (
        stable_fingerprint(receipt._evidence_payload())
        != receipt.evidence_fingerprint
        or values.get("evidence_fingerprint")
        != receipt.evidence_fingerprint
        or values.get("all_pass") is not True
        or not checks
        or not all(checks.values())
        or values.get("dataset_free_receipt_fingerprint")
        != receipt.dataset_free_receipt_fingerprint
        or values.get("real_inputs_build_fingerprint")
        != receipt.real_inputs_build_fingerprint
        or values.get("source_binding_fingerprint")
        != receipt.source_binding_fingerprint
        or values.get("bounded_population_fingerprint")
        != receipt.bounded_population_fingerprint
        or values.get("bounded_cache_fingerprint")
        != receipt.bounded_cache_fingerprint
        or values.get("sealed_v17_receipt_fingerprint")
        != receipt.v17_binding_fingerprint
        or values.get("execution_seed") != receipt.execution_seed
        or values.get("model_config_fingerprint")
        != stable_fingerprint(receipt.model_config_payload)
        or values.get("initial_model_fingerprint")
        != receipt.initial_model_fingerprint
        or values.get("gradient_rows_fingerprint")
        != stable_fingerprint(list(receipt.gradient_rows))
        or not isinstance(mass_rows, list)
        or values.get("mass_rows_fingerprint")
        != stable_fingerprint(mass_rows)
        or values.get("checks_fingerprint")
        != stable_fingerprint(checks)
    ):
        raise RuntimeError("bound PMOPE D_R gate receipt changed")


@dataclass(frozen=True, eq=False)
class CoverageStatePMOPEBoundedRunAuthorization(
    CoverageStateRunAuthorization,
):
    """Bind one singleton PMOPE run to every frozen prerequisite."""

    preflight: CoverageStateBoundedPreflight
    dataset_free_receipt: CoverageStatePMOPEDatasetFreeReceipt
    dataset_free_receipt_fingerprint: str
    dr_gate_receipt: CoverageStatePMOPEDRGateReceipt
    dr_gate_binding: tuple[tuple[str, object], ...]
    sealed_v17_receipt: CoverageStatePMOPESealedV17Receipt
    sealed_v17_receipt_fingerprint: str
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
        if not isinstance(self.preflight, CoverageStateBoundedPreflight):
            raise TypeError("preflight must be CoverageStateBoundedPreflight")
        if not isinstance(
            self.dataset_free_receipt,
            CoverageStatePMOPEDatasetFreeReceipt,
        ):
            raise TypeError(
                "dataset_free_receipt must be "
                "CoverageStatePMOPEDatasetFreeReceipt"
            )
        if not isinstance(
            self.sealed_v17_receipt,
            CoverageStatePMOPESealedV17Receipt,
        ):
            raise TypeError(
                "sealed_v17_receipt must be "
                "CoverageStatePMOPESealedV17Receipt"
            )
        self.preflight.verify_unchanged()
        self.dataset_free_receipt.verify_unchanged()
        expected_config = expected_coverage_state_pmope_config(
            self.preflight
        )
        runtime_initial, dr_initial = _common_initial_model_fingerprints(
            expected_config
        )
        _verify_bound_dr_gate_lightweight(
            self.dr_gate_receipt,
            self.dr_gate_binding,
        )
        current_sealed = verify_current_sealed_v17_controls()
        if (
            self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or not self.dataset_free_receipt.all_pass
            or self.sealed_v17_receipt.receipt_fingerprint
            != self.sealed_v17_receipt_fingerprint
            or current_sealed.receipt_fingerprint
            != self.sealed_v17_receipt_fingerprint
            or current_sealed.canonical_payload()
            != self.sealed_v17_receipt.canonical_payload()
            or tuple(
                control.objective
                for control in self.sealed_v17_receipt.controls
            )
            != COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES
            or self.implementation_binding
            != _current_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != stable_fingerprint(
                _cmif_model_config_payload(expected_config)
            )
            or runtime_initial
            != COVERAGE_STATE_PMOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            or self.expected_parameter_count
            != COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT
            or self.expected_parameter_count
            != expected_config.expected_parameter_count
            or self.objective_suite != ("pmope_joint",)
            or self.candidate_objective
            != CoverageStatePairObjective.PMOPE_JOINT.value
            or self.candidate_objective_policy != CSLF_PMOPE_POLICY
            or self.fixed_margin != COVERAGE_STATE_PMOPE_MARGIN
            or self.coverage_policy
            != CSLF_PHASE_PRESERVING_COVERAGE_POLICY
            or self.interaction_policy != CMIF_INTERACTION_POLICY
            or self.energy_policy != CMIF_ENERGY_POLICY
            or self.preflight.schedule.config.seed
            != COVERAGE_STATE_BOUNDED_SEED
            or self.preflight.schedule.config.epochs
            != COVERAGE_STATE_BOUNDED_EPOCHS
            or self.preflight.schedule.config.steps_per_epoch
            != COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
            or self.preflight.schedule.config.updates
            != COVERAGE_STATE_BOUNDED_UPDATES
            or self.preflight.schedule.schedule_fingerprint
            != COVERAGE_STATE_PMOPE_HISTORICAL_SCHEDULE_FINGERPRINT
            or self.preflight.population.bounded_cache_fingerprint
            != COVERAGE_STATE_PMOPE_HISTORICAL_CACHE_FINGERPRINT
            or dict(self.dr_gate_binding)[
                "bounded_population_fingerprint"
            ]
            != self.preflight.population.population_fingerprint
            or dict(self.dr_gate_binding)["bounded_cache_fingerprint"]
            != self.preflight.population.bounded_cache_fingerprint
            or dict(self.dr_gate_binding)[
                "sealed_v17_receipt_fingerprint"
            ]
            != self.sealed_v17_receipt_fingerprint
            or dict(self.dr_gate_binding)["execution_seed"]
            != COVERAGE_STATE_BOUNDED_SEED
            or dict(self.dr_gate_binding)["model_config_fingerprint"]
            != stable_fingerprint(
                _dr_gate_model_config_payload(expected_config)
            )
            or dict(self.dr_gate_binding)["initial_model_fingerprint"]
            != dr_initial
        ):
            raise ValueError("PMOPE bounded authorization binding changed")

    @property
    def training_authorized(self) -> bool:
        return (
            self.preflight.training_authorized
            and self.dataset_free_receipt.all_pass
            and dict(self.dr_gate_binding).get("all_pass") is True
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                COVERAGE_STATE_PMOPE_BOUNDED_AUTHORIZATION_SCHEMA
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
                "seed": 42,
                "epochs": 10,
                "steps_per_epoch": 40,
                "updates": 400,
                "objectives": 1,
            },
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "dr_gate": dict(self.dr_gate_binding),
            "sealed_v17_receipt": (
                self.sealed_v17_receipt.canonical_payload()
            ),
            "sealed_v17_receipt_fingerprint": (
                self.sealed_v17_receipt_fingerprint
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
                    COVERAGE_STATE_PMOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT
                ),
                "optimizer_fingerprint": (
                    COVERAGE_STATE_PMOPE_HISTORICAL_OPTIMIZER_FINGERPRINT
                ),
                "device_cache_fingerprint": (
                    COVERAGE_STATE_PMOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT
                ),
            },
            "checks": {
                "preflight_passed": self.preflight.training_authorized,
                "dataset_free_gate_passed": (
                    self.dataset_free_receipt.all_pass
                ),
                "D_R_gate_passed": (
                    dict(self.dr_gate_binding).get("all_pass") is True
                ),
                "D_R_gate_receipt_bound": (
                    dict(self.dr_gate_binding).get("all_pass") is True
                ),
                "sealed_v17_bound_read_only": True,
                "singleton_candidate": (
                    self.objective_suite == ("pmope_joint",)
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
        _verify_bound_dr_gate_lightweight(
            self.dr_gate_receipt,
            self.dr_gate_binding,
        )
        current_sealed = verify_current_sealed_v17_controls()
        self.preflight.verify_unchanged()
        self.dataset_free_receipt.verify_unchanged()
        if (
            current_sealed.receipt_fingerprint
            != self.sealed_v17_receipt_fingerprint
            or current_sealed.canonical_payload()
            != self.sealed_v17_receipt.canonical_payload()
            or self.implementation_binding
            != _current_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or stable_fingerprint(self.canonical_payload())
            != self.authorization_fingerprint
        ):
            raise RuntimeError(
                "PMOPE bounded authorization changed after creation"
            )

    def verify_model_config(
        self,
        model_config: CoverageStateCenteredMixedInteractionConfig,
    ) -> None:
        if (
            type(model_config)
            is not CoverageStateCenteredMixedInteractionConfig
            or stable_fingerprint(
                _cmif_model_config_payload(model_config)
            )
            != self.model_config_fingerprint
            or model_config.expected_parameter_count
            != self.expected_parameter_count
        ):
            raise PermissionError(
                "PMOPE authorization does not permit this model config"
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
                "PMOPE authorization does not permit this training run"
            )


def prepare_coverage_state_pmope_bounded_run_authorization(
    preflight: CoverageStateBoundedPreflight,
    dataset_free_receipt: CoverageStatePMOPEDatasetFreeReceipt,
    dr_gate_receipt: CoverageStatePMOPEDRGateReceipt,
    *,
    sealed_v17_receipt: CoverageStatePMOPESealedV17Receipt | None = None,
    dr_gate_canonical_payload: Mapping[str, object] | None = None,
) -> CoverageStatePMOPEBoundedRunAuthorization:
    """Bind all prerequisites to the only permitted PMOPE run."""

    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
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
    sealed = (
        verify_current_sealed_v17_controls()
        if sealed_v17_receipt is None
        else sealed_v17_receipt
    )
    current_sealed = verify_current_sealed_v17_controls()
    if (
        current_sealed.receipt_fingerprint
        != sealed.receipt_fingerprint
        or current_sealed.canonical_payload()
        != sealed.canonical_payload()
    ):
        raise RuntimeError("provided sealed v17 receipt changed")
    implementation = _current_implementation_binding()
    model_config = expected_coverage_state_pmope_config(preflight)
    result = CoverageStatePMOPEBoundedRunAuthorization(
        preflight=preflight,
        dataset_free_receipt=dataset_free_receipt,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt.receipt_fingerprint
        ),
        dr_gate_receipt=dr_gate_receipt,
        dr_gate_binding=dr_binding,
        sealed_v17_receipt=sealed,
        sealed_v17_receipt_fingerprint=sealed.receipt_fingerprint,
        implementation_binding=implementation,
        implementation_fingerprint=stable_fingerprint(
            dict(implementation)
        ),
        model_config_fingerprint=stable_fingerprint(
            _cmif_model_config_payload(model_config)
        ),
        expected_parameter_count=model_config.expected_parameter_count,
        objective_suite=tuple(
            value.value
            for value in COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES
        ),
        candidate_objective=CoverageStatePairObjective.PMOPE_JOINT.value,
        candidate_objective_policy=CSLF_PMOPE_POLICY,
        fixed_margin=COVERAGE_STATE_PMOPE_MARGIN,
        coverage_policy=CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
        interaction_policy=CMIF_INTERACTION_POLICY,
        energy_policy=CMIF_ENERGY_POLICY,
    )
    result.verify_unchanged()
    return result


def _pmope_bounded_result_checks(
    authorization: CoverageStatePMOPEBoundedRunAuthorization,
    training: CoverageStateMatchedTrainingResult,
    diagnostic: CoverageStateZeroLevelEvaluationResult,
) -> tuple[tuple[str, bool], ...]:
    """Check one PMOPE candidate without consuming historical outcomes."""

    results = training.results
    models = training.models
    result = results[0] if len(results) == 1 else None
    model_entry = models[0] if len(models) == 1 else None
    model = model_entry[1] if model_entry is not None else None
    latency = (
        dict(result.first_nonzero_gradient_update)
        if result is not None
        else {}
    )
    checks = {
        "authorization": authorization.training_authorized,
        "singleton_pmope_candidate": (
            result is not None
            and model_entry is not None
            and len(results) == len(models) == 1
            and result.objective == "pmope_joint"
            and result.objective_policy == CSLF_PMOPE_POLICY
            and model_entry[0] == "pmope_joint"
        ),
        "fixed_seed_and_budget": (
            result is not None
            and result.seed == 42
            and result.epochs == 10
            and result.steps_per_epoch == 40
            and result.completed_updates == 400
            and result.forward_calls == 400
            and result.backward_calls == 400
            and result.optimizer_steps == 400
            and result.logical_state_evaluations == 4800
            and result.finite_state_audits == 401
        ),
        "historical_training_coordinates": (
            result is not None
            and training.common_initial_model_fingerprint
            == COVERAGE_STATE_PMOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            and result.initial_model_fingerprint
            == COVERAGE_STATE_PMOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            and training.schedule_fingerprint
            == COVERAGE_STATE_PMOPE_HISTORICAL_SCHEDULE_FINGERPRINT
            and result.schedule_fingerprint
            == COVERAGE_STATE_PMOPE_HISTORICAL_SCHEDULE_FINGERPRINT
            and training.cache_fingerprint
            == COVERAGE_STATE_PMOPE_HISTORICAL_CACHE_FINGERPRINT
            and result.cache_fingerprint
            == COVERAGE_STATE_PMOPE_HISTORICAL_CACHE_FINGERPRINT
            and result.optimizer_config_fingerprint
            == COVERAGE_STATE_PMOPE_HISTORICAL_OPTIMIZER_FINGERPRINT
            and result.device_cache_fingerprint
            == COVERAGE_STATE_PMOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT
        ),
        "cmif_gradient_latency": latency
        == {
            "joint_hidden_bias": 1,
            "joint_state_weight": 1,
            "scalar_energy_weight": 0,
        },
        "exact_cmif_model": (
            model is not None
            and type(model)
            is CURELiteCenteredMixedInteractionLevelSet
            and stable_fingerprint(
                _cmif_model_config_payload(model.config)
            )
            == authorization.model_config_fingerprint
            and sum(
                parameter.numel() for parameter in model.parameters()
            )
            == 64064
        ),
        "diagnostic_checkpoint_binding": (
            model is not None
            and diagnostic.checkpoint_fingerprint
            == module_state_fingerprint(model)
        ),
        "diagnostic_D_R_binding": (
            diagnostic.cache_fingerprint
            == authorization.preflight.population.bounded_cache_fingerprint
            and diagnostic.dataset
            == authorization.preflight.population.cache.raw_catalog.dataset
            and diagnostic.split == "D_R"
        ),
        "candidate_seven_zero_level_gates": (
            diagnostic.factual_miss_gate_passed
            and diagnostic.factual_no_miss_gate_passed
            and diagnostic.clean_defined_metrics_passed
            and diagnostic.clean_compact_support_gate_passed
            and diagnostic.component_null_gate_passed
            and diagnostic.identity_null_gate_passed
            and diagnostic.diagnostic_null_gate_passed
            and diagnostic.bounded_gate_passed
        ),
        "threshold_zero_without_search": (
            diagnostic.config.residual_threshold == 0.0
            and not diagnostic.config.threshold_search_performed
            and diagnostic.config.input_representation
            == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        ),
        "diagnostic_read_only": (
            diagnostic.backward_calls == 0
            and diagnostic.optimizer_steps == 0
            and not diagnostic.config.d_v_accessed
            and not diagnostic.config.d_t_accessed
        ),
        "sealed_controls_are_non_gating": (
            tuple(
                control.objective
                for control in authorization.sealed_v17_receipt.controls
            )
            == COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES
            and authorization.sealed_v17_receipt.canonical_payload()[
                "control_outcomes_are_not_candidate_gates"
            ]
            is True
        ),
        "no_resume_or_retry": True,
        "formal800_not_executed": True,
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True, eq=False)
class CoverageStatePMOPEBoundedRunResult:
    """One PMOPE checkpoint and one read-only D_R evaluation."""

    authorization: CoverageStatePMOPEBoundedRunAuthorization
    training: CoverageStateMatchedTrainingResult
    diagnostic: CoverageStateZeroLevelEvaluationResult
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.authorization,
                CoverageStatePMOPEBoundedRunAuthorization,
            )
            or tuple(value.objective for value in self.training.results)
            != ("pmope_joint",)
            or tuple(name for name, _ in self.training.models)
            != ("pmope_joint",)
            or self.checks != tuple(sorted(self.checks))
            or len({name for name, _ in self.checks})
            != len(self.checks)
        ):
            raise ValueError("PMOPE bounded result is incomplete")

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
        expected = _pmope_bounded_result_checks(
            self.authorization,
            self.training,
            self.diagnostic,
        )
        if self.checks != expected:
            raise RuntimeError("PMOPE bounded result checks changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_PMOPE_BOUNDED_RESULT_SCHEMA,
            "runtime_splits": ["D_R"],
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "candidate_objective": "pmope_joint",
            "candidate_objective_policy": CSLF_PMOPE_POLICY,
            "fixed_margin_hex": COVERAGE_STATE_PMOPE_MARGIN.hex(),
            "training": self.training.canonical_payload(),
            "candidate_diagnostic": self.diagnostic.canonical_payload(),
            "sealed_v17_controls": (
                self.authorization
                .sealed_v17_receipt.canonical_payload()
            ),
            "historical_controls_retrained": False,
            "historical_controls_reevaluated": False,
            "historical_control_outcomes_are_candidate_gates": False,
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


def run_coverage_state_cmif_pmope_bounded_400(
    authorization: CoverageStatePMOPEBoundedRunAuthorization,
    model_config: CoverageStateCenteredMixedInteractionConfig,
    *,
    device: torch.device | str,
) -> CoverageStatePMOPEBoundedRunResult:
    """Train and evaluate only the fixed seed-42 PMOPE candidate."""

    if not isinstance(
        authorization,
        CoverageStatePMOPEBoundedRunAuthorization,
    ):
        raise TypeError("authorization must be PMOPE bounded authorization")
    if torch.device(device) != torch.device("cuda:0"):
        raise PermissionError(
            "PMOPE bounded-400 is frozen to visible cuda:0"
        )
    authorization.verify_model_config(model_config)
    preflight = authorization.preflight
    authorization.verify_for_run(
        cache=preflight.population.cache,
        schedule=preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
    )
    with _deterministic_execution(device):
        training = train_matched_coverage_state_cmif_pmope_objectives(
            model_config,
            preflight.population.cache,
            preflight.schedule,
            config=CoverageStateMatchedTrainingConfig(seed=42),
            device=device,
            authorization=authorization,
        )
        if (
            tuple(value.objective for value in training.results)
            != ("pmope_joint",)
            or tuple(name for name, _ in training.models)
            != ("pmope_joint",)
            or type(training.models[0][1])
            is not CURELiteCenteredMixedInteractionLevelSet
        ):
            raise RuntimeError(
                "PMOPE training returned a non-singleton model"
            )
        diagnostic = evaluate_coverage_state_zero_level_checkpoint(
            training.models[0][1].eval(),
            preflight.population.cache,
            device=device,
            config=CoverageStateZeroLevelEvaluationConfig(
                input_representation=(
                    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                )
            ),
        )
    result = CoverageStatePMOPEBoundedRunResult(
        authorization=authorization,
        training=training,
        diagnostic=diagnostic,
        checks=_pmope_bounded_result_checks(
            authorization,
            training,
            diagnostic,
        ),
    )
    result.verify_unchanged()
    return result


__all__ = [
    "COVERAGE_STATE_PMOPE_BOUNDED_AUTHORIZATION_SCHEMA",
    "COVERAGE_STATE_PMOPE_BOUNDED_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_PMOPE_BOUNDED_RESULT_SCHEMA",
    "COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT",
    "COVERAGE_STATE_PMOPE_HISTORICAL_CACHE_FINGERPRINT",
    "COVERAGE_STATE_PMOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT",
    "COVERAGE_STATE_PMOPE_HISTORICAL_INITIAL_MODEL_FINGERPRINT",
    "COVERAGE_STATE_PMOPE_HISTORICAL_OPTIMIZER_FINGERPRINT",
    "COVERAGE_STATE_PMOPE_HISTORICAL_SCHEDULE_FINGERPRINT",
    "CoverageStatePMOPEBoundedRunAuthorization",
    "CoverageStatePMOPEBoundedRunResult",
    "expected_coverage_state_pmope_config",
    "prepare_coverage_state_pmope_bounded_run_authorization",
    "run_coverage_state_cmif_pmope_bounded_400",
    "verify_current_sealed_v17_controls",
]
