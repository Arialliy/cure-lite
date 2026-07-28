"""Single-candidate bounded-400 protocol for v21 PAET-BFA + PMOPE.

The runner preserves the seed-42 population, cache, schedule, optimizer,
objective, and zero-threshold evaluation.  The only permitted model-side
difference from v20 is the predeclared phase-aligned evidence transport
inside the shared BFA energy.

The official run identifier is an explicit, fingerprinted input at every
layer.  This intentionally prevents the historical v20 defect in which the
CLI and result payload used different run identifiers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Callable, Mapping, TypeVar

import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_phase_aligned_evidence_transport import (
    CSLF_PAET_EQUATION_POLICY,
    CSLF_PAET_FIELD_POLICY,
    CSLF_PAET_FLIP_POLICY,
    CSLF_PAET_TRANSPORT_POLICY,
    PAET_INPUT_REPRESENTATION,
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from ..coverage_state_precomputed_cache import CoverageStateScalarCache
from ..coverage_state_schedule import CoverageStateTrainingSchedule
from ..coverage_state_sobolev import CSLF_PMOPE_POLICY
from ..frozen_base import module_state_fingerprint
from ..paired_types import tensor_content_fingerprint
from ..train.coverage_state_fused_step import (
    COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES,
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
from .coverage_state_paet_certificate import (
    COVERAGE_STATE_PAET_CERTIFICATE_DEFAULT_PAIR_BATCH_SIZE,
    COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT,
    CoverageStatePAETCertificateReceipt,
    audit_coverage_state_paet_pair_certificate,
)
from .coverage_state_paet_dataset_free import (
    COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_PAET_FORMAL_WIDTH,
    COVERAGE_STATE_PAET_MARGIN,
    CoverageStatePAETDatasetFreeReceipt,
)
from .coverage_state_paet_decision import (
    COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    CoverageStatePAETBoundedDecision,
    decide_coverage_state_paet_bounded,
)
from .coverage_state_paet_dr_gate import (
    COVERAGE_STATE_PAET_DR_EXECUTION_SEED,
    COVERAGE_STATE_PAET_DR_GATE_SCHEMA,
    CoverageStatePAETDRGateReceipt,
)
from .coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    CoverageStateMatchedTrainingConfig,
    CoverageStateMatchedTrainingResult,
    CoverageStateRunAuthorization,
    coverage_state_model_fingerprint,
    train_matched_coverage_state_paet_bfa_pmope_objectives,
)
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
    evaluate_coverage_state_zero_level_checkpoint,
)


COVERAGE_STATE_PAET_OFFICIAL_RUN_ID = COVERAGE_STATE_PAET_BOUNDED_RUN_ID
COVERAGE_STATE_PAET_BOUNDED_AUTHORIZATION_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-bounded-run-authorization-v1"
)
COVERAGE_STATE_PAET_BOUNDED_RESULT_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-bounded-run-result-v1"
)
COVERAGE_STATE_PAET_RESOURCE_MEASUREMENT_SCHEMA = (
    "cure-lite-paet-bfa-v21-training-resource-measurement-v1"
)
COVERAGE_STATE_PAET_V20_REFERENCE_SCHEMA = (
    "cure-lite-paet-bfa-v21-sealed-v20-reference-v1"
)
COVERAGE_STATE_PAET_V20_REFERENCE_RUN_ID = (
    "cure_lite_bfa_cmif_v20_pmope_bounded_400_r2"
)
COVERAGE_STATE_PAET_V20_INTERNAL_RESULT_RUN_ID = (
    "cure_lite_bfa_cmif_v20_pmope_bounded_400_r1"
)
COVERAGE_STATE_PAET_V20_COMPLETE_FINGERPRINT = (
    "8908a8c1896951e46fd737aa6f7fef2c9935e6524632b3576b8069faa026e2eb"
)
COVERAGE_STATE_PAET_V20_COMPLETE_FILE_SHA256 = (
    "a1307929615ef877726387df024c90de33b54924ea2c22de2c7f7f5a51e7f334"
)
COVERAGE_STATE_PAET_V20_BOUNDED_RESULT_FILE_SHA256 = (
    "1cb07c53fdb0ca671eaa25e34f8ed90de0f6cc5050da450bfb2e742b085c7a5e"
)
COVERAGE_STATE_PAET_V20_ZERO_LEVEL_FILE_SHA256 = (
    "fe9820d72fc796aa0e70045c48d3b40eb05e0aba2bb5a3447bff3830e9cfadc4"
)
COVERAGE_STATE_PAET_V20_RESOURCE_COMPARISON_STATUS = (
    "NOT_EVALUATED_NO_MATCHED_V20_MEASUREMENT"
)
COVERAGE_STATE_PAET_MEMORY_RATIO_LIMIT = (7, 4)
COVERAGE_STATE_PAET_STEP_TIME_RATIO_LIMIT = (2, 1)
COVERAGE_STATE_PAET_HISTORICAL_INITIAL_MODEL_FINGERPRINT = (
    "a4086bcffba4035984a8c334b3fa194910bcb7376a573f7f96ef8d36e097240d"
)
COVERAGE_STATE_PAET_HISTORICAL_RAW_INITIAL_STATE_FINGERPRINT = (
    "5e68cbbae365f9427924a722b08c490e2bd2bfb9cf709832ca81a10c1506bdea"
)
COVERAGE_STATE_PAET_HISTORICAL_SCHEDULE_FINGERPRINT = (
    "641699803ee6d0472e447c3d4150b7fbe02bc20160d2707658b0d90a175c3c9c"
)
COVERAGE_STATE_PAET_HISTORICAL_CACHE_FINGERPRINT = (
    "c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca00754c206092aaf66d8"
)
COVERAGE_STATE_PAET_HISTORICAL_OPTIMIZER_FINGERPRINT = (
    "2d058b1cad606e3c1b723aab05925efb2e873c2b3bf021aeaf0f7df40e0690f0"
)
COVERAGE_STATE_PAET_HISTORICAL_DEVICE_CACHE_FINGERPRINT = (
    "76ed2f94b4187154bad62896b93d637f131865f2e4e8dad38becb4bebc71119f"
)
COVERAGE_STATE_PAET_BOUNDED_IMPLEMENTATION_PATHS = tuple(
    dict.fromkeys(
        (
            *COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
            "cure_lite/coverage_state_phase_preserving.py",
            "cure_lite/coverage_state_binary_flip_antisymmetric.py",
            "cure_lite/coverage_state_phase_aligned_evidence_transport.py",
            "cure_lite/coverage_state_sobolev.py",
            "cure_lite/train/coverage_state_fused_step.py",
            "cure_lite/experiment/coverage_state_training.py",
            "cure_lite/experiment/coverage_state_bfa_certificate.py",
            "cure_lite/experiment/coverage_state_paet_dataset_free.py",
            "cure_lite/experiment/coverage_state_paet_dr_gate.py",
            "cure_lite/experiment/coverage_state_paet_certificate.py",
            "cure_lite/experiment/coverage_state_paet_decision.py",
            "cure_lite/experiment/coverage_state_paet_bounded_runner.py",
        )
    )
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    if run_id != COVERAGE_STATE_PAET_OFFICIAL_RUN_ID:
        raise PermissionError("PAET bounded run_id is not the frozen run")
    return run_id


def _current_implementation_binding() -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_PAET_BOUNDED_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"PAET bounded implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _model_config_payload(
    config: CoverageStatePhaseAlignedEvidenceTransportConfig,
) -> dict[str, object]:
    if type(config) is not CoverageStatePhaseAlignedEvidenceTransportConfig:
        raise TypeError("v21 requires the exact PAET-BFA config")
    return {
        "config": asdict(config),
        "model_class": "CURELitePhaseAlignedEvidenceTransportLevelSet",
        "expected_parameter_count": config.expected_parameter_count,
    }


def _certificate_model_fingerprint(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
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
    config: CoverageStatePhaseAlignedEvidenceTransportConfig,
) -> dict[str, object]:
    return {
        "model_class": (
            "CURELitePhaseAlignedEvidenceTransportLevelSet"
        ),
        "feature_channels": config.feature_channels,
        "feature_stride": config.feature_stride,
        "width": config.width,
        "parameter_count": config.expected_parameter_count,
        "parameter_tensor_count": 3,
        "field_policy": config.field_policy,
        "equation_policy": config.equation_policy,
        "flip_policy": config.flip_policy,
        "transport_policy": config.transport_policy,
        "margin_hex": COVERAGE_STATE_PAET_MARGIN.hex(),
    }


def _common_initial_model_fingerprints(
    config: CoverageStatePhaseAlignedEvidenceTransportConfig,
) -> tuple[str, str]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(
            COVERAGE_STATE_BOUNDED_SEED
        )
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(config)
    return (
        coverage_state_model_fingerprint(model),
        stable_fingerprint(
            {
                name: tensor_content_fingerprint(value)
                for name, value in sorted(model.state_dict().items())
            }
        ),
    )


def expected_coverage_state_paet_config(
    preflight: CoverageStateBoundedPreflight,
) -> CoverageStatePhaseAlignedEvidenceTransportConfig:
    """Return the only v21 bounded model configuration."""

    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    return CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_PAET_FORMAL_WIDTH,
    )


@dataclass(frozen=True)
class CoverageStatePAETV20Reference:
    """Read-only identity and metrics of the completed v20-r2 run."""

    run_id: str
    complete_fingerprint: str
    complete_file_sha256: str
    bounded_result_file_sha256: str
    zero_level_file_sha256: str
    internal_result_run_id: str
    observed: tuple[tuple[str, int], ...]
    measured_resource_reference_available: bool
    resource_comparison_status: str

    def __post_init__(self) -> None:
        expected_observed = (
            ("clean_compact_support_passed", 1),
            ("clean_outside_completion_pixels", 54),
            ("clean_target_negative_pixels", 115),
            ("component_null_passed", 16),
            ("diagnostic_null_passed", 1),
            ("factual_no_miss_passed", 16),
            ("factual_recovered", 16),
            ("factual_strict", 14),
            ("factual_target_negative_pixels", 310),
            ("factual_target_pixels", 335),
            ("identity_null_passed", 16),
            ("invalid_completion_pixels", 0),
        )
        if (
            self.run_id != COVERAGE_STATE_PAET_V20_REFERENCE_RUN_ID
            or self.complete_fingerprint
            != COVERAGE_STATE_PAET_V20_COMPLETE_FINGERPRINT
            or self.complete_file_sha256
            != COVERAGE_STATE_PAET_V20_COMPLETE_FILE_SHA256
            or self.bounded_result_file_sha256
            != COVERAGE_STATE_PAET_V20_BOUNDED_RESULT_FILE_SHA256
            or self.zero_level_file_sha256
            != COVERAGE_STATE_PAET_V20_ZERO_LEVEL_FILE_SHA256
            or self.internal_result_run_id
            != COVERAGE_STATE_PAET_V20_INTERNAL_RESULT_RUN_ID
            or self.observed != expected_observed
            or self.measured_resource_reference_available
            or self.resource_comparison_status
            != COVERAGE_STATE_PAET_V20_RESOURCE_COMPARISON_STATUS
        ):
            raise ValueError("sealed v20 reference changed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_PAET_V20_REFERENCE_SCHEMA,
            "run_id": self.run_id,
            "complete_fingerprint": self.complete_fingerprint,
            "complete_file_sha256": self.complete_file_sha256,
            "bounded_result_file_sha256": (
                self.bounded_result_file_sha256
            ),
            "zero_level_file_sha256": self.zero_level_file_sha256,
            "known_historical_internal_run_id_defect": {
                "internal_result_run_id": self.internal_result_run_id,
                "expected_cli_run_id": self.run_id,
                "propagated_to_v21": False,
            },
            "observed": dict(self.observed),
            "resource_reference": {
                "measured_reference_available": (
                    self.measured_resource_reference_available
                ),
                "working_memory_bytes": None,
                "elapsed_ns": None,
                "ns_per_update": None,
                "comparison_status": self.resource_comparison_status,
                "memory_ratio_limit": {
                    "numerator": (
                        COVERAGE_STATE_PAET_MEMORY_RATIO_LIMIT[0]
                    ),
                    "denominator": (
                        COVERAGE_STATE_PAET_MEMORY_RATIO_LIMIT[1]
                    ),
                },
                "step_time_ratio_limit": {
                    "numerator": (
                        COVERAGE_STATE_PAET_STEP_TIME_RATIO_LIMIT[0]
                    ),
                    "denominator": (
                        COVERAGE_STATE_PAET_STEP_TIME_RATIO_LIMIT[1]
                    ),
                },
                "ratio_claim_supported": False,
            },
            "read_only": True,
            "checkpoint_deserialized": False,
            "training_performed": False,
            "reevaluated": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }

    @cached_property
    def reference_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def verify_repository_coverage_state_bfa_v20_reference(
    root: Path | None = None,
) -> CoverageStatePAETV20Reference:
    """Verify the frozen v20 files as bytes and JSON, without execution."""

    repository = _repository_root() if root is None else Path(root)
    directory = (
        repository
        / "runs/irstd1k_stage_a_seed42"
        / COVERAGE_STATE_PAET_V20_REFERENCE_RUN_ID
    )
    complete_path = directory / "COMPLETE.json"
    bounded_path = directory / "receipts/bounded_result.json"
    zero_path = directory / "receipts/zero_level.json"
    if (
        not complete_path.is_file()
        or complete_path.is_symlink()
        or not bounded_path.is_file()
        or bounded_path.is_symlink()
        or not zero_path.is_file()
        or zero_path.is_symlink()
    ):
        raise FileNotFoundError("sealed v20 reference files are absent")
    if (
        file_sha256(complete_path)
        != COVERAGE_STATE_PAET_V20_COMPLETE_FILE_SHA256
        or file_sha256(bounded_path)
        != COVERAGE_STATE_PAET_V20_BOUNDED_RESULT_FILE_SHA256
        or file_sha256(zero_path)
        != COVERAGE_STATE_PAET_V20_ZERO_LEVEL_FILE_SHA256
    ):
        raise RuntimeError("sealed v20 reference bytes changed")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    bounded = json.loads(bounded_path.read_text(encoding="utf-8"))
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    result = bounded.get("result", {})
    decision = result.get("decision", {})
    observed = decision.get("observed", {})
    natural = zero.get("candidate_diagnostic", {}).get(
        "natural_diagnostics",
        [],
    )
    factual = [
        row
        for row in natural
        if row.get("state_kind") == "factual_miss"
    ]
    values = {
        **observed,
        "factual_target_negative_pixels": sum(
            int(row.get("focus_target_negative_pixels", -10**9))
            for row in factual
        ),
        "factual_target_pixels": sum(
            int(row.get("focus_target_pixels", -10**9))
            for row in factual
        ),
    }
    if (
        complete.get("run_id")
        != COVERAGE_STATE_PAET_V20_REFERENCE_RUN_ID
        or complete.get("complete_fingerprint")
        != COVERAGE_STATE_PAET_V20_COMPLETE_FINGERPRINT
        or complete.get("status") != "complete"
        or complete.get("bounded_gate_passed") is not False
        or complete.get("artifact_files", {}).get(
            "receipts/bounded_result.json"
        )
        != COVERAGE_STATE_PAET_V20_BOUNDED_RESULT_FILE_SHA256
        or result.get("run_id")
        != COVERAGE_STATE_PAET_V20_INTERNAL_RESULT_RUN_ID
        or decision.get("bounded_gate_passed") is not False
        or len(factual) != 16
    ):
        raise RuntimeError("sealed v20 reference semantics changed")
    return CoverageStatePAETV20Reference(
        run_id=COVERAGE_STATE_PAET_V20_REFERENCE_RUN_ID,
        complete_fingerprint=COVERAGE_STATE_PAET_V20_COMPLETE_FINGERPRINT,
        complete_file_sha256=COVERAGE_STATE_PAET_V20_COMPLETE_FILE_SHA256,
        bounded_result_file_sha256=(
            COVERAGE_STATE_PAET_V20_BOUNDED_RESULT_FILE_SHA256
        ),
        zero_level_file_sha256=(
            COVERAGE_STATE_PAET_V20_ZERO_LEVEL_FILE_SHA256
        ),
        internal_result_run_id=(
            COVERAGE_STATE_PAET_V20_INTERNAL_RESULT_RUN_ID
        ),
        observed=tuple(sorted((key, int(value)) for key, value in values.items())),
        measured_resource_reference_available=False,
        resource_comparison_status=(
            COVERAGE_STATE_PAET_V20_RESOURCE_COMPARISON_STATUS
        ),
    )


def _dr_gate_binding_payload(
    receipt: CoverageStatePAETDRGateReceipt,
    *,
    dataset_free_receipt_fingerprint: str,
    canonical_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(receipt, CoverageStatePAETDRGateReceipt):
        raise TypeError("dr_gate_receipt must be a PAET D_R receipt")
    if canonical_payload is None:
        payload = receipt.canonical_payload()
    else:
        receipt.verify_unchanged()
        payload = dict(canonical_payload)
    checks = dict(receipt.checks)
    probe = receipt.probe
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
            receipt.bounded_population.population_fingerprint
        ),
        "bounded_cache_fingerprint": (
            receipt.bounded_population.cache.cache_fingerprint
        ),
        "implementation_fingerprint": stable_fingerprint(
            dict(receipt.implementation_binding)
        ),
        "probe_fingerprint": stable_fingerprint(probe),
        "checks_fingerprint": stable_fingerprint(checks),
        "execution_seed": probe.get("execution_seed"),
        "runtime_splits": probe.get("runtime_splits"),
        "model_config": probe.get("model_config"),
        "initial_model_fingerprint": probe.get(
            "initial_model_fingerprint"
        ),
        "all_pass": payload.get("all_pass"),
    }
    if (
        values["schema_version"] != COVERAGE_STATE_PAET_DR_GATE_SCHEMA
        or values["dataset_free_receipt_fingerprint"]
        != dataset_free_receipt_fingerprint
        or values["execution_seed"]
        != COVERAGE_STATE_PAET_DR_EXECUTION_SEED
        or values["runtime_splits"] != ["D_R"]
        or values["all_pass"] is not True
        or payload.get("evidence_fingerprint")
        != receipt.evidence_fingerprint
        or not checks
        or not all(checks.values())
    ):
        raise PermissionError("PAET D_R gate is not a bound pass")
    return values


@dataclass(frozen=True, eq=False)
class CoverageStatePAETBoundedRunAuthorization(
    CoverageStateRunAuthorization,
):
    """All immutable prerequisites for the unique v21 bounded run."""

    run_id: str
    preflight: CoverageStateBoundedPreflight
    dataset_free_receipt: CoverageStatePAETDatasetFreeReceipt
    dataset_free_receipt_fingerprint: str
    dr_gate_receipt: CoverageStatePAETDRGateReceipt
    dr_gate_binding: tuple[tuple[str, object], ...]
    sealed_v20_reference: CoverageStatePAETV20Reference
    sealed_v20_reference_fingerprint: str
    implementation_binding: tuple[tuple[str, str], ...]
    implementation_fingerprint: str
    model_config_fingerprint: str
    expected_parameter_count: int
    objective_suite: tuple[str, ...]
    candidate_objective: str
    candidate_objective_policy: str
    fixed_margin: float
    field_policy: str
    equation_policy: str
    flip_policy: str
    transport_policy: str

    def __post_init__(self) -> None:
        self._validate_lightweight_bindings()
        _ = self.authorization_fingerprint

    def _validate_lightweight_bindings(self) -> None:
        _validate_run_id(self.run_id)
        if (
            not isinstance(self.preflight, CoverageStateBoundedPreflight)
            or not isinstance(
                self.dataset_free_receipt,
                CoverageStatePAETDatasetFreeReceipt,
            )
            or not isinstance(
                self.dr_gate_receipt,
                CoverageStatePAETDRGateReceipt,
            )
            or not isinstance(
                self.sealed_v20_reference,
                CoverageStatePAETV20Reference,
            )
        ):
            raise TypeError("PAET authorization prerequisite type changed")
        expected_config = expected_coverage_state_paet_config(
            self.preflight
        )
        training_initial, raw_initial = (
            _common_initial_model_fingerprints(expected_config)
        )
        dr = dict(self.dr_gate_binding)
        if (
            self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or not self.dataset_free_receipt.all_pass
            or self.sealed_v20_reference.reference_fingerprint
            != self.sealed_v20_reference_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != stable_fingerprint(_model_config_payload(expected_config))
            or self.expected_parameter_count
            != COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
            or self.expected_parameter_count
            != expected_config.expected_parameter_count
            or self.objective_suite != ("pmope_joint",)
            or self.candidate_objective
            != CoverageStatePairObjective.PMOPE_JOINT.value
            or self.candidate_objective_policy != CSLF_PMOPE_POLICY
            or self.fixed_margin != COVERAGE_STATE_PAET_MARGIN
            or self.field_policy != CSLF_PAET_FIELD_POLICY
            or self.equation_policy != CSLF_PAET_EQUATION_POLICY
            or self.flip_policy != CSLF_PAET_FLIP_POLICY
            or self.transport_policy != CSLF_PAET_TRANSPORT_POLICY
            or training_initial
            != COVERAGE_STATE_PAET_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            or raw_initial
            != COVERAGE_STATE_PAET_HISTORICAL_RAW_INITIAL_STATE_FINGERPRINT
            or self.preflight.schedule.config.seed
            != COVERAGE_STATE_BOUNDED_SEED
            or self.preflight.schedule.config.epochs
            != COVERAGE_STATE_BOUNDED_EPOCHS
            or self.preflight.schedule.config.steps_per_epoch
            != COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
            or self.preflight.schedule.config.updates
            != COVERAGE_STATE_BOUNDED_UPDATES
            or self.preflight.schedule.schedule_fingerprint
            != COVERAGE_STATE_PAET_HISTORICAL_SCHEDULE_FINGERPRINT
            or self.preflight.population.bounded_cache_fingerprint
            != COVERAGE_STATE_PAET_HISTORICAL_CACHE_FINGERPRINT
            or dr.get("bounded_population_fingerprint")
            != self.preflight.population.population_fingerprint
            or dr.get("bounded_cache_fingerprint")
            != self.preflight.population.bounded_cache_fingerprint
            or dr.get("execution_seed") != COVERAGE_STATE_BOUNDED_SEED
            or dr.get("model_config")
            != _dr_model_config_payload(expected_config)
            or dr.get("initial_model_fingerprint") != raw_initial
            or dr.get("all_pass") is not True
        ):
            raise ValueError("PAET bounded authorization binding changed")

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
                COVERAGE_STATE_PAET_BOUNDED_AUTHORIZATION_SCHEMA
            ),
            "run_id": self.run_id,
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
            "sealed_v20_reference": (
                self.sealed_v20_reference.canonical_payload()
            ),
            "sealed_v20_reference_fingerprint": (
                self.sealed_v20_reference_fingerprint
            ),
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "implementation_fingerprint": (
                self.implementation_fingerprint
            ),
            "model_config_fingerprint": self.model_config_fingerprint,
            "model_class": (
                "CURELitePhaseAlignedEvidenceTransportLevelSet"
            ),
            "expected_parameter_count": self.expected_parameter_count,
            "input_representation": PAET_INPUT_REPRESENTATION,
            "field_policy": self.field_policy,
            "equation_policy": self.equation_policy,
            "flip_policy": self.flip_policy,
            "transport_policy": self.transport_policy,
            "objective_suite": list(self.objective_suite),
            "candidate_objective": self.candidate_objective,
            "candidate_objective_policy": (
                self.candidate_objective_policy
            ),
            "fixed_margin_hex": self.fixed_margin.hex(),
            "historical_comparison_coordinates": {
                "reference": "sealed_v20_bfa_cmif_bounded_400_r2",
                "common_initial_model_fingerprint": (
                    COVERAGE_STATE_PAET_HISTORICAL_INITIAL_MODEL_FINGERPRINT
                ),
                "schedule_fingerprint": (
                    COVERAGE_STATE_PAET_HISTORICAL_SCHEDULE_FINGERPRINT
                ),
                "cache_fingerprint": (
                    COVERAGE_STATE_PAET_HISTORICAL_CACHE_FINGERPRINT
                ),
                "optimizer_fingerprint": (
                    COVERAGE_STATE_PAET_HISTORICAL_OPTIMIZER_FINGERPRINT
                ),
                "device_cache_fingerprint": (
                    COVERAGE_STATE_PAET_HISTORICAL_DEVICE_CACHE_FINGERPRINT
                ),
                "allowed_difference": (
                    "predeclared_phase_aligned_evidence_transport_only"
                ),
            },
            "checks": {
                "run_id_bound": (
                    self.run_id == COVERAGE_STATE_PAET_OFFICIAL_RUN_ID
                ),
                "preflight_passed": self.preflight.training_authorized,
                "dataset_free_gate_passed": (
                    self.dataset_free_receipt.all_pass
                ),
                "D_R_gate_passed": (
                    dict(self.dr_gate_binding).get("all_pass") is True
                ),
                "sealed_v20_reference_read_only": True,
                "singleton_paet_pmope_candidate": (
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
        self._validate_lightweight_bindings()
        if stable_fingerprint(
            self.canonical_payload()
        ) != self.authorization_fingerprint:
            raise RuntimeError("PAET bounded authorization changed")

    def verify_model_config(
        self,
        model_config: CoverageStatePhaseAlignedEvidenceTransportConfig,
    ) -> None:
        if (
            type(model_config)
            is not CoverageStatePhaseAlignedEvidenceTransportConfig
            or stable_fingerprint(_model_config_payload(model_config))
            != self.model_config_fingerprint
            or model_config.expected_parameter_count
            != self.expected_parameter_count
        ):
            raise PermissionError(
                "PAET authorization rejects this model config"
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
                "PAET authorization rejects this bounded run"
            )


def prepare_coverage_state_paet_bounded_run_authorization(
    preflight: CoverageStateBoundedPreflight,
    dataset_free_receipt: CoverageStatePAETDatasetFreeReceipt,
    dr_gate_receipt: CoverageStatePAETDRGateReceipt,
    *,
    run_id: str,
    sealed_v20_reference: CoverageStatePAETV20Reference | None = None,
    dr_gate_canonical_payload: Mapping[str, object] | None = None,
) -> CoverageStatePAETBoundedRunAuthorization:
    """Bind the generated, real-D_R, v20, and explicit-run prerequisites."""

    _validate_run_id(run_id)
    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    if not isinstance(
        dataset_free_receipt,
        CoverageStatePAETDatasetFreeReceipt,
    ):
        raise TypeError("dataset_free_receipt must be PAET dataset-free")
    if not isinstance(
        dr_gate_receipt,
        CoverageStatePAETDRGateReceipt,
    ):
        raise TypeError("dr_gate_receipt must be PAET D_R")
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
    reference = (
        verify_repository_coverage_state_bfa_v20_reference()
        if sealed_v20_reference is None
        else sealed_v20_reference
    )
    if not isinstance(reference, CoverageStatePAETV20Reference):
        raise TypeError("sealed_v20_reference has the wrong type")
    implementation = _current_implementation_binding()
    model_config = expected_coverage_state_paet_config(preflight)
    result = CoverageStatePAETBoundedRunAuthorization(
        run_id=run_id,
        preflight=preflight,
        dataset_free_receipt=dataset_free_receipt,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt.receipt_fingerprint
        ),
        dr_gate_receipt=dr_gate_receipt,
        dr_gate_binding=dr_binding,
        sealed_v20_reference=reference,
        sealed_v20_reference_fingerprint=(
            reference.reference_fingerprint
        ),
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
            for value in COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES
        ),
        candidate_objective=CoverageStatePairObjective.PMOPE_JOINT.value,
        candidate_objective_policy=CSLF_PMOPE_POLICY,
        fixed_margin=COVERAGE_STATE_PAET_MARGIN,
        field_policy=CSLF_PAET_FIELD_POLICY,
        equation_policy=CSLF_PAET_EQUATION_POLICY,
        flip_policy=CSLF_PAET_FLIP_POLICY,
        transport_policy=CSLF_PAET_TRANSPORT_POLICY,
    )
    result.verify_unchanged()
    return result


@dataclass(frozen=True)
class CoverageStatePAETTrainingResourceMeasurement:
    """Actual CUDA training-interval measurements without invented ratios."""

    device: str
    measurement_scope: str
    baseline_allocated_bytes: int
    baseline_reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    elapsed_ns: int
    updates: int
    parameter_count: int
    v20_reference_status: str

    def __post_init__(self) -> None:
        integer_values = (
            self.baseline_allocated_bytes,
            self.baseline_reserved_bytes,
            self.peak_allocated_bytes,
            self.peak_reserved_bytes,
            self.elapsed_ns,
            self.updates,
            self.parameter_count,
        )
        if (
            self.device != "cuda:0"
            or self.measurement_scope
            != (
                "single_training_invocation_including_device_cache_setup_"
                "and_post_verification"
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in integer_values
            )
            or self.peak_allocated_bytes
            < self.baseline_allocated_bytes
            or self.peak_reserved_bytes < self.baseline_reserved_bytes
            or self.elapsed_ns <= 0
            or self.updates != COVERAGE_STATE_BOUNDED_UPDATES
            or self.parameter_count
            != COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
            or self.v20_reference_status
            != COVERAGE_STATE_PAET_V20_RESOURCE_COMPARISON_STATUS
        ):
            raise ValueError("PAET resource measurement is malformed")

    @property
    def incremental_peak_allocated_bytes(self) -> int:
        return self.peak_allocated_bytes - self.baseline_allocated_bytes

    @property
    def incremental_peak_reserved_bytes(self) -> int:
        return self.peak_reserved_bytes - self.baseline_reserved_bytes

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                COVERAGE_STATE_PAET_RESOURCE_MEASUREMENT_SCHEMA
            ),
            "device": self.device,
            "measurement_scope": self.measurement_scope,
            "wall_time_includes_temperature_pause_if_any": True,
            "baseline_allocated_bytes": self.baseline_allocated_bytes,
            "baseline_reserved_bytes": self.baseline_reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "incremental_peak_allocated_bytes": (
                self.incremental_peak_allocated_bytes
            ),
            "incremental_peak_reserved_bytes": (
                self.incremental_peak_reserved_bytes
            ),
            "working_memory_definition": (
                "incremental_peak_allocated_bytes"
            ),
            "working_memory_bytes": (
                self.incremental_peak_allocated_bytes
            ),
            "elapsed_ns": self.elapsed_ns,
            "updates": self.updates,
            "ns_per_update": {
                "numerator": self.elapsed_ns,
                "denominator": self.updates,
            },
            "parameter_count": self.parameter_count,
            "v20_comparison": {
                "status": self.v20_reference_status,
                "measured_reference_available": False,
                "reference_working_memory_bytes": None,
                "reference_ns_per_update": None,
                "working_memory_ratio": None,
                "step_time_ratio": None,
                "working_memory_ratio_limit": {
                    "numerator": (
                        COVERAGE_STATE_PAET_MEMORY_RATIO_LIMIT[0]
                    ),
                    "denominator": (
                        COVERAGE_STATE_PAET_MEMORY_RATIO_LIMIT[1]
                    ),
                },
                "step_time_ratio_limit": {
                    "numerator": (
                        COVERAGE_STATE_PAET_STEP_TIME_RATIO_LIMIT[0]
                    ),
                    "denominator": (
                        COVERAGE_STATE_PAET_STEP_TIME_RATIO_LIMIT[1]
                    ),
                },
                "working_memory_gate_evaluated": False,
                "working_memory_gate_passed": None,
                "step_time_gate_evaluated": False,
                "step_time_gate_passed": None,
                "ratio_claim_supported": False,
                "not_a_scientific_gate": True,
            },
            "absolute_resource_run_completed": True,
            "OOM_observed": False,
        }

    @cached_property
    def measurement_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


_TrainingValue = TypeVar("_TrainingValue")


def _measure_training_invocation(
    operation: Callable[[], _TrainingValue],
    *,
    device: torch.device,
    clock: Callable[[], int] = perf_counter_ns,
) -> tuple[_TrainingValue, CoverageStatePAETTrainingResourceMeasurement]:
    """Measure exactly one completed PAET training call on CUDA zero."""

    if device != torch.device("cuda:0"):
        raise PermissionError("PAET resource measurement fixes cuda:0")
    if not callable(operation) or not callable(clock):
        raise TypeError("operation and clock must be callable")
    torch.cuda.synchronize(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    start = int(clock())
    value = operation()
    torch.cuda.synchronize(device)
    elapsed = int(clock()) - start
    measurement = CoverageStatePAETTrainingResourceMeasurement(
        device=str(device),
        measurement_scope=(
            "single_training_invocation_including_device_cache_setup_"
            "and_post_verification"
        ),
        baseline_allocated_bytes=baseline_allocated,
        baseline_reserved_bytes=baseline_reserved,
        peak_allocated_bytes=int(
            torch.cuda.max_memory_allocated(device)
        ),
        peak_reserved_bytes=int(
            torch.cuda.max_memory_reserved(device)
        ),
        elapsed_ns=elapsed,
        updates=COVERAGE_STATE_BOUNDED_UPDATES,
        parameter_count=COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT,
        v20_reference_status=(
            COVERAGE_STATE_PAET_V20_RESOURCE_COMPARISON_STATUS
        ),
    )
    return value, measurement


def _paet_bounded_result_checks(
    authorization: CoverageStatePAETBoundedRunAuthorization,
    training: CoverageStateMatchedTrainingResult,
    resource_measurement: CoverageStatePAETTrainingResourceMeasurement,
    certificate: CoverageStatePAETCertificateReceipt,
    diagnostic: CoverageStateZeroLevelEvaluationResult,
    decision: CoverageStatePAETBoundedDecision,
    *,
    run_id: str,
    training_invocations: int,
    certificate_invocations: int,
    zero_level_evaluation_invocations: int,
) -> tuple[tuple[str, bool], ...]:
    results = training.results
    models = training.models
    row = results[0] if len(results) == 1 else None
    model_entry = models[0] if len(models) == 1 else None
    model = model_entry[1] if model_entry is not None else None
    latency = (
        dict(row.first_nonzero_gradient_update)
        if row is not None
        else {}
    )
    checks = {
        "explicit_run_id_bound_at_all_runner_layers": (
            run_id
            == authorization.run_id
            == COVERAGE_STATE_PAET_OFFICIAL_RUN_ID
        ),
        "authorization_and_sealed_v20_reference": (
            authorization.training_authorized
            and authorization.sealed_v20_reference.run_id
            == COVERAGE_STATE_PAET_V20_REFERENCE_RUN_ID
        ),
        "singleton_paet_pmope_training": (
            row is not None
            and model_entry is not None
            and len(results) == len(models) == 1
            and row.objective == "pmope_joint"
            and row.objective_policy == CSLF_PMOPE_POLICY
            and model_entry[0] == "pmope_joint"
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
            == COVERAGE_STATE_PAET_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            and row.initial_model_fingerprint
            == COVERAGE_STATE_PAET_HISTORICAL_INITIAL_MODEL_FINGERPRINT
            and row.schedule_fingerprint
            == COVERAGE_STATE_PAET_HISTORICAL_SCHEDULE_FINGERPRINT
            and row.cache_fingerprint
            == COVERAGE_STATE_PAET_HISTORICAL_CACHE_FINGERPRINT
            and row.optimizer_config_fingerprint
            == COVERAGE_STATE_PAET_HISTORICAL_OPTIMIZER_FINGERPRINT
            and row.device_cache_fingerprint
            == COVERAGE_STATE_PAET_HISTORICAL_DEVICE_CACHE_FINGERPRINT
            and row.execution_device == "cuda:0"
            and training.schedule_fingerprint
            == COVERAGE_STATE_PAET_HISTORICAL_SCHEDULE_FINGERPRINT
            and training.cache_fingerprint
            == COVERAGE_STATE_PAET_HISTORICAL_CACHE_FINGERPRINT
        ),
        "paet_gradient_path": (
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
        "exact_paet_model_and_parameter_count": (
            model is not None
            and type(model)
            is CURELitePhaseAlignedEvidenceTransportLevelSet
            and stable_fingerprint(_model_config_payload(model.config))
            == authorization.model_config_fingerprint
            and sum(
                parameter.numel() for parameter in model.parameters()
            )
            == COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
        ),
        "actual_training_resource_measurement_complete": (
            resource_measurement.device == "cuda:0"
            and resource_measurement.updates
            == COVERAGE_STATE_BOUNDED_UPDATES
            and resource_measurement.parameter_count
            == COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
            and resource_measurement.elapsed_ns > 0
            and resource_measurement.peak_allocated_bytes
            >= resource_measurement.baseline_allocated_bytes
            and resource_measurement.peak_reserved_bytes
            >= resource_measurement.baseline_reserved_bytes
            and resource_measurement.v20_reference_status
            == COVERAGE_STATE_PAET_V20_RESOURCE_COMPARISON_STATUS
        ),
        "v20_resource_ratio_not_fabricated_or_scientific_gate": (
            resource_measurement.canonical_payload()["v20_comparison"][
                "working_memory_ratio"
            ]
            is None
            and resource_measurement.canonical_payload()[
                "v20_comparison"
            ]["step_time_ratio"]
            is None
            and resource_measurement.canonical_payload()[
                "v20_comparison"
            ]["not_a_scientific_gate"]
            is True
        ),
        "exact_once_execution_order_ledger": (
            training_invocations
            == certificate_invocations
            == zero_level_evaluation_invocations
            == 1
        ),
        "certificate_checkpoint_cache_and_integrity": (
            model is not None
            and certificate.cache_fingerprint
            == authorization.preflight.population.bounded_cache_fingerprint
            and certificate.model_fingerprint_before
            == _certificate_model_fingerprint(model)
            and certificate.model_fingerprint_after
            == certificate.model_fingerprint_before
            and certificate.integrity_passed
        ),
        "all_32_pair_values_reported_not_gating": (
            len(certificate.pair_certificates)
            == COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT
            and certificate.pair_batch_size
            == COVERAGE_STATE_PAET_CERTIFICATE_DEFAULT_PAIR_BATCH_SIZE
            and certificate.model_forward_invocations
            == ceil_div(
                COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT,
                COVERAGE_STATE_PAET_CERTIFICATE_DEFAULT_PAIR_BATCH_SIZE,
            )
            and certificate.canonical_payload()["diagnostic_summary"][
                "pair_result_is_bounded_gate"
            ]
            is False
        ),
        "zero_level_checkpoint_and_D_R_binding": (
            model is not None
            and diagnostic.checkpoint_fingerprint
            == module_state_fingerprint(model)
            and diagnostic.cache_fingerprint
            == authorization.preflight.population.bounded_cache_fingerprint
            and diagnostic.split == "D_R"
        ),
        "predeclared_structural_advancement_gate": (
            decision.diagnostic is diagnostic
            and decision.run_id == run_id
            and decision.bounded_gate_passed
        ),
        "same_sign_response_reported_not_gating": (
            decision.canonical_payload()[
                "same_sign_response_diagnostic"
            ]["is_gate"]
            is False
        ),
        "read_only_post_training_checks": (
            certificate.optimizer_constructed is False
            and certificate.backward_performed is False
            and certificate.training_performed is False
            and certificate.external_data_accessed is False
            and diagnostic.backward_calls == 0
            and diagnostic.optimizer_steps == 0
            and not diagnostic.config.d_v_accessed
            and not diagnostic.config.d_t_accessed
        ),
        "no_resume_retry_or_formal800_execution": True,
    }
    return tuple(sorted(checks.items()))


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True, eq=False)
class CoverageStatePAETBoundedRunResult:
    """The unique v21 run plus mandatory read-only reports."""

    run_id: str
    authorization: CoverageStatePAETBoundedRunAuthorization
    training: CoverageStateMatchedTrainingResult
    resource_measurement: CoverageStatePAETTrainingResourceMeasurement
    certificate: CoverageStatePAETCertificateReceipt
    diagnostic: CoverageStateZeroLevelEvaluationResult
    decision: CoverageStatePAETBoundedDecision
    training_invocations: int
    certificate_invocations: int
    zero_level_evaluation_invocations: int
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if (
            self.run_id != self.authorization.run_id
            or tuple(value.objective for value in self.training.results)
            != ("pmope_joint",)
            or tuple(name for name, _ in self.training.models)
            != ("pmope_joint",)
            or self.decision.diagnostic is not self.diagnostic
            or self.checks != tuple(sorted(self.checks))
            or len({name for name, _ in self.checks})
            != len(self.checks)
        ):
            raise ValueError("PAET bounded result is incomplete")

    @property
    def bounded_gate_passed(self) -> bool:
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def formal800_eligible(self) -> bool:
        return self.bounded_gate_passed

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(
            name for name, passed in self.checks if not passed
        )

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        self.training.verify_unchanged()
        self.certificate.verify()
        if self.decision.decision_fingerprint != stable_fingerprint(
            self.decision.canonical_payload()
        ):
            raise RuntimeError("PAET bounded decision changed")
        expected = _paet_bounded_result_checks(
            self.authorization,
            self.training,
            self.resource_measurement,
            self.certificate,
            self.diagnostic,
            self.decision,
            run_id=self.run_id,
            training_invocations=self.training_invocations,
            certificate_invocations=self.certificate_invocations,
            zero_level_evaluation_invocations=(
                self.zero_level_evaluation_invocations
            ),
        )
        if self.checks != expected:
            raise RuntimeError("PAET bounded result checks changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_PAET_BOUNDED_RESULT_SCHEMA,
            "run_id": self.run_id,
            "runtime_splits": ["D_R"],
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "candidate_model": "PAET-BFA",
            "candidate_objective": "pmope_joint",
            "candidate_objective_policy": CSLF_PMOPE_POLICY,
            "fixed_margin_hex": COVERAGE_STATE_PAET_MARGIN.hex(),
            "parameter_count": COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT,
            "training": self.training.canonical_payload(),
            "training_resource_measurement": (
                self.resource_measurement.canonical_payload()
            ),
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
            "sealed_v20_reference": (
                self.authorization
                .sealed_v20_reference.canonical_payload()
            ),
            "historical_candidate_retrained": False,
            "historical_candidate_reevaluated": False,
            "historical_outcome_is_candidate_gate": False,
            "pair_certificate_result_is_bounded_gate": False,
            "same_sign_response_is_gate": False,
            "resource_ratio_claim_supported": False,
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "bounded_gate_passed": self.bounded_gate_passed,
            "formal800_eligible": self.formal800_eligible,
            "formal_800_authorized": False,
            "formal_800_executed": False,
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


def run_coverage_state_paet_bfa_pmope_bounded_400(
    authorization: CoverageStatePAETBoundedRunAuthorization,
    model_config: CoverageStatePhaseAlignedEvidenceTransportConfig,
    *,
    run_id: str,
    device: torch.device | str,
) -> CoverageStatePAETBoundedRunResult:
    """Run training -> certificate -> zero-level decision exactly once."""

    if not isinstance(
        authorization,
        CoverageStatePAETBoundedRunAuthorization,
    ):
        raise TypeError("authorization must be a PAET authorization")
    _validate_run_id(run_id)
    if run_id != authorization.run_id:
        raise PermissionError(
            "PAET run_id differs from its authorization"
        )
    resolved_device = torch.device(device)
    if resolved_device != torch.device("cuda:0"):
        raise PermissionError(
            "PAET bounded-400 is frozen to visible cuda:0"
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

        def train_once() -> CoverageStateMatchedTrainingResult:
            return train_matched_coverage_state_paet_bfa_pmope_objectives(
                model_config,
                preflight.population.cache,
                preflight.schedule,
                config=CoverageStateMatchedTrainingConfig(
                    seed=COVERAGE_STATE_BOUNDED_SEED
                ),
                device=device,
                authorization=authorization,
            )

        training, resource_measurement = _measure_training_invocation(
            train_once,
            device=resolved_device,
        )
        if (
            tuple(value.objective for value in training.results)
            != ("pmope_joint",)
            or tuple(name for name, _ in training.models)
            != ("pmope_joint",)
            or type(training.models[0][1])
            is not CURELitePhaseAlignedEvidenceTransportLevelSet
        ):
            raise RuntimeError(
                "PAET training returned a non-singleton model"
            )
        model = training.models[0][1].eval()
        certificate_invocations += 1
        certificate = audit_coverage_state_paet_pair_certificate(
            model,
            preflight.population.cache,
            device=device,
            pair_batch_size=(
                COVERAGE_STATE_PAET_CERTIFICATE_DEFAULT_PAIR_BATCH_SIZE
            ),
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
        decision = decide_coverage_state_paet_bounded(
            diagnostic,
            run_id=run_id,
        )
    result = CoverageStatePAETBoundedRunResult(
        run_id=run_id,
        authorization=authorization,
        training=training,
        resource_measurement=resource_measurement,
        certificate=certificate,
        diagnostic=diagnostic,
        decision=decision,
        training_invocations=training_invocations,
        certificate_invocations=certificate_invocations,
        zero_level_evaluation_invocations=evaluation_invocations,
        checks=_paet_bounded_result_checks(
            authorization,
            training,
            resource_measurement,
            certificate,
            diagnostic,
            decision,
            run_id=run_id,
            training_invocations=training_invocations,
            certificate_invocations=certificate_invocations,
            zero_level_evaluation_invocations=evaluation_invocations,
        ),
    )
    result.verify_unchanged()
    return result


__all__ = [
    "COVERAGE_STATE_PAET_BOUNDED_AUTHORIZATION_SCHEMA",
    "COVERAGE_STATE_PAET_BOUNDED_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_PAET_BOUNDED_RESULT_SCHEMA",
    "COVERAGE_STATE_PAET_HISTORICAL_CACHE_FINGERPRINT",
    "COVERAGE_STATE_PAET_HISTORICAL_DEVICE_CACHE_FINGERPRINT",
    "COVERAGE_STATE_PAET_HISTORICAL_INITIAL_MODEL_FINGERPRINT",
    "COVERAGE_STATE_PAET_HISTORICAL_OPTIMIZER_FINGERPRINT",
    "COVERAGE_STATE_PAET_HISTORICAL_RAW_INITIAL_STATE_FINGERPRINT",
    "COVERAGE_STATE_PAET_HISTORICAL_SCHEDULE_FINGERPRINT",
    "COVERAGE_STATE_PAET_MEMORY_RATIO_LIMIT",
    "COVERAGE_STATE_PAET_OFFICIAL_RUN_ID",
    "COVERAGE_STATE_PAET_RESOURCE_MEASUREMENT_SCHEMA",
    "COVERAGE_STATE_PAET_STEP_TIME_RATIO_LIMIT",
    "COVERAGE_STATE_PAET_V20_COMPLETE_FINGERPRINT",
    "COVERAGE_STATE_PAET_V20_REFERENCE_RUN_ID",
    "COVERAGE_STATE_PAET_V20_RESOURCE_COMPARISON_STATUS",
    "CoverageStatePAETBoundedRunAuthorization",
    "CoverageStatePAETBoundedRunResult",
    "CoverageStatePAETTrainingResourceMeasurement",
    "CoverageStatePAETV20Reference",
    "expected_coverage_state_paet_config",
    "prepare_coverage_state_paet_bounded_run_authorization",
    "run_coverage_state_paet_bfa_pmope_bounded_400",
    "verify_repository_coverage_state_bfa_v20_reference",
]
