"""Strict in-memory Formal800 training core for v23 PACRE-VC.

The only advancement prerequisite is one complete v23 ``D_R`` structural
receipt (13/13 PASS), together with the dataset-free receipt and the exact
source closure to which that receipt is bound.  Bounded400 is deliberately not
an input, a prerequisite, or a success condition.

This module builds the schedule from the complete
``real_inputs.scalar_cache``, applies the formal exposure gate, constructs the
exact v23 model from scratch, and performs one PMOPE/Adam training invocation.
It has no CLI, persistence, checkpoint, calibration, inference, or performance
evaluation surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cached_property
import json
from threading import Lock
from typing import Callable, Final, Mapping

import torch

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
)
from cure_lite.coverage_state_schedule import (
    COVERAGE_STATE_FORMAL_EPOCHS,
    COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH,
    CoverageStateScheduleConfig,
    CoverageStateTrainingSchedule,
    build_coverage_state_training_schedule,
    coverage_state_formal_exposure_gate,
)
from cure_lite.coverage_state_sobolev import CSLF_PMOPE_POLICY
from cure_lite.experiment.coverage_state_bounded_runner import (
    _deterministic_execution,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRInputs,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_FORMAL_SCOPE,
    CoverageStateRunAuthorization,
    CoverageStateTrainingResult,
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
    train_coverage_state_objective,
)
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
)
from cure_lite_v23.dataset_free import (
    PACRE_FORMAL_FEATURE_CHANNELS,
    PACRE_FORMAL_FEATURE_STRIDE,
    PACRE_FORMAL_PARAMETER_COUNT,
    PACRE_FORMAL_WIDTH,
)
from cure_lite_v23.dr_gate import (
    PACRE_VC_DR_CHECK_NAMES,
    PACRE_VC_DR_PASS_DECISION,
    CoverageStatePACREDRGateReceipt,
    _validate_dataset_free_receipt,
)
from cure_lite_v23.factory import (
    PACRE_VC_PARAMETER_NAMES,
    build_pacre_vc_training_model,
)
from cure_lite_v23.pacre_vc import (
    PACRE_VC_CANDIDATE,
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)
from cure_lite_v23.protocol import (
    verify_source_closure,
)
from cure_lite_v23.training import (
    PACRE_CONFIG_FQCN,
    PACRE_MODEL_FQCN,
    PACRE_OPTIMIZER_FQCN,
    PACRE_PMOPE_OBJECTIVE,
    PACRE_PMOPE_TRAINING_CONFIG,
    PACREPMOPETrainingConfig,
)


PACRE_VC_FORMAL_SEED: Final = 42
PACRE_VC_FORMAL_UPDATES: Final = (
    COVERAGE_STATE_FORMAL_EPOCHS
    * COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH
)
PACRE_VC_FORMAL_DEVICE: Final = "cuda:0"
PACRE_VC_FORMAL_RUN_ID: Final = (
    "cure_lite_pacre_v23_vc_pmope_formal_800_seed42_r1"
)
PACRE_VC_FORMAL_AUTHORIZATION_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-formal800-authorization-v1"
)
PACRE_VC_FORMAL_RESULT_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-formal800-training-result-v1"
)
PACRE_VC_FORMAL_ATTEMPT_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-formal800-attempt-v1"
)

# These coordinates are properties of the full IRSTD-1K D_R scalar cache,
# seed-42 schedule/exposure policy, and the raw-compatible v21/v22/v23 initial
# state.  They are intentionally frozen rather than learned from the caller.
PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT: Final = (
    "569b0fb97d819cf1281ca1d148227bc1c5e229b8301065cb536656b5e578e645"
)
PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT: Final = (
    "abc1625c93dc9521b1e824ed4b2e685e867d755d8be5e7b1af3a4a5638240431"
)
PACRE_VC_FORMAL_EXPOSURE_GATE_FINGERPRINT: Final = (
    "c942578b53fd1ba9524cfcb28d504e9ea205f34af758bda6e9d3b466e5ce2c63"
)
PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT: Final = (
    "a4086bcffba4035984a8c334b3fa194910bcb7376a573f7f96ef8d36e097240d"
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _model_config_payload(
    config: CoverageStatePACREVerifierCorrectedConfig,
) -> dict[str, object]:
    if type(config) is not CoverageStatePACREVerifierCorrectedConfig:
        raise TypeError("Formal800 requires the exact PACRE-VC v23 config")
    return {
        "model_fqcn": PACRE_MODEL_FQCN,
        "config_fqcn": PACRE_CONFIG_FQCN,
        "config": asdict(config),
        "expected_parameter_count": config.expected_parameter_count,
    }


def _model_parameter_devices(
    model: CURELitePACREVerifierCorrectedLevelSet,
) -> tuple[str, ...]:
    return tuple(sorted({str(value.device) for value in model.parameters()}))


def _formal_model_binding(
    config: CoverageStatePACREVerifierCorrectedConfig,
) -> tuple[str, str, str, int]:
    """Return contract JSON/fingerprint, initial fingerprint, and parameter count."""

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(PACRE_VC_FORMAL_SEED)
        model = build_pacre_vc_training_model(config)
    if type(model) is not CURELitePACREVerifierCorrectedLevelSet:
        raise AssertionError("PACRE-VC factory returned the wrong model")
    contract = coverage_state_model_contract_payload(model)
    contract_json = canonical_json(contract)
    parameter_count = sum(value.numel() for value in model.parameters())
    return (
        contract_json,
        stable_fingerprint(contract),
        coverage_state_model_fingerprint(model),
        parameter_count,
    )


def _config_from_verified_real_inputs(
    real_inputs: CoverageStateRealDRInputs,
) -> CoverageStatePACREVerifierCorrectedConfig:
    cache = real_inputs.scalar_cache
    raw = cache.raw_catalog
    natural = raw.natural_records
    if (
        real_inputs.source_binding.dataset != "IRSTD-1K"
        or real_inputs.source_binding.split != "D_R"
        or raw.dataset != "IRSTD-1K"
        or raw.split != "D_R"
        or not natural
        or raw.feature_stride != PACRE_FORMAL_FEATURE_STRIDE
        or cache.sobolev_config.truncation_radius
        != PACRE_FORMAL_FEATURE_STRIDE
        or int(natural[0].feature.shape[1])
        != PACRE_FORMAL_FEATURE_CHANNELS
    ):
        raise PermissionError(
            "Formal800 requires the exact full IRSTD-1K D_R input contract"
        )
    config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=PACRE_FORMAL_FEATURE_CHANNELS,
        feature_stride=PACRE_FORMAL_FEATURE_STRIDE,
        width=PACRE_FORMAL_WIDTH,
    )
    if config.expected_parameter_count != PACRE_FORMAL_PARAMETER_COUNT:
        raise AssertionError("PACRE-VC formal parameter count changed")
    return config


def expected_pacre_vc_formal_config(
    real_inputs: CoverageStateRealDRInputs,
) -> CoverageStatePACREVerifierCorrectedConfig:
    """Return the only model configuration accepted by the Formal800 core."""

    if type(real_inputs) is not CoverageStateRealDRInputs:
        raise TypeError("real_inputs must have exact type CoverageStateRealDRInputs")
    real_inputs.verify_unchanged()
    return _config_from_verified_real_inputs(real_inputs)


def _validated_dataset_free_json(
    receipt: Mapping[str, object],
) -> tuple[str, str]:
    if not isinstance(receipt, Mapping):
        raise TypeError("dataset_free_receipt must be a mapping")
    fingerprint = _validate_dataset_free_receipt(receipt)
    text = canonical_json(dict(receipt))
    if not _is_sha256(fingerprint):
        raise AssertionError("dataset-free receipt fingerprint is malformed")
    return text, fingerprint


def _validated_source_closure_json(
    closure: Mapping[str, object],
) -> tuple[str, str]:
    if not isinstance(closure, Mapping):
        raise TypeError("source_closure must be a mapping")
    fingerprint = verify_source_closure(closure)
    payload = dict(closure)
    if (
        not _is_sha256(fingerprint)
        or payload.get("closure_fingerprint") != fingerprint
        or payload.get("D_R_accessed") is not False
        or payload.get("D_V_accessed") is not False
        or payload.get("D_T_accessed") is not False
        or payload.get("training_performed") is not False
    ):
        raise PermissionError("PACRE-VC source closure is not create-only")
    return canonical_json(payload), fingerprint


def _validated_dr_receipt_json(
    receipt: CoverageStatePACREDRGateReceipt,
) -> tuple[str, str]:
    """Validate the complete canonical 13/13 D_R structural receipt."""

    if type(receipt) is not CoverageStatePACREDRGateReceipt:
        raise TypeError(
            "dr_gate_receipt must have exact type CoverageStatePACREDRGateReceipt"
        )
    payload = receipt.canonical_payload()
    if type(payload) is not dict:
        raise TypeError("D_R receipt canonical payload must be a dict")
    fingerprint = receipt.receipt_fingerprint
    checks = tuple(receipt.checks)
    probe = receipt.probe
    if (
        not _is_sha256(fingerprint)
        or stable_fingerprint(payload) != fingerprint
        or tuple(name for name, _ in checks) != PACRE_VC_DR_CHECK_NAMES
        or len(checks) != 13
        or not all(type(passed) is bool and passed for _, passed in checks)
        or payload.get("checks") != dict(checks)
        or payload.get("failed_checks") != []
        or payload.get("candidate") != PACRE_VC_CANDIDATE
        or payload.get("gate_passed") is not True
        or payload.get("decision") != PACRE_VC_DR_PASS_DECISION
        or payload.get("D_R_accessed") is not True
        or payload.get("D_V_accessed") is not False
        or payload.get("D_T_accessed") is not False
        or payload.get("training_performed") is not False
        or payload.get("identifiability_only") is not True
        or payload.get("performance_claim_supported") is not False
        or not isinstance(probe, Mapping)
        or probe.get("D_R_accessed") is not True
        or probe.get("D_V_accessed") is not False
        or probe.get("D_T_accessed") is not False
        or probe.get("training_performed") is not False
        or probe.get("optimizer_constructed") is not False
        or probe.get("optimizer_steps") != 0
        or probe.get("parameter_updates") != 0
    ):
        raise PermissionError(
            "PACRE-VC Formal800 requires an exact 13/13 D_R structural PASS"
        )
    receipt.verify_sources_unchanged()
    return canonical_json(payload), fingerprint


def _validated_exposure(
    cache: CoverageStateScalarCache,
    schedule: CoverageStateTrainingSchedule,
) -> tuple[str, tuple[tuple[str, bool], ...]]:
    value = coverage_state_formal_exposure_gate(cache, schedule)
    if not isinstance(value, dict):
        raise TypeError("formal exposure gate must return a mapping")
    body = dict(value)
    fingerprint = body.pop("gate_fingerprint", None)
    raw_checks = value.get("checks")
    if not isinstance(raw_checks, Mapping):
        raise TypeError("formal exposure checks must be a mapping")
    checks = tuple(sorted(raw_checks.items()))
    if (
        not _is_sha256(fingerprint)
        or stable_fingerprint(body) != fingerprint
        or value.get("cache_fingerprint") != cache.cache_fingerprint
        or value.get("schedule_fingerprint")
        != schedule.schedule_fingerprint
        or value.get("all_pass") is not True
        or not checks
        or not all(type(passed) is bool and passed for _, passed in checks)
        or value.get("failed_checks") != []
        or value.get("D_V_accessed") is not False
        or value.get("D_T_accessed") is not False
    ):
        raise PermissionError("Formal800 exposure gate did not pass")
    return fingerprint, checks


class _FormalAttemptToken:
    """Process-local single-attempt state shared by equivalent authorizations."""

    def __init__(
        self,
        *,
        attempt_fingerprint: str,
        static_binding_fingerprint: str,
    ) -> None:
        self.attempt_fingerprint = attempt_fingerprint
        self.static_binding_fingerprint = static_binding_fingerprint
        self._lock = Lock()
        self._state = "available"
        self._claim_count = 0
        self._consume_count = 0
        self._failure_count = 0
        self._training_binding_fingerprint: str | None = None

    def claim(self, binding: str) -> None:
        with self._lock:
            if self._state != "available":
                raise PermissionError(
                    "PACRE-VC Formal800 attempt is no longer available"
                )
            if not _is_sha256(binding):
                raise ValueError("formal training binding is malformed")
            self._state = "reserved"
            self._claim_count += 1
            self._training_binding_fingerprint = binding

    def verify_reserved(self, binding: str) -> None:
        with self._lock:
            if (
                self._state != "reserved"
                or self._training_binding_fingerprint != binding
                or self._claim_count != 1
                or self._consume_count != 0
                or self._failure_count != 0
            ):
                raise PermissionError(
                    "PACRE-VC Formal800 attempt lacks this reservation"
                )

    def consume(self, binding: str) -> None:
        with self._lock:
            if (
                self._state != "reserved"
                or self._training_binding_fingerprint != binding
            ):
                raise PermissionError(
                    "PACRE-VC Formal800 attempt was not reserved for this model"
                )
            self._state = "consumed"
            self._consume_count += 1

    def verify_consumed(self, binding: str) -> None:
        with self._lock:
            if (
                self._state != "consumed"
                or self._training_binding_fingerprint != binding
                or self._claim_count != 1
                or self._consume_count != 1
                or self._failure_count != 0
            ):
                raise PermissionError(
                    "PACRE-VC Formal800 authorization was not consumed exactly once"
                )

    def fail(self) -> None:
        with self._lock:
            if self._state == "failed":
                return
            if self._state not in {"reserved", "consumed"}:
                raise PermissionError(
                    "an unavailable Formal800 attempt cannot be marked failed"
                )
            self._state = "failed"
            self._failure_count += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": self._state,
                "claim_count": self._claim_count,
                "consume_count": self._consume_count,
                "failure_count": self._failure_count,
                "training_binding_fingerprint": (
                    self._training_binding_fingerprint
                ),
            }


_FORMAL_ATTEMPT_REGISTRY_LOCK = Lock()
_FORMAL_ATTEMPT_REGISTRY: dict[str, _FormalAttemptToken] = {}


def _registered_attempt_token(
    *,
    attempt_fingerprint: str,
    static_binding_fingerprint: str,
    create: bool,
) -> _FormalAttemptToken:
    with _FORMAL_ATTEMPT_REGISTRY_LOCK:
        token = _FORMAL_ATTEMPT_REGISTRY.get(attempt_fingerprint)
        if token is None:
            if not create:
                raise PermissionError(
                    "PACRE-VC Formal800 attempt is not registered"
                )
            token = _FormalAttemptToken(
                attempt_fingerprint=attempt_fingerprint,
                static_binding_fingerprint=static_binding_fingerprint,
            )
            _FORMAL_ATTEMPT_REGISTRY[attempt_fingerprint] = token
        if token.static_binding_fingerprint != static_binding_fingerprint:
            raise PermissionError("Formal800 attempt binding collision")
        return token


@dataclass(frozen=True)
class _FormalPreparationSeal:
    real_inputs: CoverageStateRealDRInputs
    scalar_cache: CoverageStateScalarCache
    dr_gate_receipt: CoverageStatePACREDRGateReceipt
    model_config: CoverageStatePACREVerifierCorrectedConfig
    schedule: CoverageStateTrainingSchedule
    static_binding_fingerprint: str


def _static_binding_payload(
    *,
    run_id: str,
    output_claim_fingerprint: str,
    real_inputs: CoverageStateRealDRInputs,
    dataset_free_receipt_fingerprint: str,
    dr_gate_receipt_fingerprint: str,
    source_closure_fingerprint: str,
    schedule: CoverageStateTrainingSchedule,
    exposure_gate_fingerprint: str,
    exposure_gate_checks: tuple[tuple[str, bool], ...],
    model_config_fingerprint: str,
    model_contract_fingerprint: str,
    initial_model_fingerprint: str,
    expected_parameter_count: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "output_claim_fingerprint": output_claim_fingerprint,
        "real_inputs_fingerprint": real_inputs.build_fingerprint,
        "source_binding_fingerprint": (
            real_inputs.source_binding.binding_fingerprint
        ),
        "full_D_R_scalar_cache_fingerprint": (
            real_inputs.scalar_cache.cache_fingerprint
        ),
        "dataset_free_receipt_fingerprint": (
            dataset_free_receipt_fingerprint
        ),
        "D_R_gate_receipt_fingerprint": dr_gate_receipt_fingerprint,
        "source_closure_fingerprint": source_closure_fingerprint,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "exposure_gate_fingerprint": exposure_gate_fingerprint,
        "exposure_gate_checks": dict(exposure_gate_checks),
        "model_config_fingerprint": model_config_fingerprint,
        "model_contract_fingerprint": model_contract_fingerprint,
        "initial_model_fingerprint": initial_model_fingerprint,
        "expected_parameter_count": expected_parameter_count,
        "training_config_fingerprint": (
            PACRE_PMOPE_TRAINING_CONFIG.config_fingerprint
        ),
    }


@dataclass(frozen=True, eq=False)
class CoverageStatePACREVCFormal800Authorization(
    CoverageStateRunAuthorization,
):
    """Exact authorization for one from-scratch full-D_R Formal800 attempt."""

    run_id: str
    output_claim_fingerprint: str
    real_inputs: CoverageStateRealDRInputs
    dataset_free_receipt_json: str
    dataset_free_receipt_fingerprint: str
    dr_gate_receipt: CoverageStatePACREDRGateReceipt
    dr_gate_receipt_json: str
    dr_gate_receipt_fingerprint: str
    source_closure_json: str
    source_closure_fingerprint: str
    model_config: CoverageStatePACREVerifierCorrectedConfig
    model_config_fingerprint: str
    model_contract_json: str
    model_contract_fingerprint: str
    expected_parameter_count: int
    initial_model_fingerprint: str
    schedule: CoverageStateTrainingSchedule
    exposure_gate_fingerprint: str
    exposure_gate_checks: tuple[tuple[str, bool], ...]
    static_binding_fingerprint: str
    attempt_fingerprint: str
    _preparation_seal: _FormalPreparationSeal = field(
        repr=False,
        compare=False,
    )
    _attempt_token: _FormalAttemptToken = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._validate_lightweight()
        _ = self.authorization_fingerprint

    def _static_binding_payload(self) -> dict[str, object]:
        return _static_binding_payload(
            run_id=self.run_id,
            output_claim_fingerprint=self.output_claim_fingerprint,
            real_inputs=self.real_inputs,
            dataset_free_receipt_fingerprint=(
                self.dataset_free_receipt_fingerprint
            ),
            dr_gate_receipt_fingerprint=self.dr_gate_receipt_fingerprint,
            source_closure_fingerprint=self.source_closure_fingerprint,
            schedule=self.schedule,
            exposure_gate_fingerprint=self.exposure_gate_fingerprint,
            exposure_gate_checks=self.exposure_gate_checks,
            model_config_fingerprint=self.model_config_fingerprint,
            model_contract_fingerprint=self.model_contract_fingerprint,
            initial_model_fingerprint=self.initial_model_fingerprint,
            expected_parameter_count=self.expected_parameter_count,
        )

    def _validate_lightweight(self) -> None:
        frozen_config = CoverageStatePACREVerifierCorrectedConfig(
            feature_channels=PACRE_FORMAL_FEATURE_CHANNELS,
            feature_stride=PACRE_FORMAL_FEATURE_STRIDE,
            width=PACRE_FORMAL_WIDTH,
        )
        expected_static = stable_fingerprint(self._static_binding_payload())
        expected_attempt = stable_fingerprint(
            {
                "schema_version": PACRE_VC_FORMAL_ATTEMPT_SCHEMA,
                "run_id": self.run_id,
                "static_binding_fingerprint": expected_static,
            }
        )
        if (
            self.run_id != PACRE_VC_FORMAL_RUN_ID
            or type(self.real_inputs) is not CoverageStateRealDRInputs
            or type(self.dr_gate_receipt)
            is not CoverageStatePACREDRGateReceipt
            or type(self.model_config)
            is not CoverageStatePACREVerifierCorrectedConfig
            or type(self.schedule) is not CoverageStateTrainingSchedule
            or type(self._preparation_seal) is not _FormalPreparationSeal
            or type(self._attempt_token) is not _FormalAttemptToken
        ):
            raise TypeError("PACRE-VC Formal800 authorization type changed")
        if (
            self._preparation_seal.real_inputs is not self.real_inputs
            or self._preparation_seal.scalar_cache
            is not self.real_inputs.scalar_cache
            or self._preparation_seal.dr_gate_receipt
            is not self.dr_gate_receipt
            or self._preparation_seal.model_config is not self.model_config
            or self._preparation_seal.schedule is not self.schedule
            or self._preparation_seal.static_binding_fingerprint
            != expected_static
            or self.static_binding_fingerprint != expected_static
            or self.attempt_fingerprint != expected_attempt
            or self._attempt_token
            is not _registered_attempt_token(
                attempt_fingerprint=self.attempt_fingerprint,
                static_binding_fingerprint=self.static_binding_fingerprint,
                create=False,
            )
        ):
            raise PermissionError("Formal800 preparation seal changed")
        digests = (
            self.output_claim_fingerprint,
            self.dataset_free_receipt_fingerprint,
            self.dr_gate_receipt_fingerprint,
            self.source_closure_fingerprint,
            self.model_config_fingerprint,
            self.model_contract_fingerprint,
            self.initial_model_fingerprint,
            self.exposure_gate_fingerprint,
            self.static_binding_fingerprint,
            self.attempt_fingerprint,
        )
        if (
            not all(_is_sha256(value) for value in digests)
            or self.real_inputs.scalar_cache.cache_fingerprint
            != PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
            or self.schedule.cache_fingerprint
            != PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
            or self.schedule.config
            != CoverageStateScheduleConfig.formal(
                seed=PACRE_VC_FORMAL_SEED
            )
            or self.schedule.config.updates != PACRE_VC_FORMAL_UPDATES
            or self.schedule.schedule_fingerprint
            != PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT
            or self.exposure_gate_fingerprint
            != PACRE_VC_FORMAL_EXPOSURE_GATE_FINGERPRINT
            or not self.exposure_gate_checks
            or not all(passed for _, passed in self.exposure_gate_checks)
            or self.model_config != frozen_config
            or self.model_config_fingerprint
            != stable_fingerprint(_model_config_payload(frozen_config))
            or self.expected_parameter_count
            != PACRE_FORMAL_PARAMETER_COUNT
            or self.initial_model_fingerprint
            != PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT
            or stable_fingerprint(json.loads(self.model_contract_json))
            != self.model_contract_fingerprint
            or canonical_json(json.loads(self.dataset_free_receipt_json))
            != self.dataset_free_receipt_json
            or canonical_json(json.loads(self.dr_gate_receipt_json))
            != self.dr_gate_receipt_json
            or canonical_json(json.loads(self.source_closure_json))
            != self.source_closure_json
        ):
            raise PermissionError("Formal800 frozen coordinates changed")

    @property
    def prerequisites_passed(self) -> bool:
        return (
            self.dr_gate_receipt.gate_passed
            and self.dr_gate_receipt.decision
            == PACRE_VC_DR_PASS_DECISION
            and len(self.dr_gate_receipt.checks) == 13
            and all(passed for _, passed in self.dr_gate_receipt.checks)
            and bool(self.exposure_gate_checks)
            and all(passed for _, passed in self.exposure_gate_checks)
        )

    @property
    def available(self) -> bool:
        return (
            self.prerequisites_passed
            and self._attempt_token.snapshot()["state"] == "available"
        )

    @property
    def consumed(self) -> bool:
        return self._attempt_token.snapshot()["state"] == "consumed"

    @property
    def attempt_execution_ledger(self) -> dict[str, object]:
        return self._attempt_token.snapshot()

    def canonical_payload(self) -> dict[str, object]:
        self._validate_lightweight()
        return {
            "schema_version": PACRE_VC_FORMAL_AUTHORIZATION_SCHEMA,
            "run_id": self.run_id,
            "output_claim_fingerprint": self.output_claim_fingerprint,
            "scope": COVERAGE_STATE_FORMAL_SCOPE,
            "runtime_splits": ["D_R"],
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "D_R_gate_receipt_fingerprint": (
                self.dr_gate_receipt_fingerprint
            ),
            "D_R_gate_checks": dict(self.dr_gate_receipt.checks),
            "D_R_gate_check_count": len(self.dr_gate_receipt.checks),
            "D_R_gate_decision": self.dr_gate_receipt.decision,
            "source_closure_fingerprint": self.source_closure_fingerprint,
            "real_inputs_fingerprint": self.real_inputs.build_fingerprint,
            "source_binding_fingerprint": (
                self.real_inputs.source_binding.binding_fingerprint
            ),
            "full_D_R_scalar_cache_fingerprint": (
                self.real_inputs.scalar_cache.cache_fingerprint
            ),
            "schedule_fingerprint": self.schedule.schedule_fingerprint,
            "exposure_gate_fingerprint": self.exposure_gate_fingerprint,
            "exposure_gate_checks": dict(self.exposure_gate_checks),
            "budget": {
                "seed": PACRE_VC_FORMAL_SEED,
                "epochs": COVERAGE_STATE_FORMAL_EPOCHS,
                "steps_per_epoch": (
                    COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH
                ),
                "updates": PACRE_VC_FORMAL_UPDATES,
                "objectives": 1,
            },
            "model": {
                "model_fqcn": PACRE_MODEL_FQCN,
                "config_fqcn": PACRE_CONFIG_FQCN,
                "config_fingerprint": self.model_config_fingerprint,
                "contract_fingerprint": self.model_contract_fingerprint,
                "initial_fingerprint": self.initial_model_fingerprint,
                "parameter_count": self.expected_parameter_count,
            },
            "objective": PACRE_PMOPE_OBJECTIVE,
            "objective_policy": CSLF_PMOPE_POLICY,
            "optimizer_fqcn": PACRE_OPTIMIZER_FQCN,
            "training_config_fingerprint": (
                PACRE_PMOPE_TRAINING_CONFIG.config_fingerprint
            ),
            "static_binding_fingerprint": self.static_binding_fingerprint,
            "attempt_fingerprint": self.attempt_fingerprint,
            "authorization_policy": {
                "directly_authorized_by_v23_D_R_13_of_13_PASS": True,
                "bounded_400_required": False,
                "bounded_400_receipt_consumed": False,
                "bounded_400_is_final_success": False,
            },
            "training_contract": {
                "from_scratch": True,
                "process_local_claim_and_consume": True,
                "single_final_model": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "continuation_checkpoint_consumed": False,
                "checkpoint_policy": "final_model_only",
                "intermediate_checkpoint_saved": False,
                "optimizer_state_saved": False,
            },
            "formal_D_R_training_authorized": self.prerequisites_passed,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }

    @cached_property
    def authorization_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        self._validate_lightweight()
        self.real_inputs.verify_unchanged()
        expected_config = _config_from_verified_real_inputs(self.real_inputs)
        dataset_json, dataset_fingerprint = _validated_dataset_free_json(
            json.loads(self.dataset_free_receipt_json)
        )
        closure_json, closure_fingerprint = _validated_source_closure_json(
            json.loads(self.source_closure_json)
        )
        dr_json, dr_fingerprint = _validated_dr_receipt_json(
            self.dr_gate_receipt
        )
        exposure_fingerprint, exposure_checks = _validated_exposure(
            self.real_inputs.scalar_cache,
            self.schedule,
        )
        (
            contract_json,
            contract_fingerprint,
            initial_fingerprint,
            parameter_count,
        ) = _formal_model_binding(self.model_config)
        if (
            expected_config != self.model_config
            or dataset_json != self.dataset_free_receipt_json
            or dataset_fingerprint
            != self.dataset_free_receipt_fingerprint
            or closure_json != self.source_closure_json
            or closure_fingerprint != self.source_closure_fingerprint
            or dr_json != self.dr_gate_receipt_json
            or dr_fingerprint != self.dr_gate_receipt_fingerprint
            or self.dr_gate_receipt.dataset_free_receipt_fingerprint
            != dataset_fingerprint
            or self.dr_gate_receipt.source_closure_fingerprint
            != closure_fingerprint
            or self.dr_gate_receipt.real_inputs_fingerprint
            != self.real_inputs.build_fingerprint
            or self.schedule.cache_fingerprint
            != self.real_inputs.scalar_cache.cache_fingerprint
            or exposure_fingerprint != self.exposure_gate_fingerprint
            or exposure_checks != self.exposure_gate_checks
            or contract_json != self.model_contract_json
            or contract_fingerprint != self.model_contract_fingerprint
            or initial_fingerprint != self.initial_model_fingerprint
            or parameter_count != self.expected_parameter_count
            or stable_fingerprint(self._static_binding_payload())
            != self.static_binding_fingerprint
            or stable_fingerprint(self.canonical_payload())
            != self.authorization_fingerprint
        ):
            raise PermissionError("Formal800 prerequisite binding changed")

    def verify_model_config(
        self,
        model_config: CoverageStatePACREVerifierCorrectedConfig,
    ) -> None:
        if (
            type(model_config)
            is not CoverageStatePACREVerifierCorrectedConfig
            or model_config is not self.model_config
            or stable_fingerprint(_model_config_payload(model_config))
            != self.model_config_fingerprint
            or model_config.expected_parameter_count
            != self.expected_parameter_count
        ):
            raise PermissionError(
                "Formal800 authorization rejects this model config"
            )

    def _training_binding_fingerprint(
        self,
        *,
        model_config: CoverageStatePACREVerifierCorrectedConfig,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
        device: torch.device | str,
    ) -> str:
        self._validate_lightweight()
        if (
            model_config is not self.model_config
            or cache is not self.real_inputs.scalar_cache
            or schedule is not self.schedule
            or scope != COVERAGE_STATE_FORMAL_SCOPE
            or torch.device(device) != torch.device(PACRE_VC_FORMAL_DEVICE)
            or cache.cache_fingerprint
            != PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
            or schedule.schedule_fingerprint
            != PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT
            or not self.prerequisites_passed
        ):
            raise PermissionError(
                "Formal800 authorization rejects this training binding"
            )
        return stable_fingerprint(
            {
                "schema_version": (
                    "cure-lite-v23-pacre-vc-formal800-training-binding-v1"
                ),
                "authorization_fingerprint": self.authorization_fingerprint,
                "attempt_fingerprint": self.attempt_fingerprint,
                "model_contract_fingerprint": (
                    self.model_contract_fingerprint
                ),
                "initial_model_fingerprint": self.initial_model_fingerprint,
                "cache_fingerprint": cache.cache_fingerprint,
                "schedule_fingerprint": schedule.schedule_fingerprint,
                "scope": scope,
                "device": PACRE_VC_FORMAL_DEVICE,
                "objective": PACRE_PMOPE_OBJECTIVE,
                "optimizer_fqcn": PACRE_OPTIMIZER_FQCN,
            }
        )

    def claim_for_training(
        self,
        *,
        model_config: CoverageStatePACREVerifierCorrectedConfig,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
        device: torch.device | str,
    ) -> None:
        binding = self._training_binding_fingerprint(
            model_config=model_config,
            cache=cache,
            schedule=schedule,
            scope=scope,
            device=device,
        )
        self._attempt_token.claim(binding)
        try:
            self.verify_unchanged()
            self.verify_model_config(model_config)
        except BaseException:
            self._attempt_token.fail()
            raise

    def consume_for_training(
        self,
        *,
        model: CURELitePACREVerifierCorrectedLevelSet,
        model_config: CoverageStatePACREVerifierCorrectedConfig,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
        device: torch.device | str,
        objective: str,
        initial_model_fingerprint: str,
    ) -> None:
        binding = self._training_binding_fingerprint(
            model_config=model_config,
            cache=cache,
            schedule=schedule,
            scope=scope,
            device=device,
        )
        self._attempt_token.verify_reserved(binding)
        contract_fingerprint = stable_fingerprint(
            coverage_state_model_contract_payload(model)
        )
        if (
            type(model) is not CURELitePACREVerifierCorrectedLevelSet
            or model.config is not self.model_config
            or objective != PACRE_PMOPE_OBJECTIVE
            or _model_parameter_devices(model)
            != (PACRE_VC_FORMAL_DEVICE,)
            or initial_model_fingerprint != self.initial_model_fingerprint
            or coverage_state_model_fingerprint(model)
            != self.initial_model_fingerprint
            or contract_fingerprint != self.model_contract_fingerprint
            or sum(value.numel() for value in model.parameters())
            != self.expected_parameter_count
        ):
            raise PermissionError(
                "Formal800 authorization rejects the allocated model"
            )
        self._attempt_token.consume(binding)

    def verify_for_run(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> None:
        """Satisfy the generic protected-training boundary after consumption."""

        binding = self._training_binding_fingerprint(
            model_config=self.model_config,
            cache=cache,
            schedule=schedule,
            scope=scope,
            device=PACRE_VC_FORMAL_DEVICE,
        )
        self._attempt_token.verify_consumed(binding)

    def mark_failed(self) -> None:
        self._attempt_token.fail()


def prepare_pacre_vc_formal_800_authorization(
    real_inputs: CoverageStateRealDRInputs,
    model_config: CoverageStatePACREVerifierCorrectedConfig,
    *,
    dataset_free_receipt: Mapping[str, object],
    dr_gate_receipt: CoverageStatePACREDRGateReceipt,
    source_closure: Mapping[str, object],
    output_claim_fingerprint: str,
    run_id: str = PACRE_VC_FORMAL_RUN_ID,
) -> CoverageStatePACREVCFormal800Authorization:
    """Audit the direct D_R prerequisites and issue one Formal800 attempt."""

    if run_id != PACRE_VC_FORMAL_RUN_ID:
        raise PermissionError("PACRE-VC Formal800 run_id is frozen")
    if not _is_sha256(output_claim_fingerprint):
        raise ValueError(
            "output_claim_fingerprint must be one lowercase SHA-256"
        )
    if type(real_inputs) is not CoverageStateRealDRInputs:
        raise TypeError("real_inputs must have exact type CoverageStateRealDRInputs")
    if type(model_config) is not CoverageStatePACREVerifierCorrectedConfig:
        raise TypeError("model_config must have the exact PACRE-VC v23 type")
    if type(dr_gate_receipt) is not CoverageStatePACREDRGateReceipt:
        raise TypeError(
            "dr_gate_receipt must have exact type CoverageStatePACREDRGateReceipt"
        )
    real_inputs.verify_unchanged()
    expected_config = _config_from_verified_real_inputs(real_inputs)
    if model_config != expected_config:
        raise PermissionError("Formal800 model config is not frozen")

    dataset_json, dataset_fingerprint = _validated_dataset_free_json(
        dataset_free_receipt
    )
    closure_json, closure_fingerprint = _validated_source_closure_json(
        source_closure
    )
    dr_json, dr_fingerprint = _validated_dr_receipt_json(dr_gate_receipt)
    if (
        dr_gate_receipt.dataset_free_receipt_fingerprint
        != dataset_fingerprint
        or dr_gate_receipt.source_closure_fingerprint
        != closure_fingerprint
        or dr_gate_receipt.real_inputs_fingerprint
        != real_inputs.build_fingerprint
    ):
        raise PermissionError(
            "D_R receipt is bound to different formal prerequisites"
        )

    cache = real_inputs.scalar_cache
    if (
        cache.cache_fingerprint
        != PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
        or real_inputs.scalar_cache_fingerprint
        != cache.cache_fingerprint
    ):
        raise PermissionError("Formal800 requires the frozen full D_R cache")
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig.formal(seed=PACRE_VC_FORMAL_SEED),
    )
    if (
        type(schedule) is not CoverageStateTrainingSchedule
        or schedule.cache_fingerprint != cache.cache_fingerprint
        or schedule.config
        != CoverageStateScheduleConfig.formal(seed=PACRE_VC_FORMAL_SEED)
        or schedule.schedule_fingerprint
        != PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT
    ):
        raise PermissionError("Formal800 schedule coordinate changed")
    exposure_fingerprint, exposure_checks = _validated_exposure(
        cache,
        schedule,
    )
    if (
        exposure_fingerprint
        != PACRE_VC_FORMAL_EXPOSURE_GATE_FINGERPRINT
    ):
        raise PermissionError("Formal800 exposure coordinate changed")

    model_config_fingerprint = stable_fingerprint(
        _model_config_payload(model_config)
    )
    (
        model_contract_json,
        model_contract_fingerprint,
        initial_model_fingerprint,
        parameter_count,
    ) = _formal_model_binding(model_config)
    if (
        initial_model_fingerprint
        != PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT
        or parameter_count != PACRE_FORMAL_PARAMETER_COUNT
        or type(PACRE_PMOPE_TRAINING_CONFIG)
        is not PACREPMOPETrainingConfig
        or PACRE_PMOPE_TRAINING_CONFIG.seed != PACRE_VC_FORMAL_SEED
    ):
        raise PermissionError("Formal800 model/optimizer coordinate changed")

    static_payload = _static_binding_payload(
        run_id=run_id,
        output_claim_fingerprint=output_claim_fingerprint,
        real_inputs=real_inputs,
        dataset_free_receipt_fingerprint=dataset_fingerprint,
        dr_gate_receipt_fingerprint=dr_fingerprint,
        source_closure_fingerprint=closure_fingerprint,
        schedule=schedule,
        exposure_gate_fingerprint=exposure_fingerprint,
        exposure_gate_checks=exposure_checks,
        model_config_fingerprint=model_config_fingerprint,
        model_contract_fingerprint=model_contract_fingerprint,
        initial_model_fingerprint=initial_model_fingerprint,
        expected_parameter_count=parameter_count,
    )
    static_fingerprint = stable_fingerprint(static_payload)
    attempt_fingerprint = stable_fingerprint(
        {
            "schema_version": PACRE_VC_FORMAL_ATTEMPT_SCHEMA,
            "run_id": run_id,
            "static_binding_fingerprint": static_fingerprint,
        }
    )
    token = _registered_attempt_token(
        attempt_fingerprint=attempt_fingerprint,
        static_binding_fingerprint=static_fingerprint,
        create=True,
    )
    preparation_seal = _FormalPreparationSeal(
        real_inputs=real_inputs,
        scalar_cache=cache,
        dr_gate_receipt=dr_gate_receipt,
        model_config=model_config,
        schedule=schedule,
        static_binding_fingerprint=static_fingerprint,
    )
    result = CoverageStatePACREVCFormal800Authorization(
        run_id=run_id,
        output_claim_fingerprint=output_claim_fingerprint,
        real_inputs=real_inputs,
        dataset_free_receipt_json=dataset_json,
        dataset_free_receipt_fingerprint=dataset_fingerprint,
        dr_gate_receipt=dr_gate_receipt,
        dr_gate_receipt_json=dr_json,
        dr_gate_receipt_fingerprint=dr_fingerprint,
        source_closure_json=closure_json,
        source_closure_fingerprint=closure_fingerprint,
        model_config=model_config,
        model_config_fingerprint=model_config_fingerprint,
        model_contract_json=model_contract_json,
        model_contract_fingerprint=model_contract_fingerprint,
        expected_parameter_count=parameter_count,
        initial_model_fingerprint=initial_model_fingerprint,
        schedule=schedule,
        exposure_gate_fingerprint=exposure_fingerprint,
        exposure_gate_checks=exposure_checks,
        static_binding_fingerprint=static_fingerprint,
        attempt_fingerprint=attempt_fingerprint,
        _preparation_seal=preparation_seal,
        _attempt_token=token,
    )
    return result


def _formal_result_checks(
    authorization: CoverageStatePACREVCFormal800Authorization,
    model: CURELitePACREVerifierCorrectedLevelSet,
    training_result: CoverageStateTrainingResult,
    *,
    optimizer_config_fingerprint: str,
    training_invocations: int,
    source_closure_fingerprint_after: str,
) -> tuple[tuple[str, bool], ...]:
    ledger = authorization.attempt_execution_ledger
    checks = {
        "authorization_claimed_and_consumed_once": (
            ledger.get("state") == "consumed"
            and ledger.get("claim_count") == 1
            and ledger.get("consume_count") == 1
            and ledger.get("failure_count") == 0
        ),
        "direct_D_R_13_of_13_authorization": (
            authorization.dr_gate_receipt.gate_passed
            and authorization.dr_gate_receipt.decision
            == PACRE_VC_DR_PASS_DECISION
            and len(authorization.dr_gate_receipt.checks) == 13
            and all(
                passed
                for _, passed in authorization.dr_gate_receipt.checks
            )
        ),
        "bounded400_not_prerequisite_or_success": (
            authorization.canonical_payload()["authorization_policy"]
            == {
                "directly_authorized_by_v23_D_R_13_of_13_PASS": True,
                "bounded_400_required": False,
                "bounded_400_receipt_consumed": False,
                "bounded_400_is_final_success": False,
            }
        ),
        "full_D_R_cache_and_formal_schedule": (
            training_result.cache_fingerprint
            == authorization.real_inputs.scalar_cache.cache_fingerprint
            == PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
            and training_result.schedule_fingerprint
            == authorization.schedule.schedule_fingerprint
            == PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT
        ),
        "fixed_seed42_800x40_compute_ledger": (
            training_result.seed == PACRE_VC_FORMAL_SEED
            and training_result.epochs == COVERAGE_STATE_FORMAL_EPOCHS
            and training_result.steps_per_epoch
            == COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH
            and training_result.completed_updates
            == PACRE_VC_FORMAL_UPDATES
            and training_result.forward_calls
            == PACRE_VC_FORMAL_UPDATES
            and training_result.backward_calls
            == PACRE_VC_FORMAL_UPDATES
            and training_result.optimizer_steps
            == PACRE_VC_FORMAL_UPDATES
            and training_result.logical_state_evaluations
            == 12 * PACRE_VC_FORMAL_UPDATES
            and training_result.finite_state_audits
            == PACRE_VC_FORMAL_UPDATES + 1
        ),
        "exact_v23_single_PMOPE_model": (
            training_invocations == 1
            and type(model)
            is CURELitePACREVerifierCorrectedLevelSet
            and model.config is authorization.model_config
            and training_result.objective == PACRE_PMOPE_OBJECTIVE
            and training_result.objective_policy == CSLF_PMOPE_POLICY
            and training_result.execution_device
            == PACRE_VC_FORMAL_DEVICE
            and _model_parameter_devices(model)
            == (PACRE_VC_FORMAL_DEVICE,)
            and tuple(name for name, _ in model.named_parameters())
            == PACRE_VC_PARAMETER_NAMES
            and sum(value.numel() for value in model.parameters())
            == authorization.expected_parameter_count
        ),
        "fresh_exact_Adam_policy": (
            _is_sha256(optimizer_config_fingerprint)
            and training_result.optimizer_config_fingerprint
            == optimizer_config_fingerprint
            and type(PACRE_PMOPE_TRAINING_CONFIG)
            is PACREPMOPETrainingConfig
            and PACRE_PMOPE_TRAINING_CONFIG.canonical_payload()
            == PACREPMOPETrainingConfig().canonical_payload()
        ),
        "from_scratch_initial_to_one_final_model": (
            training_result.initial_model_fingerprint
            == authorization.initial_model_fingerprint
            == PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT
            and training_result.final_model_fingerprint
            == coverage_state_model_fingerprint(model)
            and training_result.final_model_fingerprint
            != training_result.initial_model_fingerprint
        ),
        "D_R_only_without_evaluation": (
            authorization.real_inputs.source_binding.split == "D_R"
            and authorization.real_inputs.scalar_cache.raw_catalog.split
            == "D_R"
            and source_closure_fingerprint_after
            == authorization.source_closure_fingerprint
        ),
        "no_resume_or_retry": (
            authorization.canonical_payload()["training_contract"]
            ["resume_allowed"]
            is False
            and authorization.canonical_payload()["training_contract"]
            ["automatic_retry_allowed"]
            is False
        ),
    }
    return tuple(sorted(checks.items()))


_FORMAL_RUN_RESULT_ISSUER = object()


class _FormalRunExecutionSeal:
    """Process-local proof that the Formal800 engine issued this terminal result.

    The seal deliberately binds the exact authorization, model, and common
    training-result objects by identity.  It is created only after the single
    protected trainer invocation returns and all terminal checks have been
    assembled.  Downstream persistence must consume the sealed run result,
    never a caller-assembled ``CoverageStateTrainingResult``.
    """

    __slots__ = (
        "authorization",
        "model",
        "training_result",
        "training_result_fingerprint",
    )

    def __init__(
        self,
        *,
        issuer: object,
        authorization: CoverageStatePACREVCFormal800Authorization,
        model: CURELitePACREVerifierCorrectedLevelSet,
        training_result: CoverageStateTrainingResult,
    ) -> None:
        if issuer is not _FORMAL_RUN_RESULT_ISSUER:
            raise PermissionError(
                "Formal800 execution seals are issued only by the engine"
            )
        self.authorization = authorization
        self.model = model
        self.training_result = training_result
        self.training_result_fingerprint = (
            training_result.result_fingerprint
        )


@dataclass(frozen=True, eq=False)
class CoverageStatePACREVCFormal800RunResult:
    """The single trained in-memory v23 model and its exact compute ledger."""

    authorization: CoverageStatePACREVCFormal800Authorization
    model: CURELitePACREVerifierCorrectedLevelSet
    training_result: CoverageStateTrainingResult
    training_result_fingerprint: str
    optimizer_config_fingerprint: str
    source_closure_fingerprint_after: str
    training_invocations: int
    checks: tuple[tuple[str, bool], ...]
    _execution_seal: _FormalRunExecutionSeal = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.authorization)
            is not CoverageStatePACREVCFormal800Authorization
            or type(self.model)
            is not CURELitePACREVerifierCorrectedLevelSet
            or type(self.training_result) is not CoverageStateTrainingResult
            or self.training_invocations != 1
            or self.checks != tuple(sorted(self.checks))
            or len({name for name, _ in self.checks}) != len(self.checks)
            or not all(type(passed) is bool for _, passed in self.checks)
        ):
            raise TypeError("Formal800 run result contains a wrong type")
        self._verify_execution_seal()
        self._validate_lightweight()
        if not self.training_complete:
            raise RuntimeError("Formal800 compute ledger did not pass")

    def _verify_execution_seal(self) -> None:
        if (
            type(self._execution_seal) is not _FormalRunExecutionSeal
            or self._execution_seal.authorization is not self.authorization
            or self._execution_seal.model is not self.model
            or self._execution_seal.training_result
            is not self.training_result
            or self._execution_seal.training_result_fingerprint
            != self.training_result_fingerprint
        ):
            raise PermissionError(
                "Formal800 run result lacks its engine-issued execution seal"
            )

    @property
    def final_model(self) -> CURELitePACREVerifierCorrectedLevelSet:
        return self.model

    @property
    def training_complete(self) -> bool:
        return bool(self.checks) and all(passed for _, passed in self.checks)

    def _validate_lightweight(self) -> None:
        self._verify_execution_seal()
        expected = _formal_result_checks(
            self.authorization,
            self.model,
            self.training_result,
            optimizer_config_fingerprint=(
                self.optimizer_config_fingerprint
            ),
            training_invocations=self.training_invocations,
            source_closure_fingerprint_after=(
                self.source_closure_fingerprint_after
            ),
        )
        if (
            self.training_result.result_fingerprint
            != self.training_result_fingerprint
            or self.training_result.final_model_fingerprint
            != coverage_state_model_fingerprint(self.model)
            or self.checks != expected
        ):
            raise RuntimeError("Formal800 final model/result binding changed")

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        closure_fingerprint = verify_source_closure(
            json.loads(self.authorization.source_closure_json)
        )
        if closure_fingerprint != self.source_closure_fingerprint_after:
            raise RuntimeError("source closure changed after Formal800")
        self._validate_lightweight()

    def canonical_payload(self) -> dict[str, object]:
        self._validate_lightweight()
        return {
            "schema_version": PACRE_VC_FORMAL_RESULT_SCHEMA,
            "run_id": self.authorization.run_id,
            "runtime_splits": ["D_R"],
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "training_result": self.training_result.canonical_payload(),
            "training_result_fingerprint": (
                self.training_result_fingerprint
            ),
            "final_model_fingerprint": (
                self.training_result.final_model_fingerprint
            ),
            "optimizer_config_fingerprint": (
                self.optimizer_config_fingerprint
            ),
            "source_closure_fingerprint_after": (
                self.source_closure_fingerprint_after
            ),
            "training_invocations": self.training_invocations,
            "checks": dict(self.checks),
            "training_complete": self.training_complete,
            "output_contract": {
                "single_final_model": True,
                "checkpoint_written": False,
                "optimizer_state_returned": False,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
            },
            "bounded_400_required": False,
            "bounded_400_is_final_success": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }

    @property
    def result_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_pacre_vc_pmope_formal_800(
    authorization: CoverageStatePACREVCFormal800Authorization,
    model_config: CoverageStatePACREVerifierCorrectedConfig,
    *,
    device: torch.device | str = PACRE_VC_FORMAL_DEVICE,
    epoch_callback: (
        Callable[[Mapping[str, object]], None] | None
    ) = None,
) -> CoverageStatePACREVCFormal800RunResult:
    """Train exactly once from scratch on full D_R and return one final model."""

    if (
        type(authorization)
        is not CoverageStatePACREVCFormal800Authorization
    ):
        raise TypeError(
            "authorization must be exact "
            "CoverageStatePACREVCFormal800Authorization"
        )
    if type(model_config) is not CoverageStatePACREVerifierCorrectedConfig:
        raise TypeError("model_config must have the exact PACRE-VC v23 type")
    resolved_device = torch.device(device)
    if resolved_device != torch.device(PACRE_VC_FORMAL_DEVICE):
        raise PermissionError(
            f"PACRE-VC Formal800 is frozen to {PACRE_VC_FORMAL_DEVICE}"
        )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PACRE-VC Formal800 requires available cuda:0")
    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError("epoch_callback must be callable or None")
    authorization.verify_model_config(model_config)
    cache = authorization.real_inputs.scalar_cache
    schedule = authorization.schedule
    authorization.claim_for_training(
        model_config=model_config,
        cache=cache,
        schedule=schedule,
        scope=COVERAGE_STATE_FORMAL_SCOPE,
        device=resolved_device,
    )

    try:
        with _deterministic_execution(resolved_device):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(
                    PACRE_VC_FORMAL_SEED
                )
                model = build_pacre_vc_training_model(model_config)
            model = model.to(device=resolved_device, dtype=torch.float32)
            initial_fingerprint = coverage_state_model_fingerprint(model)
            authorization.consume_for_training(
                model=model,
                model_config=model_config,
                cache=cache,
                schedule=schedule,
                scope=COVERAGE_STATE_FORMAL_SCOPE,
                device=resolved_device,
                objective=PACRE_PMOPE_OBJECTIVE,
                initial_model_fingerprint=initial_fingerprint,
            )
            training_config = PACRE_PMOPE_TRAINING_CONFIG
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=training_config.learning_rate,
                betas=(
                    training_config.adam_beta1,
                    training_config.adam_beta2,
                ),
                eps=training_config.adam_epsilon,
                weight_decay=training_config.weight_decay,
            )
            if type(optimizer) is not torch.optim.Adam or optimizer.state:
                raise RuntimeError("Formal800 requires one fresh exact Adam")
            optimizer_config_fingerprint = (
                coverage_state_optimizer_config_fingerprint(model, optimizer)
            )

            # Check the protected 800x40 scope here as well as inside the real
            # generic trainer.  This keeps monkeypatched tests honest without
            # weakening the production engine boundary.
            authorization.verify_for_run(
                cache=cache,
                schedule=schedule,
                scope=COVERAGE_STATE_FORMAL_SCOPE,
            )
            training_invocations = 1
            training_result = train_coverage_state_objective(
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
        if type(training_result) is not CoverageStateTrainingResult:
            raise TypeError(
                "generic Formal800 trainer returned the wrong result type"
            )
        authorization.verify_unchanged()
        closure_fingerprint_after = verify_source_closure(
            json.loads(authorization.source_closure_json)
        )
        checks = _formal_result_checks(
            authorization,
            model,
            training_result,
            optimizer_config_fingerprint=optimizer_config_fingerprint,
            training_invocations=training_invocations,
            source_closure_fingerprint_after=closure_fingerprint_after,
        )
        return CoverageStatePACREVCFormal800RunResult(
            authorization=authorization,
            model=model,
            training_result=training_result,
            training_result_fingerprint=(
                training_result.result_fingerprint
            ),
            optimizer_config_fingerprint=optimizer_config_fingerprint,
            source_closure_fingerprint_after=closure_fingerprint_after,
            training_invocations=training_invocations,
            checks=checks,
            _execution_seal=_FormalRunExecutionSeal(
                issuer=_FORMAL_RUN_RESULT_ISSUER,
                authorization=authorization,
                model=model,
                training_result=training_result,
            ),
        )
    except BaseException:
        authorization.mark_failed()
        raise


# Short compatibility spellings remain aliases to the exact v23 types/functions.
CoverageStatePACREFormal800Authorization = (
    CoverageStatePACREVCFormal800Authorization
)
CoverageStatePACREFormal800RunResult = (
    CoverageStatePACREVCFormal800RunResult
)
expected_pacre_formal_config = expected_pacre_vc_formal_config
prepare_pacre_formal_800_authorization = (
    prepare_pacre_vc_formal_800_authorization
)
run_pacre_pmope_formal_800 = run_pacre_vc_pmope_formal_800


__all__ = [
    "PACRE_VC_FORMAL_AUTHORIZATION_SCHEMA",
    "PACRE_VC_FORMAL_DEVICE",
    "PACRE_VC_FORMAL_EXPOSURE_GATE_FINGERPRINT",
    "PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT",
    "PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT",
    "PACRE_VC_FORMAL_RESULT_SCHEMA",
    "PACRE_VC_FORMAL_RUN_ID",
    "PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT",
    "PACRE_VC_FORMAL_SEED",
    "PACRE_VC_FORMAL_UPDATES",
    "CoverageStatePACREFormal800Authorization",
    "CoverageStatePACREFormal800RunResult",
    "CoverageStatePACREVCFormal800Authorization",
    "CoverageStatePACREVCFormal800RunResult",
    "expected_pacre_formal_config",
    "expected_pacre_vc_formal_config",
    "prepare_pacre_formal_800_authorization",
    "prepare_pacre_vc_formal_800_authorization",
    "run_pacre_pmope_formal_800",
    "run_pacre_vc_pmope_formal_800",
]
