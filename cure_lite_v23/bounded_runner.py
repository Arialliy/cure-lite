"""One-shot D_R-only bounded-400 runner for CURE-Lite v23 PACRE-VC.

The runner binds the generated-only prerequisite, the complete live ``D_R``
receipt graph, the bounded population/cache/schedule, the exact PACRE model
configuration, and the implementation sources before permitting one training
invocation.  Training is delegated only to
``train_pacre_vc_pmope_candidate``.  The trained checkpoint is then evaluated at
the frozen zero level and passed to the PACRE-labelled decision function.

No calibration, threshold search, ``D_V``/``D_T`` read, retry, resume, formal
800 run, or cross-backbone action is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import json
import os
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
)
from cure_lite.coverage_state_sobolev import CSLF_PMOPE_POLICY
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_EPOCHS,
    COVERAGE_STATE_BOUNDED_SEED,
    COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH,
    COVERAGE_STATE_BOUNDED_UPDATES,
    CoverageStateBoundedPreflight,
)
from cure_lite.experiment.coverage_state_bounded_runner import (
    _deterministic_execution,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRInputs,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    CoverageStateRunAuthorization,
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
)
from cure_lite.experiment.coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
    evaluate_coverage_state_zero_level_checkpoint,
)
from cure_lite.frozen_base import module_state_fingerprint

from .decision import (
    PACRE_BOUNDED_RUN_ID,
    CoverageStatePACREBoundedDecision,
    decide_coverage_state_pacre_bounded,
)
from .dr_gate import (
    PACRE_DR_PASS_DECISION,
    CoverageStatePACREDRGateReceipt,
    _validate_dataset_free_receipt,
)
from .factory import build_pacre_vc_training_model
from .pacre_vc import (
    PACRE_VC_CANDIDATE,
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)
from .training import (
    PACRE_CONFIG_FQCN,
    PACRE_MODEL_FQCN,
    PACRE_PMOPE_OBJECTIVE,
    PACRE_PMOPE_TRAINING_CONFIG,
    PACREPMOPETrainingBundle,
    train_pacre_vc_pmope_candidate,
)


PACRE_BOUNDED_AUTHORIZATION_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-pmope-bounded-400-authorization-v1"
)
PACRE_BOUNDED_RESULT_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-pmope-bounded-400-result-v1"
)
PACRE_BOUNDED_DEVICE: Final = "cuda:0"
PACRE_BOUNDED_OUTPUT_REPO_PATH: Final = (
    f"runs/irstd1k_stage_a_seed42/{PACRE_BOUNDED_RUN_ID}"
)
PACRE_BOUNDED_ATTEMPT_RECEIPT_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-pmope-bounded-400-attempt-v1"
)
PACRE_BOUNDED_ATTEMPT_IDENTITY_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-pmope-bounded-400-attempt-identity-v1"
)
PACRE_BOUNDED_VISIBLE_GPU: Final = "0"
PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG: Final = ":4096:8"
PACRE_BOUNDED_PAUSE_TEMPERATURE_C: Final = 82
PACRE_BOUNDED_RESUME_TEMPERATURE_C: Final = 75
PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH: Final = (
    "tools/run_with_gpu_temperature_control.py"
)
PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256: Final = (
    "026b751fbb59530721da1436af32f3bc924c9ed2ab3576df062a45bca7ec5e86"
)
PACRE_BOUNDED_ATTEMPT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "run_id",
        "output_repo_path",
        "config_fingerprint",
        "runtime",
        "candidate",
        "objective",
        "budget",
        "process_identity",
        "dataset_free_receipt_fingerprint",
        "dataset_free_invocations_before_claim",
        "single_attempt",
        "resume_allowed",
        "automatic_retry_allowed",
        "formal_800_authorized",
        "D_V_accessed",
        "D_T_accessed",
        "receipt_fingerprint",
    }
)
PACRE_BOUNDED_ATTEMPT_RUNTIME_FIELDS: Final = frozenset(
    {
        "device",
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "temperature_wrapper_repo_path",
        "temperature_wrapper_file_sha256",
        "pause_temperature_c",
        "resume_temperature_c",
    }
)
_PACRE_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
PACRE_BOUNDED_OUTPUT_PATH = (
    _PACRE_REPO_ROOT / PACRE_BOUNDED_OUTPUT_REPO_PATH
)
PACRE_VC_BOUNDED_AUTHORIZATION_SCHEMA: Final = (
    PACRE_BOUNDED_AUTHORIZATION_SCHEMA
)
PACRE_VC_BOUNDED_RESULT_SCHEMA: Final = PACRE_BOUNDED_RESULT_SCHEMA
PACRE_VC_BOUNDED_RUN_ID: Final = PACRE_BOUNDED_RUN_ID
PACRE_VC_BOUNDED_DEVICE: Final = PACRE_BOUNDED_DEVICE
PACRE_VC_BOUNDED_OUTPUT_REPO_PATH: Final = (
    PACRE_BOUNDED_OUTPUT_REPO_PATH
)


def _package_source_inventory() -> tuple[str, ...]:
    """Return the complete local Python source population for this run.

    The formal path deliberately binds the whole ``cure_lite`` and
    ``cure_lite_v22`` and ``cure_lite_v23`` packages.  This is stricter than
    a hand-maintained import list and therefore also covers dynamic imports,
    package initializers, and transitive data/cache/geometry helpers.
    """

    root = Path(__file__).resolve().parents[1]
    rows: list[str] = []
    for package in ("cure_lite", "cure_lite_v22", "cure_lite_v23"):
        package_root = root / package
        if (
            package_root.is_symlink()
            or not package_root.is_dir()
            or package_root.resolve(strict=True) != package_root
        ):
            raise RuntimeError(
                f"invalid PACRE bounded source package: {package}"
            )
        for path in sorted(package_root.rglob("*.py")):
            if "build" in path.relative_to(package_root).parts:
                continue
            if (
                path.is_symlink()
                or not path.is_file()
                or path.resolve(strict=True) != path
            ):
                raise RuntimeError(
                    "invalid PACRE bounded package source: "
                    f"{path.relative_to(root)}"
                )
            rows.append(str(path.relative_to(root)))
    if not rows or len(rows) != len(set(rows)):
        raise RuntimeError("PACRE bounded package source inventory is invalid")
    return tuple(rows)


PACRE_BOUNDED_IMPLEMENTATION_PATHS: Final = _package_source_inventory()


class _PACREBoundedAttemptToken:
    """One process-local state machine shared by copies of an attempt."""

    def __init__(
        self,
        *,
        attempt_fingerprint: str,
        binding_fingerprint: str,
    ) -> None:
        self.attempt_fingerprint = attempt_fingerprint
        self.binding_fingerprint = binding_fingerprint
        self.lock = Lock()
        self.state = "available"
        self.claim_count = 0
        self.consume_count = 0
        self.failure_count = 0
        self.training_binding_fingerprint: str | None = None

    def claim(self, training_binding_fingerprint: str) -> None:
        with self.lock:
            if self.state != "available":
                raise PermissionError(
                    "PACRE bounded attempt is no longer available"
                )
            if not _is_sha256(training_binding_fingerprint):
                raise ValueError(
                    "PACRE training binding fingerprint is malformed"
                )
            self.state = "reserved"
            self.claim_count += 1
            self.training_binding_fingerprint = (
                training_binding_fingerprint
            )

    def verify_reserved(self, training_binding_fingerprint: str) -> None:
        with self.lock:
            if (
                self.state != "reserved"
                or self.training_binding_fingerprint
                != training_binding_fingerprint
            ):
                raise PermissionError(
                    "PACRE bounded attempt does not hold this reservation"
                )

    def consume(self, training_binding_fingerprint: str) -> None:
        with self.lock:
            if (
                self.state != "reserved"
                or self.training_binding_fingerprint
                != training_binding_fingerprint
            ):
                raise PermissionError(
                    "PACRE bounded attempt was not reserved for this training"
                )
            self.state = "consumed"
            self.consume_count += 1

    def verify_consumed(self, training_binding_fingerprint: str) -> None:
        with self.lock:
            if (
                self.state != "consumed"
                or self.training_binding_fingerprint
                != training_binding_fingerprint
                or self.claim_count != 1
                or self.consume_count != 1
                or self.failure_count != 0
            ):
                raise PermissionError(
                    "PACRE bounded attempt was not consumed by this training"
                )

    def fail(self) -> None:
        """Make any reserved/consumed attempt terminal after an exception."""

        with self.lock:
            if self.state == "failed":
                return
            if self.state not in {"reserved", "consumed"}:
                raise PermissionError(
                    "PACRE bounded attempt cannot enter failed state"
                )
            self.state = "failed"
            self.failure_count += 1

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "state": self.state,
                "claim_count": self.claim_count,
                "consume_count": self.consume_count,
                "failure_count": self.failure_count,
                "training_binding_fingerprint": (
                    self.training_binding_fingerprint
                ),
            }


_ATTEMPT_REGISTRY_LOCK = Lock()
_ATTEMPT_REGISTRY: dict[str, _PACREBoundedAttemptToken] = {}
_MODEL_BINDING_LOCK = Lock()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _registered_attempt_token(
    *,
    run_id: str,
    attempt_fingerprint: str,
    binding_fingerprint: str,
    create: bool,
) -> _PACREBoundedAttemptToken:
    _validate_run_id(run_id)
    if not _is_sha256(attempt_fingerprint) or not _is_sha256(
        binding_fingerprint
    ):
        raise ValueError("PACRE bounded attempt fingerprint is malformed")
    with _ATTEMPT_REGISTRY_LOCK:
        token = _ATTEMPT_REGISTRY.get(run_id)
        if token is None:
            if not create:
                raise RuntimeError(
                    "PACRE bounded attempt token is not registered"
                )
            token = _PACREBoundedAttemptToken(
                attempt_fingerprint=attempt_fingerprint,
                binding_fingerprint=binding_fingerprint,
            )
            _ATTEMPT_REGISTRY[run_id] = token
        elif (
            token.binding_fingerprint != binding_fingerprint
            or token.attempt_fingerprint != attempt_fingerprint
        ):
            raise RuntimeError(
                "PACRE bounded run identity was rebound"
            )
        return token


def _validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    if run_id != PACRE_BOUNDED_RUN_ID:
        raise PermissionError("PACRE bounded run_id is not frozen")
    return run_id


def pacre_bounded_process_identity() -> dict[str, int]:
    """Return a restart-sensitive identity for the current Linux process."""

    process_id = os.getpid()
    raw = (Path("/proc") / str(process_id) / "stat").read_text(
        encoding="utf-8"
    )
    close = raw.rfind(")")
    if close < 0:
        raise RuntimeError("PACRE process identity is unavailable")
    tail = raw[close + 2 :].split()
    if len(tail) <= 19:
        raise RuntimeError("PACRE process start identity is unavailable")
    start_ticks = int(tail[19])
    if process_id < 1 or start_ticks < 1:
        raise RuntimeError("PACRE process identity is invalid")
    return {
        "process_id": process_id,
        "process_start_ticks": start_ticks,
    }


def _validate_output_attempt_receipt(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, int]]:
    if not isinstance(payload, Mapping):
        raise TypeError("PACRE output attempt must be a mapping")
    body = dict(payload)
    if frozenset(body) != PACRE_BOUNDED_ATTEMPT_FIELDS:
        raise ValueError(
            "PACRE output attempt fields differ from the fixed schema"
        )
    fingerprint = body.pop("receipt_fingerprint", None)
    runtime = body.get("runtime")
    budget = body.get("budget")
    process_identity = body.get("process_identity")
    if (
        body.get("schema_version")
        != PACRE_BOUNDED_ATTEMPT_RECEIPT_SCHEMA
        or body.get("run_id") != PACRE_BOUNDED_RUN_ID
        or body.get("output_repo_path")
        != PACRE_BOUNDED_OUTPUT_REPO_PATH
        or not _is_sha256(body.get("config_fingerprint"))
        or body.get("candidate") != PACRE_VC_CANDIDATE
        or body.get("objective") != PACRE_PMOPE_OBJECTIVE
        or budget
        != {
            "seed": COVERAGE_STATE_BOUNDED_SEED,
            "epochs": COVERAGE_STATE_BOUNDED_EPOCHS,
            "steps_per_epoch": COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH,
            "updates": COVERAGE_STATE_BOUNDED_UPDATES,
        }
        or not isinstance(process_identity, Mapping)
        or frozenset(process_identity)
        != {"process_id", "process_start_ticks"}
        or dict(process_identity)
        != pacre_bounded_process_identity()
        or not _is_sha256(
            body.get("dataset_free_receipt_fingerprint")
        )
        or body.get("dataset_free_invocations_before_claim") != 1
        or body.get("single_attempt") is not True
        or body.get("resume_allowed") is not False
        or body.get("automatic_retry_allowed") is not False
        or body.get("formal_800_authorized") is not False
        or body.get("D_V_accessed") is not False
        or body.get("D_T_accessed") is not False
        or not isinstance(runtime, Mapping)
        or frozenset(runtime)
        != PACRE_BOUNDED_ATTEMPT_RUNTIME_FIELDS
        or runtime.get("device") != PACRE_BOUNDED_DEVICE
        or runtime.get("CUDA_VISIBLE_DEVICES")
        != PACRE_BOUNDED_VISIBLE_GPU
        or runtime.get("CUBLAS_WORKSPACE_CONFIG")
        != PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG
        or runtime.get("temperature_wrapper_repo_path")
        != PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH
        or runtime.get("temperature_wrapper_file_sha256")
        != PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256
        or runtime.get("pause_temperature_c")
        != PACRE_BOUNDED_PAUSE_TEMPERATURE_C
        or runtime.get("resume_temperature_c")
        != PACRE_BOUNDED_RESUME_TEMPERATURE_C
        or not _is_sha256(fingerprint)
        or stable_fingerprint(body) != fingerprint
    ):
        raise ValueError("PACRE output attempt receipt is invalid")
    normalized_process_identity = dict(process_identity)
    return str(fingerprint), {
        "process_id": int(normalized_process_identity["process_id"]),
        "process_start_ticks": int(
            normalized_process_identity["process_start_ticks"]
        ),
    }


@dataclass(frozen=True)
class CoverageStatePACREBoundedOutputClaim:
    """Immutable fixed-directory claim required by every protected run."""

    output_repo_path: str
    attempt_receipt_json: str
    attempt_receipt_fingerprint: str
    process_id: int
    process_start_ticks: int

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def canonical_payload(self) -> dict[str, object]:
        attempt = json.loads(self.attempt_receipt_json)
        if not isinstance(attempt, dict):
            raise ValueError("PACRE output attempt must be an object")
        _validate_output_attempt_receipt(attempt)
        return {
            "schema_version": (
                "cure-lite-v23-pacre-vc-bounded-output-claim-v1"
            ),
            "run_id": PACRE_BOUNDED_RUN_ID,
            "output_repo_path": self.output_repo_path,
            "config_fingerprint": attempt["config_fingerprint"],
            "runtime": attempt["runtime"],
            "dataset_free_receipt_fingerprint": (
                attempt["dataset_free_receipt_fingerprint"]
            ),
            "attempt_receipt_fingerprint": (
                self.attempt_receipt_fingerprint
            ),
            "process_identity": {
                "process_id": self.process_id,
                "process_start_ticks": self.process_start_ticks,
            },
            "exclusive_directory_claimed": True,
            "incomplete_marker_present": True,
            "complete_marker_absent": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        }

    @property
    def claim_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        if self.output_repo_path != PACRE_BOUNDED_OUTPUT_REPO_PATH:
            raise PermissionError("PACRE output claim path changed")
        output = PACRE_BOUNDED_OUTPUT_PATH
        if (
            output.is_symlink()
            or not output.is_dir()
            or output.resolve(strict=True) != output
        ):
            raise PermissionError(
                "PACRE fixed output directory is not exclusively claimed"
            )
        attempt_path = output / "attempt.json"
        incomplete_path = output / ".incomplete"
        complete_path = output / "COMPLETE.json"
        failure_path = output / "FAILURE.json"
        if (
            attempt_path.is_symlink()
            or not attempt_path.is_file()
            or incomplete_path.is_symlink()
            or not incomplete_path.is_file()
            or complete_path.exists()
            or complete_path.is_symlink()
            or failure_path.exists()
            or failure_path.is_symlink()
        ):
            raise PermissionError(
                "PACRE fixed output claim is not active"
            )
        raw = attempt_path.read_bytes()
        expected = (self.attempt_receipt_json + "\n").encode("utf-8")
        if raw != expected:
            raise PermissionError("PACRE output attempt bytes changed")
        try:
            attempt = json.loads(self.attempt_receipt_json)
        except json.JSONDecodeError as error:
            raise ValueError("PACRE output attempt JSON is invalid") from error
        if (
            not isinstance(attempt, dict)
            or canonical_json(attempt) != self.attempt_receipt_json
        ):
            raise ValueError(
                "PACRE output attempt must be a canonical object"
            )
        fingerprint, process_identity = _validate_output_attempt_receipt(
            attempt
        )
        if (
            fingerprint != self.attempt_receipt_fingerprint
            or process_identity
            != {
                "process_id": self.process_id,
                "process_start_ticks": self.process_start_ticks,
            }
        ):
            raise PermissionError("PACRE output claim binding changed")


def load_pacre_bounded_output_claim() -> CoverageStatePACREBoundedOutputClaim:
    """Load the active fixed output claim created by the official CLI."""

    attempt_path = PACRE_BOUNDED_OUTPUT_PATH / "attempt.json"
    raw = attempt_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("PACRE output attempt is not canonical JSON")
    text = raw[:-1].decode("utf-8", errors="strict")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("PACRE output attempt is invalid JSON") from error
    if (
        not isinstance(payload, dict)
        or canonical_json(payload) != text
    ):
        raise ValueError("PACRE output attempt is not canonical JSON")
    fingerprint, process_identity = _validate_output_attempt_receipt(
        payload
    )
    return CoverageStatePACREBoundedOutputClaim(
        output_repo_path=PACRE_BOUNDED_OUTPUT_REPO_PATH,
        attempt_receipt_json=text,
        attempt_receipt_fingerprint=fingerprint,
        process_id=process_identity["process_id"],
        process_start_ticks=process_identity["process_start_ticks"],
    )


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[1]
    current_inventory = _package_source_inventory()
    if current_inventory != PACRE_BOUNDED_IMPLEMENTATION_PATHS:
        raise RuntimeError(
            "PACRE bounded package source inventory changed during execution"
        )
    rows: list[tuple[str, str]] = []
    for relative in current_inventory:
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
        ):
            raise RuntimeError(
                f"invalid PACRE bounded source path: {relative}"
            )
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


def _model_binding(
    config: CoverageStatePACREVerifierCorrectedConfig,
) -> tuple[str, str, str, str, int]:
    if type(config) is not CoverageStatePACREVerifierCorrectedConfig:
        raise TypeError(
            "model_config must have exact type "
            "CoverageStatePACREVerifierCorrectedConfig"
        )
    # ``torch.random.fork_rng`` manipulates the process-global CPU generator.
    # Serialize this generated-only construction so concurrent prerequisite
    # checks cannot observe or leave an intermediate RNG state.
    with _MODEL_BINDING_LOCK:
        cpu_state = torch.random.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                COVERAGE_STATE_BOUNDED_SEED
            )
            model = build_pacre_vc_training_model(config)
        if not torch.equal(cpu_state, torch.random.get_rng_state()):
            raise RuntimeError("PACRE model binding changed the CPU RNG state")
    contract = coverage_state_model_contract_payload(model)
    return (
        canonical_json(contract),
        stable_fingerprint(contract),
        coverage_state_model_fingerprint(model),
        f"{type(model).__module__}.{type(model).__qualname__}",
        sum(parameter.numel() for parameter in model.parameters()),
    )


def _model_parameter_devices(
    model: torch.nn.Module,
) -> tuple[str, ...]:
    """Return the actual unique parameter devices for result validation."""

    return tuple(
        sorted({str(parameter.device) for parameter in model.parameters()})
    )


def _validate_population_and_schedule(
    preflight: CoverageStateBoundedPreflight,
    real_inputs: CoverageStateRealDRInputs,
    model_config: CoverageStatePACREVerifierCorrectedConfig,
) -> None:
    if type(preflight) is not CoverageStateBoundedPreflight:
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    if not isinstance(real_inputs, CoverageStateRealDRInputs):
        raise TypeError("real_inputs must be CoverageStateRealDRInputs")
    preflight.verify_unchanged()
    real_inputs.verify_unchanged()
    population = preflight.population
    schedule = preflight.schedule
    natural_records = population.cache.raw_catalog.natural_records
    if (
        not preflight.training_authorized
        or population.seed != COVERAGE_STATE_BOUNDED_SEED
        or population.source_cache is not real_inputs.scalar_cache
        or population.source_cache_fingerprint
        != real_inputs.scalar_cache.cache_fingerprint
        or population.cache.raw_catalog.split != "D_R"
        or real_inputs.source_binding.split != "D_R"
        or real_inputs.scalar_cache.raw_catalog.split != "D_R"
        or schedule.cache_fingerprint
        != population.bounded_cache_fingerprint
        or schedule.config.seed != COVERAGE_STATE_BOUNDED_SEED
        or schedule.config.epochs != COVERAGE_STATE_BOUNDED_EPOCHS
        or schedule.config.steps_per_epoch
        != COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
        or schedule.config.updates != COVERAGE_STATE_BOUNDED_UPDATES
        or not natural_records
        or natural_records[0].feature.shape[1]
        != model_config.feature_channels
        or population.cache.raw_catalog.feature_stride
        != model_config.feature_stride
        or population.cache.sobolev_config.truncation_radius
        != model_config.feature_stride
    ):
        raise PermissionError(
            "PACRE bounded D_R population/cache/schedule binding differs"
        )


def _validated_dr_receipt_mapping(
    receipt: CoverageStatePACREDRGateReceipt,
) -> tuple[dict[str, object], str]:
    """Independently bind the v23 D_R mapping and its fingerprint.

    The bounded authorization does not trust an object's boolean properties
    alone.  It freezes the complete canonical mapping and recomputes its
    digest, then checks the data-access and zero-update terminal policy.
    """

    if type(receipt) is not CoverageStatePACREDRGateReceipt:
        raise TypeError(
            "dr_gate_receipt must be CoverageStatePACREDRGateReceipt"
        )
    mapping = receipt.canonical_payload()
    if not isinstance(mapping, Mapping):
        raise TypeError("PACRE-VC D_R receipt payload must be a mapping")
    payload = dict(mapping)
    fingerprint = receipt.receipt_fingerprint
    if (
        not _is_sha256(fingerprint)
        or stable_fingerprint(payload) != fingerprint
        or payload.get("candidate") != PACRE_VC_CANDIDATE
        or payload.get("gate_passed") is not True
        or payload.get("decision") != PACRE_DR_PASS_DECISION
        or payload.get("D_R_accessed") is not True
        or payload.get("D_V_accessed") is not False
        or payload.get("D_T_accessed") is not False
        or payload.get("training_performed") is not False
    ):
        raise PermissionError(
            "PACRE-VC D_R receipt is not a structural PASS"
        )
    return payload, fingerprint


@dataclass(frozen=True, eq=False)
class CoverageStatePACREBoundedRunAuthorization(
    CoverageStateRunAuthorization,
):
    """Immutable prerequisites plus one process-local consumption token."""

    run_id: str
    output_claim: CoverageStatePACREBoundedOutputClaim
    output_claim_fingerprint: str
    preflight: CoverageStateBoundedPreflight
    real_inputs: CoverageStateRealDRInputs
    dataset_free_receipt_json: str
    dataset_free_receipt_fingerprint: str
    dr_gate_receipt: CoverageStatePACREDRGateReceipt
    dr_gate_receipt_fingerprint: str
    model_config: CoverageStatePACREVerifierCorrectedConfig
    model_contract_json: str
    model_contract_fingerprint: str
    initial_model_fingerprint: str
    expected_parameter_count: int
    implementation_binding: tuple[tuple[str, str], ...]
    implementation_fingerprint: str
    attempt_fingerprint: str
    attempt_binding_fingerprint: str
    _attempt_token: _PACREBoundedAttemptToken = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.verify_unchanged()
        _ = self.authorization_fingerprint

    @property
    def prerequisites_passed(self) -> bool:
        return (
            self.preflight.training_authorized
            and self.dr_gate_receipt.gate_passed
            and self.dr_gate_receipt.decision
            == PACRE_DR_PASS_DECISION
        )

    @property
    def consumed(self) -> bool:
        return self._attempt_token.snapshot()["state"] == "consumed"

    @property
    def reserved(self) -> bool:
        return self._attempt_token.snapshot()["state"] == "reserved"

    @property
    def available(self) -> bool:
        return (
            self.prerequisites_passed
            and self._attempt_token.snapshot()["state"] == "available"
        )

    @property
    def attempt_execution_ledger(self) -> dict[str, object]:
        return self._attempt_token.snapshot()

    def verify_unchanged(self) -> None:
        _validate_run_id(self.run_id)
        if (
            type(self.output_claim)
            is not CoverageStatePACREBoundedOutputClaim
            or type(self.model_config)
            is not CoverageStatePACREVerifierCorrectedConfig
            or type(self.dr_gate_receipt)
            is not CoverageStatePACREDRGateReceipt
        ):
            raise TypeError("PACRE bounded prerequisite type changed")
        self.output_claim.verify_unchanged()
        dataset_free = json.loads(self.dataset_free_receipt_json)
        if not isinstance(dataset_free, dict):
            raise ValueError("dataset-free receipt JSON must be an object")
        dataset_fingerprint = _validate_dataset_free_receipt(
            dataset_free
        )
        _validate_population_and_schedule(
            self.preflight,
            self.real_inputs,
            self.model_config,
        )
        self.dr_gate_receipt.verify_unchanged(
            dataset_free_receipt=dataset_free,
            real_inputs=self.real_inputs,
            bounded_population=self.preflight.population,
        )
        dr_mapping, dr_fingerprint = _validated_dr_receipt_mapping(
            self.dr_gate_receipt
        )
        (
            contract_json,
            contract_fingerprint,
            initial_fingerprint,
            model_fqcn,
            parameter_count,
        ) = _model_binding(self.model_config)
        probe = self.dr_gate_receipt.probe
        dataset_parameter_count = dataset_free.get("parameter_count")
        if (
            dataset_fingerprint
            != self.dataset_free_receipt_fingerprint
            or self.output_claim.claim_fingerprint
            != self.output_claim_fingerprint
            or self.dr_gate_receipt.receipt_fingerprint
            != self.dr_gate_receipt_fingerprint
            or dr_fingerprint != self.dr_gate_receipt_fingerprint
            or canonical_json(dr_mapping)
            != canonical_json(self.dr_gate_receipt.canonical_payload())
            or self.dr_gate_receipt.dataset_free_receipt_fingerprint
            != dataset_fingerprint
            or self.dr_gate_receipt.real_inputs_fingerprint
            != self.real_inputs.build_fingerprint
            or self.dr_gate_receipt.population_fingerprint
            != self.preflight.population.population_fingerprint
            or self.dr_gate_receipt.cache_fingerprint
            != self.preflight.population.bounded_cache_fingerprint
            or not self.prerequisites_passed
            or dataset_parameter_count != parameter_count
            or self.model_contract_json != contract_json
            or self.model_contract_fingerprint != contract_fingerprint
            or self.initial_model_fingerprint != initial_fingerprint
            or self.expected_parameter_count != parameter_count
            or model_fqcn != PACRE_MODEL_FQCN
            or (
                f"{type(self.model_config).__module__}."
                f"{type(self.model_config).__qualname__}"
            )
            != PACRE_CONFIG_FQCN
            or probe.get("execution_seed")
            != COVERAGE_STATE_BOUNDED_SEED
            or probe.get("model_fqcn") != PACRE_MODEL_FQCN
            or probe.get("config_fqcn") != PACRE_CONFIG_FQCN
            or canonical_json(probe.get("model_contract"))
            != contract_json
            or probe.get("model_contract_fingerprint")
            != contract_fingerprint
            or probe.get("initial_model_fingerprint")
            != initial_fingerprint
            or probe.get("final_model_fingerprint")
            != initial_fingerprint
            or probe.get("D_R_accessed") is not True
            or probe.get("D_V_accessed") is not False
            or probe.get("D_T_accessed") is not False
            or probe.get("training_performed") is not False
            or self.implementation_binding
            != _implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self._attempt_token
            is not _registered_attempt_token(
                run_id=self.run_id,
                attempt_fingerprint=self.attempt_fingerprint,
                binding_fingerprint=self.attempt_binding_fingerprint,
                create=False,
            )
        ):
            raise PermissionError(
                "PACRE bounded prerequisite binding changed"
            )

    def canonical_payload(self) -> dict[str, object]:
        dataset_free = json.loads(self.dataset_free_receipt_json)
        dr_mapping, _ = _validated_dr_receipt_mapping(
            self.dr_gate_receipt
        )
        return {
            "schema_version": PACRE_BOUNDED_AUTHORIZATION_SCHEMA,
            "run_id": self.run_id,
            "output_claim": self.output_claim.canonical_payload(),
            "output_claim_fingerprint": self.output_claim_fingerprint,
            "attempt_fingerprint": self.attempt_fingerprint,
            "attempt_binding_fingerprint": (
                self.attempt_binding_fingerprint
            ),
            "scope": COVERAGE_STATE_BOUNDED_SCOPE,
            "runtime_splits": ["D_R"],
            "preflight_fingerprint": (
                self.preflight.preflight_fingerprint
            ),
            "population_fingerprint": (
                self.preflight.population.population_fingerprint
            ),
            "source_cache_fingerprint": (
                self.preflight.population.source_cache_fingerprint
            ),
            "bounded_cache_fingerprint": (
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
            "dataset_free_receipt": dataset_free,
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "real_inputs_fingerprint": (
                self.real_inputs.build_fingerprint
            ),
            "D_R_gate": dr_mapping,
            "D_R_gate_receipt_fingerprint": (
                self.dr_gate_receipt_fingerprint
            ),
            "model": {
                "model_fqcn": PACRE_MODEL_FQCN,
                "config_fqcn": PACRE_CONFIG_FQCN,
                "contract_json": self.model_contract_json,
                "contract_fingerprint": (
                    self.model_contract_fingerprint
                ),
                "initial_fingerprint": (
                    self.initial_model_fingerprint
                ),
                "parameter_count": self.expected_parameter_count,
            },
            "objective": PACRE_PMOPE_OBJECTIVE,
            "objective_policy": CSLF_PMOPE_POLICY,
            "training_config_fingerprint": (
                PACRE_PMOPE_TRAINING_CONFIG.config_fingerprint
            ),
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "implementation_fingerprint": (
                self.implementation_fingerprint
            ),
            "checks": {
                "dataset_free_passed": True,
                "complete_D_R_receipt_reverified": True,
                "bounded_preflight_passed": (
                    self.preflight.training_authorized
                ),
                "exact_pacre_model": True,
                "single_pmope_candidate": True,
                "seed42_10x40": True,
                "fresh_adam_empty_state_required": True,
                "final_checkpoint_only": True,
                "fixed_zero_threshold_without_search": True,
                "D_R_only": True,
                "formal800_requires_separate_preregistration": True,
            },
            "training_authorized": self.prerequisites_passed,
            "single_use": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "formal_800_authorized": False,
            "formal800_status": (
                "BLOCKED_PENDING_SEPARATE_PREREGISTRATION"
            ),
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
        }

    @cached_property
    def authorization_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_model_config(self, model_config: object) -> None:
        if type(model_config) is not CoverageStatePACREVerifierCorrectedConfig:
            raise PermissionError(
                "PACRE authorization rejects this model config"
            )
        contract_json, _, _, model_fqcn, parameter_count = (
            _model_binding(model_config)
        )
        if (
            contract_json != self.model_contract_json
            or model_fqcn != PACRE_MODEL_FQCN
            or parameter_count != self.expected_parameter_count
        ):
            raise PermissionError(
                "PACRE authorization rejects this model config"
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
        if (
            type(model_config) is not CoverageStatePACREVerifierCorrectedConfig
            or model_config != self.model_config
            or scope != COVERAGE_STATE_BOUNDED_SCOPE
            or cache is not self.preflight.population.cache
            or schedule is not self.preflight.schedule
            or cache.cache_fingerprint
            != self.preflight.population.bounded_cache_fingerprint
            or schedule.schedule_fingerprint
            != self.preflight.schedule.schedule_fingerprint
            or torch.device(device) != torch.device(PACRE_BOUNDED_DEVICE)
            or not self.prerequisites_passed
        ):
            raise PermissionError(
                "PACRE authorization rejects this training binding"
            )
        return stable_fingerprint(
            {
                "schema_version": (
                    "cure-lite-v23-pacre-vc-bounded-training-binding-v1"
                ),
                "run_id": self.run_id,
                "attempt_fingerprint": self.attempt_fingerprint,
                "authorization_fingerprint": (
                    self.authorization_fingerprint
                ),
                "model_contract_fingerprint": (
                    self.model_contract_fingerprint
                ),
                "initial_model_fingerprint": (
                    self.initial_model_fingerprint
                ),
                "cache_fingerprint": cache.cache_fingerprint,
                "schedule_fingerprint": (
                    schedule.schedule_fingerprint
                ),
                "scope": scope,
                "device": PACRE_BOUNDED_DEVICE,
                "objective": PACRE_PMOPE_OBJECTIVE,
            }
        )

    def verify_for_run(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> None:
        """Verify the exact training wrapper already consumed this attempt."""

        self.verify_unchanged()
        binding = self._training_binding_fingerprint(
            model_config=self.model_config,
            cache=cache,
            schedule=schedule,
            scope=scope,
            device=PACRE_BOUNDED_DEVICE,
        )
        self._attempt_token.verify_consumed(binding)

    def claim_for_training(
        self,
        *,
        model_config: CoverageStatePACREVerifierCorrectedConfig,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
        device: torch.device | str,
    ) -> None:
        """Atomically reserve before full verification or any CUDA action."""

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

    def verify_reserved_for_training(
        self,
        *,
        model_config: CoverageStatePACREVerifierCorrectedConfig,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
        device: torch.device | str,
    ) -> None:
        """Verify an earlier runner reservation without claiming again."""

        binding = self._training_binding_fingerprint(
            model_config=model_config,
            cache=cache,
            schedule=schedule,
            scope=scope,
            device=device,
        )
        self._attempt_token.verify_reserved(binding)
        self.verify_unchanged()

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
        """Bind the reserved attempt to the actual model before optimization."""

        binding = self._training_binding_fingerprint(
            model_config=model_config,
            cache=cache,
            schedule=schedule,
            scope=scope,
            device=device,
        )
        self._attempt_token.verify_reserved(binding)
        if (
            type(model) is not
            CURELitePACREVerifierCorrectedLevelSet
            or model.config is not self.model_config
            or objective != PACRE_PMOPE_OBJECTIVE
            or _model_parameter_devices(model)
            != (PACRE_BOUNDED_DEVICE,)
            or initial_model_fingerprint
            != self.initial_model_fingerprint
            or coverage_state_model_fingerprint(model)
            != self.initial_model_fingerprint
        ):
            raise PermissionError(
                "PACRE authorization rejects the allocated training state"
            )
        self._attempt_token.consume(binding)

    def mark_failed(self) -> None:
        """Record a terminal failed attempt; no retry can reuse the token."""

        self._attempt_token.fail()


def prepare_pacre_bounded_run_authorization(
    preflight: CoverageStateBoundedPreflight,
    dataset_free_receipt: Mapping[str, object],
    dr_gate_receipt: CoverageStatePACREDRGateReceipt,
    real_inputs: CoverageStateRealDRInputs,
    model_config: CoverageStatePACREVerifierCorrectedConfig,
    *,
    output_claim: CoverageStatePACREBoundedOutputClaim,
    run_id: str,
) -> CoverageStatePACREBoundedRunAuthorization:
    """Bind every prerequisite for the unique PACRE bounded-400 run."""

    _validate_run_id(run_id)
    if type(output_claim) is not CoverageStatePACREBoundedOutputClaim:
        raise TypeError(
            "output_claim must be an exact PACRE bounded output claim"
        )
    output_claim.verify_unchanged()
    if not isinstance(dataset_free_receipt, Mapping):
        raise TypeError("dataset_free_receipt must be a mapping")
    dataset_fingerprint = _validate_dataset_free_receipt(
        dataset_free_receipt
    )
    attempt_payload = json.loads(output_claim.attempt_receipt_json)
    if (
        not isinstance(attempt_payload, dict)
        or attempt_payload.get("dataset_free_receipt_fingerprint")
        != dataset_fingerprint
    ):
        raise PermissionError(
            "PACRE output claim is bound to another dataset-free receipt"
        )
    if type(dr_gate_receipt) is not CoverageStatePACREDRGateReceipt:
        raise TypeError(
            "dr_gate_receipt must be CoverageStatePACREDRGateReceipt"
        )
    _validate_population_and_schedule(
        preflight,
        real_inputs,
        model_config,
    )
    dr_gate_receipt.verify_unchanged(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population=preflight.population,
    )
    dr_mapping, dr_fingerprint = _validated_dr_receipt_mapping(
        dr_gate_receipt
    )
    (
        contract_json,
        contract_fingerprint,
        initial_fingerprint,
        model_fqcn,
        parameter_count,
    ) = _model_binding(model_config)
    if model_fqcn != PACRE_MODEL_FQCN:
        raise AssertionError("PACRE factory constructed a wrong model")
    implementation = _implementation_binding()
    implementation_fingerprint = stable_fingerprint(
        dict(implementation)
    )
    attempt_binding_fingerprint = stable_fingerprint(
        {
            "schema_version": (
                "cure-lite-v23-pacre-vc-bounded-attempt-binding-v1"
            ),
            "run_id": run_id,
            "output_claim_fingerprint": output_claim.claim_fingerprint,
            "attempt_receipt_fingerprint": (
                output_claim.attempt_receipt_fingerprint
            ),
            "preflight_fingerprint": preflight.preflight_fingerprint,
            "dataset_free_receipt_fingerprint": dataset_fingerprint,
            "D_R_gate_receipt_fingerprint": (
                dr_fingerprint
            ),
            "D_R_gate_mapping_fingerprint": stable_fingerprint(
                dr_mapping
            ),
            "real_inputs_fingerprint": real_inputs.build_fingerprint,
            "model_contract_fingerprint": contract_fingerprint,
            "initial_model_fingerprint": initial_fingerprint,
            "implementation_fingerprint": implementation_fingerprint,
        }
    )
    attempt_fingerprint = stable_fingerprint(
        {
            "schema_version": PACRE_BOUNDED_ATTEMPT_IDENTITY_SCHEMA,
            "run_id": run_id,
            "attempt_binding_fingerprint": (
                attempt_binding_fingerprint
            ),
        }
    )
    attempt_token = _registered_attempt_token(
        run_id=run_id,
        attempt_fingerprint=attempt_fingerprint,
        binding_fingerprint=attempt_binding_fingerprint,
        create=True,
    )
    result = CoverageStatePACREBoundedRunAuthorization(
        run_id=run_id,
        output_claim=output_claim,
        output_claim_fingerprint=output_claim.claim_fingerprint,
        preflight=preflight,
        real_inputs=real_inputs,
        dataset_free_receipt_json=canonical_json(
            dict(dataset_free_receipt)
        ),
        dataset_free_receipt_fingerprint=dataset_fingerprint,
        dr_gate_receipt=dr_gate_receipt,
        dr_gate_receipt_fingerprint=(
            dr_fingerprint
        ),
        model_config=model_config,
        model_contract_json=contract_json,
        model_contract_fingerprint=contract_fingerprint,
        initial_model_fingerprint=initial_fingerprint,
        expected_parameter_count=parameter_count,
        implementation_binding=implementation,
        implementation_fingerprint=implementation_fingerprint,
        attempt_fingerprint=attempt_fingerprint,
        attempt_binding_fingerprint=attempt_binding_fingerprint,
        _attempt_token=attempt_token,
    )
    result.verify_unchanged()
    return result


def _bounded_result_checks(
    authorization: CoverageStatePACREBoundedRunAuthorization,
    training: PACREPMOPETrainingBundle,
    diagnostic: CoverageStateZeroLevelEvaluationResult,
    decision: CoverageStatePACREBoundedDecision,
    *,
    run_id: str,
    training_invocations: int,
    evaluation_invocations: int,
    decision_invocations: int,
) -> tuple[tuple[str, bool], ...]:
    result = training.training_result
    parameter_devices = _model_parameter_devices(training.model)
    expected_ledger = {
        "state": "consumed",
        "claim_count": 1,
        "consume_count": 1,
        "failure_count": 0,
        "training_binding_fingerprint": (
            authorization._training_binding_fingerprint(
                model_config=authorization.model_config,
                cache=authorization.preflight.population.cache,
                schedule=authorization.preflight.schedule,
                scope=COVERAGE_STATE_BOUNDED_SCOPE,
                device=PACRE_BOUNDED_DEVICE,
            )
        ),
    }
    recomputed_decision = decide_coverage_state_pacre_bounded(
        diagnostic,
        run_id=run_id,
    )
    checks = {
        "01_authorization_consumed_once": (
            authorization.attempt_execution_ledger == expected_ledger
            and training_invocations == 1
        ),
        "02_exact_pacre_single_candidate": (
            type(training.model)
            is
            CURELitePACREVerifierCorrectedLevelSet
            and type(training.model_config)
            is CoverageStatePACREVerifierCorrectedConfig
            and training.model_config == authorization.model_config
            and result.objective == PACRE_PMOPE_OBJECTIVE
        ),
        "03_seed42_10x40_compute_ledger": (
            result.seed == COVERAGE_STATE_BOUNDED_SEED
            and result.epochs == COVERAGE_STATE_BOUNDED_EPOCHS
            and result.steps_per_epoch
            == COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
            and result.completed_updates
            == COVERAGE_STATE_BOUNDED_UPDATES
            and result.forward_calls == COVERAGE_STATE_BOUNDED_UPDATES
            and result.backward_calls == COVERAGE_STATE_BOUNDED_UPDATES
            and result.optimizer_steps == COVERAGE_STATE_BOUNDED_UPDATES
            and result.execution_device == PACRE_BOUNDED_DEVICE
            and parameter_devices == (PACRE_BOUNDED_DEVICE,)
            and result.logical_state_evaluations
            == COVERAGE_STATE_BOUNDED_UPDATES * 12
            and result.finite_state_audits
            == COVERAGE_STATE_BOUNDED_UPDATES + 1
        ),
        "04_population_cache_schedule_bound": (
            result.cache_fingerprint
            == authorization.preflight.population.bounded_cache_fingerprint
            and result.schedule_fingerprint
            == authorization.preflight.schedule.schedule_fingerprint
            and training.receipt.initial_model_fingerprint
            == authorization.initial_model_fingerprint
            and training.receipt.parameter_count
            == authorization.expected_parameter_count
        ),
        "05_read_only_D_R_zero_level_evaluation": (
            evaluation_invocations == 1
            and diagnostic.split == "D_R"
            and diagnostic.cache_fingerprint
            == authorization.preflight.population.bounded_cache_fingerprint
            and diagnostic.checkpoint_fingerprint
            == module_state_fingerprint(training.model)
            and diagnostic.backward_calls == 0
            and diagnostic.optimizer_steps == 0
            and diagnostic.config.split == "D_R"
            and diagnostic.config.residual_threshold == 0.0
            and diagnostic.config.threshold_search_performed is False
            and diagnostic.config.training_performed is False
            and diagnostic.config.input_representation
            == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
            and diagnostic.config.d_v_accessed is False
            and diagnostic.config.d_t_accessed is False
        ),
        "06_exact_pacre_decision": (
            decision_invocations == 1
            and decision.run_id == run_id == PACRE_BOUNDED_RUN_ID
            and decision.diagnostic is diagnostic
            and decision.canonical_payload()
            == recomputed_decision.canonical_payload()
            and decision.decision_fingerprint
            == recomputed_decision.decision_fingerprint
        ),
        "07_D_R_only_no_retry_or_resume": (
            authorization.attempt_execution_ledger == expected_ledger
        ),
        "08_decision_gate_passed": decision.bounded_gate_passed,
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True, eq=False)
class CoverageStatePACREBoundedRunResult:
    """One completed PACRE bounded run and its read-only decision."""

    run_id: str
    authorization: CoverageStatePACREBoundedRunAuthorization
    training: PACREPMOPETrainingBundle
    diagnostic: CoverageStateZeroLevelEvaluationResult
    decision: CoverageStatePACREBoundedDecision
    training_invocations: int
    zero_level_evaluation_invocations: int
    decision_invocations: int
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if (
            type(self.authorization)
            is not CoverageStatePACREBoundedRunAuthorization
            or type(self.training) is not PACREPMOPETrainingBundle
            or type(self.diagnostic)
            is not CoverageStateZeroLevelEvaluationResult
            or type(self.decision) is not CoverageStatePACREBoundedDecision
            or self.checks != tuple(sorted(self.checks))
            or len({name for name, _ in self.checks})
            != len(self.checks)
        ):
            raise TypeError("PACRE bounded result type changed")
        self.verify_unchanged()

    @property
    def bounded_gate_passed(self) -> bool:
        return bool(self.checks) and all(
            passed for _, passed in self.checks
        )

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
        if (
            not self.authorization.consumed
            or self.decision.decision_fingerprint
            != stable_fingerprint(self.decision.canonical_payload())
            or self.checks
            != _bounded_result_checks(
                self.authorization,
                self.training,
                self.diagnostic,
                self.decision,
                run_id=self.run_id,
                training_invocations=self.training_invocations,
                evaluation_invocations=(
                    self.zero_level_evaluation_invocations
                ),
                decision_invocations=self.decision_invocations,
            )
        ):
            raise RuntimeError("PACRE bounded result binding changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": PACRE_BOUNDED_RESULT_SCHEMA,
            "run_id": self.run_id,
            "runtime_splits": ["D_R"],
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "candidate": PACRE_VC_CANDIDATE,
            "objective": PACRE_PMOPE_OBJECTIVE,
            "objective_policy": CSLF_PMOPE_POLICY,
            "training": {
                "bundle_fingerprint": (
                    self.training.bundle_fingerprint
                ),
                "receipt": self.training.receipt.canonical_payload(),
                "result": (
                    self.training.training_result.canonical_payload()
                ),
            },
            "diagnostic": self.diagnostic.canonical_payload(),
            "decision": self.decision.canonical_payload(),
            "execution_invocations": {
                "training": self.training_invocations,
                "zero_level_evaluation": (
                    self.zero_level_evaluation_invocations
                ),
                "decision": self.decision_invocations,
            },
            "attempt_execution_ledger": (
                self.authorization.attempt_execution_ledger
            ),
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "bounded_gate_passed": self.bounded_gate_passed,
            "formal800_eligible": self.formal800_eligible,
            "formal800_status": (
                "BLOCKED_PENDING_SEPARATE_PREREGISTRATION"
            ),
            "D_R_accessed": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "threshold_search_performed": False,
            "checkpoint_selection": "final_checkpoint_only",
            "optimizer_initial_state": "fresh_empty_adam",
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "formal_800_executed": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
        }

    @cached_property
    def result_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_pacre_pmope_bounded_400(
    authorization: CoverageStatePACREBoundedRunAuthorization,
    model_config: CoverageStatePACREVerifierCorrectedConfig,
    *,
    run_id: str,
    device: torch.device | str,
) -> CoverageStatePACREBoundedRunResult:
    """Execute training, D_R zero-level evaluation, and decision once."""

    if type(authorization) is not (
        CoverageStatePACREBoundedRunAuthorization
    ):
        raise TypeError("authorization must be an exact PACRE authorization")
    _validate_run_id(run_id)
    if run_id != authorization.run_id:
        raise PermissionError(
            "PACRE run_id differs from its authorization"
        )
    resolved_device = torch.device(device)
    if resolved_device != torch.device(PACRE_BOUNDED_DEVICE):
        raise PermissionError(
            f"PACRE bounded-400 is frozen to {PACRE_BOUNDED_DEVICE}"
        )
    # Reserve before entering the deterministic context.  That context reads
    # CUDA RNG state, so no competing attempt may reach it first.
    authorization.claim_for_training(
        model_config=model_config,
        cache=authorization.preflight.population.cache,
        schedule=authorization.preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
        device=resolved_device,
    )

    training_invocations = 0
    evaluation_invocations = 0
    decision_invocations = 0
    try:
        with _deterministic_execution(resolved_device):
            training_invocations += 1
            training = train_pacre_vc_pmope_candidate(
                model_config,
                authorization.preflight.population.cache,
                authorization.preflight.schedule,
                config=PACRE_PMOPE_TRAINING_CONFIG,
                device=resolved_device,
                authorization=authorization,
            )
            if not authorization.consumed:
                raise RuntimeError(
                    "PACRE training did not consume its one-shot authorization"
                )
            if type(training) is not PACREPMOPETrainingBundle:
                raise TypeError("PACRE training returned the wrong bundle")
            training.verify_unchanged()
            model = training.model.eval()

            evaluation_invocations += 1
            diagnostic = evaluate_coverage_state_zero_level_checkpoint(
                model,
                authorization.preflight.population.cache,
                device=resolved_device,
                config=CoverageStateZeroLevelEvaluationConfig(
                    input_representation=(
                        COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                    )
                ),
            )
            if type(diagnostic) is not CoverageStateZeroLevelEvaluationResult:
                raise TypeError("PACRE evaluation returned the wrong result")

            decision_invocations += 1
            decision = decide_coverage_state_pacre_bounded(
                diagnostic,
                run_id=run_id,
            )
            if type(decision) is not CoverageStatePACREBoundedDecision:
                raise TypeError("PACRE decision returned the wrong result")
        checks = _bounded_result_checks(
            authorization,
            training,
            diagnostic,
            decision,
            run_id=run_id,
            training_invocations=training_invocations,
            evaluation_invocations=evaluation_invocations,
            decision_invocations=decision_invocations,
        )
        return CoverageStatePACREBoundedRunResult(
            run_id=run_id,
            authorization=authorization,
            training=training,
            diagnostic=diagnostic,
            decision=decision,
            training_invocations=training_invocations,
            zero_level_evaluation_invocations=evaluation_invocations,
            decision_invocations=decision_invocations,
            checks=checks,
        )
    except BaseException:
        authorization.mark_failed()
        raise


CoverageStatePACREVCBoundedOutputClaim = (
    CoverageStatePACREBoundedOutputClaim
)
CoverageStatePACREVCBoundedRunAuthorization = (
    CoverageStatePACREBoundedRunAuthorization
)
CoverageStatePACREVCBoundedRunResult = CoverageStatePACREBoundedRunResult
load_pacre_vc_bounded_output_claim = load_pacre_bounded_output_claim
prepare_pacre_vc_bounded_run_authorization = (
    prepare_pacre_bounded_run_authorization
)
run_pacre_vc_pmope_bounded_400 = run_pacre_pmope_bounded_400


__all__ = [
    "PACRE_BOUNDED_ATTEMPT_FIELDS",
    "PACRE_BOUNDED_ATTEMPT_RECEIPT_SCHEMA",
    "PACRE_BOUNDED_ATTEMPT_RUNTIME_FIELDS",
    "PACRE_BOUNDED_AUTHORIZATION_SCHEMA",
    "PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG",
    "PACRE_BOUNDED_DEVICE",
    "PACRE_BOUNDED_IMPLEMENTATION_PATHS",
    "PACRE_BOUNDED_OUTPUT_PATH",
    "PACRE_BOUNDED_OUTPUT_REPO_PATH",
    "PACRE_BOUNDED_PAUSE_TEMPERATURE_C",
    "PACRE_BOUNDED_RESUME_TEMPERATURE_C",
    "PACRE_BOUNDED_RESULT_SCHEMA",
    "PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256",
    "PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH",
    "PACRE_BOUNDED_VISIBLE_GPU",
    "PACRE_VC_BOUNDED_AUTHORIZATION_SCHEMA",
    "PACRE_VC_BOUNDED_DEVICE",
    "PACRE_VC_BOUNDED_OUTPUT_REPO_PATH",
    "PACRE_VC_BOUNDED_RESULT_SCHEMA",
    "PACRE_VC_BOUNDED_RUN_ID",
    "CoverageStatePACREBoundedOutputClaim",
    "CoverageStatePACREBoundedRunAuthorization",
    "CoverageStatePACREBoundedRunResult",
    "CoverageStatePACREVCBoundedOutputClaim",
    "CoverageStatePACREVCBoundedRunAuthorization",
    "CoverageStatePACREVCBoundedRunResult",
    "load_pacre_bounded_output_claim",
    "load_pacre_vc_bounded_output_claim",
    "pacre_bounded_process_identity",
    "prepare_pacre_bounded_run_authorization",
    "prepare_pacre_vc_bounded_run_authorization",
    "run_pacre_pmope_bounded_400",
    "run_pacre_vc_pmope_bounded_400",
]
