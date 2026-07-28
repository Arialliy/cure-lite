#!/usr/bin/env python3
"""Validate or run the unique PACRE-VC v23 Formal800 attempt.

``--validate-create-only`` verifies the fixed runtime, live source closure,
dataset-free metadata, and the terminal 13/13 ``D_R`` receipt.  It neither
claims the formal output nor constructs the full ``D_R`` tensor graph.

``--run-once`` claims the output before constructing that graph, re-verifies
the terminal ``D_R`` evidence, and performs exactly one seed-42,
800-epoch-by-40-step PMOPE/Adam invocation.  It writes one final safetensors
model and no optimizer or intermediate checkpoint.  There is no retry,
resume, overwrite, bounded-400, ``D_V``, ``D_T``, calibration, inference, or
performance-evaluation path in this command.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import BinaryIO, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite_v23.authorization import (
    frozen_real_dr_source_paths,
    protocol_root,
)
from cure_lite_v23.bounded_runner import (
    PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG,
    PACRE_BOUNDED_OUTPUT_PATH,
    PACRE_BOUNDED_PAUSE_TEMPERATURE_C,
    PACRE_BOUNDED_RESUME_TEMPERATURE_C,
    PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256,
    PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH,
    PACRE_BOUNDED_VISIBLE_GPU,
)
from cure_lite_v23.dr_gate import (
    PACRE_VC_DR_CHECK_NAMES,
    PACRE_VC_DR_PASS_DECISION,
    CoverageStatePACREDRGateReceipt,
    pacre_vc_dr_receipt_from_payload,
)
from cure_lite_v23.environment import (
    stabilize_pacre_vc_numerical_runtime,
)
from cure_lite_v23.formal_artifacts import (
    save_pacre_vc_formal_final_model,
)
from cure_lite_v23.formal_training import (
    PACRE_VC_FORMAL_DEVICE,
    PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT,
    PACRE_VC_FORMAL_RUN_ID,
    PACRE_VC_FORMAL_SEED,
    PACRE_VC_FORMAL_UPDATES,
    expected_pacre_vc_formal_config,
    prepare_pacre_vc_formal_800_authorization,
    run_pacre_vc_pmope_formal_800,
)
from cure_lite_v23.protocol import (
    fingerprinted,
    read_strict_json,
    strict_json_bytes,
    verify_fingerprinted,
    verify_source_closure,
    write_new_json,
)
from cure_lite_v23.training import PACRE_PMOPE_OBJECTIVE
from tools.verify_cure_lite_v23_pacre_vc_dr_receipt import (
    verify_terminal as verify_dr_terminal,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = PACRE_VC_FORMAL_RUN_ID
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = ROOT / OUTPUT_REPO_PATH
BOUNDED_OUTPUT_PATH = PACRE_BOUNDED_OUTPUT_PATH

FROZEN_DEVICE = PACRE_VC_FORMAL_DEVICE
FROZEN_VISIBLE_GPU = PACRE_BOUNDED_VISIBLE_GPU
FROZEN_CUBLAS_WORKSPACE_CONFIG = PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG
FROZEN_PYTHONHASHSEED = "0"
FROZEN_PAUSE_TEMPERATURE_C = PACRE_BOUNDED_PAUSE_TEMPERATURE_C
FROZEN_RESUME_TEMPERATURE_C = PACRE_BOUNDED_RESUME_TEMPERATURE_C
FROZEN_SEED = PACRE_VC_FORMAL_SEED
FROZEN_EPOCHS = 800
FROZEN_STEPS_PER_EPOCH = 40
FROZEN_UPDATES = PACRE_VC_FORMAL_UPDATES
FROZEN_PARAMETER_COUNT = 64_064

INCOMPLETE_FILE = ".incomplete"
FINAL_MODEL_DIRECTORY = "final_model"
DATASET_FREE_FILE = "dataset_free_receipt.json"
SOURCE_CLOSURE_FILE = "implementation_closure.json"

VALIDATION_SCHEMA = (
    "cure-lite-v23-pacre-vc-formal800-create-only-validation-v1"
)
ATTEMPT_SCHEMA = "cure-lite-v23-pacre-vc-formal800-cli-attempt-v1"
DR_TERMINAL_SCHEMA = (
    "cure-lite-v23-pacre-vc-formal800-D_R-terminal-binding-v1"
)
CONFIG_SCHEMA = "cure-lite-v23-pacre-vc-formal800-config-v1"
INPUTS_SCHEMA = "cure-lite-v23-pacre-vc-formal800-inputs-v1"
AUTHORIZATION_SCHEMA = (
    "cure-lite-v23-pacre-vc-formal800-authorization-wrapper-v1"
)
EPOCH_PROGRESS_SCHEMA = (
    "cure-lite-v23-pacre-vc-formal800-epoch-progress-v1"
)
TRAINING_SCHEMA = "cure-lite-v23-pacre-vc-formal800-training-v1"
DECISION_SCHEMA = "cure-lite-v23-pacre-vc-formal800-decision-v1"
COMPLETE_SCHEMA = "cure-lite-v23-pacre-vc-formal800-complete-v1"
FAILURE_SCHEMA = "cure-lite-v23-pacre-vc-formal800-failure-v1"

EXPECTED_DIRECTORIES = frozenset({"receipts", FINAL_MODEL_DIRECTORY})
EXPECTED_SCIENTIFIC_FILES = frozenset(
    {
        "attempt.json",
        "receipts/dr_terminal_verification.json",
        "receipts/config.json",
        "receipts/inputs.json",
        "receipts/authorization.json",
        "receipts/epoch_progress.jsonl",
        "receipts/training.json",
        "receipts/decision.json",
        "final_model/model.safetensors",
        "final_model/artifact.json",
    }
)


@dataclass(frozen=True)
class _StaticContext:
    runtime: dict[str, object]
    source_closure: dict[str, object]
    source_closure_fingerprint: str
    dataset_free: dict[str, object]
    dataset_free_fingerprint: str
    dr_verification: dict[str, object]
    dr_verification_fingerprint: str
    dr_receipt: CoverageStatePACREDRGateReceipt

    @property
    def binding_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "runtime": self.runtime,
                "source_closure_fingerprint": (
                    self.source_closure_fingerprint
                ),
                "dataset_free_fingerprint": (
                    self.dataset_free_fingerprint
                ),
                "D_R_terminal_verification_fingerprint": (
                    self.dr_verification_fingerprint
                ),
                "D_R_gate_receipt_fingerprint": (
                    self.dr_receipt.receipt_fingerprint
                ),
            }
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _flag_value(tokens: Sequence[str], name: str) -> str:
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == name:
            if index + 1 >= len(tokens):
                raise RuntimeError(f"temperature wrapper lacks {name}")
            values.append(tokens[index + 1])
        elif token.startswith(f"{name}="):
            values.append(token.split("=", maxsplit=1)[1])
    if len(values) != 1:
        raise RuntimeError(
            f"temperature wrapper must specify {name} exactly once"
        )
    return values[0]


def _runtime_contract() -> dict[str, object]:
    """Verify the fixed CUDA/environment/wrapper parent contract."""

    expected_environment = {
        "CUDA_VISIBLE_DEVICES": FROZEN_VISIBLE_GPU,
        "CUBLAS_WORKSPACE_CONFIG": FROZEN_CUBLAS_WORKSPACE_CONFIG,
        "PYTHONHASHSEED": FROZEN_PYTHONHASHSEED,
    }
    for name, expected in expected_environment.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(
                f"Formal800 fixes {name}={expected}"
            )
    cuda = _visible_cuda_contract()

    wrapper = ROOT / PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH
    if (
        not wrapper.is_file()
        or wrapper.is_symlink()
        or wrapper.resolve(strict=True) != wrapper
        or file_sha256(wrapper)
        != PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256
    ):
        raise RuntimeError("fixed temperature wrapper changed")

    cmdline = Path("/proc") / str(os.getppid()) / "cmdline"
    tokens = tuple(
        value.decode("utf-8", errors="strict")
        for value in cmdline.read_bytes().split(b"\0")
        if value
    )
    candidates: list[Path] = []
    for token in tokens:
        if not token.endswith("run_with_gpu_temperature_control.py"):
            continue
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            candidates.append(candidate.resolve(strict=True))
        except OSError:
            pass
    if candidates != [wrapper.resolve(strict=True)]:
        raise RuntimeError(
            "Formal800 requires the fixed GPU temperature wrapper parent"
        )
    expected_flags = {
        "--gpu": FROZEN_VISIBLE_GPU,
        "--pause-temp": str(FROZEN_PAUSE_TEMPERATURE_C),
        "--resume-temp": str(FROZEN_RESUME_TEMPERATURE_C),
    }
    for name, expected in expected_flags.items():
        if _flag_value(tokens, name) != expected:
            raise RuntimeError(f"temperature wrapper {name} changed")
    return {
        "device": FROZEN_DEVICE,
        **expected_environment,
        **cuda,
        "temperature_wrapper_repo_path": (
            PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH
        ),
        "temperature_wrapper_file_sha256": (
            PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256
        ),
        "pause_temperature_c": FROZEN_PAUSE_TEMPERATURE_C,
        "resume_temperature_c": FROZEN_RESUME_TEMPERATURE_C,
        "temperature_wrapper_parent_verified": True,
    }


def _visible_cuda_contract() -> dict[str, object]:
    """Require one visible CUDA device whose sole logical index is zero."""

    if not torch.cuda.is_available():
        raise RuntimeError("Formal800 requires available CUDA")
    visible_count = int(torch.cuda.device_count())
    if visible_count != 1:
        raise RuntimeError(
            "Formal800 requires exactly one visible CUDA device"
        )
    current = int(torch.cuda.current_device())
    if current != 0 or str(torch.device("cuda", current)) != FROZEN_DEVICE:
        raise RuntimeError("Formal800 fixes the current logical device to cuda:0")
    return {
        "cuda_available": True,
        "visible_cuda_device_count": visible_count,
        "current_cuda_logical_device": current,
    }


def _ensure_attempt_paths_absent(*, formal_output_may_exist: bool) -> None:
    if BOUNDED_OUTPUT_PATH.exists() or BOUNDED_OUTPUT_PATH.is_symlink():
        raise PermissionError(
            "v23 bounded-400 output must be absent; it has no authorization "
            "effect for Formal800"
        )
    formal_exists = OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink()
    if formal_exists and not formal_output_may_exist:
        raise FileExistsError(
            f"single-use Formal800 output already exists: {OUTPUT_PATH}"
        )
    if not formal_exists and formal_output_may_exist:
        raise RuntimeError("claimed Formal800 output disappeared")


def _load_dr_receipt(
    verification: Mapping[str, object],
) -> CoverageStatePACREDRGateReceipt:
    output_value = verification.get("output")
    if not isinstance(output_value, str):
        raise TypeError("terminal D_R verification has no output path")
    output = Path(output_value)
    wrapper = read_strict_json(output / "receipts/dr_gate.json")
    wrapper_fingerprint = verify_fingerprinted(
        wrapper,
        field="wrapper_fingerprint",
    )
    payload = wrapper.get("receipt")
    if not isinstance(payload, Mapping):
        raise TypeError("terminal D_R wrapper has no receipt")
    receipt = pacre_vc_dr_receipt_from_payload(payload)
    if (
        verification.get("run_id") != receipt.canonical_payload()["run_id"]
        or verification.get("decision") != PACRE_VC_DR_PASS_DECISION
        or verification.get("gate_passed") is not True
        or verification.get("failed_checks") != []
        or verification.get("receipt_fingerprint")
        != receipt.receipt_fingerprint
        or verification.get("wrapper_fingerprint")
        != wrapper_fingerprint
        or verification.get("formal_800_route_granted") is not True
        or verification.get("bounded_400_required") is not False
        or verification.get("bounded_400_authorization_effect") is not False
        or receipt.decision != PACRE_VC_DR_PASS_DECISION
        or not receipt.gate_passed
        or len(receipt.checks) != 13
        or tuple(name for name, _ in receipt.checks)
        != PACRE_VC_DR_CHECK_NAMES
        or not all(passed for _, passed in receipt.checks)
    ):
        raise PermissionError(
            "Formal800 requires the exact terminal v23 D_R 13/13 PASS"
        )
    return receipt


def _validate_static(
    *,
    formal_output_may_exist: bool,
) -> _StaticContext:
    """Validate metadata-only prerequisites without constructing real inputs."""

    stabilize_pacre_vc_numerical_runtime()
    _ensure_attempt_paths_absent(
        formal_output_may_exist=formal_output_may_exist
    )
    runtime = _runtime_contract()

    closure = read_strict_json(protocol_root() / SOURCE_CLOSURE_FILE)
    closure_fingerprint = verify_source_closure(closure)
    dataset_free = read_strict_json(
        protocol_root() / DATASET_FREE_FILE
    )
    dataset_fingerprint = verify_fingerprinted(dataset_free)
    verification = dict(verify_dr_terminal())
    receipt = _load_dr_receipt(verification)
    verification_fingerprint = stable_fingerprint(verification)
    if (
        not _is_sha256(closure_fingerprint)
        or not _is_sha256(dataset_fingerprint)
        or receipt.source_closure_fingerprint != closure_fingerprint
        or receipt.dataset_free_receipt_fingerprint
        != dataset_fingerprint
    ):
        raise PermissionError(
            "terminal D_R receipt differs from the live static prerequisites"
        )
    return _StaticContext(
        runtime=runtime,
        source_closure=closure,
        source_closure_fingerprint=closure_fingerprint,
        dataset_free=dataset_free,
        dataset_free_fingerprint=dataset_fingerprint,
        dr_verification=verification,
        dr_verification_fingerprint=verification_fingerprint,
        dr_receipt=receipt,
    )


def validate_create_only() -> dict[str, object]:
    """Validate the route without a claim, tensor load, or authorization."""

    context = _validate_static(formal_output_may_exist=False)
    return fingerprinted(
        {
            "schema_version": VALIDATION_SCHEMA,
            "run_id": RUN_ID,
            "mode": "validate_create_only",
            "static_contract_valid": True,
            "runtime": context.runtime,
            "source_closure_fingerprint": (
                context.source_closure_fingerprint
            ),
            "dataset_free_receipt_fingerprint": (
                context.dataset_free_fingerprint
            ),
            "D_R_terminal_verification_fingerprint": (
                context.dr_verification_fingerprint
            ),
            "D_R_gate_receipt_fingerprint": (
                context.dr_receipt.receipt_fingerprint
            ),
            "D_R_gate_check_count": len(context.dr_receipt.checks),
            "D_R_gate_passed": context.dr_receipt.gate_passed,
            "formal_output_absent": True,
            "bounded_400_output_absent": True,
            "bounded_400_required": False,
            "bounded_400_authorization_effect": False,
            "output_claimed": False,
            "real_inputs_constructed": False,
            "formal_authorization_created": False,
            "training_performed": False,
            "D_R_tensor_payload_accessed": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
            "not_a_formal_training_result": True,
        }
    )


def _attempt_payload(context: _StaticContext) -> dict[str, object]:
    return fingerprinted(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "run_id": RUN_ID,
            "output_repo_path": OUTPUT_REPO_PATH,
            "candidate": "PACRE-VC-v23",
            "objective": PACRE_PMOPE_OBJECTIVE,
            "runtime": context.runtime,
            "budget": {
                "seed": FROZEN_SEED,
                "epochs": FROZEN_EPOCHS,
                "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
                "updates": FROZEN_UPDATES,
                "from_scratch": True,
                "training_invocations": 1,
            },
            "source_closure_fingerprint": (
                context.source_closure_fingerprint
            ),
            "dataset_free_receipt_fingerprint": (
                context.dataset_free_fingerprint
            ),
            "D_R_terminal_verification_fingerprint": (
                context.dr_verification_fingerprint
            ),
            "D_R_gate_receipt_fingerprint": (
                context.dr_receipt.receipt_fingerprint
            ),
            "D_R_gate_check_count": 13,
            "D_R_gate_passed": True,
            "bounded_400_output_absent": True,
            "bounded_400_required": False,
            "bounded_400_authorization_effect": False,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "overwrite_allowed": False,
            "checkpoint_policy": "final_model_only",
            "D_R_receipt_metadata_read": True,
            "D_R_tensor_payload_accessed": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
        }
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _claim_output(attempt: Mapping[str, object]) -> Path:
    """Irreversibly claim the sole output before any full D_R construction."""

    OUTPUT_PATH.mkdir(parents=True, exist_ok=False)
    marker = OUTPUT_PATH / INCOMPLETE_FILE
    descriptor = os.open(
        marker,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    write_new_json(OUTPUT_PATH / "attempt.json", attempt)
    receipts = OUTPUT_PATH / "receipts"
    receipts.mkdir(exist_ok=False)
    _fsync_directory(receipts)
    _fsync_directory(OUTPUT_PATH)
    return receipts


def _artifact_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Formal800 output contains a symbolic link")
        if path.is_file():
            relative = str(path.relative_to(root))
            if relative not in {
                INCOMPLETE_FILE,
                "COMPLETE.json",
                "FAILURE.json",
            }:
                result[relative] = file_sha256(path)
        elif not path.is_dir():
            raise RuntimeError("Formal800 output contains a special file")
    return result


def _verify_population(*, terminal: bool) -> dict[str, str]:
    directories: set[str] = set()
    files: set[str] = set()
    for path in sorted(OUTPUT_PATH.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Formal800 output contains a symbolic link")
        relative = str(path.relative_to(OUTPUT_PATH))
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files.add(relative)
        else:
            raise RuntimeError("Formal800 output contains a special file")
    expected_files = set(EXPECTED_SCIENTIFIC_FILES)
    if terminal:
        expected_files.add("COMPLETE.json")
    else:
        expected_files.add(INCOMPLETE_FILE)
    if directories != set(EXPECTED_DIRECTORIES) or files != expected_files:
        raise RuntimeError("Formal800 output population differs")
    hashes = _artifact_hashes(OUTPUT_PATH)
    if set(hashes) != set(EXPECTED_SCIENTIFIC_FILES):
        raise RuntimeError("Formal800 scientific artifact inventory differs")
    return hashes


class _EpochProgressRecorder:
    """Durably append exactly one canonical row for every completed epoch."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: BinaryIO | None = path.open("xb")
        self.rows: list[dict[str, object]] = []

    def __call__(self, row: Mapping[str, object]) -> None:
        if self._handle is None:
            raise RuntimeError("Formal800 epoch recorder is closed")
        normalized = json.loads(
            strict_json_bytes(dict(row)).decode("utf-8")
        )
        expected_epoch = len(self.rows)
        if (
            normalized.get("epoch") != expected_epoch
            or normalized.get("objective") != PACRE_PMOPE_OBJECTIVE
            or normalized.get("completed_updates")
            != (expected_epoch + 1) * FROZEN_STEPS_PER_EPOCH
            or expected_epoch >= FROZEN_EPOCHS
        ):
            raise RuntimeError("Formal800 epoch progress row changed")
        event = fingerprinted(
            {
                "schema_version": EPOCH_PROGRESS_SCHEMA,
                "run_id": RUN_ID,
                "epoch_result": normalized,
            }
        )
        self._handle.write(strict_json_bytes(event))
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.rows.append(normalized)
        print(
            json.dumps(
                {
                    "event": "formal800_epoch_complete",
                    "run_id": RUN_ID,
                    "epoch": expected_epoch,
                    "completed_updates": normalized[
                        "completed_updates"
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )

    def close_and_verify(
        self,
        expected_rows: Sequence[Mapping[str, object]],
    ) -> None:
        if self._handle is None:
            raise RuntimeError("Formal800 epoch recorder closed twice")
        self._handle.close()
        self._handle = None
        normalized_expected = [
            json.loads(strict_json_bytes(dict(row)).decode("utf-8"))
            for row in expected_rows
        ]
        if (
            len(self.rows) != FROZEN_EPOCHS
            or self.rows != normalized_expected
            or self.rows[-1]["completed_updates"] != FROZEN_UPDATES
        ):
            raise RuntimeError(
                "Formal800 callback rows differ from the final epoch ledger"
            )

    def close_after_failure(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _validate_training_result(formal_result: object) -> None:
    training_result = formal_result.training_result
    if (
        formal_result.training_complete is not True
        or formal_result.training_invocations != 1
        or training_result.seed != FROZEN_SEED
        or training_result.epochs != FROZEN_EPOCHS
        or training_result.steps_per_epoch != FROZEN_STEPS_PER_EPOCH
        or training_result.completed_updates != FROZEN_UPDATES
        or training_result.forward_calls != FROZEN_UPDATES
        or training_result.backward_calls != FROZEN_UPDATES
        or training_result.optimizer_steps != FROZEN_UPDATES
        or training_result.logical_state_evaluations
        != 12 * FROZEN_UPDATES
        or training_result.finite_state_audits != FROZEN_UPDATES + 1
        or len(training_result.epoch_logs) != FROZEN_EPOCHS
        or training_result.objective != PACRE_PMOPE_OBJECTIVE
        or training_result.execution_device != FROZEN_DEVICE
    ):
        raise RuntimeError("Formal800 returned an incomplete compute ledger")


def _failure_payload(
    error: BaseException,
    *,
    attempt_fingerprint: str | None,
    stage: str,
    real_inputs_constructed: bool,
    training_invocations: int,
) -> dict[str, object]:
    try:
        artifacts = _artifact_hashes(OUTPUT_PATH)
    except BaseException:
        artifacts = {}
    return fingerprinted(
        {
            "schema_version": FAILURE_SCHEMA,
            "run_id": RUN_ID,
            "status": "failed_incomplete_single_attempt",
            "failed_stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "attempt_fingerprint": attempt_fingerprint,
            "artifact_files_before_failure": artifacts,
            "real_inputs_constructed": real_inputs_constructed,
            "training_invocations": training_invocations,
            "output_directory_reusable": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "bounded_400_authorization_effect": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
        }
    )


def run_once() -> dict[str, object]:
    """Consume and seal the sole fixed seed-42 Formal800 attempt."""

    context = _validate_static(formal_output_may_exist=False)
    attempt = _attempt_payload(context)
    attempt_fingerprint = str(attempt["receipt_fingerprint"])
    receipts = _claim_output(attempt)

    progress: _EpochProgressRecorder | None = None
    stage = "post_claim_D_R_terminal_reverification"
    real_inputs_constructed = False
    training_invocations = 0
    try:
        # A process that loses the exclusive mkdir never reaches this point.
        # The terminal gate and every static binding are rechecked after the
        # irreversible claim and before the first full D_R tensor is loaded.
        post_claim = _validate_static(formal_output_may_exist=True)
        if (
            post_claim.binding_fingerprint != context.binding_fingerprint
            or read_strict_json(OUTPUT_PATH / "attempt.json") != attempt
        ):
            raise RuntimeError(
                "Formal800 claim/static prerequisite changed after mkdir"
            )
        dr_terminal_receipt = fingerprinted(
            {
                "schema_version": DR_TERMINAL_SCHEMA,
                "run_id": RUN_ID,
                "attempt_fingerprint": attempt_fingerprint,
                "verification": post_claim.dr_verification,
                "verification_fingerprint": (
                    post_claim.dr_verification_fingerprint
                ),
                "D_R_gate_receipt_fingerprint": (
                    post_claim.dr_receipt.receipt_fingerprint
                ),
                "D_R_gate_check_count": len(
                    post_claim.dr_receipt.checks
                ),
                "D_R_gate_passed": True,
                "D_R_reopened": False,
                "D_R_tensor_payload_accessed": False,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
                "training_performed": False,
            }
        )
        write_new_json(
            receipts / "dr_terminal_verification.json",
            dr_terminal_receipt,
        )

        stage = "full_D_R_input_construction"
        source_paths = frozen_real_dr_source_paths()
        real_inputs = build_coverage_state_real_dr_inputs(**source_paths)
        real_inputs_constructed = True
        real_inputs.verify_unchanged()
        if (
            real_inputs.source_binding.split != "D_R"
            or real_inputs.scalar_cache.raw_catalog.split != "D_R"
            or real_inputs.scalar_cache.cache_fingerprint
            != PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
        ):
            raise PermissionError("Formal800 full D_R input changed")

        stage = "formal_authorization"
        model_config = expected_pacre_vc_formal_config(real_inputs)
        if model_config.expected_parameter_count != FROZEN_PARAMETER_COUNT:
            raise RuntimeError("Formal800 model parameter count changed")
        config = fingerprinted(
            {
                "schema_version": CONFIG_SCHEMA,
                "run_id": RUN_ID,
                "candidate": "PACRE-VC-v23",
                "split": "D_R",
                "runtime_splits": ["D_R"],
                "objective": PACRE_PMOPE_OBJECTIVE,
                "model_config": asdict(model_config),
                "parameter_count": model_config.expected_parameter_count,
                "field_threshold_hex": 0.0.hex(),
                "threshold_search_performed": False,
                "budget": {
                    "seed": FROZEN_SEED,
                    "epochs": FROZEN_EPOCHS,
                    "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
                    "updates": FROZEN_UPDATES,
                    "from_scratch": True,
                    "training_invocations": 1,
                },
                "runtime": post_claim.runtime,
                "checkpoint_policy": "final_model_only",
                "optimizer_state_saved": False,
                "intermediate_checkpoint_saved": False,
                "bounded_400_required": False,
                "bounded_400_authorization_effect": False,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
                "performance_evaluation_performed": False,
                "performance_claim_supported": False,
            }
        )
        write_new_json(receipts / "config.json", config)
        inputs = fingerprinted(
            {
                "schema_version": INPUTS_SCHEMA,
                "run_id": RUN_ID,
                "attempt_fingerprint": attempt_fingerprint,
                "real_inputs": real_inputs.canonical_payload(),
                "real_inputs_fingerprint": real_inputs.build_fingerprint,
                "full_D_R_scalar_cache_fingerprint": (
                    real_inputs.scalar_cache.cache_fingerprint
                ),
                "source_binding_fingerprint": (
                    real_inputs.source_binding.binding_fingerprint
                ),
                "source_files": {
                    name: {
                        "repo_path": str(path.relative_to(ROOT)),
                        "file_sha256": file_sha256(path),
                    }
                    for name, path in sorted(source_paths.items())
                },
                "construction_invocations": 1,
                "D_R_accessed": True,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "split_manifest_metadata_read": True,
                "D_R_tensor_payload_accessed": True,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
                "training_performed": False,
            }
        )
        write_new_json(receipts / "inputs.json", inputs)

        authorization = prepare_pacre_vc_formal_800_authorization(
            real_inputs,
            model_config,
            dataset_free_receipt=post_claim.dataset_free,
            dr_gate_receipt=post_claim.dr_receipt,
            source_closure=post_claim.source_closure,
            output_claim_fingerprint=attempt_fingerprint,
            run_id=RUN_ID,
        )
        authorization.verify_unchanged()
        if (
            not authorization.prerequisites_passed
            or not authorization.available
            or authorization.output_claim_fingerprint
            != attempt_fingerprint
        ):
            raise PermissionError(
                "Formal800 exact output-bound authorization was not issued"
            )
        authorization_receipt = fingerprinted(
            {
                "schema_version": AUTHORIZATION_SCHEMA,
                "run_id": RUN_ID,
                "attempt_fingerprint": attempt_fingerprint,
                "authorization": authorization.canonical_payload(),
                "authorization_fingerprint": (
                    authorization.authorization_fingerprint
                ),
                "formal_D_R_training_authorized": True,
                "bounded_400_receipt_consumed": False,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
                "performance_evaluation_performed": False,
            }
        )
        write_new_json(
            receipts / "authorization.json",
            authorization_receipt,
        )

        stage = "formal_800_training"
        progress = _EpochProgressRecorder(
            receipts / "epoch_progress.jsonl"
        )
        training_invocations = 1
        formal_result = run_pacre_vc_pmope_formal_800(
            authorization,
            model_config,
            device=FROZEN_DEVICE,
            epoch_callback=progress,
        )
        _validate_training_result(formal_result)
        progress.close_and_verify(
            formal_result.training_result.epoch_logs
        )
        formal_result.verify_unchanged()

        stage = "final_model_artifact"
        artifact = save_pacre_vc_formal_final_model(
            OUTPUT_PATH / FINAL_MODEL_DIRECTORY,
            formal_result=formal_result,
        )
        artifact_fingerprint = artifact.get("artifact_fingerprint")
        if not _is_sha256(artifact_fingerprint):
            raise RuntimeError("Formal800 final artifact is unbound")

        training = fingerprinted(
            {
                "schema_version": TRAINING_SCHEMA,
                "run_id": RUN_ID,
                "attempt_fingerprint": attempt_fingerprint,
                "authorization_fingerprint": (
                    authorization.authorization_fingerprint
                ),
                "formal_result": formal_result.canonical_payload(),
                "formal_result_fingerprint": (
                    formal_result.result_fingerprint
                ),
                "training_result_fingerprint": (
                    formal_result.training_result.result_fingerprint
                ),
                "final_model_artifact_fingerprint": (
                    artifact_fingerprint
                ),
                "compute_ledger": {
                    "seed": formal_result.training_result.seed,
                    "epochs": formal_result.training_result.epochs,
                    "steps_per_epoch": (
                        formal_result.training_result.steps_per_epoch
                    ),
                    "completed_updates": (
                        formal_result.training_result.completed_updates
                    ),
                    "forward_calls": (
                        formal_result.training_result.forward_calls
                    ),
                    "backward_calls": (
                        formal_result.training_result.backward_calls
                    ),
                    "optimizer_steps": (
                        formal_result.training_result.optimizer_steps
                    ),
                    "logical_state_evaluations": (
                        formal_result.training_result
                        .logical_state_evaluations
                    ),
                    "finite_state_audits": (
                        formal_result.training_result.finite_state_audits
                    ),
                    "epoch_progress_rows": len(progress.rows),
                    "training_invocations": training_invocations,
                },
                "final_checkpoint_only": True,
                "optimizer_state_saved": False,
                "intermediate_checkpoint_saved": False,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
                "performance_evaluation_performed": False,
                "performance_claim_supported": False,
            }
        )
        write_new_json(receipts / "training.json", training)

        stage = "decision"
        decision = fingerprinted(
            {
                "schema_version": DECISION_SCHEMA,
                "run_id": RUN_ID,
                "status": (
                    "FORMAL800_TRAINING_COMPLETE_"
                    "D_V_PREREGISTRATION_ELIGIBLE"
                ),
                "attempt_fingerprint": attempt_fingerprint,
                "authorization_fingerprint": (
                    authorization.authorization_fingerprint
                ),
                "formal_result_fingerprint": (
                    formal_result.result_fingerprint
                ),
                "final_model_artifact_fingerprint": (
                    artifact_fingerprint
                ),
                "formal_training_complete": True,
                "compute_ledger_complete": True,
                "bounded_400_required": False,
                "bounded_400_authorization_effect": False,
                "D_V_preregistration_eligible": True,
                "D_V_execution_authorized": False,
                "D_T_execution_authorized": False,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
                "performance_evaluation_performed": False,
                "performance_gate_passed": None,
                "performance_claim_supported": False,
                "final_model_performance_success_established": False,
                "next_action": (
                    "validate_and_run_fixed_adaptive_D_V_under_"
                    "existing_preregistration"
                ),
            }
        )
        write_new_json(receipts / "decision.json", decision)

        stage = "terminal_seal"
        _fsync_directory(receipts)
        _fsync_directory(OUTPUT_PATH / FINAL_MODEL_DIRECTORY)
        artifact_files = _verify_population(terminal=False)
        complete = fingerprinted(
            {
                "schema_version": COMPLETE_SCHEMA,
                "run_id": RUN_ID,
                "status": (
                    "FORMAL800_TRAINING_COMPLETE_"
                    "D_V_PREREGISTRATION_ELIGIBLE"
                ),
                "attempt_fingerprint": attempt_fingerprint,
                "authorization_fingerprint": (
                    authorization.authorization_fingerprint
                ),
                "formal_result_fingerprint": (
                    formal_result.result_fingerprint
                ),
                "final_model_artifact_fingerprint": (
                    artifact_fingerprint
                ),
                "decision_fingerprint": (
                    decision["receipt_fingerprint"]
                ),
                "artifact_files": artifact_files,
                "artifact_count": len(artifact_files),
                "seed": FROZEN_SEED,
                "epochs": FROZEN_EPOCHS,
                "steps_per_epoch": FROZEN_STEPS_PER_EPOCH,
                "updates": FROZEN_UPDATES,
                "training_invocations": training_invocations,
                "final_checkpoint_only": True,
                "optimizer_state_saved": False,
                "intermediate_checkpoint_saved": False,
                "bounded_400_required": False,
                "bounded_400_authorization_effect": False,
                "D_V_preregistration_eligible": True,
                "D_V_execution_authorized": False,
                "D_T_execution_authorized": False,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
                "performance_evaluation_performed": False,
                "performance_claim_supported": False,
            },
            field="complete_fingerprint",
        )
        write_new_json(OUTPUT_PATH / "COMPLETE.json", complete)
        (OUTPUT_PATH / INCOMPLETE_FILE).unlink()
        _fsync_directory(OUTPUT_PATH)
        final_hashes = _verify_population(terminal=True)
        if final_hashes != artifact_files:
            raise RuntimeError(
                "Formal800 artifacts changed while writing terminal seal"
            )
        return complete
    except BaseException as error:
        if progress is not None:
            progress.close_after_failure()
        if (
            OUTPUT_PATH.is_dir()
            and not (OUTPUT_PATH / "FAILURE.json").exists()
            and not (OUTPUT_PATH / "COMPLETE.json").exists()
        ):
            failure = _failure_payload(
                error,
                attempt_fingerprint=attempt_fingerprint,
                stage=stage,
                real_inputs_constructed=real_inputs_constructed,
                training_invocations=training_invocations,
            )
            write_new_json(OUTPUT_PATH / "FAILURE.json", failure)
            _fsync_directory(OUTPUT_PATH)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-create-only", action="store_true")
    modes.add_argument("--run-once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = (
        validate_create_only()
        if args.validate_create_only
        else run_once()
    )
    print(
        json.dumps(
            result,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
