#!/usr/bin/env python3
"""Append-only closure for CURE-Lite v24 runtime compatibility generation c2.

This module is deliberately not a scientific runner.  It can only authorize
and seal the metadata-only c2 compatibility lane after the expired c1 lane has
been terminalized.  It never imports torch, opens a dataset, starts a unit, or
creates a runtime specification.

The c2 lane retains scientific attempt ordinal 2 and the original scientific
output paths.  All c1, predecessor, and scientific runtime paths remain
protected.  A compatibility receipt is archival: its short authorization may
expire after sealing, but the receipt chronology must remain inside the
original authorization window.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import ModuleType
from typing import Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()
RUNS_ROOT = (REPOSITORY / "runs/irstd1k_stage_a_seed42").resolve()

CANDIDATE = "GCR-PACRE-v24"
STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
SCIENTIFIC_ATTEMPT_ID = "gcr_pacre_v24_D_R_zero_update_structural_r2"
SCIENTIFIC_ATTEMPT_ORDINAL = 2
RUNTIME_COMPATIBILITY_ID = "c2"
INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后运行"

OLD_UNIT_NAME = "cure-lite-v24-gcr-pacre-dr-r2.service"
C1_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service"
)
C2_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service"
)

C1_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_authorization.json"
)
C1_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_receipt.json"
)
C1_FAILURE_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_schema_compat_c1_expired_prewrite_terminal.json"
)
C2_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c2_authorization.json"
)
C2_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c2_receipt.json"
)

R10_ROOT = (
    EVIDENCE_ROOT
    / "supervisor_v2_systemd_integration_preaccess_compat_c1_r10"
)
R10_AUTHORIZATION_PATH = R10_ROOT / "control/authorization.json"
R10_RECEIPT_PATH = R10_ROOT / "control/integration-receipt.json"
R10_TERMINAL_PATH = R10_ROOT / "control/integration-terminal.json"
R10_REMOVAL_STATE_PATH = R10_ROOT / "control/removal-state.json"

C2_ENVIRONMENT_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c2.json"
)
C2_ENVIRONMENT_STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c2.json"
)
C2_ENVIRONMENT_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c2.json"
)
C2_UNIT_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c2_unit_realization_authorization.json"
)
C2_UNIT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c2_unit_realization_receipt.json"
)
C2_UNIT_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c2_unit_realization_terminal.json"
)

C2_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_spec.json"
)
C2_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c2_"
        "runtime_launch_authorization.json"
    )
)
C2_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_artifacts"
)
C2_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_gpu_lease"
)
C2_RUN_ROOT_ALIAS_PATH = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c2"
)
C2_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_receipt.json"
)

C1_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json"
)
C1_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c1_"
        "runtime_launch_authorization.json"
    )
)
C1_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_artifacts"
)
C1_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_gpu_lease"
)
C1_RUN_ROOT_ALIAS_PATH = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c1"
)
C1_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_receipt.json"
)

OLD_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_spec.json"
)
OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_runtime_launch_authorization.json"
)
OLD_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_artifacts"
)
OLD_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_gpu_lease"
)
SCIENTIFIC_RUN_ROOT = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2"
)
SCIENTIFIC_RESULT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_receipt.json"
)

C2_BRIDGE_SOURCE_PATH = Path(__file__).resolve()
C1_FAILURE_TERMINALIZER_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_compat_c1_expired_prewrite_terminal.py"
).resolve()
C2_ENVIRONMENT_WRAPPER_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_environment_preaccess_compat_c2.py"
).resolve()
C2_RELEASE_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_runtime_release_preaccess_compat_c2.py"
).resolve()
C2_SUPERVISOR_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c2.py"
).resolve()
C2_ADAPTER_SOURCE_PATH = (
    REPOSITORY
    / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c2.py"
).resolve()
C2_UNIT_REALIZER_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c2.py"
).resolve()
C2_UNIT_TEMPLATE_PATH = (
    REPOSITORY
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service.template"
).resolve()
C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_compat_c2_prewrite_failure_terminal.py"
).resolve()
C2_PREWRITE_FAILURE_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_schema_compat_c2_prewrite_failure_terminal.json"
)

C1_FAILURE_TERMINALIZER_SHA256 = (
    "72d7f8846d9bdccbdb2d15d6790d5e021b6d2db75523fe5dbe11a3d4246ca880"
)
C2_PREWRITE_FAILURE_TERMINALIZER_SHA256 = (
    "17ef3a0420c4b3d978f23270bde490805997e21dcb21f395ce7e5ac06659dc5f"
)
C2_PREWRITE_FAILURE_TERMINAL_SHA256 = (
    "6984dc9df2c905a5b7bc3b1577a4d5e8c21d1e1f895217997ed6915050e0f43d"
)
C2_PREWRITE_FAILURE_TERMINAL_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c2-"
    "prewrite-failure-terminal-v1"
)

AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c2-authorization-v1"
)
RECEIPT_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c2-receipt-v1"
)
C1_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c1-authorization-v1"
)
R10_TERMINAL_SCHEMA = (
    "cure-lite-v24-supervisor-v2-systemd-integration-terminal-v1"
)
ENVIRONMENT_POLICY_SCHEMA = "cure-lite-v24-runtime-environment-policy-v1"
ENVIRONMENT_STABILITY_SCHEMA = (
    "cure-lite-v24-runtime-environment-stability-receipt-v1"
)
ENVIRONMENT_RECEIPT_SCHEMA = (
    "cure-lite-v24-runtime-environment-audit-receipt-v1"
)
UNIT_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-actual-unit-realization-authorization-v1"
)
UNIT_RECEIPT_SCHEMA = (
    "cure-lite-v24-actual-unit-realization-receipt-v1"
)
RUNTIME_SPEC_SCHEMA = "cure-lite-v24-dr-runtime-supervisor-spec-v2"
R2_RESULT_SCHEMA = (
    "cure-lite-v24-gcr-pacre-real-dr-structural-gate-r2-v1"
)
R2_RUN_START_SCHEMA = (
    "cure-lite-v24-D_R-persistent-run-start-r2-v1"
)
R2_RUN_START_STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
R2_RUN_START_PATH_POLICY = (
    "fixed_repository_run_root_authorization_fingerprint_filename_v1"
)
R2_EXECUTION_KIND = "real_D_R"
R2_EXECUTION_SEED = 42
R2_RUN_START_FILENAME_PREFIX = (
    "gcr_pacre_v24_D_R_structural_run_start_"
)

RUNTIME_PHASE_PREACTIVATION = "preactivation"
RUNTIME_PHASE_COMMIT = "commit"
RUNTIME_PHASE_CLAIM = "claim"
RUNTIME_PHASE_VERIFY = "verify"
RUNTIME_PHASE_RUN_ONCE = "run_once"
RUNTIME_PHASE_FINALIZE_SUCCESS = "finalize_success"
RUNTIME_PHASE_FINALIZE_FAILURE = "finalize_failure"
RUNTIME_PHASES = frozenset(
    {
        RUNTIME_PHASE_PREACTIVATION,
        RUNTIME_PHASE_COMMIT,
        RUNTIME_PHASE_CLAIM,
        RUNTIME_PHASE_VERIFY,
        RUNTIME_PHASE_RUN_ONCE,
        RUNTIME_PHASE_FINALIZE_SUCCESS,
        RUNTIME_PHASE_FINALIZE_FAILURE,
    }
)
_EMPTY_RUN_ROOT_PHASES = frozenset(
    {
        RUNTIME_PHASE_COMMIT,
        RUNTIME_PHASE_CLAIM,
        RUNTIME_PHASE_VERIFY,
        RUNTIME_PHASE_RUN_ONCE,
    }
)

_R2_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "candidate",
        "execution_kind",
        "execution_seed",
        "device",
        "requested_receipt_output",
        "dataset_free_receipt_fingerprint",
        "dataset_free_receipt_file_sha256",
        "efficiency_section_fingerprint",
        "efficiency_receipt_sha256",
        "preaccess_authorization_fingerprint",
        "preaccess_authorization_file_sha256",
        "access_audit_receipt_fingerprint",
        "access_audit_receipt_file_sha256",
        "protocol_preregistration_fingerprint",
        "implementation_binding",
        "source_closure_fingerprint",
        "source_binding_fingerprint",
        "real_inputs_fingerprint",
        "population_fingerprint",
        "cache_fingerprint",
        "adapter_fingerprint",
        "run_start_marker",
        "artifact_hashes",
        "raw_observations",
        "raw_observations_fingerprint",
        "checks",
        "decision",
        "boundary",
        "receipt_fingerprint",
    }
)
_R2_RUN_START_ENVELOPE_KEYS = frozenset(
    {"path", "file_sha256", "marker_fingerprint", "payload"}
)
_R2_RUN_START_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "path_policy",
        "stage_id",
        "run_id",
        "candidate",
        "marker_path",
        "authorization_fingerprint",
        "authorization_receipt_file_sha256",
        "access_audit_receipt_fingerprint",
        "access_audit_receipt_file_sha256",
        "dataset_free_receipt_fingerprint",
        "dataset_free_receipt_file_sha256",
        "protocol_preregistration_fingerprint",
        "source_closure_fingerprint",
        "implementation_binding",
        "expected_source_binding_fingerprint",
        "expected_real_inputs_fingerprint",
        "expected_population_fingerprint",
        "expected_cache_fingerprint",
        "intent",
        "intent_fingerprint",
        "marker_fingerprint",
    }
)
_R2_RUN_INTENT_KEYS = frozenset(
    {
        "execution_kind",
        "split",
        "requested_device",
        "requested_receipt_output",
        "D_R_materialization_intended",
        "D_V_materialization_intended",
        "D_T_materialization_intended",
        "optimizer_steps_authorized",
        "parameter_updates_authorized",
        "training_authorized",
    }
)
_R2_BOUNDARY_KEYS = frozenset(
    {
        "execution_kind",
        "split",
        "D_R_accessed",
        "D_V_accessed",
        "D_T_accessed",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "optimizer_module_referenced",
        "optimizer_constructed",
        "optimizer_steps",
        "parameter_updates",
        "training_performed",
        "performance_gate_present",
        "performance_claim_supported",
        "threshold_or_ratio_gate",
    }
)
_R2_ARTIFACT_HASH_KEYS = frozenset(
    {
        "dataset_free_receipt",
        "preaccess_authorization",
        "preaccess_access_audit",
        "persistent_run_start_marker",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_UTC_SUFFIX = "Z"
_PAYLOAD_FLAGS = (
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
)
_STATE_FIELDS = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "NRestarts",
    "FragmentPath",
    "InvocationID",
    "DropInPaths",
    "NeedDaemonReload",
    "Transient",
    "Restart",
    "ExecMainPID",
    "ExecMainCode",
    "ExecMainStatus",
    "Result",
    "StateChangeTimestamp",
    "ActiveEnterTimestamp",
    "InactiveEnterTimestamp",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
)
_BRIDGE_STATE_FIELDS = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "NRestarts",
    "FragmentPath",
    "InvocationID",
)
_SCOPE_FIELDS = (
    "target_unit_id",
    "conflict_unit_ids",
    "dependency_unit_ids",
    "allowed_failed_unit_ids",
    "allowed_unit_ids",
    "allowed_manager_states",
    "require_target_ready",
    "strict_all_gpu_consumers",
)
_SOURCE_LABELS = frozenset(
    {
        "compat_bridge",
        "c1_failure_terminalizer",
        "compat_environment_wrapper",
        "compat_release",
        "compat_supervisor",
        "compat_adapter",
        "compat_unit_realizer",
        "compat_unit_template",
        "c2_prewrite_failure_terminalizer",
    }
)
_EVIDENCE_LABELS = frozenset(
    {
        "c1_failure_terminal",
        "c2_prewrite_failure_terminal",
        "r10_authorization",
        "r10_receipt",
        "environment_policy",
        "environment_stability",
        "environment_postcleanup",
        "unit_realization_authorization",
        "unit_realization_receipt",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version",
        "candidate",
        "stage_id",
        "scientific_attempt_id",
        "scientific_attempt_ordinal",
        "runtime_compatibility_id",
        "instruction_id",
        "authorization_basis",
        "authorized_uid",
        "created_at_utc",
        "issued_at_utc",
        "expires_at_utc",
        "c1_failure_terminal_root",
        "c2_prewrite_failure_terminal_root",
        "c1_expired_authorization_root",
        "r10_roots",
        "compatibility_source_roots",
        "protected_unit_states",
        "expected_evidence_paths",
        "scientific_output_contract",
        "scientific_authority",
        "mutation_authority",
        "D_R_payload_accessed",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "gpu_accessed",
        "training_started",
        "materialization_consumed",
        "authorization_fingerprint",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "candidate",
        "stage_id",
        "scientific_attempt_id",
        "scientific_attempt_ordinal",
        "runtime_compatibility_id",
        "created_at_utc",
        "compatibility_authorization_root",
        "compatibility_source_roots",
        "compatibility_evidence_roots",
        "historical_environment_contract",
        "current_environment_contract",
        "scientific_output_contract",
        "scientific_authority",
        "compatibility_closure_passed",
        "runtime_launch_authorized",
        "systemd_start_authorized",
        "automatic_retry",
        "resume",
        "D_R_payload_accessed",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "gpu_accessed",
        "training_started",
        "materialization_consumed",
        "receipt_fingerprint",
    }
)

UnitStateReader = Callable[[str], Mapping[str, object]]

# Compatibility-facing names intentionally mirror the c1 bridge interface.
# Downstream c2 realizer/supervisor/release modules must not need a permissive
# fallback or generation-specific import shim.
COMPAT_AUTHORIZATION_PATH = C2_AUTHORIZATION_PATH
COMPAT_RECEIPT_PATH = C2_RECEIPT_PATH
COMPATIBILITY_RECEIPT_PATH = C2_RECEIPT_PATH
COMPAT_UNIT_REALIZER_SOURCE_PATH = C2_UNIT_REALIZER_SOURCE_PATH
COMPAT_UNIT_NAME = C2_UNIT_NAME


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(dict(value)).encode()).hexdigest()


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", _UTC_SUFFIX)


def _parse_utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith(_UTC_SUFFIX):
        raise PermissionError(f"{name} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise PermissionError(f"{name} is malformed") from error
    if parsed.tzinfo is None or _format_utc(parsed) != value:
        raise PermissionError(f"{name} is not canonical UTC")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_fixed_path(
    supplied: str | Path,
    expected: Path,
    *,
    name: str,
) -> Path:
    path = Path(supplied).absolute()
    if path != Path(expected).absolute():
        raise PermissionError(f"{name} path changed")
    return path


def _safe_parent(path: Path) -> os.stat_result:
    parent = path.parent
    before = parent.lstat()
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or parent.resolve(strict=True) != parent
        or before.st_uid != os.getuid()
    ):
        raise PermissionError(f"unsafe evidence parent: {parent}")
    return before


def _read_regular_bytes(
    path: Path,
    *,
    sealed: bool = True,
) -> tuple[bytes, os.stat_result]:
    target = Path(path).absolute()
    parent_before = _safe_parent(target)
    flags = os.O_RDONLY
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required")
    flags |= os.O_NOFOLLOW
    directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fd = os.open(target.name, flags, dir_fd=directory_fd)
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
                or (sealed and before.st_mode & 0o022)
            ):
                raise PermissionError(f"unsafe sealed file: {target}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
        finally:
            os.close(fd)
    finally:
        os.close(directory_fd)
    parent_after = target.parent.lstat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or (
        parent_before.st_dev,
        parent_before.st_ino,
        parent_before.st_uid,
    ) != (
        parent_after.st_dev,
        parent_after.st_ino,
        parent_after.st_uid,
    ):
        raise PermissionError(f"sealed file changed while reading: {target}")
    return b"".join(chunks), before


def _source_root(path: Path) -> dict[str, object]:
    # Repository sources are owner-owned and may use the workspace's existing
    # 0664 mode.  Their exact bytes and file generation are frozen below; the
    # stricter non-writable rule remains mandatory for sealed evidence.
    raw, observed = _read_regular_bytes(path, sealed=False)
    target = Path(path).absolute()
    return {
        "path": str(target),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "owner_uid": observed.st_uid,
        "size": observed.st_size,
    }


def _validate_source_root(
    root: object,
    *,
    expected_path: Path,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    if not isinstance(root, Mapping):
        raise PermissionError("source root is malformed")
    observed = _source_root(expected_path)
    if dict(root) != observed:
        raise PermissionError(f"source generation changed: {expected_path}")
    if (
        expected_sha256 is not None
        and observed["file_sha256"] != expected_sha256
    ):
        raise PermissionError(f"frozen source hash changed: {expected_path}")
    return observed


def _write_sealed(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    target = Path(path).absolute()
    _safe_parent(target)
    if fingerprint_field in body:
        raise ValueError("fingerprint field must not be pre-populated")
    payload = dict(body)
    payload[fingerprint_field] = stable_fingerprint(payload)
    raw = (_canonical_json(payload) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required")
    flags |= os.O_NOFOLLOW
    directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fd = os.open(target.name, flags, 0o400, dir_fd=directory_fd)
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(fd, raw[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    loaded, _root = _load_sealed(
        target,
        fingerprint_field=fingerprint_field,
    )
    if loaded != payload:
        raise RuntimeError("sealed write verification failed")
    return payload


def _load_sealed(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    target = Path(path).absolute()
    raw, observed = _read_regular_bytes(target)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"sealed JSON is malformed: {target}") from error
    if (
        not isinstance(value, dict)
        or not isinstance(value.get(fingerprint_field), str)
        or _SHA256.fullmatch(value[fingerprint_field]) is None
    ):
        raise PermissionError(f"sealed fingerprint is absent: {target}")
    body = dict(value)
    fingerprint = body.pop(fingerprint_field)
    if fingerprint != stable_fingerprint(body):
        raise PermissionError(f"sealed fingerprint changed: {target}")
    if schema is not None and value.get("schema_version") != schema:
        raise PermissionError(f"sealed schema changed: {target}")
    if raw != (_canonical_json(value) + "\n").encode():
        raise PermissionError(f"sealed JSON is not canonical: {target}")
    root = {
        "path": str(target),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "fingerprint_field": fingerprint_field,
        "fingerprint": fingerprint,
        "schema_version": value.get("schema_version"),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "owner_uid": observed.st_uid,
        "size": observed.st_size,
    }
    return value, root


def _validate_sealed_root(
    root: object,
    *,
    expected_path: Path,
    fingerprint_field: str,
    schema: str | None = None,
) -> dict[str, object]:
    if not isinstance(root, Mapping):
        raise PermissionError("sealed root is malformed")
    _payload, observed = _load_sealed(
        expected_path,
        fingerprint_field=fingerprint_field,
        schema=schema,
    )
    if dict(root) != observed:
        raise PermissionError(f"sealed evidence changed: {expected_path}")
    return observed


def _require_no_payload(value: Mapping[str, object]) -> None:
    if any(value.get(field) is not False for field in _PAYLOAD_FLAGS):
        raise PermissionError("compatibility evidence accessed payload")
    if value.get("gpu_accessed", False) is not False:
        raise PermissionError("compatibility evidence accessed a GPU")
    if value.get("training_started", False) is not False:
        raise PermissionError("compatibility evidence started training")
    if value.get("materialization_consumed", False) is not False:
        raise PermissionError("scientific materialization was consumed")


def _expected_scientific_output_contract() -> dict[str, object]:
    return {
        "run_root": str(SCIENTIFIC_RUN_ROOT),
        "result_receipt": str(SCIENTIFIC_RESULT_RECEIPT_PATH),
        "compat_run_root_alias": str(C2_RUN_ROOT_ALIAS_PATH),
        "compat_result_receipt_alias": str(
            C2_RESULT_RECEIPT_ALIAS_PATH
        ),
        "original_r2_paths_retained": True,
        "compatibility_aliases_forbidden": True,
    }


def _expected_scientific_authority() -> dict[str, object]:
    return {
        "D_R_payload_authorized": False,
        "D_V_payload_authorized": False,
        "D_T_payload_authorized": False,
        "training_authorized": False,
        "materialization_authorized": False,
        "automatic_retry": False,
        "resume": False,
        "fresh_scientific_attempt": False,
    }


def _expected_mutation_authority() -> dict[str, object]:
    return {
        "compatibility_receipt_creation_authorized": True,
        "environment_scope_handoff_authorized": True,
        "environment_metadata_audit_authorized": True,
        "c2_unit_realization_authorized": True,
        "runtime_spec_creation_authorized": False,
        "runtime_launch_authorization_creation_authorized": False,
        "unit_start_authorized": False,
        "unit_enable_authorized": False,
        "payload_access_authorized": False,
    }


def _default_unit_state_reader(unit_name: str) -> Mapping[str, object]:
    command = [
        "/usr/bin/systemctl",
        "--user",
        "show",
        unit_name,
        *[f"--property={field}" for field in _STATE_FIELDS],
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    state: dict[str, object] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in _STATE_FIELDS:
            state[key] = value
    if set(state) != set(_STATE_FIELDS):
        raise PermissionError(f"unit state is incomplete: {unit_name}")
    return state


def _normalized_state(
    reader: UnitStateReader,
    unit_name: str,
) -> dict[str, object]:
    state = dict(reader(unit_name))
    if not set(_BRIDGE_STATE_FIELDS).issubset(state):
        raise PermissionError(f"unit state fields changed: {unit_name}")
    result = {field: state[field] for field in _BRIDGE_STATE_FIELDS}
    if result["Id"] != unit_name:
        raise PermissionError(f"unit identity changed: {unit_name}")
    return result


def _require_inert_state(
    state: Mapping[str, object],
    *,
    unit_name: str,
    fragment_path: str | None = None,
) -> None:
    if (
        state.get("Id") != unit_name
        or state.get("LoadState") != "loaded"
        or state.get("ActiveState") != "inactive"
        or state.get("SubState") != "dead"
        or state.get("UnitFileState") != "static"
        or state.get("NRestarts") != "0"
        or state.get("InvocationID") not in ("", None)
        or (
            fragment_path is not None
            and state.get("FragmentPath") != fragment_path
        )
    ):
        raise PermissionError(f"unit is not exact static/inert: {unit_name}")


def _source_paths() -> dict[str, Path]:
    return {
        "compat_bridge": C2_BRIDGE_SOURCE_PATH,
        "c1_failure_terminalizer": (
            C1_FAILURE_TERMINALIZER_SOURCE_PATH
        ),
        "compat_environment_wrapper": (
            C2_ENVIRONMENT_WRAPPER_SOURCE_PATH
        ),
        "compat_release": C2_RELEASE_SOURCE_PATH,
        "compat_supervisor": C2_SUPERVISOR_SOURCE_PATH,
        "compat_adapter": C2_ADAPTER_SOURCE_PATH,
        "compat_unit_realizer": C2_UNIT_REALIZER_SOURCE_PATH,
        "compat_unit_template": C2_UNIT_TEMPLATE_PATH,
        "c2_prewrite_failure_terminalizer": (
            C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH
        ),
    }


def _evidence_paths() -> dict[str, Path]:
    return {
        "c1_failure_terminal": C1_FAILURE_TERMINAL_PATH,
        "c2_prewrite_failure_terminal": (
            C2_PREWRITE_FAILURE_TERMINAL_PATH
        ),
        "r10_authorization": R10_AUTHORIZATION_PATH,
        "r10_receipt": R10_RECEIPT_PATH,
        "environment_policy": C2_ENVIRONMENT_POLICY_PATH,
        "environment_stability": C2_ENVIRONMENT_STABILITY_PATH,
        "environment_postcleanup": C2_ENVIRONMENT_POSTCLEANUP_PATH,
        "unit_realization_authorization": C2_UNIT_AUTHORIZATION_PATH,
        "unit_realization_receipt": C2_UNIT_RECEIPT_PATH,
    }


def _collect_source_roots() -> dict[str, dict[str, object]]:
    roots = {
        label: _source_root(path)
        for label, path in _source_paths().items()
    }
    if set(roots) != _SOURCE_LABELS:
        raise AssertionError("c2 source labels changed")
    if (
        roots["c1_failure_terminalizer"]["file_sha256"]
        != C1_FAILURE_TERMINALIZER_SHA256
        or roots["c2_prewrite_failure_terminalizer"]["file_sha256"]
        != C2_PREWRITE_FAILURE_TERMINALIZER_SHA256
    ):
        raise PermissionError("compatibility terminalizer source hash changed")
    return roots


def _validate_source_roots(roots: object) -> None:
    if not isinstance(roots, Mapping) or set(roots) != _SOURCE_LABELS:
        raise PermissionError("c2 source-root labels changed")
    for label, path in _source_paths().items():
        expected = None
        if label == "c1_failure_terminalizer":
            expected = C1_FAILURE_TERMINALIZER_SHA256
        elif label == "c2_prewrite_failure_terminalizer":
            expected = C2_PREWRITE_FAILURE_TERMINALIZER_SHA256
        _validate_source_root(
            roots[label],
            expected_path=path,
            expected_sha256=expected,
        )


def _always_absent_paths() -> dict[str, Path]:
    return {
        "c1_compatibility_receipt": C1_RECEIPT_PATH,
        "c1_runtime_spec": C1_RUNTIME_SPEC_PATH,
        "c1_runtime_launch_authorization": (
            C1_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "c1_runtime_artifacts": C1_RUNTIME_ARTIFACT_ROOT,
        "c1_gpu_lease": C1_GPU_LEASE_ROOT,
        "c1_run_alias": C1_RUN_ROOT_ALIAS_PATH,
        "c1_result_alias": C1_RESULT_RECEIPT_ALIAS_PATH,
        "old_runtime_spec": OLD_RUNTIME_SPEC_PATH,
        "old_runtime_launch_authorization": (
            OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "old_runtime_artifacts": OLD_RUNTIME_ARTIFACT_ROOT,
        "old_gpu_lease": OLD_GPU_LEASE_ROOT,
        "c2_run_alias": C2_RUN_ROOT_ALIAS_PATH,
        "c2_result_alias": C2_RESULT_RECEIPT_ALIAS_PATH,
        "c2_unit_terminal": C2_UNIT_TERMINAL_PATH,
    }


def _preactivation_scientific_paths() -> dict[str, Path]:
    return {
        "scientific_run_root": SCIENTIFIC_RUN_ROOT,
        "scientific_result_receipt": SCIENTIFIC_RESULT_RECEIPT_PATH,
    }


def _c2_future_paths() -> dict[str, Path]:
    return {
        "c2_runtime_spec": C2_RUNTIME_SPEC_PATH,
        "c2_runtime_launch_authorization": (
            C2_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "c2_runtime_artifacts": C2_RUNTIME_ARTIFACT_ROOT,
        "c2_gpu_lease": C2_GPU_LEASE_ROOT,
    }


def _require_absent(paths: Mapping[str, Path]) -> None:
    present = [
        f"{label}:{path}"
        for label, path in paths.items()
        if os.path.lexists(path)
    ]
    if present:
        raise PermissionError(
            "protected compatibility/scientific path exists: "
            + ",".join(present)
        )


def _resolve_runtime_phase(
    *,
    allow_runtime_activation: bool,
    runtime_phase: str | None,
) -> str:
    """Resolve one explicit phase while retaining only safe legacy behavior."""

    if type(allow_runtime_activation) is not bool:
        raise TypeError("allow_runtime_activation must be boolean")
    if runtime_phase is None:
        if allow_runtime_activation:
            raise PermissionError(
                "active runtime verification requires an explicit phase"
            )
        return RUNTIME_PHASE_PREACTIVATION
    if not isinstance(runtime_phase, str) or runtime_phase not in RUNTIME_PHASES:
        raise PermissionError("runtime phase is not an exact closed state")
    expected_activation = runtime_phase != RUNTIME_PHASE_PREACTIVATION
    if allow_runtime_activation is not expected_activation:
        raise PermissionError(
            "runtime activation flag and explicit phase disagree"
        )
    return runtime_phase


def _validate_runtime_scientific_run_root(
    *,
    require_empty: bool,
) -> None:
    """Require the exact private original-r2 run directory generation."""

    if type(require_empty) is not bool:
        raise TypeError("require_empty must be boolean")
    target = SCIENTIFIC_RUN_ROOT.absolute()
    if not os.path.lexists(target):
        raise PermissionError("scientific r2 run root is absent")
    parent_before = _safe_parent(target)
    try:
        before = target.lstat()
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or target.resolve(strict=True) != target
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o700
            or before.st_nlink < 2
        ):
            raise PermissionError(
                "scientific r2 run root is not exact private canonical"
            )
        if not hasattr(os, "O_NOFOLLOW"):
            raise RuntimeError("O_NOFOLLOW is required")
        descriptor = os.open(
            target,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            opened = os.fstat(descriptor)
            entries = tuple(os.listdir(descriptor)) if require_empty else ()
            after = os.fstat(descriptor)
            linked = target.lstat()
        finally:
            os.close(descriptor)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise PermissionError(
            "scientific r2 run root changed while validating"
        ) from error
    parent_after = target.parent.lstat()
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(observed, field)
        for field in identity_fields
        for observed in (opened, after, linked)
    ) or (
        parent_before.st_dev,
        parent_before.st_ino,
        parent_before.st_uid,
    ) != (
        parent_after.st_dev,
        parent_after.st_ino,
        parent_after.st_uid,
    ):
        raise PermissionError(
            "scientific r2 run root generation changed while validating"
        )
    if entries:
        raise PermissionError(
            "scientific r2 run root is not empty in pre-execution phase"
        )


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PermissionError(f"{name} is not exact SHA-256")
    return value


def _validate_r2_run_start_binding(
    receipt: Mapping[str, object],
) -> None:
    run_start = receipt.get("run_start_marker")
    marker = (
        run_start.get("payload")
        if isinstance(run_start, Mapping)
        else None
    )
    intent = marker.get("intent") if isinstance(marker, Mapping) else None
    if (
        not isinstance(run_start, Mapping)
        or set(run_start) != _R2_RUN_START_ENVELOPE_KEYS
        or not isinstance(marker, Mapping)
        or set(marker) != _R2_RUN_START_MARKER_KEYS
        or not isinstance(intent, Mapping)
        or set(intent) != _R2_RUN_INTENT_KEYS
    ):
        raise PermissionError("scientific r2 run-start structure changed")
    authorization_fingerprint = _require_sha256(
        receipt.get("preaccess_authorization_fingerprint"),
        name="r2 preaccess authorization fingerprint",
    )
    marker_fingerprint = _require_sha256(
        marker.get("marker_fingerprint"),
        name="r2 run-start marker fingerprint",
    )
    marker_body = dict(marker)
    marker_body.pop("marker_fingerprint")
    expected_marker_path = SCIENTIFIC_RUN_ROOT.absolute() / (
        R2_RUN_START_FILENAME_PREFIX
        + authorization_fingerprint
        + ".json"
    )
    expected_result_path = str(
        SCIENTIFIC_RESULT_RECEIPT_PATH.absolute()
    )
    implementation = receipt.get("implementation_binding")
    if (
        marker.get("schema_version") != R2_RUN_START_SCHEMA
        or marker.get("path_policy") != R2_RUN_START_PATH_POLICY
        or marker.get("stage_id") != R2_RUN_START_STAGE_ID
        or marker.get("run_id") != SCIENTIFIC_ATTEMPT_ID
        or marker.get("candidate") != CANDIDATE
        or marker.get("marker_path") != str(expected_marker_path)
        or marker.get("authorization_fingerprint")
        != authorization_fingerprint
        or marker.get("authorization_receipt_file_sha256")
        != receipt.get("preaccess_authorization_file_sha256")
        or marker.get("access_audit_receipt_fingerprint")
        != receipt.get("access_audit_receipt_fingerprint")
        or marker.get("access_audit_receipt_file_sha256")
        != receipt.get("access_audit_receipt_file_sha256")
        or marker.get("dataset_free_receipt_fingerprint")
        != receipt.get("dataset_free_receipt_fingerprint")
        or marker.get("dataset_free_receipt_file_sha256")
        != receipt.get("dataset_free_receipt_file_sha256")
        or marker.get("protocol_preregistration_fingerprint")
        != receipt.get("protocol_preregistration_fingerprint")
        or marker.get("source_closure_fingerprint")
        != receipt.get("source_closure_fingerprint")
        or marker.get("implementation_binding") != implementation
        or marker.get("expected_source_binding_fingerprint")
        != receipt.get("source_binding_fingerprint")
        or marker.get("expected_real_inputs_fingerprint")
        != receipt.get("real_inputs_fingerprint")
        or marker.get("expected_population_fingerprint")
        != receipt.get("population_fingerprint")
        or marker.get("expected_cache_fingerprint")
        != receipt.get("cache_fingerprint")
        or marker_fingerprint != stable_fingerprint(marker_body)
        or run_start.get("path") != str(expected_marker_path)
        or run_start.get("marker_fingerprint") != marker_fingerprint
        or intent.get("execution_kind") != R2_EXECUTION_KIND
        or intent.get("split") != "D_R"
        or intent.get("requested_device") != receipt.get("device")
        or intent.get("requested_receipt_output")
        != expected_result_path
        or intent.get("D_R_materialization_intended") is not True
        or intent.get("D_V_materialization_intended") is not False
        or intent.get("D_T_materialization_intended") is not False
        or type(intent.get("optimizer_steps_authorized")) is not int
        or intent.get("optimizer_steps_authorized") != 0
        or type(intent.get("parameter_updates_authorized")) is not int
        or intent.get("parameter_updates_authorized") != 0
        or intent.get("training_authorized") is not False
        or marker.get("intent_fingerprint")
        != stable_fingerprint(dict(intent))
    ):
        raise PermissionError("scientific r2 run-start binding changed")
    stored_marker, marker_root = _load_sealed(
        expected_marker_path,
        fingerprint_field="marker_fingerprint",
        schema=R2_RUN_START_SCHEMA,
    )
    if (
        stored_marker != dict(marker)
        or marker_root.get("path") != str(expected_marker_path)
        or marker_root.get("mode") != 0o444
        or marker_root.get("owner_uid") != os.getuid()
        or expected_marker_path.resolve(strict=True)
        != expected_marker_path
        or run_start.get("file_sha256")
        != marker_root.get("file_sha256")
    ):
        raise PermissionError(
            "scientific r2 persistent run-start marker changed"
        )


def _validate_r2_result_payload(
    receipt: Mapping[str, object],
) -> None:
    if set(receipt) != _R2_RESULT_KEYS:
        raise PermissionError("scientific r2 result keys changed")
    expected_result_path = str(
        SCIENTIFIC_RESULT_RECEIPT_PATH.absolute()
    )
    implementation = receipt.get("implementation_binding")
    raw = receipt.get("raw_observations")
    boundary = receipt.get("boundary")
    artifacts = receipt.get("artifact_hashes")
    if (
        receipt.get("schema_version") != R2_RESULT_SCHEMA
        or receipt.get("run_id") != SCIENTIFIC_ATTEMPT_ID
        or receipt.get("candidate") != CANDIDATE
        or receipt.get("execution_kind") != R2_EXECUTION_KIND
        or type(receipt.get("execution_seed")) is not int
        or receipt.get("execution_seed") != R2_EXECUTION_SEED
        or not isinstance(receipt.get("device"), str)
        or not receipt.get("device")
        or receipt.get("requested_receipt_output")
        != expected_result_path
        or not isinstance(implementation, Mapping)
        or receipt.get("source_closure_fingerprint")
        != stable_fingerprint(dict(implementation))
        or not isinstance(raw, Mapping)
        or receipt.get("raw_observations_fingerprint")
        != stable_fingerprint(dict(raw))
        or not isinstance(receipt.get("checks"), Mapping)
        or not isinstance(receipt.get("decision"), Mapping)
        or not isinstance(boundary, Mapping)
        or set(boundary) != _R2_BOUNDARY_KEYS
        or boundary.get("execution_kind") != R2_EXECUTION_KIND
        or boundary.get("split") != "D_R"
        or boundary.get("D_R_accessed") is not True
        or boundary.get("D_V_accessed") is not False
        or boundary.get("D_T_accessed") is not False
        or boundary.get("D_V_tensor_payload_accessed") is not False
        or boundary.get("D_T_tensor_payload_accessed") is not False
        or boundary.get("optimizer_module_referenced") is not False
        or boundary.get("optimizer_constructed") is not False
        or type(boundary.get("optimizer_steps")) is not int
        or boundary.get("optimizer_steps") != 0
        or type(boundary.get("parameter_updates")) is not int
        or boundary.get("parameter_updates") != 0
        or boundary.get("training_performed") is not False
        or boundary.get("performance_gate_present") is not False
        or boundary.get("performance_claim_supported") is not False
        or boundary.get("threshold_or_ratio_gate") is not None
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != _R2_ARTIFACT_HASH_KEYS
    ):
        raise PermissionError("scientific r2 result structure changed")
    sha_fields = (
        "dataset_free_receipt_fingerprint",
        "dataset_free_receipt_file_sha256",
        "efficiency_section_fingerprint",
        "efficiency_receipt_sha256",
        "preaccess_authorization_fingerprint",
        "preaccess_authorization_file_sha256",
        "access_audit_receipt_fingerprint",
        "access_audit_receipt_file_sha256",
        "protocol_preregistration_fingerprint",
        "source_closure_fingerprint",
        "source_binding_fingerprint",
        "real_inputs_fingerprint",
        "population_fingerprint",
        "cache_fingerprint",
        "adapter_fingerprint",
        "raw_observations_fingerprint",
        "receipt_fingerprint",
    )
    for field in sha_fields:
        _require_sha256(receipt.get(field), name=f"r2 result {field}")
    _validate_r2_run_start_binding(receipt)
    run_start = receipt["run_start_marker"]
    if (
        artifacts.get("dataset_free_receipt")
        != receipt.get("dataset_free_receipt_file_sha256")
        or artifacts.get("preaccess_authorization")
        != receipt.get("preaccess_authorization_file_sha256")
        or artifacts.get("preaccess_access_audit")
        != receipt.get("access_audit_receipt_file_sha256")
        or artifacts.get("persistent_run_start_marker")
        != run_start.get("file_sha256")
    ):
        raise PermissionError("scientific r2 artifact binding changed")


def _validate_runtime_scientific_result_receipt(
    *,
    required: bool,
) -> None:
    """Validate the exact sealed real-r2 receipt without scientific imports."""

    if type(required) is not bool:
        raise TypeError("required must be boolean")

    target = SCIENTIFIC_RESULT_RECEIPT_PATH.absolute()
    if not os.path.lexists(target):
        if required:
            raise PermissionError("scientific r2 result receipt is absent")
        return
    try:
        payload, root = _load_sealed(
            target,
            fingerprint_field="receipt_fingerprint",
            schema=R2_RESULT_SCHEMA,
        )
        if (
            root.get("path") != str(target)
            or root.get("mode") != 0o444
            or root.get("owner_uid") != os.getuid()
            or target.resolve(strict=True) != target
        ):
            raise PermissionError(
                "scientific r2 result receipt is not exactly sealed"
            )
        _validate_r2_result_payload(payload)
    except PermissionError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise PermissionError(
            "scientific r2 result receipt is not exact canonical sealed"
        ) from error


def _validate_scientific_output_phase(
    *,
    allow_runtime_activation: bool,
    runtime_phase: str | None = None,
) -> None:
    """Enforce the exact original-r2 output state for one runtime phase."""

    phase = _resolve_runtime_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    _require_absent(_always_absent_paths())
    if phase == RUNTIME_PHASE_PREACTIVATION:
        _require_absent(_preactivation_scientific_paths())
        return
    _validate_runtime_scientific_run_root(
        require_empty=phase in _EMPTY_RUN_ROOT_PHASES,
    )
    if phase in _EMPTY_RUN_ROOT_PHASES:
        _require_absent(
            {"scientific_result_receipt": SCIENTIFIC_RESULT_RECEIPT_PATH}
        )
        return
    _validate_runtime_scientific_result_receipt(
        required=phase == RUNTIME_PHASE_FINALIZE_SUCCESS,
    )


def _validate_common_identity(value: Mapping[str, object]) -> None:
    if (
        value.get("candidate") != CANDIDATE
        or value.get("stage_id") != STAGE_ID
        or value.get("scientific_attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or value.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
    ):
        raise PermissionError("c2 scientific identity changed")


def _load_verified_terminalizer() -> tuple[ModuleType, dict[str, object]]:
    raw, _observed = _read_regular_bytes(
        C1_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    if hashlib.sha256(raw).hexdigest() != C1_FAILURE_TERMINALIZER_SHA256:
        raise PermissionError("c1 terminalizer source changed")
    name = "tools._cure_lite_v24_c1_expired_terminal_verified_for_c2"
    module = ModuleType(name)
    module.__file__ = str(C1_FAILURE_TERMINALIZER_SOURCE_PATH)
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(C1_FAILURE_TERMINALIZER_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    root = _source_root(C1_FAILURE_TERMINALIZER_SOURCE_PATH)
    if (
        root["file_sha256"] != C1_FAILURE_TERMINALIZER_SHA256
        or Path(module.TERMINAL_PATH).absolute()
        != C1_FAILURE_TERMINAL_PATH.absolute()
        or module.CANDIDATE != CANDIDATE
        or module.STAGE_ID != STAGE_ID
        or module.SCIENTIFIC_ATTEMPT_ID != SCIENTIFIC_ATTEMPT_ID
        or module.SCIENTIFIC_ATTEMPT_ORDINAL
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or module.RUNTIME_COMPATIBILITY_ID != "c1"
        or module.UNIT_NAME != C1_UNIT_NAME
        or not isinstance(module.SCHEMA, str)
        or not module.SCHEMA
        or not callable(module.validate_terminal)
    ):
        raise PermissionError("c1 terminalizer interface changed")
    return module, root


def _load_verified_environment_wrapper(
    expected_root: Mapping[str, object],
) -> tuple[ModuleType, dict[str, object]]:
    """Load exactly the wrapper generation captured by c2 authorization.

    The wrapper is intentionally not hash-pinned in this bridge: doing so
    would form a source-hash cycle once the wrapper pins the c2 realizer.  The
    create-once authorization captures the wrapper's full source root, and
    every later phase requires that same file generation before and after
    execution.
    """

    if not isinstance(expected_root, Mapping):
        raise PermissionError("c2 environment wrapper root is malformed")
    expected = dict(expected_root)
    _validate_source_root(
        expected,
        expected_path=C2_ENVIRONMENT_WRAPPER_SOURCE_PATH,
    )
    raw, _observed = _read_regular_bytes(
        C2_ENVIRONMENT_WRAPPER_SOURCE_PATH,
        sealed=False,
    )
    if hashlib.sha256(raw).hexdigest() != expected.get("file_sha256"):
        raise PermissionError("c2 environment wrapper bytes changed")
    name = "tools._cure_lite_v24_environment_compat_c2_verified_for_bridge"
    module = ModuleType(name)
    module.__file__ = str(C2_ENVIRONMENT_WRAPPER_SOURCE_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                raw,
                str(C2_ENVIRONMENT_WRAPPER_SOURCE_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
        after = _source_root(C2_ENVIRONMENT_WRAPPER_SOURCE_PATH)
        if after != expected:
            raise PermissionError(
                "c2 environment wrapper generation changed while loading"
            )
        if (
            Path(module.C2_POLICY_PATH).absolute()
            != C2_ENVIRONMENT_POLICY_PATH.absolute()
            or Path(module.C2_STABILITY_PATH).absolute()
            != C2_ENVIRONMENT_STABILITY_PATH.absolute()
            or Path(module.C2_POSTCLEANUP_PATH).absolute()
            != C2_ENVIRONMENT_POSTCLEANUP_PATH.absolute()
            or Path(module.C2_REALIZATION_AUTHORIZATION_PATH).absolute()
            != C2_UNIT_AUTHORIZATION_PATH.absolute()
            or Path(module.C2_REALIZATION_RECEIPT_PATH).absolute()
            != C2_UNIT_RECEIPT_PATH.absolute()
            or module.C2_TARGET_UNIT != C2_UNIT_NAME
            or not callable(module.replay_old_scope_and_handoff)
            or not callable(module.validate_c2_environment_closure)
            or not callable(module._production_archival_validator)
        ):
            raise PermissionError("c2 environment wrapper interface changed")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, after


def _normalize_environment_contract(contract: object) -> dict[str, object]:
    if not is_dataclass(contract) or isinstance(contract, type):
        raise PermissionError("c2 environment contract is not a dataclass")
    value = asdict(contract)
    if not isinstance(value, dict):
        raise PermissionError("c2 environment contract is malformed")
    return json.loads(_canonical_json(value))


def _load_verified_prewrite_failure_terminalizer(
) -> tuple[ModuleType, dict[str, object]]:
    raw, _observed = _read_regular_bytes(
        C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    if (
        hashlib.sha256(raw).hexdigest()
        != C2_PREWRITE_FAILURE_TERMINALIZER_SHA256
    ):
        raise PermissionError("c2 prewrite terminalizer source changed")

    name = "tools._cure_lite_v24_c2_prewrite_terminal_verified_for_bridge"
    module = ModuleType(name)
    module.__file__ = str(C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH)
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    root = _source_root(C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH)
    if (
        root.get("file_sha256")
        != C2_PREWRITE_FAILURE_TERMINALIZER_SHA256
        or Path(module.TERMINAL_PATH).absolute()
        != C2_PREWRITE_FAILURE_TERMINAL_PATH.absolute()
        or module.SCHEMA != C2_PREWRITE_FAILURE_TERMINAL_SCHEMA
        or not callable(module.validate_archival)
    ):
        raise PermissionError("c2 prewrite terminalizer interface changed")
    return module, root


def _validate_prewrite_failure_terminal(
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    # T and F are independently byte-pinned before archival code can run.
    module, source_root = _load_verified_prewrite_failure_terminalizer()
    terminalizer_raw_before, terminalizer_before = _read_regular_bytes(
        C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    if (
        hashlib.sha256(terminalizer_raw_before).hexdigest()
        != C2_PREWRITE_FAILURE_TERMINALIZER_SHA256
        or _source_root(C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH)
        != source_root
    ):
        raise PermissionError("c2 prewrite terminalizer source changed")
    raw_before, before = _read_regular_bytes(
        C2_PREWRITE_FAILURE_TERMINAL_PATH,
        sealed=True,
    )
    if (
        hashlib.sha256(raw_before).hexdigest()
        != C2_PREWRITE_FAILURE_TERMINAL_SHA256
        or stat.S_IMODE(before.st_mode) != 0o444
    ):
        raise PermissionError("c2 prewrite failure terminal changed")

    payload, root = module.validate_archival(
        C2_PREWRITE_FAILURE_TERMINAL_PATH
    )
    terminalizer_raw_after, terminalizer_after = _read_regular_bytes(
        C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    raw_after, after = _read_regular_bytes(
        C2_PREWRITE_FAILURE_TERMINAL_PATH,
        sealed=True,
    )
    generation_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        terminalizer_raw_after != terminalizer_raw_before
        or any(
            getattr(terminalizer_before, field)
            != getattr(terminalizer_after, field)
            for field in generation_fields
        )
        or hashlib.sha256(terminalizer_raw_after).hexdigest()
        != C2_PREWRITE_FAILURE_TERMINALIZER_SHA256
        or _source_root(C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH)
        != source_root
        or raw_after != raw_before
        or any(
            getattr(before, field) != getattr(after, field)
            for field in generation_fields
        )
        or hashlib.sha256(raw_after).hexdigest()
        != C2_PREWRITE_FAILURE_TERMINAL_SHA256
        or not isinstance(payload, Mapping)
        or not isinstance(root, Mapping)
    ):
        raise PermissionError("c2 prewrite failure terminal changed")

    expected_root = {
        "path": str(C2_PREWRITE_FAILURE_TERMINAL_PATH.absolute()),
        "file_sha256": C2_PREWRITE_FAILURE_TERMINAL_SHA256,
        "device": after.st_dev,
        "inode": after.st_ino,
        "owner_uid": after.st_uid,
        "mode": stat.S_IMODE(after.st_mode),
        "nlink": after.st_nlink,
        "size": after.st_size,
        "terminal_fingerprint": payload.get("terminal_fingerprint"),
        "schema_version": C2_PREWRITE_FAILURE_TERMINAL_SCHEMA,
    }
    try:
        decoded = json.loads(raw_before)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(
            "c2 prewrite failure terminal is malformed"
        ) from error
    if dict(root) != expected_root or decoded != dict(payload):
        raise PermissionError("c2 prewrite failure terminal root diverged")

    identity = payload.get("identity")
    reproduction = payload.get("deterministic_reproduction")
    absence = payload.get("c2_absence_snapshot")
    continuation = payload.get("continuation_policy")
    if (
        not isinstance(identity, Mapping)
        or identity.get("candidate") != CANDIDATE
        or identity.get("stage_id") != STAGE_ID
        or identity.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or identity.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or identity.get("runtime_compatibility_id") != "c2"
        or not isinstance(reproduction, Mapping)
        or reproduction.get("validator")
        != "_validate_c1_failure_terminal"
        or reproduction.get("observation_kind")
        != "post_hoc_deterministic_reproduction_before_source_revision"
        or reproduction.get("original_call_id_claimed") is not False
        or reproduction.get("original_failure_time_claimed") is not False
        or reproduction.get("write_capable_entrypoint_invoked") is not False
        or reproduction.get("exception_type") != "PermissionError"
        or reproduction.get("exception_message")
        != "expired-prewrite terminal live closure changed"
        or reproduction.get("reproduced") is not True
        or not isinstance(absence, Mapping)
        or absence.get("all_required_paths_absent") is not True
        or absence.get("scientific_attempt_consumed") is not False
        or not isinstance(continuation, Mapping)
        or continuation.get("automatic_retry") is not False
        or continuation.get("same_source_reentry") is not False
        or continuation.get("revised_source_manual_reentry") is not True
        or continuation.get("fixed_r12_required") is not True
        or continuation.get("c3_required") is not False
        or continuation.get("scientific_attempt_consumed") is not False
        or continuation.get("runtime_launch_consumed") is not False
        or continuation.get("materialization_consumed") is not False
    ):
        raise PermissionError("c2 prewrite failure transition changed")

    # Do not consume F's payload/GPU/training claims as historical authority.
    return dict(payload), expected_root, source_root


def _validate_c1_historical_absence_snapshot(
    module: ModuleType,
    value: object,
) -> None:
    if not isinstance(value, Mapping):
        raise PermissionError("c1 historical absences are malformed")

    paths = module.ABSENCE_PATHS
    row_keys = {
        "path",
        "basename",
        "lexists",
        "parent_path",
        "parent_device",
        "parent_inode",
        "parent_owner_uid",
        "parent_owner_gid",
        "parent_mode",
        "parent_nlink",
        "parent_size",
        "parent_mtime_ns",
        "parent_ctime_ns",
    }
    if not isinstance(paths, Mapping) or set(value) != set(paths):
        raise PermissionError("c1 historical absence keys changed")

    parent_fields = tuple(
        sorted(key for key in row_keys if key.startswith("parent_"))
    )
    integer_fields = set(parent_fields) - {"parent_path"}
    parents: dict[str, tuple[object, ...]] = {}
    for name, expected_path in paths.items():
        row = value[name]
        target = Path(expected_path).absolute()
        if (
            not isinstance(row, Mapping)
            or set(row) != row_keys
            or row.get("path") != str(target)
            or row.get("basename") != target.name
            or row.get("lexists") is not False
            or row.get("parent_path") != str(target.parent)
            or any(
                isinstance(row.get(field), bool)
                or not isinstance(row.get(field), int)
                or row[field] < 0
                for field in integer_fields
            )
            or row["parent_inode"] <= 0
            or row["parent_nlink"] < 2
            or row["parent_owner_uid"] != os.getuid()
            or row["parent_mode"] > 0o7777
            or row["parent_mode"] & 0o002
        ):
            raise PermissionError(f"c1 historical absence changed: {name}")

        parent_identity = tuple(row[field] for field in parent_fields)
        previous = parents.setdefault(str(target.parent), parent_identity)
        if previous != parent_identity:
            raise PermissionError("c1 historical absence parent diverged")


def _validate_c1_historical_terminal(
    module: ModuleType,
    terminal: Mapping[str, object],
    *,
    c1_reader: Callable[[], Mapping[str, str]],
    now: datetime,
) -> None:
    identity = {
        "schema_version": module.SCHEMA,
        "candidate": module.CANDIDATE,
        "stage_id": module.STAGE_ID,
        "scientific_attempt_id": module.SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": module.SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": module.RUNTIME_COMPATIBILITY_ID,
        "unit_name": module.UNIT_NAME,
        "instruction_id": module.INSTRUCTION_ID,
        "authorization_basis": module.AUTHORIZATION_BASIS,
    }
    if (
        set(terminal) != set(module._BODY_KEYS) | {"terminal_fingerprint"}
        or any(
            terminal.get(key) != expected
            for key, expected in identity.items()
        )
    ):
        raise PermissionError("c1 historical terminal identity changed")
    if (
        terminal.get("payload_observation") != module._PAYLOAD_OBSERVATION
        or terminal.get("continuation_policy")
        != module._CONTINUATION_POLICY
        or terminal.get("outcome") != module._OUTCOME
        or terminal.get("derived_runtime_absences")
        != module._derived_runtime_absences()
    ):
        raise PermissionError("c1 historical terminal contract changed")

    current = now.astimezone(timezone.utc)
    created = module._parse_utc(
        terminal.get("created_at_utc"),
        name="terminal created_at_utc",
    )
    if created > current:
        raise PermissionError("c1 historical terminal is future-dated")

    session = module._observe_session_failure()
    evidence_roots, payloads = module._observe_evidence()
    source_roots = module._observe_source_roots()
    fragment_root = module._observe_fragment_root()
    expiry = module._validate_evidence_semantics(
        payloads,
        now=current,
    )
    current_live = module._validate_live_state(
        c1_reader(),
        unit_receipt=payloads["unit_receipt"],
    )
    receipt_fragment = payloads["unit_receipt"].get("fragment_identity")
    fragment_fields = (
        "path",
        "file_sha256",
        "device",
        "inode",
        "owner_uid",
        "mode",
        "nlink",
    )
    if (
        not isinstance(receipt_fragment, Mapping)
        or terminal.get("session_failure") != session
        or terminal.get("evidence_roots") != evidence_roots
        or terminal.get("source_roots") != source_roots
        or terminal.get("fragment_root") != fragment_root
        or terminal.get("live_unit_state") != current_live
        or terminal.get("authorization_expiry") != expiry
        or any(
            receipt_fragment.get(key) != fragment_root.get(key)
            for key in fragment_fields
        )
    ):
        raise PermissionError("c1 historical immutable snapshot changed")

    _validate_c1_historical_absence_snapshot(
        module,
        terminal.get("absence_generation_roots"),
    )
    for name, path in module.ABSENCE_PATHS.items():
        if name not in {
            "scientific_run_root",
            "scientific_result_receipt",
        }:
            module._observe_absence(path)

    bridge_expiry = module._parse_utc(
        expiry["bridge_expires_at_utc"],
        name="terminal bridge expiry",
    )
    if created <= bridge_expiry:
        raise PermissionError("c1 historical terminal predates bridge expiry")


def _validate_c1_failure_terminal(
    *,
    unit_state_reader: UnitStateReader,
    now: datetime,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    module, source_root = _load_verified_terminalizer()

    def c1_reader() -> Mapping[str, str]:
        observed = dict(unit_state_reader(C1_UNIT_NAME))
        live_keys = set(module._LIVE_KEYS)
        if not live_keys.issubset(observed):
            raise PermissionError("c1 terminal live-state fields are absent")
        return {key: str(observed[key]) for key in live_keys}

    allowed_errors = {
        "expired-prewrite terminal live closure changed",
        "required absent path exists: "
        + str(Path(module.ABSENCE_PATHS["scientific_run_root"]).absolute()),
        "required absent path exists: "
        + str(
            Path(
                module.ABSENCE_PATHS["scientific_result_receipt"]
            ).absolute()
        ),
    }
    try:
        terminal = module.validate_terminal(
            terminal_path=C1_FAILURE_TERMINAL_PATH,
            unit_state_reader=c1_reader,
            now=lambda: now,
        )
    except PermissionError as error:
        if (
            type(error) is not PermissionError
            or len(error.args) != 1
            or error.args[0] not in allowed_errors
        ):
            raise
        terminal, root = module._load_sealed(
            C1_FAILURE_TERMINAL_PATH,
            fingerprint_field="terminal_fingerprint",
            schema=module.SCHEMA,
        )
        _validate_c1_historical_terminal(
            module,
            terminal,
            c1_reader=c1_reader,
            now=now,
        )
    else:
        sealed, root = module._load_sealed(
            C1_FAILURE_TERMINAL_PATH,
            fingerprint_field="terminal_fingerprint",
            schema=module.SCHEMA,
        )
        if terminal != sealed:
            raise PermissionError(
                "c1 terminalizer returned a different terminal"
            )
    continuation = terminal["continuation_policy"]
    outcome = terminal["outcome"]
    payload = terminal["payload_observation"]
    evidence = terminal["evidence_roots"]
    expiry = terminal["authorization_expiry"]
    if (
        continuation.get("same_c1_reauthorization_allowed") is not False
        or continuation.get("same_c1_receipt_sealing_allowed") is not False
        or continuation.get("automatic_retry_allowed") is not False
        or continuation.get("resume_allowed") is not False
        or continuation.get("new_compatibility_generation_required")
        is not True
        or outcome.get("scientific_attempt_consumed") is not False
        or outcome.get("runtime_launch_consumed") is not False
        or outcome.get("materialization_consumed") is not False
        or any(payload.get(field) is not False for field in _PAYLOAD_FLAGS)
        or payload.get("gpu_accessed") is not False
        or payload.get("training_started") is not False
        or not isinstance(evidence, Mapping)
        or "r10_authorization" not in evidence
        or "r10_receipt" not in evidence
        or not isinstance(expiry, Mapping)
    ):
        raise PermissionError("c1 terminal continuation semantics changed")
    return terminal, dict(root), source_root


def _r10_roots_from_terminal(
    terminal: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    evidence = terminal.get("evidence_roots")
    if not isinstance(evidence, Mapping):
        raise PermissionError("c1 terminal evidence roots are absent")
    roots = {
        "authorization": dict(evidence["r10_authorization"]),
        "receipt": dict(evidence["r10_receipt"]),
    }
    return roots


def _authorization_times(
    authorization: Mapping[str, object],
    *,
    current: datetime,
    require_fresh: bool,
) -> tuple[datetime, datetime, datetime]:
    created = _parse_utc(
        authorization.get("created_at_utc"),
        name="c2 authorization creation",
    )
    issued = _parse_utc(
        authorization.get("issued_at_utc"),
        name="c2 authorization issuance",
    )
    expires = _parse_utc(
        authorization.get("expires_at_utc"),
        name="c2 authorization expiry",
    )
    if (
        not issued <= created <= expires
        or expires - issued > timedelta(seconds=300)
        or (require_fresh and not issued <= current <= expires)
    ):
        raise PermissionError("c2 authorization is stale or malformed")
    return created, issued, expires


def authorize_c2(
    *,
    instruction_id: str,
    authorization_basis: str,
    validity_seconds: int = 300,
    authorization_path: Path | None = None,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    if authorization_path is None:
        authorization_path = C2_AUTHORIZATION_PATH
    _require_fixed_path(
        authorization_path,
        C2_AUTHORIZATION_PATH,
        name="c2 authorization",
    )
    if (
        instruction_id != INSTRUCTION_ID
        or authorization_basis != AUTHORIZATION_BASIS
        or isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("c2 authorization input changed")
    if os.path.lexists(C2_AUTHORIZATION_PATH) or os.path.lexists(
        C2_RECEIPT_PATH
    ):
        raise FileExistsError("c2 compatibility identity is consumed")
    _validate_scientific_output_phase(
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
    )
    _require_absent(_c2_future_paths())
    issued = now().astimezone(timezone.utc)
    _failure, failure_root, failure_source_root = (
        _validate_prewrite_failure_terminal()
    )
    terminal, terminal_root, _terminalizer_root = (
        _validate_c1_failure_terminal(
            unit_state_reader=unit_state_reader,
            now=issued,
        )
    )
    terminal_evidence = terminal["evidence_roots"]
    c1_root = dict(terminal_evidence["bridge_authorization"])
    r10_roots = _r10_roots_from_terminal(terminal)
    sources = _collect_source_roots()
    if (
        sources["c2_prewrite_failure_terminalizer"]
        != failure_source_root
    ):
        raise PermissionError(
            "c2 prewrite terminalizer root diverged"
        )
    _load_verified_environment_wrapper(
        sources["compat_environment_wrapper"]
    )
    old_state = _normalized_state(unit_state_reader, OLD_UNIT_NAME)
    c1_state = _normalized_state(unit_state_reader, C1_UNIT_NAME)
    _require_inert_state(old_state, unit_name=OLD_UNIT_NAME)
    _require_inert_state(c1_state, unit_name=C1_UNIT_NAME)
    c1_expires = _parse_utc(
        terminal["authorization_expiry"]["bridge_expires_at_utc"],
        name="c1 authorization expiry",
    )
    if issued <= c1_expires:
        raise PermissionError("c2 cannot precede c1 authorization expiry")
    body: dict[str, object] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
        "instruction_id": instruction_id,
        "authorization_basis": authorization_basis,
        "authorized_uid": os.getuid(),
        "created_at_utc": _format_utc(issued),
        "issued_at_utc": _format_utc(issued),
        "expires_at_utc": _format_utc(
            issued + timedelta(seconds=validity_seconds)
        ),
        "c1_failure_terminal_root": terminal_root,
        "c2_prewrite_failure_terminal_root": failure_root,
        "c1_expired_authorization_root": c1_root,
        "r10_roots": r10_roots,
        "compatibility_source_roots": sources,
        "protected_unit_states": {
            "old": old_state,
            "c1": c1_state,
        },
        "expected_evidence_paths": {
            label: str(path)
            for label, path in _evidence_paths().items()
            if label not in {
                "c1_failure_terminal",
                "r10_authorization",
                "r10_receipt",
            }
        },
        "scientific_output_contract": (
            _expected_scientific_output_contract()
        ),
        "scientific_authority": _expected_scientific_authority(),
        "mutation_authority": _expected_mutation_authority(),
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
        "training_started": False,
        "materialization_consumed": False,
    }
    return _write_sealed(
        C2_AUTHORIZATION_PATH,
        body,
        fingerprint_field="authorization_fingerprint",
    )


def validate_c2_authorization(
    path: Path | None = None,
    *,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    require_fresh: bool = True,
    allow_runtime_activation: bool = False,
    runtime_phase: str | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> tuple[dict[str, object], dict[str, object]]:
    phase = _resolve_runtime_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    if path is None:
        path = C2_AUTHORIZATION_PATH
    fixed = _require_fixed_path(
        path,
        C2_AUTHORIZATION_PATH,
        name="c2 authorization",
    )
    authorization, root = _load_sealed(
        fixed,
        fingerprint_field="authorization_fingerprint",
        schema=AUTHORIZATION_SCHEMA,
    )
    _validate_common_identity(authorization)
    _require_no_payload(authorization)
    if set(authorization) != _AUTHORIZATION_KEYS:
        raise PermissionError("c2 authorization keys changed")
    _validate_scientific_output_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=phase,
    )
    _authorization_times(
        authorization,
        current=now().astimezone(timezone.utc),
        require_fresh=require_fresh,
    )
    current = now().astimezone(timezone.utc)
    _failure, failure_root, failure_source_root = (
        _validate_prewrite_failure_terminal()
    )
    terminal, terminal_root, _terminalizer_root = (
        _validate_c1_failure_terminal(
            unit_state_reader=unit_state_reader,
            now=current,
        )
    )
    c1_root = dict(terminal["evidence_roots"]["bridge_authorization"])
    if authorization.get("r10_roots") != _r10_roots_from_terminal(
        terminal
    ):
        raise PermissionError("c2 authorization r10 lineage changed")
    _validate_source_roots(
        authorization.get("compatibility_source_roots")
    )
    old_state = _normalized_state(unit_state_reader, OLD_UNIT_NAME)
    c1_state = _normalized_state(unit_state_reader, C1_UNIT_NAME)
    if (
        authorization.get("runtime_compatibility_id") != "c2"
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis") != AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != os.getuid()
        or authorization.get("scientific_output_contract")
        != _expected_scientific_output_contract()
        or authorization.get("scientific_authority")
        != _expected_scientific_authority()
        or authorization.get("mutation_authority")
        != _expected_mutation_authority()
        or authorization.get("expected_evidence_paths")
        != {
            label: str(path)
            for label, path in _evidence_paths().items()
            if label
            not in {
                "c1_failure_terminal",
                "r10_authorization",
                "r10_receipt",
            }
        }
        or authorization.get("c1_failure_terminal_root") != terminal_root
        or authorization.get("c2_prewrite_failure_terminal_root")
        != failure_root
        or authorization.get("compatibility_source_roots", {}).get(
            "c2_prewrite_failure_terminalizer"
        ) != failure_source_root
        or authorization.get("c1_expired_authorization_root") != c1_root
        or authorization.get("protected_unit_states")
        != {"old": old_state, "c1": c1_state}
        or authorization.get("scientific_authority", {}).get(
            "automatic_retry"
        )
        is not False
        or authorization.get("scientific_authority", {}).get("resume")
        is not False
        or authorization.get("scientific_authority", {}).get(
            "materialization_authorized"
        )
        is not False
    ):
        raise PermissionError("c2 authorization closure changed")
    return authorization, root


def validate_compat_authorization(
    path: Path | None = None,
    *,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    require_fresh: bool = True,
    require_future_absence: bool = True,
    now: Callable[[], datetime] = _utc_now,
) -> tuple[dict[str, object], dict[str, object]]:
    """c1-compatible consumer interface for the c2 authorization.

    ``require_future_absence`` controls only the c2 runtime spec/launch/
    artifact/lease namespace.  This unit-realization consumer is always
    preactivation, so original scientific outputs and every alias stay absent.
    """

    if not isinstance(require_future_absence, bool):
        raise TypeError("require_future_absence must be boolean")
    authorization, root = validate_c2_authorization(
        path=path,
        unit_state_reader=unit_state_reader,
        require_fresh=require_fresh,
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
        now=now,
    )
    _validate_scientific_output_phase(
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
    )
    if require_future_absence:
        _require_absent(_c2_future_paths())
    return authorization, root


def _validate_unit_chain(
    *,
    unit_state_reader: UnitStateReader,
    allow_runtime_activation: bool,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    datetime,
]:
    authorization, auth_root = _load_sealed(
        C2_UNIT_AUTHORIZATION_PATH,
        fingerprint_field="authorization_fingerprint",
        schema=UNIT_AUTHORIZATION_SCHEMA,
    )
    receipt, receipt_root = _load_sealed(
        C2_UNIT_RECEIPT_PATH,
        fingerprint_field="receipt_fingerprint",
        schema=UNIT_RECEIPT_SCHEMA,
    )
    _require_no_payload(authorization)
    _require_no_payload(receipt)
    issued = _parse_utc(
        authorization.get("issued_at_utc"),
        name="c2 realization issuance",
    )
    expires = _parse_utc(
        authorization.get("expires_at_utc"),
        name="c2 realization expiry",
    )
    receipt_time = _parse_utc(
        receipt.get("created_at_utc"),
        name="c2 realization receipt creation",
    )
    full_shadow = receipt.get("full_static_shadow")
    fragment = receipt.get("fragment_identity")
    if (
        authorization.get("unit_name") != C2_UNIT_NAME
        or receipt.get("unit_name") != C2_UNIT_NAME
        or receipt.get("passed") is not True
        or receipt.get("static") is not True
        or receipt.get("started") is not False
        or receipt.get("enabled") is not False
        or receipt.get("removed") is not False
        or not issued <= receipt_time <= expires
        or expires - issued > timedelta(seconds=300)
        or not isinstance(full_shadow, Mapping)
        or not isinstance(fragment, Mapping)
        or full_shadow.get("Id") != C2_UNIT_NAME
        or full_shadow.get("FragmentPath") != fragment.get("path")
        or receipt.get("manager_generation")
        != authorization.get("manager_generation")
    ):
        raise PermissionError("c2 unit realization chain changed")
    live = _normalized_state(unit_state_reader, C2_UNIT_NAME)
    if live.get("FragmentPath") != fragment.get("path"):
        raise PermissionError("c2 fragment path changed")
    fragment_root = _source_root(Path(str(fragment["path"])))
    if (
        fragment_root.get("file_sha256") != fragment.get("file_sha256")
        or fragment_root.get("device") != fragment.get("device")
        or fragment_root.get("inode") != fragment.get("inode")
    ):
        raise PermissionError("c2 fragment generation changed")
    for field in _STATE_FIELDS:
        if field in full_shadow and live.get(field) != full_shadow[field]:
            if (
                allow_runtime_activation
                and field in ("ActiveState", "SubState", "InvocationID")
            ):
                continue
            raise PermissionError(f"c2 live unit shadow changed: {field}")
    if not allow_runtime_activation:
        _require_inert_state(
            live,
            unit_name=C2_UNIT_NAME,
            fragment_path=str(fragment["path"]),
        )
    return authorization, auth_root, receipt, receipt_root, receipt_time


def _load_environment_evidence() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
]:
    policy, policy_root = _load_sealed(
        C2_ENVIRONMENT_POLICY_PATH,
        fingerprint_field="policy_fingerprint",
        schema=ENVIRONMENT_POLICY_SCHEMA,
    )
    stability, stability_root = _load_sealed(
        C2_ENVIRONMENT_STABILITY_PATH,
        fingerprint_field="stability_receipt_fingerprint",
        schema=ENVIRONMENT_STABILITY_SCHEMA,
    )
    postcleanup, postcleanup_root = _load_sealed(
        C2_ENVIRONMENT_POSTCLEANUP_PATH,
        fingerprint_field="receipt_fingerprint",
        schema=ENVIRONMENT_RECEIPT_SCHEMA,
    )
    for value in (policy, stability, postcleanup):
        _require_no_payload(value)
    return policy, stability, postcleanup, {
        "environment_policy": policy_root,
        "environment_stability": stability_root,
        "environment_postcleanup": postcleanup_root,
    }


def _collect_full_closure(
    *,
    authorization: Mapping[str, object],
    authorization_root: Mapping[str, object],
    unit_state_reader: UnitStateReader,
    allow_runtime_activation: bool,
    receipt_time: datetime,
) -> dict[str, object]:
    _failure, failure_root, failure_source_root = (
        _validate_prewrite_failure_terminal()
    )
    terminal, terminal_root, _terminalizer_root = (
        _validate_c1_failure_terminal(
            unit_state_reader=unit_state_reader,
            now=receipt_time,
        )
    )
    terminal_evidence = terminal.get("evidence_roots")
    if not isinstance(terminal_evidence, Mapping):
        raise PermissionError("c1 terminal evidence roots are absent")
    sources = authorization.get("compatibility_source_roots")
    _validate_source_roots(sources)
    if not isinstance(sources, Mapping):
        raise PermissionError("c2 source roots are malformed")
    if (
        sources.get("c2_prewrite_failure_terminalizer")
        != failure_source_root
        or authorization.get("c2_prewrite_failure_terminal_root")
        != failure_root
    ):
        raise PermissionError(
            "c2 prewrite failure lineage changed"
        )
    environment, wrapper_root = _load_verified_environment_wrapper(
        sources["compat_environment_wrapper"]
    )
    if wrapper_root != dict(sources["compat_environment_wrapper"]):
        raise PermissionError("c2 environment wrapper generation changed")
    (
        unit_authorization,
        unit_auth_root,
        unit_receipt,
        unit_receipt_root,
        unit_receipt_time,
    ) = _validate_unit_chain(
        unit_state_reader=unit_state_reader,
        allow_runtime_activation=allow_runtime_activation,
    )
    archival = environment._production_archival_validator(
        C2_UNIT_AUTHORIZATION_PATH,
        C2_UNIT_RECEIPT_PATH,
    )
    if (
        not isinstance(archival, Mapping)
        or archival.get("authorization") != unit_authorization
        or archival.get("receipt") != unit_receipt
    ):
        raise PermissionError("c2 realization archival validator diverged")
    historical_contract, current_contract, _replay_roots = (
        environment.replay_old_scope_and_handoff()
    )
    policy, stability, postcleanup, environment_roots = (
        _load_environment_evidence()
    )
    validated = environment.validate_c2_environment_closure(
        policy,
        stability,
        postcleanup,
        archival=archival,
        c2_contract=current_contract,
    )
    if (
        not isinstance(validated, Mapping)
        or validated.get("policy") != policy
        or validated.get("stability") != stability
        or validated.get("postcleanup") != postcleanup
    ):
        raise PermissionError("c2 environment wrapper returned a different closure")
    policy_time = _parse_utc(
        policy.get("created_at_utc"),
        name="c2 environment policy creation",
    )
    postcleanup_time = _parse_utc(
        postcleanup.get("created_at_utc"),
        name="c2 postcleanup creation",
    )
    c1_root = terminal_evidence.get("bridge_authorization")
    if (
        authorization.get("c1_failure_terminal_root") != terminal_root
        or authorization.get("c1_expired_authorization_root") != c1_root
        or authorization.get("r10_roots")
        != _r10_roots_from_terminal(terminal)
        or not unit_receipt_time < policy_time <= postcleanup_time
        or postcleanup_time > receipt_time
    ):
        raise PermissionError("c2 closure chronology/lineage changed")
    evidence_roots = {
        "c1_failure_terminal": terminal_root,
        "c2_prewrite_failure_terminal": failure_root,
        "r10_authorization": authorization["r10_roots"]["authorization"],
        "r10_receipt": authorization["r10_roots"]["receipt"],
        **environment_roots,
        "unit_realization_authorization": unit_auth_root,
        "unit_realization_receipt": unit_receipt_root,
    }
    if set(evidence_roots) != _EVIDENCE_LABELS:
        raise AssertionError("c2 evidence labels changed")
    return {
        "authorization_root": dict(authorization_root),
        "source_roots": dict(sources),
        "evidence_roots": evidence_roots,
        "historical_contract": _normalize_environment_contract(
            historical_contract
        ),
        "current_contract": _normalize_environment_contract(
            current_contract
        ),
        "unit_authorization": unit_authorization,
        "unit_receipt": unit_receipt,
        "policy": policy,
        "stability": stability,
        "postcleanup": postcleanup,
    }


def seal_receipt(
    *,
    receipt_path: Path | None = None,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    if receipt_path is None:
        receipt_path = C2_RECEIPT_PATH
    _require_fixed_path(
        receipt_path,
        C2_RECEIPT_PATH,
        name="c2 receipt",
    )
    if os.path.lexists(C2_RECEIPT_PATH):
        raise FileExistsError("c2 compatibility receipt already exists")
    _validate_scientific_output_phase(
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
    )
    _require_absent(_c2_future_paths())
    authorization, authorization_root = validate_c2_authorization(
        unit_state_reader=unit_state_reader,
        require_fresh=True,
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
        now=now,
    )
    created = now().astimezone(timezone.utc)
    _created, issued, expires = _authorization_times(
        authorization,
        current=created,
        require_fresh=True,
    )
    closure = _collect_full_closure(
        authorization=authorization,
        authorization_root=authorization_root,
        unit_state_reader=unit_state_reader,
        allow_runtime_activation=False,
        receipt_time=created,
    )
    if not issued <= created <= expires:
        raise PermissionError("c2 receipt is outside authorization window")
    body: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA,
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
        "created_at_utc": _format_utc(created),
        "compatibility_authorization_root": authorization_root,
        "compatibility_source_roots": closure["source_roots"],
        "compatibility_evidence_roots": closure["evidence_roots"],
        "historical_environment_contract": (
            closure["historical_contract"]
        ),
        "current_environment_contract": closure["current_contract"],
        "scientific_output_contract": (
            authorization["scientific_output_contract"]
        ),
        "scientific_authority": authorization["scientific_authority"],
        "compatibility_closure_passed": True,
        "runtime_launch_authorized": False,
        "systemd_start_authorized": False,
        "automatic_retry": False,
        "resume": False,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
        "training_started": False,
        "materialization_consumed": False,
    }
    return _write_sealed(
        C2_RECEIPT_PATH,
        body,
        fingerprint_field="receipt_fingerprint",
    )


def _validate_receipt_evidence_roots(
    roots: object,
    expected: Mapping[str, object],
) -> None:
    if (
        not isinstance(roots, Mapping)
        or set(roots) != _EVIDENCE_LABELS
        or dict(roots) != dict(expected)
    ):
        raise PermissionError("c2 receipt evidence-root labels changed")


def _validate_expected_runtime_spec_contract(
    expected_spec: Mapping[str, object],
) -> None:
    systemd = expected_spec.get("systemd")
    artifacts = expected_spec.get("artifacts")
    if (
        not isinstance(systemd, Mapping)
        or not isinstance(artifacts, Mapping)
        or expected_spec.get("attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or systemd.get("unit_name") != C2_UNIT_NAME
        or artifacts.get("root") != str(C2_RUNTIME_ARTIFACT_ROOT)
    ):
        raise PermissionError("expected spec is outside c2")


def verify_compatibility_receipt(
    path: Path | None = None,
    expected_spec: Mapping[str, object] | None = None,
    require_spec_binding: bool = False,
    allow_runtime_activation: bool = False,
    *,
    runtime_phase: str | None = None,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    phase = _resolve_runtime_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    if path is None:
        path = C2_RECEIPT_PATH
    fixed = _require_fixed_path(
        path,
        C2_RECEIPT_PATH,
        name="c2 receipt",
    )
    if type(require_spec_binding) is not bool:
        raise TypeError("c2 verification phase flags must be boolean")
    if (
        (expected_spec is None and require_spec_binding)
        or (expected_spec is not None and not require_spec_binding)
    ):
        raise PermissionError("c2 runtime-spec verification phase changed")
    receipt, receipt_root = _load_sealed(
        fixed,
        fingerprint_field="receipt_fingerprint",
        schema=RECEIPT_SCHEMA,
    )
    _validate_common_identity(receipt)
    _require_no_payload(receipt)
    if set(receipt) != _RECEIPT_KEYS:
        raise PermissionError("c2 receipt keys changed")
    if (
        receipt.get("runtime_compatibility_id") != "c2"
        or receipt.get("compatibility_closure_passed") is not True
        or receipt.get("runtime_launch_authorized") is not False
        or receipt.get("systemd_start_authorized") is not False
        or receipt.get("automatic_retry") is not False
        or receipt.get("resume") is not False
        or receipt.get("scientific_output_contract")
        != _expected_scientific_output_contract()
        or receipt.get("scientific_authority")
        != _expected_scientific_authority()
    ):
        raise PermissionError("c2 receipt semantics changed")
    authorization, authorization_root = validate_c2_authorization(
        unit_state_reader=unit_state_reader,
        require_fresh=False,
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=phase,
        now=now,
    )
    receipt_time = _parse_utc(
        receipt.get("created_at_utc"),
        name="c2 receipt creation",
    )
    _created, issued, expires = _authorization_times(
        authorization,
        current=receipt_time,
        require_fresh=False,
    )
    if not issued <= receipt_time <= expires:
        raise PermissionError("archival c2 receipt chronology changed")
    closure = _collect_full_closure(
        authorization=authorization,
        authorization_root=authorization_root,
        unit_state_reader=unit_state_reader,
        allow_runtime_activation=allow_runtime_activation,
        receipt_time=receipt_time,
    )
    _validate_receipt_evidence_roots(
        receipt.get("compatibility_evidence_roots"),
        closure["evidence_roots"],
    )
    _validate_source_roots(
        receipt.get("compatibility_source_roots")
    )
    if (
        receipt.get("compatibility_authorization_root")
        != authorization_root
        or receipt.get("compatibility_source_roots")
        != closure["source_roots"]
        or receipt.get("compatibility_evidence_roots")
        != closure["evidence_roots"]
        or receipt.get("historical_environment_contract")
        != closure["historical_contract"]
        or receipt.get("current_environment_contract")
        != closure["current_contract"]
        or receipt.get("scientific_output_contract")
        != authorization["scientific_output_contract"]
    ):
        raise PermissionError("c2 compatibility receipt lineage changed")
    _validate_scientific_output_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=phase,
    )
    if expected_spec is None:
        _require_absent(_c2_future_paths())
    else:
        if not isinstance(expected_spec, Mapping):
            raise TypeError("expected c2 runtime spec must be a mapping")
        sealed_spec, _root = _load_sealed(
            C2_RUNTIME_SPEC_PATH,
            fingerprint_field="runtime_spec_fingerprint",
            schema=RUNTIME_SPEC_SCHEMA,
        )
        if sealed_spec != dict(expected_spec):
            raise PermissionError("expected c2 runtime spec changed")
        _validate_expected_runtime_spec_contract(sealed_spec)
    result = dict(receipt)
    result["receipt_root"] = receipt_root
    return result


def verify_compatibility_prewrite_spec(
    path: Path,
    expected_spec: Mapping[str, object],
    *,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    """Validate one exact producer preview while every runtime path is absent."""

    if not isinstance(expected_spec, Mapping):
        raise TypeError("expected_spec must be a mapping")
    receipt = verify_compatibility_receipt(
        path=path,
        expected_spec=None,
        require_spec_binding=False,
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
        unit_state_reader=unit_state_reader,
        now=now,
    )
    _validate_expected_runtime_spec_contract(expected_spec)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CURE-Lite v24 c2 preaccess compatibility closure",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize-c2")
    authorize.add_argument("--instruction-id", required=True)
    authorize.add_argument("--authorization-basis", required=True)
    authorize.add_argument(
        "--validity-seconds",
        type=int,
        default=300,
    )
    subparsers.add_parser("seal-receipt")
    verify = subparsers.add_parser("verify-compatibility-receipt")
    verify.add_argument("--expected-spec", type=Path)
    verify.add_argument("--allow-runtime-activation", action="store_true")
    verify.add_argument("--runtime-phase", choices=sorted(RUNTIME_PHASES))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "authorize-c2":
        result = authorize_c2(
            instruction_id=args.instruction_id,
            authorization_basis=args.authorization_basis,
            validity_seconds=args.validity_seconds,
        )
        summary = {
            "authorization_fingerprint": result[
                "authorization_fingerprint"
            ],
            "expires_at_utc": result["expires_at_utc"],
        }
    elif args.command == "seal-receipt":
        result = seal_receipt()
        summary = {
            "receipt_fingerprint": result["receipt_fingerprint"],
            "compatibility_closure_passed": result[
                "compatibility_closure_passed"
            ],
        }
    else:
        expected = None
        if args.expected_spec is not None:
            expected, _root = _load_sealed(
                _require_fixed_path(
                    args.expected_spec,
                    C2_RUNTIME_SPEC_PATH,
                    name="c2 runtime spec",
                ),
                fingerprint_field="runtime_spec_fingerprint",
                schema=RUNTIME_SPEC_SCHEMA,
            )
        result = verify_compatibility_receipt(
            expected_spec=expected,
            require_spec_binding=expected is not None,
            allow_runtime_activation=args.allow_runtime_activation,
            runtime_phase=args.runtime_phase,
        )
        summary = {
            "receipt_fingerprint": result["receipt_fingerprint"],
            "compatibility_closure_passed": result[
                "compatibility_closure_passed"
            ],
        }
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
