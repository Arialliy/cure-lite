#!/usr/bin/env python3
"""Data-blind, create-once process supervisor for a v24 D_R attempt.

This module is deliberately standard-library-only.  It does not import the
v24 gate, any dataset package, torch, or a model implementation.  Scientific
authorization and scientific gate decisions remain the child's responsibility;
the supervisor only establishes process-lifecycle evidence.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Mapping, Sequence


RUNTIME_LAUNCH_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-D_R-structural-r2-runtime-launch-authorization-v1"
)
RUNTIME_ATTESTATION_SCHEMA = (
    "cure-lite-v24-D_R-structural-r2-runtime-attestation-v1"
)
_RUNTIME_ATTESTATION_ENV = "CURE_LITE_V24_RUNTIME_ATTESTATION_PATH"
_RUNTIME_RESERVED_ENV_PREFIX = "CURE_LITE_V24_RUNTIME_"
RUNTIME_SPEC_SCHEMA = "cure-lite-v24-dr-runtime-supervisor-spec-v2"
ATTEMPT_COMMIT_SCHEMA = "cure-lite-v24-dr-attempt-commit-v2"
MATERIALIZATION_CLAIM_SCHEMA = (
    "cure-lite-v24-dr-materialization-claim-v2"
)
RUNTIME_HEARTBEAT_SCHEMA = "cure-lite-v24-dr-runtime-heartbeat-v1"
RUNTIME_TERMINAL_SCHEMA = "cure-lite-v24-dr-runtime-terminal-v1"
SYSTEMD_TERMINAL_SCHEMA = "cure-lite-v24-dr-systemd-terminal-v1"
RUNTIME_PHASE_RECEIPT_SCHEMA = (
    "cure-lite-v24-dr-runtime-phase-receipt-v1"
)
LAUNCH_LEASE_SCHEMA = "cure-lite-v24-dr-launch-lease-v1"
CONSUMED_START_FAILURE_SCHEMA = (
    "cure-lite-v24-dr-consumed-start-failure-v1"
)

ACTUAL_EXECUTION_KIND = "actual_D_R"
_ACTUAL_CANDIDATE = "GCR-PACRE-v24"
_ACTUAL_STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
_ACTUAL_ATTEMPT_ID = "gcr_pacre_v24_D_R_zero_update_structural_r2"
_ACTUAL_UNIT_NAME = "cure-lite-v24-gcr-pacre-dr-r2.service"
_ACTUAL_INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
_ACTUAL_AUTHORIZATION_BASIS = "user instruction: 修改后运行"
_ACTUAL_SPEC_PATH = (
    "/home/md0/ly/cure_lite/protocols/IRSTD-1K/gcr_pacre_v24/"
    "runtime_evidence_r2/D_R_structural_attempt_r2_runtime_spec.json"
)
_ACTUAL_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    "/home/md0/ly/cure_lite/protocols/IRSTD-1K/gcr_pacre_v24/"
    "runtime_evidence_r2/"
    "D_R_structural_attempt_r2_runtime_launch_authorization.json"
)
_ACTUAL_SCIENTIFIC_AUTHORIZATION_PATH = (
    "/home/md0/ly/cure_lite/protocols/IRSTD-1K/gcr_pacre_v24/"
    "runtime_evidence_r2/D_R_structural_attempt_r2_authorization.json"
)
_ACTUAL_SCIENTIFIC_ACCESS_AUDIT_PATH = (
    "/home/md0/ly/cure_lite/protocols/IRSTD-1K/gcr_pacre_v24/"
    "runtime_evidence_r2/D_R_structural_attempt_r2_access_audit.json"
)
_ACTUAL_ADAPTER_PATH = (
    "/home/md0/ly/cure_lite/tools/"
    "run_cure_lite_v24_gcr_pacre_dr_gate_r2.py"
)
_ACTUAL_LEGACY_ENTRYPOINT_PATH = (
    "/home/md0/ly/cure_lite/tools/"
    "run_cure_lite_v24_gcr_pacre_dr_gate.py"
)
_ACTUAL_SOURCE_CLOSURE_FINGERPRINT_103 = (
    "28d26759a68785e9c99917fcfa8b36430c7f6e5463282d66eeab5c711e425e9f"
)
_ACTUAL_PYTHON_PATH = "/usr/bin/python3.12"
_ACTUAL_RUNTIME_DEPENDENCY_SITE_PATH = (
    "/home/md0/ly/MSHNet/.venv/lib/python3.12/site-packages"
)
_ACTUAL_RUNTIME_DEPENDENCY_SITE_DEVICE = 2304
_ACTUAL_RUNTIME_DEPENDENCY_SITE_INODE = 228331323
_ACTUAL_RUNTIME_DEPENDENCY_SITE_OWNER_UID = 1008
_ACTUAL_RUNTIME_DEPENDENCY_SITE_MODE = 0o775
DUMMY_EXECUTION_KIND = "generated_dummy"
SYSTEMD_INTEGRATION_DUMMY_KIND = "systemd_integration_dummy"

_SPEC_KEYS = {
    "schema_version",
    "execution_kind",
    "candidate",
    "stage_id",
    "attempt_id",
    "attempt_ordinal",
    "prior_attempt_count",
    "authorization",
    "scientific_preaccess",
    "child",
    "artifacts",
    "runtime",
    "environment",
    "source_bindings",
    "runtime_spec_fingerprint",
}
_CHILD_KEYS = {
    "argv",
    "argv_fingerprint",
    "cwd",
    "environment",
    "inherit_environment",
    "entrypoint_path",
}
_ARTIFACT_KEYS = {
    "root",
    "attempt_commit",
    "materialization_claim",
    "stdout_log",
    "stderr_log",
    "heartbeat_dir",
    "runtime_terminal",
    "systemd_invocation_dir",
    "launch_lease",
    "precommit_phase_receipt",
    "start_ack_receipt",
    "child_prespawn_phase_receipt",
    "consumed_start_failure_receipt",
    "gpu_lease_release_receipt",
    "runtime_attestation",
}
_RUNTIME_KEYS = {
    "shell",
    "start_new_session",
    "launch_limit",
    "automatic_retry_allowed",
    "resume_allowed",
    "restart",
    "heartbeat_interval_seconds",
    "poll_interval_seconds",
    "termination_grace_seconds",
    "systemd",
}
_SYSTEMD_KEYS = {
    "unit_name",
    "service_type",
    "kill_mode",
    "send_sigkill",
    "timeout_stop_seconds",
    "start_ack_timeout_seconds",
    "start_ack_poll_seconds",
    "unit_fragment_file_sha256",
    "immutable_shadow_properties",
    "immutable_shadow_fingerprint",
}
_SYSTEMD_IMMUTABLE_SHADOW_KEYS = {
    "Type",
    "Restart",
    "KillMode",
    "SendSIGKILL",
    "TimeoutStopUSec",
    "FragmentPath",
    "DropInPaths",
    "Transient",
    "Environment",
    "UnsetEnvironment",
    "WorkingDirectory",
    "UMask",
    "ExitType",
    "RuntimeMaxUSec",
    "WatchdogUSec",
    "OOMPolicy",
    "RemainAfterExit",
    "StandardInput",
    "StandardOutput",
    "StandardError",
    "StartLimitIntervalUSec",
    "StartLimitBurst",
    "KillSignal",
    "ExecCondition",
    "ExecStartPre",
    "ExecStart",
    "ExecStopPost",
}
_SYSTEMD_PHASE_STATE_KEYS = {
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "NRestarts",
    "NeedDaemonReload",
    "Result",
    "ExecMainCode",
    "ExecMainStatus",
    "InvocationID",
}
# Compatibility name for callers that inspect the queried static property set.
# It deliberately excludes mutable phase state in the v2 contract.
_SYSTEMD_SHADOW_KEYS = _SYSTEMD_IMMUTABLE_SHADOW_KEYS
_SYSTEMD_EXEC_MODES = {
    "ExecCondition": "claim-materialization",
    "ExecStartPre": "verify-runtime-spec",
    "ExecStart": "run-once",
    "ExecStopPost": "record-systemd-exit",
}
_SYSTEMD_EXEC_KEYS = frozenset(_SYSTEMD_EXEC_MODES)
_SYSTEMCTL_PATH = "/usr/bin/systemctl"
_CGROUP_FILESYSTEM_ROOT = Path("/sys/fs/cgroup")
_PR_SET_CHILD_SUBREAPER = 36
_SOURCE_BINDING_KEYS = {
    "supervisor_file_sha256",
    "child_entry_file_sha256",
    "prior_attempt_receipt_file_sha256",
    "runtime_environment_file_sha256",
    "r2_adapter_path",
    "r2_adapter_file_sha256",
    "legacy_gate_entrypoint_path",
    "legacy_gate_entrypoint_file_sha256",
    "python_path",
    "python_file_sha256",
    "python_device",
    "python_inode",
    "python_owner_uid",
    "python_mode",
    "runtime_dependency_site_path",
    "runtime_dependency_site_device",
    "runtime_dependency_site_inode",
    "runtime_dependency_site_owner_uid",
    "runtime_dependency_site_mode",
}
_ATTEMPT_COMMIT_KEYS = {
    "schema_version",
    "execution_kind",
    "candidate",
    "stage_id",
    "attempt_id",
    "attempt_ordinal",
    "prior_attempt_count",
    "runtime_spec_fingerprint",
    "authorization_fingerprint",
    "authorization_file_sha256",
    "precommit_phase_receipt_fingerprint",
    "precommit_phase_receipt_file_sha256",
    "launch_lease_fingerprint",
    "launch_lease_file_sha256",
    "dispatch_lease_scope",
    "planned_attempt_commit_fingerprint",
    "precommit_environment_audit_fingerprint",
    "precommit_environment_inventory_fingerprint",
    "runtime_environment_audit_valid",
    "gpu_lease_fingerprint",
    "gpu_lease_file_sha256",
    "gpu_lease_device",
    "gpu_lease_inode",
    "gpu_lease_parent_device",
    "gpu_lease_parent_inode",
    "time_utc",
    "monotonic_ns",
    "committer_pid",
    "committer_proc_starttime_ticks",
    "boot_id",
    "systemd_unit_name",
    "immutable_systemd_shadow_properties",
    "immutable_systemd_shadow_fingerprint",
    "launch_limit",
    "automatic_retry_allowed",
    "resume_allowed",
    "scientific_gate_passed",
    "attempt_commit_fingerprint",
}
_GPU_LEASE_BODY_KEYS = {
    "schema_version",
    "created_at_utc",
    "boot_id",
    "gpu_uuid",
    "runtime_spec_fingerprint",
    "attempt_id",
    "authorization_fingerprint",
    "planned_attempt_commit_fingerprint",
    "committer_pid",
    "committer_starttime",
}
_GPU_LEASE_PAYLOAD_KEYS = _GPU_LEASE_BODY_KEYS | {"lease_fingerprint"}
_AUTH_REFERENCE_KEYS = {"path", "required_schema"}
_SCIENTIFIC_PREACCESS_KEYS = {
    "authorization_path",
    "authorization_file_sha256",
    "authorization_fingerprint",
    "authorization_required_schema",
    "access_audit_path",
    "access_audit_file_sha256",
    "access_audit_fingerprint",
    "access_audit_required_schema",
    "source_closure_fingerprint_103",
}
_ENVIRONMENT_KEYS = {
    "policy_path",
    "policy_file_sha256",
    "inventory_path",
    "inventory_file_sha256",
    "cleanup_plan_path",
    "cleanup_plan_file_sha256",
    "cleanup_authorization_path",
    "cleanup_authorization_file_sha256",
    "cleanup_receipt_path",
    "cleanup_receipt_file_sha256",
    "stability_receipt_path",
    "stability_receipt_file_sha256",
    "integration_authorization_path",
    "integration_authorization_file_sha256",
    "integration_receipt_path",
    "integration_receipt_file_sha256",
    "unit_realization_authorization_path",
    "unit_realization_authorization_file_sha256",
    "unit_realization_receipt_path",
    "unit_realization_receipt_file_sha256",
    "selected_gpu_uuid",
    "selected_gpu_pci_bus_id",
    "selected_gpu_minor_number",
    "gpu_lease_path",
    "gpu_lease_tombstone_path",
}
_ENVIRONMENT_AUDIT_RECEIPT_KEYS = {
    "schema_version",
    "created_at_utc",
    "command",
    "environment_binding",
    "inventory",
    "passed",
    "error_type",
    "error_message",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "receipt_fingerprint",
}
_ENVIRONMENT_BINDING_KEYS = {
    "inventory_fingerprint",
    "boot_id",
    "runtime_directory",
    "runtime_directory_device",
    "runtime_directory_inode",
    "manager_identity",
}
_ENVIRONMENT_INVENTORY_KEYS = {
    "schema_version",
    "created_at_utc",
    "uid",
    "boot_id",
    "manager",
    "unit_scope",
    "gpu_snapshot",
    "blockers",
    "passed",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "inventory_fingerprint",
}
_ENVIRONMENT_MANAGER_KEYS = {
    "state",
    "allowed_states",
    "returncode",
    "failed_units",
    "allowed_failed_unit_ids",
    "unexpected_failed_unit_ids",
    "scoped_failed_unit_ids",
    "identity",
    "endpoint",
}
_ENVIRONMENT_MANAGER_IDENTITY_KEYS = {
    "pid",
    "starttime_ticks",
    "uid",
    "control_group",
}
_ENVIRONMENT_MANAGER_ENDPOINT_KEYS = {
    "uid",
    "runtime_directory",
    "runtime_directory_device",
    "runtime_directory_inode",
    "bus_path",
    "bus_device",
    "bus_inode",
}
_ENVIRONMENT_UNIT_SCOPE_KEYS = {
    "target_unit_id",
    "conflict_unit_ids",
    "dependency_unit_ids",
    "require_target_ready",
    "shadows",
}
_ENVIRONMENT_UNIT_SHADOW_KEYS = {
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "Restart",
    "RestartUSec",
    "NRestarts",
    "ControlGroup",
    "FragmentPath",
    "DropInPaths",
    "TriggeredBy",
    "Triggers",
    "WantedBy",
    "RequiredBy",
    "PartOf",
}
_ENVIRONMENT_GPU_SNAPSHOT_KEYS = {
    "schema_version",
    "selected_gpu_uuid",
    "expected_uid",
    "allowed_unit_ids",
    "strict_all_gpu_consumers",
    "devices",
    "first_apps",
    "second_apps",
    "process_unit_mapping",
    "observations",
    "blockers",
    "passed",
    "snapshot_fingerprint",
}
_ENVIRONMENT_GPU_DEVICE_KEYS = {
    "index",
    "uuid",
    "pci_bus_id",
    "compute_mode",
    "mig_mode",
    "driver_version",
    "minor_number",
    "mps_state",
}
_ENVIRONMENT_GPU_APP_KEYS = {
    "pid",
    "gpu_uuid",
    "process_name",
    "used_gpu_memory_mib",
}
_ENVIRONMENT_GPU_PROCESS_UNIT_KEYS = {
    "pid",
    "starttime_ticks",
    "uid",
    "gpu_uuid",
    "cgroup_path",
    "unit_id",
}
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{16,64}")
_PCI_BUS_ID = re.compile(
    r"(?:[0-9A-Fa-f]{8}:)?[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]"
)
_INTEGRATION_SCENARIO_ID = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{6,61}[a-z0-9])?-[0-9a-f]{16}"
)


def _requires_systemd_commit(spec: Mapping[str, object]) -> bool:
    return spec.get("execution_kind") in {
        ACTUAL_EXECUTION_KIND,
        SYSTEMD_INTEGRATION_DUMMY_KIND,
    }


def build_systemd_integration_identity(
    scenario_id: str,
) -> dict[str, str]:
    """Build one scenario-unique identity shared with the real harness."""

    if (
        not isinstance(scenario_id, str)
        or _INTEGRATION_SCENARIO_ID.fullmatch(scenario_id) is None
    ):
        raise ValueError("integration scenario_id is not unique and canonical")
    return {
        "candidate": "systemd-integration-dummy",
        "stage_id": f"systemd_integration_dummy_{scenario_id}",
        "attempt_id": (
            f"systemd_integration_dummy_attempt_{scenario_id}"
        ),
        "unit_name": (
            "cure-lite-v24-supervisor-integration-"
            f"{scenario_id}.service"
        ),
    }


def canonical_json(value: object) -> str:
    """Return deterministic JSON without importing repository helpers."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _deep_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return (
            set(left) == set(right)
            and all(
                _deep_exact_equal(left[key], right[key])
                for key in left
            )
        )
    if isinstance(left, list):
        return (
            len(left) == len(right)
            and all(
                _deep_exact_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    return left == right


def sealed_file_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        (canonical_json(dict(value)) + "\n").encode("utf-8")
    ).hexdigest()


def _read_fd_bytes(descriptor: int) -> bytes:
    blocks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        blocks.append(block)
    return b"".join(blocks)


def _read_stable_regular_file(
    path: str | Path,
    *,
    expected_uid: int | None = None,
    expected_mode: int | None = None,
    expected_nlink: int | None = None,
    name: str = "file",
) -> tuple[bytes, os.stat_result]:
    """Read one canonical regular-file generation through one descriptor."""

    target = Path(path)
    if not target.is_absolute():
        target = Path(os.path.abspath(target))
    parent = target.parent
    if parent.resolve(strict=True) != parent:
        raise PermissionError(f"{name} parent is not canonical")
    parent_before = parent.lstat()
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    parent_fd = os.open(parent, parent_flags)
    descriptor = -1
    try:
        parent_opened = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_opened.st_mode)
            or (parent_opened.st_dev, parent_opened.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
        ):
            raise PermissionError(f"{name} parent changed before read")
        before = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError(f"{name} is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(target.name, flags, dir_fd=parent_fd)
        opened = os.fstat(descriptor)

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_uid,
                stat.S_IMODE(value.st_mode),
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if (
            not stat.S_ISREG(opened.st_mode)
            or identity(opened) != identity(before)
            or (
                expected_uid is not None
                and opened.st_uid != expected_uid
            )
            or (
                expected_mode is not None
                and stat.S_IMODE(opened.st_mode) != expected_mode
            )
            or (
                expected_nlink is not None
                and opened.st_nlink != expected_nlink
            )
        ):
            raise PermissionError(f"{name} changed before descriptor read")
        raw = _read_fd_bytes(descriptor)
        finished = os.fstat(descriptor)
        linked = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        parent_finished = os.fstat(parent_fd)
        parent_linked = parent.lstat()
        if (
            identity(finished) != identity(opened)
            or identity(linked) != identity(opened)
            or (parent_finished.st_dev, parent_finished.st_ino)
            != (parent_opened.st_dev, parent_opened.st_ino)
            or (parent_linked.st_dev, parent_linked.st_ino)
            != (parent_opened.st_dev, parent_opened.st_ino)
        ):
            raise PermissionError(f"{name} changed during descriptor read")
        return raw, finished
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _read_stable_open_descriptor(
    descriptor: int,
    path: Path,
    *,
    expected_uid: int,
    expected_mode: int,
    expected_nlink: int,
    name: str,
) -> tuple[bytes, os.stat_result]:
    """Read and bind the already-open descriptor that carries a live lock."""

    before = os.fstat(descriptor)
    linked = path.lstat()

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_uid,
            stat.S_IMODE(value.st_mode),
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        not stat.S_ISREG(before.st_mode)
        or not stat.S_ISREG(linked.st_mode)
        or identity(linked) != identity(before)
        or before.st_uid != expected_uid
        or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_nlink != expected_nlink
    ):
        raise PermissionError(f"{name} descriptor identity is invalid")
    blocks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        block = os.pread(
            descriptor,
            min(1024 * 1024, before.st_size - offset),
            offset,
        )
        if not block:
            raise OSError(f"{name} descriptor read was truncated")
        blocks.append(block)
        offset += len(block)
    if os.pread(descriptor, 1, before.st_size):
        raise PermissionError(f"{name} grew during descriptor read")
    finished = os.fstat(descriptor)
    linked_after = path.lstat()
    if (
        identity(finished) != identity(before)
        or identity(linked_after) != identity(before)
    ):
        raise PermissionError(f"{name} changed during descriptor read")
    return b"".join(blocks), finished


def _read_stable_gpu_lease_descriptor(
    descriptor: int,
    parent_descriptor: int,
    path: Path,
    *,
    expected_uid: int,
    expected_mode: int,
    expected_nlink: int,
    name: str,
) -> tuple[bytes, os.stat_result, os.stat_result]:
    """Read one locked lease through its stable parent directory handle."""

    target = path.absolute()
    parent = target.parent
    if parent.resolve(strict=True) != parent:
        raise PermissionError(f"{name} parent is not canonical")

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_uid,
            stat.S_IMODE(value.st_mode),
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    parent_before = os.fstat(parent_descriptor)
    parent_linked_before = parent.lstat()
    before = os.fstat(descriptor)
    linked_before = os.stat(
        target.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or (parent_linked_before.st_dev, parent_linked_before.st_ino)
        != (parent_before.st_dev, parent_before.st_ino)
        or not stat.S_ISREG(before.st_mode)
        or not stat.S_ISREG(linked_before.st_mode)
        or identity(linked_before) != identity(before)
        or before.st_uid != expected_uid
        or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_nlink != expected_nlink
    ):
        raise PermissionError(f"{name} descriptor identity is invalid")

    blocks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        block = os.pread(
            descriptor,
            min(1024 * 1024, before.st_size - offset),
            offset,
        )
        if not block:
            raise OSError(f"{name} descriptor read was truncated")
        blocks.append(block)
        offset += len(block)
    if os.pread(descriptor, 1, before.st_size):
        raise PermissionError(f"{name} grew during descriptor read")

    finished = os.fstat(descriptor)
    linked_after = os.stat(
        target.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    parent_finished = os.fstat(parent_descriptor)
    parent_linked_after = parent.lstat()
    if (
        identity(finished) != identity(before)
        or identity(linked_after) != identity(before)
        or (parent_finished.st_dev, parent_finished.st_ino)
        != (parent_before.st_dev, parent_before.st_ino)
        or (parent_linked_after.st_dev, parent_linked_after.st_ino)
        != (parent_before.st_dev, parent_before.st_ino)
    ):
        raise PermissionError(f"{name} changed during descriptor read")
    return b"".join(blocks), finished, parent_finished


def file_sha256(path: str | Path) -> str:
    raw, _identity = _read_stable_regular_file(path)
    return hashlib.sha256(raw).hexdigest()


def _normalize_systemd_shadow_value(name: str, value: str) -> str:
    """Validate the whole Exec serialization before removing runtime fields."""

    if name == "WatchdogUSec":
        return (
            "disabled"
            if value in {"0", "infinity", "disabled"}
            else value
        )
    if name not in _SYSTEMD_EXEC_KEYS:
        return value
    identity = _normalized_systemd_exec_identity(value)
    return (
        f"{{ path={identity['path']} ; "
        f"argv[]={' '.join(identity['argv'])} ; "
        f"ignore_errors={identity['ignore_errors']} }}"
    )


def _normalized_systemd_exec_identity(value: str) -> dict[str, object]:
    if (
        not isinstance(value, str)
        or not value.startswith("{")
        or not value.endswith("}")
    ):
        raise ValueError("systemd execution identity is ambiguous")
    body = value[1:-1].strip()
    rows = [row.strip() for row in body.split(";")]
    if not rows or any(not row for row in rows):
        raise ValueError("systemd execution identity is ambiguous")
    allowed_static = {"path", "argv[]", "ignore_errors"}
    allowed_runtime = {
        "start_time",
        "stop_time",
        "pid",
        "code",
        "status",
    }
    fields: dict[str, str] = {}
    for row in rows:
        key, separator, raw = row.partition("=")
        if (
            separator != "="
            or key not in allowed_static | allowed_runtime
            or key in fields
            or not raw
        ):
            raise ValueError("systemd execution identity is ambiguous")
        fields[key] = raw.strip()
    if set(fields) - allowed_runtime != allowed_static:
        raise ValueError("systemd execution identity is ambiguous")
    for name in ("start_time", "stop_time"):
        if name in fields and re.fullmatch(r"\[[^\]\r\n]*\]", fields[name]) is None:
            raise ValueError("systemd execution runtime fields are malformed")
    if "pid" in fields and re.fullmatch(r"[0-9]+", fields["pid"]) is None:
        raise ValueError("systemd execution runtime fields are malformed")
    if "code" in fields and re.fullmatch(
        r"(?:\(null\)|[A-Za-z0-9_-]+)",
        fields["code"],
    ) is None:
        raise ValueError("systemd execution runtime fields are malformed")
    if "status" in fields and re.fullmatch(
        r"[0-9]+(?:/[A-Za-z0-9_-]+)?",
        fields["status"],
    ) is None:
        raise ValueError("systemd execution runtime fields are malformed")
    argv = fields["argv[]"].split()
    if (
        not argv
        or fields["path"] != argv[0]
        or fields["ignore_errors"] not in {"yes", "no"}
    ):
        raise ValueError("systemd execution path and argv diverged")
    return {
        "path": fields["path"],
        "argv": argv,
        "ignore_errors": fields["ignore_errors"],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    compact = value.replace("-", "")
    if (
        len(compact) != 32
        or any(character not in "0123456789abcdef" for character in compact)
    ):
        raise RuntimeError("Linux boot_id is unavailable or malformed")
    return value


def _proc_starttime_ticks(pid: int) -> int:
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be positive")
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing = raw.rfind(") ")
    if closing < 0:
        raise RuntimeError("proc stat has no command boundary")
    fields_from_state = raw[closing + 2 :].split()
    if len(fields_from_state) <= 19:
        raise RuntimeError("proc stat is truncated")
    value = int(fields_from_state[19])
    if value <= 0:
        raise RuntimeError("proc starttime is invalid")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_positive_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _close_descriptors_best_effort(
    *descriptors: int,
) -> BaseException | None:
    """Attempt every distinct fd close and retain the first close failure."""

    first_error: BaseException | None = None
    attempted: set[int] = set()
    for descriptor in descriptors:
        if (
            isinstance(descriptor, bool)
            or not isinstance(descriptor, int)
            or descriptor < 0
            or descriptor in attempted
        ):
            continue
        attempted.add(descriptor)
        try:
            os.close(descriptor)
        except BaseException as error:
            if first_error is None:
                first_error = error
    return first_error


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} does not have the closed schema")
    return value


def _canonical_regular_file(path: str | Path, *, name: str) -> Path:
    supplied = Path(path)
    if not supplied.is_absolute():
        supplied = Path(os.path.abspath(supplied))
    if (
        not supplied.is_file()
        or supplied.is_symlink()
        or supplied.resolve(strict=True) != supplied
        or supplied.stat().st_nlink != 1
    ):
        raise ValueError(f"{name} is not a unique canonical regular file")
    return supplied


def _read_canonical_json(
    path: str | Path,
    *,
    name: str,
    expected_mode: int | None = 0o444,
) -> dict[str, object]:
    raw, _identity = _read_stable_regular_file(
        path,
        expected_uid=os.getuid(),
        expected_mode=expected_mode,
        expected_nlink=1,
        name=name,
    )
    return _decode_canonical_json(raw, name=name)


def _decode_canonical_json(
    raw: bytes,
    *,
    name: str,
) -> dict[str, object]:
    if not raw.endswith(b"\n"):
        raise ValueError(f"{name} must end in exactly one canonical newline")
    text = raw[:-1].decode("utf-8", errors="strict")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} contains invalid JSON") from error
    if not isinstance(payload, dict) or canonical_json(payload) != text:
        raise ValueError(f"{name} is not one canonical JSON object")
    return payload


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_flags() -> int:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _write_new_json(path: Path, payload: Mapping[str, object]) -> str:
    """Write one evidence object with O_EXCL and never roll it back."""

    encoded = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    target = path.absolute()
    if target.parent.resolve(strict=True) != target.parent:
        raise PermissionError("create-once JSON parent is not canonical")
    parent_before = target.parent.lstat()
    parent_fd = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        parent_opened = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_opened.st_mode)
            or (parent_opened.st_dev, parent_opened.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
        ):
            raise PermissionError("create-once JSON parent changed before write")
        descriptor = os.open(
            target.name,
            _open_flags(),
            0o444,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("zero-byte create-once JSON write")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        linked = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != linked.st_dev
            or opened.st_ino != linked.st_ino
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o444
        ):
            raise RuntimeError("create-once JSON inode binding failed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        readback = _read_fd_bytes(descriptor)
        finished = os.fstat(descriptor)
        linked_after = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        parent_finished = os.fstat(parent_fd)
        parent_linked = target.parent.lstat()
        if (
            readback != encoded
            or finished.st_dev != opened.st_dev
            or finished.st_ino != opened.st_ino
            or finished.st_size != opened.st_size
            or finished.st_mtime_ns != opened.st_mtime_ns
            or finished.st_ctime_ns != opened.st_ctime_ns
            or linked_after.st_dev != opened.st_dev
            or linked_after.st_ino != opened.st_ino
            or linked_after.st_uid != opened.st_uid
            or linked_after.st_nlink != 1
            or stat.S_IMODE(linked_after.st_mode) != 0o444
            or (parent_finished.st_dev, parent_finished.st_ino)
            != (parent_opened.st_dev, parent_opened.st_ino)
            or (parent_linked.st_dev, parent_linked.st_ino)
            != (parent_opened.st_dev, parent_opened.st_ino)
        ):
            raise RuntimeError("create-once JSON failed descriptor readback")
        os.fsync(parent_fd)
    except BaseException:
        # A partially written create-once evidence file is itself evidence that
        # the identity was consumed.  It must never be unlinked or retried.
        try:
            os.fsync(parent_fd)
        finally:
            raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    return hashlib.sha256(encoded).hexdigest()


def _unlink_exact_json_generation(
    path: Path,
    payload: Mapping[str, object],
    *,
    expected_mode: int,
    name: str,
) -> None:
    """Unlink only the directory-entry generation carrying this exact JSON."""

    target = path.absolute()
    encoded = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    if target.parent.resolve(strict=True) != target.parent:
        raise PermissionError(f"{name} parent is not canonical")
    parent_before = target.parent.lstat()
    parent_fd = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        parent_opened = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_opened.st_mode)
            or (parent_opened.st_dev, parent_opened.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
        ):
            raise PermissionError(f"{name} parent changed before unlink")
        descriptor = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        linked = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (linked.st_dev, linked.st_ino)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != expected_mode
        ):
            raise PermissionError(f"{name} exact unlink identity is invalid")
        raw, finished = _read_stable_open_descriptor(
            descriptor,
            target,
            expected_uid=os.getuid(),
            expected_mode=expected_mode,
            expected_nlink=1,
            name=name,
        )
        linked_final = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        parent_final = os.fstat(parent_fd)
        parent_path_final = target.parent.lstat()
        if (
            raw != encoded
            or (linked_final.st_dev, linked_final.st_ino)
            != (finished.st_dev, finished.st_ino)
            or (parent_final.st_dev, parent_final.st_ino)
            != (parent_opened.st_dev, parent_opened.st_ino)
            or (parent_path_final.st_dev, parent_path_final.st_ino)
            != (parent_opened.st_dev, parent_opened.st_ino)
        ):
            raise PermissionError(f"{name} changed at exact unlink")
        os.unlink(target.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _seal(
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    materialized = dict(body)
    return {
        **materialized,
        fingerprint_field: stable_fingerprint(materialized),
    }


def _absolute_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return path


def _positive_number(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not float(value) > 0.0
    ):
        raise ValueError(f"{name} must be positive")
    return float(value)


def _validate_environment_structure(spec: Mapping[str, object]) -> None:
    environment = _require_exact_keys(
        spec["environment"],
        _ENVIRONMENT_KEYS,
        name="environment contract",
    )
    evidence_paths: list[Path] = []
    for field in sorted(_ENVIRONMENT_KEYS):
        if field.endswith("_path"):
            evidence_paths.append(
                _absolute_path(environment[field], name=f"environment.{field}")
            )
        elif field.endswith("_file_sha256") and not _is_sha256(
            environment[field]
        ):
            raise ValueError(f"environment.{field} is malformed")
    if len(evidence_paths) != len(set(evidence_paths)):
        raise ValueError("environment paths must be distinct")
    if (
        not isinstance(environment["selected_gpu_uuid"], str)
        or _GPU_UUID.fullmatch(environment["selected_gpu_uuid"]) is None
        or not isinstance(environment["selected_gpu_pci_bus_id"], str)
        or _PCI_BUS_ID.fullmatch(
            environment["selected_gpu_pci_bus_id"]
        )
        is None
        or isinstance(environment["selected_gpu_minor_number"], bool)
        or not isinstance(environment["selected_gpu_minor_number"], int)
        or environment["selected_gpu_minor_number"] < 0
    ):
        raise ValueError("selected GPU physical identity is malformed")


def _environment_evidence_bindings(
    spec: Mapping[str, object],
) -> dict[str, object]:
    environment = spec["environment"]
    if not isinstance(environment, Mapping):
        raise AssertionError("validated environment contract changed")
    definitions = (
        ("policy", "policy", "policy_fingerprint", "environment_policy_fingerprint"),
        (
            "inventory",
            "inventory receipt",
            "receipt_fingerprint",
            "environment_inventory_receipt_fingerprint",
        ),
        ("cleanup_plan", "cleanup plan", "plan_fingerprint", "cleanup_plan_fingerprint"),
        ("cleanup_authorization", "cleanup authorization", "authorization_fingerprint", "cleanup_authorization_fingerprint"),
        ("cleanup_receipt", "cleanup receipt", "cleanup_receipt_fingerprint", "cleanup_receipt_fingerprint"),
        ("stability_receipt", "stability receipt", "stability_receipt_fingerprint", "environment_stability_receipt_fingerprint"),
        ("integration_authorization", "integration authorization", "authorization_fingerprint", "user_systemd_integration_authorization_fingerprint"),
        ("integration_receipt", "integration receipt", "receipt_fingerprint", "user_systemd_integration_receipt_fingerprint"),
        ("unit_realization_authorization", "unit realization authorization", "authorization_fingerprint", "unit_realization_authorization_fingerprint"),
        ("unit_realization_receipt", "unit realization receipt", "receipt_fingerprint", "unit_realization_receipt_fingerprint"),
    )
    result: dict[str, object] = {}
    payloads: dict[str, dict[str, object]] = {}
    for prefix, name, fingerprint_field, authorization_field in definitions:
        path = _canonical_regular_file(
            str(environment[f"{prefix}_path"]),
            name=name,
        )
        expected_sha256 = environment[f"{prefix}_file_sha256"]
        payload = _read_canonical_json(path, name=name)
        if sealed_file_sha256(payload) != expected_sha256:
            raise PermissionError(f"{name} file binding changed")
        body = dict(payload)
        fingerprint = body.pop(fingerprint_field, None)
        if not _is_sha256(fingerprint) or fingerprint != stable_fingerprint(body):
            raise PermissionError(f"{name} fingerprint is invalid")
        if any(
            payload.get(field) is True
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        ):
            raise PermissionError("preauthorization environment accessed payload")
        result[authorization_field] = fingerprint
        payloads[prefix] = payload
    inventory_receipt = payloads["inventory"]
    inventory = inventory_receipt.get("inventory")
    if not isinstance(inventory, Mapping):
        raise PermissionError("environment inventory receipt has no inventory")
    nested_body = dict(inventory)
    nested_fingerprint = nested_body.pop("inventory_fingerprint", None)
    if (
        inventory_receipt.get("schema_version")
        != "cure-lite-v24-runtime-environment-audit-receipt-v1"
        or inventory_receipt.get("passed") is not True
        or not _is_sha256(nested_fingerprint)
        or nested_fingerprint != stable_fingerprint(nested_body)
    ):
        raise PermissionError("nested environment inventory is invalid")
    if spec.get("execution_kind") == ACTUAL_EXECUTION_KIND:
        _validated_bound_environment_contract(spec, payloads=payloads)
    result["environment_inventory_fingerprint"] = nested_fingerprint
    manager = inventory.get("manager")
    if not isinstance(manager, Mapping) or not isinstance(
        manager.get("endpoint"), Mapping
    ):
        raise PermissionError("environment inventory manager endpoint is absent")
    result["manager_endpoint_identity_fingerprint"] = stable_fingerprint(
        dict(manager["endpoint"])
    )
    result["gpu_lease_policy_fingerprint"] = stable_fingerprint(
        {
            "selected_gpu_uuid": environment["selected_gpu_uuid"],
            "selected_gpu_pci_bus_id": environment[
                "selected_gpu_pci_bus_id"
            ],
            "selected_gpu_minor_number": environment[
                "selected_gpu_minor_number"
            ],
            "gpu_lease_path": environment["gpu_lease_path"],
            "gpu_lease_tombstone_path": environment[
                "gpu_lease_tombstone_path"
            ],
            "cooperative_only": True,
        }
    )
    result["selected_gpu_uuid"] = environment["selected_gpu_uuid"]
    result["selected_gpu_pci_bus_id"] = environment[
        "selected_gpu_pci_bus_id"
    ]
    return result


def _load_runtime_environment_module(spec: Mapping[str, object]) -> object:
    bindings = spec["source_bindings"]
    if not isinstance(bindings, Mapping):
        raise AssertionError("validated source bindings changed")
    path = Path(__file__).with_name("cure_lite_v24_runtime_environment.py")
    expected = bindings["runtime_environment_file_sha256"]
    if not _is_sha256(expected) or file_sha256(path) != expected:
        raise PermissionError("runtime environment source binding changed")
    module_name = f"cure_lite_v24_runtime_environment_{expected[:16]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load bound runtime environment module")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _environment_payload(
    spec: Mapping[str, object],
    prefix: str,
) -> dict[str, object]:
    environment = spec["environment"]
    if not isinstance(environment, Mapping):
        raise AssertionError("validated environment contract changed")
    return _read_canonical_json(
        str(environment[f"{prefix}_path"]),
        name=f"environment {prefix}",
    )


def _strict_bound_environment_payload(
    spec: Mapping[str, object],
    *,
    prefix: str,
    name: str,
    fingerprint_field: str,
    supplied_payloads: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    """Re-read one spec-bound environment root before semantic validation."""

    environment = spec["environment"]
    if not isinstance(environment, Mapping):
        raise AssertionError("validated environment contract changed")
    path = _canonical_regular_file(
        str(environment[f"{prefix}_path"]),
        name=name,
    )
    payload = _read_canonical_json(path, name=name)
    if (
        sealed_file_sha256(payload)
        != environment[f"{prefix}_file_sha256"]
    ):
        raise PermissionError(f"{name} sealed file binding changed")
    body = dict(payload)
    fingerprint = body.pop(fingerprint_field, None)
    if not _is_sha256(fingerprint) or fingerprint != stable_fingerprint(body):
        raise PermissionError(f"{name} fingerprint is invalid")
    if supplied_payloads is not None:
        supplied = supplied_payloads.get(prefix)
        if (
            not isinstance(supplied, Mapping)
            or dict(supplied) != payload
        ):
            raise PermissionError(f"{name} changed between sealed reads")
    return payload


def _validate_closed_environment_audit_receipt(
    module: object,
    receipt: Mapping[str, object],
    *,
    expected_passed: bool,
) -> dict[str, object]:
    """Validate the exact audit-only wrapper and nested inventory schemas."""

    payload = json.loads(canonical_json(dict(receipt)))
    body = dict(payload)
    fingerprint = body.pop("receipt_fingerprint", None)
    inventory = body.get("inventory")
    if (
        set(payload) != _ENVIRONMENT_AUDIT_RECEIPT_KEYS
        or payload.get("schema_version") != module.ENVIRONMENT_RECEIPT_SCHEMA
        or payload.get("command") != "audit-only"
        or payload.get("passed") is not expected_passed
        or payload.get("error_type") is not None
        or payload.get("error_message") is not None
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
        or not isinstance(inventory, Mapping)
        or any(
            payload.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError("environment audit receipt closed schema changed")
    materialized_inventory = json.loads(canonical_json(dict(inventory)))
    nested_body = dict(materialized_inventory)
    nested_fingerprint = nested_body.pop("inventory_fingerprint", None)
    if (
        set(materialized_inventory) != _ENVIRONMENT_INVENTORY_KEYS
        or materialized_inventory.get("schema_version")
        != module.ENVIRONMENT_INVENTORY_SCHEMA
        or materialized_inventory.get("passed") is not expected_passed
        or not _is_sha256(nested_fingerprint)
        or nested_fingerprint != stable_fingerprint(nested_body)
        or any(
            materialized_inventory.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError(
            "nested environment inventory closed schema changed"
        )
    manager = materialized_inventory.get("manager")
    unit_scope = materialized_inventory.get("unit_scope")
    gpu = materialized_inventory.get("gpu_snapshot")
    if (
        not isinstance(manager, Mapping)
        or set(manager) != _ENVIRONMENT_MANAGER_KEYS
        or not isinstance(manager.get("identity"), Mapping)
        or set(manager["identity"]) != _ENVIRONMENT_MANAGER_IDENTITY_KEYS
        or not isinstance(manager.get("endpoint"), Mapping)
        or set(manager["endpoint"]) != _ENVIRONMENT_MANAGER_ENDPOINT_KEYS
        or not isinstance(unit_scope, Mapping)
        or set(unit_scope) != _ENVIRONMENT_UNIT_SCOPE_KEYS
        or not isinstance(unit_scope.get("shadows"), Mapping)
        or not isinstance(gpu, Mapping)
        or set(gpu) != _ENVIRONMENT_GPU_SNAPSHOT_KEYS
    ):
        raise PermissionError("environment inventory component schema changed")
    expected_units = (
        ([] if unit_scope.get("target_unit_id") is None else [
            unit_scope["target_unit_id"]
        ])
        + list(unit_scope.get("conflict_unit_ids", ()))
        + list(unit_scope.get("dependency_unit_ids", ()))
    )
    shadows = dict(unit_scope["shadows"])
    if (
        len(expected_units) != len(set(expected_units))
        or set(shadows) != set(expected_units)
        or any(
            not isinstance(shadow, Mapping)
            or set(shadow) != _ENVIRONMENT_UNIT_SHADOW_KEYS
            for shadow in shadows.values()
        )
    ):
        raise PermissionError("environment unit shadow closed schema changed")
    gpu_body = dict(gpu)
    gpu_fingerprint = gpu_body.pop("snapshot_fingerprint", None)
    devices = gpu.get("devices")
    first_apps = gpu.get("first_apps")
    second_apps = gpu.get("second_apps")
    process_rows = gpu.get("process_unit_mapping")
    if (
        gpu.get("schema_version") != module.GPU_DOUBLE_SNAPSHOT_SCHEMA
        or not _is_sha256(gpu_fingerprint)
        or gpu_fingerprint != stable_fingerprint(gpu_body)
        or not isinstance(devices, list)
        or any(
            not isinstance(row, Mapping)
            or set(row) != _ENVIRONMENT_GPU_DEVICE_KEYS
            for row in devices
        )
        or not isinstance(first_apps, list)
        or not isinstance(second_apps, list)
        or any(
            not isinstance(row, Mapping)
            or set(row) != _ENVIRONMENT_GPU_APP_KEYS
            for row in first_apps + second_apps
        )
        or not isinstance(process_rows, list)
        or any(
            not isinstance(row, Mapping)
            or set(row) != _ENVIRONMENT_GPU_PROCESS_UNIT_KEYS
            for row in process_rows
        )
    ):
        raise PermissionError("environment GPU snapshot closed schema changed")
    endpoint = dict(manager["endpoint"])
    identity = dict(manager["identity"])
    expected_binding = {
        "inventory_fingerprint": nested_fingerprint,
        "boot_id": materialized_inventory["boot_id"],
        "runtime_directory": endpoint["runtime_directory"],
        "runtime_directory_device": endpoint[
            "runtime_directory_device"
        ],
        "runtime_directory_inode": endpoint[
            "runtime_directory_inode"
        ],
        "manager_identity": identity,
    }
    if (
        not isinstance(payload.get("environment_binding"), Mapping)
        or set(payload["environment_binding"]) != _ENVIRONMENT_BINDING_KEYS
        or not _deep_exact_equal(
            payload["environment_binding"],
            expected_binding,
        )
    ):
        raise PermissionError("environment audit binding changed")
    return materialized_inventory


def _contract_from_stability_receipt(
    module: object,
    stability: Mapping[str, object],
) -> object:
    try:
        value = dict(stability["contract"])
        for field in (
            "conflict_unit_ids",
            "dependency_unit_ids",
            "allowed_failed_unit_ids",
            "expected_failed_unit_ids",
            "allowed_unit_ids",
            "allowed_manager_states",
        ):
            value[field] = tuple(value[field])
        value["cleanup_nrestarts_baseline"] = tuple(
            (str(item[0]), str(item[1]))
            for item in value["cleanup_nrestarts_baseline"]
        )
        value["activation_guard"] = dict(value["activation_guard"])
        contract = module.EnvironmentAuditContract(**value)
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError(
            "stability environment contract is malformed"
        ) from error
    contract = module.validate_environment_audit_contract(contract)
    if stability["contract"] != json.loads(
        canonical_json(module.asdict(contract))
    ):
        raise PermissionError("stability environment contract changed")
    return contract


def _validate_final_stability_inventory_binding(
    stability: Mapping[str, object],
    inventory: Mapping[str, object],
) -> dict[str, object]:
    samples = stability.get("samples")
    sample_count = stability.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not isinstance(samples, list)
        or sample_count != len(samples)
        or not samples
        or not isinstance(samples[-1], Mapping)
        or not isinstance(samples[-1].get("inventory"), Mapping)
        or samples[-1].get("passed") is not True
        or samples[-1].get("blockers") != []
        or not _deep_exact_equal(
            inventory,
            samples[-1]["inventory"],
        )
    ):
        raise PermissionError(
            "postcleanup inventory is not the exact final stability sample"
        )
    return dict(samples[-1])


def _validated_bound_environment_contract(
    spec: Mapping[str, object],
    payloads: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Close and replay the exact sealed environment chain for actual D_R."""

    module = _load_runtime_environment_module(spec)
    environment = spec["environment"]
    runtime = spec["runtime"]
    child = spec["child"]
    bindings = spec["source_bindings"]
    if not all(
        isinstance(value, Mapping)
        for value in (environment, runtime, child, bindings)
    ):
        raise AssertionError("validated actual runtime spec changed")
    policy = _strict_bound_environment_payload(
        spec,
        prefix="policy",
        name="environment policy",
        fingerprint_field="policy_fingerprint",
        supplied_payloads=payloads,
    )
    cleanup = _strict_bound_environment_payload(
        spec,
        prefix="cleanup_receipt",
        name="environment cleanup receipt",
        fingerprint_field="cleanup_receipt_fingerprint",
        supplied_payloads=payloads,
    )
    stability = _strict_bound_environment_payload(
        spec,
        prefix="stability_receipt",
        name="environment stability receipt",
        fingerprint_field="stability_receipt_fingerprint",
        supplied_payloads=payloads,
    )
    postcleanup = _strict_bound_environment_payload(
        spec,
        prefix="inventory",
        name="postcleanup environment receipt",
        fingerprint_field="receipt_fingerprint",
        supplied_payloads=payloads,
    )

    module.validate_environment_policy(policy)
    module.validate_environment_stability_receipt(stability)
    if (
        policy.get("toolchain") != module.current_runtime_toolchain_binding()
        or policy.get("candidate") != spec.get("candidate")
        or bindings.get("runtime_environment_file_sha256")
        != policy["toolchain"]["runtime_environment"]["file_sha256"]
        or stability.get("receipt_kind") != "sampled"
        or stability.get("passed") is not True
        or stability.get("blockers") != []
        or any(
            value.get(field) is not False
            for value in (policy, cleanup, stability, postcleanup)
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError("bound environment evidence semantics changed")

    contract = _contract_from_stability_receipt(module, stability)
    policy_body = dict(policy)
    policy_body.pop("policy_fingerprint")
    rebuilt_policy = module.build_environment_policy(
        contract,
        precleanup_root_binding=policy["precleanup_root"],
        cleanup_root_binding=policy["cleanup_root"],
        toolchain_binding=policy["toolchain"],
        minimum_sample_count=policy["sampling"]["minimum_sample_count"],
        sample_interval_seconds=policy["sampling"][
            "sample_interval_seconds"
        ],
    )
    rebuilt_body = dict(rebuilt_policy)
    rebuilt_body.pop("policy_fingerprint")
    rebuilt_body["created_at_utc"] = policy_body["created_at_utc"]
    if rebuilt_body != policy_body:
        raise PermissionError(
            "policy and stability environment contracts diverged"
        )

    roots = stability.get("root_evidence")
    if not isinstance(roots, Mapping):
        raise PermissionError("stability root evidence is absent")
    policy_root = dict(roots.get("policy", {}))
    precleanup_root = dict(policy["precleanup_root"])
    cleanup_root = dict(policy["cleanup_root"])
    if (
        dict(roots.get("precleanup_inventory_receipt", {}))
        != precleanup_root
        or dict(roots.get("cleanup_receipt", {})) != cleanup_root
        or policy_root.get("path") != environment["policy_path"]
        or policy_root.get("file_sha256")
        != environment["policy_file_sha256"]
        or cleanup_root.get("path") != environment["cleanup_receipt_path"]
        or cleanup_root.get("file_sha256")
        != environment["cleanup_receipt_file_sha256"]
        or stability.get("stability_receipt_fingerprint")
        is None
    ):
        raise PermissionError("stability root/spec lineage changed")
    _, observed_policy_root = module.load_sealed_receipt_with_evidence(
        environment["policy_path"],
        fingerprint_field="policy_fingerprint",
    )
    precleanup_receipt, observed_precleanup_root = (
        module.load_sealed_receipt_with_evidence(
            precleanup_root["path"],
            fingerprint_field="receipt_fingerprint",
        )
    )
    precleanup_inventory = precleanup_receipt.get("inventory")
    if not isinstance(precleanup_inventory, Mapping):
        raise PermissionError("precleanup inventory root is absent")
    observed_precleanup_root["inventory_fingerprint"] = (
        precleanup_inventory.get("inventory_fingerprint")
    )
    _, observed_cleanup_root = module.load_sealed_receipt_with_evidence(
        environment["cleanup_receipt_path"],
        fingerprint_field="cleanup_receipt_fingerprint",
    )
    if (
        observed_policy_root != policy_root
        or observed_precleanup_root != precleanup_root
        or observed_cleanup_root != cleanup_root
    ):
        raise PermissionError("sealed stability root identity changed")
    _validate_closed_environment_audit_receipt(
        module,
        precleanup_receipt,
        expected_passed=False,
    )

    cleanup_semantics = module.validate_cleanup_receipt_for_environment(
        cleanup,
        uid=contract.uid,
        conflict_unit_ids=contract.conflict_unit_ids,
    )
    if cleanup_semantics["partial_lineage"] is not None:
        module.verify_partial_lineage_roots(
            cleanup_semantics["partial_lineage"]
        )
    expected_manager_generation = {
        "boot_id": contract.boot_id,
        "endpoint": {
            "uid": contract.uid,
            "runtime_directory": contract.runtime_directory,
            "runtime_directory_device": contract.runtime_directory_device,
            "runtime_directory_inode": contract.runtime_directory_inode,
            "bus_path": contract.bus_path,
            "bus_device": contract.bus_device,
            "bus_inode": contract.bus_inode,
        },
        "identity": {
            "pid": contract.manager_pid,
            "starttime_ticks": contract.manager_starttime_ticks,
            "uid": contract.uid,
            "control_group": contract.manager_control_group,
        },
    }
    if (
        cleanup_semantics["cleanup_mode"] != contract.cleanup_mode
        or cleanup_semantics["quiescence_mode"] != contract.quiescence_mode
        or cleanup_semantics["cleanup_nrestarts_baseline"]
        != contract.cleanup_nrestarts_baseline
        or cleanup_semantics["activation_guard"]
        != contract.activation_guard
        or cleanup.get("boot_id") != contract.boot_id
        or cleanup.get("manager_generation")
        != expected_manager_generation
    ):
        raise PermissionError("cleanup and stability contracts diverged")

    guard_observation: dict[str, object] | None = None
    if contract.cleanup_mode == module.RECOVERY_CLEANUP_MODE:
        guard_observation = module.inspect_recovery_activation_guard(
            contract.activation_guard
        )
        if guard_observation != {
            **contract.activation_guard,
            "file_type": "symlink",
        }:
            raise PermissionError("recovery activation guard changed")

    inventory = _validate_closed_environment_audit_receipt(
        module,
        postcleanup,
        expected_passed=True,
    )
    _validate_final_stability_inventory_binding(stability, inventory)
    if inventory.get("blockers") != []:
        raise PermissionError("postcleanup inventory still has blockers")
    expected_collector_arguments = {
        "selected_gpu_index": contract.selected_gpu_index,
        "allowed_unit_ids": contract.allowed_unit_ids,
        "target_unit_id": contract.target_unit_id,
        "conflict_unit_ids": contract.conflict_unit_ids,
        "dependency_unit_ids": contract.dependency_unit_ids,
        "allowed_failed_unit_ids": contract.allowed_failed_unit_ids,
        "allowed_manager_states": contract.allowed_manager_states,
        "require_target_ready": contract.require_target_ready,
        "strict_all_gpu_consumers": contract.strict_all_gpu_consumers,
        "conflict_quiescence_mode": contract.quiescence_mode,
    }

    def replay_inventory(**kwargs: object) -> dict[str, object]:
        if kwargs != expected_collector_arguments:
            raise PermissionError("postcleanup replay scope changed")
        return json.loads(canonical_json(inventory))

    def replay_guard(
        expected_guard: Mapping[str, object],
    ) -> dict[str, object]:
        if (
            guard_observation is None
            or dict(expected_guard) != contract.activation_guard
        ):
            raise PermissionError("postcleanup recovery guard scope changed")
        return dict(guard_observation)

    replayed = module.audit_environment_once(
        contract,
        inventory_collector=replay_inventory,
        activation_guard_reader=replay_guard,
    )
    module.validate_environment_single_audit(
        replayed,
        contract=contract,
    )
    if (
        replayed.get("passed") is not True
        or replayed.get("blockers") != []
        or replayed.get("inventory") != inventory
        or any(
            replayed.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError("sealed postcleanup replay failed")

    systemd = runtime.get("systemd")
    child_environment = child.get("environment")
    if (
        contract.uid != os.getuid()
        or not isinstance(systemd, Mapping)
        or contract.target_unit_id != systemd.get("unit_name")
        or environment.get("selected_gpu_uuid")
        != contract.selected_gpu_uuid
        or environment.get("selected_gpu_pci_bus_id")
        != contract.selected_gpu_pci_bus_id
        or environment.get("selected_gpu_minor_number")
        != contract.selected_gpu_minor_number
        or not isinstance(child_environment, Mapping)
        or child_environment.get("CUDA_VISIBLE_DEVICES")
        != contract.selected_gpu_uuid
    ):
        raise PermissionError("runtime spec and environment identity diverged")
    endpoint = inventory["manager"]["endpoint"]
    return {
        "module": module,
        "contract": contract,
        "policy": policy,
        "cleanup_receipt": cleanup,
        "stability_receipt": stability,
        "postcleanup_receipt": postcleanup,
        "postcleanup_inventory": inventory,
        "manager_endpoint_identity_fingerprint": stable_fingerprint(
            dict(endpoint)
        ),
        "recovery_activation_guard_observation": guard_observation,
    }


def _verify_live_environment(
    spec: Mapping[str, object],
    *,
    phase: str,
) -> dict[str, object]:
    if phase not in {"prelease", "postlease", "child_prespawn", "finalizer"}:
        raise ValueError("runtime environment audit phase is invalid")
    validated = _validated_bound_environment_contract(spec)
    module = validated["module"]
    sealed_contract = validated["contract"]
    contract = module.replace(
        sealed_contract,
        require_target_ready=(phase in {"prelease", "postlease"}),
    )
    contract = module.validate_environment_audit_contract(contract)
    live_audit = module.audit_environment_once(contract)
    module.validate_environment_single_audit(
        live_audit,
        contract=contract,
    )
    live = live_audit.get("inventory")
    if (
        live_audit.get("passed") is not True
        or live_audit.get("blockers") != []
        or any(
            live_audit.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or not isinstance(live, Mapping)
        or live.get("passed") is not True
        or live.get("blockers") != []
    ):
        raise PermissionError("live runtime environment audit failed")
    return _seal(
        {
            "schema_version": "cure-lite-v24-runtime-live-audit-v1",
            "phase": phase,
            "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
            "inventory": live,
            "inventory_fingerprint": live["inventory_fingerprint"],
            "runtime_environment_audit_valid": True,
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        },
        fingerprint_field="environment_audit_fingerprint",
    )


def _planned_attempt_commit_fingerprint(
    spec: Mapping[str, object],
    authorization: Mapping[str, object],
) -> str:
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-v24-planned-attempt-commit-v1",
            "candidate": spec["candidate"],
            "stage_id": spec["stage_id"],
            "attempt_id": spec["attempt_id"],
            "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
            "authorization_fingerprint": authorization[
                "authorization_fingerprint"
            ],
            "environment_bindings": _environment_evidence_bindings(spec),
            "source_bindings": spec["source_bindings"],
            "unit_fragment_file_sha256": spec["runtime"]["systemd"][
                "unit_fragment_file_sha256"
            ],
            "launch_limit": 1,
            "automatic_retry_allowed": False,
            "resume_allowed": False,
        }
    )


def _acquire_external_gpu_lease(
    spec: Mapping[str, object],
    authorization: Mapping[str, object],
    *,
    planned_attempt_commit_fingerprint: str,
) -> object:
    module = _load_runtime_environment_module(spec)
    environment = spec["environment"]
    if not isinstance(environment, Mapping):
        raise AssertionError("validated environment contract changed")
    body = {
        "schema_version": module.GPU_LEASE_SCHEMA,
        "created_at_utc": _utc_now(),
        "boot_id": _boot_id(),
        "gpu_uuid": environment["selected_gpu_uuid"],
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
        "attempt_id": spec["attempt_id"],
        "authorization_fingerprint": authorization[
            "authorization_fingerprint"
        ],
        "planned_attempt_commit_fingerprint": planned_attempt_commit_fingerprint,
        "committer_pid": os.getpid(),
        "committer_starttime": _proc_starttime_ticks(os.getpid()),
    }
    return module.acquire_gpu_lease(environment["gpu_lease_path"], body)


def _external_gpu_lease_evidence(handle: object) -> dict[str, object]:
    raw, metadata, parent_metadata = _read_stable_gpu_lease_descriptor(
        handle.descriptor,
        handle.parent_descriptor,
        handle.path,
        expected_uid=os.getuid(),
        expected_mode=0o600,
        expected_nlink=1,
        name="new GPU lease",
    )
    expected = (canonical_json(handle.payload) + "\n").encode("utf-8")
    lease_body = dict(handle.payload)
    lease_fingerprint = lease_body.pop("lease_fingerprint", None)
    if (
        raw != expected
        or not _is_sha256(lease_fingerprint)
        or lease_fingerprint != stable_fingerprint(lease_body)
        or not _is_positive_int(handle.device)
        or not _is_positive_int(handle.inode)
        or not _is_positive_int(handle.parent_device)
        or not _is_positive_int(handle.parent_inode)
        or metadata.st_dev != handle.device
        or metadata.st_ino != handle.inode
        or parent_metadata.st_dev != handle.parent_device
        or parent_metadata.st_ino != handle.parent_inode
    ):
        raise PermissionError("new GPU lease payload binding is invalid")
    return {
        "handle": handle,
        "gpu_lease_fingerprint": lease_fingerprint,
        "gpu_lease_file_sha256": hashlib.sha256(raw).hexdigest(),
        "gpu_lease_device": metadata.st_dev,
        "gpu_lease_inode": metadata.st_ino,
        "gpu_lease_parent_device": parent_metadata.st_dev,
        "gpu_lease_parent_inode": parent_metadata.st_ino,
        "gpu_lease_valid": True,
    }


def _verify_active_gpu_lease(
    spec: Mapping[str, object],
    attempt_commit: Mapping[str, object],
) -> dict[str, object]:
    module = _load_runtime_environment_module(spec)
    environment = spec["environment"]
    if not isinstance(environment, Mapping):
        raise AssertionError("validated environment contract changed")
    if not _is_sha256(attempt_commit.get("authorization_fingerprint")):
        raise PermissionError("active GPU lease commit identity is invalid")
    expected_planned_fingerprint = _planned_attempt_commit_fingerprint(
        spec,
        attempt_commit,
    )
    if (
        attempt_commit.get("boot_id") != _boot_id()
        or not _is_positive_int(attempt_commit.get("committer_pid"))
        or not _is_positive_int(
            attempt_commit.get("committer_proc_starttime_ticks")
        )
        or attempt_commit.get("planned_attempt_commit_fingerprint")
        != expected_planned_fingerprint
    ):
        raise PermissionError("active GPU lease commit identity is invalid")
    path = Path(str(environment["gpu_lease_path"])).absolute()
    parent = path.parent
    if parent.resolve(strict=True) != parent:
        raise PermissionError("active GPU lease parent is not canonical")
    parent_before = parent.lstat()
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    active_error: BaseException | None = None
    try:
        parent_opened = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_opened.st_mode)
            or (parent_opened.st_dev, parent_opened.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
        ):
            raise PermissionError(
                "active GPU lease parent changed before open"
            )
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw, metadata, parent_metadata = _read_stable_gpu_lease_descriptor(
            descriptor,
            parent_descriptor,
            path,
            expected_uid=os.getuid(),
            expected_mode=0o600,
            expected_nlink=1,
            name="active GPU lease",
        )
        payload = _decode_canonical_json(raw, name="active GPU lease")
        body = dict(payload)
        fingerprint = body.pop("lease_fingerprint", None)
        file_digest = hashlib.sha256(raw).hexdigest()
        if (
            set(payload) != _GPU_LEASE_PAYLOAD_KEYS
            or set(body) != _GPU_LEASE_BODY_KEYS
            or not isinstance(payload.get("created_at_utc"), str)
            or not payload["created_at_utc"]
            or not _is_sha256(fingerprint)
            or fingerprint != stable_fingerprint(body)
            or not _is_sha256(
                attempt_commit.get("gpu_lease_file_sha256")
            )
            or any(
                not _is_positive_int(attempt_commit.get(field))
                for field in (
                    "gpu_lease_device",
                    "gpu_lease_inode",
                    "gpu_lease_parent_device",
                    "gpu_lease_parent_inode",
                )
            )
            or payload.get("schema_version") != module.GPU_LEASE_SCHEMA
            or payload.get("boot_id") != attempt_commit.get("boot_id")
            or payload.get("gpu_uuid") != environment["selected_gpu_uuid"]
            or payload.get("runtime_spec_fingerprint")
            != spec["runtime_spec_fingerprint"]
            or payload.get("attempt_id") != spec["attempt_id"]
            or payload.get("authorization_fingerprint")
            != attempt_commit.get("authorization_fingerprint")
            or payload.get("planned_attempt_commit_fingerprint")
            != expected_planned_fingerprint
            or payload.get("committer_pid")
            != attempt_commit.get("committer_pid")
            or payload.get("committer_starttime")
            != attempt_commit.get("committer_proc_starttime_ticks")
            or not _is_positive_int(payload.get("committer_pid"))
            or not _is_positive_int(payload.get("committer_starttime"))
            or fingerprint != attempt_commit.get("gpu_lease_fingerprint")
            or file_digest != attempt_commit.get("gpu_lease_file_sha256")
            or metadata.st_dev != attempt_commit.get("gpu_lease_device")
            or metadata.st_ino != attempt_commit.get("gpu_lease_inode")
            or parent_metadata.st_dev
            != attempt_commit.get("gpu_lease_parent_device")
            or parent_metadata.st_ino
            != attempt_commit.get("gpu_lease_parent_inode")
        ):
            raise PermissionError("active GPU lease identity is invalid")
        handle = module.GPULeaseHandle(
            path=path,
            descriptor=descriptor,
            parent_descriptor=parent_descriptor,
            payload=payload,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            parent_device=parent_metadata.st_dev,
            parent_inode=parent_metadata.st_ino,
        )
        descriptor = -1
        parent_descriptor = -1
        return {
            "handle": handle,
            "gpu_lease_fingerprint": fingerprint,
            "gpu_lease_file_sha256": file_digest,
            "gpu_lease_device": metadata.st_dev,
            "gpu_lease_inode": metadata.st_ino,
            "gpu_lease_parent_device": parent_metadata.st_dev,
            "gpu_lease_parent_inode": parent_metadata.st_ino,
            "gpu_lease_valid": True,
        }
    except BaseException as error:
        active_error = error
        raise
    finally:
        close_error = _close_descriptors_best_effort(
            descriptor,
            parent_descriptor,
        )
        if active_error is None and close_error is not None:
            raise close_error


def _release_external_gpu_lease(
    spec: Mapping[str, object],
    lease_evidence: Mapping[str, object],
    *,
    release_kind: str,
    attempt_consumed: bool,
    evidence_fingerprint: str,
) -> dict[str, object]:
    module = _load_runtime_environment_module(spec)
    environment = spec["environment"]
    artifacts = spec["artifacts"]
    if not isinstance(environment, Mapping) or not isinstance(artifacts, Mapping):
        raise AssertionError("validated GPU lease contract changed")
    released = module.release_gpu_lease_to_tombstone(
        lease_evidence["handle"],
        tombstone_path=environment["gpu_lease_tombstone_path"],
        release_receipt_path=artifacts["gpu_lease_release_receipt"],
        release_kind=release_kind,
        attempt_consumed=attempt_consumed,
        evidence_fingerprint=evidence_fingerprint,
    )
    return _validate_gpu_lease_release_evidence(
        spec,
        lease_evidence,
        released,
        release_kind=release_kind,
        attempt_consumed=attempt_consumed,
        evidence_fingerprint=evidence_fingerprint,
    )


def _validate_gpu_lease_release_evidence(
    spec: Mapping[str, object],
    lease_evidence: Mapping[str, object],
    release: Mapping[str, object],
    *,
    release_kind: str,
    attempt_consumed: bool,
    evidence_fingerprint: str,
) -> dict[str, object]:
    """Close the module-validated release receipt against this exact lease."""

    module = _load_runtime_environment_module(spec)
    environment = spec["environment"]
    if not isinstance(environment, Mapping):
        raise AssertionError("validated GPU lease contract changed")
    try:
        validated = module.validate_gpu_lease_release_receipt(release)
    except (TypeError, ValueError) as error:
        raise PermissionError(
            "GPU lease release receipt schema is invalid"
        ) from error
    if (
        lease_evidence.get("gpu_lease_valid") is not True
        or validated.get("release_kind") != release_kind
        or validated.get("attempt_consumed") is not attempt_consumed
        or validated.get("lease_fingerprint")
        != lease_evidence.get("gpu_lease_fingerprint")
        or validated.get("gpu_uuid")
        != environment["selected_gpu_uuid"]
        or validated.get("attempt_id") != spec["attempt_id"]
        or validated.get("evidence_fingerprint") != evidence_fingerprint
        or validated.get("active_lease_path")
        != str(environment["gpu_lease_path"])
        or validated.get("tombstone_path")
        != str(environment["gpu_lease_tombstone_path"])
        or validated.get("tombstone_file_sha256")
        != lease_evidence.get("gpu_lease_file_sha256")
        or validated.get("tombstone_device")
        != lease_evidence.get("gpu_lease_device")
        or validated.get("tombstone_inode")
        != lease_evidence.get("gpu_lease_inode")
        or validated.get("lease_parent_device")
        != lease_evidence.get("gpu_lease_parent_device")
        or validated.get("lease_parent_inode")
        != lease_evidence.get("gpu_lease_parent_inode")
    ):
        raise PermissionError("GPU lease release receipt identity is invalid")
    return validated


def _validate_spec_structure(
    payload: dict[str, object],
    *,
    loaded_spec_path: Path,
) -> None:
    _require_exact_keys(payload, _SPEC_KEYS, name="runtime spec")
    body = dict(payload)
    fingerprint = body.pop("runtime_spec_fingerprint")
    if (
        payload.get("schema_version") != RUNTIME_SPEC_SCHEMA
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
    ):
        raise ValueError("runtime spec identity or fingerprint changed")
    execution_kind = payload["execution_kind"]
    if execution_kind not in {
        ACTUAL_EXECUTION_KIND,
        DUMMY_EXECUTION_KIND,
        SYSTEMD_INTEGRATION_DUMMY_KIND,
    }:
        raise ValueError("runtime spec execution_kind is invalid")
    if execution_kind == ACTUAL_EXECUTION_KIND:
        _validate_environment_structure(payload)
    elif payload["environment"] is not None:
        raise ValueError("non-actual execution cannot carry environment evidence")
    for field in ("candidate", "stage_id", "attempt_id"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"runtime spec {field} must be nonempty text")
    for field in ("attempt_ordinal", "prior_attempt_count"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"runtime spec {field} must be nonnegative")

    authorization = payload["authorization"]
    scientific_preaccess = payload["scientific_preaccess"]
    if execution_kind == ACTUAL_EXECUTION_KIND:
        reference = _require_exact_keys(
            authorization,
            _AUTH_REFERENCE_KEYS,
            name="runtime launch authorization reference",
        )
        _absolute_path(reference["path"], name="authorization.path")
        if (
            reference["required_schema"] != RUNTIME_LAUNCH_AUTHORIZATION_SCHEMA
            or reference["path"]
            != _ACTUAL_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ):
            raise ValueError("runtime launch authorization schema is not exact")
        preaccess = _require_exact_keys(
            scientific_preaccess,
            _SCIENTIFIC_PREACCESS_KEYS,
            name="scientific preaccess contract",
        )
        for field in ("authorization_path", "access_audit_path"):
            _absolute_path(preaccess[field], name=f"scientific_preaccess.{field}")
        for field in (
            "authorization_file_sha256",
            "authorization_fingerprint",
            "access_audit_file_sha256",
            "access_audit_fingerprint",
            "source_closure_fingerprint_103",
        ):
            if not _is_sha256(preaccess[field]):
                raise ValueError(f"scientific_preaccess.{field} is malformed")
        if (
            not isinstance(preaccess["authorization_required_schema"], str)
            or not preaccess["authorization_required_schema"]
            or preaccess["authorization_required_schema"]
            == RUNTIME_LAUNCH_AUTHORIZATION_SCHEMA
            or not isinstance(preaccess["access_audit_required_schema"], str)
            or not preaccess["access_audit_required_schema"]
            or preaccess["access_audit_required_schema"]
            == RUNTIME_LAUNCH_AUTHORIZATION_SCHEMA
            or preaccess["authorization_path"]
            != _ACTUAL_SCIENTIFIC_AUTHORIZATION_PATH
            or preaccess["access_audit_path"]
            != _ACTUAL_SCIENTIFIC_ACCESS_AUDIT_PATH
            or preaccess["authorization_required_schema"]
            != "cure-lite-v24-D_R-structural-r2-authorization-v1"
            or preaccess["access_audit_required_schema"]
            != "cure-lite-v24-split-access-audit-r2-v1"
            or payload["candidate"] != _ACTUAL_CANDIDATE
            or payload["stage_id"] != _ACTUAL_STAGE_ID
            or payload["attempt_id"] != _ACTUAL_ATTEMPT_ID
            or payload["attempt_ordinal"] != 2
            or payload["prior_attempt_count"] != 1
            or preaccess["source_closure_fingerprint_103"]
            != _ACTUAL_SOURCE_CLOSURE_FINGERPRINT_103
        ):
            raise ValueError("actual r2 identity or preaccess split is not exact")
    elif authorization is not None or scientific_preaccess is not None:
        raise ValueError("non-actual execution cannot carry authorization")
    if execution_kind == SYSTEMD_INTEGRATION_DUMMY_KIND:
        stage_prefix = "systemd_integration_dummy_"
        stage_id = str(payload["stage_id"])
        if not stage_id.startswith(stage_prefix):
            raise ValueError("systemd integration stage identity is malformed")
        identity = build_systemd_integration_identity(
            stage_id[len(stage_prefix) :]
        )
        if (
            any(
                payload[key] != identity[key]
                for key in ("candidate", "stage_id", "attempt_id")
            )
            or payload["attempt_ordinal"] != 0
            or payload["prior_attempt_count"] != 0
        ):
            raise ValueError("systemd integration dummy identity is not exact")

    child = _require_exact_keys(
        payload["child"],
        _CHILD_KEYS,
        name="child contract",
    )
    argv = child["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(
            not isinstance(argument, str)
            or not argument
            or "\x00" in argument
            for argument in argv
        )
        or not Path(argv[0]).is_absolute()
        or child["argv_fingerprint"] != stable_fingerprint(argv)
    ):
        raise ValueError("child argv is not an exact absolute argv vector")
    _absolute_path(child["cwd"], name="child.cwd")
    environment = child["environment"]
    inherited = child["inherit_environment"]
    if (
        not isinstance(environment, dict)
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or "\x00" in key
            or "\x00" in value
            for key, value in environment.items()
        )
        or not isinstance(inherited, list)
        or len(inherited) != len(set(inherited))
        or any(
            not isinstance(name, str)
            or not name
            or "\x00" in name
            for name in inherited
        )
    ):
        raise ValueError("child environment contract is malformed")
    if any(
        name.startswith(_RUNTIME_RESERVED_ENV_PREFIX)
        for name in set(environment) | set(inherited)
    ):
        raise ValueError("child contract attempts to forge runtime attestation")
    entrypoint = child["entrypoint_path"]
    if entrypoint is not None:
        _absolute_path(entrypoint, name="child.entrypoint_path")

    artifacts = _require_exact_keys(
        payload["artifacts"],
        _ARTIFACT_KEYS,
        name="artifact contract",
    )
    artifact_paths = {
        key: _absolute_path(value, name=f"artifacts.{key}")
        for key, value in artifacts.items()
    }
    if len(set(artifact_paths.values())) != len(artifact_paths):
        raise ValueError("artifact paths must be distinct")
    root = artifact_paths["root"]
    for key, path in artifact_paths.items():
        if key == "root":
            continue
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("all artifacts must remain under root") from error

    runtime = _require_exact_keys(
        payload["runtime"],
        _RUNTIME_KEYS,
        name="runtime contract",
    )
    if (
        runtime["shell"] is not False
        or runtime["start_new_session"] is not True
        or runtime["launch_limit"] != 1
        or runtime["automatic_retry_allowed"] is not False
        or runtime["resume_allowed"] is not False
        or runtime["restart"] != "no"
    ):
        raise ValueError("runtime is not a shell-free one-launch contract")
    heartbeat_interval = _positive_number(
        runtime["heartbeat_interval_seconds"],
        name="runtime.heartbeat_interval_seconds",
    )
    poll_interval = _positive_number(
        runtime["poll_interval_seconds"],
        name="runtime.poll_interval_seconds",
    )
    _positive_number(
        runtime["termination_grace_seconds"],
        name="runtime.termination_grace_seconds",
    )
    if poll_interval > heartbeat_interval:
        raise ValueError("poll interval cannot exceed heartbeat interval")
    systemd = _require_exact_keys(
        runtime["systemd"],
        _SYSTEMD_KEYS,
        name="systemd contract",
    )
    if (
        not isinstance(systemd["unit_name"], str)
        or not systemd["unit_name"].endswith(".service")
        or systemd["service_type"] != "exec"
        or systemd["kill_mode"] != "mixed"
        or systemd["send_sigkill"] is not True
    ):
        raise ValueError("systemd supervision contract is not exact")
    if (
        execution_kind == ACTUAL_EXECUTION_KIND
        and systemd["unit_name"] != _ACTUAL_UNIT_NAME
    ):
        raise ValueError("actual r2 systemd unit identity is not exact")
    if execution_kind == SYSTEMD_INTEGRATION_DUMMY_KIND:
        scenario_id = str(payload["stage_id"])[
            len("systemd_integration_dummy_") :
        ]
        identity = build_systemd_integration_identity(scenario_id)
        if systemd["unit_name"] != identity["unit_name"]:
            raise ValueError(
                "systemd integration dummy unit identity is not exact"
            )
    _positive_number(
        systemd["timeout_stop_seconds"],
        name="systemd.timeout_stop_seconds",
    )
    ack_timeout = _positive_number(
        systemd["start_ack_timeout_seconds"],
        name="systemd.start_ack_timeout_seconds",
    )
    ack_poll = _positive_number(
        systemd["start_ack_poll_seconds"],
        name="systemd.start_ack_poll_seconds",
    )
    if ack_poll > ack_timeout:
        raise ValueError("start acknowledgement poll exceeds timeout")
    shadow = _require_exact_keys(
        systemd["immutable_shadow_properties"],
        _SYSTEMD_IMMUTABLE_SHADOW_KEYS,
        name="immutable systemd shadow properties",
    )
    if (
        any(not isinstance(value, str) for value in shadow.values())
        or shadow["Type"] != "exec"
        or shadow["Restart"] != "no"
        or shadow["KillMode"] != "mixed"
        or shadow["SendSIGKILL"] != "yes"
        or shadow["DropInPaths"] != ""
        or shadow["Transient"] != "no"
        or shadow["ExitType"] != "main"
        or shadow["RuntimeMaxUSec"] != "infinity"
        or shadow["WatchdogUSec"] != "disabled"
        or shadow["OOMPolicy"] != "kill"
        or shadow["RemainAfterExit"] != "no"
        or shadow["StandardInput"] != "null"
        or shadow["StartLimitBurst"] != "1"
        or shadow["KillSignal"] not in {"15", "SIGTERM"}
        or not shadow["WorkingDirectory"]
        or not shadow["UMask"]
        or not shadow["StandardOutput"]
        or not shadow["StandardError"]
        or not shadow["StartLimitIntervalUSec"]
        or not shadow["TimeoutStopUSec"]
        or not shadow["FragmentPath"]
        or any(
            shadow[name] != _normalize_systemd_shadow_value(name, shadow[name])
            for name in _SYSTEMD_EXEC_KEYS
        )
        or "claim-materialization" not in shadow["ExecCondition"]
        or "verify-runtime-spec" not in shadow["ExecStartPre"]
        or "run-once" not in shadow["ExecStart"]
        or "record-systemd-exit" not in shadow["ExecStopPost"]
        or systemd["immutable_shadow_fingerprint"]
        != stable_fingerprint(shadow)
    ):
        raise ValueError("systemd shadow properties are not exact")
    fragment_sha256 = systemd["unit_fragment_file_sha256"]
    if fragment_sha256 is not None and not _is_sha256(fragment_sha256):
        raise ValueError("systemd unit fragment binding is malformed")
    if (
        execution_kind
        in {ACTUAL_EXECUTION_KIND, SYSTEMD_INTEGRATION_DUMMY_KIND}
        and not _is_sha256(fragment_sha256)
    ):
        raise ValueError("systemd execution unit fragment binding is absent")

    bindings = _require_exact_keys(
        payload["source_bindings"],
        _SOURCE_BINDING_KEYS,
        name="source bindings",
    )
    if not _is_sha256(bindings["supervisor_file_sha256"]):
        raise ValueError("supervisor source binding is malformed")
    for field in (
        "child_entry_file_sha256",
        "prior_attempt_receipt_file_sha256",
        "runtime_environment_file_sha256",
        "r2_adapter_file_sha256",
        "legacy_gate_entrypoint_file_sha256",
        "python_file_sha256",
    ):
        if bindings[field] is not None and not _is_sha256(bindings[field]):
            raise ValueError(f"{field} is malformed")
    if execution_kind == ACTUAL_EXECUTION_KIND:
        adapter_path = _absolute_path(
            bindings["r2_adapter_path"],
            name="r2 adapter path",
        )
        _absolute_path(
            bindings["legacy_gate_entrypoint_path"],
            name="legacy gate entrypoint path",
        )
        python_path = _absolute_path(
            bindings["python_path"],
            name="actual Python path",
        )
        dependency_site_path = _absolute_path(
            bindings["runtime_dependency_site_path"],
            name="actual runtime dependency site path",
        )
        if (
            entrypoint is None
            or not _is_sha256(bindings["child_entry_file_sha256"])
            or not _is_sha256(bindings["runtime_environment_file_sha256"])
            or not _is_sha256(bindings["r2_adapter_file_sha256"])
            or not _is_sha256(
                bindings["legacy_gate_entrypoint_file_sha256"]
            )
            or str(adapter_path) != _ACTUAL_ADAPTER_PATH
            or bindings["legacy_gate_entrypoint_path"]
            != _ACTUAL_LEGACY_ENTRYPOINT_PATH
            or entrypoint != str(adapter_path)
            or bindings["child_entry_file_sha256"]
            != bindings["r2_adapter_file_sha256"]
            or not _is_sha256(
                bindings["prior_attempt_receipt_file_sha256"]
            )
            or str(python_path) != _ACTUAL_PYTHON_PATH
            or not _is_sha256(bindings["python_file_sha256"])
            or isinstance(bindings["python_owner_uid"], bool)
            or bindings["python_owner_uid"] != 0
            or bindings["python_mode"] != 0o755
            or not _is_positive_int(bindings["python_device"])
            or not _is_positive_int(bindings["python_inode"])
            or str(dependency_site_path)
            != _ACTUAL_RUNTIME_DEPENDENCY_SITE_PATH
            or bindings["runtime_dependency_site_device"]
            != _ACTUAL_RUNTIME_DEPENDENCY_SITE_DEVICE
            or bindings["runtime_dependency_site_inode"]
            != _ACTUAL_RUNTIME_DEPENDENCY_SITE_INODE
            or bindings["runtime_dependency_site_owner_uid"]
            != _ACTUAL_RUNTIME_DEPENDENCY_SITE_OWNER_UID
            or bindings["runtime_dependency_site_mode"]
            != _ACTUAL_RUNTIME_DEPENDENCY_SITE_MODE
        ):
            raise ValueError("actual source lineage bindings are incomplete")
    else:
        actual_runtime_binding_fields = {
            "python_path",
            "python_file_sha256",
            "python_device",
            "python_inode",
            "python_owner_uid",
            "python_mode",
            "runtime_dependency_site_path",
            "runtime_dependency_site_device",
            "runtime_dependency_site_inode",
            "runtime_dependency_site_owner_uid",
            "runtime_dependency_site_mode",
        }
        if any(
            bindings[field] is not None
            for field in actual_runtime_binding_fields
        ):
            raise ValueError(
                "non-actual source bindings carry actual runtime identity"
            )
    if execution_kind == ACTUAL_EXECUTION_KIND:
        if child["inherit_environment"] != []:
            raise ValueError(
                "actual runtime cannot inherit unbound environment values"
            )
        expected_actual_argv = [
            _ACTUAL_PYTHON_PATH,
            "-I",
            "-S",
            "-B",
            "-u",
            _ACTUAL_ADAPTER_PATH,
            "real",
            "--execute-real-dr",
            "--device",
            "cuda:0",
            "--runtime-launch-authorization",
            _ACTUAL_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        ]
        expected_actual_environment = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": payload["environment"][
                "selected_gpu_uuid"
            ],
        }
        if (
            entrypoint != _ACTUAL_ADAPTER_PATH
            or argv != expected_actual_argv
            or environment != expected_actual_environment
        ):
            raise ValueError(
                "actual child argv or environment is not exact"
            )
        expected_exec = {
            "ExecCondition": "claim-materialization",
            "ExecStartPre": "verify-runtime-spec",
            "ExecStart": "run-once",
            "ExecStopPost": "record-systemd-exit",
        }
        for directive, mode in expected_exec.items():
            expected_argv = [
                _ACTUAL_PYTHON_PATH,
                "-I",
                "-S",
                "-B",
                "-u",
                str(Path(__file__).resolve()),
                mode,
                "--spec",
                _ACTUAL_SPEC_PATH,
            ]
            if _normalized_systemd_exec_identity(
                shadow[directive]
            ) != {
                "path": _ACTUAL_PYTHON_PATH,
                "argv": expected_argv,
                "ignore_errors": "no",
            }:
                raise ValueError(
                    f"actual systemd {directive} argv is not exact"
                )
        forbidden_environment = {
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONINSPECT",
        }
        if forbidden_environment & set(environment):
            raise ValueError("actual child environment enables Python injection")
    if execution_kind == SYSTEMD_INTEGRATION_DUMMY_KIND:
        if (
            child["inherit_environment"] != []
            or not isinstance(entrypoint, str)
            or Path(entrypoint).name != "cure_lite_v24_dummy_child.py"
            or len(argv) < 3
            or argv[1] != "-I"
            or argv[2] != entrypoint
            or argv.count(entrypoint) != 1
            or not _is_sha256(bindings["child_entry_file_sha256"])
        ):
            raise ValueError(
                "systemd integration dummy child identity is not exact"
            )
        parsed_exec = {
            directive: _normalized_systemd_exec_identity(shadow[directive])
            for directive in _SYSTEMD_EXEC_MODES
        }
        condition_argv = parsed_exec["ExecCondition"]["argv"]
        if not isinstance(condition_argv, list) or len(condition_argv) != 7:
            raise ValueError(
                "systemd integration dummy ExecCondition argv is not exact"
            )
        bound_python = str(argv[0])
        bound_supervisor = str(Path(__file__).resolve())
        bound_spec = condition_argv[-1]
        _absolute_path(
            bound_spec,
            name="systemd integration dummy runtime spec",
        )
        if (
            Path(bound_spec).name != "runtime-spec.json"
            or bound_spec != str(loaded_spec_path)
            or shadow["WorkingDirectory"] != child["cwd"]
        ):
            raise ValueError(
                "systemd integration dummy execution roots are not exact"
            )
        for directive, mode in _SYSTEMD_EXEC_MODES.items():
            expected_argv = [
                bound_python,
                "-I",
                "-u",
                bound_supervisor,
                mode,
                "--spec",
                bound_spec,
            ]
            if parsed_exec[directive] != {
                "path": bound_python,
                "argv": expected_argv,
                "ignore_errors": "no",
            }:
                raise ValueError(
                    f"systemd integration dummy {directive} argv is not exact"
                )


def load_runtime_spec(path: str | Path) -> dict[str, object]:
    loaded_spec_path = Path(path)
    if not loaded_spec_path.is_absolute():
        loaded_spec_path = Path(os.path.abspath(loaded_spec_path))
    payload = _read_canonical_json(
        loaded_spec_path,
        name="runtime supervisor spec",
    )
    _validate_spec_structure(
        payload,
        loaded_spec_path=loaded_spec_path,
    )
    return payload



def _verify_scientific_preaccess_bindings(
    spec: Mapping[str, object],
) -> dict[str, object]:
    contract = spec["scientific_preaccess"]
    if not isinstance(contract, Mapping):
        raise PermissionError("scientific preaccess contract is absent")
    definitions = (
        (
            "authorization",
            "authorization_fingerprint",
            "authorization_required_schema",
        ),
        (
            "access_audit",
            "access_audit_fingerprint",
            "access_audit_required_schema",
        ),
    )
    result: dict[str, object] = {}
    for prefix, fingerprint_field, schema_field in definitions:
        path = _canonical_regular_file(
            str(contract[f"{prefix}_path"]),
            name=f"scientific preaccess {prefix}",
        )
        payload = _read_canonical_json(path, name=f"scientific {prefix}")
        payload_sha256 = sealed_file_sha256(payload)
        if payload_sha256 != contract[f"{prefix}_file_sha256"]:
            raise PermissionError(f"scientific preaccess {prefix} SHA changed")
        body = dict(payload)
        fingerprint = body.pop(fingerprint_field, None)
        if (
            payload.get("schema_version") != contract[schema_field]
            or fingerprint != contract[f"{prefix}_fingerprint"]
            or not _is_sha256(fingerprint)
            or fingerprint != stable_fingerprint(body)
        ):
            raise PermissionError(f"scientific preaccess {prefix} is invalid")
        result[f"scientific_preaccess_{prefix}_path"] = str(path)
        result[f"scientific_preaccess_{prefix}_file_sha256"] = payload_sha256
        result[f"scientific_preaccess_{prefix}_fingerprint"] = fingerprint
    bindings = spec["source_bindings"]
    if not isinstance(bindings, Mapping):
        raise AssertionError("validated source bindings changed")
    result.update(
        {
            "r2_adapter_path": bindings["r2_adapter_path"],
            "r2_adapter_file_sha256": bindings["r2_adapter_file_sha256"],
            "legacy_gate_entrypoint_path": bindings[
                "legacy_gate_entrypoint_path"
            ],
            "legacy_gate_entrypoint_file_sha256": bindings[
                "legacy_gate_entrypoint_file_sha256"
            ],
            "source_closure_fingerprint_103": contract[
                "source_closure_fingerprint_103"
            ],
        }
    )
    return result

def _verify_actual_authorization(
    spec: Mapping[str, object],
    *,
    spec_path: Path,
    require_fresh: bool = True,
) -> dict[str, object]:
    if not isinstance(require_fresh, bool):
        raise TypeError("authorization freshness mode must be boolean")
    reference = spec["authorization"]
    if not isinstance(reference, Mapping):
        raise PermissionError("fresh r2 authorization is absent")
    authorization_path = Path(str(reference["path"]))
    if not authorization_path.exists() or authorization_path.is_symlink():
        raise PermissionError("fresh r2 authorization is absent")
    try:
        source = _canonical_regular_file(
            authorization_path,
            name="fresh r2 authorization",
        )
        source_metadata = source.stat()
        if (
            source_metadata.st_uid != os.getuid()
            or stat.S_IMODE(source_metadata.st_mode) != 0o444
        ):
            raise PermissionError("fresh r2 authorization is writable")
        authorization = _read_canonical_json(
            source,
            name="fresh r2 authorization",
        )
    except (OSError, ValueError) as error:
        raise PermissionError("fresh r2 authorization is invalid") from error

    body = dict(authorization)
    fingerprint = body.pop("authorization_fingerprint", None)
    try:
        issued = datetime.fromisoformat(
            str(authorization["issued_at_utc"]).replace("Z", "+00:00")
        )
        expires = datetime.fromisoformat(
            str(authorization["expires_at_utc"]).replace("Z", "+00:00")
        )
    except (KeyError, ValueError) as error:
        raise PermissionError(
            "fresh r2 authorization time is invalid"
        ) from error
    required = {
        "schema_version": reference["required_schema"],
        "authorization_kind": "runtime_launch",
        "instruction_id": _ACTUAL_INSTRUCTION_ID,
        "authorization_basis": _ACTUAL_AUTHORIZATION_BASIS,
        "authorized_uid": os.getuid(),
        **_verify_scientific_preaccess_bindings(spec),
        **_environment_evidence_bindings(spec),
        "candidate": spec["candidate"],
        "stage_id": spec["stage_id"],
        "attempt_id": spec["attempt_id"],
        "attempt_ordinal": 2,
        "prior_attempt_count": 1,
        "fresh_attempt_authorized": True,
        "D_R_payload_authorized": True,
        "D_V_payload_authorized": False,
        "D_T_payload_authorized": False,
        "training_authorized": False,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
        "runtime_spec_v2_fingerprint": spec["runtime_spec_fingerprint"],
        "runtime_spec_v2_file_sha256": sealed_file_sha256(spec),
        "supervisor_v2_source_closure_fingerprint": stable_fingerprint(
            spec["source_bindings"]
        ),
        "unit_fragment_sha256": spec["runtime"]["systemd"][
            "unit_fragment_file_sha256"
        ],
        "preauthorization_D_R_payload_accessed": False,
        "preauthorization_D_V_payload_accessed": False,
        "preauthorization_D_T_payload_accessed": False,
        "runtime_spec_file_sha256": sealed_file_sha256(spec),
    }
    # Sample wall time only after the comparatively expensive sealed-evidence
    # revalidation.  Otherwise an authorization could expire while its
    # prerequisite chain is being recomputed yet still be accepted.
    now = datetime.now(timezone.utc)
    temporal = {"issued_at_utc", "expires_at_utc"}
    if (
        not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
        or any(authorization.get(key) != value for key, value in required.items())
        or set(authorization)
        != set(required) | temporal | {"authorization_fingerprint"}
        or issued.tzinfo is None
        or expires.tzinfo is None
        or expires <= issued
        or expires - issued > timedelta(seconds=300)
        or issued > now
        or (require_fresh and now > expires)
    ):
        raise PermissionError("fresh r2 authorization is invalid")
    return {
        "path": str(source),
        "authorization_fingerprint": fingerprint,
        "authorization_file_sha256": sealed_file_sha256(authorization),
    }


def _canonical_directory(path: Path, *, name: str) -> Path:
    if (
        not path.is_dir()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise ValueError(f"{name} is not a canonical directory")
    return path


def _stable_directory_identity(
    path: str | Path,
    *,
    expected_uid: int,
    expected_mode: int,
    name: str,
) -> os.stat_result:
    """Bind one canonical directory pathname to a stable open generation."""

    target = Path(path)
    if not target.is_absolute():
        target = Path(os.path.abspath(target))
    if target.resolve(strict=True) != target:
        raise PermissionError(f"{name} is not canonical")
    before = target.lstat()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(target, flags)
    try:
        opened = os.fstat(descriptor)
        linked = target.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)
            or (linked.st_dev, linked.st_ino)
            != (opened.st_dev, opened.st_ino)
            or opened.st_uid != expected_uid
            or stat.S_IMODE(opened.st_mode) != expected_mode
        ):
            raise PermissionError(f"{name} identity changed")
        return opened
    finally:
        os.close(descriptor)


def _validate_runtime_filesystem(spec: Mapping[str, object]) -> None:
    child = spec["child"]
    artifacts = spec["artifacts"]
    bindings = spec["source_bindings"]
    if (
        not isinstance(child, Mapping)
        or not isinstance(artifacts, Mapping)
        or not isinstance(bindings, Mapping)
    ):
        raise AssertionError("validated runtime spec changed")
    environment = spec["environment"]
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        if not isinstance(environment, Mapping):
            raise AssertionError("validated environment contract changed")
        _environment_evidence_bindings(spec)
        for field in ("gpu_lease_path", "gpu_lease_tombstone_path"):
            lease_path = Path(str(environment[field]))
            parent = _canonical_directory(
                lease_path.parent,
                name=f"environment.{field} parent",
            )
            metadata = parent.stat()
            if (
                metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise PermissionError(
                    "GPU lease directory must be private and owned"
                )
    elif environment is not None:
        raise AssertionError("non-actual environment contract changed")
    root = _canonical_directory(
        Path(str(artifacts["root"])),
        name="artifact root",
    )
    heartbeat_dir = _canonical_directory(
        Path(str(artifacts["heartbeat_dir"])),
        name="heartbeat directory",
    )
    systemd_invocation_dir = _canonical_directory(
        Path(str(artifacts["systemd_invocation_dir"])),
        name="systemd invocation directory",
    )
    for directory in (root, heartbeat_dir, systemd_invocation_dir):
        metadata = directory.stat()
        if (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise PermissionError(
                "runtime artifact directories must be private and owned"
            )
    _canonical_directory(Path(str(child["cwd"])), name="child cwd")
    if heartbeat_dir.parent != root:
        raise ValueError("heartbeat directory must be directly under root")
    if systemd_invocation_dir.parent != root:
        raise ValueError(
            "systemd invocation directory must be directly under root"
        )
    for key in _ARTIFACT_KEYS - {
        "root",
        "heartbeat_dir",
        "systemd_invocation_dir",
    }:
        if Path(str(artifacts[key])).parent != root:
            raise ValueError(f"{key} must be directly under artifact root")
    argv = child["argv"]
    if not isinstance(argv, list):
        raise AssertionError("validated child argv changed")
    executable = Path(argv[0])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("child executable is unavailable")
    if file_sha256(__file__) != bindings["supervisor_file_sha256"]:
        raise PermissionError("runtime supervisor source binding changed")
    entrypoint = child["entrypoint_path"]
    if entrypoint is not None:
        entrypoint_path = _canonical_regular_file(
            str(entrypoint),
            name="child entrypoint",
        )
        if (
            bindings["child_entry_file_sha256"] is not None
            and file_sha256(entrypoint_path)
            != bindings["child_entry_file_sha256"]
        ):
            raise PermissionError("child entrypoint source binding changed")
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        python_raw, python_metadata = _read_stable_regular_file(
            str(bindings["python_path"]),
            expected_uid=0,
            expected_mode=0o755,
            expected_nlink=1,
            name="actual Python executable",
        )
        dependency_site_metadata = _stable_directory_identity(
            str(bindings["runtime_dependency_site_path"]),
            expected_uid=_ACTUAL_RUNTIME_DEPENDENCY_SITE_OWNER_UID,
            expected_mode=_ACTUAL_RUNTIME_DEPENDENCY_SITE_MODE,
            name="actual runtime dependency site",
        )
        adapter_path = _canonical_regular_file(
            str(bindings["r2_adapter_path"]),
            name="r2 adapter",
        )
        legacy_path = _canonical_regular_file(
            str(bindings["legacy_gate_entrypoint_path"]),
            name="legacy gate entrypoint",
        )
        if (
            str(adapter_path) != str(entrypoint)
            or file_sha256(adapter_path)
            != bindings["r2_adapter_file_sha256"]
            or file_sha256(legacy_path)
            != bindings["legacy_gate_entrypoint_file_sha256"]
            or str(bindings["python_path"]) != _ACTUAL_PYTHON_PATH
            or hashlib.sha256(python_raw).hexdigest()
            != bindings["python_file_sha256"]
            or python_metadata.st_dev != bindings["python_device"]
            or python_metadata.st_ino != bindings["python_inode"]
            or python_metadata.st_uid != bindings["python_owner_uid"]
            or stat.S_IMODE(python_metadata.st_mode)
            != bindings["python_mode"]
            or str(bindings["runtime_dependency_site_path"])
            != _ACTUAL_RUNTIME_DEPENDENCY_SITE_PATH
            or dependency_site_metadata.st_dev
            != bindings["runtime_dependency_site_device"]
            or dependency_site_metadata.st_dev
            != _ACTUAL_RUNTIME_DEPENDENCY_SITE_DEVICE
            or dependency_site_metadata.st_ino
            != bindings["runtime_dependency_site_inode"]
            or dependency_site_metadata.st_ino
            != _ACTUAL_RUNTIME_DEPENDENCY_SITE_INODE
            or dependency_site_metadata.st_uid
            != bindings["runtime_dependency_site_owner_uid"]
            or dependency_site_metadata.st_uid
            != _ACTUAL_RUNTIME_DEPENDENCY_SITE_OWNER_UID
            or stat.S_IMODE(dependency_site_metadata.st_mode)
            != bindings["runtime_dependency_site_mode"]
            or stat.S_IMODE(dependency_site_metadata.st_mode)
            != _ACTUAL_RUNTIME_DEPENDENCY_SITE_MODE
        ):
            raise PermissionError("actual runtime source lineage changed")
    systemd = spec["runtime"]["systemd"]
    if not isinstance(systemd, Mapping):
        raise AssertionError("validated systemd contract changed")
    if spec["execution_kind"] in {
        ACTUAL_EXECUTION_KIND,
        SYSTEMD_INTEGRATION_DUMMY_KIND,
    }:
        shadow = systemd["immutable_shadow_properties"]
        if not isinstance(shadow, Mapping):
            raise AssertionError("validated systemd shadow changed")
        fragment = _canonical_regular_file(
            str(shadow["FragmentPath"]),
            name="systemd unit fragment",
        )
        if file_sha256(fragment) != systemd["unit_fragment_file_sha256"]:
            raise PermissionError("systemd unit fragment binding changed")


def _validate_precommit_artifacts(spec: Mapping[str, object]) -> None:
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise AssertionError("validated artifact contract changed")
    root = Path(str(artifacts["root"]))
    heartbeat_dir = Path(str(artifacts["heartbeat_dir"]))
    systemd_invocation_dir = Path(
        str(artifacts["systemd_invocation_dir"])
    )
    expected_entries = {
        heartbeat_dir.name,
        systemd_invocation_dir.name,
    }
    if {entry.name for entry in root.iterdir()} != expected_entries:
        raise PermissionError("precommit artifact root is not exact-empty")
    if any(heartbeat_dir.iterdir()) or any(systemd_invocation_dir.iterdir()):
        raise PermissionError("precommit runtime evidence directories are not empty")
    for key in _ARTIFACT_KEYS - {
        "root",
        "heartbeat_dir",
        "systemd_invocation_dir",
    }:
        path = Path(str(artifacts[key]))
        if path.exists() or path.is_symlink():
            raise PermissionError(f"precommit leaf already exists: {key}")
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        environment = spec["environment"]
        if not isinstance(environment, Mapping):
            raise AssertionError("validated environment contract changed")
        for field in ("gpu_lease_path", "gpu_lease_tombstone_path"):
            path = Path(str(environment[field]))
            if os.path.lexists(path):
                raise PermissionError(
                    f"precommit GPU lease path exists: {field}"
                )


def _open_new_log(path: Path) -> object:
    descriptor = os.open(path, _open_flags(), 0o600)
    return os.fdopen(descriptor, "wb", buffering=0, closefd=True)


def _log_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _verify_log_fd_and_path(
    descriptor_stat: os.stat_result,
    path_stat: os.stat_result,
    *,
    expected_identity: tuple[int, int] | None = None,
    required_mode: int | None = None,
) -> tuple[int, int]:
    identity = _log_identity(descriptor_stat)
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or descriptor_stat.st_nlink != 1
        or path_stat.st_nlink != 1
        or identity != _log_identity(path_stat)
        or (
            expected_identity is not None
            and identity != expected_identity
        )
        or (
            required_mode is not None
            and (
                stat.S_IMODE(descriptor_stat.st_mode) != required_mode
                or stat.S_IMODE(path_stat.st_mode) != required_mode
            )
        )
    ):
        raise RuntimeError("log fd/path identity is unsafe")
    return identity


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _finalize_log(handle: object, path: Path) -> dict[str, object]:
    if not hasattr(handle, "fileno") or not hasattr(handle, "close"):
        raise TypeError("log handle is invalid")
    descriptor = handle.fileno()
    read_descriptor = -1
    expected_identity: tuple[int, int] | None = None
    hashed_stat: os.stat_result | None = None
    digest: str | None = None
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.lstat(path)
        expected_identity = _verify_log_fd_and_path(
            descriptor_stat,
            path_stat,
        )
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)

        descriptor_stat = os.fstat(descriptor)
        path_stat = os.lstat(path)
        _verify_log_fd_and_path(
            descriptor_stat,
            path_stat,
            expected_identity=expected_identity,
            required_mode=0o444,
        )
        read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            read_flags |= os.O_NOFOLLOW
        read_descriptor = os.open(path, read_flags)
        hash_stat_before = os.fstat(read_descriptor)
        path_stat_before = os.lstat(path)
        _verify_log_fd_and_path(
            hash_stat_before,
            path_stat_before,
            expected_identity=expected_identity,
            required_mode=0o444,
        )
        digest = _sha256_descriptor(read_descriptor)
        hash_stat_after = os.fstat(read_descriptor)
        path_stat_after = os.lstat(path)
        _verify_log_fd_and_path(
            hash_stat_after,
            path_stat_after,
            expected_identity=expected_identity,
            required_mode=0o444,
        )
        before_version = (
            hash_stat_before.st_size,
            hash_stat_before.st_mtime_ns,
            hash_stat_before.st_ctime_ns,
        )
        after_version = (
            hash_stat_after.st_size,
            hash_stat_after.st_mtime_ns,
            hash_stat_after.st_ctime_ns,
        )
        if before_version != after_version:
            raise RuntimeError("log changed while it was hashed")
        hashed_stat = hash_stat_after
    finally:
        if read_descriptor >= 0:
            os.close(read_descriptor)
        handle.close()

    if (
        expected_identity is None
        or hashed_stat is None
        or digest is None
    ):
        raise RuntimeError("log finalization did not produce a receipt")
    _fsync_parent(path)
    final_path_stat = os.lstat(path)
    if (
        not stat.S_ISREG(final_path_stat.st_mode)
        or final_path_stat.st_nlink != 1
        or _log_identity(final_path_stat) != expected_identity
        or stat.S_IMODE(final_path_stat.st_mode) != 0o444
        or final_path_stat.st_size != hashed_stat.st_size
        or final_path_stat.st_mtime_ns != hashed_stat.st_mtime_ns
        or final_path_stat.st_ctime_ns != hashed_stat.st_ctime_ns
    ):
        raise RuntimeError("log identity changed after hashing")
    return {
        "path": str(path),
        "file_sha256": digest,
        "size_bytes": hashed_stat.st_size,
        "mode": stat.S_IMODE(hashed_stat.st_mode),
        "hardlink_count": hashed_stat.st_nlink,
    }


class _HeartbeatChain:
    def __init__(
        self,
        *,
        directory: Path,
        materialization_claim_sha256: str,
        attempt_id: str,
        systemd_invocation_id: str,
        child_pid: int,
    ) -> None:
        self._directory = directory
        self._previous_sha256 = materialization_claim_sha256
        self._attempt_id = attempt_id
        self._systemd_invocation_id = systemd_invocation_id
        self._boot_id = _boot_id()
        self._supervisor_starttime_ticks = _proc_starttime_ticks(os.getpid())
        self._child_pid = child_pid
        self._child_starttime_ticks = _proc_starttime_ticks(child_pid)
        self.sequence = 0
        self.last_file_sha256: str | None = None
        self.last_path: str | None = None

    def emit(
        self,
        event: str,
        *,
        child_pid: int | None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        path = self._directory / f"{self.sequence:012d}.json"
        body = {
            "schema_version": RUNTIME_HEARTBEAT_SCHEMA,
            "attempt_id": self._attempt_id,
            "sequence": self.sequence,
            "event": event,
            "time_utc": _utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "supervisor_pid": os.getpid(),
            "supervisor_proc_starttime_ticks": (
                self._supervisor_starttime_ticks
            ),
            "child_pid": child_pid,
            "child_proc_starttime_ticks": self._child_starttime_ticks,
            "boot_id": self._boot_id,
            "systemd_invocation_id": self._systemd_invocation_id,
            "previous_event_file_sha256": self._previous_sha256,
            "details": dict(details or {}),
        }
        payload = _seal(body, fingerprint_field="event_fingerprint")
        current_sha256 = _write_new_json(path, payload)
        self._previous_sha256 = current_sha256
        self.last_file_sha256 = current_sha256
        self.last_path = str(path)
        self.sequence += 1


def classify_child_exit(
    return_code: int | None,
    *,
    spawn_error: BaseException | None = None,
    forwarded_signals: Sequence[int] = (),
    forced_kill: bool = False,
) -> dict[str, object]:
    if spawn_error is not None:
        return {
            "category": "SPAWN_FAILED",
            "raw_return_code": None,
            "exit_status": None,
            "signal_number": None,
            "signal_name": None,
            "spawn_error_type": type(spawn_error).__name__,
            "spawn_errno": getattr(spawn_error, "errno", None),
            "forwarded_signals": list(forwarded_signals),
            "forced_kill": forced_kill,
        }
    if not isinstance(return_code, int):
        raise TypeError("return_code must be an integer after spawn")
    if return_code < 0:
        signal_number = -return_code
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = "UNKNOWN"
        category = "SIGNALED"
        exit_status = None
    else:
        signal_number = None
        signal_name = None
        exit_status = return_code
        category = "EXITED_0" if return_code == 0 else "EXITED_NONZERO"
    if forwarded_signals:
        category = f"FORWARDED_SIGNAL_THEN_{category}"
    return {
        "category": category,
        "raw_return_code": return_code,
        "exit_status": exit_status,
        "signal_number": signal_number,
        "signal_name": signal_name,
        "spawn_error_type": None,
        "spawn_errno": None,
        "forwarded_signals": list(forwarded_signals),
        "forced_kill": forced_kill,
    }


def _normalized_process_exit(return_code: int) -> int:
    if return_code < 0:
        return min(255, 128 + (-return_code))
    return min(255, return_code)


def _signal_process_group(process: subprocess.Popen[bytes], signum: int) -> bool:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return False
    return True


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _enable_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "PR_SET_CHILD_SUBREAPER failed")


def _reap_adopted_children() -> None:
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid <= 0:
            return


def _adopted_child_pids() -> set[int]:
    source = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    raw = source.read_text(encoding="ascii").strip()
    if not raw:
        return set()
    values = {int(value) for value in raw.split()}
    if any(value <= 0 for value in values):
        raise RuntimeError("adopted child PID list is malformed")
    return values


def _cgroup_other_pids(control_group: str) -> set[int]:
    root = _CGROUP_FILESYSTEM_ROOT.resolve(strict=True)
    candidate = (
        root / control_group.removeprefix("/")
    ).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PermissionError("unit cgroup escapes cgroup filesystem") from error
    sources = {candidate / "cgroup.procs"}
    sources.update(candidate.glob("**/cgroup.procs"))
    values: set[int] = set()
    for source in sorted(sources):
        resolved_source = source.resolve(strict=True)
        try:
            resolved_source.relative_to(candidate)
        except ValueError as error:
            raise PermissionError(
                "nested unit cgroup escapes frozen control group"
            ) from error
        raw = resolved_source.read_text(encoding="ascii").strip()
        if raw:
            values.update(int(value) for value in raw.split())
    if any(value <= 0 for value in values):
        raise RuntimeError("unit cgroup PID list is malformed")
    values.discard(os.getpid())
    return values


def _runtime_descendant_state(
    process_group_id: int,
    control_group: str | None,
) -> tuple[bool, set[int]]:
    _reap_adopted_children()
    pids = _adopted_child_pids()
    if control_group is not None:
        pids.update(_cgroup_other_pids(control_group))
    pids.discard(os.getpid())
    return _process_group_exists(process_group_id), pids


def _signal_runtime_descendants(
    process_group_id: int,
    pids: set[int],
    signum: int,
) -> None:
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        pass
    for pid in sorted(pids):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def _wait_runtime_quiescence(
    process_group_id: int,
    control_group: str | None,
    *,
    deadline: float,
    poll_interval: float,
) -> bool:
    while True:
        group_exists, pids = _runtime_descendant_state(
            process_group_id,
            control_group,
        )
        if not group_exists and not pids:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_interval)


def _quiesce_runtime_descendants(
    process_group_id: int,
    control_group: str | None,
    *,
    grace_seconds: float,
    poll_interval: float,
) -> list[int]:
    group_exists, pids = _runtime_descendant_state(
        process_group_id,
        control_group,
    )
    if not group_exists and not pids:
        return []
    applied: list[int] = []
    for signum in (int(signal.SIGTERM), int(signal.SIGKILL)):
        _signal_runtime_descendants(process_group_id, pids, signum)
        applied.append(signum)
        if _wait_runtime_quiescence(
            process_group_id,
            control_group,
            deadline=time.monotonic() + grace_seconds,
            poll_interval=poll_interval,
        ):
            return applied
        group_exists, pids = _runtime_descendant_state(
            process_group_id,
            control_group,
        )
    raise RuntimeError(
        "runtime descendants did not quiesce before log sealing"
    )


def _write_runtime_attestation(
    spec: Mapping[str, object],
    *,
    spec_path: Path,
    authorization: Mapping[str, object],
    attempt_commit: Mapping[str, object],
    materialization_claim: Mapping[str, object],
    gpu_lease: Mapping[str, object],
) -> dict[str, object]:
    artifacts = spec["artifacts"]
    bindings = spec["source_bindings"]
    preaccess = spec["scientific_preaccess"]
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(bindings, Mapping)
        or not isinstance(preaccess, Mapping)
    ):
        raise AssertionError("validated runtime attestation contract changed")
    if any(
        gpu_lease.get(field) != attempt_commit.get(field)
        for field in (
            "gpu_lease_fingerprint",
            "gpu_lease_file_sha256",
            "gpu_lease_device",
            "gpu_lease_inode",
            "gpu_lease_parent_device",
            "gpu_lease_parent_inode",
        )
    ):
        raise PermissionError(
            "runtime attestation GPU lease identity changed from commit"
        )
    claim_path = _canonical_regular_file(
        str(artifacts["materialization_claim"]),
        name="materialization claim",
    )
    commit_path = _canonical_regular_file(
        str(artifacts["attempt_commit"]),
        name="attempt commit",
    )
    body = {
        "schema_version": RUNTIME_ATTESTATION_SCHEMA,
        "authorization_kind": "runtime_launch",
        "execution_kind": spec["execution_kind"],
        "candidate": spec["candidate"],
        "stage_id": spec["stage_id"],
        "attempt_id": spec["attempt_id"],
        "runtime_spec_path": str(spec_path),
        "runtime_spec_file_sha256": sealed_file_sha256(spec),
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
        "runtime_launch_authorization_path": authorization["path"],
        "runtime_launch_authorization_file_sha256": authorization[
            "authorization_file_sha256"
        ],
        "runtime_launch_authorization_fingerprint": authorization[
            "authorization_fingerprint"
        ],
        "attempt_commit_path": str(commit_path),
        "attempt_commit_file_sha256": attempt_commit[
            "attempt_commit_file_sha256"
        ],
        "attempt_commit_fingerprint": attempt_commit[
            "attempt_commit_fingerprint"
        ],
        "planned_attempt_commit_fingerprint": attempt_commit[
            "planned_attempt_commit_fingerprint"
        ],
        "materialization_claim_path": str(claim_path),
        "materialization_claim_file_sha256": materialization_claim[
            "materialization_claim_file_sha256"
        ],
        "materialization_claim_fingerprint": materialization_claim[
            "materialization_claim_fingerprint"
        ],
        "active_gpu_lease_path": str(spec["environment"]["gpu_lease_path"]),
        "active_gpu_lease_file_sha256": gpu_lease[
            "gpu_lease_file_sha256"
        ],
        "active_gpu_lease_fingerprint": gpu_lease[
            "gpu_lease_fingerprint"
        ],
        "active_gpu_lease_device": gpu_lease["gpu_lease_device"],
        "active_gpu_lease_inode": gpu_lease["gpu_lease_inode"],
        "active_gpu_lease_parent_device": gpu_lease[
            "gpu_lease_parent_device"
        ],
        "active_gpu_lease_parent_inode": gpu_lease[
            "gpu_lease_parent_inode"
        ],
        "systemd_invocation_id": materialization_claim[
            "systemd_invocation_id"
        ],
        "systemd_control_group": materialization_claim[
            "systemd_control_group"
        ],
        "child_argv": list(spec["child"]["argv"]),
        "child_argv_fingerprint": spec["child"]["argv_fingerprint"],
        "r2_adapter_path": bindings["r2_adapter_path"],
        "r2_adapter_file_sha256": bindings["r2_adapter_file_sha256"],
        "legacy_gate_entrypoint_path": bindings[
            "legacy_gate_entrypoint_path"
        ],
        "legacy_gate_entrypoint_file_sha256": bindings[
            "legacy_gate_entrypoint_file_sha256"
        ],
        "source_closure_fingerprint_103": preaccess[
            "source_closure_fingerprint_103"
        ],
        "child_prespawn_environment_audit_fingerprint": (
            _read_canonical_json(
                str(artifacts["child_prespawn_phase_receipt"]),
                name="child prespawn receipt",
            )["environment_audit_fingerprint"]
        ),
        "created_at_utc": _utc_now(),
        "boot_id": _boot_id(),
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    payload = _seal(body, fingerprint_field="runtime_attestation_fingerprint")
    digest = _write_new_json(
        Path(str(artifacts["runtime_attestation"])),
        payload,
    )
    return {**payload, "runtime_attestation_file_sha256": digest}


def _current_process_argv() -> list[str]:
    try:
        raw = Path("/proc/self/cmdline").read_bytes()
        parts = raw.split(b"\0")
        if parts and parts[-1] == b"":
            parts.pop()
        if parts:
            return [os.fsdecode(part) for part in parts]
    except OSError:
        pass
    original = getattr(sys, "orig_argv", None)
    if not isinstance(original, list) or not original:
        raise PermissionError("current process argv is unavailable")
    return [str(value) for value in original]


def verify_child_runtime_attestation(
    attestation_path: str | Path,
    runtime_launch_authorization_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
    cgroup_path: str | None = None,
) -> dict[str, object]:
    """Verify the supervisor-created launch proof inside the r2 adapter."""

    effective_environment = (
        dict(os.environ) if environment is None else dict(environment)
    )
    attestation_file = _canonical_regular_file(
        attestation_path,
        name="runtime attestation",
    )
    reserved = {
        key: value
        for key, value in effective_environment.items()
        if key.startswith(_RUNTIME_RESERVED_ENV_PREFIX)
    }
    if reserved != {_RUNTIME_ATTESTATION_ENV: str(attestation_file)}:
        raise PermissionError("runtime attestation environment is not exact")
    metadata = attestation_file.stat()
    if (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o444
    ):
        raise PermissionError("runtime attestation is not immutable and owned")
    payload = _read_canonical_json(
        attestation_file,
        name="runtime attestation",
    )
    body = dict(payload)
    fingerprint = body.pop("runtime_attestation_fingerprint", None)
    expected_keys = {
        "schema_version", "authorization_kind", "execution_kind",
        "candidate", "stage_id", "attempt_id", "runtime_spec_path",
        "runtime_spec_file_sha256", "runtime_spec_fingerprint",
        "runtime_launch_authorization_path",
        "runtime_launch_authorization_file_sha256",
        "runtime_launch_authorization_fingerprint", "attempt_commit_path",
        "attempt_commit_file_sha256", "attempt_commit_fingerprint",
        "planned_attempt_commit_fingerprint", "materialization_claim_path",
        "materialization_claim_file_sha256",
        "materialization_claim_fingerprint", "active_gpu_lease_path",
        "active_gpu_lease_file_sha256", "active_gpu_lease_fingerprint",
        "active_gpu_lease_device", "active_gpu_lease_inode",
        "active_gpu_lease_parent_device",
        "active_gpu_lease_parent_inode",
        "systemd_invocation_id", "systemd_control_group", "child_argv",
        "child_argv_fingerprint", "r2_adapter_path",
        "r2_adapter_file_sha256", "legacy_gate_entrypoint_path",
        "legacy_gate_entrypoint_file_sha256",
        "source_closure_fingerprint_103",
        "child_prespawn_environment_audit_fingerprint", "created_at_utc",
        "boot_id", "D_R_payload_accessed", "D_V_payload_accessed",
        "D_T_payload_accessed", "runtime_attestation_fingerprint",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != RUNTIME_ATTESTATION_SCHEMA
        or payload.get("authorization_kind") != "runtime_launch"
        or payload.get("execution_kind") != ACTUAL_EXECUTION_KIND
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
        or payload.get("boot_id") != _boot_id()
        or not _is_sha256(payload.get("active_gpu_lease_file_sha256"))
        or not _is_sha256(payload.get("active_gpu_lease_fingerprint"))
        or any(
            not _is_positive_int(payload.get(field))
            for field in (
                "active_gpu_lease_device",
                "active_gpu_lease_inode",
                "active_gpu_lease_parent_device",
                "active_gpu_lease_parent_inode",
            )
        )
        or any(
            payload.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError("runtime attestation is invalid")
    spec_path = _canonical_regular_file(
        str(payload["runtime_spec_path"]),
        name="attested runtime spec",
    )
    spec = load_runtime_spec(spec_path)
    _validate_runtime_filesystem(spec)
    if (
        sealed_file_sha256(spec) != payload["runtime_spec_file_sha256"]
        or spec["runtime_spec_fingerprint"]
        != payload["runtime_spec_fingerprint"]
    ):
        raise PermissionError("attested runtime spec identity changed")
    supplied_authorization = _canonical_regular_file(
        runtime_launch_authorization_path,
        name="runtime launch authorization",
    )
    if str(supplied_authorization) != payload[
        "runtime_launch_authorization_path"
    ]:
        raise PermissionError("runtime launch authorization path changed")
    authorization = _verify_actual_authorization(spec, spec_path=spec_path)
    if (
        authorization["authorization_fingerprint"]
        != payload["runtime_launch_authorization_fingerprint"]
        or authorization["authorization_file_sha256"]
        != payload["runtime_launch_authorization_file_sha256"]
    ):
        raise PermissionError("attested runtime authorization changed")
    attempt_commit = _verify_attempt_commit(spec, authorization)
    if (
        str(spec["artifacts"]["attempt_commit"])
        != payload["attempt_commit_path"]
        or attempt_commit["attempt_commit_file_sha256"]
        != payload["attempt_commit_file_sha256"]
        or attempt_commit["attempt_commit_fingerprint"]
        != payload["attempt_commit_fingerprint"]
        or attempt_commit["planned_attempt_commit_fingerprint"]
        != payload["planned_attempt_commit_fingerprint"]
    ):
        raise PermissionError("attested attempt commit changed")
    invocation_id = _systemd_invocation_id(effective_environment)
    current_cgroup = _self_cgroup_path() if cgroup_path is None else cgroup_path
    if (
        invocation_id != payload["systemd_invocation_id"]
        or current_cgroup != payload["systemd_control_group"]
    ):
        raise PermissionError("attested systemd identity changed")
    claim = _verify_materialization_claim(
        spec,
        authorization=authorization,
        attempt_commit=attempt_commit,
        systemd_invocation_id=invocation_id,
        systemd_control_group=current_cgroup,
    )
    if (
        str(spec["artifacts"]["materialization_claim"])
        != payload["materialization_claim_path"]
        or claim["materialization_claim_file_sha256"]
        != payload["materialization_claim_file_sha256"]
        or claim["materialization_claim_fingerprint"]
        != payload["materialization_claim_fingerprint"]
    ):
        raise PermissionError("attested materialization claim changed")
    observed_argv = (
        _current_process_argv() if argv is None else [str(value) for value in argv]
    )
    if (
        observed_argv != payload["child_argv"]
        or stable_fingerprint(observed_argv)
        != payload["child_argv_fingerprint"]
    ):
        raise PermissionError("attested child argv changed")
    bindings = spec["source_bindings"]
    preaccess = spec["scientific_preaccess"]
    required = {
        "r2_adapter_path": bindings["r2_adapter_path"],
        "r2_adapter_file_sha256": bindings["r2_adapter_file_sha256"],
        "legacy_gate_entrypoint_path": bindings["legacy_gate_entrypoint_path"],
        "legacy_gate_entrypoint_file_sha256": bindings[
            "legacy_gate_entrypoint_file_sha256"
        ],
        "source_closure_fingerprint_103": preaccess[
            "source_closure_fingerprint_103"
        ],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise PermissionError("attested scientific source lineage changed")
    lease = _verify_active_gpu_lease(spec, attempt_commit)
    try:
        if (
            str(spec["environment"]["gpu_lease_path"])
            != payload["active_gpu_lease_path"]
            or lease["gpu_lease_file_sha256"]
            != payload["active_gpu_lease_file_sha256"]
            or lease["gpu_lease_fingerprint"]
            != payload["active_gpu_lease_fingerprint"]
            or lease["gpu_lease_device"] != payload["active_gpu_lease_device"]
            or lease["gpu_lease_inode"] != payload["active_gpu_lease_inode"]
            or lease["gpu_lease_parent_device"]
            != payload["active_gpu_lease_parent_device"]
            or lease["gpu_lease_parent_inode"]
            != payload["active_gpu_lease_parent_inode"]
        ):
            raise PermissionError("attested active GPU lease changed")
    finally:
        lease["handle"].close_without_release()
    return {
        **payload,
        "runtime_attestation_file_sha256": sealed_file_sha256(payload),
        "runtime_attestation_valid": True,
    }

def _child_environment(
    child: Mapping[str, object],
    *,
    dynamic: Mapping[str, str] | None = None,
) -> dict[str, str]:
    inherited = child["inherit_environment"]
    explicit = child["environment"]
    if not isinstance(inherited, list) or not isinstance(explicit, Mapping):
        raise AssertionError("validated child environment changed")
    environment = {
        name: os.environ[name]
        for name in inherited
        if name in os.environ
    }
    environment.update(
        {str(key): str(value) for key, value in explicit.items()}
    )
    if dynamic is not None:
        if any(
            not isinstance(key, str)
            or not key.startswith(_RUNTIME_RESERVED_ENV_PREFIX)
            or not isinstance(value, str)
            or not value
            for key, value in dynamic.items()
        ):
            raise ValueError("dynamic runtime environment is malformed")
        if set(dynamic) & set(environment):
            raise ValueError("dynamic runtime environment collides")
        environment.update(dynamic)
    return environment

def _valid_invocation_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_systemd_phase_state(
    state: Mapping[str, str],
    *,
    phase: str,
    expected_invocation_id: str | None = None,
    prior_invocation_id: str | None = None,
) -> None:
    if set(state) != _SYSTEMD_PHASE_STATE_KEYS:
        raise ValueError("systemd phase state is incomplete")
    if (
        state["LoadState"] != "loaded"
        or state["UnitFileState"] != "static"
        or state["NRestarts"] != "0"
        or state["NeedDaemonReload"] != "no"
    ):
        raise PermissionError("systemd phase state is not launch-safe")
    invocation_id = state["InvocationID"]
    if phase == "precommit":
        if (
            state["ActiveState"] != "inactive"
            or state["SubState"] != "dead"
        ):
            raise PermissionError("target unit is not inactive before commit")
        if invocation_id and not _valid_invocation_id(invocation_id):
            raise PermissionError("precommit InvocationID is malformed")
        return
    if not _valid_invocation_id(invocation_id):
        raise PermissionError("systemd start has no valid InvocationID")
    if prior_invocation_id and invocation_id == prior_invocation_id:
        raise PermissionError("systemd start did not create a new invocation")
    if expected_invocation_id is not None and invocation_id != expected_invocation_id:
        raise PermissionError("systemd phase InvocationID changed")
    if phase == "start_ack":
        if (
            state["ActiveState"],
            state["SubState"],
        ) not in {
            ("activating", "start-pre"),
            ("activating", "start"),
            ("active", "running"),
        }:
            raise PermissionError(
                "systemd start has not passed ExecCondition"
            )
        return
    if phase == "child_prespawn" and state["ActiveState"] not in {
        "activating",
        "active",
    }:
        raise PermissionError("unit is not active immediately before Popen")


def _write_phase_receipt(
    spec: Mapping[str, object],
    *,
    phase: str,
    phase_state: Mapping[str, str],
    immutable_shadow: Mapping[str, str],
    path: Path,
    launch_lease: Mapping[str, object] | None,
    environment_audit: Mapping[str, object] | None = None,
    gpu_lease: Mapping[str, object] | None = None,
) -> dict[str, object]:
    body = {
        "schema_version": RUNTIME_PHASE_RECEIPT_SCHEMA,
        "execution_kind": spec["execution_kind"],
        "candidate": spec["candidate"],
        "stage_id": spec["stage_id"],
        "attempt_id": spec["attempt_id"],
        "phase": phase,
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
        "time_utc": _utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        "boot_id": _boot_id(),
        "systemd_phase_state": dict(phase_state),
        "systemd_phase_state_fingerprint": stable_fingerprint(
            dict(phase_state)
        ),
        "immutable_shadow_fingerprint": stable_fingerprint(
            dict(immutable_shadow)
        ),
        "launch_lease_fingerprint": (
            launch_lease.get("launch_lease_fingerprint")
            if launch_lease is not None
            else None
        ),
        "dispatch_lease_scope": "attempt_dispatch_only",
        "environment_audit_fingerprint": (
            environment_audit.get("environment_audit_fingerprint")
            if environment_audit is not None
            else None
        ),
        "environment_inventory_fingerprint": (
            environment_audit.get("inventory_fingerprint")
            if environment_audit is not None
            else None
        ),
        "runtime_environment_audit_valid": (
            environment_audit.get("runtime_environment_audit_valid") is True
            if environment_audit is not None
            else False
        ),
        "gpu_lease_fingerprint": (
            gpu_lease.get("gpu_lease_fingerprint")
            if gpu_lease is not None
            else None
        ),
        "launch_limit": 1,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
        "scientific_gate_passed": None,
    }
    if (
        spec["execution_kind"] == ACTUAL_EXECUTION_KIND
        and phase in {"precommit", "child_prespawn"}
        and (
            body["runtime_environment_audit_valid"] is not True
            or not _is_sha256(body["environment_audit_fingerprint"])
            or not _is_sha256(body["environment_inventory_fingerprint"])
            or not _is_sha256(body["gpu_lease_fingerprint"])
        )
    ):
        raise PermissionError(f"{phase} environment evidence is invalid")
    payload = _seal(body, fingerprint_field="phase_receipt_fingerprint")
    file_digest = _write_new_json(path, payload)
    return {**payload, "phase_receipt_file_sha256": file_digest}


def _verify_phase_receipt(
    spec: Mapping[str, object],
    *,
    path: Path,
    expected_phase: str,
    expected_invocation_id: str | None = None,
) -> dict[str, object]:
    runtime = spec["runtime"]
    if not isinstance(runtime, Mapping) or not isinstance(
        runtime["systemd"], Mapping
    ):
        raise AssertionError("validated systemd contract changed")
    immutable_shadow = runtime["systemd"][
        "immutable_shadow_properties"
    ]
    if not isinstance(immutable_shadow, Mapping):
        raise AssertionError("validated immutable shadow changed")
    payload = _read_canonical_json(path, name=f"{expected_phase} receipt")
    body = dict(payload)
    fingerprint = body.pop("phase_receipt_fingerprint", None)
    state = payload.get("systemd_phase_state")
    if (
        not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
        or payload.get("schema_version") != RUNTIME_PHASE_RECEIPT_SCHEMA
        or payload.get("execution_kind") != spec["execution_kind"]
        or payload.get("candidate") != spec["candidate"]
        or payload.get("stage_id") != spec["stage_id"]
        or payload.get("attempt_id") != spec["attempt_id"]
        or payload.get("phase") != expected_phase
        or payload.get("runtime_spec_fingerprint")
        != spec["runtime_spec_fingerprint"]
        or payload.get("boot_id") != _boot_id()
        or not isinstance(state, dict)
        or set(state) != _SYSTEMD_PHASE_STATE_KEYS
        or payload.get("systemd_phase_state_fingerprint")
        != stable_fingerprint(state)
        or payload.get("immutable_shadow_fingerprint")
        != stable_fingerprint(immutable_shadow)
        or (
            expected_invocation_id is not None
            and state.get("InvocationID") != expected_invocation_id
        )
        or payload.get("dispatch_lease_scope") != "attempt_dispatch_only"
        or payload.get("launch_limit") != 1
        or payload.get("automatic_retry_allowed") is not False
        or payload.get("resume_allowed") is not False
    ):
        raise PermissionError(f"{expected_phase} receipt is invalid")
    if (
        spec["execution_kind"] == ACTUAL_EXECUTION_KIND
        and expected_phase in {"precommit", "child_prespawn"}
        and (
            payload.get("runtime_environment_audit_valid") is not True
            or not _is_sha256(payload.get("environment_audit_fingerprint"))
            or not _is_sha256(payload.get("environment_inventory_fingerprint"))
            or not _is_sha256(payload.get("gpu_lease_fingerprint"))
        )
    ):
        raise PermissionError(f"{expected_phase} environment evidence is invalid")
    return {
        **payload,
        "phase_receipt_file_sha256": sealed_file_sha256(payload),
    }


def _wait_for_start_ack_receipt(
    spec: Mapping[str, object],
    *,
    expected_invocation_id: str,
) -> dict[str, object]:
    runtime = spec["runtime"]
    artifacts = spec["artifacts"]
    if not isinstance(runtime, Mapping) or not isinstance(artifacts, Mapping):
        raise AssertionError("validated runtime spec changed")
    systemd = runtime["systemd"]
    if not isinstance(systemd, Mapping):
        raise AssertionError("validated systemd contract changed")
    path = Path(str(artifacts["start_ack_receipt"]))
    deadline = time.monotonic() + float(
        systemd["start_ack_timeout_seconds"]
    )
    while time.monotonic() < deadline:
        if path.is_file() and not path.is_symlink():
            return _verify_phase_receipt(
                spec,
                path=path,
                expected_phase="start_ack",
                expected_invocation_id=expected_invocation_id,
            )
        time.sleep(float(systemd["start_ack_poll_seconds"]))
    raise RuntimeError("start acknowledgement receipt did not materialize")


def _create_launch_lease(
    spec: Mapping[str, object],
    authorization: Mapping[str, object] | None,
) -> dict[str, object]:
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise AssertionError("validated artifact contract changed")
    body = {
        "schema_version": LAUNCH_LEASE_SCHEMA,
        "execution_kind": spec["execution_kind"],
        "candidate": spec["candidate"],
        "stage_id": spec["stage_id"],
        "attempt_id": spec["attempt_id"],
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
        "authorization_fingerprint": (
            authorization.get("authorization_fingerprint")
            if authorization is not None
            else None
        ),
        "time_utc": _utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        "boot_id": _boot_id(),
        "committer_pid": os.getpid(),
        "committer_proc_starttime_ticks": _proc_starttime_ticks(os.getpid()),
        "launch_limit": 1,
        "lease_scope": "attempt_dispatch_only",
        "gpu_exclusivity_claimed": False,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
    }
    payload = _seal(body, fingerprint_field="launch_lease_fingerprint")
    path = Path(str(artifacts["launch_lease"]))
    file_digest = _write_new_json(path, payload)
    return {**payload, "launch_lease_file_sha256": file_digest}


def _verify_launch_lease(
    spec: Mapping[str, object],
) -> dict[str, object]:
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise AssertionError("validated artifact contract changed")
    path = Path(str(artifacts["launch_lease"]))
    payload = _read_canonical_json(path, name="launch lease")
    body = dict(payload)
    fingerprint = body.pop("launch_lease_fingerprint", None)
    if (
        not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
        or payload.get("schema_version") != LAUNCH_LEASE_SCHEMA
        or payload.get("execution_kind") != spec["execution_kind"]
        or payload.get("candidate") != spec["candidate"]
        or payload.get("stage_id") != spec["stage_id"]
        or payload.get("attempt_id") != spec["attempt_id"]
        or payload.get("runtime_spec_fingerprint")
        != spec["runtime_spec_fingerprint"]
        or payload.get("boot_id") != _boot_id()
        or payload.get("launch_limit") != 1
        or payload.get("lease_scope") != "attempt_dispatch_only"
        or payload.get("gpu_exclusivity_claimed") is not False
        or payload.get("automatic_retry_allowed") is not False
        or payload.get("resume_allowed") is not False
    ):
        raise PermissionError("launch lease is invalid")
    return {
        **payload,
        "launch_lease_file_sha256": sealed_file_sha256(payload),
    }


def _release_uncommitted_launch_lease(
    spec: Mapping[str, object],
    lease: Mapping[str, object],
) -> None:
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise AssertionError("validated artifact contract changed")
    path = _canonical_regular_file(
        str(artifacts["launch_lease"]),
        name="uncommitted launch lease",
    )
    payload = {
        key: value
        for key, value in lease.items()
        if key != "launch_lease_file_sha256"
    }
    if (
        sealed_file_sha256(payload)
        != lease["launch_lease_file_sha256"]
    ):
        raise RuntimeError("uncommitted launch lease identity changed")
    _unlink_exact_json_generation(
        path,
        payload,
        expected_mode=0o444,
        name="uncommitted launch lease",
    )


def _attempt_commit_payload(
    spec: Mapping[str, object],
    authorization: Mapping[str, object] | None,
    systemd_shadow: Mapping[str, str],
    precommit_phase_receipt: Mapping[str, object],
    launch_lease: Mapping[str, object],
    planned_attempt_commit_fingerprint: str | None = None,
    environment_audit: Mapping[str, object] | None = None,
    gpu_lease: Mapping[str, object] | None = None,
) -> dict[str, object]:
    body = {
            "schema_version": ATTEMPT_COMMIT_SCHEMA,
            "execution_kind": spec["execution_kind"],
            "candidate": spec["candidate"],
            "stage_id": spec["stage_id"],
            "attempt_id": spec["attempt_id"],
            "attempt_ordinal": spec["attempt_ordinal"],
            "prior_attempt_count": spec["prior_attempt_count"],
            "runtime_spec_fingerprint": spec[
                "runtime_spec_fingerprint"
            ],
            "authorization_fingerprint": (
                authorization["authorization_fingerprint"]
                if authorization is not None
                else None
            ),
            "authorization_file_sha256": (
                authorization["authorization_file_sha256"]
                if authorization is not None
                else None
            ),
            "precommit_phase_receipt_fingerprint": (
                precommit_phase_receipt["phase_receipt_fingerprint"]
            ),
            "precommit_phase_receipt_file_sha256": (
                precommit_phase_receipt["phase_receipt_file_sha256"]
            ),
            "launch_lease_fingerprint": launch_lease[
                "launch_lease_fingerprint"
            ],
            "launch_lease_file_sha256": launch_lease[
                "launch_lease_file_sha256"
            ],
            "dispatch_lease_scope": "attempt_dispatch_only",
            "planned_attempt_commit_fingerprint": (
                planned_attempt_commit_fingerprint
            ),
            "precommit_environment_audit_fingerprint": (
                environment_audit.get("environment_audit_fingerprint")
                if environment_audit is not None
                else None
            ),
            "precommit_environment_inventory_fingerprint": (
                environment_audit.get("inventory_fingerprint")
                if environment_audit is not None
                else None
            ),
            "runtime_environment_audit_valid": (
                environment_audit.get("runtime_environment_audit_valid") is True
                if environment_audit is not None
                else False
            ),
            "gpu_lease_fingerprint": (
                gpu_lease.get("gpu_lease_fingerprint")
                if gpu_lease is not None
                else None
            ),
            "gpu_lease_file_sha256": (
                gpu_lease.get("gpu_lease_file_sha256")
                if gpu_lease is not None
                else None
            ),
            "gpu_lease_device": (
                gpu_lease.get("gpu_lease_device")
                if gpu_lease is not None
                else None
            ),
            "gpu_lease_inode": (
                gpu_lease.get("gpu_lease_inode")
                if gpu_lease is not None
                else None
            ),
            "gpu_lease_parent_device": (
                gpu_lease.get("gpu_lease_parent_device")
                if gpu_lease is not None
                else None
            ),
            "gpu_lease_parent_inode": (
                gpu_lease.get("gpu_lease_parent_inode")
                if gpu_lease is not None
                else None
            ),
            "time_utc": _utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "committer_pid": os.getpid(),
            "committer_proc_starttime_ticks": _proc_starttime_ticks(
                os.getpid()
            ),
            "boot_id": _boot_id(),
            "systemd_unit_name": spec["runtime"]["systemd"][
                "unit_name"
            ],
            "immutable_systemd_shadow_properties": dict(systemd_shadow),
            "immutable_systemd_shadow_fingerprint": stable_fingerprint(
                dict(systemd_shadow)
            ),
            "launch_limit": 1,
            "automatic_retry_allowed": False,
            "resume_allowed": False,
            "scientific_gate_passed": None,
    }
    gpu_identity_fields = (
        "gpu_lease_fingerprint",
        "gpu_lease_file_sha256",
        "gpu_lease_device",
        "gpu_lease_inode",
        "gpu_lease_parent_device",
        "gpu_lease_parent_inode",
    )
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        if (
            not _is_sha256(body["gpu_lease_fingerprint"])
            or not _is_sha256(body["gpu_lease_file_sha256"])
            or any(
                not _is_positive_int(body[field])
                for field in (
                    "gpu_lease_device",
                    "gpu_lease_inode",
                    "gpu_lease_parent_device",
                    "gpu_lease_parent_inode",
                )
            )
        ):
            raise PermissionError(
                "actual attempt commit GPU lease identity is invalid"
            )
    elif any(body[field] is not None for field in gpu_identity_fields):
        raise PermissionError(
            "dummy attempt commit carries GPU lease identity"
        )
    return _seal(
        body,
        fingerprint_field="attempt_commit_fingerprint",
    )


def _materialization_claim(
    spec: Mapping[str, object],
    authorization: Mapping[str, object] | None,
    attempt_commit: Mapping[str, object] | None,
    systemd_invocation_id: str,
    systemd_control_group: str | None,
) -> dict[str, object]:
    return _seal(
        {
            "schema_version": MATERIALIZATION_CLAIM_SCHEMA,
            "execution_kind": spec["execution_kind"],
            "candidate": spec["candidate"],
            "stage_id": spec["stage_id"],
            "attempt_id": spec["attempt_id"],
            "runtime_spec_fingerprint": spec[
                "runtime_spec_fingerprint"
            ],
            "authorization_fingerprint": (
                authorization["authorization_fingerprint"]
                if authorization is not None
                else None
            ),
            "attempt_commit_fingerprint": (
                attempt_commit["attempt_commit_fingerprint"]
                if attempt_commit is not None
                else None
            ),
            "attempt_commit_file_sha256": (
                attempt_commit["attempt_commit_file_sha256"]
                if attempt_commit is not None
                else None
            ),
            "time_utc": _utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "claimer_pid": os.getpid(),
            "claimer_proc_starttime_ticks": _proc_starttime_ticks(
                os.getpid()
            ),
            "boot_id": _boot_id(),
            "systemd_invocation_id": systemd_invocation_id,
            "systemd_control_group": systemd_control_group,
            "launch_limit": 1,
            "shell": False,
            "automatic_retry_allowed": False,
            "resume_allowed": False,
            "child_argv_fingerprint": spec["child"][
                "argv_fingerprint"
            ],
            "scientific_gate_passed": None,
        },
        fingerprint_field="materialization_claim_fingerprint",
    )


def _terminal_payload(
    spec: Mapping[str, object],
    *,
    materialization_claim: Mapping[str, object],
    materialization_claim_sha256: str,
    child_outcome: Mapping[str, object],
    heartbeat: _HeartbeatChain | None,
    stdout_receipt: Mapping[str, object] | None,
    stderr_receipt: Mapping[str, object] | None,
    start_ack_receipt: Mapping[str, object] | None,
    child_prespawn_receipt: Mapping[str, object] | None,
    supervisor_error: BaseException | None,
    process_group_cleanup_signals: Sequence[int],
) -> dict[str, object]:
    return _seal(
        {
            "schema_version": RUNTIME_TERMINAL_SCHEMA,
            "execution_kind": spec["execution_kind"],
            "candidate": spec["candidate"],
            "stage_id": spec["stage_id"],
            "attempt_id": spec["attempt_id"],
            "runtime_spec_fingerprint": spec[
                "runtime_spec_fingerprint"
            ],
            "materialization_claim_file_sha256": (
                materialization_claim_sha256
            ),
            "boot_id": materialization_claim["boot_id"],
            "systemd_invocation_id": materialization_claim[
                "systemd_invocation_id"
            ],
            "time_utc": _utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "child_outcome": dict(child_outcome),
            "heartbeat_event_count": (
                heartbeat.sequence if heartbeat is not None else 0
            ),
            "last_heartbeat_path": (
                heartbeat.last_path if heartbeat is not None else None
            ),
            "last_heartbeat_file_sha256": (
                heartbeat.last_file_sha256
                if heartbeat is not None
                else None
            ),
            "stdout_log": (
                dict(stdout_receipt)
                if stdout_receipt is not None
                else None
            ),
            "stderr_log": (
                dict(stderr_receipt)
                if stderr_receipt is not None
                else None
            ),
            "start_ack_receipt_fingerprint": (
                start_ack_receipt.get("phase_receipt_fingerprint")
                if start_ack_receipt is not None
                else None
            ),
            "start_ack_receipt_file_sha256": (
                start_ack_receipt.get("phase_receipt_file_sha256")
                if start_ack_receipt is not None
                else None
            ),
            "child_prespawn_phase_receipt_fingerprint": (
                child_prespawn_receipt.get("phase_receipt_fingerprint")
                if child_prespawn_receipt is not None
                else None
            ),
            "child_prespawn_phase_receipt_file_sha256": (
                child_prespawn_receipt.get("phase_receipt_file_sha256")
                if child_prespawn_receipt is not None
                else None
            ),
            "supervisor_error_type": (
                type(supervisor_error).__name__
                if supervisor_error is not None
                else None
            ),
            "process_group_cleanup_signals": list(
                process_group_cleanup_signals
            ),
            "scientific_decision": (
                "NOT_EVALUATED_BY_RUNTIME_SUPERVISOR"
            ),
            "scientific_gate_passed": None,
        },
        fingerprint_field="runtime_terminal_fingerprint",
    )


def _run_child_once(
    spec: Mapping[str, object],
    spec_path: Path,
    *,
    authorization: Mapping[str, object] | None,
    attempt_commit: Mapping[str, object] | None,
    materialization_claim: Mapping[str, object],
) -> int:
    child = spec["child"]
    artifacts = spec["artifacts"]
    runtime = spec["runtime"]
    if (
        not isinstance(child, Mapping)
        or not isinstance(artifacts, Mapping)
        or not isinstance(runtime, Mapping)
    ):
        raise AssertionError("validated runtime spec changed")

    materialization_sha256 = str(
        materialization_claim["materialization_claim_file_sha256"]
    )

    stdout_path = Path(str(artifacts["stdout_log"]))
    stderr_path = Path(str(artifacts["stderr_log"]))
    terminal_path = Path(str(artifacts["runtime_terminal"]))
    heartbeat_dir = Path(str(artifacts["heartbeat_dir"]))
    stdout_handle: object | None = None
    stderr_handle: object | None = None
    stdout_receipt: dict[str, object] | None = None
    stderr_receipt: dict[str, object] | None = None
    start_ack_receipt: dict[str, object] | None = None
    dynamic_environment: dict[str, str] | None = None
    child_prespawn_receipt: dict[str, object] | None = None
    heartbeat: _HeartbeatChain | None = None
    process: subprocess.Popen[bytes] | None = None
    spawn_error: BaseException | None = None
    supervisor_error: BaseException | None = None
    forwarded_signals: list[int] = []
    process_group_cleanup_signals: list[int] = []
    forced_kill = False
    logs_safe_to_finalize = False
    runtime_control_group = materialization_claim.get(
        "systemd_control_group"
    )
    if runtime_control_group is not None and not isinstance(
        runtime_control_group, str
    ):
        raise AssertionError("validated control group changed")

    pending_signals: list[int] = []

    def request_stop(signum: int, _frame: object) -> None:
        pending_signals.append(signum)

    handled_signals = tuple(
        value
        for value in (
            signal.SIGINT,
            signal.SIGTERM,
            getattr(signal, "SIGHUP", None),
        )
        if isinstance(value, signal.Signals)
    )
    previous_handlers: dict[signal.Signals, object] = {}

    try:
        stdout_handle = _open_new_log(stdout_path)
        stderr_handle = _open_new_log(stderr_path)
        previous_handlers = {
            value: signal.getsignal(value) for value in handled_signals
        }
        for value in handled_signals:
            signal.signal(value, request_stop)

        argv = child["argv"]
        if not isinstance(argv, list):
            raise AssertionError("validated child argv changed")
        _enable_child_subreaper()
        if (
            runtime_control_group is not None
            and _cgroup_other_pids(runtime_control_group)
        ):
            raise PermissionError("unit cgroup is not empty before Popen")
        if _requires_systemd_commit(spec):
            systemd = runtime["systemd"]
            if not isinstance(systemd, Mapping):
                raise AssertionError("validated systemd contract changed")
            invocation_id = str(
                materialization_claim["systemd_invocation_id"]
            )
            start_ack_receipt = _wait_for_start_ack_receipt(
                spec,
                expected_invocation_id=invocation_id,
            )
            prespawn_shadow = _query_systemd_shadow(
                str(systemd["unit_name"])
            )
            validate_systemd_shadow(spec, prespawn_shadow)
            prespawn_state = _query_systemd_phase_state(
                str(systemd["unit_name"])
            )
            _validate_systemd_phase_state(
                prespawn_state,
                phase="child_prespawn",
                expected_invocation_id=invocation_id,
            )
            launch_lease = _verify_launch_lease(spec)
            prespawn_environment: dict[str, object] | None = None
            prespawn_gpu: dict[str, object] | None = None
            if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
                if attempt_commit is None:
                    raise PermissionError("actual prespawn has no attempt commit")
                prespawn_gpu = _verify_active_gpu_lease(spec, attempt_commit)
                try:
                    prespawn_environment = _verify_live_environment(
                        spec,
                        phase="child_prespawn",
                    )
                    child_prespawn_receipt = _write_phase_receipt(
                        spec,
                        phase="child_prespawn",
                        phase_state=prespawn_state,
                        immutable_shadow=prespawn_shadow,
                        path=Path(
                            str(artifacts["child_prespawn_phase_receipt"])
                        ),
                        launch_lease=launch_lease,
                        environment_audit=prespawn_environment,
                        gpu_lease=prespawn_gpu,
                    )
                    if authorization is None:
                        raise PermissionError(
                            "actual prespawn has no runtime authorization"
                        )
                    _write_runtime_attestation(
                        spec,
                        spec_path=spec_path,
                        authorization=authorization,
                        attempt_commit=attempt_commit,
                        materialization_claim=materialization_claim,
                        gpu_lease=prespawn_gpu,
                    )
                    dynamic_environment = {
                        _RUNTIME_ATTESTATION_ENV: str(
                            artifacts["runtime_attestation"]
                        )
                    }
                finally:
                    prespawn_gpu["handle"].close_without_release()
            else:
                child_prespawn_receipt = _write_phase_receipt(
                    spec,
                    phase="child_prespawn",
                    phase_state=prespawn_state,
                    immutable_shadow=prespawn_shadow,
                    path=Path(
                        str(artifacts["child_prespawn_phase_receipt"])
                    ),
                    launch_lease=launch_lease,
                )
        if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
            _validate_runtime_filesystem(spec)
        # This is intentionally the only Popen call in the implementation.
        process = subprocess.Popen(
            list(argv),
            shell=False,
            start_new_session=True,
            cwd=str(child["cwd"]),
            env=_child_environment(child, dynamic=dynamic_environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
        )
        heartbeat = _HeartbeatChain(
            directory=heartbeat_dir,
            materialization_claim_sha256=materialization_sha256,
            attempt_id=str(spec["attempt_id"]),
            systemd_invocation_id=str(
                materialization_claim["systemd_invocation_id"]
            ),
            child_pid=process.pid,
        )
        heartbeat.emit(
            "child_started",
            child_pid=process.pid,
            details={"child_argv_fingerprint": child["argv_fingerprint"]},
        )

        heartbeat_interval = float(
            runtime["heartbeat_interval_seconds"]
        )
        poll_interval = float(runtime["poll_interval_seconds"])
        grace_seconds = float(runtime["termination_grace_seconds"])
        next_heartbeat = time.monotonic() + heartbeat_interval
        termination_deadline: float | None = None

        while process.poll() is None:
            while pending_signals:
                signum = pending_signals.pop(0)
                if forwarded_signals:
                    forwarded = int(signal.SIGKILL)
                    forced_kill = True
                else:
                    forwarded = signum
                    termination_deadline = time.monotonic() + grace_seconds
                if _signal_process_group(process, forwarded):
                    forwarded_signals.append(forwarded)
                    heartbeat.emit(
                        "signal_forwarded",
                        child_pid=process.pid,
                        details={
                            "signal_number": forwarded,
                            "signal_name": signal.Signals(forwarded).name,
                        },
                    )

            if (
                termination_deadline is not None
                and time.monotonic() >= termination_deadline
                and process.poll() is None
                and not forced_kill
            ):
                if _signal_process_group(process, int(signal.SIGKILL)):
                    forwarded_signals.append(int(signal.SIGKILL))
                    forced_kill = True
                    heartbeat.emit(
                        "termination_grace_expired",
                        child_pid=process.pid,
                        details={
                            "signal_number": int(signal.SIGKILL),
                            "signal_name": signal.SIGKILL.name,
                        },
                    )
            if time.monotonic() >= next_heartbeat:
                heartbeat.emit(
                    "child_running",
                    child_pid=process.pid,
                    details={
                        "termination_requested": bool(
                            forwarded_signals
                        )
                    },
                )
                next_heartbeat = time.monotonic() + heartbeat_interval
            time.sleep(poll_interval)
        return_code = process.wait()
        heartbeat.emit(
            "child_reaped",
            child_pid=process.pid,
            details={"raw_return_code": return_code},
        )
        cleanup_signals = _quiesce_runtime_descendants(
            process.pid,
            runtime_control_group,
            grace_seconds=grace_seconds,
            poll_interval=poll_interval,
        )
        process_group_cleanup_signals.extend(cleanup_signals)
        if cleanup_signals:
            heartbeat.emit(
                "runtime_descendants_quiesced",
                child_pid=process.pid,
                details={
                    "cleanup_signals": process_group_cleanup_signals
                },
            )
        logs_safe_to_finalize = True
    except BaseException as error:
        if process is None:
            spawn_error = error
        else:
            supervisor_error = error
            try:
                if process.poll() is None:
                    _signal_process_group(process, int(signal.SIGTERM))
                    forwarded_signals.append(int(signal.SIGTERM))
                    deadline = time.monotonic() + float(
                        runtime["termination_grace_seconds"]
                    )
                    while (
                        process.poll() is None
                        and time.monotonic() < deadline
                    ):
                        time.sleep(float(runtime["poll_interval_seconds"]))
                    if process.poll() is None:
                        _signal_process_group(process, int(signal.SIGKILL))
                        forwarded_signals.append(int(signal.SIGKILL))
                        forced_kill = True
                process.wait()
            except BaseException as termination_error:
                supervisor_error = supervisor_error or termination_error
            try:
                cleanup_signals = _quiesce_runtime_descendants(
                    process.pid,
                    runtime_control_group,
                    grace_seconds=float(
                        runtime["termination_grace_seconds"]
                    ),
                    poll_interval=float(runtime["poll_interval_seconds"]),
                )
                process_group_cleanup_signals.extend(cleanup_signals)
                logs_safe_to_finalize = True
            except BaseException as cleanup_error:
                supervisor_error = supervisor_error or cleanup_error
    finally:
        for value, previous in previous_handlers.items():
            signal.signal(value, previous)
        if stdout_handle is not None and logs_safe_to_finalize:
            try:
                stdout_receipt = _finalize_log(
                    stdout_handle,
                    stdout_path,
                )
            except BaseException as error:
                supervisor_error = supervisor_error or error
        elif stdout_handle is not None:
            try:
                stdout_handle.close()
            except BaseException as error:
                supervisor_error = supervisor_error or error
        if stderr_handle is not None and logs_safe_to_finalize:
            try:
                stderr_receipt = _finalize_log(
                    stderr_handle,
                    stderr_path,
                )
            except BaseException as error:
                supervisor_error = supervisor_error or error
        elif stderr_handle is not None:
            try:
                stderr_handle.close()
            except BaseException as error:
                supervisor_error = supervisor_error or error

    return_code = process.returncode if process is not None else None
    child_outcome = classify_child_exit(
        return_code,
        spawn_error=spawn_error,
        forwarded_signals=forwarded_signals,
        forced_kill=forced_kill,
    )
    terminal = _terminal_payload(
        spec,
        materialization_claim=materialization_claim,
        materialization_claim_sha256=materialization_sha256,
        child_outcome=child_outcome,
        heartbeat=heartbeat,
        stdout_receipt=stdout_receipt,
        stderr_receipt=stderr_receipt,
        start_ack_receipt=start_ack_receipt,
        child_prespawn_receipt=child_prespawn_receipt,
        supervisor_error=supervisor_error,
        process_group_cleanup_signals=process_group_cleanup_signals,
    )
    _write_new_json(terminal_path, terminal)
    if spawn_error is not None:
        return os.EX_OSERR
    if supervisor_error is not None:
        return os.EX_SOFTWARE
    if return_code is None:
        return os.EX_SOFTWARE
    return _normalized_process_exit(return_code)


def validate_systemd_shadow(
    spec: Mapping[str, object],
    observed: Mapping[str, str],
) -> None:
    runtime = spec["runtime"]
    if not isinstance(runtime, Mapping):
        raise AssertionError("validated runtime contract changed")
    systemd = runtime["systemd"]
    if not isinstance(systemd, Mapping):
        raise AssertionError("validated systemd contract changed")
    expected = systemd["immutable_shadow_properties"]
    if not isinstance(expected, Mapping):
        raise AssertionError("validated systemd shadow changed")
    expected_map = dict(expected)
    observed_map = dict(observed)
    differences: dict[str, object] = {}
    for name in sorted(set(expected_map) | set(observed_map)):
        if name not in expected_map:
            differences[name] = {
                "kind": "unexpected-property",
                "observed": observed_map[name],
            }
            continue
        if name not in observed_map:
            differences[name] = {
                "kind": "missing-property",
                "expected": expected_map[name],
            }
            continue
        if name in _SYSTEMD_EXEC_KEYS:
            try:
                expected_identity = _normalized_systemd_exec_identity(
                    str(expected_map[name])
                )
                observed_identity = _normalized_systemd_exec_identity(
                    str(observed_map[name])
                )
            except ValueError as error:
                differences[name] = {
                    "kind": "malformed-exec-identity",
                    "expected": expected_map[name],
                    "observed": observed_map[name],
                    "error": str(error),
                }
            else:
                if observed_identity != expected_identity:
                    differences[name] = {
                        "kind": "exec-identity-changed",
                        "expected": expected_identity,
                        "observed": observed_identity,
                    }
            continue
        expected_value = _normalize_systemd_shadow_value(
            name,
            str(expected_map[name]),
        )
        observed_value = _normalize_systemd_shadow_value(
            name,
            str(observed_map[name]),
        )
        if observed_value != expected_value:
            differences[name] = {
                "kind": "static-property-changed",
                "expected": expected_value,
                "observed": observed_value,
            }
    if differences:
        diagnostic = {
            "expected_fingerprint": stable_fingerprint(expected_map),
            "observed_fingerprint": stable_fingerprint(observed_map),
            "differences": differences,
        }
        raise PermissionError(
            "loaded systemd unit differs from frozen shadow:"
            + canonical_json(diagnostic)
        )



def _fixed_verified_manager_environment() -> dict[str, str]:
    """Return a non-inherited environment for one verified user manager."""

    uid = os.getuid()
    runtime = Path(f"/run/user/{uid}")
    if (
        not runtime.is_dir()
        or runtime.is_symlink()
        or runtime.resolve(strict=True) != runtime
    ):
        raise PermissionError("user runtime directory is not canonical")
    runtime_metadata = runtime.stat()
    if (
        runtime_metadata.st_uid != uid
        or stat.S_IMODE(runtime_metadata.st_mode) != 0o700
    ):
        raise PermissionError("user runtime directory is not private and owned")
    bus = runtime / "bus"
    bus_metadata = bus.lstat()
    if (
        not stat.S_ISSOCK(bus_metadata.st_mode)
        or bus_metadata.st_uid != uid
        or bus.is_symlink()
        or bus.resolve(strict=True) != bus
    ):
        raise PermissionError("user manager D-Bus endpoint is not trusted")
    return {
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={bus}",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "SYSTEMD_COLORS": "0",
        "SYSTEMD_PAGER": "",
        "SYSTEMD_PAGERSECURE": "1",
        "SYSTEMD_URLIFY": "0",
        "XDG_RUNTIME_DIR": str(runtime),
    }

def _query_systemd_shadow(unit_name: str) -> dict[str, str]:
    """Query only immutable unit identity, never mutable phase state."""

    command = [
        _SYSTEMCTL_PATH,
        "--user",
        "show",
        unit_name,
        "--no-pager",
        *[
            f"--property={name}"
            for name in sorted(_SYSTEMD_IMMUTABLE_SHADOW_KEYS)
        ],
    ]
    completed = subprocess.run(
        command,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        env=_fixed_verified_manager_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError("systemctl show failed before attempt commit")
    observed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in observed:
            raise ValueError("systemctl returned a duplicate property")
        observed[key] = _normalize_systemd_shadow_value(
            key, value
        )
    if set(observed) != _SYSTEMD_IMMUTABLE_SHADOW_KEYS:
        raise ValueError("systemctl did not return the immutable unit shadow")
    return observed


def _query_systemd_phase_state(unit_name: str) -> dict[str, str]:
    command = [
        _SYSTEMCTL_PATH,
        "--user",
        "show",
        unit_name,
        "--no-pager",
        *[
            f"--property={name}"
            for name in sorted(_SYSTEMD_PHASE_STATE_KEYS)
        ],
    ]
    completed = subprocess.run(
        command,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        env=_fixed_verified_manager_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError("systemctl phase-state query failed")
    observed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in observed:
            raise ValueError("systemctl returned duplicate phase state")
        observed[key] = value
    if set(observed) != _SYSTEMD_PHASE_STATE_KEYS:
        raise ValueError("systemctl phase-state response is incomplete")
    return observed


def _query_systemd_runtime_identity(unit_name: str) -> dict[str, str]:
    command = [
        _SYSTEMCTL_PATH,
        "--user",
        "show",
        unit_name,
        "--no-pager",
        "--property=InvocationID",
        "--property=ControlGroup",
    ]
    completed = subprocess.run(
        command,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        env=_fixed_verified_manager_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError("systemctl identity query failed")
    observed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in observed:
            raise ValueError("systemctl returned duplicate identity")
        observed[key] = value
    if set(observed) != {"InvocationID", "ControlGroup"}:
        raise ValueError("systemctl identity response is incomplete")
    invocation_id = observed["InvocationID"]
    control_group = observed["ControlGroup"]
    if (
        len(invocation_id) != 32
        or any(
            character not in "0123456789abcdef"
            for character in invocation_id
        )
        or not control_group.startswith("/")
        or control_group == "/"
    ):
        raise PermissionError("systemd runtime identity is malformed")
    return {
        "invocation_id": invocation_id,
        "control_group": control_group,
    }


def _self_cgroup_path() -> str:
    raw = Path("/proc/self/cgroup").read_text(encoding="ascii")
    candidates: list[str] = []
    for line in raw.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            candidates.append(parts[2])
    if (
        len(candidates) != 1
        or not candidates[0].startswith("/")
    ):
        raise PermissionError("unified process cgroup is unavailable")
    return candidates[0]


def _validate_live_systemd_context(
    spec: Mapping[str, object],
    systemd_invocation_id: str,
) -> str:
    runtime = spec["runtime"]
    if not isinstance(runtime, Mapping):
        raise AssertionError("validated runtime contract changed")
    systemd = runtime["systemd"]
    if not isinstance(systemd, Mapping):
        raise AssertionError("validated systemd contract changed")
    identity = _query_systemd_runtime_identity(
        str(systemd["unit_name"])
    )
    if identity["invocation_id"] != systemd_invocation_id:
        raise PermissionError("INVOCATION_ID differs from systemd manager")
    control_group = identity["control_group"]
    if _self_cgroup_path() != control_group:
        raise PermissionError("supervisor is outside the frozen unit cgroup")
    return control_group


def _verify_attempt_commit_lineage(
    spec: Mapping[str, object],
) -> dict[str, object]:
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise AssertionError("validated artifact contract changed")
    path = Path(str(artifacts["attempt_commit"]))
    payload = _read_canonical_json(path, name="r2 attempt commit")
    body = dict(payload)
    fingerprint = body.pop("attempt_commit_fingerprint", None)
    runtime = spec["runtime"]
    if not isinstance(runtime, Mapping) or not isinstance(
        runtime["systemd"], Mapping
    ):
        raise AssertionError("validated systemd contract changed")
    expected_shadow = runtime["systemd"]["immutable_shadow_properties"]
    precommit_path = Path(str(artifacts["precommit_phase_receipt"]))
    precommit = _verify_phase_receipt(
        spec,
        path=precommit_path,
        expected_phase="precommit",
    )
    precommit_state = precommit["systemd_phase_state"]
    if not isinstance(precommit_state, Mapping):
        raise PermissionError("precommit phase state is invalid")
    _validate_systemd_phase_state(precommit_state, phase="precommit")
    lease = _verify_launch_lease(spec)
    precommit_fingerprint = precommit[
        "phase_receipt_fingerprint"
    ]
    lease_fingerprint = lease["launch_lease_fingerprint"]
    if (
        precommit.get("immutable_shadow_fingerprint")
        != stable_fingerprint(expected_shadow)
        or precommit.get("launch_lease_fingerprint")
        != lease_fingerprint
    ):
        raise PermissionError("precommit phase receipt lineage is invalid")
    expected_authorization_required = (
        spec["execution_kind"] == ACTUAL_EXECUTION_KIND
    )
    expected_planned_fingerprint = (
        _planned_attempt_commit_fingerprint(spec, payload)
        if (
            expected_authorization_required
            and _is_sha256(payload.get("authorization_fingerprint"))
        )
        else None
    )
    gpu_lease_identity_fields = (
        "gpu_lease_fingerprint",
        "gpu_lease_file_sha256",
        "gpu_lease_device",
        "gpu_lease_inode",
        "gpu_lease_parent_device",
        "gpu_lease_parent_inode",
    )
    if (
        set(payload) != _ATTEMPT_COMMIT_KEYS
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
        or payload.get("schema_version") != ATTEMPT_COMMIT_SCHEMA
        or payload.get("execution_kind") != spec["execution_kind"]
        or payload.get("candidate") != spec["candidate"]
        or payload.get("stage_id") != spec["stage_id"]
        or payload.get("attempt_id") != spec["attempt_id"]
        or payload.get("attempt_ordinal") != spec["attempt_ordinal"]
        or payload.get("prior_attempt_count") != spec["prior_attempt_count"]
        or payload.get("runtime_spec_fingerprint")
        != spec["runtime_spec_fingerprint"]
        or (
            expected_authorization_required
            and not _is_sha256(payload.get("authorization_fingerprint"))
        )
        or (
            expected_authorization_required
            and not _is_sha256(payload.get("authorization_file_sha256"))
        )
        or (
            not expected_authorization_required
            and payload.get("authorization_fingerprint") is not None
        )
        or (
            not expected_authorization_required
            and payload.get("authorization_file_sha256") is not None
        )
        or payload.get("boot_id") != _boot_id()
        or not _is_positive_int(payload.get("committer_pid"))
        or not _is_positive_int(
            payload.get("committer_proc_starttime_ticks")
        )
        or payload.get("systemd_unit_name")
        != runtime["systemd"]["unit_name"]
        or payload.get("immutable_systemd_shadow_properties")
        != expected_shadow
        or payload.get("immutable_systemd_shadow_fingerprint")
        != stable_fingerprint(expected_shadow)
        or payload.get("precommit_phase_receipt_fingerprint")
        != precommit_fingerprint
        or payload.get("precommit_phase_receipt_file_sha256")
        != precommit["phase_receipt_file_sha256"]
        or payload.get("launch_lease_fingerprint") != lease_fingerprint
        or payload.get("launch_lease_file_sha256")
        != lease["launch_lease_file_sha256"]
        or payload.get("dispatch_lease_scope") != "attempt_dispatch_only"
        or (
            expected_authorization_required
            and payload.get("planned_attempt_commit_fingerprint")
            != expected_planned_fingerprint
        )
        or (
            not expected_authorization_required
            and payload.get("planned_attempt_commit_fingerprint") is not None
        )
        or (
            expected_authorization_required
            and payload.get("runtime_environment_audit_valid") is not True
        )
        or (
            expected_authorization_required
            and payload.get("precommit_environment_audit_fingerprint")
            != precommit.get("environment_audit_fingerprint")
        )
        or (
            expected_authorization_required
            and payload.get("precommit_environment_inventory_fingerprint")
            != precommit.get("environment_inventory_fingerprint")
        )
        or (
            expected_authorization_required
            and payload.get("gpu_lease_fingerprint")
            != precommit.get("gpu_lease_fingerprint")
        )
        or (
            expected_authorization_required
            and not _is_sha256(payload.get("gpu_lease_file_sha256"))
        )
        or (
            expected_authorization_required
            and (
                any(
                    not _is_positive_int(payload.get(field))
                    for field in (
                        "gpu_lease_device",
                        "gpu_lease_inode",
                        "gpu_lease_parent_device",
                        "gpu_lease_parent_inode",
                    )
                )
            )
        )
        or (
            not expected_authorization_required
            and any(
                payload.get(field) is not None
                for field in gpu_lease_identity_fields
            )
        )
        or payload.get("launch_limit") != 1
        or payload.get("automatic_retry_allowed") is not False
        or payload.get("resume_allowed") is not False
    ):
        raise PermissionError("r2 attempt commit is invalid")
    return {
        **payload,
        "attempt_commit_file_sha256": sealed_file_sha256(payload),
    }


def _verify_attempt_commit(
    spec: Mapping[str, object],
    authorization: Mapping[str, object],
) -> dict[str, object]:
    payload = _verify_attempt_commit_lineage(spec)
    if (
        payload.get("authorization_fingerprint")
        != authorization["authorization_fingerprint"]
        or payload.get("authorization_file_sha256")
        != authorization["authorization_file_sha256"]
    ):
        raise PermissionError(
            "r2 attempt commit does not match current authorization"
        )
    return payload


def _write_consumed_start_failure(
    spec: Mapping[str, object],
    *,
    category: str,
    return_code: int | None,
    stdout: str,
    stderr: str,
    error_type: str | None,
) -> None:
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise AssertionError("validated artifact contract changed")
    body = {
        "schema_version": CONSUMED_START_FAILURE_SCHEMA,
        "execution_kind": spec["execution_kind"],
        "candidate": spec["candidate"],
        "stage_id": spec["stage_id"],
        "attempt_id": spec["attempt_id"],
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
        "category": category,
        "systemctl_return_code": return_code,
        "systemctl_stdout_sha256": hashlib.sha256(
            stdout.encode("utf-8")
        ).hexdigest(),
        "systemctl_stderr_sha256": hashlib.sha256(
            stderr.encode("utf-8")
        ).hexdigest(),
        "error_type": error_type,
        "time_utc": _utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        "boot_id": _boot_id(),
        "attempt_consumed": True,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
        "scientific_gate_passed": None,
    }
    payload = _seal(
        body,
        fingerprint_field="consumed_start_failure_fingerprint",
    )
    _write_new_json(
        Path(str(artifacts["consumed_start_failure_receipt"])),
        payload,
    )


def _await_start_ack(
    spec: Mapping[str, object],
    *,
    unit_name: str,
    prior_invocation_id: str,
    immutable_shadow: Mapping[str, str],
    launch_lease: Mapping[str, object],
) -> dict[str, object]:
    runtime = spec["runtime"]
    artifacts = spec["artifacts"]
    if not isinstance(runtime, Mapping) or not isinstance(artifacts, Mapping):
        raise AssertionError("validated runtime spec changed")
    systemd = runtime["systemd"]
    if not isinstance(systemd, Mapping):
        raise AssertionError("validated systemd contract changed")
    deadline = time.monotonic() + float(
        systemd["start_ack_timeout_seconds"]
    )
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            phase_state = _query_systemd_phase_state(unit_name)
            _validate_systemd_phase_state(
                phase_state,
                phase="start_ack",
                prior_invocation_id=prior_invocation_id,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            last_error = error
        else:
            return _write_phase_receipt(
                spec,
                phase="start_ack",
                phase_state=phase_state,
                immutable_shadow=immutable_shadow,
                path=Path(str(artifacts["start_ack_receipt"])),
                launch_lease=launch_lease,
            )
        time.sleep(float(systemd["start_ack_poll_seconds"]))
    raise RuntimeError("bounded systemd start acknowledgement failed") from last_error


def commit_and_start(spec_path: str | Path) -> int:
    """Commit one systemd attempt, then issue one nonblocking start."""

    canonical_spec_path = _canonical_regular_file(
        spec_path,
        name="runtime supervisor spec",
    )
    spec = load_runtime_spec(canonical_spec_path)
    if (
        spec["execution_kind"] == ACTUAL_EXECUTION_KIND
        and str(canonical_spec_path) != _ACTUAL_SPEC_PATH
    ):
        raise PermissionError("actual runtime spec path is not exact")
    if not _requires_systemd_commit(spec):
        raise PermissionError("commit-and-start requires a systemd execution")
    # Authorization absence rejects before filesystem validation, artifacts,
    # systemctl inspection, or any process launch.
    authorization: dict[str, object] | None = None
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        authorization = _verify_actual_authorization(
            spec,
            spec_path=canonical_spec_path,
        )
    _validate_runtime_filesystem(spec)
    _validate_precommit_artifacts(spec)
    runtime = spec["runtime"]
    artifacts = spec["artifacts"]
    if not isinstance(runtime, Mapping) or not isinstance(artifacts, Mapping):
        raise AssertionError("validated runtime spec changed")
    systemd = runtime["systemd"]
    if not isinstance(systemd, Mapping):
        raise AssertionError("validated systemd contract changed")
    unit_name = str(systemd["unit_name"])
    observed = _query_systemd_shadow(unit_name)
    validate_systemd_shadow(spec, observed)
    preliminary_phase = _query_systemd_phase_state(unit_name)
    _validate_systemd_phase_state(preliminary_phase, phase="precommit")
    planned_fingerprint: str | None = None
    gpu_handle: object | None = None
    gpu_lease: dict[str, object] | None = None
    prelease_audit: dict[str, object] | None = None
    postlease_audit: dict[str, object] | None = None
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        if authorization is None:
            raise AssertionError("actual authorization disappeared")
        prelease_audit = _verify_live_environment(spec, phase="prelease")
        planned_fingerprint = _planned_attempt_commit_fingerprint(
            spec,
            authorization,
        )
        gpu_handle = _acquire_external_gpu_lease(
            spec,
            authorization,
            planned_attempt_commit_fingerprint=planned_fingerprint,
        )
        try:
            gpu_lease = _external_gpu_lease_evidence(gpu_handle)
        except BaseException:
            if not gpu_handle.closed:
                gpu_handle.close_without_release()
            raise
    lease: dict[str, object] | None = None
    evidence_write_started = False
    commit_write_started = False
    try:
        lease = _create_launch_lease(spec, authorization)
        final_observed = _query_systemd_shadow(unit_name)
        validate_systemd_shadow(spec, final_observed)
        final_phase = _query_systemd_phase_state(unit_name)
        _validate_systemd_phase_state(final_phase, phase="precommit")
        if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
            postlease_audit = _verify_live_environment(
                spec,
                phase="postlease",
            )
        evidence_write_started = True
        precommit_receipt = _write_phase_receipt(
            spec,
            phase="precommit",
            phase_state=final_phase,
            immutable_shadow=final_observed,
            path=Path(str(artifacts["precommit_phase_receipt"])),
            launch_lease=lease,
            environment_audit=postlease_audit,
            gpu_lease=gpu_lease,
        )
        commit = _attempt_commit_payload(
            spec,
            authorization,
            final_observed,
            precommit_receipt,
            lease,
            planned_attempt_commit_fingerprint=planned_fingerprint,
            environment_audit=postlease_audit,
            gpu_lease=gpu_lease,
        )
        commit_write_started = True
        _write_new_json(Path(str(artifacts["attempt_commit"])), commit)
    except BaseException:
        if lease is not None and not evidence_write_started:
            _release_uncommitted_launch_lease(spec, lease)
        if gpu_lease is not None:
            if not commit_write_started:
                release_evidence = (
                    postlease_audit or prelease_audit or {
                        "environment_audit_fingerprint": planned_fingerprint
                    }
                )
                try:
                    _release_external_gpu_lease(
                        spec,
                        gpu_lease,
                        release_kind="uncommitted_forensic",
                        attempt_consumed=False,
                        evidence_fingerprint=str(
                            release_evidence["environment_audit_fingerprint"]
                        ),
                    )
                except BaseException:
                    if not gpu_handle.closed:
                        gpu_handle.close_without_release()
            elif not gpu_handle.closed:
                gpu_handle.close_without_release()
        raise
    if gpu_handle is not None and not gpu_handle.closed:
        gpu_handle.close_without_release()

    # There is intentionally no loop and no retry around this exact start.
    try:
        completed = subprocess.run(
            [_SYSTEMCTL_PATH, "--user", "start", "--no-block", unit_name],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            env=_fixed_verified_manager_environment(),
        )
    except BaseException as error:
        _write_consumed_start_failure(
            spec,
            category="SYSTEMCTL_START_RAISED",
            return_code=None,
            stdout="",
            stderr="",
            error_type=type(error).__name__,
        )
        raise
    if completed.returncode != 0:
        _write_consumed_start_failure(
            spec,
            category="SYSTEMCTL_START_RETURNED_NONZERO",
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error_type=None,
        )
        raise RuntimeError("systemctl start failed after attempt commit")
    try:
        _await_start_ack(
            spec,
            unit_name=unit_name,
            prior_invocation_id=final_phase["InvocationID"],
            immutable_shadow=final_observed,
            launch_lease=lease,
        )
    except BaseException as error:
        _write_consumed_start_failure(
            spec,
            category="BOUNDED_START_ACK_FAILED",
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error_type=type(error).__name__,
        )
        raise
    return 0


def _systemd_invocation_id(environment: Mapping[str, str]) -> str:
    invocation_id = environment.get("INVOCATION_ID")
    if (
        not isinstance(invocation_id, str)
        or len(invocation_id) != 32
        or any(
            character not in "0123456789abcdef"
            for character in invocation_id
        )
    ):
        raise PermissionError("systemd INVOCATION_ID is absent or malformed")
    return invocation_id


def claim_materialization(
    spec_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    """ExecCondition entry point: publish the invocation-bound claim."""

    canonical_spec_path = _canonical_regular_file(
        spec_path,
        name="runtime supervisor spec",
    )
    spec = load_runtime_spec(canonical_spec_path)
    control_group: str | None = None
    authorization: dict[str, object] | None = None
    attempt_commit: dict[str, object] | None = None
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        # Missing actual authorization is rejected before any artifact.
        authorization = _verify_actual_authorization(
            spec,
            spec_path=canonical_spec_path,
        )
    _validate_runtime_filesystem(spec)
    if _requires_systemd_commit(spec):
        attempt_commit = (
            _verify_attempt_commit(spec, authorization)
            if authorization is not None
            else _verify_attempt_commit_lineage(spec)
        )
        runtime = spec["runtime"]
        if not isinstance(runtime, Mapping) or not isinstance(
            runtime["systemd"], Mapping
        ):
            raise AssertionError("validated systemd contract changed")
        observed = _query_systemd_shadow(
            str(runtime["systemd"]["unit_name"])
        )
        validate_systemd_shadow(spec, observed)
    invocation_id = _systemd_invocation_id(
        dict(os.environ) if environment is None else environment
    )
    if _requires_systemd_commit(spec):
        control_group = _validate_live_systemd_context(
            spec, invocation_id
        )
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise AssertionError("validated artifact contract changed")
    claim = _materialization_claim(
        spec,
        authorization,
        attempt_commit,
        invocation_id,
        control_group,
    )
    _write_new_json(
        Path(str(artifacts["materialization_claim"])),
        claim,
    )
    return 0


def _verify_materialization_claim(
    spec: Mapping[str, object],
    *,
    authorization: Mapping[str, object] | None,
    attempt_commit: Mapping[str, object] | None,
    systemd_invocation_id: str,
    systemd_control_group: str | None,
) -> dict[str, object]:
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise AssertionError("validated artifact contract changed")
    path = Path(str(artifacts["materialization_claim"]))
    payload = _read_canonical_json(path, name="materialization claim")
    body = dict(payload)
    fingerprint = body.pop("materialization_claim_fingerprint", None)
    expected_authorization = (
        authorization["authorization_fingerprint"]
        if authorization is not None
        else None
    )
    expected_commit = (
        attempt_commit["attempt_commit_fingerprint"]
        if attempt_commit is not None
        else None
    )
    if (
        not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
        or payload.get("schema_version") != MATERIALIZATION_CLAIM_SCHEMA
        or payload.get("runtime_spec_fingerprint")
        != spec["runtime_spec_fingerprint"]
        or payload.get("authorization_fingerprint")
        != expected_authorization
        or payload.get("attempt_commit_fingerprint") != expected_commit
        or payload.get("systemd_invocation_id") != systemd_invocation_id
        or payload.get("systemd_control_group")
        != systemd_control_group
        or payload.get("boot_id") != _boot_id()
        or not isinstance(
            payload.get("claimer_proc_starttime_ticks"),
            int,
        )
        or payload.get("launch_limit") != 1
        or payload.get("shell") is not False
        or payload.get("automatic_retry_allowed") is not False
        or payload.get("resume_allowed") is not False
    ):
        raise PermissionError("materialization claim is invalid")
    return {
        **payload,
        "materialization_claim_file_sha256": sealed_file_sha256(payload),
    }


def run_once(
    spec_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Service entry point: claim materialization, then Popen exactly once."""

    canonical_spec_path = _canonical_regular_file(
        spec_path,
        name="runtime supervisor spec",
    )
    spec = load_runtime_spec(canonical_spec_path)
    authorization: dict[str, object] | None = None
    attempt_commit: dict[str, object] | None = None
    control_group: str | None = None
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        # This remains before any attempt artifact, log, signal setup, or Popen.
        authorization = _verify_actual_authorization(
            spec,
            spec_path=canonical_spec_path,
        )
    _validate_runtime_filesystem(spec)
    if _requires_systemd_commit(spec):
        attempt_commit = (
            _verify_attempt_commit(spec, authorization)
            if authorization is not None
            else _verify_attempt_commit_lineage(spec)
        )
        runtime = spec["runtime"]
        if not isinstance(runtime, Mapping) or not isinstance(
            runtime["systemd"], Mapping
        ):
            raise AssertionError("validated systemd contract changed")
        observed = _query_systemd_shadow(
            str(runtime["systemd"]["unit_name"])
        )
        validate_systemd_shadow(spec, observed)
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        if attempt_commit is None:
            raise PermissionError("actual run_once has no attempt commit")
        entry_gpu_lease = _verify_active_gpu_lease(spec, attempt_commit)
        entry_gpu_lease["handle"].close_without_release()
    invocation_id = _systemd_invocation_id(
        dict(os.environ) if environment is None else environment
    )
    if _requires_systemd_commit(spec):
        control_group = _validate_live_systemd_context(
            spec, invocation_id
        )
    materialization_claim = _verify_materialization_claim(
        spec,
        authorization=authorization,
        attempt_commit=attempt_commit,
        systemd_invocation_id=invocation_id,
        systemd_control_group=control_group,
    )
    return _run_child_once(
        spec,
        spec_path=canonical_spec_path,
        authorization=authorization,
        attempt_commit=attempt_commit,
        materialization_claim=materialization_claim,
    )


def verify_runtime_spec(
    spec_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    """ExecStartPre entry point with no artifact or child mutation."""

    canonical_spec_path = _canonical_regular_file(
        spec_path,
        name="runtime supervisor spec",
    )
    spec = load_runtime_spec(canonical_spec_path)
    authorization: dict[str, object] | None = None
    attempt_commit: dict[str, object] | None = None
    control_group: str | None = None
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        authorization = _verify_actual_authorization(
            spec,
            spec_path=canonical_spec_path,
        )
    _validate_runtime_filesystem(spec)
    if _requires_systemd_commit(spec):
        attempt_commit = (
            _verify_attempt_commit(spec, authorization)
            if authorization is not None
            else _verify_attempt_commit_lineage(spec)
        )
        runtime = spec["runtime"]
        if not isinstance(runtime, Mapping) or not isinstance(
            runtime["systemd"], Mapping
        ):
            raise AssertionError("validated systemd contract changed")
        observed = _query_systemd_shadow(
            str(runtime["systemd"]["unit_name"])
        )
        validate_systemd_shadow(spec, observed)
    invocation_id = _systemd_invocation_id(
        dict(os.environ) if environment is None else environment
    )
    if _requires_systemd_commit(spec):
        control_group = _validate_live_systemd_context(
            spec, invocation_id
        )
    _verify_materialization_claim(
        spec,
        authorization=authorization,
        attempt_commit=attempt_commit,
        systemd_invocation_id=invocation_id,
        systemd_control_group=control_group,
    )
    return 0


def classify_systemd_exit(
    environment: Mapping[str, str],
) -> dict[str, object]:
    service_result = environment.get("SERVICE_RESULT", "unknown")
    exit_code = environment.get("EXIT_CODE", "unknown")
    exit_status = environment.get("EXIT_STATUS", "unknown")
    categories = {
        "exit-code": "SYSTEMD_MAIN_EXIT_NONZERO",
        "signal": "SYSTEMD_MAIN_SIGNAL",
        "core-dump": "SYSTEMD_MAIN_CORE_DUMP",
        "timeout": "SYSTEMD_TIMEOUT",
        "watchdog": "SYSTEMD_WATCHDOG",
        "oom-kill": "SYSTEMD_OOM_KILL",
        "resources": "SYSTEMD_RESOURCE_FAILURE",
        "protocol": "SYSTEMD_PROTOCOL_FAILURE",
        "start-limit-hit": "SYSTEMD_START_LIMIT_HIT",
        "exec-condition": "SYSTEMD_EXEC_CONDITION",
    }
    success = (
        service_result == "success"
        and exit_code == "exited"
        and exit_status == "0"
    )
    category = (
        "SYSTEMD_SERVICE_SUCCESS"
        if success
        else categories.get(service_result, "SYSTEMD_OTHER_FAILURE")
    )
    return {
        "category": category,
        "service_result": service_result,
        "exit_code": exit_code,
        "exit_status": exit_status,
        "invocation_id": environment.get("INVOCATION_ID"),
        "systemd_success": success,
        "scientific_gate_passed": None,
    }


def finalize_systemd(
    spec_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Publish manager-level exit evidence only for a consumed launch."""

    canonical_spec_path = _canonical_regular_file(
        spec_path,
        name="runtime supervisor spec",
    )
    spec = load_runtime_spec(canonical_spec_path)
    effective_environment = (
        dict(os.environ) if environment is None else environment
    )
    invocation_id = _systemd_invocation_id(effective_environment)
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise AssertionError("validated runtime spec changed")
    root = _canonical_directory(
        Path(str(artifacts["root"])),
        name="artifact root",
    )
    invocation_directory = _canonical_directory(
        Path(str(artifacts["systemd_invocation_dir"])),
        name="systemd invocation directory",
    )
    if invocation_directory.parent != root:
        raise PermissionError("systemd sidecar directory escapes artifact root")
    materialization_path = Path(str(artifacts["materialization_claim"]))
    attempt_commit_path = Path(str(artifacts["attempt_commit"]))
    attempt_commit_required = _requires_systemd_commit(spec)
    live_control_group: str | None = None
    current_authorization_valid: bool | None = None
    current_runtime_closure_valid: bool | None = None
    current_runtime_closure_error_type: str | None = None
    authorization_matches_commit: bool | None = None
    attempt_commit_fingerprint: str | None = None
    attempt_commit_sha256: str | None = None
    attempt_commit_valid = False
    attempt_commit: dict[str, object] | None = None
    if attempt_commit_required:
        live_control_group = _validate_live_systemd_context(
            spec,
            invocation_id,
        )
        if (
            not attempt_commit_path.is_file()
            or attempt_commit_path.is_symlink()
        ):
            raise PermissionError(
                "actual systemd exit has no consumed attempt commit"
            )
        attempt_commit = _verify_attempt_commit_lineage(spec)
        attempt_commit_valid = True
        attempt_commit_fingerprint = str(
            attempt_commit["attempt_commit_fingerprint"]
        )
        attempt_commit_sha256 = str(
            attempt_commit["attempt_commit_file_sha256"]
        )
        try:
            _validate_runtime_filesystem(spec)
        except (OSError, TypeError, ValueError, AssertionError) as error:
            current_runtime_closure_valid = False
            current_runtime_closure_error_type = type(error).__name__
        else:
            current_runtime_closure_valid = True
        if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
            try:
                current_authorization = _verify_actual_authorization(
                    spec,
                    spec_path=canonical_spec_path,
                    require_fresh=False,
                )
            except PermissionError:
                current_authorization_valid = False
                authorization_matches_commit = False
            else:
                current_authorization_valid = True
                authorization_matches_commit = bool(
                    attempt_commit.get("authorization_fingerprint")
                    == current_authorization["authorization_fingerprint"]
                    and attempt_commit.get("authorization_file_sha256")
                    == current_authorization["authorization_file_sha256"]
                )
        else:
            current_authorization_valid = None
            authorization_matches_commit = (
                attempt_commit.get("authorization_fingerprint") is None
                and attempt_commit.get("authorization_file_sha256") is None
            )
    fingerprint: str | None = None
    materialization_sha256: str | None = None
    claim_invocation_id: str | None = None
    claim_valid = False
    if materialization_path.is_file() and not materialization_path.is_symlink():
        materialization = _read_canonical_json(
            materialization_path,
            name="runtime materialization claim",
        )
        body = dict(materialization)
        candidate_fingerprint = body.pop(
            "materialization_claim_fingerprint",
            None,
        )
        claim_invocation_id = materialization.get("systemd_invocation_id")
        expected_authorization_fingerprint = (
            attempt_commit.get("authorization_fingerprint")
            if attempt_commit is not None
            else None
        )
        expected_commit_fingerprint = (
            attempt_commit.get("attempt_commit_fingerprint")
            if attempt_commit is not None
            else None
        )
        claim_valid = bool(
            _is_sha256(candidate_fingerprint)
            and candidate_fingerprint == stable_fingerprint(body)
            and materialization.get("schema_version")
            == MATERIALIZATION_CLAIM_SCHEMA
            and materialization.get("execution_kind")
            == spec["execution_kind"]
            and materialization.get("candidate") == spec["candidate"]
            and materialization.get("stage_id") == spec["stage_id"]
            and materialization.get("attempt_id") == spec["attempt_id"]
            and materialization.get("runtime_spec_fingerprint")
            == spec["runtime_spec_fingerprint"]
            and materialization.get("authorization_fingerprint")
            == expected_authorization_fingerprint
            and materialization.get("attempt_commit_fingerprint")
            == expected_commit_fingerprint
            and materialization.get("boot_id") == _boot_id()
            and materialization.get("systemd_control_group")
            == live_control_group
            and materialization.get("launch_limit") == 1
            and materialization.get("shell") is False
            and materialization.get("automatic_retry_allowed") is False
            and materialization.get("resume_allowed") is False
            and materialization.get("child_argv_fingerprint")
            == spec["child"]["argv_fingerprint"]
        )
        if _is_sha256(candidate_fingerprint):
            fingerprint = str(candidate_fingerprint)
        materialization_sha256 = sealed_file_sha256(materialization)
    claim_matches_invocation = bool(
        claim_valid and claim_invocation_id == invocation_id
    )
    start_ack_fingerprint: str | None = None
    start_ack_sha256: str | None = None
    start_ack_valid = False
    child_prespawn_fingerprint: str | None = None
    child_prespawn_sha256: str | None = None
    child_prespawn_valid = False
    if attempt_commit_required:
        try:
            start_ack = _verify_phase_receipt(
                spec,
                path=Path(str(artifacts["start_ack_receipt"])),
                expected_phase="start_ack",
                expected_invocation_id=invocation_id,
            )
        except (OSError, TypeError, ValueError):
            pass
        else:
            start_ack_valid = True
            start_ack_fingerprint = str(
                start_ack["phase_receipt_fingerprint"]
            )
            start_ack_sha256 = str(start_ack["phase_receipt_file_sha256"])
        try:
            child_prespawn = _verify_phase_receipt(
                spec,
                path=Path(
                    str(artifacts["child_prespawn_phase_receipt"])
                ),
                expected_phase="child_prespawn",
                expected_invocation_id=invocation_id,
            )
        except (OSError, TypeError, ValueError):
            pass
        else:
            child_prespawn_valid = True
            child_prespawn_fingerprint = str(
                child_prespawn["phase_receipt_fingerprint"]
            )
            child_prespawn_sha256 = str(
                child_prespawn["phase_receipt_file_sha256"]
            )
    audit_valid = bool(
        claim_matches_invocation
        and (
            not attempt_commit_required
            or (
                attempt_commit_valid
                and start_ack_valid
                and child_prespawn_valid
                and (
                    spec["execution_kind"]
                    != ACTUAL_EXECUTION_KIND
                    or current_authorization_valid is True
                )
                and current_runtime_closure_valid is True
                and authorization_matches_commit is True
            )
        )
    )
    systemd_outcome = classify_systemd_exit(effective_environment)
    finalizer_environment_audit_fingerprint: str | None = None
    finalizer_environment_inventory_fingerprint: str | None = None
    finalizer_environment_audit_valid: bool | None = None
    active_gpu_lease_fingerprint: str | None = None
    active_gpu_lease_file_sha256: str | None = None
    active_gpu_lease_device: int | None = None
    active_gpu_lease_inode: int | None = None
    active_gpu_lease_parent_device: int | None = None
    active_gpu_lease_parent_inode: int | None = None
    active_gpu_lease_valid: bool | None = None
    gpu_lease_release_authorized: bool | None = None
    gpu_lease_release_valid: bool | None = None
    gpu_lease_release_receipt_fingerprint: str | None = None
    gpu_lease_tombstone_file_sha256: str | None = None
    finalizer_lease: dict[str, object] | None = None
    if spec["execution_kind"] == ACTUAL_EXECUTION_KIND:
        finalizer_environment_audit_valid = False
        active_gpu_lease_valid = False
        gpu_lease_release_authorized = False
        gpu_lease_release_valid = False
        if attempt_commit is not None:
            try:
                finalizer_lease = _verify_active_gpu_lease(
                    spec,
                    attempt_commit,
                )
                active_gpu_lease_valid = True
                active_gpu_lease_fingerprint = str(
                    finalizer_lease["gpu_lease_fingerprint"]
                )
                active_gpu_lease_file_sha256 = str(
                    finalizer_lease["gpu_lease_file_sha256"]
                )
                active_gpu_lease_device = int(
                    finalizer_lease["gpu_lease_device"]
                )
                active_gpu_lease_inode = int(
                    finalizer_lease["gpu_lease_inode"]
                )
                active_gpu_lease_parent_device = int(
                    finalizer_lease["gpu_lease_parent_device"]
                )
                active_gpu_lease_parent_inode = int(
                    finalizer_lease["gpu_lease_parent_inode"]
                )
                finalizer_environment = _verify_live_environment(
                    spec,
                    phase="finalizer",
                )
                finalizer_environment_audit_valid = True
                finalizer_environment_audit_fingerprint = str(
                    finalizer_environment["environment_audit_fingerprint"]
                )
                finalizer_environment_inventory_fingerprint = str(
                    finalizer_environment["inventory_fingerprint"]
                )
            except Exception:
                pass
        gpu_lease_release_authorized = bool(
            audit_valid
            and active_gpu_lease_valid is True
            and finalizer_environment_audit_valid is True
        )
        if gpu_lease_release_authorized and finalizer_lease is not None:
            release_evidence = stable_fingerprint(
                {
                    "schema_version": "cure-lite-v24-terminal-release-evidence-v1",
                    "attempt_commit_fingerprint": attempt_commit_fingerprint,
                    "systemd_invocation_id": invocation_id,
                    "systemd_outcome": systemd_outcome,
                    "finalizer_environment_audit_fingerprint": (
                        finalizer_environment_audit_fingerprint
                    ),
                    "gpu_lease_fingerprint": (
                        active_gpu_lease_fingerprint
                    ),
                    "gpu_lease_file_sha256": (
                        active_gpu_lease_file_sha256
                    ),
                    "gpu_lease_device": active_gpu_lease_device,
                    "gpu_lease_inode": active_gpu_lease_inode,
                    "gpu_lease_parent_device": (
                        active_gpu_lease_parent_device
                    ),
                    "gpu_lease_parent_inode": (
                        active_gpu_lease_parent_inode
                    ),
                }
            )
            try:
                release = _release_external_gpu_lease(
                    spec,
                    finalizer_lease,
                    release_kind="committed_terminal",
                    attempt_consumed=True,
                    evidence_fingerprint=release_evidence,
                )
                release = _validate_gpu_lease_release_evidence(
                    spec,
                    finalizer_lease,
                    release,
                    release_kind="committed_terminal",
                    attempt_consumed=True,
                    evidence_fingerprint=release_evidence,
                )
            except Exception:
                if not finalizer_lease["handle"].closed:
                    finalizer_lease["handle"].close_without_release()
            else:
                gpu_lease_release_valid = True
                gpu_lease_release_receipt_fingerprint = str(
                    release.get("receipt_fingerprint")
                )
                gpu_lease_tombstone_file_sha256 = str(
                    release.get("tombstone_file_sha256")
                )
        elif finalizer_lease is not None:
            finalizer_lease["handle"].close_without_release()
        audit_valid = bool(
            audit_valid
            and finalizer_environment_audit_valid is True
            and active_gpu_lease_valid is True
            and gpu_lease_release_authorized is True
            and gpu_lease_release_valid is True
        )
    terminal = _seal(
        {
            "schema_version": SYSTEMD_TERMINAL_SCHEMA,
            "candidate": spec["candidate"],
            "stage_id": spec["stage_id"],
            "attempt_id": spec["attempt_id"],
            "runtime_spec_fingerprint": spec[
                "runtime_spec_fingerprint"
            ],
            "materialization_claim_fingerprint": fingerprint,
            "materialization_claim_file_sha256": materialization_sha256,
            "attempt_commit_fingerprint": attempt_commit_fingerprint,
            "attempt_commit_file_sha256": attempt_commit_sha256,
            "attempt_commit_required": attempt_commit_required,
            "attempt_commit_valid": attempt_commit_valid,
            "current_authorization_valid": current_authorization_valid,
            "current_runtime_closure_valid": current_runtime_closure_valid,
            "current_runtime_closure_error_type": (
                current_runtime_closure_error_type
            ),
            "authorization_matches_commit": authorization_matches_commit,
            "systemd_control_group": live_control_group,
            "claim_systemd_invocation_id": claim_invocation_id,
            "sidecar_systemd_invocation_id": invocation_id,
            "claim_valid": claim_valid,
            "claim_matches_invocation": claim_matches_invocation,
            "start_ack_receipt_fingerprint": start_ack_fingerprint,
            "start_ack_receipt_file_sha256": start_ack_sha256,
            "start_ack_valid": start_ack_valid,
            "child_prespawn_phase_receipt_fingerprint": (
                child_prespawn_fingerprint
            ),
            "child_prespawn_phase_receipt_file_sha256": (
                child_prespawn_sha256
            ),
            "child_prespawn_valid": child_prespawn_valid,
            "finalizer_environment_audit_fingerprint": (
                finalizer_environment_audit_fingerprint
            ),
            "finalizer_environment_inventory_fingerprint": (
                finalizer_environment_inventory_fingerprint
            ),
            "finalizer_environment_audit_valid": (
                finalizer_environment_audit_valid
            ),
            "active_gpu_lease_fingerprint": active_gpu_lease_fingerprint,
            "active_gpu_lease_file_sha256": active_gpu_lease_file_sha256,
            "active_gpu_lease_device": active_gpu_lease_device,
            "active_gpu_lease_inode": active_gpu_lease_inode,
            "active_gpu_lease_parent_device": (
                active_gpu_lease_parent_device
            ),
            "active_gpu_lease_parent_inode": active_gpu_lease_parent_inode,
            "active_gpu_lease_valid": active_gpu_lease_valid,
            "gpu_lease_release_authorized": gpu_lease_release_authorized,
            "gpu_lease_release_valid": gpu_lease_release_valid,
            "gpu_lease_release_receipt_fingerprint": (
                gpu_lease_release_receipt_fingerprint
            ),
            "gpu_lease_tombstone_file_sha256": (
                gpu_lease_tombstone_file_sha256
            ),
            "audit_valid": audit_valid,
            "time_utc": _utc_now(),
            "systemd_outcome": systemd_outcome,
            "scientific_decision": (
                "NOT_EVALUATED_BY_RUNTIME_SUPERVISOR"
            ),
            "scientific_gate_passed": None,
        },
        fingerprint_field="systemd_terminal_fingerprint",
    )
    _write_new_json(
        invocation_directory / f"{invocation_id}.json",
        terminal,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    commit = subparsers.add_parser("commit-and-start")
    commit.add_argument("--spec", required=True)
    claim = subparsers.add_parser("claim-materialization")
    claim.add_argument("--spec", required=True)
    verify = subparsers.add_parser("verify-runtime-spec")
    verify.add_argument("--spec", required=True)
    run = subparsers.add_parser("run-once")
    run.add_argument("--spec", required=True)
    finalize = subparsers.add_parser("systemd-finalize")
    finalize.add_argument("--spec", required=True)
    record = subparsers.add_parser("record-systemd-exit")
    record.add_argument("--spec", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.mode == "commit-and-start":
            return commit_and_start(arguments.spec)
        if arguments.mode == "claim-materialization":
            return claim_materialization(arguments.spec)
        if arguments.mode == "verify-runtime-spec":
            return verify_runtime_spec(arguments.spec)
        if arguments.mode == "run-once":
            return run_once(arguments.spec)
        return finalize_systemd(arguments.spec)
    except PermissionError as error:
        print(f"permission denied: {error}", file=sys.stderr, flush=True)
        return os.EX_NOPERM
    except FileExistsError as error:
        print(
            f"create-once identity already exists: {error}",
            file=sys.stderr,
            flush=True,
        )
        return os.EX_CANTCREAT
    except (TypeError, ValueError) as error:
        print(f"invalid runtime contract: {error}", file=sys.stderr, flush=True)
        return os.EX_CONFIG
    except OSError as error:
        print(f"runtime operating-system error: {error}", file=sys.stderr, flush=True)
        return os.EX_OSERR
    except RuntimeError as error:
        print(f"runtime evidence error: {error}", file=sys.stderr, flush=True)
        return os.EX_SOFTWARE


if __name__ == "__main__":
    raise SystemExit(main())
