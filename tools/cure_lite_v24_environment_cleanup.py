#!/usr/bin/env python3
"""Exact-authorized user-systemd cleanup for CURE-Lite v24.

This module is deliberately standard-library-only and data-blind.  Its
``build-plan`` command is read-only.  Its ``apply`` command can change only the
exact user units and actions bound by a sealed cleanup authorization.

The tool never invokes a shell, never disables a unit persistently, and never
executes ``reset-failed``.  Runtime masks are installed before any
stop operation.  If systemd reports a successful runtime-mask write while a
higher-priority persistent unit still wins, a separately authorized recovery
can issue only the previously undispatched exact stop and must retain the
runtime-mask symlink as an attested activation guard for later stability
sampling.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import grp
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
import sys
from typing import Callable, Mapping, Sequence


PLAN_SCHEMA = "cure-lite-v24-runtime-cleanup-plan-v2"
AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-runtime-cleanup-authorization-v2"
)
INTENT_SCHEMA = "cure-lite-v24-runtime-cleanup-intent-v2"
ACTION_RECEIPT_SCHEMA = (
    "cure-lite-v24-runtime-cleanup-action-receipt-v2"
)
FINAL_RECEIPT_SCHEMA = "cure-lite-v24-runtime-cleanup-receipt-v2"
TERMINAL_FAILURE_SCHEMA = (
    "cure-lite-v24-runtime-cleanup-terminal-failure-v2"
)
RECOVERY_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-runtime-cleanup-recovery-authorization-v1"
)
RECOVERY_INTENT_SCHEMA = (
    "cure-lite-v24-runtime-cleanup-recovery-intent-v1"
)
RECOVERY_ACTION_RECEIPT_SCHEMA = (
    "cure-lite-v24-runtime-cleanup-recovery-action-receipt-v1"
)
RECOVERY_TERMINAL_FAILURE_SCHEMA = (
    "cure-lite-v24-runtime-cleanup-recovery-terminal-failure-v1"
)

ENVIRONMENT_RECEIPT_SCHEMA = (
    "cure-lite-v24-runtime-environment-audit-receipt-v1"
)
ENVIRONMENT_INVENTORY_SCHEMA = (
    "cure-lite-v24-runtime-environment-inventory-v1"
)
GPU_SNAPSHOT_SCHEMA = "cure-lite-v24-runtime-gpu-double-snapshot-v1"

ACTUAL_R2_TARGET_UNIT = "cure-lite-v24-gcr-pacre-dr-r2.service"
GPU0_CONFLICT_UNIT = (
    "confa-v41-mshnet-nudt-clean-formal-20260718-v1.service"
)
GPU2_DNANET_UNIT = "confa-v41-dnanet-clean-formal-seed42-v2.service"
SELECTED_GPU_UUID = "GPU-12cdabd0-7910-8f4a-e4d7-e3c7867d1296"
EXPLICIT_USER_INSTRUCTION_ID = (
    "user-2026-07-30-clean-only-audited-gpu0-msh-conflict-v1"
)
RECOVERY_USER_INSTRUCTION_ID = (
    "user-2026-07-30-recover-partial-runtime-mask-stop-v1"
)
PRECLEANUP_BLOCKER = (
    f"scoped_blocker_unit_not_quiescent:{GPU0_CONFLICT_UNIT}"
)
PROTECTED_UNITS = frozenset(
    {
        ACTUAL_R2_TARGET_UNIT,
        GPU2_DNANET_UNIT,
        "basic.target",
        "default.target",
        "graphical-session.target",
        "paths.target",
        "sockets.target",
        "timers.target",
    }
)
ALLOWED_FAILED_UNITS = (
    "sctransnet-formal800-gpu2-recovery-postprocess-s42-v1.service",
    "sctransnet-formal800-gpu2-recovery-s42-v1.service",
    "snap.firmware-updater.firmware-notifier.service",
)

SYSTEMCTL_PATH = "/usr/bin/systemctl"
_UNIT_RE = re.compile(
    r"^[A-Za-z0-9_.@:\\-]+"
    r"\.(?:service|timer|socket|path|target|slice|scope|mount)$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
_ALLOWED_ACTIONS = frozenset({"mask-runtime", "stop"})
_ACTION_PHASE = {
    "mask-runtime": 0,
    "stop": 1,
}
_SNAPSHOT_PROPERTIES = (
    "Id",
    "Description",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "FragmentPath",
    "DropInPaths",
    "Restart",
    "RestartUSec",
    "NRestarts",
    "Result",
    "ControlGroup",
    "Environment",
    "ExecStart",
    "TriggeredBy",
    "Triggers",
    "WantedBy",
    "RequiredBy",
    "PartOf",
)
_SNAPSHOT_KEYS = set(_SNAPSHOT_PROPERTIES) | {
    "fragment_file_sha256",
    "dropin_file_sha256",
    "fragment_owner_uid",
    "fragment_group_gid",
    "fragment_group_name",
    "fragment_group_member_uids",
    "fragment_mode",
}
_PLAN_KEYS = {
    "schema_version",
    "candidate",
    "scope",
    "created_at_utc",
    "boot_id",
    "inventory_receipt_path",
    "inventory_receipt_file_sha256",
    "inventory_receipt_fingerprint",
    "manager_generation",
    "selected_gpu_uuid",
    "protected_units",
    "unit_snapshots",
    "actions",
    "restoration_actions",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "plan_fingerprint",
}
_ACTION_KEYS = {"ordinal", "unit_name", "action"}
_AUTHORIZATION_KEYS = {
    "schema_version",
    "candidate",
    "scope",
    "created_at_utc",
    "plan_path",
    "plan_file_sha256",
    "plan_fingerprint",
    "authorized_actions",
    "fresh_cleanup_authorized",
    "persistent_disable_authorized",
    "global_reset_failed_authorized",
    "authorization_basis",
    "explicit_user_instruction_id",
    "authorized_uid",
    "issued_at_utc",
    "expires_at_utc",
    "manager_generation",
    "executable_bindings",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "authorization_fingerprint",
}
_INTENT_KEYS = {
    "schema_version",
    "created_at_utc",
    "plan_file_sha256",
    "plan_fingerprint",
    "authorization_file_sha256",
    "authorization_fingerprint",
    "boot_id",
    "manager_generation",
    "before",
    "actions",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "intent_fingerprint",
}
_TERMINAL_FAILURE_KEYS = {
    "schema_version",
    "created_at_utc",
    "intent_fingerprint",
    "completed_action_receipt_fingerprints",
    "error_type",
    "error_message",
    "inflight_action",
    "automatic_rollback_performed",
    "runtime_mask_may_remain",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "terminal_failure_fingerprint",
}
_EVIDENCE_ROOT_KEYS = {
    "path",
    "file_sha256",
    "fingerprint_field",
    "fingerprint",
}
_RECOVERY_AUTHORIZATION_KEYS = {
    "schema_version",
    "candidate",
    "scope",
    "created_at_utc",
    "roots",
    "authorized_action",
    "partial_failure_condition",
    "activation_guard",
    "before",
    "authorization_basis",
    "explicit_user_instruction_id",
    "authorized_uid",
    "issued_at_utc",
    "expires_at_utc",
    "manager_generation",
    "executable_bindings",
    "persistent_disable_authorized",
    "global_reset_failed_authorized",
    "automatic_retry_authorized",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "recovery_authorization_fingerprint",
}
_RECOVERY_INTENT_KEYS = {
    "schema_version",
    "created_at_utc",
    "roots",
    "recovery_authorization_file_sha256",
    "recovery_authorization_fingerprint",
    "manager_generation",
    "before",
    "activation_guard",
    "action",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "recovery_intent_fingerprint",
}
_RECOVERY_ACTION_RECEIPT_KEYS = {
    "schema_version",
    "created_at_utc",
    "started_at_utc",
    "recovery_intent_fingerprint",
    "action",
    "argv",
    "shell",
    "returncode",
    "stdout",
    "stderr",
    "manager_generation",
    "before",
    "after",
    "protected_before",
    "protected_after",
    "activation_guard_before",
    "activation_guard_after",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "recovery_action_receipt_fingerprint",
}
_RECOVERY_TERMINAL_FAILURE_KEYS = {
    "schema_version",
    "created_at_utc",
    "recovery_intent_fingerprint",
    "error_type",
    "error_message",
    "inflight_action",
    "completed_recovery_action_receipt_fingerprints",
    "automatic_rollback_performed",
    "automatic_retry_performed",
    "persistent_disable_performed",
    "global_reset_failed_performed",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "recovery_terminal_failure_fingerprint",
}
_FINAL_RECEIPT_KEYS = {
    "schema_version",
    "created_at_utc",
    "intent_fingerprint",
    "action_receipt_fingerprints",
    "boot_id",
    "manager_generation",
    "after",
    "cleanup_mode",
    "activation_guard",
    "partial_lineage",
    "passed",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "cleanup_receipt_fingerprint",
}

NORMAL_CLEANUP_MODE = "runtime-mask-stop"
RECOVERY_CLEANUP_MODE = "partial-runtime-mask-stop-recovery"
NORMAL_GUARD_MODE = "effective-runtime-mask"
RECOVERY_GUARD_MODE = (
    "ineffective-runtime-mask-symlink-plus-explicit-stop"
)
PARTIAL_FAILURE_CONDITION = (
    "mask-runtime-returned-zero-but-persistent-unit-shadowed-runtime-mask;"
    "stop-was-not-dispatched"
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_fingerprint(value: object) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} keys are not exact")
    return value


def _require_unit_name(value: object) -> str:
    if not isinstance(value, str) or _UNIT_RE.fullmatch(value) is None:
        raise ValueError("unit name is not an exact supported systemd unit")
    if any(token in value for token in ("*", "?", "[", "]", "{", "}")):
        raise ValueError("wildcard unit names are forbidden")
    return value


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    if _BOOT_ID_RE.fullmatch(value) is None:
        raise RuntimeError("boot ID is malformed")
    return value


def fixed_user_manager_environment(
    uid: int | None = None,
) -> dict[str, str]:
    selected_uid = os.getuid() if uid is None else uid
    if (
        isinstance(selected_uid, bool)
        or not isinstance(selected_uid, int)
        or selected_uid < 0
    ):
        raise ValueError("uid must be a nonnegative integer")
    runtime = f"/run/user/{selected_uid}"
    return {
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime}/bus",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "SYSTEMD_COLORS": "0",
        "SYSTEMD_PAGER": "",
        "SYSTEMD_PAGERSECURE": "1",
        "SYSTEMD_URLIFY": "0",
        "XDG_RUNTIME_DIR": runtime,
    }


def validate_user_manager_endpoint(
    uid: int | None = None,
    *,
    run_user_root: Path = Path("/run/user"),
) -> dict[str, object]:
    selected_uid = os.getuid() if uid is None else uid
    runtime = (run_user_root / str(selected_uid)).absolute()
    runtime_stat = runtime.lstat()
    bus = runtime / "bus"
    bus_stat = bus.lstat()
    if (
        not stat.S_ISDIR(runtime_stat.st_mode)
        or runtime_stat.st_uid != selected_uid
        or stat.S_IMODE(runtime_stat.st_mode) & 0o077
        or runtime.is_symlink()
        or runtime.resolve(strict=True) != runtime
        or not stat.S_ISSOCK(bus_stat.st_mode)
        or bus_stat.st_uid != selected_uid
        or bus.is_symlink()
        or bus.resolve(strict=True) != bus
    ):
        raise PermissionError("user manager endpoint is not trusted")
    return {
        "uid": selected_uid,
        "runtime_path": str(runtime),
        "runtime_device": runtime_stat.st_dev,
        "runtime_inode": runtime_stat.st_ino,
        "bus_path": str(bus),
        "bus_device": bus_stat.st_dev,
        "bus_inode": bus_stat.st_ino,
    }


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        str(path.parent),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_private_parent(path: Path) -> int:
    parent = path.parent
    before = parent.lstat()
    if (
        not stat.S_ISDIR(before.st_mode)
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o700
    ):
        raise PermissionError("evidence parent must be canonical owner 0700")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(parent, flags)
    current = os.fstat(descriptor)
    if (
        (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        or current.st_uid != os.getuid()
        or stat.S_IMODE(current.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise PermissionError("evidence parent identity changed")
    return descriptor


def _fd_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def write_create_once_json(
    path: str | Path,
    body: Mapping[str, object],
    *,
    fingerprint_key: str,
) -> dict[str, object]:
    target = Path(path).absolute()
    parent_fd = _open_private_parent(target)
    payload = {
        **body,
        fingerprint_key: stable_fingerprint(body),
    }
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target.name, flags, 0o600, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        linked = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        raw = _fd_bytes(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o444
            or opened.st_size != len(encoded)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            or linked.st_nlink != 1
            or raw != encoded
        ):
            raise RuntimeError("create-once artifact fd identity is unsafe")
        os.fsync(parent_fd)
    finally:
        os.close(descriptor)
        os.close(parent_fd)
    metadata = target.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or target.is_symlink()
        or target.resolve(strict=True) != target
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or target.read_bytes() != encoded
    ):
        raise RuntimeError("create-once artifact failed self-verification")
    return payload


def load_sealed_json(path: str | Path) -> dict[str, object]:
    target = Path(path).absolute()
    parent_fd = _open_private_parent(target)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target.name, flags, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        metadata = os.fstat(descriptor)
        linked = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or (metadata.st_dev, metadata.st_ino)
            != (linked.st_dev, linked.st_ino)
            or linked.st_nlink != 1
        ):
            raise PermissionError("input artifact is not sealed")
        raw = _fd_bytes(descriptor)
        after = os.fstat(descriptor)
        linked_after = os.stat(
            target.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            (after.st_dev, after.st_ino, after.st_size)
            != (metadata.st_dev, metadata.st_ino, metadata.st_size)
            or (linked_after.st_dev, linked_after.st_ino)
            != (metadata.st_dev, metadata.st_ino)
        ):
            raise PermissionError("input artifact changed while reading")
    finally:
        os.close(descriptor)
        os.close(parent_fd)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("input artifact is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("input artifact must contain an object")
    if raw != (canonical_json(payload) + "\n").encode("utf-8"):
        raise ValueError("input artifact is not canonical JSON")
    return payload


def parse_systemctl_show(text: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            raise ValueError("systemctl show output is malformed")
        key, value = line.split("=", 1)
        if key in output:
            raise ValueError("systemctl show output has duplicate keys")
        output[key] = value
    if set(output) != set(_SNAPSHOT_PROPERTIES):
        raise ValueError("systemctl show property set changed")
    return output


def normalize_exec_start(raw: str) -> dict[str, str]:
    if not isinstance(raw, str) or not raw:
        raise PermissionError("ExecStart is absent")
    pairs = re.findall(
        r"(?:^|[;{])\s*(path|argv\[\]|ignore_errors)=([^;}]*)",
        raw,
    )
    normalized = {key: value.strip() for key, value in pairs}
    if (
        len(pairs) != 3
        or set(normalized) != {"path", "argv[]", "ignore_errors"}
        or any(not value for value in normalized.values())
    ):
        raise PermissionError("ExecStart identity is ambiguous")
    return {
        "path": normalized["path"],
        "argv": normalized["argv[]"],
        "ignore_errors": normalized["ignore_errors"],
    }


def trusted_fragment_identity(
    path: Path,
    metadata: os.stat_result,
) -> dict[str, object]:
    mode = stat.S_IMODE(metadata.st_mode)
    if mode not in {0o644, 0o664}:
        raise PermissionError("unit fragment mode is outside 0644/0664 policy")
    try:
        group = grp.getgrgid(metadata.st_gid)
        primary_names = {
            row.pw_name for row in pwd.getpwall() if row.pw_gid == metadata.st_gid
        }
        member_names = primary_names | set(group.gr_mem)
        member_uids = sorted({pwd.getpwnam(name).pw_uid for name in member_names})
    except (KeyError, OSError) as error:
        raise PermissionError("fragment group membership is not enumerable") from error
    if mode == 0o664 and (
        metadata.st_gid != os.getgid()
        or group.gr_gid != os.getgid()
        or member_uids != [os.getuid()]
    ):
        raise PermissionError("0664 fragment group is not exclusive to current UID")
    return {
        "fragment_owner_uid": metadata.st_uid,
        "fragment_group_gid": metadata.st_gid,
        "fragment_group_name": group.gr_name,
        "fragment_group_member_uids": member_uids,
        "fragment_mode": mode,
    }


def run_systemctl(
    argv: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    exact = tuple(argv)
    _validate_systemctl_argv(exact)
    return runner(
        list(exact),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
        env=fixed_user_manager_environment(),
    )


def _snapshot_argv(unit_name: str) -> tuple[str, ...]:
    unit = _require_unit_name(unit_name)
    argv: list[str] = [SYSTEMCTL_PATH, "--user", "show", unit, "--no-pager"]
    for name in _SNAPSHOT_PROPERTIES:
        argv.extend(("-p", name))
    return tuple(argv)


def _validate_systemctl_argv(argv: Sequence[str]) -> tuple[str, ...]:
    exact = tuple(argv)
    allowed_action = False
    if len(exact) == 5 and exact[:4] == (
        SYSTEMCTL_PATH,
        "--user",
        "mask",
        "--runtime",
    ):
        _require_unit_name(exact[4])
        allowed_action = exact[4] == GPU0_CONFLICT_UNIT
    elif len(exact) == 4 and exact[:3] == (
        SYSTEMCTL_PATH,
        "--user",
        "stop",
    ):
        _require_unit_name(exact[3])
        allowed_action = exact[3] == GPU0_CONFLICT_UNIT
    elif len(exact) >= 7 and exact[:3] == (
        SYSTEMCTL_PATH,
        "--user",
        "show",
    ):
        unit = _require_unit_name(exact[3])
        allowed_action = (
            unit in {GPU0_CONFLICT_UNIT, GPU2_DNANET_UNIT}
            and exact == _snapshot_argv(unit)
        )
    if not allowed_action:
        raise ValueError("systemctl argv is outside the exact cleanup allowlist")
    return exact


def query_unit_snapshot(
    unit_name: str,
    *,
    command_runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] = run_systemctl,
) -> dict[str, object]:
    unit = _require_unit_name(unit_name)
    argv = _snapshot_argv(unit)
    _validate_systemctl_argv(argv)
    completed = command_runner(argv)
    if completed.returncode != 0:
        raise RuntimeError(f"systemctl show failed for {unit}")
    values = parse_systemctl_show(completed.stdout)
    if values["Id"] != unit:
        raise RuntimeError("systemctl unit identity mismatch")
    fragment = values["FragmentPath"]
    fragment_sha: str | None = None
    if fragment:
        path = Path(fragment)
        metadata = path.lstat()
        if (
            not path.is_absolute()
            or not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or path.resolve(strict=True) != path
            or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        ):
            raise PermissionError("unit fragment is not trusted")
        fragment_sha = file_sha256(path)
    if values["DropInPaths"]:
        raise PermissionError("unknown drop-in closure is forbidden")
    fragment_identity = (
        trusted_fragment_identity(path, metadata) if fragment else {
            "fragment_owner_uid": None,
            "fragment_group_gid": None,
            "fragment_group_name": None,
            "fragment_group_member_uids": [],
            "fragment_mode": None,
        }
    )
    values["ExecStart"] = normalize_exec_start(values["ExecStart"])
    return {
        **values,
        "fragment_file_sha256": fragment_sha,
        "dropin_file_sha256": {},
        **fragment_identity,
    }


def _action(
    ordinal: int,
    unit_name: str,
    action: str,
) -> dict[str, object]:
    if action not in _ALLOWED_ACTIONS:
        raise ValueError("cleanup action is not allowed")
    return {
        "ordinal": ordinal,
        "unit_name": _require_unit_name(unit_name),
        "action": action,
    }


def _without_fingerprint(
    payload: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    body = dict(payload)
    fingerprint = body.pop(key, None)
    if (
        not isinstance(fingerprint, str)
        or _SHA256_RE.fullmatch(fingerprint) is None
        or stable_fingerprint(body) != fingerprint
    ):
        raise PermissionError(f"{key} is invalid")
    return body


def _parse_utc_timestamp(value: object, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise PermissionError(f"{name} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PermissionError(f"{name} is not a UTC timestamp") from error
    if parsed.tzinfo is None:
        raise PermissionError(f"{name} is not timezone-aware")
    return parsed


def _evidence_root(
    path: str | Path,
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    target = Path(path).absolute()
    payload = load_sealed_json(target)
    _without_fingerprint(payload, fingerprint_field)
    return {
        "path": str(target),
        "file_sha256": file_sha256(target),
        "fingerprint_field": fingerprint_field,
        "fingerprint": payload[fingerprint_field],
    }


def _load_bound_evidence_root(
    value: object,
    *,
    expected_path: str | Path,
    fingerprint_field: str,
) -> dict[str, object]:
    root = dict(
        _require_exact_keys(
            value,
            _EVIDENCE_ROOT_KEYS,
            name="sealed evidence root",
        )
    )
    target = Path(expected_path).absolute()
    if (
        root.get("path") != str(target)
        or root.get("file_sha256") != file_sha256(target)
        or root.get("fingerprint_field") != fingerprint_field
    ):
        raise PermissionError("sealed evidence root binding changed")
    payload = load_sealed_json(target)
    _without_fingerprint(payload, fingerprint_field)
    if root.get("fingerprint") != payload.get(fingerprint_field):
        raise PermissionError("sealed evidence root fingerprint changed")
    return payload


def _validate_no_payload(value: Mapping[str, object]) -> None:
    if (
        value.get("payload_authority") != "none"
        or value.get("D_R_payload_accessed") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("cleanup recovery evidence accessed payload")


def _validate_archived_authorization(
    value: object,
    *,
    plan_path: Path,
    plan: Mapping[str, object],
) -> dict[str, object]:
    authorization = dict(
        _require_exact_keys(
            value,
            _AUTHORIZATION_KEYS,
            name="archived cleanup authorization",
        )
    )
    _validate_no_payload(authorization)
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or authorization.get("candidate") != "GCR-PACRE-v24"
        or authorization.get("scope")
        != "user-systemd-environment-cleanup"
        or authorization.get("plan_path") != str(plan_path.absolute())
        or authorization.get("plan_file_sha256") != file_sha256(plan_path)
        or authorization.get("plan_fingerprint")
        != plan.get("plan_fingerprint")
        or authorization.get("authorized_actions") != plan.get("actions")
        or authorization.get("fresh_cleanup_authorized") is not True
        or authorization.get("persistent_disable_authorized") is not False
        or authorization.get("global_reset_failed_authorized") is not False
        or authorization.get("explicit_user_instruction_id")
        != EXPLICIT_USER_INSTRUCTION_ID
        or authorization.get("authorized_uid") != os.getuid()
        or authorization.get("manager_generation")
        != plan.get("manager_generation")
        or not isinstance(authorization.get("authorization_basis"), str)
        or not str(authorization["authorization_basis"]).strip()
    ):
        raise PermissionError("archived cleanup authorization lineage changed")
    issued = _parse_utc_timestamp(
        authorization.get("issued_at_utc"),
        name="archived cleanup authorization issuance",
    )
    expires = _parse_utc_timestamp(
        authorization.get("expires_at_utc"),
        name="archived cleanup authorization expiry",
    )
    created = _parse_utc_timestamp(
        authorization.get("created_at_utc"),
        name="archived cleanup authorization creation",
    )
    bindings = authorization.get("executable_bindings")
    if (
        expires <= issued
        or expires - issued > timedelta(minutes=5)
        or not issued <= created <= expires
        or not isinstance(bindings, dict)
        or set(bindings)
        != {"cleanup_tool", "environment_auditor", "python", "systemctl"}
    ):
        raise PermissionError(
            "archived cleanup authorization lifetime or tools are invalid"
        )
    for binding in bindings.values():
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "file_sha256"}
            or not isinstance(binding.get("path"), str)
            or not Path(str(binding["path"])).is_absolute()
            or not isinstance(binding.get("file_sha256"), str)
            or _SHA256_RE.fullmatch(str(binding["file_sha256"])) is None
        ):
            raise PermissionError(
                "archived cleanup executable binding is malformed"
            )
    _without_fingerprint(
        authorization,
        "authorization_fingerprint",
    )
    return authorization


def read_runtime_mask_activation_guard(
    manager_generation: Mapping[str, object],
) -> dict[str, object]:
    endpoint = manager_generation.get("endpoint")
    if not isinstance(endpoint, Mapping):
        raise PermissionError("manager endpoint is absent from activation guard")
    runtime_directory = endpoint.get("runtime_directory")
    uid = endpoint.get("uid")
    if (
        not isinstance(runtime_directory, str)
        or not Path(runtime_directory).is_absolute()
        or uid != os.getuid()
    ):
        raise PermissionError("activation guard runtime identity changed")
    path = (
        Path(runtime_directory)
        / "systemd"
        / "user"
        / GPU0_CONFLICT_UNIT
    ).absolute()
    expected_path = (
        Path(f"/run/user/{os.getuid()}")
        / "systemd"
        / "user"
        / GPU0_CONFLICT_UNIT
    )
    if path != expected_path:
        raise PermissionError("activation guard path changed")
    metadata = path.lstat()
    target = os.readlink(path)
    if (
        not stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or target != "/dev/null"
    ):
        raise PermissionError("runtime mask activation guard is not exact")
    return {
        "mode": RECOVERY_GUARD_MODE,
        "unit_name": GPU0_CONFLICT_UNIT,
        "path": str(path),
        "target": target,
        "owner_uid": metadata.st_uid,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "observed_unit_file_state": "enabled",
    }


def _validate_recovery_guard(value: object) -> dict[str, object]:
    guard = dict(
        _require_exact_keys(
            value,
            {
                "mode",
                "unit_name",
                "path",
                "target",
                "owner_uid",
                "device",
                "inode",
                "observed_unit_file_state",
            },
            name="cleanup recovery activation guard",
        )
    )
    expected_path = (
        Path(f"/run/user/{os.getuid()}")
        / "systemd"
        / "user"
        / GPU0_CONFLICT_UNIT
    )
    if (
        guard.get("mode") != RECOVERY_GUARD_MODE
        or guard.get("unit_name") != GPU0_CONFLICT_UNIT
        or guard.get("target") != "/dev/null"
        or guard.get("owner_uid") != os.getuid()
        or guard.get("observed_unit_file_state") != "enabled"
        or not isinstance(guard.get("path"), str)
        or not Path(str(guard["path"])).is_absolute()
        or Path(str(guard["path"])) != expected_path
        or isinstance(guard.get("device"), bool)
        or not isinstance(guard.get("device"), int)
        or guard["device"] < 0
        or isinstance(guard.get("inode"), bool)
        or not isinstance(guard.get("inode"), int)
        or guard["inode"] <= 0
    ):
        raise PermissionError("cleanup recovery activation guard changed")
    return guard


def validate_environment_receipt(value: object) -> dict[str, object]:
    receipt = dict(value) if isinstance(value, dict) else {}
    body = _without_fingerprint(receipt, "receipt_fingerprint")
    expected_receipt_keys = {
        "schema_version", "created_at_utc", "command",
        "environment_binding", "inventory", "passed",
        "error_type", "error_message", "D_R_payload_accessed",
        "D_V_payload_accessed", "D_T_payload_accessed",
    }
    if set(body) != expected_receipt_keys:
        raise PermissionError("environment receipt keys are not exact")
    inventory = body.get("inventory")
    if (
        body.get("schema_version") != ENVIRONMENT_RECEIPT_SCHEMA
        or body.get("command") != "audit-only"
        or body.get("passed") is not False
        or body.get("error_type") is not None
        or body.get("error_message") is not None
        or body.get("D_R_payload_accessed") is not False
        or body.get("D_V_payload_accessed") is not False
        or body.get("D_T_payload_accessed") is not False
        or not isinstance(inventory, dict)
    ):
        raise PermissionError("environment receipt is not exact precleanup evidence")
    inventory_body = _without_fingerprint(inventory, "inventory_fingerprint")
    expected_inventory_keys = {
        "schema_version", "created_at_utc", "uid", "boot_id",
        "manager", "unit_scope", "gpu_snapshot", "blockers", "passed",
        "D_R_payload_accessed", "D_V_payload_accessed",
        "D_T_payload_accessed",
    }
    if set(inventory_body) != expected_inventory_keys:
        raise PermissionError("environment inventory keys are not exact")
    uid = os.getuid()
    manager = inventory.get("manager")
    scope = inventory.get("unit_scope")
    gpu = inventory.get("gpu_snapshot")
    if not all(isinstance(item, dict) for item in (manager, scope, gpu)):
        raise PermissionError("environment inventory nested objects are invalid")
    assert isinstance(manager, dict) and isinstance(scope, dict)
    assert isinstance(gpu, dict)
    _without_fingerprint(gpu, "snapshot_fingerprint")
    identity = manager.get("identity")
    endpoint = manager.get("endpoint")
    shadows = scope.get("shadows")
    environment_binding = body.get("environment_binding")
    manager_keys = {
        "state", "allowed_states", "returncode", "failed_units",
        "allowed_failed_unit_ids", "unexpected_failed_unit_ids",
        "scoped_failed_unit_ids", "identity", "endpoint",
    }
    scope_keys = {
        "target_unit_id", "conflict_unit_ids", "dependency_unit_ids",
        "require_target_ready", "shadows",
    }
    endpoint_keys = {
        "uid", "runtime_directory", "runtime_directory_device",
        "runtime_directory_inode", "bus_path", "bus_device", "bus_inode",
    }
    binding_keys = {
        "inventory_fingerprint", "boot_id", "runtime_directory",
        "runtime_directory_device", "runtime_directory_inode",
        "manager_identity",
    }
    gpu_keys = {
        "schema_version", "selected_gpu_uuid", "expected_uid",
        "allowed_unit_ids", "strict_all_gpu_consumers", "devices",
        "first_apps", "second_apps", "process_unit_mapping",
        "observations", "blockers", "passed", "snapshot_fingerprint",
    }
    devices = gpu.get("devices")
    selected_devices = (
        [row for row in devices if isinstance(row, dict)
         and row.get("uuid") == SELECTED_GPU_UUID]
        if isinstance(devices, list) else []
    )
    mappings = gpu.get("process_unit_mapping")
    unsafe_selected_mapping = (
        not isinstance(mappings, list)
        or any(
            not isinstance(row, dict)
            or (
                row.get("gpu_uuid") == SELECTED_GPU_UUID
                and row.get("unit_id") != GPU0_CONFLICT_UNIT
            )
            or (
                row.get("unit_id") == GPU2_DNANET_UNIT
                and row.get("gpu_uuid") == SELECTED_GPU_UUID
            )
            for row in (mappings if isinstance(mappings, list) else [])
        )
    )
    try:
        receipt_time = datetime.fromisoformat(
            str(body.get("created_at_utc")).replace("Z", "+00:00")
        )
        inventory_time = datetime.fromisoformat(
            str(inventory.get("created_at_utc")).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise PermissionError("environment receipt time is invalid") from error
    now = datetime.now(timezone.utc)
    shadow_keys = {
        "Id", "LoadState", "ActiveState", "SubState", "UnitFileState",
        "Restart", "RestartUSec", "NRestarts", "ControlGroup", "FragmentPath",
        "DropInPaths", "TriggeredBy", "Triggers", "WantedBy",
        "RequiredBy", "PartOf",
    }
    target_shadow = shadows.get(ACTUAL_R2_TARGET_UNIT) if isinstance(shadows, dict) else None
    conflict_shadow = shadows.get(GPU0_CONFLICT_UNIT) if isinstance(shadows, dict) else None
    target_activation_closure = (
        {
            field: target_shadow.get(field)
            for field in (
                "TriggeredBy", "Triggers", "WantedBy", "RequiredBy", "PartOf"
            )
        }
        if isinstance(target_shadow, dict)
        else None
    )
    conflict_activation_closure = (
        {
            field: conflict_shadow.get(field)
            for field in (
                "TriggeredBy", "Triggers", "RequiredBy", "PartOf"
            )
        }
        if isinstance(conflict_shadow, dict)
        else None
    )
    conflict_wanted_by = (
        str(conflict_shadow.get("WantedBy", "")).split()
        if isinstance(conflict_shadow, dict)
        else []
    )
    endpoint_integers = (
        endpoint.get("runtime_directory_device"),
        endpoint.get("runtime_directory_inode"),
        endpoint.get("bus_device"),
        endpoint.get("bus_inode"),
    ) if isinstance(endpoint, dict) else ()
    expected_cgroup = (
        f"/user.slice/user-{uid}.slice/user@{uid}.service/init.scope"
    )
    if (
        inventory.get("schema_version") != ENVIRONMENT_INVENTORY_SCHEMA
        or inventory.get("uid") != uid
        or receipt_time.tzinfo is None
        or inventory_time.tzinfo is None
        or not inventory_time <= receipt_time <= now
        or now - inventory_time > timedelta(minutes=5)
        or _BOOT_ID_RE.fullmatch(str(inventory.get("boot_id"))) is None
        or inventory.get("passed") is not False
        or inventory.get("blockers") != [PRECLEANUP_BLOCKER]
        or inventory.get("D_R_payload_accessed") is not False
        or inventory.get("D_V_payload_accessed") is not False
        or inventory.get("D_T_payload_accessed") is not False
        or scope.get("target_unit_id") != ACTUAL_R2_TARGET_UNIT
        or scope.get("conflict_unit_ids") != [GPU0_CONFLICT_UNIT]
        or scope.get("dependency_unit_ids") != []
        or not isinstance(shadows, dict)
        or set(shadows) != {ACTUAL_R2_TARGET_UNIT, GPU0_CONFLICT_UNIT}
        or not isinstance(target_shadow, dict)
        or not isinstance(conflict_shadow, dict)
        or set(target_shadow) != shadow_keys
        or set(conflict_shadow) != shadow_keys
        or target_shadow.get("Id") != ACTUAL_R2_TARGET_UNIT
        or conflict_shadow.get("Id") != GPU0_CONFLICT_UNIT
        or target_activation_closure != {
            "TriggeredBy": "", "Triggers": "", "WantedBy": "",
            "RequiredBy": "", "PartOf": "",
        }
        or conflict_activation_closure != {
            "TriggeredBy": "", "Triggers": "", "RequiredBy": "",
            "PartOf": "",
        }
        or len(conflict_wanted_by) != len(set(conflict_wanted_by))
        or set(conflict_wanted_by) - {
            "default.target", "graphical-session.target"
        }
        or conflict_shadow.get("LoadState") != "loaded"
        or conflict_shadow.get("Restart") != "on-failure"
        or conflict_shadow.get("RestartUSec") != "30s"
        or conflict_shadow.get("DropInPaths") != ""
        or (
            conflict_shadow.get("ActiveState") == "inactive"
            and conflict_shadow.get("SubState") == "dead"
            and conflict_shadow.get("UnitFileState") == "masked-runtime"
        )
        or gpu.get("schema_version") != GPU_SNAPSHOT_SCHEMA
        or gpu.get("selected_gpu_uuid") != SELECTED_GPU_UUID
        or gpu.get("expected_uid") != uid
        or set(gpu) != gpu_keys
        or gpu.get("allowed_unit_ids") != [GPU0_CONFLICT_UNIT]
        or gpu.get("strict_all_gpu_consumers") is not False
        or unsafe_selected_mapping
        or len(selected_devices) != 1
        or selected_devices[0].get("index") != 0
        or selected_devices[0].get("minor_number") != 0
        or selected_devices[0].get("mps_state") != "not_observed"
        or gpu.get("blockers") != []
        or gpu.get("passed") is not True
        or not isinstance(identity, dict)
        or set(manager) != manager_keys
        or manager.get("state") not in {"running", "degraded"}
        or manager.get("allowed_failed_unit_ids") != list(ALLOWED_FAILED_UNITS)
        or manager.get("unexpected_failed_unit_ids") != []
        or manager.get("scoped_failed_unit_ids") != []
        or set(scope) != scope_keys
        or set(identity) != {"pid", "starttime_ticks", "uid", "control_group"}
        or identity.get("uid") != uid
        or isinstance(identity.get("pid"), bool)
        or not isinstance(identity.get("pid"), int)
        or identity["pid"] <= 0
        or isinstance(identity.get("starttime_ticks"), bool)
        or not isinstance(identity.get("starttime_ticks"), int)
        or identity["starttime_ticks"] <= 0
        or identity.get("control_group") != expected_cgroup
        or not isinstance(endpoint, dict)
        or set(endpoint) != endpoint_keys
        or endpoint.get("uid") != uid
        or endpoint.get("runtime_directory") != f"/run/user/{uid}"
        or endpoint.get("bus_path") != f"/run/user/{uid}/bus"
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in endpoint_integers
        )
        or not isinstance(environment_binding, dict)
        or set(environment_binding) != binding_keys
        or environment_binding.get("inventory_fingerprint")
        != inventory.get("inventory_fingerprint")
        or environment_binding.get("boot_id") != inventory.get("boot_id")
        or environment_binding.get("manager_identity") != identity
        or environment_binding.get("runtime_directory")
        != endpoint.get("runtime_directory")
        or environment_binding.get("runtime_directory_device")
        != endpoint.get("runtime_directory_device")
        or environment_binding.get("runtime_directory_inode")
        != endpoint.get("runtime_directory_inode")
    ):
        raise PermissionError("environment inventory scope or identity is invalid")
    return receipt


def _manager_generation_from_inventory(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    inventory = receipt["inventory"]
    assert isinstance(inventory, dict)
    manager = inventory["manager"]
    assert isinstance(manager, dict)
    return {
        "boot_id": inventory["boot_id"],
        "identity": manager["identity"],
        "endpoint": manager["endpoint"],
    }


def validate_live_manager_generation(
    expected: Mapping[str, object],
) -> dict[str, object]:
    if _boot_id() != expected.get("boot_id"):
        raise RuntimeError("manager boot generation changed")
    identity = expected.get("identity")
    endpoint_expected = expected.get("endpoint")
    if not isinstance(identity, dict) or not isinstance(endpoint_expected, dict):
        raise RuntimeError("manager generation is malformed")
    endpoint = validate_user_manager_endpoint()
    normalized = {
        "uid": endpoint["uid"],
        "runtime_directory": endpoint["runtime_path"],
        "runtime_directory_device": endpoint["runtime_device"],
        "runtime_directory_inode": endpoint["runtime_inode"],
        "bus_path": endpoint["bus_path"],
        "bus_device": endpoint["bus_device"],
        "bus_inode": endpoint["bus_inode"],
    }
    if normalized != endpoint_expected:
        raise RuntimeError("manager endpoint generation changed")
    pid = identity.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RuntimeError("manager PID is invalid")
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = stat_text.rfind(")")
    fields = stat_text[close + 2:].split()
    if close < 0 or len(fields) <= 19:
        raise RuntimeError("manager stat identity is malformed")
    starttime = int(fields[19])
    cgroup_rows = Path(f"/proc/{pid}/cgroup").read_text(
        encoding="ascii"
    ).splitlines()
    unified = [row.split("::", 1)[1] for row in cgroup_rows if "::" in row]
    if (
        starttime != identity.get("starttime_ticks")
        or unified != [identity.get("control_group")]
        or Path(f"/proc/{pid}").stat().st_uid != os.getuid()
    ):
        raise RuntimeError("manager PID/starttime/cgroup generation changed")
    return dict(expected)


def validate_plan(value: object) -> dict[str, object]:
    plan = dict(_require_exact_keys(value, _PLAN_KEYS, name="plan"))
    if (
        plan["schema_version"] != PLAN_SCHEMA
        or plan["candidate"] != "GCR-PACRE-v24"
        or plan["scope"] != "user-systemd-environment-cleanup"
        or plan["payload_authority"] != "none"
        or plan["D_R_payload_accessed"] is not False
        or plan["D_V_payload_accessed"] is not False
        or plan["D_T_payload_accessed"] is not False
        or not isinstance(plan["created_at_utc"], str)
        or not isinstance(plan["inventory_receipt_path"], str)
        or not Path(plan["inventory_receipt_path"]).is_absolute()
        or not isinstance(plan["inventory_receipt_file_sha256"], str)
        or _SHA256_RE.fullmatch(
            plan["inventory_receipt_file_sha256"]
        )
        is None
        or not isinstance(plan["boot_id"], str)
        or _BOOT_ID_RE.fullmatch(plan["boot_id"]) is None
        or plan["selected_gpu_uuid"] != SELECTED_GPU_UUID
        or not isinstance(plan["manager_generation"], dict)
    ):
        raise ValueError("cleanup plan identity is invalid")
    protected_raw = plan["protected_units"]
    if not isinstance(protected_raw, list):
        raise ValueError("protected_units must be a list")
    protected = [_require_unit_name(item) for item in protected_raw]
    if protected != sorted(set(protected)):
        raise ValueError("protected_units must be sorted and unique")
    if protected != sorted(PROTECTED_UNITS):
        raise ValueError("protected_units are not the hard-coded protection set")
    snapshots_raw = plan["unit_snapshots"]
    if not isinstance(snapshots_raw, dict):
        raise ValueError("unit_snapshots must be an object")
    snapshots: dict[str, object] = {}
    for unit, raw in snapshots_raw.items():
        exact_unit = _require_unit_name(unit)
        snapshot = dict(
            _require_exact_keys(raw, _SNAPSHOT_KEYS, name="unit snapshot")
        )
        if snapshot["Id"] != exact_unit:
            raise ValueError("snapshot unit identity mismatch")
        if exact_unit not in {GPU0_CONFLICT_UNIT, GPU2_DNANET_UNIT}:
            raise ValueError("cleanup snapshot is outside the exact observed scope")
        fingerprint = snapshot["fragment_file_sha256"]
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or _SHA256_RE.fullmatch(fingerprint) is None
        ):
            raise ValueError("fragment SHA is invalid")
        snapshots[exact_unit] = snapshot
        if (
            snapshot["LoadState"] != "loaded"
            or snapshot["Restart"] != "on-failure"
            or snapshot["FragmentPath"] == ""
            or snapshot["fragment_owner_uid"] != os.getuid()
            or snapshot["fragment_mode"] not in {0o644, 0o664}
            or not isinstance(snapshot["fragment_group_gid"], int)
            or not isinstance(snapshot["fragment_group_name"], str)
            or not snapshot["fragment_group_name"]
            or not isinstance(snapshot["fragment_group_member_uids"], list)
            or (
                snapshot["fragment_mode"] == 0o664
                and (
                    snapshot["fragment_group_gid"] != os.getgid()
                    or snapshot["fragment_group_member_uids"] != [os.getuid()]
                )
            )
            or snapshot["DropInPaths"] != ""
            or snapshot["dropin_file_sha256"] != {}
            or (
                exact_unit == GPU0_CONFLICT_UNIT
                and (
                    snapshot["TriggeredBy"] != ""
                    or snapshot["Triggers"] != ""
                    or snapshot["RequiredBy"] != ""
                    or snapshot["PartOf"] != ""
                    or set(str(snapshot["WantedBy"]).split())
                    - {"default.target", "graphical-session.target"}
                )
            )
            or (
                exact_unit == GPU2_DNANET_UNIT
                and (
                    snapshot["ActiveState"] != "activating"
                    or snapshot["SubState"] != "auto-restart"
                )
            )
        ):
            raise ValueError("GPU0 unit activation/drop-in closure is not exact")
    actions_raw = plan["actions"]
    if not isinstance(actions_raw, list) or not actions_raw:
        raise ValueError("cleanup actions must be a nonempty list")
    actions: list[dict[str, object]] = []
    previous_phase = -1
    for ordinal, raw in enumerate(actions_raw):
        action = dict(
            _require_exact_keys(raw, _ACTION_KEYS, name="cleanup action")
        )
        if action["ordinal"] != ordinal:
            raise ValueError("cleanup action ordinals are not contiguous")
        unit = _require_unit_name(action["unit_name"])
        name = action["action"]
        if not isinstance(name, str) or name not in _ALLOWED_ACTIONS:
            raise ValueError("cleanup action is unsupported")
        if unit in protected:
            raise ValueError("cleanup action targets a protected unit")
        if unit not in snapshots:
            raise ValueError("cleanup action lacks a unit snapshot")
        phase = _ACTION_PHASE[name]
        if phase < previous_phase:
            raise ValueError("cleanup action phase order is unsafe")
        previous_phase = phase
        actions.append(action)
    masked = {
        item["unit_name"]
        for item in actions
        if item["action"] == "mask-runtime"
    }
    stopped = {
        item["unit_name"]
        for item in actions
        if item["action"] == "stop"
    }
    if not stopped.issubset(masked):
        raise ValueError("every stopped unit must first be runtime-masked")
    expected_actions = [
        _action(0, GPU0_CONFLICT_UNIT, "mask-runtime"),
        _action(1, GPU0_CONFLICT_UNIT, "stop"),
    ]
    if actions != expected_actions or set(snapshots) != {
        GPU0_CONFLICT_UNIT,
        GPU2_DNANET_UNIT,
    }:
        raise ValueError("cleanup actions are not the exact GPU0 sequence")
    restoration = plan["restoration_actions"]
    if not isinstance(restoration, list):
        raise ValueError("restoration_actions must be a list")
    if restoration != [{
        "unit_name": GPU0_CONFLICT_UNIT,
        "suggested_action": "unmask-runtime",
        "requires_separate_authorization": True,
    }]:
        raise ValueError("restoration actions are not exact")
    expected = plan.pop("plan_fingerprint")
    if (
        not isinstance(expected, str)
        or stable_fingerprint(plan) != expected
    ):
        raise ValueError("cleanup plan fingerprint mismatch")
    return {**plan, "plan_fingerprint": expected}


def build_plan(
    *,
    inventory_receipt_path: Path,
    mask_units: Sequence[str],
    stop_units: Sequence[str],
    protected_units: Sequence[str] = (),
    snapshot_reader: Callable[[str], dict[str, object]] = (
        query_unit_snapshot
    ),
) -> dict[str, object]:
    inventory = inventory_receipt_path.absolute()
    environment_receipt = validate_environment_receipt(
        load_sealed_json(inventory)
    )
    masks = sorted({_require_unit_name(item) for item in mask_units})
    stops = sorted({_require_unit_name(item) for item in stop_units})
    protected = sorted(
        {_require_unit_name(item) for item in protected_units}
    )
    if (
        masks != [GPU0_CONFLICT_UNIT]
        or stops != [GPU0_CONFLICT_UNIT]
    ):
        raise ValueError("only exact runtime-mask then stop of GPU0 conflict is allowed")
    if protected and protected != sorted(PROTECTED_UNITS):
        raise ValueError("caller protection set differs from hard-coded policy")
    protected = sorted(PROTECTED_UNITS)
    units = sorted(set(masks) | set(stops) | {GPU2_DNANET_UNIT})
    if (set(masks) | set(stops)) & set(protected):
        raise ValueError("cleanup plan targets a protected unit")
    snapshots = {unit: snapshot_reader(unit) for unit in units}
    actions: list[dict[str, object]] = []
    for unit in masks:
        actions.append(_action(len(actions), unit, "mask-runtime"))
    for unit in stops:
        actions.append(_action(len(actions), unit, "stop"))
    restoration = [
        {
            "unit_name": unit,
            "suggested_action": "unmask-runtime",
            "requires_separate_authorization": True,
        }
        for unit in masks
    ]
    body: dict[str, object] = {
        "schema_version": PLAN_SCHEMA,
        "candidate": "GCR-PACRE-v24",
        "scope": "user-systemd-environment-cleanup",
        "created_at_utc": utc_now(),
        "boot_id": _boot_id(),
        "inventory_receipt_path": str(inventory),
        "inventory_receipt_file_sha256": file_sha256(inventory),
        "inventory_receipt_fingerprint": environment_receipt[
            "receipt_fingerprint"
        ],
        "manager_generation": _manager_generation_from_inventory(
            environment_receipt
        ),
        "selected_gpu_uuid": SELECTED_GPU_UUID,
        "protected_units": protected,
        "unit_snapshots": snapshots,
        "actions": actions,
        "restoration_actions": restoration,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    materialized = {
        **body,
        "plan_fingerprint": stable_fingerprint(body),
    }
    return validate_plan(materialized)


def validate_authorization(
    value: object,
    *,
    plan_path: Path,
    plan: Mapping[str, object],
) -> dict[str, object]:
    authorization = dict(
        _require_exact_keys(
            value,
            _AUTHORIZATION_KEYS,
            name="cleanup authorization",
        )
    )
    expected_actions = plan["actions"]
    if (
        authorization["schema_version"] != AUTHORIZATION_SCHEMA
        or authorization["candidate"] != "GCR-PACRE-v24"
        or authorization["scope"] != "user-systemd-environment-cleanup"
        or authorization["plan_path"] != str(plan_path.absolute())
        or authorization["plan_file_sha256"] != file_sha256(plan_path)
        or authorization["plan_fingerprint"]
        != plan["plan_fingerprint"]
        or authorization["authorized_actions"] != expected_actions
        or authorization["fresh_cleanup_authorized"] is not True
        or authorization["persistent_disable_authorized"] is not False
        or authorization["global_reset_failed_authorized"] is not False
        or authorization["payload_authority"] != "none"
        or authorization["D_R_payload_accessed"] is not False
        or authorization["D_V_payload_accessed"] is not False
        or authorization["D_T_payload_accessed"] is not False
        or authorization["explicit_user_instruction_id"]
        != EXPLICIT_USER_INSTRUCTION_ID
        or authorization["authorized_uid"] != os.getuid()
        or authorization["manager_generation"] != plan["manager_generation"]
        or authorization["executable_bindings"] != _executable_bindings()
        or not isinstance(authorization["authorization_basis"], str)
        or not authorization["authorization_basis"].strip()
    ):
        raise PermissionError("cleanup authorization does not bind the plan")
    try:
        issued = datetime.fromisoformat(
            str(authorization["issued_at_utc"]).replace("Z", "+00:00")
        )
        expires = datetime.fromisoformat(
            str(authorization["expires_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise PermissionError("cleanup authorization time is invalid") from error
    now = datetime.now(timezone.utc)
    if (
        issued.tzinfo is None
        or expires.tzinfo is None
        or not issued <= now <= expires
        or expires - issued > timedelta(minutes=5)
    ):
        raise PermissionError("cleanup authorization is stale or not yet valid")
    expected = authorization.pop("authorization_fingerprint")
    if (
        not isinstance(expected, str)
        or stable_fingerprint(authorization) != expected
    ):
        raise PermissionError("cleanup authorization fingerprint mismatch")
    return {
        **authorization,
        "authorization_fingerprint": expected,
    }


def build_authorization(
    *,
    plan_path: Path,
    plan: Mapping[str, object],
    authorization_basis: str,
    explicit_user_instruction_id: str,
    validity_seconds: int = 300,
) -> dict[str, object]:
    if not authorization_basis.strip():
        raise ValueError("authorization basis must be nonempty")
    if explicit_user_instruction_id != EXPLICIT_USER_INSTRUCTION_ID:
        raise PermissionError("explicit user cleanup instruction is absent")
    if (
        isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("authorization validity must be 1..300 seconds")
    issued = datetime.now(timezone.utc)
    body: dict[str, object] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "candidate": "GCR-PACRE-v24",
        "scope": "user-systemd-environment-cleanup",
        "created_at_utc": utc_now(),
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (
            issued + timedelta(seconds=validity_seconds)
        ).isoformat().replace("+00:00", "Z"),
        "plan_path": str(plan_path.absolute()),
        "plan_file_sha256": file_sha256(plan_path),
        "plan_fingerprint": plan["plan_fingerprint"],
        "authorized_actions": plan["actions"],
        "fresh_cleanup_authorized": True,
        "persistent_disable_authorized": False,
        "global_reset_failed_authorized": False,
        "authorization_basis": authorization_basis,
        "explicit_user_instruction_id": explicit_user_instruction_id,
        "authorized_uid": os.getuid(),
        "manager_generation": plan["manager_generation"],
        "executable_bindings": _executable_bindings(),
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {
        **body,
        "authorization_fingerprint": stable_fingerprint(body),
    }


def _executable_bindings() -> dict[str, object]:
    rows = {
        "cleanup_tool": Path(__file__).resolve(),
        "environment_auditor": (
            Path(__file__).resolve().parent
            / "cure_lite_v24_runtime_environment.py"
        ),
        "python": Path(sys.executable).resolve(),
        "systemctl": Path(SYSTEMCTL_PATH).resolve(),
    }
    bindings: dict[str, object] = {}
    for name, path in rows.items():
        metadata = path.lstat()
        if (
            not path.is_absolute()
            or not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or path.resolve(strict=True) != path
            or metadata.st_nlink != 1
        ):
            raise PermissionError(f"{name} executable binding is unsafe")
        bindings[name] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
        }
    return bindings


def validate_partial_cleanup_failure_lineage(
    *,
    plan_path: Path,
    original_authorization_path: Path,
    intent_path: Path,
    terminal_failure_path: Path,
) -> dict[str, object]:
    plan_target = plan_path.absolute()
    authorization_target = original_authorization_path.absolute()
    intent_target = intent_path.absolute()
    terminal_target = terminal_failure_path.absolute()
    plan = validate_plan(load_sealed_json(plan_target))
    authorization = _validate_archived_authorization(
        load_sealed_json(authorization_target),
        plan_path=plan_target,
        plan=plan,
    )
    intent = dict(
        _require_exact_keys(
            load_sealed_json(intent_target),
            _INTENT_KEYS,
            name="original cleanup intent",
        )
    )
    _without_fingerprint(intent, "intent_fingerprint")
    _validate_no_payload(intent)
    if (
        intent.get("schema_version") != INTENT_SCHEMA
        or intent.get("plan_file_sha256") != file_sha256(plan_target)
        or intent.get("plan_fingerprint") != plan["plan_fingerprint"]
        or intent.get("authorization_file_sha256")
        != file_sha256(authorization_target)
        or intent.get("authorization_fingerprint")
        != authorization["authorization_fingerprint"]
        or intent.get("boot_id") != plan["boot_id"]
        or intent.get("manager_generation") != plan["manager_generation"]
        or intent.get("actions") != plan["actions"]
    ):
        raise PermissionError("original cleanup intent lineage changed")
    before = intent.get("before")
    if (
        not isinstance(before, dict)
        or set(before) != {GPU0_CONFLICT_UNIT, GPU2_DNANET_UNIT}
    ):
        raise PermissionError("original cleanup intent snapshots changed")
    snapshots = plan["unit_snapshots"]
    assert isinstance(snapshots, dict)
    for unit in (GPU0_CONFLICT_UNIT, GPU2_DNANET_UNIT):
        observed = before.get(unit)
        if not isinstance(observed, dict):
            raise PermissionError("original cleanup snapshot is malformed")
        _validate_snapshot_identity(snapshots[unit], observed)

    issued = _parse_utc_timestamp(
        authorization["issued_at_utc"],
        name="original cleanup authorization issuance",
    )
    expires = _parse_utc_timestamp(
        authorization["expires_at_utc"],
        name="original cleanup authorization expiry",
    )
    intent_created = _parse_utc_timestamp(
        intent.get("created_at_utc"),
        name="original cleanup intent creation",
    )
    if not issued <= intent_created <= expires:
        raise PermissionError(
            "original cleanup intent was outside its authorization window"
        )

    terminal = dict(
        _require_exact_keys(
            load_sealed_json(terminal_target),
            _TERMINAL_FAILURE_KEYS,
            name="original cleanup terminal failure",
        )
    )
    _without_fingerprint(
        terminal,
        "terminal_failure_fingerprint",
    )
    _validate_no_payload(terminal)
    inflight = terminal.get("inflight_action")
    expected_inflight_keys = {
        "action",
        "argv",
        "started_at_utc",
        "dispatch_attempted",
        "completion_observed",
        "returncode",
        "stdout",
        "stderr",
    }
    if not isinstance(inflight, dict) or set(inflight) != expected_inflight_keys:
        raise PermissionError("original cleanup in-flight evidence changed")
    expected_action = plan["actions"][0]
    expected_argv = _command_for_action(expected_action)
    expected_stderr = (
        "Created symlink "
        f"/run/user/{os.getuid()}/systemd/user/{GPU0_CONFLICT_UNIT} "
        "-> /dev/null.\n"
    )
    if (
        terminal.get("schema_version") != TERMINAL_FAILURE_SCHEMA
        or terminal.get("intent_fingerprint")
        != intent["intent_fingerprint"]
        or terminal.get("completed_action_receipt_fingerprints") != []
        or terminal.get("error_type") != "RuntimeError"
        or terminal.get("error_message")
        != "runtime mask was not observed immediately"
        or terminal.get("automatic_rollback_performed") is not False
        or terminal.get("runtime_mask_may_remain") is not False
        or inflight.get("action") != expected_action
        or inflight.get("argv") != expected_argv
        or inflight.get("dispatch_attempted") is not True
        or inflight.get("completion_observed") is not True
        or inflight.get("returncode") != 0
        or inflight.get("stdout") != ""
        or inflight.get("stderr") != expected_stderr
    ):
        raise PermissionError("original partial cleanup condition changed")
    started = _parse_utc_timestamp(
        inflight.get("started_at_utc"),
        name="original mask action start",
    )
    terminal_created = _parse_utc_timestamp(
        terminal.get("created_at_utc"),
        name="original cleanup terminal creation",
    )
    if not (
        issued <= intent_created <= started <= terminal_created <= expires
    ):
        raise PermissionError(
            "original partial cleanup chronology changed"
        )
    roots = {
        "plan": _evidence_root(
            plan_target,
            fingerprint_field="plan_fingerprint",
        ),
        "original_authorization": _evidence_root(
            authorization_target,
            fingerprint_field="authorization_fingerprint",
        ),
        "original_intent": _evidence_root(
            intent_target,
            fingerprint_field="intent_fingerprint",
        ),
        "original_terminal_failure": _evidence_root(
            terminal_target,
            fingerprint_field="terminal_failure_fingerprint",
        ),
    }
    return {
        "plan": plan,
        "authorization": authorization,
        "intent": intent,
        "terminal_failure": terminal,
        "roots": roots,
    }


def build_recovery_authorization(
    *,
    plan_path: Path,
    original_authorization_path: Path,
    intent_path: Path,
    terminal_failure_path: Path,
    authorization_basis: str,
    explicit_user_instruction_id: str,
    validity_seconds: int = 300,
    snapshot_reader: Callable[[str], dict[str, object]] = (
        query_unit_snapshot
    ),
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ] = read_runtime_mask_activation_guard,
) -> dict[str, object]:
    if (
        not isinstance(authorization_basis, str)
        or not authorization_basis.strip()
    ):
        raise ValueError("recovery authorization basis must be nonempty")
    if explicit_user_instruction_id != RECOVERY_USER_INSTRUCTION_ID:
        raise PermissionError("explicit user recovery instruction is absent")
    if (
        isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("recovery authorization validity must be 1..300 seconds")
    lineage = validate_partial_cleanup_failure_lineage(
        plan_path=plan_path,
        original_authorization_path=original_authorization_path,
        intent_path=intent_path,
        terminal_failure_path=terminal_failure_path,
    )
    plan = lineage["plan"]
    assert isinstance(plan, dict)
    generation = validate_live_manager_generation(
        plan["manager_generation"]
    )
    plan_snapshots = plan["unit_snapshots"]
    assert isinstance(plan_snapshots, dict)
    before = {
        GPU0_CONFLICT_UNIT: snapshot_reader(GPU0_CONFLICT_UNIT),
        GPU2_DNANET_UNIT: snapshot_reader(GPU2_DNANET_UNIT),
    }
    _validate_snapshot_identity(
        plan_snapshots[GPU0_CONFLICT_UNIT],
        before[GPU0_CONFLICT_UNIT],
    )
    _validate_snapshot_identity(
        plan_snapshots[GPU2_DNANET_UNIT],
        before[GPU2_DNANET_UNIT],
    )
    conflict = before[GPU0_CONFLICT_UNIT]
    if (
        conflict.get("UnitFileState") != "enabled"
        or conflict.get("ActiveState") != "activating"
        or conflict.get("SubState") != "auto-restart"
        or conflict.get("TriggeredBy") != ""
        or conflict.get("Triggers") != ""
    ):
        raise PermissionError(
            "recovery authorization requires the exact restart-loop state"
        )
    activation_guard = _validate_recovery_guard(
        activation_guard_reader(generation)
    )
    if activation_guard["observed_unit_file_state"] != conflict["UnitFileState"]:
        raise PermissionError("activation guard and unit state disagree")
    issued = datetime.now(timezone.utc)
    roots = lineage["roots"]
    assert isinstance(roots, dict)
    body: dict[str, object] = {
        "schema_version": RECOVERY_AUTHORIZATION_SCHEMA,
        "candidate": "GCR-PACRE-v24",
        "scope": "user-systemd-partial-cleanup-recovery",
        "created_at_utc": utc_now(),
        "roots": roots,
        "authorized_action": plan["actions"][1],
        "partial_failure_condition": PARTIAL_FAILURE_CONDITION,
        "activation_guard": activation_guard,
        "before": before,
        "authorization_basis": authorization_basis,
        "explicit_user_instruction_id": explicit_user_instruction_id,
        "authorized_uid": os.getuid(),
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (
            issued + timedelta(seconds=validity_seconds)
        ).isoformat().replace("+00:00", "Z"),
        "manager_generation": generation,
        "executable_bindings": _executable_bindings(),
        "persistent_disable_authorized": False,
        "global_reset_failed_authorized": False,
        "automatic_retry_authorized": False,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {
        **body,
        "recovery_authorization_fingerprint": stable_fingerprint(body),
    }


def validate_recovery_authorization(
    value: object,
    *,
    plan_path: Path,
    original_authorization_path: Path,
    intent_path: Path,
    terminal_failure_path: Path,
    snapshot_reader: Callable[[str], dict[str, object]] = (
        query_unit_snapshot
    ),
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ] = read_runtime_mask_activation_guard,
    require_fresh: bool = True,
) -> dict[str, object]:
    authorization = dict(
        _require_exact_keys(
            value,
            _RECOVERY_AUTHORIZATION_KEYS,
            name="cleanup recovery authorization",
        )
    )
    _without_fingerprint(
        authorization,
        "recovery_authorization_fingerprint",
    )
    _validate_no_payload(authorization)
    lineage = validate_partial_cleanup_failure_lineage(
        plan_path=plan_path,
        original_authorization_path=original_authorization_path,
        intent_path=intent_path,
        terminal_failure_path=terminal_failure_path,
    )
    plan = lineage["plan"]
    roots = lineage["roots"]
    assert isinstance(plan, dict)
    assert isinstance(roots, dict)
    if (
        authorization.get("schema_version")
        != RECOVERY_AUTHORIZATION_SCHEMA
        or authorization.get("candidate") != "GCR-PACRE-v24"
        or authorization.get("scope")
        != "user-systemd-partial-cleanup-recovery"
        or authorization.get("roots") != roots
        or authorization.get("authorized_action") != plan["actions"][1]
        or authorization.get("partial_failure_condition")
        != PARTIAL_FAILURE_CONDITION
        or authorization.get("explicit_user_instruction_id")
        != RECOVERY_USER_INSTRUCTION_ID
        or authorization.get("authorized_uid") != os.getuid()
        or authorization.get("manager_generation")
        != plan["manager_generation"]
        or authorization.get("executable_bindings") != _executable_bindings()
        or authorization.get("persistent_disable_authorized") is not False
        or authorization.get("global_reset_failed_authorized") is not False
        or authorization.get("automatic_retry_authorized") is not False
        or not isinstance(authorization.get("authorization_basis"), str)
        or not str(authorization["authorization_basis"]).strip()
    ):
        raise PermissionError("cleanup recovery authorization changed")
    issued = _parse_utc_timestamp(
        authorization.get("issued_at_utc"),
        name="cleanup recovery authorization issuance",
    )
    expires = _parse_utc_timestamp(
        authorization.get("expires_at_utc"),
        name="cleanup recovery authorization expiry",
    )
    now = datetime.now(timezone.utc)
    if (
        expires <= issued
        or expires - issued > timedelta(seconds=300)
        or (require_fresh and not issued <= now <= expires)
    ):
        raise PermissionError("cleanup recovery authorization is stale")
    generation = validate_live_manager_generation(
        authorization["manager_generation"]
    )
    guard = _validate_recovery_guard(
        activation_guard_reader(generation)
    )
    if guard != authorization.get("activation_guard"):
        raise PermissionError("cleanup recovery activation guard drifted")
    before = authorization.get("before")
    if (
        not isinstance(before, dict)
        or set(before) != {GPU0_CONFLICT_UNIT, GPU2_DNANET_UNIT}
    ):
        raise PermissionError("cleanup recovery snapshots changed")
    current = {
        GPU0_CONFLICT_UNIT: snapshot_reader(GPU0_CONFLICT_UNIT),
        GPU2_DNANET_UNIT: snapshot_reader(GPU2_DNANET_UNIT),
    }
    for unit in (GPU0_CONFLICT_UNIT, GPU2_DNANET_UNIT):
        expected = before.get(unit)
        if not isinstance(expected, dict):
            raise PermissionError("cleanup recovery snapshot is malformed")
        _validate_snapshot_identity(expected, current[unit])
    conflict = current[GPU0_CONFLICT_UNIT]
    if (
        conflict.get("UnitFileState") != "enabled"
        or conflict.get("ActiveState") != "activating"
        or conflict.get("SubState") != "auto-restart"
        or conflict.get("TriggeredBy") != ""
        or conflict.get("Triggers") != ""
    ):
        raise PermissionError("cleanup recovery precondition is not live")
    return authorization


def _command_for_action(action: Mapping[str, object]) -> list[str]:
    unit = _require_unit_name(action["unit_name"])
    name = action["action"]
    if name == "mask-runtime":
        return [
            SYSTEMCTL_PATH,
            "--user",
            "mask",
            "--runtime",
            unit,
        ]
    if name == "stop":
        return [SYSTEMCTL_PATH, "--user", "stop", unit]
    raise ValueError("cleanup action is unsupported")


def _validate_snapshot_identity(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
    *,
    require_original_unit_file_state: bool = True,
) -> None:
    for key in (
        "Id",
        "LoadState",
        "FragmentPath",
        "DropInPaths",
        "Restart",
        "RestartUSec",
        "Environment",
        "ExecStart",
        "TriggeredBy",
        "Triggers",
        "WantedBy",
        "RequiredBy",
        "PartOf",
        "fragment_file_sha256",
        "dropin_file_sha256",
        "fragment_owner_uid",
        "fragment_group_gid",
        "fragment_group_name",
        "fragment_group_member_uids",
        "fragment_mode",
    ):
        if observed[key] != expected[key]:
            raise RuntimeError(f"unit identity drifted at {key}")
    if (
        require_original_unit_file_state
        and observed["UnitFileState"] != expected["UnitFileState"]
    ):
        raise RuntimeError("unit identity drifted at UnitFileState")
    try:
        if int(observed["NRestarts"]) < int(expected["NRestarts"]):
            raise RuntimeError("NRestarts moved backwards")
    except (TypeError, ValueError) as error:
        raise RuntimeError("NRestarts is malformed") from error


def execute_cleanup(
    *,
    plan_path: Path,
    authorization_path: Path,
    receipt_directory: Path,
    snapshot_reader: Callable[[str], dict[str, object]] = (
        query_unit_snapshot
    ),
    command_runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] = run_systemctl,
) -> dict[str, object]:
    plan = validate_plan(load_sealed_json(plan_path))
    authorization = validate_authorization(
        load_sealed_json(authorization_path),
        plan_path=plan_path,
        plan=plan,
    )
    inventory_path = Path(plan["inventory_receipt_path"])
    inventory_receipt = validate_environment_receipt(
        load_sealed_json(inventory_path)
    )
    if (
        file_sha256(inventory_path) != plan["inventory_receipt_file_sha256"]
        or inventory_receipt["receipt_fingerprint"]
        != plan["inventory_receipt_fingerprint"]
        or _manager_generation_from_inventory(inventory_receipt)
        != plan["manager_generation"]
    ):
        raise RuntimeError("inventory receipt drifted")
    if _boot_id() != plan["boot_id"]:
        raise RuntimeError("boot changed after cleanup planning")
    generation = validate_live_manager_generation(plan["manager_generation"])
    snapshots = plan["unit_snapshots"]
    assert isinstance(snapshots, dict)
    before = {}
    for unit, expected in snapshots.items():
        observed = snapshot_reader(unit)
        assert isinstance(expected, dict)
        _validate_snapshot_identity(expected, observed)
        before[unit] = observed

    receipt_root = receipt_directory.absolute()
    receipt_parent_fd = _open_private_parent(receipt_root)
    os.close(receipt_parent_fd)
    receipt_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    receipt_root.chmod(0o700)
    _fsync_parent(receipt_root)
    intent_body: dict[str, object] = {
        "schema_version": INTENT_SCHEMA,
        "created_at_utc": utc_now(),
        "plan_file_sha256": file_sha256(plan_path),
        "plan_fingerprint": plan["plan_fingerprint"],
        "authorization_file_sha256": file_sha256(
            authorization_path
        ),
        "authorization_fingerprint": authorization[
            "authorization_fingerprint"
        ],
        "boot_id": plan["boot_id"],
        "manager_generation": generation,
        "before": before,
        "actions": plan["actions"],
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    intent = write_create_once_json(
        receipt_root / "cleanup-intent.json",
        intent_body,
        fingerprint_key="intent_fingerprint",
    )

    action_receipts: list[dict[str, object]] = []
    inflight_action: dict[str, object] | None = None
    try:
        for raw_action in plan["actions"]:
            assert isinstance(raw_action, dict)
            validate_live_manager_generation(plan["manager_generation"])
            immediate_before = snapshot_reader(GPU0_CONFLICT_UNIT)
            protected_before = snapshot_reader(GPU2_DNANET_UNIT)
            _validate_snapshot_identity(
                snapshots[GPU0_CONFLICT_UNIT],
                immediate_before,
                require_original_unit_file_state=(
                    raw_action["action"] == "mask-runtime"
                ),
            )
            _validate_snapshot_identity(
                snapshots[GPU2_DNANET_UNIT],
                protected_before,
            )
            if raw_action["action"] == "stop" and (
                immediate_before["UnitFileState"] != "masked-runtime"
            ):
                raise RuntimeError("stop is forbidden before live runtime mask")
            argv = _command_for_action(raw_action)
            _validate_systemctl_argv(argv)
            started = utc_now()
            inflight_action = {
                "action": raw_action,
                "argv": argv,
                "started_at_utc": started,
                "dispatch_attempted": True,
                "completion_observed": False,
            }
            completed = command_runner(argv)
            inflight_action.update({
                "completion_observed": True,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            })
            validate_live_manager_generation(plan["manager_generation"])
            immediate_after = snapshot_reader(GPU0_CONFLICT_UNIT)
            protected_after = snapshot_reader(GPU2_DNANET_UNIT)
            _validate_snapshot_identity(
                snapshots[GPU0_CONFLICT_UNIT],
                immediate_after,
                require_original_unit_file_state=False,
            )
            _validate_snapshot_identity(
                snapshots[GPU2_DNANET_UNIT],
                protected_after,
            )
            if immediate_after["UnitFileState"] != "masked-runtime":
                raise RuntimeError("runtime mask was not observed immediately")
            if raw_action["action"] == "stop" and (
                immediate_after["ActiveState"] != "inactive"
                or immediate_after["SubState"] != "dead"
            ):
                raise RuntimeError("GPU0 conflict did not become inactive")
            action_body: dict[str, object] = {
                "schema_version": ACTION_RECEIPT_SCHEMA,
                "created_at_utc": utc_now(),
                "started_at_utc": started,
                "intent_fingerprint": intent["intent_fingerprint"],
                "action": raw_action,
                "argv": argv,
                "shell": False,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "manager_generation": plan["manager_generation"],
                "before": immediate_before,
                "after": immediate_after,
                "protected_before": protected_before,
                "protected_after": protected_after,
                "payload_authority": "none",
                "D_R_payload_accessed": False,
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            }
            action_receipt = write_create_once_json(
                receipt_root / f"action-{raw_action['ordinal']:03d}.json",
                action_body,
                fingerprint_key="action_receipt_fingerprint",
            )
            action_receipts.append(action_receipt)
            inflight_action = None
            if completed.returncode != 0:
                raise RuntimeError(
                    "authorized cleanup action failed; inspect action receipt"
                )
        validate_live_manager_generation(plan["manager_generation"])
        after = {
            GPU0_CONFLICT_UNIT: snapshot_reader(GPU0_CONFLICT_UNIT),
            GPU2_DNANET_UNIT: snapshot_reader(GPU2_DNANET_UNIT),
        }
        _validate_snapshot_identity(
            snapshots[GPU0_CONFLICT_UNIT],
            after[GPU0_CONFLICT_UNIT],
            require_original_unit_file_state=False,
        )
        _validate_snapshot_identity(
            snapshots[GPU2_DNANET_UNIT],
            after[GPU2_DNANET_UNIT],
        )
        validate_live_manager_generation(plan["manager_generation"])
    except BaseException as error:
        completed_runtime_mask_inflight = bool(
            isinstance(inflight_action, dict)
            and isinstance(inflight_action.get("action"), dict)
            and inflight_action["action"].get("action") == "mask-runtime"
            and inflight_action.get("dispatch_attempted") is True
            and inflight_action.get("completion_observed") is True
            and inflight_action.get("returncode") == 0
        )
        failure_body: dict[str, object] = {
            "schema_version": TERMINAL_FAILURE_SCHEMA,
            "created_at_utc": utc_now(),
            "intent_fingerprint": intent["intent_fingerprint"],
            "completed_action_receipt_fingerprints": [
                item["action_receipt_fingerprint"] for item in action_receipts
            ],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "inflight_action": inflight_action,
            "automatic_rollback_performed": False,
            "runtime_mask_may_remain": (
                bool(action_receipts) or completed_runtime_mask_inflight
            ),
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        try:
            write_create_once_json(
                receipt_root / "cleanup-terminal-failure.json",
                failure_body,
                fingerprint_key="terminal_failure_fingerprint",
            )
        finally:
            raise
    masked = {
        item["unit_name"]
        for item in plan["actions"]
        if item["action"] == "mask-runtime"
    }
    stopped = {
        item["unit_name"]
        for item in plan["actions"]
        if item["action"] == "stop"
    }
    for unit in masked:
        if after[unit]["UnitFileState"] != "masked-runtime":
            raise RuntimeError(f"{unit} is not runtime-masked")
    for unit in stopped:
        if (
            after[unit]["ActiveState"] != "inactive"
            or after[unit]["SubState"] != "dead"
        ):
            raise RuntimeError(f"{unit} is not inactive after stop")
    final_body: dict[str, object] = {
        "schema_version": FINAL_RECEIPT_SCHEMA,
        "created_at_utc": utc_now(),
        "intent_fingerprint": intent["intent_fingerprint"],
        "action_receipt_fingerprints": [
            item["action_receipt_fingerprint"]
            for item in action_receipts
        ],
        "boot_id": plan["boot_id"],
        "manager_generation": plan["manager_generation"],
        "after": after,
        "cleanup_mode": NORMAL_CLEANUP_MODE,
        "activation_guard": {
            "mode": NORMAL_GUARD_MODE,
            "unit_name": GPU0_CONFLICT_UNIT,
            "observed_unit_file_state": "masked-runtime",
        },
        "partial_lineage": None,
        "passed": True,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return write_create_once_json(
        receipt_root / "cleanup-receipt.json",
        final_body,
        fingerprint_key="cleanup_receipt_fingerprint",
    )


def execute_partial_cleanup_recovery(
    *,
    plan_path: Path,
    original_authorization_path: Path,
    intent_path: Path,
    terminal_failure_path: Path,
    recovery_authorization_path: Path,
    receipt_directory: Path,
    snapshot_reader: Callable[[str], dict[str, object]] = (
        query_unit_snapshot
    ),
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ] = read_runtime_mask_activation_guard,
    command_runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] = run_systemctl,
) -> dict[str, object]:
    plan_target = plan_path.absolute()
    original_authorization_target = original_authorization_path.absolute()
    original_intent_target = intent_path.absolute()
    terminal_target = terminal_failure_path.absolute()
    recovery_authorization_target = recovery_authorization_path.absolute()
    recovery_authorization = validate_recovery_authorization(
        load_sealed_json(recovery_authorization_target),
        plan_path=plan_target,
        original_authorization_path=original_authorization_target,
        intent_path=original_intent_target,
        terminal_failure_path=terminal_target,
        snapshot_reader=snapshot_reader,
        activation_guard_reader=activation_guard_reader,
    )
    lineage = validate_partial_cleanup_failure_lineage(
        plan_path=plan_target,
        original_authorization_path=original_authorization_target,
        intent_path=original_intent_target,
        terminal_failure_path=terminal_target,
    )
    plan = lineage["plan"]
    original_roots = lineage["roots"]
    terminal = lineage["terminal_failure"]
    assert isinstance(plan, dict)
    assert isinstance(original_roots, dict)
    assert isinstance(terminal, dict)
    generation = validate_live_manager_generation(
        recovery_authorization["manager_generation"]
    )
    before = {
        GPU0_CONFLICT_UNIT: snapshot_reader(GPU0_CONFLICT_UNIT),
        GPU2_DNANET_UNIT: snapshot_reader(GPU2_DNANET_UNIT),
    }
    authorization_before = recovery_authorization["before"]
    assert isinstance(authorization_before, dict)
    for unit in (GPU0_CONFLICT_UNIT, GPU2_DNANET_UNIT):
        expected = authorization_before.get(unit)
        if not isinstance(expected, dict):
            raise PermissionError("recovery authorization snapshot is malformed")
        _validate_snapshot_identity(expected, before[unit])
    conflict_before = before[GPU0_CONFLICT_UNIT]
    if (
        conflict_before.get("UnitFileState") != "enabled"
        or conflict_before.get("ActiveState") != "activating"
        or conflict_before.get("SubState") != "auto-restart"
        or conflict_before.get("TriggeredBy") != ""
        or conflict_before.get("Triggers") != ""
    ):
        raise PermissionError("recovery stop precondition changed")
    guard_before = _validate_recovery_guard(
        activation_guard_reader(generation)
    )
    if guard_before != recovery_authorization["activation_guard"]:
        raise PermissionError("recovery activation guard changed before intent")

    receipt_root = receipt_directory.absolute()
    receipt_parent_fd = _open_private_parent(receipt_root)
    os.close(receipt_parent_fd)
    receipt_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    receipt_root.chmod(0o700)
    _fsync_parent(receipt_root)
    recovery_authorization_root = _evidence_root(
        recovery_authorization_target,
        fingerprint_field="recovery_authorization_fingerprint",
    )
    action = dict(recovery_authorization["authorized_action"])
    intent_body: dict[str, object] = {
        "schema_version": RECOVERY_INTENT_SCHEMA,
        "created_at_utc": utc_now(),
        "roots": original_roots,
        "recovery_authorization_file_sha256": file_sha256(
            recovery_authorization_target
        ),
        "recovery_authorization_fingerprint": recovery_authorization[
            "recovery_authorization_fingerprint"
        ],
        "manager_generation": generation,
        "before": before,
        "activation_guard": guard_before,
        "action": action,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    recovery_intent = write_create_once_json(
        receipt_root / "cleanup-recovery-intent.json",
        intent_body,
        fingerprint_key="recovery_intent_fingerprint",
    )

    inflight_action: dict[str, object] | None = None
    action_receipt: dict[str, object] | None = None
    try:
        validate_live_manager_generation(generation)
        immediate_before = snapshot_reader(GPU0_CONFLICT_UNIT)
        protected_before = snapshot_reader(GPU2_DNANET_UNIT)
        _validate_snapshot_identity(
            conflict_before,
            immediate_before,
        )
        _validate_snapshot_identity(
            before[GPU2_DNANET_UNIT],
            protected_before,
        )
        if (
            immediate_before.get("UnitFileState") != "enabled"
            or immediate_before.get("ActiveState") != "activating"
            or immediate_before.get("SubState") != "auto-restart"
            or immediate_before.get("TriggeredBy") != ""
            or immediate_before.get("Triggers") != ""
        ):
            raise PermissionError("recovery stop pre-dispatch state changed")
        guard_dispatch = _validate_recovery_guard(
            activation_guard_reader(generation)
        )
        if guard_dispatch != guard_before:
            raise PermissionError(
                "recovery activation guard changed before dispatch"
            )
        argv = _command_for_action(action)
        if argv != [
            SYSTEMCTL_PATH,
            "--user",
            "stop",
            GPU0_CONFLICT_UNIT,
        ]:
            raise PermissionError("recovery command is not the exact stop")
        _validate_systemctl_argv(argv)
        started = utc_now()
        inflight_action = {
            "action": action,
            "argv": argv,
            "started_at_utc": started,
            "dispatch_attempted": True,
            "completion_observed": False,
        }
        completed = command_runner(argv)
        inflight_action.update({
            "completion_observed": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        })
        validate_live_manager_generation(generation)
        immediate_after = snapshot_reader(GPU0_CONFLICT_UNIT)
        protected_after = snapshot_reader(GPU2_DNANET_UNIT)
        _validate_snapshot_identity(
            immediate_before,
            immediate_after,
        )
        _validate_snapshot_identity(
            protected_before,
            protected_after,
        )
        guard_after = _validate_recovery_guard(
            activation_guard_reader(generation)
        )
        if (
            guard_after != guard_before
            or immediate_after.get("UnitFileState") != "enabled"
            or immediate_after.get("ActiveState") != "inactive"
            or immediate_after.get("SubState") != "dead"
            or immediate_after.get("TriggeredBy") != ""
            or immediate_after.get("Triggers") != ""
            or immediate_after.get("NRestarts")
            != immediate_before.get("NRestarts")
        ):
            raise RuntimeError(
                "recovery stop did not produce the exact guarded quiescent state"
            )
        action_body: dict[str, object] = {
            "schema_version": RECOVERY_ACTION_RECEIPT_SCHEMA,
            "created_at_utc": utc_now(),
            "started_at_utc": started,
            "recovery_intent_fingerprint": recovery_intent[
                "recovery_intent_fingerprint"
            ],
            "action": action,
            "argv": argv,
            "shell": False,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "manager_generation": generation,
            "before": immediate_before,
            "after": immediate_after,
            "protected_before": protected_before,
            "protected_after": protected_after,
            "activation_guard_before": guard_before,
            "activation_guard_after": guard_after,
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        action_receipt = write_create_once_json(
            receipt_root / "cleanup-recovery-action-001.json",
            action_body,
            fingerprint_key="recovery_action_receipt_fingerprint",
        )
        if completed.returncode != 0:
            raise RuntimeError("authorized recovery stop returned nonzero")
        inflight_action = None
        final_after = {
            GPU0_CONFLICT_UNIT: snapshot_reader(GPU0_CONFLICT_UNIT),
            GPU2_DNANET_UNIT: snapshot_reader(GPU2_DNANET_UNIT),
        }
        _validate_snapshot_identity(
            immediate_after,
            final_after[GPU0_CONFLICT_UNIT],
        )
        _validate_snapshot_identity(
            protected_after,
            final_after[GPU2_DNANET_UNIT],
        )
        final_guard = _validate_recovery_guard(
            activation_guard_reader(generation)
        )
        if (
            final_guard != guard_after
            or final_after[GPU0_CONFLICT_UNIT].get("UnitFileState")
            != "enabled"
            or final_after[GPU0_CONFLICT_UNIT].get("ActiveState")
            != "inactive"
            or final_after[GPU0_CONFLICT_UNIT].get("SubState") != "dead"
            or final_after[GPU0_CONFLICT_UNIT].get("NRestarts")
            != immediate_after.get("NRestarts")
        ):
            raise RuntimeError("recovery final state did not remain quiescent")
        validate_live_manager_generation(generation)
    except BaseException as error:
        failure_body: dict[str, object] = {
            "schema_version": RECOVERY_TERMINAL_FAILURE_SCHEMA,
            "created_at_utc": utc_now(),
            "recovery_intent_fingerprint": recovery_intent[
                "recovery_intent_fingerprint"
            ],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "inflight_action": inflight_action,
            "completed_recovery_action_receipt_fingerprints": (
                []
                if action_receipt is None
                else [
                    action_receipt[
                        "recovery_action_receipt_fingerprint"
                    ]
                ]
            ),
            "automatic_rollback_performed": False,
            "automatic_retry_performed": False,
            "persistent_disable_performed": False,
            "global_reset_failed_performed": False,
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        try:
            write_create_once_json(
                receipt_root / "cleanup-recovery-terminal-failure.json",
                failure_body,
                fingerprint_key="recovery_terminal_failure_fingerprint",
            )
        finally:
            raise

    assert action_receipt is not None
    recovery_intent_root = _evidence_root(
        receipt_root / "cleanup-recovery-intent.json",
        fingerprint_field="recovery_intent_fingerprint",
    )
    recovery_action_root = _evidence_root(
        receipt_root / "cleanup-recovery-action-001.json",
        fingerprint_field="recovery_action_receipt_fingerprint",
    )
    partial_lineage = {
        **original_roots,
        "recovery_authorization": recovery_authorization_root,
        "recovery_intent": recovery_intent_root,
        "recovery_action_receipt": recovery_action_root,
        "legacy_runtime_mask_may_remain_false_reconciled": (
            terminal.get("runtime_mask_may_remain") is False
        ),
        "original_stop_dispatched": False,
    }
    final_body: dict[str, object] = {
        "schema_version": FINAL_RECEIPT_SCHEMA,
        "created_at_utc": utc_now(),
        "intent_fingerprint": recovery_intent[
            "recovery_intent_fingerprint"
        ],
        "action_receipt_fingerprints": [
            action_receipt["recovery_action_receipt_fingerprint"]
        ],
        "boot_id": plan["boot_id"],
        "manager_generation": generation,
        "after": final_after,
        "cleanup_mode": RECOVERY_CLEANUP_MODE,
        "activation_guard": final_guard,
        "partial_lineage": partial_lineage,
        "passed": True,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return write_create_once_json(
        receipt_root / "cleanup-receipt.json",
        final_body,
        fingerprint_key="cleanup_receipt_fingerprint",
    )


def validate_final_cleanup_receipt(value: object) -> dict[str, object]:
    receipt = dict(
        _require_exact_keys(
            value,
            _FINAL_RECEIPT_KEYS,
            name="final cleanup receipt",
        )
    )
    _without_fingerprint(receipt, "cleanup_receipt_fingerprint")
    _validate_no_payload(receipt)
    after = receipt.get("after")
    actions = receipt.get("action_receipt_fingerprints")
    if (
        receipt.get("schema_version") != FINAL_RECEIPT_SCHEMA
        or receipt.get("passed") is not True
        or not isinstance(receipt.get("created_at_utc"), str)
        or not isinstance(receipt.get("intent_fingerprint"), str)
        or _SHA256_RE.fullmatch(str(receipt["intent_fingerprint"])) is None
        or not isinstance(receipt.get("boot_id"), str)
        or _BOOT_ID_RE.fullmatch(str(receipt["boot_id"])) is None
        or not isinstance(receipt.get("manager_generation"), dict)
        or not isinstance(after, dict)
        or set(after) != {GPU0_CONFLICT_UNIT, GPU2_DNANET_UNIT}
        or not isinstance(actions, list)
        or any(
            not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None
            for item in actions
        )
    ):
        raise PermissionError("final cleanup receipt identity changed")
    conflict = after[GPU0_CONFLICT_UNIT]
    if not isinstance(conflict, dict):
        raise PermissionError("final cleanup conflict snapshot is malformed")
    mode = receipt.get("cleanup_mode")
    if mode == NORMAL_CLEANUP_MODE:
        expected_guard = {
            "mode": NORMAL_GUARD_MODE,
            "unit_name": GPU0_CONFLICT_UNIT,
            "observed_unit_file_state": "masked-runtime",
        }
        if (
            actions is None
            or len(actions) != 2
            or receipt.get("activation_guard") != expected_guard
            or receipt.get("partial_lineage") is not None
            or conflict.get("UnitFileState") != "masked-runtime"
            or conflict.get("ActiveState") != "inactive"
            or conflict.get("SubState") != "dead"
        ):
            raise PermissionError("normal cleanup receipt semantics changed")
    elif mode == RECOVERY_CLEANUP_MODE:
        guard = _validate_recovery_guard(receipt.get("activation_guard"))
        lineage = receipt.get("partial_lineage")
        expected_lineage_keys = {
            "plan",
            "original_authorization",
            "original_intent",
            "original_terminal_failure",
            "recovery_authorization",
            "recovery_intent",
            "recovery_action_receipt",
            "legacy_runtime_mask_may_remain_false_reconciled",
            "original_stop_dispatched",
        }
        if (
            len(actions) != 1
            or not isinstance(lineage, dict)
            or set(lineage) != expected_lineage_keys
            or lineage.get(
                "legacy_runtime_mask_may_remain_false_reconciled"
            )
            is not True
            or lineage.get("original_stop_dispatched") is not False
            or conflict.get("UnitFileState") != "enabled"
            or conflict.get("ActiveState") != "inactive"
            or conflict.get("SubState") != "dead"
            or guard.get("observed_unit_file_state")
            != conflict.get("UnitFileState")
        ):
            raise PermissionError("recovery cleanup receipt semantics changed")
        for name in expected_lineage_keys - {
            "legacy_runtime_mask_may_remain_false_reconciled",
            "original_stop_dispatched",
        }:
            root = dict(
                _require_exact_keys(
                    lineage[name],
                    _EVIDENCE_ROOT_KEYS,
                    name=f"recovery lineage {name}",
                )
            )
            if (
                not isinstance(root.get("path"), str)
                or not Path(str(root["path"])).is_absolute()
                or not isinstance(root.get("fingerprint_field"), str)
                or not isinstance(root.get("file_sha256"), str)
                or _SHA256_RE.fullmatch(str(root["file_sha256"])) is None
                or not isinstance(root.get("fingerprint"), str)
                or _SHA256_RE.fullmatch(str(root["fingerprint"])) is None
            ):
                raise PermissionError(
                    f"recovery lineage {name} is malformed"
                )
    else:
        raise PermissionError("cleanup receipt mode is unsupported")
    return receipt


def _build_plan_cli(args: argparse.Namespace) -> int:
    plan = build_plan(
        inventory_receipt_path=Path(args.inventory_receipt),
        mask_units=args.mask_unit,
        stop_units=args.stop_unit,
        protected_units=args.protected_unit,
    )
    body = dict(plan)
    fingerprint = body.pop("plan_fingerprint")
    written = write_create_once_json(
        args.output,
        body,
        fingerprint_key="plan_fingerprint",
    )
    if written["plan_fingerprint"] != fingerprint:
        raise RuntimeError("plan fingerprint changed during write")
    return 0


def _authorize_cli(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).absolute()
    plan = validate_plan(load_sealed_json(plan_path))
    authorization = build_authorization(
        plan_path=plan_path,
        plan=plan,
        authorization_basis=args.authorization_basis,
        explicit_user_instruction_id=args.explicit_user_instruction_id,
    )
    body = dict(authorization)
    fingerprint = body.pop("authorization_fingerprint")
    written = write_create_once_json(
        args.output,
        body,
        fingerprint_key="authorization_fingerprint",
    )
    if written["authorization_fingerprint"] != fingerprint:
        raise RuntimeError("authorization fingerprint changed during write")
    return 0


def _apply_cli(args: argparse.Namespace) -> int:
    execute_cleanup(
        plan_path=Path(args.plan).absolute(),
        authorization_path=Path(args.authorization).absolute(),
        receipt_directory=Path(args.receipt_directory).absolute(),
    )
    return 0


def _authorize_recovery_cli(args: argparse.Namespace) -> int:
    authorization = build_recovery_authorization(
        plan_path=Path(args.plan).absolute(),
        original_authorization_path=Path(
            args.original_authorization
        ).absolute(),
        intent_path=Path(args.intent).absolute(),
        terminal_failure_path=Path(args.terminal_failure).absolute(),
        authorization_basis=args.authorization_basis,
        explicit_user_instruction_id=args.explicit_user_instruction_id,
        validity_seconds=args.validity_seconds,
    )
    body = dict(authorization)
    fingerprint = body.pop("recovery_authorization_fingerprint")
    written = write_create_once_json(
        args.output,
        body,
        fingerprint_key="recovery_authorization_fingerprint",
    )
    if written["recovery_authorization_fingerprint"] != fingerprint:
        raise RuntimeError(
            "recovery authorization fingerprint changed during write"
        )
    return 0


def _apply_recovery_cli(args: argparse.Namespace) -> int:
    execute_partial_cleanup_recovery(
        plan_path=Path(args.plan).absolute(),
        original_authorization_path=Path(
            args.original_authorization
        ).absolute(),
        intent_path=Path(args.intent).absolute(),
        terminal_failure_path=Path(args.terminal_failure).absolute(),
        recovery_authorization_path=Path(
            args.recovery_authorization
        ).absolute(),
        receipt_directory=Path(args.receipt_directory).absolute(),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact-authorized CURE-Lite v24 user-systemd cleanup",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("build-plan")
    plan.add_argument("--inventory-receipt", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--mask-unit", action="append", default=[])
    plan.add_argument("--stop-unit", action="append", default=[])
    plan.add_argument("--protected-unit", action="append", default=[])
    plan.set_defaults(handler=_build_plan_cli)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--plan", required=True)
    authorize.add_argument("--output", required=True)
    authorize.add_argument("--authorization-basis", required=True)
    authorize.add_argument("--explicit-user-instruction-id", required=True)
    authorize.set_defaults(handler=_authorize_cli)

    apply = subparsers.add_parser("apply")
    apply.add_argument("--plan", required=True)
    apply.add_argument("--authorization", required=True)
    apply.add_argument("--receipt-directory", required=True)
    apply.set_defaults(handler=_apply_cli)

    recovery_authorize = subparsers.add_parser("authorize-recovery")
    recovery_authorize.add_argument("--plan", required=True)
    recovery_authorize.add_argument(
        "--original-authorization",
        required=True,
    )
    recovery_authorize.add_argument("--intent", required=True)
    recovery_authorize.add_argument("--terminal-failure", required=True)
    recovery_authorize.add_argument("--output", required=True)
    recovery_authorize.add_argument(
        "--authorization-basis",
        required=True,
    )
    recovery_authorize.add_argument(
        "--explicit-user-instruction-id",
        required=True,
    )
    recovery_authorize.add_argument(
        "--validity-seconds",
        type=int,
        default=300,
    )
    recovery_authorize.set_defaults(handler=_authorize_recovery_cli)

    recovery_apply = subparsers.add_parser("apply-recovery")
    recovery_apply.add_argument("--plan", required=True)
    recovery_apply.add_argument(
        "--original-authorization",
        required=True,
    )
    recovery_apply.add_argument("--intent", required=True)
    recovery_apply.add_argument("--terminal-failure", required=True)
    recovery_apply.add_argument(
        "--recovery-authorization",
        required=True,
    )
    recovery_apply.add_argument("--receipt-directory", required=True)
    recovery_apply.set_defaults(handler=_apply_recovery_cli)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
