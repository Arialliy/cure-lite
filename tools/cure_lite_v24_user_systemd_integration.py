#!/usr/bin/env python3
"""Create-only, payload-free supervisor-v2 user-systemd integration harness."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence


try:
    from tools import cure_lite_v24_realize_systemd_unit as realizer
except ModuleNotFoundError:
    _REALIZER_PATH = Path(__file__).with_name(
        "cure_lite_v24_realize_systemd_unit.py"
    )
    _MODULE_SPEC = importlib.util.spec_from_file_location(
        "cure_lite_v24_realize_systemd_unit_isolated", _REALIZER_PATH
    )
    if _MODULE_SPEC is None or _MODULE_SPEC.loader is None:
        raise RuntimeError("cannot load the isolated integration realizer")
    realizer = importlib.util.module_from_spec(_MODULE_SPEC)
    sys.modules[_MODULE_SPEC.name] = realizer
    _MODULE_SPEC.loader.exec_module(realizer)


AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-supervisor-v2-systemd-integration-authorization-v2"
)
INTEGRATION_TERMINAL_SCHEMA = (
    "cure-lite-v24-supervisor-v2-systemd-integration-terminal-v1"
)
INTEGRATION_RECEIPT_SCHEMA = (
    "cure-lite-v24-supervisor-v2-systemd-integration-receipt-v1"
)
REMOVAL_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-supervisor-v2-integration-removal-authorization-v1"
)
REMOVAL_STATE_SCHEMA = (
    "cure-lite-v24-supervisor-v2-integration-removal-state-v1"
)
RUNTIME_SPEC_SCHEMA = "cure-lite-v24-dr-runtime-supervisor-spec-v2"
ATTEMPT_COMMIT_SCHEMA = "cure-lite-v24-dr-attempt-commit-v2"
MATERIALIZATION_CLAIM_SCHEMA = "cure-lite-v24-dr-materialization-claim-v2"
LAUNCH_LEASE_SCHEMA = "cure-lite-v24-dr-launch-lease-v1"
PHASE_RECEIPT_SCHEMA = "cure-lite-v24-dr-runtime-phase-receipt-v1"
RUNTIME_TERMINAL_SCHEMA = "cure-lite-v24-dr-runtime-terminal-v1"
SYSTEMD_TERMINAL_SCHEMA = "cure-lite-v24-dr-systemd-terminal-v1"
DUMMY_ARTIFACT_SCHEMA = "cure-lite-v24-user-systemd-dummy-child-v1"
EXECUTION_KIND = "systemd_integration_dummy"
INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后运行"

SYSTEMCTL = "/usr/bin/systemctl"
SYSTEMD_PATH = "/usr/bin/systemd-path"
SYSTEMD_ANALYZE = "/usr/bin/systemd-analyze"

_SCENARIO = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{6,61}[a-z0-9])?-[0-9a-f]{16}"
)
_SHA = re.compile(r"[0-9a-f]{64}")
_BOOT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
_INVOCATION = re.compile(r"[0-9a-f]{32}")
_TOKENS = {
    "@@SCENARIO_ID@@", "@@SCENARIO_ROOT@@", "@@UNIT_NAME@@",
    "@@PYTHON_PATH@@", "@@SUPERVISOR_PATH@@", "@@RUNTIME_SPEC_PATH@@",
}
_EXEC_MODES = {
    "ExecCondition": "claim-materialization",
    "ExecStartPre": "verify-runtime-spec",
    "ExecStart": "run-once",
    "ExecStopPost": "record-systemd-exit",
}
_SYSTEMD_OUTCOME_CATEGORIES = {
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
_SYSTEMD_OUTCOME_KEYS = {
    "category",
    "service_result",
    "exit_code",
    "exit_status",
    "invocation_id",
    "systemd_success",
    "scientific_gate_passed",
}
_IMMUTABLE_SHADOW_KEYS = {
    "Type", "Restart", "KillMode", "SendSIGKILL", "TimeoutStopUSec",
    "FragmentPath", "DropInPaths", "Transient", "Environment",
    "UnsetEnvironment", "WorkingDirectory", "UMask", "ExitType",
    "RuntimeMaxUSec", "WatchdogUSec", "OOMPolicy", "RemainAfterExit",
    "StandardInput", "StandardOutput", "StandardError",
    "StartLimitIntervalUSec", "StartLimitBurst", "KillSignal",
    "ExecCondition", "ExecStartPre", "ExecStart", "ExecStopPost",
}
_SPEC_ARTIFACT_NAMES = {
    "attempt_commit": "attempt-commit.json",
    "materialization_claim": "materialization-claim.json",
    "stdout_log": "stdout.log",
    "stderr_log": "stderr.log",
    "heartbeat_dir": "heartbeat",
    "runtime_terminal": "runtime-terminal.json",
    "systemd_invocation_dir": "systemd-invocations",
    "launch_lease": "launch-lease.json",
    "precommit_phase_receipt": "precommit-phase.json",
    "start_ack_receipt": "start-ack.json",
    "child_prespawn_phase_receipt": "child-prespawn.json",
    "consumed_start_failure_receipt": "consumed-start-failure.json",
    "gpu_lease_release_receipt": "gpu-lease-release.json",
    "runtime_attestation": "runtime-attestation.json",
}
_AUTH_KEYS = {
    "schema_version", "authorization_fingerprint", "instruction_id",
    "authorization_basis", "issued_at_utc", "expires_at_utc", "authorized_uid",
    "scenario_id", "identity", "scenario_root", "control_root", "runtime_root",
    "unit_directory", "unit_path_policy", "manager_generation",
    "template_binding", "rendered_fragment", "runtime_spec_binding",
    "executable_bindings", "control_artifacts", "integration_authorized",
    "actual_r2_authorized", "unit_realization_authorized",
    "unit_removal_authorized", "enable_authorized", "direct_start_authorized",
    "payload_authority", "D_R_payload_accessed", "D_V_payload_accessed",
    "D_T_payload_accessed", "gpu_access_authorized",
}
_REMOVAL_AUTH_KEYS = {
    "schema_version", "removal_authorization_fingerprint", "scenario_id",
    "unit_name", "authorization_fingerprint",
    "integration_terminal_fingerprint", "runtime_spec_fingerprint",
    "supervisor_evidence", "fragment_identity", "inactive_static_state",
    "manager_generation", "unit_path_policy", "issued_at_utc",
    "expires_at_utc", "remove_authorized", "daemon_reload_authorized",
    "not_found_verification_authorized", "enable_authorized",
    "start_authorized", "payload_authority", "D_R_payload_accessed",
    "D_V_payload_accessed", "D_T_payload_accessed",
}


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
ManagerReader = Callable[[], dict[str, object]]


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def stable_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _deep_exact_equal(left: object, right: object) -> bool:
    """Compare JSON-like policy values without bool/int coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            set(left) == set(right)
            and all(_deep_exact_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _deep_exact_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _validate_unit_path_policy_transition(
    before: object,
    after: object,
    *,
    authorized_uid: int,
    allow_generator_late_inode_rotation: bool = False,
) -> None:
    """Require an exact policy, optionally allowing one safe late-generator swap."""

    if _deep_exact_equal(before, after):
        return
    if allow_generator_late_inode_rotation is not True:
        raise PermissionError("user unit search path policy changed")
    if (
        type(before) is not dict
        or type(after) is not dict
        or isinstance(authorized_uid, bool)
        or not isinstance(authorized_uid, int)
        or authorized_uid < 0
    ):
        raise PermissionError("user unit search path policy transition is malformed")
    before_paths = before.get("ordered_unit_paths")
    after_paths = after.get("ordered_unit_paths")
    runtime_directory = before.get("runtime_directory")
    if (
        type(before_paths) is not list
        or type(after_paths) is not list
        or len(before_paths) != len(after_paths)
        or not isinstance(runtime_directory, str)
        or not Path(runtime_directory).is_absolute()
    ):
        raise PermissionError("user unit search path policy transition is malformed")
    expected_path = str(Path(runtime_directory).parent / "generator.late")
    before_matches = [
        index for index, row in enumerate(before_paths)
        if type(row) is dict and row.get("path") == expected_path
    ]
    after_matches = [
        index for index, row in enumerate(after_paths)
        if type(row) is dict and row.get("path") == expected_path
    ]
    changed_indexes = [
        index for index, (old_row, new_row) in enumerate(
            zip(before_paths, after_paths)
        )
        if not _deep_exact_equal(old_row, new_row)
    ]
    if (
        len(before_matches) != 1
        or before_matches != after_matches
        or changed_indexes != before_matches
    ):
        raise PermissionError(
            "only one same-index generator.late inode rotation is allowed"
        )
    index = changed_indexes[0]
    old_row = before_paths[index]
    new_row = after_paths[index]
    exact_keys = {"path", "exists", "device", "inode", "owner_uid", "mode"}
    if (
        type(old_row) is not dict
        or type(new_row) is not dict
        or set(old_row) != exact_keys
        or set(new_row) != exact_keys
        or old_row["path"] != expected_path
        or new_row["path"] != expected_path
        or old_row["exists"] is not True
        or new_row["exists"] is not True
    ):
        raise PermissionError("generator.late policy row schema is not exact")
    for row in (old_row, new_row):
        device = row["device"]
        inode = row["inode"]
        owner_uid = row["owner_uid"]
        mode = row["mode"]
        if (
            isinstance(device, bool)
            or not isinstance(device, int)
            or device < 0
            or isinstance(inode, bool)
            or not isinstance(inode, int)
            or inode <= 0
            or isinstance(owner_uid, bool)
            or not isinstance(owner_uid, int)
            or owner_uid != authorized_uid
            or isinstance(mode, bool)
            or not isinstance(mode, int)
            or not 0 <= mode <= 0o7777
            or mode & 0o022
        ):
            raise PermissionError("generator.late policy row identity is unsafe")
    if (
        old_row["device"] != new_row["device"]
        or old_row["owner_uid"] != new_row["owner_uid"]
        or old_row["mode"] != new_row["mode"]
        or old_row["inode"] == new_row["inode"]
    ):
        raise PermissionError("generator.late rotation changed more than its inode")
    normalized_after = deepcopy(after)
    normalized_after["ordered_unit_paths"][index]["inode"] = old_row["inode"]
    if not _deep_exact_equal(before, normalized_after):
        raise PermissionError(
            "generator.late rotation accompanied another policy change"
        )


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
) -> tuple[bytes, os.stat_result]:
    """Read one canonical regular-file generation through one descriptor."""

    target = Path(path).absolute()
    parent = target.parent
    if parent.resolve(strict=True) != parent:
        raise PermissionError("bound file parent is not canonical")
    parent_before = parent.lstat()
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
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
            raise PermissionError("bound file parent changed before read")
        before = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError("bound file is not a regular file")
        flags = os.O_RDONLY | os.O_CLOEXEC
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
            raise PermissionError("bound file changed before descriptor read")
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
            raise PermissionError("bound file changed during descriptor read")
        return raw, finished
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def file_sha256(path: str | Path) -> str:
    raw, _identity = _read_stable_regular_file(path)
    return hashlib.sha256(raw).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_supervisor_v2_identity(scenario_id: str) -> dict[str, str]:
    if not isinstance(scenario_id, str) or _SCENARIO.fullmatch(scenario_id) is None:
        raise ValueError("integration scenario_id is not unique and canonical")
    identity = {
        "candidate": "systemd-integration-dummy",
        "stage_id": f"systemd_integration_dummy_{scenario_id}",
        "attempt_id": f"systemd_integration_dummy_attempt_{scenario_id}",
        "unit_name": f"{realizer.INTEGRATION_UNIT_PREFIX}{scenario_id}.service",
    }
    realizer.validate_integration_unit_name(identity["unit_name"])
    return identity


def _private_directory(path: Path, *, create: bool = False) -> dict[str, object]:
    target = path.absolute()
    if create:
        target.mkdir(mode=0o700)
    current = target.lstat()
    if (
        target.is_symlink() or not stat.S_ISDIR(current.st_mode)
        or target.resolve(strict=True) != target or current.st_uid != os.getuid()
        or stat.S_IMODE(current.st_mode) != 0o700
    ):
        raise PermissionError("integration directory must be canonical owned mode 0700")
    return {
        "path": str(target), "device": current.st_dev, "inode": current.st_ino,
        "owner_uid": current.st_uid, "mode": stat.S_IMODE(current.st_mode),
    }


def _validate_private_directory(binding: Mapping[str, object]) -> None:
    if _private_directory(Path(str(binding["path"]))) != dict(binding):
        raise PermissionError("bound private directory changed")


def _file_binding(path: Path, *, allow_symlink: bool = False) -> dict[str, object]:
    supplied = path.absolute()
    if any(character.isspace() for character in str(supplied)):
        raise ValueError("integration paths must be whitespace-free")
    linked = supplied.lstat()
    supplied_is_symlink = stat.S_ISLNK(linked.st_mode)
    link_text = os.readlink(supplied) if supplied_is_symlink else None
    resolved = supplied.resolve(strict=True)
    if (
        (supplied_is_symlink and not allow_symlink)
        or (not supplied_is_symlink and resolved != supplied)
    ):
        raise PermissionError("bound source is not a safe regular file")
    raw, current = _read_stable_regular_file(
        resolved,
        expected_nlink=1,
    )
    linked_after = supplied.lstat()
    if (
        (linked.st_dev, linked.st_ino, linked.st_uid, linked.st_mode,
         linked.st_nlink, linked.st_size, linked.st_mtime_ns, linked.st_ctime_ns)
        != (
            linked_after.st_dev,
            linked_after.st_ino,
            linked_after.st_uid,
            linked_after.st_mode,
            linked_after.st_nlink,
            linked_after.st_size,
            linked_after.st_mtime_ns,
            linked_after.st_ctime_ns,
        )
        or (supplied_is_symlink and os.readlink(supplied) != link_text)
        or supplied.resolve(strict=True) != resolved
    ):
        raise PermissionError("bound source changed during identity capture")
    return {
        "path": str(supplied), "resolved_path": str(resolved),
        "path_is_symlink": supplied_is_symlink,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "device": current.st_dev, "inode": current.st_ino,
        "owner_uid": current.st_uid, "mode": stat.S_IMODE(current.st_mode),
    }


def _validate_file_binding(binding: Mapping[str, object]) -> None:
    observed = _file_binding(
        Path(str(binding["path"])), allow_symlink=bool(binding["path_is_symlink"])
    )
    if observed != dict(binding):
        raise PermissionError("bound file identity changed")


def _read_bound_utf8(binding: Mapping[str, object]) -> str:
    raw, current = _read_stable_regular_file(
        Path(str(binding["resolved_path"])),
        expected_nlink=1,
    )
    if (
        hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
        or current.st_dev != binding["device"]
        or current.st_ino != binding["inode"]
        or current.st_uid != binding["owner_uid"]
        or stat.S_IMODE(current.st_mode) != binding["mode"]
    ):
        raise PermissionError("bound UTF-8 source changed before use")
    return raw.decode("utf-8")


def _write_sealed(
    path: Path, body: Mapping[str, object], *, fingerprint_field: str,
) -> dict[str, object]:
    target = path.absolute()
    parent_before = _private_directory(target.parent)
    if fingerprint_field in body:
        raise ValueError("body already contains its fingerprint")
    materialized = dict(body)
    payload = {**materialized, fingerprint_field: stable_fingerprint(materialized)}
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    parent_fd = os.open(
        target.parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        parent_opened = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_opened.st_mode)
            or parent_opened.st_dev != parent_before["device"]
            or parent_opened.st_ino != parent_before["inode"]
            or parent_opened.st_uid != parent_before["owner_uid"]
            or stat.S_IMODE(parent_opened.st_mode) != parent_before["mode"]
        ):
            raise PermissionError(
                "sealed integration evidence parent changed before write"
            )
        descriptor = os.open(
            target.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o444, dir_fd=parent_fd,
        )
        os.fchmod(descriptor, 0o444)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("zero-byte integration evidence write")
            offset += written
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        linked = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            opened.st_dev != linked.st_dev or opened.st_ino != linked.st_ino
            or opened.st_uid != os.getuid() or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o444
        ):
            raise PermissionError("sealed integration evidence identity is unsafe")
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
            or parent_linked.st_uid != parent_opened.st_uid
            or stat.S_IMODE(parent_linked.st_mode)
            != stat.S_IMODE(parent_opened.st_mode)
        ):
            raise RuntimeError("sealed integration evidence readback changed")
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    return payload


def _read_sealed(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str | None = None,
    expected_mode: int = 0o444,
) -> dict[str, object]:
    raw, _identity = _read_stable_regular_file(
        path,
        expected_uid=os.getuid(),
        expected_mode=expected_mode,
        expected_nlink=1,
    )
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or raw != (canonical_json(payload) + "\n").encode():
        raise ValueError("integration evidence is not canonical JSON")
    body = dict(payload)
    fingerprint = body.pop(fingerprint_field, None)
    if (
        not isinstance(fingerprint, str) or _SHA.fullmatch(fingerprint) is None
        or fingerprint != stable_fingerprint(body)
        or (schema is not None and payload.get("schema_version") != schema)
    ):
        raise PermissionError("integration evidence fingerprint or schema is invalid")
    return payload


def _fixed_environment() -> dict[str, str]:
    uid = os.getuid()
    runtime = f"/run/user/{uid}"
    return {
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime}/bus",
        "HOME": pwd.getpwuid(uid).pw_dir, "LANG": "C", "LC_ALL": "C",
        "PATH": "/usr/bin:/bin", "SYSTEMD_COLORS": "0",
        "XDG_RUNTIME_DIR": runtime,
    }


def run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), shell=False, check=False, capture_output=True, text=True,
        timeout=30.0, env=_fixed_environment(),
    )


def collect_manager_generation() -> dict[str, object]:
    uid = os.getuid()
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip().lower()
    if _BOOT_ID.fullmatch(boot_id) is None:
        raise RuntimeError("boot ID is malformed")
    runtime = Path(f"/run/user/{uid}")
    bus = runtime / "bus"
    runtime_stat = runtime.lstat()
    bus_stat = bus.lstat()
    if (
        runtime.is_symlink() or not stat.S_ISDIR(runtime_stat.st_mode)
        or runtime_stat.st_uid != uid or stat.S_IMODE(runtime_stat.st_mode) != 0o700
        or bus.is_symlink() or not stat.S_ISSOCK(bus_stat.st_mode)
        or bus_stat.st_uid != uid
    ):
        raise PermissionError("user manager endpoint is unsafe")
    cgroup = f"/user.slice/user-{uid}.slice/user@{uid}.service/init.scope"
    members = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "cgroup.procs"
    candidates: list[dict[str, object]] = []
    for row in members.read_text(encoding="ascii").splitlines():
        pid = int(row)
        root = Path(f"/proc/{pid}")
        stat_row = (root / "stat").read_text(encoding="ascii")
        closing = stat_row.rfind(") ")
        fields = stat_row[closing + 2:].split() if closing >= 0 else []
        status = (root / "status").read_text(encoding="ascii")
        uid_rows = [line for line in status.splitlines() if line.startswith("Uid:")]
        cgroup_rows = (root / "cgroup").read_text(encoding="utf-8").splitlines()
        unified = [line.split("::", 1)[1] for line in cgroup_rows if "::" in line]
        argv = [
            item.decode("utf-8", errors="strict")
            for item in (root / "cmdline").read_bytes().split(b"\0") if item
        ]
        if (
            len(fields) > 19 and len(uid_rows) == 1 and len(unified) == 1
            and argv and int(uid_rows[0].split()[1]) == uid
            and unified[0] == cgroup and Path(argv[0]).name == "systemd"
            and "--user" in argv[1:]
        ):
            candidates.append({
                "pid": pid, "starttime_ticks": int(fields[19]), "uid": uid,
                "control_group": cgroup,
            })
    if len(candidates) != 1:
        raise RuntimeError("user manager process generation is ambiguous")
    return {
        "boot_id": boot_id, "identity": candidates[0],
        "endpoint": {
            "uid": uid, "runtime_directory": str(runtime),
            "runtime_device": runtime_stat.st_dev,
            "runtime_inode": runtime_stat.st_ino, "bus_path": str(bus),
            "bus_device": bus_stat.st_dev, "bus_inode": bus_stat.st_ino,
        },
    }


def _validate_manager_generation(value: Mapping[str, object]) -> None:
    uid = os.getuid()
    identity = value.get("identity")
    endpoint = value.get("endpoint")
    if (
        set(value) != {"boot_id", "identity", "endpoint"}
        or _BOOT_ID.fullmatch(str(value.get("boot_id"))) is None
        or not isinstance(identity, Mapping) or not isinstance(endpoint, Mapping)
        or set(identity) != {"pid", "starttime_ticks", "uid", "control_group"}
        or set(endpoint) != {
            "uid", "runtime_directory", "runtime_device", "runtime_inode",
            "bus_path", "bus_device", "bus_inode",
        }
        or isinstance(identity.get("uid"), bool)
        or not isinstance(identity.get("uid"), int)
        or identity.get("uid") != uid
        or identity.get("control_group")
        != f"/user.slice/user-{uid}.slice/user@{uid}.service/init.scope"
        or isinstance(endpoint.get("uid"), bool)
        or not isinstance(endpoint.get("uid"), int)
        or endpoint.get("uid") != uid
        or endpoint.get("runtime_directory") != f"/run/user/{uid}"
        or endpoint.get("bus_path") != f"/run/user/{uid}/bus"
    ):
        raise PermissionError("manager generation is not exact")
    numeric = (
        identity.get("pid"), identity.get("starttime_ticks"),
        endpoint.get("runtime_device"), endpoint.get("runtime_inode"),
        endpoint.get("bus_device"), endpoint.get("bus_inode"),
    )
    for item in numeric:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise PermissionError("manager generation numeric identity is malformed")


def _directives(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in text.splitlines():
        if not row or row.startswith(("#", "[")):
            continue
        key, separator, value = row.partition("=")
        if separator != "=":
            raise ValueError("integration unit directive is malformed")
        result.setdefault(key, []).append(value)
    return result


def render_integration_fragment(
    *, template_text: str, scenario_id: str, scenario_root: Path,
    unit_name: str, python_path: Path, supervisor_path: Path,
    runtime_spec_path: Path,
) -> str:
    identity = build_supervisor_v2_identity(scenario_id)
    if identity["unit_name"] != unit_name or not template_text.endswith("\n"):
        raise PermissionError("integration template identity changed")
    replacements = {
        "@@SCENARIO_ID@@": scenario_id, "@@SCENARIO_ROOT@@": str(scenario_root),
        "@@UNIT_NAME@@": unit_name, "@@PYTHON_PATH@@": str(python_path),
        "@@SUPERVISOR_PATH@@": str(supervisor_path),
        "@@RUNTIME_SPEC_PATH@@": str(runtime_spec_path),
    }
    if any(not Path(value).is_absolute() for token, value in replacements.items()
           if token not in {"@@SCENARIO_ID@@", "@@UNIT_NAME@@"}):
        raise ValueError("rendered integration paths must be absolute")
    if any(character.isspace() for value in replacements.values()
           for character in value):
        raise ValueError("rendered integration values must be whitespace-free")
    if any(template_text.count(token) < 1 for token in _TOKENS):
        raise ValueError("integration template placeholder closure changed")
    rendered = template_text
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if "@@" in rendered or "[Install]" in rendered or "systemd-run" in rendered:
        raise PermissionError("rendered integration fragment is installable or transient")
    directives = _directives(rendered)
    for directive, mode in _EXEC_MODES.items():
        expected = (
            f"{python_path} -I -u {supervisor_path} {mode} "
            f"--spec {runtime_spec_path}"
        )
        if directives.get(directive) != [expected]:
            raise PermissionError(f"integration {directive} is not supervisor-v2 exact")
    exact = {
        "Description": [f"CURE-Lite v24 supervisor-v2 integration {scenario_id}"],
        "StartLimitIntervalSec": ["infinity"], "StartLimitBurst": ["1"],
        "Type": ["exec"], "ExitType": ["main"],
        "WorkingDirectory": [str(scenario_root)], "UMask": ["0077"],
        "Restart": ["no"], "RestartSec": ["0"], "KillMode": ["mixed"],
        "KillSignal": ["SIGTERM"], "SendSIGKILL": ["yes"],
        "TimeoutStartSec": ["30s"], "TimeoutStopSec": ["30s"],
        "RuntimeMaxSec": ["infinity"], "WatchdogSec": ["0"],
        "OOMPolicy": ["kill"], "RemainAfterExit": ["no"],
        "StandardInput": ["null"], "StandardOutput": ["journal"],
        "StandardError": ["journal"], "SyslogIdentifier": [unit_name],
        "SuccessExitStatus": ["0"],
    }
    if any(directives.get(key) != value for key, value in exact.items()):
        raise PermissionError("integration static directive closure changed")
    if set(directives) != set(exact) | set(_EXEC_MODES):
        raise PermissionError("integration template has an unauthorized directive")
    return rendered


def _exec_shadow(argv: Sequence[str]) -> str:
    return f"{{ path={argv[0]} ; argv[]={' '.join(argv)} ; ignore_errors=no }}"


def _normalize_queried_exec_shadow(value: str) -> str:
    """Fully parse one systemd Exec value before discarding runtime fields."""

    if (
        not isinstance(value, str)
        or not value.startswith("{")
        or not value.endswith("}")
    ):
        raise ValueError("immutable systemd Exec identity is ambiguous")
    rows = [row.strip() for row in value[1:-1].strip().split(";")]
    if not rows or any(not row for row in rows):
        raise ValueError("immutable systemd Exec identity is ambiguous")
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
            raise ValueError("immutable systemd Exec identity is ambiguous")
        fields[key] = raw.strip()
    if set(fields) - allowed_runtime != allowed_static:
        raise ValueError("immutable systemd Exec identity is ambiguous")
    for name in ("start_time", "stop_time"):
        if name in fields and re.fullmatch(r"\[[^\]\r\n]*\]", fields[name]) is None:
            raise ValueError("immutable systemd Exec runtime fields are malformed")
    if "pid" in fields and re.fullmatch(r"[0-9]+", fields["pid"]) is None:
        raise ValueError("immutable systemd Exec runtime fields are malformed")
    if "code" in fields and re.fullmatch(
        r"(?:\(null\)|[A-Za-z0-9_-]+)",
        fields["code"],
    ) is None:
        raise ValueError("immutable systemd Exec runtime fields are malformed")
    if "status" in fields and re.fullmatch(
        r"[0-9]+(?:/[A-Za-z0-9_-]+)?",
        fields["status"],
    ) is None:
        raise ValueError("immutable systemd Exec runtime fields are malformed")
    argv = fields["argv[]"].split()
    if (
        not argv
        or fields["path"] != argv[0]
        or fields["ignore_errors"] not in {"yes", "no"}
    ):
        raise ValueError("immutable systemd Exec static identity is malformed")
    return _exec_shadow(argv).replace(
        "ignore_errors=no",
        f"ignore_errors={fields['ignore_errors']}",
    )


def _normalize_queried_shadow_value(name: str, value: str) -> str:
    if name in _EXEC_MODES:
        return _normalize_queried_exec_shadow(value)
    if name == "WatchdogUSec":
        return (
            "disabled"
            if value in {"0", "infinity", "disabled"}
            else value
        )
    return value


def _immutable_shadow(
    *, fragment_path: Path, scenario_root: Path, python_path: Path,
    supervisor_path: Path, runtime_spec_path: Path,
) -> dict[str, str]:
    result = {
        "Type": "exec", "Restart": "no", "KillMode": "mixed",
        "SendSIGKILL": "yes", "TimeoutStopUSec": "30s",
        "FragmentPath": str(fragment_path), "DropInPaths": "",
        "Transient": "no", "Environment": "", "UnsetEnvironment": "",
        "WorkingDirectory": str(scenario_root), "UMask": "0077",
        "ExitType": "main", "RuntimeMaxUSec": "infinity",
        "WatchdogUSec": "disabled",
        "OOMPolicy": "kill", "RemainAfterExit": "no", "StandardInput": "null",
        "StandardOutput": "journal", "StandardError": "journal",
        "StartLimitIntervalUSec": "infinity", "StartLimitBurst": "1",
        "KillSignal": "15",
    }
    for directive, mode in _EXEC_MODES.items():
        argv = [
            str(python_path), "-I", "-u", str(supervisor_path), mode,
            "--spec", str(runtime_spec_path),
        ]
        result[directive] = _exec_shadow(argv)
    return result


def _build_runtime_spec_body(
    *, identity: Mapping[str, str], scenario_root: Path, runtime_root: Path,
    fragment_path: Path, fragment_sha: str, python_path: Path,
    supervisor_path: Path, supervisor_sha256: str,
    dummy_child_path: Path, dummy_child_sha256: str,
    runtime_spec_path: Path,
) -> dict[str, object]:
    artifacts = {
        key: str(runtime_root / name) for key, name in _SPEC_ARTIFACT_NAMES.items()
    }
    artifacts["root"] = str(runtime_root)
    child_argv = [
        str(python_path), "-I", str(dummy_child_path), "--artifact",
        str(runtime_root / "dummy-child.json"), "--scenario-id",
        identity["stage_id"][len("systemd_integration_dummy_"):],
        "--wait-seconds", "0.05",
    ]
    shadow = _immutable_shadow(
        fragment_path=fragment_path, scenario_root=scenario_root,
        python_path=python_path, supervisor_path=supervisor_path,
        runtime_spec_path=runtime_spec_path,
    )
    return {
        "schema_version": RUNTIME_SPEC_SCHEMA, "execution_kind": EXECUTION_KIND,
        "candidate": identity["candidate"], "stage_id": identity["stage_id"],
        "attempt_id": identity["attempt_id"], "attempt_ordinal": 0,
        "prior_attempt_count": 0, "authorization": None,
        "scientific_preaccess": None,
        "child": {
            "argv": child_argv, "argv_fingerprint": stable_fingerprint(child_argv),
            "cwd": str(scenario_root), "environment": {},
            "inherit_environment": [], "entrypoint_path": str(dummy_child_path),
        },
        "artifacts": artifacts,
        "runtime": {
            "shell": False, "start_new_session": True, "launch_limit": 1,
            "automatic_retry_allowed": False, "resume_allowed": False,
            "restart": "no", "heartbeat_interval_seconds": 0.05,
            "poll_interval_seconds": 0.01, "termination_grace_seconds": 0.5,
            "systemd": {
                "unit_name": identity["unit_name"], "service_type": "exec",
                "kill_mode": "mixed", "send_sigkill": True,
                "timeout_stop_seconds": 30.0, "start_ack_timeout_seconds": 5.0,
                "start_ack_poll_seconds": 0.02,
                "unit_fragment_file_sha256": fragment_sha,
                "immutable_shadow_properties": shadow,
                "immutable_shadow_fingerprint": stable_fingerprint(shadow),
            },
        },
        "environment": None,
        "source_bindings": {
            "supervisor_file_sha256": supervisor_sha256,
            "child_entry_file_sha256": dummy_child_sha256,
            "prior_attempt_receipt_file_sha256": None,
            "runtime_environment_file_sha256": None,
            "r2_adapter_path": None, "r2_adapter_file_sha256": None,
            "legacy_gate_entrypoint_path": None,
            "legacy_gate_entrypoint_file_sha256": None,
            "python_path": None, "python_file_sha256": None,
            "python_device": None, "python_inode": None,
            "python_owner_uid": None, "python_mode": None,
            "runtime_dependency_site_path": None,
            "runtime_dependency_site_device": None,
            "runtime_dependency_site_inode": None,
            "runtime_dependency_site_owner_uid": None,
            "runtime_dependency_site_mode": None,
        },
    }


def create_production_authorization(
    scenario_root: Path, *, scenario_id: str, template_path: Path,
    python_path: Path, supervisor_path: Path, realizer_path: Path,
    dummy_child_path: Path, instruction_id: str, authorization_basis: str,
    validity_seconds: int = 300, runner: CommandRunner = run_command,
    manager_reader: ManagerReader = collect_manager_generation,
) -> dict[str, object]:
    if instruction_id != INSTRUCTION_ID or authorization_basis != AUTHORIZATION_BASIS:
        raise PermissionError("modify-then-run instruction is not exact")
    if (
        isinstance(validity_seconds, bool) or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300 or not scenario_root.is_absolute()
    ):
        raise ValueError("integration authorization validity or root is invalid")
    identity = build_supervisor_v2_identity(scenario_id)
    scenario_binding = _private_directory(scenario_root, create=True)
    control_root = scenario_root / "control"
    runtime_root = scenario_root / "runtime"
    control_binding = _private_directory(control_root, create=True)
    runtime_binding = _private_directory(runtime_root, create=True)
    _private_directory(runtime_root / "heartbeat", create=True)
    _private_directory(runtime_root / "systemd-invocations", create=True)
    policy = realizer.freeze_user_unit_path_policy(
        identity["unit_name"], runner=runner
    )
    unit_directory = Path(str(policy["runtime_directory"]))
    fragment_path = unit_directory / identity["unit_name"]
    spec_path = control_root / "runtime-spec.json"
    authorization_path = control_root / "authorization.json"
    template_binding = _file_binding(template_path)
    executable_bindings = {
        "python": _file_binding(python_path),
        "supervisor": _file_binding(supervisor_path),
        "integration_tool": _file_binding(Path(__file__).resolve()),
        "realizer": _file_binding(realizer_path),
        "dummy_child": _file_binding(dummy_child_path),
        "systemd_path": _file_binding(Path(SYSTEMD_PATH), allow_symlink=True),
        "systemd_analyze": _file_binding(Path(SYSTEMD_ANALYZE), allow_symlink=True),
        "systemctl": _file_binding(Path(SYSTEMCTL), allow_symlink=True),
    }
    rendered = render_integration_fragment(
        template_text=_read_bound_utf8(template_binding),
        scenario_id=scenario_id, scenario_root=scenario_root,
        unit_name=identity["unit_name"], python_path=python_path,
        supervisor_path=supervisor_path, runtime_spec_path=spec_path,
    )
    rendered_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    spec_body = _build_runtime_spec_body(
        identity=identity, scenario_root=scenario_root, runtime_root=runtime_root,
        fragment_path=fragment_path, fragment_sha=rendered_sha,
        python_path=python_path, supervisor_path=supervisor_path,
        supervisor_sha256=str(
            executable_bindings["supervisor"]["file_sha256"]
        ),
        dummy_child_path=dummy_child_path,
        dummy_child_sha256=str(
            executable_bindings["dummy_child"]["file_sha256"]
        ),
        runtime_spec_path=spec_path,
    )
    for binding in executable_bindings.values():
        _validate_file_binding(binding)
    _validate_file_binding(template_binding)
    spec = _write_sealed(
        spec_path, spec_body, fingerprint_field="runtime_spec_fingerprint"
    )
    manager = manager_reader()
    _validate_manager_generation(manager)
    issued = datetime.now(timezone.utc)
    control_artifacts = {
        "integration_terminal": str(control_root / "integration-terminal.json"),
        "removal_authorization": str(control_root / "removal-authorization.json"),
        "removal_state": str(control_root / "removal-state.json"),
        "integration_receipt": str(control_root / "integration-receipt.json"),
        "dummy_artifact": str(runtime_root / "dummy-child.json"),
    }
    body = {
        "schema_version": AUTHORIZATION_SCHEMA, "instruction_id": instruction_id,
        "authorization_basis": authorization_basis,
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (
            issued + timedelta(seconds=validity_seconds)
        ).isoformat().replace("+00:00", "Z"),
        "authorized_uid": os.getuid(), "scenario_id": scenario_id,
        "identity": dict(identity), "scenario_root": scenario_binding,
        "control_root": control_binding, "runtime_root": runtime_binding,
        "unit_directory": str(unit_directory), "unit_path_policy": policy,
        "manager_generation": manager, "template_binding": template_binding,
        "rendered_fragment": {"utf8_text": rendered, "sha256": rendered_sha},
        "runtime_spec_binding": {
            "path": str(spec_path),
            "file_sha256": sealed_file_sha256(spec),
            "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
            "required_schema": RUNTIME_SPEC_SCHEMA,
        },
        "executable_bindings": executable_bindings,
        "control_artifacts": control_artifacts,
        "integration_authorized": True, "actual_r2_authorized": False,
        "unit_realization_authorized": True, "unit_removal_authorized": False,
        "enable_authorized": False, "direct_start_authorized": False,
        "payload_authority": "none", "D_R_payload_accessed": False,
        "D_V_payload_accessed": False, "D_T_payload_accessed": False,
        "gpu_access_authorized": False,
    }
    return _write_sealed(
        authorization_path, body, fingerprint_field="authorization_fingerprint"
    )


def load_integration_authorization(
    path: Path, *, runner: CommandRunner = run_command,
    manager_reader: ManagerReader = collect_manager_generation,
    allow_installed_fragment: bool = False,
) -> dict[str, object]:
    authorization = _read_sealed(
        path, fingerprint_field="authorization_fingerprint",
        schema=AUTHORIZATION_SCHEMA,
    )
    if set(authorization) != _AUTH_KEYS:
        raise PermissionError("integration authorization keys are not exact")
    if (
        authorization["instruction_id"] != INSTRUCTION_ID
        or authorization["authorization_basis"] != AUTHORIZATION_BASIS
        or authorization["authorized_uid"] != os.getuid()
        or authorization["integration_authorized"] is not True
        or authorization["actual_r2_authorized"] is not False
        or authorization["unit_realization_authorized"] is not True
        or authorization["unit_removal_authorized"] is not False
        or authorization["enable_authorized"] is not False
        or authorization["direct_start_authorized"] is not False
        or authorization["payload_authority"] != "none"
        or any(authorization[key] is not False for key in (
            "D_R_payload_accessed", "D_V_payload_accessed",
            "D_T_payload_accessed", "gpu_access_authorized"
        ))
    ):
        raise PermissionError("integration authorization is not payload-free exact")
    issued = datetime.fromisoformat(str(authorization["issued_at_utc"]).replace("Z", "+00:00"))
    expires = datetime.fromisoformat(str(authorization["expires_at_utc"]).replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if issued.tzinfo is None or expires.tzinfo is None or not issued <= now <= expires or expires - issued > timedelta(seconds=300):
        raise PermissionError("integration authorization is stale")
    identity = build_supervisor_v2_identity(str(authorization["scenario_id"]))
    if authorization["identity"] != identity:
        raise PermissionError("integration scenario identity changed")
    for key in ("scenario_root", "control_root", "runtime_root"):
        _validate_private_directory(authorization[key])
    executable_bindings = authorization["executable_bindings"]
    expected_executables = {
        "python", "supervisor", "integration_tool", "realizer", "dummy_child",
        "systemd_path", "systemd_analyze", "systemctl",
    }
    if not isinstance(executable_bindings, Mapping) or set(executable_bindings) != expected_executables:
        raise PermissionError("integration executable bindings are not exact")
    for binding in executable_bindings.values():
        _validate_file_binding(binding)
    _validate_file_binding(authorization["template_binding"])
    spec_binding = authorization["runtime_spec_binding"]
    if not isinstance(spec_binding, Mapping) or set(spec_binding) != {
        "path", "file_sha256", "runtime_spec_fingerprint", "required_schema"
    }:
        raise PermissionError("runtime spec binding is not exact")
    spec = _read_sealed(
        Path(str(spec_binding["path"])), fingerprint_field="runtime_spec_fingerprint",
        schema=RUNTIME_SPEC_SCHEMA,
    )
    if (
        spec_binding["required_schema"] != RUNTIME_SPEC_SCHEMA
        or sealed_file_sha256(spec) != spec_binding["file_sha256"]
        or spec["runtime_spec_fingerprint"] != spec_binding["runtime_spec_fingerprint"]
        or spec.get("environment") is not None
        or spec.get("execution_kind") != EXECUTION_KIND
        or spec.get("source_bindings", {}).get(
            "supervisor_file_sha256"
        ) != executable_bindings["supervisor"]["file_sha256"]
        or spec.get("source_bindings", {}).get(
            "child_entry_file_sha256"
        ) != executable_bindings["dummy_child"]["file_sha256"]
    ):
        raise PermissionError("runtime spec binding changed or carries environment")
    rendered = render_integration_fragment(
        template_text=_read_bound_utf8(authorization["template_binding"]),
        scenario_id=str(authorization["scenario_id"]),
        scenario_root=Path(str(authorization["scenario_root"]["path"])),
        unit_name=identity["unit_name"],
        python_path=Path(str(executable_bindings["python"]["path"])),
        supervisor_path=Path(str(executable_bindings["supervisor"]["path"])),
        runtime_spec_path=Path(str(spec_binding["path"])),
    )
    if authorization["rendered_fragment"] != {
        "utf8_text": rendered,
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }:
        raise PermissionError("rendered integration fragment binding changed")
    manager = manager_reader()
    _validate_manager_generation(manager)
    if manager != authorization["manager_generation"]:
        raise PermissionError("user manager generation changed")
    fragment = Path(str(authorization["unit_directory"])) / identity["unit_name"]
    observed_policy = realizer.freeze_user_unit_path_policy(
        identity["unit_name"], runner=runner,
        allowed_fragment=fragment if allow_installed_fragment else None,
    )
    _validate_unit_path_policy_transition(
        authorization["unit_path_policy"],
        observed_policy,
        authorized_uid=authorization["authorized_uid"],
    )
    return authorization


def _query_immutable_shadow(unit_name: str, *, runner: CommandRunner) -> dict[str, str]:
    argv = (
        SYSTEMCTL, "--user", "show", unit_name, "--no-pager",
        *[f"--property={name}" for name in sorted(_IMMUTABLE_SHADOW_KEYS)],
    )
    completed = runner(argv)
    if completed.returncode != 0:
        raise RuntimeError("immutable systemd shadow query failed")
    result: dict[str, str] = {}
    for row in completed.stdout.splitlines():
        if not row or "=" not in row:
            continue
        key, value = row.split("=", 1)
        if key in result:
            raise ValueError("immutable shadow has duplicate properties")
        result[key] = _normalize_queried_shadow_value(key, value)
    if set(result) != _IMMUTABLE_SHADOW_KEYS:
        raise ValueError("immutable systemd shadow is incomplete")
    return result


def _validate_systemd_outcome(outcome: object) -> Mapping[str, object]:
    if not isinstance(outcome, Mapping) or set(outcome) != _SYSTEMD_OUTCOME_KEYS:
        raise PermissionError("systemd sidecar outcome is malformed")
    service_result = outcome.get("service_result")
    exit_code = outcome.get("exit_code")
    exit_status = outcome.get("exit_status")
    invocation_id = outcome.get("invocation_id")
    if (
        not isinstance(service_result, str)
        or not service_result
        or "\x00" in service_result
        or not isinstance(exit_code, str)
        or not exit_code
        or "\x00" in exit_code
        or not isinstance(exit_status, str)
        or not exit_status
        or "\x00" in exit_status
        or not isinstance(outcome.get("category"), str)
        or not isinstance(outcome.get("systemd_success"), bool)
        or outcome.get("scientific_gate_passed") is not None
        or not isinstance(invocation_id, str)
        or _INVOCATION.fullmatch(invocation_id) is None
    ):
        raise PermissionError("systemd sidecar outcome fields are malformed")
    expected_success = (
        service_result == "success"
        and exit_code == "exited"
        and exit_status == "0"
    )
    expected_category = (
        "SYSTEMD_SERVICE_SUCCESS"
        if expected_success
        else _SYSTEMD_OUTCOME_CATEGORIES.get(
            service_result,
            "SYSTEMD_OTHER_FAILURE",
        )
    )
    if (
        outcome.get("systemd_success") is not expected_success
        or outcome.get("category") != expected_category
    ):
        raise PermissionError("systemd sidecar outcome is internally inconsistent")
    return outcome


def _validate_systemd_sidecar_causality(
    sidecar: Mapping[str, object],
) -> None:
    outcome = _validate_systemd_outcome(sidecar.get("systemd_outcome"))
    claim_valid = sidecar.get("claim_valid")
    claim_matches = sidecar.get("claim_matches_invocation")
    start_ack_valid = sidecar.get("start_ack_valid")
    child_prespawn_valid = sidecar.get("child_prespawn_valid")
    if (
        (start_ack_valid is True and claim_valid is not True)
        or (
            child_prespawn_valid is True
            and start_ack_valid is not True
        )
        or (claim_matches is True and claim_valid is not True)
    ):
        raise PermissionError("systemd sidecar causal lineage is impossible")
    if sidecar.get("audit_valid") is True and (
        claim_valid is not True
        or claim_matches is not True
        or start_ack_valid is not True
        or child_prespawn_valid is not True
    ):
        raise PermissionError(
            "valid systemd sidecar audit lacks its complete causal lineage"
        )
    if (
        outcome.get("category") == "SYSTEMD_EXEC_CONDITION"
        or outcome.get("service_result") == "exec-condition"
    ) and (
        outcome.get("category") != "SYSTEMD_EXEC_CONDITION"
        or outcome.get("service_result") != "exec-condition"
        or outcome.get("systemd_success") is not False
        or sidecar.get("audit_valid") is not False
        or not isinstance(claim_valid, bool)
        or claim_matches is not False
        or start_ack_valid is not False
        or child_prespawn_valid is not False
    ):
        raise PermissionError(
            "systemd ExecCondition failure lineage is impossible"
        )


def _load_single_systemd_sidecar(
    authorization: Mapping[str, object],
) -> tuple[dict[str, object], Path] | None:
    """Load one invocation sidecar without requiring a runtime terminal."""

    spec = _read_sealed(
        Path(str(authorization["runtime_spec_binding"]["path"])),
        fingerprint_field="runtime_spec_fingerprint",
        schema=RUNTIME_SPEC_SCHEMA,
    )
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise PermissionError("runtime artifacts contract is malformed")
    invocation_dir = Path(str(artifacts["systemd_invocation_dir"]))
    current = invocation_dir.lstat()
    if (
        invocation_dir.is_symlink()
        or not stat.S_ISDIR(current.st_mode)
        or invocation_dir.resolve(strict=True) != invocation_dir
        or current.st_uid != os.getuid()
        or stat.S_IMODE(current.st_mode) != 0o700
    ):
        raise PermissionError("systemd invocation directory is not exact")
    rows = list(invocation_dir.iterdir())
    if not rows:
        return None
    if len(rows) != 1:
        raise PermissionError("systemd terminal evidence is ambiguous")
    sidecar_path = rows[0]
    invocation_id = sidecar_path.stem
    if (
        sidecar_path.name != f"{invocation_id}.json"
        or _INVOCATION.fullmatch(invocation_id) is None
    ):
        raise PermissionError("systemd terminal filename is malformed")
    sidecar = _read_sealed(
        sidecar_path,
        fingerprint_field="systemd_terminal_fingerprint",
        schema=SYSTEMD_TERMINAL_SCHEMA,
        expected_mode=0o444,
    )
    common = {
        "candidate": spec["candidate"],
        "stage_id": spec["stage_id"],
        "attempt_id": spec["attempt_id"],
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
    }
    outcome = sidecar.get("systemd_outcome")
    if (
        any(sidecar.get(name) != value for name, value in common.items())
        or sidecar.get("sidecar_systemd_invocation_id") != invocation_id
        or not isinstance(outcome, Mapping)
        or outcome.get("invocation_id") != invocation_id
        or not isinstance(outcome.get("systemd_success"), bool)
        or not isinstance(sidecar.get("audit_valid"), bool)
        or sidecar.get("attempt_commit_required") is not True
        or sidecar.get("attempt_commit_valid") is not True
        or sidecar.get("current_authorization_valid") is not None
        or sidecar.get("current_runtime_closure_valid") is not True
        or sidecar.get("current_runtime_closure_error_type") is not None
        or sidecar.get("authorization_matches_commit") is not True
        or sidecar.get("scientific_gate_passed") is not None
    ):
        raise PermissionError("systemd sidecar lineage is not exact")

    attempt_path = Path(str(artifacts["attempt_commit"]))
    attempt = _read_sealed(
        attempt_path,
        fingerprint_field="attempt_commit_fingerprint",
        schema=ATTEMPT_COMMIT_SCHEMA,
    )
    if (
        any(attempt.get(name) != value for name, value in common.items())
        or sidecar.get("attempt_commit_fingerprint")
        != attempt.get("attempt_commit_fingerprint")
        or sidecar.get("attempt_commit_file_sha256")
        != sealed_file_sha256(attempt)
    ):
        raise PermissionError("systemd sidecar attempt lineage changed")

    def _verify_optional_phase(
        *,
        artifact_key: str,
        valid_key: str,
        fingerprint_key: str,
        sha_key: str,
        expected_phase: str,
        expected_mode: int,
    ) -> None:
        path = Path(str(artifacts[artifact_key]))
        valid = sidecar.get(valid_key)
        if valid is False:
            if (
                os.path.lexists(path)
                or sidecar.get(fingerprint_key) is not None
                or sidecar.get(sha_key) is not None
            ):
                raise PermissionError(
                    f"systemd sidecar {expected_phase} absence changed"
                )
            return
        if valid is not True:
            raise PermissionError(
                f"systemd sidecar {expected_phase} validity is malformed"
            )
        receipt = _read_sealed(
            path,
            fingerprint_field="phase_receipt_fingerprint",
            schema=PHASE_RECEIPT_SCHEMA,
            expected_mode=expected_mode,
        )
        state = receipt.get("systemd_phase_state")
        if (
            any(receipt.get(name) != value for name, value in common.items())
            or receipt.get("phase") != expected_phase
            or not isinstance(state, Mapping)
            or state.get("InvocationID") != invocation_id
            or sidecar.get(fingerprint_key)
            != receipt.get("phase_receipt_fingerprint")
            or sidecar.get(sha_key) != sealed_file_sha256(receipt)
        ):
            raise PermissionError(
                f"systemd sidecar {expected_phase} lineage changed"
            )

    _verify_optional_phase(
        artifact_key="start_ack_receipt",
        valid_key="start_ack_valid",
        fingerprint_key="start_ack_receipt_fingerprint",
        sha_key="start_ack_receipt_file_sha256",
        expected_phase="start_ack",
        expected_mode=0o444,
    )
    _verify_optional_phase(
        artifact_key="child_prespawn_phase_receipt",
        valid_key="child_prespawn_valid",
        fingerprint_key="child_prespawn_phase_receipt_fingerprint",
        sha_key="child_prespawn_phase_receipt_file_sha256",
        expected_phase="child_prespawn",
        expected_mode=0o444,
    )

    claim_path = Path(str(artifacts["materialization_claim"]))
    if sidecar.get("claim_valid") is False:
        if (
            os.path.lexists(claim_path)
            or sidecar.get("materialization_claim_fingerprint") is not None
            or sidecar.get("materialization_claim_file_sha256") is not None
            or sidecar.get("claim_systemd_invocation_id") is not None
            or sidecar.get("claim_matches_invocation") is not False
        ):
            raise PermissionError("systemd sidecar claim absence changed")
    elif sidecar.get("claim_valid") is True:
        claim = _read_sealed(
            claim_path,
            fingerprint_field="materialization_claim_fingerprint",
            schema=MATERIALIZATION_CLAIM_SCHEMA,
            expected_mode=0o444,
        )
        if (
            any(claim.get(name) != value for name, value in common.items())
            or claim.get("systemd_invocation_id") != invocation_id
            or sidecar.get("claim_systemd_invocation_id") != invocation_id
            or sidecar.get("claim_matches_invocation") is not True
            or sidecar.get("materialization_claim_fingerprint")
            != claim.get("materialization_claim_fingerprint")
            or sidecar.get("materialization_claim_file_sha256")
            != sealed_file_sha256(claim)
        ):
            raise PermissionError("systemd sidecar claim lineage changed")
    else:
        raise PermissionError("systemd sidecar claim validity is malformed")
    _validate_systemd_sidecar_causality(sidecar)
    if any(
        sidecar.get(key) is not None
        for key in (
            "finalizer_environment_audit_fingerprint",
            "finalizer_environment_inventory_fingerprint",
            "finalizer_environment_audit_valid",
            "active_gpu_lease_fingerprint",
            "active_gpu_lease_file_sha256",
            "active_gpu_lease_device",
            "active_gpu_lease_inode",
            "active_gpu_lease_parent_device",
            "active_gpu_lease_parent_inode",
            "active_gpu_lease_valid",
            "gpu_lease_release_authorized",
            "gpu_lease_release_valid",
            "gpu_lease_release_receipt_fingerprint",
            "gpu_lease_tombstone_file_sha256",
        )
    ):
        raise PermissionError("systemd dummy sidecar contains GPU evidence")
    return sidecar, sidecar_path


def _sidecar_summary(
    sidecar: Mapping[str, object],
    path: Path,
) -> dict[str, object]:
    outcome = sidecar["systemd_outcome"]
    if not isinstance(outcome, Mapping):
        raise AssertionError("validated sidecar outcome changed")
    return {
        "evidence_kind": "partial-systemd-terminal",
        "invocation_id": sidecar["sidecar_systemd_invocation_id"],
        "systemd_terminal_fingerprint": sidecar[
            "systemd_terminal_fingerprint"
        ],
        "systemd_terminal_file_sha256": sealed_file_sha256(sidecar),
        "audit_valid": sidecar["audit_valid"],
        "claim_valid": sidecar["claim_valid"],
        "start_ack_valid": sidecar["start_ack_valid"],
        "child_prespawn_valid": sidecar["child_prespawn_valid"],
        "systemd_outcome": dict(outcome),
    }


def _sidecar_failure_error(
    sidecar: Mapping[str, object],
) -> RuntimeError:
    outcome = sidecar["systemd_outcome"]
    if not isinstance(outcome, Mapping):
        raise AssertionError("validated sidecar outcome changed")
    return RuntimeError(
        "systemd terminal rejected integration:"
        f"{outcome.get('category')}:"
        f"{sidecar.get('systemd_terminal_fingerprint')}"
    )


def _validate_supervisor_evidence(
    authorization: Mapping[str, object],
) -> dict[str, object]:
    spec = _read_sealed(
        Path(str(authorization["runtime_spec_binding"]["path"])),
        fingerprint_field="runtime_spec_fingerprint", schema=RUNTIME_SPEC_SCHEMA,
    )
    artifacts = spec["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise PermissionError("runtime artifacts contract is malformed")
    common = {
        "candidate": spec["candidate"], "stage_id": spec["stage_id"],
        "attempt_id": spec["attempt_id"],
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
    }
    definitions = {
        "attempt_commit": (
            "attempt_commit_fingerprint",
            ATTEMPT_COMMIT_SCHEMA,
            0o444,
        ),
        "launch_lease": (
            "launch_lease_fingerprint",
            LAUNCH_LEASE_SCHEMA,
            0o444,
        ),
        "materialization_claim": (
            "materialization_claim_fingerprint",
            MATERIALIZATION_CLAIM_SCHEMA,
            0o444,
        ),
        "precommit_phase_receipt": (
            "phase_receipt_fingerprint",
            PHASE_RECEIPT_SCHEMA,
            0o444,
        ),
        "start_ack_receipt": (
            "phase_receipt_fingerprint",
            PHASE_RECEIPT_SCHEMA,
            0o444,
        ),
        "child_prespawn_phase_receipt": (
            "phase_receipt_fingerprint",
            PHASE_RECEIPT_SCHEMA,
            0o444,
        ),
        "runtime_terminal": (
            "runtime_terminal_fingerprint",
            RUNTIME_TERMINAL_SCHEMA,
            0o444,
        ),
    }
    evidence: dict[str, dict[str, object]] = {}
    for key, (fingerprint_field, schema, expected_mode) in definitions.items():
        payload = _read_sealed(
            Path(str(artifacts[key])),
            fingerprint_field=fingerprint_field,
            schema=schema,
            expected_mode=expected_mode,
        )
        if (
            payload.get("execution_kind") != EXECUTION_KIND
            or any(payload.get(name) != value for name, value in common.items())
        ):
            raise PermissionError(f"{key} lineage differs from runtime spec")
        evidence[key] = payload
    phase_names = {
        "precommit_phase_receipt": "precommit",
        "start_ack_receipt": "start_ack",
        "child_prespawn_phase_receipt": "child_prespawn",
    }
    boot_id = authorization["manager_generation"]["boot_id"]
    immutable_fingerprint = spec["runtime"]["systemd"][
        "immutable_shadow_fingerprint"
    ]
    for key, phase in phase_names.items():
        state = evidence[key].get("systemd_phase_state")
        if (
            evidence[key].get("phase") != phase
            or evidence[key].get("boot_id") != boot_id
            or not isinstance(state, Mapping)
            or evidence[key].get("systemd_phase_state_fingerprint")
            != stable_fingerprint(dict(state))
            or evidence[key].get("immutable_shadow_fingerprint")
            != immutable_fingerprint
            or evidence[key].get("runtime_environment_audit_valid") is not False
            or evidence[key].get("environment_audit_fingerprint") is not None
            or evidence[key].get("environment_inventory_fingerprint") is not None
            or evidence[key].get("gpu_lease_fingerprint") is not None
        ):
            raise PermissionError(f"{phase} phase receipt is not exact")
    invocation_values = []
    for key in ("start_ack_receipt", "child_prespawn_phase_receipt"):
        state = evidence[key].get("systemd_phase_state")
        if not isinstance(state, Mapping):
            raise PermissionError(f"{key} has no systemd state")
        invocation_values.append(state.get("InvocationID"))
    invocation_values.extend((
        evidence["materialization_claim"].get("systemd_invocation_id"),
        evidence["runtime_terminal"].get("systemd_invocation_id"),
    ))
    invocation_id = invocation_values[0]
    if _INVOCATION.fullmatch(str(invocation_id)) is None or any(
        value != invocation_id for value in invocation_values
    ):
        raise PermissionError("runtime evidence InvocationID lineage diverged")
    invocation_dir = Path(str(artifacts["systemd_invocation_dir"]))
    rows = list(invocation_dir.iterdir())
    if len(rows) != 1 or rows[0].name != f"{invocation_id}.json":
        raise PermissionError("systemd terminal evidence is absent or ambiguous")
    systemd_terminal = _read_sealed(
        rows[0], fingerprint_field="systemd_terminal_fingerprint",
        schema=SYSTEMD_TERMINAL_SCHEMA, expected_mode=0o444,
    )
    _validate_systemd_sidecar_causality(systemd_terminal)
    if (
        any(systemd_terminal.get(name) != value for name, value in common.items())
        or systemd_terminal.get("claim_systemd_invocation_id") != invocation_id
        or systemd_terminal.get("sidecar_systemd_invocation_id") != invocation_id
        or systemd_terminal.get("audit_valid") is not True
        or not isinstance(systemd_terminal.get("systemd_outcome"), Mapping)
        or systemd_terminal["systemd_outcome"].get("systemd_success") is not True
    ):
        raise PermissionError("systemd terminal is not same-invocation successful")
    child_outcome = evidence["runtime_terminal"].get("child_outcome")
    if (
        not isinstance(child_outcome, Mapping)
        or child_outcome.get("category") != "EXITED_0"
        or evidence["runtime_terminal"].get("supervisor_error_type") is not None
    ):
        raise PermissionError("runtime terminal did not record one successful child")
    attempt = evidence["attempt_commit"]
    lease = evidence["launch_lease"]
    claim = evidence["materialization_claim"]
    precommit = evidence["precommit_phase_receipt"]
    start_ack = evidence["start_ack_receipt"]
    child_prespawn = evidence["child_prespawn_phase_receipt"]
    attempt_sha = sealed_file_sha256(attempt)
    claim_sha = sealed_file_sha256(claim)
    precommit_sha = sealed_file_sha256(precommit)
    start_ack_sha = sealed_file_sha256(start_ack)
    child_prespawn_sha = sealed_file_sha256(child_prespawn)
    if (
        attempt.get("authorization_fingerprint") is not None
        or attempt.get("authorization_file_sha256") is not None
        or any(
            payload.get("boot_id") != boot_id
            for payload in (
                attempt, lease, claim, precommit, start_ack,
                child_prespawn, evidence["runtime_terminal"],
            )
        )
        or lease.get("launch_limit") != 1
        or lease.get("lease_scope") != "attempt_dispatch_only"
        or lease.get("automatic_retry_allowed") is not False
        or lease.get("resume_allowed") is not False
        or claim.get("launch_limit") != 1
        or claim.get("shell") is not False
        or claim.get("automatic_retry_allowed") is not False
        or claim.get("resume_allowed") is not False
        or attempt.get("launch_lease_fingerprint")
        != lease.get("launch_lease_fingerprint")
        or attempt.get("launch_lease_file_sha256")
        != sealed_file_sha256(lease)
        or attempt.get("precommit_phase_receipt_fingerprint")
        != precommit.get("phase_receipt_fingerprint")
        or attempt.get("precommit_phase_receipt_file_sha256") != precommit_sha
        or claim.get("authorization_fingerprint") is not None
        or claim.get("attempt_commit_fingerprint")
        != attempt.get("attempt_commit_fingerprint")
        or claim.get("attempt_commit_file_sha256") != attempt_sha
        or any(
            receipt.get("launch_lease_fingerprint")
            != lease.get("launch_lease_fingerprint")
            for receipt in (precommit, start_ack, child_prespawn)
        )
        or evidence["runtime_terminal"].get(
            "materialization_claim_file_sha256"
        ) != claim_sha
        or evidence["runtime_terminal"].get("start_ack_receipt_fingerprint")
        != start_ack.get("phase_receipt_fingerprint")
        or evidence["runtime_terminal"].get("start_ack_receipt_file_sha256")
        != start_ack_sha
        or evidence["runtime_terminal"].get(
            "child_prespawn_phase_receipt_fingerprint"
        ) != child_prespawn.get("phase_receipt_fingerprint")
        or evidence["runtime_terminal"].get(
            "child_prespawn_phase_receipt_file_sha256"
        ) != child_prespawn_sha
        or attempt.get("runtime_environment_audit_valid") is not False
        or any(attempt.get(key) is not None for key in (
            "precommit_environment_audit_fingerprint",
            "precommit_environment_inventory_fingerprint", "gpu_lease_fingerprint",
            "gpu_lease_file_sha256", "gpu_lease_device", "gpu_lease_inode",
            "gpu_lease_parent_device", "gpu_lease_parent_inode",
        ))
        or lease.get("gpu_exclusivity_claimed") is not False
        or systemd_terminal.get("materialization_claim_fingerprint")
        != claim.get("materialization_claim_fingerprint")
        or systemd_terminal.get("materialization_claim_file_sha256") != claim_sha
        or systemd_terminal.get("attempt_commit_fingerprint")
        != attempt.get("attempt_commit_fingerprint")
        or systemd_terminal.get("attempt_commit_file_sha256") != attempt_sha
        or systemd_terminal.get("attempt_commit_required") is not True
        or systemd_terminal.get("attempt_commit_valid") is not True
        or systemd_terminal.get("current_authorization_valid") is not None
        or systemd_terminal.get("current_runtime_closure_valid") is not True
        or systemd_terminal.get("current_runtime_closure_error_type") is not None
        or systemd_terminal.get("authorization_matches_commit") is not True
        or systemd_terminal.get("claim_valid") is not True
        or systemd_terminal.get("claim_matches_invocation") is not True
        or systemd_terminal.get("start_ack_receipt_fingerprint")
        != start_ack.get("phase_receipt_fingerprint")
        or systemd_terminal.get("start_ack_receipt_file_sha256") != start_ack_sha
        or systemd_terminal.get("start_ack_valid") is not True
        or systemd_terminal.get(
            "child_prespawn_phase_receipt_fingerprint"
        ) != child_prespawn.get("phase_receipt_fingerprint")
        or systemd_terminal.get(
            "child_prespawn_phase_receipt_file_sha256"
        ) != child_prespawn_sha
        or systemd_terminal.get("child_prespawn_valid") is not True
        or systemd_terminal.get("finalizer_environment_audit_valid") is not None
        or any(systemd_terminal.get(key) is not None for key in (
            "active_gpu_lease_fingerprint",
            "active_gpu_lease_file_sha256",
            "active_gpu_lease_device", "active_gpu_lease_inode",
            "active_gpu_lease_parent_device",
            "active_gpu_lease_parent_inode",
            "active_gpu_lease_valid",
            "gpu_lease_release_authorized", "gpu_lease_release_valid",
            "gpu_lease_release_receipt_fingerprint", "gpu_lease_tombstone_file_sha256",
        ))
        or Path(str(artifacts["runtime_attestation"])).exists()
        or Path(str(artifacts["gpu_lease_release_receipt"])).exists()
    ):
        raise PermissionError("integration dummy forged environment/GPU evidence")
    dummy = _read_sealed(
        Path(str(authorization["control_artifacts"]["dummy_artifact"])),
        fingerprint_field="dummy_artifact_fingerprint",
        schema=DUMMY_ARTIFACT_SCHEMA,
        expected_mode=0o444,
    )
    if (
        dummy.get("scenario_id") != authorization["scenario_id"]
        or dummy.get("dataset_accessed") is not False
        or dummy.get("gpu_accessed") is not False
        or dummy.get("torch_imported") is not False
    ):
        raise PermissionError("dummy child evidence is not payload-free")
    return {
        "invocation_id": invocation_id,
        "attempt_commit_fingerprint": attempt["attempt_commit_fingerprint"],
        "launch_lease_fingerprint": lease["launch_lease_fingerprint"],
        "materialization_claim_fingerprint": evidence["materialization_claim"]["materialization_claim_fingerprint"],
        "precommit_fingerprint": evidence["precommit_phase_receipt"]["phase_receipt_fingerprint"],
        "start_ack_fingerprint": evidence["start_ack_receipt"]["phase_receipt_fingerprint"],
        "child_prespawn_fingerprint": evidence["child_prespawn_phase_receipt"]["phase_receipt_fingerprint"],
        "runtime_terminal_fingerprint": evidence["runtime_terminal"]["runtime_terminal_fingerprint"],
        "systemd_terminal_fingerprint": systemd_terminal["systemd_terminal_fingerprint"],
        "runtime_attestation_absent": True, "gpu_lease_evidence_absent": True,
    }


def _terminal_body(
    authorization: Mapping[str, object], *, passed: bool,
    completed_actions: Sequence[str], evidence: Mapping[str, object] | None,
    error: BaseException | None,
) -> dict[str, object]:
    return {
        "schema_version": INTEGRATION_TERMINAL_SCHEMA,
        "scenario_id": authorization["scenario_id"],
        "identity": authorization["identity"],
        "authorization_fingerprint": authorization["authorization_fingerprint"],
        "runtime_spec_fingerprint": authorization["runtime_spec_binding"]["runtime_spec_fingerprint"],
        "created_at_utc": _utc_now(), "passed": passed,
        "completed_actions": list(completed_actions),
        "supervisor_evidence": dict(evidence) if evidence is not None else None,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
        "direct_systemctl_start_attempted": False, "enable_attempted": False,
        "remove_attempted": False, "payload_authority": "none",
        "D_R_payload_accessed": False, "D_V_payload_accessed": False,
        "D_T_payload_accessed": False, "gpu_accessed": False,
    }


def _observed_fragment_identity(path: Path) -> dict[str, object]:
    raw, current = _read_stable_regular_file(
        path,
        expected_uid=os.getuid(),
        expected_mode=0o600,
        expected_nlink=1,
    )
    return {
        "fragment_path": str(path),
        "fragment_sha256": hashlib.sha256(raw).hexdigest(),
        "device": current.st_dev, "inode": current.st_ino,
        "owner_uid": current.st_uid, "mode": stat.S_IMODE(current.st_mode),
        "nlink": current.st_nlink,
    }


def _load_removal_authorization(
    path: Path, *, authorization: Mapping[str, object],
    terminal: Mapping[str, object], expected_fragment_identity: Mapping[str, object],
    runner: CommandRunner, manager_reader: ManagerReader,
) -> dict[str, object]:
    removal = _read_sealed(
        path, fingerprint_field="removal_authorization_fingerprint",
        schema=REMOVAL_AUTHORIZATION_SCHEMA,
    )
    if set(removal) != _REMOVAL_AUTH_KEYS:
        raise PermissionError("removal authorization keys are not exact")
    issued = datetime.fromisoformat(
        str(removal["issued_at_utc"]).replace("Z", "+00:00")
    )
    expires = datetime.fromisoformat(
        str(removal["expires_at_utc"]).replace("Z", "+00:00")
    )
    now = datetime.now(timezone.utc)
    identity = authorization["identity"]
    if (
        issued.tzinfo is None or expires.tzinfo is None
        or not issued <= now <= expires
        or expires - issued > timedelta(seconds=300)
        or removal["scenario_id"] != authorization["scenario_id"]
        or removal["unit_name"] != identity["unit_name"]
        or removal["authorization_fingerprint"]
        != authorization["authorization_fingerprint"]
        or removal["integration_terminal_fingerprint"]
        != terminal["integration_terminal_fingerprint"]
        or removal["runtime_spec_fingerprint"]
        != authorization["runtime_spec_binding"]["runtime_spec_fingerprint"]
        or removal["supervisor_evidence"] != terminal["supervisor_evidence"]
        or removal["fragment_identity"] != dict(expected_fragment_identity)
        or removal["remove_authorized"] is not True
        or removal["daemon_reload_authorized"] is not True
        or removal["not_found_verification_authorized"] is not True
        or removal["enable_authorized"] is not False
        or removal["start_authorized"] is not False
        or removal["payload_authority"] != "none"
        or any(removal[key] is not False for key in (
            "D_R_payload_accessed", "D_V_payload_accessed",
            "D_T_payload_accessed",
        ))
    ):
        raise PermissionError("removal authorization is stale or not exact")
    manager = manager_reader()
    _validate_manager_generation(manager)
    if manager != removal["manager_generation"]:
        raise PermissionError("user manager generation changed before removal")
    fragment = Path(str(expected_fragment_identity["fragment_path"]))
    policy = realizer.freeze_user_unit_path_policy(
        str(identity["unit_name"]), runner=runner, allowed_fragment=fragment,
    )
    _validate_unit_path_policy_transition(
        removal["unit_path_policy"],
        policy,
        authorized_uid=authorization["authorized_uid"],
    )
    if _observed_fragment_identity(fragment) != dict(expected_fragment_identity):
        raise PermissionError("fragment identity changed before removal")
    current_state = realizer.query_unit_properties(
        str(identity["unit_name"]), runner=runner
    )
    if (
        current_state != removal["inactive_static_state"]
        or current_state.get("ActiveState") != "inactive"
        or current_state.get("SubState") != "dead"
    ):
        raise PermissionError("unit is not terminal and inactive before removal")
    return removal


def run_authorized_integration(
    authorization_path: Path, *, execute: bool,
    runner: CommandRunner = run_command,
    manager_reader: ManagerReader = collect_manager_generation,
    timeout_seconds: float = 10.0,
) -> dict[str, object]:
    if not execute:
        raise PermissionError("explicit authorized integration execution is required")
    completed_actions: list[str] = []
    authorization: dict[str, object] | None = None
    evidence: dict[str, object] | None = None
    terminal_written = False
    removal_attempted = False
    fragment: Path | None = None
    try:
        authorization = load_integration_authorization(
            authorization_path, runner=runner, manager_reader=manager_reader
        )
        identity = authorization["identity"]
        fragment = Path(str(authorization["unit_directory"])) / str(identity["unit_name"])
        before = realizer.query_unit_properties(str(identity["unit_name"]), runner=runner)
        if before.get("LoadState") != "not-found" or before.get("FragmentPath") != "":
            raise PermissionError("integration unit exists before realization")
        plan = realizer.build_realization_plan(
            unit_name=str(identity["unit_name"]),
            unit_directory=Path(str(authorization["unit_directory"])),
            fragment_text=str(authorization["rendered_fragment"]["utf8_text"]),
            expected_fragment_sha256=str(authorization["rendered_fragment"]["sha256"]),
            execute_authorized=True, removal_authorized=False,
        )
        fragment_identity = realizer.realize_static_fragment(plan, execute=True)
        completed_actions.append("realize-static-fragment")
        pre_reload_policy = realizer.freeze_user_unit_path_policy(
            str(identity["unit_name"]), runner=runner, allowed_fragment=fragment
        )
        _validate_unit_path_policy_transition(
            authorization["unit_path_policy"],
            pre_reload_policy,
            authorized_uid=authorization["authorized_uid"],
        )
        realizer.daemon_reload(execute=True, runner=runner)
        completed_actions.append("daemon-reload-after-realization")
        post_reload_policy = realizer.freeze_user_unit_path_policy(
            str(identity["unit_name"]), runner=runner, allowed_fragment=fragment,
        )
        _validate_unit_path_policy_transition(
            authorization["unit_path_policy"],
            post_reload_policy,
            authorized_uid=authorization["authorized_uid"],
            allow_generator_late_inode_rotation=True,
        )
        static_state = realizer.query_unit_properties(str(identity["unit_name"]), runner=runner)
        realizer.validate_realized_static_unit(plan, static_state)
        immutable = _query_immutable_shadow(str(identity["unit_name"]), runner=runner)
        spec = _read_sealed(
            Path(str(authorization["runtime_spec_binding"]["path"])),
            fingerprint_field="runtime_spec_fingerprint", schema=RUNTIME_SPEC_SCHEMA,
        )
        if immutable != spec["runtime"]["systemd"]["immutable_shadow_properties"]:
            raise PermissionError("loaded supervisor-v2 immutable shadow changed")
        precommit_policy = realizer.freeze_user_unit_path_policy(
            str(identity["unit_name"]), runner=runner, allowed_fragment=fragment,
        )
        _validate_unit_path_policy_transition(
            post_reload_policy,
            precommit_policy,
            authorized_uid=authorization["authorized_uid"],
        )
        manager = manager_reader()
        _validate_manager_generation(manager)
        if manager != authorization["manager_generation"]:
            raise PermissionError("user manager generation changed before commit")
        supervisor_binding = authorization["executable_bindings"]["supervisor"]
        python_binding = authorization["executable_bindings"]["python"]
        command = (
            str(python_binding["path"]), "-I", "-u",
            str(supervisor_binding["path"]), "commit-and-start", "--spec",
            str(authorization["runtime_spec_binding"]["path"]),
        )
        result = runner(command)
        if result.returncode != 0:
            loaded_sidecar = _load_single_systemd_sidecar(authorization)
            if loaded_sidecar is not None:
                sidecar, sidecar_path = loaded_sidecar
                evidence = _sidecar_summary(sidecar, sidecar_path)
                outcome = sidecar["systemd_outcome"]
                if not isinstance(outcome, Mapping):
                    raise AssertionError(
                        "validated sidecar outcome changed"
                    )
                if (
                    outcome.get("systemd_success") is not True
                    or sidecar.get("audit_valid") is not True
                ):
                    raise _sidecar_failure_error(sidecar)
            raise RuntimeError("supervisor commit-and-start failed")
        completed_actions.append("supervisor-commit-and-start")
        deadline = time.monotonic() + timeout_seconds
        terminal_path = Path(str(spec["artifacts"]["runtime_terminal"]))
        invocation_dir = Path(str(spec["artifacts"]["systemd_invocation_dir"]))
        while True:
            loaded_sidecar = _load_single_systemd_sidecar(authorization)
            if loaded_sidecar is not None:
                sidecar, sidecar_path = loaded_sidecar
                evidence = _sidecar_summary(sidecar, sidecar_path)
                outcome = sidecar["systemd_outcome"]
                if not isinstance(outcome, Mapping):
                    raise AssertionError(
                        "validated sidecar outcome changed"
                    )
                if (
                    outcome.get("systemd_success") is not True
                    or sidecar.get("audit_valid") is not True
                ):
                    raise _sidecar_failure_error(sidecar)
                if terminal_path.exists():
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError("supervisor terminal evidence did not arrive")
            time.sleep(0.01)
        evidence = _validate_supervisor_evidence(authorization)
        completed_actions.append("verify-supervisor-evidence")
        control = authorization["control_artifacts"]
        terminal = _write_sealed(
            Path(str(control["integration_terminal"])),
            _terminal_body(
                authorization, passed=True, completed_actions=completed_actions,
                evidence=evidence, error=None,
            ),
            fingerprint_field="integration_terminal_fingerprint",
        )
        terminal_written = True
        inactive = realizer.wait_until_unit_inactive_static(
            plan,
            query=lambda unit: realizer.query_unit_properties(unit, runner=runner),
            timeout_seconds=timeout_seconds, poll_seconds=0.01,
        )
        removal_manager = manager_reader()
        _validate_manager_generation(removal_manager)
        if removal_manager != authorization["manager_generation"]:
            raise PermissionError("user manager generation changed after terminal")
        removal_policy = realizer.freeze_user_unit_path_policy(
            str(identity["unit_name"]), runner=runner, allowed_fragment=fragment,
        )
        _validate_unit_path_policy_transition(
            post_reload_policy,
            removal_policy,
            authorized_uid=authorization["authorized_uid"],
        )
        removal_issued = datetime.now(timezone.utc)
        removal_body = {
            "schema_version": REMOVAL_AUTHORIZATION_SCHEMA,
            "scenario_id": authorization["scenario_id"],
            "unit_name": identity["unit_name"],
            "authorization_fingerprint": authorization["authorization_fingerprint"],
            "integration_terminal_fingerprint": terminal["integration_terminal_fingerprint"],
            "runtime_spec_fingerprint": authorization["runtime_spec_binding"]["runtime_spec_fingerprint"],
            "supervisor_evidence": evidence, "fragment_identity": fragment_identity,
            "inactive_static_state": inactive, "manager_generation": removal_manager,
            "unit_path_policy": removal_policy,
            "issued_at_utc": removal_issued.isoformat().replace("+00:00", "Z"),
            "expires_at_utc": (
                removal_issued + timedelta(seconds=300)
            ).isoformat().replace("+00:00", "Z"),
            "remove_authorized": True, "daemon_reload_authorized": True,
            "not_found_verification_authorized": True,
            "enable_authorized": False, "start_authorized": False,
            "payload_authority": "none",
            "D_R_payload_accessed": False, "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        removal_auth = _write_sealed(
            Path(str(control["removal_authorization"])), removal_body,
            fingerprint_field="removal_authorization_fingerprint",
        )
        removal_auth = _load_removal_authorization(
            Path(str(control["removal_authorization"])),
            authorization=authorization, terminal=terminal,
            expected_fragment_identity=fragment_identity, runner=runner,
            manager_reader=manager_reader,
        )
        removal_attempted = True
        removal_plan = realizer.build_realization_plan(
            unit_name=plan.unit_name, unit_directory=plan.unit_directory,
            fragment_text=str(authorization["rendered_fragment"]["utf8_text"]),
            expected_fragment_sha256=plan.fragment_sha256,
            execute_authorized=True, removal_authorized=True,
        )
        realizer.remove_integration_fragment(removal_plan, execute=True)
        completed_actions.append("remove-authorized-fragment")
        realizer.daemon_reload(execute=True, runner=runner)
        completed_actions.append("daemon-reload-after-removal")
        post_remove_policy = realizer.freeze_user_unit_path_policy(
            str(identity["unit_name"]), runner=runner,
        )
        _validate_unit_path_policy_transition(
            removal_policy,
            post_remove_policy,
            authorized_uid=authorization["authorized_uid"],
            allow_generator_late_inode_rotation=True,
        )
        not_found = realizer.wait_until_unit_not_found(
            plan.unit_name,
            query=lambda unit: realizer.query_unit_properties(unit, runner=runner),
            timeout_seconds=timeout_seconds, poll_seconds=0.01,
        )
        expected_not_found = {
            "LoadState": "not-found",
            "UnitFileState": "",
            "ActiveState": "inactive",
            "SubState": "dead",
            "FragmentPath": "",
            "DropInPaths": "",
            "Transient": "no",
            "Restart": "no",
            "NRestarts": "0",
            "NeedDaemonReload": "no",
        }
        if not _deep_exact_equal(not_found, expected_not_found):
            raise PermissionError("removed integration unit is not exact not-found")
        final_policy = realizer.freeze_user_unit_path_policy(
            str(identity["unit_name"]), runner=runner,
        )
        _validate_unit_path_policy_transition(
            post_remove_policy,
            final_policy,
            authorized_uid=authorization["authorized_uid"],
        )
        post_remove_manager = manager_reader()
        _validate_manager_generation(post_remove_manager)
        if not _deep_exact_equal(
            removal_auth["manager_generation"], post_remove_manager
        ):
            raise PermissionError(
                "user manager generation changed after removal"
            )
        if os.path.lexists(plan.fragment_path):
            raise PermissionError(
                "authorized fragment reappeared after removal"
            )
        removal_state = _write_sealed(
            Path(str(control["removal_state"])),
            {
                "schema_version": REMOVAL_STATE_SCHEMA,
                "scenario_id": authorization["scenario_id"],
                "unit_name": plan.unit_name,
                "removal_authorization_fingerprint": removal_auth["removal_authorization_fingerprint"],
                "passed": True, "remove_attempted": True,
                "fragment_absent": True,
                "not_found_state": not_found, "completed_actions": completed_actions,
                "error_type": None, "error_message": None,
                "payload_authority": "none", "D_R_payload_accessed": False,
                "D_V_payload_accessed": False, "D_T_payload_accessed": False,
            },
            fingerprint_field="removal_state_fingerprint",
        )
        receipt = _write_sealed(
            Path(str(control["integration_receipt"])),
            {
                "schema_version": INTEGRATION_RECEIPT_SCHEMA,
                "scenario_id": authorization["scenario_id"],
                "identity": authorization["identity"],
                "authorization_path": str(authorization_path.absolute()),
                "authorization_file_sha256": sealed_file_sha256(
                    authorization
                ),
                "authorization_fingerprint": authorization[
                    "authorization_fingerprint"
                ],
                "integration_terminal_path": str(
                    control["integration_terminal"]
                ),
                "integration_terminal_file_sha256": sealed_file_sha256(
                    terminal
                ),
                "integration_terminal_fingerprint": terminal[
                    "integration_terminal_fingerprint"
                ],
                "removal_authorization_path": str(
                    control["removal_authorization"]
                ),
                "removal_authorization_file_sha256": sealed_file_sha256(
                    removal_auth
                ),
                "removal_authorization_fingerprint": removal_auth[
                    "removal_authorization_fingerprint"
                ],
                "removal_state_path": str(control["removal_state"]),
                "removal_state_file_sha256": sealed_file_sha256(
                    removal_state
                ),
                "removal_state_fingerprint": removal_state[
                    "removal_state_fingerprint"
                ],
                "supervisor_evidence": evidence,
                "fragment_removed": not os.path.lexists(
                    plan.fragment_path
                ),
                "post_removal_unit_state": not_found,
                "passed": True,
                "payload_authority": "none",
                "D_R_payload_accessed": False,
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
                "gpu_accessed": False,
            },
            fingerprint_field="receipt_fingerprint",
        )
        return {
            "terminal": terminal,
            "removal_state": removal_state,
            "receipt": receipt,
        }
    except BaseException as error:
        if authorization is not None:
            control = authorization["control_artifacts"]
            if not terminal_written and not os.path.lexists(str(control["integration_terminal"])):
                _write_sealed(
                    Path(str(control["integration_terminal"])),
                    _terminal_body(
                        authorization, passed=False, completed_actions=completed_actions,
                        evidence=evidence, error=error,
                    ),
                    fingerprint_field="integration_terminal_fingerprint",
                )
            if not os.path.lexists(str(control["removal_state"])):
                _write_sealed(
                    Path(str(control["removal_state"])),
                    {
                        "schema_version": REMOVAL_STATE_SCHEMA,
                        "scenario_id": authorization["scenario_id"],
                        "unit_name": authorization["identity"]["unit_name"],
                        "removal_authorization_fingerprint": None,
                        "passed": False, "remove_attempted": removal_attempted,
                        "fragment_absent": (
                            fragment is not None and not os.path.lexists(fragment)
                        ),
                        "not_found_state": None, "completed_actions": completed_actions,
                        "error_type": type(error).__name__, "error_message": str(error),
                        "payload_authority": "none", "D_R_payload_accessed": False,
                        "D_V_payload_accessed": False, "D_T_payload_accessed": False,
                    },
                    fingerprint_field="removal_state_fingerprint",
                )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build-authorization")
    build.add_argument("--scenario-root", type=Path, required=True)
    build.add_argument("--scenario-id", required=True)
    build.add_argument("--template", type=Path, required=True)
    build.add_argument("--python", type=Path, required=True)
    build.add_argument("--supervisor", type=Path, required=True)
    build.add_argument("--realizer", type=Path, required=True)
    build.add_argument("--dummy-child", type=Path, required=True)
    build.add_argument("--instruction-id", required=True)
    build.add_argument("--authorization-basis", required=True)
    build.add_argument("--validity-seconds", type=int, default=300)
    verify = sub.add_parser("verify-authorization")
    verify.add_argument("--authorization", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--execute-authorized-integration", action="store_true")
    run.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "build-authorization":
        create_production_authorization(
            args.scenario_root, scenario_id=args.scenario_id,
            template_path=args.template, python_path=args.python,
            supervisor_path=args.supervisor, realizer_path=args.realizer,
            dummy_child_path=args.dummy_child, instruction_id=args.instruction_id,
            authorization_basis=args.authorization_basis,
            validity_seconds=args.validity_seconds,
        )
        return 0
    if args.mode == "verify-authorization":
        load_integration_authorization(args.authorization)
        return 0
    run_authorized_integration(
        args.authorization, execute=args.execute_authorized_integration,
        timeout_seconds=args.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
