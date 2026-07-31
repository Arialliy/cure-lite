#!/usr/bin/env python3
"""Create-only realization of the exact CURE-Lite v24 actual r2 user unit.

The tool is standard-library-only and payload-blind.  It can install exactly
one static runtime user unit after a fresh, sealed authorization.  It never
enables, starts, stops, removes, or resets a unit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
from typing import Callable, Mapping, Sequence


ACTUAL_UNIT = "cure-lite-v24-gcr-pacre-dr-r2.service"
CANDIDATE = "GCR-PACRE-v24"
STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
ATTEMPT_ID = "gcr_pacre_v24_D_R_zero_update_structural_r2"
INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后运行"
AUTHORIZATION_SCHEMA = "cure-lite-v24-actual-unit-realization-authorization-v1"
RECEIPT_SCHEMA = "cure-lite-v24-actual-unit-realization-receipt-v1"
TERMINAL_SCHEMA = "cure-lite-v24-actual-unit-realization-terminal-v1"
SUPERVISOR_SPEC_SCHEMA = "cure-lite-v24-dr-runtime-supervisor-spec-v2"

SYSTEMD_PATH = "/usr/bin/systemd-path"
SYSTEMD_ANALYZE = "/usr/bin/systemd-analyze"
SYSTEMCTL = "/usr/bin/systemctl"
PYTHON_PATH = Path("/usr/bin/python3.12")

_SHA = re.compile(r"[0-9a-f]{64}")
_BOOT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
_PLACEHOLDERS = {
    "@@PYTHON_PATH@@",
    "@@SUPERVISOR_PATH@@",
    "@@RUNTIME_SPEC_PATH@@",
}
_ACTIONS = [
    "install-runtime-static-fragment",
    "daemon-reload",
    "verify-static-shadow",
]
_EXEC_MODES = {
    "ExecCondition": "claim-materialization",
    "ExecStartPre": "verify-runtime-spec",
    "ExecStart": "run-once",
    "ExecStopPost": "record-systemd-exit",
}
_FILE_BINDING_KEYS = {
    "path", "resolved_path", "path_is_symlink", "file_sha256",
    "device", "inode", "owner_uid", "mode",
}
_SHADOW_PROPERTIES = (
    "Id", "Description", "LoadState", "ActiveState", "SubState",
    "UnitFileState", "FragmentPath", "DropInPaths", "Transient",
    "Restart", "RestartUSec", "NRestarts", "NeedDaemonReload", "Type",
    "ExitType", "KillMode", "KillSignal", "SendSIGKILL",
    "TimeoutStartUSec", "TimeoutStopUSec", "RuntimeMaxUSec",
    "WatchdogUSec", "OOMPolicy", "RemainAfterExit", "StandardInput",
    "StandardOutput", "StandardError", "StartLimitIntervalUSec",
    "StartLimitBurst", "UMask", "Environment", "UnsetEnvironment",
    "WorkingDirectory", "TriggeredBy", "Triggers", "WantedBy",
    "RequiredBy", "PartOf", "SyslogIdentifier", "SuccessExitStatus",
    "ExecCondition", "ExecStartPre", "ExecStart", "ExecStopPost",
)
_AUTH_KEYS = {
    "schema_version", "candidate", "stage_id", "attempt_id", "unit_name",
    "instruction_id", "authorization_basis", "authorized_uid",
    "created_at_utc", "issued_at_utc", "expires_at_utc", "actions",
    "persistent_install_authorized", "enable_authorized", "start_authorized",
    "remove_authorized", "unit_directory", "unit_path_policy",
    "manager_generation", "template_binding", "rendered_fragment",
    "runtime_spec_binding", "executable_bindings", "payload_authority",
    "expected_static_shadow",
    "D_R_payload_accessed", "D_V_payload_accessed", "D_T_payload_accessed",
    "authorization_fingerprint",
}


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ManagerReader = Callable[[], dict[str, object]]


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def stable_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _deep_exact_equal(left: object, right: object) -> bool:
    """Compare JSON-shaped evidence without Python numeric coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return (
            set(left) == set(right)
            and all(_deep_exact_equal(left[key], right[key]) for key in left)
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


def _stat_generation(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _stat_snapshot(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid,
        value.st_gid, value.st_nlink, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _parent_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid,
        stat.S_IMODE(value.st_mode),
    )


def _open_stable_parent(
    target: Path, *, private: bool = False,
) -> tuple[int, os.stat_result]:
    parent = target.parent
    before = os.lstat(parent)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or parent.resolve(strict=True) != parent
        or (
            private
            and (
                before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o700
            )
        )
    ):
        requirement = (
            "evidence parent must be owned mode 0700"
            if private else "bound file parent is not canonical"
        )
        raise PermissionError(requirement)
    parent_fd = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(parent_fd)
    if (
        _parent_identity(opened) != _parent_identity(before)
        or not stat.S_ISDIR(opened.st_mode)
    ):
        os.close(parent_fd)
        raise PermissionError("bound file parent changed while opening")
    return parent_fd, before


def _verify_parent_generation(
    parent: Path, parent_fd: int, expected: os.stat_result,
) -> None:
    opened = os.fstat(parent_fd)
    linked = os.lstat(parent)
    if (
        _parent_identity(opened) != _parent_identity(expected)
        or _parent_identity(linked) != _parent_identity(expected)
        or not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
    ):
        raise PermissionError("bound file parent generation changed")


def _read_all_fd(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _stable_read_file(
    path: str | Path, *, private_parent: bool = False,
) -> tuple[bytes, dict[str, object]]:
    target = Path(path).absolute()
    if target.name in {"", ".", ".."}:
        raise ValueError("bound file must have one basename")
    parent_fd, parent_before = _open_stable_parent(
        target, private=private_parent,
    )
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError("bound file is not regular")
        raw = _read_all_fd(descriptor)
        after = os.fstat(descriptor)
        linked = os.stat(
            target.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            _stat_snapshot(after) != _stat_snapshot(before)
            or _stat_snapshot(linked) != _stat_snapshot(after)
            or not stat.S_ISREG(linked.st_mode)
        ):
            raise PermissionError("bound file identity changed while reading")
        _verify_parent_generation(target.parent, parent_fd, parent_before)
        identity = {
            "path": str(target), "resolved_path": str(target),
            "path_is_symlink": False,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "device": after.st_dev, "inode": after.st_ino,
            "owner_uid": after.st_uid, "mode": stat.S_IMODE(after.st_mode),
            "nlink": after.st_nlink,
        }
        return raw, identity
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def file_sha256(path: str | Path) -> str:
    return str(_stable_read_file(path)[1]["file_sha256"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fixed_environment() -> dict[str, str]:
    uid = os.getuid()
    runtime = f"/run/user/{uid}"
    return {
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime}/bus",
        "HOME": pwd.getpwuid(uid).pw_dir,
        "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin",
        "SYSTEMD_COLORS": "0", "XDG_RUNTIME_DIR": runtime,
    }


def _run(argv: Sequence[str], *, runner: CommandRunner = subprocess.run):
    exact = tuple(argv)
    allowed = {
        (SYSTEMD_PATH, "--suffix=systemd/user", "user-runtime"),
        (SYSTEMD_ANALYZE, "--user", "unit-paths", "--no-pager"),
        (SYSTEMCTL, "--user", "daemon-reload"),
        _shadow_argv(),
    }
    if exact not in allowed:
        raise ValueError("command is outside the actual-realization allowlist")
    return runner(
        list(exact), shell=False, check=False, capture_output=True, text=True,
        timeout=30.0, env=_fixed_environment(),
    )


def _shadow_argv() -> tuple[str, ...]:
    argv: list[str] = [SYSTEMCTL, "--user", "show", ACTUAL_UNIT, "--no-pager"]
    for name in _SHADOW_PROPERTIES:
        argv.extend(("-p", name))
    return tuple(argv)


def _parse_show(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in text.splitlines():
        if "=" not in row:
            raise ValueError("systemctl show output is malformed")
        key, value = row.split("=", 1)
        if key in result:
            raise ValueError("systemctl show output has duplicate keys")
        result[key] = (
            "disabled"
            if key == "WatchdogUSec"
            and value in {"0", "infinity", "disabled"}
            else value
        )
    expected = set(_SHADOW_PROPERTIES)
    missing = expected - set(result)
    extra = set(result) - expected
    if extra or (
        missing
        and (
            result.get("LoadState") != "not-found"
            or not missing.issubset(_EXEC_MODES)
        )
    ):
        raise ValueError("systemctl show property set is not exact")
    for name in missing:
        result[name] = ""
    if set(result) != expected:
        raise AssertionError("normalized systemctl property set is incomplete")
    return result


def query_shadow(*, runner: CommandRunner = subprocess.run) -> dict[str, str]:
    completed = _run(_shadow_argv(), runner=runner)
    if completed.returncode != 0:
        raise RuntimeError("systemctl show failed")
    result = _parse_show(completed.stdout)
    if result["Id"] != ACTUAL_UNIT:
        raise PermissionError("systemctl unit identity changed")
    return result


def _require_no_shadow(shadow: Mapping[str, str]) -> None:
    if (
        shadow.get("Id") != ACTUAL_UNIT
        or shadow.get("LoadState") != "not-found"
        or shadow.get("ActiveState") != "inactive"
        or shadow.get("SubState") != "dead"
        or shadow.get("UnitFileState") != ""
        or shadow.get("FragmentPath") != ""
        or shadow.get("DropInPaths") != ""
        or any(shadow.get(name) for name in (
            "TriggeredBy", "Triggers", "WantedBy", "RequiredBy", "PartOf"
        ))
        or any(shadow.get(name) for name in _EXEC_MODES)
    ):
        raise PermissionError("an actual-unit shadow already exists")


def _path_row(path: Path) -> dict[str, object]:
    if not os.path.lexists(path):
        return {"path": str(path), "exists": False}
    linked = path.lstat()
    if path.is_symlink():
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
        if (
            not stat.S_ISLNK(linked.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or resolved.is_symlink()
            or resolved.resolve(strict=True) != resolved
            or linked.st_uid not in {0, os.getuid()}
            or metadata.st_uid not in {0, os.getuid()}
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise PermissionError(
                "unit search path symlink target is not trusted"
            )
        linked_after = path.lstat()
        resolved_after = path.resolve(strict=True)
        metadata_after = resolved_after.stat()
        if (
            _stat_snapshot(linked_after) != _stat_snapshot(linked)
            or resolved_after != resolved
            or _stat_snapshot(metadata_after) != _stat_snapshot(metadata)
        ):
            raise PermissionError("unit search path symlink changed while observed")
        return {
            "path": str(path),
            "exists": True,
            "path_is_symlink": True,
            "link_target": os.readlink(path),
            "link_device": linked.st_dev,
            "link_inode": linked.st_ino,
            "link_owner_uid": linked.st_uid,
            "link_mode": stat.S_IMODE(linked.st_mode),
            "link_nlink": linked.st_nlink,
            "resolved_path": str(resolved),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "owner_uid": metadata.st_uid,
            "mode": stat.S_IMODE(metadata.st_mode),
        }
    metadata = linked
    if not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError("unit search path is not a real directory")
    if path.resolve(strict=True) != path:
        raise PermissionError("unit search path is not canonical")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        linked_after = path.lstat()
        if (
            _stat_snapshot(opened) != _stat_snapshot(metadata)
            or _stat_snapshot(linked_after) != _stat_snapshot(metadata)
        ):
            raise PermissionError("unit search path changed while observed")
    finally:
        os.close(descriptor)
    return {
        "path": str(path), "exists": True, "device": metadata.st_dev,
        "inode": metadata.st_ino, "owner_uid": metadata.st_uid,
        "mode": stat.S_IMODE(metadata.st_mode),
    }


def _observe_unit_path_policy(
    *, runner: CommandRunner, allowed_fragment: Path | None,
) -> dict[str, object]:
    resolved = _run(
        (SYSTEMD_PATH, "--suffix=systemd/user", "user-runtime"),
        runner=runner,
    )
    analyzed = _run(
        (SYSTEMD_ANALYZE, "--user", "unit-paths", "--no-pager"),
        runner=runner,
    )
    if resolved.returncode != 0 or analyzed.returncode != 0:
        raise RuntimeError("user unit path discovery failed")
    runtime_rows = [row for row in resolved.stdout.splitlines() if row]
    path_rows = [row for row in analyzed.stdout.splitlines() if row]
    if len(runtime_rows) != 1 or not path_rows or len(path_rows) != len(set(path_rows)):
        raise PermissionError("user unit path discovery is ambiguous")
    runtime_directory = Path(runtime_rows[0])
    paths = [Path(row) for row in path_rows]
    if (
        not runtime_directory.is_absolute()
        or any(not path.is_absolute() for path in paths)
        or paths.count(runtime_directory) != 1
    ):
        raise PermissionError("runtime user unit directory has no unique priority")
    runtime_identity = _path_row(runtime_directory)
    if (
        runtime_identity.get("exists") is not True
        or runtime_identity.get("owner_uid") != os.getuid()
        or int(runtime_identity.get("mode", 0)) & 0o022
    ):
        raise PermissionError("runtime user unit directory is not owned and safe")
    identities = [_path_row(path) for path in paths]
    for path in paths:
        candidate = path / ACTUAL_UNIT
        if os.path.lexists(candidate) and candidate != allowed_fragment:
            raise PermissionError("an actual-unit fragment shadows the target")
        if candidate == allowed_fragment and os.path.lexists(candidate):
            _, observed_fragment = _stable_read_file(candidate)
            if (
                observed_fragment["owner_uid"] != os.getuid()
                or observed_fragment["nlink"] != 1
                or observed_fragment["mode"] != 0o600
            ):
                raise PermissionError("the installed actual fragment is unsafe")
    return {
        "systemd_path_argv": [
            SYSTEMD_PATH, "--suffix=systemd/user", "user-runtime"
        ],
        "systemd_path_stdout": resolved.stdout,
        "systemd_analyze_argv": [
            SYSTEMD_ANALYZE, "--user", "unit-paths", "--no-pager"
        ],
        "systemd_analyze_stdout": analyzed.stdout,
        "ordered_unit_paths": identities,
        "runtime_directory": str(runtime_directory),
        "runtime_directory_priority": paths.index(runtime_directory),
        "runtime_directory_identity": runtime_identity,
    }


def freeze_unit_path_policy(
    *, runner: CommandRunner = subprocess.run,
) -> dict[str, object]:
    return _observe_unit_path_policy(runner=runner, allowed_fragment=None)


def _path_policies_are_exact(
    expected: Mapping[str, object], observed: Mapping[str, object],
) -> bool:
    return _deep_exact_equal(expected, observed)


def _revalidate_unit_path_policy(
    expected: Mapping[str, object], *, runner: CommandRunner,
) -> dict[str, object]:
    observed = freeze_unit_path_policy(runner=runner)
    if not _path_policies_are_exact(expected, observed):
        raise PermissionError("user unit search path policy changed")
    return observed


def _revalidate_installed_path_policy(
    expected: Mapping[str, object], *, fragment: Path, runner: CommandRunner,
) -> dict[str, object]:
    observed = _observe_unit_path_policy(
        runner=runner, allowed_fragment=fragment.absolute()
    )
    if not _path_policies_are_exact(expected, observed):
        raise PermissionError("installed user unit search path policy changed")
    return observed


def _validate_daemon_reload_path_policy_transition(
    before: Mapping[str, object], after: Mapping[str, object], *,
    runtime_directory: Path, authorized_uid: int,
) -> dict[str, object]:
    """Accept deep equality or one daemon-reload generator.late inode rotation."""

    if _path_policies_are_exact(before, after):
        return json.loads(canonical_json(after))
    before_normalized = json.loads(canonical_json(before))
    after_normalized = json.loads(canonical_json(after))
    before_rows = before_normalized.get("ordered_unit_paths")
    after_rows = after_normalized.get("ordered_unit_paths")
    target = str(runtime_directory.parent / "generator.late")
    if (
        not isinstance(before_rows, list)
        or not isinstance(after_rows, list)
        or len(before_rows) != len(after_rows)
    ):
        raise PermissionError("daemon-reload user unit path policy changed")
    before_indexes = [
        index for index, row in enumerate(before_rows)
        if isinstance(row, dict) and row.get("path") == target
    ]
    after_indexes = [
        index for index, row in enumerate(after_rows)
        if isinstance(row, dict) and row.get("path") == target
    ]
    if (
        len(before_indexes) != 1
        or before_indexes != after_indexes
    ):
        raise PermissionError("daemon-reload generator.late index changed")
    index = before_indexes[0]
    before_row = before_rows[index]
    after_row = after_rows[index]
    row_keys = {"path", "exists", "device", "inode", "owner_uid", "mode"}
    if set(before_row) != row_keys or set(after_row) != row_keys:
        raise PermissionError("daemon-reload generator.late schema changed")
    before_inode = before_row["inode"]
    after_inode = after_row["inode"]
    devices = (before_row["device"], after_row["device"])
    modes = (before_row["mode"], after_row["mode"])
    owners = (before_row["owner_uid"], after_row["owner_uid"])
    if (
        before_row["exists"] is not True
        or after_row["exists"] is not True
        or any(
            isinstance(owner, bool)
            or not isinstance(owner, int)
            or owner != authorized_uid
            for owner in owners
        )
        or any(
            isinstance(device, bool)
            or not isinstance(device, int)
            or device < 0
            for device in devices
        )
        or any(
            isinstance(mode, bool)
            or not isinstance(mode, int)
            or mode < 0
            or mode > 0o7777
            or mode & 0o022
            for mode in modes
        )
        or isinstance(before_inode, bool)
        or not isinstance(before_inode, int)
        or before_inode <= 0
        or isinstance(after_inode, bool)
        or not isinstance(after_inode, int)
        or after_inode <= 0
        or before_inode == after_inode
    ):
        raise PermissionError("daemon-reload generator.late identity is invalid")
    after_row["inode"] = before_inode
    if not _path_policies_are_exact(before_normalized, after_normalized):
        raise PermissionError("daemon-reload user unit path policy changed")
    return json.loads(canonical_json(after))


def _read_process(pid: int) -> dict[str, object]:
    root = Path(f"/proc/{pid}")
    first = (root / "stat").read_text(encoding="ascii")
    close = first.rfind(") ")
    fields = first[close + 2:].split() if close >= 0 else []
    if len(fields) <= 19:
        raise RuntimeError("manager process stat is malformed")
    starttime = int(fields[19])
    status = (root / "status").read_text(encoding="ascii")
    uid_rows = [row for row in status.splitlines() if row.startswith("Uid:")]
    cgroup_rows = (root / "cgroup").read_text(encoding="utf-8").splitlines()
    unified = [row.split("::", 1)[1] for row in cgroup_rows if "::" in row]
    argv = tuple(
        value.decode("utf-8", errors="strict")
        for value in (root / "cmdline").read_bytes().split(b"\0") if value
    )
    if len(uid_rows) != 1 or len(unified) != 1 or not argv:
        raise RuntimeError("manager process identity is malformed")
    uid = int(uid_rows[0].split()[1])
    return {
        "pid": pid, "starttime_ticks": starttime, "uid": uid,
        "control_group": unified[0], "argv": list(argv),
    }


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
        not stat.S_ISDIR(runtime_stat.st_mode)
        or runtime.is_symlink()
        or runtime_stat.st_uid != uid
        or stat.S_IMODE(runtime_stat.st_mode) != 0o700
        or not stat.S_ISSOCK(bus_stat.st_mode)
        or bus.is_symlink()
        or bus_stat.st_uid != uid
    ):
        raise PermissionError("user manager endpoint is unsafe")
    cgroup = f"/user.slice/user-{uid}.slice/user@{uid}.service/init.scope"
    members = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "cgroup.procs"
    pids = [int(row) for row in members.read_text(encoding="ascii").splitlines()]
    candidates = []
    for pid in pids:
        identity = _read_process(pid)
        argv = identity["argv"]
        if (
            identity["uid"] == uid
            and identity["control_group"] == cgroup
            and Path(str(argv[0])).name == "systemd"
            and "--user" in argv[1:]
        ):
            candidates.append(identity)
    if len(candidates) != 1:
        raise RuntimeError("user manager identity is ambiguous")
    identity = dict(candidates[0])
    identity.pop("argv")
    return {
        "boot_id": boot_id,
        "identity": identity,
        "endpoint": {
            "uid": uid, "runtime_directory": str(runtime),
            "runtime_directory_device": runtime_stat.st_dev,
            "runtime_directory_inode": runtime_stat.st_ino,
            "bus_path": str(bus), "bus_device": bus_stat.st_dev,
            "bus_inode": bus_stat.st_ino,
        },
    }


def _validate_manager_generation(value: Mapping[str, object]) -> None:
    uid = os.getuid()
    cgroup = f"/user.slice/user-{uid}.slice/user@{uid}.service/init.scope"
    if set(value) != {"boot_id", "identity", "endpoint"}:
        raise PermissionError("manager generation keys are not exact")
    identity = value.get("identity")
    endpoint = value.get("endpoint")
    if not isinstance(identity, Mapping) or not isinstance(endpoint, Mapping):
        raise PermissionError("manager generation objects are malformed")
    if (
        not isinstance(value.get("boot_id"), str)
        or _BOOT_ID.fullmatch(str(value["boot_id"])) is None
        or set(identity) != {
            "pid", "starttime_ticks", "uid", "control_group"
        }
        or set(endpoint) != {
            "uid", "runtime_directory", "runtime_directory_device",
            "runtime_directory_inode", "bus_path", "bus_device", "bus_inode",
        }
        or isinstance(identity.get("uid"), bool)
        or not isinstance(identity.get("uid"), int)
        or identity.get("uid") != uid
        or identity.get("control_group") != cgroup
        or isinstance(endpoint.get("uid"), bool)
        or not isinstance(endpoint.get("uid"), int)
        or endpoint.get("uid") != uid
        or endpoint.get("runtime_directory") != f"/run/user/{uid}"
        or endpoint.get("bus_path") != f"/run/user/{uid}/bus"
    ):
        raise PermissionError("manager generation identity is not exact")
    numeric = (
        identity.get("pid"), identity.get("starttime_ticks"),
        endpoint.get("runtime_directory_device"),
        endpoint.get("runtime_directory_inode"), endpoint.get("bus_device"),
        endpoint.get("bus_inode"),
    )
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
           for item in numeric):
        raise PermissionError("manager generation numeric identity is invalid")


def _file_binding(
    path: Path, *, allow_symlink: bool = False,
) -> dict[str, object]:
    supplied = path.absolute()
    if any(character.isspace() for character in str(supplied)):
        raise ValueError("bound paths must be whitespace-free")
    raw, observed = _stable_read_file(supplied)
    if (
        allow_symlink and observed["path_is_symlink"] is not False
    ):
        raise PermissionError("bound file symlinks are not supported")
    if int(observed["nlink"]) != 1:
        raise PermissionError("bound file identity is unsafe")
    del raw
    binding = dict(observed)
    binding.pop("nlink")
    return binding


def _read_file_binding(
    path: Path, *, allow_symlink: bool = False,
) -> tuple[bytes, dict[str, object]]:
    supplied = path.absolute()
    if any(character.isspace() for character in str(supplied)):
        raise ValueError("bound paths must be whitespace-free")
    raw, observed = _stable_read_file(supplied)
    if (
        allow_symlink and observed["path_is_symlink"] is not False
    ):
        raise PermissionError("bound file symlinks are not supported")
    if int(observed["nlink"]) != 1:
        raise PermissionError("bound file identity is unsafe")
    binding = dict(observed)
    binding.pop("nlink")
    return raw, binding


def _validate_python_binding(binding: Mapping[str, object]) -> None:
    if (
        binding.get("path") != str(PYTHON_PATH)
        or binding.get("resolved_path") != str(PYTHON_PATH)
        or binding.get("path_is_symlink") is not False
        or isinstance(binding.get("owner_uid"), bool)
        or not isinstance(binding.get("owner_uid"), int)
        or binding.get("owner_uid") != 0
        or isinstance(binding.get("mode"), bool)
        or not isinstance(binding.get("mode"), int)
        or int(binding["mode"]) & 0o022
    ):
        raise PermissionError("python binding is not the fixed trusted interpreter")


def _validate_binding(binding: Mapping[str, object]) -> bytes:
    if set(binding) != _FILE_BINDING_KEYS:
        raise PermissionError("bound file schema changed")
    numeric = (
        binding.get("device"), binding.get("inode"),
        binding.get("owner_uid"), binding.get("mode"),
    )
    if (
        not isinstance(binding.get("path"), str)
        or not isinstance(binding.get("resolved_path"), str)
        or binding.get("path_is_symlink") is not False
        or not isinstance(binding.get("file_sha256"), str)
        or _SHA.fullmatch(str(binding["file_sha256"])) is None
        or any(isinstance(item, bool) or not isinstance(item, int)
               for item in numeric)
    ):
        raise PermissionError("bound file field types changed")
    raw, observed = _read_file_binding(
        Path(binding["path"]),
        allow_symlink=binding["path_is_symlink"],
    )
    if not _deep_exact_equal(observed, dict(binding)):
        raise PermissionError("bound file changed")
    return raw


def render_fragment(
    template_text: str, *, python_path: Path, supervisor_path: Path,
    runtime_spec_path: Path,
) -> str:
    if not template_text.endswith("\n") or "[Install]" in template_text:
        raise ValueError("actual template must be newline-terminated and static")
    replacements = {
        "@@PYTHON_PATH@@": str(python_path.absolute()),
        "@@SUPERVISOR_PATH@@": str(supervisor_path.absolute()),
        "@@RUNTIME_SPEC_PATH@@": str(runtime_spec_path.absolute()),
    }
    if any(character.isspace() for value in replacements.values() for character in value):
        raise ValueError("rendered supervisor paths must be whitespace-free")
    if any(template_text.count(token) != 4 for token in _PLACEHOLDERS):
        raise ValueError("actual template placeholder closure changed")
    rendered = template_text
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if "@@" in rendered or "[Install]" in rendered:
        raise ValueError("actual template was not completely rendered")
    _validate_rendered_directives(
        rendered, python_path=python_path, supervisor_path=supervisor_path,
        runtime_spec_path=runtime_spec_path,
    )
    return rendered


def _directives(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in text.splitlines():
        if not row or row.startswith(("#", "[")):
            continue
        key, separator, value = row.partition("=")
        if separator != "=":
            raise ValueError("unit directive is malformed")
        result.setdefault(key, []).append(value)
    return result


def _expected_exec(
    *, python_path: Path, supervisor_path: Path, runtime_spec_path: Path,
) -> dict[str, list[str]]:
    result = {}
    for directive, mode in _EXEC_MODES.items():
        result[directive] = [
            str(python_path.absolute()), "-I", "-S", "-B", "-u",
            str(supervisor_path.absolute()), mode, "--spec",
            str(runtime_spec_path.absolute()),
        ]
    return result


def _validate_rendered_directives(
    text: str, *, python_path: Path, supervisor_path: Path,
    runtime_spec_path: Path,
) -> None:
    directives = _directives(text)
    expected_exec = _expected_exec(
        python_path=python_path, supervisor_path=supervisor_path,
        runtime_spec_path=runtime_spec_path,
    )
    for name, argv in expected_exec.items():
        if directives.get(name) != [" ".join(argv)]:
            raise PermissionError(f"{name} is not the exact supervisor argv")
    exact = {
        "Description": [
            "CURE-Lite v24 GCR-PACRE D_R structural fresh attempt r2"
        ],
        "Type": ["exec"], "ExitType": ["main"], "UMask": ["0077"],
        "WorkingDirectory": ["/home/md0/ly/cure_lite"],
        "Environment": [
            "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        ],
        "Restart": ["no"], "RestartSec": ["0"],
        "StartLimitIntervalSec": ["infinity"], "StartLimitBurst": ["1"],
        "KillMode": ["mixed"], "KillSignal": ["SIGTERM"],
        "SendSIGKILL": ["yes"], "WatchdogSec": ["0"],
        "RemainAfterExit": ["no"], "StandardInput": ["null"],
        "StandardOutput": ["journal"], "StandardError": ["journal"],
        "TimeoutStartSec": ["5min"], "TimeoutStopSec": ["10min"],
        "RuntimeMaxSec": ["infinity"], "OOMPolicy": ["kill"],
        "SyslogIdentifier": ["cure-lite-v24-gcr-pacre-dr-r2"],
        "SuccessExitStatus": ["0"],
    }
    if any(directives.get(key) != value for key, value in exact.items()):
        raise PermissionError("actual unit static directive closure changed")
    if set(directives) != set(exact) | set(_EXEC_MODES):
        raise PermissionError("actual unit has an unauthorized directive")
    if any(key in directives for key in ("WantedBy", "RequiredBy", "Alias")):
        raise PermissionError("actual unit is installable or aliased")


def _expected_static_shadow(fragment_path: Path) -> dict[str, str]:
    return {
        "Id": ACTUAL_UNIT,
        "Description": "CURE-Lite v24 GCR-PACRE D_R structural fresh attempt r2",
        "LoadState": "loaded", "ActiveState": "inactive", "SubState": "dead",
        "UnitFileState": "static", "FragmentPath": str(fragment_path.absolute()),
        "DropInPaths": "", "Transient": "no", "Restart": "no",
        "RestartUSec": "0", "NRestarts": "0", "NeedDaemonReload": "no",
        "Type": "exec", "ExitType": "main", "KillMode": "mixed",
        "KillSignal": "15", "SendSIGKILL": "yes",
        "TimeoutStartUSec": "5min", "TimeoutStopUSec": "10min",
        "RuntimeMaxUSec": "infinity", "WatchdogUSec": "disabled",
        "OOMPolicy": "kill", "RemainAfterExit": "no",
        "StandardInput": "null", "StandardOutput": "journal",
        "StandardError": "journal", "StartLimitIntervalUSec": "infinity",
        "StartLimitBurst": "1", "UMask": "0077",
        "Environment": (
            "PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_ORDER=PCI_BUS_ID"
        ),
        "UnsetEnvironment": "", "WorkingDirectory": "/home/md0/ly/cure_lite",
        "TriggeredBy": "", "Triggers": "", "WantedBy": "",
        "RequiredBy": "", "PartOf": "",
        "SyslogIdentifier": "cure-lite-v24-gcr-pacre-dr-r2",
        "SuccessExitStatus": "0 0",
    }


def _private_parent(path: Path) -> Path:
    target = path.absolute()
    parent_fd, _ = _open_stable_parent(target, private=True)
    os.close(parent_fd)
    return target


def _write_create_once_json_bound(
    path: Path, body: Mapping[str, object], *, fingerprint_field: str,
) -> tuple[dict[str, object], dict[str, object]]:
    target = Path(path).absolute()
    if fingerprint_field in body:
        raise ValueError("body already contains fingerprint")
    materialized = dict(body)
    payload = {
        **materialized, fingerprint_field: stable_fingerprint(materialized)
    }
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    parent_fd, parent_before = _open_stable_parent(target, private=True)
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o444,
            dir_fd=parent_fd,
        )
        os.fchmod(descriptor, 0o444)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("zero-byte evidence write")
            offset += written
        opened = os.fstat(descriptor)
        linked = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _stat_snapshot(opened) != _stat_snapshot(linked)
            or opened.st_uid != os.getuid() or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o444
        ):
            raise PermissionError("created evidence identity is unsafe")
        os.fsync(descriptor)
        if os.pread(descriptor, len(encoded) + 1, 0) != encoded:
            raise RuntimeError("created evidence failed descriptor readback")
        after = os.fstat(descriptor)
        linked_after = os.stat(
            target.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            _stat_snapshot(after) != _stat_snapshot(opened)
            or _stat_snapshot(linked_after) != _stat_snapshot(after)
        ):
            raise PermissionError("created evidence changed during readback")
        os.fsync(descriptor)
        os.fsync(parent_fd)
        _verify_parent_generation(target.parent, parent_fd, parent_before)
        identity = {
            "path": str(target),
            "file_sha256": hashlib.sha256(encoded).hexdigest(),
            "device": after.st_dev, "inode": after.st_ino,
            "owner_uid": after.st_uid, "mode": stat.S_IMODE(after.st_mode),
            "nlink": after.st_nlink,
        }
        return payload, identity
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def write_create_once_json(
    path: Path, body: Mapping[str, object], *, fingerprint_field: str,
) -> dict[str, object]:
    return _write_create_once_json_bound(
        path, body, fingerprint_field=fingerprint_field,
    )[0]


def _load_sealed_json_bound(
    path: Path, fingerprint_field: str,
) -> tuple[dict[str, object], dict[str, object]]:
    target = Path(path).absolute()
    raw, identity = _stable_read_file(target, private_parent=True)
    if (
        identity["owner_uid"] != os.getuid()
        or identity["nlink"] != 1
        or identity["mode"] != 0o444
    ):
        raise PermissionError("input evidence is not sealed")
    payload = json.loads(raw.decode("utf-8"))
    if (
        not isinstance(payload, dict)
        or raw != (canonical_json(payload) + "\n").encode("utf-8")
    ):
        raise ValueError("input evidence is not canonical JSON")
    body = dict(payload)
    fingerprint = body.pop(fingerprint_field, None)
    if not isinstance(fingerprint, str) or _SHA.fullmatch(fingerprint) is None:
        raise PermissionError("input evidence fingerprint is malformed")
    if fingerprint != stable_fingerprint(body):
        raise PermissionError("input evidence fingerprint is invalid")
    return payload, identity


def load_sealed_json(path: Path, fingerprint_field: str) -> dict[str, object]:
    return _load_sealed_json_bound(path, fingerprint_field)[0]


def create_authorization(
    authorization_path: Path, *, template_path: Path, python_path: Path,
    supervisor_path: Path, runtime_spec_path: Path,
    authorization_basis: str, instruction_id: str,
    validity_seconds: int = 300, runner: CommandRunner = subprocess.run,
    manager_reader: ManagerReader = collect_manager_generation,
) -> dict[str, object]:
    if instruction_id != INSTRUCTION_ID:
        raise PermissionError("modify-then-run instruction is not bound")
    paths = (
        authorization_path, template_path, python_path, supervisor_path,
        runtime_spec_path,
    )
    if any(not Path(path).is_absolute() for path in paths):
        raise ValueError("all authorization paths must be absolute")
    if str(python_path) != str(PYTHON_PATH):
        raise PermissionError("python path is not /usr/bin/python3.12")
    if (
        not isinstance(validity_seconds, int)
        or isinstance(validity_seconds, bool)
        or authorization_basis != AUTHORIZATION_BASIS
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("authorization basis or validity is invalid")
    policy = freeze_unit_path_policy(runner=runner)
    _require_no_shadow(query_shadow(runner=runner))
    template_raw, template_binding = _read_file_binding(template_path)
    executable_bindings = {
        "realization_tool": _file_binding(Path(__file__).resolve()),
        "python": _file_binding(python_path),
        "supervisor": _file_binding(supervisor_path),
        "systemd_path": _file_binding(Path(SYSTEMD_PATH)),
        "systemd_analyze": _file_binding(Path(SYSTEMD_ANALYZE)),
        "systemctl": _file_binding(Path(SYSTEMCTL)),
    }
    _validate_python_binding(executable_bindings["python"])
    if os.path.lexists(runtime_spec_path):
        raise PermissionError("future runtime spec path must be absent")
    spec_parent = runtime_spec_path.parent
    spec_parent_identity = _path_row(spec_parent)
    if spec_parent_identity.get("exists") is not True:
        raise PermissionError("future runtime spec parent must exist")
    runtime_binding = {
        "kind": "future-absent-runtime-spec-v2",
        "runtime_spec_path": str(runtime_spec_path.absolute()),
        "runtime_spec_parent_identity": spec_parent_identity,
        "absent_at_authorization": True,
        "required_schema": SUPERVISOR_SPEC_SCHEMA,
    }
    try:
        template_text = template_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("actual template is not strict UTF-8") from error
    rendered = render_fragment(
        template_text,
        python_path=python_path, supervisor_path=supervisor_path,
        runtime_spec_path=runtime_spec_path,
    )
    rendered_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    manager_generation = manager_reader()
    _validate_manager_generation(manager_generation)
    expected_shadow = _expected_static_shadow(
        Path(str(policy["runtime_directory"])) / ACTUAL_UNIT
    )
    issued = datetime.now(timezone.utc)
    body: dict[str, object] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "candidate": CANDIDATE, "stage_id": STAGE_ID,
        "attempt_id": ATTEMPT_ID, "unit_name": ACTUAL_UNIT,
        "instruction_id": instruction_id,
        "authorization_basis": authorization_basis,
        "authorized_uid": os.getuid(), "created_at_utc": _utc_now(),
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (
            issued + timedelta(seconds=validity_seconds)
        ).isoformat().replace("+00:00", "Z"),
        "actions": list(_ACTIONS), "persistent_install_authorized": False,
        "enable_authorized": False, "start_authorized": False,
        "remove_authorized": False,
        "unit_directory": policy["runtime_directory"],
        "unit_path_policy": policy,
        "manager_generation": manager_generation,
        "template_binding": template_binding,
        "rendered_fragment": {"utf8_text": rendered, "sha256": rendered_sha},
        "runtime_spec_binding": runtime_binding,
        "executable_bindings": executable_bindings,
        "expected_static_shadow": expected_shadow,
        "payload_authority": "none", "D_R_payload_accessed": False,
        "D_V_payload_accessed": False, "D_T_payload_accessed": False,
    }
    # Bind-render-bind: a same-byte inode substitution after the first read
    # must not be sealed into an authorization as if it were the same source.
    if not _deep_exact_equal(_file_binding(template_path), template_binding):
        raise PermissionError("actual template changed before authorization commit")
    for name, binding in executable_bindings.items():
        if not _deep_exact_equal(
            _file_binding(Path(str(binding["path"]))), binding,
        ):
            raise PermissionError(
                f"actual realization executable changed before commit:{name}"
            )
    return write_create_once_json(
        authorization_path, body, fingerprint_field="authorization_fingerprint"
    )


def validate_authorization(
    path: Path, *, runner: CommandRunner, manager_reader: ManagerReader,
    return_identity: bool = False,
) -> dict[str, object] | tuple[dict[str, object], dict[str, object]]:
    authorization, authorization_identity = _load_sealed_json_bound(
        path, "authorization_fingerprint",
    )
    if set(authorization) != _AUTH_KEYS:
        raise PermissionError("actual realization authorization keys are not exact")
    if (
        authorization["schema_version"] != AUTHORIZATION_SCHEMA
        or authorization["candidate"] != CANDIDATE
        or authorization["stage_id"] != STAGE_ID
        or authorization["attempt_id"] != ATTEMPT_ID
        or authorization["unit_name"] != ACTUAL_UNIT
        or authorization["instruction_id"] != INSTRUCTION_ID
        or authorization["authorization_basis"] != AUTHORIZATION_BASIS
        or isinstance(authorization["authorized_uid"], bool)
        or not isinstance(authorization["authorized_uid"], int)
        or authorization["authorized_uid"] != os.getuid()
        or not _deep_exact_equal(authorization["actions"], _ACTIONS)
        or authorization["persistent_install_authorized"] is not False
        or authorization["enable_authorized"] is not False
        or authorization["start_authorized"] is not False
        or authorization["remove_authorized"] is not False
        or authorization["payload_authority"] != "none"
        or authorization["D_R_payload_accessed"] is not False
        or authorization["D_V_payload_accessed"] is not False
        or authorization["D_T_payload_accessed"] is not False
    ):
        raise PermissionError("actual realization authorization is not exact")
    issued = datetime.fromisoformat(
        str(authorization["issued_at_utc"]).replace("Z", "+00:00")
    )
    expires = datetime.fromisoformat(
        str(authorization["expires_at_utc"]).replace("Z", "+00:00")
    )
    now = datetime.now(timezone.utc)
    if (
        issued.tzinfo is None or expires.tzinfo is None
        or not issued <= now <= expires or expires - issued > timedelta(minutes=5)
    ):
        raise PermissionError("actual realization authorization is stale")
    executable_bindings = authorization["executable_bindings"]
    if not isinstance(executable_bindings, Mapping) or set(executable_bindings) != {
        "realization_tool", "python", "supervisor", "systemd_path",
        "systemd_analyze", "systemctl",
    }:
        raise PermissionError("executable binding keys are not exact")
    for binding in executable_bindings.values():
        if not isinstance(binding, Mapping):
            raise PermissionError("an executable binding is malformed")
        _validate_binding(binding)
    _validate_python_binding(executable_bindings["python"])
    template_raw = _validate_binding(authorization["template_binding"])
    runtime_binding = authorization["runtime_spec_binding"]
    if not isinstance(runtime_binding, Mapping) or set(runtime_binding) != {
        "kind", "runtime_spec_path", "runtime_spec_parent_identity",
        "absent_at_authorization", "required_schema",
    }:
        raise PermissionError("runtime spec binding keys are not exact")
    if (
        runtime_binding["kind"] != "future-absent-runtime-spec-v2"
        or runtime_binding["required_schema"] != SUPERVISOR_SPEC_SCHEMA
        or not Path(str(runtime_binding["runtime_spec_path"])).is_absolute()
        or runtime_binding["absent_at_authorization"] is not True
    ):
        raise PermissionError("runtime spec binding is malformed")
    runtime_spec_path = Path(str(runtime_binding["runtime_spec_path"]))
    if (
        not _deep_exact_equal(
            _path_row(runtime_spec_path.parent),
            runtime_binding["runtime_spec_parent_identity"],
        )
        or os.path.lexists(runtime_spec_path)
    ):
        raise PermissionError("future runtime spec path state changed")
    _validate_manager_generation(authorization["manager_generation"])
    current_generation = manager_reader()
    _validate_manager_generation(current_generation)
    if not _deep_exact_equal(
        current_generation, authorization["manager_generation"],
    ):
        raise PermissionError("user manager generation changed")
    _revalidate_unit_path_policy(authorization["unit_path_policy"], runner=runner)
    _require_no_shadow(query_shadow(runner=runner))
    template_path = Path(authorization["template_binding"]["path"])
    if template_path != Path(str(authorization["template_binding"]["path"])):
        raise PermissionError("template binding path changed")
    try:
        template_text = template_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("actual template is not strict UTF-8") from error
    rendered = render_fragment(
        template_text,
        python_path=Path(authorization["executable_bindings"]["python"]["path"]),
        supervisor_path=Path(
            authorization["executable_bindings"]["supervisor"]["path"]
        ),
        runtime_spec_path=Path(runtime_binding["runtime_spec_path"]),
    )
    if not _deep_exact_equal(
        authorization["rendered_fragment"],
        {
            "utf8_text": rendered,
            "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        },
    ):
        raise PermissionError("rendered actual fragment binding changed")
    expected_shadow = _expected_static_shadow(
        Path(str(authorization["unit_directory"])) / ACTUAL_UNIT
    )
    if not _deep_exact_equal(
        authorization["expected_static_shadow"], expected_shadow,
    ):
        raise PermissionError("authorized full static shadow changed")
    if return_identity:
        return authorization, authorization_identity
    return authorization


def _install_fragment(
    unit_directory: Path, rendered: bytes,
) -> dict[str, object]:
    fragment = unit_directory / ACTUAL_UNIT
    directory_fd, directory_metadata = _open_stable_parent(fragment)
    if (
        directory_metadata.st_uid != os.getuid()
        or stat.S_IMODE(directory_metadata.st_mode) & 0o022
    ):
        os.close(directory_fd)
        raise PermissionError("authorized runtime unit directory changed")
    descriptor = -1
    try:
        descriptor = os.open(
            ACTUAL_UNIT,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(rendered):
            written = os.write(descriptor, rendered[offset:])
            if written <= 0:
                raise OSError("zero-byte fragment write")
            offset += written
        opened = os.fstat(descriptor)
        linked = os.stat(ACTUAL_UNIT, dir_fd=directory_fd, follow_symlinks=False)
        if (
            _stat_snapshot(opened) != _stat_snapshot(linked)
            or opened.st_uid != os.getuid() or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise PermissionError("installed actual fragment identity is unsafe")
        os.fsync(descriptor)
        if os.pread(descriptor, len(rendered) + 1, 0) != rendered:
            raise RuntimeError("installed actual fragment descriptor readback changed")
        after = os.fstat(descriptor)
        linked_after = os.stat(
            ACTUAL_UNIT, dir_fd=directory_fd, follow_symlinks=False,
        )
        if (
            _stat_snapshot(after) != _stat_snapshot(opened)
            or _stat_snapshot(linked_after) != _stat_snapshot(after)
        ):
            raise PermissionError("installed actual fragment changed during readback")
        os.fsync(directory_fd)
        _verify_parent_generation(
            unit_directory, directory_fd, directory_metadata,
        )
        return {
            "path": str(fragment),
            "file_sha256": hashlib.sha256(rendered).hexdigest(),
            "device": after.st_dev, "inode": after.st_ino,
            "owner_uid": after.st_uid, "mode": stat.S_IMODE(after.st_mode),
            "nlink": after.st_nlink,
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)


def _normalize_exec(raw: str) -> dict[str, object]:
    pairs = re.findall(
        r"(?:^|[;{])\s*(path|argv\[\]|ignore_errors)=([^;}]*)(?=;|})", raw
    )
    if len(pairs) != 3 or len({key for key, _ in pairs}) != 3:
        raise PermissionError("systemd execution identity is ambiguous")
    values = {key: value.strip() for key, value in pairs}
    return {
        "path": values["path"], "argv": values["argv[]"].split(),
        "ignore_errors": values["ignore_errors"],
    }


def validate_installed_shadow(
    shadow: Mapping[str, str], *, fragment_identity: Mapping[str, object],
    authorization: Mapping[str, object],
) -> dict[str, object]:
    fragment_path = str(fragment_identity["path"])
    expected = _expected_static_shadow(Path(fragment_path))
    if not _deep_exact_equal(
        authorization["expected_static_shadow"], expected,
    ):
        raise PermissionError("authorized installed shadow contract changed")
    if set(shadow) != set(_SHADOW_PROPERTIES):
        raise PermissionError("installed actual unit property closure changed")
    if any(shadow.get(key) != value for key, value in expected.items()):
        raise PermissionError("installed actual unit shadow is not exact static")
    runtime_binding = authorization["runtime_spec_binding"]
    expected_exec = _expected_exec(
        python_path=Path(authorization["executable_bindings"]["python"]["path"]),
        supervisor_path=Path(
            authorization["executable_bindings"]["supervisor"]["path"]
        ),
        runtime_spec_path=Path(runtime_binding["runtime_spec_path"]),
    )
    normalized = {name: _normalize_exec(shadow[name]) for name in _EXEC_MODES}
    for name, argv in expected_exec.items():
        if not _deep_exact_equal(
            normalized[name],
            {"path": argv[0], "argv": argv, "ignore_errors": "no"},
        ):
            raise PermissionError(f"installed {name} argv changed")
    _, observed = _stable_read_file(fragment_path)
    live_fragment_identity = {
        key: observed[key]
        for key in (
            "path", "file_sha256", "device", "inode", "owner_uid", "mode",
            "nlink",
        )
    }
    if (
        not _deep_exact_equal(live_fragment_identity, dict(fragment_identity))
        or observed["owner_uid"] != os.getuid()
        or observed["nlink"] != 1
        or observed["mode"] != 0o600
    ):
        raise PermissionError("installed actual fragment identity drifted")
    return {**dict(shadow), **{f"normalized_{k}": v for k, v in normalized.items()}}


def realize_actual_unit(
    authorization_path: Path, *, receipt_path: Path, terminal_path: Path,
    runner: CommandRunner = subprocess.run,
    manager_reader: ManagerReader = collect_manager_generation,
) -> dict[str, object]:
    completed_actions: list[str] = []
    fragment_identity: dict[str, object] | None = None
    expected_fragment_path: Path | None = None
    authorization_fingerprint: str | None = None
    post_reload_policy: dict[str, object] | None = None
    daemon_reload_attempted = False
    try:
        authorization, authorization_identity = validate_authorization(
            authorization_path, runner=runner, manager_reader=manager_reader,
            return_identity=True,
        )
        authorization_fingerprint = str(
            authorization["authorization_fingerprint"]
        )
        rendered = authorization["rendered_fragment"]["utf8_text"].encode("utf-8")
        expected_fragment_path = (
            Path(str(authorization["unit_directory"])) / ACTUAL_UNIT
        )
        fragment_identity = _install_fragment(
            Path(authorization["unit_directory"]), rendered
        )
        if fragment_identity["file_sha256"] != authorization[
            "rendered_fragment"
        ]["sha256"]:
            raise PermissionError("installed fragment SHA differs from authorization")
        completed_actions.append(_ACTIONS[0])
        _revalidate_installed_path_policy(
            authorization["unit_path_policy"],
            fragment=Path(str(fragment_identity["path"])), runner=runner,
        )
        current_generation = manager_reader()
        _validate_manager_generation(current_generation)
        if not _deep_exact_equal(
            current_generation, authorization["manager_generation"],
        ):
            raise PermissionError("manager changed before daemon-reload")
        daemon_reload_attempted = True
        completed = _run((SYSTEMCTL, "--user", "daemon-reload"), runner=runner)
        if completed.returncode != 0:
            raise RuntimeError("user manager daemon-reload failed")
        observed_post_reload_policy = _observe_unit_path_policy(
            runner=runner,
            allowed_fragment=Path(str(fragment_identity["path"])).absolute(),
        )
        post_reload_policy = _validate_daemon_reload_path_policy_transition(
            authorization["unit_path_policy"],
            observed_post_reload_policy,
            runtime_directory=Path(str(authorization["unit_directory"])),
            authorized_uid=authorization["authorized_uid"],
        )
        completed_actions.append(_ACTIONS[1])
        current_generation = manager_reader()
        _validate_manager_generation(current_generation)
        if not _deep_exact_equal(
            current_generation, authorization["manager_generation"],
        ):
            raise PermissionError("manager changed after daemon-reload")
        _revalidate_installed_path_policy(
            post_reload_policy,
            fragment=Path(str(fragment_identity["path"])), runner=runner,
        )
        shadow = validate_installed_shadow(
            query_shadow(runner=runner), fragment_identity=fragment_identity,
            authorization=authorization,
        )
        completed_actions.append(_ACTIONS[2])
        _revalidate_installed_path_policy(
            post_reload_policy,
            fragment=Path(str(fragment_identity["path"])), runner=runner,
        )
        current_generation = manager_reader()
        _validate_manager_generation(current_generation)
        if not _deep_exact_equal(
            current_generation, authorization["manager_generation"],
        ):
            raise PermissionError("manager changed after static-shadow validation")
        if os.path.lexists(Path(str(
            authorization["runtime_spec_binding"]["runtime_spec_path"]
        ))):
            raise PermissionError("runtime spec appeared during realization")
        _, live_authorization_identity = _stable_read_file(
            authorization_path, private_parent=True,
        )
        if not _deep_exact_equal(
            live_authorization_identity, authorization_identity,
        ):
            raise PermissionError(
                "actual realization authorization identity was replaced"
            )
        if (
            not _deep_exact_equal(
                _file_binding(Path(str(
                    authorization["template_binding"]["path"]
                ))),
                authorization["template_binding"],
            )
        ):
            raise PermissionError("actual template changed before receipt commit")
        for name, binding in authorization["executable_bindings"].items():
            if not _deep_exact_equal(
                _file_binding(Path(str(binding["path"]))), binding,
            ):
                raise PermissionError(
                    f"actual realization executable changed before receipt:{name}"
                )
        _revalidate_installed_path_policy(
            post_reload_policy,
            fragment=Path(str(fragment_identity["path"])), runner=runner,
        )
        current_generation = manager_reader()
        _validate_manager_generation(current_generation)
        if not _deep_exact_equal(
            current_generation, authorization["manager_generation"],
        ):
            raise PermissionError("manager changed before receipt commit")
        _, live_fragment = _stable_read_file(
            Path(str(fragment_identity["path"])),
        )
        if not _deep_exact_equal(
            {
                key: live_fragment[key]
                for key in (
                    "file_sha256", "device", "inode", "owner_uid", "mode",
                    "nlink",
                )
            },
            {
                key: fragment_identity[key]
                for key in (
                    "file_sha256", "device", "inode", "owner_uid", "mode",
                    "nlink",
                )
            },
        ):
            raise PermissionError(
                "actual fragment changed before realization receipt commit"
            )
        if os.path.lexists(Path(str(
            authorization["runtime_spec_binding"]["runtime_spec_path"]
        ))):
            raise PermissionError("runtime spec appeared before receipt commit")
        body = {
            "schema_version": RECEIPT_SCHEMA, "candidate": CANDIDATE,
            "stage_id": STAGE_ID, "attempt_id": ATTEMPT_ID,
            "unit_name": ACTUAL_UNIT, "created_at_utc": _utc_now(),
            "authorization_path": str(authorization_path.absolute()),
            "authorization_file_sha256": authorization_identity["file_sha256"],
            "authorization_fingerprint": authorization_fingerprint,
            "instruction_id": INSTRUCTION_ID,
            "manager_generation": authorization["manager_generation"],
            "unit_path_policy": post_reload_policy,
            "template_binding": authorization["template_binding"],
            "rendered_fragment": authorization["rendered_fragment"],
            "runtime_spec_binding": authorization["runtime_spec_binding"],
            "expected_future_runtime_spec_path": authorization[
                "runtime_spec_binding"
            ]["runtime_spec_path"],
            "runtime_spec_absent_at_receipt": True,
            "executable_bindings": authorization["executable_bindings"],
            "fragment_identity": fragment_identity,
            "full_static_shadow": shadow,
            "completed_actions": completed_actions,
            "static": True, "enabled": False, "started": False,
            "removed": False, "passed": True, "payload_authority": "none",
            "D_R_payload_accessed": False, "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        return write_create_once_json(
            receipt_path, body, fingerprint_field="receipt_fingerprint"
        )
    except BaseException as error:
        body = {
            "schema_version": TERMINAL_SCHEMA, "candidate": CANDIDATE,
            "stage_id": STAGE_ID, "attempt_id": ATTEMPT_ID,
            "unit_name": ACTUAL_UNIT, "created_at_utc": _utc_now(),
            "authorization_path": str(authorization_path.absolute()),
            "authorization_fingerprint": authorization_fingerprint,
            "completed_actions": completed_actions,
            "fragment_identity": fragment_identity,
            "fragment_may_remain": (
                fragment_identity is not None
                or (
                    expected_fragment_path is not None
                    and os.path.lexists(expected_fragment_path)
                )
            ),
            "automatic_removal_performed": False,
            "daemon_reload_attempted": daemon_reload_attempted,
            "enable_attempted": False, "start_attempted": False,
            "remove_attempted": False,
            "error_type": type(error).__name__, "error_message": str(error),
            "passed": False, "payload_authority": "none",
            "D_R_payload_accessed": False, "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        try:
            write_create_once_json(
                terminal_path, body, fingerprint_field="terminal_fingerprint"
            )
        finally:
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    authorize = sub.add_parser("authorize")
    authorize.add_argument("--output", type=Path, required=True)
    authorize.add_argument("--template", type=Path, required=True)
    authorize.add_argument("--python", type=Path, required=True)
    authorize.add_argument("--supervisor", type=Path, required=True)
    authorize.add_argument("--runtime-spec", type=Path, required=True)
    authorize.add_argument("--authorization-basis", required=True)
    authorize.add_argument("--instruction-id", required=True)
    authorize.add_argument("--validity-seconds", type=int, default=300)
    apply = sub.add_parser("apply")
    apply.add_argument("--authorization", type=Path, required=True)
    apply.add_argument("--receipt", type=Path, required=True)
    apply.add_argument("--terminal-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "authorize":
        create_authorization(
            args.output, template_path=args.template, python_path=args.python,
            supervisor_path=args.supervisor, runtime_spec_path=args.runtime_spec,
            authorization_basis=args.authorization_basis,
            instruction_id=args.instruction_id,
            validity_seconds=args.validity_seconds,
        )
        return 0
    realize_actual_unit(
        args.authorization, receipt_path=args.receipt,
        terminal_path=args.terminal_receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
