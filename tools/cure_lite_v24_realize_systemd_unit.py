"""Fail-closed helpers for realizing one authorized user-systemd dummy unit.

This module deliberately does not know about the real r2 unit.  It accepts only
the dedicated integration-unit namespace and exposes mutation primitives that
require an already verified plan plus an explicit execution flag.  The
integration orchestrator is responsible for authorization and receipts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
import time
from typing import Callable, Mapping, Sequence


SYSTEMD_PATH = "/usr/bin/systemd-path"
SYSTEMD_ANALYZE = "/usr/bin/systemd-analyze"
SYSTEMCTL_PATH = "/usr/bin/systemctl"
INTEGRATION_UNIT_PREFIX = "cure-lite-v24-supervisor-integration-"
_UNIT_NAME = re.compile(
    rf"{re.escape(INTEGRATION_UNIT_PREFIX)}"
    r"[a-z0-9](?:[a-z0-9-]{6,61}[a-z0-9])?-[0-9a-f]{16}\.service"
)
_STATIC_PROPERTIES = (
    "LoadState",
    "UnitFileState",
    "ActiveState",
    "SubState",
    "FragmentPath",
    "DropInPaths",
    "Transient",
    "Restart",
    "NRestarts",
    "NeedDaemonReload",
)


@dataclass(frozen=True)
class RealizationPlan:
    unit_name: str
    unit_directory: Path
    fragment_path: Path
    fragment_bytes: bytes
    fragment_sha256: str
    owner_uid: int
    execute_authorized: bool
    removal_authorized: bool


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
UnitQuery = Callable[[str], Mapping[str, str]]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fd_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def validate_integration_unit_name(unit_name: str) -> None:
    if not isinstance(unit_name, str) or _UNIT_NAME.fullmatch(unit_name) is None:
        raise PermissionError(
            "only unique CURE-Lite v24 integration unit names are permitted"
        )
    if unit_name == "cure-lite-v24-gcr-pacre-dr-r2.service":
        raise PermissionError("the actual r2 unit is never an integration unit")


def _fixed_command_environment() -> dict[str, str]:
    uid = os.getuid()
    runtime = f"/run/user/{uid}"
    return {
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime}/bus",
        "HOME": pwd.getpwuid(uid).pw_dir,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "SYSTEMD_COLORS": "0",
        "XDG_RUNTIME_DIR": runtime,
    }


def run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    if not isinstance(argv, (tuple, list)) or not argv:
        raise ValueError("command argv must be non-empty")
    return subprocess.run(
        list(argv),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        env=_fixed_command_environment(),
    )


def discover_user_unit_directory(
    *,
    runtime: bool,
    runner: CommandRunner = run_command,
) -> Path:
    """Resolve the exact user unit directory with systemd-path --suffix.

    ``user-runtime`` and ``user-configuration`` denote directory roots.  The
    ``--suffix=systemd/user`` option is required; treating either root itself as
    a unit directory is an error.
    """

    key = "user-runtime" if runtime else "user-configuration"
    argv = (
        SYSTEMD_PATH,
        "--suffix=systemd/user",
        key,
    )
    completed = runner(argv)
    if completed.returncode != 0 or completed.stderr.strip():
        raise RuntimeError("systemd-path failed to resolve the user unit directory")
    rows = [row for row in completed.stdout.splitlines() if row]
    if len(rows) != 1:
        raise RuntimeError("systemd-path returned an ambiguous user unit directory")
    raw = Path(rows[0])
    if not raw.is_absolute() or raw.name != "user" or raw.parent.name != "systemd":
        raise RuntimeError("systemd-path returned an invalid user unit directory")
    return raw


def _unit_path_identity(path: Path) -> dict[str, object]:
    if not os.path.lexists(path):
        return {"path": str(path), "exists": False}
    linked = path.lstat()
    if path.is_symlink():
        resolved = path.resolve(strict=True)
        current = resolved.stat()
        if (
            not stat.S_ISLNK(linked.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or resolved.is_symlink()
            or resolved.resolve(strict=True) != resolved
            or linked.st_uid not in {0, os.getuid()}
            or current.st_uid not in {0, os.getuid()}
            or stat.S_IMODE(current.st_mode) & 0o022
        ):
            raise PermissionError(
                "user unit search path symlink target is not trusted"
            )
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
            "device": current.st_dev,
            "inode": current.st_ino,
            "owner_uid": current.st_uid,
            "mode": stat.S_IMODE(current.st_mode),
        }
    current = linked
    if (
        not stat.S_ISDIR(current.st_mode)
        or path.resolve(strict=True) != path
    ):
        raise PermissionError("user unit search path is not a canonical directory")
    return {
        "path": str(path), "exists": True, "device": current.st_dev,
        "inode": current.st_ino, "owner_uid": current.st_uid,
        "mode": stat.S_IMODE(current.st_mode),
    }


def freeze_user_unit_path_policy(
    unit_name: str,
    *,
    runner: CommandRunner = run_command,
    allowed_fragment: Path | None = None,
) -> dict[str, object]:
    """Freeze the complete user-unit search order and reject every shadow."""

    validate_integration_unit_name(unit_name)
    runtime_call = (
        SYSTEMD_PATH, "--suffix=systemd/user", "user-runtime"
    )
    analyze_call = (
        SYSTEMD_ANALYZE, "--user", "unit-paths", "--no-pager"
    )
    runtime_result = runner(runtime_call)
    analyze_result = runner(analyze_call)
    if runtime_result.returncode != 0 or analyze_result.returncode != 0:
        raise RuntimeError("cannot freeze the user unit search path")
    runtime_rows = [row for row in runtime_result.stdout.splitlines() if row]
    path_rows = [row for row in analyze_result.stdout.splitlines() if row]
    if (
        len(runtime_rows) != 1 or not path_rows
        or len(path_rows) != len(set(path_rows))
    ):
        raise PermissionError("user unit search path discovery is ambiguous")
    runtime_directory = Path(runtime_rows[0])
    paths = [Path(row) for row in path_rows]
    if (
        not runtime_directory.is_absolute()
        or any(not path.is_absolute() for path in paths)
        or paths.count(runtime_directory) != 1
    ):
        raise PermissionError("runtime unit directory has no unique search priority")
    runtime_identity = _unit_path_identity(runtime_directory)
    if (
        runtime_identity.get("exists") is not True
        or runtime_identity.get("owner_uid") != os.getuid()
        or int(runtime_identity.get("mode", 0)) & 0o022
    ):
        raise PermissionError("runtime unit directory is not owned and safe")
    normalized_allowed = (
        allowed_fragment.absolute() if allowed_fragment is not None else None
    )
    for path in paths:
        candidate = path / unit_name
        if os.path.lexists(candidate) and candidate != normalized_allowed:
            raise PermissionError("integration unit is shadowed in the search path")
        if candidate == normalized_allowed and os.path.lexists(candidate):
            current = candidate.lstat()
            if (
                candidate.is_symlink() or not stat.S_ISREG(current.st_mode)
                or candidate.resolve(strict=True) != candidate
                or current.st_uid != os.getuid() or current.st_nlink != 1
                or stat.S_IMODE(current.st_mode) != 0o600
            ):
                raise PermissionError("authorized integration fragment is unsafe")
    return {
        "systemd_path_argv": list(runtime_call),
        "systemd_path_stdout": runtime_result.stdout,
        "systemd_analyze_argv": list(analyze_call),
        "systemd_analyze_stdout": analyze_result.stdout,
        "ordered_unit_paths": [_unit_path_identity(path) for path in paths],
        "runtime_directory": str(runtime_directory),
        "runtime_directory_priority": paths.index(runtime_directory),
        "runtime_directory_identity": runtime_identity,
    }


def _open_verified_directory(path: Path, *, owner_uid: int) -> int:
    if not path.is_absolute():
        raise ValueError("unit directory must be absolute")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise PermissionError("unit directory must be a real directory")
    if before.st_uid != owner_uid:
        raise PermissionError("unit directory owner is not authorized")
    if before.st_mode & 0o022:
        raise PermissionError("unit directory must not be group/other writable")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        if (
            current.st_dev != before.st_dev
            or current.st_ino != before.st_ino
            or not stat.S_ISDIR(current.st_mode)
            or current.st_uid != owner_uid
            or current.st_mode & 0o022
        ):
            raise PermissionError("unit directory changed during verification")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def build_realization_plan(
    *,
    unit_name: str,
    unit_directory: Path,
    fragment_text: str,
    expected_fragment_sha256: str,
    owner_uid: int | None = None,
    execute_authorized: bool,
    removal_authorized: bool,
) -> RealizationPlan:
    validate_integration_unit_name(unit_name)
    if not isinstance(fragment_text, str) or not fragment_text.endswith("\n"):
        raise ValueError("fragment must be newline-terminated UTF-8 text")
    encoded = fragment_text.encode("utf-8")
    observed_sha256 = _sha256_bytes(encoded)
    if observed_sha256 != expected_fragment_sha256:
        raise PermissionError("rendered fragment SHA does not match authorization")
    directory = Path(unit_directory)
    if not directory.is_absolute():
        raise ValueError("unit directory must be absolute")
    fragment_path = directory / unit_name
    if fragment_path.parent != directory:
        raise PermissionError("fragment escaped the authorized unit directory")
    return RealizationPlan(
        unit_name=unit_name,
        unit_directory=directory,
        fragment_path=fragment_path,
        fragment_bytes=encoded,
        fragment_sha256=observed_sha256,
        owner_uid=os.getuid() if owner_uid is None else owner_uid,
        execute_authorized=execute_authorized,
        removal_authorized=removal_authorized,
    )


def realize_static_fragment(
    plan: RealizationPlan,
    *,
    execute: bool,
) -> dict[str, object]:
    """Create one exact fragment with O_EXCL, fsync, and inode verification."""

    validate_integration_unit_name(plan.unit_name)
    if not execute or plan.execute_authorized is not True:
        raise PermissionError("explicit authorized realization is required")
    if plan.fragment_path != plan.unit_directory / plan.unit_name:
        raise PermissionError("fragment path is not bound to the unit name")
    if _sha256_bytes(plan.fragment_bytes) != plan.fragment_sha256:
        raise PermissionError("fragment bytes changed after plan verification")

    directory_fd = _open_verified_directory(
        plan.unit_directory,
        owner_uid=plan.owner_uid,
    )
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(
            plan.unit_name,
            flags,
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(plan.fragment_bytes):
            written = os.write(descriptor, plan.fragment_bytes[offset:])
            if written <= 0:
                raise OSError("short write while creating systemd fragment")
            offset += written
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        linked = os.stat(
            plan.unit_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != plan.owner_uid
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size != len(plan.fragment_bytes)
            or opened.st_dev != linked.st_dev
            or opened.st_ino != linked.st_ino
            or opened.st_nlink != linked.st_nlink
        ):
            raise PermissionError("created fragment identity is unsafe")
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)

    if file_sha256(plan.fragment_path) != plan.fragment_sha256:
        raise PermissionError("created fragment content does not match authorization")
    final = os.lstat(plan.fragment_path)
    return {
        "fragment_path": str(plan.fragment_path),
        "fragment_sha256": plan.fragment_sha256,
        "device": final.st_dev,
        "inode": final.st_ino,
        "owner_uid": final.st_uid,
        "mode": stat.S_IMODE(final.st_mode),
        "nlink": final.st_nlink,
    }


def daemon_reload(
    *,
    execute: bool,
    runner: CommandRunner = run_command,
) -> None:
    if not execute:
        raise PermissionError("daemon-reload requires explicit execution")
    completed = runner((SYSTEMCTL_PATH, "--user", "daemon-reload"))
    if completed.returncode != 0:
        raise RuntimeError("user-systemd daemon-reload failed")


def query_unit_properties(
    unit_name: str,
    *,
    runner: CommandRunner = run_command,
) -> dict[str, str]:
    validate_integration_unit_name(unit_name)
    argv = (
        SYSTEMCTL_PATH,
        "--user",
        "show",
        unit_name,
        "--no-pager",
        *[f"--property={name}" for name in _STATIC_PROPERTIES],
    )
    completed = runner(argv)
    if completed.returncode != 0:
        raise RuntimeError("systemctl show failed")
    result: dict[str, str] = {}
    for row in completed.stdout.splitlines():
        if not row or "=" not in row:
            continue
        key, value = row.split("=", 1)
        if key in result:
            raise ValueError("systemctl returned a duplicate property")
        result[key] = value
    if set(result) != set(_STATIC_PROPERTIES):
        raise ValueError("systemctl returned an incomplete property set")
    return result


def validate_realized_static_unit(
    plan: RealizationPlan,
    properties: Mapping[str, str],
) -> None:
    expected = {
        "LoadState": "loaded",
        "UnitFileState": "static",
        "ActiveState": "inactive",
        "SubState": "dead",
        "FragmentPath": str(plan.fragment_path),
        "DropInPaths": "",
        "Transient": "no",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
    }
    if dict(properties) != expected:
        raise PermissionError("realized user-systemd unit is not exact and static")
    current = os.lstat(plan.fragment_path)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_uid != plan.owner_uid
        or current.st_nlink != 1
        or stat.S_IMODE(current.st_mode) != 0o600
        or file_sha256(plan.fragment_path) != plan.fragment_sha256
    ):
        raise PermissionError("realized fragment changed after daemon-reload")


def wait_until_unit_inactive_static(
    plan: RealizationPlan,
    *,
    query: UnitQuery,
    timeout_seconds: float = 5.0,
    poll_seconds: float = 0.05,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """Wait for inactive/dead while rejecting any immutable-unit drift."""

    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("polling intervals must be positive")
    immutable = {
        "LoadState": "loaded",
        "UnitFileState": "static",
        "FragmentPath": str(plan.fragment_path),
        "DropInPaths": "",
        "Transient": "no",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
    }
    deadline = monotonic() + timeout_seconds
    while True:
        observed = dict(query(plan.unit_name))
        if any(observed.get(key) != value for key, value in immutable.items()):
            raise PermissionError("integration unit identity drifted while stopping")
        current = os.lstat(plan.fragment_path)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_uid != plan.owner_uid
            or current.st_nlink != 1
            or stat.S_IMODE(current.st_mode) != 0o600
            or file_sha256(plan.fragment_path) != plan.fragment_sha256
        ):
            raise PermissionError("integration fragment drifted while stopping")
        if (
            observed.get("ActiveState") == "inactive"
            and observed.get("SubState") == "dead"
        ):
            return observed
        if monotonic() >= deadline:
            raise TimeoutError("integration unit did not become inactive/dead")
        sleep(poll_seconds)


def remove_integration_fragment(
    plan: RealizationPlan,
    *,
    execute: bool,
) -> None:
    """Remove only the exact authorized dummy fragment, never an actual unit."""

    validate_integration_unit_name(plan.unit_name)
    if (
        not execute
        or plan.execute_authorized is not True
        or plan.removal_authorized is not True
    ):
        raise PermissionError("explicit dummy-fragment removal authorization required")
    directory_fd = _open_verified_directory(
        plan.unit_directory,
        owner_uid=plan.owner_uid,
    )
    fragment_fd: int | None = None
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fragment_fd = os.open(plan.unit_name, flags, dir_fd=directory_fd)
        opened = os.fstat(fragment_fd)
        linked = os.stat(
            plan.unit_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != plan.owner_uid
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_dev != linked.st_dev
            or opened.st_ino != linked.st_ino
            or linked.st_nlink != 1
        ):
            raise PermissionError("dummy fragment is not an exact regular file")
        if _fd_sha256(fragment_fd) != plan.fragment_sha256:
            raise PermissionError("dummy fragment SHA changed before removal")
        linked_again = os.stat(
            plan.unit_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            linked_again.st_dev != opened.st_dev
            or linked_again.st_ino != opened.st_ino
            or linked_again.st_nlink != 1
        ):
            raise PermissionError("dummy fragment changed before unlink")
        os.unlink(plan.unit_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        if fragment_fd is not None:
            os.close(fragment_fd)
        os.close(directory_fd)


def wait_until_unit_not_found(
    unit_name: str,
    *,
    query: UnitQuery,
    timeout_seconds: float = 5.0,
    poll_seconds: float = 0.05,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("polling intervals must be positive")
    deadline = monotonic() + timeout_seconds
    while True:
        observed = dict(query(unit_name))
        if (
            observed.get("LoadState") == "not-found"
            and observed.get("ActiveState") == "inactive"
            and observed.get("SubState") == "dead"
            and observed.get("FragmentPath") == ""
            and observed.get("UnitFileState") == ""
        ):
            return observed
        if monotonic() >= deadline:
            raise TimeoutError("removed integration unit did not become not-found")
        sleep(poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only user-systemd unit-directory discovery helper"
    )
    parser.add_argument(
        "mode",
        choices=("discover-runtime-directory", "discover-configuration-directory"),
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    runtime = arguments.mode == "discover-runtime-directory"
    print(discover_user_unit_directory(runtime=runtime))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
