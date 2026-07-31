#!/usr/bin/env python3
"""One-shot recovery for the failed CURE-Lite v24 dummy integration r3.

The tool has exactly two phases.  ``authorize`` seals a short-lived
authorization for one already existing fragment.  ``apply`` consumes that
authorization once and permits only:

    exact unlink -> user daemon-reload -> exact not-found verification

It cannot start, stop, enable, retry, access a GPU, or inspect any dataset.
The archived r3 failure lineage and the live inactive/static unit must both
remain exact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Callable, Mapping, Sequence


try:
    from tools import cure_lite_v24_realize_systemd_unit as realizer
    from tools import cure_lite_v24_user_systemd_integration as integration
except ModuleNotFoundError:
    _TOOLS = Path(__file__).resolve().parent

    def _load(name: str, filename: str) -> object:
        module_spec = importlib.util.spec_from_file_location(
            name,
            _TOOLS / filename,
        )
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"cannot load isolated r3 recovery dependency:{name}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        return module

    realizer = _load(
        "cure_lite_v24_realize_systemd_unit_r3_recovery",
        "cure_lite_v24_realize_systemd_unit.py",
    )
    integration = _load(
        "cure_lite_v24_user_systemd_integration_r3_recovery",
        "cure_lite_v24_user_systemd_integration.py",
    )


REPOSITORY = Path("/home/md0/ly/cure_lite")
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
)
SCENARIO_ROOT = EVIDENCE_ROOT / "supervisor_v2_systemd_integration_r3"
CONTROL_ROOT = SCENARIO_ROOT / "control"
RUNTIME_ROOT = SCENARIO_ROOT / "runtime"
HEARTBEAT_ROOT = RUNTIME_ROOT / "heartbeat"
INVOCATION_ROOT = RUNTIME_ROOT / "systemd-invocations"

SCENARIO_ID = "supervisor-v2-dummy-r3-202607300430cafe"
UNIT_NAME = (
    "cure-lite-v24-supervisor-integration-"
    "supervisor-v2-dummy-r3-202607300430cafe.service"
)
STAGE_ID = f"systemd_integration_dummy_{SCENARIO_ID}"
ATTEMPT_ID = f"systemd_integration_dummy_attempt_{SCENARIO_ID}"
CANDIDATE = "systemd-integration-dummy"
EXECUTION_KIND = "systemd_integration_dummy"
INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后运行"
AUTHORIZED_UID = 1008
INVOCATION_ID = "ecef2827ab67470fadd42342272de6db"

EXPECTED_RUNTIME_UNIT_DIRECTORY = Path("/run/user/1008/systemd/user")
EXPECTED_FRAGMENT_PATH = EXPECTED_RUNTIME_UNIT_DIRECTORY / UNIT_NAME
EXPECTED_FRAGMENT_SHA256 = (
    "14db29ad9d63cc768aad05165fd6d699fa38b852a3bc4123db57550f38035bd3"
)
EXPECTED_FRAGMENT_DEVICE = 54
EXPECTED_FRAGMENT_INODE = 38294
EXPECTED_FRAGMENT_OWNER_UID = 1008
EXPECTED_FRAGMENT_MODE = 0o600
EXPECTED_FRAGMENT_NLINK = 1

ORIGINAL_AUTHORIZATION_PATH = CONTROL_ROOT / "authorization.json"
ORIGINAL_RUNTIME_SPEC_PATH = CONTROL_ROOT / "runtime-spec.json"
ORIGINAL_LAUNCH_LEASE_PATH = RUNTIME_ROOT / "launch-lease.json"
ORIGINAL_PRECOMMIT_PATH = RUNTIME_ROOT / "precommit-phase.json"
ORIGINAL_ATTEMPT_COMMIT_PATH = RUNTIME_ROOT / "attempt-commit.json"
ORIGINAL_START_ACK_PATH = RUNTIME_ROOT / "start-ack.json"
ORIGINAL_SIDECAR_PATH = INVOCATION_ROOT / f"{INVOCATION_ID}.json"
ORIGINAL_TERMINAL_PATH = CONTROL_ROOT / "integration-terminal.json"
ORIGINAL_REMOVAL_STATE_PATH = CONTROL_ROOT / "removal-state.json"

RECOVERY_AUTHORIZATION_PATH = (
    CONTROL_ROOT / "recovery-removal-authorization.json"
)
RECOVERY_INTENT_PATH = CONTROL_ROOT / "recovery-removal-intent.json"
RECOVERY_TERMINAL_PATH = CONTROL_ROOT / "recovery-removal-terminal.json"

AUTHORIZATION_SCHEMA = "cure-lite-v24-dummy-r3-recovery-authorization-v1"
INTENT_SCHEMA = "cure-lite-v24-dummy-r3-recovery-intent-v1"
TERMINAL_SCHEMA = "cure-lite-v24-dummy-r3-recovery-terminal-v1"

_SHA = re.compile(r"[0-9a-f]{64}")
_BOOT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
_FILE_BINDING_KEYS = {
    "path",
    "resolved_path",
    "path_is_symlink",
    "file_sha256",
    "device",
    "inode",
    "owner_uid",
    "mode",
}
_FRAGMENT_IDENTITY_KEYS = {
    "fragment_path",
    "fragment_sha256",
    "device",
    "inode",
    "owner_uid",
    "mode",
    "nlink",
}

_ARCHIVED_EVIDENCE_ANCHORS: dict[str, dict[str, object]] = {
    "authorization": {
        "path": str(ORIGINAL_AUTHORIZATION_PATH),
        "file_sha256": (
            "a1d3d91a7c83d37e814e7f58e5f0a3bab539e0ef317294ecb839228ae83324fd"
        ),
        "fingerprint_field": "authorization_fingerprint",
        "fingerprint": (
            "ef7a2959d1a9becbda13f0544fa2ff29bdb96dce9d3f2a231fce3814a9ce9162"
        ),
        "schema_version": (
            "cure-lite-v24-supervisor-v2-systemd-integration-authorization-v2"
        ),
        "mode": 0o444,
    },
    "runtime_spec": {
        "path": str(ORIGINAL_RUNTIME_SPEC_PATH),
        "file_sha256": (
            "819eb43a9f06cc0f2be0a8bdf3a2c31a579ca08873ca76e9134f9ebc062bac09"
        ),
        "fingerprint_field": "runtime_spec_fingerprint",
        "fingerprint": (
            "f1486339a6db8a855c8648448398e013f01ceb296dc8d6647a8477957e3651f4"
        ),
        "schema_version": "cure-lite-v24-dr-runtime-supervisor-spec-v2",
        "mode": 0o444,
    },
    "launch_lease": {
        "path": str(ORIGINAL_LAUNCH_LEASE_PATH),
        "file_sha256": (
            "ea9601b70f1b5997d01f810cedac001408fbefc6886cdd18c334d1b5468ed7b8"
        ),
        "fingerprint_field": "launch_lease_fingerprint",
        "fingerprint": (
            "4c7ce93852018682aad974192af8c2dbf5a1a8437462eb6067e21442e6369ca6"
        ),
        "schema_version": "cure-lite-v24-dr-launch-lease-v1",
        "mode": 0o444,
    },
    "precommit": {
        "path": str(ORIGINAL_PRECOMMIT_PATH),
        "file_sha256": (
            "88852a28add05dc8eb4ba6c802111ec8ea8a0b60018facbe9bf8e6769b62b404"
        ),
        "fingerprint_field": "phase_receipt_fingerprint",
        "fingerprint": (
            "0943bbd5b3ef545b73a7a3cc8e326949a7e942e312e58d6116c082064f9ac8de"
        ),
        "schema_version": "cure-lite-v24-dr-runtime-phase-receipt-v1",
        "mode": 0o444,
    },
    "attempt_commit": {
        "path": str(ORIGINAL_ATTEMPT_COMMIT_PATH),
        "file_sha256": (
            "e11967beee0417a9b614a2152b51adacb6bc1b9fc98eef40b3510189a3ded52b"
        ),
        "fingerprint_field": "attempt_commit_fingerprint",
        "fingerprint": (
            "60f6cd8222be8d714f509792c9fb98eb18d71de9992d56e9dd5e1973a5dfc892"
        ),
        "schema_version": "cure-lite-v24-dr-attempt-commit-v2",
        "mode": 0o444,
    },
    "start_ack": {
        "path": str(ORIGINAL_START_ACK_PATH),
        "file_sha256": (
            "48b8eaa99129d356c3aa00fc2730d9fffbf32b88d6c79a6403517b9257e02709"
        ),
        "fingerprint_field": "phase_receipt_fingerprint",
        "fingerprint": (
            "643e83a0514f22dea22584c8f8ac2461c4d1d1d27cbc6720253ea7d642f2177e"
        ),
        "schema_version": "cure-lite-v24-dr-runtime-phase-receipt-v1",
        "mode": 0o444,
    },
    "systemd_sidecar": {
        "path": str(ORIGINAL_SIDECAR_PATH),
        "file_sha256": (
            "849276b7849d3acfa87e7474f9e7c6986f99d169808e42a99e2e8774dcb7ea97"
        ),
        "fingerprint_field": "systemd_terminal_fingerprint",
        "fingerprint": (
            "3c3962e4a902efdc39da05dbd92c438b48b80475c36b5df4aef6b5b1d8c40074"
        ),
        "schema_version": "cure-lite-v24-dr-systemd-terminal-v1",
        "mode": 0o400,
    },
    "integration_terminal": {
        "path": str(ORIGINAL_TERMINAL_PATH),
        "file_sha256": (
            "b27cccaab88af2ec2dc2a44114dfb1c1494c42ae7b45e7f52bb56c8798b623b8"
        ),
        "fingerprint_field": "integration_terminal_fingerprint",
        "fingerprint": (
            "33c0ec3be2cb13004a3980b29974711da0267e120f4077649aab1589f6e16caa"
        ),
        "schema_version": (
            "cure-lite-v24-supervisor-v2-systemd-integration-terminal-v1"
        ),
        "mode": 0o444,
    },
    "removal_state": {
        "path": str(ORIGINAL_REMOVAL_STATE_PATH),
        "file_sha256": (
            "57e7f63ee0bfe60058fc43db67b5b8a515c8e8f3fc0201877db812307f9d9a09"
        ),
        "fingerprint_field": "removal_state_fingerprint",
        "fingerprint": (
            "45dfc13b60f3b0ee388639abdee23ba165333b6ab691d1e3562fa57f2e73b612"
        ),
        "schema_version": (
            "cure-lite-v24-supervisor-v2-integration-removal-state-v1"
        ),
        "mode": 0o444,
    },
}

_AUTHORIZATION_KEYS = {
    "schema_version",
    "scenario_id",
    "unit_name",
    "instruction_id",
    "authorization_basis",
    "authorized_uid",
    "issued_at_utc",
    "expires_at_utc",
    "archived_roots",
    "archived_executable_bindings",
    "current_recovery_tool_binding",
    "current_required_executable_bindings",
    "manager_generation",
    "unit_path_policy",
    "fragment_identity",
    "inactive_static_state",
    "authorized_action",
    "remove_authorized",
    "daemon_reload_authorized",
    "not_found_verification_authorized",
    "start_authorized",
    "stop_authorized",
    "enable_authorized",
    "automatic_retry_authorized",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "gpu_accessed",
    "recovery_authorization_fingerprint",
}
_INTENT_KEYS = {
    "schema_version",
    "created_at_utc",
    "scenario_id",
    "unit_name",
    "recovery_authorization_path",
    "recovery_authorization_file_sha256",
    "recovery_authorization_fingerprint",
    "fragment_identity",
    "inactive_static_state",
    "authorized_action",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "gpu_accessed",
    "recovery_intent_fingerprint",
}
_RECOVERY_TERMINAL_KEYS = {
    "schema_version",
    "created_at_utc",
    "scenario_id",
    "unit_name",
    "recovery_authorization_fingerprint",
    "recovery_intent_fingerprint",
    "action_started_at_utc",
    "completed_actions",
    "fragment_absent",
    "post_removal_unit_state",
    "passed",
    "error_type",
    "error_message",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "gpu_accessed",
    "recovery_terminal_fingerprint",
}


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    """Identity and mutation-sensitive fields required for every FD read."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_nlink,
        stat.S_IMODE(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_fd_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    return b"".join(chunks)


def _read_regular_file_snapshot(
    path: str | Path,
    *,
    expected_owner_uid: int | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    """Read one unchanged regular-file object through O_NOFOLLOW.

    ``pre`` is the path lstat before open, ``opened`` is the initial fstat,
    ``post`` is fstat after the complete FD read, and ``path_post`` is the
    final path lstat.  All mutation-sensitive identity fields must agree.
    """

    target = Path(path).absolute()
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("r3 recovery requires O_NOFOLLOW")
    try:
        pre = target.lstat()
    except OSError as error:
        raise PermissionError("r3 snapshot path is unavailable before open") from error
    if (
        stat.S_ISLNK(pre.st_mode)
        or not stat.S_ISREG(pre.st_mode)
        or pre.st_nlink != 1
    ):
        raise PermissionError("r3 snapshot path is not a safe regular file")
    try:
        resolved_pre = target.resolve(strict=True)
    except OSError as error:
        raise PermissionError("r3 snapshot path cannot be resolved before open") from error
    if resolved_pre != target:
        raise PermissionError("r3 snapshot path is not canonical")

    descriptor = -1
    try:
        try:
            descriptor = os.open(
                target,
                os.O_RDONLY | os.O_CLOEXEC | nofollow,
            )
        except OSError as error:
            raise PermissionError(
                "r3 snapshot O_NOFOLLOW open failed"
            ) from error
        opened = os.fstat(descriptor)
        data = _read_fd_bytes(descriptor)
        post = os.fstat(descriptor)
        try:
            path_post = target.lstat()
            resolved_post = target.resolve(strict=True)
        except OSError as error:
            raise PermissionError(
                "r3 snapshot path changed after FD read"
            ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    identities = (
        _stat_identity(pre),
        _stat_identity(opened),
        _stat_identity(post),
        _stat_identity(path_post),
    )
    if (
        any(not stat.S_ISREG(value.st_mode) for value in (pre, opened, post, path_post))
        or len(set(identities)) != 1
        or resolved_post != target
        or len(data) != post.st_size
        or (
            expected_owner_uid is not None
            and post.st_uid != expected_owner_uid
        )
        or (
            expected_mode is not None
            and stat.S_IMODE(post.st_mode) != expected_mode
        )
    ):
        raise PermissionError(
            "r3 snapshot pre/open/post/path identity changed"
        )
    return data, post


def file_sha256(path: str | Path) -> str:
    data, _current = _read_regular_file_snapshot(path)
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: object, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} timestamp is malformed") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} timestamp is naive")
    return parsed


def _assert_repository() -> None:
    if Path(__file__).resolve().parents[1] != REPOSITORY:
        raise PermissionError("r3 recovery repository path changed")
    if os.getuid() != AUTHORIZED_UID:
        raise PermissionError("r3 recovery must run as the archived authorized UID")


def _private_directory(path: Path) -> dict[str, object]:
    target = path.absolute()
    current = target.lstat()
    if (
        target.is_symlink()
        or not stat.S_ISDIR(current.st_mode)
        or target.resolve(strict=True) != target
        or current.st_uid != AUTHORIZED_UID
        or stat.S_IMODE(current.st_mode) != 0o700
    ):
        raise PermissionError("r3 recovery directory is not exact private storage")
    return {
        "path": str(target),
        "device": current.st_dev,
        "inode": current.st_ino,
        "owner_uid": current.st_uid,
        "mode": stat.S_IMODE(current.st_mode),
    }


def _file_binding(path: Path, *, allow_symlink: bool = False) -> dict[str, object]:
    supplied = path.absolute()
    if (
        any(character.isspace() for character in str(supplied))
        or allow_symlink
    ):
        raise PermissionError("r3 recovery dependency cannot follow a symlink")
    data, current = _read_regular_file_snapshot(supplied)
    return {
        "path": str(supplied),
        "resolved_path": str(supplied),
        "path_is_symlink": False,
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "device": current.st_dev,
        "inode": current.st_ino,
        "owner_uid": current.st_uid,
        "mode": stat.S_IMODE(current.st_mode),
    }


def _current_required_bindings() -> dict[str, object]:
    expected_integration = (
        REPOSITORY / "tools/cure_lite_v24_user_systemd_integration.py"
    )
    expected_realizer = (
        REPOSITORY / "tools/cure_lite_v24_realize_systemd_unit.py"
    )
    integration_path = Path(str(integration.__file__)).absolute()
    realizer_path = Path(str(realizer.__file__)).absolute()
    integration_realizer_path = Path(
        str(integration.realizer.__file__)
    ).absolute()
    if (
        integration_path != expected_integration
        or realizer_path != expected_realizer
        or integration_realizer_path != expected_realizer
        or str(realizer.SYSTEMCTL_PATH) != "/usr/bin/systemctl"
        or str(realizer.SYSTEMD_PATH) != "/usr/bin/systemd-path"
        or str(realizer.SYSTEMD_ANALYZE) != "/usr/bin/systemd-analyze"
    ):
        raise PermissionError("r3 recovery dependency path changed")
    return {
        "integration_library": _file_binding(integration_path),
        "realizer_library": _file_binding(realizer_path),
        "systemctl": _file_binding(Path(str(realizer.SYSTEMCTL_PATH))),
        "systemd_path": _file_binding(Path(str(realizer.SYSTEMD_PATH))),
        "systemd_analyze": _file_binding(Path(str(realizer.SYSTEMD_ANALYZE))),
    }


def _current_recovery_tool_binding() -> dict[str, object]:
    return _file_binding(Path(__file__).absolute())


def _validate_file_binding_shape(
    value: object,
    *,
    name: str,
    expected_path: str | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FILE_BINDING_KEYS:
        raise PermissionError(f"r3 archived executable binding malformed:{name}")
    binding = dict(value)
    if (
        (expected_path is not None and binding.get("path") != expected_path)
        or not isinstance(binding.get("resolved_path"), str)
        or not isinstance(binding.get("path_is_symlink"), bool)
        or _SHA.fullmatch(str(binding.get("file_sha256"))) is None
        or any(
            isinstance(binding.get(field), bool)
            or not isinstance(binding.get(field), int)
            or int(binding[field]) < 0
            for field in ("device", "inode", "owner_uid", "mode")
        )
    ):
        raise PermissionError(f"r3 archived executable binding malformed:{name}")
    return binding


def _write_sealed(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    target = path.absolute()
    parent_binding = _private_directory(target.parent)
    parent_pre = target.parent.lstat()
    if parent_binding != {
        "path": str(target.parent),
        "device": parent_pre.st_dev,
        "inode": parent_pre.st_ino,
        "owner_uid": parent_pre.st_uid,
        "mode": stat.S_IMODE(parent_pre.st_mode),
    }:
        raise PermissionError("r3 recovery evidence parent changed before open")
    if fingerprint_field in body:
        raise ValueError("r3 recovery body already contains fingerprint")
    payload = {
        **dict(body),
        fingerprint_field: stable_fingerprint(dict(body)),
    }
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("r3 recovery requires O_NOFOLLOW")
    parent_fd = os.open(
        target.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | nofollow,
    )
    descriptor = -1
    try:
        parent_opened = os.fstat(parent_fd)

        def _parent_core(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_uid,
                value.st_nlink,
                stat.S_IMODE(value.st_mode),
            )

        if (
            not stat.S_ISDIR(parent_pre.st_mode)
            or not stat.S_ISDIR(parent_opened.st_mode)
            or _parent_core(parent_pre) != _parent_core(parent_opened)
        ):
            raise PermissionError(
                "r3 recovery evidence parent changed at open"
            )
        descriptor = os.open(
            target.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | nofollow,
            0o444,
            dir_fd=parent_fd,
        )
        os.fchmod(descriptor, 0o444)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("zero-byte r3 recovery evidence write")
            offset += written
        os.fsync(descriptor)
        read_pre = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        readback = _read_fd_bytes(descriptor)
        read_post = os.fstat(descriptor)
        path_post = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            any(
                not stat.S_ISREG(value.st_mode)
                for value in (read_pre, read_post, path_post)
            )
            or len(
                {
                    _stat_identity(read_pre),
                    _stat_identity(read_post),
                    _stat_identity(path_post),
                }
            )
            != 1
            or read_post.st_uid != AUTHORIZED_UID
            or read_post.st_nlink != 1
            or stat.S_IMODE(read_post.st_mode) != 0o444
            or read_post.st_size != len(encoded)
            or readback != encoded
        ):
            raise PermissionError("r3 recovery sealed evidence identity is unsafe")
        os.fsync(parent_fd)
        parent_post = os.fstat(parent_fd)
        parent_path_post = target.parent.lstat()
        if (
            any(
                not stat.S_ISDIR(value.st_mode)
                for value in (parent_post, parent_path_post)
            )
            or _parent_core(parent_pre) != _parent_core(parent_post)
            or _parent_core(parent_post) != _parent_core(parent_path_post)
            or _stat_identity(parent_post)
            != _stat_identity(parent_path_post)
            or target.parent.resolve(strict=True) != target.parent
        ):
            raise PermissionError(
                "r3 recovery evidence parent changed after write"
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    return payload


def _read_recovery_sealed_snapshot(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str,
) -> tuple[dict[str, object], str, tuple[int, ...]]:
    target = path.absolute()
    raw, current = _read_regular_file_snapshot(
        target,
        expected_owner_uid=AUTHORIZED_UID,
        expected_mode=0o444,
    )
    payload = json.loads(raw.decode("utf-8"))
    if (
        not isinstance(payload, dict)
        or raw != (canonical_json(payload) + "\n").encode("utf-8")
    ):
        raise ValueError("r3 recovery evidence is not canonical JSON")
    body = dict(payload)
    fingerprint = body.pop(fingerprint_field, None)
    if (
        not isinstance(fingerprint, str)
        or _SHA.fullmatch(fingerprint) is None
        or fingerprint != stable_fingerprint(body)
        or payload.get("schema_version") != schema
    ):
        raise PermissionError("r3 recovery fingerprint or schema is invalid")
    return (
        payload,
        hashlib.sha256(raw).hexdigest(),
        _stat_identity(current),
    )


def _read_recovery_sealed(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str,
) -> dict[str, object]:
    payload, _file_digest, _identity = _read_recovery_sealed_snapshot(
        path,
        fingerprint_field=fingerprint_field,
        schema=schema,
    )
    return payload


def _read_archived(name: str) -> dict[str, object]:
    anchor = _ARCHIVED_EVIDENCE_ANCHORS[name]
    target = Path(str(anchor["path"]))
    expected_mode = int(anchor["mode"])
    raw, _current = _read_regular_file_snapshot(
        target,
        expected_owner_uid=AUTHORIZED_UID,
        expected_mode=expected_mode,
    )
    if (
        hashlib.sha256(raw).hexdigest() != anchor["file_sha256"]
    ):
        raise PermissionError(f"r3 archived evidence identity changed:{name}")
    payload = json.loads(raw.decode("utf-8"))
    if (
        not isinstance(payload, dict)
        or raw != (canonical_json(payload) + "\n").encode("utf-8")
        or payload.get("schema_version") != anchor["schema_version"]
    ):
        raise PermissionError(f"r3 archived evidence encoding changed:{name}")
    body = dict(payload)
    fingerprint = body.pop(str(anchor["fingerprint_field"]), None)
    if (
        fingerprint != anchor["fingerprint"]
        or fingerprint != stable_fingerprint(body)
    ):
        raise PermissionError(f"r3 archived evidence fingerprint changed:{name}")
    return payload


def _archived_roots() -> dict[str, object]:
    return json.loads(canonical_json(_ARCHIVED_EVIDENCE_ANCHORS))


def _no_payload(value: Mapping[str, object]) -> None:
    if (
        value.get("payload_authority") != "none"
        or value.get("D_R_payload_accessed") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or value.get("gpu_accessed", False) is not False
    ):
        raise PermissionError("r3 archived integration is not payload-free")


def _identity() -> dict[str, str]:
    return {
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "attempt_id": ATTEMPT_ID,
        "unit_name": UNIT_NAME,
    }


def _common_lineage(value: Mapping[str, object]) -> bool:
    return (
        value.get("candidate") == CANDIDATE
        and value.get("stage_id") == STAGE_ID
        and value.get("attempt_id") == ATTEMPT_ID
        and value.get("execution_kind") == EXECUTION_KIND
        and value.get("runtime_spec_fingerprint")
        == _ARCHIVED_EVIDENCE_ANCHORS["runtime_spec"]["fingerprint"]
    )


def _validate_archived_inventory() -> None:
    if set(SCENARIO_ROOT.iterdir()) != {CONTROL_ROOT, RUNTIME_ROOT}:
        raise PermissionError("r3 scenario root inventory changed")
    expected_control = {
        ORIGINAL_AUTHORIZATION_PATH,
        ORIGINAL_RUNTIME_SPEC_PATH,
        ORIGINAL_TERMINAL_PATH,
        ORIGINAL_REMOVAL_STATE_PATH,
    }
    optional_recovery = {
        RECOVERY_AUTHORIZATION_PATH,
        RECOVERY_INTENT_PATH,
        RECOVERY_TERMINAL_PATH,
    }
    actual_control = set(CONTROL_ROOT.iterdir())
    if (
        not expected_control.issubset(actual_control)
        or not actual_control.issubset(expected_control | optional_recovery)
    ):
        raise PermissionError("r3 control evidence inventory changed")
    for path in actual_control & optional_recovery:
        current = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(current.st_mode)
            or current.st_uid != AUTHORIZED_UID
            or current.st_nlink != 1
            or stat.S_IMODE(current.st_mode) != 0o444
        ):
            raise PermissionError("r3 recovery evidence inventory is unsafe")
    expected_runtime = {
        ORIGINAL_LAUNCH_LEASE_PATH,
        ORIGINAL_PRECOMMIT_PATH,
        ORIGINAL_ATTEMPT_COMMIT_PATH,
        ORIGINAL_START_ACK_PATH,
        HEARTBEAT_ROOT,
        INVOCATION_ROOT,
    }
    if set(RUNTIME_ROOT.iterdir()) != expected_runtime:
        raise PermissionError(
            "r3 runtime reached claim/child/GPU/payload/terminal evidence"
        )
    if any(HEARTBEAT_ROOT.iterdir()):
        raise PermissionError("r3 runtime heartbeat proves a child was reached")
    if set(INVOCATION_ROOT.iterdir()) != {ORIGINAL_SIDECAR_PATH}:
        raise PermissionError("r3 systemd sidecar inventory changed")


def _sealed_original_chain() -> dict[str, object]:
    _assert_repository()
    authorization = _read_archived("authorization")
    runtime_spec = _read_archived("runtime_spec")
    launch_lease = _read_archived("launch_lease")
    precommit = _read_archived("precommit")
    attempt_commit = _read_archived("attempt_commit")
    start_ack = _read_archived("start_ack")
    sidecar = _read_archived("systemd_sidecar")
    terminal = _read_archived("integration_terminal")
    removal_state = _read_archived("removal_state")
    _validate_archived_inventory()

    _no_payload(authorization)
    _no_payload(terminal)
    _no_payload(removal_state)
    expected_identity = _identity()
    expected_actions = [
        "realize-static-fragment",
        "daemon-reload-after-realization",
        "supervisor-commit-and-start",
    ]
    issued = _timestamp(
        authorization.get("issued_at_utc"),
        name="r3 integration issuance",
    )
    expires = _timestamp(
        authorization.get("expires_at_utc"),
        name="r3 integration expiry",
    )
    lease_time = _timestamp(launch_lease.get("time_utc"), name="r3 launch lease")
    precommit_time = _timestamp(precommit.get("time_utc"), name="r3 precommit")
    commit_time = _timestamp(
        attempt_commit.get("time_utc"),
        name="r3 attempt commit",
    )
    start_ack_time = _timestamp(start_ack.get("time_utc"), name="r3 start ack")
    sidecar_time = _timestamp(sidecar.get("time_utc"), name="r3 sidecar")
    terminal_time = _timestamp(
        terminal.get("created_at_utc"),
        name="r3 integration terminal",
    )
    if not (
        issued
        <= lease_time
        <= precommit_time
        <= commit_time
        <= start_ack_time
        <= sidecar_time
        <= terminal_time
        <= expires
    ):
        raise PermissionError("r3 archived chronology changed")
    if expires <= issued or expires - issued > timedelta(seconds=300):
        raise PermissionError("r3 archived authorization interval changed")

    manager = authorization.get("manager_generation")
    boot_id = (
        manager.get("boot_id") if isinstance(manager, Mapping) else None
    )
    if (
        _BOOT_ID.fullmatch(str(boot_id)) is None
        or authorization.get("scenario_id") != SCENARIO_ID
        or authorization.get("identity") != expected_identity
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis") != AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != AUTHORIZED_UID
        or authorization.get("integration_authorized") is not True
        or authorization.get("actual_r2_authorized") is not False
        or authorization.get("unit_realization_authorized") is not True
        or authorization.get("unit_removal_authorized") is not False
        or authorization.get("direct_start_authorized") is not False
        or authorization.get("enable_authorized") is not False
        or authorization.get("gpu_access_authorized") is not False
        or authorization.get("unit_directory")
        != str(EXPECTED_RUNTIME_UNIT_DIRECTORY)
        or authorization.get("rendered_fragment", {}).get("sha256")
        != EXPECTED_FRAGMENT_SHA256
        or authorization.get("runtime_spec_binding", {}).get("path")
        != str(ORIGINAL_RUNTIME_SPEC_PATH)
        or authorization.get("runtime_spec_binding", {}).get("file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["runtime_spec"]["file_sha256"]
        or authorization.get("runtime_spec_binding", {}).get(
            "runtime_spec_fingerprint"
        )
        != _ARCHIVED_EVIDENCE_ANCHORS["runtime_spec"]["fingerprint"]
    ):
        raise PermissionError("r3 archived authorization lineage changed")
    rendered = authorization["rendered_fragment"]
    if (
        hashlib.sha256(
            str(rendered.get("utf8_text")).encode("utf-8")
        ).hexdigest()
        != EXPECTED_FRAGMENT_SHA256
    ):
        raise PermissionError("r3 archived rendered fragment changed")

    for key in ("scenario_root", "control_root", "runtime_root"):
        binding = authorization.get(key)
        if not isinstance(binding, Mapping):
            raise PermissionError(f"r3 archived directory binding missing:{key}")
        expected_path = {
            "scenario_root": SCENARIO_ROOT,
            "control_root": CONTROL_ROOT,
            "runtime_root": RUNTIME_ROOT,
        }[key]
        if dict(binding) != _private_directory(expected_path):
            raise PermissionError(f"r3 archived directory identity changed:{key}")

    control_artifacts = authorization.get("control_artifacts")
    expected_control_artifacts = {
        "dummy_artifact": str(RUNTIME_ROOT / "dummy-child.json"),
        "integration_receipt": str(CONTROL_ROOT / "integration-receipt.json"),
        "integration_terminal": str(ORIGINAL_TERMINAL_PATH),
        "removal_authorization": str(CONTROL_ROOT / "removal-authorization.json"),
        "removal_state": str(ORIGINAL_REMOVAL_STATE_PATH),
    }
    if control_artifacts != expected_control_artifacts:
        raise PermissionError("r3 archived control-artifact contract changed")

    archived = authorization.get("executable_bindings")
    expected_executables = {
        "python": "/home/md0/ly/MSHNet/.venv/bin/python",
        "supervisor": str(REPOSITORY / "tools/cure_lite_v24_runtime_supervisor.py"),
        "integration_tool": str(
            REPOSITORY / "tools/cure_lite_v24_user_systemd_integration.py"
        ),
        "realizer": str(
            REPOSITORY / "tools/cure_lite_v24_realize_systemd_unit.py"
        ),
        "dummy_child": str(REPOSITORY / "tools/cure_lite_v24_dummy_child.py"),
        "systemd_path": "/usr/bin/systemd-path",
        "systemd_analyze": "/usr/bin/systemd-analyze",
        "systemctl": "/usr/bin/systemctl",
    }
    if not isinstance(archived, Mapping) or set(archived) != set(
        expected_executables
    ):
        raise PermissionError("r3 archived executable root set changed")
    archived_bindings: dict[str, object] = {}
    for name, expected_path in expected_executables.items():
        archived_bindings[name] = _validate_file_binding_shape(
            archived[name],
            name=name,
            expected_path=expected_path,
        )
    template = _validate_file_binding_shape(
        authorization.get("template_binding"),
        name="template",
        expected_path=str(
            REPOSITORY
            / "deploy/systemd/cure-lite-v24-supervisor-integration.service.template"
        ),
    )

    artifacts = runtime_spec.get("artifacts")
    expected_artifacts = {
        "attempt_commit": str(ORIGINAL_ATTEMPT_COMMIT_PATH),
        "child_prespawn_phase_receipt": str(RUNTIME_ROOT / "child-prespawn.json"),
        "consumed_start_failure_receipt": str(
            RUNTIME_ROOT / "consumed-start-failure.json"
        ),
        "gpu_lease_release_receipt": str(RUNTIME_ROOT / "gpu-lease-release.json"),
        "heartbeat_dir": str(HEARTBEAT_ROOT),
        "launch_lease": str(ORIGINAL_LAUNCH_LEASE_PATH),
        "materialization_claim": str(RUNTIME_ROOT / "materialization-claim.json"),
        "precommit_phase_receipt": str(ORIGINAL_PRECOMMIT_PATH),
        "root": str(RUNTIME_ROOT),
        "runtime_attestation": str(RUNTIME_ROOT / "runtime-attestation.json"),
        "runtime_terminal": str(RUNTIME_ROOT / "runtime-terminal.json"),
        "start_ack_receipt": str(ORIGINAL_START_ACK_PATH),
        "stderr_log": str(RUNTIME_ROOT / "stderr.log"),
        "stdout_log": str(RUNTIME_ROOT / "stdout.log"),
        "systemd_invocation_dir": str(INVOCATION_ROOT),
    }
    runtime_contract = runtime_spec.get("runtime")
    systemd_contract = (
        runtime_contract.get("systemd")
        if isinstance(runtime_contract, Mapping)
        else None
    )
    if (
        not _common_lineage(runtime_spec)
        or runtime_spec.get("attempt_ordinal") != 0
        or runtime_spec.get("prior_attempt_count") != 0
        or runtime_spec.get("authorization") is not None
        or runtime_spec.get("environment") is not None
        or runtime_spec.get("scientific_preaccess") is not None
        or artifacts != expected_artifacts
        or not isinstance(runtime_contract, Mapping)
        or runtime_contract.get("automatic_retry_allowed") is not False
        or runtime_contract.get("resume_allowed") is not False
        or runtime_contract.get("launch_limit") != 1
        or runtime_contract.get("restart") != "no"
        or not isinstance(systemd_contract, Mapping)
        or systemd_contract.get("unit_name") != UNIT_NAME
        or systemd_contract.get("unit_fragment_file_sha256")
        != EXPECTED_FRAGMENT_SHA256
        or runtime_spec.get("source_bindings", {}).get(
            "supervisor_file_sha256"
        )
        != archived_bindings["supervisor"]["file_sha256"]
        or runtime_spec.get("source_bindings", {}).get(
            "child_entry_file_sha256"
        )
        != archived_bindings["dummy_child"]["file_sha256"]
    ):
        raise PermissionError("r3 archived runtime-spec lineage changed")

    immutable_fingerprint = systemd_contract.get(
        "immutable_shadow_fingerprint"
    )
    if (
        not _common_lineage(launch_lease)
        or launch_lease.get("boot_id") != boot_id
        or launch_lease.get("authorization_fingerprint") is not None
        or launch_lease.get("gpu_exclusivity_claimed") is not False
        or launch_lease.get("launch_limit") != 1
        or launch_lease.get("automatic_retry_allowed") is not False
        or launch_lease.get("resume_allowed") is not False
        or launch_lease.get("lease_scope") != "attempt_dispatch_only"
        or not _common_lineage(precommit)
        or precommit.get("boot_id") != boot_id
        or precommit.get("phase") != "precommit"
        or precommit.get("launch_lease_fingerprint")
        != launch_lease.get("launch_lease_fingerprint")
        or precommit.get("immutable_shadow_fingerprint")
        != immutable_fingerprint
        or precommit.get("runtime_environment_audit_valid") is not False
        or precommit.get("environment_audit_fingerprint") is not None
        or precommit.get("environment_inventory_fingerprint") is not None
        or precommit.get("gpu_lease_fingerprint") is not None
        or precommit.get("scientific_gate_passed") is not None
        or not _common_lineage(attempt_commit)
        or attempt_commit.get("boot_id") != boot_id
        or attempt_commit.get("attempt_ordinal") != 0
        or attempt_commit.get("prior_attempt_count") != 0
        or attempt_commit.get("authorization_fingerprint") is not None
        or attempt_commit.get("authorization_file_sha256") is not None
        or attempt_commit.get("gpu_lease_fingerprint") is not None
        or attempt_commit.get("gpu_lease_file_sha256") is not None
        or attempt_commit.get("gpu_lease_device") is not None
        or attempt_commit.get("gpu_lease_inode") is not None
        or attempt_commit.get("planned_attempt_commit_fingerprint") is not None
        or attempt_commit.get("launch_lease_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["launch_lease"]["file_sha256"]
        or attempt_commit.get("launch_lease_fingerprint")
        != launch_lease.get("launch_lease_fingerprint")
        or attempt_commit.get("precommit_phase_receipt_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["precommit"]["file_sha256"]
        or attempt_commit.get("precommit_phase_receipt_fingerprint")
        != precommit.get("phase_receipt_fingerprint")
        or attempt_commit.get("immutable_systemd_shadow_fingerprint")
        != immutable_fingerprint
        or attempt_commit.get("automatic_retry_allowed") is not False
        or attempt_commit.get("resume_allowed") is not False
        or attempt_commit.get("scientific_gate_passed") is not None
        or not _common_lineage(start_ack)
        or start_ack.get("boot_id") != boot_id
        or start_ack.get("phase") != "start_ack"
        or start_ack.get("launch_lease_fingerprint")
        != launch_lease.get("launch_lease_fingerprint")
        or start_ack.get("immutable_shadow_fingerprint")
        != immutable_fingerprint
        or start_ack.get("gpu_lease_fingerprint") is not None
        or start_ack.get("scientific_gate_passed") is not None
        or start_ack.get("systemd_phase_state", {}).get("InvocationID")
        != INVOCATION_ID
    ):
        raise PermissionError("r3 archived pre-child commit chain changed")
    monotonic_values = [
        launch_lease.get("monotonic_ns"),
        precommit.get("monotonic_ns"),
        attempt_commit.get("monotonic_ns"),
        start_ack.get("monotonic_ns"),
    ]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in monotonic_values
        )
        or monotonic_values != sorted(monotonic_values)
        or len(set(monotonic_values)) != len(monotonic_values)
    ):
        raise PermissionError("r3 archived monotonic chronology changed")

    if (
        sidecar.get("candidate") != CANDIDATE
        or sidecar.get("stage_id") != STAGE_ID
        or sidecar.get("attempt_id") != ATTEMPT_ID
        or sidecar.get("runtime_spec_fingerprint")
        != _ARCHIVED_EVIDENCE_ANCHORS["runtime_spec"]["fingerprint"]
        or sidecar.get("sidecar_systemd_invocation_id") != INVOCATION_ID
        or sidecar.get("attempt_commit_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["attempt_commit"]["file_sha256"]
        or sidecar.get("attempt_commit_fingerprint")
        != attempt_commit.get("attempt_commit_fingerprint")
        or sidecar.get("start_ack_receipt_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["start_ack"]["file_sha256"]
        or sidecar.get("start_ack_receipt_fingerprint")
        != start_ack.get("phase_receipt_fingerprint")
        or sidecar.get("attempt_commit_required") is not True
        or sidecar.get("attempt_commit_valid") is not True
        or sidecar.get("start_ack_valid") is not True
        or sidecar.get("authorization_matches_commit") is not True
        or sidecar.get("current_runtime_closure_valid") is not True
        or sidecar.get("claim_valid") is not False
        or sidecar.get("claim_matches_invocation") is not False
        or sidecar.get("claim_systemd_invocation_id") is not None
        or sidecar.get("materialization_claim_file_sha256") is not None
        or sidecar.get("materialization_claim_fingerprint") is not None
        or sidecar.get("child_prespawn_valid") is not False
        or sidecar.get("child_prespawn_phase_receipt_file_sha256") is not None
        or sidecar.get("child_prespawn_phase_receipt_fingerprint") is not None
        or sidecar.get("active_gpu_lease_valid") is not None
        or sidecar.get("active_gpu_lease_fingerprint") is not None
        or sidecar.get("gpu_lease_release_authorized") is not None
        or sidecar.get("gpu_lease_release_valid") is not None
        or sidecar.get("gpu_lease_release_receipt_fingerprint") is not None
        or sidecar.get("gpu_lease_tombstone_file_sha256") is not None
        or sidecar.get("audit_valid") is not False
        or sidecar.get("scientific_gate_passed") is not None
        or sidecar.get("scientific_decision")
        != "NOT_EVALUATED_BY_RUNTIME_SUPERVISOR"
        or sidecar.get("systemd_outcome", {}).get("category")
        != "SYSTEMD_EXEC_CONDITION"
        or sidecar.get("systemd_outcome", {}).get("service_result")
        != "exec-condition"
        or sidecar.get("systemd_outcome", {}).get("systemd_success") is not False
    ):
        raise PermissionError("r3 archived zero-claim sidecar changed")

    authorization_fingerprint = authorization["authorization_fingerprint"]
    runtime_spec_fingerprint = runtime_spec["runtime_spec_fingerprint"]
    if (
        terminal.get("scenario_id") != SCENARIO_ID
        or terminal.get("identity") != expected_identity
        or terminal.get("authorization_fingerprint")
        != authorization_fingerprint
        or terminal.get("runtime_spec_fingerprint")
        != runtime_spec_fingerprint
        or terminal.get("passed") is not False
        or terminal.get("completed_actions") != expected_actions
        or terminal.get("supervisor_evidence") is not None
        or terminal.get("error_type") != "TimeoutError"
        or terminal.get("error_message")
        != "supervisor terminal evidence did not arrive"
        or terminal.get("direct_systemctl_start_attempted") is not False
        or terminal.get("enable_attempted") is not False
        or terminal.get("remove_attempted") is not False
        or removal_state.get("scenario_id") != SCENARIO_ID
        or removal_state.get("unit_name") != UNIT_NAME
        or removal_state.get("removal_authorization_fingerprint") is not None
        or removal_state.get("passed") is not False
        or removal_state.get("remove_attempted") is not False
        or removal_state.get("fragment_absent") is not False
        or removal_state.get("not_found_state") is not None
        or removal_state.get("completed_actions") != expected_actions
        or removal_state.get("error_type") != "TimeoutError"
        or removal_state.get("error_message")
        != "supervisor terminal evidence did not arrive"
    ):
        raise PermissionError("r3 archived failed integration terminal changed")

    return {
        "authorization": authorization,
        "runtime_spec": runtime_spec,
        "launch_lease": launch_lease,
        "precommit": precommit,
        "attempt_commit": attempt_commit,
        "start_ack": start_ack,
        "sidecar": sidecar,
        "terminal": terminal,
        "removal_state": removal_state,
        "archived_roots": _archived_roots(),
        "archived_executable_bindings": archived_bindings,
        "archived_template_binding": template,
    }


def _validated_recovery_path_policy(
    archived: Mapping[str, object],
    observed: Mapping[str, object],
) -> dict[str, object]:
    """Allow only generator.late inode regeneration caused by daemon-reload."""

    archived_body = dict(archived)
    observed_body = dict(observed)
    archived_rows_value = archived_body.pop("ordered_unit_paths", None)
    observed_rows_value = observed_body.pop("ordered_unit_paths", None)
    if (
        archived_body != observed_body
        or not isinstance(archived_rows_value, list)
        or not isinstance(observed_rows_value, list)
    ):
        raise PermissionError("r3 user-unit path policy changed")
    archived_rows = {
        str(row["path"]): dict(row)
        for row in archived_rows_value
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    observed_rows = {
        str(row["path"]): dict(row)
        for row in observed_rows_value
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    archived_order = [
        str(row.get("path"))
        for row in archived_rows_value
        if isinstance(row, Mapping)
    ]
    observed_order = [
        str(row.get("path"))
        for row in observed_rows_value
        if isinstance(row, Mapping)
    ]
    if (
        len(archived_rows) != len(archived_rows_value)
        or len(observed_rows) != len(observed_rows_value)
        or set(archived_rows) != set(observed_rows)
        or archived_order != observed_order
    ):
        raise PermissionError("r3 user-unit search path set changed")
    allowed = "/run/user/1008/systemd/generator.late"
    for path, expected in archived_rows.items():
        current = observed_rows[path]
        if path != allowed:
            if current != expected:
                raise PermissionError(f"r3 user-unit path identity changed:{path}")
            continue
        expected_without_inode = dict(expected)
        current_without_inode = dict(current)
        old_inode = expected_without_inode.pop("inode", None)
        new_inode = current_without_inode.pop("inode", None)
        if (
            expected_without_inode != current_without_inode
            or isinstance(old_inode, bool)
            or not isinstance(old_inode, int)
            or old_inode <= 0
            or isinstance(new_inode, bool)
            or not isinstance(new_inode, int)
            or new_inode <= 0
        ):
            raise PermissionError(
                "r3 authorized daemon-reload generator identity changed"
            )
    return json.loads(canonical_json(dict(observed)))


def _fragment_identity(path: Path) -> dict[str, object]:
    if path != EXPECTED_FRAGMENT_PATH:
        raise PermissionError("r3 fragment is not the exact private regular file")
    data, current = _read_regular_file_snapshot(
        path,
        expected_owner_uid=EXPECTED_FRAGMENT_OWNER_UID,
        expected_mode=EXPECTED_FRAGMENT_MODE,
    )
    if current.st_nlink != EXPECTED_FRAGMENT_NLINK:
        raise PermissionError("r3 fragment link count changed")
    return {
        "fragment_path": str(path),
        "fragment_sha256": hashlib.sha256(data).hexdigest(),
        "device": current.st_dev,
        "inode": current.st_ino,
        "owner_uid": current.st_uid,
        "mode": stat.S_IMODE(current.st_mode),
        "nlink": current.st_nlink,
    }


def _expected_fragment_identity() -> dict[str, object]:
    return {
        "fragment_path": str(EXPECTED_FRAGMENT_PATH),
        "fragment_sha256": EXPECTED_FRAGMENT_SHA256,
        "device": EXPECTED_FRAGMENT_DEVICE,
        "inode": EXPECTED_FRAGMENT_INODE,
        "owner_uid": EXPECTED_FRAGMENT_OWNER_UID,
        "mode": EXPECTED_FRAGMENT_MODE,
        "nlink": EXPECTED_FRAGMENT_NLINK,
    }


def _live_context(
    *,
    runner: object = integration.run_command,
    manager_reader: Callable[[], dict[str, object]] = (
        integration.collect_manager_generation
    ),
) -> dict[str, object]:
    chain = _sealed_original_chain()
    authorization = chain["authorization"]
    observed_policy = realizer.freeze_user_unit_path_policy(
        UNIT_NAME,
        runner=runner,
        allowed_fragment=EXPECTED_FRAGMENT_PATH,
    )
    policy = _validated_recovery_path_policy(
        authorization["unit_path_policy"],
        observed_policy,
    )
    manager = manager_reader()
    integration._validate_manager_generation(manager)
    if manager != authorization["manager_generation"]:
        raise PermissionError("r3 user manager generation changed")

    rendered = authorization["rendered_fragment"]
    plan = realizer.build_realization_plan(
        unit_name=UNIT_NAME,
        unit_directory=EXPECTED_RUNTIME_UNIT_DIRECTORY,
        fragment_text=str(rendered["utf8_text"]),
        expected_fragment_sha256=EXPECTED_FRAGMENT_SHA256,
        execute_authorized=True,
        removal_authorized=True,
    )
    state = realizer.query_unit_properties(UNIT_NAME, runner=runner)
    expected_state = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "FragmentPath": str(EXPECTED_FRAGMENT_PATH),
        "DropInPaths": "",
        "Transient": "no",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
    }
    realizer.validate_realized_static_unit(plan, state)
    fragment_identity = _fragment_identity(EXPECTED_FRAGMENT_PATH)
    if (
        fragment_identity != _expected_fragment_identity()
        or state != expected_state
        or plan.fragment_path != EXPECTED_FRAGMENT_PATH
        or plan.fragment_sha256 != EXPECTED_FRAGMENT_SHA256
        or plan.execute_authorized is not True
        or plan.removal_authorized is not True
    ):
        raise PermissionError("r3 fragment is not exact inactive static state")
    return {
        **chain,
        "manager_generation": manager,
        "unit_path_policy": policy,
        "fragment_identity": fragment_identity,
        "inactive_static_state": state,
        "plan": plan,
    }


def _assert_current_bindings(
    *,
    expected_recovery_tool_binding: Mapping[str, object],
    expected_required_bindings: Mapping[str, object],
) -> None:
    if (
        _current_recovery_tool_binding()
        != dict(expected_recovery_tool_binding)
        or _current_required_bindings() != dict(expected_required_bindings)
    ):
        raise PermissionError("r3 recovery implementation changed after authorization")


def _revalidate_preunlink_state(
    *,
    plan: object,
    expected_manager_generation: Mapping[str, object],
    expected_unit_path_policy: Mapping[str, object],
    expected_inactive_static_state: Mapping[str, str],
    runner: object,
    manager_reader: Callable[[], dict[str, object]],
) -> None:
    """Recheck manager, search paths, and inactive/static state after intent."""

    manager = manager_reader()
    integration._validate_manager_generation(manager)
    if manager != dict(expected_manager_generation):
        raise PermissionError("r3 user manager changed after recovery intent")
    policy = realizer.freeze_user_unit_path_policy(
        UNIT_NAME,
        runner=runner,
        allowed_fragment=EXPECTED_FRAGMENT_PATH,
    )
    if policy != dict(expected_unit_path_policy):
        raise PermissionError("r3 user-unit path policy changed after recovery intent")
    state = realizer.query_unit_properties(UNIT_NAME, runner=runner)
    realizer.validate_realized_static_unit(plan, state)
    if state != dict(expected_inactive_static_state):
        raise PermissionError("r3 unit state changed after recovery intent")


def _revalidate_manager_generation(
    expected_manager_generation: Mapping[str, object],
    *,
    manager_reader: Callable[[], dict[str, object]],
) -> None:
    """Do not reload a different user-manager generation after unlink."""

    manager = manager_reader()
    integration._validate_manager_generation(manager)
    if manager != dict(expected_manager_generation):
        raise PermissionError("r3 user manager changed before daemon-reload")


def _observe_fragment_absent(path: Path) -> bool:
    return not os.path.lexists(path)


def _remove_exact_authorized_fragment(
    plan: object,
    *,
    expected_identity: Mapping[str, object],
    authorization: Mapping[str, object],
    intent: Mapping[str, object],
    expected_recovery_tool_binding: Mapping[str, object],
    expected_required_bindings: Mapping[str, object],
    expected_manager_generation: Mapping[str, object],
    expected_unit_path_policy: Mapping[str, object],
    expected_inactive_static_state: Mapping[str, str],
    runner: object,
    manager_reader: Callable[[], dict[str, object]],
    on_action_started: Callable[[str], None],
) -> None:
    """Unlink the authorization-bound inode after four exact fresh checks."""

    realizer.validate_integration_unit_name(plan.unit_name)
    if (
        set(expected_identity) != _FRAGMENT_IDENTITY_KEYS
        or dict(expected_identity) != _expected_fragment_identity()
        or plan.unit_name != UNIT_NAME
        or plan.unit_directory != EXPECTED_RUNTIME_UNIT_DIRECTORY
        or plan.fragment_path != EXPECTED_FRAGMENT_PATH
        or plan.fragment_sha256 != EXPECTED_FRAGMENT_SHA256
        or plan.owner_uid != EXPECTED_FRAGMENT_OWNER_UID
        or plan.execute_authorized is not True
        or plan.removal_authorized is not True
    ):
        raise PermissionError("r3 fragment removal identity is not exact")
    _assert_current_bindings(
        expected_recovery_tool_binding=expected_recovery_tool_binding,
        expected_required_bindings=expected_required_bindings,
    )

    directory_fd = realizer._open_verified_directory(
        plan.unit_directory,
        owner_uid=EXPECTED_FRAGMENT_OWNER_UID,
    )
    fragment_fd: int | None = None
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        fragment_fd = os.open(plan.unit_name, flags, dir_fd=directory_fd)

        def _check_exact(check_ordinal: int) -> None:
            opened = os.fstat(fragment_fd)
            linked = os.stat(
                plan.unit_name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )

            def _exact(value: os.stat_result) -> bool:
                return (
                    stat.S_ISREG(value.st_mode)
                    and value.st_dev == EXPECTED_FRAGMENT_DEVICE
                    and value.st_ino == EXPECTED_FRAGMENT_INODE
                    and value.st_uid == EXPECTED_FRAGMENT_OWNER_UID
                    and stat.S_IMODE(value.st_mode) == EXPECTED_FRAGMENT_MODE
                    and value.st_nlink == EXPECTED_FRAGMENT_NLINK
                )

            if (
                not _exact(opened)
                or not _exact(linked)
                or opened.st_dev != linked.st_dev
                or opened.st_ino != linked.st_ino
                or realizer._fd_sha256(fragment_fd)
                != EXPECTED_FRAGMENT_SHA256
            ):
                raise PermissionError(
                    f"r3 fragment changed at exact check {check_ordinal}"
                )

        _check_exact(1)
        _check_exact(2)
        _assert_current_bindings(
            expected_recovery_tool_binding=expected_recovery_tool_binding,
            expected_required_bindings=expected_required_bindings,
        )
        _revalidate_preunlink_state(
            plan=plan,
            expected_manager_generation=expected_manager_generation,
            expected_unit_path_policy=expected_unit_path_policy,
            expected_inactive_static_state=expected_inactive_static_state,
            runner=runner,
            manager_reader=manager_reader,
        )
        _assert_current_bindings(
            expected_recovery_tool_binding=expected_recovery_tool_binding,
            expected_required_bindings=expected_required_bindings,
        )
        _check_exact(3)
        action_started = datetime.now(timezone.utc)
        issued = _timestamp(
            authorization.get("issued_at_utc"),
            name="r3 recovery issuance",
        )
        intent_created = _timestamp(
            intent.get("created_at_utc"),
            name="r3 recovery intent",
        )
        expires = _timestamp(
            authorization.get("expires_at_utc"),
            name="r3 recovery expiry",
        )
        if not issued <= intent_created <= action_started <= expires:
            raise PermissionError("r3 recovery authorization expired before unlink")
        on_action_started(
            action_started.isoformat().replace("+00:00", "Z")
        )
        _check_exact(4)
        if datetime.now(timezone.utc) > expires:
            raise PermissionError(
                "r3 recovery authorization expired at exact unlink"
            )
        os.unlink(plan.unit_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        if fragment_fd is not None:
            os.close(fragment_fd)
        os.close(directory_fd)


def create_recovery_authorization(
    *,
    validity_seconds: int = 300,
    runner: object = integration.run_command,
    manager_reader: Callable[[], dict[str, object]] = (
        integration.collect_manager_generation
    ),
) -> dict[str, object]:
    _assert_repository()
    if (
        isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("r3 recovery authorization validity must be in [1,300]")
    if any(
        os.path.lexists(path)
        for path in (
            RECOVERY_AUTHORIZATION_PATH,
            RECOVERY_INTENT_PATH,
            RECOVERY_TERMINAL_PATH,
        )
    ):
        raise FileExistsError("r3 recovery identity is already consumed")

    tool_before = _current_recovery_tool_binding()
    required_before = _current_required_bindings()
    context = _live_context(runner=runner, manager_reader=manager_reader)
    tool_after = _current_recovery_tool_binding()
    required_after = _current_required_bindings()
    if tool_before != tool_after or required_before != required_after:
        raise PermissionError("r3 recovery implementation changed during authorization")

    issued = datetime.now(timezone.utc)
    action = {
        "ordinal": 0,
        "action": "remove-exact-runtime-static-fragment",
        "unit_name": UNIT_NAME,
        "fragment_path": str(EXPECTED_FRAGMENT_PATH),
        "then": ["daemon-reload", "verify-not-found"],
    }
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "scenario_id": SCENARIO_ID,
        "unit_name": UNIT_NAME,
        "instruction_id": INSTRUCTION_ID,
        "authorization_basis": AUTHORIZATION_BASIS,
        "authorized_uid": AUTHORIZED_UID,
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (
            issued + timedelta(seconds=validity_seconds)
        ).isoformat().replace("+00:00", "Z"),
        "archived_roots": context["archived_roots"],
        "archived_executable_bindings": context[
            "archived_executable_bindings"
        ],
        "current_recovery_tool_binding": tool_after,
        "current_required_executable_bindings": required_after,
        "manager_generation": context["manager_generation"],
        "unit_path_policy": context["unit_path_policy"],
        "fragment_identity": context["fragment_identity"],
        "inactive_static_state": context["inactive_static_state"],
        "authorized_action": action,
        "remove_authorized": True,
        "daemon_reload_authorized": True,
        "not_found_verification_authorized": True,
        "start_authorized": False,
        "stop_authorized": False,
        "enable_authorized": False,
        "automatic_retry_authorized": False,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
    }
    return _write_sealed(
        RECOVERY_AUTHORIZATION_PATH,
        body,
        fingerprint_field="recovery_authorization_fingerprint",
    )


def load_recovery_authorization(
    *,
    require_fresh: bool,
    runner: object = integration.run_command,
    manager_reader: Callable[[], dict[str, object]] = (
        integration.collect_manager_generation
    ),
) -> tuple[dict[str, object], dict[str, object]]:
    (
        authorization,
        authorization_file_sha256,
        authorization_stat_identity,
    ) = _read_recovery_sealed_snapshot(
        RECOVERY_AUTHORIZATION_PATH,
        fingerprint_field="recovery_authorization_fingerprint",
        schema=AUTHORIZATION_SCHEMA,
    )
    if set(authorization) != _AUTHORIZATION_KEYS:
        raise PermissionError("r3 recovery authorization keys changed")
    _no_payload(authorization)
    issued = _timestamp(
        authorization.get("issued_at_utc"),
        name="r3 recovery issuance",
    )
    expires = _timestamp(
        authorization.get("expires_at_utc"),
        name="r3 recovery expiry",
    )
    now = datetime.now(timezone.utc)
    expected_action = {
        "ordinal": 0,
        "action": "remove-exact-runtime-static-fragment",
        "unit_name": UNIT_NAME,
        "fragment_path": str(EXPECTED_FRAGMENT_PATH),
        "then": ["daemon-reload", "verify-not-found"],
    }
    if (
        authorization.get("scenario_id") != SCENARIO_ID
        or authorization.get("unit_name") != UNIT_NAME
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis") != AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != AUTHORIZED_UID
        or expires <= issued
        or expires - issued > timedelta(seconds=300)
        or issued > now
        or (require_fresh and now > expires)
        or authorization.get("archived_roots") != _archived_roots()
        or authorization.get("authorized_action") != expected_action
        or authorization.get("remove_authorized") is not True
        or authorization.get("daemon_reload_authorized") is not True
        or authorization.get("not_found_verification_authorized") is not True
        or authorization.get("start_authorized") is not False
        or authorization.get("stop_authorized") is not False
        or authorization.get("enable_authorized") is not False
        or authorization.get("automatic_retry_authorized") is not False
    ):
        raise PermissionError("r3 recovery authorization is stale or changed")
    expected_tool = authorization.get("current_recovery_tool_binding")
    expected_required = authorization.get(
        "current_required_executable_bindings"
    )
    if not isinstance(expected_tool, Mapping) or not isinstance(
        expected_required,
        Mapping,
    ):
        raise PermissionError("r3 recovery current bindings are malformed")
    _assert_current_bindings(
        expected_recovery_tool_binding=expected_tool,
        expected_required_bindings=expected_required,
    )
    context = _live_context(runner=runner, manager_reader=manager_reader)
    _assert_current_bindings(
        expected_recovery_tool_binding=expected_tool,
        expected_required_bindings=expected_required,
    )
    if (
        authorization.get("archived_executable_bindings")
        != context["archived_executable_bindings"]
        or authorization.get("manager_generation")
        != context["manager_generation"]
        or authorization.get("unit_path_policy")
        != context["unit_path_policy"]
        or authorization.get("fragment_identity")
        != context["fragment_identity"]
        or authorization.get("inactive_static_state")
        != context["inactive_static_state"]
    ):
        raise PermissionError("r3 recovery authorization/live closure changed")
    (
        final_authorization,
        final_authorization_file_sha256,
        final_authorization_stat_identity,
    ) = _read_recovery_sealed_snapshot(
        RECOVERY_AUTHORIZATION_PATH,
        fingerprint_field="recovery_authorization_fingerprint",
        schema=AUTHORIZATION_SCHEMA,
    )
    if (
        final_authorization != authorization
        or final_authorization_file_sha256 != authorization_file_sha256
        or final_authorization_stat_identity != authorization_stat_identity
        or (
            require_fresh
            and datetime.now(timezone.utc) > expires
        )
    ):
        raise PermissionError(
            "r3 recovery authorization changed during live closure"
        )
    context = dict(context)
    context["recovery_authorization_file_sha256"] = (
        final_authorization_file_sha256
    )
    context["recovery_authorization_stat_identity"] = (
        final_authorization_stat_identity
    )
    return authorization, context


def execute_recovery(
    *,
    execute: bool,
    timeout_seconds: float = 10.0,
    runner: object = integration.run_command,
    manager_reader: Callable[[], dict[str, object]] = (
        integration.collect_manager_generation
    ),
) -> dict[str, object]:
    if not execute:
        raise PermissionError("explicit r3 recovery execution is required")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise ValueError("r3 recovery timeout must be finite and positive")
    if os.path.lexists(RECOVERY_INTENT_PATH) or os.path.lexists(
        RECOVERY_TERMINAL_PATH
    ):
        raise FileExistsError("r3 recovery execution identity is consumed")

    authorization, context = load_recovery_authorization(
        require_fresh=True,
        runner=runner,
        manager_reader=manager_reader,
    )
    (
        intent_authorization,
        intent_authorization_file_sha256,
        intent_authorization_stat_identity,
    ) = _read_recovery_sealed_snapshot(
        RECOVERY_AUTHORIZATION_PATH,
        fingerprint_field="recovery_authorization_fingerprint",
        schema=AUTHORIZATION_SCHEMA,
    )
    if (
        intent_authorization != authorization
        or intent_authorization_file_sha256
        != context["recovery_authorization_file_sha256"]
        or intent_authorization_stat_identity
        != context["recovery_authorization_stat_identity"]
    ):
        raise PermissionError(
            "r3 recovery authorization changed before intent"
        )
    intent = _write_sealed(
        RECOVERY_INTENT_PATH,
        {
            "schema_version": INTENT_SCHEMA,
            "created_at_utc": _utc_now(),
            "scenario_id": SCENARIO_ID,
            "unit_name": UNIT_NAME,
            "recovery_authorization_path": str(RECOVERY_AUTHORIZATION_PATH),
            "recovery_authorization_file_sha256": (
                intent_authorization_file_sha256
            ),
            "recovery_authorization_fingerprint": authorization[
                "recovery_authorization_fingerprint"
            ],
            "fragment_identity": context["fragment_identity"],
            "inactive_static_state": context["inactive_static_state"],
            "authorized_action": authorization["authorized_action"],
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_accessed": False,
        },
        fingerprint_field="recovery_intent_fingerprint",
    )
    if set(intent) != _INTENT_KEYS:
        raise RuntimeError("r3 recovery intent schema changed")

    completed: list[str] = []
    post_state: dict[str, str] | None = None
    action_started_at_utc: str | None = None
    error: BaseException | None = None

    def _record_action_started(value: str) -> None:
        nonlocal action_started_at_utc
        if action_started_at_utc is not None:
            raise RuntimeError("r3 recovery action-start trace repeated")
        _timestamp(value, name="r3 recovery action start")
        action_started_at_utc = value

    try:
        _remove_exact_authorized_fragment(
            context["plan"],
            expected_identity=context["fragment_identity"],
            authorization=authorization,
            intent=intent,
            expected_recovery_tool_binding=authorization[
                "current_recovery_tool_binding"
            ],
            expected_required_bindings=authorization[
                "current_required_executable_bindings"
            ],
            expected_manager_generation=authorization[
                "manager_generation"
            ],
            expected_unit_path_policy=authorization["unit_path_policy"],
            expected_inactive_static_state=authorization[
                "inactive_static_state"
            ],
            runner=runner,
            manager_reader=manager_reader,
            on_action_started=_record_action_started,
        )
        completed.append("remove-exact-runtime-static-fragment")
        _revalidate_manager_generation(
            authorization["manager_generation"],
            manager_reader=manager_reader,
        )
        realizer.daemon_reload(execute=True, runner=runner)
        completed.append("daemon-reload")
        post_state = realizer.wait_until_unit_not_found(
            UNIT_NAME,
            query=lambda unit: realizer.query_unit_properties(
                unit,
                runner=runner,
            ),
            timeout_seconds=float(timeout_seconds),
            poll_seconds=0.01,
        )
        completed.append("verify-not-found")
    except BaseException as caught:
        error = caught

    fragment_absent = False
    try:
        fragment_absent = _observe_fragment_absent(
            context["plan"].fragment_path
        )
    except BaseException as caught:
        if error is None:
            error = caught
    terminal_created = _utc_now()
    terminal = _write_sealed(
        RECOVERY_TERMINAL_PATH,
        {
            "schema_version": TERMINAL_SCHEMA,
            "created_at_utc": terminal_created,
            "scenario_id": SCENARIO_ID,
            "unit_name": UNIT_NAME,
            "recovery_authorization_fingerprint": authorization[
                "recovery_authorization_fingerprint"
            ],
            "recovery_intent_fingerprint": intent[
                "recovery_intent_fingerprint"
            ],
            "action_started_at_utc": action_started_at_utc,
            "completed_actions": completed,
            "fragment_absent": fragment_absent,
            "post_removal_unit_state": post_state,
            "passed": (
                error is None
                and fragment_absent
                and completed
                == [
                    "remove-exact-runtime-static-fragment",
                    "daemon-reload",
                    "verify-not-found",
                ]
            ),
            "error_type": (
                type(error).__name__ if error is not None else None
            ),
            "error_message": str(error) if error is not None else None,
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_accessed": False,
        },
        fingerprint_field="recovery_terminal_fingerprint",
    )
    if set(terminal) != _RECOVERY_TERMINAL_KEYS:
        raise RuntimeError("r3 recovery terminal schema changed")
    intent_time = _timestamp(intent["created_at_utc"], name="r3 recovery intent")
    terminal_time = _timestamp(
        terminal["created_at_utc"],
        name="r3 recovery terminal",
    )
    if action_started_at_utc is None:
        if terminal_time < intent_time:
            raise RuntimeError("r3 recovery failure chronology changed")
    elif not (
        intent_time
        <= _timestamp(
            action_started_at_utc,
            name="r3 recovery action start",
        )
        <= terminal_time
    ):
        raise RuntimeError("r3 recovery action chronology changed")
    if error is not None:
        raise error
    if terminal["passed"] is not True:
        raise RuntimeError("r3 recovery did not reach exact terminal PASS")
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--validity-seconds", type=int, default=300)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--execute-authorized-removal", action="store_true")
    apply.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "authorize":
        result = create_recovery_authorization(
            validity_seconds=arguments.validity_seconds
        )
        print(
            json.dumps(
                {
                    "path": str(RECOVERY_AUTHORIZATION_PATH),
                    "recovery_authorization_fingerprint": result[
                        "recovery_authorization_fingerprint"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    result = execute_recovery(
        execute=arguments.execute_authorized_removal,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "path": str(RECOVERY_TERMINAL_PATH),
                "passed": result["passed"],
                "recovery_terminal_fingerprint": result[
                    "recovery_terminal_fingerprint"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
