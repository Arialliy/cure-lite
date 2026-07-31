#!/usr/bin/env python3
"""Data-blind runtime environment evidence for CURE-Lite v24.

This module is deliberately standard-library-only.  It may inspect the user
systemd manager, procfs, cgroup metadata, and ``nvidia-smi`` metadata.  It has
no dataset/model imports and no command capable of changing systemd or GPU
state.

The public CLI exposes read-only audit/stability operations plus create-once
policy/receipt sealing.  GPU lease functions are library primitives for the
supervisor and are intentionally not exposed as CLI commands.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence


ENVIRONMENT_INVENTORY_SCHEMA = (
    "cure-lite-v24-runtime-environment-inventory-v1"
)
ENVIRONMENT_RECEIPT_SCHEMA = (
    "cure-lite-v24-runtime-environment-audit-receipt-v1"
)
ENVIRONMENT_POLICY_SCHEMA = (
    "cure-lite-v24-runtime-environment-policy-v1"
)
ENVIRONMENT_SINGLE_AUDIT_SCHEMA = (
    "cure-lite-v24-runtime-environment-single-audit-v1"
)
ENVIRONMENT_STABILITY_RECEIPT_SCHEMA = (
    "cure-lite-v24-runtime-environment-stability-receipt-v1"
)
CLEANUP_RECEIPT_SCHEMA = "cure-lite-v24-runtime-cleanup-receipt-v2"
GPU0_CONFLICT_UNIT = (
    "confa-v41-mshnet-nudt-clean-formal-20260718-v1.service"
)
NORMAL_CLEANUP_MODE = "runtime-mask-stop"
RECOVERY_CLEANUP_MODE = "partial-runtime-mask-stop-recovery"
NORMAL_GUARD_MODE = "effective-runtime-mask"
RECOVERY_GUARD_MODE = (
    "ineffective-runtime-mask-symlink-plus-explicit-stop"
)
NORMAL_QUIESCENCE_MODE = "masked-runtime-inactive-dead"
RECOVERY_QUIESCENCE_MODE = (
    "enabled-inactive-dead-with-sealed-runtime-symlink-guard"
)
_CLEANUP_FINAL_KEYS = {
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
_NORMAL_GUARD_KEYS = {
    "mode", "unit_name", "observed_unit_file_state",
}
_RECOVERY_GUARD_KEYS = {
    "mode", "unit_name", "path", "target", "owner_uid", "device",
    "inode", "observed_unit_file_state",
}
_PARTIAL_LINEAGE_KEYS = {
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
_PARTIAL_LINEAGE_ROOT_FINGERPRINT_FIELDS = {
    "plan": "plan_fingerprint",
    "original_authorization": "authorization_fingerprint",
    "original_intent": "intent_fingerprint",
    "original_terminal_failure": "terminal_failure_fingerprint",
    "recovery_authorization": "recovery_authorization_fingerprint",
    "recovery_intent": "recovery_intent_fingerprint",
    "recovery_action_receipt": "recovery_action_receipt_fingerprint",
}
GPU_DOUBLE_SNAPSHOT_SCHEMA = (
    "cure-lite-v24-runtime-gpu-double-snapshot-v1"
)
ACTIVATION_CLOSURE_FIELDS = (
    "TriggeredBy",
    "Triggers",
    "WantedBy",
    "RequiredBy",
    "PartOf",
)
NORMAL_CONFLICT_ACTIVATION_CLOSURE = {
    field: "" for field in ACTIVATION_CLOSURE_FIELDS
}
RECOVERY_CONFLICT_ACTIVATION_CLOSURE = {
    "TriggeredBy": "",
    "Triggers": "",
    "WantedBy": "default.target",
    "RequiredBy": "",
    "PartOf": "",
}
GPU_LEASE_SCHEMA = "cure-lite-v24-gpu-lease-v1"
GPU_LEASE_RELEASE_SCHEMA = (
    "cure-lite-v24-gpu-lease-release-complete-v1"
)

SYSTEMCTL_PATH = "/usr/bin/systemctl"
NVIDIA_SMI_PATH = "/usr/bin/nvidia-smi"
READ_ONLY_EXECUTABLES = frozenset({SYSTEMCTL_PATH, NVIDIA_SMI_PATH})

GPU_QUERY_ARGV = (
    NVIDIA_SMI_PATH,
    "--query-gpu=index,uuid,pci.bus_id,compute_mode,mig.mode.current,"
    "driver_version",
    "--format=csv,noheader,nounits",
)
GPU_APPS_QUERY_ARGV = (
    NVIDIA_SMI_PATH,
    "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits",
)

_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{16,}$")
_PCI_BUS_RE = re.compile(
    r"^(?:[0-9a-fA-F]{4,8}:)?[0-9a-fA-F]{2}:"
    r"[0-9a-fA-F]{2}\.[0-7]$"
)
_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:\\-]+\.service$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)


def canonical_json(value: object) -> str:
    """Return deterministic finite JSON."""

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
    """Compare JSON-like evidence without Python bool/int/float coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if len(left) != len(right):
            return False
        for left_key, left_value in left.items():
            matching_keys = [
                right_key
                for right_key in right
                if _deep_exact_equal(left_key, right_key)
            ]
            if (
                len(matching_keys) != 1
                or not _deep_exact_equal(
                    left_value,
                    right[matching_keys[0]],
                )
            ):
                return False
        return True
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _deep_exact_equal(first, second)
            for first, second in zip(left, right)
        )
    return bool(left == right)


def _is_nonbool_int(value: object, *, minimum: int = 0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= minimum
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_systemd_duration_usec(value: str) -> int:
    """Parse one C-locale systemd duration and conservatively ceil to usec."""

    if value == "0":
        return 0
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("systemd duration is malformed")
    units = {
        "us": 1,
        "usec": 1,
        "µs": 1,
        "ms": 1_000,
        "s": 1_000_000,
        "min": 60_000_000,
        "h": 3_600_000_000,
        "d": 86_400_000_000,
        "w": 604_800_000_000,
    }
    token_pattern = re.compile(
        r"(?P<amount>[0-9]+(?:\.[0-9]+)?)\s*"
        r"(?P<unit>usec|us|µs|ms|min|s|h|d|w)"
    )
    position = 0
    total = Decimal(0)
    matched = False
    try:
        for match in token_pattern.finditer(value):
            if value[position:match.start()].strip():
                raise ValueError("systemd duration has an unknown token")
            matched = True
            amount = Decimal(match.group("amount"))
            total += amount * units[match.group("unit")]
            position = match.end()
    except (InvalidOperation, KeyError) as error:
        raise ValueError("systemd duration is malformed") from error
    if not matched or value[position:].strip():
        raise ValueError("systemd duration has an unknown token")
    integral = total.to_integral_value(rounding=ROUND_CEILING)
    if integral < 0 or integral > 2**63 - 1:
        raise ValueError("systemd duration is outside the supported range")
    return int(integral)


def current_runtime_toolchain_binding() -> dict[str, dict[str, object]]:
    """Bind the exact auditor and read-only executables used by this process."""

    entries = (
        ("runtime_environment", Path(__file__).resolve()),
        ("python", Path(sys.executable).resolve()),
        ("systemctl", Path(SYSTEMCTL_PATH).resolve()),
        ("nvidia_smi", Path(NVIDIA_SMI_PATH).resolve()),
    )
    bindings: dict[str, dict[str, object]] = {}
    for label, path in entries:
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise PermissionError(f"{label} executable binding is not regular")
        bindings[label] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
        }
    return bindings


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_gpu_uuid(value: object) -> bool:
    return isinstance(value, str) and _UUID_RE.fullmatch(value) is not None


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _new_absolute_path(value: str | Path, *, name: str) -> Path:
    """Validate the lexical part of a path before opening its parent."""

    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if path.name in {"", ".", ".."} or path.parent / path.name != path:
        raise ValueError(f"{name} basename is invalid")
    return path


def _validate_directory_metadata(
    metadata: os.stat_result,
    *,
    name: str,
    owner_uid: int,
    private: bool,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{name} must be a canonical directory")
    if metadata.st_uid != owner_uid:
        raise PermissionError(f"{name} owner changed")
    mode = stat.S_IMODE(metadata.st_mode)
    if private and mode != 0o700:
        raise PermissionError(f"{name} must have mode 0700")
    if not private and mode & 0o022:
        raise PermissionError(f"{name} is group/world writable")


def _verify_parent_path_generation(
    parent: Path,
    descriptor: int,
    *,
    expected_device: int,
    expected_inode: int,
    name: str,
    owner_uid: int,
    private: bool,
) -> os.stat_result:
    """Bind a saved directory fd to the generation still named by ``parent``."""

    descriptor_metadata = os.fstat(descriptor)
    _validate_directory_metadata(
        descriptor_metadata,
        name=name,
        owner_uid=owner_uid,
        private=private,
    )
    if (
        (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
        != (expected_device, expected_inode)
    ):
        raise PermissionError(f"{name} generation changed")
    try:
        resolved = parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PermissionError(f"{name} generation changed") from error
    if resolved != parent:
        raise PermissionError(f"{name} is no longer canonical")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    recheck_descriptor = os.open(parent, flags)
    try:
        recheck_metadata = os.fstat(recheck_descriptor)
        path_metadata = parent.lstat()
        _validate_directory_metadata(
            recheck_metadata,
            name=name,
            owner_uid=owner_uid,
            private=private,
        )
        _validate_directory_metadata(
            path_metadata,
            name=name,
            owner_uid=owner_uid,
            private=private,
        )
        if (
            (recheck_metadata.st_dev, recheck_metadata.st_ino)
            != (expected_device, expected_inode)
            or (path_metadata.st_dev, path_metadata.st_ino)
            != (expected_device, expected_inode)
        ):
            raise PermissionError(f"{name} generation changed")
    except BaseException:
        _close_descriptors_best_effort(recheck_descriptor)
        raise
    close_error = _close_descriptors_best_effort(recheck_descriptor)
    if close_error is not None:
        raise close_error
    return descriptor_metadata


def _open_stable_parent_directory(
    target: Path,
    *,
    name: str,
    owner_uid: int,
    private: bool,
) -> tuple[int, os.stat_result]:
    """Open and generation-bind the canonical parent of one new/existing file."""

    parent = target.parent
    try:
        before = parent.lstat()
    except OSError as error:
        raise ValueError(f"{name} must be a canonical directory") from error
    _validate_directory_metadata(
        before,
        name=name,
        owner_uid=owner_uid,
        private=private,
    )
    try:
        resolved = parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{name} must be a canonical directory") from error
    if resolved != parent:
        raise ValueError(f"{name} must be a canonical directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(parent, flags)
    try:
        opened = os.fstat(descriptor)
        _validate_directory_metadata(
            opened,
            name=name,
            owner_uid=owner_uid,
            private=private,
        )
        if (
            (before.st_dev, before.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise PermissionError(f"{name} generation changed while opening")
        _verify_parent_path_generation(
            parent,
            descriptor,
            expected_device=opened.st_dev,
            expected_inode=opened.st_ino,
            name=name,
            owner_uid=owner_uid,
            private=private,
        )
        return descriptor, opened
    except BaseException:
        _close_descriptors_best_effort(descriptor)
        raise


def _read_all_from_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    blocks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(blocks)
        blocks.append(block)


def _write_all_to_fd(descriptor: int, encoded: bytes) -> None:
    view = memoryview(encoded)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short create-once file write")
        written += count


def _close_descriptors_best_effort(
    *descriptors: int,
) -> BaseException | None:
    """Attempt every distinct fd close and return the first close error."""

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


def _linked_entry_metadata(
    parent_descriptor: int,
    basename: str,
) -> os.stat_result:
    return os.stat(
        basename,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )


def _same_stable_file_metadata(
    first: os.stat_result,
    second: os.stat_result,
) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_uid",
        "st_nlink",
        "st_mode",
        "st_size",
        "st_mtime_ns",
    )
    return all(getattr(first, field) == getattr(second, field) for field in fields)


def _canonical_existing_directory(
    path: Path,
    *,
    name: str,
    owner_uid: int | None = None,
    private: bool = False,
) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if (
        not path.is_dir()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise ValueError(f"{name} must be a canonical directory")
    metadata = path.stat()
    if owner_uid is not None and metadata.st_uid != owner_uid:
        raise PermissionError(f"{name} owner changed")
    mode = stat.S_IMODE(metadata.st_mode)
    if private and mode != 0o700:
        raise PermissionError(f"{name} must have mode 0700")
    if not private and mode & 0o022:
        raise PermissionError(f"{name} is group/world writable")
    return path


def _canonical_new_path(
    value: str | Path,
    *,
    name: str,
    private_parent: bool = False,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    _canonical_existing_directory(
        path.parent,
        name=f"{name} parent",
        owner_uid=os.getuid(),
        private=private_parent,
    )
    if path.name in {"", ".", ".."} or path.parent / path.name != path:
        raise ValueError(f"{name} basename is invalid")
    return path


def write_create_once_receipt(
    path: str | Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str = "receipt_fingerprint",
    while_open_guard: Callable[
        [int, int, os.stat_result, os.stat_result],
        None,
    ]
    | None = None,
) -> dict[str, object]:
    """Seal one receipt, optionally cross-checking while both receipt fds live."""

    target = _new_absolute_path(path, name="receipt path")
    if (
        not isinstance(fingerprint_field, str)
        or not fingerprint_field
        or fingerprint_field
        in {"path", "device", "inode", "size", "mtime_ns", "file_sha256"}
    ):
        raise ValueError("receipt fingerprint field is invalid")
    if while_open_guard is not None and not callable(while_open_guard):
        raise ValueError("receipt while-open guard must be callable")
    if fingerprint_field in body:
        raise ValueError("receipt body already contains its fingerprint")
    materialized = dict(body)
    payload = {
        **materialized,
        fingerprint_field: stable_fingerprint(materialized),
    }
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    selected_uid = os.getuid()
    parent_descriptor, parent_metadata = _open_stable_parent_directory(
        target,
        name="receipt path parent",
        owner_uid=selected_uid,
        private=False,
    )
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            flags,
            0o444,
            dir_fd=parent_descriptor,
        )
        _write_all_to_fd(descriptor, encoded)
        os.fsync(descriptor)
        before_fd = os.fstat(descriptor)
        observed = _read_all_from_fd(descriptor)
        after_fd = os.fstat(descriptor)
        linked = _linked_entry_metadata(parent_descriptor, target.name)
        observed_sha256 = hashlib.sha256(observed).hexdigest()
        expected_sha256 = hashlib.sha256(encoded).hexdigest()
        if (
            not _same_stable_file_metadata(before_fd, after_fd)
            or not _same_stable_file_metadata(after_fd, linked)
            or not stat.S_ISREG(after_fd.st_mode)
            or after_fd.st_uid != os.getuid()
            or after_fd.st_nlink != 1
            or stat.S_IMODE(after_fd.st_mode) != 0o444
            or observed != encoded
            or observed_sha256 != expected_sha256
        ):
            raise RuntimeError("create-once receipt failed self-verification")
        os.fsync(parent_descriptor)
        _verify_parent_path_generation(
            target.parent,
            parent_descriptor,
            expected_device=parent_metadata.st_dev,
            expected_inode=parent_metadata.st_ino,
            name="receipt path parent",
            owner_uid=selected_uid,
            private=False,
        )
        if while_open_guard is not None:
            while_open_guard(
                descriptor,
                parent_descriptor,
                after_fd,
                parent_metadata,
            )
            guarded_before_fd = os.fstat(descriptor)
            guarded_observed = _read_all_from_fd(descriptor)
            guarded_after_fd = os.fstat(descriptor)
            guarded_linked = _linked_entry_metadata(
                parent_descriptor,
                target.name,
            )
            if (
                not _same_stable_file_metadata(
                    after_fd,
                    guarded_before_fd,
                )
                or not _same_stable_file_metadata(
                    guarded_before_fd,
                    guarded_after_fd,
                )
                or not _same_stable_file_metadata(
                    guarded_after_fd,
                    guarded_linked,
                )
                or guarded_observed != observed
                or hashlib.sha256(guarded_observed).hexdigest()
                != expected_sha256
            ):
                raise RuntimeError(
                    "create-once receipt changed during while-open guard"
                )
            _verify_parent_path_generation(
                target.parent,
                parent_descriptor,
                expected_device=parent_metadata.st_dev,
                expected_inode=parent_metadata.st_ino,
                name="receipt path parent",
                owner_uid=selected_uid,
                private=False,
            )
    except BaseException:
        # A partial create-once artifact remains evidence and is never removed.
        try:
            if descriptor >= 0:
                os.fsync(descriptor)
        except BaseException:
            pass
        try:
            os.fsync(parent_descriptor)
        except BaseException:
            pass
        _close_descriptors_best_effort(
            descriptor,
            parent_descriptor,
        )
        raise
    close_error = _close_descriptors_best_effort(
        descriptor,
        parent_descriptor,
    )
    if close_error is not None:
        raise close_error
    return payload


def load_sealed_receipt_with_evidence(
    path: str | Path,
    *,
    fingerprint_field: str = "receipt_fingerprint",
) -> tuple[dict[str, object], dict[str, object]]:
    """Load a sealed receipt through one fd and bind its stable identity."""

    target = Path(path).absolute()
    target = _new_absolute_path(target, name="input receipt path")
    if (
        not isinstance(fingerprint_field, str)
        or not fingerprint_field
        or fingerprint_field
        in {"path", "device", "inode", "size", "mtime_ns", "file_sha256"}
    ):
        raise ValueError("receipt fingerprint field is invalid")
    selected_uid = os.getuid()
    parent_descriptor, parent_metadata = _open_stable_parent_directory(
        target,
        name="input receipt parent",
        owner_uid=selected_uid,
        private=False,
    )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        before_path = _linked_entry_metadata(
            parent_descriptor,
            target.name,
        )
        if (
            not stat.S_ISREG(before_path.st_mode)
            or before_path.st_uid != selected_uid
            or before_path.st_nlink != 1
            or stat.S_IMODE(before_path.st_mode) != 0o444
        ):
            raise PermissionError("input receipt is not sealed")
        descriptor = os.open(
            target.name,
            flags,
            dir_fd=parent_descriptor,
        )
        before_fd = os.fstat(descriptor)
        raw = _read_all_from_fd(descriptor)
        after_fd = os.fstat(descriptor)
        after_path = _linked_entry_metadata(
            parent_descriptor,
            target.name,
        )
        if (
            not _same_stable_file_metadata(before_path, before_fd)
            or not _same_stable_file_metadata(before_fd, after_fd)
            or not _same_stable_file_metadata(after_fd, after_path)
        ):
            raise PermissionError("input receipt changed while being read")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("input receipt is invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("input receipt must contain an object")
        if raw != (canonical_json(payload) + "\n").encode("utf-8"):
            raise ValueError("input receipt is not canonical JSON")
        body = dict(payload)
        fingerprint = body.pop(fingerprint_field, None)
        if (
            not _is_sha256(fingerprint)
            or fingerprint != stable_fingerprint(body)
        ):
            raise PermissionError("input receipt fingerprint is invalid")
        final_fd = os.fstat(descriptor)
        final_path = _linked_entry_metadata(
            parent_descriptor,
            target.name,
        )
        if (
            not _same_stable_file_metadata(after_fd, final_fd)
            or not _same_stable_file_metadata(final_fd, final_path)
        ):
            raise PermissionError("input receipt changed while being read")
        _verify_parent_path_generation(
            target.parent,
            parent_descriptor,
            expected_device=parent_metadata.st_dev,
            expected_inode=parent_metadata.st_ino,
            name="input receipt parent",
            owner_uid=selected_uid,
            private=False,
        )
        evidence: dict[str, object] = {
            "path": str(target),
            "device": final_fd.st_dev,
            "inode": final_fd.st_ino,
            "size": final_fd.st_size,
            "mtime_ns": final_fd.st_mtime_ns,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            fingerprint_field: fingerprint,
        }
        return payload, evidence
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _validate_sealed_receipt_evidence(
    value: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    evidence = dict(value)
    expected = {
        "path",
        "device",
        "inode",
        "size",
        "mtime_ns",
        "file_sha256",
        fingerprint_field,
    }
    if (
        not isinstance(fingerprint_field, str)
        or not fingerprint_field
        or set(evidence) != expected
        or not isinstance(evidence.get("path"), str)
        or not Path(evidence["path"]).is_absolute()
        or not _is_sha256(evidence.get("file_sha256"))
        or not _is_sha256(evidence.get(fingerprint_field))
        or any(
            isinstance(evidence.get(field), bool)
            or not isinstance(evidence.get(field), int)
            or evidence[field] < 0
            for field in ("device", "inode", "size", "mtime_ns")
        )
    ):
        raise ValueError("sealed receipt evidence is malformed")
    return evidence


def load_sealed_receipt(
    path: str | Path,
    *,
    fingerprint_field: str = "receipt_fingerprint",
) -> dict[str, object]:
    """Load one canonical, owner-bound, immutable fingerprinted receipt."""

    payload, _ = load_sealed_receipt_with_evidence(
        path,
        fingerprint_field=fingerprint_field,
    )
    return payload


def verify_sealed_receipt_evidence(
    path: str | Path,
    expected_evidence: Mapping[str, object],
    *,
    fingerprint_field: str = "receipt_fingerprint",
) -> None:
    """Fail if a root path no longer names the exact loaded sealed receipt."""

    expected = _validate_sealed_receipt_evidence(
        expected_evidence,
        fingerprint_field=fingerprint_field,
    )
    _, observed = load_sealed_receipt_with_evidence(
        path,
        fingerprint_field=fingerprint_field,
    )
    if observed != expected:
        raise PermissionError("sealed receipt root changed during stability gate")


def _validated_nrestarts(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdigit()
        or str(int(value)) != value
    ):
        raise ValueError(f"{name} NRestarts is malformed")
    return value


def _expected_conflict_activation_closure(
    cleanup_mode: str,
) -> dict[str, str]:
    if cleanup_mode == NORMAL_CLEANUP_MODE:
        return dict(NORMAL_CONFLICT_ACTIVATION_CLOSURE)
    if cleanup_mode == RECOVERY_CLEANUP_MODE:
        return dict(RECOVERY_CONFLICT_ACTIVATION_CLOSURE)
    raise ValueError("cleanup mode has no activation closure")


def _validate_partial_lineage(
    value: object,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("recovery partial lineage must be an object")
    lineage = json.loads(canonical_json(dict(value)))
    if set(lineage) != _PARTIAL_LINEAGE_KEYS:
        raise ValueError("recovery partial lineage closed schema changed")
    if (
        lineage["legacy_runtime_mask_may_remain_false_reconciled"] is not True
        or lineage["original_stop_dispatched"] is not False
    ):
        raise PermissionError("recovery partial lineage semantics changed")
    for name, fingerprint_field in (
        _PARTIAL_LINEAGE_ROOT_FINGERPRINT_FIELDS.items()
    ):
        root = lineage[name]
        if not isinstance(root, Mapping):
            raise ValueError(f"recovery lineage root is malformed:{name}")
        evidence = dict(root)
        if (
            set(evidence)
            != {"path", "file_sha256", "fingerprint_field", "fingerprint"}
            or not isinstance(evidence.get("path"), str)
            or not Path(evidence["path"]).is_absolute()
            or not _is_sha256(evidence.get("file_sha256"))
            or evidence.get("fingerprint_field") != fingerprint_field
            or not _is_sha256(evidence.get("fingerprint"))
        ):
            raise ValueError(f"recovery lineage root is malformed:{name}")
    return lineage


def verify_partial_lineage_roots(
    partial_lineage: Mapping[str, object],
) -> None:
    """Re-read every recovery root without accepting path/content drift."""

    lineage = _validate_partial_lineage(partial_lineage)
    for name, fingerprint_field in (
        _PARTIAL_LINEAGE_ROOT_FINGERPRINT_FIELDS.items()
    ):
        expected = dict(lineage[name])
        payload, observed = load_sealed_receipt_with_evidence(
            expected["path"],
            fingerprint_field=fingerprint_field,
        )
        if (
            observed["file_sha256"] != expected["file_sha256"]
            or payload.get(fingerprint_field) != expected["fingerprint"]
        ):
            raise PermissionError(
                f"recovery lineage root changed:{name}"
            )


def inspect_recovery_activation_guard(
    expected_guard: Mapping[str, object],
) -> dict[str, object]:
    """Read one runtime-mask symlink by lstat/readlink, never following it."""

    guard = dict(expected_guard)
    if set(guard) != _RECOVERY_GUARD_KEYS:
        raise ValueError("recovery activation guard closed schema changed")
    path = Path(str(guard["path"]))
    metadata = path.lstat()
    target = os.readlink(path)
    observation = {
        "mode": RECOVERY_GUARD_MODE,
        "unit_name": guard["unit_name"],
        "path": str(path),
        "target": target,
        "owner_uid": metadata.st_uid,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "observed_unit_file_state": guard["observed_unit_file_state"],
        "file_type": "symlink" if stat.S_ISLNK(metadata.st_mode) else "other",
    }
    expected_observation = {
        **guard,
        "file_type": "symlink",
    }
    if observation != expected_observation:
        raise PermissionError("recovery activation guard identity changed")
    return observation


def validate_cleanup_receipt_for_environment(
    value: Mapping[str, object],
    *,
    uid: int,
    conflict_unit_ids: Sequence[str],
) -> dict[str, object]:
    """Validate the exact normal or one narrowly authorized recovery closure."""

    payload = json.loads(canonical_json(dict(value)))
    body = dict(payload)
    fingerprint = body.get("cleanup_receipt_fingerprint")
    fingerprint_body = dict(body)
    fingerprint_body.pop("cleanup_receipt_fingerprint", None)
    conflicts = _validated_unit_tuple(
        conflict_unit_ids,
        name="cleanup conflict",
    )
    if (
        set(body) != _CLEANUP_FINAL_KEYS
        or body.get("schema_version") != CLEANUP_RECEIPT_SCHEMA
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(fingerprint_body)
        or body.get("passed") is not True
        or body.get("payload_authority") != "none"
        or any(
            body.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or not _is_sha256(body.get("intent_fingerprint"))
        or not isinstance(body.get("action_receipt_fingerprints"), list)
        or any(
            not _is_sha256(item)
            for item in body["action_receipt_fingerprints"]
        )
        or len(set(body["action_receipt_fingerprints"]))
        != len(body["action_receipt_fingerprints"])
        or not isinstance(body.get("after"), Mapping)
    ):
        raise ValueError("cleanup final receipt closed schema changed")
    after = {
        str(name): dict(shadow)
        for name, shadow in dict(body["after"]).items()
    }
    if any(unit not in after for unit in conflicts):
        raise PermissionError("cleanup receipt omits a conflict unit")

    mode = body.get("cleanup_mode")
    guard_value = body.get("activation_guard")
    if not isinstance(guard_value, Mapping):
        raise ValueError("cleanup activation guard must be an object")
    guard = dict(guard_value)
    if mode == NORMAL_CLEANUP_MODE:
        if (
            set(guard) != _NORMAL_GUARD_KEYS
            or len(body["action_receipt_fingerprints"]) != 2
            or len(conflicts) != 1
            or guard
            != {
                "mode": NORMAL_GUARD_MODE,
                "unit_name": conflicts[0],
                "observed_unit_file_state": "masked-runtime",
            }
            or body.get("partial_lineage") is not None
        ):
            raise PermissionError("normal cleanup activation closure changed")
        quiescence_mode = NORMAL_QUIESCENCE_MODE
        required_unit_file_state = "masked-runtime"
        partial_lineage = None
    elif mode == RECOVERY_CLEANUP_MODE:
        expected_path = (
            f"/run/user/{uid}/systemd/user/{GPU0_CONFLICT_UNIT}"
        )
        if (
            conflicts != (GPU0_CONFLICT_UNIT,)
            or len(body["action_receipt_fingerprints"]) != 1
            or set(guard) != _RECOVERY_GUARD_KEYS
            or guard.get("mode") != RECOVERY_GUARD_MODE
            or guard.get("unit_name") != GPU0_CONFLICT_UNIT
            or guard.get("path") != expected_path
            or guard.get("target") != "/dev/null"
            or guard.get("owner_uid") != uid
            or guard.get("observed_unit_file_state") != "enabled"
            or isinstance(guard.get("device"), bool)
            or not isinstance(guard.get("device"), int)
            or guard["device"] < 0
            or isinstance(guard.get("inode"), bool)
            or not isinstance(guard.get("inode"), int)
            or guard["inode"] <= 0
        ):
            raise PermissionError("recovery activation guard changed")
        partial_lineage = _validate_partial_lineage(
            body.get("partial_lineage")
        )
        quiescence_mode = RECOVERY_QUIESCENCE_MODE
        required_unit_file_state = "enabled"
    else:
        raise PermissionError("cleanup mode is not authorized")

    nrestarts: list[tuple[str, str]] = []
    expected_activation_closure = _expected_conflict_activation_closure(
        str(mode)
    )
    for unit in conflicts:
        shadow = after[unit]
        if (
            shadow.get("LoadState") != "loaded"
            or shadow.get("ActiveState") != "inactive"
            or shadow.get("SubState") != "dead"
            or shadow.get("UnitFileState") != required_unit_file_state
            or {
                field: shadow.get(field)
                for field in ACTIVATION_CLOSURE_FIELDS
            }
            != expected_activation_closure
        ):
            raise PermissionError("cleanup conflict unit is not quiescent")
        nrestarts.append(
            (
                unit,
                _validated_nrestarts(
                    shadow.get("NRestarts"),
                    name=f"cleanup {unit}",
                ),
            )
        )
    return {
        "payload": payload,
        "cleanup_mode": mode,
        "quiescence_mode": quiescence_mode,
        "activation_guard": guard,
        "partial_lineage": partial_lineage,
        "cleanup_nrestarts_baseline": tuple(nrestarts),
    }


def fixed_user_manager_environment(uid: int | None = None) -> dict[str, str]:
    """Return an exact environment for querying one user's manager."""

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
    """Verify that the fixed D-Bus endpoint is a private socket for ``uid``."""

    selected_uid = os.getuid() if uid is None else uid
    if (
        isinstance(selected_uid, bool)
        or not isinstance(selected_uid, int)
        or selected_uid < 0
    ):
        raise ValueError("uid must be a nonnegative integer")
    runtime = (run_user_root / str(selected_uid)).absolute()
    _canonical_existing_directory(
        runtime,
        name="user runtime directory",
        owner_uid=selected_uid,
        private=True,
    )
    runtime_metadata = runtime.stat()
    bus = runtime / "bus"
    metadata = bus.lstat()
    final_runtime_metadata = runtime.stat()
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != selected_uid
        or bus.is_symlink()
        or bus.resolve(strict=True) != bus
        or (
            runtime_metadata.st_dev,
            runtime_metadata.st_ino,
        )
        != (
            final_runtime_metadata.st_dev,
            final_runtime_metadata.st_ino,
        )
    ):
        raise PermissionError("user manager D-Bus endpoint is not trusted")
    return {
        "uid": selected_uid,
        "runtime_directory": str(runtime),
        "runtime_directory_device": runtime_metadata.st_dev,
        "runtime_directory_inode": runtime_metadata.st_ino,
        "bus_path": str(bus),
        "bus_device": metadata.st_dev,
        "bus_inode": metadata.st_ino,
    }


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def _validate_read_only_argv(argv: Sequence[str]) -> tuple[str, ...]:
    materialized = tuple(argv)
    if (
        not materialized
        or any(
            not isinstance(value, str)
            or not value
            or "\x00" in value
            for value in materialized
        )
        or materialized[0] not in READ_ONLY_EXECUTABLES
    ):
        raise PermissionError("command is outside the read-only allowlist")
    if materialized[0] == SYSTEMCTL_PATH:
        allowed = (
            "is-system-running" in materialized
            or "list-units" in materialized
            or "show" in materialized
            or "--failed" in materialized
        )
        forbidden = {
            "start",
            "stop",
            "restart",
            "try-restart",
            "reload",
            "enable",
            "disable",
            "mask",
            "unmask",
            "reset-failed",
            "daemon-reload",
            "set-property",
            "kill",
        }
        if not allowed or forbidden.intersection(materialized):
            raise PermissionError("systemctl argv is not audit-only")
    elif materialized not in {GPU_QUERY_ARGV, GPU_APPS_QUERY_ARGV}:
        raise PermissionError("nvidia-smi argv is not an exact metadata query")
    return materialized


def run_read_only_command(
    argv: Sequence[str],
    *,
    timeout_seconds: float = 15.0,
    uid: int | None = None,
) -> CommandResult:
    """Run one exact metadata command without a shell or inherited env."""

    materialized = _validate_read_only_argv(argv)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0.0 < float(timeout_seconds) <= 60.0
    ):
        raise ValueError("timeout_seconds is outside the audit envelope")
    completed = subprocess.run(
        list(materialized),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=float(timeout_seconds),
        env=fixed_user_manager_environment(uid),
    )
    return CommandResult(
        argv=materialized,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


@dataclass(frozen=True)
class GPUDevice:
    index: int
    uuid: str
    pci_bus_id: str
    compute_mode: str
    mig_mode: str | None
    driver_version: str
    minor_number: int | None = None
    mps_state: str = "unknown"


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    gpu_uuid: str
    process_name: str
    used_gpu_memory_mib: int | None


def _csv_rows(text: str) -> list[list[str]]:
    try:
        return [
            [cell.strip() for cell in row]
            for row in csv.reader(text.splitlines(), strict=True)
            if row
        ]
    except csv.Error as error:
        raise ValueError("nvidia-smi returned malformed CSV") from error


def parse_gpu_inventory(text: str) -> tuple[GPUDevice, ...]:
    rows = _csv_rows(text)
    devices: list[GPUDevice] = []
    for row in rows:
        if len(row) != 6:
            raise ValueError("GPU inventory row width changed")
        try:
            index = int(row[0])
        except ValueError as error:
            raise ValueError("GPU index is not an integer") from error
        uuid = row[1]
        if (
            index < 0
            or not _is_gpu_uuid(uuid)
            or _PCI_BUS_RE.fullmatch(row[2]) is None
            or not row[3]
            or not row[5]
        ):
            raise ValueError("GPU inventory identity is malformed")
        mig = None if row[4] in {"N/A", "[N/A]", ""} else row[4]
        devices.append(
            GPUDevice(
                index=index,
                uuid=uuid,
                pci_bus_id=row[2],
                compute_mode=row[3],
                mig_mode=mig,
                driver_version=row[5],
            )
        )
    if not devices:
        raise ValueError("GPU inventory is empty")
    if len({row.index for row in devices}) != len(devices) or len(
        {row.uuid for row in devices}
    ) != len(devices):
        raise ValueError("GPU inventory contains duplicate identities")
    return tuple(sorted(devices, key=lambda row: row.index))


def parse_gpu_processes(text: str) -> tuple[GPUProcess, ...]:
    rows = _csv_rows(text)
    processes: list[GPUProcess] = []
    for row in rows:
        if len(row) != 4:
            raise ValueError("GPU process row width changed")
        try:
            pid = int(row[0])
        except ValueError as error:
            raise ValueError("GPU process PID is not an integer") from error
        memory: int | None
        if row[3] in {"N/A", "[N/A]", ""}:
            memory = None
        else:
            try:
                memory = int(row[3])
            except ValueError as error:
                raise ValueError("GPU memory is not an integer") from error
        if (
            pid <= 0
            or not _is_gpu_uuid(row[1])
            or not row[2]
            or memory is not None
            and memory < 0
        ):
            raise ValueError("GPU process identity is malformed")
        processes.append(
            GPUProcess(
                pid=pid,
                gpu_uuid=row[1],
                process_name=row[2],
                used_gpu_memory_mib=memory,
            )
        )
    identities = {(row.pid, row.gpu_uuid) for row in processes}
    if len(identities) != len(processes):
        raise ValueError("GPU process list contains duplicate identities")
    return tuple(sorted(processes, key=lambda row: (row.gpu_uuid, row.pid)))


def bind_gpu_driver_metadata(
    devices: Sequence[GPUDevice],
    *,
    driver_root: Path = Path("/proc/driver/nvidia/gpus"),
) -> tuple[GPUDevice, ...]:
    """Bind UUID/bus inventory to the kernel driver's device minor."""

    bound: list[GPUDevice] = []
    for device in devices:
        pci_fields = device.pci_bus_id.split(":")
        proc_bus_id = device.pci_bus_id
        if len(pci_fields) == 3 and len(pci_fields[0]) > 4:
            proc_bus_id = ":".join(
                (pci_fields[0][-4:], pci_fields[1], pci_fields[2])
            )
        candidates = (
            driver_root / device.pci_bus_id / "information",
            driver_root / device.pci_bus_id.lower() / "information",
            driver_root / proc_bus_id / "information",
            driver_root / proc_bus_id.lower() / "information",
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None or source.is_symlink():
            bound.append(replace(device, minor_number=None))
            continue
        values: dict[str, str] = {}
        for line in source.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
        uuid = values.get("GPU UUID")
        minor_text = values.get("Device Minor")
        try:
            minor = int(minor_text) if minor_text is not None else None
        except ValueError:
            minor = None
        if uuid != device.uuid or minor is None or minor < 0:
            bound.append(replace(device, minor_number=None))
        else:
            bound.append(replace(device, minor_number=minor))
    return tuple(bound)


def detect_mps_state(*, proc_root: Path = Path("/proc")) -> str:
    """Return ``enabled_observed``, ``not_observed``, or fail-safe unknown."""

    unknown = False
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return "unknown"
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().lower()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, PermissionError):
            unknown = True
            continue
        if b"nvidia-cuda-mps-control" in command or b"nvidia-cuda-mps-server" in command:
            return "enabled_observed"
    return "unknown" if unknown else "not_observed"


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    starttime_ticks: int
    uid: int
    cgroup_path: str
    argv: tuple[str, ...]


def read_boot_id(
    *,
    source: Path = Path("/proc/sys/kernel/random/boot_id"),
) -> str:
    """Read and validate the kernel boot identity."""

    if (
        not source.is_absolute()
        or not source.is_file()
        or source.is_symlink()
        or source.resolve(strict=True) != source
    ):
        raise RuntimeError("boot-id source is not a canonical file")
    value = source.read_text(encoding="ascii").strip().lower()
    if _BOOT_ID_RE.fullmatch(value) is None:
        raise RuntimeError("kernel boot ID is malformed")
    return value


def _proc_starttime(raw: str) -> int:
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


def _proc_uid(text: str) -> int:
    for line in text.splitlines():
        if line.startswith("Uid:"):
            values = line.split()[1:]
            if len(values) != 4:
                break
            uid = int(values[0])
            if uid < 0:
                break
            return uid
    raise RuntimeError("proc status has no valid Uid")


def _unified_cgroup(text: str) -> str:
    matches: list[str] = []
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            matches.append(fields[2])
    if len(matches) != 1:
        raise RuntimeError("proc cgroup has no unique unified path")
    path = matches[0]
    if (
        not path.startswith("/")
        or "\x00" in path
        or ".." in PurePosixPath(path).parts
    ):
        raise RuntimeError("proc cgroup path is malformed")
    return path.rstrip("/") or "/"


def read_process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> ProcessIdentity:
    """Read one PID identity and reject reuse during the read."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be positive")
    root = proc_root / str(pid)
    first_stat = (root / "stat").read_text(encoding="ascii")
    first_starttime = _proc_starttime(first_stat)
    uid = _proc_uid((root / "status").read_text(encoding="ascii"))
    cgroup = _unified_cgroup(
        (root / "cgroup").read_text(encoding="utf-8")
    )
    cmdline = (root / "cmdline").read_bytes()
    argv = tuple(
        part.decode("utf-8", errors="backslashreplace")
        for part in cmdline.rstrip(b"\x00").split(b"\x00")
        if part
    )
    second_starttime = _proc_starttime(
        (root / "stat").read_text(encoding="ascii")
    )
    if first_starttime != second_starttime:
        raise RuntimeError("PID identity changed during procfs read")
    return ProcessIdentity(
        pid=pid,
        starttime_ticks=first_starttime,
        uid=uid,
        cgroup_path=cgroup,
        argv=argv,
    )


def read_user_manager_identity(
    uid: int,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    process_reader: Callable[[int], ProcessIdentity] = read_process_identity,
) -> ProcessIdentity:
    """Resolve the user manager from init.scope and verify PID identity."""

    if isinstance(uid, bool) or not isinstance(uid, int) or uid < 0:
        raise ValueError("uid must be a nonnegative integer")
    expected_cgroup = (
        f"/user.slice/user-{uid}.slice/user@{uid}.service/init.scope"
    )
    members = cgroup_root / expected_cgroup.lstrip("/") / "cgroup.procs"
    if (
        not members.is_absolute()
        or not members.is_file()
        or members.is_symlink()
        or members.resolve(strict=True) != members
    ):
        raise RuntimeError("user manager cgroup.procs is not canonical")

    def member_pids() -> tuple[int, ...]:
        try:
            values = tuple(
                int(line)
                for line in members.read_text(encoding="ascii").splitlines()
                if line
            )
        except ValueError as error:
            raise RuntimeError(
                "user manager cgroup.procs is malformed"
            ) from error
        if (
            not values
            or len(set(values)) != len(values)
            or any(pid <= 0 for pid in values)
        ):
            raise RuntimeError("user manager cgroup membership is invalid")
        return values

    def is_manager(identity: ProcessIdentity) -> bool:
        return (
            identity.uid == uid
            and identity.cgroup_path == expected_cgroup
            and bool(identity.argv)
            and Path(identity.argv[0]).name == "systemd"
            and "--user" in identity.argv[1:]
        )

    first_members = member_pids()
    candidates: list[ProcessIdentity] = []
    for pid in first_members:
        try:
            identity = process_reader(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if is_manager(identity):
            candidates.append(identity)
    if len(candidates) != 1:
        raise RuntimeError("user manager process identity is ambiguous")
    first = candidates[0]

    second_members = member_pids()
    if first.pid not in second_members:
        raise RuntimeError("user manager left init.scope during verification")
    second = process_reader(first.pid)
    if (
        not is_manager(second)
        or second.starttime_ticks != first.starttime_ticks
    ):
        raise RuntimeError("user manager process identity changed")
    return second


def parse_systemctl_show(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        if "=" not in line:
            raise ValueError("systemctl show output is malformed")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError("systemctl show output has duplicate keys")
        values[key] = value
    return values


def map_process_to_user_unit(
    process: ProcessIdentity,
    unit_control_groups: Mapping[str, str],
    *,
    expected_uid: int,
) -> str | None:
    """Map a process by cgroup ancestry, never by process name/cmdline."""

    if process.uid != expected_uid:
        return None
    matches: list[tuple[int, str]] = []
    for unit, control_group in unit_control_groups.items():
        if (
            not isinstance(unit, str)
            or _UNIT_RE.fullmatch(unit) is None
            or not isinstance(control_group, str)
            or not control_group.startswith("/")
            or ".." in PurePosixPath(control_group).parts
        ):
            raise ValueError("unit ControlGroup inventory is malformed")
        normalized = control_group.rstrip("/") or "/"
        if normalized == "/":
            raise ValueError("unit ControlGroup cannot be the cgroup root")
        if process.cgroup_path == normalized or process.cgroup_path.startswith(
            normalized + "/"
        ):
            matches.append((len(normalized), unit))
    if not matches:
        return None
    matches.sort(reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise RuntimeError("process cgroup maps ambiguously")
    return matches[0][1]


def _process_map(
    processes: Sequence[ProcessIdentity],
) -> dict[int, ProcessIdentity]:
    materialized = {row.pid: row for row in processes}
    if len(materialized) != len(processes):
        raise ValueError("process identity snapshot contains duplicate PIDs")
    return materialized


def verify_gpu_double_snapshot(
    *,
    devices: Sequence[GPUDevice],
    first_apps: Sequence[GPUProcess],
    first_processes: Sequence[ProcessIdentity],
    second_processes: Sequence[ProcessIdentity],
    second_apps: Sequence[GPUProcess],
    selected_gpu_uuid: str,
    expected_uid: int,
    allowed_unit_ids: Sequence[str],
    unit_control_groups: Mapping[str, str],
    strict_all_gpu_consumers: bool = False,
) -> dict[str, object]:
    """Verify a race-detecting GPU/process/cgroup double snapshot."""

    device_rows = tuple(devices)
    known_uuids = {row.uuid for row in device_rows}
    if (
        isinstance(expected_uid, bool)
        or not isinstance(expected_uid, int)
        or expected_uid < 0
        or not _is_gpu_uuid(selected_gpu_uuid)
        or selected_gpu_uuid not in known_uuids
        or len(known_uuids) != len(device_rows)
    ):
        raise ValueError("selected GPU UUID is absent or ambiguous")
    if not isinstance(strict_all_gpu_consumers, bool):
        raise ValueError("strict all-GPU policy must be boolean")
    allowed = tuple(allowed_unit_ids)
    if len(set(allowed)) != len(allowed) or any(
        _UNIT_RE.fullmatch(value) is None for value in allowed
    ):
        raise ValueError("allowed unit IDs are malformed")

    first_keys = {(row.pid, row.gpu_uuid) for row in first_apps}
    second_keys = {(row.pid, row.gpu_uuid) for row in second_apps}
    if (
        len(first_keys) != len(first_apps)
        or len(second_keys) != len(second_apps)
    ):
        raise ValueError("GPU application snapshots contain duplicates")
    blockers: list[str] = []
    observations: list[str] = []
    selected_device = next(
        row for row in device_rows if row.uuid == selected_gpu_uuid
    )
    if selected_device.minor_number is None:
        blockers.append("selected_gpu_minor_number_unknown")
    if selected_device.mps_state == "unknown":
        blockers.append("selected_gpu_mps_state_unknown")
    elif selected_device.mps_state != "not_observed":
        blockers.append(
            f"selected_gpu_mps_state_blocked:{selected_device.mps_state}"
        )
    selected_first_keys = {
        key for key in first_keys if key[1] == selected_gpu_uuid
    }
    selected_second_keys = {
        key for key in second_keys if key[1] == selected_gpu_uuid
    }
    if selected_first_keys != selected_second_keys:
        blockers.append("selected_gpu_process_set_changed_between_snapshots")
    if first_keys != second_keys:
        observations.append("gpu_process_set_changed_between_snapshots")
        if strict_all_gpu_consumers:
            blockers.append("gpu_process_set_changed_between_snapshots")
    for _, uuid in first_keys | second_keys:
        if uuid not in known_uuids:
            finding = f"unknown_gpu_uuid:{uuid}"
            observations.append(finding)
            if strict_all_gpu_consumers:
                blockers.append(finding)

    before = _process_map(first_processes)
    after = _process_map(second_processes)
    unit_rows: list[dict[str, object]] = []
    for pid, uuid in sorted(first_keys | second_keys):
        first = before.get(pid)
        second = after.get(pid)
        hard_gate = uuid == selected_gpu_uuid or strict_all_gpu_consumers
        if first is None or second is None:
            finding = f"missing_proc_identity:{pid}"
            observations.append(finding)
            if hard_gate:
                blockers.append(finding)
            continue
        if (
            first.starttime_ticks != second.starttime_ticks
            or first.uid != second.uid
            or first.cgroup_path != second.cgroup_path
        ):
            finding = f"pid_identity_changed:{pid}"
            observations.append(finding)
            if hard_gate:
                blockers.append(finding)
            continue
        unit = map_process_to_user_unit(
            second,
            unit_control_groups,
            expected_uid=expected_uid,
        )
        unit_rows.append(
            {
                "pid": pid,
                "starttime_ticks": second.starttime_ticks,
                "uid": second.uid,
                "gpu_uuid": uuid,
                "cgroup_path": second.cgroup_path,
                "unit_id": unit,
            }
        )
        if unit is None:
            finding = f"unmapped_gpu_process:{pid}"
            observations.append(finding)
            if hard_gate:
                blockers.append(finding)
        if uuid == selected_gpu_uuid and unit not in allowed:
            blockers.append(f"selected_gpu_consumer_not_allowed:{pid}")

    unique_blockers = sorted(set(blockers))
    body: dict[str, object] = {
        "schema_version": GPU_DOUBLE_SNAPSHOT_SCHEMA,
        "selected_gpu_uuid": selected_gpu_uuid,
        "expected_uid": expected_uid,
        "allowed_unit_ids": list(allowed),
        "strict_all_gpu_consumers": strict_all_gpu_consumers,
        "devices": [asdict(row) for row in device_rows],
        "first_apps": [asdict(row) for row in first_apps],
        "second_apps": [asdict(row) for row in second_apps],
        "process_unit_mapping": unit_rows,
        "observations": sorted(set(observations)),
        "blockers": unique_blockers,
        "passed": not unique_blockers,
    }
    return {
        **body,
        "snapshot_fingerprint": stable_fingerprint(body),
    }


def _list_user_unit_control_groups(
    runner: Callable[[Sequence[str]], CommandResult],
) -> dict[str, str]:
    listed = runner(
        (
            SYSTEMCTL_PATH,
            "--user",
            "list-units",
            "--type=service",
            "--all",
            "--no-legend",
            "--plain",
            "--no-pager",
        )
    )
    if listed.returncode != 0:
        raise RuntimeError("systemctl list-units failed")
    units: list[str] = []
    for line in listed.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if _UNIT_RE.fullmatch(fields[0]) is None:
            raise ValueError("systemctl list-units returned an invalid unit")
        units.append(fields[0])
    if len(set(units)) != len(units):
        raise ValueError("systemctl list-units returned duplicate units")
    control_groups: dict[str, str] = {}
    for unit in units:
        shown = runner(
            (
                SYSTEMCTL_PATH,
                "--user",
                "show",
                unit,
                "-p",
                "Id",
                "-p",
                "ControlGroup",
                "--no-pager",
            )
        )
        if shown.returncode != 0:
            raise RuntimeError(f"systemctl show failed for {unit}")
        values = parse_systemctl_show(shown.stdout)
        if set(values) != {"Id", "ControlGroup"} or values["Id"] != unit:
            raise ValueError("unit identity/control-group output changed")
        if values["ControlGroup"]:
            control_groups[unit] = values["ControlGroup"]
    return control_groups


def _validated_unit_tuple(
    values: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized) or any(
        not isinstance(value, str) or _UNIT_RE.fullmatch(value) is None
        for value in materialized
    ):
        raise ValueError(f"{name} unit IDs are malformed")
    return materialized


def _query_scoped_unit_shadows(
    runner: Callable[[Sequence[str]], CommandResult],
    unit_ids: Sequence[str],
) -> dict[str, dict[str, str]]:
    properties = (
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
        *ACTIVATION_CLOSURE_FIELDS,
    )
    shadows: dict[str, dict[str, str]] = {}
    for unit in unit_ids:
        argv: list[str] = [SYSTEMCTL_PATH, "--user", "show", unit]
        for property_name in properties:
            argv.extend(("-p", property_name))
        argv.append("--no-pager")
        shown = runner(tuple(argv))
        if shown.returncode != 0:
            raise RuntimeError(f"systemctl show failed for scoped unit {unit}")
        values = parse_systemctl_show(shown.stdout)
        if set(values) != set(properties) or values["Id"] != unit:
            raise ValueError("scoped unit shadow changed schema or identity")
        shadows[unit] = values
    return shadows


def collect_environment_inventory(
    *,
    selected_gpu_index: int,
    allowed_unit_ids: Sequence[str] = (),
    target_unit_id: str | None = None,
    conflict_unit_ids: Sequence[str] = (),
    dependency_unit_ids: Sequence[str] = (),
    allowed_failed_unit_ids: Sequence[str] = (),
    allowed_manager_states: Sequence[str] = ("running", "degraded"),
    require_target_ready: bool = False,
    strict_all_gpu_consumers: bool = False,
    conflict_quiescence_mode: str = NORMAL_QUIESCENCE_MODE,
    command_runner: Callable[[Sequence[str]], CommandResult] = (
        run_read_only_command
    ),
    process_reader: Callable[[int], ProcessIdentity] = read_process_identity,
    endpoint_validator: Callable[[int], Mapping[str, object]] = (
        validate_user_manager_endpoint
    ),
    boot_id_reader: Callable[[], str] = read_boot_id,
    manager_identity_reader: Callable[[int], ProcessIdentity] = (
        read_user_manager_identity
    ),
    driver_metadata_binder: Callable[
        [Sequence[GPUDevice]], Sequence[GPUDevice]
    ] = bind_gpu_driver_metadata,
    mps_detector: Callable[[], str] = detect_mps_state,
    uid: int | None = None,
) -> dict[str, object]:
    """Collect one strict read-only manager/GPU double-snapshot inventory."""

    selected_uid = os.getuid() if uid is None else uid
    if (
        isinstance(selected_gpu_index, bool)
        or not isinstance(selected_gpu_index, int)
        or selected_gpu_index < 0
        or isinstance(selected_uid, bool)
        or not isinstance(selected_uid, int)
        or selected_uid < 0
        or not isinstance(strict_all_gpu_consumers, bool)
        or conflict_quiescence_mode
        not in {NORMAL_QUIESCENCE_MODE, RECOVERY_QUIESCENCE_MODE}
    ):
        raise ValueError("selected GPU index, uid, or strict policy is invalid")
    target_units = () if target_unit_id is None else (target_unit_id,)
    target_units = _validated_unit_tuple(target_units, name="target")
    conflict_units = _validated_unit_tuple(
        conflict_unit_ids,
        name="conflict",
    )
    if (
        conflict_quiescence_mode == RECOVERY_QUIESCENCE_MODE
        and conflict_units != (GPU0_CONFLICT_UNIT,)
    ):
        raise PermissionError(
            "recovery quiescence is restricted to the exact GPU0 conflict"
        )
    dependency_units = _validated_unit_tuple(
        dependency_unit_ids,
        name="dependency",
    )
    allowed_failed = _validated_unit_tuple(
        allowed_failed_unit_ids,
        name="allowed failed",
    )
    scoped_units = target_units + conflict_units + dependency_units
    if len(set(scoped_units)) != len(scoped_units):
        raise ValueError("target/conflict/dependency unit scopes overlap")
    if set(scoped_units).intersection(allowed_failed):
        raise ValueError("scoped units cannot be unrelated failed allowlist")
    manager_states = tuple(allowed_manager_states)
    known_manager_states = {
        "running",
        "degraded",
        "starting",
        "stopping",
        "maintenance",
        "offline",
        "unknown",
    }
    if (
        not manager_states
        or len(set(manager_states)) != len(manager_states)
        or not set(manager_states).issubset(known_manager_states)
    ):
        raise ValueError("allowed manager states are malformed")
    endpoint = dict(endpoint_validator(selected_uid))
    expected_runtime = f"/run/user/{selected_uid}"
    required_endpoint_keys = {
        "uid",
        "runtime_directory",
        "runtime_directory_device",
        "runtime_directory_inode",
        "bus_path",
        "bus_device",
        "bus_inode",
    }
    if set(endpoint) != required_endpoint_keys:
        raise RuntimeError("user manager endpoint receipt schema changed")
    integer_fields = (
        endpoint["runtime_directory_device"],
        endpoint["runtime_directory_inode"],
        endpoint["bus_device"],
        endpoint["bus_inode"],
    )
    if (
        endpoint["uid"] != selected_uid
        or endpoint["runtime_directory"] != expected_runtime
        or endpoint["bus_path"] != f"{expected_runtime}/bus"
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in integer_fields
        )
    ):
        raise RuntimeError("user manager endpoint receipt identity changed")
    boot_id = boot_id_reader().lower()
    if _BOOT_ID_RE.fullmatch(boot_id) is None:
        raise RuntimeError("kernel boot ID is malformed")
    manager_identity = manager_identity_reader(selected_uid)
    expected_manager_cgroup = (
        f"/user.slice/user-{selected_uid}.slice/"
        f"user@{selected_uid}.service/init.scope"
    )
    if (
        not isinstance(manager_identity, ProcessIdentity)
        or manager_identity.pid <= 0
        or manager_identity.starttime_ticks <= 0
        or manager_identity.uid != selected_uid
        or manager_identity.cgroup_path != expected_manager_cgroup
    ):
        raise RuntimeError("user manager process identity is invalid")
    manager = command_runner(
        (SYSTEMCTL_PATH, "--user", "is-system-running", "--no-pager")
    )
    manager_state = manager.stdout.strip()
    if manager_state not in known_manager_states:
        raise ValueError("user manager state is malformed")
    failed = command_runner(
        (
            SYSTEMCTL_PATH,
            "--user",
            "--failed",
            "--all",
            "--no-legend",
            "--plain",
            "--no-pager",
        )
    )
    if failed.returncode not in {0, 1}:
        raise RuntimeError("failed-unit inventory command failed")
    failed_units = [
        line.split()[0]
        for line in failed.stdout.splitlines()
        if line.split()
    ]
    if (
        len(set(failed_units)) != len(failed_units)
        or any(_UNIT_RE.fullmatch(unit) is None for unit in failed_units)
    ):
        raise ValueError("failed-unit inventory contains an invalid unit")

    scoped_shadows = _query_scoped_unit_shadows(
        command_runner,
        scoped_units,
    )

    gpu_result = command_runner(GPU_QUERY_ARGV)
    if gpu_result.returncode != 0:
        raise RuntimeError("GPU inventory command failed")
    devices = tuple(
        driver_metadata_binder(parse_gpu_inventory(gpu_result.stdout))
    )
    first_mps_state = mps_detector()
    selected = [row for row in devices if row.index == selected_gpu_index]
    if len(selected) != 1:
        raise ValueError("selected GPU index is absent or ambiguous")

    first_result = command_runner(GPU_APPS_QUERY_ARGV)
    if first_result.returncode != 0:
        raise RuntimeError("first GPU process query failed")
    first_apps = parse_gpu_processes(first_result.stdout)
    first_processes = tuple(
        process_reader(pid) for pid in sorted({row.pid for row in first_apps})
    )
    control_groups = _list_user_unit_control_groups(command_runner)
    second_result = command_runner(GPU_APPS_QUERY_ARGV)
    if second_result.returncode != 0:
        raise RuntimeError("second GPU process query failed")
    second_apps = parse_gpu_processes(second_result.stdout)
    second_mps_state = mps_detector()
    if first_mps_state == second_mps_state == "not_observed":
        mps_state = "not_observed"
    elif "enabled_observed" in {first_mps_state, second_mps_state}:
        mps_state = "enabled_observed"
    else:
        mps_state = "unknown"
    devices = tuple(replace(row, mps_state=mps_state) for row in devices)
    second_processes = tuple(
        process_reader(pid)
        for pid in sorted(
            {row.pid for row in first_apps}
            | {row.pid for row in second_apps}
        )
    )
    gpu_snapshot = verify_gpu_double_snapshot(
        devices=devices,
        first_apps=first_apps,
        first_processes=first_processes,
        second_processes=second_processes,
        second_apps=second_apps,
        selected_gpu_uuid=selected[0].uuid,
        expected_uid=selected_uid,
        allowed_unit_ids=allowed_unit_ids,
        unit_control_groups=control_groups,
        strict_all_gpu_consumers=strict_all_gpu_consumers,
    )
    blockers = list(gpu_snapshot["blockers"])
    if manager_state not in manager_states:
        blockers.append(f"user_manager_state_not_allowed:{manager_state}")
    failed_set = set(failed_units)
    unexpected_failed = sorted(failed_set - set(allowed_failed))
    scoped_failed = sorted(failed_set.intersection(scoped_units))
    if unexpected_failed:
        blockers.append("unexpected_failed_user_units")
    if scoped_failed:
        blockers.append("scoped_user_unit_failed")
    for unit in conflict_units + dependency_units:
        shadow = scoped_shadows[unit]
        expected_unit_file_state = (
            "enabled"
            if (
                unit in conflict_units
                and conflict_quiescence_mode == RECOVERY_QUIESCENCE_MODE
            )
            else "masked-runtime"
        )
        expected_activation_closure = (
            _expected_conflict_activation_closure(
                RECOVERY_CLEANUP_MODE
                if (
                    unit in conflict_units
                    and conflict_quiescence_mode
                    == RECOVERY_QUIESCENCE_MODE
                )
                else NORMAL_CLEANUP_MODE
            )
        )
        if (
            shadow["LoadState"] != "loaded"
            or shadow["ActiveState"] != "inactive"
            or shadow["SubState"] != "dead"
            or shadow["UnitFileState"] != expected_unit_file_state
            or {
                field: shadow[field]
                for field in ACTIVATION_CLOSURE_FIELDS
            }
            != expected_activation_closure
        ):
            blockers.append(f"scoped_blocker_unit_not_quiescent:{unit}")
    if require_target_ready:
        if len(target_units) != 1:
            raise ValueError("require_target_ready needs one target unit")
        target_shadow = scoped_shadows[target_units[0]]
        expected_target = {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "static",
            "Restart": "no",
            "NRestarts": "0",
        }
        if any(
            target_shadow[field] != expected
            for field, expected in expected_target.items()
        ):
            blockers.append("target_unit_not_ready")
    body: dict[str, object] = {
        "schema_version": ENVIRONMENT_INVENTORY_SCHEMA,
        "created_at_utc": utc_now(),
        "uid": selected_uid,
        "boot_id": boot_id,
        "manager": {
            "state": manager_state,
            "allowed_states": list(manager_states),
            "returncode": manager.returncode,
            "failed_units": failed_units,
            "allowed_failed_unit_ids": list(allowed_failed),
            "unexpected_failed_unit_ids": unexpected_failed,
            "scoped_failed_unit_ids": scoped_failed,
            "identity": {
                "pid": manager_identity.pid,
                "starttime_ticks": manager_identity.starttime_ticks,
                "uid": manager_identity.uid,
                "control_group": manager_identity.cgroup_path,
            },
            "endpoint": endpoint,
        },
        "unit_scope": {
            "target_unit_id": target_unit_id,
            "conflict_unit_ids": list(conflict_units),
            "dependency_unit_ids": list(dependency_units),
            "require_target_ready": require_target_ready,
            "shadows": scoped_shadows,
        },
        "gpu_snapshot": gpu_snapshot,
        "blockers": sorted(set(blockers)),
        "passed": not blockers,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {
        **body,
        "inventory_fingerprint": stable_fingerprint(body),
    }


@dataclass(frozen=True)
class EnvironmentAuditContract:
    """Closed runtime identity and scope contract for one live audit."""

    uid: int
    boot_id: str
    runtime_directory: str
    runtime_directory_device: int
    runtime_directory_inode: int
    manager_pid: int
    bus_path: str
    bus_device: int
    bus_inode: int
    manager_starttime_ticks: int
    manager_control_group: str
    selected_gpu_index: int
    selected_gpu_uuid: str
    selected_gpu_pci_bus_id: str
    selected_gpu_minor_number: int
    target_unit_id: str | None
    conflict_unit_ids: tuple[str, ...]
    dependency_unit_ids: tuple[str, ...]
    allowed_failed_unit_ids: tuple[str, ...]
    expected_failed_unit_ids: tuple[str, ...]
    maximum_restart_usec: int
    maximum_trigger_usec: int
    required_stability_window_usec: int
    cleanup_mode: str
    quiescence_mode: str
    cleanup_nrestarts_baseline: tuple[tuple[str, str], ...]
    activation_guard: dict[str, object]
    allowed_unit_ids: tuple[str, ...]
    allowed_manager_states: tuple[str, ...]
    require_target_ready: bool = False
    strict_all_gpu_consumers: bool = False

def validate_environment_audit_contract(
    contract: EnvironmentAuditContract,
) -> EnvironmentAuditContract:
    if not isinstance(contract, EnvironmentAuditContract):
        raise TypeError("environment audit contract type changed")
    target_values = () if contract.target_unit_id is None else (
        contract.target_unit_id,
    )
    targets = _validated_unit_tuple(target_values, name="target")
    conflicts = _validated_unit_tuple(
        contract.conflict_unit_ids,
        name="conflict",
    )
    dependencies = _validated_unit_tuple(
        contract.dependency_unit_ids,
        name="dependency",
    )
    allowed_failed = _validated_unit_tuple(
        contract.allowed_failed_unit_ids,
        name="allowed failed",
    )
    expected_failed = _validated_unit_tuple(
        contract.expected_failed_unit_ids,
        name="expected failed",
    )
    allowed_units = _validated_unit_tuple(
        contract.allowed_unit_ids,
        name="allowed",
    )
    try:
        baseline = tuple(
            (str(unit), _validated_nrestarts(value, name=f"contract {unit}"))
            for unit, value in contract.cleanup_nrestarts_baseline
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "environment cleanup NRestarts baseline is malformed"
        ) from error
    if not isinstance(contract.activation_guard, Mapping):
        raise ValueError("environment activation guard is malformed")
    guard = dict(contract.activation_guard)
    states = tuple(contract.allowed_manager_states)
    known_states = {
        "running", "degraded", "starting", "stopping", "maintenance",
        "offline", "unknown",
    }
    expected_runtime = f"/run/user/{contract.uid}"
    expected_bus = f"{expected_runtime}/bus"
    expected_control_group = (
        f"/user.slice/user-{contract.uid}.slice/"
        f"user@{contract.uid}.service/init.scope"
    )
    integers = (
        contract.uid,
        contract.runtime_directory_device,
        contract.runtime_directory_inode,
        contract.bus_device,
        contract.bus_inode,
        contract.manager_pid,
        contract.manager_starttime_ticks,
        contract.selected_gpu_index,
        contract.selected_gpu_minor_number,
        contract.maximum_restart_usec,
        contract.maximum_trigger_usec,
        contract.required_stability_window_usec,
    )
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integers)
        or contract.manager_pid <= 0
        or contract.manager_starttime_ticks <= 0
        or _BOOT_ID_RE.fullmatch(contract.boot_id) is None
        or contract.runtime_directory != expected_runtime
        or contract.manager_control_group != expected_control_group
        or contract.bus_path != expected_bus
        or not _is_gpu_uuid(contract.selected_gpu_uuid)
        or _PCI_BUS_RE.fullmatch(contract.selected_gpu_pci_bus_id) is None
        or targets != target_values
        or conflicts != contract.conflict_unit_ids
        or dependencies != contract.dependency_unit_ids
        or allowed_failed != contract.allowed_failed_unit_ids
        or expected_failed != contract.expected_failed_unit_ids
        or allowed_units != contract.allowed_unit_ids
        or not states or len(set(states)) != len(states)
        or not set(states).issubset(known_states)
        or not isinstance(contract.require_target_ready, bool)
        or not isinstance(contract.strict_all_gpu_consumers, bool)
        or not set(expected_failed).issubset(allowed_failed)
        or not _deep_exact_equal(
            baseline,
            contract.cleanup_nrestarts_baseline,
        )
        or not _deep_exact_equal(
            tuple(unit for unit, _ in baseline),
            conflicts,
        )
        or contract.required_stability_window_usec
        != max(contract.maximum_restart_usec, contract.maximum_trigger_usec)
    ):
        raise ValueError("environment audit contract is malformed")
    if contract.cleanup_mode == NORMAL_CLEANUP_MODE:
        if (
            contract.quiescence_mode != NORMAL_QUIESCENCE_MODE
            or set(guard) != _NORMAL_GUARD_KEYS
            or len(conflicts) != 1
            or not _deep_exact_equal(
                guard,
                {
                    "mode": NORMAL_GUARD_MODE,
                    "unit_name": conflicts[0],
                    "observed_unit_file_state": "masked-runtime",
                },
            )
        ):
            raise ValueError("normal environment quiescence contract changed")
    elif contract.cleanup_mode == RECOVERY_CLEANUP_MODE:
        if (
            contract.quiescence_mode != RECOVERY_QUIESCENCE_MODE
            or conflicts != (GPU0_CONFLICT_UNIT,)
            or set(guard) != _RECOVERY_GUARD_KEYS
            or guard.get("mode") != RECOVERY_GUARD_MODE
            or guard.get("unit_name") != GPU0_CONFLICT_UNIT
            or guard.get("path")
            != f"/run/user/{contract.uid}/systemd/user/{GPU0_CONFLICT_UNIT}"
            or guard.get("target") != "/dev/null"
            or not _deep_exact_equal(
                guard.get("owner_uid"),
                contract.uid,
            )
            or guard.get("observed_unit_file_state") != "enabled"
            or any(
                isinstance(guard.get(field), bool)
                or not isinstance(guard.get(field), int)
                or guard[field] < 0
                for field in ("device", "inode")
            )
            or guard.get("inode") == 0
        ):
            raise ValueError("recovery environment quiescence contract changed")
    else:
        raise ValueError("environment cleanup mode is malformed")
    scoped = targets + conflicts + dependencies
    if len(set(scoped)) != len(scoped) or set(scoped) & set(allowed_failed):
        raise ValueError("environment audit contract scopes overlap")
    return contract



def environment_audit_contract_from_inventory(
    inventory: Mapping[str, object],
    *,
    selected_gpu_index: int,
    target_unit_id: str | None,
    conflict_unit_ids: Sequence[str],
    dependency_unit_ids: Sequence[str],
    allowed_failed_unit_ids: Sequence[str],
    cleanup_mode: str,
    quiescence_mode: str,
    cleanup_nrestarts_baseline: Sequence[Sequence[str]],
    activation_guard: Mapping[str, object],
    allowed_unit_ids: Sequence[str] = (),
    allowed_manager_states: Sequence[str] = ("running", "degraded"),
    require_target_ready: bool = False,
    strict_all_gpu_consumers: bool = False,
) -> EnvironmentAuditContract:
    body = dict(inventory)
    fingerprint = body.pop("inventory_fingerprint", None)
    if (
        body.get("schema_version") != ENVIRONMENT_INVENTORY_SCHEMA
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
    ):
        raise PermissionError("precleanup inventory fingerprint is invalid")
    _validate_environment_inventory_identity_types(inventory)
    try:
        unit_scope = dict(inventory["unit_scope"])
        manager = dict(inventory["manager"])
        endpoint = dict(manager["endpoint"])
        identity = dict(manager["identity"])
        gpu = dict(inventory["gpu_snapshot"])
        devices = [dict(item) for item in gpu["devices"]]
        shadows = {key: dict(value) for key, value in dict(unit_scope["shadows"]).items()}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("precleanup inventory structure is malformed") from error
    expected_scope = {
        "target_unit_id": target_unit_id,
        "conflict_unit_ids": list(conflict_unit_ids),
        "dependency_unit_ids": list(dependency_unit_ids),
        "require_target_ready": require_target_ready,
    }
    observed_scope = {
        "target_unit_id": unit_scope.get("target_unit_id"),
        "conflict_unit_ids": unit_scope.get("conflict_unit_ids"),
        "dependency_unit_ids": unit_scope.get("dependency_unit_ids"),
        "require_target_ready": unit_scope.get("require_target_ready"),
    }
    if not _deep_exact_equal(observed_scope, expected_scope):
        raise PermissionError("precleanup inventory unit scope changed")
    if (
        not _deep_exact_equal(
            manager.get("allowed_failed_unit_ids"),
            list(allowed_failed_unit_ids),
        )
        or not _deep_exact_equal(
            manager.get("allowed_states"),
            list(allowed_manager_states),
        )
        or not _deep_exact_equal(
            gpu.get("allowed_unit_ids"),
            list(allowed_unit_ids),
        )
        or gpu.get("strict_all_gpu_consumers") is not strict_all_gpu_consumers
    ):
        raise PermissionError("precleanup inventory policy scope changed")
    scoped_units = (
        (() if target_unit_id is None else (target_unit_id,))
        + tuple(conflict_unit_ids)
        + tuple(dependency_unit_ids)
    )
    if set(shadows) != set(scoped_units):
        raise PermissionError("precleanup scoped unit shadow set changed")
    restart_usec_by_unit: dict[str, int] = {}
    for unit in scoped_units:
        shadow = shadows[unit]
        restart_value = shadow.get("RestartUSec")
        if not isinstance(restart_value, str):
            raise ValueError("precleanup RestartUSec evidence is malformed")
        restart_usec_by_unit[unit] = parse_systemd_duration_usec(restart_value)
        expected_precleanup_closure = (
            _expected_conflict_activation_closure(
                RECOVERY_CLEANUP_MODE
            )
            if (
                unit in tuple(conflict_unit_ids)
                and cleanup_mode == RECOVERY_CLEANUP_MODE
            )
            else _expected_conflict_activation_closure(
                NORMAL_CLEANUP_MODE
            )
        )
        if not _deep_exact_equal(
            {
                field: shadow.get(field)
                for field in ACTIVATION_CLOSURE_FIELDS
            },
            expected_precleanup_closure,
        ):
            raise PermissionError(
                "precleanup trigger period is not closed; policy cannot be derived"
            )
    failed_value = manager.get("failed_units")
    if not isinstance(failed_value, list):
        raise ValueError("precleanup failed-unit evidence is malformed")
    expected_failed_unit_ids = _validated_unit_tuple(
        tuple(sorted(failed_value)),
        name="expected failed",
    )
    maximum_restart_usec = max(restart_usec_by_unit.values(), default=0)
    maximum_trigger_usec = 0
    required_stability_window_usec = max(
        maximum_restart_usec,
        maximum_trigger_usec,
    )
    selected = [
        device for device in devices
        if _deep_exact_equal(device.get("index"), selected_gpu_index)
    ]
    if len(selected) != 1:
        raise ValueError("precleanup selected GPU is absent or ambiguous")
    device = selected[0]
    if not _deep_exact_equal(
        device.get("uuid"),
        gpu.get("selected_gpu_uuid"),
    ):
        raise PermissionError("precleanup selected GPU identity changed")
    contract = EnvironmentAuditContract(
        uid=inventory["uid"],
        bus_path=endpoint["bus_path"],
        bus_device=endpoint["bus_device"],
        bus_inode=endpoint["bus_inode"],
        boot_id=inventory["boot_id"],
        runtime_directory=endpoint["runtime_directory"],
        runtime_directory_device=endpoint["runtime_directory_device"],
        runtime_directory_inode=endpoint["runtime_directory_inode"],
        manager_pid=identity["pid"],
        manager_starttime_ticks=identity["starttime_ticks"],
        manager_control_group=identity["control_group"],
        selected_gpu_index=selected_gpu_index,
        selected_gpu_uuid=device["uuid"],
        selected_gpu_pci_bus_id=device["pci_bus_id"],
        selected_gpu_minor_number=device["minor_number"],
        target_unit_id=target_unit_id,
        conflict_unit_ids=tuple(conflict_unit_ids),
        dependency_unit_ids=tuple(dependency_unit_ids),
        allowed_failed_unit_ids=tuple(allowed_failed_unit_ids),
        expected_failed_unit_ids=expected_failed_unit_ids,
        maximum_restart_usec=maximum_restart_usec,
        maximum_trigger_usec=maximum_trigger_usec,
        required_stability_window_usec=required_stability_window_usec,
        cleanup_mode=cleanup_mode,
        quiescence_mode=quiescence_mode,
        cleanup_nrestarts_baseline=tuple(
            (str(item[0]), str(item[1]))
            for item in cleanup_nrestarts_baseline
        ),
        activation_guard=dict(activation_guard),
        allowed_unit_ids=tuple(allowed_unit_ids),
        allowed_manager_states=tuple(allowed_manager_states),
        require_target_ready=require_target_ready,
        strict_all_gpu_consumers=strict_all_gpu_consumers,
    )
    return validate_environment_audit_contract(contract)


def _validate_environment_inventory_identity_types(
    inventory: Mapping[str, object],
) -> None:
    """Reject numeric/type coercions in contract-bearing inventory fields."""

    inventory_keys = {
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
    manager_keys = {
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
    identity_keys = {"pid", "starttime_ticks", "uid", "control_group"}
    endpoint_keys = {
        "uid",
        "runtime_directory",
        "runtime_directory_device",
        "runtime_directory_inode",
        "bus_path",
        "bus_device",
        "bus_inode",
    }
    scope_keys = {
        "target_unit_id",
        "conflict_unit_ids",
        "dependency_unit_ids",
        "require_target_ready",
        "shadows",
    }
    shadow_keys = {
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
        *ACTIVATION_CLOSURE_FIELDS,
    }
    gpu_keys = {
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
    mapping_keys = {
        "pid",
        "starttime_ticks",
        "uid",
        "gpu_uuid",
        "cgroup_path",
        "unit_id",
    }
    try:
        manager = dict(inventory["manager"])
        identity = dict(manager["identity"])
        endpoint = dict(manager["endpoint"])
        scope = dict(inventory["unit_scope"])
        shadows = {
            unit: dict(value)
            for unit, value in dict(scope["shadows"]).items()
        }
        gpu = dict(inventory["gpu_snapshot"])
        devices = [dict(item) for item in gpu["devices"]]
        first_apps = [dict(item) for item in gpu["first_apps"]]
        second_apps = [dict(item) for item in gpu["second_apps"]]
        mappings = [dict(item) for item in gpu["process_unit_mapping"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "environment inventory identity structure is malformed"
        ) from error
    target_unit = scope.get("target_unit_id")
    try:
        target_units = (
            ()
            if target_unit is None
            else _validated_unit_tuple((target_unit,), name="inventory target")
        )
        conflict_units = _validated_unit_tuple(
            scope.get("conflict_unit_ids"),
            name="inventory conflict",
        )
        dependency_units = _validated_unit_tuple(
            scope.get("dependency_unit_ids"),
            name="inventory dependency",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "environment inventory unit scope is malformed"
        ) from error
    scoped_units = target_units + conflict_units + dependency_units
    if (
        len(set(scoped_units)) != len(scoped_units)
        or set(shadows) != set(scoped_units)
    ):
        raise ValueError(
            "environment inventory shadow scope changed"
        )
    known_manager_states = {
        "running",
        "degraded",
        "starting",
        "stopping",
        "maintenance",
        "offline",
        "unknown",
    }
    try:
        allowed_states = tuple(manager["allowed_states"])
        failed_units = _validated_unit_tuple(
            manager["failed_units"],
            name="inventory failed",
        )
        allowed_failed_units = _validated_unit_tuple(
            manager["allowed_failed_unit_ids"],
            name="inventory allowed failed",
        )
        unexpected_failed_units = _validated_unit_tuple(
            manager["unexpected_failed_unit_ids"],
            name="inventory unexpected failed",
        )
        scoped_failed_units = _validated_unit_tuple(
            manager["scoped_failed_unit_ids"],
            name="inventory scoped failed",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "environment inventory manager scope is malformed"
        ) from error
    manager_state = manager.get("state")
    expected_manager_returncode = (
        0 if manager_state == "running" else 1
    )
    if (
        not allowed_states
        or any(
            not isinstance(state, str)
            or state not in known_manager_states
            for state in allowed_states
        )
        or len(set(allowed_states)) != len(allowed_states)
        or not isinstance(manager_state, str)
        or manager_state not in allowed_states
        or not _deep_exact_equal(
            manager.get("returncode"),
            expected_manager_returncode,
        )
        or not _deep_exact_equal(
            unexpected_failed_units,
            tuple(sorted(set(failed_units) - set(allowed_failed_units))),
        )
        or not _deep_exact_equal(
            scoped_failed_units,
            tuple(sorted(set(failed_units) & set(scoped_units))),
        )
    ):
        raise ValueError(
            "environment inventory manager state semantics changed"
        )
    inventory_body = dict(inventory)
    inventory_fingerprint = inventory_body.pop(
        "inventory_fingerprint",
        None,
    )
    inventory_blockers = inventory.get("blockers")
    if (
        set(inventory) != inventory_keys
        or inventory.get("schema_version") != ENVIRONMENT_INVENTORY_SCHEMA
        or not _is_sha256(inventory_fingerprint)
        or inventory_fingerprint != stable_fingerprint(inventory_body)
        or set(manager) != manager_keys
        or set(identity) != identity_keys
        or set(endpoint) != endpoint_keys
        or set(scope) != scope_keys
        or not isinstance(scope.get("require_target_ready"), bool)
        or not isinstance(inventory_blockers, list)
        or any(not isinstance(item, str) for item in inventory_blockers)
        or inventory_blockers != sorted(set(inventory_blockers))
        or inventory.get("passed") is not (not inventory_blockers)
        or any(
            inventory.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or any(
            set(shadow) != shadow_keys
            or any(not isinstance(value, str) for value in shadow.values())
            or shadow.get("Id") != unit
            for unit, shadow in shadows.items()
        )
    ):
        raise ValueError(
            "environment inventory closed identity schema changed"
        )
    gpu_body = dict(gpu)
    gpu_fingerprint = gpu_body.pop("snapshot_fingerprint", None)
    gpu_blockers = gpu.get("blockers")
    gpu_observations = gpu.get("observations")
    if (
        set(gpu) != gpu_keys
        or gpu.get("schema_version") != GPU_DOUBLE_SNAPSHOT_SCHEMA
        or not _is_sha256(gpu_fingerprint)
        or gpu_fingerprint != stable_fingerprint(gpu_body)
        or not isinstance(gpu.get("strict_all_gpu_consumers"), bool)
        or not isinstance(gpu_blockers, list)
        or any(not isinstance(item, str) for item in gpu_blockers)
        or gpu_blockers != sorted(set(gpu_blockers))
        or gpu.get("passed") is not (not gpu_blockers)
        or not isinstance(gpu_observations, list)
        or any(
            not isinstance(item, str)
            for item in gpu_observations
        )
        or gpu_observations != sorted(set(gpu_observations))
        or not _is_gpu_uuid(gpu.get("selected_gpu_uuid"))
        or not isinstance(gpu.get("allowed_unit_ids"), list)
        or any(
            not isinstance(unit, str)
            or _UNIT_RE.fullmatch(unit) is None
            for unit in gpu.get("allowed_unit_ids", ())
        )
    ):
        raise PermissionError(
            "environment inventory GPU snapshot sealing is invalid"
        )
    endpoint_integer_minimums = {
        "uid": 0,
        "runtime_directory_device": 0,
        "runtime_directory_inode": 1,
        "bus_device": 0,
        "bus_inode": 1,
    }
    identity_integer_minimums = {
        "pid": 1,
        "starttime_ticks": 1,
        "uid": 0,
    }
    if (
        not _is_nonbool_int(inventory.get("uid"), minimum=0)
        or not _is_nonbool_int(manager.get("returncode"), minimum=0)
        or any(
            not _is_nonbool_int(
                endpoint.get(field),
                minimum=minimum,
            )
            for field, minimum in endpoint_integer_minimums.items()
        )
        or any(
            not _is_nonbool_int(
                identity.get(field),
                minimum=minimum,
            )
            for field, minimum in identity_integer_minimums.items()
        )
        or not _is_nonbool_int(gpu.get("expected_uid"), minimum=0)
    ):
        raise ValueError(
            "environment inventory manager numeric identity is malformed"
        )
    for device in devices:
        minor_number = device.get("minor_number")
        if (
            set(device)
            != {
                "index",
                "uuid",
                "pci_bus_id",
                "compute_mode",
                "mig_mode",
                "driver_version",
                "minor_number",
                "mps_state",
            }
            or not _is_nonbool_int(device.get("index"), minimum=0)
            or (
                minor_number is not None
                and not _is_nonbool_int(minor_number, minimum=0)
            )
        ):
            raise ValueError(
                "environment inventory GPU numeric identity is malformed"
            )
    for app in first_apps + second_apps:
        memory = app.get("used_gpu_memory_mib")
        if (
            set(app)
            != {
                "pid",
                "gpu_uuid",
                "process_name",
                "used_gpu_memory_mib",
            }
            or not _is_nonbool_int(app.get("pid"), minimum=1)
            or not _is_gpu_uuid(app.get("gpu_uuid"))
            or not isinstance(app.get("process_name"), str)
            or (
                memory is not None
                and not _is_nonbool_int(memory, minimum=0)
            )
        ):
            raise ValueError(
                "environment inventory GPU process identity is malformed"
            )
    for mapping in mappings:
        if (
            set(mapping) != mapping_keys
            or not _is_nonbool_int(mapping.get("pid"), minimum=1)
            or not _is_nonbool_int(
                mapping.get("starttime_ticks"),
                minimum=1,
            )
            or not _is_nonbool_int(
                mapping.get("uid"),
                minimum=0,
            )
            or not _is_gpu_uuid(mapping.get("gpu_uuid"))
            or not isinstance(mapping.get("cgroup_path"), str)
            or (
                mapping.get("unit_id") is not None
                and (
                    not isinstance(mapping.get("unit_id"), str)
                    or _UNIT_RE.fullmatch(mapping["unit_id"]) is None
                )
            )
        ):
            raise ValueError(
                "environment inventory GPU mapping identity is malformed"
            )
    selected_devices = [
        device
        for device in devices
        if _deep_exact_equal(
            device.get("uuid"),
            gpu.get("selected_gpu_uuid"),
        )
    ]
    if len(selected_devices) != 1:
        raise ValueError(
            "environment inventory selected GPU is ambiguous"
        )
    selected_device = selected_devices[0]
    required_gpu_blockers: set[str] = set()
    if selected_device.get("minor_number") is None:
        required_gpu_blockers.add("selected_gpu_minor_number_unknown")
    selected_mps_state = selected_device.get("mps_state")
    if selected_mps_state == "unknown":
        required_gpu_blockers.add("selected_gpu_mps_state_unknown")
    elif selected_mps_state != "not_observed":
        required_gpu_blockers.add(
            f"selected_gpu_mps_state_blocked:{selected_mps_state}"
        )
    app_keys = {
        (app["pid"], app["gpu_uuid"])
        for app in first_apps + second_apps
    }
    mapping_keys_observed = [
        (mapping["pid"], mapping["gpu_uuid"])
        for mapping in mappings
    ]
    if (
        not required_gpu_blockers.issubset(set(gpu_blockers))
        or not set(gpu_blockers).issubset(set(inventory_blockers))
        or len(set(mapping_keys_observed)) != len(mapping_keys_observed)
        or any(
            key not in app_keys
            for key in mapping_keys_observed
        )
        or any(
            not _deep_exact_equal(
                mapping.get("uid"),
                gpu.get("expected_uid"),
            )
            for mapping in mappings
        )
    ):
        raise PermissionError(
            "environment inventory GPU semantics are not propagated"
        )
    for unit, shadow in shadows.items():
        try:
            _validated_nrestarts(
                shadow.get("NRestarts"),
                name=f"inventory {unit}",
            )
            parse_systemd_duration_usec(shadow.get("RestartUSec"))
        except ValueError as error:
            raise ValueError(
                "environment inventory restart evidence is malformed"
            ) from error


def audit_environment_once(
    contract: EnvironmentAuditContract,
    *,
    inventory_collector: Callable[..., dict[str, object]] = collect_environment_inventory,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ] = inspect_recovery_activation_guard,
) -> dict[str, object]:
    contract = validate_environment_audit_contract(contract)
    inventory = inventory_collector(
        selected_gpu_index=contract.selected_gpu_index,
        allowed_unit_ids=contract.allowed_unit_ids,
        target_unit_id=contract.target_unit_id,
        conflict_unit_ids=contract.conflict_unit_ids,
        dependency_unit_ids=contract.dependency_unit_ids,
        allowed_failed_unit_ids=contract.allowed_failed_unit_ids,
        allowed_manager_states=contract.allowed_manager_states,
        require_target_ready=contract.require_target_ready,
        strict_all_gpu_consumers=contract.strict_all_gpu_consumers,
        conflict_quiescence_mode=contract.quiescence_mode,
    )
    _validate_environment_inventory_identity_types(inventory)
    blockers = list(inventory.get("blockers", ()))
    try:
        manager = dict(inventory["manager"])
        endpoint = dict(manager["endpoint"])
        identity = dict(manager["identity"])
        scope = dict(inventory["unit_scope"])
        gpu = dict(inventory["gpu_snapshot"])
        devices = [dict(item) for item in gpu["devices"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("live environment inventory is malformed") from error
    expected_endpoint = {
        "uid": contract.uid,
        "runtime_directory": contract.runtime_directory,
        "runtime_directory_device": contract.runtime_directory_device,
        "runtime_directory_inode": contract.runtime_directory_inode,
        "bus_path": contract.bus_path,
        "bus_device": contract.bus_device,
        "bus_inode": contract.bus_inode,
    }
    observed_endpoint = {
        "uid": endpoint.get("uid"),
        "runtime_directory": endpoint.get("runtime_directory"),
        "runtime_directory_device": endpoint.get("runtime_directory_device"),
        "runtime_directory_inode": endpoint.get("runtime_directory_inode"),
        "bus_path": endpoint.get("bus_path"),
        "bus_device": endpoint.get("bus_device"),
        "bus_inode": endpoint.get("bus_inode"),
    }
    if not _deep_exact_equal(observed_endpoint, expected_endpoint):
        blockers.append("manager_endpoint_generation_changed")
    if (
        not _deep_exact_equal(inventory.get("uid"), contract.uid)
        or not _deep_exact_equal(
            inventory.get("boot_id"),
            contract.boot_id,
        )
    ):
        blockers.append("boot_or_uid_generation_changed")
    expected_manager = {
        "pid": contract.manager_pid,
        "starttime_ticks": contract.manager_starttime_ticks,
        "uid": contract.uid,
        "control_group": contract.manager_control_group,
    }
    observed_manager = {
        "pid": identity.get("pid"),
        "starttime_ticks": identity.get("starttime_ticks"),
        "uid": identity.get("uid"),
        "control_group": identity.get("control_group"),
    }
    if not _deep_exact_equal(observed_manager, expected_manager):
        blockers.append("user_manager_process_generation_changed")
    expected_scope = {
        "target_unit_id": contract.target_unit_id,
        "conflict_unit_ids": list(contract.conflict_unit_ids),
        "dependency_unit_ids": list(contract.dependency_unit_ids),
        "require_target_ready": contract.require_target_ready,
    }
    observed_scope = {
        "target_unit_id": scope.get("target_unit_id"),
        "conflict_unit_ids": scope.get("conflict_unit_ids"),
        "dependency_unit_ids": scope.get("dependency_unit_ids"),
        "require_target_ready": scope.get("require_target_ready"),
    }
    if not _deep_exact_equal(observed_scope, expected_scope):
        blockers.append("unit_scope_changed")
    if not _deep_exact_equal(
        manager.get("allowed_failed_unit_ids"),
        list(contract.allowed_failed_unit_ids),
    ):
        blockers.append("allowed_failed_unit_scope_changed")
    observed_failed = manager.get("failed_units")
    if (
        not isinstance(observed_failed, list)
        or any(not isinstance(unit, str) for unit in observed_failed)
        or not _deep_exact_equal(
            tuple(sorted(observed_failed)),
            contract.expected_failed_unit_ids,
        )
    ):
        blockers.append("failed_unit_set_changed")
    if not _deep_exact_equal(
        manager.get("allowed_states"),
        list(contract.allowed_manager_states),
    ):
        blockers.append("allowed_manager_states_changed")
    expected_gpu_scope = {
        "selected_gpu_uuid": contract.selected_gpu_uuid,
        "expected_uid": contract.uid,
        "allowed_unit_ids": list(contract.allowed_unit_ids),
        "strict_all_gpu_consumers": contract.strict_all_gpu_consumers,
    }
    observed_gpu_scope = {
        field: gpu.get(field)
        for field in expected_gpu_scope
    }
    if not _deep_exact_equal(observed_gpu_scope, expected_gpu_scope):
        blockers.append("gpu_policy_scope_changed")
    selected = [
        device for device in devices
        if _deep_exact_equal(
            device.get("index"),
            contract.selected_gpu_index,
        )
    ]
    selected_device: dict[str, object] | None
    if len(selected) != 1:
        blockers.append("selected_gpu_identity_absent_or_ambiguous")
        selected_device = None
    else:
        selected_device = selected[0]
        expected_gpu = (
            contract.selected_gpu_uuid,
            contract.selected_gpu_pci_bus_id,
            contract.selected_gpu_minor_number,
        )
        observed_gpu = (
            selected_device.get("uuid"),
            selected_device.get("pci_bus_id"),
            selected_device.get("minor_number"),
        )
        if not _deep_exact_equal(observed_gpu, expected_gpu):
            blockers.append("selected_gpu_physical_identity_changed")
    shadows = dict(scope.get("shadows", {}))
    projection: dict[str, object] = {}
    baseline_by_unit = dict(contract.cleanup_nrestarts_baseline)
    guard_observation: dict[str, object] | None = None
    if contract.cleanup_mode == RECOVERY_CLEANUP_MODE:
        try:
            guard_observation = activation_guard_reader(
                contract.activation_guard
            )
            if not _deep_exact_equal(
                guard_observation,
                {
                    **contract.activation_guard,
                    "file_type": "symlink",
                },
            ):
                blockers.append("activation_guard_identity_changed")
        except (OSError, PermissionError, ValueError) as error:
            blockers.append(
                f"activation_guard_unreadable_or_changed:{type(error).__name__}"
            )
    scoped_units = (
        (() if contract.target_unit_id is None else (contract.target_unit_id,))
        + contract.conflict_unit_ids
        + contract.dependency_unit_ids
    )
    for unit in scoped_units:
        shadow = dict(shadows.get(unit, {}))
        observed_restart_usec = parse_systemd_duration_usec(
            shadow.get("RestartUSec")
        )
        if observed_restart_usec > contract.maximum_restart_usec:
            blockers.append(f"restart_usec_exceeds_contract:{unit}")
        projection[unit] = {
            "NRestarts": shadow.get("NRestarts"),
            "activation_closure": {
                field: shadow.get(field)
                for field in ACTIVATION_CLOSURE_FIELDS
            },
            "state": {
                "LoadState": shadow.get("LoadState"),
                "ActiveState": shadow.get("ActiveState"),
                "SubState": shadow.get("SubState"),
                "UnitFileState": shadow.get("UnitFileState"),
            },
            "quiescence_mode": (
                contract.quiescence_mode
                if unit in contract.conflict_unit_ids
                else None
            ),
            "cleanup_nrestarts_baseline": baseline_by_unit.get(unit),
            "activation_guard": (
                dict(contract.activation_guard)
                if unit in contract.conflict_unit_ids
                else None
            ),
            "activation_guard_observation": (
                guard_observation
                if unit in contract.conflict_unit_ids
                else None
            ),
        }
        if unit in contract.conflict_unit_ids:
            expected_unit_file_state = (
                "enabled"
                if contract.cleanup_mode == RECOVERY_CLEANUP_MODE
                else "masked-runtime"
            )
            observed_state = {
                "LoadState": shadow.get("LoadState"),
                "ActiveState": shadow.get("ActiveState"),
                "SubState": shadow.get("SubState"),
                "UnitFileState": shadow.get("UnitFileState"),
            }
            expected_state = {
                "LoadState": "loaded",
                "ActiveState": "inactive",
                "SubState": "dead",
                "UnitFileState": expected_unit_file_state,
            }
            if not _deep_exact_equal(observed_state, expected_state):
                blockers.append(f"conflict_not_quiescent:{unit}")
            if not _deep_exact_equal(
                {
                    field: shadow.get(field)
                    for field in ACTIVATION_CLOSURE_FIELDS
                },
                _expected_conflict_activation_closure(
                    contract.cleanup_mode
                ),
            ):
                blockers.append(f"activation_closure_changed:{unit}")
            if not _deep_exact_equal(
                shadow.get("NRestarts"),
                baseline_by_unit.get(unit),
            ):
                blockers.append(f"nrestarts_changed_from_cleanup:{unit}")
    selected_uuid = gpu.get("selected_gpu_uuid")
    apps = list(gpu.get("first_apps", ())) + list(
        gpu.get("second_apps", ())
    )
    for app in apps:
        row = dict(app)
        if _deep_exact_equal(row.get("gpu_uuid"), selected_uuid):
            blockers.append(f"selected_gpu_not_empty:{row.get('pid')}")
    unique_blockers = sorted(set(blockers))
    body: dict[str, object] = {
        "schema_version": ENVIRONMENT_SINGLE_AUDIT_SCHEMA,
        "created_at_utc": utc_now(),
        "contract": json.loads(canonical_json(asdict(contract))),
        "inventory": inventory,
        "state_projection": projection,
        "blockers": unique_blockers,
        "passed": not unique_blockers,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {
        **body,
        "single_audit_fingerprint": stable_fingerprint(body),
    }


def validate_environment_single_audit(
    value: Mapping[str, object],
    *,
    contract: EnvironmentAuditContract,
) -> dict[str, object]:
    """Validate the exact sampled-audit and state-projection schemas."""

    contract = validate_environment_audit_contract(contract)
    payload = json.loads(canonical_json(dict(value)))
    body = dict(payload)
    fingerprint = body.pop("single_audit_fingerprint", None)
    exact_keys = {
        "schema_version",
        "created_at_utc",
        "contract",
        "inventory",
        "state_projection",
        "blockers",
        "passed",
        "D_R_payload_accessed",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
    }
    blockers = body.get("blockers")
    if (
        set(body) != exact_keys
        or body.get("schema_version") != ENVIRONMENT_SINGLE_AUDIT_SCHEMA
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
        or not _deep_exact_equal(
            body.get("contract"),
            json.loads(canonical_json(asdict(contract))),
        )
        or not isinstance(body.get("inventory"), Mapping)
        or not isinstance(body.get("state_projection"), Mapping)
        or not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or any(not isinstance(item, str) for item in blockers)
        or body.get("passed") is not (not blockers)
        or any(
            body.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise ValueError("single environment audit closed schema changed")
    scoped_units = (
        (() if contract.target_unit_id is None else (contract.target_unit_id,))
        + contract.conflict_unit_ids
        + contract.dependency_unit_ids
    )
    projection = dict(body["state_projection"])
    if set(projection) != set(scoped_units):
        raise ValueError("single environment projection scope changed")
    baseline = dict(contract.cleanup_nrestarts_baseline)
    closure_keys = set(ACTIVATION_CLOSURE_FIELDS)
    state_keys = {
        "LoadState", "ActiveState", "SubState", "UnitFileState",
    }
    projection_keys = {
        "NRestarts",
        "activation_closure",
        "state",
        "quiescence_mode",
        "cleanup_nrestarts_baseline",
        "activation_guard",
        "activation_guard_observation",
    }
    for unit in scoped_units:
        row = dict(projection[unit])
        if (
            set(row) != projection_keys
            or not isinstance(row.get("activation_closure"), Mapping)
            or set(row["activation_closure"]) != closure_keys
            or any(
                not isinstance(value, str)
                for value in row["activation_closure"].values()
            )
            or not isinstance(row.get("state"), Mapping)
            or set(row["state"]) != state_keys
            or any(
                not isinstance(value, str)
                for value in row["state"].values()
            )
        ):
            raise ValueError(
                f"single environment projection schema changed:{unit}"
            )
        try:
            _validated_nrestarts(
                row.get("NRestarts"),
                name=f"single projection {unit}",
            )
        except ValueError as error:
            raise ValueError(
                f"single environment NRestarts malformed:{unit}"
            ) from error
        is_conflict = unit in contract.conflict_unit_ids
        if is_conflict:
            if (
                not _deep_exact_equal(
                    row.get("quiescence_mode"),
                    contract.quiescence_mode,
                )
                or not _deep_exact_equal(
                    row.get("cleanup_nrestarts_baseline"),
                    baseline.get(unit),
                )
                or not _deep_exact_equal(
                    row.get("activation_guard"),
                    contract.activation_guard,
                )
            ):
                raise PermissionError(
                    f"single environment quiescence binding changed:{unit}"
                )
            observation = row.get("activation_guard_observation")
            if contract.cleanup_mode == RECOVERY_CLEANUP_MODE:
                expected_observation = {
                    **contract.activation_guard,
                    "file_type": "symlink",
                }
                if (
                    not isinstance(observation, Mapping)
                    or set(observation)
                    != _RECOVERY_GUARD_KEYS | {"file_type"}
                    or not _deep_exact_equal(
                        dict(observation),
                        expected_observation,
                    )
                ):
                    raise ValueError(
                        f"single activation guard observation malformed:{unit}"
                    )
            elif observation is not None:
                raise ValueError(
                    f"normal activation guard observation is not empty:{unit}"
                )
        elif any(
            row.get(field) is not None
            for field in (
                "quiescence_mode",
                "cleanup_nrestarts_baseline",
                "activation_guard",
                "activation_guard_observation",
            )
        ):
            raise ValueError(
                f"non-conflict quiescence projection is not empty:{unit}"
            )
    return payload


def build_environment_policy(
    contract: EnvironmentAuditContract,
    *,
    precleanup_root_binding: Mapping[str, object],
    cleanup_root_binding: Mapping[str, object],
    toolchain_binding: Mapping[str, object],
    minimum_sample_count: int,
    sample_interval_seconds: float,
) -> dict[str, object]:
    root_binding = dict(precleanup_root_binding)
    root_keys = {
        "path", "device", "inode", "size", "mtime_ns", "file_sha256",
        "receipt_fingerprint", "inventory_fingerprint",
    }
    if (
        set(root_binding) != root_keys
        or not isinstance(root_binding.get("path"), str)
        or not Path(root_binding["path"]).is_absolute()
        or not _is_sha256(root_binding.get("file_sha256"))
        or not _is_sha256(root_binding.get("receipt_fingerprint"))
        or not _is_sha256(root_binding.get("inventory_fingerprint"))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                root_binding.get("device"), root_binding.get("inode"),
                root_binding.get("size"), root_binding.get("mtime_ns"),
            )
        )
    ):
        raise ValueError("precleanup policy root binding is malformed")
    cleanup_binding = dict(cleanup_root_binding)
    cleanup_root_keys = {
        "path", "device", "inode", "size", "mtime_ns", "file_sha256",
        "cleanup_receipt_fingerprint",
    }
    if (
        set(cleanup_binding) != cleanup_root_keys
        or not isinstance(cleanup_binding.get("path"), str)
        or not Path(cleanup_binding["path"]).is_absolute()
        or not _is_sha256(cleanup_binding.get("file_sha256"))
        or not _is_sha256(
            cleanup_binding.get("cleanup_receipt_fingerprint")
        )
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in (
                cleanup_binding.get("device"),
                cleanup_binding.get("inode"),
                cleanup_binding.get("size"),
                cleanup_binding.get("mtime_ns"),
            )
        )
    ):
        raise ValueError("cleanup policy root binding is malformed")
    toolchain: dict[str, dict[str, object]] = {}
    if set(toolchain_binding) != {"runtime_environment", "python", "systemctl", "nvidia_smi"}:
        raise ValueError("policy toolchain binding set changed")
    for label, value in toolchain_binding.items():
        item = dict(value)
        if set(item) != {"path", "file_sha256"} or not isinstance(item["path"], str) or not Path(item["path"]).is_absolute() or not _is_sha256(item["file_sha256"]):
            raise ValueError(f"policy {label} binding is malformed")
        toolchain[label] = item
    contract = validate_environment_audit_contract(contract)
    if (
        isinstance(minimum_sample_count, bool)
        or not isinstance(minimum_sample_count, int)
        or minimum_sample_count < 2
        or isinstance(sample_interval_seconds, bool)
        or not isinstance(sample_interval_seconds, (int, float))
        or not math.isfinite(float(sample_interval_seconds))
        or float(sample_interval_seconds) < 0.0
    ):
        raise ValueError("environment stability sampling policy is malformed")
    minimum_window = (minimum_sample_count - 1) * float(sample_interval_seconds)
    if (
        contract.target_unit_id is None
        or contract.selected_gpu_index != 0
        or len(contract.conflict_unit_ids) != 1
        or contract.strict_all_gpu_consumers
        or minimum_window < contract.required_stability_window_usec / 1_000_000.0
    ):
        raise ValueError("environment policy invariants are not satisfied")
    body: dict[str, object] = {
        "schema_version": ENVIRONMENT_POLICY_SCHEMA,
        "candidate": "GCR-PACRE-v24",
        "scope": "runtime-environment-stability",
        "created_at_utc": utc_now(),
        "uid": contract.uid,
        "precleanup_root": root_binding,
        "cleanup_root": cleanup_binding,
        "toolchain": toolchain,
        "manager_generation": {
            "bus_path": contract.bus_path,
            "bus_device": contract.bus_device,
            "bus_inode": contract.bus_inode,
            "boot_id": contract.boot_id,
            "runtime_directory": contract.runtime_directory,
            "runtime_directory_device": contract.runtime_directory_device,
            "runtime_directory_inode": contract.runtime_directory_inode,
            "pid": contract.manager_pid,
            "starttime_ticks": contract.manager_starttime_ticks,
            "control_group": contract.manager_control_group,
        },
        "unit_scope": {
            "target_unit_id": contract.target_unit_id,
            "conflict_unit_ids": list(contract.conflict_unit_ids),
            "dependency_unit_ids": list(contract.dependency_unit_ids),
            "allowed_failed_unit_ids": list(contract.allowed_failed_unit_ids),
            "allowed_unit_ids": list(contract.allowed_unit_ids),
            "expected_failed_unit_ids": list(contract.expected_failed_unit_ids),
            "require_target_ready": contract.require_target_ready,
        },
        "selected_gpu": {
            "index": contract.selected_gpu_index,
            "uuid": contract.selected_gpu_uuid,
            "pci_bus_id": contract.selected_gpu_pci_bus_id,
            "minor_number": contract.selected_gpu_minor_number,
        },
        "allowed_manager_states": list(contract.allowed_manager_states),
        "strict_all_gpu_consumers": contract.strict_all_gpu_consumers,
        "postcleanup_contract": {
            "cleanup_mode": contract.cleanup_mode,
            "quiescence_mode": contract.quiescence_mode,
            "cleanup_nrestarts_baseline": [
                list(item) for item in contract.cleanup_nrestarts_baseline
            ],
            "activation_guard": dict(contract.activation_guard),
        },
        "sampling": {
            "minimum_sample_count": minimum_sample_count,
            "sample_interval_seconds": float(sample_interval_seconds),
            "minimum_window_seconds": minimum_window,
            "maximum_restart_usec": contract.maximum_restart_usec,
            "maximum_trigger_usec": contract.maximum_trigger_usec,
            "required_stability_window_usec": contract.required_stability_window_usec,
            "window_basis": "sealed-precleanup-restart-and-trigger-period",
        },
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {
        **body,
        "policy_fingerprint": stable_fingerprint(body),
    }


def validate_environment_policy(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    body = dict(payload)
    fingerprint = body.pop("policy_fingerprint", None)
    if not _is_sha256(fingerprint):
        raise ValueError("environment policy fingerprint is malformed")
    try:
        manager = dict(body["manager_generation"])
        scope = dict(body["unit_scope"])
        gpu = dict(body["selected_gpu"])
        sampling = dict(body["sampling"])
        precleanup_root = dict(body["precleanup_root"])
        cleanup_root = dict(body["cleanup_root"])
        postcleanup = dict(body["postcleanup_contract"])
        toolchain = dict(body["toolchain"])
        contract = EnvironmentAuditContract(
            uid=body["uid"],
            boot_id=manager["boot_id"],
            runtime_directory=manager["runtime_directory"],
            runtime_directory_device=manager["runtime_directory_device"],
            runtime_directory_inode=manager["runtime_directory_inode"],
            bus_path=manager["bus_path"],
            bus_device=manager["bus_device"],
            bus_inode=manager["bus_inode"],
            manager_pid=manager["pid"],
            manager_starttime_ticks=manager["starttime_ticks"],
            manager_control_group=manager["control_group"],
            selected_gpu_index=gpu["index"],
            selected_gpu_uuid=gpu["uuid"],
            selected_gpu_pci_bus_id=gpu["pci_bus_id"],
            selected_gpu_minor_number=gpu["minor_number"],
            target_unit_id=scope["target_unit_id"],
            conflict_unit_ids=tuple(scope["conflict_unit_ids"]),
            dependency_unit_ids=tuple(scope["dependency_unit_ids"]),
            allowed_failed_unit_ids=tuple(scope["allowed_failed_unit_ids"]),
            expected_failed_unit_ids=tuple(scope["expected_failed_unit_ids"]),
            maximum_restart_usec=sampling["maximum_restart_usec"],
            maximum_trigger_usec=sampling["maximum_trigger_usec"],
            required_stability_window_usec=sampling["required_stability_window_usec"],
            cleanup_mode=postcleanup["cleanup_mode"],
            quiescence_mode=postcleanup["quiescence_mode"],
            cleanup_nrestarts_baseline=tuple(
                (str(item[0]), str(item[1]))
                for item in postcleanup["cleanup_nrestarts_baseline"]
            ),
            activation_guard=dict(postcleanup["activation_guard"]),
            allowed_unit_ids=tuple(scope["allowed_unit_ids"]),
            allowed_manager_states=tuple(body["allowed_manager_states"]),
            require_target_ready=scope["require_target_ready"],
            strict_all_gpu_consumers=body["strict_all_gpu_consumers"],
        )
        minimum_sample_count = sampling["minimum_sample_count"]
        sample_interval_seconds = sampling["sample_interval_seconds"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("environment policy structure is malformed") from error
    rebuilt = build_environment_policy(
        contract,
        precleanup_root_binding=precleanup_root,
        cleanup_root_binding=cleanup_root,
        toolchain_binding=toolchain,
        minimum_sample_count=minimum_sample_count,
        sample_interval_seconds=sample_interval_seconds,
    )
    rebuilt_body = dict(rebuilt)
    rebuilt_body.pop("policy_fingerprint")
    rebuilt_body["created_at_utc"] = body.get("created_at_utc")
    if body != rebuilt_body:
        raise ValueError("environment policy closed schema changed")
    if fingerprint != stable_fingerprint(body):
        raise PermissionError("environment policy fingerprint is invalid")
    return payload


def write_environment_policy(
    path: str | Path,
    policy: Mapping[str, object],
) -> dict[str, object]:
    validated = validate_environment_policy(policy)
    body = dict(validated)
    body.pop("policy_fingerprint")
    return write_create_once_receipt(
        path,
        body,
        fingerprint_field="policy_fingerprint",
    )


def prepare_environment_stability_contract(
    precleanup_inventory_receipt_path: str | Path,
    cleanup_receipt_path: str | Path,
    *,
    selected_gpu_index: int,
    target_unit_id: str | None,
    conflict_unit_ids: Sequence[str],
    dependency_unit_ids: Sequence[str],
    allowed_failed_unit_ids: Sequence[str],
    allowed_unit_ids: Sequence[str] = (),
    allowed_manager_states: Sequence[str] = ("running", "degraded"),
    require_target_ready: bool = False,
    strict_all_gpu_consumers: bool = False,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ] = inspect_recovery_activation_guard,
) -> tuple[EnvironmentAuditContract, dict[str, object]]:
    pre_path = Path(precleanup_inventory_receipt_path).absolute()
    cleanup_path = Path(cleanup_receipt_path).absolute()
    precleanup, pre_evidence = load_sealed_receipt_with_evidence(pre_path)
    cleanup, cleanup_evidence = load_sealed_receipt_with_evidence(
        cleanup_path,
        fingerprint_field="cleanup_receipt_fingerprint",
    )
    if precleanup.get("schema_version") != ENVIRONMENT_RECEIPT_SCHEMA:
        raise ValueError("precleanup receipt schema changed")
    if any(
        precleanup.get(field) is not False
        for field in ("D_R_payload_accessed", "D_V_payload_accessed", "D_T_payload_accessed")
    ) or any(
        cleanup.get(field) is not False
        for field in ("D_R_payload_accessed", "D_V_payload_accessed", "D_T_payload_accessed")
    ):
        raise PermissionError("environment root evidence accessed payload")
    inventory_value = precleanup.get("inventory")
    if not isinstance(inventory_value, Mapping):
        raise ValueError("precleanup receipt has no inventory")
    inventory = dict(inventory_value)
    cleanup_semantics = validate_cleanup_receipt_for_environment(
        cleanup,
        uid=inventory.get("uid"),
        conflict_unit_ids=conflict_unit_ids,
    )
    partial_lineage = cleanup_semantics["partial_lineage"]
    if partial_lineage is not None:
        verify_partial_lineage_roots(partial_lineage)
        observed_guard = activation_guard_reader(
            cleanup_semantics["activation_guard"]
        )
        if not _deep_exact_equal(
            observed_guard,
            {
                **cleanup_semantics["activation_guard"],
                "file_type": "symlink",
            },
        ):
            raise PermissionError(
                "recovery activation guard changed before policy sampling"
            )
    expected_precleanup_blockers = [
        f"scoped_blocker_unit_not_quiescent:{unit}"
        for unit in conflict_unit_ids
    ]
    if (
        precleanup.get("passed") is not False
        or inventory.get("passed") is not False
        or not _deep_exact_equal(
            inventory.get("blockers"),
            expected_precleanup_blockers,
        )
    ):
        raise PermissionError("precleanup is not the exact conflict-only FAIL root")
    contract = environment_audit_contract_from_inventory(
        inventory,
        selected_gpu_index=selected_gpu_index,
        target_unit_id=target_unit_id,
        conflict_unit_ids=conflict_unit_ids,
        dependency_unit_ids=dependency_unit_ids,
        allowed_failed_unit_ids=allowed_failed_unit_ids,
        cleanup_mode=cleanup_semantics["cleanup_mode"],
        quiescence_mode=cleanup_semantics["quiescence_mode"],
        cleanup_nrestarts_baseline=cleanup_semantics[
            "cleanup_nrestarts_baseline"
        ],
        activation_guard=cleanup_semantics["activation_guard"],
        allowed_unit_ids=allowed_unit_ids,
        allowed_manager_states=allowed_manager_states,
        require_target_ready=require_target_ready,
        strict_all_gpu_consumers=strict_all_gpu_consumers,
    )
    try:
        binding = dict(precleanup["environment_binding"])
        manager = dict(inventory["manager"])
        endpoint = dict(manager["endpoint"])
        identity = dict(manager["identity"])
        after = dict(cleanup["after"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("environment root binding is malformed") from error
    expected_binding = {
        "inventory_fingerprint": inventory["inventory_fingerprint"],
        "boot_id": contract.boot_id,
        "runtime_directory": contract.runtime_directory,
        "runtime_directory_device": contract.runtime_directory_device,
        "runtime_directory_inode": contract.runtime_directory_inode,
        "manager_identity": {
            "pid": contract.manager_pid,
            "starttime_ticks": contract.manager_starttime_ticks,
            "uid": contract.uid,
            "control_group": contract.manager_control_group,
        },
    }
    if not _deep_exact_equal(binding, expected_binding):
        raise PermissionError("precleanup environment binding changed")
    expected_cleanup_generation = {
        "boot_id": contract.boot_id,
        "identity": expected_binding["manager_identity"],
        "endpoint": endpoint,
    }
    if (
        not _deep_exact_equal(
            cleanup.get("boot_id"),
            contract.boot_id,
        )
        or not _deep_exact_equal(
            cleanup.get("manager_generation"),
            expected_cleanup_generation,
        )
    ):
        raise PermissionError("cleanup manager generation binding changed")
    root_evidence = {
        "precleanup_inventory_receipt": {
            **pre_evidence,
            "inventory_fingerprint": inventory["inventory_fingerprint"],
        },
        "cleanup_receipt": cleanup_evidence,
    }
    return contract, root_evidence


def evaluate_environment_stability(
    contract: EnvironmentAuditContract,
    root_evidence: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
    *,
    sample_interval_seconds: float,
    sample_monotonic_seconds: Sequence[float],
) -> dict[str, object]:
    contract = validate_environment_audit_contract(contract)
    rows = tuple(dict(sample) for sample in samples)
    if len(rows) < 2:
        raise ValueError("environment stability requires at least two samples")
    if (
        isinstance(sample_interval_seconds, bool)
        or not isinstance(sample_interval_seconds, (int, float))
        or not math.isfinite(float(sample_interval_seconds))
        or float(sample_interval_seconds) < 0.0
    ):
        raise ValueError("sample interval must be finite and nonnegative")
    timestamps = tuple(sample_monotonic_seconds)
    if (
        len(timestamps) != len(rows)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in timestamps
        )
        or any(
            float(current) < float(previous)
            for previous, current in zip(timestamps, timestamps[1:])
        )
    ):
        raise ValueError("sample monotonic timestamps are malformed")
    observed_window_seconds = float(timestamps[-1]) - float(timestamps[0])
    blockers: list[str] = []
    if observed_window_seconds < contract.required_stability_window_usec / 1_000_000.0:
        blockers.append("observed_stability_window_too_short")
    expected_contract = json.loads(canonical_json(asdict(contract)))
    baseline: dict[str, object] | None = None
    cleanup_baseline = dict(contract.cleanup_nrestarts_baseline)
    for index, sample in enumerate(rows):
        try:
            validate_environment_single_audit(sample, contract=contract)
        except (PermissionError, TypeError, ValueError):
            blockers.append(f"sample_fingerprint_invalid:{index}")
            continue
        if not _deep_exact_equal(
            sample.get("contract"),
            expected_contract,
        ):
            blockers.append(f"sample_contract_changed:{index}")
        if sample.get("passed") is not True:
            blockers.append(f"single_audit_failed:{index}")
        projection_value = sample.get("state_projection")
        if not isinstance(projection_value, Mapping):
            blockers.append(f"sample_projection_malformed:{index}")
            continue
        projection = dict(projection_value)
        if baseline is None:
            baseline = projection
        else:
            for unit in sorted(set(baseline) | set(projection)):
                before = dict(baseline.get(unit, {}))
                current = dict(projection.get(unit, {}))
                if not _deep_exact_equal(
                    before.get("NRestarts"),
                    current.get("NRestarts"),
                ):
                    blockers.append(f"nrestarts_drift:{unit}:{index}")
                if not _deep_exact_equal(
                    before.get("activation_closure"),
                    current.get("activation_closure"),
                ):
                    blockers.append(f"activation_closure_drift:{unit}:{index}")
        for unit in contract.conflict_unit_ids:
            unit_projection = dict(projection.get(unit, {}))
            state = dict(unit_projection.get("state", {}))
            expected_unit_file_state = (
                "enabled"
                if contract.cleanup_mode == RECOVERY_CLEANUP_MODE
                else "masked-runtime"
            )
            if not _deep_exact_equal(
                state,
                {
                    "LoadState": "loaded",
                    "ActiveState": "inactive",
                    "SubState": "dead",
                    "UnitFileState": expected_unit_file_state,
                },
            ):
                blockers.append(f"conflict_not_quiescent:{unit}:{index}")
            if not _deep_exact_equal(
                unit_projection.get("NRestarts"),
                cleanup_baseline.get(unit),
            ):
                blockers.append(
                    f"nrestarts_changed_from_cleanup:{unit}:{index}"
                )
            if not _deep_exact_equal(
                unit_projection.get("activation_closure"),
                _expected_conflict_activation_closure(
                    contract.cleanup_mode
                ),
            ):
                blockers.append(
                    f"activation_closure_changed:{unit}:{index}"
                )
            if not _deep_exact_equal(
                {
                    "quiescence_mode": unit_projection.get(
                        "quiescence_mode"
                    ),
                    "cleanup_nrestarts_baseline": unit_projection.get(
                        "cleanup_nrestarts_baseline"
                    ),
                    "activation_guard": unit_projection.get(
                        "activation_guard"
                    ),
                },
                {
                    "quiescence_mode": contract.quiescence_mode,
                    "cleanup_nrestarts_baseline": cleanup_baseline.get(
                        unit
                    ),
                    "activation_guard": contract.activation_guard,
                },
            ):
                blockers.append(
                    f"quiescence_projection_changed:{unit}:{index}"
                )
            expected_guard_observation = (
                {
                    **contract.activation_guard,
                    "file_type": "symlink",
                }
                if contract.cleanup_mode == RECOVERY_CLEANUP_MODE
                else None
            )
            if not _deep_exact_equal(
                unit_projection.get("activation_guard_observation"),
                expected_guard_observation,
            ):
                blockers.append(
                    f"activation_guard_changed:{unit}:{index}"
                )
        inventory_value = sample.get("inventory")
        if not isinstance(inventory_value, Mapping):
            blockers.append(f"sample_inventory_malformed:{index}")
            continue
        gpu = dict(inventory_value["gpu_snapshot"])
        selected_uuid = gpu.get("selected_gpu_uuid")
        apps = list(gpu.get("first_apps", ())) + list(gpu.get("second_apps", ()))
        for app in apps:
            row = dict(app)
            if not _deep_exact_equal(
                row.get("gpu_uuid"),
                selected_uuid,
            ):
                continue
            blockers.append(f"selected_gpu_not_empty:{row.get('pid')}:{index}")
    unique_blockers = sorted(set(blockers))
    body: dict[str, object] = {
        "schema_version": ENVIRONMENT_STABILITY_RECEIPT_SCHEMA,
        "receipt_kind": "sampled",
        "created_at_utc": utc_now(),
        "contract": expected_contract,
        "root_evidence": dict(root_evidence),
        "sample_count": len(rows),
        "sample_interval_seconds": float(sample_interval_seconds),
        "minimum_window_seconds": (len(rows) - 1) * float(sample_interval_seconds),
        "sample_monotonic_seconds": [float(value) for value in timestamps],
        "observed_window_seconds": observed_window_seconds,
        "required_stability_window_usec": contract.required_stability_window_usec,
        "samples": list(rows),
        "blockers": unique_blockers,
        "passed": not unique_blockers,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {
        **body,
        "stability_receipt_fingerprint": stable_fingerprint(body),
    }

def validate_environment_stability_receipt(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the closed sampled or canonical-exception receipt schema."""

    payload = json.loads(canonical_json(dict(value)))
    body = dict(payload)
    fingerprint = body.pop("stability_receipt_fingerprint", None)
    if not _is_sha256(fingerprint) or fingerprint != stable_fingerprint(body):
        raise PermissionError("stability receipt fingerprint is invalid")
    common_valid = (
        body.get("schema_version") == ENVIRONMENT_STABILITY_RECEIPT_SCHEMA
        and body.get("payload_authority") == "none"
        and all(
            body.get(field) is False
            for field in (
                "D_R_payload_accessed", "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    )
    if not common_valid:
        raise ValueError("stability receipt no-payload schema is malformed")
    if body.get("receipt_kind") == "exception":
        expected = {
            "schema_version", "receipt_kind", "created_at_utc", "command",
            "requested_roots", "requested_scope", "sample_count",
            "sample_interval_seconds", "samples", "blockers", "passed",
            "error_type", "error_message", "payload_authority",
            "D_R_payload_accessed", "D_V_payload_accessed",
            "D_T_payload_accessed",
        }
        if (
            set(body) != expected
            or body.get("command") != "stability-gate"
            or body.get("sample_count") != 0
            or body.get("samples") != []
            or body.get("blockers") != ["stability_gate_exception"]
            or body.get("passed") is not False
            or not isinstance(body.get("error_type"), str)
            or not isinstance(body.get("error_message"), str)
        ):
            raise ValueError("exception stability receipt closed schema changed")
        return payload
    expected = {
        "schema_version", "receipt_kind", "created_at_utc", "contract",
        "root_evidence", "sample_count", "sample_interval_seconds",
        "minimum_window_seconds", "sample_monotonic_seconds",
        "observed_window_seconds", "required_stability_window_usec",
        "samples", "blockers", "passed", "payload_authority",
        "D_R_payload_accessed", "D_V_payload_accessed",
        "D_T_payload_accessed",
    }
    if body.get("receipt_kind") != "sampled" or set(body) != expected:
        raise ValueError("sampled stability receipt closed schema changed")
    contract_value = dict(body["contract"])
    tuple_fields = {
        "conflict_unit_ids", "dependency_unit_ids", "allowed_failed_unit_ids",
        "expected_failed_unit_ids", "allowed_unit_ids", "allowed_manager_states",
    }
    for field in tuple_fields:
        contract_value[field] = tuple(contract_value[field])
    contract_value["cleanup_nrestarts_baseline"] = tuple(
        (str(item[0]), str(item[1]))
        for item in contract_value["cleanup_nrestarts_baseline"]
    )
    contract_value["activation_guard"] = dict(
        contract_value["activation_guard"]
    )
    contract = validate_environment_audit_contract(
        EnvironmentAuditContract(**contract_value)
    )
    normalized_contract = json.loads(canonical_json(asdict(contract)))
    timestamps = body.get("sample_monotonic_seconds")
    samples = body.get("samples")
    blockers = body.get("blockers")
    if (
        not _deep_exact_equal(body.get("contract"), normalized_contract)
        or isinstance(body.get("sample_count"), bool)
        or not isinstance(body.get("sample_count"), int)
        or body["sample_count"] < 2
        or isinstance(body.get("sample_interval_seconds"), bool)
        or not isinstance(body.get("sample_interval_seconds"), (int, float))
        or not math.isfinite(float(body["sample_interval_seconds"]))
        or float(body["sample_interval_seconds"]) < 0.0
        or body.get("minimum_window_seconds")
        != (body["sample_count"] - 1) * float(body["sample_interval_seconds"])
        or body["minimum_window_seconds"]
        < body["required_stability_window_usec"] / 1_000_000.0
        or isinstance(body.get("required_stability_window_usec"), bool)
        or not isinstance(body.get("required_stability_window_usec"), int)
        or body["required_stability_window_usec"] < 0
        or not isinstance(samples, list)
        or not isinstance(timestamps, list)
        or len(samples) != body.get("sample_count")
        or len(timestamps) != len(samples)
        or len(samples) < 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item)) or float(item) < 0.0
            for item in timestamps
        )
        or any(float(b) < float(a) for a, b in zip(timestamps, timestamps[1:]))
        or isinstance(body.get("observed_window_seconds"), bool)
        or not isinstance(body.get("observed_window_seconds"), (int, float))
        or not math.isfinite(float(body["observed_window_seconds"]))
        or body.get("observed_window_seconds")
        != float(timestamps[-1]) - float(timestamps[0])
        or body.get("required_stability_window_usec")
        != contract.required_stability_window_usec
        or not isinstance(blockers, list)
        or (
            ("observed_stability_window_too_short" in blockers)
            is not (
                body["observed_window_seconds"]
                < body["required_stability_window_usec"] / 1_000_000.0
            )
        )
        or blockers != sorted(set(blockers))
        or any(not isinstance(item, str) for item in blockers)
        or body.get("passed") is not (not blockers)
    ):
        raise ValueError("sampled stability receipt semantics are malformed")
    roots = dict(body["root_evidence"])
    if set(roots) != {"precleanup_inventory_receipt", "cleanup_receipt", "policy"}:
        raise ValueError("stability root evidence set changed")
    base_keys = {"path", "device", "inode", "size", "mtime_ns", "file_sha256"}
    root_specific = {
        "precleanup_inventory_receipt": {"receipt_fingerprint", "inventory_fingerprint"},
        "cleanup_receipt": {"cleanup_receipt_fingerprint"},
        "policy": {"policy_fingerprint"},
    }
    for name, extra_keys in root_specific.items():
        evidence = dict(roots[name])
        if (
            set(evidence) != base_keys | extra_keys
            or not isinstance(evidence.get("path"), str)
            or not Path(evidence["path"]).is_absolute()
            or any(not _is_sha256(evidence.get(field)) for field in {"file_sha256"} | extra_keys)
            or any(
                isinstance(evidence.get(field), bool)
                or not isinstance(evidence.get(field), int)
                or evidence[field] < 0
                for field in ("device", "inode", "size", "mtime_ns")
            )
        ):
            raise ValueError(f"stability {name} evidence is malformed")
    replayed_samples: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        try:
            canonical_sample = validate_environment_single_audit(
                sample,
                contract=contract,
            )
            inventory = dict(canonical_sample["inventory"])
            inventory_body = dict(inventory)
            inventory_fingerprint = inventory_body.pop(
                "inventory_fingerprint",
                None,
            )
            inventory_blockers = inventory.get("blockers")
            if (
                inventory.get("schema_version")
                != ENVIRONMENT_INVENTORY_SCHEMA
                or not _is_sha256(inventory_fingerprint)
                or inventory_fingerprint
                != stable_fingerprint(inventory_body)
                or not isinstance(inventory_blockers, list)
                or inventory_blockers
                != sorted(set(inventory_blockers))
                or any(
                    not isinstance(item, str)
                    for item in inventory_blockers
                )
                or inventory.get("passed") is not (not inventory_blockers)
                or any(
                    inventory.get(field) is not False
                    for field in (
                        "D_R_payload_accessed",
                        "D_V_payload_accessed",
                        "D_T_payload_accessed",
                    )
                )
            ):
                raise PermissionError(
                    "sample replay inventory sealing is invalid"
                )

            def replay_inventory_collector(
                **_kwargs: object,
            ) -> dict[str, object]:
                return json.loads(canonical_json(inventory))

            def replay_activation_guard(
                expected: Mapping[str, object],
            ) -> dict[str, object]:
                if not _deep_exact_equal(
                    dict(expected),
                    contract.activation_guard,
                ):
                    raise PermissionError(
                        "sample replay activation guard changed"
                    )
                return {
                    **contract.activation_guard,
                    "file_type": "symlink",
                }

            replayed = audit_environment_once(
                contract,
                inventory_collector=replay_inventory_collector,
                activation_guard_reader=replay_activation_guard,
            )
            replayed_body = dict(replayed)
            replayed_body.pop("single_audit_fingerprint")
            replayed_body["created_at_utc"] = canonical_sample[
                "created_at_utc"
            ]
            replayed = {
                **replayed_body,
                "single_audit_fingerprint": stable_fingerprint(
                    replayed_body
                ),
            }
            replayed = json.loads(canonical_json(replayed))
        except (PermissionError, TypeError, ValueError) as error:
            raise PermissionError(
                f"stability sample semantic replay invalid:{index}"
            ) from error
        if not _deep_exact_equal(replayed, canonical_sample):
            raise PermissionError(
                f"stability sample semantic replay changed:{index}"
            )
        replayed_samples.append(replayed)
    replayed_stability = evaluate_environment_stability(
        contract,
        roots,
        replayed_samples,
        sample_interval_seconds=body["sample_interval_seconds"],
        sample_monotonic_seconds=timestamps,
    )
    replayed_stability_body = dict(replayed_stability)
    replayed_stability_body.pop("stability_receipt_fingerprint")
    replayed_stability_body["created_at_utc"] = body["created_at_utc"]
    replayed_stability = {
        **replayed_stability_body,
        "stability_receipt_fingerprint": stable_fingerprint(
            replayed_stability_body
        ),
    }
    replayed_stability = json.loads(canonical_json(replayed_stability))
    if not _deep_exact_equal(replayed_stability, payload):
        raise PermissionError(
            "stability receipt semantic replay changed"
        )
    return payload



def run_environment_stability_gate(
    precleanup_inventory_receipt_path: str | Path,
    cleanup_receipt_path: str | Path,
    *,
    selected_gpu_index: int,
    target_unit_id: str | None,
    conflict_unit_ids: Sequence[str],
    dependency_unit_ids: Sequence[str],
    allowed_failed_unit_ids: Sequence[str],
    sample_count: int,
    sample_interval_seconds: float,
    policy_path: str | Path,
    allowed_unit_ids: Sequence[str] = (),
    allowed_manager_states: Sequence[str] = ("running", "degraded"),
    require_target_ready: bool = False,
    strict_all_gpu_consumers: bool = False,
    inventory_collector: Callable[..., dict[str, object]] = collect_environment_inventory,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ] = inspect_recovery_activation_guard,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 2
        or isinstance(sample_interval_seconds, bool)
        or not isinstance(sample_interval_seconds, (int, float))
        or not math.isfinite(float(sample_interval_seconds))
        or float(sample_interval_seconds) < 0.0
    ):
        raise ValueError("stability sample count or interval is invalid")
    if policy_path is None:
        raise PermissionError("sealed environment policy is required")
    contract, root_evidence = prepare_environment_stability_contract(
        precleanup_inventory_receipt_path,
        cleanup_receipt_path,
        selected_gpu_index=selected_gpu_index,
        target_unit_id=target_unit_id,
        conflict_unit_ids=conflict_unit_ids,
        dependency_unit_ids=dependency_unit_ids,
        allowed_failed_unit_ids=allowed_failed_unit_ids,
        allowed_unit_ids=allowed_unit_ids,
        allowed_manager_states=allowed_manager_states,
        require_target_ready=require_target_ready,
        strict_all_gpu_consumers=strict_all_gpu_consumers,
        activation_guard_reader=activation_guard_reader,
    )
    if policy_path is not None:
        sealed_policy_path = Path(policy_path).absolute()
        policy_value, policy_evidence = load_sealed_receipt_with_evidence(
            sealed_policy_path,
            fingerprint_field="policy_fingerprint",
        )
        policy = validate_environment_policy(
            policy_value
        )
        precleanup_root = dict(root_evidence["precleanup_inventory_receipt"])
        toolchain = current_runtime_toolchain_binding()
        if not _deep_exact_equal(
            policy.get("precleanup_root"),
            precleanup_root,
        ):
            raise PermissionError("sealed policy precleanup root binding changed")
        cleanup_root = dict(root_evidence["cleanup_receipt"])
        if not _deep_exact_equal(
            policy.get("cleanup_root"),
            cleanup_root,
        ):
            raise PermissionError("sealed policy cleanup root binding changed")
        if not _deep_exact_equal(policy.get("toolchain"), toolchain):
            raise PermissionError("sealed policy runtime toolchain binding changed")
        sampling = dict(policy["sampling"])
        if (
            sample_count < sampling["minimum_sample_count"]
            or float(sample_interval_seconds) < sampling["sample_interval_seconds"]
            or (sample_count - 1) * float(sample_interval_seconds)
            < sampling["required_stability_window_usec"] / 1_000_000.0
        ):
            raise PermissionError("runtime sampling is weaker than sealed policy")
        expected_policy = build_environment_policy(
            contract,
            precleanup_root_binding=precleanup_root,
            cleanup_root_binding=cleanup_root,
            toolchain_binding=toolchain,
            minimum_sample_count=sampling["minimum_sample_count"],
            sample_interval_seconds=sampling["sample_interval_seconds"],
        )
        expected_body = dict(expected_policy)
        expected_body.pop("policy_fingerprint")
        policy_body = dict(policy)
        policy_body.pop("policy_fingerprint")
        expected_body["created_at_utc"] = policy_body["created_at_utc"]
        if not _deep_exact_equal(policy_body, expected_body):
            raise PermissionError("sealed policy does not match root contract")
        root_evidence["policy"] = policy_evidence
    samples: list[dict[str, object]] = []
    sample_monotonic_seconds: list[float] = []
    for index in range(sample_count):
        samples.append(
            audit_environment_once(
                contract,
                inventory_collector=inventory_collector,
                activation_guard_reader=activation_guard_reader,
            )
        )
        sample_monotonic_seconds.append(float(monotonic_clock()))
        if index + 1 < sample_count and sample_interval_seconds > 0.0:
            sleeper(float(sample_interval_seconds))
    pre_verify = dict(root_evidence["precleanup_inventory_receipt"])
    pre_verify.pop("inventory_fingerprint")
    verify_sealed_receipt_evidence(
        precleanup_inventory_receipt_path,
        pre_verify,
    )
    verify_sealed_receipt_evidence(
        cleanup_receipt_path,
        root_evidence["cleanup_receipt"],
        fingerprint_field="cleanup_receipt_fingerprint",
    )
    verify_sealed_receipt_evidence(
        policy_path,
        root_evidence["policy"],
        fingerprint_field="policy_fingerprint",
    )
    if not _deep_exact_equal(
        current_runtime_toolchain_binding(),
        toolchain,
    ):
        raise PermissionError("runtime toolchain changed during stability gate")
    if contract.cleanup_mode == RECOVERY_CLEANUP_MODE:
        final_guard = activation_guard_reader(contract.activation_guard)
        if not _deep_exact_equal(
            final_guard,
            {
                **contract.activation_guard,
                "file_type": "symlink",
            },
        ):
            raise PermissionError(
                "recovery activation guard changed after stability samples"
            )
        cleanup_payload = load_sealed_receipt(
            cleanup_receipt_path,
            fingerprint_field="cleanup_receipt_fingerprint",
        )
        partial_lineage = cleanup_payload.get("partial_lineage")
        if not isinstance(partial_lineage, Mapping):
            raise PermissionError(
                "recovery cleanup partial lineage disappeared"
            )
        verify_partial_lineage_roots(partial_lineage)
    result = evaluate_environment_stability(
        contract,
        root_evidence,
        samples,
        sample_interval_seconds=float(sample_interval_seconds),
        sample_monotonic_seconds=sample_monotonic_seconds,
    )
    return validate_environment_stability_receipt(result)

@dataclass
class GPULeaseHandle:
    path: Path
    descriptor: int
    parent_descriptor: int
    payload: dict[str, object]
    device: int
    inode: int
    parent_device: int
    parent_inode: int
    closed: bool = False

    def close_without_release(self) -> None:
        """Drop this process's lock without unlinking any evidence."""

        if self.closed:
            return
        # Invalidate the object before touching either fd.  This makes a
        # repeated call a no-op even when unlock/close itself reports an error.
        lease_descriptor = self.descriptor
        parent_descriptor = self.parent_descriptor
        self.descriptor = -1
        self.parent_descriptor = -1
        self.closed = True
        first_error: BaseException | None = None
        try:
            fcntl.flock(lease_descriptor, fcntl.LOCK_UN)
        except BaseException as error:
            first_error = error
        try:
            os.close(lease_descriptor)
        except BaseException as error:
            if first_error is None:
                first_error = error
        if parent_descriptor != lease_descriptor:
            try:
                os.close(parent_descriptor)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _validate_lease_body(body: Mapping[str, object]) -> dict[str, object]:
    # The planned commit fingerprint binds deterministic future content. The
    # commit artifact must not exist yet; lease acquisition intentionally does
    # no filesystem lookup for it.
    expected = {
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
    value = dict(body)
    if set(value) != expected:
        raise ValueError("GPU lease body does not have the closed schema")
    if (
        value["schema_version"] != GPU_LEASE_SCHEMA
        or not isinstance(value["created_at_utc"], str)
        or _BOOT_ID_RE.fullmatch(str(value["boot_id"])) is None
        or not _is_gpu_uuid(value["gpu_uuid"])
        or not _is_sha256(value["runtime_spec_fingerprint"])
        or not isinstance(value["attempt_id"], str)
        or not value["attempt_id"]
        or not _is_sha256(value["authorization_fingerprint"])
        or not _is_sha256(value["planned_attempt_commit_fingerprint"])
    ):
        raise ValueError("GPU lease identity is malformed")
    for field in ("committer_pid", "committer_starttime"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"GPU lease {field} must be positive")
    return value


def acquire_gpu_lease(
    path: str | Path,
    body: Mapping[str, object],
) -> GPULeaseHandle:
    """Create and exclusively lock one private transient GPU lease."""

    target = _new_absolute_path(path, name="GPU lease path")
    materialized = _validate_lease_body(body)
    payload = {
        **materialized,
        "lease_fingerprint": stable_fingerprint(materialized),
    }
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    selected_uid = os.getuid()
    parent_descriptor, parent_metadata = _open_stable_parent_directory(
        target,
        name="GPU lease path parent",
        owner_uid=selected_uid,
        private=True,
    )
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write_all_to_fd(descriptor, encoded)
        os.fsync(descriptor)
        before_fd = os.fstat(descriptor)
        observed = _read_all_from_fd(descriptor)
        after_fd = os.fstat(descriptor)
        linked = _linked_entry_metadata(parent_descriptor, target.name)
        if (
            not _same_stable_file_metadata(before_fd, after_fd)
            or not _same_stable_file_metadata(after_fd, linked)
            or not stat.S_ISREG(after_fd.st_mode)
            or after_fd.st_uid != os.getuid()
            or stat.S_IMODE(after_fd.st_mode) != 0o600
            or after_fd.st_nlink != 1
            or observed != encoded
        ):
            raise RuntimeError("GPU lease failed self-verification")
        os.fsync(parent_descriptor)
        _verify_parent_path_generation(
            target.parent,
            parent_descriptor,
            expected_device=parent_metadata.st_dev,
            expected_inode=parent_metadata.st_ino,
            name="GPU lease path parent",
            owner_uid=selected_uid,
            private=True,
        )
        return GPULeaseHandle(
            path=target,
            descriptor=descriptor,
            parent_descriptor=parent_descriptor,
            payload=payload,
            device=after_fd.st_dev,
            inode=after_fd.st_ino,
            parent_device=parent_metadata.st_dev,
            parent_inode=parent_metadata.st_ino,
        )
    except BaseException:
        # Never unlink a partially materialized lease automatically.
        try:
            if descriptor >= 0:
                os.fsync(descriptor)
        except BaseException:
            pass
        try:
            os.fsync(parent_descriptor)
        except BaseException:
            pass
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except BaseException:
                pass
        _close_descriptors_best_effort(
            descriptor,
            parent_descriptor,
        )
        raise


def _rename_noreplace(
    source_parent_descriptor: int,
    source_basename: str,
    destination_parent_descriptor: int,
    destination_basename: str,
) -> None:
    """Linux ``renameat2(RENAME_NOREPLACE)`` with no unsafe fallback."""

    import ctypes

    for name, value in (
        ("source", source_basename),
        ("destination", destination_basename),
    ):
        if (
            not isinstance(value, str)
            or value in {"", ".", ".."}
            or Path(value).name != value
        ):
            raise ValueError(f"rename {name} basename is invalid")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    rename_noreplace = 1
    result = renameat2(
        source_parent_descriptor,
        os.fsencode(source_basename),
        destination_parent_descriptor,
        os.fsencode(destination_basename),
        rename_noreplace,
    )
    if result != 0:
        value = ctypes.get_errno()
        if value == errno.EEXIST:
            raise FileExistsError(
                value,
                os.strerror(value),
                destination_basename,
            )
        raise OSError(value, os.strerror(value), destination_basename)


def _directory_entry_absent(
    parent_descriptor: int,
    basename: str,
) -> bool:
    try:
        _linked_entry_metadata(parent_descriptor, basename)
    except FileNotFoundError:
        return True
    return False


_GPU_LEASE_RELEASE_BODY_KEYS = {
    "schema_version",
    "released_at_utc",
    "release_kind",
    "attempt_consumed",
    "lease_fingerprint",
    "gpu_uuid",
    "attempt_id",
    "evidence_fingerprint",
    "tombstone_path",
    "tombstone_file_sha256",
    "tombstone_device",
    "tombstone_inode",
    "active_lease_path",
    "active_lease_absent",
    "lease_parent_device",
    "lease_parent_inode",
}


def _validate_gpu_lease_release_body(
    value: Mapping[str, object],
) -> dict[str, object]:
    body = dict(value)
    release_kind = body.get("release_kind")
    tombstone_path = body.get("tombstone_path")
    active_path = body.get("active_lease_path")
    if (
        set(body) != _GPU_LEASE_RELEASE_BODY_KEYS
        or body.get("schema_version") != GPU_LEASE_RELEASE_SCHEMA
        or not isinstance(body.get("released_at_utc"), str)
        or not body["released_at_utc"]
        or release_kind not in {
            "committed_terminal",
            "uncommitted_forensic",
        }
        or body.get("attempt_consumed")
        is not (release_kind == "committed_terminal")
        or not _is_sha256(body.get("lease_fingerprint"))
        or not _is_gpu_uuid(body.get("gpu_uuid"))
        or not isinstance(body.get("attempt_id"), str)
        or not body["attempt_id"]
        or not _is_sha256(body.get("evidence_fingerprint"))
        or not isinstance(tombstone_path, str)
        or not Path(tombstone_path).is_absolute()
        or not isinstance(active_path, str)
        or not Path(active_path).is_absolute()
        or Path(tombstone_path).parent != Path(active_path).parent
        or not _is_sha256(body.get("tombstone_file_sha256"))
        or body.get("active_lease_absent") is not True
        or any(
            isinstance(body.get(field), bool)
            or not isinstance(body.get(field), int)
            or body[field] <= 0
            for field in (
                "tombstone_device",
                "tombstone_inode",
                "lease_parent_device",
                "lease_parent_inode",
            )
        )
    ):
        raise ValueError("GPU lease release receipt body is malformed")
    return body


def validate_gpu_lease_release_receipt(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact immutable GPU lease release receipt schema."""

    payload = dict(value)
    fingerprint = payload.get("receipt_fingerprint")
    body = dict(payload)
    body.pop("receipt_fingerprint", None)
    _validate_gpu_lease_release_body(body)
    if (
        set(payload) != _GPU_LEASE_RELEASE_BODY_KEYS
        | {"receipt_fingerprint"}
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
    ):
        raise ValueError("GPU lease release receipt fingerprint is malformed")
    return payload


def _verify_open_gpu_lease_tombstone(
    handle: GPULeaseHandle,
    *,
    active: Path,
    tombstone: Path,
    encoded: bytes,
    owner_uid: int,
) -> tuple[os.stat_result, bytes]:
    """Verify the tombstone closure through the still-locked lease fds."""

    _verify_parent_path_generation(
        active.parent,
        handle.parent_descriptor,
        expected_device=handle.parent_device,
        expected_inode=handle.parent_inode,
        name="GPU lease path parent",
        owner_uid=owner_uid,
        private=True,
    )
    before_linked = _linked_entry_metadata(
        handle.parent_descriptor,
        tombstone.name,
    )
    before_fd = os.fstat(handle.descriptor)
    observed = _read_all_from_fd(handle.descriptor)
    after_fd = os.fstat(handle.descriptor)
    after_linked = _linked_entry_metadata(
        handle.parent_descriptor,
        tombstone.name,
    )
    if (
        not _same_stable_file_metadata(before_linked, before_fd)
        or not _same_stable_file_metadata(before_fd, after_fd)
        or not _same_stable_file_metadata(after_fd, after_linked)
        or not stat.S_ISREG(after_fd.st_mode)
        or after_fd.st_uid != owner_uid
        or after_fd.st_nlink != 1
        or stat.S_IMODE(after_fd.st_mode) != 0o444
        or (after_fd.st_dev, after_fd.st_ino)
        != (handle.device, handle.inode)
        or observed != encoded
        or not _directory_entry_absent(
            handle.parent_descriptor,
            active.name,
        )
    ):
        raise RuntimeError("GPU lease tombstone failed self-verification")
    _verify_parent_path_generation(
        active.parent,
        handle.parent_descriptor,
        expected_device=handle.parent_device,
        expected_inode=handle.parent_inode,
        name="GPU lease path parent",
        owner_uid=owner_uid,
        private=True,
    )
    return after_fd, observed


def _verify_gpu_lease_tombstone_after_close(
    *,
    active: Path,
    tombstone: Path,
    release_receipt: Mapping[str, object],
    owner_uid: int,
) -> None:
    """Freshly bind and verify the named release closure after lease fd close."""

    receipt = validate_gpu_lease_release_receipt(release_receipt)
    if (
        receipt["active_lease_path"] != str(active)
        or receipt["tombstone_path"] != str(tombstone)
    ):
        raise PermissionError(
            "GPU lease post-close receipt paths changed"
        )
    parent_descriptor = -1
    tombstone_descriptor = -1
    try:
        parent_descriptor, parent_metadata = (
            _open_stable_parent_directory(
                tombstone,
                name="GPU lease post-close parent",
                owner_uid=owner_uid,
                private=True,
            )
        )
        if (
            parent_metadata.st_dev != receipt["lease_parent_device"]
            or parent_metadata.st_ino != receipt["lease_parent_inode"]
            or not _directory_entry_absent(
                parent_descriptor,
                active.name,
            )
        ):
            raise PermissionError(
                "GPU lease post-close parent closure changed"
            )
        before_linked = _linked_entry_metadata(
            parent_descriptor,
            tombstone.name,
        )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        tombstone_descriptor = os.open(
            tombstone.name,
            flags,
            dir_fd=parent_descriptor,
        )
        before_fd = os.fstat(tombstone_descriptor)
        observed = _read_all_from_fd(tombstone_descriptor)
        after_fd = os.fstat(tombstone_descriptor)
        after_linked = _linked_entry_metadata(
            parent_descriptor,
            tombstone.name,
        )
        if (
            not _same_stable_file_metadata(before_linked, before_fd)
            or not _same_stable_file_metadata(before_fd, after_fd)
            or not _same_stable_file_metadata(after_fd, after_linked)
            or not stat.S_ISREG(after_fd.st_mode)
            or after_fd.st_uid != owner_uid
            or after_fd.st_nlink != 1
            or stat.S_IMODE(after_fd.st_mode) != 0o444
            or after_fd.st_dev != receipt["tombstone_device"]
            or after_fd.st_ino != receipt["tombstone_inode"]
            or hashlib.sha256(observed).hexdigest()
            != receipt["tombstone_file_sha256"]
            or not _directory_entry_absent(
                parent_descriptor,
                active.name,
            )
        ):
            raise PermissionError(
                "GPU lease post-close tombstone closure changed"
            )
        _verify_parent_path_generation(
            tombstone.parent,
            parent_descriptor,
            expected_device=receipt["lease_parent_device"],
            expected_inode=receipt["lease_parent_inode"],
            name="GPU lease post-close parent",
            owner_uid=owner_uid,
            private=True,
        )
    except BaseException:
        _close_descriptors_best_effort(
            tombstone_descriptor,
            parent_descriptor,
        )
        raise
    close_error = _close_descriptors_best_effort(
        tombstone_descriptor,
        parent_descriptor,
    )
    if close_error is not None:
        raise close_error


def release_gpu_lease_to_tombstone(
    handle: GPULeaseHandle,
    *,
    tombstone_path: str | Path,
    release_receipt_path: str | Path,
    release_kind: str,
    attempt_consumed: bool,
    evidence_fingerprint: str,
) -> dict[str, object]:
    """Atomically retire a lease to an immutable tombstone, then receipt it."""

    if handle.closed:
        raise RuntimeError("GPU lease handle is already closed")
    for field in (
        "device",
        "inode",
        "parent_device",
        "parent_inode",
    ):
        item = getattr(handle, field)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"GPU lease handle {field} is malformed")
    for field in ("descriptor", "parent_descriptor"):
        item = getattr(handle, field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"GPU lease handle {field} is malformed")
    if handle.descriptor == handle.parent_descriptor:
        raise ValueError("GPU lease descriptors must be distinct")
    if release_kind not in {
        "committed_terminal",
        "uncommitted_forensic",
    }:
        raise ValueError("GPU lease release_kind is invalid")
    if attempt_consumed is not (release_kind == "committed_terminal"):
        raise ValueError("GPU lease consumed/release semantics disagree")
    if not _is_sha256(evidence_fingerprint):
        raise ValueError("release evidence fingerprint is malformed")
    active = _new_absolute_path(
        handle.path,
        name="GPU lease active path",
    )
    tombstone = _new_absolute_path(
        tombstone_path,
        name="GPU lease tombstone",
    )
    if tombstone.parent != active.parent:
        raise ValueError("GPU lease tombstone must share the lease directory")
    receipt = _canonical_new_path(
        release_receipt_path,
        name="GPU lease release receipt",
    )
    if receipt.parent == active.parent:
        raise ValueError(
            "GPU lease release receipt must use a different directory"
        )

    selected_uid = os.getuid()
    _verify_parent_path_generation(
        active.parent,
        handle.parent_descriptor,
        expected_device=handle.parent_device,
        expected_inode=handle.parent_inode,
        name="GPU lease path parent",
        owner_uid=selected_uid,
        private=True,
    )

    encoded = (
        canonical_json(handle.payload) + "\n"
    ).encode("utf-8")
    active_before = _linked_entry_metadata(
        handle.parent_descriptor,
        active.name,
    )
    descriptor_before = os.fstat(handle.descriptor)
    observed = _read_all_from_fd(handle.descriptor)
    descriptor_after = os.fstat(handle.descriptor)
    active_after = _linked_entry_metadata(
        handle.parent_descriptor,
        active.name,
    )
    if (
        not _same_stable_file_metadata(
            active_before,
            descriptor_before,
        )
        or not _same_stable_file_metadata(
            descriptor_before,
            descriptor_after,
        )
        or not _same_stable_file_metadata(
            descriptor_after,
            active_after,
        )
        or not stat.S_ISREG(descriptor_after.st_mode)
        or descriptor_after.st_uid != selected_uid
        or descriptor_after.st_nlink != 1
        or stat.S_IMODE(descriptor_after.st_mode) != 0o600
        or (descriptor_after.st_dev, descriptor_after.st_ino)
        != (handle.device, handle.inode)
        or observed != encoded
    ):
        raise RuntimeError("GPU lease identity changed before release")

    _rename_noreplace(
        handle.parent_descriptor,
        active.name,
        handle.parent_descriptor,
        tombstone.name,
    )
    os.fchmod(handle.descriptor, 0o444)
    os.fsync(handle.descriptor)
    os.fsync(handle.parent_descriptor)
    tombstone_metadata, tombstone_observed = (
        _verify_open_gpu_lease_tombstone(
            handle,
            active=active,
            tombstone=tombstone,
            encoded=encoded,
            owner_uid=selected_uid,
        )
    )
    active_absent = True

    release_body: dict[str, object] = {
        "schema_version": GPU_LEASE_RELEASE_SCHEMA,
        "released_at_utc": utc_now(),
        "release_kind": release_kind,
        "attempt_consumed": attempt_consumed,
        "lease_fingerprint": handle.payload["lease_fingerprint"],
        "gpu_uuid": handle.payload["gpu_uuid"],
        "attempt_id": handle.payload["attempt_id"],
        "evidence_fingerprint": evidence_fingerprint,
        "tombstone_path": str(tombstone),
        "tombstone_file_sha256": hashlib.sha256(
            tombstone_observed
        ).hexdigest(),
        "tombstone_device": tombstone_metadata.st_dev,
        "tombstone_inode": tombstone_metadata.st_ino,
        "active_lease_path": str(active),
        "active_lease_absent": active_absent,
        "lease_parent_device": handle.parent_device,
        "lease_parent_inode": handle.parent_inode,
    }
    _validate_gpu_lease_release_body(release_body)

    def cross_verify_while_receipt_open(
        receipt_descriptor: int,
        receipt_parent_descriptor: int,
        receipt_metadata: os.stat_result,
        receipt_parent_metadata: os.stat_result,
    ) -> None:
        guarded_receipt_parent = os.fstat(
            receipt_parent_descriptor
        )
        if (
            receipt_descriptor == handle.descriptor
            or receipt_parent_descriptor == handle.parent_descriptor
            or (
                receipt_parent_metadata.st_dev,
                receipt_parent_metadata.st_ino,
            )
            == (handle.parent_device, handle.parent_inode)
            or not _same_stable_file_metadata(
                receipt_metadata,
                os.fstat(receipt_descriptor),
            )
            or (
                guarded_receipt_parent.st_dev,
                guarded_receipt_parent.st_ino,
            )
            != (
                receipt_parent_metadata.st_dev,
                receipt_parent_metadata.st_ino,
            )
        ):
            raise PermissionError(
                "GPU lease and release receipt generations overlap incorrectly"
            )
        guarded_metadata, guarded_observed = (
            _verify_open_gpu_lease_tombstone(
                handle,
                active=active,
                tombstone=tombstone,
                encoded=encoded,
                owner_uid=selected_uid,
            )
        )
        if (
            not _same_stable_file_metadata(
                tombstone_metadata,
                guarded_metadata,
            )
            or guarded_observed != tombstone_observed
            or guarded_metadata.st_dev != release_body["tombstone_device"]
            or guarded_metadata.st_ino != release_body["tombstone_inode"]
            or hashlib.sha256(guarded_observed).hexdigest()
            != release_body["tombstone_file_sha256"]
        ):
            raise PermissionError(
                "GPU lease closure changed during receipt sealing"
            )

    released = write_create_once_receipt(
        receipt,
        release_body,
        while_open_guard=cross_verify_while_receipt_open,
    )
    released = validate_gpu_lease_release_receipt(released)

    # The receipt fds have closed only after the cross-generation guard.  The
    # lease fds remain open for one last check before deliberately closing.
    final_tombstone_metadata, final_observed = (
        _verify_open_gpu_lease_tombstone(
            handle,
            active=active,
            tombstone=tombstone,
            encoded=encoded,
            owner_uid=selected_uid,
        )
    )
    if (
        not _same_stable_file_metadata(
            tombstone_metadata,
            final_tombstone_metadata,
        )
        or final_observed != tombstone_observed
        or not _directory_entry_absent(
            handle.parent_descriptor,
            active.name,
        )
    ):
        raise RuntimeError(
            "GPU lease closure changed after release receipt"
        )
    handle.close_without_release()
    _verify_gpu_lease_tombstone_after_close(
        active=active,
        tombstone=tombstone,
        release_receipt=released,
        owner_uid=selected_uid,
    )
    return released


def _verify_policy_bound_audit_closure(
    sampled_result: Mapping[str, object],
    policy: Mapping[str, object],
    *,
    policy_evidence: Mapping[str, object],
    stability_evidence: Mapping[str, object],
    policy_path: str | Path,
    stability_receipt_path: str | Path,
    precleanup_inventory_receipt_path: str | Path,
    cleanup_receipt_path: str | Path,
) -> None:
    """Recheck the exact policy/root/toolchain generation around receipt sealing."""

    roots = dict(sampled_result["root_evidence"])
    precleanup_evidence = dict(roots["precleanup_inventory_receipt"])
    cleanup_evidence = dict(roots["cleanup_receipt"])
    root_policy_evidence = dict(roots["policy"])
    expected_paths = {
        "precleanup": str(
            Path(precleanup_inventory_receipt_path).absolute()
        ),
        "cleanup": str(Path(cleanup_receipt_path).absolute()),
        "policy": str(Path(policy_path).absolute()),
        "stability": str(Path(stability_receipt_path).absolute()),
    }
    if (
        precleanup_evidence.get("path") != expected_paths["precleanup"]
        or cleanup_evidence.get("path") != expected_paths["cleanup"]
        or root_policy_evidence.get("path") != expected_paths["policy"]
        or policy_evidence.get("path") != expected_paths["policy"]
        or stability_evidence.get("path") != expected_paths["stability"]
        or not _deep_exact_equal(
            policy.get("precleanup_root"),
            precleanup_evidence,
        )
        or not _deep_exact_equal(
            policy.get("cleanup_root"),
            cleanup_evidence,
        )
        or not _deep_exact_equal(
            root_policy_evidence,
            dict(policy_evidence),
        )
        or not _deep_exact_equal(
            policy.get("policy_fingerprint"),
            root_policy_evidence.get("policy_fingerprint"),
        )
        or not _deep_exact_equal(
            sampled_result.get("stability_receipt_fingerprint"),
            stability_evidence.get("stability_receipt_fingerprint"),
        )
        or not _deep_exact_equal(
            policy.get("toolchain"),
            current_runtime_toolchain_binding(),
        )
    ):
        raise PermissionError(
            "policy-bound audit closure changed before receipt sealing"
        )
    precleanup_verify = dict(precleanup_evidence)
    precleanup_verify.pop("inventory_fingerprint")
    verify_sealed_receipt_evidence(
        precleanup_inventory_receipt_path,
        precleanup_verify,
    )
    verify_sealed_receipt_evidence(
        cleanup_receipt_path,
        cleanup_evidence,
        fingerprint_field="cleanup_receipt_fingerprint",
    )
    verify_sealed_receipt_evidence(
        policy_path,
        policy_evidence,
        fingerprint_field="policy_fingerprint",
    )
    verify_sealed_receipt_evidence(
        stability_receipt_path,
        stability_evidence,
        fingerprint_field="stability_receipt_fingerprint",
    )
    if not _deep_exact_equal(
        policy.get("toolchain"),
        current_runtime_toolchain_binding(),
    ):
        raise PermissionError(
            "policy-bound audit toolchain changed during closure recheck"
        )


def _contract_from_sampled_environment_receipt(
    sampled_result: Mapping[str, object],
) -> EnvironmentAuditContract:
    try:
        contract_value = dict(sampled_result["contract"])
        for field in (
            "conflict_unit_ids",
            "dependency_unit_ids",
            "allowed_failed_unit_ids",
            "expected_failed_unit_ids",
            "allowed_unit_ids",
            "allowed_manager_states",
        ):
            contract_value[field] = tuple(contract_value[field])
        contract_value["cleanup_nrestarts_baseline"] = tuple(
            (str(item[0]), str(item[1]))
            for item in contract_value["cleanup_nrestarts_baseline"]
        )
        contract_value["activation_guard"] = dict(
            contract_value["activation_guard"]
        )
        return validate_environment_audit_contract(
            EnvironmentAuditContract(**contract_value)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "policy-bound sampled environment contract is malformed"
        ) from error


def _validate_policy_bound_requested_scope(
    args: argparse.Namespace,
    policy: Mapping[str, object],
) -> None:
    scope = dict(policy["unit_scope"])
    selected_gpu = dict(policy["selected_gpu"])
    requested = {
        "target_unit_id": args.target_unit,
        "conflict_unit_ids": list(args.conflict_unit),
        "dependency_unit_ids": list(args.dependency_unit),
        "allowed_failed_unit_ids": list(args.allow_failed_unit),
        "allowed_unit_ids": list(args.allow_unit),
        "allowed_manager_states": list(
            args.allow_manager_state or ("running", "degraded")
        ),
        "selected_gpu_index": args.selected_gpu_index,
        "require_target_ready": args.require_target_ready,
        "strict_all_gpu_consumers": args.strict_all_gpu_consumers,
    }
    expected = {
        "target_unit_id": scope["target_unit_id"],
        "conflict_unit_ids": scope["conflict_unit_ids"],
        "dependency_unit_ids": scope["dependency_unit_ids"],
        "allowed_failed_unit_ids": scope["allowed_failed_unit_ids"],
        "allowed_unit_ids": scope["allowed_unit_ids"],
        "allowed_manager_states": policy["allowed_manager_states"],
        "selected_gpu_index": selected_gpu["index"],
        "require_target_ready": scope["require_target_ready"],
        "strict_all_gpu_consumers": policy["strict_all_gpu_consumers"],
    }
    if not _deep_exact_equal(requested, expected):
        raise PermissionError(
            "policy-bound audit requested scope differs from sealed policy"
        )


def _audit_only(args: argparse.Namespace) -> int:
    output = Path(args.output).absolute()
    body: dict[str, object]
    receipt_while_open_guard: Callable[
        [int, int, os.stat_result, os.stat_result],
        None,
    ] | None = None
    try:
        policy_bound_roots = (
            args.policy,
            args.precleanup_inventory_receipt,
            args.cleanup_receipt,
            args.stability_receipt,
        )
        if any(value is not None for value in policy_bound_roots):
            if not all(value is not None for value in policy_bound_roots):
                raise ValueError(
                    "policy-bound audit requires policy, precleanup inventory "
                    "receipt, cleanup receipt, and stability receipt together"
                )
            policy_value, policy_evidence = (
                load_sealed_receipt_with_evidence(
                    args.policy,
                    fingerprint_field="policy_fingerprint",
                )
            )
            policy = validate_environment_policy(policy_value)
            _validate_policy_bound_requested_scope(args, policy)
            stability_value, stability_evidence = (
                load_sealed_receipt_with_evidence(
                    args.stability_receipt,
                    fingerprint_field="stability_receipt_fingerprint",
                )
            )
            sampled_result = validate_environment_stability_receipt(
                stability_value
            )
            sampling = dict(policy["sampling"])
            samples = sampled_result.get("samples")
            if (
                sampled_result.get("receipt_kind") != "sampled"
                or sampled_result.get("passed") is not True
                or sampled_result.get("blockers") != []
                or not isinstance(samples, list)
                or isinstance(sampled_result.get("sample_count"), bool)
                or not isinstance(sampled_result.get("sample_count"), int)
                or sampled_result["sample_count"] != len(samples)
                or sampled_result["sample_count"]
                < sampling["minimum_sample_count"]
                or sampled_result["sample_interval_seconds"]
                < sampling["sample_interval_seconds"]
                or (
                    sampled_result["sample_count"] - 1
                ) * sampled_result["sample_interval_seconds"]
                < sampling["required_stability_window_usec"] / 1_000_000.0
            ):
                raise PermissionError(
                    "policy-bound postcleanup stability audit did not pass"
                )
            contract = _contract_from_sampled_environment_receipt(
                sampled_result
            )
            roots = dict(sampled_result["root_evidence"])
            toolchain = current_runtime_toolchain_binding()
            expected_policy = build_environment_policy(
                contract,
                precleanup_root_binding=roots[
                    "precleanup_inventory_receipt"
                ],
                cleanup_root_binding=roots["cleanup_receipt"],
                toolchain_binding=toolchain,
                minimum_sample_count=sampling["minimum_sample_count"],
                sample_interval_seconds=sampling["sample_interval_seconds"],
            )
            expected_policy_body = dict(expected_policy)
            expected_policy_body.pop("policy_fingerprint")
            policy_body = dict(policy)
            policy_body.pop("policy_fingerprint")
            expected_policy_body["created_at_utc"] = policy_body[
                "created_at_utc"
            ]
            if (
                not _deep_exact_equal(
                    policy_body,
                    expected_policy_body,
                )
                or not _deep_exact_equal(
                    roots.get("policy"),
                    policy_evidence,
                )
            ):
                raise PermissionError(
                    "sealed stability receipt is not bound to policy scope"
                )
            last_sample_index = sampled_result["sample_count"] - 1
            last_sample = validate_environment_single_audit(
                samples[last_sample_index],
                contract=contract,
            )
            inventory = dict(last_sample["inventory"])
            inventory_body = dict(inventory)
            inventory_fingerprint = inventory_body.pop(
                "inventory_fingerprint",
                None,
            )
            if (
                inventory.get("schema_version")
                != ENVIRONMENT_INVENTORY_SCHEMA
                or not _is_sha256(inventory_fingerprint)
                or inventory_fingerprint
                != stable_fingerprint(inventory_body)
                or inventory.get("passed") is not True
                or inventory.get("blockers") != []
                or any(
                    inventory.get(field) is not False
                    for field in (
                        "D_R_payload_accessed",
                        "D_V_payload_accessed",
                        "D_T_payload_accessed",
                    )
                )
            ):
                raise PermissionError(
                    "policy-bound postcleanup inventory is invalid"
                )
            _verify_policy_bound_audit_closure(
                sampled_result,
                policy,
                policy_evidence=policy_evidence,
                stability_evidence=stability_evidence,
                policy_path=args.policy,
                stability_receipt_path=args.stability_receipt,
                precleanup_inventory_receipt_path=(
                    args.precleanup_inventory_receipt
                ),
                cleanup_receipt_path=args.cleanup_receipt,
            )

            def policy_bound_receipt_guard(
                _receipt_descriptor: int,
                _receipt_parent_descriptor: int,
                _receipt_metadata: os.stat_result,
                _receipt_parent_metadata: os.stat_result,
            ) -> None:
                _verify_policy_bound_audit_closure(
                    sampled_result,
                    policy,
                    policy_evidence=policy_evidence,
                    stability_evidence=stability_evidence,
                    policy_path=args.policy,
                    stability_receipt_path=args.stability_receipt,
                    precleanup_inventory_receipt_path=(
                        args.precleanup_inventory_receipt
                    ),
                    cleanup_receipt_path=args.cleanup_receipt,
                )

            receipt_while_open_guard = policy_bound_receipt_guard
        else:
            inventory = collect_environment_inventory(
                selected_gpu_index=args.selected_gpu_index,
                allowed_unit_ids=tuple(args.allow_unit),
                target_unit_id=args.target_unit,
                conflict_unit_ids=tuple(args.conflict_unit),
                dependency_unit_ids=tuple(args.dependency_unit),
                allowed_failed_unit_ids=tuple(args.allow_failed_unit),
                allowed_manager_states=tuple(
                    args.allow_manager_state or ("running", "degraded")
                ),
                require_target_ready=args.require_target_ready,
                strict_all_gpu_consumers=args.strict_all_gpu_consumers,
            )
        endpoint = inventory["manager"]["endpoint"]
        manager_identity = inventory["manager"]["identity"]
        environment_binding = {
            "inventory_fingerprint": inventory["inventory_fingerprint"],
            "boot_id": inventory["boot_id"],
            "runtime_directory": endpoint["runtime_directory"],
            "runtime_directory_device": endpoint["runtime_directory_device"],
            "runtime_directory_inode": endpoint["runtime_directory_inode"],
            "manager_identity": manager_identity,
        }
        body = {
            "schema_version": ENVIRONMENT_RECEIPT_SCHEMA,
            "created_at_utc": utc_now(),
            "command": "audit-only",
            "environment_binding": environment_binding,
            "inventory": inventory,
            "passed": inventory["passed"],
            "error_type": None,
            "error_message": None,
        }
        returncode = 0 if inventory["passed"] else 1
    except Exception as error:
        receipt_while_open_guard = None
        body = {
            "schema_version": ENVIRONMENT_RECEIPT_SCHEMA,
            "created_at_utc": utc_now(),
            "command": "audit-only",
            "environment_binding": None,
            "inventory": None,
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        returncode = 1
    body.update(
        {
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
    )
    write_create_once_receipt(
        output,
        body,
        while_open_guard=receipt_while_open_guard,
    )
    return returncode

def _create_policy(args: argparse.Namespace) -> int:
    contract, roots = prepare_environment_stability_contract(
        args.precleanup_inventory_receipt,
        args.cleanup_receipt,
        selected_gpu_index=args.selected_gpu_index,
        target_unit_id=args.target_unit,
        conflict_unit_ids=tuple(args.conflict_unit),
        dependency_unit_ids=tuple(args.dependency_unit),
        allowed_failed_unit_ids=tuple(args.allow_failed_unit),
        allowed_unit_ids=tuple(args.allow_unit),
        allowed_manager_states=tuple(
            args.allow_manager_state or ("running", "degraded")
        ),
        require_target_ready=args.require_target_ready,
        strict_all_gpu_consumers=False,
    )
    policy = build_environment_policy(
        contract,
        precleanup_root_binding=roots["precleanup_inventory_receipt"],
        cleanup_root_binding=roots["cleanup_receipt"],
        toolchain_binding=current_runtime_toolchain_binding(),
        minimum_sample_count=args.sample_count,
        sample_interval_seconds=args.interval_seconds,
    )
    write_environment_policy(args.output, policy)
    return 0



def _stability_gate(args: argparse.Namespace) -> int:
    output = Path(args.output).absolute()
    try:
        result = run_environment_stability_gate(
            args.precleanup_inventory_receipt,
            args.cleanup_receipt,
            selected_gpu_index=args.selected_gpu_index,
            target_unit_id=args.target_unit,
            conflict_unit_ids=tuple(args.conflict_unit),
            dependency_unit_ids=tuple(args.dependency_unit),
            allowed_failed_unit_ids=tuple(args.allow_failed_unit),
            sample_count=args.sample_count,
            sample_interval_seconds=args.interval_seconds,
            allowed_unit_ids=tuple(args.allow_unit),
            allowed_manager_states=tuple(
                args.allow_manager_state or ("running", "degraded")
            ),
            require_target_ready=args.require_target_ready,
            strict_all_gpu_consumers=args.strict_all_gpu_consumers,
            policy_path=args.policy,
        )
        returncode = 0 if result["passed"] else 1
    except Exception as error:
        result = {
            "schema_version": ENVIRONMENT_STABILITY_RECEIPT_SCHEMA,
            "receipt_kind": "exception",
            "created_at_utc": utc_now(),
            "command": "stability-gate",
            "requested_roots": {
                "precleanup_inventory_receipt_path": str(
                    Path(args.precleanup_inventory_receipt).absolute()
                ),
                "cleanup_receipt_path": str(
                    Path(args.cleanup_receipt).absolute()
                ),
                "policy_path": (
                    None if args.policy is None
                    else str(Path(args.policy).absolute())
                ),
            },
            "requested_scope": {
                "target_unit_id": args.target_unit,
                "conflict_unit_ids": list(args.conflict_unit),
                "dependency_unit_ids": list(args.dependency_unit),
                "allowed_failed_unit_ids": list(args.allow_failed_unit),
                "allowed_unit_ids": list(args.allow_unit),
                "selected_gpu_index": args.selected_gpu_index,
            },
            "sample_count": 0,
            "sample_interval_seconds": (
                float(args.interval_seconds)
                if math.isfinite(float(args.interval_seconds))
                else repr(args.interval_seconds)
            ),
            "samples": [],
            "blockers": ["stability_gate_exception"],
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "payload_authority": "none",
            "D_T_payload_accessed": False,
        }
        returncode = 1
    body = dict(result)
    body.pop("stability_receipt_fingerprint", None)
    written = write_create_once_receipt(
        output,
        body,
        fingerprint_field="stability_receipt_fingerprint",
    )
    validate_environment_stability_receipt(written)
    return returncode



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CURE-Lite v24 read-only runtime environment auditor",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit-only",
        help="read systemd/procfs/nvidia-smi metadata and write one receipt",
    )
    audit.add_argument("--output", required=True)
    audit.add_argument("--selected-gpu-index", required=True, type=int)
    audit.add_argument("--allow-unit", action="append", default=[])
    audit.add_argument("--target-unit")
    audit.add_argument("--conflict-unit", action="append", default=[])
    audit.add_argument("--dependency-unit", action="append", default=[])
    audit.add_argument("--allow-failed-unit", action="append", default=[])
    audit.add_argument(
        "--allow-manager-state",
        action="append",
        default=None,
    )
    audit.add_argument("--require-target-ready", action="store_true")
    audit.add_argument("--strict-all-gpu-consumers", action="store_true")
    audit.add_argument("--policy")
    audit.add_argument("--precleanup-inventory-receipt")
    audit.add_argument("--cleanup-receipt")
    audit.add_argument("--stability-receipt")
    audit.set_defaults(handler=_audit_only)
    create_policy = subparsers.add_parser(
        "create-policy",
        help="seal a closed runtime environment policy from precleanup evidence",
    )
    create_policy.add_argument("--output", required=True)
    create_policy.add_argument("--precleanup-inventory-receipt", required=True)
    create_policy.add_argument("--cleanup-receipt", required=True)
    create_policy.add_argument("--selected-gpu-index", required=True, type=int)
    create_policy.add_argument("--target-unit", required=True)
    create_policy.add_argument("--conflict-unit", action="append", default=[])
    create_policy.add_argument("--dependency-unit", action="append", default=[])
    create_policy.add_argument("--allow-failed-unit", action="append", default=[])
    create_policy.add_argument("--allow-unit", action="append", default=[])
    create_policy.add_argument(
        "--allow-manager-state",
        action="append",
        default=None,
    )
    create_policy.add_argument("--sample-count", type=int, default=2)
    create_policy.add_argument("--interval-seconds", type=float, default=30.0)
    create_policy.add_argument("--require-target-ready", action="store_true")
    create_policy.set_defaults(handler=_create_policy)

    stability = subparsers.add_parser(
        "stability-gate",
        help="sample a sealed post-cleanup environment without mutation",
    )
    stability.add_argument("--output", required=True)
    stability.add_argument("--precleanup-inventory-receipt", required=True)
    stability.add_argument("--cleanup-receipt", required=True)
    stability.add_argument("--policy", required=True)
    stability.add_argument("--selected-gpu-index", required=True, type=int)
    stability.add_argument("--target-unit", required=True)
    stability.add_argument("--conflict-unit", action="append", default=[])
    stability.add_argument("--dependency-unit", action="append", default=[])
    stability.add_argument("--allow-failed-unit", action="append", default=[])
    stability.add_argument("--allow-unit", action="append", default=[])
    stability.add_argument(
        "--allow-manager-state",
        action="append",
        default=None,
    )
    stability.add_argument("--sample-count", type=int, default=2)
    stability.add_argument("--interval-seconds", type=float, default=30.0)
    stability.add_argument("--require-target-ready", action="store_true")
    stability.add_argument("--strict-all-gpu-consumers", action="store_true")
    stability.set_defaults(handler=_stability_gate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
