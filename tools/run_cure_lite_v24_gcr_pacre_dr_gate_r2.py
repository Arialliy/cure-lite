#!/usr/bin/env python3
"""Fresh-r2 execution-identity adapter for the frozen v24 D_R gate.

The 103-file r1 numerical/scientific closure is byte-preserved.  This outer
entry point changes only attempt identity schemas and create-once artifact
paths in a fresh isolated Python process, then delegates to the already-frozen
gate CLI.  A future r2 runtime authorization must bind this adapter's path and
SHA in addition to the unchanged scientific closure.

This is intentionally not a retry switch.  It refuses to configure after any
real preaccess/run-start token has been issued in the process, and it refuses
to run if the legacy CLI was imported before the identity transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
ACTUAL_PYTHON_PATH = Path("/usr/bin/python3.12")
RUNTIME_DEPENDENCY_SITE_PATH = Path(
    "/home/md0/ly/MSHNet/.venv/lib/python3.12/site-packages"
)
EXPECTED_RUNTIME_DEPENDENCY_SITE_BINDING = {
    "path": str(RUNTIME_DEPENDENCY_SITE_PATH),
    "device": 2304,
    "inode": 228331323,
    "owner_uid": 1008,
    "mode": 0o775,
}
_DIRECT_RUNTIME_BOOTSTRAPPED = False
_DIRECT_RUNTIME_SITE_BINDING: dict[str, object] | None = None
_DIRECT_RUNTIME_SITE_AUTHORITY_BINDING: dict[str, object] | None = None
_DIRECT_RUNTIME_SITE_DESCRIPTOR: int | None = None


def _close_descriptors_best_effort(*descriptors: int) -> BaseException | None:
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


def _open_runtime_dependency_site_authority(
    site_path: Path,
    expected_binding: Mapping[str, object],
) -> tuple[int, dict[str, object], dict[str, object]]:
    """Open one exact site generation and expose only its retained procfd."""

    target = Path(site_path)
    if (
        not target.is_absolute()
        or target.name in {"", ".", ".."}
        or target.parent / target.name != target
    ):
        raise ValueError("runtime dependency site path must be absolute")
    expected = dict(expected_binding)
    if (
        set(expected)
        != {"path", "device", "inode", "owner_uid", "mode"}
        or expected.get("path") != str(target)
        or any(
            isinstance(expected.get(field), bool)
            or not isinstance(expected.get(field), int)
            or expected[field] < 0
            for field in ("device", "inode", "owner_uid", "mode")
        )
        or expected["inode"] <= 0
    ):
        raise ValueError("runtime dependency site binding is malformed")

    parent = target.parent
    parent_before = parent.lstat()
    site_before = target.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or not stat.S_ISDIR(site_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or stat.S_ISLNK(site_before.st_mode)
        or parent.resolve(strict=True) != parent
        or target.resolve(strict=True) != target
        or parent_before.st_uid != os.getuid()
        or site_before.st_uid != os.getuid()
        or stat.S_IMODE(parent_before.st_mode) & 0o002
        or stat.S_IMODE(site_before.st_mode) & 0o002
    ):
        raise PermissionError(
            "actual runtime dependency site identity is unsafe"
        )

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = -1
    site_descriptor = -1
    try:
        parent_descriptor = os.open(parent, directory_flags)
        parent_opened = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_opened.st_mode)
            or (
                parent_opened.st_dev,
                parent_opened.st_ino,
                parent_opened.st_uid,
                stat.S_IMODE(parent_opened.st_mode),
            )
            != (
                parent_before.st_dev,
                parent_before.st_ino,
                parent_before.st_uid,
                stat.S_IMODE(parent_before.st_mode),
            )
        ):
            raise PermissionError(
                "runtime dependency site parent changed while opening"
            )
        linked_before = os.stat(
            target.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        site_descriptor = os.open(
            target.name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(site_descriptor)
        linked_after = os.stat(
            target.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        observed = {
            "path": str(target),
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "owner_uid": opened.st_uid,
            "mode": stat.S_IMODE(opened.st_mode),
        }
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (
                linked_before.st_dev,
                linked_before.st_ino,
                linked_before.st_uid,
                stat.S_IMODE(linked_before.st_mode),
            )
            != (
                site_before.st_dev,
                site_before.st_ino,
                site_before.st_uid,
                stat.S_IMODE(site_before.st_mode),
            )
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_uid,
                stat.S_IMODE(opened.st_mode),
            )
            != (
                linked_after.st_dev,
                linked_after.st_ino,
                linked_after.st_uid,
                stat.S_IMODE(linked_after.st_mode),
            )
            or observed != expected
        ):
            raise PermissionError(
                "actual runtime dependency site generation changed"
            )

        authority_path = Path(f"/proc/self/fd/{site_descriptor}")
        authority_lstat = authority_path.lstat()
        authority_metadata = authority_path.stat()
        authority_binding = {
            "path": str(authority_path),
            "device": authority_metadata.st_dev,
            "inode": authority_metadata.st_ino,
            "owner_uid": authority_metadata.st_uid,
            "mode": stat.S_IMODE(authority_metadata.st_mode),
        }
        final_parent = parent.lstat()
        final_site = target.lstat()
        if (
            not stat.S_ISLNK(authority_lstat.st_mode)
            or not stat.S_ISDIR(authority_metadata.st_mode)
            or authority_binding
            != {
                **observed,
                "path": str(authority_path),
            }
            or (
                final_parent.st_dev,
                final_parent.st_ino,
                final_parent.st_uid,
                stat.S_IMODE(final_parent.st_mode),
            )
            != (
                parent_opened.st_dev,
                parent_opened.st_ino,
                parent_opened.st_uid,
                stat.S_IMODE(parent_opened.st_mode),
            )
            or (
                final_site.st_dev,
                final_site.st_ino,
                final_site.st_uid,
                stat.S_IMODE(final_site.st_mode),
            )
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_uid,
                stat.S_IMODE(opened.st_mode),
            )
        ):
            raise PermissionError(
                "runtime dependency site authority changed before use"
            )
    except BaseException:
        _close_descriptors_best_effort(
            site_descriptor,
            parent_descriptor,
        )
        raise
    close_error = _close_descriptors_best_effort(parent_descriptor)
    if close_error is not None:
        _close_descriptors_best_effort(site_descriptor)
        raise close_error
    return site_descriptor, observed, authority_binding


def _verify_direct_runtime_site_authority() -> dict[str, object]:
    descriptor = _DIRECT_RUNTIME_SITE_DESCRIPTOR
    binding = _DIRECT_RUNTIME_SITE_BINDING
    authority = _DIRECT_RUNTIME_SITE_AUTHORITY_BINDING
    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 0
        or binding is None
        or authority is None
    ):
        raise PermissionError("runtime dependency site authority is absent")
    descriptor_metadata = os.fstat(descriptor)
    authority_path = Path(str(authority["path"]))
    authority_lstat = authority_path.lstat()
    authority_metadata = authority_path.stat()
    observed = {
        "path": str(authority_path),
        "device": descriptor_metadata.st_dev,
        "inode": descriptor_metadata.st_ino,
        "owner_uid": descriptor_metadata.st_uid,
        "mode": stat.S_IMODE(descriptor_metadata.st_mode),
    }
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or not stat.S_ISLNK(authority_lstat.st_mode)
        or (
            authority_metadata.st_dev,
            authority_metadata.st_ino,
            authority_metadata.st_uid,
            stat.S_IMODE(authority_metadata.st_mode),
        )
        != (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
            descriptor_metadata.st_uid,
            stat.S_IMODE(descriptor_metadata.st_mode),
        )
        or observed != authority
        or {
            **observed,
            "path": str(RUNTIME_DEPENDENCY_SITE_PATH),
        }
        != binding
    ):
        raise PermissionError(
            "runtime dependency procfd authority generation changed"
        )
    return dict(authority)


def _validate_bound_runtime_module_origin(
    origin: str,
    *,
    site_descriptor: int,
    authority_path: str,
    site_device: int,
    site_inode: int,
) -> str:
    """Bind one imported module origin to the retained site generation."""

    if not isinstance(origin, str) or not origin:
        raise PermissionError("runtime dependency module has no origin")
    authority = Path(authority_path)
    authority_metadata = authority.stat()
    descriptor_metadata = os.fstat(site_descriptor)
    prefix = str(authority) + os.sep
    if (
        not origin.startswith(prefix)
        or (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
        )
        != (site_device, site_inode)
        or (
            authority_metadata.st_dev,
            authority_metadata.st_ino,
        )
        != (site_device, site_inode)
    ):
        raise PermissionError(
            "runtime dependency import escaped the site authority"
        )
    relative = os.path.relpath(origin, str(authority))
    relative_path = Path(relative)
    if (
        relative in {"", ".", ".."}
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise PermissionError("runtime dependency module origin is malformed")
    before = os.stat(
        relative,
        dir_fd=site_descriptor,
        follow_symlinks=True,
    )
    origin_metadata = Path(origin).stat()
    resolved_authority = authority.resolve(strict=True)
    resolved_origin = Path(origin).resolve(strict=True)
    try:
        resolved_origin.relative_to(resolved_authority)
    except ValueError as error:
        raise PermissionError(
            "runtime dependency import resolved outside its authority"
        ) from error
    after = os.stat(
        relative,
        dir_fd=site_descriptor,
        follow_symlinks=True,
    )
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_size",
        "st_mtime_ns",
    )
    if (
        not stat.S_ISREG(origin_metadata.st_mode)
        or any(
            getattr(before, field) != getattr(origin_metadata, field)
            or getattr(origin_metadata, field) != getattr(after, field)
            for field in identity_fields
        )
    ):
        raise PermissionError(
            "runtime dependency module origin generation changed"
        )
    return origin


def _bootstrap_direct_actual_runtime() -> None:
    """Add the bound venv packages without importing ``site`` or ``.pth``."""

    global _DIRECT_RUNTIME_BOOTSTRAPPED
    global _DIRECT_RUNTIME_SITE_AUTHORITY_BINDING
    global _DIRECT_RUNTIME_SITE_BINDING
    global _DIRECT_RUNTIME_SITE_DESCRIPTOR
    if _DIRECT_RUNTIME_BOOTSTRAPPED:
        raise RuntimeError("actual runtime dependency bootstrap repeated")
    executable = Path(sys.executable)
    executable_metadata = executable.lstat()
    if (
        executable != ACTUAL_PYTHON_PATH
        or executable.is_symlink()
        or executable.resolve(strict=True) != executable
        or not stat.S_ISREG(executable_metadata.st_mode)
        or executable_metadata.st_uid != 0
        or executable_metadata.st_nlink != 1
        or stat.S_IMODE(executable_metadata.st_mode) & 0o022
        or sys.flags.isolated != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or any(
            name in sys.modules
            for name in ("site", "sitecustomize", "usercustomize")
        )
        or any(
            "site-packages" in value or "dist-packages" in value
            for value in sys.path
        )
    ):
        raise PermissionError(
            "actual adapter requires the exact root Python -I -S runtime"
        )
    (
        site_descriptor,
        observed_site_binding,
        authority_binding,
    ) = _open_runtime_dependency_site_authority(
        RUNTIME_DEPENDENCY_SITE_PATH,
        EXPECTED_RUNTIME_DEPENDENCY_SITE_BINDING,
    )
    try:
        sys.path.insert(0, str(authority_binding["path"]))
        sys.path.insert(0, str(REPOSITORY))
    except BaseException:
        _close_descriptors_best_effort(site_descriptor)
        raise
    _DIRECT_RUNTIME_SITE_DESCRIPTOR = site_descriptor
    _DIRECT_RUNTIME_SITE_BINDING = observed_site_binding
    _DIRECT_RUNTIME_SITE_AUTHORITY_BINDING = authority_binding
    _verify_direct_runtime_site_authority()
    _DIRECT_RUNTIME_BOOTSTRAPPED = True


if __name__ == "__main__":
    _bootstrap_direct_actual_runtime()

if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from cure_lite.cache.schema import canonical_json
from cure_lite_v24 import dr_gate


R2_EXECUTION_IDENTITY_SCHEMA = (
    "cure-lite-v24-D_R-structural-r2-execution-identity-v1"
)
EXPECTED_R1_SCIENTIFIC_PATH_COUNT = 103
EXPECTED_R1_SOURCE_CLOSURE_FINGERPRINT = (
    "28d26759a68785e9c99917fcfa8b36430c7f6e5463282d66eeab5c711e425e9f"
)
_LEGACY_CLI_MODULE = "tools.run_cure_lite_v24_gcr_pacre_dr_gate"
_ALLOWED_R2_MODES = frozenset(
    {"preaccess-create", "preaccess-verify", "real"}
)
_RUNTIME_LAUNCH_AUTHORIZATION_OPTION = (
    "--runtime-launch-authorization"
)
_RUNTIME_ATTESTATION_ENV = (
    "CURE_LITE_V24_RUNTIME_ATTESTATION_PATH"
)
_RUNTIME_RESERVED_ENV_PREFIX = "CURE_LITE_V24_RUNTIME_"

_R1_EXPECTED = {
    "GCR_PACRE_DR_GATE_SCHEMA": (
        "cure-lite-v24-gcr-pacre-real-dr-structural-gate-v1"
    ),
    "GCR_PACRE_DR_RUN_ID": (
        "gcr_pacre_v24_D_R_zero_update_structural_r1"
    ),
    "GCR_PACRE_DR_PREACCESS_SCHEMA": (
        "cure-lite-v24-D_R-structural-preaccess-authorization-v1"
    ),
    "GCR_PACRE_DR_PREACCESS_STAGE_ID": (
        "gcr_pacre_v24_D_R_structural"
    ),
    "GCR_PACRE_DR_PREACCESS_STATUS": (
        "GCR_PACRE_V24_D_R_STRUCTURAL_PREACCESS_AUTHORIZED"
    ),
    "GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA": (
        "cure-lite-v24-split-access-audit-v1"
    ),
    "GCR_PACRE_DR_RUN_START_SCHEMA": (
        "cure-lite-v24-D_R-persistent-run-start-v1"
    ),
    "GCR_PACRE_DR_RUN_START_PARENT": (
        "runs/irstd1k_stage_a_seed42"
    ),
    "GCR_PACRE_DR_ACCESS_AUDIT_PATH": (
        "protocols/IRSTD-1K/gcr_pacre_v24/"
        "D_R_structural_access_audit.json"
    ),
    "GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH": (
        "protocols/IRSTD-1K/gcr_pacre_v24/"
        "D_R_structural_authorization.json"
    ),
    "GCR_PACRE_DR_RECEIPT_PATH": (
        "protocols/IRSTD-1K/gcr_pacre_v24/"
        "D_R_structural_receipt.json"
    ),
}

_R2_IDENTITY = {
    "GCR_PACRE_DR_GATE_SCHEMA": (
        "cure-lite-v24-gcr-pacre-real-dr-structural-gate-r2-v1"
    ),
    "GCR_PACRE_DR_RUN_ID": (
        "gcr_pacre_v24_D_R_zero_update_structural_r2"
    ),
    "GCR_PACRE_DR_PREACCESS_SCHEMA": (
        "cure-lite-v24-D_R-structural-r2-authorization-v1"
    ),
    "GCR_PACRE_DR_PREACCESS_STAGE_ID": (
        "gcr_pacre_v24_D_R_structural_r2"
    ),
    "GCR_PACRE_DR_PREACCESS_STATUS": (
        "GCR_PACRE_V24_D_R_STRUCTURAL_R2_AUTHORIZED"
    ),
    "GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA": (
        "cure-lite-v24-split-access-audit-v1"
    ),
    "GCR_PACRE_DR_RUN_START_SCHEMA": (
        "cure-lite-v24-D_R-persistent-run-start-r2-v1"
    ),
    "GCR_PACRE_DR_RUN_START_PARENT": (
        "runs/irstd1k_stage_a_seed42/"
        "gcr_pacre_v24_D_R_structural_attempt_r2"
    ),
    "GCR_PACRE_DR_ACCESS_AUDIT_PATH": (
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_access_audit.json"
    ),
    "GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH": (
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_authorization.json"
    ),
    "GCR_PACRE_DR_RECEIPT_PATH": (
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_receipt.json"
    ),
}


def stable_fingerprint(value: object) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _private_r2_directory(relative: str) -> Path:
    path = (REPOSITORY / relative).absolute()
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise PermissionError(
            "r2 evidence directory must be canonical, owned, and mode 0700"
        )
    return path


def _project_r2_run_start_marker_path(
    authorization_fingerprint: object,
) -> Path:
    """Project the future r2 marker path without materializing its parent."""

    if (
        type(authorization_fingerprint) is not str
        or len(authorization_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in authorization_fingerprint
        )
    ):
        raise ValueError("authorization fingerprint must be exact SHA-256")
    relative = Path(
        str(_R2_IDENTITY["GCR_PACRE_DR_RUN_START_PARENT"])
    )
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RuntimeError("r2 run-start parent identity is malformed")
    parent = (REPOSITORY / relative).absolute()
    anchor = parent.parent
    anchor_metadata = anchor.lstat()
    if (
        not stat.S_ISDIR(anchor_metadata.st_mode)
        or stat.S_ISLNK(anchor_metadata.st_mode)
        or anchor.resolve(strict=True) != anchor
        or anchor_metadata.st_uid != os.getuid()
        or stat.S_IMODE(anchor_metadata.st_mode) & 0o002
    ):
        raise PermissionError("r2 run-start parent anchor is unsafe")
    if parent.exists() or parent.is_symlink():
        _private_r2_directory(str(relative))
    return parent / (
        "gcr_pacre_v24_D_R_structural_run_start_"
        f"{authorization_fingerprint}.json"
    )


def _project_live_r2_preaccess_run_start_marker_path(
    preaccess_token: object,
) -> Path:
    """Summary-only projection for one live private r2 preaccess token."""

    if not dr_gate._is_live_real_preaccess_token(preaccess_token):
        raise PermissionError(
            "run-start projection requires a private preaccess token"
        )
    return _project_r2_run_start_marker_path(
        preaccess_token.authorization_fingerprint
    )


def _r2_protocol_path(name: str) -> Path:
    return (
        REPOSITORY / str(_R2_IDENTITY[name])
    ).absolute()


def _verify_or_seal_evidence(
    path: Path,
    *,
    fingerprint_field: str,
    seal: bool,
) -> dict[str, object]:
    if path.parent != _private_r2_directory(
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
    ):
        raise PermissionError("r2 protocol evidence escaped its private root")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = path.lstat()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        raw = b"".join(chunks)
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or descriptor_metadata.st_uid != os.getuid()
            or descriptor_metadata.st_nlink != 1
            or (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
            or path.is_symlink()
            or path.resolve(strict=True) != path
        ):
            raise PermissionError("r2 protocol evidence identity is unsafe")
        payload = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(payload, dict)
            or raw != (canonical_json(payload) + "\n").encode("utf-8")
        ):
            raise ValueError("r2 protocol evidence is not canonical JSON")
        body = dict(payload)
        fingerprint = body.pop(fingerprint_field, None)
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or fingerprint != stable_fingerprint(body)
        ):
            raise PermissionError("r2 protocol evidence fingerprint is invalid")
        if seal:
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            _fsync_parent(path)
        final_descriptor = os.fstat(descriptor)
        final_path = path.lstat()
        os.lseek(descriptor, 0, os.SEEK_SET)
        final_chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            final_chunks.append(block)
        if (
            stat.S_IMODE(final_descriptor.st_mode) != 0o444
            or final_descriptor.st_uid != os.getuid()
            or final_descriptor.st_nlink != 1
            or (final_descriptor.st_dev, final_descriptor.st_ino)
            != (final_path.st_dev, final_path.st_ino)
            or b"".join(final_chunks) != raw
        ):
            raise RuntimeError("r2 protocol evidence was not sealed exactly")
        return payload
    finally:
        os.close(descriptor)


def _verify_scientific_preaccess(*, seal: bool) -> None:
    _verify_or_seal_evidence(
        _r2_protocol_path("GCR_PACRE_DR_ACCESS_AUDIT_PATH"),
        fingerprint_field="receipt_fingerprint",
        seal=seal,
    )
    _verify_or_seal_evidence(
        _r2_protocol_path(
            "GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH"
        ),
        fingerprint_field="authorization_fingerprint",
        seal=seal,
    )


def _issued_token_registries_are_empty() -> bool:
    registries = (
        dr_gate._ISSUED_REAL_PREACCESS_TOKENS,
        dr_gate._ISSUED_REAL_RUN_START_TOKENS,
        dr_gate._CONSUMED_REAL_PREACCESS_TOKENS,
        dr_gate._STARTED_REAL_GATE_TOKENS,
    )
    return all(not value for value in registries)


def configure_r2_execution_identity() -> dict[str, object]:
    """Apply the exact r1→r2 identity transition once in a fresh process."""

    if _LEGACY_CLI_MODULE in sys.modules:
        raise RuntimeError(
            "legacy gate CLI was imported before r2 identity configuration"
        )
    if not _issued_token_registries_are_empty():
        raise RuntimeError(
            "r2 identity cannot be configured after token issuance"
        )
    observed = {
        name: getattr(dr_gate, name, None)
        for name in sorted(_R1_EXPECTED)
    }
    if observed != {
        name: _R1_EXPECTED[name] for name in sorted(_R1_EXPECTED)
    }:
        raise RuntimeError("frozen r1 identity constants changed")

    scientific_paths = tuple(dr_gate.GCR_PACRE_DR_IMPLEMENTATION_PATHS)
    implementation_binding = dr_gate._implementation_binding()
    source_closure_fingerprint = stable_fingerprint(
        dict(implementation_binding)
    )
    if (
        len(scientific_paths) != EXPECTED_R1_SCIENTIFIC_PATH_COUNT
        or len(implementation_binding)
        != EXPECTED_R1_SCIENTIFIC_PATH_COUNT
        or tuple(path for path, _ in implementation_binding)
        != scientific_paths
        or source_closure_fingerprint
        != EXPECTED_R1_SOURCE_CLOSURE_FINGERPRINT
    ):
        raise RuntimeError("frozen r1 scientific source closure changed")

    for name, value in _R2_IDENTITY.items():
        setattr(dr_gate, name, value)
    transitioned = {
        name: getattr(dr_gate, name, None)
        for name in sorted(_R2_IDENTITY)
    }
    expected = {
        name: _R2_IDENTITY[name] for name in sorted(_R2_IDENTITY)
    }
    if transitioned != expected:
        raise RuntimeError("r2 execution identity transition was incomplete")

    adapter_relative = str(Path(__file__).resolve().relative_to(REPOSITORY))
    if adapter_relative in scientific_paths:
        raise RuntimeError(
            "r2 execution adapter entered the frozen r1 scientific closure"
        )
    body: dict[str, object] = {
        "schema_version": R2_EXECUTION_IDENTITY_SCHEMA,
        "candidate": "GCR-PACRE-v24",
        "attempt_ordinal": 2,
        "prior_attempt_count": 1,
        "prior_attempt_status": (
            "OBSERVABILITY_LOST_NO_AUTHENTICATED_DECISION"
        ),
        "identity_transition": {
            name: {
                "r1": _R1_EXPECTED[name],
                "r2": _R2_IDENTITY[name],
            }
            for name in sorted(_R1_EXPECTED)
        },
        "frozen_scientific_path_count": len(scientific_paths),
        "frozen_scientific_paths_fingerprint": stable_fingerprint(
            list(scientific_paths)
        ),
        "frozen_scientific_source_closure_fingerprint": (
            source_closure_fingerprint
        ),
        "adapter_repo_path": adapter_relative,
        "numerical_or_scientific_change_authorized": False,
        "D_R_payload_authorized_by_adapter": False,
        "D_V_payload_authorized": False,
        "D_T_payload_authorized": False,
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
        "parameter_updates_authorized": 0,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
    }
    return {
        **body,
        "execution_identity_fingerprint": stable_fingerprint(body),
    }


def _identity_summary() -> int:
    print(canonical_json(configure_r2_execution_identity()))
    return 0


def _split_runtime_launch_authorization(
    argv: Sequence[str],
) -> tuple[list[str], str | None]:
    """Remove the adapter-only launch authorization from delegated argv."""

    materialized = [str(value) for value in argv]
    delegated: list[str] = []
    authorization: str | None = None
    ordinal = 0
    while ordinal < len(materialized):
        value = materialized[ordinal]
        if value.startswith(
            _RUNTIME_LAUNCH_AUTHORIZATION_OPTION + "="
        ):
            raise PermissionError(
                "runtime launch authorization requires a separate path argv"
            )
        if value != _RUNTIME_LAUNCH_AUTHORIZATION_OPTION:
            delegated.append(value)
            ordinal += 1
            continue
        if authorization is not None or ordinal + 1 >= len(materialized):
            raise PermissionError(
                "runtime launch authorization must occur exactly once"
            )
        authorization = materialized[ordinal + 1]
        if (
            not authorization
            or authorization.startswith("-")
            or not Path(authorization).is_absolute()
        ):
            raise PermissionError(
                "runtime launch authorization path must be absolute"
            )
        ordinal += 2
    return delegated, authorization


def _reject_reserved_runtime_environment() -> None:
    reserved = sorted(
        name
        for name in os.environ
        if name.startswith(_RUNTIME_RESERVED_ENV_PREFIX)
    )
    if reserved:
        raise PermissionError(
            "runtime attestation environment is forbidden outside real mode"
        )


def _verify_no_runtime_site_customization() -> dict[str, object]:
    if not _DIRECT_RUNTIME_BOOTSTRAPPED:
        raise PermissionError(
            "direct runtime site customization check has no bootstrap"
        )
    authority = _verify_direct_runtime_site_authority()
    if (
        sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
        or str(RUNTIME_DEPENDENCY_SITE_PATH) in sys.path
        or str(authority["path"]) not in sys.path
    ):
        raise PermissionError(
            "runtime site, pth, or customization processing was enabled"
        )
    return authority


def _runtime_import_smoke() -> int:
    if (
        not _DIRECT_RUNTIME_BOOTSTRAPPED
        or _DIRECT_RUNTIME_SITE_BINDING is None
    ):
        raise PermissionError(
            "runtime import smoke requires the direct isolated bootstrap"
        )
    authority = _verify_no_runtime_site_customization()
    import numpy
    import torch

    descriptor = _DIRECT_RUNTIME_SITE_DESCRIPTOR
    if isinstance(descriptor, bool) or not isinstance(descriptor, int):
        raise PermissionError("runtime dependency site descriptor is absent")
    origins = {
        "numpy": str(numpy.__file__),
        "torch": str(torch.__file__),
    }
    validated_origins = {
        name: _validate_bound_runtime_module_origin(
            origin,
            site_descriptor=descriptor,
            authority_path=str(authority["path"]),
            site_device=int(authority["device"]),
            site_inode=int(authority["inode"]),
        )
        for name, origin in origins.items()
    }
    final_authority = _verify_no_runtime_site_customization()
    cuda_initialized = bool(torch.cuda.is_initialized())
    if final_authority != authority or cuda_initialized:
        raise PermissionError(
            "runtime dependency imports escaped the bound site or touched CUDA"
        )
    body: dict[str, object] = {
        "schema_version": "cure-lite-v24-runtime-import-smoke-v1",
        "python_executable": sys.executable,
        "isolated": sys.flags.isolated == 1,
        "no_site": sys.flags.no_site == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "runtime_dependency_site_binding": dict(
            _DIRECT_RUNTIME_SITE_BINDING
        ),
        "runtime_dependency_site_authority_binding": authority,
        "runtime_dependency_import_mode": (
            "retained-procfd-no-site-processing"
        ),
        "pth_files_processed": False,
        "module_origins": validated_origins,
        "module_versions": {
            "numpy": str(numpy.__version__),
            "torch": str(torch.__version__),
        },
        "site_module_imported_after_dependencies": "site" in sys.modules,
        "sitecustomize_imported": "sitecustomize" in sys.modules,
        "usercustomize_imported": "usercustomize" in sys.modules,
        "torch_cuda_initialized": cuda_initialized,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
    }
    payload = {
        **body,
        "receipt_fingerprint": stable_fingerprint(body),
    }
    print(canonical_json(payload))
    return 0


def _verify_runtime_launch(
    runtime_launch_authorization_path: str,
) -> dict[str, object]:
    attestation_path = os.environ.get(_RUNTIME_ATTESTATION_ENV)
    if not attestation_path or not Path(attestation_path).is_absolute():
        raise PermissionError(
            "supervisor runtime attestation path is required"
        )
    from tools.cure_lite_v24_runtime_supervisor import (
        verify_child_runtime_attestation,
    )

    result = verify_child_runtime_attestation(
        attestation_path,
        runtime_launch_authorization_path,
    )
    if result.get("runtime_attestation_valid") is not True:
        raise PermissionError("runtime attestation did not validate")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    if _DIRECT_RUNTIME_BOOTSTRAPPED:
        _verify_no_runtime_site_customization()
    materialized = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--r2-execution-identity-summary",
        action="store_true",
    )
    parser.add_argument(
        "--runtime-import-smoke",
        action="store_true",
    )
    known, remaining = parser.parse_known_args(materialized)
    if (
        known.r2_execution_identity_summary
        and known.runtime_import_smoke
    ):
        raise ValueError("adapter inspection modes are mutually exclusive")
    if known.runtime_import_smoke:
        if remaining:
            raise ValueError(
                "runtime import smoke accepts no gate CLI arguments"
            )
        _reject_reserved_runtime_environment()
        return _runtime_import_smoke()
    if known.r2_execution_identity_summary:
        if remaining:
            raise ValueError(
                "identity summary accepts no gate CLI arguments"
            )
        _reject_reserved_runtime_environment()
        return _identity_summary()

    if not materialized or materialized[0] not in _ALLOWED_R2_MODES:
        raise PermissionError(
            "r2 identity adapter permits only real/preaccess modes"
        )
    delegated_argv, runtime_launch_authorization = (
        _split_runtime_launch_authorization(materialized)
    )
    mode = delegated_argv[0]
    if mode == "real":
        if not _DIRECT_RUNTIME_BOOTSTRAPPED:
            raise PermissionError(
                "real mode requires direct root Python -I -S execution"
            )
        if runtime_launch_authorization is None:
            raise PermissionError(
                "real mode requires runtime launch authorization"
            )
    else:
        if runtime_launch_authorization is not None:
            raise PermissionError(
                "runtime launch authorization is forbidden in preaccess modes"
            )
        _reject_reserved_runtime_environment()
    configure_r2_execution_identity()
    if mode == "real":
        assert runtime_launch_authorization is not None
        _verify_runtime_launch(runtime_launch_authorization)
    # Import only after the identity transition.  The delegated module uses
    # ``from dr_gate import ...`` and therefore freezes the r2 values above.
    from tools import run_cure_lite_v24_gcr_pacre_dr_gate as delegated

    delegated_paths = {
        "GCR_PACRE_DR_ACCESS_AUDIT_PATH": (
            delegated.GCR_PACRE_DR_ACCESS_AUDIT_PATH
        ),
        "GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH": (
            delegated.GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH
        ),
        "GCR_PACRE_DR_RECEIPT_PATH": (
            delegated.GCR_PACRE_DR_RECEIPT_PATH
        ),
    }
    if delegated_paths != {
        name: _R2_IDENTITY[name] for name in sorted(delegated_paths)
    }:
        raise RuntimeError("delegated CLI did not inherit all r2 paths")
    if _DIRECT_RUNTIME_BOOTSTRAPPED:
        _verify_no_runtime_site_customization()
    if mode in {"preaccess-verify", "real"}:
        _verify_scientific_preaccess(seal=False)
    if mode == "real":
        _private_r2_directory(
            str(_R2_IDENTITY["GCR_PACRE_DR_RUN_START_PARENT"])
        )
    original_marker_resolver = (
        delegated.required_gcr_pacre_dr_run_start_marker_path
    )
    if original_marker_resolver is not (
        dr_gate.required_gcr_pacre_dr_run_start_marker_path
    ):
        raise RuntimeError(
            "delegated CLI run-start marker resolver identity changed"
        )
    if mode in {"preaccess-create", "preaccess-verify"}:
        delegated.required_gcr_pacre_dr_run_start_marker_path = (
            _project_live_r2_preaccess_run_start_marker_path
        )
    try:
        result = int(delegated.main(delegated_argv))
    finally:
        delegated.required_gcr_pacre_dr_run_start_marker_path = (
            original_marker_resolver
        )
    if _DIRECT_RUNTIME_BOOTSTRAPPED:
        _verify_no_runtime_site_customization()
    if mode == "preaccess-create":
        _verify_scientific_preaccess(seal=True)
    elif mode == "real":
        _verify_or_seal_evidence(
            _r2_protocol_path("GCR_PACRE_DR_RECEIPT_PATH"),
            fingerprint_field="receipt_fingerprint",
            seal=True,
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
