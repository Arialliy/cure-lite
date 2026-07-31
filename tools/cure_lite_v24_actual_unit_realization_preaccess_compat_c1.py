#!/usr/bin/env python3
"""Create-only realizer for the r2 preaccess compatibility-c1 unit.

The scientific attempt remains r2.  Before any predecessor byte is executed,
this module reads the fixed frozen predecessor through a no-follow descriptor,
proves its regular-file generation is stable, and verifies its frozen SHA-256.
Only then are those already-verified bytes compiled in an isolated namespace.

The compatibility lane may create exactly one new static user-unit fragment
and ask the user manager to daemon-reload.  It cannot start, enable, stop,
remove, or mutate the protected predecessor unit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType
from typing import Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
COMPAT_REALIZER_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c1.py"
).resolve()
FROZEN_REALIZER_PATH = (
    REPOSITORY / "tools/cure_lite_v24_actual_unit_realization.py"
).resolve()
FROZEN_REALIZER_SHA256 = (
    "0d66bc4007366588ed1393b21092cc57d58e0f7fca084f7266a00e6818703fd9"
)

EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()
COMPAT_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service"
)
PROTECTED_PREDECESSOR_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2.service"
)
COMPAT_UNIT_DIRECTORY = Path(f"/run/user/{os.getuid()}/systemd/user")
PROTECTED_PREDECESSOR_FRAGMENT_PATH = (
    COMPAT_UNIT_DIRECTORY / PROTECTED_PREDECESSOR_UNIT
)
COMPAT_TEMPLATE_PATH = (
    REPOSITORY
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service.template"
).resolve()
COMPAT_TEMPLATE_SHA256 = (
    "c67384b712b1e9f5573c16e2ee75b163124335176a9218e17ce8f59b35d009d5"
)
COMPAT_SUPERVISOR_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c1.py"
).resolve()
COMPAT_BRIDGE_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility.py"
).resolve()
COMPAT_BRIDGE_SOURCE_SHA256 = (
    "fe715af48867f166d2e15727e0190844cfd79fb5c02fa5a440d294bb7f29e084"
)
COMPAT_BRIDGE_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_authorization.json"
)
COMPAT_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json"
)
COMPAT_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c1_unit_realization_authorization.json"
)
COMPAT_RECEIPT_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c1_unit_realization_receipt.json"
)
COMPAT_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c1_unit_realization_terminal.json"
)

COMPATIBILITY_CLOSURE_SCHEMA = (
    "cure-lite-v24-preaccess-compat-c1-unit-realization-closure-v1"
)
COMPATIBILITY_CLOSURE_KEY = "compatibility_closure"
_GENERATION_FIELDS = (
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
_SOURCE_BINDING_KEYS = {
    "path",
    "resolved_path",
    "path_is_symlink",
    "file_sha256",
    "device",
    "inode",
    "owner_uid",
    "owner_gid",
    "mode",
    "nlink",
    "size",
    "mtime_ns",
    "ctime_ns",
}
_PARENT_BINDING_KEYS = {
    "path",
    "device",
    "inode",
    "owner_uid",
    "owner_gid",
    "mode",
}
_CLOSURE_KEYS = {
    "schema_version",
    "scientific_attempt_ordinal",
    "runtime_compatibility_generation",
    "compat_source_generation",
    "frozen_predecessor_source_generation",
    "template_generation",
    "bridge_validator_source_generation",
    "bridge_compat_authorization_root",
    "bridge_authorization_window",
    "protected_predecessor_unit",
    "fixed_paths",
    "mutation_authority",
    "payload_authority",
    "automatic_retry_authorized",
    "resume_authorized",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "closure_fingerprint",
}
_MUTATION_AUTHORITY = {
    "install_new_static_compat_fragment": True,
    "daemon_reload": True,
    "verify_new_static_shadow": True,
    "enable": False,
    "start": False,
    "stop": False,
    "remove": False,
    "protected_predecessor_fragment_mutation": False,
    "protected_predecessor_unit_mutation": False,
}
_RECEIPT_BODY_KEYS = {
    "schema_version",
    "candidate",
    "stage_id",
    "attempt_id",
    "unit_name",
    "created_at_utc",
    "authorization_path",
    "authorization_file_sha256",
    "authorization_fingerprint",
    "instruction_id",
    "manager_generation",
    "unit_path_policy",
    "template_binding",
    "rendered_fragment",
    "runtime_spec_binding",
    "expected_future_runtime_spec_path",
    "runtime_spec_absent_at_receipt",
    "executable_bindings",
    "fragment_identity",
    "full_static_shadow",
    "completed_actions",
    "static",
    "enabled",
    "started",
    "removed",
    "passed",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
}
_FRAGMENT_IDENTITY_KEYS = {
    "path",
    "file_sha256",
    "device",
    "inode",
    "owner_uid",
    "mode",
    "nlink",
}
_RUNTIME_DYNAMIC_SHADOW_FIELDS = {
    "ActiveState",
    "SubState",
}
_BRIDGE_ROOT_VOLATILE_PARENT_FIELDS = {
    "parent_size",
    "parent_mtime_ns",
    "parent_ctime_ns",
}
_BRIDGE_LANE_REQUIRED_TRUE = {
    "compatibility_source_preparation_authorized",
    "environment_metadata_audit_authorized",
    "dummy_systemd_integration_authorized",
    "compat_unit_realization_authorized",
    "compat_unit_fragment_install_authorized",
    "compat_daemon_reload_authorized",
}
_BRIDGE_LANE_REQUIRED_FALSE = {
    "compat_enable_authorized",
    "compat_start_authorized",
    "compat_stop_authorized",
    "compat_remove_authorized",
    "predecessor_unit_mutation_authorized",
    "runtime_spec_creation_authorized_by_this_receipt",
    "runtime_launch_authorization_authorized_by_this_receipt",
    "payload_access_authorized",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8"),
    ).hexdigest()


def _stat_snapshot(value: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(value, name)) for name in _GENERATION_FIELDS)


def _stable_regular_read(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Read one canonical regular-file generation without following links."""

    target = Path(path)
    if not target.is_absolute() or target.name in {"", ".", ".."}:
        raise ValueError("source path must be an absolute file path")
    before = os.lstat(target)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or target.resolve(strict=True) != target
    ):
        raise PermissionError("bound source is not a canonical regular file")
    descriptor = os.open(
        target,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        linked_after = os.stat(target, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if (
        _stat_snapshot(before) != _stat_snapshot(opened)
        or _stat_snapshot(opened) != _stat_snapshot(after)
        or _stat_snapshot(after) != _stat_snapshot(linked_after)
        or not stat.S_ISREG(opened.st_mode)
    ):
        raise PermissionError("bound source generation changed while reading")
    raw = b"".join(chunks)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise PermissionError("bound source SHA-256 changed")
    return raw, {
        "path": str(target),
        "resolved_path": str(target),
        "path_is_symlink": False,
        "file_sha256": digest,
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "owner_uid": opened.st_uid,
        "owner_gid": opened.st_gid,
        "mode": stat.S_IMODE(opened.st_mode),
        "nlink": opened.st_nlink,
        "size": opened.st_size,
        "mtime_ns": opened.st_mtime_ns,
        "ctime_ns": opened.st_ctime_ns,
    }


def _same_generation(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    return (
        set(observed) == _SOURCE_BINDING_KEYS
        and set(expected) == _SOURCE_BINDING_KEYS
        and _canonical_json(observed) == _canonical_json(expected)
    )


def _require_running_source_generation() -> dict[str, object]:
    _, observed = _stable_regular_read(COMPAT_REALIZER_PATH)
    if not _same_generation(observed, _COMPAT_SOURCE_LOAD_BINDING):
        raise PermissionError(
            "executing compatibility realizer source generation was replaced",
        )
    return observed


def _require_frozen_predecessor_generation() -> dict[str, object]:
    _, observed = _stable_regular_read(
        FROZEN_REALIZER_PATH,
        expected_sha256=FROZEN_REALIZER_SHA256,
    )
    if not _same_generation(observed, _FROZEN_REALIZER_LOAD_BINDING):
        raise PermissionError(
            "frozen predecessor source generation was replaced",
        )
    return observed


def _parse_utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PermissionError(f"{name} timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PermissionError(f"{name} timestamp is malformed") from error
    if parsed.tzinfo is None:
        raise PermissionError(f"{name} timestamp is naive")
    return parsed.astimezone(timezone.utc)


def _bridge_roots_same(
    observed: object,
    expected: object,
) -> bool:
    """Compare a bridge sealed root while allowing sibling creation."""

    if (
        not isinstance(observed, Mapping)
        or not isinstance(expected, Mapping)
        or set(observed) != set(expected)
    ):
        return False
    return all(
        observed[key] == expected[key]
        for key in observed
        if key not in _BRIDGE_ROOT_VOLATILE_PARENT_FIELDS
    )


def _load_bridge_policy_from_verified_bytes(
    raw: bytes,
) -> ModuleType:
    name = (
        "tools._cure_lite_v24_preaccess_schema_compatibility_"
        "for_unit_realization_c1"
    )
    module = ModuleType(name)
    module.__file__ = str(COMPAT_BRIDGE_SOURCE_PATH)
    module.__package__ = "tools"
    code = compile(
        raw,
        str(COMPAT_BRIDGE_SOURCE_PATH),
        "exec",
        dont_inherit=True,
    )
    exec(code, module.__dict__)
    return module


def _validate_bridge_authorization(
    *,
    require_fresh: bool = True,
    require_future_absence: bool,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Run the fixed bridge's strict validator from one verified generation."""

    raw, source = _stable_regular_read(
        COMPAT_BRIDGE_SOURCE_PATH,
        expected_sha256=COMPAT_BRIDGE_SOURCE_SHA256,
    )
    policy = _load_bridge_policy_from_verified_bytes(raw)
    if (
        Path(policy.COMPAT_AUTHORIZATION_PATH)
        != COMPAT_BRIDGE_AUTHORIZATION_PATH
        or Path(policy.COMPAT_UNIT_REALIZER_SOURCE_PATH)
        != COMPAT_REALIZER_PATH
        or policy.COMPAT_UNIT_NAME != COMPAT_UNIT
    ):
        raise PermissionError("bridge compatibility interface changed")
    authorization, root = policy.validate_compat_authorization(
        COMPAT_BRIDGE_AUTHORIZATION_PATH,
        require_fresh=require_fresh,
        require_future_absence=require_future_absence,
    )
    _, source_after = _stable_regular_read(
        COMPAT_BRIDGE_SOURCE_PATH,
        expected_sha256=COMPAT_BRIDGE_SOURCE_SHA256,
    )
    if not _same_generation(source_after, source):
        raise PermissionError(
            "bridge validator source generation changed while executing",
        )
    lane = authorization.get("compat_lane_authority")
    scientific = authorization.get("scientific_authority")
    if (
        authorization.get("schema_version")
        != "cure-lite-v24-r2-preaccess-schema-compat-c1-authorization-v1"
        or authorization.get("scientific_attempt_ordinal") != 2
        or authorization.get("runtime_compatibility_id") != "c1"
        or authorization.get("authorized_uid") != os.getuid()
        or not isinstance(lane, Mapping)
        or any(lane.get(field) is not True for field in _BRIDGE_LANE_REQUIRED_TRUE)
        or any(
            lane.get(field) is not False
            for field in _BRIDGE_LANE_REQUIRED_FALSE
        )
        or not isinstance(scientific, Mapping)
        or scientific.get("scientific_attempt_ordinal") != 2
        or scientific.get("runtime_compatibility_id") != "c1"
        or scientific.get("fresh_scientific_attempt") is not False
        or scientific.get("automatic_retry") is not False
        or scientific.get("resume") is not False
        or scientific.get("allowed_splits") != ["D_R"]
        or root.get("path") != str(COMPAT_BRIDGE_AUTHORIZATION_PATH)
        or root.get("file_sha256") is None
        or root.get("fingerprint")
        != authorization.get("authorization_fingerprint")
        or root.get("mode") != 0o444
        or root.get("nlink") != 1
    ):
        raise PermissionError(
            "bridge authorization does not authorize the narrow unit lane",
        )
    return dict(authorization), dict(root), source


def _bridge_authorization_window(
    authorization: Mapping[str, object],
    root: Mapping[str, object],
) -> dict[str, object]:
    fingerprint = authorization.get("authorization_fingerprint")
    file_sha256 = root.get("file_sha256")
    if (
        root.get("path") != str(COMPAT_BRIDGE_AUTHORIZATION_PATH)
        or not isinstance(file_sha256, str)
        or len(file_sha256) != 64
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or root.get("fingerprint") != fingerprint
    ):
        raise PermissionError(
            "bridge authorization generation root is off-path or malformed",
        )
    created = _parse_utc(
        authorization.get("created_at_utc"),
        name="bridge authorization created",
    )
    issued = _parse_utc(
        authorization.get("issued_at_utc"),
        name="bridge authorization issued",
    )
    expires = _parse_utc(
        authorization.get("expires_at_utc"),
        name="bridge authorization expires",
    )
    if not issued <= created <= expires:
        raise PermissionError("bridge authorization chronology changed")
    return {
        "path": str(COMPAT_BRIDGE_AUTHORIZATION_PATH),
        "file_sha256": file_sha256,
        "authorization_fingerprint": fingerprint,
        "created_at_utc": authorization["created_at_utc"],
        "issued_at_utc": authorization["issued_at_utc"],
        "expires_at_utc": authorization["expires_at_utc"],
    }


def _require_timestamp_inside_bridge_window(
    value: object,
    *,
    window: Mapping[str, object],
    name: str,
) -> None:
    observed = _parse_utc(value, name=name)
    issued = _parse_utc(
        window.get("issued_at_utc"),
        name="sealed bridge authorization issued",
    )
    expires = _parse_utc(
        window.get("expires_at_utc"),
        name="sealed bridge authorization expires",
    )
    if not issued <= observed <= expires:
        raise PermissionError(f"{name} is outside bridge authorization window")


def _validate_unit_authorization_window(
    authorization: Mapping[str, object],
) -> tuple[datetime, datetime]:
    created = _parse_utc(
        authorization.get("created_at_utc"),
        name="unit authorization created",
    )
    issued = _parse_utc(
        authorization.get("issued_at_utc"),
        name="unit authorization issued",
    )
    expires = _parse_utc(
        authorization.get("expires_at_utc"),
        name="unit authorization expires",
    )
    if (
        not issued <= created <= expires
        or (expires - issued).total_seconds() > 300
    ):
        raise PermissionError(
            "unit realization authorization window is malformed",
        )
    return issued, expires


def _require_unit_authorization_fresh(
    authorization: Mapping[str, object],
) -> None:
    issued, expires = _validate_unit_authorization_window(authorization)
    if not issued <= datetime.now(timezone.utc) <= expires:
        raise PermissionError("unit realization authorization is stale")


def _require_create_targets_absent() -> None:
    protected = (
        COMPAT_AUTHORIZATION_PATH,
        COMPAT_RECEIPT_PATH,
        COMPAT_TERMINAL_PATH,
        COMPAT_UNIT_DIRECTORY / COMPAT_UNIT,
    )
    if any(os.path.lexists(path) for path in protected):
        raise PermissionError(
            "compatibility realization create-once target already exists",
        )


def _require_apply_targets_pristine() -> None:
    protected = (
        COMPAT_RECEIPT_PATH,
        COMPAT_TERMINAL_PATH,
        COMPAT_UNIT_DIRECTORY / COMPAT_UNIT,
    )
    if any(os.path.lexists(path) for path in protected):
        raise PermissionError(
            "compatibility realization is terminal or already materialized",
        )


def _parent_binding(path: Path) -> dict[str, object]:
    parent = Path(path)
    linked = os.lstat(parent)
    if (
        not stat.S_ISDIR(linked.st_mode)
        or stat.S_ISLNK(linked.st_mode)
        or parent.resolve(strict=True) != parent
    ):
        raise PermissionError("protected unit parent is not canonical")
    descriptor = os.open(
        parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        linked_after = os.lstat(parent)
    finally:
        os.close(descriptor)
    stable = (
        linked.st_dev,
        linked.st_ino,
        linked.st_mode,
        linked.st_uid,
        linked.st_gid,
    )
    if stable != (
        opened.st_dev,
        opened.st_ino,
        opened.st_mode,
        opened.st_uid,
        opened.st_gid,
    ) or stable != (
        linked_after.st_dev,
        linked_after.st_ino,
        linked_after.st_mode,
        linked_after.st_uid,
        linked_after.st_gid,
    ):
        raise PermissionError("protected unit parent generation changed")
    return {
        "path": str(parent),
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "owner_uid": opened.st_uid,
        "owner_gid": opened.st_gid,
        "mode": stat.S_IMODE(opened.st_mode),
    }


def _observe_protected_predecessor_fragment() -> dict[str, object]:
    target = PROTECTED_PREDECESSOR_FRAGMENT_PATH
    if (
        not target.is_absolute()
        or target.parent != COMPAT_UNIT_DIRECTORY
        or target.name != PROTECTED_PREDECESSOR_UNIT
    ):
        raise PermissionError("protected predecessor fragment path changed")
    parent = _parent_binding(target.parent)
    try:
        _, binding = _stable_regular_read(target)
    except FileNotFoundError:
        _, parent_fd_generation = (
            None,
            _parent_binding(target.parent),
        )
        if parent_fd_generation != parent:
            raise PermissionError(
                "protected predecessor parent changed during absence proof",
            )
        binding = None
    return {
        "unit_name": PROTECTED_PREDECESSOR_UNIT,
        "fragment_path": str(target),
        "fragment_exists": binding is not None,
        "fragment_generation": binding,
        "parent_generation": parent,
        "mutation_authorized": False,
    }


def _legacy_binding_projection(
    binding: Mapping[str, object],
) -> dict[str, object]:
    return {
        "path": binding["path"],
        "resolved_path": binding["resolved_path"],
        "path_is_symlink": binding["path_is_symlink"],
        "file_sha256": binding["file_sha256"],
        "device": binding["device"],
        "inode": binding["inode"],
        "owner_uid": binding["owner_uid"],
        "mode": binding["mode"],
    }


def _fixed_paths() -> dict[str, str]:
    return {
        "compat_realizer": str(COMPAT_REALIZER_PATH),
        "frozen_predecessor_realizer": str(FROZEN_REALIZER_PATH),
        "template": str(COMPAT_TEMPLATE_PATH),
        "python": str(Path("/usr/bin/python3.12")),
        "supervisor": str(COMPAT_SUPERVISOR_PATH),
        "bridge_validator": str(COMPAT_BRIDGE_SOURCE_PATH),
        "bridge_compat_authorization": str(
            COMPAT_BRIDGE_AUTHORIZATION_PATH,
        ),
        "runtime_spec": str(COMPAT_RUNTIME_SPEC_PATH),
        "unit_directory": str(COMPAT_UNIT_DIRECTORY),
        "compat_unit_fragment": str(
            COMPAT_UNIT_DIRECTORY / COMPAT_UNIT,
        ),
        "protected_predecessor_fragment": str(
            PROTECTED_PREDECESSOR_FRAGMENT_PATH,
        ),
        "authorization": str(COMPAT_AUTHORIZATION_PATH),
        "receipt": str(COMPAT_RECEIPT_PATH),
        "terminal": str(COMPAT_TERMINAL_PATH),
    }


def _require_exact_path(
    supplied: Path,
    expected: Path,
    *,
    name: str,
) -> None:
    if (
        not isinstance(supplied, Path)
        or not supplied.is_absolute()
        or supplied != expected
    ):
        raise PermissionError(f"compatibility realizer path changed:{name}")


def _require_authorize_paths(
    *,
    authorization_path: Path,
    template_path: Path,
    python_path: Path,
    supervisor_path: Path,
    runtime_spec_path: Path,
) -> None:
    for name, supplied, expected in (
        ("authorization", authorization_path, COMPAT_AUTHORIZATION_PATH),
        ("template", template_path, COMPAT_TEMPLATE_PATH),
        ("python", python_path, Path("/usr/bin/python3.12")),
        ("supervisor", supervisor_path, COMPAT_SUPERVISOR_PATH),
        ("runtime_spec", runtime_spec_path, COMPAT_RUNTIME_SPEC_PATH),
    ):
        _require_exact_path(supplied, expected, name=name)


def _require_apply_paths(
    *,
    authorization_path: Path,
    receipt_path: Path,
    terminal_path: Path,
) -> None:
    for name, supplied, expected in (
        ("authorization", authorization_path, COMPAT_AUTHORIZATION_PATH),
        ("receipt", receipt_path, COMPAT_RECEIPT_PATH),
        ("terminal", terminal_path, COMPAT_TERMINAL_PATH),
    ):
        _require_exact_path(supplied, expected, name=name)


# Seal both source generations before executing any frozen predecessor byte.
if (
    Path(__file__).absolute() != COMPAT_REALIZER_PATH
    or Path(__file__).resolve() != COMPAT_REALIZER_PATH
):
    raise PermissionError("compatibility realizer was loaded from an off-path source")
_COMPAT_SOURCE_LOAD_BYTES, _COMPAT_SOURCE_LOAD_BINDING = _stable_regular_read(
    COMPAT_REALIZER_PATH,
)
_FROZEN_REALIZER_BYTES, _FROZEN_REALIZER_LOAD_BINDING = _stable_regular_read(
    FROZEN_REALIZER_PATH,
    expected_sha256=FROZEN_REALIZER_SHA256,
)


def _load_frozen_realizer_from_verified_bytes() -> ModuleType:
    name = (
        "tools._cure_lite_v24_actual_unit_realization_frozen_"
        "for_preaccess_compat_c1"
    )
    module = ModuleType(name)
    module.__file__ = str(FROZEN_REALIZER_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        code = compile(
            _FROZEN_REALIZER_BYTES,
            str(FROZEN_REALIZER_PATH),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


legacy = _load_frozen_realizer_from_verified_bytes()
_frozen_expected_static_shadow = legacy._expected_static_shadow
_frozen_write_create_once_json = legacy.write_create_once_json
_frozen_create_authorization = legacy.create_authorization
_frozen_validate_authorization = legacy.validate_authorization
_frozen_realize_actual_unit = legacy.realize_actual_unit


def _compat_expected_static_shadow(fragment_path: Path) -> dict[str, str]:
    result = dict(_frozen_expected_static_shadow(fragment_path))
    if (
        result.get("Id") != COMPAT_UNIT
        or result.get("SuccessExitStatus") != "0 0"
    ):
        raise RuntimeError("frozen static-shadow expectation changed")
    result["SuccessExitStatus"] = "0"
    return result


def _require_fixed_authorization_body(
    body: Mapping[str, object],
) -> None:
    executable_bindings = body.get("executable_bindings")
    runtime_binding = body.get("runtime_spec_binding")
    if (
        body.get("unit_name") != COMPAT_UNIT
        or body.get("unit_directory") != str(COMPAT_UNIT_DIRECTORY)
        or not isinstance(executable_bindings, Mapping)
        or not isinstance(runtime_binding, Mapping)
        or runtime_binding.get("runtime_spec_path")
        != str(COMPAT_RUNTIME_SPEC_PATH)
        or executable_bindings.get("realization_tool", {}).get("path")
        != str(COMPAT_REALIZER_PATH)
        or executable_bindings.get("python", {}).get("path")
        != "/usr/bin/python3.12"
        or executable_bindings.get("supervisor", {}).get("path")
        != str(COMPAT_SUPERVISOR_PATH)
        or body.get("template_binding", {}).get("path")
        != str(COMPAT_TEMPLATE_PATH)
        or body.get("actions") != list(legacy._ACTIONS)
        or body.get("persistent_install_authorized") is not False
        or body.get("enable_authorized") is not False
        or body.get("start_authorized") is not False
        or body.get("remove_authorized") is not False
        or body.get("payload_authority") != "none"
        or body.get("D_R_payload_accessed") is not False
        or body.get("D_V_payload_accessed") is not False
        or body.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError(
            "compatibility realization authorization lane is not exact",
        )


def _build_compatibility_closure(
    authorization_body: Mapping[str, object],
) -> dict[str, object]:
    _require_fixed_authorization_body(authorization_body)
    compat_source = _require_running_source_generation()
    predecessor_source = _require_frozen_predecessor_generation()
    (
        bridge_authorization,
        bridge_authorization_root,
        bridge_source,
    ) = _validate_bridge_authorization(require_future_absence=True)
    if (
        bridge_authorization.get("instruction_id")
        != authorization_body.get("instruction_id")
        or bridge_authorization.get("authorization_basis")
        != authorization_body.get("authorization_basis")
    ):
        raise PermissionError(
            "unit realization is not caused by the bridge authorization",
        )
    bridge_window = _bridge_authorization_window(
        bridge_authorization,
        bridge_authorization_root,
    )
    for field in ("created_at_utc", "issued_at_utc"):
        _require_timestamp_inside_bridge_window(
            authorization_body.get(field),
            window=bridge_window,
            name=f"unit authorization {field}",
        )
    _, template = _stable_regular_read(
        COMPAT_TEMPLATE_PATH,
        expected_sha256=COMPAT_TEMPLATE_SHA256,
    )
    if (
        _legacy_binding_projection(compat_source)
        != authorization_body["executable_bindings"]["realization_tool"]
        or _legacy_binding_projection(template)
        != authorization_body["template_binding"]
    ):
        raise PermissionError(
            "compatibility source/template authorization binding changed",
        )
    closure: dict[str, object] = {
        "schema_version": COMPATIBILITY_CLOSURE_SCHEMA,
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_generation": "c1",
        "compat_source_generation": compat_source,
        "frozen_predecessor_source_generation": predecessor_source,
        "template_generation": template,
        "bridge_validator_source_generation": bridge_source,
        "bridge_compat_authorization_root": bridge_authorization_root,
        "bridge_authorization_window": bridge_window,
        "protected_predecessor_unit": (
            _observe_protected_predecessor_fragment()
        ),
        "fixed_paths": _fixed_paths(),
        "mutation_authority": deepcopy(_MUTATION_AUTHORITY),
        "payload_authority": "none",
        "automatic_retry_authorized": False,
        "resume_authorized": False,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    closure["closure_fingerprint"] = _fingerprint(closure)
    return closure


def _validate_compatibility_closure(
    authorization: Mapping[str, object],
    *,
    require_fresh: bool = True,
    require_bridge_fresh: bool = True,
    require_future_absence: bool = True,
) -> dict[str, object]:
    _require_fixed_authorization_body(authorization)
    _validate_unit_authorization_window(authorization)
    if require_fresh:
        _require_unit_authorization_fresh(authorization)
    closure = authorization.get(COMPATIBILITY_CLOSURE_KEY)
    if not isinstance(closure, Mapping) or set(closure) != _CLOSURE_KEYS:
        raise PermissionError("compatibility authorization closure is malformed")
    body = dict(closure)
    fingerprint = body.pop("closure_fingerprint", None)
    if (
        not isinstance(fingerprint, str)
        or fingerprint != _fingerprint(body)
        or closure.get("schema_version") != COMPATIBILITY_CLOSURE_SCHEMA
        or closure.get("scientific_attempt_ordinal") != 2
        or closure.get("runtime_compatibility_generation") != "c1"
        or closure.get("fixed_paths") != _fixed_paths()
        or closure.get("mutation_authority") != _MUTATION_AUTHORITY
        or closure.get("payload_authority") != "none"
        or closure.get("automatic_retry_authorized") is not False
        or closure.get("resume_authorized") is not False
        or closure.get("D_R_payload_accessed") is not False
        or closure.get("D_V_payload_accessed") is not False
        or closure.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("compatibility authorization semantics changed")
    source_fields = (
        "compat_source_generation",
        "frozen_predecessor_source_generation",
        "template_generation",
        "bridge_validator_source_generation",
    )
    if any(
        not isinstance(closure.get(field), Mapping)
        or set(closure[field]) != _SOURCE_BINDING_KEYS
        for field in source_fields
    ):
        raise PermissionError(
            "compatibility source generation binding is malformed",
        )
    compat_source = _require_running_source_generation()
    predecessor_source = _require_frozen_predecessor_generation()
    (
        bridge_authorization,
        bridge_authorization_root,
        bridge_source,
    ) = _validate_bridge_authorization(
        require_fresh=require_bridge_fresh,
        require_future_absence=require_future_absence,
    )
    bridge_window = _bridge_authorization_window(
        bridge_authorization,
        bridge_authorization_root,
    )
    _, template = _stable_regular_read(
        COMPAT_TEMPLATE_PATH,
        expected_sha256=COMPAT_TEMPLATE_SHA256,
    )
    protected = _observe_protected_predecessor_fragment()
    if (
        not _same_generation(
            closure["compat_source_generation"],
            compat_source,
        )
        or not _same_generation(
            closure["frozen_predecessor_source_generation"],
            predecessor_source,
        )
        or not _same_generation(
            closure["template_generation"],
            template,
        )
        or not _same_generation(
            closure["bridge_validator_source_generation"],
            bridge_source,
        )
        or not _bridge_roots_same(
            closure.get("bridge_compat_authorization_root"),
            bridge_authorization_root,
        )
        or closure.get("bridge_authorization_window") != bridge_window
        or bridge_authorization.get("instruction_id")
        != authorization.get("instruction_id")
        or bridge_authorization.get("authorization_basis")
        != authorization.get("authorization_basis")
        or closure.get("protected_predecessor_unit") != protected
        or _legacy_binding_projection(compat_source)
        != authorization["executable_bindings"]["realization_tool"]
        or _legacy_binding_projection(template)
        != authorization["template_binding"]
    ):
        raise PermissionError(
            "compatibility bridge/source/template/protected-unit generation changed",
        )
    for field in ("created_at_utc", "issued_at_utc"):
        _require_timestamp_inside_bridge_window(
            authorization.get(field),
            window=bridge_window,
            name=f"unit authorization {field}",
        )
    protected_value = closure["protected_predecessor_unit"]
    if (
        not isinstance(protected_value, Mapping)
        or protected_value.get("unit_name") != PROTECTED_PREDECESSOR_UNIT
        or protected_value.get("fragment_path")
        != str(PROTECTED_PREDECESSOR_FRAGMENT_PATH)
        or protected_value.get("mutation_authorized") is not False
        or not isinstance(protected_value.get("fragment_exists"), bool)
        or set(protected_value.get("parent_generation", {}))
        != _PARENT_BINDING_KEYS
        or (
            protected_value.get("fragment_exists") is True
            and (
                not isinstance(
                    protected_value.get("fragment_generation"),
                    Mapping,
                )
                or set(protected_value["fragment_generation"])
                != _SOURCE_BINDING_KEYS
            )
        )
        or (
            protected_value.get("fragment_exists") is False
            and protected_value.get("fragment_generation") is not None
        )
    ):
        raise PermissionError("protected predecessor binding is malformed")
    return dict(closure)


def _validate_receipt_transitive_binding(
    body: Mapping[str, object],
    *,
    require_fresh: bool = True,
    require_bridge_fresh: bool = True,
    require_future_absence: bool = True,
) -> dict[str, object]:
    expected_receipt_keys = set(_RECEIPT_BODY_KEYS)
    if "receipt_fingerprint" in body:
        expected_receipt_keys.add("receipt_fingerprint")
    if (
        set(body) != expected_receipt_keys
        or body.get("schema_version") != legacy.RECEIPT_SCHEMA
        or body.get("unit_name") != COMPAT_UNIT
    ):
        raise PermissionError(
            "compatibility receipt exact key/identity closure changed",
        )
    authorization_path = Path(str(body.get("authorization_path", "")))
    _require_exact_path(
        authorization_path,
        COMPAT_AUTHORIZATION_PATH,
        name="receipt_authorization",
    )
    authorization, identity = legacy._load_sealed_json_bound(
        authorization_path,
        "authorization_fingerprint",
    )
    if (
        set(authorization) != set(legacy._AUTH_KEYS)
        or authorization.get("schema_version")
        != legacy.AUTHORIZATION_SCHEMA
        or authorization.get("candidate") != legacy.CANDIDATE
        or authorization.get("stage_id") != legacy.STAGE_ID
        or authorization.get("attempt_id") != legacy.ATTEMPT_ID
        or authorization.get("unit_name") != COMPAT_UNIT
        or authorization.get("instruction_id") != legacy.INSTRUCTION_ID
        or authorization.get("authorization_basis")
        != legacy.AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != os.getuid()
    ):
        raise PermissionError(
            "compatibility authorization archival identity changed",
        )
    closure = _validate_compatibility_closure(
        authorization,
        require_fresh=require_fresh,
        require_bridge_fresh=require_bridge_fresh,
        require_future_absence=require_future_absence,
    )
    _require_timestamp_inside_bridge_window(
        body.get("created_at_utc"),
        window=closure["bridge_authorization_window"],
        name="unit realization receipt created_at_utc",
    )
    receipt_created = _parse_utc(
        body.get("created_at_utc"),
        name="unit realization receipt created_at_utc",
    )
    unit_issued = _parse_utc(
        authorization.get("issued_at_utc"),
        name="unit authorization issued_at_utc",
    )
    unit_expires = _parse_utc(
        authorization.get("expires_at_utc"),
        name="unit authorization expires_at_utc",
    )
    if not unit_issued <= receipt_created <= unit_expires:
        raise PermissionError(
            "unit receipt is outside unit authorization window",
        )
    if (
        body.get("candidate") != authorization.get("candidate")
        or body.get("stage_id") != authorization.get("stage_id")
        or body.get("attempt_id") != authorization.get("attempt_id")
        or body.get("instruction_id") != authorization.get("instruction_id")
        or body.get("authorization_file_sha256")
        != identity.get("file_sha256")
        or body.get("authorization_fingerprint")
        != authorization.get("authorization_fingerprint")
        or body.get("executable_bindings")
        != authorization.get("executable_bindings")
        or body.get("template_binding")
        != authorization.get("template_binding")
        or body.get("manager_generation")
        != authorization.get("manager_generation")
        or body.get("rendered_fragment")
        != authorization.get("rendered_fragment")
        or body.get("runtime_spec_binding")
        != authorization.get("runtime_spec_binding")
        or body.get("expected_future_runtime_spec_path")
        != str(COMPAT_RUNTIME_SPEC_PATH)
        or body.get("runtime_spec_absent_at_receipt") is not True
        or body.get("completed_actions") != list(legacy._ACTIONS)
        or body.get("static") is not True
        or body.get("enabled") is not False
        or body.get("started") is not False
        or body.get("removed") is not False
        or body.get("passed") is not True
        or body.get("payload_authority") != "none"
        or body.get("D_R_payload_accessed") is not False
        or body.get("D_V_payload_accessed") is not False
        or body.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError(
            "compatibility receipt does not transitively bind authorization",
        )
    legacy._validate_manager_generation(authorization["manager_generation"])
    transitioned_policy = (
        legacy._validate_daemon_reload_path_policy_transition(
            authorization["unit_path_policy"],
            body["unit_path_policy"],
            runtime_directory=COMPAT_UNIT_DIRECTORY,
            authorized_uid=os.getuid(),
        )
    )
    if transitioned_policy != body["unit_path_policy"]:
        raise PermissionError("compatibility receipt path policy changed")
    fragment_identity = body.get("fragment_identity")
    if (
        not isinstance(fragment_identity, Mapping)
        or set(fragment_identity) != _FRAGMENT_IDENTITY_KEYS
        or fragment_identity.get("path")
        != str(COMPAT_UNIT_DIRECTORY / COMPAT_UNIT)
        or fragment_identity.get("file_sha256")
        != authorization["rendered_fragment"]["sha256"]
        or fragment_identity.get("owner_uid") != os.getuid()
        or fragment_identity.get("mode") != 0o600
        or fragment_identity.get("nlink") != 1
    ):
        raise PermissionError(
            "compatibility receipt fragment identity is malformed",
        )
    _, live_fragment = legacy._stable_read_file(
        COMPAT_UNIT_DIRECTORY / COMPAT_UNIT,
    )
    live_fragment_projection = {
        key: live_fragment[key]
        for key in _FRAGMENT_IDENTITY_KEYS
    }
    if live_fragment_projection != dict(fragment_identity):
        raise PermissionError(
            "compatibility receipt fragment generation changed",
        )
    sealed_shadow = body.get("full_static_shadow")
    if (
        not isinstance(sealed_shadow, Mapping)
        or not set(legacy._SHADOW_PROPERTIES).issubset(sealed_shadow)
    ):
        raise PermissionError(
            "compatibility receipt static shadow is malformed",
        )
    raw_sealed_shadow = {
        name: sealed_shadow[name]
        for name in legacy._SHADOW_PROPERTIES
    }
    validated_sealed_shadow = legacy.validate_installed_shadow(
        raw_sealed_shadow,
        fragment_identity=fragment_identity,
        authorization=authorization,
    )
    if validated_sealed_shadow != dict(sealed_shadow):
        raise PermissionError(
            "compatibility receipt sealed static shadow changed",
        )
    return closure


def _compat_write_create_once_json(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    materialized = dict(body)
    if materialized.get("schema_version") == legacy.AUTHORIZATION_SCHEMA:
        if COMPATIBILITY_CLOSURE_KEY in materialized:
            raise ValueError("compatibility closure was already supplied")
        _require_exact_path(
            Path(path),
            COMPAT_AUTHORIZATION_PATH,
            name="authorization_output",
        )
        materialized[COMPATIBILITY_CLOSURE_KEY] = (
            _build_compatibility_closure(materialized)
        )
    elif materialized.get("schema_version") == legacy.RECEIPT_SCHEMA:
        _require_exact_path(
            Path(path),
            COMPAT_RECEIPT_PATH,
            name="receipt_output",
        )
        if os.path.lexists(COMPAT_TERMINAL_PATH):
            raise PermissionError(
                "PASS receipt cannot coexist with a terminal receipt",
            )
        _validate_receipt_transitive_binding(materialized)
    elif materialized.get("schema_version") == legacy.TERMINAL_SCHEMA:
        _require_exact_path(
            Path(path),
            COMPAT_TERMINAL_PATH,
            name="terminal_output",
        )
        if os.path.lexists(COMPAT_RECEIPT_PATH):
            raise PermissionError(
                "terminal receipt cannot coexist with a PASS receipt",
            )
    return _frozen_write_create_once_json(
        path,
        materialized,
        fingerprint_field=fingerprint_field,
    )


def create_authorization(
    authorization_path: Path,
    *,
    template_path: Path,
    python_path: Path,
    supervisor_path: Path,
    runtime_spec_path: Path,
    authorization_basis: str,
    instruction_id: str,
    validity_seconds: int = 300,
    runner=subprocess.run,
    manager_reader=None,
) -> dict[str, object]:
    verify_compatibility_identity()
    _require_authorize_paths(
        authorization_path=authorization_path,
        template_path=template_path,
        python_path=python_path,
        supervisor_path=supervisor_path,
        runtime_spec_path=runtime_spec_path,
    )
    _require_create_targets_absent()
    kwargs = {
        "template_path": template_path,
        "python_path": python_path,
        "supervisor_path": supervisor_path,
        "runtime_spec_path": runtime_spec_path,
        "authorization_basis": authorization_basis,
        "instruction_id": instruction_id,
        "validity_seconds": validity_seconds,
        "runner": runner,
    }
    if manager_reader is not None:
        kwargs["manager_reader"] = manager_reader
    result = _frozen_create_authorization(
        authorization_path,
        **kwargs,
    )
    _validate_compatibility_closure(result)
    return result


def validate_authorization(
    path: Path,
    *,
    runner,
    manager_reader,
    return_identity: bool = False,
):
    _require_exact_path(
        Path(path),
        COMPAT_AUTHORIZATION_PATH,
        name="authorization_input",
    )
    result = _frozen_validate_authorization(
        path,
        runner=runner,
        manager_reader=manager_reader,
        return_identity=return_identity,
    )
    authorization = result[0] if return_identity else result
    _validate_compatibility_closure(authorization)
    return result


def realize_actual_unit(
    authorization_path: Path,
    *,
    receipt_path: Path,
    terminal_path: Path,
    runner=subprocess.run,
    manager_reader=None,
) -> dict[str, object]:
    verify_compatibility_identity()
    _require_apply_paths(
        authorization_path=authorization_path,
        receipt_path=receipt_path,
        terminal_path=terminal_path,
    )
    _require_apply_targets_pristine()
    kwargs = {
        "receipt_path": receipt_path,
        "terminal_path": terminal_path,
        "runner": runner,
    }
    if manager_reader is not None:
        kwargs["manager_reader"] = manager_reader
    result = _frozen_realize_actual_unit(
        authorization_path,
        **kwargs,
    )
    authorization, identity = legacy._load_sealed_json_bound(
        authorization_path,
        "authorization_fingerprint",
    )
    _validate_compatibility_closure(
        authorization,
        require_fresh=False,
        require_bridge_fresh=False,
        require_future_absence=True,
    )
    _validate_receipt_transitive_binding(
        result,
        require_fresh=False,
        require_bridge_fresh=False,
        require_future_absence=True,
    )
    if (
        result.get("authorization_file_sha256")
        != identity.get("file_sha256")
    ):
        raise PermissionError("compatibility receipt authorization changed")
    return result


def _validate_live_archival_state(
    authorization: Mapping[str, object],
    receipt: Mapping[str, object],
    *,
    runner,
    manager_reader,
    allow_runtime_activation: bool,
) -> None:
    manager = manager_reader()
    legacy._validate_manager_generation(manager)
    if manager != authorization["manager_generation"]:
        raise PermissionError(
            "compatibility archival manager generation changed",
        )
    fragment = COMPAT_UNIT_DIRECTORY / COMPAT_UNIT
    path_policy = legacy._observe_unit_path_policy(
        runner=runner,
        allowed_fragment=fragment,
    )
    if path_policy != receipt["unit_path_policy"]:
        raise PermissionError(
            "compatibility archival unit path policy changed",
        )
    live_shadow = legacy.query_shadow(runner=runner)
    if not allow_runtime_activation:
        validated = legacy.validate_installed_shadow(
            live_shadow,
            fragment_identity=receipt["fragment_identity"],
            authorization=authorization,
        )
        if validated != receipt["full_static_shadow"]:
            raise PermissionError(
                "compatibility archival inactive shadow changed",
            )
        return
    if set(live_shadow) != set(legacy._SHADOW_PROPERTIES):
        raise PermissionError(
            "compatibility runtime shadow property closure changed",
        )
    expected = legacy._expected_static_shadow(fragment)
    for name, expected_value in expected.items():
        if (
            name not in _RUNTIME_DYNAMIC_SHADOW_FIELDS
            and live_shadow.get(name) != expected_value
        ):
            raise PermissionError(
                "compatibility runtime immutable shadow changed",
            )
    if live_shadow.get("ActiveState") not in {
        "inactive",
        "activating",
        "active",
        "deactivating",
        "failed",
    }:
        raise PermissionError("compatibility runtime active state is invalid")
    expected_exec = legacy._expected_exec(
        python_path=Path(
            authorization["executable_bindings"]["python"]["path"],
        ),
        supervisor_path=Path(
            authorization["executable_bindings"]["supervisor"]["path"],
        ),
        runtime_spec_path=COMPAT_RUNTIME_SPEC_PATH,
    )
    for directive, argv in expected_exec.items():
        normalized = legacy._normalize_exec(live_shadow[directive])
        if normalized != {
            "path": argv[0],
            "argv": argv,
            "ignore_errors": "no",
        }:
            raise PermissionError(
                f"compatibility runtime {directive} changed",
            )


def validate_archival_realization_chain(
    authorization_path: Path = COMPAT_AUTHORIZATION_PATH,
    receipt_path: Path = COMPAT_RECEIPT_PATH,
    *,
    runner=subprocess.run,
    manager_reader=None,
    allow_runtime_activation: bool = False,
) -> dict[str, object]:
    """Validate a completed chain after its short-lived windows have closed."""

    _require_exact_path(
        Path(authorization_path),
        COMPAT_AUTHORIZATION_PATH,
        name="archival_authorization",
    )
    _require_exact_path(
        Path(receipt_path),
        COMPAT_RECEIPT_PATH,
        name="archival_receipt",
    )
    authorization, authorization_identity = legacy._load_sealed_json_bound(
        authorization_path,
        "authorization_fingerprint",
    )
    receipt, receipt_identity = legacy._load_sealed_json_bound(
        receipt_path,
        "receipt_fingerprint",
    )
    closure = _validate_receipt_transitive_binding(
        receipt,
        require_fresh=False,
        require_bridge_fresh=False,
        require_future_absence=False,
    )
    if (
        receipt.get("authorization_file_sha256")
        != authorization_identity.get("file_sha256")
        or receipt.get("authorization_fingerprint")
        != authorization.get("authorization_fingerprint")
    ):
        raise PermissionError("archival realization chain is not exact PASS")
    if os.path.lexists(COMPAT_TERMINAL_PATH):
        raise PermissionError(
            "PASS realization cannot coexist with a terminal receipt",
        )
    if manager_reader is None:
        manager_reader = legacy.collect_manager_generation
    _validate_live_archival_state(
        authorization,
        receipt,
        runner=runner,
        manager_reader=manager_reader,
        allow_runtime_activation=allow_runtime_activation,
    )
    return {
        "authorization": authorization,
        "receipt": receipt,
        "compatibility_closure": closure,
        "authorization_identity": authorization_identity,
        "receipt_identity": receipt_identity,
    }


def _configure_isolated_namespace() -> None:
    """Apply the minimal deterministic compatibility patch after safe load."""

    legacy.ACTUAL_UNIT = COMPAT_UNIT
    # Frozen create_authorization therefore binds the actual executing wrapper.
    legacy.__file__ = str(COMPAT_REALIZER_PATH)
    legacy._expected_static_shadow = _compat_expected_static_shadow
    legacy._AUTH_KEYS = set(legacy._AUTH_KEYS) | {
        COMPATIBILITY_CLOSURE_KEY,
    }
    legacy.write_create_once_json = _compat_write_create_once_json
    legacy.create_authorization = create_authorization
    legacy.validate_authorization = validate_authorization
    legacy.realize_actual_unit = realize_actual_unit
    legacy.validate_archival_realization_chain = (
        validate_archival_realization_chain
    )


_configure_isolated_namespace()


def verify_compatibility_identity() -> dict[str, object]:
    compat_source = _require_running_source_generation()
    predecessor_source = _require_frozen_predecessor_generation()
    if (
        legacy.ACTUAL_UNIT != COMPAT_UNIT
        or legacy.__file__ != str(COMPAT_REALIZER_PATH)
        or legacy._expected_static_shadow is not _compat_expected_static_shadow
        or legacy.write_create_once_json is not _compat_write_create_once_json
        or legacy.validate_authorization is not validate_authorization
        or legacy.create_authorization is not create_authorization
        or legacy.realize_actual_unit is not realize_actual_unit
        or legacy.validate_archival_realization_chain
        is not validate_archival_realization_chain
        or COMPATIBILITY_CLOSURE_KEY not in legacy._AUTH_KEYS
        or PROTECTED_PREDECESSOR_UNIT == COMPAT_UNIT
    ):
        raise PermissionError("compatibility realizer isolated patch changed")
    return {
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_generation": "c1",
        "unit_name": COMPAT_UNIT,
        "runtime_spec_path": str(COMPAT_RUNTIME_SPEC_PATH),
        "authorization_path": str(COMPAT_AUTHORIZATION_PATH),
        "receipt_path": str(COMPAT_RECEIPT_PATH),
        "terminal_path": str(COMPAT_TERMINAL_PATH),
        "template_path": str(COMPAT_TEMPLATE_PATH),
        "supervisor_path": str(COMPAT_SUPERVISOR_PATH),
        "bridge_validator_path": str(COMPAT_BRIDGE_SOURCE_PATH),
        "bridge_validator_file_sha256": COMPAT_BRIDGE_SOURCE_SHA256,
        "bridge_compat_authorization_path": str(
            COMPAT_BRIDGE_AUTHORIZATION_PATH,
        ),
        "frozen_realizer_path": str(FROZEN_REALIZER_PATH),
        "frozen_realizer_file_sha256": FROZEN_REALIZER_SHA256,
        "frozen_realizer_generation": predecessor_source,
        "compat_realizer_generation": compat_source,
        "success_exit_status_authorized_value": "0",
        "new_static_fragment_install_authorized": True,
        "daemon_reload_authorized": True,
        "predecessor_unit_mutation_authorized": False,
        "enable_authorized": False,
        "start_authorized": False,
        "stop_authorized": False,
        "remove_authorized": False,
        "automatic_retry_authorized": False,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


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
            args.output,
            template_path=args.template,
            python_path=args.python,
            supervisor_path=args.supervisor,
            runtime_spec_path=args.runtime_spec,
            authorization_basis=args.authorization_basis,
            instruction_id=args.instruction_id,
            validity_seconds=args.validity_seconds,
        )
        return 0
    realize_actual_unit(
        args.authorization,
        receipt_path=args.receipt,
        terminal_path=args.terminal_receipt,
    )
    return 0


def __getattr__(name: str):
    return getattr(legacy, name)


if __name__ == "__main__":
    raise SystemExit(main())
