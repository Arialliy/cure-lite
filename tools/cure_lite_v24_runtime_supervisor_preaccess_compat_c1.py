#!/usr/bin/env python3
"""Compatibility-c1 supervisor for the frozen v24 r2 runtime.

The frozen supervisor remains the implementation authority.  This module
executes its hash-bound source in an isolated module namespace and changes
only the actual-runtime path identities plus the one preaccess audit-schema
comparison that was encoded with a fictional r2-only schema.  The original
scientific audit is still read and verified directly; the compatibility
receipt is policy evidence, never a surrogate audit.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    # The production unit intentionally executes Python with -I -S.  Import
    # only this exact repository root so the sealed compatibility-policy
    # verifier remains reachable in that isolated interpreter.
    sys.path.insert(0, str(REPOSITORY))
FROZEN_SUPERVISOR_PATH = (
    REPOSITORY / "tools/cure_lite_v24_runtime_supervisor.py"
).resolve()
FROZEN_SUPERVISOR_SHA256 = (
    "b955ba8ffe869d324cc9319f8031180989746053d7ceec5e50bd12eb19faeeed"
)

COMPATIBILITY_GENERATION = "c1"
AUTHORITATIVE_ACCESS_AUDIT_SCHEMA = (
    "cure-lite-v24-split-access-audit-v1"
)
FICTIONAL_ACCESS_AUDIT_SCHEMA = (
    "cure-lite-v24-split-access-audit-r2-v1"
)
COMPAT_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service"
)
COMPAT_RUNTIME_SPEC_PATH = (
    "/home/md0/ly/cure_lite/protocols/IRSTD-1K/gcr_pacre_v24/"
    "runtime_evidence_r2/"
    "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json"
)
COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    "/home/md0/ly/cure_lite/protocols/IRSTD-1K/gcr_pacre_v24/"
    "runtime_evidence_r2/"
    "D_R_structural_attempt_r2_preaccess_compat_c1_"
    "runtime_launch_authorization.json"
)
COMPAT_ADAPTER_PATH = (
    "/home/md0/ly/cure_lite/tools/"
    "run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c1.py"
)
COMPAT_SUPERVISOR_PATH = str(Path(__file__).resolve())
COMPATIBILITY_RECEIPT_PATH = (
    "/home/md0/ly/cure_lite/protocols/IRSTD-1K/gcr_pacre_v24/"
    "runtime_evidence_r2/"
    "r2_preaccess_schema_compat_c1_receipt.json"
)
COMPAT_POLICY_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility.py"
).resolve()
COMPAT_POLICY_SOURCE_SHA256 = (
    "fe715af48867f166d2e15727e0190844cfd79fb5c02fa5a440d294bb7f29e084"
)


_FILE_IDENTITY_FIELDS = (
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
_PARENT_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
    "st_nlink",
)


def _stable_source_bytes(
    path: Path,
) -> tuple[bytes, dict[str, int]]:
    target = Path(path).absolute()
    parent = target.parent
    parent_before = parent.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or parent.resolve(strict=True) != parent
        or target.resolve(strict=True) != target
    ):
        raise PermissionError("frozen runtime supervisor source is unsafe")
    directory_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        parent_opened = os.fstat(directory_fd)
        before = os.stat(
            target.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
        ):
            raise PermissionError(
                "frozen runtime supervisor source is unsafe"
            )
        descriptor = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
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
        finally:
            os.close(descriptor)
        path_after = os.stat(
            target.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        parent_after = os.fstat(directory_fd)
    finally:
        os.close(directory_fd)
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(after, field)
        or getattr(after, field) != getattr(path_after, field)
        for field in _FILE_IDENTITY_FIELDS
    ) or any(
        getattr(parent_before, field)
        != getattr(parent_opened, field)
        or getattr(parent_opened, field)
        != getattr(parent_after, field)
        for field in _PARENT_IDENTITY_FIELDS
    ):
        raise PermissionError(
            "frozen runtime supervisor source changed while reading"
        )
    identity = {
        field: int(getattr(path_after, field))
        for field in _FILE_IDENTITY_FIELDS
    }
    identity.update({
        f"parent_{field}": int(getattr(parent_after, field))
        for field in _PARENT_IDENTITY_FIELDS
    })
    return b"".join(chunks), identity


def _verified_source_bytes(
    path: Path,
    expected_sha256: str,
) -> tuple[bytes, dict[str, int]]:
    """Read, hash, and retain one exact regular source generation."""

    raw, identity = _stable_source_bytes(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError("frozen runtime supervisor source changed")
    return raw, identity


def _verify_frozen_supervisor_source() -> str:
    _raw, identity = _verified_source_bytes(
        FROZEN_SUPERVISOR_PATH,
        FROZEN_SUPERVISOR_SHA256,
    )
    if identity != _FROZEN_SUPERVISOR_LOAD_IDENTITY:
        raise PermissionError(
            "frozen runtime supervisor generation was replaced"
        )
    return FROZEN_SUPERVISOR_SHA256


def _load_frozen_supervisor() -> tuple[ModuleType, dict[str, int]]:
    raw, identity = _verified_source_bytes(
        FROZEN_SUPERVISOR_PATH,
        FROZEN_SUPERVISOR_SHA256,
    )
    name = (
        "tools._cure_lite_v24_runtime_supervisor_frozen_"
        "for_preaccess_compat_c1"
    )
    module = ModuleType(name)
    module.__file__ = str(FROZEN_SUPERVISOR_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                raw,
                str(FROZEN_SUPERVISOR_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, identity


legacy, _FROZEN_SUPERVISOR_LOAD_IDENTITY = _load_frozen_supervisor()
_frozen_validate_spec_structure = legacy._validate_spec_structure


def _configure_frozen_supervisor() -> None:
    """Apply the isolated compatibility runtime identity."""

    legacy.__file__ = COMPAT_SUPERVISOR_PATH
    legacy._ACTUAL_UNIT_NAME = COMPAT_UNIT_NAME
    legacy._ACTUAL_SPEC_PATH = COMPAT_RUNTIME_SPEC_PATH
    legacy._ACTUAL_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
        COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
    )
    legacy._ACTUAL_ADAPTER_PATH = COMPAT_ADAPTER_PATH
    legacy._validate_spec_structure = _validate_spec_structure


def _load_verified_compatibility_policy() -> tuple[
    ModuleType,
    dict[str, int],
]:
    raw, identity = _stable_source_bytes(COMPAT_POLICY_SOURCE_PATH)
    if hashlib.sha256(raw).hexdigest() != COMPAT_POLICY_SOURCE_SHA256:
        raise PermissionError(
            "preaccess compatibility policy source changed"
        )
    name = (
        "tools._cure_lite_v24_preaccess_schema_compatibility_"
        "verified_for_runtime_supervisor_c1"
    )
    module = ModuleType(name)
    module.__file__ = str(COMPAT_POLICY_SOURCE_PATH)
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(COMPAT_POLICY_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    _raw_after, identity_after = _stable_source_bytes(
        COMPAT_POLICY_SOURCE_PATH
    )
    if identity_after != identity:
        raise PermissionError(
            "preaccess compatibility policy generation changed"
        )
    return module, identity


def _validate_policy_receipt_contract(
    result: object,
    *,
    policy: ModuleType,
    policy_generation: Mapping[str, int],
) -> Mapping[str, object]:
    if not isinstance(result, Mapping):
        raise PermissionError(
            "preaccess compatibility receipt verifier returned no evidence"
        )
    receipt = result
    source_roots = receipt.get("compatibility_source_roots")
    schema_compatibility = receipt.get("schema_compatibility")
    if (
        not isinstance(source_roots, Mapping)
        or not isinstance(schema_compatibility, Mapping)
        or schema_compatibility.get("producer_schema")
        != AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        or schema_compatibility.get(
            "scientific_authorization_bound_schema"
        )
        != AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        or schema_compatibility.get(
            "compatibility_consumer_required_schema"
        )
        != AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        or schema_compatibility.get(
            "buggy_frozen_consumer_expected_schema"
        )
        != FICTIONAL_ACCESS_AUDIT_SCHEMA
        or schema_compatibility.get("accept_either_schema") is not False
    ):
        raise PermissionError(
            "preaccess compatibility receipt schema contract changed"
        )
    expected_sources = {
        "compat_policy": COMPAT_POLICY_SOURCE_PATH,
        "compat_supervisor": Path(COMPAT_SUPERVISOR_PATH),
        "compat_adapter": Path(COMPAT_ADAPTER_PATH),
    }
    for label, expected_path in expected_sources.items():
        root = source_roots.get(label)
        if (
            not isinstance(root, Mapping)
            or root.get("path") != str(expected_path)
            or root.get("file_sha256")
            != legacy.file_sha256(expected_path)
        ):
            raise PermissionError(
                f"preaccess compatibility receipt {label} changed"
            )
    _raw_after, policy_generation_after = _stable_source_bytes(
        COMPAT_POLICY_SOURCE_PATH
    )
    if policy_generation_after != policy_generation:
        raise PermissionError(
            "preaccess compatibility policy generation changed"
        )
    for old_path in (
        policy.OLD_RUNTIME_SPEC_PATH,
        policy.OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
    ):
        if os.path.lexists(old_path):
            raise PermissionError(
                "frozen failed-generation runtime path is no longer absent"
            )
    return receipt


def _verify_policy_compatibility_receipt(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    """Delegate the sealed policy receipt to its sole owning module."""

    policy, policy_generation = _load_verified_compatibility_policy()
    policy_path = getattr(policy, "COMPAT_RECEIPT_PATH", None)
    if (
        policy_path is None
        or Path(os.path.abspath(policy_path))
        != Path(COMPATIBILITY_RECEIPT_PATH)
    ):
        raise PermissionError(
            "preaccess compatibility receipt path interface changed"
        )
    verifier = getattr(policy, "verify_compatibility_receipt", None)
    if not callable(verifier):
        raise PermissionError(
            "preaccess compatibility receipt verifier is unavailable"
        )
    result = verifier(
        Path(COMPATIBILITY_RECEIPT_PATH),
        expected_spec=dict(payload),
        require_spec_binding=True,
        allow_runtime_activation=True,
    )
    return _validate_policy_receipt_contract(
        result,
        policy=policy,
        policy_generation=policy_generation,
    )


def _verify_policy_compatibility_prewrite(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    policy, policy_generation = _load_verified_compatibility_policy()
    verifier = getattr(
        policy,
        "verify_compatibility_prewrite_spec",
        None,
    )
    if not callable(verifier):
        raise PermissionError(
            "preaccess compatibility prewrite verifier is unavailable"
        )
    result = verifier(
        Path(COMPATIBILITY_RECEIPT_PATH),
        dict(payload),
    )
    return _validate_policy_receipt_contract(
        result,
        policy=policy,
        policy_generation=policy_generation,
    )


def _validate_actual_identity_before_projection(
    payload: dict[str, object],
) -> None:
    """Validate the authentic spec seal and the authoritative v1 schema."""

    body = dict(payload)
    fingerprint = body.pop("runtime_spec_fingerprint", None)
    preaccess = payload.get("scientific_preaccess")
    if (
        payload.get("schema_version") != legacy.RUNTIME_SPEC_SCHEMA
        or not legacy._is_sha256(fingerprint)
        or fingerprint != legacy.stable_fingerprint(body)
        or not isinstance(preaccess, Mapping)
        or preaccess.get("access_audit_required_schema")
        != AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
    ):
        raise ValueError(
            "compatibility actual runtime requires the sealed original "
            "v1 access-audit schema"
        )
    if (
        preaccess.get("access_audit_required_schema")
        == FICTIONAL_ACCESS_AUDIT_SCHEMA
    ):
        raise ValueError("fictional r2 access-audit schema is forbidden")


def _validate_actual_spec_structure_core(
    payload: dict[str, object],
    *,
    loaded_spec_path: Path,
) -> None:
    if Path(os.path.abspath(loaded_spec_path)) != Path(
        COMPAT_RUNTIME_SPEC_PATH
    ):
        raise ValueError("compatibility actual runtime spec path is not exact")
    _validate_actual_identity_before_projection(payload)

    projected = deepcopy(payload)
    preaccess = projected.get("scientific_preaccess")
    if not isinstance(preaccess, dict):
        raise ValueError("scientific preaccess contract is malformed")
    preaccess["access_audit_required_schema"] = (
        FICTIONAL_ACCESS_AUDIT_SCHEMA
    )
    projected_body = dict(projected)
    projected_body.pop("runtime_spec_fingerprint", None)
    projected["runtime_spec_fingerprint"] = legacy.stable_fingerprint(
        projected_body
    )
    _frozen_validate_spec_structure(
        projected,
        loaded_spec_path=loaded_spec_path,
    )


def _validate_spec_structure(
    payload: dict[str, object],
    *,
    loaded_spec_path: Path,
) -> None:
    """Retain every frozen check while projecting its one mistaken literal."""

    _verify_frozen_supervisor_source()
    _configure_frozen_supervisor()
    if payload.get("execution_kind") != legacy.ACTUAL_EXECUTION_KIND:
        _frozen_validate_spec_structure(
            payload,
            loaded_spec_path=loaded_spec_path,
        )
        return
    _validate_actual_spec_structure_core(
        payload,
        loaded_spec_path=loaded_spec_path,
    )
    _verify_policy_compatibility_receipt(payload)


def validate_prewrite_spec(payload: Mapping[str, object]) -> None:
    """Validate the exact producer view before any runtime path is created."""

    _verify_frozen_supervisor_source()
    _configure_frozen_supervisor()
    materialized = dict(payload)
    if materialized.get("execution_kind") != legacy.ACTUAL_EXECUTION_KIND:
        raise ValueError(
            "compatibility prewrite preview must be actual D_R"
        )
    _validate_actual_spec_structure_core(
        materialized,
        loaded_spec_path=Path(COMPAT_RUNTIME_SPEC_PATH),
    )
    _verify_policy_compatibility_prewrite(materialized)


def verify_compatibility_identity() -> dict[str, object]:
    """Return the non-mutating identity of this runtime compatibility layer."""

    _verify_frozen_supervisor_source()
    _configure_frozen_supervisor()
    if (
        legacy.__file__ != COMPAT_SUPERVISOR_PATH
        or legacy._ACTUAL_UNIT_NAME != COMPAT_UNIT_NAME
        or legacy._ACTUAL_SPEC_PATH != COMPAT_RUNTIME_SPEC_PATH
        or legacy._ACTUAL_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        != COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        or legacy._ACTUAL_ADAPTER_PATH != COMPAT_ADAPTER_PATH
        or legacy._validate_spec_structure is not _validate_spec_structure
    ):
        raise PermissionError("compatibility supervisor identity changed")
    return {
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_generation": COMPATIBILITY_GENERATION,
        "unit_name": COMPAT_UNIT_NAME,
        "runtime_spec_path": COMPAT_RUNTIME_SPEC_PATH,
        "runtime_launch_authorization_path": (
            COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "supervisor_path": COMPAT_SUPERVISOR_PATH,
        "adapter_path": COMPAT_ADAPTER_PATH,
        "compatibility_receipt_path": COMPATIBILITY_RECEIPT_PATH,
        "authoritative_access_audit_schema": (
            AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        ),
        "fictional_access_audit_schema_accepted": False,
        "frozen_supervisor_path": str(FROZEN_SUPERVISOR_PATH),
        "frozen_supervisor_file_sha256": FROZEN_SUPERVISOR_SHA256,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def verify_child_runtime_attestation(
    attestation_path: str | Path,
    runtime_launch_authorization_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
    cgroup_path: str | None = None,
) -> dict[str, object]:
    """Verify an attestation through the compatibility supervisor."""

    _verify_frozen_supervisor_source()
    _configure_frozen_supervisor()
    return legacy.verify_child_runtime_attestation(
        attestation_path,
        runtime_launch_authorization_path,
        environment=environment,
        argv=argv,
        cgroup_path=cgroup_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    _verify_frozen_supervisor_source()
    _configure_frozen_supervisor()
    return int(legacy.main(argv))


def __getattr__(name: str):
    return getattr(legacy, name)


_configure_frozen_supervisor()


if __name__ == "__main__":
    raise SystemExit(main())
