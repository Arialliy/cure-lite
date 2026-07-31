#!/usr/bin/env python3
"""Fresh runtime-environment scope handoff for compatibility generation c3.

This module does not reinterpret the frozen precleanup receipt as evidence
about the c3 unit.  It first replays that receipt with its original target and
``require_target_ready=False``.  It then creates a new contract by replacing
only those two fields and delegates policy construction and stability sampling
to the hash-pinned runtime-environment implementation.

The production entry point remains fail-closed until the c3 realization bridge
has been frozen and ``C3_BRIDGE_SHA256`` has been replaced.  Pure in-memory
functions accept an injected archival validator for unit tests.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import MappingProxyType, ModuleType
from typing import Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()

FROZEN_ENVIRONMENT_PATH = (
    REPOSITORY / "tools/cure_lite_v24_runtime_environment.py"
).resolve()
FROZEN_ENVIRONMENT_SHA256 = (
    "a40465786ce3537346372df5991bb6788d44feddfd497ec83a1dc302fb8b2fea"
)

OLD_TARGET_UNIT = "cure-lite-v24-gcr-pacre-dr-r2.service"
C3_TARGET_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c3.service"
)
C3_FRAGMENT_PATH = (
    Path(f"/run/user/{os.getuid()}/systemd/user") / C3_TARGET_UNIT
)

PRECLEANUP_PATH = (
    EVIDENCE_ROOT / "runtime_environment_precleanup_receipt.json"
)
CLEANUP_RECEIPT_PATH = (
    EVIDENCE_ROOT / "environment_cleanup_recovery_r1/cleanup-receipt.json"
)
C3_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c3.json"
)
C3_STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c3.json"
)
C3_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c3.json"
)
C3_REALIZATION_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c3_unit_realization_authorization.json"
)
C3_REALIZATION_RECEIPT_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c3_unit_realization_receipt.json"
)
C3_BRIDGE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c3.py"
).resolve()
C3_BRIDGE_SHA256 = (
    "cdbbe4355b29519d2b3da858732bc8531396a59f5a3f1cfacdb578323fe33de1"
)

SELECTED_GPU_INDEX = 0
CONFLICT_UNIT_IDS = (
    "confa-v41-mshnet-nudt-clean-formal-20260718-v1.service",
)
DEPENDENCY_UNIT_IDS: tuple[str, ...] = ()
ALLOWED_FAILED_UNIT_IDS = (
    "sctransnet-formal800-gpu2-recovery-postprocess-s42-v1.service",
    "sctransnet-formal800-gpu2-recovery-s42-v1.service",
    "snap.firmware-updater.firmware-notifier.service",
)
ALLOWED_UNIT_IDS = CONFLICT_UNIT_IDS
ALLOWED_MANAGER_STATES = ("running", "degraded")
SAMPLE_COUNT = 2
SAMPLE_INTERVAL_SECONDS = 30.0


def _require_c3_namespace() -> None:
    evidence_paths = (
        C3_POLICY_PATH,
        C3_STABILITY_PATH,
        C3_POSTCLEANUP_PATH,
        C3_REALIZATION_AUTHORIZATION_PATH,
        C3_REALIZATION_RECEIPT_PATH,
    )
    if (
        "preaccess-compat-c3" not in C3_TARGET_UNIT
        or C3_FRAGMENT_PATH.name != C3_TARGET_UNIT
        or len(set(evidence_paths)) != len(evidence_paths)
        or any("compat_c3" not in path.name for path in evidence_paths)
        or "compat_c3" not in C3_BRIDGE_PATH.name
        or any("compat_c2" in str(path) for path in evidence_paths)
    ):
        raise PermissionError("c3 environment namespace is not exact")

_HANDOFF_FIELDS = frozenset({"target_unit_id", "require_target_ready"})
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
_PARENT_GUARD_IDENTITY_FIELDS = _PARENT_IDENTITY_FIELDS + (
    "st_mtime_ns",
    "st_ctime_ns",
)

ArchivalValidator = Callable[
    [Path, Path],
    Mapping[str, object],
]


def _stable_source_bytes(
    path: Path,
) -> tuple[bytes, dict[str, int]]:
    """Read one exact regular-file generation through a no-follow fd."""

    target = Path(path).absolute()
    parent = target.parent
    parent_before = parent.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or parent.resolve(strict=True) != parent
        or target.resolve(strict=True) != target
    ):
        raise PermissionError("runtime environment source path is unsafe")
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
            raise PermissionError("runtime environment source is unsafe")
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
        raise PermissionError("runtime environment source generation changed")
    raw = b"".join(chunks)
    identity = {
        field: int(getattr(path_after, field))
        for field in _FILE_IDENTITY_FIELDS
    }
    identity.update(
        {
            f"parent_{field}": int(getattr(parent_after, field))
            for field in _PARENT_IDENTITY_FIELDS
        }
    )
    return raw, identity


def _load_frozen_environment() -> tuple[ModuleType, dict[str, int]]:
    raw, identity = _stable_source_bytes(FROZEN_ENVIRONMENT_PATH)
    if hashlib.sha256(raw).hexdigest() != FROZEN_ENVIRONMENT_SHA256:
        raise PermissionError("frozen runtime environment source changed")
    name = (
        "tools._cure_lite_v24_runtime_environment_frozen_"
        "for_preaccess_compat_c3"
    )
    module = ModuleType(name)
    module.__file__ = str(FROZEN_ENVIRONMENT_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                raw,
                str(FROZEN_ENVIRONMENT_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    raw_after, identity_after = _stable_source_bytes(
        FROZEN_ENVIRONMENT_PATH
    )
    if (
        hashlib.sha256(raw_after).hexdigest()
        != FROZEN_ENVIRONMENT_SHA256
        or identity_after != identity
    ):
        sys.modules.pop(name, None)
        raise PermissionError(
            "frozen runtime environment generation changed while loading"
        )
    sys.modules.pop(name, None)
    return module, identity


_loaded_frozen, _FROZEN_LOAD_IDENTITY = _load_frozen_environment()

_FROZEN_READ_API = frozenset(
    {
        "ENVIRONMENT_INVENTORY_SCHEMA",
        "ENVIRONMENT_RECEIPT_SCHEMA",
        "GPU_DOUBLE_SNAPSHOT_SCHEMA",
        "RECOVERY_CLEANUP_MODE",
        "RECOVERY_GUARD_MODE",
        "RECOVERY_QUIESCENCE_MODE",
        "EnvironmentAuditContract",
        "_deep_exact_equal",
        "audit_environment_once",
        "build_environment_policy",
        "canonical_json",
        "current_runtime_toolchain_binding",
        "evaluate_environment_stability",
        "load_sealed_receipt",
        "load_sealed_receipt_with_evidence",
        "prepare_environment_stability_contract",
        "run_environment_stability_gate",
        "stable_fingerprint",
        "utc_now",
        "validate_environment_audit_contract",
        "validate_environment_policy",
        "validate_environment_stability_receipt",
        "verify_sealed_receipt_evidence",
    }
)
_FROZEN_MUTATION_API = frozenset(
    {
        "_audit_only",
        "_create_policy",
        "_rename_noreplace",
        "_stability_gate",
        "_write_all_to_fd",
        "acquire_gpu_lease",
        "build_parser",
        "main",
        "release_gpu_lease_to_tombstone",
        "write_create_once_receipt",
        "write_environment_policy",
    }
)


def _project_frozen_read_generation(
    module: ModuleType,
) -> tuple[
    MappingProxyType,
    MappingProxyType,
    MappingProxyType,
]:
    missing = sorted(
        name for name in _FROZEN_READ_API if name not in module.__dict__
    )
    mutation_missing = sorted(
        name for name in _FROZEN_MUTATION_API if name not in module.__dict__
    )
    if missing or mutation_missing:
        raise PermissionError(
            "frozen runtime environment API generation changed"
        )
    for name in _FROZEN_MUTATION_API:
        candidate = module.__dict__.pop(name)
        if not callable(candidate):
            raise PermissionError(
                f"frozen mutation API is not callable:{name}"
            )
    projection = {
        name: module.__dict__[name]
        for name in sorted(_FROZEN_READ_API)
    }
    all_callables = {
        name: value
        for name, value in module.__dict__.items()
        if callable(value)
    }
    generation = {
        name: (
            id(value),
            id(getattr(value, "__code__", None)),
            repr(getattr(value, "__defaults__", None)),
            repr(getattr(value, "__kwdefaults__", None)),
        )
        for name, value in all_callables.items()
    }
    for name, candidate in all_callables.items():
        globals_value = getattr(candidate, "__globals__", None)
        if isinstance(globals_value, dict) and any(
            mutation_name in globals_value
            for mutation_name in _FROZEN_MUTATION_API
        ):
            raise PermissionError(
                f"frozen callable retains mutation API:{name}"
            )
    return (
        MappingProxyType(projection),
        MappingProxyType(all_callables),
        MappingProxyType(generation),
    )


class _FrozenReadFacade:
    """Immutable strict allowlist over one hash-pinned read generation."""

    __slots__ = ("__projection",)

    def __init__(self, projection: Mapping[str, object]) -> None:
        object.__setattr__(
            self,
            "_FrozenReadFacade__projection",
            MappingProxyType(dict(projection)),
        )

    def __getattribute__(self, name: str):
        if name in {"__class__", "__dir__"}:
            return object.__getattribute__(self, name)
        projection = object.__getattribute__(
            self,
            "_FrozenReadFacade__projection",
        )
        try:
            return projection[name]
        except KeyError as error:
            raise AttributeError(
                f"frozen read API is unavailable:{name}"
            ) from error

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("frozen read facade is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("frozen read facade is immutable")

    def __dir__(self) -> list[str]:
        projection = object.__getattribute__(
            self,
            "_FrozenReadFacade__projection",
        )
        return sorted(projection)


(
    _FROZEN_PROJECTION,
    _FROZEN_ALL_CALLABLES,
    _FROZEN_CALLABLE_GENERATION,
) = (
    _project_frozen_read_generation(_loaded_frozen)
)
frozen = _FrozenReadFacade(_FROZEN_PROJECTION)
_frozen = frozen

def _bind_frozen_facade_guard(
    expected_facade: object,
    expected_projection: object,
) -> Callable[[], None]:
    def require() -> None:
        if (
            frozen is not expected_facade
            or _frozen is not expected_facade
            or object.__getattribute__(
                expected_facade,
                "_FrozenReadFacade__projection",
            )
            is not expected_projection
        ):
            raise PermissionError(
                "frozen read facade generation changed"
            )

    return require


_require_frozen_facade_identity = _bind_frozen_facade_guard(
    frozen,
    object.__getattribute__(
        frozen,
        "_FrozenReadFacade__projection",
    ),
)
del _bind_frozen_facade_guard
del _project_frozen_read_generation
del _FrozenReadFacade


def _require_frozen_environment_source() -> None:
    _require_frozen_facade_identity()
    raw, identity = _stable_source_bytes(FROZEN_ENVIRONMENT_PATH)
    if (
        hashlib.sha256(raw).hexdigest()
        != FROZEN_ENVIRONMENT_SHA256
        or identity != _FROZEN_LOAD_IDENTITY
    ):
        raise PermissionError(
            "frozen runtime environment generation was replaced"
        )
    _require_frozen_callable_generation()
    _require_frozen_facade_identity()


def _require_frozen_callable_generation() -> None:
    if (
        set(_FROZEN_PROJECTION) != _FROZEN_READ_API
        or set(_FROZEN_ALL_CALLABLES)
        != set(_FROZEN_CALLABLE_GENERATION)
        or any(
            name in _FROZEN_ALL_CALLABLES
            for name in _FROZEN_MUTATION_API
        )
    ):
        raise PermissionError("frozen read projection generation changed")
    for name, observed in _FROZEN_ALL_CALLABLES.items():
        expected = _FROZEN_CALLABLE_GENERATION[name]
        if (
            id(observed) != expected[0]
            or id(getattr(observed, "__code__", None)) != expected[1]
            or repr(getattr(observed, "__defaults__", None))
            != expected[2]
            or repr(getattr(observed, "__kwdefaults__", None))
            != expected[3]
        ):
            raise PermissionError(
                f"frozen callable generation changed:{name}"
            )
        globals_value = getattr(observed, "__globals__", None)
        if not isinstance(globals_value, dict):
            continue
        if any(
            mutation_name in globals_value
            for mutation_name in _FROZEN_MUTATION_API
        ):
            raise PermissionError(
                f"frozen callable regained mutation API:{name}"
            )
        for dependency_name, dependency in (
            _FROZEN_ALL_CALLABLES.items()
        ):
            if (
                dependency_name in globals_value
                and globals_value[dependency_name] is not dependency
            ):
                raise PermissionError(
                    "frozen callable dependency changed:"
                    f"{name}:{dependency_name}"
                )
        for dependency_name, dependency in _FROZEN_PROJECTION.items():
            if (
                callable(dependency)
                or dependency_name not in globals_value
            ):
                continue
            if globals_value[dependency_name] != dependency:
                raise PermissionError(
                    "frozen constant dependency changed:"
                    f"{name}:{dependency_name}"
                )


def _no_payload(value: Mapping[str, object], *, name: str) -> None:
    boolean_controls = frozenset(
        {
            "cuda_context_created",
            "gpu_accessed",
            "gpu_compute_authorized",
            "gpu_compute_performed",
            "gpu_compute_started",
            "gpu_payload_accessed",
            "materialization_authorized",
            "materialization_consumed",
            "optimizer_constructed",
            "optimizer_module_referenced",
            "training_authorized",
            "training_performed",
            "training_started",
        }
    )
    numeric_controls = frozenset(
        {
            "epochs_completed",
            "gpu_kernel_launches",
            "optimizer_steps",
            "parameter_updates",
            "training_steps",
        }
    )
    strict_none_authorities = frozenset(
        {
            "gpu_authority",
            "payload_authority",
            "training_authority",
        }
    )
    protected_resource_tokens = (
        "cache",
        "checkpoint",
        "real_data",
        "real_dataset",
    )
    permission_suffixes = (
        "_accessed",
        "_authorized",
        "_consumed",
        "_loaded",
        "_materialized",
    )
    if any(
        value.get(field) is not False
        for field in (
            "D_R_payload_accessed",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
        )
    ):
        raise PermissionError(f"{name} accessed payload or training")

    def _visit(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                lowered = key.lower()
                child_path = f"{path}.{key}"
                payload_permission = (
                    lowered
                    in {
                        "payload_accessed",
                        "payload_authorized",
                        "payload_access_authorized",
                    }
                    or lowered.endswith("_payload_accessed")
                    or lowered.endswith("_payload_authorized")
                    or lowered.endswith("_payload_access_authorized")
                )
                dataset_permission = (
                    lowered.startswith(("d_r_", "d_v_", "d_t_"))
                    and lowered.endswith(("_accessed", "_authorized"))
                )
                protected_resource_permission = (
                    any(
                        token in lowered
                        for token in protected_resource_tokens
                    )
                    and lowered.endswith(permission_suffixes)
                )
                invalid = (
                    lowered in boolean_controls
                    and child is not False
                    or payload_permission
                    and child is not False
                    or dataset_permission
                    and child is not False
                    or protected_resource_permission
                    and child is not False
                    or lowered in numeric_controls
                    and (type(child) is not int or child != 0)
                    or lowered in strict_none_authorities
                    and child not in (None, "none")
                    or lowered == "mutation_authority"
                    and not isinstance(child, Mapping)
                    and child not in (None, "none")
                )
                if invalid:
                    raise PermissionError(
                        f"{child_path} accessed payload or training"
                    )
                _visit(child, child_path)
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                _visit(child, f"{path}[{index}]")

    _visit(value, name)


def _strict_utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} is not a strict UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} is malformed") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(
        parsed
    ):
        raise ValueError(f"{name} is not UTC")
    return parsed


_SEALED_EVIDENCE_COMMON_FIELDS = frozenset(
    {
        "path",
        "device",
        "inode",
        "size",
        "mtime_ns",
        "file_sha256",
    }
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_sealed_root(
    evidence: Mapping[str, object],
    *,
    path: Path,
    fingerprint_field: str,
) -> dict[str, object]:
    root = dict(evidence)
    expected_fields = _SEALED_EVIDENCE_COMMON_FIELDS | {
        fingerprint_field
    }
    if (
        set(root) != expected_fields
        or Path(str(root.get("path", ""))).absolute()
        != path.absolute()
        or not _is_sha256(root.get("file_sha256"))
        or not _is_sha256(root.get(fingerprint_field))
        or any(
            type(root.get(field)) is not int
            or root[field] < 0
            for field in ("device", "inode", "size", "mtime_ns")
        )
    ):
        raise PermissionError("sealed live evidence root changed")
    return root


def _require_live_sealed_metadata(
    path: Path,
    evidence: Mapping[str, object],
) -> None:
    target = path.absolute()
    metadata = target.lstat()
    if (
        target.resolve(strict=True) != target
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_dev != evidence["device"]
        or metadata.st_ino != evidence["inode"]
        or metadata.st_size != evidence["size"]
        or metadata.st_mtime_ns != evidence["mtime_ns"]
    ):
        raise PermissionError("sealed live evidence metadata changed")


def _read_live_sealed(
    path: Path,
    fingerprint_field: str,
) -> tuple[dict[str, object], dict[str, object]]:
    _require_frozen_environment_source()
    payload, evidence = _frozen.load_sealed_receipt_with_evidence(
        path,
        fingerprint_field=fingerprint_field,
    )
    root = _validate_sealed_root(
        evidence,
        path=path,
        fingerprint_field=fingerprint_field,
    )
    _require_live_sealed_metadata(path, root)
    _no_payload(payload, name=f"sealed live evidence:{path.name}")
    return dict(payload), root


def _verify_live_sealed(
    path: Path,
    evidence: Mapping[str, object],
    fingerprint_field: str,
) -> None:
    _require_frozen_environment_source()
    root = _validate_sealed_root(
        evidence,
        path=path,
        fingerprint_field=fingerprint_field,
    )
    _frozen.verify_sealed_receipt_evidence(
        path,
        root,
        fingerprint_field=fingerprint_field,
    )
    _require_live_sealed_metadata(path, root)


_LIVE_ROOT_SPECS = MappingProxyType(
    {
        "policy": (C3_POLICY_PATH, "policy_fingerprint"),
        "stability": (
            C3_STABILITY_PATH,
            "stability_receipt_fingerprint",
        ),
    }
)


def _verify_exact_live_roots(
    live_roots: Mapping[str, Mapping[str, object]] | None,
    *,
    required: frozenset[str],
) -> dict[str, dict[str, object]]:
    if live_roots is None:
        if required:
            raise PermissionError("required c3 live roots are absent")
        return {}
    if set(live_roots) != required:
        raise PermissionError("c3 live-root lane changed")
    roots: dict[str, dict[str, object]] = {}
    for name in sorted(required):
        try:
            path, fingerprint_field = _LIVE_ROOT_SPECS[name]
            supplied = live_roots[name]
        except (KeyError, TypeError) as error:
            raise PermissionError(
                "c3 live-root lane is malformed"
            ) from error
        root = _validate_sealed_root(
            supplied,
            path=path,
            fingerprint_field=fingerprint_field,
        )
        _verify_live_sealed(path, root, fingerprint_field)
        roots[name] = root
    return roots


def replay_old_scope_and_handoff(
    *,
    prepare: Callable[..., tuple[object, dict[str, object]]] | None = None,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ]
    | None = None,
) -> tuple[object, object, dict[str, object]]:
    """Replay the old scope, then replace exactly target and readiness."""

    _require_c3_namespace()

    _require_frozen_environment_source()
    if prepare is None:
        prepare = _frozen.prepare_environment_stability_contract
    kwargs: dict[str, object] = {
        "selected_gpu_index": SELECTED_GPU_INDEX,
        "target_unit_id": OLD_TARGET_UNIT,
        "conflict_unit_ids": CONFLICT_UNIT_IDS,
        "dependency_unit_ids": DEPENDENCY_UNIT_IDS,
        "allowed_failed_unit_ids": ALLOWED_FAILED_UNIT_IDS,
        "allowed_unit_ids": ALLOWED_UNIT_IDS,
        "allowed_manager_states": ALLOWED_MANAGER_STATES,
        "require_target_ready": False,
        "strict_all_gpu_consumers": False,
    }
    if activation_guard_reader is not None:
        kwargs["activation_guard_reader"] = activation_guard_reader
    old_contract, roots = prepare(
        PRECLEANUP_PATH,
        CLEANUP_RECEIPT_PATH,
        **kwargs,
    )
    old_contract = _frozen.validate_environment_audit_contract(old_contract)
    if (
        old_contract.target_unit_id != OLD_TARGET_UNIT
        or old_contract.require_target_ready is not False
        or old_contract.selected_gpu_index != SELECTED_GPU_INDEX
        or old_contract.conflict_unit_ids != CONFLICT_UNIT_IDS
        or old_contract.dependency_unit_ids != DEPENDENCY_UNIT_IDS
        or old_contract.allowed_failed_unit_ids
        != ALLOWED_FAILED_UNIT_IDS
        or old_contract.allowed_unit_ids != ALLOWED_UNIT_IDS
        or old_contract.allowed_manager_states
        != ALLOWED_MANAGER_STATES
        or old_contract.strict_all_gpu_consumers is not False
        or set(roots)
        != {"precleanup_inventory_receipt", "cleanup_receipt"}
        or Path(
            str(roots["precleanup_inventory_receipt"]["path"])
        ).absolute()
        != PRECLEANUP_PATH.absolute()
        or Path(str(roots["cleanup_receipt"]["path"])).absolute()
        != CLEANUP_RECEIPT_PATH.absolute()
    ):
        raise PermissionError("frozen old-scope replay is not exact")
    c3_contract = replace(
        old_contract,
        target_unit_id=C3_TARGET_UNIT,
        require_target_ready=True,
    )
    c3_contract = _frozen.validate_environment_audit_contract(c3_contract)
    old_projection = asdict(old_contract)
    c3_projection = asdict(c3_contract)
    for field in fields(old_contract):
        if field.name in _HANDOFF_FIELDS:
            continue
        if not _frozen._deep_exact_equal(
            old_projection[field.name],
            c3_projection[field.name],
        ):
            raise PermissionError(
                f"c3 scope handoff changed field:{field.name}"
            )
    if (
        c3_contract.target_unit_id != C3_TARGET_UNIT
        or c3_contract.require_target_ready is not True
    ):
        raise PermissionError("c3 scope handoff did not close target readiness")
    return old_contract, c3_contract, dict(roots)


def _manager_generation_from_contract(
    contract: object,
) -> dict[str, object]:
    return {
        "boot_id": contract.boot_id,
        "identity": {
            "pid": contract.manager_pid,
            "starttime_ticks": contract.manager_starttime_ticks,
            "uid": contract.uid,
            "control_group": contract.manager_control_group,
        },
        "endpoint": {
            "uid": contract.uid,
            "runtime_directory": contract.runtime_directory,
            "runtime_directory_device":
                contract.runtime_directory_device,
            "runtime_directory_inode":
                contract.runtime_directory_inode,
            "bus_path": contract.bus_path,
            "bus_device": contract.bus_device,
            "bus_inode": contract.bus_inode,
        },
    }


def validate_c3_realization_archival(
    archival: Mapping[str, object],
    *,
    contract: object,
) -> dict[str, object]:
    """Bind the c3 contract to one exact archival realization PASS."""

    try:
        authorization = dict(archival["authorization"])
        receipt = dict(archival["receipt"])
        manager = dict(receipt["manager_generation"])
        fragment = dict(receipt["fragment_identity"])
        shadow = dict(receipt["full_static_shadow"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("c3 realization archival structure is malformed") from error
    _no_payload(authorization, name="c3 realization authorization")
    _no_payload(receipt, name="c3 realization receipt")
    expected_manager = _manager_generation_from_contract(contract)
    expected_shadow = {
        "Id": C3_TARGET_UNIT,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "Restart": "no",
        "NRestarts": "0",
        "FragmentPath": str(C3_FRAGMENT_PATH),
    }
    if (
        authorization.get("unit_name") != C3_TARGET_UNIT
        or receipt.get("unit_name") != C3_TARGET_UNIT
        or receipt.get("passed") is not True
        or receipt.get("static") is not True
        or receipt.get("enabled") is not False
        or receipt.get("started") is not False
        or not _frozen._deep_exact_equal(
            authorization.get("manager_generation"),
            expected_manager,
        )
        or not _frozen._deep_exact_equal(manager, expected_manager)
        or fragment.get("path") != str(C3_FRAGMENT_PATH)
        or shadow.get("FragmentPath") != fragment.get("path")
        or any(
            shadow.get(key) != value
            for key, value in expected_shadow.items()
        )
    ):
        raise PermissionError(
            "c3 realization is not an exact live-ready archival PASS"
        )
    return {
        "authorization": authorization,
        "receipt": receipt,
        "fragment": fragment,
        "shadow": shadow,
    }


def _load_verified_c3_bridge() -> ModuleType:
    _require_c3_namespace()
    if (
        C3_BRIDGE_SHA256 == "__TO_BE_FROZEN__"
        or len(C3_BRIDGE_SHA256) != 64
        or any(character not in "0123456789abcdef"
               for character in C3_BRIDGE_SHA256)
    ):
        raise PermissionError("c3 realization bridge is not frozen")
    raw, identity = _stable_source_bytes(C3_BRIDGE_PATH)
    if hashlib.sha256(raw).hexdigest() != C3_BRIDGE_SHA256:
        raise PermissionError("c3 realization bridge source changed")
    name = (
        "tools._cure_lite_v24_actual_unit_realization_"
        "preaccess_compat_c3_verified_for_environment"
    )
    module = ModuleType(name)
    module.__file__ = str(C3_BRIDGE_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(raw, str(C3_BRIDGE_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    raw_after, identity_after = _stable_source_bytes(C3_BRIDGE_PATH)
    if (
        hashlib.sha256(raw_after).hexdigest() != C3_BRIDGE_SHA256
        or identity_after != identity
    ):
        sys.modules.pop(name, None)
        raise PermissionError(
            "c3 realization bridge generation changed while loading"
        )
    return module


def _production_archival_validator(
    authorization_path: Path,
    receipt_path: Path,
) -> Mapping[str, object]:
    bridge = _load_verified_c3_bridge()
    validator = getattr(bridge, "validate_archival_realization_chain", None)
    if not callable(validator):
        raise PermissionError("c3 archival realization validator is absent")
    return validator(authorization_path, receipt_path)


def _resolve_archival(
    validator: ArchivalValidator,
    *,
    contract: object,
) -> dict[str, object]:
    archival = validator(
        C3_REALIZATION_AUTHORIZATION_PATH,
        C3_REALIZATION_RECEIPT_PATH,
    )
    if not isinstance(archival, Mapping):
        raise PermissionError("c3 archival validator returned no closure")
    return validate_c3_realization_archival(archival, contract=contract)



def _validate_c3_policy_contract(
    policy: Mapping[str, object],
    *,
    c3_contract: object,
    roots: Mapping[str, object],
    archival: Mapping[str, object],
) -> dict[str, object]:
    value = _frozen.validate_environment_policy(policy)
    expected = _frozen.build_environment_policy(
        c3_contract,
        precleanup_root_binding=roots["precleanup_inventory_receipt"],
        cleanup_root_binding=roots["cleanup_receipt"],
        toolchain_binding=value["toolchain"],
        minimum_sample_count=SAMPLE_COUNT,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
    )
    expected_body = dict(expected)
    expected_body.pop("policy_fingerprint")
    policy_body = dict(value)
    policy_body.pop("policy_fingerprint")
    expected_body["created_at_utc"] = policy_body.get("created_at_utc")
    if not _frozen._deep_exact_equal(policy_body, expected_body):
        raise PermissionError("c3 policy differs from exact handoff contract")
    _no_payload(value, name="c3 environment policy")
    if not (
        _strict_utc(
            archival["receipt"]["created_at_utc"],
            name="c3 realization receipt",
        )
        < _strict_utc(
            value["created_at_utc"],
            name="c3 policy",
        )
    ):
        raise PermissionError("c3 policy predates unit realization")
    return dict(value)


def build_c3_policy_in_memory(
    *,
    realization_validator: ArchivalValidator,
    prepare: Callable[..., tuple[object, dict[str, object]]] | None = None,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ]
    | None = None,
    toolchain_reader: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build, but do not write, the fixed c3 fresh environment policy."""

    old_contract, c3_contract, roots = replay_old_scope_and_handoff(
        prepare=prepare,
        activation_guard_reader=activation_guard_reader,
    )
    del old_contract
    archival = _resolve_archival(
        realization_validator,
        contract=c3_contract,
    )
    if toolchain_reader is None:
        toolchain_reader = _frozen.current_runtime_toolchain_binding
    policy = _frozen.build_environment_policy(
        c3_contract,
        precleanup_root_binding=roots["precleanup_inventory_receipt"],
        cleanup_root_binding=roots["cleanup_receipt"],
        toolchain_binding=toolchain_reader(),
        minimum_sample_count=SAMPLE_COUNT,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
    )
    policy = _frozen.validate_environment_policy(policy)
    if not (
        _strict_utc(
            archival["receipt"]["created_at_utc"],
            name="c3 realization receipt",
        )
        < _strict_utc(policy["created_at_utc"], name="c3 policy")
    ):
        raise PermissionError("c3 policy predates unit realization")
    return policy


def run_c3_stability_in_memory(
    policy: Mapping[str, object],
    *,
    realization_validator: ArchivalValidator,
    prepare: Callable[..., tuple[object, dict[str, object]]] | None = None,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ]
    | None = None,
    inventory_collector: Callable[..., dict[str, object]] | None = None,
    sleeper: Callable[[float], None] | None = None,
    monotonic_clock: Callable[[], float] | None = None,
    gate: Callable[..., dict[str, object]] | None = None,
    live_roots: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Run the frozen c3 gate and preserve any supplied live policy root."""

    _require_frozen_environment_source()
    verified_live_roots = None
    if live_roots is not None:
        verified_live_roots = _verify_exact_live_roots(
            live_roots,
            required=frozenset({"policy"}),
        )
    _old, c3_contract, roots = replay_old_scope_and_handoff(
        prepare=prepare,
        activation_guard_reader=activation_guard_reader,
    )
    archival = _resolve_archival(
        realization_validator,
        contract=c3_contract,
    )
    _frozen.validate_environment_policy(policy)
    expected = _frozen.build_environment_policy(
        c3_contract,
        precleanup_root_binding=roots["precleanup_inventory_receipt"],
        cleanup_root_binding=roots["cleanup_receipt"],
        toolchain_binding=policy["toolchain"],
        minimum_sample_count=SAMPLE_COUNT,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
    )
    expected_body = dict(expected)
    expected_body.pop("policy_fingerprint")
    policy_body = dict(policy)
    policy_body.pop("policy_fingerprint")
    expected_body["created_at_utc"] = policy_body.get("created_at_utc")
    if not _frozen._deep_exact_equal(policy_body, expected_body):
        raise PermissionError("c3 policy differs from exact handoff contract")
    if gate is None:
        gate = _frozen.run_environment_stability_gate
    call_kwargs: dict[str, object] = {
        "selected_gpu_index": SELECTED_GPU_INDEX,
        "target_unit_id": C3_TARGET_UNIT,
        "conflict_unit_ids": CONFLICT_UNIT_IDS,
        "dependency_unit_ids": DEPENDENCY_UNIT_IDS,
        "allowed_failed_unit_ids": ALLOWED_FAILED_UNIT_IDS,
        "sample_count": SAMPLE_COUNT,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "policy_path": C3_POLICY_PATH,
        "allowed_unit_ids": ALLOWED_UNIT_IDS,
        "allowed_manager_states": ALLOWED_MANAGER_STATES,
        "require_target_ready": True,
        "strict_all_gpu_consumers": False,
    }
    if inventory_collector is not None:
        call_kwargs["inventory_collector"] = inventory_collector
    if activation_guard_reader is not None:
        call_kwargs["activation_guard_reader"] = activation_guard_reader
    if sleeper is not None:
        call_kwargs["sleeper"] = sleeper
    if monotonic_clock is not None:
        call_kwargs["monotonic_clock"] = monotonic_clock
    stability = gate(
        PRECLEANUP_PATH,
        CLEANUP_RECEIPT_PATH,
        **call_kwargs,
    )
    _require_frozen_environment_source()
    validate_c3_environment_closure(
        policy,
        stability,
        None,
        archival=archival,
        c3_contract=c3_contract,
        live_roots=verified_live_roots,
    )
    return dict(stability)


def build_c3_postcleanup_in_memory(
    policy: Mapping[str, object],
    stability: Mapping[str, object],
    *,
    realization_validator: ArchivalValidator,
    prepare: Callable[..., tuple[object, dict[str, object]]] | None = None,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ]
    | None = None,
    live_roots: Mapping[str, Mapping[str, object]] | None = None,
    clock: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Reuse the exact final stability inventory in a fresh audit receipt."""

    verified_live_roots = None
    if live_roots is not None:
        verified_live_roots = _verify_exact_live_roots(
            live_roots,
            required=frozenset({"policy", "stability"}),
        )
    _old, c3_contract, _roots = replay_old_scope_and_handoff(
        prepare=prepare,
        activation_guard_reader=activation_guard_reader,
    )
    archival = _resolve_archival(
        realization_validator,
        contract=c3_contract,
    )
    validate_c3_environment_closure(
        policy,
        stability,
        None,
        archival=archival,
        c3_contract=c3_contract,
        live_roots=verified_live_roots,
    )
    samples = list(stability["samples"])
    inventory = json.loads(
        _frozen.canonical_json(samples[-1]["inventory"])
    )
    endpoint = dict(inventory["manager"]["endpoint"])
    body = {
        "schema_version": _frozen.ENVIRONMENT_RECEIPT_SCHEMA,
        "created_at_utc": (clock or _frozen.utc_now)(),
        "command": "audit-only",
        "environment_binding": {
            "inventory_fingerprint": inventory["inventory_fingerprint"],
            "boot_id": inventory["boot_id"],
            "runtime_directory": endpoint["runtime_directory"],
            "runtime_directory_device":
                endpoint["runtime_directory_device"],
            "runtime_directory_inode": endpoint["runtime_directory_inode"],
            "manager_identity": inventory["manager"]["identity"],
        },
        "inventory": inventory,
        "passed": True,
        "error_type": None,
        "error_message": None,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    postcleanup = {
        **body,
        "receipt_fingerprint": _frozen.stable_fingerprint(body),
    }
    validate_c3_environment_closure(
        policy,
        stability,
        postcleanup,
        archival=archival,
        c3_contract=c3_contract,
        live_roots=verified_live_roots,
    )
    return postcleanup


def _normalized_contract(contract: object) -> dict[str, object]:
    return json.loads(_frozen.canonical_json(asdict(contract)))


def validate_c3_environment_closure(
    policy: Mapping[str, object],
    stability: Mapping[str, object],
    postcleanup: Mapping[str, object] | None,
    *,
    archival: Mapping[str, object],
    c3_contract: object,
    live_roots: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Validate all cross-bindings omitted by the generic frozen validator.

    Omitting ``live_roots`` is an in-memory-only convenience when neither
    fixed lane exists.  Once a fixed lane exists, this function derives and
    verifies the applicable sealed root instead of silently downgrading to an
    in-memory validation.
    """

    verified_live_roots = None
    live_required = frozenset()
    fixed_state = (
        os.path.lexists(C3_POLICY_PATH),
        os.path.lexists(C3_STABILITY_PATH),
    )
    fixed_postcleanup = os.path.lexists(C3_POSTCLEANUP_PATH)
    fixed_postcleanup_root: dict[str, object] | None = None
    if fixed_state == (False, True):
        raise PermissionError(
            "c3 closure fixed live-root state is partial"
        )
    if fixed_postcleanup and fixed_state != (True, True):
        raise PermissionError(
            "c3 fixed postcleanup predecessors are incomplete"
        )
    if fixed_state == (True, False):
        if postcleanup is not None:
            raise PermissionError(
                "c3 postcleanup fixed live roots are incomplete"
            )
        live_policy, policy_root = _read_live_sealed(
            C3_POLICY_PATH,
            fingerprint_field="policy_fingerprint",
        )
        if not _frozen._deep_exact_equal(live_policy, policy):
            raise PermissionError(
                "c3 fixed policy payload changed"
            )
        if live_roots is not None:
            supplied = _verify_exact_live_roots(
                live_roots,
                required=frozenset({"policy"}),
            )
            if not _frozen._deep_exact_equal(
                supplied["policy"],
                policy_root,
            ):
                raise PermissionError(
                    "c3 supplied policy root changed"
                )
        live_roots = {"policy": policy_root}
    elif fixed_state == (True, True):
        live_policy, policy_root = _read_live_sealed(
            C3_POLICY_PATH,
            fingerprint_field="policy_fingerprint",
        )
        live_stability, stability_root = _read_live_sealed(
            C3_STABILITY_PATH,
            fingerprint_field="stability_receipt_fingerprint",
        )
        if (
            not _frozen._deep_exact_equal(live_policy, policy)
            or not _frozen._deep_exact_equal(
                live_stability,
                stability,
            )
        ):
            raise PermissionError(
                "c3 fixed environment payload changed"
            )
        fixed_roots = {
            "policy": policy_root,
            "stability": stability_root,
        }
        if live_roots is not None:
            supplied_required = frozenset(live_roots)
            if supplied_required not in {
                frozenset({"policy"}),
                frozenset({"policy", "stability"}),
            }:
                raise PermissionError(
                    "c3 closure supplied live-root lane changed"
                )
            supplied = _verify_exact_live_roots(
                live_roots,
                required=supplied_required,
            )
            if any(
                not _frozen._deep_exact_equal(
                    supplied[name],
                    fixed_roots[name],
                )
                for name in supplied_required
            ):
                raise PermissionError(
                    "c3 supplied fixed live root changed"
                )
        live_roots = fixed_roots
    if fixed_postcleanup:
        if postcleanup is None:
            raise PermissionError(
                "c3 fixed postcleanup payload was omitted"
            )
        live_postcleanup, fixed_postcleanup_root = _read_live_sealed(
            C3_POSTCLEANUP_PATH,
            fingerprint_field="receipt_fingerprint",
        )
        if not _frozen._deep_exact_equal(
            live_postcleanup,
            postcleanup,
        ):
            raise PermissionError(
                "c3 fixed postcleanup payload changed"
            )
    if live_roots is not None:
        live_required = frozenset(live_roots)
        allowed = (
            {frozenset({"policy"}), frozenset({"policy", "stability"})}
            if postcleanup is None
            else {frozenset({"policy", "stability"})}
        )
        if live_required not in allowed:
            raise PermissionError("c3 closure live-root lane changed")
        verified_live_roots = _verify_exact_live_roots(
            live_roots,
            required=live_required,
        )
    policy_value = _frozen.validate_environment_policy(policy)
    stability_value = _frozen.validate_environment_stability_receipt(
        stability
    )
    archival_value = validate_c3_realization_archival(
        archival,
        contract=c3_contract,
    )
    _no_payload(policy_value, name="c3 environment policy")
    _no_payload(stability_value, name="c3 environment stability")
    scope = dict(policy_value["unit_scope"])
    sampling = dict(policy_value["sampling"])
    contract_value = dict(stability_value["contract"])
    samples = list(stability_value["samples"])
    expected_policy = _frozen.build_environment_policy(
        c3_contract,
        precleanup_root_binding=policy_value["precleanup_root"],
        cleanup_root_binding=policy_value["cleanup_root"],
        toolchain_binding=policy_value["toolchain"],
        minimum_sample_count=SAMPLE_COUNT,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
    )
    expected_policy_body = dict(expected_policy)
    expected_policy_body.pop("policy_fingerprint")
    policy_body = dict(policy_value)
    policy_body.pop("policy_fingerprint")
    expected_policy_body["created_at_utc"] = policy_body.get(
        "created_at_utc"
    )
    roots = dict(stability_value["root_evidence"])
    policy_root = dict(roots.get("policy", {}))
    if (
        scope.get("target_unit_id") != C3_TARGET_UNIT
        or scope.get("require_target_ready") is not True
        or not _frozen._deep_exact_equal(
            policy_body,
            expected_policy_body,
        )
        or sampling.get("minimum_sample_count") != SAMPLE_COUNT
        or sampling.get("sample_interval_seconds")
        != SAMPLE_INTERVAL_SECONDS
        or Path(str(policy_value["precleanup_root"]["path"])).absolute()
        != PRECLEANUP_PATH.absolute()
        or Path(str(policy_value["cleanup_root"]["path"])).absolute()
        != CLEANUP_RECEIPT_PATH.absolute()
        or Path(str(policy_root.get("path", ""))).absolute()
        != C3_POLICY_PATH.absolute()
        or policy_root.get("policy_fingerprint")
        != policy_value.get("policy_fingerprint")
        or verified_live_roots is not None
        and not _frozen._deep_exact_equal(
            policy_root,
            verified_live_roots["policy"],
        )
        or not _frozen._deep_exact_equal(
            roots.get("precleanup_inventory_receipt"),
            policy_value.get("precleanup_root"),
        )
        or not _frozen._deep_exact_equal(
            roots.get("cleanup_receipt"),
            policy_value.get("cleanup_root"),
        )
        or not _frozen._deep_exact_equal(
            contract_value,
            _normalized_contract(c3_contract),
        )
        or stability_value.get("sample_count") != SAMPLE_COUNT
        or stability_value.get("sample_interval_seconds")
        != SAMPLE_INTERVAL_SECONDS
        or len(samples) != SAMPLE_COUNT
        or stability_value.get("passed") is not True
        or stability_value.get("blockers") != []
    ):
        raise PermissionError("c3 stability scope handoff is not exact")
    realization_time = _strict_utc(
        archival_value["receipt"]["created_at_utc"],
        name="c3 realization receipt",
    )
    policy_time = _strict_utc(
        policy_value["created_at_utc"],
        name="c3 policy",
    )
    if not realization_time < policy_time:
        raise PermissionError("c3 realization/policy chronology changed")
    prior_time = policy_time
    for index, sample in enumerate(samples):
        sample_time = _strict_utc(
            sample.get("created_at_utc"),
            name=f"c3 stability sample:{index}",
        )
        try:
            inventory = dict(sample["inventory"])
            inventory_scope = dict(inventory["unit_scope"])
            target_shadow = dict(
                inventory_scope["shadows"][C3_TARGET_UNIT]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"c3 target sample is malformed:{index}"
            ) from error
        expected_target = {
            "Id": C3_TARGET_UNIT,
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "static",
            "Restart": "no",
            "NRestarts": "0",
            "FragmentPath": archival_value["fragment"]["path"],
        }
        if (
            not prior_time < sample_time
            or inventory_scope.get("target_unit_id") != C3_TARGET_UNIT
            or inventory_scope.get("require_target_ready") is not True
            or any(
                target_shadow.get(key) != value
                for key, value in expected_target.items()
            )
            or not _frozen._deep_exact_equal(
                _manager_generation_from_inventory(inventory),
                _manager_generation_from_contract(c3_contract),
            )
        ):
            raise PermissionError(
                f"c3 target sample cross-binding changed:{index}"
            )
        _no_payload(sample, name=f"c3 stability sample:{index}")
        _no_payload(inventory, name=f"c3 inventory:{index}")
        prior_time = sample_time
    if postcleanup is None:
        if verified_live_roots is not None:
            _verify_exact_live_roots(
                verified_live_roots,
                required=live_required,
            )
        return {
            "policy": dict(policy_value),
            "stability": dict(stability_value),
            "realization": archival_value,
        }
    post = dict(postcleanup)
    body = dict(post)
    fingerprint = body.pop("receipt_fingerprint", None)
    expected_keys = {
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
    }
    final_inventory = dict(samples[-1]["inventory"])
    post_time = _strict_utc(
        post.get("created_at_utc"),
        name="c3 postcleanup",
    )
    if (
        set(body) != expected_keys
        or post.get("schema_version") != _frozen.ENVIRONMENT_RECEIPT_SCHEMA
        or post.get("command") != "audit-only"
        or post.get("passed") is not True
        or post.get("error_type") is not None
        or post.get("error_message") is not None
        or fingerprint != _frozen.stable_fingerprint(body)
        or not _frozen._deep_exact_equal(
            post.get("inventory"),
            final_inventory,
        )
        or not _frozen._deep_exact_equal(
            post.get("environment_binding"),
            _environment_binding_from_inventory(final_inventory),
        )
        or not prior_time < post_time
    ):
        raise PermissionError("c3 postcleanup cross-binding changed")
    _no_payload(post, name="c3 postcleanup")
    if verified_live_roots is not None:
        _verify_exact_live_roots(
            verified_live_roots,
            required=live_required,
        )
    if fixed_postcleanup_root is not None:
        _verify_live_sealed(
            C3_POSTCLEANUP_PATH,
            fixed_postcleanup_root,
            "receipt_fingerprint",
        )
    return {
        "policy": dict(policy_value),
        "stability": dict(stability_value),
        "postcleanup": post,
        "realization": archival_value,
    }


def _manager_generation_from_inventory(
    inventory: Mapping[str, object],
) -> dict[str, object]:
    manager = dict(inventory["manager"])
    return {
        "boot_id": inventory["boot_id"],
        "identity": dict(manager["identity"]),
        "endpoint": dict(manager["endpoint"]),
    }


def _environment_binding_from_inventory(
    inventory: Mapping[str, object],
) -> dict[str, object]:
    manager = dict(inventory["manager"])
    endpoint = dict(manager["endpoint"])
    return {
        "inventory_fingerprint": inventory["inventory_fingerprint"],
        "boot_id": inventory["boot_id"],
        "runtime_directory": endpoint["runtime_directory"],
        "runtime_directory_device": endpoint["runtime_directory_device"],
        "runtime_directory_inode": endpoint["runtime_directory_inode"],
        "manager_identity": manager["identity"],
    }


def _write_create_once(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise PermissionError("c3 environment writes are CLI-closure only")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CURE-Lite v24 c3 fresh environment scope handoff",
    )
    parser.add_argument(
        "command",
        choices=("create-policy", "stability-gate", "postcleanup"),
    )
    return parser


def _validate_cli_payload(
    payload: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    value = dict(payload)
    _no_payload(value, name="c3 environment CLI payload")
    if fingerprint_field == "policy_fingerprint":
        return _frozen.validate_environment_policy(value)
    if fingerprint_field == "stability_receipt_fingerprint":
        return _frozen.validate_environment_stability_receipt(value)
    if fingerprint_field != "receipt_fingerprint":
        raise PermissionError("c3 environment fingerprint field changed")
    body = dict(value)
    fingerprint = body.pop("receipt_fingerprint", None)
    expected_keys = {
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
    }
    if (
        set(body) != expected_keys
        or value.get("schema_version")
        != _frozen.ENVIRONMENT_RECEIPT_SCHEMA
        or value.get("command") != "audit-only"
        or value.get("passed") is not True
        or value.get("error_type") is not None
        or value.get("error_message") is not None
        or fingerprint != _frozen.stable_fingerprint(body)
    ):
        raise PermissionError("c3 postcleanup payload is not closed")
    _strict_utc(value.get("created_at_utc"), name="c3 postcleanup")
    return value


def _bind_private_cli() -> Callable[[Sequence[str] | None], int]:
    fixed_identity = (
        FROZEN_ENVIRONMENT_PATH,
        FROZEN_ENVIRONMENT_SHA256,
        C3_TARGET_UNIT,
        C3_FRAGMENT_PATH,
        PRECLEANUP_PATH,
        CLEANUP_RECEIPT_PATH,
        C3_POLICY_PATH,
        C3_STABILITY_PATH,
        C3_POSTCLEANUP_PATH,
        C3_REALIZATION_AUTHORIZATION_PATH,
        C3_REALIZATION_RECEIPT_PATH,
        C3_BRIDGE_PATH,
        C3_BRIDGE_SHA256,
        SELECTED_GPU_INDEX,
        CONFLICT_UNIT_IDS,
        DEPENDENCY_UNIT_IDS,
        ALLOWED_FAILED_UNIT_IDS,
        ALLOWED_UNIT_IDS,
        ALLOWED_MANAGER_STATES,
        SAMPLE_COUNT,
        SAMPLE_INTERVAL_SECONDS,
    )
    fixed_policy_path = C3_POLICY_PATH
    fixed_stability_path = C3_STABILITY_PATH
    fixed_postcleanup_path = C3_POSTCLEANUP_PATH

    def require_fixed_identity() -> None:
        current_identity = (
            FROZEN_ENVIRONMENT_PATH,
            FROZEN_ENVIRONMENT_SHA256,
            C3_TARGET_UNIT,
            C3_FRAGMENT_PATH,
            PRECLEANUP_PATH,
            CLEANUP_RECEIPT_PATH,
            C3_POLICY_PATH,
            C3_STABILITY_PATH,
            C3_POSTCLEANUP_PATH,
            C3_REALIZATION_AUTHORIZATION_PATH,
            C3_REALIZATION_RECEIPT_PATH,
            C3_BRIDGE_PATH,
            C3_BRIDGE_SHA256,
            SELECTED_GPU_INDEX,
            CONFLICT_UNIT_IDS,
            DEPENDENCY_UNIT_IDS,
            ALLOWED_FAILED_UNIT_IDS,
            ALLOWED_UNIT_IDS,
            ALLOWED_MANAGER_STATES,
            SAMPLE_COUNT,
            SAMPLE_INTERVAL_SECONDS,
        )
        if current_identity != fixed_identity:
            raise PermissionError("c3 environment fixed identity changed")
        _require_c3_namespace()
        _require_frozen_environment_source()

    def require_order(fingerprint_field: str) -> Path:
        paths = (
            fixed_policy_path,
            fixed_stability_path,
            fixed_postcleanup_path,
        )
        state = tuple(os.path.lexists(path) for path in paths)
        transitions = {
            "policy_fingerprint": ((False, False, False), paths[0]),
            "stability_receipt_fingerprint": (
                (True, False, False),
                paths[1],
            ),
            "receipt_fingerprint": ((True, True, False), paths[2]),
        }
        transition = transitions.get(fingerprint_field)
        if transition is None or state != transition[0]:
            raise PermissionError("c3 environment write order is invalid")
        return transition[1]

    def bound_main(argv: Sequence[str] | None = None) -> int:
        args = _parser().parse_args(argv)
        if args.command not in {
            "create-policy",
            "stability-gate",
            "postcleanup",
        }:
            raise PermissionError("c3 environment command changed")
        require_fixed_identity()
        # R3 is hash-verified before any evidence transition is inspected or
        # any policy/stability/postcleanup builder can execute.
        _load_verified_c3_bridge()
        validator = _production_archival_validator
        policy_root: dict[str, object] | None = None
        stability_root: dict[str, object] | None = None

        def same_generation(
            left: os.stat_result,
            right: os.stat_result,
            fields_to_check: Sequence[str],
        ) -> bool:
            return all(
                getattr(left, field) == getattr(right, field)
                for field in fields_to_check
            )

        def write_fixed_lane(
            payload: Mapping[str, object],
            lane: str,
        ) -> dict[str, object]:
            lanes = {
                "policy": (
                    fixed_policy_path,
                    "policy_fingerprint",
                    (),
                    (True, False, False),
                ),
                "stability": (
                    fixed_stability_path,
                    "stability_receipt_fingerprint",
                    ("policy",),
                    (True, True, False),
                ),
                "postcleanup": (
                    fixed_postcleanup_path,
                    "receipt_fingerprint",
                    ("policy", "stability"),
                    (True, True, True),
                ),
            }
            if lane not in lanes:
                raise PermissionError("c3 environment write lane changed")
            (
                path,
                fingerprint_field,
                guard_names,
                expected_state,
            ) = lanes[lane]

            def require_lane_state() -> None:
                observed = tuple(
                    os.path.lexists(candidate)
                    for candidate in (
                        fixed_policy_path,
                        fixed_stability_path,
                        fixed_postcleanup_path,
                    )
                )
                if observed != expected_state:
                    raise PermissionError("c3 fixed lane state changed")
            require_fixed_identity()
            if require_order(fingerprint_field) != path:
                raise PermissionError("c3 fixed write target changed")
            value = _validate_cli_payload(
                payload,
                fingerprint_field=fingerprint_field,
            )
            body = dict(value)
            supplied_fingerprint = body.pop(fingerprint_field, None)
            if supplied_fingerprint != _frozen.stable_fingerprint(body):
                raise PermissionError("c3 fixed write fingerprint changed")
            sealed_value = {
                **body,
                fingerprint_field: supplied_fingerprint,
            }
            encoded = (
                _frozen.canonical_json(sealed_value) + "\n"
            ).encode("utf-8")
            target = Path(path).absolute()

            def revalidate_lane_closure() -> None:
                _historical, guard_contract, guard_roots = (
                    replay_old_scope_and_handoff()
                )
                del _historical
                guard_archival = _resolve_archival(
                    validator,
                    contract=guard_contract,
                )
                current_value, current_root = _read_live_sealed(
                    target,
                    fingerprint_field,
                )
                if not _frozen._deep_exact_equal(
                    current_value,
                    sealed_value,
                ):
                    raise PermissionError(
                        "c3 fixed lane payload changed while open"
                    )
                if lane == "policy":
                    _validate_c3_policy_contract(
                        current_value,
                        c3_contract=guard_contract,
                        roots=guard_roots,
                        archival=guard_archival,
                    )
                    return
                current_policy, current_policy_root = _read_live_sealed(
                    fixed_policy_path,
                    "policy_fingerprint",
                )
                if (
                    policy_root is None
                    or not _frozen._deep_exact_equal(
                        current_policy_root,
                        policy_root,
                    )
                ):
                    raise PermissionError(
                        "c3 fixed policy root changed while open"
                    )
                _validate_c3_policy_contract(
                    current_policy,
                    c3_contract=guard_contract,
                    roots=guard_roots,
                    archival=guard_archival,
                )
                if lane == "stability":
                    validate_c3_environment_closure(
                        current_policy,
                        current_value,
                        None,
                        archival=guard_archival,
                        c3_contract=guard_contract,
                        live_roots={
                            "policy": current_policy_root,
                            "stability": current_root,
                        },
                    )
                    return
                current_stability, current_stability_root = (
                    _read_live_sealed(
                        fixed_stability_path,
                        "stability_receipt_fingerprint",
                    )
                )
                if (
                    stability_root is None
                    or not _frozen._deep_exact_equal(
                        current_stability_root,
                        stability_root,
                    )
                ):
                    raise PermissionError(
                        "c3 fixed stability root changed while open"
                    )
                validate_c3_environment_closure(
                    current_policy,
                    current_stability,
                    current_value,
                    archival=guard_archival,
                    c3_contract=guard_contract,
                    live_roots={
                        "policy": current_policy_root,
                        "stability": current_stability_root,
                    },
                )

            parent = target.parent
            parent_before = parent.lstat()
            if (
                target != path
                or target.name in {"", ".", ".."}
                or target.parent / target.name != target
                or not stat.S_ISDIR(parent_before.st_mode)
                or stat.S_ISLNK(parent_before.st_mode)
                or parent.resolve(strict=True) != parent
                or parent_before.st_uid != os.getuid()
                or stat.S_IMODE(parent_before.st_mode) & 0o022
                or not hasattr(os, "O_NOFOLLOW")
                or not hasattr(os, "O_DIRECTORY")
            ):
                raise PermissionError("c3 fixed write parent is unsafe")
            directory_fd = os.open(
                parent,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
            )
            descriptor = -1
            try:
                parent_opened = os.fstat(directory_fd)
                if not same_generation(
                    parent_before,
                    parent_opened,
                    _PARENT_IDENTITY_FIELDS,
                ):
                    raise PermissionError(
                        "c3 fixed write parent generation changed"
                    )
                descriptor = os.open(
                    target.name,
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
                os.fchmod(descriptor, 0o600)
                created = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(created.st_mode)
                    or created.st_uid != os.getuid()
                    or created.st_nlink != 1
                    or stat.S_IMODE(created.st_mode) != 0o600
                ):
                    raise PermissionError(
                        "c3 fixed write private generation changed"
                    )
                offset = 0
                while offset < len(encoded):
                    written = os.write(descriptor, encoded[offset:])
                    if written <= 0:
                        raise OSError("short c3 fixed evidence write")
                    offset += written
                os.fsync(descriptor)
                private_written = os.fstat(descriptor)
                private_readback = os.pread(
                    descriptor,
                    len(encoded) + 1,
                    0,
                )
                if (
                    not stat.S_ISREG(private_written.st_mode)
                    or private_written.st_uid != os.getuid()
                    or private_written.st_nlink != 1
                    or stat.S_IMODE(private_written.st_mode) != 0o600
                    or private_written.st_size != len(encoded)
                    or private_readback != encoded
                ):
                    raise PermissionError(
                        "c3 fixed write private readback changed"
                    )
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
                sealed = os.fstat(descriptor)
                linked = os.stat(
                    target.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                sealed_readback = os.pread(
                    descriptor,
                    len(encoded) + 1,
                    0,
                )
                if (
                    not stat.S_ISREG(sealed.st_mode)
                    or sealed.st_uid != os.getuid()
                    or sealed.st_nlink != 1
                    or stat.S_IMODE(sealed.st_mode) != 0o444
                    or sealed.st_size != len(encoded)
                    or not same_generation(
                        sealed,
                        linked,
                        _FILE_IDENTITY_FIELDS,
                    )
                    or sealed_readback != encoded
                    or hashlib.sha256(sealed_readback).hexdigest()
                    != hashlib.sha256(encoded).hexdigest()
                ):
                    raise PermissionError(
                        "c3 fixed write sealed generation changed"
                    )
                os.fsync(directory_fd)
                parent_after_create = os.fstat(directory_fd)
                parent_linked = parent.lstat()
                if (
                    not same_generation(
                        parent_opened,
                        parent_after_create,
                        _PARENT_IDENTITY_FIELDS,
                    )
                    or not same_generation(
                        parent_after_create,
                        parent_linked,
                        _PARENT_IDENTITY_FIELDS,
                    )
                ):
                    raise PermissionError(
                        "c3 fixed write parent generation changed"
                    )
                require_fixed_identity()
                _load_verified_c3_bridge()
                available_roots = {
                    "policy": policy_root,
                    "stability": stability_root,
                }
                for guard_name in guard_names:
                    guard_root = available_roots[guard_name]
                    if guard_root is None:
                        raise PermissionError(
                            "c3 fixed while-open guard is absent"
                        )
                    guard_path, guard_field = _LIVE_ROOT_SPECS[
                        guard_name
                    ]
                    _verify_live_sealed(
                        guard_path,
                        guard_root,
                        guard_field,
                    )
                require_lane_state()
                revalidate_lane_closure()
                require_lane_state()
                guarded_before = os.fstat(descriptor)
                guarded_readback = os.pread(
                    descriptor,
                    len(encoded) + 1,
                    0,
                )
                guarded_after = os.fstat(descriptor)
                guarded_linked = os.stat(
                    target.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                parent_guarded = os.fstat(directory_fd)
                parent_guarded_path = parent.lstat()
                if (
                    not same_generation(
                        sealed,
                        guarded_before,
                        _FILE_IDENTITY_FIELDS,
                    )
                    or not same_generation(
                        guarded_before,
                        guarded_after,
                        _FILE_IDENTITY_FIELDS,
                    )
                    or not same_generation(
                        guarded_after,
                        guarded_linked,
                        _FILE_IDENTITY_FIELDS,
                    )
                    or guarded_readback != encoded
                    or not same_generation(
                        parent_after_create,
                        parent_guarded,
                        _PARENT_GUARD_IDENTITY_FIELDS,
                    )
                    or not same_generation(
                        parent_guarded,
                        parent_guarded_path,
                        _PARENT_GUARD_IDENTITY_FIELDS,
                    )
                ):
                    raise PermissionError(
                        "c3 fixed while-open closure changed"
                    )
            except BaseException:
                for candidate in (descriptor, directory_fd):
                    if candidate < 0:
                        continue
                    try:
                        os.fsync(candidate)
                    except BaseException:
                        pass
                for candidate in (descriptor, directory_fd):
                    if candidate < 0:
                        continue
                    try:
                        os.close(candidate)
                    except BaseException:
                        pass
                raise
            os.close(descriptor)
            os.close(directory_fd)
            require_fixed_identity()
            _load_verified_c3_bridge()
            require_lane_state()
            parent_final = parent.lstat()
            if not same_generation(
                parent_guarded_path,
                parent_final,
                _PARENT_GUARD_IDENTITY_FIELDS,
            ):
                raise PermissionError(
                    "c3 fixed parent changed after close"
                )
            for guard_name in guard_names:
                guard_root = available_roots[guard_name]
                if guard_root is None:
                    raise PermissionError(
                        "c3 fixed post-close guard is absent"
                    )
                guard_path, guard_field = _LIVE_ROOT_SPECS[guard_name]
                _verify_live_sealed(
                    guard_path,
                    guard_root,
                    guard_field,
                )
            observed_value, observed_root = _read_live_sealed(
                target,
                fingerprint_field,
            )
            if not _frozen._deep_exact_equal(
                observed_value,
                sealed_value,
            ):
                raise PermissionError("c3 fixed write reload changed")
            _verify_live_sealed(
                target,
                observed_root,
                fingerprint_field,
            )
            require_lane_state()
            parent_reloaded = parent.lstat()
            if not same_generation(
                parent_final,
                parent_reloaded,
                _PARENT_GUARD_IDENTITY_FIELDS,
            ):
                raise PermissionError(
                    "c3 fixed parent changed during reload"
                )
            for guard_name in guard_names:
                guard_root = available_roots[guard_name]
                if guard_root is None:
                    raise PermissionError(
                        "c3 fixed terminal guard is absent"
                    )
                guard_path, guard_field = _LIVE_ROOT_SPECS[guard_name]
                _verify_live_sealed(
                    guard_path,
                    guard_root,
                    guard_field,
                )
            _verify_live_sealed(
                target,
                observed_root,
                fingerprint_field,
            )
            require_lane_state()
            parent_terminal = parent.lstat()
            if not same_generation(
                parent_reloaded,
                parent_terminal,
                _PARENT_GUARD_IDENTITY_FIELDS,
            ):
                raise PermissionError(
                    "c3 fixed parent changed before return"
                )
            return observed_value
        if args.command == "create-policy":
            require_order("policy_fingerprint")
            policy = build_c3_policy_in_memory(
                realization_validator=validator,
            )
            write_fixed_lane(policy, "policy")
            return 0
        if args.command == "stability-gate":
            require_order("stability_receipt_fingerprint")
            policy, policy_root = _read_live_sealed(
                fixed_policy_path,
                fingerprint_field="policy_fingerprint",
            )
            _validate_cli_payload(
                policy,
                fingerprint_field="policy_fingerprint",
            )
            stability = run_c3_stability_in_memory(
                policy,
                realization_validator=validator,
                live_roots={"policy": policy_root},
            )
            write_fixed_lane(stability, "stability")
            return 0
        require_order("receipt_fingerprint")
        if args.command != "postcleanup":
            raise PermissionError("c3 environment command changed")

        policy, policy_root = _read_live_sealed(
            fixed_policy_path,
            fingerprint_field="policy_fingerprint",
        )
        _validate_cli_payload(
            policy,
            fingerprint_field="policy_fingerprint",
        )
        stability, stability_root = _read_live_sealed(
            fixed_stability_path,
            fingerprint_field="stability_receipt_fingerprint",
        )
        _validate_cli_payload(
            stability,
            fingerprint_field="stability_receipt_fingerprint",
        )
        postcleanup = build_c3_postcleanup_in_memory(
            policy,
            stability,
            realization_validator=validator,
            live_roots={
                "policy": policy_root,
                "stability": stability_root,
            },
        )
        write_fixed_lane(postcleanup, "postcleanup")
        return 0

    return bound_main


main = _bind_private_cli()
del _bind_private_cli
del _loaded_frozen


if __name__ == "__main__":
    raise SystemExit(main())
