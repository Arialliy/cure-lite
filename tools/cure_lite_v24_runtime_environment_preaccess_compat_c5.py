#!/usr/bin/env python3
"""Fresh runtime-environment scope handoff for compatibility generation c5.

This module does not reinterpret the frozen precleanup receipt as evidence
about the c5 unit.  It first replays that receipt with its original target and
``require_target_ready=False``.  It then creates a new contract by replacing
only those two fields and delegates policy construction and stability sampling
to the hash-pinned runtime-environment implementation.

The production entry point remains fail-closed until the B5, R5, and C4
terminal producers have been frozen and their explicit SHA-256 sentinels have
been replaced.  Pure in-memory functions accept injected producer validators
for unit tests; foreign evidence is never re-canonicalized by this module.
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
import subprocess
import sys
import time
from types import MappingProxyType, ModuleType
from typing import Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()
RUNS_ROOT = (REPOSITORY / "runs/irstd1k_stage_a_seed42").resolve()

FROZEN_ENVIRONMENT_PATH = (
    REPOSITORY / "tools/cure_lite_v24_runtime_environment.py"
).resolve()
FROZEN_ENVIRONMENT_SHA256 = (
    "a40465786ce3537346372df5991bb6788d44feddfd497ec83a1dc302fb8b2fea"
)

OLD_TARGET_UNIT = "cure-lite-v24-gcr-pacre-dr-r2.service"
C5_TARGET_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c5.service"
)
C5_FRAGMENT_PATH = (
    Path(f"/run/user/{os.getuid()}/systemd/user") / C5_TARGET_UNIT
)

PRECLEANUP_PATH = (
    EVIDENCE_ROOT / "runtime_environment_precleanup_receipt.json"
)
CLEANUP_RECEIPT_PATH = (
    EVIDENCE_ROOT / "environment_cleanup_recovery_r1/cleanup-receipt.json"
)
C5_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c5.json"
)
C5_STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c5.json"
)
C5_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c5.json"
)
C5_REALIZATION_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c5_unit_realization_authorization.json"
)
C5_REALIZATION_RECEIPT_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c5_unit_realization_receipt.json"
)
C5_BRIDGE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c5.py"
).resolve()
C5_BRIDGE_SHA256 = (
    "dbe35cd096554c4fd4c64b34213b0f7ac3ccb79e396f6d1d8e620c2c4c1d1be5"
)

C5_COMPATIBILITY_BRIDGE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c5.py"
).resolve()
C5_COMPATIBILITY_BRIDGE_SHA256 = (
    "388843b9b840db41610d57543f4982666cdf442ba81fa5acb208033de062319f"
)

C4_FAILURE_TERMINALIZER_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_compat_c4_receipt_seal_failure_terminal.py"
).resolve()
C4_FAILURE_TERMINALIZER_SHA256 = (
    "3cf56e803d6d7b39c995125d17d145b5c8625a4eea03de6cf4c6118c9bc777c0"
)
C4_FAILURE_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_schema_compat_c4_receipt_seal_failure_terminal.json"
)
C4_FAILURE_TERMINAL_SHA256 = (
    "567b22e9839dad2d27168c36206b66be9b2b91d98269e9b9ce087ee3becea733"
)
C4_FAILURE_TERMINAL_FINGERPRINT = (
    "d86ef0c432237043e39119c56cfb6602b7df7f8b62069f836ac6c3d08b75b622"
)
C4_TARGET_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service"
)
C4_FRAGMENT_PATH = (
    Path(f"/run/user/{os.getuid()}/systemd/user") / C4_TARGET_UNIT
)

C5_SCOPE_HANDOFF_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_scope_handoff_preaccess_compat_c5.json"
)
# Backward-readable alias for the path name used during the C5 design review.
C5_HANDOFF_PATH = C5_SCOPE_HANDOFF_PATH
C5_STABILITY_ATTEMPT_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_attempt_preaccess_compat_c5.json"
)
C5_STABILITY_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_terminal_preaccess_compat_c5.json"
)
C5_HANDOFF_SCHEMA = (
    "cure-lite-v24-runtime-environment-scope-handoff-"
    "preaccess-compat-c5-v1"
)
C5_STABILITY_ATTEMPT_SCHEMA = (
    "cure-lite-v24-runtime-environment-stability-attempt-"
    "preaccess-compat-c5-v1"
)
C5_STABILITY_TERMINAL_SCHEMA = (
    "cure-lite-v24-runtime-environment-stability-terminal-"
    "preaccess-compat-c5-v1"
)

_C4_REQUIRED_ABSENT_PATHS = MappingProxyType(
    {
        "B4_compatibility_receipt": (
            EVIDENCE_ROOT / "r2_preaccess_schema_compat_c4_receipt.json"
        ),
        "R4_unit_terminal": (
            EVIDENCE_ROOT
            / "r2_preaccess_compat_c4_unit_realization_terminal.json"
        ),
        "E4_environment_terminal": (
            EVIDENCE_ROOT
            / "runtime_environment_stability_terminal_preaccess_compat_c4.json"
        ),
        "r14_integration_root": (
            EVIDENCE_ROOT
            / "supervisor_v2_systemd_integration_preaccess_compat_c4_r14"
        ),
        "L4_C4_runtime_spec": (
            EVIDENCE_ROOT
            / "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_spec.json"
        ),
        "L4_C4_runtime_launch_authorization": (
            EVIDENCE_ROOT
            / (
                "D_R_structural_attempt_r2_preaccess_compat_c4_"
                "runtime_launch_authorization.json"
            )
        ),
        "C4_runtime_artifacts": (
            EVIDENCE_ROOT
            / "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_artifacts"
        ),
        "C4_gpu_lease": (
            EVIDENCE_ROOT
            / "D_R_structural_attempt_r2_preaccess_compat_c4_gpu_lease"
        ),
        "C4_run_alias": (
            RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c4"
        ),
        "C4_result_alias": (
            EVIDENCE_ROOT
            / "D_R_structural_attempt_r2_preaccess_compat_c4_receipt.json"
        ),
        "direct_runtime_spec": (
            EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_spec.json"
        ),
        "direct_runtime_launch_authorization": (
            EVIDENCE_ROOT
            / "D_R_structural_attempt_r2_runtime_launch_authorization.json"
        ),
        "direct_runtime_artifacts": (
            EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_artifacts"
        ),
        "direct_gpu_lease": (
            EVIDENCE_ROOT / "D_R_structural_attempt_r2_gpu_lease"
        ),
        "scientific_run_root": (
            RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2"
        ),
        "scientific_result_receipt": (
            EVIDENCE_ROOT / "D_R_structural_attempt_r2_receipt.json"
        ),
    }
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


def _require_c5_namespace() -> None:
    evidence_paths = (
        C5_POLICY_PATH,
        C5_STABILITY_PATH,
        C5_POSTCLEANUP_PATH,
        C5_REALIZATION_AUTHORIZATION_PATH,
        C5_REALIZATION_RECEIPT_PATH,
        C5_SCOPE_HANDOFF_PATH,
        C5_STABILITY_ATTEMPT_PATH,
        C5_STABILITY_TERMINAL_PATH,
    )
    if (
        "preaccess-compat-c5" not in C5_TARGET_UNIT
        or C5_FRAGMENT_PATH.name != C5_TARGET_UNIT
        or len(set(evidence_paths)) != len(evidence_paths)
        or any("compat_c5" not in path.name for path in evidence_paths)
        or "compat_c5" not in C5_BRIDGE_PATH.name
        or any(
            marker in str(path)
            for marker in ("compat_c2", "compat_c3")
            for path in evidence_paths
        )
        or "compat_c4" not in C4_FAILURE_TERMINAL_PATH.name
        or "compat_c4" not in C4_FAILURE_TERMINALIZER_PATH.name
        or "compatibility_c5" not in C5_COMPATIBILITY_BRIDGE_PATH.name
    ):
        raise PermissionError("c5 environment namespace is not exact")


def _canonical_json(value: object) -> str:
    """Canonical profile owned by E5 (UTF-8, never producer fallback)."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _stable_fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()

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
_R5_SUPPLIED_IDENTITY_FIELDS = frozenset(
    {
        "path",
        "resolved_path",
        "path_is_symlink",
        "file_sha256",
        "device",
        "inode",
        "owner_uid",
        "mode",
        "nlink",
    }
)
_R5_ARCHIVAL_ROOT_FIELDS = frozenset(
    {
        *_R5_SUPPLIED_IDENTITY_FIELDS,
        "owner_gid",
        "size",
        "mtime_ns",
        "ctime_ns",
        "parent_path",
        "parent_device",
        "parent_inode",
        "parent_owner_uid",
        "parent_owner_gid",
        "parent_mode",
    }
)

ArchivalValidator = Callable[
    [Path, Path],
    Mapping[str, object],
]
PhaseGuardValidator = Callable[[], Mapping[str, object]]
EvidenceWriter = Callable[
    [Mapping[str, object]],
    tuple[Mapping[str, object], Mapping[str, object]],
]
EvidenceReader = Callable[
    [],
    tuple[Mapping[str, object], Mapping[str, object]],
]
LaneStateReader = Callable[[], Mapping[str, bool]]


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
        "for_preaccess_compat_c5"
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
        "stable_fingerprint",
        "utc_now",
        "validate_environment_audit_contract",
        "validate_environment_policy",
        "validate_environment_stability_receipt",
        "verify_sealed_receipt_evidence",
    }
)
_FROZEN_FORBIDDEN_API = frozenset({"run_environment_stability_gate"})
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
    removed_api = _FROZEN_MUTATION_API | _FROZEN_FORBIDDEN_API
    mutation_missing = sorted(
        name for name in removed_api if name not in module.__dict__
    )
    if missing or mutation_missing:
        raise PermissionError(
            "frozen runtime environment API generation changed"
        )
    for name in removed_api:
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
            for mutation_name in removed_api
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
            for name in (_FROZEN_MUTATION_API | _FROZEN_FORBIDDEN_API)
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
            for mutation_name in (
                _FROZEN_MUTATION_API | _FROZEN_FORBIDDEN_API
            )
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
        "policy": (C5_POLICY_PATH, "policy_fingerprint"),
        "stability": (
            C5_STABILITY_PATH,
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
            raise PermissionError("required c5 live roots are absent")
        return {}
    if set(live_roots) != required:
        raise PermissionError("c5 live-root lane changed")
    roots: dict[str, dict[str, object]] = {}
    for name in sorted(required):
        try:
            path, fingerprint_field = _LIVE_ROOT_SPECS[name]
            supplied = live_roots[name]
        except (KeyError, TypeError) as error:
            raise PermissionError(
                "c5 live-root lane is malformed"
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

    _require_c5_namespace()

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
    c5_contract = replace(
        old_contract,
        target_unit_id=C5_TARGET_UNIT,
        require_target_ready=True,
    )
    c5_contract = _frozen.validate_environment_audit_contract(c5_contract)
    old_projection = asdict(old_contract)
    c5_projection = asdict(c5_contract)
    for field in fields(old_contract):
        if field.name in _HANDOFF_FIELDS:
            continue
        if not _frozen._deep_exact_equal(
            old_projection[field.name],
            c5_projection[field.name],
        ):
            raise PermissionError(
                f"c5 scope handoff changed field:{field.name}"
            )
    if (
        c5_contract.target_unit_id != C5_TARGET_UNIT
        or c5_contract.require_target_ready is not True
    ):
        raise PermissionError("c5 scope handoff did not close target readiness")
    return old_contract, c5_contract, dict(roots)


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


def validate_c5_realization_archival(
    archival: Mapping[str, object],
    *,
    contract: object,
) -> dict[str, object]:
    """Bind the c5 contract to one exact archival realization PASS."""

    try:
        authorization = dict(archival["authorization"])
        receipt = dict(archival["receipt"])
        authorization_identity = _validate_r5_archival_root(
            archival["authorization_identity"],
            path=C5_REALIZATION_AUTHORIZATION_PATH,
        )
        receipt_identity = _validate_r5_archival_root(
            archival["receipt_identity"],
            path=C5_REALIZATION_RECEIPT_PATH,
        )
        manager = dict(receipt["manager_generation"])
        fragment = dict(receipt["fragment_identity"])
        shadow = dict(receipt["full_static_shadow"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("c5 realization archival structure is malformed") from error
    _no_payload(authorization, name="c5 realization authorization")
    _no_payload(receipt, name="c5 realization receipt")
    expected_manager = _manager_generation_from_contract(contract)
    expected_shadow = {
        "Id": C5_TARGET_UNIT,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "Restart": "no",
        "NRestarts": "0",
        "FragmentPath": str(C5_FRAGMENT_PATH),
    }
    if (
        authorization.get("unit_name") != C5_TARGET_UNIT
        or receipt.get("unit_name") != C5_TARGET_UNIT
        or receipt.get("passed") is not True
        or receipt.get("static") is not True
        or receipt.get("enabled") is not False
        or receipt.get("started") is not False
        or receipt.get("authorization_file_sha256")
        != authorization_identity["file_sha256"]
        or not _frozen._deep_exact_equal(
            authorization.get("manager_generation"),
            expected_manager,
        )
        or not _frozen._deep_exact_equal(manager, expected_manager)
        or fragment.get("path") != str(C5_FRAGMENT_PATH)
        or shadow.get("FragmentPath") != fragment.get("path")
        or any(
            shadow.get(key) != value
            for key, value in expected_shadow.items()
        )
    ):
        raise PermissionError(
            "c5 realization is not an exact live-ready archival PASS"
        )
    return {
        "authorization": authorization,
        "receipt": receipt,
        "authorization_identity": authorization_identity,
        "receipt_identity": receipt_identity,
        "fragment": fragment,
        "shadow": shadow,
    }


def _validate_r5_archival_root(
    value: object,
    *,
    path: Path,
) -> dict[str, object]:
    """Validate one immutable R5 evidence-file generation root."""

    if not isinstance(value, Mapping):
        raise PermissionError("r5 archival evidence root is malformed")
    root = dict(value)
    integer_fields = _R5_ARCHIVAL_ROOT_FIELDS - {
        "path",
        "resolved_path",
        "path_is_symlink",
        "file_sha256",
        "parent_path",
    }
    target = path.absolute()
    if (
        set(root) != _R5_ARCHIVAL_ROOT_FIELDS
        or Path(str(root.get("path", ""))).absolute() != target
        or root.get("resolved_path") != str(target)
        or root.get("path_is_symlink") is not False
        or not _is_sha256(root.get("file_sha256"))
        or any(
            isinstance(root.get(field), bool)
            or not isinstance(root.get(field), int)
            or root[field] < 0
            for field in integer_fields
        )
        or root.get("owner_uid") != os.getuid()
        or root.get("mode") != 0o444
        or root.get("nlink") != 1
        or root.get("parent_path") != str(target.parent)
        or root.get("parent_owner_uid") != os.getuid()
        or root.get("parent_mode", 0) & 0o022
    ):
        raise PermissionError("r5 archival evidence root changed")
    return root


def _bind_r5_archival_root(
    path: Path,
    supplied_value: object,
) -> dict[str, object]:
    """Extend R5's sealed identity with size and nanosecond timestamps."""

    if not isinstance(supplied_value, Mapping):
        raise PermissionError("r5 supplied evidence identity is malformed")
    supplied = dict(supplied_value)
    if set(supplied) != _R5_SUPPLIED_IDENTITY_FIELDS:
        raise PermissionError("r5 supplied evidence identity changed")
    target = path.absolute()
    raw, generation = _stable_source_bytes(target)
    expected_supplied = {
        "path": str(target),
        "resolved_path": str(target),
        "path_is_symlink": False,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "device": generation["st_dev"],
        "inode": generation["st_ino"],
        "owner_uid": generation["st_uid"],
        "mode": stat.S_IMODE(generation["st_mode"]),
        "nlink": generation["st_nlink"],
    }
    if not _frozen._deep_exact_equal(supplied, expected_supplied):
        raise PermissionError("r5 evidence changed after archival validation")
    root = {
        **expected_supplied,
        "owner_gid": generation["st_gid"],
        "size": generation["st_size"],
        "mtime_ns": generation["st_mtime_ns"],
        "ctime_ns": generation["st_ctime_ns"],
        "parent_path": str(target.parent),
        "parent_device": generation["parent_st_dev"],
        "parent_inode": generation["parent_st_ino"],
        "parent_owner_uid": generation["parent_st_uid"],
        "parent_owner_gid": generation["parent_st_gid"],
        "parent_mode": stat.S_IMODE(generation["parent_st_mode"]),
    }
    return _validate_r5_archival_root(root, path=target)


def _load_verified_c5_bridge() -> ModuleType:
    _require_c5_namespace()
    # R5 is the sole producer validator for its authorization/receipt chain.
    # Its fixed B5 source pin is independently checked here before R5 bytes
    # are compiled, so neither producer can be silently swapped.
    _verify_fixed_source(
        C5_COMPATIBILITY_BRIDGE_PATH,
        C5_COMPATIBILITY_BRIDGE_SHA256,
        name="b5 compatibility bridge",
    )
    if (
        C5_BRIDGE_SHA256 == "__TO_BE_FROZEN__"
        or len(C5_BRIDGE_SHA256) != 64
        or any(character not in "0123456789abcdef"
               for character in C5_BRIDGE_SHA256)
    ):
        raise PermissionError("c5 realization bridge is not frozen")
    raw, identity = _stable_source_bytes(C5_BRIDGE_PATH)
    if hashlib.sha256(raw).hexdigest() != C5_BRIDGE_SHA256:
        raise PermissionError("c5 realization bridge source changed")
    name = (
        "tools._cure_lite_v24_actual_unit_realization_"
        "preaccess_compat_c5_verified_for_environment"
    )
    module = ModuleType(name)
    module.__file__ = str(C5_BRIDGE_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(raw, str(C5_BRIDGE_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
        if (
            Path(module.COMPAT_BRIDGE_SOURCE_PATH).absolute()
            != C5_COMPATIBILITY_BRIDGE_PATH.absolute()
            or module.COMPAT_BRIDGE_SOURCE_SHA256
            != C5_COMPATIBILITY_BRIDGE_SHA256
            or module.COMPAT_UNIT != C5_TARGET_UNIT
            or not callable(module.validate_archival_realization_chain)
        ):
            raise PermissionError("R5/B5 producer interface changed")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    raw_after, identity_after = _stable_source_bytes(C5_BRIDGE_PATH)
    if (
        hashlib.sha256(raw_after).hexdigest() != C5_BRIDGE_SHA256
        or identity_after != identity
    ):
        sys.modules.pop(name, None)
        raise PermissionError(
            "c5 realization bridge generation changed while loading"
        )
    return module


def _production_archival_validator(
    authorization_path: Path,
    receipt_path: Path,
) -> Mapping[str, object]:
    bridge = _load_verified_c5_bridge()
    validator = getattr(bridge, "validate_archival_realization_chain", None)
    if not callable(validator):
        raise PermissionError("c5 archival realization validator is absent")
    archival = validator(authorization_path, receipt_path)
    if not isinstance(archival, Mapping):
        raise PermissionError("c5 archival realization validator returned no closure")
    result = dict(archival)
    result["authorization_identity"] = _bind_r5_archival_root(
        authorization_path,
        result.get("authorization_identity"),
    )
    result["receipt_identity"] = _bind_r5_archival_root(
        receipt_path,
        result.get("receipt_identity"),
    )
    return result


def _require_frozen_sha256(value: object, *, name: str) -> str:
    if (
        value == "__TO_BE_FROZEN__"
        or not isinstance(value, str)
        or not _is_sha256(value)
    ):
        raise PermissionError(f"{name} is not frozen")
    return value


def _verify_fixed_source(path: Path, digest: object, *, name: str) -> None:
    expected = _require_frozen_sha256(digest, name=name)
    raw, identity = _stable_source_bytes(path)
    raw_after, identity_after = _stable_source_bytes(path)
    if (
        hashlib.sha256(raw).hexdigest() != expected
        or raw_after != raw
        or identity_after != identity
    ):
        raise PermissionError(f"{name} generation changed")


def _load_verified_c4_failure_terminalizer() -> ModuleType:
    """Load the fixed C4 terminal producer's archival validator only."""

    expected = _require_frozen_sha256(
        C4_FAILURE_TERMINALIZER_SHA256,
        name="c4 receipt-seal terminalizer",
    )
    raw, identity = _stable_source_bytes(C4_FAILURE_TERMINALIZER_PATH)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise PermissionError("c4 receipt-seal terminalizer source changed")
    name = (
        "tools._cure_lite_v24_preaccess_compat_c4_receipt_seal_"
        "failure_terminal_verified_for_c5"
    )
    module = ModuleType(name)
    module.__file__ = str(C4_FAILURE_TERMINALIZER_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                raw,
                str(C4_FAILURE_TERMINALIZER_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    raw_after, identity_after = _stable_source_bytes(
        C4_FAILURE_TERMINALIZER_PATH
    )
    if (
        hashlib.sha256(raw_after).hexdigest() != expected
        or raw_after != raw
        or identity_after != identity
    ):
        sys.modules.pop(name, None)
        raise PermissionError(
            "c4 receipt-seal terminalizer changed while loading"
        )
    return module


def _validate_c5_phase_guard(
    value: Mapping[str, object],
    *,
    require_live_absence: bool = False,
) -> dict[str, object]:
    try:
        guard = dict(value)
        terminal_root = dict(guard["c4_failure_terminal_root"])
        unit_state = dict(guard["c4_unit_state"])
        absent_paths = dict(guard["c4_absent_outputs"])
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError("c5 phase guard is malformed") from error
    expected_terminal_sha = _require_frozen_sha256(
        C4_FAILURE_TERMINAL_SHA256,
        name="c4 receipt-seal failure terminal",
    )
    expected_terminal_fingerprint = _require_frozen_sha256(
        C4_FAILURE_TERMINAL_FINGERPRINT,
        name="c4 receipt-seal failure terminal fingerprint",
    )
    expected_absent = {
        name: str(path.absolute())
        for name, path in _C4_REQUIRED_ABSENT_PATHS.items()
    }
    expected_state = {
        "Id": C4_TARGET_UNIT,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "FragmentPath": str(C4_FRAGMENT_PATH),
        "InvocationID": "",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
    }
    required_root_fields = {
        "path",
        "file_sha256",
        "terminal_fingerprint",
        "schema_version",
        "device",
        "inode",
        "owner_uid",
        "mode",
        "nlink",
        "size",
        "mtime_ns",
    }
    if (
        set(guard)
        != {
            "c4_failure_terminal_root",
            "c4_unit_state",
            "c4_absent_outputs",
            "D_R_payload_accessed",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
        }
        or not required_root_fields.issubset(terminal_root)
        or Path(str(terminal_root.get("path", ""))).absolute()
        != C4_FAILURE_TERMINAL_PATH.absolute()
        or not _is_sha256(terminal_root.get("file_sha256"))
        or not _is_sha256(terminal_root.get("terminal_fingerprint"))
        or terminal_root.get("file_sha256")
        != expected_terminal_sha
        or terminal_root.get("terminal_fingerprint")
        != expected_terminal_fingerprint
        or "compat-c4-receipt-seal-failure" not in str(
            terminal_root.get("schema_version", "")
        )
        or terminal_root.get("owner_uid") != os.getuid()
        or terminal_root.get("mode") != 0o444
        or terminal_root.get("nlink") != 1
        or unit_state != expected_state
        or absent_paths != expected_absent
        or require_live_absence
        and any(
            os.path.lexists(path)
            for path in _C4_REQUIRED_ABSENT_PATHS.values()
        )
    ):
        raise PermissionError("c4 terminal/inert/absence guard changed")
    _no_payload(guard, name="c5 phase guard")
    return guard


def _production_c5_phase_guard() -> Mapping[str, object]:
    terminalizer = _load_verified_c4_failure_terminalizer()
    _require_frozen_sha256(
        C4_FAILURE_TERMINAL_SHA256,
        name="c4 receipt-seal failure terminal",
    )
    _require_frozen_sha256(
        C4_FAILURE_TERMINAL_FINGERPRINT,
        name="c4 receipt-seal failure terminal fingerprint",
    )
    validator = getattr(terminalizer, "validate_archival", None)
    state_reader = getattr(terminalizer, "_read_unit_state", None)
    terminal_absences = getattr(terminalizer, "ABSENT_OUTPUT_PATHS", None)
    if (
        not callable(validator)
        or not callable(state_reader)
        or not isinstance(terminal_absences, Mapping)
        or {
            str(name): Path(path).absolute()
            for name, path in terminal_absences.items()
        }
        != {
            name: path.absolute()
            for name, path in _C4_REQUIRED_ABSENT_PATHS.items()
        }
    ):
        raise PermissionError("c4 failure terminalizer read API changed")
    before = {
        name: str(path)
        for name, path in _C4_REQUIRED_ABSENT_PATHS.items()
        if os.path.lexists(path)
    }
    terminal_payload, terminal_root = validator(C4_FAILURE_TERMINAL_PATH)
    payload_observation = terminal_payload.get("payload_observation")
    if not isinstance(payload_observation, Mapping):
        raise PermissionError("c4 failure terminal payload closure changed")
    _no_payload(
        payload_observation,
        name="c4 failure terminal payload observation",
    )
    unit_state = state_reader(C4_TARGET_UNIT, expected="static")
    after = {
        name: str(path)
        for name, path in _C4_REQUIRED_ABSENT_PATHS.items()
        if os.path.lexists(path)
    }
    if before or after:
        raise PermissionError("c4 forbidden future output appeared")
    return _validate_c5_phase_guard(
        {
            "c4_failure_terminal_root": terminal_root,
            "c4_unit_state": unit_state,
            "c4_absent_outputs": {
                name: str(path.absolute())
                for name, path in _C4_REQUIRED_ABSENT_PATHS.items()
            },
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        },
        require_live_absence=True,
    )


def _resolve_archival(
    validator: ArchivalValidator,
    *,
    contract: object,
) -> dict[str, object]:
    archival = validator(
        C5_REALIZATION_AUTHORIZATION_PATH,
        C5_REALIZATION_RECEIPT_PATH,
    )
    if not isinstance(archival, Mapping):
        raise PermissionError("c5 archival validator returned no closure")
    return validate_c5_realization_archival(archival, contract=contract)



def _validate_c5_policy_contract(
    policy: Mapping[str, object],
    *,
    c5_contract: object,
    roots: Mapping[str, object],
    archival: Mapping[str, object],
) -> dict[str, object]:
    value = _frozen.validate_environment_policy(policy)
    expected = _frozen.build_environment_policy(
        c5_contract,
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
        raise PermissionError("c5 policy differs from exact handoff contract")
    _no_payload(value, name="c5 environment policy")
    if not (
        _strict_utc(
            archival["receipt"]["created_at_utc"],
            name="c5 realization receipt",
        )
        < _strict_utc(
            value["created_at_utc"],
            name="c5 policy",
        )
    ):
        raise PermissionError("c5 policy predates unit realization")
    return dict(value)


_CONTRACT_TUPLE_FIELDS = frozenset(
    {
        "conflict_unit_ids",
        "dependency_unit_ids",
        "allowed_failed_unit_ids",
        "expected_failed_unit_ids",
        "allowed_unit_ids",
        "allowed_manager_states",
    }
)


def _contract_from_mapping(value: Mapping[str, object]) -> object:
    projection = dict(value)
    for name in _CONTRACT_TUPLE_FIELDS:
        projection[name] = tuple(projection[name])
    projection["cleanup_nrestarts_baseline"] = tuple(
        (str(item[0]), str(item[1]))
        for item in projection["cleanup_nrestarts_baseline"]
    )
    projection["activation_guard"] = dict(projection["activation_guard"])
    return _frozen.validate_environment_audit_contract(
        _frozen.EnvironmentAuditContract(**projection)
    )


def _normalized_contract_value(contract: object) -> dict[str, object]:
    return json.loads(_canonical_json(asdict(contract)))


def _validate_custom_live_root(
    value: Mapping[str, object],
    *,
    path: Path,
    fingerprint_field: str,
) -> dict[str, object]:
    return _validate_sealed_root(
        value,
        path=path,
        fingerprint_field=fingerprint_field,
    )


def validate_c5_scope_handoff(
    value: Mapping[str, object],
    *,
    expected_old_contract: object | None = None,
    expected_c5_contract: object | None = None,
    expected_roots: Mapping[str, object] | None = None,
    expected_archival: Mapping[str, object] | None = None,
    expected_phase_guard: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate one closed, metadata-only C4-to-C5 scope handoff."""

    payload = json.loads(_canonical_json(dict(value)))
    body = dict(payload)
    fingerprint = body.pop("scope_handoff_fingerprint", None)
    expected_keys = {
        "schema_version",
        "created_at_utc",
        "runtime_compatibility_id",
        "old_contract",
        "c5_contract",
        "changed_fields",
        "root_evidence",
        "realization",
        "phase_guard",
        "toolchain",
        "sampling",
        "payload_authority",
        "D_R_payload_accessed",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "gpu_accessed",
        "training_started",
        "materialization_consumed",
    }
    if (
        set(body) != expected_keys
        or payload.get("schema_version") != C5_HANDOFF_SCHEMA
        or payload.get("runtime_compatibility_id") != "c5"
        or payload.get("changed_fields")
        != ["require_target_ready", "target_unit_id"]
        or fingerprint != _stable_fingerprint(body)
        or payload.get("payload_authority") != "none"
    ):
        raise PermissionError("c5 scope handoff schema changed")
    _strict_utc(payload.get("created_at_utc"), name="c5 scope handoff")
    try:
        old_contract = _contract_from_mapping(payload["old_contract"])
        c5_contract = _contract_from_mapping(payload["c5_contract"])
        roots = dict(payload["root_evidence"])
        archival = dict(payload["realization"])
        phase_guard = _validate_c5_phase_guard(payload["phase_guard"])
        toolchain = dict(payload["toolchain"])
        sampling = dict(payload["sampling"])
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError("c5 scope handoff content is malformed") from error
    old_projection = asdict(old_contract)
    c5_projection = asdict(c5_contract)
    if (
        old_contract.target_unit_id != OLD_TARGET_UNIT
        or old_contract.require_target_ready is not False
        or c5_contract.target_unit_id != C5_TARGET_UNIT
        or c5_contract.require_target_ready is not True
        or set(roots)
        != {"precleanup_inventory_receipt", "cleanup_receipt"}
        or Path(str(roots["precleanup_inventory_receipt"].get("path", ""))).absolute()
        != PRECLEANUP_PATH.absolute()
        or Path(str(roots["cleanup_receipt"].get("path", ""))).absolute()
        != CLEANUP_RECEIPT_PATH.absolute()
        or sampling
        != {
            "sample_count": SAMPLE_COUNT,
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        }
        or not toolchain
        or any(
            not _frozen._deep_exact_equal(
                old_projection[field.name],
                c5_projection[field.name],
            )
            for field in fields(old_contract)
            if field.name not in _HANDOFF_FIELDS
        )
    ):
        raise PermissionError("c5 scope handoff semantics changed")
    archival = validate_c5_realization_archival(
        archival,
        contract=c5_contract,
    )
    if not (
        _strict_utc(
            archival["receipt"]["created_at_utc"],
            name="c5 realization receipt",
        )
        < _strict_utc(
            payload["created_at_utc"],
            name="c5 scope handoff",
        )
    ):
        raise PermissionError("c5 scope handoff predates realization")
    comparisons = (
        (expected_old_contract, old_contract),
        (expected_c5_contract, c5_contract),
        (expected_roots, roots),
        (expected_archival, archival),
        (expected_phase_guard, phase_guard),
    )
    for expected, observed in comparisons:
        if expected is None:
            continue
        left = (
            _normalized_contract_value(expected)
            if hasattr(expected, "target_unit_id")
            else expected
        )
        right = (
            _normalized_contract_value(observed)
            if hasattr(observed, "target_unit_id")
            else observed
        )
        if not _frozen._deep_exact_equal(left, right):
            raise PermissionError("c5 scope handoff live binding changed")
    _no_payload(payload, name="c5 scope handoff")
    return payload


def build_c5_scope_handoff_in_memory(
    *,
    realization_validator: ArchivalValidator,
    phase_guard_validator: PhaseGuardValidator,
    prepare: Callable[..., tuple[object, dict[str, object]]] | None = None,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ]
    | None = None,
    toolchain_reader: Callable[[], Mapping[str, object]] | None = None,
    clock: Callable[[], str] | None = None,
) -> dict[str, object]:
    old_contract, c5_contract, roots = replay_old_scope_and_handoff(
        prepare=prepare,
        activation_guard_reader=activation_guard_reader,
    )
    archival = _resolve_archival(
        realization_validator,
        contract=c5_contract,
    )
    phase_guard = _validate_c5_phase_guard(phase_guard_validator())
    toolchain = dict(
        (toolchain_reader or _frozen.current_runtime_toolchain_binding)()
    )
    body: dict[str, object] = {
        "schema_version": C5_HANDOFF_SCHEMA,
        "created_at_utc": (clock or _frozen.utc_now)(),
        "runtime_compatibility_id": "c5",
        "old_contract": _normalized_contract_value(old_contract),
        "c5_contract": _normalized_contract_value(c5_contract),
        "changed_fields": ["require_target_ready", "target_unit_id"],
        "root_evidence": dict(roots),
        "realization": archival,
        "phase_guard": phase_guard,
        "toolchain": toolchain,
        "sampling": {
            "sample_count": SAMPLE_COUNT,
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        },
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
        "training_started": False,
        "materialization_consumed": False,
    }
    return validate_c5_scope_handoff(
        {
            **body,
            "scope_handoff_fingerprint": _stable_fingerprint(body),
        },
        expected_old_contract=old_contract,
        expected_c5_contract=c5_contract,
        expected_roots=roots,
        expected_archival=archival,
        expected_phase_guard=phase_guard,
    )


def build_c5_policy_in_memory(
    scope_handoff: Mapping[str, object],
    *,
    realization_validator: ArchivalValidator,
    prepare: Callable[..., tuple[object, dict[str, object]]] | None = None,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ]
    | None = None,
    toolchain_reader: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build, but do not write, the fixed c5 fresh environment policy."""

    old_contract, c5_contract, roots = replay_old_scope_and_handoff(
        prepare=prepare,
        activation_guard_reader=activation_guard_reader,
    )
    archival = _resolve_archival(
        realization_validator,
        contract=c5_contract,
    )
    validate_c5_scope_handoff(
        scope_handoff,
        expected_old_contract=old_contract,
        expected_c5_contract=c5_contract,
        expected_roots=roots,
        expected_archival=archival,
    )
    if toolchain_reader is None:
        toolchain_reader = _frozen.current_runtime_toolchain_binding
    policy = _frozen.build_environment_policy(
        c5_contract,
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
            name="c5 realization receipt",
        )
        < _strict_utc(policy["created_at_utc"], name="c5 policy")
    ):
        raise PermissionError("c5 policy predates unit realization")
    return policy


def validate_c5_stability_attempt(
    value: Mapping[str, object],
    *,
    expected_scope_handoff_root: Mapping[str, object] | None = None,
    expected_policy_root: Mapping[str, object] | None = None,
    expected_contract: object | None = None,
    expected_roots: Mapping[str, object] | None = None,
    expected_phase_guard: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate the irreversible, pre-first-sample C5 attempt commit."""

    payload = json.loads(_canonical_json(dict(value)))
    body = dict(payload)
    fingerprint = body.pop("stability_attempt_fingerprint", None)
    expected_keys = {
        "schema_version",
        "created_at_utc",
        "command",
        "runtime_compatibility_id",
        "target_unit_id",
        "contract",
        "root_evidence",
        "scope_handoff_root",
        "policy_root",
        "toolchain",
        "phase_guard",
        "sample_count",
        "sample_interval_seconds",
        "automatic_retry_allowed",
        "resume_allowed",
        "payload_authority",
        "D_R_payload_accessed",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "gpu_accessed",
        "training_started",
        "materialization_consumed",
    }
    if (
        set(body) != expected_keys
        or payload.get("schema_version") != C5_STABILITY_ATTEMPT_SCHEMA
        or payload.get("command") != "stability-gate"
        or payload.get("runtime_compatibility_id") != "c5"
        or payload.get("target_unit_id") != C5_TARGET_UNIT
        or payload.get("sample_count") != SAMPLE_COUNT
        or payload.get("sample_interval_seconds")
        != SAMPLE_INTERVAL_SECONDS
        or payload.get("automatic_retry_allowed") is not False
        or payload.get("resume_allowed") is not False
        or payload.get("payload_authority") != "none"
        or fingerprint != _stable_fingerprint(body)
    ):
        raise PermissionError("c5 stability attempt schema changed")
    _strict_utc(payload.get("created_at_utc"), name="c5 stability attempt")
    try:
        contract = _contract_from_mapping(payload["contract"])
        roots = dict(payload["root_evidence"])
        handoff_root = _validate_custom_live_root(
            payload["scope_handoff_root"],
            path=C5_SCOPE_HANDOFF_PATH,
            fingerprint_field="scope_handoff_fingerprint",
        )
        policy_root = _validate_custom_live_root(
            payload["policy_root"],
            path=C5_POLICY_PATH,
            fingerprint_field="policy_fingerprint",
        )
        phase_guard = _validate_c5_phase_guard(payload["phase_guard"])
        toolchain = dict(payload["toolchain"])
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError("c5 stability attempt content is malformed") from error
    if (
        contract.target_unit_id != C5_TARGET_UNIT
        or contract.require_target_ready is not True
        or set(roots)
        != {"precleanup_inventory_receipt", "cleanup_receipt"}
        or not toolchain
    ):
        raise PermissionError("c5 stability attempt semantics changed")
    comparisons = (
        (expected_scope_handoff_root, handoff_root),
        (expected_policy_root, policy_root),
        (
            None
            if expected_contract is None
            else _normalized_contract_value(expected_contract),
            _normalized_contract_value(contract),
        ),
        (expected_roots, roots),
        (expected_phase_guard, phase_guard),
    )
    for expected, observed in comparisons:
        if expected is not None and not _frozen._deep_exact_equal(
            expected,
            observed,
        ):
            raise PermissionError("c5 stability attempt binding changed")
    _no_payload(payload, name="c5 stability attempt")
    return payload


def _build_c5_stability_attempt(
    *,
    contract: object,
    roots: Mapping[str, object],
    scope_handoff_root: Mapping[str, object],
    policy_root: Mapping[str, object],
    toolchain: Mapping[str, object],
    phase_guard: Mapping[str, object],
    clock: Callable[[], str],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": C5_STABILITY_ATTEMPT_SCHEMA,
        "created_at_utc": clock(),
        "command": "stability-gate",
        "runtime_compatibility_id": "c5",
        "target_unit_id": C5_TARGET_UNIT,
        "contract": _normalized_contract_value(contract),
        "root_evidence": dict(roots),
        "scope_handoff_root": dict(scope_handoff_root),
        "policy_root": dict(policy_root),
        "toolchain": dict(toolchain),
        "phase_guard": dict(phase_guard),
        "sample_count": SAMPLE_COUNT,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
        "training_started": False,
        "materialization_consumed": False,
    }
    return validate_c5_stability_attempt(
        {
            **body,
            "stability_attempt_fingerprint": _stable_fingerprint(
                body
            ),
        },
        expected_scope_handoff_root=scope_handoff_root,
        expected_policy_root=policy_root,
        expected_contract=contract,
        expected_roots=roots,
        expected_phase_guard=phase_guard,
    )


def validate_c5_stability_terminal(
    value: Mapping[str, object],
    *,
    expected_attempt_root: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate the failure-only C5 stability terminal."""

    payload = json.loads(_canonical_json(dict(value)))
    body = dict(payload)
    fingerprint = body.pop("stability_terminal_fingerprint", None)
    expected_keys = {
        "schema_version",
        "created_at_utc",
        "command",
        "runtime_compatibility_id",
        "outcome",
        "attempt_root",
        "completed_sample_count",
        "completed_sleep_count",
        "failure_phase",
        "error_type",
        "error_message",
        "passed",
        "blockers",
        "success_stability_absent_at_creation",
        "automatic_retry_allowed",
        "resume_allowed",
        "payload_authority",
        "D_R_payload_accessed",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "gpu_accessed",
        "training_started",
        "materialization_consumed",
    }
    if (
        set(body) != expected_keys
        or payload.get("schema_version") != C5_STABILITY_TERMINAL_SCHEMA
        or payload.get("command") != "stability-gate"
        or payload.get("runtime_compatibility_id") != "c5"
        or payload.get("outcome") != "failed"
        or type(payload.get("completed_sample_count")) is not int
        or not 0 <= payload["completed_sample_count"] <= SAMPLE_COUNT
        or type(payload.get("completed_sleep_count")) is not int
        or not 0 <= payload["completed_sleep_count"] <= 1
        or payload.get("failure_phase")
        not in {"pre-sample", "between-samples", "post-sample", "success-seal"}
        or not isinstance(payload.get("error_type"), str)
        or not payload["error_type"]
        or not isinstance(payload.get("error_message"), str)
        or payload.get("passed") is not False
        or payload.get("blockers") != ["c5_stability_gate_exception"]
        or payload.get("success_stability_absent_at_creation") is not True
        or payload.get("automatic_retry_allowed") is not False
        or payload.get("resume_allowed") is not False
        or payload.get("payload_authority") != "none"
        or fingerprint != _stable_fingerprint(body)
    ):
        raise PermissionError("c5 stability terminal schema changed")
    _strict_utc(payload.get("created_at_utc"), name="c5 stability terminal")
    attempt_root = _validate_custom_live_root(
        payload["attempt_root"],
        path=C5_STABILITY_ATTEMPT_PATH,
        fingerprint_field="stability_attempt_fingerprint",
    )
    if expected_attempt_root is not None and not _frozen._deep_exact_equal(
        attempt_root,
        expected_attempt_root,
    ):
        raise PermissionError("c5 stability terminal attempt root changed")
    _no_payload(payload, name="c5 stability terminal")
    return payload


def _build_c5_stability_terminal(
    *,
    attempt_root: Mapping[str, object],
    completed_sample_count: int,
    completed_sleep_count: int,
    failure_phase: str,
    error: BaseException,
    clock: Callable[[], str],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": C5_STABILITY_TERMINAL_SCHEMA,
        "created_at_utc": clock(),
        "command": "stability-gate",
        "runtime_compatibility_id": "c5",
        "outcome": "failed",
        "attempt_root": dict(attempt_root),
        "completed_sample_count": completed_sample_count,
        "completed_sleep_count": completed_sleep_count,
        "failure_phase": failure_phase,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "passed": False,
        "blockers": ["c5_stability_gate_exception"],
        "success_stability_absent_at_creation": True,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
        "training_started": False,
        "materialization_consumed": False,
    }
    return validate_c5_stability_terminal(
        {
            **body,
            "stability_terminal_fingerprint": _stable_fingerprint(
                body
            ),
        },
        expected_attempt_root=attempt_root,
    )


def _validate_c5_stability_lane_state(
    value: Mapping[str, bool],
    *,
    expected: tuple[bool, bool, bool],
) -> dict[str, bool]:
    state = dict(value)
    if (
        set(state) != {"attempt", "success", "terminal"}
        or any(type(item) is not bool for item in state.values())
        or tuple(state[name] for name in ("attempt", "success", "terminal"))
        != expected
    ):
        raise PermissionError("c5 stability lane state changed")
    return state


def run_c5_stability_in_memory(
    scope_handoff: Mapping[str, object],
    policy: Mapping[str, object],
    *,
    realization_validator: ArchivalValidator,
    phase_guard_validator: PhaseGuardValidator,
    scope_handoff_reader: EvidenceReader,
    policy_reader: EvidenceReader,
    attempt_writer: EvidenceWriter,
    attempt_reader: EvidenceReader,
    success_writer: EvidenceWriter,
    terminal_writer: EvidenceWriter,
    lane_state_reader: LaneStateReader,
    prepare: Callable[..., tuple[object, dict[str, object]]] | None = None,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ]
    | None = None,
    inventory_collector: Callable[..., dict[str, object]] | None = None,
    sleeper: Callable[[float], None] | None = None,
    monotonic_clock: Callable[[], float] | None = None,
    toolchain_reader: Callable[[], Mapping[str, object]] | None = None,
    clock: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Commit once, sample C5 directly, and seal one terminal outcome."""

    if not all(
        callable(candidate)
        for candidate in (
            realization_validator,
            phase_guard_validator,
            scope_handoff_reader,
            policy_reader,
            attempt_writer,
            attempt_reader,
            success_writer,
            terminal_writer,
            lane_state_reader,
        )
    ):
        raise PermissionError("c5 stability closure dependency is absent")
    sleep_once = sleeper or time.sleep
    monotonic = monotonic_clock or time.monotonic
    read_toolchain = (
        toolchain_reader or _frozen.current_runtime_toolchain_binding
    )
    utc_clock = clock or _frozen.utc_now
    attempt_value: dict[str, object] | None = None
    attempt_root: dict[str, object] | None = None
    baseline: dict[str, object] | None = None
    completed_sample_count = 0
    completed_sleep_count = 0
    failure_phase = "pre-sample"

    def checkpoint(*, require_attempt: bool) -> dict[str, object]:
        _require_frozen_environment_source()
        old_contract, c5_contract, roots = replay_old_scope_and_handoff(
            prepare=prepare,
            activation_guard_reader=activation_guard_reader,
        )
        archival = _resolve_archival(
            realization_validator,
            contract=c5_contract,
        )
        current_guard = _validate_c5_phase_guard(
            phase_guard_validator()
        )
        live_handoff, live_handoff_root = scope_handoff_reader()
        live_handoff = validate_c5_scope_handoff(
            live_handoff,
            expected_old_contract=old_contract,
            expected_c5_contract=c5_contract,
            expected_roots=roots,
            expected_archival=archival,
            expected_phase_guard=current_guard,
        )
        live_handoff_root = _validate_custom_live_root(
            live_handoff_root,
            path=C5_SCOPE_HANDOFF_PATH,
            fingerprint_field="scope_handoff_fingerprint",
        )
        live_policy, live_policy_root = policy_reader()
        live_policy_root = _validate_custom_live_root(
            live_policy_root,
            path=C5_POLICY_PATH,
            fingerprint_field="policy_fingerprint",
        )
        live_policy = _validate_c5_policy_contract(
            live_policy,
            c5_contract=c5_contract,
            roots=roots,
            archival=archival,
        )
        current_toolchain = dict(read_toolchain())
        if (
            not _frozen._deep_exact_equal(live_handoff, scope_handoff)
            or not _frozen._deep_exact_equal(live_policy, policy)
            or live_handoff_root.get("scope_handoff_fingerprint")
            != live_handoff.get("scope_handoff_fingerprint")
            or live_policy_root.get("policy_fingerprint")
            != live_policy.get("policy_fingerprint")
            or not _frozen._deep_exact_equal(
                live_handoff.get("toolchain"),
                current_toolchain,
            )
            or not _frozen._deep_exact_equal(
                live_policy.get("toolchain"),
                current_toolchain,
            )
        ):
            raise PermissionError("c5 checkpoint policy/source binding changed")
        observed = {
            "old_contract": _normalized_contract_value(old_contract),
            "c5_contract": _normalized_contract_value(c5_contract),
            "roots": dict(roots),
            "archival": archival,
            "phase_guard": current_guard,
            "scope_handoff_root": live_handoff_root,
            "policy_root": live_policy_root,
            "toolchain": current_toolchain,
        }
        nonlocal baseline
        if baseline is None:
            baseline = json.loads(_canonical_json(observed))
        elif not _frozen._deep_exact_equal(observed, baseline):
            raise PermissionError("c5 checkpoint generation drifted")
        if require_attempt:
            if attempt_value is None or attempt_root is None:
                raise PermissionError("c5 committed attempt is unavailable")
            live_attempt, live_attempt_root = attempt_reader()
            live_attempt = validate_c5_stability_attempt(
                live_attempt,
                expected_scope_handoff_root=live_handoff_root,
                expected_policy_root=live_policy_root,
                expected_contract=c5_contract,
                expected_roots=roots,
                expected_phase_guard=current_guard,
            )
            live_attempt_root = _validate_custom_live_root(
                live_attempt_root,
                path=C5_STABILITY_ATTEMPT_PATH,
                fingerprint_field="stability_attempt_fingerprint",
            )
            if (
                not _frozen._deep_exact_equal(live_attempt, attempt_value)
                or not _frozen._deep_exact_equal(
                    live_attempt_root,
                    attempt_root,
                )
                or live_attempt_root.get("stability_attempt_fingerprint")
                != live_attempt.get("stability_attempt_fingerprint")
            ):
                raise PermissionError("c5 committed attempt changed")
        _require_frozen_environment_source()
        return observed

    try:
        _validate_c5_stability_lane_state(
            lane_state_reader(),
            expected=(False, False, False),
        )
        pre = checkpoint(require_attempt=False)
        c5_contract = _contract_from_mapping(pre["c5_contract"])
        attempt_value = _build_c5_stability_attempt(
            contract=c5_contract,
            roots=pre["roots"],
            scope_handoff_root=pre["scope_handoff_root"],
            policy_root=pre["policy_root"],
            toolchain=pre["toolchain"],
            phase_guard=pre["phase_guard"],
            clock=utc_clock,
        )
        if not (
            _strict_utc(policy["created_at_utc"], name="c5 policy")
            < _strict_utc(
                attempt_value["created_at_utc"],
                name="c5 stability attempt",
            )
        ):
            raise PermissionError("c5 stability attempt predates policy")
        written_attempt, written_attempt_root = attempt_writer(attempt_value)
        written_attempt = validate_c5_stability_attempt(
            written_attempt,
            expected_scope_handoff_root=pre["scope_handoff_root"],
            expected_policy_root=pre["policy_root"],
            expected_contract=c5_contract,
            expected_roots=pre["roots"],
            expected_phase_guard=pre["phase_guard"],
        )
        attempt_root = _validate_custom_live_root(
            written_attempt_root,
            path=C5_STABILITY_ATTEMPT_PATH,
            fingerprint_field="stability_attempt_fingerprint",
        )
        if (
            not _frozen._deep_exact_equal(written_attempt, attempt_value)
            or attempt_root.get("stability_attempt_fingerprint")
            != written_attempt.get("stability_attempt_fingerprint")
        ):
            raise PermissionError("c5 stability attempt seal changed")
        _validate_c5_stability_lane_state(
            lane_state_reader(),
            expected=(True, False, False),
        )
        checkpoint(require_attempt=True)

        audit_kwargs: dict[str, object] = {}
        if inventory_collector is not None:
            audit_kwargs["inventory_collector"] = inventory_collector
        if activation_guard_reader is not None:
            audit_kwargs["activation_guard_reader"] = (
                activation_guard_reader
            )
        samples: list[dict[str, object]] = []
        sample_monotonic_seconds: list[float] = []
        samples.append(
            _frozen.audit_environment_once(c5_contract, **audit_kwargs)
        )
        completed_sample_count = 1
        sample_monotonic_seconds.append(float(monotonic()))
        failure_phase = "between-samples"
        sleep_once(SAMPLE_INTERVAL_SECONDS)
        completed_sleep_count = 1
        checkpoint(require_attempt=True)
        samples.append(
            _frozen.audit_environment_once(c5_contract, **audit_kwargs)
        )
        completed_sample_count = 2
        sample_monotonic_seconds.append(float(monotonic()))
        failure_phase = "post-sample"
        post = checkpoint(require_attempt=True)
        root_evidence = dict(post["roots"])
        root_evidence["policy"] = dict(post["policy_root"])
        stability = _frozen.evaluate_environment_stability(
            c5_contract,
            root_evidence,
            samples,
            sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
            sample_monotonic_seconds=sample_monotonic_seconds,
        )
        stability = _frozen.validate_environment_stability_receipt(
            stability
        )
        if stability.get("passed") is not True or stability.get("blockers") != []:
            raise PermissionError("c5 direct stability evaluation failed")
        validate_c5_environment_closure(
            scope_handoff,
            attempt_value,
            policy,
            stability,
            None,
            archival=post["archival"],
            c5_contract=c5_contract,
        )
        failure_phase = "success-seal"
        sealed_stability, stability_root = success_writer(stability)
        sealed_stability = _frozen.validate_environment_stability_receipt(
            sealed_stability
        )
        stability_root = _validate_custom_live_root(
            stability_root,
            path=C5_STABILITY_PATH,
            fingerprint_field="stability_receipt_fingerprint",
        )
        if (
            not _frozen._deep_exact_equal(sealed_stability, stability)
            or stability_root.get("stability_receipt_fingerprint")
            != sealed_stability.get("stability_receipt_fingerprint")
        ):
            raise PermissionError("c5 success stability seal changed")
        _validate_c5_stability_lane_state(
            lane_state_reader(),
            expected=(True, True, False),
        )
        validate_c5_environment_closure(
            scope_handoff,
            attempt_value,
            policy,
            sealed_stability,
            None,
            archival=post["archival"],
            c5_contract=c5_contract,
        )
        return dict(sealed_stability)
    except BaseException as error:
        if attempt_root is not None:
            try:
                state = _validate_c5_stability_lane_state(
                    lane_state_reader(),
                    expected=(True, False, False),
                )
            except BaseException:
                state = dict(lane_state_reader())
            if state == {
                "attempt": True,
                "success": False,
                "terminal": False,
            }:
                terminal = _build_c5_stability_terminal(
                    attempt_root=attempt_root,
                    completed_sample_count=completed_sample_count,
                    completed_sleep_count=completed_sleep_count,
                    failure_phase=failure_phase,
                    error=error,
                    clock=utc_clock,
                )
                try:
                    sealed_terminal, terminal_root = terminal_writer(
                        terminal
                    )
                    validate_c5_stability_terminal(
                        sealed_terminal,
                        expected_attempt_root=attempt_root,
                    )
                    terminal_root = _validate_custom_live_root(
                        terminal_root,
                        path=C5_STABILITY_TERMINAL_PATH,
                        fingerprint_field="stability_terminal_fingerprint",
                    )
                    if (
                        terminal_root.get(
                            "stability_terminal_fingerprint"
                        )
                        != sealed_terminal.get(
                            "stability_terminal_fingerprint"
                        )
                    ):
                        raise PermissionError(
                            "c5 failure terminal seal changed"
                        )
                    _validate_c5_stability_lane_state(
                        lane_state_reader(),
                        expected=(True, False, True),
                    )
                except BaseException as terminal_error:
                    terminal_error.add_note(
                        "C5 ATTEMPT is consumed; same-generation retry is forbidden"
                    )
                    raise terminal_error from error
        raise


def build_c5_postcleanup_in_memory(
    scope_handoff: Mapping[str, object],
    stability_attempt: Mapping[str, object],
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
    _old, c5_contract, _roots = replay_old_scope_and_handoff(
        prepare=prepare,
        activation_guard_reader=activation_guard_reader,
    )
    archival = _resolve_archival(
        realization_validator,
        contract=c5_contract,
    )
    validate_c5_environment_closure(
        scope_handoff,
        stability_attempt,
        policy,
        stability,
        None,
        archival=archival,
        c5_contract=c5_contract,
        live_roots=verified_live_roots,
    )
    samples = list(stability["samples"])
    inventory = json.loads(
        _canonical_json(samples[-1]["inventory"])
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
        "receipt_fingerprint": _stable_fingerprint(body),
    }
    validate_c5_environment_closure(
        scope_handoff,
        stability_attempt,
        policy,
        stability,
        postcleanup,
        archival=archival,
        c5_contract=c5_contract,
        live_roots=verified_live_roots,
    )
    return postcleanup


def _normalized_contract(contract: object) -> dict[str, object]:
    return json.loads(_canonical_json(asdict(contract)))


def validate_c5_environment_closure(
    scope_handoff: Mapping[str, object],
    stability_attempt: Mapping[str, object],
    policy: Mapping[str, object],
    stability: Mapping[str, object],
    postcleanup: Mapping[str, object] | None,
    *,
    archival: Mapping[str, object],
    c5_contract: object,
    live_roots: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Validate all cross-bindings omitted by the generic frozen validator.

    Omitting ``live_roots`` is an in-memory-only convenience when neither
    fixed lane exists.  Once a fixed lane exists, this function derives and
    verifies the applicable sealed root instead of silently downgrading to an
    in-memory validation.
    """

    if os.path.lexists(C5_STABILITY_TERMINAL_PATH):
        raise PermissionError(
            "c5 PASS stability and failure terminal are mutually exclusive"
        )
    verified_live_roots = None
    live_required = frozenset()
    fixed_handoff = os.path.lexists(C5_SCOPE_HANDOFF_PATH)
    fixed_attempt = os.path.lexists(C5_STABILITY_ATTEMPT_PATH)
    fixed_handoff_root: dict[str, object] | None = None
    fixed_attempt_root: dict[str, object] | None = None
    fixed_state = (
        os.path.lexists(C5_POLICY_PATH),
        os.path.lexists(C5_STABILITY_PATH),
    )
    fixed_postcleanup = os.path.lexists(C5_POSTCLEANUP_PATH)
    fixed_postcleanup_root: dict[str, object] | None = None
    if fixed_state == (False, True):
        raise PermissionError(
            "c5 closure fixed live-root state is partial"
        )
    if fixed_attempt and not fixed_handoff:
        raise PermissionError("c5 fixed attempt has no scope handoff")
    if any(fixed_state) or fixed_postcleanup:
        if not fixed_handoff or not fixed_attempt:
            raise PermissionError(
                "c5 fixed environment predecessors are incomplete"
            )
    if fixed_handoff:
        live_handoff, fixed_handoff_root = _read_live_sealed(
            C5_SCOPE_HANDOFF_PATH,
            fingerprint_field="scope_handoff_fingerprint",
        )
        if not _frozen._deep_exact_equal(live_handoff, scope_handoff):
            raise PermissionError("c5 fixed scope handoff changed")
    if fixed_attempt:
        live_attempt, fixed_attempt_root = _read_live_sealed(
            C5_STABILITY_ATTEMPT_PATH,
            fingerprint_field="stability_attempt_fingerprint",
        )
        if not _frozen._deep_exact_equal(
            live_attempt,
            stability_attempt,
        ):
            raise PermissionError("c5 fixed stability attempt changed")
    if fixed_postcleanup and fixed_state != (True, True):
        raise PermissionError(
            "c5 fixed postcleanup predecessors are incomplete"
        )
    if fixed_state == (True, False):
        if postcleanup is not None:
            raise PermissionError(
                "c5 postcleanup fixed live roots are incomplete"
            )
        live_policy, policy_root = _read_live_sealed(
            C5_POLICY_PATH,
            fingerprint_field="policy_fingerprint",
        )
        if not _frozen._deep_exact_equal(live_policy, policy):
            raise PermissionError(
                "c5 fixed policy payload changed"
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
                    "c5 supplied policy root changed"
                )
        live_roots = {"policy": policy_root}
    elif fixed_state == (True, True):
        live_policy, policy_root = _read_live_sealed(
            C5_POLICY_PATH,
            fingerprint_field="policy_fingerprint",
        )
        live_stability, stability_root = _read_live_sealed(
            C5_STABILITY_PATH,
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
                "c5 fixed environment payload changed"
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
                    "c5 closure supplied live-root lane changed"
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
                    "c5 supplied fixed live root changed"
                )
        live_roots = fixed_roots
    if fixed_postcleanup:
        if postcleanup is None:
            raise PermissionError(
                "c5 fixed postcleanup payload was omitted"
            )
        live_postcleanup, fixed_postcleanup_root = _read_live_sealed(
            C5_POSTCLEANUP_PATH,
            fingerprint_field="receipt_fingerprint",
        )
        if not _frozen._deep_exact_equal(
            live_postcleanup,
            postcleanup,
        ):
            raise PermissionError(
                "c5 fixed postcleanup payload changed"
            )
    if live_roots is not None:
        live_required = frozenset(live_roots)
        allowed = (
            {frozenset({"policy"}), frozenset({"policy", "stability"})}
            if postcleanup is None
            else {frozenset({"policy", "stability"})}
        )
        if live_required not in allowed:
            raise PermissionError("c5 closure live-root lane changed")
        verified_live_roots = _verify_exact_live_roots(
            live_roots,
            required=live_required,
        )
    policy_value = _frozen.validate_environment_policy(policy)
    stability_value = _frozen.validate_environment_stability_receipt(
        stability
    )
    archival_value = validate_c5_realization_archival(
        archival,
        contract=c5_contract,
    )
    handoff_value = validate_c5_scope_handoff(
        scope_handoff,
        expected_c5_contract=c5_contract,
        expected_archival=archival_value,
    )
    _no_payload(policy_value, name="c5 environment policy")
    _no_payload(stability_value, name="c5 environment stability")
    scope = dict(policy_value["unit_scope"])
    sampling = dict(policy_value["sampling"])
    contract_value = dict(stability_value["contract"])
    samples = list(stability_value["samples"])
    expected_policy = _frozen.build_environment_policy(
        c5_contract,
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
    attempt_value = validate_c5_stability_attempt(
        stability_attempt,
        expected_policy_root=policy_root,
        expected_contract=c5_contract,
        expected_roots={
            "precleanup_inventory_receipt": roots.get(
                "precleanup_inventory_receipt"
            ),
            "cleanup_receipt": roots.get("cleanup_receipt"),
        },
        expected_phase_guard=handoff_value["phase_guard"],
    )
    attempt_handoff_root = dict(attempt_value["scope_handoff_root"])
    if (
        fixed_handoff_root is not None
        and not _frozen._deep_exact_equal(
            fixed_handoff_root,
            attempt_handoff_root,
        )
        or fixed_attempt_root is not None
        and fixed_attempt_root.get("stability_attempt_fingerprint")
        != attempt_value.get("stability_attempt_fingerprint")
    ):
        raise PermissionError("c5 fixed handoff/attempt root changed")
    if (
        scope.get("target_unit_id") != C5_TARGET_UNIT
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
        != C5_POLICY_PATH.absolute()
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
            handoff_value.get("root_evidence"),
            {
                "precleanup_inventory_receipt": roots.get(
                    "precleanup_inventory_receipt"
                ),
                "cleanup_receipt": roots.get("cleanup_receipt"),
            },
        )
        or not _frozen._deep_exact_equal(
            handoff_value.get("toolchain"),
            policy_value.get("toolchain"),
        )
        or not _frozen._deep_exact_equal(
            attempt_value.get("toolchain"),
            policy_value.get("toolchain"),
        )
        or not _frozen._deep_exact_equal(
            contract_value,
            _normalized_contract(c5_contract),
        )
        or stability_value.get("sample_count") != SAMPLE_COUNT
        or stability_value.get("sample_interval_seconds")
        != SAMPLE_INTERVAL_SECONDS
        or len(samples) != SAMPLE_COUNT
        or stability_value.get("passed") is not True
        or stability_value.get("blockers") != []
    ):
        raise PermissionError("c5 stability scope handoff is not exact")
    realization_time = _strict_utc(
        archival_value["receipt"]["created_at_utc"],
        name="c5 realization receipt",
    )
    handoff_time = _strict_utc(
        handoff_value["created_at_utc"],
        name="c5 scope handoff",
    )
    policy_time = _strict_utc(
        policy_value["created_at_utc"],
        name="c5 policy",
    )
    attempt_time = _strict_utc(
        attempt_value["created_at_utc"],
        name="c5 stability attempt",
    )
    if not realization_time < handoff_time < policy_time < attempt_time:
        raise PermissionError("c5 metadata/stability chronology changed")
    prior_time = attempt_time
    for index, sample in enumerate(samples):
        sample_time = _strict_utc(
            sample.get("created_at_utc"),
            name=f"c5 stability sample:{index}",
        )
        try:
            inventory = dict(sample["inventory"])
            inventory_scope = dict(inventory["unit_scope"])
            target_shadow = dict(
                inventory_scope["shadows"][C5_TARGET_UNIT]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"c5 target sample is malformed:{index}"
            ) from error
        expected_target = {
            "Id": C5_TARGET_UNIT,
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
            or inventory_scope.get("target_unit_id") != C5_TARGET_UNIT
            or inventory_scope.get("require_target_ready") is not True
            or any(
                target_shadow.get(key) != value
                for key, value in expected_target.items()
            )
            or not _frozen._deep_exact_equal(
                _manager_generation_from_inventory(inventory),
                _manager_generation_from_contract(c5_contract),
            )
        ):
            raise PermissionError(
                f"c5 target sample cross-binding changed:{index}"
            )
        _no_payload(sample, name=f"c5 stability sample:{index}")
        _no_payload(inventory, name=f"c5 inventory:{index}")
        prior_time = sample_time
    stability_time = _strict_utc(
        stability_value.get("created_at_utc"),
        name="c5 stability receipt",
    )
    if not prior_time < stability_time:
        raise PermissionError("c5 stability receipt chronology changed")
    prior_time = stability_time
    if postcleanup is None:
        if verified_live_roots is not None:
            _verify_exact_live_roots(
                verified_live_roots,
                required=live_required,
            )
        return {
            "scope_handoff": dict(handoff_value),
            "stability_attempt": dict(attempt_value),
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
        name="c5 postcleanup",
    )
    if (
        set(body) != expected_keys
        or post.get("schema_version") != _frozen.ENVIRONMENT_RECEIPT_SCHEMA
        or post.get("command") != "audit-only"
        or post.get("passed") is not True
        or post.get("error_type") is not None
        or post.get("error_message") is not None
        or fingerprint != _stable_fingerprint(body)
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
        raise PermissionError("c5 postcleanup cross-binding changed")
    _no_payload(post, name="c5 postcleanup")
    if verified_live_roots is not None:
        _verify_exact_live_roots(
            verified_live_roots,
            required=live_required,
        )
    if fixed_postcleanup_root is not None:
        _verify_live_sealed(
            C5_POSTCLEANUP_PATH,
            fixed_postcleanup_root,
            "receipt_fingerprint",
        )
    return {
        "scope_handoff": dict(handoff_value),
        "stability_attempt": dict(attempt_value),
        "policy": dict(policy_value),
        "stability": dict(stability_value),
        "postcleanup": post,
        "realization": archival_value,
    }


def load_c5_environment_closure() -> dict[str, object]:
    """Load one sealed E5 PASS closure through E5's fixed producer API.

    This is the only evidence-loading interface exposed to B5.  The function
    delegates R5 authorization/receipt parsing to the hash-pinned R5 producer,
    then reads each E5-owned lane exactly once through the no-follow sealed
    reader and returns the five immutable roots alongside their payloads.
    """

    _require_c5_namespace()
    _require_frozen_environment_source()
    _load_verified_c5_bridge()
    _load_verified_c4_failure_terminalizer()
    _require_frozen_sha256(
        C4_FAILURE_TERMINAL_SHA256,
        name="c4 receipt-seal failure terminal",
    )
    _require_frozen_sha256(
        C4_FAILURE_TERMINAL_FINGERPRINT,
        name="c4 receipt-seal failure terminal fingerprint",
    )
    if os.path.lexists(C5_STABILITY_TERMINAL_PATH):
        raise PermissionError("c5 environment failure terminal closes PASS lane")
    _old_contract, c5_contract, _roots = replay_old_scope_and_handoff()
    archival = _production_archival_validator(
        C5_REALIZATION_AUTHORIZATION_PATH,
        C5_REALIZATION_RECEIPT_PATH,
    )
    scope_handoff, scope_handoff_root = _read_live_sealed(
        C5_SCOPE_HANDOFF_PATH,
        "scope_handoff_fingerprint",
    )
    stability_attempt, stability_attempt_root = _read_live_sealed(
        C5_STABILITY_ATTEMPT_PATH,
        "stability_attempt_fingerprint",
    )
    policy, policy_root = _read_live_sealed(
        C5_POLICY_PATH,
        "policy_fingerprint",
    )
    stability, stability_root = _read_live_sealed(
        C5_STABILITY_PATH,
        "stability_receipt_fingerprint",
    )
    postcleanup, postcleanup_root = _read_live_sealed(
        C5_POSTCLEANUP_PATH,
        "receipt_fingerprint",
    )
    validated = validate_c5_environment_closure(
        scope_handoff,
        stability_attempt,
        policy,
        stability,
        postcleanup,
        archival=archival,
        c5_contract=c5_contract,
        live_roots={
            "policy": policy_root,
            "stability": stability_root,
        },
    )
    for expected_name, expected_value in (
        ("scope_handoff", scope_handoff),
        ("stability_attempt", stability_attempt),
        ("policy", policy),
        ("stability", stability),
        ("postcleanup", postcleanup),
    ):
        if not _frozen._deep_exact_equal(
            validated.get(expected_name),
            expected_value,
        ):
            raise PermissionError("c5 environment producer returned drift")
    return {
        "scope_handoff": dict(scope_handoff),
        "stability_attempt": dict(stability_attempt),
        "policy": dict(policy),
        "stability": dict(stability),
        "postcleanup": dict(postcleanup),
        "evidence_roots": {
            "environment_scope_handoff": dict(scope_handoff_root),
            "environment_stability_attempt": dict(stability_attempt_root),
            "environment_policy": dict(policy_root),
            "environment_stability": dict(stability_root),
            "environment_postcleanup": dict(postcleanup_root),
        },
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
    raise PermissionError("c5 environment writes are CLI-closure only")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CURE-Lite v24 c5 fresh environment scope handoff",
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
    _no_payload(value, name="c5 environment CLI payload")
    if fingerprint_field == "scope_handoff_fingerprint":
        return validate_c5_scope_handoff(value)
    if fingerprint_field == "stability_attempt_fingerprint":
        return validate_c5_stability_attempt(value)
    if fingerprint_field == "stability_terminal_fingerprint":
        return validate_c5_stability_terminal(value)
    if fingerprint_field == "policy_fingerprint":
        return _frozen.validate_environment_policy(value)
    if fingerprint_field == "stability_receipt_fingerprint":
        return _frozen.validate_environment_stability_receipt(value)
    if fingerprint_field != "receipt_fingerprint":
        raise PermissionError("c5 environment fingerprint field changed")
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
        or fingerprint != _stable_fingerprint(body)
    ):
        raise PermissionError("c5 postcleanup payload is not closed")
    _strict_utc(value.get("created_at_utc"), name="c5 postcleanup")
    return value


def _bind_private_cli() -> Callable[[Sequence[str] | None], int]:
    fixed_identity = (
        FROZEN_ENVIRONMENT_PATH,
        FROZEN_ENVIRONMENT_SHA256,
        C5_TARGET_UNIT,
        C5_FRAGMENT_PATH,
        C5_COMPATIBILITY_BRIDGE_PATH,
        C5_COMPATIBILITY_BRIDGE_SHA256,
        C4_FAILURE_TERMINALIZER_PATH,
        C4_FAILURE_TERMINALIZER_SHA256,
        C4_FAILURE_TERMINAL_PATH,
        C4_FAILURE_TERMINAL_SHA256,
        C4_FAILURE_TERMINAL_FINGERPRINT,
        C4_TARGET_UNIT,
        C4_FRAGMENT_PATH,
        tuple(_C4_REQUIRED_ABSENT_PATHS.items()),
        PRECLEANUP_PATH,
        CLEANUP_RECEIPT_PATH,
        C5_SCOPE_HANDOFF_PATH,
        C5_POLICY_PATH,
        C5_STABILITY_ATTEMPT_PATH,
        C5_STABILITY_PATH,
        C5_STABILITY_TERMINAL_PATH,
        C5_POSTCLEANUP_PATH,
        C5_REALIZATION_AUTHORIZATION_PATH,
        C5_REALIZATION_RECEIPT_PATH,
        C5_BRIDGE_PATH,
        C5_BRIDGE_SHA256,
        C5_HANDOFF_SCHEMA,
        C5_STABILITY_ATTEMPT_SCHEMA,
        C5_STABILITY_TERMINAL_SCHEMA,
        SELECTED_GPU_INDEX,
        CONFLICT_UNIT_IDS,
        DEPENDENCY_UNIT_IDS,
        ALLOWED_FAILED_UNIT_IDS,
        ALLOWED_UNIT_IDS,
        ALLOWED_MANAGER_STATES,
        SAMPLE_COUNT,
        SAMPLE_INTERVAL_SECONDS,
    )
    fixed_handoff_path = C5_SCOPE_HANDOFF_PATH
    fixed_policy_path = C5_POLICY_PATH
    fixed_attempt_path = C5_STABILITY_ATTEMPT_PATH
    fixed_stability_path = C5_STABILITY_PATH
    fixed_terminal_path = C5_STABILITY_TERMINAL_PATH
    fixed_postcleanup_path = C5_POSTCLEANUP_PATH

    def require_fixed_identity() -> None:
        current_identity = (
            FROZEN_ENVIRONMENT_PATH,
            FROZEN_ENVIRONMENT_SHA256,
            C5_TARGET_UNIT,
            C5_FRAGMENT_PATH,
            C5_COMPATIBILITY_BRIDGE_PATH,
            C5_COMPATIBILITY_BRIDGE_SHA256,
            C4_FAILURE_TERMINALIZER_PATH,
            C4_FAILURE_TERMINALIZER_SHA256,
            C4_FAILURE_TERMINAL_PATH,
            C4_FAILURE_TERMINAL_SHA256,
            C4_FAILURE_TERMINAL_FINGERPRINT,
            C4_TARGET_UNIT,
            C4_FRAGMENT_PATH,
            tuple(_C4_REQUIRED_ABSENT_PATHS.items()),
            PRECLEANUP_PATH,
            CLEANUP_RECEIPT_PATH,
            C5_SCOPE_HANDOFF_PATH,
            C5_POLICY_PATH,
            C5_STABILITY_ATTEMPT_PATH,
            C5_STABILITY_PATH,
            C5_STABILITY_TERMINAL_PATH,
            C5_POSTCLEANUP_PATH,
            C5_REALIZATION_AUTHORIZATION_PATH,
            C5_REALIZATION_RECEIPT_PATH,
            C5_BRIDGE_PATH,
            C5_BRIDGE_SHA256,
            C5_HANDOFF_SCHEMA,
            C5_STABILITY_ATTEMPT_SCHEMA,
            C5_STABILITY_TERMINAL_SCHEMA,
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
            raise PermissionError("c5 environment fixed identity changed")
        _require_c5_namespace()
        _require_frozen_environment_source()

    def require_order(fingerprint_field: str) -> Path:
        paths = (
            fixed_handoff_path,
            fixed_policy_path,
            fixed_attempt_path,
            fixed_stability_path,
            fixed_terminal_path,
            fixed_postcleanup_path,
        )
        state = tuple(os.path.lexists(path) for path in paths)
        transitions = {
            "scope_handoff_fingerprint": (
                (False, False, False, False, False, False),
                paths[0],
            ),
            "policy_fingerprint": (
                (True, False, False, False, False, False),
                paths[1],
            ),
            "stability_attempt_fingerprint": (
                (True, True, False, False, False, False),
                paths[2],
            ),
            "stability_receipt_fingerprint": (
                (True, True, True, False, False, False),
                paths[3],
            ),
            "stability_terminal_fingerprint": (
                (True, True, True, False, False, False),
                paths[4],
            ),
            "receipt_fingerprint": (
                (True, True, True, True, False, False),
                paths[5],
            ),
        }
        transition = transitions.get(fingerprint_field)
        if transition is None or state != transition[0]:
            raise PermissionError("c5 environment write order is invalid")
        return transition[1]

    def bound_main(argv: Sequence[str] | None = None) -> int:
        args = _parser().parse_args(argv)
        if args.command not in {
            "create-policy",
            "stability-gate",
            "postcleanup",
        }:
            raise PermissionError("c5 environment command changed")
        require_fixed_identity()
        # R5 is hash-verified before any evidence transition is inspected or
        # any policy/stability/postcleanup builder can execute.
        _load_verified_c5_bridge()
        validator = _production_archival_validator
        phase_guard_validator = _production_c5_phase_guard
        handoff_root: dict[str, object] | None = None
        policy_root: dict[str, object] | None = None
        attempt_root: dict[str, object] | None = None
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
        ) -> tuple[dict[str, object], dict[str, object]]:
            nonlocal handoff_root, policy_root, attempt_root, stability_root
            lanes = {
                "handoff": (
                    fixed_handoff_path,
                    "scope_handoff_fingerprint",
                    (),
                    (True, False, False, False, False, False),
                ),
                "policy": (
                    fixed_policy_path,
                    "policy_fingerprint",
                    ("handoff",),
                    (True, True, False, False, False, False),
                ),
                "attempt": (
                    fixed_attempt_path,
                    "stability_attempt_fingerprint",
                    ("handoff", "policy"),
                    (True, True, True, False, False, False),
                ),
                "stability": (
                    fixed_stability_path,
                    "stability_receipt_fingerprint",
                    ("handoff", "policy", "attempt"),
                    (True, True, True, True, False, False),
                ),
                "terminal": (
                    fixed_terminal_path,
                    "stability_terminal_fingerprint",
                    ("handoff", "policy", "attempt"),
                    (True, True, True, False, True, False),
                ),
                "postcleanup": (
                    fixed_postcleanup_path,
                    "receipt_fingerprint",
                    ("handoff", "policy", "attempt", "stability"),
                    (True, True, True, True, False, True),
                ),
            }
            if lane not in lanes:
                raise PermissionError("c5 environment write lane changed")
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
                        fixed_handoff_path,
                        fixed_policy_path,
                        fixed_attempt_path,
                        fixed_stability_path,
                        fixed_terminal_path,
                        fixed_postcleanup_path,
                    )
                )
                if observed != expected_state:
                    raise PermissionError("c5 fixed lane state changed")
            require_fixed_identity()
            if require_order(fingerprint_field) != path:
                raise PermissionError("c5 fixed write target changed")
            value = _validate_cli_payload(
                payload,
                fingerprint_field=fingerprint_field,
            )
            body = dict(value)
            supplied_fingerprint = body.pop(fingerprint_field, None)
            if supplied_fingerprint != _stable_fingerprint(body):
                raise PermissionError("c5 fixed write fingerprint changed")
            sealed_value = {
                **body,
                fingerprint_field: supplied_fingerprint,
            }
            encoded = (
                _canonical_json(sealed_value) + "\n"
            ).encode("utf-8")
            target = Path(path).absolute()

            def revalidate_lane_closure() -> None:
                historical_contract, guard_contract, guard_roots = (
                    replay_old_scope_and_handoff()
                )
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
                        "c5 fixed lane payload changed while open"
                    )
                guard_phase = (
                    None
                    if lane == "terminal"
                    else _validate_c5_phase_guard(
                        phase_guard_validator()
                    )
                )
                if lane == "handoff":
                    validate_c5_scope_handoff(
                        current_value,
                        expected_old_contract=historical_contract,
                        expected_c5_contract=guard_contract,
                        expected_roots=guard_roots,
                        expected_archival=guard_archival,
                        expected_phase_guard=guard_phase,
                    )
                    return
                current_handoff, current_handoff_root = (
                    _read_live_sealed(
                        fixed_handoff_path,
                        "scope_handoff_fingerprint",
                    )
                )
                validate_c5_scope_handoff(
                    current_handoff,
                    expected_old_contract=historical_contract,
                    expected_c5_contract=guard_contract,
                    expected_roots=guard_roots,
                    expected_archival=guard_archival,
                    expected_phase_guard=guard_phase,
                )
                if (
                    handoff_root is None
                    or not _frozen._deep_exact_equal(
                        current_handoff_root,
                        handoff_root,
                    )
                ):
                    raise PermissionError(
                        "c5 fixed handoff root changed while open"
                    )
                if lane == "policy":
                    _validate_c5_policy_contract(
                        current_value,
                        c5_contract=guard_contract,
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
                        "c5 fixed policy root changed while open"
                    )
                _validate_c5_policy_contract(
                    current_policy,
                    c5_contract=guard_contract,
                    roots=guard_roots,
                    archival=guard_archival,
                )
                if lane == "attempt":
                    validate_c5_stability_attempt(
                        current_value,
                        expected_scope_handoff_root=current_handoff_root,
                        expected_policy_root=current_policy_root,
                        expected_contract=guard_contract,
                        expected_roots=guard_roots,
                        expected_phase_guard=guard_phase,
                    )
                    return
                current_attempt, current_attempt_root = (
                    _read_live_sealed(
                        fixed_attempt_path,
                        "stability_attempt_fingerprint",
                    )
                )
                validate_c5_stability_attempt(
                    current_attempt,
                    expected_scope_handoff_root=current_handoff_root,
                    expected_policy_root=current_policy_root,
                    expected_contract=guard_contract,
                    expected_roots=guard_roots,
                    expected_phase_guard=current_handoff["phase_guard"],
                )
                if (
                    attempt_root is None
                    or not _frozen._deep_exact_equal(
                        current_attempt_root,
                        attempt_root,
                    )
                ):
                    raise PermissionError(
                        "c5 fixed attempt root changed while open"
                    )
                if lane == "terminal":
                    validate_c5_stability_terminal(
                        current_value,
                        expected_attempt_root=current_attempt_root,
                    )
                    if os.path.lexists(fixed_stability_path):
                        raise PermissionError(
                            "c5 failure terminal conflicts with success"
                        )
                    return
                if lane == "stability":
                    validate_c5_environment_closure(
                        current_handoff,
                        current_attempt,
                        current_policy,
                        current_value,
                        None,
                        archival=guard_archival,
                        c5_contract=guard_contract,
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
                        "c5 fixed stability root changed while open"
                    )
                validate_c5_environment_closure(
                    current_handoff,
                    current_attempt,
                    current_policy,
                    current_stability,
                    current_value,
                    archival=guard_archival,
                    c5_contract=guard_contract,
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
                raise PermissionError("c5 fixed write parent is unsafe")
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
                        "c5 fixed write parent generation changed"
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
                        "c5 fixed write private generation changed"
                    )
                offset = 0
                while offset < len(encoded):
                    written = os.write(descriptor, encoded[offset:])
                    if written <= 0:
                        raise OSError("short c5 fixed evidence write")
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
                        "c5 fixed write private readback changed"
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
                        "c5 fixed write sealed generation changed"
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
                        "c5 fixed write parent generation changed"
                    )
                require_fixed_identity()
                _load_verified_c5_bridge()
                available_roots = {
                    "handoff": handoff_root,
                    "policy": policy_root,
                    "attempt": attempt_root,
                    "stability": stability_root,
                }
                guard_specs = {
                    "handoff": (
                        fixed_handoff_path,
                        "scope_handoff_fingerprint",
                    ),
                    "policy": (
                        fixed_policy_path,
                        "policy_fingerprint",
                    ),
                    "attempt": (
                        fixed_attempt_path,
                        "stability_attempt_fingerprint",
                    ),
                    "stability": (
                        fixed_stability_path,
                        "stability_receipt_fingerprint",
                    ),
                }
                for guard_name in guard_names:
                    guard_root = available_roots[guard_name]
                    if guard_root is None:
                        raise PermissionError(
                            "c5 fixed while-open guard is absent"
                        )
                    guard_path, guard_field = guard_specs[guard_name]
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
                        "c5 fixed while-open closure changed"
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
            _load_verified_c5_bridge()
            require_lane_state()
            parent_final = parent.lstat()
            if not same_generation(
                parent_guarded_path,
                parent_final,
                _PARENT_GUARD_IDENTITY_FIELDS,
            ):
                raise PermissionError(
                    "c5 fixed parent changed after close"
                )
            for guard_name in guard_names:
                guard_root = available_roots[guard_name]
                if guard_root is None:
                    raise PermissionError(
                        "c5 fixed post-close guard is absent"
                    )
                guard_path, guard_field = guard_specs[guard_name]
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
                raise PermissionError("c5 fixed write reload changed")
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
                    "c5 fixed parent changed during reload"
                )
            for guard_name in guard_names:
                guard_root = available_roots[guard_name]
                if guard_root is None:
                    raise PermissionError(
                        "c5 fixed terminal guard is absent"
                    )
                guard_path, guard_field = guard_specs[guard_name]
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
                    "c5 fixed parent changed before return"
                )
            if lane == "handoff":
                handoff_root = observed_root
            elif lane == "policy":
                policy_root = observed_root
            elif lane == "attempt":
                attempt_root = observed_root
            elif lane == "stability":
                stability_root = observed_root
            return observed_value, observed_root
        if args.command == "create-policy":
            require_order("scope_handoff_fingerprint")
            handoff = build_c5_scope_handoff_in_memory(
                realization_validator=validator,
                phase_guard_validator=phase_guard_validator,
            )
            write_fixed_lane(handoff, "handoff")
            require_order("policy_fingerprint")
            policy = build_c5_policy_in_memory(
                handoff,
                realization_validator=validator,
            )
            write_fixed_lane(policy, "policy")
            return 0
        if args.command == "stability-gate":
            require_order("stability_attempt_fingerprint")
            handoff, handoff_root = _read_live_sealed(
                fixed_handoff_path,
                fingerprint_field="scope_handoff_fingerprint",
            )
            policy, policy_root = _read_live_sealed(
                fixed_policy_path,
                fingerprint_field="policy_fingerprint",
            )
            _validate_cli_payload(
                policy,
                fingerprint_field="policy_fingerprint",
            )
            run_c5_stability_in_memory(
                handoff,
                policy,
                realization_validator=validator,
                phase_guard_validator=phase_guard_validator,
                scope_handoff_reader=lambda: _read_live_sealed(
                    fixed_handoff_path,
                    "scope_handoff_fingerprint",
                ),
                policy_reader=lambda: _read_live_sealed(
                    fixed_policy_path,
                    "policy_fingerprint",
                ),
                attempt_writer=lambda value: write_fixed_lane(
                    value,
                    "attempt",
                ),
                attempt_reader=lambda: _read_live_sealed(
                    fixed_attempt_path,
                    "stability_attempt_fingerprint",
                ),
                success_writer=lambda value: write_fixed_lane(
                    value,
                    "stability",
                ),
                terminal_writer=lambda value: write_fixed_lane(
                    value,
                    "terminal",
                ),
                lane_state_reader=lambda: {
                    "attempt": os.path.lexists(fixed_attempt_path),
                    "success": os.path.lexists(fixed_stability_path),
                    "terminal": os.path.lexists(fixed_terminal_path),
                },
            )
            return 0
        require_order("receipt_fingerprint")
        if args.command != "postcleanup":
            raise PermissionError("c5 environment command changed")

        handoff, handoff_root = _read_live_sealed(
            fixed_handoff_path,
            fingerprint_field="scope_handoff_fingerprint",
        )
        attempt, attempt_root = _read_live_sealed(
            fixed_attempt_path,
            fingerprint_field="stability_attempt_fingerprint",
        )
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
        postcleanup = build_c5_postcleanup_in_memory(
            handoff,
            attempt,
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
