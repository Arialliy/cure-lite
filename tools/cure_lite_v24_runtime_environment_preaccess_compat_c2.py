#!/usr/bin/env python3
"""Fresh runtime-environment scope handoff for compatibility generation c2.

This module does not reinterpret the frozen precleanup receipt as evidence
about the c2 unit.  It first replays that receipt with its original target and
``require_target_ready=False``.  It then creates a new contract by replacing
only those two fields and delegates policy construction and stability sampling
to the hash-pinned runtime-environment implementation.

The production entry point remains fail-closed until the c2 realization bridge
has been frozen and ``C2_BRIDGE_SHA256`` has been replaced.  Pure in-memory
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
from types import ModuleType
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
C2_TARGET_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service"
)
C2_FRAGMENT_PATH = (
    Path(f"/run/user/{os.getuid()}/systemd/user") / C2_TARGET_UNIT
)

PRECLEANUP_PATH = (
    EVIDENCE_ROOT / "runtime_environment_precleanup_receipt.json"
)
CLEANUP_RECEIPT_PATH = (
    EVIDENCE_ROOT / "environment_cleanup_recovery_r1/cleanup-receipt.json"
)
C2_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c2.json"
)
C2_STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c2.json"
)
C2_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c2.json"
)
C2_REALIZATION_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c2_unit_realization_authorization.json"
)
C2_REALIZATION_RECEIPT_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c2_unit_realization_receipt.json"
)
C2_BRIDGE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c2.py"
).resolve()
C2_BRIDGE_SHA256 = (
    "06144d42292294603631eb778fd8bc789d40b31a73b42dcf3ceeffe95fa9f7d0"
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
        "for_preaccess_compat_c2"
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
    return module, identity


frozen, _FROZEN_LOAD_IDENTITY = _load_frozen_environment()


def _require_frozen_environment_source() -> None:
    raw, identity = _stable_source_bytes(FROZEN_ENVIRONMENT_PATH)
    if (
        hashlib.sha256(raw).hexdigest()
        != FROZEN_ENVIRONMENT_SHA256
        or identity != _FROZEN_LOAD_IDENTITY
    ):
        raise PermissionError(
            "frozen runtime environment generation was replaced"
        )


def _no_payload(value: Mapping[str, object], *, name: str) -> None:
    if (
        value.get("payload_authority") not in (None, "none")
        or any(
            value.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError(f"{name} accessed scientific payload")


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


def _exact_prepare_arguments(
    args: Sequence[object],
    kwargs: Mapping[str, object],
    *,
    target_unit_id: str,
    require_target_ready: bool,
) -> None:
    if (
        len(args) != 2
        or Path(args[0]).absolute() != PRECLEANUP_PATH.absolute()
        or Path(args[1]).absolute() != CLEANUP_RECEIPT_PATH.absolute()
        or kwargs.get("selected_gpu_index") != SELECTED_GPU_INDEX
        or kwargs.get("target_unit_id") != target_unit_id
        or tuple(kwargs.get("conflict_unit_ids", ()))
        != CONFLICT_UNIT_IDS
        or tuple(kwargs.get("dependency_unit_ids", ()))
        != DEPENDENCY_UNIT_IDS
        or tuple(kwargs.get("allowed_failed_unit_ids", ()))
        != ALLOWED_FAILED_UNIT_IDS
        or tuple(kwargs.get("allowed_unit_ids", ())) != ALLOWED_UNIT_IDS
        or tuple(kwargs.get("allowed_manager_states", ()))
        != ALLOWED_MANAGER_STATES
        or kwargs.get("require_target_ready")
        is not require_target_ready
        or kwargs.get("strict_all_gpu_consumers", False) is not False
    ):
        raise PermissionError("runtime environment scope request changed")


def replay_old_scope_and_handoff(
    *,
    prepare: Callable[..., tuple[object, dict[str, object]]] | None = None,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ]
    | None = None,
) -> tuple[object, object, dict[str, object]]:
    """Replay the old scope, then replace exactly target and readiness."""

    _require_frozen_environment_source()
    if prepare is None:
        prepare = frozen.prepare_environment_stability_contract
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
    old_contract = frozen.validate_environment_audit_contract(old_contract)
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
    c2_contract = replace(
        old_contract,
        target_unit_id=C2_TARGET_UNIT,
        require_target_ready=True,
    )
    c2_contract = frozen.validate_environment_audit_contract(c2_contract)
    old_projection = asdict(old_contract)
    c2_projection = asdict(c2_contract)
    for field in fields(old_contract):
        if field.name in _HANDOFF_FIELDS:
            continue
        if not frozen._deep_exact_equal(
            old_projection[field.name],
            c2_projection[field.name],
        ):
            raise PermissionError(
                f"c2 scope handoff changed field:{field.name}"
            )
    if (
        c2_contract.target_unit_id != C2_TARGET_UNIT
        or c2_contract.require_target_ready is not True
    ):
        raise PermissionError("c2 scope handoff did not close target readiness")
    return old_contract, c2_contract, dict(roots)


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


def validate_c2_realization_archival(
    archival: Mapping[str, object],
    *,
    contract: object,
) -> dict[str, object]:
    """Bind the c2 contract to one exact archival realization PASS."""

    try:
        authorization = dict(archival["authorization"])
        receipt = dict(archival["receipt"])
        manager = dict(receipt["manager_generation"])
        fragment = dict(receipt["fragment_identity"])
        shadow = dict(receipt["full_static_shadow"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("c2 realization archival structure is malformed") from error
    _no_payload(authorization, name="c2 realization authorization")
    _no_payload(receipt, name="c2 realization receipt")
    expected_manager = _manager_generation_from_contract(contract)
    expected_shadow = {
        "Id": C2_TARGET_UNIT,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "Restart": "no",
        "NRestarts": "0",
        "FragmentPath": str(C2_FRAGMENT_PATH),
    }
    if (
        authorization.get("unit_name") != C2_TARGET_UNIT
        or receipt.get("unit_name") != C2_TARGET_UNIT
        or receipt.get("passed") is not True
        or receipt.get("static") is not True
        or receipt.get("enabled") is not False
        or receipt.get("started") is not False
        or not frozen._deep_exact_equal(
            authorization.get("manager_generation"),
            expected_manager,
        )
        or not frozen._deep_exact_equal(manager, expected_manager)
        or fragment.get("path") != str(C2_FRAGMENT_PATH)
        or shadow.get("FragmentPath") != fragment.get("path")
        or any(
            shadow.get(key) != value
            for key, value in expected_shadow.items()
        )
    ):
        raise PermissionError(
            "c2 realization is not an exact live-ready archival PASS"
        )
    return {
        "authorization": authorization,
        "receipt": receipt,
        "fragment": fragment,
        "shadow": shadow,
    }


def _load_verified_c2_bridge() -> ModuleType:
    if (
        C2_BRIDGE_SHA256 == "__TO_BE_FROZEN__"
        or len(C2_BRIDGE_SHA256) != 64
        or any(character not in "0123456789abcdef"
               for character in C2_BRIDGE_SHA256)
    ):
        raise PermissionError("c2 realization bridge is not frozen")
    raw, identity = _stable_source_bytes(C2_BRIDGE_PATH)
    if hashlib.sha256(raw).hexdigest() != C2_BRIDGE_SHA256:
        raise PermissionError("c2 realization bridge source changed")
    name = (
        "tools._cure_lite_v24_actual_unit_realization_"
        "preaccess_compat_c2_verified_for_environment"
    )
    module = ModuleType(name)
    module.__file__ = str(C2_BRIDGE_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(raw, str(C2_BRIDGE_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    raw_after, identity_after = _stable_source_bytes(C2_BRIDGE_PATH)
    if (
        hashlib.sha256(raw_after).hexdigest() != C2_BRIDGE_SHA256
        or identity_after != identity
    ):
        sys.modules.pop(name, None)
        raise PermissionError(
            "c2 realization bridge generation changed while loading"
        )
    return module


def _production_archival_validator(
    authorization_path: Path,
    receipt_path: Path,
) -> Mapping[str, object]:
    bridge = _load_verified_c2_bridge()
    validator = getattr(bridge, "validate_archival_realization_chain", None)
    if not callable(validator):
        raise PermissionError("c2 archival realization validator is absent")
    return validator(authorization_path, receipt_path)


def _resolve_archival(
    validator: ArchivalValidator,
    *,
    contract: object,
) -> dict[str, object]:
    archival = validator(
        C2_REALIZATION_AUTHORIZATION_PATH,
        C2_REALIZATION_RECEIPT_PATH,
    )
    if not isinstance(archival, Mapping):
        raise PermissionError("c2 archival validator returned no closure")
    return validate_c2_realization_archival(archival, contract=contract)


def build_c2_policy_in_memory(
    *,
    realization_validator: ArchivalValidator,
    prepare: Callable[..., tuple[object, dict[str, object]]] | None = None,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ]
    | None = None,
    toolchain_reader: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build, but do not write, the fixed c2 fresh environment policy."""

    old_contract, c2_contract, roots = replay_old_scope_and_handoff(
        prepare=prepare,
        activation_guard_reader=activation_guard_reader,
    )
    del old_contract
    archival = _resolve_archival(
        realization_validator,
        contract=c2_contract,
    )
    if toolchain_reader is None:
        toolchain_reader = frozen.current_runtime_toolchain_binding
    policy = frozen.build_environment_policy(
        c2_contract,
        precleanup_root_binding=roots["precleanup_inventory_receipt"],
        cleanup_root_binding=roots["cleanup_receipt"],
        toolchain_binding=toolchain_reader(),
        minimum_sample_count=SAMPLE_COUNT,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
    )
    policy = frozen.validate_environment_policy(policy)
    if not (
        _strict_utc(
            archival["receipt"]["created_at_utc"],
            name="c2 realization receipt",
        )
        < _strict_utc(policy["created_at_utc"], name="c2 policy")
    ):
        raise PermissionError("c2 policy predates unit realization")
    return policy


def _patched_prepare(
    c2_contract: object,
    roots: Mapping[str, object],
) -> Callable[..., tuple[object, dict[str, object]]]:
    def prepare(
        *args: object,
        **kwargs: object,
    ) -> tuple[object, dict[str, object]]:
        _exact_prepare_arguments(
            args,
            kwargs,
            target_unit_id=C2_TARGET_UNIT,
            require_target_ready=True,
        )
        return c2_contract, dict(roots)

    return prepare


def run_c2_stability_in_memory(
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
) -> dict[str, object]:
    """Run the frozen stability gate with an exact patched c2 prepare."""

    _require_frozen_environment_source()
    _old, c2_contract, roots = replay_old_scope_and_handoff(
        prepare=prepare,
        activation_guard_reader=activation_guard_reader,
    )
    archival = _resolve_archival(
        realization_validator,
        contract=c2_contract,
    )
    frozen.validate_environment_policy(policy)
    expected = frozen.build_environment_policy(
        c2_contract,
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
    if not frozen._deep_exact_equal(policy_body, expected_body):
        raise PermissionError("c2 policy differs from exact handoff contract")
    if gate is None:
        gate = frozen.run_environment_stability_gate
    call_kwargs: dict[str, object] = {
        "selected_gpu_index": SELECTED_GPU_INDEX,
        "target_unit_id": C2_TARGET_UNIT,
        "conflict_unit_ids": CONFLICT_UNIT_IDS,
        "dependency_unit_ids": DEPENDENCY_UNIT_IDS,
        "allowed_failed_unit_ids": ALLOWED_FAILED_UNIT_IDS,
        "sample_count": SAMPLE_COUNT,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "policy_path": C2_POLICY_PATH,
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
    prior_prepare = frozen.prepare_environment_stability_contract
    frozen.prepare_environment_stability_contract = _patched_prepare(
        c2_contract,
        roots,
    )
    try:
        stability = gate(
            PRECLEANUP_PATH,
            CLEANUP_RECEIPT_PATH,
            **call_kwargs,
        )
    finally:
        frozen.prepare_environment_stability_contract = prior_prepare
    _require_frozen_environment_source()
    validate_c2_environment_closure(
        policy,
        stability,
        None,
        archival=archival,
        c2_contract=c2_contract,
    )
    return dict(stability)


def build_c2_postcleanup_in_memory(
    policy: Mapping[str, object],
    stability: Mapping[str, object],
    *,
    realization_validator: ArchivalValidator,
    prepare: Callable[..., tuple[object, dict[str, object]]] | None = None,
    activation_guard_reader: Callable[
        [Mapping[str, object]], dict[str, object]
    ]
    | None = None,
) -> dict[str, object]:
    """Reuse the exact final stability inventory in a fresh audit receipt."""

    _old, c2_contract, _roots = replay_old_scope_and_handoff(
        prepare=prepare,
        activation_guard_reader=activation_guard_reader,
    )
    archival = _resolve_archival(
        realization_validator,
        contract=c2_contract,
    )
    validate_c2_environment_closure(
        policy,
        stability,
        None,
        archival=archival,
        c2_contract=c2_contract,
    )
    samples = list(stability["samples"])
    inventory = json.loads(
        frozen.canonical_json(samples[-1]["inventory"])
    )
    endpoint = dict(inventory["manager"]["endpoint"])
    body = {
        "schema_version": frozen.ENVIRONMENT_RECEIPT_SCHEMA,
        "created_at_utc": frozen.utc_now(),
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
        "receipt_fingerprint": frozen.stable_fingerprint(body),
    }
    validate_c2_environment_closure(
        policy,
        stability,
        postcleanup,
        archival=archival,
        c2_contract=c2_contract,
    )
    return postcleanup


def _normalized_contract(contract: object) -> dict[str, object]:
    return json.loads(frozen.canonical_json(asdict(contract)))


def validate_c2_environment_closure(
    policy: Mapping[str, object],
    stability: Mapping[str, object],
    postcleanup: Mapping[str, object] | None,
    *,
    archival: Mapping[str, object],
    c2_contract: object,
) -> dict[str, object]:
    """Validate all cross-bindings omitted by the generic frozen validator."""

    policy_value = frozen.validate_environment_policy(policy)
    stability_value = frozen.validate_environment_stability_receipt(
        stability
    )
    archival_value = validate_c2_realization_archival(
        archival,
        contract=c2_contract,
    )
    _no_payload(policy_value, name="c2 environment policy")
    _no_payload(stability_value, name="c2 environment stability")
    scope = dict(policy_value["unit_scope"])
    sampling = dict(policy_value["sampling"])
    contract_value = dict(stability_value["contract"])
    samples = list(stability_value["samples"])
    expected_policy = frozen.build_environment_policy(
        c2_contract,
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
        scope.get("target_unit_id") != C2_TARGET_UNIT
        or scope.get("require_target_ready") is not True
        or not frozen._deep_exact_equal(
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
        != C2_POLICY_PATH.absolute()
        or policy_root.get("policy_fingerprint")
        != policy_value.get("policy_fingerprint")
        or not frozen._deep_exact_equal(
            roots.get("precleanup_inventory_receipt"),
            policy_value.get("precleanup_root"),
        )
        or not frozen._deep_exact_equal(
            roots.get("cleanup_receipt"),
            policy_value.get("cleanup_root"),
        )
        or not frozen._deep_exact_equal(
            contract_value,
            _normalized_contract(c2_contract),
        )
        or stability_value.get("sample_count") != SAMPLE_COUNT
        or stability_value.get("sample_interval_seconds")
        != SAMPLE_INTERVAL_SECONDS
        or len(samples) != SAMPLE_COUNT
        or stability_value.get("passed") is not True
        or stability_value.get("blockers") != []
    ):
        raise PermissionError("c2 stability scope handoff is not exact")
    realization_time = _strict_utc(
        archival_value["receipt"]["created_at_utc"],
        name="c2 realization receipt",
    )
    policy_time = _strict_utc(
        policy_value["created_at_utc"],
        name="c2 policy",
    )
    if not realization_time < policy_time:
        raise PermissionError("c2 realization/policy chronology changed")
    prior_time = policy_time
    for index, sample in enumerate(samples):
        sample_time = _strict_utc(
            sample.get("created_at_utc"),
            name=f"c2 stability sample:{index}",
        )
        try:
            inventory = dict(sample["inventory"])
            inventory_scope = dict(inventory["unit_scope"])
            target_shadow = dict(
                inventory_scope["shadows"][C2_TARGET_UNIT]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"c2 target sample is malformed:{index}"
            ) from error
        expected_target = {
            "Id": C2_TARGET_UNIT,
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
            or inventory_scope.get("target_unit_id") != C2_TARGET_UNIT
            or inventory_scope.get("require_target_ready") is not True
            or any(
                target_shadow.get(key) != value
                for key, value in expected_target.items()
            )
            or not frozen._deep_exact_equal(
                _manager_generation_from_inventory(inventory),
                _manager_generation_from_contract(c2_contract),
            )
        ):
            raise PermissionError(
                f"c2 target sample cross-binding changed:{index}"
            )
        _no_payload(sample, name=f"c2 stability sample:{index}")
        _no_payload(inventory, name=f"c2 inventory:{index}")
        prior_time = sample_time
    if postcleanup is None:
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
        name="c2 postcleanup",
    )
    if (
        set(body) != expected_keys
        or post.get("schema_version") != frozen.ENVIRONMENT_RECEIPT_SCHEMA
        or post.get("command") != "audit-only"
        or post.get("passed") is not True
        or post.get("error_type") is not None
        or post.get("error_message") is not None
        or fingerprint != frozen.stable_fingerprint(body)
        or not frozen._deep_exact_equal(
            post.get("inventory"),
            final_inventory,
        )
        or not frozen._deep_exact_equal(
            post.get("environment_binding"),
            _environment_binding_from_inventory(final_inventory),
        )
        or not prior_time < post_time
    ):
        raise PermissionError("c2 postcleanup cross-binding changed")
    _no_payload(post, name="c2 postcleanup")
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


def _write_create_once(
    path: Path,
    payload: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    body = dict(payload)
    body.pop(fingerprint_field)
    return frozen.write_create_once_receipt(
        path,
        body,
        fingerprint_field=fingerprint_field,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CURE-Lite v24 c2 fresh environment scope handoff",
    )
    parser.add_argument(
        "command",
        choices=("create-policy", "stability-gate", "postcleanup"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Loading the not-yet-frozen bridge is the first production-side action.
    # This intentionally fails before any evidence path can be written.
    validator = _production_archival_validator
    _load_verified_c2_bridge()
    if args.command == "create-policy":
        policy = build_c2_policy_in_memory(
            realization_validator=validator,
        )
        _write_create_once(
            C2_POLICY_PATH,
            policy,
            fingerprint_field="policy_fingerprint",
        )
        return 0
    policy = frozen.load_sealed_receipt(
        C2_POLICY_PATH,
        fingerprint_field="policy_fingerprint",
    )
    if args.command == "stability-gate":
        stability = run_c2_stability_in_memory(
            policy,
            realization_validator=validator,
        )
        _write_create_once(
            C2_STABILITY_PATH,
            stability,
            fingerprint_field="stability_receipt_fingerprint",
        )
        return 0
    stability = frozen.load_sealed_receipt(
        C2_STABILITY_PATH,
        fingerprint_field="stability_receipt_fingerprint",
    )
    postcleanup = build_c2_postcleanup_in_memory(
        policy,
        stability,
        realization_validator=validator,
    )
    _write_create_once(
        C2_POSTCLEANUP_PATH,
        postcleanup,
        fingerprint_field="receipt_fingerprint",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
