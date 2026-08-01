from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import stat
import sys
from types import MappingProxyType, ModuleType
from typing import Mapping

import pytest

from tools import (
    cure_lite_v24_runtime_environment_preaccess_compat_c5 as compat,
)


environment = compat.frozen
UID = os.getuid()
BOOT_ID = "12345678-1234-1234-1234-123456789abc"
GPU_UUID = "GPU-12cdabd0-7910-8f4a-e4d7-e3c7867d1296"
PCI_BUS_ID = "00000000:02:00.0"


def _contract(*, target: str = compat.OLD_TARGET_UNIT):
    conflict = compat.CONFLICT_UNIT_IDS[0]
    guard = {
        "mode": environment.RECOVERY_GUARD_MODE,
        "unit_name": conflict,
        "path": f"/run/user/{UID}/systemd/user/{conflict}",
        "target": "/dev/null",
        "owner_uid": UID,
        "device": 54,
        "inode": 99,
        "observed_unit_file_state": "enabled",
    }
    return environment.validate_environment_audit_contract(
        environment.EnvironmentAuditContract(
            uid=UID,
            boot_id=BOOT_ID,
            runtime_directory=f"/run/user/{UID}",
            runtime_directory_device=54,
            runtime_directory_inode=1,
            manager_pid=1808,
            bus_path=f"/run/user/{UID}/bus",
            bus_device=54,
            bus_inode=61,
            manager_starttime_ticks=769,
            manager_control_group=(
                f"/user.slice/user-{UID}.slice/"
                f"user@{UID}.service/init.scope"
            ),
            selected_gpu_index=0,
            selected_gpu_uuid=GPU_UUID,
            selected_gpu_pci_bus_id=PCI_BUS_ID,
            selected_gpu_minor_number=0,
            target_unit_id=target,
            conflict_unit_ids=compat.CONFLICT_UNIT_IDS,
            dependency_unit_ids=compat.DEPENDENCY_UNIT_IDS,
            allowed_failed_unit_ids=compat.ALLOWED_FAILED_UNIT_IDS,
            expected_failed_unit_ids=compat.ALLOWED_FAILED_UNIT_IDS,
            maximum_restart_usec=30_000_000,
            maximum_trigger_usec=0,
            required_stability_window_usec=30_000_000,
            cleanup_mode=environment.RECOVERY_CLEANUP_MODE,
            quiescence_mode=environment.RECOVERY_QUIESCENCE_MODE,
            cleanup_nrestarts_baseline=((conflict, "7"),),
            activation_guard=guard,
            allowed_unit_ids=compat.ALLOWED_UNIT_IDS,
            allowed_manager_states=compat.ALLOWED_MANAGER_STATES,
            require_target_ready=False,
            strict_all_gpu_consumers=False,
        )
    )


def _file_evidence(
    path: Path,
    *,
    digest: str,
) -> dict[str, object]:
    return {
        "path": str(path.absolute()),
        "device": 2304,
        "inode": 100,
        "size": 1000,
        "mtime_ns": 1234,
        "file_sha256": digest,
    }


def _roots() -> dict[str, object]:
    return {
        "precleanup_inventory_receipt": {
            **_file_evidence(compat.PRECLEANUP_PATH, digest="a" * 64),
            "receipt_fingerprint": "b" * 64,
            "inventory_fingerprint": "c" * 64,
        },
        "cleanup_receipt": {
            **_file_evidence(
                compat.CLEANUP_RECEIPT_PATH,
                digest="d" * 64,
            ),
            "cleanup_receipt_fingerprint": "e" * 64,
        },
    }


def _toolchain() -> dict[str, object]:
    return {
        name: {
            "path": f"/test/{name}",
            "file_sha256": character * 64,
        }
        for name, character in (
            ("runtime_environment", "1"),
            ("python", "2"),
            ("systemctl", "3"),
            ("nvidia_smi", "4"),
        )
    }


def _prepare(old_contract, roots, *, calls: list[str] | None = None):
    def prepare(*args, **kwargs):
        if calls is not None:
            calls.append("prepare")
        assert tuple(Path(item).absolute() for item in args) == (
            compat.PRECLEANUP_PATH.absolute(),
            compat.CLEANUP_RECEIPT_PATH.absolute(),
        )
        assert kwargs["target_unit_id"] == compat.OLD_TARGET_UNIT
        assert kwargs["require_target_ready"] is False
        return old_contract, deepcopy(roots)

    return prepare


def _manager(contract) -> dict[str, object]:
    return compat._manager_generation_from_contract(contract)


def _r5_archival_root(
    path: Path,
    *,
    digest: str,
    inode: int,
) -> dict[str, object]:
    return {
        "path": str(path.absolute()),
        "resolved_path": str(path.absolute()),
        "path_is_symlink": False,
        "file_sha256": digest,
        "device": 2304,
        "inode": inode,
        "owner_uid": UID,
        "owner_gid": UID,
        "mode": 0o444,
        "nlink": 1,
        "size": 1000,
        "mtime_ns": 1234,
        "ctime_ns": 1235,
        "parent_path": str(path.absolute().parent),
        "parent_device": 2304,
        "parent_inode": 99,
        "parent_owner_uid": UID,
        "parent_owner_gid": UID,
        "parent_mode": 0o700,
    }


def _archival(
    contract,
    *,
    created_at: str = "2026-07-30T00:00:01Z",
) -> dict[str, object]:
    manager = _manager(contract)
    authorization_digest = "9" * 64
    fragment = {
        "path": str(compat.C5_FRAGMENT_PATH),
        "file_sha256": "f" * 64,
    }
    shadow = {
        "Id": compat.C5_TARGET_UNIT,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "Restart": "no",
        "NRestarts": "0",
        "FragmentPath": str(compat.C5_FRAGMENT_PATH),
    }
    no_payload = {
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {
        "authorization": {
            "unit_name": compat.C5_TARGET_UNIT,
            "manager_generation": deepcopy(manager),
            **no_payload,
        },
        "receipt": {
            "unit_name": compat.C5_TARGET_UNIT,
            "created_at_utc": created_at,
            "authorization_file_sha256": authorization_digest,
            "manager_generation": deepcopy(manager),
            "fragment_identity": fragment,
            "full_static_shadow": shadow,
            "passed": True,
            "static": True,
            "enabled": False,
            "started": False,
            **no_payload,
        },
        "authorization_identity": _r5_archival_root(
            compat.C5_REALIZATION_AUTHORIZATION_PATH,
            digest=authorization_digest,
            inode=501,
        ),
        "receipt_identity": _r5_archival_root(
            compat.C5_REALIZATION_RECEIPT_PATH,
            digest="8" * 64,
            inode=502,
        ),
    }


def _validator(archival):
    def validate(authorization_path: Path, receipt_path: Path):
        assert authorization_path == compat.C5_REALIZATION_AUTHORIZATION_PATH
        assert receipt_path == compat.C5_REALIZATION_RECEIPT_PATH
        return deepcopy(archival)

    return validate


def _phase_guard() -> dict[str, object]:
    return {
        "c4_failure_terminal_root": {
            "path": str(compat.C4_FAILURE_TERMINAL_PATH),
            "file_sha256": compat.C4_FAILURE_TERMINAL_SHA256,
            "terminal_fingerprint": (
                compat.C4_FAILURE_TERMINAL_FINGERPRINT
            ),
            "schema_version": (
                "cure-lite-v24-r2-preaccess-schema-compat-c4-"
                "receipt-seal-failure-terminal-v1"
            ),
            "device": 2304,
            "inode": 777,
            "owner_uid": UID,
            "owner_gid": UID,
            "mode": 0o444,
            "nlink": 1,
            "size": 1000,
            "mtime_ns": 1,
            "ctime_ns": 1,
        },
        "c4_unit_state": {
            "Id": compat.C4_TARGET_UNIT,
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "static",
            "FragmentPath": str(compat.C4_FRAGMENT_PATH),
            "InvocationID": "",
            "Restart": "no",
            "NRestarts": "0",
            "NeedDaemonReload": "no",
        },
        "c4_absent_outputs": {
            name: str(path.absolute())
            for name, path in compat._C4_REQUIRED_ABSENT_PATHS.items()
        },
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def _shadow(unit: str, *, target: bool) -> dict[str, str]:
    return {
        "Id": unit,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static" if target else "enabled",
        "Restart": "no" if target else "on-failure",
        "RestartUSec": "0" if target else "30s",
        "NRestarts": "0" if target else "7",
        "ControlGroup": "",
        "FragmentPath": (
            str(compat.C5_FRAGMENT_PATH)
            if target
            else f"/home/test/{unit}"
        ),
        "DropInPaths": "",
        "TriggeredBy": "",
        "Triggers": "",
        "WantedBy": "" if target else "default.target",
        "RequiredBy": "",
        "PartOf": "",
    }


def _inventory(contract) -> dict[str, object]:
    conflict = compat.CONFLICT_UNIT_IDS[0]
    gpu_body: dict[str, object] = {
        "schema_version": environment.GPU_DOUBLE_SNAPSHOT_SCHEMA,
        "selected_gpu_uuid": GPU_UUID,
        "expected_uid": UID,
        "allowed_unit_ids": list(compat.ALLOWED_UNIT_IDS),
        "strict_all_gpu_consumers": False,
        "devices": [
            {
                "index": 0,
                "uuid": GPU_UUID,
                "pci_bus_id": PCI_BUS_ID,
                "compute_mode": "Default",
                "mig_mode": None,
                "driver_version": "580.126.09",
                "minor_number": 0,
                "mps_state": "not_observed",
            }
        ],
        "first_apps": [],
        "second_apps": [],
        "process_unit_mapping": [],
        "observations": [],
        "blockers": [],
        "passed": True,
    }
    gpu = {
        **gpu_body,
        "snapshot_fingerprint": environment.stable_fingerprint(gpu_body),
    }
    manager = _manager(contract)
    body: dict[str, object] = {
        "schema_version": environment.ENVIRONMENT_INVENTORY_SCHEMA,
        "created_at_utc": environment.utc_now(),
        "uid": UID,
        "boot_id": BOOT_ID,
        "manager": {
            "state": "degraded",
            "allowed_states": list(compat.ALLOWED_MANAGER_STATES),
            "returncode": 1,
            "failed_units": list(compat.ALLOWED_FAILED_UNIT_IDS),
            "allowed_failed_unit_ids": list(
                compat.ALLOWED_FAILED_UNIT_IDS
            ),
            "unexpected_failed_unit_ids": [],
            "scoped_failed_unit_ids": [],
            "identity": manager["identity"],
            "endpoint": manager["endpoint"],
        },
        "unit_scope": {
            "target_unit_id": compat.C5_TARGET_UNIT,
            "conflict_unit_ids": list(compat.CONFLICT_UNIT_IDS),
            "dependency_unit_ids": [],
            "require_target_ready": True,
            "shadows": {
                compat.C5_TARGET_UNIT: _shadow(
                    compat.C5_TARGET_UNIT,
                    target=True,
                ),
                conflict: _shadow(conflict, target=False),
            },
        },
        "gpu_snapshot": gpu,
        "blockers": [],
        "passed": True,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {
        **body,
        "inventory_fingerprint": environment.stable_fingerprint(body),
    }


def _sealed_root(
    path: Path,
    field: str,
    fingerprint: str,
    *,
    inode: int,
) -> dict[str, object]:
    return {
        **_file_evidence(path, digest=hex(inode % 16)[2:] * 64),
        "inode": inode,
        field: fingerprint,
    }


def _bundle() -> dict[str, object]:
    old = _contract()
    roots = _roots()
    _old, c5, replayed = compat.replay_old_scope_and_handoff(
        prepare=_prepare(old, roots),
    )
    archival = _archival(c5)
    guard = _phase_guard()
    handoff = compat.build_c5_scope_handoff_in_memory(
        realization_validator=_validator(archival),
        phase_guard_validator=lambda: deepcopy(guard),
        prepare=_prepare(old, roots),
        toolchain_reader=_toolchain,
    )
    policy = compat.build_c5_policy_in_memory(
        handoff,
        realization_validator=_validator(archival),
        prepare=_prepare(old, roots),
        toolchain_reader=_toolchain,
    )
    handoff_root = _sealed_root(
        compat.C5_SCOPE_HANDOFF_PATH,
        "scope_handoff_fingerprint",
        handoff["scope_handoff_fingerprint"],
        inode=201,
    )
    policy_root = _sealed_root(
        compat.C5_POLICY_PATH,
        "policy_fingerprint",
        policy["policy_fingerprint"],
        inode=202,
    )
    return {
        "old": old,
        "c5": c5,
        "roots": replayed,
        "archival": archival,
        "guard": guard,
        "handoff": handoff,
        "policy": policy,
        "handoff_root": handoff_root,
        "policy_root": policy_root,
        "inventory": _inventory(c5),
    }


def _runtime(
    bundle: Mapping[str, object],
    *,
    monotonic_values: tuple[float, float] = (0.0, 30.0),
    guard_drift_call: int | None = None,
    prepare_drift_call: int | None = None,
    archival_drift_call: int | None = None,
    archival_drift_lane: str = "authorization_identity",
    archival_drift_field: str = "ctime_ns",
) -> dict[str, object]:
    state = {"attempt": False, "success": False, "terminal": False}
    evidence: dict[str, object] = {}
    events: list[object] = []
    guard_calls = 0
    prepare_calls = 0
    archival_calls = 0
    monotonic_iterator = iter(monotonic_values)

    def phase_guard():
        nonlocal guard_calls
        guard_calls += 1
        events.append(("guard", guard_calls))
        value = deepcopy(bundle["guard"])
        if guard_calls == guard_drift_call:
            value["c4_unit_state"]["NRestarts"] = "1"
        return value

    def prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        events.append(("prepare", prepare_calls))
        value = deepcopy(bundle["roots"])
        if prepare_calls == prepare_drift_call:
            value["cleanup_receipt"]["file_sha256"] = "0" * 64
        assert kwargs["target_unit_id"] == compat.OLD_TARGET_UNIT
        return bundle["old"], value

    def realization_validator(
        authorization_path: Path,
        receipt_path: Path,
    ) -> dict[str, object]:
        nonlocal archival_calls
        archival_calls += 1
        events.append(("archival", archival_calls))
        assert authorization_path == compat.C5_REALIZATION_AUTHORIZATION_PATH
        assert receipt_path == compat.C5_REALIZATION_RECEIPT_PATH
        value = deepcopy(bundle["archival"])
        if archival_calls == archival_drift_call:
            value[archival_drift_lane][archival_drift_field] += 1
        return value

    def attempt_writer(payload):
        assert not state["attempt"]
        assert not any(state.values())
        events.append("attempt-write")
        state["attempt"] = True
        value = deepcopy(payload)
        root = _sealed_root(
            compat.C5_STABILITY_ATTEMPT_PATH,
            "stability_attempt_fingerprint",
            value["stability_attempt_fingerprint"],
            inode=203,
        )
        evidence["attempt"] = value
        evidence["attempt_root"] = root
        return deepcopy(value), deepcopy(root)

    def attempt_reader():
        return (
            deepcopy(evidence["attempt"]),
            deepcopy(evidence["attempt_root"]),
        )

    def success_writer(payload):
        assert state == {
            "attempt": True,
            "success": False,
            "terminal": False,
        }
        events.append("success-write")
        state["success"] = True
        value = deepcopy(payload)
        root = _sealed_root(
            compat.C5_STABILITY_PATH,
            "stability_receipt_fingerprint",
            value["stability_receipt_fingerprint"],
            inode=204,
        )
        evidence["stability"] = value
        evidence["stability_root"] = root
        return deepcopy(value), deepcopy(root)

    def terminal_writer(payload):
        assert state == {
            "attempt": True,
            "success": False,
            "terminal": False,
        }
        events.append("terminal-write")
        state["terminal"] = True
        value = deepcopy(payload)
        root = _sealed_root(
            compat.C5_STABILITY_TERMINAL_PATH,
            "stability_terminal_fingerprint",
            value["stability_terminal_fingerprint"],
            inode=205,
        )
        evidence["terminal"] = value
        evidence["terminal_root"] = root
        return deepcopy(value), deepcopy(root)

    def collector(**kwargs):
        assert kwargs["target_unit_id"] == compat.C5_TARGET_UNIT
        assert state["attempt"] is True
        events.append("audit")
        return deepcopy(bundle["inventory"])

    def sleep(seconds: float):
        events.append(("sleep", seconds))

    def monotonic():
        value = next(monotonic_iterator)
        events.append(("monotonic", value))
        return value

    def utc_clock() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    kwargs = {
        "realization_validator": realization_validator,
        "phase_guard_validator": phase_guard,
        "scope_handoff_reader": lambda: (
            deepcopy(bundle["handoff"]),
            deepcopy(bundle["handoff_root"]),
        ),
        "policy_reader": lambda: (
            deepcopy(bundle["policy"]),
            deepcopy(bundle["policy_root"]),
        ),
        "attempt_writer": attempt_writer,
        "attempt_reader": attempt_reader,
        "success_writer": success_writer,
        "terminal_writer": terminal_writer,
        "lane_state_reader": lambda: dict(state),
        "prepare": prepare,
        "inventory_collector": collector,
        "activation_guard_reader": lambda guard: {
            **dict(guard),
            "file_type": "symlink",
        },
        "sleeper": sleep,
        "monotonic_clock": monotonic,
        "toolchain_reader": _toolchain,
        "clock": utc_clock,
    }
    return {
        "state": state,
        "evidence": evidence,
        "events": events,
        "kwargs": kwargs,
    }


def test_all_foreign_producer_pins_are_frozen() -> None:
    assert compat.FROZEN_ENVIRONMENT_SHA256 == (
        "a40465786ce3537346372df5991bb6788d44feddfd497ec83a1dc302fb8b2fea"
    )
    assert compat.C5_BRIDGE_SHA256 == (
        "dbe35cd096554c4fd4c64b34213b0f7ac3ccb79e396f6d1d8e620c2c4c1d1be5"
    )
    assert compat.C5_COMPATIBILITY_BRIDGE_SHA256 == (
        "388843b9b840db41610d57543f4982666cdf442ba81fa5acb208033de062319f"
    )
    assert compat.C4_FAILURE_TERMINALIZER_SHA256 == (
        "3cf56e803d6d7b39c995125d17d145b5c8625a4eea03de6cf4c6118c9bc777c0"
    )
    assert compat.C4_FAILURE_TERMINAL_SHA256 == (
        "567b22e9839dad2d27168c36206b66be9b2b91d98269e9b9ce087ee3becea733"
    )
    assert compat.C4_FAILURE_TERMINAL_FINGERPRINT == (
        "d86ef0c432237043e39119c56cfb6602b7df7f8b62069f836ac6c3d08b75b622"
    )
    assert not {
        name: value
        for name, value in vars(compat).items()
        if value == "__TO_BE_FROZEN__"
    }
    compat._require_frozen_environment_source()
    compat._verify_fixed_source(
        compat.C5_COMPATIBILITY_BRIDGE_PATH,
        compat.C5_COMPATIBILITY_BRIDGE_SHA256,
        name="b5 compatibility bridge",
    )
    bridge = compat._load_verified_c5_bridge()
    assert callable(bridge.validate_archival_realization_chain)
    terminalizer = compat._load_verified_c4_failure_terminalizer()
    assert callable(terminalizer.validate_archival)


def test_real_r5_b5_and_c4_terminal_interop() -> None:
    """Load all three fixed foreign producers without touching systemd."""

    producer = compat._load_verified_c5_bridge()

    assert Path(producer.__file__).resolve() == compat.C5_BRIDGE_PATH
    assert producer.COMPAT_BRIDGE_SOURCE_PATH == (
        compat.C5_COMPATIBILITY_BRIDGE_PATH
    )
    assert producer.COMPAT_BRIDGE_SOURCE_SHA256 == (
        compat.C5_COMPATIBILITY_BRIDGE_SHA256
    )
    assert producer.COMPAT_UNIT == compat.C5_TARGET_UNIT
    assert callable(producer.validate_archival_realization_chain)
    assert hashlib.sha256(compat.C5_BRIDGE_PATH.read_bytes()).hexdigest() == (
        compat.C5_BRIDGE_SHA256
    )

    terminalizer = compat._load_verified_c4_failure_terminalizer()
    terminal, terminal_root = terminalizer.validate_archival(
        compat.C4_FAILURE_TERMINAL_PATH
    )
    assert terminal_root["file_sha256"] == (
        compat.C4_FAILURE_TERMINAL_SHA256
    )
    assert terminal_root["terminal_fingerprint"] == (
        compat.C4_FAILURE_TERMINAL_FINGERPRINT
    )
    assert terminal["terminal_fingerprint"] == (
        compat.C4_FAILURE_TERMINAL_FINGERPRINT
    )
    payload_observation = terminal["payload_observation"]
    assert all(
        payload_observation[field] is False
        for field in (
            "D_R_payload_accessed",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
            "gpu_compute_accessed",
            "training_started",
            "scientific_attempt_consumed",
        )
    )
    assert all(
        payload_observation[field] == 0
        for field in (
            "scientific_samples_processed",
            "optimizer_steps",
            "parameter_updates",
        )
    )


def test_r5_archival_root_rejects_same_bytes_inode_replacement(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sealed.json"
    target.write_bytes(b"{}\n")
    target.chmod(0o444)
    raw, generation = compat._stable_source_bytes(target)
    supplied = {
        "path": str(target.absolute()),
        "resolved_path": str(target.absolute()),
        "path_is_symlink": False,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "device": generation["st_dev"],
        "inode": generation["st_ino"],
        "owner_uid": generation["st_uid"],
        "mode": stat.S_IMODE(generation["st_mode"]),
        "nlink": generation["st_nlink"],
    }
    root = compat._bind_r5_archival_root(target, supplied)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(raw)
    replacement.chmod(0o444)
    replacement.replace(target)
    assert target.stat().st_ino != root["inode"]
    with pytest.raises(PermissionError, match="changed after archival"):
        compat._bind_r5_archival_root(target, supplied)


def test_realization_archival_retains_and_checks_file_roots() -> None:
    contract = _contract(target=compat.C5_TARGET_UNIT)
    archival = _archival(contract)
    validated = compat.validate_c5_realization_archival(
        archival,
        contract=contract,
    )
    assert validated["authorization_identity"]["inode"] == 501
    assert validated["receipt_identity"]["inode"] == 502
    archival["authorization_identity"]["inode"] = 999
    changed = compat.validate_c5_realization_archival(
        archival,
        contract=contract,
    )
    assert changed != validated
    bundle = _bundle()
    with pytest.raises(PermissionError, match="handoff live binding"):
        compat.validate_c5_scope_handoff(
            bundle["handoff"],
            expected_archival=changed,
        )


def test_c5_paths_are_disjoint_and_use_final_contract_names() -> None:
    paths = {
        compat.C5_SCOPE_HANDOFF_PATH,
        compat.C5_STABILITY_ATTEMPT_PATH,
        compat.C5_POLICY_PATH,
        compat.C5_STABILITY_PATH,
        compat.C5_STABILITY_TERMINAL_PATH,
        compat.C5_POSTCLEANUP_PATH,
    }
    assert len(paths) == 6
    assert all("compat_c5" in path.name for path in paths)
    assert all("compat_c3" not in str(path) for path in paths)
    assert compat.C5_HANDOFF_PATH == compat.C5_SCOPE_HANDOFF_PATH
    compat._require_c5_namespace()


def test_high_level_gate_is_unreachable_by_ast_and_runtime() -> None:
    source = Path(compat.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.attr
            if isinstance(function, ast.Attribute)
            else function.id
            if isinstance(function, ast.Name)
            else None
        )
        if name == "run_environment_stability_gate":
            forbidden_calls.append(node.lineno)
    assert forbidden_calls == []
    assert "collect_environment_inventory" not in source
    with pytest.raises(AttributeError):
        getattr(compat.frozen, "run_environment_stability_gate")
    assert "run_environment_stability_gate" not in compat._FROZEN_PROJECTION
    assert "run_environment_stability_gate" not in compat._FROZEN_ALL_CALLABLES


def test_scope_handoff_and_attempt_public_validators_are_closed() -> None:
    bundle = _bundle()
    handoff = compat.validate_c5_scope_handoff(bundle["handoff"])
    assert handoff["changed_fields"] == [
        "require_target_ready",
        "target_unit_id",
    ]
    for field in (
        "D_R_payload_accessed",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "gpu_accessed",
        "training_started",
        "materialization_consumed",
    ):
        assert handoff[field] is False
    runtime = _runtime(bundle)
    pre_contract = bundle["c5"]
    attempt = compat._build_c5_stability_attempt(
        contract=pre_contract,
        roots=bundle["roots"],
        scope_handoff_root=bundle["handoff_root"],
        policy_root=bundle["policy_root"],
        toolchain=_toolchain(),
        phase_guard=bundle["guard"],
        clock=runtime["kwargs"]["clock"],
    )
    assert compat.validate_c5_stability_attempt(attempt) == attempt
    for field in (
        "gpu_accessed",
        "training_started",
        "materialization_consumed",
    ):
        assert attempt[field] is False


def test_direct_gate_commits_before_exact_two_audits_and_one_sleep() -> None:
    bundle = _bundle()
    runtime = _runtime(bundle)
    result = compat.run_c5_stability_in_memory(
        bundle["handoff"],
        bundle["policy"],
        **runtime["kwargs"],
    )
    events = runtime["events"]
    assert result["passed"] is True
    assert result["sample_count"] == 2
    assert result["sample_monotonic_seconds"] == [0.0, 30.0]
    assert events.count("attempt-write") == 1
    assert events.count("audit") == 2
    assert events.count(("sleep", 30.0)) == 1
    assert events.index("attempt-write") < events.index("audit")
    assert events.count("success-write") == 1
    assert "terminal-write" not in events
    assert runtime["state"] == {
        "attempt": True,
        "success": True,
        "terminal": False,
    }
    assert len([event for event in events if event[0] == "guard"]) == 4


def test_stability_receipt_chronology_rejects_resealed_past_timestamp() -> None:
    bundle = _bundle()
    runtime = _runtime(bundle)
    compat.run_c5_stability_in_memory(
        bundle["handoff"],
        bundle["policy"],
        **runtime["kwargs"],
    )
    stability = deepcopy(runtime["evidence"]["stability"])
    stability["created_at_utc"] = stability["samples"][-1][
        "created_at_utc"
    ]
    body = dict(stability)
    body.pop("stability_receipt_fingerprint")
    stability["stability_receipt_fingerprint"] = (
        environment.stable_fingerprint(body)
    )
    assert environment.validate_environment_stability_receipt(
        stability
    ) == stability
    with pytest.raises(PermissionError, match="receipt chronology"):
        compat.validate_c5_environment_closure(
            bundle["handoff"],
            runtime["evidence"]["attempt"],
            bundle["policy"],
            stability,
            None,
            archival=bundle["archival"],
            c5_contract=bundle["c5"],
        )


@pytest.mark.parametrize(
    ("lane", "field"),
    (
        ("authorization_identity", "inode"),
        ("authorization_identity", "ctime_ns"),
        ("receipt_identity", "size"),
        ("receipt_identity", "mtime_ns"),
    ),
)
def test_r4_root_drift_after_first_audit_burns_attempt(
    lane: str,
    field: str,
) -> None:
    bundle = _bundle()
    runtime = _runtime(
        bundle,
        archival_drift_call=3,
        archival_drift_lane=lane,
        archival_drift_field=field,
    )
    with pytest.raises(PermissionError, match="handoff live binding|checkpoint"):
        compat.run_c5_stability_in_memory(
            bundle["handoff"],
            bundle["policy"],
            **runtime["kwargs"],
        )
    assert runtime["events"].count("audit") == 1
    assert runtime["state"] == {
        "attempt": True,
        "success": False,
        "terminal": True,
    }


def test_postcleanup_chronology_requires_stability_before_post() -> None:
    bundle = _bundle()
    runtime = _runtime(bundle)
    stability = compat.run_c5_stability_in_memory(
        bundle["handoff"],
        bundle["policy"],
        **runtime["kwargs"],
    )
    post = compat.build_c5_postcleanup_in_memory(
        bundle["handoff"],
        runtime["evidence"]["attempt"],
        bundle["policy"],
        stability,
        realization_validator=_validator(bundle["archival"]),
        prepare=_prepare(bundle["old"], bundle["roots"]),
    )
    forged = deepcopy(post)
    forged["created_at_utc"] = stability["created_at_utc"]
    body = dict(forged)
    body.pop("receipt_fingerprint")
    forged["receipt_fingerprint"] = environment.stable_fingerprint(body)
    with pytest.raises(PermissionError, match="postcleanup cross-binding"):
        compat.validate_c5_environment_closure(
            bundle["handoff"],
            runtime["evidence"]["attempt"],
            bundle["policy"],
            stability,
            forged,
            archival=bundle["archival"],
            c5_contract=bundle["c5"],
        )


def test_observed_29_999_second_window_is_terminal_failure() -> None:
    bundle = _bundle()
    runtime = _runtime(bundle, monotonic_values=(0.0, 29.999))
    with pytest.raises(PermissionError, match="direct stability"):
        compat.run_c5_stability_in_memory(
            bundle["handoff"],
            bundle["policy"],
            **runtime["kwargs"],
        )
    assert runtime["state"] == {
        "attempt": True,
        "success": False,
        "terminal": True,
    }
    terminal = compat.validate_c5_stability_terminal(
        runtime["evidence"]["terminal"],
        expected_attempt_root=runtime["evidence"]["attempt_root"],
    )
    assert terminal["completed_sample_count"] == 2
    assert terminal["completed_sleep_count"] == 1
    assert terminal["failure_phase"] == "post-sample"


@pytest.mark.parametrize(
    ("drift_call", "expected_audits"),
    ((2, 0), (3, 1), (4, 2)),
)
def test_pre_between_post_guard_drift_burns_attempt_without_retry(
    drift_call: int,
    expected_audits: int,
) -> None:
    bundle = _bundle()
    runtime = _runtime(bundle, guard_drift_call=drift_call)
    with pytest.raises(PermissionError):
        compat.run_c5_stability_in_memory(
            bundle["handoff"],
            bundle["policy"],
            **runtime["kwargs"],
        )
    assert runtime["events"].count("audit") == expected_audits
    assert runtime["events"].count("attempt-write") == 1
    assert runtime["events"].count("terminal-write") == 1
    assert runtime["state"] == {
        "attempt": True,
        "success": False,
        "terminal": True,
    }


def test_source_root_drift_between_samples_is_terminal_failure() -> None:
    bundle = _bundle()
    runtime = _runtime(bundle, prepare_drift_call=3)
    with pytest.raises(PermissionError, match="handoff live binding|checkpoint"):
        compat.run_c5_stability_in_memory(
            bundle["handoff"],
            bundle["policy"],
            **runtime["kwargs"],
        )
    assert runtime["events"].count("audit") == 1
    assert runtime["state"] == {
        "attempt": True,
        "success": False,
        "terminal": True,
    }


def test_consumed_success_cannot_reenter_or_write_a_second_attempt() -> None:
    bundle = _bundle()
    runtime = _runtime(bundle)
    compat.run_c5_stability_in_memory(
        bundle["handoff"],
        bundle["policy"],
        **runtime["kwargs"],
    )
    audit_count = runtime["events"].count("audit")
    with pytest.raises(PermissionError, match="lane state"):
        compat.run_c5_stability_in_memory(
            bundle["handoff"],
            bundle["policy"],
            **runtime["kwargs"],
        )
    assert runtime["events"].count("attempt-write") == 1
    assert runtime["events"].count("audit") == audit_count


def test_run_signature_has_no_gate_or_live_mutation_escape() -> None:
    parameters = inspect.signature(
        compat.run_c5_stability_in_memory
    ).parameters
    assert "gate" not in parameters
    assert "live_roots" not in parameters
    assert {
        "attempt_writer",
        "attempt_reader",
        "success_writer",
        "terminal_writer",
        "lane_state_reader",
    }.issubset(parameters)


def test_public_legacy_writer_remains_closed() -> None:
    with pytest.raises(PermissionError):
        compat._write_create_once(
            compat.C5_SCOPE_HANDOFF_PATH,
            {},
            fingerprint_field="scope_handoff_fingerprint",
        )
    assert "_bind_private_cli" not in vars(compat)


def _isolated_module(tmp_path: Path) -> ModuleType:
    repository = tmp_path / "isolated_repo"
    tools_root = repository / "tools"
    evidence_root = (
        repository
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
    )
    tools_root.mkdir(parents=True, mode=0o700)
    evidence_root.mkdir(parents=True, mode=0o700)
    os.chmod(evidence_root, 0o700)
    source_path = tools_root / Path(compat.__file__).name
    frozen_path = tools_root / compat.FROZEN_ENVIRONMENT_PATH.name
    shutil.copy2(Path(compat.__file__), source_path)
    shutil.copy2(compat.FROZEN_ENVIRONMENT_PATH, frozen_path)
    name = f"isolated_e4_{id(tmp_path)}"
    specification = importlib.util.spec_from_file_location(name, source_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    assert module.REPOSITORY == repository.resolve()
    return module


def _lane_payload(
    module: ModuleType,
    lane: str,
    fingerprint_field: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "lane": lane,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
        "training_started": False,
        "materialization_consumed": False,
    }
    if lane == "handoff":
        body["phase_guard"] = {"closed": True}
    return {
        **body,
        fingerprint_field: module.frozen.stable_fingerprint(body),
    }


def _fd_is_open(path: Path) -> bool:
    for descriptor_path in Path("/proc/self/fd").iterdir():
        try:
            if descriptor_path.resolve() == path:
                return True
        except (FileNotFoundError, OSError):
            continue
    return False


def _prepare_isolated(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, dict[str, object]], list[tuple[str, bool]]]:
    payloads = {
        "handoff": _lane_payload(
            module, "handoff", "scope_handoff_fingerprint"
        ),
        "policy": _lane_payload(
            module, "policy", "policy_fingerprint"
        ),
        "attempt": _lane_payload(
            module, "attempt", "stability_attempt_fingerprint"
        ),
        "stability": _lane_payload(
            module, "stability", "stability_receipt_fingerprint"
        ),
        "terminal": _lane_payload(
            module, "terminal", "stability_terminal_fingerprint"
        ),
        "postcleanup": _lane_payload(
            module, "postcleanup", "receipt_fingerprint"
        ),
    }
    events: list[tuple[str, bool]] = []
    contract = object()
    monkeypatch.setattr(module, "_load_verified_c5_bridge", lambda: object())
    monkeypatch.setattr(
        module,
        "_production_c5_phase_guard",
        lambda: {"closed": True},
    )
    monkeypatch.setattr(
        module,
        "_validate_c5_phase_guard",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        module,
        "_validate_cli_payload",
        lambda payload, *, fingerprint_field: dict(payload),
    )
    monkeypatch.setattr(
        module,
        "replay_old_scope_and_handoff",
        lambda *_args, **_kwargs: (object(), contract, {}),
    )
    monkeypatch.setattr(
        module,
        "_resolve_archival",
        lambda *_args, **_kwargs: {},
    )

    def track(name: str, path: Path):
        def validate(value, *_args, **_kwargs):
            events.append((name, _fd_is_open(path)))
            return dict(value)

        return validate

    monkeypatch.setattr(
        module,
        "validate_c5_scope_handoff",
        track("handoff", module.C5_SCOPE_HANDOFF_PATH),
    )
    monkeypatch.setattr(
        module,
        "_validate_c5_policy_contract",
        track("policy", module.C5_POLICY_PATH),
    )
    monkeypatch.setattr(
        module,
        "validate_c5_stability_attempt",
        track("attempt", module.C5_STABILITY_ATTEMPT_PATH),
    )
    monkeypatch.setattr(
        module,
        "validate_c5_stability_terminal",
        track("terminal", module.C5_STABILITY_TERMINAL_PATH),
    )
    monkeypatch.setattr(
        module,
        "validate_c5_environment_closure",
        track("closure", module.C5_STABILITY_PATH),
    )
    monkeypatch.setattr(
        module,
        "build_c5_scope_handoff_in_memory",
        lambda **_kwargs: dict(payloads["handoff"]),
    )
    monkeypatch.setattr(
        module,
        "build_c5_policy_in_memory",
        lambda _handoff, **_kwargs: dict(payloads["policy"]),
    )
    return payloads, events


def test_fixed_success_writer_revalidates_full_closure_while_fd_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_module(tmp_path)
    payloads, events = _prepare_isolated(module, monkeypatch)
    assert module.main(["create-policy"]) == 0

    def run(_handoff, _policy, **kwargs):
        kwargs["attempt_writer"](dict(payloads["attempt"]))
        kwargs["success_writer"](dict(payloads["stability"]))
        return dict(payloads["stability"])

    monkeypatch.setattr(module, "run_c5_stability_in_memory", run)
    assert module.main(["stability-gate"]) == 0
    assert ("handoff", True) in events
    assert ("policy", True) in events
    assert ("attempt", True) in events
    assert ("closure", True) in events
    assert module.C5_STABILITY_ATTEMPT_PATH.exists()
    assert module.C5_STABILITY_PATH.exists()
    assert not module.C5_STABILITY_TERMINAL_PATH.exists()
    for path in (
        module.C5_SCOPE_HANDOFF_PATH,
        module.C5_POLICY_PATH,
        module.C5_STABILITY_ATTEMPT_PATH,
        module.C5_STABILITY_PATH,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_failure_terminal_is_while_open_and_attempt_blocks_reentry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_module(tmp_path)
    payloads, events = _prepare_isolated(module, monkeypatch)
    assert module.main(["create-policy"]) == 0
    calls = 0

    def run(_handoff, _policy, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["attempt_writer"](dict(payloads["attempt"]))
        kwargs["terminal_writer"](dict(payloads["terminal"]))
        raise PermissionError("injected sampled failure")

    monkeypatch.setattr(module, "run_c5_stability_in_memory", run)
    with pytest.raises(PermissionError, match="injected sampled failure"):
        module.main(["stability-gate"])
    assert ("terminal", True) in events
    assert module.C5_STABILITY_ATTEMPT_PATH.exists()
    assert module.C5_STABILITY_TERMINAL_PATH.exists()
    assert not module.C5_STABILITY_PATH.exists()
    with pytest.raises(PermissionError, match="write order"):
        module.main(["stability-gate"])
    assert calls == 1


def test_attempt_write_failure_burns_generation_and_never_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_module(tmp_path)
    payloads, _events = _prepare_isolated(module, monkeypatch)
    assert module.main(["create-policy"]) == 0
    calls = 0

    def run(_handoff, _policy, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["attempt_writer"](dict(payloads["attempt"]))
        raise AssertionError("attempt writer must fail first")

    monkeypatch.setattr(module, "run_c5_stability_in_memory", run)
    real_write = module.os.write

    def fail_write(_descriptor: int, _raw: bytes) -> int:
        raise OSError("injected attempt write failure")

    monkeypatch.setattr(module.os, "write", fail_write)
    with pytest.raises(OSError, match="injected attempt write failure"):
        module.main(["stability-gate"])
    assert module.C5_STABILITY_ATTEMPT_PATH.exists()
    assert (
        stat.S_IMODE(module.C5_STABILITY_ATTEMPT_PATH.stat().st_mode)
        == 0o600
    )
    monkeypatch.setattr(module.os, "write", real_write)
    with pytest.raises(PermissionError, match="write order"):
        module.main(["stability-gate"])
    assert calls == 1


def test_e5_canonical_profile_preserves_utf8_foreign_text() -> None:
    value = {
        "authorization_basis": "用户指令：修改后继续",
        "generation": "B5→R5→E5",
    }
    encoded = compat._canonical_json(value).encode("utf-8")
    assert "用户指令：修改后继续" in encoded.decode("utf-8")
    assert b"\\u7528" not in encoded
    assert compat._stable_fingerprint(value) == hashlib.sha256(
        encoded
    ).hexdigest()
    source = Path(compat.__file__).read_text(encoding="utf-8")
    assert "ensure_ascii=False" in source
    assert "ensure_ascii=True" not in source
    assert "for ensure_ascii in" not in source


def test_r5_foreign_payload_is_returned_only_by_fixed_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign = {
        "authorization": {
            "authorization_basis": "用户指令：修改后继续",
        },
        "receipt": {"producer": "R5"},
        "authorization_identity": {"path": "/authorization"},
        "receipt_identity": {"path": "/receipt"},
    }

    class Producer:
        @staticmethod
        def validate_archival_realization_chain(
            authorization_path: Path,
            receipt_path: Path,
        ) -> dict[str, object]:
            assert authorization_path == compat.C5_REALIZATION_AUTHORIZATION_PATH
            assert receipt_path == compat.C5_REALIZATION_RECEIPT_PATH
            return deepcopy(foreign)

    monkeypatch.setattr(compat, "_load_verified_c5_bridge", lambda: Producer)
    monkeypatch.setattr(
        compat,
        "_bind_r5_archival_root",
        lambda _path, root: dict(root),
    )
    result = compat._production_archival_validator(
        compat.C5_REALIZATION_AUTHORIZATION_PATH,
        compat.C5_REALIZATION_RECEIPT_PATH,
    )
    assert result["authorization"]["authorization_basis"] == (
        "用户指令：修改后继续"
    )


def test_archival_phase_guard_does_not_recheck_future_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = tmp_path / "future-runtime-output"
    paths = MappingProxyType({"future": future})
    monkeypatch.setattr(compat, "_C4_REQUIRED_ABSENT_PATHS", paths)
    guard = _phase_guard()
    guard["c4_absent_outputs"] = {"future": str(future.absolute())}
    future.write_text("created after E5 archival\n", encoding="utf-8")
    assert compat._validate_c5_phase_guard(guard) == guard
    with pytest.raises(PermissionError, match="absence guard"):
        compat._validate_c5_phase_guard(guard, require_live_absence=True)


def test_load_c5_environment_closure_returns_five_producer_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = {
        "scope_handoff": {"lane": "handoff"},
        "stability_attempt": {"lane": "attempt"},
        "policy": {"lane": "policy"},
        "stability": {"lane": "stability"},
        "postcleanup": {"lane": "postcleanup"},
    }
    lanes = {
        compat.C5_SCOPE_HANDOFF_PATH: (
            "scope_handoff",
            "scope_handoff_fingerprint",
        ),
        compat.C5_STABILITY_ATTEMPT_PATH: (
            "stability_attempt",
            "stability_attempt_fingerprint",
        ),
        compat.C5_POLICY_PATH: ("policy", "policy_fingerprint"),
        compat.C5_STABILITY_PATH: (
            "stability",
            "stability_receipt_fingerprint",
        ),
        compat.C5_POSTCLEANUP_PATH: (
            "postcleanup",
            "receipt_fingerprint",
        ),
    }
    monkeypatch.setattr(compat, "_require_c5_namespace", lambda: None)
    monkeypatch.setattr(
        compat, "_require_frozen_environment_source", lambda: None
    )
    monkeypatch.setattr(compat, "_load_verified_c5_bridge", lambda: object())
    monkeypatch.setattr(
        compat, "_load_verified_c4_failure_terminalizer", lambda: object()
    )
    monkeypatch.setattr(compat, "C4_FAILURE_TERMINAL_SHA256", "a" * 64)
    monkeypatch.setattr(
        compat, "C4_FAILURE_TERMINAL_FINGERPRINT", "b" * 64
    )
    monkeypatch.setattr(compat.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(
        compat,
        "replay_old_scope_and_handoff",
        lambda: (object(), "c5-contract", {}),
    )
    monkeypatch.setattr(
        compat,
        "_production_archival_validator",
        lambda *_args: {"producer": "R5"},
    )

    def read(path: Path, field: str):
        name, expected_field = lanes[path]
        assert field == expected_field
        return deepcopy(payloads[name]), {
            "path": str(path),
            "file_sha256": name[0] * 64,
            field: name[-1] * 64,
        }

    monkeypatch.setattr(compat, "_read_live_sealed", read)
    monkeypatch.setattr(
        compat,
        "validate_c5_environment_closure",
        lambda handoff, attempt, policy, stability, postcleanup, **_kwargs: {
            "scope_handoff": handoff,
            "stability_attempt": attempt,
            "policy": policy,
            "stability": stability,
            "postcleanup": postcleanup,
        },
    )
    loaded = compat.load_c5_environment_closure()
    assert {name: loaded[name] for name in payloads} == payloads
    assert set(loaded["evidence_roots"]) == {
        "environment_scope_handoff",
        "environment_stability_attempt",
        "environment_policy",
        "environment_stability",
        "environment_postcleanup",
    }


def test_e5_cli_and_source_have_no_training_or_retry_lane() -> None:
    parser = compat._parser()
    choices = parser._actions[1].choices
    assert tuple(choices) == (
        "create-policy",
        "stability-gate",
        "postcleanup",
    )
    source = Path(compat.__file__).read_text(encoding="utf-8")
    assert '"automatic_retry_allowed": True' not in source
    assert "retry(" not in source
    assert "train(" not in source
    assert compat.SAMPLE_COUNT == 2
    assert compat.SAMPLE_INTERVAL_SECONDS == 30.0
