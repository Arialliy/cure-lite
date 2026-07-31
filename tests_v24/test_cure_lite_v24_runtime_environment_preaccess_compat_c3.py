from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta
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
    cure_lite_v24_runtime_environment_preaccess_compat_c3 as compat,
)


environment = compat.frozen
UID = 1008
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
        "path": str(path),
        "device": 2304,
        "inode": 100,
        "size": 1000,
        "mtime_ns": 1234,
        "file_sha256": digest,
    }


def _roots() -> dict[str, object]:
    return {
        "precleanup_inventory_receipt": {
            **_file_evidence(
                compat.PRECLEANUP_PATH,
                digest="a" * 64,
            ),
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


def _prepare(old_contract, roots):
    def prepare(*args, **kwargs):
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


def _archival(
    contract,
    *,
    created_at: str = "2026-07-30T00:00:01Z",
) -> dict[str, object]:
    manager = _manager(contract)
    fragment = {
        "path": str(compat.C3_FRAGMENT_PATH),
        "file_sha256": "f" * 64,
    }
    shadow = {
        "Id": compat.C3_TARGET_UNIT,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "Restart": "no",
        "NRestarts": "0",
        "FragmentPath": str(compat.C3_FRAGMENT_PATH),
    }
    return {
        "authorization": {
            "unit_name": compat.C3_TARGET_UNIT,
            "manager_generation": deepcopy(manager),
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        },
        "receipt": {
            "unit_name": compat.C3_TARGET_UNIT,
            "created_at_utc": created_at,
            "manager_generation": deepcopy(manager),
            "fragment_identity": fragment,
            "full_static_shadow": shadow,
            "passed": True,
            "static": True,
            "enabled": False,
            "started": False,
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        },
    }


def _validator(archival):
    def validate(authorization_path: Path, receipt_path: Path):
        assert authorization_path == (
            compat.C3_REALIZATION_AUTHORIZATION_PATH
        )
        assert receipt_path == compat.C3_REALIZATION_RECEIPT_PATH
        return deepcopy(archival)

    return validate


def _shadow(
    unit: str,
    *,
    target: bool,
) -> dict[str, str]:
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
            str(compat.C3_FRAGMENT_PATH)
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
        "created_at_utc": "2026-07-30T00:00:02.500000Z",
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
            "target_unit_id": compat.C3_TARGET_UNIT,
            "conflict_unit_ids": list(compat.CONFLICT_UNIT_IDS),
            "dependency_unit_ids": [],
            "require_target_ready": True,
            "shadows": {
                compat.C3_TARGET_UNIT: _shadow(
                    compat.C3_TARGET_UNIT,
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


def _sample(
    contract,
    inventory,
    *,
    created_at: str,
) -> dict[str, object]:
    sample = environment.audit_environment_once(
        contract,
        inventory_collector=lambda **_kwargs: deepcopy(inventory),
        activation_guard_reader=lambda guard: {
            **dict(guard),
            "file_type": "symlink",
        },
    )
    body = dict(sample)
    body.pop("single_audit_fingerprint")
    body["created_at_utc"] = created_at
    return {
        **body,
        "single_audit_fingerprint": environment.stable_fingerprint(body),
    }


def _policy_and_stability(monkeypatch: pytest.MonkeyPatch):
    old_contract = _contract()
    roots = _roots()
    _old, c3_contract, replayed_roots = (
        compat.replay_old_scope_and_handoff(
            prepare=_prepare(old_contract, roots),
        )
    )
    archival = _archival(c3_contract)
    policy = compat.build_c3_policy_in_memory(
        realization_validator=_validator(archival),
        prepare=_prepare(old_contract, roots),
        toolchain_reader=_toolchain,
    )
    inventory = _inventory(c3_contract)
    policy_time = datetime.fromisoformat(
        str(policy["created_at_utc"])[0:-1] + "+00:00"
    )

    def after_policy(seconds: int) -> str:
        return (
            (policy_time + timedelta(seconds=seconds))
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    samples = [
        _sample(
            c3_contract,
            inventory,
            created_at=after_policy(1),
        ),
        _sample(
            c3_contract,
            inventory,
            created_at=after_policy(2),
        ),
    ]
    stability_roots = deepcopy(replayed_roots)
    stability_roots["policy"] = {
        **_file_evidence(
            compat.C3_POLICY_PATH,
            digest="5" * 64,
        ),
        "policy_fingerprint": policy["policy_fingerprint"],
    }
    stability = environment.evaluate_environment_stability(
        c3_contract,
        stability_roots,
        samples,
        sample_interval_seconds=compat.SAMPLE_INTERVAL_SECONDS,
        sample_monotonic_seconds=[0.0, 30.0],
    )
    stability = environment.validate_environment_stability_receipt(
        stability
    )
    return (
        old_contract,
        c3_contract,
        roots,
        archival,
        policy,
        stability,
    )


def test_frozen_source_is_hash_before_exec_loaded() -> None:
    assert (
        compat.FROZEN_ENVIRONMENT_SHA256
        == "a40465786ce3537346372df5991bb6788d44feddfd497ec83a1dc302fb8b2fea"
    )
    assert compat._FROZEN_LOAD_IDENTITY
    compat._require_frozen_environment_source()


def test_old_scope_replay_then_only_target_and_readiness_handoff() -> None:
    old = _contract()
    old_returned, c3, roots = compat.replay_old_scope_and_handoff(
        prepare=_prepare(old, _roots()),
    )
    assert old_returned.target_unit_id == compat.OLD_TARGET_UNIT
    assert old_returned.require_target_ready is False
    assert c3.target_unit_id == compat.C3_TARGET_UNIT
    assert c3.require_target_ready is True
    for field, old_value in asdict(old_returned).items():
        if field not in {"target_unit_id", "require_target_ready"}:
            assert asdict(c3)[field] == old_value
    assert roots == _roots()


@pytest.mark.parametrize(
    "drift",
    ("target", "require", "other_scope"),
)
def test_old_scope_replay_rejects_scope_drift(drift: str) -> None:
    old = _contract()
    if drift == "target":
        old = replace(old, target_unit_id=compat.C3_TARGET_UNIT)
    elif drift == "require":
        old = replace(old, require_target_ready=True)
    else:
        old = replace(old, allowed_manager_states=("running",))
    with pytest.raises(PermissionError):
        compat.replay_old_scope_and_handoff(
            prepare=_prepare(old, _roots()),
        )


def test_policy_is_fresh_c3_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _old,
        c3,
        roots,
        archival,
        policy,
        _stability,
    ) = _policy_and_stability(monkeypatch)
    assert policy["unit_scope"] == {
        "target_unit_id": compat.C3_TARGET_UNIT,
        "conflict_unit_ids": list(compat.CONFLICT_UNIT_IDS),
        "dependency_unit_ids": [],
        "allowed_failed_unit_ids": list(
            compat.ALLOWED_FAILED_UNIT_IDS
        ),
        "allowed_unit_ids": list(compat.ALLOWED_UNIT_IDS),
        "expected_failed_unit_ids": list(
            compat.ALLOWED_FAILED_UNIT_IDS
        ),
        "require_target_ready": True,
    }
    assert policy["sampling"]["minimum_sample_count"] == 2
    assert policy["sampling"]["sample_interval_seconds"] == 30.0
    assert policy["precleanup_root"] == roots[
        "precleanup_inventory_receipt"
    ]
    compat.validate_c3_realization_archival(archival, contract=c3)


def test_realization_bridge_source_is_production_frozen() -> None:
    digest = hashlib.sha256(compat.C3_BRIDGE_PATH.read_bytes()).hexdigest()
    assert compat.C3_BRIDGE_SHA256 == digest
    assert compat.C3_BRIDGE_SHA256 != "__TO_BE_FROZEN__"
    module = compat._load_verified_c3_bridge()
    assert callable(module.validate_archival_realization_chain)


def test_stability_uses_frozen_gate_directly_and_exact_2x30(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        old,
        _c3,
        roots,
        archival,
        policy,
        expected_stability,
    ) = _policy_and_stability(monkeypatch)
    observed: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def gate(*args, **kwargs):
        observed.append((args, dict(kwargs)))
        assert tuple(Path(item).absolute() for item in args) == (
            compat.PRECLEANUP_PATH.absolute(),
            compat.CLEANUP_RECEIPT_PATH.absolute(),
        )
        assert kwargs["target_unit_id"] == compat.C3_TARGET_UNIT
        assert kwargs["require_target_ready"] is True
        assert kwargs["sample_count"] == 2
        assert kwargs["sample_interval_seconds"] == 30.0
        return deepcopy(expected_stability)

    result = compat.run_c3_stability_in_memory(
        policy,
        realization_validator=_validator(archival),
        prepare=_prepare(old, roots),
        gate=gate,
    )
    assert result == expected_stability
    assert len(observed) == 1
    assert "_patched_prepare" not in vars(compat)


def test_postcleanup_reuses_exact_final_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        old,
        c3,
        roots,
        archival,
        policy,
        stability,
    ) = _policy_and_stability(monkeypatch)
    final_time = datetime.fromisoformat(
        str(stability["samples"][-1]["created_at_utc"])[0:-1]
        + "+00:00"
    )
    post_time = (
        (final_time + timedelta(seconds=1))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    post = compat.build_c3_postcleanup_in_memory(
        policy,
        stability,
        realization_validator=_validator(archival),
        prepare=_prepare(old, roots),
        clock=lambda: post_time,
    )
    assert post["inventory"] == stability["samples"][-1]["inventory"]
    closed = compat.validate_c3_environment_closure(
        policy,
        stability,
        post,
        archival=archival,
        c3_contract=c3,
    )
    assert closed["postcleanup"] == post
    assert all(
        post[field] is False
        for field in (
            "D_R_payload_accessed",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
        )
    )


@pytest.mark.parametrize(
    "drift",
    ("target", "require", "other_scope", "manager", "fragment", "chronology"),
)
def test_cross_binding_rejects_resealed_or_archival_drift(
    drift: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _old,
        c3,
        _roots_value,
        archival,
        policy,
        stability,
    ) = _policy_and_stability(monkeypatch)
    supplied_contract = c3
    if drift == "target":
        supplied_contract = replace(
            c3,
            target_unit_id=compat.OLD_TARGET_UNIT,
        )
    elif drift == "require":
        supplied_contract = replace(c3, require_target_ready=False)
    elif drift == "other_scope":
        supplied_contract = replace(
            c3,
            allowed_manager_states=("running",),
        )
    elif drift == "manager":
        archival["receipt"]["manager_generation"]["identity"][
            "starttime_ticks"
        ] += 1
    elif drift == "fragment":
        archival["receipt"]["fragment_identity"]["path"] = (
            "/run/user/1008/systemd/user/wrong.service"
        )
    else:
        archival["receipt"]["created_at_utc"] = (
            str(stability["samples"][-1]["created_at_utc"])
        )
    with pytest.raises((PermissionError, ValueError)):
        compat.validate_c3_environment_closure(
            policy,
            stability,
            None,
            archival=archival,
            c3_contract=supplied_contract,
        )


def test_public_and_legacy_write_capabilities_are_closed() -> None:
    assert compat.frozen is compat._frozen
    for view in (compat.frozen, compat._frozen):
        for entry in (
            "main",
            "write_create_once_receipt",
            "write_environment_policy",
        ):
            with pytest.raises(AttributeError):
                getattr(view, entry)
        for entry in (
            "_module",
            "_GuardedFrozenView__module",
            "_write_create_once",
        ):
            with pytest.raises(AttributeError):
                getattr(view, entry)
    with pytest.raises(PermissionError):
        compat._write_create_once(
            compat.C3_POLICY_PATH,
            {},
            fingerprint_field="policy_fingerprint",
        )
    assert "_context" not in inspect.signature(
        compat._write_create_once
    ).parameters
    assert "_WRITE_CONTEXT" not in vars(compat)
    assert "_loaded_frozen" not in vars(compat)
    assert "_bind_private_cli" not in vars(compat)
    assert (
        "tools._cure_lite_v24_runtime_environment_frozen_"
        "for_preaccess_compat_c3"
    ) not in sys.modules


def test_frozen_source_and_r3_pin_reject_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(compat, "FROZEN_ENVIRONMENT_SHA256", "0" * 64)
        with pytest.raises(PermissionError):
            compat._require_frozen_environment_source()
    with monkeypatch.context() as scoped:
        scoped.setattr(compat, "C3_BRIDGE_SHA256", "__TO_BE_FROZEN__")
        with pytest.raises(PermissionError):
            compat._load_verified_c3_bridge()


@pytest.mark.parametrize("alias", ("frozen", "_frozen"))
def test_frozen_facade_alias_replacement_is_rejected(
    alias: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(compat, alias, object())
        with pytest.raises(PermissionError, match="facade generation"):
            compat._require_frozen_environment_source()
    compat._require_frozen_environment_source()


def test_frozen_facade_projection_replacement_is_rejected() -> None:
    facade = compat.frozen
    slot = "_FrozenReadFacade__projection"
    original = object.__getattribute__(facade, slot)
    replacement = MappingProxyType(dict(original))
    assert replacement is not original
    object.__setattr__(facade, slot, replacement)
    try:
        with pytest.raises(PermissionError, match="facade generation"):
            compat._require_frozen_environment_source()
    finally:
        object.__setattr__(facade, slot, original)
    compat._require_frozen_environment_source()


def test_main_rejects_fixed_path_identity_drift_before_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reached = False

    def build(**_kwargs):
        nonlocal reached
        reached = True
        raise AssertionError("builder must stay unreachable")

    monkeypatch.setattr(compat, "build_c3_policy_in_memory", build)
    monkeypatch.setattr(
        compat,
        "C3_POLICY_PATH",
        tmp_path / "runtime_environment_policy_preaccess_compat_c3.json",
    )
    with pytest.raises(PermissionError, match="fixed identity changed"):
        compat.main(["create-policy"])
    assert reached is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("command", "state", "order_valid"),
    (
        ("create-policy", (False, False, False), True),
        ("create-policy", (True, False, False), False),
        ("stability-gate", (True, False, False), True),
        ("stability-gate", (False, False, False), False),
        ("postcleanup", (True, True, False), True),
        ("postcleanup", (True, False, False), False),
    ),
)
def test_main_enforces_policy_stability_postcleanup_order_before_payload(
    command: str,
    state: tuple[bool, bool, bool],
    order_valid: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = (
        compat.C3_POLICY_PATH,
        compat.C3_STABILITY_PATH,
        compat.C3_POSTCLEANUP_PATH,
    )
    observed = {path: present for path, present in zip(paths, state)}

    def reached(*_args, **_kwargs):
        raise RuntimeError("order accepted")

    monkeypatch.setattr(
        compat.os.path,
        "lexists",
        lambda path: observed.get(Path(path), False),
    )
    monkeypatch.setattr(
        compat,
        "_load_verified_c3_bridge",
        lambda: object(),
    )
    monkeypatch.setattr(
        compat,
        "build_c3_policy_in_memory",
        reached,
    )
    monkeypatch.setattr(compat, "_read_live_sealed", reached)
    if order_valid:
        with pytest.raises(RuntimeError, match="order accepted"):
            compat.main([command])
    else:
        with pytest.raises(PermissionError, match="write order"):
            compat.main([command])


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("D_R_cached_tensor_payload_accessed", True),
        ("D_V_tensor_payload_accessed", True),
        ("D_T_payload_accessed", 1),
        ("training_started", True),
        ("optimizer_steps", 1),
        ("parameter_updates", False),
        ("gpu_compute_performed", True),
        ("gpu_kernel_launches", 1),
    ),
)
def test_nested_evidence_rejects_payload_training_or_gpu_compute(
    field: str,
    value: object,
) -> None:
    evidence = {
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "nested": [{"deeper": {field: value}}],
    }
    with pytest.raises(PermissionError):
        compat._no_payload(evidence, name="nested evidence")


def test_nested_evidence_accepts_only_false_or_zero_controls() -> None:
    evidence = {
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "nested": [
            {
                "D_V_tensor_payload_accessed": False,
                "training_started": False,
                "optimizer_steps": 0,
                "parameter_updates": 0,
                "gpu_compute_performed": False,
                "gpu_kernel_launches": 0,
            }
        ],
    }
    compat._no_payload(evidence, name="nested evidence")

def _isolated_compatibility_module(tmp_path: Path) -> ModuleType:
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
    bridge_path = tools_root / compat.C3_BRIDGE_PATH.name
    shutil.copy2(Path(compat.__file__), source_path)
    shutil.copy2(compat.FROZEN_ENVIRONMENT_PATH, frozen_path)
    shutil.copy2(compat.C3_BRIDGE_PATH, bridge_path)
    module_name = f"isolated_e3_{id(tmp_path)}"
    specification = importlib.util.spec_from_file_location(
        module_name,
        source_path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    assert module.REPOSITORY == repository.resolve()
    assert module.EVIDENCE_ROOT.is_relative_to(repository.resolve())
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
        "training_started": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
    }
    return {
        **body,
        fingerprint_field: module.frozen.stable_fingerprint(body),
    }


def _prepare_isolated_cli(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, object]]:
    payloads = {
        "policy": _lane_payload(
            module, "policy", "policy_fingerprint"
        ),
        "stability": _lane_payload(
            module,
            "stability",
            "stability_receipt_fingerprint",
        ),
        "postcleanup": _lane_payload(
            module, "postcleanup", "receipt_fingerprint"
        ),
    }
    monkeypatch.setattr(
        module,
        "_load_verified_c3_bridge",
        lambda: object(),
    )
    monkeypatch.setattr(
        module,
        "_validate_cli_payload",
        lambda payload, *, fingerprint_field: dict(payload),
    )
    monkeypatch.setattr(
        module,
        "build_c3_policy_in_memory",
        lambda **_kwargs: dict(payloads["policy"]),
    )
    monkeypatch.setattr(
        module,
        "run_c3_stability_in_memory",
        lambda _policy, **_kwargs: dict(payloads["stability"]),
    )
    monkeypatch.setattr(
        module,
        "build_c3_postcleanup_in_memory",
        lambda _policy, _stability, **_kwargs: dict(
            payloads["postcleanup"]
        ),
    )
    contract = object()
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
    monkeypatch.setattr(
        module,
        "_validate_c3_policy_contract",
        lambda policy, **_kwargs: dict(policy),
    )
    monkeypatch.setattr(
        module,
        "validate_c3_environment_closure",
        lambda *_args, **_kwargs: {},
    )
    return payloads


def test_fixed_three_lane_writer_seals_exact_generations_in_tmp_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_compatibility_module(tmp_path)
    payloads = _prepare_isolated_cli(module, monkeypatch)
    mode_transitions: list[int] = []
    create_calls: list[tuple[int, int]] = []
    real_fchmod = module.os.fchmod
    real_open = module.os.open

    def tracked_fchmod(descriptor: int, mode: int) -> None:
        mode_transitions.append(mode)
        real_fchmod(descriptor, mode)

    def tracked_open(
        path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if flags & module.os.O_CREAT:
            create_calls.append((flags, mode))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "fchmod", tracked_fchmod)
    monkeypatch.setattr(module.os, "open", tracked_open)
    assert module.main(["create-policy"]) == 0
    assert module.main(["stability-gate"]) == 0
    assert module.main(["postcleanup"]) == 0
    assert mode_transitions == [0o600, 0o444] * 3
    assert len(create_calls) == 3
    for flags, mode in create_calls:
        assert flags & module.os.O_EXCL
        assert flags & module.os.O_NOFOLLOW
        assert mode == 0o600
    lanes = (
        (module.C3_POLICY_PATH, "policy_fingerprint", payloads["policy"]),
        (
            module.C3_STABILITY_PATH,
            "stability_receipt_fingerprint",
            payloads["stability"],
        ),
        (
            module.C3_POSTCLEANUP_PATH,
            "receipt_fingerprint",
            payloads["postcleanup"],
        ),
    )
    for path, fingerprint_field, expected in lanes:
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert path.stat().st_nlink == 1
        observed, root = module._read_live_sealed(
            path, fingerprint_field
        )
        assert observed == expected
        assert set(root) == {
            "path",
            "device",
            "inode",
            "size",
            "mtime_ns",
            "file_sha256",
            fingerprint_field,
        }
        assert path.read_bytes() == (
            module.frozen.canonical_json(expected) + "\n"
        ).encode("utf-8")

def test_fixed_writer_failure_is_preserved_and_never_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_compatibility_module(tmp_path)
    payloads = _prepare_isolated_cli(module, monkeypatch)
    build_calls = 0

    def build(**_kwargs):
        nonlocal build_calls
        build_calls += 1
        return dict(payloads["policy"])

    monkeypatch.setattr(module, "build_c3_policy_in_memory", build)
    real_write = module.os.write

    def fail_write(_descriptor: int, _raw: bytes) -> int:
        raise OSError("injected fixed-lane write failure")

    monkeypatch.setattr(module.os, "write", fail_write)
    with pytest.raises(OSError, match="injected fixed-lane"):
        module.main(["create-policy"])
    assert build_calls == 1
    assert module.C3_POLICY_PATH.exists()
    assert stat.S_IMODE(module.C3_POLICY_PATH.stat().st_mode) == 0o600
    assert module.C3_POLICY_PATH.stat().st_size == 0
    monkeypatch.setattr(module.os, "write", real_write)
    with pytest.raises(PermissionError, match="write order"):
        module.main(["create-policy"])
    assert build_calls == 1
    assert module.C3_POLICY_PATH.exists()

def test_postcleanup_cross_roots_are_verified_while_output_fd_is_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_compatibility_module(tmp_path)
    _prepare_isolated_cli(module, monkeypatch)
    assert module.main(["create-policy"]) == 0
    assert module.main(["stability-gate"]) == 0
    real_verify = module._verify_live_sealed
    events: list[tuple[Path, bool, bool]] = []

    def output_fd_is_open() -> bool:
        for descriptor_path in Path("/proc/self/fd").iterdir():
            try:
                if descriptor_path.resolve() == module.C3_POSTCLEANUP_PATH:
                    return True
            except (FileNotFoundError, OSError):
                continue
        return False

    def tracked_verify(path, evidence, fingerprint_field):
        target = Path(path).absolute()
        events.append(
            (
                target,
                module.C3_POSTCLEANUP_PATH.exists(),
                output_fd_is_open(),
            )
        )
        return real_verify(path, evidence, fingerprint_field)

    monkeypatch.setattr(module, "_verify_live_sealed", tracked_verify)
    assert module.main(["postcleanup"]) == 0
    guarded_while_open = [
        event
        for event in events
        if event[0]
        in {module.C3_POLICY_PATH, module.C3_STABILITY_PATH}
        and event[1]
        and event[2]
    ]
    assert {event[0] for event in guarded_while_open} == {
        module.C3_POLICY_PATH,
        module.C3_STABILITY_PATH,
    }
    guarded_after_close = [
        event
        for event in events
        if event[0]
        in {module.C3_POLICY_PATH, module.C3_STABILITY_PATH}
        and event[1]
        and not event[2]
    ]
    assert {event[0] for event in guarded_after_close} == {
        module.C3_POLICY_PATH,
        module.C3_STABILITY_PATH,
    }
    final_output_checks = [
        event for event in events if event[0] == module.C3_POSTCLEANUP_PATH
    ]
    assert final_output_checks
    assert all(not event[2] for event in final_output_checks)

def test_postcleanup_guard_tamper_fails_closed_and_burns_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_compatibility_module(tmp_path)
    _prepare_isolated_cli(module, monkeypatch)
    assert module.main(["create-policy"]) == 0
    assert module.main(["stability-gate"]) == 0
    real_verify = module._verify_live_sealed
    tampered = False
    observed_open_fd = False

    def tamper_during_guard(path, evidence, fingerprint_field):
        nonlocal tampered, observed_open_fd
        target = Path(path).absolute()
        if (
            target == module.C3_POLICY_PATH
            and module.C3_POSTCLEANUP_PATH.exists()
            and not tampered
        ):
            for descriptor_path in Path("/proc/self/fd").iterdir():
                try:
                    observed_open_fd = (
                        observed_open_fd
                        or descriptor_path.resolve()
                        == module.C3_POSTCLEANUP_PATH
                    )
                except (FileNotFoundError, OSError):
                    continue
            module.os.chmod(module.C3_POLICY_PATH, 0o600)
            tampered = True
        return real_verify(path, evidence, fingerprint_field)

    monkeypatch.setattr(
        module, "_verify_live_sealed", tamper_during_guard
    )
    with pytest.raises(PermissionError):
        module.main(["postcleanup"])
    assert tampered is True
    assert observed_open_fd is True
    assert module.C3_POSTCLEANUP_PATH.exists()
    assert (
        stat.S_IMODE(module.C3_POSTCLEANUP_PATH.stat().st_mode)
        == 0o444
    )
    with pytest.raises(PermissionError, match="write order"):
        module.main(["postcleanup"])

def test_live_root_exact7_rejects_shape_and_mode_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_compatibility_module(tmp_path)
    _prepare_isolated_cli(module, monkeypatch)
    assert module.main(["create-policy"]) == 0
    _payload, root = module._read_live_sealed(
        module.C3_POLICY_PATH, "policy_fingerprint"
    )
    with pytest.raises(PermissionError):
        module._verify_exact_live_roots(
            {"policy": {**root, "extra": "forbidden"}},
            required=frozenset({"policy"}),
        )
    module.os.chmod(module.C3_POLICY_PATH, 0o400)
    with pytest.raises(PermissionError):
        module._verify_live_sealed(
            module.C3_POLICY_PATH,
            root,
            "policy_fingerprint",
        )


def test_main_closure_cannot_reach_an_arbitrary_path_writer() -> None:
    assert tuple(inspect.signature(compat.main).parameters) == ("argv",)
    pending = [
        cell.cell_contents
        for cell in (compat.main.__closure__ or ())
    ]
    visited: set[int] = set()
    reachable_functions = []
    while pending:
        value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        if inspect.isfunction(value):
            reachable_functions.append(value)
            pending.extend(
                cell.cell_contents
                for cell in (value.__closure__ or ())
            )
        elif isinstance(value, (tuple, list, frozenset)):
            pending.extend(value)
        elif isinstance(value, Mapping):
            pending.extend(value.values())
    assert reachable_functions
    for function in reachable_functions:
        assert "O_CREAT" not in function.__code__.co_names
        assert "fchmod" not in function.__code__.co_names
        assert "write_fixed_lane" not in function.__code__.co_names
    assert "_bind_private_cli" not in vars(compat)
    assert "write_fixed_lane" not in vars(compat)


def _seal_fixed_environment_payload(path: Path, payload: Mapping[str, object]) -> None:
    path.write_bytes((environment.canonical_json(payload) + "\n").encode("utf-8"))
    path.chmod(0o444)


def _fixed_environment_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    policy_path = tmp_path / "runtime_environment_policy_preaccess_compat_c3.json"
    stability_path = tmp_path / "runtime_environment_stability_receipt_preaccess_compat_c3.json"
    monkeypatch.setattr(compat, "C3_POLICY_PATH", policy_path)
    monkeypatch.setattr(compat, "C3_STABILITY_PATH", stability_path)
    monkeypatch.setattr(
        compat,
        "_LIVE_ROOT_SPECS",
        {
            "policy": (policy_path, "policy_fingerprint"),
            "stability": (
                stability_path, "stability_receipt_fingerprint"
            ),
        },
    )
    old, c3, roots, archival, policy, stability = _policy_and_stability(monkeypatch)
    return old, c3, roots, archival, policy, stability


def _seal_fixed_environment_values(policy, stability):
    _seal_fixed_environment_payload(compat.C3_POLICY_PATH, policy)
    _loaded, policy_root = compat._read_live_sealed(
        compat.C3_POLICY_PATH, "policy_fingerprint"
    )
    body = deepcopy(stability)
    body["root_evidence"]["policy"] = policy_root
    body.pop("stability_receipt_fingerprint")
    closed = {
        **body,
        "stability_receipt_fingerprint": environment.stable_fingerprint(body),
    }
    closed = environment.validate_environment_stability_receipt(closed)
    _seal_fixed_environment_payload(compat.C3_STABILITY_PATH, closed)
    return closed


def _fixed_post(old, roots, archival, policy, stability):
    final_time = datetime.fromisoformat(
        str(stability["samples"][-1]["created_at_utc"])[0:-1] + "+00:00"
    )
    return compat.build_c3_postcleanup_in_memory(
        policy,
        stability,
        realization_validator=_validator(archival),
        prepare=_prepare(old, roots),
        clock=lambda: (
            (final_time + timedelta(seconds=1))
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        ),
    )


def test_closure_none_roots_auto_binds_fixed_policy_and_stability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old, c3, roots, archival, policy, stability = _fixed_environment_values(
        monkeypatch, tmp_path
    )
    stability = _seal_fixed_environment_values(policy, stability)
    post = _fixed_post(old, roots, archival, policy, stability)
    original_read = compat._read_live_sealed
    observed: list[Path] = []
    def tracked(path: Path, fingerprint_field: str):
        observed.append(Path(path))
        return original_read(path, fingerprint_field)
    monkeypatch.setattr(compat, "_read_live_sealed", tracked)
    result = compat.validate_c3_environment_closure(
        policy, stability, post, archival=archival, c3_contract=c3
    )
    assert result["postcleanup"] == post
    assert observed == [compat.C3_POLICY_PATH, compat.C3_STABILITY_PATH]


@pytest.mark.parametrize("drift", ("same-content-new-inode", "payload"))
def test_closure_none_roots_rejects_replaced_fixed_policy(
    drift: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old, c3, roots, archival, policy, stability = _fixed_environment_values(
        monkeypatch, tmp_path
    )
    stability = _seal_fixed_environment_values(policy, stability)
    post = _fixed_post(old, roots, archival, policy, stability)
    old_inode = compat.C3_POLICY_PATH.stat().st_ino
    replacement = tmp_path / "replacement-policy.json"
    replacement_payload = deepcopy(policy)
    if drift == "payload":
        body = dict(replacement_payload)
        body.pop("policy_fingerprint")
        body["created_at_utc"] = "2026-07-30T00:00:03Z"
        replacement_payload = {
            **body, "policy_fingerprint": environment.stable_fingerprint(body)
        }
        environment.validate_environment_policy(replacement_payload)
    _seal_fixed_environment_payload(replacement, replacement_payload)
    os.replace(replacement, compat.C3_POLICY_PATH)
    assert compat.C3_POLICY_PATH.stat().st_ino != old_inode
    with pytest.raises(PermissionError):
        compat.validate_c3_environment_closure(
            policy, stability, post, archival=archival, c3_contract=c3
        )


@pytest.mark.parametrize("missing", ("policy", "stability"))
def test_closure_none_roots_rejects_partial_fixed_state(
    missing: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old, c3, roots, archival, policy, stability = _fixed_environment_values(
        monkeypatch, tmp_path
    )
    stability = _seal_fixed_environment_values(policy, stability)
    post = _fixed_post(old, roots, archival, policy, stability)
    path = compat.C3_POLICY_PATH if missing == "policy" else compat.C3_STABILITY_PATH
    path.unlink()
    with pytest.raises(PermissionError, match="partial|incomplete"):
        compat.validate_c3_environment_closure(
            policy, stability, post, archival=archival, c3_contract=c3
        )


def test_closure_none_roots_with_no_fixed_lanes_is_in_memory_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old, c3, roots, archival, policy, stability = _fixed_environment_values(
        monkeypatch, tmp_path
    )
    post = _fixed_post(old, roots, archival, policy, stability)
    assert not compat.C3_POLICY_PATH.exists()
    assert not compat.C3_STABILITY_PATH.exists()
    result = compat.validate_c3_environment_closure(
        policy, stability, post, archival=archival, c3_contract=c3
    )
    assert result["postcleanup"] == post


def test_closure_explicit_policy_root_auto_upgrades_fixed_tt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _old, c3, _roots_value, archival, policy, stability = (
        _fixed_environment_values(monkeypatch, tmp_path)
    )
    stability = _seal_fixed_environment_values(policy, stability)
    _loaded_policy, policy_root = compat._read_live_sealed(
        compat.C3_POLICY_PATH, "policy_fingerprint"
    )
    original_read = compat._read_live_sealed
    observed: list[Path] = []
    def tracked(path: Path, fingerprint_field: str):
        observed.append(Path(path))
        return original_read(path, fingerprint_field)
    monkeypatch.setattr(compat, "_read_live_sealed", tracked)
    result = compat.validate_c3_environment_closure(
        policy,
        stability,
        None,
        archival=archival,
        c3_contract=c3,
        live_roots={"policy": policy_root},
    )
    assert result["stability"] == stability
    assert observed[:2] == [
        compat.C3_POLICY_PATH, compat.C3_STABILITY_PATH
    ]


def test_closure_explicit_policy_root_cannot_skip_unsealed_fixed_stability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _old, c3, _roots_value, archival, policy, stability = (
        _fixed_environment_values(monkeypatch, tmp_path)
    )
    stability = _seal_fixed_environment_values(policy, stability)
    _loaded_policy, policy_root = compat._read_live_sealed(
        compat.C3_POLICY_PATH, "policy_fingerprint"
    )
    compat.C3_STABILITY_PATH.chmod(0o400)
    with pytest.raises(PermissionError):
        compat.validate_c3_environment_closure(
            policy,
            stability,
            None,
            archival=archival,
            c3_contract=c3,
            live_roots={"policy": policy_root},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("payload_authority", {}),
        ("payload_authority", "full"),
        ("gpu_authority", {"mode": "none"}),
        ("gpu_authority", "compute"),
        ("training_authority", {"allowed": False}),
        ("training_authority", "allowed"),
    ),
)
def test_strict_authority_fields_reject_mappings_and_non_none_values(
    field: str, value: object
) -> None:
    evidence = {
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "nested": {"deeper": {field: value}},
    }
    with pytest.raises(PermissionError):
        compat._no_payload(evidence, name="strict authority evidence")


def test_mutation_authority_mapping_recurses_but_cannot_hide_authority() -> None:
    safe = {
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "mutation_authority": {
            "phase": {
                "payload_authority": "none",
                "gpu_authority": None,
                "training_authority": "none",
                "optimizer_steps": 0,
            }
        },
    }
    compat._no_payload(safe, name="recursive mutation evidence")
    unsafe = deepcopy(safe)
    unsafe["mutation_authority"]["phase"]["gpu_authority"] = "compute"
    with pytest.raises(PermissionError):
        compat._no_payload(unsafe, name="recursive mutation evidence")


@pytest.mark.parametrize(
    ("command", "pre_lanes", "output_lane"),
    (
        ("stability-gate", ("policy",), "stability"),
        ("postcleanup", ("policy", "stability"), "postcleanup"),
    ),
)
def test_fixed_writer_replays_contract_archival_and_closure_while_fd_open(
    command: str,
    pre_lanes: tuple[str, ...],
    output_lane: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_compatibility_module(tmp_path)
    payloads = _prepare_isolated_cli(module, monkeypatch)
    lane_paths = {
        "policy": module.C3_POLICY_PATH,
        "stability": module.C3_STABILITY_PATH,
        "postcleanup": module.C3_POSTCLEANUP_PATH,
    }
    lane_fields = {
        "policy": "policy_fingerprint",
        "stability": "stability_receipt_fingerprint",
        "postcleanup": "receipt_fingerprint",
    }
    for lane in pre_lanes:
        _seal_fixed_environment_payload(lane_paths[lane], payloads[lane])
    output_path = lane_paths[output_lane]
    events: list[tuple[str, bool]] = []
    def output_fd_is_open() -> bool:
        for descriptor_path in Path("/proc/self/fd").iterdir():
            try:
                if descriptor_path.resolve() == output_path:
                    return True
            except (FileNotFoundError, OSError):
                continue
        return False
    contract = object()
    def replay(*_args, **_kwargs):
        events.append(("contract", output_fd_is_open()))
        return object(), contract, {}
    def archival(*_args, **_kwargs):
        events.append(("archival", output_fd_is_open()))
        return {"closed": True}
    def closure(*_args, **_kwargs):
        events.append(("closure", output_fd_is_open()))
        return {"closed": True}
    monkeypatch.setattr(module, "replay_old_scope_and_handoff", replay)
    monkeypatch.setattr(module, "_resolve_archival", archival)
    monkeypatch.setattr(module, "validate_c3_environment_closure", closure)
    monkeypatch.setattr(
        module,
        "_validate_c3_policy_contract",
        lambda policy, **_kwargs: dict(policy),
    )
    assert module.main([command]) == 0
    assert {name for name, _open in events} == {
        "contract", "archival", "closure"
    }
    assert all(is_open for _name, is_open in events)
    assert output_path.exists()
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o444
    assert lane_fields[output_lane] in payloads[output_lane]


def test_future_command_cannot_fall_through_to_postcleanup_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _isolated_compatibility_module(tmp_path)
    payloads = _prepare_isolated_cli(module, monkeypatch)
    _seal_fixed_environment_payload(module.C3_POLICY_PATH, payloads["policy"])
    _seal_fixed_environment_payload(
        module.C3_STABILITY_PATH, payloads["stability"]
    )
    class FutureArguments:
        command = "future-lane"
    class FutureParser:
        def parse_args(self, _argv):
            return FutureArguments()
    monkeypatch.setattr(module, "_parser", lambda: FutureParser())
    with pytest.raises(PermissionError):
        module.main([])
    assert not module.C3_POSTCLEANUP_PATH.exists()


def test_policy_writer_replays_contract_archival_and_policy_while_fd_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isolated_compatibility_module(tmp_path)
    _prepare_isolated_cli(module, monkeypatch)
    output_path = module.C3_POLICY_PATH
    events: list[tuple[str, bool]] = []

    def output_fd_is_open() -> bool:
        for descriptor_path in Path("/proc/self/fd").iterdir():
            try:
                if descriptor_path.resolve() == output_path:
                    return True
            except (FileNotFoundError, OSError):
                continue
        return False

    contract = object()

    def replay(*_args, **_kwargs):
        events.append(("contract", output_fd_is_open()))
        return object(), contract, {}

    def archival(*_args, **_kwargs):
        events.append(("archival", output_fd_is_open()))
        return {"closed": True}

    def validate_policy(policy, **_kwargs):
        events.append(("policy", output_fd_is_open()))
        return dict(policy)

    monkeypatch.setattr(module, "replay_old_scope_and_handoff", replay)
    monkeypatch.setattr(module, "_resolve_archival", archival)
    monkeypatch.setattr(
        module,
        "_validate_c3_policy_contract",
        validate_policy,
    )
    assert module.main(["create-policy"]) == 0
    assert {name for name, _is_open in events} == {
        "contract",
        "archival",
        "policy",
    }
    assert all(is_open for _name, is_open in events)
    assert output_path.exists()
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o444
