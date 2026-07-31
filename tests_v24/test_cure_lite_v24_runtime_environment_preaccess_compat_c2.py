from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from typing import Mapping

import pytest

from tools import (
    cure_lite_v24_runtime_environment_preaccess_compat_c2 as compat,
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
        "path": str(compat.C2_FRAGMENT_PATH),
        "file_sha256": "f" * 64,
    }
    shadow = {
        "Id": compat.C2_TARGET_UNIT,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "Restart": "no",
        "NRestarts": "0",
        "FragmentPath": str(compat.C2_FRAGMENT_PATH),
    }
    return {
        "authorization": {
            "unit_name": compat.C2_TARGET_UNIT,
            "manager_generation": deepcopy(manager),
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        },
        "receipt": {
            "unit_name": compat.C2_TARGET_UNIT,
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
            compat.C2_REALIZATION_AUTHORIZATION_PATH
        )
        assert receipt_path == compat.C2_REALIZATION_RECEIPT_PATH
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
            str(compat.C2_FRAGMENT_PATH)
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
            "target_unit_id": compat.C2_TARGET_UNIT,
            "conflict_unit_ids": list(compat.CONFLICT_UNIT_IDS),
            "dependency_unit_ids": [],
            "require_target_ready": True,
            "shadows": {
                compat.C2_TARGET_UNIT: _shadow(
                    compat.C2_TARGET_UNIT,
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
    _old, c2_contract, replayed_roots = (
        compat.replay_old_scope_and_handoff(
            prepare=_prepare(old_contract, roots),
        )
    )
    archival = _archival(c2_contract)
    monkeypatch.setattr(
        environment,
        "utc_now",
        lambda: "2026-07-30T00:00:02Z",
    )
    policy = compat.build_c2_policy_in_memory(
        realization_validator=_validator(archival),
        prepare=_prepare(old_contract, roots),
        toolchain_reader=_toolchain,
    )
    inventory = _inventory(c2_contract)
    samples = [
        _sample(
            c2_contract,
            inventory,
            created_at="2026-07-30T00:00:03Z",
        ),
        _sample(
            c2_contract,
            inventory,
            created_at="2026-07-30T00:00:04Z",
        ),
    ]
    stability_roots = deepcopy(replayed_roots)
    stability_roots["policy"] = {
        **_file_evidence(
            compat.C2_POLICY_PATH,
            digest="5" * 64,
        ),
        "policy_fingerprint": policy["policy_fingerprint"],
    }
    stability = environment.evaluate_environment_stability(
        c2_contract,
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
        c2_contract,
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
    old_returned, c2, roots = compat.replay_old_scope_and_handoff(
        prepare=_prepare(old, _roots()),
    )
    assert old_returned.target_unit_id == compat.OLD_TARGET_UNIT
    assert old_returned.require_target_ready is False
    assert c2.target_unit_id == compat.C2_TARGET_UNIT
    assert c2.require_target_ready is True
    for field, old_value in asdict(old_returned).items():
        if field not in {"target_unit_id", "require_target_ready"}:
            assert asdict(c2)[field] == old_value
    assert roots == _roots()


@pytest.mark.parametrize(
    "drift",
    ("target", "require", "other_scope"),
)
def test_old_scope_replay_rejects_scope_drift(drift: str) -> None:
    old = _contract()
    if drift == "target":
        old = replace(old, target_unit_id=compat.C2_TARGET_UNIT)
    elif drift == "require":
        old = replace(old, require_target_ready=True)
    else:
        old = replace(old, allowed_manager_states=("running",))
    with pytest.raises(PermissionError):
        compat.replay_old_scope_and_handoff(
            prepare=_prepare(old, _roots()),
        )


def test_policy_is_fresh_c2_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _old,
        c2,
        roots,
        archival,
        policy,
        _stability,
    ) = _policy_and_stability(monkeypatch)
    assert policy["unit_scope"] == {
        "target_unit_id": compat.C2_TARGET_UNIT,
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
    compat.validate_c2_realization_archival(archival, contract=c2)


def test_realization_bridge_source_is_production_frozen() -> None:
    digest = hashlib.sha256(compat.C2_BRIDGE_PATH.read_bytes()).hexdigest()
    assert compat.C2_BRIDGE_SHA256 == digest
    assert compat.C2_BRIDGE_SHA256 != "__TO_BE_FROZEN__"
    module = compat._load_verified_c2_bridge()
    assert callable(module.validate_archival_realization_chain)


def test_stability_uses_patched_prepare_and_exact_2x30(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        old,
        c2,
        roots,
        archival,
        policy,
        expected_stability,
    ) = _policy_and_stability(monkeypatch)
    observed: list[tuple[object, dict[str, object]]] = []

    def gate(*args, **kwargs):
        prepared = environment.prepare_environment_stability_contract(
            args[0],
            args[1],
            selected_gpu_index=kwargs["selected_gpu_index"],
            target_unit_id=kwargs["target_unit_id"],
            conflict_unit_ids=kwargs["conflict_unit_ids"],
            dependency_unit_ids=kwargs["dependency_unit_ids"],
            allowed_failed_unit_ids=kwargs["allowed_failed_unit_ids"],
            allowed_unit_ids=kwargs["allowed_unit_ids"],
            allowed_manager_states=kwargs["allowed_manager_states"],
            require_target_ready=kwargs["require_target_ready"],
            strict_all_gpu_consumers=kwargs[
                "strict_all_gpu_consumers"
            ],
        )
        observed.append(prepared)
        assert kwargs["sample_count"] == 2
        assert kwargs["sample_interval_seconds"] == 30.0
        return deepcopy(expected_stability)

    prior = environment.prepare_environment_stability_contract
    result = compat.run_c2_stability_in_memory(
        policy,
        realization_validator=_validator(archival),
        prepare=_prepare(old, roots),
        gate=gate,
    )
    assert result == expected_stability
    assert observed[0][0] == c2
    assert observed[0][1] == roots
    assert environment.prepare_environment_stability_contract is prior


def test_postcleanup_reuses_exact_final_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        old,
        c2,
        roots,
        archival,
        policy,
        stability,
    ) = _policy_and_stability(monkeypatch)
    monkeypatch.setattr(
        environment,
        "utc_now",
        lambda: "2026-07-30T00:00:06Z",
    )
    post = compat.build_c2_postcleanup_in_memory(
        policy,
        stability,
        realization_validator=_validator(archival),
        prepare=_prepare(old, roots),
    )
    assert post["inventory"] == stability["samples"][-1]["inventory"]
    closed = compat.validate_c2_environment_closure(
        policy,
        stability,
        post,
        archival=archival,
        c2_contract=c2,
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
        c2,
        _roots_value,
        archival,
        policy,
        stability,
    ) = _policy_and_stability(monkeypatch)
    supplied_contract = c2
    if drift == "target":
        supplied_contract = replace(
            c2,
            target_unit_id=compat.OLD_TARGET_UNIT,
        )
    elif drift == "require":
        supplied_contract = replace(c2, require_target_ready=False)
    elif drift == "other_scope":
        supplied_contract = replace(
            c2,
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
            "2026-07-30T00:00:07Z"
        )
    with pytest.raises((PermissionError, ValueError)):
        compat.validate_c2_environment_closure(
            policy,
            stability,
            None,
            archival=archival,
            c2_contract=supplied_contract,
        )
