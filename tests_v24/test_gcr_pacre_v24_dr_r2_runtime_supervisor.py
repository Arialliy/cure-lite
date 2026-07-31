from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from tools import cure_lite_v24_runtime_environment as runtime_environment
from tools import cure_lite_v24_runtime_supervisor as supervisor


REPOSITORY = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = (
    REPOSITORY / "tools/cure_lite_v24_runtime_supervisor.py"
)
DUMMY_INVOCATION_ID = "a" * 32
STRICT_ENVIRONMENT_VALIDATOR = (
    supervisor._validated_bound_environment_contract
)
LIVE_ENVIRONMENT_VERIFIER = supervisor._verify_live_environment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        path.chmod(0o600)
    path.write_text(
        supervisor.canonical_json(payload) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)


def _reseal_runtime_spec(
    path: Path,
    payload: dict[str, object],
) -> None:
    systemd = payload["runtime"]["systemd"]
    systemd["immutable_shadow_fingerprint"] = supervisor.stable_fingerprint(
        systemd["immutable_shadow_properties"]
    )
    body = dict(payload)
    body.pop("runtime_spec_fingerprint", None)
    payload["runtime_spec_fingerprint"] = supervisor.stable_fingerprint(body)
    _write_canonical(path, payload)



def _environment_contract(tmp_path: Path) -> dict[str, object]:
    evidence_root = tmp_path / "environment-evidence"
    evidence_root.mkdir()
    selected_uuid = "GPU-12cdabd0-7910-8f4a-e4d7-e3c7867d1296"
    selected_pci = "00000000:02:00.0"
    definitions = {
        "policy": "policy_fingerprint",
        "cleanup_plan": "plan_fingerprint",
        "cleanup_authorization": "authorization_fingerprint",
        "cleanup_receipt": "cleanup_receipt_fingerprint",
        "stability_receipt": "stability_receipt_fingerprint",
        "integration_authorization": "authorization_fingerprint",
        "integration_receipt": "receipt_fingerprint",
        "unit_realization_authorization": "authorization_fingerprint",
        "unit_realization_receipt": "receipt_fingerprint",
    }
    contract: dict[str, object] = {}
    for prefix, fingerprint_field in definitions.items():
        body: dict[str, object] = {
            "schema_version": f"generated-{prefix}-v1",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        payload = {
            **body,
            fingerprint_field: supervisor.stable_fingerprint(body),
        }
        path = (evidence_root / f"{prefix}.json").resolve()
        _write_canonical(path, payload)
        path.chmod(0o444)
        contract[f"{prefix}_path"] = str(path)
        contract[f"{prefix}_file_sha256"] = _sha256(path)
    nested_inventory_body: dict[str, object] = {
        "schema_version": "cure-lite-v24-runtime-environment-inventory-v1",
        "uid": os.getuid(),
        "boot_id": "00000000-0000-0000-0000-000000000000",
        "manager": {
            "endpoint": {
                "uid": os.getuid(),
                "runtime_directory": f"/run/user/{os.getuid()}",
                "runtime_directory_device": 1,
                "runtime_directory_inode": 2,
                "bus_path": f"/run/user/{os.getuid()}/bus",
                "bus_device": 3,
                "bus_inode": 4,
            },
            "allowed_failed_unit_ids": [],
            "allowed_states": ["running", "degraded"],
        },
        "unit_scope": {
            "conflict_unit_ids": [],
            "dependency_unit_ids": [],
        },
        "gpu_snapshot": {
            "selected_gpu_uuid": selected_uuid,
            "devices": [{
                "index": 0,
                "uuid": selected_uuid,
                "pci_bus_id": selected_pci,
                "minor_number": 0,
            }],
            "allowed_unit_ids": [],
            "strict_all_gpu_consumers": False,
            "first_apps": [],
            "second_apps": [],
        },
        "passed": True,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    nested_inventory = {
        **nested_inventory_body,
        "inventory_fingerprint": supervisor.stable_fingerprint(
            nested_inventory_body
        ),
    }
    inventory_receipt_body: dict[str, object] = {
        "schema_version": (
            "cure-lite-v24-runtime-environment-audit-receipt-v1"
        ),
        "command": "audit-only",
        "inventory": nested_inventory,
        "passed": True,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    inventory_receipt = {
        **inventory_receipt_body,
        "receipt_fingerprint": supervisor.stable_fingerprint(
            inventory_receipt_body
        ),
    }
    inventory_path = (evidence_root / "inventory.json").resolve()
    _write_canonical(inventory_path, inventory_receipt)
    inventory_path.chmod(0o444)
    contract["inventory_path"] = str(inventory_path)
    contract["inventory_file_sha256"] = _sha256(inventory_path)
    lease_root = tmp_path / "gpu-leases"
    lease_root.mkdir(mode=0o700)
    contract.update(
        {
            "selected_gpu_uuid": selected_uuid,
            "selected_gpu_pci_bus_id": selected_pci,
            "selected_gpu_minor_number": 0,
            "gpu_lease_path": str((lease_root / "active.lease").resolve()),
            "gpu_lease_tombstone_path": str(
                (lease_root / "released.lease").resolve()
            ),
        }
    )
    return contract


def _scientific_preaccess_contract(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "scientific-preaccess"
    root.mkdir()
    contract: dict[str, object] = {
        "source_closure_fingerprint_103": (
            supervisor._ACTUAL_SOURCE_CLOSURE_FINGERPRINT_103
        ),
    }
    definitions = (
        (
            "authorization",
            "authorization_fingerprint",
            "cure-lite-v24-D_R-structural-r2-authorization-v1",
        ),
        (
            "access_audit",
            "access_audit_fingerprint",
            "cure-lite-v24-split-access-audit-r2-v1",
        ),
    )
    for prefix, fingerprint_field, schema in definitions:
        body: dict[str, object] = {
            "schema_version": schema,
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        payload = {
            **body,
            fingerprint_field: supervisor.stable_fingerprint(body),
        }
        path = (root / f"{prefix}.json").resolve()
        _write_canonical(path, payload)
        path.chmod(0o444)
        contract[f"{prefix}_path"] = str(path)
        contract[f"{prefix}_file_sha256"] = _sha256(path)
        contract[f"{prefix}_fingerprint"] = payload[fingerprint_field]
        contract[f"{prefix}_required_schema"] = schema
    return contract

@pytest.fixture(autouse=True)
def _fixed_manager_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        supervisor,
        "_fixed_verified_manager_environment",
        lambda: {
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1234/bus",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "XDG_RUNTIME_DIR": "/run/user/1234",
        },
    )
    monkeypatch.setattr(
        supervisor,
        "_ACTUAL_SPEC_PATH",
        str((tmp_path / "runtime-spec.json").resolve()),
    )
    monkeypatch.setattr(
        supervisor,
        "_ACTUAL_RUNTIME_LAUNCH_AUTHORIZATION_PATH",
        str((tmp_path / "absent-r2-authorization.json").resolve()),
    )
    monkeypatch.setattr(
        supervisor,
        "_ACTUAL_SCIENTIFIC_AUTHORIZATION_PATH",
        str(
            (
                tmp_path
                / "scientific-preaccess/authorization.json"
            ).resolve()
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_ACTUAL_SCIENTIFIC_ACCESS_AUDIT_PATH",
        str(
            (
                tmp_path
                / "scientific-preaccess/access_audit.json"
            ).resolve()
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_ACTUAL_ADAPTER_PATH",
        str(SUPERVISOR_PATH.resolve()),
    )
    monkeypatch.setattr(
        supervisor,
        "_ACTUAL_LEGACY_ENTRYPOINT_PATH",
        str(SUPERVISOR_PATH.resolve()),
    )
    # The long-standing supervisor unit fixtures use deliberately generic
    # environment receipts.  Production actual_D_R now has a separate strict
    # validator, exercised directly by the dedicated tests below.
    monkeypatch.setattr(
        supervisor,
        "_validated_bound_environment_contract",
        lambda _spec, payloads=None: None,
    )
    monkeypatch.setattr(
        supervisor,
        "_verify_live_environment",
        lambda _spec, *, phase: _fake_live_audit(phase),
    )

def _dummy_spec(
    tmp_path: Path,
    argv: list[str],
    *,
    actual: bool = False,
    integration: bool = False,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    heartbeat = root / "heartbeat"
    heartbeat.mkdir(mode=0o700)
    systemd_invocations = root / "systemd-invocations"
    systemd_invocations.mkdir(mode=0o700)
    assert not (actual and integration)
    integration_identity = (
        supervisor.build_systemd_integration_identity(
            "generated-case-0123456789abcdef"
        )
        if integration
        else None
    )
    execution_kind = supervisor.DUMMY_EXECUTION_KIND
    if actual:
        execution_kind = supervisor.ACTUAL_EXECUTION_KIND
    elif integration:
        execution_kind = supervisor.SYSTEMD_INTEGRATION_DUMMY_KIND
        dummy_child = tmp_path / "cure_lite_v24_dummy_child.py"
        dummy_child.write_text("raise SystemExit(0)\n", encoding="utf-8")
        argv = [
            str(Path(sys.executable).resolve()),
            "-I",
            str(dummy_child.resolve()),
        ]
    supervisor_python = (
        supervisor._ACTUAL_PYTHON_PATH
        if actual
        else str(Path(sys.executable).resolve())
    )
    supervisor_python_flags = "-I -S -B -u" if actual else "-I -u"
    actual_python = (
        Path(supervisor._ACTUAL_PYTHON_PATH) if actual else None
    )
    actual_python_metadata = (
        actual_python.stat() if actual_python is not None else None
    )
    dependency_site = (
        Path(supervisor._ACTUAL_RUNTIME_DEPENDENCY_SITE_PATH)
        if actual
        else None
    )
    dependency_site_metadata = (
        dependency_site.stat() if dependency_site is not None else None
    )
    authorization_path = (
        tmp_path / "absent-r2-authorization.json"
    ).resolve()
    environment_contract = _environment_contract(tmp_path) if actual else None
    if actual:
        argv = [
            supervisor._ACTUAL_PYTHON_PATH,
            "-I",
            "-S",
            "-B",
            "-u",
            supervisor._ACTUAL_ADAPTER_PATH,
            "real",
            "--execute-real-dr",
            "--device",
            "cuda:0",
            "--runtime-launch-authorization",
            str(authorization_path),
        ]
    body: dict[str, object] = {
        "schema_version": supervisor.RUNTIME_SPEC_SCHEMA,
        "execution_kind": execution_kind,
        "candidate": (
            "GCR-PACRE-v24"
            if actual
            else "systemd-integration-dummy"
            if integration
            else "generated-dummy"
        ),
        "stage_id": (
            "gcr_pacre_v24_D_R_structural_r2"
            if actual
            else integration_identity["stage_id"]
            if integration_identity is not None
            else "generated_dummy_runtime"
        ),
        "attempt_id": (
            "gcr_pacre_v24_D_R_zero_update_structural_r2"
            if actual
            else integration_identity["attempt_id"]
            if integration_identity is not None
            else "generated_dummy_attempt"
        ),
        "attempt_ordinal": 2 if actual else 0,
        "prior_attempt_count": 1 if actual else 0,
        "authorization": (
            {
                "path": str(authorization_path),
                "required_schema": (
                    "cure-lite-v24-D_R-structural-r2-runtime-launch-authorization-v1"
                ),
            }
            if actual
            else None
        ),
        "scientific_preaccess": (
            _scientific_preaccess_contract(tmp_path) if actual else None
        ),
        "child": {
            "argv": argv,
            "argv_fingerprint": supervisor.stable_fingerprint(argv),
            "cwd": str(tmp_path.resolve()),
            "environment": (
                {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": environment_contract[
                        "selected_gpu_uuid"
                    ],
                }
                if actual
                else {}
            ),
            "inherit_environment": [],
            "entrypoint_path": (
                str(SUPERVISOR_PATH)
                if actual
                else str(dummy_child.resolve())
                if integration
                else None
            ),
        },
        "artifacts": {
            "root": str(root.resolve()),
            "attempt_commit": str(
                (root / "attempt-commit.json").resolve()
            ),
            "materialization_claim": str(
                (root / "materialization-claim.json").resolve()
            ),
            "stdout_log": str((root / "stdout.log").resolve()),
            "stderr_log": str((root / "stderr.log").resolve()),
            "heartbeat_dir": str(heartbeat.resolve()),
            "runtime_terminal": str((root / "terminal.json").resolve()),
            "systemd_invocation_dir": str(
                systemd_invocations.resolve()
            ),
            "launch_lease": str((root / "launch-lease.json").resolve()),
            "precommit_phase_receipt": str(
                (root / "precommit-phase.json").resolve()
            ),
            "start_ack_receipt": str(
                (root / "start-ack.json").resolve()
            ),
            "child_prespawn_phase_receipt": str(
                (root / "child-prespawn.json").resolve()
            ),
            "runtime_attestation": str(
                (root / "runtime-attestation.json").resolve()
            ),
            "gpu_lease_release_receipt": str(
                (root / "gpu-lease-release.json").resolve()
            ),
            "consumed_start_failure_receipt": str(
                (root / "consumed-start-failure.json").resolve()
            ),
        },
        "runtime": {
            "shell": False,
            "start_new_session": True,
            "launch_limit": 1,
            "automatic_retry_allowed": False,
            "resume_allowed": False,
            "restart": "no",
            "heartbeat_interval_seconds": 0.02,
            "poll_interval_seconds": 0.005,
            "termination_grace_seconds": 0.2,
            "systemd": {
                "unit_name": (
                    supervisor._ACTUAL_UNIT_NAME
                    if actual
                    else integration_identity["unit_name"]
                    if integration_identity is not None
                    else "cure-lite-v24-generated-dummy.service"
                ),
                "service_type": "exec",
                "kill_mode": "mixed",
                "send_sigkill": True,
                "timeout_stop_seconds": 1.0,
                "start_ack_timeout_seconds": 0.2,
                "start_ack_poll_seconds": 0.005,
                "unit_fragment_file_sha256": (
                    _sha256(SUPERVISOR_PATH)
                    if actual or integration
                    else None
                ),
                "immutable_shadow_properties": {
                    "Type": "exec",
                    "Restart": "no",
                    "KillMode": "mixed",
                    "SendSIGKILL": "yes",
                    "TimeoutStopUSec": "1s",
                    "FragmentPath": str(SUPERVISOR_PATH),
                    "DropInPaths": "",
                    "Transient": "no",
                    "Environment": "PYTHONUNBUFFERED=1",
                    "UnsetEnvironment": "",
                    "WorkingDirectory": str(tmp_path.resolve()),
                    "UMask": "0077",
                    "ExitType": "main",
                    "RuntimeMaxUSec": "infinity",
                    "WatchdogUSec": "disabled",
                    "OOMPolicy": "kill",
                    "RemainAfterExit": "no",
                    "StandardInput": "null",
                    "StandardOutput": "journal",
                    "StandardError": "journal",
                    "StartLimitIntervalUSec": "infinity",
                    "StartLimitBurst": "1",
                    "KillSignal": "15",
                    "ExecCondition": (
                        "{ path="
                        f"{supervisor_python}"
                        " ; argv[]="
                        f"{supervisor_python} {supervisor_python_flags} "
                        f"{SUPERVISOR_PATH} claim-materialization --spec "
                        f"{tmp_path / 'runtime-spec.json'}"
                        " ; ignore_errors=no }"
                    ),
                    "ExecStartPre": (
                        "{ path="
                        f"{supervisor_python}"
                        " ; argv[]="
                        f"{supervisor_python} {supervisor_python_flags} "
                        f"{SUPERVISOR_PATH} verify-runtime-spec --spec "
                        f"{tmp_path / 'runtime-spec.json'}"
                        " ; ignore_errors=no }"
                    ),
                    "ExecStart": (
                        "{ path="
                        f"{supervisor_python}"
                        " ; argv[]="
                        f"{supervisor_python} {supervisor_python_flags} "
                        f"{SUPERVISOR_PATH} run-once --spec "
                        f"{tmp_path / 'runtime-spec.json'}"
                        " ; ignore_errors=no }"
                    ),
                    "ExecStopPost": (
                        "{ path="
                        f"{supervisor_python}"
                        " ; argv[]="
                        f"{supervisor_python} {supervisor_python_flags} "
                        f"{SUPERVISOR_PATH} record-systemd-exit --spec "
                        f"{tmp_path / 'runtime-spec.json'}"
                        " ; ignore_errors=no }"
                    ),
                },
                "immutable_shadow_fingerprint": "TBD",
            },
        },
        "environment": environment_contract,
        "source_bindings": {
            "supervisor_file_sha256": _sha256(SUPERVISOR_PATH),
            "child_entry_file_sha256": (
                _sha256(SUPERVISOR_PATH)
                if actual
                else _sha256(dummy_child)
                if integration
                else None
            ),
            "runtime_environment_file_sha256": _sha256(
                REPOSITORY / "tools/cure_lite_v24_runtime_environment.py"
            ),
            "r2_adapter_path": str(SUPERVISOR_PATH) if actual else None,
            "r2_adapter_file_sha256": (
                _sha256(SUPERVISOR_PATH) if actual else None
            ),
            "legacy_gate_entrypoint_path": (
                str(SUPERVISOR_PATH) if actual else None
            ),
            "legacy_gate_entrypoint_file_sha256": (
                _sha256(SUPERVISOR_PATH) if actual else None
            ),
            "prior_attempt_receipt_file_sha256": (
                "1" * 64 if actual else None
            ),
            "python_path": (
                str(actual_python) if actual_python is not None else None
            ),
            "python_file_sha256": (
                _sha256(actual_python)
                if actual_python is not None
                else None
            ),
            "python_device": (
                actual_python_metadata.st_dev
                if actual_python_metadata is not None
                else None
            ),
            "python_inode": (
                actual_python_metadata.st_ino
                if actual_python_metadata is not None
                else None
            ),
            "python_owner_uid": (
                actual_python_metadata.st_uid
                if actual_python_metadata is not None
                else None
            ),
            "python_mode": (
                stat.S_IMODE(actual_python_metadata.st_mode)
                if actual_python_metadata is not None
                else None
            ),
            "runtime_dependency_site_path": (
                str(dependency_site)
                if dependency_site is not None
                else None
            ),
            "runtime_dependency_site_device": (
                dependency_site_metadata.st_dev
                if dependency_site_metadata is not None
                else None
            ),
            "runtime_dependency_site_inode": (
                dependency_site_metadata.st_ino
                if dependency_site_metadata is not None
                else None
            ),
            "runtime_dependency_site_owner_uid": (
                dependency_site_metadata.st_uid
                if dependency_site_metadata is not None
                else None
            ),
            "runtime_dependency_site_mode": (
                stat.S_IMODE(dependency_site_metadata.st_mode)
                if dependency_site_metadata is not None
                else None
            ),
        },
    }
    systemd = body["runtime"]["systemd"]
    systemd["immutable_shadow_fingerprint"] = supervisor.stable_fingerprint(
        systemd["immutable_shadow_properties"]
    )
    spec = {
        **body,
        "runtime_spec_fingerprint": supervisor.stable_fingerprint(body),
    }
    path = tmp_path / "runtime-spec.json"
    _write_canonical(path, spec)
    return path, spec


def _claim_and_run(spec_path: Path) -> int:
    assert supervisor.claim_materialization(
        spec_path,
        environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
    ) == 0
    assert supervisor.verify_runtime_spec(
        spec_path,
        environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
    ) == 0
    return supervisor.run_once(
        spec_path,
        environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
    )


def _read_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _phase_state(
    *,
    invocation_id: str = "",
    active_state: str = "inactive",
    sub_state: str = "dead",
) -> dict[str, str]:
    return {
        "LoadState": "loaded",
        "ActiveState": active_state,
        "SubState": sub_state,
        "UnitFileState": "static",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
        "Result": "success",
        "ExecMainCode": "0",
        "ExecMainStatus": "0",
        "InvocationID": invocation_id,
    }


def test_canonical_json_mode_and_umask_are_exact(
    tmp_path: Path,
) -> None:
    payload = {"schema_version": "generated-mode-test", "value": 1}
    sealed = tmp_path / "sealed.json"
    previous_umask = os.umask(0o077)
    try:
        supervisor._write_new_json(sealed, payload)
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(sealed.stat().st_mode) == 0o444
    assert supervisor._read_canonical_json(
        sealed,
        name="generated sealed evidence",
    ) == payload
    sealed.chmod(0o666)
    with pytest.raises(PermissionError, match="descriptor read"):
        supervisor._read_canonical_json(
            sealed,
            name="generated sealed evidence",
        )


def test_write_new_json_rejects_parent_directory_generation_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    displaced = tmp_path / "displaced-control"
    path = control / "sealed.json"
    original_reader = supervisor._read_fd_bytes
    replaced = False

    def replace_parent_during_readback(descriptor: int) -> bytes:
        nonlocal replaced
        data = original_reader(descriptor)
        if not replaced:
            replaced = True
            control.rename(displaced)
            control.mkdir(mode=0o700)
            replacement = control / path.name
            replacement.write_bytes(data)
            replacement.chmod(0o444)
        return data

    monkeypatch.setattr(
        supervisor,
        "_read_fd_bytes",
        replace_parent_during_readback,
    )
    with pytest.raises(RuntimeError, match="descriptor readback"):
        supervisor._write_new_json(
            path,
            {"schema_version": "test-parent-v1", "value": "exact"},
        )
    assert replaced is True
    assert (displaced / path.name).exists()
    assert path.exists()


def test_gpu_lease_evidence_rejects_same_content_inode_replacement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "active.lease"
    body = {
        "schema_version": "test-gpu-lease-v1",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
    }
    payload = {
        **body,
        "lease_fingerprint": supervisor.stable_fingerprint(body),
    }
    encoded = (supervisor.canonical_json(payload) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    path.chmod(0o600)
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    parent = os.fstat(parent_descriptor)
    descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    original = os.fstat(descriptor)
    replacement = tmp_path / "replacement.lease"
    replacement.write_bytes(encoded)
    replacement.chmod(0o600)
    os.replace(replacement, path)
    handle = SimpleNamespace(
        descriptor=descriptor,
        path=path,
        payload=payload,
        device=original.st_dev,
        inode=original.st_ino,
        parent_descriptor=parent_descriptor,
        parent_device=parent.st_dev,
        parent_inode=parent.st_ino,
    )
    try:
        with pytest.raises(PermissionError, match="descriptor identity"):
            supervisor._external_gpu_lease_evidence(handle)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def test_gpu_lease_evidence_rejects_parent_directory_replacement(
    tmp_path: Path,
) -> None:
    lease_root = tmp_path / "gpu-leases"
    lease_root.mkdir(mode=0o700)
    path = lease_root / "active.lease"
    body = {
        "schema_version": "test-gpu-lease-v1",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
    }
    payload = {
        **body,
        "lease_fingerprint": supervisor.stable_fingerprint(body),
    }
    encoded = (supervisor.canonical_json(payload) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    path.chmod(0o600)
    parent_descriptor = os.open(
        lease_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    parent = os.fstat(parent_descriptor)
    lease = os.fstat(descriptor)
    displaced = tmp_path / "displaced-gpu-leases"
    lease_root.rename(displaced)
    lease_root.mkdir(mode=0o700)
    path.write_bytes(encoded)
    path.chmod(0o600)
    handle = SimpleNamespace(
        descriptor=descriptor,
        parent_descriptor=parent_descriptor,
        path=path,
        payload=payload,
        device=lease.st_dev,
        inode=lease.st_ino,
        parent_device=parent.st_dev,
        parent_inode=parent.st_ino,
    )
    try:
        with pytest.raises(PermissionError, match="descriptor identity"):
            supervisor._external_gpu_lease_evidence(handle)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)



def _fake_live_audit(phase: str) -> dict[str, object]:
    inventory_fingerprint = supervisor.stable_fingerprint(
        {"generated_inventory_phase": phase}
    )
    return supervisor._seal(
        {
            "schema_version": "cure-lite-v24-runtime-live-audit-v1",
            "phase": phase,
            "runtime_spec_fingerprint": "generated-by-fixture",
            "inventory": {"passed": True},
            "inventory_fingerprint": inventory_fingerprint,
            "runtime_environment_audit_valid": True,
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        },
        fingerprint_field="environment_audit_fingerprint",
    )

def _authorize_actual_spec(
    spec_path: Path,
    spec: dict[str, object],
) -> dict[str, object]:
    reference = spec["authorization"]
    assert isinstance(reference, dict)
    issued = datetime.now(timezone.utc)
    body = {
        "schema_version": reference["required_schema"],
        "authorization_kind": "runtime_launch",
        "instruction_id": supervisor._ACTUAL_INSTRUCTION_ID,
        "authorization_basis": supervisor._ACTUAL_AUTHORIZATION_BASIS,
        "authorized_uid": os.getuid(),
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (
            issued + timedelta(seconds=300)
        ).isoformat().replace("+00:00", "Z"),
        **supervisor._verify_scientific_preaccess_bindings(spec),
        **supervisor._environment_evidence_bindings(spec),
        "candidate": spec["candidate"],
        "stage_id": spec["stage_id"],
        "attempt_id": spec["attempt_id"],
        "attempt_ordinal": 2,
        "prior_attempt_count": 1,
        "fresh_attempt_authorized": True,
        "D_R_payload_authorized": True,
        "D_V_payload_authorized": False,
        "D_T_payload_authorized": False,
        "training_authorized": False,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
        "runtime_spec_file_sha256": _sha256(spec_path),
        "runtime_spec_v2_fingerprint": spec["runtime_spec_fingerprint"],
        "runtime_spec_v2_file_sha256": _sha256(spec_path),
        "supervisor_v2_source_closure_fingerprint": (
            supervisor.stable_fingerprint(spec["source_bindings"])
        ),
        "unit_fragment_sha256": spec["runtime"]["systemd"][
            "unit_fragment_file_sha256"
        ],
        "preauthorization_D_R_payload_accessed": False,
        "preauthorization_D_V_payload_accessed": False,
        "preauthorization_D_T_payload_accessed": False,
    }
    authorization = {
        **body,
        "authorization_fingerprint": supervisor.stable_fingerprint(body),
    }
    authorization_path = Path(str(reference["path"]))
    _write_canonical(authorization_path, authorization)
    authorization_path.chmod(0o444)
    return supervisor._verify_actual_authorization(
        spec,
        spec_path=spec_path,
    )


def _commit_actual_spec(
    spec_path: Path,
    spec: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    authorization = _authorize_actual_spec(spec_path, spec)
    runtime = spec["runtime"]
    assert isinstance(runtime, dict)
    systemd = runtime["systemd"]
    assert isinstance(systemd, dict)
    shadow = systemd["immutable_shadow_properties"]
    assert isinstance(shadow, dict)
    artifacts = spec["artifacts"]
    assert isinstance(artifacts, dict)
    planned = supervisor._planned_attempt_commit_fingerprint(
        spec,
        authorization,
    )
    gpu_handle = supervisor._acquire_external_gpu_lease(
        spec,
        authorization,
        planned_attempt_commit_fingerprint=planned,
    )
    gpu_lease = supervisor._external_gpu_lease_evidence(gpu_handle)
    lease = supervisor._create_launch_lease(spec, authorization)
    audit = _fake_live_audit("postlease")
    precommit = supervisor._write_phase_receipt(
        spec,
        phase="precommit",
        phase_state=_phase_state(),
        immutable_shadow=shadow,
        path=Path(str(artifacts["precommit_phase_receipt"])),
        launch_lease=lease,
        environment_audit=audit,
        gpu_lease=gpu_lease,
    )
    commit = supervisor._attempt_commit_payload(
        spec,
        authorization,
        shadow,
        precommit,
        lease,
        planned_attempt_commit_fingerprint=planned,
        environment_audit=audit,
        gpu_lease=gpu_lease,
    )
    supervisor._write_new_json(
        Path(str(artifacts["attempt_commit"])),
        commit,
    )
    gpu_handle.close_without_release()
    return authorization, commit


def _commit_integration_spec(
    spec: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    runtime = spec["runtime"]
    artifacts = spec["artifacts"]
    assert isinstance(runtime, dict)
    assert isinstance(artifacts, dict)
    systemd = runtime["systemd"]
    assert isinstance(systemd, dict)
    shadow = systemd["immutable_shadow_properties"]
    assert isinstance(shadow, dict)
    lease = supervisor._create_launch_lease(spec, None)
    precommit = supervisor._write_phase_receipt(
        spec,
        phase="precommit",
        phase_state=_phase_state(),
        immutable_shadow=shadow,
        path=Path(str(artifacts["precommit_phase_receipt"])),
        launch_lease=lease,
    )
    commit = supervisor._attempt_commit_payload(
        spec,
        None,
        shadow,
        precommit,
        lease,
    )
    supervisor._write_new_json(
        Path(str(artifacts["attempt_commit"])),
        commit,
    )
    return lease, commit


def _rewrite_attempt_commit(
    spec: dict[str, object],
    payload: dict[str, object],
) -> None:
    body = dict(payload)
    body.pop("attempt_commit_fingerprint", None)
    payload["attempt_commit_fingerprint"] = supervisor.stable_fingerprint(
        body
    )
    artifacts = spec["artifacts"]
    assert isinstance(artifacts, dict)
    _write_canonical(Path(str(artifacts["attempt_commit"])), payload)


def _rewrite_active_gpu_lease(
    spec: dict[str, object],
    attempt_commit: dict[str, object],
    mutation: object,
) -> dict[str, object]:
    environment = spec["environment"]
    assert isinstance(environment, dict)
    path = Path(str(environment["gpu_lease_path"]))
    payload = _read_json(path)
    mutation(payload)
    body = dict(payload)
    body.pop("lease_fingerprint", None)
    payload["lease_fingerprint"] = supervisor.stable_fingerprint(body)
    encoded = (supervisor.canonical_json(payload) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    path.chmod(0o600)
    changed = dict(attempt_commit)
    changed["gpu_lease_fingerprint"] = payload["lease_fingerprint"]
    changed["gpu_lease_file_sha256"] = hashlib.sha256(encoded).hexdigest()
    return changed


def test_actual_gpu_lease_commit_identity_is_exact_and_reopenable(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    authorization, commit = _commit_actual_spec(spec_path, spec)
    verified_commit = supervisor._verify_attempt_commit(spec, authorization)
    for field in (
        "gpu_lease_fingerprint",
        "gpu_lease_file_sha256",
        "gpu_lease_device",
        "gpu_lease_inode",
        "gpu_lease_parent_device",
        "gpu_lease_parent_inode",
    ):
        assert verified_commit[field] == commit[field]
    lease = supervisor._verify_active_gpu_lease(spec, verified_commit)
    try:
        for field in (
            "gpu_lease_fingerprint",
            "gpu_lease_file_sha256",
            "gpu_lease_device",
            "gpu_lease_inode",
            "gpu_lease_parent_device",
            "gpu_lease_parent_inode",
        ):
            assert lease[field] == commit[field]
    finally:
        lease["handle"].close_without_release()


def test_runtime_attestation_consumes_gpu_lease_parent_identity(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    authorization, _commit = _commit_actual_spec(spec_path, spec)
    attempt_commit = supervisor._verify_attempt_commit(spec, authorization)
    gpu_lease = supervisor._verify_active_gpu_lease(spec, attempt_commit)
    artifacts = spec["artifacts"]
    runtime = spec["runtime"]
    assert isinstance(artifacts, dict)
    assert isinstance(runtime, dict)
    systemd = runtime["systemd"]
    assert isinstance(systemd, dict)
    launch_lease = supervisor._verify_launch_lease(spec)
    supervisor._write_phase_receipt(
        spec,
        phase="child_prespawn",
        phase_state=_phase_state(
            invocation_id=DUMMY_INVOCATION_ID,
            active_state="active",
            sub_state="running",
        ),
        immutable_shadow=systemd["immutable_shadow_properties"],
        path=Path(str(artifacts["child_prespawn_phase_receipt"])),
        launch_lease=launch_lease,
        environment_audit=_fake_live_audit("child_prespawn"),
        gpu_lease=gpu_lease,
    )
    claim = supervisor._materialization_claim(
        spec,
        authorization,
        attempt_commit,
        DUMMY_INVOCATION_ID,
        "/generated.slice/r2.service",
    )
    claim_digest = supervisor._write_new_json(
        Path(str(artifacts["materialization_claim"])),
        claim,
    )
    materialization_claim = {
        **claim,
        "materialization_claim_file_sha256": claim_digest,
    }
    attestation = supervisor._write_runtime_attestation(
        spec,
        spec_path=spec_path,
        authorization=authorization,
        attempt_commit=attempt_commit,
        materialization_claim=materialization_claim,
        gpu_lease=gpu_lease,
    )
    for suffix in ("parent_device", "parent_inode"):
        assert attestation[f"active_gpu_lease_{suffix}"] == (
            attempt_commit[f"gpu_lease_{suffix}"]
        )
    gpu_lease["handle"].close_without_release()
    verified = supervisor.verify_child_runtime_attestation(
        artifacts["runtime_attestation"],
        authorization["path"],
        environment={
            "INVOCATION_ID": DUMMY_INVOCATION_ID,
            supervisor._RUNTIME_ATTESTATION_ENV: str(
                artifacts["runtime_attestation"]
            ),
        },
        argv=spec["child"]["argv"],
        cgroup_path="/generated.slice/r2.service",
    )
    assert verified["runtime_attestation_valid"] is True


def test_active_gpu_lease_rejects_nonclosed_payload_schema(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorization, commit = _commit_actual_spec(spec_path, spec)
    changed = _rewrite_active_gpu_lease(
        spec,
        commit,
        lambda payload: payload.__setitem__(
            "unexpected_identity",
            None,
        ),
    )
    with pytest.raises(PermissionError, match="active GPU lease identity"):
        supervisor._verify_active_gpu_lease(spec, changed)


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("boot_id", "11111111-1111-1111-1111-111111111111"),
        ("committer_pid", 2**30),
        ("committer_starttime", 2**62),
    ),
)
def test_active_gpu_lease_rejects_commit_process_identity_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorization, commit = _commit_actual_spec(spec_path, spec)
    changed = _rewrite_active_gpu_lease(
        spec,
        commit,
        lambda payload: payload.__setitem__(field, replacement),
    )
    with pytest.raises(PermissionError, match="active GPU lease identity"):
        supervisor._verify_active_gpu_lease(spec, changed)


def test_active_gpu_lease_recomputes_planned_commit_fingerprint(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorization, commit = _commit_actual_spec(spec_path, spec)
    forged = "0" * 64
    changed = _rewrite_active_gpu_lease(
        spec,
        commit,
        lambda payload: payload.__setitem__(
            "planned_attempt_commit_fingerprint",
            forged,
        ),
    )
    changed["planned_attempt_commit_fingerprint"] = forged
    with pytest.raises(
        PermissionError,
        match="active GPU lease commit identity",
    ):
        supervisor._verify_active_gpu_lease(spec, changed)


def test_active_gpu_lease_cleanup_closes_parent_after_lease_close_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorization, commit = _commit_actual_spec(spec_path, spec)
    changed = dict(commit)
    changed["gpu_lease_file_sha256"] = "0" * 64
    environment = spec["environment"]
    assert isinstance(environment, dict)
    lease_path = Path(str(environment["gpu_lease_path"]))
    real_open = os.open
    real_close = os.close
    descriptors: dict[str, int] = {}
    close_order: list[str] = []

    def tracking_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if (
            isinstance(path, (str, os.PathLike))
            and Path(path) == lease_path.parent
            and dir_fd is None
        ):
            descriptors["parent"] = descriptor
        elif (
            path == lease_path.name
            and dir_fd == descriptors.get("parent")
        ):
            descriptors["lease"] = descriptor
        return descriptor

    def faulting_close(descriptor: int) -> None:
        if descriptor == descriptors.get("lease"):
            close_order.append("lease")
            real_close(descriptor)
            raise OSError("generated lease close fault")
        if descriptor == descriptors.get("parent"):
            close_order.append("parent")
        real_close(descriptor)

    monkeypatch.setattr(supervisor.os, "open", tracking_open)
    monkeypatch.setattr(supervisor.os, "close", faulting_close)
    with pytest.raises(PermissionError, match="active GPU lease identity"):
        supervisor._verify_active_gpu_lease(spec, changed)
    assert close_order == ["lease", "parent"]
    with pytest.raises(OSError):
        os.fstat(descriptors["parent"])


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("gpu_lease_fingerprint", "0" * 64),
        ("gpu_lease_file_sha256", "0" * 64),
        ("gpu_lease_device", 2**62),
        ("gpu_lease_inode", 2**62),
        ("gpu_lease_parent_device", 2**62),
        ("gpu_lease_parent_inode", 2**62),
    ),
)
def test_active_gpu_lease_rejects_every_commit_identity_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorization, commit = _commit_actual_spec(spec_path, spec)
    changed = dict(commit)
    changed[field] = replacement
    with pytest.raises(PermissionError, match="active GPU lease identity"):
        supervisor._verify_active_gpu_lease(spec, changed)


def test_active_gpu_lease_rejects_same_bytes_new_inode(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorization, commit = _commit_actual_spec(spec_path, spec)
    environment = spec["environment"]
    assert isinstance(environment, dict)
    path = Path(str(environment["gpu_lease_path"]))
    replacement = path.with_name("replacement.lease")
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, path)
    with pytest.raises(PermissionError, match="active GPU lease identity"):
        supervisor._verify_active_gpu_lease(spec, commit)


def test_active_gpu_lease_rejects_parent_generation_replacement(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorization, commit = _commit_actual_spec(spec_path, spec)
    environment = spec["environment"]
    assert isinstance(environment, dict)
    path = Path(str(environment["gpu_lease_path"]))
    original_parent = path.parent
    displaced = original_parent.with_name("displaced-gpu-leases")
    original_parent.rename(displaced)
    original_parent.mkdir(mode=0o700)
    (displaced / path.name).rename(path)
    assert path.stat().st_ino == commit["gpu_lease_inode"]
    with pytest.raises(PermissionError, match="active GPU lease identity"):
        supervisor._verify_active_gpu_lease(spec, commit)


@pytest.mark.parametrize(
    "field",
    (
        "gpu_lease_device",
        "gpu_lease_inode",
        "gpu_lease_parent_device",
        "gpu_lease_parent_inode",
    ),
)
@pytest.mark.parametrize("invalid", (True, 0, -1))
def test_actual_attempt_commit_rejects_nonpositive_or_bool_identity(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorization, commit = _commit_actual_spec(spec_path, spec)
    changed = dict(commit)
    changed[field] = invalid
    _rewrite_attempt_commit(spec, changed)
    with pytest.raises(PermissionError, match="attempt commit is invalid"):
        supervisor._verify_attempt_commit_lineage(spec)


def test_integration_dummy_attempt_commit_closes_gpu_identity_with_none(
    tmp_path: Path,
) -> None:
    _spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        integration=True,
    )
    _lease, commit = _commit_integration_spec(spec)
    for field in (
        "gpu_lease_fingerprint",
        "gpu_lease_file_sha256",
        "gpu_lease_device",
        "gpu_lease_inode",
        "gpu_lease_parent_device",
        "gpu_lease_parent_inode",
    ):
        assert commit[field] is None
    verified = supervisor._verify_attempt_commit_lineage(spec)
    assert set(verified) == (
        supervisor._ATTEMPT_COMMIT_KEYS
        | {"attempt_commit_file_sha256"}
    )


def test_attempt_commit_rejects_nonclosed_schema(
    tmp_path: Path,
) -> None:
    _spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        integration=True,
    )
    _lease, commit = _commit_integration_spec(spec)
    changed = {**commit, "unexpected_gpu_identity": None}
    _rewrite_attempt_commit(spec, changed)
    with pytest.raises(PermissionError, match="attempt commit is invalid"):
        supervisor._verify_attempt_commit_lineage(spec)


def test_actual_python_and_dependency_site_bindings_are_exact(
    tmp_path: Path,
) -> None:
    spec_path, _spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    spec = supervisor.load_runtime_spec(spec_path)
    assert spec["child"]["argv"][:5] == [
        "/usr/bin/python3.12",
        "-I",
        "-S",
        "-B",
        "-u",
    ]
    bindings = spec["source_bindings"]
    assert bindings["python_path"] == supervisor._ACTUAL_PYTHON_PATH
    assert bindings["runtime_dependency_site_path"] == (
        supervisor._ACTUAL_RUNTIME_DEPENDENCY_SITE_PATH
    )
    assert bindings["runtime_dependency_site_device"] == (
        supervisor._ACTUAL_RUNTIME_DEPENDENCY_SITE_DEVICE
    )
    assert bindings["runtime_dependency_site_inode"] == (
        supervisor._ACTUAL_RUNTIME_DEPENDENCY_SITE_INODE
    )
    assert bindings["runtime_dependency_site_owner_uid"] == (
        supervisor._ACTUAL_RUNTIME_DEPENDENCY_SITE_OWNER_UID
    )
    assert bindings["runtime_dependency_site_mode"] == (
        supervisor._ACTUAL_RUNTIME_DEPENDENCY_SITE_MODE
    )
    supervisor._validate_runtime_filesystem(spec)


def test_actual_dependency_site_identity_is_revalidated_each_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, _spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    spec = supervisor.load_runtime_spec(spec_path)
    original = supervisor._stable_directory_identity

    def changed_dependency_site(
        path: str | Path,
        *,
        expected_uid: int,
        expected_mode: int,
        name: str,
    ) -> os.stat_result:
        identity = original(
            path,
            expected_uid=expected_uid,
            expected_mode=expected_mode,
            name=name,
        )
        if name != "actual runtime dependency site":
            return identity
        values = list(identity)
        values[1] = identity.st_ino + 1
        return os.stat_result(values)

    monkeypatch.setattr(
        supervisor,
        "_stable_directory_identity",
        changed_dependency_site,
    )
    with pytest.raises(
        PermissionError,
        match="actual runtime source lineage changed",
    ):
        supervisor._validate_runtime_filesystem(spec)


def test_module_is_stdlib_only_and_has_no_scientific_import() -> None:
    tree = ast.parse(SUPERVISOR_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= {
        "__future__",
            "argparse",
            "ctypes",
            "fcntl",
            "datetime",
            "hashlib",
            "importlib",
            "json",
            "os",
            "pathlib",
            "re",
            "signal",
            "stat",
            "subprocess",
        "sys",
        "time",
        "typing",
    }
    assert not (
        imported_roots
        & {
            "cure_lite",
            "cure_lite_v24",
            "torch",
            "numpy",
            "safetensors",
        }
    )
    popen_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Popen"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(popen_calls) == 1


def test_actual_without_r2_authorization_fails_before_artifact_or_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [
            sys.executable,
            "-I",
            str(SUPERVISOR_PATH),
            "--generated-never-run",
        ],
        actual=True,
    )
    called = False
    systemctl_called = False

    def forbidden_popen(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(supervisor.subprocess, "Popen", forbidden_popen)
    def forbidden_run(*_args: object, **_kwargs: object) -> object:
        nonlocal systemctl_called
        systemctl_called = True
        raise AssertionError("systemctl must not be reached")

    monkeypatch.setattr(supervisor.subprocess, "run", forbidden_run)
    for mode in (
        "commit-and-start",
        "claim-materialization",
        "verify-runtime-spec",
        "run-once",
    ):
        assert supervisor.main([mode, "--spec", str(spec_path)]) == os.EX_NOPERM
    assert called is False
    assert systemctl_called is False
    artifacts = spec["artifacts"]
    assert isinstance(artifacts, dict)
    for key in supervisor._ARTIFACT_KEYS - {
        "root",
        "heartbeat_dir",
        "systemd_invocation_dir",
    }:
        assert not Path(str(artifacts[key])).exists()
    assert list(Path(str(artifacts["heartbeat_dir"])).iterdir()) == []


def test_v2_separates_immutable_shadow_from_phase_state(
    tmp_path: Path,
) -> None:
    spec_path, _spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
        integration=True,
    )
    spec = supervisor.load_runtime_spec(spec_path)
    systemd = spec["runtime"]["systemd"]
    immutable = systemd["immutable_shadow_properties"]
    assert "ActiveState" not in immutable
    assert "NRestarts" not in immutable
    assert "NeedDaemonReload" not in immutable
    supervisor.validate_systemd_shadow(spec, dict(immutable))
    supervisor._validate_systemd_phase_state(
        _phase_state(),
        phase="precommit",
    )
    with pytest.raises(PermissionError):
        supervisor._validate_systemd_phase_state(
            _phase_state(active_state="active", sub_state="running"),
            phase="precommit",
        )
    with pytest.raises(
        PermissionError,
        match="has not passed ExecCondition",
    ):
        supervisor._validate_systemd_phase_state(
            _phase_state(
                invocation_id=DUMMY_INVOCATION_ID,
                active_state="activating",
                sub_state="condition",
            ),
            phase="start_ack",
        )
    supervisor._validate_systemd_phase_state(
        _phase_state(
            invocation_id=DUMMY_INVOCATION_ID,
            active_state="activating",
            sub_state="start-pre",
        ),
        phase="start_ack",
    )


@pytest.mark.parametrize(
    "directive",
    tuple(supervisor._SYSTEMD_EXEC_MODES),
)
@pytest.mark.parametrize(
    "mutation",
    ("wrong-mode-with-decoy", "extra-argv", "swapped-flags", "ignore-errors"),
)
def test_integration_dummy_exec_contract_is_exact_after_refingerprinting(
    tmp_path: Path,
    directive: str,
    mutation: str,
) -> None:
    spec_path, original = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
        integration=True,
    )
    mutated = deepcopy(original)
    shadow = mutated["runtime"]["systemd"]["immutable_shadow_properties"]
    identity = supervisor._normalized_systemd_exec_identity(
        shadow[directive]
    )
    argv = list(identity["argv"])
    ignore_errors = str(identity["ignore_errors"])
    if mutation == "wrong-mode-with-decoy":
        expected_mode = supervisor._SYSTEMD_EXEC_MODES[directive]
        argv[4] = (
            "verify-runtime-spec"
            if expected_mode != "verify-runtime-spec"
            else "run-once"
        )
        argv.append(expected_mode)
    elif mutation == "extra-argv":
        argv.append("--unauthorized-decoy")
    elif mutation == "swapped-flags":
        argv[1], argv[2] = argv[2], argv[1]
    elif mutation == "ignore-errors":
        ignore_errors = "yes"
    else:
        raise AssertionError("unreachable mutation")
    shadow[directive] = (
        f"{{ path={identity['path']} ; argv[]={' '.join(argv)}"
        f" ; ignore_errors={ignore_errors} }}"
    )
    _reseal_runtime_spec(spec_path, mutated)
    with pytest.raises(
        ValueError,
        match=rf"systemd integration dummy {directive} argv is not exact",
    ):
        supervisor.load_runtime_spec(spec_path)


def test_integration_dummy_exec_spec_path_is_the_loaded_generation(
    tmp_path: Path,
) -> None:
    spec_path, original = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
        integration=True,
    )
    mutated = deepcopy(original)
    shadow = mutated["runtime"]["systemd"]["immutable_shadow_properties"]
    alternate = (tmp_path / "other-generation" / "runtime-spec.json").resolve()
    for directive in supervisor._SYSTEMD_EXEC_MODES:
        identity = supervisor._normalized_systemd_exec_identity(
            shadow[directive]
        )
        argv = list(identity["argv"])
        argv[-1] = str(alternate)
        shadow[directive] = (
            f"{{ path={identity['path']} ; argv[]={' '.join(argv)}"
            f" ; ignore_errors={identity['ignore_errors']} }}"
        )
    _reseal_runtime_spec(spec_path, mutated)
    with pytest.raises(
        ValueError,
        match="systemd integration dummy execution roots are not exact",
    ):
        supervisor.load_runtime_spec(spec_path)


def test_integration_commit_orders_lease_final_audit_commit_and_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
        integration=True,
    )
    artifacts = spec["artifacts"]
    immutable = spec["runtime"]["systemd"][
        "immutable_shadow_properties"
    ]
    events: list[str] = []
    shadow_calls = 0
    phase_calls = 0

    def query_shadow(_unit: str) -> dict[str, str]:
        nonlocal shadow_calls
        shadow_calls += 1
        events.append(f"shadow-{shadow_calls}")
        if shadow_calls == 2:
            assert Path(str(artifacts["launch_lease"])).is_file()
        return dict(immutable)

    def query_phase(_unit: str) -> dict[str, str]:
        nonlocal phase_calls
        phase_calls += 1
        events.append(f"phase-{phase_calls}")
        if phase_calls <= 2:
            return _phase_state()
        return _phase_state(
            invocation_id=DUMMY_INVOCATION_ID,
            active_state="activating",
            sub_state="start",
        )

    starts = 0

    def start_once(*_args: object, **_kwargs: object) -> object:
        nonlocal starts
        starts += 1
        events.append("start")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(supervisor, "_query_systemd_shadow", query_shadow)
    monkeypatch.setattr(supervisor, "_query_systemd_phase_state", query_phase)
    monkeypatch.setattr(supervisor.subprocess, "run", start_once)
    assert supervisor.commit_and_start(spec_path) == 0
    assert starts == 1
    assert events[:4] == ["shadow-1", "phase-1", "shadow-2", "phase-2"]
    assert events[-2:] == ["start", "phase-3"]
    assert Path(str(artifacts["attempt_commit"])).is_file()
    assert Path(str(artifacts["precommit_phase_receipt"])).is_file()
    assert Path(str(artifacts["start_ack_receipt"])).is_file()
    assert not Path(str(artifacts["consumed_start_failure_receipt"])).exists()


def test_commit_materialization_uncertainty_preserves_launch_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
        integration=True,
    )
    artifacts = spec["artifacts"]
    immutable = spec["runtime"]["systemd"][
        "immutable_shadow_properties"
    ]
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_shadow",
        lambda _unit: dict(immutable),
    )
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_phase_state",
        lambda _unit: _phase_state(),
    )
    original_write = supervisor._write_new_json

    def uncertain_write(path: Path, payload: object) -> str:
        if path == Path(str(artifacts["attempt_commit"])):
            raise OSError(5, "generated uncertain commit write")
        return original_write(path, payload)

    monkeypatch.setattr(supervisor, "_write_new_json", uncertain_write)
    with pytest.raises(OSError):
        supervisor.commit_and_start(spec_path)
    assert Path(str(artifacts["launch_lease"])).is_file()
    assert Path(str(artifacts["precommit_phase_receipt"])).is_file()


def test_final_live_audit_failure_releases_only_uncommitted_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
        integration=True,
    )
    artifacts = spec["artifacts"]
    immutable = spec["runtime"]["systemd"][
        "immutable_shadow_properties"
    ]
    calls = 0

    def drifting_shadow(_unit: str) -> dict[str, str]:
        nonlocal calls
        calls += 1
        value = dict(immutable)
        if calls == 2:
            value["Restart"] = "on-failure"
        return value

    monkeypatch.setattr(supervisor, "_query_systemd_shadow", drifting_shadow)
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_phase_state",
        lambda _unit: _phase_state(),
    )
    with pytest.raises(PermissionError):
        supervisor.commit_and_start(spec_path)
    assert not Path(str(artifacts["launch_lease"])).exists()
    assert not Path(str(artifacts["attempt_commit"])).exists()


@pytest.mark.parametrize(
    ("start_return_code", "ack_fails", "expected_category"),
    [
        (7, False, "SYSTEMCTL_START_RETURNED_NONZERO"),
        (0, True, "BOUNDED_START_ACK_FAILED"),
    ],
)
def test_consumed_start_failure_receipt_and_no_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start_return_code: int,
    ack_fails: bool,
    expected_category: str,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
        integration=True,
    )
    artifacts = spec["artifacts"]
    immutable = spec["runtime"]["systemd"][
        "immutable_shadow_properties"
    ]
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_shadow",
        lambda _unit: dict(immutable),
    )
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_phase_state",
        lambda _unit: _phase_state(),
    )
    starts = 0

    def start_once(*_args: object, **_kwargs: object) -> object:
        nonlocal starts
        starts += 1
        return subprocess.CompletedProcess(
            [], start_return_code, stdout="generated", stderr="failure"
        )

    monkeypatch.setattr(supervisor.subprocess, "run", start_once)
    if ack_fails:
        monkeypatch.setattr(
            supervisor,
            "_await_start_ack",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("generated ack failure")
            ),
        )
    with pytest.raises(RuntimeError):
        supervisor.commit_and_start(spec_path)
    assert starts == 1
    receipt = _read_json(artifacts["consumed_start_failure_receipt"])
    assert receipt["category"] == expected_category
    assert receipt["attempt_consumed"] is True
    assert receipt["automatic_retry_allowed"] is False
    with pytest.raises(PermissionError):
        supervisor.commit_and_start(spec_path)
    assert starts == 1


def test_systemctl_start_exception_is_consumed_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
        integration=True,
    )
    artifacts = spec["artifacts"]
    immutable = spec["runtime"]["systemd"][
        "immutable_shadow_properties"
    ]
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_shadow",
        lambda _unit: dict(immutable),
    )
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_phase_state",
        lambda _unit: _phase_state(),
    )
    starts = 0

    def start_raises(*_args: object, **_kwargs: object) -> object:
        nonlocal starts
        starts += 1
        raise OSError(5, "generated systemctl exec failure")

    monkeypatch.setattr(supervisor.subprocess, "run", start_raises)
    with pytest.raises(OSError):
        supervisor.commit_and_start(spec_path)
    assert starts == 1
    receipt = _read_json(artifacts["consumed_start_failure_receipt"])
    assert receipt["category"] == "SYSTEMCTL_START_RAISED"
    assert receipt["attempt_consumed"] is True
    with pytest.raises(PermissionError):
        supervisor.commit_and_start(spec_path)
    assert starts == 1


def test_systemd_integration_dummy_prespawn_receipt_precedes_only_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
        integration=True,
    )
    artifacts = spec["artifacts"]
    immutable = spec["runtime"]["systemd"][
        "immutable_shadow_properties"
    ]
    lease, _commit = _commit_integration_spec(spec)
    start_state = _phase_state(
        invocation_id=DUMMY_INVOCATION_ID,
        active_state="activating",
        sub_state="start",
    )
    supervisor._write_phase_receipt(
        spec,
        phase="start_ack",
        phase_state=start_state,
        immutable_shadow=immutable,
        path=Path(str(artifacts["start_ack_receipt"])),
        launch_lease=lease,
    )
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_shadow",
        lambda _unit: dict(immutable),
    )
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_phase_state",
        lambda _unit: dict(start_state),
    )
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_runtime_identity",
        lambda _unit: {
            "invocation_id": DUMMY_INVOCATION_ID,
            "control_group": "/generated.slice/integration.service",
        },
    )
    monkeypatch.setattr(
        supervisor,
        "_self_cgroup_path",
        lambda: "/generated.slice/integration.service",
    )
    monkeypatch.setattr(supervisor, "_cgroup_other_pids", lambda _cg: set())
    assert supervisor.claim_materialization(
        spec_path,
        environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
    ) == 0
    original_popen = subprocess.Popen
    popen_calls = 0

    def checked_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal popen_calls
        popen_calls += 1
        assert Path(
            str(artifacts["child_prespawn_phase_receipt"])
        ).is_file()
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(supervisor.subprocess, "Popen", checked_popen)
    assert supervisor.run_once(
        spec_path,
        environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
    ) == 0
    assert popen_calls == 1
    prespawn = _read_json(artifacts["child_prespawn_phase_receipt"])
    assert prespawn["phase"] == "child_prespawn"
    assert prespawn["automatic_retry_allowed"] is False
    terminal = _read_json(artifacts["runtime_terminal"])
    assert terminal["start_ack_receipt_fingerprint"] is not None
    assert terminal["child_prespawn_phase_receipt_fingerprint"] is not None
    assert supervisor.finalize_systemd(
        spec_path,
        environment={
            "SERVICE_RESULT": "success",
            "EXIT_CODE": "exited",
            "EXIT_STATUS": "0",
            "INVOCATION_ID": DUMMY_INVOCATION_ID,
        },
    ) == 0
    sidecar = _read_json(
        Path(str(artifacts["systemd_invocation_dir"]))
        / f"{DUMMY_INVOCATION_ID}.json"
    )
    assert sidecar["start_ack_valid"] is True
    assert sidecar["child_prespawn_valid"] is True
    assert sidecar["audit_valid"] is True


def test_shell_false_exactly_one_launch_and_cross_process_replay_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "literal-argument.txt"
    injected = tmp_path / "must-not-exist"
    literal = f"; touch {injected}"
    code = (
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')"
    )
    spec_path, spec = _dummy_spec(
        tmp_path,
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(output),
            literal,
        ],
    )
    original_popen = subprocess.Popen
    calls: list[dict[str, object]] = []

    def recording_popen(
        *args: object,
        **kwargs: object,
    ) -> subprocess.Popen[bytes]:
        calls.append(dict(kwargs))
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(supervisor.subprocess, "Popen", recording_popen)
    assert _claim_and_run(spec_path) == 0
    assert output.read_text(encoding="utf-8") == literal
    assert injected.exists() is False
    assert len(calls) == 1
    assert calls[0]["shell"] is False
    assert calls[0]["start_new_session"] is True

    monkeypatch.setenv("INVOCATION_ID", DUMMY_INVOCATION_ID)
    assert supervisor.main(["run-once", "--spec", str(spec_path)]) == (
        os.EX_CANTCREAT
    )
    assert len(calls) == 1
    terminal = _read_json(spec["artifacts"]["runtime_terminal"])
    assert terminal["child_outcome"]["category"] == "EXITED_0"
    assert terminal["scientific_gate_passed"] is None


def test_spawn_failure_consumes_claim_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
    )
    calls = 0

    def broken_popen(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise OSError(5, "generated spawn failure")

    monkeypatch.setattr(supervisor.subprocess, "Popen", broken_popen)
    assert supervisor.claim_materialization(
        spec_path,
        environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
    ) == 0
    monkeypatch.setenv("INVOCATION_ID", DUMMY_INVOCATION_ID)
    assert (
        supervisor.main(["run-once", "--spec", str(spec_path)])
        == os.EX_OSERR
    )
    assert calls == 1
    terminal = _read_json(spec["artifacts"]["runtime_terminal"])
    assert terminal["child_outcome"]["category"] == "SPAWN_FAILED"

    assert (
        supervisor.main(["run-once", "--spec", str(spec_path)])
        == os.EX_CANTCREAT
    )
    assert calls == 1


def test_numbered_heartbeats_are_o_excl_hash_chained(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [
            sys.executable,
            "-I",
            "-c",
            "import time; time.sleep(0.09); raise SystemExit(42)",
        ],
    )
    assert _claim_and_run(spec_path) == 42
    artifacts = spec["artifacts"]
    heartbeat_paths = sorted(
        Path(str(artifacts["heartbeat_dir"])).glob("*.json")
    )
    assert len(heartbeat_paths) >= 3
    previous_sha256 = _sha256(
        Path(str(artifacts["materialization_claim"]))
    )
    for sequence, path in enumerate(heartbeat_paths):
        event = _read_json(path)
        assert event["sequence"] == sequence
        assert event["previous_event_file_sha256"] == previous_sha256
        assert path.stat().st_mode & 0o777 == 0o444
        assert path.stat().st_nlink == 1
        previous_sha256 = _sha256(path)
    terminal = _read_json(artifacts["runtime_terminal"])
    assert terminal["child_outcome"]["category"] == "EXITED_NONZERO"
    assert terminal["heartbeat_event_count"] == len(heartbeat_paths)
    assert terminal["last_heartbeat_file_sha256"] == previous_sha256


def test_sigterm_is_forwarded_to_dummy_child_process_group(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "child-ready"
    child_code = (
        "from pathlib import Path; import signal,sys,time; "
        f"p=Path({str(ready)!r}); "
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(23)); "
        "p.write_text('ready', encoding='utf-8'); "
        "\nwhile True: time.sleep(0.01)"
    )
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", child_code],
    )
    assert supervisor.claim_materialization(
        spec_path,
        environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
    ) == 0
    child_environment = os.environ.copy()
    child_environment["INVOCATION_ID"] = DUMMY_INVOCATION_ID
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(SUPERVISOR_PATH),
            "run-once",
            "--spec",
            str(spec_path),
        ],
        cwd=REPOSITORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_environment,
    )
    deadline = time.monotonic() + 3.0
    while not ready.exists() and time.monotonic() < deadline:
        assert process.poll() is None
        time.sleep(0.01)
    assert ready.exists()
    os.kill(process.pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=3.0)
    assert process.returncode == 23, (stdout, stderr)
    terminal = _read_json(spec["artifacts"]["runtime_terminal"])
    outcome = terminal["child_outcome"]
    assert outcome["category"] == "FORWARDED_SIGNAL_THEN_EXITED_NONZERO"
    assert outcome["forwarded_signals"] == [int(signal.SIGTERM)]


@pytest.mark.parametrize(
    ("return_code", "signals", "expected"),
    [
        (0, (), "EXITED_0"),
        (7, (), "EXITED_NONZERO"),
        (-int(signal.SIGTERM), (), "SIGNALED"),
        (7, (int(signal.SIGTERM),), "FORWARDED_SIGNAL_THEN_EXITED_NONZERO"),
    ],
)
def test_child_exit_classification_is_mechanical(
    return_code: int,
    signals: tuple[int, ...],
    expected: str,
) -> None:
    outcome = supervisor.classify_child_exit(
        return_code,
        forwarded_signals=signals,
    )
    assert outcome["category"] == expected
    assert "scientific_gate_passed" not in outcome


@pytest.mark.parametrize(
    ("service_result", "exit_code", "exit_status", "expected"),
    [
        ("success", "exited", "0", "SYSTEMD_SERVICE_SUCCESS"),
        ("exit-code", "exited", "7", "SYSTEMD_MAIN_EXIT_NONZERO"),
        ("signal", "killed", "TERM", "SYSTEMD_MAIN_SIGNAL"),
        ("core-dump", "dumped", "SEGV", "SYSTEMD_MAIN_CORE_DUMP"),
        ("timeout", "killed", "KILL", "SYSTEMD_TIMEOUT"),
        ("watchdog", "killed", "ABRT", "SYSTEMD_WATCHDOG"),
        ("oom-kill", "killed", "KILL", "SYSTEMD_OOM_KILL"),
        ("resources", "exited", "1", "SYSTEMD_RESOURCE_FAILURE"),
        ("protocol", "exited", "1", "SYSTEMD_PROTOCOL_FAILURE"),
        (
            "start-limit-hit",
            "exited",
            "1",
            "SYSTEMD_START_LIMIT_HIT",
        ),
        (
            "exec-condition",
            "exited",
            "1",
            "SYSTEMD_EXEC_CONDITION",
        ),
        ("novel-result", "unknown", "unknown", "SYSTEMD_OTHER_FAILURE"),
    ],
)
def test_systemd_exit_classification_is_dummy_mapping_only(
    service_result: str,
    exit_code: str,
    exit_status: str,
    expected: str,
) -> None:
    outcome = supervisor.classify_systemd_exit(
        {
            "SERVICE_RESULT": service_result,
            "EXIT_CODE": exit_code,
            "EXIT_STATUS": exit_status,
            "INVOCATION_ID": "a" * 32,
        }
    )
    assert outcome["category"] == expected
    assert outcome["scientific_gate_passed"] is None


def test_systemd_sidecars_distinguish_first_and_second_invocation(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
    )
    preclaim_environment = {
        "SERVICE_RESULT": "success",
        "EXIT_CODE": "exited",
        "EXIT_STATUS": "0",
        "INVOCATION_ID": "b" * 32,
    }
    assert supervisor.finalize_systemd(
        spec_path,
        environment=preclaim_environment,
    ) == 0
    sidecar_dir = Path(
        str(spec["artifacts"]["systemd_invocation_dir"])
    )
    preclaim = _read_json(sidecar_dir / f"{'b' * 32}.json")
    assert preclaim["audit_valid"] is False
    assert preclaim["claim_valid"] is False

    assert supervisor.claim_materialization(
        spec_path,
        environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
    ) == 0
    first_environment = {
        "SERVICE_RESULT": "success",
        "EXIT_CODE": "exited",
        "EXIT_STATUS": "0",
        "INVOCATION_ID": DUMMY_INVOCATION_ID,
    }
    assert supervisor.finalize_systemd(
        spec_path,
        environment=first_environment,
    ) == 0
    terminal = _read_json(
        sidecar_dir / f"{DUMMY_INVOCATION_ID}.json"
    )
    assert (
        terminal["systemd_outcome"]["category"]
        == "SYSTEMD_SERVICE_SUCCESS"
    )
    assert terminal["audit_valid"] is True
    assert terminal["scientific_gate_passed"] is None

    second_invocation = "c" * 32
    with pytest.raises(FileExistsError):
        supervisor.claim_materialization(
            spec_path,
            environment={"INVOCATION_ID": second_invocation},
        )
    second_environment = {
        "SERVICE_RESULT": "exec-condition",
        "EXIT_CODE": "exited",
        "EXIT_STATUS": str(os.EX_CANTCREAT),
        "INVOCATION_ID": second_invocation,
    }
    assert supervisor.finalize_systemd(
        spec_path,
        environment=second_environment,
    ) == 0
    second = _read_json(sidecar_dir / f"{second_invocation}.json")
    assert second["audit_valid"] is False
    assert second["claim_valid"] is True
    assert second["claim_matches_invocation"] is False
    with pytest.raises(FileExistsError):
        supervisor.finalize_systemd(
            spec_path,
            environment=second_environment,
        )


def test_systemd_exec_shadow_normalization_removes_only_runtime_fields() -> None:
    static = (
        "{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 -I -u "
        "/tmp/supervisor.py run-once --spec /tmp/spec.json ; "
        "ignore_errors=no"
    )
    first = (
        static
        + " ; start_time=[Wed 2026-07-29 22:00:00 CST]"
        + " ; stop_time=[Wed 2026-07-29 22:00:01 CST]"
        + " ; pid=100 ; code=exited ; status=0 }"
    )
    second = (
        static
        + " ; start_time=[Wed 2026-07-29 23:00:00 CST]"
        + " ; stop_time=[n/a]"
        + " ; pid=999 ; code=(null) ; status=0/0 }"
    )
    normalized_first = supervisor._normalize_systemd_shadow_value(
        "ExecStart", first
    )
    normalized_second = supervisor._normalize_systemd_shadow_value(
        "ExecStart", second
    )
    assert normalized_first == normalized_second
    assert "argv[]=/usr/bin/python3 -I -u" in normalized_first
    assert "pid=" not in normalized_first
    assert supervisor._normalize_systemd_shadow_value(
        "Environment", "A=B"
    ) == "A=B"
    for malformed in (
        static + " ; pid=evil }",
        static + " ; start_time=garbage }",
        static + " ; pid=1 ; pid=2 }",
        static.replace("ignore_errors=no", "ignore_errors=garbage") + " }",
    ):
        with pytest.raises(
            ValueError,
            match="runtime fields|ambiguous|path and argv diverged",
        ):
            supervisor._normalize_systemd_shadow_value(
                "ExecStart",
                malformed,
            )


def test_systemd_shadow_query_validates_raw_exec_before_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, _spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
    )
    spec = supervisor.load_runtime_spec(spec_path)
    expected = dict(
        spec["runtime"]["systemd"]["immutable_shadow_properties"]
    )
    monkeypatch.setattr(
        supervisor,
        "_fixed_verified_manager_environment",
        lambda: {},
    )

    def install_query_result(shadow: dict[str, str]) -> None:
        stdout = "".join(
            f"{name}={shadow[name]}\n"
            for name in sorted(supervisor._SYSTEMD_IMMUTABLE_SHADOW_KEYS)
        )
        monkeypatch.setattr(
            supervisor.subprocess,
            "run",
            lambda argv, **_kwargs: subprocess.CompletedProcess(
                argv,
                0,
                stdout,
                "",
            ),
        )

    live = dict(expected)
    for name in supervisor._SYSTEMD_EXEC_KEYS:
        live[name] = (
            live[name][:-2]
            + " ; start_time=[n/a] ; stop_time=[n/a]"
            + " ; pid=0 ; code=(null) ; status=0/0 }"
        )
    install_query_result(live)
    assert supervisor._query_systemd_shadow("dummy.service") == expected

    for suffix in (
        " ; pid=evil }",
        " ; start_time=garbage }",
        " ; pid=1 ; pid=2 }",
    ):
        malformed = dict(expected)
        malformed["ExecStart"] = malformed["ExecStart"][:-2] + suffix
        install_query_result(malformed)
        with pytest.raises(ValueError, match="runtime fields|ambiguous"):
            supervisor._query_systemd_shadow("dummy.service")
    malformed = dict(expected)
    malformed["ExecStart"] = malformed["ExecStart"].replace(
        "ignore_errors=no",
        "ignore_errors=garbage",
    )
    install_query_result(malformed)
    with pytest.raises(ValueError, match="path and argv diverged"):
        supervisor._query_systemd_shadow("dummy.service")


def test_systemd_shadow_static_mutation_is_rejected(tmp_path: Path) -> None:
    spec_path, _spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
    )
    spec = supervisor.load_runtime_spec(spec_path)
    runtime = spec["runtime"]
    assert isinstance(runtime, dict)
    systemd = runtime["systemd"]
    assert isinstance(systemd, dict)
    observed = dict(systemd["immutable_shadow_properties"])
    observed["Restart"] = "on-failure"
    with pytest.raises(
        PermissionError,
        match="static-property-changed",
    ):
        supervisor.validate_systemd_shadow(spec, observed)


def test_systemd_shadow_exec_identity_ignores_only_runtime_serialization(
    tmp_path: Path,
) -> None:
    spec_path, _spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
    )
    spec = supervisor.load_runtime_spec(spec_path)
    expected = dict(
        spec["runtime"]["systemd"]["immutable_shadow_properties"]
    )
    observed = dict(expected)
    for name in supervisor._SYSTEMD_EXEC_KEYS:
        observed[name] = (
            observed[name][:-2]
            + " ; pid=9182 ; status=0/0"
            + " ; start_time=[Wed 2026-07-30 04:55:34 CST]"
            + " ; code=(null) ; stop_time=[n/a] }"
        )
    supervisor.validate_systemd_shadow(spec, observed)

    changed = dict(observed)
    changed["ExecCondition"] = changed["ExecCondition"].replace(
        "claim-materialization",
        "run-once",
        1,
    )
    with pytest.raises(
        PermissionError,
        match="exec-identity-changed",
    ) as caught:
        supervisor.validate_systemd_shadow(spec, changed)
    message = str(caught.value)
    assert "expected_fingerprint" in message
    assert "observed_fingerprint" in message
    assert "ExecCondition" in message
    for unauthorized in (
        expected["ExecStart"][:-2] + " ; unauthorized=yes }",
        expected["ExecStart"] + " GARBAGE",
    ):
        malformed = dict(expected)
        malformed["ExecStart"] = unauthorized
        with pytest.raises(
            PermissionError,
            match="malformed-exec-identity",
        ):
            supervisor.validate_systemd_shadow(spec, malformed)


def test_watchdog_disabled_contract_normalizes_phase_dependent_values(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
    )
    loaded = supervisor.load_runtime_spec(spec_path)
    assert loaded["runtime"]["systemd"]["immutable_shadow_properties"][
        "WatchdogUSec"
    ] == "disabled"

    expected = loaded["runtime"]["systemd"]["immutable_shadow_properties"]
    for live_value in ("infinity", "0"):
        observed = dict(expected)
        observed["WatchdogUSec"] = live_value
        supervisor.validate_systemd_shadow(loaded, observed)
    changed = dict(expected)
    changed["WatchdogUSec"] = "1s"
    with pytest.raises(
        PermissionError,
        match="static-property-changed",
    ):
        supervisor.validate_systemd_shadow(loaded, changed)

    invalid = json.loads(json.dumps(spec))
    systemd = invalid["runtime"]["systemd"]
    systemd["immutable_shadow_properties"]["WatchdogUSec"] = "0"
    systemd["immutable_shadow_fingerprint"] = supervisor.stable_fingerprint(
        systemd["immutable_shadow_properties"]
    )
    body = {
        key: value
        for key, value in invalid.items()
        if key != "runtime_spec_fingerprint"
    }
    invalid["runtime_spec_fingerprint"] = supervisor.stable_fingerprint(body)
    _write_canonical(spec_path, invalid)
    with pytest.raises(
        ValueError, match="systemd shadow properties are not exact"
    ):
        supervisor.load_runtime_spec(spec_path)


def test_actual_claim_rejects_forged_invocation_before_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _commit_actual_spec(spec_path, spec)
    runtime = spec["runtime"]
    assert isinstance(runtime, dict)
    systemd = runtime["systemd"]
    assert isinstance(systemd, dict)
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_shadow",
        lambda _unit: dict(systemd["immutable_shadow_properties"]),
    )
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_runtime_identity",
        lambda _unit: {
            "invocation_id": "b" * 32,
            "control_group": "/generated.slice/r2.service",
        },
    )
    monkeypatch.setattr(
        supervisor,
        "_self_cgroup_path",
        lambda: "/generated.slice/r2.service",
    )
    with pytest.raises(PermissionError):
        supervisor.claim_materialization(
            spec_path,
            environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
        )
    artifacts = spec["artifacts"]
    assert isinstance(artifacts, dict)
    assert not Path(str(artifacts["materialization_claim"])).exists()


def _prepare_actual_finalizer_case(
    spec_path: Path,
    spec: dict[str, object],
) -> dict[str, object]:
    authorization, commit = _commit_actual_spec(spec_path, spec)
    attempt_commit = supervisor._verify_attempt_commit(spec, authorization)
    artifacts = spec["artifacts"]
    runtime = spec["runtime"]
    assert isinstance(artifacts, dict)
    assert isinstance(runtime, dict)
    systemd = runtime["systemd"]
    assert isinstance(systemd, dict)
    launch_lease = supervisor._verify_launch_lease(spec)
    supervisor._write_phase_receipt(
        spec,
        phase="start_ack",
        phase_state=_phase_state(
            invocation_id=DUMMY_INVOCATION_ID,
            active_state="active",
            sub_state="running",
        ),
        immutable_shadow=systemd["immutable_shadow_properties"],
        path=Path(str(artifacts["start_ack_receipt"])),
        launch_lease=launch_lease,
    )
    gpu_lease = supervisor._verify_active_gpu_lease(spec, attempt_commit)
    supervisor._write_phase_receipt(
        spec,
        phase="child_prespawn",
        phase_state=_phase_state(
            invocation_id=DUMMY_INVOCATION_ID,
            active_state="active",
            sub_state="running",
        ),
        immutable_shadow=systemd["immutable_shadow_properties"],
        path=Path(str(artifacts["child_prespawn_phase_receipt"])),
        launch_lease=launch_lease,
        environment_audit=_fake_live_audit("child_prespawn"),
        gpu_lease=gpu_lease,
    )
    gpu_lease["handle"].close_without_release()
    claim = supervisor._materialization_claim(
        spec,
        authorization,
        attempt_commit,
        DUMMY_INVOCATION_ID,
        "/generated.slice/r2.service",
    )
    supervisor._write_new_json(
        Path(str(artifacts["materialization_claim"])),
        claim,
    )
    return commit


def test_actual_finalizer_without_commit_has_zero_artifact_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    monkeypatch.setattr(
        supervisor,
        "_validate_live_systemd_context",
        lambda _spec, _invocation: "/generated.slice/r2.service",
    )
    with pytest.raises(PermissionError):
        supervisor.finalize_systemd(
            spec_path,
            environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
        )
    artifacts = spec["artifacts"]
    assert isinstance(artifacts, dict)
    assert list(
        Path(str(artifacts["systemd_invocation_dir"])).iterdir()
    ) == []


def test_actual_finalizer_preserves_commit_after_authorization_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorization, commit = _commit_actual_spec(spec_path, spec)
    reference = spec["authorization"]
    assert isinstance(reference, dict)
    authorization_path = Path(str(reference["path"]))
    authorization_path.chmod(0o600)
    authorization_path.unlink()
    monkeypatch.setattr(
        supervisor,
        "_validate_live_systemd_context",
        lambda _spec, _invocation: "/generated.slice/r2.service",
    )
    environment = {
        "SERVICE_RESULT": "exec-condition",
        "EXIT_CODE": "exited",
        "EXIT_STATUS": str(os.EX_NOPERM),
        "INVOCATION_ID": DUMMY_INVOCATION_ID,
    }
    assert supervisor.finalize_systemd(
        spec_path,
        environment=environment,
    ) == 0
    artifacts = spec["artifacts"]
    assert isinstance(artifacts, dict)
    sidecar = _read_json(
        Path(str(artifacts["systemd_invocation_dir"]))
        / f"{DUMMY_INVOCATION_ID}.json"
    )
    assert sidecar["attempt_commit_valid"] is True
    assert sidecar["current_authorization_valid"] is False
    assert sidecar["current_runtime_closure_valid"] is True
    assert sidecar["current_runtime_closure_error_type"] is None
    assert sidecar["authorization_matches_commit"] is False
    for field in (
        "gpu_lease_file_sha256",
        "gpu_lease_device",
        "gpu_lease_inode",
        "gpu_lease_parent_device",
        "gpu_lease_parent_inode",
    ):
        assert sidecar[f"active_{field}"] == commit[field]
    assert sidecar["audit_valid"] is False


def test_actual_finalizer_consumes_release_parent_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    authorization, commit = _commit_actual_spec(spec_path, spec)
    attempt_commit = supervisor._verify_attempt_commit(spec, authorization)
    artifacts = spec["artifacts"]
    runtime = spec["runtime"]
    assert isinstance(artifacts, dict)
    assert isinstance(runtime, dict)
    systemd = runtime["systemd"]
    assert isinstance(systemd, dict)
    launch_lease = supervisor._verify_launch_lease(spec)
    supervisor._write_phase_receipt(
        spec,
        phase="start_ack",
        phase_state=_phase_state(
            invocation_id=DUMMY_INVOCATION_ID,
            active_state="active",
            sub_state="running",
        ),
        immutable_shadow=systemd["immutable_shadow_properties"],
        path=Path(str(artifacts["start_ack_receipt"])),
        launch_lease=launch_lease,
    )
    gpu_lease = supervisor._verify_active_gpu_lease(spec, attempt_commit)
    supervisor._write_phase_receipt(
        spec,
        phase="child_prespawn",
        phase_state=_phase_state(
            invocation_id=DUMMY_INVOCATION_ID,
            active_state="active",
            sub_state="running",
        ),
        immutable_shadow=systemd["immutable_shadow_properties"],
        path=Path(str(artifacts["child_prespawn_phase_receipt"])),
        launch_lease=launch_lease,
        environment_audit=_fake_live_audit("child_prespawn"),
        gpu_lease=gpu_lease,
    )
    gpu_lease["handle"].close_without_release()
    claim = supervisor._materialization_claim(
        spec,
        authorization,
        attempt_commit,
        DUMMY_INVOCATION_ID,
        "/generated.slice/r2.service",
    )
    supervisor._write_new_json(
        Path(str(artifacts["materialization_claim"])),
        claim,
    )
    monkeypatch.setattr(
        supervisor,
        "_validate_live_systemd_context",
        lambda _spec, _invocation: "/generated.slice/r2.service",
    )
    assert supervisor.finalize_systemd(
        spec_path,
        environment={
            "SERVICE_RESULT": "success",
            "EXIT_CODE": "exited",
            "EXIT_STATUS": "0",
            "INVOCATION_ID": DUMMY_INVOCATION_ID,
        },
    ) == 0
    release = _read_json(artifacts["gpu_lease_release_receipt"])
    assert release["tombstone_device"] == commit["gpu_lease_device"]
    assert release["tombstone_inode"] == commit["gpu_lease_inode"]
    assert release["lease_parent_device"] == (
        commit["gpu_lease_parent_device"]
    )
    assert release["lease_parent_inode"] == (
        commit["gpu_lease_parent_inode"]
    )
    sidecar = _read_json(
        Path(str(artifacts["systemd_invocation_dir"]))
        / f"{DUMMY_INVOCATION_ID}.json"
    )
    assert sidecar["gpu_lease_release_valid"] is True
    assert sidecar["audit_valid"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_key",
        "tombstone_sha",
        "release_kind",
        "gpu_uuid",
        "attempt_id",
        "active_path",
        "tombstone_path",
        "evidence_fingerprint",
    ),
)
def test_actual_finalizer_rejects_forged_release_receipt_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    commit = _prepare_actual_finalizer_case(spec_path, spec)
    environment = spec["environment"]
    artifacts = spec["artifacts"]
    assert isinstance(environment, dict)
    assert isinstance(artifacts, dict)
    module = supervisor._load_runtime_environment_module(spec)

    def forged_release(
        _spec: object,
        lease: dict[str, object],
        *,
        release_kind: str,
        attempt_consumed: bool,
        evidence_fingerprint: str,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": module.GPU_LEASE_RELEASE_SCHEMA,
            "released_at_utc": "generated-release-time",
            "release_kind": release_kind,
            "attempt_consumed": attempt_consumed,
            "lease_fingerprint": lease["gpu_lease_fingerprint"],
            "gpu_uuid": environment["selected_gpu_uuid"],
            "attempt_id": spec["attempt_id"],
            "evidence_fingerprint": evidence_fingerprint,
            "tombstone_path": environment["gpu_lease_tombstone_path"],
            "tombstone_file_sha256": lease["gpu_lease_file_sha256"],
            "tombstone_device": lease["gpu_lease_device"],
            "tombstone_inode": lease["gpu_lease_inode"],
            "active_lease_path": environment["gpu_lease_path"],
            "active_lease_absent": True,
            "lease_parent_device": lease["gpu_lease_parent_device"],
            "lease_parent_inode": lease["gpu_lease_parent_inode"],
        }
        if mutation == "extra_key":
            body["unexpected_release_identity"] = None
        elif mutation == "tombstone_sha":
            body["tombstone_file_sha256"] = "0" * 64
        elif mutation == "release_kind":
            body["release_kind"] = "uncommitted_forensic"
            body["attempt_consumed"] = False
        elif mutation == "gpu_uuid":
            body["gpu_uuid"] = (
                "GPU-00000000-0000-0000-0000-000000000000"
            )
        elif mutation == "attempt_id":
            body["attempt_id"] = "forged-attempt"
        elif mutation == "active_path":
            body["active_lease_path"] = str(
                Path(str(environment["gpu_lease_path"])).with_name(
                    "forged-active.lease"
                )
            )
        elif mutation == "tombstone_path":
            body["tombstone_path"] = str(
                Path(str(environment["gpu_lease_tombstone_path"])).with_name(
                    "forged-released.lease"
                )
            )
        elif mutation == "evidence_fingerprint":
            body["evidence_fingerprint"] = "0" * 64
        else:
            raise AssertionError("unknown generated release mutation")
        return {
            **body,
            "receipt_fingerprint": module.stable_fingerprint(body),
        }

    monkeypatch.setattr(
        supervisor,
        "_release_external_gpu_lease",
        forged_release,
    )
    monkeypatch.setattr(
        supervisor,
        "_validate_live_systemd_context",
        lambda _spec, _invocation: "/generated.slice/r2.service",
    )
    assert supervisor.finalize_systemd(
        spec_path,
        environment={
            "SERVICE_RESULT": "success",
            "EXIT_CODE": "exited",
            "EXIT_STATUS": "0",
            "INVOCATION_ID": DUMMY_INVOCATION_ID,
        },
    ) == 0
    sidecar = _read_json(
        Path(str(artifacts["systemd_invocation_dir"]))
        / f"{DUMMY_INVOCATION_ID}.json"
    )
    assert sidecar["active_gpu_lease_valid"] is True
    assert sidecar["gpu_lease_release_authorized"] is True
    assert sidecar["gpu_lease_release_valid"] is False
    assert sidecar["audit_valid"] is False
    assert Path(str(environment["gpu_lease_path"])).is_file()
    assert commit["gpu_lease_file_sha256"] == _sha256(
        Path(str(environment["gpu_lease_path"]))
    )


def test_actual_finalizer_marks_current_runtime_closure_drift_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorization, _commit = _commit_actual_spec(spec_path, spec)
    runtime = spec["runtime"]
    assert isinstance(runtime, dict)
    systemd = runtime["systemd"]
    assert isinstance(systemd, dict)
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_shadow",
        lambda _unit: dict(systemd["immutable_shadow_properties"]),
    )
    monkeypatch.setattr(
        supervisor,
        "_query_systemd_runtime_identity",
        lambda _unit: {
            "invocation_id": DUMMY_INVOCATION_ID,
            "control_group": "/generated.slice/r2.service",
        },
    )
    monkeypatch.setattr(
        supervisor,
        "_self_cgroup_path",
        lambda: "/generated.slice/r2.service",
    )
    assert supervisor.claim_materialization(
        spec_path,
        environment={"INVOCATION_ID": DUMMY_INVOCATION_ID},
    ) == 0

    def generated_runtime_drift(_spec: object) -> None:
        raise PermissionError("generated runtime closure drift")

    monkeypatch.setattr(
        supervisor,
        "_validate_runtime_filesystem",
        generated_runtime_drift,
    )
    environment = {
        "SERVICE_RESULT": "success",
        "EXIT_CODE": "exited",
        "EXIT_STATUS": "0",
        "INVOCATION_ID": DUMMY_INVOCATION_ID,
    }
    assert supervisor.finalize_systemd(
        spec_path,
        environment=environment,
    ) == 0
    artifacts = spec["artifacts"]
    assert isinstance(artifacts, dict)
    sidecar = _read_json(
        Path(str(artifacts["systemd_invocation_dir"]))
        / f"{DUMMY_INVOCATION_ID}.json"
    )
    assert sidecar["attempt_commit_valid"] is True
    assert sidecar["current_authorization_valid"] is True
    assert sidecar["authorization_matches_commit"] is True
    assert sidecar["claim_valid"] is True
    assert sidecar["claim_matches_invocation"] is True
    assert sidecar["current_runtime_closure_valid"] is False
    assert sidecar["current_runtime_closure_error_type"] == "PermissionError"
    assert sidecar["audit_valid"] is False


def test_setsid_grandchild_is_quiesced_before_logs_are_sealed(
    tmp_path: Path,
) -> None:
    child_code = """
import os
import signal
import time

pid = os.fork()
if pid == 0:
    os.setsid()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    print("generated-grandchild-ready", flush=True)
    time.sleep(10)
    raise SystemExit(0)
raise SystemExit(0)
"""
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", child_code],
    )
    started = time.monotonic()
    assert _claim_and_run(spec_path) == 0
    assert time.monotonic() - started < 3.0
    artifacts = spec["artifacts"]
    assert isinstance(artifacts, dict)
    terminal = _read_json(artifacts["runtime_terminal"])
    assert int(signal.SIGTERM) in terminal["process_group_cleanup_signals"]
    for key in ("stdout_log", "stderr_log"):
        receipt = terminal[key]
        assert receipt["mode"] == 0o444
        assert receipt["hardlink_count"] == 1


def test_nonquiescent_descendant_path_never_seals_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
    )

    def refuse_quiescence(*_args: object, **_kwargs: object) -> list[int]:
        raise RuntimeError("generated nonquiescent descendant")

    monkeypatch.setattr(
        supervisor,
        "_quiesce_runtime_descendants",
        refuse_quiescence,
    )
    assert _claim_and_run(spec_path) == os.EX_SOFTWARE
    artifacts = spec["artifacts"]
    assert isinstance(artifacts, dict)
    terminal = _read_json(artifacts["runtime_terminal"])
    assert terminal["supervisor_error_type"] == "RuntimeError"
    assert terminal["stdout_log"] is None
    assert terminal["stderr_log"] is None
    for key in ("stdout_log", "stderr_log"):
        assert Path(str(artifacts[key])).stat().st_mode & 0o777 == 0o600


def test_log_hardlink_is_rejected_before_sealing(tmp_path: Path) -> None:
    path = tmp_path / "generated.log"
    handle = supervisor._open_new_log(path)
    handle.write(b"generated\n")
    os.link(path, tmp_path / "generated-hardlink.log")
    with pytest.raises(RuntimeError):
        supervisor._finalize_log(handle, path)
    assert handle.closed
    assert path.stat().st_nlink == 2
    assert path.stat().st_mode & 0o777 == 0o600


def test_v2_environment_chain_is_bound_into_actual_authorization(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    authorization = _authorize_actual_spec(spec_path, spec)
    payload = _read_json(authorization["path"])
    expected = supervisor._environment_evidence_bindings(spec)
    for key, value in expected.items():
        assert payload[key] == value
    assert payload["runtime_spec_v2_fingerprint"] == spec[
        "runtime_spec_fingerprint"
    ]
    assert payload["supervisor_v2_source_closure_fingerprint"] == (
        supervisor.stable_fingerprint(spec["source_bindings"])
    )
    assert payload["preauthorization_D_R_payload_accessed"] is False
    assert payload["preauthorization_D_V_payload_accessed"] is False
    assert payload["preauthorization_D_T_payload_accessed"] is False


def test_actual_authorization_freshness_is_start_only_but_archivally_valid(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    authorization = _authorize_actual_spec(spec_path, spec)
    path = Path(str(authorization["path"]))
    payload = _read_json(path)
    past = datetime.now(timezone.utc) - timedelta(minutes=10)
    payload["issued_at_utc"] = past.isoformat().replace("+00:00", "Z")
    payload["expires_at_utc"] = (
        past + timedelta(seconds=300)
    ).isoformat().replace("+00:00", "Z")
    body = dict(payload)
    body.pop("authorization_fingerprint")
    payload["authorization_fingerprint"] = supervisor.stable_fingerprint(
        body
    )
    path.chmod(0o600)
    _write_canonical(path, payload)
    path.chmod(0o444)

    with pytest.raises(PermissionError, match="fresh r2 authorization"):
        supervisor._verify_actual_authorization(
            spec,
            spec_path=spec_path,
        )
    archived = supervisor._verify_actual_authorization(
        spec,
        spec_path=spec_path,
        require_fresh=False,
    )
    assert archived["authorization_fingerprint"] == payload[
        "authorization_fingerprint"
    ]


def test_actual_authorization_over_300_seconds_is_never_valid(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    authorization = _authorize_actual_spec(spec_path, spec)
    path = Path(str(authorization["path"]))
    payload = _read_json(path)
    issued = datetime.now(timezone.utc) - timedelta(seconds=1)
    payload["issued_at_utc"] = issued.isoformat().replace("+00:00", "Z")
    payload["expires_at_utc"] = (
        issued + timedelta(seconds=301)
    ).isoformat().replace("+00:00", "Z")
    body = dict(payload)
    body.pop("authorization_fingerprint")
    payload["authorization_fingerprint"] = supervisor.stable_fingerprint(
        body
    )
    path.chmod(0o600)
    _write_canonical(path, payload)
    path.chmod(0o444)
    with pytest.raises(PermissionError, match="fresh r2 authorization"):
        supervisor._verify_actual_authorization(
            spec,
            spec_path=spec_path,
            require_fresh=False,
        )


def test_environment_evidence_drift_rejects_actual_authorization(
    tmp_path: Path,
) -> None:
    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", str(SUPERVISOR_PATH)],
        actual=True,
    )
    _authorize_actual_spec(spec_path, spec)
    environment = spec["environment"]
    assert isinstance(environment, dict)
    policy = Path(str(environment["policy_path"]))
    policy.chmod(0o600)
    changed = _read_json(policy)
    changed["unexpected_drift"] = True
    _write_canonical(policy, changed)
    policy.chmod(0o444)
    with pytest.raises(PermissionError, match="file binding changed"):
        supervisor._verify_actual_authorization(spec, spec_path=spec_path)


def _sampled_inventory_binding_fixture(
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    first = {
        "schema_version": "generated-inventory-v1",
        "created_at_utc": "2026-07-30T00:00:01Z",
        "passed": True,
    }
    final = {
        "schema_version": "generated-inventory-v1",
        "created_at_utc": "2026-07-30T00:00:02Z",
        "passed": True,
    }
    stability = {
        "receipt_kind": "sampled",
        "sample_count": 2,
        "samples": [
            {"inventory": first, "passed": True, "blockers": []},
            {"inventory": final, "passed": True, "blockers": []},
        ],
        "passed": True,
        "blockers": [],
    }
    return stability, first, final


def test_final_stability_inventory_binding_accepts_exact_last_sample() -> None:
    stability, _first, final = _sampled_inventory_binding_fixture()
    selected = supervisor._validate_final_stability_inventory_binding(
        stability,
        deepcopy(final),
    )
    assert selected["inventory"] == final


def test_final_stability_inventory_binding_rejects_valid_first_sample() -> None:
    stability, first, _final = _sampled_inventory_binding_fixture()
    with pytest.raises(
        PermissionError,
        match="exact final stability sample",
    ):
        supervisor._validate_final_stability_inventory_binding(
            stability,
            deepcopy(first),
        )


def test_closed_environment_binding_comparison_is_type_exact(
    tmp_path: Path,
) -> None:
    landed = (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "runtime_environment_postcleanup_receipt_post_c3.json"
    )
    receipt = _read_json(landed)
    assert supervisor._validate_closed_environment_audit_receipt(
        runtime_environment,
        receipt,
        expected_passed=True,
    )
    changed = deepcopy(receipt)
    binding = changed["environment_binding"]
    assert isinstance(binding, dict)
    assert binding["runtime_directory_inode"] == 1
    binding["runtime_directory_inode"] = True
    body = dict(changed)
    body.pop("receipt_fingerprint")
    changed["receipt_fingerprint"] = supervisor.stable_fingerprint(body)
    path = (tmp_path / "binding-type-drift.json").resolve()
    _write_canonical(path, changed)
    with pytest.raises(
        PermissionError,
        match="environment audit binding changed",
    ):
        supervisor._validate_closed_environment_audit_receipt(
            runtime_environment,
            _read_json(path),
            expected_passed=True,
        )


class _EnvironmentModuleProxy:
    def __init__(self, base: object) -> None:
        self.base = base
        self.live_contracts: list[object] = []
        self.live_inventory: dict[str, object] | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self.base, name)

    def inspect_recovery_activation_guard(
        self,
        expected_guard: dict[str, object],
    ) -> dict[str, object]:
        return {**expected_guard, "file_type": "symlink"}

    def audit_environment_once(
        self,
        contract: object,
        **kwargs: object,
    ) -> dict[str, object]:
        if kwargs:
            return self.base.audit_environment_once(contract, **kwargs)
        if self.live_inventory is None:
            raise AssertionError("generated live inventory was not installed")
        self.live_contracts.append(contract)
        return self.base.audit_environment_once(
            contract,
            inventory_collector=lambda **_arguments: json.loads(
                supervisor.canonical_json(self.live_inventory)
            ),
            activation_guard_reader=self.inspect_recovery_activation_guard,
        )


def _landed_strict_environment_case(
) -> tuple[dict[str, object], _EnvironmentModuleProxy]:
    evidence = (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
    ).resolve()
    policy = evidence / "runtime_environment_policy_post_c3.json"
    cleanup = (
        evidence
        / "environment_cleanup_recovery_r1/cleanup-receipt.json"
    )
    stability = (
        evidence / "runtime_environment_stability_receipt_post_c3.json"
    )
    postcleanup = (
        evidence / "runtime_environment_postcleanup_receipt_post_c3.json"
    )
    selected_uuid = "GPU-12cdabd0-7910-8f4a-e4d7-e3c7867d1296"
    environment = {
        "policy_path": str(policy),
        "policy_file_sha256": _sha256(policy),
        "cleanup_receipt_path": str(cleanup),
        "cleanup_receipt_file_sha256": _sha256(cleanup),
        "stability_receipt_path": str(stability),
        "stability_receipt_file_sha256": _sha256(stability),
        "inventory_path": str(postcleanup),
        "inventory_file_sha256": _sha256(postcleanup),
        "selected_gpu_uuid": selected_uuid,
        "selected_gpu_pci_bus_id": "00000000:02:00.0",
        "selected_gpu_minor_number": 0,
    }
    spec: dict[str, object] = {
        "execution_kind": supervisor.ACTUAL_EXECUTION_KIND,
        "candidate": "GCR-PACRE-v24",
        "runtime_spec_fingerprint": "f" * 64,
        "environment": environment,
        "source_bindings": {
            "runtime_environment_file_sha256": _sha256(
                REPOSITORY
                / "tools/cure_lite_v24_runtime_environment.py"
            ),
        },
        "runtime": {
            "systemd": {
                "unit_name": "cure-lite-v24-gcr-pacre-dr-r2.service",
            },
        },
        "child": {
            "environment": {
                "CUDA_VISIBLE_DEVICES": selected_uuid,
            },
        },
    }
    base = supervisor._load_runtime_environment_module(spec)
    return spec, _EnvironmentModuleProxy(base)


def test_strict_environment_validator_closes_recovery_chain_without_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, module = _landed_strict_environment_case()
    monkeypatch.setattr(
        supervisor,
        "_load_runtime_environment_module",
        lambda _spec: module,
    )
    validated = STRICT_ENVIRONMENT_VALIDATOR(spec)
    contract = validated["contract"]
    assert contract.cleanup_mode == module.RECOVERY_CLEANUP_MODE
    assert contract.quiescence_mode == module.RECOVERY_QUIESCENCE_MODE
    assert contract.cleanup_nrestarts_baseline == (
        (
            "confa-v41-mshnet-nudt-clean-formal-20260718-v1.service",
            "18739",
        ),
    )
    assert validated["postcleanup_inventory"]["blockers"] == []
    assert validated["recovery_activation_guard_observation"] == {
        **contract.activation_guard,
        "file_type": "symlink",
    }


def test_strict_environment_validator_rejects_semantic_postcleanup_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, module = _landed_strict_environment_case()
    monkeypatch.setattr(
        supervisor,
        "_load_runtime_environment_module",
        lambda _spec: module,
    )
    environment = spec["environment"]
    assert isinstance(environment, dict)
    changed = _read_json(environment["inventory_path"])
    inventory = changed["inventory"]
    assert isinstance(inventory, dict)
    scope = inventory["unit_scope"]
    assert isinstance(scope, dict)
    shadows = scope["shadows"]
    assert isinstance(shadows, dict)
    conflict = (
        "confa-v41-mshnet-nudt-clean-formal-20260718-v1.service"
    )
    shadow = shadows[conflict]
    assert isinstance(shadow, dict)
    shadow["NRestarts"] = "18740"
    inventory_body = dict(inventory)
    inventory_body.pop("inventory_fingerprint")
    inventory["inventory_fingerprint"] = supervisor.stable_fingerprint(
        inventory_body
    )
    binding = changed["environment_binding"]
    assert isinstance(binding, dict)
    binding["inventory_fingerprint"] = inventory["inventory_fingerprint"]
    receipt_body = dict(changed)
    receipt_body.pop("receipt_fingerprint")
    changed["receipt_fingerprint"] = supervisor.stable_fingerprint(
        receipt_body
    )
    changed_path = (tmp_path / "changed-postcleanup.json").resolve()
    _write_canonical(changed_path, changed)
    changed_path.chmod(0o444)
    environment["inventory_path"] = str(changed_path)
    environment["inventory_file_sha256"] = _sha256(changed_path)
    with pytest.raises(
        PermissionError,
        match="postcleanup inventory is not the exact final stability sample",
    ):
        STRICT_ENVIRONMENT_VALIDATOR(spec)


def test_live_environment_audit_uses_recovery_contract_not_normal_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, module = _landed_strict_environment_case()
    monkeypatch.setattr(
        supervisor,
        "_load_runtime_environment_module",
        lambda _spec: module,
    )
    validated = STRICT_ENVIRONMENT_VALIDATOR(spec)
    module.live_inventory = validated["postcleanup_inventory"]
    monkeypatch.setattr(
        supervisor,
        "_validated_bound_environment_contract",
        lambda _spec: validated,
    )
    receipt = LIVE_ENVIRONMENT_VERIFIER(spec, phase="finalizer")
    assert receipt["runtime_environment_audit_valid"] is True
    assert receipt["inventory"]["blockers"] == []
    assert len(module.live_contracts) == 1
    live_contract = module.live_contracts[0]
    assert live_contract.cleanup_mode == module.RECOVERY_CLEANUP_MODE
    assert live_contract.quiescence_mode == module.RECOVERY_QUIESCENCE_MODE
    assert live_contract.require_target_ready is False


def test_systemctl_query_uses_only_fixed_verified_manager_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_environment = supervisor._fixed_verified_manager_environment()
    captured: dict[str, object] = {}

    def fake_run(argv: object, **kwargs: object) -> object:
        captured["argv"] = argv
        captured.update(kwargs)
        stdout = "\n".join(
            f"{key}={value}"
            for key, value in _phase_state().items()
        )
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    assert supervisor._query_systemd_phase_state("generated.service") == (
        _phase_state()
    )
    assert captured["env"] == expected_environment
    assert "CUDA_VISIBLE_DEVICES" not in expected_environment
    assert captured["shell"] is False


def test_integration_identity_is_unique_per_scenario_and_spec_bound(
    tmp_path: Path,
) -> None:
    first = supervisor.build_systemd_integration_identity(
        "success-case-0123456789abcdef"
    )
    second = supervisor.build_systemd_integration_identity(
        "ack-fault-fedcba9876543210"
    )
    assert first["attempt_id"] != second["attempt_id"]
    assert first["unit_name"] != second["unit_name"]
    assert first["unit_name"].endswith(".service")
    with pytest.raises(ValueError, match="unique and canonical"):
        supervisor.build_systemd_integration_identity("fixed")

    spec_path, spec = _dummy_spec(
        tmp_path,
        [sys.executable, "-I", "-c", "raise SystemExit(0)"],
        integration=True,
    )
    identity = supervisor.build_systemd_integration_identity(
        "generated-case-0123456789abcdef"
    )
    assert spec["attempt_id"] == identity["attempt_id"]
    assert spec["runtime"]["systemd"]["unit_name"] == identity["unit_name"]
    assert supervisor.load_runtime_spec(spec_path)["stage_id"] == (
        identity["stage_id"]
    )
