from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import cure_lite_v24_environment_cleanup as cleanup


BOOT_ID = "12345678-1234-1234-1234-123456789abc"
UNIT = cleanup.GPU0_CONFLICT_UNIT
TARGET = cleanup.ACTUAL_R2_TARGET_UNIT
GPU2 = cleanup.GPU2_DNANET_UNIT


def _snapshot(
    fragment: Path,
    *,
    unit: str = UNIT,
    unit_file_state: str = "enabled",
    active_state: str = "activating",
    sub_state: str = "auto-restart",
    nrestarts: str = "10",
    triggered_by: str = "",
    dropins: str = "",
) -> dict[str, object]:
    identity = cleanup.trusted_fragment_identity(fragment, fragment.stat())
    return {
        "Id": unit,
        "Description": f"audited restart loop {unit}",
        "LoadState": "loaded",
        "ActiveState": active_state,
        "SubState": sub_state,
        "UnitFileState": unit_file_state,
        "FragmentPath": str(fragment),
        "DropInPaths": dropins,
        "Restart": "on-failure",
        "RestartUSec": "30s",
        "NRestarts": nrestarts,
        "Result": "exit-code",
        "ControlGroup": "",
        "Environment": "CUDA_VISIBLE_DEVICES=0",
        "ExecStart": {
            "path": "/usr/bin/python3",
            "argv": "/usr/bin/python3 train.py",
            "ignore_errors": "no",
        },
        "TriggeredBy": triggered_by,
        "Triggers": "",
        "WantedBy": "default.target",
        "RequiredBy": "",
        "PartOf": "",
        "fragment_file_sha256": cleanup.file_sha256(fragment),
        "dropin_file_sha256": {},
        **identity,
    }


def _environment_receipt(
    tmp_path: Path,
    *,
    gpu_uuid: str = cleanup.SELECTED_GPU_UUID,
    blockers: list[str] | None = None,
) -> Path:
    uid = os.getuid()
    observed = datetime.now(timezone.utc)
    inventory_created = observed.isoformat().replace("+00:00", "Z")
    receipt_created = inventory_created
    identity = {
        "pid": 123,
        "starttime_ticks": 456,
        "uid": uid,
        "control_group": (
            f"/user.slice/user-{uid}.slice/user@{uid}.service/init.scope"
        ),
    }
    endpoint = {
        "uid": uid,
        "runtime_directory": f"/run/user/{uid}",
        "runtime_directory_device": 11,
        "runtime_directory_inode": 12,
        "bus_path": f"/run/user/{uid}/bus",
        "bus_device": 11,
        "bus_inode": 13,
    }
    gpu_body: dict[str, object] = {
        "schema_version": cleanup.GPU_SNAPSHOT_SCHEMA,
        "selected_gpu_uuid": gpu_uuid,
        "expected_uid": uid,
        "allowed_unit_ids": [UNIT],
        "strict_all_gpu_consumers": False,
        "devices": [{
            "index": 0,
            "uuid": gpu_uuid,
            "pci_bus_id": "00000000:02:00.0",
            "compute_mode": "Default",
            "mig_mode": None,
            "driver_version": "580.126.09",
            "minor_number": 0,
            "mps_state": "not_observed",
        }],
        "first_apps": [],
        "second_apps": [],
        "process_unit_mapping": [],
        "observations": [],
        "blockers": [],
        "passed": True,
    }
    gpu = {
        **gpu_body,
        "snapshot_fingerprint": cleanup.stable_fingerprint(gpu_body),
    }
    inventory_body: dict[str, object] = {
        "schema_version": cleanup.ENVIRONMENT_INVENTORY_SCHEMA,
        "created_at_utc": inventory_created,
        "uid": uid,
        "boot_id": BOOT_ID,
        "manager": {
            "state": "degraded",
            "allowed_states": ["running", "degraded"],
            "returncode": 1,
            "failed_units": [],
            "allowed_failed_unit_ids": list(cleanup.ALLOWED_FAILED_UNITS),
            "unexpected_failed_unit_ids": [],
            "scoped_failed_unit_ids": [],
            "identity": identity,
            "endpoint": endpoint,
        },
        "unit_scope": {
            "target_unit_id": TARGET,
            "conflict_unit_ids": [UNIT],
            "dependency_unit_ids": [],
            "require_target_ready": False,
            "shadows": {
                TARGET: {
                    "Id": TARGET, "LoadState": "not-found",
                    "ActiveState": "inactive", "SubState": "dead",
                    "UnitFileState": "", "Restart": "no",
                    "RestartUSec": "0",
                    "NRestarts": "0", "ControlGroup": "",
                    "FragmentPath": "", "DropInPaths": "",
                    "TriggeredBy": "", "Triggers": "", "WantedBy": "",
                    "RequiredBy": "", "PartOf": "",
                },
                UNIT: {
                    "Id": UNIT, "LoadState": "loaded",
                    "ActiveState": "activating", "SubState": "auto-restart",
                    "UnitFileState": "enabled", "Restart": "on-failure",
                    "RestartUSec": "30s",
                    "NRestarts": "10", "ControlGroup": "",
                    "FragmentPath": "/tmp/audited-gpu0.service",
                    "DropInPaths": "",
                    "TriggeredBy": "", "Triggers": "",
                    "WantedBy": "default.target", "RequiredBy": "",
                    "PartOf": "",
                },
            },
        },
        "gpu_snapshot": gpu,
        "blockers": (
            [cleanup.PRECLEANUP_BLOCKER] if blockers is None else blockers
        ),
        "passed": False,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    inventory = {
        **inventory_body,
        "inventory_fingerprint": cleanup.stable_fingerprint(inventory_body),
    }
    binding = {
        "inventory_fingerprint": inventory["inventory_fingerprint"],
        "boot_id": BOOT_ID,
        "runtime_directory": endpoint["runtime_directory"],
        "runtime_directory_device": endpoint["runtime_directory_device"],
        "runtime_directory_inode": endpoint["runtime_directory_inode"],
        "manager_identity": identity,
    }
    receipt_body: dict[str, object] = {
        "schema_version": cleanup.ENVIRONMENT_RECEIPT_SCHEMA,
        "created_at_utc": receipt_created,
        "command": "audit-only",
        "environment_binding": binding,
        "inventory": inventory,
        "passed": False,
        "error_type": None,
        "error_message": None,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    path = tmp_path / "environment-receipt.json"
    cleanup.write_create_once_json(
        path, receipt_body, fingerprint_key="receipt_fingerprint"
    )
    return path


def _sealed_plan_and_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object], Path]:
    fragment = tmp_path / "gpu0-conflict.service"
    fragment.write_text("[Service]\nRestart=on-failure\n", encoding="utf-8")
    environment_receipt = _environment_receipt(tmp_path)
    monkeypatch.setattr(cleanup, "_boot_id", lambda: BOOT_ID)
    plan = cleanup.build_plan(
        inventory_receipt_path=environment_receipt,
        mask_units=[UNIT],
        stop_units=[UNIT],
        snapshot_reader=lambda unit: _snapshot(fragment, unit=unit),
    )
    plan_path = tmp_path / "plan.json"
    plan_body = dict(plan)
    plan_body.pop("plan_fingerprint")
    cleanup.write_create_once_json(
        plan_path, plan_body, fingerprint_key="plan_fingerprint"
    )
    authorization = cleanup.build_authorization(
        plan_path=plan_path,
        plan=plan,
        authorization_basis="user explicitly requested exact GPU0 cleanup",
        explicit_user_instruction_id=cleanup.EXPLICIT_USER_INSTRUCTION_ID,
    )
    authorization_path = tmp_path / "authorization.json"
    auth_body = dict(authorization)
    auth_body.pop("authorization_fingerprint")
    cleanup.write_create_once_json(
        authorization_path,
        auth_body,
        fingerprint_key="authorization_fingerprint",
    )
    return plan_path, authorization_path, plan, fragment


def _sealed_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, dict[str, object], Path]:
    plan_path, authorization_path, plan, fragment = (
        _sealed_plan_and_authorization(tmp_path, monkeypatch)
    )
    authorization = cleanup.load_sealed_json(authorization_path)
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    before = {
        UNIT: _snapshot(fragment, nrestarts="11"),
        GPU2: _snapshot(fragment, unit=GPU2, nrestarts="11"),
    }
    intent_body: dict[str, object] = {
        "schema_version": cleanup.INTENT_SCHEMA,
        "created_at_utc": created,
        "plan_file_sha256": cleanup.file_sha256(plan_path),
        "plan_fingerprint": plan["plan_fingerprint"],
        "authorization_file_sha256": cleanup.file_sha256(
            authorization_path
        ),
        "authorization_fingerprint": authorization[
            "authorization_fingerprint"
        ],
        "boot_id": plan["boot_id"],
        "manager_generation": plan["manager_generation"],
        "before": before,
        "actions": plan["actions"],
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    intent_path = tmp_path / "cleanup-intent.json"
    intent = cleanup.write_create_once_json(
        intent_path,
        intent_body,
        fingerprint_key="intent_fingerprint",
    )
    terminal_body: dict[str, object] = {
        "schema_version": cleanup.TERMINAL_FAILURE_SCHEMA,
        "created_at_utc": created,
        "intent_fingerprint": intent["intent_fingerprint"],
        "completed_action_receipt_fingerprints": [],
        "error_type": "RuntimeError",
        "error_message": "runtime mask was not observed immediately",
        "inflight_action": {
            "action": plan["actions"][0],
            "argv": [
                cleanup.SYSTEMCTL_PATH,
                "--user",
                "mask",
                "--runtime",
                UNIT,
            ],
            "started_at_utc": created,
            "dispatch_attempted": True,
            "completion_observed": True,
            "returncode": 0,
            "stdout": "",
            "stderr": (
                "Created symlink "
                f"/run/user/{os.getuid()}/systemd/user/{UNIT} "
                "-> /dev/null.\n"
            ),
        },
        "automatic_rollback_performed": False,
        # This is the historical instrumentation bug being reconciled.
        "runtime_mask_may_remain": False,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    terminal_path = tmp_path / "cleanup-terminal-failure.json"
    cleanup.write_create_once_json(
        terminal_path,
        terminal_body,
        fingerprint_key="terminal_failure_fingerprint",
    )
    return (
        plan_path,
        authorization_path,
        intent_path,
        terminal_path,
        plan,
        fragment,
    )


def _guard() -> dict[str, object]:
    return {
        "mode": cleanup.RECOVERY_GUARD_MODE,
        "unit_name": UNIT,
        "path": f"/run/user/{os.getuid()}/systemd/user/{UNIT}",
        "target": "/dev/null",
        "owner_uid": os.getuid(),
        "device": 11,
        "inode": 12,
        "observed_unit_file_state": "enabled",
    }


def test_plan_is_exact_gpu0_only_and_deeply_validates_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment = tmp_path / "gpu0.service"
    fragment.write_text("[Service]\n", encoding="utf-8")
    receipt = _environment_receipt(tmp_path)
    monkeypatch.setattr(cleanup, "_boot_id", lambda: BOOT_ID)
    plan = cleanup.build_plan(
        inventory_receipt_path=receipt,
        mask_units=[UNIT],
        stop_units=[UNIT],
        snapshot_reader=lambda unit: _snapshot(fragment, unit=unit),
    )
    assert plan["protected_units"] == sorted(cleanup.PROTECTED_UNITS)
    assert [row["action"] for row in plan["actions"]] == [
        "mask-runtime", "stop"
    ]
    assert set(plan["unit_snapshots"]) == {UNIT, GPU2}
    assert all(row["unit_name"] != GPU2 for row in plan["actions"])
    assert plan["payload_authority"] == "none"
    assert plan["D_R_payload_accessed"] is False
    assert plan["D_V_payload_accessed"] is False
    assert plan["D_T_payload_accessed"] is False
    assert cleanup.validate_plan(plan) == plan

    with pytest.raises(ValueError, match="only exact"):
        cleanup.build_plan(
            inventory_receipt_path=receipt,
            mask_units=[TARGET],
            stop_units=[TARGET],
            snapshot_reader=lambda unit: _snapshot(fragment, unit=unit),
        )


def test_environment_receipt_wrong_gpu_or_blocker_fails_closed(
    tmp_path: Path,
) -> None:
    wrong_gpu = _environment_receipt(tmp_path, gpu_uuid="GPU-" + "a" * 32)
    with pytest.raises(PermissionError, match="scope or identity"):
        cleanup.validate_environment_receipt(cleanup.load_sealed_json(wrong_gpu))

    second = tmp_path / "second"
    second.mkdir(mode=0o700)
    wrong_blocker = _environment_receipt(second, blockers=[])
    with pytest.raises(PermissionError, match="scope or identity"):
        cleanup.validate_environment_receipt(
            cleanup.load_sealed_json(wrong_blocker)
        )


def test_environment_receipt_binds_actual_allowed_failed_units(
    tmp_path: Path,
) -> None:
    path = _environment_receipt(tmp_path)
    receipt = cleanup.load_sealed_json(path)
    assert receipt["inventory"]["manager"]["allowed_failed_unit_ids"] == [
        "sctransnet-formal800-gpu2-recovery-postprocess-s42-v1.service",
        "sctransnet-formal800-gpu2-recovery-s42-v1.service",
        "snap.firmware-updater.firmware-notifier.service",
    ]

    body = dict(receipt)
    body.pop("receipt_fingerprint")
    inventory = dict(body["inventory"])
    inventory.pop("inventory_fingerprint")
    manager = dict(inventory["manager"])
    manager["allowed_failed_unit_ids"] = [GPU2]
    inventory["manager"] = manager
    inventory["inventory_fingerprint"] = cleanup.stable_fingerprint(inventory)
    body["inventory"] = inventory
    body["environment_binding"] = {
        **body["environment_binding"],
        "inventory_fingerprint": inventory["inventory_fingerprint"],
    }
    body["receipt_fingerprint"] = cleanup.stable_fingerprint(body)
    with pytest.raises(PermissionError, match="scope or identity"):
        cleanup.validate_environment_receipt(body)


def test_environment_receipt_rejects_activation_closure_drift(
    tmp_path: Path,
) -> None:
    path = _environment_receipt(tmp_path)
    receipt = cleanup.load_sealed_json(path)
    body = dict(receipt)
    body.pop("receipt_fingerprint")
    inventory = dict(body["inventory"])
    inventory.pop("inventory_fingerprint")
    scope = dict(inventory["unit_scope"])
    shadows = {name: dict(value) for name, value in scope["shadows"].items()}
    shadows[UNIT]["TriggeredBy"] = "rogue.timer"
    scope["shadows"] = shadows
    inventory["unit_scope"] = scope
    inventory["inventory_fingerprint"] = cleanup.stable_fingerprint(inventory)
    body["inventory"] = inventory
    body["environment_binding"] = {
        **body["environment_binding"],
        "inventory_fingerprint": inventory["inventory_fingerprint"],
    }
    body["receipt_fingerprint"] = cleanup.stable_fingerprint(body)
    with pytest.raises(PermissionError, match="scope or identity"):
        cleanup.validate_environment_receipt(body)


def test_execstart_normalization_ignores_only_runtime_churn() -> None:
    first = (
        "{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 train.py ; "
        "ignore_errors=no ; start_time=[Wed 2026-07-30 10:00:00 CST] ; "
        "stop_time=[n/a] ; pid=100 ; code=(null) ; status=0/0 }"
    )
    second = (
        "{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 train.py ; "
        "ignore_errors=no ; start_time=[Wed 2026-07-30 10:01:00 CST] ; "
        "stop_time=[Wed 2026-07-30 10:00:59 CST] ; pid=200 ; "
        "code=exited ; status=1/FAILURE }"
    )
    assert cleanup.normalize_exec_start(first) == cleanup.normalize_exec_start(
        second
    ) == {
        "path": "/usr/bin/python3",
        "argv": "/usr/bin/python3 train.py",
        "ignore_errors": "no",
    }
    with pytest.raises(PermissionError, match="ambiguous"):
        cleanup.normalize_exec_start(
            "{ path=/a ; path=/b ; argv[]=/a ; ignore_errors=no }"
        )


def test_0664_fragment_requires_current_gid_with_exclusive_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment = tmp_path / "group-writable.service"
    fragment.write_text("[Service]\n", encoding="utf-8")
    fragment.chmod(0o664)
    current = SimpleNamespace(pw_name="current", pw_gid=os.getgid())
    extra = SimpleNamespace(pw_name="extra", pw_gid=os.getgid())
    group = SimpleNamespace(
        gr_gid=os.getgid(), gr_name="exclusive", gr_mem=[]
    )
    monkeypatch.setattr(cleanup.grp, "getgrgid", lambda gid: group)
    monkeypatch.setattr(cleanup.pwd, "getpwall", lambda: [current])
    monkeypatch.setattr(
        cleanup.pwd,
        "getpwnam",
        lambda name: SimpleNamespace(
            pw_uid=os.getuid() if name == "current" else os.getuid() + 1
        ),
    )
    identity = cleanup.trusted_fragment_identity(fragment, fragment.stat())
    assert identity["fragment_mode"] == 0o664
    assert identity["fragment_group_gid"] == os.getgid()
    assert identity["fragment_group_member_uids"] == [os.getuid()]

    monkeypatch.setattr(cleanup.pwd, "getpwall", lambda: [current, extra])
    with pytest.raises(PermissionError, match="exclusive"):
        cleanup.trusted_fragment_identity(fragment, fragment.stat())


def test_trigger_or_dropin_closure_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment = tmp_path / "gpu0.service"
    fragment.write_text("[Service]\n", encoding="utf-8")
    receipt = _environment_receipt(tmp_path)
    monkeypatch.setattr(cleanup, "_boot_id", lambda: BOOT_ID)
    with pytest.raises(ValueError, match="closure"):
        cleanup.build_plan(
            inventory_receipt_path=receipt,
            mask_units=[UNIT],
            stop_units=[UNIT],
            snapshot_reader=lambda unit: _snapshot(
                fragment,
                unit=unit,
                triggered_by=("rogue.timer" if unit == UNIT else ""),
            ),
        )


def test_authorization_binds_instruction_binaries_uid_generation_and_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, authorization_path, plan, _ = _sealed_plan_and_authorization(
        tmp_path, monkeypatch
    )
    verified = cleanup.validate_authorization(
        cleanup.load_sealed_json(authorization_path),
        plan_path=plan_path,
        plan=plan,
    )
    assert verified["explicit_user_instruction_id"] == (
        cleanup.EXPLICIT_USER_INSTRUCTION_ID
    )
    assert verified["payload_authority"] == "none"
    assert verified["D_R_payload_accessed"] is False
    assert verified["D_V_payload_accessed"] is False
    assert verified["D_T_payload_accessed"] is False
    assert set(verified["executable_bindings"]) == {
        "cleanup_tool", "environment_auditor", "python", "systemctl"
    }
    with pytest.raises(PermissionError, match="instruction"):
        cleanup.build_authorization(
            plan_path=plan_path,
            plan=plan,
            authorization_basis="not enough",
            explicit_user_instruction_id="wrong",
        )

    raw = cleanup.load_sealed_json(authorization_path)
    expired = dict(raw)
    expired.pop("authorization_fingerprint")
    past = datetime.now(timezone.utc) - timedelta(minutes=10)
    expired["issued_at_utc"] = past.isoformat().replace("+00:00", "Z")
    expired["expires_at_utc"] = (past + timedelta(seconds=10)).isoformat().replace(
        "+00:00", "Z"
    )
    expired["authorization_fingerprint"] = cleanup.stable_fingerprint(expired)
    with pytest.raises(PermissionError, match="stale"):
        cleanup.validate_authorization(expired, plan_path=plan_path, plan=plan)


def test_systemctl_allowlist_is_exact_and_reset_is_disabled() -> None:
    forbidden = (
        [cleanup.SYSTEMCTL_PATH, "--user", "start", UNIT],
        [cleanup.SYSTEMCTL_PATH, "--user", "restart", UNIT],
        [cleanup.SYSTEMCTL_PATH, "--user", "unmask", UNIT],
        [cleanup.SYSTEMCTL_PATH, "--user", "reset-failed", UNIT],
        [cleanup.SYSTEMCTL_PATH, "--user", "stop", TARGET],
        [cleanup.SYSTEMCTL_PATH, "--user", "stop", GPU2],
    )
    for argv in forbidden:
        with pytest.raises(ValueError, match="exact cleanup allowlist"):
            cleanup.run_systemctl(
                argv,
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(
                    args[0], 0, "", ""
                ),
            )
    with pytest.raises(SystemExit):
        cleanup.build_parser().parse_args([
            "build-plan",
            "--inventory-receipt", "/tmp/inventory.json",
            "--output", "/tmp/plan.json",
            "--exact-reset-unit", UNIT,
        ])


def test_execute_rechecks_every_boundary_and_seals_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, authorization_path, plan, fragment = (
        _sealed_plan_and_authorization(tmp_path, monkeypatch)
    )
    generations: list[dict[str, object]] = []
    monkeypatch.setattr(
        cleanup,
        "validate_live_manager_generation",
        lambda expected: generations.append(dict(expected)) or dict(expected),
    )
    snapshot_calls = 0

    def snapshots(unit: str) -> dict[str, object]:
        nonlocal snapshot_calls
        if unit == GPU2:
            return _snapshot(fragment, unit=GPU2)
        assert unit == UNIT
        snapshot_calls += 1
        if snapshot_calls <= 2:
            return _snapshot(fragment, nrestarts="11")
        if snapshot_calls <= 4:
            return _snapshot(
                fragment, unit_file_state="masked-runtime", nrestarts="11"
            )
        return _snapshot(
            fragment,
            unit_file_state="masked-runtime",
            active_state="inactive",
            sub_state="dead",
            nrestarts="11",
        )

    commands: list[list[str]] = []

    def runner(argv):
        commands.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    receipt = cleanup.execute_cleanup(
        plan_path=plan_path,
        authorization_path=authorization_path,
        receipt_directory=tmp_path / "receipts",
        snapshot_reader=snapshots,
        command_runner=runner,
    )
    assert commands == [
        [cleanup.SYSTEMCTL_PATH, "--user", "mask", "--runtime", UNIT],
        [cleanup.SYSTEMCTL_PATH, "--user", "stop", UNIT],
    ]
    assert receipt["passed"] is True
    assert receipt["payload_authority"] == "none"
    assert receipt["D_R_payload_accessed"] is False
    assert receipt["D_V_payload_accessed"] is False
    assert receipt["D_T_payload_accessed"] is False
    assert len(generations) >= 6
    assert receipt["manager_generation"] == plan["manager_generation"]
    assert not (tmp_path / "receipts/cleanup-terminal-failure.json").exists()
    for path in (tmp_path / "receipts").iterdir():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["payload_authority"] == "none"
        assert payload["D_R_payload_accessed"] is False
        assert payload["D_V_payload_accessed"] is False
        assert payload["D_T_payload_accessed"] is False
    assert all(
        path.stat().st_mode & 0o777 == 0o444
        for path in (tmp_path / "receipts").iterdir()
    )


def test_partial_failure_is_terminal_and_never_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, authorization_path, _, fragment = (
        _sealed_plan_and_authorization(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(
        cleanup,
        "validate_live_manager_generation",
        lambda expected: dict(expected),
    )
    snapshot_calls = 0

    def snapshots(unit: str) -> dict[str, object]:
        nonlocal snapshot_calls
        if unit == GPU2:
            return _snapshot(fragment, unit=GPU2)
        snapshot_calls += 1
        if snapshot_calls <= 2:
            return _snapshot(fragment)
        return _snapshot(fragment, unit_file_state="masked-runtime")

    commands: list[list[str]] = []

    def runner(argv):
        commands.append(list(argv))
        if argv[2] == "stop":
            raise TimeoutError("simulated stop timeout")
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    with pytest.raises(TimeoutError, match="simulated"):
        cleanup.execute_cleanup(
            plan_path=plan_path,
            authorization_path=authorization_path,
            receipt_directory=tmp_path / "failure-receipts",
            snapshot_reader=snapshots,
            command_runner=runner,
        )
    terminal = json.loads(
        (tmp_path / "failure-receipts/cleanup-terminal-failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert terminal["automatic_rollback_performed"] is False
    assert terminal["runtime_mask_may_remain"] is True
    assert terminal["inflight_action"]["action"]["action"] == "stop"
    assert terminal["payload_authority"] == "none"
    assert terminal["D_R_payload_accessed"] is False
    assert terminal["D_V_payload_accessed"] is False
    assert terminal["D_T_payload_accessed"] is False
    assert all(command[2] != "unmask" for command in commands)


def test_ineffective_mask_failure_marks_possible_runtime_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, authorization_path, _, fragment = (
        _sealed_plan_and_authorization(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(
        cleanup,
        "validate_live_manager_generation",
        lambda expected: dict(expected),
    )
    snapshot_calls = 0

    def snapshots(unit: str) -> dict[str, object]:
        nonlocal snapshot_calls
        if unit == GPU2:
            return _snapshot(fragment, unit=GPU2)
        snapshot_calls += 1
        return _snapshot(fragment, nrestarts=str(10 + snapshot_calls))

    with pytest.raises(RuntimeError, match="not observed"):
        cleanup.execute_cleanup(
            plan_path=plan_path,
            authorization_path=authorization_path,
            receipt_directory=tmp_path / "ineffective-mask",
            snapshot_reader=snapshots,
            command_runner=lambda argv: subprocess.CompletedProcess(
                list(argv),
                0,
                "",
                "Created symlink -> /dev/null.\n",
            ),
        )
    terminal = json.loads(
        (
            tmp_path
            / "ineffective-mask"
            / "cleanup-terminal-failure.json"
        ).read_text(encoding="utf-8")
    )
    assert terminal["completed_action_receipt_fingerprints"] == []
    assert terminal["inflight_action"]["action"]["action"] == "mask-runtime"
    assert terminal["inflight_action"]["returncode"] == 0
    assert terminal["runtime_mask_may_remain"] is True


def test_partial_mask_recovery_executes_only_one_stop_and_seals_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        plan_path,
        original_authorization_path,
        intent_path,
        terminal_path,
        plan,
        fragment,
    ) = _sealed_partial_failure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cleanup,
        "validate_live_manager_generation",
        lambda expected: dict(expected),
    )
    stopped = False

    def snapshots(unit: str) -> dict[str, object]:
        if unit == GPU2:
            return _snapshot(
                fragment,
                unit=GPU2,
                nrestarts="11",
            )
        return _snapshot(
            fragment,
            nrestarts="11",
            active_state="inactive" if stopped else "activating",
            sub_state="dead" if stopped else "auto-restart",
        )

    authorization = cleanup.build_recovery_authorization(
        plan_path=plan_path,
        original_authorization_path=original_authorization_path,
        intent_path=intent_path,
        terminal_failure_path=terminal_path,
        authorization_basis=(
            "user requested modification then run; reconcile exact partial "
            "runtime mask and stop only the audited GPU0 conflict"
        ),
        explicit_user_instruction_id=cleanup.RECOVERY_USER_INSTRUCTION_ID,
        validity_seconds=300,
        snapshot_reader=snapshots,
        activation_guard_reader=lambda generation: _guard(),
    )
    authorization_path = tmp_path / "recovery-authorization.json"
    authorization_body = dict(authorization)
    authorization_body.pop("recovery_authorization_fingerprint")
    cleanup.write_create_once_json(
        authorization_path,
        authorization_body,
        fingerprint_key="recovery_authorization_fingerprint",
    )
    commands: list[list[str]] = []

    def runner(argv):
        nonlocal stopped
        commands.append(list(argv))
        stopped = True
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    receipt = cleanup.execute_partial_cleanup_recovery(
        plan_path=plan_path,
        original_authorization_path=original_authorization_path,
        intent_path=intent_path,
        terminal_failure_path=terminal_path,
        recovery_authorization_path=authorization_path,
        receipt_directory=tmp_path / "recovery-receipts",
        snapshot_reader=snapshots,
        activation_guard_reader=lambda generation: _guard(),
        command_runner=runner,
    )
    assert commands == [
        [cleanup.SYSTEMCTL_PATH, "--user", "stop", UNIT]
    ]
    assert receipt["cleanup_mode"] == cleanup.RECOVERY_CLEANUP_MODE
    assert receipt["activation_guard"] == _guard()
    assert receipt["after"][UNIT]["UnitFileState"] == "enabled"
    assert receipt["after"][UNIT]["ActiveState"] == "inactive"
    assert receipt["after"][UNIT]["SubState"] == "dead"
    assert receipt["partial_lineage"][
        "legacy_runtime_mask_may_remain_false_reconciled"
    ] is True
    assert receipt["partial_lineage"]["original_stop_dispatched"] is False
    assert cleanup.validate_final_cleanup_receipt(receipt) == receipt
    assert len(receipt["action_receipt_fingerprints"]) == 1
    assert all(
        path.stat().st_mode & 0o777 == 0o444
        for path in (tmp_path / "recovery-receipts").iterdir()
    )


def test_partial_mask_recovery_fails_before_dispatch_on_guard_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        plan_path,
        original_authorization_path,
        intent_path,
        terminal_path,
        _,
        fragment,
    ) = _sealed_partial_failure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cleanup,
        "validate_live_manager_generation",
        lambda expected: dict(expected),
    )

    def snapshots(unit: str) -> dict[str, object]:
        return _snapshot(
            fragment,
            unit=unit,
            nrestarts="11",
        )

    authorization = cleanup.build_recovery_authorization(
        plan_path=plan_path,
        original_authorization_path=original_authorization_path,
        intent_path=intent_path,
        terminal_failure_path=terminal_path,
        authorization_basis="exact guarded recovery",
        explicit_user_instruction_id=cleanup.RECOVERY_USER_INSTRUCTION_ID,
        snapshot_reader=snapshots,
        activation_guard_reader=lambda generation: _guard(),
    )
    body = dict(authorization)
    body.pop("recovery_authorization_fingerprint")
    authorization_path = tmp_path / "recovery-authorization.json"
    cleanup.write_create_once_json(
        authorization_path,
        body,
        fingerprint_key="recovery_authorization_fingerprint",
    )
    drifted = {**_guard(), "inode": 99}
    commands: list[list[str]] = []
    with pytest.raises(PermissionError, match="guard drifted"):
        cleanup.execute_partial_cleanup_recovery(
            plan_path=plan_path,
            original_authorization_path=original_authorization_path,
            intent_path=intent_path,
            terminal_failure_path=terminal_path,
            recovery_authorization_path=authorization_path,
            receipt_directory=tmp_path / "must-not-exist",
            snapshot_reader=snapshots,
            activation_guard_reader=lambda generation: drifted,
            command_runner=lambda argv: commands.append(list(argv)),
        )
    assert commands == []
    assert not (tmp_path / "must-not-exist").exists()


def test_sealed_plan_rejects_hardlink_and_parent_must_be_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _, _, _ = _sealed_plan_and_authorization(tmp_path, monkeypatch)
    alias = tmp_path / "plan-hardlink.json"
    os.link(plan_path, alias)
    with pytest.raises(PermissionError, match="sealed"):
        cleanup.load_sealed_json(plan_path)

    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="owner 0700"):
        cleanup.write_create_once_json(
            public / "receipt.json",
            {"schema_version": "forbidden"},
            fingerprint_key="receipt_fingerprint",
        )
