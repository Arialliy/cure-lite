from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from tools import cure_lite_v24_dummy_child as dummy_child
from tools import cure_lite_v24_realize_systemd_unit as realizer
from tools import cure_lite_v24_runtime_supervisor as supervisor
from tools import cure_lite_v24_user_systemd_integration as integration


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "deploy/systemd/cure-lite-v24-supervisor-integration.service.template"
INTEGRATION_TOOL = ROOT / "tools/cure_lite_v24_user_systemd_integration.py"
REALIZER = ROOT / "tools/cure_lite_v24_realize_systemd_unit.py"
SUPERVISOR = ROOT / "tools/cure_lite_v24_runtime_supervisor.py"
DUMMY_CHILD = ROOT / "tools/cure_lite_v24_dummy_child.py"
PYTHON = Path(sys.executable).resolve()
SCENARIO = "success-case-0123456789abcdef"
INVOCATION = "b" * 32


def _completed(argv, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def _manager() -> dict[str, object]:
    uid = os.getuid()
    return {
        "boot_id": "12345678-1234-1234-1234-123456789abc",
        "identity": {
            "pid": 111, "starttime_ticks": 222, "uid": uid,
            "control_group": f"/user.slice/user-{uid}.slice/user@{uid}.service/init.scope",
        },
        "endpoint": {
            "uid": uid, "runtime_directory": f"/run/user/{uid}",
            "runtime_device": 11, "runtime_inode": 12,
            "bus_path": f"/run/user/{uid}/bus", "bus_device": 13,
            "bus_inode": 14,
        },
    }


def _phase_state(invocation: str, *, active="activating", sub="start-pre"):
    return {
        "LoadState": "loaded", "ActiveState": active, "SubState": sub,
        "UnitFileState": "static", "NRestarts": "0",
        "NeedDaemonReload": "no", "Result": "success", "ExecMainCode": "0",
        "ExecMainStatus": "0", "InvocationID": invocation,
    }


def _seal(path: Path, body: dict[str, object], field: str) -> dict[str, object]:
    return integration._write_sealed(path, body, fingerprint_field=field)


def test_write_sealed_rejects_parent_directory_generation_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    displaced = tmp_path / "displaced-control"
    path = control / "sealed.json"
    original_reader = integration._read_fd_bytes
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
        integration,
        "_read_fd_bytes",
        replace_parent_during_readback,
    )
    with pytest.raises(RuntimeError, match="readback changed"):
        integration._write_sealed(
            path,
            {"schema_version": "test-parent-v1", "value": "exact"},
            fingerprint_field="test_fingerprint",
        )
    assert replaced is True
    assert (displaced / path.name).exists()
    assert path.exists()


def test_realizer_freezes_safe_system_unit_path_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "canonical-system-unit-path"
    target.mkdir(mode=0o755)
    alias = tmp_path / "xdg-system-unit-path"
    alias.symlink_to(target, target_is_directory=True)
    identity = realizer._unit_path_identity(alias)
    assert identity["path_is_symlink"] is True
    assert identity["link_target"] == str(target)
    assert identity["resolved_path"] == str(target)
    assert identity["device"] == target.stat().st_dev
    assert identity["inode"] == target.stat().st_ino

    target.chmod(0o777)
    with pytest.raises(PermissionError, match="not trusted"):
        realizer._unit_path_identity(alias)


class FakeRunner:
    def __init__(self, unit_dir: Path, alternate: Path) -> None:
        self.unit_dir = unit_dir
        self.alternate = alternate
        self.calls: list[tuple[str, ...]] = []
        self.loaded = False
        self.commit_failure = False
        self.watchdog_usec_override: str | None = None
        self.authorization: dict[str, object] | None = None

    def __call__(self, argv):
        command = tuple(argv)
        self.calls.append(command)
        if command[0] == realizer.SYSTEMD_PATH:
            return _completed(argv, stdout=f"{self.unit_dir}\n")
        if command[0] == realizer.SYSTEMD_ANALYZE:
            return _completed(argv, stdout=f"{self.alternate}\n{self.unit_dir}\n")
        if command[:3] == (realizer.SYSTEMCTL_PATH, "--user", "daemon-reload"):
            self.loaded = any(self.unit_dir.iterdir())
            return _completed(argv)
        if command[:3] == (realizer.SYSTEMCTL_PATH, "--user", "show"):
            if any(value.startswith("--property=Type") for value in command):
                assert self.authorization is not None
                spec = integration._read_sealed(
                    Path(str(self.authorization["runtime_spec_binding"]["path"])),
                    fingerprint_field="runtime_spec_fingerprint",
                    schema=integration.RUNTIME_SPEC_SCHEMA,
                )
                values = dict(
                    spec["runtime"]["systemd"]["immutable_shadow_properties"]
                )
                values["WatchdogUSec"] = (
                    self.watchdog_usec_override
                    if self.watchdog_usec_override is not None
                    else "infinity"
                )
            else:
                fragment = (
                    self.unit_dir / self.authorization["identity"]["unit_name"]
                    if self.authorization is not None else self.unit_dir / "absent"
                )
                if self.loaded and fragment.exists():
                    values = {
                        "LoadState": "loaded", "UnitFileState": "static",
                        "ActiveState": "inactive", "SubState": "dead",
                        "FragmentPath": str(fragment), "DropInPaths": "",
                        "Transient": "no", "Restart": "no", "NRestarts": "0",
                        "NeedDaemonReload": "no",
                    }
                else:
                    values = {
                        "LoadState": "not-found", "UnitFileState": "",
                        "ActiveState": "inactive", "SubState": "dead",
                        "FragmentPath": "", "DropInPaths": "",
                        "Transient": "no", "Restart": "no", "NRestarts": "0",
                        "NeedDaemonReload": "no",
                    }
            return _completed(
                argv, stdout="".join(f"{key}={value}\n" for key, value in values.items())
            )
        if "commit-and-start" in command:
            if self.commit_failure:
                return _completed(argv, returncode=1, stderr="mock commit failure")
            self._materialize_supervisor_evidence()
            return _completed(argv)
        raise AssertionError(f"unexpected command: {command}")

    def _materialize_supervisor_evidence(self) -> None:
        assert self.authorization is not None
        spec = integration._read_sealed(
            Path(str(self.authorization["runtime_spec_binding"]["path"])),
            fingerprint_field="runtime_spec_fingerprint",
            schema=integration.RUNTIME_SPEC_SCHEMA,
        )
        artifacts = {key: Path(value) for key, value in spec["artifacts"].items()}
        common = {
            "execution_kind": integration.EXECUTION_KIND,
            "candidate": spec["candidate"], "stage_id": spec["stage_id"],
            "attempt_id": spec["attempt_id"],
            "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
        }
        lease = _seal(
            artifacts["launch_lease"],
            {"schema_version": integration.LAUNCH_LEASE_SCHEMA, **common,
             "boot_id": _manager()["boot_id"], "launch_limit": 1,
             "lease_scope": "attempt_dispatch_only",
             "gpu_exclusivity_claimed": False,
             "automatic_retry_allowed": False, "resume_allowed": False},
            "launch_lease_fingerprint",
        )
        precommit_state = _phase_state("", active="inactive", sub="dead")
        phases = {}
        immutable_fingerprint = spec["runtime"]["systemd"][
            "immutable_shadow_fingerprint"
        ]
        for key, phase, state in (
            ("precommit_phase_receipt", "precommit", precommit_state),
            ("start_ack_receipt", "start_ack", _phase_state(INVOCATION)),
            ("child_prespawn_phase_receipt", "child_prespawn", _phase_state(INVOCATION, active="active", sub="running")),
        ):
            phases[key] = _seal(
                artifacts[key],
                {"schema_version": integration.PHASE_RECEIPT_SCHEMA, **common,
                 "phase": phase, "systemd_phase_state": state,
                 "boot_id": _manager()["boot_id"],
                 "systemd_phase_state_fingerprint": integration.stable_fingerprint(state),
                 "immutable_shadow_fingerprint": immutable_fingerprint,
                 "runtime_environment_audit_valid": False,
                 "environment_audit_fingerprint": None,
                 "environment_inventory_fingerprint": None,
                 "gpu_lease_fingerprint": None,
                 "launch_lease_fingerprint": lease["launch_lease_fingerprint"],
                 "launch_limit": 1, "automatic_retry_allowed": False,
                 "resume_allowed": False},
                "phase_receipt_fingerprint",
            )
        attempt = _seal(
            artifacts["attempt_commit"],
            {"schema_version": integration.ATTEMPT_COMMIT_SCHEMA, **common,
             "boot_id": _manager()["boot_id"],
             "authorization_fingerprint": None, "authorization_file_sha256": None,
             "runtime_environment_audit_valid": False,
             "precommit_environment_audit_fingerprint": None,
             "precommit_environment_inventory_fingerprint": None,
             "gpu_lease_fingerprint": None, "gpu_lease_file_sha256": None,
             "gpu_lease_device": None, "gpu_lease_inode": None,
             "launch_lease_fingerprint": lease["launch_lease_fingerprint"],
             "launch_lease_file_sha256": integration.file_sha256(artifacts["launch_lease"]),
             "precommit_phase_receipt_fingerprint": phases["precommit_phase_receipt"]["phase_receipt_fingerprint"],
             "precommit_phase_receipt_file_sha256": integration.file_sha256(artifacts["precommit_phase_receipt"])},
            "attempt_commit_fingerprint",
        )
        claim = _seal(
            artifacts["materialization_claim"],
            {"schema_version": integration.MATERIALIZATION_CLAIM_SCHEMA, **common,
             "boot_id": _manager()["boot_id"],
             "authorization_fingerprint": None,
             "attempt_commit_fingerprint": attempt["attempt_commit_fingerprint"],
             "attempt_commit_file_sha256": integration.file_sha256(artifacts["attempt_commit"]),
             "systemd_invocation_id": INVOCATION,
             "systemd_control_group": "/user.slice/integration.scope",
             "launch_limit": 1, "shell": False,
             "automatic_retry_allowed": False, "resume_allowed": False},
            "materialization_claim_fingerprint",
        )
        _seal(
            artifacts["runtime_terminal"],
            {"schema_version": integration.RUNTIME_TERMINAL_SCHEMA, **common,
             "boot_id": _manager()["boot_id"],
             "systemd_invocation_id": INVOCATION,
             "child_outcome": {"category": "EXITED_0", "raw_return_code": 0},
             "supervisor_error_type": None,
             "materialization_claim_file_sha256": integration.file_sha256(artifacts["materialization_claim"]),
             "start_ack_receipt_fingerprint": phases["start_ack_receipt"]["phase_receipt_fingerprint"],
             "start_ack_receipt_file_sha256": integration.file_sha256(artifacts["start_ack_receipt"]),
             "child_prespawn_phase_receipt_fingerprint": phases["child_prespawn_phase_receipt"]["phase_receipt_fingerprint"],
             "child_prespawn_phase_receipt_file_sha256": integration.file_sha256(artifacts["child_prespawn_phase_receipt"])},
            "runtime_terminal_fingerprint",
        )
        _seal(
            artifacts["systemd_invocation_dir"] / f"{INVOCATION}.json",
            {"schema_version": integration.SYSTEMD_TERMINAL_SCHEMA, **common,
             "claim_systemd_invocation_id": INVOCATION,
             "sidecar_systemd_invocation_id": INVOCATION,
             "audit_valid": True,
             "systemd_outcome": {
                 "systemd_success": True,
                 "category": "SYSTEMD_SERVICE_SUCCESS",
                 "service_result": "success",
                 "exit_code": "exited",
                 "exit_status": "0",
                 "invocation_id": INVOCATION,
                 "scientific_gate_passed": None,
             },
             "materialization_claim_fingerprint": claim["materialization_claim_fingerprint"],
             "materialization_claim_file_sha256": integration.file_sha256(artifacts["materialization_claim"]),
             "attempt_commit_fingerprint": attempt["attempt_commit_fingerprint"],
             "attempt_commit_file_sha256": integration.file_sha256(artifacts["attempt_commit"]),
             "attempt_commit_required": True, "attempt_commit_valid": True,
             "current_authorization_valid": None,
             "current_runtime_closure_valid": True,
             "current_runtime_closure_error_type": None,
             "authorization_matches_commit": True,
             "claim_valid": True, "claim_matches_invocation": True,
             "start_ack_receipt_fingerprint": phases["start_ack_receipt"]["phase_receipt_fingerprint"],
             "start_ack_receipt_file_sha256": integration.file_sha256(artifacts["start_ack_receipt"]),
             "start_ack_valid": True,
             "child_prespawn_phase_receipt_fingerprint": phases["child_prespawn_phase_receipt"]["phase_receipt_fingerprint"],
             "child_prespawn_phase_receipt_file_sha256": integration.file_sha256(artifacts["child_prespawn_phase_receipt"]),
             "child_prespawn_valid": True,
             "finalizer_environment_audit_valid": None,
             "active_gpu_lease_fingerprint": None, "active_gpu_lease_valid": None,
             "gpu_lease_release_authorized": None, "gpu_lease_release_valid": None,
             "gpu_lease_release_receipt_fingerprint": None,
             "gpu_lease_tombstone_file_sha256": None},
            "systemd_terminal_fingerprint",
        )
        _seal(
            Path(str(self.authorization["control_artifacts"]["dummy_artifact"])),
            {"schema_version": integration.DUMMY_ARTIFACT_SCHEMA,
             "scenario_id": self.authorization["scenario_id"],
             "dataset_accessed": False, "gpu_accessed": False,
             "torch_imported": False, "pid": 123},
            "dummy_artifact_fingerprint",
        )


def _workspace(tmp_path: Path):
    unit_dir = tmp_path / "run/systemd/user"
    alternate = tmp_path / "config/systemd/user"
    unit_dir.mkdir(parents=True)
    alternate.mkdir(parents=True)
    unit_dir.chmod(0o700)
    alternate.chmod(0o755)
    runner = FakeRunner(unit_dir.resolve(), alternate.resolve())
    scenario_root = (tmp_path / "scenario").resolve()
    authorization = integration.create_production_authorization(
        scenario_root, scenario_id=SCENARIO, template_path=TEMPLATE.resolve(),
        python_path=PYTHON, supervisor_path=SUPERVISOR.resolve(),
        realizer_path=REALIZER.resolve(), dummy_child_path=DUMMY_CHILD.resolve(),
        instruction_id=integration.INSTRUCTION_ID,
        authorization_basis=integration.AUTHORIZATION_BASIS,
        runner=runner, manager_reader=lambda: deepcopy(_manager()),
    )
    runner.authorization = authorization
    return scenario_root, runner, authorization


def test_builder_creates_private_unique_spec_and_authorization(tmp_path: Path) -> None:
    scenario_root, runner, authorization = _workspace(tmp_path)
    control = scenario_root / "control"
    spec_path = control / "runtime-spec.json"
    auth_path = control / "authorization.json"
    assert stat.S_IMODE(scenario_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(spec_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o444
    spec = integration._read_sealed(
        spec_path, fingerprint_field="runtime_spec_fingerprint",
        schema=integration.RUNTIME_SPEC_SCHEMA,
    )
    assert spec["environment"] is None
    assert spec["execution_kind"] == integration.EXECUTION_KIND
    assert spec["runtime"]["systemd"]["immutable_shadow_properties"][
        "WatchdogUSec"
    ] == "disabled"
    loaded_by_supervisor = supervisor.load_runtime_spec(spec_path)
    assert (
        loaded_by_supervisor["runtime_spec_fingerprint"]
        == spec["runtime_spec_fingerprint"]
    )
    assert authorization["unit_removal_authorized"] is False
    assert authorization["direct_start_authorized"] is False
    assert authorization["unit_path_policy"]["runtime_directory_priority"] == 1
    assert runner.calls[:2] == [
        (realizer.SYSTEMD_PATH, "--suffix=systemd/user", "user-runtime"),
        (realizer.SYSTEMD_ANALYZE, "--user", "unit-paths", "--no-pager"),
    ]


def test_builder_rejects_user_mutable_python_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "python-target"
    target.write_bytes(b"generated-python-target\n")
    target.chmod(0o555)
    linked = tmp_path / "python-linked"
    linked.symlink_to(target)
    unit_dir = tmp_path / "run/systemd/user"
    alternate = tmp_path / "config/systemd/user"
    unit_dir.mkdir(parents=True)
    alternate.mkdir(parents=True)
    unit_dir.chmod(0o700)
    alternate.chmod(0o755)
    runner = FakeRunner(unit_dir.resolve(), alternate.resolve())
    with pytest.raises(PermissionError, match="safe regular file"):
        integration.create_production_authorization(
            (tmp_path / "scenario").resolve(),
            scenario_id=SCENARIO,
            template_path=TEMPLATE.resolve(),
            python_path=linked,
            supervisor_path=SUPERVISOR.resolve(),
            realizer_path=REALIZER.resolve(),
            dummy_child_path=DUMMY_CHILD.resolve(),
            instruction_id=integration.INSTRUCTION_ID,
            authorization_basis=integration.AUTHORIZATION_BASIS,
            runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
        )


def test_template_four_phases_are_exact_supervisor_v2() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    assert text.count("ExecCondition=") == 1
    assert text.count("ExecStartPre=") == 1
    assert text.count("ExecStart=") == 1
    assert text.count("ExecStopPost=") == 1
    for mode in integration._EXEC_MODES.values():
        assert f" {mode} --spec @@RUNTIME_SPEC_PATH@@" in text
    assert "INTEGRATION_TOOL" not in text
    assert "DUMMY_CHILD" not in text
    assert "[Install]" not in text and "systemd-run" not in text
    assert "WatchdogSec=0\n" in text


def test_watchdog_disabled_contract_accepts_both_phases_and_rejects_enabled(
    tmp_path: Path,
) -> None:
    scenario_root, runner, authorization = _workspace(tmp_path)
    spec = integration._read_sealed(
        scenario_root / "control/runtime-spec.json",
        fingerprint_field="runtime_spec_fingerprint",
        schema=integration.RUNTIME_SPEC_SCHEMA,
    )
    assert spec["runtime"]["systemd"]["immutable_shadow_properties"][
        "WatchdogUSec"
    ] == "disabled"
    runner.watchdog_usec_override = "0"
    observed = integration._query_immutable_shadow(
        authorization["identity"]["unit_name"],
        runner=runner,
    )
    assert observed["WatchdogUSec"] == "disabled"
    runner.watchdog_usec_override = "1s"
    with pytest.raises(
        PermissionError, match="loaded supervisor-v2 immutable shadow changed"
    ):
        integration.run_authorized_integration(
            scenario_root / "control/authorization.json",
            execute=True,
            runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
            timeout_seconds=0.1,
        )
    assert not any("commit-and-start" in command for command in runner.calls)


def test_success_uses_commit_and_start_then_terminal_authorized_removal(
    tmp_path: Path,
) -> None:
    scenario_root, runner, authorization = _workspace(tmp_path)
    result = integration.run_authorized_integration(
        scenario_root / "control/authorization.json", execute=True,
        runner=runner, manager_reader=lambda: deepcopy(_manager()),
        timeout_seconds=0.2,
    )
    assert result["terminal"]["passed"] is True
    evidence = result["terminal"]["supervisor_evidence"]
    assert evidence["invocation_id"] == INVOCATION
    assert evidence["runtime_attestation_absent"] is True
    assert evidence["gpu_lease_evidence_absent"] is True
    assert result["removal_state"]["passed"] is True
    assert result["removal_state"]["fragment_absent"] is True
    assert result["receipt"]["schema_version"] == (
        integration.INTEGRATION_RECEIPT_SCHEMA
    )
    assert result["receipt"]["passed"] is True
    assert result["receipt"]["fragment_removed"] is True
    assert result["receipt"]["D_R_payload_accessed"] is False
    assert result["receipt"]["D_V_payload_accessed"] is False
    assert result["receipt"]["D_T_payload_accessed"] is False
    spec = integration._read_sealed(
        Path(str(authorization["runtime_spec_binding"]["path"])),
        fingerprint_field="runtime_spec_fingerprint",
        schema=integration.RUNTIME_SPEC_SCHEMA,
    )
    artifact_modes = {
        key: stat.S_IMODE(Path(str(spec["artifacts"][key])).stat().st_mode)
        for key in (
            "attempt_commit",
            "launch_lease",
            "materialization_claim",
            "precommit_phase_receipt",
            "start_ack_receipt",
            "child_prespawn_phase_receipt",
            "runtime_terminal",
        )
    }
    assert artifact_modes == {
        "attempt_commit": 0o444,
        "launch_lease": 0o444,
        "materialization_claim": 0o444,
        "precommit_phase_receipt": 0o444,
        "start_ack_receipt": 0o444,
        "child_prespawn_phase_receipt": 0o444,
        "runtime_terminal": 0o444,
    }
    invocation_path = next(
        Path(str(spec["artifacts"]["systemd_invocation_dir"])).iterdir()
    )
    assert stat.S_IMODE(invocation_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(
        Path(
            str(authorization["control_artifacts"]["dummy_artifact"])
        ).stat().st_mode
    ) == 0o444
    control = authorization["control_artifacts"]
    removal_path = Path(str(control["removal_authorization"]))
    assert removal_path.is_file()
    assert stat.S_IMODE(removal_path.stat().st_mode) == 0o444
    removal = integration._read_sealed(
        removal_path, fingerprint_field="removal_authorization_fingerprint",
        schema=integration.REMOVAL_AUTHORIZATION_SCHEMA,
    )
    issued = datetime.fromisoformat(removal["issued_at_utc"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(removal["expires_at_utc"].replace("Z", "+00:00"))
    assert timedelta(0) < expires - issued <= timedelta(seconds=300)
    assert issued <= datetime.now(timezone.utc) <= expires
    joined = " ".join(" ".join(call) for call in runner.calls)
    assert "commit-and-start" in joined
    assert f" {realizer.SYSTEMCTL_PATH} --user start " not in f" {joined} "
    assert " enable " not in f" {joined} "
    commit_calls = [call for call in runner.calls if "commit-and-start" in call]
    assert len(commit_calls) == 1


def test_exec_condition_sidecar_fails_fast_without_runtime_terminal(
    tmp_path: Path,
) -> None:
    scenario_root, runner, authorization = _workspace(tmp_path)
    original_materialize = runner._materialize_supervisor_evidence

    def materialize_failure() -> None:
        original_materialize()
        spec = integration._read_sealed(
            Path(str(authorization["runtime_spec_binding"]["path"])),
            fingerprint_field="runtime_spec_fingerprint",
            schema=integration.RUNTIME_SPEC_SCHEMA,
        )
        artifacts = {
            key: Path(value) for key, value in spec["artifacts"].items()
        }
        for key in (
            "materialization_claim",
            "start_ack_receipt",
            "child_prespawn_phase_receipt",
            "runtime_terminal",
        ):
            artifacts[key].unlink()
        Path(
            str(authorization["control_artifacts"]["dummy_artifact"])
        ).unlink()
        sidecar_path = (
            artifacts["systemd_invocation_dir"] / f"{INVOCATION}.json"
        )
        sidecar_path.unlink()
        attempt = integration._read_sealed(
            artifacts["attempt_commit"],
            fingerprint_field="attempt_commit_fingerprint",
            schema=integration.ATTEMPT_COMMIT_SCHEMA,
        )
        _seal(
            sidecar_path,
            {
                "schema_version": integration.SYSTEMD_TERMINAL_SCHEMA,
                "candidate": spec["candidate"],
                "stage_id": spec["stage_id"],
                "attempt_id": spec["attempt_id"],
                "runtime_spec_fingerprint": spec[
                    "runtime_spec_fingerprint"
                ],
                "materialization_claim_fingerprint": None,
                "materialization_claim_file_sha256": None,
                "attempt_commit_fingerprint": attempt[
                    "attempt_commit_fingerprint"
                ],
                "attempt_commit_file_sha256": integration.file_sha256(
                    artifacts["attempt_commit"]
                ),
                "attempt_commit_required": True,
                "attempt_commit_valid": True,
                "current_authorization_valid": None,
                "current_runtime_closure_valid": True,
                "current_runtime_closure_error_type": None,
                "authorization_matches_commit": True,
                "systemd_control_group": "/user.slice/integration.scope",
                "claim_systemd_invocation_id": None,
                "sidecar_systemd_invocation_id": INVOCATION,
                "claim_valid": False,
                "claim_matches_invocation": False,
                "start_ack_receipt_fingerprint": None,
                "start_ack_receipt_file_sha256": None,
                "start_ack_valid": False,
                "child_prespawn_phase_receipt_fingerprint": None,
                "child_prespawn_phase_receipt_file_sha256": None,
                "child_prespawn_valid": False,
                "finalizer_environment_audit_fingerprint": None,
                "finalizer_environment_inventory_fingerprint": None,
                "finalizer_environment_audit_valid": None,
                "active_gpu_lease_fingerprint": None,
                "active_gpu_lease_valid": None,
                "gpu_lease_release_authorized": None,
                "gpu_lease_release_valid": None,
                "gpu_lease_release_receipt_fingerprint": None,
                "gpu_lease_tombstone_file_sha256": None,
                "audit_valid": False,
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "systemd_outcome": {
                    "category": "SYSTEMD_EXEC_CONDITION",
                    "exit_code": "unknown",
                    "exit_status": "unknown",
                    "invocation_id": INVOCATION,
                    "scientific_gate_passed": None,
                    "service_result": "exec-condition",
                    "systemd_success": False,
                },
                "scientific_decision": (
                    "NOT_EVALUATED_BY_RUNTIME_SUPERVISOR"
                ),
                "scientific_gate_passed": None,
            },
            "systemd_terminal_fingerprint",
        )

    runner._materialize_supervisor_evidence = materialize_failure
    with pytest.raises(
        RuntimeError,
        match="SYSTEMD_EXEC_CONDITION",
    ):
        integration.run_authorized_integration(
            scenario_root / "control/authorization.json",
            execute=True,
            runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
            timeout_seconds=5.0,
        )
    terminal = integration._read_sealed(
        Path(
            str(
                authorization["control_artifacts"][
                    "integration_terminal"
                ]
            )
        ),
        fingerprint_field="integration_terminal_fingerprint",
        schema=integration.INTEGRATION_TERMINAL_SCHEMA,
    )
    partial = terminal["supervisor_evidence"]
    assert terminal["passed"] is False
    assert terminal["error_type"] == "RuntimeError"
    assert "SYSTEMD_EXEC_CONDITION" in terminal["error_message"]
    assert partial["evidence_kind"] == "partial-systemd-terminal"
    assert partial["audit_valid"] is False
    assert partial["systemd_outcome"]["systemd_success"] is False
    assert not Path(
        str(
            integration._read_sealed(
                Path(str(authorization["runtime_spec_binding"]["path"])),
                fingerprint_field="runtime_spec_fingerprint",
                schema=integration.RUNTIME_SPEC_SCHEMA,
            )["artifacts"]["runtime_terminal"]
        )
    ).exists()


def test_systemd_sidecar_rejects_impossible_phase_causality() -> None:
    base = {
        "audit_valid": False,
        "claim_valid": False,
        "claim_matches_invocation": False,
        "start_ack_valid": False,
        "child_prespawn_valid": False,
        "systemd_outcome": {
            "category": "SYSTEMD_EXEC_CONDITION",
            "service_result": "exec-condition",
            "exit_code": "unknown",
            "exit_status": "unknown",
            "invocation_id": INVOCATION,
            "systemd_success": False,
            "scientific_gate_passed": None,
        },
    }
    integration._validate_systemd_sidecar_causality(base)
    impossible = dict(base)
    impossible["start_ack_valid"] = True
    with pytest.raises(PermissionError, match="causal lineage"):
        integration._validate_systemd_sidecar_causality(impossible)

    complete = {
        "audit_valid": True,
        "claim_valid": True,
        "claim_matches_invocation": True,
        "start_ack_valid": True,
        "child_prespawn_valid": True,
        "systemd_outcome": {
            "category": "SYSTEMD_SERVICE_SUCCESS",
            "service_result": "success",
            "exit_code": "exited",
            "exit_status": "0",
            "invocation_id": INVOCATION,
            "systemd_success": True,
            "scientific_gate_passed": None,
        },
    }
    integration._validate_systemd_sidecar_causality(complete)
    for missing in (
        "claim_valid",
        "claim_matches_invocation",
        "start_ack_valid",
        "child_prespawn_valid",
    ):
        incomplete = dict(complete)
        incomplete[missing] = False
        with pytest.raises(PermissionError, match="causal lineage"):
            integration._validate_systemd_sidecar_causality(incomplete)

    preexisting_claim = deepcopy(base)
    preexisting_claim["claim_valid"] = True
    integration._validate_systemd_sidecar_causality(preexisting_claim)
    matching_claim = deepcopy(preexisting_claim)
    matching_claim["claim_matches_invocation"] = True
    with pytest.raises(
        PermissionError,
        match="ExecCondition failure lineage",
    ):
        integration._validate_systemd_sidecar_causality(matching_claim)


@pytest.mark.parametrize(
    (
        "service_result",
        "exit_code",
        "exit_status",
        "expected_category",
    ),
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
        ("start-limit-hit", "exited", "1", "SYSTEMD_START_LIMIT_HIT"),
        ("exec-condition", "exited", "1", "SYSTEMD_EXEC_CONDITION"),
        ("novel-result", "unknown", "unknown", "SYSTEMD_OTHER_FAILURE"),
        (
            "exec-condition",
            "unknown",
            "unknown",
            "SYSTEMD_EXEC_CONDITION",
        ),
        ("success", "exited", "1", "SYSTEMD_OTHER_FAILURE"),
    ],
)
def test_systemd_outcome_validator_round_trips_supervisor_classifier(
    service_result: str,
    exit_code: str,
    exit_status: str,
    expected_category: str,
) -> None:
    outcome = supervisor.classify_systemd_exit(
        {
            "SERVICE_RESULT": service_result,
            "EXIT_CODE": exit_code,
            "EXIT_STATUS": exit_status,
            "INVOCATION_ID": INVOCATION,
        }
    )
    assert outcome["category"] == expected_category
    assert integration._validate_systemd_outcome(outcome) == outcome

    wrong_category = dict(outcome)
    wrong_category["category"] = "SYSTEMD_SERVICE_SUCCESS"
    if expected_category == "SYSTEMD_SERVICE_SUCCESS":
        wrong_category["category"] = "SYSTEMD_OTHER_FAILURE"
    with pytest.raises(PermissionError, match="internally inconsistent"):
        integration._validate_systemd_outcome(wrong_category)

    wrong_success = dict(outcome)
    wrong_success["systemd_success"] = not outcome["systemd_success"]
    with pytest.raises(PermissionError, match="internally inconsistent"):
        integration._validate_systemd_outcome(wrong_success)


def test_systemd_outcome_validator_rejects_open_or_malformed_schema() -> None:
    valid = supervisor.classify_systemd_exit(
        {
            "SERVICE_RESULT": "success",
            "EXIT_CODE": "exited",
            "EXIT_STATUS": "0",
            "INVOCATION_ID": INVOCATION,
        }
    )
    missing = dict(valid)
    missing.pop("exit_status")
    extra = {**valid, "unauthorized": False}
    malformed_invocation = {**valid, "invocation_id": "A" * 32}
    scientific_claim = {**valid, "scientific_gate_passed": True}
    for malformed in (
        missing,
        extra,
        malformed_invocation,
        scientific_claim,
    ):
        with pytest.raises(PermissionError):
            integration._validate_systemd_outcome(malformed)


def test_immutable_shadow_query_validates_raw_exec_before_normalization(
    tmp_path: Path,
) -> None:
    expected = integration._immutable_shadow(
        fragment_path=tmp_path / "dummy.service",
        scenario_root=tmp_path,
        python_path=Path("/usr/bin/python3.12"),
        supervisor_path=tmp_path / "supervisor.py",
        runtime_spec_path=tmp_path / "runtime-spec.json",
    )

    def query(shadow: dict[str, str]) -> dict[str, str]:
        stdout = "".join(
            f"{name}={shadow[name]}\n"
            for name in sorted(integration._IMMUTABLE_SHADOW_KEYS)
        )
        return integration._query_immutable_shadow(
            "dummy.service",
            runner=lambda argv: _completed(argv, stdout=stdout),
        )

    live = dict(expected)
    for name in integration._EXEC_MODES:
        live[name] = (
            live[name][:-2]
            + " ; start_time=[n/a] ; stop_time=[n/a]"
            + " ; pid=0 ; code=(null) ; status=0/0 }"
        )
    assert query(live) == expected

    for suffix in (
        " ; pid=evil }",
        " ; start_time=garbage }",
        " ; pid=1 ; pid=2 }",
    ):
        malformed = dict(expected)
        malformed["ExecStart"] = malformed["ExecStart"][:-2] + suffix
        with pytest.raises(ValueError, match="runtime fields|ambiguous"):
            query(malformed)
    malformed = dict(expected)
    malformed["ExecStart"] = malformed["ExecStart"].replace(
        "ignore_errors=no",
        "ignore_errors=garbage",
    )
    with pytest.raises(ValueError, match="static identity"):
        query(malformed)


def test_manager_generation_drift_after_terminal_blocks_removal_with_receipt(
    tmp_path: Path,
) -> None:
    scenario_root, runner, authorization = _workspace(tmp_path)
    calls = 0

    def drifting_manager() -> dict[str, object]:
        nonlocal calls
        calls += 1
        value = deepcopy(_manager())
        if calls >= 3:
            value["identity"]["starttime_ticks"] = 999
        return value

    with pytest.raises(PermissionError, match="changed after terminal"):
        integration.run_authorized_integration(
            scenario_root / "control/authorization.json", execute=True,
            runner=runner, manager_reader=drifting_manager,
            timeout_seconds=0.2,
        )
    control = authorization["control_artifacts"]
    terminal = integration._read_sealed(
        Path(str(control["integration_terminal"])),
        fingerprint_field="integration_terminal_fingerprint",
        schema=integration.INTEGRATION_TERMINAL_SCHEMA,
    )
    removal = integration._read_sealed(
        Path(str(control["removal_state"])),
        fingerprint_field="removal_state_fingerprint",
        schema=integration.REMOVAL_STATE_SCHEMA,
    )
    assert terminal["passed"] is True
    assert removal["passed"] is False and removal["remove_attempted"] is False
    assert not Path(str(control["removal_authorization"])).exists()
    fragment = Path(str(authorization["unit_directory"])) / authorization["identity"]["unit_name"]
    assert fragment.exists()


def test_commit_failure_seals_terminal_and_removal_state_without_removing(
    tmp_path: Path,
) -> None:
    scenario_root, runner, authorization = _workspace(tmp_path)
    runner.commit_failure = True
    with pytest.raises(RuntimeError, match="commit-and-start"):
        integration.run_authorized_integration(
            scenario_root / "control/authorization.json", execute=True,
            runner=runner, manager_reader=lambda: deepcopy(_manager()),
            timeout_seconds=0.1,
        )
    control = authorization["control_artifacts"]
    terminal = integration._read_sealed(
        Path(str(control["integration_terminal"])),
        fingerprint_field="integration_terminal_fingerprint",
        schema=integration.INTEGRATION_TERMINAL_SCHEMA,
    )
    removal = integration._read_sealed(
        Path(str(control["removal_state"])),
        fingerprint_field="removal_state_fingerprint",
        schema=integration.REMOVAL_STATE_SCHEMA,
    )
    assert terminal["passed"] is False
    assert terminal["direct_systemctl_start_attempted"] is False
    assert removal["passed"] is False and removal["remove_attempted"] is False
    assert not Path(str(control["removal_authorization"])).exists()
    fragment = Path(str(authorization["unit_directory"])) / authorization["identity"]["unit_name"]
    assert fragment.exists()


def test_full_search_path_shadow_rejected_before_authorization(tmp_path: Path) -> None:
    unit_dir = tmp_path / "run/systemd/user"
    alternate = tmp_path / "config/systemd/user"
    unit_dir.mkdir(parents=True)
    alternate.mkdir(parents=True)
    unit_dir.chmod(0o700)
    alternate.chmod(0o755)
    identity = integration.build_supervisor_v2_identity(SCENARIO)
    (alternate / identity["unit_name"]).write_text("shadow\n", encoding="utf-8")
    runner = FakeRunner(unit_dir.resolve(), alternate.resolve())
    with pytest.raises(PermissionError, match="shadowed"):
        integration.create_production_authorization(
            (tmp_path / "scenario").resolve(), scenario_id=SCENARIO,
            template_path=TEMPLATE.resolve(), python_path=PYTHON,
            supervisor_path=SUPERVISOR.resolve(), realizer_path=REALIZER.resolve(),
            dummy_child_path=DUMMY_CHILD.resolve(),
            instruction_id=integration.INSTRUCTION_ID,
            authorization_basis=integration.AUTHORIZATION_BASIS,
            runner=runner, manager_reader=lambda: deepcopy(_manager()),
        )


def test_no_execute_flag_has_no_command_or_fragment_effect(tmp_path: Path) -> None:
    scenario_root, runner, authorization = _workspace(tmp_path)
    before = len(runner.calls)
    with pytest.raises(PermissionError, match="explicit"):
        integration.run_authorized_integration(
            scenario_root / "control/authorization.json", execute=False,
            runner=runner, manager_reader=lambda: deepcopy(_manager()),
        )
    assert len(runner.calls) == before
    fragment = Path(str(authorization["unit_directory"])) / authorization["identity"]["unit_name"]
    assert not fragment.exists()


def test_dummy_child_is_create_once_sealed_and_payload_free(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    artifact = (tmp_path / "dummy.json").resolve()
    digest = dummy_child.run_dummy_child(
        artifact=artifact, scenario_id=SCENARIO, wait_seconds=0,
    )
    assert digest == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o444
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["dataset_accessed"] is False
    assert payload["gpu_accessed"] is False
    assert payload["torch_imported"] is False
    assert payload["dummy_artifact_fingerprint"] == integration.stable_fingerprint(
        {key: value for key, value in payload.items() if key != "dummy_artifact_fingerprint"}
    )
    with pytest.raises(FileExistsError):
        dummy_child.run_dummy_child(
            artifact=artifact, scenario_id=SCENARIO, wait_seconds=0,
        )


def test_realizer_fragment_is_0600_create_only_and_hardlink_safe(tmp_path: Path) -> None:
    unit_dir = tmp_path / "unit"
    unit_dir.mkdir()
    unit_dir.chmod(0o700)
    unit = f"{realizer.INTEGRATION_UNIT_PREFIX}{SCENARIO}.service"
    text = "[Service]\nType=oneshot\n"
    plan = realizer.build_realization_plan(
        unit_name=unit, unit_directory=unit_dir.resolve(), fragment_text=text,
        expected_fragment_sha256=hashlib.sha256(text.encode()).hexdigest(),
        execute_authorized=True, removal_authorized=True,
    )
    identity = realizer.realize_static_fragment(plan, execute=True)
    assert identity["mode"] == 0o600 and identity["nlink"] == 1
    with pytest.raises(FileExistsError):
        realizer.realize_static_fragment(plan, execute=True)
    os.link(plan.fragment_path, plan.fragment_path.with_suffix(".alias"))
    with pytest.raises(PermissionError, match="regular file"):
        realizer.remove_integration_fragment(plan, execute=True)


def test_isolated_cli_help_does_not_import_supervisor_module() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", str(INTEGRATION_TOOL), "--help"],
        shell=False, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "build-authorization" in completed.stdout
    assert "verify-authorization" in completed.stdout


def test_scenario_identity_is_unique_and_never_actual() -> None:
    first = integration.build_supervisor_v2_identity(SCENARIO)
    second = integration.build_supervisor_v2_identity(
        "fault-case-fedcba9876543210"
    )
    assert first["unit_name"] != second["unit_name"]
    assert first["attempt_id"] != second["attempt_id"]
    assert first["unit_name"] != "cure-lite-v24-gcr-pacre-dr-r2.service"
