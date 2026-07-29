from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tools import cure_lite_v24_runtime_supervisor as supervisor


REPOSITORY = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = (
    REPOSITORY / "tools/cure_lite_v24_runtime_supervisor.py"
)
DUMMY_INVOCATION_ID = "a" * 32


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        supervisor.canonical_json(payload) + "\n",
        encoding="utf-8",
    )


def _dummy_spec(
    tmp_path: Path,
    argv: list[str],
    *,
    actual: bool = False,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "runtime"
    root.mkdir()
    heartbeat = root / "heartbeat"
    heartbeat.mkdir()
    systemd_invocations = root / "systemd-invocations"
    systemd_invocations.mkdir()
    execution_kind = (
        supervisor.ACTUAL_EXECUTION_KIND
        if actual
        else supervisor.DUMMY_EXECUTION_KIND
    )
    body: dict[str, object] = {
        "schema_version": supervisor.RUNTIME_SPEC_SCHEMA,
        "execution_kind": execution_kind,
        "candidate": "GCR-PACRE-v24" if actual else "generated-dummy",
        "stage_id": (
            "gcr_pacre_v24_D_R_structural_r2"
            if actual
            else "generated_dummy_runtime"
        ),
        "attempt_id": (
            "gcr_pacre_v24_D_R_zero_update_structural_r2"
            if actual
            else "generated_dummy_attempt"
        ),
        "attempt_ordinal": 2 if actual else 0,
        "prior_attempt_count": 1 if actual else 0,
        "authorization": (
            {
                "path": str((tmp_path / "absent-r2-authorization.json").resolve()),
                "required_schema": (
                    "cure-lite-v24-D_R-structural-r2-authorization-v1"
                ),
            }
            if actual
            else None
        ),
        "child": {
            "argv": argv,
            "argv_fingerprint": supervisor.stable_fingerprint(argv),
            "cwd": str(tmp_path.resolve()),
            "environment": {},
            "inherit_environment": [],
            "entrypoint_path": (
                str(SUPERVISOR_PATH) if actual else None
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
                "unit_name": "cure-lite-v24-generated-dummy.service",
                "service_type": "exec",
                "kill_mode": "mixed",
                "send_sigkill": True,
                "timeout_stop_seconds": 1.0,
                "unit_fragment_file_sha256": (
                    _sha256(SUPERVISOR_PATH) if actual else None
                ),
                "shadow_properties": {
                    "Type": "exec",
                    "Restart": "no",
                    "NRestarts": "0",
                    "KillMode": "mixed",
                    "SendSIGKILL": "yes",
                    "TimeoutStopUSec": "1s",
                    "FragmentPath": str(SUPERVISOR_PATH),
                    "DropInPaths": "",
                    "Transient": "no",
                    "NeedDaemonReload": "no",
                    "Environment": "PYTHONUNBUFFERED=1",
                    "UnsetEnvironment": "",
                    "WorkingDirectory": str(tmp_path.resolve()),
                    "UMask": "0077",
                    "ExitType": "main",
                    "RuntimeMaxUSec": "infinity",
                    "WatchdogUSec": "0",
                    "OOMPolicy": "kill",
                    "RemainAfterExit": "no",
                    "StandardInput": "null",
                    "StandardOutput": "journal",
                    "StandardError": "journal",
                    "StartLimitIntervalUSec": "infinity",
                    "StartLimitBurst": "1",
                    "KillSignal": "15",
                    "ExecCondition": (
                        f"{sys.executable} -I {SUPERVISOR_PATH} "
                        "claim-materialization --spec "
                        f"{tmp_path / 'runtime-spec.json'}"
                    ),
                    "ExecStartPre": (
                        f"{sys.executable} -I {SUPERVISOR_PATH} "
                        "verify-runtime-spec --spec "
                        f"{tmp_path / 'runtime-spec.json'}"
                    ),
                    "ExecStart": (
                        f"{sys.executable} -I {SUPERVISOR_PATH} "
                        f"run-once --spec {tmp_path / 'runtime-spec.json'}"
                    ),
                    "ExecStopPost": (
                        f"{sys.executable} -I {SUPERVISOR_PATH} "
                        "record-systemd-exit --spec "
                        f"{tmp_path / 'runtime-spec.json'}"
                    ),
                },
                "shadow_fingerprint": "TBD",
            },
        },
        "source_bindings": {
            "supervisor_file_sha256": _sha256(SUPERVISOR_PATH),
            "child_entry_file_sha256": (
                _sha256(SUPERVISOR_PATH) if actual else None
            ),
            "prior_attempt_receipt_file_sha256": (
                "1" * 64 if actual else None
            ),
        },
    }
    systemd = body["runtime"]["systemd"]
    systemd["shadow_fingerprint"] = supervisor.stable_fingerprint(
        systemd["shadow_properties"]
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


def _authorize_actual_spec(
    spec_path: Path,
    spec: dict[str, object],
) -> dict[str, object]:
    reference = spec["authorization"]
    assert isinstance(reference, dict)
    body = {
        "schema_version": reference["required_schema"],
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
    shadow = systemd["shadow_properties"]
    assert isinstance(shadow, dict)
    commit = supervisor._attempt_commit_payload(
        spec,
        authorization,
        shadow,
    )
    artifacts = spec["artifacts"]
    assert isinstance(artifacts, dict)
    supervisor._write_new_json(
        Path(str(artifacts["attempt_commit"])),
        commit,
    )
    return authorization, commit


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
            "datetime",
            "hashlib",
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
    for key in (
        "attempt_commit",
        "materialization_claim",
        "stdout_log",
        "stderr_log",
        "runtime_terminal",
    ):
        assert not Path(str(artifacts[key])).exists()
    assert list(Path(str(artifacts["heartbeat_dir"])).iterdir()) == []


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
    observed = dict(systemd["shadow_properties"])
    observed["Restart"] = "on-failure"
    with pytest.raises(PermissionError):
        supervisor.validate_systemd_shadow(spec, observed)


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
        lambda _unit: dict(systemd["shadow_properties"]),
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
    _authorization, _commit = _commit_actual_spec(spec_path, spec)
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
    assert sidecar["audit_valid"] is False


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
        lambda _unit: dict(systemd["shadow_properties"]),
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
