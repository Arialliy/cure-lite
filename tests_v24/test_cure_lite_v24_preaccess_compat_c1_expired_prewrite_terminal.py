from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO
    / "tools/cure_lite_v24_preaccess_compat_c1_expired_prewrite_terminal.py"
)
SPEC = importlib.util.spec_from_file_location("expired_c1_terminal", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
terminal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(terminal)

AFTER_EXPIRY = datetime(2026, 7, 30, 6, 30, tzinfo=timezone.utc)
BEFORE_EXPIRY = datetime(2026, 7, 30, 6, 27, tzinfo=timezone.utc)


def _seal(
    path: Path,
    body: dict[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    payload = dict(body)
    payload[fingerprint_field] = terminal.stable_fingerprint(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(terminal.canonical_json(payload) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return payload


def _live(fragment: Path) -> dict[str, str]:
    return {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "FragmentPath": str(fragment),
        "DropInPaths": "",
        "NeedDaemonReload": "no",
        "Transient": "no",
        "Restart": "no",
        "NRestarts": "0",
        "InvocationID": "",
        "ExecMainPID": "0",
        "ExecMainCode": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "StateChangeTimestamp": "",
        "ActiveEnterTimestamp": "",
        "InactiveEnterTimestamp": "",
        "ExecMainStartTimestamp": "",
        "ExecMainExitTimestamp": "",
    }


def _write_session(path: Path) -> tuple[bytes, bytes]:
    arguments = '{"cmd":"create-policy --require-target-ready"}'
    output_text = (
        "Process exited with code 1\n"
        "PermissionError: precleanup inventory unit scope changed\n"
    )
    call = {
        "timestamp": terminal.CALL_TIMESTAMP_UTC,
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "exec_command",
            "arguments": arguments,
            "call_id": terminal.CALL_ID,
        },
    }
    output = {
        "timestamp": terminal.OUTPUT_TIMESTAMP_UTC,
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "call_id": terminal.CALL_ID,
            "output": output_text,
        },
    }
    call_raw = terminal.canonical_json(call).encode()
    output_raw = terminal.canonical_json(output).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(call_raw + b"\n" + output_raw + b"\n")
    return call_raw, output_raw


@pytest.fixture()
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    repo = tmp_path / "repo"
    evidence = repo / "evidence"
    runs = repo / "runs"
    evidence.mkdir(parents=True)
    runs.mkdir(parents=True)
    terminal_path = evidence / "expired-terminal.json"
    fragment = repo / "runtime" / terminal.UNIT_NAME
    fragment.parent.mkdir(parents=True)
    fragment.write_text("static fragment\n", encoding="utf-8")
    fragment.chmod(0o600)
    fragment_sha = hashlib.sha256(fragment.read_bytes()).hexdigest()

    sources: dict[str, tuple[Path, str]] = {}
    for name in terminal.SOURCE_BINDINGS:
        path = repo / "sources" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        sources[name] = (path, hashlib.sha256(path.read_bytes()).hexdigest())

    absence_paths = {
        name: (
            runs / path.name
            if name in {"scientific_run_root", "compat_run_root_alias"}
            else evidence / path.name
        )
        for name, path in terminal.ABSENCE_PATHS.items()
    }

    forensic_path = evidence / "forensic.json"
    bridge_path = evidence / "bridge-auth.json"
    unit_auth_path = evidence / "unit-auth.json"
    unit_receipt_path = evidence / "unit-receipt.json"
    r10_auth_path = evidence / "r10-auth.json"
    r10_receipt_path = evidence / "r10-receipt.json"
    science_auth_path = evidence / "science-auth.json"
    audit_path = evidence / "audit.json"

    forensic = _seal(
        forensic_path,
        {
            "schema_version": terminal.EVIDENCE_BINDINGS[
                "forensic_terminal"
            ][2],
            "forensic_closure_passed": True,
            "payload_observation": {
                "D_R_payload_accessed": False,
                "D_T_payload_accessed": False,
                "D_V_payload_accessed": False,
                "attempt_commit_present": False,
                "materialization_claim_present": False,
                "scientific_decision_present": False,
                "training_started": False,
            },
        },
        fingerprint_field="terminal_fingerprint",
    )
    del forensic
    bridge = _seal(
        bridge_path,
        {
            "schema_version": terminal.EVIDENCE_BINDINGS[
                "bridge_authorization"
            ][2],
            "created_at_utc": "2026-07-30T06:23:07.622325Z",
            "issued_at_utc": "2026-07-30T06:23:07.622325Z",
            "expires_at_utc": "2026-07-30T06:28:07.622325Z",
            "instruction_id": terminal.INSTRUCTION_ID,
            "authorization_basis": terminal.AUTHORIZATION_BASIS,
            "compat_lane_authority": {
                "compat_start_authorized": False,
                "compat_enable_authorized": False,
                "payload_access_authorized": False,
                "runtime_spec_creation_authorized_by_this_receipt": False,
                "runtime_launch_authorization_authorized_by_this_receipt": False,
            },
            "scientific_authority": {
                "automatic_retry": False,
                "resume": False,
                "training_authorized": False,
                "D_V_payload_authorized": False,
                "D_T_payload_authorized": False,
            },
        },
        fingerprint_field="authorization_fingerprint",
    )
    unit_auth = _seal(
        unit_auth_path,
        {
            "schema_version": terminal.EVIDENCE_BINDINGS[
                "unit_authorization"
            ][2],
            "created_at_utc": "2026-07-30T06:23:25.025653Z",
            "issued_at_utc": "2026-07-30T06:23:25.025636Z",
            "expires_at_utc": "2026-07-30T06:28:25.025636Z",
            "instruction_id": terminal.INSTRUCTION_ID,
            "authorization_basis": terminal.AUTHORIZATION_BASIS,
            "unit_name": terminal.UNIT_NAME,
        },
        fingerprint_field="authorization_fingerprint",
    )
    fragment_stat = fragment.stat()
    receipt = _seal(
        unit_receipt_path,
        {
            "schema_version": terminal.EVIDENCE_BINDINGS["unit_receipt"][2],
            "created_at_utc": "2026-07-30T06:23:55.817536Z",
            "authorization_fingerprint": unit_auth[
                "authorization_fingerprint"
            ],
            "unit_name": terminal.UNIT_NAME,
            "passed": True,
            "static": True,
            "enabled": False,
            "started": False,
            "runtime_spec_absent_at_receipt": True,
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "fragment_identity": {
                "path": str(fragment),
                "file_sha256": fragment_sha,
                "device": fragment_stat.st_dev,
                "inode": fragment_stat.st_ino,
                "owner_uid": fragment_stat.st_uid,
                "mode": stat.S_IMODE(fragment_stat.st_mode),
                "nlink": fragment_stat.st_nlink,
            },
            "full_static_shadow": {
                key: value
                for key, value in _live(fragment).items()
                if key
                in {
                    "LoadState",
                    "ActiveState",
                    "SubState",
                    "UnitFileState",
                    "FragmentPath",
                    "NeedDaemonReload",
                    "Transient",
                    "Restart",
                    "NRestarts",
                }
            },
        },
        fingerprint_field="receipt_fingerprint",
    )
    del receipt
    r10_auth = _seal(
        r10_auth_path,
        {
            "schema_version": terminal.EVIDENCE_BINDINGS[
                "r10_authorization"
            ][2],
            "scenario_id": "dummy-c1-r10",
        },
        fingerprint_field="authorization_fingerprint",
    )
    _seal(
        r10_receipt_path,
        {
            "schema_version": terminal.EVIDENCE_BINDINGS["r10_receipt"][2],
            "authorization_fingerprint": r10_auth[
                "authorization_fingerprint"
            ],
            "passed": True,
            "fragment_removed": True,
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_accessed": False,
            "post_removal_unit_state": {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "FragmentPath": "",
                "NRestarts": "0",
            },
        },
        fingerprint_field="receipt_fingerprint",
    )
    _seal(
        science_auth_path,
        {
            "schema_version": terminal.EVIDENCE_BINDINGS[
                "scientific_authorization"
            ][2],
            "D_R_payload_authorized": True,
            "D_V_payload_authorized": False,
            "D_T_payload_authorized": False,
            "training_authorized": False,
        },
        fingerprint_field="authorization_fingerprint",
    )
    _seal(
        audit_path,
        {
            "schema_version": terminal.EVIDENCE_BINDINGS[
                "scientific_access_audit"
            ][2],
            "observed_payloads": [],
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        },
        fingerprint_field="receipt_fingerprint",
    )

    bindings = {
        "forensic_terminal": (
            forensic_path,
            "terminal_fingerprint",
            terminal.EVIDENCE_BINDINGS["forensic_terminal"][2],
        ),
        "bridge_authorization": (
            bridge_path,
            "authorization_fingerprint",
            terminal.EVIDENCE_BINDINGS["bridge_authorization"][2],
        ),
        "unit_authorization": (
            unit_auth_path,
            "authorization_fingerprint",
            terminal.EVIDENCE_BINDINGS["unit_authorization"][2],
        ),
        "unit_receipt": (
            unit_receipt_path,
            "receipt_fingerprint",
            terminal.EVIDENCE_BINDINGS["unit_receipt"][2],
        ),
        "r10_authorization": (
            r10_auth_path,
            "authorization_fingerprint",
            terminal.EVIDENCE_BINDINGS["r10_authorization"][2],
        ),
        "r10_receipt": (
            r10_receipt_path,
            "receipt_fingerprint",
            terminal.EVIDENCE_BINDINGS["r10_receipt"][2],
        ),
        "scientific_authorization": (
            science_auth_path,
            "authorization_fingerprint",
            terminal.EVIDENCE_BINDINGS["scientific_authorization"][2],
        ),
        "scientific_access_audit": (
            audit_path,
            "receipt_fingerprint",
            terminal.EVIDENCE_BINDINGS["scientific_access_audit"][2],
        ),
    }

    session = repo / "session.jsonl"
    call_raw, output_raw = _write_session(session)
    prefix = session.read_bytes()
    session_stat = session.stat()

    monkeypatch.setattr(terminal, "TERMINAL_PATH", terminal_path)
    monkeypatch.setattr(terminal, "FRAGMENT_PATH", fragment)
    monkeypatch.setattr(terminal, "EXPECTED_FRAGMENT_SHA256", fragment_sha)
    monkeypatch.setattr(terminal, "SOURCE_BINDINGS", sources)
    monkeypatch.setattr(terminal, "ABSENCE_PATHS", absence_paths)
    monkeypatch.setattr(terminal, "EVIDENCE_BINDINGS", bindings)
    monkeypatch.setattr(terminal, "SESSION_PATH", session)
    monkeypatch.setattr(terminal, "SESSION_CALL_LINE", 1)
    monkeypatch.setattr(terminal, "SESSION_OUTPUT_LINE", 2)
    monkeypatch.setattr(terminal, "SESSION_PREFIX_SIZE", len(prefix))
    monkeypatch.setattr(
        terminal,
        "SESSION_PREFIX_SHA256",
        hashlib.sha256(prefix).hexdigest(),
    )
    monkeypatch.setattr(
        terminal, "SESSION_EXPECTED_DEVICE", session_stat.st_dev
    )
    monkeypatch.setattr(
        terminal, "SESSION_EXPECTED_INODE", session_stat.st_ino
    )
    monkeypatch.setattr(
        terminal, "SESSION_EXPECTED_UID", session_stat.st_uid
    )
    monkeypatch.setattr(
        terminal, "SESSION_EXPECTED_GID", session_stat.st_gid
    )
    monkeypatch.setattr(
        terminal,
        "SESSION_EXPECTED_MODE",
        stat.S_IMODE(session_stat.st_mode),
    )
    monkeypatch.setattr(
        terminal, "CALL_RAW_SHA256", hashlib.sha256(call_raw).hexdigest()
    )
    monkeypatch.setattr(
        terminal,
        "CALL_RECORD_WITH_LF_SHA256",
        hashlib.sha256(call_raw + b"\n").hexdigest(),
    )
    monkeypatch.setattr(
        terminal,
        "OUTPUT_RAW_SHA256",
        hashlib.sha256(output_raw).hexdigest(),
    )
    monkeypatch.setattr(
        terminal,
        "OUTPUT_RECORD_WITH_LF_SHA256",
        hashlib.sha256(output_raw + b"\n").hexdigest(),
    )
    monkeypatch.setattr(
        terminal,
        "CALL_ARGUMENTS_SHA256",
        hashlib.sha256(
            '{"cmd":"create-policy --require-target-ready"}'.encode()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        terminal,
        "OUTPUT_PAYLOAD_SHA256",
        hashlib.sha256(
            (
                "Process exited with code 1\n"
                "PermissionError: precleanup inventory unit scope changed\n"
            ).encode()
        ).hexdigest(),
    )

    return {
        "terminal": terminal_path,
        "fragment": fragment,
        "sources": sources,
        "session": session,
        "live": _live(fragment),
        "absence_paths": absence_paths,
    }


def _create(case: dict[str, object]) -> dict[str, object]:
    return terminal.create_terminal(
        terminal_path=case["terminal"],
        unit_state_reader=lambda: case["live"],
        now=lambda: AFTER_EXPIRY,
    )


def _validate(case: dict[str, object]) -> dict[str, object]:
    return terminal.validate_terminal(
        terminal_path=case["terminal"],
        unit_state_reader=lambda: case["live"],
        now=lambda: AFTER_EXPIRY,
    )


def test_create_once_and_validate_exact_terminal(
    case: dict[str, object],
) -> None:
    payload = _create(case)
    assert payload["schema_version"] == terminal.SCHEMA
    assert payload["continuation_policy"] == terminal._CONTINUATION_POLICY
    assert payload["payload_observation"] == terminal._PAYLOAD_OBSERVATION
    assert payload["outcome"]["scientific_attempt_consumed"] is False
    assert stat.S_IMODE(case["terminal"].stat().st_mode) == 0o444
    assert _validate(case) == payload
    with pytest.raises(FileExistsError):
        _create(case)


def test_unexpired_bridge_authorization_is_rejected(
    case: dict[str, object],
) -> None:
    with pytest.raises(PermissionError, match="chronology"):
        terminal.create_terminal(
            terminal_path=case["terminal"],
            unit_state_reader=lambda: case["live"],
            now=lambda: BEFORE_EXPIRY,
        )
    assert not case["terminal"].exists()


def test_future_runtime_path_appearing_is_rejected(
    case: dict[str, object],
) -> None:
    future = case["absence_paths"]["compat_runtime_spec"]
    future.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="required absent path exists"):
        _create(case)
    assert not case["terminal"].exists()


def test_future_runtime_path_after_seal_breaks_validation(
    case: dict[str, object],
) -> None:
    _create(case)
    future = case["absence_paths"]["compat_runtime_launch_authorization"]
    future.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="required absent path exists"):
        _validate(case)


def test_source_drift_between_observations_is_rejected(
    case: dict[str, object],
) -> None:
    source = case["sources"]["compat_release"][0]

    def drift() -> None:
        source.write_text("changed\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="source hash changed"):
        terminal.create_terminal(
            terminal_path=case["terminal"],
            unit_state_reader=lambda: case["live"],
            now=lambda: AFTER_EXPIRY,
            between_observations=drift,
        )


def test_session_record_drift_is_rejected(case: dict[str, object]) -> None:
    session = case["session"]
    session.write_bytes(
        session.read_bytes().replace(b"create-policy", b"create-policx")
    )
    with pytest.raises(PermissionError, match="session fixed prefix changed"):
        _create(case)


def test_live_unit_drift_is_rejected(case: dict[str, object]) -> None:
    changed = dict(case["live"])
    changed["InvocationID"] = "unexpected"
    with pytest.raises(PermissionError, match="never-started static"):
        terminal.create_terminal(
            terminal_path=case["terminal"],
            unit_state_reader=lambda: changed,
            now=lambda: AFTER_EXPIRY,
        )


def test_extra_terminal_key_is_rejected(case: dict[str, object]) -> None:
    payload = _create(case)
    path = case["terminal"]
    path.chmod(0o600)
    payload["unexpected"] = True
    body = dict(payload)
    body.pop("terminal_fingerprint")
    payload["terminal_fingerprint"] = terminal.stable_fingerprint(body)
    path.write_text(terminal.canonical_json(payload) + "\n", encoding="utf-8")
    path.chmod(0o444)
    with pytest.raises(PermissionError, match="exact keys"):
        _validate(case)


def test_fixed_terminal_path_is_enforced(
    case: dict[str, object], tmp_path: Path
) -> None:
    with pytest.raises(PermissionError, match="not fixed"):
        terminal.create_terminal(
            terminal_path=tmp_path / "other.json",
            unit_state_reader=lambda: case["live"],
            now=lambda: AFTER_EXPIRY,
        )


def test_cli_surface_has_only_create_and_validate() -> None:
    help_text = terminal.build_parser().format_help()
    assert "create-terminal" in help_text
    assert "validate-terminal" in help_text
