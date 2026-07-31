from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests_v24 import (
    test_gcr_pacre_v24_dr_r2_runtime_supervisor as frozen_fixtures,
)
from tools import cure_lite_v24_preaccess_schema_compatibility as policy
from tools import cure_lite_v24_runtime_supervisor as frozen_supervisor
from tools import (
    cure_lite_v24_runtime_supervisor_preaccess_compat_c1 as supervisor,
)
from tools import (
    run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c1
    as adapter,
)


REPOSITORY = Path(__file__).resolve().parents[1]
LEGACY_GATE = (
    REPOSITORY / "tools/run_cure_lite_v24_gcr_pacre_dr_gate.py"
).resolve()
SCIENTIFIC_AUTHORIZATION = Path(
    frozen_supervisor._ACTUAL_SCIENTIFIC_AUTHORIZATION_PATH
)
SCIENTIFIC_ACCESS_AUDIT = Path(
    frozen_supervisor._ACTUAL_SCIENTIFIC_ACCESS_AUDIT_PATH
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal_spec(payload: dict[str, object]) -> None:
    systemd = payload["runtime"]["systemd"]
    systemd["immutable_shadow_fingerprint"] = (
        supervisor.legacy.stable_fingerprint(
            systemd["immutable_shadow_properties"]
        )
    )
    body = dict(payload)
    body.pop("runtime_spec_fingerprint", None)
    payload["runtime_spec_fingerprint"] = (
        supervisor.legacy.stable_fingerprint(body)
    )


def _actual_v1_spec(tmp_path: Path) -> dict[str, object]:
    _path, payload = frozen_fixtures._dummy_spec(
        tmp_path,
        [],
        actual=True,
    )
    scientific_authorization = json.loads(
        SCIENTIFIC_AUTHORIZATION.read_text(encoding="utf-8")
    )
    scientific_audit = json.loads(
        SCIENTIFIC_ACCESS_AUDIT.read_text(encoding="utf-8")
    )
    payload["authorization"] = {
        "path": supervisor.COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        "required_schema": (
            supervisor.legacy.RUNTIME_LAUNCH_AUTHORIZATION_SCHEMA
        ),
    }
    payload["scientific_preaccess"] = {
        "authorization_path": str(SCIENTIFIC_AUTHORIZATION),
        "authorization_file_sha256": _sha256(
            SCIENTIFIC_AUTHORIZATION
        ),
        "authorization_fingerprint": scientific_authorization[
            "authorization_fingerprint"
        ],
        "authorization_required_schema": scientific_authorization[
            "schema_version"
        ],
        "access_audit_path": str(SCIENTIFIC_ACCESS_AUDIT),
        "access_audit_file_sha256": _sha256(
            SCIENTIFIC_ACCESS_AUDIT
        ),
        "access_audit_fingerprint": scientific_audit[
            "receipt_fingerprint"
        ],
        "access_audit_required_schema": (
            supervisor.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        ),
        "source_closure_fingerprint_103": (
            frozen_supervisor._ACTUAL_SOURCE_CLOSURE_FINGERPRINT_103
        ),
    }
    adapter_path = Path(supervisor.COMPAT_ADAPTER_PATH)
    supervisor_path = Path(supervisor.COMPAT_SUPERVISOR_PATH)
    child_argv = [
        frozen_supervisor._ACTUAL_PYTHON_PATH,
        "-I",
        "-S",
        "-B",
        "-u",
        str(adapter_path),
        "real",
        "--execute-real-dr",
        "--device",
        "cuda:0",
        "--runtime-launch-authorization",
        supervisor.COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
    ]
    payload["child"]["argv"] = child_argv
    payload["child"]["argv_fingerprint"] = (
        supervisor.legacy.stable_fingerprint(child_argv)
    )
    payload["child"]["entrypoint_path"] = str(adapter_path)
    bindings = payload["source_bindings"]
    bindings.update(
        {
            "supervisor_file_sha256": _sha256(supervisor_path),
            "child_entry_file_sha256": _sha256(adapter_path),
            "r2_adapter_path": str(adapter_path),
            "r2_adapter_file_sha256": _sha256(adapter_path),
            "legacy_gate_entrypoint_path": str(LEGACY_GATE),
            "legacy_gate_entrypoint_file_sha256": _sha256(LEGACY_GATE),
        }
    )
    systemd = payload["runtime"]["systemd"]
    systemd["unit_name"] = supervisor.COMPAT_UNIT_NAME
    shadow = systemd["immutable_shadow_properties"]
    modes = {
        "ExecCondition": "claim-materialization",
        "ExecStartPre": "verify-runtime-spec",
        "ExecStart": "run-once",
        "ExecStopPost": "record-systemd-exit",
    }
    for directive, mode in modes.items():
        argv = (
            f"{frozen_supervisor._ACTUAL_PYTHON_PATH} -I -S -B -u "
            f"{supervisor_path} {mode} --spec "
            f"{supervisor.COMPAT_RUNTIME_SPEC_PATH}"
        )
        shadow[directive] = (
            f"{{ path={frozen_supervisor._ACTUAL_PYTHON_PATH} ; "
            f"argv[]={argv} ; ignore_errors=no }}"
        )
    _reseal_spec(payload)
    return payload


def _accepted_bridge() -> dict[str, object]:
    schema = {
        "producer_schema": supervisor.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA,
        "scientific_authorization_bound_schema": (
            supervisor.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        ),
        "compatibility_consumer_required_schema": (
            supervisor.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        ),
        "buggy_frozen_consumer_expected_schema": (
            supervisor.FICTIONAL_ACCESS_AUDIT_SCHEMA
        ),
        "accept_either_schema": False,
    }
    return {
        "schema_compatibility": schema,
        "compatibility_source_roots": {
            "compat_supervisor": {
                "path": supervisor.COMPAT_SUPERVISOR_PATH,
                "file_sha256": _sha256(
                    Path(supervisor.COMPAT_SUPERVISOR_PATH)
                ),
            },
            "compat_adapter": {
                "path": supervisor.COMPAT_ADAPTER_PATH,
                "file_sha256": _sha256(
                    Path(supervisor.COMPAT_ADAPTER_PATH)
                ),
            },
        },
    }


def test_actual_original_v1_passes_without_mutating_the_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _actual_v1_spec(tmp_path)
    original = deepcopy(payload)
    monkeypatch.setattr(
        supervisor,
        "_verify_policy_compatibility_receipt",
        lambda _payload: _accepted_bridge(),
    )

    supervisor._validate_spec_structure(
        payload,
        loaded_spec_path=Path(supervisor.COMPAT_RUNTIME_SPEC_PATH),
    )

    assert payload == original
    assert payload["scientific_preaccess"][
        "access_audit_required_schema"
    ] == supervisor.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA


def test_actual_fictional_r2_v1_schema_is_rejected_before_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _actual_v1_spec(tmp_path)
    payload["scientific_preaccess"][
        "access_audit_required_schema"
    ] = supervisor.FICTIONAL_ACCESS_AUDIT_SCHEMA
    _reseal_spec(payload)
    bridge_called = False

    def forbidden_bridge(_payload: object) -> dict[str, object]:
        nonlocal bridge_called
        bridge_called = True
        return _accepted_bridge()

    monkeypatch.setattr(
        supervisor,
        "_verify_policy_compatibility_receipt",
        forbidden_bridge,
    )
    with pytest.raises(ValueError, match="original v1"):
        supervisor._validate_spec_structure(
            payload,
            loaded_spec_path=Path(
                supervisor.COMPAT_RUNTIME_SPEC_PATH
            ),
        )
    assert bridge_called is False


def test_actual_spec_is_rejected_from_every_off_path_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _actual_v1_spec(tmp_path)
    bridge_called = False

    def forbidden_bridge(_payload: object) -> dict[str, object]:
        nonlocal bridge_called
        bridge_called = True
        return _accepted_bridge()

    monkeypatch.setattr(
        supervisor,
        "_verify_policy_compatibility_receipt",
        forbidden_bridge,
    )
    with pytest.raises(ValueError, match="spec path is not exact"):
        supervisor._validate_spec_structure(
            payload,
            loaded_spec_path=(tmp_path / "copied-runtime-spec.json").resolve(),
        )
    assert bridge_called is False


@pytest.mark.parametrize("receipt_kind", ["missing", "drift"])
def test_policy_bridge_missing_or_drift_fails_closed(
    receipt_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = (tmp_path / "r2_preaccess_schema_compat_c1_receipt.json").resolve()
    if receipt_kind == "drift":
        receipt.write_text("{}\n", encoding="utf-8")
        receipt.chmod(0o444)
    monkeypatch.setattr(
        supervisor,
        "COMPATIBILITY_RECEIPT_PATH",
        str(receipt),
    )
    monkeypatch.setattr(policy, "COMPAT_RECEIPT_PATH", receipt)

    with pytest.raises((OSError, PermissionError, ValueError)):
        supervisor._verify_policy_compatibility_receipt({})


def test_frozen_adapter_rejects_the_new_compat_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _actual_v1_spec(tmp_path)
    monkeypatch.setattr(
        supervisor,
        "_verify_policy_compatibility_receipt",
        lambda _payload: _accepted_bridge(),
    )
    supervisor._validate_spec_structure(
        payload,
        loaded_spec_path=Path(supervisor.COMPAT_RUNTIME_SPEC_PATH),
    )
    with pytest.raises(ValueError):
        frozen_supervisor._validate_spec_structure(
            payload,
            loaded_spec_path=Path(supervisor.COMPAT_RUNTIME_SPEC_PATH),
        )


def test_compat_adapter_calls_only_the_new_attestation_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    attestation = "/tmp/compat-c1-attestation.json"
    authorization = "/tmp/compat-c1-launch-authorization.json"
    monkeypatch.setenv(
        adapter.legacy._RUNTIME_ATTESTATION_ENV,
        attestation,
    )

    def new_verifier(
        attestation_path: str,
        authorization_path: str,
    ) -> dict[str, object]:
        calls.append((attestation_path, authorization_path))
        return {"runtime_attestation_valid": True}

    def old_verifier(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("frozen supervisor verifier was called")

    monkeypatch.setattr(
        supervisor,
        "verify_child_runtime_attestation",
        new_verifier,
    )
    monkeypatch.setattr(
        frozen_supervisor,
        "verify_child_runtime_attestation",
        old_verifier,
    )

    result = adapter._verify_runtime_launch(authorization)

    assert result["runtime_attestation_valid"] is True
    assert calls == [(attestation, authorization)]


@pytest.mark.parametrize(
    ("module", "path_constant", "verify_name"),
    [
        (
            supervisor,
            "FROZEN_SUPERVISOR_PATH",
            "_verify_frozen_supervisor_source",
        ),
        (
            adapter,
            "FROZEN_ADAPTER_PATH",
            "_verify_frozen_adapter_source",
        ),
    ],
)
def test_frozen_source_hash_drift_fails_closed(
    module: object,
    path_constant: str,
    verify_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drift = (tmp_path / f"{path_constant}.py").resolve()
    drift.write_text("# drift\n", encoding="utf-8")
    monkeypatch.setattr(module, path_constant, drift)
    with pytest.raises(PermissionError, match="source changed"):
        getattr(module, verify_name)()


def test_identity_summary_reports_new_adapter_and_preserves_r2_science() -> None:
    command = [
        "/usr/bin/python3.12",
        "-I",
        "-S",
        "-B",
        "-u",
        adapter.COMPAT_ADAPTER_PATH,
        "--r2-execution-identity-summary",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    summary = json.loads(completed.stdout)
    transition = summary["identity_transition"]
    assert summary["adapter_repo_path"] == (
        "tools/"
        "run_cure_lite_v24_gcr_pacre_dr_gate_r2_"
        "preaccess_compat_c1.py"
    )
    assert transition["GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA"]["r2"] == (
        supervisor.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
    )
    assert transition["GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH"]["r2"] == (
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_authorization.json"
    )
    assert transition["GCR_PACRE_DR_RECEIPT_PATH"]["r2"] == (
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_receipt.json"
    )


def test_isolated_supervisor_can_import_the_fixed_policy_module() -> None:
    source = Path(supervisor.COMPAT_SUPERVISOR_PATH)
    code = (
        "import pathlib,runpy;"
        f"ns=runpy.run_path({str(source)!r});"
        "policy=__import__("
        "'tools.cure_lite_v24_preaccess_schema_compatibility',"
        "fromlist=['COMPATIBILITY_RECEIPT_PATH']);"
        f"assert pathlib.Path(policy.__file__).resolve()=="
        f"pathlib.Path({str(policy.__file__)!r}).resolve();"
        "assert str(pathlib.Path(ns['REPOSITORY'])) in __import__('sys').path"
    )
    completed = subprocess.run(
        [
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            "-c",
            code,
        ],
        cwd="/",
        check=False,
        capture_output=True,
        text=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr


def test_systemd_dummy_structure_uses_final_compat_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor._configure_frozen_supervisor()
    monkeypatch.setattr(
        frozen_fixtures,
        "supervisor",
        supervisor.legacy,
    )
    monkeypatch.setattr(
        frozen_fixtures,
        "SUPERVISOR_PATH",
        Path(supervisor.COMPAT_SUPERVISOR_PATH),
    )
    path, payload = frozen_fixtures._dummy_spec(
        tmp_path,
        [],
        integration=True,
    )

    supervisor._validate_spec_structure(
        payload,
        loaded_spec_path=path.resolve(),
    )
    exec_start = payload["runtime"]["systemd"][
        "immutable_shadow_properties"
    ]["ExecStart"]
    assert supervisor.COMPAT_SUPERVISOR_PATH in exec_start
    assert not os.path.lexists(
        tmp_path / "r2_preaccess_schema_compat_c1_receipt.json"
    )
