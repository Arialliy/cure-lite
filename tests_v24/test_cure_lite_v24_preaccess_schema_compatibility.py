from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
from typing import Callable

import pytest

from tools import cure_lite_v24_preaccess_schema_compatibility as compat


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/usr/bin/python3.12")
FIXED_NOW = datetime(2026, 7, 30, 8, 0, 0, tzinfo=timezone.utc)


def _write_regular(
    path: Path,
    raw: bytes,
    *,
    mode: int = 0o664,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def _seal(
    path: Path,
    body: dict[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    payload = deepcopy(body)
    assert fingerprint_field not in payload
    payload[fingerprint_field] = compat.stable_fingerprint(payload)
    _write_regular(
        path,
        (compat.canonical_json(payload) + "\n").encode("utf-8"),
        mode=0o444,
    )
    return payload


def _replace_sealed(
    path: Path,
    payload: dict[str, object],
    *,
    fingerprint_field: str,
    mutate: Callable[[dict[str, object]], None],
) -> dict[str, object]:
    body = deepcopy(payload)
    body.pop(fingerprint_field)
    mutate(body)
    path.chmod(0o600)
    path.unlink()
    return _seal(path, body, fingerprint_field=fingerprint_field)


def _scientific_audit_body() -> dict[str, object]:
    return {
        "schema_version": compat.SCIENTIFIC_ACCESS_AUDIT_SCHEMA,
        "stage_id": compat.STAGE_ID,
        "allowed_splits": ["D_R"],
        "observed_payloads": [],
        "event_log_fingerprint": "1" * 64,
        "source_manifest_fingerprint": "2" * 64,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def _scientific_authorization_body(
    audit_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": compat.SCIENTIFIC_AUTHORIZATION_SCHEMA,
        "candidate": compat.CANDIDATE,
        "stage_id": compat.STAGE_ID,
        "run_id": compat.SCIENTIFIC_ATTEMPT_ID,
        "status": "GCR_PACRE_V24_D_R_STRUCTURAL_R2_AUTHORIZED",
        "allowed_purposes": ["zero_update_structural_gate"],
        "allowed_splits": ["D_R"],
        "D_R_payload_authorized": True,
        "D_V_payload_authorized": False,
        "D_T_payload_authorized": False,
        "training_authorized": False,
        "expires_after_single_materialization": True,
        "access_audit_receipt_fingerprint": audit_fingerprint,
        "dataset_free_receipt_file_sha256": "3" * 64,
        "dataset_free_receipt_fingerprint": "4" * 64,
        "efficiency_receipt_sha256": "5" * 64,
        "efficiency_section_fingerprint": "6" * 64,
        "expected_cache_fingerprint": "7" * 64,
        "expected_population_fingerprint": "8" * 64,
        "expected_real_inputs_fingerprint": "9" * 64,
        "manifest_file_sha256": "a" * 64,
        "protocol_preregistration_fingerprint": "b" * 64,
        "source_binding_fingerprint": "c" * 64,
        "source_closure_fingerprint": (
            compat.SOURCE_CLOSURE_FINGERPRINT_103
        ),
        "state_index_file_sha256": "d" * 64,
    }


def _fake_unit_state(fragment: Path) -> dict[str, object]:
    return {
        "boot_id": "11111111-2222-3333-4444-555555555555",
        "unit_name": compat.OLD_UNIT_NAME,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "Result": "success",
        "ExecMainCode": 0,
        "ExecMainStatus": 0,
        "NRestarts": 0,
        "InvocationID": "",
        "StateChangeTimestampMonotonic": 0,
        "InactiveEnterTimestampMonotonic": 0,
        "ActiveEnterTimestampMonotonic": 0,
        "FragmentPath": str(fragment),
    }


class Case(SimpleNamespace):
    repo: Path
    evidence: Path
    session: Path
    terminal: Path
    authorization: Path
    receipt: Path
    frozen_paths: dict[str, Path]
    component_sources: dict[str, Path]
    component_evidence: dict[str, Path]
    unit_state: dict[str, object]

    def reader(self) -> dict[str, object]:
        return deepcopy(self.unit_state)


def _patch_path(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: Path,
) -> None:
    monkeypatch.setattr(compat, name, value.absolute())


def _case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Case:
    monkeypatch.setattr(
        compat,
        "_validate_compat_unit_realization_chain",
        lambda **_kwargs: None,
    )
    repo = tmp_path / "repo"
    evidence = (
        repo / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
    )
    evidence.mkdir(parents=True)
    evidence.chmod(0o700)
    tools = repo / "tools"
    tools.mkdir()
    runs_parent = repo / "runs/irstd1k_stage_a_seed42"
    runs_parent.mkdir(parents=True)
    session = repo / "sessions/failure.jsonl"
    session.parent.mkdir(parents=True)
    unit_dir = repo / "run/user/systemd/user"
    unit_dir.mkdir(parents=True)
    fragment = unit_dir / compat.OLD_UNIT_NAME

    paths = {
        "FORENSIC_TERMINAL_PATH": (
            evidence / "r2_preaccess_schema_compat_c1_forensic_terminal.json"
        ),
        "COMPAT_AUTHORIZATION_PATH": (
            evidence / "r2_preaccess_schema_compat_c1_authorization.json"
        ),
        "COMPAT_RECEIPT_PATH": (
            evidence / "r2_preaccess_schema_compat_c1_receipt.json"
        ),
        "COMPATIBILITY_RECEIPT_PATH": (
            evidence / "r2_preaccess_schema_compat_c1_receipt.json"
        ),
        "SCIENTIFIC_AUTHORIZATION_PATH": (
            evidence / "D_R_structural_attempt_r2_authorization.json"
        ),
        "SCIENTIFIC_ACCESS_AUDIT_PATH": (
            evidence / "D_R_structural_attempt_r2_access_audit.json"
        ),
        "OLD_RUNTIME_SPEC_PATH": (
            evidence / "D_R_structural_attempt_r2_runtime_spec.json"
        ),
        "OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH": (
            evidence
            / "D_R_structural_attempt_r2_runtime_launch_authorization.json"
        ),
        "OLD_RUNTIME_ARTIFACT_ROOT": (
            evidence / "D_R_structural_attempt_r2_runtime_artifacts"
        ),
        "OLD_GPU_LEASE_ROOT": (
            evidence / "D_R_structural_attempt_r2_gpu_lease"
        ),
        "SCIENTIFIC_RESULT_RECEIPT_PATH": (
            evidence / "D_R_structural_attempt_r2_receipt.json"
        ),
        "SCIENTIFIC_RUN_ROOT": (
            runs_parent / "gcr_pacre_v24_D_R_structural_attempt_r2"
        ),
        "COMPAT_RUNTIME_SPEC_PATH": (
            evidence
            / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json"
        ),
        "COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH": (
            evidence
            / "D_R_structural_attempt_r2_preaccess_compat_c1_"
            "runtime_launch_authorization.json"
        ),
        "COMPAT_RUNTIME_ARTIFACT_ROOT": (
            evidence
            / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_artifacts"
        ),
        "COMPAT_GPU_LEASE_ROOT": (
            evidence
            / "D_R_structural_attempt_r2_preaccess_compat_c1_gpu_lease"
        ),
        "COMPAT_RUN_ROOT_ALIAS_PATH": (
            runs_parent
            / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c1"
        ),
        "COMPAT_RESULT_RECEIPT_ALIAS_PATH": (
            evidence
            / "D_R_structural_attempt_r2_preaccess_compat_c1_receipt.json"
        ),
        "COMPAT_UNIT_REALIZATION_TERMINAL_PATH": (
            evidence
            / "r2_preaccess_compat_c1_unit_realization_terminal.json"
        ),
        "OLD_RELEASE_PATH": (
            tools / "cure_lite_v24_actual_runtime_release.py"
        ),
        "OLD_SUPERVISOR_PATH": (
            tools / "cure_lite_v24_runtime_supervisor.py"
        ),
        "OLD_ADAPTER_PATH": (
            tools / "run_cure_lite_v24_gcr_pacre_dr_gate_r2.py"
        ),
        "PROTOCOL_VERIFIER_PATH": (
            tools / "gcr_pacre_v24_protocol.py"
        ),
        "OLD_UNIT_REALIZER_PATH": (
            tools / "cure_lite_v24_actual_unit_realization.py"
        ),
        "UNIT_RECOVERY_AUTHORIZATION_PATH": (
            evidence / "r2_unit_realization_recovery_authorization.json"
        ),
        "UNIT_RECOVERY_RECEIPT_PATH": (
            evidence / "r2_unit_realization_recovery_receipt.json"
        ),
        "OLD_UNIT_FRAGMENT_PATH": fragment,
        "SESSION_PATH": session,
    }
    monkeypatch.setattr(compat, "REPOSITORY", repo.absolute())
    monkeypatch.setattr(compat, "EVIDENCE_ROOT", evidence.absolute())
    for name, path in paths.items():
        _patch_path(monkeypatch, name, path)

    frozen_regular = {
        "old_release": paths["OLD_RELEASE_PATH"],
        "old_supervisor": paths["OLD_SUPERVISOR_PATH"],
        "old_adapter": paths["OLD_ADAPTER_PATH"],
        "protocol_verifier": paths["PROTOCOL_VERIFIER_PATH"],
        "old_unit_realizer": paths["OLD_UNIT_REALIZER_PATH"],
        "old_unit_fragment": paths["OLD_UNIT_FRAGMENT_PATH"],
    }
    for label, path in frozen_regular.items():
        _write_regular(
            path,
            f"fixed-{label}\n".encode("ascii"),
            mode=0o600 if label == "old_unit_fragment" else 0o664,
        )

    audit = _seal(
        paths["SCIENTIFIC_ACCESS_AUDIT_PATH"],
        _scientific_audit_body(),
        fingerprint_field="receipt_fingerprint",
    )
    authorization = _seal(
        paths["SCIENTIFIC_AUTHORIZATION_PATH"],
        _scientific_authorization_body(
            str(audit["receipt_fingerprint"])
        ),
        fingerprint_field="authorization_fingerprint",
    )
    del authorization
    _seal(
        paths["UNIT_RECOVERY_AUTHORIZATION_PATH"],
        {
            "schema_version": compat.UNIT_RECOVERY_AUTHORIZATION_SCHEMA,
            "kind": "test-recovery-authorization",
        },
        fingerprint_field="authorization_fingerprint",
    )
    _seal(
        paths["UNIT_RECOVERY_RECEIPT_PATH"],
        {
            "schema_version": compat.UNIT_RECOVERY_RECEIPT_SCHEMA,
            "kind": "test-recovery-receipt",
        },
        fingerprint_field="receipt_fingerprint",
    )

    expected_paths = {
        "scientific_authorization": (
            paths["SCIENTIFIC_AUTHORIZATION_PATH"]
        ),
        "scientific_access_audit": (
            paths["SCIENTIFIC_ACCESS_AUDIT_PATH"]
        ),
        "old_release": paths["OLD_RELEASE_PATH"],
        "old_supervisor": paths["OLD_SUPERVISOR_PATH"],
        "old_adapter": paths["OLD_ADAPTER_PATH"],
        "protocol_verifier": paths["PROTOCOL_VERIFIER_PATH"],
        "old_unit_realizer": paths["OLD_UNIT_REALIZER_PATH"],
        "unit_recovery_authorization": (
            paths["UNIT_RECOVERY_AUTHORIZATION_PATH"]
        ),
        "unit_recovery_receipt": (
            paths["UNIT_RECOVERY_RECEIPT_PATH"]
        ),
        "old_unit_fragment": paths["OLD_UNIT_FRAGMENT_PATH"],
    }
    monkeypatch.setattr(
        compat,
        "EXPECTED_FILE_SHA256",
        {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in expected_paths.items()
        },
    )

    command = " ".join(
        [
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            "-u",
            str(paths["OLD_RELEASE_PATH"]),
            "build-spec",
            "--environment-policy",
            str(evidence / "environment-policy.json"),
        ]
    )
    arguments_raw = json.dumps(
        {
            "cmd": command,
            "workdir": str(repo),
            "yield_time_ms": 30000,
            "max_output_tokens": 30000,
        },
        separators=(",", ":"),
    )
    output_text = (
        "Chunk ID: test\n"
        "Wall time: 0.1 seconds\n"
        "Process exited with code 1\n"
        "Output:\n"
        "Traceback (most recent call last):\n"
        "  build_spec(\n"
        "  validate_release_closure(\n"
        "  _validate_scientific_preaccess()\n"
        "  _load_sealed(\n"
        "PermissionError: release evidence fingerprint/schema changed: "
        f"{paths['SCIENTIFIC_ACCESS_AUDIT_PATH']}\n"
    )
    call_id = "call_test_prewrite_failure"
    call_record = {
        "timestamp": "2026-07-30T04:14:34.424Z",
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "exec_command",
            "arguments": arguments_raw,
            "call_id": call_id,
        },
    }
    output_record = {
        "timestamp": "2026-07-30T04:14:34.892Z",
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_text,
        },
    }
    call_raw = json.dumps(
        call_record,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    output_raw = json.dumps(
        output_record,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    _write_regular(
        session,
        b'{"padding":1}\n'
        + b'{"padding":2}\n'
        + call_raw
        + b"\n"
        + output_raw
        + b"\n",
    )
    monkeypatch.setattr(compat, "SESSION_CALL_LINE", 3)
    monkeypatch.setattr(compat, "SESSION_OUTPUT_LINE", 4)
    monkeypatch.setattr(compat, "SESSION_CALL_ID", call_id)
    monkeypatch.setattr(
        compat,
        "SESSION_CALL_RAW_SHA256",
        hashlib.sha256(call_raw).hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "SESSION_OUTPUT_RAW_SHA256",
        hashlib.sha256(output_raw).hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "SESSION_CALL_RECORD_WITH_LF_SHA256",
        hashlib.sha256(call_raw + b"\n").hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "SESSION_OUTPUT_RECORD_WITH_LF_SHA256",
        hashlib.sha256(output_raw + b"\n").hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "SESSION_CALL_ARGUMENTS_SHA256",
        hashlib.sha256(arguments_raw.encode("utf-8")).hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "SESSION_OUTPUT_PAYLOAD_SHA256",
        hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    )

    component_root = repo / "compat-c1"
    component_sources = {
        "compat_policy": component_root / "compat_policy.py",
        "compat_release": component_root / "compat_release.py",
        "compat_supervisor": component_root / "compat_supervisor.py",
        "compat_adapter": component_root / "compat_adapter.py",
        "compat_unit_realizer": component_root / "compat_realizer.py",
        "compat_unit_template": component_root / "compat.service.in",
    }
    for label, path in component_sources.items():
        _write_regular(path, f"new-{label}\n".encode("ascii"))
    component_evidence = {
        "compat_environment_policy": (
            component_root / "evidence/environment-policy.json"
        ),
        "compat_environment_stability": (
            component_root / "evidence/environment-stability.json"
        ),
        "compat_environment_postcleanup": (
            component_root / "evidence/environment-postcleanup.json"
        ),
        "compat_integration_authorization": (
            component_root / "evidence/integration-authorization.json"
        ),
        "compat_integration_receipt": (
            component_root / "evidence/integration-receipt.json"
        ),
        "compat_unit_realization_authorization": (
            component_root / "evidence/realization-authorization.json"
        ),
        "compat_unit_realization_receipt": (
            component_root / "evidence/realization-receipt.json"
        ),
    }
    for label, path in component_evidence.items():
        _seal(
            path,
            {
                "schema_version": f"test-{label}-v1",
                "kind": label,
            },
            fingerprint_field=(
                compat._COMPAT_EVIDENCE_FINGERPRINT_FIELDS[label]
            ),
        )

    fixed_source_constants = {
        "COMPAT_POLICY_SOURCE_PATH": "compat_policy",
        "COMPAT_RELEASE_SOURCE_PATH": "compat_release",
        "COMPAT_SUPERVISOR_SOURCE_PATH": "compat_supervisor",
        "COMPAT_ADAPTER_SOURCE_PATH": "compat_adapter",
        "COMPAT_UNIT_REALIZER_SOURCE_PATH": "compat_unit_realizer",
        "COMPAT_UNIT_TEMPLATE_PATH": "compat_unit_template",
    }
    for constant, label in fixed_source_constants.items():
        _patch_path(monkeypatch, constant, component_sources[label])
    fixed_evidence_constants = {
        "COMPAT_ENVIRONMENT_POLICY_PATH": (
            "compat_environment_policy"
        ),
        "COMPAT_ENVIRONMENT_STABILITY_PATH": (
            "compat_environment_stability"
        ),
        "COMPAT_ENVIRONMENT_POSTCLEANUP_PATH": (
            "compat_environment_postcleanup"
        ),
        "COMPAT_INTEGRATION_AUTHORIZATION_PATH": (
            "compat_integration_authorization"
        ),
        "COMPAT_INTEGRATION_RECEIPT_PATH": (
            "compat_integration_receipt"
        ),
        "COMPAT_UNIT_REALIZATION_AUTHORIZATION_PATH": (
            "compat_unit_realization_authorization"
        ),
        "COMPAT_UNIT_REALIZATION_RECEIPT_PATH": (
            "compat_unit_realization_receipt"
        ),
    }
    for constant, label in fixed_evidence_constants.items():
        _patch_path(monkeypatch, constant, component_evidence[label])

    case = Case(
        repo=repo,
        evidence=evidence,
        session=session,
        terminal=paths["FORENSIC_TERMINAL_PATH"],
        authorization=paths["COMPAT_AUTHORIZATION_PATH"],
        receipt=paths["COMPAT_RECEIPT_PATH"],
        frozen_paths=expected_paths,
        component_sources=component_sources,
        component_evidence=component_evidence,
        unit_state=_fake_unit_state(fragment),
    )
    return case


def _create_terminal(case: Case) -> dict[str, object]:
    return compat.create_forensic_terminal(
        unit_state_reader=case.reader,
        now=lambda: FIXED_NOW,
    )


def _authorize(case: Case) -> dict[str, object]:
    return compat.authorize_compat(
        instruction_id=compat.INSTRUCTION_ID,
        authorization_basis=compat.AUTHORIZATION_BASIS,
        validity_seconds=300,
        unit_state_reader=case.reader,
        now=lambda: FIXED_NOW,
    )


def _receipt_kwargs(case: Case) -> dict[str, object]:
    return {
        "compat_release_path": (
            case.component_sources["compat_release"]
        ),
        "compat_supervisor_path": (
            case.component_sources["compat_supervisor"]
        ),
        "compat_adapter_path": (
            case.component_sources["compat_adapter"]
        ),
        "compat_unit_realizer_path": (
            case.component_sources["compat_unit_realizer"]
        ),
        "compat_unit_template_path": (
            case.component_sources["compat_unit_template"]
        ),
        "compat_environment_policy_path": (
            case.component_evidence["compat_environment_policy"]
        ),
        "compat_environment_stability_path": (
            case.component_evidence["compat_environment_stability"]
        ),
        "compat_environment_postcleanup_path": (
            case.component_evidence["compat_environment_postcleanup"]
        ),
        "compat_integration_authorization_path": (
            case.component_evidence[
                "compat_integration_authorization"
            ]
        ),
        "compat_integration_receipt_path": (
            case.component_evidence["compat_integration_receipt"]
        ),
        "compat_unit_realization_authorization_path": (
            case.component_evidence[
                "compat_unit_realization_authorization"
            ]
        ),
        "compat_unit_realization_receipt_path": (
            case.component_evidence[
                "compat_unit_realization_receipt"
            ]
        ),
        "unit_state_reader": case.reader,
        "now": lambda: FIXED_NOW,
    }


def _seal_chain(
    case: Case,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    terminal = _create_terminal(case)
    authorization = _authorize(case)
    receipt = compat.seal_receipt(**_receipt_kwargs(case))
    return terminal, authorization, receipt


def _expected_spec(receipt: dict[str, object]) -> dict[str, object]:
    sources = receipt["compatibility_source_roots"]
    body: dict[str, object] = {
        "schema_version": compat.RUNTIME_SPEC_SCHEMA,
        "candidate": compat.CANDIDATE,
        "stage_id": compat.STAGE_ID,
        "attempt_id": compat.SCIENTIFIC_ATTEMPT_ID,
        "attempt_ordinal": 2,
        "scientific_preaccess": {
            "authorization_path": str(
                compat.SCIENTIFIC_AUTHORIZATION_PATH
            ),
            "authorization_required_schema": (
                compat.SCIENTIFIC_AUTHORIZATION_SCHEMA
            ),
            "access_audit_path": str(
                compat.SCIENTIFIC_ACCESS_AUDIT_PATH
            ),
            "access_audit_required_schema": (
                compat.SCIENTIFIC_ACCESS_AUDIT_SCHEMA
            ),
            "source_closure_fingerprint_103": (
                compat.SOURCE_CLOSURE_FINGERPRINT_103
            ),
        },
        "artifacts": {
            "root": str(compat.COMPAT_RUNTIME_ARTIFACT_ROOT),
        },
        "runtime": {
            "systemd": {"unit_name": compat.COMPAT_UNIT_NAME},
        },
        "environment": {
            "gpu_lease_path": str(
                compat.COMPAT_GPU_LEASE_ROOT / "active.json"
            ),
        },
        "child": {
            "entrypoint_path": str(
                compat.COMPAT_ADAPTER_SOURCE_PATH
            ),
        },
        "source_bindings": {
            "r2_adapter_path": sources["compat_adapter"]["path"],
            "r2_adapter_file_sha256": (
                sources["compat_adapter"]["file_sha256"]
            ),
            "child_entry_file_sha256": (
                sources["compat_adapter"]["file_sha256"]
            ),
            "supervisor_file_sha256": (
                sources["compat_supervisor"]["file_sha256"]
            ),
        },
    }
    return {
        **body,
        "runtime_spec_fingerprint": compat.stable_fingerprint(body),
    }


def test_isolated_cli_help_has_three_metadata_only_commands() -> None:
    result = subprocess.run(
        [
            str(PYTHON),
            "-I",
            "-S",
            "-B",
            "-u",
            str(Path(compat.__file__).resolve()),
            "--help",
        ],
        cwd=ROOT,
        shell=False,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "create-forensic-terminal" in result.stdout
    assert "authorize-compat" in result.stdout
    assert "seal-receipt" in result.stdout


def test_production_constants_bind_exact_session_and_frozen_sources() -> None:
    assert compat.SESSION_CALL_LINE == 22512
    assert compat.SESSION_OUTPUT_LINE == 22513
    assert compat.SESSION_CALL_RECORD_WITH_LF_SHA256 == (
        "a9959fa90e9b209ab38e379196f74e8ce2949c78b36d46c1362472900f80289e"
    )
    assert compat.SESSION_OUTPUT_RECORD_WITH_LF_SHA256 == (
        "09130fe3eec9ecc25c87b342bebd26809587c7f636d214ff6ae7a9d6af2778e1"
    )
    assert compat.SESSION_CALL_RAW_SHA256 == (
        "fe3685d462cff7ebc5b3588f074abc46befa32be14bce83bab13c45fc64ecb39"
    )
    assert compat.SESSION_OUTPUT_RAW_SHA256 == (
        "1499b7ef4adfdb48ed1e75595fe3a724d956f6bad20a8713aad94f5d42217b4a"
    )
    assert compat.COMPATIBILITY_RECEIPT_PATH == compat.COMPAT_RECEIPT_PATH
    assert compat.COMPAT_RUNTIME_SPEC_PATH.name == (
        "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json"
    )
    assert compat.COMPAT_RUNTIME_ARTIFACT_ROOT.name == (
        "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_artifacts"
    )
    for label, (path, _fingerprint, _schema) in (
        compat._frozen_root_specs().items()
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            compat.EXPECTED_FILE_SHA256[label]
        )


def test_full_chain_is_append_only_runtime_only_and_alias_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)

    terminal, authorization, receipt = _seal_chain(case)
    validated, receipt_root = compat.validate_compatibility_receipt(
        unit_state_reader=case.reader,
        now=lambda: FIXED_NOW + timedelta(hours=1),
    )

    assert terminal["retrospective"] is True
    assert terminal["original_release_generated_terminal"] is False
    assert terminal["failure_phase"] == (
        "prewrite_scientific_preaccess_schema_validation"
    )
    assert terminal["session_failure"]["exit_code"] == 1
    assert terminal["session_failure"]["line_hash_contract"] == {
        "record_hash_includes_trailing_lf": True,
        "raw_json_text_hash_excludes_trailing_lf": True,
        "line_ending": "LF",
    }
    assert authorization["scientific_authority"] == (
        compat._SCIENTIFIC_AUTHORITY
    )
    assert authorization["scientific_attempt_ordinal"] == 2
    assert authorization["runtime_compatibility_id"] == "c1"
    assert authorization["scientific_authority"][
        "fresh_scientific_attempt"
    ] is False
    assert authorization["scientific_authority"][
        "automatic_retry"
    ] is False
    assert authorization["scientific_authority"]["resume"] is False
    assert authorization["scientific_authority"]["allowed_splits"] == [
        "D_R"
    ]
    assert authorization["scientific_authority"][
        "exactly_one_first_materialization"
    ] is True
    assert authorization["mutation_authority"][
        "daemon_reload_authorized"
    ] is False
    assert authorization["compat_lane_authority"][
        "compat_unit_realization_authorized"
    ] is True
    assert authorization["compat_lane_authority"][
        "compat_daemon_reload_authorized"
    ] is True
    assert authorization["compat_lane_authority"][
        "compat_start_authorized"
    ] is False
    assert receipt["schema_compatibility"]["alias_created"] is False
    assert receipt["schema_compatibility"]["audit_rewritten"] is False
    assert receipt["schema_compatibility"][
        "compatibility_consumer_required_schema"
    ] == compat.SCIENTIFIC_ACCESS_AUDIT_SCHEMA
    assert receipt["schema_compatibility"][
        "accept_either_schema"
    ] is False
    assert receipt["runtime_launch_authorized"] is False
    assert receipt["systemd_mutation_authorized"] is False
    assert validated == receipt
    assert receipt_root["path"] == str(case.receipt)
    assert set(receipt["compatibility_source_roots"]) == (
        compat._COMPAT_SOURCE_LABELS
    )
    assert set(receipt["compatibility_evidence_roots"]) == (
        compat._COMPAT_EVIDENCE_LABELS
    )
    assert set(receipt["frozen_generation_roots"]) == (
        compat._FROZEN_ROOT_LABELS
    )
    assert set(receipt["absence_generation_roots"]) == (
        compat._ABSENCE_LABELS
    )
    for path in (case.terminal, case.authorization, case.receipt):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert path.stat().st_nlink == 1
    assert not any(
        os.path.lexists(path)
        for path in compat._absence_paths().values()
    )
    assert sorted(path.name for path in case.evidence.iterdir()) == sorted(
        [
            case.terminal.name,
            case.authorization.name,
            case.receipt.name,
            case.frozen_paths["scientific_authorization"].name,
            case.frozen_paths["scientific_access_audit"].name,
            case.frozen_paths["unit_recovery_authorization"].name,
            case.frozen_paths["unit_recovery_receipt"].name,
        ]
    )


def test_same_byte_source_inode_substitution_between_observations_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    target = case.frozen_paths["old_release"]
    raw = target.read_bytes()
    old_inode = target.stat().st_ino

    def replace_same_bytes() -> None:
        target.rename(target.with_name(target.name + ".retained-generation"))
        _write_regular(target, raw)
        assert target.stat().st_ino != old_inode

    with pytest.raises(
        PermissionError,
        match="changed between observations",
    ):
        compat.create_forensic_terminal(
            unit_state_reader=case.reader,
            between_observations=replace_same_bytes,
            now=lambda: FIXED_NOW,
        )
    assert not case.terminal.exists()


def test_source_content_drift_before_terminal_commit_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    target = case.frozen_paths["protocol_verifier"]

    def drift() -> None:
        target.write_bytes(b"drifted protocol verifier\n")

    with pytest.raises(PermissionError, match="identity changed"):
        compat.create_forensic_terminal(
            unit_state_reader=case.reader,
            between_observations=drift,
            now=lambda: FIXED_NOW,
        )
    assert not case.terminal.exists()


def test_external_session_may_append_but_fixed_failure_lines_must_not_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _create_terminal(case)
    with case.session.open("ab") as stream:
        stream.write(b'{"later_append_only_record":true}\n')

    authorization = _authorize(case)
    assert authorization["runtime_compatibility_id"] == "c1"

    raw = case.session.read_bytes()
    case.session.write_bytes(
        raw.replace(
            b'"call_test_prewrite_failure"',
            b'"call_test_prewrite_tampered"',
            1,
        )
    )
    with pytest.raises(
        PermissionError,
        match="fixed forensic session line changed",
    ):
        compat.validate_forensic_terminal(
            unit_state_reader=case.reader,
        )


def test_absent_old_spec_race_before_terminal_commit_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)

    def create_forbidden_spec() -> None:
        _write_regular(
            compat.OLD_RUNTIME_SPEC_PATH,
            b"forbidden\n",
            mode=0o444,
        )

    with pytest.raises(
        PermissionError,
        match="required absent path exists",
    ):
        compat.create_forensic_terminal(
            unit_state_reader=case.reader,
            between_observations=create_forbidden_spec,
            now=lambda: FIXED_NOW,
        )
    assert not case.terminal.exists()


def test_old_unit_state_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    calls = 0

    def reader() -> dict[str, object]:
        nonlocal calls
        calls += 1
        state = case.reader()
        if calls >= 2:
            state["NRestarts"] = 1
        return state

    with pytest.raises(
        PermissionError,
        match="untouched static unit",
    ):
        compat.create_forensic_terminal(
            unit_state_reader=reader,
            now=lambda: FIXED_NOW,
        )
    assert not case.terminal.exists()


@pytest.mark.parametrize(
    ("operation", "path_name"),
    [
        ("terminal", "terminal"),
        ("authorization", "authorization"),
        ("receipt", "receipt"),
    ],
)
def test_offpath_output_is_rejected_before_observation_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    path_name: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    calls = 0

    def reader() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return case.reader()

    offpath = tmp_path / f"offpath-{path_name}.json"
    if operation == "terminal":
        invoke = lambda: compat.create_forensic_terminal(
            terminal_path=offpath,
            unit_state_reader=reader,
            now=lambda: FIXED_NOW,
        )
    elif operation == "authorization":
        _create_terminal(case)
        calls = 0
        invoke = lambda: compat.authorize_compat(
            instruction_id="explicit",
            authorization_basis="explicit",
            authorization_path=offpath,
            unit_state_reader=reader,
            now=lambda: FIXED_NOW,
        )
    else:
        _create_terminal(case)
        _authorize(case)
        calls = 0
        kwargs = _receipt_kwargs(case)
        kwargs["receipt_path"] = offpath
        kwargs["unit_state_reader"] = reader
        invoke = lambda: compat.seal_receipt(**kwargs)

    with pytest.raises(PermissionError, match="fixed path"):
        invoke()
    assert calls == 0
    assert not offpath.exists()


@pytest.mark.parametrize("artifact", ["terminal", "authorization", "receipt"])
def test_extra_top_level_fields_are_rejected_even_with_valid_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    terminal, authorization, receipt = _seal_chain(case)
    definitions = {
        "terminal": (
            case.terminal,
            terminal,
            "terminal_fingerprint",
            lambda: compat.validate_forensic_terminal(
                unit_state_reader=case.reader
            ),
        ),
        "authorization": (
            case.authorization,
            authorization,
            "authorization_fingerprint",
            lambda: compat.validate_compat_authorization(
                unit_state_reader=case.reader,
                require_fresh=False,
                now=lambda: FIXED_NOW,
            ),
        ),
        "receipt": (
            case.receipt,
            receipt,
            "receipt_fingerprint",
            lambda: compat.validate_compatibility_receipt(
                unit_state_reader=case.reader,
                now=lambda: FIXED_NOW,
            ),
        ),
    }
    path, payload, fingerprint_field, validator = definitions[artifact]
    _replace_sealed(
        path,
        payload,
        fingerprint_field=fingerprint_field,
        mutate=lambda body: body.__setitem__("extra", False),
    )

    with pytest.raises(PermissionError, match="keys changed"):
        validator()


def test_authorization_must_be_fresh_when_receipt_is_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _create_terminal(case)
    _authorize(case)
    kwargs = _receipt_kwargs(case)
    kwargs["now"] = lambda: FIXED_NOW + timedelta(seconds=301)

    with pytest.raises(PermissionError, match="stale"):
        compat.seal_receipt(**kwargs)
    assert not case.receipt.exists()


def test_authorization_identity_is_fixed_to_the_user_instruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _create_terminal(case)
    with pytest.raises(
        ValueError,
        match="authorization input is invalid",
    ):
        compat.authorize_compat(
            instruction_id="different-instruction",
            authorization_basis=compat.AUTHORIZATION_BASIS,
            validity_seconds=300,
            unit_state_reader=case.reader,
            now=lambda: FIXED_NOW,
        )
    assert not case.authorization.exists()


def test_failed_unit_terminal_permanently_blocks_bridge_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _create_terminal(case)
    _authorize(case)
    _seal(
        compat.COMPAT_UNIT_REALIZATION_TERMINAL_PATH,
        {
            "schema_version": "test-unit-realization-terminal-v1",
            "passed": False,
        },
        fingerprint_field="terminal_fingerprint",
    )
    with pytest.raises(PermissionError, match="absent path exists"):
        compat.seal_receipt(**_receipt_kwargs(case))
    assert not case.receipt.exists()


def test_unit_realization_semantic_validator_failure_blocks_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _create_terminal(case)
    _authorize(case)

    def reject_chain(**_kwargs: object) -> None:
        raise PermissionError("unit realization semantic failure")

    monkeypatch.setattr(
        compat,
        "_validate_compat_unit_realization_chain",
        reject_chain,
    )
    with pytest.raises(
        PermissionError,
        match="unit realization semantic failure",
    ):
        compat.seal_receipt(**_receipt_kwargs(case))
    assert not case.receipt.exists()


def test_same_byte_compat_source_substitution_before_receipt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _create_terminal(case)
    _authorize(case)
    target = case.component_sources["compat_supervisor"]
    raw = target.read_bytes()
    old_inode = target.stat().st_ino

    def replace_same_bytes() -> None:
        target.rename(target.with_name(target.name + ".retained-generation"))
        _write_regular(target, raw)
        assert target.stat().st_ino != old_inode

    kwargs = _receipt_kwargs(case)
    kwargs["between_observations"] = replace_same_bytes
    with pytest.raises(
        PermissionError,
        match="changed between observations",
    ):
        compat.seal_receipt(**kwargs)
    assert not case.receipt.exists()


def test_same_byte_policy_source_substitution_before_receipt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _create_terminal(case)
    _authorize(case)
    target = case.component_sources["compat_policy"]
    raw = target.read_bytes()

    def replace_same_bytes() -> None:
        target.rename(target.with_name(target.name + ".old-generation"))
        _write_regular(target, raw)

    kwargs = _receipt_kwargs(case)
    kwargs["between_observations"] = replace_same_bytes
    with pytest.raises(
        PermissionError,
        match="changed between observations",
    ):
        compat.seal_receipt(**kwargs)
    assert not case.receipt.exists()


def test_old_adapter_cannot_be_reused_as_compat_adapter_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _create_terminal(case)
    _authorize(case)
    kwargs = _receipt_kwargs(case)
    kwargs["compat_adapter_path"] = case.frozen_paths["old_adapter"]

    with pytest.raises(
        PermissionError,
        match="fixed paths changed",
    ):
        compat.seal_receipt(**kwargs)
    assert not case.receipt.exists()


def test_archival_receipt_validation_can_skip_only_future_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _seal_chain(case)
    _write_regular(
        compat.COMPAT_RUNTIME_SPEC_PATH,
        b"later sealed by a separately authorized release\n",
        mode=0o444,
    )

    with pytest.raises(
        PermissionError,
        match="required absent path exists",
    ):
        compat.validate_compatibility_receipt(
            unit_state_reader=case.reader,
            require_future_absence=True,
            now=lambda: FIXED_NOW + timedelta(hours=1),
        )
    payload, _root = compat.validate_compatibility_receipt(
        unit_state_reader=case.reader,
        require_future_absence=False,
        now=lambda: FIXED_NOW + timedelta(hours=1),
    )
    assert payload["compatibility_closure_passed"] is True


def test_consumer_verifier_exactly_binds_expected_spec_after_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _terminal, _authorization, receipt = _seal_chain(case)
    expected_spec = _expected_spec(receipt)
    _write_regular(
        compat.COMPAT_RUNTIME_SPEC_PATH,
        (
            compat.canonical_json(expected_spec) + "\n"
        ).encode("utf-8"),
        mode=0o444,
    )

    verified = compat.verify_compatibility_receipt(
        case.receipt,
        expected_spec=expected_spec,
        require_spec_binding=True,
        unit_state_reader=case.reader,
        now=lambda: FIXED_NOW + timedelta(hours=1),
    )
    assert verified == receipt

    drifted = deepcopy(expected_spec)
    drifted["scientific_preaccess"][
        "access_audit_required_schema"
    ] = compat.BUGGY_CONSUMER_ACCESS_AUDIT_SCHEMA
    drifted_body = deepcopy(drifted)
    drifted_body.pop("runtime_spec_fingerprint")
    drifted["runtime_spec_fingerprint"] = compat.stable_fingerprint(
        drifted_body
    )
    with pytest.raises(
        PermissionError,
        match="differs from consumer view",
    ):
        compat.verify_compatibility_receipt(
            case.receipt,
            expected_spec=drifted,
            require_spec_binding=True,
            unit_state_reader=case.reader,
            now=lambda: FIXED_NOW + timedelta(hours=1),
        )


def test_prewrite_preview_is_separate_from_fixed_runtime_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _terminal, _authorization, receipt = _seal_chain(case)
    expected_spec = _expected_spec(receipt)

    assert compat.verify_compatibility_prewrite_spec(
        case.receipt,
        expected_spec,
        unit_state_reader=case.reader,
        now=lambda: FIXED_NOW + timedelta(hours=1),
    ) == receipt
    assert not compat.COMPAT_RUNTIME_SPEC_PATH.exists()

    with pytest.raises(
        PermissionError,
        match="binding phase is inconsistent",
    ):
        compat.verify_compatibility_receipt(
            case.receipt,
            expected_spec=expected_spec,
            require_spec_binding=False,
            unit_state_reader=case.reader,
            now=lambda: FIXED_NOW + timedelta(hours=1),
        )


def test_runtime_activation_flag_reaches_unit_chain_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _seal_chain(case)
    observed: list[bool] = []

    def capture(**kwargs: object) -> None:
        observed.append(bool(kwargs["allow_runtime_activation"]))

    monkeypatch.setattr(
        compat,
        "_validate_compat_unit_realization_chain",
        capture,
    )
    compat.validate_compatibility_receipt(
        case.receipt,
        require_future_absence=True,
        allow_runtime_activation=True,
        unit_state_reader=case.reader,
        now=lambda: FIXED_NOW + timedelta(hours=1),
    )
    assert observed == [True]
