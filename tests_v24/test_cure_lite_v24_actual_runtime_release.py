from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

from tools import cure_lite_v24_actual_runtime_release as release
from tests_v24 import test_cure_lite_v24_environment_cleanup as cleanup_fixtures
from tests_v24 import (
    test_cure_lite_v24_actual_unit_realization as realization_fixtures,
)


def _seal(
    path: Path, body: dict[str, object], field: str,
) -> dict[str, object]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    payload = {**body, field: release.stable_fingerprint(body)}
    path.write_text(release.canonical_json(payload) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return payload


def _patch_release_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = (tmp_path / "runtime_evidence_r2").resolve()
    root.mkdir(mode=0o700)
    values = {
        "EVIDENCE_ROOT": root,
        "RUNTIME_SPEC_PATH": root / "D_R_structural_attempt_r2_runtime_spec.json",
        "RUNTIME_LAUNCH_AUTHORIZATION_PATH": root / "D_R_structural_attempt_r2_runtime_launch_authorization.json",
        "SCIENTIFIC_AUTHORIZATION_PATH": root / "D_R_structural_attempt_r2_authorization.json",
        "SCIENTIFIC_ACCESS_AUDIT_PATH": root / "D_R_structural_attempt_r2_access_audit.json",
        "PRIOR_R1_INTERRUPTION_RECEIPT_PATH": root / "r1-interruption.json",
        "RUNTIME_ARTIFACT_ROOT": root / "D_R_structural_attempt_r2_runtime_artifacts",
        "GPU_LEASE_ROOT": root / "D_R_structural_attempt_r2_gpu_lease",
    }
    for name, value in values.items():
        monkeypatch.setattr(release, name, value)
    return root


def _fake_shadow(fragment: Path) -> dict[str, str]:
    result = {key: f"value-{key}" for key in release.supervisor._SYSTEMD_IMMUTABLE_SHADOW_KEYS}
    result.update({
        "Type": "exec", "Restart": "no", "KillMode": "mixed",
        "SendSIGKILL": "yes", "TimeoutStopUSec": "10min",
        "FragmentPath": str(fragment), "DropInPaths": "", "Transient": "no",
        "Environment": "PYTHONUNBUFFERED=1", "UnsetEnvironment": "",
        "WorkingDirectory": str(release.REPOSITORY), "UMask": "0077",
        "ExitType": "main", "RuntimeMaxUSec": "infinity",
        "WatchdogUSec": "disabled",
        "OOMPolicy": "kill", "RemainAfterExit": "no", "StandardInput": "null",
        "StandardOutput": "journal", "StandardError": "journal",
        "StartLimitIntervalUSec": "infinity", "StartLimitBurst": "1",
        "KillSignal": "15",
    })
    modes = {
        "ExecCondition": "claim-materialization",
        "ExecStartPre": "verify-runtime-spec",
        "ExecStart": "run-once",
        "ExecStopPost": "record-systemd-exit",
    }
    for directive, mode in modes.items():
        argv = (
            f"{release.PYTHON_PATH} -I -S -B -u {release.SUPERVISOR_PATH} "
            f"{mode} --spec {release.RUNTIME_SPEC_PATH}"
        )
        result[directive] = (
            f"{{ path={release.PYTHON_PATH} ; argv[]={argv} ; ignore_errors=no }}"
        )
    return result


def _fake_closure(tmp_path: Path) -> dict[str, object]:
    fragment = tmp_path / "actual.service"
    fragment.write_text("[Service]\n", encoding="utf-8")
    fragment.chmod(0o600)
    shadow = _fake_shadow(fragment)
    python = release._trusted_python_binding()
    realizer_python = {
        "path": python["path"], "resolved_path": python["path"],
        "path_is_symlink": False,
        "file_sha256": python["file_sha256"],
        "device": python["device"], "inode": python["inode"],
        "owner_uid": python["owner_uid"], "mode": python["mode"],
    }
    scientific_authorization = {
        "schema_version": "cure-lite-v24-D_R-structural-r2-authorization-v1",
        "authorization_fingerprint": "a" * 64,
    }
    scientific_audit = {
        "schema_version": "cure-lite-v24-split-access-audit-r2-v1",
        "receipt_fingerprint": "b" * 64,
    }
    return {
        "environment": {"policy": {"selected_gpu": {
            "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "pci_bus_id": "00000000:02:00.0", "minor_number": 0,
        }}},
        "realization": {
            "authorization": {
                "executable_bindings": {"python": realizer_python},
            },
            "receipt": {"fragment_identity": {
                "file_sha256": release.file_sha256(fragment),
            }, "executable_bindings": {"python": realizer_python}},
            "live_shadow": shadow,
        },
        "scientific": {
            "authorization": scientific_authorization, "audit": scientific_audit,
        },
    }


def _input_files(tmp_path: Path) -> dict[str, Path]:
    names = {
        "policy_path": "policy.json", "precleanup_path": "pre.json",
        "cleanup_plan_path": "plan.json", "cleanup_authorization_path": "cleanup-auth.json",
        "cleanup_receipt_path": "cleanup-receipt.json", "stability_path": "stability.json",
        "postcleanup_path": "post.json", "integration_authorization_path": "integration-auth.json",
        "integration_receipt_path": "integration-receipt.json",
        "realization_authorization_path": "realization-auth.json",
        "realization_receipt_path": "realization-receipt.json",
    }
    result = {}
    for key, name in names.items():
        path = (tmp_path / "inputs" / name).resolve()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        path.chmod(0o444)
        result[key] = path
    return result


def _create_scientific_and_prior_files() -> None:
    _seal(
        release.SCIENTIFIC_AUTHORIZATION_PATH,
        {"schema_version": "cure-lite-v24-D_R-structural-r2-authorization-v1"},
        "authorization_fingerprint",
    )
    _seal(
        release.SCIENTIFIC_ACCESS_AUDIT_PATH,
        {"schema_version": "cure-lite-v24-split-access-audit-r2-v1"},
        "receipt_fingerprint",
    )
    _seal(
        release.PRIOR_R1_INTERRUPTION_RECEIPT_PATH,
        {"schema_version": "cure-lite-v24-D_R-structural-interruption-receipt-v2"},
        "receipt_fingerprint",
    )


def _create_valid_scientific_preaccess() -> tuple[dict[str, object], dict[str, object]]:
    audit_body = {
        key: "1" * 64
        for key in release._SCIENTIFIC_AUDIT_KEYS
        if key != "receipt_fingerprint"
    }
    audit_body.update({
        "schema_version": "cure-lite-v24-split-access-audit-r2-v1",
        "stage_id": release.STAGE_ID, "allowed_splits": ["D_R"],
        "observed_payloads": [], "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    })
    audit = _seal(
        release.SCIENTIFIC_ACCESS_AUDIT_PATH, audit_body,
        "receipt_fingerprint",
    )
    authorization_body = {
        key: "2" * 64
        for key in release._SCIENTIFIC_AUTHORIZATION_KEYS
        if key != "authorization_fingerprint"
    }
    authorization_body.update({
        "schema_version": "cure-lite-v24-D_R-structural-r2-authorization-v1",
        "candidate": release.CANDIDATE, "stage_id": release.STAGE_ID,
        "run_id": release.ATTEMPT_ID,
        "status": "GCR_PACRE_V24_D_R_STRUCTURAL_R2_AUTHORIZED",
        "D_R_payload_authorized": True, "D_V_payload_authorized": False,
        "D_T_payload_authorized": False, "training_authorized": False,
        "expires_after_single_materialization": True,
        "allowed_splits": ["D_R"],
        "allowed_purposes": ["zero_update_structural_gate"],
        "source_closure_fingerprint": release.SOURCE_CLOSURE_FINGERPRINT_103,
        "access_audit_receipt_fingerprint": audit["receipt_fingerprint"],
    })
    authorization = _seal(
        release.SCIENTIFIC_AUTHORIZATION_PATH, authorization_body,
        "authorization_fingerprint",
    )
    return authorization, audit


def _create_valid_prior_interruption() -> dict[str, object]:
    body = {
        key: None for key in release._PRIOR_INTERRUPTION_KEYS
        if key != "receipt_fingerprint"
    }
    body.update({
        "schema_version": "cure-lite-v24-D_R-structural-interruption-receipt-v2",
        "candidate": release.CANDIDATE,
        "run_id": "gcr_pacre_v24_D_R_zero_update_structural_r1",
        "authoritative_observation": {
            "classification": "EXECUTION_OBSERVABILITY_LOST_NO_DECISION",
            "scientific_model_failure_established": False,
            "structural_gate_failure_established": False,
        },
        "protocol_disposition": {
            "attempt_ordinal": 1, "official_D_R_receipt_valid": False,
            "same_run_resume_allowed": False, "automatic_retry_allowed": False,
            "fresh_attempt_currently_authorized": False,
        },
    })
    return _seal(
        release.PRIOR_R1_INTERRUPTION_RECEIPT_PATH, body,
        "receipt_fingerprint",
    )


def _sealed_recovery_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    cleanup = release.cleanup
    (
        plan_path, original_authorization_path, original_intent_path,
        original_terminal_path, plan, fragment,
    ) = cleanup_fixtures._sealed_partial_failure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cleanup, "validate_live_manager_generation",
        lambda expected: dict(expected),
    )
    stopped = False

    def snapshots(unit: str) -> dict[str, object]:
        if unit == cleanup.GPU2_DNANET_UNIT:
            return cleanup_fixtures._snapshot(
                fragment, unit=unit, nrestarts="11",
            )
        return cleanup_fixtures._snapshot(
            fragment, nrestarts="11",
            active_state="inactive" if stopped else "activating",
            sub_state="dead" if stopped else "auto-restart",
        )

    recovery_authorization = cleanup.build_recovery_authorization(
        plan_path=plan_path,
        original_authorization_path=original_authorization_path,
        intent_path=original_intent_path,
        terminal_failure_path=original_terminal_path,
        authorization_basis="sealed test recovery only",
        explicit_user_instruction_id=cleanup.RECOVERY_USER_INSTRUCTION_ID,
        snapshot_reader=snapshots,
        activation_guard_reader=lambda _generation: cleanup_fixtures._guard(),
    )
    recovery_authorization_path = tmp_path / "recovery-authorization.json"
    recovery_authorization_body = dict(recovery_authorization)
    recovery_authorization_body.pop("recovery_authorization_fingerprint")
    cleanup.write_create_once_json(
        recovery_authorization_path, recovery_authorization_body,
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
        intent_path=original_intent_path,
        terminal_failure_path=original_terminal_path,
        recovery_authorization_path=recovery_authorization_path,
        receipt_directory=tmp_path / "recovery-receipts",
        snapshot_reader=snapshots,
        activation_guard_reader=lambda _generation: cleanup_fixtures._guard(),
        command_runner=runner,
    )
    assert commands == [[
        cleanup.SYSTEMCTL_PATH, "--user", "stop", cleanup.GPU0_CONFLICT_UNIT,
    ]]
    return {
        "precleanup_path": Path(str(plan["inventory_receipt_path"])),
        "plan_path": plan_path,
        "authorization_path": original_authorization_path,
        "receipt_path": tmp_path / "recovery-receipts/cleanup-receipt.json",
        "receipt": receipt,
    }


def _rebind_recovery_chain(
    chain: dict[str, object],
    output_root: Path,
    *,
    authorization_updates: dict[str, object] | None = None,
    intent_updates: dict[str, object] | None = None,
    action_updates: dict[str, object] | None = None,
) -> Path:
    """Reseal a semantically linked recovery chain after a focused mutation."""

    receipt = deepcopy(chain["receipt"])
    lineage = receipt["partial_lineage"]
    authorization = release.cleanup.load_sealed_json(
        lineage["recovery_authorization"]["path"]
    )
    intent = release.cleanup.load_sealed_json(
        lineage["recovery_intent"]["path"]
    )
    action = release.cleanup.load_sealed_json(
        lineage["recovery_action_receipt"]["path"]
    )

    authorization_body = dict(authorization)
    authorization_body.pop("recovery_authorization_fingerprint")
    authorization_body.update(authorization_updates or {})
    authorization_path = output_root / "recovery-authorization.json"
    rebound_authorization = _seal(
        authorization_path,
        authorization_body,
        "recovery_authorization_fingerprint",
    )

    intent_body = dict(intent)
    intent_body.pop("recovery_intent_fingerprint")
    intent_body.update(intent_updates or {})
    intent_body.update({
        "recovery_authorization_file_sha256": release.file_sha256(
            authorization_path
        ),
        "recovery_authorization_fingerprint": rebound_authorization[
            "recovery_authorization_fingerprint"
        ],
    })
    intent_path = output_root / "recovery-intent.json"
    rebound_intent = _seal(
        intent_path, intent_body, "recovery_intent_fingerprint",
    )

    action_body = dict(action)
    action_body.pop("recovery_action_receipt_fingerprint")
    action_body.update(action_updates or {})
    action_body["recovery_intent_fingerprint"] = rebound_intent[
        "recovery_intent_fingerprint"
    ]
    action_path = output_root / "recovery-action.json"
    rebound_action = _seal(
        action_path, action_body, "recovery_action_receipt_fingerprint",
    )

    def root(
        path: Path, payload: dict[str, object], fingerprint_field: str,
    ) -> dict[str, str]:
        return {
            "path": str(path),
            "file_sha256": release.file_sha256(path),
            "fingerprint_field": fingerprint_field,
            "fingerprint": str(payload[fingerprint_field]),
        }

    receipt.pop("cleanup_receipt_fingerprint")
    receipt["partial_lineage"].update({
        "recovery_authorization": root(
            authorization_path,
            rebound_authorization,
            "recovery_authorization_fingerprint",
        ),
        "recovery_intent": root(
            intent_path, rebound_intent, "recovery_intent_fingerprint",
        ),
        "recovery_action_receipt": root(
            action_path,
            rebound_action,
            "recovery_action_receipt_fingerprint",
        ),
    })
    receipt["intent_fingerprint"] = rebound_intent[
        "recovery_intent_fingerprint"
    ]
    receipt["action_receipt_fingerprints"] = [
        rebound_action["recovery_action_receipt_fingerprint"]
    ]
    return_path = output_root / "cleanup-receipt.json"
    _seal(return_path, receipt, "cleanup_receipt_fingerprint")
    return return_path


def _sealed_normal_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    cleanup = release.cleanup
    plan_path, authorization_path, plan, fragment = (
        cleanup_fixtures._sealed_plan_and_authorization(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(
        cleanup, "validate_live_manager_generation",
        lambda expected: dict(expected),
    )
    conflict_calls = 0

    def snapshots(unit: str) -> dict[str, object]:
        nonlocal conflict_calls
        if unit == cleanup.GPU2_DNANET_UNIT:
            return cleanup_fixtures._snapshot(fragment, unit=unit)
        conflict_calls += 1
        if conflict_calls <= 2:
            return cleanup_fixtures._snapshot(fragment, nrestarts="11")
        if conflict_calls <= 4:
            return cleanup_fixtures._snapshot(
                fragment, unit_file_state="masked-runtime", nrestarts="11",
            )
        return cleanup_fixtures._snapshot(
            fragment, unit_file_state="masked-runtime",
            active_state="inactive", sub_state="dead", nrestarts="11",
        )

    receipt_root = tmp_path / "normal-receipts"
    cleanup.execute_cleanup(
        plan_path=plan_path, authorization_path=authorization_path,
        receipt_directory=receipt_root, snapshot_reader=snapshots,
        command_runner=lambda argv: subprocess.CompletedProcess(
            list(argv), 0, "", "",
        ),
    )
    return {
        "precleanup_path": Path(str(plan["inventory_receipt_path"])),
        "plan_path": plan_path, "authorization_path": authorization_path,
        "receipt_path": receipt_root / "cleanup-receipt.json",
    }


def test_production_output_paths_are_exact_and_not_cli_overridable() -> None:
    expected = (
        "/home/md0/ly/cure_lite/protocols/IRSTD-1K/gcr_pacre_v24/"
        "runtime_evidence_r2/"
    )
    assert str(release.RUNTIME_SPEC_PATH) == expected + "D_R_structural_attempt_r2_runtime_spec.json"
    assert str(release.RUNTIME_LAUNCH_AUTHORIZATION_PATH) == expected + "D_R_structural_attempt_r2_runtime_launch_authorization.json"
    help_text = release._parser().format_help()
    assert "--output" not in help_text and "--runtime-spec" not in help_text


def test_direct_file_cli_help_smoke() -> None:
    completed = subprocess.run(
        [
            str(release.PYTHON_PATH), "-I", "-S", "-B",
            str(Path(release.__file__)), "--help",
        ],
        cwd="/tmp", text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "build-spec" in completed.stdout and "authorize-launch" in completed.stdout


def test_cleanup_release_chain_accepts_only_complete_recovery_final_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _sealed_recovery_cleanup(tmp_path, monkeypatch)
    result = release._validate_cleanup_chain(
        precleanup_path=chain["precleanup_path"],
        plan_path=chain["plan_path"],
        authorization_path=chain["authorization_path"],
        receipt_path=chain["receipt_path"],
    )
    assert result["receipt"]["cleanup_mode"] == release.cleanup.RECOVERY_CLEANUP_MODE
    assert result["recovery_action_receipt"]["action"] == {
        "ordinal": 1, "unit_name": release.cleanup.GPU0_CONFLICT_UNIT,
        "action": "stop",
    }
    assert result["recovery_action_receipt"]["returncode"] == 0
    assert result["original_terminal_failure"]["inflight_action"]["action"] == {
        "ordinal": 0, "unit_name": release.cleanup.GPU0_CONFLICT_UNIT,
        "action": "mask-runtime",
    }
    assert result["original_terminal_failure"]["inflight_action"]["returncode"] == 0
    lineage = result["receipt"]["partial_lineage"]
    assert set(lineage) == {
        "plan", "original_authorization", "original_intent",
        "original_terminal_failure", "recovery_authorization",
        "recovery_intent", "recovery_action_receipt",
        "legacy_runtime_mask_may_remain_false_reconciled",
        "original_stop_dispatched",
    }
    assert lineage["legacy_runtime_mask_may_remain_false_reconciled"] is True
    assert lineage["original_stop_dispatched"] is False
    receipt_parent = Path(chain["receipt_path"]).parent
    assert not (receipt_parent / "cleanup-intent.json").exists()
    assert not (receipt_parent / "action-000.json").exists()
    assert not (receipt_parent / "action-001.json").exists()


def test_cleanup_release_chain_rejects_valid_normal_final_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal_root = tmp_path / "normal"
    normal_root.mkdir(mode=0o700)
    chain = _sealed_normal_cleanup(normal_root, monkeypatch)
    normal = release.cleanup.load_sealed_json(chain["receipt_path"])
    assert release.cleanup.validate_final_cleanup_receipt(normal) == normal
    assert normal["cleanup_mode"] == release.cleanup.NORMAL_CLEANUP_MODE
    with pytest.raises(PermissionError, match="not the recovery final"):
        release._validate_cleanup_chain(**chain)


def test_cleanup_release_chain_rejects_recovery_root_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _sealed_recovery_cleanup(tmp_path, monkeypatch)
    body = deepcopy(chain["receipt"])
    body.pop("cleanup_receipt_fingerprint")
    lineage = deepcopy(body["partial_lineage"])
    lineage["recovery_action_receipt"]["file_sha256"] = "0" * 64
    body["partial_lineage"] = lineage
    bad_path = Path(chain["receipt_path"]).parent / "bad-root-final.json"
    _seal(bad_path, body, "cleanup_receipt_fingerprint")
    with pytest.raises(PermissionError, match="root binding"):
        release._validate_cleanup_chain(
            precleanup_path=chain["precleanup_path"],
            plan_path=chain["plan_path"],
            authorization_path=chain["authorization_path"],
            receipt_path=bad_path,
        )


def test_cleanup_release_chain_rejects_rehashed_nonzero_recovery_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _sealed_recovery_cleanup(tmp_path, monkeypatch)
    receipt = chain["receipt"]
    action_root = receipt["partial_lineage"]["recovery_action_receipt"]
    action = release.cleanup.load_sealed_json(action_root["path"])
    action_body = dict(action)
    action_body.pop("recovery_action_receipt_fingerprint")
    action_body["returncode"] = 1
    bad_action_path = Path(chain["receipt_path"]).parent / "bad-action.json"
    bad_action = _seal(
        bad_action_path, action_body, "recovery_action_receipt_fingerprint",
    )
    bad_action_root = {
        "path": str(bad_action_path),
        "file_sha256": release.file_sha256(bad_action_path),
        "fingerprint_field": "recovery_action_receipt_fingerprint",
        "fingerprint": bad_action["recovery_action_receipt_fingerprint"],
    }
    final_body = deepcopy(receipt)
    final_body.pop("cleanup_receipt_fingerprint")
    final_body["partial_lineage"]["recovery_action_receipt"] = bad_action_root
    final_body["action_receipt_fingerprints"] = [
        bad_action["recovery_action_receipt_fingerprint"]
    ]
    bad_final_path = Path(chain["receipt_path"]).parent / "bad-action-final.json"
    bad_final = _seal(
        bad_final_path, final_body, "cleanup_receipt_fingerprint",
    )
    assert release.cleanup.validate_final_cleanup_receipt(bad_final) == bad_final
    with pytest.raises(PermissionError, match="unique rc0 stop"):
        release._validate_cleanup_chain(
            precleanup_path=chain["precleanup_path"],
            plan_path=chain["plan_path"],
            authorization_path=chain["authorization_path"],
            receipt_path=bad_final_path,
        )


@pytest.mark.parametrize("drift", ["manager", "guard"])
def test_cleanup_release_chain_rejects_final_manager_or_guard_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str,
) -> None:
    chain = _sealed_recovery_cleanup(tmp_path, monkeypatch)
    body = deepcopy(chain["receipt"])
    body.pop("cleanup_receipt_fingerprint")
    if drift == "manager":
        body["manager_generation"] = {"unbound_generation": True}
    else:
        body["activation_guard"]["inode"] += 1
    bad_path = Path(chain["receipt_path"]).parent / f"bad-{drift}-final.json"
    bad = _seal(bad_path, body, "cleanup_receipt_fingerprint")
    assert release.cleanup.validate_final_cleanup_receipt(bad) == bad
    with pytest.raises(PermissionError, match="final receipt lineage"):
        release._validate_cleanup_chain(
            precleanup_path=chain["precleanup_path"],
            plan_path=chain["plan_path"],
            authorization_path=chain["authorization_path"],
            receipt_path=bad_path,
        )


@pytest.mark.parametrize(
    ("edge", "target"),
    [
        ("intent_before_issued", "intent"),
        ("action_started_before_intent", "action_started"),
        ("action_created_before_started", "action_created"),
        ("action_created_after_expiry", "action_created_after_expiry"),
    ],
)
def test_cleanup_release_chain_rejects_each_recovery_chronology_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edge: str,
    target: str,
) -> None:
    chain = _sealed_recovery_cleanup(tmp_path, monkeypatch)
    lineage = chain["receipt"]["partial_lineage"]
    authorization = release.cleanup.load_sealed_json(
        lineage["recovery_authorization"]["path"]
    )
    intent = release.cleanup.load_sealed_json(
        lineage["recovery_intent"]["path"]
    )
    action = release.cleanup.load_sealed_json(
        lineage["recovery_action_receipt"]["path"]
    )

    def parsed(value: object) -> datetime:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    def rendered(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    intent_updates: dict[str, object] = {}
    action_updates: dict[str, object] = {}
    if target == "intent":
        intent_updates["created_at_utc"] = rendered(
            parsed(authorization["issued_at_utc"]) - timedelta(seconds=1)
        )
    elif target == "action_started":
        action_updates["started_at_utc"] = rendered(
            parsed(intent["created_at_utc"]) - timedelta(seconds=1)
        )
    elif target == "action_created":
        action_updates["created_at_utc"] = rendered(
            parsed(action["started_at_utc"]) - timedelta(seconds=1)
        )
    else:
        action_updates["created_at_utc"] = rendered(
            parsed(authorization["expires_at_utc"]) + timedelta(seconds=1)
        )

    bad_receipt_path = _rebind_recovery_chain(
        chain,
        tmp_path / f"{edge}-rebound",
        intent_updates=intent_updates,
        action_updates=action_updates,
    )
    with pytest.raises(PermissionError, match="recovery chronology"):
        release._validate_cleanup_chain(
            precleanup_path=chain["precleanup_path"],
            plan_path=chain["plan_path"],
            authorization_path=chain["authorization_path"],
            receipt_path=bad_receipt_path,
        )


def test_cleanup_release_chain_accepts_historical_environment_auditor_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _sealed_recovery_cleanup(tmp_path, monkeypatch)
    lineage = chain["receipt"]["partial_lineage"]
    authorization = release.cleanup.load_sealed_json(
        lineage["recovery_authorization"]["path"]
    )
    bindings = deepcopy(authorization["executable_bindings"])
    current_sha = bindings["environment_auditor"]["file_sha256"]
    historical_sha = "0" * 64 if current_sha != "0" * 64 else "1" * 64
    bindings["environment_auditor"]["file_sha256"] = historical_sha

    receipt_path = _rebind_recovery_chain(
        chain,
        tmp_path / "historical-rebound",
        authorization_updates={"executable_bindings": bindings},
    )
    result = release._validate_cleanup_chain(
        precleanup_path=chain["precleanup_path"],
        plan_path=chain["plan_path"],
        authorization_path=chain["authorization_path"],
        receipt_path=receipt_path,
    )
    assert (
        result["recovery_authorization"]["executable_bindings"][
            "environment_auditor"
        ]["file_sha256"]
        == historical_sha
    )


@pytest.mark.parametrize("binding_name", ["cleanup_tool", "python", "systemctl"])
def test_cleanup_release_chain_rejects_nonhistorical_executable_sha_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding_name: str,
) -> None:
    chain = _sealed_recovery_cleanup(tmp_path, monkeypatch)
    lineage = chain["receipt"]["partial_lineage"]
    authorization = release.cleanup.load_sealed_json(
        lineage["recovery_authorization"]["path"]
    )
    bindings = deepcopy(authorization["executable_bindings"])
    current_sha = bindings[binding_name]["file_sha256"]
    changed_sha = "0" * 64 if current_sha != "0" * 64 else "1" * 64
    bindings[binding_name]["file_sha256"] = changed_sha

    receipt_path = _rebind_recovery_chain(
        chain,
        tmp_path / f"{binding_name}-rebound",
        authorization_updates={"executable_bindings": bindings},
    )
    with pytest.raises(
        PermissionError,
        match=rf"archived recovery executable changed:{binding_name}",
    ):
        release._validate_cleanup_chain(
            precleanup_path=chain["precleanup_path"],
            plan_path=chain["plan_path"],
            authorization_path=chain["authorization_path"],
            receipt_path=receipt_path,
        )


def test_build_spec_is_create_only_private_and_exact_actual_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_release_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(release.supervisor, "_ACTUAL_SPEC_PATH", str(release.RUNTIME_SPEC_PATH))
    monkeypatch.setattr(
        release.supervisor, "_ACTUAL_RUNTIME_LAUNCH_AUTHORIZATION_PATH",
        str(release.RUNTIME_LAUNCH_AUTHORIZATION_PATH),
    )
    monkeypatch.setattr(
        release.supervisor, "_ACTUAL_SCIENTIFIC_AUTHORIZATION_PATH",
        str(release.SCIENTIFIC_AUTHORIZATION_PATH),
    )
    monkeypatch.setattr(
        release.supervisor, "_ACTUAL_SCIENTIFIC_ACCESS_AUDIT_PATH",
        str(release.SCIENTIFIC_ACCESS_AUDIT_PATH),
    )
    _create_scientific_and_prior_files()
    inputs = _input_files(tmp_path)
    closure = _fake_closure(tmp_path)
    calls = 0

    def validate(**_kwargs):
        nonlocal calls
        calls += 1
        return deepcopy(closure)

    monkeypatch.setattr(release, "validate_release_closure", validate)
    spec = release.build_spec(
        **inputs, shadow_reader=lambda: {}, manager_reader=lambda: {},
        summary_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "{}", ""),
    )
    assert calls == 2
    assert stat.S_IMODE(release.RUNTIME_SPEC_PATH.stat().st_mode) == 0o444
    for path in (
        release.RUNTIME_ARTIFACT_ROOT,
        release.RUNTIME_ARTIFACT_ROOT / "heartbeat",
        release.RUNTIME_ARTIFACT_ROOT / "systemd-invocations",
        release.GPU_LEASE_ROOT,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert spec["execution_kind"] == release.supervisor.ACTUAL_EXECUTION_KIND
    assert spec["attempt_ordinal"] == 2 and spec["prior_attempt_count"] == 1
    assert spec["authorization"]["path"] == str(release.RUNTIME_LAUNCH_AUTHORIZATION_PATH)
    assert spec["child"]["argv"] == [
        str(release.PYTHON_PATH), "-I", "-S", "-B", "-u",
        str(release.ADAPTER_PATH), "real",
        "--execute-real-dr", "--device", "cuda:0",
        "--runtime-launch-authorization", str(release.RUNTIME_LAUNCH_AUTHORIZATION_PATH),
    ]
    assert spec["child"]["environment"]["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
    source = spec["source_bindings"]
    python_metadata = release.PYTHON_PATH.stat()
    assert source["python_path"] == "/usr/bin/python3.12"
    assert source["python_device"] == python_metadata.st_dev
    assert source["python_inode"] == python_metadata.st_ino
    assert source["python_owner_uid"] == 0
    assert source["python_mode"] == 0o755
    assert source["runtime_dependency_site_path"] == str(
        release.RUNTIME_DEPENDENCY_SITE_PATH
    )
    assert source["runtime_dependency_site_device"] == 2304
    assert source["runtime_dependency_site_inode"] == 228331323
    assert source["runtime_dependency_site_owner_uid"] == os.getuid() == 1008
    assert source["runtime_dependency_site_mode"] == 0o775
    assert spec["runtime"]["launch_limit"] == 1
    assert spec["runtime"]["automatic_retry_allowed"] is False
    assert (
        release.supervisor.load_runtime_spec(release.RUNTIME_SPEC_PATH)[
            "runtime_spec_fingerprint"
        ]
        == spec["runtime_spec_fingerprint"]
    )
    artifacts = spec["artifacts"]
    directory_keys = {"root", "heartbeat_dir", "systemd_invocation_dir"}
    assert all(
        not os.path.lexists(Path(value))
        for key, value in artifacts.items() if key not in directory_keys
    )
    assert not (release.GPU_LEASE_ROOT / "active.json").exists()
    with pytest.raises(FileExistsError):
        release.build_spec(
            **inputs, shadow_reader=lambda: {}, manager_reader=lambda: {},
            summary_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "{}", ""),
        )


def _minimal_spec_for_authorize(
    realization: dict[str, object],
) -> dict[str, object]:
    root = release.RUNTIME_ARTIFACT_ROOT
    artifacts = release._artifact_paths()
    scientific_authorization, scientific_authorization_identity = (
        release._load_sealed(
            release.SCIENTIFIC_AUTHORIZATION_PATH,
            fingerprint_field="authorization_fingerprint",
            return_identity=True,
        )
    )
    scientific_audit, scientific_audit_identity = release._load_sealed(
        release.SCIENTIFIC_ACCESS_AUDIT_PATH,
        fingerprint_field="receipt_fingerprint",
        return_identity=True,
    )
    body = {
        "schema_version": release.supervisor.RUNTIME_SPEC_SCHEMA,
        "candidate": release.CANDIDATE, "stage_id": release.STAGE_ID,
        "attempt_id": release.ATTEMPT_ID,
        "scientific_preaccess": {
            "authorization_file_sha256":
                scientific_authorization_identity["file_sha256"],
            "authorization_fingerprint":
                scientific_authorization["authorization_fingerprint"],
            "access_audit_file_sha256":
                scientific_audit_identity["file_sha256"],
            "access_audit_fingerprint":
                scientific_audit["receipt_fingerprint"],
        },
        "environment": {
            "gpu_lease_path": str(release.GPU_LEASE_ROOT / "active.json"),
            "gpu_lease_tombstone_path": str(release.GPU_LEASE_ROOT / "released.json"),
        },
        "artifacts": artifacts,
        "source_bindings": release._collect_source_bindings(realization),
        "runtime": {"systemd": {"unit_fragment_file_sha256": "4" * 64}},
    }
    return {
        **body,
        "runtime_spec_fingerprint": release.stable_fingerprint(body),
    }


def test_environment_authorization_bindings_delegate_to_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {"environment": {"generated": "contract"}}
    expected = {
        "environment_policy_fingerprint": "5" * 64,
        "selected_gpu_uuid": "GPU-generated",
    }
    observed: list[object] = []

    def verify(value: object) -> dict[str, object]:
        observed.append(value)
        return expected

    monkeypatch.setattr(
        release.supervisor,
        "_environment_evidence_bindings",
        verify,
    )
    assert release._environment_authorization_bindings(spec) == expected
    assert observed == [spec]


def test_authorize_launch_reloads_spec_rechecks_closure_and_writes_intended_freshness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_release_paths(monkeypatch, tmp_path)
    _create_scientific_and_prior_files()
    artifacts = release._artifact_paths()
    release._create_runtime_directories_and_verify_leaves(artifacts)
    closure = _fake_closure(tmp_path)
    spec = _minimal_spec_for_authorize(closure["realization"])
    release.RUNTIME_SPEC_PATH.write_text(release.canonical_json(spec) + "\n", encoding="utf-8")
    release.RUNTIME_SPEC_PATH.chmod(0o444)
    loaded = 0
    rechecked = 0

    def validate_spec(value, *, loaded_spec_path):
        nonlocal loaded
        loaded += 1
        assert value == spec
        assert loaded_spec_path == release.RUNTIME_SPEC_PATH

    def validate(**_kwargs):
        nonlocal rechecked
        rechecked += 1
        return deepcopy(closure)

    monkeypatch.setattr(
        release.supervisor, "_validate_spec_structure", validate_spec,
    )
    monkeypatch.setattr(release, "_release_inputs_from_spec", lambda _spec: {})
    monkeypatch.setattr(release, "validate_release_closure", validate)
    env_bindings = {
        "environment_policy_fingerprint": "5" * 64,
        "environment_inventory_receipt_fingerprint": "6" * 64,
    }
    monkeypatch.setattr(release, "_environment_authorization_bindings", lambda _spec: env_bindings)
    scientific = {
        "scientific_preaccess_authorization_path": str(release.SCIENTIFIC_AUTHORIZATION_PATH),
        "scientific_preaccess_authorization_file_sha256": release.file_sha256(release.SCIENTIFIC_AUTHORIZATION_PATH),
        "scientific_preaccess_authorization_fingerprint":
            spec["scientific_preaccess"]["authorization_fingerprint"],
        "scientific_preaccess_access_audit_path": str(release.SCIENTIFIC_ACCESS_AUDIT_PATH),
        "scientific_preaccess_access_audit_file_sha256": release.file_sha256(release.SCIENTIFIC_ACCESS_AUDIT_PATH),
        "scientific_preaccess_access_audit_fingerprint":
            spec["scientific_preaccess"]["access_audit_fingerprint"],
        "r2_adapter_path": str(release.ADAPTER_PATH),
        "r2_adapter_file_sha256": release.file_sha256(release.ADAPTER_PATH),
        "legacy_gate_entrypoint_path": str(release.LEGACY_GATE_PATH),
        "legacy_gate_entrypoint_file_sha256": release.file_sha256(release.LEGACY_GATE_PATH),
        "source_closure_fingerprint_103": release.SOURCE_CLOSURE_FINGERPRINT_103,
    }
    monkeypatch.setattr(release.supervisor, "_verify_scientific_preaccess_bindings", lambda _spec: scientific)
    consumer_checks = 0

    def verify_consumer(
        value: object,
        *,
        spec_path: Path,
        require_fresh: bool,
    ) -> dict[str, object]:
        nonlocal consumer_checks
        consumer_checks += 1
        assert value == spec
        assert spec_path == release.RUNTIME_SPEC_PATH
        assert require_fresh is True
        payload = release._read_json(
            release.RUNTIME_LAUNCH_AUTHORIZATION_PATH
        )
        return {
            "path": str(release.RUNTIME_LAUNCH_AUTHORIZATION_PATH),
            "authorization_fingerprint": payload[
                "authorization_fingerprint"
            ],
            "authorization_file_sha256": release.file_sha256(
                release.RUNTIME_LAUNCH_AUTHORIZATION_PATH
            ),
        }

    monkeypatch.setattr(
        release.supervisor,
        "_verify_actual_authorization",
        verify_consumer,
    )
    authorization = release.authorize_launch(
        shadow_reader=lambda: {}, manager_reader=lambda: {}, validity_seconds=120,
    )
    assert loaded == 1 and rechecked == 1 and consumer_checks == 1
    assert stat.S_IMODE(release.RUNTIME_LAUNCH_AUTHORIZATION_PATH.stat().st_mode) == 0o444
    issued = datetime.fromisoformat(authorization["issued_at_utc"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(authorization["expires_at_utc"].replace("Z", "+00:00"))
    assert expires - issued == timedelta(seconds=120)
    assert authorization["instruction_id"] == release.INSTRUCTION_ID
    assert authorization["authorization_basis"] == release.AUTHORIZATION_BASIS
    assert authorization["authorized_uid"] == os.getuid()
    assert authorization["D_R_payload_authorized"] is True
    for key in ("D_V_payload_authorized", "D_T_payload_authorized", "training_authorized", "resume_allowed", "automatic_retry_allowed"):
        assert authorization[key] is False
    with pytest.raises(FileExistsError):
        release.authorize_launch(shadow_reader=lambda: {}, manager_reader=lambda: {})


def test_authorize_launch_rejects_more_than_300_seconds_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_release_paths(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=r"\[1,300\]"):
        release.authorize_launch(
            shadow_reader=lambda: {}, manager_reader=lambda: {},
            validity_seconds=301,
        )
    assert not os.path.lexists(release.RUNTIME_LAUNCH_AUTHORIZATION_PATH)


def _write_integration_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    _patch_release_paths(monkeypatch, tmp_path)
    control = tmp_path / "integration-control"
    control.mkdir(mode=0o700)
    now = datetime.now(timezone.utc)
    scenario_id = "scenario-0123456789abcdef"
    identity = release.integration.build_supervisor_v2_identity(scenario_id)
    runtime_spec_fingerprint = "1" * 64
    rendered_sha256 = "2" * 64
    uid = os.getuid()
    unit_directory = tmp_path / "integration-units"
    manager_generation = {
        "boot_id": "12345678-1234-1234-1234-123456789abc",
        "identity": {
            "pid": 111, "starttime_ticks": 222, "uid": uid,
            "control_group":
                f"/user.slice/user-{uid}.slice/user@{uid}.service/init.scope",
        },
        "endpoint": {
            "uid": uid, "runtime_directory": f"/run/user/{uid}",
            "runtime_device": 11, "runtime_inode": 12,
            "bus_path": f"/run/user/{uid}/bus",
            "bus_device": 13, "bus_inode": 14,
        },
    }
    unit_path_policy = {
        "runtime_directory": str(unit_directory),
        "ordered_unit_paths": [],
    }
    paths = {
        "integration_terminal": str(control / "terminal.json"),
        "removal_authorization": str(control / "removal-auth.json"),
        "removal_state": str(control / "removal-state.json"),
        "integration_receipt": str(control / "receipt.json"),
    }
    auth_path = control / "authorization.json"
    auth_body = {
        key: None for key in release.integration._AUTH_KEYS
        if key != "authorization_fingerprint"
    }
    auth_body.update({
        "schema_version": release.integration.AUTHORIZATION_SCHEMA,
        "instruction_id": release.integration.INSTRUCTION_ID,
        "authorization_basis": release.integration.AUTHORIZATION_BASIS,
        "authorized_uid": uid,
        "issued_at_utc": now.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (now + timedelta(seconds=300)).isoformat().replace("+00:00", "Z"),
        "integration_authorized": True, "actual_r2_authorized": False,
        "unit_realization_authorized": True,
        "unit_removal_authorized": False,
        "direct_start_authorized": False, "enable_authorized": False,
        "scenario_id": scenario_id, "identity": identity,
        "unit_directory": str(unit_directory),
        "unit_path_policy": unit_path_policy,
        "manager_generation": manager_generation,
        "rendered_fragment": {
            "utf8_text": "[Service]\n", "sha256": rendered_sha256,
        },
        "runtime_spec_binding": {
            "runtime_spec_fingerprint": runtime_spec_fingerprint,
        },
        "control_artifacts": paths, "D_R_payload_accessed": False,
        "D_V_payload_accessed": False, "D_T_payload_accessed": False,
        "payload_authority": "none", "gpu_access_authorized": False,
    })
    auth = _seal(auth_path, auth_body, "authorization_fingerprint")
    evidence = {
        "invocation_id": "a" * 32,
        "attempt_commit_fingerprint": "3" * 64,
        "launch_lease_fingerprint": "4" * 64,
        "materialization_claim_fingerprint": "5" * 64,
        "precommit_fingerprint": "6" * 64,
        "start_ack_fingerprint": "7" * 64,
        "child_prespawn_fingerprint": "8" * 64,
        "runtime_terminal_fingerprint": "9" * 64,
        "systemd_terminal_fingerprint": "a" * 64,
        "runtime_attestation_absent": True,
        "gpu_lease_evidence_absent": True,
    }
    terminal = _seal(Path(paths["integration_terminal"]), {
        "schema_version": release.integration.INTEGRATION_TERMINAL_SCHEMA,
        "scenario_id": scenario_id, "identity": identity,
        "authorization_fingerprint": auth["authorization_fingerprint"],
        "runtime_spec_fingerprint": runtime_spec_fingerprint,
        "created_at_utc": now.isoformat().replace("+00:00", "Z"),
        "supervisor_evidence": evidence, "passed": True,
        "completed_actions":
            deepcopy(release._INTEGRATION_TERMINAL_ACTIONS),
        "error_type": None, "error_message": None,
        "direct_systemctl_start_attempted": False,
        "enable_attempted": False, "remove_attempted": False,
        "payload_authority": "none", "gpu_accessed": False,
        "D_R_payload_accessed": False, "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }, "integration_terminal_fingerprint")
    removal_auth_body = {
        key: None for key in release.integration._REMOVAL_AUTH_KEYS
        if key != "removal_authorization_fingerprint"
    }
    removal_auth_body.update({
        "schema_version": release.integration.REMOVAL_AUTHORIZATION_SCHEMA,
        "scenario_id": scenario_id, "unit_name": identity["unit_name"],
        "authorization_fingerprint": auth["authorization_fingerprint"],
        "integration_terminal_fingerprint":
            terminal["integration_terminal_fingerprint"],
        "runtime_spec_fingerprint": runtime_spec_fingerprint,
        "supervisor_evidence": evidence,
        "fragment_identity": {
            "fragment_path": str(unit_directory / identity["unit_name"]),
            "fragment_sha256": rendered_sha256,
            "device": 21, "inode": 22, "owner_uid": uid,
            "mode": 0o600, "nlink": 1,
        },
        "inactive_static_state": {
            "LoadState": "loaded", "UnitFileState": "static",
            "ActiveState": "inactive", "SubState": "dead",
            "FragmentPath": str(unit_directory / identity["unit_name"]),
            "DropInPaths": "", "Transient": "no", "Restart": "no",
            "NRestarts": "0", "NeedDaemonReload": "no",
        },
        "manager_generation": manager_generation,
        "unit_path_policy": unit_path_policy,
        "issued_at_utc": now.isoformat().replace("+00:00", "Z"),
        "expires_at_utc":
            (now + timedelta(seconds=300)).isoformat().replace("+00:00", "Z"),
        "remove_authorized": True, "daemon_reload_authorized": True,
        "not_found_verification_authorized": True,
        "enable_authorized": False, "start_authorized": False,
        "D_R_payload_accessed": False, "D_V_payload_accessed": False,
        "D_T_payload_accessed": False, "payload_authority": "none",
    })
    removal_auth = _seal(
        Path(paths["removal_authorization"]), removal_auth_body,
        "removal_authorization_fingerprint",
    )
    removal = _seal(Path(paths["removal_state"]), {
        "schema_version": release.integration.REMOVAL_STATE_SCHEMA,
        "scenario_id": scenario_id, "unit_name": identity["unit_name"],
        "removal_authorization_fingerprint":
            removal_auth["removal_authorization_fingerprint"],
        "passed": True, "remove_attempted": True, "fragment_absent": True,
        "not_found_state": deepcopy(release._INTEGRATION_NOT_FOUND_STATE),
        "completed_actions":
            deepcopy(release._INTEGRATION_REMOVAL_ACTIONS),
        "error_type": None, "error_message": None,
        "payload_authority": "none",
        "D_R_payload_accessed": False, "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }, "removal_state_fingerprint")
    receipt_body = {
        "schema_version": release.integration.INTEGRATION_RECEIPT_SCHEMA,
        "scenario_id": scenario_id, "identity": identity,
        "authorization_path": str(auth_path), "authorization_file_sha256": release.file_sha256(auth_path),
        "authorization_fingerprint": auth["authorization_fingerprint"],
        "integration_terminal_path": paths["integration_terminal"],
        "integration_terminal_file_sha256": release.file_sha256(paths["integration_terminal"]),
        "integration_terminal_fingerprint": terminal["integration_terminal_fingerprint"],
        "removal_authorization_path": paths["removal_authorization"],
        "removal_authorization_file_sha256": release.file_sha256(paths["removal_authorization"]),
        "removal_authorization_fingerprint": removal_auth["removal_authorization_fingerprint"],
        "removal_state_path": paths["removal_state"],
        "removal_state_file_sha256": release.file_sha256(paths["removal_state"]),
        "removal_state_fingerprint": removal["removal_state_fingerprint"],
        "supervisor_evidence": evidence, "fragment_removed": True, "passed": True,
        "post_removal_unit_state":
            deepcopy(release._INTEGRATION_NOT_FOUND_STATE),
        "payload_authority": "none",
        "gpu_accessed": False,
        "D_R_payload_accessed": False, "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    receipt_path = Path(paths["integration_receipt"])
    receipt = _seal(receipt_path, receipt_body, "receipt_fingerprint")
    return {
        "authorization_path": auth_path,
        "receipt_path": receipt_path,
        "receipt_body": receipt_body,
        "receipt": receipt,
        "paths": paths,
    }


def test_integration_final_wrapper_is_required_and_lineage_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _write_integration_chain(tmp_path, monkeypatch)
    result = release._validate_integration_chain(
        authorization_path=chain["authorization_path"],
        receipt_path=chain["receipt_path"],
    )
    assert (
        result["receipt"]["receipt_fingerprint"]
        == chain["receipt"]["receipt_fingerprint"]
    )


@pytest.mark.parametrize(
    ("field", "drift"),
    [
        ("scenario_id", "other-scenario-fedcba9876543210"),
        ("identity", {"unit_name": "drifted.service"}),
        ("removal_authorization_path", "/tmp/drifted-removal-auth.json"),
        ("removal_authorization_file_sha256", "0" * 64),
        ("removal_state_path", "/tmp/drifted-removal-state.json"),
        ("removal_state_file_sha256", "0" * 64),
        ("post_removal_unit_state", {"LoadState": "loaded"}),
        ("payload_authority", "scientific-payload"),
        ("gpu_accessed", True),
        ("authorization_fingerprint", "0" * 64),
        ("integration_terminal_fingerprint", "0" * 64),
        ("removal_authorization_fingerprint", "0" * 64),
        ("removal_state_fingerprint", "0" * 64),
    ],
)
def test_integration_receipt_resealed_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    field: str, drift: object,
) -> None:
    chain = _write_integration_chain(tmp_path, monkeypatch)
    receipt_path = chain["receipt_path"]
    receipt_path.unlink()
    bad = deepcopy(chain["receipt_body"])
    bad[field] = drift
    resealed = _seal(receipt_path, bad, "receipt_fingerprint")
    assert (
        resealed["receipt_fingerprint"]
        == release.stable_fingerprint(bad)
    )
    assert (
        resealed["receipt_fingerprint"]
        != chain["receipt"]["receipt_fingerprint"]
    )
    with pytest.raises(PermissionError, match="complete PASS"):
        release._validate_integration_chain(
            authorization_path=chain["authorization_path"],
            receipt_path=receipt_path,
        )


def _set_nested_drift(
    payload: dict[str, object], path: tuple[str, ...], drift: object,
) -> None:
    target = payload
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = deepcopy(drift)


def _coordinated_reseal_integration_children(
    chain: dict[str, object], *, target: str,
    field_path: tuple[str, ...], drift: object,
) -> None:
    paths = chain["paths"]
    assert isinstance(paths, dict)
    authorization_path = chain["authorization_path"]
    terminal_path = Path(paths["integration_terminal"])
    removal_authorization_path = Path(paths["removal_authorization"])
    removal_state_path = Path(paths["removal_state"])
    receipt_path = chain["receipt_path"]

    authorization = release._load_sealed(
        authorization_path,
        fingerprint_field="authorization_fingerprint",
    )
    if target == "authorization":
        _set_nested_drift(authorization, field_path, drift)
        authorization = _replace_sealed(
            authorization_path, authorization, "authorization_fingerprint",
        )

    terminal = release._load_sealed(
        terminal_path,
        fingerprint_field="integration_terminal_fingerprint",
    )
    if target == "terminal":
        _set_nested_drift(terminal, field_path, drift)
    terminal["authorization_fingerprint"] = authorization[
        "authorization_fingerprint"
    ]
    terminal = _replace_sealed(
        terminal_path, terminal, "integration_terminal_fingerprint",
    )

    removal_authorization = release._load_sealed(
        removal_authorization_path,
        fingerprint_field="removal_authorization_fingerprint",
    )
    removal_authorization["integration_terminal_fingerprint"] = terminal[
        "integration_terminal_fingerprint"
    ]
    removal_authorization["authorization_fingerprint"] = authorization[
        "authorization_fingerprint"
    ]
    if target == "authorization" and field_path[:1] == (
        "manager_generation",
    ):
        removal_authorization["manager_generation"] = deepcopy(
            authorization["manager_generation"]
        )
    removal_authorization["supervisor_evidence"] = deepcopy(
        terminal["supervisor_evidence"]
    )
    if target == "removal_authorization":
        _set_nested_drift(removal_authorization, field_path, drift)
    removal_authorization = _replace_sealed(
        removal_authorization_path, removal_authorization,
        "removal_authorization_fingerprint",
    )

    removal_state = release._load_sealed(
        removal_state_path,
        fingerprint_field="removal_state_fingerprint",
    )
    removal_state["removal_authorization_fingerprint"] = (
        removal_authorization["removal_authorization_fingerprint"]
    )
    if target == "terminal" and field_path == ("completed_actions",):
        removal_state["completed_actions"] = [
            *terminal["completed_actions"],
            "remove-authorized-fragment",
            "daemon-reload-after-removal",
        ]
    if target == "removal_state":
        _set_nested_drift(removal_state, field_path, drift)
    removal_state = _replace_sealed(
        removal_state_path, removal_state, "removal_state_fingerprint",
    )

    receipt = release._load_sealed(
        receipt_path, fingerprint_field="receipt_fingerprint",
    )
    receipt.update({
        "authorization_file_sha256":
            release.file_sha256(authorization_path),
        "authorization_fingerprint":
            authorization["authorization_fingerprint"],
        "integration_terminal_file_sha256":
            release.file_sha256(terminal_path),
        "integration_terminal_fingerprint":
            terminal["integration_terminal_fingerprint"],
        "removal_authorization_file_sha256":
            release.file_sha256(removal_authorization_path),
        "removal_authorization_fingerprint":
            removal_authorization["removal_authorization_fingerprint"],
        "removal_state_file_sha256":
            release.file_sha256(removal_state_path),
        "removal_state_fingerprint":
            removal_state["removal_state_fingerprint"],
        "supervisor_evidence": deepcopy(terminal["supervisor_evidence"]),
        "post_removal_unit_state":
            deepcopy(removal_state["not_found_state"]),
    })
    _replace_sealed(receipt_path, receipt, "receipt_fingerprint")


@pytest.mark.parametrize(
    ("target", "field_path", "drift"),
    [
        (
            "authorization",
            ("manager_generation", "identity", "uid"),
            float(os.getuid()),
        ),
        (
            "authorization",
            ("manager_generation", "endpoint", "uid"),
            float(os.getuid()),
        ),
        ("terminal", ("passed",), False),
        ("terminal", ("direct_systemctl_start_attempted",), True),
        ("terminal", ("enable_attempted",), True),
        ("terminal", ("remove_attempted",), True),
        ("terminal", ("payload_authority",), "scientific-payload"),
        ("terminal", ("D_R_payload_accessed",), True),
        ("terminal", ("D_V_payload_accessed",), True),
        ("terminal", ("D_T_payload_accessed",), True),
        ("terminal", ("gpu_accessed",), True),
        (
            "terminal", ("completed_actions",),
            release._INTEGRATION_TERMINAL_ACTIONS[:-1],
        ),
        (
            "terminal", ("completed_actions",),
            list(reversed(release._INTEGRATION_TERMINAL_ACTIONS)),
        ),
        (
            "terminal", ("completed_actions",),
            [*release._INTEGRATION_TERMINAL_ACTIONS, "extra-action"],
        ),
        (
            "terminal",
            ("supervisor_evidence", "runtime_attestation_absent"), False,
        ),
        (
            "terminal",
            ("supervisor_evidence", "gpu_lease_evidence_absent"), False,
        ),
        ("terminal", ("created_at_utc",), "1970-01-01T00:00:00Z"),
        ("removal_authorization", ("remove_authorized",), False),
        ("removal_authorization", ("daemon_reload_authorized",), False),
        (
            "removal_authorization",
            ("not_found_verification_authorized",), False,
        ),
        ("removal_authorization", ("enable_authorized",), True),
        ("removal_authorization", ("start_authorized",), True),
        (
            "removal_authorization", ("payload_authority",),
            "scientific-payload",
        ),
        ("removal_authorization", ("D_R_payload_accessed",), True),
        ("removal_authorization", ("D_V_payload_accessed",), True),
        ("removal_authorization", ("D_T_payload_accessed",), True),
        (
            "removal_authorization", ("fragment_identity", "device"), True,
        ),
        (
            "removal_authorization", ("fragment_identity", "inode"), True,
        ),
        (
            "removal_authorization",
            ("fragment_identity", "owner_uid"), True,
        ),
        (
            "removal_authorization", ("fragment_identity", "mode"), 384.0,
        ),
        (
            "removal_authorization", ("fragment_identity", "nlink"), True,
        ),
        (
            "removal_authorization",
            ("inactive_static_state", "ActiveState"), "active",
        ),
        (
            "removal_authorization",
            ("manager_generation", "identity", "pid"), 111.0,
        ),
        (
            "removal_authorization", ("issued_at_utc",),
            "1970-01-01T00:00:00Z",
        ),
        ("removal_state", ("passed",), False),
        ("removal_state", ("remove_attempted",), False),
        ("removal_state", ("fragment_absent",), False),
        ("removal_state", ("payload_authority",), "scientific-payload"),
        ("removal_state", ("D_R_payload_accessed",), True),
        ("removal_state", ("D_V_payload_accessed",), True),
        ("removal_state", ("D_T_payload_accessed",), True),
        (
            "removal_state", ("completed_actions",),
            release._INTEGRATION_REMOVAL_ACTIONS[:-1],
        ),
        (
            "removal_state", ("completed_actions",),
            list(reversed(release._INTEGRATION_REMOVAL_ACTIONS)),
        ),
        (
            "removal_state", ("completed_actions",),
            [*release._INTEGRATION_REMOVAL_ACTIONS, "extra-action"],
        ),
        (
            "removal_state", ("not_found_state", "LoadState"), "loaded",
        ),
    ],
)
def test_integration_coordinated_resealed_child_safety_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target: str, field_path: tuple[str, ...], drift: object,
) -> None:
    chain = _write_integration_chain(tmp_path, monkeypatch)
    _coordinated_reseal_integration_children(
        chain, target=target, field_path=field_path, drift=drift,
    )
    with pytest.raises(PermissionError):
        release._validate_integration_chain(
            authorization_path=chain["authorization_path"],
            receipt_path=chain["receipt_path"],
        )


def test_identity_summary_rejects_103_closure_drift() -> None:
    body = {
        "schema_version": "cure-lite-v24-D_R-structural-r2-execution-identity-v1",
        "attempt_ordinal": 2, "prior_attempt_count": 1,
        "prior_attempt_status": "OBSERVABILITY_LOST_NO_AUTHENTICATED_DECISION",
        "frozen_scientific_path_count": 103,
        "frozen_scientific_source_closure_fingerprint": release.SOURCE_CLOSURE_FINGERPRINT_103,
        "numerical_or_scientific_change_authorized": False,
        "D_R_payload_authorized_by_adapter": False,
        "D_V_payload_authorized": False, "D_T_payload_authorized": False,
        "training_authorized": False, "optimizer_steps_authorized": 0,
        "parameter_updates_authorized": 0, "automatic_retry_allowed": False,
        "resume_allowed": False,
    }
    payload = {**body, "execution_identity_fingerprint": release.stable_fingerprint(body)}
    runner = lambda argv: subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
    assert release._validate_identity_summary(runner)["frozen_scientific_path_count"] == 103
    drifted_body = dict(body)
    drifted_body["frozen_scientific_source_closure_fingerprint"] = "0" * 64
    drifted = {**drifted_body, "execution_identity_fingerprint": release.stable_fingerprint(drifted_body)}
    with pytest.raises(PermissionError, match="103 closure"):
        release._validate_identity_summary(
            lambda argv: subprocess.CompletedProcess(argv, 0, json.dumps(drifted), "")
        )


def test_audit_payload_access_is_rejected(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    inventory_body = {
        "passed": True, "blockers": [], "D_R_payload_accessed": False,
        "D_V_payload_accessed": False, "D_T_payload_accessed": False,
    }
    inventory = {**inventory_body, "inventory_fingerprint": release.stable_fingerprint(inventory_body)}
    path = tmp_path / "post.json"
    _seal(path, {
        "schema_version": release.runtime_environment.ENVIRONMENT_RECEIPT_SCHEMA,
        "created_at_utc": "2026-07-30T00:00:00Z",
        "command": "audit-only", "passed": True, "inventory": inventory,
        "environment_binding": {}, "error_type": None, "error_message": None,
        "D_R_payload_accessed": True, "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }, "receipt_fingerprint")
    with pytest.raises(PermissionError, match="accessed"):
        release._validate_audit_receipt(path, passed=True)


def _mock_environment_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    postcleanup_inventory_source: str,
    binding_device: object = 54,
) -> dict[str, object]:
    paths = {
        name: (tmp_path / f"{name}.json").resolve()
        for name in (
            "policy", "precleanup", "cleanup", "stability",
            "postcleanup",
        )
    }
    identity = {
        name: {"file_sha256": character * 64}
        for name, character in (
            ("policy", "1"),
            ("precleanup", "2"),
            ("cleanup", "3"),
            ("stability", "4"),
            ("postcleanup", "5"),
        )
    }
    manager_identity = {
        "pid": 123,
        "starttime_ticks": 456,
        "uid": os.getuid(),
        "control_group": (
            f"/user.slice/user-{os.getuid()}.slice/"
            f"user@{os.getuid()}.service/init.scope"
        ),
    }
    endpoint = {
        "uid": os.getuid(),
        "runtime_directory": f"/run/user/{os.getuid()}",
        "runtime_directory_device": 54,
        "runtime_directory_inode": 1,
        "bus_path": f"/run/user/{os.getuid()}/bus",
        "bus_device": 54,
        "bus_inode": 61,
    }

    def inventory(label: str) -> dict[str, object]:
        body = {
            "schema_version": (
                "cure-lite-v24-runtime-environment-inventory-v1"
            ),
            "created_at_utc": f"2026-07-30T00:00:0{label}Z",
            "uid": os.getuid(),
            "boot_id": "12345678-1234-1234-1234-123456789abc",
            "manager": {
                "identity": deepcopy(manager_identity),
                "endpoint": deepcopy(endpoint),
            },
            "blockers": [],
            "passed": True,
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        return {
            **body,
            "inventory_fingerprint": release.stable_fingerprint(body),
        }

    first_inventory = inventory("1")
    final_inventory = inventory("2")
    selected_inventory = (
        final_inventory
        if postcleanup_inventory_source == "final"
        else first_inventory
    )
    expected_binding = release._environment_binding_from_inventory(
        selected_inventory
    )
    expected_binding["runtime_directory_device"] = binding_device
    postcleanup = {
        "inventory": deepcopy(selected_inventory),
        "environment_binding": expected_binding,
    }
    policy = {
        "selected_gpu": {
            "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "pci_bus_id": "00000000:02:00.0",
            "minor_number": 0,
        },
        "policy_fingerprint": "a" * 64,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    precleanup = {"receipt_fingerprint": "b" * 64}
    cleanup_receipt = {"cleanup_receipt_fingerprint": "c" * 64}
    contract = {
        "selected_gpu_uuid": policy["selected_gpu"]["uuid"],
        "selected_gpu_pci_bus_id": policy["selected_gpu"]["pci_bus_id"],
        "selected_gpu_minor_number": 0,
        "boot_id": final_inventory["boot_id"],
        "manager_pid": manager_identity["pid"],
        "manager_starttime_ticks": manager_identity["starttime_ticks"],
        "uid": manager_identity["uid"],
        "manager_control_group": manager_identity["control_group"],
    }
    stability = {
        "receipt_kind": "sampled",
        "sample_count": 2,
        "samples": [
            {
                "inventory": deepcopy(first_inventory),
                "passed": True,
                "blockers": [],
            },
            {
                "inventory": deepcopy(final_inventory),
                "passed": True,
                "blockers": [],
            },
        ],
        "contract": contract,
        "root_evidence": {
            "precleanup_inventory_receipt": {
                "file_sha256": identity["precleanup"]["file_sha256"],
                "receipt_fingerprint": precleanup[
                    "receipt_fingerprint"
                ],
            },
            "cleanup_receipt": {
                "file_sha256": identity["cleanup"]["file_sha256"],
                "cleanup_receipt_fingerprint": cleanup_receipt[
                    "cleanup_receipt_fingerprint"
                ],
            },
            "policy": {
                "file_sha256": identity["policy"]["file_sha256"],
                "policy_fingerprint": policy["policy_fingerprint"],
            },
        },
        "passed": True,
        "blockers": [],
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    loaded = {
        paths["policy"]: (policy, identity["policy"]),
        paths["stability"]: (stability, identity["stability"]),
        paths["cleanup"]: (cleanup_receipt, identity["cleanup"]),
    }

    def fake_load_sealed(
        path: Path,
        *,
        fingerprint_field: str,
        schema: str,
        return_identity: bool = False,
    ) -> object:
        payload, file_identity = loaded[Path(path)]
        if return_identity:
            return payload, file_identity
        return payload

    def fake_validate_audit(
        path: Path,
        *,
        passed: bool,
        return_identity: bool = False,
    ) -> object:
        if Path(path) == paths["precleanup"]:
            payload = precleanup
            file_identity = identity["precleanup"]
        else:
            payload = postcleanup
            file_identity = identity["postcleanup"]
        if return_identity:
            return payload, file_identity
        return payload

    monkeypatch.setattr(release, "_load_sealed", fake_load_sealed)
    monkeypatch.setattr(
        release,
        "_validate_audit_receipt",
        fake_validate_audit,
    )
    monkeypatch.setattr(
        release.runtime_environment,
        "validate_environment_policy",
        lambda value: value,
    )
    monkeypatch.setattr(
        release.runtime_environment,
        "validate_environment_stability_receipt",
        lambda value: value,
    )
    return release._validate_environment_chain(
        policy_path=paths["policy"],
        precleanup_path=paths["precleanup"],
        cleanup_receipt_path=paths["cleanup"],
        stability_path=paths["stability"],
        postcleanup_path=paths["postcleanup"],
    )


def test_environment_chain_accepts_exact_final_stability_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _mock_environment_chain(
        tmp_path,
        monkeypatch,
        postcleanup_inventory_source="final",
    )
    assert validated["postcleanup"]["inventory"] == (
        validated["stability"]["samples"][-1]["inventory"]
    )


def test_environment_chain_rejects_semantically_valid_nonfinal_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        PermissionError,
        match="exact final stability sample",
    ):
        _mock_environment_chain(
            tmp_path,
            monkeypatch,
            postcleanup_inventory_source="first",
        )


def test_environment_chain_binding_comparison_is_type_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        PermissionError,
        match="binding differs",
    ):
        _mock_environment_chain(
            tmp_path,
            monkeypatch,
            postcleanup_inventory_source="final",
            binding_device=True,
        )


def test_scientific_preaccess_and_prior_interruption_are_exact_and_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_release_paths(monkeypatch, tmp_path)
    authorization, audit = _create_valid_scientific_preaccess()
    prior = _create_valid_prior_interruption()
    scientific = release._validate_scientific_preaccess()
    interruption = release._validate_prior_interruption()
    assert scientific["authorization"]["authorization_fingerprint"] == authorization["authorization_fingerprint"]
    assert scientific["audit"]["receipt_fingerprint"] == audit["receipt_fingerprint"]
    assert interruption["receipt_fingerprint"] == prior["receipt_fingerprint"]
    assert stat.S_IMODE(release.SCIENTIFIC_AUTHORIZATION_PATH.stat().st_mode) == 0o444
    assert stat.S_IMODE(release.SCIENTIFIC_ACCESS_AUDIT_PATH.stat().st_mode) == 0o444

    bad_path = release.EVIDENCE_ROOT / "bad-scientific-auth.json"
    bad = dict(authorization)
    bad.pop("authorization_fingerprint")
    bad["unreviewed_extension"] = True
    _seal(bad_path, bad, "authorization_fingerprint")
    monkeypatch.setattr(release, "SCIENTIFIC_AUTHORIZATION_PATH", bad_path)
    with pytest.raises(PermissionError, match="preaccess identity"):
        release._validate_scientific_preaccess()


def test_tool_contains_no_systemctl_mutation_command() -> None:
    text = Path(release.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        literal_tokens = {
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
        }
        assert not (
            "daemon-reload" in literal_tokens
            and "SYSTEMCTL" in ast.dump(node).upper()
        )
    assert "systemctl start" not in text
    assert "systemctl enable" not in text
    assert "systemctl stop" not in text


def _realization_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, object]:
    (
        evidence, unit_dir, alternate, generator_late, runtime_spec,
    ) = realization_fixtures._workspace(tmp_path)
    monkeypatch.setattr(release, "RUNTIME_SPEC_PATH", runtime_spec)
    template = tmp_path / "actual-release-template.service"
    template.write_text(
        realization_fixtures.TEMPLATE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runner = realization_fixtures.FakeRunner(
        unit_dir, alternate, generator_late,
    )
    runner.runtime_spec = runtime_spec
    authorization_path = evidence / "authorization.json"
    release.actual_realizer.create_authorization(
        authorization_path, template_path=template,
        python_path=release.PYTHON_PATH,
        supervisor_path=realization_fixtures.SUPERVISOR,
        runtime_spec_path=runtime_spec,
        authorization_basis=release.actual_realizer.AUTHORIZATION_BASIS,
        instruction_id=release.actual_realizer.INSTRUCTION_ID,
        runner=runner,
        manager_reader=lambda: deepcopy(realization_fixtures._manager()),
    )
    receipt_path = evidence / "receipt.json"
    release.actual_realizer.realize_actual_unit(
        authorization_path, receipt_path=receipt_path,
        terminal_path=evidence / "terminal.json", runner=runner,
        manager_reader=lambda: deepcopy(realization_fixtures._manager()),
    )
    return authorization_path, receipt_path, runner


def _validate_realization_fixture(
    authorization_path: Path, receipt_path: Path, runner: object,
) -> dict[str, object]:
    return release._validate_realization_chain(
        authorization_path=authorization_path,
        receipt_path=receipt_path,
        shadow_reader=lambda: release.actual_realizer.query_shadow(
            runner=runner,
        ),
        manager_reader=lambda: deepcopy(realization_fixtures._manager()),
        unit_path_policy_reader=lambda fragment: (
            release.actual_realizer._observe_unit_path_policy(
                runner=runner, allowed_fragment=fragment,
            )
        ),
    )


def _replace_sealed(
    path: Path, payload: dict[str, object], fingerprint_field: str,
) -> dict[str, object]:
    body = deepcopy(payload)
    body.pop(fingerprint_field, None)
    path.unlink()
    return _seal(path, body, fingerprint_field)


def test_realization_release_replays_exact_policy_shadow_and_manager_brackets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )
    manager_reads = 0
    policy_reads = 0

    def manager_reader() -> dict[str, object]:
        nonlocal manager_reads
        manager_reads += 1
        return deepcopy(realization_fixtures._manager())

    def policy_reader(fragment: Path) -> dict[str, object]:
        nonlocal policy_reads
        policy_reads += 1
        return release.actual_realizer._observe_unit_path_policy(
            runner=runner, allowed_fragment=fragment,
        )

    result = release._validate_realization_chain(
        authorization_path=authorization_path, receipt_path=receipt_path,
        shadow_reader=lambda: release.actual_realizer.query_shadow(runner=runner),
        manager_reader=manager_reader,
        unit_path_policy_reader=policy_reader,
    )
    assert result["receipt"]["passed"] is True
    assert manager_reads == 4
    assert policy_reads == 2


@pytest.mark.parametrize(
    ("field", "drift"),
    [
        ("candidate", "drifted-candidate"),
        ("stage_id", "drifted-stage"),
        ("attempt_id", "drifted-attempt"),
        ("unit_name", "drifted.service"),
        ("instruction_id", "drifted-instruction"),
        ("created_at_utc", "1970-01-01T00:00:00Z"),
        ("payload_authority", "scientific-payload"),
        ("completed_actions", ["install-runtime-static-fragment"]),
    ],
)
def test_realization_release_rejects_resealed_receipt_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    field: str, drift: object,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )
    del runner
    receipt = release._load_sealed(
        receipt_path, fingerprint_field="receipt_fingerprint",
    )
    receipt[field] = drift
    resealed = _replace_sealed(
        receipt_path, receipt, "receipt_fingerprint",
    )
    assert resealed["receipt_fingerprint"] == release.stable_fingerprint({
        key: value for key, value in resealed.items()
        if key != "receipt_fingerprint"
    })
    with pytest.raises(PermissionError, match="realization"):
        release._validate_realization_chain(
            authorization_path=authorization_path,
            receipt_path=receipt_path,
            shadow_reader=lambda: pytest.fail(
                "shadow read after receipt semantic drift"
            ),
            manager_reader=lambda: pytest.fail(
                "manager read after receipt semantic drift"
            ),
            unit_path_policy_reader=lambda _fragment: pytest.fail(
                "policy read after receipt semantic drift"
            ),
        )


@pytest.mark.parametrize(
    ("field", "drift"),
    [
        ("actions", ["install-runtime-static-fragment"]),
        ("payload_authority", "scientific-payload"),
        ("created_at_utc", "1970-01-01T00:00:00Z"),
        ("issued_at_utc", "2099-01-01T00:00:00Z"),
    ],
)
def test_realization_release_rejects_resealed_authorization_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    field: str, drift: object,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )
    del runner
    authorization = release._load_sealed(
        authorization_path, fingerprint_field="authorization_fingerprint",
    )
    authorization[field] = drift
    rebound_authorization = _replace_sealed(
        authorization_path, authorization, "authorization_fingerprint",
    )
    receipt = release._load_sealed(
        receipt_path, fingerprint_field="receipt_fingerprint",
    )
    receipt["authorization_fingerprint"] = rebound_authorization[
        "authorization_fingerprint"
    ]
    receipt["authorization_file_sha256"] = release.file_sha256(
        authorization_path
    )
    _replace_sealed(receipt_path, receipt, "receipt_fingerprint")
    with pytest.raises(PermissionError, match="realization|authorization"):
        release._validate_realization_chain(
            authorization_path=authorization_path,
            receipt_path=receipt_path,
            shadow_reader=lambda: pytest.fail(
                "shadow read after authorization semantic drift"
            ),
            manager_reader=lambda: pytest.fail(
                "manager read after authorization semantic drift"
            ),
            unit_path_policy_reader=lambda _fragment: pytest.fail(
                "policy read after authorization semantic drift"
            ),
        )


@pytest.mark.parametrize(
    "drift_kind",
    [
        "template-binding", "rendered-fragment", "runtime-binding",
        "systemctl-binding", "expected-shadow", "template-symlink-int",
        "systemctl-symlink-int", "runtime-parent-exists-int",
    ],
)
def test_realization_release_rejects_resealed_nested_authorization_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift_kind: str,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )
    del runner
    authorization = release._load_sealed(
        authorization_path, fingerprint_field="authorization_fingerprint",
    )
    echoed_field: str | None = None
    if drift_kind == "template-binding":
        authorization["template_binding"]["file_sha256"] = "0" * 64
        echoed_field = "template_binding"
    elif drift_kind == "rendered-fragment":
        authorization["rendered_fragment"]["sha256"] = "0" * 64
        echoed_field = "rendered_fragment"
    elif drift_kind == "runtime-binding":
        authorization["runtime_spec_binding"]["kind"] = "drifted"
        echoed_field = "runtime_spec_binding"
    elif drift_kind == "systemctl-binding":
        authorization["executable_bindings"]["systemctl"][
            "file_sha256"
        ] = "0" * 64
        echoed_field = "executable_bindings"
    elif drift_kind == "template-symlink-int":
        authorization["template_binding"]["path_is_symlink"] = 0
        echoed_field = "template_binding"
    elif drift_kind == "systemctl-symlink-int":
        authorization["executable_bindings"]["systemctl"][
            "path_is_symlink"
        ] = 0
        echoed_field = "executable_bindings"
    elif drift_kind == "runtime-parent-exists-int":
        authorization["runtime_spec_binding"][
            "runtime_spec_parent_identity"
        ]["exists"] = 1
        echoed_field = "runtime_spec_binding"
    else:
        authorization["expected_static_shadow"]["Restart"] = "always"
    rebound_authorization = _replace_sealed(
        authorization_path, authorization, "authorization_fingerprint",
    )
    receipt = release._load_sealed(
        receipt_path, fingerprint_field="receipt_fingerprint",
    )
    receipt["authorization_fingerprint"] = rebound_authorization[
        "authorization_fingerprint"
    ]
    receipt["authorization_file_sha256"] = release.file_sha256(
        authorization_path
    )
    if echoed_field is not None:
        receipt[echoed_field] = deepcopy(rebound_authorization[echoed_field])
    _replace_sealed(receipt_path, receipt, "receipt_fingerprint")
    with pytest.raises(PermissionError, match="realization|bound file"):
        release._validate_realization_chain(
            authorization_path=authorization_path,
            receipt_path=receipt_path,
            shadow_reader=lambda: pytest.fail(
                "shadow read after nested authorization drift"
            ),
            manager_reader=lambda: pytest.fail(
                "manager read after nested authorization drift"
            ),
            unit_path_policy_reader=lambda _fragment: pytest.fail(
                "policy read after nested authorization drift"
            ),
        )


def test_realization_release_rejects_resealed_fragment_identity_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )
    del runner
    receipt = release._load_sealed(
        receipt_path, fingerprint_field="receipt_fingerprint",
    )
    receipt["fragment_identity"]["unclosed_extension"] = True
    _replace_sealed(receipt_path, receipt, "receipt_fingerprint")
    with pytest.raises(PermissionError, match="fragment identity"):
        release._validate_realization_chain(
            authorization_path=authorization_path,
            receipt_path=receipt_path,
            shadow_reader=lambda: pytest.fail(
                "shadow read after fragment identity extension"
            ),
            manager_reader=lambda: pytest.fail(
                "manager read after fragment identity extension"
            ),
            unit_path_policy_reader=lambda _fragment: pytest.fail(
                "policy read after fragment identity extension"
            ),
        )


@pytest.mark.parametrize(
    ("field", "drift"),
    [
        ("device", True),
        ("inode", True),
        ("owner_uid", True),
        ("mode", 384.0),
        ("nlink", True),
    ],
)
def test_realization_release_rejects_fragment_numeric_type_coercion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    field: str, drift: object,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )
    del runner
    receipt = release._load_sealed(
        receipt_path, fingerprint_field="receipt_fingerprint",
    )
    receipt["fragment_identity"][field] = drift
    _replace_sealed(receipt_path, receipt, "receipt_fingerprint")
    with pytest.raises(PermissionError, match="fragment"):
        release._validate_realization_chain(
            authorization_path=authorization_path,
            receipt_path=receipt_path,
            shadow_reader=lambda: pytest.fail(
                "shadow read after fragment numeric coercion"
            ),
            manager_reader=lambda: pytest.fail(
                "manager read after fragment numeric coercion"
            ),
            unit_path_policy_reader=lambda _fragment: pytest.fail(
                "policy read after fragment numeric coercion"
            ),
        )


@pytest.mark.parametrize(
    "drift_kind",
    ["extra-normalized-key", "missing-normalized-key", "normalized-value"],
)
def test_realization_release_rejects_resealed_shadow_closure_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift_kind: str,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )
    receipt = release._load_sealed(
        receipt_path, fingerprint_field="receipt_fingerprint",
    )
    shadow = receipt["full_static_shadow"]
    assert isinstance(shadow, dict)
    if drift_kind == "extra-normalized-key":
        shadow["normalized_Unexpected"] = {
            "path": "/drift", "argv": [], "ignore_errors": "no",
        }
    elif drift_kind == "missing-normalized-key":
        shadow.pop("normalized_ExecStart")
    else:
        shadow["normalized_ExecStart"]["ignore_errors"] = "yes"
    _replace_sealed(receipt_path, receipt, "receipt_fingerprint")
    with pytest.raises(PermissionError, match="shadow closure"):
        _validate_realization_fixture(
            authorization_path, receipt_path, runner,
        )


def test_realization_release_rejects_second_generator_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )
    runner.rotate_generator_late()
    with pytest.raises(PermissionError, match="live path policy"):
        release._validate_realization_chain(
            authorization_path=authorization_path,
            receipt_path=receipt_path,
            shadow_reader=lambda: release.actual_realizer.query_shadow(
                runner=runner,
            ),
            manager_reader=lambda: deepcopy(realization_fixtures._manager()),
            unit_path_policy_reader=lambda fragment: (
                release.actual_realizer._observe_unit_path_policy(
                    runner=runner, allowed_fragment=fragment,
                )
            ),
        )


def test_realization_release_rejects_non_generator_receipt_policy_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )
    del runner
    receipt = release._load_sealed(
        receipt_path, fingerprint_field="receipt_fingerprint",
    )
    body = deepcopy(receipt)
    body.pop("receipt_fingerprint")
    body["unit_path_policy"]["runtime_directory_priority"] += 1
    bad_receipt = receipt_path.with_name("receipt-illegal-policy.json")
    _seal(bad_receipt, body, "receipt_fingerprint")
    with pytest.raises(PermissionError, match="path policy changed"):
        release._validate_realization_chain(
            authorization_path=authorization_path,
            receipt_path=bad_receipt,
            shadow_reader=lambda: pytest.fail("shadow read after bad transition"),
            manager_reader=lambda: pytest.fail("manager read after bad transition"),
            unit_path_policy_reader=lambda _fragment: pytest.fail(
                "live policy read after bad transition",
            ),
        )


def test_realization_release_rejects_shadow_drift_before_second_policy_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )

    def drifted_shadow() -> dict[str, str]:
        shadow = release.actual_realizer.query_shadow(runner=runner)
        shadow["Restart"] = "always"
        return shadow

    with pytest.raises(PermissionError, match="shadow"):
        release._validate_realization_chain(
            authorization_path=authorization_path,
            receipt_path=receipt_path, shadow_reader=drifted_shadow,
            manager_reader=lambda: deepcopy(realization_fixtures._manager()),
            unit_path_policy_reader=lambda fragment: (
                release.actual_realizer._observe_unit_path_policy(
                    runner=runner, allowed_fragment=fragment,
                )
            ),
        )


def test_realization_release_rejects_same_byte_fragment_race_after_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path, receipt_path, runner = _realization_chain(
        tmp_path, monkeypatch,
    )
    policy_reads = 0

    def policy_reader(fragment: Path) -> dict[str, object]:
        nonlocal policy_reads
        policy_reads += 1
        policy = release.actual_realizer._observe_unit_path_policy(
            runner=runner, allowed_fragment=fragment,
        )
        if policy_reads == 2:
            raw = fragment.read_bytes()
            fragment.rename(
                fragment.with_name(f"{fragment.name}.before-release-race"),
            )
            fragment.write_bytes(raw)
            fragment.chmod(0o600)
        return policy

    with pytest.raises(PermissionError, match="fragment identity drifted"):
        release._validate_realization_chain(
            authorization_path=authorization_path,
            receipt_path=receipt_path,
            shadow_reader=lambda: release.actual_realizer.query_shadow(
                runner=runner,
            ),
            manager_reader=lambda: deepcopy(realization_fixtures._manager()),
            unit_path_policy_reader=policy_reader,
        )


def test_release_stable_load_rejects_same_byte_inode_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    path = tmp_path / "sealed.json"
    _seal(path, {"schema_version": "race-v1"}, "receipt_fingerprint")
    original = release._read_all_fd
    replaced = False

    def read_then_replace(descriptor: int) -> bytes:
        nonlocal replaced
        raw = original(descriptor)
        if not replaced:
            replaced = True
            path.rename(path.with_name(f"{path.name}.before-race"))
            path.write_bytes(raw)
            path.chmod(0o444)
        return raw

    monkeypatch.setattr(release, "_read_all_fd", read_then_replace)
    with pytest.raises(PermissionError, match="identity changed"):
        release._load_sealed(
            path, fingerprint_field="receipt_fingerprint",
        )


def test_release_write_rejects_parent_generation_replacement_before_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "release-parent"
    parent.mkdir(mode=0o700)
    target = parent / "release.json"
    original = release.os.pread
    rotated = False

    def pread_then_rotate(
        descriptor: int, length: int, offset: int,
    ) -> bytes:
        nonlocal rotated
        raw = original(descriptor, length, offset)
        if not rotated:
            rotated = True
            parent.rename(tmp_path / "release-parent-before")
            parent.mkdir(mode=0o700)
        return raw

    monkeypatch.setattr(release.os, "pread", pread_then_rotate)
    with pytest.raises(PermissionError, match="parent generation changed"):
        release._write_sealed(
            target, {"schema_version": "race-output-v1"},
            fingerprint_field="receipt_fingerprint",
        )
