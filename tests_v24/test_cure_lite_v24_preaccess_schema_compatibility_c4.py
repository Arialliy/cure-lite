from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c4.py"
)


@pytest.fixture
def bridge():
    name = "cure_lite_v24_preaccess_schema_compatibility_c4_tested"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loaded_state(unit: str, *, fragment: str = "") -> dict[str, str]:
    return {
        "Id": unit,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "NRestarts": "0",
        "FragmentPath": fragment,
        "InvocationID": "",
    }


def _missing_state(unit: str) -> dict[str, str]:
    return {
        "Id": unit,
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "",
        "NRestarts": "0",
        "FragmentPath": "",
        "InvocationID": "",
    }


def _payload_flags() -> dict[str, bool]:
    return {
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
        "training_started": False,
        "materialization_consumed": False,
    }


def test_generation_identity_and_paths_are_disjoint(bridge) -> None:
    assert bridge.RUNTIME_COMPATIBILITY_ID == "c4"
    assert bridge.SCIENTIFIC_ATTEMPT_ORDINAL == 2
    assert bridge.C3_UNIT_NAME.endswith("compat-c3.service")
    assert bridge.C4_UNIT_NAME.endswith("compat-c4.service")
    assert bridge.C3_UNIT_NAME != bridge.C4_UNIT_NAME
    assert bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_PATH.name == (
        "r2_preaccess_schema_compat_c3_"
        "environment_stability_failure_terminal.json"
    )
    assert bridge.C4_ENVIRONMENT_SCOPE_HANDOFF_PATH.name == (
        "runtime_environment_scope_handoff_preaccess_compat_c4.json"
    )
    assert bridge.C4_ENVIRONMENT_STABILITY_ATTEMPT_PATH.name == (
        "runtime_environment_stability_attempt_preaccess_compat_c4.json"
    )
    assert bridge.C4_ENVIRONMENT_STABILITY_TERMINAL_PATH.name == (
        "runtime_environment_stability_terminal_preaccess_compat_c4.json"
    )
    assert bridge.C4_RUNTIME_SPEC_PATH != bridge.C3_RUNTIME_SPEC_PATH
    assert bridge.C4_RUN_ROOT_ALIAS_PATH != bridge.C3_RUN_ROOT_ALIAS_PATH


def test_blind_rename_did_not_corrupt_historical_c2_hashes(bridge) -> None:
    assert bridge.C2_MODE_CONTRACT_FAILURE_TERMINAL_SHA256 == (
        "e478e0cc3516c97b5eea91c615a64cd7ee4020a9d22ddc863c7b04375331f9e7"
    )
    assert bridge.C2_PREWRITE_FAILURE_TERMINAL_SHA256 == (
        "6984dc9df2c905a5b7bc3b1577a4d5e8c21d1e1f895217997ed6915050e0f43d"
    )
    payload, _root, _source = (
        bridge._validate_mode_contract_failure_terminal()
    )
    continuation = payload["continuation_policy"]
    assert continuation["c3_required"] is True
    assert "c4_required" not in continuation


def test_direct_predecessor_terminalizer_pin_matches_frozen_source(
    bridge,
) -> None:
    raw = bridge.C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH.read_bytes()
    if bridge.C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256 == (
        "__TO_BE_FROZEN__"
    ):
        assert bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256 == (
            "__TO_BE_FROZEN__"
        )
        return
    assert hashlib.sha256(raw).hexdigest() == (
        bridge.C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256
    )
    assert bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SCHEMA == (
        "cure-lite-v24-r2-preaccess-schema-compat-c3-"
        "environment-stability-failure-terminal-v1"
    )


def test_terminal_hash_is_pinned_and_sentinel_remains_fail_closed(
    bridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256 == (
        "527eb5c12c92e19dac8f797868de2bc8462e53b8113c24f6e701e0e54a26180a"
    )
    assert hashlib.sha256(
        bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_PATH.read_bytes()
    ).hexdigest() == bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256
    monkeypatch.setattr(
        bridge,
        "C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256",
        "__TO_BE_FROZEN__",
    )
    with pytest.raises(PermissionError, match="hashes are not frozen"):
        bridge._require_frozen_c3_failure_hashes()


def test_frozen_production_c3_fail_boundary_validates_archivally(bridge) -> None:
    payload, root, source_root = (
        bridge._validate_c3_environment_failure_terminal()
    )
    assert payload["identity"]["runtime_compatibility_id"] == "c3"
    assert payload["continuation_policy"]["c4_required"] is True
    assert payload["payload_observation"]["samples_processed"] == 0
    assert root["file_sha256"] == (
        bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256
    )
    assert root["terminal_fingerprint"] == (
        bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_FINGERPRINT
    )
    assert source_root["file_sha256"] == (
        bridge.C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256
    )


def test_authorize_stops_on_sentinel_before_any_audit(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        bridge, "C4_AUTHORIZATION_PATH", tmp_path / "authorization.json"
    )
    monkeypatch.setattr(bridge, "C4_RECEIPT_PATH", tmp_path / "receipt.json")
    monkeypatch.setattr(
        bridge,
        "C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256",
        "__TO_BE_FROZEN__",
    )
    monkeypatch.setattr(
        bridge,
        "_validate_scientific_output_phase",
        lambda **_kwargs: calls.append("scientific"),
    )
    with pytest.raises(PermissionError, match="hashes are not frozen"):
        bridge.authorize_c4(
            instruction_id=bridge.INSTRUCTION_ID,
            authorization_basis=bridge.AUTHORIZATION_BASIS,
        )
    assert calls == []


def test_byte_pinned_terminalizer_loader_accepts_only_exact_interface(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "terminalizer.py"
    terminal = tmp_path / "terminal.json"
    source.write_text(
        "\n".join(
            (
                f"TERMINAL_PATH = {str(terminal)!r}",
                f"SCHEMA = {bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SCHEMA!r}",
                f"CANDIDATE = {bridge.CANDIDATE!r}",
                f"STAGE_ID = {bridge.STAGE_ID!r}",
                f"SCIENTIFIC_ATTEMPT_ID = {bridge.SCIENTIFIC_ATTEMPT_ID!r}",
                f"SCIENTIFIC_ATTEMPT_ORDINAL = {bridge.SCIENTIFIC_ATTEMPT_ORDINAL!r}",
                "RUNTIME_COMPATIBILITY_ID = 'c3'",
                f"C3_UNIT_NAME = {bridge.C3_UNIT_NAME!r}",
                "def validate_archival(path=None):",
                "    return {}, {}",
                "",
            )
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH", source
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINAL_PATH", terminal
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256", digest
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256", "a" * 64
    )
    module, root = bridge._load_verified_c3_environment_failure_terminalizer()
    assert module.RUNTIME_COMPATIBILITY_ID == "c3"
    assert root["file_sha256"] == digest

    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256", "b" * 64
    )
    with pytest.raises(PermissionError, match="terminalizer source changed"):
        bridge._load_verified_c3_environment_failure_terminalizer()


def _c3_failure_payload(bridge) -> dict[str, object]:
    return {
        "schema_version": bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SCHEMA,
        "identity": {
            "candidate": bridge.CANDIDATE,
            "stage_id": bridge.STAGE_ID,
            "scientific_attempt_id": bridge.SCIENTIFIC_ATTEMPT_ID,
            "scientific_attempt_ordinal": bridge.SCIENTIFIC_ATTEMPT_ORDINAL,
            "runtime_compatibility_id": "c3",
            "sealed_at_utc": "2026-07-31T10:00:00.000000Z",
        },
        "unit_realization_closure": {
            "R3_receipt_passed": True,
            "static": True,
            "enabled": False,
            "started": False,
            "removed": False,
            "payload_authority": "none",
            "unit_name": bridge.C3_UNIT_NAME,
        },
        "environment_stability_failure": {
            "known_subcommand": "stability-gate",
            "attempt_count": 1,
            "retry": False,
            "samples_collected": 0,
            "expected_exception_type": "PermissionError",
            "expected_exception_message": (
                "precleanup inventory unit scope changed"
            ),
        },
        "deterministic_reproduction": {
            "reproduced": True,
            "samples_collected": 0,
        },
        "continuation_policy": {
            "automatic_retry": False,
            "same_c3_reentry": False,
            "same_c3_reauthorization_allowed": False,
            "same_c3_metadata_repair_allowed": False,
            "c3_environment_gate_reentry_allowed": False,
            "c3_environment_gate_repair_allowed": False,
            "c4_required": True,
            "new_explicit_authorization_required": True,
            "scientific_attempt_id": bridge.SCIENTIFIC_ATTEMPT_ID,
            "scientific_attempt_ordinal": bridge.SCIENTIFIC_ATTEMPT_ORDINAL,
            "scientific_attempt_consumed": False,
            "c3_authorization_consumed": True,
            "unit_realization_consumed": True,
            "environment_metadata_attempt_consumed": True,
            "runtime_launch_consumed": False,
            "materialization_consumed": False,
        },
        "payload_observation": {
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_accessed": False,
            "training_started": False,
            "samples_processed": 0,
            "optimizer_steps": 0,
            "parameter_updates": 0,
        },
    }


def test_direct_predecessor_is_validated_as_sealed_fail_not_pass(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "terminalizer.py"
    source.write_text("# frozen terminalizer fixture\n", encoding="utf-8")
    terminal = tmp_path / "terminal.json"
    payload = _c3_failure_payload(bridge)
    terminal.write_text(
        bridge._canonical_json(payload) + "\n", encoding="utf-8"
    )
    terminal.chmod(0o444)
    source_root = bridge._source_root(source)
    terminal_digest = hashlib.sha256(terminal.read_bytes()).hexdigest()
    terminal_root = {
        "path": str(terminal.absolute()),
        "file_sha256": terminal_digest,
        "mode": 0o444,
        "schema_version": bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SCHEMA,
        "terminal_fingerprint": "d" * 64,
        "terminalizer_source_root": source_root,
    }
    module = SimpleNamespace(
        validate_archival=lambda _path: (payload, terminal_root)
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH", source
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINAL_PATH", terminal
    )
    monkeypatch.setattr(
        bridge,
        "C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256",
        source_root["file_sha256"],
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256", terminal_digest
    )
    monkeypatch.setattr(
        bridge,
        "C3_ENVIRONMENT_FAILURE_TERMINAL_FINGERPRINT",
        "d" * 64,
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_c3_environment_failure_terminalizer",
        lambda: (module, source_root),
    )
    validated, root, returned_source = (
        bridge._validate_c3_environment_failure_terminal()
    )
    assert validated == payload
    assert root == terminal_root
    assert returned_source == source_root
    assert "passed" not in validated
    assert validated["continuation_policy"]["c4_required"] is True
    assert validated["payload_observation"]["samples_processed"] == 0


def test_direct_predecessor_drift_during_archival_is_rejected(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "terminalizer.py"
    source.write_text("# stable source\n", encoding="utf-8")
    terminal = tmp_path / "terminal.json"
    terminal.write_text("{}\n", encoding="utf-8")
    terminal.chmod(0o444)
    source_root = bridge._source_root(source)
    original_digest = hashlib.sha256(terminal.read_bytes()).hexdigest()

    def validate(_path):
        terminal.chmod(0o644)
        terminal.write_text('{"drift":true}\n', encoding="utf-8")
        terminal.chmod(0o444)
        return {}, {}

    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH", source
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINAL_PATH", terminal
    )
    monkeypatch.setattr(
        bridge,
        "C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256",
        source_root["file_sha256"],
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256", original_digest
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_c3_environment_failure_terminalizer",
        lambda: (SimpleNamespace(validate_archival=validate), source_root),
    )
    with pytest.raises(PermissionError, match="lineage changed"):
        bridge._validate_c3_environment_failure_terminal()


def test_sealed_writer_is_create_once_and_readback_verified(
    bridge, tmp_path: Path
) -> None:
    path = tmp_path / "authorization.json"
    payload = bridge._write_sealed(
        path,
        {"schema_version": "fixture-v1", **_payload_flags()},
        fingerprint_field="authorization_fingerprint",
    )
    assert path.stat().st_mode & 0o777 == 0o444
    loaded, root = bridge._load_sealed(
        path,
        fingerprint_field="authorization_fingerprint",
        schema="fixture-v1",
    )
    assert loaded == payload
    assert root["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        bridge._write_sealed(
            path,
            {"schema_version": "fixture-v1", **_payload_flags()},
            fingerprint_field="authorization_fingerprint",
        )


def test_source_closure_requires_all_c4_and_r14_sources_final(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths: dict[str, Path] = {}
    for label in bridge._SOURCE_LABELS:
        path = tmp_path / f"{label}.txt"
        path.write_text(f"frozen:{label}\n", encoding="utf-8")
        paths[label] = path
    monkeypatch.setattr(bridge, "_source_paths", lambda: paths)
    monkeypatch.setattr(
        bridge,
        "C1_FAILURE_TERMINALIZER_SHA256",
        hashlib.sha256(paths["c1_failure_terminalizer"].read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        bridge,
        "C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SHA256",
        hashlib.sha256(
            paths["c2_mode_contract_failure_terminalizer"].read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        bridge,
        "C2_PREWRITE_FAILURE_TERMINALIZER_SHA256",
        hashlib.sha256(
            paths["c2_prewrite_failure_terminalizer"].read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        bridge,
        "C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256",
        hashlib.sha256(
            paths["c3_environment_failure_terminalizer"].read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256", "a" * 64
    )
    roots = bridge._collect_source_roots()
    assert set(roots) == bridge._SOURCE_LABELS
    assert {
        "compat_bridge",
        "compat_environment_wrapper",
        "compat_unit_realizer",
        "compat_supervisor",
        "compat_release",
        "compat_adapter",
        "compat_unit_template",
        "r14_integration_wrapper",
        "r14_shared_realizer",
        "r14_dummy_child",
        "r14_dummy_unit_template",
    }.issubset(roots)

    paths["compat_environment_wrapper"].write_text(
        "binding = '__TO_BE_FROZEN__'\n", encoding="utf-8"
    )
    with pytest.raises(PermissionError, match="unfrozen binding"):
        bridge._collect_source_roots()


def test_protected_units_include_live_c3_and_empty_c4(bridge) -> None:
    states = {
        bridge.OLD_UNIT_NAME: _loaded_state(bridge.OLD_UNIT_NAME),
        bridge.C1_UNIT_NAME: _loaded_state(bridge.C1_UNIT_NAME),
        bridge.C2_UNIT_NAME: _missing_state(bridge.C2_UNIT_NAME),
        bridge.C3_UNIT_NAME: _loaded_state(
            bridge.C3_UNIT_NAME,
            fragment=str(bridge.C3_UNIT_FRAGMENT_PATH),
        ),
        bridge.C4_UNIT_NAME: _missing_state(bridge.C4_UNIT_NAME),
    }
    observed = bridge._collect_protected_unit_states(states.__getitem__)
    assert set(observed) == {"old", "c1", "c2", "c3"}
    assert observed["c3"]["UnitFileState"] == "static"
    target = bridge._collect_preauthorization_target_unit_state(
        states.__getitem__
    )
    assert target["LoadState"] == "not-found"

    states[bridge.C3_UNIT_NAME]["FragmentPath"] = "/wrong"
    with pytest.raises(PermissionError, match="static/inert"):
        bridge._collect_protected_unit_states(states.__getitem__)


def test_c4_path_sets_encode_append_only_boundary(bridge) -> None:
    permanent = bridge._always_absent_paths()
    preauthorization = bridge._c4_preauthorization_paths()
    future = bridge._c4_future_paths()
    assert {
        "c3_compatibility_receipt",
        "c3_environment_stability",
        "c3_environment_postcleanup",
        "c3_unit_terminal",
        "c3_runtime_spec",
        "c3_runtime_launch_authorization",
        "c3_runtime_artifacts",
        "c3_gpu_lease",
        "c3_run_alias",
        "c3_result_alias",
    }.issubset(permanent)
    assert {
        "c4_environment_policy",
        "c4_environment_scope_handoff",
        "c4_environment_stability_attempt",
        "c4_environment_stability_terminal",
        "c4_environment_stability",
        "c4_environment_postcleanup",
        "c4_unit_authorization",
        "c4_unit_receipt",
        "c4_unit_terminal",
        "c4_unit_fragment",
    } == set(preauthorization)
    assert set(future) == {
        "c4_runtime_spec",
        "c4_runtime_launch_authorization",
        "c4_runtime_artifacts",
        "c4_gpu_lease",
    }


def _write_environment_fixture(
    bridge,
    path: Path,
    *,
    schema: str,
    fingerprint_field: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": schema,
        **_payload_flags(),
    }
    return bridge._write_sealed(
        path, body, fingerprint_field=fingerprint_field
    )


def test_environment_loader_requires_handoff_attempt_pass_and_no_terminal(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        "C4_ENVIRONMENT_SCOPE_HANDOFF_PATH": tmp_path / "handoff.json",
        "C4_ENVIRONMENT_STABILITY_ATTEMPT_PATH": tmp_path / "attempt.json",
        "C4_ENVIRONMENT_POLICY_PATH": tmp_path / "policy.json",
        "C4_ENVIRONMENT_STABILITY_PATH": tmp_path / "stability.json",
        "C4_ENVIRONMENT_POSTCLEANUP_PATH": tmp_path / "postcleanup.json",
        "C4_ENVIRONMENT_STABILITY_TERMINAL_PATH": tmp_path / "terminal.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(bridge, name, path)
    handoff = _write_environment_fixture(
        bridge,
        paths["C4_ENVIRONMENT_SCOPE_HANDOFF_PATH"],
        schema=bridge.ENVIRONMENT_SCOPE_HANDOFF_SCHEMA,
        fingerprint_field="scope_handoff_fingerprint",
    )
    attempt = _write_environment_fixture(
        bridge,
        paths["C4_ENVIRONMENT_STABILITY_ATTEMPT_PATH"],
        schema=bridge.ENVIRONMENT_STABILITY_ATTEMPT_SCHEMA,
        fingerprint_field="stability_attempt_fingerprint",
    )
    policy = _write_environment_fixture(
        bridge,
        paths["C4_ENVIRONMENT_POLICY_PATH"],
        schema=bridge.ENVIRONMENT_POLICY_SCHEMA,
        fingerprint_field="policy_fingerprint",
    )
    stability = _write_environment_fixture(
        bridge,
        paths["C4_ENVIRONMENT_STABILITY_PATH"],
        schema=bridge.ENVIRONMENT_STABILITY_SCHEMA,
        fingerprint_field="stability_receipt_fingerprint",
    )
    postcleanup = _write_environment_fixture(
        bridge,
        paths["C4_ENVIRONMENT_POSTCLEANUP_PATH"],
        schema=bridge.ENVIRONMENT_RECEIPT_SCHEMA,
        fingerprint_field="receipt_fingerprint",
    )
    loaded = bridge._load_environment_evidence()
    assert loaded[:5] == (
        handoff,
        attempt,
        policy,
        stability,
        postcleanup,
    )
    assert set(loaded[5]) == {
        "environment_scope_handoff",
        "environment_stability_attempt",
        "environment_policy",
        "environment_stability",
        "environment_postcleanup",
    }

    paths["C4_ENVIRONMENT_STABILITY_TERMINAL_PATH"].write_text(
        "failure\n", encoding="utf-8"
    )
    with pytest.raises(PermissionError, match="stability_terminal"):
        bridge._load_environment_evidence()


def _fake_c1_terminal(bridge) -> dict[str, object]:
    return {
        "evidence_roots": {
            "bridge_authorization": {"path": "c1-authorization"},
            "r10_authorization": {"path": "r10-authorization"},
            "r10_receipt": {"path": "r10-receipt"},
        },
        "authorization_expiry": {
            "bridge_expires_at_utc": "2026-07-31T09:00:00.000000Z"
        },
    }


def test_authorize_binds_direct_fail_all_sources_units_and_new_evidence(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization_path = tmp_path / "c4-authorization.json"
    receipt_path = tmp_path / "c4-receipt.json"
    monkeypatch.setattr(bridge, "C4_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(bridge, "C4_RECEIPT_PATH", receipt_path)
    monkeypatch.setattr(bridge, "_require_frozen_c3_failure_hashes", lambda: None)
    monkeypatch.setattr(
        bridge, "_validate_scientific_output_phase", lambda **_kwargs: None
    )
    monkeypatch.setattr(bridge, "_require_absent", lambda _paths: None)
    prewrite_root = {"path": "c2-prewrite"}
    prewrite_source = {"path": "c2-prewrite-source"}
    c2_failure_root = {"path": "c2-mode-failure"}
    c2_failure_source = {"path": "c2-mode-source"}
    c3_failure_root = {"path": "c3-environment-failure"}
    c3_failure_source = {"path": "c3-terminalizer-source"}
    c3_failure = {
        "identity": {"sealed_at_utc": "2026-07-31T10:00:00.000000Z"}
    }
    c1_terminal = _fake_c1_terminal(bridge)
    c1_terminal_root = {"path": "c1-terminal"}
    monkeypatch.setattr(
        bridge,
        "_validate_c2_prewrite_failure_terminal",
        lambda: ({}, prewrite_root, prewrite_source),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_mode_contract_failure_terminal",
        lambda: (
            {"identity": {"sealed_at_utc": "2026-07-31T09:30:00.000000Z"}},
            c2_failure_root,
            c2_failure_source,
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_c3_environment_failure_terminal",
        lambda: (c3_failure, c3_failure_root, c3_failure_source),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_c1_failure_terminal",
        lambda **_kwargs: (c1_terminal, c1_terminal_root, {"path": "c1-source"}),
    )
    sources = {label: {"path": label} for label in bridge._SOURCE_LABELS}
    sources["c2_prewrite_failure_terminalizer"] = prewrite_source
    sources["c2_mode_contract_failure_terminalizer"] = c2_failure_source
    sources["c3_environment_failure_terminalizer"] = c3_failure_source
    monkeypatch.setattr(bridge, "_collect_source_roots", lambda: sources)
    monkeypatch.setattr(
        bridge,
        "_load_verified_environment_wrapper",
        lambda root: (SimpleNamespace(), root),
    )
    protected = {name: {"Id": name} for name in ("old", "c1", "c2", "c3")}
    target_state = _missing_state(bridge.C4_UNIT_NAME)
    monkeypatch.setattr(
        bridge, "_collect_protected_unit_states", lambda _reader: protected
    )
    monkeypatch.setattr(
        bridge,
        "_collect_preauthorization_target_unit_state",
        lambda _reader: target_state,
    )
    written: dict[str, object] = {}

    def write(_path, body, *, fingerprint_field):
        written.update(body)
        result = dict(body)
        result[fingerprint_field] = bridge.stable_fingerprint(body)
        return result

    monkeypatch.setattr(bridge, "_write_sealed", write)
    result = bridge.authorize_c4(
        instruction_id=bridge.INSTRUCTION_ID,
        authorization_basis=bridge.AUTHORIZATION_BASIS,
        now=lambda: datetime(2026, 7, 31, 11, tzinfo=timezone.utc),
    )
    assert result["runtime_compatibility_id"] == "c4"
    assert written["c3_environment_failure_terminal_root"] == c3_failure_root
    assert written["compatibility_source_roots"] == sources
    assert written["protected_unit_states"] == protected
    assert written["preauthorization_target_unit_state"] == target_state
    expected = written["expected_evidence_paths"]
    assert "c3_environment_failure_terminal" in expected
    assert "environment_scope_handoff" in expected
    assert "environment_stability_attempt" in expected
    assert "environment_stability_terminal" in expected
    assert written["scientific_attempt_ordinal"] == 2
    assert written["training_started"] is False


@dataclass(frozen=True)
class _Contract:
    target_unit_id: str
    require_target_ready: bool


class _Environment:
    def __init__(self, bridge, unit_authorization, unit_receipt):
        self.bridge = bridge
        self.unit_authorization = unit_authorization
        self.unit_receipt = unit_receipt
        self.seen: tuple[object, ...] | None = None

    def _production_archival_validator(self, authorization_path, receipt_path):
        assert Path(authorization_path) == self.bridge.C4_UNIT_AUTHORIZATION_PATH
        assert Path(receipt_path) == self.bridge.C4_UNIT_RECEIPT_PATH
        return {
            "authorization": self.unit_authorization,
            "receipt": self.unit_receipt,
        }

    def replay_old_scope_and_handoff(self):
        return (
            _Contract(self.bridge.OLD_UNIT_NAME, False),
            _Contract(self.bridge.C4_UNIT_NAME, True),
            {"handoff": "replayed"},
        )

    def validate_c4_environment_closure(
        self,
        scope_handoff,
        stability_attempt,
        policy,
        stability,
        postcleanup,
        *,
        archival,
        c4_contract,
    ):
        self.seen = (
            scope_handoff,
            stability_attempt,
            policy,
            stability,
            postcleanup,
            archival,
            c4_contract,
        )
        return {
            "scope_handoff": scope_handoff,
            "stability_attempt": stability_attempt,
            "policy": policy,
            "stability": stability,
            "postcleanup": postcleanup,
            "realization": archival,
        }


def test_full_receipt_closure_binds_c3_r4_handoff_attempt_and_e4(
    bridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    prewrite_root = {"path": "c2-prewrite"}
    prewrite_source = {"path": "c2-prewrite-source"}
    failure_root = {"path": "c2-failure"}
    failure_source = {"path": "c2-failure-source"}
    c3_root = {"path": "c3-failure"}
    c3_source = {"path": "c3-failure-source"}
    c3_failure = {
        "identity": {"sealed_at_utc": "2026-07-31T10:00:00.000000Z"}
    }
    c1_terminal = _fake_c1_terminal(bridge)
    c1_terminal_root = {"path": "c1-terminal"}
    protected = {name: {"Id": name} for name in ("old", "c1", "c2", "c3")}
    sources = {label: {"path": label} for label in bridge._SOURCE_LABELS}
    sources["c2_prewrite_failure_terminalizer"] = prewrite_source
    sources["c2_mode_contract_failure_terminalizer"] = failure_source
    sources["c3_environment_failure_terminalizer"] = c3_source
    monkeypatch.setattr(
        bridge,
        "_validate_c2_prewrite_failure_terminal",
        lambda: ({}, prewrite_root, prewrite_source),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_mode_contract_failure_terminal",
        lambda: ({}, failure_root, failure_source),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_c3_environment_failure_terminal",
        lambda: (c3_failure, c3_root, c3_source),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_c1_failure_terminal",
        lambda **_kwargs: (c1_terminal, c1_terminal_root, {"path": "c1-source"}),
    )
    monkeypatch.setattr(
        bridge, "_collect_protected_unit_states", lambda _reader: protected
    )
    monkeypatch.setattr(bridge, "_validate_source_roots", lambda _roots: None)
    unit_authorization = {"unit_name": bridge.C4_UNIT_NAME}
    unit_receipt = {"unit_name": bridge.C4_UNIT_NAME}
    environment = _Environment(bridge, unit_authorization, unit_receipt)
    monkeypatch.setattr(
        bridge,
        "_load_verified_environment_wrapper",
        lambda root: (environment, dict(root)),
    )
    unit_receipt_time = datetime(2026, 7, 31, 10, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(
        bridge,
        "_validate_unit_chain",
        lambda **_kwargs: (
            unit_authorization,
            {"path": "r4-authorization"},
            unit_receipt,
            {"path": "r4-receipt"},
            unit_receipt_time,
        ),
    )
    scope_handoff = {"name": "handoff"}
    stability_attempt = {"name": "attempt"}
    policy = {"created_at_utc": "2026-07-31T10:02:00.000000Z"}
    stability = {"passed": True}
    postcleanup = {"created_at_utc": "2026-07-31T10:03:00.000000Z"}
    environment_roots = {
        "environment_scope_handoff": {"path": "handoff"},
        "environment_stability_attempt": {"path": "attempt"},
        "environment_policy": {"path": "policy"},
        "environment_stability": {"path": "stability"},
        "environment_postcleanup": {"path": "postcleanup"},
    }
    monkeypatch.setattr(
        bridge,
        "_load_environment_evidence",
        lambda: (
            scope_handoff,
            stability_attempt,
            policy,
            stability,
            postcleanup,
            environment_roots,
        ),
    )
    authorization = {
        "protected_unit_states": protected,
        "preauthorization_target_unit_state": _missing_state(
            bridge.C4_UNIT_NAME
        ),
        "compatibility_source_roots": sources,
        "c1_failure_terminal_root": c1_terminal_root,
        "c1_expired_authorization_root": c1_terminal["evidence_roots"][
            "bridge_authorization"
        ],
        "c2_mode_contract_failure_terminal_root": failure_root,
        "c2_prewrite_failure_terminal_root": prewrite_root,
        "c3_environment_failure_terminal_root": c3_root,
        "r10_roots": bridge._r10_roots_from_terminal(c1_terminal),
    }
    closure = bridge._collect_full_closure(
        authorization=authorization,
        authorization_root={"path": "b4-authorization"},
        unit_state_reader=lambda _unit: {},
        allow_runtime_activation=False,
        receipt_time=datetime(2026, 7, 31, 10, 4, tzinfo=timezone.utc),
    )
    assert closure["evidence_roots"]["c3_environment_failure_terminal"] == c3_root
    assert closure["evidence_roots"]["unit_realization_receipt"] == {
        "path": "r4-receipt"
    }
    assert closure["evidence_roots"]["environment_scope_handoff"] == {
        "path": "handoff"
    }
    assert closure["evidence_roots"]["environment_stability_attempt"] == {
        "path": "attempt"
    }
    assert closure["scope_handoff"] == scope_handoff
    assert closure["stability_attempt"] == stability_attempt
    assert environment.seen is not None


def test_no_scientific_or_fixed_absolute_performance_authority(bridge) -> None:
    authority = bridge._expected_scientific_authority()
    assert authority == {
        "D_R_payload_authorized": False,
        "D_V_payload_authorized": False,
        "D_T_payload_authorized": False,
        "training_authorized": False,
        "materialization_authorized": False,
        "automatic_retry": False,
        "resume": False,
        "fresh_scientific_attempt": False,
    }
    names = set(bridge._AUTHORIZATION_KEYS) | set(bridge._RECEIPT_KEYS)
    assert not any("absolute" in name or "threshold" in name for name in names)


def test_cli_is_c4_only(bridge) -> None:
    parser = bridge.build_parser()
    args = parser.parse_args(
        [
            "authorize-c4",
            "--instruction-id",
            bridge.INSTRUCTION_ID,
            "--authorization-basis",
            bridge.AUTHORIZATION_BASIS,
        ]
    )
    assert args.command == "authorize-c4"
    with pytest.raises(SystemExit):
        parser.parse_args(["authorize-c3"])
