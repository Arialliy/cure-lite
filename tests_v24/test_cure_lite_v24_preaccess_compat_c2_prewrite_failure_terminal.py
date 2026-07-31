from __future__ import annotations

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
    / "tools/cure_lite_v24_preaccess_compat_c2_prewrite_failure_terminal.py"
)


def _load():
    name = "cure_lite_v24_c2_prewrite_failure_terminal_tested"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def terminalizer():
    return _load()


def _not_found_state(unit: str) -> dict[str, str]:
    return {
        "Id": unit,
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "NRestarts": "0",
        "FragmentPath": "",
        "InvocationID": "",
        "Restart": "no",
    }


def test_fixed_failed_generation_hashes_are_exact(terminalizer) -> None:
    expected = {
        "bridge_B0": (
            "6212d744bb644a857880212da698926fec95dbe23ac80c330e7e359cc00151aa"
        ),
        "realizer_R0": (
            "8d19ebc528d3b7b0a2ad0a8edd15b22aca401948c7981e8eedde67fdd1c42066"
        ),
        "supervisor_S0": (
            "83124d5d47333818126e19022eb5a848f4ae3c345831136ec80b6a7333bd6aab"
        ),
        "environment_E0": (
            "d6e5f38cc735fca5f069bb5dfbbf124c44513018765308918cfe8ff6ce269fee"
        ),
        "release_L0": (
            "becdd5f5e38e757d76318fe554ff72f9d33c421369856f134e0bc728c89f548a"
        ),
        "adapter": (
            "510826b8345948803cf976d180edb1764575874d22379e4223cb6f97dc96728f"
        ),
        "template": (
            "4e485e5ba86a79b9244fb73d5add1a7015d71aa5f16f56479bd9c2c200d12967"
        ),
    }
    assert {
        label: digest
        for label, (_path, digest) in terminalizer.FAILED_SOURCE_BINDINGS.items()
    } == expected


def test_r11_hashes_and_scenario_are_exact(terminalizer) -> None:
    assert terminalizer.R11_SCENARIO_ID.endswith(
        "compat-c2-r11-20260730c2000011"
    )
    assert {
        label: digest
        for label, (_path, digest) in terminalizer.R11_FILE_BINDINGS.items()
    } == {
        "authorization": (
            "00815dd1c6e78eae63fcbff9c49b7aeea9a9ca311ab455ddc89b401c0f9a3ccf"
        ),
        "integration_terminal": (
            "bf5882e9f89165e5c0b4223243fa5873b8893d75b45811770b5b9f53b53d0cb7"
        ),
        "receipt": (
            "1d415709b23173baa9710c73385e25a492972a5c491e281cc437133dcdce1eea"
        ),
        "removal_authorization": (
            "4b32e8204ea68d11daa10f50d35c8f608d9241852d5f91d36f9d711ff048bbc7"
        ),
        "removal_state": (
            "c19f5eb7495a7642ce91a52cd654a3f8e256ef324975fc47d1ea50f11c67bd59"
        ),
    }


def test_source_never_calls_write_capable_c2_authorizer() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "bridge.authorize_c2(" not in text
    assert "systemctl --user start" not in text
    compile(text, str(SOURCE), "exec")


def test_reproduction_accepts_only_exact_permission_error(
    terminalizer,
) -> None:
    class Bridge:
        _default_unit_state_reader = object()

        @staticmethod
        def _validate_c1_failure_terminal(**_kwargs):
            raise PermissionError(terminalizer.EXPECTED_ERROR)

    result = terminalizer._reproduce_failure(Bridge)
    assert result["reproduced"] is True
    assert result["write_capable_entrypoint_invoked"] is False
    assert result["original_call_id_claimed"] is False
    assert result["original_failure_time_claimed"] is False


def test_reproduction_rejects_wrong_permission_error(terminalizer) -> None:
    class Bridge:
        _default_unit_state_reader = object()

        @staticmethod
        def _validate_c1_failure_terminal(**_kwargs):
            raise PermissionError("different")

    with pytest.raises(PermissionError, match="did not reproduce exactly"):
        terminalizer._reproduce_failure(Bridge)


def test_reproduction_does_not_relabel_other_exception(terminalizer) -> None:
    class Bridge:
        _default_unit_state_reader = object()

        @staticmethod
        def _validate_c1_failure_terminal(**_kwargs):
            raise FileExistsError(terminalizer.EXPECTED_ERROR)

    with pytest.raises(FileExistsError, match=terminalizer.EXPECTED_ERROR):
        terminalizer._reproduce_failure(Bridge)


def test_absence_snapshot_requires_every_path_absent(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        "one": tmp_path / "one",
        "two": tmp_path / "two",
    }
    monkeypatch.setattr(terminalizer, "C2_OUTPUT_PATHS", paths)
    bridge = SimpleNamespace(
        _default_unit_state_reader=lambda unit: _not_found_state(unit)
    )
    observed = datetime(2026, 7, 30, tzinfo=timezone.utc)
    result = terminalizer._collect_absence_snapshot(
        bridge,
        observed_at=observed,
    )
    assert result["all_required_paths_absent"] is True
    assert result["scientific_attempt_consumed"] is False
    paths["two"].write_text("present", encoding="utf-8")
    with pytest.raises(PermissionError, match="already exists"):
        terminalizer._collect_absence_snapshot(
            bridge,
            observed_at=observed,
        )


def test_absence_snapshot_requires_not_found_unit(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        terminalizer,
        "C2_OUTPUT_PATHS",
        {"one": tmp_path / "one"},
    )
    state = _not_found_state(terminalizer.C2_UNIT_NAME)
    state["LoadState"] = "loaded"
    bridge = SimpleNamespace(
        _default_unit_state_reader=lambda _unit: state
    )
    with pytest.raises(PermissionError, match="not-found/inert"):
        terminalizer._collect_absence_snapshot(
            bridge,
            observed_at=datetime.now(timezone.utc),
        )


def test_create_body_is_failure_only_and_non_consuming(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "terminal.json"
    monkeypatch.setattr(terminalizer, "TERMINAL_PATH", output)
    roots = {
        "supervisor_S0": {"path": "supervisor", "file_sha256": "a" * 64},
        "c1_terminal": {"path": "c1", "file_sha256": "b" * 64},
    }
    monkeypatch.setattr(
        terminalizer, "_failed_generation_roots", lambda: roots
    )
    bridge = object()
    release = object()
    monkeypatch.setattr(
        terminalizer,
        "_load_verified_module",
        lambda path, **_kwargs: (
            bridge if path == terminalizer.BRIDGE_PATH else release
        ),
    )
    monkeypatch.setattr(
        terminalizer,
        "_reproduce_failure",
        lambda _bridge: {
            "validator": "_validate_c1_failure_terminal",
            "observation_kind": (
                "post_hoc_deterministic_reproduction_before_source_revision"
            ),
            "original_call_id_claimed": False,
            "original_failure_time_claimed": False,
            "write_capable_entrypoint_invoked": False,
            "exception_type": "PermissionError",
            "exception_message": terminalizer.EXPECTED_ERROR,
            "reproduced": True,
        },
    )
    identities = {"authorization": {"path": "auth"}}
    closure = {
        "authorization": {"scenario_id": terminalizer.R11_SCENARIO_ID},
        "receipt": {"passed": True, "fragment_removed": True},
        "identities": identities,
    }
    r11_roots = {
        label: {"path": label}
        for label in terminalizer.R11_FILE_BINDINGS
    }
    monkeypatch.setattr(
        terminalizer,
        "_validate_r11_pass",
        lambda _release: (closure, r11_roots),
    )
    monkeypatch.setattr(
        terminalizer,
        "_collect_absence_snapshot",
        lambda _bridge, **_kwargs: {
            "observed_at_utc": "2026-07-30T00:00:00.000000Z",
            "exact_absent_paths": {},
            "all_required_paths_absent": True,
            "c2_unit_state": {},
            "D_R_materialized": False,
            "D_V_materialized": False,
            "D_T_materialized": False,
            "scientific_attempt_consumed": False,
        },
    )
    monkeypatch.setattr(
        terminalizer,
        "_file_root",
        lambda _path, **_kwargs: {"path": "terminalizer"},
    )
    captured: dict[str, object] = {}

    def capture(_path, body):
        captured.update(body)
        return dict(body)

    monkeypatch.setattr(terminalizer, "_write_create_once", capture)
    terminalizer.create_terminal()
    assert captured["payload_observation"] == {
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_access_evidence_present": False,
        "training_evidence_present": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
    }
    assert captured["continuation_policy"] == {
        "automatic_retry": False,
        "same_source_reentry": False,
        "revised_source_manual_reentry": True,
        "fixed_r12_required": True,
        "c3_required": False,
        "scientific_attempt_consumed": False,
        "runtime_launch_consumed": False,
        "materialization_consumed": False,
    }


def _write_sealed(
    terminalizer,
    path: Path,
    body: dict[str, object],
) -> None:
    payload = dict(body)
    payload["terminal_fingerprint"] = terminalizer.stable_fingerprint(body)
    path.write_bytes(terminalizer._canonical_bytes(payload) + b"\n")
    path.chmod(0o444)


def test_archival_validation_does_not_require_live_c2_absence(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = tmp_path / "failure.json"
    c1 = tmp_path / "c1.json"
    c1.write_bytes(b"c1\n")
    c1.chmod(0o444)
    c1_digest = hashlib.sha256(c1.read_bytes()).hexdigest()
    monkeypatch.setattr(terminalizer, "TERMINAL_PATH", terminal)
    monkeypatch.setattr(terminalizer, "C1_TERMINAL_PATH", c1)
    monkeypatch.setattr(terminalizer, "C1_TERMINAL_SHA256", c1_digest)

    r11_bindings = {}
    r11_roots = {}
    for label in terminalizer.R11_FILE_BINDINGS:
        path = tmp_path / f"{label}.json"
        path.write_bytes((label + "\n").encode())
        path.chmod(0o444)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        r11_bindings[label] = (path, digest)
        r11_roots[label] = terminalizer._file_root(
            path,
            expected_sha256=digest,
            expected_mode=0o444,
        )
    monkeypatch.setattr(
        terminalizer, "R11_FILE_BINDINGS", r11_bindings
    )
    c1_root = terminalizer._file_root(
        c1,
        expected_sha256=c1_digest,
        expected_mode=0o444,
    )
    failed_roots = {
        label: {"path": str(path.absolute()), "file_sha256": digest}
        for label, (path, digest) in terminalizer.FAILED_SOURCE_BINDINGS.items()
    }
    failed_roots["c1_terminal"] = c1_root
    body = {
        "schema_version": terminalizer.SCHEMA,
        "identity": {
            "candidate": "GCR-PACRE-v24",
            "scientific_attempt_ordinal": 2,
            "runtime_compatibility_id": "c2",
        },
        "terminalizer_source_root": terminalizer._file_root(SOURCE),
        "failed_generation_roots": failed_roots,
        "r11_pass_closure": {
            "integration_root": str(terminalizer.R11_ROOT.absolute()),
            "scenario_id": terminalizer.R11_SCENARIO_ID,
            "authorization_root": r11_roots["authorization"],
            "integration_terminal_root": r11_roots[
                "integration_terminal"
            ],
            "receipt_root": r11_roots["receipt"],
            "removal_authorization_root": r11_roots[
                "removal_authorization"
            ],
            "removal_state_root": r11_roots["removal_state"],
            "supervisor_source_root": failed_roots["supervisor_S0"],
            "passed": True,
            "fragment_removed": True,
        },
        "deterministic_reproduction": {
            "exception_type": "PermissionError",
            "exception_message": terminalizer.EXPECTED_ERROR,
            "reproduced": True,
            "write_capable_entrypoint_invoked": False,
        },
        "c2_absence_snapshot": {
            "all_required_paths_absent": True,
            "scientific_attempt_consumed": False,
        },
        "payload_observation": {
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_access_evidence_present": False,
            "training_evidence_present": False,
            "optimizer_steps": 0,
            "parameter_updates": 0,
        },
        "continuation_policy": {
            "automatic_retry": False,
            "same_source_reentry": False,
            "revised_source_manual_reentry": True,
            "fixed_r12_required": True,
            "scientific_attempt_consumed": False,
        },
    }
    _write_sealed(terminalizer, terminal, body)

    # This path appeared after the historical observation.  Archival
    # validation must not inspect it or compare parent directory metadata.
    future = tmp_path / "future-c2-artifact"
    future.mkdir()
    payload, root = terminalizer.validate_archival()
    assert payload["continuation_policy"]["automatic_retry"] is False
    assert root["mode"] == 0o444


def test_stable_fingerprint_is_canonical(terminalizer) -> None:
    assert terminalizer.stable_fingerprint({"b": 2, "a": 1}) == (
        terminalizer.stable_fingerprint({"a": 1, "b": 2})
    )
    encoded = b'{"a":1,"b":2}'
    assert terminalizer.stable_fingerprint({"b": 2, "a": 1}) == (
        hashlib.sha256(encoded).hexdigest()
    )
