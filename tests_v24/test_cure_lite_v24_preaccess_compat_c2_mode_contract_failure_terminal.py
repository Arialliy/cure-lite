from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY
    / "tools/"
    "cure_lite_v24_preaccess_compat_c2_mode_contract_failure_terminal.py"
)


def _load():
    name = (
        "cure_lite_v24_preaccess_compat_c2_"
        "mode_contract_failure_terminal_tested"
    )
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def terminalizer():
    return _load()


def _snapshot(path: Path):
    target = Path(path)
    if not os.path.lexists(target):
        return None
    observed = target.lstat()
    result = (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_uid,
        observed.st_gid,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )
    if stat.S_ISREG(observed.st_mode):
        return result, hashlib.sha256(target.read_bytes()).hexdigest()
    return result, None


def _rewrite_terminal(module, path: Path, payload: dict[str, object]) -> None:
    body = deepcopy(payload)
    body.pop("terminal_fingerprint", None)
    body["terminal_fingerprint"] = module.stable_fingerprint(body)
    raw = module._canonical_bytes(body) + b"\n"
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(0o444)


def test_fixed_names_hashes_and_real_failed_authorization(
    terminalizer,
) -> None:
    assert terminalizer.TERMINAL_PATH.name == (
        "r2_preaccess_schema_compat_c2_"
        "mode_contract_failure_terminal.json"
    )
    assert terminalizer.SCHEMA == (
        "cure-lite-v24-r2-preaccess-schema-compat-c2-"
        "mode-contract-failure-terminal-v1"
    )
    assert terminalizer.C2_UNIT_FRAGMENT_PATH == (
        Path(f"/run/user/{os.getuid()}/systemd/user")
        / terminalizer.C2_UNIT_NAME
    )

    for path, digest in terminalizer.FAILED_SOURCE_BINDINGS.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for path, digest in terminalizer.R12_FILE_BINDINGS.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert stat.S_IMODE(path.stat().st_mode) == 0o444

    authorization, root = terminalizer._validate_failed_authorization()
    assert root["file_sha256"] == (
        terminalizer.FAILED_AUTHORIZATION_SHA256
    )
    assert root["mode"] == 0o400
    assert authorization["authorization_fingerprint"] == (
        terminalizer.FAILED_AUTHORIZATION_FINGERPRINT
    )
    assert authorization["expires_at_utc"] == (
        "2026-07-30T15:42:31.190746Z"
    )
    raw = terminalizer.FAILED_AUTHORIZATION_PATH.read_bytes()
    assert b"\\u4fee\\u6539\\u540e\\u8fd0\\u884c" in raw
    assert "修改后运行".encode("utf-8") not in raw


def test_real_read_only_reproduction_is_exact_and_nonmutating(
    terminalizer,
) -> None:
    watched = {
        label: _snapshot(path)
        for label, path in terminalizer.ABSENT_OUTPUT_PATHS.items()
    }
    authorization_before = _snapshot(
        terminalizer.FAILED_AUTHORIZATION_PATH
    )
    production_terminal_before = _snapshot(terminalizer.TERMINAL_PATH)

    reproduction = terminalizer._reproduce_failure()

    assert reproduction == {
        "validator": "_validate_c2_bridge_authorization",
        "require_fresh": False,
        "require_future_absence": True,
        "observation_kind": "read_only_post_hoc_reproduction",
        "write_capable_entrypoint_invoked": False,
        "exception_type": "PermissionError",
        "exception_message": terminalizer.EXPECTED_ERROR,
        "exception_args": [terminalizer.EXPECTED_ERROR],
        "reproduced": True,
    }
    assert {
        label: _snapshot(path)
        for label, path in terminalizer.ABSENT_OUTPUT_PATHS.items()
    } == watched
    assert _snapshot(terminalizer.FAILED_AUTHORIZATION_PATH) == (
        authorization_before
    )
    assert _snapshot(terminalizer.TERMINAL_PATH) == (
        production_terminal_before
    )


def test_real_historical_absence_includes_exact_runtime_fragment(
    terminalizer,
) -> None:
    bridge, _root = terminalizer._load_verified_module(
        terminalizer.BRIDGE_PATH,
        expected_sha256=(
            terminalizer.FAILED_SOURCE_BINDINGS["bridge_B"][1]
        ),
        name="tools._mode_contract_test_bridge",
    )
    try:
        observed = terminalizer._collect_historical_absence_observation(
            bridge,
            observed_at=terminalizer._utc_now(),
        )
    finally:
        sys.modules.pop(bridge.__name__, None)
    assert observed["all_required_paths_absent"] is True
    assert observed["historical_observation_only"] is True
    assert observed["future_state_authority"] is False
    assert observed["archival_live_absence_recheck_required"] is False
    assert set(observed["exact_absent_paths"]) == set(
        terminalizer.ABSENT_OUTPUT_PATHS
    )
    assert observed["exact_absent_paths"]["unit_fragment"] == str(
        terminalizer.C2_UNIT_FRAGMENT_PATH
    )
    assert observed["c2_unit_state"]["LoadState"] == "not-found"
    assert observed["c2_unit_state"]["FragmentPath"] == ""


def test_writer_forces_0444_and_exact_fd_readback_under_restrictive_umask(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "terminal.json"

    def validate(selected):
        raw = Path(selected).read_bytes()
        payload = json.loads(raw)
        return payload, {
            "mode": stat.S_IMODE(Path(selected).stat().st_mode),
        }

    monkeypatch.setattr(terminalizer, "validate_archival", validate)
    previous = os.umask(0o077)
    try:
        payload = terminalizer._write_create_once(
            path,
            {"schema_version": "test-only"},
        )
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert path.read_bytes() == (
        terminalizer._canonical_bytes(payload) + b"\n"
    )
    with pytest.raises(FileExistsError):
        terminalizer._write_create_once(
            path,
            {"schema_version": "second-write"},
        )

    short = tmp_path / "short-readback.json"
    monkeypatch.setattr(terminalizer.os, "pread", lambda *_args: b"")
    with pytest.raises(PermissionError, match="fd seal/readback"):
        terminalizer._write_create_once(
            short,
            {"schema_version": "short-readback"},
        )


def test_temp_terminal_full_create_and_archival_without_live_absence_recheck(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = terminalizer.TERMINAL_PATH
    production_before = _snapshot(production)
    selected = tmp_path / production.name
    monkeypatch.setattr(terminalizer, "TERMINAL_PATH", selected)

    payload = terminalizer.create_terminal()
    assert stat.S_IMODE(selected.stat().st_mode) == 0o444
    assert selected.stat().st_nlink == 1
    assert selected.read_bytes() == (
        terminalizer._canonical_bytes(payload) + b"\n"
    )
    assert payload["failed_authorization"]["observed_mode"] == 0o400
    assert payload["mode_contract_failure"] == {
        "producer": (
            "cure_lite_v24_preaccess_schema_compatibility_c2."
            "_write_sealed"
        ),
        "consumer": (
            "cure_lite_v24_actual_unit_realization_"
            "preaccess_compat_c2._validate_c2_bridge_authorization"
        ),
        "producer_observed_mode": 0o400,
        "consumer_required_mode": 0o444,
        "mode_contract_mismatch": True,
        "original_write_capable_entrypoint_invoked": True,
        "original_call_artifact_claimed": False,
        "original_failure_time_claimed": False,
        "failed_before_unit_authorization_write": True,
        "unit_authorization_written": False,
        "unit_terminal_written": False,
    }
    assert payload["continuation_policy"] == (
        terminalizer._CONTINUATION_POLICY
    )
    assert payload["continuation_policy"]["automatic_retry"] is False
    assert payload["continuation_policy"]["same_c2_reentry"] is False
    assert payload["continuation_policy"]["c3_required"] is True
    assert (
        payload["continuation_policy"]["scientific_attempt_consumed"]
        is False
    )
    assert payload["payload_observation"] == (
        terminalizer._PAYLOAD_OBSERVATION
    )
    assert payload["payload_observation"]["optimizer_steps"] == 0
    assert payload["payload_observation"]["parameter_updates"] == 0

    def forbidden_live_observation(*_args, **_kwargs):
        raise AssertionError(
            "archival validation re-checked a live absence"
        )

    monkeypatch.setattr(
        terminalizer,
        "_collect_historical_absence_observation",
        forbidden_live_observation,
    )
    archived, root = terminalizer.validate_archival()
    assert archived == payload
    assert root["mode"] == 0o444
    assert root["terminalizer_source_root"] == (
        payload["terminalizer_source_root"]
    )
    assert _snapshot(production) == production_before

    before_duplicate = selected.read_bytes()
    with pytest.raises(FileExistsError):
        terminalizer.create_terminal()
    assert selected.read_bytes() == before_duplicate


def test_archival_rejects_recomputed_t_or_absence_lineage_tamper(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / terminalizer.TERMINAL_PATH.name
    monkeypatch.setattr(terminalizer, "TERMINAL_PATH", selected)
    payload = terminalizer.create_terminal()
    valid = selected.read_bytes()

    changed_t = deepcopy(payload)
    changed_t["terminalizer_source_root"]["file_sha256"] = "0" * 64
    _rewrite_terminal(terminalizer, selected, changed_t)
    with pytest.raises(PermissionError, match="terminalizer source lineage"):
        terminalizer.validate_archival()

    selected.chmod(0o600)
    selected.write_bytes(valid)
    selected.chmod(0o444)
    changed_absence = deepcopy(payload)
    changed_absence["historical_absence_observation"][
        "exact_absent_paths"
    ].pop("unit_fragment")
    _rewrite_terminal(terminalizer, selected, changed_absence)
    with pytest.raises(PermissionError, match="historical absence"):
        terminalizer.validate_archival()


@pytest.mark.parametrize(
    "raised",
    (
        PermissionError("near c2 bridge error"),
        RuntimeError("c2 bridge does not authorize the narrow unit lane"),
    ),
)
def test_reproduction_rejects_nonexact_exception(
    terminalizer,
    monkeypatch: pytest.MonkeyPatch,
    raised: BaseException,
) -> None:
    class FakeRealizer:
        __name__ = "tools._fake_mode_failure_realizer"

        @staticmethod
        def _validate_c2_bridge_authorization(**_kwargs):
            raise raised

    monkeypatch.setattr(
        terminalizer,
        "_load_verified_module",
        lambda *_args, **_kwargs: (FakeRealizer(), {}),
    )
    with pytest.raises(
        (PermissionError, RuntimeError),
        match="reproduce exactly|narrow unit lane",
    ):
        terminalizer._reproduce_failure()


def test_create_terminal_rejects_nonfixed_path(
    terminalizer,
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="path is not fixed"):
        terminalizer.create_terminal(tmp_path / "not-fixed.json")
