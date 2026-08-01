from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY
    / "tools/"
    "cure_lite_v24_preaccess_compat_c3_"
    "environment_stability_failure_terminal.py"
)


def _load():
    name = (
        "cure_lite_v24_preaccess_compat_c3_"
        "environment_stability_failure_terminal_tested"
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
    identity = (
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
        return identity, hashlib.sha256(target.read_bytes()).hexdigest()
    return identity, None


def _rewrite_terminal(module, path: Path, payload: dict[str, object]) -> None:
    body = deepcopy(payload)
    body.pop("terminal_fingerprint", None)
    body["terminal_fingerprint"] = module.stable_fingerprint(body)
    raw = module._canonical_bytes(body) + b"\n"
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(0o444)


def _static_state(module, **changes: str) -> dict[str, str]:
    state = {
        "Id": module.C3_UNIT_NAME,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "FragmentPath": str(module.C3_UNIT_FRAGMENT_PATH),
        "InvocationID": "",
        "Restart": "no",
        "NRestarts": "0",
    }
    state.update(changes)
    return state


def _runner_for(state: dict[str, str]):
    def runner(argv, **kwargs):
        assert argv[:4] == [
            "/usr/bin/systemctl",
            "--user",
            "show",
            state["Id"],
        ]
        assert argv[4] == "--no-pager"
        assert argv[5].startswith("--property=")
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": 10,
        }
        stdout = "".join(
            f"{key}={state[key]}\n"
            for key in reversed(tuple(state))
        )
        return SimpleNamespace(
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    return runner


def _isolate_absences(module, tmp_path: Path, monkeypatch) -> dict[str, Path]:
    paths = {
        label: tmp_path / "future" / f"{index:02d}-{label}"
        for index, label in enumerate(module.ABSENT_OUTPUT_PATHS)
    }
    monkeypatch.setattr(module, "ABSENT_OUTPUT_PATHS", paths)
    return paths


def _assert_forensic_partial(module, selected: Path) -> None:
    assert os.path.lexists(selected)
    assert stat.S_IMODE(selected.stat().st_mode) == 0o400
    assert selected.stat().st_nlink == 1
    payload = json.loads(selected.read_bytes())
    assert payload["schema_version"] == module.SCHEMA
    assert "terminal_fingerprint" in payload


def test_fixed_paths_hashes_and_real_sealed_inputs(terminalizer) -> None:
    assert terminalizer.TERMINAL_PATH.name == (
        "r2_preaccess_schema_compat_c3_"
        "environment_stability_failure_terminal.json"
    )
    assert terminalizer.SCHEMA == (
        "cure-lite-v24-r2-preaccess-schema-compat-c3-"
        "environment-stability-failure-terminal-v1"
    )
    assert terminalizer.C3_UNIT_FRAGMENT_PATH == (
        Path(f"/run/user/{os.getuid()}/systemd/user")
        / terminalizer.C3_UNIT_NAME
    )

    for path, digest in terminalizer.FAILED_SOURCE_BINDINGS.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    fixed_files = (
        (
            terminalizer.B3_AUTHORIZATION_PATH,
            terminalizer.B3_AUTHORIZATION_SHA256,
            0o444,
        ),
        (
            terminalizer.R3_AUTHORIZATION_PATH,
            terminalizer.R3_AUTHORIZATION_SHA256,
            0o444,
        ),
        (
            terminalizer.R3_RECEIPT_PATH,
            terminalizer.R3_RECEIPT_SHA256,
            0o444,
        ),
        (
            terminalizer.C3_POLICY_PATH,
            terminalizer.C3_POLICY_SHA256,
            0o444,
        ),
        (
            terminalizer.PRECLEANUP_PATH,
            terminalizer.PRECLEANUP_SHA256,
            0o444,
        ),
        (
            terminalizer.CLEANUP_RECEIPT_PATH,
            terminalizer.CLEANUP_RECEIPT_SHA256,
            0o444,
        ),
        (
            terminalizer.C3_UNIT_FRAGMENT_PATH,
            terminalizer.C3_FRAGMENT_SHA256,
            0o600,
        ),
    )
    for path, digest, mode in fixed_files:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert stat.S_IMODE(path.stat().st_mode) == mode

    values, roots, fingerprints = terminalizer._validate_fixed_inputs()
    assert set(values) == {
        "B3_authorization",
        "R3_authorization",
        "R3_receipt",
        "historical_precleanup",
        "fixed_cleanup_receipt",
        "C3_environment_policy",
    }
    assert set(roots) == set(fingerprints) == {
        *values,
        "C3_unit_fragment",
    }
    assert values["B3_authorization"]["expires_at_utc"] == (
        terminalizer.B3_AUTHORIZATION_EXPIRES_AT
    )
    assert values["R3_authorization"]["expires_at_utc"] == (
        terminalizer.R3_AUTHORIZATION_EXPIRES_AT
    )
    assert values["R3_receipt"]["passed"] is True
    assert values["R3_receipt"]["static"] is True
    assert values["R3_receipt"]["enabled"] is False
    assert values["R3_receipt"]["started"] is False
    assert values["fixed_cleanup_receipt"]["passed"] is True
    assert values["fixed_cleanup_receipt"][
        "cleanup_receipt_fingerprint"
    ] == terminalizer.CLEANUP_RECEIPT_FINGERPRINT
    assert roots["C3_unit_fragment"]["file_sha256"] == (
        terminalizer.C3_FRAGMENT_SHA256
    )


def test_tmp_cleanup_receipt_canonical_and_semantic_tamper_rejected(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = json.loads(terminalizer.CLEANUP_RECEIPT_PATH.read_bytes())

    semantic = deepcopy(original)
    semantic["passed"] = False
    semantic_body = dict(semantic)
    semantic_body.pop("cleanup_receipt_fingerprint")
    semantic_fingerprint = terminalizer._fingerprint(
        semantic_body,
        ensure_ascii=False,
    )
    semantic["cleanup_receipt_fingerprint"] = semantic_fingerprint
    semantic_raw = terminalizer._canonical_bytes(semantic) + b"\n"
    semantic_path = tmp_path / "semantic-cleanup.json"
    semantic_path.write_bytes(semantic_raw)
    semantic_path.chmod(0o444)
    monkeypatch.setattr(
        terminalizer,
        "CLEANUP_RECEIPT_PATH",
        semantic_path,
    )
    monkeypatch.setattr(
        terminalizer,
        "CLEANUP_RECEIPT_SHA256",
        hashlib.sha256(semantic_raw).hexdigest(),
    )
    monkeypatch.setattr(
        terminalizer,
        "CLEANUP_RECEIPT_FINGERPRINT",
        semantic_fingerprint,
    )
    with pytest.raises(PermissionError, match="cleanup receipt lineage"):
        terminalizer._validate_cleanup_receipt()

    noncanonical_raw = (
        json.dumps(original, indent=2, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    noncanonical_path = tmp_path / "noncanonical-cleanup.json"
    noncanonical_path.write_bytes(noncanonical_raw)
    noncanonical_path.chmod(0o444)
    monkeypatch.setattr(
        terminalizer,
        "CLEANUP_RECEIPT_PATH",
        noncanonical_path,
    )
    monkeypatch.setattr(
        terminalizer,
        "CLEANUP_RECEIPT_SHA256",
        hashlib.sha256(noncanonical_raw).hexdigest(),
    )
    monkeypatch.setattr(
        terminalizer,
        "CLEANUP_RECEIPT_FINGERPRINT",
        original["cleanup_receipt_fingerprint"],
    )
    with pytest.raises(PermissionError, match="layout changed|not canonical"):
        terminalizer._validate_cleanup_receipt()


def test_scope_reproduction_is_pure_exact_and_nonmutating(
    terminalizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watched = {
        "B3": terminalizer.B3_AUTHORIZATION_PATH,
        "R3_auth": terminalizer.R3_AUTHORIZATION_PATH,
        "R3_receipt": terminalizer.R3_RECEIPT_PATH,
        "policy": terminalizer.C3_POLICY_PATH,
        "precleanup": terminalizer.PRECLEANUP_PATH,
        "cleanup": terminalizer.CLEANUP_RECEIPT_PATH,
        "fragment": terminalizer.C3_UNIT_FRAGMENT_PATH,
        "production_terminal": terminalizer.TERMINAL_PATH,
    }
    before = {label: _snapshot(path) for label, path in watched.items()}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reproduction reached a prohibited live API")

    monkeypatch.setattr(terminalizer, "_utc_now", forbidden)
    monkeypatch.setattr(terminalizer, "_write_create_once", forbidden)
    monkeypatch.setattr(terminalizer, "_read_c3_unit_state", forbidden)
    monkeypatch.setattr(terminalizer.subprocess, "run", forbidden)

    reproduction = terminalizer._reproduce_scope_mismatch()

    assert reproduction["observation_kind"] == (
        "post_hoc_deterministic_read_only_reproduction"
    )
    assert reproduction["observed_precleanup_scope"] == {
        "target_unit_id": terminalizer.OLD_TARGET_UNIT,
        "conflict_unit_ids": [
            "confa-v41-mshnet-nudt-clean-formal-20260718-v1.service"
        ],
        "dependency_unit_ids": [],
        "require_target_ready": False,
    }
    assert reproduction["requested_C3_scope"] == {
        "target_unit_id": terminalizer.C3_UNIT_NAME,
        "conflict_unit_ids": [
            "confa-v41-mshnet-nudt-clean-formal-20260718-v1.service"
        ],
        "dependency_unit_ids": [],
        "require_target_ready": True,
    }
    assert reproduction["mismatch_fields"] == [
        "target_unit_id",
        "require_target_ready",
    ]
    assert reproduction["exception_type"] == "PermissionError"
    assert reproduction["exception_message"] == terminalizer.EXPECTED_ERROR
    assert reproduction["exception_args"] == [terminalizer.EXPECTED_ERROR]
    assert reproduction["sealed_cleanup_receipt_root"]["file_sha256"] == (
        terminalizer.CLEANUP_RECEIPT_SHA256
    )
    assert reproduction["sealed_cleanup_receipt_fingerprint"] == (
        terminalizer.CLEANUP_RECEIPT_FINGERPRINT
    )
    assert reproduction["samples_collected"] == 0
    for field in (
        "E3_module_loaded",
        "E3_gate_invoked",
        "inventory_collector_invoked",
        "activation_guard_reader_invoked",
        "sleeper_invoked",
        "monotonic_clock_invoked",
        "writer_invoked",
    ):
        assert reproduction[field] is False
    assert {label: _snapshot(path) for label, path in watched.items()} == before


@pytest.mark.parametrize(
    "policy_update,error",
    (
        ({"require_target_ready": False}, "no longer exact"),
        (
            {
                "conflict_unit_ids": [
                    "confa-v41-mshnet-nudt-clean-formal-20260718-v1.service",
                    "unexpected.service",
                ]
            },
            "no longer exact",
        ),
    ),
)
def test_scope_reproduction_rejects_nonexact_mismatch(
    terminalizer,
    monkeypatch: pytest.MonkeyPatch,
    policy_update: dict[str, object],
    error: str,
) -> None:
    precleanup, precleanup_root = terminalizer._validate_precleanup()
    policy, policy_root = terminalizer._validate_policy()
    changed = deepcopy(policy)
    changed["unit_scope"].update(policy_update)
    monkeypatch.setattr(
        terminalizer,
        "_validate_precleanup",
        lambda: (deepcopy(precleanup), deepcopy(precleanup_root)),
    )
    monkeypatch.setattr(
        terminalizer,
        "_validate_policy",
        lambda: (deepcopy(changed), deepcopy(policy_root)),
    )
    with pytest.raises(PermissionError, match=error):
        terminalizer._reproduce_scope_mismatch()


def test_injected_unit_state_is_exact_and_fail_closed(terminalizer) -> None:
    state = _static_state(terminalizer)
    assert terminalizer._read_c3_unit_state(
        runner=_runner_for(state)
    ) == state

    invoked = _static_state(terminalizer, InvocationID="nonempty")
    with pytest.raises(PermissionError, match="not exact static/inert"):
        terminalizer._read_c3_unit_state(
            runner=_runner_for(invoked)
        )

    failed = lambda *_args, **_kwargs: SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="manager failed",
    )
    with pytest.raises(PermissionError, match="state query failed"):
        terminalizer._read_c3_unit_state(runner=failed)


def test_historical_observation_uses_injected_state_and_tmp_absences(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _isolate_absences(terminalizer, tmp_path, monkeypatch)
    observed_at = datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc)
    state = _static_state(terminalizer)
    observed = terminalizer._collect_historical_state_observation(
        observed_at=observed_at,
        runner=_runner_for(state),
    )
    assert observed["all_required_paths_absent"] is True
    assert observed["exact_absent_paths"] == {
        label: str(path.absolute()) for label, path in paths.items()
    }
    assert observed["C3_unit_state"] == state
    assert observed["historical_observation_only"] is True
    assert observed["archival_live_absence_recheck_required"] is False
    assert observed["archival_live_manager_recheck_required"] is False

    existing = next(iter(paths.values()))
    existing.parent.mkdir(parents=True)
    existing.write_text("later output", encoding="utf-8")
    with pytest.raises(PermissionError, match="future output exists"):
        terminalizer._collect_historical_state_observation(
            observed_at=observed_at,
            runner=_runner_for(state),
        )


def test_writer_forces_0444_exclusive_and_exact_fd_readback(
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
    assert path.stat().st_nlink == 1
    assert path.read_bytes() == terminalizer._canonical_bytes(payload) + b"\n"
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


def test_writer_detects_parent_identity_race(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "parent-race.json"
    monkeypatch.setattr(
        terminalizer,
        "validate_archival",
        lambda *_args: ({}, {}),
    )
    real_fstat = terminalizer.os.fstat
    directory_fstat_calls = 0

    def changed_fstat(descriptor):
        nonlocal directory_fstat_calls
        observed = real_fstat(descriptor)
        if stat.S_ISDIR(observed.st_mode):
            directory_fstat_calls += 1
            if directory_fstat_calls == 2:
                return SimpleNamespace(
                    st_dev=observed.st_dev,
                    st_ino=observed.st_ino + 1,
                )
        return observed

    monkeypatch.setattr(terminalizer.os, "fstat", changed_fstat)
    with pytest.raises(PermissionError, match="parent changed after create"):
        terminalizer._write_create_once(
            path,
            {"schema_version": "parent-race"},
        )


def test_writer_detects_parent_path_rebinding(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "parent-rebind.json"
    monkeypatch.setattr(
        terminalizer,
        "validate_archival",
        lambda *_args: ({}, {}),
    )
    real_lstat = terminalizer.Path.lstat
    parent_lstat_calls = 0

    def changed_lstat(selected):
        nonlocal parent_lstat_calls
        observed = real_lstat(selected)
        if selected == tmp_path:
            parent_lstat_calls += 1
            if parent_lstat_calls == 2:
                fields = list(observed)
                fields[1] = observed.st_ino + 1
                return os.stat_result(fields)
        return observed

    monkeypatch.setattr(terminalizer.Path, "lstat", changed_lstat)
    with pytest.raises(PermissionError, match="parent changed after create"):
        terminalizer._write_create_once(
            path,
            {"schema_version": "parent-rebind"},
        )


def test_temp_full_create_and_archival_never_rechecks_live_state(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = terminalizer.TERMINAL_PATH
    production_before = _snapshot(production)
    selected = tmp_path / production.name
    monkeypatch.setattr(terminalizer, "TERMINAL_PATH", selected)
    paths = _isolate_absences(terminalizer, tmp_path, monkeypatch)
    observed_at = datetime(
        2026,
        7,
        31,
        10,
        0,
        0,
        123456,
        tzinfo=timezone.utc,
    )
    monkeypatch.setattr(terminalizer, "_utc_now", lambda: observed_at)
    state = _static_state(terminalizer)

    payload = terminalizer.create_terminal(
        runner=_runner_for(state),
    )
    assert stat.S_IMODE(selected.stat().st_mode) == 0o444
    assert selected.stat().st_nlink == 1
    assert selected.read_bytes() == (
        terminalizer._canonical_bytes(payload) + b"\n"
    )
    failure = payload["environment_stability_failure"]
    assert failure["known_subcommand"] == "stability-gate"
    assert failure["known_cleanup_input_path"] == str(
        terminalizer.CLEANUP_RECEIPT_PATH
    )
    assert failure["original_call_artifact_available"] is False
    assert failure["original_argv_claimed"] is False
    assert failure["original_traceback_artifact_available"] is False
    assert failure["original_traceback_claimed"] is False
    assert failure["attempt_count"] == 1
    assert failure["retry"] is False
    assert failure["samples_collected"] == 0
    assert payload["authorization_expiry"]["B3_expired"] is True
    assert payload["authorization_expiry"]["R3_expired"] is True
    assert payload["authorization_expiry"][
        "B3_sealed_by_compatibility_receipt"
    ] is False
    assert payload["unit_realization_closure"] == (
        terminalizer._UNIT_REALIZATION_CLOSURE
    )
    assert payload["payload_observation"] == terminalizer._PAYLOAD_OBSERVATION
    assert payload["payload_observation"]["optimizer_steps"] == 0
    assert payload["continuation_policy"] == (
        terminalizer._CONTINUATION_POLICY
    )
    assert payload["continuation_policy"]["same_c3_reentry"] is False
    assert payload["continuation_policy"]["c4_required"] is True
    assert payload["continuation_policy"][
        "scientific_attempt_consumed"
    ] is False
    assert payload["continuation_policy"][
        "unit_realization_consumed"
    ] is True

    # Later state is deliberately different.  An archival read must not
    # reinterpret the historical absence or manager-state observation.
    later = next(iter(paths.values()))
    later.parent.mkdir(parents=True, exist_ok=True)
    later.write_text("later state", encoding="utf-8")

    def forbidden_live(*_args, **_kwargs):
        raise AssertionError("archival validation consulted live state")

    monkeypatch.setattr(
        terminalizer,
        "_collect_historical_state_observation",
        forbidden_live,
    )
    monkeypatch.setattr(terminalizer, "_read_c3_unit_state", forbidden_live)
    monkeypatch.setattr(terminalizer, "_utc_now", forbidden_live)
    monkeypatch.setattr(terminalizer.subprocess, "run", forbidden_live)

    archived, root = terminalizer.validate_archival()
    assert archived == payload
    assert root["mode"] == 0o444
    assert root["schema_version"] == terminalizer.SCHEMA
    assert root["terminal_fingerprint"] == payload["terminal_fingerprint"]
    assert root["terminalizer_source_root"] == (
        payload["terminalizer_source_root"]
    )
    assert _snapshot(production) == production_before

    before_duplicate = selected.read_bytes()
    with pytest.raises(FileExistsError):
        terminalizer.create_terminal(runner=forbidden_live)
    assert selected.read_bytes() == before_duplicate


def test_while_open_future_output_drift_fails_closed_and_keeps_partial(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = terminalizer.TERMINAL_PATH
    production_before = _snapshot(production)
    selected = tmp_path / production.name
    monkeypatch.setattr(terminalizer, "TERMINAL_PATH", selected)
    paths = _isolate_absences(terminalizer, tmp_path, monkeypatch)
    monkeypatch.setattr(
        terminalizer,
        "_utc_now",
        lambda: datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
    )
    original_guard = terminalizer._revalidate_open_creation_guard
    appeared = next(iter(paths.values()))

    def inject_future_output(**kwargs):
        appeared.parent.mkdir(parents=True, exist_ok=True)
        appeared.write_text("appeared while fd open", encoding="utf-8")
        return original_guard(**kwargs)

    monkeypatch.setattr(
        terminalizer,
        "_revalidate_open_creation_guard",
        inject_future_output,
    )
    with pytest.raises(PermissionError, match="future output appeared"):
        terminalizer.create_terminal(
            runner=_runner_for(_static_state(terminalizer)),
        )
    _assert_forensic_partial(terminalizer, selected)
    assert _snapshot(production) == production_before


def test_while_open_unit_drift_fails_closed_and_keeps_partial(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = terminalizer.TERMINAL_PATH
    production_before = _snapshot(production)
    selected = tmp_path / production.name
    monkeypatch.setattr(terminalizer, "TERMINAL_PATH", selected)
    _isolate_absences(terminalizer, tmp_path, monkeypatch)
    monkeypatch.setattr(
        terminalizer,
        "_utc_now",
        lambda: datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
    )
    calls = 0

    def drifting_runner(argv, **kwargs):
        nonlocal calls
        calls += 1
        state = _static_state(
            terminalizer,
            **({} if calls == 1 else {"NRestarts": "1"}),
        )
        return _runner_for(state)(argv, **kwargs)

    with pytest.raises(PermissionError, match="unit state drifted"):
        terminalizer.create_terminal(runner=drifting_runner)
    assert calls == 2
    _assert_forensic_partial(terminalizer, selected)
    assert _snapshot(production) == production_before


def test_while_open_source_root_drift_fails_closed_and_keeps_partial(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = terminalizer.TERMINAL_PATH
    production_before = _snapshot(production)
    selected = tmp_path / production.name
    monkeypatch.setattr(terminalizer, "TERMINAL_PATH", selected)
    _isolate_absences(terminalizer, tmp_path, monkeypatch)
    monkeypatch.setattr(
        terminalizer,
        "_utc_now",
        lambda: datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
    )
    original_guard = terminalizer._revalidate_open_creation_guard

    def inject_source_drift(**kwargs):
        real_reader = terminalizer._fixed_generation_roots

        def changed_roots():
            roots = deepcopy(real_reader())
            roots["bridge_B3"]["file_sha256"] = "0" * 64
            return roots

        terminalizer._fixed_generation_roots = changed_roots
        try:
            return original_guard(**kwargs)
        finally:
            terminalizer._fixed_generation_roots = real_reader

    monkeypatch.setattr(
        terminalizer,
        "_revalidate_open_creation_guard",
        inject_source_drift,
    )
    with pytest.raises(PermissionError, match="input/source drifted"):
        terminalizer.create_terminal(
            runner=_runner_for(_static_state(terminalizer)),
        )
    _assert_forensic_partial(terminalizer, selected)
    assert _snapshot(production) == production_before


def test_archival_rejects_terminalizer_absence_and_reproduction_tamper(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / terminalizer.TERMINAL_PATH.name
    monkeypatch.setattr(terminalizer, "TERMINAL_PATH", selected)
    _isolate_absences(terminalizer, tmp_path, monkeypatch)
    monkeypatch.setattr(
        terminalizer,
        "_utc_now",
        lambda: datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
    )
    payload = terminalizer.create_terminal(
        runner=_runner_for(_static_state(terminalizer)),
    )
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
    changed_absence["historical_state_observation"][
        "exact_absent_paths"
    ].pop("C3_environment_stability")
    _rewrite_terminal(terminalizer, selected, changed_absence)
    with pytest.raises(PermissionError, match="historical C3 state"):
        terminalizer.validate_archival()

    selected.chmod(0o600)
    selected.write_bytes(valid)
    selected.chmod(0o444)
    changed_cleanup = deepcopy(payload)
    changed_cleanup["sealed_input_roots"]["fixed_cleanup_receipt"][
        "root"
    ]["file_sha256"] = "0" * 64
    _rewrite_terminal(terminalizer, selected, changed_cleanup)
    with pytest.raises(PermissionError, match="sealed C3 input lineage"):
        terminalizer.validate_archival()

    selected.chmod(0o600)
    selected.write_bytes(valid)
    selected.chmod(0o444)
    changed_reproduction = deepcopy(payload)
    changed_reproduction["deterministic_reproduction"][
        "requested_C3_scope"
    ]["target_unit_id"] = "near-c3.service"
    _rewrite_terminal(terminalizer, selected, changed_reproduction)
    with pytest.raises(
        PermissionError,
        match="failure semantics",
    ):
        terminalizer.validate_archival()

    selected.chmod(0o600)
    selected.write_bytes(valid)
    selected.chmod(0o444)
    changed_type = deepcopy(payload)
    changed_type["payload_observation"]["gpu_accessed"] = 0
    _rewrite_terminal(terminalizer, selected, changed_type)
    with pytest.raises(PermissionError, match="failure semantics"):
        terminalizer.validate_archival()


def test_create_requires_both_authorizations_expired_before_writer(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / terminalizer.TERMINAL_PATH.name
    monkeypatch.setattr(terminalizer, "TERMINAL_PATH", selected)
    _isolate_absences(terminalizer, tmp_path, monkeypatch)
    monkeypatch.setattr(
        terminalizer,
        "_utc_now",
        lambda: datetime(
            2026,
            7,
            31,
            9,
            21,
            50,
            tzinfo=timezone.utc,
        ),
    )

    def forbidden_writer(*_args, **_kwargs):
        raise AssertionError("writer reached before both expirations")

    monkeypatch.setattr(terminalizer, "_write_create_once", forbidden_writer)
    with pytest.raises(PermissionError, match="not both expired"):
        terminalizer.create_terminal(
            runner=_runner_for(_static_state(terminalizer)),
        )
    assert not os.path.lexists(selected)


def test_create_terminal_rejects_nonfixed_path(
    terminalizer,
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="path is not fixed"):
        terminalizer.create_terminal(tmp_path / "not-fixed.json")
