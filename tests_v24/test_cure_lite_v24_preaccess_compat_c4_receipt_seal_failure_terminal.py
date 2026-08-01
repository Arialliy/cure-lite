from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_compat_c4_"
    "receipt_seal_failure_terminal.py"
)
PRODUCTION_TERMINAL = (
    REPOSITORY
    / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
    "r2_preaccess_schema_compat_c4_receipt_seal_failure_terminal.json"
)


@pytest.fixture
def terminalizer():
    name = "tools._test_c4_receipt_seal_failure_terminal"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module


def test_production_terminal_is_valid_when_present(terminalizer) -> None:
    """Keep this suite rerunnable after the one-time production seal."""

    if not os.path.lexists(PRODUCTION_TERMINAL):
        pytest.skip("production C4 failure terminal is not sealed yet")
    payload, root = terminalizer.validate_archival(PRODUCTION_TERMINAL)
    assert payload["schema_version"] == terminalizer.SCHEMA
    assert root["mode"] == 0o444
    assert root["nlink"] == 1


def _state_stdout(module, unit_name: str, *, drift: bool = False) -> str:
    is_c4 = unit_name == module.C4_UNIT_NAME
    values = {
        "Id": unit_name,
        "LoadState": "loaded" if is_c4 else "not-found",
        "ActiveState": "active" if drift and is_c4 else "inactive",
        "SubState": "running" if drift and is_c4 else "dead",
        "UnitFileState": "static" if is_c4 else "",
        "FragmentPath": str(module.C4_UNIT_FRAGMENT_PATH) if is_c4 else "",
        "InvocationID": "drift" if drift and is_c4 else "",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
    }
    return "\n".join(f"{key}={value}" for key, value in values.items()) + "\n"


def _runner(module, *, drift_after: int | None = None):
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        unit_name = command[3]
        drift = drift_after is not None and calls >= drift_after
        return subprocess.CompletedProcess(
            command,
            0,
            _state_stdout(module, unit_name, drift=drift),
            "",
        )

    return run


def _rewrite_terminal(module, path: Path, mutator) -> None:
    os.chmod(path, 0o600)
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    body = dict(value)
    body.pop("terminal_fingerprint", None)
    value["terminal_fingerprint"] = module.stable_fingerprint(body)
    path.write_bytes(module._canonical_bytes(value) + b"\n")
    os.chmod(path, 0o444)


def test_fixed_paths_schema_and_source_hashes(terminalizer) -> None:
    module = terminalizer
    assert module.TERMINAL_PATH == PRODUCTION_TERMINAL
    assert module.SCHEMA == (
        "cure-lite-v24-r2-preaccess-schema-compat-c4-"
        "receipt-seal-failure-terminal-v1"
    )
    assert module.CANDIDATE == "GCR-PACRE-v24"
    assert module.RUNTIME_COMPATIBILITY_ID == "c4"
    assert module.SCIENTIFIC_ATTEMPT_ORDINAL == 2
    assert module.C4_UNIT_FRAGMENT_PATH == Path(
        "/run/user/1008/systemd/user/"
        "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service"
    )
    for path, expected in module.SOURCE_BINDINGS.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_frozen_inputs_and_b4_authorized_roots_close(terminalizer) -> None:
    module = terminalizer
    values, roots, raws, generations = module._validate_fixed_inputs()
    assert set(values) == set(module._INPUT_SPECS)
    assert roots["B4_authorization"]["file_sha256"] == (
        module.B4_AUTHORIZATION_SHA256
    )
    assert roots["R4_authorization"]["file_sha256"] == (
        module.R4_AUTHORIZATION_SHA256
    )
    assert roots["R4_receipt"]["file_sha256"] == module.R4_RECEIPT_SHA256
    assert roots["C4_unit_fragment"]["file_sha256"] == (
        module.C4_FRAGMENT_SHA256
    )
    assert set(generations["B4_authorized_source_roots"]) == set(
        module._AUTHORIZED_SOURCE_LABELS
    )
    assert set(generations["B4_authorized_live_generations"]) == set(
        module._AUTHORIZED_SOURCE_LABELS
    )
    for label, spec in module._INPUT_SPECS.items():
        assert hashlib.sha256(raws[label]).hexdigest() == spec[1]
        assert roots[label]["mode"] == 0o444
        assert roots[label]["nlink"] == 1


def test_b4_ascii_and_r4_utf8_profiles_are_exact(terminalizer) -> None:
    module = terminalizer
    values, _roots, raws, _generations = module._validate_fixed_inputs()
    b4 = values["B4_authorization"]
    b4_body = dict(b4)
    assert b4_body.pop("authorization_fingerprint") == (
        module.B4_AUTHORIZATION_FINGERPRINT
    )
    assert raws["B4_authorization"] == (
        module._canonical_bytes(b4, ensure_ascii=True) + b"\n"
    )
    assert module._fingerprint(b4_body, ensure_ascii=True) == (
        module.B4_AUTHORIZATION_FINGERPRINT
    )

    r4 = values["R4_authorization"]
    r4_body = dict(r4)
    assert r4_body.pop("authorization_fingerprint") == (
        module.R4_AUTHORIZATION_FINGERPRINT
    )
    assert raws["R4_authorization"] == (
        module._canonical_bytes(r4, ensure_ascii=False) + b"\n"
    )
    assert raws["R4_authorization"] != (
        module._canonical_bytes(r4, ensure_ascii=True) + b"\n"
    )
    assert module._fingerprint(r4_body, ensure_ascii=False) == (
        module.R4_AUTHORIZATION_FINGERPRINT
    )
    assert module._fingerprint(r4_body, ensure_ascii=True) == (
        module.R4_AUTHORIZATION_OLD_B4_ASCII_FINGERPRINT
    )


def test_deterministic_first_failure_and_dual_mode_control(
    terminalizer,
) -> None:
    module = terminalizer
    values, _roots, raws, _generations = module._validate_fixed_inputs()
    reproduction = module._deterministic_reproduction(values, raws)
    assert reproduction["observation_kind"] == (
        "post_hoc_pure_byte_recalculation"
    )
    assert reproduction["first_failure_reproduced"] is True
    assert reproduction["load_order"][0] == "R4_authorization"
    assert reproduction["R4_receipt_control"] == {
        "path": str(module.R4_RECEIPT_PATH.absolute()),
        "claimed_fingerprint": module.R4_RECEIPT_FINGERPRINT,
        "utf8_recomputed_fingerprint": module.R4_RECEIPT_FINGERPRINT,
        "ascii_recomputed_fingerprint": module.R4_RECEIPT_FINGERPRINT,
        "dual_profile_equal": True,
        "profile_mismatch": False,
    }
    latent = reproduction["E4_scope_handoff_latent_mismatch"]
    assert latent["producer_utf8_recomputed_fingerprint"] == (
        module.E4_SCOPE_HANDOFF_FINGERPRINT
    )
    assert latent["old_B4_ascii_recomputed_fingerprint"] == (
        module.E4_SCOPE_HANDOFF_OLD_B4_ASCII_FINGERPRINT
    )
    assert latent["producer_specific_loader_required"] is True
    assert reproduction["old_B4_module_imported"] is False
    assert reproduction["old_B4_seal_called"] is False
    assert reproduction["R4_E4_writer_called"] is False


def test_r4_e4_success_closure_and_payload_boundary(terminalizer) -> None:
    module = terminalizer
    values, roots, _raws, _generations = module._validate_fixed_inputs()
    closure = module._metadata_success_closure(values, roots)
    assert closure["r4_unit_realization_passed"] is True
    assert closure["r4_static_unit_verified"] is True
    assert closure["e4_scope_handoff_present"] is True
    assert closure["e4_stability_attempt_count"] == 1
    assert closure["e4_environment_sample_count"] == 2
    assert closure["e4_stability_passed"] is True
    assert closure["e4_postcleanup_passed"] is True
    assert closure["c4_compatibility_receipt_present"] is False
    assert set(closure["E4_five_piece_roots"]) == {
        "E4_scope_handoff",
        "E4_stability_attempt",
        "E4_policy",
        "E4_stability_receipt",
        "E4_postcleanup",
    }
    assert module._PAYLOAD_OBSERVATION["scientific_samples_processed"] == 0
    assert module._PAYLOAD_OBSERVATION["environment_metadata_samples"] == 2
    assert module._PAYLOAD_OBSERVATION["parameter_updates"] == 0


def test_failure_observation_is_honest_and_consumer_aligned(
    terminalizer,
) -> None:
    module = terminalizer
    failure = module._failure_record()
    observation = module._ORIGINAL_EXECUTION_OBSERVATION
    assert failure["first_rejected_path"] == str(
        module.R4_AUTHORIZATION_PATH.absolute()
    )
    assert failure["producer_canonical_profile"] == (
        "compact_sorted_ensure_ascii_false_utf8"
    )
    assert failure["consumer_canonical_profile"] == (
        "compact_sorted_ensure_ascii_true_utf8"
    )
    assert failure["receipt_writer_reached"] is False
    assert failure["receipt_sealed"] is False
    assert failure["expected_exception_observation_kind"] == (
        "post_hoc_deterministic_expectation"
    )
    assert observation["attempt_count"] == 1
    assert observation["control_plane_observed_exit_code"] == 1
    assert observation["durable_original_execution_artifact"] is False
    assert observation["durable_process_result_artifact_available"] is False
    assert observation["exit_code_independently_verifiable"] is False
    for name in ("argv", "stdout", "stderr", "traceback"):
        assert observation[f"original_{name}_claimed"] is False


def test_unit_state_reader_accepts_only_exact_static_and_not_found(
    terminalizer,
) -> None:
    module = terminalizer
    good = _runner(module)
    c4 = module._read_unit_state(
        module.C4_UNIT_NAME,
        expected="static",
        runner=good,
    )
    r14 = module._read_unit_state(
        module.R14_DUMMY_UNIT_NAME,
        expected="not-found",
        runner=good,
    )
    assert c4["InvocationID"] == ""
    assert c4["UnitFileState"] == "static"
    assert r14["LoadState"] == "not-found"
    with pytest.raises(PermissionError, match="not exact static/inert"):
        module._read_unit_state(
            module.C4_UNIT_NAME,
            expected="static",
            runner=_runner(module, drift_after=1),
        )


def test_fixed_absences_are_nofollow_parent_bound(terminalizer) -> None:
    module = terminalizer
    rows = module._observe_absences()
    assert set(rows) == set(module.ABSENT_OUTPUT_PATHS)
    assert module._exact_absent_paths() == {
        label: str(path.absolute())
        for label, path in module.ABSENT_OUTPUT_PATHS.items()
    }
    for label, row in rows.items():
        assert row["path"] == str(module.ABSENT_OUTPUT_PATHS[label].absolute())
        assert row["lexists"] is False
        assert row["parent_owner_uid"] == os.getuid()
        assert not row["parent_mode"] & 0o002


def test_duplicate_keys_and_nonfinite_json_are_rejected(
    terminalizer,
    tmp_path: Path,
) -> None:
    module = terminalizer
    path = tmp_path / "bad.json"
    with pytest.raises(PermissionError, match="malformed"):
        module._parse_json(b'{"x":1,"x":2}', path=path)
    with pytest.raises(PermissionError, match="malformed"):
        module._parse_json(b'{"x":NaN}', path=path)


def test_full_temp_create_is_exclusive_canonical_and_archival(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = terminalizer
    target = tmp_path / "terminal.json"
    monkeypatch.setattr(module, "TERMINAL_PATH", target)
    guard_calls = 0
    original_guard = module._revalidate_open_creation_guard

    def counted_guard(**kwargs):
        nonlocal guard_calls
        guard_calls += 1
        return original_guard(**kwargs)

    monkeypatch.setattr(
        module,
        "_revalidate_open_creation_guard",
        counted_guard,
    )
    payload = module.create_terminal(runner=_runner(module))
    assert guard_calls == 2
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert target.stat().st_nlink == 1
    assert target.read_bytes() == module._canonical_bytes(payload) + b"\n"
    assert set(payload) == set(module._TOP_LEVEL_KEYS)
    assert payload["terminal_fingerprint"] == module.stable_fingerprint(
        {key: value for key, value in payload.items() if key != "terminal_fingerprint"}
    )
    checked, root = module.validate_archival()
    assert checked == payload
    assert root["file_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert root["mode"] == 0o444
    assert root["schema_version"] == module.SCHEMA
    assert root["terminal_fingerprint"] == payload["terminal_fingerprint"]
    assert root["terminalizer_source_root"] == payload[
        "terminalizer_source_root"
    ]
    with pytest.raises(FileExistsError, match="already exists"):
        module.create_terminal(runner=_runner(module))


def test_archival_never_rechecks_clock_manager_or_absences(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = terminalizer
    target = tmp_path / "terminal.json"
    monkeypatch.setattr(module, "TERMINAL_PATH", target)
    expected = module.create_terminal(runner=_runner(module))

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("archival performed a forbidden live recheck")

    monkeypatch.setattr(module, "_utc_now", forbidden)
    monkeypatch.setattr(module, "_read_unit_state", forbidden)
    monkeypatch.setattr(module, "_observe_absences", forbidden)
    observed, _root = module.validate_archival()
    assert observed == expected


def test_semantic_tamper_cannot_be_hidden_by_refingerprinting(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = terminalizer
    target = tmp_path / "terminal.json"
    monkeypatch.setattr(module, "TERMINAL_PATH", target)
    module.create_terminal(runner=_runner(module))

    def mutate(value):
        value["original_execution_observation"][
            "original_stderr_claimed"
        ] = True

    _rewrite_terminal(module, target, mutate)
    with pytest.raises(PermissionError, match="semantics changed"):
        module.validate_archival()


def test_preopen_source_drift_fails_without_creating_terminal(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = terminalizer
    target = tmp_path / "terminal.json"
    monkeypatch.setattr(module, "TERMINAL_PATH", target)
    original = module._validate_fixed_inputs
    calls = 0

    def drifting_inputs():
        nonlocal calls
        calls += 1
        values, roots, raws, generations = original()
        if calls == 2:
            generations = copy.deepcopy(generations)
            generations["old_B4_compatibility_bridge"]["inode"] += 1
        return values, roots, raws, generations

    monkeypatch.setattr(module, "_validate_fixed_inputs", drifting_inputs)
    with pytest.raises(PermissionError, match="changed frozen C4 inputs"):
        module.create_terminal(runner=_runner(module))
    assert not os.path.lexists(target)


def test_while_open_guard_failure_leaves_fail_closed_partial(
    terminalizer,
    tmp_path: Path,
) -> None:
    module = terminalizer
    target = tmp_path / "partial.json"

    def drift() -> None:
        raise PermissionError("injected while-open drift")

    with pytest.raises(PermissionError, match="injected while-open drift"):
        module._write_create_once(
            target,
            {"schema_version": "test"},
            while_open_guard=drift,
        )
    assert target.exists()
    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    assert target.stat().st_nlink == 1


def test_existing_forbidden_output_blocks_before_terminal_write(
    terminalizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = terminalizer
    target = tmp_path / "terminal.json"
    forbidden = tmp_path / "runtime-spec.json"
    forbidden.write_text("forbidden", encoding="utf-8")
    monkeypatch.setattr(module, "TERMINAL_PATH", target)
    monkeypatch.setattr(
        module,
        "ABSENT_OUTPUT_PATHS",
        {"L4_C4_runtime_spec": forbidden},
    )
    with pytest.raises(PermissionError, match="forbidden C4 output exists"):
        module.create_terminal(runner=_runner(module))
    assert not os.path.lexists(target)


def test_expiry_and_fixed_path_fail_closed(
    terminalizer,
    tmp_path: Path,
) -> None:
    module = terminalizer
    values, _roots, _raws, _generations = module._validate_fixed_inputs()
    too_early = datetime(2026, 7, 31, 15, 21, 30, tzinfo=timezone.utc)
    with pytest.raises(PermissionError, match="not both expired"):
        module._authorization_expiry(values, observed_at=too_early)
    with pytest.raises(PermissionError, match="path is not fixed"):
        module.create_terminal(tmp_path / "not-fixed.json", runner=_runner(module))
    with pytest.raises(PermissionError, match="path is not fixed"):
        module.validate_archival(tmp_path / "not-fixed.json")


def test_cli_surface_and_ast_have_no_retry_or_payload_entrypoints(
    terminalizer,
) -> None:
    module = terminalizer
    assert module._parser().parse_args(["create-terminal"]).command == (
        "create-terminal"
    )
    assert module._parser().parse_args(["validate-terminal"]).command == (
        "validate-terminal"
    )
    with pytest.raises(SystemExit):
        module._parser().parse_args(["create-terminal", "--force"])
    with pytest.raises(SystemExit):
        module._parser().parse_args(["retry"])

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert imported.isdisjoint({"torch", "numpy", "PIL", "pynvml"})
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called_names.isdisjoint(
        {"seal_receipt", "_load_sealed", "stability_gate", "sleep"}
    )
    assert called_attributes.isdisjoint(
        {"seal_receipt", "_load_sealed", "stability_gate", "sleep", "cuda"}
    )
