from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tests_v24 import (
    test_gcr_pacre_v24_dr_r2_runtime_supervisor as frozen_fixtures,
)
from tools import cure_lite_v24_runtime_supervisor as frozen_supervisor
from tools import (
    cure_lite_v24_preaccess_schema_compatibility_c3 as actual_bridge,
)
from tools import (
    cure_lite_v24_runtime_supervisor_preaccess_compat_c3 as supervisor,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SCIENTIFIC_AUTHORIZATION = Path(
    frozen_supervisor._ACTUAL_SCIENTIFIC_AUTHORIZATION_PATH
)
SCIENTIFIC_ACCESS_AUDIT = Path(
    frozen_supervisor._ACTUAL_SCIENTIFIC_ACCESS_AUDIT_PATH
)
LEGACY_GATE = (
    REPOSITORY / "tools/run_cure_lite_v24_gcr_pacre_dr_gate.py"
).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _enable_guarded_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        supervisor,
        "_require_source_generations",
        lambda: None,
    )


def _reseal_spec(payload: dict[str, object]) -> None:
    systemd = payload["runtime"]["systemd"]
    systemd["immutable_shadow_fingerprint"] = (
        supervisor.compat_c1.legacy.stable_fingerprint(
            systemd["immutable_shadow_properties"]
        )
    )
    body = dict(payload)
    body.pop("runtime_spec_fingerprint", None)
    payload["runtime_spec_fingerprint"] = (
        supervisor.compat_c1.legacy.stable_fingerprint(body)
    )


def _actual_c3_spec(tmp_path: Path) -> dict[str, object]:
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
            supervisor.compat_c1.legacy.RUNTIME_LAUNCH_AUTHORIZATION_SCHEMA
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
            supervisor.compat_c1.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
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
        supervisor.compat_c1.legacy.stable_fingerprint(child_argv)
    )
    payload["child"]["entrypoint_path"] = str(adapter_path)
    payload["source_bindings"].update(
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


def _policy_result() -> dict[str, object]:
    c1 = supervisor.compat_c1
    schema = {
        "producer_schema": c1.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA,
        "scientific_authorization_bound_schema": (
            c1.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        ),
        "compatibility_consumer_required_schema": (
            c1.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        ),
        "buggy_frozen_consumer_expected_schema": (
            c1.FICTIONAL_ACCESS_AUDIT_SCHEMA
        ),
        "accept_either_schema": False,
    }
    return {
        "schema_compatibility": schema,
        "compatibility_source_roots": {
            label: {
                "path": str(path),
                "file_sha256": c1.legacy.file_sha256(path),
            }
            for label, path in {
                "compat_policy": supervisor.COMPAT_POLICY_SOURCE_PATH,
                "compat_supervisor": Path(
                    supervisor.COMPAT_SUPERVISOR_PATH
                ),
                "compat_adapter": Path(supervisor.COMPAT_ADAPTER_PATH),
            }.items()
        },
    }


def test_metadata_only_identity_is_fixed_and_bridge_is_frozen() -> None:
    summary = supervisor.describe_compatibility_identity()

    assert summary["production_ready"] is True
    assert summary["bridge_source_frozen_at_load"] is True
    assert summary["compatibility_policy_source_sha256"] == (
        "3bf4caabfce8fd302b74b59b021bb37ba839f3c11226fde9b347ed2e574badb7"
    )
    assert summary["runtime_compatibility_generation"] == "c3"
    assert summary["scientific_attempt_ordinal"] == 2
    assert summary["unit_name"].endswith(
        "preaccess-compat-c3.service"
    )
    assert summary["runtime_spec_path"].endswith(
        "preaccess_compat_c3_runtime_spec.json"
    )
    assert summary["compatibility_receipt_path"].endswith(
        "schema_compat_c3_receipt.json"
    )
    assert summary["scientific_identity_changed"] is False
    assert summary["scientific_output_paths_changed"] is False
    assert summary["automatic_retry_allowed"] is False
    assert summary["resume_allowed"] is False
    assert summary["D_R_payload_accessed"] is False
    assert summary["D_V_payload_accessed"] is False
    assert summary["D_T_payload_accessed"] is False


def _force_placeholder_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        supervisor,
        "COMPAT_POLICY_SOURCE_SHA256",
        "__TO_BE_FROZEN__",
    )
    monkeypatch.setattr(supervisor, "_BRIDGE_LOAD", None)


@pytest.mark.parametrize(
    "call",
    [
        lambda: supervisor.verify_compatibility_identity(),
        lambda: supervisor.validate_prewrite_spec({}),
        lambda: supervisor._validate_spec_structure(
            {},
            loaded_spec_path=Path(supervisor.COMPAT_RUNTIME_SPEC_PATH),
        ),
        lambda: supervisor.verify_child_runtime_attestation("a", "b"),
        lambda: supervisor.main(["--help"]),
        lambda: supervisor.commit_and_start(
            supervisor.COMPAT_RUNTIME_SPEC_PATH
        ),
    ],
)
def test_placeholder_fails_closed_before_every_runtime_entrypoint(
    call,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_placeholder_guard(monkeypatch)
    with pytest.raises(
        PermissionError,
        match="bridge source generation is not frozen",
    ):
        call()


def test_direct_cli_help_succeeds_after_bridge_freeze() -> None:
    completed = subprocess.run(
        [
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            supervisor.COMPAT_SUPERVISOR_PATH,
            "--help",
        ],
        cwd="/",
        check=False,
        capture_output=True,
        text=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    for command in (
        "commit-and-start",
        "claim-materialization",
        "verify-runtime-spec",
        "run-once",
        "systemd-finalize",
        "record-systemd-exit",
    ):
        assert command in completed.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["commit-and-start", "--spec", "/tmp/c3-guard-absent.json"],
        ["claim-materialization", "--spec", "/tmp/c3-guard-absent.json"],
        ["verify-runtime-spec", "--spec", "/tmp/c3-guard-absent.json"],
        ["run-once", "--spec", "/tmp/c3-guard-absent.json"],
        ["systemd-finalize", "--spec", "/tmp/c3-guard-absent.json"],
        ["record-systemd-exit", "--spec", "/tmp/c3-guard-absent.json"],
    ],
)
def test_raw_legacy_main_commands_cannot_bypass_placeholder_guard(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_placeholder_guard(monkeypatch)
    with pytest.raises(
        PermissionError,
        match="bridge source generation is not frozen",
    ):
        supervisor.compat_c1.legacy.main(argv)


def test_raw_c1_main_cannot_bypass_placeholder_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_placeholder_guard(monkeypatch)
    with pytest.raises(
        PermissionError,
        match="bridge source generation is not frozen",
    ):
        supervisor.compat_c1.main(["--help"])


@pytest.mark.parametrize(
    "name",
    [
        "commit_and_start",
        "claim_materialization",
        "verify_runtime_spec",
        "run_once",
        "finalize_systemd",
    ],
)
def test_raw_inherited_runtime_callables_are_guarded(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_placeholder_guard(monkeypatch)
    inherited = getattr(supervisor.compat_c1.legacy, name)
    with pytest.raises(
        PermissionError,
        match="bridge source generation is not frozen",
    ):
        inherited("/tmp/c3-guard-absent.json")


def test_guard_installation_exposes_no_raw_or_dunder_wrapped_callable() -> None:
    assert supervisor._RAW_LEGACY_RUNTIME_ENTRYPOINTS == {}
    guarded = (
        supervisor._C1_MAIN,
        supervisor.compat_c1.main,
        supervisor.compat_c1.legacy.main,
        supervisor.compat_c1.legacy.commit_and_start,
        supervisor.compat_c1.legacy.claim_materialization,
        supervisor.compat_c1.legacy.verify_runtime_spec,
        supervisor.compat_c1.legacy.run_once,
        supervisor.compat_c1.legacy.finalize_systemd,
        supervisor.compat_c1.legacy.load_runtime_spec,
        supervisor.compat_c1.legacy.verify_child_runtime_attestation,
        supervisor.commit_and_start,
        supervisor.claim_materialization,
        supervisor.verify_runtime_spec,
        supervisor.run_once,
        supervisor.finalize_systemd,
        supervisor.load_runtime_spec,
    )
    assert all(not hasattr(function, "__wrapped__") for function in guarded)


@pytest.mark.parametrize(
    "call",
    (
        lambda: supervisor.claim_materialization("/tmp/c3-guard-absent.json"),
        lambda: supervisor.verify_runtime_spec("/tmp/c3-guard-absent.json"),
        lambda: supervisor.run_once("/tmp/c3-guard-absent.json"),
        lambda: supervisor.finalize_systemd("/tmp/c3-guard-absent.json"),
        lambda: supervisor.load_runtime_spec("/tmp/c3-guard-absent.json"),
        lambda: supervisor.compat_c1.main(["--help"]),
        lambda: supervisor.compat_c1.commit_and_start(
            "/tmp/c3-guard-absent.json"
        ),
        lambda: supervisor.compat_c1.legacy.main(["--help"]),
        lambda: supervisor.compat_c1.legacy.load_runtime_spec(
            "/tmp/c3-guard-absent.json"
        ),
        lambda: supervisor.compat_c1.legacy.verify_child_runtime_attestation(
            "/tmp/c3-attestation-absent.json",
            "/tmp/c3-launch-absent.json",
        ),
    ),
)
def test_every_explicit_or_inherited_entrypoint_is_placeholder_guarded(
    call,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_placeholder_guard(monkeypatch)
    with pytest.raises(
        PermissionError,
        match="bridge source generation is not frozen",
    ):
        call()


def test_c1_source_hash_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drift = (tmp_path / "compat-c1.py").resolve()
    drift.write_text("# drift\n", encoding="utf-8")
    monkeypatch.setattr(
        supervisor,
        "FROZEN_COMPAT_C1_SUPERVISOR_PATH",
        drift,
    )
    with pytest.raises(PermissionError, match="source hash changed"):
        supervisor._require_self_and_c1_generations()


def test_bridge_hash_or_generation_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = (tmp_path / "bridge.py").resolve()
    bridge.write_text("VALUE = 1\n", encoding="utf-8")
    raw, generation = supervisor._stable_source_bytes(bridge)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(supervisor, "COMPAT_POLICY_SOURCE_PATH", bridge)
    monkeypatch.setattr(
        supervisor,
        "COMPAT_POLICY_SOURCE_SHA256",
        digest,
    )
    monkeypatch.setattr(
        supervisor,
        "_BRIDGE_LOAD",
        (generation, digest),
    )
    supervisor._require_source_generations()

    bridge.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="source hash changed"):
        supervisor._require_source_generations()


def test_self_generation_replacement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dict(supervisor._SELF_LOAD_GENERATION)
    changed["st_size"] += 1
    monkeypatch.setattr(supervisor, "_SELF_LOAD_GENERATION", changed)
    with pytest.raises(PermissionError, match="generation changed"):
        supervisor._require_self_and_c1_generations()


def test_verified_identity_accepts_only_fixed_c3_runtime_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_guarded_calls(monkeypatch)
    result = supervisor.verify_compatibility_identity()

    assert result["runtime_compatibility_generation"] == "c3"
    assert result["unit_name"] == supervisor.COMPAT_UNIT_NAME
    assert result["runtime_spec_path"] == supervisor.COMPAT_RUNTIME_SPEC_PATH
    assert result["runtime_launch_authorization_path"] == (
        supervisor.COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
    )
    assert result["adapter_path"] == supervisor.COMPAT_ADAPTER_PATH
    assert result["compatibility_receipt_path"] == (
        supervisor.COMPATIBILITY_RECEIPT_PATH
    )
    assert result["production_ready"] is True
    serialized = json.dumps(result, sort_keys=True)
    assert "preaccess-compat-c1.service" not in serialized
    assert "preaccess-compat-c2.service" not in serialized
    assert "preaccess_compat_c1_runtime_spec.json" not in serialized
    assert "preaccess_compat_c2_runtime_spec.json" not in serialized


def test_identity_rejects_a_c2_receipt_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_guarded_calls(monkeypatch)
    original = supervisor._C1_VERIFY_COMPATIBILITY_IDENTITY

    def drifted() -> dict[str, object]:
        result = dict(original())
        result["compatibility_receipt_path"] = supervisor._C2_RECEIPT_PATH
        return result

    monkeypatch.setattr(
        supervisor,
        "_C1_VERIFY_COMPATIBILITY_IDENTITY",
        drifted,
    )
    with pytest.raises(PermissionError, match="identity changed"):
        supervisor.verify_compatibility_identity()


@pytest.mark.parametrize(
    "off_path",
    [
        supervisor._C1_RUNTIME_SPEC_PATH,
        supervisor._C2_RUNTIME_SPEC_PATH,
        supervisor._OLD_RUNTIME_SPEC_PATH,
    ],
)
def test_c1_c2_and_old_runtime_specs_are_rejected_before_receipt_use(
    off_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_guarded_calls(monkeypatch)
    bridge_called = False

    def forbidden() -> object:
        nonlocal bridge_called
        bridge_called = True
        raise AssertionError("bridge must not load for an off-path spec")

    monkeypatch.setattr(
        supervisor,
        "_C1_LOAD_VERIFIED_COMPATIBILITY_POLICY",
        forbidden,
    )
    payload = {
        "execution_kind": (
            supervisor.compat_c1.legacy.ACTUAL_EXECUTION_KIND
        )
    }
    with pytest.raises(ValueError, match="spec path is not exact"):
        supervisor._validate_spec_structure(
            payload,
            loaded_spec_path=Path(off_path),
        )
    assert bridge_called is False


def test_c1_receipt_interface_is_rejected_before_verifier_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_guarded_calls(monkeypatch)
    called = False

    def verifier(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        return {}

    policy = SimpleNamespace(
        COMPAT_RECEIPT_PATH=Path(supervisor._C1_RECEIPT_PATH),
        verify_compatibility_receipt=verifier,
    )
    monkeypatch.setattr(
        supervisor,
        "_C1_LOAD_VERIFIED_COMPATIBILITY_POLICY",
        lambda: (policy, {}),
    )
    with pytest.raises(PermissionError, match="path interface changed"):
        supervisor._verify_policy_compatibility_receipt({})
    assert called is False


@pytest.mark.parametrize(
    "runtime_phase, allow_runtime_activation",
    [
        ("preactivation", False),
        ("commit", True),
        ("claim", True),
        ("verify", True),
        ("run_once", True),
        ("finalize_success", True),
        ("finalize_failure", True),
    ],
)
def test_runtime_receipt_verification_uses_activation_phase(
    runtime_phase: str,
    allow_runtime_activation: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_guarded_calls(monkeypatch)
    calls: list[tuple[object, object, bool, bool, str]] = []

    def verifier(
        path: Path,
        *,
        expected_spec: dict[str, object],
        require_spec_binding: bool,
        allow_runtime_activation: bool,
        runtime_phase: str,
    ) -> dict[str, object]:
        calls.append(
            (
                path,
                expected_spec,
                require_spec_binding,
                allow_runtime_activation,
                runtime_phase,
            )
        )
        return _policy_result()

    policy = SimpleNamespace(
        COMPAT_RECEIPT_PATH=Path(
            supervisor.COMPATIBILITY_RECEIPT_PATH
        ),
        OLD_RUNTIME_SPEC_PATH=Path(supervisor._OLD_RUNTIME_SPEC_PATH),
        OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH=(
            Path(supervisor._OLD_RUNTIME_SPEC_PATH + ".launch")
        ),
        verify_compatibility_receipt=verifier,
    )
    _raw, generation = supervisor.compat_c1._stable_source_bytes(
        supervisor.COMPAT_POLICY_SOURCE_PATH
    )
    monkeypatch.setattr(
        supervisor,
        "_C1_LOAD_VERIFIED_COMPATIBILITY_POLICY",
        lambda: (policy, generation),
    )
    payload = {"runtime": "c3"}

    token = supervisor._ACTIVE_RUNTIME_PHASE.set(runtime_phase)
    try:
        result = supervisor._verify_policy_compatibility_receipt(payload)
    finally:
        supervisor._ACTIVE_RUNTIME_PHASE.reset(token)

    assert result["schema_compatibility"]["accept_either_schema"] is False
    assert calls == [
        (
            Path(supervisor.COMPATIBILITY_RECEIPT_PATH),
            payload,
            True,
            allow_runtime_activation,
            runtime_phase,
        )
    ]


@pytest.mark.parametrize(
    "runtime_phase",
    [
        "commit",
        "claim",
        "verify",
        "run_once",
        "finalize_success",
        "finalize_failure",
    ],
)
def test_phase_guarded_entrypoint_sets_and_resets_context(
    runtime_phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_guarded_calls(monkeypatch)
    observed: list[str] = []

    def probe(value: str) -> str:
        observed.append(supervisor._ACTIVE_RUNTIME_PHASE.get())
        return value

    wrapped = supervisor._phase_guarded_inherited_callable(
        probe,
        runtime_phase,
    )

    assert wrapped("ok") == "ok"
    assert observed == [runtime_phase]
    assert supervisor._ACTIVE_RUNTIME_PHASE.get() == "preactivation"


@pytest.mark.parametrize("runtime_phase", ("commit", "claim", "verify", "run_once"))
def test_phase_guard_drives_actual_bridge_preexecution_state_machine(
    runtime_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise S3 phase propagation against real B3 phase code, not a stub."""

    _enable_guarded_calls(monkeypatch)
    monkeypatch.setattr(
        actual_bridge,
        "SCIENTIFIC_RUN_ROOT",
        tmp_path / "scientific-r2-run",
    )
    monkeypatch.setattr(
        actual_bridge,
        "SCIENTIFIC_RESULT_RECEIPT_PATH",
        tmp_path / "scientific-r2-result.json",
    )
    monkeypatch.setattr(actual_bridge, "_always_absent_paths", lambda: {})
    observed: list[str] = []

    def verify_actual_bridge_phase() -> None:
        selected = supervisor._ACTIVE_RUNTIME_PHASE.get()
        actual_bridge._validate_scientific_output_phase(
            allow_runtime_activation=True,
            runtime_phase=selected,
        )
        observed.append(selected)

    wrapped = supervisor._phase_guarded_inherited_callable(
        verify_actual_bridge_phase,
        runtime_phase,
    )
    wrapped()

    assert observed == [runtime_phase]
    assert supervisor._ACTIVE_RUNTIME_PHASE.get() == "preactivation"
    assert not (tmp_path / "scientific-r2-run").exists()
    assert not (tmp_path / "scientific-r2-result.json").exists()


def test_phase_guard_resets_context_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_guarded_calls(monkeypatch)

    def fail() -> None:
        assert supervisor._ACTIVE_RUNTIME_PHASE.get() == "commit"
        raise RuntimeError("expected probe failure")

    wrapped = supervisor._phase_guarded_inherited_callable(
        fail,
        "commit",
    )

    with pytest.raises(RuntimeError, match="expected probe failure"):
        wrapped()
    assert supervisor._ACTIVE_RUNTIME_PHASE.get() == "preactivation"


@pytest.mark.parametrize(
    "environment, expected_phase",
    [
        (
            {
                "SERVICE_RESULT": "success",
                "EXIT_CODE": "exited",
                "EXIT_STATUS": "0",
            },
            "finalize_success",
        ),
        ({"SERVICE_RESULT": "exit-code"}, "finalize_failure"),
        ({}, "finalize_failure"),
    ],
)
def test_finalization_phase_comes_only_from_manager_outcome(
    environment: dict[str, str],
    expected_phase: str,
) -> None:
    assert (
        supervisor._finalization_runtime_phase(
            (),
            {"environment": environment},
        )
        == expected_phase
    )


def test_prewrite_receipt_verification_never_allows_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_guarded_calls(monkeypatch)
    calls: list[tuple[Path, dict[str, object]]] = []

    def verifier(
        path: Path,
        expected_spec: dict[str, object],
    ) -> dict[str, object]:
        calls.append((path, expected_spec))
        return _policy_result()

    policy = SimpleNamespace(
        verify_compatibility_prewrite_spec=verifier,
        OLD_RUNTIME_SPEC_PATH=Path(supervisor._OLD_RUNTIME_SPEC_PATH),
        OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH=(
            Path(supervisor._OLD_RUNTIME_SPEC_PATH + ".launch")
        ),
    )
    _raw, generation = supervisor.compat_c1._stable_source_bytes(
        supervisor.COMPAT_POLICY_SOURCE_PATH
    )
    monkeypatch.setattr(
        supervisor.compat_c1,
        "_load_verified_compatibility_policy",
        lambda: (policy, generation),
    )
    payload = {"runtime": "c3-preview"}

    result = supervisor._verify_policy_compatibility_prewrite(payload)

    assert result["schema_compatibility"]["accept_either_schema"] is False
    assert calls == [
        (Path(supervisor.COMPATIBILITY_RECEIPT_PATH), payload)
    ]


def test_extra_runtime_spec_field_is_rejected_before_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_guarded_calls(monkeypatch)
    payload = _actual_c3_spec(tmp_path)
    payload["unexpected_c3_field"] = "forbidden"
    _reseal_spec(payload)
    bridge_called = False

    def forbidden() -> object:
        nonlocal bridge_called
        bridge_called = True
        raise AssertionError("bridge must not load for an invalid spec")

    monkeypatch.setattr(
        supervisor,
        "_C1_LOAD_VERIFIED_COMPATIBILITY_POLICY",
        forbidden,
    )
    with pytest.raises(ValueError, match="closed schema"):
        supervisor._validate_spec_structure(
            payload,
            loaded_spec_path=Path(supervisor.COMPAT_RUNTIME_SPEC_PATH),
        )
    assert bridge_called is False
