from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c2.py"
)


def _load():
    name = "cure_lite_v24_preaccess_schema_compatibility_c2_tested"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bridge():
    return _load()


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _state(module, unit: str, fragment: str = "") -> dict[str, object]:
    state = {field: "" for field in module._STATE_FIELDS}
    state.update(
        {
            "Id": unit,
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "static",
            "NRestarts": "0",
            "FragmentPath": fragment,
            "InvocationID": "",
            "Restart": "no",
        }
    )
    return state


def test_bridge_has_no_fictional_handoff_or_hash_cycle_sentinels() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    forbidden = (
        "C2_SCOPE_HANDOFF_AUTHORIZATION_PATH",
        "C2_SCOPE_HANDOFF_RECEIPT_PATH",
        "C2_SCOPE_HANDOFF_AUTHORIZATION_SCHEMA",
        "C2_SCOPE_HANDOFF_RECEIPT_SCHEMA",
        "_validate_scope_handoff_contract",
        "C2_ENVIRONMENT_WRAPPER_SHA256",
        "__TO_BE_FROZEN__",
    )
    assert all(token not in text for token in forbidden)
    compile(text, str(SOURCE), "exec")


def test_terminalizer_path_hash_and_interface_are_exact(bridge) -> None:
    assert bridge.C1_FAILURE_TERMINALIZER_SHA256 == (
        "72d7f8846d9bdccbdb2d15d6790d5e021b6d2db75523fe5dbe11a3d4246ca880"
    )
    module, root = bridge._load_verified_terminalizer()
    assert Path(module.TERMINAL_PATH) == bridge.C1_FAILURE_TERMINAL_PATH
    assert module.SCHEMA == (
        "cure-lite-v24-r2-preaccess-schema-compat-c1-"
        "expired-prewrite-terminal-v1"
    )
    assert root["file_sha256"] == bridge.C1_FAILURE_TERMINALIZER_SHA256
    assert callable(module.validate_terminal)


def test_prewrite_terminalizer_and_failure_file_are_double_fixed(
    bridge,
) -> None:
    assert bridge.C2_PREWRITE_FAILURE_TERMINALIZER_SHA256 == (
        "17ef3a0420c4b3d978f23270bde490805997e21dcb21f395ce7e5ac06659dc5f"
    )
    assert bridge.C2_PREWRITE_FAILURE_TERMINAL_SHA256 == (
        "6984dc9df2c905a5b7bc3b1577a4d5e8c21d1e1f895217997ed6915050e0f43d"
    )
    terminalizer, source_root = (
        bridge._load_verified_prewrite_failure_terminalizer()
    )
    assert Path(terminalizer.TERMINAL_PATH) == (
        bridge.C2_PREWRITE_FAILURE_TERMINAL_PATH
    )
    assert terminalizer.SCHEMA == (
        bridge.C2_PREWRITE_FAILURE_TERMINAL_SCHEMA
    )
    assert source_root["file_sha256"] == (
        bridge.C2_PREWRITE_FAILURE_TERMINALIZER_SHA256
    )

    payload, failure_root, observed_source_root = (
        bridge._validate_prewrite_failure_terminal()
    )
    assert payload["schema_version"] == (
        bridge.C2_PREWRITE_FAILURE_TERMINAL_SCHEMA
    )
    assert failure_root["path"] == str(
        bridge.C2_PREWRITE_FAILURE_TERMINAL_PATH.absolute()
    )
    assert failure_root["file_sha256"] == (
        bridge.C2_PREWRITE_FAILURE_TERMINAL_SHA256
    )
    assert observed_source_root == source_root


def test_prewrite_failure_lineage_labels_are_explicit(bridge) -> None:
    assert "c2_prewrite_failure_terminalizer" in bridge._SOURCE_LABELS
    assert "c2_prewrite_failure_terminal" in bridge._EVIDENCE_LABELS
    assert "c2_prewrite_failure_terminal_root" in (
        bridge._AUTHORIZATION_KEYS
    )
    assert bridge._source_paths()[
        "c2_prewrite_failure_terminalizer"
    ] == bridge.C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH
    assert bridge._evidence_paths()[
        "c2_prewrite_failure_terminal"
    ] == bridge.C2_PREWRITE_FAILURE_TERMINAL_PATH


@pytest.mark.parametrize("drift_target", ("terminalizer", "failure"))
def test_prewrite_archival_detects_postvalidation_generation_drift(
    bridge,
    monkeypatch: pytest.MonkeyPatch,
    drift_target: str,
) -> None:
    payload, failure_root, source_root = (
        bridge._validate_prewrite_failure_terminal()
    )
    terminalizer_path = (
        bridge.C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH
    )
    failure_path = bridge.C2_PREWRITE_FAILURE_TERMINAL_PATH
    terminalizer_raw = terminalizer_path.read_bytes()
    failure_raw = failure_path.read_bytes()

    def observed(path: Path, *, inode_delta: int = 0):
        value = path.stat()
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino + inode_delta,
            st_mode=value.st_mode,
            st_uid=value.st_uid,
            st_gid=value.st_gid,
            st_nlink=value.st_nlink,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
            st_ctime_ns=value.st_ctime_ns,
        )

    calls = {"terminalizer": 0, "failure": 0}

    def read(path: Path, *, sealed: bool = True):
        selected = Path(path).absolute()
        if selected == terminalizer_path.absolute():
            calls["terminalizer"] += 1
            delta = int(
                drift_target == "terminalizer"
                and calls["terminalizer"] == 2
            )
            return terminalizer_raw, observed(
                terminalizer_path,
                inode_delta=delta,
            )
        assert selected == failure_path.absolute()
        assert sealed is True
        calls["failure"] += 1
        delta = int(
            drift_target == "failure" and calls["failure"] == 2
        )
        return failure_raw, observed(failure_path, inode_delta=delta)

    module = SimpleNamespace(
        validate_archival=lambda _path: (payload, failure_root),
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_prewrite_failure_terminalizer",
        lambda: (module, source_root),
    )
    monkeypatch.setattr(bridge, "_read_regular_bytes", read)
    monkeypatch.setattr(bridge, "_source_root", lambda _path: source_root)

    with pytest.raises(PermissionError, match="changed"):
        bridge._validate_prewrite_failure_terminal()
    assert calls == {"terminalizer": 2, "failure": 2}


def test_real_environment_wrapper_loads_from_captured_source_root(bridge) -> None:
    root = bridge._source_root(bridge.C2_ENVIRONMENT_WRAPPER_SOURCE_PATH)
    module, observed = bridge._load_verified_environment_wrapper(root)
    assert observed == root
    assert module.C2_TARGET_UNIT == bridge.C2_UNIT_NAME
    assert Path(module.C2_POLICY_PATH) == bridge.C2_ENVIRONMENT_POLICY_PATH
    assert callable(module.replay_old_scope_and_handoff)
    assert callable(module.validate_c2_environment_closure)


def test_environment_wrapper_generation_drift_fails_closed(bridge) -> None:
    root = bridge._source_root(bridge.C2_ENVIRONMENT_WRAPPER_SOURCE_PATH)
    root["file_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="generation changed"):
        bridge._load_verified_environment_wrapper(root)


def test_environment_wrapper_interface_change_fails_closed(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("C2_TARGET_UNIT = 'wrong.service'\n", encoding="utf-8")
    monkeypatch.setattr(bridge, "C2_ENVIRONMENT_WRAPPER_SOURCE_PATH", wrapper)
    root = bridge._source_root(wrapper)
    with pytest.raises((AttributeError, PermissionError)):
        bridge._load_verified_environment_wrapper(root)


@dataclass(frozen=True)
class _Contract:
    target_unit_id: str
    require_target_ready: bool
    selected_gpu_index: int = 0


class _Environment:
    def __init__(self, bridge, unit_authorization, unit_receipt):
        self.bridge = bridge
        self.unit_authorization = unit_authorization
        self.unit_receipt = unit_receipt
        self.validations: list[dict[str, object]] = []

    def _production_archival_validator(self, authorization_path, receipt_path):
        assert Path(authorization_path) == self.bridge.C2_UNIT_AUTHORIZATION_PATH
        assert Path(receipt_path) == self.bridge.C2_UNIT_RECEIPT_PATH
        return {
            "authorization": self.unit_authorization,
            "receipt": self.unit_receipt,
        }

    def replay_old_scope_and_handoff(self):
        return (
            _Contract(self.bridge.OLD_UNIT_NAME, False),
            _Contract(self.bridge.C2_UNIT_NAME, True),
            {"precleanup_inventory_receipt": {}, "cleanup_receipt": {}},
        )

    def validate_c2_environment_closure(
        self, policy, stability, postcleanup, *, archival, c2_contract
    ):
        self.validations.append(
            {
                "archival": archival,
                "contract": c2_contract,
            }
        )
        return {
            "policy": policy,
            "stability": stability,
            "postcleanup": postcleanup,
            "realization": archival,
        }


def _closure_inputs(bridge, monkeypatch: pytest.MonkeyPatch):
    base = datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc)
    terminal_root = {"path": "terminal", "terminal_fingerprint": "a" * 64}
    c1_root = {"path": "c1-auth", "authorization_fingerprint": "b" * 64}
    r10 = {
        "authorization": {"path": "r10-auth"},
        "receipt": {"path": "r10-receipt"},
    }
    terminal = {
        "evidence_roots": {
            "bridge_authorization": c1_root,
            "r10_authorization": r10["authorization"],
            "r10_receipt": r10["receipt"],
        }
    }
    sources = {
        label: {"path": label, "file_sha256": str(index) * 64}
        for index, label in enumerate(sorted(bridge._SOURCE_LABELS), 1)
    }
    failure_root = {"path": "c2-prewrite-failure"}
    failure_source_root = sources["c2_prewrite_failure_terminalizer"]
    wrapper_root = sources["compat_environment_wrapper"]
    unit_authorization = {"unit_name": bridge.C2_UNIT_NAME}
    unit_receipt = {
        "unit_name": bridge.C2_UNIT_NAME,
        "created_at_utc": _utc(base + timedelta(seconds=10)),
    }
    environment = _Environment(bridge, unit_authorization, unit_receipt)
    policy = {"created_at_utc": _utc(base + timedelta(seconds=20))}
    stability = {"passed": True}
    postcleanup = {"created_at_utc": _utc(base + timedelta(seconds=35))}
    env_roots = {
        "environment_policy": {"path": "policy"},
        "environment_stability": {"path": "stability"},
        "environment_postcleanup": {"path": "postcleanup"},
    }
    unit_auth_root = {"path": "unit-auth"}
    unit_receipt_root = {"path": "unit-receipt"}
    monkeypatch.setattr(
        bridge,
        "_validate_prewrite_failure_terminal",
        lambda: ({}, failure_root, failure_source_root),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_c1_failure_terminal",
        lambda **_kwargs: (terminal, terminal_root, {"path": "terminalizer"}),
    )
    monkeypatch.setattr(bridge, "_validate_source_roots", lambda _roots: None)
    monkeypatch.setattr(
        bridge,
        "_load_verified_environment_wrapper",
        lambda root: (environment, dict(root)),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_unit_chain",
        lambda **_kwargs: (
            unit_authorization,
            unit_auth_root,
            unit_receipt,
            unit_receipt_root,
            base + timedelta(seconds=10),
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_load_environment_evidence",
        lambda: (policy, stability, postcleanup, env_roots),
    )
    authorization = {
        "compatibility_source_roots": sources,
        "c1_failure_terminal_root": terminal_root,
        "c2_prewrite_failure_terminal_root": failure_root,
        "c1_expired_authorization_root": c1_root,
        "r10_roots": r10,
    }
    return SimpleNamespace(
        base=base,
        terminal=terminal,
        sources=sources,
        failure_root=failure_root,
        failure_source_root=failure_source_root,
        environment=environment,
        policy=policy,
        stability=stability,
        postcleanup=postcleanup,
        authorization=authorization,
    )


def test_full_closure_delegates_to_real_wrapper_contract(
    bridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _closure_inputs(bridge, monkeypatch)
    closure = bridge._collect_full_closure(
        authorization=values.authorization,
        authorization_root={"path": "c2-auth"},
        unit_state_reader=lambda _unit: {},
        allow_runtime_activation=False,
        receipt_time=values.base + timedelta(seconds=40),
    )
    assert set(closure["evidence_roots"]) == bridge._EVIDENCE_LABELS
    assert all("handoff" not in label for label in closure["evidence_roots"])
    assert closure["historical_contract"]["target_unit_id"] == (
        bridge.OLD_UNIT_NAME
    )
    assert closure["current_contract"] == {
        "require_target_ready": True,
        "selected_gpu_index": 0,
        "target_unit_id": bridge.C2_UNIT_NAME,
    }
    assert len(values.environment.validations) == 1


def test_realization_archival_divergence_fails_closed(
    bridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _closure_inputs(bridge, monkeypatch)
    values.environment.unit_receipt = {"unit_name": "wrong.service"}
    with pytest.raises(PermissionError, match="archival validator diverged"):
        bridge._collect_full_closure(
            authorization=values.authorization,
            authorization_root={},
            unit_state_reader=lambda _unit: {},
            allow_runtime_activation=False,
            receipt_time=values.base + timedelta(seconds=40),
        )


def test_environment_must_follow_realization(
    bridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _closure_inputs(bridge, monkeypatch)
    values.policy["created_at_utc"] = _utc(values.base + timedelta(seconds=5))
    with pytest.raises(PermissionError, match="chronology/lineage"):
        bridge._collect_full_closure(
            authorization=values.authorization,
            authorization_root={},
            unit_state_reader=lambda _unit: {},
            allow_runtime_activation=False,
            receipt_time=values.base + timedelta(seconds=40),
        )


def test_authorize_captures_and_loads_wrapper_root(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc)
    authorization_path = tmp_path / "authorization.json"
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setattr(bridge, "C2_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(bridge, "COMPAT_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(bridge, "C2_RECEIPT_PATH", receipt_path)
    terminal_root = {"path": "terminal"}
    c1_root = {"path": "c1-auth"}
    terminal = {
        "evidence_roots": {
            "bridge_authorization": c1_root,
            "r10_authorization": {"path": "r10-auth"},
            "r10_receipt": {"path": "r10-receipt"},
        },
        "authorization_expiry": {
            "bridge_expires_at_utc": _utc(base - timedelta(seconds=1))
        },
    }
    sources = {
        label: {"path": label, "file_sha256": str(index) * 64}
        for index, label in enumerate(sorted(bridge._SOURCE_LABELS), 1)
    }
    failure_root = {"path": "c2-prewrite-failure"}
    failure_source_root = sources["c2_prewrite_failure_terminalizer"]
    monkeypatch.setattr(
        bridge,
        "_validate_prewrite_failure_terminal",
        lambda: ({}, failure_root, failure_source_root),
    )
    loaded: list[object] = []
    monkeypatch.setattr(bridge, "_require_absent", lambda _paths: None)
    monkeypatch.setattr(
        bridge,
        "_validate_c1_failure_terminal",
        lambda **_kwargs: (terminal, terminal_root, {"path": "terminalizer"}),
    )
    monkeypatch.setattr(bridge, "_collect_source_roots", lambda: sources)
    monkeypatch.setattr(
        bridge,
        "_load_verified_environment_wrapper",
        lambda root: (loaded.append(root) or object(), dict(root)),
    )
    written: dict[str, object] = {}

    def write(_path, body, *, fingerprint_field):
        written.update(body)
        return {**body, fingerprint_field: "f" * 64}

    monkeypatch.setattr(bridge, "_write_sealed", write)
    reader = lambda unit: _state(bridge, unit)
    result = bridge.authorize_c2(
        instruction_id=bridge.INSTRUCTION_ID,
        authorization_basis=bridge.AUTHORIZATION_BASIS,
        unit_state_reader=reader,
        now=lambda: base,
    )
    assert loaded == [sources["compat_environment_wrapper"]]
    assert written["compatibility_source_roots"] == sources
    assert written["c2_prewrite_failure_terminal_root"] == failure_root
    assert sources["c2_prewrite_failure_terminalizer"] == (
        failure_source_root
    )
    assert result["scientific_attempt_ordinal"] == 2
    assert result["scientific_authority"]["automatic_retry"] is False


def test_authorization_window_remains_bounded(bridge) -> None:
    with pytest.raises(ValueError, match="input changed"):
        bridge.authorize_c2(
            instruction_id=bridge.INSTRUCTION_ID,
            authorization_basis=bridge.AUTHORIZATION_BASIS,
            validity_seconds=301,
        )


def test_future_absence_is_phase_aware(
    bridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bridge,
        "validate_c2_authorization",
        lambda **_kwargs: ({"ok": True}, {"path": "auth"}),
    )
    calls: list[object] = []
    monkeypatch.setattr(bridge, "_require_absent", calls.append)
    bridge.validate_compat_authorization(require_future_absence=False)
    assert calls == [
        bridge._always_absent_paths(),
        bridge._preactivation_scientific_paths(),
    ]
    calls.clear()
    bridge.validate_compat_authorization(require_future_absence=True)
    assert calls == [
        bridge._always_absent_paths(),
        bridge._preactivation_scientific_paths(),
        bridge._c2_future_paths(),
    ]


def test_original_scientific_outputs_are_not_permanent_absences(bridge) -> None:
    permanent = bridge._always_absent_paths()
    preactivation = bridge._preactivation_scientific_paths()
    future = bridge._c2_future_paths()
    assert set(preactivation) == {
        "scientific_run_root",
        "scientific_result_receipt",
    }
    assert not set(preactivation) & set(permanent)
    assert {
        "c1_runtime_spec",
        "c1_runtime_launch_authorization",
        "c1_runtime_artifacts",
        "c1_gpu_lease",
        "old_runtime_spec",
        "old_runtime_launch_authorization",
        "old_runtime_artifacts",
        "old_gpu_lease",
        "c1_run_alias",
        "c1_result_alias",
        "c2_run_alias",
        "c2_result_alias",
        "c2_unit_terminal",
    }.issubset(permanent)
    assert set(future) == {
        "c2_runtime_spec",
        "c2_runtime_launch_authorization",
        "c2_runtime_artifacts",
        "c2_gpu_lease",
    }


def _redirect_scientific_outputs(
    bridge,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    run_root = tmp_path / "scientific-r2-run"
    result = tmp_path / "scientific-r2-result.json"
    monkeypatch.setattr(bridge, "SCIENTIFIC_RUN_ROOT", run_root)
    monkeypatch.setattr(bridge, "SCIENTIFIC_RESULT_RECEIPT_PATH", result)
    monkeypatch.setattr(bridge, "_always_absent_paths", lambda: {})
    return run_root, result


def _seal_payload(
    bridge,
    path: Path,
    body: dict[str, object],
    *,
    fingerprint_field: str,
    mode: int = 0o444,
) -> dict[str, object]:
    payload = {
        **body,
        fingerprint_field: bridge.stable_fingerprint(body),
    }
    if path.exists():
        path.chmod(0o600)
    path.write_text(bridge._canonical_json(payload) + "\n", encoding="utf-8")
    path.chmod(mode)
    return payload


def _write_arbitrary_fake_result(
    bridge,
    path: Path,
    *,
    mode: int = 0o444,
) -> None:
    body = {
        "schema_version": "test-scientific-r2-result-v1",
        "D_R_payload_accessed": True,
    }
    _seal_payload(
        bridge,
        path,
        body,
        fingerprint_field="receipt_fingerprint",
        mode=mode,
    )


def _write_structural_real_r2_result(
    bridge,
    run_root: Path,
    result: Path,
) -> dict[str, object]:
    implementation = {"frozen/scientific.py": "1" * 64}
    source_closure = bridge.stable_fingerprint(implementation)
    authorization_fingerprint = "2" * 64
    authorization_file_sha256 = "3" * 64
    access_fingerprint = "4" * 64
    access_file_sha256 = "5" * 64
    dataset_fingerprint = "6" * 64
    dataset_file_sha256 = "7" * 64
    protocol_fingerprint = "8" * 64
    source_binding_fingerprint = "f" * 64
    real_inputs_fingerprint = "0" * 64
    population_fingerprint = "1" * 64
    cache_fingerprint = "2" * 64
    marker_path = run_root / (
        bridge.R2_RUN_START_FILENAME_PREFIX
        + authorization_fingerprint
        + ".json"
    )
    intent = {
        "execution_kind": bridge.R2_EXECUTION_KIND,
        "split": "D_R",
        "requested_device": "cuda:0",
        "requested_receipt_output": str(result.absolute()),
        "D_R_materialization_intended": True,
        "D_V_materialization_intended": False,
        "D_T_materialization_intended": False,
        "optimizer_steps_authorized": 0,
        "parameter_updates_authorized": 0,
        "training_authorized": False,
    }
    marker_body = {
        "schema_version": bridge.R2_RUN_START_SCHEMA,
        "path_policy": bridge.R2_RUN_START_PATH_POLICY,
        "stage_id": bridge.R2_RUN_START_STAGE_ID,
        "run_id": bridge.SCIENTIFIC_ATTEMPT_ID,
        "candidate": bridge.CANDIDATE,
        "marker_path": str(marker_path),
        "authorization_fingerprint": authorization_fingerprint,
        "authorization_receipt_file_sha256": authorization_file_sha256,
        "access_audit_receipt_fingerprint": access_fingerprint,
        "access_audit_receipt_file_sha256": access_file_sha256,
        "dataset_free_receipt_fingerprint": dataset_fingerprint,
        "dataset_free_receipt_file_sha256": dataset_file_sha256,
        "protocol_preregistration_fingerprint": protocol_fingerprint,
        "source_closure_fingerprint": source_closure,
        "implementation_binding": implementation,
        "expected_source_binding_fingerprint": source_binding_fingerprint,
        "expected_real_inputs_fingerprint": real_inputs_fingerprint,
        "expected_population_fingerprint": population_fingerprint,
        "expected_cache_fingerprint": cache_fingerprint,
        "intent": intent,
        "intent_fingerprint": bridge.stable_fingerprint(intent),
    }
    marker = _seal_payload(
        bridge,
        marker_path,
        marker_body,
        fingerprint_field="marker_fingerprint",
    )
    marker_file_sha256 = hashlib.sha256(marker_path.read_bytes()).hexdigest()
    raw = {"structural_fixture": "no-scientific-import"}
    boundary = {
        "execution_kind": bridge.R2_EXECUTION_KIND,
        "split": "D_R",
        "D_R_accessed": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "D_V_tensor_payload_accessed": False,
        "D_T_tensor_payload_accessed": False,
        "optimizer_module_referenced": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "performance_gate_present": False,
        "performance_claim_supported": False,
        "threshold_or_ratio_gate": None,
    }
    body = {
        "schema_version": bridge.R2_RESULT_SCHEMA,
        "run_id": bridge.SCIENTIFIC_ATTEMPT_ID,
        "candidate": bridge.CANDIDATE,
        "execution_kind": bridge.R2_EXECUTION_KIND,
        "execution_seed": bridge.R2_EXECUTION_SEED,
        "device": "cuda:0",
        "requested_receipt_output": str(result.absolute()),
        "dataset_free_receipt_fingerprint": dataset_fingerprint,
        "dataset_free_receipt_file_sha256": dataset_file_sha256,
        "efficiency_section_fingerprint": "d" * 64,
        "efficiency_receipt_sha256": "e" * 64,
        "preaccess_authorization_fingerprint": authorization_fingerprint,
        "preaccess_authorization_file_sha256": authorization_file_sha256,
        "access_audit_receipt_fingerprint": access_fingerprint,
        "access_audit_receipt_file_sha256": access_file_sha256,
        "protocol_preregistration_fingerprint": protocol_fingerprint,
        "implementation_binding": implementation,
        "source_closure_fingerprint": source_closure,
        "source_binding_fingerprint": source_binding_fingerprint,
        "real_inputs_fingerprint": real_inputs_fingerprint,
        "population_fingerprint": population_fingerprint,
        "cache_fingerprint": cache_fingerprint,
        "adapter_fingerprint": "3" * 64,
        "run_start_marker": {
            "path": str(marker_path),
            "file_sha256": marker_file_sha256,
            "marker_fingerprint": marker["marker_fingerprint"],
            "payload": marker,
        },
        "artifact_hashes": {
            "dataset_free_receipt": dataset_file_sha256,
            "preaccess_authorization": authorization_file_sha256,
            "preaccess_access_audit": access_file_sha256,
            "persistent_run_start_marker": marker_file_sha256,
        },
        "raw_observations": raw,
        "raw_observations_fingerprint": bridge.stable_fingerprint(raw),
        "checks": {"structural_fixture": {"passed": True}},
        "decision": {"gate_passed": True},
        "boundary": boundary,
    }
    return _seal_payload(
        bridge,
        result,
        body,
        fingerprint_field="receipt_fingerprint",
    )


def _rewrite_result(bridge, path: Path, payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("receipt_fingerprint")
    _seal_payload(
        bridge,
        path,
        body,
        fingerprint_field="receipt_fingerprint",
    )


def _validate_phase(bridge, phase: str) -> None:
    bridge._validate_scientific_output_phase(
        allow_runtime_activation=(
            phase != bridge.RUNTIME_PHASE_PREACTIVATION
        ),
        runtime_phase=phase,
    )


def test_runtime_phase_enum_is_exact(bridge) -> None:
    assert bridge.RUNTIME_PHASES == {
        "preactivation",
        "commit",
        "claim",
        "verify",
        "run_once",
        "finalize_success",
        "finalize_failure",
    }


def test_legacy_active_boolean_without_phase_fails_closed(bridge) -> None:
    with pytest.raises(PermissionError, match="explicit phase"):
        bridge._validate_scientific_output_phase(
            allow_runtime_activation=True,
        )


@pytest.mark.parametrize(
    "allow_runtime_activation,runtime_phase",
    (
        (False, "commit"),
        (True, "preactivation"),
        (True, "unknown"),
    ),
)
def test_runtime_phase_and_activation_flag_must_agree(
    bridge,
    allow_runtime_activation: bool,
    runtime_phase: str,
) -> None:
    with pytest.raises(PermissionError, match="phase"):
        bridge._validate_scientific_output_phase(
            allow_runtime_activation=allow_runtime_activation,
            runtime_phase=runtime_phase,
        )


@pytest.mark.parametrize("phase", ("commit", "claim", "verify", "run_once"))
def test_preexecution_phases_require_and_accept_empty_private_run_root(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    run_root, _result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o700)
    _validate_phase(bridge, phase)


@pytest.mark.parametrize(
    "phase",
    (
        "commit",
        "claim",
        "verify",
        "run_once",
        "finalize_success",
        "finalize_failure",
    ),
)
def test_every_active_phase_requires_original_run_root(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    _redirect_scientific_outputs(bridge, monkeypatch, tmp_path)
    with pytest.raises(PermissionError, match="run root is absent"):
        _validate_phase(bridge, phase)


def test_preactivation_phase_rejects_existing_original_run_root(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, _result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o700)
    with pytest.raises(PermissionError, match="protected compatibility"):
        bridge._validate_scientific_output_phase(
            allow_runtime_activation=False,
        )


def test_finalize_result_presence_matrix(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, _result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o700)
    _validate_phase(bridge, "finalize_failure")
    with pytest.raises(PermissionError, match="result receipt is absent"):
        _validate_phase(bridge, "finalize_success")


@pytest.mark.parametrize("phase", ("finalize_success", "finalize_failure"))
def test_finalize_accepts_exact_structural_real_r2_result(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    run_root, result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o700)
    _write_structural_real_r2_result(bridge, run_root, result)
    _validate_phase(bridge, phase)


@pytest.mark.parametrize("phase", ("finalize_success", "finalize_failure"))
def test_finalize_rejects_arbitrary_self_fingerprinted_fake_receipt(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    run_root, result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o700)
    _write_arbitrary_fake_result(bridge, result)
    with pytest.raises(PermissionError, match="schema"):
        _validate_phase(bridge, phase)


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "run_id",
        "candidate",
        "execution_kind",
        "seed",
        "requested_result",
        "D_R",
        "D_V",
        "D_T",
        "optimizer_steps",
        "parameter_updates",
        "training",
    ),
)
def test_finalize_rejects_resealed_r2_identity_or_boundary_mutation(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    run_root, result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o700)
    payload = _write_structural_real_r2_result(bridge, run_root, result)
    if mutation == "schema":
        payload["schema_version"] = "wrong"
    elif mutation == "run_id":
        payload["run_id"] = "wrong"
    elif mutation == "candidate":
        payload["candidate"] = "wrong"
    elif mutation == "execution_kind":
        payload["execution_kind"] = "generated"
    elif mutation == "seed":
        payload["execution_seed"] = 43
    elif mutation == "requested_result":
        payload["requested_receipt_output"] = str(tmp_path / "alias.json")
    else:
        boundary = payload["boundary"]
        assert isinstance(boundary, dict)
        if mutation == "D_R":
            boundary["D_R_accessed"] = False
        elif mutation == "D_V":
            boundary["D_V_accessed"] = True
        elif mutation == "D_T":
            boundary["D_T_accessed"] = True
        elif mutation == "optimizer_steps":
            boundary["optimizer_steps"] = 1
        elif mutation == "parameter_updates":
            boundary["parameter_updates"] = 1
        elif mutation == "training":
            boundary["training_performed"] = True
    _rewrite_result(bridge, result, payload)
    with pytest.raises(PermissionError):
        _validate_phase(bridge, "finalize_success")


@pytest.mark.parametrize("phase", ("commit", "claim", "verify", "run_once"))
def test_preexecution_phases_reject_nonempty_run_root(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    run_root, _result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o700)
    (run_root / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(PermissionError, match="not empty"):
        _validate_phase(bridge, phase)


@pytest.mark.parametrize("phase", ("commit", "claim", "verify", "run_once"))
def test_preexecution_phases_reject_any_result_receipt(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    run_root, result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o700)
    _write_arbitrary_fake_result(bridge, result)
    with pytest.raises(PermissionError, match="protected compatibility"):
        _validate_phase(bridge, phase)


def test_finalize_binds_exact_persistent_run_start_marker(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o700)
    payload = _write_structural_real_r2_result(bridge, run_root, result)
    run_start = payload["run_start_marker"]
    assert isinstance(run_start, dict)
    marker_path = Path(str(run_start["path"]))
    marker_path.chmod(0o644)
    with pytest.raises(PermissionError, match="run-start marker"):
        _validate_phase(bridge, "finalize_success")


def test_runtime_phase_rejects_wrong_run_root_mode(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, _result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="scientific r2 run root"):
        _validate_phase(bridge, "commit")


def test_runtime_phase_rejects_unsealed_result_mode(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    run_root.mkdir(mode=0o700)
    _write_structural_real_r2_result(bridge, run_root, result)
    result.chmod(0o644)
    with pytest.raises(PermissionError, match="scientific r2 result"):
        _validate_phase(bridge, "finalize_failure")


@pytest.mark.parametrize("target_kind", ("run", "result"))
def test_runtime_phase_rejects_scientific_output_symlinks(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    run_root, result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    if target_kind == "run":
        actual = tmp_path / "actual-run"
        actual.mkdir(mode=0o700)
        run_root.symlink_to(actual, target_is_directory=True)
        match = "scientific r2 run root"
    else:
        run_root.mkdir(mode=0o700)
        actual = tmp_path / "actual-result.json"
        _write_arbitrary_fake_result(bridge, actual)
        result.symlink_to(actual)
        match = "scientific r2 result"
    with pytest.raises(PermissionError, match=match):
        _validate_phase(
            bridge,
            "commit" if target_kind == "run" else "finalize_failure",
        )


def test_runtime_phase_still_rejects_compatibility_alias(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_root, _result = _redirect_scientific_outputs(
        bridge, monkeypatch, tmp_path
    )
    alias = tmp_path / "compat-c2-run-alias"
    alias.mkdir(mode=0o700)
    monkeypatch.setattr(
        bridge,
        "_always_absent_paths",
        lambda: {"c2_run_alias": alias},
    )
    with pytest.raises(PermissionError, match="protected compatibility"):
        _validate_phase(bridge, "commit")


def test_c1_compatible_downstream_api_is_preserved(bridge) -> None:
    assert bridge.COMPAT_AUTHORIZATION_PATH == bridge.C2_AUTHORIZATION_PATH
    assert bridge.COMPAT_RECEIPT_PATH == bridge.C2_RECEIPT_PATH
    assert bridge.COMPATIBILITY_RECEIPT_PATH == bridge.C2_RECEIPT_PATH
    assert bridge.COMPAT_UNIT_REALIZER_SOURCE_PATH == (
        bridge.C2_UNIT_REALIZER_SOURCE_PATH
    )
    assert bridge.COMPAT_UNIT_NAME == bridge.C2_UNIT_NAME
    receipt_parameters = inspect.signature(
        bridge.verify_compatibility_receipt
    ).parameters
    assert {
        "path",
        "expected_spec",
        "require_spec_binding",
        "allow_runtime_activation",
        "runtime_phase",
        "unit_state_reader",
        "now",
    }.issubset(receipt_parameters)
    prewrite_parameters = inspect.signature(
        bridge.verify_compatibility_prewrite_spec
    ).parameters
    assert {"expected_spec", "unit_state_reader", "now"}.issubset(
        prewrite_parameters
    )
    authorization_parameters = inspect.signature(
        bridge.validate_c2_authorization
    ).parameters
    assert {
        "allow_runtime_activation",
        "runtime_phase",
    }.issubset(authorization_parameters)


def test_receipt_evidence_roots_are_exact_not_schema_guessed(bridge) -> None:
    expected = {label: {"path": label} for label in bridge._EVIDENCE_LABELS}
    bridge._validate_receipt_evidence_roots(expected, expected)
    changed = dict(expected)
    changed["environment_policy"] = {"path": "other"}
    with pytest.raises(PermissionError, match="evidence-root"):
        bridge._validate_receipt_evidence_roots(changed, expected)


def test_authority_never_grants_retry_resume_or_payload(bridge) -> None:
    scientific = bridge._expected_scientific_authority()
    mutation = bridge._expected_mutation_authority()
    assert scientific["automatic_retry"] is False
    assert scientific["resume"] is False
    assert scientific["materialization_authorized"] is False
    assert mutation["unit_start_authorized"] is False
    assert mutation["unit_enable_authorized"] is False
    assert mutation["payload_access_authorized"] is False


def _minimal_c1_terminal() -> dict[str, object]:
    return {
        "continuation_policy": {
            "same_c1_reauthorization_allowed": False,
            "same_c1_receipt_sealing_allowed": False,
            "automatic_retry_allowed": False,
            "resume_allowed": False,
            "new_compatibility_generation_required": True,
        },
        "outcome": {
            "scientific_attempt_consumed": False,
            "runtime_launch_consumed": False,
            "materialization_consumed": False,
        },
        "payload_observation": {
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_accessed": False,
            "training_started": False,
        },
        "evidence_roots": {
            "r10_authorization": {"path": "r10-auth"},
            "r10_receipt": {"path": "r10-receipt"},
        },
        "authorization_expiry": {
            "bridge_expires_at_utc": "2026-07-30T00:00:00.000000Z",
        },
    }


def _fake_c1_terminalizer(
    tmp_path: Path,
    terminal: dict[str, object],
):
    module = SimpleNamespace(
        SCHEMA="fake-c1-terminal-v1",
        _LIVE_KEYS={"LoadState"},
        ABSENCE_PATHS={
            "compat_runtime_spec": tmp_path / "c1-runtime-spec.json",
            "scientific_run_root": tmp_path / "scientific-run",
            "scientific_result_receipt": (
                tmp_path / "scientific-result.json"
            ),
        },
        error=PermissionError("unset"),
    )

    def validate_terminal(**_kwargs):
        raise module.error

    module.validate_terminal = validate_terminal
    module._load_sealed = lambda *_args, **_kwargs: (
        terminal,
        {"path": "c1-terminal"},
    )
    return module


@pytest.mark.parametrize(
    "error_kind",
    ("live_closure", "scientific_run", "scientific_result"),
)
def test_c1_historical_fallback_accepts_exact_whitelisted_errors(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_kind: str,
) -> None:
    terminal = _minimal_c1_terminal()
    module = _fake_c1_terminalizer(tmp_path, terminal)
    messages = {
        "live_closure": "expired-prewrite terminal live closure changed",
        "scientific_run": "required absent path exists: "
        + str(Path(module.ABSENCE_PATHS["scientific_run_root"]).absolute()),
        "scientific_result": "required absent path exists: "
        + str(
            Path(
                module.ABSENCE_PATHS["scientific_result_receipt"]
            ).absolute()
        ),
    }
    module.error = PermissionError(messages[error_kind])
    observed: list[tuple[object, object, object, datetime]] = []
    monkeypatch.setattr(
        bridge,
        "_load_verified_terminalizer",
        lambda: (module, {"path": "terminalizer"}),
    )

    def historical(
        supplied_module,
        supplied_terminal,
        *,
        c1_reader,
        now,
    ) -> None:
        observed.append(
            (supplied_module, supplied_terminal, c1_reader(), now)
        )

    monkeypatch.setattr(
        bridge,
        "_validate_c1_historical_terminal",
        historical,
    )
    current = datetime(2026, 7, 30, 13, 0, tzinfo=timezone.utc)
    returned, root, source_root = bridge._validate_c1_failure_terminal(
        unit_state_reader=lambda _unit: {"LoadState": "loaded"},
        now=current,
    )
    assert returned is terminal
    assert root == {"path": "c1-terminal"}
    assert source_root == {"path": "terminalizer"}
    assert observed == [
        (module, terminal, {"LoadState": "loaded"}, current)
    ]


def test_c1_historical_fallback_rejects_nonexact_errors(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DerivedPermissionError(PermissionError):
        pass

    terminal = _minimal_c1_terminal()
    module = _fake_c1_terminalizer(tmp_path, terminal)
    live_message = "expired-prewrite terminal live closure changed"
    errors = (
        PermissionError(
            "required absent path exists: "
            + str(module.ABSENCE_PATHS["compat_runtime_spec"].absolute())
        ),
        PermissionError(live_message, "extra-argument"),
        DerivedPermissionError(live_message),
        PermissionError(live_message + " "),
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_terminalizer",
        lambda: (module, {"path": "terminalizer"}),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_c1_historical_terminal",
        lambda *_args, **_kwargs: pytest.fail(
            "nonexact error reached historical fallback"
        ),
    )
    current = datetime(2026, 7, 30, 13, 0, tzinfo=timezone.utc)
    for expected_error in errors:
        module.error = expected_error
        with pytest.raises(PermissionError) as caught:
            bridge._validate_c1_failure_terminal(
                unit_state_reader=lambda _unit: {"LoadState": "loaded"},
                now=current,
            )
        assert caught.value is expected_error


def _historical_absence_row(path: Path) -> dict[str, object]:
    target = path.absolute()
    return {
        "path": str(target),
        "basename": target.name,
        "lexists": False,
        "parent_path": str(target.parent),
        "parent_device": 1,
        "parent_inode": 2,
        "parent_owner_uid": os.getuid(),
        "parent_owner_gid": os.getgid(),
        "parent_mode": 0o700,
        "parent_nlink": 13,
        "parent_size": 4096,
        "parent_mtime_ns": 10,
        "parent_ctime_ns": 10,
    }


def test_historical_absence_snapshot_accepts_sealed_parent_metadata(
    bridge,
    tmp_path: Path,
) -> None:
    paths = {
        "first": tmp_path / "first.json",
        "second": tmp_path / "second",
    }
    module = SimpleNamespace(ABSENCE_PATHS=paths)
    snapshot = {
        name: _historical_absence_row(path)
        for name, path in paths.items()
    }
    bridge._validate_c1_historical_absence_snapshot(module, snapshot)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_key",
        "wrong_path",
        "wrong_basename",
        "lexists",
        "bool_inode",
        "parent_divergence",
        "world_writable",
    ),
)
def test_historical_absence_snapshot_rejects_tampering(
    bridge,
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = {
        "first": tmp_path / "first.json",
        "second": tmp_path / "second",
    }
    module = SimpleNamespace(ABSENCE_PATHS=paths)
    snapshot = {
        name: _historical_absence_row(path)
        for name, path in paths.items()
    }
    if mutation == "missing_key":
        snapshot["first"].pop("parent_ctime_ns")
    elif mutation == "wrong_path":
        snapshot["first"]["path"] = str(tmp_path / "other")
    elif mutation == "wrong_basename":
        snapshot["first"]["basename"] = "other"
    elif mutation == "lexists":
        snapshot["first"]["lexists"] = True
    elif mutation == "bool_inode":
        snapshot["first"]["parent_inode"] = True
    elif mutation == "parent_divergence":
        snapshot["second"]["parent_nlink"] = 14
    elif mutation == "world_writable":
        snapshot["first"]["parent_mode"] = 0o702
        snapshot["second"]["parent_mode"] = 0o702
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(PermissionError):
        bridge._validate_c1_historical_absence_snapshot(module, snapshot)


def _historical_c1_fixture(bridge, tmp_path: Path):
    absence_paths = {
        "compat_runtime_spec": tmp_path / "c1-runtime-spec.json",
        "scientific_run_root": tmp_path / "scientific-run",
        "scientific_result_receipt": tmp_path / "scientific-result.json",
    }
    absences = {
        name: _historical_absence_row(path)
        for name, path in absence_paths.items()
    }
    payload_observation = {"sealed": True}
    continuation = {"new_generation_required": True}
    outcome = {"terminal": True}
    derived = {"attempt_commit": {"absent": True}}
    session = {"fixed_prefix": True}
    evidence_roots = {"unit_receipt": {"path": "receipt-root"}}
    source_roots = {"c1_source": {"path": "source-root"}}
    fragment = {
        "path": "fragment",
        "file_sha256": "a" * 64,
        "device": 1,
        "inode": 2,
        "owner_uid": os.getuid(),
        "mode": 0o600,
        "nlink": 1,
    }
    live = {"LoadState": "loaded"}
    expiry = {
        "bridge_expires_at_utc": "2026-07-30T12:00:00.000000Z"
    }
    body = {
        "schema_version": "c1-historical-test-v1",
        "candidate": bridge.CANDIDATE,
        "stage_id": bridge.STAGE_ID,
        "scientific_attempt_id": bridge.SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": bridge.SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": "c1",
        "unit_name": bridge.C1_UNIT_NAME,
        "created_at_utc": "2026-07-30T12:01:00.000000Z",
        "instruction_id": bridge.INSTRUCTION_ID,
        "authorization_basis": bridge.AUTHORIZATION_BASIS,
        "session_failure": session,
        "evidence_roots": evidence_roots,
        "source_roots": source_roots,
        "fragment_root": fragment,
        "live_unit_state": live,
        "absence_generation_roots": absences,
        "derived_runtime_absences": derived,
        "authorization_expiry": expiry,
        "payload_observation": payload_observation,
        "continuation_policy": continuation,
        "outcome": outcome,
    }
    terminal = {**body, "terminal_fingerprint": "f" * 64}
    payloads = {
        "unit_receipt": {"fragment_identity": dict(fragment)}
    }
    absence_calls: list[Path] = []

    def validate_live(state, *, unit_receipt):
        if dict(state) != live or unit_receipt is not payloads["unit_receipt"]:
            raise PermissionError("current c1 live state changed")
        return dict(state)

    def observe_absence(path: Path):
        selected = Path(path).absolute()
        absence_calls.append(selected)
        if os.path.lexists(selected):
            raise PermissionError(f"required absent path exists: {selected}")
        return {"path": str(selected), "lexists": False}

    module = SimpleNamespace(
        SCHEMA=body["schema_version"],
        CANDIDATE=bridge.CANDIDATE,
        STAGE_ID=bridge.STAGE_ID,
        SCIENTIFIC_ATTEMPT_ID=bridge.SCIENTIFIC_ATTEMPT_ID,
        SCIENTIFIC_ATTEMPT_ORDINAL=bridge.SCIENTIFIC_ATTEMPT_ORDINAL,
        RUNTIME_COMPATIBILITY_ID="c1",
        UNIT_NAME=bridge.C1_UNIT_NAME,
        INSTRUCTION_ID=bridge.INSTRUCTION_ID,
        AUTHORIZATION_BASIS=bridge.AUTHORIZATION_BASIS,
        _BODY_KEYS=set(body),
        _PAYLOAD_OBSERVATION=payload_observation,
        _CONTINUATION_POLICY=continuation,
        _OUTCOME=outcome,
        ABSENCE_PATHS=absence_paths,
        _derived_runtime_absences=lambda: derived,
        _parse_utc=bridge._parse_utc,
        _observe_session_failure=lambda: session,
        _observe_evidence=lambda: (evidence_roots, payloads),
        _observe_source_roots=lambda: source_roots,
        _observe_fragment_root=lambda: fragment,
        _validate_evidence_semantics=(
            lambda _payloads, *, now: expiry
        ),
        _validate_live_state=validate_live,
        _observe_absence=observe_absence,
    )
    return module, terminal, live, absence_paths, absence_calls


def test_historical_terminal_rechecks_only_c1_specific_current_absence(
    bridge,
    tmp_path: Path,
) -> None:
    module, terminal, live, paths, calls = _historical_c1_fixture(
        bridge,
        tmp_path,
    )
    paths["scientific_run_root"].mkdir()
    paths["scientific_result_receipt"].write_text("shared", encoding="utf-8")
    bridge._validate_c1_historical_terminal(
        module,
        terminal,
        c1_reader=lambda: live,
        now=datetime(2026, 7, 30, 13, 0, tzinfo=timezone.utc),
    )
    assert calls == [paths["compat_runtime_spec"].absolute()]


@pytest.mark.parametrize("mutation", ("live_state", "c1_path_present"))
def test_historical_terminal_rejects_current_c1_drift(
    bridge,
    tmp_path: Path,
    mutation: str,
) -> None:
    module, terminal, live, paths, _calls = _historical_c1_fixture(
        bridge,
        tmp_path,
    )
    reader = lambda: live
    if mutation == "live_state":
        reader = lambda: {"LoadState": "active"}
    else:
        paths["compat_runtime_spec"].write_text("present", encoding="utf-8")
    with pytest.raises(PermissionError):
        bridge._validate_c1_historical_terminal(
            module,
            terminal,
            c1_reader=reader,
            now=datetime(2026, 7, 30, 13, 0, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("mutation", ("failure_root", "terminalizer_root"))
def test_full_closure_rejects_prewrite_failure_lineage_drift(
    bridge,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    values = _closure_inputs(bridge, monkeypatch)
    authorization = deepcopy(values.authorization)
    if mutation == "failure_root":
        authorization["c2_prewrite_failure_terminal_root"] = {
            "path": "wrong-failure-root"
        }
    else:
        authorization["compatibility_source_roots"][
            "c2_prewrite_failure_terminalizer"
        ] = {"path": "wrong-terminalizer"}
    with pytest.raises(PermissionError, match="prewrite failure lineage"):
        bridge._collect_full_closure(
            authorization=authorization,
            authorization_root={"path": "c2-auth"},
            unit_state_reader=lambda _unit: {},
            allow_runtime_activation=False,
            receipt_time=values.base + timedelta(seconds=40),
        )


def test_receipt_and_source_root_sets_require_prewrite_lineage(bridge) -> None:
    evidence = {
        label: {"path": label}
        for label in bridge._EVIDENCE_LABELS
    }
    missing_failure = dict(evidence)
    missing_failure.pop("c2_prewrite_failure_terminal")
    with pytest.raises(PermissionError, match="evidence-root"):
        bridge._validate_receipt_evidence_roots(
            missing_failure,
            evidence,
        )

    sources = bridge._collect_source_roots()
    missing_terminalizer = dict(sources)
    missing_terminalizer.pop("c2_prewrite_failure_terminalizer")
    with pytest.raises(PermissionError, match="source-root labels"):
        bridge._validate_source_roots(missing_terminalizer)
