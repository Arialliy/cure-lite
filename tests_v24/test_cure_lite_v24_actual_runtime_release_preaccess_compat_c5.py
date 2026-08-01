from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import (
    cure_lite_v24_actual_runtime_release_preaccess_compat_c5 as compat,
)


def _source_roots() -> dict[str, dict[str, object]]:
    return {
        label: {"path": str(path), "file_sha256": digest}
        for label, (path, digest) in {
            "compat_bridge": (
                compat.COMPAT_BRIDGE_PATH,
                compat.COMPAT_BRIDGE_SHA256,
            ),
            "compat_environment_wrapper": (
                compat.COMPAT_ENVIRONMENT_PATH,
                compat.COMPAT_ENVIRONMENT_SHA256,
            ),
            "compat_release": (
                Path(compat.__file__).resolve(),
                compat._sha256_file(Path(compat.__file__).resolve()),
            ),
            "compat_supervisor": (
                compat.COMPAT_SUPERVISOR_PATH,
                compat.COMPAT_SUPERVISOR_SHA256,
            ),
            "compat_adapter": (
                compat.COMPAT_ADAPTER_PATH,
                compat.COMPAT_ADAPTER_SHA256,
            ),
            "compat_unit_realizer": (
                compat.COMPAT_REALIZER_PATH,
                compat.COMPAT_REALIZER_SHA256,
            ),
            "compat_unit_template": (
                compat.COMPAT_UNIT_TEMPLATE_PATH,
                compat.COMPAT_UNIT_TEMPLATE_SHA256,
            ),
        }.items()
    }


def _unit_path_policy(inode: int) -> dict[str, object]:
    uid = os.getuid()
    runtime_directory = f"/run/user/{uid}/systemd/user"
    return {
        "runtime_directory": runtime_directory,
        "ordered_unit_paths": [
            {
                "path": str(
                    Path(runtime_directory).parent / "generator.late"
                ),
                "exists": True,
                "device": 17,
                "inode": inode,
                "owner_uid": uid,
                "mode": 0o700,
            },
            {
                "path": "/usr/lib/systemd/user",
                "exists": True,
                "device": 23,
                "inode": 29,
                "owner_uid": 0,
                "mode": 0o755,
            },
        ],
    }


def test_c5_namespace_and_scientific_identity_are_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compat.compat_realizer,
        "verify_compatibility_identity",
        lambda: {"unit_name": compat.COMPAT_UNIT},
        raising=False,
    )
    monkeypatch.setattr(
        compat.compat_supervisor,
        "verify_compatibility_identity",
        lambda: {
            "unit_name": compat.COMPAT_UNIT,
            "runtime_spec_path": str(compat.COMPAT_RUNTIME_SPEC_PATH),
        },
        raising=False,
    )
    identity = compat.verify_compatibility_identity()
    assert identity["scientific_attempt_ordinal"] == 2
    assert identity["runtime_compatibility_generation"] == "c5"
    assert identity["allowed_splits"] == ["D_R"]
    assert identity["D_R_structural_update_count"] == 0
    assert identity["optimizer_steps_authorized"] == 0
    assert identity["parameter_updates_authorized"] == 0
    assert identity["fresh_scientific_attempt"] is False
    assert identity["automatic_retry_allowed"] is False
    assert identity["resume_allowed"] is False
    assert identity["D_R_payload_accessed"] is False
    assert identity["D_V_payload_accessed"] is False
    assert identity["D_T_payload_accessed"] is False
    assert identity["c4_mutation_authorized"] is False
    assert identity["c4_reentry_authorized"] is False
    assert identity["c4_r14_must_remain_absent"] is True
    assert "compat_c5" in identity["runtime_spec_path"]
    assert "compat_c5_r14" in identity["r14_root"]
    assert "dummy-compat-c5-r14" in identity["r14_scenario_id"]
    assert identity["r14_dummy_unit"].endswith(".service")
    assert identity["unit_template_path"] == str(
        compat.COMPAT_UNIT_TEMPLATE_PATH
    )
    assert set(identity["protected_c4_runtime_paths"]) == {
        str(path) for path in compat.C4_RUNTIME_PATHS
    }


def test_all_c5_component_pins_are_frozen_and_match_sources() -> None:
    expected = {
        compat.COMPAT_BRIDGE_PATH: (
            "388843b9b840db41610d57543f4982666cdf442ba81fa5acb208033de062319f"
        ),
        compat.COMPAT_REALIZER_PATH: (
            "dbe35cd096554c4fd4c64b34213b0f7ac3ccb79e396f6d1d8e620c2c4c1d1be5"
        ),
        compat.COMPAT_ENVIRONMENT_PATH: (
            "69c6f3f77acd68de94cf839dfece53ccc9c81858b3867d17edfda894792b13fb"
        ),
        compat.COMPAT_SUPERVISOR_PATH: (
            "12c93e469b03e5b4b6f626e875a0934f603061c840b6614221748ac2cdd3dda2"
        ),
        compat.COMPAT_ADAPTER_PATH: (
            "bc31c82378291ef19d747c1594e0c5a9bf92b9e6fc410a82224603e2a00e8f6f"
        ),
        compat.COMPAT_UNIT_TEMPLATE_PATH: (
            "f2a3da0862addb90e61301c97e0d5c1d109e8cbf59ad86c2e5130235f8387216"
        ),
    }
    observed = {
        compat.COMPAT_BRIDGE_PATH: compat.COMPAT_BRIDGE_SHA256,
        compat.COMPAT_REALIZER_PATH: compat.COMPAT_REALIZER_SHA256,
        compat.COMPAT_ENVIRONMENT_PATH: compat.COMPAT_ENVIRONMENT_SHA256,
        compat.COMPAT_SUPERVISOR_PATH: compat.COMPAT_SUPERVISOR_SHA256,
        compat.COMPAT_ADAPTER_PATH: compat.COMPAT_ADAPTER_SHA256,
        compat.COMPAT_UNIT_TEMPLATE_PATH: (
            compat.COMPAT_UNIT_TEMPLATE_SHA256
        ),
    }
    assert observed == expected
    assert all(
        compat._sha256_file(path) == digest
        for path, digest in expected.items()
    )
    compat._require_component_sources()


def test_b5_real_source_root_collector_accepts_every_frozen_c5_source() -> None:
    from tools import cure_lite_v24_preaccess_schema_compatibility_c5 as b5

    roots = b5._collect_source_roots()
    assert roots["compat_environment_wrapper"]["file_sha256"] == (
        compat.COMPAT_ENVIRONMENT_SHA256
    )
    assert roots["compat_release"]["file_sha256"] == compat._sha256_file(
        Path(compat.__file__).resolve()
    )


def test_frozen_authorities_are_hash_before_exec_bound() -> None:
    for path, digest in (
        (compat.FROZEN_RELEASE_PATH, compat.FROZEN_RELEASE_SHA256),
        (compat.R14_INTEGRATION_PATH, compat.R14_INTEGRATION_SHA256),
        (
            compat.R14_SHARED_REALIZER_PATH,
            compat.R14_SHARED_REALIZER_SHA256,
        ),
        (compat.R14_DUMMY_CHILD_PATH, compat.R14_DUMMY_CHILD_SHA256),
        (compat.R14_DUMMY_TEMPLATE_PATH, compat.R14_DUMMY_TEMPLATE_SHA256),
    ):
        assert compat._sha256_file(path) == digest


def test_c4_runtime_r14_and_evidence_namespaces_are_protected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = tuple(tmp_path / f"direct-{index}" for index in range(4))
    c1 = tuple(tmp_path / f"c1-{index}" for index in range(4))
    c2 = tuple(tmp_path / f"c2-{index}" for index in range(4))
    c3 = tuple(tmp_path / f"c3-{index}" for index in range(4))
    c4 = tuple(tmp_path / f"c4-{index}" for index in range(4))
    aliases = tuple(tmp_path / f"alias-{index}" for index in range(10))
    c4_r14 = tmp_path / "compat-c4-r14"
    monkeypatch.setattr(compat, "BLOCKED_RUNTIME_PATHS", direct)
    monkeypatch.setattr(compat, "C1_RUNTIME_PATHS", c1)
    monkeypatch.setattr(compat, "C2_RUNTIME_PATHS", c2)
    monkeypatch.setattr(compat, "C3_RUNTIME_PATHS", c3)
    monkeypatch.setattr(compat, "C4_RUNTIME_PATHS", c4)
    monkeypatch.setattr(compat, "C4_R14_ROOT", c4_r14)
    monkeypatch.setattr(compat, "FORBIDDEN_SCIENTIFIC_ALIASES", aliases)
    c4[2].mkdir()
    with pytest.raises(PermissionError, match="c1-c4 predecessor"):
        compat._require_lane_separation()
    c4[2].rmdir()
    c4_r14.mkdir()
    with pytest.raises(PermissionError, match="c1-c4 predecessor"):
        compat._require_lane_separation()
    assert set(compat.C4_PROTECTED_EVIDENCE_PATHS).isdisjoint(
        {
            compat.COMPAT_RUNTIME_SPEC_PATH,
            compat.COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
            compat.REALIZATION_AUTHORIZATION_PATH,
            compat.REALIZATION_RECEIPT_PATH,
            compat.COMPATIBILITY_RECEIPT_PATH,
        }
    )


def test_disjoint_gate_freezes_c4_and_c5_control_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compat,
        "COMPAT_RUNTIME_SPEC_PATH",
        compat.C4_RUNTIME_PATHS[0],
    )
    with pytest.raises(PermissionError, match="identity/path isolation"):
        compat._require_disjoint_c5_runtime_identity()
    monkeypatch.undo()
    monkeypatch.setattr(compat, "INTEGRATION_ROOT", compat.C4_R14_ROOT)
    with pytest.raises(PermissionError, match="identity/path isolation"):
        compat._require_disjoint_c5_runtime_identity()


def test_l5_canonical_profile_is_utf8_single_profile() -> None:
    value = {"授权依据": "用户指令：修改后继续", "finite": 1.25}
    encoded = compat._canonical_json(value)
    assert "修改后继续" in encoded
    assert "\\u4fee" not in encoded
    assert encoded == json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert compat._stable_fingerprint(value) == hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()


def test_l5_loader_accepts_utf8_and_rejects_ascii_escaped_self_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "runtime-spec.json").absolute()
    monkeypatch.setattr(compat, "COMPAT_RUNTIME_SPEC_PATH", path)
    body = {
        "schema_version": "runtime-spec-v1",
        "authorization_basis": "用户指令：修改后继续",
    }
    payload = {
        **body,
        "runtime_spec_fingerprint": compat._stable_fingerprint(body),
    }
    path.write_text(
        compat._canonical_json(payload) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    assert compat._load_l5_sealed(
        path,
        fingerprint_field="runtime_spec_fingerprint",
        schema="runtime-spec-v1",
    ) == payload
    path.chmod(0o644)
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    with pytest.raises(PermissionError, match="canonical JSON"):
        compat._load_l5_sealed(
            path,
            fingerprint_field="runtime_spec_fingerprint",
            schema="runtime-spec-v1",
        )


def test_b5_unicode_receipt_uses_fixed_producer_validator_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "runtime_compatibility_id": "c5",
        "authorization_basis": "用户指令：修改后继续",
        "automatic_retry": False,
        "resume": False,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "compatibility_source_roots": _source_roots(),
    }
    calls: list[tuple[object, object, bool, bool]] = []
    monkeypatch.setattr(
        compat.compat_bridge,
        "verify_compatibility_receipt",
        lambda path, **kwargs: calls.append(
            (
                path,
                kwargs["expected_spec"],
                kwargs["require_spec_binding"],
                kwargs["allow_runtime_activation"],
            )
        )
        or result,
        raising=False,
    )
    monkeypatch.setattr(
        compat,
        "_verify_environment_cross_binding",
        lambda value: {"receipt": value},
    )
    monkeypatch.setattr(
        compat,
        "_stable_fingerprint",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("L5 must not re-fingerprint B5")
        ),
    )
    expected = {"runtime_spec_fingerprint": "a" * 64}
    assert compat._verify_compatibility_receipt(
        expected_spec=expected,
        require_spec_binding=True,
    ) == result
    assert calls == [
        (compat.COMPATIBILITY_RECEIPT_PATH, expected, True, False)
    ]


def test_r5_archival_uses_r5_validator_and_e5_root_binder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_identity = {
        "path": str(compat.REALIZATION_AUTHORIZATION_PATH),
        "file_sha256": "a" * 64,
    }
    receipt_identity = {
        "path": str(compat.REALIZATION_RECEIPT_PATH),
        "file_sha256": "b" * 64,
    }
    authorization = {
        "authorization_fingerprint": "c" * 64,
    }
    receipt = {
        "authorization_file_sha256": "a" * 64,
        "authorization_fingerprint": "c" * 64,
    }
    archival = {
        "authorization": authorization,
        "receipt": receipt,
        "authorization_identity": authorization_identity,
        "receipt_identity": receipt_identity,
        "compatibility_closure": {"passed": True},
    }
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        compat.compat_realizer,
        "validate_archival_realization_chain",
        lambda *args, **kwargs: calls.append((*args, kwargs)) or archival,
        raising=False,
    )
    monkeypatch.setattr(
        compat.compat_realizer,
        "_validate_c5_receipt_contract",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        compat.compat_environment,
        "_bind_r5_archival_root",
        lambda path, value: {**dict(value), "path": str(path)},
        raising=False,
    )
    result = compat._load_r5_archival()
    assert result["compatibility_closure"] == {"passed": True}
    assert calls == [
        (
            compat.REALIZATION_AUTHORIZATION_PATH,
            compat.REALIZATION_RECEIPT_PATH,
            {"allow_runtime_activation": False},
        )
    ]


def test_e5_cross_binding_uses_e5_sealed_readers_and_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = SimpleNamespace(name="old")
    c5 = SimpleNamespace(name="c5")
    archival = {"authorization": {}, "receipt": {}}
    normalized_old = {"说明": "历史"}
    normalized_c5 = {"说明": "当前"}
    monkeypatch.setattr(
        compat.compat_environment,
        "replay_old_scope_and_handoff",
        lambda: (old, c5, {}),
        raising=False,
    )
    monkeypatch.setattr(compat, "_load_r5_archival", lambda: archival)
    monkeypatch.setattr(
        compat.compat_environment,
        "validate_c5_realization_archival",
        lambda value, *, contract: value,
        raising=False,
    )
    roots: dict[str, dict[str, object]] = {}

    def read_live(path: Path, fingerprint_field: str):
        value = {"path": str(path), "说明": "修改后继续"}
        root = {"path": str(path), "field": fingerprint_field}
        roots[str(path)] = root
        return value, root

    monkeypatch.setattr(
        compat.compat_environment,
        "_read_live_sealed",
        read_live,
        raising=False,
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        compat.compat_environment,
        "validate_c5_environment_closure",
        lambda *args, **kwargs: calls.append((*args, kwargs))
        or {"realization": archival},
        raising=False,
    )
    monkeypatch.setattr(
        compat.compat_environment,
        "_normalized_contract",
        lambda value: normalized_old if value is old else normalized_c5,
        raising=False,
    )
    receipt = {
        "historical_environment_contract": normalized_old,
        "current_environment_contract": normalized_c5,
    }
    assert compat._verify_environment_cross_binding(receipt) == {
        "realization": archival
    }
    kwargs = calls[0][-1]
    assert kwargs["c5_contract"] is c5
    assert kwargs["live_roots"] == {
        "policy": roots[str(compat.POLICY_PATH)],
        "stability": roots[str(compat.STABILITY_PATH)],
    }


def test_generator_late_inode_rotation_is_the_only_accepted_policy_drift() -> None:
    before = _unit_path_policy(11)
    after = _unit_path_policy(12)
    compat.r14_integration._validate_unit_path_policy_transition(
        before,
        after,
        authorized_uid=os.getuid(),
        allow_generator_late_inode_rotation=True,
    )
    unsafe_mode = deepcopy(after)
    unsafe_mode["ordered_unit_paths"][0]["mode"] = 0o755
    with pytest.raises(PermissionError, match="more than its inode"):
        compat.r14_integration._validate_unit_path_policy_transition(
            before,
            unsafe_mode,
            authorized_uid=os.getuid(),
            allow_generator_late_inode_rotation=True,
        )
    unsafe_other = deepcopy(after)
    unsafe_other["ordered_unit_paths"][1]["inode"] += 1
    with pytest.raises(PermissionError, match="only one same-index"):
        compat.r14_integration._validate_unit_path_policy_transition(
            before,
            unsafe_other,
            authorized_uid=os.getuid(),
            allow_generator_late_inode_rotation=True,
        )


def test_r14_preheartbeat_timeline_is_complete_and_archival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoNowDateTime(datetime):
        @classmethod
        def now(cls, *_args, **_kwargs):
            raise AssertionError("archival validation must not read now")

    keys = (
        "launch_lease",
        "precommit_phase_receipt",
        "attempt_commit",
        "materialization_claim",
        "start_ack_receipt",
        "child_prespawn_phase_receipt",
    )
    artifacts = {key: f"/archived/{key}.json" for key in keys}
    payloads: dict[str, dict[str, object]] = {}
    for index, key in enumerate(keys, start=1):
        payloads[artifacts[key]] = {
            "attempt_id": "attempt-r14-c5",
            "boot_id": "boot-r14-c5",
            "time_utc": f"2020-01-01T00:00:0{index}Z",
            "monotonic_ns": index,
        }
    payloads[artifacts["attempt_commit"]]["systemd_unit_name"] = (
        compat.R14_DUMMY_UNIT
    )
    payloads[artifacts["materialization_claim"]][
        "systemd_control_group"
    ] = f"/user.slice/{compat.R14_DUMMY_UNIT}"
    monkeypatch.setattr(compat, "datetime", NoNowDateTime)
    monkeypatch.setattr(
        compat.r14_integration,
        "_read_sealed",
        lambda path, **_kwargs: deepcopy(payloads[str(path)]),
    )
    spec = {"attempt_id": "attempt-r14-c5", "artifacts": artifacts}
    assert tuple(compat._validate_r14_preheartbeat_timeline(spec)) == keys
    payloads[artifacts["start_ack_receipt"]]["monotonic_ns"] = 3
    with pytest.raises(PermissionError, match="preheartbeat chronology"):
        compat._validate_r14_preheartbeat_timeline(spec)


def test_r14_heartbeat_chain_is_continuous_and_hash_linked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_root = tmp_path / "heartbeat"
    heartbeat_root.mkdir()
    first_path = heartbeat_root / "000000000000.json"
    second_path = heartbeat_root / "000000000001.json"
    first_path.touch()
    second_path.touch()
    terminal_path = tmp_path / "runtime-terminal.json"
    terminal_path.touch()
    claim = {
        "attempt_id": "attempt-r14-c5",
        "boot_id": "boot-r14-c5",
        "time_utc": "2020-01-01T00:00:04Z",
        "monotonic_ns": 4,
    }
    anchor = {
        "time_utc": "2020-01-01T00:00:06Z",
        "monotonic_ns": 6,
    }
    invocation_id = "a" * 32
    first = {
        "attempt_id": "attempt-r14-c5",
        "boot_id": "boot-r14-c5",
        "sequence": 0,
        "time_utc": "2020-01-01T00:00:07Z",
        "monotonic_ns": 7,
        "systemd_invocation_id": invocation_id,
        "previous_event_file_sha256": (
            compat.r14_integration.sealed_file_sha256(claim)
        ),
        "supervisor_pid": 10,
        "supervisor_proc_starttime_ticks": 20,
        "child_pid": 30,
        "child_proc_starttime_ticks": 40,
    }
    second = {
        **first,
        "sequence": 1,
        "time_utc": "2020-01-01T00:00:08Z",
        "monotonic_ns": 8,
        "previous_event_file_sha256": (
            compat.r14_integration.sealed_file_sha256(first)
        ),
    }
    terminal = {
        "attempt_id": "attempt-r14-c5",
        "boot_id": "boot-r14-c5",
        "systemd_invocation_id": invocation_id,
        "heartbeat_event_count": 2,
        "last_heartbeat_path": str(second_path),
        "last_heartbeat_file_sha256": (
            compat.r14_integration.sealed_file_sha256(second)
        ),
        "time_utc": "2020-01-01T00:00:09Z",
        "monotonic_ns": 9,
    }
    payloads = {
        str(first_path): first,
        str(second_path): second,
        str(terminal_path): terminal,
    }
    monkeypatch.setattr(
        compat.r14_integration,
        "_read_sealed",
        lambda path, **_kwargs: deepcopy(payloads[str(path)]),
    )
    spec = {
        "attempt_id": "attempt-r14-c5",
        "artifacts": {
            "heartbeat_dir": str(heartbeat_root),
            "runtime_terminal": str(terminal_path),
        },
    }
    assert compat._validate_r14_heartbeat_chain(
        spec,
        invocation_id=invocation_id,
        claim=claim,
        chronology_anchor=anchor,
    ) == terminal
    payloads[str(second_path)]["previous_event_file_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="hash/time chain"):
        compat._validate_r14_heartbeat_chain(
            spec,
            invocation_id=invocation_id,
            claim=claim,
            chronology_anchor=anchor,
        )


def test_r14_complete_runtime_chain_orchestrates_all_archival_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = (compat.INTEGRATION_ROOT / "runtime").absolute()
    artifacts = {
        "root": str(runtime_root),
        **{
            key: str(runtime_root / name)
            for key, name in compat.r14_integration._SPEC_ARTIFACT_NAMES.items()
        },
    }
    spec = {
        "runtime_spec_fingerprint": "a" * 64,
        "attempt_id": "attempt-r14-c5",
        "artifacts": artifacts,
        "runtime": {"systemd": {"unit_name": compat.R14_DUMMY_UNIT}},
        "source_bindings": {
            "supervisor_file_sha256": compat.COMPAT_SUPERVISOR_SHA256,
            "child_entry_file_sha256": compat.R14_DUMMY_CHILD_SHA256,
        },
        "child": {"entrypoint_path": str(compat.R14_DUMMY_CHILD_PATH)},
        "execution_kind": "systemd_integration_dummy",
        "authorization": None,
        "environment": None,
        "scientific_preaccess": None,
    }
    invocation_id = "b" * 32
    evidence = {"invocation_id": invocation_id}
    terminal = {
        "supervisor_evidence": evidence,
        "created_at_utc": "2020-01-01T00:00:05Z",
    }
    cgroup = f"/user.slice/{compat.R14_DUMMY_UNIT}"
    timeline = {
        "launch_lease": {"time_utc": "2020-01-01T00:00:02Z"},
        "attempt_commit": {"systemd_unit_name": compat.R14_DUMMY_UNIT},
        "materialization_claim": {"systemd_control_group": cgroup},
        "child_prespawn_phase_receipt": {},
    }
    runtime_terminal = {
        "time_utc": "2020-01-01T00:00:03Z",
        "stdout_log": {"path": artifacts["stdout_log"]},
        "stderr_log": {"path": artifacts["stderr_log"]},
    }
    sidecar = {
        "time_utc": "2020-01-01T00:00:04Z",
        "systemd_control_group": cgroup,
    }
    dummy = {
        "scenario_id": compat.R14_SCENARIO_ID,
        "dataset_accessed": False,
        "gpu_accessed": False,
        "torch_imported": False,
    }
    authorization = {
        "issued_at_utc": "2020-01-01T00:00:01Z",
        "expires_at_utc": "2020-01-01T00:04:59Z",
        "runtime_spec_binding": {
            "path": str(compat.INTEGRATION_RUNTIME_SPEC_PATH),
            "file_sha256": "spec-file-sha",
            "runtime_spec_fingerprint": "a" * 64,
        },
        "executable_bindings": {
            name: {"name": name}
            for name in (
                "integration_tool",
                "realizer",
                "dummy_child",
                "supervisor",
            )
        },
        "template_binding": {"name": "template"},
        "identity": {"unit_name": compat.R14_DUMMY_UNIT},
        "control_artifacts": {
            "dummy_artifact": str(runtime_root / "dummy-child.json")
        },
    }
    sidecar_path = Path(artifacts["systemd_invocation_dir"]) / (
        invocation_id + ".json"
    )
    payloads = {
        str(compat.INTEGRATION_RUNTIME_SPEC_PATH): spec,
        str(compat.INTEGRATION_TERMINAL_PATH): terminal,
        str(sidecar_path): sidecar,
        str(runtime_root / "dummy-child.json"): dummy,
    }
    monkeypatch.setattr(
        compat.r14_integration,
        "_read_sealed",
        lambda path, **_kwargs: deepcopy(payloads[str(path)]),
    )
    monkeypatch.setattr(
        compat.r14_integration,
        "file_sha256",
        lambda _path: "spec-file-sha",
    )
    monkeypatch.setattr(
        compat.r14_integration,
        "_validate_supervisor_evidence",
        lambda _authorization: evidence,
    )
    source_calls: list[str] = []
    monkeypatch.setattr(
        compat,
        "_require_r14_source_binding",
        lambda _binding, *, path, digest: source_calls.append(str(path)),
    )
    monkeypatch.setattr(
        compat,
        "_validate_r14_preheartbeat_timeline",
        lambda _spec: timeline,
    )
    monkeypatch.setattr(
        compat,
        "_validate_r14_heartbeat_chain",
        lambda *_args, **_kwargs: runtime_terminal,
    )
    monkeypatch.setattr(compat, "_validate_r14_log", lambda *_a, **_k: None)
    b5_receipt = {"created_at_utc": "2020-01-01T00:00:00Z"}
    result = compat._validate_r14_runtime_chain(
        {"authorization": authorization},
        b5_receipt=b5_receipt,
    )
    assert result == {
        "runtime_spec": spec,
        "supervisor_evidence": evidence,
        "terminal": terminal,
        "dummy": dummy,
        "sidecar": sidecar,
    }
    assert source_calls == [
        str(compat.R14_INTEGRATION_PATH),
        str(compat.R14_SHARED_REALIZER_PATH),
        str(compat.R14_DUMMY_CHILD_PATH),
        str(compat.COMPAT_SUPERVISOR_PATH),
        str(compat.R14_DUMMY_TEMPLATE_PATH),
    ]


def test_r14_chronology_has_no_current_time_or_reordering_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoNowDateTime(datetime):
        @classmethod
        def now(cls, *_args, **_kwargs):
            raise AssertionError("archival validation must not read now")

    monkeypatch.setattr(compat, "datetime", NoNowDateTime)
    values = {
        "b5_receipt": {"created_at_utc": "2020-01-01T00:00:00Z"},
        "authorization": {
            "issued_at_utc": "2020-01-01T00:00:01Z",
            "expires_at_utc": "2020-01-01T00:04:59Z",
        },
        "timeline": {
            "launch_lease": {"time_utc": "2020-01-01T00:00:02Z"}
        },
        "runtime_terminal": {"time_utc": "2020-01-01T00:00:03Z"},
        "sidecar": {"time_utc": "2020-01-01T00:00:04Z"},
        "integration_terminal": {
            "created_at_utc": "2020-01-01T00:00:05Z"
        },
    }
    compat._validate_r14_archival_chronology(**values)
    reordered = deepcopy(values)
    reordered["sidecar"]["time_utc"] = "2020-01-01T00:00:02Z"
    with pytest.raises(PermissionError, match="archival authorization chronology"):
        compat._validate_r14_archival_chronology(**reordered)


def _fixed_build_argv() -> list[str]:
    return [
        "build-spec",
        "--environment-policy",
        str(compat.POLICY_PATH),
        "--precleanup-receipt",
        str(compat.PRECLEANUP_PATH),
        "--cleanup-plan",
        str(compat.CLEANUP_PLAN_PATH),
        "--cleanup-authorization",
        str(compat.CLEANUP_AUTHORIZATION_PATH),
        "--cleanup-receipt",
        str(compat.CLEANUP_RECEIPT_PATH),
        "--stability-receipt",
        str(compat.STABILITY_PATH),
        "--postcleanup-audit",
        str(compat.POSTCLEANUP_PATH),
        "--integration-authorization",
        str(compat.INTEGRATION_AUTHORIZATION_PATH),
        "--integration-receipt",
        str(compat.INTEGRATION_RECEIPT_PATH),
        "--unit-realization-authorization",
        str(compat.REALIZATION_AUTHORIZATION_PATH),
        "--unit-realization-receipt",
        str(compat.REALIZATION_RECEIPT_PATH),
    ]


def test_public_inherited_release_surfaces_are_read_only_facades() -> None:
    assert not isinstance(compat.legacy, ModuleType)
    assert not isinstance(compat.compat_c1, ModuleType)
    assert not isinstance(compat.compat_c1.legacy, ModuleType)
    assert compat.legacy.UNIT_NAME == compat.COMPAT_UNIT
    assert compat.compat_c1.COMPAT_UNIT == compat.COMPAT_UNIT
    for view, name in (
        (compat.legacy, "main"),
        (compat.legacy, "build_spec"),
        (compat.legacy, "authorize_launch"),
        (compat.legacy, "_write_sealed"),
        (compat.legacy, "_write_sealed_bound"),
        (compat.legacy, "_private_directory"),
        (
            compat.legacy,
            "_create_runtime_directories_and_verify_leaves",
        ),
        (compat.compat_c1, "main"),
        (compat.compat_c1, "_frozen_write_sealed"),
        (
            compat.compat_c1,
            "_frozen_create_runtime_directories",
        ),
    ):
        with pytest.raises(PermissionError, match="private"):
            getattr(view, name)
    with pytest.raises(PermissionError, match="read-only"):
        compat.legacy.UNIT_NAME = "drift"
    with pytest.raises(PermissionError, match="read-only"):
        compat.compat_c1.COMPAT_UNIT = "drift"


def test_every_internal_runtime_mutator_requires_l5_capability_before_io(
    tmp_path: Path,
) -> None:
    target = tmp_path / "forbidden-runtime-spec.json"
    artifacts = {"root": str(tmp_path / "forbidden-root")}
    calls = (
        lambda: compat._legacy.main(["authorize-launch"]),
        lambda: compat._legacy.build_spec(),
        lambda: compat._legacy.authorize_launch(),
        lambda: compat._legacy._write_sealed(
            target,
            {},
            fingerprint_field="runtime_spec_fingerprint",
        ),
        lambda: compat._legacy._write_sealed_bound(
            target,
            {},
            fingerprint_field="runtime_spec_fingerprint",
        ),
        lambda: compat._legacy._private_directory(target, create=True),
        lambda: compat._legacy._create_runtime_directories_and_verify_leaves(
            artifacts
        ),
        lambda: compat._compat_c1._frozen_write_sealed(
            target,
            {},
            fingerprint_field="runtime_spec_fingerprint",
        ),
        lambda: compat._compat_c1._frozen_create_runtime_directories(
            artifacts
        ),
    )
    for call in calls:
        with pytest.raises(PermissionError):
            call()
    assert not target.exists()
    assert not (tmp_path / "forbidden-root").exists()


def test_exact_suboperation_validator_rejects_wrong_path_body_and_artifacts(
    tmp_path: Path,
) -> None:
    argv = tuple(_fixed_build_argv())
    body = {"schema_version": "test-runtime-spec-v1"}
    approved = {
        **body,
        "runtime_spec_fingerprint": compat._stable_fingerprint(body),
    }
    token = compat._APPROVED_PREWRITE_SPEC.set(approved)
    try:
        compat._validate_delegated_suboperation(
            "compat-write-sealed",
            "build-spec",
            (compat.COMPAT_RUNTIME_SPEC_PATH, body),
            {"fingerprint_field": "runtime_spec_fingerprint"},
            delegated_argv=argv,
        )
        with pytest.raises(PermissionError, match="write binding"):
            compat._validate_delegated_suboperation(
                "compat-write-sealed",
                "build-spec",
                (tmp_path / "wrong.json", body),
                {"fingerprint_field": "runtime_spec_fingerprint"},
                delegated_argv=argv,
            )
        with pytest.raises(PermissionError, match="write binding"):
            compat._validate_delegated_suboperation(
                "compat-write-sealed",
                "build-spec",
                (compat.COMPAT_RUNTIME_SPEC_PATH, {**body, "drift": True}),
                {"fingerprint_field": "runtime_spec_fingerprint"},
                delegated_argv=argv,
            )
    finally:
        compat._APPROVED_PREWRITE_SPEC.reset(token)
    exact_artifacts = compat._legacy._artifact_paths()
    compat._validate_delegated_suboperation(
        "create-runtime-directories",
        "build-spec",
        (exact_artifacts,),
        {},
        delegated_argv=argv,
    )
    with pytest.raises(PermissionError, match="runtime-directory"):
        compat._validate_delegated_suboperation(
            "create-runtime-directories",
            "build-spec",
            ({**exact_artifacts, "root": str(tmp_path)},),
            {},
            delegated_argv=argv,
        )
    with pytest.raises(PermissionError, match="path binding"):
        compat._validate_delegated_suboperation(
            "private-directory",
            "build-spec",
            (tmp_path / "wrong-directory",),
            {"create": True},
            delegated_argv=argv,
        )
    assert not (tmp_path / "wrong.json").exists()
    assert not (tmp_path / "wrong-directory").exists()


@pytest.mark.parametrize(
    ("owner", "name"),
    (
        ("legacy", "_write_sealed"),
        ("legacy", "_write_sealed_bound"),
        ("legacy", "_private_directory"),
        ("legacy", "_create_runtime_directories_and_verify_leaves"),
        ("legacy", "validate_release_closure"),
        ("c1", "_compat_write_sealed"),
        ("c1", "_frozen_write_sealed"),
        ("c1", "_frozen_create_runtime_directories"),
    ),
)
def test_inherited_guard_alias_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    name: str,
) -> None:
    module = compat._legacy if owner == "legacy" else compat._compat_c1
    monkeypatch.setattr(module, name, lambda *_args, **_kwargs: None)
    with pytest.raises(PermissionError, match="guard identity changed"):
        compat.verify_compatibility_identity()


def test_active_command_cannot_widen_suboperation_and_capability_resets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = tmp_path / "forbidden-active-directory"

    def fail_inside_nested_operation(*_args, **_kwargs):
        raise RuntimeError("forced nested operation failure")

    def reject_after_probe():
        with pytest.raises(PermissionError):
            compat._legacy._private_directory(forbidden, create=True)
        nested_caller = compat._guard_release_callable(
            fail_inside_nested_operation,
            "authorize-launch",
            operation="private-directory",
            caller_codes=(reject_after_probe.__code__,),
        )
        with pytest.raises(RuntimeError, match="nested operation failure"):
            nested_caller(compat.EVIDENCE_ROOT)
        compat._require_no_active_l5_suboperation()
        with pytest.raises(PermissionError, match="delegated caller changed"):
            compat._legacy._private_directory(compat.EVIDENCE_ROOT)
        raise RuntimeError("stop after active-capability probe")

    monkeypatch.setattr(
        compat,
        "_verify_active_phase_compatibility_receipt",
        reject_after_probe,
    )
    with pytest.raises(RuntimeError, match="active-capability probe"):
        compat.main(["authorize-launch"])
    assert not forbidden.exists()
    with pytest.raises(PermissionError, match="delegation is not authorized"):
        compat._require_l5_delegation("authorize-launch")
    compat._require_no_active_l5_suboperation()


def test_guard_and_validator_global_rebinding_cannot_bypass_public_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compat,
        "_validate_delegated_suboperation",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(PermissionError, match="guard identity changed"):
        compat.verify_compatibility_identity()
    monkeypatch.undo()
    monkeypatch.setattr(
        compat,
        "_require_inherited_release_guard_identities",
        lambda: None,
    )
    monkeypatch.setattr(
        compat._compat_c1,
        "_frozen_create_runtime_directories",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(PermissionError, match="guard identity changed"):
        compat.verify_compatibility_identity()


def test_source_has_no_scientific_entry_retry_resume_or_dv_dt() -> None:
    source = Path(compat.__file__).read_text(encoding="utf-8")
    assert '"automatic_retry_allowed": False' in source
    assert '"resume_allowed": False' in source
    assert '"D_V_payload_accessed": False' in source
    assert '"D_T_payload_accessed": False' in source
    assert '"D_R_structural_update_count": 0' in source
    assert "systemctl" not in source
    assert "Popen" not in source
    assert "def train" not in source
    assert "def run_scientific" not in source
