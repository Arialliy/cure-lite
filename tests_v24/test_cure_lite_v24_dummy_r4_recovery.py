from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import cure_lite_v24_dummy_r4_recovery as recovery


def test_r4_archived_anchors_are_complete_and_exact() -> None:
    expected = {
        "authorization": (
            "3c4c68e81591c2f2b349432d078c48075427382a8d01d4b9a990dfa620d23dd9",
            "f0f3d6ff9064595d93c98253dac295bae3e2713f2313ca1e7fa2291427320a63",
        ),
        "runtime_spec": (
            "df853b4789638e1cac2dcb3cbc721160c63d5e8cbac1111c113f4ac57a6a737c",
            "7bbec06c9e920ceae7ace6433cbd96bb9b5abe260a8383af0591674d5918da6e",
        ),
        "launch_lease": (
            "ed69c1aee649b0e4b6ad658749424c7b7e0086eab89133126d29117a7e5d309b",
            "d5be45b011adec5b2dc2a656c1bf9685424e941405638eb70907cd8326a309bc",
        ),
        "precommit": (
            "41465e296bf9d9367c7a26ca3bbd872f4154c2153aea02e4efdc3123955a5de9",
            "9eafa2e9688b976e8d54b0c8911f0c6fabc201b7208566b183f6028edbf1c2b2",
        ),
        "attempt_commit": (
            "a3f2b7a2b904dcde154e8889ef0358088024a8b8d3b692220cfe8e2f10729f04",
            "2447a42b3d7ce7e1724a850c3cfa481d2c127204d7811e3cdb9827c7656e3c4b",
        ),
        "systemd_sidecar": (
            "46f793ba45182e16a6fb2466cfb325eedf185c29cf3c9a1b9fcf589d18b9ea39",
            "8ef43c821e656681dc132c6d4572dca9026cf0ca40a372bd9052a9cd90014d08",
        ),
        "consumed_start_failure": (
            "7cde7ee0dca2fe6b0edfe5c678baa1342af226a251e5bfa49de45ee8e9a6c28e",
            "71777aab978d6e827d4e6fc2f1d991e5e789df79d60b424765cae949f5b0c02a",
        ),
        "integration_terminal": (
            "845a1b87517b435ae67969c0fde27b824e33f3b451113495bef206b21cb50fcc",
            "117a7011a87dc3a7a544cb98842a2ecb6f95b5643c8856a7d38ad30ac66e00e5",
        ),
        "removal_state": (
            "409e8484cc7dcdb05d1614f5e7d425de9380290f23d838a5bdf6e6781080c121",
            "78a8fa0b7769f0830a9069c2850fb419e979c551f46b174bc8017bc3853f3763",
        ),
    }
    assert set(recovery._ARCHIVED_EVIDENCE_ANCHORS) == set(expected)
    for name, (file_digest, fingerprint) in expected.items():
        anchor = recovery._ARCHIVED_EVIDENCE_ANCHORS[name]
        assert anchor["file_sha256"] == file_digest
        assert anchor["fingerprint"] == fingerprint
        assert anchor["mode"] == 0o444
        assert Path(str(anchor["path"])).is_absolute()


def test_r4_original_failure_chain_is_sealed_and_payload_free() -> None:
    chain = recovery._sealed_original_chain()
    assert chain["authorization"]["scenario_id"] == recovery.SCENARIO_ID
    assert chain["sidecar"]["claim_valid"] is False
    assert chain["sidecar"]["start_ack_valid"] is False
    assert chain["sidecar"]["child_prespawn_valid"] is False
    assert chain["consumed"]["attempt_consumed"] is True
    assert chain["consumed"]["automatic_retry_allowed"] is False
    assert chain["terminal"]["passed"] is False
    assert chain["removal"]["remove_attempted"] is False


def test_r4_identity_is_independent_and_binds_frozen_r3_hardening() -> None:
    r3 = recovery.hardened
    r4_paths = {
        recovery.RECOVERY_AUTHORIZATION_PATH,
        recovery.RECOVERY_INTENT_PATH,
        recovery.RECOVERY_TERMINAL_PATH,
    }
    r3_paths = {
        r3.RECOVERY_AUTHORIZATION_PATH,
        r3.RECOVERY_INTENT_PATH,
        r3.RECOVERY_TERMINAL_PATH,
    }
    assert r4_paths.isdisjoint(r3_paths)
    assert all(path.name.startswith("r4-exact-recovery-") for path in r4_paths)
    assert {
        recovery.AUTHORIZATION_SCHEMA,
        recovery.INTENT_SCHEMA,
        recovery.TERMINAL_SCHEMA,
    }.isdisjoint(
        {
            r3.AUTHORIZATION_SCHEMA,
            r3.INTENT_SCHEMA,
            r3.TERMINAL_SCHEMA,
        }
    )
    assert recovery.SCENARIO_ID != r3.SCENARIO_ID
    assert (
        hashlib.sha256(Path(str(r3.__file__)).read_bytes()).hexdigest()
        == recovery.FROZEN_HARDENED_IO_SHA256
        == "b3d7fd5b98f70db98ec637dbbed3bc4f428a9290cb32cdc39e6fe46f0cc0a7f4"
    )
    bindings = recovery._current_required_bindings()
    assert (
        bindings["hardened_io_library"]["file_sha256"]
        == recovery.FROZEN_HARDENED_IO_SHA256
    )


def test_only_generator_late_inode_regeneration_is_accepted() -> None:
    base = {
        "runtime_directory": "/run/user/1008/systemd/user",
        "runtime_directory_priority": 7,
        "ordered_unit_paths": [
            {
                "path": "/run/user/1008/systemd/user",
                "exists": True,
                "device": 54,
                "inode": 37880,
                "owner_uid": 1008,
                "mode": 0o755,
            },
            {
                "path": "/run/user/1008/systemd/generator.late",
                "exists": True,
                "device": 54,
                "inode": 38465,
                "owner_uid": 1008,
                "mode": 0o755,
            },
        ],
    }
    regenerated = deepcopy(base)
    regenerated["ordered_unit_paths"][1]["inode"] = 39001
    assert recovery._validated_path_policy(base, regenerated) == regenerated

    unsafe = deepcopy(regenerated)
    unsafe["ordered_unit_paths"][0]["inode"] = 39002
    with pytest.raises(PermissionError, match="unit path changed"):
        recovery._validated_path_policy(base, unsafe)
    reordered = deepcopy(regenerated)
    reordered["ordered_unit_paths"].reverse()
    with pytest.raises(PermissionError, match="ordered unit path set changed"):
        recovery._validated_path_policy(base, reordered)


def test_sealed_read_rejects_same_content_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    path = control / "sealed.json"
    recovery.hardened._write_sealed(
        path,
        {"schema_version": "r4-test-read-v1", "value": "exact"},
        fingerprint_field="test_fingerprint",
    )
    replacement = control / "replacement.json"
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o444)
    original_reader = recovery.hardened._read_fd_bytes
    replaced = False

    def replace_after_read(descriptor: int) -> bytes:
        nonlocal replaced
        data = original_reader(descriptor)
        if not replaced:
            replaced = True
            os.replace(replacement, path)
        return data

    monkeypatch.setattr(
        recovery.hardened,
        "_read_fd_bytes",
        replace_after_read,
    )
    with pytest.raises(PermissionError, match="identity changed"):
        recovery._read_recovery_snapshot(
            path,
            fingerprint_field="test_fingerprint",
            schema="r4-test-read-v1",
        )
    assert replaced is True


def test_sealed_write_rejects_same_content_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    path = control / "sealed.json"
    replacement = control / "replacement.json"
    original_reader = recovery.hardened._read_fd_bytes
    replaced = False

    def replace_during_readback(descriptor: int) -> bytes:
        nonlocal replaced
        data = original_reader(descriptor)
        if not replaced:
            replaced = True
            replacement.write_bytes(data)
            replacement.chmod(0o444)
            os.replace(replacement, path)
        return data

    monkeypatch.setattr(
        recovery.hardened,
        "_read_fd_bytes",
        replace_during_readback,
    )
    with pytest.raises(PermissionError, match="identity"):
        recovery.hardened._write_sealed(
            path,
            {"schema_version": "r4-test-write-v1", "value": "exact"},
            fingerprint_field="test_fingerprint",
        )
    assert replaced is True


def test_sealed_write_rejects_parent_directory_generation_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    displaced = tmp_path / "displaced-control"
    path = control / "sealed.json"
    original_reader = recovery.hardened._read_fd_bytes
    replaced = False

    def replace_parent_during_readback(descriptor: int) -> bytes:
        nonlocal replaced
        data = original_reader(descriptor)
        if not replaced:
            replaced = True
            control.rename(displaced)
            control.mkdir(mode=0o700)
            replacement = control / path.name
            replacement.write_bytes(data)
            replacement.chmod(0o444)
        return data

    monkeypatch.setattr(
        recovery.hardened,
        "_read_fd_bytes",
        replace_parent_during_readback,
    )
    with pytest.raises(PermissionError, match="parent changed"):
        recovery.hardened._write_sealed(
            path,
            {"schema_version": "r4-test-parent-v1", "value": "exact"},
            fingerprint_field="test_fingerprint",
        )
    assert replaced is True
    assert (displaced / path.name).exists()
    assert path.exists()


def _patch_exact_fragment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, object, dict[str, object]]:
    runtime = tmp_path / "systemd" / "user"
    runtime.mkdir(parents=True, mode=0o700)
    fragment = runtime / recovery.UNIT_NAME
    text = "[Unit]\nDescription=r4 exact recovery test\n"
    fragment.write_text(text, encoding="utf-8")
    fragment.chmod(0o600)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    observed = fragment.lstat()
    monkeypatch.setattr(recovery, "EXPECTED_RUNTIME_UNIT_DIRECTORY", runtime)
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_PATH", fragment)
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_SHA256", digest)
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_DEVICE", observed.st_dev)
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_INODE", observed.st_ino)
    monkeypatch.setattr(
        recovery,
        "EXPECTED_FRAGMENT_OWNER_UID",
        observed.st_uid,
    )
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_MODE", 0o600)
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_NLINK", 1)
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_SIZE", observed.st_size)
    monkeypatch.setattr(
        recovery,
        "EXPECTED_FRAGMENT_MTIME_NS",
        observed.st_mtime_ns,
    )
    monkeypatch.setattr(
        recovery,
        "EXPECTED_FRAGMENT_CTIME_NS",
        observed.st_ctime_ns,
    )
    plan = recovery.realizer.build_realization_plan(
        unit_name=recovery.UNIT_NAME,
        unit_directory=runtime,
        fragment_text=text,
        expected_fragment_sha256=digest,
        execute_authorized=True,
        removal_authorized=True,
    )
    return fragment, plan, recovery._expected_fragment_identity()


def test_fragment_snapshot_rejects_symlink_and_same_content_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment, _plan, _identity = _patch_exact_fragment(
        tmp_path / "replacement",
        monkeypatch,
    )
    replacement = fragment.with_name("same-content.service")
    replacement.write_bytes(fragment.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, fragment)
    with pytest.raises(PermissionError, match="identity changed"):
        recovery._fragment_identity(fragment)

    target = tmp_path / "symlink" / "target.service"
    target.parent.mkdir(parents=True)
    target.write_text("exact\n", encoding="utf-8")
    target.chmod(0o600)
    linked = target.with_name(recovery.UNIT_NAME)
    linked.symlink_to(target)
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_PATH", linked)
    with pytest.raises(PermissionError):
        recovery._fragment_identity(linked)


def _patch_unlink_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, str], dict[str, str]]:
    tool_binding = {"tool": "stable"}
    required_bindings = {"libraries": "stable"}
    monkeypatch.setattr(
        recovery,
        "_current_recovery_tool_binding",
        lambda: tool_binding,
    )
    monkeypatch.setattr(
        recovery,
        "_current_required_bindings",
        lambda: required_bindings,
    )
    monkeypatch.setattr(
        recovery,
        "_revalidate_preunlink",
        lambda **_kwargs: None,
    )
    return tool_binding, required_bindings


def _unlink_times() -> tuple[dict[str, str], dict[str, str]]:
    now = datetime.now(timezone.utc)
    return (
        {
            "issued_at_utc": (now - timedelta(seconds=1)).isoformat(),
            "expires_at_utc": (now + timedelta(seconds=60)).isoformat(),
        },
        {"created_at_utc": now.isoformat()},
    )


def _remove(
    plan: object,
    identity: dict[str, object],
    authorization: dict[str, str],
    intent: dict[str, str],
    tool_binding: dict[str, str],
    required_bindings: dict[str, str],
    *,
    on_action_started: object = lambda _value: None,
) -> None:
    recovery._remove_exact_fragment(
        plan,
        expected_identity=identity,
        authorization=authorization,
        intent=intent,
        recovery_tool=tool_binding,
        required_bindings=required_bindings,
        manager_generation={"manager": "stable"},
        unit_path_policy={"paths": "stable"},
        inactive_state={"state": "inactive"},
        runner=object(),
        manager_reader=lambda: {"manager": "stable"},
        on_action_started=on_action_started,
    )


def test_exact_unlink_uses_four_same_fd_hash_and_parent_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment, plan, identity = _patch_exact_fragment(tmp_path, monkeypatch)
    tool_binding, required_bindings = _patch_unlink_closure(monkeypatch)
    authorization, intent = _unlink_times()
    original_fd_sha256 = recovery.realizer._fd_sha256
    hash_checks: list[int] = []

    def counted_fd_sha256(descriptor: int) -> str:
        hash_checks.append(descriptor)
        return original_fd_sha256(descriptor)

    monkeypatch.setattr(
        recovery.realizer,
        "_fd_sha256",
        counted_fd_sha256,
    )
    action_trace: list[str] = []
    _remove(
        plan,
        identity,
        authorization,
        intent,
        tool_binding,
        required_bindings,
        on_action_started=action_trace.append,
    )
    assert not fragment.exists()
    assert len(hash_checks) == 4
    assert len(set(hash_checks)) == 1
    assert len(action_trace) == 1


def test_exact_unlink_rejects_replacement_expiry_and_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_binding, required_bindings = _patch_unlink_closure(monkeypatch)
    authorization, intent = _unlink_times()

    fragment, plan, identity = _patch_exact_fragment(
        tmp_path / "replacement",
        monkeypatch,
    )
    replacement = fragment.with_name("same-content.service")
    replacement.write_bytes(fragment.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, fragment)
    with pytest.raises(PermissionError, match="check 1"):
        _remove(
            plan,
            identity,
            authorization,
            intent,
            tool_binding,
            required_bindings,
        )
    assert fragment.exists()

    fragment, plan, identity = _patch_exact_fragment(
        tmp_path / "expired",
        monkeypatch,
    )
    now = datetime.now(timezone.utc)
    expired = {
        "issued_at_utc": (now - timedelta(seconds=2)).isoformat(),
        "expires_at_utc": (now - timedelta(seconds=1)).isoformat(),
    }
    with pytest.raises(PermissionError, match="expired before unlink"):
        _remove(
            plan,
            identity,
            expired,
            intent,
            tool_binding,
            required_bindings,
        )
    assert fragment.exists()

    fragment, plan, identity = _patch_exact_fragment(
        tmp_path / "binding-drift",
        monkeypatch,
    )
    calls = 0

    def drifting_bindings() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return (
            required_bindings
            if calls == 1
            else {"libraries": "changed"}
        )

    monkeypatch.setattr(
        recovery,
        "_current_required_bindings",
        drifting_bindings,
    )
    with pytest.raises(PermissionError, match="implementation changed"):
        _remove(
            plan,
            identity,
            authorization,
            intent,
            tool_binding,
            required_bindings,
        )
    assert fragment.exists()


def test_exact_unlink_rejects_parent_directory_generation_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment, plan, identity = _patch_exact_fragment(tmp_path, monkeypatch)
    tool_binding, required_bindings = _patch_unlink_closure(monkeypatch)
    authorization, intent = _unlink_times()
    runtime = fragment.parent
    displaced = runtime.with_name("displaced-user")

    def replace_parent(**_kwargs: object) -> None:
        content = fragment.read_bytes()
        runtime.rename(displaced)
        runtime.mkdir(mode=0o700)
        replacement = runtime / recovery.UNIT_NAME
        replacement.write_bytes(content)
        replacement.chmod(0o600)

    monkeypatch.setattr(recovery, "_revalidate_preunlink", replace_parent)
    with pytest.raises(PermissionError, match="check 3"):
        _remove(
            plan,
            identity,
            authorization,
            intent,
            tool_binding,
            required_bindings,
        )
    assert (displaced / recovery.UNIT_NAME).exists()
    assert (runtime / recovery.UNIT_NAME).exists()


def test_post_intent_preunlink_revalidation_rejects_state_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = {"manager": "exact"}
    policy = {"paths": "exact"}
    expected_state = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "FragmentPath": str(recovery.EXPECTED_FRAGMENT_PATH),
        "DropInPaths": "",
        "Transient": "no",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
    }
    observed_state = dict(expected_state)
    monkeypatch.setattr(
        recovery.integration,
        "_validate_manager_generation",
        lambda _value: None,
    )
    monkeypatch.setattr(
        recovery.realizer,
        "freeze_user_unit_path_policy",
        lambda *_args, **_kwargs: policy,
    )
    monkeypatch.setattr(
        recovery.realizer,
        "query_unit_properties",
        lambda *_args, **_kwargs: observed_state,
    )
    monkeypatch.setattr(
        recovery.realizer,
        "validate_realized_static_unit",
        lambda *_args, **_kwargs: None,
    )
    recovery._revalidate_preunlink(
        plan=SimpleNamespace(),
        manager_generation=manager,
        unit_path_policy=policy,
        inactive_state=expected_state,
        runner=object(),
        manager_reader=lambda: manager,
    )
    observed_state["ActiveState"] = "active"
    observed_state["SubState"] = "running"
    with pytest.raises(PermissionError, match="unit state changed"):
        recovery._revalidate_preunlink(
            plan=SimpleNamespace(),
            manager_generation=manager,
            unit_path_policy=policy,
            inactive_state=expected_state,
            runner=object(),
            manager_reader=lambda: manager,
        )


def _mock_two_phase_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object], dict[str, str], dict[str, str]]:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    fragment = runtime / recovery.UNIT_NAME
    fragment.write_text("r4 recovery mock fragment\n", encoding="utf-8")
    fragment.chmod(0o600)
    monkeypatch.setattr(
        recovery,
        "RECOVERY_AUTHORIZATION_PATH",
        control / "r4-exact-recovery-authorization.json",
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_INTENT_PATH",
        control / "r4-exact-recovery-intent.json",
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_TERMINAL_PATH",
        control / "r4-exact-recovery-terminal.json",
    )
    tool_binding = {"tool": "stable"}
    required_bindings = {"libraries": "stable"}
    monkeypatch.setattr(
        recovery,
        "_current_recovery_tool_binding",
        lambda: tool_binding,
    )
    monkeypatch.setattr(
        recovery,
        "_current_required_bindings",
        lambda: required_bindings,
    )
    state = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "FragmentPath": str(recovery.EXPECTED_FRAGMENT_PATH),
        "DropInPaths": "",
        "Transient": "no",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
    }
    context = {
        "archived_roots": recovery._archived_roots(),
        "archived_executable_bindings": {"historical": "fixed"},
        "manager_generation": {"manager": "fixed"},
        "unit_path_policy": {"paths": "fixed"},
        "fragment_identity": recovery._expected_fragment_identity(),
        "inactive_static_state": state,
        "plan": SimpleNamespace(fragment_path=fragment),
    }
    monkeypatch.setattr(
        recovery,
        "_live_context",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        recovery,
        "_revalidate_manager",
        lambda *_args, **_kwargs: None,
    )
    return fragment, context, tool_binding, required_bindings


def test_two_phase_success_is_create_once_and_unlink_reload_notfound_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment, context, tool_binding, required_bindings = (
        _mock_two_phase_context(tmp_path, monkeypatch)
    )
    authorization = recovery.create_recovery_authorization(
        validity_seconds=120
    )
    assert authorization["remove_authorized"] is True
    assert authorization["daemon_reload_authorized"] is True
    assert authorization["start_authorized"] is False
    assert authorization["stop_authorized"] is False
    assert authorization["enable_authorized"] is False
    assert authorization["automatic_retry_authorized"] is False
    with pytest.raises(FileExistsError, match="consumed"):
        recovery.create_recovery_authorization(validity_seconds=120)
    authorization_file_sha256 = hashlib.sha256(
        recovery.RECOVERY_AUTHORIZATION_PATH.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        recovery.hardened,
        "file_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("authorization digest must use the sealed FD")
        ),
    )
    calls: list[str] = []
    expected_required_bindings = required_bindings

    def remove_with_intent(
        _plan: object,
        *,
        expected_identity: object,
        recovery_tool: object,
        required_bindings: object,
        intent: dict[str, object],
        on_action_started: object,
        **_closure: object,
    ) -> None:
        assert recovery.RECOVERY_INTENT_PATH.exists()
        assert expected_identity == context["fragment_identity"]
        assert recovery_tool == tool_binding
        assert required_bindings == expected_required_bindings
        assert intent["recovery_intent_fingerprint"]
        on_action_started(recovery._utc_now())
        fragment.unlink()
        calls.append("unlink")

    monkeypatch.setattr(
        recovery,
        "_remove_exact_fragment",
        remove_with_intent,
    )
    monkeypatch.setattr(
        recovery.realizer,
        "daemon_reload",
        lambda *, execute, runner: calls.append(f"reload:{execute}"),
    )
    monkeypatch.setattr(
        recovery.realizer,
        "wait_until_unit_not_found",
        lambda *_args, **_kwargs: {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "FragmentPath": "",
        },
    )
    terminal = recovery.execute_recovery(execute=True)
    assert terminal["passed"] is True
    assert terminal["fragment_absent"] is True
    assert terminal["completed_actions"] == [
        "remove-exact-r4-runtime-static-fragment",
        "daemon-reload",
        "verify-not-found",
    ]
    assert calls == ["unlink", "reload:True"]
    intent = recovery._read_recovery(
        recovery.RECOVERY_INTENT_PATH,
        fingerprint_field="recovery_intent_fingerprint",
        schema=recovery.INTENT_SCHEMA,
    )
    assert (
        intent["recovery_authorization_file_sha256"]
        == authorization_file_sha256
    )
    with pytest.raises(FileExistsError, match="consumed"):
        recovery.execute_recovery(execute=True)


def test_action_started_failure_is_terminalized_and_cannot_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment, _context, _tool_binding, _required_bindings = (
        _mock_two_phase_context(tmp_path, monkeypatch)
    )
    recovery.create_recovery_authorization(validity_seconds=120)

    def fail_after_action_started(
        _plan: object,
        *,
        on_action_started: object,
        **_closure: object,
    ) -> None:
        assert recovery.RECOVERY_INTENT_PATH.exists()
        on_action_started(recovery._utc_now())
        raise RuntimeError("synthetic r4 action-start failure")

    monkeypatch.setattr(
        recovery,
        "_remove_exact_fragment",
        fail_after_action_started,
    )
    with pytest.raises(RuntimeError, match="synthetic r4 action-start failure"):
        recovery.execute_recovery(execute=True)
    assert fragment.exists()
    terminal = recovery._read_recovery(
        recovery.RECOVERY_TERMINAL_PATH,
        fingerprint_field="recovery_terminal_fingerprint",
        schema=recovery.TERMINAL_SCHEMA,
    )
    assert terminal["passed"] is False
    assert terminal["action_started_at_utc"] is not None
    assert terminal["completed_actions"] == []
    assert terminal["error_type"] == "RuntimeError"
    assert terminal["error_message"] == "synthetic r4 action-start failure"
    with pytest.raises(FileExistsError, match="consumed"):
        recovery.execute_recovery(execute=True)
