from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import cure_lite_v24_dummy_r3_recovery as recovery


def test_r3_archived_anchors_are_complete_and_untruncated() -> None:
    expected = {
        "authorization": (
            "a1d3d91a7c83d37e814e7f58e5f0a3bab539e0ef317294ecb839228ae83324fd",
            "ef7a2959d1a9becbda13f0544fa2ff29bdb96dce9d3f2a231fce3814a9ce9162",
            0o444,
        ),
        "runtime_spec": (
            "819eb43a9f06cc0f2be0a8bdf3a2c31a579ca08873ca76e9134f9ebc062bac09",
            "f1486339a6db8a855c8648448398e013f01ceb296dc8d6647a8477957e3651f4",
            0o444,
        ),
        "launch_lease": (
            "ea9601b70f1b5997d01f810cedac001408fbefc6886cdd18c334d1b5468ed7b8",
            "4c7ce93852018682aad974192af8c2dbf5a1a8437462eb6067e21442e6369ca6",
            0o444,
        ),
        "precommit": (
            "88852a28add05dc8eb4ba6c802111ec8ea8a0b60018facbe9bf8e6769b62b404",
            "0943bbd5b3ef545b73a7a3cc8e326949a7e942e312e58d6116c082064f9ac8de",
            0o444,
        ),
        "attempt_commit": (
            "e11967beee0417a9b614a2152b51adacb6bc1b9fc98eef40b3510189a3ded52b",
            "60f6cd8222be8d714f509792c9fb98eb18d71de9992d56e9dd5e1973a5dfc892",
            0o444,
        ),
        "start_ack": (
            "48b8eaa99129d356c3aa00fc2730d9fffbf32b88d6c79a6403517b9257e02709",
            "643e83a0514f22dea22584c8f8ac2461c4d1d1d27cbc6720253ea7d642f2177e",
            0o444,
        ),
        "systemd_sidecar": (
            "849276b7849d3acfa87e7474f9e7c6986f99d169808e42a99e2e8774dcb7ea97",
            "3c3962e4a902efdc39da05dbd92c438b48b80475c36b5df4aef6b5b1d8c40074",
            0o400,
        ),
        "integration_terminal": (
            "b27cccaab88af2ec2dc2a44114dfb1c1494c42ae7b45e7f52bb56c8798b623b8",
            "33c0ec3be2cb13004a3980b29974711da0267e120f4077649aab1589f6e16caa",
            0o444,
        ),
        "removal_state": (
            "57e7f63ee0bfe60058fc43db67b5b8a515c8e8f3fc0201877db812307f9d9a09",
            "45dfc13b60f3b0ee388639abdee23ba165333b6ab691d1e3562fa57f2e73b612",
            0o444,
        ),
    }
    assert set(recovery._ARCHIVED_EVIDENCE_ANCHORS) == set(expected)
    for name, (file_digest, fingerprint, mode) in expected.items():
        anchor = recovery._ARCHIVED_EVIDENCE_ANCHORS[name]
        assert anchor["file_sha256"] == file_digest
        assert anchor["fingerprint"] == fingerprint
        assert anchor["mode"] == mode
        assert Path(str(anchor["path"])).is_absolute()
        assert len(str(anchor["file_sha256"])) == 64
        assert len(str(anchor["fingerprint"])) == 64


def test_archived_reader_accepts_exact_mode_0400_and_rejects_mode_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sidecar.json"
    body = {"schema_version": "test-sidecar-v1", "value": "sealed"}
    payload = {
        **body,
        "systemd_terminal_fingerprint": recovery.stable_fingerprint(body),
    }
    encoded = recovery.canonical_json(payload) + "\n"
    path.write_text(encoded, encoding="utf-8")
    path.chmod(0o400)
    anchor = {
        "path": str(path),
        "file_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "fingerprint_field": "systemd_terminal_fingerprint",
        "fingerprint": payload["systemd_terminal_fingerprint"],
        "schema_version": "test-sidecar-v1",
        "mode": 0o400,
    }
    monkeypatch.setitem(
        recovery._ARCHIVED_EVIDENCE_ANCHORS,
        "temporary_sidecar",
        anchor,
    )
    assert recovery._read_archived("temporary_sidecar") == payload
    path.chmod(0o444)
    with pytest.raises(PermissionError, match="identity changed"):
        recovery._read_archived("temporary_sidecar")


def test_fd_snapshot_rejects_symlink_for_hash_binding_and_fragment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "regular.txt"
    target.write_text("exact\n", encoding="utf-8")
    target.chmod(0o600)
    linked = tmp_path / "linked.txt"
    linked.symlink_to(target)

    with pytest.raises(PermissionError, match="safe regular file"):
        recovery.file_sha256(linked)
    with pytest.raises(PermissionError, match="safe regular file"):
        recovery._file_binding(linked)
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_PATH", linked)
    monkeypatch.setattr(
        recovery,
        "EXPECTED_FRAGMENT_OWNER_UID",
        os.getuid(),
    )
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_MODE", 0o600)
    with pytest.raises(PermissionError, match="safe regular file"):
        recovery._fragment_identity(linked)


def test_archived_fd_snapshot_rejects_same_content_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "archive.json"
    replacement = tmp_path / "replacement.json"
    body = {"schema_version": "test-archive-v1", "value": "exact"}
    payload = {
        **body,
        "archive_fingerprint": recovery.stable_fingerprint(body),
    }
    encoded = (recovery.canonical_json(payload) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    path.chmod(0o444)
    replacement.write_bytes(encoded)
    replacement.chmod(0o444)
    monkeypatch.setitem(
        recovery._ARCHIVED_EVIDENCE_ANCHORS,
        "replacement_archive",
        {
            "path": str(path),
            "file_sha256": hashlib.sha256(encoded).hexdigest(),
            "fingerprint_field": "archive_fingerprint",
            "fingerprint": payload["archive_fingerprint"],
            "schema_version": "test-archive-v1",
            "mode": 0o444,
        },
    )
    original_reader = recovery._read_fd_bytes
    replaced = False

    def replace_after_read(descriptor: int) -> bytes:
        nonlocal replaced
        data = original_reader(descriptor)
        if not replaced:
            replaced = True
            os.replace(replacement, path)
        return data

    monkeypatch.setattr(
        recovery,
        "_read_fd_bytes",
        replace_after_read,
    )
    with pytest.raises(PermissionError, match="identity changed"):
        recovery._read_archived("replacement_archive")
    assert replaced is True
    assert path.read_bytes() == encoded


def test_recovery_reader_rejects_same_inode_drift_during_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    path = control / "sealed.json"
    recovery._write_sealed(
        path,
        {"schema_version": "test-recovery-v1", "value": "before"},
        fingerprint_field="test_fingerprint",
    )
    original_reader = recovery._read_fd_bytes
    mutated = False

    def mutate_after_read(descriptor: int) -> bytes:
        nonlocal mutated
        data = original_reader(descriptor)
        if not mutated:
            mutated = True
            path.chmod(0o600)
            path.write_text(
                '{"schema_version":"test-recovery-v1","value":"after"}\n',
                encoding="utf-8",
            )
            path.chmod(0o444)
        return data

    monkeypatch.setattr(recovery, "_read_fd_bytes", mutate_after_read)
    with pytest.raises(PermissionError, match="identity changed"):
        recovery._read_recovery_sealed(
            path,
            fingerprint_field="test_fingerprint",
            schema="test-recovery-v1",
        )
    assert mutated is True


def test_write_sealed_readback_rejects_same_content_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    path = control / "sealed.json"
    replacement = control / "replacement.json"
    original_reader = recovery._read_fd_bytes
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
        recovery,
        "_read_fd_bytes",
        replace_during_readback,
    )
    with pytest.raises(PermissionError, match="identity"):
        recovery._write_sealed(
            path,
            {"schema_version": "test-write-v1", "value": "exact"},
            fingerprint_field="test_fingerprint",
        )
    assert replaced is True


def test_write_sealed_rejects_parent_directory_generation_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    displaced = tmp_path / "displaced-control"
    path = control / "sealed.json"
    original_reader = recovery._read_fd_bytes
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
        recovery,
        "_read_fd_bytes",
        replace_parent_during_readback,
    )
    with pytest.raises(PermissionError, match="parent changed"):
        recovery._write_sealed(
            path,
            {"schema_version": "test-parent-v1", "value": "exact"},
            fingerprint_field="test_fingerprint",
        )
    assert replaced is True
    assert (displaced / path.name).exists()
    assert path.exists()


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
                "inode": 38247,
                "owner_uid": 1008,
                "mode": 0o755,
            },
        ],
    }
    regenerated = deepcopy(base)
    regenerated["ordered_unit_paths"][1]["inode"] = 39001
    assert recovery._validated_recovery_path_policy(
        base,
        regenerated,
    ) == regenerated

    unsafe = deepcopy(regenerated)
    unsafe["ordered_unit_paths"][0]["inode"] = 39002
    with pytest.raises(PermissionError, match="path identity changed"):
        recovery._validated_recovery_path_policy(base, unsafe)
    reordered = deepcopy(regenerated)
    reordered["ordered_unit_paths"].reverse()
    with pytest.raises(PermissionError, match="search path set changed"):
        recovery._validated_recovery_path_policy(base, reordered)


def _patch_exact_fragment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, object, dict[str, object]]:
    runtime = tmp_path / "systemd" / "user"
    runtime.mkdir(parents=True, mode=0o700)
    fragment = runtime / recovery.UNIT_NAME
    text = "[Unit]\nDescription=r3 exact recovery test\n"
    fragment.write_text(text, encoding="utf-8")
    fragment.chmod(0o600)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    plan = recovery.realizer.build_realization_plan(
        unit_name=recovery.UNIT_NAME,
        unit_directory=runtime,
        fragment_text=text,
        expected_fragment_sha256=digest,
        execute_authorized=True,
        removal_authorized=True,
    )
    observed = fragment.lstat()
    identity = {
        "fragment_path": str(fragment),
        "fragment_sha256": digest,
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "owner_uid": observed.st_uid,
        "mode": 0o600,
        "nlink": 1,
    }
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
    return fragment, plan, identity


def test_exact_unlink_performs_four_full_checks_immediately_before_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment, plan, identity = _patch_exact_fragment(tmp_path, monkeypatch)
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
    fresh_checks: list[str] = []
    monkeypatch.setattr(
        recovery,
        "_revalidate_preunlink_state",
        lambda **_kwargs: fresh_checks.append("fresh"),
    )
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
    now = datetime.now(timezone.utc)
    authorization = {
        "issued_at_utc": (now - timedelta(seconds=1)).isoformat(),
        "expires_at_utc": (now + timedelta(seconds=60)).isoformat(),
    }
    intent = {"created_at_utc": now.isoformat()}
    action_trace: list[str] = []
    recovery._remove_exact_authorized_fragment(
        plan,
        expected_identity=identity,
        authorization=authorization,
        intent=intent,
        expected_recovery_tool_binding=tool_binding,
        expected_required_bindings=required_bindings,
        expected_manager_generation={"manager": "stable"},
        expected_unit_path_policy={"paths": "stable"},
        expected_inactive_static_state={"state": "inactive"},
        runner=object(),
        manager_reader=lambda: {"manager": "stable"},
        on_action_started=action_trace.append,
    )
    assert not fragment.exists()
    assert len(hash_checks) == 4
    assert len(action_trace) == 1
    assert fresh_checks == ["fresh"]


def test_exact_unlink_rejects_replacement_expiry_and_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "_revalidate_preunlink_state",
        lambda **_kwargs: None,
    )
    now = datetime.now(timezone.utc)
    authorization = {
        "issued_at_utc": (now - timedelta(seconds=1)).isoformat(),
        "expires_at_utc": (now + timedelta(seconds=60)).isoformat(),
    }
    intent = {"created_at_utc": now.isoformat()}

    fragment, plan, identity = _patch_exact_fragment(
        tmp_path / "replacement",
        monkeypatch,
    )
    replacement = fragment.with_name("same-content.service")
    replacement.write_bytes(fragment.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, fragment)
    with pytest.raises(PermissionError, match="exact check 1"):
        recovery._remove_exact_authorized_fragment(
            plan,
            expected_identity=identity,
            authorization=authorization,
            intent=intent,
            expected_recovery_tool_binding=tool_binding,
            expected_required_bindings=required_bindings,
            expected_manager_generation={"manager": "stable"},
            expected_unit_path_policy={"paths": "stable"},
            expected_inactive_static_state={"state": "inactive"},
            runner=object(),
            manager_reader=lambda: {"manager": "stable"},
            on_action_started=lambda _value: None,
        )
    assert fragment.exists()

    fragment, plan, identity = _patch_exact_fragment(
        tmp_path / "expired",
        monkeypatch,
    )
    expired = {
        "issued_at_utc": (now - timedelta(seconds=2)).isoformat(),
        "expires_at_utc": (now - timedelta(seconds=1)).isoformat(),
    }
    with pytest.raises(PermissionError, match="expired before unlink"):
        recovery._remove_exact_authorized_fragment(
            plan,
            expected_identity=identity,
            authorization=expired,
            intent=intent,
            expected_recovery_tool_binding=tool_binding,
            expected_required_bindings=required_bindings,
            expected_manager_generation={"manager": "stable"},
            expected_unit_path_policy={"paths": "stable"},
            expected_inactive_static_state={"state": "inactive"},
            runner=object(),
            manager_reader=lambda: {"manager": "stable"},
            on_action_started=lambda _value: None,
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
        recovery._remove_exact_authorized_fragment(
            plan,
            expected_identity=identity,
            authorization=authorization,
            intent=intent,
            expected_recovery_tool_binding=tool_binding,
            expected_required_bindings=required_bindings,
            expected_manager_generation={"manager": "stable"},
            expected_unit_path_policy={"paths": "stable"},
            expected_inactive_static_state={"state": "inactive"},
            runner=object(),
            manager_reader=lambda: {"manager": "stable"},
            on_action_started=lambda _value: None,
        )
    assert fragment.exists()


def test_post_intent_preunlink_revalidation_rejects_state_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_manager = {"manager": "exact"}
    expected_policy = {"paths": "exact"}
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
        lambda *_args, **_kwargs: expected_policy,
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
    recovery._revalidate_preunlink_state(
        plan=SimpleNamespace(),
        expected_manager_generation=expected_manager,
        expected_unit_path_policy=expected_policy,
        expected_inactive_static_state=expected_state,
        runner=object(),
        manager_reader=lambda: expected_manager,
    )
    observed_state["ActiveState"] = "active"
    observed_state["SubState"] = "running"
    with pytest.raises(PermissionError, match="unit state changed"):
        recovery._revalidate_preunlink_state(
            plan=SimpleNamespace(),
            expected_manager_generation=expected_manager,
            expected_unit_path_policy=expected_policy,
            expected_inactive_static_state=expected_state,
            runner=object(),
            manager_reader=lambda: expected_manager,
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
    fragment.write_text("r3 recovery mock fragment\n", encoding="utf-8")
    fragment.chmod(0o600)
    recovery_authorization_path = control / "recovery-authorization.json"
    recovery_intent_path = control / "recovery-intent.json"
    recovery_terminal_path = control / "recovery-terminal.json"
    monkeypatch.setattr(
        recovery,
        "RECOVERY_AUTHORIZATION_PATH",
        recovery_authorization_path,
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_INTENT_PATH",
        recovery_intent_path,
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_TERMINAL_PATH",
        recovery_terminal_path,
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
    fragment_identity = recovery._expected_fragment_identity()
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
        "fragment_identity": fragment_identity,
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
        "_revalidate_manager_generation",
        lambda *_args, **_kwargs: None,
    )
    return fragment, context, tool_binding, required_bindings


def test_two_phase_success_is_create_once_and_single_action(
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
    assert authorization["start_authorized"] is False
    assert authorization["automatic_retry_authorized"] is False
    with pytest.raises(FileExistsError):
        recovery.create_recovery_authorization(validity_seconds=120)
    authorization_file_sha256 = hashlib.sha256(
        recovery.RECOVERY_AUTHORIZATION_PATH.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        recovery,
        "file_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("authorization digest must use its payload FD snapshot")
        ),
    )

    calls: list[str] = []

    def remove_with_intent(
        _plan: object,
        *,
        expected_identity: object,
        authorization: object,
        intent: object,
        expected_recovery_tool_binding: object,
        expected_required_bindings: object,
        on_action_started: object,
        **_fresh_state: object,
    ) -> None:
        assert recovery.RECOVERY_INTENT_PATH.exists()
        assert expected_identity == context["fragment_identity"]
        assert expected_recovery_tool_binding == tool_binding
        assert expected_required_bindings == required_bindings
        assert intent["recovery_intent_fingerprint"]
        on_action_started(recovery._utc_now())
        fragment.unlink()
        calls.append("remove")

    monkeypatch.setattr(
        recovery,
        "_remove_exact_authorized_fragment",
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
    assert calls == ["remove", "reload:True"]
    intent = recovery._read_recovery_sealed(
        recovery.RECOVERY_INTENT_PATH,
        fingerprint_field="recovery_intent_fingerprint",
        schema=recovery.INTENT_SCHEMA,
    )
    assert (
        intent["recovery_authorization_file_sha256"]
        == authorization_file_sha256
    )
    with pytest.raises(FileExistsError):
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
        expected_identity: object,
        authorization: object,
        intent: object,
        expected_recovery_tool_binding: object,
        expected_required_bindings: object,
        on_action_started: object,
        **_fresh_state: object,
    ) -> None:
        assert recovery.RECOVERY_INTENT_PATH.exists()
        on_action_started(recovery._utc_now())
        raise RuntimeError("synthetic action-start failure")

    monkeypatch.setattr(
        recovery,
        "_remove_exact_authorized_fragment",
        fail_after_action_started,
    )
    with pytest.raises(RuntimeError, match="synthetic action-start failure"):
        recovery.execute_recovery(execute=True)
    assert fragment.exists()
    terminal = recovery._read_recovery_sealed(
        recovery.RECOVERY_TERMINAL_PATH,
        fingerprint_field="recovery_terminal_fingerprint",
        schema=recovery.TERMINAL_SCHEMA,
    )
    assert terminal["passed"] is False
    assert terminal["action_started_at_utc"] is not None
    assert terminal["completed_actions"] == []
    assert terminal["error_type"] == "RuntimeError"
    assert terminal["error_message"] == "synthetic action-start failure"
    with pytest.raises(FileExistsError):
        recovery.execute_recovery(execute=True)


def test_post_unlink_observation_failure_is_terminalized_and_cannot_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment, _context, _tool_binding, _required_bindings = (
        _mock_two_phase_context(tmp_path, monkeypatch)
    )
    recovery.create_recovery_authorization(validity_seconds=120)

    def remove_once(
        _plan: object,
        *,
        on_action_started: object,
        **_kwargs: object,
    ) -> None:
        on_action_started(recovery._utc_now())
        fragment.unlink()

    monkeypatch.setattr(
        recovery,
        "_remove_exact_authorized_fragment",
        remove_once,
    )
    monkeypatch.setattr(
        recovery.realizer,
        "daemon_reload",
        lambda **_kwargs: None,
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
    monkeypatch.setattr(
        recovery,
        "_observe_fragment_absent",
        lambda _path: (_ for _ in ()).throw(
            OSError("synthetic post-unlink observation failure")
        ),
    )
    with pytest.raises(
        OSError,
        match="synthetic post-unlink observation failure",
    ):
        recovery.execute_recovery(execute=True)
    terminal = recovery._read_recovery_sealed(
        recovery.RECOVERY_TERMINAL_PATH,
        fingerprint_field="recovery_terminal_fingerprint",
        schema=recovery.TERMINAL_SCHEMA,
    )
    assert terminal["passed"] is False
    assert terminal["fragment_absent"] is False
    assert terminal["completed_actions"] == [
        "remove-exact-runtime-static-fragment",
        "daemon-reload",
        "verify-not-found",
    ]
    assert terminal["error_type"] == "OSError"
    with pytest.raises(FileExistsError):
        recovery.execute_recovery(execute=True)
