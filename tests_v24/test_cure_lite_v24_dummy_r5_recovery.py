from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import cure_lite_v24_dummy_r5_recovery as recovery


def test_r5_archived_anchors_are_complete_and_exact() -> None:
    expected = {
        "authorization": (
            "4b299e67735962dc35c04ecc10ba81e741449a463c17a9c8ec50285615b8666f",
            "356a941fe5ce56d138a442164973fcbf39f1668d8ce4b87253e7d6ae7832af83",
        ),
        "runtime_spec": (
            "e99fdc7610cd9791aaba5bcff653046a430b2d6698ab420bda24ad488db0f73b",
            "8b9f07d26a8b3f799fafd2c8a39351a64bbf985c791e97e3dab51aa825ad3e35",
        ),
        "launch_lease": (
            "0bc7461b7c351fb0e0ad0754a45cb3fbe3c0323acd82e9c2c1438acc1daa8c46",
            "c9dca67f6102b16b463c1fabfc8ca0f514aadd67ef8155671a5a120c18a7da06",
        ),
        "precommit": (
            "d86aca87e8701290a93d4e509e32b18c954cb76d1f021ab971c95f80ed958bb7",
            "80aff33a41deef930ab69dffde1cf2c03b9410dd4422a0d32b2dc86de445d6d8",
        ),
        "attempt_commit": (
            "032b42ea061cf1a44c6f8bd6747c51c3b7a03447fec54a78771583d457cf0fbe",
            "e577893103ca34b69e1fbcd0049a3b03d252ffb0e53d471624bed0f5871edebb",
        ),
        "materialization_claim": (
            "534825b3197d38b0b15111efa9ac6a45ed0371e1f1d11516059815208ac3098e",
            "317223cf5052d6bb60758ffc0773acdcf18243c99148a0265ec685fb64a0d965",
        ),
        "start_ack": (
            "19a599fb8869cb992923289e6cfef4d7c2f34ccb79f5e223016cf701a24fdf2c",
            "2f8f10a03bcfc1b9156de5ebe2a76f13e25911620016385e520a743df0ee03d3",
        ),
        "child_prespawn": (
            "60347268dd7aa566880dda243b5ad81713024a4ffac1bfed654809f62f52c11c",
            "93568c319f68e71a532846c6a6c3abe78e47bee99465784b027172f3c79382b1",
        ),
        "heartbeat_0": (
            "32962b374bc7fba15a7ece542b99aef36d391fe0f666eea292902478346be294",
            "48de1dc1da75656413eb3d403481a90542e3bc89b2e42bffa5df6bac9b23e9b5",
        ),
        "heartbeat_1": (
            "b88cbf7968a147a0a784ad0d0c64088a0c708427a56101f5bf85391064013a93",
            "059679a3caf922853d5f024e81b10d8a2cc772cbb83e9d5baeee47f5ffe85623",
        ),
        "heartbeat_2": (
            "ed4dbbcb8b984c98cf9cd0f9617551d27b48de9c40674101db759f02364e9dcd",
            "870f3a6f02a0a1a00fa211a46720c47e92839f704af144cc4794f738e476af68",
        ),
        "runtime_terminal": (
            "f9fcb1097de3954fb0f07cb778810b8e234a01075d0aec96d564f09ca3215475",
            "ff8071066b68ec8733e02e0bc523f75ff8ecd8072e240752fcb1ee1973d40d88",
        ),
        "systemd_sidecar": (
            "17aaf28d9f1b63fece0b61de1d399716d2f33b552212c36aae84f5a262db7348",
            "ceb3cc0294eca53c7287baa1a524c80c9086c9ae0f4fe4534c54fddec0520594",
        ),
        "dummy_artifact": (
            "73a25ee8213a2f40a8c34a14fa26eddd6b30e856b606919abb278360b789b9d1",
            "5937a41d2cebf73a58fc895f464cc368c1854668fe850a28f475b7a4c417cbe2",
        ),
        "stdout_log": (
            "d52a241a7082efe38a5bc6cb7b647d7c5bc49fb1cdbfda0342acc1077ac1b20c",
            None,
        ),
        "stderr_log": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            None,
        ),
        "integration_terminal": (
            "3d606b9277f9921a65df4c9469dc3a991b376c9ecfee89ca35424d0029c44cd6",
            "c8a83b36f4d6b8e9b5fd2f27197f74b11dcd669d7e28c0552568c656cbd2e4d0",
        ),
        "removal_state": (
            "4a744432b298fff1de41a55e2f4e4f6d566ad65e511b3b06954ff4919b69a122",
            "52ff8242e80c28cf4e2f04a60d7731a1ba8bda9ed4f56d09d74446d4942f9f35",
        ),
    }
    assert set(recovery._ARCHIVED_EVIDENCE_ANCHORS) == set(expected)
    for name, (file_digest, fingerprint) in expected.items():
        anchor = recovery._ARCHIVED_EVIDENCE_ANCHORS[name]
        assert anchor["file_sha256"] == file_digest
        assert anchor["fingerprint"] == fingerprint
        assert anchor["mode"] == 0o444
        assert Path(str(anchor["path"])).is_absolute()


def test_r5_success_chain_and_cleanup_failure_are_sealed() -> None:
    chain = recovery._sealed_original_chain()
    assert chain["authorization"]["scenario_id"] == recovery.SCENARIO_ID
    assert chain["sidecar"]["claim_valid"] is True
    assert chain["sidecar"]["start_ack_valid"] is True
    assert chain["sidecar"]["child_prespawn_valid"] is True
    assert chain["sidecar"]["audit_valid"] is True
    assert chain["sidecar"]["systemd_outcome"]["systemd_success"] is True
    assert chain["runtime_terminal"]["child_outcome"]["raw_return_code"] == 0
    assert chain["integration_terminal"]["passed"] is True
    assert chain["removal"]["passed"] is False
    assert chain["removal"]["remove_attempted"] is False
    assert (
        chain["removal"]["error_message"]
        == "unit search paths changed after terminal"
    )
    assert chain["dummy"]["dataset_accessed"] is False
    assert chain["dummy"]["gpu_accessed"] is False


def test_r5_identity_is_independent_and_binds_frozen_r4_and_r3() -> None:
    r4 = recovery.hardened
    r3 = recovery.io
    r5_paths = {
        recovery.RECOVERY_AUTHORIZATION_PATH,
        recovery.RECOVERY_INTENT_PATH,
        recovery.RECOVERY_TERMINAL_PATH,
    }
    r4_paths = {
        r4.RECOVERY_AUTHORIZATION_PATH,
        r4.RECOVERY_INTENT_PATH,
        r4.RECOVERY_TERMINAL_PATH,
    }
    assert r5_paths.isdisjoint(r4_paths)
    assert all(path.name.startswith("r5-exact-recovery-") for path in r5_paths)
    assert {
        recovery.AUTHORIZATION_SCHEMA,
        recovery.INTENT_SCHEMA,
        recovery.TERMINAL_SCHEMA,
    }.isdisjoint(
        {
            r4.AUTHORIZATION_SCHEMA,
            r4.INTENT_SCHEMA,
            r4.TERMINAL_SCHEMA,
        }
    )
    assert recovery.SCENARIO_ID != r4.SCENARIO_ID
    assert (
        hashlib.sha256(Path(str(r4.__file__)).read_bytes()).hexdigest()
        == recovery.FROZEN_R4_RECOVERY_SHA256
    )
    assert (
        hashlib.sha256(Path(str(r3.__file__)).read_bytes()).hexdigest()
        == recovery.FROZEN_R3_IO_SHA256
    )
    bindings = recovery._current_required_bindings()
    assert (
        bindings["r4_recovery_template"]["file_sha256"]
        == recovery.FROZEN_R4_RECOVERY_SHA256
    )
    assert (
        bindings["hardened_io_library"]["file_sha256"]
        == recovery.FROZEN_R3_IO_SHA256
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
                "inode": 38643,
                "owner_uid": 1008,
                "mode": 0o755,
            },
        ],
    }
    regenerated = deepcopy(base)
    regenerated["ordered_unit_paths"][1]["inode"] = 38692
    assert recovery._validated_path_policy(base, regenerated) == regenerated
    unsafe = deepcopy(regenerated)
    unsafe["ordered_unit_paths"][0]["inode"] = 39002
    with pytest.raises(PermissionError, match="unit path changed"):
        recovery._validated_path_policy(base, unsafe)
    inserted = deepcopy(regenerated)
    inserted["ordered_unit_paths"].insert(
        1,
        {"path": "/unexpected/systemd/user", "exists": False},
    )
    with pytest.raises(PermissionError, match="ordered unit path set changed"):
        recovery._validated_path_policy(base, inserted)


def test_sealed_read_rejects_same_content_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    path = control / "sealed.json"
    recovery.io._write_sealed(
        path,
        {"schema_version": "r5-test-read-v1", "value": "exact"},
        fingerprint_field="test_fingerprint",
    )
    replacement = control / "replacement.json"
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o444)
    original_reader = recovery.io._read_fd_bytes
    replaced = False

    def replace_after_read(descriptor: int) -> bytes:
        nonlocal replaced
        data = original_reader(descriptor)
        if not replaced:
            replaced = True
            os.replace(replacement, path)
        return data

    monkeypatch.setattr(recovery.io, "_read_fd_bytes", replace_after_read)
    with pytest.raises(PermissionError, match="identity changed"):
        recovery._read_recovery_snapshot(
            path,
            fingerprint_field="test_fingerprint",
            schema="r5-test-read-v1",
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
    original_reader = recovery.io._read_fd_bytes
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
        recovery.io,
        "_read_fd_bytes",
        replace_during_readback,
    )
    with pytest.raises(PermissionError, match="identity"):
        recovery.io._write_sealed(
            path,
            {"schema_version": "r5-test-write-v1", "value": "exact"},
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
    original_reader = recovery.io._read_fd_bytes
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
        recovery.io,
        "_read_fd_bytes",
        replace_parent_during_readback,
    )
    with pytest.raises(PermissionError, match="parent changed"):
        recovery.io._write_sealed(
            path,
            {"schema_version": "r5-test-parent-v1", "value": "exact"},
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
    text = "[Unit]\nDescription=r5 exact recovery test\n"
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
    fragment.write_text("r5 recovery mock fragment\n", encoding="utf-8")
    fragment.chmod(0o600)
    monkeypatch.setattr(
        recovery,
        "RECOVERY_AUTHORIZATION_PATH",
        control / "r5-exact-recovery-authorization.json",
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_INTENT_PATH",
        control / "r5-exact-recovery-intent.json",
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_TERMINAL_PATH",
        control / "r5-exact-recovery-terminal.json",
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
        recovery.io,
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
        "remove-exact-r5-runtime-static-fragment",
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
        raise RuntimeError("synthetic r5 action-start failure")

    monkeypatch.setattr(
        recovery,
        "_remove_exact_fragment",
        fail_after_action_started,
    )
    with pytest.raises(RuntimeError, match="synthetic r5 action-start failure"):
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
    assert terminal["error_message"] == "synthetic r5 action-start failure"
    with pytest.raises(FileExistsError, match="consumed"):
        recovery.execute_recovery(execute=True)
