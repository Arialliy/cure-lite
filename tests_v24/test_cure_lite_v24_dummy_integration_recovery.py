from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import cure_lite_v24_dummy_integration_recovery as recovery


def test_only_authorized_generator_late_inode_regeneration_is_accepted() -> None:
    base = {
        "systemd_path_argv": ["systemd-path"],
        "systemd_path_stdout": "/run/user/1008/systemd/user\n",
        "systemd_analyze_argv": ["systemd-analyze"],
        "systemd_analyze_stdout": "paths\n",
        "runtime_directory": "/run/user/1008/systemd/user",
        "runtime_directory_priority": 1,
        "runtime_directory_identity": {
            "path": "/run/user/1008/systemd/user",
            "exists": True,
            "device": 54,
            "inode": 10,
            "owner_uid": os.getuid(),
            "mode": 0o755,
        },
        "ordered_unit_paths": [
            {
                "path": "/run/user/1008/systemd/user",
                "exists": True,
                "device": 54,
                "inode": 10,
                "owner_uid": os.getuid(),
                "mode": 0o755,
            },
            {
                "path": f"/run/user/{os.getuid()}/systemd/generator.late",
                "exists": True,
                "device": 54,
                "inode": 20,
                "owner_uid": os.getuid(),
                "mode": 0o755,
            },
        ],
    }
    regenerated = deepcopy(base)
    regenerated["ordered_unit_paths"][1]["inode"] = 21
    assert recovery._validated_recovery_path_policy(
        base,
        regenerated,
    ) == regenerated

    unsafe = deepcopy(regenerated)
    unsafe["ordered_unit_paths"][0]["inode"] = 11
    with pytest.raises(PermissionError, match="path identity changed"):
        recovery._validated_recovery_path_policy(base, unsafe)


def test_archived_source_binding_is_structural_not_current() -> None:
    binding = {
        "path": str(
            recovery.REPOSITORY
            / "tools/cure_lite_v24_runtime_supervisor.py"
        ),
        "resolved_path": "/historical/resolved/path",
        "path_is_symlink": False,
        "file_sha256": "a" * 64,
        "device": 1,
        "inode": 2,
        "owner_uid": os.getuid(),
        "mode": 0o664,
    }
    assert recovery._validate_archived_file_binding(
        binding,
        name="supervisor",
        expected_path=binding["path"],
    ) == binding
    changed = dict(binding)
    changed["path"] = "/tmp/not-authorized"
    with pytest.raises(PermissionError, match="archived executable"):
        recovery._validate_archived_file_binding(
            changed,
            name="supervisor",
            expected_path=binding["path"],
        )


def test_two_phase_recovery_is_create_once_and_single_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    fragment = runtime / recovery.UNIT_NAME
    original_auth_path = control / "authorization.json"
    spec_path = control / "runtime-spec.json"
    terminal_path = control / "integration-terminal.json"
    removal_path = control / "removal-state.json"
    recovery_auth_path = control / "recovery-authorization.json"
    intent_path = control / "recovery-intent.json"
    recovery_terminal_path = control / "recovery-terminal.json"

    original = recovery.integration._write_sealed(
        original_auth_path,
        {"schema_version": "generated-original"},
        fingerprint_field="authorization_fingerprint",
    )
    spec = recovery.integration._write_sealed(
        spec_path,
        {"schema_version": "generated-spec"},
        fingerprint_field="runtime_spec_fingerprint",
    )
    terminal = recovery.integration._write_sealed(
        terminal_path,
        {"schema_version": "generated-terminal"},
        fingerprint_field="integration_terminal_fingerprint",
    )
    removal = recovery.integration._write_sealed(
        removal_path,
        {"schema_version": "generated-removal"},
        fingerprint_field="removal_state_fingerprint",
    )
    state = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "FragmentPath": str(fragment),
        "DropInPaths": "",
        "Transient": "no",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
    }
    fragment_identity = {
        "fragment_path": str(fragment),
        "fragment_sha256": "b" * 64,
        "device": 1,
        "inode": 2,
        "owner_uid": os.getuid(),
        "mode": 0o600,
        "nlink": 1,
    }
    context = {
        "authorization": original,
        "terminal": terminal,
        "removal_state": removal,
        "spec": spec,
        "spec_path": spec_path,
        "archived_executable_bindings": {"historical": True},
        "current_required_executable_bindings": {"current": True},
        "manager_generation": {"generated": "manager"},
        "unit_path_policy": {"generated": "policy"},
        "fragment_identity": fragment_identity,
        "inactive_static_state": state,
        "plan": SimpleNamespace(fragment_path=fragment),
    }
    monkeypatch.setattr(
        recovery,
        "ORIGINAL_AUTHORIZATION_PATH",
        original_auth_path,
    )
    monkeypatch.setattr(recovery, "ORIGINAL_TERMINAL_PATH", terminal_path)
    monkeypatch.setattr(
        recovery,
        "ORIGINAL_REMOVAL_STATE_PATH",
        removal_path,
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_AUTHORIZATION_PATH",
        recovery_auth_path,
    )
    monkeypatch.setattr(recovery, "RECOVERY_INTENT_PATH", intent_path)
    monkeypatch.setattr(
        recovery,
        "RECOVERY_TERMINAL_PATH",
        recovery_terminal_path,
    )
    monkeypatch.setattr(recovery, "_live_context", lambda **_kwargs: context)

    authorization = recovery.create_recovery_authorization(
        validity_seconds=120
    )
    assert authorization["remove_authorized"] is True
    assert authorization["start_authorized"] is False
    assert authorization["stop_authorized"] is False
    with pytest.raises(FileExistsError):
        recovery.create_recovery_authorization(validity_seconds=120)

    calls: list[str] = []

    def remove_with_intent(
        _plan: object,
        *,
        expected_identity: object,
        authorization: object,
        intent: object,
        on_action_started: object,
    ) -> None:
        assert recovery.RECOVERY_INTENT_PATH.exists()
        assert expected_identity == fragment_identity
        assert authorization["recovery_authorization_fingerprint"]
        assert intent["recovery_intent_fingerprint"]
        on_action_started(
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        calls.append("remove:True")

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
    result = recovery.execute_recovery(execute=True)
    assert result["passed"] is True
    assert result["fragment_absent"] is True
    assert calls == ["remove:True", "reload:True"]
    with pytest.raises(FileExistsError):
        recovery.execute_recovery(execute=True)


def test_exact_removal_rejects_authorized_inode_drift_and_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "systemd" / "user"
    runtime.mkdir(parents=True, mode=0o700)
    fragment = runtime / recovery.UNIT_NAME
    fragment_text = "[Unit]\nDescription=exact recovery test\n"
    fragment.write_text(fragment_text, encoding="utf-8")
    fragment.chmod(0o600)
    digest = hashlib.sha256(fragment_text.encode("utf-8")).hexdigest()
    plan = recovery.realizer.build_realization_plan(
        unit_name=recovery.UNIT_NAME,
        unit_directory=runtime,
        fragment_text=fragment_text,
        expected_fragment_sha256=digest,
        execute_authorized=True,
        removal_authorized=True,
    )
    observed = recovery.integration._observed_fragment_identity(fragment)
    now = datetime.now(timezone.utc)
    authorization = {
        "issued_at_utc": (now - timedelta(seconds=1)).isoformat(),
        "expires_at_utc": (now + timedelta(seconds=60)).isoformat(),
    }
    intent = {"created_at_utc": now.isoformat()}
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_PATH", fragment)
    monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_SHA256", digest)
    monkeypatch.setattr(
        recovery,
        "EXPECTED_FRAGMENT_DEVICE",
        observed["device"],
    )
    monkeypatch.setattr(
        recovery,
        "EXPECTED_FRAGMENT_INODE",
        observed["inode"],
    )

    drifted = dict(observed)
    drifted["inode"] = int(observed["inode"]) + 1
    with pytest.raises(PermissionError, match="removal identity"):
        recovery._remove_exact_authorized_fragment(
            plan,
            expected_identity=drifted,
            authorization=authorization,
            intent=intent,
            on_action_started=lambda _value: None,
        )
    assert fragment.exists()

    expired = {
        "issued_at_utc": (now - timedelta(seconds=2)).isoformat(),
        "expires_at_utc": (now - timedelta(seconds=1)).isoformat(),
    }
    with pytest.raises(PermissionError, match="expired before unlink"):
        recovery._remove_exact_authorized_fragment(
            plan,
            expected_identity=observed,
            authorization=expired,
            intent=intent,
            on_action_started=lambda _value: None,
        )
    assert fragment.exists()


def test_exact_removal_success_and_same_content_replacement_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    authorization = {
        "issued_at_utc": (now - timedelta(seconds=1)).isoformat(),
        "expires_at_utc": (now + timedelta(seconds=60)).isoformat(),
    }
    intent = {"created_at_utc": now.isoformat()}
    fragment_text = "[Unit]\nDescription=exact recovery test\n"
    digest = hashlib.sha256(fragment_text.encode("utf-8")).hexdigest()

    def prepare(name: str) -> tuple[Path, object, dict[str, object]]:
        runtime = tmp_path / name / "systemd" / "user"
        runtime.mkdir(parents=True, mode=0o700)
        fragment = runtime / recovery.UNIT_NAME
        fragment.write_text(fragment_text, encoding="utf-8")
        fragment.chmod(0o600)
        plan = recovery.realizer.build_realization_plan(
            unit_name=recovery.UNIT_NAME,
            unit_directory=runtime,
            fragment_text=fragment_text,
            expected_fragment_sha256=digest,
            execute_authorized=True,
            removal_authorized=True,
        )
        observed = recovery.integration._observed_fragment_identity(fragment)
        monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_PATH", fragment)
        monkeypatch.setattr(recovery, "EXPECTED_FRAGMENT_SHA256", digest)
        monkeypatch.setattr(
            recovery,
            "EXPECTED_FRAGMENT_DEVICE",
            observed["device"],
        )
        monkeypatch.setattr(
            recovery,
            "EXPECTED_FRAGMENT_INODE",
            observed["inode"],
        )
        return fragment, plan, observed

    fragment, plan, observed = prepare("success")
    action_trace: list[str] = []
    recovery._remove_exact_authorized_fragment(
        plan,
        expected_identity=observed,
        authorization=authorization,
        intent=intent,
        on_action_started=action_trace.append,
    )
    assert not fragment.exists()
    assert len(action_trace) == 1

    fragment, plan, observed = prepare("replacement")
    replacement = fragment.with_name("same-content-replacement.service")
    replacement.write_text(fragment_text, encoding="utf-8")
    replacement.chmod(0o600)
    replacement_identity = recovery.integration._observed_fragment_identity(
        replacement
    )
    assert replacement_identity["inode"] != observed["inode"]
    os.replace(replacement, fragment)
    action_trace = []
    with pytest.raises(PermissionError, match="inode changed"):
        recovery._remove_exact_authorized_fragment(
            plan,
            expected_identity=observed,
            authorization=authorization,
            intent=intent,
            on_action_started=action_trace.append,
        )
    assert fragment.exists()
    assert action_trace == []
