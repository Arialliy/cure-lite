from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests_v24 import test_cure_lite_v24_actual_unit_realization as fixtures
from tools import (
    cure_lite_v24_actual_unit_realization_preaccess_compat_c1 as compat,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service.template"
)
SUPERVISOR = (
    ROOT
    / "tools/"
    "cure_lite_v24_runtime_supervisor_preaccess_compat_c1.py"
)
SPEC_NAME = (
    "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json"
)


def _fake_bridge_observation(
    *,
    issued_delta: timedelta = timedelta(minutes=-1),
    expires_delta: timedelta = timedelta(minutes=4),
    root_overrides: dict[str, object] | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    now = datetime.now(timezone.utc)
    issued = now + issued_delta
    expires = now + expires_delta
    fingerprint = "a" * 64
    authorization = {
        "instruction_id": compat.legacy.INSTRUCTION_ID,
        "authorization_basis": compat.legacy.AUTHORIZATION_BASIS,
        "authorization_fingerprint": fingerprint,
        "created_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": expires.isoformat().replace("+00:00", "Z"),
    }
    root: dict[str, object] = {
        "path": str(compat.COMPAT_BRIDGE_AUTHORIZATION_PATH),
        "file_sha256": "b" * 64,
        "fingerprint": fingerprint,
        "device": 10,
        "inode": 20,
        "parent_size": 30,
        "parent_mtime_ns": 40,
        "parent_ctime_ns": 50,
    }
    if root_overrides:
        root.update(root_overrides)
    source = compat._stable_regular_read(
        compat.COMPAT_BRIDGE_SOURCE_PATH,
    )[1]
    return authorization, root, source


def _install_fake_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _fake_bridge_observation()
    monkeypatch.setattr(
        compat,
        "_validate_bridge_authorization",
        lambda **_kwargs: deepcopy(observation),
    )


def _compat_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fixtures, "realization", compat.legacy)
    monkeypatch.setattr(fixtures, "TEMPLATE", TEMPLATE)
    monkeypatch.setattr(fixtures, "SUPERVISOR", SUPERVISOR)


def _bind_test_lane(
    monkeypatch: pytest.MonkeyPatch,
    *,
    evidence: Path,
    unit_dir: Path,
    runtime_spec: Path,
) -> tuple[Path, Path, Path, Path]:
    authorization = evidence / "compat-authorization.json"
    receipt = evidence / "compat-receipt.json"
    terminal = evidence / "compat-terminal.json"
    protected = unit_dir / compat.PROTECTED_PREDECESSOR_UNIT
    monkeypatch.setattr(compat, "COMPAT_AUTHORIZATION_PATH", authorization)
    monkeypatch.setattr(compat, "COMPAT_RECEIPT_PATH", receipt)
    monkeypatch.setattr(compat, "COMPAT_TERMINAL_PATH", terminal)
    monkeypatch.setattr(compat, "COMPAT_RUNTIME_SPEC_PATH", runtime_spec)
    monkeypatch.setattr(compat, "COMPAT_UNIT_DIRECTORY", unit_dir)
    monkeypatch.setattr(
        compat,
        "PROTECTED_PREDECESSOR_FRAGMENT_PATH",
        protected,
    )
    return authorization, receipt, terminal, protected


def _authorize_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    bridge_validator=None,
) -> tuple[
    dict[str, object],
    Path,
    Path,
    Path,
    Path,
    fixtures.FakeRunner,
]:
    _compat_fixture(monkeypatch)
    if bridge_validator is None:
        _install_fake_bridge(monkeypatch)
    else:
        monkeypatch.setattr(
            compat,
            "_validate_bridge_authorization",
            bridge_validator,
        )
    evidence, unit_dir, alternate, generator_late, runtime_spec = (
        fixtures._workspace(tmp_path)
    )
    runtime_spec = runtime_spec.with_name(SPEC_NAME)
    authorization, receipt, terminal, protected = _bind_test_lane(
        monkeypatch,
        evidence=evidence,
        unit_dir=unit_dir,
        runtime_spec=runtime_spec,
    )
    protected.write_bytes(b"protected predecessor fragment\n")
    protected.chmod(0o600)
    runner = fixtures.FakeRunner(unit_dir, alternate, generator_late)
    runner.runtime_spec = runtime_spec
    payload = compat.create_authorization(
        authorization,
        template_path=TEMPLATE,
        python_path=fixtures.PYTHON,
        supervisor_path=SUPERVISOR,
        runtime_spec_path=runtime_spec,
        authorization_basis=compat.legacy.AUTHORIZATION_BASIS,
        instruction_id=compat.legacy.INSTRUCTION_ID,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
    )
    return (
        payload,
        authorization,
        receipt,
        terminal,
        protected,
        runner,
    )


def test_predecessor_hash_is_checked_before_any_predecessor_byte_executes(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    wrapper = tools / compat.COMPAT_REALIZER_PATH.name
    shutil.copy2(compat.COMPAT_REALIZER_PATH, wrapper)
    marker = tmp_path / "predecessor-executed"
    malicious = tools / compat.FROZEN_REALIZER_PATH.name
    malicious.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(wrapper), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "bound source SHA-256 changed" in completed.stderr
    assert not marker.exists()


def test_frozen_predecessor_same_bytes_inode_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor = tmp_path / "predecessor.py"
    predecessor.write_bytes(b"same frozen bytes\n")
    predecessor.chmod(0o644)
    raw, generation = compat._stable_regular_read(predecessor)
    monkeypatch.setattr(compat, "FROZEN_REALIZER_PATH", predecessor)
    monkeypatch.setattr(
        compat,
        "FROZEN_REALIZER_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "_FROZEN_REALIZER_LOAD_BINDING",
        generation,
    )
    predecessor.rename(tmp_path / "predecessor-old-generation.py")
    predecessor.write_bytes(raw)
    predecessor.chmod(0o644)

    with pytest.raises(PermissionError, match="generation was replaced"):
        compat._require_frozen_predecessor_generation()


def test_bridge_sha_is_checked_before_any_bridge_byte_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "bridge-executed"
    malicious = tmp_path / "bridge.py"
    malicious.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(compat, "COMPAT_BRIDGE_SOURCE_PATH", malicious)

    with pytest.raises(PermissionError, match="bound source SHA-256"):
        compat._validate_bridge_authorization(
            require_fresh=True,
            require_future_absence=True,
        )
    assert not marker.exists()


def test_compat_realizer_is_a_sealed_narrow_transitive_binding() -> None:
    identity = compat.verify_compatibility_identity()
    assert identity["scientific_attempt_ordinal"] == 2
    assert identity["runtime_compatibility_generation"] == "c1"
    assert identity["unit_name"] == compat.COMPAT_UNIT
    assert identity["runtime_spec_path"] == str(
        compat.COMPAT_RUNTIME_SPEC_PATH,
    )
    assert identity["frozen_realizer_path"] == str(
        compat.FROZEN_REALIZER_PATH,
    )
    assert identity["frozen_realizer_file_sha256"] == (
        compat.FROZEN_REALIZER_SHA256
    )
    assert identity["bridge_validator_file_sha256"] == (
        compat.COMPAT_BRIDGE_SOURCE_SHA256
    )
    assert identity["new_static_fragment_install_authorized"] is True
    assert identity["daemon_reload_authorized"] is True
    for field in (
        "predecessor_unit_mutation_authorized",
        "enable_authorized",
        "start_authorized",
        "stop_authorized",
        "remove_authorized",
        "automatic_retry_authorized",
        "D_R_payload_accessed",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
    ):
        assert identity[field] is False
    assert compat.legacy.ACTUAL_UNIT == compat.COMPAT_UNIT
    assert compat.legacy.validate_authorization is compat.validate_authorization
    assert compat.legacy.realize_actual_unit is compat.realize_actual_unit
    assert (
        compat.COMPATIBILITY_CLOSURE_KEY
        in compat.legacy._AUTH_KEYS
    )


def test_compat_fragment_accepts_only_live_explicit_zero_representation(
    tmp_path: Path,
) -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    rendered = compat.legacy.render_fragment(
        text,
        python_path=fixtures.PYTHON,
        supervisor_path=SUPERVISOR,
        runtime_spec_path=tmp_path / SPEC_NAME,
    )
    assert "SuccessExitStatus=0\n" in rendered
    expected = compat.legacy._expected_static_shadow(
        tmp_path / compat.COMPAT_UNIT,
    )
    assert expected["Id"] == compat.COMPAT_UNIT
    assert expected["SuccessExitStatus"] == "0"


@pytest.mark.parametrize(
    "changed",
    [
        "authorization",
        "template",
        "python",
        "supervisor",
        "runtime_spec",
    ],
)
def test_authorize_rejects_every_off_path_before_frozen_create(
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    values = {
        "authorization_path": compat.COMPAT_AUTHORIZATION_PATH,
        "template_path": compat.COMPAT_TEMPLATE_PATH,
        "python_path": Path("/usr/bin/python3.12"),
        "supervisor_path": compat.COMPAT_SUPERVISOR_PATH,
        "runtime_spec_path": compat.COMPAT_RUNTIME_SPEC_PATH,
    }
    key = (
        "authorization_path"
        if changed == "authorization"
        else f"{changed}_path"
    )
    values[key] = Path(f"/tmp/off-path-{changed}")
    monkeypatch.setattr(
        compat,
        "_frozen_create_authorization",
        lambda *_args, **_kwargs: pytest.fail(
            "frozen create reached after off-path input",
        ),
    )

    with pytest.raises(PermissionError, match="path changed"):
        compat.create_authorization(
            values["authorization_path"],
            template_path=values["template_path"],
            python_path=values["python_path"],
            supervisor_path=values["supervisor_path"],
            runtime_spec_path=values["runtime_spec_path"],
            authorization_basis=compat.legacy.AUTHORIZATION_BASIS,
            instruction_id=compat.legacy.INSTRUCTION_ID,
        )


@pytest.mark.parametrize(
    "changed",
    ["authorization", "receipt", "terminal"],
)
def test_apply_rejects_every_off_path_before_frozen_realize(
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    values = {
        "authorization_path": compat.COMPAT_AUTHORIZATION_PATH,
        "receipt_path": compat.COMPAT_RECEIPT_PATH,
        "terminal_path": compat.COMPAT_TERMINAL_PATH,
    }
    values[f"{changed}_path"] = Path(f"/tmp/off-path-{changed}")
    monkeypatch.setattr(
        compat,
        "_frozen_realize_actual_unit",
        lambda *_args, **_kwargs: pytest.fail(
            "frozen realization reached after off-path input",
        ),
    )

    with pytest.raises(PermissionError, match="path changed"):
        compat.realize_actual_unit(
            values["authorization_path"],
            receipt_path=values["receipt_path"],
            terminal_path=values["terminal_path"],
        )


def test_unit_authorization_cannot_exist_before_bridge_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_bridge(**_kwargs):
        raise FileNotFoundError("fixed bridge authorization is absent")

    with pytest.raises(
        FileNotFoundError,
        match="bridge authorization is absent",
    ):
        _authorize_fixture(
            monkeypatch,
            tmp_path,
            bridge_validator=missing_bridge,
        )
    assert not (
        tmp_path / "evidence/compat-authorization.json"
    ).exists()


def test_expired_bridge_authorization_cannot_cause_unit_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expired = _fake_bridge_observation(
        issued_delta=timedelta(minutes=-5),
        expires_delta=timedelta(minutes=-1),
    )

    with pytest.raises(
        PermissionError,
        match="outside bridge authorization window",
    ):
        _authorize_fixture(
            monkeypatch,
            tmp_path,
            bridge_validator=lambda **_kwargs: deepcopy(expired),
        )
    assert not (
        tmp_path / "evidence/compat-authorization.json"
    ).exists()


def test_off_path_bridge_authorization_root_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    off_path = _fake_bridge_observation(
        root_overrides={"path": "/tmp/off-path-bridge-authorization.json"},
    )

    with pytest.raises(PermissionError, match="off-path or malformed"):
        _authorize_fixture(
            monkeypatch,
            tmp_path,
            bridge_validator=lambda **_kwargs: deepcopy(off_path),
        )
    assert not (
        tmp_path / "evidence/compat-authorization.json"
    ).exists()


def test_authorization_and_receipt_bind_full_compatibility_closure_without(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor_before = compat._stable_regular_read(
        compat.FROZEN_REALIZER_PATH,
        expected_sha256=compat.FROZEN_REALIZER_SHA256,
    )[1]
    (
        authorization_payload,
        authorization,
        receipt,
        terminal,
        protected,
        runner,
    ) = _authorize_fixture(monkeypatch, tmp_path)
    closure = authorization_payload[compat.COMPATIBILITY_CLOSURE_KEY]
    assert closure["compat_source_generation"]["path"] == str(
        compat.COMPAT_REALIZER_PATH,
    )
    assert closure["frozen_predecessor_source_generation"] == (
        predecessor_before
    )
    assert closure["template_generation"]["file_sha256"] == (
        compat.COMPAT_TEMPLATE_SHA256
    )
    assert closure["bridge_compat_authorization_root"]["path"] == str(
        compat.COMPAT_BRIDGE_AUTHORIZATION_PATH,
    )
    assert closure["bridge_authorization_window"][
        "authorization_fingerprint"
    ] == "a" * 64
    assert closure["bridge_validator_source_generation"]["path"] == str(
        compat.COMPAT_BRIDGE_SOURCE_PATH,
    )
    assert closure["protected_predecessor_unit"]["fragment_path"] == str(
        protected,
    )
    assert closure["protected_predecessor_unit"][
        "mutation_authorized"
    ] is False
    assert closure["mutation_authority"] == compat._MUTATION_AUTHORITY

    realized = compat.realize_actual_unit(
        authorization,
        receipt_path=receipt,
        terminal_path=terminal,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
    )
    fragment = compat.COMPAT_UNIT_DIRECTORY / compat.COMPAT_UNIT
    assert realized["passed"] is True
    assert realized["started"] is False
    assert realized["enabled"] is False
    assert realized["removed"] is False
    assert realized["full_static_shadow"]["SuccessExitStatus"] == "0"
    assert realized["authorization_path"] == str(authorization)
    assert realized["authorization_fingerprint"] == (
        authorization_payload["authorization_fingerprint"]
    )
    assert fragment.is_file()
    assert protected.read_bytes() == b"protected predecessor fragment\n"
    assert not terminal.exists()
    assert compat._stable_regular_read(
        compat.FROZEN_REALIZER_PATH,
        expected_sha256=compat.FROZEN_REALIZER_SHA256,
    )[1] == predecessor_before
    predecessor_name = compat.PROTECTED_PREDECESSOR_UNIT
    assert all(
        predecessor_name not in command
        for command in runner.commands
    )
    mutating_systemctl = [
        command
        for command in runner.commands
        if command[0] == compat.legacy.SYSTEMCTL
        and len(command) >= 3
        and command[2] in {"start", "enable", "stop", "disable", "remove"}
    ]
    assert mutating_systemctl == []
    monkeypatch.setattr(
        compat,
        "_require_unit_authorization_fresh",
        lambda *_args, **_kwargs: pytest.fail(
            "archival validation required a live authorization window",
        ),
    )
    archival = compat.validate_archival_realization_chain(
        authorization,
        receipt,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
    )
    assert archival["receipt"]["passed"] is True


def test_same_bytes_bridge_authorization_generation_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        payload,
        authorization,
        _receipt,
        _terminal,
        _protected,
        runner,
    ) = _authorize_fixture(monkeypatch, tmp_path)
    closure = payload[compat.COMPATIBILITY_CLOSURE_KEY]
    window = closure["bridge_authorization_window"]
    bridge_authorization = {
        "instruction_id": compat.legacy.INSTRUCTION_ID,
        "authorization_basis": compat.legacy.AUTHORIZATION_BASIS,
        "authorization_fingerprint": window[
            "authorization_fingerprint"
        ],
        "created_at_utc": window["created_at_utc"],
        "issued_at_utc": window["issued_at_utc"],
        "expires_at_utc": window["expires_at_utc"],
    }
    replaced_root = deepcopy(
        closure["bridge_compat_authorization_root"],
    )
    replaced_root["inode"] = int(replaced_root["inode"]) + 1
    bridge_source = deepcopy(
        closure["bridge_validator_source_generation"],
    )
    monkeypatch.setattr(
        compat,
        "_validate_bridge_authorization",
        lambda **_kwargs: (
            deepcopy(bridge_authorization),
            deepcopy(replaced_root),
            deepcopy(bridge_source),
        ),
    )

    with pytest.raises(PermissionError, match="bridge/source"):
        compat.validate_authorization(
            authorization,
            runner=runner,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )


def test_archival_validator_rejects_self_consistent_minimal_fake_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        payload,
        authorization,
        receipt,
        _terminal,
        _protected,
        runner,
    ) = _authorize_fixture(monkeypatch, tmp_path)
    minimal = {
        "schema_version": compat.legacy.RECEIPT_SCHEMA,
        "unit_name": compat.COMPAT_UNIT,
        "authorization_path": str(authorization),
        "authorization_fingerprint": payload[
            "authorization_fingerprint"
        ],
        "passed": True,
    }
    compat._frozen_write_create_once_json(
        receipt,
        minimal,
        fingerprint_field="receipt_fingerprint",
    )

    with pytest.raises(
        PermissionError,
        match="receipt exact key/identity closure changed",
    ):
        compat.validate_archival_realization_chain(
            authorization,
            receipt,
            runner=runner,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )


def test_stale_unit_authorization_is_rejected_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        payload,
        _authorization,
        _receipt,
        _terminal,
        _protected,
        _runner,
    ) = _authorize_fixture(monkeypatch, tmp_path)
    stale = deepcopy(payload)
    now = datetime.now(timezone.utc)
    issued = (now - timedelta(minutes=2)).isoformat().replace(
        "+00:00",
        "Z",
    )
    stale["created_at_utc"] = issued
    stale["issued_at_utc"] = issued
    stale["expires_at_utc"] = (
        now - timedelta(minutes=1)
    ).isoformat().replace("+00:00", "Z")

    with pytest.raises(
        PermissionError,
        match="unit realization authorization is stale",
    ):
        compat._validate_compatibility_closure(stale)


def test_same_byte_protected_fragment_replacement_fails_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _payload,
        authorization,
        _receipt,
        _terminal,
        protected,
        runner,
    ) = _authorize_fixture(monkeypatch, tmp_path)
    raw = protected.read_bytes()
    protected.rename(protected.with_name("protected-old-generation.service"))
    protected.write_bytes(raw)
    protected.chmod(0o600)

    with pytest.raises(PermissionError, match="protected-unit generation"):
        compat.validate_authorization(
            authorization,
            runner=runner,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )
    assert not (
        compat.COMPAT_UNIT_DIRECTORY / compat.COMPAT_UNIT
    ).exists()
    assert (
        compat.legacy.SYSTEMCTL,
        "--user",
        "daemon-reload",
    ) not in runner.commands


def test_preinstall_terminal_permanently_blocks_same_authorization_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _payload,
        authorization,
        receipt,
        terminal,
        _protected,
        runner,
    ) = _authorize_fixture(monkeypatch, tmp_path)

    class PreinstallFailureRunner:
        def __init__(self, delegate: fixtures.FakeRunner) -> None:
            self.delegate = delegate
            self.commands: list[tuple[str, ...]] = []

        def __call__(self, argv: list[str], **kwargs: object):
            command = tuple(argv)
            self.commands.append(command)
            if command == compat.legacy._shadow_argv():
                return type("Completed", (), {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "forced preinstall read failure",
                })()
            return self.delegate(argv, **kwargs)

    failing = PreinstallFailureRunner(runner)
    with pytest.raises(RuntimeError, match="systemctl show failed"):
        compat.realize_actual_unit(
            authorization,
            receipt_path=receipt,
            terminal_path=terminal,
            runner=failing,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )
    assert terminal.is_file()
    assert not receipt.exists()
    assert not (
        compat.COMPAT_UNIT_DIRECTORY / compat.COMPAT_UNIT
    ).exists()
    assert (
        compat.legacy.SYSTEMCTL,
        "--user",
        "daemon-reload",
    ) not in failing.commands

    commands_before_replay = list(failing.commands)
    with pytest.raises(PermissionError, match="terminal or already"):
        compat.realize_actual_unit(
            authorization,
            receipt_path=receipt,
            terminal_path=terminal,
            runner=failing,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )
    assert failing.commands == commands_before_replay
    assert terminal.is_file()
    assert not receipt.exists()


def test_resealed_permission_expansion_is_rejected_by_closure_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        payload,
        _authorization,
        _receipt,
        _terminal,
        _protected,
        _runner,
    ) = _authorize_fixture(monkeypatch, tmp_path)
    drifted = deepcopy(payload)
    closure = drifted[compat.COMPATIBILITY_CLOSURE_KEY]
    closure["mutation_authority"]["start"] = True
    closure_body = dict(closure)
    closure_body.pop("closure_fingerprint")
    closure["closure_fingerprint"] = compat._fingerprint(closure_body)

    with pytest.raises(
        PermissionError,
        match="authorization semantics changed",
    ):
        compat._validate_compatibility_closure(drifted)
