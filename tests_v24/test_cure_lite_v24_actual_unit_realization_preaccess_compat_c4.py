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
    cure_lite_v24_actual_unit_realization_preaccess_compat_c4 as c4,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service.template"
)
BRIDGE = (
    ROOT / "tools/cure_lite_v24_preaccess_schema_compatibility_c4.py"
)
SUPERVISOR = (
    ROOT
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c4.py"
)
SPEC_NAME = (
    "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_spec.json"
)


@pytest.fixture
def frozen_bridge(monkeypatch: pytest.MonkeyPatch) -> str:
    digest = hashlib.sha256(BRIDGE.read_bytes()).hexdigest()
    monkeypatch.setattr(c4, "COMPAT_BRIDGE_SOURCE_SHA256", digest)
    c4._configure_c4_identity()
    return digest


def _loaded_state(unit: str, *, fragment: str = "") -> dict[str, str]:
    return {
        "Id": unit,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "NRestarts": "0",
        "FragmentPath": fragment,
        "InvocationID": "",
    }


def _missing_state(unit: str) -> dict[str, str]:
    return {
        "Id": unit,
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "",
        "NRestarts": "0",
        "FragmentPath": "",
        "InvocationID": "",
    }


def _authorized_unit_states() -> dict[str, object]:
    return {
        "protected_unit_states": {
            "old": _loaded_state(c4.PROTECTED_ORIGINAL_UNIT),
            "c1": _loaded_state(c4.PROTECTED_C1_UNIT),
            "c2": _missing_state(c4.PROTECTED_C2_UNIT),
            "c3": _loaded_state(
                c4.PROTECTED_C3_UNIT,
                fragment=str(c4.PROTECTED_C3_FRAGMENT_PATH),
            ),
        },
        "preauthorization_target_unit_state": _missing_state(
            c4.COMPAT_UNIT,
        ),
    }


def _fake_bridge_observation() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    now = datetime.now(timezone.utc)
    fingerprint = "a" * 64
    authorization = {
        "instruction_id": c4.compat_c1.legacy.INSTRUCTION_ID,
        "authorization_basis": c4.compat_c1.legacy.AUTHORIZATION_BASIS,
        "authorization_fingerprint": fingerprint,
        "created_at_utc": (now - timedelta(minutes=1))
        .isoformat()
        .replace("+00:00", "Z"),
        "issued_at_utc": (now - timedelta(minutes=1))
        .isoformat()
        .replace("+00:00", "Z"),
        "expires_at_utc": (now + timedelta(minutes=4))
        .isoformat()
        .replace("+00:00", "Z"),
        "schema_version": c4.BRIDGE_AUTHORIZATION_SCHEMA,
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_id": "c4",
        "authorized_uid": os.getuid(),
        "mutation_authority": deepcopy(
            c4.BRIDGE_MUTATION_AUTHORITY,
        ),
        "scientific_authority": deepcopy(
            c4.BRIDGE_SCIENTIFIC_AUTHORITY,
        ),
        **_authorized_unit_states(),
    }
    root = {
        "path": str(c4.COMPAT_BRIDGE_AUTHORIZATION_PATH),
        "file_sha256": "b" * 64,
        "fingerprint": fingerprint,
        "device": 10,
        "inode": 20,
        "mode": 0o444,
        "owner_uid": os.getuid(),
    }
    source = c4.compat_c1._stable_regular_read(
        c4.COMPAT_BRIDGE_SOURCE_PATH,
    )[1]
    return authorization, root, source


def _temporary_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    fixtures.FakeRunner,
]:
    monkeypatch.setattr(fixtures, "realization", c4.legacy)
    monkeypatch.setattr(fixtures, "TEMPLATE", TEMPLATE)
    monkeypatch.setattr(fixtures, "SUPERVISOR", SUPERVISOR)
    evidence, unit_dir, alternate, generator_late, runtime_spec = (
        fixtures._workspace(tmp_path)
    )
    runtime_spec = runtime_spec.with_name(SPEC_NAME)
    authorization = evidence / "r2_preaccess_compat_c4_unit_realization_authorization.json"
    receipt = evidence / "r2_preaccess_compat_c4_unit_realization_receipt.json"
    terminal = evidence / "r2_preaccess_compat_c4_unit_realization_terminal.json"
    old = unit_dir / c4.PROTECTED_ORIGINAL_UNIT
    c1_fragment = unit_dir / c4.PROTECTED_C1_UNIT
    c2_fragment = unit_dir / c4.PROTECTED_C2_UNIT
    c3_fragment = unit_dir / c4.PROTECTED_C3_UNIT
    old.write_bytes(b"protected original\n")
    c1_fragment.write_bytes(b"protected c1\n")
    c2_fragment.write_bytes(b"protected c2\n")
    c3_fragment.write_bytes(b"protected c3\n")
    monkeypatch.setattr(c4, "COMPAT_AUTHORIZATION_PATH", authorization)
    monkeypatch.setattr(c4, "COMPAT_RECEIPT_PATH", receipt)
    monkeypatch.setattr(c4, "COMPAT_TERMINAL_PATH", terminal)
    monkeypatch.setattr(c4, "COMPAT_RUNTIME_SPEC_PATH", runtime_spec)
    monkeypatch.setattr(c4, "COMPAT_UNIT_DIRECTORY", unit_dir)
    monkeypatch.setattr(c4, "PROTECTED_ORIGINAL_FRAGMENT_PATH", old)
    monkeypatch.setattr(c4, "PROTECTED_C1_FRAGMENT_PATH", c1_fragment)
    monkeypatch.setattr(c4, "PROTECTED_C2_FRAGMENT_PATH", c2_fragment)
    monkeypatch.setattr(c4, "PROTECTED_C3_FRAGMENT_PATH", c3_fragment)
    bridge_observation = _fake_bridge_observation()
    monkeypatch.setattr(
        c4,
        "_validate_c4_bridge_authorization",
        lambda **_kwargs: deepcopy(bridge_observation),
    )
    runner = fixtures.FakeRunner(unit_dir, alternate, generator_late)
    runner.runtime_spec = runtime_spec
    payload = c4.create_authorization(
        authorization,
        template_path=TEMPLATE,
        python_path=fixtures.PYTHON,
        supervisor_path=SUPERVISOR,
        runtime_spec_path=runtime_spec,
        authorization_basis=c4.compat_c1.legacy.AUTHORIZATION_BASIS,
        instruction_id=c4.compat_c1.legacy.INSTRUCTION_ID,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
    )
    return (
        payload,
        authorization,
        receipt,
        terminal,
        old,
        c1_fragment,
        c2_fragment,
        c3_fragment,
        runner,
    )


def test_c1_predecessor_hash_is_checked_before_any_byte_executes(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    wrapper = tools / c4.COMPAT_REALIZER_PATH.name
    shutil.copy2(c4.COMPAT_REALIZER_PATH, wrapper)
    marker = tmp_path / "predecessor-executed"
    malicious = tools / c4.FROZEN_REALIZER_PATH.name
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


def test_cold_help_is_c4_only_and_nonmutating() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(c4.COMPAT_REALIZER_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0
    normalized = " ".join(completed.stdout.split())
    assert "runtime compatibility generation c4" in normalized
    assert "original r2 and c1/c2/c3 units are immutable" in normalized
    assert "{authorize,apply}" in completed.stdout


def test_predecessor_same_bytes_generation_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor = tmp_path / "c1.py"
    predecessor.write_bytes(b"frozen c1 bytes\n")
    raw, generation = c4._stable_source_bytes(predecessor)
    monkeypatch.setattr(c4, "FROZEN_REALIZER_PATH", predecessor)
    monkeypatch.setattr(
        c4,
        "FROZEN_REALIZER_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(c4, "_C1_LOAD_GENERATION", generation)
    predecessor.rename(tmp_path / "old-c1.py")
    predecessor.write_bytes(raw)

    with pytest.raises(PermissionError, match="generation changed"):
        c4._require_source_generations()


def test_bridge_source_is_production_frozen() -> None:
    digest = hashlib.sha256(BRIDGE.read_bytes()).hexdigest()
    if c4.COMPAT_BRIDGE_SOURCE_SHA256 == "__TO_BE_FROZEN__":
        with pytest.raises(PermissionError, match="not frozen"):
            c4.verify_compatibility_identity()
        return
    assert c4.COMPAT_BRIDGE_SOURCE_SHA256 == digest
    assert c4.COMPAT_BRIDGE_SOURCE_SHA256 != "__TO_BE_FROZEN__"


def test_bridge_hash_is_checked_before_any_bridge_byte_executes(
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
    monkeypatch.setattr(c4, "COMPAT_BRIDGE_SOURCE_PATH", malicious)
    monkeypatch.setattr(c4, "COMPAT_BRIDGE_SOURCE_SHA256", "0" * 64)

    with pytest.raises(PermissionError, match="bound source SHA-256"):
        c4._validate_c4_bridge_authorization(
            require_fresh=True,
            require_future_absence=True,
        )
    assert not marker.exists()


def test_c4_identity_is_disjoint_and_transitively_binds_c1(
    frozen_bridge: str,
) -> None:
    identity = c4.verify_compatibility_identity()
    assert identity["scientific_attempt_ordinal"] == 2
    assert identity["runtime_compatibility_generation"] == "c4"
    assert identity["unit_name"] == c4.COMPAT_UNIT
    assert identity["frozen_realizer_path"] == str(
        c4.FROZEN_REALIZER_PATH,
    )
    assert identity["frozen_realizer_file_sha256"] == (
        "7bfc5944378d552f9f12654da5234762452f8dc5ee49f1bced47554bcbd58ece"
    )
    assert identity["bridge_validator_file_sha256"] == frozen_bridge
    assert c4.COMPAT_UNIT not in {
        c4.PROTECTED_ORIGINAL_UNIT,
        c4.PROTECTED_C1_UNIT,
        c4.PROTECTED_C2_UNIT,
        c4.PROTECTED_C3_UNIT,
    }
    assert len({c4.PROTECTED_ORIGINAL_UNIT, c4.PROTECTED_C1_UNIT,
                c4.PROTECTED_C2_UNIT, c4.PROTECTED_C3_UNIT,
                c4.COMPAT_UNIT}) == 5
    assert identity["protected_original_unit_mutation_authorized"] is False
    assert identity["protected_c1_unit_mutation_authorized"] is False
    assert identity["protected_c2_unit_mutation_authorized"] is False
    assert identity["protected_c3_unit_mutation_authorized"] is False
    assert identity["compat_bridge_file_sha256"] == (
        "ad660b7afe7ca87f690bc9565bd6674684c2b62824394751a39114a6efcf178a"
    )
    assert identity["compat_supervisor_file_sha256"] == (
        "faffe980cba4cad668a7d0f525bed8f2005950503d46f2b7c6888d79813c64ce"
    )
    assert identity["compat_template_file_sha256"] == (
        "5cc782eeeb4c29b9682e66b0413d8d0d938d3deacc4770a0f31e9711ce600dce"
    )
    assert identity["persistent_install_authorized"] is False
    for field in (
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


def test_all_c1_runtime_and_evidence_literals_are_rebound_to_c4(
    frozen_bridge: str,
) -> None:
    c4.verify_compatibility_identity()
    assert c4.compat_c1.COMPAT_UNIT == c4.COMPAT_UNIT
    assert c4.compat_c1.COMPAT_RUNTIME_SPEC_PATH == (
        c4.COMPAT_RUNTIME_SPEC_PATH
    )
    assert c4.compat_c1.COMPAT_AUTHORIZATION_PATH == (
        c4.COMPAT_AUTHORIZATION_PATH
    )
    assert c4.compat_c1.COMPAT_RECEIPT_PATH == c4.COMPAT_RECEIPT_PATH
    assert c4.compat_c1.COMPAT_TERMINAL_PATH == c4.COMPAT_TERMINAL_PATH
    assert c4.compat_c1.COMPAT_BRIDGE_AUTHORIZATION_PATH == (
        c4.COMPAT_BRIDGE_AUTHORIZATION_PATH
    )
    assert c4.compat_c1.legacy.ACTUAL_UNIT == c4.COMPAT_UNIT
    assert c4.compat_c1.legacy.__file__ == str(c4.COMPAT_REALIZER_PATH)


def test_template_is_exactly_frozen_and_c4_only() -> None:
    raw, binding = c4.compat_c1._stable_regular_read(
        TEMPLATE,
        expected_sha256=c4.COMPAT_TEMPLATE_SHA256,
    )
    text = raw.decode()
    assert binding["file_sha256"] == (
        "5cc782eeeb4c29b9682e66b0413d8d0d938d3deacc4770a0f31e9711ce600dce"
    )
    assert "compatibility c4 authorization" in text
    assert "preaccess_compat_c1" not in text
    assert "SuccessExitStatus=0\n" in text


def test_template_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = tmp_path / "changed.template"
    changed.write_bytes(TEMPLATE.read_bytes() + b"\n")
    monkeypatch.setattr(c4, "COMPAT_TEMPLATE_PATH", changed)
    c4._configure_c4_identity()
    with pytest.raises(PermissionError, match="SHA-256 changed"):
        c4.compat_c1._stable_regular_read(
            c4.compat_c1.COMPAT_TEMPLATE_PATH,
            expected_sha256=c4.compat_c1.COMPAT_TEMPLATE_SHA256,
        )


def test_old_c1_c2_and_c3_fragments_are_generation_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit_dir = tmp_path / "systemd-user"
    unit_dir.mkdir()
    old = unit_dir / c4.PROTECTED_ORIGINAL_UNIT
    c1_fragment = unit_dir / c4.PROTECTED_C1_UNIT
    c2_fragment = unit_dir / c4.PROTECTED_C2_UNIT
    c3_fragment = unit_dir / c4.PROTECTED_C3_UNIT
    old.write_bytes(b"old protected\n")
    c1_fragment.write_bytes(b"c1 protected changed\n")
    c2_fragment.write_bytes(b"c2 protected unchanged\n")
    c3_fragment.write_bytes(b"c3 protected unchanged\n")
    monkeypatch.setattr(c4, "COMPAT_UNIT_DIRECTORY", unit_dir)
    monkeypatch.setattr(c4, "PROTECTED_ORIGINAL_FRAGMENT_PATH", old)
    monkeypatch.setattr(c4, "PROTECTED_C1_FRAGMENT_PATH", c1_fragment)
    monkeypatch.setattr(c4, "PROTECTED_C2_FRAGMENT_PATH", c2_fragment)
    monkeypatch.setattr(c4, "PROTECTED_C3_FRAGMENT_PATH", c3_fragment)

    before = c4._protected_units()
    assert before["original_r2"]["mutation_authorized"] is False
    assert before["compat_c1"]["mutation_authorized"] is False
    assert before["compat_c2"]["mutation_authorized"] is False
    assert before["compat_c3"]["mutation_authorized"] is False
    assert before["original_r2"]["fragment_exists"] is True
    assert before["compat_c1"]["fragment_exists"] is True
    assert before["compat_c2"]["fragment_exists"] is True
    assert before["compat_c3"]["fragment_exists"] is True

    c1_fragment.unlink()
    c1_fragment.write_bytes(b"c1 protected\n")
    assert c4._protected_units() != before


def _base_c1_closure() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": (
            "cure-lite-v24-preaccess-compat-c1-"
            "unit-realization-closure-v1"
        ),
        "runtime_compatibility_generation": "c1",
        "fixed_paths": c4._fixed_paths(),
    }
    body["closure_fingerprint"] = c4.compat_c1._fingerprint(body)
    return body


def test_c4_closure_rewrites_generation_and_seals_all_protected_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = {
        "original_r2": {"mutation_authorized": False},
        "compat_c1": {"mutation_authorized": False},
        "compat_c2": {"mutation_authorized": False},
        "compat_c3": {"mutation_authorized": False},
    }
    monkeypatch.setattr(c4, "_C1_BUILD_CLOSURE", lambda _body: _base_c1_closure())
    monkeypatch.setattr(c4, "_protected_units", lambda: deepcopy(protected))
    closure = c4._build_c4_closure({})
    assert closure["schema_version"] == c4.COMPATIBILITY_CLOSURE_SCHEMA
    assert closure["runtime_compatibility_generation"] == "c4"
    assert closure[c4.PROTECTED_UNITS_KEY] == protected
    body = dict(closure)
    fingerprint = body.pop("closure_fingerprint")
    assert fingerprint == c4.compat_c1._fingerprint(body)


def test_c4_closure_rejects_protected_generation_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = {
        "original_r2": {"generation": 1},
        "compat_c1": {"generation": 1},
        "compat_c2": {"generation": 1},
        "compat_c3": {"generation": 1},
    }
    monkeypatch.setattr(c4, "_C1_BUILD_CLOSURE", lambda _body: _base_c1_closure())
    monkeypatch.setattr(c4, "_protected_units", lambda: deepcopy(protected))
    closure = c4._build_c4_closure({})
    authorization = {c4.COMPATIBILITY_CLOSURE_KEY: closure}
    monkeypatch.setattr(
        c4,
        "_protected_units",
        lambda: {
            "original_r2": {"generation": 1},
            "compat_c1": {"generation": 2},
            "compat_c2": {"generation": 1},
            "compat_c3": {"generation": 1},
        },
    )
    with pytest.raises(PermissionError, match="identity changed"):
        c4._validate_c4_closure(authorization)


def test_temporary_authorize_apply_and_archival_chain_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    (
        authorization_body,
        authorization,
        receipt,
        terminal,
        old,
        c1_fragment,
        c2_fragment,
        c3_fragment,
        runner,
    ) = _temporary_chain(monkeypatch, tmp_path)
    old_before = old.read_bytes()
    c1_before = c1_fragment.read_bytes()
    c2_before = c2_fragment.read_bytes()
    c3_before = c3_fragment.read_bytes()
    closure = authorization_body[c4.COMPATIBILITY_CLOSURE_KEY]
    assert closure["runtime_compatibility_generation"] == "c4"
    assert set(closure[c4.PROTECTED_UNITS_KEY]) == {
        "original_r2",
        "compat_c1",
        "compat_c2",
        "compat_c3",
    }

    result = c4.realize_actual_unit(
        authorization,
        receipt_path=receipt,
        terminal_path=terminal,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
    )
    assert result["passed"] is True
    assert result["unit_name"] == c4.COMPAT_UNIT
    assert receipt.exists()
    assert not terminal.exists()
    assert old.read_bytes() == old_before
    assert c1_fragment.read_bytes() == c1_before
    assert c2_fragment.read_bytes() == c2_before
    assert c3_fragment.read_bytes() == c3_before
    assert result["completed_actions"] == [
        "install-runtime-static-fragment",
        "daemon-reload",
        "verify-static-shadow",
    ]
    assert result["runtime_spec_absent_at_receipt"] is True
    assert {
        field: result["full_static_shadow"][field]
        for field in c4._RECEIPT_STATIC_STATE
    } == c4._RECEIPT_STATIC_STATE
    archival = c4.validate_archival_realization_chain(
        authorization,
        receipt,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
        allow_runtime_activation=False,
    )
    assert archival["compatibility_closure"][
        "runtime_compatibility_generation"
    ] == "c4"


def test_preinstall_failure_is_terminal_and_never_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    (
        _payload,
        authorization,
        receipt,
        terminal,
        _old,
        _c1_fragment,
        _c2_fragment,
        _c3_fragment,
        runner,
    ) = _temporary_chain(monkeypatch, tmp_path)

    class PreinstallFailureRunner:
        def __init__(self, delegate: fixtures.FakeRunner) -> None:
            self.delegate = delegate
            self.commands: list[tuple[str, ...]] = []

        def __call__(self, argv: list[str], **kwargs: object):
            command = tuple(argv)
            self.commands.append(command)
            if command == c4.compat_c1.legacy._shadow_argv():
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": 1,
                        "stdout": "",
                        "stderr": "forced preinstall read failure",
                    },
                )()
            return self.delegate(argv, **kwargs)

    failing = PreinstallFailureRunner(runner)
    with pytest.raises(RuntimeError, match="systemctl show failed"):
        c4.realize_actual_unit(
            authorization,
            receipt_path=receipt,
            terminal_path=terminal,
            runner=failing,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )
    assert terminal.is_file()
    assert not receipt.exists()
    assert not (c4.COMPAT_UNIT_DIRECTORY / c4.COMPAT_UNIT).exists()
    assert (
        c4.compat_c1.legacy.SYSTEMCTL,
        "--user",
        "daemon-reload",
    ) not in failing.commands

    commands_before_replay = list(failing.commands)
    with pytest.raises(PermissionError, match="terminal or already"):
        c4.realize_actual_unit(
            authorization,
            receipt_path=receipt,
            terminal_path=terminal,
            runner=failing,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )
    assert failing.commands == commands_before_replay


@pytest.mark.parametrize("protected_name", ("c1", "c2", "c3"))
def test_protected_generation_drift_after_authorization_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
    protected_name: str,
) -> None:
    (
        _payload,
        authorization,
        receipt,
        terminal,
        _old,
        c1_fragment,
        c2_fragment,
        c3_fragment,
        runner,
    ) = _temporary_chain(monkeypatch, tmp_path)
    protected = {
        "c1": c1_fragment,
        "c2": c2_fragment,
        "c3": c3_fragment,
    }[protected_name]
    protected.unlink()
    protected.write_bytes(f"protected {protected_name} drift\n".encode())

    with pytest.raises(
        PermissionError,
        match="compatibility closure identity changed",
    ):
        c4.realize_actual_unit(
            authorization,
            receipt_path=receipt,
            terminal_path=terminal,
            runner=runner,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )
    assert not receipt.exists()
    assert not (c4.COMPAT_UNIT_DIRECTORY / c4.COMPAT_UNIT).exists()


@pytest.mark.parametrize("consumed", ["authorization", "receipt", "terminal", "fragment"])
def test_terminal_or_replay_target_blocks_create_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
    consumed: str,
) -> None:
    evidence = tmp_path / "evidence"
    unit_dir = tmp_path / "units"
    evidence.mkdir()
    unit_dir.mkdir()
    paths = {
        "authorization": evidence / "r2_preaccess_compat_c4_unit_realization_authorization.json",
        "receipt": evidence / "r2_preaccess_compat_c4_unit_realization_receipt.json",
        "terminal": evidence / "r2_preaccess_compat_c4_unit_realization_terminal.json",
        "fragment": unit_dir / c4.COMPAT_UNIT,
    }
    paths[consumed].write_bytes(b"consumed\n")
    monkeypatch.setattr(c4, "COMPAT_AUTHORIZATION_PATH", paths["authorization"])
    monkeypatch.setattr(c4, "COMPAT_RECEIPT_PATH", paths["receipt"])
    monkeypatch.setattr(c4, "COMPAT_TERMINAL_PATH", paths["terminal"])
    monkeypatch.setattr(c4, "COMPAT_UNIT_DIRECTORY", unit_dir)
    monkeypatch.setattr(c4, "PROTECTED_ORIGINAL_FRAGMENT_PATH", unit_dir / c4.PROTECTED_ORIGINAL_UNIT)
    monkeypatch.setattr(c4, "PROTECTED_C1_FRAGMENT_PATH", unit_dir / c4.PROTECTED_C1_UNIT)
    monkeypatch.setattr(c4, "PROTECTED_C2_FRAGMENT_PATH", unit_dir / c4.PROTECTED_C2_UNIT)
    monkeypatch.setattr(c4, "PROTECTED_C3_FRAGMENT_PATH", unit_dir / c4.PROTECTED_C3_UNIT)
    c4._configure_c4_identity()

    with pytest.raises(PermissionError, match="already exists"):
        c4.compat_c1._require_create_targets_absent()


def test_static_shadow_targets_only_c4_and_never_authorizes_activation(
    tmp_path: Path,
    frozen_bridge: str,
) -> None:
    c4.verify_compatibility_identity()
    shadow = c4.compat_c1.legacy._expected_static_shadow(
        tmp_path / c4.COMPAT_UNIT,
    )
    assert shadow["Id"] == c4.COMPAT_UNIT
    assert shadow["UnitFileState"] == "static"
    assert shadow["ActiveState"] == "inactive"
    assert shadow["SubState"] == "dead"
    assert shadow["NRestarts"] == "0"
    assert shadow["Restart"] == "no"
    assert shadow["NeedDaemonReload"] == "no"
    assert shadow["InvocationID"] == ""


@pytest.mark.parametrize("allow_runtime_activation", [False, True])
def test_archival_validation_preserves_phase_aware_activation_flag(
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
    allow_runtime_activation: bool,
) -> None:
    observed: dict[str, object] = {}

    def fake(*args, **kwargs):
        observed.update(kwargs)
        return {"passed": True, "receipt": {}}

    monkeypatch.setattr(c4, "_C1_VALIDATE_ARCHIVAL_CHAIN", fake)
    monkeypatch.setattr(
        c4,
        "_validate_c4_receipt_contract",
        lambda receipt: dict(receipt),
    )
    result = c4.validate_archival_realization_chain(
        allow_runtime_activation=allow_runtime_activation,
    )
    assert result == {"passed": True, "receipt": {}}
    assert observed["allow_runtime_activation"] is allow_runtime_activation


def test_wrapper_exposes_no_start_enable_stop_remove_or_retry_command() -> None:
    parser = c4._parser()
    assert set(parser._subparsers._group_actions[0].choices) == {
        "authorize",
        "apply",
    }
    authority = c4.compat_c1._MUTATION_AUTHORITY
    for forbidden in ("enable", "start", "stop", "remove"):
        assert authority[forbidden] is False
    assert authority["protected_original_r2_unit_mutation"] is False
    assert authority["protected_c1_unit_mutation"] is False
    assert authority["protected_c2_unit_mutation"] is False
    assert authority["protected_c3_unit_mutation"] is False
    assert authority["c4_unit_realization_authorized"] is True
    assert c4.compat_c1.legacy._ACTIONS == list(c4._REALIZATION_ACTIONS)


def _install_real_tmp_bridge_policy(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: int,
) -> Path:
    raw, _source = c4.compat_c1._stable_regular_read(
        c4.COMPAT_BRIDGE_SOURCE_PATH,
        expected_sha256=c4.COMPAT_BRIDGE_SOURCE_SHA256,
    )
    policy = c4._load_bridge_policy_from_verified_bytes(raw)
    path = tmp_path / "c4-bridge-authorization.json"
    body = {
        "schema_version": c4.BRIDGE_AUTHORIZATION_SCHEMA,
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_id": "c4",
        "authorized_uid": os.getuid(),
        "mutation_authority": deepcopy(c4.BRIDGE_MUTATION_AUTHORITY),
        "scientific_authority": deepcopy(c4.BRIDGE_SCIENTIFIC_AUTHORITY),
        **_authorized_unit_states(),
    }
    policy._write_sealed(
        path,
        body,
        fingerprint_field="authorization_fingerprint",
    )
    path.chmod(mode)
    monkeypatch.setattr(c4, "COMPAT_BRIDGE_AUTHORIZATION_PATH", path)
    policy.COMPAT_AUTHORIZATION_PATH = path
    policy.COMPAT_UNIT_REALIZER_SOURCE_PATH = c4.COMPAT_REALIZER_PATH
    policy.COMPAT_UNIT_NAME = c4.COMPAT_UNIT
    policy.RUNTIME_COMPATIBILITY_ID = "c4"

    def validate(
        supplied: Path,
        *,
        require_fresh: bool,
        require_future_absence: bool,
    ):
        del require_fresh, require_future_absence
        assert supplied == path
        return policy._load_sealed(
            path,
            fingerprint_field="authorization_fingerprint",
            schema=c4.BRIDGE_AUTHORIZATION_SCHEMA,
        )

    policy.validate_compat_authorization = validate
    monkeypatch.setattr(
        c4,
        "_load_bridge_policy_from_verified_bytes",
        lambda _raw: policy,
    )
    return path


def test_real_tmp_b4_authorization_mode_0444_is_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    path = _install_real_tmp_bridge_policy(
        monkeypatch=monkeypatch, tmp_path=tmp_path, mode=0o444,
    )
    authorization, root, source = c4._validate_c4_bridge_authorization(
        require_fresh=True, require_future_absence=True,
    )
    assert path.parent == tmp_path
    assert root["mode"] == 0o444
    assert root["path"] == str(path)
    assert authorization["mutation_authority"] == c4.BRIDGE_MUTATION_AUTHORITY
    assert source["file_sha256"] == frozen_bridge

@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("target_loaded", "not-found/inert"),
        ("missing_c3", "protected unit-state closure"),
        ("c3_active", "static/inert"),
        ("c3_wrong_fragment", "static/inert"),
        ("extra_target_field", "not-found/inert"),
    ),
)
def test_b4_authorized_target_and_protected_states_are_exactly_consumed(
    case: str,
    message: str,
) -> None:
    authorization = _fake_bridge_observation()[0]
    if case == "target_loaded":
        authorization["preauthorization_target_unit_state"] = (
            _loaded_state(c4.COMPAT_UNIT)
        )
    elif case == "missing_c3":
        del authorization["protected_unit_states"]["c3"]
    elif case == "c3_active":
        authorization["protected_unit_states"]["c3"][
            "ActiveState"
        ] = "active"
    elif case == "c3_wrong_fragment":
        authorization["protected_unit_states"]["c3"][
            "FragmentPath"
        ] = "/wrong"
    else:
        authorization["preauthorization_target_unit_state"][
            "unexpected"
        ] = "drift"
    with pytest.raises(PermissionError, match=message):
        c4._validate_b4_authorized_unit_states(authorization)



@pytest.mark.parametrize("mode", (0o400, 0o440, 0o644, 0o446))
def test_real_tmp_b4_authorization_nonexact_mode_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
    mode: int,
) -> None:
    del frozen_bridge
    path = _install_real_tmp_bridge_policy(
        monkeypatch=monkeypatch, tmp_path=tmp_path, mode=mode,
    )
    assert path.parent == tmp_path
    with pytest.raises(PermissionError, match="unsafe sealed file"):
        c4._validate_c4_bridge_authorization(
            require_fresh=True, require_future_absence=True,
        )


def test_public_legacy_surface_has_no_base_write_bypass() -> None:
    assert "__getattr__" not in c4.__dict__
    for name in (
        "_frozen_create_authorization",
        "_frozen_realize_actual_unit",
        "_frozen_write_create_once_json",
        "_compat_write_create_once_json",
    ):
        with pytest.raises(AttributeError, match="bypass is unavailable"):
            getattr(c4.legacy, name)
    with pytest.raises(PermissionError, match="CLI entry point is disabled"):
        c4.legacy.main([])
    with pytest.raises(PermissionError, match="write helper is disabled"):
        c4.legacy.write_create_once_json(Path("unused"), {})


def test_legacy_write_entries_fail_closed_on_b4_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c4, "COMPAT_BRIDGE_SOURCE_SHA256", "__TO_BE_FROZEN__")
    for entry in (
        c4.legacy.create_authorization,
        c4.legacy.validate_authorization,
        c4.legacy.realize_actual_unit,
        c4.legacy.validate_archival_realization_chain,
    ):
        with pytest.raises(PermissionError, match="not frozen"):
            entry()
