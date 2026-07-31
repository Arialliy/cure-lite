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
    cure_lite_v24_actual_unit_realization_preaccess_compat_c2 as c2,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service.template"
)
BRIDGE = (
    ROOT / "tools/cure_lite_v24_preaccess_schema_compatibility_c2.py"
)
SUPERVISOR = (
    ROOT
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c2.py"
)
SPEC_NAME = (
    "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_spec.json"
)


@pytest.fixture
def frozen_bridge() -> str:
    digest = hashlib.sha256(BRIDGE.read_bytes()).hexdigest()
    assert c2.COMPAT_BRIDGE_SOURCE_SHA256 == digest
    return digest


def _fake_bridge_observation() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    now = datetime.now(timezone.utc)
    fingerprint = "a" * 64
    authorization = {
        "instruction_id": c2.compat_c1.legacy.INSTRUCTION_ID,
        "authorization_basis": c2.compat_c1.legacy.AUTHORIZATION_BASIS,
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
    }
    root = {
        "path": str(c2.COMPAT_BRIDGE_AUTHORIZATION_PATH),
        "file_sha256": "b" * 64,
        "fingerprint": fingerprint,
        "device": 10,
        "inode": 20,
        "mode": 0o444,
    }
    source = c2.compat_c1._stable_regular_read(
        c2.COMPAT_BRIDGE_SOURCE_PATH,
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
    fixtures.FakeRunner,
]:
    monkeypatch.setattr(fixtures, "realization", c2.legacy)
    monkeypatch.setattr(fixtures, "TEMPLATE", TEMPLATE)
    monkeypatch.setattr(fixtures, "SUPERVISOR", SUPERVISOR)
    evidence, unit_dir, alternate, generator_late, runtime_spec = (
        fixtures._workspace(tmp_path)
    )
    runtime_spec = runtime_spec.with_name(SPEC_NAME)
    authorization = evidence / "c2-authorization.json"
    receipt = evidence / "c2-receipt.json"
    terminal = evidence / "c2-terminal.json"
    old = unit_dir / c2.PROTECTED_ORIGINAL_UNIT
    c1_fragment = unit_dir / c2.PROTECTED_C1_UNIT
    old.write_bytes(b"protected original\n")
    c1_fragment.write_bytes(b"protected c1\n")
    monkeypatch.setattr(c2, "COMPAT_AUTHORIZATION_PATH", authorization)
    monkeypatch.setattr(c2, "COMPAT_RECEIPT_PATH", receipt)
    monkeypatch.setattr(c2, "COMPAT_TERMINAL_PATH", terminal)
    monkeypatch.setattr(c2, "COMPAT_RUNTIME_SPEC_PATH", runtime_spec)
    monkeypatch.setattr(c2, "COMPAT_UNIT_DIRECTORY", unit_dir)
    monkeypatch.setattr(c2, "PROTECTED_ORIGINAL_FRAGMENT_PATH", old)
    monkeypatch.setattr(c2, "PROTECTED_C1_FRAGMENT_PATH", c1_fragment)
    bridge_observation = _fake_bridge_observation()
    monkeypatch.setattr(
        c2,
        "_validate_c2_bridge_authorization",
        lambda **_kwargs: deepcopy(bridge_observation),
    )
    runner = fixtures.FakeRunner(unit_dir, alternate, generator_late)
    runner.runtime_spec = runtime_spec
    payload = c2.create_authorization(
        authorization,
        template_path=TEMPLATE,
        python_path=fixtures.PYTHON,
        supervisor_path=SUPERVISOR,
        runtime_spec_path=runtime_spec,
        authorization_basis=c2.compat_c1.legacy.AUTHORIZATION_BASIS,
        instruction_id=c2.compat_c1.legacy.INSTRUCTION_ID,
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
        runner,
    )


def test_c1_predecessor_hash_is_checked_before_any_byte_executes(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    wrapper = tools / c2.COMPAT_REALIZER_PATH.name
    shutil.copy2(c2.COMPAT_REALIZER_PATH, wrapper)
    marker = tmp_path / "predecessor-executed"
    malicious = tools / c2.FROZEN_REALIZER_PATH.name
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


def test_predecessor_same_bytes_generation_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor = tmp_path / "c1.py"
    predecessor.write_bytes(b"frozen c1 bytes\n")
    raw, generation = c2._stable_source_bytes(predecessor)
    monkeypatch.setattr(c2, "FROZEN_REALIZER_PATH", predecessor)
    monkeypatch.setattr(
        c2,
        "FROZEN_REALIZER_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(c2, "_C1_LOAD_GENERATION", generation)
    predecessor.rename(tmp_path / "old-c1.py")
    predecessor.write_bytes(raw)

    with pytest.raises(PermissionError, match="generation changed"):
        c2._require_source_generations()


def test_bridge_source_is_production_frozen() -> None:
    digest = hashlib.sha256(BRIDGE.read_bytes()).hexdigest()
    assert c2.COMPAT_BRIDGE_SOURCE_SHA256 == digest
    assert c2.COMPAT_BRIDGE_SOURCE_SHA256 != "__TO_BE_FROZEN__"


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
    monkeypatch.setattr(c2, "COMPAT_BRIDGE_SOURCE_PATH", malicious)
    monkeypatch.setattr(c2, "COMPAT_BRIDGE_SOURCE_SHA256", "0" * 64)

    with pytest.raises(PermissionError, match="bound source SHA-256"):
        c2._validate_c2_bridge_authorization(
            require_fresh=True,
            require_future_absence=True,
        )
    assert not marker.exists()


def test_c2_identity_is_disjoint_and_transitively_binds_c1(
    frozen_bridge: str,
) -> None:
    identity = c2.verify_compatibility_identity()
    assert identity["scientific_attempt_ordinal"] == 2
    assert identity["runtime_compatibility_generation"] == "c2"
    assert identity["unit_name"] == c2.COMPAT_UNIT
    assert identity["frozen_realizer_path"] == str(
        c2.FROZEN_REALIZER_PATH,
    )
    assert identity["frozen_realizer_file_sha256"] == (
        "7bfc5944378d552f9f12654da5234762452f8dc5ee49f1bced47554bcbd58ece"
    )
    assert identity["bridge_validator_file_sha256"] == frozen_bridge
    assert c2.COMPAT_UNIT not in {
        c2.PROTECTED_ORIGINAL_UNIT,
        c2.PROTECTED_C1_UNIT,
    }
    assert c2.PROTECTED_ORIGINAL_UNIT != c2.PROTECTED_C1_UNIT
    assert identity["protected_original_unit_mutation_authorized"] is False
    assert identity["protected_c1_unit_mutation_authorized"] is False
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


def test_all_c1_runtime_and_evidence_literals_are_rebound_to_c2(
    frozen_bridge: str,
) -> None:
    c2.verify_compatibility_identity()
    assert c2.compat_c1.COMPAT_UNIT == c2.COMPAT_UNIT
    assert c2.compat_c1.COMPAT_RUNTIME_SPEC_PATH == (
        c2.COMPAT_RUNTIME_SPEC_PATH
    )
    assert c2.compat_c1.COMPAT_AUTHORIZATION_PATH == (
        c2.COMPAT_AUTHORIZATION_PATH
    )
    assert c2.compat_c1.COMPAT_RECEIPT_PATH == c2.COMPAT_RECEIPT_PATH
    assert c2.compat_c1.COMPAT_TERMINAL_PATH == c2.COMPAT_TERMINAL_PATH
    assert c2.compat_c1.COMPAT_BRIDGE_AUTHORIZATION_PATH == (
        c2.COMPAT_BRIDGE_AUTHORIZATION_PATH
    )
    assert c2.compat_c1.legacy.ACTUAL_UNIT == c2.COMPAT_UNIT
    assert c2.compat_c1.legacy.__file__ == str(c2.COMPAT_REALIZER_PATH)


def test_template_is_exactly_frozen_and_c2_only() -> None:
    raw, binding = c2.compat_c1._stable_regular_read(
        TEMPLATE,
        expected_sha256=c2.COMPAT_TEMPLATE_SHA256,
    )
    text = raw.decode()
    assert binding["file_sha256"] == (
        "4e485e5ba86a79b9244fb73d5add1a7015d71aa5f16f56479bd9c2c200d12967"
    )
    assert "compatibility c2 authorization" in text
    assert "preaccess_compat_c1" not in text
    assert "SuccessExitStatus=0\n" in text


def test_template_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = tmp_path / "changed.template"
    changed.write_bytes(TEMPLATE.read_bytes() + b"\n")
    monkeypatch.setattr(c2, "COMPAT_TEMPLATE_PATH", changed)
    c2._configure_c2_identity()
    with pytest.raises(PermissionError, match="SHA-256 changed"):
        c2.compat_c1._stable_regular_read(
            c2.compat_c1.COMPAT_TEMPLATE_PATH,
            expected_sha256=c2.compat_c1.COMPAT_TEMPLATE_SHA256,
        )


def test_old_and_c1_fragments_are_both_generation_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit_dir = tmp_path / "systemd-user"
    unit_dir.mkdir()
    old = unit_dir / c2.PROTECTED_ORIGINAL_UNIT
    c1_fragment = unit_dir / c2.PROTECTED_C1_UNIT
    old.write_bytes(b"old protected\n")
    c1_fragment.write_bytes(b"c1 protected changed\n")
    monkeypatch.setattr(c2, "COMPAT_UNIT_DIRECTORY", unit_dir)
    monkeypatch.setattr(c2, "PROTECTED_ORIGINAL_FRAGMENT_PATH", old)
    monkeypatch.setattr(c2, "PROTECTED_C1_FRAGMENT_PATH", c1_fragment)

    before = c2._protected_units()
    assert before["original_r2"]["mutation_authorized"] is False
    assert before["compat_c1"]["mutation_authorized"] is False
    assert before["original_r2"]["fragment_exists"] is True
    assert before["compat_c1"]["fragment_exists"] is True

    c1_fragment.unlink()
    c1_fragment.write_bytes(b"c1 protected\n")
    assert c2._protected_units() != before


def _base_c1_closure() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": (
            "cure-lite-v24-preaccess-compat-c1-"
            "unit-realization-closure-v1"
        ),
        "runtime_compatibility_generation": "c1",
        "fixed_paths": c2._fixed_paths(),
    }
    body["closure_fingerprint"] = c2.compat_c1._fingerprint(body)
    return body


def test_c2_closure_rewrites_generation_and_seals_both_protected_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = {
        "original_r2": {"mutation_authorized": False},
        "compat_c1": {"mutation_authorized": False},
    }
    monkeypatch.setattr(c2, "_C1_BUILD_CLOSURE", lambda _body: _base_c1_closure())
    monkeypatch.setattr(c2, "_protected_units", lambda: deepcopy(protected))
    closure = c2._build_c2_closure({})
    assert closure["schema_version"] == c2.COMPATIBILITY_CLOSURE_SCHEMA
    assert closure["runtime_compatibility_generation"] == "c2"
    assert closure[c2.PROTECTED_UNITS_KEY] == protected
    body = dict(closure)
    fingerprint = body.pop("closure_fingerprint")
    assert fingerprint == c2.compat_c1._fingerprint(body)


def test_c2_closure_rejects_protected_generation_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = {
        "original_r2": {"generation": 1},
        "compat_c1": {"generation": 1},
    }
    monkeypatch.setattr(c2, "_C1_BUILD_CLOSURE", lambda _body: _base_c1_closure())
    monkeypatch.setattr(c2, "_protected_units", lambda: deepcopy(protected))
    closure = c2._build_c2_closure({})
    authorization = {c2.COMPATIBILITY_CLOSURE_KEY: closure}
    monkeypatch.setattr(
        c2,
        "_protected_units",
        lambda: {
            "original_r2": {"generation": 1},
            "compat_c1": {"generation": 2},
        },
    )
    with pytest.raises(PermissionError, match="identity changed"):
        c2._validate_c2_closure(authorization)


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
        runner,
    ) = _temporary_chain(monkeypatch, tmp_path)
    old_before = old.read_bytes()
    c1_before = c1_fragment.read_bytes()
    closure = authorization_body[c2.COMPATIBILITY_CLOSURE_KEY]
    assert closure["runtime_compatibility_generation"] == "c2"
    assert set(closure[c2.PROTECTED_UNITS_KEY]) == {
        "original_r2",
        "compat_c1",
    }

    result = c2.realize_actual_unit(
        authorization,
        receipt_path=receipt,
        terminal_path=terminal,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
    )
    assert result["passed"] is True
    assert result["unit_name"] == c2.COMPAT_UNIT
    assert receipt.exists()
    assert not terminal.exists()
    assert old.read_bytes() == old_before
    assert c1_fragment.read_bytes() == c1_before
    archival = c2.validate_archival_realization_chain(
        authorization,
        receipt,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
        allow_runtime_activation=False,
    )
    assert archival["compatibility_closure"][
        "runtime_compatibility_generation"
    ] == "c2"


def test_protected_c1_generation_drift_after_authorization_is_rejected(
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
        c1_fragment,
        runner,
    ) = _temporary_chain(monkeypatch, tmp_path)
    c1_fragment.unlink()
    c1_fragment.write_bytes(b"protected c1\n")

    with pytest.raises(
        PermissionError,
        match="compatibility closure identity changed",
    ):
        c2.realize_actual_unit(
            authorization,
            receipt_path=receipt,
            terminal_path=terminal,
            runner=runner,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )
    assert not receipt.exists()
    assert not (c2.COMPAT_UNIT_DIRECTORY / c2.COMPAT_UNIT).exists()


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
        "authorization": evidence / "auth.json",
        "receipt": evidence / "receipt.json",
        "terminal": evidence / "terminal.json",
        "fragment": unit_dir / c2.COMPAT_UNIT,
    }
    paths[consumed].write_bytes(b"consumed\n")
    monkeypatch.setattr(c2, "COMPAT_AUTHORIZATION_PATH", paths["authorization"])
    monkeypatch.setattr(c2, "COMPAT_RECEIPT_PATH", paths["receipt"])
    monkeypatch.setattr(c2, "COMPAT_TERMINAL_PATH", paths["terminal"])
    monkeypatch.setattr(c2, "COMPAT_UNIT_DIRECTORY", unit_dir)
    c2._configure_c2_identity()

    with pytest.raises(PermissionError, match="already exists"):
        c2.compat_c1._require_create_targets_absent()


def test_static_shadow_targets_only_c2_and_never_authorizes_activation(
    tmp_path: Path,
    frozen_bridge: str,
) -> None:
    c2.verify_compatibility_identity()
    shadow = c2.compat_c1.legacy._expected_static_shadow(
        tmp_path / c2.COMPAT_UNIT,
    )
    assert shadow["Id"] == c2.COMPAT_UNIT
    assert shadow["UnitFileState"] == "static"
    assert shadow["ActiveState"] == "inactive"
    assert shadow["SubState"] == "dead"
    assert shadow["NRestarts"] == "0"


@pytest.mark.parametrize("allow_runtime_activation", [False, True])
def test_archival_validation_preserves_phase_aware_activation_flag(
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
    allow_runtime_activation: bool,
) -> None:
    observed: dict[str, object] = {}

    def fake(*args, **kwargs):
        observed.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(c2, "_C1_VALIDATE_ARCHIVAL_CHAIN", fake)
    result = c2.validate_archival_realization_chain(
        allow_runtime_activation=allow_runtime_activation,
    )
    assert result == {"passed": True}
    assert observed["allow_runtime_activation"] is allow_runtime_activation


def test_wrapper_exposes_no_start_enable_stop_remove_or_retry_command() -> None:
    parser = c2._parser()
    assert set(parser._subparsers._group_actions[0].choices) == {
        "authorize",
        "apply",
    }
    authority = c2.compat_c1._MUTATION_AUTHORITY
    for forbidden in ("enable", "start", "stop", "remove"):
        assert authority[forbidden] is False
    assert authority["protected_original_r2_unit_mutation"] is False
    assert authority["protected_c1_unit_mutation"] is False
