from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import FunctionType, ModuleType, SimpleNamespace

import pytest

from tests_v24 import test_cure_lite_v24_actual_unit_realization as fixtures
from tools import (
    cure_lite_v24_actual_unit_realization_preaccess_compat_c5 as c5,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c5.service.template"
)
BRIDGE = (
    ROOT / "tools/cure_lite_v24_preaccess_schema_compatibility_c5.py"
)
SUPERVISOR = (
    ROOT
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c5.py"
)
SPEC_NAME = (
    "D_R_structural_attempt_r2_preaccess_compat_c5_runtime_spec.json"
)


def _b5_user_authority() -> tuple[str, str]:
    from tools import (
        cure_lite_v24_preaccess_schema_compatibility_c5 as b5,
    )

    return b5.INSTRUCTION_ID, b5.AUTHORIZATION_BASIS


def test_r5_owns_one_utf8_canonical_profile_and_required_interface() -> None:
    value = {"authorization_basis": "user instruction: 修改后继续"}
    canonical = c5._canonical_json(value)
    assert "修改后继续" in canonical
    assert "\\u4fee" not in canonical
    assert c5._fingerprint(value) == hashlib.sha256(
        canonical.encode("utf-8"),
    ).hexdigest()
    assert c5.COMPAT_UNIT.endswith("preaccess-compat-c5.service")
    assert c5.COMPAT_BRIDGE_AUTHORIZATION_PATH.name.endswith(
        "schema_compat_c5_authorization.json"
    )
    assert c5.COMPAT_AUTHORIZATION_PATH.name.endswith(
        "compat_c5_unit_realization_authorization.json"
    )
    assert c5.COMPAT_RECEIPT_PATH.name.endswith(
        "compat_c5_unit_realization_receipt.json"
    )
    assert c5.COMPAT_TERMINAL_PATH.name.endswith(
        "compat_c5_unit_realization_terminal.json"
    )
    assert callable(c5.validate_archival_realization_chain)


def test_b5_verified_loader_accepts_the_exact_r5_producer_interface() -> None:
    from tools import (  # local import keeps this an explicit interop check
        cure_lite_v24_preaccess_schema_compatibility_c5 as b5,
    )

    expected = b5._source_root(b5.C5_UNIT_REALIZER_SOURCE_PATH)
    producer, observed = b5._load_verified_unit_realizer(expected)
    assert observed == expected
    assert producer.COMPAT_UNIT == b5.C5_UNIT_NAME
    assert Path(producer.COMPAT_BRIDGE_AUTHORIZATION_PATH) == (
        b5.C5_AUTHORIZATION_PATH
    )
    assert Path(producer.COMPAT_AUTHORIZATION_PATH) == (
        b5.C5_UNIT_AUTHORIZATION_PATH
    )
    assert Path(producer.COMPAT_RECEIPT_PATH) == b5.C5_UNIT_RECEIPT_PATH
    assert Path(producer.COMPAT_TERMINAL_PATH) == b5.C5_UNIT_TERMINAL_PATH
    assert producer.B5_INSTRUCTION_ID == b5.INSTRUCTION_ID
    assert producer.B5_AUTHORIZATION_BASIS == b5.AUTHORIZATION_BASIS
    assert producer.compat_c1.legacy.INSTRUCTION_ID == b5.INSTRUCTION_ID
    assert (
        producer.compat_c1.legacy.AUTHORIZATION_BASIS
        == b5.AUTHORIZATION_BASIS
    )
    assert callable(producer.validate_archival_realization_chain)


@pytest.fixture
def frozen_bridge() -> str:
    digest = hashlib.sha256(BRIDGE.read_bytes()).hexdigest()
    supervisor_digest = hashlib.sha256(SUPERVISOR.read_bytes()).hexdigest()
    assert c5.COMPAT_BRIDGE_SOURCE_SHA256 == digest
    assert c5.COMPAT_SUPERVISOR_SHA256 == supervisor_digest
    c5._configure_c5_identity()
    return digest


@pytest.fixture(autouse=True)
def restore_delegated_c5_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep same-process test doubles from leaking into the next test."""

    yield
    monkeypatch.undo()
    c5._configure_c5_identity()


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
            "old": _loaded_state(
                c5.PROTECTED_ORIGINAL_UNIT,
                fragment=str(c5.PROTECTED_ORIGINAL_FRAGMENT_PATH),
            ),
            "c1": _loaded_state(
                c5.PROTECTED_C1_UNIT,
                fragment=str(c5.PROTECTED_C1_FRAGMENT_PATH),
            ),
            "c2": _missing_state(c5.PROTECTED_C2_UNIT),
            "c3": _loaded_state(
                c5.PROTECTED_C3_UNIT,
                fragment=str(c5.PROTECTED_C3_FRAGMENT_PATH),
            ),
            "c4": _loaded_state(
                c5.PROTECTED_C4_UNIT,
                fragment=str(c5.PROTECTED_C4_FRAGMENT_PATH),
            ),
        },
        "preauthorization_target_unit_state": _missing_state(
            c5.COMPAT_UNIT,
        ),
    }


def _fake_bridge_observation() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    instruction_id, authorization_basis = _b5_user_authority()
    now = datetime.now(timezone.utc)
    fingerprint = "a" * 64
    authorization = {
        "instruction_id": instruction_id,
        "authorization_basis": authorization_basis,
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
        "schema_version": c5.BRIDGE_AUTHORIZATION_SCHEMA,
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_id": "c5",
        "authorized_uid": os.getuid(),
        "mutation_authority": deepcopy(
            c5.BRIDGE_MUTATION_AUTHORITY,
        ),
        "scientific_authority": deepcopy(
            c5.BRIDGE_SCIENTIFIC_AUTHORITY,
        ),
        **_authorized_unit_states(),
    }
    root = {
        "path": str(c5.COMPAT_BRIDGE_AUTHORIZATION_PATH),
        "file_sha256": "b" * 64,
        "fingerprint": fingerprint,
        "device": 10,
        "inode": 20,
        "mode": 0o444,
        "owner_uid": os.getuid(),
        "size": 4096,
    }
    source = c5.compat_c1._stable_regular_read(
        c5.COMPAT_BRIDGE_SOURCE_PATH,
    )[1]
    return authorization, root, source


def _temporary_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    real_b5_authorization: bool = False,
    runner_probe: object | None = None,
    manager_probe: object | None = None,
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
    monkeypatch.setattr(fixtures, "realization", c5.legacy)
    monkeypatch.setattr(fixtures, "TEMPLATE", TEMPLATE)
    monkeypatch.setattr(fixtures, "SUPERVISOR", SUPERVISOR)
    evidence, unit_dir, alternate, generator_late, runtime_spec = (
        fixtures._workspace(tmp_path)
    )
    runtime_spec = runtime_spec.with_name(SPEC_NAME)
    authorization = evidence / "r2_preaccess_compat_c5_unit_realization_authorization.json"
    receipt = evidence / "r2_preaccess_compat_c5_unit_realization_receipt.json"
    terminal = evidence / "r2_preaccess_compat_c5_unit_realization_terminal.json"
    old = unit_dir / c5.PROTECTED_ORIGINAL_UNIT
    c1_fragment = unit_dir / c5.PROTECTED_C1_UNIT
    c2_fragment = unit_dir / c5.PROTECTED_C2_UNIT
    c3_fragment = unit_dir / c5.PROTECTED_C3_UNIT
    c4_fragment = unit_dir / c5.PROTECTED_C4_UNIT
    old.write_bytes(b"protected original\n")
    c1_fragment.write_bytes(b"protected c1\n")
    c2_fragment.write_bytes(b"protected c2\n")
    c3_fragment.write_bytes(b"protected c3\n")
    c4_fragment.write_bytes(b"protected c4\n")
    monkeypatch.setattr(c5, "COMPAT_AUTHORIZATION_PATH", authorization)
    monkeypatch.setattr(c5, "COMPAT_RECEIPT_PATH", receipt)
    monkeypatch.setattr(c5, "COMPAT_TERMINAL_PATH", terminal)
    monkeypatch.setattr(c5, "COMPAT_RUNTIME_SPEC_PATH", runtime_spec)
    monkeypatch.setattr(c5, "COMPAT_UNIT_DIRECTORY", unit_dir)
    monkeypatch.setattr(c5, "PROTECTED_ORIGINAL_FRAGMENT_PATH", old)
    monkeypatch.setattr(c5, "PROTECTED_C1_FRAGMENT_PATH", c1_fragment)
    monkeypatch.setattr(c5, "PROTECTED_C2_FRAGMENT_PATH", c2_fragment)
    monkeypatch.setattr(c5, "PROTECTED_C3_FRAGMENT_PATH", c3_fragment)
    monkeypatch.setattr(c5, "PROTECTED_C4_FRAGMENT_PATH", c4_fragment)
    monkeypatch.setattr(c5, "_require_canonical_c5_identity", lambda: None)
    # The isolated fixture intentionally replaces delegated callables and paths.
    # Production entry points retain both identity sentries.
    monkeypatch.setattr(
        c5,
        "_require_transitive_callable_identities",
        lambda: None,
    )
    monkeypatch.setattr(c5, "_require_c1_alias_identities", lambda: None)
    if real_b5_authorization:
        _install_real_tmp_bridge_policy(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            mode=0o444,
        )
    else:
        bridge_observation = _fake_bridge_observation()
        monkeypatch.setattr(
            c5,
            "_validate_c5_bridge_authorization",
            lambda **_kwargs: deepcopy(bridge_observation),
        )
    c5._configure_c5_identity()
    runner = fixtures.FakeRunner(unit_dir, alternate, generator_late)
    runner.runtime_spec = runtime_spec
    create_runner = runner
    if runner_probe is not None:
        def create_runner(argv: list[str], **kwargs: object):
            runner_probe()
            return runner(argv, **kwargs)

    def manager_reader():
        if manager_probe is not None:
            manager_probe()
        return deepcopy(fixtures._manager())

    instruction_id, authorization_basis = _b5_user_authority()
    payload = c5.create_authorization(
        authorization,
        template_path=TEMPLATE,
        python_path=fixtures.PYTHON,
        supervisor_path=SUPERVISOR,
        runtime_spec_path=runtime_spec,
        authorization_basis=authorization_basis,
        instruction_id=instruction_id,
        runner=create_runner,
        manager_reader=manager_reader,
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
    wrapper = tools / c5.COMPAT_REALIZER_PATH.name
    shutil.copy2(c5.COMPAT_REALIZER_PATH, wrapper)
    marker = tmp_path / "predecessor-executed"
    malicious = tools / c5.FROZEN_REALIZER_PATH.name
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


def test_cold_help_is_c5_only_and_nonmutating() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(c5.COMPAT_REALIZER_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0
    normalized = " ".join(completed.stdout.split())
    assert "runtime compatibility generation c5" in normalized
    assert "original r2 and c1/c2/c3/c4 units are immutable" in normalized
    assert "{authorize,apply}" in completed.stdout


def test_predecessor_same_bytes_generation_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        c5,
        "COMPAT_BRIDGE_SOURCE_SHA256",
        hashlib.sha256(BRIDGE.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        c5,
        "COMPAT_SUPERVISOR_SHA256",
        hashlib.sha256(SUPERVISOR.read_bytes()).hexdigest(),
    )
    predecessor = tmp_path / "c1.py"
    predecessor.write_bytes(b"frozen c1 bytes\n")
    raw, generation = c5._stable_source_bytes(predecessor)
    monkeypatch.setattr(c5, "FROZEN_REALIZER_PATH", predecessor)
    monkeypatch.setattr(
        c5,
        "FROZEN_REALIZER_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(c5, "_C1_LOAD_GENERATION", generation)
    predecessor.rename(tmp_path / "old-c1.py")
    predecessor.write_bytes(raw)

    with pytest.raises(PermissionError, match="generation changed"):
        c5._require_source_generations()


def test_bridge_source_is_production_frozen() -> None:
    digest = hashlib.sha256(BRIDGE.read_bytes()).hexdigest()
    assert c5.COMPAT_BRIDGE_SOURCE_SHA256 == digest
    assert c5.COMPAT_BRIDGE_SOURCE_SHA256 != "__TO_BE_FROZEN__"


def test_supervisor_source_is_production_frozen() -> None:
    digest = hashlib.sha256(SUPERVISOR.read_bytes()).hexdigest()
    assert c5.COMPAT_SUPERVISOR_SHA256 == digest
    assert c5.COMPAT_SUPERVISOR_SHA256 != "__TO_BE_FROZEN__"


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
    monkeypatch.setattr(c5, "COMPAT_BRIDGE_SOURCE_PATH", malicious)
    monkeypatch.setattr(c5, "COMPAT_BRIDGE_SOURCE_SHA256", "0" * 64)
    monkeypatch.setattr(
        c5,
        "COMPAT_SUPERVISOR_SHA256",
        hashlib.sha256(SUPERVISOR.read_bytes()).hexdigest(),
    )

    with pytest.raises(PermissionError, match="bound source SHA-256"):
        c5._validate_c5_bridge_authorization(
            require_fresh=True,
            require_future_absence=True,
        )
    assert not marker.exists()


def test_c5_identity_is_disjoint_and_transitively_binds_c1(
    frozen_bridge: str,
) -> None:
    instruction_id, authorization_basis = _b5_user_authority()
    identity = c5.verify_compatibility_identity()
    assert identity["scientific_attempt_ordinal"] == 2
    assert identity["runtime_compatibility_generation"] == "c5"
    assert identity["unit_name"] == c5.COMPAT_UNIT
    assert identity["frozen_realizer_path"] == str(
        c5.FROZEN_REALIZER_PATH,
    )
    assert identity["frozen_realizer_file_sha256"] == (
        "7bfc5944378d552f9f12654da5234762452f8dc5ee49f1bced47554bcbd58ece"
    )
    assert identity["bridge_validator_file_sha256"] == frozen_bridge
    assert c5.COMPAT_UNIT not in {
        c5.PROTECTED_ORIGINAL_UNIT,
        c5.PROTECTED_C1_UNIT,
        c5.PROTECTED_C2_UNIT,
        c5.PROTECTED_C3_UNIT,
        c5.PROTECTED_C4_UNIT,
    }
    assert len({c5.PROTECTED_ORIGINAL_UNIT, c5.PROTECTED_C1_UNIT,
                c5.PROTECTED_C2_UNIT, c5.PROTECTED_C3_UNIT,
                c5.PROTECTED_C4_UNIT, c5.COMPAT_UNIT}) == 6
    assert identity["protected_original_unit_mutation_authorized"] is False
    assert identity["protected_c1_unit_mutation_authorized"] is False
    assert identity["protected_c2_unit_mutation_authorized"] is False
    assert identity["protected_c3_unit_mutation_authorized"] is False
    assert identity["protected_c4_unit_mutation_authorized"] is False
    assert identity["compat_bridge_file_sha256"] == frozen_bridge
    assert identity["compat_supervisor_file_sha256"] == hashlib.sha256(
        SUPERVISOR.read_bytes(),
    ).hexdigest()
    assert identity["compat_template_file_sha256"] == (
        "f2a3da0862addb90e61301c97e0d5c1d109e8cbf59ad86c2e5130235f8387216"
    )
    assert identity["instruction_id"] == instruction_id
    assert identity["authorization_basis"] == authorization_basis
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


def test_all_c1_runtime_and_evidence_literals_are_rebound_to_c5(
    frozen_bridge: str,
) -> None:
    instruction_id, authorization_basis = _b5_user_authority()
    c5.verify_compatibility_identity()
    assert c5.compat_c1.COMPAT_UNIT == c5.COMPAT_UNIT
    assert c5.compat_c1.COMPAT_RUNTIME_SPEC_PATH == (
        c5.COMPAT_RUNTIME_SPEC_PATH
    )
    assert c5.compat_c1.COMPAT_AUTHORIZATION_PATH == (
        c5.COMPAT_AUTHORIZATION_PATH
    )
    assert c5.compat_c1.COMPAT_RECEIPT_PATH == c5.COMPAT_RECEIPT_PATH
    assert c5.compat_c1.COMPAT_TERMINAL_PATH == c5.COMPAT_TERMINAL_PATH
    assert c5.compat_c1.COMPAT_BRIDGE_AUTHORIZATION_PATH == (
        c5.COMPAT_BRIDGE_AUTHORIZATION_PATH
    )
    assert c5.compat_c1.legacy.ACTUAL_UNIT == c5.COMPAT_UNIT
    assert c5.compat_c1.legacy.__file__ == str(c5.COMPAT_REALIZER_PATH)
    assert c5.compat_c1.legacy.INSTRUCTION_ID == instruction_id
    assert c5.compat_c1.legacy.AUTHORIZATION_BASIS == authorization_basis


@pytest.mark.parametrize(
    ("field", "drift"),
    (
        ("INSTRUCTION_ID", "user-2026-07-30-modify-then-run-v1"),
        ("AUTHORIZATION_BASIS", "user instruction: 修改后运行"),
    ),
)
def test_b5_user_authority_drift_fails_before_c1_delegation(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    drift: str,
) -> None:
    raw, _source = c5._stable_source_bytes(
        c5.COMPAT_BRIDGE_SOURCE_PATH,
        expected_sha256=c5.COMPAT_BRIDGE_SOURCE_SHA256,
    )
    policy = c5._load_bridge_policy_from_verified_bytes(raw)
    setattr(policy, field, drift)
    monkeypatch.setattr(
        c5,
        "_load_bridge_policy_from_verified_bytes",
        lambda _raw: policy,
    )
    with pytest.raises(PermissionError, match="user-authority contract"):
        c5.verify_compatibility_identity()


def test_template_is_exactly_frozen_and_c5_only() -> None:
    raw, binding = c5.compat_c1._stable_regular_read(
        TEMPLATE,
        expected_sha256=c5.COMPAT_TEMPLATE_SHA256,
    )
    text = raw.decode()
    assert binding["file_sha256"] == (
        "f2a3da0862addb90e61301c97e0d5c1d109e8cbf59ad86c2e5130235f8387216"
    )
    assert "compatibility c5 authorization" in text
    assert "preaccess_compat_c1" not in text
    assert "SuccessExitStatus=0\n" in text
    # ``fresh attempt`` is the inherited static systemd Description only.
    # Sealed scientific authority remains the same r2 attempt and forbids a
    # fresh scientific attempt, retry, resume, payload access, and training.
    assert "D_R structural fresh attempt r2" in text
    assert c5.BRIDGE_SCIENTIFIC_AUTHORITY["fresh_scientific_attempt"] is False
    assert c5.BRIDGE_SCIENTIFIC_AUTHORITY["automatic_retry"] is False
    assert c5.BRIDGE_SCIENTIFIC_AUTHORITY["resume"] is False
    assert c5.BRIDGE_SCIENTIFIC_AUTHORITY["training_authorized"] is False


def test_template_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = tmp_path / "changed.template"
    changed.write_bytes(TEMPLATE.read_bytes() + b"\n")
    monkeypatch.setattr(c5, "COMPAT_TEMPLATE_PATH", changed)
    with pytest.raises(PermissionError, match="canonical path/unit"):
        c5._configure_c5_identity()


def test_old_c1_c2_c3_and_c4_fragments_are_generation_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit_dir = tmp_path / "systemd-user"
    unit_dir.mkdir()
    old = unit_dir / c5.PROTECTED_ORIGINAL_UNIT
    c1_fragment = unit_dir / c5.PROTECTED_C1_UNIT
    c2_fragment = unit_dir / c5.PROTECTED_C2_UNIT
    c3_fragment = unit_dir / c5.PROTECTED_C3_UNIT
    c4_fragment = unit_dir / c5.PROTECTED_C4_UNIT
    old.write_bytes(b"old protected\n")
    c1_fragment.write_bytes(b"c1 protected changed\n")
    c2_fragment.write_bytes(b"c2 protected unchanged\n")
    c3_fragment.write_bytes(b"c3 protected unchanged\n")
    c4_fragment.write_bytes(b"c4 protected unchanged\n")
    monkeypatch.setattr(c5, "COMPAT_UNIT_DIRECTORY", unit_dir)
    monkeypatch.setattr(c5, "PROTECTED_ORIGINAL_FRAGMENT_PATH", old)
    monkeypatch.setattr(c5, "PROTECTED_C1_FRAGMENT_PATH", c1_fragment)
    monkeypatch.setattr(c5, "PROTECTED_C2_FRAGMENT_PATH", c2_fragment)
    monkeypatch.setattr(c5, "PROTECTED_C3_FRAGMENT_PATH", c3_fragment)
    monkeypatch.setattr(c5, "PROTECTED_C4_FRAGMENT_PATH", c4_fragment)

    before = c5._protected_units()
    assert before["original_r2"]["mutation_authorized"] is False
    assert before["compat_c1"]["mutation_authorized"] is False
    assert before["compat_c2"]["mutation_authorized"] is False
    assert before["compat_c3"]["mutation_authorized"] is False
    assert before["compat_c4"]["mutation_authorized"] is False
    assert before["original_r2"]["fragment_exists"] is True
    assert before["compat_c1"]["fragment_exists"] is True
    assert before["compat_c2"]["fragment_exists"] is True
    assert before["compat_c3"]["fragment_exists"] is True
    assert before["compat_c4"]["fragment_exists"] is True

    c1_fragment.unlink()
    c1_fragment.write_bytes(b"c1 protected\n")
    assert c5._protected_units() != before


def _base_c1_closure() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": (
            "cure-lite-v24-preaccess-compat-c1-"
            "unit-realization-closure-v1"
        ),
        "runtime_compatibility_generation": "c1",
        "fixed_paths": c5._fixed_paths(),
    }
    body["closure_fingerprint"] = c5.compat_c1._fingerprint(body)
    return body


def test_c5_closure_rewrites_generation_and_seals_all_protected_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = {
        "original_r2": {"mutation_authorized": False},
        "compat_c1": {"mutation_authorized": False},
        "compat_c2": {"mutation_authorized": False},
        "compat_c3": {"mutation_authorized": False},
        "compat_c4": {"mutation_authorized": False},
    }
    monkeypatch.setattr(c5, "_C1_BUILD_CLOSURE", lambda _body: _base_c1_closure())
    monkeypatch.setattr(c5, "_require_c1_alias_identities", lambda: None)
    monkeypatch.setattr(c5, "_require_transitive_callable_identities", lambda: None)
    monkeypatch.setattr(c5, "_protected_units", lambda: deepcopy(protected))
    closure = c5._build_c5_closure({})
    assert closure["schema_version"] == c5.COMPATIBILITY_CLOSURE_SCHEMA
    assert closure["runtime_compatibility_generation"] == "c5"
    assert closure[c5.PROTECTED_UNITS_KEY] == protected
    body = dict(closure)
    fingerprint = body.pop("closure_fingerprint")
    assert fingerprint == c5.compat_c1._fingerprint(body)


def test_c5_closure_rejects_protected_generation_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = {
        "original_r2": {"generation": 1},
        "compat_c1": {"generation": 1},
        "compat_c2": {"generation": 1},
        "compat_c3": {"generation": 1},
        "compat_c4": {"generation": 1},
    }
    monkeypatch.setattr(c5, "_C1_BUILD_CLOSURE", lambda _body: _base_c1_closure())
    monkeypatch.setattr(c5, "_require_c1_alias_identities", lambda: None)
    monkeypatch.setattr(c5, "_require_transitive_callable_identities", lambda: None)
    monkeypatch.setattr(c5, "_protected_units", lambda: deepcopy(protected))
    closure = c5._build_c5_closure({})
    authorization = {c5.COMPATIBILITY_CLOSURE_KEY: closure}
    monkeypatch.setattr(
        c5,
        "_protected_units",
        lambda: {
            "original_r2": {"generation": 1},
            "compat_c1": {"generation": 2},
            "compat_c2": {"generation": 1},
            "compat_c3": {"generation": 1},
            "compat_c4": {"generation": 1},
        },
    )
    with pytest.raises(PermissionError, match="identity changed"):
        c5._validate_c5_closure(authorization)


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
    c4_fragment = old.parent / c5.PROTECTED_C4_UNIT
    c4_before = c4_fragment.read_bytes()
    closure = authorization_body[c5.COMPATIBILITY_CLOSURE_KEY]
    assert closure["runtime_compatibility_generation"] == "c5"
    assert set(closure[c5.PROTECTED_UNITS_KEY]) == {
        "original_r2",
        "compat_c1",
        "compat_c2",
        "compat_c3",
        "compat_c4",
    }

    result = c5.realize_actual_unit(
        authorization,
        receipt_path=receipt,
        terminal_path=terminal,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
    )
    assert result["passed"] is True
    assert result["unit_name"] == c5.COMPAT_UNIT
    assert receipt.exists()
    assert not terminal.exists()
    assert old.read_bytes() == old_before
    assert c1_fragment.read_bytes() == c1_before
    assert c2_fragment.read_bytes() == c2_before
    assert c3_fragment.read_bytes() == c3_before
    assert c4_fragment.read_bytes() == c4_before
    assert result["completed_actions"] == [
        "install-runtime-static-fragment",
        "daemon-reload",
        "verify-static-shadow",
    ]
    assert result["runtime_spec_absent_at_receipt"] is True
    assert {
        field: result["full_static_shadow"][field]
        for field in c5._RECEIPT_STATIC_STATE
    } == c5._RECEIPT_STATIC_STATE
    archival = c5.validate_archival_realization_chain(
        authorization,
        receipt,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
        allow_runtime_activation=False,
    )
    assert archival["compatibility_closure"][
        "runtime_compatibility_generation"
    ] == "c5"


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
            if command == c5.compat_c1.legacy._shadow_argv():
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
        c5.realize_actual_unit(
            authorization,
            receipt_path=receipt,
            terminal_path=terminal,
            runner=failing,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )
    assert terminal.is_file()
    assert not receipt.exists()
    assert not (c5.COMPAT_UNIT_DIRECTORY / c5.COMPAT_UNIT).exists()
    assert (
        c5.compat_c1.legacy.SYSTEMCTL,
        "--user",
        "daemon-reload",
    ) not in failing.commands

    commands_before_replay = list(failing.commands)
    with pytest.raises(PermissionError, match="terminal or already"):
        c5.realize_actual_unit(
            authorization,
            receipt_path=receipt,
            terminal_path=terminal,
            runner=failing,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )
    assert failing.commands == commands_before_replay


@pytest.mark.parametrize("protected_name", ("c1", "c2", "c3", "c4"))
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
        "c4": c1_fragment.parent / c5.PROTECTED_C4_UNIT,
    }[protected_name]
    protected.unlink()
    protected.write_bytes(f"protected {protected_name} drift\n".encode())

    with pytest.raises(
        PermissionError,
        match="compatibility closure identity changed",
    ):
        c5.realize_actual_unit(
            authorization,
            receipt_path=receipt,
            terminal_path=terminal,
            runner=runner,
            manager_reader=lambda: deepcopy(fixtures._manager()),
        )
    assert not receipt.exists()
    assert not (c5.COMPAT_UNIT_DIRECTORY / c5.COMPAT_UNIT).exists()


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
        "authorization": evidence / "r2_preaccess_compat_c5_unit_realization_authorization.json",
        "receipt": evidence / "r2_preaccess_compat_c5_unit_realization_receipt.json",
        "terminal": evidence / "r2_preaccess_compat_c5_unit_realization_terminal.json",
        "fragment": unit_dir / c5.COMPAT_UNIT,
    }
    paths[consumed].write_bytes(b"consumed\n")
    monkeypatch.setattr(c5, "COMPAT_AUTHORIZATION_PATH", paths["authorization"])
    monkeypatch.setattr(c5, "COMPAT_RECEIPT_PATH", paths["receipt"])
    monkeypatch.setattr(c5, "COMPAT_TERMINAL_PATH", paths["terminal"])
    monkeypatch.setattr(c5, "COMPAT_UNIT_DIRECTORY", unit_dir)
    monkeypatch.setattr(c5, "PROTECTED_ORIGINAL_FRAGMENT_PATH", unit_dir / c5.PROTECTED_ORIGINAL_UNIT)
    monkeypatch.setattr(c5, "PROTECTED_C1_FRAGMENT_PATH", unit_dir / c5.PROTECTED_C1_UNIT)
    monkeypatch.setattr(c5, "PROTECTED_C2_FRAGMENT_PATH", unit_dir / c5.PROTECTED_C2_UNIT)
    monkeypatch.setattr(c5, "PROTECTED_C3_FRAGMENT_PATH", unit_dir / c5.PROTECTED_C3_UNIT)
    monkeypatch.setattr(c5, "PROTECTED_C4_FRAGMENT_PATH", unit_dir / c5.PROTECTED_C4_UNIT)
    monkeypatch.setattr(c5, "_require_canonical_c5_identity", lambda: None)
    c5._configure_c5_identity()

    with pytest.raises(PermissionError, match="already exists"):
        c5.compat_c1._require_create_targets_absent()


def test_static_shadow_targets_only_c5_and_never_authorizes_activation(
    tmp_path: Path,
    frozen_bridge: str,
) -> None:
    c5.verify_compatibility_identity()
    shadow = c5.compat_c1.legacy._expected_static_shadow(
        tmp_path / c5.COMPAT_UNIT,
    )
    assert shadow["Id"] == c5.COMPAT_UNIT
    assert shadow["UnitFileState"] == "static"
    assert shadow["ActiveState"] == "inactive"
    assert shadow["SubState"] == "dead"
    assert shadow["NRestarts"] == "0"
    assert shadow["Restart"] == "no"
    assert shadow["NeedDaemonReload"] == "no"
    assert shadow["InvocationID"] == ""


def _runtime_shadow_runner(
    delegate: fixtures.FakeRunner,
    overrides: dict[str, str],
):
    def run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        completed = delegate(argv, **kwargs)
        if tuple(argv) != c5.compat_c1.legacy._shadow_argv():
            return completed
        values = c5.compat_c1.legacy._parse_show(completed.stdout)
        values.update(overrides)
        stdout = "".join(
            f"{name}={values[name]}\n"
            for name in c5.compat_c1.legacy._SHADOW_PROPERTIES
        )
        return SimpleNamespace(
            returncode=completed.returncode,
            stdout=stdout,
            stderr=completed.stderr,
        )

    return run


def test_runtime_dynamic_shadow_contract_is_exact_and_identity_guarded(
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    del frozen_bridge
    identity = c5.verify_compatibility_identity()
    assert identity["runtime_dynamic_shadow_fields"] == [
        "ActiveState",
        "InvocationID",
        "SubState",
    ]
    assert frozenset(c5.compat_c1._RUNTIME_DYNAMIC_SHADOW_FIELDS) == (
        frozenset({"ActiveState", "SubState", "InvocationID"})
    )
    monkeypatch.setattr(
        c5._compat_c1,
        "_RUNTIME_DYNAMIC_SHADOW_FIELDS",
        {"ActiveState", "SubState", "InvocationID", "NRestarts"},
    )
    with pytest.raises(PermissionError, match="contract changed"):
        c5.verify_compatibility_identity()
    monkeypatch.setattr(
        c5._compat_c1,
        "_RUNTIME_DYNAMIC_SHADOW_FIELDS",
        {"ActiveState", "SubState", "InvocationID"},
    )
    monkeypatch.setattr(
        c5,
        "_R5_RUNTIME_DYNAMIC_SHADOW_FIELDS",
        frozenset(
            {"ActiveState", "SubState", "InvocationID", "NRestarts"}
        ),
    )
    with pytest.raises(PermissionError, match="contract changed"):
        c5.verify_compatibility_identity()


def test_archival_active_phase_allows_only_three_dynamic_shadow_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    del frozen_bridge
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
    c5.realize_actual_unit(
        authorization,
        receipt_path=receipt,
        terminal_path=terminal,
        runner=runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
    )
    active = {
        "ActiveState": "active",
        "SubState": "running",
        "InvocationID": "0123456789abcdef0123456789abcdef",
    }
    active_runner = _runtime_shadow_runner(runner, active)
    with pytest.raises(PermissionError):
        c5.validate_archival_realization_chain(
            authorization,
            receipt,
            runner=active_runner,
            manager_reader=lambda: deepcopy(fixtures._manager()),
            allow_runtime_activation=False,
        )
    archival = c5.validate_archival_realization_chain(
        authorization,
        receipt,
        runner=active_runner,
        manager_reader=lambda: deepcopy(fixtures._manager()),
        runtime_phase=c5.RUNTIME_PHASE_RUN_ONCE,
    )
    assert archival["receipt"]["passed"] is True

    for field, value in (
        ("NRestarts", "1"),
        ("FragmentPath", "/wrong/c5.service"),
        ("Restart", "always"),
        ("NeedDaemonReload", "yes"),
    ):
        drifted_runner = _runtime_shadow_runner(
            runner,
            {**active, field: value},
        )
        with pytest.raises(
            PermissionError,
            match="runtime immutable shadow changed",
        ):
            c5.validate_archival_realization_chain(
                authorization,
                receipt,
                runner=drifted_runner,
                manager_reader=lambda: deepcopy(fixtures._manager()),
                runtime_phase=c5.RUNTIME_PHASE_RUN_ONCE,
            )


@pytest.mark.parametrize(
    ("runtime_phase", "allow_runtime_activation"),
    (
        (c5.RUNTIME_PHASE_PREACTIVATION, False),
        (c5.RUNTIME_PHASE_RUN_ONCE, True),
    ),
)
def test_archival_validation_preserves_phase_aware_activation_flag(
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
    runtime_phase: str,
    allow_runtime_activation: bool,
) -> None:
    observed: dict[str, object] = {}

    def fake(*args, **kwargs):
        observed.update(kwargs)
        observed["context_phase"] = c5._RUNTIME_PHASE_CONTEXT.get()
        return {"passed": True, "receipt": {}}

    monkeypatch.setattr(c5, "_C1_VALIDATE_ARCHIVAL_CHAIN", fake)
    monkeypatch.setattr(c5, "_require_c1_alias_identities", lambda: None)
    monkeypatch.setattr(c5, "_require_transitive_callable_identities", lambda: None)
    monkeypatch.setattr(
        c5,
        "_validate_c5_receipt_contract",
        lambda receipt: dict(receipt),
    )
    result = c5.validate_archival_realization_chain(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    assert result == {"passed": True, "receipt": {}}
    assert observed["allow_runtime_activation"] is allow_runtime_activation
    assert "runtime_phase" not in observed
    assert observed["context_phase"] == runtime_phase
    assert c5._RUNTIME_PHASE_CONTEXT.get() == (
        c5.RUNTIME_PHASE_PREACTIVATION
    )


@pytest.mark.parametrize(
    ("allow_runtime_activation", "runtime_phase", "message"),
    (
        (True, None, "requires an explicit phase"),
        (False, c5.RUNTIME_PHASE_RUN_ONCE, "disagree"),
        (True, c5.RUNTIME_PHASE_PREACTIVATION, "disagree"),
        (False, "unknown", "exact closed state"),
    ),
)
def test_archival_runtime_phase_is_closed_and_flag_consistent(
    allow_runtime_activation: bool,
    runtime_phase: str | None,
    message: str,
) -> None:
    with pytest.raises(PermissionError, match=message):
        c5.validate_archival_realization_chain(
            allow_runtime_activation=allow_runtime_activation,
            runtime_phase=runtime_phase,
        )


def test_fresh_validation_forces_and_resets_preactivation_context(
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    del frozen_bridge
    observed: list[tuple[str, str]] = []

    def record(name: str, result: object):
        def call(*_args: object, **_kwargs: object):
            observed.append((name, c5._RUNTIME_PHASE_CONTEXT.get()))
            return result

        return call

    monkeypatch.setattr(
        c5, "_C1_VALIDATE_AUTHORIZATION", record("validate", {}),
    )
    monkeypatch.setattr(c5, "_require_c1_alias_identities", lambda: None)
    monkeypatch.setattr(c5, "_require_transitive_callable_identities", lambda: None)
    outer = c5._RUNTIME_PHASE_CONTEXT.set(c5.RUNTIME_PHASE_RUN_ONCE)
    try:
        c5.validate_authorization()
        assert c5._RUNTIME_PHASE_CONTEXT.get() == c5.RUNTIME_PHASE_RUN_ONCE
    finally:
        c5._RUNTIME_PHASE_CONTEXT.reset(outer)
    assert observed == [("validate", c5.RUNTIME_PHASE_PREACTIVATION)]


def test_archival_runtime_phase_context_resets_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    del frozen_bridge

    def fail(*_args: object, **_kwargs: object):
        assert c5._RUNTIME_PHASE_CONTEXT.get() == c5.RUNTIME_PHASE_RUN_ONCE
        raise PermissionError("forced archival failure")

    monkeypatch.setattr(c5, "_C1_VALIDATE_ARCHIVAL_CHAIN", fail)
    monkeypatch.setattr(c5, "_require_c1_alias_identities", lambda: None)
    monkeypatch.setattr(c5, "_require_transitive_callable_identities", lambda: None)
    with pytest.raises(PermissionError, match="forced archival failure"):
        c5.validate_archival_realization_chain(
            runtime_phase=c5.RUNTIME_PHASE_RUN_ONCE,
        )
    assert c5._RUNTIME_PHASE_CONTEXT.get() == (
        c5.RUNTIME_PHASE_PREACTIVATION
    )


def test_wrapper_exposes_no_start_enable_stop_remove_or_retry_command() -> None:
    parser = c5._parser()
    assert set(parser._subparsers._group_actions[0].choices) == {
        "authorize",
        "apply",
    }
    authority = c5.compat_c1._MUTATION_AUTHORITY
    for forbidden in ("enable", "start", "stop", "remove"):
        assert authority[forbidden] is False
    assert authority["protected_original_r2_unit_mutation"] is False
    assert authority["protected_c1_unit_mutation"] is False
    assert authority["protected_c2_unit_mutation"] is False
    assert authority["protected_c3_unit_mutation"] is False
    assert authority["protected_c4_unit_mutation"] is False
    assert authority["c5_unit_realization_authorized"] is True
    assert c5.compat_c1.legacy._ACTIONS == list(c5._REALIZATION_ACTIONS)


def _install_real_tmp_bridge_policy(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: int,
) -> tuple[Path, object]:
    raw, _source = c5.compat_c1._stable_regular_read(
        c5.COMPAT_BRIDGE_SOURCE_PATH,
        expected_sha256=c5.COMPAT_BRIDGE_SOURCE_SHA256,
    )
    policy = c5._load_bridge_policy_from_verified_bytes(raw)
    path = tmp_path / "r2_preaccess_schema_compat_c5_authorization.json"
    current = datetime.now(timezone.utc)
    body = {
        "schema_version": c5.BRIDGE_AUTHORIZATION_SCHEMA,
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_id": "c5",
        "instruction_id": policy.INSTRUCTION_ID,
        "authorization_basis": policy.AUTHORIZATION_BASIS,
        "created_at_utc": policy._format_utc(
            current - timedelta(minutes=1),
        ),
        "issued_at_utc": policy._format_utc(
            current - timedelta(minutes=2),
        ),
        "expires_at_utc": policy._format_utc(
            current + timedelta(minutes=4),
        ),
        "authorized_uid": os.getuid(),
        "mutation_authority": deepcopy(c5.BRIDGE_MUTATION_AUTHORITY),
        "scientific_authority": deepcopy(c5.BRIDGE_SCIENTIFIC_AUTHORITY),
        **_authorized_unit_states(),
    }
    policy._write_sealed(
        path,
        body,
        fingerprint_field="authorization_fingerprint",
    )
    path.chmod(mode)
    monkeypatch.setattr(c5, "COMPAT_BRIDGE_AUTHORIZATION_PATH", path)
    policy.COMPAT_AUTHORIZATION_PATH = path
    policy.COMPAT_UNIT_REALIZER_SOURCE_PATH = c5.COMPAT_REALIZER_PATH
    policy.COMPAT_UNIT_NAME = c5.COMPAT_UNIT
    policy.RUNTIME_COMPATIBILITY_ID = "c5"

    phase_calls: list[tuple[bool, str]] = []

    def validate_c5_authorization(
        path: Path | None = None,
        *,
        unit_state_reader: object = None,
        require_fresh: bool,
        allow_runtime_activation: bool,
        runtime_phase: str | None,
        now: object = None,
    ):
        del unit_state_reader, require_fresh, now
        assert path == policy.COMPAT_AUTHORIZATION_PATH
        assert runtime_phase is not None
        phase_calls.append((allow_runtime_activation, runtime_phase))
        assert allow_runtime_activation is (
            runtime_phase != c5.RUNTIME_PHASE_PREACTIVATION
        )
        return policy._load_sealed(
            policy.COMPAT_AUTHORIZATION_PATH,
            fingerprint_field="authorization_fingerprint",
            schema=c5.BRIDGE_AUTHORIZATION_SCHEMA,
        )

    def validate_scientific_output_phase(
        *,
        allow_runtime_activation: bool,
        runtime_phase: str,
    ) -> None:
        assert policy._resolve_runtime_phase(
            allow_runtime_activation=allow_runtime_activation,
            runtime_phase=runtime_phase,
        ) == runtime_phase

    # Keep the real B5 public producer adapter and UTF-8 sealed loader.  Only
    # its large predecessor/current-state closure is replaced by this isolated
    # sealed fixture; this catches public signature and phase-contract drift.
    policy.validate_c5_authorization = validate_c5_authorization
    policy._validate_scientific_output_phase = validate_scientific_output_phase
    policy._c5_future_paths = lambda: {}
    policy._test_phase_calls = phase_calls
    monkeypatch.setattr(
        c5,
        "_load_bridge_policy_from_verified_bytes",
        lambda _raw: policy,
    )
    return path, policy


def test_real_b5_constants_create_r5_authorization_in_tmp_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    del frozen_bridge
    instruction_id, authorization_basis = _b5_user_authority()
    (
        payload,
        authorization,
        receipt,
        terminal,
        _old,
        _c1_fragment,
        _c2_fragment,
        _c3_fragment,
        runner,
    ) = _temporary_chain(
        monkeypatch,
        tmp_path,
        real_b5_authorization=True,
    )
    assert authorization.parent.is_relative_to(tmp_path)
    assert authorization.is_file()
    assert payload["instruction_id"] == instruction_id
    assert payload["authorization_basis"] == authorization_basis
    assert c5.compat_c1.legacy.INSTRUCTION_ID == instruction_id
    assert c5.compat_c1.legacy.AUTHORIZATION_BASIS == authorization_basis
    assert not receipt.exists()
    assert not terminal.exists()
    assert not (c5.COMPAT_UNIT_DIRECTORY / c5.COMPAT_UNIT).exists()
    assert runner.reloaded is False


def test_real_tmp_b5_authorization_mode_0444_is_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    path, policy = _install_real_tmp_bridge_policy(
        monkeypatch=monkeypatch, tmp_path=tmp_path, mode=0o444,
    )
    authorization, root, source = c5._validate_c5_bridge_authorization(
        require_fresh=True, require_future_absence=True,
    )
    assert path.parent == tmp_path
    assert root["mode"] == 0o444
    assert root["path"] == str(path)
    assert authorization["mutation_authority"] == c5.BRIDGE_MUTATION_AUTHORITY
    assert authorization["authorization_basis"] == (
        policy.AUTHORIZATION_BASIS
    )
    assert authorization["instruction_id"] == policy.INSTRUCTION_ID
    raw = path.read_bytes()
    assert "修改后继续".encode() in raw
    body = dict(authorization)
    producer_fingerprint = body.pop("authorization_fingerprint")
    ascii_fingerprint = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    ).hexdigest()
    assert producer_fingerprint != ascii_fingerprint
    assert source["file_sha256"] == frozen_bridge
    assert policy._test_phase_calls == [
        (False, c5.RUNTIME_PHASE_PREACTIVATION),
    ]


def test_bridge_consumer_forwards_exact_context_runtime_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    del frozen_bridge
    _path, policy = _install_real_tmp_bridge_policy(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        mode=0o444,
    )
    token = c5._RUNTIME_PHASE_CONTEXT.set(c5.RUNTIME_PHASE_FINALIZE_SUCCESS)
    try:
        c5._validate_c5_bridge_authorization(
            require_fresh=False,
            require_future_absence=False,
        )
    finally:
        c5._RUNTIME_PHASE_CONTEXT.reset(token)
    assert policy._test_phase_calls == [
        (True, c5.RUNTIME_PHASE_FINALIZE_SUCCESS),
    ]
    assert c5._RUNTIME_PHASE_CONTEXT.get() == (
        c5.RUNTIME_PHASE_PREACTIVATION
    )


def test_bridge_consumer_rejects_b5_phase_vocabulary_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    del frozen_bridge
    _path, policy = _install_real_tmp_bridge_policy(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        mode=0o444,
    )
    policy.RUNTIME_PHASES = frozenset(
        set(policy.RUNTIME_PHASES) | {"unexpected"}
    )
    with pytest.raises(PermissionError, match="interface changed"):
        c5._validate_c5_bridge_authorization(
            require_fresh=False,
            require_future_absence=False,
        )
    assert policy._test_phase_calls == []

@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("target_loaded", "not-found/inert"),
        ("missing_c3", "protected unit-state closure"),
        ("missing_c4", "protected unit-state closure"),
        ("c3_active", "static/inert"),
        ("c3_wrong_fragment", "static/inert"),
        ("c4_wrong_fragment", "static/inert"),
        ("old_wrong_fragment", "static/inert"),
        ("c1_wrong_fragment", "static/inert"),
        ("extra_target_field", "not-found/inert"),
    ),
)
def test_b5_authorized_target_and_protected_states_are_exactly_consumed(
    case: str,
    message: str,
) -> None:
    authorization = _fake_bridge_observation()[0]
    if case == "target_loaded":
        authorization["preauthorization_target_unit_state"] = (
            _loaded_state(c5.COMPAT_UNIT)
        )
    elif case == "missing_c3":
        del authorization["protected_unit_states"]["c3"]
    elif case == "missing_c4":
        del authorization["protected_unit_states"]["c4"]
    elif case == "c3_active":
        authorization["protected_unit_states"]["c3"][
            "ActiveState"
        ] = "active"
    elif case == "c3_wrong_fragment":
        authorization["protected_unit_states"]["c3"][
            "FragmentPath"
        ] = "/wrong"
    elif case == "c4_wrong_fragment":
        authorization["protected_unit_states"]["c4"][
            "FragmentPath"
        ] = "/wrong"
    elif case == "old_wrong_fragment":
        authorization["protected_unit_states"]["old"][
            "FragmentPath"
        ] = "/wrong"
    elif case == "c1_wrong_fragment":
        authorization["protected_unit_states"]["c1"][
            "FragmentPath"
        ] = "/wrong"
    else:
        authorization["preauthorization_target_unit_state"][
            "unexpected"
        ] = "drift"
    with pytest.raises(PermissionError, match=message):
        c5._validate_b5_authorized_unit_states(authorization)



@pytest.mark.parametrize("mode", (0o400, 0o440, 0o644, 0o446))
def test_real_tmp_b5_authorization_nonexact_mode_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
    mode: int,
) -> None:
    del frozen_bridge
    path, _policy = _install_real_tmp_bridge_policy(
        monkeypatch=monkeypatch, tmp_path=tmp_path, mode=mode,
    )
    assert path.parent == tmp_path
    with pytest.raises(PermissionError, match="unsafe sealed file"):
        c5._validate_c5_bridge_authorization(
            require_fresh=True, require_future_absence=True,
        )


def test_public_legacy_surface_has_no_base_write_bypass() -> None:
    assert "__getattr__" not in c5.__dict__
    assert not isinstance(c5.compat_c1, ModuleType)
    assert not isinstance(c5.compat_c1.legacy, ModuleType)
    for name in (
        "_frozen_create_authorization",
        "_frozen_realize_actual_unit",
        "_frozen_write_create_once_json",
        "_compat_write_create_once_json",
        "_write_create_once_json_bound",
        "_install_fragment",
    ):
        with pytest.raises(AttributeError, match="bypass is unavailable"):
            getattr(c5.legacy, name)
        with pytest.raises(AttributeError, match="raw surface is unavailable"):
            getattr(c5.compat_c1, name)
    with pytest.raises(PermissionError, match="CLI entry point is disabled"):
        c5.legacy.main([])
    with pytest.raises(PermissionError, match="write helper is disabled"):
        c5.legacy.write_create_once_json(Path("unused"), {})


def test_every_internal_raw_mutator_requires_r5_capability_before_io(
    tmp_path: Path,
) -> None:
    target = tmp_path / "must-not-exist.json"
    fragment = tmp_path / c5.COMPAT_UNIT
    write_calls = (
        c5._compat_c1._frozen_write_create_once_json,
        c5._compat_c1._compat_write_create_once_json,
        c5._compat_c1.legacy._write_create_once_json_bound,
        c5._compat_c1.legacy.write_create_once_json,
    )
    for entry in write_calls:
        with pytest.raises(PermissionError, match="delegat|contract"):
            entry(
                target,
                {"schema_version": "unknown"},
                fingerprint_field="fingerprint",
            )
    for entry in (
        c5._compat_c1._frozen_create_authorization,
        c5._compat_c1.create_authorization,
        c5._compat_c1.legacy.create_authorization,
        c5._C1_CREATE_AUTHORIZATION,
    ):
        with pytest.raises(PermissionError, match="delegat|contract"):
            entry()
    for entry in (
        c5._compat_c1._frozen_realize_actual_unit,
        c5._compat_c1.realize_actual_unit,
        c5._compat_c1.legacy.realize_actual_unit,
        c5._C1_REALIZE_ACTUAL_UNIT,
    ):
        with pytest.raises(PermissionError, match="delegation"):
            entry()
    with pytest.raises(PermissionError, match="delegation"):
        c5._compat_c1.legacy._install_fragment(tmp_path, b"forbidden")
    with pytest.raises(PermissionError, match="CLI is disabled"):
        c5._compat_c1.main([])
    with pytest.raises(PermissionError, match="CLI is disabled"):
        c5._compat_c1.legacy.main([])
    assert not target.exists()
    assert not fragment.exists()


@pytest.mark.parametrize(
    "argv",
    (
        ("/usr/bin/touch", "/tmp/r5-forbidden"),
        ("/usr/bin/systemctl", "--user", "daemon-reload"),
    ),
)
def test_guarded_runner_rejects_mutating_or_unknown_command_before_runner(
    argv: tuple[str, ...],
) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(PermissionError, match="delegat|outside closure"):
        c5._compat_c1.legacy._run(argv, runner=runner)
    assert calls == []


@pytest.mark.parametrize(
    "entry",
    (c5.create_authorization, c5.realize_actual_unit),
)
def test_mutation_capability_is_reset_after_delegated_failure(
    entry: object,
    tmp_path: Path,
    frozen_bridge: str,
) -> None:
    del frozen_bridge
    with pytest.raises(PermissionError, match="delegated path"):
        entry()
    target = tmp_path / "capability-must-be-reset.json"
    with pytest.raises(PermissionError, match="delegat|contract"):
        c5._compat_c1._frozen_write_create_once_json(
            target,
            {"schema_version": "unknown"},
            fingerprint_field="fingerprint",
        )
    assert not target.exists()


@pytest.mark.parametrize("callback_kind", ("runner", "manager_reader"))
def test_external_callbacks_cannot_reenter_active_mutation_capability(
    callback_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_bridge: str,
) -> None:
    del frozen_bridge
    target = tmp_path / "callback-reentry-must-not-exist.json"
    attempts: list[str] = []

    def probe() -> None:
        attempts.append(callback_kind)
        with pytest.raises(PermissionError, match="delegat|contract"):
            c5._compat_c1._frozen_write_create_once_json(
                target,
                {"schema_version": "unknown"},
                fingerprint_field="fingerprint",
            )

    kwargs = {f"{callback_kind.split('_')[0]}_probe": probe}
    _temporary_chain(monkeypatch, tmp_path, **kwargs)
    assert attempts
    assert not target.exists()


@pytest.mark.parametrize(
    "drift",
    (
        "candidate",
        "actions",
        "shadow",
        "mutation",
        "auth_keys",
        "schema",
        "file_binding_keys",
        "sha_regex",
        "boot_regex",
        "parent_binding_keys",
        "generation_fields",
        "source_binding_keys",
        "bridge_root_volatile_fields",
    ),
)
def test_delegated_exact_global_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    if drift == "candidate":
        monkeypatch.setattr(c5._compat_c1.legacy, "CANDIDATE", "DRIFT")
    elif drift == "actions":
        monkeypatch.setattr(
            c5._compat_c1.legacy,
            "_ACTIONS",
            [*c5._compat_c1.legacy._ACTIONS, "DRIFT"],
        )
    elif drift == "shadow":
        monkeypatch.setattr(
            c5._compat_c1.legacy,
            "_SHADOW_PROPERTIES",
            (*c5._compat_c1.legacy._SHADOW_PROPERTIES, "DriftProperty"),
        )
    elif drift == "mutation":
        value = dict(c5._compat_c1._MUTATION_AUTHORITY)
        value["start"] = True
        monkeypatch.setattr(c5._compat_c1, "_MUTATION_AUTHORITY", value)
    elif drift == "auth_keys":
        monkeypatch.setattr(
            c5._compat_c1.legacy,
            "_AUTH_KEYS",
            {*c5._compat_c1.legacy._AUTH_KEYS, "drift"},
        )
    elif drift == "schema":
        monkeypatch.setattr(
            c5._compat_c1.legacy,
            "AUTHORIZATION_SCHEMA",
            "drift-schema",
        )
    elif drift == "file_binding_keys":
        monkeypatch.setattr(
            c5._compat_c1.legacy,
            "_FILE_BINDING_KEYS",
            {"path"},
        )
    elif drift == "sha_regex":
        monkeypatch.setattr(c5._compat_c1.legacy, "_SHA", re.compile(".*"))
    elif drift == "boot_regex":
        monkeypatch.setattr(
            c5._compat_c1.legacy,
            "_BOOT_ID",
            re.compile(".*"),
        )
    elif drift == "parent_binding_keys":
        monkeypatch.setattr(c5._compat_c1, "_PARENT_BINDING_KEYS", {"path"})
    elif drift == "generation_fields":
        monkeypatch.setattr(c5._compat_c1, "_GENERATION_FIELDS", ("st_dev",))
    elif drift == "source_binding_keys":
        monkeypatch.setattr(c5._compat_c1, "_SOURCE_BINDING_KEYS", {"path"})
    else:
        monkeypatch.setattr(
            c5._compat_c1,
            "_BRIDGE_ROOT_VOLATILE_PARENT_FIELDS",
            set(),
        )
    with pytest.raises(PermissionError, match="exact-global contract"):
        c5.verify_compatibility_identity()


@pytest.mark.parametrize(
    ("owner", "name"),
    (
        ("c1", "_frozen_expected_static_shadow"),
        ("c1", "_bridge_roots_same"),
        ("c1", "_observe_protected_predecessor_fragment"),
        ("c1", "_configure_isolated_namespace"),
        ("base", "_validate_binding"),
        ("base", "_require_no_shadow"),
        ("base", "_stable_read_file"),
    ),
)
def test_every_transitive_source_callable_identity_is_guarded(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    name: str,
) -> None:
    module = c5._compat_c1 if owner == "c1" else c5._compat_c1.legacy
    monkeypatch.setattr(module, name, lambda *_args, **_kwargs: None)
    with pytest.raises(PermissionError, match="callable identity changed"):
        c5.verify_compatibility_identity()


def test_transitive_callable_guard_covers_complete_loaded_source_maps() -> None:
    pristine = {
        label: function_count
        for label, function_count, _entry_count in c5._PRISTINE_CALLABLE_COVERAGE
    }
    active = {
        label: function_count
        for label, function_count, _entry_count in c5._ACTIVE_CALLABLE_COVERAGE
    }
    assert pristine["c1"] >= 32
    assert pristine["base"] >= 44
    assert active["c1"] >= 30
    assert active["base"] >= 41


@pytest.mark.parametrize(
    ("owner", "name"),
    (
        ("c1", "Path"),
        ("c1", "Mapping"),
        ("c1", "datetime"),
        ("c1", "timezone"),
        ("base", "timedelta"),
        ("c1", "deepcopy"),
        ("c1", "ModuleType"),
        ("c1", "os"),
        ("c1", "stat"),
        ("c1", "json"),
        ("c1", "hashlib"),
        ("base", "re"),
        ("base", "pwd"),
        ("c1", "subprocess"),
    ),
)
def test_every_active_delegated_import_identity_is_guarded(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    name: str,
) -> None:
    module = c5._compat_c1 if owner == "c1" else c5._compat_c1.legacy
    monkeypatch.setattr(module, name, object())
    with pytest.raises(PermissionError, match="callable identity changed"):
        c5.verify_compatibility_identity()


@pytest.mark.parametrize(
    "state",
    ("code", "defaults", "kwdefaults", "closure"),
)
def test_transitive_function_state_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    if state == "code":
        target = c5._compat_c1.legacy._validate_binding
        assert isinstance(target, FunctionType)
        monkeypatch.setattr(
            target,
            "__code__",
            (lambda *_args, **_kwargs: None).__code__,
        )
    elif state == "defaults":
        target = c5._compat_c1.legacy.validate_archival_realization_chain
        assert target.__defaults__ is not None
        monkeypatch.setattr(
            target,
            "__defaults__",
            (*target.__defaults__, None),
        )
    elif state == "kwdefaults":
        target = c5._compat_c1.legacy.query_shadow
        assert target.__kwdefaults__ is not None
        monkeypatch.setattr(
            target,
            "__kwdefaults__",
            {**target.__kwdefaults__, "runner": object()},
        )
    else:
        target = c5._compat_c1.legacy._run
        assert target.__closure__ is not None
        cell = target.__closure__[1]
        monkeypatch.setattr(cell, "cell_contents", ("drift",))
    with pytest.raises(PermissionError, match="callable state changed"):
        c5.verify_compatibility_identity()


@pytest.mark.parametrize(
    ("alias", "entry"),
    (
        ("_invoke_c1_authorize", "create_authorization"),
        ("_invoke_c1_apply", "realize_actual_unit"),
        ("_validate_c5_receipt_contract", "create_authorization"),
    ),
)
def test_critical_r5_alias_drift_is_rejected_before_argument_handling(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
    entry: str,
) -> None:
    monkeypatch.setattr(c5, alias, lambda *_args, **_kwargs: None)
    with pytest.raises(PermissionError, match="critical R5 callable identity changed"):
        getattr(c5, entry)()


@pytest.mark.parametrize(
    ("contract", "field", "replacement"),
    (
        ("bridge_mutation", "unit_start_authorized", 0),
        ("bridge_scientific", "training_authorized", 0),
        ("c1_mutation", "daemon_reload", 1),
    ),
)
def test_authority_values_are_type_exact_not_bool_int_equal(
    monkeypatch: pytest.MonkeyPatch,
    contract: str,
    field: str,
    replacement: int,
) -> None:
    if contract == "bridge_mutation":
        value = dict(c5.BRIDGE_MUTATION_AUTHORITY)
        value[field] = replacement
        monkeypatch.setattr(c5, "BRIDGE_MUTATION_AUTHORITY", value)
    elif contract == "bridge_scientific":
        value = dict(c5.BRIDGE_SCIENTIFIC_AUTHORITY)
        value[field] = replacement
        monkeypatch.setattr(c5, "BRIDGE_SCIENTIFIC_AUTHORITY", value)
    else:
        value = dict(c5._compat_c1._MUTATION_AUTHORITY)
        value[field] = replacement
        monkeypatch.setattr(c5._compat_c1, "_MUTATION_AUTHORITY", value)
    with pytest.raises(PermissionError):
        c5.verify_compatibility_identity()


@pytest.mark.parametrize(
    "generation_name",
    (
        "_C1_LOAD_GENERATION",
        "_SELF_LOAD_GENERATION",
        "_SUPERVISOR_LOAD_GENERATION",
        "_BRIDGE_LOAD_GENERATION",
        "_TEMPLATE_LOAD_GENERATION",
        "BASE_REALIZER_GENERATION",
    ),
)
def test_generation_values_are_type_exact_not_bool_int_equal(
    monkeypatch: pytest.MonkeyPatch,
    generation_name: str,
) -> None:
    generation = dict(getattr(c5, generation_name))
    assert generation["nlink"] == 1
    assert type(generation["nlink"]) is int
    generation["nlink"] = True
    monkeypatch.setattr(c5, generation_name, generation)
    with pytest.raises(PermissionError):
        c5.verify_compatibility_identity()


@pytest.mark.parametrize(
    "drift",
    (
        "closure",
        "actions",
        "receipt",
        "bridge_authority",
        "phase",
        "context",
        "source_fields",
        "self_generation",
        "c1_generation",
    ),
)
def test_r5_owned_contract_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    if drift == "closure":
        monkeypatch.setattr(c5, "COMPATIBILITY_CLOSURE_SCHEMA", "drift")
    elif drift == "actions":
        monkeypatch.setattr(
            c5,
            "_REALIZATION_ACTIONS",
            (*c5._REALIZATION_ACTIONS, "drift"),
        )
    elif drift == "receipt":
        monkeypatch.setattr(
            c5,
            "_RECEIPT_STATIC_STATE",
            {**c5._RECEIPT_STATIC_STATE, "NRestarts": "1"},
        )
    elif drift == "bridge_authority":
        monkeypatch.setattr(
            c5,
            "BRIDGE_MUTATION_AUTHORITY",
            {**c5.BRIDGE_MUTATION_AUTHORITY, "unit_start_authorized": True},
        )
    elif drift == "phase":
        monkeypatch.setattr(
            c5,
            "RUNTIME_PHASES",
            frozenset({*c5.RUNTIME_PHASES, "drift"}),
        )
    elif drift == "context":
        monkeypatch.setattr(
            c5,
            "_RUNTIME_PHASE_CONTEXT",
            ContextVar("drift", default=c5.RUNTIME_PHASE_PREACTIVATION),
        )
    elif drift == "source_fields":
        monkeypatch.setattr(c5, "_SOURCE_FIELDS", ("st_dev",))
    elif drift == "self_generation":
        value = dict(c5._SELF_LOAD_GENERATION)
        value["inode"] = int(value["inode"]) + 1
        monkeypatch.setattr(c5, "_SELF_LOAD_GENERATION", value)
    else:
        value = dict(c5._C1_LOAD_GENERATION)
        value["inode"] = int(value["inode"]) + 1
        monkeypatch.setattr(c5, "_C1_LOAD_GENERATION", value)
    with pytest.raises(PermissionError, match="realizer-owned contract"):
        c5.verify_compatibility_identity()


def test_base_realizer_public_projection_cannot_replace_private_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = tmp_path / "cure_lite_v24_actual_unit_realization.py"
    replacement.write_bytes(c5.BASE_REALIZER_PATH.read_bytes())
    _raw, generation = c5._stable_source_bytes(replacement)
    monkeypatch.setattr(c5, "BASE_REALIZER_PATH", replacement)
    monkeypatch.setattr(
        c5,
        "BASE_REALIZER_SHA256",
        hashlib.sha256(replacement.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(c5, "BASE_REALIZER_GENERATION", generation)
    with pytest.raises(PermissionError, match="public identity changed"):
        c5._require_base_realizer_generation()


@pytest.mark.parametrize(
    "hash_field",
    ("COMPAT_BRIDGE_SOURCE_SHA256", "COMPAT_SUPERVISOR_SHA256"),
)
def test_legacy_write_entries_fail_closed_on_injected_placeholder(
    monkeypatch: pytest.MonkeyPatch,
    hash_field: str,
) -> None:
    monkeypatch.setattr(c5, hash_field, "__TO_BE_FROZEN__")
    for entry in (
        c5.legacy.create_authorization,
        c5.legacy.validate_authorization,
        c5.legacy.realize_actual_unit,
        c5.legacy.validate_archival_realization_chain,
    ):
        with pytest.raises(PermissionError, match="not frozen"):
            entry()
