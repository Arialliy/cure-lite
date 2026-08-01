from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c5.py"
)


@pytest.fixture
def bridge():
    name = "cure_lite_v24_preaccess_schema_compatibility_c5_tested"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _states(bridge) -> dict[str, dict[str, str]]:
    return {
        bridge.OLD_UNIT_NAME: _loaded_state(
            bridge.OLD_UNIT_NAME,
            fragment=str(bridge.OLD_UNIT_FRAGMENT_PATH),
        ),
        bridge.C1_UNIT_NAME: _loaded_state(
            bridge.C1_UNIT_NAME,
            fragment=str(bridge.C1_UNIT_FRAGMENT_PATH),
        ),
        bridge.C2_UNIT_NAME: _missing_state(bridge.C2_UNIT_NAME),
        bridge.C3_UNIT_NAME: _loaded_state(
            bridge.C3_UNIT_NAME,
            fragment=str(bridge.C3_UNIT_FRAGMENT_PATH),
        ),
        bridge.C4_UNIT_NAME: _loaded_state(
            bridge.C4_UNIT_NAME,
            fragment=str(bridge.C4_UNIT_FRAGMENT_PATH),
        ),
        bridge.C5_UNIT_NAME: _missing_state(bridge.C5_UNIT_NAME),
    }


def _source_root(bridge, path: Path) -> dict[str, object]:
    return bridge._source_root(path)


def _full_source_root(bridge, path: Path) -> dict[str, object]:
    root = _source_root(bridge, path)
    observed = path.lstat()
    root.update(
        {
            "owner_gid": observed.st_gid,
            "nlink": observed.st_nlink,
            "mtime_ns": observed.st_mtime_ns,
            "ctime_ns": observed.st_ctime_ns,
        }
    )
    return root


def _payload_flags() -> dict[str, bool]:
    return {
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
        "training_started": False,
        "materialization_consumed": False,
    }


def test_generation_identity_and_all_c5_paths_are_disjoint(bridge) -> None:
    assert bridge.RUNTIME_COMPATIBILITY_ID == "c5"
    assert bridge.SCIENTIFIC_ATTEMPT_ORDINAL == 2
    assert bridge.SCIENTIFIC_ATTEMPT_ID.endswith("structural_r2")
    assert bridge.C4_UNIT_NAME.endswith("compat-c4.service")
    assert bridge.C5_UNIT_NAME.endswith("compat-c5.service")
    assert bridge.C4_UNIT_NAME != bridge.C5_UNIT_NAME
    assert bridge.C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH.name == (
        "r2_preaccess_schema_compat_c4_receipt_seal_failure_terminal.json"
    )
    assert bridge.C5_AUTHORIZATION_PATH.name.endswith(
        "preaccess_schema_compat_c5_authorization.json"
    )
    assert bridge.C5_RECEIPT_PATH.name.endswith(
        "preaccess_schema_compat_c5_receipt.json"
    )
    assert bridge.C5_TERMINAL_PATH.name.endswith(
        "preaccess_schema_compat_c5_terminal.json"
    )
    assert bridge.C5_RUNTIME_SPEC_PATH != bridge.C4_RUNTIME_SPEC_PATH
    assert bridge.C5_RUN_ROOT_ALIAS_PATH != bridge.C4_RUN_ROOT_ALIAS_PATH


def test_c5_owned_json_profile_is_utf8_not_ascii_escaped(bridge) -> None:
    value = {"authorization_basis": "user instruction: 修改后继续"}
    encoded = bridge._canonical_json(value).encode("utf-8")
    assert "修改后继续".encode("utf-8") in encoded
    assert b"\\u4fee" not in encoded
    assert bridge.stable_fingerprint(value) == hashlib.sha256(encoded).hexdigest()


def test_frozen_c1_consumer_projection_is_explicit(bridge) -> None:
    assert bridge._expected_schema_compatibility() == {
        "producer_schema": "cure-lite-v24-split-access-audit-v1",
        "scientific_authorization_bound_schema": (
            "cure-lite-v24-split-access-audit-v1"
        ),
        "compatibility_consumer_required_schema": (
            "cure-lite-v24-split-access-audit-v1"
        ),
        "buggy_frozen_consumer_expected_schema": (
            "cure-lite-v24-split-access-audit-r2-v1"
        ),
        "accept_either_schema": False,
    }
    paths = bridge._source_paths()
    assert paths["compat_policy"] == paths["compat_bridge"]
    assert {"compat_policy", "compat_bridge"}.issubset(
        bridge._SOURCE_LABELS
    )


def test_c5_owned_sealed_writer_is_create_once_and_utf8(
    bridge, tmp_path: Path
) -> None:
    path = tmp_path / "authorization.json"
    payload = bridge._write_sealed(
        path,
        {
            "schema_version": "fixture-v1",
            "authorization_basis": "修改后继续",
            **_payload_flags(),
        },
        fingerprint_field="authorization_fingerprint",
    )
    assert path.stat().st_mode & 0o777 == 0o444
    assert "修改后继续".encode() in path.read_bytes()
    loaded, root = bridge._load_sealed(
        path,
        fingerprint_field="authorization_fingerprint",
        schema="fixture-v1",
    )
    assert loaded == payload
    assert root["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        bridge._write_sealed(
            path,
            {"schema_version": "fixture-v1"},
            fingerprint_field="authorization_fingerprint",
        )


def test_c4_terminal_sentinel_blocks_before_any_audit(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        bridge,
        "C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256",
        bridge._TO_BE_FROZEN_SHA256,
    )
    monkeypatch.setattr(
        bridge, "C5_AUTHORIZATION_PATH", tmp_path / "authorization.json"
    )
    monkeypatch.setattr(bridge, "C5_RECEIPT_PATH", tmp_path / "receipt.json")
    monkeypatch.setattr(bridge, "C5_TERMINAL_PATH", tmp_path / "terminal.json")
    monkeypatch.setattr(
        bridge,
        "_validate_scientific_output_phase",
        lambda **_kwargs: calls.append("scientific"),
    )
    with pytest.raises(PermissionError, match="c4 receipt-seal failure"):
        bridge.authorize_c5(
            instruction_id=bridge.INSTRUCTION_ID,
            authorization_basis=bridge.AUTHORIZATION_BASIS,
        )
    assert calls == []


@pytest.mark.parametrize(
    "source",
    [
        'PIN = "__TO_BE_FROZEN__"\n',
        'PIN = ("__TO_BE_FROZEN__")\n',
        'PIN = (\n    "__TO_BE_FROZEN__"\n)\n',
        'PIN: str = (\n    "__TO_BE_FROZEN__"\n)\n',
    ],
)
def test_unfrozen_binding_detector_rejects_parenthesized_python_pins(
    bridge, tmp_path: Path, source: str
) -> None:
    path = tmp_path / "producer.py"
    assert bridge._has_unfrozen_binding(
        source.encode("utf-8"),
        path=path,
    )


def test_unfrozen_binding_detector_ignores_only_the_fail_closed_comparison(
    bridge, tmp_path: Path
) -> None:
    path = tmp_path / "producer.py"
    source = b'if PIN == "__TO_BE_FROZEN__":\n    raise PermissionError\n'
    assert not bridge._has_unfrozen_binding(source, path=path)


def test_frozen_production_c4_failure_terminal_closes_in_b5(bridge) -> None:
    payload, root, source_root = (
        bridge._validate_c4_receipt_seal_failure_terminal()
    )
    assert payload["identity"]["runtime_compatibility_id"] == "c4"
    assert payload["continuation_policy"]["c5_required"] is True
    assert root["file_sha256"] == (
        bridge.C4_RECEIPT_SEAL_FAILURE_TERMINAL_SHA256
    )
    assert root["terminal_fingerprint"] == (
        bridge.C4_RECEIPT_SEAL_FAILURE_TERMINAL_FINGERPRINT
    )
    assert source_root["file_sha256"] == (
        bridge.C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256
    )


def test_frozen_production_c3_failure_terminal_is_historical_c4_transition(
    bridge,
) -> None:
    payload, root, source_root = (
        bridge._validate_c3_environment_failure_terminal()
    )
    continuation = payload["continuation_policy"]
    assert payload["identity"]["runtime_compatibility_id"] == "c3"
    assert continuation["c4_required"] is True
    assert "c5_required" not in continuation
    assert root["file_sha256"] == (
        bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256
    )
    assert root["terminal_fingerprint"] == (
        bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_FINGERPRINT
    )
    assert source_root["file_sha256"] == (
        bridge.C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256
    )


def _install_c3_transition_fixture(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation: str,
) -> None:
    producer, _source_root_value = (
        bridge._load_verified_c3_environment_failure_terminalizer()
    )
    payload, _terminal_root = producer.validate_archival(
        bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_PATH
    )
    payload = copy.deepcopy(payload)
    continuation = payload["continuation_policy"]
    if mutation == "missing_c4":
        continuation.pop("c4_required")
    elif mutation == "false_c4":
        continuation["c4_required"] = False
    elif mutation == "c5_substitute":
        continuation.pop("c4_required")
        continuation["c5_required"] = True
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unknown C3 mutation: {mutation}")

    source = tmp_path / f"c3-terminalizer-{mutation}.py"
    source.write_text("# coherent B5 consumer fixture\n", encoding="utf-8")
    terminal = tmp_path / f"c3-terminal-{mutation}.json"
    terminal.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    terminal.chmod(0o444)
    source_root = bridge._source_root(source)
    terminal_sha = hashlib.sha256(terminal.read_bytes()).hexdigest()
    terminal_fingerprint = "3" * 64
    root = {
        "path": str(terminal.absolute()),
        "file_sha256": terminal_sha,
        "mode": 0o444,
        "schema_version": bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SCHEMA,
        "terminal_fingerprint": terminal_fingerprint,
        "terminalizer_source_root": {
            "path": str(source.absolute()),
            "file_sha256": source_root["file_sha256"],
        },
    }
    module = SimpleNamespace(
        validate_archival=lambda _path: (payload, root),
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH", source
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINAL_PATH", terminal
    )
    monkeypatch.setattr(
        bridge,
        "C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256",
        source_root["file_sha256"],
    )
    monkeypatch.setattr(
        bridge, "C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256", terminal_sha
    )
    monkeypatch.setattr(
        bridge,
        "C3_ENVIRONMENT_FAILURE_TERMINAL_FINGERPRINT",
        terminal_fingerprint,
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_c3_environment_failure_terminalizer",
        lambda: (module, source_root),
    )


@pytest.mark.parametrize(
    "mutation",
    ["missing_c4", "false_c4", "c5_substitute"],
)
def test_c3_transition_requires_c4_and_never_accepts_c5_as_substitute(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _install_c3_transition_fixture(
        bridge,
        tmp_path,
        monkeypatch,
        mutation=mutation,
    )
    with pytest.raises(
        PermissionError,
        match="c3 environment-failure transition changed",
    ):
        bridge._validate_c3_environment_failure_terminal()


def _install_c4_transition_fixture(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation: str,
) -> None:
    producer, _source_root_value = (
        bridge._load_verified_c4_receipt_seal_failure_terminalizer()
    )
    payload, _terminal_root = producer.validate_archival(
        bridge.C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH
    )
    payload = copy.deepcopy(payload)
    continuation = payload["continuation_policy"]
    if mutation == "missing_c5":
        continuation.pop("c5_required")
    elif mutation == "false_c5":
        continuation["c5_required"] = False
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unknown C4 mutation: {mutation}")

    source = tmp_path / f"c4-terminalizer-{mutation}.py"
    source.write_text("# coherent B5 consumer fixture\n", encoding="utf-8")
    terminal = tmp_path / f"c4-terminal-{mutation}.json"
    terminal.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    terminal.chmod(0o444)
    source_root = bridge._source_root(source)
    terminal_sha = hashlib.sha256(terminal.read_bytes()).hexdigest()
    terminal_fingerprint = "4" * 64
    root = {
        "path": str(terminal.absolute()),
        "file_sha256": terminal_sha,
        "schema_version": (
            bridge.C4_RECEIPT_SEAL_FAILURE_TERMINAL_SCHEMA
        ),
        "terminal_fingerprint": terminal_fingerprint,
        "terminalizer_source_root": _full_source_root(bridge, source),
    }
    module = SimpleNamespace(
        validate_archival=lambda _path: (payload, root),
    )
    monkeypatch.setattr(
        bridge,
        "C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH",
        source,
    )
    monkeypatch.setattr(
        bridge, "C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH", terminal
    )
    monkeypatch.setattr(
        bridge,
        "C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256",
        source_root["file_sha256"],
    )
    monkeypatch.setattr(
        bridge, "C4_RECEIPT_SEAL_FAILURE_TERMINAL_SHA256", terminal_sha
    )
    monkeypatch.setattr(
        bridge,
        "C4_RECEIPT_SEAL_FAILURE_TERMINAL_FINGERPRINT",
        terminal_fingerprint,
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_c4_receipt_seal_failure_terminalizer",
        lambda: (module, source_root),
    )


@pytest.mark.parametrize("mutation", ["missing_c5", "false_c5"])
def test_c4_transition_remains_the_only_direct_c5_authority(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _install_c4_transition_fixture(
        bridge,
        tmp_path,
        monkeypatch,
        mutation=mutation,
    )
    with pytest.raises(
        PermissionError,
        match="c4 receipt-seal failure transition changed",
    ):
        bridge._validate_c4_receipt_seal_failure_terminal()


@pytest.mark.parametrize("field", ["owner_gid", "nlink", "mtime_ns", "ctime_ns"])
def test_c4_terminalizer_full_generation_drift_is_rejected(
    bridge, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    producer, source_root = (
        bridge._load_verified_c4_receipt_seal_failure_terminalizer()
    )
    payload, returned_root = producer.validate_archival(
        bridge.C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH
    )
    drifted_root = copy.deepcopy(returned_root)
    drifted_root["terminalizer_source_root"][field] += 1
    monkeypatch.setattr(
        bridge,
        "_load_verified_c4_receipt_seal_failure_terminalizer",
        lambda: (
            SimpleNamespace(
                validate_archival=lambda _path: (payload, drifted_root)
            ),
            source_root,
        ),
    )
    with pytest.raises(PermissionError, match="transition changed"):
        bridge._validate_c4_receipt_seal_failure_terminal()


def test_frozen_b4_loader_uses_exact_producer_and_interface(bridge) -> None:
    module, root = bridge._load_verified_c4_bridge()
    assert module.RUNTIME_COMPATIBILITY_ID == "c4"
    assert module._canonical_json({"x": "修改"}) != bridge._canonical_json(
        {"x": "修改"}
    )
    assert root["file_sha256"] == bridge.C4_BRIDGE_SHA256
    assert callable(module.validate_c4_authorization)


def test_b4_authorization_is_validated_by_b4_producer_not_b5_loader(
    bridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer, source_root = bridge._load_verified_c4_bridge()
    authorization, root = producer._load_sealed(
        bridge.C4_AUTHORIZATION_PATH,
        fingerprint_field="authorization_fingerprint",
        schema=bridge.C4_AUTHORIZATION_SCHEMA,
    )
    calls: list[dict[str, object]] = []

    def validate(*_args, **kwargs):
        calls.append(dict(kwargs))
        return authorization, root

    fake = SimpleNamespace(
        validate_c4_authorization=validate,
        RUNTIME_PHASE_PREACTIVATION=producer.RUNTIME_PHASE_PREACTIVATION,
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_c4_bridge",
        lambda: (fake, source_root),
    )
    monkeypatch.setattr(
        bridge,
        "_load_sealed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("B5 loader must not read B4 authorization")
        ),
    )
    authorization, root, source = bridge._validate_c4_authorization_archival(
        unit_state_reader=lambda _unit: {},
        allow_runtime_activation=False,
        runtime_phase=bridge.RUNTIME_PHASE_PREACTIVATION,
        now=bridge._utc_now,
    )
    assert authorization["runtime_compatibility_id"] == "c4"
    assert root["file_sha256"] == hashlib.sha256(
        bridge.C4_AUTHORIZATION_PATH.read_bytes()
    ).hexdigest()
    assert source["file_sha256"] == bridge.C4_BRIDGE_SHA256
    assert calls[0]["require_fresh"] is False


def _c4_failure_payload(bridge) -> dict[str, object]:
    return {
        "schema_version": bridge.C4_RECEIPT_SEAL_FAILURE_TERMINAL_SCHEMA,
        "identity": {
            "candidate": bridge.CANDIDATE,
            "stage_id": bridge.STAGE_ID,
            "scientific_attempt_id": bridge.SCIENTIFIC_ATTEMPT_ID,
            "scientific_attempt_ordinal": bridge.SCIENTIFIC_ATTEMPT_ORDINAL,
            "runtime_compatibility_id": "c4",
            "failure_stage": "B4_compatibility_receipt_seal",
            "sealed_at_utc": "2026-07-31T16:00:00.000000Z",
        },
        "b4_receipt_seal_failure": {
            "first_rejected_path": str(
                bridge.C4_UNIT_AUTHORIZATION_PATH.absolute()
            ),
            "fingerprint_field": "authorization_fingerprint",
            "producer_canonical_profile": (
                "compact_sorted_ensure_ascii_false_utf8"
            ),
            "consumer_canonical_profile": (
                "compact_sorted_ensure_ascii_true_utf8"
            ),
            "producer_fingerprint": (
                "543f794fd27e6277471eb2e52ab290a228415091c3071070cf3f0920c3d28c10"
            ),
            "consumer_recomputed_fingerprint": (
                "11b4f19ae10d7b032af4eb7611e8b36155be6cf577149450128d6b439b14cb44"
            ),
            "profile_mismatch": True,
            "receipt_writer_reached": False,
            "receipt_sealed": False,
        },
        "original_execution_observation": {
            "attempt_count": 1,
            "control_plane_observed_exit_code": 1,
            "durable_original_execution_artifact": False,
            "exit_code_independently_verifiable": False,
            "original_argv_claimed": False,
            "original_stdout_claimed": False,
            "original_stderr_claimed": False,
            "original_traceback_claimed": False,
        },
        "metadata_success_closure": {
            "r4_unit_realization_passed": True,
            "r4_static_unit_verified": True,
            "e4_scope_handoff_present": True,
            "e4_stability_attempt_count": 1,
            "e4_environment_sample_count": 2,
            "e4_stability_passed": True,
            "e4_postcleanup_passed": True,
            "c4_compatibility_receipt_present": False,
        },
        "authorization_expiry": {
            "B4_expired": True,
            "R4_expired": True,
            "B4_compatibility_receipt_absent_at_observation": True,
            "B4_sealed_by_compatibility_receipt": False,
        },
        "deterministic_reproduction": {
            "first_failure_stage": (
                "R4_authorization_fingerprint_validation"
            ),
            "first_failure_reproduced": True,
            "retry_or_replay_performed": False,
            "systemd_mutation_performed": False,
            "gpu_or_payload_accessed": False,
            "old_B4_seal_called": False,
            "R4_E4_writer_called": False,
        },
        "historical_state_observation": {
            "historical_observation_only": True,
            "future_state_authority": False,
            "archival_live_absence_recheck_required": False,
            "archival_live_manager_recheck_required": False,
        },
        "payload_observation": {
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_compute_accessed": False,
            "training_started": False,
            "scientific_samples_processed": 0,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "scientific_attempt_consumed": False,
        },
        "continuation_policy": {
            "automatic_retry": False,
            "same_c4_reentry": False,
            "same_c4_reauthorization": False,
            "same_c4_source_repair": False,
            "same_c4_loader_patch": False,
            "same_c4_receipt_seal_reentry": False,
            "r4_e4_reentry": False,
            "r14_l4_runtime_scientific_launch": False,
            "c5_required": True,
            "new_explicit_authorization_required": True,
            "scientific_attempt_consumed": False,
            "terminal_grants_c5_reuse_authority": False,
            "b4_authorization_consumed": True,
            "r4_unit_realization_consumed": True,
            "e4_metadata_attempt_consumed": True,
            "runtime_launch_consumed": False,
            "runtime_materialization_consumed": False,
            "scientific_attempt_id": bridge.SCIENTIFIC_ATTEMPT_ID,
            "scientific_attempt_ordinal": bridge.SCIENTIFIC_ATTEMPT_ORDINAL,
            "scientific_attempt_id_unchanged": True,
            "scientific_attempt_ordinal_unchanged": True,
        },
    }


def test_c4_failure_terminal_is_consumed_only_via_terminalizer(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "terminalizer.py"
    source.write_text("# fixture terminalizer\n", encoding="utf-8")
    terminal = tmp_path / "terminal.json"
    terminal.write_text("{}\n", encoding="utf-8")
    terminal.chmod(0o444)
    payload = _c4_failure_payload(bridge)
    source_root = _source_root(bridge, source)
    terminal_sha = hashlib.sha256(terminal.read_bytes()).hexdigest()
    terminal_fp = "d" * 64
    root = {
        "path": str(terminal.absolute()),
        "file_sha256": terminal_sha,
        "terminal_fingerprint": terminal_fp,
        "schema_version": bridge.C4_RECEIPT_SEAL_FAILURE_TERMINAL_SCHEMA,
        "terminalizer_source_root": _full_source_root(bridge, source),
    }
    module = SimpleNamespace(validate_archival=lambda _path: (payload, root))
    monkeypatch.setattr(
        bridge,
        "C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH",
        source,
    )
    monkeypatch.setattr(
        bridge, "C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH", terminal
    )
    monkeypatch.setattr(
        bridge,
        "C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256",
        source_root["file_sha256"],
    )
    monkeypatch.setattr(
        bridge, "C4_RECEIPT_SEAL_FAILURE_TERMINAL_SHA256", terminal_sha
    )
    monkeypatch.setattr(
        bridge, "C4_RECEIPT_SEAL_FAILURE_TERMINAL_FINGERPRINT", terminal_fp
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_c4_receipt_seal_failure_terminalizer",
        lambda: (module, source_root),
    )
    validated, validated_root, returned_source = (
        bridge._validate_c4_receipt_seal_failure_terminal()
    )
    assert validated == payload
    assert validated_root == root
    assert returned_source == source_root
    assert validated["continuation_policy"]["c5_required"] is True
    assert "passed" not in validated


def test_c4_failure_terminal_rejects_any_same_generation_reentry(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _c4_failure_payload(bridge)
    payload["continuation_policy"]["same_c4_reentry"] = True
    source = tmp_path / "terminalizer.py"
    source.write_text("# fixture\n", encoding="utf-8")
    terminal = tmp_path / "terminal.json"
    terminal.write_text("{}\n", encoding="utf-8")
    terminal.chmod(0o444)
    source_root = _source_root(bridge, source)
    digest = hashlib.sha256(terminal.read_bytes()).hexdigest()
    root = {
        "path": str(terminal.absolute()),
        "file_sha256": digest,
        "terminal_fingerprint": "e" * 64,
        "schema_version": bridge.C4_RECEIPT_SEAL_FAILURE_TERMINAL_SCHEMA,
        "terminalizer_source_root": _full_source_root(bridge, source),
    }
    monkeypatch.setattr(
        bridge,
        "C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH",
        source,
    )
    monkeypatch.setattr(
        bridge, "C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH", terminal
    )
    monkeypatch.setattr(
        bridge,
        "C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256",
        source_root["file_sha256"],
    )
    monkeypatch.setattr(
        bridge, "C4_RECEIPT_SEAL_FAILURE_TERMINAL_SHA256", digest
    )
    monkeypatch.setattr(
        bridge, "C4_RECEIPT_SEAL_FAILURE_TERMINAL_FINGERPRINT", "e" * 64
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_c4_receipt_seal_failure_terminalizer",
        lambda: (
            SimpleNamespace(validate_archival=lambda _path: (payload, root)),
            source_root,
        ),
    )
    with pytest.raises(PermissionError, match="transition changed"):
        bridge._validate_c4_receipt_seal_failure_terminal()


def test_protected_units_include_exact_c4_and_missing_c5(bridge) -> None:
    states = _states(bridge)
    protected = bridge._collect_protected_unit_states(states.__getitem__)
    assert set(protected) == {"old", "c1", "c2", "c3", "c4"}
    assert protected["c4"]["FragmentPath"] == str(
        bridge.C4_UNIT_FRAGMENT_PATH
    )
    target = bridge._collect_preauthorization_target_unit_state(
        states.__getitem__
    )
    assert target["LoadState"] == "not-found"
    states[bridge.C4_UNIT_NAME]["FragmentPath"] = "/wrong"
    with pytest.raises(PermissionError, match="static/inert"):
        bridge._collect_protected_unit_states(states.__getitem__)


@pytest.mark.parametrize("unit_label", ["old", "c1"])
def test_protected_old_fragment_paths_are_exact(bridge, unit_label: str) -> None:
    states = _states(bridge)
    unit_name = {
        "old": bridge.OLD_UNIT_NAME,
        "c1": bridge.C1_UNIT_NAME,
    }[unit_label]
    states[unit_name]["FragmentPath"] = "/wrong/fragment.service"
    with pytest.raises(PermissionError, match="static/inert"):
        bridge._collect_protected_unit_states(states.__getitem__)


def test_append_only_path_sets_protect_c4_and_isolate_c5(bridge) -> None:
    permanent = bridge._always_absent_paths()
    preauthorization = bridge._c5_preauthorization_paths()
    future = bridge._c5_future_paths()
    assert {
        "c4_compatibility_receipt",
        "c4_environment_stability_terminal",
        "c4_unit_terminal",
        "c4_runtime_spec",
        "c4_runtime_launch_authorization",
        "c4_runtime_artifacts",
        "c4_gpu_lease",
        "c4_run_alias",
        "c4_result_alias",
    }.issubset(permanent)
    assert "c5_compatibility_terminal" in preauthorization
    assert set(future) == {
        "c5_runtime_spec",
        "c5_runtime_launch_authorization",
        "c5_runtime_artifacts",
        "c5_gpu_lease",
    }


def _environment_payloads() -> dict[str, dict[str, object]]:
    return {
        "scope_handoff": _payload_flags(),
        "stability_attempt": _payload_flags(),
        "policy": _payload_flags(),
        "stability": _payload_flags(),
        "postcleanup": _payload_flags(),
    }


def test_e5_evidence_is_loaded_only_by_fixed_e5_producer(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _environment_payloads()
    roots = {
        label: {"path": f"/{label}", "file_sha256": "a" * 64}
        for label in {
            "environment_scope_handoff",
            "environment_stability_attempt",
            "environment_policy",
            "environment_stability",
            "environment_postcleanup",
        }
    }
    producer = SimpleNamespace(
        load_c5_environment_closure=lambda: {
            **payloads,
            "evidence_roots": roots,
        }
    )
    monkeypatch.setattr(
        bridge, "C5_ENVIRONMENT_STABILITY_TERMINAL_PATH", tmp_path / "absent"
    )
    monkeypatch.setattr(
        bridge,
        "_load_sealed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("B5 loader must not read E5 evidence")
        ),
    )
    loaded = bridge._load_environment_evidence(producer)
    assert loaded[0] == payloads["scope_handoff"]
    assert loaded[-1] == roots


def test_e5_producer_root_label_drift_is_rejected(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = SimpleNamespace(
        load_c5_environment_closure=lambda: {
            **_environment_payloads(),
            "evidence_roots": {"environment_policy": {}},
        }
    )
    monkeypatch.setattr(
        bridge, "C5_ENVIRONMENT_STABILITY_TERMINAL_PATH", tmp_path / "absent"
    )
    with pytest.raises(PermissionError, match="root labels"):
        bridge._load_environment_evidence(producer)


@pytest.mark.parametrize(
    ("allow_runtime_activation", "runtime_phase"),
    [
        (False, "preactivation"),
        (True, "run_once"),
        (True, "finalize_success"),
    ],
)
def test_r5_evidence_is_loaded_only_by_fixed_r5_producer(
    bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_runtime_activation: bool,
    runtime_phase: str,
) -> None:
    fragment = tmp_path / "c5.service"
    fragment.write_text("[Service]\nRestart=no\n", encoding="utf-8")
    fragment_root = bridge._source_root(fragment)
    now = "2026-07-31T17:00:00.000000Z"
    authorization = {
        "unit_name": bridge.C5_UNIT_NAME,
        "issued_at_utc": "2026-07-31T16:59:00.000000Z",
        "expires_at_utc": "2026-07-31T17:04:00.000000Z",
        "manager_generation": {"x": 1},
        **_payload_flags(),
    }
    receipt = {
        "unit_name": bridge.C5_UNIT_NAME,
        "created_at_utc": now,
        "passed": True,
        "static": True,
        "started": False,
        "enabled": False,
        "removed": False,
        "manager_generation": {"x": 1},
        "fragment_identity": {
            "path": str(fragment),
            "file_sha256": fragment_root["file_sha256"],
            "device": fragment_root["device"],
            "inode": fragment_root["inode"],
        },
        "full_static_shadow": {
            "Id": bridge.C5_UNIT_NAME,
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "static",
            "NRestarts": "0",
            "FragmentPath": str(fragment),
            "InvocationID": "",
            "DropInPaths": [],
            "NeedDaemonReload": "no",
            "Transient": "no",
            "Restart": "no",
            "ExecMainPID": "0",
            "ExecMainCode": "0",
            "ExecMainStatus": "0",
            "Result": "success",
        },
        **_payload_flags(),
    }
    calls: list[dict[str, object]] = []

    def validate_archival(*_args, **kwargs):
        calls.append(dict(kwargs))
        return {
            "authorization": authorization,
            "authorization_identity": {"path": "/auth"},
            "receipt": receipt,
            "receipt_identity": {"path": "/receipt"},
        }

    producer = SimpleNamespace(
        validate_archival_realization_chain=validate_archival,
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_unit_realizer",
        lambda _root: (producer, {"source": "root"}),
    )
    monkeypatch.setattr(
        bridge,
        "_load_sealed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("B5 loader must not read R5 evidence")
        ),
    )
    state = _loaded_state(bridge.C5_UNIT_NAME, fragment=str(fragment))
    if allow_runtime_activation:
        state.update(
            {
                "ActiveState": "active",
                "SubState": "running",
                "InvocationID": "0123456789abcdef0123456789abcdef",
            }
        )
    result = bridge._validate_unit_chain(
        realizer_root={"source": "root"},
        unit_state_reader=lambda _unit: state,
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    assert result[0] == authorization
    assert result[2] == receipt
    assert calls == [
        {
            "allow_runtime_activation": allow_runtime_activation,
            "runtime_phase": runtime_phase,
        }
    ]


@pytest.mark.parametrize(
    ("allow_runtime_activation", "runtime_phase"),
    [(False, "preactivation"), (True, "run_once"), (True, "finalize_success")],
)
def test_compat_authorization_adapter_preserves_exact_runtime_phase(
    bridge,
    monkeypatch: pytest.MonkeyPatch,
    allow_runtime_activation: bool,
    runtime_phase: str,
) -> None:
    calls: list[dict[str, object]] = []

    def validate(**kwargs):
        calls.append(dict(kwargs))
        return {**_payload_flags()}, {"path": "/authorization"}

    phase_checks: list[dict[str, object]] = []
    monkeypatch.setattr(bridge, "validate_c5_authorization", validate)
    monkeypatch.setattr(
        bridge,
        "_validate_scientific_output_phase",
        lambda **kwargs: phase_checks.append(dict(kwargs)),
    )
    result = bridge.validate_compat_authorization(
        unit_state_reader=lambda _unit: {},
        require_fresh=False,
        require_future_absence=False,
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    assert result[1] == {"path": "/authorization"}
    assert calls[0]["allow_runtime_activation"] is allow_runtime_activation
    assert calls[0]["runtime_phase"] == runtime_phase
    assert phase_checks == [
        {
            "allow_runtime_activation": allow_runtime_activation,
            "runtime_phase": runtime_phase,
        }
    ]


def test_foreign_producer_calls_are_static_and_no_dual_profile_fallback(
    bridge,
) -> None:
    source = SOURCE.read_text(encoding="utf-8")
    unit_body = source.split("def _validate_unit_chain", 1)[1].split(
        "def _load_environment_evidence", 1
    )[0]
    environment_body = source.split(
        "def _load_environment_evidence", 1
    )[1].split("def _collect_full_closure", 1)[0]
    assert "validate_archival_realization_chain" in unit_body
    assert "_load_sealed(" not in unit_body
    assert "load_c5_environment_closure" in environment_body
    assert "_load_sealed(" not in environment_body
    assert "ensure_ascii=True" not in source
    assert "for ensure_ascii in" not in source


def _mock_authorization_dependencies(
    bridge, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    # Keep the two direct archival predecessor validators and the source-root
    # collector real.  This prevents authorization tests from masking a
    # consumer/producer continuation-field mismatch.
    roots = bridge._collect_source_roots()
    c4_bridge_root = roots["c4_bridge"]
    c1_terminal = {
        "authorization_expiry": {
            "bridge_expires_at_utc": "2026-07-30T00:00:00.000000Z"
        },
        "evidence_roots": {
            "bridge_authorization": {"path": "/c1-auth"},
            "r10_authorization": {"path": "/r10-auth"},
            "r10_receipt": {"path": "/r10-receipt"},
        },
    }
    monkeypatch.setattr(
        bridge, "_validate_scientific_output_phase", lambda **_kwargs: None
    )
    monkeypatch.setattr(bridge, "_require_absent", lambda _paths: None)
    monkeypatch.setattr(
        bridge,
        "_validate_c2_prewrite_failure_terminal",
        lambda: (
            {},
            {"path": "/c2-prewrite"},
            roots["c2_prewrite_failure_terminalizer"],
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_mode_contract_failure_terminal",
        lambda: (
            {"identity": {"sealed_at_utc": "2026-07-30T01:00:00.000000Z"}},
            {"path": "/c2-mode"},
            roots["c2_mode_contract_failure_terminalizer"],
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_c4_authorization_archival",
        lambda **_kwargs: (
            {"runtime_compatibility_id": "c4"},
            {"path": "/c4-auth"},
            c4_bridge_root,
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_validate_c1_failure_terminal",
        lambda **_kwargs: (c1_terminal, {"path": "/c1-terminal"}, {}),
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_environment_wrapper",
        lambda root: (SimpleNamespace(), root),
    )
    monkeypatch.setattr(
        bridge,
        "_load_verified_unit_realizer",
        lambda root: (SimpleNamespace(), root),
    )
    return roots, c1_terminal


def test_authorize_and_validate_c5_capture_direct_c4_failure_lineage(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_path = tmp_path / "c5-authorization.json"
    receipt_path = tmp_path / "c5-receipt.json"
    terminal_path = tmp_path / "c5-terminal.json"
    monkeypatch.setattr(bridge, "C5_AUTHORIZATION_PATH", auth_path)
    monkeypatch.setattr(bridge, "C5_RECEIPT_PATH", receipt_path)
    monkeypatch.setattr(bridge, "C5_TERMINAL_PATH", terminal_path)
    roots, _terminal = _mock_authorization_dependencies(bridge, monkeypatch)
    states = _states(bridge)
    issued = datetime(2026, 7, 31, 17, 0, tzinfo=timezone.utc)
    authorization = bridge.authorize_c5(
        instruction_id=bridge.INSTRUCTION_ID,
        authorization_basis=bridge.AUTHORIZATION_BASIS,
        unit_state_reader=states.__getitem__,
        now=lambda: issued,
    )
    assert authorization["runtime_compatibility_id"] == "c5"
    assert authorization["scientific_attempt_ordinal"] == 2
    assert authorization["c4_authorization_root"] == {"path": "/c4-auth"}
    assert authorization["c3_environment_failure_terminal_root"][
        "file_sha256"
    ] == bridge.C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256
    assert authorization["c4_receipt_seal_failure_terminal_root"][
        "file_sha256"
    ] == bridge.C4_RECEIPT_SEAL_FAILURE_TERMINAL_SHA256
    assert authorization["compatibility_source_roots"] == roots
    assert authorization["protected_unit_states"]["c4"]["LoadState"] == (
        "loaded"
    )
    assert "修改后继续".encode() in auth_path.read_bytes()
    validated, root = bridge.validate_c5_authorization(
        unit_state_reader=states.__getitem__,
        require_fresh=True,
        now=lambda: issued,
    )
    assert validated == authorization
    assert root["path"] == str(auth_path.absolute())


def test_seal_receipt_refuses_existing_c5_terminal(
    bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    terminal = tmp_path / "terminal.json"
    terminal.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(bridge, "C5_RECEIPT_PATH", receipt)
    monkeypatch.setattr(bridge, "C5_TERMINAL_PATH", terminal)
    with pytest.raises(FileExistsError, match="identity is consumed"):
        bridge.seal_receipt()


def test_parser_exposes_only_metadata_c5_commands(bridge) -> None:
    parser = bridge.build_parser()
    help_text = parser.format_help()
    assert "authorize-c5" in help_text
    assert "seal-receipt" in help_text
    assert "verify-compatibility-receipt" in help_text
    for forbidden in ("start", "train", "D_V", "D_T", "retry", "resume"):
        assert forbidden not in help_text


def test_bridge_has_no_scientific_or_systemd_mutation_imports(bridge) -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import numpy" not in source
    assert "systemctl\", \"--user\", \"start" not in source
    assert "daemon-reload" not in source
    assert "ensure_ascii=False" in source
    assert bridge._expected_scientific_authority() == {
        "D_R_payload_authorized": False,
        "D_V_payload_authorized": False,
        "D_T_payload_authorized": False,
        "training_authorized": False,
        "materialization_authorized": False,
        "automatic_retry": False,
        "resume": False,
        "fresh_scientific_attempt": False,
    }
