from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tools import (
    cure_lite_v24_actual_runtime_release_preaccess_compat_c3 as compat,
)


def _valid_r13_release_closure() -> dict[str, object]:
    source_identity = compat._SUPERVISOR_LOAD_IDENTITY
    assert source_identity is not None
    supervisor = compat.COMPAT_SUPERVISOR_PATH.absolute()
    return {
        "integration": {
            "authorization": {
                "scenario_root": {
                    "path": str(compat.INTEGRATION_ROOT.absolute()),
                },
                "executable_bindings": {
                    "supervisor": {
                        "path": str(supervisor),
                        "resolved_path": str(supervisor),
                        "path_is_symlink": False,
                        "file_sha256": compat.COMPAT_SUPERVISOR_SHA256,
                        "device": source_identity["st_dev"],
                        "inode": source_identity["st_ino"],
                        "owner_uid": source_identity["st_uid"],
                        "mode": stat.S_IMODE(source_identity["st_mode"]),
                    },
                },
            },
            "identities": {
                "authorization": {
                    "path": str(
                        compat.INTEGRATION_AUTHORIZATION_PATH.absolute()
                    ),
                },
                "receipt": {
                    "path": str(
                        compat.INTEGRATION_RECEIPT_PATH.absolute()
                    ),
                },
            },
        },
    }


def test_c3_identity_is_disjoint_and_retains_scientific_r2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compat, "_configure", lambda: None)
    monkeypatch.setattr(compat.legacy, "UNIT_NAME", compat.COMPAT_UNIT)
    monkeypatch.setattr(
        compat.legacy, "RUNTIME_SPEC_PATH", compat.COMPAT_RUNTIME_SPEC_PATH
    )
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
    assert identity["runtime_compatibility_generation"] == "c3"
    assert identity["fresh_scientific_attempt"] is False
    assert identity["automatic_retry_allowed"] is False
    assert identity["resume_allowed"] is False
    assert identity["D_R_payload_accessed"] is False
    assert identity["D_V_payload_accessed"] is False
    assert identity["D_T_payload_accessed"] is False
    assert identity["frozen_c1_release_file_sha256"] == (
        "395a013ff4f14160a0ac4e9845497caf9ecbaa6f2eeb3aa88fad54b63f514cfa"
    )
    assert "compat_c3" in identity["runtime_spec_path"]
    assert identity["runtime_spec_path"] != str(compat.C1_RUNTIME_PATHS[0])
    assert identity["runtime_spec_path"] != str(compat.C2_RUNTIME_PATHS[0])
    assert str(compat.SCIENTIFIC_RUN_ROOT).endswith(
        "gcr_pacre_v24_D_R_structural_attempt_r2"
    )


@pytest.mark.parametrize(
    ("name", "replacement"),
    (
        ("COMPAT_RUNTIME_SPEC_PATH", "c1-runtime-spec"),
        ("COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH", "c1-authorization"),
        ("COMPAT_RUNTIME_ARTIFACT_ROOT", "c1-artifacts"),
        ("COMPAT_GPU_LEASE_ROOT", "c1-lease"),
        ("COMPAT_RUN_ROOT_ALIAS_PATH", "scientific-run"),
        ("COMPAT_RESULT_RECEIPT_ALIAS_PATH", "scientific-receipt"),
    ),
)
def test_c3_runtime_identity_rejects_path_reuse(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: str,
) -> None:
    values = {
        "c1-runtime-spec": compat.C1_RUNTIME_PATHS[0],
        "c1-authorization": compat.C1_RUNTIME_PATHS[1],
        "c1-artifacts": compat.C1_RUNTIME_PATHS[2],
        "c1-lease": compat.C1_RUNTIME_PATHS[3],
        "scientific-run": compat.SCIENTIFIC_RUN_ROOT,
        "scientific-receipt": compat.SCIENTIFIC_RESULT_RECEIPT_PATH,
    }
    monkeypatch.setattr(compat, name, values[replacement])
    with pytest.raises(PermissionError, match="identity/path isolation"):
        compat._require_disjoint_c3_runtime_identity()


def test_c3_runtime_identity_rejects_unit_or_scientific_target_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compat,
        "COMPAT_UNIT",
        "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service",
    )
    with pytest.raises(PermissionError, match="identity/path isolation"):
        compat._require_disjoint_c3_runtime_identity()
    monkeypatch.undo()
    monkeypatch.setattr(
        compat,
        "SCIENTIFIC_RUN_ROOT",
        compat.COMPAT_RUN_ROOT_ALIAS_PATH,
    )
    with pytest.raises(PermissionError, match="identity/path isolation"):
        compat._require_disjoint_c3_runtime_identity()


def test_frozen_c1_release_is_hash_before_exec_bound() -> None:
    assert compat._sha256_file(compat.FROZEN_RELEASE_PATH) == (
        compat.FROZEN_RELEASE_SHA256
    )
    assert compat.FROZEN_RELEASE_SHA256 == (
        "395a013ff4f14160a0ac4e9845497caf9ecbaa6f2eeb3aa88fad54b63f514cfa"
    )

def test_frozen_sibling_hashes_match_loaded_generations() -> None:
    paths = (
        compat.COMPAT_BRIDGE_PATH,
        compat.COMPAT_REALIZER_PATH,
        compat.COMPAT_SUPERVISOR_PATH,
        compat.COMPAT_ENVIRONMENT_PATH,
    )
    digests = (
        compat.COMPAT_BRIDGE_SHA256,
        compat.COMPAT_REALIZER_SHA256,
        compat.COMPAT_SUPERVISOR_SHA256,
        compat.COMPAT_ENVIRONMENT_SHA256,
    )
    identities = (
        compat._BRIDGE_LOAD_IDENTITY,
        compat._REALIZER_LOAD_IDENTITY,
        compat._SUPERVISOR_LOAD_IDENTITY,
        compat._ENVIRONMENT_LOAD_IDENTITY,
    )
    for path, digest, identity in zip(paths, digests, identities, strict=True):
        assert compat._sha256_file(path) == digest
        assert identity is not None
    compat._require_component_sources()


def test_unfrozen_sibling_hashes_are_production_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compat, "COMPAT_BRIDGE_SHA256", "__TO_BE_FROZEN__")
    monkeypatch.setattr(compat, "_BRIDGE_LOAD_IDENTITY", None)
    with pytest.raises(PermissionError, match="hash is not frozen"):
        compat._require_component_sources()


def test_l3_has_no_inherited_public_writer_projection() -> None:
    for module in (compat, compat.compat_c1):
        assert "__getattr__" not in module.__dict__
        for name in ("build_spec", "authorize_launch"):
            assert name not in module.__dict__
            with pytest.raises(AttributeError):
                getattr(module, name)


@pytest.mark.parametrize(
    "call",
    (
        lambda: compat.legacy.main(["authorize-launch"]),
        lambda: compat.legacy.build_spec(),
        lambda: compat.legacy.authorize_launch(),
        lambda: compat.compat_c1.main(["authorize-launch"]),
        lambda: compat.legacy._write_sealed(
            Path("/tmp/c3-release-guarded.json"),
            {},
            fingerprint_field="runtime_spec_fingerprint",
        ),
        lambda: compat.legacy._write_sealed_bound(
            Path("/tmp/c3-release-guarded.json"),
            {},
            fingerprint_field="runtime_spec_fingerprint",
        ),
        lambda: compat.compat_c1._frozen_write_sealed(
            Path("/tmp/c3-release-guarded.json"),
            {},
            fingerprint_field="runtime_spec_fingerprint",
        ),
        lambda: compat.legacy._create_runtime_directories_and_verify_leaves({}),
        lambda: compat.compat_c1._frozen_create_runtime_directories({}),
        lambda: compat.legacy._private_directory(
            Path("/tmp/c3-release-guarded-directory"),
            create=True,
        ),
    ),
)
def test_every_inherited_mutation_entry_requires_l3_main_capability(call) -> None:
    with pytest.raises(
        PermissionError,
        match="(c3 release (delegation|phase)|c1 release entrypoint)",
    ):
        call()
    assert not Path("/tmp/c3-release-guarded.json").exists()
    assert not Path("/tmp/c3-release-guarded-directory").exists()


def test_inherited_guards_publish_no_dunder_wrapped_raw_callable() -> None:
    guarded = (
        compat.legacy.main,
        compat.legacy.build_spec,
        compat.legacy.authorize_launch,
        compat.legacy._write_sealed,
        compat.legacy._write_sealed_bound,
        compat.legacy._create_runtime_directories_and_verify_leaves,
        compat.legacy._private_directory,
        compat.compat_c1.main,
        compat.compat_c1._frozen_write_sealed,
    )
    assert all(not hasattr(function, "__wrapped__") for function in guarded)


def test_l3_delegation_has_no_public_capability_minter() -> None:
    assert "__getattr__" not in compat.__dict__
    assert not hasattr(compat, "_activate_l3_delegation")
    assert not hasattr(compat, "_build_l3_delegation_gate")
    assert not hasattr(compat, "_reset_l3_delegation")


def test_l3_delegation_binds_the_exact_main_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(command="authorize-launch")
    monkeypatch.setattr(compat, "_require_frozen_release_source", lambda: None)
    monkeypatch.setattr(compat, "_require_component_sources", lambda: None)
    monkeypatch.setattr(
        compat.legacy,
        "_parser",
        lambda: SimpleNamespace(parse_args=lambda _argv: args),
    )
    monkeypatch.setattr(compat, "verify_compatibility_identity", lambda: {})
    monkeypatch.setattr(compat, "_require_lane_separation", lambda: None)
    monkeypatch.setattr(
        compat,
        "_verify_active_phase_compatibility_receipt",
        lambda: {},
    )
    monkeypatch.setattr(
        compat.legacy,
        "main",
        lambda _argv: compat._require_l3_delegation(
            "authorize-launch",
            argv=["authorize-launch", "--validity-seconds", "1"],
        ),
    )
    with pytest.raises(PermissionError, match="delegation is not authorized"):
        compat.main(["authorize-launch"])
    assert compat._ACTIVE_COMMAND.get() is None



def test_same_bytes_inode_replacement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (tmp_path / "c1-release.py").absolute()
    raw = b"VALUE = 1\n"
    source.write_bytes(raw)
    _bytes, identity = compat._stable_source_bytes(source)
    monkeypatch.setattr(compat, "FROZEN_RELEASE_PATH", source)
    monkeypatch.setattr(
        compat, "FROZEN_RELEASE_SHA256", hashlib.sha256(raw).hexdigest()
    )
    monkeypatch.setattr(
        compat, "_FROZEN_RELEASE_LOAD_IDENTITY", identity
    )
    compat._require_frozen_release_source()
    replacement = tmp_path / "replacement.py"
    replacement.write_bytes(raw)
    replacement.replace(source)
    with pytest.raises(PermissionError, match="generation changed"):
        compat._require_frozen_release_source()


@pytest.mark.parametrize("lane", ("direct", "c1", "c2"))
def test_predecessor_runtime_materialization_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
) -> None:
    direct = tuple(tmp_path / f"direct-{index}" for index in range(4))
    c1 = tuple(tmp_path / f"c1-{index}" for index in range(4))
    c2 = tuple(tmp_path / f"c2-{index}" for index in range(4))
    aliases = tuple(tmp_path / f"alias-{index}" for index in range(4))
    monkeypatch.setattr(compat, "BLOCKED_RUNTIME_PATHS", direct)
    monkeypatch.setattr(compat, "C1_RUNTIME_PATHS", c1)
    monkeypatch.setattr(compat, "C2_RUNTIME_PATHS", c2)
    monkeypatch.setattr(compat, "FORBIDDEN_SCIENTIFIC_ALIASES", aliases)
    target = {"direct": direct, "c1": c1, "c2": c2}[lane][2]
    target.mkdir()
    with pytest.raises(PermissionError, match="predecessor runtime lane"):
        compat._require_lane_separation()


def test_historical_or_c3_scientific_alias_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = tuple(tmp_path / f"alias-{index}" for index in range(4))
    monkeypatch.setattr(compat, "BLOCKED_RUNTIME_PATHS", ())
    monkeypatch.setattr(compat, "C1_RUNTIME_PATHS", ())
    monkeypatch.setattr(compat, "C2_RUNTIME_PATHS", ())
    monkeypatch.setattr(compat, "FORBIDDEN_SCIENTIFIC_ALIASES", aliases)
    aliases[1].write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="scientific alias"):
        compat._require_lane_separation()


@pytest.mark.parametrize(
    "future_name",
    (
        "COMPAT_RUNTIME_SPEC_PATH",
        "COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH",
        "COMPAT_RUNTIME_ARTIFACT_ROOT",
        "COMPAT_GPU_LEASE_ROOT",
    ),
)
def test_build_phase_requires_every_c3_future_path_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    future_name: str,
) -> None:
    for name in (
        "COMPAT_RUNTIME_SPEC_PATH",
        "COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH",
        "COMPAT_RUNTIME_ARTIFACT_ROOT",
        "COMPAT_GPU_LEASE_ROOT",
    ):
        monkeypatch.setattr(compat, name, tmp_path / name)
    getattr(compat, future_name).write_text("early\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="already materialized"):
        compat._require_build_phase_pristine()


@pytest.mark.parametrize(
    "name", ("SCIENTIFIC_RUN_ROOT", "SCIENTIFIC_RESULT_RECEIPT_PATH")
)
def test_authorize_requires_scientific_outputs_pristine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setattr(compat, "SCIENTIFIC_RUN_ROOT", tmp_path / "run")
    monkeypatch.setattr(
        compat, "SCIENTIFIC_RESULT_RECEIPT_PATH", tmp_path / "receipt"
    )
    getattr(compat, name).write_text("premature\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="before c3 launch"):
        compat._require_prelaunch_scientific_pristine()


def test_build_and_authorize_phases_use_exact_bridge_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        compat,
        "_require_build_phase_pristine",
        lambda: None,
    )
    monkeypatch.setattr(
        compat,
        "_require_prelaunch_scientific_pristine",
        lambda: None,
    )
    monkeypatch.setattr(
        compat,
        "_load_fixed_compat_runtime_spec",
        lambda: {"runtime_spec_fingerprint": "a" * 64},
    )
    monkeypatch.setattr(
        compat,
        "_verify_compatibility_receipt",
        lambda **kwargs: calls.append(
            (kwargs["expected_spec"], kwargs["require_spec_binding"])
        )
        or {"runtime_compatibility_id": "c3"},
    )
    token = compat._ACTIVE_COMMAND.set("build-spec")
    try:
        compat._verify_active_phase_compatibility_receipt()
    finally:
        compat._ACTIVE_COMMAND.reset(token)
    token = compat._ACTIVE_COMMAND.set("authorize-launch")
    try:
        compat._verify_active_phase_compatibility_receipt()
    finally:
        compat._ACTIVE_COMMAND.reset(token)
    assert calls == [
        (None, False),
        ({"runtime_spec_fingerprint": "a" * 64}, True),
    ]


def test_bridge_receipt_arguments_and_environment_cross_binding_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_spec = {"runtime_spec_fingerprint": "b" * 64}
    calls: list[tuple[object, object, bool, bool]] = []
    result = {
        "runtime_compatibility_id": "c3",
        "automatic_retry": False,
        "resume": False,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "compatibility_source_roots": {
            "compat_bridge": {
                "path": str(compat.COMPAT_BRIDGE_PATH),
                "file_sha256": compat.COMPAT_BRIDGE_SHA256,
            }
        },
    }
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
    env_calls: list[object] = []
    monkeypatch.setattr(
        compat,
        "_verify_environment_cross_binding",
        lambda receipt: env_calls.append(receipt) or {},
    )
    assert compat._verify_compatibility_receipt(
        expected_spec=expected_spec,
        require_spec_binding=True,
    ) == result
    assert calls == [
        (
            compat.COMPATIBILITY_RECEIPT_PATH,
            expected_spec,
            True,
            False,
        )
    ]
    assert env_calls == [result]


def test_environment_cross_binding_calls_authoritative_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = SimpleNamespace(name="old")
    c3 = SimpleNamespace(name="c3")
    archival = {
        "authorization": {"unit_name": compat.COMPAT_UNIT},
        "receipt": {"unit_name": compat.COMPAT_UNIT},
    }
    normalized_old = {"name": "old"}
    normalized_c3 = {"name": "c3"}
    monkeypatch.setattr(
        compat.compat_environment,
        "replay_old_scope_and_handoff",
        lambda: (old, c3, {}),
        raising=False,
    )
    monkeypatch.setattr(
        compat.compat_realizer,
        "validate_archival_realization_chain",
        lambda *_args: archival,
        raising=False,
    )
    monkeypatch.setattr(
        compat.compat_environment,
        "validate_c3_realization_archival",
        lambda value, *, contract: value,
        raising=False,
    )
    monkeypatch.setattr(
        compat.compat_environment,
        "frozen",
        SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(
        compat.compat_environment.frozen,
        "load_sealed_receipt",
        lambda path, **_kwargs: {"path": str(path)},
        raising=False,
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        compat.compat_environment,
        "validate_c3_environment_closure",
        lambda *args, **kwargs: calls.append((*args, kwargs))
        or {"realization": archival},
        raising=False,
    )
    monkeypatch.setattr(
        compat.compat_environment,
        "_normalized_contract",
        lambda value: normalized_old if value is old else normalized_c3,
        raising=False,
    )
    receipt = {
        "historical_environment_contract": normalized_old,
        "current_environment_contract": normalized_c3,
    }
    closure = compat._verify_environment_cross_binding(receipt)
    assert closure["realization"] == archival
    assert len(calls) == 1
    assert calls[0][-1]["archival"] == archival
    assert calls[0][-1]["c3_contract"] is c3


def test_release_closure_orders_bridge_before_frozen_and_archival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        compat,
        "_require_l3_delegation",
        lambda _command, **_kwargs: None,
    )
    command_token = compat._ACTIVE_COMMAND.set("build-spec")
    monkeypatch.setattr(
        compat,
        "_require_lane_separation",
        lambda: events.append("lane"),
    )
    monkeypatch.setattr(
        compat,
        "_require_fixed_inputs",
        lambda _values: events.append("paths"),
    )
    monkeypatch.setattr(
        compat,
        "_verify_active_phase_compatibility_receipt",
        lambda: events.append("bridge")
        or {
            "compatibility_evidence_roots": {
                "unit_realization_authorization": {
                    "path": str(compat.REALIZATION_AUTHORIZATION_PATH)
                },
                "unit_realization_receipt": {
                    "path": str(compat.REALIZATION_RECEIPT_PATH)
                },
            }
        },
    )
    monkeypatch.setattr(
        compat,
        "_require_frozen_release_source",
        lambda: events.append("hash"),
    )
    monkeypatch.setattr(
        compat,
        "_require_component_sources",
        lambda: events.append("sources"),
    )
    monkeypatch.setattr(
        compat,
        "_frozen_validate_release_closure",
        lambda **_kwargs: events.append("frozen")
        or {
            "realization": {
                "authorization": {"a": 1},
                "receipt": {"r": 1},
            }
        },
    )
    monkeypatch.setattr(
        compat,
        "_require_final_r13_supervisor_binding",
        lambda _result: events.append("r13"),
    )
    monkeypatch.setattr(
        compat.compat_realizer,
        "validate_archival_realization_chain",
        lambda *_args: events.append("archival")
        or {
            "authorization": {"a": 1},
            "receipt": {"r": 1},
            "compatibility_closure": {"passed": True},
        },
        raising=False,
    )
    try:
        compat._compat_validate_release_closure()
    finally:
        compat._ACTIVE_COMMAND.reset(command_token)
    assert events == [
        "lane", "paths", "bridge", "hash", "sources", "frozen",
        "r13", "archival",
    ]


def test_r13_binding_accepts_current_final_supervisor_generation() -> None:
    compat._require_final_r13_supervisor_binding(
        _valid_r13_release_closure()
    )


@pytest.mark.parametrize(
    "field",
    (
        "path",
        "resolved_path",
        "file_sha256",
        "device",
        "inode",
        "owner_uid",
        "mode",
    ),
)
def test_r13_binding_rejects_stale_supervisor_generation(
    field: str,
) -> None:
    closure = _valid_r13_release_closure()
    binding = closure["integration"]["authorization"][
        "executable_bindings"
    ]["supervisor"]
    if field in {"path", "resolved_path"}:
        binding[field] = "/tmp/stale-c3-supervisor.py"
    elif field == "file_sha256":
        binding[field] = "0" * 64
    else:
        binding[field] += 1
    with pytest.raises(PermissionError, match="r13 integration"):
        compat._require_final_r13_supervisor_binding(closure)


@pytest.mark.parametrize(
    "location",
    ("authorization", "receipt", "scenario_root"),
)
def test_r13_binding_rejects_wrong_fixed_root(location: str) -> None:
    closure = _valid_r13_release_closure()
    integration = closure["integration"]
    if location == "authorization":
        integration["identities"]["authorization"]["path"] = (
            "/tmp/stale-r13/control/authorization.json"
        )
    elif location == "receipt":
        integration["identities"]["receipt"]["path"] = (
            "/tmp/stale-r13/control/integration-receipt.json"
        )
    else:
        integration["authorization"]["scenario_root"]["path"] = (
            "/tmp/stale-r13"
        )
    with pytest.raises(PermissionError, match="r13 integration"):
        compat._require_final_r13_supervisor_binding(closure)


def test_r13_binding_rejects_non_single_link_current_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = _valid_r13_release_closure()
    identity = deepcopy(compat._SUPERVISOR_LOAD_IDENTITY)
    assert identity is not None
    identity["st_nlink"] = 2
    monkeypatch.setattr(compat, "_SUPERVISOR_LOAD_IDENTITY", identity)
    with pytest.raises(PermissionError, match="r13 integration"):
        compat._require_final_r13_supervisor_binding(closure)


def test_exact_preview_context_is_required_before_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"schema_version": "preview"}
    preview = compat._materialize_spec_body(
        compat.COMPAT_RUNTIME_SPEC_PATH,
        body,
        fingerprint_field="runtime_spec_fingerprint",
    )
    calls: list[object] = []
    monkeypatch.setattr(
        compat,
        "_frozen_write_sealed",
        lambda *args, **kwargs: calls.append((args, kwargs)) or preview,
    )
    monkeypatch.setattr(
        compat,
        "_require_l3_delegation",
        lambda _command, **_kwargs: None,
    )
    token = compat._APPROVED_PREWRITE_SPEC.set(preview)
    try:
        assert compat._compat_write_sealed(
            compat.COMPAT_RUNTIME_SPEC_PATH,
            body,
            fingerprint_field="runtime_spec_fingerprint",
        ) == preview
    finally:
        compat._APPROVED_PREWRITE_SPEC.reset(token)
    assert len(calls) == 1
    drifted = dict(preview)
    drifted["schema_version"] = "drift"
    token = compat._APPROVED_PREWRITE_SPEC.set(drifted)
    try:
        with pytest.raises(PermissionError, match="mutation-free preview"):
            compat._compat_write_sealed(
                compat.COMPAT_RUNTIME_SPEC_PATH,
                body,
                fingerprint_field="runtime_spec_fingerprint",
            )
    finally:
        compat._APPROVED_PREWRITE_SPEC.reset(token)


def test_only_authoritative_v1_audit_is_projected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = (tmp_path / "audit.json").absolute()
    monkeypatch.setattr(
        compat.legacy, "SCIENTIFIC_ACCESS_AUDIT_PATH", audit
    )
    calls: list[tuple[Path, str | None]] = []
    monkeypatch.setattr(
        compat.compat_c1,
        "_require_frozen_release_source",
        lambda: None,
    )
    monkeypatch.setattr(
        compat.compat_c1,
        "_frozen_load_sealed",
        lambda path, **kwargs: calls.append((path, kwargs["schema"]))
        or {"schema_version": kwargs["schema"]},
    )
    compat._compat_load_sealed(
        audit,
        fingerprint_field="receipt_fingerprint",
        schema=compat.FICTIONAL_ACCESS_AUDIT_SCHEMA,
    )
    assert calls == [(audit, compat.AUTHORITATIVE_ACCESS_AUDIT_SCHEMA)]


def test_off_path_build_input_is_rejected() -> None:
    values = {
        "policy_path": Path("/tmp/off-path-policy.json"),
        "precleanup_path": compat.PRECLEANUP_PATH,
        "cleanup_plan_path": compat.CLEANUP_PLAN_PATH,
        "cleanup_authorization_path": compat.CLEANUP_AUTHORIZATION_PATH,
        "cleanup_receipt_path": compat.CLEANUP_RECEIPT_PATH,
        "stability_path": compat.STABILITY_PATH,
        "postcleanup_path": compat.POSTCLEANUP_PATH,
        "integration_authorization_path": (
            compat.INTEGRATION_AUTHORIZATION_PATH
        ),
        "integration_receipt_path": compat.INTEGRATION_RECEIPT_PATH,
        "realization_authorization_path": (
            compat.REALIZATION_AUTHORIZATION_PATH
        ),
        "realization_receipt_path": compat.REALIZATION_RECEIPT_PATH,
    }
    with pytest.raises(PermissionError, match="policy_path"):
        compat._require_fixed_inputs(values)


def test_r13_and_c3_environment_inputs_are_fixed() -> None:
    assert compat.INTEGRATION_ROOT.name.endswith("compat_c3_r13")
    assert "compat_c3" in compat.POLICY_PATH.name
    assert "compat_c3" in compat.STABILITY_PATH.name
    assert "compat_c3" in compat.POSTCLEANUP_PATH.name
    assert "compat_c3" in compat.REALIZATION_AUTHORIZATION_PATH.name
    assert "compat_c3" in compat.REALIZATION_RECEIPT_PATH.name


def test_delegation_rechecks_hashes_immediately_before_frozen_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    args = SimpleNamespace(command="authorize-launch")
    monkeypatch.setattr(
        compat,
        "_require_frozen_release_source",
        lambda: events.append("hash"),
    )
    monkeypatch.setattr(
        compat,
        "_require_component_sources",
        lambda: events.append("sources"),
    )
    monkeypatch.setattr(
        compat.legacy,
        "_parser",
        lambda: SimpleNamespace(parse_args=lambda _argv: args),
    )
    monkeypatch.setattr(
        compat,
        "verify_compatibility_identity",
        lambda: events.append("identity") or {},
    )
    monkeypatch.setattr(
        compat,
        "_require_lane_separation",
        lambda: events.append("lane"),
    )
    monkeypatch.setattr(
        compat,
        "_verify_active_phase_compatibility_receipt",
        lambda: events.append("bridge") or {},
    )
    monkeypatch.setattr(
        compat.legacy,
        "main",
        lambda delegated_argv: compat._require_l3_delegation(
            "authorize-launch",
            argv=delegated_argv,
        )
        or events.append("frozen")
        or 0,
    )
    assert compat.main(["authorize-launch"]) == 0
    assert "identity" in events
    assert events[-3:] == ["hash", "sources", "frozen"]
    assert compat._ACTIVE_COMMAND.get() is None
    command_token = compat._ACTIVE_COMMAND.set("authorize-launch")
    try:
        with pytest.raises(PermissionError, match="delegation is not authorized"):
            compat._require_l3_delegation("authorize-launch")
    finally:
        compat._ACTIVE_COMMAND.reset(command_token)


def test_source_declares_no_retry_resume_or_dv_dt_authority() -> None:
    source = compat.COMPAT_BRIDGE_PATH.parent / (
        "cure_lite_v24_actual_runtime_release_preaccess_compat_c3.py"
    )
    text = source.read_text(encoding="utf-8")
    assert '"automatic_retry_allowed": False' in text
    assert '"resume_allowed": False' in text
    assert '"D_V_payload_accessed": False' in text
    assert '"D_T_payload_accessed": False' in text
    assert "systemctl" not in text
    assert "subprocess" not in text
