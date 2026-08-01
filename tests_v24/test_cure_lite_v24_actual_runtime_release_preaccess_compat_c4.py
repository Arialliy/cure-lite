from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tools import (
    cure_lite_v24_actual_runtime_release_preaccess_compat_c4 as compat,
)


def _valid_r14_release_closure() -> dict[str, object]:
    source_identity = compat._SUPERVISOR_LOAD_IDENTITY
    assert source_identity is not None
    supervisor = compat.COMPAT_SUPERVISOR_PATH.absolute()
    return {
        "integration": {
            "authorization": {
                "scenario_id": compat.R14_SCENARIO_ID,
                "identity": compat.r14_integration.build_supervisor_v2_identity(
                    compat.R14_SCENARIO_ID
                ),
                "scenario_root": {
                    "path": str(compat.INTEGRATION_ROOT.absolute()),
                },
                "control_root": {
                    "path": str(
                        (compat.INTEGRATION_ROOT / "control").absolute()
                    ),
                },
                "runtime_root": {
                    "path": str(
                        (compat.INTEGRATION_ROOT / "runtime").absolute()
                    ),
                },
                "runtime_spec_binding": {
                    "path": str(compat.INTEGRATION_RUNTIME_SPEC_PATH),
                },
                "control_artifacts": {
                    "integration_terminal": str(
                        compat.INTEGRATION_TERMINAL_PATH
                    ),
                    "removal_authorization": str(
                        compat.INTEGRATION_REMOVAL_AUTHORIZATION_PATH
                    ),
                    "removal_state": str(
                        compat.INTEGRATION_REMOVAL_STATE_PATH
                    ),
                    "integration_receipt": str(
                        compat.INTEGRATION_RECEIPT_PATH
                    ),
                    "dummy_artifact": str(
                        compat.INTEGRATION_ROOT / "runtime/dummy-child.json"
                    ),
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
                "integration_terminal": {
                    "path": str(compat.INTEGRATION_TERMINAL_PATH),
                },
                "removal_authorization": {
                    "path": str(
                        compat.INTEGRATION_REMOVAL_AUTHORIZATION_PATH
                    ),
                },
                "removal_state": {
                    "path": str(compat.INTEGRATION_REMOVAL_STATE_PATH),
                },
            },
        },
    }


def _unit_path_policy(inode: int) -> dict[str, object]:
    uid = os.getuid()
    runtime_directory = f"/run/user/{uid}/systemd/user"
    return {
        "runtime_directory": runtime_directory,
        "ordered_unit_paths": [
            {
                "path": str(
                    Path(runtime_directory).parent / "generator.late"
                ),
                "exists": True,
                "device": 17,
                "inode": inode,
                "owner_uid": uid,
                "mode": 0o700,
            },
            {
                "path": "/usr/lib/systemd/user",
                "exists": True,
                "device": 23,
                "inode": 29,
                "owner_uid": 0,
                "mode": 0o755,
            },
        ],
    }


def _mock_release_snapshot(
    *,
    policy: dict[str, object],
    authorization_inode: int = 101,
    receipt_inode: int = 202,
) -> dict[str, object]:
    archival = {
        "authorization": {"authorization": "sealed"},
        "receipt": {
            "receipt": "sealed",
            "unit_path_policy": _unit_path_policy(11),
        },
        "authorization_identity": {"inode": authorization_inode},
        "receipt_identity": {"inode": receipt_inode},
        "compatibility_closure": {"passed": True},
    }
    return {
        "archival": archival,
        "fragment": {"inode": 303},
        "shadow": {"Id": compat.COMPAT_UNIT},
        "manager": {"boot_id": "same-manager"},
        "live_unit_path_policy": policy,
    }


def _run_mocked_release_closure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    monkeypatch.setattr(
        compat,
        "_require_l4_delegation",
        lambda _command, **_kwargs: None,
    )
    monkeypatch.setattr(compat, "_require_lane_separation", lambda: None)
    monkeypatch.setattr(compat, "_require_fixed_inputs", lambda _values: None)
    monkeypatch.setattr(compat, "_require_frozen_release_source", lambda: None)
    monkeypatch.setattr(compat, "_require_component_sources", lambda: None)
    monkeypatch.setattr(
        compat,
        "_verify_active_phase_compatibility_receipt",
        lambda: {
            "compatibility_evidence_roots": {
                "unit_realization_authorization": {
                    "path": str(compat.REALIZATION_AUTHORIZATION_PATH),
                },
                "unit_realization_receipt": {
                    "path": str(compat.REALIZATION_RECEIPT_PATH),
                },
            },
        },
    )
    snapshots = iter((deepcopy(before), deepcopy(after)))
    monkeypatch.setattr(
        compat,
        "_snapshot_production_c4",
        lambda **_kwargs: next(snapshots),
    )
    archival = before["archival"]
    result = {
        "integration": {},
        "realization": {
            "authorization": deepcopy(archival["authorization"]),
            "receipt": deepcopy(archival["receipt"]),
            "authorization_identity": deepcopy(
                archival["authorization_identity"]
            ),
            "receipt_identity": deepcopy(archival["receipt_identity"]),
        },
    }
    monkeypatch.setattr(
        compat,
        "_frozen_validate_release_closure",
        lambda **_kwargs: result,
    )
    monkeypatch.setattr(
        compat,
        "_require_final_r14_supervisor_binding",
        lambda _result: None,
    )
    monkeypatch.setattr(
        compat,
        "_validate_r14_runtime_chain",
        lambda _integration: {
            "supervisor_evidence": {"invocation_id": "a" * 32},
        },
    )
    monkeypatch.setattr(
        compat.compat_environment,
        "_bind_r4_archival_root",
        lambda _path, value: value,
    )
    token = compat._ACTIVE_COMMAND.set("build-spec")
    try:
        return compat._compat_validate_release_closure()
    finally:
        compat._ACTIVE_COMMAND.reset(token)


def test_c4_identity_is_disjoint_and_retains_scientific_r2(
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
    assert identity["runtime_compatibility_generation"] == "c4"
    assert identity["fresh_scientific_attempt"] is False
    assert identity["automatic_retry_allowed"] is False
    assert identity["resume_allowed"] is False
    assert identity["D_R_payload_accessed"] is False
    assert identity["D_V_payload_accessed"] is False
    assert identity["D_T_payload_accessed"] is False
    assert identity["frozen_c1_release_file_sha256"] == (
        "395a013ff4f14160a0ac4e9845497caf9ecbaa6f2eeb3aa88fad54b63f514cfa"
    )
    assert "compat_c4" in identity["runtime_spec_path"]
    assert identity["runtime_spec_path"] != str(compat.C1_RUNTIME_PATHS[0])
    assert identity["runtime_spec_path"] != str(compat.C2_RUNTIME_PATHS[0])
    assert identity["runtime_spec_path"] != str(compat.C3_RUNTIME_PATHS[0])
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
        ("COMPAT_RUNTIME_SPEC_PATH", "c3-runtime-spec"),
        ("COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH", "c3-authorization"),
        ("COMPAT_RUNTIME_ARTIFACT_ROOT", "c3-artifacts"),
        ("COMPAT_GPU_LEASE_ROOT", "c3-lease"),
        ("COMPAT_RUN_ROOT_ALIAS_PATH", "scientific-run"),
        ("COMPAT_RESULT_RECEIPT_ALIAS_PATH", "scientific-receipt"),
    ),
)
def test_c4_runtime_identity_rejects_path_reuse(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: str,
) -> None:
    values = {
        "c1-runtime-spec": compat.C1_RUNTIME_PATHS[0],
        "c1-authorization": compat.C1_RUNTIME_PATHS[1],
        "c1-artifacts": compat.C1_RUNTIME_PATHS[2],
        "c1-lease": compat.C1_RUNTIME_PATHS[3],
        "c3-runtime-spec": compat.C3_RUNTIME_PATHS[0],
        "c3-authorization": compat.C3_RUNTIME_PATHS[1],
        "c3-artifacts": compat.C3_RUNTIME_PATHS[2],
        "c3-lease": compat.C3_RUNTIME_PATHS[3],
        "scientific-run": compat.SCIENTIFIC_RUN_ROOT,
        "scientific-receipt": compat.SCIENTIFIC_RESULT_RECEIPT_PATH,
    }
    monkeypatch.setattr(compat, name, values[replacement])
    with pytest.raises(PermissionError, match="identity/path isolation"):
        compat._require_disjoint_c4_runtime_identity()


def test_c4_runtime_identity_rejects_unit_or_scientific_target_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compat,
        "COMPAT_UNIT",
        "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service",
    )
    with pytest.raises(PermissionError, match="identity/path isolation"):
        compat._require_disjoint_c4_runtime_identity()
    monkeypatch.undo()
    monkeypatch.setattr(
        compat,
        "SCIENTIFIC_RUN_ROOT",
        compat.COMPAT_RUN_ROOT_ALIAS_PATH,
    )
    with pytest.raises(PermissionError, match="identity/path isolation"):
        compat._require_disjoint_c4_runtime_identity()


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
        compat.COMPAT_ADAPTER_PATH,
        compat.R14_INTEGRATION_PATH,
        compat.R14_SHARED_REALIZER_PATH,
        compat.R14_DUMMY_CHILD_PATH,
        compat.R14_DUMMY_TEMPLATE_PATH,
    )
    digests = (
        compat.COMPAT_BRIDGE_SHA256,
        compat.COMPAT_REALIZER_SHA256,
        compat.COMPAT_SUPERVISOR_SHA256,
        compat.COMPAT_ENVIRONMENT_SHA256,
        compat.COMPAT_ADAPTER_SHA256,
        compat.R14_INTEGRATION_SHA256,
        compat.R14_SHARED_REALIZER_SHA256,
        compat.R14_DUMMY_CHILD_SHA256,
        compat.R14_DUMMY_TEMPLATE_SHA256,
    )
    identities = (
        compat._BRIDGE_LOAD_IDENTITY,
        compat._REALIZER_LOAD_IDENTITY,
        compat._SUPERVISOR_LOAD_IDENTITY,
        compat._ENVIRONMENT_LOAD_IDENTITY,
        compat._ADAPTER_LOAD_IDENTITY,
        compat._R14_INTEGRATION_LOAD_IDENTITY,
        compat._R14_SHARED_REALIZER_LOAD_IDENTITY,
        compat._R14_DUMMY_CHILD_LOAD_IDENTITY,
        compat._R14_DUMMY_TEMPLATE_LOAD_IDENTITY,
    )
    for path, digest, identity in zip(paths, digests, identities, strict=True):
        assert compat._sha256_file(path) == digest
        assert identity is not None
    compat._require_component_sources()


def test_all_l4_component_pins_are_exact() -> None:
    assert {
        "bridge": compat.COMPAT_BRIDGE_SHA256,
        "realizer": compat.COMPAT_REALIZER_SHA256,
        "supervisor": compat.COMPAT_SUPERVISOR_SHA256,
        "environment": compat.COMPAT_ENVIRONMENT_SHA256,
        "adapter": compat.COMPAT_ADAPTER_SHA256,
        "r14_integration": compat.R14_INTEGRATION_SHA256,
        "r14_realizer": compat.R14_SHARED_REALIZER_SHA256,
        "r14_child": compat.R14_DUMMY_CHILD_SHA256,
        "r14_template": compat.R14_DUMMY_TEMPLATE_SHA256,
    } == {
        "bridge": "ad660b7afe7ca87f690bc9565bd6674684c2b62824394751a39114a6efcf178a",
        "realizer": "8708f8a13d74623f510992e23c6c23e1c4bfe70db09092c04fe56d44d29c5b65",
        "supervisor": "faffe980cba4cad668a7d0f525bed8f2005950503d46f2b7c6888d79813c64ce",
        "environment": "f4335efdb3865efe68dbbb6aac5f7977fd2157452b557f83428e4dd4a5d8932b",
        "adapter": "fa5fe28eacf3980720616a9b54dc1dc878f1e1280883f643a49ec4c4800e92ed",
        "r14_integration": "10f6814deab43cfa4513b813546759b1e6508272a18ee026ebecb2cf8535a187",
        "r14_realizer": "131b89f186b064629354165a9454b976145b5d76fbf36053292b2997e73cf6b6",
        "r14_child": "d57e21c7450c58faba8d6915ffe09647b19313f3f734e8ec2847a34001ab27b9",
        "r14_template": "8df34c4ea07d2dfe3b23f0c8407df66b9f282bb43f7275c245ee110f90f568c8",
    }


def test_unfrozen_sibling_hashes_are_production_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compat, "COMPAT_BRIDGE_SHA256", "__TO_BE_FROZEN__")
    monkeypatch.setattr(compat, "_BRIDGE_LOAD_IDENTITY", None)
    with pytest.raises(PermissionError, match="hash is not frozen"):
        compat._require_component_sources()


def test_l4_has_no_inherited_public_writer_projection() -> None:
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
            Path("/tmp/c4-release-guarded.json"),
            {},
            fingerprint_field="runtime_spec_fingerprint",
        ),
        lambda: compat.legacy._write_sealed_bound(
            Path("/tmp/c4-release-guarded.json"),
            {},
            fingerprint_field="runtime_spec_fingerprint",
        ),
        lambda: compat.compat_c1._frozen_write_sealed(
            Path("/tmp/c4-release-guarded.json"),
            {},
            fingerprint_field="runtime_spec_fingerprint",
        ),
        lambda: compat.legacy._create_runtime_directories_and_verify_leaves({}),
        lambda: compat.compat_c1._frozen_create_runtime_directories({}),
        lambda: compat.legacy._private_directory(
            Path("/tmp/c4-release-guarded-directory"),
            create=True,
        ),
    ),
)
def test_every_inherited_mutation_entry_requires_l4_main_capability(call) -> None:
    with pytest.raises(
        PermissionError,
        match="(c4 release (delegation|phase)|c1 release entrypoint)",
    ):
        call()
    assert not Path("/tmp/c4-release-guarded.json").exists()
    assert not Path("/tmp/c4-release-guarded-directory").exists()


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


def test_l4_delegation_has_no_public_capability_minter() -> None:
    assert "__getattr__" not in compat.__dict__
    assert not hasattr(compat, "_activate_l4_delegation")
    assert not hasattr(compat, "_build_l4_delegation_gate")
    assert not hasattr(compat, "_reset_l4_delegation")


def test_l4_delegation_binds_the_exact_main_argv(
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
        lambda _argv: compat._require_l4_delegation(
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


@pytest.mark.parametrize("lane", ("direct", "c1", "c2", "c3"))
def test_predecessor_runtime_materialization_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
) -> None:
    direct = tuple(tmp_path / f"direct-{index}" for index in range(4))
    c1 = tuple(tmp_path / f"c1-{index}" for index in range(4))
    c2 = tuple(tmp_path / f"c2-{index}" for index in range(4))
    c3 = tuple(tmp_path / f"c3-{index}" for index in range(4))
    aliases = tuple(tmp_path / f"alias-{index}" for index in range(4))
    monkeypatch.setattr(compat, "BLOCKED_RUNTIME_PATHS", direct)
    monkeypatch.setattr(compat, "C1_RUNTIME_PATHS", c1)
    monkeypatch.setattr(compat, "C2_RUNTIME_PATHS", c2)
    monkeypatch.setattr(compat, "C3_RUNTIME_PATHS", c3)
    monkeypatch.setattr(compat, "FORBIDDEN_SCIENTIFIC_ALIASES", aliases)
    target = {"direct": direct, "c1": c1, "c2": c2, "c3": c3}[lane][2]
    target.mkdir()
    with pytest.raises(PermissionError, match="predecessor runtime lane"):
        compat._require_lane_separation()


def test_historical_or_c4_scientific_alias_fails_closed(
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


def test_disjoint_gate_freezes_complete_predecessor_alias_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = list(compat.FORBIDDEN_SCIENTIFIC_ALIASES)
    changed[4] = tmp_path / "retargeted-c3-alias"
    monkeypatch.setattr(
        compat,
        "FORBIDDEN_SCIENTIFIC_ALIASES",
        tuple(changed),
    )
    with pytest.raises(PermissionError, match="identity/path isolation"):
        compat._require_disjoint_c4_runtime_identity()


@pytest.mark.parametrize(
    "future_name",
    (
        "COMPAT_RUNTIME_SPEC_PATH",
        "COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH",
        "COMPAT_RUNTIME_ARTIFACT_ROOT",
        "COMPAT_GPU_LEASE_ROOT",
    ),
)
def test_build_phase_requires_every_c4_future_path_absent(
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
    with pytest.raises(PermissionError, match="before c4 launch"):
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
        or {"runtime_compatibility_id": "c4"},
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
        "runtime_compatibility_id": "c4",
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
    c4 = SimpleNamespace(name="c4")
    archival = {
        "authorization": {"unit_name": compat.COMPAT_UNIT},
        "receipt": {"unit_name": compat.COMPAT_UNIT},
    }
    normalized_old = {"name": "old"}
    normalized_c4 = {"name": "c4"}
    monkeypatch.setattr(
        compat.compat_environment,
        "replay_old_scope_and_handoff",
        lambda: (old, c4, {}),
        raising=False,
    )
    monkeypatch.setattr(
        compat,
        "_load_r4_archival",
        lambda: archival,
    )
    monkeypatch.setattr(
        compat.compat_environment,
        "validate_c4_realization_archival",
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
        "validate_c4_environment_closure",
        lambda *args, **kwargs: calls.append((*args, kwargs))
        or {"realization": archival},
        raising=False,
    )
    monkeypatch.setattr(
        compat.compat_environment,
        "_normalized_contract",
        lambda value: normalized_old if value is old else normalized_c4,
        raising=False,
    )
    receipt = {
        "historical_environment_contract": normalized_old,
        "current_environment_contract": normalized_c4,
    }
    closure = compat._verify_environment_cross_binding(receipt)
    assert closure["realization"] == archival
    assert len(calls) == 1
    assert calls[0][:5] == (
        {"path": str(compat.SCOPE_HANDOFF_PATH)},
        {"path": str(compat.STABILITY_ATTEMPT_PATH)},
        {"path": str(compat.POLICY_PATH)},
        {"path": str(compat.STABILITY_PATH)},
        {"path": str(compat.POSTCLEANUP_PATH)},
    )
    assert calls[0][-1]["archival"] == archival
    assert calls[0][-1]["c4_contract"] is c4


def test_release_closure_orders_bridge_before_frozen_and_archival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    archival = {
        "authorization": {"a": 1},
        "receipt": {"r": 1, "unit_path_policy": {}},
        "authorization_identity": {"path": "auth"},
        "receipt_identity": {"path": "receipt"},
        "compatibility_closure": {"passed": True},
    }
    monkeypatch.setattr(
        compat,
        "_require_l4_delegation",
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
            "integration": {},
            "realization": {
                "authorization": {"a": 1},
                "receipt": {"r": 1, "unit_path_policy": {}},
                "authorization_identity": {"path": "auth"},
                "receipt_identity": {"path": "receipt"},
            }
        },
    )
    monkeypatch.setattr(
        compat,
        "_require_final_r14_supervisor_binding",
        lambda _result: events.append("r14"),
    )
    snapshots = iter(("snapshot-before", "snapshot-after"))
    monkeypatch.setattr(
        compat,
        "_snapshot_production_c4",
        lambda **_kwargs: events.append(next(snapshots))
        or {
            "archival": deepcopy(archival),
            "fragment": {"inode": 1},
            "shadow": {"Id": compat.COMPAT_UNIT},
            "manager": {"boot_id": "same"},
            "live_unit_path_policy": {},
        },
    )
    monkeypatch.setattr(
        compat,
        "_validate_r14_runtime_chain",
        lambda _integration: events.append("r14-chain")
        or {"supervisor_evidence": {"invocation_id": "a" * 32}},
    )
    monkeypatch.setattr(
        compat.compat_environment,
        "_bind_r4_archival_root",
        lambda _path, value: value,
    )
    try:
        compat._compat_validate_release_closure()
    finally:
        compat._ACTIVE_COMMAND.reset(command_token)
    assert events == [
        "lane", "paths", "bridge", "hash", "sources",
        "snapshot-before", "frozen", "r14", "r14-chain",
        "snapshot-after",
    ]


def test_r14_binding_accepts_current_final_supervisor_generation() -> None:
    compat._require_final_r14_supervisor_binding(
        _valid_r14_release_closure()
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
def test_r14_binding_rejects_stale_supervisor_generation(
    field: str,
) -> None:
    closure = _valid_r14_release_closure()
    binding = closure["integration"]["authorization"][
        "executable_bindings"
    ]["supervisor"]
    if field in {"path", "resolved_path"}:
        binding[field] = "/tmp/stale-c4-supervisor.py"
    elif field == "file_sha256":
        binding[field] = "0" * 64
    else:
        binding[field] += 1
    with pytest.raises(PermissionError, match="r14 integration"):
        compat._require_final_r14_supervisor_binding(closure)


@pytest.mark.parametrize(
    "location",
    ("authorization", "receipt", "scenario_root"),
)
def test_r14_binding_rejects_wrong_fixed_root(location: str) -> None:
    closure = _valid_r14_release_closure()
    integration = closure["integration"]
    if location == "authorization":
        integration["identities"]["authorization"]["path"] = (
            "/tmp/stale-r14/control/authorization.json"
        )
    elif location == "receipt":
        integration["identities"]["receipt"]["path"] = (
            "/tmp/stale-r14/control/integration-receipt.json"
        )
    else:
        integration["authorization"]["scenario_root"]["path"] = (
            "/tmp/stale-r14"
        )
    with pytest.raises(PermissionError, match="r14 integration"):
        compat._require_final_r14_supervisor_binding(closure)


@pytest.mark.parametrize(
    "location",
    (
        "runtime_spec",
        "control_root",
        "runtime_root",
        "dummy_artifact",
        "integration_terminal_identity",
        "removal_authorization_identity",
        "removal_state_identity",
    ),
)
def test_r14_binding_rejects_any_of_six_control_or_fixed_root_drifts(
    location: str,
) -> None:
    closure = _valid_r14_release_closure()
    integration = closure["integration"]
    authorization = integration["authorization"]
    if location == "runtime_spec":
        authorization["runtime_spec_binding"]["path"] = "/tmp/runtime-spec.json"
    elif location == "control_root":
        authorization["control_root"]["path"] = "/tmp/control"
    elif location == "runtime_root":
        authorization["runtime_root"]["path"] = "/tmp/runtime"
    elif location == "dummy_artifact":
        authorization["control_artifacts"]["dummy_artifact"] = (
            "/tmp/dummy.json"
        )
    else:
        identity_name = location.removesuffix("_identity")
        integration["identities"][identity_name]["path"] = "/tmp/stale.json"
    with pytest.raises(PermissionError, match="r14 integration"):
        compat._require_final_r14_supervisor_binding(closure)


def test_r14_binding_rejects_non_single_link_current_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = _valid_r14_release_closure()
    identity = deepcopy(compat._SUPERVISOR_LOAD_IDENTITY)
    assert identity is not None
    identity["st_nlink"] = 2
    monkeypatch.setattr(compat, "_SUPERVISOR_LOAD_IDENTITY", identity)
    with pytest.raises(PermissionError, match="r14 integration"):
        compat._require_final_r14_supervisor_binding(closure)


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
        "_require_l4_delegation",
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


def test_r14_and_c4_environment_inputs_are_fixed() -> None:
    assert compat.INTEGRATION_ROOT.name.endswith("compat_c4_r14")
    assert "compat_c4" in compat.POLICY_PATH.name
    assert "compat_c4" in compat.STABILITY_PATH.name
    assert "compat_c4" in compat.POSTCLEANUP_PATH.name
    assert "compat_c4" in compat.REALIZATION_AUTHORIZATION_PATH.name
    assert "compat_c4" in compat.REALIZATION_RECEIPT_PATH.name


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
        lambda delegated_argv: compat._require_l4_delegation(
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
            compat._require_l4_delegation("authorize-launch")
    finally:
        compat._ACTIVE_COMMAND.reset(command_token)


def test_generator_late_only_inode_rotation_is_the_only_safe_policy_drift() -> None:
    before = _unit_path_policy(11)
    after = _unit_path_policy(12)
    compat.r14_integration._validate_unit_path_policy_transition(
        before,
        after,
        authorized_uid=os.getuid(),
        allow_generator_late_inode_rotation=True,
    )
    unsafe_mode = deepcopy(after)
    unsafe_mode["ordered_unit_paths"][0]["mode"] = 0o755
    with pytest.raises(PermissionError, match="more than its inode"):
        compat.r14_integration._validate_unit_path_policy_transition(
            before,
            unsafe_mode,
            authorized_uid=os.getuid(),
            allow_generator_late_inode_rotation=True,
        )
    unsafe_other_row = deepcopy(after)
    unsafe_other_row["ordered_unit_paths"][1]["inode"] += 1
    with pytest.raises(PermissionError, match="only one same-index"):
        compat.r14_integration._validate_unit_path_policy_transition(
            before,
            unsafe_other_row,
            authorized_uid=os.getuid(),
            allow_generator_late_inode_rotation=True,
        )


def test_release_closure_accepts_safe_generator_late_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _mock_release_snapshot(policy=_unit_path_policy(11))
    after = _mock_release_snapshot(policy=_unit_path_policy(12))
    result = _run_mocked_release_closure(
        monkeypatch,
        before=before,
        after=after,
    )
    assert result["realization"]["authorization"] == (
        before["archival"]["authorization"]
    )


def test_release_closure_rejects_r4_root_generation_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _mock_release_snapshot(policy=_unit_path_policy(11))
    after = _mock_release_snapshot(
        policy=_unit_path_policy(11),
        authorization_inode=999,
    )
    with pytest.raises(PermissionError, match="archival release closure"):
        _run_mocked_release_closure(
            monkeypatch,
            before=before,
            after=after,
        )


def test_release_closure_rejects_non_generator_policy_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _mock_release_snapshot(policy=_unit_path_policy(11))
    unsafe_policy = _unit_path_policy(12)
    unsafe_policy["ordered_unit_paths"][1]["inode"] += 1
    after = _mock_release_snapshot(policy=unsafe_policy)
    with pytest.raises(PermissionError, match="only one same-index"):
        _run_mocked_release_closure(
            monkeypatch,
            before=before,
            after=after,
        )


def test_production_snapshot_projects_exact_r4_fragment_and_static_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fragment_path = (tmp_path / "c4.service").absolute()
    fragment = {
        "path": str(fragment_path),
        "file_sha256": "a" * 64,
        "device": 1,
        "inode": 2,
        "owner_uid": os.getuid(),
        "mode": 0o600,
        "nlink": 1,
    }
    shadow = {
        "Id": compat.COMPAT_UNIT,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
        "InvocationID": "",
    }
    manager = {"boot_id": "exact-manager"}
    policy = _unit_path_policy(11)
    archival = {
        "authorization": {"unit_name": compat.COMPAT_UNIT},
        "receipt": {
            "fragment_identity": fragment,
            "full_static_shadow": shadow,
            "manager_generation": manager,
            "unit_path_policy": policy,
        },
        "authorization_identity": {"inode": 3},
        "receipt_identity": {"inode": 4},
        "compatibility_closure": {"passed": True},
    }
    monkeypatch.setattr(compat, "_load_r4_archival", lambda: archival)
    monkeypatch.setattr(
        compat.compat_realizer.compat_c1.legacy,
        "_stable_read_file",
        lambda _path: (
            b"fragment\n",
            {
                **fragment,
                "resolved_path": str(fragment_path),
                "path_is_symlink": False,
            },
        ),
    )
    monkeypatch.setattr(
        compat.compat_realizer.compat_c1.legacy,
        "validate_installed_shadow",
        lambda value, **_kwargs: dict(value),
    )
    observed = compat._snapshot_production_c4(
        shadow_reader=lambda: deepcopy(shadow),
        manager_reader=lambda: deepcopy(manager),
        unit_path_policy_reader=lambda _path: deepcopy(policy),
    )
    assert observed["fragment"] == fragment
    assert observed["shadow"] == shadow
    assert observed["manager"] == manager


def test_load_r4_archival_extends_real_sealed_identity_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = compat.compat_realizer.compat_c1.legacy
    authorization_path = (tmp_path / "authorization.json").absolute()
    receipt_path = (tmp_path / "receipt.json").absolute()
    terminal_path = (tmp_path / "terminal.json").absolute()
    closure_key = compat.compat_realizer.COMPATIBILITY_CLOSURE_KEY
    authorization_body = {closure_key: {"passed": True}}
    authorization = {
        **authorization_body,
        "authorization_fingerprint": legacy.stable_fingerprint(
            authorization_body
        ),
    }
    authorization_path.write_text(
        legacy.canonical_json(authorization) + "\n",
        encoding="utf-8",
    )
    authorization_path.chmod(0o444)
    _, authorization_identity = legacy._stable_read_file(
        authorization_path,
        private_parent=True,
    )
    receipt_body = {
        "authorization_file_sha256": authorization_identity["file_sha256"],
        "authorization_fingerprint": authorization[
            "authorization_fingerprint"
        ],
    }
    receipt = {
        **receipt_body,
        "receipt_fingerprint": legacy.stable_fingerprint(receipt_body),
    }
    receipt_path.write_text(
        legacy.canonical_json(receipt) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)
    monkeypatch.setattr(
        compat,
        "REALIZATION_AUTHORIZATION_PATH",
        authorization_path,
    )
    monkeypatch.setattr(compat, "REALIZATION_RECEIPT_PATH", receipt_path)
    monkeypatch.setattr(compat, "REALIZATION_TERMINAL_PATH", terminal_path)
    monkeypatch.setattr(
        compat.compat_realizer.compat_c1,
        "_validate_receipt_transitive_binding",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        compat.compat_realizer,
        "_validate_c4_receipt_contract",
        lambda value: value,
    )
    archival = compat._load_r4_archival()
    for name in ("authorization_identity", "receipt_identity"):
        root = archival[name]
        assert set(root) == compat.compat_environment._R4_ARCHIVAL_ROOT_FIELDS
        assert root["mode"] == 0o444
        assert root["nlink"] == 1
        assert root["parent_path"] == str(tmp_path.absolute())


def test_r14_preheartbeat_timeline_is_complete_and_current_time_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoNowDateTime(datetime):
        @classmethod
        def now(cls, *_args, **_kwargs):
            raise AssertionError("archival validation must not read current time")

    keys = (
        "launch_lease",
        "precommit_phase_receipt",
        "attempt_commit",
        "materialization_claim",
        "start_ack_receipt",
        "child_prespawn_phase_receipt",
    )
    artifacts = {key: f"/archived/{key}.json" for key in keys}
    payloads: dict[str, dict[str, object]] = {}
    for index, key in enumerate(keys, start=1):
        payloads[artifacts[key]] = {
            "attempt_id": "attempt-r14",
            "boot_id": "boot-r14",
            "time_utc": f"2020-01-01T00:00:0{index}Z",
            "monotonic_ns": index,
        }
    payloads[artifacts["attempt_commit"]]["systemd_unit_name"] = (
        compat.R14_DUMMY_UNIT
    )
    payloads[artifacts["materialization_claim"]][
        "systemd_control_group"
    ] = f"/user.slice/{compat.R14_DUMMY_UNIT}"
    monkeypatch.setattr(compat, "datetime", NoNowDateTime)
    monkeypatch.setattr(
        compat.r14_integration,
        "_read_sealed",
        lambda path, **_kwargs: deepcopy(payloads[str(path)]),
    )
    spec = {"attempt_id": "attempt-r14", "artifacts": artifacts}
    timeline = compat._validate_r14_preheartbeat_timeline(spec)
    assert tuple(timeline) == keys
    payloads[artifacts["start_ack_receipt"]]["monotonic_ns"] = 3
    with pytest.raises(PermissionError, match="preheartbeat chronology"):
        compat._validate_r14_preheartbeat_timeline(spec)


def test_r14_heartbeat_chain_is_continuous_hash_linked_and_archival(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoNowDateTime(datetime):
        @classmethod
        def now(cls, *_args, **_kwargs):
            raise AssertionError("archival validation must not read current time")

    heartbeat_root = tmp_path / "heartbeat"
    heartbeat_root.mkdir()
    first_path = heartbeat_root / "000000000000.json"
    second_path = heartbeat_root / "000000000001.json"
    first_path.touch()
    second_path.touch()
    terminal_path = tmp_path / "runtime-terminal.json"
    terminal_path.touch()
    claim = {
        "attempt_id": "attempt-r14",
        "boot_id": "boot-r14",
        "time_utc": "2020-01-01T00:00:04Z",
        "monotonic_ns": 4,
    }
    anchor = {
        "time_utc": "2020-01-01T00:00:06Z",
        "monotonic_ns": 6,
    }
    invocation_id = "a" * 32
    first = {
        "attempt_id": "attempt-r14",
        "boot_id": "boot-r14",
        "sequence": 0,
        "time_utc": "2020-01-01T00:00:07Z",
        "monotonic_ns": 7,
        "systemd_invocation_id": invocation_id,
        "previous_event_file_sha256": (
            compat.r14_integration.sealed_file_sha256(claim)
        ),
        "supervisor_pid": 10,
        "supervisor_proc_starttime_ticks": 20,
        "child_pid": 30,
        "child_proc_starttime_ticks": 40,
    }
    second = {
        **first,
        "sequence": 1,
        "time_utc": "2020-01-01T00:00:08Z",
        "monotonic_ns": 8,
        "previous_event_file_sha256": (
            compat.r14_integration.sealed_file_sha256(first)
        ),
    }
    terminal = {
        "attempt_id": "attempt-r14",
        "boot_id": "boot-r14",
        "systemd_invocation_id": invocation_id,
        "heartbeat_event_count": 2,
        "last_heartbeat_path": str(second_path),
        "last_heartbeat_file_sha256": (
            compat.r14_integration.sealed_file_sha256(second)
        ),
        "time_utc": "2020-01-01T00:00:09Z",
        "monotonic_ns": 9,
    }
    payloads = {
        str(first_path): first,
        str(second_path): second,
        str(terminal_path): terminal,
    }
    monkeypatch.setattr(compat, "datetime", NoNowDateTime)
    monkeypatch.setattr(
        compat.r14_integration,
        "_read_sealed",
        lambda path, **_kwargs: deepcopy(payloads[str(path)]),
    )
    spec = {
        "attempt_id": "attempt-r14",
        "artifacts": {
            "heartbeat_dir": str(heartbeat_root),
            "runtime_terminal": str(terminal_path),
        },
    }
    assert compat._validate_r14_heartbeat_chain(
        spec,
        invocation_id=invocation_id,
        claim=claim,
        chronology_anchor=anchor,
    ) == terminal
    payloads[str(second_path)]["previous_event_file_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="hash/time chain"):
        compat._validate_r14_heartbeat_chain(
            spec,
            invocation_id=invocation_id,
            claim=claim,
            chronology_anchor=anchor,
        )
    payloads[str(second_path)] = second
    second_path.rename(heartbeat_root / "000000000002.json")
    with pytest.raises(PermissionError, match="sequence is not continuous"):
        compat._validate_r14_heartbeat_chain(
            spec,
            invocation_id=invocation_id,
            claim=claim,
            chronology_anchor=anchor,
        )


def test_r14_archival_chronology_uses_no_current_time_and_rejects_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoNowDateTime(datetime):
        @classmethod
        def now(cls, *_args, **_kwargs):
            raise AssertionError("archival validation must not read current time")

    monkeypatch.setattr(compat, "datetime", NoNowDateTime)
    values = {
        "b4_receipt": {"created_at_utc": "2020-01-01T00:00:00Z"},
        "authorization": {
            "issued_at_utc": "2020-01-01T00:00:01Z",
            "expires_at_utc": "2020-01-01T00:04:59Z",
        },
        "timeline": {
            "launch_lease": {"time_utc": "2020-01-01T00:00:02Z"},
        },
        "runtime_terminal": {"time_utc": "2020-01-01T00:00:03Z"},
        "sidecar": {"time_utc": "2020-01-01T00:00:04Z"},
        "integration_terminal": {
            "created_at_utc": "2020-01-01T00:00:05Z",
        },
    }
    compat._validate_r14_archival_chronology(**values)
    reordered = deepcopy(values)
    reordered["sidecar"]["time_utc"] = "2020-01-01T00:00:02Z"
    with pytest.raises(PermissionError, match="archival authorization chronology"):
        compat._validate_r14_archival_chronology(**reordered)


def test_r14_structured_binding_ignores_legitimate_service_path_tokens() -> None:
    closure = _valid_r14_release_closure()
    authorization = closure["integration"]["authorization"]
    authorization["manager_generation"] = {
        "identity": {
            "control_group": (
                f"/user.slice/user-{os.getuid()}.slice/"
                f"user@{os.getuid()}.service/init.scope"
            ),
        },
    }
    authorization["template_binding"] = {
        "path": str(compat.R14_DUMMY_TEMPLATE_PATH),
    }
    compat._require_final_r14_supervisor_binding(closure)


def test_r14_source_binding_is_exact_and_inode_bound() -> None:
    binding = compat.r14_integration._file_binding(
        compat.R14_DUMMY_CHILD_PATH
    )
    compat._require_r14_source_binding(
        binding,
        path=compat.R14_DUMMY_CHILD_PATH,
        digest=compat.R14_DUMMY_CHILD_SHA256,
    )
    drifted = dict(binding)
    drifted["inode"] += 1
    with pytest.raises(PermissionError, match="source binding changed"):
        compat._require_r14_source_binding(
            drifted,
            path=compat.R14_DUMMY_CHILD_PATH,
            digest=compat.R14_DUMMY_CHILD_SHA256,
        )


def test_source_declares_no_retry_resume_or_dv_dt_authority() -> None:
    source = compat.COMPAT_BRIDGE_PATH.parent / (
        "cure_lite_v24_actual_runtime_release_preaccess_compat_c4.py"
    )
    text = source.read_text(encoding="utf-8")
    assert '"automatic_retry_allowed": False' in text
    assert '"resume_allowed": False' in text
    assert '"D_V_payload_accessed": False' in text
    assert '"D_T_payload_accessed": False' in text
    assert "systemctl" not in text
    assert "Popen" not in text
    assert "service_tokens" not in text
    assert "re.findall" not in text
