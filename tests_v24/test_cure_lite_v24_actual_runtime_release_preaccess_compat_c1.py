from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools import (
    cure_lite_v24_actual_runtime_release_preaccess_compat_c1 as compat,
)


def _seal(
    path: Path,
    body: dict[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    payload = {
        **body,
        fingerprint_field: compat.legacy.stable_fingerprint(body),
    }
    path.write_text(
        compat.legacy.canonical_json(payload) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return payload


def test_compat_release_identity_is_disjoint_and_transitively_bound() -> None:
    identity = compat.verify_compatibility_identity()
    assert identity["scientific_attempt_ordinal"] == 2
    assert identity["runtime_compatibility_generation"] == "c1"
    assert identity["fresh_scientific_attempt"] is False
    assert identity["automatic_retry_allowed"] is False
    assert identity["resume_allowed"] is False
    assert identity["authoritative_access_audit_schema"] == (
        "cure-lite-v24-split-access-audit-v1"
    )
    assert identity["fictional_access_audit_schema_accepted"] is False
    assert identity["frozen_release_file_sha256"] == (
        compat.FROZEN_RELEASE_SHA256
    )
    assert compat._sha256_file(compat.FROZEN_RELEASE_PATH) == (
        compat.FROZEN_RELEASE_SHA256
    )
    assert compat.COMPAT_RUNTIME_SPEC_PATH != (
        compat.BLOCKED_RUNTIME_SPEC_PATH
    )
    assert compat.COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH != (
        compat.BLOCKED_RUNTIME_LAUNCH_AUTHORIZATION_PATH
    )
    assert compat.COMPAT_RUNTIME_ARTIFACT_ROOT != (
        compat.BLOCKED_RUNTIME_ARTIFACT_ROOT
    )
    assert compat.COMPAT_GPU_LEASE_ROOT != (
        compat.BLOCKED_GPU_LEASE_ROOT
    )


def test_frozen_release_same_bytes_inode_replacement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (tmp_path / "frozen-release.py").absolute()
    raw = b"VALUE = 1\n"
    source.write_bytes(raw)
    _verified, identity = compat._stable_source_bytes(source)
    monkeypatch.setattr(compat, "FROZEN_RELEASE_PATH", source)
    monkeypatch.setattr(
        compat,
        "FROZEN_RELEASE_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "_FROZEN_RELEASE_LOAD_IDENTITY",
        identity,
    )
    compat._require_frozen_release_source()

    replacement = tmp_path / "replacement.py"
    replacement.write_bytes(raw)
    replacement.replace(source)
    with pytest.raises(
        PermissionError,
        match="generation was replaced",
    ):
        compat._require_frozen_release_source()


def test_frozen_release_executes_the_exact_verified_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"VERIFIED_SENTINEL = 17\n"
    identity = {"generation": 1}
    calls: list[Path] = []

    def verified_once(path: Path):
        calls.append(Path(path))
        return raw, identity

    monkeypatch.setattr(
        compat,
        "FROZEN_RELEASE_PATH",
        Path("/fixed/frozen-release.py"),
    )
    monkeypatch.setattr(
        compat,
        "FROZEN_RELEASE_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "_stable_source_bytes",
        verified_once,
    )
    module_name = (
        "tools._cure_lite_v24_actual_runtime_release_frozen_"
        "for_preaccess_compat_c1"
    )
    prior = sys.modules.get(module_name)
    try:
        loaded, observed_identity = compat._load_frozen_release()
    finally:
        if prior is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior
    assert calls == [Path("/fixed/frozen-release.py")]
    assert loaded.VERIFIED_SENTINEL == 17
    assert observed_identity == identity


def test_actual_spec_write_uses_exact_approved_preview_after_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"schema_version": "preview-test"}
    preview = compat._materialize_spec_body(
        compat.COMPAT_RUNTIME_SPEC_PATH,
        body,
        fingerprint_field="runtime_spec_fingerprint",
    )
    calls: list[tuple[Path, dict[str, object], str]] = []
    monkeypatch.setattr(
        compat.compat_supervisor,
        "validate_prewrite_spec",
        lambda _payload: pytest.fail(
            "post-directory writer must not replay future-absence policy"
        ),
    )
    monkeypatch.setattr(
        compat,
        "_frozen_write_sealed",
        lambda path, materialized, *, fingerprint_field: calls.append(
            (Path(path), dict(materialized), fingerprint_field)
        )
        or preview,
    )
    token = compat._APPROVED_PREWRITE_SPEC.set(preview)
    try:
        result = compat._compat_write_sealed(
            compat.COMPAT_RUNTIME_SPEC_PATH,
            body,
            fingerprint_field="runtime_spec_fingerprint",
        )
    finally:
        compat._APPROVED_PREWRITE_SPEC.reset(token)
    assert result == preview
    assert calls == [
        (
            compat.COMPAT_RUNTIME_SPEC_PATH,
            body,
            "runtime_spec_fingerprint",
        )
    ]


def test_actual_spec_write_rejects_any_preview_drift_before_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"schema_version": "preview-test"}
    preview = compat._materialize_spec_body(
        compat.COMPAT_RUNTIME_SPEC_PATH,
        body,
        fingerprint_field="runtime_spec_fingerprint",
    )
    drifted = dict(preview)
    drifted["schema_version"] = "drift"
    monkeypatch.setattr(
        compat,
        "_frozen_write_sealed",
        lambda *_args, **_kwargs: pytest.fail(
            "drifted spec must not be written"
        ),
    )
    token = compat._APPROVED_PREWRITE_SPEC.set(drifted)
    try:
        with pytest.raises(
            PermissionError,
            match="differs from its mutation-free preview",
        ):
            compat._compat_write_sealed(
                compat.COMPAT_RUNTIME_SPEC_PATH,
                body,
                fingerprint_field="runtime_spec_fingerprint",
            )
    finally:
        compat._APPROVED_PREWRITE_SPEC.reset(token)


def test_only_fixed_authoritative_v1_audit_projects_the_frozen_literal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = tmp_path / "audit.json"
    audit_path = audit.absolute()
    monkeypatch.setattr(
        compat.legacy,
        "SCIENTIFIC_ACCESS_AUDIT_PATH",
        audit_path,
    )
    expected = _seal(
        audit,
        {
            "schema_version": (
                "cure-lite-v24-split-access-audit-v1"
            ),
        },
        fingerprint_field="receipt_fingerprint",
    )
    loaded = compat._compat_load_sealed(
        audit_path,
        fingerprint_field="receipt_fingerprint",
        schema="cure-lite-v24-split-access-audit-r2-v1",
    )
    assert loaded == expected

    audit.chmod(0o600)
    audit.unlink()
    _seal(
        audit,
        {
            "schema_version": (
                "cure-lite-v24-split-access-audit-r2-v1"
            ),
        },
        fingerprint_field="receipt_fingerprint",
    )
    with pytest.raises(
        PermissionError,
        match="fingerprint/schema changed",
    ):
        compat._compat_load_sealed(
            audit_path,
            fingerprint_field="receipt_fingerprint",
            schema="cure-lite-v24-split-access-audit-r2-v1",
        )


def test_schema_projection_is_never_applied_to_an_off_path_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = (tmp_path / "fixed-audit.json").absolute()
    decoy = (tmp_path / "decoy-audit.json").absolute()
    monkeypatch.setattr(
        compat.legacy,
        "SCIENTIFIC_ACCESS_AUDIT_PATH",
        fixed,
    )
    _seal(
        decoy,
        {
            "schema_version": (
                "cure-lite-v24-split-access-audit-v1"
            ),
        },
        fingerprint_field="receipt_fingerprint",
    )
    with pytest.raises(
        PermissionError,
        match="fingerprint/schema changed",
    ):
        compat._compat_load_sealed(
            decoy,
            fingerprint_field="receipt_fingerprint",
            schema="cure-lite-v24-split-access-audit-r2-v1",
        )


@pytest.mark.parametrize(
    "blocked_name",
    [
        "BLOCKED_RUNTIME_SPEC_PATH",
        "BLOCKED_RUNTIME_LAUNCH_AUTHORIZATION_PATH",
        "BLOCKED_RUNTIME_ARTIFACT_ROOT",
        "BLOCKED_GPU_LEASE_ROOT",
    ],
)
def test_any_blocked_lane_materialization_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_name: str,
) -> None:
    blocked = tmp_path / blocked_name
    monkeypatch.setattr(compat, blocked_name, blocked)
    blocked.write_text("blocked\n", encoding="utf-8")
    with pytest.raises(
        PermissionError,
        match="predecessor runtime lane",
    ):
        compat._require_lane_separation()


@pytest.mark.parametrize(
    "alias_name",
    [
        "COMPAT_RUN_ROOT_ALIAS_PATH",
        "COMPAT_RESULT_RECEIPT_ALIAS_PATH",
    ],
)
def test_any_compat_scientific_alias_is_always_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_name: str,
) -> None:
    alias = tmp_path / alias_name
    monkeypatch.setattr(compat, alias_name, alias)
    alias.write_text("forbidden alias\n", encoding="utf-8")
    with pytest.raises(
        PermissionError,
        match="scientific alias",
    ):
        compat._require_lane_separation()


@pytest.mark.parametrize(
    "path_name",
    [
        "SCIENTIFIC_RUN_ROOT",
        "SCIENTIFIC_RESULT_RECEIPT_PATH",
    ],
)
def test_prelaunch_scientific_output_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_name: str,
) -> None:
    path = tmp_path / path_name
    monkeypatch.setattr(compat, path_name, path)
    path.write_text("premature scientific output\n", encoding="utf-8")
    with pytest.raises(
        PermissionError,
        match="exists before launch authorization",
    ):
        compat._require_prelaunch_scientific_pristine()


def test_build_phase_requires_all_runtime_namespaces_absent_and_unbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "COMPAT_RUNTIME_SPEC_PATH",
        "COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH",
        "COMPAT_RUNTIME_ARTIFACT_ROOT",
        "COMPAT_GPU_LEASE_ROOT",
    ):
        monkeypatch.setattr(compat, name, tmp_path / name)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        compat,
        "_verify_compatibility_receipt",
        lambda **kwargs: calls.append(
            (
                kwargs["expected_spec"],
                kwargs["require_spec_binding"],
            )
        )
        or {"passed": True},
    )
    token = compat._ACTIVE_COMMAND.set("build-spec")
    try:
        assert compat._verify_active_phase_compatibility_receipt() == {
            "passed": True
        }
    finally:
        compat._ACTIVE_COMMAND.reset(token)
    assert calls == [(None, False)]


@pytest.mark.parametrize(
    "future_name",
    [
        "COMPAT_RUNTIME_SPEC_PATH",
        "COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH",
        "COMPAT_RUNTIME_ARTIFACT_ROOT",
        "COMPAT_GPU_LEASE_ROOT",
    ],
)
def test_build_phase_rejects_any_early_runtime_materialization(
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
    early = getattr(compat, future_name)
    early.write_text("early\n", encoding="utf-8")
    token = compat._ACTIVE_COMMAND.set("build-spec")
    try:
        with pytest.raises(
            PermissionError,
            match="already materialized",
        ):
            compat._verify_active_phase_compatibility_receipt()
    finally:
        compat._ACTIVE_COMMAND.reset(token)


def test_authorize_phase_reads_only_fixed_sealed_spec_and_requires_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = (tmp_path / "fixed-runtime-spec.json").absolute()
    expected = _seal(
        fixed,
        {
            "schema_version": (
                compat.compat_supervisor.RUNTIME_SPEC_SCHEMA
            ),
        },
        fingerprint_field="runtime_spec_fingerprint",
    )
    monkeypatch.setattr(compat, "COMPAT_RUNTIME_SPEC_PATH", fixed)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        compat,
        "_verify_compatibility_receipt",
        lambda **kwargs: calls.append(
            (
                kwargs["expected_spec"],
                kwargs["require_spec_binding"],
            )
        )
        or {"passed": True},
    )
    token = compat._ACTIVE_COMMAND.set("authorize-launch")
    try:
        assert compat._verify_active_phase_compatibility_receipt() == {
            "passed": True
        }
    finally:
        compat._ACTIVE_COMMAND.reset(token)
    assert calls == [(expected, True)]


def test_authorize_phase_never_accepts_a_valid_off_path_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = (tmp_path / "missing-fixed-runtime-spec.json").absolute()
    decoy = (tmp_path / "decoy-runtime-spec.json").absolute()
    _seal(
        decoy,
        {
            "schema_version": (
                compat.compat_supervisor.RUNTIME_SPEC_SCHEMA
            ),
        },
        fingerprint_field="runtime_spec_fingerprint",
    )
    monkeypatch.setattr(compat, "COMPAT_RUNTIME_SPEC_PATH", fixed)
    token = compat._ACTIVE_COMMAND.set("authorize-launch")
    try:
        with pytest.raises(FileNotFoundError):
            compat._verify_active_phase_compatibility_receipt()
    finally:
        compat._ACTIVE_COMMAND.reset(token)


def test_bridge_verifier_receives_exact_phase_binding_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_spec = {"runtime_spec_fingerprint": "a" * 64}
    calls: list[tuple[object, object, object]] = []
    result = {
        "passed": True,
        "compatibility_source_roots": {
            "compat_policy": {
                "path": str(compat.COMPAT_POLICY_SOURCE_PATH),
                "file_sha256": compat.COMPAT_POLICY_SOURCE_SHA256,
            },
        },
    }
    policy = SimpleNamespace(
        COMPATIBILITY_RECEIPT_PATH=compat.COMPATIBILITY_RECEIPT_PATH,
        verify_compatibility_receipt=(
            lambda path, **kwargs: calls.append(
                (
                    path,
                    kwargs["expected_spec"],
                    kwargs["require_spec_binding"],
                )
            )
            or result
        ),
    )
    identity = {"generation": 1}
    monkeypatch.setattr(
        compat,
        "_load_verified_compatibility_policy",
        lambda: (policy, identity),
    )
    monkeypatch.setattr(
        compat,
        "_stable_source_bytes",
        lambda path: (
            b"",
            identity,
        )
        if Path(path) == compat.COMPAT_POLICY_SOURCE_PATH
        else pytest.fail(f"unexpected stable read: {path}"),
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
        )
    ]


def test_cli_rejects_off_path_build_inputs_before_frozen_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = [
        "build-spec",
        "--environment-policy", "/tmp/off-path-policy.json",
        "--precleanup-receipt", str(compat.PRECLEANUP_PATH),
        "--cleanup-plan", str(compat.CLEANUP_PLAN_PATH),
        "--cleanup-authorization",
        str(compat.CLEANUP_AUTHORIZATION_PATH),
        "--cleanup-receipt", str(compat.CLEANUP_RECEIPT_PATH),
        "--stability-receipt", str(compat.STABILITY_PATH),
        "--postcleanup-audit", str(compat.POSTCLEANUP_PATH),
        "--integration-authorization",
        str(compat.INTEGRATION_AUTHORIZATION_PATH),
        "--integration-receipt",
        str(compat.INTEGRATION_RECEIPT_PATH),
        "--unit-realization-authorization",
        str(compat.REALIZATION_AUTHORIZATION_PATH),
        "--unit-realization-receipt",
        str(compat.REALIZATION_RECEIPT_PATH),
    ]
    monkeypatch.setattr(
        compat.legacy,
        "main",
        lambda _argv: pytest.fail("frozen main must not run"),
    )
    with pytest.raises(
        PermissionError,
        match="input path changed: policy_path",
    ):
        compat.main(argv)


def test_frozen_hash_is_rechecked_immediately_before_delegation(
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
        compat.legacy,
        "_parser",
        lambda: SimpleNamespace(
            parse_args=lambda _argv: args,
        ),
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
        lambda _argv: events.append("frozen") or 0,
    )
    assert compat.main(["authorize-launch"]) == 0
    assert events[-2:] == ["hash", "frozen"]


def test_release_closure_checks_bridge_before_frozen_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        compat,
        "_require_lane_separation",
        lambda: events.append("lane"),
    )
    monkeypatch.setattr(
        compat,
        "_require_fixed_inputs",
        lambda values: events.append("paths"),
    )
    monkeypatch.setattr(
        compat,
        "_verify_active_phase_compatibility_receipt",
        lambda: events.append("bridge") or {"passed": True},
    )
    monkeypatch.setattr(
        compat,
        "_require_frozen_release_source",
        lambda: events.append("hash"),
    )
    monkeypatch.setattr(
        compat,
        "_frozen_validate_release_closure",
        lambda **kwargs: events.append("frozen")
        or {
            "passed": True,
            "realization": {
                "authorization": {"auth": 1},
                "receipt": {"receipt": 1},
            },
        },
    )
    monkeypatch.setattr(
        compat.compat_realizer,
        "validate_archival_realization_chain",
        lambda *_args, **_kwargs: events.append("archival")
        or {
            "authorization": {"auth": 1},
            "receipt": {"receipt": 1},
            "compatibility_closure": {"passed": True},
        },
    )
    result = compat._compat_validate_release_closure()
    assert result["passed"] is True
    assert events == [
        "lane",
        "paths",
        "bridge",
        "hash",
        "frozen",
        "archival",
    ]
