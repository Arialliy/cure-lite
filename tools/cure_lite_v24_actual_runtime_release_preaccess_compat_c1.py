#!/usr/bin/env python3
"""Create-only release producer for r2 runtime compatibility generation c1.

This lane does not create a new scientific attempt.  It verifies the sealed
prewrite schema-mismatch closure, consumes the authoritative generic-v1
preaccess audit directly, and delegates every other release check to the
frozen release implementation.  All runtime, unit, artifact, authorization,
and GPU-lease paths are disjoint from the blocked predecessor lane.
"""

from __future__ import annotations

from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()

FROZEN_RELEASE_PATH = (
    REPOSITORY / "tools/cure_lite_v24_actual_runtime_release.py"
).resolve()
FROZEN_RELEASE_SHA256 = (
    "258dcae12a7799ccf63a39dd191fce67170728fb38a705223c0bc1c9fd1b387d"
)
COMPAT_SUPERVISOR_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c1.py"
).resolve()
COMPAT_ADAPTER_PATH = (
    REPOSITORY
    / "tools/"
    "run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c1.py"
).resolve()
COMPAT_REALIZER_PATH = (
    REPOSITORY
    / "tools/"
    "cure_lite_v24_actual_unit_realization_preaccess_compat_c1.py"
).resolve()
COMPAT_REALIZER_SHA256 = (
    "7bfc5944378d552f9f12654da5234762452f8dc5ee49f1bced47554bcbd58ece"
)
COMPAT_SUPERVISOR_SHA256 = (
    "7e2182da4f818bda5567c677194a1daf4cf02ce6874754acbc6b42095bd77447"
)

COMPAT_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service"
)
COMPAT_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json"
)
COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c1_"
        "runtime_launch_authorization.json"
    )
)
COMPAT_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_artifacts"
)
COMPAT_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_gpu_lease"
)
COMPAT_RUN_ROOT_ALIAS_PATH = (
    REPOSITORY
    / "runs/irstd1k_stage_a_seed42/"
    "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c1"
)
COMPAT_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_receipt.json"
)
SCIENTIFIC_RUN_ROOT = (
    REPOSITORY
    / "runs/irstd1k_stage_a_seed42/"
    "gcr_pacre_v24_D_R_structural_attempt_r2"
)
SCIENTIFIC_RESULT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_receipt.json"
)
COMPATIBILITY_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_receipt.json"
)
COMPAT_POLICY_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility.py"
).resolve()
COMPAT_POLICY_SOURCE_SHA256 = (
    "fe715af48867f166d2e15727e0190844cfd79fb5c02fa5a440d294bb7f29e084"
)

POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c1.json"
)
PRECLEANUP_PATH = (
    EVIDENCE_ROOT / "runtime_environment_precleanup_receipt.json"
)
CLEANUP_PLAN_PATH = EVIDENCE_ROOT / "environment_cleanup_plan.json"
CLEANUP_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "environment_cleanup_authorization.json"
)
CLEANUP_RECEIPT_PATH = (
    EVIDENCE_ROOT / "environment_cleanup_recovery_r1/cleanup-receipt.json"
)
STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c1.json"
)
POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c1.json"
)
INTEGRATION_ROOT = (
    EVIDENCE_ROOT
    / "supervisor_v2_systemd_integration_preaccess_compat_c1_r10"
)
INTEGRATION_AUTHORIZATION_PATH = (
    INTEGRATION_ROOT / "control/authorization.json"
)
INTEGRATION_RECEIPT_PATH = (
    INTEGRATION_ROOT / "control/integration-receipt.json"
)
REALIZATION_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c1_unit_realization_authorization.json"
)
REALIZATION_RECEIPT_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c1_unit_realization_receipt.json"
)
REALIZATION_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c1_unit_realization_terminal.json"
)

BLOCKED_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_spec.json"
)
BLOCKED_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_runtime_launch_authorization.json"
)
BLOCKED_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_artifacts"
)
BLOCKED_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_gpu_lease"
)

AUTHORITATIVE_ACCESS_AUDIT_SCHEMA = (
    "cure-lite-v24-split-access-audit-v1"
)
FICTIONAL_ACCESS_AUDIT_SCHEMA = (
    "cure-lite-v24-split-access-audit-r2-v1"
)
_ALLOWED_COMMANDS = frozenset({"build-spec", "authorize-launch"})
_ACTIVE_COMMAND: ContextVar[str | None] = ContextVar(
    "cure_lite_v24_compat_release_command",
    default=None,
)
_APPROVED_PREWRITE_SPEC: ContextVar[
    Mapping[str, object] | None
] = ContextVar(
    "cure_lite_v24_compat_release_approved_prewrite_spec",
    default=None,
)


_FILE_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_PARENT_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
    "st_nlink",
)


def _stable_source_bytes(
    path: Path,
) -> tuple[bytes, dict[str, int]]:
    """Read one exact regular-file generation through one no-follow fd."""

    target = Path(path).absolute()
    parent = target.parent
    parent_before = parent.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or parent.resolve(strict=True) != parent
        or target.resolve(strict=True) != target
    ):
        raise PermissionError("compatibility release source is unsafe")
    directory_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        parent_opened = os.fstat(directory_fd)
        before = os.stat(
            target.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
        ):
            raise PermissionError(
                "compatibility release source is unsafe"
            )
        descriptor = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = os.stat(
            target.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        parent_after = os.fstat(directory_fd)
    finally:
        os.close(directory_fd)
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(after, field)
        or getattr(after, field) != getattr(path_after, field)
        for field in _FILE_IDENTITY_FIELDS
    ) or any(
        getattr(parent_before, field)
        != getattr(parent_opened, field)
        or getattr(parent_opened, field)
        != getattr(parent_after, field)
        for field in _PARENT_IDENTITY_FIELDS
    ):
        raise PermissionError("compatibility release source changed")
    raw = b"".join(chunks)
    identity = {
        field: int(getattr(path_after, field))
        for field in _FILE_IDENTITY_FIELDS
    }
    identity.update({
        f"parent_{field}": int(getattr(parent_after, field))
        for field in _PARENT_IDENTITY_FIELDS
    })
    return raw, identity


def _sha256_file(path: Path) -> str:
    raw, _identity = _stable_source_bytes(path)
    return hashlib.sha256(raw).hexdigest()


def _require_frozen_release_source() -> None:
    raw, identity = _stable_source_bytes(FROZEN_RELEASE_PATH)
    if hashlib.sha256(raw).hexdigest() != FROZEN_RELEASE_SHA256:
        raise PermissionError("frozen runtime release source changed")
    loaded_identity = globals().get("_FROZEN_RELEASE_LOAD_IDENTITY")
    if (
        loaded_identity is not None
        and identity != loaded_identity
    ):
        raise PermissionError(
            "frozen runtime release generation was replaced"
        )


def _load_frozen_release() -> tuple[ModuleType, dict[str, int]]:
    raw, identity = _stable_source_bytes(FROZEN_RELEASE_PATH)
    if hashlib.sha256(raw).hexdigest() != FROZEN_RELEASE_SHA256:
        raise PermissionError("frozen runtime release source changed")
    name = (
        "tools._cure_lite_v24_actual_runtime_release_frozen_"
        "for_preaccess_compat_c1"
    )
    module = ModuleType(name)
    module.__file__ = str(FROZEN_RELEASE_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                raw,
                str(FROZEN_RELEASE_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, identity


legacy, _FROZEN_RELEASE_LOAD_IDENTITY = _load_frozen_release()


def _load_verified_compat_module(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> tuple[ModuleType, dict[str, int]]:
    raw, identity = _stable_source_bytes(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError(
            "compatibility runtime component source changed"
        )
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                raw,
                str(path),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _raw_after, identity_after = _stable_source_bytes(path)
    if identity_after != identity:
        sys.modules.pop(name, None)
        raise PermissionError(
            "compatibility runtime component generation changed"
        )
    return module, identity


compat_realizer, _COMPAT_REALIZER_LOAD_IDENTITY = (
    _load_verified_compat_module(
        COMPAT_REALIZER_PATH,
        COMPAT_REALIZER_SHA256,
        name=(
            "tools._cure_lite_v24_actual_unit_realization_"
            "preaccess_compat_c1_verified_for_release"
        ),
    )
)
compat_supervisor, _COMPAT_SUPERVISOR_LOAD_IDENTITY = (
    _load_verified_compat_module(
        COMPAT_SUPERVISOR_PATH,
        COMPAT_SUPERVISOR_SHA256,
        name=(
            "tools._cure_lite_v24_runtime_supervisor_"
            "preaccess_compat_c1_verified_for_release"
        ),
    )
)


_frozen_load_sealed = legacy._load_sealed
_frozen_validate_release_closure = legacy.validate_release_closure
_frozen_write_sealed = legacy._write_sealed
_frozen_create_runtime_directories = (
    legacy._create_runtime_directories_and_verify_leaves
)


def _load_verified_compatibility_policy() -> tuple[
    ModuleType,
    dict[str, int],
]:
    raw, identity = _stable_source_bytes(COMPAT_POLICY_SOURCE_PATH)
    if hashlib.sha256(raw).hexdigest() != COMPAT_POLICY_SOURCE_SHA256:
        raise PermissionError(
            "compatibility policy source changed"
        )
    name = (
        "tools._cure_lite_v24_preaccess_schema_compatibility_"
        "verified_for_runtime_release_c1"
    )
    module = ModuleType(name)
    module.__file__ = str(COMPAT_POLICY_SOURCE_PATH)
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(COMPAT_POLICY_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    _raw_after, identity_after = _stable_source_bytes(
        COMPAT_POLICY_SOURCE_PATH
    )
    if identity_after != identity:
        raise PermissionError(
            "compatibility policy generation changed while executing"
        )
    return module, identity


def _compat_load_sealed(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str | None = None,
    canonical_required: bool = True,
    return_identity: bool = False,
    private_parent: bool = False,
):
    _require_frozen_release_source()
    target = Path(path).absolute()
    if (
        target == legacy.SCIENTIFIC_ACCESS_AUDIT_PATH.absolute()
        and fingerprint_field == "receipt_fingerprint"
        and schema == FICTIONAL_ACCESS_AUDIT_SCHEMA
    ):
        schema = AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
    return _frozen_load_sealed(
        target,
        fingerprint_field=fingerprint_field,
        schema=schema,
        canonical_required=canonical_required,
        return_identity=return_identity,
        private_parent=private_parent,
    )


def _verify_compatibility_receipt(
    *,
    expected_spec: Mapping[str, object] | None = None,
    require_spec_binding: bool,
) -> Mapping[str, object]:
    policy, policy_generation = _load_verified_compatibility_policy()

    policy_path = getattr(
        policy,
        "COMPATIBILITY_RECEIPT_PATH",
        getattr(policy, "RECEIPT_PATH", None),
    )
    if (
        policy_path is None
        or Path(policy_path).absolute()
        != COMPATIBILITY_RECEIPT_PATH.absolute()
    ):
        raise PermissionError(
            "compatibility receipt path interface changed",
        )
    verifier = getattr(policy, "verify_compatibility_receipt", None)
    if not callable(verifier):
        raise PermissionError(
            "compatibility receipt verifier is unavailable",
        )
    result = verifier(
        COMPATIBILITY_RECEIPT_PATH,
        expected_spec=(
            None if expected_spec is None else dict(expected_spec)
        ),
        require_spec_binding=require_spec_binding,
    )
    if not isinstance(result, Mapping):
        raise PermissionError("compatibility receipt is not verified")
    source_roots = result.get("compatibility_source_roots")
    policy_root = (
        source_roots.get("compat_policy")
        if isinstance(source_roots, Mapping)
        else None
    )
    _raw_after, policy_generation_after = _stable_source_bytes(
        COMPAT_POLICY_SOURCE_PATH
    )
    if (
        not isinstance(policy_root, Mapping)
        or policy_root.get("path") != str(COMPAT_POLICY_SOURCE_PATH)
        or policy_root.get("file_sha256")
        != COMPAT_POLICY_SOURCE_SHA256
        or policy_generation_after != policy_generation
    ):
        raise PermissionError(
            "compatibility receipt policy source binding changed"
        )
    return result


def _require_lane_separation() -> None:
    blocked = (
        BLOCKED_RUNTIME_SPEC_PATH,
        BLOCKED_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        BLOCKED_RUNTIME_ARTIFACT_ROOT,
        BLOCKED_GPU_LEASE_ROOT,
    )
    if any(os.path.lexists(path) for path in blocked):
        raise PermissionError(
            "blocked predecessor runtime lane is no longer pristine",
        )
    aliases = (
        COMPAT_RUN_ROOT_ALIAS_PATH,
        COMPAT_RESULT_RECEIPT_ALIAS_PATH,
    )
    if any(os.path.lexists(path) for path in aliases):
        raise PermissionError(
            "compatibility scientific alias is forbidden",
        )


def _require_build_phase_pristine() -> None:
    future_runtime_paths = (
        COMPAT_RUNTIME_SPEC_PATH,
        COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        COMPAT_RUNTIME_ARTIFACT_ROOT,
        COMPAT_GPU_LEASE_ROOT,
    )
    if any(os.path.lexists(path) for path in future_runtime_paths):
        raise PermissionError(
            "compatibility build-spec namespace is already materialized",
        )


def _require_prelaunch_scientific_pristine() -> None:
    if any(
        os.path.lexists(path)
        for path in (
            SCIENTIFIC_RUN_ROOT,
            SCIENTIFIC_RESULT_RECEIPT_PATH,
        )
    ):
        raise PermissionError(
            "scientific r2 output exists before launch authorization"
        )


def _load_fixed_compat_runtime_spec() -> dict[str, object]:
    _require_frozen_release_source()
    payload = _frozen_load_sealed(
        COMPAT_RUNTIME_SPEC_PATH,
        fingerprint_field="runtime_spec_fingerprint",
        schema=compat_supervisor.RUNTIME_SPEC_SCHEMA,
    )
    if not isinstance(payload, Mapping):
        raise PermissionError(
            "fixed compatibility runtime spec is not a mapping",
        )
    return dict(payload)


def _verify_active_phase_compatibility_receipt() -> Mapping[str, object]:
    command = _ACTIVE_COMMAND.get()
    if command == "build-spec":
        _require_build_phase_pristine()
        return _verify_compatibility_receipt(
            expected_spec=None,
            require_spec_binding=False,
        )
    if command == "authorize-launch":
        _require_prelaunch_scientific_pristine()
        spec = _load_fixed_compat_runtime_spec()
        return _verify_compatibility_receipt(
            expected_spec=spec,
            require_spec_binding=True,
        )
    raise PermissionError(
        "compatibility release phase is not explicitly selected",
    )


def _require_fixed_inputs(values: Mapping[str, object]) -> None:
    expected = {
        "policy_path": POLICY_PATH,
        "precleanup_path": PRECLEANUP_PATH,
        "cleanup_plan_path": CLEANUP_PLAN_PATH,
        "cleanup_authorization_path": CLEANUP_AUTHORIZATION_PATH,
        "cleanup_receipt_path": CLEANUP_RECEIPT_PATH,
        "stability_path": STABILITY_PATH,
        "postcleanup_path": POSTCLEANUP_PATH,
        "integration_authorization_path": (
            INTEGRATION_AUTHORIZATION_PATH
        ),
        "integration_receipt_path": INTEGRATION_RECEIPT_PATH,
        "realization_authorization_path": (
            REALIZATION_AUTHORIZATION_PATH
        ),
        "realization_receipt_path": REALIZATION_RECEIPT_PATH,
    }
    for name, expected_path in expected.items():
        supplied = values.get(name)
        if (
            not isinstance(supplied, Path)
            or supplied.absolute() != expected_path.absolute()
        ):
            raise PermissionError(
                f"compatibility release input path changed: {name}",
            )


def _compat_validate_release_closure(**kwargs):
    _require_lane_separation()
    _require_fixed_inputs(kwargs)
    _verify_active_phase_compatibility_receipt()
    _require_frozen_release_source()
    result = _frozen_validate_release_closure(**kwargs)
    archival = compat_realizer.validate_archival_realization_chain(
        REALIZATION_AUTHORIZATION_PATH,
        REALIZATION_RECEIPT_PATH,
    )
    realization = result.get("realization")
    if (
        not isinstance(realization, Mapping)
        or not isinstance(archival, Mapping)
        or not legacy._deep_exact_equal(
            realization.get("authorization"),
            archival.get("authorization"),
        )
        or not legacy._deep_exact_equal(
            realization.get("receipt"),
            archival.get("receipt"),
        )
        or not isinstance(
            archival.get("compatibility_closure"),
            Mapping,
        )
    ):
        raise PermissionError(
            "compatibility realization archival closure diverged"
        )
    return result


def _materialize_spec_body(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    if (
        Path(path).absolute() != COMPAT_RUNTIME_SPEC_PATH.absolute()
        or fingerprint_field != "runtime_spec_fingerprint"
        or "runtime_spec_fingerprint" in body
    ):
        raise PermissionError(
            "compatibility prewrite spec output contract changed"
        )
    return {
        **dict(body),
        "runtime_spec_fingerprint": legacy.stable_fingerprint(body),
    }


def _validate_prewrite_spec_body(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    payload = _materialize_spec_body(
        path,
        body,
        fingerprint_field=fingerprint_field,
    )
    compat_supervisor.validate_prewrite_spec(payload)
    return payload


def _compat_write_sealed(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    payload = _materialize_spec_body(
        path,
        body,
        fingerprint_field=fingerprint_field,
    )
    preview = _APPROVED_PREWRITE_SPEC.get()
    if (
        not isinstance(preview, Mapping)
        or not legacy._deep_exact_equal(payload, dict(preview))
    ):
        raise PermissionError(
            "runtime spec differs from its mutation-free preview"
        )
    return _frozen_write_sealed(
        path,
        body,
        fingerprint_field=fingerprint_field,
    )


def _preview_build_spec(args: object) -> dict[str, object]:
    """Run the exact producer with both mutation points replaced by memory."""

    captured: list[dict[str, object]] = []
    prior_create = legacy._create_runtime_directories_and_verify_leaves
    prior_write = legacy._write_sealed

    def no_runtime_directory_mutation(
        artifacts: Mapping[str, str],
    ) -> None:
        _require_build_phase_pristine()
        if not legacy._deep_exact_equal(
            dict(artifacts),
            legacy._artifact_paths(),
        ):
            raise PermissionError(
                "compatibility preview artifact contract changed"
            )

    def capture_spec(
        path: Path,
        body: Mapping[str, object],
        *,
        fingerprint_field: str,
    ) -> dict[str, object]:
        payload = _validate_prewrite_spec_body(
            path,
            body,
            fingerprint_field=fingerprint_field,
        )
        captured.append(payload)
        return payload

    legacy._create_runtime_directories_and_verify_leaves = (
        no_runtime_directory_mutation
    )
    legacy._write_sealed = capture_spec
    try:
        result = legacy.build_spec(
            policy_path=args.environment_policy,
            precleanup_path=args.precleanup_receipt,
            cleanup_plan_path=args.cleanup_plan,
            cleanup_authorization_path=args.cleanup_authorization,
            cleanup_receipt_path=args.cleanup_receipt,
            stability_path=args.stability_receipt,
            postcleanup_path=args.postcleanup_audit,
            integration_authorization_path=(
                args.integration_authorization
            ),
            integration_receipt_path=args.integration_receipt,
            realization_authorization_path=(
                args.unit_realization_authorization
            ),
            realization_receipt_path=args.unit_realization_receipt,
            shadow_reader=legacy._default_shadow_reader,
            manager_reader=(
                legacy.actual_realizer.collect_manager_generation
            ),
        )
    finally:
        legacy._create_runtime_directories_and_verify_leaves = prior_create
        legacy._write_sealed = prior_write
    if (
        len(captured) != 1
        or not legacy._deep_exact_equal(result, captured[0])
        or any(
            os.path.lexists(path)
            for path in (
                COMPAT_RUNTIME_SPEC_PATH,
                COMPAT_RUNTIME_ARTIFACT_ROOT,
                COMPAT_GPU_LEASE_ROOT,
            )
        )
    ):
        raise PermissionError(
            "compatibility build-spec preview was not mutation-free"
        )
    return captured[0]


def _require_compat_component_sources() -> None:
    for path, expected_sha256, expected_identity in (
        (
            COMPAT_REALIZER_PATH,
            COMPAT_REALIZER_SHA256,
            _COMPAT_REALIZER_LOAD_IDENTITY,
        ),
        (
            COMPAT_SUPERVISOR_PATH,
            COMPAT_SUPERVISOR_SHA256,
            _COMPAT_SUPERVISOR_LOAD_IDENTITY,
        ),
    ):
        raw, identity = _stable_source_bytes(path)
        if (
            hashlib.sha256(raw).hexdigest() != expected_sha256
            or identity != expected_identity
        ):
            raise PermissionError(
                "compatibility runtime component generation was replaced"
            )


def _configure() -> None:
    _require_frozen_release_source()
    _require_compat_component_sources()
    compat_realizer.verify_compatibility_identity()
    compat_supervisor.verify_compatibility_identity()
    legacy.actual_realizer = compat_realizer.legacy
    legacy.supervisor = compat_supervisor
    legacy._load_sealed = _compat_load_sealed
    legacy.validate_release_closure = _compat_validate_release_closure
    legacy._write_sealed = _compat_write_sealed
    legacy.UNIT_NAME = COMPAT_UNIT
    legacy.SUPERVISOR_PATH = COMPAT_SUPERVISOR_PATH
    legacy.ADAPTER_PATH = COMPAT_ADAPTER_PATH
    legacy.RUNTIME_SPEC_PATH = COMPAT_RUNTIME_SPEC_PATH
    legacy.RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
        COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
    )
    legacy.RUNTIME_ARTIFACT_ROOT = COMPAT_RUNTIME_ARTIFACT_ROOT
    legacy.GPU_LEASE_ROOT = COMPAT_GPU_LEASE_ROOT


_configure()


def verify_compatibility_identity() -> dict[str, object]:
    _configure()
    if (
        legacy.UNIT_NAME != COMPAT_UNIT
        or legacy.SUPERVISOR_PATH != COMPAT_SUPERVISOR_PATH
        or legacy.ADAPTER_PATH != COMPAT_ADAPTER_PATH
        or legacy.RUNTIME_SPEC_PATH != COMPAT_RUNTIME_SPEC_PATH
        or legacy.RUNTIME_LAUNCH_AUTHORIZATION_PATH
        != COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        or legacy.RUNTIME_ARTIFACT_ROOT != COMPAT_RUNTIME_ARTIFACT_ROOT
        or legacy.GPU_LEASE_ROOT != COMPAT_GPU_LEASE_ROOT
    ):
        raise PermissionError("compatibility release identity changed")
    return {
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_generation": "c1",
        "unit_name": COMPAT_UNIT,
        "runtime_spec_path": str(COMPAT_RUNTIME_SPEC_PATH),
        "runtime_launch_authorization_path": str(
            COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        ),
        "runtime_artifact_root": str(COMPAT_RUNTIME_ARTIFACT_ROOT),
        "gpu_lease_root": str(COMPAT_GPU_LEASE_ROOT),
        "compatibility_receipt_path": str(
            COMPATIBILITY_RECEIPT_PATH,
        ),
        "authoritative_access_audit_schema": (
            AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        ),
        "fictional_access_audit_schema_accepted": False,
        "frozen_release_path": str(FROZEN_RELEASE_PATH),
        "frozen_release_file_sha256": FROZEN_RELEASE_SHA256,
        "supervisor_path": str(COMPAT_SUPERVISOR_PATH),
        "adapter_path": str(COMPAT_ADAPTER_PATH),
        "realizer_path": str(COMPAT_REALIZER_PATH),
        "fresh_scientific_attempt": False,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    materialized = list(sys.argv[1:] if argv is None else argv)
    _require_frozen_release_source()
    args = legacy._parser().parse_args(materialized)
    if args.command not in _ALLOWED_COMMANDS:
        raise PermissionError(
            "compatibility release command is not allowed",
        )
    if args.command == "build-spec":
        _require_fixed_inputs(
            {
                "policy_path": args.environment_policy,
                "precleanup_path": args.precleanup_receipt,
                "cleanup_plan_path": args.cleanup_plan,
                "cleanup_authorization_path": (
                    args.cleanup_authorization
                ),
                "cleanup_receipt_path": args.cleanup_receipt,
                "stability_path": args.stability_receipt,
                "postcleanup_path": args.postcleanup_audit,
                "integration_authorization_path": (
                    args.integration_authorization
                ),
                "integration_receipt_path": (
                    args.integration_receipt
                ),
                "realization_authorization_path": (
                    args.unit_realization_authorization
                ),
                "realization_receipt_path": (
                    args.unit_realization_receipt
                ),
            }
        )
    verify_compatibility_identity()
    _require_lane_separation()
    token = _ACTIVE_COMMAND.set(args.command)
    preview_token = None
    try:
        _verify_active_phase_compatibility_receipt()
        if args.command == "build-spec":
            preview = _preview_build_spec(args)
            repeated_preview = _preview_build_spec(args)
            if not legacy._deep_exact_equal(
                preview,
                repeated_preview,
            ):
                raise PermissionError(
                    "compatibility build-spec preview is nondeterministic"
                )
            preview_token = _APPROVED_PREWRITE_SPEC.set(preview)
        _require_frozen_release_source()
        return int(legacy.main(materialized))
    finally:
        if preview_token is not None:
            _APPROVED_PREWRITE_SPEC.reset(preview_token)
        _ACTIVE_COMMAND.reset(token)


def __getattr__(name: str):
    return getattr(legacy, name)


if __name__ == "__main__":
    raise SystemExit(main())
