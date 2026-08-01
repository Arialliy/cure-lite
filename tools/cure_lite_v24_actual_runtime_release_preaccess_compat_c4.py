#!/usr/bin/env python3
"""Create-only runtime release wrapper for compatibility generation c4.

The already audited c1 release producer is the implementation authority.  Its
exact bytes are read through no-follow file descriptors, hashed, and compiled
only after the frozen hash matches.  This wrapper then narrows that producer to
the disjoint c4 runtime namespace and adds three c4-specific consumers:

* the sealed c4 compatibility receipt (including its environment lineage);
* the fresh c4 environment scope cross-binding;
* the archival c4 unit-realization closure.

This module can only build one runtime specification or authorize its one
launch.  It never starts systemd, accesses a payload, retries, resumes, creates
a new scientific attempt, or authorizes D_V/D_T.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
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
    REPOSITORY
    / "tools/cure_lite_v24_actual_runtime_release_preaccess_compat_c1.py"
).resolve()
FROZEN_RELEASE_SHA256 = (
    "395a013ff4f14160a0ac4e9845497caf9ecbaa6f2eeb3aa88fad54b63f514cfa"
)
COMPAT_BRIDGE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c4.py"
).resolve()
COMPAT_BRIDGE_SHA256 = (
    "ad660b7afe7ca87f690bc9565bd6674684c2b62824394751a39114a6efcf178a"
)
COMPAT_REALIZER_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c4.py"
).resolve()
COMPAT_REALIZER_SHA256 = (
    "8708f8a13d74623f510992e23c6c23e1c4bfe70db09092c04fe56d44d29c5b65"
)
COMPAT_SUPERVISOR_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c4.py"
).resolve()
COMPAT_SUPERVISOR_SHA256 = (
    "faffe980cba4cad668a7d0f525bed8f2005950503d46f2b7c6888d79813c64ce"
)
COMPAT_ENVIRONMENT_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_environment_preaccess_compat_c4.py"
).resolve()
COMPAT_ENVIRONMENT_SHA256 = (
    "f4335efdb3865efe68dbbb6aac5f7977fd2157452b557f83428e4dd4a5d8932b"
)
COMPAT_ADAPTER_PATH = (
    REPOSITORY
    / "tools/"
    "run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c4.py"
).resolve()
COMPAT_ADAPTER_SHA256 = (
    "fa5fe28eacf3980720616a9b54dc1dc878f1e1280883f643a49ec4c4800e92ed"
)
R14_INTEGRATION_PATH = (
    REPOSITORY / "tools/cure_lite_v24_user_systemd_integration.py"
).resolve()
R14_INTEGRATION_SHA256 = (
    "10f6814deab43cfa4513b813546759b1e6508272a18ee026ebecb2cf8535a187"
)
R14_SHARED_REALIZER_PATH = (
    REPOSITORY / "tools/cure_lite_v24_realize_systemd_unit.py"
).resolve()
R14_SHARED_REALIZER_SHA256 = (
    "131b89f186b064629354165a9454b976145b5d76fbf36053292b2997e73cf6b6"
)
R14_DUMMY_CHILD_PATH = (
    REPOSITORY / "tools/cure_lite_v24_dummy_child.py"
).resolve()
R14_DUMMY_CHILD_SHA256 = (
    "d57e21c7450c58faba8d6915ffe09647b19313f3f734e8ec2847a34001ab27b9"
)
R14_DUMMY_TEMPLATE_PATH = (
    REPOSITORY
    / "deploy/systemd/cure-lite-v24-supervisor-integration.service.template"
).resolve()
R14_DUMMY_TEMPLATE_SHA256 = (
    "8df34c4ea07d2dfe3b23f0c8407df66b9f282bb43f7275c245ee110f90f568c8"
)

COMPAT_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service"
)
COMPAT_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_spec.json"
)
COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c4_"
        "runtime_launch_authorization.json"
    )
)
COMPAT_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_artifacts"
)
COMPAT_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_gpu_lease"
)
COMPAT_RUN_ROOT_ALIAS_PATH = (
    REPOSITORY
    / "runs/irstd1k_stage_a_seed42/"
    "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c4"
)
COMPAT_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_receipt.json"
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
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c4_receipt.json"
)

POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c4.json"
)
SCOPE_HANDOFF_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_scope_handoff_preaccess_compat_c4.json"
)
STABILITY_ATTEMPT_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_attempt_preaccess_compat_c4.json"
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
    / "runtime_environment_stability_receipt_preaccess_compat_c4.json"
)
POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c4.json"
)
INTEGRATION_ROOT = (
    EVIDENCE_ROOT
    / "supervisor_v2_systemd_integration_preaccess_compat_c4_r14"
)
INTEGRATION_AUTHORIZATION_PATH = (
    INTEGRATION_ROOT / "control/authorization.json"
)
INTEGRATION_RECEIPT_PATH = (
    INTEGRATION_ROOT / "control/integration-receipt.json"
)
INTEGRATION_RUNTIME_SPEC_PATH = (
    INTEGRATION_ROOT / "control/runtime-spec.json"
)
INTEGRATION_TERMINAL_PATH = (
    INTEGRATION_ROOT / "control/integration-terminal.json"
)
INTEGRATION_REMOVAL_AUTHORIZATION_PATH = (
    INTEGRATION_ROOT / "control/removal-authorization.json"
)
INTEGRATION_REMOVAL_STATE_PATH = (
    INTEGRATION_ROOT / "control/removal-state.json"
)
R14_SCENARIO_ID = (
    "supervisor-v2-dummy-compat-c4-r14-20260731c4000014"
)
R14_DUMMY_UNIT = (
    "cure-lite-v24-supervisor-integration-"
    "supervisor-v2-dummy-compat-c4-r14-20260731c4000014.service"
)
REALIZATION_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c4_unit_realization_authorization.json"
)
REALIZATION_RECEIPT_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c4_unit_realization_receipt.json"
)
REALIZATION_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c4_unit_realization_terminal.json"
)

BLOCKED_RUNTIME_PATHS = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_spec.json",
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_runtime_launch_authorization.json",
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_artifacts",
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_gpu_lease",
)
C1_RUNTIME_PATHS = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json",
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c1_"
        "runtime_launch_authorization.json"
    ),
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_artifacts",
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_gpu_lease",
)
C2_RUNTIME_PATHS = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_spec.json",
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c2_"
        "runtime_launch_authorization.json"
    ),
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_artifacts",
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_gpu_lease",
)
C3_RUNTIME_PATHS = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_runtime_spec.json",
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c3_"
        "runtime_launch_authorization.json"
    ),
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_runtime_artifacts",
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_gpu_lease",
)

FORBIDDEN_SCIENTIFIC_ALIASES = (
    REPOSITORY
    / "runs/irstd1k_stage_a_seed42/"
    "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c1",
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_receipt.json",
    REPOSITORY
    / "runs/irstd1k_stage_a_seed42/"
    "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c2",
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_receipt.json",
    REPOSITORY
    / "runs/irstd1k_stage_a_seed42/"
    "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c3",
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_receipt.json",
    COMPAT_RUN_ROOT_ALIAS_PATH,
    COMPAT_RESULT_RECEIPT_ALIAS_PATH,
)

AUTHORITATIVE_ACCESS_AUDIT_SCHEMA = (
    "cure-lite-v24-split-access-audit-v1"
)
FICTIONAL_ACCESS_AUDIT_SCHEMA = (
    "cure-lite-v24-split-access-audit-r2-v1"
)
_ALLOWED_COMMANDS = frozenset({"build-spec", "authorize-launch"})
_FILE_FIELDS = (
    "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
    "st_size", "st_mtime_ns", "st_ctime_ns",
)
_PARENT_FIELDS = (
    "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
)


def _build_disjoint_c4_runtime_identity_gate():
    """Freeze the c4 runtime identity and its separation from older lanes."""

    expected_unit = COMPAT_UNIT
    expected_runtime_paths = (
        COMPAT_RUNTIME_SPEC_PATH.absolute(),
        COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH.absolute(),
        COMPAT_RUNTIME_ARTIFACT_ROOT.absolute(),
        COMPAT_GPU_LEASE_ROOT.absolute(),
    )
    expected_predecessor_paths = tuple(
        path.absolute()
        for path in (
            *BLOCKED_RUNTIME_PATHS,
            *C1_RUNTIME_PATHS,
            *C2_RUNTIME_PATHS,
            *C3_RUNTIME_PATHS,
        )
    )
    expected_aliases = (
        COMPAT_RUN_ROOT_ALIAS_PATH.absolute(),
        COMPAT_RESULT_RECEIPT_ALIAS_PATH.absolute(),
    )
    expected_forbidden_aliases = tuple(
        path.absolute() for path in FORBIDDEN_SCIENTIFIC_ALIASES
    )
    expected_scientific_paths = (
        SCIENTIFIC_RUN_ROOT.absolute(),
        SCIENTIFIC_RESULT_RECEIPT_PATH.absolute(),
    )

    def require() -> None:
        runtime_paths = (
            Path(COMPAT_RUNTIME_SPEC_PATH).absolute(),
            Path(COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH).absolute(),
            Path(COMPAT_RUNTIME_ARTIFACT_ROOT).absolute(),
            Path(COMPAT_GPU_LEASE_ROOT).absolute(),
        )
        predecessor_paths = tuple(
            Path(path).absolute()
            for path in (
                *BLOCKED_RUNTIME_PATHS,
                *C1_RUNTIME_PATHS,
                *C2_RUNTIME_PATHS,
                *C3_RUNTIME_PATHS,
            )
        )
        aliases = (
            Path(COMPAT_RUN_ROOT_ALIAS_PATH).absolute(),
            Path(COMPAT_RESULT_RECEIPT_ALIAS_PATH).absolute(),
        )
        forbidden_aliases = tuple(
            Path(path).absolute() for path in FORBIDDEN_SCIENTIFIC_ALIASES
        )
        scientific_paths = (
            Path(SCIENTIFIC_RUN_ROOT).absolute(),
            Path(SCIENTIFIC_RESULT_RECEIPT_PATH).absolute(),
        )
        if (
            COMPAT_UNIT != expected_unit
            or runtime_paths != expected_runtime_paths
            or predecessor_paths != expected_predecessor_paths
            or aliases != expected_aliases
            or forbidden_aliases != expected_forbidden_aliases
            or scientific_paths != expected_scientific_paths
            or len(set(runtime_paths)) != len(runtime_paths)
            or set(runtime_paths).intersection(predecessor_paths)
            or set(runtime_paths).intersection(scientific_paths)
            or len(set(aliases)) != len(aliases)
            or set(aliases).intersection(
                (*runtime_paths, *predecessor_paths, *scientific_paths)
            )
            or len(set(scientific_paths)) != len(scientific_paths)
        ):
            raise PermissionError("c4 runtime identity/path isolation changed")

    return require


_require_disjoint_c4_runtime_identity = (
    _build_disjoint_c4_runtime_identity_gate()
)


def _stable_source_bytes(path: Path) -> tuple[bytes, dict[str, int]]:
    """Read one canonical regular-file generation through no-follow fds."""

    target = Path(path).absolute()
    parent = target.parent
    parent_before = parent.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or parent.resolve(strict=True) != parent
        or target.resolve(strict=True) != target
    ):
        raise PermissionError("c4 release source path is unsafe")
    directory_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        parent_opened = os.fstat(directory_fd)
        before = os.stat(
            target.name, dir_fd=directory_fd, follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
        ):
            raise PermissionError("c4 release source is unsafe")
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
            finished = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        linked = os.stat(
            target.name, dir_fd=directory_fd, follow_symlinks=False,
        )
        parent_finished = os.fstat(directory_fd)
    finally:
        os.close(directory_fd)
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(finished, field)
        or getattr(finished, field) != getattr(linked, field)
        for field in _FILE_FIELDS
    ) or any(
        getattr(parent_before, field) != getattr(parent_opened, field)
        or getattr(parent_opened, field) != getattr(parent_finished, field)
        for field in _PARENT_FIELDS
    ):
        raise PermissionError("c4 release source generation changed")
    identity = {
        field: int(getattr(linked, field)) for field in _FILE_FIELDS
    }
    identity.update({
        f"parent_{field}": int(getattr(parent_finished, field))
        for field in _PARENT_FIELDS
    })
    return b"".join(chunks), identity


def _load_verified(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> tuple[ModuleType, dict[str, int]]:
    raw, identity = _stable_source_bytes(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError("bound c4 release component source changed")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(raw, str(path), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    raw_after, after = _stable_source_bytes(path)
    if raw_after != raw or after != identity:
        sys.modules.pop(name, None)
        raise PermissionError("bound c4 release component was replaced")
    return module, identity

def _load_when_frozen(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> tuple[ModuleType, dict[str, int] | None]:
    """Keep development imports inert until a sibling hash is frozen."""

    if (
        len(expected_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sha256
        )
    ):
        module = ModuleType(name)
        module.__file__ = str(path)
        module.__package__ = "tools"
        return module, None
    return _load_verified(path, expected_sha256, name=name)


def _bind_when_frozen(
    path: Path,
    expected_sha256: str,
) -> dict[str, int] | None:
    if (
        len(expected_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sha256
        )
    ):
        return None
    raw, identity = _stable_source_bytes(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError("bound c4 release component source changed")
    return identity

compat_c1, _FROZEN_RELEASE_LOAD_IDENTITY = _load_verified(
    FROZEN_RELEASE_PATH,
    FROZEN_RELEASE_SHA256,
    name=(
        "tools._cure_lite_v24_actual_runtime_release_preaccess_"
        "compat_c1_verified_for_c4"
    ),
)
compat_bridge, _BRIDGE_LOAD_IDENTITY = _load_when_frozen(
    COMPAT_BRIDGE_PATH,
    COMPAT_BRIDGE_SHA256,
    name="tools._cure_lite_v24_preaccess_compat_c4_verified_for_release",
)
compat_realizer, _REALIZER_LOAD_IDENTITY = _load_when_frozen(
    COMPAT_REALIZER_PATH,
    COMPAT_REALIZER_SHA256,
    name="tools._cure_lite_v24_realizer_compat_c4_verified_for_release",
)
compat_supervisor, _SUPERVISOR_LOAD_IDENTITY = _load_when_frozen(
    COMPAT_SUPERVISOR_PATH,
    COMPAT_SUPERVISOR_SHA256,
    name="tools._cure_lite_v24_supervisor_compat_c4_verified_for_release",
)
compat_environment, _ENVIRONMENT_LOAD_IDENTITY = _load_when_frozen(
    COMPAT_ENVIRONMENT_PATH,
    COMPAT_ENVIRONMENT_SHA256,
    name="tools._cure_lite_v24_environment_compat_c4_verified_for_release",
)
r14_integration, _R14_INTEGRATION_LOAD_IDENTITY = _load_when_frozen(
    R14_INTEGRATION_PATH,
    R14_INTEGRATION_SHA256,
    name="tools._cure_lite_v24_r14_integration_verified_for_release",
)
r14_shared_realizer, _R14_SHARED_REALIZER_LOAD_IDENTITY = (
    _load_when_frozen(
        R14_SHARED_REALIZER_PATH,
        R14_SHARED_REALIZER_SHA256,
        name="tools._cure_lite_v24_r14_realizer_verified_for_release",
    )
)
_ADAPTER_LOAD_IDENTITY = _bind_when_frozen(
    COMPAT_ADAPTER_PATH,
    COMPAT_ADAPTER_SHA256,
)
_R14_DUMMY_CHILD_LOAD_IDENTITY = _bind_when_frozen(
    R14_DUMMY_CHILD_PATH,
    R14_DUMMY_CHILD_SHA256,
)
_R14_DUMMY_TEMPLATE_LOAD_IDENTITY = _bind_when_frozen(
    R14_DUMMY_TEMPLATE_PATH,
    R14_DUMMY_TEMPLATE_SHA256,
)

legacy = compat_c1.legacy
_ACTIVE_COMMAND = compat_c1._ACTIVE_COMMAND
_APPROVED_PREWRITE_SPEC = compat_c1._APPROVED_PREWRITE_SPEC
_frozen_validate_release_closure = (
    compat_c1._frozen_validate_release_closure
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_stable_source_bytes(path)[0]).hexdigest()


def _require_frozen_release_source() -> None:
    raw, identity = _stable_source_bytes(FROZEN_RELEASE_PATH)
    if (
        hashlib.sha256(raw).hexdigest() != FROZEN_RELEASE_SHA256
        or identity != _FROZEN_RELEASE_LOAD_IDENTITY
    ):
        raise PermissionError("frozen c1 release generation changed")


def _require_component_sources() -> None:
    for path, digest, identity in (
        (COMPAT_BRIDGE_PATH, COMPAT_BRIDGE_SHA256, _BRIDGE_LOAD_IDENTITY),
        (COMPAT_REALIZER_PATH, COMPAT_REALIZER_SHA256, _REALIZER_LOAD_IDENTITY),
        (
            COMPAT_SUPERVISOR_PATH,
            COMPAT_SUPERVISOR_SHA256,
            _SUPERVISOR_LOAD_IDENTITY,
        ),
        (
            COMPAT_ENVIRONMENT_PATH,
            COMPAT_ENVIRONMENT_SHA256,
            _ENVIRONMENT_LOAD_IDENTITY,
        ),
        (COMPAT_ADAPTER_PATH, COMPAT_ADAPTER_SHA256, _ADAPTER_LOAD_IDENTITY),
        (
            R14_INTEGRATION_PATH,
            R14_INTEGRATION_SHA256,
            _R14_INTEGRATION_LOAD_IDENTITY,
        ),
        (
            R14_SHARED_REALIZER_PATH,
            R14_SHARED_REALIZER_SHA256,
            _R14_SHARED_REALIZER_LOAD_IDENTITY,
        ),
        (
            R14_DUMMY_CHILD_PATH,
            R14_DUMMY_CHILD_SHA256,
            _R14_DUMMY_CHILD_LOAD_IDENTITY,
        ),
        (
            R14_DUMMY_TEMPLATE_PATH,
            R14_DUMMY_TEMPLATE_SHA256,
            _R14_DUMMY_TEMPLATE_LOAD_IDENTITY,
        ),
    ):
        if (
            identity is None
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise PermissionError("c4 release component hash is not frozen")
        raw, observed = _stable_source_bytes(path)
        if hashlib.sha256(raw).hexdigest() != digest or observed != identity:
            raise PermissionError("c4 release component generation changed")


def _build_l4_delegation_gate():
    """Create the sole delegated main and its closure-private capability."""

    seal = object()
    state: ContextVar[
        tuple[object, str, tuple[str, ...]] | None
    ] = ContextVar("cure_lite_v24_c4_release_delegation", default=None)

    def require(
        command: str,
        *,
        argv: Sequence[str] | None = None,
    ) -> None:
        capability = state.get()
        if (
            command not in _ALLOWED_COMMANDS
            or capability is None
            or capability[0] is not seal
            or capability[1] != command
            or _ACTIVE_COMMAND.get() != command
            or (
                argv is not None
                and capability[2] != tuple(str(value) for value in argv)
            )
        ):
            raise PermissionError("c4 release delegation is not authorized")
        _require_frozen_release_source()
        _require_component_sources()

    def delegated_main(argv: Sequence[str] | None = None) -> int:
        materialized = list(sys.argv[1:] if argv is None else argv)
        _require_frozen_release_source()
        _require_component_sources()
        args = legacy._parser().parse_args(materialized)
        if args.command not in _ALLOWED_COMMANDS:
            raise PermissionError("c4 release command is not allowed")
        if args.command == "build-spec":
            _require_fixed_inputs(_fixed_build_values(args))
        verify_compatibility_identity()
        _require_lane_separation()
        command_token = _ACTIVE_COMMAND.set(args.command)
        delegation_token = state.set(
            (
                seal,
                args.command,
                tuple(str(value) for value in materialized),
            )
        )
        preview_token = None
        try:
            _verify_active_phase_compatibility_receipt()
            if args.command == "build-spec":
                preview = _preview_build_spec(args)
                repeated = _preview_build_spec(args)
                if not legacy._deep_exact_equal(preview, repeated):
                    raise PermissionError(
                        "c4 build-spec preview is nondeterministic"
                    )
                preview_token = _APPROVED_PREWRITE_SPEC.set(preview)
            _require_frozen_release_source()
            _require_component_sources()
            return int(legacy.main(materialized))
        finally:
            if preview_token is not None:
                _APPROVED_PREWRITE_SPEC.reset(preview_token)
            state.reset(delegation_token)
            _ACTIVE_COMMAND.reset(command_token)

    return delegated_main, require


main, _require_l4_delegation = _build_l4_delegation_gate()
del _build_l4_delegation_gate


def _guard_release_callable(
    function,
    command: str | None,
    *,
    bind_argv: bool = False,
):
    """Guard one inherited release callable without exposing ``__wrapped__``."""

    def guarded(*args, **kwargs):
        selected = command
        if selected is None:
            selected = _ACTIVE_COMMAND.get()
        if selected not in _ALLOWED_COMMANDS:
            raise PermissionError("c4 release phase is not explicitly selected")
        delegated_argv: Sequence[str] | None = None
        if bind_argv:
            supplied = args[0] if args else kwargs.get("argv")
            delegated_argv = list(
                sys.argv[1:] if supplied is None else supplied
            )
        _require_l4_delegation(selected, argv=delegated_argv)
        return function(*args, **kwargs)

    guarded.__name__ = getattr(function, "__name__", "guarded_c4_release")
    guarded.__doc__ = getattr(function, "__doc__", None)
    return guarded


def _require_lane_separation() -> None:
    if any(
        os.path.lexists(path)
        for path in (
            *BLOCKED_RUNTIME_PATHS,
            *C1_RUNTIME_PATHS,
            *C2_RUNTIME_PATHS,
            *C3_RUNTIME_PATHS,
        )
    ):
        raise PermissionError(
            "direct, c1, c2, or c3 predecessor runtime lane is no longer pristine"
        )
    if any(os.path.lexists(path) for path in FORBIDDEN_SCIENTIFIC_ALIASES):
        raise PermissionError("runtime compatibility scientific alias exists")


def _require_build_phase_pristine() -> None:
    if any(
        os.path.lexists(path)
        for path in (
            COMPAT_RUNTIME_SPEC_PATH,
            COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
            COMPAT_RUNTIME_ARTIFACT_ROOT,
            COMPAT_GPU_LEASE_ROOT,
        )
    ):
        raise PermissionError("c4 build-spec namespace is already materialized")


def _require_prelaunch_scientific_pristine() -> None:
    if any(
        os.path.lexists(path)
        for path in (
            SCIENTIFIC_RUN_ROOT,
            SCIENTIFIC_RESULT_RECEIPT_PATH,
        )
    ):
        raise PermissionError(
            "scientific r2 output exists before c4 launch authorization"
        )


def _load_fixed_compat_runtime_spec() -> dict[str, object]:
    payload = compat_c1._frozen_load_sealed(
        COMPAT_RUNTIME_SPEC_PATH,
        fingerprint_field="runtime_spec_fingerprint",
        schema=compat_supervisor.RUNTIME_SPEC_SCHEMA,
    )
    if not isinstance(payload, Mapping):
        raise PermissionError("fixed c4 runtime spec is not a mapping")
    return dict(payload)


def _load_r4_archival() -> dict[str, object]:
    sealed = compat_realizer.compat_c1.legacy
    authorization, authorization_identity = sealed._load_sealed_json_bound(
        REALIZATION_AUTHORIZATION_PATH,
        "authorization_fingerprint",
    )
    receipt, receipt_identity = sealed._load_sealed_json_bound(
        REALIZATION_RECEIPT_PATH,
        "receipt_fingerprint",
    )
    authorization_root = compat_environment._bind_r4_archival_root(
        REALIZATION_AUTHORIZATION_PATH,
        authorization_identity,
    )
    receipt_root = compat_environment._bind_r4_archival_root(
        REALIZATION_RECEIPT_PATH,
        receipt_identity,
    )
    compat_realizer.compat_c1._validate_receipt_transitive_binding(
        receipt,
        require_fresh=False,
        require_bridge_fresh=False,
        require_future_absence=False,
    )
    compat_realizer._validate_c4_receipt_contract(receipt)
    if (
        os.path.lexists(REALIZATION_TERMINAL_PATH)
        or receipt.get("authorization_file_sha256")
        != authorization_identity.get("file_sha256")
        or receipt.get("authorization_fingerprint")
        != authorization.get("authorization_fingerprint")
    ):
        raise PermissionError("R4 archival realization is not exact PASS")
    authorization_root_after = compat_environment._bind_r4_archival_root(
        REALIZATION_AUTHORIZATION_PATH,
        authorization_identity,
    )
    receipt_root_after = compat_environment._bind_r4_archival_root(
        REALIZATION_RECEIPT_PATH,
        receipt_identity,
    )
    if (
        not legacy._deep_exact_equal(
            authorization_root,
            authorization_root_after,
        )
        or not legacy._deep_exact_equal(receipt_root, receipt_root_after)
    ):
        raise PermissionError("R4 archival evidence generation changed")
    return {
        "authorization": dict(authorization),
        "receipt": dict(receipt),
        "authorization_identity": dict(authorization_root_after),
        "receipt_identity": dict(receipt_root_after),
        "compatibility_closure": dict(
            authorization[compat_realizer.COMPATIBILITY_CLOSURE_KEY]
        ),
    }


def _verify_environment_cross_binding(
    bridge_receipt: Mapping[str, object],
) -> dict[str, object]:
    old_contract, c4_contract, _roots = (
        compat_environment.replay_old_scope_and_handoff()
    )
    archival = _load_r4_archival()
    archival = compat_environment.validate_c4_realization_archival(
        archival,
        contract=c4_contract,
    )
    policy = compat_environment.frozen.load_sealed_receipt(
        POLICY_PATH,
        fingerprint_field="policy_fingerprint",
    )
    handoff = compat_environment.frozen.load_sealed_receipt(
        SCOPE_HANDOFF_PATH,
        fingerprint_field="scope_handoff_fingerprint",
    )
    attempt = compat_environment.frozen.load_sealed_receipt(
        STABILITY_ATTEMPT_PATH,
        fingerprint_field="stability_attempt_fingerprint",
    )
    stability = compat_environment.frozen.load_sealed_receipt(
        STABILITY_PATH,
        fingerprint_field="stability_receipt_fingerprint",
    )
    postcleanup = compat_environment.frozen.load_sealed_receipt(
        POSTCLEANUP_PATH,
        fingerprint_field="receipt_fingerprint",
    )
    closure = compat_environment.validate_c4_environment_closure(
        handoff,
        attempt,
        policy,
        stability,
        postcleanup,
        archival=archival,
        c4_contract=c4_contract,
    )
    if (
        bridge_receipt.get("historical_environment_contract")
        != compat_environment._normalized_contract(old_contract)
        or bridge_receipt.get("current_environment_contract")
        != compat_environment._normalized_contract(c4_contract)
        or not legacy._deep_exact_equal(
            closure.get("realization"), archival,
        )
    ):
        raise PermissionError(
            "c4 bridge/environment/realization cross-binding changed"
        )
    return closure


def _verify_compatibility_receipt(
    *,
    expected_spec: Mapping[str, object] | None = None,
    require_spec_binding: bool,
) -> Mapping[str, object]:
    result = compat_bridge.verify_compatibility_receipt(
        COMPATIBILITY_RECEIPT_PATH,
        expected_spec=(
            None if expected_spec is None else dict(expected_spec)
        ),
        require_spec_binding=require_spec_binding,
        allow_runtime_activation=False,
    )
    roots = result.get("compatibility_source_roots")
    bridge_root = (
        roots.get("compat_bridge") if isinstance(roots, Mapping) else None
    )
    if (
        not isinstance(result, Mapping)
        or result.get("runtime_compatibility_id") != "c4"
        or result.get("automatic_retry") is not False
        or result.get("resume") is not False
        or result.get("D_R_payload_accessed") is not False
        or result.get("D_V_payload_accessed") is not False
        or result.get("D_T_payload_accessed") is not False
        or not isinstance(bridge_root, Mapping)
        or bridge_root.get("path") != str(COMPAT_BRIDGE_PATH)
        or bridge_root.get("file_sha256") != COMPAT_BRIDGE_SHA256
    ):
        raise PermissionError("c4 compatibility receipt is not exact")
    _verify_environment_cross_binding(result)
    return result


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
        return _verify_compatibility_receipt(
            expected_spec=_load_fixed_compat_runtime_spec(),
            require_spec_binding=True,
        )
    raise PermissionError("c4 release phase is not explicitly selected")


def _require_fixed_inputs(values: Mapping[str, object]) -> None:
    expected = {
        "policy_path": POLICY_PATH,
        "precleanup_path": PRECLEANUP_PATH,
        "cleanup_plan_path": CLEANUP_PLAN_PATH,
        "cleanup_authorization_path": CLEANUP_AUTHORIZATION_PATH,
        "cleanup_receipt_path": CLEANUP_RECEIPT_PATH,
        "stability_path": STABILITY_PATH,
        "postcleanup_path": POSTCLEANUP_PATH,
        "integration_authorization_path": INTEGRATION_AUTHORIZATION_PATH,
        "integration_receipt_path": INTEGRATION_RECEIPT_PATH,
        "realization_authorization_path": REALIZATION_AUTHORIZATION_PATH,
        "realization_receipt_path": REALIZATION_RECEIPT_PATH,
    }
    for name, expected_path in expected.items():
        supplied = values.get(name)
        if (
            not isinstance(supplied, Path)
            or supplied.absolute() != expected_path.absolute()
        ):
            raise PermissionError(f"c4 release input path changed: {name}")


def _require_final_r14_supervisor_binding(
    closure: Mapping[str, object],
) -> None:
    """Bind the archived r14 proof to this exact supervisor generation."""

    integration = closure.get("integration")
    if not isinstance(integration, Mapping):
        raise PermissionError(
            "r14 integration is not bound to the final c4 supervisor "
            "generation"
        )
    authorization = integration.get("authorization")
    identities = integration.get("identities")
    if (
        not isinstance(authorization, Mapping)
        or not isinstance(identities, Mapping)
    ):
        raise PermissionError(
            "r14 integration is not bound to the final c4 supervisor "
            "generation"
        )
    authorization_identity = identities.get("authorization")
    receipt_identity = identities.get("receipt")
    terminal_identity = identities.get("integration_terminal")
    removal_authorization_identity = identities.get(
        "removal_authorization"
    )
    removal_state_identity = identities.get("removal_state")
    scenario_root = authorization.get("scenario_root")
    control_root = authorization.get("control_root")
    runtime_root = authorization.get("runtime_root")
    runtime_spec_binding = authorization.get("runtime_spec_binding")
    executable_bindings = authorization.get("executable_bindings")
    if (
        not isinstance(authorization_identity, Mapping)
        or not isinstance(receipt_identity, Mapping)
        or not isinstance(terminal_identity, Mapping)
        or not isinstance(removal_authorization_identity, Mapping)
        or not isinstance(removal_state_identity, Mapping)
        or not isinstance(scenario_root, Mapping)
        or not isinstance(control_root, Mapping)
        or not isinstance(runtime_root, Mapping)
        or not isinstance(runtime_spec_binding, Mapping)
        or not isinstance(executable_bindings, Mapping)
    ):
        raise PermissionError(
            "r14 integration is not bound to the final c4 supervisor "
            "generation"
        )
    supervisor_binding = executable_bindings.get("supervisor")
    source_identity = _SUPERVISOR_LOAD_IDENTITY
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_nlink",
    )
    if (
        not isinstance(supervisor_binding, Mapping)
        or not isinstance(source_identity, Mapping)
        or any(
            isinstance(source_identity.get(field), bool)
            or not isinstance(source_identity.get(field), int)
            for field in identity_fields
        )
        or not stat.S_ISREG(source_identity["st_mode"])
        or source_identity["st_nlink"] != 1
    ):
        raise PermissionError(
            "r14 integration is not bound to the final c4 supervisor "
            "generation"
        )
    supervisor_path = COMPAT_SUPERVISOR_PATH.absolute()
    expected_binding = {
        "path": str(supervisor_path),
        "resolved_path": str(supervisor_path),
        "path_is_symlink": False,
        "file_sha256": COMPAT_SUPERVISOR_SHA256,
        "device": source_identity["st_dev"],
        "inode": source_identity["st_ino"],
        "owner_uid": source_identity["st_uid"],
        "mode": stat.S_IMODE(source_identity["st_mode"]),
    }
    authorization_path = INTEGRATION_AUTHORIZATION_PATH.absolute()
    receipt_path = INTEGRATION_RECEIPT_PATH.absolute()
    integration_root = INTEGRATION_ROOT.absolute()
    expected_identity = r14_integration.build_supervisor_v2_identity(
        R14_SCENARIO_ID,
    )
    control = authorization.get("control_artifacts")
    expected_control = {
        "integration_terminal": str(INTEGRATION_TERMINAL_PATH),
        "removal_authorization": str(
            INTEGRATION_REMOVAL_AUTHORIZATION_PATH
        ),
        "removal_state": str(INTEGRATION_REMOVAL_STATE_PATH),
        "integration_receipt": str(INTEGRATION_RECEIPT_PATH),
        "dummy_artifact": str(INTEGRATION_ROOT / "runtime/dummy-child.json"),
    }
    if (
        authorization_path.parent.parent != integration_root
        or receipt_path.parent.parent != integration_root
        or authorization.get("scenario_id") != R14_SCENARIO_ID
        or authorization.get("identity") != expected_identity
        or expected_identity.get("unit_name") != R14_DUMMY_UNIT
        or not isinstance(control, Mapping)
        or dict(control) != expected_control
        or runtime_spec_binding.get("path")
        != str(INTEGRATION_RUNTIME_SPEC_PATH)
        or authorization_identity.get("path")
        != str(authorization_path)
        or receipt_identity.get("path") != str(receipt_path)
        or terminal_identity.get("path")
        != str(INTEGRATION_TERMINAL_PATH)
        or removal_authorization_identity.get("path")
        != str(INTEGRATION_REMOVAL_AUTHORIZATION_PATH)
        or removal_state_identity.get("path")
        != str(INTEGRATION_REMOVAL_STATE_PATH)
        or scenario_root.get("path") != str(integration_root)
        or control_root.get("path") != str(integration_root / "control")
        or runtime_root.get("path") != str(integration_root / "runtime")
        or dict(supervisor_binding) != expected_binding
    ):
        raise PermissionError(
            "r14 integration is not bound to the final c4 supervisor "
            "generation"
        )


def _parse_archival_utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PermissionError(f"{name} timestamp is malformed")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PermissionError(f"{name} timestamp is malformed") from error
    if result.tzinfo is None:
        raise PermissionError(f"{name} timestamp is naive")
    offset = result.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise PermissionError(f"{name} timestamp is not UTC")
    return result


def _require_r14_source_binding(
    binding: object,
    *,
    path: Path,
    digest: str,
) -> None:
    if not isinstance(binding, Mapping):
        raise PermissionError("r14 source binding is absent")
    raw, identity = _stable_source_bytes(path)
    expected = {
        "path": str(path),
        "resolved_path": str(path),
        "path_is_symlink": False,
        "file_sha256": digest,
        "device": identity["st_dev"],
        "inode": identity["st_ino"],
        "owner_uid": identity["st_uid"],
        "mode": stat.S_IMODE(identity["st_mode"]),
    }
    if (
        hashlib.sha256(raw).hexdigest() != digest
        or dict(binding) != expected
    ):
        raise PermissionError("r14 source binding changed")


def _validate_r14_log(binding: object, *, expected_path: Path) -> None:
    if not isinstance(binding, Mapping):
        raise PermissionError("r14 log binding is absent")
    raw, observed = r14_integration._read_stable_regular_file(
        expected_path,
        expected_uid=os.getuid(),
        expected_mode=0o444,
        expected_nlink=1,
    )
    if (
        binding.get("path") != str(expected_path)
        or binding.get("file_sha256")
        != hashlib.sha256(raw).hexdigest()
        or binding.get("size_bytes") != len(raw)
        or binding.get("hardlink_count") != observed.st_nlink
        or binding.get("mode") != stat.S_IMODE(observed.st_mode)
    ):
        raise PermissionError("r14 runtime log binding changed")


def _validate_r14_preheartbeat_timeline(
    spec: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PermissionError("r14 runtime artifacts are absent")
    definitions = (
        (
            "launch_lease",
            "launch_lease_fingerprint",
            r14_integration.LAUNCH_LEASE_SCHEMA,
        ),
        (
            "precommit_phase_receipt",
            "phase_receipt_fingerprint",
            r14_integration.PHASE_RECEIPT_SCHEMA,
        ),
        (
            "attempt_commit",
            "attempt_commit_fingerprint",
            r14_integration.ATTEMPT_COMMIT_SCHEMA,
        ),
        (
            "materialization_claim",
            "materialization_claim_fingerprint",
            r14_integration.MATERIALIZATION_CLAIM_SCHEMA,
        ),
        (
            "start_ack_receipt",
            "phase_receipt_fingerprint",
            r14_integration.PHASE_RECEIPT_SCHEMA,
        ),
        (
            "child_prespawn_phase_receipt",
            "phase_receipt_fingerprint",
            r14_integration.PHASE_RECEIPT_SCHEMA,
        ),
    )
    loaded: dict[str, dict[str, object]] = {}
    previous_time: datetime | None = None
    previous_monotonic: int | None = None
    boot_id: object = None
    for name, fingerprint_field, schema in definitions:
        payload = r14_integration._read_sealed(
            Path(str(artifacts[name])),
            fingerprint_field=fingerprint_field,
            schema=schema,
            expected_mode=0o444,
        )
        created = _parse_archival_utc(
            payload.get("time_utc"),
            name=f"r14 {name}",
        )
        monotonic = payload.get("monotonic_ns")
        if boot_id is None:
            boot_id = payload.get("boot_id")
        if (
            payload.get("attempt_id") != spec.get("attempt_id")
            or payload.get("boot_id") != boot_id
            or isinstance(monotonic, bool)
            or not isinstance(monotonic, int)
            or monotonic <= 0
            or (
                previous_monotonic is not None
                and monotonic <= previous_monotonic
            )
            or (previous_time is not None and created < previous_time)
        ):
            raise PermissionError("r14 preheartbeat chronology changed")
        loaded[name] = payload
        previous_time = created
        previous_monotonic = monotonic
    if (
        loaded["attempt_commit"].get("systemd_unit_name")
        != R14_DUMMY_UNIT
        or loaded["materialization_claim"].get("systemd_control_group")
        is None
    ):
        raise PermissionError("r14 preheartbeat unit identity changed")
    return loaded


def _validate_r14_heartbeat_chain(
    spec: Mapping[str, object],
    *,
    invocation_id: str,
    claim: Mapping[str, object],
    chronology_anchor: Mapping[str, object],
) -> dict[str, object]:
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PermissionError("r14 runtime artifacts are absent")
    heartbeat_root = Path(str(artifacts.get("heartbeat_dir", "")))
    rows = sorted(heartbeat_root.iterdir())
    terminal = r14_integration._read_sealed(
        Path(str(artifacts["runtime_terminal"])),
        fingerprint_field="runtime_terminal_fingerprint",
        schema=r14_integration.RUNTIME_TERMINAL_SCHEMA,
        expected_mode=0o444,
    )
    previous_sha = r14_integration.sealed_file_sha256(claim)
    previous_monotonic = chronology_anchor.get("monotonic_ns")
    previous_time = _parse_archival_utc(
        chronology_anchor.get("time_utc"),
        name="r14 child-prespawn chronology anchor",
    )
    if (
        isinstance(previous_monotonic, bool)
        or not isinstance(previous_monotonic, int)
        or previous_monotonic <= 0
    ):
        raise PermissionError("r14 materialization claim chronology changed")
    heartbeat_process_identity: tuple[object, ...] | None = None
    if not rows:
        raise PermissionError("r14 heartbeat chain is empty")
    for sequence, path in enumerate(rows):
        if path.name != f"{sequence:012d}.json":
            raise PermissionError("r14 heartbeat sequence is not continuous")
        heartbeat = r14_integration._read_sealed(
            path,
            fingerprint_field="event_fingerprint",
            schema="cure-lite-v24-dr-runtime-heartbeat-v1",
            expected_mode=0o444,
        )
        monotonic = heartbeat.get("monotonic_ns")
        created = _parse_archival_utc(
            heartbeat.get("time_utc"),
            name=f"r14 heartbeat:{sequence}",
        )
        process_identity = (
            heartbeat.get("supervisor_pid"),
            heartbeat.get("supervisor_proc_starttime_ticks"),
            heartbeat.get("child_pid"),
            heartbeat.get("child_proc_starttime_ticks"),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in process_identity
        ):
            raise PermissionError("r14 heartbeat process identity changed")
        if heartbeat_process_identity is None:
            heartbeat_process_identity = process_identity
        elif process_identity != heartbeat_process_identity:
            raise PermissionError("r14 heartbeat process identity changed")
        if heartbeat.get("sequence") != sequence:
            raise PermissionError("r14 heartbeat identity changed")
        if (
            heartbeat.get("attempt_id") != spec.get("attempt_id")
            or heartbeat.get("boot_id") != claim.get("boot_id")
            or heartbeat.get("systemd_invocation_id") != invocation_id
            or heartbeat.get("previous_event_file_sha256") != previous_sha
            or isinstance(monotonic, bool)
            or not isinstance(monotonic, int)
            or monotonic <= previous_monotonic
            or created < previous_time
        ):
            raise PermissionError("r14 heartbeat hash/time chain changed")
        previous_sha = r14_integration.sealed_file_sha256(heartbeat)
        previous_monotonic = monotonic
        previous_time = created
    if (
        terminal.get("heartbeat_event_count") != len(rows)
        or terminal.get("last_heartbeat_path") != str(rows[-1])
        or terminal.get("last_heartbeat_file_sha256") != previous_sha
        or terminal.get("attempt_id") != spec.get("attempt_id")
        or terminal.get("systemd_invocation_id") != invocation_id
        or terminal.get("boot_id") != claim.get("boot_id")
    ):
        raise PermissionError("r14 runtime terminal heartbeat anchor changed")
    terminal_time = _parse_archival_utc(
        terminal.get("time_utc"),
        name="r14 runtime terminal",
    )
    terminal_monotonic = terminal.get("monotonic_ns")
    if (
        terminal_time < previous_time
        or isinstance(terminal_monotonic, bool)
        or not isinstance(terminal_monotonic, int)
        or terminal_monotonic <= previous_monotonic
    ):
        raise PermissionError("r14 heartbeat terminal chronology changed")
    return terminal


def _validate_r14_archival_chronology(
    *,
    b4_receipt: Mapping[str, object],
    authorization: Mapping[str, object],
    timeline: Mapping[str, Mapping[str, object]],
    runtime_terminal: Mapping[str, object],
    sidecar: Mapping[str, object],
    integration_terminal: Mapping[str, object],
) -> None:
    b4_time = _parse_archival_utc(
        b4_receipt.get("created_at_utc"),
        name="B4 receipt",
    )
    issued = _parse_archival_utc(
        authorization.get("issued_at_utc"),
        name="r14 authorization issuance",
    )
    expires = _parse_archival_utc(
        authorization.get("expires_at_utc"),
        name="r14 authorization expiry",
    )
    lease_time = _parse_archival_utc(
        timeline["launch_lease"].get("time_utc"),
        name="r14 launch lease",
    )
    runtime_terminal_time = _parse_archival_utc(
        runtime_terminal.get("time_utc"),
        name="r14 runtime terminal",
    )
    sidecar_time = _parse_archival_utc(
        sidecar.get("time_utc"),
        name="r14 systemd sidecar",
    )
    terminal_time = _parse_archival_utc(
        integration_terminal.get("created_at_utc"),
        name="r14 integration terminal",
    )
    if (
        not (
            b4_time
            <= issued
            <= lease_time
            <= runtime_terminal_time
            <= sidecar_time
            <= terminal_time
            <= expires
        )
        or issued >= expires
        or (expires - issued).total_seconds() > 300
    ):
        raise PermissionError("r14 archival authorization chronology changed")


def _validate_r14_runtime_chain(
    integration_closure: Mapping[str, object],
) -> dict[str, object]:
    authorization = integration_closure.get("authorization")
    if not isinstance(authorization, Mapping):
        raise PermissionError("r14 authorization is absent")
    spec = r14_integration._read_sealed(
        INTEGRATION_RUNTIME_SPEC_PATH,
        fingerprint_field="runtime_spec_fingerprint",
        schema=r14_integration.RUNTIME_SPEC_SCHEMA,
    )
    runtime_binding = authorization.get("runtime_spec_binding")
    if (
        not isinstance(runtime_binding, Mapping)
        or runtime_binding.get("path") != str(INTEGRATION_RUNTIME_SPEC_PATH)
        or runtime_binding.get("file_sha256")
        != r14_integration.file_sha256(INTEGRATION_RUNTIME_SPEC_PATH)
        or runtime_binding.get("runtime_spec_fingerprint")
        != spec.get("runtime_spec_fingerprint")
    ):
        raise PermissionError("r14 runtime-spec control binding changed")
    evidence = r14_integration._validate_supervisor_evidence(
        authorization,
    )
    terminal = r14_integration._read_sealed(
        INTEGRATION_TERMINAL_PATH,
        fingerprint_field="integration_terminal_fingerprint",
        schema=r14_integration.INTEGRATION_TERMINAL_SCHEMA,
    )
    if terminal.get("supervisor_evidence") != evidence:
        raise PermissionError("r14 supervisor evidence summary changed")

    bindings = authorization.get("executable_bindings")
    if not isinstance(bindings, Mapping):
        raise PermissionError("r14 executable source closure is absent")
    for name, path, digest in (
        ("integration_tool", R14_INTEGRATION_PATH, R14_INTEGRATION_SHA256),
        ("realizer", R14_SHARED_REALIZER_PATH, R14_SHARED_REALIZER_SHA256),
        ("dummy_child", R14_DUMMY_CHILD_PATH, R14_DUMMY_CHILD_SHA256),
        ("supervisor", COMPAT_SUPERVISOR_PATH, COMPAT_SUPERVISOR_SHA256),
    ):
        _require_r14_source_binding(
            bindings.get(name),
            path=path,
            digest=digest,
        )
    _require_r14_source_binding(
        authorization.get("template_binding"),
        path=R14_DUMMY_TEMPLATE_PATH,
        digest=R14_DUMMY_TEMPLATE_SHA256,
    )

    runtime_root = (INTEGRATION_ROOT / "runtime").absolute()
    artifacts = spec.get("artifacts")
    expected_artifacts = {
        "root": str(runtime_root),
        **{
            key: str(runtime_root / name)
            for key, name in r14_integration._SPEC_ARTIFACT_NAMES.items()
        },
    }
    systemd = spec.get("runtime", {}).get("systemd", {})
    source_bindings = spec.get("source_bindings")
    child = spec.get("child")
    if (
        not isinstance(artifacts, Mapping)
        or dict(artifacts) != expected_artifacts
        or not isinstance(systemd, Mapping)
        or systemd.get("unit_name") != R14_DUMMY_UNIT
        or not isinstance(source_bindings, Mapping)
        or source_bindings.get("supervisor_file_sha256")
        != COMPAT_SUPERVISOR_SHA256
        or source_bindings.get("child_entry_file_sha256")
        != R14_DUMMY_CHILD_SHA256
        or not isinstance(child, Mapping)
        or child.get("entrypoint_path") != str(R14_DUMMY_CHILD_PATH)
        or spec.get("execution_kind") != "systemd_integration_dummy"
        or spec.get("authorization") is not None
        or spec.get("environment") is not None
        or spec.get("scientific_preaccess") is not None
    ):
        raise PermissionError("r14 dummy runtime-spec closure changed")

    invocation_id = str(evidence["invocation_id"])
    timeline = _validate_r14_preheartbeat_timeline(spec)
    claim = timeline["materialization_claim"]
    runtime_terminal = _validate_r14_heartbeat_chain(
        spec,
        invocation_id=invocation_id,
        claim=claim,
        chronology_anchor=timeline["child_prespawn_phase_receipt"],
    )
    _validate_r14_log(
        runtime_terminal.get("stdout_log"),
        expected_path=Path(str(artifacts["stdout_log"])),
    )
    _validate_r14_log(
        runtime_terminal.get("stderr_log"),
        expected_path=Path(str(artifacts["stderr_log"])),
    )
    sidecar_path = (
        Path(str(artifacts["systemd_invocation_dir"]))
        / f"{invocation_id}.json"
    )
    sidecar = r14_integration._read_sealed(
        sidecar_path,
        fingerprint_field="systemd_terminal_fingerprint",
        schema=r14_integration.SYSTEMD_TERMINAL_SCHEMA,
        expected_mode=0o444,
    )
    expected_cgroup_suffix = f"/{R14_DUMMY_UNIT}"
    if (
        claim.get("systemd_control_group")
        != sidecar.get("systemd_control_group")
        or not str(claim.get("systemd_control_group", "")).endswith(
            expected_cgroup_suffix
        )
    ):
        raise PermissionError("r14 InvocationID/cgroup crosslink changed")

    dummy = r14_integration._read_sealed(
        Path(str(authorization["control_artifacts"]["dummy_artifact"])),
        fingerprint_field="dummy_artifact_fingerprint",
        schema=r14_integration.DUMMY_ARTIFACT_SCHEMA,
        expected_mode=0o444,
    )
    if (
        authorization.get("identity", {}).get("unit_name")
        != R14_DUMMY_UNIT
        or systemd.get("unit_name") != R14_DUMMY_UNIT
        or timeline["attempt_commit"].get("systemd_unit_name")
        != R14_DUMMY_UNIT
        or dummy.get("scenario_id") != R14_SCENARIO_ID
        or dummy.get("dataset_accessed") is not False
        or dummy.get("gpu_accessed") is not False
        or dummy.get("torch_imported") is not False
        or any(
            os.path.lexists(Path(str(artifacts[name])))
            for name in (
                "runtime_attestation",
                "gpu_lease_release_receipt",
                "consumed_start_failure_receipt",
            )
        )
    ):
        raise PermissionError("r14 dummy/payload boundary changed")

    b4_receipt, _root = compat_bridge._load_sealed(
        COMPATIBILITY_RECEIPT_PATH,
        fingerprint_field="receipt_fingerprint",
        schema=compat_bridge.RECEIPT_SCHEMA,
    )
    _validate_r14_archival_chronology(
        b4_receipt=b4_receipt,
        authorization=authorization,
        timeline=timeline,
        runtime_terminal=runtime_terminal,
        sidecar=sidecar,
        integration_terminal=terminal,
    )
    return {
        "runtime_spec": dict(spec),
        "supervisor_evidence": dict(evidence),
        "terminal": dict(terminal),
        "dummy": dict(dummy),
        "sidecar": dict(sidecar),
    }


def _snapshot_production_c4(
    *,
    shadow_reader=None,
    manager_reader=None,
    unit_path_policy_reader=None,
) -> dict[str, object]:
    archival = _load_r4_archival()
    authorization = archival["authorization"]
    receipt = archival["receipt"]
    fragment_identity = receipt["fragment_identity"]
    fragment_path = Path(str(fragment_identity["path"]))
    if manager_reader is None:
        manager_reader = compat_realizer.compat_c1.legacy.collect_manager_generation
    if shadow_reader is None:
        shadow_reader = compat_realizer.compat_c1.legacy.query_shadow
    if unit_path_policy_reader is None:
        unit_path_policy_reader = (
            lambda fragment: compat_realizer.compat_c1.legacy._observe_unit_path_policy(
                runner=__import__("subprocess").run,
                allowed_fragment=fragment,
            )
        )
    _, live_fragment = compat_realizer.compat_c1.legacy._stable_read_file(
        fragment_path,
    )
    live_fragment_projection = {
        key: live_fragment[key] for key in fragment_identity
    }
    manager = dict(manager_reader())
    live_policy = dict(unit_path_policy_reader(fragment_path))
    r14_integration._validate_unit_path_policy_transition(
        receipt["unit_path_policy"],
        live_policy,
        authorized_uid=os.getuid(),
        allow_generator_late_inode_rotation=True,
    )
    shadow = dict(shadow_reader())
    validated_shadow = (
        compat_realizer.compat_c1.legacy.validate_installed_shadow(
            shadow,
            fragment_identity=fragment_identity,
            authorization=authorization,
        )
    )
    expected_static = {
        "Id": COMPAT_UNIT,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
        "InvocationID": "",
    }
    if (
        live_fragment_projection != dict(fragment_identity)
        or not legacy._deep_exact_equal(
            manager,
            receipt["manager_generation"],
        )
        or not legacy._deep_exact_equal(
            validated_shadow,
            receipt["full_static_shadow"],
        )
        or any(
            shadow.get(name) != value
            for name, value in expected_static.items()
        )
        or fragment_identity.get("mode") != 0o600
        or fragment_identity.get("nlink") != 1
    ):
        raise PermissionError("production C4 static fragment changed")
    return {
        "archival": archival,
        "fragment": dict(fragment_identity),
        "shadow": dict(shadow),
        "manager": manager,
        "live_unit_path_policy": live_policy,
    }


def _compat_validate_release_closure(**kwargs):
    command = _ACTIVE_COMMAND.get()
    if command not in _ALLOWED_COMMANDS:
        raise PermissionError("c4 release phase is not explicitly selected")
    _require_l4_delegation(command)
    _require_lane_separation()
    _require_fixed_inputs(kwargs)
    bridge = _verify_active_phase_compatibility_receipt()
    _require_frozen_release_source()
    _require_component_sources()
    reader_kwargs = {
        "shadow_reader": kwargs.get("shadow_reader"),
        "manager_reader": kwargs.get("manager_reader"),
        "unit_path_policy_reader": kwargs.get("unit_path_policy_reader"),
    }
    before = _snapshot_production_c4(**reader_kwargs)
    sealed_policy = before["archival"]["receipt"]["unit_path_policy"]
    supplied_policy_reader = kwargs.get("unit_path_policy_reader")

    def frozen_policy_reader(fragment: Path) -> Mapping[str, object]:
        if supplied_policy_reader is None:
            live = compat_realizer.compat_c1.legacy._observe_unit_path_policy(
                runner=__import__("subprocess").run,
                allowed_fragment=fragment,
            )
        else:
            live = supplied_policy_reader(fragment)
        r14_integration._validate_unit_path_policy_transition(
            sealed_policy,
            live,
            authorized_uid=os.getuid(),
            allow_generator_late_inode_rotation=True,
        )
        return dict(sealed_policy)

    frozen_kwargs = dict(kwargs)
    frozen_kwargs["unit_path_policy_reader"] = frozen_policy_reader
    result = _frozen_validate_release_closure(**frozen_kwargs)
    _require_final_r14_supervisor_binding(result)
    r14 = _validate_r14_runtime_chain(result["integration"])
    after = _snapshot_production_c4(**reader_kwargs)
    archival = before["archival"]
    after_archival = after["archival"]
    realization = result.get("realization")
    evidence = bridge.get("compatibility_evidence_roots")
    realization_authorization_root = None
    realization_receipt_root = None
    if isinstance(realization, Mapping):
        realization_authorization_root = (
            compat_environment._bind_r4_archival_root(
                REALIZATION_AUTHORIZATION_PATH,
                realization.get("authorization_identity"),
            )
        )
        realization_receipt_root = (
            compat_environment._bind_r4_archival_root(
                REALIZATION_RECEIPT_PATH,
                realization.get("receipt_identity"),
            )
        )
    r14_integration._validate_unit_path_policy_transition(
        before["live_unit_path_policy"],
        after["live_unit_path_policy"],
        authorized_uid=os.getuid(),
        allow_generator_late_inode_rotation=True,
    )
    if (
        not isinstance(realization, Mapping)
        or not isinstance(archival, Mapping)
        or not isinstance(after_archival, Mapping)
        or not legacy._deep_exact_equal(archival, after_archival)
        or not legacy._deep_exact_equal(
            realization.get("authorization"),
            archival.get("authorization"),
        )
        or not legacy._deep_exact_equal(
            realization.get("receipt"),
            archival.get("receipt"),
        )
        or not legacy._deep_exact_equal(
            realization_authorization_root,
            archival.get("authorization_identity"),
        )
        or not legacy._deep_exact_equal(
            realization_receipt_root,
            archival.get("receipt_identity"),
        )
        or not isinstance(archival.get("compatibility_closure"), Mapping)
        or not isinstance(r14.get("supervisor_evidence"), Mapping)
        or not legacy._deep_exact_equal(
            {
                "fragment": before["fragment"],
                "shadow": before["shadow"],
                "manager": before["manager"],
            },
            {
                "fragment": after["fragment"],
                "shadow": after["shadow"],
                "manager": after["manager"],
            },
        )
        or not isinstance(evidence, Mapping)
        or evidence.get("unit_realization_authorization", {}).get("path")
        != str(REALIZATION_AUTHORIZATION_PATH)
        or evidence.get("unit_realization_receipt", {}).get("path")
        != str(REALIZATION_RECEIPT_PATH)
    ):
        raise PermissionError("c4 archival release closure diverged")
    return result


def _configure() -> None:
    _require_frozen_release_source()
    _require_component_sources()
    _require_disjoint_c4_runtime_identity()
    compat_c1.FROZEN_RELEASE_PATH = FROZEN_RELEASE_PATH
    compat_c1.FROZEN_RELEASE_SHA256 = FROZEN_RELEASE_SHA256
    compat_c1._FROZEN_RELEASE_LOAD_IDENTITY = (
        _FROZEN_RELEASE_LOAD_IDENTITY
    )
    compat_c1.COMPAT_REALIZER_PATH = COMPAT_REALIZER_PATH
    compat_c1.COMPAT_REALIZER_SHA256 = COMPAT_REALIZER_SHA256
    compat_c1.COMPAT_SUPERVISOR_PATH = COMPAT_SUPERVISOR_PATH
    compat_c1.COMPAT_SUPERVISOR_SHA256 = COMPAT_SUPERVISOR_SHA256
    compat_c1.compat_realizer = compat_realizer
    compat_c1.compat_supervisor = compat_supervisor
    r14_integration.realizer = r14_shared_realizer
    legacy.integration = r14_integration
    compat_c1.COMPAT_UNIT = COMPAT_UNIT
    compat_c1.COMPAT_RUNTIME_SPEC_PATH = COMPAT_RUNTIME_SPEC_PATH
    compat_c1.COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
        COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
    )
    compat_c1.COMPAT_RUNTIME_ARTIFACT_ROOT = COMPAT_RUNTIME_ARTIFACT_ROOT
    compat_c1.COMPAT_GPU_LEASE_ROOT = COMPAT_GPU_LEASE_ROOT
    compat_c1.COMPAT_RUN_ROOT_ALIAS_PATH = COMPAT_RUN_ROOT_ALIAS_PATH
    compat_c1.COMPAT_RESULT_RECEIPT_ALIAS_PATH = (
        COMPAT_RESULT_RECEIPT_ALIAS_PATH
    )
    compat_c1.SCIENTIFIC_RUN_ROOT = SCIENTIFIC_RUN_ROOT
    compat_c1.SCIENTIFIC_RESULT_RECEIPT_PATH = (
        SCIENTIFIC_RESULT_RECEIPT_PATH
    )
    compat_c1.COMPATIBILITY_RECEIPT_PATH = COMPATIBILITY_RECEIPT_PATH
    compat_c1.POLICY_PATH = POLICY_PATH
    compat_c1.STABILITY_PATH = STABILITY_PATH
    compat_c1.POSTCLEANUP_PATH = POSTCLEANUP_PATH
    compat_c1.INTEGRATION_AUTHORIZATION_PATH = (
        INTEGRATION_AUTHORIZATION_PATH
    )
    compat_c1.INTEGRATION_RECEIPT_PATH = INTEGRATION_RECEIPT_PATH
    compat_c1.REALIZATION_AUTHORIZATION_PATH = (
        REALIZATION_AUTHORIZATION_PATH
    )
    compat_c1.REALIZATION_RECEIPT_PATH = REALIZATION_RECEIPT_PATH
    compat_c1._require_lane_separation = _require_lane_separation
    compat_c1._require_build_phase_pristine = (
        _require_build_phase_pristine
    )
    compat_c1._require_prelaunch_scientific_pristine = (
        _require_prelaunch_scientific_pristine
    )
    compat_c1._load_fixed_compat_runtime_spec = (
        _load_fixed_compat_runtime_spec
    )
    compat_c1._verify_compatibility_receipt = (
        _verify_compatibility_receipt
    )
    compat_c1._verify_active_phase_compatibility_receipt = (
        _verify_active_phase_compatibility_receipt
    )
    compat_c1._require_fixed_inputs = _require_fixed_inputs
    compat_c1._compat_validate_release_closure = (
        _compat_validate_release_closure
    )
    compat_c1._materialize_spec_body = _materialize_spec_body
    compat_c1._validate_prewrite_spec_body = _validate_prewrite_spec_body
    compat_c1._compat_write_sealed = _compat_write_sealed
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


_frozen_write_sealed = _guard_release_callable(
    compat_c1._frozen_write_sealed,
    "build-spec",
)


def _materialize_spec_body(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    if (
        Path(path).absolute() != COMPAT_RUNTIME_SPEC_PATH.absolute()
        or fingerprint_field != "runtime_spec_fingerprint"
        or fingerprint_field in body
    ):
        raise PermissionError("c4 prewrite spec output contract changed")
    return {
        **dict(body),
        fingerprint_field: legacy.stable_fingerprint(body),
    }


def _validate_prewrite_spec_body(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    payload = _materialize_spec_body(
        path, body, fingerprint_field=fingerprint_field,
    )
    compat_supervisor.validate_prewrite_spec(payload)
    return payload


def _compat_write_sealed(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    _require_l4_delegation("build-spec")
    payload = _materialize_spec_body(
        path, body, fingerprint_field=fingerprint_field,
    )
    preview = _APPROVED_PREWRITE_SPEC.get()
    if (
        not isinstance(preview, Mapping)
        or not legacy._deep_exact_equal(payload, dict(preview))
    ):
        raise PermissionError("runtime spec differs from its mutation-free preview")
    return _frozen_write_sealed(
        path, body, fingerprint_field=fingerprint_field,
    )


def _preview_build_spec(args: object) -> dict[str, object]:
    _require_l4_delegation("build-spec")
    return compat_c1._preview_build_spec(args)


def _compat_load_sealed(path: Path, **kwargs):
    _require_frozen_release_source()
    target = Path(path).absolute()
    if (
        target == legacy.SCIENTIFIC_ACCESS_AUDIT_PATH.absolute()
        and kwargs.get("fingerprint_field") == "receipt_fingerprint"

        and kwargs.get("schema") == FICTIONAL_ACCESS_AUDIT_SCHEMA
    ):
        kwargs["schema"] = AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
    return compat_c1._frozen_load_sealed(target, **kwargs)


def verify_compatibility_identity() -> dict[str, object]:
    _configure()
    realizer = compat_realizer.verify_compatibility_identity()
    supervisor = compat_supervisor.verify_compatibility_identity()
    if (
        realizer.get("unit_name") != COMPAT_UNIT
        or supervisor.get("unit_name") != COMPAT_UNIT
        or supervisor.get("runtime_spec_path")
        != str(COMPAT_RUNTIME_SPEC_PATH)
        or legacy.UNIT_NAME != COMPAT_UNIT
        or legacy.RUNTIME_SPEC_PATH != COMPAT_RUNTIME_SPEC_PATH
    ):
        raise PermissionError("c4 release identity changed")
    return {
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_generation": "c4",
        "unit_name": COMPAT_UNIT,
        "runtime_spec_path": str(COMPAT_RUNTIME_SPEC_PATH),
        "runtime_launch_authorization_path": str(
            COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "runtime_artifact_root": str(COMPAT_RUNTIME_ARTIFACT_ROOT),
        "gpu_lease_root": str(COMPAT_GPU_LEASE_ROOT),
        "compatibility_receipt_path": str(COMPATIBILITY_RECEIPT_PATH),
        "authoritative_access_audit_schema": (
            AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        ),
        "fictional_access_audit_schema_accepted": False,
        "frozen_c1_release_path": str(FROZEN_RELEASE_PATH),
        "frozen_c1_release_file_sha256": FROZEN_RELEASE_SHA256,
        "bridge_path": str(COMPAT_BRIDGE_PATH),
        "bridge_file_sha256": COMPAT_BRIDGE_SHA256,
        "supervisor_path": str(COMPAT_SUPERVISOR_PATH),
        "supervisor_file_sha256": COMPAT_SUPERVISOR_SHA256,
        "adapter_path": str(COMPAT_ADAPTER_PATH),
        "adapter_file_sha256": COMPAT_ADAPTER_SHA256,
        "realizer_path": str(COMPAT_REALIZER_PATH),
        "realizer_file_sha256": COMPAT_REALIZER_SHA256,
        "environment_path": str(COMPAT_ENVIRONMENT_PATH),
        "environment_file_sha256": COMPAT_ENVIRONMENT_SHA256,
        "scope_handoff_path": str(SCOPE_HANDOFF_PATH),
        "stability_attempt_path": str(STABILITY_ATTEMPT_PATH),
        "r14_root": str(INTEGRATION_ROOT),
        "r14_scenario_id": R14_SCENARIO_ID,
        "r14_dummy_unit": R14_DUMMY_UNIT,
        "r14_integration_file_sha256": R14_INTEGRATION_SHA256,
        "r14_shared_realizer_file_sha256": R14_SHARED_REALIZER_SHA256,
        "r14_dummy_child_file_sha256": R14_DUMMY_CHILD_SHA256,
        "r14_dummy_template_file_sha256": R14_DUMMY_TEMPLATE_SHA256,
        "scientific_run_root": str(SCIENTIFIC_RUN_ROOT),
        "scientific_result_receipt_path": str(
            SCIENTIFIC_RESULT_RECEIPT_PATH
        ),
        "allowed_splits": ["D_R"],
        "optimizer_steps_authorized": 0,
        "parameter_updates_authorized": 0,
        "training_authorized": False,
        "gpu_accessed": False,
        "fresh_scientific_attempt": False,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def _fixed_build_values(args: object) -> dict[str, object]:
    return {
        "policy_path": args.environment_policy,
        "precleanup_path": args.precleanup_receipt,
        "cleanup_plan_path": args.cleanup_plan,
        "cleanup_authorization_path": args.cleanup_authorization,
        "cleanup_receipt_path": args.cleanup_receipt,
        "stability_path": args.stability_receipt,
        "postcleanup_path": args.postcleanup_audit,
        "integration_authorization_path": args.integration_authorization,
        "integration_receipt_path": args.integration_receipt,
        "realization_authorization_path": (
            args.unit_realization_authorization
        ),
        "realization_receipt_path": args.unit_realization_receipt,
    }


def _install_inherited_release_guards() -> None:
    """Make every inherited mutation entry require one L4 main capability."""

    guarded_main = _guard_release_callable(
        legacy.main,
        None,
        bind_argv=True,
    )
    guarded_build = _guard_release_callable(legacy.build_spec, "build-spec")
    guarded_authorize = _guard_release_callable(
        legacy.authorize_launch,
        "authorize-launch",
    )
    guarded_bound_writer = _guard_release_callable(
        legacy._write_sealed_bound,
        None,
    )
    guarded_private_directory = _guard_release_callable(
        legacy._private_directory,
        None,
    )
    guarded_runtime_directories = _guard_release_callable(
        legacy._create_runtime_directories_and_verify_leaves,
        "build-spec",
    )

    def reject_c1_entrypoint(*_args, **_kwargs):
        raise PermissionError(
            "c1 release entrypoint is not a c4 delegation target"
        )

    compat_c1.__dict__.pop("__getattr__", None)
    legacy.main = guarded_main
    legacy.build_spec = guarded_build
    legacy.authorize_launch = guarded_authorize
    legacy._write_sealed_bound = guarded_bound_writer
    legacy._private_directory = guarded_private_directory
    legacy._create_runtime_directories_and_verify_leaves = (
        guarded_runtime_directories
    )
    legacy._write_sealed = _compat_write_sealed
    legacy.validate_release_closure = _compat_validate_release_closure
    compat_c1.main = reject_c1_entrypoint
    compat_c1.verify_compatibility_identity = reject_c1_entrypoint
    compat_c1._configure = reject_c1_entrypoint
    compat_c1._frozen_write_sealed = _frozen_write_sealed
    compat_c1._compat_write_sealed = _compat_write_sealed


_install_inherited_release_guards()


if __name__ == "__main__":
    raise SystemExit(main())
