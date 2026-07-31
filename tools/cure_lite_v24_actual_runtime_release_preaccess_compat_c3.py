#!/usr/bin/env python3
"""Create-only runtime release wrapper for compatibility generation c3.

The already audited c1 release producer is the implementation authority.  Its
exact bytes are read through no-follow file descriptors, hashed, and compiled
only after the frozen hash matches.  This wrapper then narrows that producer to
the disjoint c3 runtime namespace and adds three c3-specific consumers:

* the sealed c3 compatibility receipt (including its environment lineage);
* the fresh c3 environment scope cross-binding;
* the archival c3 unit-realization closure.

This module can only build one runtime specification or authorize its one
launch.  It never starts systemd, accesses a payload, retries, resumes, creates
a new scientific attempt, or authorizes D_V/D_T.
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
    REPOSITORY
    / "tools/cure_lite_v24_actual_runtime_release_preaccess_compat_c1.py"
).resolve()
FROZEN_RELEASE_SHA256 = (
    "395a013ff4f14160a0ac4e9845497caf9ecbaa6f2eeb3aa88fad54b63f514cfa"
)
COMPAT_BRIDGE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c3.py"
).resolve()
COMPAT_BRIDGE_SHA256 = (
    "3bf4caabfce8fd302b74b59b021bb37ba839f3c11226fde9b347ed2e574badb7"
)
COMPAT_REALIZER_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c3.py"
).resolve()
COMPAT_REALIZER_SHA256 = (
    "cdbbe4355b29519d2b3da858732bc8531396a59f5a3f1cfacdb578323fe33de1"
)
COMPAT_SUPERVISOR_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c3.py"
).resolve()
COMPAT_SUPERVISOR_SHA256 = (
    "536df32b66d8de3891aad4b454c886ec6b93617bc6759db098b2a442a4209afd"
)
COMPAT_ENVIRONMENT_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_environment_preaccess_compat_c3.py"
).resolve()
COMPAT_ENVIRONMENT_SHA256 = (
    "7bf9e268ffbd11491fc3f5efdd7a89611492c4e58ef587c0d9752a46e4e85a7e"
)
COMPAT_ADAPTER_PATH = (
    REPOSITORY
    / "tools/"
    "run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c3.py"
).resolve()

COMPAT_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c3.service"
)
COMPAT_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_runtime_spec.json"
)
COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c3_"
        "runtime_launch_authorization.json"
    )
)
COMPAT_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_runtime_artifacts"
)
COMPAT_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_gpu_lease"
)
COMPAT_RUN_ROOT_ALIAS_PATH = (
    REPOSITORY
    / "runs/irstd1k_stage_a_seed42/"
    "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c3"
)
COMPAT_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_receipt.json"
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
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c3_receipt.json"
)

POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c3.json"
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
    / "runtime_environment_stability_receipt_preaccess_compat_c3.json"
)
POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c3.json"
)
INTEGRATION_ROOT = (
    EVIDENCE_ROOT
    / "supervisor_v2_systemd_integration_preaccess_compat_c3_r13"
)
INTEGRATION_AUTHORIZATION_PATH = (
    INTEGRATION_ROOT / "control/authorization.json"
)
INTEGRATION_RECEIPT_PATH = (
    INTEGRATION_ROOT / "control/integration-receipt.json"
)
REALIZATION_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c3_unit_realization_authorization.json"
)
REALIZATION_RECEIPT_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c3_unit_realization_receipt.json"
)
REALIZATION_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c3_unit_realization_terminal.json"
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


def _build_disjoint_c3_runtime_identity_gate():
    """Freeze the c3 runtime identity and its separation from older lanes."""

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
        )
    )
    expected_aliases = (
        COMPAT_RUN_ROOT_ALIAS_PATH.absolute(),
        COMPAT_RESULT_RECEIPT_ALIAS_PATH.absolute(),
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
            )
        )
        aliases = (
            Path(COMPAT_RUN_ROOT_ALIAS_PATH).absolute(),
            Path(COMPAT_RESULT_RECEIPT_ALIAS_PATH).absolute(),
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
            raise PermissionError("c3 runtime identity/path isolation changed")

    return require


_require_disjoint_c3_runtime_identity = (
    _build_disjoint_c3_runtime_identity_gate()
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
        raise PermissionError("c3 release source path is unsafe")
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
            raise PermissionError("c3 release source is unsafe")
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
        raise PermissionError("c3 release source generation changed")
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
        raise PermissionError("bound c3 release component source changed")
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
        raise PermissionError("bound c3 release component was replaced")
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

compat_c1, _FROZEN_RELEASE_LOAD_IDENTITY = _load_verified(
    FROZEN_RELEASE_PATH,
    FROZEN_RELEASE_SHA256,
    name=(
        "tools._cure_lite_v24_actual_runtime_release_preaccess_"
        "compat_c1_verified_for_c3"
    ),
)
compat_bridge, _BRIDGE_LOAD_IDENTITY = _load_when_frozen(
    COMPAT_BRIDGE_PATH,
    COMPAT_BRIDGE_SHA256,
    name="tools._cure_lite_v24_preaccess_compat_c3_verified_for_release",
)
compat_realizer, _REALIZER_LOAD_IDENTITY = _load_when_frozen(
    COMPAT_REALIZER_PATH,
    COMPAT_REALIZER_SHA256,
    name="tools._cure_lite_v24_realizer_compat_c3_verified_for_release",
)
compat_supervisor, _SUPERVISOR_LOAD_IDENTITY = _load_when_frozen(
    COMPAT_SUPERVISOR_PATH,
    COMPAT_SUPERVISOR_SHA256,
    name="tools._cure_lite_v24_supervisor_compat_c3_verified_for_release",
)
compat_environment, _ENVIRONMENT_LOAD_IDENTITY = _load_when_frozen(
    COMPAT_ENVIRONMENT_PATH,
    COMPAT_ENVIRONMENT_SHA256,
    name="tools._cure_lite_v24_environment_compat_c3_verified_for_release",
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
    ):
        if (
            identity is None
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise PermissionError("c3 release component hash is not frozen")
        raw, observed = _stable_source_bytes(path)
        if hashlib.sha256(raw).hexdigest() != digest or observed != identity:
            raise PermissionError("c3 release component generation changed")


def _build_l3_delegation_gate():
    """Create the sole delegated main and its closure-private capability."""

    seal = object()
    state: ContextVar[
        tuple[object, str, tuple[str, ...]] | None
    ] = ContextVar("cure_lite_v24_c3_release_delegation", default=None)

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
            raise PermissionError("c3 release delegation is not authorized")
        _require_frozen_release_source()
        _require_component_sources()

    def delegated_main(argv: Sequence[str] | None = None) -> int:
        materialized = list(sys.argv[1:] if argv is None else argv)
        _require_frozen_release_source()
        _require_component_sources()
        args = legacy._parser().parse_args(materialized)
        if args.command not in _ALLOWED_COMMANDS:
            raise PermissionError("c3 release command is not allowed")
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
                        "c3 build-spec preview is nondeterministic"
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


main, _require_l3_delegation = _build_l3_delegation_gate()
del _build_l3_delegation_gate


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
            raise PermissionError("c3 release phase is not explicitly selected")
        delegated_argv: Sequence[str] | None = None
        if bind_argv:
            supplied = args[0] if args else kwargs.get("argv")
            delegated_argv = list(
                sys.argv[1:] if supplied is None else supplied
            )
        _require_l3_delegation(selected, argv=delegated_argv)
        return function(*args, **kwargs)

    guarded.__name__ = getattr(function, "__name__", "guarded_c3_release")
    guarded.__doc__ = getattr(function, "__doc__", None)
    return guarded


def _require_lane_separation() -> None:
    if any(
        os.path.lexists(path)
        for path in (*BLOCKED_RUNTIME_PATHS, *C1_RUNTIME_PATHS, *C2_RUNTIME_PATHS)
    ):
        raise PermissionError(
            "direct, c1, or c2 predecessor runtime lane is no longer pristine"
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
        raise PermissionError("c3 build-spec namespace is already materialized")


def _require_prelaunch_scientific_pristine() -> None:
    if any(
        os.path.lexists(path)
        for path in (
            SCIENTIFIC_RUN_ROOT,
            SCIENTIFIC_RESULT_RECEIPT_PATH,
        )
    ):
        raise PermissionError(
            "scientific r2 output exists before c3 launch authorization"
        )


def _load_fixed_compat_runtime_spec() -> dict[str, object]:
    payload = compat_c1._frozen_load_sealed(
        COMPAT_RUNTIME_SPEC_PATH,
        fingerprint_field="runtime_spec_fingerprint",
        schema=compat_supervisor.RUNTIME_SPEC_SCHEMA,
    )
    if not isinstance(payload, Mapping):
        raise PermissionError("fixed c3 runtime spec is not a mapping")
    return dict(payload)


def _verify_environment_cross_binding(
    bridge_receipt: Mapping[str, object],
) -> dict[str, object]:
    old_contract, c3_contract, _roots = (
        compat_environment.replay_old_scope_and_handoff()
    )
    archival = compat_realizer.validate_archival_realization_chain(
        REALIZATION_AUTHORIZATION_PATH,
        REALIZATION_RECEIPT_PATH,
    )
    archival = compat_environment.validate_c3_realization_archival(
        archival,
        contract=c3_contract,
    )
    policy = compat_environment.frozen.load_sealed_receipt(
        POLICY_PATH,
        fingerprint_field="policy_fingerprint",
    )
    stability = compat_environment.frozen.load_sealed_receipt(
        STABILITY_PATH,
        fingerprint_field="stability_receipt_fingerprint",
    )
    postcleanup = compat_environment.frozen.load_sealed_receipt(
        POSTCLEANUP_PATH,
        fingerprint_field="receipt_fingerprint",
    )
    closure = compat_environment.validate_c3_environment_closure(
        policy,
        stability,
        postcleanup,
        archival=archival,
        c3_contract=c3_contract,
    )
    if (
        bridge_receipt.get("historical_environment_contract")
        != compat_environment._normalized_contract(old_contract)
        or bridge_receipt.get("current_environment_contract")
        != compat_environment._normalized_contract(c3_contract)
        or not legacy._deep_exact_equal(
            closure.get("realization"), archival,
        )
    ):
        raise PermissionError(
            "c3 bridge/environment/realization cross-binding changed"
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
        or result.get("runtime_compatibility_id") != "c3"
        or result.get("automatic_retry") is not False
        or result.get("resume") is not False
        or result.get("D_R_payload_accessed") is not False
        or result.get("D_V_payload_accessed") is not False
        or result.get("D_T_payload_accessed") is not False
        or not isinstance(bridge_root, Mapping)
        or bridge_root.get("path") != str(COMPAT_BRIDGE_PATH)
        or bridge_root.get("file_sha256") != COMPAT_BRIDGE_SHA256
    ):
        raise PermissionError("c3 compatibility receipt is not exact")
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
    raise PermissionError("c3 release phase is not explicitly selected")


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
            raise PermissionError(f"c3 release input path changed: {name}")


def _require_final_r13_supervisor_binding(
    closure: Mapping[str, object],
) -> None:
    """Bind the archived r13 proof to this exact supervisor generation."""

    integration = closure.get("integration")
    if not isinstance(integration, Mapping):
        raise PermissionError(
            "r13 integration is not bound to the final c3 supervisor "
            "generation"
        )
    authorization = integration.get("authorization")
    identities = integration.get("identities")
    if (
        not isinstance(authorization, Mapping)
        or not isinstance(identities, Mapping)
    ):
        raise PermissionError(
            "r13 integration is not bound to the final c3 supervisor "
            "generation"
        )
    authorization_identity = identities.get("authorization")
    receipt_identity = identities.get("receipt")
    scenario_root = authorization.get("scenario_root")
    executable_bindings = authorization.get("executable_bindings")
    if (
        not isinstance(authorization_identity, Mapping)
        or not isinstance(receipt_identity, Mapping)
        or not isinstance(scenario_root, Mapping)
        or not isinstance(executable_bindings, Mapping)
    ):
        raise PermissionError(
            "r13 integration is not bound to the final c3 supervisor "
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
            "r13 integration is not bound to the final c3 supervisor "
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
    if (
        authorization_path.parent.parent != integration_root
        or receipt_path.parent.parent != integration_root
        or authorization_identity.get("path")
        != str(authorization_path)
        or receipt_identity.get("path") != str(receipt_path)
        or scenario_root.get("path") != str(integration_root)
        or dict(supervisor_binding) != expected_binding
    ):
        raise PermissionError(
            "r13 integration is not bound to the final c3 supervisor "
            "generation"
        )


def _compat_validate_release_closure(**kwargs):
    command = _ACTIVE_COMMAND.get()
    if command not in _ALLOWED_COMMANDS:
        raise PermissionError("c3 release phase is not explicitly selected")
    _require_l3_delegation(command)
    _require_lane_separation()
    _require_fixed_inputs(kwargs)
    bridge = _verify_active_phase_compatibility_receipt()
    _require_frozen_release_source()
    _require_component_sources()
    result = _frozen_validate_release_closure(**kwargs)
    _require_final_r13_supervisor_binding(result)
    archival = compat_realizer.validate_archival_realization_chain(
        REALIZATION_AUTHORIZATION_PATH,
        REALIZATION_RECEIPT_PATH,
    )
    realization = result.get("realization")
    evidence = bridge.get("compatibility_evidence_roots")
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
        or not isinstance(archival.get("compatibility_closure"), Mapping)
        or not isinstance(evidence, Mapping)
        or evidence.get("unit_realization_authorization", {}).get("path")
        != str(REALIZATION_AUTHORIZATION_PATH)
        or evidence.get("unit_realization_receipt", {}).get("path")
        != str(REALIZATION_RECEIPT_PATH)
    ):
        raise PermissionError("c3 archival release closure diverged")
    return result


def _configure() -> None:
    _require_frozen_release_source()
    _require_component_sources()
    _require_disjoint_c3_runtime_identity()
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
        raise PermissionError("c3 prewrite spec output contract changed")
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
    _require_l3_delegation("build-spec")
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
    _require_l3_delegation("build-spec")
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
        raise PermissionError("c3 release identity changed")
    return {
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_generation": "c3",
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
        "adapter_path": str(COMPAT_ADAPTER_PATH),
        "realizer_path": str(COMPAT_REALIZER_PATH),
        "environment_path": str(COMPAT_ENVIRONMENT_PATH),
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
    """Make every inherited mutation entry require one L3 main capability."""

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
            "c1 release entrypoint is not a c3 delegation target"
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
