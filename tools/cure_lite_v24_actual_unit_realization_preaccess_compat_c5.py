#!/usr/bin/env python3
"""Create-only unit realizer for runtime compatibility generation c5.

The audited c1 realizer is read through a no-follow descriptor and its frozen
SHA-256 is proved before any predecessor byte is compiled.  This wrapper then
rebinds that verified implementation to the disjoint c5 unit namespace.

Only creation of the new c5 static fragment and ``daemon-reload`` are
authorized.  The original r2 and c1/c2/c3/c4 units are immutable.  This module
never starts, enables, stops, removes, retries, resumes, or accesses a
scientific payload.
"""

from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import FunctionType, ModuleType
from typing import Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
COMPAT_REALIZER_PATH = Path(__file__).resolve()
FROZEN_REALIZER_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c1.py"
).resolve()
FROZEN_REALIZER_SHA256 = (
    "7bfc5944378d552f9f12654da5234762452f8dc5ee49f1bced47554bcbd58ece"
)

EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()
COMPAT_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c5.service"
)
PROTECTED_ORIGINAL_UNIT = "cure-lite-v24-gcr-pacre-dr-r2.service"
PROTECTED_C1_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service"
)
PROTECTED_C2_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service"
)
PROTECTED_C3_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c3.service"
)
PROTECTED_C4_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service"
)
COMPAT_UNIT_DIRECTORY = Path(f"/run/user/{os.getuid()}/systemd/user")
PROTECTED_ORIGINAL_FRAGMENT_PATH = (
    COMPAT_UNIT_DIRECTORY / PROTECTED_ORIGINAL_UNIT
)
PROTECTED_C1_FRAGMENT_PATH = COMPAT_UNIT_DIRECTORY / PROTECTED_C1_UNIT
PROTECTED_C2_FRAGMENT_PATH = COMPAT_UNIT_DIRECTORY / PROTECTED_C2_UNIT
PROTECTED_C3_FRAGMENT_PATH = COMPAT_UNIT_DIRECTORY / PROTECTED_C3_UNIT
PROTECTED_C4_FRAGMENT_PATH = COMPAT_UNIT_DIRECTORY / PROTECTED_C4_UNIT
COMPAT_TEMPLATE_PATH = (
    REPOSITORY
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c5.service.template"
).resolve()
COMPAT_TEMPLATE_SHA256 = (
    "f2a3da0862addb90e61301c97e0d5c1d109e8cbf59ad86c2e5130235f8387216"
)
COMPAT_SUPERVISOR_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c5.py"
).resolve()
COMPAT_SUPERVISOR_SHA256 = (
    "12c93e469b03e5b4b6f626e875a0934f603061c840b6614221748ac2cdd3dda2"
)
COMPAT_BRIDGE_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c5.py"
).resolve()
COMPAT_BRIDGE_SOURCE_SHA256 = (
    "388843b9b840db41610d57543f4982666cdf442ba81fa5acb208033de062319f"
)
COMPAT_BRIDGE_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c5_authorization.json"
)
COMPAT_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c5_runtime_spec.json"
)
COMPAT_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c5_unit_realization_authorization.json"
)
COMPAT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c5_unit_realization_receipt.json"
)
COMPAT_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c5_unit_realization_terminal.json"
)

COMPATIBILITY_CLOSURE_SCHEMA = (
    "cure-lite-v24-preaccess-compat-c5-unit-realization-closure-v1"
)
COMPATIBILITY_CLOSURE_KEY = "compatibility_closure"
PROTECTED_UNITS_KEY = "protected_unit_generations"
BRIDGE_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c5-authorization-v1"
)
B5_INSTRUCTION_ID = "user-2026-07-31-modify-after-c4-failure-v1"
B5_AUTHORIZATION_BASIS = "user instruction: 修改后继续"
BRIDGE_MUTATION_AUTHORITY = {
    "compatibility_receipt_creation_authorized": True,
    "compatibility_terminal_creation_authorized": True,
    "environment_scope_handoff_authorized": True,
    "environment_metadata_audit_authorized": True,
    "c5_unit_realization_authorized": True,
    "c4_unit_mutation_authorized": False,
    "c4_evidence_mutation_authorized": False,
    "runtime_spec_creation_authorized": False,
    "runtime_launch_authorization_creation_authorized": False,
    "unit_start_authorized": False,
    "unit_enable_authorized": False,
    "payload_access_authorized": False,
}
BRIDGE_SCIENTIFIC_AUTHORITY = {
    "D_R_payload_authorized": False,
    "D_V_payload_authorized": False,
    "D_T_payload_authorized": False,
    "training_authorized": False,
    "materialization_authorized": False,
    "automatic_retry": False,
    "resume": False,
    "fresh_scientific_attempt": False,
}
_BRIDGE_STATE_KEYS = frozenset(
    {
        "Id",
        "LoadState",
        "ActiveState",
        "SubState",
        "UnitFileState",
        "NRestarts",
        "FragmentPath",
        "InvocationID",
    }
)
_REALIZATION_ACTIONS = (
    "install-runtime-static-fragment",
    "daemon-reload",
    "verify-static-shadow",
)
_RECEIPT_STATIC_STATE = {
    "LoadState": "loaded",
    "ActiveState": "inactive",
    "SubState": "dead",
    "UnitFileState": "static",
    "Restart": "no",
    "NRestarts": "0",
    "NeedDaemonReload": "no",
    "InvocationID": "",
}
RUNTIME_PHASE_PREACTIVATION = "preactivation"
RUNTIME_PHASE_COMMIT = "commit"
RUNTIME_PHASE_CLAIM = "claim"
RUNTIME_PHASE_VERIFY = "verify"
RUNTIME_PHASE_RUN_ONCE = "run_once"
RUNTIME_PHASE_FINALIZE_SUCCESS = "finalize_success"
RUNTIME_PHASE_FINALIZE_FAILURE = "finalize_failure"
RUNTIME_PHASES = frozenset(
    {
        RUNTIME_PHASE_PREACTIVATION,
        RUNTIME_PHASE_COMMIT,
        RUNTIME_PHASE_CLAIM,
        RUNTIME_PHASE_VERIFY,
        RUNTIME_PHASE_RUN_ONCE,
        RUNTIME_PHASE_FINALIZE_SUCCESS,
        RUNTIME_PHASE_FINALIZE_FAILURE,
    }
)
_RUNTIME_PHASE_CONTEXT: ContextVar[str] = ContextVar(
    "cure_lite_v24_r5_runtime_phase",
    default=RUNTIME_PHASE_PREACTIVATION,
)
_SOURCE_FIELDS = (
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


def _canonical_json(value: object) -> str:
    """Canonical profile owned by R5; never used for foreign B5 evidence."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _resolve_runtime_phase(
    *,
    allow_runtime_activation: bool,
    runtime_phase: str | None,
) -> str:
    if type(allow_runtime_activation) is not bool:
        raise TypeError("allow_runtime_activation must be boolean")
    if runtime_phase is None:
        if allow_runtime_activation:
            raise PermissionError(
                "active runtime verification requires an explicit phase",
            )
        return RUNTIME_PHASE_PREACTIVATION
    if (
        not isinstance(runtime_phase, str)
        or runtime_phase not in RUNTIME_PHASES
    ):
        raise PermissionError("runtime phase is not an exact closed state")
    expected_activation = runtime_phase != RUNTIME_PHASE_PREACTIVATION
    if allow_runtime_activation is not expected_activation:
        raise PermissionError(
            "runtime activation flag and explicit phase disagree",
        )
    return runtime_phase


def _stable_source_bytes(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, object]]:
    target = Path(path)
    before = os.lstat(target)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or target.resolve(strict=True) != target
    ):
        raise PermissionError("compatibility realizer source is unsafe")
    descriptor = os.open(
        target,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
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
        linked = os.stat(target, follow_symlinks=False)
    finally:
        os.close(descriptor)
    snapshots = [
        tuple(int(getattr(item, field)) for field in _SOURCE_FIELDS)
        for item in (before, opened, finished, linked)
    ]
    if any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
        raise PermissionError(
            "compatibility realizer source changed during read",
        )
    raw = b"".join(chunks)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise PermissionError("bound source SHA-256 changed")
    return raw, {
        "path": str(target),
        "resolved_path": str(target),
        "path_is_symlink": False,
        "file_sha256": digest,
        "device": finished.st_dev,
        "inode": finished.st_ino,
        "owner_uid": finished.st_uid,
        "owner_gid": finished.st_gid,
        "mode": stat.S_IMODE(finished.st_mode),
        "nlink": finished.st_nlink,
        "size": finished.st_size,
        "mtime_ns": finished.st_mtime_ns,
        "ctime_ns": finished.st_ctime_ns,
    }


def _load_frozen_c1() -> tuple[ModuleType, dict[str, object]]:
    raw, generation = _stable_source_bytes(
        FROZEN_REALIZER_PATH,
        expected_sha256=FROZEN_REALIZER_SHA256,
    )
    name = (
        "tools._cure_lite_v24_actual_unit_realization_preaccess_"
        "compat_c1_verified_for_c5"
    )
    module = ModuleType(name)
    module.__file__ = str(FROZEN_REALIZER_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                raw,
                str(FROZEN_REALIZER_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, generation


def _typed_projection(
    value: object,
    seen: frozenset[int] = frozenset(),
) -> object:
    """Return a recursively type-exact, deterministic comparison projection."""

    value_type = type(value)
    if value is None or value_type in {
        bool,
        int,
        float,
        complex,
        str,
        bytes,
    }:
        return value_type, value
    if isinstance(value, FunctionType) or isinstance(value, ModuleType):
        return value_type, value
    marker = id(value)
    if marker in seen:
        return value_type, "cycle", marker
    nested_seen = seen | {marker}
    if value_type is dict:
        rows = [
            (
                _typed_projection(key, nested_seen),
                _typed_projection(item, nested_seen),
            )
            for key, item in value.items()
        ]
        return value_type, tuple(sorted(rows, key=repr))
    if value_type in {tuple, list}:
        return value_type, tuple(
            _typed_projection(item, nested_seen) for item in value
        )
    if value_type in {set, frozenset}:
        rows = [
            _typed_projection(item, nested_seen) for item in value
        ]
        return value_type, tuple(sorted(rows, key=repr))
    return value_type, value


def _type_exact_equal(left: object, right: object) -> bool:
    return _typed_projection(left) == _typed_projection(right)


def _build_callable_state_guard(
    *owners: tuple[str, ModuleType],
    require_active_names: bool,
    previous_guard=None,
    critical_globals: Mapping[str, object] | None = None,
):
    """Capture callable internals without trusting later module globals.

    The first (pre-rebind) guard preserves every loaded Python function even
    when R5 intentionally replaces its public module name.  The second guard
    additionally freezes the active post-rebind name map.  This is an
    entry-time integrity boundary for a fresh isolated CLI process; it is not
    a claim of resistance to hostile in-process reflection or concurrent
    mutation.
    """

    typed = _typed_projection

    def cell_value(cell):
        try:
            return cell.cell_contents
        except ValueError:
            return _EMPTY_CLOSURE_CELL

    def capture(function: FunctionType):
        closure = function.__closure__
        return (
            function.__code__,
            function.__globals__,
            function.__defaults__,
            typed(function.__defaults__),
            function.__kwdefaults__,
            typed(function.__kwdefaults__),
            closure,
            None
            if closure is None
            else tuple(
                (cell, typed(cell_value(cell))) for cell in closure
            ),
            function.__annotations__,
            typed(function.__annotations__),
            function.__dict__,
            typed(function.__dict__),
        )

    def state_same(function: FunctionType, expected) -> bool:
        (
            code,
            global_namespace,
            defaults,
            defaults_typed,
            kwdefaults,
            kwdefaults_typed,
            closure,
            closure_cells,
            annotations,
            annotations_typed,
            function_dict,
            function_dict_typed,
        ) = expected
        if (
            function.__code__ is not code
            or function.__globals__ is not global_namespace
            or function.__defaults__ is not defaults
            or typed(function.__defaults__) != defaults_typed
            or function.__kwdefaults__ is not kwdefaults
            or typed(function.__kwdefaults__) != kwdefaults_typed
            or function.__closure__ is not closure
            or function.__annotations__ is not annotations
            or typed(function.__annotations__) != annotations_typed
            or function.__dict__ is not function_dict
            or typed(function.__dict__) != function_dict_typed
        ):
            return False
        if closure is None:
            return closure_cells is None
        assert closure_cells is not None
        return len(closure) == len(closure_cells) and all(
            cell is expected_cell
            and typed(cell_value(cell)) == expected_value
            for cell, (expected_cell, expected_value) in zip(
                closure,
                closure_cells,
            )
        )

    active_import_identities = frozenset(
        {
            "Path",
            "Mapping",
            "datetime",
            "timezone",
            "timedelta",
            "deepcopy",
            "ModuleType",
            "os",
            "stat",
            "json",
            "hashlib",
            "re",
            "pwd",
            "subprocess",
        }
    )
    owner_modules = tuple(module for _label, module in owners)
    callable_maps = tuple(
        (
            label,
            module,
            tuple(
                (
                    name,
                    value,
                    capture(value) if isinstance(value, FunctionType) else None,
                )
                for name, value in vars(module).items()
                if callable(value)
                or (
                    name in active_import_identities
                    and isinstance(value, ModuleType)
                )
            ),
        )
        for label, module in owners
    )
    critical_map = tuple(
        (
            name,
            value,
            capture(value) if isinstance(value, FunctionType) else None,
        )
        for name, value in (critical_globals or {}).items()
    )
    coverage = tuple(
        (
            label,
            sum(
                isinstance(value, FunctionType)
                for value in vars(module).values()
            ),
            len(entries),
        )
        for (label, module), (_x, _y, entries) in zip(
            owners,
            callable_maps,
        )
    )

    def require() -> None:
        if previous_guard is not None:
            previous_guard()
        if tuple(module for _label, module in owners) != owner_modules:
            raise PermissionError("c5 delegated module identity changed")
        for _label, module, entries in callable_maps:
            for name, expected, state in entries:
                if require_active_names and getattr(module, name, None) is not expected:
                    raise PermissionError(
                        "c5 transitive callable identity changed",
                    )
                if state is not None and not state_same(expected, state):
                    raise PermissionError(
                        "c5 transitive callable state changed",
                    )
        for name, expected, state in critical_map:
            if globals().get(name) is not expected:
                raise PermissionError("c5 critical R5 callable identity changed")
            if state is not None and not state_same(expected, state):
                raise PermissionError("c5 critical R5 callable state changed")

    return require, coverage


_EMPTY_CLOSURE_CELL = object()
_compat_c1, _C1_LOAD_GENERATION = _load_frozen_c1()
_require_pristine_callable_states, _PRISTINE_CALLABLE_COVERAGE = (
    _build_callable_state_guard(
        ("c1", _compat_c1),
        ("base", _compat_c1.legacy),
        require_active_names=False,
    )
)
BASE_REALIZER_PATH = Path(_compat_c1.FROZEN_REALIZER_PATH)
BASE_REALIZER_SHA256 = _compat_c1.FROZEN_REALIZER_SHA256
BASE_REALIZER_GENERATION = dict(
    _compat_c1._FROZEN_REALIZER_LOAD_BINDING,
)


def _build_base_realizer_guard(
    path: Path,
    digest: str,
    generation: Mapping[str, object],
):
    expected_path = (
        REPOSITORY / "tools/cure_lite_v24_actual_unit_realization.py"
    ).resolve()
    expected_digest = (
        "0d66bc4007366588ed1393b21092cc57d58e0f7fca084f7266a00e6818703fd9"
    )
    expected_generation = dict(generation)
    if path != expected_path or digest != expected_digest:
        raise PermissionError("c5 imported base realizer identity changed")

    def require() -> dict[str, object]:
        if (
            BASE_REALIZER_PATH != expected_path
            or BASE_REALIZER_SHA256 != expected_digest
            or not _type_exact_equal(
                BASE_REALIZER_GENERATION,
                expected_generation,
            )
        ):
            raise PermissionError("c5 base realizer public identity changed")
        _, observed = _stable_source_bytes(
            expected_path,
            expected_sha256=expected_digest,
        )
        if not _type_exact_equal(observed, expected_generation):
            raise PermissionError(
                "c5 base realizer source generation changed",
            )
        return observed

    return require


_require_base_realizer_generation = _build_base_realizer_guard(
    BASE_REALIZER_PATH,
    BASE_REALIZER_SHA256,
    BASE_REALIZER_GENERATION,
)
del _build_base_realizer_guard
_SELF_LOAD_BYTES, _SELF_LOAD_GENERATION = _stable_source_bytes(
    COMPAT_REALIZER_PATH,
)
_SUPERVISOR_LOAD_BYTES, _SUPERVISOR_LOAD_GENERATION = (
    _stable_source_bytes(
        COMPAT_SUPERVISOR_PATH,
    )
)
_BRIDGE_LOAD_BYTES, _BRIDGE_LOAD_GENERATION = _stable_source_bytes(
    COMPAT_BRIDGE_SOURCE_PATH,
)
_TEMPLATE_LOAD_BYTES, _TEMPLATE_LOAD_GENERATION = _stable_source_bytes(
    COMPAT_TEMPLATE_PATH,
    expected_sha256=COMPAT_TEMPLATE_SHA256,
)

_C1_BUILD_CLOSURE = _compat_c1._build_compatibility_closure
_C1_VALIDATE_CLOSURE = _compat_c1._validate_compatibility_closure
_C1_FIXED_PATHS = _compat_c1._fixed_paths
_C1_VERIFY_IDENTITY = _compat_c1.verify_compatibility_identity
_C1_CREATE_AUTHORIZATION_RAW = _compat_c1.create_authorization
_C1_VALIDATE_AUTHORIZATION = _compat_c1.validate_authorization
_C1_REALIZE_ACTUAL_UNIT_RAW = _compat_c1.realize_actual_unit
_C1_VALIDATE_ARCHIVAL_CHAIN = _compat_c1.validate_archival_realization_chain
_C1_COMPAT_EXPECTED_STATIC_SHADOW = (
    _compat_c1._compat_expected_static_shadow
)
_C1_RUNTIME_DYNAMIC_SHADOW_FIELDS = frozenset(
    _compat_c1._RUNTIME_DYNAMIC_SHADOW_FIELDS
)


def _build_c1_alias_guard(**captured):
    expected = tuple(captured.items())

    def require() -> None:
        if any(globals().get(name) is not value for name, value in expected):
            raise PermissionError("c5 captured C1 callable identity changed")

    return require


_require_c1_alias_identities = _build_c1_alias_guard(
    _C1_BUILD_CLOSURE=_C1_BUILD_CLOSURE,
    _C1_VALIDATE_CLOSURE=_C1_VALIDATE_CLOSURE,
    _C1_FIXED_PATHS=_C1_FIXED_PATHS,
    _C1_VERIFY_IDENTITY=_C1_VERIFY_IDENTITY,
    _C1_VALIDATE_AUTHORIZATION=_C1_VALIDATE_AUTHORIZATION,
    _C1_VALIDATE_ARCHIVAL_CHAIN=_C1_VALIDATE_ARCHIVAL_CHAIN,
    _C1_COMPAT_EXPECTED_STATIC_SHADOW=(
        _C1_COMPAT_EXPECTED_STATIC_SHADOW
    ),
)
del _build_c1_alias_guard
_EXPECTED_C1_RUNTIME_DYNAMIC_SHADOW_FIELDS = frozenset(
    {"ActiveState", "SubState"}
)
_R5_RUNTIME_DYNAMIC_SHADOW_FIELDS = (
    _EXPECTED_C1_RUNTIME_DYNAMIC_SHADOW_FIELDS | {"InvocationID"}
)


def _build_exact_global_lane(
    module: ModuleType,
):
    legacy_module = module.legacy
    base_scalars = {
        name: getattr(legacy_module, name)
        for name in (
            "CANDIDATE",
            "STAGE_ID",
            "ATTEMPT_ID",
            "AUTHORIZATION_SCHEMA",
            "RECEIPT_SCHEMA",
            "TERMINAL_SCHEMA",
            "SUPERVISOR_SPEC_SCHEMA",
            "PYTHON_PATH",
            "SYSTEMD_PATH",
            "SYSTEMD_ANALYZE",
            "SYSTEMCTL",
        )
    }
    base_exec_modes = tuple(sorted(legacy_module._EXEC_MODES.items()))
    base_placeholders = frozenset(legacy_module._PLACEHOLDERS)
    base_file_binding_keys = frozenset(legacy_module._FILE_BINDING_KEYS)
    base_sha_pattern = legacy_module._SHA
    base_boot_id_pattern = legacy_module._BOOT_ID
    base_auth_keys = frozenset(legacy_module._AUTH_KEYS)
    c1_generation_fields = tuple(module._GENERATION_FIELDS)
    c1_source_binding_keys = frozenset(module._SOURCE_BINDING_KEYS)
    c1_parent_binding_keys = frozenset(module._PARENT_BINDING_KEYS)
    c1_closure_keys = frozenset(module._CLOSURE_KEYS)
    c1_receipt_keys = frozenset(module._RECEIPT_BODY_KEYS)
    c1_fragment_identity_keys = frozenset(module._FRAGMENT_IDENTITY_KEYS)
    c1_bridge_root_volatile_fields = frozenset(
        module._BRIDGE_ROOT_VOLATILE_PARENT_FIELDS,
    )
    c1_mutation_items = tuple(sorted(module._MUTATION_AUTHORITY.items()))
    c5_mutation = dict(c1_mutation_items)
    c5_mutation.update(
        {
            "c5_unit_realization_authorized": True,
            "protected_original_r2_unit_mutation": False,
            "protected_c1_unit_mutation": False,
            "protected_c2_unit_mutation": False,
            "protected_c3_unit_mutation": False,
            "protected_c4_unit_mutation": False,
        },
    )
    c5_mutation_items = tuple(sorted(c5_mutation.items()))
    base_shadow = tuple(legacy_module._SHADOW_PROPERTIES)
    c5_shadow = tuple(
        name for name in base_shadow if name != "InvocationID"
    ) + ("InvocationID",)
    c5_actions = tuple(_REALIZATION_ACTIONS)
    own_contract = (
        COMPATIBILITY_CLOSURE_SCHEMA,
        COMPATIBILITY_CLOSURE_KEY,
        PROTECTED_UNITS_KEY,
        BRIDGE_AUTHORIZATION_SCHEMA,
        B5_INSTRUCTION_ID,
        B5_AUTHORIZATION_BASIS,
        tuple(sorted(BRIDGE_MUTATION_AUTHORITY.items())),
        tuple(sorted(BRIDGE_SCIENTIFIC_AUTHORITY.items())),
        frozenset(_BRIDGE_STATE_KEYS),
        tuple(_REALIZATION_ACTIONS),
        tuple(sorted(_RECEIPT_STATIC_STATE.items())),
        RUNTIME_PHASE_PREACTIVATION,
        RUNTIME_PHASE_COMMIT,
        RUNTIME_PHASE_CLAIM,
        RUNTIME_PHASE_VERIFY,
        RUNTIME_PHASE_RUN_ONCE,
        RUNTIME_PHASE_FINALIZE_SUCCESS,
        RUNTIME_PHASE_FINALIZE_FAILURE,
        frozenset(RUNTIME_PHASES),
        _RUNTIME_PHASE_CONTEXT,
        tuple(_SOURCE_FIELDS),
        frozenset(_C1_RUNTIME_DYNAMIC_SHADOW_FIELDS),
        frozenset(_EXPECTED_C1_RUNTIME_DYNAMIC_SHADOW_FIELDS),
        frozenset(_R5_RUNTIME_DYNAMIC_SHADOW_FIELDS),
        tuple(sorted(_C1_LOAD_GENERATION.items())),
        tuple(sorted(_SELF_LOAD_GENERATION.items())),
        tuple(sorted(_SUPERVISOR_LOAD_GENERATION.items())),
        tuple(sorted(_BRIDGE_LOAD_GENERATION.items())),
        tuple(sorted(_TEMPLATE_LOAD_GENERATION.items())),
    )

    def require_own_contract() -> None:
        observed = (
            COMPATIBILITY_CLOSURE_SCHEMA,
            COMPATIBILITY_CLOSURE_KEY,
            PROTECTED_UNITS_KEY,
            BRIDGE_AUTHORIZATION_SCHEMA,
            B5_INSTRUCTION_ID,
            B5_AUTHORIZATION_BASIS,
            tuple(sorted(BRIDGE_MUTATION_AUTHORITY.items())),
            tuple(sorted(BRIDGE_SCIENTIFIC_AUTHORITY.items())),
            frozenset(_BRIDGE_STATE_KEYS),
            tuple(_REALIZATION_ACTIONS),
            tuple(sorted(_RECEIPT_STATIC_STATE.items())),
            RUNTIME_PHASE_PREACTIVATION,
            RUNTIME_PHASE_COMMIT,
            RUNTIME_PHASE_CLAIM,
            RUNTIME_PHASE_VERIFY,
            RUNTIME_PHASE_RUN_ONCE,
            RUNTIME_PHASE_FINALIZE_SUCCESS,
            RUNTIME_PHASE_FINALIZE_FAILURE,
            frozenset(RUNTIME_PHASES),
            _RUNTIME_PHASE_CONTEXT,
            tuple(_SOURCE_FIELDS),
            frozenset(_C1_RUNTIME_DYNAMIC_SHADOW_FIELDS),
            frozenset(_EXPECTED_C1_RUNTIME_DYNAMIC_SHADOW_FIELDS),
            frozenset(_R5_RUNTIME_DYNAMIC_SHADOW_FIELDS),
            tuple(sorted(_C1_LOAD_GENERATION.items())),
            tuple(sorted(_SELF_LOAD_GENERATION.items())),
            tuple(sorted(_SUPERVISOR_LOAD_GENERATION.items())),
            tuple(sorted(_BRIDGE_LOAD_GENERATION.items())),
            tuple(sorted(_TEMPLATE_LOAD_GENERATION.items())),
        )
        if not _type_exact_equal(observed, own_contract):
            raise PermissionError("c5 realizer-owned contract changed")

    def apply_exact_globals(
        *,
        instruction_id: str,
        authorization_basis: str,
    ) -> None:
        require_own_contract()
        for name, value in base_scalars.items():
            setattr(legacy_module, name, value)
        legacy_module.ACTUAL_UNIT = COMPAT_UNIT
        legacy_module.__file__ = str(COMPAT_REALIZER_PATH)
        legacy_module.INSTRUCTION_ID = instruction_id
        legacy_module.AUTHORIZATION_BASIS = authorization_basis
        legacy_module._ACTIONS = list(c5_actions)
        legacy_module._SHADOW_PROPERTIES = c5_shadow
        legacy_module._EXEC_MODES = dict(base_exec_modes)
        legacy_module._PLACEHOLDERS = set(base_placeholders)
        legacy_module._FILE_BINDING_KEYS = set(base_file_binding_keys)
        legacy_module._SHA = base_sha_pattern
        legacy_module._BOOT_ID = base_boot_id_pattern
        legacy_module._AUTH_KEYS = set(base_auth_keys)
        module._GENERATION_FIELDS = c1_generation_fields
        module._SOURCE_BINDING_KEYS = set(c1_source_binding_keys)
        module._PARENT_BINDING_KEYS = set(c1_parent_binding_keys)
        module._CLOSURE_KEYS = set(c1_closure_keys)
        module._RECEIPT_BODY_KEYS = set(c1_receipt_keys)
        module._FRAGMENT_IDENTITY_KEYS = set(c1_fragment_identity_keys)
        module._BRIDGE_ROOT_VOLATILE_PARENT_FIELDS = set(
            c1_bridge_root_volatile_fields,
        )
        module._MUTATION_AUTHORITY = dict(c5_mutation_items)
        module._RUNTIME_DYNAMIC_SHADOW_FIELDS = set(
            _R5_RUNTIME_DYNAMIC_SHADOW_FIELDS,
        )

    def require_exact_globals() -> None:
        require_own_contract()
        expected_scalars = {
            **base_scalars,
            "ACTUAL_UNIT": COMPAT_UNIT,
            "INSTRUCTION_ID": B5_INSTRUCTION_ID,
            "AUTHORIZATION_BASIS": B5_AUTHORIZATION_BASIS,
            "__file__": str(COMPAT_REALIZER_PATH),
        }
        if (
            any(
                getattr(legacy_module, name, None) != value
                for name, value in expected_scalars.items()
            )
            or tuple(legacy_module._ACTIONS) != c5_actions
            or tuple(legacy_module._SHADOW_PROPERTIES) != c5_shadow
            or tuple(sorted(legacy_module._EXEC_MODES.items()))
            != base_exec_modes
            or frozenset(legacy_module._PLACEHOLDERS) != base_placeholders
            or frozenset(legacy_module._FILE_BINDING_KEYS)
            != base_file_binding_keys
            or legacy_module._SHA is not base_sha_pattern
            or legacy_module._BOOT_ID is not base_boot_id_pattern
            or frozenset(legacy_module._AUTH_KEYS) != base_auth_keys
            or tuple(module._GENERATION_FIELDS) != c1_generation_fields
            or frozenset(module._SOURCE_BINDING_KEYS)
            != c1_source_binding_keys
            or frozenset(module._PARENT_BINDING_KEYS)
            != c1_parent_binding_keys
            or frozenset(module._CLOSURE_KEYS) != c1_closure_keys
            or frozenset(module._RECEIPT_BODY_KEYS) != c1_receipt_keys
            or frozenset(module._FRAGMENT_IDENTITY_KEYS)
            != c1_fragment_identity_keys
            or frozenset(module._BRIDGE_ROOT_VOLATILE_PARENT_FIELDS)
            != c1_bridge_root_volatile_fields
            or not _type_exact_equal(
                dict(module._MUTATION_AUTHORITY),
                dict(c5_mutation_items),
            )
            or frozenset(module._RUNTIME_DYNAMIC_SHADOW_FIELDS)
            != _R5_RUNTIME_DYNAMIC_SHADOW_FIELDS
        ):
            raise PermissionError("c5 delegated exact-global contract changed")

    def mutation_authority() -> dict[str, bool]:
        return dict(c5_mutation_items)

    return apply_exact_globals, require_exact_globals, mutation_authority


(
    _apply_exact_frozen_globals,
    _require_exact_frozen_globals,
    _exact_mutation_authority,
) = _build_exact_global_lane(_compat_c1)
del _build_exact_global_lane


def _build_canonical_c5_identity_guard():
    expected = (
        REPOSITORY,
        COMPAT_REALIZER_PATH,
        FROZEN_REALIZER_PATH,
        FROZEN_REALIZER_SHA256,
        EVIDENCE_ROOT,
        COMPAT_UNIT,
        PROTECTED_ORIGINAL_UNIT,
        PROTECTED_C1_UNIT,
        PROTECTED_C2_UNIT,
        PROTECTED_C3_UNIT,
        PROTECTED_C4_UNIT,
        COMPAT_UNIT_DIRECTORY,
        PROTECTED_ORIGINAL_FRAGMENT_PATH,
        PROTECTED_C1_FRAGMENT_PATH,
        PROTECTED_C2_FRAGMENT_PATH,
        PROTECTED_C3_FRAGMENT_PATH,
        PROTECTED_C4_FRAGMENT_PATH,
        COMPAT_TEMPLATE_PATH,
        COMPAT_TEMPLATE_SHA256,
        COMPAT_SUPERVISOR_PATH,
        COMPAT_SUPERVISOR_SHA256,
        COMPAT_BRIDGE_SOURCE_PATH,
        COMPAT_BRIDGE_SOURCE_SHA256,
        COMPAT_BRIDGE_AUTHORIZATION_PATH,
        COMPAT_RUNTIME_SPEC_PATH,
        COMPAT_AUTHORIZATION_PATH,
        COMPAT_RECEIPT_PATH,
        COMPAT_TERMINAL_PATH,
    )

    def require() -> None:
        observed = (
            REPOSITORY,
            COMPAT_REALIZER_PATH,
            FROZEN_REALIZER_PATH,
            FROZEN_REALIZER_SHA256,
            EVIDENCE_ROOT,
            COMPAT_UNIT,
            PROTECTED_ORIGINAL_UNIT,
            PROTECTED_C1_UNIT,
            PROTECTED_C2_UNIT,
            PROTECTED_C3_UNIT,
            PROTECTED_C4_UNIT,
            COMPAT_UNIT_DIRECTORY,
            PROTECTED_ORIGINAL_FRAGMENT_PATH,
            PROTECTED_C1_FRAGMENT_PATH,
            PROTECTED_C2_FRAGMENT_PATH,
            PROTECTED_C3_FRAGMENT_PATH,
            PROTECTED_C4_FRAGMENT_PATH,
            COMPAT_TEMPLATE_PATH,
            COMPAT_TEMPLATE_SHA256,
            COMPAT_SUPERVISOR_PATH,
            COMPAT_SUPERVISOR_SHA256,
            COMPAT_BRIDGE_SOURCE_PATH,
            COMPAT_BRIDGE_SOURCE_SHA256,
            COMPAT_BRIDGE_AUTHORIZATION_PATH,
            COMPAT_RUNTIME_SPEC_PATH,
            COMPAT_AUTHORIZATION_PATH,
            COMPAT_RECEIPT_PATH,
            COMPAT_TERMINAL_PATH,
        )
        if not _type_exact_equal(observed, expected):
            raise PermissionError("c5 canonical path/unit identity changed")

    return require


_require_canonical_c5_identity = _build_canonical_c5_identity_guard()
del _build_canonical_c5_identity_guard


def _require_runtime_dynamic_shadow_contract() -> None:
    observed = frozenset(_compat_c1._RUNTIME_DYNAMIC_SHADOW_FIELDS)
    if (
        _C1_RUNTIME_DYNAMIC_SHADOW_FIELDS
        != _EXPECTED_C1_RUNTIME_DYNAMIC_SHADOW_FIELDS
        or _R5_RUNTIME_DYNAMIC_SHADOW_FIELDS
        != frozenset({"ActiveState", "SubState", "InvocationID"})
        or observed
        not in {
            _EXPECTED_C1_RUNTIME_DYNAMIC_SHADOW_FIELDS,
            _R5_RUNTIME_DYNAMIC_SHADOW_FIELDS,
        }
    ):
        raise PermissionError(
            "c5 runtime dynamic-shadow field contract changed",
        )


def _require_frozen_c5_sources() -> None:
    for label, digest in (
        ("bridge source", COMPAT_BRIDGE_SOURCE_SHA256),
        ("supervisor", COMPAT_SUPERVISOR_SHA256),
        ("template", COMPAT_TEMPLATE_SHA256),
    ):
        if (
            digest == "__TO_BE_FROZEN__"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise PermissionError(f"c5 {label} generation is not frozen")


def _require_source_generations() -> None:
    _require_frozen_c5_sources()
    _require_base_realizer_generation()
    _, predecessor = _stable_source_bytes(
        FROZEN_REALIZER_PATH,
        expected_sha256=FROZEN_REALIZER_SHA256,
    )
    _, own = _stable_source_bytes(COMPAT_REALIZER_PATH)
    _, supervisor = _stable_source_bytes(
        COMPAT_SUPERVISOR_PATH,
        expected_sha256=COMPAT_SUPERVISOR_SHA256,
    )
    _, bridge = _stable_source_bytes(
        COMPAT_BRIDGE_SOURCE_PATH,
        expected_sha256=COMPAT_BRIDGE_SOURCE_SHA256,
    )
    _, template = _stable_source_bytes(
        COMPAT_TEMPLATE_PATH,
        expected_sha256=COMPAT_TEMPLATE_SHA256,
    )
    if (
        not _type_exact_equal(predecessor, _C1_LOAD_GENERATION)
        or not _type_exact_equal(own, _SELF_LOAD_GENERATION)
        or not _type_exact_equal(
            supervisor,
            _SUPERVISOR_LOAD_GENERATION,
        )
        or not _type_exact_equal(bridge, _BRIDGE_LOAD_GENERATION)
        or not _type_exact_equal(template, _TEMPLATE_LOAD_GENERATION)
    ):
        raise PermissionError(
            "c5 realizer/source closure generation changed",
        )


def _load_bridge_policy_from_verified_bytes(raw: bytes) -> ModuleType:
    name = (
        "tools._cure_lite_v24_preaccess_schema_compatibility_"
        "c5_for_unit_realization"
    )
    module = ModuleType(name)
    module.__file__ = str(COMPAT_BRIDGE_SOURCE_PATH)
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(COMPAT_BRIDGE_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    return module


def _require_b5_authority_contract() -> tuple[str, str]:
    """Read the frozen B5 generation and return its exact user authority."""

    _require_frozen_c5_sources()
    raw, generation = _stable_source_bytes(
        COMPAT_BRIDGE_SOURCE_PATH,
        expected_sha256=COMPAT_BRIDGE_SOURCE_SHA256,
    )
    if not _type_exact_equal(generation, _BRIDGE_LOAD_GENERATION):
        raise PermissionError("c5 bridge authority source generation changed")
    policy = _load_bridge_policy_from_verified_bytes(raw)
    if (
        getattr(policy, "INSTRUCTION_ID", None) != B5_INSTRUCTION_ID
        or getattr(policy, "AUTHORIZATION_BASIS", None)
        != B5_AUTHORIZATION_BASIS
    ):
        raise PermissionError("c5 B5 user-authority contract changed")
    return policy.INSTRUCTION_ID, policy.AUTHORIZATION_BASIS


def _require_static_authorized_state(
    state: object,
    *,
    unit_name: str,
    fragment_path: str | None = None,
) -> None:
    if (
        not isinstance(state, Mapping)
        or set(state) != _BRIDGE_STATE_KEYS
        or state.get("Id") != unit_name
        or state.get("LoadState") != "loaded"
        or state.get("ActiveState") != "inactive"
        or state.get("SubState") != "dead"
        or state.get("UnitFileState") != "static"
        or state.get("NRestarts") != "0"
        or state.get("InvocationID") != ""
        or not isinstance(state.get("FragmentPath"), str)
        or (
            fragment_path is not None
            and state.get("FragmentPath") != fragment_path
        )
    ):
        raise PermissionError(
            f"B5 authorized unit is not exact static/inert: {unit_name}",
        )


def _require_missing_authorized_state(
    state: object,
    *,
    unit_name: str,
) -> None:
    if (
        not isinstance(state, Mapping)
        or set(state) != _BRIDGE_STATE_KEYS
        or state.get("Id") != unit_name
        or state.get("LoadState") != "not-found"
        or state.get("ActiveState") != "inactive"
        or state.get("SubState") != "dead"
        or state.get("UnitFileState") != ""
        or state.get("NRestarts") != "0"
        or state.get("FragmentPath") != ""
        or state.get("InvocationID") != ""
    ):
        raise PermissionError(
            f"B5 authorized unit is not exact not-found/inert: {unit_name}",
        )


def _validate_b5_authorized_unit_states(
    authorization: Mapping[str, object],
) -> None:
    protected = authorization.get("protected_unit_states")
    if not isinstance(protected, Mapping) or set(protected) != {
        "old",
        "c1",
        "c2",
        "c3",
        "c4",
    }:
        raise PermissionError("B5 protected unit-state closure changed")
    _require_static_authorized_state(
        protected["old"],
        unit_name=PROTECTED_ORIGINAL_UNIT,
        fragment_path=str(PROTECTED_ORIGINAL_FRAGMENT_PATH),
    )
    _require_static_authorized_state(
        protected["c1"],
        unit_name=PROTECTED_C1_UNIT,
        fragment_path=str(PROTECTED_C1_FRAGMENT_PATH),
    )
    _require_missing_authorized_state(
        protected["c2"],
        unit_name=PROTECTED_C2_UNIT,
    )
    _require_static_authorized_state(
        protected["c3"],
        unit_name=PROTECTED_C3_UNIT,
        fragment_path=str(PROTECTED_C3_FRAGMENT_PATH),
    )
    _require_static_authorized_state(
        protected["c4"],
        unit_name=PROTECTED_C4_UNIT,
        fragment_path=str(PROTECTED_C4_FRAGMENT_PATH),
    )
    _require_missing_authorized_state(
        authorization.get("preauthorization_target_unit_state"),
        unit_name=COMPAT_UNIT,
    )


def _validate_c5_bridge_authorization(
    *,
    require_fresh: bool = True,
    require_future_absence: bool,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    _require_frozen_c5_sources()
    raw, source = _compat_c1._stable_regular_read(
        COMPAT_BRIDGE_SOURCE_PATH,
        expected_sha256=COMPAT_BRIDGE_SOURCE_SHA256,
    )
    policy = _load_bridge_policy_from_verified_bytes(raw)
    if (
        Path(policy.COMPAT_AUTHORIZATION_PATH)
        != COMPAT_BRIDGE_AUTHORIZATION_PATH
        or Path(policy.COMPAT_UNIT_REALIZER_SOURCE_PATH)
        != COMPAT_REALIZER_PATH
        or policy.COMPAT_UNIT_NAME != COMPAT_UNIT
        or policy.RUNTIME_COMPATIBILITY_ID != "c5"
        or frozenset(policy.RUNTIME_PHASES) != RUNTIME_PHASES
        or policy.RUNTIME_PHASE_PREACTIVATION
        != RUNTIME_PHASE_PREACTIVATION
        or policy.RUNTIME_PHASE_COMMIT != RUNTIME_PHASE_COMMIT
        or policy.RUNTIME_PHASE_CLAIM != RUNTIME_PHASE_CLAIM
        or policy.RUNTIME_PHASE_VERIFY != RUNTIME_PHASE_VERIFY
        or policy.RUNTIME_PHASE_RUN_ONCE != RUNTIME_PHASE_RUN_ONCE
        or policy.RUNTIME_PHASE_FINALIZE_SUCCESS
        != RUNTIME_PHASE_FINALIZE_SUCCESS
        or policy.RUNTIME_PHASE_FINALIZE_FAILURE
        != RUNTIME_PHASE_FINALIZE_FAILURE
        or policy.INSTRUCTION_ID != B5_INSTRUCTION_ID
        or policy.AUTHORIZATION_BASIS != B5_AUTHORIZATION_BASIS
        or not callable(policy.validate_compat_authorization)
    ):
        raise PermissionError("c5 bridge compatibility interface changed")
    runtime_phase = _RUNTIME_PHASE_CONTEXT.get()
    allow_runtime_activation = (
        runtime_phase != RUNTIME_PHASE_PREACTIVATION
    )
    _resolve_runtime_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    authorization, root = policy.validate_compat_authorization(
        COMPAT_BRIDGE_AUTHORIZATION_PATH,
        require_fresh=require_fresh,
        require_future_absence=require_future_absence,
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    _, source_after = _compat_c1._stable_regular_read(
        COMPAT_BRIDGE_SOURCE_PATH,
        expected_sha256=COMPAT_BRIDGE_SOURCE_SHA256,
    )
    mutation = authorization.get("mutation_authority")
    scientific = authorization.get("scientific_authority")
    file_sha256 = root.get("file_sha256")
    _validate_b5_authorized_unit_states(authorization)
    if (
        not _compat_c1._same_generation(source_after, source)
        or authorization.get("schema_version")
        != BRIDGE_AUTHORIZATION_SCHEMA
        or authorization.get("scientific_attempt_ordinal") != 2
        or authorization.get("runtime_compatibility_id") != "c5"
        or authorization.get("instruction_id") != B5_INSTRUCTION_ID
        or authorization.get("authorization_basis")
        != B5_AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != os.getuid()
        or not _type_exact_equal(mutation, BRIDGE_MUTATION_AUTHORITY)
        or not _type_exact_equal(
            scientific,
            BRIDGE_SCIENTIFIC_AUTHORITY,
        )
        or root.get("path") != str(COMPAT_BRIDGE_AUTHORIZATION_PATH)
        or root.get("fingerprint")
        != authorization.get("authorization_fingerprint")
        or not isinstance(file_sha256, str)
        or len(file_sha256) != 64
        or any(character not in "0123456789abcdef"
               for character in file_sha256)
        or root.get("mode") != 0o444
        or root.get("owner_uid") != os.getuid()
        or not isinstance(root.get("device"), int)
        or root.get("device", -1) < 0
        or not isinstance(root.get("inode"), int)
        or root.get("inode", 0) <= 0
        or not isinstance(root.get("size"), int)
        or root.get("size", 0) <= 0
    ):
        raise PermissionError(
            "c5 bridge does not authorize the narrow unit lane",
        )
    return dict(authorization), dict(root), source


def _observe_unit(
    unit_name: str,
    fragment_path: Path,
) -> dict[str, object]:
    if (
        fragment_path.parent != COMPAT_UNIT_DIRECTORY
        or fragment_path.name != unit_name
        or unit_name == COMPAT_UNIT
    ):
        raise PermissionError("protected unit path changed")
    parent = _compat_c1._parent_binding(fragment_path.parent)
    try:
        _, binding = _compat_c1._stable_regular_read(fragment_path)
    except FileNotFoundError:
        if _compat_c1._parent_binding(fragment_path.parent) != parent:
            raise PermissionError(
                "protected unit parent changed during absence proof",
            )
        binding = None
    return {
        "unit_name": unit_name,
        "fragment_path": str(fragment_path),
        "fragment_exists": binding is not None,
        "fragment_generation": binding,
        "parent_generation": parent,
        "mutation_authorized": False,
    }


def _protected_units() -> dict[str, object]:
    _require_disjoint_c5_namespace()
    return {
        "original_r2": _observe_unit(
            PROTECTED_ORIGINAL_UNIT,
            PROTECTED_ORIGINAL_FRAGMENT_PATH,
        ),
        "compat_c1": _observe_unit(
            PROTECTED_C1_UNIT,
            PROTECTED_C1_FRAGMENT_PATH,
        ),
        "compat_c2": _observe_unit(
            PROTECTED_C2_UNIT,
            PROTECTED_C2_FRAGMENT_PATH,
        ),
        "compat_c3": _observe_unit(
            PROTECTED_C3_UNIT,
            PROTECTED_C3_FRAGMENT_PATH,
        ),
        "compat_c4": _observe_unit(
            PROTECTED_C4_UNIT,
            PROTECTED_C4_FRAGMENT_PATH,
        ),
    }


def _fixed_paths() -> dict[str, str]:
    _require_disjoint_c5_namespace()
    result = dict(_C1_FIXED_PATHS())
    result.update(
        {
            "protected_original_fragment": str(
                PROTECTED_ORIGINAL_FRAGMENT_PATH,
            ),
            "protected_c1_fragment": str(PROTECTED_C1_FRAGMENT_PATH),
            "protected_c2_fragment": str(PROTECTED_C2_FRAGMENT_PATH),
            "protected_c3_fragment": str(PROTECTED_C3_FRAGMENT_PATH),
            "protected_c4_fragment": str(PROTECTED_C4_FRAGMENT_PATH),
        },
    )
    return result


def _require_disjoint_c5_namespace() -> None:
    units = (
        PROTECTED_ORIGINAL_UNIT,
        PROTECTED_C1_UNIT,
        PROTECTED_C2_UNIT,
        PROTECTED_C3_UNIT,
        PROTECTED_C4_UNIT,
        COMPAT_UNIT,
    )
    fragments = (
        PROTECTED_ORIGINAL_FRAGMENT_PATH,
        PROTECTED_C1_FRAGMENT_PATH,
        PROTECTED_C2_FRAGMENT_PATH,
        PROTECTED_C3_FRAGMENT_PATH,
        PROTECTED_C4_FRAGMENT_PATH,
        COMPAT_UNIT_DIRECTORY / COMPAT_UNIT,
    )
    evidence = (
        COMPAT_BRIDGE_AUTHORIZATION_PATH,
        COMPAT_RUNTIME_SPEC_PATH,
        COMPAT_AUTHORIZATION_PATH,
        COMPAT_RECEIPT_PATH,
        COMPAT_TERMINAL_PATH,
    )
    if (
        len(set(units)) != len(units)
        or len(set(fragments)) != len(fragments)
        or len(set(evidence)) != len(evidence)
        or any(
            fragment.parent != COMPAT_UNIT_DIRECTORY
            or fragment.name != unit
            for unit, fragment in zip(units, fragments)
        )
        or "preaccess-compat-c5" not in COMPAT_UNIT
        or any("compat_c5" not in path.name for path in evidence)
    ):
        raise PermissionError("c5 unit/evidence namespace is not disjoint")


def _build_c5_closure(
    authorization_body: Mapping[str, object],
) -> dict[str, object]:
    _require_c1_alias_identities()
    closure = dict(_C1_BUILD_CLOSURE(authorization_body))
    closure["schema_version"] = COMPATIBILITY_CLOSURE_SCHEMA
    closure["runtime_compatibility_generation"] = "c5"
    closure[PROTECTED_UNITS_KEY] = _protected_units()
    body = dict(closure)
    body.pop("closure_fingerprint", None)
    closure["closure_fingerprint"] = _compat_c1._fingerprint(body)
    return closure


def _validate_c5_closure(
    authorization: Mapping[str, object],
    *,
    require_fresh: bool = True,
    require_bridge_fresh: bool = True,
    require_future_absence: bool = True,
) -> dict[str, object]:
    _require_c1_alias_identities()
    closure = authorization.get(COMPATIBILITY_CLOSURE_KEY)
    if (
        not isinstance(closure, Mapping)
        or closure.get("schema_version") != COMPATIBILITY_CLOSURE_SCHEMA
        or closure.get("runtime_compatibility_generation") != "c5"
        or closure.get(PROTECTED_UNITS_KEY) != _protected_units()
        or closure.get("fixed_paths") != _fixed_paths()
    ):
        raise PermissionError("c5 compatibility closure identity changed")
    body = dict(closure)
    fingerprint = body.pop("closure_fingerprint", None)
    if fingerprint != _compat_c1._fingerprint(body):
        raise PermissionError("c5 compatibility closure fingerprint changed")

    projected = deepcopy(dict(authorization))
    projected_closure = dict(projected[COMPATIBILITY_CLOSURE_KEY])
    projected_closure.pop(PROTECTED_UNITS_KEY, None)
    projected_closure["schema_version"] = (
        "cure-lite-v24-preaccess-compat-c1-unit-realization-closure-v1"
    )
    projected_closure["runtime_compatibility_generation"] = "c1"
    projected_body = dict(projected_closure)
    projected_body.pop("closure_fingerprint", None)
    projected_closure["closure_fingerprint"] = _compat_c1._fingerprint(
        projected_body,
    )
    projected[COMPATIBILITY_CLOSURE_KEY] = projected_closure
    _C1_VALIDATE_CLOSURE(
        projected,
        require_fresh=require_fresh,
        require_bridge_fresh=require_bridge_fresh,
        require_future_absence=require_future_absence,
    )
    return dict(closure)


def _configure_c5_identity() -> None:
    """Rebind every c1 runtime identity before any delegated operation."""

    _require_canonical_c5_identity()
    _require_disjoint_c5_namespace()
    _require_runtime_dynamic_shadow_contract()
    instruction_id, authorization_basis = _require_b5_authority_contract()
    _compat_c1.COMPAT_REALIZER_PATH = COMPAT_REALIZER_PATH
    _compat_c1.FROZEN_REALIZER_PATH = FROZEN_REALIZER_PATH
    _compat_c1.FROZEN_REALIZER_SHA256 = FROZEN_REALIZER_SHA256
    _compat_c1._COMPAT_SOURCE_LOAD_BINDING = _SELF_LOAD_GENERATION
    _compat_c1._FROZEN_REALIZER_LOAD_BINDING = _C1_LOAD_GENERATION
    _compat_c1.COMPAT_UNIT = COMPAT_UNIT
    _compat_c1.PROTECTED_PREDECESSOR_UNIT = PROTECTED_C1_UNIT
    _compat_c1.COMPAT_UNIT_DIRECTORY = COMPAT_UNIT_DIRECTORY
    _compat_c1.PROTECTED_PREDECESSOR_FRAGMENT_PATH = (
        PROTECTED_C1_FRAGMENT_PATH
    )
    _compat_c1.COMPAT_TEMPLATE_PATH = COMPAT_TEMPLATE_PATH
    _compat_c1.COMPAT_TEMPLATE_SHA256 = COMPAT_TEMPLATE_SHA256
    _compat_c1.COMPAT_SUPERVISOR_PATH = COMPAT_SUPERVISOR_PATH
    _compat_c1.COMPAT_BRIDGE_SOURCE_PATH = COMPAT_BRIDGE_SOURCE_PATH
    _compat_c1.COMPAT_BRIDGE_SOURCE_SHA256 = COMPAT_BRIDGE_SOURCE_SHA256
    _compat_c1.COMPAT_BRIDGE_AUTHORIZATION_PATH = (
        COMPAT_BRIDGE_AUTHORIZATION_PATH
    )
    _compat_c1.COMPAT_RUNTIME_SPEC_PATH = COMPAT_RUNTIME_SPEC_PATH
    _compat_c1.COMPAT_AUTHORIZATION_PATH = COMPAT_AUTHORIZATION_PATH
    _compat_c1.COMPAT_RECEIPT_PATH = COMPAT_RECEIPT_PATH
    _compat_c1.COMPAT_TERMINAL_PATH = COMPAT_TERMINAL_PATH
    _compat_c1.__doc__ = __doc__
    _compat_c1._canonical_json = _canonical_json
    _compat_c1._fingerprint = _fingerprint
    _compat_c1.COMPATIBILITY_CLOSURE_KEY = COMPATIBILITY_CLOSURE_KEY
    _compat_c1.COMPATIBILITY_CLOSURE_SCHEMA = (
        "cure-lite-v24-preaccess-compat-c1-unit-realization-closure-v1"
    )
    _apply_exact_frozen_globals(
        instruction_id=instruction_id,
        authorization_basis=authorization_basis,
    )
    _compat_c1._fixed_paths = _fixed_paths
    _compat_c1._validate_bridge_authorization = (
        _validate_c5_bridge_authorization
    )
    _compat_c1._build_compatibility_closure = _build_c5_closure
    _compat_c1._validate_compatibility_closure = _validate_c5_closure
    _compat_c1._compat_expected_static_shadow = (
        _c5_expected_static_shadow
    )
    _compat_c1._configure_isolated_namespace()
    _install_mutation_guards()
    _require_exact_frozen_globals()
    _require_mutation_guard_identities()


def _c5_expected_static_shadow(fragment_path: Path) -> dict[str, str]:
    result = dict(_C1_COMPAT_EXPECTED_STATIC_SHADOW(fragment_path))
    if (
        result.get("Id") != COMPAT_UNIT
        or result.get("LoadState") != "loaded"
        or result.get("ActiveState") != "inactive"
        or result.get("SubState") != "dead"
        or result.get("UnitFileState") != "static"
        or result.get("Restart") != "no"
        or result.get("NRestarts") != "0"
        or result.get("NeedDaemonReload") != "no"
    ):
        raise RuntimeError("frozen static-shadow contract changed")
    result["InvocationID"] = ""
    return result


def _require_rebound_c1_identity() -> None:
    expected_paths = {
        "COMPAT_REALIZER_PATH": COMPAT_REALIZER_PATH,
        "FROZEN_REALIZER_PATH": FROZEN_REALIZER_PATH,
        "COMPAT_UNIT_DIRECTORY": COMPAT_UNIT_DIRECTORY,
        "PROTECTED_PREDECESSOR_FRAGMENT_PATH": (
            PROTECTED_C1_FRAGMENT_PATH
        ),
        "COMPAT_TEMPLATE_PATH": COMPAT_TEMPLATE_PATH,
        "COMPAT_SUPERVISOR_PATH": COMPAT_SUPERVISOR_PATH,
        "COMPAT_BRIDGE_SOURCE_PATH": COMPAT_BRIDGE_SOURCE_PATH,
        "COMPAT_BRIDGE_AUTHORIZATION_PATH": (
            COMPAT_BRIDGE_AUTHORIZATION_PATH
        ),
        "COMPAT_RUNTIME_SPEC_PATH": COMPAT_RUNTIME_SPEC_PATH,
        "COMPAT_AUTHORIZATION_PATH": COMPAT_AUTHORIZATION_PATH,
        "COMPAT_RECEIPT_PATH": COMPAT_RECEIPT_PATH,
        "COMPAT_TERMINAL_PATH": COMPAT_TERMINAL_PATH,
    }
    if (
        any(
            getattr(_compat_c1, name, None) != value
            for name, value in expected_paths.items()
        )
        or _compat_c1.FROZEN_REALIZER_SHA256 != FROZEN_REALIZER_SHA256
        or _compat_c1.COMPAT_TEMPLATE_SHA256 != COMPAT_TEMPLATE_SHA256
        or _compat_c1.COMPAT_BRIDGE_SOURCE_SHA256
        != COMPAT_BRIDGE_SOURCE_SHA256
        or _compat_c1.COMPAT_UNIT != COMPAT_UNIT
        or _compat_c1.PROTECTED_PREDECESSOR_UNIT != PROTECTED_C1_UNIT
        or _compat_c1.COMPATIBILITY_CLOSURE_KEY
        != COMPATIBILITY_CLOSURE_KEY
        or _compat_c1.COMPATIBILITY_CLOSURE_SCHEMA
        != "cure-lite-v24-preaccess-compat-c1-unit-realization-closure-v1"
        or not _type_exact_equal(
            _compat_c1._COMPAT_SOURCE_LOAD_BINDING,
            _SELF_LOAD_GENERATION,
        )
        or not _type_exact_equal(
            _compat_c1._FROZEN_REALIZER_LOAD_BINDING,
            _C1_LOAD_GENERATION,
        )
        or _compat_c1._fixed_paths is not _fixed_paths
        or _compat_c1._validate_bridge_authorization
        is not _validate_c5_bridge_authorization
        or _compat_c1._build_compatibility_closure
        is not _build_c5_closure
        or _compat_c1._validate_compatibility_closure
        is not _validate_c5_closure
        or _compat_c1._compat_expected_static_shadow
        is not _c5_expected_static_shadow
        or _compat_c1.legacy._expected_static_shadow
        is not _c5_expected_static_shadow
        or _C1_CREATE_AUTHORIZATION
        is not _compat_c1.create_authorization
        or _C1_REALIZE_ACTUAL_UNIT
        is not _compat_c1.realize_actual_unit
    ):
        raise PermissionError("c5 delegated rebound identity changed")


def _validate_c5_receipt_contract(
    receipt: object,
) -> dict[str, object]:
    if not isinstance(receipt, Mapping):
        raise PermissionError("c5 realization receipt is malformed")
    shadow = receipt.get("full_static_shadow")
    if (
        receipt.get("unit_name") != COMPAT_UNIT
        or receipt.get("completed_actions")
        != list(_REALIZATION_ACTIONS)
        or receipt.get("runtime_spec_absent_at_receipt") is not True
        or receipt.get("static") is not True
        or receipt.get("enabled") is not False
        or receipt.get("started") is not False
        or receipt.get("removed") is not False
        or receipt.get("passed") is not True
        or not isinstance(shadow, Mapping)
        or shadow.get("Id") != COMPAT_UNIT
        or any(
            shadow.get(field) != expected
            for field, expected in _RECEIPT_STATIC_STATE.items()
        )
    ):
        raise PermissionError("c5 receipt is not exact static/inert PASS")
    return dict(receipt)


def verify_compatibility_identity() -> dict[str, object]:
    _require_frozen_c5_sources()
    _require_transitive_callable_identities()
    _require_c1_alias_identities()
    _require_canonical_c5_identity()
    _require_exact_frozen_globals()
    _require_mutation_guard_identities()
    _require_rebound_c1_identity()
    _require_source_generations()
    _configure_c5_identity()
    _require_transitive_callable_identities()
    _require_rebound_c1_identity()
    if (
        frozenset(_compat_c1._RUNTIME_DYNAMIC_SHADOW_FIELDS)
        != _R5_RUNTIME_DYNAMIC_SHADOW_FIELDS
        or _compat_c1.legacy.INSTRUCTION_ID != B5_INSTRUCTION_ID
        or _compat_c1.legacy.AUTHORIZATION_BASIS
        != B5_AUTHORIZATION_BASIS
    ):
        raise PermissionError(
            "c5 runtime dynamic-shadow identity changed",
        )
    predecessor = _compat_c1._require_frozen_predecessor_generation()
    own = _compat_c1._require_running_source_generation()
    result = dict(_C1_VERIFY_IDENTITY())
    expected = {
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_generation": "c1",
        "unit_name": COMPAT_UNIT,
        "runtime_spec_path": str(COMPAT_RUNTIME_SPEC_PATH),
        "authorization_path": str(COMPAT_AUTHORIZATION_PATH),
        "receipt_path": str(COMPAT_RECEIPT_PATH),
        "terminal_path": str(COMPAT_TERMINAL_PATH),
        "template_path": str(COMPAT_TEMPLATE_PATH),
        "supervisor_path": str(COMPAT_SUPERVISOR_PATH),
        "bridge_validator_path": str(COMPAT_BRIDGE_SOURCE_PATH),
        "bridge_validator_file_sha256": COMPAT_BRIDGE_SOURCE_SHA256,
        "bridge_compat_authorization_path": str(
            COMPAT_BRIDGE_AUTHORIZATION_PATH,
        ),
        "frozen_realizer_path": str(FROZEN_REALIZER_PATH),
        "frozen_realizer_file_sha256": FROZEN_REALIZER_SHA256,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise PermissionError("c5 realizer transitive identity changed")
    result.update(
        {
            "runtime_compatibility_generation": "c5",
            "frozen_c1_realizer_generation": predecessor,
            "compat_c5_realizer_generation": own,
            "base_realizer_path": str(BASE_REALIZER_PATH),
            "base_realizer_file_sha256": BASE_REALIZER_SHA256,
            "base_realizer_generation": dict(BASE_REALIZER_GENERATION),
            "protected_original_unit": PROTECTED_ORIGINAL_UNIT,
            "protected_c1_unit": PROTECTED_C1_UNIT,
            "protected_c2_unit": PROTECTED_C2_UNIT,
            "protected_c3_unit": PROTECTED_C3_UNIT,
            "protected_c4_unit": PROTECTED_C4_UNIT,
            "protected_original_unit_mutation_authorized": False,
            "protected_c1_unit_mutation_authorized": False,
            "protected_c2_unit_mutation_authorized": False,
            "protected_c3_unit_mutation_authorized": False,
            "protected_c4_unit_mutation_authorized": False,
            "compat_supervisor_file_sha256": (
                COMPAT_SUPERVISOR_SHA256
            ),
            "compat_template_file_sha256": COMPAT_TEMPLATE_SHA256,
            "compat_bridge_file_sha256": COMPAT_BRIDGE_SOURCE_SHA256,
            "instruction_id": B5_INSTRUCTION_ID,
            "authorization_basis": B5_AUTHORIZATION_BASIS,
            "realization_actions": list(_REALIZATION_ACTIONS),
            "runtime_dynamic_shadow_fields": sorted(
                _R5_RUNTIME_DYNAMIC_SHADOW_FIELDS
            ),
            "persistent_install_authorized": False,
        },
    )
    return result


def _build_mutation_delegation_lane(
    module: ModuleType,
    *,
    c1_create,
    c1_realize,
    frozen_write,
    frozen_create,
    frozen_realize,
    compat_write,
    bound_write,
    install_fragment,
    base_run,
    c1_main,
    base_main,
    frozen_validate,
):
    legacy_module = module.legacy
    context: ContextVar[object | None] = ContextVar(
        "cure_lite_v24_r5_mutation_capability",
        default=None,
    )
    authorize_capability = object()
    apply_capability = object()
    r5_shadow_properties = tuple(
        name
        for name in legacy_module._SHADOW_PROPERTIES
        if name != "InvocationID"
    ) + ("InvocationID",)
    shadow_command: list[str] = [
        legacy_module.SYSTEMCTL,
        "--user",
        "show",
        COMPAT_UNIT,
        "--no-pager",
    ]
    for property_name in r5_shadow_properties:
        shadow_command.extend(("-p", property_name))
    read_only_commands = frozenset(
        {
            (
                legacy_module.SYSTEMD_PATH,
                "--suffix=systemd/user",
                "user-runtime",
            ),
            (
                legacy_module.SYSTEMD_ANALYZE,
                "--user",
                "unit-paths",
                "--no-pager",
            ),
            tuple(shadow_command),
        },
    )
    daemon_reload_command = (
        legacy_module.SYSTEMCTL,
        "--user",
        "daemon-reload",
    )

    def require(operation: str) -> None:
        expected = {
            "authorize": authorize_capability,
            "apply": apply_capability,
        }[operation]
        if context.get() is not expected:
            raise PermissionError(
                f"c5 {operation} mutation requires R5 delegation",
            )

    def require_caller(*functions) -> None:
        caller = sys._getframe(2).f_code
        if all(caller is not function.__code__ for function in functions):
            raise PermissionError("c5 delegated sub-operation changed")

    def deprivilege_callbacks(kwargs):
        guarded_kwargs = dict(kwargs)
        for name in ("runner", "manager_reader"):
            callback = guarded_kwargs.get(name)
            if callback is None:
                continue

            def without_capability(*args, __callback=callback, **inner):
                token = context.set(None)
                try:
                    return __callback(*args, **inner)
                finally:
                    context.reset(token)

            guarded_kwargs[name] = without_capability
        return guarded_kwargs

    def path_argument(args, kwargs, position: int, name: str) -> Path:
        value = args[position] if len(args) > position else kwargs.get(name)
        if not isinstance(value, Path):
            raise PermissionError(f"c5 delegated path is not exact:{name}")
        return value

    def require_authorize_call(args, kwargs) -> None:
        require("authorize")
        if (
            path_argument(args, kwargs, 0, "authorization_path")
            != COMPAT_AUTHORIZATION_PATH
            or kwargs.get("template_path") != COMPAT_TEMPLATE_PATH
            or kwargs.get("python_path") != Path("/usr/bin/python3.12")
            or kwargs.get("supervisor_path") != COMPAT_SUPERVISOR_PATH
            or kwargs.get("runtime_spec_path") != COMPAT_RUNTIME_SPEC_PATH
            or kwargs.get("instruction_id") != B5_INSTRUCTION_ID
            or kwargs.get("authorization_basis")
            != B5_AUTHORIZATION_BASIS
        ):
            raise PermissionError("c5 delegated authorize call changed")

    def require_apply_call(args, kwargs) -> None:
        require("apply")
        if (
            path_argument(args, kwargs, 0, "authorization_path")
            != COMPAT_AUTHORIZATION_PATH
            or kwargs.get("receipt_path") != COMPAT_RECEIPT_PATH
            or kwargs.get("terminal_path") != COMPAT_TERMINAL_PATH
        ):
            raise PermissionError("c5 delegated apply call changed")

    def require_write_call(args, kwargs) -> None:
        capability = context.get()
        if capability not in {authorize_capability, apply_capability}:
            raise PermissionError(
                "c5 write mutation requires R5 delegation",
            )
        path = path_argument(args, kwargs, 0, "path")
        body = args[1] if len(args) > 1 else kwargs.get("body")
        fingerprint = kwargs.get("fingerprint_field")
        if not isinstance(body, Mapping):
            raise PermissionError("c5 delegated write body is malformed")
        schema = body.get("schema_version")
        expected = None
        if capability is authorize_capability:
            expected = (
                COMPAT_AUTHORIZATION_PATH,
                legacy_module.AUTHORIZATION_SCHEMA,
                "authorization_fingerprint",
            )
        elif capability is apply_capability:
            if schema == legacy_module.RECEIPT_SCHEMA:
                expected = (
                    COMPAT_RECEIPT_PATH,
                    legacy_module.RECEIPT_SCHEMA,
                    "receipt_fingerprint",
                )
            elif schema == legacy_module.TERMINAL_SCHEMA:
                expected = (
                    COMPAT_TERMINAL_PATH,
                    legacy_module.TERMINAL_SCHEMA,
                    "terminal_fingerprint",
                )
        if expected != (path, schema, fingerprint):
            raise PermissionError("c5 delegated write contract changed")

    def guarded_c1_create(*args, **kwargs):
        require_authorize_call(args, kwargs)
        require_caller(invoke_authorize)
        return c1_create(*args, **kwargs)

    def guarded_frozen_create(*args, **kwargs):
        require_authorize_call(args, kwargs)
        require_caller(c1_create)
        return frozen_create(*args, **kwargs)

    def guarded_c1_realize(*args, **kwargs):
        require_apply_call(args, kwargs)
        require_caller(invoke_apply)
        return c1_realize(*args, **kwargs)

    def guarded_frozen_realize(*args, **kwargs):
        require_apply_call(args, kwargs)
        require_caller(c1_realize)
        return frozen_realize(*args, **kwargs)

    def guarded_compat_write(*args, **kwargs):
        require_write_call(args, kwargs)
        require_caller(frozen_create, frozen_realize)
        return compat_write(*args, **kwargs)

    def guarded_frozen_write(*args, **kwargs):
        require_write_call(args, kwargs)
        require_caller(compat_write)
        return frozen_write(*args, **kwargs)

    def guarded_bound_write(*args, **kwargs):
        require_write_call(args, kwargs)
        require_caller(frozen_write)
        return bound_write(*args, **kwargs)

    def guarded_install_fragment(*args, **kwargs):
        require("apply")
        require_caller(frozen_realize)
        unit_directory = (
            args[0] if args else kwargs.get("unit_directory")
        )
        rendered = args[1] if len(args) > 1 else kwargs.get("rendered")
        if (
            unit_directory != COMPAT_UNIT_DIRECTORY
            or not isinstance(rendered, bytes)
            or not rendered
        ):
            raise PermissionError("c5 delegated fragment install changed")
        return install_fragment(*args, **kwargs)

    def guarded_run(*args, **kwargs):
        argv = args[0] if args else kwargs.get("argv")
        command = tuple(argv or ())
        if command == daemon_reload_command:
            require("apply")
            require_caller(frozen_realize)
        elif command not in read_only_commands:
            raise PermissionError("c5 delegated command is outside closure")
        return base_run(*args, **kwargs)

    def guarded_c1_main(*args, **kwargs):
        del args, kwargs
        raise PermissionError("c5 delegated C1 CLI is disabled")

    def guarded_base_main(*args, **kwargs):
        del args, kwargs
        raise PermissionError("c5 delegated base CLI is disabled")

    def install_guards() -> None:
        module._frozen_write_create_once_json = guarded_frozen_write
        module._frozen_create_authorization = guarded_frozen_create
        module._frozen_realize_actual_unit = guarded_frozen_realize
        module._compat_write_create_once_json = guarded_compat_write
        module.create_authorization = guarded_c1_create
        module.realize_actual_unit = guarded_c1_realize
        module.main = guarded_c1_main
        legacy_module._write_create_once_json_bound = guarded_bound_write
        legacy_module._install_fragment = guarded_install_fragment
        legacy_module.write_create_once_json = guarded_compat_write
        legacy_module.create_authorization = guarded_c1_create
        legacy_module.realize_actual_unit = guarded_c1_realize
        legacy_module.main = guarded_base_main
        legacy_module._run = guarded_run

    def require_guard_identities() -> None:
        if (
            module._frozen_write_create_once_json
            is not guarded_frozen_write
            or module._frozen_create_authorization
            is not guarded_frozen_create
            or module._frozen_validate_authorization is not frozen_validate
            or module._frozen_realize_actual_unit
            is not guarded_frozen_realize
            or module._compat_write_create_once_json
            is not guarded_compat_write
            or module.create_authorization is not guarded_c1_create
            or module.realize_actual_unit is not guarded_c1_realize
            or module.main is not guarded_c1_main
            or legacy_module._write_create_once_json_bound
            is not guarded_bound_write
            or legacy_module._install_fragment is not guarded_install_fragment
            or legacy_module.write_create_once_json
            is not guarded_compat_write
            or legacy_module.create_authorization is not guarded_c1_create
            or legacy_module.realize_actual_unit is not guarded_c1_realize
            or legacy_module.main is not guarded_base_main
            or legacy_module._run is not guarded_run
        ):
            raise PermissionError("c5 mutation guard identity changed")

    def invoke_authorize(*args, **kwargs):
        verify_compatibility_identity()
        phase_token = _RUNTIME_PHASE_CONTEXT.set(
            RUNTIME_PHASE_PREACTIVATION,
        )
        capability_token = context.set(authorize_capability)
        try:
            return guarded_c1_create(
                *args,
                **deprivilege_callbacks(kwargs),
            )
        finally:
            context.reset(capability_token)
            _RUNTIME_PHASE_CONTEXT.reset(phase_token)

    def invoke_apply(*args, **kwargs):
        verify_compatibility_identity()
        phase_token = _RUNTIME_PHASE_CONTEXT.set(
            RUNTIME_PHASE_PREACTIVATION,
        )
        capability_token = context.set(apply_capability)
        try:
            return guarded_c1_realize(
                *args,
                **deprivilege_callbacks(kwargs),
            )
        finally:
            context.reset(capability_token)
            _RUNTIME_PHASE_CONTEXT.reset(phase_token)

    return (
        install_guards,
        require_guard_identities,
        invoke_authorize,
        invoke_apply,
        guarded_c1_create,
        guarded_c1_realize,
    )


(
    _install_mutation_guards,
    _require_mutation_guard_identities,
    _invoke_c1_authorize,
    _invoke_c1_apply,
    _C1_CREATE_AUTHORIZATION,
    _C1_REALIZE_ACTUAL_UNIT,
) = _build_mutation_delegation_lane(
    _compat_c1,
    c1_create=_C1_CREATE_AUTHORIZATION_RAW,
    c1_realize=_C1_REALIZE_ACTUAL_UNIT_RAW,
    frozen_write=_compat_c1._frozen_write_create_once_json,
    frozen_create=_compat_c1._frozen_create_authorization,
    frozen_realize=_compat_c1._frozen_realize_actual_unit,
    compat_write=_compat_c1._compat_write_create_once_json,
    bound_write=_compat_c1.legacy._write_create_once_json_bound,
    install_fragment=_compat_c1.legacy._install_fragment,
    base_run=_compat_c1.legacy._run,
    c1_main=_compat_c1.main,
    base_main=_compat_c1.legacy.main,
    frozen_validate=_compat_c1._frozen_validate_authorization,
)
del _build_mutation_delegation_lane
del _C1_CREATE_AUTHORIZATION_RAW
del _C1_REALIZE_ACTUAL_UNIT_RAW


def _build_public_mutation_entries(
    authorize_entry,
    apply_entry,
    receipt_validator,
):
    """Closure-bind critical entries before exposing the public write lane."""

    def create_authorization(*args, **kwargs):
        return authorize_entry(*args, **kwargs)

    def realize_actual_unit(*args, **kwargs):
        result = apply_entry(*args, **kwargs)
        return receipt_validator(result)

    return create_authorization, realize_actual_unit


create_authorization, realize_actual_unit = _build_public_mutation_entries(
    _invoke_c1_authorize,
    _invoke_c1_apply,
    _validate_c5_receipt_contract,
)
del _build_public_mutation_entries


def validate_authorization(*args, **kwargs):
    verify_compatibility_identity()
    token = _RUNTIME_PHASE_CONTEXT.set(RUNTIME_PHASE_PREACTIVATION)
    try:
        return _C1_VALIDATE_AUTHORIZATION(*args, **kwargs)
    finally:
        _RUNTIME_PHASE_CONTEXT.reset(token)


def validate_archival_realization_chain(*args, **kwargs):
    runtime_phase = kwargs.pop("runtime_phase", None)
    if "allow_runtime_activation" in kwargs:
        allow_runtime_activation = kwargs["allow_runtime_activation"]
    else:
        allow_runtime_activation = (
            runtime_phase is not None
            and runtime_phase != RUNTIME_PHASE_PREACTIVATION
        )
    phase = _resolve_runtime_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    kwargs["allow_runtime_activation"] = (
        phase != RUNTIME_PHASE_PREACTIVATION
    )
    verify_compatibility_identity()
    token = _RUNTIME_PHASE_CONTEXT.set(phase)
    try:
        result = _C1_VALIDATE_ARCHIVAL_CHAIN(*args, **kwargs)
    finally:
        _RUNTIME_PHASE_CONTEXT.reset(token)
    if not isinstance(result, Mapping):
        raise PermissionError("c5 archival realization chain is malformed")
    _validate_c5_receipt_contract(result.get("receipt"))
    return result


def _parser():
    _require_canonical_c5_identity()
    _require_transitive_callable_identities()
    _require_exact_frozen_globals()
    _require_mutation_guard_identities()
    _require_rebound_c1_identity()
    return _compat_c1._parser()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "authorize":
        create_authorization(
            args.output,
            template_path=args.template,
            python_path=args.python,
            supervisor_path=args.supervisor,
            runtime_spec_path=args.runtime_spec,
            authorization_basis=args.authorization_basis,
            instruction_id=args.instruction_id,
            validity_seconds=args.validity_seconds,
        )
        return 0
    realize_actual_unit(
        args.authorization,
        receipt_path=args.receipt,
        terminal_path=args.terminal_receipt,
    )
    return 0


_configure_c5_identity()
(
    _require_transitive_callable_identities,
    _ACTIVE_CALLABLE_COVERAGE,
) = _build_callable_state_guard(
    ("c1", _compat_c1),
    ("base", _compat_c1.legacy),
    require_active_names=True,
    previous_guard=_require_pristine_callable_states,
    critical_globals={
        "_invoke_c1_authorize": _invoke_c1_authorize,
        "_invoke_c1_apply": _invoke_c1_apply,
        "_validate_c5_receipt_contract": _validate_c5_receipt_contract,
    },
)
del _build_callable_state_guard


class _GuardedLegacyView:
    """Strict facade exposing only audited reads and R5 public entries."""

    __slots__ = ()
    _READ_ALLOWLIST = frozenset(
        {
            "ACTUAL_UNIT",
            "AUTHORIZATION_BASIS",
            "AUTHORIZATION_SCHEMA",
            "INSTRUCTION_ID",
            "RECEIPT_SCHEMA",
            "SUPERVISOR_SPEC_SCHEMA",
            "SYSTEMCTL",
            "SYSTEMD_ANALYZE",
            "SYSTEMD_PATH",
            "TERMINAL_SCHEMA",
            "__file__",
            "_ACTIONS",
            "_AUTH_KEYS",
            "_EXEC_MODES",
            "_FILE_BINDING_KEYS",
            "_SHADOW_PROPERTIES",
            "_expected_exec",
            "_expected_static_shadow",
            "_observe_unit_path_policy",
            "_parse_show",
            "_path_row",
            "_read_all_fd",
            "_read_file_binding",
            "_require_no_shadow",
            "_run",
            "_shadow_argv",
            "_stable_read_file",
            "_validate_daemon_reload_path_policy_transition",
            "_validate_manager_generation",
            "_validate_python_binding",
            "collect_manager_generation",
            "freeze_unit_path_policy",
            "load_sealed_json",
            "query_shadow",
            "render_fragment",
            "validate_installed_shadow",
        }
    )

    def __getattr__(self, name: str):
        if name not in self._READ_ALLOWLIST:
            raise AttributeError(f"legacy write bypass is unavailable:{name}")
        value = getattr(_compat_c1.legacy, name)
        if isinstance(value, (dict, list, set)):
            return deepcopy(value)
        return value

    def create_authorization(self, *args, **kwargs):
        return create_authorization(*args, **kwargs)

    def validate_authorization(self, *args, **kwargs):
        return validate_authorization(*args, **kwargs)

    def realize_actual_unit(self, *args, **kwargs):
        return realize_actual_unit(*args, **kwargs)

    def validate_archival_realization_chain(self, *args, **kwargs):
        return validate_archival_realization_chain(*args, **kwargs)

    def write_create_once_json(self, *args, **kwargs):
        del args, kwargs
        raise PermissionError("legacy write helper is disabled")

    def main(self, *args, **kwargs):
        del args, kwargs
        raise PermissionError("legacy CLI entry point is disabled")


class _ReadOnlyCompatC1View:
    """Defensive read facade; raw C1 modules and mutators stay internal."""

    __slots__ = ()
    _READ_ALLOWLIST = frozenset(
        {
            "COMPAT_AUTHORIZATION_PATH",
            "COMPAT_BRIDGE_AUTHORIZATION_PATH",
            "COMPAT_RECEIPT_PATH",
            "COMPAT_RUNTIME_SPEC_PATH",
            "COMPAT_TEMPLATE_PATH",
            "COMPAT_TEMPLATE_SHA256",
            "COMPAT_TERMINAL_PATH",
            "COMPAT_UNIT",
            "_MUTATION_AUTHORITY",
            "_RUNTIME_DYNAMIC_SHADOW_FIELDS",
            "_fingerprint",
            "_parent_binding",
            "_require_create_targets_absent",
            "_same_generation",
            "_stable_regular_read",
        }
    )

    @property
    def legacy(self) -> _GuardedLegacyView:
        return legacy

    def __getattr__(self, name: str):
        if name not in self._READ_ALLOWLIST:
            raise AttributeError(f"C1 raw surface is unavailable:{name}")
        value = getattr(_compat_c1, name)
        if isinstance(value, (dict, list, set)):
            return deepcopy(value)
        if isinstance(value, ModuleType):
            raise AttributeError(f"C1 module surface is unavailable:{name}")
        return value


legacy = _GuardedLegacyView()
compat_c1 = _ReadOnlyCompatC1View()


if __name__ == "__main__":
    raise SystemExit(main())
