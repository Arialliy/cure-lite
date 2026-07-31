#!/usr/bin/env python3
"""Create-only unit realizer for runtime compatibility generation c2.

The audited c1 realizer is read through a no-follow descriptor and its frozen
SHA-256 is proved before any predecessor byte is compiled.  This wrapper then
rebinds that verified implementation to the disjoint c2 unit namespace.

Only creation of the new c2 static fragment and ``daemon-reload`` are
authorized.  The original r2 unit and the c1 unit are both immutable.  This
module never starts, enables, stops, removes, retries, resumes, or accesses a
scientific payload.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
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
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service"
)
PROTECTED_ORIGINAL_UNIT = "cure-lite-v24-gcr-pacre-dr-r2.service"
PROTECTED_C1_UNIT = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service"
)
COMPAT_UNIT_DIRECTORY = Path(f"/run/user/{os.getuid()}/systemd/user")
PROTECTED_ORIGINAL_FRAGMENT_PATH = (
    COMPAT_UNIT_DIRECTORY / PROTECTED_ORIGINAL_UNIT
)
PROTECTED_C1_FRAGMENT_PATH = COMPAT_UNIT_DIRECTORY / PROTECTED_C1_UNIT
COMPAT_TEMPLATE_PATH = (
    REPOSITORY
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service.template"
).resolve()
COMPAT_TEMPLATE_SHA256 = (
    "4e485e5ba86a79b9244fb73d5add1a7015d71aa5f16f56479bd9c2c200d12967"
)
COMPAT_SUPERVISOR_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c2.py"
).resolve()
COMPAT_BRIDGE_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c2.py"
).resolve()
COMPAT_BRIDGE_SOURCE_SHA256 = (
    "f077a3b91850deb3c6652afbcc3a77a6932cdb3cb353e06c32bb81d9da4afedf"
)
COMPAT_BRIDGE_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c2_authorization.json"
)
COMPAT_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_spec.json"
)
COMPAT_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c2_unit_realization_authorization.json"
)
COMPAT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c2_unit_realization_receipt.json"
)
COMPAT_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c2_unit_realization_terminal.json"
)

COMPATIBILITY_CLOSURE_SCHEMA = (
    "cure-lite-v24-preaccess-compat-c2-unit-realization-closure-v1"
)
COMPATIBILITY_CLOSURE_KEY = "compatibility_closure"
PROTECTED_UNITS_KEY = "protected_unit_generations"
BRIDGE_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c2-authorization-v1"
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
        "compat_c1_verified_for_c2"
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


compat_c1, _C1_LOAD_GENERATION = _load_frozen_c1()
_SELF_LOAD_BYTES, _SELF_LOAD_GENERATION = _stable_source_bytes(
    COMPAT_REALIZER_PATH,
)

_C1_BUILD_CLOSURE = compat_c1._build_compatibility_closure
_C1_VALIDATE_CLOSURE = compat_c1._validate_compatibility_closure
_C1_FIXED_PATHS = compat_c1._fixed_paths
_C1_VERIFY_IDENTITY = compat_c1.verify_compatibility_identity
_C1_CREATE_AUTHORIZATION = compat_c1.create_authorization
_C1_VALIDATE_AUTHORIZATION = compat_c1.validate_authorization
_C1_REALIZE_ACTUAL_UNIT = compat_c1.realize_actual_unit
_C1_VALIDATE_ARCHIVAL_CHAIN = compat_c1.validate_archival_realization_chain


def _require_source_generations() -> None:
    _, predecessor = _stable_source_bytes(
        FROZEN_REALIZER_PATH,
        expected_sha256=FROZEN_REALIZER_SHA256,
    )
    _, own = _stable_source_bytes(COMPAT_REALIZER_PATH)
    if (
        predecessor != _C1_LOAD_GENERATION
        or own != _SELF_LOAD_GENERATION
    ):
        raise PermissionError(
            "c2 realizer or frozen c1 predecessor generation changed",
        )


def _load_bridge_policy_from_verified_bytes(raw: bytes) -> ModuleType:
    name = (
        "tools._cure_lite_v24_preaccess_schema_compatibility_"
        "c2_for_unit_realization"
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


def _validate_c2_bridge_authorization(
    *,
    require_fresh: bool = True,
    require_future_absence: bool,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    if COMPAT_BRIDGE_SOURCE_SHA256 == "__TO_BE_FROZEN__":
        raise PermissionError("c2 bridge source generation is not frozen")
    raw, source = compat_c1._stable_regular_read(
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
        or policy.RUNTIME_COMPATIBILITY_ID != "c2"
    ):
        raise PermissionError("c2 bridge compatibility interface changed")
    authorization, root = policy.validate_compat_authorization(
        COMPAT_BRIDGE_AUTHORIZATION_PATH,
        require_fresh=require_fresh,
        require_future_absence=require_future_absence,
    )
    _, source_after = compat_c1._stable_regular_read(
        COMPAT_BRIDGE_SOURCE_PATH,
        expected_sha256=COMPAT_BRIDGE_SOURCE_SHA256,
    )
    mutation = authorization.get("mutation_authority")
    scientific = authorization.get("scientific_authority")
    if (
        not compat_c1._same_generation(source_after, source)
        or authorization.get("schema_version")
        != BRIDGE_AUTHORIZATION_SCHEMA
        or authorization.get("scientific_attempt_ordinal") != 2
        or authorization.get("runtime_compatibility_id") != "c2"
        or authorization.get("authorized_uid") != os.getuid()
        or not isinstance(mutation, Mapping)
        or mutation.get("c2_unit_realization_authorized") is not True
        or mutation.get("unit_start_authorized") is not False
        or mutation.get("unit_enable_authorized") is not False
        or mutation.get("payload_access_authorized") is not False
        or not isinstance(scientific, Mapping)
        or scientific.get("fresh_scientific_attempt") is not False
        or scientific.get("automatic_retry") is not False
        or scientific.get("resume") is not False
        or scientific.get("materialization_authorized") is not False
        or root.get("path") != str(COMPAT_BRIDGE_AUTHORIZATION_PATH)
        or root.get("fingerprint")
        != authorization.get("authorization_fingerprint")
        or root.get("mode") != 0o444
    ):
        raise PermissionError(
            "c2 bridge does not authorize the narrow unit lane",
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
    parent = compat_c1._parent_binding(fragment_path.parent)
    try:
        _, binding = compat_c1._stable_regular_read(fragment_path)
    except FileNotFoundError:
        if compat_c1._parent_binding(fragment_path.parent) != parent:
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
    return {
        "original_r2": _observe_unit(
            PROTECTED_ORIGINAL_UNIT,
            PROTECTED_ORIGINAL_FRAGMENT_PATH,
        ),
        "compat_c1": _observe_unit(
            PROTECTED_C1_UNIT,
            PROTECTED_C1_FRAGMENT_PATH,
        ),
    }


def _fixed_paths() -> dict[str, str]:
    result = dict(_C1_FIXED_PATHS())
    result.update(
        {
            "protected_original_fragment": str(
                PROTECTED_ORIGINAL_FRAGMENT_PATH,
            ),
            "protected_c1_fragment": str(PROTECTED_C1_FRAGMENT_PATH),
        },
    )
    return result


def _build_c2_closure(
    authorization_body: Mapping[str, object],
) -> dict[str, object]:
    closure = dict(_C1_BUILD_CLOSURE(authorization_body))
    closure["schema_version"] = COMPATIBILITY_CLOSURE_SCHEMA
    closure["runtime_compatibility_generation"] = "c2"
    closure[PROTECTED_UNITS_KEY] = _protected_units()
    body = dict(closure)
    body.pop("closure_fingerprint", None)
    closure["closure_fingerprint"] = compat_c1._fingerprint(body)
    return closure


def _validate_c2_closure(
    authorization: Mapping[str, object],
    *,
    require_fresh: bool = True,
    require_bridge_fresh: bool = True,
    require_future_absence: bool = True,
) -> dict[str, object]:
    closure = authorization.get(COMPATIBILITY_CLOSURE_KEY)
    if (
        not isinstance(closure, Mapping)
        or closure.get("schema_version") != COMPATIBILITY_CLOSURE_SCHEMA
        or closure.get("runtime_compatibility_generation") != "c2"
        or closure.get(PROTECTED_UNITS_KEY) != _protected_units()
        or closure.get("fixed_paths") != _fixed_paths()
    ):
        raise PermissionError("c2 compatibility closure identity changed")
    body = dict(closure)
    fingerprint = body.pop("closure_fingerprint", None)
    if fingerprint != compat_c1._fingerprint(body):
        raise PermissionError("c2 compatibility closure fingerprint changed")

    projected = deepcopy(dict(authorization))
    projected_closure = dict(projected[COMPATIBILITY_CLOSURE_KEY])
    projected_closure.pop(PROTECTED_UNITS_KEY, None)
    projected_closure["schema_version"] = (
        "cure-lite-v24-preaccess-compat-c1-unit-realization-closure-v1"
    )
    projected_closure["runtime_compatibility_generation"] = "c1"
    projected_body = dict(projected_closure)
    projected_body.pop("closure_fingerprint", None)
    projected_closure["closure_fingerprint"] = compat_c1._fingerprint(
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


def _configure_c2_identity() -> None:
    """Rebind every c1 runtime identity before any delegated operation."""

    compat_c1.COMPAT_REALIZER_PATH = COMPAT_REALIZER_PATH
    compat_c1.FROZEN_REALIZER_PATH = FROZEN_REALIZER_PATH
    compat_c1.FROZEN_REALIZER_SHA256 = FROZEN_REALIZER_SHA256
    compat_c1._COMPAT_SOURCE_LOAD_BINDING = _SELF_LOAD_GENERATION
    compat_c1._FROZEN_REALIZER_LOAD_BINDING = _C1_LOAD_GENERATION
    compat_c1.COMPAT_UNIT = COMPAT_UNIT
    compat_c1.PROTECTED_PREDECESSOR_UNIT = PROTECTED_C1_UNIT
    compat_c1.COMPAT_UNIT_DIRECTORY = COMPAT_UNIT_DIRECTORY
    compat_c1.PROTECTED_PREDECESSOR_FRAGMENT_PATH = (
        PROTECTED_C1_FRAGMENT_PATH
    )
    compat_c1.COMPAT_TEMPLATE_PATH = COMPAT_TEMPLATE_PATH
    compat_c1.COMPAT_TEMPLATE_SHA256 = COMPAT_TEMPLATE_SHA256
    compat_c1.COMPAT_SUPERVISOR_PATH = COMPAT_SUPERVISOR_PATH
    compat_c1.COMPAT_BRIDGE_SOURCE_PATH = COMPAT_BRIDGE_SOURCE_PATH
    compat_c1.COMPAT_BRIDGE_SOURCE_SHA256 = COMPAT_BRIDGE_SOURCE_SHA256
    compat_c1.COMPAT_BRIDGE_AUTHORIZATION_PATH = (
        COMPAT_BRIDGE_AUTHORIZATION_PATH
    )
    compat_c1.COMPAT_RUNTIME_SPEC_PATH = COMPAT_RUNTIME_SPEC_PATH
    compat_c1.COMPAT_AUTHORIZATION_PATH = COMPAT_AUTHORIZATION_PATH
    compat_c1.COMPAT_RECEIPT_PATH = COMPAT_RECEIPT_PATH
    compat_c1.COMPAT_TERMINAL_PATH = COMPAT_TERMINAL_PATH
    compat_c1.COMPATIBILITY_CLOSURE_KEY = COMPATIBILITY_CLOSURE_KEY
    compat_c1.COMPATIBILITY_CLOSURE_SCHEMA = (
        "cure-lite-v24-preaccess-compat-c1-unit-realization-closure-v1"
    )
    compat_c1._MUTATION_AUTHORITY = {
        **compat_c1._MUTATION_AUTHORITY,
        "protected_original_r2_unit_mutation": False,
        "protected_c1_unit_mutation": False,
    }
    compat_c1._fixed_paths = _fixed_paths
    compat_c1._validate_bridge_authorization = (
        _validate_c2_bridge_authorization
    )
    compat_c1._build_compatibility_closure = _build_c2_closure
    compat_c1._validate_compatibility_closure = _validate_c2_closure
    compat_c1.legacy.ACTUAL_UNIT = COMPAT_UNIT
    compat_c1.legacy.__file__ = str(COMPAT_REALIZER_PATH)
    compat_c1._configure_isolated_namespace()


def verify_compatibility_identity() -> dict[str, object]:
    _require_source_generations()
    if COMPAT_BRIDGE_SOURCE_SHA256 == "__TO_BE_FROZEN__":
        raise PermissionError("c2 bridge source generation is not frozen")
    _configure_c2_identity()
    predecessor = compat_c1._require_frozen_predecessor_generation()
    own = compat_c1._require_running_source_generation()
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
        raise PermissionError("c2 realizer transitive identity changed")
    result.update(
        {
            "runtime_compatibility_generation": "c2",
            "frozen_c1_realizer_generation": predecessor,
            "compat_c2_realizer_generation": own,
            "protected_original_unit": PROTECTED_ORIGINAL_UNIT,
            "protected_c1_unit": PROTECTED_C1_UNIT,
            "protected_original_unit_mutation_authorized": False,
            "protected_c1_unit_mutation_authorized": False,
        },
    )
    return result


def create_authorization(*args, **kwargs):
    verify_compatibility_identity()
    return _C1_CREATE_AUTHORIZATION(*args, **kwargs)


def validate_authorization(*args, **kwargs):
    verify_compatibility_identity()
    return _C1_VALIDATE_AUTHORIZATION(*args, **kwargs)


def realize_actual_unit(*args, **kwargs):
    verify_compatibility_identity()
    return _C1_REALIZE_ACTUAL_UNIT(*args, **kwargs)


def validate_archival_realization_chain(*args, **kwargs):
    verify_compatibility_identity()
    return _C1_VALIDATE_ARCHIVAL_CHAIN(*args, **kwargs)


def _parser():
    _configure_c2_identity()
    return compat_c1._parser()


def main(argv: Sequence[str] | None = None) -> int:
    _configure_c2_identity()
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


legacy = compat_c1.legacy
_configure_c2_identity()


def __getattr__(name: str):
    return getattr(compat_c1, name)


if __name__ == "__main__":
    raise SystemExit(main())
