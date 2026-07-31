#!/usr/bin/env python3
"""Seal the consumed c2 0400/0444 authorization-mode contract failure.

This terminalizer is forensic and metadata-only.  It fixes the exact c2
source generation, the create-once c2 authorization, and the r12 dummy
integration PASS.  Its only failure reproduction calls the read-only
realizer validator with ``require_fresh=False`` and
``require_future_absence=True``.  It never calls an authorization writer,
unit realizer, runtime launcher, payload reader, or training entry point.

Creation records a contemporaneous absence observation.  Archival validation
does not re-check those live absences: later filesystem state is not allowed
to rewrite what was observed when this terminal was sealed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
RUNS_ROOT = (REPOSITORY / "runs/irstd1k_stage_a_seed42").resolve()

TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_schema_compat_c2_mode_contract_failure_terminal.json"
)
SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c2-"
    "mode-contract-failure-terminal-v1"
)

CANDIDATE = "GCR-PACRE-v24"
STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
SCIENTIFIC_ATTEMPT_ID = (
    "gcr_pacre_v24_D_R_zero_update_structural_r2"
)
SCIENTIFIC_ATTEMPT_ORDINAL = 2
RUNTIME_COMPATIBILITY_ID = "c2"
C2_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service"
)

EXPECTED_ERROR = "c2 bridge does not authorize the narrow unit lane"
FAILED_AUTHORIZATION_SHA256 = (
    "860eaf3ff26bf1d17791cf9c292492f26c16abf6c6f15645f46b1fd138af0ae7"
)
FAILED_AUTHORIZATION_FINGERPRINT = (
    "803fc1e2834e903e9562316da9c46cc8f5d8b43bedc61f42a7d197284582107e"
)
FAILED_AUTHORIZATION_CREATED_AT = "2026-07-30T15:37:31.190746Z"
FAILED_AUTHORIZATION_ISSUED_AT = "2026-07-30T15:37:31.190746Z"
FAILED_AUTHORIZATION_EXPIRES_AT = "2026-07-30T15:42:31.190746Z"
FAILED_AUTHORIZATION_MODE = 0o400
REQUIRED_AUTHORIZATION_MODE = 0o444

FAILED_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c2_authorization.json"
)

BRIDGE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c2.py"
).resolve()
REALIZER_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c2.py"
).resolve()
SUPERVISOR_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c2.py"
).resolve()
ENVIRONMENT_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_environment_preaccess_compat_c2.py"
).resolve()
RELEASE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_runtime_release_preaccess_compat_c2.py"
).resolve()
ADAPTER_PATH = (
    REPOSITORY
    / "tools/"
    "run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c2.py"
).resolve()
TEMPLATE_PATH = (
    REPOSITORY
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service.template"
).resolve()

FAILED_SOURCE_BINDINGS: dict[str, tuple[Path, str]] = {
    "bridge_B": (
        BRIDGE_PATH,
        "f077a3b91850deb3c6652afbcc3a77a6932cdb3cb353e06c32bb81d9da4afedf",
    ),
    "realizer_R": (
        REALIZER_PATH,
        "06144d42292294603631eb778fd8bc789d40b31a73b42dcf3ceeffe95fa9f7d0",
    ),
    "supervisor_S": (
        SUPERVISOR_PATH,
        "ae9462d85a1b86d1b9ce8aad5dcc1c12b0baa3af2f7c54a84fd89ee0d692c07c",
    ),
    "environment_E": (
        ENVIRONMENT_PATH,
        "e820411771e81d93b672ab8b5f334692bdac62350c46faa878083d85df08756f",
    ),
    "release_L": (
        RELEASE_PATH,
        "08518cdffd8cc063a6e466afdd023492e09e031026a251e6c9ca07c90ea3c4b0",
    ),
    "adapter": (
        ADAPTER_PATH,
        "510826b8345948803cf976d180edb1764575874d22379e4223cb6f97dc96728f",
    ),
    "template": (
        TEMPLATE_PATH,
        "4e485e5ba86a79b9244fb73d5add1a7015d71aa5f16f56479bd9c2c200d12967",
    ),
}

_AUTH_SOURCE_LABELS = {
    "bridge_B": "compat_bridge",
    "realizer_R": "compat_unit_realizer",
    "supervisor_S": "compat_supervisor",
    "environment_E": "compat_environment_wrapper",
    "release_L": "compat_release",
    "adapter": "compat_adapter",
    "template": "compat_unit_template",
}

R12_SCENARIO_ID = (
    "supervisor-v2-dummy-compat-c2-r12-20260730c2000012"
)
R12_ROOT = (
    EVIDENCE_ROOT
    / "supervisor_v2_systemd_integration_preaccess_compat_c2_r12"
)
R12_AUTHORIZATION_PATH = R12_ROOT / "control/authorization.json"
R12_RECEIPT_PATH = R12_ROOT / "control/integration-receipt.json"
R12_FILE_BINDINGS: dict[str, tuple[Path, str]] = {
    "authorization": (
        R12_AUTHORIZATION_PATH,
        "bbc475373e21515ef69bc0f3fdd7f290430ef24ca0bc31c721cbd3c3b4047090",
    ),
    "integration_terminal": (
        R12_ROOT / "control/integration-terminal.json",
        "3eb16b9a29abef1654890e67ce34daf848bbd95dc52f2971410ca18263c4006b",
    ),
    "receipt": (
        R12_RECEIPT_PATH,
        "a1e5f4c00f21d847e6c4d4dbfc989862c54a7899760a956e4284c71d83afc39b",
    ),
    "removal_authorization": (
        R12_ROOT / "control/removal-authorization.json",
        "0153f876fe19da644624f93550ccad18f365922de8788bfa440eac740211ca1d",
    ),
    "removal_state": (
        R12_ROOT / "control/removal-state.json",
        "bb64c7f24ca3ec0a1a2b7e05917955a6cefa63fa1ff26e1d69b5d729ab9b7611",
    ),
}

C2_UNIT_FRAGMENT_PATH = (
    Path(f"/run/user/{os.getuid()}/systemd/user") / C2_UNIT_NAME
)

ABSENT_OUTPUT_PATHS: dict[str, Path] = {
    "compatibility_receipt": (
        EVIDENCE_ROOT / "r2_preaccess_schema_compat_c2_receipt.json"
    ),
    "unit_authorization": (
        EVIDENCE_ROOT
        / "r2_preaccess_compat_c2_unit_realization_authorization.json"
    ),
    "unit_receipt": (
        EVIDENCE_ROOT
        / "r2_preaccess_compat_c2_unit_realization_receipt.json"
    ),
    "unit_terminal": (
        EVIDENCE_ROOT
        / "r2_preaccess_compat_c2_unit_realization_terminal.json"
    ),
    "unit_fragment": C2_UNIT_FRAGMENT_PATH,
    "environment_policy": (
        EVIDENCE_ROOT
        / "runtime_environment_policy_preaccess_compat_c2.json"
    ),
    "environment_stability": (
        EVIDENCE_ROOT
        / "runtime_environment_stability_receipt_preaccess_compat_c2.json"
    ),
    "environment_postcleanup": (
        EVIDENCE_ROOT
        / "runtime_environment_postcleanup_receipt_preaccess_compat_c2.json"
    ),
    "runtime_spec": (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_spec.json"
    ),
    "runtime_launch_authorization": (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c2_"
        "runtime_launch_authorization.json"
    ),
    "runtime_artifacts": (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_artifacts"
    ),
    "gpu_lease": (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c2_gpu_lease"
    ),
    "compat_run_alias": (
        RUNS_ROOT
        / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c2"
    ),
    "compat_result_alias": (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c2_receipt.json"
    ),
    "scientific_run_root": (
        RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2"
    ),
    "scientific_result_receipt": (
        EVIDENCE_ROOT / "D_R_structural_attempt_r2_receipt.json"
    ),
}

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "identity",
        "terminalizer_source_root",
        "failed_generation_roots",
        "failed_authorization",
        "r12_pass_closure",
        "mode_contract_failure",
        "deterministic_reproduction",
        "historical_absence_observation",
        "payload_observation",
        "continuation_policy",
        "terminal_fingerprint",
    }
)

_CONTINUATION_POLICY = {
    "automatic_retry": False,
    "same_c2_reentry": False,
    "same_c2_reauthorization_allowed": False,
    "same_c2_metadata_repair_allowed": False,
    "c3_required": True,
    "new_explicit_authorization_required": True,
    "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
    "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
    "scientific_attempt_consumed": False,
    "unit_realization_consumed": False,
    "runtime_launch_consumed": False,
    "materialization_consumed": False,
}

_PAYLOAD_OBSERVATION = {
    "D_R_payload_accessed": False,
    "D_V_payload_accessed": False,
    "D_T_payload_accessed": False,
    "gpu_accessed": False,
    "training_started": False,
    "optimizer_steps": 0,
    "parameter_updates": 0,
    "zero_step_basis": (
        "authorization denies payload/training and no unit/runtime/"
        "scientific output existed at terminal observation"
    ),
}

_STAT_FIELDS = (
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


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def stable_fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _c2_canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _c2_fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_c2_canonical_bytes(value)).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(value, field)) for field in _STAT_FIELDS)


def _root_from(
    target: Path,
    raw: bytes,
    observed: os.stat_result,
) -> dict[str, object]:
    return {
        "path": str(target),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "owner_uid": observed.st_uid,
        "owner_gid": observed.st_gid,
        "mode": stat.S_IMODE(observed.st_mode),
        "nlink": observed.st_nlink,
        "size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def _read_regular(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, object]]:
    target = Path(path).absolute()
    parent = target.parent
    parent_before = parent.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or parent.resolve(strict=True) != parent
    ):
        raise PermissionError(f"unsafe parent for fixed file: {parent}")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        parent_opened = os.fstat(parent_fd)
        if (
            parent_opened.st_dev,
            parent_opened.st_ino,
        ) != (
            parent_before.st_dev,
            parent_before.st_ino,
        ):
            raise PermissionError(f"fixed-file parent changed: {parent}")
        descriptor = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        linked = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        parent_after = os.fstat(parent_fd)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(linked)
            or not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or target.resolve(strict=True) != target
            or (
                expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode
            )
            or (
                parent_after.st_dev,
                parent_after.st_ino,
            )
            != (
                parent_before.st_dev,
                parent_before.st_ino,
            )
        ):
            raise PermissionError(f"unsafe fixed file: {target}")
        raw = b"".join(chunks)
        root = _root_from(target, raw, before)
        if (
            expected_sha256 is not None
            and root["file_sha256"] != expected_sha256
        ):
            raise PermissionError(f"frozen file hash changed: {target}")
        return raw, root
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _load_verified_module(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
) -> tuple[ModuleType, dict[str, object]]:
    raw, before = _read_regular(
        path,
        expected_sha256=expected_sha256,
    )
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(raw, str(path), "exec", dont_inherit=True),
            module.__dict__,
        )
        _raw_after, after = _read_regular(
            path,
            expected_sha256=expected_sha256,
        )
        if after != before:
            raise PermissionError(f"module changed while loading: {path}")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, before


def _fixed_generation_roots() -> dict[str, dict[str, object]]:
    return {
        label: _read_regular(path, expected_sha256=digest)[1]
        for label, (path, digest) in FAILED_SOURCE_BINDINGS.items()
    }


def _validate_failed_authorization(
) -> tuple[dict[str, object], dict[str, object]]:
    raw, root = _read_regular(
        FAILED_AUTHORIZATION_PATH,
        expected_sha256=FAILED_AUTHORIZATION_SHA256,
        expected_mode=FAILED_AUTHORIZATION_MODE,
    )
    if (
        not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
    ):
        raise PermissionError("failed c2 authorization layout changed")
    try:
        authorization = json.loads(raw[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(
            "failed c2 authorization is not JSON"
        ) from error
    if (
        not isinstance(authorization, dict)
        or raw != _c2_canonical_bytes(authorization) + b"\n"
    ):
        raise PermissionError(
            "failed c2 authorization is not canonical"
        )
    body = dict(authorization)
    fingerprint = body.pop("authorization_fingerprint", None)
    if (
        fingerprint != FAILED_AUTHORIZATION_FINGERPRINT
        or _c2_fingerprint(body) != fingerprint
        or authorization.get("schema_version")
        != "cure-lite-v24-r2-preaccess-schema-compat-c2-authorization-v1"
        or authorization.get("candidate") != CANDIDATE
        or authorization.get("stage_id") != STAGE_ID
        or authorization.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or authorization.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or authorization.get("runtime_compatibility_id")
        != RUNTIME_COMPATIBILITY_ID
        or authorization.get("authorized_uid") != os.getuid()
        or authorization.get("created_at_utc")
        != FAILED_AUTHORIZATION_CREATED_AT
        or authorization.get("issued_at_utc")
        != FAILED_AUTHORIZATION_ISSUED_AT
        or authorization.get("expires_at_utc")
        != FAILED_AUTHORIZATION_EXPIRES_AT
        or authorization.get("D_R_payload_accessed") is not False
        or authorization.get("D_V_payload_accessed") is not False
        or authorization.get("D_T_payload_accessed") is not False
        or authorization.get("gpu_accessed") is not False
        or authorization.get("training_started") is not False
        or authorization.get("materialization_consumed") is not False
    ):
        raise PermissionError(
            "failed c2 authorization identity changed"
        )
    mutation = authorization.get("mutation_authority")
    scientific = authorization.get("scientific_authority")
    if (
        not isinstance(mutation, Mapping)
        or mutation.get("c2_unit_realization_authorized") is not True
        or mutation.get("unit_start_authorized") is not False
        or mutation.get("unit_enable_authorized") is not False
        or mutation.get("payload_access_authorized") is not False
        or mutation.get("runtime_spec_creation_authorized") is not False
        or mutation.get(
            "runtime_launch_authorization_creation_authorized"
        )
        is not False
        or not isinstance(scientific, Mapping)
        or scientific.get("fresh_scientific_attempt") is not False
        or scientific.get("automatic_retry") is not False
        or scientific.get("resume") is not False
        or scientific.get("materialization_authorized") is not False
        or scientific.get("training_authorized") is not False
        or scientific.get("D_R_payload_authorized") is not False
        or scientific.get("D_V_payload_authorized") is not False
        or scientific.get("D_T_payload_authorized") is not False
    ):
        raise PermissionError(
            "failed c2 authorization authority changed"
        )
    source_roots = authorization.get("compatibility_source_roots")
    live_roots = _fixed_generation_roots()
    if not isinstance(source_roots, Mapping):
        raise PermissionError("failed authorization source roots changed")
    for label, auth_label in _AUTH_SOURCE_LABELS.items():
        stored = source_roots.get(auth_label)
        live = live_roots[label]
        expected = {
            "path": live["path"],
            "file_sha256": live["file_sha256"],
            "device": live["device"],
            "inode": live["inode"],
            "mode": live["mode"],
            "owner_uid": live["owner_uid"],
            "size": live["size"],
        }
        if not isinstance(stored, Mapping) or dict(stored) != expected:
            raise PermissionError(
                f"failed authorization source lineage changed: {label}"
            )
    return authorization, root


def _validate_r12_pass(
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    roots = {
        label: _read_regular(
            path,
            expected_sha256=digest,
            expected_mode=0o444,
        )[1]
        for label, (path, digest) in R12_FILE_BINDINGS.items()
    }
    release, _source = _load_verified_module(
        RELEASE_PATH,
        expected_sha256=FAILED_SOURCE_BINDINGS["release_L"][1],
        name="tools._cure_lite_v24_mode_failure_release_L",
    )
    try:
        closure = release.compat_c1.legacy._validate_integration_chain(
            authorization_path=R12_AUTHORIZATION_PATH,
            receipt_path=R12_RECEIPT_PATH,
        )
        release._require_final_r12_supervisor_binding(
            {"integration": closure}
        )
    finally:
        sys.modules.pop(release.__name__, None)
    authorization = closure.get("authorization")
    receipt = closure.get("receipt")
    identities = closure.get("identities")
    if (
        not isinstance(authorization, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(identities, Mapping)
        or authorization.get("scenario_id") != R12_SCENARIO_ID
        or authorization.get("actual_r2_authorized") is not False
        or authorization.get("payload_authority") != "none"
        or receipt.get("passed") is not True
        or receipt.get("fragment_removed") is not True
        or receipt.get("gpu_accessed") is not False
        or receipt.get("payload_authority") != "none"
        or any(
            authorization.get(field) is not False
            or receipt.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError("r12 is not the exact payload-free PASS")
    return {
        "integration_root": str(R12_ROOT.absolute()),
        "scenario_id": R12_SCENARIO_ID,
        "authorization_root": roots["authorization"],
        "integration_terminal_root": roots["integration_terminal"],
        "receipt_root": roots["receipt"],
        "removal_authorization_root": roots["removal_authorization"],
        "removal_state_root": roots["removal_state"],
        "validated_identity_roots": dict(identities),
        "passed": True,
        "fragment_removed": True,
        "payload_authority": "none",
        "gpu_accessed": False,
    }, roots


def _reproduce_failure() -> dict[str, object]:
    realizer, _source = _load_verified_module(
        REALIZER_PATH,
        expected_sha256=FAILED_SOURCE_BINDINGS["realizer_R"][1],
        name="tools._cure_lite_v24_mode_failure_realizer_R",
    )
    try:
        try:
            realizer._validate_c2_bridge_authorization(
                require_fresh=False,
                require_future_absence=True,
            )
        except PermissionError as error:
            if type(error) is not PermissionError or error.args != (
                EXPECTED_ERROR,
            ):
                raise PermissionError(
                    "c2 mode-contract failure did not reproduce exactly"
                ) from error
        else:
            raise PermissionError(
                "c2 mode-contract failure no longer reproduces"
            )
    finally:
        sys.modules.pop(realizer.__name__, None)
    return {
        "validator": "_validate_c2_bridge_authorization",
        "require_fresh": False,
        "require_future_absence": True,
        "observation_kind": "read_only_post_hoc_reproduction",
        "write_capable_entrypoint_invoked": False,
        "exception_type": "PermissionError",
        "exception_message": EXPECTED_ERROR,
        "exception_args": [EXPECTED_ERROR],
        "reproduced": True,
    }


def _exact_absent_paths() -> dict[str, str]:
    return {
        label: str(path.absolute())
        for label, path in ABSENT_OUTPUT_PATHS.items()
    }


def _collect_historical_absence_observation(
    bridge: ModuleType,
    *,
    observed_at: datetime,
) -> dict[str, object]:
    before = {
        label: str(path)
        for label, path in ABSENT_OUTPUT_PATHS.items()
        if os.path.lexists(path)
    }
    state = dict(bridge._default_unit_state_reader(C2_UNIT_NAME))
    after = {
        label: str(path)
        for label, path in ABSENT_OUTPUT_PATHS.items()
        if os.path.lexists(path)
    }
    if before or after:
        raise PermissionError(
            f"c2 post-authorization output exists: {before or after}"
        )
    if (
        state.get("Id") != C2_UNIT_NAME
        or state.get("LoadState") != "not-found"
        or state.get("ActiveState") != "inactive"
        or state.get("SubState") != "dead"
        or state.get("UnitFileState") not in ("", None)
        or state.get("NRestarts") != "0"
        or state.get("FragmentPath") not in ("", None)
        or state.get("InvocationID") not in ("", None)
        or state.get("Restart") != "no"
    ):
        raise PermissionError("c2 unit is not exact not-found/inert")
    return {
        "observed_at_utc": _format_utc(observed_at),
        "exact_absent_paths": _exact_absent_paths(),
        "all_required_paths_absent": True,
        "c2_unit_state": state,
        "historical_observation_only": True,
        "future_state_authority": False,
        "archival_live_absence_recheck_required": False,
    }


def _write_create_once(
    path: Path,
    body: Mapping[str, object],
) -> dict[str, object]:
    target = Path(path).absolute()
    payload = dict(body)
    if "terminal_fingerprint" in payload:
        raise ValueError("terminal fingerprint must not be pre-populated")
    payload["terminal_fingerprint"] = stable_fingerprint(payload)
    raw = _canonical_bytes(payload) + b"\n"

    parent = target.parent
    parent_before = parent.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or parent.resolve(strict=True) != parent
        or parent_before.st_uid != os.getuid()
    ):
        raise PermissionError("unsafe failure-terminal parent")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        parent_opened = os.fstat(parent_fd)
        if (
            parent_opened.st_dev,
            parent_opened.st_ino,
        ) != (
            parent_before.st_dev,
            parent_before.st_ino,
        ):
            raise PermissionError("failure-terminal parent changed")
        descriptor = os.open(
            target.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short failure-terminal write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)

        opened = os.fstat(descriptor)
        linked = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        readback = os.pread(descriptor, len(raw) + 1, 0)
        finished = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_identity(opened) != _stat_identity(linked)
            or _stat_identity(opened) != _stat_identity(finished)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o444
            or opened.st_size != len(raw)
            or readback != raw
        ):
            raise PermissionError(
                "failure-terminal fd seal/readback changed"
            )
        os.fsync(parent_fd)
        parent_finished = os.fstat(parent_fd)
        if (
            parent_finished.st_dev,
            parent_finished.st_ino,
        ) != (
            parent_before.st_dev,
            parent_before.st_ino,
        ):
            raise PermissionError(
                "failure-terminal parent changed after create"
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    validated, _root = validate_archival(target)
    if validated != payload:
        raise RuntimeError("failure-terminal archival readback changed")
    return validated


def create_terminal(path: Path | None = None) -> dict[str, object]:
    selected = TERMINAL_PATH if path is None else Path(path).absolute()
    if selected != TERMINAL_PATH.absolute():
        raise PermissionError("mode-contract terminal path is not fixed")
    if os.path.lexists(selected):
        raise FileExistsError("mode-contract terminal already exists")

    failed_roots = _fixed_generation_roots()
    authorization, authorization_root = _validate_failed_authorization()
    r12, _r12_roots = _validate_r12_pass()
    reproduction = _reproduce_failure()

    # The read-only reproduction must not mutate its inputs.
    post_authorization, post_authorization_root = (
        _validate_failed_authorization()
    )
    if (
        post_authorization != authorization
        or post_authorization_root != authorization_root
        or _fixed_generation_roots() != failed_roots
    ):
        raise PermissionError(
            "read-only failure reproduction changed its inputs"
        )

    bridge, _bridge_source = _load_verified_module(
        BRIDGE_PATH,
        expected_sha256=FAILED_SOURCE_BINDINGS["bridge_B"][1],
        name="tools._cure_lite_v24_mode_failure_bridge_B",
    )
    try:
        observed_at = _utc_now()
        absence = _collect_historical_absence_observation(
            bridge,
            observed_at=observed_at,
        )
    finally:
        sys.modules.pop(bridge.__name__, None)

    body: dict[str, object] = {
        "schema_version": SCHEMA,
        "identity": {
            "candidate": CANDIDATE,
            "stage_id": STAGE_ID,
            "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
            "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
            "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
            "sealed_at_utc": _format_utc(observed_at),
        },
        "terminalizer_source_root": _read_regular(
            Path(__file__).resolve()
        )[1],
        "failed_generation_roots": failed_roots,
        "failed_authorization": {
            "root": authorization_root,
            "schema_version": authorization["schema_version"],
            "authorization_fingerprint": (
                authorization["authorization_fingerprint"]
            ),
            "created_at_utc": authorization["created_at_utc"],
            "issued_at_utc": authorization["issued_at_utc"],
            "expires_at_utc": authorization["expires_at_utc"],
            "observed_mode": authorization_root["mode"],
        },
        "r12_pass_closure": r12,
        "mode_contract_failure": {
            "producer": (
                "cure_lite_v24_preaccess_schema_compatibility_c2."
                "_write_sealed"
            ),
            "consumer": (
                "cure_lite_v24_actual_unit_realization_"
                "preaccess_compat_c2._validate_c2_bridge_authorization"
            ),
            "producer_observed_mode": FAILED_AUTHORIZATION_MODE,
            "consumer_required_mode": REQUIRED_AUTHORIZATION_MODE,
            "mode_contract_mismatch": True,
            "original_write_capable_entrypoint_invoked": True,
            "original_call_artifact_claimed": False,
            "original_failure_time_claimed": False,
            "failed_before_unit_authorization_write": True,
            "unit_authorization_written": False,
            "unit_terminal_written": False,
        },
        "deterministic_reproduction": reproduction,
        "historical_absence_observation": absence,
        "payload_observation": dict(_PAYLOAD_OBSERVATION),
        "continuation_policy": dict(_CONTINUATION_POLICY),
    }
    return _write_create_once(selected, body)


def _validate_historical_absence_record(value: object) -> None:
    if not isinstance(value, Mapping):
        raise PermissionError("historical absence record is malformed")
    state = value.get("c2_unit_state")
    if (
        value.get("exact_absent_paths") != _exact_absent_paths()
        or value.get("all_required_paths_absent") is not True
        or value.get("historical_observation_only") is not True
        or value.get("future_state_authority") is not False
        or value.get("archival_live_absence_recheck_required") is not False
        or not isinstance(value.get("observed_at_utc"), str)
        or not isinstance(state, Mapping)
        or state.get("Id") != C2_UNIT_NAME
        or state.get("LoadState") != "not-found"
        or state.get("ActiveState") != "inactive"
        or state.get("SubState") != "dead"
        or state.get("NRestarts") != "0"
        or state.get("FragmentPath") not in ("", None)
        or state.get("InvocationID") not in ("", None)
        or state.get("Restart") != "no"
    ):
        raise PermissionError("historical absence record changed")


def validate_archival(
    path: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    selected = TERMINAL_PATH if path is None else Path(path).absolute()
    if selected != TERMINAL_PATH.absolute():
        raise PermissionError("mode-contract terminal path is not fixed")
    raw, terminal_root = _read_regular(
        selected,
        expected_mode=0o444,
    )
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise PermissionError("mode-contract terminal layout changed")
    try:
        payload = json.loads(raw[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(
            "mode-contract terminal is not JSON"
        ) from error
    if (
        not isinstance(payload, dict)
        or set(payload) != _TOP_LEVEL_KEYS
        or payload.get("schema_version") != SCHEMA
        or raw != _canonical_bytes(payload) + b"\n"
    ):
        raise PermissionError(
            "mode-contract terminal schema/layout changed"
        )
    fingerprint = payload.get("terminal_fingerprint")
    body = dict(payload)
    body.pop("terminal_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or fingerprint != stable_fingerprint(body)
    ):
        raise PermissionError(
            "mode-contract terminal fingerprint changed"
        )

    identity = payload.get("identity")
    failure = payload.get("mode_contract_failure")
    reproduction = payload.get("deterministic_reproduction")
    if (
        not isinstance(identity, Mapping)
        or identity.get("candidate") != CANDIDATE
        or identity.get("stage_id") != STAGE_ID
        or identity.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or identity.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or identity.get("runtime_compatibility_id") != "c2"
        or not isinstance(identity.get("sealed_at_utc"), str)
        or not isinstance(failure, Mapping)
        or failure.get("producer_observed_mode")
        != FAILED_AUTHORIZATION_MODE
        or failure.get("consumer_required_mode")
        != REQUIRED_AUTHORIZATION_MODE
        or failure.get("mode_contract_mismatch") is not True
        or failure.get("original_write_capable_entrypoint_invoked")
        is not True
        or failure.get("original_call_artifact_claimed") is not False
        or failure.get("original_failure_time_claimed") is not False
        or failure.get("failed_before_unit_authorization_write")
        is not True
        or failure.get("unit_authorization_written") is not False
        or failure.get("unit_terminal_written") is not False
        or not isinstance(reproduction, Mapping)
        or reproduction.get("validator")
        != "_validate_c2_bridge_authorization"
        or reproduction.get("require_fresh") is not False
        or reproduction.get("require_future_absence") is not True
        or reproduction.get("write_capable_entrypoint_invoked")
        is not False
        or reproduction.get("exception_type") != "PermissionError"
        or reproduction.get("exception_message") != EXPECTED_ERROR
        or reproduction.get("exception_args") != [EXPECTED_ERROR]
        or reproduction.get("reproduced") is not True
        or payload.get("payload_observation") != _PAYLOAD_OBSERVATION
        or payload.get("continuation_policy") != _CONTINUATION_POLICY
    ):
        raise PermissionError(
            "mode-contract terminal semantics changed"
        )
    _validate_historical_absence_record(
        payload.get("historical_absence_observation")
    )

    terminalizer_root = _read_regular(Path(__file__).resolve())[1]
    if payload.get("terminalizer_source_root") != terminalizer_root:
        raise PermissionError("terminalizer source lineage changed")
    failed_roots = _fixed_generation_roots()
    if payload.get("failed_generation_roots") != failed_roots:
        raise PermissionError("failed source-generation lineage changed")

    authorization, authorization_root = _validate_failed_authorization()
    stored_authorization = payload.get("failed_authorization")
    if (
        not isinstance(stored_authorization, Mapping)
        or stored_authorization.get("root") != authorization_root
        or stored_authorization.get("schema_version")
        != authorization.get("schema_version")
        or stored_authorization.get("authorization_fingerprint")
        != FAILED_AUTHORIZATION_FINGERPRINT
        or stored_authorization.get("created_at_utc")
        != FAILED_AUTHORIZATION_CREATED_AT
        or stored_authorization.get("issued_at_utc")
        != FAILED_AUTHORIZATION_ISSUED_AT
        or stored_authorization.get("expires_at_utc")
        != FAILED_AUTHORIZATION_EXPIRES_AT
        or stored_authorization.get("observed_mode")
        != FAILED_AUTHORIZATION_MODE
    ):
        raise PermissionError("failed authorization lineage changed")

    r12, _r12_roots = _validate_r12_pass()
    if payload.get("r12_pass_closure") != r12:
        raise PermissionError("r12 PASS lineage changed")

    # T is the fixed terminalizer source root stored in F.  No live absence
    # check occurs above or below this point.
    terminal_root["terminal_fingerprint"] = fingerprint
    terminal_root["schema_version"] = SCHEMA
    terminal_root["terminalizer_source_root"] = terminalizer_root
    return payload, terminal_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("create-terminal")
    subparsers.add_parser("validate-terminal")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create-terminal":
        create_terminal()
    else:
        validate_archival()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
