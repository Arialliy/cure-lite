#!/usr/bin/env python3
"""Seal the single c3 environment-stability scope-handoff failure.

This terminalizer is forensic and metadata-only.  It binds the exact c3
source generation, the expired B3/R3 authorizations, the R3 PASS receipt,
the installed static c3 fragment, the sealed c3 environment policy, the
historical precleanup inventory, and the fixed cleanup receipt.  Its
deterministic reproduction compares only sealed JSON scope projections.  It
does not import or call the c3
environment gate, an inventory collector, a sleeper, a clock, or a writer.

No raw command/traceback artifact survived the original call.  Consequently
the terminal records only the known entrypoint and subcommand, and expressly
does not claim an exact original argv or traceback.

Creation records contemporaneous future-output absences and the inert c3
unit state.  Archival validation validates the sealed record and its fixed
input lineage, but does not re-check later absences or later manager state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()
RUNS_ROOT = (REPOSITORY / "runs/irstd1k_stage_a_seed42").resolve()

TERMINAL_PATH = (
    EVIDENCE_ROOT
    / (
        "r2_preaccess_schema_compat_c3_"
        "environment_stability_failure_terminal.json"
    )
)
SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c3-"
    "environment-stability-failure-terminal-v1"
)

CANDIDATE = "GCR-PACRE-v24"
STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
SCIENTIFIC_ATTEMPT_ID = "gcr_pacre_v24_D_R_zero_update_structural_r2"
SCIENTIFIC_ATTEMPT_ORDINAL = 2
RUNTIME_COMPATIBILITY_ID = "c3"

OLD_TARGET_UNIT = "cure-lite-v24-gcr-pacre-dr-r2.service"
C3_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c3.service"
)
C3_UNIT_FRAGMENT_PATH = (
    Path(f"/run/user/{os.getuid()}/systemd/user") / C3_UNIT_NAME
)

EXPECTED_ERROR = "precleanup inventory unit scope changed"

BRIDGE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c3.py"
).resolve()
REALIZER_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c3.py"
).resolve()
ENVIRONMENT_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_environment_preaccess_compat_c3.py"
).resolve()
SUPERVISOR_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c3.py"
).resolve()
RELEASE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_runtime_release_preaccess_compat_c3.py"
).resolve()
ADAPTER_PATH = (
    REPOSITORY
    / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c3.py"
).resolve()
TEMPLATE_PATH = (
    REPOSITORY
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c3.service.template"
).resolve()
FROZEN_ENVIRONMENT_PATH = (
    REPOSITORY / "tools/cure_lite_v24_runtime_environment.py"
).resolve()

FAILED_SOURCE_BINDINGS: dict[str, tuple[Path, str]] = {
    "bridge_B3": (
        BRIDGE_PATH,
        "3bf4caabfce8fd302b74b59b021bb37ba839f3c11226fde9b347ed2e574badb7",
    ),
    "realizer_R3": (
        REALIZER_PATH,
        "cdbbe4355b29519d2b3da858732bc8531396a59f5a3f1cfacdb578323fe33de1",
    ),
    "environment_E3": (
        ENVIRONMENT_PATH,
        "7bf9e268ffbd11491fc3f5efdd7a89611492c4e58ef587c0d9752a46e4e85a7e",
    ),
    "supervisor_S3": (
        SUPERVISOR_PATH,
        "536df32b66d8de3891aad4b454c886ec6b93617bc6759db098b2a442a4209afd",
    ),
    "release_L3": (
        RELEASE_PATH,
        "01f7f3ff171942bf26cfeeaa1cec0888480daf37fbe601395fab5b5ca7b6579a",
    ),
    "adapter": (
        ADAPTER_PATH,
        "002feefc40f73bf6e3c1a12445fceceee9cb8508130f3635970d479cea627863",
    ),
    "template": (
        TEMPLATE_PATH,
        "a7b7e63dd39603cabac75e4341203b1d231365caa3d5fc8c39819aacc5350edf",
    ),
    "frozen_environment": (
        FROZEN_ENVIRONMENT_PATH,
        "a40465786ce3537346372df5991bb6788d44feddfd497ec83a1dc302fb8b2fea",
    ),
}

B3_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c3_authorization.json"
)
B3_AUTHORIZATION_SHA256 = (
    "55b7bcb79bbbe5dc35719474d049874b2a0186728cb52c028e4458f015753afe"
)
B3_AUTHORIZATION_FINGERPRINT = (
    "64e535d14f0fa1210a10ae3e0366751001c3182889e6ccbb6502b81743593703"
)
B3_AUTHORIZATION_EXPIRES_AT = "2026-07-31T09:21:42.106021Z"
B3_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c3_receipt.json"
)

R3_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c3_unit_realization_authorization.json"
)
R3_AUTHORIZATION_SHA256 = (
    "acbf200597eaab497ec6114cd30184e4eb280cb32ebf2d1fa839cc30a2d1a02c"
)
R3_AUTHORIZATION_FINGERPRINT = (
    "e68b5d4170154101f0810cfa3cb1e4e89ef16581436c4e5fea553d1906bfecc2"
)
R3_AUTHORIZATION_EXPIRES_AT = "2026-07-31T09:21:55.214198Z"

R3_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c3_unit_realization_receipt.json"
)
R3_RECEIPT_SHA256 = (
    "25025b26ec227ba48d851ac063e901284e1f227367fad273da84a35b5d8047fc"
)
R3_RECEIPT_FINGERPRINT = (
    "6c02e6cd51b31a97a34f6014606e8ad8bf8970873bfac4f7e30820a3bcc409e1"
)
R3_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c3_unit_realization_terminal.json"
)

C3_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c3.json"
)
C3_POLICY_SHA256 = (
    "babb3f37f659ce5c885c1bba4d6c95bbbcfecd7726171d6bf9976548e13dc744"
)
C3_POLICY_FINGERPRINT = (
    "2ce59433e79deb8134befdc88d7556c70b057145c8222c9097b987198c58b63e"
)

PRECLEANUP_PATH = (
    EVIDENCE_ROOT / "runtime_environment_precleanup_receipt.json"
)
PRECLEANUP_SHA256 = (
    "57e2310163c28b43419cb2eb5f141e354b83135f50d57a052046c528801ca744"
)
PRECLEANUP_RECEIPT_FINGERPRINT = (
    "446f93e30f219dfd0f6a2d7a8d51b2a787b7b9c7956f462412cdae839cbb2582"
)
PRECLEANUP_INVENTORY_FINGERPRINT = (
    "242ad174c21be4d37a93e197c38059a9ba06eb61cd112abe1f641572a8e0a1f3"
)

CLEANUP_RECEIPT_PATH = (
    EVIDENCE_ROOT
    / "environment_cleanup_recovery_r1/cleanup-receipt.json"
)
CLEANUP_RECEIPT_SHA256 = (
    "511090ba1da235ff5383970ad7ec8ae456c030386700059eec029828b6edb762"
)
CLEANUP_RECEIPT_FINGERPRINT = (
    "b2a630de1afb5b239e410a97240b99a0f0b310ff180e193461d29d4cb2ca58e5"
)

C3_FRAGMENT_SHA256 = (
    "855a8e061c548cf2559cd94bd9d3271573a2f04c14d035d3040b77102ac072f0"
)

C3_STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c3.json"
)
C3_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c3.json"
)
C3_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_runtime_spec.json"
)
C3_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c3_"
        "runtime_launch_authorization.json"
    )
)
C3_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_runtime_artifacts"
)
C3_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_gpu_lease"
)
C3_RUN_ROOT_ALIAS_PATH = (
    RUNS_ROOT
    / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c3"
)
C3_RESULT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_receipt.json"
)
SCIENTIFIC_RUN_ROOT = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2"
)
SCIENTIFIC_RESULT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_receipt.json"
)

ABSENT_OUTPUT_PATHS: dict[str, Path] = {
    "B3_compatibility_receipt": B3_RECEIPT_PATH,
    "R3_unit_terminal": R3_TERMINAL_PATH,
    "C3_environment_stability": C3_STABILITY_PATH,
    "C3_environment_postcleanup": C3_POSTCLEANUP_PATH,
    "C3_runtime_spec": C3_RUNTIME_SPEC_PATH,
    "C3_runtime_launch_authorization": (
        C3_RUNTIME_LAUNCH_AUTHORIZATION_PATH
    ),
    "C3_runtime_artifacts": C3_RUNTIME_ARTIFACT_ROOT,
    "C3_gpu_lease": C3_GPU_LEASE_ROOT,
    "C3_run_alias": C3_RUN_ROOT_ALIAS_PATH,
    "C3_result_alias": C3_RESULT_ALIAS_PATH,
    "scientific_run_root": SCIENTIFIC_RUN_ROOT,
    "scientific_result_receipt": SCIENTIFIC_RESULT_RECEIPT_PATH,
}

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "identity",
        "terminalizer_source_root",
        "failed_generation_roots",
        "sealed_input_roots",
        "authorization_expiry",
        "unit_realization_closure",
        "environment_stability_failure",
        "deterministic_reproduction",
        "historical_state_observation",
        "payload_observation",
        "continuation_policy",
        "terminal_fingerprint",
    }
)

_CONTINUATION_POLICY = {
    "automatic_retry": False,
    "same_c3_reentry": False,
    "same_c3_reauthorization_allowed": False,
    "same_c3_metadata_repair_allowed": False,
    "c3_environment_gate_reentry_allowed": False,
    "c3_environment_gate_repair_allowed": False,
    "c4_required": True,
    "new_explicit_authorization_required": True,
    "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
    "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
    "scientific_attempt_consumed": False,
    "c3_authorization_consumed": True,
    "unit_realization_consumed": True,
    "environment_metadata_attempt_consumed": True,
    "runtime_launch_consumed": False,
    "materialization_consumed": False,
}

_UNIT_REALIZATION_CLOSURE = {
    "R3_receipt_passed": True,
    "static": True,
    "enabled": False,
    "started": False,
    "removed": False,
    "payload_authority": "none",
    "fragment_sha256": C3_FRAGMENT_SHA256,
    "unit_name": C3_UNIT_NAME,
}

_FAILURE_CONTROL_FLOW = {
    "E3_calls_frozen_run_environment_stability_gate": True,
    "frozen_gate_prepares_contract_before_sampling": True,
    "scope_validator_raises_before_collector": True,
    "collector_reached": False,
    "sleep_reached": False,
    "monotonic_clock_reached": False,
    "writer_reached": False,
}

_PAYLOAD_OBSERVATION = {
    "D_R_payload_accessed": False,
    "D_V_payload_accessed": False,
    "D_T_payload_accessed": False,
    "gpu_accessed": False,
    "training_started": False,
    "samples_processed": 0,
    "optimizer_steps": 0,
    "parameter_updates": 0,
    "zero_step_basis": (
        "B3/R3/policy grant no payload authority, the static c3 unit "
        "has no invocation, and runtime/scientific outputs were absent "
        "at terminal observation"
    ),
}

_STATE_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "FragmentPath",
    "InvocationID",
    "Restart",
    "NRestarts",
)

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


def _canonical_bytes(
    value: Mapping[str, object],
    *,
    ensure_ascii: bool = False,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    ).encode("utf-8")


def stable_fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _deep_exact_equal(left: object, right: object) -> bool:
    """Compare JSON-shaped values without bool/int coercion."""
    try:
        left_raw = json.dumps(
            left,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        right_raw = json.dumps(
            right,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return False
    return left_raw == right_raw


def _fingerprint(
    value: Mapping[str, object],
    *,
    ensure_ascii: bool,
) -> str:
    return hashlib.sha256(
        _canonical_bytes(value, ensure_ascii=ensure_ascii)
    ).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _parse_utc(value: object, *, name: str) -> datetime:
    if (
        not isinstance(value, str)
        or not value.endswith("Z")
        or value.count("Z") != 1
    ):
        raise PermissionError(f"{name} is not exact UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise PermissionError(f"{name} is malformed") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(
        parsed
    ):
        raise PermissionError(f"{name} is not UTC")
    return parsed


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
        opened_parent = os.fstat(parent_fd)
        if (
            opened_parent.st_dev,
            opened_parent.st_ino,
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
        parent_linked_after = parent.lstat()
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
            or not stat.S_ISDIR(parent_linked_after.st_mode)
            or stat.S_ISLNK(parent_linked_after.st_mode)
            or (
                parent_linked_after.st_dev,
                parent_linked_after.st_ino,
            )
            != (
                parent_before.st_dev,
                parent_before.st_ino,
            )
            or parent.resolve(strict=True) != parent
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


def _load_canonical_json(
    path: Path,
    *,
    expected_sha256: str,
    fingerprint_field: str,
    expected_fingerprint: str,
    ensure_ascii: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    raw, root = _read_regular(
        path,
        expected_sha256=expected_sha256,
        expected_mode=0o444,
    )
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise PermissionError(f"sealed JSON layout changed: {path}")
    try:
        value = json.loads(raw[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"sealed JSON is malformed: {path}") from error
    if (
        not isinstance(value, dict)
        or raw
        != _canonical_bytes(value, ensure_ascii=ensure_ascii) + b"\n"
    ):
        raise PermissionError(f"sealed JSON is not canonical: {path}")
    body = dict(value)
    observed_fingerprint = body.pop(fingerprint_field, None)
    if (
        observed_fingerprint != expected_fingerprint
        or _fingerprint(body, ensure_ascii=ensure_ascii)
        != expected_fingerprint
    ):
        raise PermissionError(f"sealed JSON fingerprint changed: {path}")
    return value, root


def _fixed_generation_roots() -> dict[str, dict[str, object]]:
    return {
        label: _read_regular(path, expected_sha256=digest)[1]
        for label, (path, digest) in FAILED_SOURCE_BINDINGS.items()
    }


def _validate_b3_authorization(
) -> tuple[dict[str, object], dict[str, object]]:
    value, root = _load_canonical_json(
        B3_AUTHORIZATION_PATH,
        expected_sha256=B3_AUTHORIZATION_SHA256,
        fingerprint_field="authorization_fingerprint",
        expected_fingerprint=B3_AUTHORIZATION_FINGERPRINT,
        ensure_ascii=True,
    )
    mutation = value.get("mutation_authority")
    scientific = value.get("scientific_authority")
    if (
        value.get("schema_version")
        != "cure-lite-v24-r2-preaccess-schema-compat-c3-authorization-v1"
        or value.get("candidate") != CANDIDATE
        or value.get("stage_id") != STAGE_ID
        or value.get("scientific_attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or value.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or value.get("runtime_compatibility_id")
        != RUNTIME_COMPATIBILITY_ID
        or value.get("authorized_uid") != os.getuid()
        or value.get("expires_at_utc") != B3_AUTHORIZATION_EXPIRES_AT
        or value.get("materialization_consumed") is not False
        or value.get("D_R_payload_accessed") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or value.get("gpu_accessed") is not False
        or value.get("training_started") is not False
        or not isinstance(mutation, Mapping)
        or mutation.get("c3_unit_realization_authorized") is not True
        or mutation.get("environment_metadata_audit_authorized")
        is not True
        or mutation.get("environment_scope_handoff_authorized")
        is not True
        or mutation.get("payload_access_authorized") is not False
        or mutation.get("runtime_spec_creation_authorized") is not False
        or mutation.get(
            "runtime_launch_authorization_creation_authorized"
        )
        is not False
        or mutation.get("unit_start_authorized") is not False
        or mutation.get("unit_enable_authorized") is not False
        or not isinstance(scientific, Mapping)
        or scientific.get("automatic_retry") is not False
        or scientific.get("resume") is not False
        or scientific.get("materialization_authorized") is not False
        or scientific.get("training_authorized") is not False
        or scientific.get("D_R_payload_authorized") is not False
        or scientific.get("D_V_payload_authorized") is not False
        or scientific.get("D_T_payload_authorized") is not False
    ):
        raise PermissionError("B3 authorization identity/authority changed")
    return value, root


def _validate_r3_authorization(
) -> tuple[dict[str, object], dict[str, object]]:
    value, root = _load_canonical_json(
        R3_AUTHORIZATION_PATH,
        expected_sha256=R3_AUTHORIZATION_SHA256,
        fingerprint_field="authorization_fingerprint",
        expected_fingerprint=R3_AUTHORIZATION_FINGERPRINT,
        ensure_ascii=False,
    )
    closure = value.get("compatibility_closure")
    bridge_window = (
        closure.get("bridge_authorization_window")
        if isinstance(closure, Mapping)
        else None
    )
    bridge_root = (
        closure.get("bridge_compat_authorization_root")
        if isinstance(closure, Mapping)
        else None
    )
    rendered = value.get("rendered_fragment")
    if (
        value.get("schema_version")
        != "cure-lite-v24-actual-unit-realization-authorization-v1"
        or value.get("candidate") != CANDIDATE
        or value.get("stage_id") != STAGE_ID
        or value.get("attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or value.get("unit_name") != C3_UNIT_NAME
        or value.get("authorized_uid") != os.getuid()
        or value.get("expires_at_utc") != R3_AUTHORIZATION_EXPIRES_AT
        or value.get("payload_authority") != "none"
        or value.get("start_authorized") is not False
        or value.get("enable_authorized") is not False
        or value.get("persistent_install_authorized") is not False
        or value.get("remove_authorized") is not False
        or value.get("D_R_payload_accessed") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or value.get("actions")
        != [
            "install-runtime-static-fragment",
            "daemon-reload",
            "verify-static-shadow",
        ]
        or not isinstance(rendered, Mapping)
        or rendered.get("sha256") != C3_FRAGMENT_SHA256
        or not isinstance(closure, Mapping)
        or closure.get("schema_version")
        != "cure-lite-v24-preaccess-compat-c3-unit-realization-closure-v1"
        or closure.get("runtime_compatibility_generation") != "c3"
        or closure.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or closure.get("payload_authority") != "none"
        or closure.get("automatic_retry_authorized") is not False
        or closure.get("resume_authorized") is not False
        or not isinstance(bridge_window, Mapping)
        or bridge_window.get("file_sha256") != B3_AUTHORIZATION_SHA256
        or bridge_window.get("authorization_fingerprint")
        != B3_AUTHORIZATION_FINGERPRINT
        or not isinstance(bridge_root, Mapping)
        or bridge_root.get("file_sha256") != B3_AUTHORIZATION_SHA256
        or bridge_root.get("fingerprint")
        != B3_AUTHORIZATION_FINGERPRINT
    ):
        raise PermissionError("R3 authorization identity/authority changed")
    return value, root


def _validate_r3_receipt(
) -> tuple[dict[str, object], dict[str, object]]:
    value, root = _load_canonical_json(
        R3_RECEIPT_PATH,
        expected_sha256=R3_RECEIPT_SHA256,
        fingerprint_field="receipt_fingerprint",
        expected_fingerprint=R3_RECEIPT_FINGERPRINT,
        ensure_ascii=False,
    )
    fragment = value.get("fragment_identity")
    shadow = value.get("full_static_shadow")
    if (
        value.get("schema_version")
        != "cure-lite-v24-actual-unit-realization-receipt-v1"
        or value.get("candidate") != CANDIDATE
        or value.get("stage_id") != STAGE_ID
        or value.get("attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or value.get("unit_name") != C3_UNIT_NAME
        or value.get("authorization_file_sha256")
        != R3_AUTHORIZATION_SHA256
        or value.get("authorization_fingerprint")
        != R3_AUTHORIZATION_FINGERPRINT
        or value.get("passed") is not True
        or value.get("static") is not True
        or value.get("enabled") is not False
        or value.get("started") is not False
        or value.get("removed") is not False
        or value.get("payload_authority") != "none"
        or value.get("D_R_payload_accessed") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or value.get("runtime_spec_absent_at_receipt") is not True
        or value.get("completed_actions")
        != [
            "install-runtime-static-fragment",
            "daemon-reload",
            "verify-static-shadow",
        ]
        or not isinstance(fragment, Mapping)
        or fragment.get("path") != str(C3_UNIT_FRAGMENT_PATH)
        or fragment.get("file_sha256") != C3_FRAGMENT_SHA256
        or fragment.get("mode") != 0o600
        or fragment.get("owner_uid") != os.getuid()
        or fragment.get("nlink") != 1
        or not isinstance(shadow, Mapping)
        or shadow.get("Id") != C3_UNIT_NAME
        or shadow.get("LoadState") != "loaded"
        or shadow.get("ActiveState") != "inactive"
        or shadow.get("SubState") != "dead"
        or shadow.get("UnitFileState") != "static"
        or shadow.get("Restart") != "no"
        or shadow.get("NRestarts") != "0"
        or shadow.get("FragmentPath") != str(C3_UNIT_FRAGMENT_PATH)
    ):
        raise PermissionError("R3 PASS/static receipt changed")
    return value, root


def _validate_precleanup(
) -> tuple[dict[str, object], dict[str, object]]:
    value, root = _load_canonical_json(
        PRECLEANUP_PATH,
        expected_sha256=PRECLEANUP_SHA256,
        fingerprint_field="receipt_fingerprint",
        expected_fingerprint=PRECLEANUP_RECEIPT_FINGERPRINT,
        ensure_ascii=False,
    )
    inventory = value.get("inventory")
    if not isinstance(inventory, Mapping):
        raise PermissionError("historical precleanup inventory is missing")
    inventory_body = dict(inventory)
    inventory_fingerprint = inventory_body.pop(
        "inventory_fingerprint",
        None,
    )
    scope = inventory.get("unit_scope")
    if (
        inventory_fingerprint != PRECLEANUP_INVENTORY_FINGERPRINT
        or _fingerprint(inventory_body, ensure_ascii=False)
        != PRECLEANUP_INVENTORY_FINGERPRINT
        or value.get("schema_version")
        != "cure-lite-v24-runtime-environment-audit-receipt-v1"
        or inventory.get("schema_version")
        != "cure-lite-v24-runtime-environment-inventory-v1"
        or not isinstance(scope, Mapping)
        or scope.get("target_unit_id") != OLD_TARGET_UNIT
        or scope.get("conflict_unit_ids")
        != ["confa-v41-mshnet-nudt-clean-formal-20260718-v1.service"]
        or scope.get("dependency_unit_ids") != []
        or scope.get("require_target_ready") is not False
        or value.get("D_R_payload_accessed") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or inventory.get("D_R_payload_accessed") is not False
        or inventory.get("D_V_payload_accessed") is not False
        or inventory.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("historical precleanup lineage/scope changed")
    return value, root


def _validate_cleanup_receipt(
) -> tuple[dict[str, object], dict[str, object]]:
    value, root = _load_canonical_json(
        CLEANUP_RECEIPT_PATH,
        expected_sha256=CLEANUP_RECEIPT_SHA256,
        fingerprint_field="cleanup_receipt_fingerprint",
        expected_fingerprint=CLEANUP_RECEIPT_FINGERPRINT,
        ensure_ascii=False,
    )
    guard = value.get("activation_guard")
    partial = value.get("partial_lineage")
    if (
        value.get("schema_version")
        != "cure-lite-v24-runtime-cleanup-receipt-v2"
        or value.get("passed") is not True
        or value.get("cleanup_mode")
        != "partial-runtime-mask-stop-recovery"
        or value.get("payload_authority") != "none"
        or value.get("D_R_payload_accessed") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or not isinstance(guard, Mapping)
        or guard.get("unit_name")
        != "confa-v41-mshnet-nudt-clean-formal-20260718-v1.service"
        or guard.get("path")
        != (
            "/run/user/1008/systemd/user/"
            "confa-v41-mshnet-nudt-clean-formal-20260718-v1.service"
        )
        or guard.get("target") != "/dev/null"
        or guard.get("mode")
        != "ineffective-runtime-mask-symlink-plus-explicit-stop"
        or guard.get("observed_unit_file_state") != "enabled"
        or not isinstance(partial, Mapping)
        or partial.get("original_stop_dispatched") is not False
        or partial.get("legacy_runtime_mask_may_remain_false_reconciled")
        is not True
        or value.get("action_receipt_fingerprints")
        != [
            "cf94f733b23d14b661593ff1c5b39d761a135f4ad90ca1ef2d87db443cf535ca"
        ]
        or value.get("intent_fingerprint")
        != "a6f440e63c593e019302801efeaf8a48fb05620d27176fbc1bec986d2f80f003"
    ):
        raise PermissionError("fixed cleanup receipt lineage changed")
    return value, root


def _validate_policy(
) -> tuple[dict[str, object], dict[str, object]]:
    value, root = _load_canonical_json(
        C3_POLICY_PATH,
        expected_sha256=C3_POLICY_SHA256,
        fingerprint_field="policy_fingerprint",
        expected_fingerprint=C3_POLICY_FINGERPRINT,
        ensure_ascii=False,
    )
    scope = value.get("unit_scope")
    precleanup = value.get("precleanup_root")
    cleanup = value.get("cleanup_root")
    sampling = value.get("sampling")
    if (
        value.get("schema_version")
        != "cure-lite-v24-runtime-environment-policy-v1"
        or value.get("candidate") != CANDIDATE
        or value.get("scope") != "runtime-environment-stability"
        or value.get("uid") != os.getuid()
        or value.get("payload_authority") != "none"
        or value.get("D_R_payload_accessed") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or not isinstance(scope, Mapping)
        or scope.get("target_unit_id") != C3_UNIT_NAME
        or scope.get("conflict_unit_ids")
        != ["confa-v41-mshnet-nudt-clean-formal-20260718-v1.service"]
        or scope.get("dependency_unit_ids") != []
        or scope.get("require_target_ready") is not True
        or not isinstance(precleanup, Mapping)
        or precleanup.get("path") != str(PRECLEANUP_PATH)
        or precleanup.get("file_sha256") != PRECLEANUP_SHA256
        or precleanup.get("receipt_fingerprint")
        != PRECLEANUP_RECEIPT_FINGERPRINT
        or precleanup.get("inventory_fingerprint")
        != PRECLEANUP_INVENTORY_FINGERPRINT
        or not isinstance(cleanup, Mapping)
        or cleanup.get("path") != str(CLEANUP_RECEIPT_PATH)
        or cleanup.get("file_sha256") != CLEANUP_RECEIPT_SHA256
        or cleanup.get("cleanup_receipt_fingerprint")
        != CLEANUP_RECEIPT_FINGERPRINT
        or not isinstance(sampling, Mapping)
        or sampling.get("minimum_sample_count") != 2
        or sampling.get("sample_interval_seconds") != 30.0
        or sampling.get("minimum_window_seconds") != 30.0
    ):
        raise PermissionError("C3 environment policy scope changed")
    return value, root


def _validate_fixed_inputs(
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    b3, b3_root = _validate_b3_authorization()
    r3_authorization, r3_authorization_root = (
        _validate_r3_authorization()
    )
    r3_receipt, r3_receipt_root = _validate_r3_receipt()
    precleanup, precleanup_root = _validate_precleanup()
    cleanup, cleanup_root = _validate_cleanup_receipt()
    policy, policy_root = _validate_policy()
    fragment_raw, fragment_root = _read_regular(
        C3_UNIT_FRAGMENT_PATH,
        expected_sha256=C3_FRAGMENT_SHA256,
        expected_mode=0o600,
    )
    rendered = r3_authorization["rendered_fragment"]
    if (
        not isinstance(rendered, Mapping)
        or not isinstance(rendered.get("utf8_text"), str)
        or fragment_raw != rendered["utf8_text"].encode("utf-8")
        or r3_receipt["fragment_identity"]["file_sha256"]
        != fragment_root["file_sha256"]
    ):
        raise PermissionError("installed C3 fragment binding changed")
    values = {
        "B3_authorization": b3,
        "R3_authorization": r3_authorization,
        "R3_receipt": r3_receipt,
        "historical_precleanup": precleanup,
        "fixed_cleanup_receipt": cleanup,
        "C3_environment_policy": policy,
    }
    roots = {
        "B3_authorization": b3_root,
        "R3_authorization": r3_authorization_root,
        "R3_receipt": r3_receipt_root,
        "historical_precleanup": precleanup_root,
        "fixed_cleanup_receipt": cleanup_root,
        "C3_environment_policy": policy_root,
        "C3_unit_fragment": fragment_root,
    }
    fingerprints = {
        "B3_authorization": {
            "field": "authorization_fingerprint",
            "value": B3_AUTHORIZATION_FINGERPRINT,
        },
        "R3_authorization": {
            "field": "authorization_fingerprint",
            "value": R3_AUTHORIZATION_FINGERPRINT,
        },
        "R3_receipt": {
            "field": "receipt_fingerprint",
            "value": R3_RECEIPT_FINGERPRINT,
        },
        "historical_precleanup": {
            "field": "receipt_fingerprint",
            "value": PRECLEANUP_RECEIPT_FINGERPRINT,
            "inventory_fingerprint": PRECLEANUP_INVENTORY_FINGERPRINT,
        },
        "fixed_cleanup_receipt": {
            "field": "cleanup_receipt_fingerprint",
            "value": CLEANUP_RECEIPT_FINGERPRINT,
        },
        "C3_environment_policy": {
            "field": "policy_fingerprint",
            "value": C3_POLICY_FINGERPRINT,
        },
        "C3_unit_fragment": {
            "field": "file_sha256",
            "value": C3_FRAGMENT_SHA256,
        },
    }
    return values, roots, fingerprints


def _scope_projection(scope: Mapping[str, object]) -> dict[str, object]:
    return {
        "target_unit_id": scope.get("target_unit_id"),
        "conflict_unit_ids": scope.get("conflict_unit_ids"),
        "dependency_unit_ids": scope.get("dependency_unit_ids"),
        "require_target_ready": scope.get("require_target_ready"),
    }


def _raise_if_precleanup_scope_changed(
    observed_scope: Mapping[str, object],
    expected_scope: Mapping[str, object],
) -> None:
    if dict(observed_scope) != dict(expected_scope):
        raise PermissionError(EXPECTED_ERROR)


def _reproduce_scope_mismatch() -> dict[str, object]:
    """Reproduce only the sealed scope contradiction; perform no live audit."""
    precleanup, _precleanup_root = _validate_precleanup()
    _cleanup, cleanup_root = _validate_cleanup_receipt()
    policy, _policy_root = _validate_policy()
    inventory = precleanup["inventory"]
    if not isinstance(inventory, Mapping):
        raise PermissionError("historical precleanup inventory is missing")
    old_scope_value = inventory.get("unit_scope")
    c3_scope_value = policy.get("unit_scope")
    if (
        not isinstance(old_scope_value, Mapping)
        or not isinstance(c3_scope_value, Mapping)
    ):
        raise PermissionError("scope reproduction input changed")
    old_scope = _scope_projection(old_scope_value)
    c3_scope = _scope_projection(c3_scope_value)
    mismatch_fields = [
        field
        for field in old_scope
        if old_scope[field] != c3_scope[field]
    ]
    if mismatch_fields != ["target_unit_id", "require_target_ready"]:
        raise PermissionError("C3 scope mismatch is no longer exact")
    try:
        _raise_if_precleanup_scope_changed(old_scope, c3_scope)
    except PermissionError as error:
        if type(error) is not PermissionError or error.args != (
            EXPECTED_ERROR,
        ):
            raise PermissionError(
                "C3 scope failure did not reproduce exactly"
            ) from error
    else:
        raise PermissionError("C3 scope failure no longer reproduces")
    return {
        "observation_kind": (
            "post_hoc_deterministic_read_only_reproduction"
        ),
        "sealed_inputs_only": True,
        "sealed_cleanup_receipt_root": cleanup_root,
        "sealed_cleanup_receipt_fingerprint": (
            CLEANUP_RECEIPT_FINGERPRINT
        ),
        "observed_precleanup_scope": old_scope,
        "requested_C3_scope": c3_scope,
        "mismatch_fields": mismatch_fields,
        "validator_contract": (
            "environment_audit_contract_from_inventory scope projection"
        ),
        "frozen_validator_source_path": str(FROZEN_ENVIRONMENT_PATH),
        "frozen_validator_source_sha256": (
            FAILED_SOURCE_BINDINGS["frozen_environment"][1]
        ),
        "exception_type": "PermissionError",
        "exception_message": EXPECTED_ERROR,
        "exception_args": [EXPECTED_ERROR],
        "reproduced": True,
        "E3_module_loaded": False,
        "E3_gate_invoked": False,
        "inventory_collector_invoked": False,
        "activation_guard_reader_invoked": False,
        "sleeper_invoked": False,
        "monotonic_clock_invoked": False,
        "writer_invoked": False,
        "samples_collected": 0,
    }


def _require_authorizations_expired(
    values: Mapping[str, Mapping[str, object]],
    *,
    observed_at: datetime,
) -> dict[str, object]:
    b3_expiry = _parse_utc(
        values["B3_authorization"].get("expires_at_utc"),
        name="B3 authorization expiry",
    )
    r3_expiry = _parse_utc(
        values["R3_authorization"].get("expires_at_utc"),
        name="R3 authorization expiry",
    )
    if observed_at <= b3_expiry or observed_at <= r3_expiry:
        raise PermissionError(
            "B3/R3 authorizations are not both expired"
        )
    return {
        "observed_at_utc": _format_utc(observed_at),
        "B3_expires_at_utc": B3_AUTHORIZATION_EXPIRES_AT,
        "B3_expired": True,
        "B3_compatibility_receipt_absent_at_observation": True,
        "B3_sealed_by_compatibility_receipt": False,
        "R3_expires_at_utc": R3_AUTHORIZATION_EXPIRES_AT,
        "R3_expired": True,
        "R3_PASS_receipt_exists": True,
    }


def _read_c3_unit_state(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    command = [
        "/usr/bin/systemctl",
        "--user",
        "show",
        C3_UNIT_NAME,
        "--no-pager",
        "--property=" + ",".join(_STATE_PROPERTIES),
    ]
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if (
        completed.returncode != 0
        or not isinstance(completed.stdout, str)
        or not isinstance(completed.stderr, str)
        or completed.stderr != ""
    ):
        raise PermissionError("C3 unit state query failed")
    state: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" not in line:
            raise PermissionError("C3 unit state output changed")
        key, value = line.split("=", 1)
        if key in state:
            raise PermissionError("C3 unit state property repeated")
        state[key] = value
    if set(state) != set(_STATE_PROPERTIES):
        raise PermissionError("C3 unit state properties changed")
    if (
        state.get("Id") != C3_UNIT_NAME
        or state.get("LoadState") != "loaded"
        or state.get("ActiveState") != "inactive"
        or state.get("SubState") != "dead"
        or state.get("UnitFileState") != "static"
        or state.get("FragmentPath") != str(C3_UNIT_FRAGMENT_PATH)
        or state.get("InvocationID") != ""
        or state.get("Restart") != "no"
        or state.get("NRestarts") != "0"
    ):
        raise PermissionError("C3 unit is not exact static/inert")
    return state


def _exact_absent_paths() -> dict[str, str]:
    return {
        label: str(path.absolute())
        for label, path in ABSENT_OUTPUT_PATHS.items()
    }


def _existing_output_paths() -> dict[str, str]:
    return {
        label: str(path)
        for label, path in ABSENT_OUTPUT_PATHS.items()
        if os.path.lexists(path)
    }


def _collect_historical_state_observation(
    *,
    observed_at: datetime,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    before = _existing_output_paths()
    state = _read_c3_unit_state(runner=runner)
    after = _existing_output_paths()
    if before or after:
        raise PermissionError(
            f"C3 future output exists: {before or after}"
        )
    return {
        "observed_at_utc": _format_utc(observed_at),
        "exact_absent_paths": _exact_absent_paths(),
        "all_required_paths_absent": True,
        "B3_authorization_unsealed_basis": (
            "B3 compatibility receipt absent at observation"
        ),
        "C3_unit_state": state,
        "historical_observation_only": True,
        "future_state_authority": False,
        "archival_live_absence_recheck_required": False,
        "archival_live_manager_recheck_required": False,
    }


def _revalidate_open_creation_guard(
    *,
    expected_generation_roots: Mapping[str, object],
    expected_values: Mapping[str, object],
    expected_input_roots: Mapping[str, object],
    expected_input_fingerprints: Mapping[str, object],
    expected_terminalizer_source_root: Mapping[str, object],
    expected_unit_state: Mapping[str, str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    """Revalidate all mutable observations while the terminal fd is open."""
    before_absence = _existing_output_paths()
    if before_absence:
        raise PermissionError(
            f"C3 future output appeared while terminal open: {before_absence}"
        )

    generation_before = _fixed_generation_roots()
    source_before = _read_regular(Path(__file__).resolve())[1]
    values_before, roots_before, fingerprints_before = (
        _validate_fixed_inputs()
    )
    try:
        state_before = _read_c3_unit_state(runner=runner)
    except PermissionError as error:
        raise PermissionError(
            "C3 unit state drifted while terminal open"
        ) from error
    values_after, roots_after, fingerprints_after = (
        _validate_fixed_inputs()
    )
    source_after = _read_regular(Path(__file__).resolve())[1]
    generation_after = _fixed_generation_roots()
    try:
        state_after = _read_c3_unit_state(runner=runner)
    except PermissionError as error:
        raise PermissionError(
            "C3 unit state drifted while terminal open"
        ) from error
    after_absence = _existing_output_paths()

    if after_absence:
        raise PermissionError(
            f"C3 future output appeared while terminal open: {after_absence}"
        )
    if (
        not _deep_exact_equal(state_before, expected_unit_state)
        or not _deep_exact_equal(state_after, expected_unit_state)
    ):
        raise PermissionError("C3 unit state drifted while terminal open")
    expected_groups = (
        (generation_before, expected_generation_roots),
        (generation_after, expected_generation_roots),
        (source_before, expected_terminalizer_source_root),
        (source_after, expected_terminalizer_source_root),
        (values_before, expected_values),
        (values_after, expected_values),
        (roots_before, expected_input_roots),
        (roots_after, expected_input_roots),
        (fingerprints_before, expected_input_fingerprints),
        (fingerprints_after, expected_input_fingerprints),
    )
    if any(
        not _deep_exact_equal(observed, expected)
        for observed, expected in expected_groups
    ):
        raise PermissionError(
            "fixed C3 input/source drifted while terminal open"
        )


def _write_create_once(
    path: Path,
    body: Mapping[str, object],
    *,
    preseal_guard: Callable[[], None] | None = None,
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
        opened_parent = os.fstat(parent_fd)
        if (
            opened_parent.st_dev,
            opened_parent.st_ino,
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
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short failure-terminal write")
            offset += written
        os.fsync(descriptor)
        os.fsync(parent_fd)
        if preseal_guard is not None:
            preseal_guard()
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
        finished_parent = os.fstat(parent_fd)
        linked_parent = parent.lstat()
        if (
            (
                finished_parent.st_dev,
                finished_parent.st_ino,
            )
            != (
                parent_before.st_dev,
                parent_before.st_ino,
            )
            or not stat.S_ISDIR(linked_parent.st_mode)
            or stat.S_ISLNK(linked_parent.st_mode)
            or (
                linked_parent.st_dev,
                linked_parent.st_ino,
            )
            != (
                parent_before.st_dev,
                parent_before.st_ino,
            )
            or parent.resolve(strict=True) != parent
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


def create_terminal(
    path: Path | None = None,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    selected = TERMINAL_PATH if path is None else Path(path).absolute()
    if selected != TERMINAL_PATH.absolute():
        raise PermissionError(
            "environment-stability terminal path is not fixed"
        )
    if os.path.lexists(selected):
        raise FileExistsError(
            "environment-stability terminal already exists"
        )

    generation_roots = _fixed_generation_roots()
    values, input_roots, input_fingerprints = _validate_fixed_inputs()
    reproduction = _reproduce_scope_mismatch()
    post_values, post_roots, post_fingerprints = (
        _validate_fixed_inputs()
    )
    if (
        post_values != values
        or post_roots != input_roots
        or post_fingerprints != input_fingerprints
        or _fixed_generation_roots() != generation_roots
    ):
        raise PermissionError(
            "read-only scope reproduction changed its inputs"
        )

    observed_at = _utc_now()
    expiry = _require_authorizations_expired(
        values,
        observed_at=observed_at,
    )
    historical = _collect_historical_state_observation(
        observed_at=observed_at,
        runner=runner,
    )

    final_values, final_roots, final_fingerprints = (
        _validate_fixed_inputs()
    )
    if (
        final_values != values
        or final_roots != input_roots
        or final_fingerprints != input_fingerprints
        or _fixed_generation_roots() != generation_roots
    ):
        raise PermissionError(
            "fixed C3 evidence changed before terminal seal"
        )

    failure = {
        "known_entrypoint_path": str(ENVIRONMENT_PATH),
        "known_subcommand": "stability-gate",
        "known_precleanup_input_path": str(PRECLEANUP_PATH),
        "known_cleanup_input_path": str(CLEANUP_RECEIPT_PATH),
        "known_policy_input_path": str(C3_POLICY_PATH),
        "original_call_artifact_available": False,
        "original_argv_claimed": False,
        "original_traceback_artifact_available": False,
        "original_traceback_claimed": False,
        "original_failure_time_claimed": False,
        "attempt_count": 1,
        "retry": False,
        "samples_collected": 0,
        "expected_exception_type": "PermissionError",
        "expected_exception_message": EXPECTED_ERROR,
        "failure_observation": (
            "post_hoc_deterministic_read_only_reproduction"
        ),
        "frozen_control_flow": dict(_FAILURE_CONTROL_FLOW),
    }
    terminalizer_source_root = _read_regular(
        Path(__file__).resolve()
    )[1]
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
        "terminalizer_source_root": terminalizer_source_root,
        "failed_generation_roots": generation_roots,
        "sealed_input_roots": {
            label: {
                "root": input_roots[label],
                **input_fingerprints[label],
            }
            for label in input_roots
        },
        "authorization_expiry": expiry,
        "unit_realization_closure": dict(_UNIT_REALIZATION_CLOSURE),
        "environment_stability_failure": failure,
        "deterministic_reproduction": reproduction,
        "historical_state_observation": historical,
        "payload_observation": dict(_PAYLOAD_OBSERVATION),
        "continuation_policy": dict(_CONTINUATION_POLICY),
    }

    historical_state = historical.get("C3_unit_state")
    if not isinstance(historical_state, Mapping):
        raise PermissionError("historical C3 unit state disappeared")

    def preseal_guard() -> None:
        _revalidate_open_creation_guard(
            expected_generation_roots=generation_roots,
            expected_values=values,
            expected_input_roots=input_roots,
            expected_input_fingerprints=input_fingerprints,
            expected_terminalizer_source_root=terminalizer_source_root,
            expected_unit_state=dict(historical_state),
            runner=runner,
        )

    return _write_create_once(
        selected,
        body,
        preseal_guard=preseal_guard,
    )


def _validate_historical_state_record(
    value: object,
    *,
    sealed_at: datetime,
) -> None:
    if not isinstance(value, Mapping):
        raise PermissionError("historical C3 state record is malformed")
    state = value.get("C3_unit_state")
    if (
        set(value)
        != {
            "observed_at_utc",
            "exact_absent_paths",
            "all_required_paths_absent",
            "B3_authorization_unsealed_basis",
            "C3_unit_state",
            "historical_observation_only",
            "future_state_authority",
            "archival_live_absence_recheck_required",
            "archival_live_manager_recheck_required",
        }
        or value.get("exact_absent_paths") != _exact_absent_paths()
        or value.get("all_required_paths_absent") is not True
        or value.get("B3_authorization_unsealed_basis")
        != "B3 compatibility receipt absent at observation"
        or value.get("historical_observation_only") is not True
        or value.get("future_state_authority") is not False
        or value.get("archival_live_absence_recheck_required") is not False
        or value.get("archival_live_manager_recheck_required") is not False
        or _parse_utc(
            value.get("observed_at_utc"),
            name="historical state observation",
        )
        != sealed_at
        or not isinstance(state, Mapping)
        or set(state) != set(_STATE_PROPERTIES)
        or state.get("Id") != C3_UNIT_NAME
        or state.get("LoadState") != "loaded"
        or state.get("ActiveState") != "inactive"
        or state.get("SubState") != "dead"
        or state.get("UnitFileState") != "static"
        or state.get("FragmentPath") != str(C3_UNIT_FRAGMENT_PATH)
        or state.get("InvocationID") != ""
        or state.get("Restart") != "no"
        or state.get("NRestarts") != "0"
    ):
        raise PermissionError("historical C3 state record changed")


def _validate_expiry_record(
    value: object,
    *,
    sealed_at: datetime,
) -> None:
    if not isinstance(value, Mapping):
        raise PermissionError("authorization expiry record is malformed")
    b3_expiry = _parse_utc(
        value.get("B3_expires_at_utc"),
        name="stored B3 expiry",
    )
    r3_expiry = _parse_utc(
        value.get("R3_expires_at_utc"),
        name="stored R3 expiry",
    )
    if (
        set(value)
        != {
            "observed_at_utc",
            "B3_expires_at_utc",
            "B3_expired",
            "B3_compatibility_receipt_absent_at_observation",
            "B3_sealed_by_compatibility_receipt",
            "R3_expires_at_utc",
            "R3_expired",
            "R3_PASS_receipt_exists",
        }
        or value.get("B3_expires_at_utc") != B3_AUTHORIZATION_EXPIRES_AT
        or value.get("B3_expired") is not True
        or value.get(
            "B3_compatibility_receipt_absent_at_observation"
        )
        is not True
        or value.get("B3_sealed_by_compatibility_receipt") is not False
        or value.get("R3_expires_at_utc") != R3_AUTHORIZATION_EXPIRES_AT
        or value.get("R3_expired") is not True
        or value.get("R3_PASS_receipt_exists") is not True
        or not isinstance(value.get("observed_at_utc"), str)
        or _parse_utc(
            value.get("observed_at_utc"),
            name="expiry observation",
        )
        != sealed_at
        or sealed_at <= b3_expiry
        or sealed_at <= r3_expiry
    ):
        raise PermissionError("authorization expiry record changed")


def validate_archival(
    path: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    selected = TERMINAL_PATH if path is None else Path(path).absolute()
    if selected != TERMINAL_PATH.absolute():
        raise PermissionError(
            "environment-stability terminal path is not fixed"
        )
    raw, terminal_root = _read_regular(
        selected,
        expected_mode=0o444,
    )
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise PermissionError(
            "environment-stability terminal layout changed"
        )
    try:
        payload = json.loads(raw[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(
            "environment-stability terminal is not JSON"
        ) from error
    if (
        not isinstance(payload, dict)
        or set(payload) != _TOP_LEVEL_KEYS
        or payload.get("schema_version") != SCHEMA
        or raw != _canonical_bytes(payload) + b"\n"
    ):
        raise PermissionError(
            "environment-stability terminal schema/layout changed"
        )
    fingerprint = payload.get("terminal_fingerprint")
    body = dict(payload)
    body.pop("terminal_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or fingerprint != stable_fingerprint(body)
    ):
        raise PermissionError(
            "environment-stability terminal fingerprint changed"
        )

    identity = payload.get("identity")
    failure = payload.get("environment_stability_failure")
    reproduction = payload.get("deterministic_reproduction")
    closure = payload.get("unit_realization_closure")
    if not isinstance(identity, Mapping):
        raise PermissionError("failure terminal identity changed")
    sealed_at = _parse_utc(
        identity.get("sealed_at_utc"),
        name="failure terminal sealed_at",
    )
    expected_reproduction = _reproduce_scope_mismatch()
    if (
        set(identity)
        != {
            "candidate",
            "stage_id",
            "scientific_attempt_id",
            "scientific_attempt_ordinal",
            "runtime_compatibility_id",
            "sealed_at_utc",
        }
        or identity.get("candidate") != CANDIDATE
        or identity.get("stage_id") != STAGE_ID
        or identity.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or identity.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or identity.get("runtime_compatibility_id") != "c3"
        or not isinstance(failure, Mapping)
        or set(failure)
        != {
            "known_entrypoint_path",
            "known_subcommand",
            "known_precleanup_input_path",
            "known_cleanup_input_path",
            "known_policy_input_path",
            "original_call_artifact_available",
            "original_argv_claimed",
            "original_traceback_artifact_available",
            "original_traceback_claimed",
            "original_failure_time_claimed",
            "attempt_count",
            "retry",
            "samples_collected",
            "expected_exception_type",
            "expected_exception_message",
            "failure_observation",
            "frozen_control_flow",
        }
        or failure.get("known_entrypoint_path") != str(ENVIRONMENT_PATH)
        or failure.get("known_subcommand") != "stability-gate"
        or failure.get("known_precleanup_input_path")
        != str(PRECLEANUP_PATH)
        or failure.get("known_cleanup_input_path")
        != str(CLEANUP_RECEIPT_PATH)
        or failure.get("known_policy_input_path") != str(C3_POLICY_PATH)
        or failure.get("original_call_artifact_available") is not False
        or failure.get("original_argv_claimed") is not False
        or failure.get("original_traceback_artifact_available")
        is not False
        or failure.get("original_traceback_claimed") is not False
        or failure.get("original_failure_time_claimed") is not False
        or failure.get("attempt_count") != 1
        or failure.get("retry") is not False
        or failure.get("samples_collected") != 0
        or failure.get("expected_exception_type") != "PermissionError"
        or failure.get("expected_exception_message") != EXPECTED_ERROR
        or failure.get("failure_observation")
        != "post_hoc_deterministic_read_only_reproduction"
        or not _deep_exact_equal(reproduction, expected_reproduction)
        or not _deep_exact_equal(closure, _UNIT_REALIZATION_CLOSURE)
        or not _deep_exact_equal(
            payload.get("payload_observation"),
            _PAYLOAD_OBSERVATION,
        )
        or not _deep_exact_equal(
            payload.get("continuation_policy"),
            _CONTINUATION_POLICY,
        )
    ):
        raise PermissionError(
            "environment-stability failure semantics changed"
        )
    if not _deep_exact_equal(
        failure.get("frozen_control_flow"),
        _FAILURE_CONTROL_FLOW,
    ):
        raise PermissionError("frozen failure control flow changed")
    _validate_historical_state_record(
        payload.get("historical_state_observation"),
        sealed_at=sealed_at,
    )
    _validate_expiry_record(
        payload.get("authorization_expiry"),
        sealed_at=sealed_at,
    )

    terminalizer_root = _read_regular(Path(__file__).resolve())[1]
    if not _deep_exact_equal(
        payload.get("terminalizer_source_root"),
        terminalizer_root,
    ):
        raise PermissionError("terminalizer source lineage changed")
    generation_roots = _fixed_generation_roots()
    if not _deep_exact_equal(
        payload.get("failed_generation_roots"),
        generation_roots,
    ):
        raise PermissionError("failed C3 generation lineage changed")
    _values, roots, fingerprints = _validate_fixed_inputs()
    expected_inputs = {
        label: {"root": roots[label], **fingerprints[label]}
        for label in roots
    }
    if not _deep_exact_equal(
        payload.get("sealed_input_roots"),
        expected_inputs,
    ):
        raise PermissionError("sealed C3 input lineage changed")

    # Archival validation intentionally does not inspect ABSENT_OUTPUT_PATHS,
    # call systemctl, or consult the current clock.
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
