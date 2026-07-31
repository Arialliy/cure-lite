#!/usr/bin/env python3
"""Seal or validate the expired, unmaterialized v24 compatibility-c1 lane.

This tool has exactly one mutation: ``create-terminal`` may create the fixed
terminal JSON once.  It never creates environment/runtime evidence, touches a
dataset or GPU, or changes a systemd unit.  ``validate-terminal`` is read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY
    / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
)
RUN_PARENT = REPOSITORY / "runs/irstd1k_stage_a_seed42"

SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c1-"
    "expired-prewrite-terminal-v1"
)
CANDIDATE = "GCR-PACRE-v24"
STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
SCIENTIFIC_ATTEMPT_ID = "gcr_pacre_v24_D_R_zero_update_structural_r2"
SCIENTIFIC_ATTEMPT_ORDINAL = 2
RUNTIME_COMPATIBILITY_ID = "c1"
UNIT_NAME = "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service"
INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后运行"

TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_schema_compat_c1_expired_prewrite_terminal.json"
)
FORENSIC_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_forensic_terminal.json"
)
BRIDGE_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_authorization.json"
)
UNIT_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c1_unit_realization_authorization.json"
)
UNIT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c1_unit_realization_receipt.json"
)
R10_ROOT = (
    EVIDENCE_ROOT
    / "supervisor_v2_systemd_integration_preaccess_compat_c1_r10"
)
R10_AUTHORIZATION_PATH = R10_ROOT / "control/authorization.json"
R10_RECEIPT_PATH = R10_ROOT / "control/integration-receipt.json"
SCIENTIFIC_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_authorization.json"
)
SCIENTIFIC_ACCESS_AUDIT_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_access_audit.json"
)
FRAGMENT_PATH = (
    Path("/run/user") / str(os.getuid()) / "systemd/user" / UNIT_NAME
)

SESSION_PATH = Path(
    "/home/md0/ly/.codex/sessions/2026/07/28/"
    "rollout-2026-07-28T17-15-33-019fa802-0a9c-7ac3-bdd7-14aad146788f.jsonl"
)
SESSION_CALL_LINE = 23922
SESSION_OUTPUT_LINE = 23923
SESSION_PREFIX_SIZE = 47_888_063
SESSION_PREFIX_SHA256 = (
    "d9e0d6c8d68f1fa0cd82713ed1b9d276a75ebbd056ae45bafb8e26b834089275"
)
SESSION_EXPECTED_DEVICE = 2304
SESSION_EXPECTED_INODE = 228729190
SESSION_EXPECTED_UID = 1008
SESSION_EXPECTED_GID = 1008
SESSION_EXPECTED_MODE = 0o664
CALL_ID = "call_B6cHhD5WAqcHNl3m0gyRXHYP"
CALL_TIMESTAMP_UTC = "2026-07-30T06:26:54.454Z"
OUTPUT_TIMESTAMP_UTC = "2026-07-30T06:26:54.740Z"
CALL_RAW_SHA256 = (
    "8467eeb7573e7a0f99a417ec8e38fac05641733e5a5821e3bc47f201b24213d1"
)
CALL_RECORD_WITH_LF_SHA256 = (
    "44f0eca03e668ddc819de50523b13f61ee0a689ae76525e9528bd2a95f71e8d7"
)
CALL_ARGUMENTS_SHA256 = (
    "72ec137b1b315e8ef1634bd93614587a98a4ac469b0f558e6d0ce3a412dffde2"
)
OUTPUT_RAW_SHA256 = (
    "fbe9ff298290e8b61e7732711139cf447945c2e27e0c9d7518d5f0b249b9074c"
)
OUTPUT_RECORD_WITH_LF_SHA256 = (
    "3fbd235afd5aa9d2f9f156870dec942c5f21d1e04269f14da002b18052f34d82"
)
OUTPUT_PAYLOAD_SHA256 = (
    "3ecc5c0d94e85cab6a27f4fac60138051271f6e1518aca57481eabfdcfde3e77"
)

SOURCE_BINDINGS: dict[str, tuple[Path, str]] = {
    "compat_policy": (
        REPOSITORY / "tools/cure_lite_v24_preaccess_schema_compatibility.py",
        "fe715af48867f166d2e15727e0190844cfd79fb5c02fa5a440d294bb7f29e084",
    ),
    "compat_release": (
        REPOSITORY
        / "tools/cure_lite_v24_actual_runtime_release_preaccess_compat_c1.py",
        "395a013ff4f14160a0ac4e9845497caf9ecbaa6f2eeb3aa88fad54b63f514cfa",
    ),
    "compat_supervisor": (
        REPOSITORY
        / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c1.py",
        "7e2182da4f818bda5567c677194a1daf4cf02ce6874754acbc6b42095bd77447",
    ),
    "compat_adapter": (
        REPOSITORY
        / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c1.py",
        "95b4ae6e731cf4cec4c016a2d9ce20bce4e15a2ec7751e782eade9ee73344c77",
    ),
    "compat_unit_realizer": (
        REPOSITORY
        / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c1.py",
        "7bfc5944378d552f9f12654da5234762452f8dc5ee49f1bced47554bcbd58ece",
    ),
    "compat_unit_template": (
        REPOSITORY
        / "deploy/systemd/"
        "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service.template",
        "c67384b712b1e9f5573c16e2ee75b163124335176a9218e17ce8f59b35d009d5",
    ),
    "runtime_environment_tool": (
        REPOSITORY / "tools/cure_lite_v24_runtime_environment.py",
        "a40465786ce3537346372df5991bb6788d44feddfd497ec83a1dc302fb8b2fea",
    ),
}
EXPECTED_FRAGMENT_SHA256 = (
    "71b8f449c3ba67a80c45cd0a97c80bcd60adbae6aa6d6a3f0e1fb6625caa7c01"
)

ABSENCE_PATHS: dict[str, Path] = {
    "compat_environment_policy": (
        EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c1.json"
    ),
    "compat_environment_stability": (
        EVIDENCE_ROOT
        / "runtime_environment_stability_receipt_preaccess_compat_c1.json"
    ),
    "compat_environment_postcleanup": (
        EVIDENCE_ROOT
        / "runtime_environment_postcleanup_receipt_preaccess_compat_c1.json"
    ),
    "compatibility_receipt": (
        EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_receipt.json"
    ),
    "compat_runtime_spec": (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json"
    ),
    "compat_runtime_launch_authorization": (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c1_"
        "runtime_launch_authorization.json"
    ),
    "compat_runtime_artifact_root": (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_artifacts"
    ),
    "compat_gpu_lease_root": (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c1_gpu_lease"
    ),
    "scientific_run_root": (
        RUN_PARENT / "gcr_pacre_v24_D_R_structural_attempt_r2"
    ),
    "scientific_result_receipt": (
        EVIDENCE_ROOT / "D_R_structural_attempt_r2_receipt.json"
    ),
    "compat_run_root_alias": (
        RUN_PARENT
        / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c1"
    ),
    "compat_result_receipt_alias": (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c1_receipt.json"
    ),
    "compat_unit_realization_terminal": (
        EVIDENCE_ROOT
        / "r2_preaccess_compat_c1_unit_realization_terminal.json"
    ),
}

EVIDENCE_BINDINGS: dict[str, tuple[Path, str, str]] = {
    "forensic_terminal": (
        FORENSIC_TERMINAL_PATH,
        "terminal_fingerprint",
        "cure-lite-v24-r2-preaccess-schema-compat-c1-forensic-terminal-v1",
    ),
    "bridge_authorization": (
        BRIDGE_AUTHORIZATION_PATH,
        "authorization_fingerprint",
        "cure-lite-v24-r2-preaccess-schema-compat-c1-authorization-v1",
    ),
    "unit_authorization": (
        UNIT_AUTHORIZATION_PATH,
        "authorization_fingerprint",
        "cure-lite-v24-actual-unit-realization-authorization-v1",
    ),
    "unit_receipt": (
        UNIT_RECEIPT_PATH,
        "receipt_fingerprint",
        "cure-lite-v24-actual-unit-realization-receipt-v1",
    ),
    "r10_authorization": (
        R10_AUTHORIZATION_PATH,
        "authorization_fingerprint",
        "cure-lite-v24-supervisor-v2-systemd-integration-authorization-v2",
    ),
    "r10_receipt": (
        R10_RECEIPT_PATH,
        "receipt_fingerprint",
        "cure-lite-v24-supervisor-v2-systemd-integration-receipt-v1",
    ),
    "scientific_authorization": (
        SCIENTIFIC_AUTHORIZATION_PATH,
        "authorization_fingerprint",
        "cure-lite-v24-D_R-structural-r2-authorization-v1",
    ),
    "scientific_access_audit": (
        SCIENTIFIC_ACCESS_AUDIT_PATH,
        "receipt_fingerprint",
        "cure-lite-v24-split-access-audit-v1",
    ),
}

_LIVE_KEYS = {
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "FragmentPath",
    "DropInPaths",
    "NeedDaemonReload",
    "Transient",
    "Restart",
    "NRestarts",
    "InvocationID",
    "ExecMainPID",
    "ExecMainCode",
    "ExecMainStatus",
    "Result",
    "StateChangeTimestamp",
    "ActiveEnterTimestamp",
    "InactiveEnterTimestamp",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
}
_PAYLOAD_OBSERVATION = {
    "D_R_payload_accessed": False,
    "D_V_payload_accessed": False,
    "D_T_payload_accessed": False,
    "gpu_accessed": False,
    "training_started": False,
    "attempt_commit_present": False,
    "materialization_claim_present": False,
    "scientific_decision_present": False,
}
_CONTINUATION_POLICY = {
    "same_c1_reauthorization_allowed": False,
    "same_c1_receipt_sealing_allowed": False,
    "same_c1_release_allowed": False,
    "automatic_retry_allowed": False,
    "resume_allowed": False,
    "new_compatibility_generation_required": True,
    "runtime_launch_authorized_by_terminal": False,
    "systemd_mutation_authorized_by_terminal": False,
    "payload_access_authorized_by_terminal": False,
}
_OUTCOME = {
    "failure_phase": "compatibility_closure_prewrite_environment_policy",
    "terminal_reason": (
        "bridge_authorization_expired_after_environment_policy_command_failed_"
        "and_before_compatibility_receipt_or_runtime_materialization"
    ),
    "terminal_origin": "append-only-create-once-expired-prewrite-closure",
    "terminal_closure_passed": True,
    "compatibility_closure_passed": False,
    "scientific_attempt_consumed": False,
    "runtime_launch_consumed": False,
    "materialization_consumed": False,
}
_BODY_KEYS = {
    "schema_version",
    "candidate",
    "stage_id",
    "scientific_attempt_id",
    "scientific_attempt_ordinal",
    "runtime_compatibility_id",
    "unit_name",
    "created_at_utc",
    "instruction_id",
    "authorization_basis",
    "session_failure",
    "evidence_roots",
    "source_roots",
    "fragment_root",
    "live_unit_state",
    "absence_generation_roots",
    "derived_runtime_absences",
    "authorization_expiry",
    "payload_observation",
    "continuation_policy",
    "outcome",
}

UnitStateReader = Callable[[], Mapping[str, str]]
Clock = Callable[[], datetime]
ObservationHook = Callable[[], None]


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json(value: Mapping[str, object]) -> str:
    return _canonical_bytes(value).decode("utf-8")


def stable_fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_object(raw: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PermissionError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise PermissionError(f"{name} is not a JSON object")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_read(path: Path, *, expected_mode: int | None = None) -> tuple[bytes, dict[str, object]]:
    target = Path(path).absolute()
    parent = target.parent
    if parent.resolve(strict=True) != parent:
        raise PermissionError(f"non-canonical parent: {parent}")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or (
                expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode
            )
        ):
            raise PermissionError(f"unsafe file identity: {target}")
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
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(linked)
        ):
            raise PermissionError(f"file changed during read: {target}")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise PermissionError(f"short read: {target}")
        root = {
            "path": str(target),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "device": before.st_dev,
            "inode": before.st_ino,
            "owner_uid": before.st_uid,
            "owner_gid": before.st_gid,
            "mode": stat.S_IMODE(before.st_mode),
            "nlink": before.st_nlink,
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }
        return raw, root
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _load_sealed(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str,
) -> tuple[dict[str, object], dict[str, object]]:
    raw, root = _stable_read(path, expected_mode=0o444)
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise PermissionError(f"sealed JSON layout changed: {path}")
    payload = _json_object(raw[:-1], name=str(path))
    if payload.get("schema_version") != schema:
        raise PermissionError(f"sealed JSON schema changed: {path}")
    fingerprint = payload.get(fingerprint_field)
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise PermissionError(f"sealed JSON fingerprint missing: {path}")
    body = dict(payload)
    del body[fingerprint_field]
    if stable_fingerprint(body) != fingerprint:
        raise PermissionError(f"sealed JSON fingerprint changed: {path}")
    if raw != _canonical_bytes(payload) + b"\n":
        raise PermissionError(f"sealed JSON canonical encoding changed: {path}")
    root[fingerprint_field] = fingerprint
    root["schema_version"] = schema
    return payload, root


def _parse_utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PermissionError(f"{name} is not UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise PermissionError(f"{name} is invalid") from error
    if parsed.tzinfo is None:
        raise PermissionError(f"{name} is naive")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _observe_source_roots() -> dict[str, object]:
    result: dict[str, object] = {}
    for name, (path, expected_sha256) in SOURCE_BINDINGS.items():
        _, root = _stable_read(path)
        if root["file_sha256"] != expected_sha256:
            raise PermissionError(f"compatibility source hash changed: {name}")
        result[name] = root
    return result


def _observe_fragment_root() -> dict[str, object]:
    _, root = _stable_read(FRAGMENT_PATH, expected_mode=0o600)
    if root["file_sha256"] != EXPECTED_FRAGMENT_SHA256:
        raise PermissionError("compatibility fragment hash changed")
    return root


def _observe_absence(path: Path) -> dict[str, object]:
    target = Path(path).absolute()
    parent = target.parent
    if parent.resolve(strict=True) != parent:
        raise PermissionError(f"absence parent is non-canonical: {parent}")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(parent_fd)
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise PermissionError(f"required absent path exists: {target}")
        after = os.fstat(parent_fd)
        if _stat_identity(before) != _stat_identity(after):
            raise PermissionError(f"absence parent changed: {parent}")
        return {
            "path": str(target),
            "basename": target.name,
            "lexists": False,
            "parent_path": str(parent),
            "parent_device": before.st_dev,
            "parent_inode": before.st_ino,
            "parent_owner_uid": before.st_uid,
            "parent_owner_gid": before.st_gid,
            "parent_mode": stat.S_IMODE(before.st_mode),
            "parent_nlink": before.st_nlink,
            "parent_size": before.st_size,
            "parent_mtime_ns": before.st_mtime_ns,
            "parent_ctime_ns": before.st_ctime_ns,
        }
    finally:
        os.close(parent_fd)


def _same_absence(left: object, right: object) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    if set(left) != set(right):
        return False
    volatile = {"parent_size", "parent_mtime_ns", "parent_ctime_ns"}
    return all(left[key] == right[key] for key in left if key not in volatile)


def _observe_absences() -> dict[str, object]:
    return {
        name: _observe_absence(path)
        for name, path in ABSENCE_PATHS.items()
    }


def _observe_evidence() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    roots: dict[str, object] = {}
    payloads: dict[str, dict[str, object]] = {}
    for name, (path, fingerprint_field, schema) in EVIDENCE_BINDINGS.items():
        payload, root = _load_sealed(
            path,
            fingerprint_field=fingerprint_field,
            schema=schema,
        )
        roots[name] = root
        payloads[name] = payload
    return roots, payloads


def _observe_session_failure() -> dict[str, object]:
    target = SESSION_PATH.absolute()
    parent_fd = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        stable_fields = (
            before.st_dev,
            before.st_ino,
            before.st_uid,
            before.st_gid,
            stat.S_IMODE(before.st_mode),
            before.st_nlink,
        )
        expected_fields = (
            SESSION_EXPECTED_DEVICE,
            SESSION_EXPECTED_INODE,
            SESSION_EXPECTED_UID,
            SESSION_EXPECTED_GID,
            SESSION_EXPECTED_MODE,
            1,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or stable_fields != expected_fields
            or before.st_size < SESSION_PREFIX_SIZE
        ):
            raise PermissionError("session generation identity changed")
        prefix = b""
        while len(prefix) < SESSION_PREFIX_SIZE:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, SESSION_PREFIX_SIZE - len(prefix)),
            )
            if not chunk:
                break
            prefix += chunk
        after = os.fstat(descriptor)
        linked = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        for observed in (after, linked):
            if (
                observed.st_dev,
                observed.st_ino,
                observed.st_uid,
                observed.st_gid,
                stat.S_IMODE(observed.st_mode),
                observed.st_nlink,
            ) != stable_fields:
                raise PermissionError("session generation changed during read")
        if (
            len(prefix) != SESSION_PREFIX_SIZE
            or hashlib.sha256(prefix).hexdigest() != SESSION_PREFIX_SHA256
        ):
            raise PermissionError("session fixed prefix changed")
        lines = prefix.splitlines(keepends=True)
        if len(lines) != SESSION_OUTPUT_LINE:
            raise PermissionError("session fixed prefix line count changed")
        call_record = lines[SESSION_CALL_LINE - 1]
        output_record = lines[SESSION_OUTPUT_LINE - 1]
        if not call_record.endswith(b"\n") or not output_record.endswith(b"\n"):
            raise PermissionError("session records lack LF terminators")
        checks = (
            (call_record[:-1], CALL_RAW_SHA256, "call raw"),
            (call_record, CALL_RECORD_WITH_LF_SHA256, "call record"),
            (output_record[:-1], OUTPUT_RAW_SHA256, "output raw"),
            (output_record, OUTPUT_RECORD_WITH_LF_SHA256, "output record"),
        )
        for raw, expected, name in checks:
            if hashlib.sha256(raw).hexdigest() != expected:
                raise PermissionError(f"session {name} hash changed")
        call = _json_object(call_record[:-1], name="session call record")
        output = _json_object(output_record[:-1], name="session output record")
        call_payload = call.get("payload")
        output_payload = output.get("payload")
        if not isinstance(call_payload, Mapping) or not isinstance(
            output_payload, Mapping
        ):
            raise PermissionError("session payload records changed")
        arguments = call_payload.get("arguments")
        output_text = output_payload.get("output")
        if (
            call.get("timestamp") != CALL_TIMESTAMP_UTC
            or output.get("timestamp") != OUTPUT_TIMESTAMP_UTC
            or call_payload.get("type") != "function_call"
            or call_payload.get("name") != "exec_command"
            or call_payload.get("call_id") != CALL_ID
            or output_payload.get("type") != "function_call_output"
            or output_payload.get("call_id") != CALL_ID
            or not isinstance(arguments, str)
            or not isinstance(output_text, str)
            or hashlib.sha256(arguments.encode()).hexdigest()
            != CALL_ARGUMENTS_SHA256
            or hashlib.sha256(output_text.encode()).hexdigest()
            != OUTPUT_PAYLOAD_SHA256
            or "Process exited with code 1" not in output_text
            or "PermissionError: precleanup inventory unit scope changed"
            not in output_text
        ):
            raise PermissionError("session call/output semantics changed")
        return {
            "session_path": str(target),
            "session_device": before.st_dev,
            "session_inode": before.st_ino,
            "session_owner_uid": before.st_uid,
            "session_owner_gid": before.st_gid,
            "session_mode": stat.S_IMODE(before.st_mode),
            "session_nlink": before.st_nlink,
            "prefix_line_count": SESSION_OUTPUT_LINE,
            "prefix_size": SESSION_PREFIX_SIZE,
            "prefix_sha256": SESSION_PREFIX_SHA256,
            "call_line_number": SESSION_CALL_LINE,
            "call_timestamp_utc": CALL_TIMESTAMP_UTC,
            "call_id": CALL_ID,
            "tool_name": "exec_command",
            "call_line_raw_sha256": CALL_RAW_SHA256,
            "call_record_with_lf_sha256": CALL_RECORD_WITH_LF_SHA256,
            "call_arguments_sha256": CALL_ARGUMENTS_SHA256,
            "output_line_number": SESSION_OUTPUT_LINE,
            "output_timestamp_utc": OUTPUT_TIMESTAMP_UTC,
            "output_line_raw_sha256": OUTPUT_RAW_SHA256,
            "output_record_with_lf_sha256": OUTPUT_RECORD_WITH_LF_SHA256,
            "output_payload_sha256": OUTPUT_PAYLOAD_SHA256,
            "exit_code": 1,
            "error_type": "PermissionError",
            "error_message": "precleanup inventory unit scope changed",
            "failed_output_path": str(
                ABSENCE_PATHS["compat_environment_policy"]
            ),
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def collect_live_unit_state() -> dict[str, str]:
    properties = ",".join(sorted(_LIVE_KEYS))
    completed = subprocess.run(
        (
            "/usr/bin/systemctl",
            "--user",
            "show",
            UNIT_NAME,
            "--no-pager",
            f"--property={properties}",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("failed to query compatibility unit")
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator != "=" or key in result:
            raise PermissionError("ambiguous systemd property output")
        result[key] = value
    if set(result) != _LIVE_KEYS:
        raise PermissionError("systemd property closure changed")
    return result


def _validate_live_state(
    state: Mapping[str, str],
    *,
    unit_receipt: Mapping[str, object],
) -> dict[str, str]:
    observed = dict(state)
    if set(observed) != _LIVE_KEYS:
        raise PermissionError("live unit state keys changed")
    required = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "FragmentPath": str(FRAGMENT_PATH),
        "DropInPaths": "",
        "NeedDaemonReload": "no",
        "Transient": "no",
        "Restart": "no",
        "NRestarts": "0",
        "InvocationID": "",
        "ExecMainPID": "0",
        "ExecMainCode": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "StateChangeTimestamp": "",
        "ActiveEnterTimestamp": "",
        "InactiveEnterTimestamp": "",
        "ExecMainStartTimestamp": "",
        "ExecMainExitTimestamp": "",
    }
    if observed != required:
        raise PermissionError("compatibility unit is not never-started static")
    shadow = unit_receipt.get("full_static_shadow")
    if not isinstance(shadow, Mapping):
        raise PermissionError("unit receipt static shadow missing")
    for key in (
        "LoadState",
        "ActiveState",
        "SubState",
        "UnitFileState",
        "FragmentPath",
        "NeedDaemonReload",
        "Transient",
        "Restart",
        "NRestarts",
    ):
        if shadow.get(key) != required[key]:
            raise PermissionError("unit receipt/live shadow mismatch")
    return observed


def _validate_evidence_semantics(
    payloads: Mapping[str, Mapping[str, object]],
    *,
    now: datetime,
) -> dict[str, object]:
    bridge = payloads["bridge_authorization"]
    unit_auth = payloads["unit_authorization"]
    unit_receipt = payloads["unit_receipt"]
    r10 = payloads["r10_receipt"]
    audit = payloads["scientific_access_audit"]
    scientific_auth = payloads["scientific_authorization"]
    forensic = payloads["forensic_terminal"]

    bridge_created = _parse_utc(
        bridge.get("created_at_utc"), name="bridge created_at_utc"
    )
    bridge_issued = _parse_utc(
        bridge.get("issued_at_utc"), name="bridge issued_at_utc"
    )
    bridge_expires = _parse_utc(
        bridge.get("expires_at_utc"), name="bridge expires_at_utc"
    )
    unit_created = _parse_utc(
        unit_auth.get("created_at_utc"), name="unit auth created_at_utc"
    )
    unit_issued = _parse_utc(
        unit_auth.get("issued_at_utc"), name="unit auth issued_at_utc"
    )
    unit_expires = _parse_utc(
        unit_auth.get("expires_at_utc"), name="unit auth expires_at_utc"
    )
    receipt_created = _parse_utc(
        unit_receipt.get("created_at_utc"), name="unit receipt created_at_utc"
    )
    call_time = _parse_utc(CALL_TIMESTAMP_UTC, name="session call timestamp")
    output_time = _parse_utc(
        OUTPUT_TIMESTAMP_UTC, name="session output timestamp"
    )
    current = now.astimezone(timezone.utc)
    if (
        bridge_created != bridge_issued
        or not bridge_issued < bridge_expires < current
        or not bridge_issued <= unit_issued
        or not unit_issued <= unit_created <= receipt_created <= unit_expires
        or not receipt_created <= bridge_expires
        or not bridge_issued <= call_time <= output_time <= bridge_expires
    ):
        raise PermissionError("expired authorization chronology changed")
    if (
        bridge.get("instruction_id") != INSTRUCTION_ID
        or bridge.get("authorization_basis") != AUTHORIZATION_BASIS
        or unit_auth.get("instruction_id") != INSTRUCTION_ID
        or unit_auth.get("authorization_basis") != AUTHORIZATION_BASIS
        or unit_auth.get("unit_name") != UNIT_NAME
        or unit_receipt.get("unit_name") != UNIT_NAME
        or unit_receipt.get("authorization_fingerprint")
        != unit_auth.get("authorization_fingerprint")
        or unit_receipt.get("passed") is not True
        or unit_receipt.get("static") is not True
        or unit_receipt.get("enabled") is not False
        or unit_receipt.get("started") is not False
        or unit_receipt.get("runtime_spec_absent_at_receipt") is not True
        or unit_receipt.get("payload_authority") != "none"
        or any(
            unit_receipt.get(key) is not False
            for key in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError("unit authorization/receipt semantics changed")
    compat_lane = bridge.get("compat_lane_authority")
    scientific_lane = bridge.get("scientific_authority")
    if (
        not isinstance(compat_lane, Mapping)
        or compat_lane.get("compat_start_authorized") is not False
        or compat_lane.get("compat_enable_authorized") is not False
        or compat_lane.get("payload_access_authorized") is not False
        or compat_lane.get(
            "runtime_spec_creation_authorized_by_this_receipt"
        )
        is not False
        or compat_lane.get(
            "runtime_launch_authorization_authorized_by_this_receipt"
        )
        is not False
        or not isinstance(scientific_lane, Mapping)
        or scientific_lane.get("automatic_retry") is not False
        or scientific_lane.get("resume") is not False
        or scientific_lane.get("training_authorized") is not False
        or scientific_lane.get("D_V_payload_authorized") is not False
        or scientific_lane.get("D_T_payload_authorized") is not False
    ):
        raise PermissionError("bridge authority closure changed")
    post_removal = r10.get("post_removal_unit_state")
    if (
        r10.get("passed") is not True
        or r10.get("fragment_removed") is not True
        or r10.get("payload_authority") != "none"
        or any(
            r10.get(key) is not False
            for key in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
                "gpu_accessed",
            )
        )
        or not isinstance(post_removal, Mapping)
        or post_removal.get("LoadState") != "not-found"
        or post_removal.get("ActiveState") != "inactive"
        or post_removal.get("SubState") != "dead"
        or post_removal.get("FragmentPath") != ""
        or post_removal.get("NRestarts") != "0"
    ):
        raise PermissionError("r10 PASS/removal closure changed")
    if (
        audit.get("observed_payloads") != []
        or audit.get("D_V_payload_accessed") is not False
        or audit.get("D_T_payload_accessed") is not False
        or scientific_auth.get("D_R_payload_authorized") is not True
        or scientific_auth.get("D_V_payload_authorized") is not False
        or scientific_auth.get("D_T_payload_authorized") is not False
        or scientific_auth.get("training_authorized") is not False
        or forensic.get("forensic_closure_passed") is not True
        or forensic.get("payload_observation") != {
            "D_R_payload_accessed": False,
            "D_T_payload_accessed": False,
            "D_V_payload_accessed": False,
            "attempt_commit_present": False,
            "materialization_claim_present": False,
            "scientific_decision_present": False,
            "training_started": False,
        }
    ):
        raise PermissionError("scientific/payload predecessor closure changed")
    return {
        "bridge_created_at_utc": _format_utc(bridge_created),
        "bridge_issued_at_utc": _format_utc(bridge_issued),
        "bridge_expires_at_utc": _format_utc(bridge_expires),
        "bridge_authorization_expired": True,
        "unit_authorization_created_at_utc": _format_utc(unit_created),
        "unit_authorization_issued_at_utc": _format_utc(unit_issued),
        "unit_authorization_expires_at_utc": _format_utc(unit_expires),
        "unit_receipt_created_at_utc": _format_utc(receipt_created),
        "unit_receipt_within_bridge_and_unit_windows": True,
        "environment_policy_failure_within_bridge_window": True,
        "same_c1_authorization_path_is_create_once_and_expired": True,
    }


def _derived_runtime_absences() -> dict[str, object]:
    artifact_root = ABSENCE_PATHS["compat_runtime_artifact_root"]
    return {
        "attempt_commit": {
            "path": str(artifact_root / "attempt-commit.json"),
            "absent_by_parent_root": "compat_runtime_artifact_root",
        },
        "materialization_claim": {
            "path": str(artifact_root / "materialization-claim.json"),
            "absent_by_parent_root": "compat_runtime_artifact_root",
        },
    }


def _collect_snapshot(
    *,
    unit_state_reader: UnitStateReader,
    now: datetime,
) -> dict[str, object]:
    session = _observe_session_failure()
    evidence_roots, payloads = _observe_evidence()
    source_roots = _observe_source_roots()
    fragment_root = _observe_fragment_root()
    live_state = _validate_live_state(
        unit_state_reader(),
        unit_receipt=payloads["unit_receipt"],
    )
    absences = _observe_absences()
    expiry = _validate_evidence_semantics(payloads, now=now)
    receipt_fragment = payloads["unit_receipt"].get("fragment_identity")
    if not isinstance(receipt_fragment, Mapping):
        raise PermissionError("unit receipt fragment identity missing")
    for key in (
        "path",
        "file_sha256",
        "device",
        "inode",
        "owner_uid",
        "mode",
        "nlink",
    ):
        if receipt_fragment.get(key) != fragment_root.get(key):
            raise PermissionError("live fragment differs from unit receipt")
    return {
        "session_failure": session,
        "evidence_roots": evidence_roots,
        "source_roots": source_roots,
        "fragment_root": fragment_root,
        "live_unit_state": live_state,
        "absence_generation_roots": absences,
        "derived_runtime_absences": _derived_runtime_absences(),
        "authorization_expiry": expiry,
    }


def _snapshots_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        if key != "absence_generation_roots":
            if left[key] != right[key]:
                return False
            continue
        left_abs = left[key]
        right_abs = right[key]
        if (
            not isinstance(left_abs, Mapping)
            or not isinstance(right_abs, Mapping)
            or set(left_abs) != set(right_abs)
            or any(
                not _same_absence(left_abs[name], right_abs[name])
                for name in left_abs
            )
        ):
            return False
    return True


def _fixed_terminal(path: Path | None) -> Path:
    selected = TERMINAL_PATH if path is None else Path(path).absolute()
    if selected != TERMINAL_PATH.absolute():
        raise PermissionError("expired-prewrite terminal path is not fixed")
    return selected


def _require_absent(path: Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"create-once terminal already exists: {path}")


def _write_create_once(path: Path, body: Mapping[str, object]) -> dict[str, object]:
    payload = dict(body)
    payload["terminal_fingerprint"] = stable_fingerprint(body)
    raw = _canonical_bytes(payload) + b"\n"
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent_fd,
        )
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        if (
            observed.st_uid != os.getuid()
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o444
            or observed.st_size != len(raw)
        ):
            raise PermissionError("created terminal identity is unsafe")
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    sealed, _ = _load_sealed(
        path,
        fingerprint_field="terminal_fingerprint",
        schema=SCHEMA,
    )
    return sealed


def create_terminal(
    *,
    terminal_path: Path | None = None,
    unit_state_reader: UnitStateReader = collect_live_unit_state,
    now: Clock = lambda: datetime.now(timezone.utc),
    between_observations: ObservationHook | None = None,
) -> dict[str, object]:
    path = _fixed_terminal(terminal_path)
    _require_absent(path)
    observed_at = now().astimezone(timezone.utc)
    first = _collect_snapshot(
        unit_state_reader=unit_state_reader,
        now=observed_at,
    )
    if between_observations is not None:
        between_observations()
    second = _collect_snapshot(
        unit_state_reader=unit_state_reader,
        now=observed_at,
    )
    if not _snapshots_equal(first, second):
        raise PermissionError("expired-prewrite closure changed between observations")
    _require_absent(path)
    body: dict[str, object] = {
        "schema_version": SCHEMA,
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
        "unit_name": UNIT_NAME,
        "created_at_utc": _format_utc(observed_at),
        "instruction_id": INSTRUCTION_ID,
        "authorization_basis": AUTHORIZATION_BASIS,
        **second,
        "payload_observation": dict(_PAYLOAD_OBSERVATION),
        "continuation_policy": dict(_CONTINUATION_POLICY),
        "outcome": dict(_OUTCOME),
    }
    if set(body) != _BODY_KEYS:
        raise AssertionError("internal expired-prewrite terminal keys changed")
    return _write_create_once(path, body)


def validate_terminal(
    *,
    terminal_path: Path | None = None,
    unit_state_reader: UnitStateReader = collect_live_unit_state,
    now: Clock = lambda: datetime.now(timezone.utc),
) -> dict[str, object]:
    path = _fixed_terminal(terminal_path)
    payload, _ = _load_sealed(
        path,
        fingerprint_field="terminal_fingerprint",
        schema=SCHEMA,
    )
    if set(payload) != _BODY_KEYS | {"terminal_fingerprint"}:
        raise PermissionError("expired-prewrite terminal exact keys changed")
    identity = {
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
        "unit_name": UNIT_NAME,
        "instruction_id": INSTRUCTION_ID,
        "authorization_basis": AUTHORIZATION_BASIS,
    }
    if any(payload.get(key) != value for key, value in identity.items()):
        raise PermissionError("expired-prewrite terminal identity changed")
    if (
        payload.get("payload_observation") != _PAYLOAD_OBSERVATION
        or payload.get("continuation_policy") != _CONTINUATION_POLICY
        or payload.get("outcome") != _OUTCOME
        or payload.get("derived_runtime_absences")
        != _derived_runtime_absences()
    ):
        raise PermissionError("expired-prewrite terminal contract changed")
    created = _parse_utc(
        payload.get("created_at_utc"), name="terminal created_at_utc"
    )
    current = now().astimezone(timezone.utc)
    if created > current:
        raise PermissionError("terminal creation time is in the future")
    observed = _collect_snapshot(
        unit_state_reader=unit_state_reader,
        now=current,
    )
    stored = {
        key: payload[key]
        for key in (
            "session_failure",
            "evidence_roots",
            "source_roots",
            "fragment_root",
            "live_unit_state",
            "absence_generation_roots",
            "derived_runtime_absences",
            "authorization_expiry",
        )
    }
    if not _snapshots_equal(stored, observed):
        raise PermissionError("expired-prewrite terminal live closure changed")
    bridge_expiry = _parse_utc(
        payload["authorization_expiry"]["bridge_expires_at_utc"],
        name="terminal bridge expiry",
    )
    if created <= bridge_expiry:
        raise PermissionError("terminal was sealed before bridge expiry")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "create-terminal",
        help="create the fixed expired c1 terminal exactly once",
    )
    sub.add_parser(
        "validate-terminal",
        help="read-only validation of the fixed expired c1 terminal",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create-terminal":
        payload = create_terminal()
    elif args.command == "validate-terminal":
        payload = validate_terminal()
    else:  # pragma: no cover
        raise AssertionError("unknown command")
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
