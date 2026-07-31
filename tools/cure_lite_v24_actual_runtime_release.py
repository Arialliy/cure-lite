#!/usr/bin/env python3
"""Create-only release builder for the single CURE-Lite v24 actual D_R r2 run.

This tool is data-blind.  It validates already-sealed execution evidence,
freezes the actual supervisor-v2 runtime spec, and only then creates a short-
lived launch authorization.  It never starts, enables, reloads, or removes a
systemd unit and it never imports a dataset or model package.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Callable, Mapping, Sequence

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from tools import cure_lite_v24_actual_unit_realization as actual_realizer
from tools import (
    cure_lite_v24_actual_unit_realization_recovery as realization_recovery,
)
from tools import cure_lite_v24_environment_cleanup as cleanup
from tools import cure_lite_v24_runtime_environment as runtime_environment
from tools import cure_lite_v24_runtime_supervisor as supervisor
from tools import cure_lite_v24_user_systemd_integration as integration


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()
RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_spec.json"
)
RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT /
    "D_R_structural_attempt_r2_runtime_launch_authorization.json"
)
SCIENTIFIC_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_authorization.json"
)
SCIENTIFIC_ACCESS_AUDIT_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_access_audit.json"
)
PRIOR_R1_INTERRUPTION_RECEIPT_PATH = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/"
    "D_R_structural_attempt_r1_interruption_receipt_v2.json"
).resolve()
RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_artifacts"
)
GPU_LEASE_ROOT = EVIDENCE_ROOT / "D_R_structural_attempt_r2_gpu_lease"

SUPERVISOR_PATH = (REPOSITORY / "tools/cure_lite_v24_runtime_supervisor.py").resolve()
RUNTIME_ENVIRONMENT_PATH = (
    REPOSITORY / "tools/cure_lite_v24_runtime_environment.py"
).resolve()
ADAPTER_PATH = (
    REPOSITORY / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2.py"
).resolve()
LEGACY_GATE_PATH = (
    REPOSITORY / "tools/run_cure_lite_v24_gcr_pacre_dr_gate.py"
).resolve()
PYTHON_PATH = Path("/usr/bin/python3.12")
RUNTIME_DEPENDENCY_SITE_PATH = Path(
    "/home/md0/ly/MSHNet/.venv/lib/python3.12/site-packages"
)
RUNTIME_DEPENDENCY_SITE_DEVICE = 2304
RUNTIME_DEPENDENCY_SITE_INODE = 228331323
RUNTIME_DEPENDENCY_SITE_OWNER_UID = 1008
RUNTIME_DEPENDENCY_SITE_MODE = 0o775

CANDIDATE = "GCR-PACRE-v24"
STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
ATTEMPT_ID = "gcr_pacre_v24_D_R_zero_update_structural_r2"
UNIT_NAME = "cure-lite-v24-gcr-pacre-dr-r2.service"
INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后运行"
SOURCE_CLOSURE_FINGERPRINT_103 = (
    "28d26759a68785e9c99917fcfa8b36430c7f6e5463282d66eeab5c711e425e9f"
)
RELEASE_SCHEMA = "cure-lite-v24-actual-runtime-release-spec-roots-v1"

_SHA = re.compile(r"[0-9a-f]{64}")
_AUDIT_RECEIPT_KEYS = {
    "schema_version", "created_at_utc", "command", "environment_binding",
    "inventory", "passed", "error_type", "error_message",
    "D_R_payload_accessed", "D_V_payload_accessed", "D_T_payload_accessed",
    "receipt_fingerprint",
}
_SCIENTIFIC_AUTHORIZATION_KEYS = {
    "schema_version", "candidate", "stage_id", "run_id", "status",
    "D_R_payload_authorized", "D_V_payload_authorized",
    "D_T_payload_authorized", "training_authorized", "allowed_splits",
    "allowed_purposes", "expires_after_single_materialization",
    "access_audit_receipt_fingerprint", "dataset_free_receipt_file_sha256",
    "dataset_free_receipt_fingerprint", "efficiency_receipt_sha256",
    "efficiency_section_fingerprint", "expected_cache_fingerprint",
    "expected_population_fingerprint", "expected_real_inputs_fingerprint",
    "manifest_file_sha256", "protocol_preregistration_fingerprint",
    "source_binding_fingerprint", "source_closure_fingerprint",
    "state_index_file_sha256", "authorization_fingerprint",
}
_SCIENTIFIC_AUDIT_KEYS = {
    "schema_version", "stage_id", "allowed_splits", "observed_payloads",
    "event_log_fingerprint", "source_manifest_fingerprint",
    "D_V_payload_accessed", "D_T_payload_accessed", "receipt_fingerprint",
}
_PRIOR_INTERRUPTION_KEYS = {
    "schema_version", "receipt_role", "candidate", "stage_id", "run_id",
    "created_at_local", "supersession", "corrections",
    "authoritative_observation", "official_receipt_evidence",
    "bounded_known_path_search", "runtime_scope", "threat_model",
    "protocol_disposition", "receipt_fingerprint",
}
_INTEGRATION_RECEIPT_KEYS = {
    "schema_version", "scenario_id", "identity", "authorization_path",
    "authorization_file_sha256", "authorization_fingerprint",
    "integration_terminal_path", "integration_terminal_file_sha256",
    "integration_terminal_fingerprint", "removal_authorization_path",
    "removal_authorization_file_sha256",
    "removal_authorization_fingerprint", "removal_state_path",
    "removal_state_file_sha256", "removal_state_fingerprint",
    "supervisor_evidence", "fragment_removed", "post_removal_unit_state",
    "passed", "payload_authority", "D_R_payload_accessed",
    "D_V_payload_accessed", "D_T_payload_accessed", "gpu_accessed",
    "receipt_fingerprint",
}
_INTEGRATION_TERMINAL_KEYS = {
    "schema_version", "scenario_id", "identity", "authorization_fingerprint",
    "runtime_spec_fingerprint", "created_at_utc", "passed",
    "completed_actions", "supervisor_evidence", "error_type",
    "error_message", "direct_systemctl_start_attempted", "enable_attempted",
    "remove_attempted", "payload_authority", "D_R_payload_accessed",
    "D_V_payload_accessed", "D_T_payload_accessed", "gpu_accessed",
    "integration_terminal_fingerprint",
}
_INTEGRATION_REMOVAL_STATE_KEYS = {
    "schema_version", "scenario_id", "unit_name",
    "removal_authorization_fingerprint", "passed", "remove_attempted",
    "fragment_absent", "not_found_state", "completed_actions", "error_type",
    "error_message", "payload_authority", "D_R_payload_accessed",
    "D_V_payload_accessed", "D_T_payload_accessed",
    "removal_state_fingerprint",
}
_INTEGRATION_NOT_FOUND_STATE = {
    "LoadState": "not-found",
    "UnitFileState": "",
    "ActiveState": "inactive",
    "SubState": "dead",
    "FragmentPath": "",
    "DropInPaths": "",
    "Transient": "no",
    "Restart": "no",
    "NRestarts": "0",
    "NeedDaemonReload": "no",
}
_INTEGRATION_TERMINAL_ACTIONS = [
    "realize-static-fragment",
    "daemon-reload-after-realization",
    "supervisor-commit-and-start",
    "verify-supervisor-evidence",
]
_INTEGRATION_REMOVAL_ACTIONS = [
    *_INTEGRATION_TERMINAL_ACTIONS,
    "remove-authorized-fragment",
    "daemon-reload-after-removal",
]
_INTEGRATION_SUPERVISOR_EVIDENCE_KEYS = {
    "invocation_id", "attempt_commit_fingerprint",
    "launch_lease_fingerprint", "materialization_claim_fingerprint",
    "precommit_fingerprint", "start_ack_fingerprint",
    "child_prespawn_fingerprint", "runtime_terminal_fingerprint",
    "systemd_terminal_fingerprint", "runtime_attestation_absent",
    "gpu_lease_evidence_absent",
}
_INTEGRATION_FRAGMENT_IDENTITY_KEYS = {
    "fragment_path", "fragment_sha256", "device", "inode", "owner_uid",
    "mode", "nlink",
}
_REALIZATION_RECEIPT_KEYS = {
    "schema_version", "candidate", "stage_id", "attempt_id", "unit_name",
    "created_at_utc", "authorization_path", "authorization_file_sha256",
    "authorization_fingerprint", "instruction_id", "manager_generation",
    "unit_path_policy", "template_binding", "rendered_fragment",
    "runtime_spec_binding", "expected_future_runtime_spec_path",
    "runtime_spec_absent_at_receipt", "executable_bindings",
    "fragment_identity", "full_static_shadow", "completed_actions", "static",
    "enabled", "started", "removed", "passed", "payload_authority",
    "D_R_payload_accessed", "D_V_payload_accessed", "D_T_payload_accessed",
    "receipt_fingerprint",
}
_REALIZATION_FRAGMENT_IDENTITY_KEYS = {
    "path", "file_sha256", "device", "inode", "owner_uid", "mode", "nlink",
}
_CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
_ShadowReader = Callable[[], Mapping[str, str]]
_ManagerReader = Callable[[], Mapping[str, object]]
_UnitPathPolicyReader = Callable[[Path], Mapping[str, object]]


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def stable_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _deep_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return (
            set(left) == set(right)
            and all(_deep_exact_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return (
            len(left) == len(right)
            and all(
                _deep_exact_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    return left == right


def _stat_generation(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _stat_snapshot(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid,
        value.st_gid, value.st_nlink, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _parent_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid,
        stat.S_IMODE(value.st_mode),
    )


def _open_stable_parent(
    target: Path, *, private: bool = False,
) -> tuple[int, os.stat_result]:
    parent = target.parent
    before = os.lstat(parent)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or parent.resolve(strict=True) != parent
        or (
            private
            and (
                before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o700
            )
        )
    ):
        raise PermissionError(
            "release directory must be canonical owned mode 0700"
            if private else "release input parent is not canonical"
        )
    parent_fd = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(parent_fd)
    if (
        _parent_identity(opened) != _parent_identity(before)
        or not stat.S_ISDIR(opened.st_mode)
    ):
        os.close(parent_fd)
        raise PermissionError("release input parent changed while opening")
    return parent_fd, before


def _verify_parent_generation(
    parent: Path, parent_fd: int, expected: os.stat_result,
) -> None:
    opened = os.fstat(parent_fd)
    linked = os.lstat(parent)
    if (
        _parent_identity(opened) != _parent_identity(expected)
        or _parent_identity(linked) != _parent_identity(expected)
        or not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
    ):
        raise PermissionError("release input parent generation changed")


def _read_all_fd(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _stable_read_file(
    path: str | Path, *, private_parent: bool = False,
) -> tuple[bytes, dict[str, object]]:
    target = Path(path).absolute()
    if target.name in {"", ".", ".."}:
        raise ValueError("release file must have one basename")
    parent_fd, parent_before = _open_stable_parent(
        target, private=private_parent,
    )
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError("release input is not a regular file")
        raw = _read_all_fd(descriptor)
        after = os.fstat(descriptor)
        linked = os.stat(
            target.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            _stat_snapshot(after) != _stat_snapshot(before)
            or _stat_snapshot(linked) != _stat_snapshot(after)
            or not stat.S_ISREG(linked.st_mode)
        ):
            raise PermissionError("release input identity changed while reading")
        _verify_parent_generation(target.parent, parent_fd, parent_before)
        return raw, {
            "path": str(target),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "device": after.st_dev, "inode": after.st_ino,
            "owner_uid": after.st_uid, "mode": stat.S_IMODE(after.st_mode),
            "nlink": after.st_nlink,
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def file_sha256(path: str | Path) -> str:
    return str(_stable_read_file(path)[1]["file_sha256"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _private_directory(path: Path, *, create: bool = False) -> dict[str, object]:
    target = path.absolute()
    if create:
        target.mkdir(mode=0o700, parents=False, exist_ok=False)
    current = os.lstat(target)
    if (
        stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode)
        or target.resolve(strict=True) != target or current.st_uid != os.getuid()
        or stat.S_IMODE(current.st_mode) != 0o700
    ):
        raise PermissionError("release directory must be canonical owned mode 0700")
    descriptor = os.open(
        target,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        linked = os.lstat(target)
        if (
            _stat_snapshot(opened) != _stat_snapshot(current)
            or _stat_snapshot(linked) != _stat_snapshot(current)
        ):
            raise PermissionError("release directory changed while observed")
    finally:
        os.close(descriptor)
    return {
        "path": str(target), "device": current.st_dev, "inode": current.st_ino,
        "owner_uid": current.st_uid, "mode": stat.S_IMODE(current.st_mode),
    }


def _decode_json(
    raw: bytes, *, path: Path, canonical_required: bool = True,
) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("release evidence must be one JSON object")
    if canonical_required and raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise ValueError("release evidence is not canonical JSON")
    return value


def _read_json(
    path: Path, *, canonical_required: bool = True,
) -> dict[str, object]:
    raw, _ = _stable_read_file(path)
    return _decode_json(
        raw, path=Path(path).absolute(),
        canonical_required=canonical_required,
    )


def _load_sealed(
    path: Path, *, fingerprint_field: str, schema: str | None = None,
    canonical_required: bool = True, return_identity: bool = False,
    private_parent: bool = False,
) -> (
    dict[str, object]
    | tuple[dict[str, object], dict[str, object]]
):
    target = path.absolute()
    raw, identity = _stable_read_file(
        target, private_parent=private_parent,
    )
    if (
        identity["owner_uid"] != os.getuid()
        or identity["nlink"] != 1
        or identity["mode"] != 0o444
    ):
        raise PermissionError(f"release evidence is not sealed: {target}")
    payload = _decode_json(
        raw, path=target, canonical_required=canonical_required,
    )
    body = dict(payload)
    fingerprint = body.pop(fingerprint_field, None)
    if (
        not isinstance(fingerprint, str) or _SHA.fullmatch(fingerprint) is None
        or fingerprint != stable_fingerprint(body)
        or (schema is not None and payload.get("schema_version") != schema)
    ):
        raise PermissionError(f"release evidence fingerprint/schema changed: {target}")
    if return_identity:
        return payload, identity
    return payload


def _write_sealed_bound(
    path: Path, body: Mapping[str, object], *, fingerprint_field: str,
) -> tuple[dict[str, object], dict[str, object]]:
    target = path.absolute()
    if fingerprint_field in body:
        raise ValueError("release output body already contains fingerprint")
    materialized = dict(body)
    payload = {**materialized, fingerprint_field: stable_fingerprint(materialized)}
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    parent_fd, parent_before = _open_stable_parent(target, private=True)
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o444, dir_fd=parent_fd,
        )
        os.fchmod(descriptor, 0o444)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("zero-byte release evidence write")
            offset += written
        opened = os.fstat(descriptor)
        linked = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _stat_snapshot(opened) != _stat_snapshot(linked)
            or opened.st_uid != os.getuid() or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o444
        ):
            raise PermissionError("release output identity is unsafe")
        os.fsync(descriptor)
        if os.pread(descriptor, len(encoded) + 1, 0) != encoded:
            raise RuntimeError("release output descriptor readback changed")
        after = os.fstat(descriptor)
        linked_after = os.stat(
            target.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            _stat_snapshot(after) != _stat_snapshot(opened)
            or _stat_snapshot(linked_after) != _stat_snapshot(after)
        ):
            raise PermissionError("release output changed during readback")
        os.fsync(descriptor)
        os.fsync(parent_fd)
        _verify_parent_generation(target.parent, parent_fd, parent_before)
        return payload, {
            "path": str(target),
            "file_sha256": hashlib.sha256(encoded).hexdigest(),
            "device": after.st_dev, "inode": after.st_ino,
            "owner_uid": after.st_uid, "mode": stat.S_IMODE(after.st_mode),
            "nlink": after.st_nlink,
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _write_sealed(
    path: Path, body: Mapping[str, object], *, fingerprint_field: str,
) -> dict[str, object]:
    return _write_sealed_bound(
        path, body, fingerprint_field=fingerprint_field,
    )[0]


def _no_payload(payload: Mapping[str, object], *, require_fields: bool = True) -> None:
    for field in ("D_R_payload_accessed", "D_V_payload_accessed", "D_T_payload_accessed"):
        if (require_fields and field not in payload) or payload.get(field) is not False:
            raise PermissionError("prelaunch evidence accessed a scientific payload")


def _bounded_lifetime(payload: Mapping[str, object], *, require_fresh: bool) -> None:
    try:
        issued = datetime.fromisoformat(str(payload["issued_at_utc"]).replace("Z", "+00:00"))
        expires = datetime.fromisoformat(str(payload["expires_at_utc"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as error:
        raise PermissionError("authorization time is malformed") from error
    now = datetime.now(timezone.utc)
    if (
        issued.tzinfo is None or expires.tzinfo is None or expires <= issued
        or expires - issued > timedelta(seconds=300)
        or (require_fresh and not issued <= now <= expires)
    ):
        raise PermissionError("authorization is stale or exceeds 300 seconds")


def _evidence_timestamp(
    payload: Mapping[str, object], field: str,
) -> datetime:
    value = payload.get(field)
    if not isinstance(value, str):
        raise PermissionError(f"{field} is not a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PermissionError(f"{field} is malformed") from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timedelta(0)
        or not value.endswith("Z")
    ):
        raise PermissionError(f"{field} is not canonical UTC")
    return parsed


def _file_binding(path: Path) -> dict[str, object]:
    target = path.absolute()
    _, observed = _stable_read_file(target)
    if observed["nlink"] != 1:
        raise PermissionError("release source binding is not a canonical regular file")
    return {
        key: observed[key] for key in (
            "path", "file_sha256", "device", "inode", "owner_uid", "mode",
        )
    }


def _directory_binding(path: Path) -> dict[str, object]:
    target = path.absolute()
    if target.resolve(strict=True) != target:
        raise PermissionError("runtime dependency site path is not canonical")
    parent_fd, parent_before = _open_stable_parent(target)
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        linked = os.stat(
            target.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _stat_snapshot(linked) != _stat_snapshot(opened)
        ):
            raise PermissionError("runtime dependency site is not a stable directory")
        _verify_parent_generation(target.parent, parent_fd, parent_before)
        return {
            "path": str(target), "device": opened.st_dev,
            "inode": opened.st_ino, "owner_uid": opened.st_uid,
            "owner_gid": opened.st_gid, "mode": stat.S_IMODE(opened.st_mode),
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _trusted_python_binding() -> dict[str, object]:
    binding = _file_binding(PYTHON_PATH)
    if (
        binding["path"] != str(PYTHON_PATH)
        or binding["owner_uid"] != 0
        or binding["mode"] != 0o755
    ):
        raise PermissionError("release python is not the fixed root-owned interpreter")
    return binding


def _runtime_dependency_site_binding() -> dict[str, object]:
    observed = _directory_binding(RUNTIME_DEPENDENCY_SITE_PATH)
    exact = {
        "path": str(RUNTIME_DEPENDENCY_SITE_PATH),
        "device": RUNTIME_DEPENDENCY_SITE_DEVICE,
        "inode": RUNTIME_DEPENDENCY_SITE_INODE,
        "owner_uid": RUNTIME_DEPENDENCY_SITE_OWNER_UID,
        "mode": RUNTIME_DEPENDENCY_SITE_MODE,
    }
    if (
        {key: observed[key] for key in exact} != exact
        or observed["owner_uid"] != os.getuid()
        or int(observed["mode"]) & 0o002
        or (
            int(observed["mode"]) & 0o020
            and observed["owner_gid"] != os.getgid()
        )
    ):
        raise PermissionError("runtime dependency site binding changed")
    return exact


def _realizer_python_source_fields(
    realization: Mapping[str, object],
) -> dict[str, object]:
    authorization = realization.get("authorization")
    receipt = realization.get("receipt")
    if not isinstance(authorization, Mapping) or not isinstance(receipt, Mapping):
        raise PermissionError("realization python binding closure is absent")
    authorization_bindings = authorization.get("executable_bindings")
    receipt_bindings = receipt.get("executable_bindings")
    if (
        not isinstance(authorization_bindings, Mapping)
        or not isinstance(receipt_bindings, Mapping)
        or not _deep_exact_equal(
            authorization_bindings.get("python"),
            receipt_bindings.get("python"),
        )
        or not isinstance(authorization_bindings.get("python"), Mapping)
    ):
        raise PermissionError("realization python binding changed across receipt")
    realized = dict(authorization_bindings["python"])
    live = _trusted_python_binding()
    expected_realizer = {
        "path": live["path"], "resolved_path": live["path"],
        "path_is_symlink": False, "file_sha256": live["file_sha256"],
        "device": live["device"], "inode": live["inode"],
        "owner_uid": live["owner_uid"], "mode": live["mode"],
    }
    if not _deep_exact_equal(realized, expected_realizer):
        raise PermissionError("realization python binding differs from live release")
    return {
        "python_path": live["path"],
        "python_file_sha256": live["file_sha256"],
        "python_device": live["device"], "python_inode": live["inode"],
        "python_owner_uid": live["owner_uid"], "python_mode": live["mode"],
    }


def _collect_source_bindings(
    realization: Mapping[str, object] | None = None,
    expected_prior_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if realization is None:
        live_python = _trusted_python_binding()
        python_fields = {
            "python_path": live_python["path"],
            "python_file_sha256": live_python["file_sha256"],
            "python_device": live_python["device"],
            "python_inode": live_python["inode"],
            "python_owner_uid": live_python["owner_uid"],
            "python_mode": live_python["mode"],
        }
    else:
        python_fields = _realizer_python_source_fields(realization)
    site = _runtime_dependency_site_binding()
    supervisor_binding = _file_binding(SUPERVISOR_PATH)
    adapter_binding = _file_binding(ADAPTER_PATH)
    prior_binding = _file_binding(PRIOR_R1_INTERRUPTION_RECEIPT_PATH)
    if (
        expected_prior_identity is not None
        and any(
            prior_binding[key] != expected_prior_identity[key]
            for key in (
                "file_sha256", "device", "inode", "owner_uid", "mode",
            )
        )
    ):
        raise PermissionError("prior attempt receipt identity was replaced")
    environment_binding = _file_binding(RUNTIME_ENVIRONMENT_PATH)
    legacy_binding = _file_binding(LEGACY_GATE_PATH)
    return {
        "supervisor_file_sha256": supervisor_binding["file_sha256"],
        "child_entry_file_sha256": adapter_binding["file_sha256"],
        "prior_attempt_receipt_file_sha256": prior_binding["file_sha256"],
        "runtime_environment_file_sha256": environment_binding["file_sha256"],
        "r2_adapter_path": adapter_binding["path"],
        "r2_adapter_file_sha256": adapter_binding["file_sha256"],
        "legacy_gate_entrypoint_path": legacy_binding["path"],
        "legacy_gate_entrypoint_file_sha256": legacy_binding["file_sha256"],
        **python_fields,
        "runtime_dependency_site_path": site["path"],
        "runtime_dependency_site_device": site["device"],
        "runtime_dependency_site_inode": site["inode"],
        "runtime_dependency_site_owner_uid": site["owner_uid"],
        "runtime_dependency_site_mode": site["mode"],
    }


def _validate_source_bindings(
    value: object, *, realization: Mapping[str, object] | None = None,
    expected_prior_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = _collect_source_bindings(
        realization,
        expected_prior_identity=expected_prior_identity,
    )
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise PermissionError("actual runtime source bindings changed")
    return expected


def _default_summary_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), shell=False, check=False, capture_output=True, text=True,
        cwd=REPOSITORY, timeout=60.0,
        env={"HOME": os.environ.get("HOME", "/tmp"), "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )


def _validate_identity_summary(runner: _CommandRunner) -> dict[str, object]:
    argv = (
        str(PYTHON_PATH), "-I", "-S", "-B", str(ADAPTER_PATH),
        "--r2-execution-identity-summary",
    )
    completed = runner(argv)
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("r2 adapter identity summary failed")
    try:
        summary = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("r2 adapter identity summary is not JSON") from error
    if not isinstance(summary, dict):
        raise ValueError("r2 adapter identity summary is not an object")
    body = dict(summary)
    fingerprint = body.pop("execution_identity_fingerprint", None)
    if (
        fingerprint != stable_fingerprint(body)
        or summary.get("schema_version")
        != "cure-lite-v24-D_R-structural-r2-execution-identity-v1"
        or summary.get("attempt_ordinal") != 2
        or summary.get("prior_attempt_count") != 1
        or summary.get("prior_attempt_status")
        != "OBSERVABILITY_LOST_NO_AUTHENTICATED_DECISION"
        or summary.get("frozen_scientific_path_count") != 103
        or summary.get("frozen_scientific_source_closure_fingerprint")
        != SOURCE_CLOSURE_FINGERPRINT_103
        or summary.get("numerical_or_scientific_change_authorized") is not False
        or summary.get("D_R_payload_authorized_by_adapter") is not False
        or summary.get("D_V_payload_authorized") is not False
        or summary.get("D_T_payload_authorized") is not False
        or summary.get("training_authorized") is not False
        or summary.get("optimizer_steps_authorized") != 0
        or summary.get("parameter_updates_authorized") != 0
        or summary.get("automatic_retry_allowed") is not False
        or summary.get("resume_allowed") is not False
    ):
        raise PermissionError("r2 adapter identity/103 closure changed")
    return summary


def _manager_generation_from_inventory(receipt: Mapping[str, object]) -> dict[str, object]:
    inventory = receipt.get("inventory")
    manager = inventory.get("manager") if isinstance(inventory, Mapping) else None
    if not isinstance(inventory, Mapping) or not isinstance(manager, Mapping):
        raise PermissionError("environment inventory manager generation is absent")
    return {
        "boot_id": inventory.get("boot_id"),
        "identity": manager.get("identity"), "endpoint": manager.get("endpoint"),
    }


def _environment_binding_from_inventory(
    inventory: Mapping[str, object],
) -> dict[str, object]:
    manager = inventory.get("manager")
    endpoint = manager.get("endpoint") if isinstance(manager, Mapping) else None
    identity = manager.get("identity") if isinstance(manager, Mapping) else None
    if (
        not isinstance(manager, Mapping)
        or not isinstance(endpoint, Mapping)
        or not isinstance(identity, Mapping)
    ):
        raise PermissionError(
            "environment inventory manager binding is absent"
        )
    return {
        "inventory_fingerprint": inventory.get("inventory_fingerprint"),
        "boot_id": inventory.get("boot_id"),
        "runtime_directory": endpoint.get("runtime_directory"),
        "runtime_directory_device": endpoint.get(
            "runtime_directory_device"
        ),
        "runtime_directory_inode": endpoint.get(
            "runtime_directory_inode"
        ),
        "manager_identity": dict(identity),
    }


def _validate_audit_receipt(
    path: Path, *, passed: bool, return_identity: bool = False,
) -> (
    dict[str, object]
    | tuple[dict[str, object], dict[str, object]]
):
    receipt, identity = _load_sealed(
        path, fingerprint_field="receipt_fingerprint",
        schema=runtime_environment.ENVIRONMENT_RECEIPT_SCHEMA,
        return_identity=True,
    )
    inventory = receipt.get("inventory")
    if (
        set(receipt) != _AUDIT_RECEIPT_KEYS
        or receipt.get("command") != "audit-only"
        or receipt.get("passed") is not passed
        or not isinstance(inventory, Mapping) or inventory.get("passed") is not passed
    ):
        raise PermissionError("environment audit PASS/FAIL semantics changed")
    _no_payload(receipt)
    _no_payload(inventory)
    nested = dict(inventory)
    fingerprint = nested.pop("inventory_fingerprint", None)
    if fingerprint != stable_fingerprint(nested):
        raise PermissionError("nested environment inventory fingerprint changed")
    if passed:
        if inventory.get("blockers") != []:
            raise PermissionError("postcleanup audit still has blockers")
    elif inventory.get("blockers") != [cleanup.PRECLEANUP_BLOCKER]:
        raise PermissionError("precleanup FAIL is not the exact known blocker")
    if return_identity:
        return receipt, identity
    return receipt


def _validate_cleanup_prerequisites(
    *, precleanup_path: Path, plan_path: Path, authorization_path: Path,
) -> tuple[
    dict[str, object], dict[str, object], dict[str, object],
    dict[str, dict[str, object]],
]:
    precleanup, precleanup_identity = _validate_audit_receipt(
        precleanup_path, passed=False, return_identity=True,
    )
    plan, plan_identity = _load_sealed(
        plan_path, fingerprint_field="plan_fingerprint", schema=cleanup.PLAN_SCHEMA,
        return_identity=True,
    )
    cleanup.validate_plan(plan)
    authorization, authorization_identity = _load_sealed(
        authorization_path, fingerprint_field="authorization_fingerprint",
        schema=cleanup.AUTHORIZATION_SCHEMA, return_identity=True,
    )
    if set(authorization) != cleanup._AUTHORIZATION_KEYS:
        raise PermissionError("cleanup authorization schema keys changed")
    _bounded_lifetime(authorization, require_fresh=False)
    _no_payload(authorization)
    if (
        plan.get("inventory_receipt_path") != str(precleanup_path.absolute())
        or plan.get("inventory_receipt_file_sha256")
        != precleanup_identity["file_sha256"]
        or plan.get("inventory_receipt_fingerprint") != precleanup["receipt_fingerprint"]
        or authorization.get("plan_path") != str(plan_path.absolute())
        or authorization.get("plan_file_sha256")
        != plan_identity["file_sha256"]
        or authorization.get("plan_fingerprint") != plan["plan_fingerprint"]
        or authorization.get("fresh_cleanup_authorized") is not True
        or authorization.get("persistent_disable_authorized") is not False
        or authorization.get("global_reset_failed_authorized") is not False
        or authorization.get("authorized_uid") != os.getuid()
    ):
        raise PermissionError("cleanup plan/authorization lineage changed")
    return precleanup, plan, authorization, {
        "precleanup": precleanup_identity,
        "plan": plan_identity,
        "authorization": authorization_identity,
    }


def _validate_normal_cleanup_chain(
    *, precleanup_path: Path, plan_path: Path, authorization_path: Path,
    receipt_path: Path,
) -> dict[str, dict[str, object]]:
    """Retained for isolated legacy tests; production release never calls it."""
    precleanup, plan, authorization, identities = _validate_cleanup_prerequisites(
        precleanup_path=precleanup_path, plan_path=plan_path,
        authorization_path=authorization_path,
    )
    receipt, receipt_identity = _load_sealed(
        receipt_path, fingerprint_field="cleanup_receipt_fingerprint",
        schema=cleanup.FINAL_RECEIPT_SCHEMA, return_identity=True,
    )
    _no_payload(receipt)
    intent_path = receipt_path.parent / "cleanup-intent.json"
    intent, intent_identity = _load_sealed(
        intent_path, fingerprint_field="intent_fingerprint", schema=cleanup.INTENT_SCHEMA,
        return_identity=True,
    )
    _no_payload(intent)
    if (
        receipt.get("passed") is not True
        or intent.get("plan_file_sha256") != identities["plan"]["file_sha256"]
        or intent.get("plan_fingerprint") != plan["plan_fingerprint"]
        or intent.get("authorization_file_sha256")
        != identities["authorization"]["file_sha256"]
        or intent.get("authorization_fingerprint") != authorization["authorization_fingerprint"]
        or receipt.get("intent_fingerprint") != intent["intent_fingerprint"]
        or receipt.get("boot_id") != plan.get("boot_id")
        or receipt.get("manager_generation") != plan.get("manager_generation")
    ):
        raise PermissionError("cleanup PASS receipt lineage changed")
    action_fingerprints = receipt.get("action_receipt_fingerprints")
    if not isinstance(action_fingerprints, list) or len(action_fingerprints) != len(plan["actions"]):
        raise PermissionError("cleanup action receipt closure is incomplete")
    for ordinal, expected in enumerate(action_fingerprints):
        action = _load_sealed(
            receipt_path.parent / f"action-{ordinal:03d}.json",
            fingerprint_field="action_receipt_fingerprint",
            schema=cleanup.ACTION_RECEIPT_SCHEMA,
        )
        _no_payload(action)
        if (
            action["action_receipt_fingerprint"] != expected
            or action.get("intent_fingerprint") != intent["intent_fingerprint"]
            or action.get("action") != plan["actions"][ordinal]
            or action.get("returncode") != 0
        ):
            raise PermissionError("cleanup action receipt lineage changed")
    return {
        "precleanup": precleanup, "plan": plan,
        "authorization": authorization, "receipt": receipt,
        "identities": {
            **identities, "intent": intent_identity,
            "receipt": receipt_identity,
        },
    }


def _cleanup_no_payload(payload: Mapping[str, object]) -> None:
    _no_payload(payload)
    if payload.get("payload_authority") != "none":
        raise PermissionError("cleanup recovery evidence has payload authority")


def _cleanup_timestamp(value: object, *, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise PermissionError(f"{name} is malformed") from error
    if parsed.tzinfo is None:
        raise PermissionError(f"{name} is not timezone-aware")
    return parsed


def _validate_archived_recovery_executable_bindings(
    value: object,
) -> None:
    """Validate historical tool identities without pretending all are current.

    The recovery authorization was valid at dispatch and is sealed into the
    intent/action/final lineage.  The environment auditor was subsequently
    revised to encode the observed recovery-mode stability semantics, so its
    historical SHA is evidence rather than a requirement on the current file.
    The mutating cleanup tool and fixed executables, however, must still match.
    """

    if not isinstance(value, Mapping):
        raise PermissionError("archived recovery executable bindings changed")
    bindings = {str(name): dict(binding) for name, binding in value.items()}
    expected_current = cleanup._executable_bindings()
    if set(bindings) != set(expected_current):
        raise PermissionError("archived recovery executable binding set changed")
    for name, binding in bindings.items():
        expected = expected_current[name]
        if (
            set(binding) != {"path", "file_sha256"}
            or binding.get("path") != expected["path"]
            or not isinstance(binding.get("file_sha256"), str)
            or _SHA.fullmatch(str(binding["file_sha256"])) is None
        ):
            raise PermissionError(
                f"archived recovery executable binding changed:{name}"
            )
        if (
            name != "environment_auditor"
            and binding["file_sha256"] != expected["file_sha256"]
        ):
            raise PermissionError(
                f"archived recovery executable changed:{name}"
            )


def _load_cleanup_lineage_root(
    value: object, *, name: str, fingerprint_field: str, schema: str,
    exact_keys: set[str], expected_path: Path | None = None,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != cleanup._EVIDENCE_ROOT_KEYS:
        raise PermissionError(f"cleanup recovery {name} root keys changed")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise PermissionError(f"cleanup recovery {name} root path changed")
    target = Path(raw_path).absolute()
    if raw_path != str(target) or (
        expected_path is not None and target != expected_path.absolute()
    ):
        raise PermissionError(f"cleanup recovery {name} root path changed")
    if value.get("fingerprint_field") != fingerprint_field:
        raise PermissionError(f"cleanup recovery {name} fingerprint field changed")
    payload, identity = _load_sealed(
        target, fingerprint_field=fingerprint_field, schema=schema,
        return_identity=True, private_parent=True,
    )
    if (
        set(payload) != exact_keys
        or value.get("file_sha256") != identity["file_sha256"]
        or value.get("fingerprint") != payload[fingerprint_field]
    ):
        raise PermissionError(f"cleanup recovery {name} root binding changed")
    return target, payload, identity


def _validate_cleanup_chain(
    *, precleanup_path: Path, plan_path: Path, authorization_path: Path,
    receipt_path: Path,
) -> dict[str, dict[str, object]]:
    """Verify only the authoritative partial-mask recovery cleanup closure."""
    (
        precleanup, plan, original_authorization, prerequisite_identities,
    ) = _validate_cleanup_prerequisites(
        precleanup_path=precleanup_path, plan_path=plan_path,
        authorization_path=authorization_path,
    )
    expected_actions = [
        {
            "ordinal": 0, "unit_name": cleanup.GPU0_CONFLICT_UNIT,
            "action": "mask-runtime",
        },
        {
            "ordinal": 1, "unit_name": cleanup.GPU0_CONFLICT_UNIT,
            "action": "stop",
        },
    ]
    if plan.get("actions") != expected_actions:
        raise PermissionError("cleanup recovery plan is not mask then one stop")

    receipt, receipt_identity = _load_sealed(
        receipt_path, fingerprint_field="cleanup_receipt_fingerprint",
        schema=cleanup.FINAL_RECEIPT_SCHEMA, return_identity=True,
        private_parent=True,
    )
    cleanup.validate_final_cleanup_receipt(receipt)
    _cleanup_no_payload(receipt)
    if receipt.get("cleanup_mode") != cleanup.RECOVERY_CLEANUP_MODE:
        raise PermissionError("production cleanup is not the recovery final receipt")
    lineage = receipt.get("partial_lineage")
    if not isinstance(lineage, Mapping):
        raise PermissionError("cleanup recovery flat partial lineage is absent")

    plan_root_path, rooted_plan, rooted_plan_identity = _load_cleanup_lineage_root(
        lineage.get("plan"), name="plan", fingerprint_field="plan_fingerprint",
        schema=cleanup.PLAN_SCHEMA, exact_keys=cleanup._PLAN_KEYS,
        expected_path=plan_path,
    )
    (
        original_authorization_root_path,
        rooted_original_authorization,
        rooted_original_authorization_identity,
    ) = (
        _load_cleanup_lineage_root(
            lineage.get("original_authorization"),
            name="original authorization",
            fingerprint_field="authorization_fingerprint",
            schema=cleanup.AUTHORIZATION_SCHEMA,
            exact_keys=cleanup._AUTHORIZATION_KEYS,
            expected_path=authorization_path,
        )
    )
    (
        original_intent_path, original_intent, original_intent_identity,
    ) = _load_cleanup_lineage_root(
        lineage.get("original_intent"), name="original intent",
        fingerprint_field="intent_fingerprint", schema=cleanup.INTENT_SCHEMA,
        exact_keys=cleanup._INTENT_KEYS,
    )
    (
        original_terminal_path, original_terminal, original_terminal_identity,
    ) = _load_cleanup_lineage_root(
        lineage.get("original_terminal_failure"),
        name="original terminal failure",
        fingerprint_field="terminal_failure_fingerprint",
        schema=cleanup.TERMINAL_FAILURE_SCHEMA,
        exact_keys=cleanup._TERMINAL_FAILURE_KEYS,
    )
    if rooted_plan != plan or rooted_original_authorization != original_authorization:
        raise PermissionError("cleanup recovery prerequisite roots changed")

    original = cleanup.validate_partial_cleanup_failure_lineage(
        plan_path=plan_root_path,
        original_authorization_path=original_authorization_root_path,
        intent_path=original_intent_path,
        terminal_failure_path=original_terminal_path,
    )
    terminal_inflight = original_terminal.get("inflight_action")
    if (
        original.get("plan") != plan
        or original.get("authorization") != original_authorization
        or original.get("intent") != original_intent
        or original.get("terminal_failure") != original_terminal
        or original_terminal.get("runtime_mask_may_remain") is not False
        or original_terminal.get("completed_action_receipt_fingerprints") != []
        or not isinstance(terminal_inflight, Mapping)
        or terminal_inflight.get("action") != expected_actions[0]
        or terminal_inflight.get("argv") != [
            cleanup.SYSTEMCTL_PATH, "--user", "mask", "--runtime",
            cleanup.GPU0_CONFLICT_UNIT,
        ]
        or terminal_inflight.get("dispatch_attempted") is not True
        or terminal_inflight.get("completion_observed") is not True
        or terminal_inflight.get("returncode") != 0
    ):
        raise PermissionError("original cleanup was not the exact rc0 mask/no-stop failure")

    (
        recovery_authorization_path,
        recovery_authorization,
        recovery_authorization_identity,
    ) = (
        _load_cleanup_lineage_root(
            lineage.get("recovery_authorization"),
            name="recovery authorization",
            fingerprint_field="recovery_authorization_fingerprint",
            schema=cleanup.RECOVERY_AUTHORIZATION_SCHEMA,
            exact_keys=cleanup._RECOVERY_AUTHORIZATION_KEYS,
        )
    )
    (
        recovery_intent_path, recovery_intent, recovery_intent_identity,
    ) = _load_cleanup_lineage_root(
        lineage.get("recovery_intent"), name="recovery intent",
        fingerprint_field="recovery_intent_fingerprint",
        schema=cleanup.RECOVERY_INTENT_SCHEMA,
        exact_keys=cleanup._RECOVERY_INTENT_KEYS,
    )
    (
        recovery_action_path, recovery_action, recovery_action_identity,
    ) = _load_cleanup_lineage_root(
        lineage.get("recovery_action_receipt"),
        name="recovery action receipt",
        fingerprint_field="recovery_action_receipt_fingerprint",
        schema=cleanup.RECOVERY_ACTION_RECEIPT_SCHEMA,
        exact_keys=cleanup._RECOVERY_ACTION_RECEIPT_KEYS,
    )
    del recovery_intent_path, recovery_action_path
    for payload in (recovery_authorization, recovery_intent, recovery_action):
        _cleanup_no_payload(payload)

    original_roots = {
        key: lineage[key] for key in (
            "plan", "original_authorization", "original_intent",
            "original_terminal_failure",
        )
    }
    if original.get("roots") != original_roots:
        raise PermissionError("original cleanup sealed roots changed")
    authorization_guard = cleanup._validate_recovery_guard(
        recovery_authorization.get("activation_guard")
    )
    _validate_archived_recovery_executable_bindings(
        recovery_authorization.get("executable_bindings")
    )
    _bounded_lifetime(recovery_authorization, require_fresh=False)
    if (
        recovery_authorization.get("candidate") != CANDIDATE
        or recovery_authorization.get("scope")
        != "user-systemd-partial-cleanup-recovery"
        or recovery_authorization.get("roots") != original_roots
        or recovery_authorization.get("authorized_action") != expected_actions[1]
        or recovery_authorization.get("partial_failure_condition")
        != cleanup.PARTIAL_FAILURE_CONDITION
        or recovery_authorization.get("explicit_user_instruction_id")
        != cleanup.RECOVERY_USER_INSTRUCTION_ID
        or recovery_authorization.get("authorized_uid") != os.getuid()
        or recovery_authorization.get("manager_generation")
        != plan.get("manager_generation")
        or recovery_authorization.get("persistent_disable_authorized") is not False
        or recovery_authorization.get("global_reset_failed_authorized") is not False
        or recovery_authorization.get("automatic_retry_authorized") is not False
        or not isinstance(recovery_authorization.get("authorization_basis"), str)
        or not str(recovery_authorization["authorization_basis"]).strip()
    ):
        raise PermissionError("archived cleanup recovery authorization changed")

    recovery_before = recovery_authorization.get("before")
    if (
        not isinstance(recovery_before, Mapping)
        or set(recovery_before)
        != {cleanup.GPU0_CONFLICT_UNIT, cleanup.GPU2_DNANET_UNIT}
    ):
        raise PermissionError("cleanup recovery before snapshots changed")
    if (
        recovery_intent.get("roots") != original_roots
        or recovery_intent.get("recovery_authorization_file_sha256")
        != recovery_authorization_identity["file_sha256"]
        or recovery_intent.get("recovery_authorization_fingerprint")
        != recovery_authorization["recovery_authorization_fingerprint"]
        or recovery_intent.get("manager_generation") != plan.get("manager_generation")
        or recovery_intent.get("before") != recovery_before
        or recovery_intent.get("activation_guard") != authorization_guard
        or recovery_intent.get("action") != expected_actions[1]
    ):
        raise PermissionError("cleanup recovery intent lineage changed")

    issued = _cleanup_timestamp(
        recovery_authorization.get("issued_at_utc"), name="recovery authorization issuance",
    )
    expires = _cleanup_timestamp(
        recovery_authorization.get("expires_at_utc"), name="recovery authorization expiry",
    )
    intent_created = _cleanup_timestamp(
        recovery_intent.get("created_at_utc"), name="recovery intent creation",
    )
    action_started = _cleanup_timestamp(
        recovery_action.get("started_at_utc"), name="recovery action start",
    )
    action_created = _cleanup_timestamp(
        recovery_action.get("created_at_utc"), name="recovery action creation",
    )
    if not (
        issued
        <= intent_created
        <= action_started
        <= action_created
        <= expires
    ):
        raise PermissionError("cleanup recovery chronology changed")

    expected_stop_argv = [
        cleanup.SYSTEMCTL_PATH, "--user", "stop", cleanup.GPU0_CONFLICT_UNIT,
    ]
    action_before = recovery_action.get("before")
    action_after = recovery_action.get("after")
    protected_before = recovery_action.get("protected_before")
    protected_after = recovery_action.get("protected_after")
    if not isinstance(action_before, Mapping):
        raise PermissionError("cleanup recovery action-before snapshot changed")
    action_guard_before = cleanup._validate_recovery_guard(
        recovery_action.get("activation_guard_before")
    )
    action_guard_after = cleanup._validate_recovery_guard(
        recovery_action.get("activation_guard_after")
    )
    if (
        recovery_action.get("recovery_intent_fingerprint")
        != recovery_intent["recovery_intent_fingerprint"]
        or recovery_action.get("action") != expected_actions[1]
        or recovery_action.get("argv") != expected_stop_argv
        or recovery_action.get("shell") is not False
        or recovery_action.get("returncode") != 0
        or not isinstance(recovery_action.get("stdout"), str)
        or not isinstance(recovery_action.get("stderr"), str)
        or recovery_action.get("manager_generation") != plan.get("manager_generation")
        or action_before != recovery_before[cleanup.GPU0_CONFLICT_UNIT]
        or protected_before != recovery_before[cleanup.GPU2_DNANET_UNIT]
        or protected_after != protected_before
        or action_guard_before != authorization_guard
        or action_guard_after != authorization_guard
        or not isinstance(action_after, Mapping)
        or action_after.get("UnitFileState") != "enabled"
        or action_after.get("ActiveState") != "inactive"
        or action_after.get("SubState") != "dead"
        or action_after.get("TriggeredBy") != ""
        or action_after.get("Triggers") != ""
        or action_after.get("NRestarts") != action_before.get("NRestarts")
    ):
        raise PermissionError("cleanup recovery is not the unique rc0 stop action")

    final_guard = cleanup._validate_recovery_guard(receipt.get("activation_guard"))
    final_after = receipt.get("after")
    if (
        receipt.get("intent_fingerprint")
        != recovery_intent["recovery_intent_fingerprint"]
        or receipt.get("action_receipt_fingerprints") != [
            recovery_action["recovery_action_receipt_fingerprint"]
        ]
        or receipt.get("boot_id") != plan.get("boot_id")
        or receipt.get("manager_generation") != plan.get("manager_generation")
        or final_guard != authorization_guard
        or not isinstance(final_after, Mapping)
        or final_after.get(cleanup.GPU0_CONFLICT_UNIT) != action_after
        or final_after.get(cleanup.GPU2_DNANET_UNIT) != protected_after
        or lineage.get("legacy_runtime_mask_may_remain_false_reconciled") is not True
        or lineage.get("original_stop_dispatched") is not False
        or lineage.get("recovery_authorization")
        != {
            "path": str(recovery_authorization_path),
            "file_sha256": recovery_authorization_identity["file_sha256"],
            "fingerprint_field": "recovery_authorization_fingerprint",
            "fingerprint": recovery_authorization[
                "recovery_authorization_fingerprint"
            ],
        }
    ):
        raise PermissionError("cleanup recovery final receipt lineage changed")
    return {
        "precleanup": precleanup, "plan": plan,
        "authorization": original_authorization,
        "original_intent": original_intent,
        "original_terminal_failure": original_terminal,
        "recovery_authorization": recovery_authorization,
        "recovery_intent": recovery_intent,
        "recovery_action_receipt": recovery_action,
        "receipt": receipt,
        "identities": {
            **prerequisite_identities,
            "receipt": receipt_identity,
            "rooted_plan": rooted_plan_identity,
            "rooted_original_authorization":
                rooted_original_authorization_identity,
            "original_intent": original_intent_identity,
            "original_terminal_failure": original_terminal_identity,
            "recovery_authorization": recovery_authorization_identity,
            "recovery_intent": recovery_intent_identity,
            "recovery_action_receipt": recovery_action_identity,
        },
    }


def _validate_environment_chain(
    *, policy_path: Path, precleanup_path: Path, cleanup_receipt_path: Path,
    stability_path: Path, postcleanup_path: Path,
) -> dict[str, dict[str, object]]:
    policy, policy_identity = _load_sealed(
        policy_path, fingerprint_field="policy_fingerprint",
        schema=runtime_environment.ENVIRONMENT_POLICY_SCHEMA,
        return_identity=True,
    )
    runtime_environment.validate_environment_policy(policy)
    _no_payload(policy)
    stability, stability_identity = _load_sealed(
        stability_path, fingerprint_field="stability_receipt_fingerprint",
        schema=runtime_environment.ENVIRONMENT_STABILITY_RECEIPT_SCHEMA,
        return_identity=True,
    )
    runtime_environment.validate_environment_stability_receipt(stability)
    _no_payload(stability)
    postcleanup, postcleanup_identity = _validate_audit_receipt(
        postcleanup_path, passed=True, return_identity=True,
    )
    roots = stability.get("root_evidence")
    precleanup, precleanup_identity = _validate_audit_receipt(
        precleanup_path, passed=False, return_identity=True,
    )
    cleanup_receipt, cleanup_receipt_identity = _load_sealed(
        cleanup_receipt_path, fingerprint_field="cleanup_receipt_fingerprint",
        schema=cleanup.FINAL_RECEIPT_SCHEMA, return_identity=True,
    )
    samples = stability.get("samples")
    sample_count = stability.get("sample_count")
    if (
        stability.get("receipt_kind") != "sampled"
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not isinstance(samples, list)
        or sample_count != len(samples)
        or not samples
        or not isinstance(samples[-1], Mapping)
        or not isinstance(samples[-1].get("inventory"), Mapping)
        or samples[-1].get("passed") is not True
        or samples[-1].get("blockers") != []
    ):
        raise PermissionError(
            "environment stability final sample is not a closed PASS"
        )
    final_inventory = samples[-1]["inventory"]
    postcleanup_inventory = postcleanup.get("inventory")
    if (
        not isinstance(postcleanup_inventory, Mapping)
        or not _deep_exact_equal(
            postcleanup_inventory,
            final_inventory,
        )
    ):
        raise PermissionError(
            "postcleanup inventory is not the exact final stability sample"
        )
    expected_environment_binding = _environment_binding_from_inventory(
        final_inventory
    )
    if not _deep_exact_equal(
        postcleanup.get("environment_binding"),
        expected_environment_binding,
    ):
        raise PermissionError(
            "postcleanup environment binding differs from final stability sample"
        )
    if (
        stability.get("passed") is not True or stability.get("blockers") != []
        or not isinstance(roots, Mapping)
        or roots.get("precleanup_inventory_receipt", {}).get("file_sha256")
        != precleanup_identity["file_sha256"]
        or roots.get("precleanup_inventory_receipt", {}).get("receipt_fingerprint")
        != precleanup["receipt_fingerprint"]
        or roots.get("cleanup_receipt", {}).get("file_sha256")
        != cleanup_receipt_identity["file_sha256"]
        or roots.get("cleanup_receipt", {}).get("cleanup_receipt_fingerprint")
        != cleanup_receipt["cleanup_receipt_fingerprint"]
        or roots.get("policy", {}).get("file_sha256")
        != policy_identity["file_sha256"]
        or roots.get("policy", {}).get("policy_fingerprint")
        != policy["policy_fingerprint"]
    ):
        raise PermissionError("environment stability root lineage changed")
    contract = stability.get("contract")
    selected = policy.get("selected_gpu")
    manager = _manager_generation_from_inventory(postcleanup)
    if (
        not isinstance(contract, Mapping) or not isinstance(selected, Mapping)
        or contract.get("selected_gpu_uuid") != selected.get("uuid")
        or contract.get("selected_gpu_pci_bus_id") != selected.get("pci_bus_id")
        or contract.get("selected_gpu_minor_number") != selected.get("minor_number")
        or manager.get("boot_id") != contract.get("boot_id")
        or manager.get("identity") != {
            "pid": contract.get("manager_pid"),
            "starttime_ticks": contract.get("manager_starttime_ticks"),
            "uid": contract.get("uid"),
            "control_group": contract.get("manager_control_group"),
        }
    ):
        raise PermissionError("postcleanup audit differs from stability contract")
    return {
        "policy": policy, "stability": stability, "postcleanup": postcleanup,
        "identities": {
            "policy": policy_identity,
            "stability": stability_identity,
            "postcleanup": postcleanup_identity,
            "precleanup": precleanup_identity,
            "cleanup_receipt": cleanup_receipt_identity,
        },
    }


def _require_exact_int(
    value: object, *, name: str, minimum: int,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise PermissionError(f"{name} is not an exact integer identity")
    return value


def _validate_realization_file_binding(
    binding: object, *, name: str,
) -> bytes:
    if type(binding) is not dict or set(binding) != (
        actual_realizer._FILE_BINDING_KEYS
    ):
        raise PermissionError(f"{name} binding schema is not type-exact")
    if (
        not isinstance(binding.get("path"), str)
        or not isinstance(binding.get("resolved_path"), str)
        or type(binding.get("path_is_symlink")) is not bool
        or not isinstance(binding.get("file_sha256"), str)
        or _SHA.fullmatch(str(binding.get("file_sha256"))) is None
    ):
        raise PermissionError(f"{name} binding scalar types changed")
    _require_exact_int(
        binding.get("device"), name=f"{name} device", minimum=0,
    )
    _require_exact_int(
        binding.get("inode"), name=f"{name} inode", minimum=1,
    )
    _require_exact_int(
        binding.get("owner_uid"), name=f"{name} owner uid", minimum=0,
    )
    _require_exact_int(
        binding.get("mode"), name=f"{name} mode",
        minimum=0, maximum=0o7777,
    )
    raw, observed = actual_realizer._read_file_binding(
        Path(binding["path"]),
        allow_symlink=binding["path_is_symlink"],
    )
    if not _deep_exact_equal(observed, binding):
        raise PermissionError(f"{name} sealed/live binding changed")
    return raw


def _validate_realization_fragment_identity(
    fragment_identity: object, *, expected_path: Path,
) -> dict[str, object]:
    if (
        type(fragment_identity) is not dict
        or set(fragment_identity) != _REALIZATION_FRAGMENT_IDENTITY_KEYS
        or not isinstance(fragment_identity.get("path"), str)
        or fragment_identity.get("path") != str(expected_path.absolute())
        or not isinstance(fragment_identity.get("file_sha256"), str)
        or _SHA.fullmatch(str(fragment_identity.get("file_sha256"))) is None
    ):
        raise PermissionError("actual unit fragment identity schema changed")
    _require_exact_int(
        fragment_identity.get("device"),
        name="actual fragment device", minimum=0,
    )
    _require_exact_int(
        fragment_identity.get("inode"),
        name="actual fragment inode", minimum=1,
    )
    owner_uid = _require_exact_int(
        fragment_identity.get("owner_uid"),
        name="actual fragment owner uid", minimum=0,
    )
    mode = _require_exact_int(
        fragment_identity.get("mode"),
        name="actual fragment mode", minimum=0, maximum=0o7777,
    )
    nlink = _require_exact_int(
        fragment_identity.get("nlink"),
        name="actual fragment link count", minimum=1,
    )
    if owner_uid != os.getuid() or mode != 0o600 or nlink != 1:
        raise PermissionError("actual unit fragment identity is unsafe")
    return fragment_identity


def _validate_integration_supervisor_evidence(
    evidence: object,
) -> dict[str, object]:
    if type(evidence) is not dict or set(evidence) != (
        _INTEGRATION_SUPERVISOR_EVIDENCE_KEYS
    ):
        raise PermissionError("integration supervisor evidence is not CLOSED")
    if (
        re.fullmatch(r"[0-9a-f]{32}", str(evidence["invocation_id"])) is None
        or any(
            not isinstance(evidence[field], str)
            or _SHA.fullmatch(evidence[field]) is None
            for field in (
                _INTEGRATION_SUPERVISOR_EVIDENCE_KEYS
                - {
                    "invocation_id", "runtime_attestation_absent",
                    "gpu_lease_evidence_absent",
                }
            )
        )
        or evidence["runtime_attestation_absent"] is not True
        or evidence["gpu_lease_evidence_absent"] is not True
    ):
        raise PermissionError(
            "integration supervisor evidence safety semantics changed"
        )
    return evidence


def _validate_integration_authorization_semantics(
    authorization: Mapping[str, object],
    *, expected_identity: Mapping[str, str],
) -> None:
    manager_generation = authorization.get("manager_generation")
    runtime_spec_binding = authorization.get("runtime_spec_binding")
    rendered_fragment = authorization.get("rendered_fragment")
    if (
        not isinstance(manager_generation, Mapping)
        or not isinstance(runtime_spec_binding, Mapping)
        or not isinstance(rendered_fragment, Mapping)
    ):
        raise PermissionError("integration authorization lineage is incomplete")
    integration._validate_manager_generation(manager_generation)
    authorized_uid = _require_exact_int(
        authorization.get("authorized_uid"),
        name="integration authorized uid", minimum=0,
    )
    del authorized_uid
    if (
        authorization.get("instruction_id") != integration.INSTRUCTION_ID
        or authorization.get("authorization_basis")
        != integration.AUTHORIZATION_BASIS
        or not _deep_exact_equal(
            authorization.get("identity"), expected_identity
        )
        or authorization.get("unit_realization_authorized") is not True
        or authorization.get("unit_removal_authorized") is not False
        or authorization.get("integration_authorized") is not True
        or authorization.get("actual_r2_authorized") is not False
        or authorization.get("direct_start_authorized") is not False
        or authorization.get("enable_authorized") is not False
        or authorization.get("payload_authority") != "none"
        or authorization.get("gpu_access_authorized") is not False
        or not isinstance(runtime_spec_binding.get(
            "runtime_spec_fingerprint"
        ), str)
        or _SHA.fullmatch(str(runtime_spec_binding.get(
            "runtime_spec_fingerprint"
        ))) is None
        or not isinstance(rendered_fragment.get("sha256"), str)
        or _SHA.fullmatch(str(rendered_fragment.get("sha256"))) is None
    ):
        raise PermissionError(
            "integration authorization safety semantics changed"
        )


def _validate_integration_terminal_semantics(
    terminal: Mapping[str, object], *,
    authorization: Mapping[str, object],
    expected_identity: Mapping[str, str],
) -> dict[str, object]:
    if set(terminal) != _INTEGRATION_TERMINAL_KEYS:
        raise PermissionError("integration terminal schema keys changed")
    _no_payload(terminal)
    evidence = _validate_integration_supervisor_evidence(
        terminal.get("supervisor_evidence")
    )
    created = _evidence_timestamp(terminal, "created_at_utc")
    issued = _evidence_timestamp(authorization, "issued_at_utc")
    expires = _evidence_timestamp(authorization, "expires_at_utc")
    runtime_spec_binding = authorization["runtime_spec_binding"]
    if (
        not issued <= created <= expires
        or terminal.get("scenario_id") != authorization["scenario_id"]
        or not _deep_exact_equal(
            terminal.get("identity"), expected_identity
        )
        or terminal.get("authorization_fingerprint")
        != authorization["authorization_fingerprint"]
        or terminal.get("runtime_spec_fingerprint")
        != runtime_spec_binding["runtime_spec_fingerprint"]
        or terminal.get("passed") is not True
        or not _deep_exact_equal(
            terminal.get("completed_actions"),
            _INTEGRATION_TERMINAL_ACTIONS,
        )
        or terminal.get("error_type") is not None
        or terminal.get("error_message") is not None
        or terminal.get("direct_systemctl_start_attempted") is not False
        or terminal.get("enable_attempted") is not False
        or terminal.get("remove_attempted") is not False
        or terminal.get("payload_authority") != "none"
        or terminal.get("gpu_accessed") is not False
    ):
        raise PermissionError("integration terminal is not exact safe PASS")
    return evidence


def _validate_integration_removal_authorization_semantics(
    removal: Mapping[str, object], *,
    authorization: Mapping[str, object],
    terminal: Mapping[str, object],
    expected_identity: Mapping[str, str],
    evidence: Mapping[str, object],
) -> None:
    if set(removal) != integration._REMOVAL_AUTH_KEYS:
        raise PermissionError("integration removal authorization schema keys changed")
    _bounded_lifetime(removal, require_fresh=False)
    _no_payload(removal)
    issued = _evidence_timestamp(removal, "issued_at_utc")
    terminal_created = _evidence_timestamp(terminal, "created_at_utc")
    authorization_expires = _evidence_timestamp(
        authorization, "expires_at_utc",
    )
    fragment_identity = removal.get("fragment_identity")
    inactive_state = removal.get("inactive_static_state")
    manager_generation = removal.get("manager_generation")
    unit_path_policy = removal.get("unit_path_policy")
    rendered_fragment = authorization["rendered_fragment"]
    fragment_path = str(
        Path(str(authorization["unit_directory"]))
        / expected_identity["unit_name"]
    )
    if (
        type(fragment_identity) is not dict
        or set(fragment_identity) != _INTEGRATION_FRAGMENT_IDENTITY_KEYS
        or type(inactive_state) is not dict
        or not isinstance(manager_generation, Mapping)
        or not isinstance(unit_path_policy, Mapping)
    ):
        raise PermissionError(
            "integration removal authorization closure changed"
        )
    device = _require_exact_int(
        fragment_identity.get("device"),
        name="integration fragment device", minimum=0,
    )
    inode = _require_exact_int(
        fragment_identity.get("inode"),
        name="integration fragment inode", minimum=1,
    )
    owner_uid = _require_exact_int(
        fragment_identity.get("owner_uid"),
        name="integration fragment owner", minimum=0,
    )
    mode = _require_exact_int(
        fragment_identity.get("mode"),
        name="integration fragment mode", minimum=0, maximum=0o7777,
    )
    nlink = _require_exact_int(
        fragment_identity.get("nlink"),
        name="integration fragment link count", minimum=1,
    )
    del device, inode
    expected_inactive = {
        "LoadState": "loaded", "UnitFileState": "static",
        "ActiveState": "inactive", "SubState": "dead",
        "FragmentPath": fragment_path, "DropInPaths": "",
        "Transient": "no", "Restart": "no", "NRestarts": "0",
        "NeedDaemonReload": "no",
    }
    integration._validate_manager_generation(manager_generation)
    integration._validate_unit_path_policy_transition(
        authorization["unit_path_policy"], unit_path_policy,
        authorized_uid=int(authorization["authorized_uid"]),
        allow_generator_late_inode_rotation=True,
    )
    if (
        not terminal_created <= issued <= authorization_expires
        or removal.get("scenario_id") != authorization["scenario_id"]
        or removal.get("unit_name") != expected_identity["unit_name"]
        or removal.get("authorization_fingerprint")
        != authorization["authorization_fingerprint"]
        or removal.get("integration_terminal_fingerprint")
        != terminal["integration_terminal_fingerprint"]
        or removal.get("runtime_spec_fingerprint")
        != authorization["runtime_spec_binding"]["runtime_spec_fingerprint"]
        or not _deep_exact_equal(
            removal.get("supervisor_evidence"), evidence
        )
        or fragment_identity.get("fragment_path") != fragment_path
        or fragment_identity.get("fragment_sha256")
        != rendered_fragment["sha256"]
        or owner_uid != authorization["authorized_uid"]
        or mode != 0o600
        or nlink != 1
        or not _deep_exact_equal(inactive_state, expected_inactive)
        or not _deep_exact_equal(
            manager_generation, authorization["manager_generation"]
        )
        or removal.get("remove_authorized") is not True
        or removal.get("daemon_reload_authorized") is not True
        or removal.get("not_found_verification_authorized") is not True
        or removal.get("enable_authorized") is not False
        or removal.get("start_authorized") is not False
        or removal.get("payload_authority") != "none"
    ):
        raise PermissionError(
            "integration removal authorization is not exact safe PASS"
        )


def _validate_integration_removal_state_semantics(
    removal_state: Mapping[str, object], *,
    authorization: Mapping[str, object],
    removal_authorization: Mapping[str, object],
    expected_identity: Mapping[str, str],
) -> None:
    if set(removal_state) != _INTEGRATION_REMOVAL_STATE_KEYS:
        raise PermissionError("integration removal state schema keys changed")
    _no_payload(removal_state)
    if (
        removal_state.get("scenario_id") != authorization["scenario_id"]
        or removal_state.get("unit_name") != expected_identity["unit_name"]
        or removal_state.get("removal_authorization_fingerprint")
        != removal_authorization["removal_authorization_fingerprint"]
        or removal_state.get("passed") is not True
        or removal_state.get("remove_attempted") is not True
        or removal_state.get("fragment_absent") is not True
        or not _deep_exact_equal(
            removal_state.get("not_found_state"),
            _INTEGRATION_NOT_FOUND_STATE,
        )
        or not _deep_exact_equal(
            removal_state.get("completed_actions"),
            _INTEGRATION_REMOVAL_ACTIONS,
        )
        or removal_state.get("error_type") is not None
        or removal_state.get("error_message") is not None
        or removal_state.get("payload_authority") != "none"
    ):
        raise PermissionError("integration removal state is not exact safe PASS")


def _validate_integration_chain(
    *, authorization_path: Path, receipt_path: Path,
) -> dict[str, dict[str, object]]:
    authorization, authorization_identity = _load_sealed(
        authorization_path, fingerprint_field="authorization_fingerprint",
        schema=integration.AUTHORIZATION_SCHEMA, return_identity=True,
    )
    if set(authorization) != integration._AUTH_KEYS:
        raise PermissionError("integration authorization schema keys changed")
    _bounded_lifetime(authorization, require_fresh=False)
    _no_payload(authorization)
    receipt, receipt_identity = _load_sealed(
        receipt_path, fingerprint_field="receipt_fingerprint",
        schema=integration.INTEGRATION_RECEIPT_SCHEMA, return_identity=True,
    )
    if set(receipt) != _INTEGRATION_RECEIPT_KEYS:
        raise PermissionError("integration PASS receipt schema keys changed")
    _no_payload(receipt)
    control = authorization.get("control_artifacts")
    if not isinstance(control, Mapping):
        raise PermissionError("integration control artifact closure is absent")
    control_names = (
        "integration_terminal", "removal_authorization", "removal_state",
        "integration_receipt",
    )
    if any(
        not isinstance(control.get(name), str)
        or not Path(str(control[name])).is_absolute()
        for name in control_names
    ):
        raise PermissionError("integration control artifact paths are not exact")
    scenario_id = authorization.get("scenario_id")
    if not isinstance(scenario_id, str):
        raise PermissionError("integration scenario identity is absent")
    try:
        expected_identity = integration.build_supervisor_v2_identity(scenario_id)
    except ValueError as error:
        raise PermissionError("integration scenario identity is not canonical") from error
    terminal_path = Path(str(control["integration_terminal"]))
    removal_authorization_path = Path(str(control["removal_authorization"]))
    removal_state_path = Path(str(control["removal_state"]))
    removal_state, removal_state_identity = _load_sealed(
        removal_state_path,
        fingerprint_field="removal_state_fingerprint",
        schema=integration.REMOVAL_STATE_SCHEMA, return_identity=True,
    )
    if set(removal_state) != _INTEGRATION_REMOVAL_STATE_KEYS:
        raise PermissionError("integration removal state schema keys changed")
    _no_payload(removal_state)
    terminal, terminal_identity = _load_sealed(
        terminal_path,
        fingerprint_field="integration_terminal_fingerprint",
        schema=integration.INTEGRATION_TERMINAL_SCHEMA, return_identity=True,
    )
    if set(terminal) != _INTEGRATION_TERMINAL_KEYS:
        raise PermissionError("integration terminal schema keys changed")
    _no_payload(terminal)
    removal_authorization, removal_authorization_identity = _load_sealed(
        removal_authorization_path,
        fingerprint_field="removal_authorization_fingerprint",
        schema=integration.REMOVAL_AUTHORIZATION_SCHEMA,
        return_identity=True,
    )
    if set(removal_authorization) != integration._REMOVAL_AUTH_KEYS:
        raise PermissionError("integration removal authorization schema keys changed")
    _bounded_lifetime(removal_authorization, require_fresh=False)
    _no_payload(removal_authorization)
    authorization_identity_value = authorization.get("identity")
    runtime_spec_binding = authorization.get("runtime_spec_binding")
    if (
        not isinstance(authorization_identity_value, Mapping)
        or not isinstance(runtime_spec_binding, Mapping)
    ):
        raise PermissionError("integration authorization lineage is incomplete")
    _validate_integration_authorization_semantics(
        authorization, expected_identity=expected_identity,
    )
    terminal_evidence = _validate_integration_terminal_semantics(
        terminal, authorization=authorization,
        expected_identity=expected_identity,
    )
    _validate_integration_removal_authorization_semantics(
        removal_authorization, authorization=authorization,
        terminal=terminal, expected_identity=expected_identity,
        evidence=terminal_evidence,
    )
    _validate_integration_removal_state_semantics(
        removal_state, authorization=authorization,
        removal_authorization=removal_authorization,
        expected_identity=expected_identity,
    )
    unit_name = expected_identity["unit_name"]
    if (
        authorization.get("integration_authorized") is not True
        or authorization.get("actual_r2_authorized") is not False
        or authorization.get("direct_start_authorized") is not False
        or authorization.get("enable_authorized") is not False
        or authorization.get("payload_authority") != "none"
        or authorization.get("gpu_access_authorized") is not False
        or not _deep_exact_equal(
            authorization_identity_value, expected_identity
        )
        or receipt.get("passed") is not True
        or receipt.get("payload_authority") != "none"
        or receipt.get("gpu_accessed") is not False
        or receipt.get("scenario_id") != scenario_id
        or not _deep_exact_equal(
            receipt.get("identity"), authorization_identity_value
        )
        or receipt.get("authorization_fingerprint")
        != authorization["authorization_fingerprint"]
        or receipt.get("authorization_path") != authorization_identity["path"]
        or authorization_identity["path"] != str(authorization_path.absolute())
        or receipt.get("authorization_file_sha256")
        != authorization_identity["file_sha256"]
        or receipt.get("integration_terminal_path")
        != terminal_identity["path"]
        or terminal_identity["path"] != str(control["integration_terminal"])
        or receipt.get("integration_terminal_file_sha256")
        != terminal_identity["file_sha256"]
        or receipt.get("integration_terminal_fingerprint")
        != terminal["integration_terminal_fingerprint"]
        or receipt.get("removal_authorization_path")
        != removal_authorization_identity["path"]
        or removal_authorization_identity["path"]
        != str(control["removal_authorization"])
        or receipt.get("removal_authorization_file_sha256")
        != removal_authorization_identity["file_sha256"]
        or receipt.get("removal_authorization_fingerprint")
        != removal_authorization["removal_authorization_fingerprint"]
        or receipt.get("removal_state_path")
        != removal_state_identity["path"]
        or removal_state_identity["path"] != str(control["removal_state"])
        or receipt.get("removal_state_file_sha256")
        != removal_state_identity["file_sha256"]
        or receipt.get("removal_state_fingerprint")
        != removal_state["removal_state_fingerprint"]
        or receipt_identity["path"] != str(receipt_path.absolute())
        or receipt_identity["path"] != str(control["integration_receipt"])
        or not isinstance(receipt.get("supervisor_evidence"), Mapping)
        or not _deep_exact_equal(
            receipt.get("supervisor_evidence"),
            terminal.get("supervisor_evidence"),
        )
        or terminal.get("scenario_id") != scenario_id
        or not _deep_exact_equal(
            terminal.get("identity"), authorization_identity_value
        )
        or terminal.get("authorization_fingerprint")
        != authorization["authorization_fingerprint"]
        or terminal.get("runtime_spec_fingerprint")
        != runtime_spec_binding.get("runtime_spec_fingerprint")
        or terminal.get("passed") is not True
        or terminal.get("error_type") is not None
        or terminal.get("error_message") is not None
        or terminal.get("payload_authority") != "none"
        or terminal.get("gpu_accessed") is not False
        or removal_authorization.get("scenario_id") != scenario_id
        or removal_authorization.get("unit_name") != unit_name
        or removal_authorization.get("authorization_fingerprint")
        != authorization["authorization_fingerprint"]
        or removal_authorization.get("integration_terminal_fingerprint")
        != terminal["integration_terminal_fingerprint"]
        or removal_authorization.get("runtime_spec_fingerprint")
        != runtime_spec_binding.get("runtime_spec_fingerprint")
        or not _deep_exact_equal(
            removal_authorization.get("supervisor_evidence"),
            terminal.get("supervisor_evidence"),
        )
        or removal_authorization.get("remove_authorized") is not True
        or removal_authorization.get("daemon_reload_authorized") is not True
        or removal_authorization.get(
            "not_found_verification_authorized"
        ) is not True
        or removal_authorization.get("enable_authorized") is not False
        or removal_authorization.get("start_authorized") is not False
        or removal_authorization.get("payload_authority") != "none"
        or removal_state.get("scenario_id") != scenario_id
        or removal_state.get("unit_name") != unit_name
        or removal_state.get("removal_authorization_fingerprint")
        != removal_authorization["removal_authorization_fingerprint"]
        or removal_state.get("payload_authority") != "none"
        or receipt.get("fragment_removed") is not True
        or removal_state.get("passed") is not True
        or removal_state.get("remove_attempted") is not True
        or removal_state.get("fragment_absent") is not True
        or removal_state.get("error_type") is not None
        or removal_state.get("error_message") is not None
        or not _deep_exact_equal(
            removal_state.get("not_found_state"),
            _INTEGRATION_NOT_FOUND_STATE,
        )
        or not _deep_exact_equal(
            receipt.get("post_removal_unit_state"),
            removal_state.get("not_found_state"),
        )
    ):
        raise PermissionError("true user-systemd integration is not a complete PASS")
    return {
        "authorization": authorization, "receipt": receipt,
        "identities": {
            "authorization": authorization_identity,
            "receipt": receipt_identity,
            "removal_state": removal_state_identity,
            "integration_terminal": terminal_identity,
            "removal_authorization": removal_authorization_identity,
        },
    }


def _validate_normal_realization_chain(
    *, authorization_path: Path, receipt_path: Path,
    shadow_reader: _ShadowReader, manager_reader: _ManagerReader,
    unit_path_policy_reader: _UnitPathPolicyReader | None = None,
) -> dict[str, object]:
    authorization, authorization_identity = _load_sealed(
        authorization_path, fingerprint_field="authorization_fingerprint",
        schema=actual_realizer.AUTHORIZATION_SCHEMA, return_identity=True,
    )
    if set(authorization) != actual_realizer._AUTH_KEYS:
        raise PermissionError("actual realization authorization schema keys changed")
    _bounded_lifetime(authorization, require_fresh=False)
    _no_payload(authorization)
    receipt, receipt_identity = _load_sealed(
        receipt_path, fingerprint_field="receipt_fingerprint",
        schema=actual_realizer.RECEIPT_SCHEMA, return_identity=True,
    )
    if set(receipt) != _REALIZATION_RECEIPT_KEYS:
        raise PermissionError("actual realization PASS receipt schema keys changed")
    _no_payload(receipt)
    authorization_created = _evidence_timestamp(
        authorization, "created_at_utc",
    )
    authorization_issued = _evidence_timestamp(
        authorization, "issued_at_utc",
    )
    authorization_expires = _evidence_timestamp(
        authorization, "expires_at_utc",
    )
    receipt_created = _evidence_timestamp(receipt, "created_at_utc")
    if not (
        authorization_issued
        <= authorization_created
        <= receipt_created
        <= authorization_expires
    ):
        raise PermissionError(
            "actual realization authorization/receipt chronology changed"
        )
    runtime_spec_binding = authorization.get("runtime_spec_binding")
    full_static_shadow = receipt.get("full_static_shadow")
    if (
        not isinstance(runtime_spec_binding, Mapping)
        or not isinstance(full_static_shadow, Mapping)
    ):
        raise PermissionError("actual realization semantic closure is incomplete")
    if (
        authorization.get("candidate") != CANDIDATE
        or authorization.get("stage_id") != STAGE_ID
        or authorization.get("attempt_id") != ATTEMPT_ID
        or authorization.get("unit_name") != UNIT_NAME
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis") != AUTHORIZATION_BASIS
        or isinstance(authorization.get("authorized_uid"), bool)
        or not isinstance(authorization.get("authorized_uid"), int)
        or authorization.get("authorized_uid") != os.getuid()
        or not _deep_exact_equal(
            authorization.get("actions"), actual_realizer._ACTIONS
        )
        or authorization.get("payload_authority") != "none"
        or any(authorization.get(field) is not False for field in (
            "persistent_install_authorized", "enable_authorized",
            "start_authorized", "remove_authorized",
        ))
        or runtime_spec_binding.get("runtime_spec_path")
        != str(RUNTIME_SPEC_PATH)
        or receipt.get("candidate") != CANDIDATE
        or receipt.get("candidate") != authorization.get("candidate")
        or receipt.get("stage_id") != STAGE_ID
        or receipt.get("stage_id") != authorization.get("stage_id")
        or receipt.get("attempt_id") != ATTEMPT_ID
        or receipt.get("attempt_id") != authorization.get("attempt_id")
        or receipt.get("unit_name") != UNIT_NAME
        or receipt.get("unit_name") != authorization.get("unit_name")
        or receipt.get("instruction_id") != INSTRUCTION_ID
        or receipt.get("instruction_id") != authorization.get("instruction_id")
        or receipt.get("payload_authority") != "none"
        or receipt.get("passed") is not True or receipt.get("static") is not True
        or receipt.get("enabled") is not False or receipt.get("started") is not False
        or receipt.get("removed") is not False
        or receipt.get("runtime_spec_absent_at_receipt") is not True
        or receipt.get("expected_future_runtime_spec_path") != str(RUNTIME_SPEC_PATH)
        or receipt.get("authorization_path") != str(authorization_path.absolute())
        or receipt.get("authorization_file_sha256")
        != authorization_identity["file_sha256"]
        or receipt.get("authorization_fingerprint")
        != authorization["authorization_fingerprint"]
        or not _deep_exact_equal(
            receipt.get("manager_generation"),
            authorization.get("manager_generation"),
        )
        or not _deep_exact_equal(
            receipt.get("template_binding"),
            authorization.get("template_binding"),
        )
        or not _deep_exact_equal(
            receipt.get("rendered_fragment"),
            authorization.get("rendered_fragment"),
        )
        or not _deep_exact_equal(
            receipt.get("runtime_spec_binding"), runtime_spec_binding,
        )
        or not _deep_exact_equal(
            receipt.get("executable_bindings"),
            authorization.get("executable_bindings"),
        )
        or not _deep_exact_equal(
            receipt.get("completed_actions"), authorization.get("actions"),
        )
        or not _deep_exact_equal(
            receipt.get("completed_actions"), actual_realizer._ACTIONS,
        )
    ):
        raise PermissionError("actual unit realization chain is not exact PASS")
    executable_bindings = authorization.get("executable_bindings")
    template_binding = authorization.get("template_binding")
    unit_path_policy = authorization.get("unit_path_policy")
    if (
        not isinstance(executable_bindings, Mapping)
        or set(executable_bindings) != {
            "realization_tool", "python", "supervisor", "systemd_path",
            "systemd_analyze", "systemctl",
        }
        or not isinstance(template_binding, Mapping)
        or not isinstance(unit_path_policy, Mapping)
        or set(runtime_spec_binding) != {
            "kind", "runtime_spec_path", "runtime_spec_parent_identity",
            "absent_at_authorization", "required_schema",
        }
    ):
        raise PermissionError(
            "actual realization authorization binding closure changed"
        )
    for name, binding in executable_bindings.items():
        _validate_realization_file_binding(
            binding, name=f"actual realization executable:{name}",
        )
    actual_realizer._validate_python_binding(executable_bindings["python"])
    template_raw = _validate_realization_file_binding(
        template_binding, name="actual realization template",
    )
    runtime_spec_path = Path(str(runtime_spec_binding["runtime_spec_path"]))
    unit_directory = Path(str(authorization.get("unit_directory")))
    if (
        runtime_spec_binding.get("kind")
        != "future-absent-runtime-spec-v2"
        or runtime_spec_binding.get("required_schema")
        != actual_realizer.SUPERVISOR_SPEC_SCHEMA
        or runtime_spec_binding.get("absent_at_authorization") is not True
        or not runtime_spec_path.is_absolute()
        or runtime_spec_path != RUNTIME_SPEC_PATH
        or not _deep_exact_equal(
            actual_realizer._path_row(runtime_spec_path.parent),
            runtime_spec_binding.get("runtime_spec_parent_identity"),
        )
        or not unit_directory.is_absolute()
        or unit_path_policy.get("runtime_directory") != str(unit_directory)
    ):
        raise PermissionError(
            "actual realization future runtime binding changed"
        )
    try:
        template_text = template_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PermissionError(
            "actual realization template is not strict UTF-8"
        ) from error
    rendered = actual_realizer.render_fragment(
        template_text,
        python_path=Path(str(executable_bindings["python"]["path"])),
        supervisor_path=Path(str(executable_bindings["supervisor"]["path"])),
        runtime_spec_path=runtime_spec_path,
    )
    if (
        not _deep_exact_equal(
            authorization.get("rendered_fragment"),
            {
                "utf8_text": rendered,
                "sha256": hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
            },
        )
        or not _deep_exact_equal(
            authorization.get("expected_static_shadow"),
            actual_realizer._expected_static_shadow(
                unit_directory / UNIT_NAME,
            ),
        )
    ):
        raise PermissionError(
            "actual realization rendered/static authorization changed"
        )
    expected_manager = authorization["manager_generation"]
    actual_realizer._validate_manager_generation(expected_manager)
    python_fields = _realizer_python_source_fields({
        "authorization": authorization, "receipt": receipt,
    })
    del python_fields
    transitioned_policy = (
        actual_realizer._validate_daemon_reload_path_policy_transition(
            authorization["unit_path_policy"], receipt["unit_path_policy"],
            runtime_directory=unit_directory,
            authorized_uid=int(authorization["authorized_uid"]),
        )
    )
    if not _deep_exact_equal(
        transitioned_policy, receipt["unit_path_policy"]
    ):
        raise PermissionError("actual unit generator.late transition changed")
    fragment_identity = _validate_realization_fragment_identity(
        receipt.get("fragment_identity"),
        expected_path=unit_directory / UNIT_NAME,
    )

    if unit_path_policy_reader is None:
        def unit_path_policy_reader(fragment: Path) -> Mapping[str, object]:
            return actual_realizer._observe_unit_path_policy(
                runner=subprocess.run, allowed_fragment=fragment.absolute(),
            )

    def require_manager_exact() -> dict[str, object]:
        live = dict(manager_reader())
        actual_realizer._validate_manager_generation(live)
        if not _deep_exact_equal(live, expected_manager):
            raise PermissionError("actual unit manager generation changed")
        return live

    fragment_path = Path(str(fragment_identity["path"]))
    _, live_fragment_identity = _stable_read_file(fragment_path)
    if not _deep_exact_equal(live_fragment_identity, fragment_identity):
        raise PermissionError("actual unit fragment identity changed")
    require_manager_exact()
    live_policy_before_shadow = dict(unit_path_policy_reader(fragment_path))
    if not _deep_exact_equal(
        live_policy_before_shadow, receipt["unit_path_policy"]
    ):
        raise PermissionError("actual unit live path policy changed")
    require_manager_exact()
    shadow = dict(shadow_reader())
    validated = actual_realizer.validate_installed_shadow(
        shadow, fragment_identity=fragment_identity,
        authorization=authorization,
    )
    if not _deep_exact_equal(full_static_shadow, validated):
        raise PermissionError(
            "actual unit sealed/live validated shadow closure changed"
        )
    require_manager_exact()
    live_policy_after_shadow = dict(unit_path_policy_reader(fragment_path))
    if not _deep_exact_equal(
        live_policy_after_shadow, receipt["unit_path_policy"]
    ):
        raise PermissionError(
            "actual unit path policy rotated again after static shadow"
        )
    validated_after_policy = actual_realizer.validate_installed_shadow(
        shadow, fragment_identity=fragment_identity,
        authorization=authorization,
    )
    if (
        not _deep_exact_equal(validated_after_policy, validated)
        or not _deep_exact_equal(full_static_shadow, validated_after_policy)
    ):
        raise PermissionError("actual unit fragment/shadow view changed")
    validated = validated_after_policy
    require_manager_exact()
    return {
        "authorization": authorization, "receipt": receipt,
        "authorization_identity": authorization_identity,
        "receipt_identity": receipt_identity,
        "live_unit_path_policy": live_policy_after_shadow,
        "live_shadow": shadow, "validated_shadow": validated,
    }


def _validate_realization_chain(
    *, authorization_path: Path, receipt_path: Path,
    shadow_reader: _ShadowReader, manager_reader: _ManagerReader,
    unit_path_policy_reader: _UnitPathPolicyReader | None = None,
) -> dict[str, object]:
    """Validate either the original PASS or the exact retained-failure closure."""

    routed = _load_sealed(
        authorization_path,
        fingerprint_field="authorization_fingerprint",
    )
    schema = routed.get("schema_version")
    if schema == actual_realizer.AUTHORIZATION_SCHEMA:
        return _validate_normal_realization_chain(
            authorization_path=authorization_path,
            receipt_path=receipt_path,
            shadow_reader=shadow_reader,
            manager_reader=manager_reader,
            unit_path_policy_reader=unit_path_policy_reader,
        )
    if schema != realization_recovery.RECOVERY_AUTHORIZATION_SCHEMA:
        raise PermissionError(
            "actual realization authorization schema is unsupported",
        )
    if unit_path_policy_reader is None:
        def unit_path_policy_reader(
            fragment: Path,
        ) -> Mapping[str, object]:
            return actual_realizer._observe_unit_path_policy(
                runner=subprocess.run,
                allowed_fragment=fragment.absolute(),
            )
    return realization_recovery.validate_release_recovery_chain(
        recovery_authorization_path=authorization_path,
        recovery_receipt_path=receipt_path,
        shadow_reader=shadow_reader,
        manager_reader=manager_reader,
        unit_path_policy_reader=unit_path_policy_reader,
    )


def _validate_scientific_preaccess() -> dict[str, dict[str, object]]:
    authorization, authorization_identity = _load_sealed(
        SCIENTIFIC_AUTHORIZATION_PATH,
        fingerprint_field="authorization_fingerprint",
        schema="cure-lite-v24-D_R-structural-r2-authorization-v1",
        return_identity=True,
    )
    audit, audit_identity = _load_sealed(
        SCIENTIFIC_ACCESS_AUDIT_PATH, fingerprint_field="receipt_fingerprint",
        schema="cure-lite-v24-split-access-audit-r2-v1",
        return_identity=True,
    )
    authorization_sha_fields = {
        key for key in _SCIENTIFIC_AUTHORIZATION_KEYS
        if key.endswith("_sha256") or key.endswith("_fingerprint")
    }
    audit_sha_fields = {
        "event_log_fingerprint", "source_manifest_fingerprint",
        "receipt_fingerprint",
    }
    if (
        set(authorization) != _SCIENTIFIC_AUTHORIZATION_KEYS
        or set(audit) != _SCIENTIFIC_AUDIT_KEYS
        or any(
            not isinstance(authorization.get(key), str)
            or _SHA.fullmatch(str(authorization[key])) is None
            for key in authorization_sha_fields
        )
        or any(
            not isinstance(audit.get(key), str)
            or _SHA.fullmatch(str(audit[key])) is None
            for key in audit_sha_fields
        )
        or authorization.get("candidate") != CANDIDATE
        or authorization.get("stage_id") != STAGE_ID
        or authorization.get("run_id") != ATTEMPT_ID
        or authorization.get("status") != "GCR_PACRE_V24_D_R_STRUCTURAL_R2_AUTHORIZED"
        or authorization.get("D_R_payload_authorized") is not True
        or authorization.get("D_V_payload_authorized") is not False
        or authorization.get("D_T_payload_authorized") is not False
        or authorization.get("training_authorized") is not False
        or authorization.get("expires_after_single_materialization") is not True
        or authorization.get("allowed_splits") != ["D_R"]
        or authorization.get("allowed_purposes") != ["zero_update_structural_gate"]
        or authorization.get("source_closure_fingerprint")
        != SOURCE_CLOSURE_FINGERPRINT_103
        or authorization.get("access_audit_receipt_fingerprint")
        != audit["receipt_fingerprint"]
        or audit.get("stage_id") != STAGE_ID
        or audit.get("allowed_splits") != ["D_R"]
        or audit.get("observed_payloads") != []
        or audit.get("D_V_payload_accessed") is not False
        or audit.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("scientific preaccess identity or no-access state changed")
    return {
        "authorization": authorization, "audit": audit,
        "identities": {
            "authorization": authorization_identity,
            "audit": audit_identity,
        },
    }


def _validate_prior_interruption(
    *, return_identity: bool = False,
) -> (
    dict[str, object]
    | tuple[dict[str, object], dict[str, object]]
):
    receipt, identity = _load_sealed(
        PRIOR_R1_INTERRUPTION_RECEIPT_PATH,
        fingerprint_field="receipt_fingerprint",
        schema="cure-lite-v24-D_R-structural-interruption-receipt-v2",
        canonical_required=False, return_identity=True,
    )
    disposition = receipt.get("protocol_disposition")
    observation = receipt.get("authoritative_observation")
    if (
        set(receipt) != _PRIOR_INTERRUPTION_KEYS
        or receipt.get("candidate") != CANDIDATE
        or receipt.get("run_id") != "gcr_pacre_v24_D_R_zero_update_structural_r1"
        or not isinstance(disposition, Mapping) or not isinstance(observation, Mapping)
        or observation.get("classification") != "EXECUTION_OBSERVABILITY_LOST_NO_DECISION"
        or observation.get("scientific_model_failure_established") is not False
        or observation.get("structural_gate_failure_established") is not False
        or disposition.get("attempt_ordinal") != 1
        or disposition.get("official_D_R_receipt_valid") is not False
        or disposition.get("same_run_resume_allowed") is not False
        or disposition.get("automatic_retry_allowed") is not False
        or disposition.get("fresh_attempt_currently_authorized") is not False
    ):
        raise PermissionError("r1 interruption receipt no-decision meaning changed")
    if return_identity:
        return receipt, identity
    return receipt


def _release_inputs_from_spec(spec: Mapping[str, object]) -> dict[str, Path]:
    environment = spec["environment"]
    if not isinstance(environment, Mapping):
        raise PermissionError("actual spec environment contract is absent")
    plan_path = Path(str(environment["cleanup_plan_path"]))
    plan = _load_sealed(
        plan_path, fingerprint_field="plan_fingerprint", schema=cleanup.PLAN_SCHEMA,
    )
    return {
        "policy_path": Path(str(environment["policy_path"])),
        "precleanup_path": Path(str(plan["inventory_receipt_path"])),
        "cleanup_plan_path": plan_path,
        "cleanup_authorization_path": Path(str(environment["cleanup_authorization_path"])),
        "cleanup_receipt_path": Path(str(environment["cleanup_receipt_path"])),
        "stability_path": Path(str(environment["stability_receipt_path"])),
        "postcleanup_path": Path(str(environment["inventory_path"])),
        "integration_authorization_path": Path(str(environment["integration_authorization_path"])),
        "integration_receipt_path": Path(str(environment["integration_receipt_path"])),
        "realization_authorization_path": Path(str(environment["unit_realization_authorization_path"])),
        "realization_receipt_path": Path(str(environment["unit_realization_receipt_path"])),
    }


def validate_release_closure(
    *, policy_path: Path, precleanup_path: Path, cleanup_plan_path: Path,
    cleanup_authorization_path: Path, cleanup_receipt_path: Path,
    stability_path: Path, postcleanup_path: Path,
    integration_authorization_path: Path, integration_receipt_path: Path,
    realization_authorization_path: Path, realization_receipt_path: Path,
    shadow_reader: _ShadowReader, manager_reader: _ManagerReader,
    summary_runner: _CommandRunner = _default_summary_runner,
    unit_path_policy_reader: _UnitPathPolicyReader | None = None,
) -> dict[str, object]:
    _private_directory(EVIDENCE_ROOT)
    cleanup_chain = _validate_cleanup_chain(
        precleanup_path=precleanup_path, plan_path=cleanup_plan_path,
        authorization_path=cleanup_authorization_path,
        receipt_path=cleanup_receipt_path,
    )
    environment_chain = _validate_environment_chain(
        policy_path=policy_path, precleanup_path=precleanup_path,
        cleanup_receipt_path=cleanup_receipt_path,
        stability_path=stability_path, postcleanup_path=postcleanup_path,
    )
    integration_chain = _validate_integration_chain(
        authorization_path=integration_authorization_path,
        receipt_path=integration_receipt_path,
    )
    realization_chain = _validate_realization_chain(
        authorization_path=realization_authorization_path,
        receipt_path=realization_receipt_path, shadow_reader=shadow_reader,
        manager_reader=manager_reader,
        unit_path_policy_reader=unit_path_policy_reader,
    )
    scientific = _validate_scientific_preaccess()
    prior, prior_identity = _validate_prior_interruption(return_identity=True)
    summary = _validate_identity_summary(summary_runner)
    if (
        cleanup_chain["identities"]["precleanup"]
        != environment_chain["identities"]["precleanup"]
        or cleanup_chain["identities"]["receipt"]
        != environment_chain["identities"]["cleanup_receipt"]
    ):
        raise PermissionError(
            "release input identity changed between closure validators"
        )
    input_identities = {
        "policy": environment_chain["identities"]["policy"],
        "inventory": environment_chain["identities"]["postcleanup"],
        "cleanup_plan": cleanup_chain["identities"]["plan"],
        "cleanup_authorization": cleanup_chain["identities"]["authorization"],
        "cleanup_receipt": cleanup_chain["identities"]["receipt"],
        "stability_receipt": environment_chain["identities"]["stability"],
        "integration_authorization":
            integration_chain["identities"]["authorization"],
        "integration_receipt": integration_chain["identities"]["receipt"],
        "unit_realization_authorization":
            realization_chain["authorization_identity"],
        "unit_realization_receipt": realization_chain["receipt_identity"],
    }
    return {
        "cleanup": cleanup_chain, "environment": environment_chain,
        "integration": integration_chain, "realization": realization_chain,
        "scientific": scientific, "prior": prior, "identity_summary": summary,
        "prior_identity": prior_identity, "input_identities": input_identities,
    }


def _artifact_paths() -> dict[str, str]:
    root = RUNTIME_ARTIFACT_ROOT
    return {
        "root": str(root), "attempt_commit": str(root / "attempt-commit.json"),
        "materialization_claim": str(root / "materialization-claim.json"),
        "stdout_log": str(root / "stdout.log"), "stderr_log": str(root / "stderr.log"),
        "heartbeat_dir": str(root / "heartbeat"),
        "runtime_terminal": str(root / "runtime-terminal.json"),
        "systemd_invocation_dir": str(root / "systemd-invocations"),
        "launch_lease": str(root / "launch-lease.json"),
        "precommit_phase_receipt": str(root / "precommit-phase.json"),
        "start_ack_receipt": str(root / "start-ack.json"),
        "child_prespawn_phase_receipt": str(root / "child-prespawn.json"),
        "consumed_start_failure_receipt": str(root / "consumed-start-failure.json"),
        "gpu_lease_release_receipt": str(root / "gpu-lease-release.json"),
        "runtime_attestation": str(root / "runtime-attestation.json"),
    }


def _environment_contract(inputs: Mapping[str, Path], closure: Mapping[str, object]) -> dict[str, object]:
    policy = closure["environment"]["policy"]
    selected = policy["selected_gpu"]
    definitions = {
        "policy": inputs["policy_path"], "inventory": inputs["postcleanup_path"],
        "cleanup_plan": inputs["cleanup_plan_path"],
        "cleanup_authorization": inputs["cleanup_authorization_path"],
        "cleanup_receipt": inputs["cleanup_receipt_path"],
        "stability_receipt": inputs["stability_path"],
        "integration_authorization": inputs["integration_authorization_path"],
        "integration_receipt": inputs["integration_receipt_path"],
        "unit_realization_authorization": inputs["realization_authorization_path"],
        "unit_realization_receipt": inputs["realization_receipt_path"],
    }
    identities = closure.get("input_identities")
    result: dict[str, object] = {}
    for prefix, path in definitions.items():
        result[f"{prefix}_path"] = str(path.absolute())
        if isinstance(identities, Mapping) and isinstance(
            identities.get(prefix), Mapping,
        ):
            identity = identities[prefix]
        else:
            # Isolated tests may provide a synthetic closure; production
            # closure validation always supplies the same-descriptor identity.
            identity = _file_binding(path)
        result[f"{prefix}_file_sha256"] = identity["file_sha256"]
    result.update({
        "selected_gpu_uuid": selected["uuid"],
        "selected_gpu_pci_bus_id": selected["pci_bus_id"],
        "selected_gpu_minor_number": selected["minor_number"],
        "gpu_lease_path": str(GPU_LEASE_ROOT / "active.json"),
        "gpu_lease_tombstone_path": str(GPU_LEASE_ROOT / "released.json"),
    })
    return result


def _immutable_shadow(realization: Mapping[str, object]) -> dict[str, str]:
    live = realization["live_shadow"]
    return {key: str(live[key]) for key in supervisor._SYSTEMD_IMMUTABLE_SHADOW_KEYS}


def _create_runtime_directories_and_verify_leaves(artifacts: Mapping[str, str]) -> None:
    _private_directory(RUNTIME_ARTIFACT_ROOT, create=True)
    _private_directory(Path(artifacts["heartbeat_dir"]), create=True)
    _private_directory(Path(artifacts["systemd_invocation_dir"]), create=True)
    _private_directory(GPU_LEASE_ROOT, create=True)
    directory_keys = {"root", "heartbeat_dir", "systemd_invocation_dir"}
    leaves = [Path(value) for key, value in artifacts.items() if key not in directory_keys]
    leaves.extend((GPU_LEASE_ROOT / "active.json", GPU_LEASE_ROOT / "released.json"))
    if any(os.path.lexists(path) for path in leaves):
        raise FileExistsError("runtime artifact/GPU lease leaf already exists")


def build_spec(
    *, policy_path: Path, precleanup_path: Path, cleanup_plan_path: Path,
    cleanup_authorization_path: Path, cleanup_receipt_path: Path,
    stability_path: Path, postcleanup_path: Path,
    integration_authorization_path: Path, integration_receipt_path: Path,
    realization_authorization_path: Path, realization_receipt_path: Path,
    shadow_reader: _ShadowReader, manager_reader: _ManagerReader,
    summary_runner: _CommandRunner = _default_summary_runner,
) -> dict[str, object]:
    if os.path.lexists(RUNTIME_SPEC_PATH) or os.path.lexists(RUNTIME_LAUNCH_AUTHORIZATION_PATH):
        raise FileExistsError("actual runtime spec/authorization identity is already consumed")
    inputs = {
        "policy_path": policy_path.absolute(), "precleanup_path": precleanup_path.absolute(),
        "cleanup_plan_path": cleanup_plan_path.absolute(),
        "cleanup_authorization_path": cleanup_authorization_path.absolute(),
        "cleanup_receipt_path": cleanup_receipt_path.absolute(),
        "stability_path": stability_path.absolute(), "postcleanup_path": postcleanup_path.absolute(),
        "integration_authorization_path": integration_authorization_path.absolute(),
        "integration_receipt_path": integration_receipt_path.absolute(),
        "realization_authorization_path": realization_authorization_path.absolute(),
        "realization_receipt_path": realization_receipt_path.absolute(),
    }
    closure = validate_release_closure(
        **inputs, shadow_reader=shadow_reader, manager_reader=manager_reader,
        summary_runner=summary_runner,
    )
    artifacts = _artifact_paths()
    environment_contract = _environment_contract(inputs, closure)
    shadow = _immutable_shadow(closure["realization"])
    fragment_sha = closure["realization"]["receipt"]["fragment_identity"]["file_sha256"]
    source_bindings = _collect_source_bindings(
        closure["realization"],
        expected_prior_identity=closure.get("prior_identity"),
    )
    child_argv = [
        str(PYTHON_PATH), "-I", "-S", "-B", "-u", str(ADAPTER_PATH), "real",
        "--execute-real-dr", "--device", "cuda:0",
        "--runtime-launch-authorization", str(RUNTIME_LAUNCH_AUTHORIZATION_PATH),
    ]
    child_environment = {
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": environment_contract["selected_gpu_uuid"],
    }
    scientific = closure["scientific"]
    scientific_identities = scientific.get("identities")
    if not isinstance(scientific_identities, Mapping):
        scientific_identities = {
            "authorization": _file_binding(SCIENTIFIC_AUTHORIZATION_PATH),
            "audit": _file_binding(SCIENTIFIC_ACCESS_AUDIT_PATH),
        }
    body: dict[str, object] = {
        "schema_version": supervisor.RUNTIME_SPEC_SCHEMA,
        "execution_kind": supervisor.ACTUAL_EXECUTION_KIND,
        "candidate": CANDIDATE, "stage_id": STAGE_ID, "attempt_id": ATTEMPT_ID,
        "attempt_ordinal": 2, "prior_attempt_count": 1,
        "authorization": {
            "path": str(RUNTIME_LAUNCH_AUTHORIZATION_PATH),
            "required_schema": supervisor.RUNTIME_LAUNCH_AUTHORIZATION_SCHEMA,
        },
        "scientific_preaccess": {
            "authorization_path": str(SCIENTIFIC_AUTHORIZATION_PATH),
            "authorization_file_sha256":
                scientific_identities["authorization"]["file_sha256"],
            "authorization_fingerprint": scientific["authorization"]["authorization_fingerprint"],
            "authorization_required_schema": scientific["authorization"]["schema_version"],
            "access_audit_path": str(SCIENTIFIC_ACCESS_AUDIT_PATH),
            "access_audit_file_sha256":
                scientific_identities["audit"]["file_sha256"],
            "access_audit_fingerprint": scientific["audit"]["receipt_fingerprint"],
            "access_audit_required_schema": scientific["audit"]["schema_version"],
            "source_closure_fingerprint_103": SOURCE_CLOSURE_FINGERPRINT_103,
        },
        "child": {
            "argv": child_argv, "argv_fingerprint": stable_fingerprint(child_argv),
            "cwd": str(REPOSITORY), "environment": child_environment,
            "inherit_environment": [], "entrypoint_path": str(ADAPTER_PATH),
        },
        "artifacts": artifacts,
        "runtime": {
            "shell": False, "start_new_session": True, "launch_limit": 1,
            "automatic_retry_allowed": False, "resume_allowed": False,
            "restart": "no", "heartbeat_interval_seconds": 5.0,
            "poll_interval_seconds": 0.1, "termination_grace_seconds": 300.0,
            "systemd": {
                "unit_name": UNIT_NAME, "service_type": "exec", "kill_mode": "mixed",
                "send_sigkill": True, "timeout_stop_seconds": 600.0,
                "start_ack_timeout_seconds": 30.0, "start_ack_poll_seconds": 0.1,
                "unit_fragment_file_sha256": fragment_sha,
                "immutable_shadow_properties": shadow,
                "immutable_shadow_fingerprint": stable_fingerprint(shadow),
            },
        },
        "environment": environment_contract,
        "source_bindings": source_bindings,
    }
    # Recheck every input before the first release mutation.
    precommit_closure = validate_release_closure(
        **inputs, shadow_reader=shadow_reader, manager_reader=manager_reader,
        summary_runner=summary_runner,
    )
    if (
        precommit_closure.get("input_identities")
        != closure.get("input_identities")
        or precommit_closure.get("prior_identity")
        != closure.get("prior_identity")
        or precommit_closure.get("scientific", {}).get("identities")
        != closure.get("scientific", {}).get("identities")
    ):
        raise PermissionError("release input identity generation changed at precommit")
    _validate_source_bindings(
        source_bindings,
        realization=precommit_closure["realization"],
        expected_prior_identity=precommit_closure.get("prior_identity"),
    )
    _create_runtime_directories_and_verify_leaves(artifacts)
    return _write_sealed(
        RUNTIME_SPEC_PATH, body, fingerprint_field="runtime_spec_fingerprint"
    )


def _environment_authorization_bindings(spec: Mapping[str, object]) -> dict[str, object]:
    # Use the production supervisor's verifier as the sole field-construction
    # authority.  This prevents launch-authorizer and launch-consumer drift and
    # also replays the sealed recovery-mode environment contract before either
    # building or validating an authorization.
    return supervisor._environment_evidence_bindings(spec)


def validate_launch_authorization(
    authorization: Mapping[str, object], *, spec: Mapping[str, object],
    spec_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = dict(authorization)
    body = dict(payload)
    fingerprint = body.pop("authorization_fingerprint", None)
    _bounded_lifetime(payload, require_fresh=True)
    source_bindings = _validate_source_bindings(spec.get("source_bindings"))
    loaded_spec, observed_spec_identity = _load_sealed(
        RUNTIME_SPEC_PATH,
        fingerprint_field="runtime_spec_fingerprint",
        schema=supervisor.RUNTIME_SPEC_SCHEMA,
        return_identity=True,
    )
    if (
        loaded_spec != dict(spec)
        or (
            spec_identity is not None
            and any(
                observed_spec_identity[key] != spec_identity[key]
                for key in (
                    "file_sha256", "device", "inode",
                    "owner_uid", "mode", "nlink",
                )
            )
        )
    ):
        raise PermissionError("runtime spec changed from authorization input")
    spec_identity = observed_spec_identity
    scientific_authorization, scientific_authorization_identity = _load_sealed(
        SCIENTIFIC_AUTHORIZATION_PATH,
        fingerprint_field="authorization_fingerprint",
        return_identity=True,
    )
    scientific_audit, scientific_audit_identity = _load_sealed(
        SCIENTIFIC_ACCESS_AUDIT_PATH,
        fingerprint_field="receipt_fingerprint",
        return_identity=True,
    )
    if (
        scientific_authorization_identity["file_sha256"]
        != spec["scientific_preaccess"]["authorization_file_sha256"]
        or scientific_audit_identity["file_sha256"]
        != spec["scientific_preaccess"]["access_audit_file_sha256"]
        or scientific_authorization.get("authorization_fingerprint")
        != spec["scientific_preaccess"]["authorization_fingerprint"]
        or scientific_audit.get("receipt_fingerprint")
        != spec["scientific_preaccess"]["access_audit_fingerprint"]
    ):
        raise PermissionError("scientific preaccess files changed")
    required = {
        "schema_version": supervisor.RUNTIME_LAUNCH_AUTHORIZATION_SCHEMA,
        "authorization_kind": "runtime_launch", "instruction_id": INSTRUCTION_ID,
        "authorization_basis": AUTHORIZATION_BASIS, "authorized_uid": os.getuid(),
        **_environment_authorization_bindings(spec),
        "scientific_preaccess_authorization_path": str(SCIENTIFIC_AUTHORIZATION_PATH),
        "scientific_preaccess_authorization_file_sha256":
            scientific_authorization_identity["file_sha256"],
        "scientific_preaccess_authorization_fingerprint": spec["scientific_preaccess"]["authorization_fingerprint"],
        "scientific_preaccess_access_audit_path": str(SCIENTIFIC_ACCESS_AUDIT_PATH),
        "scientific_preaccess_access_audit_file_sha256":
            scientific_audit_identity["file_sha256"],
        "scientific_preaccess_access_audit_fingerprint": spec["scientific_preaccess"]["access_audit_fingerprint"],
        "r2_adapter_path": str(ADAPTER_PATH),
        "r2_adapter_file_sha256":
            source_bindings["r2_adapter_file_sha256"],
        "legacy_gate_entrypoint_path": str(LEGACY_GATE_PATH),
        "legacy_gate_entrypoint_file_sha256":
            source_bindings["legacy_gate_entrypoint_file_sha256"],
        "source_closure_fingerprint_103": SOURCE_CLOSURE_FINGERPRINT_103,
        "candidate": CANDIDATE, "stage_id": STAGE_ID, "attempt_id": ATTEMPT_ID,
        "attempt_ordinal": 2, "prior_attempt_count": 1,
        "fresh_attempt_authorized": True, "D_R_payload_authorized": True,
        "D_V_payload_authorized": False, "D_T_payload_authorized": False,
        "training_authorized": False, "resume_allowed": False,
        "automatic_retry_allowed": False,
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
        "runtime_spec_file_sha256": spec_identity["file_sha256"],
        "runtime_spec_v2_fingerprint": spec["runtime_spec_fingerprint"],
        "runtime_spec_v2_file_sha256": spec_identity["file_sha256"],
        "supervisor_v2_source_closure_fingerprint": stable_fingerprint(spec["source_bindings"]),
        "unit_fragment_sha256": spec["runtime"]["systemd"]["unit_fragment_file_sha256"],
        "preauthorization_D_R_payload_accessed": False,
        "preauthorization_D_V_payload_accessed": False,
        "preauthorization_D_T_payload_accessed": False,
    }
    temporal = {"issued_at_utc", "expires_at_utc"}
    if (
        fingerprint != stable_fingerprint(body)
        or set(body) != set(required) | temporal
        or any(payload.get(key) != value for key, value in required.items())
    ):
        raise PermissionError("intended fresh runtime launch authorization is invalid")
    return payload


def authorize_launch(
    *, shadow_reader: _ShadowReader, manager_reader: _ManagerReader,
    summary_runner: _CommandRunner = _default_summary_runner,
    validity_seconds: int = 300,
) -> dict[str, object]:
    if (
        isinstance(validity_seconds, bool) or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("launch authorization validity must be in [1,300]")
    if os.path.lexists(RUNTIME_LAUNCH_AUTHORIZATION_PATH):
        raise FileExistsError("runtime launch authorization identity is consumed")
    # Parse and hash through one stable descriptor, then invoke the production
    # supervisor's structure validator on those exact bytes.
    spec, spec_identity = _load_sealed(
        RUNTIME_SPEC_PATH,
        fingerprint_field="runtime_spec_fingerprint",
        schema=supervisor.RUNTIME_SPEC_SCHEMA,
        return_identity=True,
    )
    supervisor._validate_spec_structure(
        spec, loaded_spec_path=RUNTIME_SPEC_PATH,
    )
    inputs = _release_inputs_from_spec(spec)
    closure = validate_release_closure(
        **inputs, shadow_reader=shadow_reader, manager_reader=manager_reader,
        summary_runner=summary_runner,
    )
    _validate_source_bindings(
        spec.get("source_bindings"), realization=closure["realization"],
        expected_prior_identity=closure.get("prior_identity"),
    )
    artifacts = spec["artifacts"]
    for key in ("root", "heartbeat_dir", "systemd_invocation_dir"):
        _private_directory(Path(str(artifacts[key])))
    directory_keys = {"root", "heartbeat_dir", "systemd_invocation_dir"}
    if any(
        os.path.lexists(Path(str(value)))
        for key, value in artifacts.items() if key not in directory_keys
    ) or any(os.path.lexists(path) for path in (
        Path(str(spec["environment"]["gpu_lease_path"])),
        Path(str(spec["environment"]["gpu_lease_tombstone_path"])),
    )):
        raise PermissionError("runtime/GPU artifact namespace is not pristine")
    issued = datetime.now(timezone.utc)
    body = {
        "schema_version": supervisor.RUNTIME_LAUNCH_AUTHORIZATION_SCHEMA,
        "authorization_kind": "runtime_launch", "instruction_id": INSTRUCTION_ID,
        "authorization_basis": AUTHORIZATION_BASIS, "authorized_uid": os.getuid(),
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (
            issued + timedelta(seconds=validity_seconds)
        ).isoformat().replace("+00:00", "Z"),
        **_environment_authorization_bindings(spec),
        **supervisor._verify_scientific_preaccess_bindings(spec),
        "candidate": spec["candidate"], "stage_id": spec["stage_id"],
        "attempt_id": spec["attempt_id"], "attempt_ordinal": 2,
        "prior_attempt_count": 1, "fresh_attempt_authorized": True,
        "D_R_payload_authorized": True, "D_V_payload_authorized": False,
        "D_T_payload_authorized": False, "training_authorized": False,
        "resume_allowed": False, "automatic_retry_allowed": False,
        "runtime_spec_fingerprint": spec["runtime_spec_fingerprint"],
        "runtime_spec_file_sha256": spec_identity["file_sha256"],
        "runtime_spec_v2_fingerprint": spec["runtime_spec_fingerprint"],
        "runtime_spec_v2_file_sha256": spec_identity["file_sha256"],
        "supervisor_v2_source_closure_fingerprint": stable_fingerprint(spec["source_bindings"]),
        "unit_fragment_sha256": spec["runtime"]["systemd"]["unit_fragment_file_sha256"],
        "preauthorization_D_R_payload_accessed": False,
        "preauthorization_D_V_payload_accessed": False,
        "preauthorization_D_T_payload_accessed": False,
    }
    preview = {
        **body,
        "authorization_fingerprint": stable_fingerprint(body),
    }
    validate_launch_authorization(
        preview, spec=spec, spec_identity=spec_identity,
    )
    written, written_identity = _write_sealed_bound(
        RUNTIME_LAUNCH_AUTHORIZATION_PATH, body,
        fingerprint_field="authorization_fingerprint",
    )
    validated = validate_launch_authorization(
        written, spec=spec, spec_identity=spec_identity,
    )
    consumer_view = supervisor._verify_actual_authorization(
        spec,
        spec_path=RUNTIME_SPEC_PATH,
        require_fresh=True,
    )
    if consumer_view != {
        "path": str(RUNTIME_LAUNCH_AUTHORIZATION_PATH),
        "authorization_fingerprint": validated[
            "authorization_fingerprint"
        ],
        "authorization_file_sha256": written_identity["file_sha256"],
    }:
        raise PermissionError(
            "runtime launch authorization producer/consumer views diverged"
        )
    live_written = _file_binding(RUNTIME_LAUNCH_AUTHORIZATION_PATH)
    if any(
        live_written[key] != written_identity[key]
        for key in ("file_sha256", "device", "inode", "owner_uid", "mode")
    ):
        raise PermissionError("runtime launch authorization identity was replaced")
    return validated


def _default_shadow_reader() -> Mapping[str, str]:
    return actual_realizer.query_shadow(runner=subprocess.run)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-spec")
    for option in (
        "environment-policy", "precleanup-receipt", "cleanup-plan",
        "cleanup-authorization", "cleanup-receipt", "stability-receipt",
        "postcleanup-audit", "integration-authorization", "integration-receipt",
        "unit-realization-authorization", "unit-realization-receipt",
    ):
        build.add_argument(f"--{option}", type=Path, required=True)
    authorize = sub.add_parser("authorize-launch")
    authorize.add_argument("--validity-seconds", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-spec":
        build_spec(
            policy_path=args.environment_policy,
            precleanup_path=args.precleanup_receipt,
            cleanup_plan_path=args.cleanup_plan,
            cleanup_authorization_path=args.cleanup_authorization,
            cleanup_receipt_path=args.cleanup_receipt,
            stability_path=args.stability_receipt,
            postcleanup_path=args.postcleanup_audit,
            integration_authorization_path=args.integration_authorization,
            integration_receipt_path=args.integration_receipt,
            realization_authorization_path=args.unit_realization_authorization,
            realization_receipt_path=args.unit_realization_receipt,
            shadow_reader=_default_shadow_reader,
            manager_reader=actual_realizer.collect_manager_generation,
        )
    else:
        authorize_launch(
            shadow_reader=_default_shadow_reader,
            manager_reader=actual_realizer.collect_manager_generation,
            validity_seconds=args.validity_seconds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
