#!/usr/bin/env python3
"""Create the immutable C4/B4 receipt-seal failure boundary.

This tool is forensic and metadata-only.  It validates the frozen B4, R4,
and E4 generations, reproduces the JSON canonicalization disagreement using
pure byte calculations, records the exact inert C4 state and future-output
absences, and creates one read-only terminal record.  It never imports or
calls the old B4 receipt sealer, an R4/E4 writer, a runtime launcher, a GPU
tool, or a scientific payload reader.

No durable argv, stdout, stderr, traceback, or process-result artifact from
the original B4 invocation survived.  The historical exit status is therefore
kept as a control-plane observation and is explicitly not independently
verifiable.  The exception below is only a post-hoc deterministic expectation.
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
    / "r2_preaccess_schema_compat_c4_receipt_seal_failure_terminal.json"
)
SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c4-"
    "receipt-seal-failure-terminal-v1"
)

CANDIDATE = "GCR-PACRE-v24"
STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
SCIENTIFIC_ATTEMPT_ID = "gcr_pacre_v24_D_R_zero_update_structural_r2"
SCIENTIFIC_ATTEMPT_ORDINAL = 2
RUNTIME_COMPATIBILITY_ID = "c4"
C4_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service"
)
C4_UNIT_FRAGMENT_PATH = (
    Path(f"/run/user/{os.getuid()}/systemd/user") / C4_UNIT_NAME
)
R14_DUMMY_UNIT_NAME = (
    "cure-lite-v24-supervisor-integration-"
    "supervisor-v2-dummy-compat-c4-r14-20260731c4000014.service"
)

B4_SOURCE_PATH = (
    REPOSITORY / "tools/cure_lite_v24_preaccess_schema_compatibility_c4.py"
).resolve()
R4_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c4.py"
).resolve()
R4_NATIVE_CANONICALIZER_SOURCE_PATH = (
    REPOSITORY / "tools/cure_lite_v24_actual_unit_realization.py"
).resolve()
E4_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_environment_preaccess_compat_c4.py"
).resolve()

SOURCE_BINDINGS: dict[str, tuple[Path, str]] = {
    "old_B4_compatibility_bridge": (
        B4_SOURCE_PATH,
        "ad660b7afe7ca87f690bc9565bd6674684c2b62824394751a39114a6efcf178a",
    ),
    "R4_unit_realization_wrapper": (
        R4_SOURCE_PATH,
        "8708f8a13d74623f510992e23c6c23e1c4bfe70db09092c04fe56d44d29c5b65",
    ),
    "R4_native_canonicalizer": (
        R4_NATIVE_CANONICALIZER_SOURCE_PATH,
        "0d66bc4007366588ed1393b21092cc57d58e0f7fca084f7266a00e6818703fd9",
    ),
    "E4_environment_wrapper": (
        E4_SOURCE_PATH,
        "f4335efdb3865efe68dbbb6aac5f7977fd2157452b557f83428e4dd4a5d8932b",
    ),
}

B4_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c4_authorization.json"
)
B4_AUTHORIZATION_SHA256 = (
    "372b073186e65eebb31e0bed78683721c8b09a3f8c8af592b723686e8939a67b"
)
B4_AUTHORIZATION_FINGERPRINT = (
    "7f3735bdea424a3a7987e943c438770b6d226622140ff946648909199c369503"
)
B4_AUTHORIZATION_ISSUED_AT = "2026-07-31T15:16:22.413088Z"
B4_AUTHORIZATION_EXPIRES_AT = "2026-07-31T15:21:22.413088Z"
B4_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c4_receipt.json"
)

R4_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c4_unit_realization_authorization.json"
)
R4_AUTHORIZATION_SHA256 = (
    "78ba5ebbec468717187cb2f6f00c48d57a416e1b5bf78b829f762f76566ee8ae"
)
R4_AUTHORIZATION_FINGERPRINT = (
    "543f794fd27e6277471eb2e52ab290a228415091c3071070cf3f0920c3d28c10"
)
R4_AUTHORIZATION_OLD_B4_ASCII_FINGERPRINT = (
    "11b4f19ae10d7b032af4eb7611e8b36155be6cf577149450128d6b439b14cb44"
)
R4_AUTHORIZATION_ASCII_FILE_SHA256 = (
    "bc73c31fd53543cda288354a522e0e0880331d3a82642d2ed253ea494888ccbe"
)
R4_AUTHORIZATION_ASCII_FILE_SIZE = 19856
R4_AUTHORIZATION_ISSUED_AT = "2026-07-31T15:16:40.032974Z"
R4_AUTHORIZATION_EXPIRES_AT = "2026-07-31T15:21:40.032974Z"

R4_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c4_unit_realization_receipt.json"
)
R4_RECEIPT_SHA256 = (
    "eda9a4263dd1ba66028f1676b713bbc3d32c80b8d11fb7d8b5809a0accd0ac85"
)
R4_RECEIPT_FINGERPRINT = (
    "016bdd614ce0c184b005db1e0d8a42fc6884279fad1e8d602035408bd92ec55f"
)
R4_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c4_unit_realization_terminal.json"
)
C4_FRAGMENT_SHA256 = (
    "d57a881190025ab570021719ace74c3106501f9cc3d4e29a3c85c73dc26e5597"
)

E4_SCOPE_HANDOFF_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_scope_handoff_preaccess_compat_c4.json"
)
E4_SCOPE_HANDOFF_SHA256 = (
    "2e2145521218272b92ffcf0e8d0fd88e19bd72bd7a49fcc69a1d75cd0d012f9a"
)
E4_SCOPE_HANDOFF_FINGERPRINT = (
    "46e79fde3b75dadf7849a210e04ae0924bd8edc763a97916644f90285fe69dd8"
)
E4_SCOPE_HANDOFF_OLD_B4_ASCII_FINGERPRINT = (
    "28290b4ff3f2917aad872a5d8c5fba5a34860246f2d2413714fdffbd684872b5"
)
E4_STABILITY_ATTEMPT_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_attempt_preaccess_compat_c4.json"
)
E4_STABILITY_ATTEMPT_SHA256 = (
    "f8af44c4782960acc064b159ee117b015f4dfc7a328256a25f818ea3b9af4aed"
)
E4_STABILITY_ATTEMPT_FINGERPRINT = (
    "670afe791f8d4b7facca19eadacd899f0adbe6ab5be31c81b51f38dc60fb1c00"
)
E4_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c4.json"
)
E4_POLICY_SHA256 = (
    "2e4f2ddef0d2fcba40edf4f83f511c24d52a28b310ebc6c864615c81285f74f7"
)
E4_POLICY_FINGERPRINT = (
    "04b0ca6e8f91dcc332dc3773388419cfb1394db0d3761dacb16213dc9759f75a"
)
E4_STABILITY_RECEIPT_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c4.json"
)
E4_STABILITY_RECEIPT_SHA256 = (
    "d676aac771ae8473dc00d708aaf8035a93c58de5c7250f5f8123e0d25edb75e6"
)
E4_STABILITY_RECEIPT_FINGERPRINT = (
    "57b41839625188b03b904b0d7fd62404058a98b225f23f791e3d09bff7dd679b"
)
E4_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c4.json"
)
E4_POSTCLEANUP_SHA256 = (
    "e309de8d2aee7bfe3dbbda6dd07a7be608fc0a7fc67ba39716a47b11d4dd440e"
)
E4_POSTCLEANUP_FINGERPRINT = (
    "f3c1d9effd659edda94773558f4b9726a7cdc61b1a3681d797ac2f30b345c136"
)
E4_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_terminal_preaccess_compat_c4.json"
)

R14_INTEGRATION_ROOT = (
    EVIDENCE_ROOT
    / "supervisor_v2_systemd_integration_preaccess_compat_c4_r14"
)
L4_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_spec.json"
)
L4_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c4_"
        "runtime_launch_authorization.json"
    )
)
C4_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_artifacts"
)
C4_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_gpu_lease"
)
C4_RUN_ROOT_ALIAS_PATH = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c4"
)
C4_RESULT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_receipt.json"
)
DIRECT_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_spec.json"
)
DIRECT_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_runtime_launch_authorization.json"
)
DIRECT_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_artifacts"
)
DIRECT_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_gpu_lease"
)
SCIENTIFIC_RUN_ROOT = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2"
)
SCIENTIFIC_RESULT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_receipt.json"
)

ABSENT_OUTPUT_PATHS: dict[str, Path] = {
    "B4_compatibility_receipt": B4_RECEIPT_PATH,
    "R4_unit_terminal": R4_TERMINAL_PATH,
    "E4_environment_terminal": E4_TERMINAL_PATH,
    "r14_integration_root": R14_INTEGRATION_ROOT,
    "L4_C4_runtime_spec": L4_RUNTIME_SPEC_PATH,
    "L4_C4_runtime_launch_authorization": (
        L4_RUNTIME_LAUNCH_AUTHORIZATION_PATH
    ),
    "C4_runtime_artifacts": C4_RUNTIME_ARTIFACT_ROOT,
    "C4_gpu_lease": C4_GPU_LEASE_ROOT,
    "C4_run_alias": C4_RUN_ROOT_ALIAS_PATH,
    "C4_result_alias": C4_RESULT_ALIAS_PATH,
    "direct_runtime_spec": DIRECT_RUNTIME_SPEC_PATH,
    "direct_runtime_launch_authorization": (
        DIRECT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
    ),
    "direct_runtime_artifacts": DIRECT_RUNTIME_ARTIFACT_ROOT,
    "direct_gpu_lease": DIRECT_GPU_LEASE_ROOT,
    "scientific_run_root": SCIENTIFIC_RUN_ROOT,
    "scientific_result_receipt": SCIENTIFIC_RESULT_RECEIPT_PATH,
}

_INPUT_SPECS: dict[str, tuple[Path, str, str, str, bool, str]] = {
    "B4_authorization": (
        B4_AUTHORIZATION_PATH,
        B4_AUTHORIZATION_SHA256,
        "authorization_fingerprint",
        B4_AUTHORIZATION_FINGERPRINT,
        True,
        "cure-lite-v24-r2-preaccess-schema-compat-c4-authorization-v1",
    ),
    "R4_authorization": (
        R4_AUTHORIZATION_PATH,
        R4_AUTHORIZATION_SHA256,
        "authorization_fingerprint",
        R4_AUTHORIZATION_FINGERPRINT,
        False,
        "cure-lite-v24-actual-unit-realization-authorization-v1",
    ),
    "R4_receipt": (
        R4_RECEIPT_PATH,
        R4_RECEIPT_SHA256,
        "receipt_fingerprint",
        R4_RECEIPT_FINGERPRINT,
        False,
        "cure-lite-v24-actual-unit-realization-receipt-v1",
    ),
    "E4_scope_handoff": (
        E4_SCOPE_HANDOFF_PATH,
        E4_SCOPE_HANDOFF_SHA256,
        "scope_handoff_fingerprint",
        E4_SCOPE_HANDOFF_FINGERPRINT,
        False,
        (
            "cure-lite-v24-runtime-environment-scope-handoff-"
            "preaccess-compat-c4-v1"
        ),
    ),
    "E4_stability_attempt": (
        E4_STABILITY_ATTEMPT_PATH,
        E4_STABILITY_ATTEMPT_SHA256,
        "stability_attempt_fingerprint",
        E4_STABILITY_ATTEMPT_FINGERPRINT,
        False,
        (
            "cure-lite-v24-runtime-environment-stability-attempt-"
            "preaccess-compat-c4-v1"
        ),
    ),
    "E4_policy": (
        E4_POLICY_PATH,
        E4_POLICY_SHA256,
        "policy_fingerprint",
        E4_POLICY_FINGERPRINT,
        False,
        "cure-lite-v24-runtime-environment-policy-v1",
    ),
    "E4_stability_receipt": (
        E4_STABILITY_RECEIPT_PATH,
        E4_STABILITY_RECEIPT_SHA256,
        "stability_receipt_fingerprint",
        E4_STABILITY_RECEIPT_FINGERPRINT,
        False,
        "cure-lite-v24-runtime-environment-stability-receipt-v1",
    ),
    "E4_postcleanup": (
        E4_POSTCLEANUP_PATH,
        E4_POSTCLEANUP_SHA256,
        "receipt_fingerprint",
        E4_POSTCLEANUP_FINGERPRINT,
        False,
        "cure-lite-v24-runtime-environment-audit-receipt-v1",
    ),
}

_AUTHORIZED_SOURCE_LABELS = frozenset(
    {
        "c1_failure_terminalizer",
        "c2_mode_contract_failure_terminalizer",
        "c2_prewrite_failure_terminalizer",
        "c3_environment_failure_terminalizer",
        "compat_adapter",
        "compat_bridge",
        "compat_environment_wrapper",
        "compat_release",
        "compat_supervisor",
        "compat_unit_realizer",
        "compat_unit_template",
        "r14_dummy_child",
        "r14_dummy_unit_template",
        "r14_integration_wrapper",
        "r14_shared_realizer",
    }
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "identity",
        "terminalizer_source_root",
        "failed_generation_roots",
        "B4_authorization_root",
        "metadata_success_closure",
        "authorization_expiry",
        "b4_receipt_seal_failure",
        "deterministic_reproduction",
        "original_execution_observation",
        "historical_state_observation",
        "payload_observation",
        "continuation_policy",
        "terminal_fingerprint",
    }
)

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
    "NeedDaemonReload",
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

_PAYLOAD_OBSERVATION = {
    "D_R_payload_accessed": False,
    "D_V_payload_accessed": False,
    "D_T_payload_accessed": False,
    "gpu_compute_accessed": False,
    "training_started": False,
    "scientific_samples_processed": 0,
    "optimizer_steps": 0,
    "parameter_updates": 0,
    "environment_metadata_samples": 2,
    "scientific_attempt_consumed": False,
    "zero_update_basis": (
        "R4/E4 grant no payload authority; the C4 unit had no invocation; "
        "r14/L4/runtime/scientific outputs were absent at observation"
    ),
}

_CONTINUATION_POLICY = {
    "automatic_retry": False,
    "same_c4_reentry": False,
    "same_c4_reauthorization": False,
    "same_c4_source_repair": False,
    "same_c4_loader_patch": False,
    "same_c4_receipt_seal_reentry": False,
    "r4_e4_reentry": False,
    "r14_l4_runtime_scientific_launch": False,
    "c5_required": True,
    "new_explicit_authorization_required": True,
    "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
    "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
    "scientific_attempt_id_unchanged": True,
    "scientific_attempt_ordinal_unchanged": True,
    "b4_authorization_consumed": True,
    "r4_unit_realization_consumed": True,
    "e4_metadata_attempt_consumed": True,
    "e4_metadata_attempt_successful": True,
    "runtime_launch_consumed": False,
    "runtime_materialization_consumed": False,
    "scientific_attempt_consumed": False,
    "terminal_grants_c5_reuse_authority": False,
}

_ORIGINAL_EXECUTION_OBSERVATION = {
    "observation_kind": "historical_control_plane_observation",
    "control_plane_observed_exit_code": 1,
    "attempt_count": 1,
    "automatic_retry": False,
    "durable_original_execution_artifact": False,
    "durable_process_result_artifact_available": False,
    "exit_code_independently_verifiable": False,
    "original_call_artifact_available": False,
    "original_argv_claimed": False,
    "original_stdout_artifact_available": False,
    "original_stdout_claimed": False,
    "original_stderr_artifact_available": False,
    "original_stderr_claimed": False,
    "original_traceback_artifact_available": False,
    "original_traceback_claimed": False,
    "original_failure_time_claimed": False,
}

_EXPECTED_EXCEPTION_MESSAGE = (
    "sealed fingerprint changed: " + str(R4_AUTHORIZATION_PATH.absolute())
)


def _canonical_bytes(
    value: object,
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


def _fingerprint(
    value: Mapping[str, object],
    *,
    ensure_ascii: bool,
) -> str:
    return hashlib.sha256(
        _canonical_bytes(value, ensure_ascii=ensure_ascii)
    ).hexdigest()


def _deep_exact_equal(left: object, right: object) -> bool:
    try:
        return _canonical_bytes(left) == _canonical_bytes(right)
    except (TypeError, ValueError):
        return False


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


def _reject_duplicate_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _parse_json(raw: bytes, *, path: Path) -> dict[str, object]:
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PermissionError(f"sealed JSON is malformed: {path}") from error
    if not isinstance(value, dict):
        raise PermissionError(f"sealed JSON root is not an object: {path}")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(value, field)) for field in _STAT_FIELDS)


def _root_from(
    target: Path,
    raw: bytes,
    observed: os.stat_result,
) -> dict[str, object]:
    return {
        "path": str(target.absolute()),
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
    """Read one regular file through stable no-follow parent/file fds."""
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
        if (opened_parent.st_dev, opened_parent.st_ino) != (
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
        linked_parent = parent.lstat()
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
            or (os.fstat(parent_fd).st_dev, os.fstat(parent_fd).st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or not stat.S_ISDIR(linked_parent.st_mode)
            or stat.S_ISLNK(linked_parent.st_mode)
            or (linked_parent.st_dev, linked_parent.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
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
    expected_schema: str,
) -> tuple[dict[str, object], dict[str, object], bytes]:
    raw, root = _read_regular(
        path,
        expected_sha256=expected_sha256,
        expected_mode=0o444,
    )
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise PermissionError(f"sealed JSON layout changed: {path}")
    value = _parse_json(raw[:-1], path=path)
    if raw != _canonical_bytes(value, ensure_ascii=ensure_ascii) + b"\n":
        raise PermissionError(f"sealed JSON is not canonical: {path}")
    body = dict(value)
    claimed = body.pop(fingerprint_field, None)
    if (
        value.get("schema_version") != expected_schema
        or claimed != expected_fingerprint
        or _fingerprint(body, ensure_ascii=ensure_ascii)
        != expected_fingerprint
    ):
        raise PermissionError(f"sealed JSON fingerprint changed: {path}")
    return value, root, raw


def _assert_generation_binding(
    embedded: object,
    live_root: Mapping[str, object],
    *,
    name: str,
) -> None:
    if not isinstance(embedded, Mapping):
        raise PermissionError(f"{name} generation binding is missing")
    fields = {
        "path",
        "file_sha256",
        "device",
        "inode",
        "owner_uid",
        "owner_gid",
        "mode",
        "nlink",
        "size",
        "mtime_ns",
        "ctime_ns",
    }
    compared = fields.intersection(embedded)
    if not {"path", "file_sha256"}.issubset(compared) or any(
        embedded.get(field) != live_root.get(field) for field in compared
    ):
        raise PermissionError(f"{name} generation binding changed")


def _source_generations() -> dict[str, object]:
    return {
        label: _read_regular(path, expected_sha256=digest)[1]
        for label, (path, digest) in SOURCE_BINDINGS.items()
    }


def _validate_authorized_source_roots(
    b4: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    embedded = b4.get("compatibility_source_roots")
    if not isinstance(embedded, Mapping) or set(embedded) != set(
        _AUTHORIZED_SOURCE_LABELS
    ):
        raise PermissionError("B4 authorized source-root set changed")
    live: dict[str, dict[str, object]] = {}
    for label in sorted(_AUTHORIZED_SOURCE_LABELS):
        binding = embedded.get(label)
        if not isinstance(binding, Mapping):
            raise PermissionError(f"B4 source root is missing: {label}")
        path = binding.get("path")
        digest = binding.get("file_sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise PermissionError(f"B4 source root is malformed: {label}")
        raw, root = _read_regular(Path(path), expected_sha256=digest)
        del raw
        _assert_generation_binding(binding, root, name=f"B4 {label}")
        live[label] = root
    return live


def _validate_b4_authorization(value: Mapping[str, object]) -> None:
    mutation = value.get("mutation_authority")
    scientific = value.get("scientific_authority")
    if (
        value.get("candidate") != CANDIDATE
        or value.get("stage_id") != STAGE_ID
        or value.get("scientific_attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or value.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or value.get("runtime_compatibility_id") != "c4"
        or value.get("authorized_uid") != os.getuid()
        or value.get("issued_at_utc") != B4_AUTHORIZATION_ISSUED_AT
        or value.get("created_at_utc") != B4_AUTHORIZATION_ISSUED_AT
        or value.get("expires_at_utc") != B4_AUTHORIZATION_EXPIRES_AT
        or value.get("authorization_basis")
        != "user instruction: 修改后运行"
        or value.get("instruction_id")
        != "user-2026-07-30-modify-then-run-v1"
        or value.get("materialization_consumed") is not False
        or any(
            value.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
                "gpu_accessed",
                "training_started",
            )
        )
        or not isinstance(mutation, Mapping)
        or mutation.get("c4_unit_realization_authorized") is not True
        or mutation.get("compatibility_receipt_creation_authorized")
        is not True
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
        raise PermissionError("B4 authorization identity/authority changed")


def _validate_r4_authorization(
    value: Mapping[str, object],
    *,
    b4_root: Mapping[str, object],
    sources: Mapping[str, object],
) -> None:
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
    if (
        value.get("candidate") != CANDIDATE
        or value.get("stage_id") != STAGE_ID
        or value.get("attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or value.get("unit_name") != C4_UNIT_NAME
        or value.get("authorized_uid") != os.getuid()
        or value.get("issued_at_utc") != R4_AUTHORIZATION_ISSUED_AT
        or value.get("expires_at_utc") != R4_AUTHORIZATION_EXPIRES_AT
        or value.get("authorization_basis")
        != "user instruction: 修改后运行"
        or value.get("payload_authority") != "none"
        or value.get("start_authorized") is not False
        or value.get("enable_authorized") is not False
        or value.get("persistent_install_authorized") is not False
        or value.get("remove_authorized") is not False
        or any(
            value.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or value.get("actions")
        != [
            "install-runtime-static-fragment",
            "daemon-reload",
            "verify-static-shadow",
        ]
        or not isinstance(closure, Mapping)
        or closure.get("schema_version")
        != "cure-lite-v24-preaccess-compat-c4-unit-realization-closure-v1"
        or closure.get("runtime_compatibility_generation") != "c4"
        or closure.get("scientific_attempt_ordinal") != 2
        or closure.get("payload_authority") != "none"
        or closure.get("automatic_retry_authorized") is not False
        or closure.get("resume_authorized") is not False
        or not isinstance(bridge_window, Mapping)
        or bridge_window.get("file_sha256") != B4_AUTHORIZATION_SHA256
        or bridge_window.get("authorization_fingerprint")
        != B4_AUTHORIZATION_FINGERPRINT
        or not isinstance(bridge_root, Mapping)
        or bridge_root.get("file_sha256") != B4_AUTHORIZATION_SHA256
        or bridge_root.get("fingerprint")
        != B4_AUTHORIZATION_FINGERPRINT
        or value.get("rendered_fragment", {}).get("sha256")
        != C4_FRAGMENT_SHA256
    ):
        raise PermissionError("R4 authorization identity/closure changed")
    _assert_generation_binding(bridge_root, b4_root, name="R4 B4 root")
    _assert_generation_binding(
        closure.get("bridge_validator_source_generation"),
        sources["old_B4_compatibility_bridge"],
        name="R4 old B4 source",
    )
    _assert_generation_binding(
        closure.get("compat_source_generation"),
        sources["R4_unit_realization_wrapper"],
        name="R4 wrapper source",
    )


def _validate_r4_receipt(
    value: Mapping[str, object],
    *,
    authorization: Mapping[str, object],
    authorization_root: Mapping[str, object],
) -> None:
    fragment = value.get("fragment_identity")
    shadow = value.get("full_static_shadow")
    if (
        value.get("candidate") != CANDIDATE
        or value.get("stage_id") != STAGE_ID
        or value.get("attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or value.get("unit_name") != C4_UNIT_NAME
        or value.get("authorization_path")
        != str(R4_AUTHORIZATION_PATH.absolute())
        or value.get("authorization_file_sha256")
        != authorization_root.get("file_sha256")
        or value.get("authorization_fingerprint")
        != authorization.get("authorization_fingerprint")
        or value.get("passed") is not True
        or value.get("static") is not True
        or value.get("enabled") is not False
        or value.get("started") is not False
        or value.get("removed") is not False
        or value.get("runtime_spec_absent_at_receipt") is not True
        or value.get("payload_authority") != "none"
        or any(
            value.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or value.get("completed_actions")
        != [
            "install-runtime-static-fragment",
            "daemon-reload",
            "verify-static-shadow",
        ]
        or not isinstance(fragment, Mapping)
        or fragment.get("path") != str(C4_UNIT_FRAGMENT_PATH)
        or fragment.get("file_sha256") != C4_FRAGMENT_SHA256
        or fragment.get("mode") != 0o600
        or fragment.get("owner_uid") != os.getuid()
        or fragment.get("nlink") != 1
        or not isinstance(shadow, Mapping)
        or shadow.get("Id") != C4_UNIT_NAME
        or shadow.get("LoadState") != "loaded"
        or shadow.get("ActiveState") != "inactive"
        or shadow.get("SubState") != "dead"
        or shadow.get("UnitFileState") != "static"
        or shadow.get("FragmentPath") != str(C4_UNIT_FRAGMENT_PATH)
        or shadow.get("InvocationID") != ""
        or shadow.get("Restart") != "no"
        or shadow.get("NRestarts") != "0"
        or shadow.get("NeedDaemonReload") != "no"
    ):
        raise PermissionError("R4 PASS/static receipt changed")


def _validate_e4_semantics(
    values: Mapping[str, Mapping[str, object]],
    roots: Mapping[str, Mapping[str, object]],
) -> None:
    handoff = values["E4_scope_handoff"]
    attempt = values["E4_stability_attempt"]
    policy = values["E4_policy"]
    stability = values["E4_stability_receipt"]
    postcleanup = values["E4_postcleanup"]
    realization = handoff.get("realization")
    sampling = handoff.get("sampling")
    attempt_handoff = attempt.get("scope_handoff_root")
    attempt_policy = attempt.get("policy_root")
    stability_roots = stability.get("root_evidence")
    if (
        handoff.get("runtime_compatibility_id") != "c4"
        or handoff.get("payload_authority") != "none"
        or handoff.get("materialization_consumed") is not False
        or handoff.get("gpu_accessed") is not False
        or handoff.get("training_started") is not False
        or any(
            handoff.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or handoff.get("changed_fields")
        != ["require_target_ready", "target_unit_id"]
        or not isinstance(sampling, Mapping)
        or sampling.get("sample_count") != 2
        or sampling.get("sample_interval_seconds") != 30.0
        or not isinstance(realization, Mapping)
        or not _deep_exact_equal(
            realization.get("authorization"), values["R4_authorization"]
        )
        or not _deep_exact_equal(
            realization.get("receipt"), values["R4_receipt"]
        )
        or attempt.get("runtime_compatibility_id") != "c4"
        or attempt.get("target_unit_id") != C4_UNIT_NAME
        or attempt.get("payload_authority") != "none"
        or attempt.get("materialization_consumed") is not False
        or attempt.get("gpu_accessed") is not False
        or attempt.get("training_started") is not False
        or attempt.get("automatic_retry_allowed") is not False
        or attempt.get("resume_allowed") is not False
        or attempt.get("sample_count") != 2
        or attempt.get("sample_interval_seconds") != 30.0
        or any(
            attempt.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or not isinstance(attempt_handoff, Mapping)
        or attempt_handoff.get("file_sha256")
        != roots["E4_scope_handoff"].get("file_sha256")
        or attempt_handoff.get("scope_handoff_fingerprint")
        != E4_SCOPE_HANDOFF_FINGERPRINT
        or not isinstance(attempt_policy, Mapping)
        or attempt_policy.get("file_sha256")
        != roots["E4_policy"].get("file_sha256")
        or attempt_policy.get("policy_fingerprint")
        != E4_POLICY_FINGERPRINT
        or policy.get("candidate") != CANDIDATE
        or policy.get("scope") != "runtime-environment-stability"
        or policy.get("uid") != os.getuid()
        or policy.get("payload_authority") != "none"
        or any(
            policy.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or stability.get("passed") is not True
        or stability.get("blockers") != []
        or stability.get("receipt_kind") != "sampled"
        or stability.get("payload_authority") != "none"
        or stability.get("sample_count") != 2
        or stability.get("sample_interval_seconds") != 30.0
        or stability.get("minimum_window_seconds") != 30.0
        or not isinstance(stability.get("observed_window_seconds"), float)
        or stability.get("observed_window_seconds", 0.0) < 30.0
        or any(
            stability.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or not isinstance(stability_roots, Mapping)
        or stability_roots.get("policy", {}).get("file_sha256")
        != E4_POLICY_SHA256
        or stability_roots.get("policy", {}).get("policy_fingerprint")
        != E4_POLICY_FINGERPRINT
        or postcleanup.get("passed") is not True
        or postcleanup.get("error_type") is not None
        or postcleanup.get("error_message") is not None
        or any(
            postcleanup.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError("E4 five-piece PASS closure changed")
    authorization_identity = realization.get("authorization_identity")
    receipt_identity = realization.get("receipt_identity")
    _assert_generation_binding(
        authorization_identity,
        roots["R4_authorization"],
        name="E4 R4 authorization",
    )
    _assert_generation_binding(
        receipt_identity,
        roots["R4_receipt"],
        name="E4 R4 receipt",
    )


def _validate_fixed_inputs(
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, bytes],
    dict[str, object],
]:
    values: dict[str, dict[str, object]] = {}
    roots: dict[str, dict[str, object]] = {}
    raws: dict[str, bytes] = {}
    for label, spec in _INPUT_SPECS.items():
        value, root, raw = _load_canonical_json(
            spec[0],
            expected_sha256=spec[1],
            fingerprint_field=spec[2],
            expected_fingerprint=spec[3],
            ensure_ascii=spec[4],
            expected_schema=spec[5],
        )
        values[label] = value
        roots[label] = root
        raws[label] = raw

    sources = _source_generations()
    _validate_b4_authorization(values["B4_authorization"])
    authorized_sources = _validate_authorized_source_roots(
        values["B4_authorization"]
    )
    _validate_r4_authorization(
        values["R4_authorization"],
        b4_root=roots["B4_authorization"],
        sources=sources,
    )
    _validate_r4_receipt(
        values["R4_receipt"],
        authorization=values["R4_authorization"],
        authorization_root=roots["R4_authorization"],
    )
    _validate_e4_semantics(values, roots)

    fragment_raw, fragment_root = _read_regular(
        C4_UNIT_FRAGMENT_PATH,
        expected_sha256=C4_FRAGMENT_SHA256,
        expected_mode=0o600,
    )
    rendered = values["R4_authorization"].get("rendered_fragment")
    if (
        not isinstance(rendered, Mapping)
        or not isinstance(rendered.get("utf8_text"), str)
        or fragment_raw != rendered["utf8_text"].encode("utf-8")
    ):
        raise PermissionError("installed C4 fragment binding changed")
    roots["C4_unit_fragment"] = fragment_root
    raws["C4_unit_fragment"] = fragment_raw
    generations = {
        **sources,
        "B4_authorized_source_roots": values["B4_authorization"][
            "compatibility_source_roots"
        ],
        "B4_authorized_live_generations": authorized_sources,
    }
    return values, roots, raws, generations


def _old_b4_fingerprint_check(
    value: Mapping[str, object],
    *,
    path: Path,
    fingerprint_field: str,
) -> None:
    body = dict(value)
    claimed = body.pop(fingerprint_field, None)
    if claimed != _fingerprint(body, ensure_ascii=True):
        raise PermissionError(f"sealed fingerprint changed: {path.absolute()}")


def _deterministic_reproduction(
    values: Mapping[str, Mapping[str, object]],
    raws: Mapping[str, bytes],
) -> dict[str, object]:
    r4_authorization = values["R4_authorization"]
    r4_body = dict(r4_authorization)
    r4_claimed = r4_body.pop("authorization_fingerprint", None)
    producer = _fingerprint(r4_body, ensure_ascii=False)
    consumer = _fingerprint(r4_body, ensure_ascii=True)
    ascii_file = _canonical_bytes(r4_authorization, ensure_ascii=True) + b"\n"
    try:
        _old_b4_fingerprint_check(
            r4_authorization,
            path=R4_AUTHORIZATION_PATH,
            fingerprint_field="authorization_fingerprint",
        )
    except PermissionError as error:
        if type(error) is not PermissionError or error.args != (
            _EXPECTED_EXCEPTION_MESSAGE,
        ):
            raise PermissionError(
                "old B4 failure did not reproduce exactly"
            ) from error
    else:
        raise PermissionError("old B4 failure no longer reproduces")

    r4_receipt = values["R4_receipt"]
    receipt_body = dict(r4_receipt)
    receipt_claimed = receipt_body.pop("receipt_fingerprint", None)
    receipt_utf8 = _fingerprint(receipt_body, ensure_ascii=False)
    receipt_ascii = _fingerprint(receipt_body, ensure_ascii=True)

    handoff = values["E4_scope_handoff"]
    handoff_body = dict(handoff)
    handoff_claimed = handoff_body.pop("scope_handoff_fingerprint", None)
    handoff_utf8 = _fingerprint(handoff_body, ensure_ascii=False)
    handoff_ascii = _fingerprint(handoff_body, ensure_ascii=True)

    if (
        r4_claimed != producer != consumer
        or producer != R4_AUTHORIZATION_FINGERPRINT
        or consumer != R4_AUTHORIZATION_OLD_B4_ASCII_FINGERPRINT
        or raws["R4_authorization"]
        != _canonical_bytes(r4_authorization, ensure_ascii=False) + b"\n"
        or raws["R4_authorization"] == ascii_file
        or hashlib.sha256(ascii_file).hexdigest()
        != R4_AUTHORIZATION_ASCII_FILE_SHA256
        or len(ascii_file) != R4_AUTHORIZATION_ASCII_FILE_SIZE
        or receipt_claimed != receipt_utf8
        or receipt_utf8 != receipt_ascii
        or receipt_utf8 != R4_RECEIPT_FINGERPRINT
        or handoff_claimed != handoff_utf8
        or handoff_utf8 != E4_SCOPE_HANDOFF_FINGERPRINT
        or handoff_ascii != E4_SCOPE_HANDOFF_OLD_B4_ASCII_FINGERPRINT
        or handoff_utf8 == handoff_ascii
    ):
        raise PermissionError("canonicalization reproduction changed")

    return {
        "observation_kind": "post_hoc_pure_byte_recalculation",
        "old_B4_module_imported": False,
        "old_B4_loader_called": False,
        "old_B4_seal_called": False,
        "R4_E4_writer_called": False,
        "retry_or_replay_performed": False,
        "systemd_mutation_performed": False,
        "sleep_performed": False,
        "gpu_or_payload_accessed": False,
        "load_order": [
            "R4_authorization",
            "R4_receipt",
            "E4_scope_handoff",
            "E4_stability_attempt",
            "E4_policy",
            "E4_stability_receipt",
            "E4_postcleanup",
        ],
        "first_failure_stage": "R4_authorization_fingerprint_validation",
        "R4_authorization": {
            "path": str(R4_AUTHORIZATION_PATH.absolute()),
            "raw_file_sha256": R4_AUTHORIZATION_SHA256,
            "claimed_producer_fingerprint": r4_claimed,
            "producer_utf8_recomputed_fingerprint": producer,
            "old_B4_ascii_recomputed_fingerprint": consumer,
            "raw_is_utf8_canonical": True,
            "raw_is_ascii_canonical": False,
            "hypothetical_ascii_file_sha256": (
                R4_AUTHORIZATION_ASCII_FILE_SHA256
            ),
            "hypothetical_ascii_file_size": R4_AUTHORIZATION_ASCII_FILE_SIZE,
            "unicode_source_field": "authorization_basis",
            "unicode_source_value": "user instruction: 修改后运行",
        },
        "R4_receipt_control": {
            "path": str(R4_RECEIPT_PATH.absolute()),
            "claimed_fingerprint": receipt_claimed,
            "utf8_recomputed_fingerprint": receipt_utf8,
            "ascii_recomputed_fingerprint": receipt_ascii,
            "dual_profile_equal": True,
            "profile_mismatch": False,
        },
        "E4_scope_handoff_latent_mismatch": {
            "path": str(E4_SCOPE_HANDOFF_PATH.absolute()),
            "claimed_producer_fingerprint": handoff_claimed,
            "producer_utf8_recomputed_fingerprint": handoff_utf8,
            "old_B4_ascii_recomputed_fingerprint": handoff_ascii,
            "profile_mismatch": True,
            "producer_specific_loader_required": True,
        },
        "expected_exception_type": "PermissionError",
        "expected_exception_message": _EXPECTED_EXCEPTION_MESSAGE,
        "expected_exception_observation_kind": (
            "post_hoc_deterministic_expectation"
        ),
        "first_failure_reproduced": True,
    }


def _b4_authorization_record(
    values: Mapping[str, Mapping[str, object]],
    roots: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    value = values["B4_authorization"]
    return {
        "root": roots["B4_authorization"],
        "schema_version": value["schema_version"],
        "fingerprint_field": "authorization_fingerprint",
        "authorization_fingerprint": B4_AUTHORIZATION_FINGERPRINT,
        "canonical_profile": "compact_sorted_ensure_ascii_true_utf8",
        "issued_at_utc": B4_AUTHORIZATION_ISSUED_AT,
        "expires_at_utc": B4_AUTHORIZATION_EXPIRES_AT,
        "authorized_source_roots": value["compatibility_source_roots"],
    }


def _metadata_success_closure(
    values: Mapping[str, Mapping[str, object]],
    roots: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    five_piece: dict[str, object] = {}
    for label in (
        "E4_scope_handoff",
        "E4_stability_attempt",
        "E4_policy",
        "E4_stability_receipt",
        "E4_postcleanup",
    ):
        spec = _INPUT_SPECS[label]
        five_piece[label] = {
            "root": roots[label],
            "schema_version": spec[5],
            "fingerprint_field": spec[2],
            "fingerprint": spec[3],
            "canonical_profile": "compact_sorted_ensure_ascii_false_utf8",
        }
    return {
        "r4_unit_realization_passed": True,
        "r4_static_unit_verified": True,
        "e4_scope_handoff_present": True,
        "e4_stability_attempt_count": 1,
        "e4_environment_sample_count": 2,
        "e4_stability_passed": True,
        "e4_postcleanup_passed": True,
        "c4_compatibility_receipt_present": False,
        "R4_authorization": {
            "root": roots["R4_authorization"],
            "schema_version": values["R4_authorization"]["schema_version"],
            "fingerprint_field": "authorization_fingerprint",
            "fingerprint": R4_AUTHORIZATION_FINGERPRINT,
            "canonical_profile": "compact_sorted_ensure_ascii_false_utf8",
        },
        "R4_receipt": {
            "root": roots["R4_receipt"],
            "schema_version": values["R4_receipt"]["schema_version"],
            "fingerprint_field": "receipt_fingerprint",
            "fingerprint": R4_RECEIPT_FINGERPRINT,
            "authorization_file_sha256": R4_AUTHORIZATION_SHA256,
            "authorization_fingerprint": R4_AUTHORIZATION_FINGERPRINT,
            "passed": True,
            "static": True,
            "enabled": False,
            "started": False,
            "removed": False,
            "runtime_spec_absent_at_receipt": True,
        },
        "C4_unit_fragment_root": roots["C4_unit_fragment"],
        "E4_five_piece_roots": five_piece,
    }


def _failure_record() -> dict[str, object]:
    return {
        "first_rejected_path": str(R4_AUTHORIZATION_PATH.absolute()),
        "fingerprint_field": "authorization_fingerprint",
        "producer_canonical_profile": (
            "compact_sorted_ensure_ascii_false_utf8"
        ),
        "consumer_canonical_profile": (
            "compact_sorted_ensure_ascii_true_utf8"
        ),
        "producer_fingerprint": R4_AUTHORIZATION_FINGERPRINT,
        "consumer_recomputed_fingerprint": (
            R4_AUTHORIZATION_OLD_B4_ASCII_FINGERPRINT
        ),
        "profile_mismatch": True,
        "first_rejected_unicode_field": "authorization_basis",
        "first_rejected_unicode_value": "user instruction: 修改后运行",
        "R4_authorization_loaded_before_R4_receipt": True,
        "R4_authorization_loaded_before_E4_five_piece": True,
        "receipt_writer_reached": False,
        "receipt_sealed": False,
        "expected_exception_type": "PermissionError",
        "expected_exception_message": _EXPECTED_EXCEPTION_MESSAGE,
        "expected_exception_observation_kind": (
            "post_hoc_deterministic_expectation"
        ),
    }


def _authorization_expiry(
    values: Mapping[str, Mapping[str, object]],
    *,
    observed_at: datetime,
) -> dict[str, object]:
    b4_expiry = _parse_utc(
        values["B4_authorization"].get("expires_at_utc"),
        name="B4 authorization expiry",
    )
    r4_expiry = _parse_utc(
        values["R4_authorization"].get("expires_at_utc"),
        name="R4 authorization expiry",
    )
    if observed_at <= b4_expiry or observed_at <= r4_expiry:
        raise PermissionError("B4/R4 authorizations are not both expired")
    return {
        "observed_at_utc": _format_utc(observed_at),
        "B4_issued_at_utc": B4_AUTHORIZATION_ISSUED_AT,
        "B4_expires_at_utc": B4_AUTHORIZATION_EXPIRES_AT,
        "B4_expired": True,
        "B4_compatibility_receipt_absent_at_observation": True,
        "B4_sealed_by_compatibility_receipt": False,
        "R4_issued_at_utc": R4_AUTHORIZATION_ISSUED_AT,
        "R4_expires_at_utc": R4_AUTHORIZATION_EXPIRES_AT,
        "R4_expired": True,
        "R4_PASS_receipt_exists": True,
        "R4_receipt_created_at_utc": "2026-07-31T15:17:45.508407Z",
        "E4_scope_handoff_created_at_utc": "2026-07-31T15:18:13.092723Z",
        "E4_policy_created_at_utc": "2026-07-31T15:18:17.421732Z",
        "E4_stability_attempt_created_at_utc": (
            "2026-07-31T15:18:39.340216Z"
        ),
        "E4_stability_receipt_created_at_utc": (
            "2026-07-31T15:19:24.730431Z"
        ),
        "E4_postcleanup_created_at_utc": "2026-07-31T15:20:08.679742Z",
        "terminal_sealed_after_latest_authorization_expiry": True,
        "original_failure_time_claimed": False,
    }


def _read_unit_state(
    unit_name: str,
    *,
    expected: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    command = [
        "/usr/bin/systemctl",
        "--user",
        "show",
        unit_name,
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
        raise PermissionError(f"unit state query failed: {unit_name}")
    state: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" not in line:
            raise PermissionError(f"unit state output changed: {unit_name}")
        key, value = line.split("=", 1)
        if key in state:
            raise PermissionError(f"unit state property repeated: {unit_name}")
        state[key] = value
    if set(state) != set(_STATE_PROPERTIES):
        raise PermissionError(f"unit state properties changed: {unit_name}")
    common = (
        state.get("Id") == unit_name
        and state.get("ActiveState") == "inactive"
        and state.get("SubState") == "dead"
        and state.get("InvocationID") == ""
        and state.get("Restart") == "no"
        and state.get("NRestarts") == "0"
        and state.get("NeedDaemonReload") == "no"
    )
    if expected == "static":
        valid = (
            common
            and state.get("LoadState") == "loaded"
            and state.get("UnitFileState") == "static"
            and state.get("FragmentPath") == str(C4_UNIT_FRAGMENT_PATH)
        )
    elif expected == "not-found":
        valid = (
            common
            and state.get("LoadState") == "not-found"
            and state.get("UnitFileState") == ""
            and state.get("FragmentPath") == ""
        )
    else:
        raise ValueError(f"unknown expected unit state: {expected}")
    if not valid:
        raise PermissionError(f"unit is not exact {expected}/inert: {unit_name}")
    return state


def _observe_absence(path: Path) -> dict[str, object]:
    target = Path(path).absolute()
    parent = target.parent
    before = parent.lstat()
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or parent.resolve(strict=True) != parent
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o002
    ):
        raise PermissionError(f"unsafe absent-path parent: {parent}")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(parent_fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise PermissionError(f"absent-path parent changed: {parent}")
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise PermissionError(f"forbidden C4 output exists: {target}")
        linked = parent.lstat()
        if (
            (linked.st_dev, linked.st_ino) != (before.st_dev, before.st_ino)
            or stat.S_ISLNK(linked.st_mode)
            or parent.resolve(strict=True) != parent
        ):
            raise PermissionError(f"absent-path parent changed: {parent}")
        return {
            "path": str(target),
            "basename": target.name,
            "lexists": False,
            "parent_path": str(parent),
            "parent_device": opened.st_dev,
            "parent_inode": opened.st_ino,
            "parent_owner_uid": opened.st_uid,
            "parent_owner_gid": opened.st_gid,
            "parent_mode": stat.S_IMODE(opened.st_mode),
            "parent_nlink": opened.st_nlink,
        }
    finally:
        os.close(parent_fd)


def _observe_absences() -> dict[str, dict[str, object]]:
    return {
        label: _observe_absence(path)
        for label, path in ABSENT_OUTPUT_PATHS.items()
    }


def _exact_absent_paths() -> dict[str, str]:
    return {
        label: str(path.absolute())
        for label, path in ABSENT_OUTPUT_PATHS.items()
    }


def _collect_historical_state(
    *,
    observed_at: datetime,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, object]:
    before = _observe_absences()
    c4_state = _read_unit_state(
        C4_UNIT_NAME,
        expected="static",
        runner=runner,
    )
    r14_state = _read_unit_state(
        R14_DUMMY_UNIT_NAME,
        expected="not-found",
        runner=runner,
    )
    after = _observe_absences()
    if not _deep_exact_equal(before, after):
        raise PermissionError("C4 absence generation drifted during observation")
    return {
        "observed_at_utc": _format_utc(observed_at),
        "exact_absent_paths": _exact_absent_paths(),
        "absence_generation_roots": before,
        "all_required_paths_absent": True,
        "B4_authorization_unsealed_basis": (
            "B4 compatibility receipt absent at observation"
        ),
        "C4_unit_state": c4_state,
        "R14_dummy_unit_state": r14_state,
        "historical_observation_only": True,
        "future_state_authority": False,
        "archival_live_absence_recheck_required": False,
        "archival_live_manager_recheck_required": False,
    }


def _revalidate_open_creation_guard(
    *,
    expected_values: Mapping[str, object],
    expected_roots: Mapping[str, object],
    expected_raws: Mapping[str, object],
    expected_generations: Mapping[str, object],
    expected_terminalizer_source_root: Mapping[str, object],
    expected_absences: Mapping[str, object],
    expected_c4_state: Mapping[str, str],
    expected_r14_state: Mapping[str, str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    for phase in ("before", "after"):
        observed_absences = _observe_absences()
        values, roots, raws, generations = _validate_fixed_inputs()
        source = _read_regular(Path(__file__).resolve())[1]
        c4_state = _read_unit_state(
            C4_UNIT_NAME,
            expected="static",
            runner=runner,
        )
        r14_state = _read_unit_state(
            R14_DUMMY_UNIT_NAME,
            expected="not-found",
            runner=runner,
        )
        groups = (
            (observed_absences, expected_absences),
            (values, expected_values),
            (roots, expected_roots),
            (generations, expected_generations),
            (source, expected_terminalizer_source_root),
            (c4_state, expected_c4_state),
            (r14_state, expected_r14_state),
        )
        if raws != expected_raws or any(
            not _deep_exact_equal(observed, expected)
            for observed, expected in groups
        ):
            raise PermissionError(
                f"fixed C4 input/state drifted {phase} while terminal open"
            )


def _write_create_once(
    path: Path,
    body: Mapping[str, object],
    *,
    while_open_guard: Callable[[], None] | None = None,
    post_close_guard: Callable[[], None] | None = None,
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
        or stat.S_IMODE(parent_before.st_mode) & 0o022
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
        if (opened_parent.st_dev, opened_parent.st_ino) != (
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
        if while_open_guard is not None:
            while_open_guard()
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
            raise PermissionError("failure-terminal fd seal/readback changed")
        os.fsync(parent_fd)
        linked_parent = parent.lstat()
        if (
            (linked_parent.st_dev, linked_parent.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or stat.S_ISLNK(linked_parent.st_mode)
            or parent.resolve(strict=True) != parent
        ):
            raise PermissionError("failure-terminal parent changed after create")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    if post_close_guard is not None:
        post_close_guard()
    validated, _root = validate_archival(target)
    if not _deep_exact_equal(validated, payload):
        raise RuntimeError("failure-terminal archival readback changed")
    return validated


def create_terminal(
    path: Path | None = None,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    selected = TERMINAL_PATH if path is None else Path(path).absolute()
    if selected != TERMINAL_PATH.absolute():
        raise PermissionError("C4 receipt-seal terminal path is not fixed")
    if os.path.lexists(selected):
        raise FileExistsError("C4 receipt-seal terminal already exists")

    values, roots, raws, generations = _validate_fixed_inputs()
    reproduction = _deterministic_reproduction(values, raws)
    post_values, post_roots, post_raws, post_generations = (
        _validate_fixed_inputs()
    )
    if post_raws != raws or any(
        not _deep_exact_equal(observed, expected)
        for observed, expected in (
            (post_values, values),
            (post_roots, roots),
            (post_generations, generations),
        )
    ):
        raise PermissionError("pure reproduction changed frozen C4 inputs")

    observed_at = _utc_now()
    expiry = _authorization_expiry(values, observed_at=observed_at)
    historical = _collect_historical_state(
        observed_at=observed_at,
        runner=runner,
    )
    terminalizer_source_root = _read_regular(Path(__file__).resolve())[1]

    final_values, final_roots, final_raws, final_generations = (
        _validate_fixed_inputs()
    )
    if final_raws != raws or any(
        not _deep_exact_equal(observed, expected)
        for observed, expected in (
            (final_values, values),
            (final_roots, roots),
            (final_generations, generations),
        )
    ):
        raise PermissionError("fixed C4 evidence changed before terminal seal")

    body: dict[str, object] = {
        "schema_version": SCHEMA,
        "identity": {
            "candidate": CANDIDATE,
            "stage_id": STAGE_ID,
            "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
            "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
            "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
            "failure_stage": "B4_compatibility_receipt_seal",
            "sealed_at_utc": _format_utc(observed_at),
        },
        "terminalizer_source_root": terminalizer_source_root,
        "failed_generation_roots": generations,
        "B4_authorization_root": _b4_authorization_record(values, roots),
        "metadata_success_closure": _metadata_success_closure(values, roots),
        "authorization_expiry": expiry,
        "b4_receipt_seal_failure": _failure_record(),
        "deterministic_reproduction": reproduction,
        "original_execution_observation": dict(
            _ORIGINAL_EXECUTION_OBSERVATION
        ),
        "historical_state_observation": historical,
        "payload_observation": dict(_PAYLOAD_OBSERVATION),
        "continuation_policy": dict(_CONTINUATION_POLICY),
    }

    absences = historical.get("absence_generation_roots")
    c4_state = historical.get("C4_unit_state")
    r14_state = historical.get("R14_dummy_unit_state")
    if (
        not isinstance(absences, Mapping)
        or not isinstance(c4_state, Mapping)
        or not isinstance(r14_state, Mapping)
    ):
        raise PermissionError("historical creation guard disappeared")

    def guard() -> None:
        _revalidate_open_creation_guard(
            expected_values=values,
            expected_roots=roots,
            expected_raws=raws,
            expected_generations=generations,
            expected_terminalizer_source_root=terminalizer_source_root,
            expected_absences=absences,
            expected_c4_state=dict(c4_state),
            expected_r14_state=dict(r14_state),
            runner=runner,
        )

    return _write_create_once(
        selected,
        body,
        while_open_guard=guard,
        post_close_guard=guard,
    )


def _validate_identity(value: object) -> datetime:
    if not isinstance(value, Mapping) or set(value) != {
        "candidate",
        "stage_id",
        "scientific_attempt_id",
        "scientific_attempt_ordinal",
        "runtime_compatibility_id",
        "failure_stage",
        "sealed_at_utc",
    }:
        raise PermissionError("C4 failure-terminal identity changed")
    sealed_at = _parse_utc(
        value.get("sealed_at_utc"),
        name="C4 failure terminal sealed_at",
    )
    if (
        value.get("candidate") != CANDIDATE
        or value.get("stage_id") != STAGE_ID
        or value.get("scientific_attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or value.get("scientific_attempt_ordinal") != 2
        or value.get("runtime_compatibility_id") != "c4"
        or value.get("failure_stage") != "B4_compatibility_receipt_seal"
    ):
        raise PermissionError("C4 failure-terminal identity changed")
    return sealed_at


def _validate_expiry_record(
    value: object,
    *,
    sealed_at: datetime,
    values: Mapping[str, Mapping[str, object]],
) -> None:
    expected = _authorization_expiry(values, observed_at=sealed_at)
    if not _deep_exact_equal(value, expected):
        raise PermissionError("C4 authorization-expiry record changed")


def _validate_stored_unit_state(
    value: object,
    *,
    unit_name: str,
    expected: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_STATE_PROPERTIES):
        raise PermissionError(f"stored unit state changed: {unit_name}")
    common = (
        value.get("Id") == unit_name
        and value.get("ActiveState") == "inactive"
        and value.get("SubState") == "dead"
        and value.get("InvocationID") == ""
        and value.get("Restart") == "no"
        and value.get("NRestarts") == "0"
        and value.get("NeedDaemonReload") == "no"
    )
    if expected == "static":
        valid = (
            common
            and value.get("LoadState") == "loaded"
            and value.get("UnitFileState") == "static"
            and value.get("FragmentPath") == str(C4_UNIT_FRAGMENT_PATH)
        )
    else:
        valid = (
            common
            and value.get("LoadState") == "not-found"
            and value.get("UnitFileState") == ""
            and value.get("FragmentPath") == ""
        )
    if not valid:
        raise PermissionError(f"stored unit state changed: {unit_name}")


def _validate_historical_state(
    value: object,
    *,
    sealed_at: datetime,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "observed_at_utc",
        "exact_absent_paths",
        "absence_generation_roots",
        "all_required_paths_absent",
        "B4_authorization_unsealed_basis",
        "C4_unit_state",
        "R14_dummy_unit_state",
        "historical_observation_only",
        "future_state_authority",
        "archival_live_absence_recheck_required",
        "archival_live_manager_recheck_required",
    }:
        raise PermissionError("historical C4 state record changed")
    rows = value.get("absence_generation_roots")
    row_keys = {
        "path",
        "basename",
        "lexists",
        "parent_path",
        "parent_device",
        "parent_inode",
        "parent_owner_uid",
        "parent_owner_gid",
        "parent_mode",
        "parent_nlink",
    }
    if (
        _parse_utc(value.get("observed_at_utc"), name="C4 observation")
        != sealed_at
        or value.get("exact_absent_paths") != _exact_absent_paths()
        or value.get("all_required_paths_absent") is not True
        or value.get("B4_authorization_unsealed_basis")
        != "B4 compatibility receipt absent at observation"
        or value.get("historical_observation_only") is not True
        or value.get("future_state_authority") is not False
        or value.get("archival_live_absence_recheck_required") is not False
        or value.get("archival_live_manager_recheck_required") is not False
        or not isinstance(rows, Mapping)
        or set(rows) != set(ABSENT_OUTPUT_PATHS)
    ):
        raise PermissionError("historical C4 state record changed")
    for label, path in ABSENT_OUTPUT_PATHS.items():
        row = rows.get(label)
        if (
            not isinstance(row, Mapping)
            or set(row) != row_keys
            or row.get("path") != str(path.absolute())
            or row.get("basename") != path.name
            or row.get("lexists") is not False
            or row.get("parent_path") != str(path.absolute().parent)
            or row.get("parent_owner_uid") != os.getuid()
            or not isinstance(row.get("parent_device"), int)
            or not isinstance(row.get("parent_inode"), int)
            or not isinstance(row.get("parent_mode"), int)
            or row.get("parent_mode", 0) & 0o002
            or not isinstance(row.get("parent_nlink"), int)
        ):
            raise PermissionError(f"historical absence changed: {label}")
    _validate_stored_unit_state(
        value.get("C4_unit_state"),
        unit_name=C4_UNIT_NAME,
        expected="static",
    )
    _validate_stored_unit_state(
        value.get("R14_dummy_unit_state"),
        unit_name=R14_DUMMY_UNIT_NAME,
        expected="not-found",
    )


def validate_archival(
    path: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    selected = TERMINAL_PATH if path is None else Path(path).absolute()
    if selected != TERMINAL_PATH.absolute():
        raise PermissionError("C4 receipt-seal terminal path is not fixed")
    raw, terminal_root = _read_regular(selected, expected_mode=0o444)
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise PermissionError("C4 receipt-seal terminal layout changed")
    payload = _parse_json(raw[:-1], path=selected)
    if (
        set(payload) != _TOP_LEVEL_KEYS
        or payload.get("schema_version") != SCHEMA
        or raw != _canonical_bytes(payload) + b"\n"
    ):
        raise PermissionError("C4 receipt-seal terminal schema/layout changed")
    fingerprint = payload.get("terminal_fingerprint")
    body = dict(payload)
    body.pop("terminal_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != stable_fingerprint(
        body
    ):
        raise PermissionError("C4 receipt-seal terminal fingerprint changed")

    sealed_at = _validate_identity(payload.get("identity"))
    values, roots, raws, generations = _validate_fixed_inputs()
    reproduction = _deterministic_reproduction(values, raws)
    expected_groups = (
        (payload.get("failed_generation_roots"), generations),
        (payload.get("B4_authorization_root"), _b4_authorization_record(values, roots)),
        (
            payload.get("metadata_success_closure"),
            _metadata_success_closure(values, roots),
        ),
        (payload.get("b4_receipt_seal_failure"), _failure_record()),
        (payload.get("deterministic_reproduction"), reproduction),
        (
            payload.get("original_execution_observation"),
            _ORIGINAL_EXECUTION_OBSERVATION,
        ),
        (payload.get("payload_observation"), _PAYLOAD_OBSERVATION),
        (payload.get("continuation_policy"), _CONTINUATION_POLICY),
    )
    if any(
        not _deep_exact_equal(observed, expected)
        for observed, expected in expected_groups
    ):
        raise PermissionError("C4 receipt-seal terminal semantics changed")
    _validate_expiry_record(
        payload.get("authorization_expiry"),
        sealed_at=sealed_at,
        values=values,
    )
    _validate_historical_state(
        payload.get("historical_state_observation"),
        sealed_at=sealed_at,
    )

    terminalizer_root = _read_regular(Path(__file__).resolve())[1]
    if not _deep_exact_equal(
        payload.get("terminalizer_source_root"), terminalizer_root
    ):
        raise PermissionError("C4 terminalizer source lineage changed")

    # Archival validation intentionally does not inspect the current clock,
    # current systemd state, or current/future path absences.
    terminal_root["terminal_fingerprint"] = fingerprint
    terminal_root["schema_version"] = SCHEMA
    terminal_root["terminalizer_source_root"] = terminalizer_root
    return payload, terminal_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("create-terminal", allow_abbrev=False)
    subparsers.add_parser("validate-terminal", allow_abbrev=False)
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
