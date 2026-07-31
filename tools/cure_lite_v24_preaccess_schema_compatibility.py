#!/usr/bin/env python3
"""Append-only closure for the v24 r2 preaccess-schema prewrite failure.

This tool does not repair, replace, alias, or execute the frozen r2 runtime
chain.  It can only:

* seal a retrospective terminal for the already-recorded ``build-spec``
  failure;
* seal a short-lived authorization for one *runtime compatibility* generation
  while preserving scientific attempt ordinal 2; and
* seal a compatibility receipt which binds a new, independently named runtime
  closure and the unchanged scientific ``...split-access-audit-v1`` evidence.

No command in this module starts, enables, reloads, stops, or removes a unit.
No command imports a dataset, model, torch, or the scientific gate.  All three
outputs have fixed create-once paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
from typing import Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY
    / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).absolute()

FORENSIC_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_forensic_terminal.json"
)
COMPAT_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_authorization.json"
)
COMPAT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_receipt.json"
)
# Consumer-facing spelling used by the compatibility release.  Both names are
# the same fixed path; neither is a configurable output.
COMPATIBILITY_RECEIPT_PATH = COMPAT_RECEIPT_PATH

SCIENTIFIC_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_authorization.json"
)
SCIENTIFIC_ACCESS_AUDIT_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_access_audit.json"
)
OLD_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_spec.json"
)
OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_runtime_launch_authorization.json"
)
OLD_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_artifacts"
)
OLD_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_gpu_lease"
)
SCIENTIFIC_RESULT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_receipt.json"
)
SCIENTIFIC_RUN_ROOT = (
    REPOSITORY
    / "runs/irstd1k_stage_a_seed42/"
    "gcr_pacre_v24_D_R_structural_attempt_r2"
).absolute()

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
# These two paths are prohibited aliases.  A compatibility runtime must retain
# the original r2 scientific output identity, not create a second scientific
# run/result namespace.
COMPAT_RUN_ROOT_ALIAS_PATH = (
    REPOSITORY
    / "runs/irstd1k_stage_a_seed42/"
    "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c1"
).absolute()
COMPAT_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_receipt.json"
)

COMPAT_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service"
)
INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后运行"
COMPAT_POLICY_SOURCE_PATH = Path(__file__).resolve()
COMPAT_RELEASE_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_runtime_release_preaccess_compat_c1.py"
).absolute()
COMPAT_SUPERVISOR_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c1.py"
).absolute()
COMPAT_ADAPTER_SOURCE_PATH = (
    REPOSITORY
    / "tools/"
    "run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c1.py"
).absolute()
COMPAT_UNIT_REALIZER_SOURCE_PATH = (
    REPOSITORY
    / "tools/"
    "cure_lite_v24_actual_unit_realization_preaccess_compat_c1.py"
).absolute()
COMPAT_UNIT_TEMPLATE_PATH = (
    REPOSITORY
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service.template"
).absolute()

COMPAT_ENVIRONMENT_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c1.json"
)
COMPAT_ENVIRONMENT_STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c1.json"
)
COMPAT_ENVIRONMENT_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c1.json"
)
COMPAT_INTEGRATION_ROOT = (
    EVIDENCE_ROOT
    / "supervisor_v2_systemd_integration_preaccess_compat_c1_r10"
)
COMPAT_INTEGRATION_AUTHORIZATION_PATH = (
    COMPAT_INTEGRATION_ROOT / "control/authorization.json"
)
COMPAT_INTEGRATION_RECEIPT_PATH = (
    COMPAT_INTEGRATION_ROOT / "control/integration-receipt.json"
)
COMPAT_UNIT_REALIZATION_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c1_unit_realization_authorization.json"
)
COMPAT_UNIT_REALIZATION_RECEIPT_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c1_unit_realization_receipt.json"
)
COMPAT_UNIT_REALIZATION_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c1_unit_realization_terminal.json"
)

OLD_RELEASE_PATH = (
    REPOSITORY / "tools/cure_lite_v24_actual_runtime_release.py"
).absolute()
OLD_SUPERVISOR_PATH = (
    REPOSITORY / "tools/cure_lite_v24_runtime_supervisor.py"
).absolute()
OLD_ADAPTER_PATH = (
    REPOSITORY / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2.py"
).absolute()
PROTOCOL_VERIFIER_PATH = (
    REPOSITORY / "tools/gcr_pacre_v24_protocol.py"
).absolute()
OLD_UNIT_REALIZER_PATH = (
    REPOSITORY / "tools/cure_lite_v24_actual_unit_realization.py"
).absolute()
UNIT_RECOVERY_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_unit_realization_recovery_authorization.json"
)
UNIT_RECOVERY_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_unit_realization_recovery_receipt.json"
)

OLD_UNIT_NAME = "cure-lite-v24-gcr-pacre-dr-r2.service"
OLD_UNIT_FRAGMENT_PATH = Path(
    f"/run/user/{os.getuid()}/systemd/user/{OLD_UNIT_NAME}"
).absolute()
SYSTEMCTL = "/usr/bin/systemctl"

SESSION_PATH = Path(
    "/home/md0/ly/.codex/sessions/2026/07/28/"
    "rollout-2026-07-28T17-15-33-019fa802-0a9c-7ac3-bdd7-14aad146788f.jsonl"
).absolute()
SESSION_CALL_LINE = 22512
SESSION_OUTPUT_LINE = 22513
SESSION_CALL_ID = "call_fYljES6cQ2IXSYvQmD6z2Ag6"
SESSION_CALL_RAW_SHA256 = (
    "fe3685d462cff7ebc5b3588f074abc46befa32be14bce83bab13c45fc64ecb39"
)
SESSION_OUTPUT_RAW_SHA256 = (
    "1499b7ef4adfdb48ed1e75595fe3a724d956f6bad20a8713aad94f5d42217b4a"
)
SESSION_CALL_RECORD_WITH_LF_SHA256 = (
    "a9959fa90e9b209ab38e379196f74e8ce2949c78b36d46c1362472900f80289e"
)
SESSION_OUTPUT_RECORD_WITH_LF_SHA256 = (
    "09130fe3eec9ecc25c87b342bebd26809587c7f636d214ff6ae7a9d6af2778e1"
)
SESSION_CALL_ARGUMENTS_SHA256 = (
    "b3603e154f91a0bf7e1de21ca9e3be09abfbd7e176a2d32aa1cad2050d533675"
)
SESSION_OUTPUT_PAYLOAD_SHA256 = (
    "f8dd5e62d91460ed79adc241ed0986477e5c0f228fc41856b29a329153c1b490"
)

EXPECTED_FILE_SHA256 = {
    "scientific_authorization": (
        "a7e14f788ad6158d380125da0567d7e3187b353b8dca4949c22322314aa1ae38"
    ),
    "scientific_access_audit": (
        "6d7b492a0401ebd285e83b28a5f94494a88fadfa6a7be09c3d9624cf11c6de92"
    ),
    "old_release": (
        "258dcae12a7799ccf63a39dd191fce67170728fb38a705223c0bc1c9fd1b387d"
    ),
    "old_supervisor": (
        "b955ba8ffe869d324cc9319f8031180989746053d7ceec5e50bd12eb19faeeed"
    ),
    "old_adapter": (
        "5cbfd073d7df8f4257079c71e6f05110d31b383c48abe6e0e9127ee154785495"
    ),
    "protocol_verifier": (
        "d516c8c390e3fe84c7bd3644a419c8089cceedd2eb82f5f8ce32d9a8cb6766ca"
    ),
    "old_unit_realizer": (
        "0d66bc4007366588ed1393b21092cc57d58e0f7fca084f7266a00e6818703fd9"
    ),
    "unit_recovery_authorization": (
        "9439e509e174a0828a4efac36b18164fb1d4da983e1bd96998f2a03416cd050c"
    ),
    "unit_recovery_receipt": (
        "c1e30c7c3e8af351f39221bce925b912ee5abb64859391934591d244ca2c56f4"
    ),
    "old_unit_fragment": (
        "efe4e7194bc85154963d2379909dd8ea2990439f8058fa639d6b00fa7c5a33f6"
    ),
}

CANDIDATE = "GCR-PACRE-v24"
STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
SCIENTIFIC_ATTEMPT_ID = "gcr_pacre_v24_D_R_zero_update_structural_r2"
SCIENTIFIC_ATTEMPT_ORDINAL = 2
RUNTIME_COMPATIBILITY_ID = "c1"
SOURCE_CLOSURE_FINGERPRINT_103 = (
    "28d26759a68785e9c99917fcfa8b36430c7f6e5463282d66eeab5c711e425e9f"
)
SCIENTIFIC_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-D_R-structural-r2-authorization-v1"
)
SCIENTIFIC_ACCESS_AUDIT_SCHEMA = "cure-lite-v24-split-access-audit-v1"
RUNTIME_SPEC_SCHEMA = "cure-lite-v24-dr-runtime-supervisor-spec-v2"
BUGGY_CONSUMER_ACCESS_AUDIT_SCHEMA = (
    "cure-lite-v24-split-access-audit-r2-v1"
)
UNIT_RECOVERY_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-actual-unit-realization-recovery-authorization-v1"
)
UNIT_RECOVERY_RECEIPT_SCHEMA = (
    "cure-lite-v24-actual-unit-realization-recovery-receipt-v1"
)

FORENSIC_TERMINAL_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c1-forensic-terminal-v1"
)
COMPAT_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c1-authorization-v1"
)
COMPAT_RECEIPT_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c1-receipt-v1"
)

_SHA = re.compile(r"[0-9a-f]{64}")
_EXIT_CODE = re.compile(r"(?:^|\n)Process exited with code ([0-9]+)(?:\n|$)")
_BOOT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)

_FILE_ROOT_KEYS = {
    "path",
    "file_sha256",
    "device",
    "inode",
    "size",
    "mtime_ns",
    "ctime_ns",
    "owner_uid",
    "owner_gid",
    "mode",
    "nlink",
    "parent_path",
    "parent_device",
    "parent_inode",
    "parent_size",
    "parent_mtime_ns",
    "parent_ctime_ns",
    "parent_owner_uid",
    "parent_owner_gid",
    "parent_mode",
    "parent_nlink",
}
_SEALED_ROOT_KEYS = _FILE_ROOT_KEYS | {
    "schema_version",
    "fingerprint_field",
    "fingerprint",
}
_ABSENCE_ROOT_KEYS = {
    "path",
    "basename",
    "lexists",
    "parent_path",
    "parent_device",
    "parent_inode",
    "parent_size",
    "parent_mtime_ns",
    "parent_ctime_ns",
    "parent_owner_uid",
    "parent_owner_gid",
    "parent_mode",
    "parent_nlink",
    "observation_scope",
}
_UNIT_STATE_KEYS = {
    "boot_id",
    "unit_name",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "Result",
    "ExecMainCode",
    "ExecMainStatus",
    "NRestarts",
    "InvocationID",
    "StateChangeTimestampMonotonic",
    "InactiveEnterTimestampMonotonic",
    "ActiveEnterTimestampMonotonic",
    "FragmentPath",
}
_SESSION_FAILURE_KEYS = {
    "session_root",
    "call_line_number",
    "output_line_number",
    "call_line_raw_sha256",
    "output_line_raw_sha256",
    "call_record_with_lf_sha256",
    "output_record_with_lf_sha256",
    "line_hash_contract",
    "call_arguments_sha256",
    "output_payload_sha256",
    "call_id",
    "call_timestamp_utc",
    "output_timestamp_utc",
    "tool_name",
    "workdir",
    "command",
    "argv",
    "exit_code",
    "stack_phase",
    "error_type",
    "error_message",
}

_SCIENTIFIC_AUDIT_KEYS = {
    "schema_version",
    "stage_id",
    "allowed_splits",
    "observed_payloads",
    "event_log_fingerprint",
    "source_manifest_fingerprint",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "receipt_fingerprint",
}
_SCIENTIFIC_AUTHORIZATION_KEYS = {
    "schema_version",
    "candidate",
    "stage_id",
    "run_id",
    "status",
    "allowed_purposes",
    "allowed_splits",
    "D_R_payload_authorized",
    "D_V_payload_authorized",
    "D_T_payload_authorized",
    "training_authorized",
    "expires_after_single_materialization",
    "access_audit_receipt_fingerprint",
    "dataset_free_receipt_file_sha256",
    "dataset_free_receipt_fingerprint",
    "efficiency_receipt_sha256",
    "efficiency_section_fingerprint",
    "expected_cache_fingerprint",
    "expected_population_fingerprint",
    "expected_real_inputs_fingerprint",
    "manifest_file_sha256",
    "protocol_preregistration_fingerprint",
    "source_binding_fingerprint",
    "source_closure_fingerprint",
    "state_index_file_sha256",
    "authorization_fingerprint",
}

_ABSENCE_LABELS = {
    "old_runtime_spec",
    "old_runtime_launch_authorization",
    "old_runtime_artifact_root",
    "old_gpu_lease_root",
    "scientific_run_root",
    "scientific_result_receipt",
    "compat_runtime_spec",
    "compat_runtime_launch_authorization",
    "compat_runtime_artifact_root",
    "compat_gpu_lease_root",
    "compat_run_root_alias",
    "compat_result_receipt_alias",
    "compat_unit_realization_terminal",
}
_ALWAYS_PROTECTED_ABSENCE_LABELS = {
    "old_runtime_spec",
    "old_runtime_launch_authorization",
    "old_runtime_artifact_root",
    "old_gpu_lease_root",
    "compat_run_root_alias",
    "compat_result_receipt_alias",
    "compat_unit_realization_terminal",
}
_FROZEN_ROOT_LABELS = {
    "scientific_authorization",
    "scientific_access_audit",
    "old_release",
    "old_supervisor",
    "old_adapter",
    "protocol_verifier",
    "old_unit_realizer",
    "unit_recovery_authorization",
    "unit_recovery_receipt",
    "old_unit_fragment",
}
_COMPAT_SOURCE_LABELS = {
    "compat_policy",
    "compat_release",
    "compat_supervisor",
    "compat_adapter",
    "compat_unit_realizer",
    "compat_unit_template",
}
_COMPAT_EVIDENCE_LABELS = {
    "compat_environment_policy",
    "compat_environment_stability",
    "compat_environment_postcleanup",
    "compat_integration_authorization",
    "compat_integration_receipt",
    "compat_unit_realization_authorization",
    "compat_unit_realization_receipt",
}
_COMPAT_EVIDENCE_FINGERPRINT_FIELDS = {
    "compat_environment_policy": "policy_fingerprint",
    "compat_environment_stability": "stability_receipt_fingerprint",
    "compat_environment_postcleanup": "receipt_fingerprint",
    "compat_integration_authorization": "authorization_fingerprint",
    "compat_integration_receipt": "receipt_fingerprint",
    "compat_unit_realization_authorization": "authorization_fingerprint",
    "compat_unit_realization_receipt": "receipt_fingerprint",
}
_RUNTIME_CONTRACT_KEYS = {
    "unit_name",
    "runtime_spec_path",
    "runtime_launch_authorization_path",
    "runtime_artifact_root",
    "gpu_lease_root",
    "scientific_authorization_path",
    "scientific_access_audit_path",
    "scientific_run_root",
    "scientific_result_receipt_path",
    "compatibility_receipt_path",
}

_MUTATION_AUTHORITY = {
    "forensic_terminal_creation_authorized": True,
    "compatibility_authorization_creation_authorized": True,
    "compatibility_receipt_creation_authorized": True,
    "runtime_spec_creation_authorized": False,
    "runtime_launch_authorization_creation_authorized": False,
    "unit_install_authorized": False,
    "daemon_reload_authorized": False,
    "enable_authorized": False,
    "start_authorized": False,
    "stop_authorized": False,
    "remove_authorized": False,
    "payload_access_authorized_by_this_tool": False,
}
_COMPAT_LANE_AUTHORITY = {
    "compatibility_source_preparation_authorized": True,
    "environment_metadata_audit_authorized": True,
    "dummy_systemd_integration_authorized": True,
    "compat_unit_realization_authorized": True,
    "compat_unit_fragment_install_authorized": True,
    "compat_daemon_reload_authorized": True,
    "compat_enable_authorized": False,
    "compat_start_authorized": False,
    "compat_stop_authorized": False,
    "compat_remove_authorized": False,
    "predecessor_unit_mutation_authorized": False,
    "runtime_spec_creation_authorized_by_this_receipt": False,
    "runtime_launch_authorization_authorized_by_this_receipt": False,
    "payload_access_authorized": False,
}
_SCIENTIFIC_AUTHORITY = {
    "scientific_attempt_ordinal": 2,
    "runtime_compatibility_id": "c1",
    "fresh_scientific_attempt": False,
    "automatic_retry": False,
    "resume": False,
    "allowed_splits": ["D_R"],
    "D_R_payload_authorized": True,
    "D_V_payload_authorized": False,
    "D_T_payload_authorized": False,
    "training_authorized": False,
    "exactly_one_first_materialization": True,
    "expires_after_single_materialization": True,
    "materialization_authority": (
        "deferred_to_separate_fresh_runtime_launch_authorization"
    ),
}
_PAYLOAD_OBSERVATION = {
    "D_R_payload_accessed": False,
    "D_V_payload_accessed": False,
    "D_T_payload_accessed": False,
    "training_started": False,
    "scientific_decision_present": False,
    "attempt_commit_present": False,
    "materialization_claim_present": False,
}
_SCHEMA_COMPATIBILITY = {
    "producer_schema": SCIENTIFIC_ACCESS_AUDIT_SCHEMA,
    "scientific_authorization_bound_schema": (
        SCIENTIFIC_ACCESS_AUDIT_SCHEMA
    ),
    "buggy_frozen_consumer_expected_schema": (
        BUGGY_CONSUMER_ACCESS_AUDIT_SCHEMA
    ),
    "compatibility_consumer_required_schema": (
        SCIENTIFIC_ACCESS_AUDIT_SCHEMA
    ),
    "compatibility_kind": (
        "exact-existing-v1-runtime-consumer-correction"
    ),
    "accept_either_schema": False,
    "audit_rewritten": False,
    "authorization_rewritten": False,
    "alias_created": False,
    "old_fixed_path_reused_for_new_evidence": False,
    "scientific_scope_changed": False,
}

_FORENSIC_TERMINAL_KEYS = {
    "schema_version",
    "candidate",
    "stage_id",
    "scientific_attempt_id",
    "scientific_attempt_ordinal",
    "runtime_compatibility_id",
    "created_at_utc",
    "retrospective",
    "terminal_origin",
    "original_release_generated_terminal",
    "failed_release_passed",
    "failure_phase",
    "runtime_spec_write_attempted",
    "runtime_launch_authorization_write_attempted",
    "runtime_directories_create_attempted",
    "unit_start_attempted",
    "session_failure",
    "schema_compatibility",
    "frozen_generation_roots",
    "absence_generation_roots",
    "old_unit_state",
    "payload_observation",
    "mutation_authority",
    "forensic_closure_passed",
    "terminal_fingerprint",
}
_COMPAT_AUTHORIZATION_KEYS = {
    "schema_version",
    "candidate",
    "stage_id",
    "scientific_attempt_id",
    "scientific_attempt_ordinal",
    "runtime_compatibility_id",
    "instruction_id",
    "authorization_basis",
    "authorized_uid",
    "created_at_utc",
    "issued_at_utc",
    "expires_at_utc",
    "forensic_terminal_root",
    "schema_compatibility",
    "scientific_authority",
    "frozen_generation_roots",
    "absence_generation_roots",
    "old_unit_state",
    "payload_observation",
    "mutation_authority",
    "compat_lane_authority",
    "authorization_fingerprint",
}
_COMPAT_RECEIPT_KEYS = {
    "schema_version",
    "candidate",
    "stage_id",
    "scientific_attempt_id",
    "scientific_attempt_ordinal",
    "runtime_compatibility_id",
    "created_at_utc",
    "forensic_terminal_root",
    "compatibility_authorization_root",
    "instruction_id",
    "authorization_basis",
    "schema_compatibility",
    "scientific_authority",
    "frozen_generation_roots",
    "compatibility_source_roots",
    "compatibility_evidence_roots",
    "compatibility_runtime_contract",
    "absence_generation_roots",
    "scientific_output_contract",
    "old_unit_state",
    "payload_observation",
    "mutation_authority",
    "compat_lane_authority",
    "runtime_launch_authorized",
    "systemd_mutation_authorized",
    "compatibility_closure_passed",
    "receipt_fingerprint",
}

UnitStateReader = Callable[[], Mapping[str, object]]
ObservationHook = Callable[[], None]
Runner = Callable[..., subprocess.CompletedProcess[str]]


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _deep_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            set(left) == set(right)
            and all(
                _deep_exact_equal(left[key], right[key])
                for key in left
            )
        )
    if isinstance(left, list):
        return (
            len(left) == len(right)
            and all(
                _deep_exact_equal(a, b)
                for a, b in zip(left, right, strict=True)
            )
        )
    return left == right


def _utc_now_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise PermissionError(f"{name} is not an exact UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PermissionError(f"{name} is not an exact UTC timestamp") from error
    if parsed.tzinfo is None:
        raise PermissionError(f"{name} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _require_fixed_path(
    supplied: Path,
    expected: Path,
    *,
    name: str,
) -> Path:
    actual = Path(supplied).absolute()
    fixed = Path(expected).absolute()
    if actual != fixed:
        raise PermissionError(f"{name} differs from its fixed path")
    return actual


def _stat_fields(metadata: os.stat_result) -> dict[str, int]:
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "owner_uid": metadata.st_uid,
        "owner_gid": metadata.st_gid,
        "mode": stat.S_IMODE(metadata.st_mode),
        "nlink": metadata.st_nlink,
    }


def _parent_fields(path: Path, metadata: os.stat_result) -> dict[str, object]:
    return {
        "parent_path": str(path),
        "parent_device": metadata.st_dev,
        "parent_inode": metadata.st_ino,
        "parent_size": metadata.st_size,
        "parent_mtime_ns": metadata.st_mtime_ns,
        "parent_ctime_ns": metadata.st_ctime_ns,
        "parent_owner_uid": metadata.st_uid,
        "parent_owner_gid": metadata.st_gid,
        "parent_mode": stat.S_IMODE(metadata.st_mode),
        "parent_nlink": metadata.st_nlink,
    }


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _read_regular_root(path: Path) -> tuple[bytes, dict[str, object]]:
    target = Path(path).absolute()
    if target.name in {"", ".", ".."}:
        raise ValueError("bound file path has no basename")
    if target.parent.resolve(strict=True) != target.parent:
        raise PermissionError("bound file parent is not canonical")
    parent_before = target.parent.lstat()
    if not stat.S_ISDIR(parent_before.st_mode):
        raise PermissionError("bound file parent is not a directory")
    parent_fd = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        parent_opened = os.fstat(parent_fd)
        if _stat_fields(parent_opened) != _stat_fields(parent_before):
            raise PermissionError("bound file parent changed before open")
        descriptor = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError("bound path is not a regular file")
        raw = _read_all(descriptor)
        after = os.fstat(descriptor)
        linked = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        parent_after = os.fstat(parent_fd)
        parent_linked = target.parent.lstat()
        if (
            _stat_fields(after) != _stat_fields(before)
            or _stat_fields(linked) != _stat_fields(after)
            or _stat_fields(parent_after) != _stat_fields(parent_opened)
            or _stat_fields(parent_linked) != _stat_fields(parent_opened)
            or len(raw) != after.st_size
        ):
            raise PermissionError("bound file generation changed while read")
        root: dict[str, object] = {
            "path": str(target),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            **_stat_fields(after),
            **_parent_fields(target.parent, parent_after),
        }
        if set(root) != _FILE_ROOT_KEYS:
            raise AssertionError("internal file-root schema is incomplete")
        if (
            root["owner_uid"] != os.getuid()
            or root["nlink"] != 1
        ):
            raise PermissionError("bound file owner/link identity is unsafe")
        return raw, root
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _decode_canonical_json(raw: bytes, *, name: str) -> dict[str, object]:
    if not raw.endswith(b"\n"):
        raise ValueError(f"{name} lacks its canonical newline")
    try:
        payload = json.loads(raw[:-1].decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict canonical JSON") from error
    if (
        not isinstance(payload, dict)
        or raw != (canonical_json(payload) + "\n").encode("utf-8")
    ):
        raise ValueError(f"{name} is not one canonical JSON object")
    return payload


def _sealed_root(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str | None,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    raw, file_root = _read_regular_root(path)
    if file_root["mode"] != 0o444:
        raise PermissionError("sealed predecessor mode is not 0444")
    payload = _decode_canonical_json(raw, name=str(path))
    body = dict(payload)
    fingerprint = body.pop(fingerprint_field, None)
    if (
        not isinstance(fingerprint, str)
        or _SHA.fullmatch(fingerprint) is None
        or fingerprint != stable_fingerprint(body)
        or (
            schema is not None
            and payload.get("schema_version") != schema
        )
        or (
            expected_sha256 is not None
            and file_root["file_sha256"] != expected_sha256
        )
    ):
        raise PermissionError("sealed predecessor identity changed")
    root = {
        **file_root,
        "schema_version": payload.get("schema_version"),
        "fingerprint_field": fingerprint_field,
        "fingerprint": fingerprint,
    }
    if set(root) != _SEALED_ROOT_KEYS:
        raise AssertionError("internal sealed-root schema is incomplete")
    return payload, root


def _regular_root(
    path: Path,
    *,
    expected_sha256: str | None = None,
    required_mode: int | None = None,
) -> dict[str, object]:
    _raw, root = _read_regular_root(path)
    if (
        (
            expected_sha256 is not None
            and root["file_sha256"] != expected_sha256
        )
        or (
            required_mode is not None
            and root["mode"] != required_mode
        )
    ):
        raise PermissionError("regular predecessor identity changed")
    return root


def _validate_file_root(
    value: object,
    *,
    expected_path: Path,
    expected_sha256: str | None = None,
    required_mode: int | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FILE_ROOT_KEYS:
        raise PermissionError("file generation-root schema changed")
    root = dict(value)
    if root.get("path") != str(Path(expected_path).absolute()):
        raise PermissionError("file generation-root path changed")
    observed = _regular_root(
        expected_path,
        expected_sha256=expected_sha256,
        required_mode=required_mode,
    )
    if not _same_generation_root(observed, root):
        raise PermissionError("file generation was replaced")
    return observed


def _validate_sealed_root(
    value: object,
    *,
    expected_path: Path,
    fingerprint_field: str,
    schema: str | None,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != _SEALED_ROOT_KEYS:
        raise PermissionError("sealed generation-root schema changed")
    root = dict(value)
    if root.get("path") != str(Path(expected_path).absolute()):
        raise PermissionError("sealed generation-root path changed")
    payload, observed = _sealed_root(
        expected_path,
        fingerprint_field=fingerprint_field,
        schema=schema,
        expected_sha256=expected_sha256,
    )
    if not _same_generation_root(observed, root):
        raise PermissionError("sealed generation was replaced")
    return payload, observed


def _same_generation_root(
    left: object,
    right: object,
) -> bool:
    """Compare one file generation while allowing append-only siblings.

    Every stored root retains the complete parent generation at observation.
    The forensic terminal, authorization, and receipt intentionally share a
    parent with several predecessors; creating a later sibling changes the
    directory size/timestamps but not the predecessor or parent directory
    entry generation.  File fields and stable parent identity fields remain
    exact, so same-byte inode substitution is still rejected.
    """

    if (
        not isinstance(left, Mapping)
        or not isinstance(right, Mapping)
        or set(left) != set(right)
    ):
        return False
    volatile_parent = {
        "parent_size",
        "parent_mtime_ns",
        "parent_ctime_ns",
    }
    return all(
        _deep_exact_equal(left[key], right[key])
        for key in left
        if key not in volatile_parent
    )


def _observe_absence(path: Path) -> dict[str, object]:
    target = Path(path).absolute()
    if target.name in {"", ".", ".."}:
        raise ValueError("absence path has no basename")
    if target.parent.resolve(strict=True) != target.parent:
        raise PermissionError("absence parent is not canonical")
    parent_before = target.parent.lstat()
    if not stat.S_ISDIR(parent_before.st_mode):
        raise PermissionError("absence parent is not a directory")
    parent_fd = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(parent_fd)
        if _stat_fields(opened) != _stat_fields(parent_before):
            raise PermissionError("absence parent changed before observation")
        try:
            os.stat(
                target.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise PermissionError(f"required absent path exists: {target}")
        linked = target.parent.lstat()
        finished = os.fstat(parent_fd)
        if (
            _stat_fields(linked) != _stat_fields(opened)
            or _stat_fields(finished) != _stat_fields(opened)
        ):
            raise PermissionError("absence parent changed while observed")
        result: dict[str, object] = {
            "path": str(target),
            "basename": target.name,
            "lexists": False,
            **_parent_fields(target.parent, finished),
            "observation_scope": (
                "full-parent-generation-at-observation;"
                "later-append-only-sibling-evidence-may-change-parent-times"
            ),
        }
        if set(result) != _ABSENCE_ROOT_KEYS:
            raise AssertionError("internal absence-root schema is incomplete")
        return result
    finally:
        os.close(parent_fd)


def _validate_absence_root(
    value: object,
    *,
    expected_path: Path,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ABSENCE_ROOT_KEYS:
        raise PermissionError("absence generation-root schema changed")
    root = dict(value)
    target = Path(expected_path).absolute()
    if (
        root.get("path") != str(target)
        or root.get("basename") != target.name
        or root.get("lexists") is not False
        or root.get("parent_path") != str(target.parent)
    ):
        raise PermissionError("absence generation-root path changed")
    observed = _observe_absence(target)
    # The three closure artifacts live beside several future-absent leaves.
    # Their create-once writes legitimately change parent times and directory
    # size.  The directory-entry generation is therefore preserved by exact
    # path/device/inode/owner/mode/nlink plus a fresh full observation, rather
    # than pretending the historical parent timestamps can remain fixed.
    stable_parent_fields = {
        "parent_path",
        "parent_device",
        "parent_inode",
        "parent_owner_uid",
        "parent_owner_gid",
        "parent_mode",
        "parent_nlink",
    }
    if any(root[field] != observed[field] for field in stable_parent_fields):
        raise PermissionError("absence parent generation was replaced")
    return observed


def _open_create_parent(target: Path) -> tuple[int, os.stat_result]:
    if target.parent.resolve(strict=True) != target.parent:
        raise PermissionError("create-once parent is not canonical")
    before = target.parent.lstat()
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.getuid()
    ):
        raise PermissionError("create-once parent is unsafe")
    descriptor = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino)
        != (before.st_dev, before.st_ino)
    ):
        os.close(descriptor)
        raise PermissionError("create-once parent changed before open")
    return descriptor, opened


def _write_sealed(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    target = Path(path).absolute()
    payload = dict(body)
    if fingerprint_field in payload:
        raise ValueError("fingerprint field already exists")
    payload[fingerprint_field] = stable_fingerprint(payload)
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    parent_fd, parent_before = _open_create_parent(target)
    descriptor = -1
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(
            target.name,
            flags,
            0o444,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("short create-once evidence write")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        linked = os.stat(
            target.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_fields(linked) != _stat_fields(opened)
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != 0o444
            or opened.st_nlink != 1
        ):
            raise RuntimeError("created evidence inode binding failed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        readback = _read_all(descriptor)
        finished = os.fstat(descriptor)
        parent_finished = os.fstat(parent_fd)
        parent_linked = target.parent.lstat()
        if (
            readback != encoded
            or _stat_fields(finished) != _stat_fields(opened)
            or (parent_finished.st_dev, parent_finished.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or (parent_linked.st_dev, parent_linked.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
        ):
            raise RuntimeError("created evidence failed descriptor readback")
        os.fsync(parent_fd)
    except BaseException:
        # A partial create-once artifact remains forensic evidence and is never
        # unlinked or retried automatically.
        try:
            os.fsync(parent_fd)
        finally:
            raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    return payload


def _load_sealed_output(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str,
    expected_keys: set[str],
) -> tuple[dict[str, object], dict[str, object]]:
    payload, root = _sealed_root(
        path,
        fingerprint_field=fingerprint_field,
        schema=schema,
    )
    if set(payload) != expected_keys:
        raise PermissionError("compatibility evidence keys changed")
    return payload, root


def _absence_paths() -> dict[str, Path]:
    return {
        "old_runtime_spec": OLD_RUNTIME_SPEC_PATH,
        "old_runtime_launch_authorization": (
            OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "old_runtime_artifact_root": OLD_RUNTIME_ARTIFACT_ROOT,
        "old_gpu_lease_root": OLD_GPU_LEASE_ROOT,
        "scientific_run_root": SCIENTIFIC_RUN_ROOT,
        "scientific_result_receipt": SCIENTIFIC_RESULT_RECEIPT_PATH,
        "compat_runtime_spec": COMPAT_RUNTIME_SPEC_PATH,
        "compat_runtime_launch_authorization": (
            COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "compat_runtime_artifact_root": COMPAT_RUNTIME_ARTIFACT_ROOT,
        "compat_gpu_lease_root": COMPAT_GPU_LEASE_ROOT,
        "compat_run_root_alias": COMPAT_RUN_ROOT_ALIAS_PATH,
        "compat_result_receipt_alias": (
            COMPAT_RESULT_RECEIPT_ALIAS_PATH
        ),
        "compat_unit_realization_terminal": (
            COMPAT_UNIT_REALIZATION_TERMINAL_PATH
        ),
    }


def _collect_absences() -> dict[str, dict[str, object]]:
    result = {
        label: _observe_absence(path)
        for label, path in _absence_paths().items()
    }
    if set(result) != _ABSENCE_LABELS:
        raise AssertionError("internal absence closure is incomplete")
    return result


def _validate_absences(
    value: object,
    *,
    labels: set[str] = _ABSENCE_LABELS,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _ABSENCE_LABELS:
        raise PermissionError("absence closure labels changed")
    if not labels.issubset(_ABSENCE_LABELS):
        raise AssertionError("unknown absence validation scope")
    paths = _absence_paths()
    for label in labels:
        _validate_absence_root(value[label], expected_path=paths[label])


def _validate_scientific_preaccess(
    authorization: Mapping[str, object],
    audit: Mapping[str, object],
) -> None:
    if (
        set(authorization) != _SCIENTIFIC_AUTHORIZATION_KEYS
        or set(audit) != _SCIENTIFIC_AUDIT_KEYS
        or authorization.get("candidate") != CANDIDATE
        or authorization.get("stage_id") != STAGE_ID
        or authorization.get("run_id") != SCIENTIFIC_ATTEMPT_ID
        or authorization.get("allowed_splits") != ["D_R"]
        or authorization.get("D_R_payload_authorized") is not True
        or authorization.get("D_V_payload_authorized") is not False
        or authorization.get("D_T_payload_authorized") is not False
        or authorization.get("training_authorized") is not False
        or authorization.get("expires_after_single_materialization")
        is not True
        or authorization.get("source_closure_fingerprint")
        != SOURCE_CLOSURE_FINGERPRINT_103
        or audit.get("stage_id") != STAGE_ID
        or audit.get("allowed_splits") != ["D_R"]
        or audit.get("observed_payloads") != []
        or audit.get("D_V_payload_accessed") is not False
        or audit.get("D_T_payload_accessed") is not False
        or authorization.get("access_audit_receipt_fingerprint")
        != audit.get("receipt_fingerprint")
    ):
        raise PermissionError("scientific r2 preaccess identity changed")


def _frozen_root_specs() -> dict[
    str,
    tuple[Path, str | None, str | None],
]:
    # (path, fingerprint_field, schema); None fingerprint means regular file.
    return {
        "scientific_authorization": (
            SCIENTIFIC_AUTHORIZATION_PATH,
            "authorization_fingerprint",
            SCIENTIFIC_AUTHORIZATION_SCHEMA,
        ),
        "scientific_access_audit": (
            SCIENTIFIC_ACCESS_AUDIT_PATH,
            "receipt_fingerprint",
            SCIENTIFIC_ACCESS_AUDIT_SCHEMA,
        ),
        "old_release": (OLD_RELEASE_PATH, None, None),
        "old_supervisor": (OLD_SUPERVISOR_PATH, None, None),
        "old_adapter": (OLD_ADAPTER_PATH, None, None),
        "protocol_verifier": (PROTOCOL_VERIFIER_PATH, None, None),
        "old_unit_realizer": (OLD_UNIT_REALIZER_PATH, None, None),
        "unit_recovery_authorization": (
            UNIT_RECOVERY_AUTHORIZATION_PATH,
            "authorization_fingerprint",
            UNIT_RECOVERY_AUTHORIZATION_SCHEMA,
        ),
        "unit_recovery_receipt": (
            UNIT_RECOVERY_RECEIPT_PATH,
            "receipt_fingerprint",
            UNIT_RECOVERY_RECEIPT_SCHEMA,
        ),
        "old_unit_fragment": (OLD_UNIT_FRAGMENT_PATH, None, None),
    }


def _collect_frozen_roots() -> dict[str, dict[str, object]]:
    roots: dict[str, dict[str, object]] = {}
    scientific: dict[str, dict[str, object]] = {}
    for label, (path, fingerprint_field, schema) in (
        _frozen_root_specs().items()
    ):
        expected_sha = EXPECTED_FILE_SHA256[label]
        if fingerprint_field is None:
            roots[label] = _regular_root(
                path,
                expected_sha256=expected_sha,
                required_mode=0o600
                if label == "old_unit_fragment"
                else None,
            )
        else:
            payload, roots[label] = _sealed_root(
                path,
                fingerprint_field=fingerprint_field,
                schema=schema,
                expected_sha256=expected_sha,
            )
            scientific[label] = payload
    if set(roots) != _FROZEN_ROOT_LABELS:
        raise AssertionError("internal frozen closure is incomplete")
    _validate_scientific_preaccess(
        scientific["scientific_authorization"],
        scientific["scientific_access_audit"],
    )
    return roots


def _validate_frozen_roots(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _FROZEN_ROOT_LABELS:
        raise PermissionError("frozen generation-root labels changed")
    scientific: dict[str, dict[str, object]] = {}
    for label, (path, fingerprint_field, schema) in (
        _frozen_root_specs().items()
    ):
        expected_sha = EXPECTED_FILE_SHA256[label]
        if fingerprint_field is None:
            _validate_file_root(
                value[label],
                expected_path=path,
                expected_sha256=expected_sha,
                required_mode=0o600
                if label == "old_unit_fragment"
                else None,
            )
        else:
            payload, _root = _validate_sealed_root(
                value[label],
                expected_path=path,
                fingerprint_field=fingerprint_field,
                schema=schema,
                expected_sha256=expected_sha,
            )
            scientific[label] = payload
    _validate_scientific_preaccess(
        scientific["scientific_authorization"],
        scientific["scientific_access_audit"],
    )


def _fixed_systemd_environment() -> dict[str, str]:
    uid = os.getuid()
    runtime = f"/run/user/{uid}"
    return {
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime}/bus",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "SYSTEMD_COLORS": "0",
        "XDG_RUNTIME_DIR": runtime,
    }


def collect_old_unit_state(
    *,
    runner: Runner = subprocess.run,
) -> dict[str, object]:
    properties = [
        "LoadState",
        "ActiveState",
        "SubState",
        "UnitFileState",
        "Result",
        "ExecMainCode",
        "ExecMainStatus",
        "NRestarts",
        "InvocationID",
        "StateChangeTimestampMonotonic",
        "InactiveEnterTimestampMonotonic",
        "ActiveEnterTimestampMonotonic",
        "FragmentPath",
    ]
    argv: list[str] = [
        SYSTEMCTL,
        "--user",
        "show",
        OLD_UNIT_NAME,
        "--no-pager",
    ]
    for name in properties:
        argv.extend(("-p", name))
    completed = runner(
        argv,
        shell=False,
        check=False,
        text=True,
        capture_output=True,
        timeout=30.0,
        env=_fixed_systemd_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError("read-only systemctl show failed")
    parsed: dict[str, str] = {}
    for row in completed.stdout.splitlines():
        if "=" not in row:
            raise ValueError("systemctl show output is malformed")
        key, value = row.split("=", 1)
        if key in parsed:
            raise ValueError("systemctl show output has duplicate keys")
        parsed[key] = value
    if set(parsed) != set(properties):
        raise PermissionError("systemctl show property set changed")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    result: dict[str, object] = {
        "boot_id": boot_id,
        "unit_name": OLD_UNIT_NAME,
        **parsed,
    }
    return _normalize_unit_state(result)


def _normalize_unit_state(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != _UNIT_STATE_KEYS:
        raise PermissionError("old unit-state keys changed")
    result = dict(value)
    integer_fields = (
        "ExecMainCode",
        "ExecMainStatus",
        "NRestarts",
        "StateChangeTimestampMonotonic",
        "InactiveEnterTimestampMonotonic",
        "ActiveEnterTimestampMonotonic",
    )
    for field in integer_fields:
        raw = result[field]
        if isinstance(raw, str):
            if not raw.isdigit():
                raise PermissionError("old unit numeric state is malformed")
            result[field] = int(raw)
        elif isinstance(raw, bool) or not isinstance(raw, int):
            raise PermissionError("old unit numeric state is malformed")
    if (
        not isinstance(result["boot_id"], str)
        or _BOOT_ID.fullmatch(result["boot_id"]) is None
        or result["unit_name"] != OLD_UNIT_NAME
        or result["LoadState"] != "loaded"
        or result["ActiveState"] != "inactive"
        or result["SubState"] != "dead"
        or result["UnitFileState"] != "static"
        or result["Result"] != "success"
        or result["ExecMainCode"] != 0
        or result["ExecMainStatus"] != 0
        or result["NRestarts"] != 0
        or result["InvocationID"] != ""
        or result["StateChangeTimestampMonotonic"] != 0
        or result["InactiveEnterTimestampMonotonic"] != 0
        or result["ActiveEnterTimestampMonotonic"] != 0
        or result["FragmentPath"] != str(OLD_UNIT_FRAGMENT_PATH)
    ):
        raise PermissionError("old unit is not an untouched static unit")
    return result


def _read_unit_state(reader: UnitStateReader) -> dict[str, object]:
    return _normalize_unit_state(reader())


def _observe_session_failure() -> dict[str, object]:
    raw, session_root = _read_regular_root(SESSION_PATH)
    lines = raw.splitlines(keepends=True)
    if len(lines) < SESSION_OUTPUT_LINE:
        raise PermissionError("forensic session is shorter than fixed lines")
    call_record_raw = lines[SESSION_CALL_LINE - 1]
    output_record_raw = lines[SESSION_OUTPUT_LINE - 1]
    if (
        not call_record_raw.endswith(b"\n")
        or call_record_raw.endswith(b"\r\n")
        or not output_record_raw.endswith(b"\n")
        or output_record_raw.endswith(b"\r\n")
    ):
        raise PermissionError("fixed forensic records lack exact LF framing")
    call_record_with_lf_sha = hashlib.sha256(call_record_raw).hexdigest()
    output_record_with_lf_sha = hashlib.sha256(
        output_record_raw
    ).hexdigest()
    call_raw = call_record_raw[:-1]
    output_raw = output_record_raw[:-1]
    call_raw_sha = hashlib.sha256(call_raw).hexdigest()
    output_raw_sha = hashlib.sha256(output_raw).hexdigest()
    if (
        call_raw_sha != SESSION_CALL_RAW_SHA256
        or output_raw_sha != SESSION_OUTPUT_RAW_SHA256
        or call_record_with_lf_sha
        != SESSION_CALL_RECORD_WITH_LF_SHA256
        or output_record_with_lf_sha
        != SESSION_OUTPUT_RECORD_WITH_LF_SHA256
    ):
        raise PermissionError("fixed forensic session line changed")
    try:
        call_record = json.loads(call_raw.decode("utf-8", errors="strict"))
        output_record = json.loads(
            output_raw.decode("utf-8", errors="strict")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fixed forensic session lines are invalid") from error
    call_payload = call_record.get("payload")
    output_payload = output_record.get("payload")
    if not isinstance(call_payload, Mapping) or not isinstance(
        output_payload, Mapping
    ):
        raise PermissionError("fixed forensic payload is absent")
    arguments_raw = call_payload.get("arguments")
    output_text = output_payload.get("output")
    if not isinstance(arguments_raw, str) or not isinstance(output_text, str):
        raise PermissionError("fixed forensic command/output is absent")
    if (
        hashlib.sha256(arguments_raw.encode("utf-8")).hexdigest()
        != SESSION_CALL_ARGUMENTS_SHA256
        or hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        != SESSION_OUTPUT_PAYLOAD_SHA256
    ):
        raise PermissionError("fixed forensic payload changed")
    try:
        arguments = json.loads(arguments_raw)
    except json.JSONDecodeError as error:
        raise ValueError("fixed exec arguments are invalid") from error
    if not isinstance(arguments, dict):
        raise PermissionError("fixed exec arguments are not an object")
    command = arguments.get("cmd")
    workdir = arguments.get("workdir")
    if not isinstance(command, str) or not isinstance(workdir, str):
        raise PermissionError("fixed exec command/workdir is absent")
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as error:
        raise PermissionError("fixed exec command cannot be materialized") from error
    if (
        call_record.get("type") != "response_item"
        or output_record.get("type") != "response_item"
        or call_payload.get("type") != "function_call"
        or output_payload.get("type") != "function_call_output"
        or call_payload.get("name") != "exec_command"
        or call_payload.get("call_id") != SESSION_CALL_ID
        or output_payload.get("call_id") != SESSION_CALL_ID
        or workdir != str(REPOSITORY)
        or len(argv) < 7
        or argv[:5] != [
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            "-u",
        ]
        or argv[5] != str(OLD_RELEASE_PATH)
        or argv[6] != "build-spec"
    ):
        raise PermissionError("fixed forensic invocation identity changed")
    exit_match = _EXIT_CODE.search(output_text)
    if exit_match is None or int(exit_match.group(1)) != 1:
        raise PermissionError("fixed forensic exit code is not 1")
    required_stack = (
        "build_spec(",
        "validate_release_closure(",
        "_validate_scientific_preaccess()",
        "_load_sealed(",
        "PermissionError: release evidence fingerprint/schema changed:",
        str(SCIENTIFIC_ACCESS_AUDIT_PATH),
    )
    if any(fragment not in output_text for fragment in required_stack):
        raise PermissionError("fixed forensic prewrite stack changed")
    result: dict[str, object] = {
        "session_root": session_root,
        "call_line_number": SESSION_CALL_LINE,
        "output_line_number": SESSION_OUTPUT_LINE,
        "call_line_raw_sha256": call_raw_sha,
        "output_line_raw_sha256": output_raw_sha,
        "call_record_with_lf_sha256": call_record_with_lf_sha,
        "output_record_with_lf_sha256": output_record_with_lf_sha,
        "line_hash_contract": {
            "record_hash_includes_trailing_lf": True,
            "raw_json_text_hash_excludes_trailing_lf": True,
            "line_ending": "LF",
        },
        "call_arguments_sha256": SESSION_CALL_ARGUMENTS_SHA256,
        "output_payload_sha256": SESSION_OUTPUT_PAYLOAD_SHA256,
        "call_id": SESSION_CALL_ID,
        "call_timestamp_utc": call_record.get("timestamp"),
        "output_timestamp_utc": output_record.get("timestamp"),
        "tool_name": "exec_command",
        "workdir": workdir,
        "command": command,
        "argv": argv,
        "exit_code": 1,
        "stack_phase": (
            "build_spec.validate_release_closure."
            "_validate_scientific_preaccess._load_sealed"
        ),
        "error_type": "PermissionError",
        "error_message": (
            "release evidence fingerprint/schema changed: "
            f"{SCIENTIFIC_ACCESS_AUDIT_PATH}"
        ),
    }
    if set(result) != _SESSION_FAILURE_KEYS:
        raise AssertionError("internal session failure schema is incomplete")
    _parse_utc(result["call_timestamp_utc"], name="failure call timestamp")
    _parse_utc(
        result["output_timestamp_utc"],
        name="failure output timestamp",
    )
    return result


def _validate_session_failure(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _SESSION_FAILURE_KEYS:
        raise PermissionError("forensic session-binding keys changed")
    observed = _observe_session_failure()
    stored = dict(value)
    stored_root = stored.get("session_root")
    observed_root = observed.get("session_root")
    if (
        not isinstance(stored_root, Mapping)
        or not isinstance(observed_root, Mapping)
    ):
        raise PermissionError("forensic session root is absent")
    stable_root_fields = {
        "path",
        "device",
        "inode",
        "owner_uid",
        "owner_gid",
        "mode",
        "nlink",
        "parent_path",
        "parent_device",
        "parent_inode",
        "parent_owner_uid",
        "parent_owner_gid",
        "parent_mode",
        "parent_nlink",
    }
    stored_without_root = dict(stored)
    observed_without_root = dict(observed)
    stored_without_root.pop("session_root")
    observed_without_root.pop("session_root")
    if (
        not _deep_exact_equal(
            stored_without_root,
            observed_without_root,
        )
        or any(
            stored_root.get(field) != observed_root.get(field)
            for field in stable_root_fields
        )
        or not isinstance(stored_root.get("size"), int)
        or not isinstance(observed_root.get("size"), int)
        or observed_root["size"] < stored_root["size"]
        or not isinstance(stored_root.get("mtime_ns"), int)
        or not isinstance(observed_root.get("mtime_ns"), int)
        or observed_root["mtime_ns"] < stored_root["mtime_ns"]
        or not isinstance(stored_root.get("ctime_ns"), int)
        or not isinstance(observed_root.get("ctime_ns"), int)
        or observed_root["ctime_ns"] < stored_root["ctime_ns"]
    ):
        raise PermissionError("forensic session generation was replaced")


def _collect_live_snapshot(
    unit_state_reader: UnitStateReader,
) -> dict[str, object]:
    return {
        "session_failure": _observe_session_failure(),
        "frozen_generation_roots": _collect_frozen_roots(),
        "absence_generation_roots": _collect_absences(),
        "old_unit_state": _read_unit_state(unit_state_reader),
    }


def _validate_live_snapshot(
    snapshot: Mapping[str, object],
    *,
    unit_state_reader: UnitStateReader,
    require_future_absence: bool = True,
) -> None:
    if set(snapshot) != {
        "session_failure",
        "frozen_generation_roots",
        "absence_generation_roots",
        "old_unit_state",
    }:
        raise PermissionError("live prewrite snapshot keys changed")
    _validate_session_failure(snapshot["session_failure"])
    _validate_frozen_roots(snapshot["frozen_generation_roots"])
    _validate_absences(
        snapshot["absence_generation_roots"],
        labels=(
            _ABSENCE_LABELS
            if require_future_absence
            else _ALWAYS_PROTECTED_ABSENCE_LABELS
        ),
    )
    observed_state = _read_unit_state(unit_state_reader)
    if not _deep_exact_equal(observed_state, snapshot["old_unit_state"]):
        raise PermissionError("old unit state changed")


def _double_observe_snapshot(
    unit_state_reader: UnitStateReader,
    *,
    between_observations: ObservationHook | None,
) -> dict[str, object]:
    first = _collect_live_snapshot(unit_state_reader)
    if between_observations is not None:
        between_observations()
    second = _collect_live_snapshot(unit_state_reader)
    if not _deep_exact_equal(first, second):
        raise PermissionError("prewrite closure changed between observations")
    return second


def _require_closure_outputs_absent(
    *,
    terminal_may_exist: bool,
    authorization_may_exist: bool,
) -> None:
    paths = (
        (FORENSIC_TERMINAL_PATH, terminal_may_exist, "forensic terminal"),
        (
            COMPAT_AUTHORIZATION_PATH,
            authorization_may_exist,
            "compatibility authorization",
        ),
        (COMPAT_RECEIPT_PATH, False, "compatibility receipt"),
    )
    for path, may_exist, name in paths:
        exists = os.path.lexists(path)
        if exists != may_exist:
            expectation = "present" if may_exist else "absent"
            raise PermissionError(f"{name} must be {expectation}")


def _validate_common_identity(payload: Mapping[str, object]) -> None:
    if (
        payload.get("candidate") != CANDIDATE
        or payload.get("stage_id") != STAGE_ID
        or payload.get("scientific_attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or payload.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or payload.get("runtime_compatibility_id")
        != RUNTIME_COMPATIBILITY_ID
    ):
        raise PermissionError("compatibility identity changed")


def _validate_common_contracts(payload: Mapping[str, object]) -> None:
    if (
        not _deep_exact_equal(
            payload.get("schema_compatibility"),
            _SCHEMA_COMPATIBILITY,
        )
        or not _deep_exact_equal(
            payload.get("payload_observation"),
            _PAYLOAD_OBSERVATION,
        )
        or not _deep_exact_equal(
            payload.get("mutation_authority"),
            _MUTATION_AUTHORITY,
        )
    ):
        raise PermissionError("compatibility safety contract changed")


def create_forensic_terminal(
    *,
    terminal_path: Path | None = None,
    unit_state_reader: UnitStateReader = collect_old_unit_state,
    between_observations: ObservationHook | None = None,
    now: Callable[[], datetime] = _utc_now_datetime,
) -> dict[str, object]:
    if terminal_path is None:
        terminal_path = FORENSIC_TERMINAL_PATH
    _require_fixed_path(
        terminal_path,
        FORENSIC_TERMINAL_PATH,
        name="forensic terminal",
    )
    _require_closure_outputs_absent(
        terminal_may_exist=False,
        authorization_may_exist=False,
    )
    snapshot = _double_observe_snapshot(
        unit_state_reader,
        between_observations=between_observations,
    )
    body: dict[str, object] = {
        "schema_version": FORENSIC_TERMINAL_SCHEMA,
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
        "created_at_utc": _format_utc(now()),
        "retrospective": True,
        "terminal_origin": (
            "append-only-forensic-closure;"
            "not-the-frozen-release-process"
        ),
        "original_release_generated_terminal": False,
        "failed_release_passed": False,
        "failure_phase": (
            "prewrite_scientific_preaccess_schema_validation"
        ),
        "runtime_spec_write_attempted": False,
        "runtime_launch_authorization_write_attempted": False,
        "runtime_directories_create_attempted": False,
        "unit_start_attempted": False,
        "session_failure": snapshot["session_failure"],
        "schema_compatibility": dict(_SCHEMA_COMPATIBILITY),
        "frozen_generation_roots": (
            snapshot["frozen_generation_roots"]
        ),
        "absence_generation_roots": (
            snapshot["absence_generation_roots"]
        ),
        "old_unit_state": snapshot["old_unit_state"],
        "payload_observation": dict(_PAYLOAD_OBSERVATION),
        "mutation_authority": dict(_MUTATION_AUTHORITY),
        "forensic_closure_passed": True,
    }
    if set(body) | {"terminal_fingerprint"} != _FORENSIC_TERMINAL_KEYS:
        raise AssertionError("internal forensic terminal keys are incomplete")
    # The second observation above is the commit observation.  Revalidate all
    # roots once more without accepting an intervening generation.
    _validate_live_snapshot(snapshot, unit_state_reader=unit_state_reader)
    return _write_sealed(
        FORENSIC_TERMINAL_PATH,
        body,
        fingerprint_field="terminal_fingerprint",
    )


def validate_forensic_terminal(
    path: Path | None = None,
    *,
    unit_state_reader: UnitStateReader = collect_old_unit_state,
    require_future_absence: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    if path is None:
        path = FORENSIC_TERMINAL_PATH
    fixed = _require_fixed_path(
        path,
        FORENSIC_TERMINAL_PATH,
        name="forensic terminal",
    )
    terminal, root = _load_sealed_output(
        fixed,
        fingerprint_field="terminal_fingerprint",
        schema=FORENSIC_TERMINAL_SCHEMA,
        expected_keys=_FORENSIC_TERMINAL_KEYS,
    )
    _validate_common_identity(terminal)
    _validate_common_contracts(terminal)
    if (
        terminal.get("retrospective") is not True
        or terminal.get("original_release_generated_terminal") is not False
        or terminal.get("failed_release_passed") is not False
        or terminal.get("failure_phase")
        != "prewrite_scientific_preaccess_schema_validation"
        or any(
            terminal.get(field) is not False
            for field in (
                "runtime_spec_write_attempted",
                "runtime_launch_authorization_write_attempted",
                "runtime_directories_create_attempted",
                "unit_start_attempted",
            )
        )
        or terminal.get("forensic_closure_passed") is not True
    ):
        raise PermissionError("retrospective terminal semantics changed")
    _validate_live_snapshot(
        {
            "session_failure": terminal["session_failure"],
            "frozen_generation_roots": (
                terminal["frozen_generation_roots"]
            ),
            "absence_generation_roots": (
                terminal["absence_generation_roots"]
            ),
            "old_unit_state": terminal["old_unit_state"],
        },
        unit_state_reader=unit_state_reader,
        require_future_absence=require_future_absence,
    )
    return terminal, root


def _validate_authorization_freshness(
    authorization: Mapping[str, object],
    *,
    current: datetime,
    require_fresh: bool,
) -> tuple[datetime, datetime, datetime]:
    created = _parse_utc(
        authorization.get("created_at_utc"),
        name="compatibility authorization created_at_utc",
    )
    issued = _parse_utc(
        authorization.get("issued_at_utc"),
        name="compatibility authorization issued_at_utc",
    )
    expires = _parse_utc(
        authorization.get("expires_at_utc"),
        name="compatibility authorization expires_at_utc",
    )
    current = current.astimezone(timezone.utc)
    if (
        not issued <= created <= expires
        or expires - issued > timedelta(minutes=5)
        or (
            require_fresh
            and not issued <= current <= expires
        )
    ):
        raise PermissionError("compatibility authorization is stale")
    return created, issued, expires


def authorize_compat(
    *,
    instruction_id: str,
    authorization_basis: str,
    validity_seconds: int = 300,
    authorization_path: Path | None = None,
    unit_state_reader: UnitStateReader = collect_old_unit_state,
    between_observations: ObservationHook | None = None,
    now: Callable[[], datetime] = _utc_now_datetime,
) -> dict[str, object]:
    if authorization_path is None:
        authorization_path = COMPAT_AUTHORIZATION_PATH
    _require_fixed_path(
        authorization_path,
        COMPAT_AUTHORIZATION_PATH,
        name="compatibility authorization",
    )
    if (
        instruction_id != INSTRUCTION_ID
        or authorization_basis != AUTHORIZATION_BASIS
        or isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("compatibility authorization input is invalid")
    _require_closure_outputs_absent(
        terminal_may_exist=True,
        authorization_may_exist=False,
    )
    terminal, terminal_root = validate_forensic_terminal(
        unit_state_reader=unit_state_reader,
    )
    snapshot = _double_observe_snapshot(
        unit_state_reader,
        between_observations=between_observations,
    )
    issued = now().astimezone(timezone.utc)
    body: dict[str, object] = {
        "schema_version": COMPAT_AUTHORIZATION_SCHEMA,
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
        "instruction_id": instruction_id,
        "authorization_basis": authorization_basis,
        "authorized_uid": os.getuid(),
        "created_at_utc": _format_utc(issued),
        "issued_at_utc": _format_utc(issued),
        "expires_at_utc": _format_utc(
            issued + timedelta(seconds=validity_seconds)
        ),
        "forensic_terminal_root": terminal_root,
        "schema_compatibility": dict(_SCHEMA_COMPATIBILITY),
        "scientific_authority": dict(_SCIENTIFIC_AUTHORITY),
        "frozen_generation_roots": (
            snapshot["frozen_generation_roots"]
        ),
        "absence_generation_roots": (
            snapshot["absence_generation_roots"]
        ),
        "old_unit_state": snapshot["old_unit_state"],
        "payload_observation": dict(_PAYLOAD_OBSERVATION),
        "mutation_authority": dict(_MUTATION_AUTHORITY),
        "compat_lane_authority": dict(_COMPAT_LANE_AUTHORITY),
    }
    if (
        set(body) | {"authorization_fingerprint"}
        != _COMPAT_AUTHORIZATION_KEYS
    ):
        raise AssertionError(
            "internal compatibility authorization keys are incomplete"
        )
    if terminal["scientific_attempt_ordinal"] != 2:
        raise PermissionError("forensic terminal no longer binds r2")
    _validate_live_snapshot(snapshot, unit_state_reader=unit_state_reader)
    return _write_sealed(
        COMPAT_AUTHORIZATION_PATH,
        body,
        fingerprint_field="authorization_fingerprint",
    )


def validate_compat_authorization(
    path: Path | None = None,
    *,
    unit_state_reader: UnitStateReader = collect_old_unit_state,
    require_fresh: bool = True,
    require_future_absence: bool = True,
    now: Callable[[], datetime] = _utc_now_datetime,
) -> tuple[dict[str, object], dict[str, object]]:
    if path is None:
        path = COMPAT_AUTHORIZATION_PATH
    fixed = _require_fixed_path(
        path,
        COMPAT_AUTHORIZATION_PATH,
        name="compatibility authorization",
    )
    authorization, root = _load_sealed_output(
        fixed,
        fingerprint_field="authorization_fingerprint",
        schema=COMPAT_AUTHORIZATION_SCHEMA,
        expected_keys=_COMPAT_AUTHORIZATION_KEYS,
    )
    _validate_common_identity(authorization)
    _validate_common_contracts(authorization)
    if (
        authorization.get("authorized_uid") != os.getuid()
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis")
        != AUTHORIZATION_BASIS
        or not _deep_exact_equal(
            authorization.get("scientific_authority"),
            _SCIENTIFIC_AUTHORITY,
        )
        or not _deep_exact_equal(
            authorization.get("compat_lane_authority"),
            _COMPAT_LANE_AUTHORITY,
        )
    ):
        raise PermissionError("compatibility authorization semantics changed")
    _validate_authorization_freshness(
        authorization,
        current=now(),
        require_fresh=require_fresh,
    )
    terminal, observed_terminal_root = validate_forensic_terminal(
        unit_state_reader=unit_state_reader,
        require_future_absence=require_future_absence,
    )
    del terminal
    if not _same_generation_root(
        authorization.get("forensic_terminal_root"),
        observed_terminal_root,
    ):
        raise PermissionError("forensic terminal generation changed")
    _validate_live_snapshot(
        {
            "session_failure": (
                _observe_session_failure()
            ),
            "frozen_generation_roots": (
                authorization["frozen_generation_roots"]
            ),
            "absence_generation_roots": (
                authorization["absence_generation_roots"]
            ),
            "old_unit_state": authorization["old_unit_state"],
        },
        unit_state_reader=unit_state_reader,
        require_future_absence=require_future_absence,
    )
    return authorization, root


def _component_source_paths(
    *,
    compat_release_path: Path,
    compat_supervisor_path: Path,
    compat_adapter_path: Path,
    compat_unit_realizer_path: Path,
    compat_unit_template_path: Path,
) -> dict[str, Path]:
    return {
        "compat_policy": COMPAT_POLICY_SOURCE_PATH,
        "compat_release": Path(compat_release_path).absolute(),
        "compat_supervisor": Path(compat_supervisor_path).absolute(),
        "compat_adapter": Path(compat_adapter_path).absolute(),
        "compat_unit_realizer": Path(
            compat_unit_realizer_path
        ).absolute(),
        "compat_unit_template": Path(
            compat_unit_template_path
        ).absolute(),
    }


def _fixed_component_source_paths() -> dict[str, Path]:
    return {
        "compat_policy": COMPAT_POLICY_SOURCE_PATH,
        "compat_release": COMPAT_RELEASE_SOURCE_PATH,
        "compat_supervisor": COMPAT_SUPERVISOR_SOURCE_PATH,
        "compat_adapter": COMPAT_ADAPTER_SOURCE_PATH,
        "compat_unit_realizer": COMPAT_UNIT_REALIZER_SOURCE_PATH,
        "compat_unit_template": COMPAT_UNIT_TEMPLATE_PATH,
    }


def _component_evidence_paths(
    *,
    compat_environment_policy_path: Path,
    compat_environment_stability_path: Path,
    compat_environment_postcleanup_path: Path,
    compat_integration_authorization_path: Path,
    compat_integration_receipt_path: Path,
    compat_unit_realization_authorization_path: Path,
    compat_unit_realization_receipt_path: Path,
) -> dict[str, Path]:
    return {
        "compat_environment_policy": Path(
            compat_environment_policy_path
        ).absolute(),
        "compat_environment_stability": Path(
            compat_environment_stability_path
        ).absolute(),
        "compat_environment_postcleanup": Path(
            compat_environment_postcleanup_path
        ).absolute(),
        "compat_integration_authorization": Path(
            compat_integration_authorization_path
        ).absolute(),
        "compat_integration_receipt": Path(
            compat_integration_receipt_path
        ).absolute(),
        "compat_unit_realization_authorization": Path(
            compat_unit_realization_authorization_path
        ).absolute(),
        "compat_unit_realization_receipt": Path(
            compat_unit_realization_receipt_path
        ).absolute(),
    }


def _fixed_component_evidence_paths() -> dict[str, Path]:
    return {
        "compat_environment_policy": COMPAT_ENVIRONMENT_POLICY_PATH,
        "compat_environment_stability": (
            COMPAT_ENVIRONMENT_STABILITY_PATH
        ),
        "compat_environment_postcleanup": (
            COMPAT_ENVIRONMENT_POSTCLEANUP_PATH
        ),
        "compat_integration_authorization": (
            COMPAT_INTEGRATION_AUTHORIZATION_PATH
        ),
        "compat_integration_receipt": COMPAT_INTEGRATION_RECEIPT_PATH,
        "compat_unit_realization_authorization": (
            COMPAT_UNIT_REALIZATION_AUTHORIZATION_PATH
        ),
        "compat_unit_realization_receipt": (
            COMPAT_UNIT_REALIZATION_RECEIPT_PATH
        ),
    }


def _validate_component_path_set(
    sources: Mapping[str, Path],
    evidence: Mapping[str, Path],
) -> None:
    if (
        set(sources) != _COMPAT_SOURCE_LABELS
        or set(evidence) != _COMPAT_EVIDENCE_LABELS
        or not _deep_exact_equal(
            dict(sources),
            _fixed_component_source_paths(),
        )
        or not _deep_exact_equal(
            dict(evidence),
            _fixed_component_evidence_paths(),
        )
    ):
        raise PermissionError(
            "compatibility component labels or fixed paths changed"
        )
    paths = list(sources.values()) + list(evidence.values())
    forbidden = {
        Path(value[0]).absolute()
        for value in _frozen_root_specs().values()
    } | {
        FORENSIC_TERMINAL_PATH,
        COMPAT_AUTHORIZATION_PATH,
        COMPAT_RECEIPT_PATH,
        *list(_absence_paths().values()),
    }
    if (
        len(set(paths)) != len(paths)
        or any(not path.is_absolute() for path in paths)
        or any(path in forbidden for path in paths)
    ):
        raise PermissionError(
            "compatibility components are aliased or off-closure"
        )


def _collect_component_roots(
    sources: Mapping[str, Path],
    evidence: Mapping[str, Path],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    _validate_component_path_set(sources, evidence)
    source_roots = {
        label: _regular_root(path)
        for label, path in sources.items()
    }
    evidence_roots: dict[str, dict[str, object]] = {}
    for label, path in evidence.items():
        payload, root = _sealed_root(
            path,
            fingerprint_field=_COMPAT_EVIDENCE_FINGERPRINT_FIELDS[label],
            schema=None,
        )
        if (
            not isinstance(payload.get("schema_version"), str)
            or not payload["schema_version"]
        ):
            raise PermissionError(
                "compatibility evidence schema is absent"
            )
        evidence_roots[label] = root
    return source_roots, evidence_roots


def _validate_component_roots(
    source_roots: object,
    evidence_roots: object,
    *,
    sources: Mapping[str, Path] | None = None,
    evidence: Mapping[str, Path] | None = None,
) -> None:
    if (
        not isinstance(source_roots, Mapping)
        or set(source_roots) != _COMPAT_SOURCE_LABELS
        or not isinstance(evidence_roots, Mapping)
        or set(evidence_roots) != _COMPAT_EVIDENCE_LABELS
    ):
        raise PermissionError("compatibility component-root labels changed")
    if sources is None:
        sources = {
            label: Path(str(source_roots[label]["path"]))
            for label in _COMPAT_SOURCE_LABELS
        }
    if evidence is None:
        evidence = {
            label: Path(str(evidence_roots[label]["path"]))
            for label in _COMPAT_EVIDENCE_LABELS
        }
    _validate_component_path_set(sources, evidence)
    for label, path in sources.items():
        _validate_file_root(
            source_roots[label],
            expected_path=path,
        )
    for label, path in evidence.items():
        root = evidence_roots[label]
        if (
            not isinstance(root, Mapping)
            or not isinstance(root.get("schema_version"), str)
        ):
            raise PermissionError("compatibility evidence root malformed")
        _validate_sealed_root(
            root,
            expected_path=path,
            fingerprint_field=(
                _COMPAT_EVIDENCE_FINGERPRINT_FIELDS[label]
            ),
            schema=str(root["schema_version"]),
        )


def _load_compat_unit_realizer_validator(raw: bytes):
    """Execute exactly the source generation already bound by the receipt."""

    from types import ModuleType

    name = (
        "tools._cure_lite_v24_actual_unit_realization_"
        "preaccess_compat_c1_for_bridge_validation"
    )
    module = ModuleType(name)
    module.__file__ = str(COMPAT_UNIT_REALIZER_SOURCE_PATH)
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(COMPAT_UNIT_REALIZER_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    return module


def _validate_compat_unit_realization_chain(
    *,
    source_root: Mapping[str, object],
    authorization_root: Mapping[str, object],
    receipt_root: Mapping[str, object],
    allow_runtime_activation: bool = False,
) -> None:
    """Require the unit evidence to be a causal, non-replayed c1 PASS."""

    if not isinstance(allow_runtime_activation, bool):
        raise TypeError("allow_runtime_activation must be boolean")
    raw, observed_source = _read_regular_root(
        COMPAT_UNIT_REALIZER_SOURCE_PATH
    )
    if not _same_generation_root(observed_source, source_root):
        raise PermissionError(
            "compatibility unit realizer generation was replaced"
        )
    realizer = _load_compat_unit_realizer_validator(raw)
    validator = getattr(
        realizer,
        "validate_archival_realization_chain",
        None,
    )
    if not callable(validator):
        raise PermissionError(
            "compatibility unit archival validator is unavailable"
        )
    result = validator(
        COMPAT_UNIT_REALIZATION_AUTHORIZATION_PATH,
        COMPAT_UNIT_REALIZATION_RECEIPT_PATH,
        allow_runtime_activation=allow_runtime_activation,
    )
    if not isinstance(result, Mapping):
        raise PermissionError(
            "compatibility unit archival validator returned no chain"
        )
    authorization = result.get("authorization")
    receipt = result.get("receipt")
    closure = result.get("compatibility_closure")
    if (
        not isinstance(authorization, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(closure, Mapping)
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis")
        != AUTHORIZATION_BASIS
        or authorization.get("unit_name") != COMPAT_UNIT_NAME
        or receipt.get("unit_name") != COMPAT_UNIT_NAME
        or receipt.get("passed") is not True
        or receipt.get("started") is not False
        or receipt.get("enabled") is not False
        or receipt.get("removed") is not False
        or closure.get("automatic_retry_authorized") is not False
        or closure.get("resume_authorized") is not False
        or closure.get("payload_authority") != "none"
        or closure.get("D_R_payload_accessed") is not False
        or closure.get("D_V_payload_accessed") is not False
        or closure.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError(
            "compatibility unit realization semantics changed"
        )
    _, source_after = _read_regular_root(
        COMPAT_UNIT_REALIZER_SOURCE_PATH
    )
    authorization_after, authorization_after_root = _sealed_root(
        COMPAT_UNIT_REALIZATION_AUTHORIZATION_PATH,
        fingerprint_field="authorization_fingerprint",
        schema=str(authorization.get("schema_version")),
    )
    receipt_after, receipt_after_root = _sealed_root(
        COMPAT_UNIT_REALIZATION_RECEIPT_PATH,
        fingerprint_field="receipt_fingerprint",
        schema=str(receipt.get("schema_version")),
    )
    if (
        not _same_generation_root(source_after, source_root)
        or not _same_generation_root(
            authorization_after_root,
            authorization_root,
        )
        or not _same_generation_root(
            receipt_after_root,
            receipt_root,
        )
        or not _deep_exact_equal(
            dict(authorization_after),
            dict(authorization),
        )
        or not _deep_exact_equal(dict(receipt_after), dict(receipt))
    ):
        raise PermissionError(
            "compatibility unit realization changed during validation"
        )


def _expected_runtime_contract() -> dict[str, str]:
    return {
        "unit_name": COMPAT_UNIT_NAME,
        "runtime_spec_path": str(COMPAT_RUNTIME_SPEC_PATH),
        "runtime_launch_authorization_path": str(
            COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "runtime_artifact_root": str(COMPAT_RUNTIME_ARTIFACT_ROOT),
        "gpu_lease_root": str(COMPAT_GPU_LEASE_ROOT),
        "scientific_authorization_path": str(
            SCIENTIFIC_AUTHORIZATION_PATH
        ),
        "scientific_access_audit_path": str(
            SCIENTIFIC_ACCESS_AUDIT_PATH
        ),
        "scientific_run_root": str(SCIENTIFIC_RUN_ROOT),
        "scientific_result_receipt_path": str(
            SCIENTIFIC_RESULT_RECEIPT_PATH
        ),
        "compatibility_receipt_path": str(COMPAT_RECEIPT_PATH),
    }


def _validate_runtime_contract(value: object) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _RUNTIME_CONTRACT_KEYS
        or not _deep_exact_equal(
            dict(value),
            _expected_runtime_contract(),
        )
    ):
        raise PermissionError("compatibility runtime contract changed")
    return dict(value)


def _validate_expected_spec_contract(
    expected_spec: Mapping[str, object],
    *,
    receipt: Mapping[str, object] | None = None,
) -> None:
    body = dict(expected_spec)
    fingerprint = body.pop("runtime_spec_fingerprint", None)
    preaccess = expected_spec.get("scientific_preaccess")
    artifacts = expected_spec.get("artifacts")
    runtime = expected_spec.get("runtime")
    environment = expected_spec.get("environment")
    child = expected_spec.get("child")
    source_bindings = expected_spec.get("source_bindings")
    if (
        expected_spec.get("schema_version") != RUNTIME_SPEC_SCHEMA
        or not isinstance(fingerprint, str)
        or _SHA.fullmatch(fingerprint) is None
        or fingerprint != stable_fingerprint(body)
        or not isinstance(preaccess, Mapping)
        or not isinstance(artifacts, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(environment, Mapping)
        or not isinstance(child, Mapping)
        or not isinstance(source_bindings, Mapping)
    ):
        raise PermissionError(
            "expected compatibility runtime spec is incomplete"
        )
    systemd = runtime.get("systemd")
    if not isinstance(systemd, Mapping):
        raise PermissionError(
            "expected compatibility systemd contract is incomplete"
        )
    gpu_lease_path = environment.get("gpu_lease_path")
    if (
        expected_spec.get("candidate") != CANDIDATE
        or expected_spec.get("stage_id") != STAGE_ID
        or expected_spec.get("attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or expected_spec.get("attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or preaccess.get("authorization_path")
        != str(SCIENTIFIC_AUTHORIZATION_PATH)
        or preaccess.get("authorization_required_schema")
        != SCIENTIFIC_AUTHORIZATION_SCHEMA
        or preaccess.get("access_audit_path")
        != str(SCIENTIFIC_ACCESS_AUDIT_PATH)
        or preaccess.get("access_audit_required_schema")
        != SCIENTIFIC_ACCESS_AUDIT_SCHEMA
        or preaccess.get("source_closure_fingerprint_103")
        != SOURCE_CLOSURE_FINGERPRINT_103
        or artifacts.get("root") != str(COMPAT_RUNTIME_ARTIFACT_ROOT)
        or systemd.get("unit_name") != COMPAT_UNIT_NAME
        or child.get("entrypoint_path")
        != str(COMPAT_ADAPTER_SOURCE_PATH)
        or not isinstance(gpu_lease_path, str)
        or Path(gpu_lease_path).absolute().parent
        != COMPAT_GPU_LEASE_ROOT
    ):
        raise PermissionError(
            "expected spec is outside compatibility c1"
        )
    if receipt is not None:
        roots = receipt.get("compatibility_source_roots")
        if not isinstance(roots, Mapping):
            raise PermissionError(
                "compatibility receipt source roots are absent"
            )
        adapter_root = roots.get("compat_adapter")
        supervisor_root = roots.get("compat_supervisor")
        if (
            not isinstance(adapter_root, Mapping)
            or not isinstance(supervisor_root, Mapping)
            or source_bindings.get("r2_adapter_path")
            != adapter_root.get("path")
            or source_bindings.get("r2_adapter_file_sha256")
            != adapter_root.get("file_sha256")
            or source_bindings.get("child_entry_file_sha256")
            != adapter_root.get("file_sha256")
            or source_bindings.get("supervisor_file_sha256")
            != supervisor_root.get("file_sha256")
        ):
            raise PermissionError(
                "expected spec source roots are not receipt-bound"
            )


def seal_receipt(
    *,
    compat_release_path: Path,
    compat_supervisor_path: Path,
    compat_adapter_path: Path,
    compat_unit_realizer_path: Path,
    compat_unit_template_path: Path,
    compat_environment_policy_path: Path,
    compat_environment_stability_path: Path,
    compat_environment_postcleanup_path: Path,
    compat_integration_authorization_path: Path,
    compat_integration_receipt_path: Path,
    compat_unit_realization_authorization_path: Path,
    compat_unit_realization_receipt_path: Path,
    receipt_path: Path | None = None,
    unit_state_reader: UnitStateReader = collect_old_unit_state,
    between_observations: ObservationHook | None = None,
    now: Callable[[], datetime] = _utc_now_datetime,
) -> dict[str, object]:
    if receipt_path is None:
        receipt_path = COMPAT_RECEIPT_PATH
    _require_fixed_path(
        receipt_path,
        COMPAT_RECEIPT_PATH,
        name="compatibility receipt",
    )
    _require_closure_outputs_absent(
        terminal_may_exist=True,
        authorization_may_exist=True,
    )
    authorization, authorization_root = validate_compat_authorization(
        unit_state_reader=unit_state_reader,
        require_fresh=True,
        now=now,
    )
    terminal, terminal_root = validate_forensic_terminal(
        unit_state_reader=unit_state_reader,
    )
    sources = _component_source_paths(
        compat_release_path=compat_release_path,
        compat_supervisor_path=compat_supervisor_path,
        compat_adapter_path=compat_adapter_path,
        compat_unit_realizer_path=compat_unit_realizer_path,
        compat_unit_template_path=compat_unit_template_path,
    )
    evidence = _component_evidence_paths(
        compat_environment_policy_path=compat_environment_policy_path,
        compat_environment_stability_path=(
            compat_environment_stability_path
        ),
        compat_environment_postcleanup_path=(
            compat_environment_postcleanup_path
        ),
        compat_integration_authorization_path=(
            compat_integration_authorization_path
        ),
        compat_integration_receipt_path=compat_integration_receipt_path,
        compat_unit_realization_authorization_path=(
            compat_unit_realization_authorization_path
        ),
        compat_unit_realization_receipt_path=(
            compat_unit_realization_receipt_path
        ),
    )
    snapshot_first = _collect_live_snapshot(unit_state_reader)
    sources_first, evidence_first = _collect_component_roots(
        sources,
        evidence,
    )
    if between_observations is not None:
        between_observations()
    snapshot_second = _collect_live_snapshot(unit_state_reader)
    sources_second, evidence_second = _collect_component_roots(
        sources,
        evidence,
    )
    if (
        not _deep_exact_equal(snapshot_first, snapshot_second)
        or not _deep_exact_equal(sources_first, sources_second)
        or not _deep_exact_equal(evidence_first, evidence_second)
    ):
        raise PermissionError(
            "compatibility closure changed between observations"
        )
    _validate_compat_unit_realization_chain(
        source_root=sources_second["compat_unit_realizer"],
        authorization_root=evidence_second[
            "compat_unit_realization_authorization"
        ],
        receipt_root=evidence_second[
            "compat_unit_realization_receipt"
        ],
    )
    created = now().astimezone(timezone.utc)
    _created_auth, issued, expires = _validate_authorization_freshness(
        authorization,
        current=created,
        require_fresh=True,
    )
    if not issued <= created <= expires:
        raise PermissionError("receipt is outside authorization lifetime")
    body: dict[str, object] = {
        "schema_version": COMPAT_RECEIPT_SCHEMA,
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
        "created_at_utc": _format_utc(created),
        "forensic_terminal_root": terminal_root,
        "compatibility_authorization_root": authorization_root,
        "instruction_id": authorization["instruction_id"],
        "authorization_basis": authorization["authorization_basis"],
        "schema_compatibility": dict(_SCHEMA_COMPATIBILITY),
        "scientific_authority": dict(_SCIENTIFIC_AUTHORITY),
        "frozen_generation_roots": (
            snapshot_second["frozen_generation_roots"]
        ),
        "compatibility_source_roots": sources_second,
        "compatibility_evidence_roots": evidence_second,
        "compatibility_runtime_contract": _expected_runtime_contract(),
        "absence_generation_roots": (
            snapshot_second["absence_generation_roots"]
        ),
        "scientific_output_contract": {
            "run_root": str(SCIENTIFIC_RUN_ROOT),
            "result_receipt": str(SCIENTIFIC_RESULT_RECEIPT_PATH),
            "compat_run_root_alias": str(COMPAT_RUN_ROOT_ALIAS_PATH),
            "compat_result_receipt_alias": str(
                COMPAT_RESULT_RECEIPT_ALIAS_PATH
            ),
            "original_r2_paths_retained": True,
            "compatibility_aliases_forbidden": True,
        },
        "old_unit_state": snapshot_second["old_unit_state"],
        "payload_observation": dict(_PAYLOAD_OBSERVATION),
        "mutation_authority": dict(_MUTATION_AUTHORITY),
        "compat_lane_authority": dict(_COMPAT_LANE_AUTHORITY),
        "runtime_launch_authorized": False,
        "systemd_mutation_authorized": False,
        "compatibility_closure_passed": True,
    }
    if set(body) | {"receipt_fingerprint"} != _COMPAT_RECEIPT_KEYS:
        raise AssertionError("internal compatibility receipt keys incomplete")
    if (
        terminal["forensic_closure_passed"] is not True
        or not _same_generation_root(
            authorization["forensic_terminal_root"],
            terminal_root,
        )
    ):
        raise PermissionError("compatibility predecessor chain changed")
    _validate_live_snapshot(
        snapshot_second,
        unit_state_reader=unit_state_reader,
    )
    _validate_component_roots(
        sources_second,
        evidence_second,
        sources=sources,
        evidence=evidence,
    )
    return _write_sealed(
        COMPAT_RECEIPT_PATH,
        body,
        fingerprint_field="receipt_fingerprint",
    )


def validate_compatibility_receipt(
    path: Path | None = None,
    *,
    unit_state_reader: UnitStateReader = collect_old_unit_state,
    require_future_absence: bool = True,
    allow_runtime_activation: bool = False,
    now: Callable[[], datetime] = _utc_now_datetime,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(allow_runtime_activation, bool):
        raise TypeError("allow_runtime_activation must be boolean")
    if path is None:
        path = COMPAT_RECEIPT_PATH
    fixed = _require_fixed_path(
        path,
        COMPAT_RECEIPT_PATH,
        name="compatibility receipt",
    )
    receipt, root = _load_sealed_output(
        fixed,
        fingerprint_field="receipt_fingerprint",
        schema=COMPAT_RECEIPT_SCHEMA,
        expected_keys=_COMPAT_RECEIPT_KEYS,
    )
    _validate_common_identity(receipt)
    _validate_common_contracts(receipt)
    if (
        not _deep_exact_equal(
            receipt.get("scientific_authority"),
            _SCIENTIFIC_AUTHORITY,
        )
        or receipt.get("runtime_launch_authorized") is not False
        or receipt.get("systemd_mutation_authorized") is not False
        or receipt.get("compatibility_closure_passed") is not True
        or not _deep_exact_equal(
            receipt.get("compat_lane_authority"),
            _COMPAT_LANE_AUTHORITY,
        )
    ):
        raise PermissionError("compatibility receipt semantics changed")
    authorization, authorization_root = (
        validate_compat_authorization(
            unit_state_reader=unit_state_reader,
            require_fresh=False,
            require_future_absence=require_future_absence,
            now=now,
        )
    )
    terminal, terminal_root = validate_forensic_terminal(
        unit_state_reader=unit_state_reader,
        require_future_absence=require_future_absence,
    )
    del terminal
    if (
        not _same_generation_root(
            receipt["compatibility_authorization_root"],
            authorization_root,
        )
        or not _same_generation_root(
            receipt["forensic_terminal_root"],
            terminal_root,
        )
        or receipt["instruction_id"] != authorization["instruction_id"]
        or receipt["authorization_basis"]
        != authorization["authorization_basis"]
    ):
        raise PermissionError("compatibility receipt lineage changed")
    created = _parse_utc(
        receipt["created_at_utc"],
        name="compatibility receipt created_at_utc",
    )
    _created_auth, issued, expires = _validate_authorization_freshness(
        authorization,
        current=created,
        require_fresh=False,
    )
    if not issued <= created <= expires:
        raise PermissionError("receipt chronology changed")
    _validate_frozen_roots(receipt["frozen_generation_roots"])
    _validate_component_roots(
        receipt["compatibility_source_roots"],
        receipt["compatibility_evidence_roots"],
    )
    _validate_compat_unit_realization_chain(
        source_root=receipt["compatibility_source_roots"][
            "compat_unit_realizer"
        ],
        authorization_root=receipt["compatibility_evidence_roots"][
            "compat_unit_realization_authorization"
        ],
        receipt_root=receipt["compatibility_evidence_roots"][
            "compat_unit_realization_receipt"
        ],
        allow_runtime_activation=allow_runtime_activation,
    )
    _validate_runtime_contract(receipt["compatibility_runtime_contract"])
    output_contract = receipt.get("scientific_output_contract")
    if (
        not isinstance(output_contract, Mapping)
        or set(output_contract)
        != {
            "run_root",
            "result_receipt",
            "compat_run_root_alias",
            "compat_result_receipt_alias",
            "original_r2_paths_retained",
            "compatibility_aliases_forbidden",
        }
        or output_contract.get("run_root") != str(SCIENTIFIC_RUN_ROOT)
        or output_contract.get("result_receipt")
        != str(SCIENTIFIC_RESULT_RECEIPT_PATH)
        or output_contract.get("compat_run_root_alias")
        != str(COMPAT_RUN_ROOT_ALIAS_PATH)
        or output_contract.get("compat_result_receipt_alias")
        != str(COMPAT_RESULT_RECEIPT_ALIAS_PATH)
        or output_contract.get("original_r2_paths_retained") is not True
        or output_contract.get("compatibility_aliases_forbidden") is not True
    ):
        raise PermissionError("scientific output/alias contract changed")
    observed_state = _read_unit_state(unit_state_reader)
    if not _deep_exact_equal(observed_state, receipt["old_unit_state"]):
        raise PermissionError("old unit state changed after receipt")
    _validate_absences(
        receipt["absence_generation_roots"],
        labels=(
            _ABSENCE_LABELS
            if require_future_absence
            else _ALWAYS_PROTECTED_ABSENCE_LABELS
        ),
    )
    return receipt, root


def verify_compatibility_receipt(
    path: Path | None = None,
    expected_spec: Mapping[str, object] | None = None,
    require_spec_binding: bool = False,
    allow_runtime_activation: bool = False,
    *,
    unit_state_reader: UnitStateReader = collect_old_unit_state,
    now: Callable[[], datetime] = _utc_now_datetime,
) -> dict[str, object]:
    """Consumer-facing verifier for the compatibility release/supervisor.

    A release which has not yet written a spec gets the strongest absence
    replay.  A consumer supplying an in-memory or already-sealed spec instead
    gets an exact compatibility-lane contract check; the receipt never becomes
    a surrogate scientific audit or a wildcard schema waiver.
    """

    if not isinstance(require_spec_binding, bool):
        raise TypeError("require_spec_binding must be boolean")
    if not isinstance(allow_runtime_activation, bool):
        raise TypeError("allow_runtime_activation must be boolean")
    if expected_spec is not None and not isinstance(expected_spec, Mapping):
        raise TypeError("expected_spec must be a mapping or None")
    if (
        (expected_spec is None and require_spec_binding)
        or (
            expected_spec is not None
            and not require_spec_binding
        )
    ):
        raise PermissionError(
            "compatibility spec binding phase is inconsistent"
        )
    receipt, _root = validate_compatibility_receipt(
        path=path,
        unit_state_reader=unit_state_reader,
        require_future_absence=expected_spec is None,
        allow_runtime_activation=allow_runtime_activation,
        now=now,
    )
    _validate_runtime_contract(receipt["compatibility_runtime_contract"])
    if expected_spec is not None:
        sealed_spec, _spec_root = _sealed_root(
            COMPAT_RUNTIME_SPEC_PATH,
            fingerprint_field="runtime_spec_fingerprint",
            schema=RUNTIME_SPEC_SCHEMA,
        )
        if not _deep_exact_equal(
            dict(sealed_spec),
            dict(expected_spec),
        ):
            raise PermissionError(
                "fixed compatibility runtime spec differs from consumer view"
            )
        _validate_expected_spec_contract(expected_spec, receipt=receipt)
    return receipt


def verify_compatibility_prewrite_spec(
    path: Path,
    expected_spec: Mapping[str, object],
    *,
    unit_state_reader: UnitStateReader = collect_old_unit_state,
    now: Callable[[], datetime] = _utc_now_datetime,
) -> dict[str, object]:
    """Validate one exact producer preview while runtime paths stay absent."""

    if not isinstance(expected_spec, Mapping):
        raise TypeError("expected_spec must be a mapping")
    receipt, _root = validate_compatibility_receipt(
        path=path,
        unit_state_reader=unit_state_reader,
        require_future_absence=True,
        allow_runtime_activation=False,
        now=now,
    )
    _validate_runtime_contract(receipt["compatibility_runtime_contract"])
    _validate_expected_spec_contract(expected_spec, receipt=receipt)
    return receipt


def _add_component_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--compat-release", type=Path, required=True)
    parser.add_argument("--compat-supervisor", type=Path, required=True)
    parser.add_argument("--compat-adapter", type=Path, required=True)
    parser.add_argument("--compat-unit-realizer", type=Path, required=True)
    parser.add_argument("--compat-unit-template", type=Path, required=True)
    parser.add_argument(
        "--compat-environment-policy",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--compat-environment-stability",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--compat-environment-postcleanup",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--compat-integration-authorization",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--compat-integration-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--compat-unit-realization-authorization",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--compat-unit-realization-receipt",
        type=Path,
        required=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append-only forensic/schema bridge for the v24 scientific-r2 "
            "runtime-compatibility-c1 prewrite failure."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "create-forensic-terminal",
        help="seal the exact retrospective prewrite failure",
    )
    authorize = sub.add_parser(
        "authorize-compat",
        help="seal a short-lived metadata-only compatibility authorization",
    )
    authorize.add_argument("--instruction-id", required=True)
    authorize.add_argument("--authorization-basis", required=True)
    authorize.add_argument(
        "--validity-seconds",
        type=int,
        default=300,
    )
    receipt = sub.add_parser(
        "seal-receipt",
        help="bind a new compatibility closure without launching it",
    )
    _add_component_arguments(receipt)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create-forensic-terminal":
        payload = create_forensic_terminal()
    elif args.command == "authorize-compat":
        payload = authorize_compat(
            instruction_id=args.instruction_id,
            authorization_basis=args.authorization_basis,
            validity_seconds=args.validity_seconds,
        )
    elif args.command == "seal-receipt":
        payload = seal_receipt(
            compat_release_path=args.compat_release,
            compat_supervisor_path=args.compat_supervisor,
            compat_adapter_path=args.compat_adapter,
            compat_unit_realizer_path=args.compat_unit_realizer,
            compat_unit_template_path=args.compat_unit_template,
            compat_environment_policy_path=(
                args.compat_environment_policy
            ),
            compat_environment_stability_path=(
                args.compat_environment_stability
            ),
            compat_environment_postcleanup_path=(
                args.compat_environment_postcleanup
            ),
            compat_integration_authorization_path=(
                args.compat_integration_authorization
            ),
            compat_integration_receipt_path=(
                args.compat_integration_receipt
            ),
            compat_unit_realization_authorization_path=(
                args.compat_unit_realization_authorization
            ),
            compat_unit_realization_receipt_path=(
                args.compat_unit_realization_receipt
            ),
        )
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError("unknown compatibility command")
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
