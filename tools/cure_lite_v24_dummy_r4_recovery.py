#!/usr/bin/env python3
"""Exact one-shot recovery for the failed CURE-Lite v24 dummy r4 run.

Only one already archived r4 identity is accepted.  ``authorize`` may seal a
short-lived authorization; ``apply`` may consume it once for the sole sequence

    remove-exact-r4-fragment -> daemon-reload -> verify-not-found

No start, stop, enable, retry, GPU, or dataset authority exists here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Callable, Mapping, Sequence


try:
    from tools import cure_lite_v24_dummy_r3_recovery as hardened
except ModuleNotFoundError:
    _HARDENED_PATH = Path(__file__).with_name(
        "cure_lite_v24_dummy_r3_recovery.py"
    )
    _HARDENED_SPEC = importlib.util.spec_from_file_location(
        "cure_lite_v24_dummy_r3_recovery_for_r4",
        _HARDENED_PATH,
    )
    if _HARDENED_SPEC is None or _HARDENED_SPEC.loader is None:
        raise RuntimeError("cannot load frozen r3 hardened I/O for r4 recovery")
    hardened = importlib.util.module_from_spec(_HARDENED_SPEC)
    sys.modules[_HARDENED_SPEC.name] = hardened
    _HARDENED_SPEC.loader.exec_module(hardened)


integration = hardened.integration
realizer = hardened.realizer

REPOSITORY = Path("/home/md0/ly/cure_lite")
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
)
SCENARIO_ROOT = EVIDENCE_ROOT / "supervisor_v2_systemd_integration_r4"
CONTROL_ROOT = SCENARIO_ROOT / "control"
RUNTIME_ROOT = SCENARIO_ROOT / "runtime"
HEARTBEAT_ROOT = RUNTIME_ROOT / "heartbeat"
INVOCATION_ROOT = RUNTIME_ROOT / "systemd-invocations"

SCENARIO_ID = "supervisor-v2-dummy-r4-202607300520b16b"
UNIT_NAME = (
    "cure-lite-v24-supervisor-integration-"
    "supervisor-v2-dummy-r4-202607300520b16b.service"
)
STAGE_ID = f"systemd_integration_dummy_{SCENARIO_ID}"
ATTEMPT_ID = f"systemd_integration_dummy_attempt_{SCENARIO_ID}"
CANDIDATE = "systemd-integration-dummy"
EXECUTION_KIND = "systemd_integration_dummy"
INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后运行"
AUTHORIZED_UID = 1008
INVOCATION_ID = "eafd931aeeee4c02a1bbd2a919eec215"

EXPECTED_RUNTIME_UNIT_DIRECTORY = Path("/run/user/1008/systemd/user")
EXPECTED_FRAGMENT_PATH = EXPECTED_RUNTIME_UNIT_DIRECTORY / UNIT_NAME
EXPECTED_FRAGMENT_SHA256 = (
    "27f3cc19aa04649b02151798eb3bb50bd731e0d72d8694e963d34810240fa061"
)
EXPECTED_FRAGMENT_DEVICE = 54
EXPECTED_FRAGMENT_INODE = 38551
EXPECTED_FRAGMENT_OWNER_UID = 1008
EXPECTED_FRAGMENT_MODE = 0o600
EXPECTED_FRAGMENT_NLINK = 1
EXPECTED_FRAGMENT_SIZE = 1883
EXPECTED_FRAGMENT_MTIME_NS = 1785363309482597446
EXPECTED_FRAGMENT_CTIME_NS = 1785363309482597446

ORIGINAL_AUTHORIZATION_PATH = CONTROL_ROOT / "authorization.json"
ORIGINAL_RUNTIME_SPEC_PATH = CONTROL_ROOT / "runtime-spec.json"
ORIGINAL_LAUNCH_LEASE_PATH = RUNTIME_ROOT / "launch-lease.json"
ORIGINAL_PRECOMMIT_PATH = RUNTIME_ROOT / "precommit-phase.json"
ORIGINAL_ATTEMPT_COMMIT_PATH = RUNTIME_ROOT / "attempt-commit.json"
ORIGINAL_SIDECAR_PATH = INVOCATION_ROOT / f"{INVOCATION_ID}.json"
ORIGINAL_CONSUMED_FAILURE_PATH = RUNTIME_ROOT / "consumed-start-failure.json"
ORIGINAL_TERMINAL_PATH = CONTROL_ROOT / "integration-terminal.json"
ORIGINAL_REMOVAL_STATE_PATH = CONTROL_ROOT / "removal-state.json"

RECOVERY_AUTHORIZATION_PATH = (
    CONTROL_ROOT / "r4-exact-recovery-authorization.json"
)
RECOVERY_INTENT_PATH = CONTROL_ROOT / "r4-exact-recovery-intent.json"
RECOVERY_TERMINAL_PATH = CONTROL_ROOT / "r4-exact-recovery-terminal.json"

AUTHORIZATION_SCHEMA = "cure-lite-v24-dummy-r4-exact-recovery-authorization-v1"
INTENT_SCHEMA = "cure-lite-v24-dummy-r4-exact-recovery-intent-v1"
TERMINAL_SCHEMA = "cure-lite-v24-dummy-r4-exact-recovery-terminal-v1"
FROZEN_HARDENED_IO_SHA256 = (
    "b3d7fd5b98f70db98ec637dbbed3bc4f428a9290cb32cdc39e6fe46f0cc0a7f4"
)

_SHA = re.compile(r"[0-9a-f]{64}")
_BOOT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
_FILE_BINDING_KEYS = {
    "path",
    "resolved_path",
    "path_is_symlink",
    "file_sha256",
    "device",
    "inode",
    "owner_uid",
    "mode",
}
_FRAGMENT_IDENTITY_KEYS = {
    "fragment_path",
    "fragment_sha256",
    "device",
    "inode",
    "owner_uid",
    "mode",
    "nlink",
    "size",
    "mtime_ns",
    "ctime_ns",
}

_ARCHIVED_EVIDENCE_ANCHORS: dict[str, dict[str, object]] = {
    "authorization": {
        "path": str(ORIGINAL_AUTHORIZATION_PATH),
        "file_sha256": (
            "3c4c68e81591c2f2b349432d078c48075427382a8d01d4b9a990dfa620d23dd9"
        ),
        "fingerprint_field": "authorization_fingerprint",
        "fingerprint": (
            "f0f3d6ff9064595d93c98253dac295bae3e2713f2313ca1e7fa2291427320a63"
        ),
        "schema_version": (
            "cure-lite-v24-supervisor-v2-systemd-integration-authorization-v2"
        ),
        "mode": 0o444,
    },
    "runtime_spec": {
        "path": str(ORIGINAL_RUNTIME_SPEC_PATH),
        "file_sha256": (
            "df853b4789638e1cac2dcb3cbc721160c63d5e8cbac1111c113f4ac57a6a737c"
        ),
        "fingerprint_field": "runtime_spec_fingerprint",
        "fingerprint": (
            "7bbec06c9e920ceae7ace6433cbd96bb9b5abe260a8383af0591674d5918da6e"
        ),
        "schema_version": "cure-lite-v24-dr-runtime-supervisor-spec-v2",
        "mode": 0o444,
    },
    "launch_lease": {
        "path": str(ORIGINAL_LAUNCH_LEASE_PATH),
        "file_sha256": (
            "ed69c1aee649b0e4b6ad658749424c7b7e0086eab89133126d29117a7e5d309b"
        ),
        "fingerprint_field": "launch_lease_fingerprint",
        "fingerprint": (
            "d5be45b011adec5b2dc2a656c1bf9685424e941405638eb70907cd8326a309bc"
        ),
        "schema_version": "cure-lite-v24-dr-launch-lease-v1",
        "mode": 0o444,
    },
    "precommit": {
        "path": str(ORIGINAL_PRECOMMIT_PATH),
        "file_sha256": (
            "41465e296bf9d9367c7a26ca3bbd872f4154c2153aea02e4efdc3123955a5de9"
        ),
        "fingerprint_field": "phase_receipt_fingerprint",
        "fingerprint": (
            "9eafa2e9688b976e8d54b0c8911f0c6fabc201b7208566b183f6028edbf1c2b2"
        ),
        "schema_version": "cure-lite-v24-dr-runtime-phase-receipt-v1",
        "mode": 0o444,
    },
    "attempt_commit": {
        "path": str(ORIGINAL_ATTEMPT_COMMIT_PATH),
        "file_sha256": (
            "a3f2b7a2b904dcde154e8889ef0358088024a8b8d3b692220cfe8e2f10729f04"
        ),
        "fingerprint_field": "attempt_commit_fingerprint",
        "fingerprint": (
            "2447a42b3d7ce7e1724a850c3cfa481d2c127204d7811e3cdb9827c7656e3c4b"
        ),
        "schema_version": "cure-lite-v24-dr-attempt-commit-v2",
        "mode": 0o444,
    },
    "systemd_sidecar": {
        "path": str(ORIGINAL_SIDECAR_PATH),
        "file_sha256": (
            "46f793ba45182e16a6fb2466cfb325eedf185c29cf3c9a1b9fcf589d18b9ea39"
        ),
        "fingerprint_field": "systemd_terminal_fingerprint",
        "fingerprint": (
            "8ef43c821e656681dc132c6d4572dca9026cf0ca40a372bd9052a9cd90014d08"
        ),
        "schema_version": "cure-lite-v24-dr-systemd-terminal-v1",
        "mode": 0o444,
    },
    "consumed_start_failure": {
        "path": str(ORIGINAL_CONSUMED_FAILURE_PATH),
        "file_sha256": (
            "7cde7ee0dca2fe6b0edfe5c678baa1342af226a251e5bfa49de45ee8e9a6c28e"
        ),
        "fingerprint_field": "consumed_start_failure_fingerprint",
        "fingerprint": (
            "71777aab978d6e827d4e6fc2f1d991e5e789df79d60b424765cae949f5b0c02a"
        ),
        "schema_version": "cure-lite-v24-dr-consumed-start-failure-v1",
        "mode": 0o444,
    },
    "integration_terminal": {
        "path": str(ORIGINAL_TERMINAL_PATH),
        "file_sha256": (
            "845a1b87517b435ae67969c0fde27b824e33f3b451113495bef206b21cb50fcc"
        ),
        "fingerprint_field": "integration_terminal_fingerprint",
        "fingerprint": (
            "117a7011a87dc3a7a544cb98842a2ecb6f95b5643c8856a7d38ad30ac66e00e5"
        ),
        "schema_version": (
            "cure-lite-v24-supervisor-v2-systemd-integration-terminal-v1"
        ),
        "mode": 0o444,
    },
    "removal_state": {
        "path": str(ORIGINAL_REMOVAL_STATE_PATH),
        "file_sha256": (
            "409e8484cc7dcdb05d1614f5e7d425de9380290f23d838a5bdf6e6781080c121"
        ),
        "fingerprint_field": "removal_state_fingerprint",
        "fingerprint": (
            "78a8fa0b7769f0830a9069c2850fb419e979c551f46b174bc8017bc3853f3763"
        ),
        "schema_version": (
            "cure-lite-v24-supervisor-v2-integration-removal-state-v1"
        ),
        "mode": 0o444,
    },
}

_AUTHORIZATION_KEYS = {
    "schema_version",
    "scenario_id",
    "unit_name",
    "instruction_id",
    "authorization_basis",
    "authorized_uid",
    "issued_at_utc",
    "expires_at_utc",
    "archived_roots",
    "archived_executable_bindings",
    "current_recovery_tool_binding",
    "current_required_executable_bindings",
    "manager_generation",
    "unit_path_policy",
    "fragment_identity",
    "inactive_static_state",
    "authorized_action",
    "remove_authorized",
    "daemon_reload_authorized",
    "not_found_verification_authorized",
    "start_authorized",
    "stop_authorized",
    "enable_authorized",
    "automatic_retry_authorized",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "gpu_accessed",
    "recovery_authorization_fingerprint",
}
_INTENT_KEYS = {
    "schema_version",
    "created_at_utc",
    "scenario_id",
    "unit_name",
    "recovery_authorization_path",
    "recovery_authorization_file_sha256",
    "recovery_authorization_fingerprint",
    "fragment_identity",
    "inactive_static_state",
    "authorized_action",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "gpu_accessed",
    "recovery_intent_fingerprint",
}
_TERMINAL_KEYS = {
    "schema_version",
    "created_at_utc",
    "scenario_id",
    "unit_name",
    "recovery_authorization_fingerprint",
    "recovery_intent_fingerprint",
    "action_started_at_utc",
    "completed_actions",
    "fragment_absent",
    "post_removal_unit_state",
    "passed",
    "error_type",
    "error_message",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "gpu_accessed",
    "recovery_terminal_fingerprint",
}


def _timestamp(value: object, *, name: str) -> datetime:
    return hardened._timestamp(value, name=name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _assert_repository() -> None:
    if (
        Path(__file__).resolve().parents[1] != REPOSITORY
        or os.getuid() != AUTHORIZED_UID
    ):
        raise PermissionError("r4 recovery repository or UID changed")


def _binding(path: Path) -> dict[str, object]:
    return hardened._file_binding(path.absolute())


def _current_recovery_tool_binding() -> dict[str, object]:
    return _binding(Path(__file__).absolute())


def _current_required_bindings() -> dict[str, object]:
    hardened_path = Path(str(hardened.__file__)).absolute()
    integration_path = Path(str(integration.__file__)).absolute()
    realizer_path = Path(str(realizer.__file__)).absolute()
    expected_hardened = REPOSITORY / "tools/cure_lite_v24_dummy_r3_recovery.py"
    expected_integration = (
        REPOSITORY / "tools/cure_lite_v24_user_systemd_integration.py"
    )
    expected_realizer = (
        REPOSITORY / "tools/cure_lite_v24_realize_systemd_unit.py"
    )
    if (
        hardened_path != expected_hardened
        or integration_path != expected_integration
        or realizer_path != expected_realizer
        or Path(str(integration.realizer.__file__)).absolute()
        != expected_realizer
        or str(realizer.SYSTEMCTL_PATH) != "/usr/bin/systemctl"
        or str(realizer.SYSTEMD_PATH) != "/usr/bin/systemd-path"
        or str(realizer.SYSTEMD_ANALYZE) != "/usr/bin/systemd-analyze"
    ):
        raise PermissionError("r4 recovery dependency origin changed")
    hardened_binding = _binding(hardened_path)
    if hardened_binding["file_sha256"] != FROZEN_HARDENED_IO_SHA256:
        raise PermissionError("frozen r3 hardened I/O changed")
    return {
        "hardened_io_library": hardened_binding,
        "integration_library": _binding(integration_path),
        "realizer_library": _binding(realizer_path),
        "systemctl": _binding(Path("/usr/bin/systemctl")),
        "systemd_path": _binding(Path("/usr/bin/systemd-path")),
        "systemd_analyze": _binding(Path("/usr/bin/systemd-analyze")),
    }


def _validate_binding_shape(
    value: object,
    *,
    name: str,
    expected_path: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FILE_BINDING_KEYS:
        raise PermissionError(f"r4 archived binding malformed:{name}")
    binding = dict(value)
    if (
        binding.get("path") != expected_path
        or not isinstance(binding.get("resolved_path"), str)
        or not isinstance(binding.get("path_is_symlink"), bool)
        or _SHA.fullmatch(str(binding.get("file_sha256"))) is None
        or any(
            isinstance(binding.get(field), bool)
            or not isinstance(binding.get(field), int)
            or int(binding[field]) < 0
            for field in ("device", "inode", "owner_uid", "mode")
        )
    ):
        raise PermissionError(f"r4 archived binding malformed:{name}")
    return binding


def _read_archived(name: str) -> dict[str, object]:
    anchor = _ARCHIVED_EVIDENCE_ANCHORS[name]
    raw, _current = hardened._read_regular_file_snapshot(
        Path(str(anchor["path"])),
        expected_owner_uid=AUTHORIZED_UID,
        expected_mode=int(anchor["mode"]),
    )
    if hashlib.sha256(raw).hexdigest() != anchor["file_sha256"]:
        raise PermissionError(f"r4 archived file digest changed:{name}")
    payload = json.loads(raw.decode("utf-8"))
    if (
        not isinstance(payload, dict)
        or raw
        != (hardened.canonical_json(payload) + "\n").encode("utf-8")
        or payload.get("schema_version") != anchor["schema_version"]
    ):
        raise PermissionError(f"r4 archived encoding changed:{name}")
    body = dict(payload)
    fingerprint = body.pop(str(anchor["fingerprint_field"]), None)
    if (
        fingerprint != anchor["fingerprint"]
        or fingerprint != hardened.stable_fingerprint(body)
    ):
        raise PermissionError(f"r4 archived fingerprint changed:{name}")
    return payload


def _archived_roots() -> dict[str, object]:
    return json.loads(hardened.canonical_json(_ARCHIVED_EVIDENCE_ANCHORS))


def _no_payload(value: Mapping[str, object]) -> None:
    if (
        value.get("payload_authority") != "none"
        or value.get("D_R_payload_accessed") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or value.get("gpu_accessed", False) is not False
    ):
        raise PermissionError("r4 recovery lineage is not payload-free")


def _identity() -> dict[str, str]:
    return {
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "attempt_id": ATTEMPT_ID,
        "unit_name": UNIT_NAME,
    }


def _common(value: Mapping[str, object]) -> bool:
    return (
        value.get("candidate") == CANDIDATE
        and value.get("stage_id") == STAGE_ID
        and value.get("attempt_id") == ATTEMPT_ID
        and value.get("execution_kind") == EXECUTION_KIND
        and value.get("runtime_spec_fingerprint")
        == _ARCHIVED_EVIDENCE_ANCHORS["runtime_spec"]["fingerprint"]
    )


def _private_directory(path: Path) -> dict[str, object]:
    binding = hardened._private_directory(path)
    if binding["owner_uid"] != AUTHORIZED_UID:
        raise PermissionError("r4 recovery directory owner changed")
    return binding


def _validate_inventory() -> None:
    if set(SCENARIO_ROOT.iterdir()) != {CONTROL_ROOT, RUNTIME_ROOT}:
        raise PermissionError("r4 scenario inventory changed")
    expected_control = {
        ORIGINAL_AUTHORIZATION_PATH,
        ORIGINAL_RUNTIME_SPEC_PATH,
        ORIGINAL_TERMINAL_PATH,
        ORIGINAL_REMOVAL_STATE_PATH,
    }
    recovery_paths = {
        RECOVERY_AUTHORIZATION_PATH,
        RECOVERY_INTENT_PATH,
        RECOVERY_TERMINAL_PATH,
    }
    actual_control = set(CONTROL_ROOT.iterdir())
    if (
        not expected_control.issubset(actual_control)
        or not actual_control.issubset(expected_control | recovery_paths)
    ):
        raise PermissionError("r4 control inventory changed")
    for path in actual_control & recovery_paths:
        hardened._read_regular_file_snapshot(
            path,
            expected_owner_uid=AUTHORIZED_UID,
            expected_mode=0o444,
        )
    expected_runtime = {
        ORIGINAL_LAUNCH_LEASE_PATH,
        ORIGINAL_PRECOMMIT_PATH,
        ORIGINAL_ATTEMPT_COMMIT_PATH,
        ORIGINAL_SIDECAR_PATH.parent,
        ORIGINAL_CONSUMED_FAILURE_PATH,
        HEARTBEAT_ROOT,
    }
    if set(RUNTIME_ROOT.iterdir()) != expected_runtime:
        raise PermissionError(
            "r4 reached claim/start-ack/child/GPU/payload/runtime-terminal"
        )
    if any(HEARTBEAT_ROOT.iterdir()):
        raise PermissionError("r4 heartbeat proves child execution")
    if set(INVOCATION_ROOT.iterdir()) != {ORIGINAL_SIDECAR_PATH}:
        raise PermissionError("r4 sidecar inventory changed")


def _sealed_original_chain() -> dict[str, object]:
    _assert_repository()
    authorization = _read_archived("authorization")
    spec = _read_archived("runtime_spec")
    lease = _read_archived("launch_lease")
    precommit = _read_archived("precommit")
    attempt = _read_archived("attempt_commit")
    sidecar = _read_archived("systemd_sidecar")
    consumed = _read_archived("consumed_start_failure")
    terminal = _read_archived("integration_terminal")
    removal = _read_archived("removal_state")
    _validate_inventory()
    _no_payload(authorization)
    _no_payload(terminal)
    _no_payload(removal)

    issued = _timestamp(authorization.get("issued_at_utc"), name="r4 issuance")
    expires = _timestamp(authorization.get("expires_at_utc"), name="r4 expiry")
    lease_time = _timestamp(lease.get("time_utc"), name="r4 lease")
    precommit_time = _timestamp(precommit.get("time_utc"), name="r4 precommit")
    attempt_time = _timestamp(attempt.get("time_utc"), name="r4 attempt")
    sidecar_time = _timestamp(sidecar.get("time_utc"), name="r4 sidecar")
    consumed_time = _timestamp(consumed.get("time_utc"), name="r4 consumed")
    terminal_time = _timestamp(
        terminal.get("created_at_utc"),
        name="r4 terminal",
    )
    if (
        expires <= issued
        or expires - issued > timedelta(seconds=300)
        or not (
            issued
            <= lease_time
            <= precommit_time
            <= attempt_time
            <= sidecar_time
            <= consumed_time
            <= terminal_time
            <= expires
        )
    ):
        raise PermissionError("r4 archived chronology changed")

    manager = authorization.get("manager_generation")
    boot_id = manager.get("boot_id") if isinstance(manager, Mapping) else None
    if (
        _BOOT_ID.fullmatch(str(boot_id)) is None
        or authorization.get("scenario_id") != SCENARIO_ID
        or authorization.get("identity") != _identity()
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis") != AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != AUTHORIZED_UID
        or authorization.get("integration_authorized") is not True
        or authorization.get("actual_r2_authorized") is not False
        or authorization.get("unit_realization_authorized") is not True
        or authorization.get("unit_removal_authorized") is not False
        or authorization.get("direct_start_authorized") is not False
        or authorization.get("enable_authorized") is not False
        or authorization.get("gpu_access_authorized") is not False
        or authorization.get("unit_directory")
        != str(EXPECTED_RUNTIME_UNIT_DIRECTORY)
        or authorization.get("rendered_fragment", {}).get("sha256")
        != EXPECTED_FRAGMENT_SHA256
        or hashlib.sha256(
            str(
                authorization.get("rendered_fragment", {}).get("utf8_text")
            ).encode("utf-8")
        ).hexdigest()
        != EXPECTED_FRAGMENT_SHA256
        or authorization.get("runtime_spec_binding", {}).get("path")
        != str(ORIGINAL_RUNTIME_SPEC_PATH)
        or authorization.get("runtime_spec_binding", {}).get("file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["runtime_spec"]["file_sha256"]
        or authorization.get("runtime_spec_binding", {}).get(
            "runtime_spec_fingerprint"
        )
        != _ARCHIVED_EVIDENCE_ANCHORS["runtime_spec"]["fingerprint"]
    ):
        raise PermissionError("r4 archived authorization changed")

    for name, path in {
        "scenario_root": SCENARIO_ROOT,
        "control_root": CONTROL_ROOT,
        "runtime_root": RUNTIME_ROOT,
    }.items():
        if authorization.get(name) != _private_directory(path):
            raise PermissionError(f"r4 archived directory changed:{name}")
    expected_control = {
        "dummy_artifact": str(RUNTIME_ROOT / "dummy-child.json"),
        "integration_receipt": str(CONTROL_ROOT / "integration-receipt.json"),
        "integration_terminal": str(ORIGINAL_TERMINAL_PATH),
        "removal_authorization": str(CONTROL_ROOT / "removal-authorization.json"),
        "removal_state": str(ORIGINAL_REMOVAL_STATE_PATH),
    }
    if authorization.get("control_artifacts") != expected_control:
        raise PermissionError("r4 control artifact contract changed")

    expected_executable_paths = {
        "python": "/usr/bin/python3.12",
        "supervisor": str(REPOSITORY / "tools/cure_lite_v24_runtime_supervisor.py"),
        "integration_tool": str(
            REPOSITORY / "tools/cure_lite_v24_user_systemd_integration.py"
        ),
        "realizer": str(
            REPOSITORY / "tools/cure_lite_v24_realize_systemd_unit.py"
        ),
        "dummy_child": str(REPOSITORY / "tools/cure_lite_v24_dummy_child.py"),
        "systemd_path": "/usr/bin/systemd-path",
        "systemd_analyze": "/usr/bin/systemd-analyze",
        "systemctl": "/usr/bin/systemctl",
    }
    archived = authorization.get("executable_bindings")
    if not isinstance(archived, Mapping) or set(archived) != set(
        expected_executable_paths
    ):
        raise PermissionError("r4 archived executable set changed")
    archived_bindings = {
        name: _validate_binding_shape(
            archived[name],
            name=name,
            expected_path=path,
        )
        for name, path in expected_executable_paths.items()
    }
    template = _validate_binding_shape(
        authorization.get("template_binding"),
        name="template",
        expected_path=str(
            REPOSITORY
            / "deploy/systemd/cure-lite-v24-supervisor-integration.service.template"
        ),
    )

    expected_artifacts = {
        "attempt_commit": str(ORIGINAL_ATTEMPT_COMMIT_PATH),
        "child_prespawn_phase_receipt": str(RUNTIME_ROOT / "child-prespawn.json"),
        "consumed_start_failure_receipt": str(
            ORIGINAL_CONSUMED_FAILURE_PATH
        ),
        "gpu_lease_release_receipt": str(RUNTIME_ROOT / "gpu-lease-release.json"),
        "heartbeat_dir": str(HEARTBEAT_ROOT),
        "launch_lease": str(ORIGINAL_LAUNCH_LEASE_PATH),
        "materialization_claim": str(RUNTIME_ROOT / "materialization-claim.json"),
        "precommit_phase_receipt": str(ORIGINAL_PRECOMMIT_PATH),
        "root": str(RUNTIME_ROOT),
        "runtime_attestation": str(RUNTIME_ROOT / "runtime-attestation.json"),
        "runtime_terminal": str(RUNTIME_ROOT / "runtime-terminal.json"),
        "start_ack_receipt": str(RUNTIME_ROOT / "start-ack.json"),
        "stderr_log": str(RUNTIME_ROOT / "stderr.log"),
        "stdout_log": str(RUNTIME_ROOT / "stdout.log"),
        "systemd_invocation_dir": str(INVOCATION_ROOT),
    }
    runtime = spec.get("runtime")
    systemd = runtime.get("systemd") if isinstance(runtime, Mapping) else None
    source_bindings = spec.get("source_bindings")
    if (
        not _common(spec)
        or spec.get("attempt_ordinal") != 0
        or spec.get("prior_attempt_count") != 0
        or spec.get("authorization") is not None
        or spec.get("environment") is not None
        or spec.get("scientific_preaccess") is not None
        or spec.get("artifacts") != expected_artifacts
        or not isinstance(runtime, Mapping)
        or runtime.get("automatic_retry_allowed") is not False
        or runtime.get("resume_allowed") is not False
        or runtime.get("launch_limit") != 1
        or runtime.get("restart") != "no"
        or not isinstance(systemd, Mapping)
        or systemd.get("unit_name") != UNIT_NAME
        or systemd.get("unit_fragment_file_sha256")
        != EXPECTED_FRAGMENT_SHA256
        or not isinstance(source_bindings, Mapping)
        or source_bindings.get("supervisor_file_sha256")
        != archived_bindings["supervisor"]["file_sha256"]
        or source_bindings.get("child_entry_file_sha256")
        != archived_bindings["dummy_child"]["file_sha256"]
    ):
        raise PermissionError("r4 archived runtime spec changed")

    immutable = systemd.get("immutable_shadow_fingerprint")
    if (
        not _common(lease)
        or lease.get("boot_id") != boot_id
        or lease.get("authorization_fingerprint") is not None
        or lease.get("gpu_exclusivity_claimed") is not False
        or lease.get("launch_limit") != 1
        or lease.get("automatic_retry_allowed") is not False
        or lease.get("resume_allowed") is not False
        or lease.get("lease_scope") != "attempt_dispatch_only"
        or not _common(precommit)
        or precommit.get("boot_id") != boot_id
        or precommit.get("phase") != "precommit"
        or precommit.get("launch_lease_fingerprint")
        != lease.get("launch_lease_fingerprint")
        or precommit.get("immutable_shadow_fingerprint") != immutable
        or precommit.get("runtime_environment_audit_valid") is not False
        or precommit.get("environment_audit_fingerprint") is not None
        or precommit.get("environment_inventory_fingerprint") is not None
        or precommit.get("gpu_lease_fingerprint") is not None
        or precommit.get("scientific_gate_passed") is not None
        or not _common(attempt)
        or attempt.get("boot_id") != boot_id
        or attempt.get("attempt_ordinal") != 0
        or attempt.get("prior_attempt_count") != 0
        or attempt.get("authorization_fingerprint") is not None
        or attempt.get("authorization_file_sha256") is not None
        or attempt.get("gpu_lease_fingerprint") is not None
        or attempt.get("gpu_lease_file_sha256") is not None
        or attempt.get("gpu_lease_device") is not None
        or attempt.get("gpu_lease_inode") is not None
        or attempt.get("planned_attempt_commit_fingerprint") is not None
        or attempt.get("launch_lease_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["launch_lease"]["file_sha256"]
        or attempt.get("launch_lease_fingerprint")
        != lease.get("launch_lease_fingerprint")
        or attempt.get("precommit_phase_receipt_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["precommit"]["file_sha256"]
        or attempt.get("precommit_phase_receipt_fingerprint")
        != precommit.get("phase_receipt_fingerprint")
        or attempt.get("immutable_systemd_shadow_fingerprint") != immutable
        or attempt.get("automatic_retry_allowed") is not False
        or attempt.get("resume_allowed") is not False
        or attempt.get("scientific_gate_passed") is not None
    ):
        raise PermissionError("r4 archived lease/commit chain changed")
    monotonic = [
        lease.get("monotonic_ns"),
        precommit.get("monotonic_ns"),
        attempt.get("monotonic_ns"),
        consumed.get("monotonic_ns"),
    ]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in monotonic
        )
        or monotonic != sorted(monotonic)
        or len(set(monotonic)) != len(monotonic)
    ):
        raise PermissionError("r4 monotonic chronology changed")

    if (
        sidecar.get("candidate") != CANDIDATE
        or sidecar.get("stage_id") != STAGE_ID
        or sidecar.get("attempt_id") != ATTEMPT_ID
        or sidecar.get("runtime_spec_fingerprint")
        != spec.get("runtime_spec_fingerprint")
        or sidecar.get("sidecar_systemd_invocation_id") != INVOCATION_ID
        or sidecar.get("attempt_commit_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["attempt_commit"]["file_sha256"]
        or sidecar.get("attempt_commit_fingerprint")
        != attempt.get("attempt_commit_fingerprint")
        or sidecar.get("attempt_commit_required") is not True
        or sidecar.get("attempt_commit_valid") is not True
        or sidecar.get("authorization_matches_commit") is not True
        or sidecar.get("current_runtime_closure_valid") is not True
        or sidecar.get("claim_valid") is not False
        or sidecar.get("claim_matches_invocation") is not False
        or sidecar.get("claim_systemd_invocation_id") is not None
        or sidecar.get("materialization_claim_file_sha256") is not None
        or sidecar.get("materialization_claim_fingerprint") is not None
        or sidecar.get("start_ack_valid") is not False
        or sidecar.get("start_ack_receipt_file_sha256") is not None
        or sidecar.get("start_ack_receipt_fingerprint") is not None
        or sidecar.get("child_prespawn_valid") is not False
        or sidecar.get("child_prespawn_phase_receipt_file_sha256") is not None
        or sidecar.get("child_prespawn_phase_receipt_fingerprint") is not None
        or sidecar.get("active_gpu_lease_valid") is not None
        or sidecar.get("active_gpu_lease_fingerprint") is not None
        or sidecar.get("gpu_lease_release_authorized") is not None
        or sidecar.get("gpu_lease_release_valid") is not None
        or sidecar.get("gpu_lease_release_receipt_fingerprint") is not None
        or sidecar.get("gpu_lease_tombstone_file_sha256") is not None
        or sidecar.get("audit_valid") is not False
        or sidecar.get("scientific_gate_passed") is not None
        or sidecar.get("scientific_decision")
        != "NOT_EVALUATED_BY_RUNTIME_SUPERVISOR"
        or sidecar.get("systemd_outcome", {}).get("category")
        != "SYSTEMD_EXEC_CONDITION"
        or sidecar.get("systemd_outcome", {}).get("service_result")
        != "exec-condition"
        or sidecar.get("systemd_outcome", {}).get("systemd_success") is not False
    ):
        raise PermissionError("r4 partial sidecar changed")

    empty_sha = hashlib.sha256(b"").hexdigest()
    if (
        not _common(consumed)
        or consumed.get("boot_id") != boot_id
        or consumed.get("attempt_consumed") is not True
        or consumed.get("category") != "BOUNDED_START_ACK_FAILED"
        or consumed.get("error_type") != "RuntimeError"
        or consumed.get("automatic_retry_allowed") is not False
        or consumed.get("resume_allowed") is not False
        or consumed.get("scientific_gate_passed") is not None
        or consumed.get("systemctl_return_code") != 0
        or consumed.get("systemctl_stdout_sha256") != empty_sha
        or consumed.get("systemctl_stderr_sha256") != empty_sha
    ):
        raise PermissionError("r4 consumed start failure changed")

    expected_actions = [
        "realize-static-fragment",
        "daemon-reload-after-realization",
    ]
    error_message = (
        "systemd terminal rejected integration:SYSTEMD_EXEC_CONDITION:"
        "8ef43c821e656681dc132c6d4572dca9026cf0ca40a372bd9052a9cd90014d08"
    )
    expected_evidence = {
        "audit_valid": False,
        "child_prespawn_valid": False,
        "claim_valid": False,
        "evidence_kind": "partial-systemd-terminal",
        "invocation_id": INVOCATION_ID,
        "start_ack_valid": False,
        "systemd_outcome": sidecar["systemd_outcome"],
        "systemd_terminal_file_sha256": _ARCHIVED_EVIDENCE_ANCHORS[
            "systemd_sidecar"
        ]["file_sha256"],
        "systemd_terminal_fingerprint": sidecar[
            "systemd_terminal_fingerprint"
        ],
    }
    if (
        terminal.get("scenario_id") != SCENARIO_ID
        or terminal.get("identity") != _identity()
        or terminal.get("authorization_fingerprint")
        != authorization.get("authorization_fingerprint")
        or terminal.get("runtime_spec_fingerprint")
        != spec.get("runtime_spec_fingerprint")
        or terminal.get("passed") is not False
        or terminal.get("completed_actions") != expected_actions
        or terminal.get("supervisor_evidence") != expected_evidence
        or terminal.get("error_type") != "RuntimeError"
        or terminal.get("error_message") != error_message
        or terminal.get("direct_systemctl_start_attempted") is not False
        or terminal.get("enable_attempted") is not False
        or terminal.get("remove_attempted") is not False
        or removal.get("scenario_id") != SCENARIO_ID
        or removal.get("unit_name") != UNIT_NAME
        or removal.get("removal_authorization_fingerprint") is not None
        or removal.get("passed") is not False
        or removal.get("remove_attempted") is not False
        or removal.get("fragment_absent") is not False
        or removal.get("not_found_state") is not None
        or removal.get("completed_actions") != expected_actions
        or removal.get("error_type") != "RuntimeError"
        or removal.get("error_message") != error_message
    ):
        raise PermissionError("r4 archived failure terminal changed")
    return {
        "authorization": authorization,
        "spec": spec,
        "lease": lease,
        "precommit": precommit,
        "attempt": attempt,
        "sidecar": sidecar,
        "consumed": consumed,
        "terminal": terminal,
        "removal": removal,
        "archived_roots": _archived_roots(),
        "archived_executable_bindings": archived_bindings,
        "archived_template_binding": template,
    }


def _validated_path_policy(
    archived: Mapping[str, object],
    observed: Mapping[str, object],
) -> dict[str, object]:
    archived_body = dict(archived)
    observed_body = dict(observed)
    archived_rows_value = archived_body.pop("ordered_unit_paths", None)
    observed_rows_value = observed_body.pop("ordered_unit_paths", None)
    if (
        archived_body != observed_body
        or not isinstance(archived_rows_value, list)
        or not isinstance(observed_rows_value, list)
    ):
        raise PermissionError("r4 unit path policy changed")
    archived_order = [
        str(row.get("path"))
        for row in archived_rows_value
        if isinstance(row, Mapping)
    ]
    observed_order = [
        str(row.get("path"))
        for row in observed_rows_value
        if isinstance(row, Mapping)
    ]
    archived_rows = {
        str(row["path"]): dict(row)
        for row in archived_rows_value
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    observed_rows = {
        str(row["path"]): dict(row)
        for row in observed_rows_value
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    if (
        len(archived_rows) != len(archived_rows_value)
        or len(observed_rows) != len(observed_rows_value)
        or set(archived_rows) != set(observed_rows)
        or archived_order != observed_order
    ):
        raise PermissionError("r4 ordered unit path set changed")
    allowed = "/run/user/1008/systemd/generator.late"
    for path, expected in archived_rows.items():
        current = observed_rows[path]
        if path != allowed:
            if current != expected:
                raise PermissionError(f"r4 unit path changed:{path}")
            continue
        expected_without_inode = dict(expected)
        current_without_inode = dict(current)
        old_inode = expected_without_inode.pop("inode", None)
        new_inode = current_without_inode.pop("inode", None)
        if (
            expected_without_inode != current_without_inode
            or isinstance(old_inode, bool)
            or not isinstance(old_inode, int)
            or old_inode <= 0
            or isinstance(new_inode, bool)
            or not isinstance(new_inode, int)
            or new_inode <= 0
        ):
            raise PermissionError("r4 generator.late identity changed")
    return json.loads(hardened.canonical_json(dict(observed)))


def _expected_fragment_identity() -> dict[str, object]:
    return {
        "fragment_path": str(EXPECTED_FRAGMENT_PATH),
        "fragment_sha256": EXPECTED_FRAGMENT_SHA256,
        "device": EXPECTED_FRAGMENT_DEVICE,
        "inode": EXPECTED_FRAGMENT_INODE,
        "owner_uid": EXPECTED_FRAGMENT_OWNER_UID,
        "mode": EXPECTED_FRAGMENT_MODE,
        "nlink": EXPECTED_FRAGMENT_NLINK,
        "size": EXPECTED_FRAGMENT_SIZE,
        "mtime_ns": EXPECTED_FRAGMENT_MTIME_NS,
        "ctime_ns": EXPECTED_FRAGMENT_CTIME_NS,
    }


def _fragment_identity(path: Path) -> dict[str, object]:
    if path != EXPECTED_FRAGMENT_PATH:
        raise PermissionError("r4 fragment path changed")
    data, current = hardened._read_regular_file_snapshot(
        path,
        expected_owner_uid=EXPECTED_FRAGMENT_OWNER_UID,
        expected_mode=EXPECTED_FRAGMENT_MODE,
    )
    observed = {
        "fragment_path": str(path),
        "fragment_sha256": hashlib.sha256(data).hexdigest(),
        "device": current.st_dev,
        "inode": current.st_ino,
        "owner_uid": current.st_uid,
        "mode": stat.S_IMODE(current.st_mode),
        "nlink": current.st_nlink,
        "size": current.st_size,
        "mtime_ns": current.st_mtime_ns,
        "ctime_ns": current.st_ctime_ns,
    }
    if observed != _expected_fragment_identity():
        raise PermissionError("r4 fragment identity changed")
    return observed


def _live_context(
    *,
    runner: object = integration.run_command,
    manager_reader: Callable[[], dict[str, object]] = (
        integration.collect_manager_generation
    ),
) -> dict[str, object]:
    chain = _sealed_original_chain()
    authorization = chain["authorization"]
    policy = realizer.freeze_user_unit_path_policy(
        UNIT_NAME,
        runner=runner,
        allowed_fragment=EXPECTED_FRAGMENT_PATH,
    )
    policy = _validated_path_policy(authorization["unit_path_policy"], policy)
    manager = manager_reader()
    integration._validate_manager_generation(manager)
    if manager != authorization["manager_generation"]:
        raise PermissionError("r4 manager generation changed")
    rendered = authorization["rendered_fragment"]
    plan = realizer.build_realization_plan(
        unit_name=UNIT_NAME,
        unit_directory=EXPECTED_RUNTIME_UNIT_DIRECTORY,
        fragment_text=str(rendered["utf8_text"]),
        expected_fragment_sha256=EXPECTED_FRAGMENT_SHA256,
        execute_authorized=True,
        removal_authorized=True,
    )
    state = realizer.query_unit_properties(UNIT_NAME, runner=runner)
    expected_state = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "static",
        "FragmentPath": str(EXPECTED_FRAGMENT_PATH),
        "DropInPaths": "",
        "Transient": "no",
        "Restart": "no",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
    }
    realizer.validate_realized_static_unit(plan, state)
    fragment = _fragment_identity(EXPECTED_FRAGMENT_PATH)
    if state != expected_state:
        raise PermissionError("r4 unit is not exact inactive static state")
    return {
        **chain,
        "manager_generation": manager,
        "unit_path_policy": policy,
        "fragment_identity": fragment,
        "inactive_static_state": state,
        "plan": plan,
    }


def _assert_current_bindings(
    *,
    recovery_tool: Mapping[str, object],
    required: Mapping[str, object],
) -> None:
    if (
        _current_recovery_tool_binding() != dict(recovery_tool)
        or _current_required_bindings() != dict(required)
    ):
        raise PermissionError("r4 recovery implementation changed")


def _revalidate_preunlink(
    *,
    plan: object,
    manager_generation: Mapping[str, object],
    unit_path_policy: Mapping[str, object],
    inactive_state: Mapping[str, str],
    runner: object,
    manager_reader: Callable[[], dict[str, object]],
) -> None:
    manager = manager_reader()
    integration._validate_manager_generation(manager)
    if manager != dict(manager_generation):
        raise PermissionError("r4 manager changed after intent")
    policy = realizer.freeze_user_unit_path_policy(
        UNIT_NAME,
        runner=runner,
        allowed_fragment=EXPECTED_FRAGMENT_PATH,
    )
    if policy != dict(unit_path_policy):
        raise PermissionError("r4 unit path changed after intent")
    state = realizer.query_unit_properties(UNIT_NAME, runner=runner)
    realizer.validate_realized_static_unit(plan, state)
    if state != dict(inactive_state):
        raise PermissionError("r4 unit state changed after intent")


def _revalidate_manager(
    manager_generation: Mapping[str, object],
    *,
    manager_reader: Callable[[], dict[str, object]],
) -> None:
    manager = manager_reader()
    integration._validate_manager_generation(manager)
    if manager != dict(manager_generation):
        raise PermissionError("r4 manager changed before daemon-reload")


def _observe_fragment_absent(path: Path) -> bool:
    return not os.path.lexists(path)


def _remove_exact_fragment(
    plan: object,
    *,
    expected_identity: Mapping[str, object],
    authorization: Mapping[str, object],
    intent: Mapping[str, object],
    recovery_tool: Mapping[str, object],
    required_bindings: Mapping[str, object],
    manager_generation: Mapping[str, object],
    unit_path_policy: Mapping[str, object],
    inactive_state: Mapping[str, str],
    runner: object,
    manager_reader: Callable[[], dict[str, object]],
    on_action_started: Callable[[str], None],
) -> None:
    realizer.validate_integration_unit_name(plan.unit_name)
    if (
        set(expected_identity) != _FRAGMENT_IDENTITY_KEYS
        or dict(expected_identity) != _expected_fragment_identity()
        or plan.unit_name != UNIT_NAME
        or plan.unit_directory != EXPECTED_RUNTIME_UNIT_DIRECTORY
        or plan.fragment_path != EXPECTED_FRAGMENT_PATH
        or plan.fragment_sha256 != EXPECTED_FRAGMENT_SHA256
        or plan.owner_uid != EXPECTED_FRAGMENT_OWNER_UID
        or plan.execute_authorized is not True
        or plan.removal_authorized is not True
    ):
        raise PermissionError("r4 removal identity is not exact")
    _assert_current_bindings(
        recovery_tool=recovery_tool,
        required=required_bindings,
    )
    directory_fd = realizer._open_verified_directory(
        plan.unit_directory,
        owner_uid=EXPECTED_FRAGMENT_OWNER_UID,
    )
    fragment_fd: int | None = None
    try:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("r4 recovery requires O_NOFOLLOW")
        fragment_fd = os.open(
            plan.unit_name,
            os.O_RDONLY | os.O_CLOEXEC | nofollow,
            dir_fd=directory_fd,
        )

        def _check(ordinal: int) -> None:
            directory_opened = os.fstat(directory_fd)
            directory_linked = os.stat(
                plan.unit_directory,
                follow_symlinks=False,
            )

            def _directory_core(value: os.stat_result) -> tuple[int, ...]:
                return (
                    value.st_dev,
                    value.st_ino,
                    value.st_uid,
                    value.st_nlink,
                    stat.S_IMODE(value.st_mode),
                )

            opened = os.fstat(fragment_fd)
            linked = os.stat(
                plan.unit_name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )

            def _exact(value: os.stat_result) -> bool:
                return (
                    stat.S_ISREG(value.st_mode)
                    and value.st_dev == EXPECTED_FRAGMENT_DEVICE
                    and value.st_ino == EXPECTED_FRAGMENT_INODE
                    and value.st_uid == EXPECTED_FRAGMENT_OWNER_UID
                    and stat.S_IMODE(value.st_mode) == EXPECTED_FRAGMENT_MODE
                    and value.st_nlink == EXPECTED_FRAGMENT_NLINK
                    and value.st_size == EXPECTED_FRAGMENT_SIZE
                    and value.st_mtime_ns == EXPECTED_FRAGMENT_MTIME_NS
                    and value.st_ctime_ns == EXPECTED_FRAGMENT_CTIME_NS
                )

            if (
                not stat.S_ISDIR(directory_opened.st_mode)
                or not stat.S_ISDIR(directory_linked.st_mode)
                or _directory_core(directory_opened)
                != _directory_core(directory_linked)
                or not _exact(opened)
                or not _exact(linked)
                or opened.st_dev != linked.st_dev
                or opened.st_ino != linked.st_ino
                or realizer._fd_sha256(fragment_fd)
                != EXPECTED_FRAGMENT_SHA256
            ):
                raise PermissionError(f"r4 fragment changed at check {ordinal}")

        _check(1)
        _check(2)
        _assert_current_bindings(
            recovery_tool=recovery_tool,
            required=required_bindings,
        )
        _revalidate_preunlink(
            plan=plan,
            manager_generation=manager_generation,
            unit_path_policy=unit_path_policy,
            inactive_state=inactive_state,
            runner=runner,
            manager_reader=manager_reader,
        )
        _assert_current_bindings(
            recovery_tool=recovery_tool,
            required=required_bindings,
        )
        _check(3)
        action_started = datetime.now(timezone.utc)
        issued = _timestamp(
            authorization.get("issued_at_utc"),
            name="r4 recovery issuance",
        )
        intent_time = _timestamp(
            intent.get("created_at_utc"),
            name="r4 recovery intent",
        )
        expires = _timestamp(
            authorization.get("expires_at_utc"),
            name="r4 recovery expiry",
        )
        if not issued <= intent_time <= action_started <= expires:
            raise PermissionError("r4 recovery expired before unlink")
        on_action_started(action_started.isoformat().replace("+00:00", "Z"))
        _check(4)
        if datetime.now(timezone.utc) > expires:
            raise PermissionError("r4 recovery expired at unlink")
        os.unlink(plan.unit_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        if fragment_fd is not None:
            os.close(fragment_fd)
        os.close(directory_fd)


def create_recovery_authorization(
    *,
    validity_seconds: int = 300,
    runner: object = integration.run_command,
    manager_reader: Callable[[], dict[str, object]] = (
        integration.collect_manager_generation
    ),
) -> dict[str, object]:
    _assert_repository()
    if (
        isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("r4 recovery validity must be in [1,300]")
    if any(
        os.path.lexists(path)
        for path in (
            RECOVERY_AUTHORIZATION_PATH,
            RECOVERY_INTENT_PATH,
            RECOVERY_TERMINAL_PATH,
        )
    ):
        raise FileExistsError("r4 recovery identity is consumed")
    tool_before = _current_recovery_tool_binding()
    required_before = _current_required_bindings()
    context = _live_context(runner=runner, manager_reader=manager_reader)
    tool_after = _current_recovery_tool_binding()
    required_after = _current_required_bindings()
    if tool_before != tool_after or required_before != required_after:
        raise PermissionError("r4 recovery changed during authorization")
    issued = datetime.now(timezone.utc)
    action = {
        "ordinal": 0,
        "action": "remove-exact-r4-runtime-static-fragment",
        "unit_name": UNIT_NAME,
        "fragment_path": str(EXPECTED_FRAGMENT_PATH),
        "then": ["daemon-reload", "verify-not-found"],
    }
    return hardened._write_sealed(
        RECOVERY_AUTHORIZATION_PATH,
        {
            "schema_version": AUTHORIZATION_SCHEMA,
            "scenario_id": SCENARIO_ID,
            "unit_name": UNIT_NAME,
            "instruction_id": INSTRUCTION_ID,
            "authorization_basis": AUTHORIZATION_BASIS,
            "authorized_uid": AUTHORIZED_UID,
            "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
            "expires_at_utc": (
                issued + timedelta(seconds=validity_seconds)
            ).isoformat().replace("+00:00", "Z"),
            "archived_roots": context["archived_roots"],
            "archived_executable_bindings": context[
                "archived_executable_bindings"
            ],
            "current_recovery_tool_binding": tool_after,
            "current_required_executable_bindings": required_after,
            "manager_generation": context["manager_generation"],
            "unit_path_policy": context["unit_path_policy"],
            "fragment_identity": context["fragment_identity"],
            "inactive_static_state": context["inactive_static_state"],
            "authorized_action": action,
            "remove_authorized": True,
            "daemon_reload_authorized": True,
            "not_found_verification_authorized": True,
            "start_authorized": False,
            "stop_authorized": False,
            "enable_authorized": False,
            "automatic_retry_authorized": False,
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_accessed": False,
        },
        fingerprint_field="recovery_authorization_fingerprint",
    )


def _read_recovery_snapshot(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str,
) -> tuple[dict[str, object], str, tuple[int, ...]]:
    return hardened._read_recovery_sealed_snapshot(
        path,
        fingerprint_field=fingerprint_field,
        schema=schema,
    )


def _read_recovery(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str,
) -> dict[str, object]:
    payload, _digest, _identity_value = _read_recovery_snapshot(
        path,
        fingerprint_field=fingerprint_field,
        schema=schema,
    )
    return payload


def load_recovery_authorization(
    *,
    require_fresh: bool,
    runner: object = integration.run_command,
    manager_reader: Callable[[], dict[str, object]] = (
        integration.collect_manager_generation
    ),
) -> tuple[dict[str, object], dict[str, object]]:
    authorization, digest, stat_identity = _read_recovery_snapshot(
        RECOVERY_AUTHORIZATION_PATH,
        fingerprint_field="recovery_authorization_fingerprint",
        schema=AUTHORIZATION_SCHEMA,
    )
    if set(authorization) != _AUTHORIZATION_KEYS:
        raise PermissionError("r4 recovery authorization keys changed")
    _no_payload(authorization)
    issued = _timestamp(authorization.get("issued_at_utc"), name="r4 issuance")
    expires = _timestamp(authorization.get("expires_at_utc"), name="r4 expiry")
    now = datetime.now(timezone.utc)
    expected_action = {
        "ordinal": 0,
        "action": "remove-exact-r4-runtime-static-fragment",
        "unit_name": UNIT_NAME,
        "fragment_path": str(EXPECTED_FRAGMENT_PATH),
        "then": ["daemon-reload", "verify-not-found"],
    }
    if (
        authorization.get("scenario_id") != SCENARIO_ID
        or authorization.get("unit_name") != UNIT_NAME
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis") != AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != AUTHORIZED_UID
        or expires <= issued
        or expires - issued > timedelta(seconds=300)
        or issued > now
        or (require_fresh and now > expires)
        or authorization.get("archived_roots") != _archived_roots()
        or authorization.get("authorized_action") != expected_action
        or authorization.get("remove_authorized") is not True
        or authorization.get("daemon_reload_authorized") is not True
        or authorization.get("not_found_verification_authorized") is not True
        or authorization.get("start_authorized") is not False
        or authorization.get("stop_authorized") is not False
        or authorization.get("enable_authorized") is not False
        or authorization.get("automatic_retry_authorized") is not False
    ):
        raise PermissionError("r4 recovery authorization changed or expired")
    tool = authorization.get("current_recovery_tool_binding")
    required = authorization.get("current_required_executable_bindings")
    if not isinstance(tool, Mapping) or not isinstance(required, Mapping):
        raise PermissionError("r4 recovery current bindings malformed")
    _assert_current_bindings(recovery_tool=tool, required=required)
    context = _live_context(runner=runner, manager_reader=manager_reader)
    _assert_current_bindings(recovery_tool=tool, required=required)
    if (
        authorization.get("archived_executable_bindings")
        != context["archived_executable_bindings"]
        or authorization.get("manager_generation")
        != context["manager_generation"]
        or authorization.get("unit_path_policy")
        != context["unit_path_policy"]
        or authorization.get("fragment_identity")
        != context["fragment_identity"]
        or authorization.get("inactive_static_state")
        != context["inactive_static_state"]
    ):
        raise PermissionError("r4 recovery live closure changed")
    final_auth, final_digest, final_identity = _read_recovery_snapshot(
        RECOVERY_AUTHORIZATION_PATH,
        fingerprint_field="recovery_authorization_fingerprint",
        schema=AUTHORIZATION_SCHEMA,
    )
    if (
        final_auth != authorization
        or final_digest != digest
        or final_identity != stat_identity
        or (require_fresh and datetime.now(timezone.utc) > expires)
    ):
        raise PermissionError("r4 authorization changed during live closure")
    context = dict(context)
    context["recovery_authorization_file_sha256"] = final_digest
    context["recovery_authorization_stat_identity"] = final_identity
    return authorization, context


def execute_recovery(
    *,
    execute: bool,
    timeout_seconds: float = 10.0,
    runner: object = integration.run_command,
    manager_reader: Callable[[], dict[str, object]] = (
        integration.collect_manager_generation
    ),
) -> dict[str, object]:
    if not execute:
        raise PermissionError("explicit r4 recovery execution required")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise ValueError("r4 recovery timeout must be finite positive")
    if os.path.lexists(RECOVERY_INTENT_PATH) or os.path.lexists(
        RECOVERY_TERMINAL_PATH
    ):
        raise FileExistsError("r4 recovery execution identity consumed")
    authorization, context = load_recovery_authorization(
        require_fresh=True,
        runner=runner,
        manager_reader=manager_reader,
    )
    intent_auth, intent_digest, intent_identity = _read_recovery_snapshot(
        RECOVERY_AUTHORIZATION_PATH,
        fingerprint_field="recovery_authorization_fingerprint",
        schema=AUTHORIZATION_SCHEMA,
    )
    if (
        intent_auth != authorization
        or intent_digest != context["recovery_authorization_file_sha256"]
        or intent_identity != context["recovery_authorization_stat_identity"]
    ):
        raise PermissionError("r4 authorization changed before intent")
    intent = hardened._write_sealed(
        RECOVERY_INTENT_PATH,
        {
            "schema_version": INTENT_SCHEMA,
            "created_at_utc": _utc_now(),
            "scenario_id": SCENARIO_ID,
            "unit_name": UNIT_NAME,
            "recovery_authorization_path": str(RECOVERY_AUTHORIZATION_PATH),
            "recovery_authorization_file_sha256": intent_digest,
            "recovery_authorization_fingerprint": authorization[
                "recovery_authorization_fingerprint"
            ],
            "fragment_identity": context["fragment_identity"],
            "inactive_static_state": context["inactive_static_state"],
            "authorized_action": authorization["authorized_action"],
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_accessed": False,
        },
        fingerprint_field="recovery_intent_fingerprint",
    )
    if set(intent) != _INTENT_KEYS:
        raise RuntimeError("r4 recovery intent schema changed")
    completed: list[str] = []
    post_state: dict[str, str] | None = None
    action_started: str | None = None
    error: BaseException | None = None

    def _record(value: str) -> None:
        nonlocal action_started
        if action_started is not None:
            raise RuntimeError("r4 recovery action start repeated")
        _timestamp(value, name="r4 action start")
        action_started = value

    try:
        _remove_exact_fragment(
            context["plan"],
            expected_identity=context["fragment_identity"],
            authorization=authorization,
            intent=intent,
            recovery_tool=authorization["current_recovery_tool_binding"],
            required_bindings=authorization[
                "current_required_executable_bindings"
            ],
            manager_generation=authorization["manager_generation"],
            unit_path_policy=authorization["unit_path_policy"],
            inactive_state=authorization["inactive_static_state"],
            runner=runner,
            manager_reader=manager_reader,
            on_action_started=_record,
        )
        completed.append("remove-exact-r4-runtime-static-fragment")
        _revalidate_manager(
            authorization["manager_generation"],
            manager_reader=manager_reader,
        )
        realizer.daemon_reload(execute=True, runner=runner)
        completed.append("daemon-reload")
        post_state = realizer.wait_until_unit_not_found(
            UNIT_NAME,
            query=lambda unit: realizer.query_unit_properties(
                unit,
                runner=runner,
            ),
            timeout_seconds=float(timeout_seconds),
            poll_seconds=0.01,
        )
        completed.append("verify-not-found")
    except BaseException as caught:
        error = caught
    fragment_absent = False
    try:
        fragment_absent = _observe_fragment_absent(context["plan"].fragment_path)
    except BaseException as caught:
        if error is None:
            error = caught
    terminal = hardened._write_sealed(
        RECOVERY_TERMINAL_PATH,
        {
            "schema_version": TERMINAL_SCHEMA,
            "created_at_utc": _utc_now(),
            "scenario_id": SCENARIO_ID,
            "unit_name": UNIT_NAME,
            "recovery_authorization_fingerprint": authorization[
                "recovery_authorization_fingerprint"
            ],
            "recovery_intent_fingerprint": intent[
                "recovery_intent_fingerprint"
            ],
            "action_started_at_utc": action_started,
            "completed_actions": completed,
            "fragment_absent": fragment_absent,
            "post_removal_unit_state": post_state,
            "passed": (
                error is None
                and fragment_absent
                and completed
                == [
                    "remove-exact-r4-runtime-static-fragment",
                    "daemon-reload",
                    "verify-not-found",
                ]
            ),
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error) if error is not None else None,
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_accessed": False,
        },
        fingerprint_field="recovery_terminal_fingerprint",
    )
    if set(terminal) != _TERMINAL_KEYS:
        raise RuntimeError("r4 recovery terminal schema changed")
    intent_time = _timestamp(intent["created_at_utc"], name="r4 intent")
    terminal_time = _timestamp(terminal["created_at_utc"], name="r4 terminal")
    if action_started is None:
        if terminal_time < intent_time:
            raise RuntimeError("r4 failure chronology changed")
    elif not (
        intent_time
        <= _timestamp(action_started, name="r4 action start")
        <= terminal_time
    ):
        raise RuntimeError("r4 action chronology changed")
    if error is not None:
        raise error
    if terminal["passed"] is not True:
        raise RuntimeError("r4 recovery did not reach PASS")
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--validity-seconds", type=int, default=300)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--execute-authorized-removal", action="store_true")
    apply.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "authorize":
        result = create_recovery_authorization(
            validity_seconds=arguments.validity_seconds
        )
        print(
            json.dumps(
                {
                    "path": str(RECOVERY_AUTHORIZATION_PATH),
                    "recovery_authorization_fingerprint": result[
                        "recovery_authorization_fingerprint"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    result = execute_recovery(
        execute=arguments.execute_authorized_removal,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "path": str(RECOVERY_TERMINAL_PATH),
                "passed": result["passed"],
                "recovery_terminal_fingerprint": result[
                    "recovery_terminal_fingerprint"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
