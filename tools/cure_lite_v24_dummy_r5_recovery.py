#!/usr/bin/env python3
"""Exact one-shot recovery for the completed CURE-Lite v24 dummy r5 run.

The archived r5 execution passed, but its original cleanup policy rejected a
post-terminal unit-path generation change before attempting removal.  This
tool accepts only that one archived r5 identity and exposes only:

    remove-exact-r5-fragment -> daemon-reload -> verify-not-found

It grants no start, stop, enable, retry, payload, dataset, or GPU authority.
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
    from tools import cure_lite_v24_dummy_r4_recovery as hardened
except ModuleNotFoundError:
    _HARDENED_PATH = Path(__file__).with_name(
        "cure_lite_v24_dummy_r4_recovery.py"
    )
    _HARDENED_SPEC = importlib.util.spec_from_file_location(
        "cure_lite_v24_dummy_r4_recovery_for_r5",
        _HARDENED_PATH,
    )
    if _HARDENED_SPEC is None or _HARDENED_SPEC.loader is None:
        raise RuntimeError("cannot load frozen r4 recovery template")
    hardened = importlib.util.module_from_spec(_HARDENED_SPEC)
    sys.modules[_HARDENED_SPEC.name] = hardened
    _HARDENED_SPEC.loader.exec_module(hardened)


io = hardened.hardened
integration = hardened.integration
realizer = hardened.realizer

REPOSITORY = Path("/home/md0/ly/cure_lite")
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
)
SCENARIO_ROOT = EVIDENCE_ROOT / "supervisor_v2_systemd_integration_r5"
CONTROL_ROOT = SCENARIO_ROOT / "control"
RUNTIME_ROOT = SCENARIO_ROOT / "runtime"
HEARTBEAT_ROOT = RUNTIME_ROOT / "heartbeat"
INVOCATION_ROOT = RUNTIME_ROOT / "systemd-invocations"

SCENARIO_ID = "supervisor-v2-dummy-r5-202607300630d15c"
UNIT_NAME = (
    "cure-lite-v24-supervisor-integration-"
    "supervisor-v2-dummy-r5-202607300630d15c.service"
)
STAGE_ID = f"systemd_integration_dummy_{SCENARIO_ID}"
ATTEMPT_ID = f"systemd_integration_dummy_attempt_{SCENARIO_ID}"
CANDIDATE = "systemd-integration-dummy"
EXECUTION_KIND = "systemd_integration_dummy"
INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后运行"
AUTHORIZED_UID = 1008
INVOCATION_ID = "01385dfdbc204761bba11016c14916a3"

EXPECTED_RUNTIME_UNIT_DIRECTORY = Path("/run/user/1008/systemd/user")
EXPECTED_FRAGMENT_PATH = EXPECTED_RUNTIME_UNIT_DIRECTORY / UNIT_NAME
EXPECTED_FRAGMENT_SHA256 = (
    "455419092a210c071e3a8793a07ff659b10e0a272599e2be1dbfbb18ff5a5558"
)
EXPECTED_FRAGMENT_DEVICE = 54
EXPECTED_FRAGMENT_INODE = 38689
EXPECTED_FRAGMENT_OWNER_UID = 1008
EXPECTED_FRAGMENT_MODE = 0o600
EXPECTED_FRAGMENT_NLINK = 1
EXPECTED_FRAGMENT_SIZE = 1883
EXPECTED_FRAGMENT_MTIME_NS = 1785364655973626201
EXPECTED_FRAGMENT_CTIME_NS = 1785364655973626201

ORIGINAL_AUTHORIZATION_PATH = CONTROL_ROOT / "authorization.json"
ORIGINAL_RUNTIME_SPEC_PATH = CONTROL_ROOT / "runtime-spec.json"
ORIGINAL_INTEGRATION_TERMINAL_PATH = (
    CONTROL_ROOT / "integration-terminal.json"
)
ORIGINAL_REMOVAL_STATE_PATH = CONTROL_ROOT / "removal-state.json"
ORIGINAL_LAUNCH_LEASE_PATH = RUNTIME_ROOT / "launch-lease.json"
ORIGINAL_PRECOMMIT_PATH = RUNTIME_ROOT / "precommit-phase.json"
ORIGINAL_ATTEMPT_COMMIT_PATH = RUNTIME_ROOT / "attempt-commit.json"
ORIGINAL_CLAIM_PATH = RUNTIME_ROOT / "materialization-claim.json"
ORIGINAL_START_ACK_PATH = RUNTIME_ROOT / "start-ack.json"
ORIGINAL_CHILD_PRESPAWN_PATH = RUNTIME_ROOT / "child-prespawn.json"
ORIGINAL_RUNTIME_TERMINAL_PATH = RUNTIME_ROOT / "runtime-terminal.json"
ORIGINAL_DUMMY_PATH = RUNTIME_ROOT / "dummy-child.json"
ORIGINAL_STDOUT_PATH = RUNTIME_ROOT / "stdout.log"
ORIGINAL_STDERR_PATH = RUNTIME_ROOT / "stderr.log"
ORIGINAL_SIDECAR_PATH = INVOCATION_ROOT / f"{INVOCATION_ID}.json"
ORIGINAL_HEARTBEAT_PATHS = tuple(
    HEARTBEAT_ROOT / f"{sequence:012d}.json" for sequence in range(3)
)

RECOVERY_AUTHORIZATION_PATH = (
    CONTROL_ROOT / "r5-exact-recovery-authorization.json"
)
RECOVERY_INTENT_PATH = CONTROL_ROOT / "r5-exact-recovery-intent.json"
RECOVERY_TERMINAL_PATH = CONTROL_ROOT / "r5-exact-recovery-terminal.json"

AUTHORIZATION_SCHEMA = "cure-lite-v24-dummy-r5-exact-recovery-authorization-v1"
INTENT_SCHEMA = "cure-lite-v24-dummy-r5-exact-recovery-intent-v1"
TERMINAL_SCHEMA = "cure-lite-v24-dummy-r5-exact-recovery-terminal-v1"
FROZEN_R4_RECOVERY_SHA256 = (
    "2ed22b91e1dec93c904b7f66654f45eacf7bfa06207efe3bff5d05da490ab98b"
)
FROZEN_R3_IO_SHA256 = (
    "b3d7fd5b98f70db98ec637dbbed3bc4f428a9290cb32cdc39e6fe46f0cc0a7f4"
)

EXPECTED_MANAGER_GENERATION = {
    "boot_id": "fa197e27-4b03-489a-85d7-f27be6bfed7d",
    "endpoint": {
        "bus_device": 54,
        "bus_inode": 61,
        "bus_path": "/run/user/1008/bus",
        "runtime_device": 54,
        "runtime_directory": "/run/user/1008",
        "runtime_inode": 1,
        "uid": 1008,
    },
    "identity": {
        "control_group": (
            "/user.slice/user-1008.slice/user@1008.service/init.scope"
        ),
        "pid": 1808,
        "starttime_ticks": 769,
        "uid": 1008,
    },
}

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


def _json_anchor(
    path: Path,
    file_sha256: str,
    fingerprint_field: str,
    fingerprint: str,
    schema_version: str,
) -> dict[str, object]:
    return {
        "path": str(path),
        "file_sha256": file_sha256,
        "fingerprint_field": fingerprint_field,
        "fingerprint": fingerprint,
        "schema_version": schema_version,
        "mode": 0o444,
        "kind": "canonical-json",
    }


def _raw_anchor(path: Path, file_sha256: str) -> dict[str, object]:
    return {
        "path": str(path),
        "file_sha256": file_sha256,
        "fingerprint_field": None,
        "fingerprint": None,
        "schema_version": None,
        "mode": 0o444,
        "kind": "raw",
    }


_ARCHIVED_EVIDENCE_ANCHORS: dict[str, dict[str, object]] = {
    "authorization": _json_anchor(
        ORIGINAL_AUTHORIZATION_PATH,
        "4b299e67735962dc35c04ecc10ba81e741449a463c17a9c8ec50285615b8666f",
        "authorization_fingerprint",
        "356a941fe5ce56d138a442164973fcbf39f1668d8ce4b87253e7d6ae7832af83",
        "cure-lite-v24-supervisor-v2-systemd-integration-authorization-v2",
    ),
    "runtime_spec": _json_anchor(
        ORIGINAL_RUNTIME_SPEC_PATH,
        "e99fdc7610cd9791aaba5bcff653046a430b2d6698ab420bda24ad488db0f73b",
        "runtime_spec_fingerprint",
        "8b9f07d26a8b3f799fafd2c8a39351a64bbf985c791e97e3dab51aa825ad3e35",
        "cure-lite-v24-dr-runtime-supervisor-spec-v2",
    ),
    "launch_lease": _json_anchor(
        ORIGINAL_LAUNCH_LEASE_PATH,
        "0bc7461b7c351fb0e0ad0754a45cb3fbe3c0323acd82e9c2c1438acc1daa8c46",
        "launch_lease_fingerprint",
        "c9dca67f6102b16b463c1fabfc8ca0f514aadd67ef8155671a5a120c18a7da06",
        "cure-lite-v24-dr-launch-lease-v1",
    ),
    "precommit": _json_anchor(
        ORIGINAL_PRECOMMIT_PATH,
        "d86aca87e8701290a93d4e509e32b18c954cb76d1f021ab971c95f80ed958bb7",
        "phase_receipt_fingerprint",
        "80aff33a41deef930ab69dffde1cf2c03b9410dd4422a0d32b2dc86de445d6d8",
        "cure-lite-v24-dr-runtime-phase-receipt-v1",
    ),
    "attempt_commit": _json_anchor(
        ORIGINAL_ATTEMPT_COMMIT_PATH,
        "032b42ea061cf1a44c6f8bd6747c51c3b7a03447fec54a78771583d457cf0fbe",
        "attempt_commit_fingerprint",
        "e577893103ca34b69e1fbcd0049a3b03d252ffb0e53d471624bed0f5871edebb",
        "cure-lite-v24-dr-attempt-commit-v2",
    ),
    "materialization_claim": _json_anchor(
        ORIGINAL_CLAIM_PATH,
        "534825b3197d38b0b15111efa9ac6a45ed0371e1f1d11516059815208ac3098e",
        "materialization_claim_fingerprint",
        "317223cf5052d6bb60758ffc0773acdcf18243c99148a0265ec685fb64a0d965",
        "cure-lite-v24-dr-materialization-claim-v2",
    ),
    "start_ack": _json_anchor(
        ORIGINAL_START_ACK_PATH,
        "19a599fb8869cb992923289e6cfef4d7c2f34ccb79f5e223016cf701a24fdf2c",
        "phase_receipt_fingerprint",
        "2f8f10a03bcfc1b9156de5ebe2a76f13e25911620016385e520a743df0ee03d3",
        "cure-lite-v24-dr-runtime-phase-receipt-v1",
    ),
    "child_prespawn": _json_anchor(
        ORIGINAL_CHILD_PRESPAWN_PATH,
        "60347268dd7aa566880dda243b5ad81713024a4ffac1bfed654809f62f52c11c",
        "phase_receipt_fingerprint",
        "93568c319f68e71a532846c6a6c3abe78e47bee99465784b027172f3c79382b1",
        "cure-lite-v24-dr-runtime-phase-receipt-v1",
    ),
    "heartbeat_0": _json_anchor(
        ORIGINAL_HEARTBEAT_PATHS[0],
        "32962b374bc7fba15a7ece542b99aef36d391fe0f666eea292902478346be294",
        "event_fingerprint",
        "48de1dc1da75656413eb3d403481a90542e3bc89b2e42bffa5df6bac9b23e9b5",
        "cure-lite-v24-dr-runtime-heartbeat-v1",
    ),
    "heartbeat_1": _json_anchor(
        ORIGINAL_HEARTBEAT_PATHS[1],
        "b88cbf7968a147a0a784ad0d0c64088a0c708427a56101f5bf85391064013a93",
        "event_fingerprint",
        "059679a3caf922853d5f024e81b10d8a2cc772cbb83e9d5baeee47f5ffe85623",
        "cure-lite-v24-dr-runtime-heartbeat-v1",
    ),
    "heartbeat_2": _json_anchor(
        ORIGINAL_HEARTBEAT_PATHS[2],
        "ed4dbbcb8b984c98cf9cd0f9617551d27b48de9c40674101db759f02364e9dcd",
        "event_fingerprint",
        "870f3a6f02a0a1a00fa211a46720c47e92839f704af144cc4794f738e476af68",
        "cure-lite-v24-dr-runtime-heartbeat-v1",
    ),
    "runtime_terminal": _json_anchor(
        ORIGINAL_RUNTIME_TERMINAL_PATH,
        "f9fcb1097de3954fb0f07cb778810b8e234a01075d0aec96d564f09ca3215475",
        "runtime_terminal_fingerprint",
        "ff8071066b68ec8733e02e0bc523f75ff8ecd8072e240752fcb1ee1973d40d88",
        "cure-lite-v24-dr-runtime-terminal-v1",
    ),
    "systemd_sidecar": _json_anchor(
        ORIGINAL_SIDECAR_PATH,
        "17aaf28d9f1b63fece0b61de1d399716d2f33b552212c36aae84f5a262db7348",
        "systemd_terminal_fingerprint",
        "ceb3cc0294eca53c7287baa1a524c80c9086c9ae0f4fe4534c54fddec0520594",
        "cure-lite-v24-dr-systemd-terminal-v1",
    ),
    "dummy_artifact": _json_anchor(
        ORIGINAL_DUMMY_PATH,
        "73a25ee8213a2f40a8c34a14fa26eddd6b30e856b606919abb278360b789b9d1",
        "dummy_artifact_fingerprint",
        "5937a41d2cebf73a58fc895f464cc368c1854668fe850a28f475b7a4c417cbe2",
        "cure-lite-v24-user-systemd-dummy-child-v1",
    ),
    "stdout_log": _raw_anchor(
        ORIGINAL_STDOUT_PATH,
        "d52a241a7082efe38a5bc6cb7b647d7c5bc49fb1cdbfda0342acc1077ac1b20c",
    ),
    "stderr_log": _raw_anchor(
        ORIGINAL_STDERR_PATH,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "integration_terminal": _json_anchor(
        ORIGINAL_INTEGRATION_TERMINAL_PATH,
        "3d606b9277f9921a65df4c9469dc3a991b376c9ecfee89ca35424d0029c44cd6",
        "integration_terminal_fingerprint",
        "c8a83b36f4d6b8e9b5fd2f27197f74b11dcd669d7e28c0552568c656cbd2e4d0",
        "cure-lite-v24-supervisor-v2-systemd-integration-terminal-v1",
    ),
    "removal_state": _json_anchor(
        ORIGINAL_REMOVAL_STATE_PATH,
        "4a744432b298fff1de41a55e2f4e4f6d566ad65e511b3b06954ff4919b69a122",
        "removal_state_fingerprint",
        "52ff8242e80c28cf4e2f04a60d7731a1ba8bda9ed4f56d09d74446d4942f9f35",
        "cure-lite-v24-supervisor-v2-integration-removal-state-v1",
    ),
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
    return io._timestamp(value, name=name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _assert_repository() -> None:
    if (
        Path(__file__).resolve().parents[1] != REPOSITORY
        or os.getuid() != AUTHORIZED_UID
    ):
        raise PermissionError("r5 recovery repository or UID changed")


def _binding(path: Path) -> dict[str, object]:
    return io._file_binding(path.absolute())


def _current_recovery_tool_binding() -> dict[str, object]:
    return _binding(Path(__file__).absolute())


def _current_required_bindings() -> dict[str, object]:
    r4_path = Path(str(hardened.__file__)).absolute()
    r3_path = Path(str(io.__file__)).absolute()
    integration_path = Path(str(integration.__file__)).absolute()
    realizer_path = Path(str(realizer.__file__)).absolute()
    expected_r4 = REPOSITORY / "tools/cure_lite_v24_dummy_r4_recovery.py"
    expected_r3 = REPOSITORY / "tools/cure_lite_v24_dummy_r3_recovery.py"
    expected_integration = (
        REPOSITORY / "tools/cure_lite_v24_user_systemd_integration.py"
    )
    expected_realizer = (
        REPOSITORY / "tools/cure_lite_v24_realize_systemd_unit.py"
    )
    if (
        r4_path != expected_r4
        or r3_path != expected_r3
        or integration_path != expected_integration
        or realizer_path != expected_realizer
        or Path(str(integration.realizer.__file__)).absolute()
        != expected_realizer
        or str(realizer.SYSTEMCTL_PATH) != "/usr/bin/systemctl"
        or str(realizer.SYSTEMD_PATH) != "/usr/bin/systemd-path"
        or str(realizer.SYSTEMD_ANALYZE) != "/usr/bin/systemd-analyze"
    ):
        raise PermissionError("r5 recovery dependency origin changed")
    r4_binding = _binding(r4_path)
    r3_binding = _binding(r3_path)
    if r4_binding["file_sha256"] != FROZEN_R4_RECOVERY_SHA256:
        raise PermissionError("frozen r4 recovery template changed")
    if r3_binding["file_sha256"] != FROZEN_R3_IO_SHA256:
        raise PermissionError("frozen r3 hardened I/O changed")
    return {
        "r4_recovery_template": r4_binding,
        "hardened_io_library": r3_binding,
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
        raise PermissionError(f"r5 archived binding malformed:{name}")
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
        raise PermissionError(f"r5 archived binding malformed:{name}")
    return binding


def _read_anchor_bytes(name: str) -> bytes:
    anchor = _ARCHIVED_EVIDENCE_ANCHORS[name]
    raw, _current = io._read_regular_file_snapshot(
        Path(str(anchor["path"])),
        expected_owner_uid=AUTHORIZED_UID,
        expected_mode=int(anchor["mode"]),
    )
    if hashlib.sha256(raw).hexdigest() != anchor["file_sha256"]:
        raise PermissionError(f"r5 archived file digest changed:{name}")
    return raw


def _read_archived(name: str) -> dict[str, object]:
    anchor = _ARCHIVED_EVIDENCE_ANCHORS[name]
    if anchor["kind"] != "canonical-json":
        raise ValueError(f"r5 archive is not JSON:{name}")
    raw = _read_anchor_bytes(name)
    payload = json.loads(raw.decode("utf-8"))
    if (
        not isinstance(payload, dict)
        or raw != (io.canonical_json(payload) + "\n").encode("utf-8")
        or payload.get("schema_version") != anchor["schema_version"]
    ):
        raise PermissionError(f"r5 archived encoding changed:{name}")
    body = dict(payload)
    fingerprint = body.pop(str(anchor["fingerprint_field"]), None)
    if (
        fingerprint != anchor["fingerprint"]
        or fingerprint != io.stable_fingerprint(body)
    ):
        raise PermissionError(f"r5 archived fingerprint changed:{name}")
    return payload


def _read_archived_raw(name: str) -> bytes:
    if _ARCHIVED_EVIDENCE_ANCHORS[name]["kind"] != "raw":
        raise ValueError(f"r5 archive is not raw:{name}")
    return _read_anchor_bytes(name)


def _archived_roots() -> dict[str, object]:
    return json.loads(io.canonical_json(_ARCHIVED_EVIDENCE_ANCHORS))


def _no_payload(value: Mapping[str, object]) -> None:
    if (
        value.get("payload_authority") != "none"
        or value.get("D_R_payload_accessed") is not False
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or value.get("gpu_accessed", False) is not False
    ):
        raise PermissionError("r5 recovery lineage is not payload-free")


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
    binding = io._private_directory(path)
    if binding["owner_uid"] != AUTHORIZED_UID:
        raise PermissionError("r5 recovery directory owner changed")
    return binding


def _validate_inventory() -> None:
    if set(SCENARIO_ROOT.iterdir()) != {CONTROL_ROOT, RUNTIME_ROOT}:
        raise PermissionError("r5 scenario inventory changed")
    expected_control = {
        ORIGINAL_AUTHORIZATION_PATH,
        ORIGINAL_RUNTIME_SPEC_PATH,
        ORIGINAL_INTEGRATION_TERMINAL_PATH,
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
        raise PermissionError("r5 control inventory changed")
    for path in actual_control & recovery_paths:
        io._read_regular_file_snapshot(
            path,
            expected_owner_uid=AUTHORIZED_UID,
            expected_mode=0o444,
        )
    expected_runtime = {
        ORIGINAL_LAUNCH_LEASE_PATH,
        ORIGINAL_PRECOMMIT_PATH,
        ORIGINAL_ATTEMPT_COMMIT_PATH,
        ORIGINAL_CLAIM_PATH,
        ORIGINAL_START_ACK_PATH,
        ORIGINAL_CHILD_PRESPAWN_PATH,
        ORIGINAL_RUNTIME_TERMINAL_PATH,
        ORIGINAL_DUMMY_PATH,
        ORIGINAL_STDOUT_PATH,
        ORIGINAL_STDERR_PATH,
        HEARTBEAT_ROOT,
        INVOCATION_ROOT,
    }
    if set(RUNTIME_ROOT.iterdir()) != expected_runtime:
        raise PermissionError("r5 success evidence inventory changed")
    if set(HEARTBEAT_ROOT.iterdir()) != set(ORIGINAL_HEARTBEAT_PATHS):
        raise PermissionError("r5 heartbeat inventory changed")
    if set(INVOCATION_ROOT.iterdir()) != {ORIGINAL_SIDECAR_PATH}:
        raise PermissionError("r5 sidecar inventory changed")


def _load_original_chain() -> dict[str, object]:
    chain = {
        "authorization": _read_archived("authorization"),
        "spec": _read_archived("runtime_spec"),
        "lease": _read_archived("launch_lease"),
        "precommit": _read_archived("precommit"),
        "attempt": _read_archived("attempt_commit"),
        "claim": _read_archived("materialization_claim"),
        "start_ack": _read_archived("start_ack"),
        "child_prespawn": _read_archived("child_prespawn"),
        "heartbeats": [
            _read_archived(f"heartbeat_{sequence}") for sequence in range(3)
        ],
        "runtime_terminal": _read_archived("runtime_terminal"),
        "sidecar": _read_archived("systemd_sidecar"),
        "dummy": _read_archived("dummy_artifact"),
        "stdout": _read_archived_raw("stdout_log"),
        "stderr": _read_archived_raw("stderr_log"),
        "integration_terminal": _read_archived("integration_terminal"),
        "removal": _read_archived("removal_state"),
    }
    return chain


def _validate_original_chronology(chain: Mapping[str, object]) -> None:
    authorization = chain["authorization"]
    if not isinstance(authorization, Mapping):
        raise PermissionError("r5 authorization malformed")
    issued = _timestamp(authorization.get("issued_at_utc"), name="r5 issuance")
    expires = _timestamp(authorization.get("expires_at_utc"), name="r5 expiry")
    ordered = [
        chain["lease"],
        chain["precommit"],
        chain["attempt"],
        chain["claim"],
        chain["start_ack"],
        chain["child_prespawn"],
        *chain["heartbeats"],
        chain["runtime_terminal"],
        chain["sidecar"],
    ]
    times = [
        _timestamp(value.get("time_utc"), name=f"r5 event {index}")
        for index, value in enumerate(ordered)
        if isinstance(value, Mapping)
    ]
    integration_terminal = chain["integration_terminal"]
    if not isinstance(integration_terminal, Mapping):
        raise PermissionError("r5 integration terminal malformed")
    times.append(
        _timestamp(
            integration_terminal.get("created_at_utc"),
            name="r5 integration terminal",
        )
    )
    if (
        expires <= issued
        or expires - issued > timedelta(seconds=300)
        or len(times) != len(ordered) + 1
        or [issued, *times, expires] != sorted([issued, *times, expires])
    ):
        raise PermissionError("r5 archived wall chronology changed")
    monotonic_values = [
        value.get("monotonic_ns")
        for value in ordered[:-1]
        if isinstance(value, Mapping)
    ]
    if (
        len(monotonic_values) != len(ordered) - 1
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in monotonic_values
        )
        or monotonic_values != sorted(monotonic_values)
        or len(set(monotonic_values)) != len(monotonic_values)
    ):
        raise PermissionError("r5 archived monotonic chronology changed")


def _validate_original_authorization_and_spec(
    chain: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], str, str]:
    authorization = chain["authorization"]
    spec = chain["spec"]
    if not isinstance(authorization, Mapping) or not isinstance(spec, Mapping):
        raise PermissionError("r5 authorization/spec malformed")
    _no_payload(authorization)
    manager = authorization.get("manager_generation")
    rendered = authorization.get("rendered_fragment")
    if (
        manager != EXPECTED_MANAGER_GENERATION
        or not isinstance(manager, Mapping)
        or _BOOT_ID.fullmatch(str(manager.get("boot_id"))) is None
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
        or not isinstance(rendered, Mapping)
        or rendered.get("sha256") != EXPECTED_FRAGMENT_SHA256
        or hashlib.sha256(
            str(rendered.get("utf8_text")).encode("utf-8")
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
        raise PermissionError("r5 archived authorization changed")

    for name, path in {
        "scenario_root": SCENARIO_ROOT,
        "control_root": CONTROL_ROOT,
        "runtime_root": RUNTIME_ROOT,
    }.items():
        if authorization.get(name) != _private_directory(path):
            raise PermissionError(f"r5 archived directory changed:{name}")
    expected_control = {
        "dummy_artifact": str(ORIGINAL_DUMMY_PATH),
        "integration_receipt": str(CONTROL_ROOT / "integration-receipt.json"),
        "integration_terminal": str(ORIGINAL_INTEGRATION_TERMINAL_PATH),
        "removal_authorization": str(
            CONTROL_ROOT / "removal-authorization.json"
        ),
        "removal_state": str(ORIGINAL_REMOVAL_STATE_PATH),
    }
    if authorization.get("control_artifacts") != expected_control:
        raise PermissionError("r5 control artifact contract changed")

    expected_executable_paths = {
        "python": "/usr/bin/python3.12",
        "supervisor": str(
            REPOSITORY / "tools/cure_lite_v24_runtime_supervisor.py"
        ),
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
        raise PermissionError("r5 archived executable set changed")
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
        "child_prespawn_phase_receipt": str(ORIGINAL_CHILD_PRESPAWN_PATH),
        "consumed_start_failure_receipt": str(
            RUNTIME_ROOT / "consumed-start-failure.json"
        ),
        "gpu_lease_release_receipt": str(
            RUNTIME_ROOT / "gpu-lease-release.json"
        ),
        "heartbeat_dir": str(HEARTBEAT_ROOT),
        "launch_lease": str(ORIGINAL_LAUNCH_LEASE_PATH),
        "materialization_claim": str(ORIGINAL_CLAIM_PATH),
        "precommit_phase_receipt": str(ORIGINAL_PRECOMMIT_PATH),
        "root": str(RUNTIME_ROOT),
        "runtime_attestation": str(RUNTIME_ROOT / "runtime-attestation.json"),
        "runtime_terminal": str(ORIGINAL_RUNTIME_TERMINAL_PATH),
        "start_ack_receipt": str(ORIGINAL_START_ACK_PATH),
        "stderr_log": str(ORIGINAL_STDERR_PATH),
        "stdout_log": str(ORIGINAL_STDOUT_PATH),
        "systemd_invocation_dir": str(INVOCATION_ROOT),
    }
    runtime = spec.get("runtime")
    systemd = runtime.get("systemd") if isinstance(runtime, Mapping) else None
    child = spec.get("child")
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
        or runtime.get("shell") is not False
        or runtime.get("start_new_session") is not True
        or not isinstance(systemd, Mapping)
        or systemd.get("unit_name") != UNIT_NAME
        or systemd.get("unit_fragment_file_sha256")
        != EXPECTED_FRAGMENT_SHA256
        or not isinstance(child, Mapping)
        or child.get("entrypoint_path")
        != str(REPOSITORY / "tools/cure_lite_v24_dummy_child.py")
        or child.get("cwd") != str(SCENARIO_ROOT)
        or child.get("environment") != {}
        or child.get("inherit_environment") != []
        or not isinstance(source_bindings, Mapping)
        or source_bindings.get("supervisor_file_sha256")
        != archived_bindings["supervisor"]["file_sha256"]
        or source_bindings.get("child_entry_file_sha256")
        != archived_bindings["dummy_child"]["file_sha256"]
    ):
        raise PermissionError("r5 archived runtime spec changed")
    immutable = systemd.get("immutable_shadow_fingerprint")
    child_argv_fingerprint = child.get("argv_fingerprint")
    if (
        _SHA.fullmatch(str(immutable)) is None
        or _SHA.fullmatch(str(child_argv_fingerprint)) is None
    ):
        raise PermissionError("r5 immutable/child binding changed")
    return (
        archived_bindings,
        template,
        str(immutable),
        str(child_argv_fingerprint),
    )


def _expected_phase_state(
    *,
    active: str,
    sub: str,
    invocation_id: str,
) -> dict[str, str]:
    return {
        "ActiveState": active,
        "ExecMainCode": "0",
        "ExecMainStatus": "0",
        "InvocationID": invocation_id,
        "LoadState": "loaded",
        "NRestarts": "0",
        "NeedDaemonReload": "no",
        "Result": "success",
        "SubState": sub,
        "UnitFileState": "static",
    }


def _validate_phase_receipt(
    value: object,
    *,
    phase: str,
    state: Mapping[str, str],
    immutable: str,
    lease_fingerprint: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PermissionError(f"r5 phase malformed:{phase}")
    receipt = dict(value)
    if (
        not _common(receipt)
        or receipt.get("boot_id")
        != EXPECTED_MANAGER_GENERATION["boot_id"]
        or receipt.get("phase") != phase
        or receipt.get("launch_lease_fingerprint") != lease_fingerprint
        or receipt.get("immutable_shadow_fingerprint") != immutable
        or receipt.get("dispatch_lease_scope") != "attempt_dispatch_only"
        or receipt.get("launch_limit") != 1
        or receipt.get("automatic_retry_allowed") is not False
        or receipt.get("resume_allowed") is not False
        or receipt.get("runtime_environment_audit_valid") is not False
        or receipt.get("environment_audit_fingerprint") is not None
        or receipt.get("environment_inventory_fingerprint") is not None
        or receipt.get("gpu_lease_fingerprint") is not None
        or receipt.get("scientific_gate_passed") is not None
        or receipt.get("systemd_phase_state") != dict(state)
    ):
        raise PermissionError(f"r5 phase receipt changed:{phase}")
    return receipt


def _validate_original_success_runtime(
    chain: Mapping[str, object],
    *,
    immutable: str,
    child_argv_fingerprint: str,
) -> None:
    lease = chain["lease"]
    precommit = chain["precommit"]
    attempt = chain["attempt"]
    claim = chain["claim"]
    start_ack = chain["start_ack"]
    child_prespawn = chain["child_prespawn"]
    heartbeats = chain["heartbeats"]
    runtime_terminal = chain["runtime_terminal"]
    sidecar = chain["sidecar"]
    dummy = chain["dummy"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            lease,
            attempt,
            claim,
            runtime_terminal,
            sidecar,
            dummy,
        )
    ):
        raise PermissionError("r5 runtime evidence malformed")
    lease_fingerprint = str(
        _ARCHIVED_EVIDENCE_ANCHORS["launch_lease"]["fingerprint"]
    )
    attempt_fingerprint = str(
        _ARCHIVED_EVIDENCE_ANCHORS["attempt_commit"]["fingerprint"]
    )
    claim_fingerprint = str(
        _ARCHIVED_EVIDENCE_ANCHORS["materialization_claim"]["fingerprint"]
    )
    start_ack_fingerprint = str(
        _ARCHIVED_EVIDENCE_ANCHORS["start_ack"]["fingerprint"]
    )
    child_fingerprint = str(
        _ARCHIVED_EVIDENCE_ANCHORS["child_prespawn"]["fingerprint"]
    )
    runtime_terminal_fingerprint = str(
        _ARCHIVED_EVIDENCE_ANCHORS["runtime_terminal"]["fingerprint"]
    )
    if (
        not _common(lease)
        or lease.get("boot_id") != EXPECTED_MANAGER_GENERATION["boot_id"]
        or lease.get("authorization_fingerprint") is not None
        or lease.get("gpu_exclusivity_claimed") is not False
        or lease.get("launch_limit") != 1
        or lease.get("automatic_retry_allowed") is not False
        or lease.get("resume_allowed") is not False
        or lease.get("lease_scope") != "attempt_dispatch_only"
        or lease.get("launch_lease_fingerprint") != lease_fingerprint
    ):
        raise PermissionError("r5 archived launch lease changed")
    precommit_receipt = _validate_phase_receipt(
        precommit,
        phase="precommit",
        state=_expected_phase_state(
            active="inactive",
            sub="dead",
            invocation_id="",
        ),
        immutable=immutable,
        lease_fingerprint=lease_fingerprint,
    )
    if (
        not _common(attempt)
        or attempt.get("boot_id") != EXPECTED_MANAGER_GENERATION["boot_id"]
        or attempt.get("attempt_ordinal") != 0
        or attempt.get("prior_attempt_count") != 0
        or attempt.get("systemd_unit_name") != UNIT_NAME
        or attempt.get("dispatch_lease_scope") != "attempt_dispatch_only"
        or attempt.get("launch_limit") != 1
        or attempt.get("authorization_fingerprint") is not None
        or attempt.get("authorization_file_sha256") is not None
        or attempt.get("gpu_lease_fingerprint") is not None
        or attempt.get("gpu_lease_file_sha256") is not None
        or attempt.get("gpu_lease_device") is not None
        or attempt.get("gpu_lease_inode") is not None
        or attempt.get("planned_attempt_commit_fingerprint") is not None
        or attempt.get("launch_lease_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["launch_lease"]["file_sha256"]
        or attempt.get("launch_lease_fingerprint") != lease_fingerprint
        or attempt.get("precommit_phase_receipt_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["precommit"]["file_sha256"]
        or attempt.get("precommit_phase_receipt_fingerprint")
        != precommit_receipt.get("phase_receipt_fingerprint")
        or attempt.get("immutable_systemd_shadow_fingerprint") != immutable
        or attempt.get("automatic_retry_allowed") is not False
        or attempt.get("resume_allowed") is not False
        or attempt.get("runtime_environment_audit_valid") is not False
        or attempt.get("scientific_gate_passed") is not None
        or attempt.get("attempt_commit_fingerprint") != attempt_fingerprint
    ):
        raise PermissionError("r5 archived attempt commit changed")
    expected_control_group = (
        "/user.slice/user-1008.slice/user@1008.service/app.slice/" + UNIT_NAME
    )
    if (
        not _common(claim)
        or claim.get("boot_id") != EXPECTED_MANAGER_GENERATION["boot_id"]
        or claim.get("attempt_commit_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["attempt_commit"]["file_sha256"]
        or claim.get("attempt_commit_fingerprint") != attempt_fingerprint
        or claim.get("authorization_fingerprint") is not None
        or claim.get("child_argv_fingerprint") != child_argv_fingerprint
        or claim.get("materialization_claim_fingerprint")
        != claim_fingerprint
        or claim.get("systemd_invocation_id") != INVOCATION_ID
        or claim.get("systemd_control_group") != expected_control_group
        or claim.get("shell") is not False
        or claim.get("launch_limit") != 1
        or claim.get("automatic_retry_allowed") is not False
        or claim.get("resume_allowed") is not False
        or claim.get("scientific_gate_passed") is not None
    ):
        raise PermissionError("r5 archived materialization claim changed")
    start_receipt = _validate_phase_receipt(
        start_ack,
        phase="start_ack",
        state=_expected_phase_state(
            active="activating",
            sub="start-pre",
            invocation_id=INVOCATION_ID,
        ),
        immutable=immutable,
        lease_fingerprint=lease_fingerprint,
    )
    child_receipt = _validate_phase_receipt(
        child_prespawn,
        phase="child_prespawn",
        state=_expected_phase_state(
            active="active",
            sub="running",
            invocation_id=INVOCATION_ID,
        ),
        immutable=immutable,
        lease_fingerprint=lease_fingerprint,
    )
    if (
        start_receipt.get("phase_receipt_fingerprint")
        != start_ack_fingerprint
        or child_receipt.get("phase_receipt_fingerprint")
        != child_fingerprint
    ):
        raise PermissionError("r5 phase fingerprint chain changed")

    if not isinstance(heartbeats, list) or len(heartbeats) != 3:
        raise PermissionError("r5 heartbeat chain malformed")
    heartbeat_hashes = [
        str(_ARCHIVED_EVIDENCE_ANCHORS[f"heartbeat_{index}"]["file_sha256"])
        for index in range(3)
    ]
    expected_previous = [
        str(
            _ARCHIVED_EVIDENCE_ANCHORS["materialization_claim"][
                "file_sha256"
            ]
        ),
        heartbeat_hashes[0],
        heartbeat_hashes[1],
    ]
    expected_events = ["child_started", "child_running", "child_reaped"]
    child_pid: int | None = None
    child_starttime: int | None = None
    supervisor_pid: int | None = None
    supervisor_starttime: int | None = None
    for index, value in enumerate(heartbeats):
        if not isinstance(value, Mapping):
            raise PermissionError("r5 heartbeat malformed")
        current_child_pid = value.get("child_pid")
        current_child_start = value.get("child_proc_starttime_ticks")
        current_supervisor_pid = value.get("supervisor_pid")
        current_supervisor_start = value.get(
            "supervisor_proc_starttime_ticks"
        )
        if index == 0:
            child_pid = current_child_pid
            child_starttime = current_child_start
            supervisor_pid = current_supervisor_pid
            supervisor_starttime = current_supervisor_start
        if (
            value.get("attempt_id") != ATTEMPT_ID
            or value.get("boot_id")
            != EXPECTED_MANAGER_GENERATION["boot_id"]
            or value.get("systemd_invocation_id") != INVOCATION_ID
            or value.get("sequence") != index
            or value.get("event") != expected_events[index]
            or value.get("previous_event_file_sha256")
            != expected_previous[index]
            or current_child_pid != child_pid
            or current_child_start != child_starttime
            or current_supervisor_pid != supervisor_pid
            or current_supervisor_start != supervisor_starttime
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item <= 0
                for item in (
                    current_child_pid,
                    current_child_start,
                    current_supervisor_pid,
                    current_supervisor_start,
                )
            )
        ):
            raise PermissionError(f"r5 heartbeat changed:{index}")
    if (
        heartbeats[0].get("details")
        != {"child_argv_fingerprint": child_argv_fingerprint}
        or heartbeats[1].get("details")
        != {"termination_requested": False}
        or heartbeats[2].get("details") != {"raw_return_code": 0}
        or dummy.get("scenario_id") != SCENARIO_ID
        or dummy.get("pid") != child_pid
        or dummy.get("dataset_accessed") is not False
        or dummy.get("gpu_accessed") is not False
        or dummy.get("torch_imported") is not False
    ):
        raise PermissionError("r5 dummy child evidence changed")

    expected_child_outcome = {
        "category": "EXITED_0",
        "exit_status": 0,
        "forced_kill": False,
        "forwarded_signals": [],
        "raw_return_code": 0,
        "signal_name": None,
        "signal_number": None,
        "spawn_errno": None,
        "spawn_error_type": None,
    }
    stdout = chain["stdout"]
    stderr = chain["stderr"]
    expected_stdout = b"CURE-Lite v24 user-systemd dummy child\n"
    if stdout != expected_stdout or stderr != b"":
        raise PermissionError("r5 archived logs changed")
    expected_stdout_binding = {
        "file_sha256": _ARCHIVED_EVIDENCE_ANCHORS["stdout_log"][
            "file_sha256"
        ],
        "hardlink_count": 1,
        "mode": 0o444,
        "path": str(ORIGINAL_STDOUT_PATH),
        "size_bytes": len(expected_stdout),
    }
    expected_stderr_binding = {
        "file_sha256": _ARCHIVED_EVIDENCE_ANCHORS["stderr_log"][
            "file_sha256"
        ],
        "hardlink_count": 1,
        "mode": 0o444,
        "path": str(ORIGINAL_STDERR_PATH),
        "size_bytes": 0,
    }
    if (
        not _common(runtime_terminal)
        or runtime_terminal.get("boot_id")
        != EXPECTED_MANAGER_GENERATION["boot_id"]
        or runtime_terminal.get("systemd_invocation_id") != INVOCATION_ID
        or runtime_terminal.get("child_outcome") != expected_child_outcome
        or runtime_terminal.get("process_group_cleanup_signals") != []
        or runtime_terminal.get("supervisor_error_type") is not None
        or runtime_terminal.get("heartbeat_event_count") != 3
        or runtime_terminal.get("last_heartbeat_path")
        != str(ORIGINAL_HEARTBEAT_PATHS[2])
        or runtime_terminal.get("last_heartbeat_file_sha256")
        != heartbeat_hashes[2]
        or runtime_terminal.get("materialization_claim_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["materialization_claim"][
            "file_sha256"
        ]
        or runtime_terminal.get("start_ack_receipt_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["start_ack"]["file_sha256"]
        or runtime_terminal.get("start_ack_receipt_fingerprint")
        != start_ack_fingerprint
        or runtime_terminal.get("child_prespawn_phase_receipt_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["child_prespawn"]["file_sha256"]
        or runtime_terminal.get("child_prespawn_phase_receipt_fingerprint")
        != child_fingerprint
        or runtime_terminal.get("stdout_log") != expected_stdout_binding
        or runtime_terminal.get("stderr_log") != expected_stderr_binding
        or runtime_terminal.get("runtime_terminal_fingerprint")
        != runtime_terminal_fingerprint
        or runtime_terminal.get("scientific_gate_passed") is not None
        or runtime_terminal.get("scientific_decision")
        != "NOT_EVALUATED_BY_RUNTIME_SUPERVISOR"
    ):
        raise PermissionError("r5 archived runtime terminal changed")

    expected_systemd_outcome = {
        "category": "SYSTEMD_SERVICE_SUCCESS",
        "exit_code": "exited",
        "exit_status": "0",
        "invocation_id": INVOCATION_ID,
        "scientific_gate_passed": None,
        "service_result": "success",
        "systemd_success": True,
    }
    if (
        sidecar.get("candidate") != CANDIDATE
        or sidecar.get("stage_id") != STAGE_ID
        or sidecar.get("attempt_id") != ATTEMPT_ID
        or sidecar.get("runtime_spec_fingerprint")
        != _ARCHIVED_EVIDENCE_ANCHORS["runtime_spec"]["fingerprint"]
        or sidecar.get("sidecar_systemd_invocation_id") != INVOCATION_ID
        or sidecar.get("claim_systemd_invocation_id") != INVOCATION_ID
        or sidecar.get("systemd_control_group") != expected_control_group
        or sidecar.get("attempt_commit_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["attempt_commit"]["file_sha256"]
        or sidecar.get("attempt_commit_fingerprint") != attempt_fingerprint
        or sidecar.get("materialization_claim_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["materialization_claim"][
            "file_sha256"
        ]
        or sidecar.get("materialization_claim_fingerprint")
        != claim_fingerprint
        or sidecar.get("start_ack_receipt_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["start_ack"]["file_sha256"]
        or sidecar.get("start_ack_receipt_fingerprint")
        != start_ack_fingerprint
        or sidecar.get("child_prespawn_phase_receipt_file_sha256")
        != _ARCHIVED_EVIDENCE_ANCHORS["child_prespawn"]["file_sha256"]
        or sidecar.get("child_prespawn_phase_receipt_fingerprint")
        != child_fingerprint
        or sidecar.get("attempt_commit_required") is not True
        or sidecar.get("attempt_commit_valid") is not True
        or sidecar.get("authorization_matches_commit") is not True
        or sidecar.get("current_runtime_closure_valid") is not True
        or sidecar.get("current_runtime_closure_error_type") is not None
        or sidecar.get("claim_valid") is not True
        or sidecar.get("claim_matches_invocation") is not True
        or sidecar.get("start_ack_valid") is not True
        or sidecar.get("child_prespawn_valid") is not True
        or sidecar.get("audit_valid") is not True
        or sidecar.get("active_gpu_lease_valid") is not None
        or sidecar.get("active_gpu_lease_fingerprint") is not None
        or sidecar.get("gpu_lease_release_authorized") is not None
        or sidecar.get("gpu_lease_release_valid") is not None
        or sidecar.get("gpu_lease_release_receipt_fingerprint") is not None
        or sidecar.get("gpu_lease_tombstone_file_sha256") is not None
        or sidecar.get("systemd_outcome") != expected_systemd_outcome
        or sidecar.get("scientific_gate_passed") is not None
        or sidecar.get("scientific_decision")
        != "NOT_EVALUATED_BY_RUNTIME_SUPERVISOR"
    ):
        raise PermissionError("r5 archived systemd terminal changed")


def _validate_original_terminals(chain: Mapping[str, object]) -> None:
    authorization = chain["authorization"]
    spec = chain["spec"]
    terminal = chain["integration_terminal"]
    removal = chain["removal"]
    if not all(
        isinstance(value, Mapping)
        for value in (authorization, spec, terminal, removal)
    ):
        raise PermissionError("r5 terminal evidence malformed")
    _no_payload(terminal)
    _no_payload(removal)
    expected_actions = [
        "realize-static-fragment",
        "daemon-reload-after-realization",
        "supervisor-commit-and-start",
        "verify-supervisor-evidence",
    ]
    expected_evidence = {
        "attempt_commit_fingerprint": _ARCHIVED_EVIDENCE_ANCHORS[
            "attempt_commit"
        ]["fingerprint"],
        "child_prespawn_fingerprint": _ARCHIVED_EVIDENCE_ANCHORS[
            "child_prespawn"
        ]["fingerprint"],
        "gpu_lease_evidence_absent": True,
        "invocation_id": INVOCATION_ID,
        "launch_lease_fingerprint": _ARCHIVED_EVIDENCE_ANCHORS[
            "launch_lease"
        ]["fingerprint"],
        "materialization_claim_fingerprint": _ARCHIVED_EVIDENCE_ANCHORS[
            "materialization_claim"
        ]["fingerprint"],
        "precommit_fingerprint": _ARCHIVED_EVIDENCE_ANCHORS["precommit"][
            "fingerprint"
        ],
        "runtime_attestation_absent": True,
        "runtime_terminal_fingerprint": _ARCHIVED_EVIDENCE_ANCHORS[
            "runtime_terminal"
        ]["fingerprint"],
        "start_ack_fingerprint": _ARCHIVED_EVIDENCE_ANCHORS["start_ack"][
            "fingerprint"
        ],
        "systemd_terminal_fingerprint": _ARCHIVED_EVIDENCE_ANCHORS[
            "systemd_sidecar"
        ]["fingerprint"],
    }
    if (
        terminal.get("scenario_id") != SCENARIO_ID
        or terminal.get("identity") != _identity()
        or terminal.get("authorization_fingerprint")
        != authorization.get("authorization_fingerprint")
        or terminal.get("runtime_spec_fingerprint")
        != spec.get("runtime_spec_fingerprint")
        or terminal.get("passed") is not True
        or terminal.get("completed_actions") != expected_actions
        or terminal.get("supervisor_evidence") != expected_evidence
        or terminal.get("error_type") is not None
        or terminal.get("error_message") is not None
        or terminal.get("direct_systemctl_start_attempted") is not False
        or terminal.get("enable_attempted") is not False
        or terminal.get("remove_attempted") is not False
    ):
        raise PermissionError("r5 archived successful terminal changed")
    if (
        removal.get("scenario_id") != SCENARIO_ID
        or removal.get("unit_name") != UNIT_NAME
        or removal.get("removal_authorization_fingerprint") is not None
        or removal.get("passed") is not False
        or removal.get("remove_attempted") is not False
        or removal.get("fragment_absent") is not False
        or removal.get("not_found_state") is not None
        or removal.get("completed_actions") != expected_actions
        or removal.get("error_type") != "PermissionError"
        or removal.get("error_message")
        != "unit search paths changed after terminal"
    ):
        raise PermissionError("r5 archived cleanup-policy failure changed")


def _sealed_original_chain() -> dict[str, object]:
    _assert_repository()
    _validate_inventory()
    chain = _load_original_chain()
    _validate_original_chronology(chain)
    archived_bindings, template, immutable, child_argv = (
        _validate_original_authorization_and_spec(chain)
    )
    _validate_original_success_runtime(
        chain,
        immutable=immutable,
        child_argv_fingerprint=child_argv,
    )
    _validate_original_terminals(chain)
    return {
        **chain,
        "archived_roots": _archived_roots(),
        "archived_executable_bindings": archived_bindings,
        "archived_template_binding": template,
    }


def _validated_path_policy(
    archived: Mapping[str, object],
    observed: Mapping[str, object],
) -> dict[str, object]:
    """Accept only generator.late inode regeneration in the normalized policy."""

    archived_body = dict(archived)
    observed_body = dict(observed)
    archived_rows_value = archived_body.pop("ordered_unit_paths", None)
    observed_rows_value = observed_body.pop("ordered_unit_paths", None)
    if (
        archived_body != observed_body
        or not isinstance(archived_rows_value, list)
        or not isinstance(observed_rows_value, list)
    ):
        raise PermissionError("r5 unit path policy changed")
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
        raise PermissionError("r5 ordered unit path set changed")
    allowed = "/run/user/1008/systemd/generator.late"
    for path, expected in archived_rows.items():
        current = observed_rows[path]
        if path != allowed:
            if current != expected:
                raise PermissionError(f"r5 unit path changed:{path}")
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
            raise PermissionError("r5 generator.late identity changed")
    return json.loads(io.canonical_json(dict(observed)))


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
        raise PermissionError("r5 fragment path changed")
    data, current = io._read_regular_file_snapshot(
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
        raise PermissionError("r5 fragment identity changed")
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
    if not isinstance(authorization, Mapping):
        raise PermissionError("r5 authorization malformed")
    policy = realizer.freeze_user_unit_path_policy(
        UNIT_NAME,
        runner=runner,
        allowed_fragment=EXPECTED_FRAGMENT_PATH,
    )
    policy = _validated_path_policy(authorization["unit_path_policy"], policy)
    manager = manager_reader()
    integration._validate_manager_generation(manager)
    if (
        manager != EXPECTED_MANAGER_GENERATION
        or manager != authorization["manager_generation"]
    ):
        raise PermissionError("r5 manager generation changed")
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
        raise PermissionError("r5 unit is not exact inactive static state")
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
        raise PermissionError("r5 recovery implementation changed")


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
        raise PermissionError("r5 manager changed after intent")
    policy = realizer.freeze_user_unit_path_policy(
        UNIT_NAME,
        runner=runner,
        allowed_fragment=EXPECTED_FRAGMENT_PATH,
    )
    if policy != dict(unit_path_policy):
        raise PermissionError("r5 unit path changed after intent")
    state = realizer.query_unit_properties(UNIT_NAME, runner=runner)
    realizer.validate_realized_static_unit(plan, state)
    if state != dict(inactive_state):
        raise PermissionError("r5 unit state changed after intent")


def _revalidate_manager(
    manager_generation: Mapping[str, object],
    *,
    manager_reader: Callable[[], dict[str, object]],
) -> None:
    manager = manager_reader()
    integration._validate_manager_generation(manager)
    if manager != dict(manager_generation):
        raise PermissionError("r5 manager changed before daemon-reload")


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
        raise PermissionError("r5 removal identity is not exact")
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
            raise RuntimeError("r5 recovery requires O_NOFOLLOW")
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
                raise PermissionError(f"r5 fragment changed at check {ordinal}")

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
            name="r5 recovery issuance",
        )
        intent_time = _timestamp(
            intent.get("created_at_utc"),
            name="r5 recovery intent",
        )
        expires = _timestamp(
            authorization.get("expires_at_utc"),
            name="r5 recovery expiry",
        )
        if not issued <= intent_time <= action_started <= expires:
            raise PermissionError("r5 recovery expired before unlink")
        on_action_started(action_started.isoformat().replace("+00:00", "Z"))
        _check(4)
        if datetime.now(timezone.utc) > expires:
            raise PermissionError("r5 recovery expired at unlink")
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
        raise ValueError("r5 recovery validity must be in [1,300]")
    if any(
        os.path.lexists(path)
        for path in (
            RECOVERY_AUTHORIZATION_PATH,
            RECOVERY_INTENT_PATH,
            RECOVERY_TERMINAL_PATH,
        )
    ):
        raise FileExistsError("r5 recovery identity is consumed")
    tool_before = _current_recovery_tool_binding()
    required_before = _current_required_bindings()
    context = _live_context(runner=runner, manager_reader=manager_reader)
    tool_after = _current_recovery_tool_binding()
    required_after = _current_required_bindings()
    if tool_before != tool_after or required_before != required_after:
        raise PermissionError("r5 recovery changed during authorization")
    issued = datetime.now(timezone.utc)
    action = {
        "ordinal": 0,
        "action": "remove-exact-r5-runtime-static-fragment",
        "unit_name": UNIT_NAME,
        "fragment_path": str(EXPECTED_FRAGMENT_PATH),
        "then": ["daemon-reload", "verify-not-found"],
    }
    return io._write_sealed(
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
    return io._read_recovery_sealed_snapshot(
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
        raise PermissionError("r5 recovery authorization keys changed")
    _no_payload(authorization)
    issued = _timestamp(authorization.get("issued_at_utc"), name="r5 issuance")
    expires = _timestamp(authorization.get("expires_at_utc"), name="r5 expiry")
    now = datetime.now(timezone.utc)
    expected_action = {
        "ordinal": 0,
        "action": "remove-exact-r5-runtime-static-fragment",
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
        raise PermissionError("r5 recovery authorization changed or expired")
    tool = authorization.get("current_recovery_tool_binding")
    required = authorization.get("current_required_executable_bindings")
    if not isinstance(tool, Mapping) or not isinstance(required, Mapping):
        raise PermissionError("r5 recovery current bindings malformed")
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
        raise PermissionError("r5 recovery live closure changed")
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
        raise PermissionError("r5 authorization changed during live closure")
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
        raise PermissionError("explicit r5 recovery execution required")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise ValueError("r5 recovery timeout must be finite positive")
    if os.path.lexists(RECOVERY_INTENT_PATH) or os.path.lexists(
        RECOVERY_TERMINAL_PATH
    ):
        raise FileExistsError("r5 recovery execution identity consumed")
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
        raise PermissionError("r5 authorization changed before intent")
    intent = io._write_sealed(
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
        raise RuntimeError("r5 recovery intent schema changed")
    completed: list[str] = []
    post_state: dict[str, str] | None = None
    action_started: str | None = None
    error: BaseException | None = None

    def _record(value: str) -> None:
        nonlocal action_started
        if action_started is not None:
            raise RuntimeError("r5 recovery action start repeated")
        _timestamp(value, name="r5 action start")
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
        completed.append("remove-exact-r5-runtime-static-fragment")
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
    terminal = io._write_sealed(
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
                    "remove-exact-r5-runtime-static-fragment",
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
        raise RuntimeError("r5 recovery terminal schema changed")
    intent_time = _timestamp(intent["created_at_utc"], name="r5 intent")
    terminal_time = _timestamp(terminal["created_at_utc"], name="r5 terminal")
    if action_started is None:
        if terminal_time < intent_time:
            raise RuntimeError("r5 failure chronology changed")
    elif not (
        intent_time
        <= _timestamp(action_started, name="r5 action start")
        <= terminal_time
    ):
        raise RuntimeError("r5 action chronology changed")
    if error is not None:
        raise error
    if terminal["passed"] is not True:
        raise RuntimeError("r5 recovery did not reach PASS")
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
