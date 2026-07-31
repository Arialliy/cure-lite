#!/usr/bin/env python3
"""One-shot recovery for the failed CURE-Lite v24 dummy integration r2.

This tool is deliberately narrower than the integration harness.  It can
authorize and remove only the exact inactive runtime fragment left by the
pre-commit WatchdogUSec normalization failure.  It cannot start, stop, enable,
or inspect any dataset/GPU payload.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import stat
from typing import Callable, Mapping, Sequence

try:
    from tools import cure_lite_v24_realize_systemd_unit as realizer
    from tools import cure_lite_v24_user_systemd_integration as integration
except ModuleNotFoundError:
    import importlib.util
    import sys

    _TOOLS = Path(__file__).resolve().parent

    def _load(name: str, filename: str) -> object:
        spec = importlib.util.spec_from_file_location(name, _TOOLS / filename)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load isolated recovery dependency:{name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    realizer = _load(
        "cure_lite_v24_realize_systemd_unit_recovery",
        "cure_lite_v24_realize_systemd_unit.py",
    )
    integration = _load(
        "cure_lite_v24_user_systemd_integration_recovery",
        "cure_lite_v24_user_systemd_integration.py",
    )


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY
    / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()
SCENARIO_ROOT = (
    EVIDENCE_ROOT / "supervisor_v2_systemd_integration_r2"
).resolve()
CONTROL_ROOT = (SCENARIO_ROOT / "control").resolve()
ORIGINAL_AUTHORIZATION_PATH = (CONTROL_ROOT / "authorization.json").resolve()
ORIGINAL_RUNTIME_SPEC_PATH = (CONTROL_ROOT / "runtime-spec.json").resolve()
ORIGINAL_TERMINAL_PATH = (CONTROL_ROOT / "integration-terminal.json").resolve()
ORIGINAL_REMOVAL_STATE_PATH = (CONTROL_ROOT / "removal-state.json").resolve()
RECOVERY_AUTHORIZATION_PATH = (
    CONTROL_ROOT / "recovery-removal-authorization.json"
).resolve()
RECOVERY_INTENT_PATH = (
    CONTROL_ROOT / "recovery-removal-intent.json"
).resolve()
RECOVERY_TERMINAL_PATH = (
    CONTROL_ROOT / "recovery-removal-terminal.json"
).resolve()

SCENARIO_ID = "supervisor-v2-dummy-202607300349beef"
UNIT_NAME = (
    "cure-lite-v24-supervisor-integration-"
    "supervisor-v2-dummy-202607300349beef.service"
)
INSTRUCTION_ID = "user-2026-07-30-modify-then-run-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后运行"
FAILURE_MESSAGE = "loaded supervisor-v2 immutable shadow changed"
EXPECTED_RUNTIME_UNIT_DIRECTORY = Path(
    f"/run/user/{os.getuid()}/systemd/user"
)
EXPECTED_FRAGMENT_PATH = EXPECTED_RUNTIME_UNIT_DIRECTORY / UNIT_NAME
EXPECTED_FRAGMENT_SHA256 = (
    "6263ce736fe4575317aaf5c2bbfeb438f79070382c9b4af6565193b73621f8f8"
)
EXPECTED_FRAGMENT_DEVICE = 54
EXPECTED_FRAGMENT_INODE = 38142
_ARCHIVED_EVIDENCE_ANCHORS = {
    "original_authorization": {
        "path": str(ORIGINAL_AUTHORIZATION_PATH),
        "file_sha256": (
            "07ee121ed9fc79e410a4c0bd9d34965f963b958c1ec3b4c07c6271414de3e2d2"
        ),
        "fingerprint_field": "authorization_fingerprint",
        "fingerprint": (
            "de90d2fdf32f395374b00d475902a2fde65941dfe02ae0630d1f6e3e6c9fc9c5"
        ),
    },
    "runtime_spec": {
        "path": str(ORIGINAL_RUNTIME_SPEC_PATH),
        "file_sha256": (
            "9976a156fa9a6931159303ab5fda884ff3b01342bece3d8c491064ba9e639910"
        ),
        "fingerprint_field": "runtime_spec_fingerprint",
        "fingerprint": (
            "c6f431839ebf727153efc9b80848cc0be1728ab4489fc39ee5b8fba6279a3c32"
        ),
    },
    "integration_terminal": {
        "path": str(ORIGINAL_TERMINAL_PATH),
        "file_sha256": (
            "49149c69fcf528333bc915e9dcfbb6c8d2c5fe7d17d8559cf4260ae0c1473028"
        ),
        "fingerprint_field": "integration_terminal_fingerprint",
        "fingerprint": (
            "98d83e4964721fbfeb877392f4c80b59606e021faaf592a3986a6f8b91362d0c"
        ),
    },
    "removal_state": {
        "path": str(ORIGINAL_REMOVAL_STATE_PATH),
        "file_sha256": (
            "a3bd6afcd7b12b8f3ceac7e2e3d66fcb0902a86e5679dfb87279168db9b1096e"
        ),
        "fingerprint_field": "removal_state_fingerprint",
        "fingerprint": (
            "490fd9c1fdee8932456eee3e77a4238b59a270308c80dd6dd168deecec700c98"
        ),
    },
}
AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-dummy-integration-recovery-authorization-v1"
)
INTENT_SCHEMA = "cure-lite-v24-dummy-integration-recovery-intent-v1"
TERMINAL_SCHEMA = "cure-lite-v24-dummy-integration-recovery-terminal-v1"
_SHA = re.compile(r"[0-9a-f]{64}")
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
_TERMINAL_KEYS = {
    "schema_version",
    "scenario_id",
    "identity",
    "authorization_fingerprint",
    "runtime_spec_fingerprint",
    "created_at_utc",
    "passed",
    "completed_actions",
    "supervisor_evidence",
    "error_type",
    "error_message",
    "direct_systemctl_start_attempted",
    "enable_attempted",
    "remove_attempted",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "gpu_accessed",
    "integration_terminal_fingerprint",
}
_REMOVAL_STATE_KEYS = {
    "schema_version",
    "scenario_id",
    "unit_name",
    "removal_authorization_fingerprint",
    "passed",
    "remove_attempted",
    "fragment_absent",
    "not_found_state",
    "completed_actions",
    "error_type",
    "error_message",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "removal_state_fingerprint",
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
_RECOVERY_TERMINAL_KEYS = {
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
    if not isinstance(value, str):
        raise ValueError(f"{name} timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} timestamp is malformed") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} timestamp is naive")
    return parsed


def _no_payload(value: Mapping[str, object]) -> None:
    if (
        value.get("payload_authority") != "none"
        or any(
            value.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or value.get("gpu_accessed", False) is not False
    ):
        raise PermissionError("dummy recovery evidence is not payload-free")


def _root(
    path: Path,
    payload: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    return {
        "path": str(path),
        "file_sha256": integration.file_sha256(path),
        "fingerprint_field": fingerprint_field,
        "fingerprint": payload[fingerprint_field],
    }


def _validate_root(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "file_sha256",
        "fingerprint_field",
        "fingerprint",
    }:
        raise PermissionError(f"archived root is malformed:{name}")
    if (
        not isinstance(value.get("path"), str)
        or not Path(str(value["path"])).is_absolute()
        or not isinstance(value.get("fingerprint_field"), str)
        or _SHA.fullmatch(str(value.get("file_sha256"))) is None
        or _SHA.fullmatch(str(value.get("fingerprint"))) is None
    ):
        raise PermissionError(f"archived root is malformed:{name}")


def _validate_archived_file_binding(
    value: object,
    *,
    name: str,
    expected_path: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FILE_BINDING_KEYS:
        raise PermissionError(f"archived executable binding malformed:{name}")
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
        raise PermissionError(f"archived executable binding malformed:{name}")
    return binding


def _sealed_original_chain() -> dict[str, object]:
    authorization = integration._read_sealed(
        ORIGINAL_AUTHORIZATION_PATH,
        fingerprint_field="authorization_fingerprint",
        schema=integration.AUTHORIZATION_SCHEMA,
    )
    terminal = integration._read_sealed(
        ORIGINAL_TERMINAL_PATH,
        fingerprint_field="integration_terminal_fingerprint",
        schema=integration.INTEGRATION_TERMINAL_SCHEMA,
    )
    removal_state = integration._read_sealed(
        ORIGINAL_REMOVAL_STATE_PATH,
        fingerprint_field="removal_state_fingerprint",
        schema=integration.REMOVAL_STATE_SCHEMA,
    )
    if (
        set(authorization) != integration._AUTH_KEYS
        or set(terminal) != _TERMINAL_KEYS
        or set(removal_state) != _REMOVAL_STATE_KEYS
    ):
        raise PermissionError("dummy r2 archived evidence schema changed")
    _no_payload(authorization)
    _no_payload(terminal)
    _no_payload(removal_state)
    issued = _timestamp(
        authorization.get("issued_at_utc"),
        name="archived integration issuance",
    )
    expires = _timestamp(
        authorization.get("expires_at_utc"),
        name="archived integration expiry",
    )
    terminal_created = _timestamp(
        terminal.get("created_at_utc"),
        name="archived integration terminal",
    )
    identity = integration.build_supervisor_v2_identity(SCENARIO_ID)
    expected_actions = [
        "realize-static-fragment",
        "daemon-reload-after-realization",
    ]
    if (
        authorization.get("scenario_id") != SCENARIO_ID
        or authorization.get("identity") != identity
        or identity.get("unit_name") != UNIT_NAME
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis") != AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != os.getuid()
        or authorization.get("integration_authorized") is not True
        or authorization.get("actual_r2_authorized") is not False
        or authorization.get("unit_realization_authorized") is not True
        or authorization.get("unit_removal_authorized") is not False
        or authorization.get("enable_authorized") is not False
        or authorization.get("direct_start_authorized") is not False
        or authorization.get("gpu_access_authorized") is not False
        or expires <= issued
        or expires - issued > timedelta(seconds=300)
        or not issued <= terminal_created <= expires
        or terminal.get("scenario_id") != SCENARIO_ID
        or terminal.get("identity") != identity
        or terminal.get("authorization_fingerprint")
        != authorization["authorization_fingerprint"]
        or terminal.get("runtime_spec_fingerprint")
        != authorization["runtime_spec_binding"]["runtime_spec_fingerprint"]
        or terminal.get("passed") is not False
        or terminal.get("completed_actions") != expected_actions
        or terminal.get("supervisor_evidence") is not None
        or terminal.get("error_type") != "PermissionError"
        or terminal.get("error_message") != FAILURE_MESSAGE
        or terminal.get("direct_systemctl_start_attempted") is not False
        or terminal.get("enable_attempted") is not False
        or terminal.get("remove_attempted") is not False
        or terminal.get("gpu_accessed") is not False
        or removal_state.get("scenario_id") != SCENARIO_ID
        or removal_state.get("unit_name") != UNIT_NAME
        or removal_state.get("passed") is not False
        or removal_state.get("remove_attempted") is not False
        or removal_state.get("fragment_absent") is not False
        or removal_state.get("not_found_state") is not None
        or removal_state.get("completed_actions") != expected_actions
        or removal_state.get("error_type") != "PermissionError"
        or removal_state.get("error_message") != FAILURE_MESSAGE
        or removal_state.get("removal_authorization_fingerprint") is not None
    ):
        raise PermissionError("dummy r2 failure lineage is not exact")
    for key in ("scenario_root", "control_root", "runtime_root"):
        integration._validate_private_directory(authorization[key])
    if (
        authorization["scenario_root"]["path"] != str(SCENARIO_ROOT)
        or authorization["control_root"]["path"] != str(CONTROL_ROOT)
        or authorization.get("unit_directory")
        != str(EXPECTED_RUNTIME_UNIT_DIRECTORY)
        or authorization["unit_path_policy"].get("runtime_directory")
        != str(EXPECTED_RUNTIME_UNIT_DIRECTORY)
    ):
        raise PermissionError("dummy r2 archived directory roots changed")
    control = authorization.get("control_artifacts")
    if (
        not isinstance(control, Mapping)
        or control.get("integration_terminal") != str(ORIGINAL_TERMINAL_PATH)
        or control.get("removal_state") != str(ORIGINAL_REMOVAL_STATE_PATH)
    ):
        raise PermissionError("dummy r2 control artifact roots changed")

    spec_binding = authorization.get("runtime_spec_binding")
    if not isinstance(spec_binding, Mapping) or set(spec_binding) != {
        "path",
        "file_sha256",
        "runtime_spec_fingerprint",
        "required_schema",
    }:
        raise PermissionError("dummy r2 runtime spec binding malformed")
    if spec_binding.get("path") != str(ORIGINAL_RUNTIME_SPEC_PATH):
        raise PermissionError("dummy r2 runtime spec path changed")
    spec_path = ORIGINAL_RUNTIME_SPEC_PATH
    spec = integration._read_sealed(
        spec_path,
        fingerprint_field="runtime_spec_fingerprint",
        schema=integration.RUNTIME_SPEC_SCHEMA,
    )
    if (
        spec_binding.get("required_schema") != integration.RUNTIME_SPEC_SCHEMA
        or spec_binding.get("file_sha256")
        != integration.file_sha256(spec_path)
        or spec_binding.get("runtime_spec_fingerprint")
        != spec.get("runtime_spec_fingerprint")
        or spec.get("execution_kind") != integration.EXECUTION_KIND
        or spec.get("environment") is not None
        or spec["runtime"]["systemd"]["unit_name"] != UNIT_NAME
        or spec["runtime"]["systemd"]["unit_fragment_file_sha256"]
        != authorization["rendered_fragment"]["sha256"]
        or authorization["rendered_fragment"]["sha256"]
        != EXPECTED_FRAGMENT_SHA256
    ):
        raise PermissionError("dummy r2 runtime spec lineage changed")
    archived = authorization.get("executable_bindings")
    if not isinstance(archived, Mapping) or set(archived) != {
        "python",
        "supervisor",
        "integration_tool",
        "realizer",
        "dummy_child",
        "systemd_path",
        "systemd_analyze",
        "systemctl",
    }:
        raise PermissionError("dummy r2 executable root set changed")
    _validate_archived_file_binding(
        archived["supervisor"],
        name="supervisor",
        expected_path=str(REPOSITORY / "tools/cure_lite_v24_runtime_supervisor.py"),
    )
    _validate_archived_file_binding(
        archived["integration_tool"],
        name="integration_tool",
        expected_path=str(
            REPOSITORY / "tools/cure_lite_v24_user_systemd_integration.py"
        ),
    )
    if (
        spec["source_bindings"]["supervisor_file_sha256"]
        != archived["supervisor"]["file_sha256"]
        or spec["source_bindings"]["child_entry_file_sha256"]
        != archived["dummy_child"]["file_sha256"]
        or authorization["rendered_fragment"]["utf8_text"].encode("utf-8")
        != (
            Path(str(authorization["unit_directory"])) / UNIT_NAME
        ).read_bytes()
    ):
        raise PermissionError("dummy r2 archived executable/fragment lineage changed")
    observed_anchors = {
        "original_authorization": _root(
            ORIGINAL_AUTHORIZATION_PATH,
            authorization,
            fingerprint_field="authorization_fingerprint",
        ),
        "runtime_spec": _root(
            ORIGINAL_RUNTIME_SPEC_PATH,
            spec,
            fingerprint_field="runtime_spec_fingerprint",
        ),
        "integration_terminal": _root(
            ORIGINAL_TERMINAL_PATH,
            terminal,
            fingerprint_field="integration_terminal_fingerprint",
        ),
        "removal_state": _root(
            ORIGINAL_REMOVAL_STATE_PATH,
            removal_state,
            fingerprint_field="removal_state_fingerprint",
        ),
    }
    if observed_anchors != _ARCHIVED_EVIDENCE_ANCHORS:
        raise PermissionError("dummy r2 archived evidence anchor changed")
    current_required: dict[str, object] = {}
    for name in (
        "python",
        "realizer",
        "dummy_child",
        "systemd_path",
        "systemd_analyze",
        "systemctl",
    ):
        integration._validate_file_binding(archived[name])
        current_required[name] = dict(archived[name])
    # The archived integration-tool binding intentionally proves the exact
    # historical r2 producer.  Recovery itself imports the *current* module
    # for sealed I/O and read-only manager/path checks, so bind that current
    # dependency separately rather than silently trusting a changed helper.
    current_required["integration_recovery_library"] = (
        integration._file_binding(Path(integration.__file__).resolve())
    )
    integration._validate_file_binding(authorization["template_binding"])
    return {
        "authorization": authorization,
        "terminal": terminal,
        "removal_state": removal_state,
        "spec": spec,
        "spec_path": spec_path,
        "archived_executable_bindings": dict(archived),
        "current_required_executable_bindings": current_required,
    }


def _validated_recovery_path_policy(
    original: Mapping[str, object],
    observed: Mapping[str, object],
) -> dict[str, object]:
    """Allow only the daemon-reload regeneration inode already caused by r2."""

    original_body = dict(original)
    observed_body = dict(observed)
    original_rows_value = original_body.pop("ordered_unit_paths", None)
    observed_rows_value = observed_body.pop("ordered_unit_paths", None)
    if (
        original_body != observed_body
        or not isinstance(original_rows_value, list)
        or not isinstance(observed_rows_value, list)
    ):
        raise PermissionError("dummy r2 user-unit path policy changed")
    original_rows = {
        str(row["path"]): dict(row)
        for row in original_rows_value
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    observed_rows = {
        str(row["path"]): dict(row)
        for row in observed_rows_value
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    if (
        len(original_rows) != len(original_rows_value)
        or len(observed_rows) != len(observed_rows_value)
        or set(original_rows) != set(observed_rows)
    ):
        raise PermissionError("dummy r2 user-unit search path set changed")
    allowed = f"/run/user/{os.getuid()}/systemd/generator.late"
    for path, expected in original_rows.items():
        current = observed_rows[path]
        if path != allowed:
            if current != expected:
                raise PermissionError(
                    f"dummy r2 user-unit path identity changed:{path}"
                )
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
            raise PermissionError(
                "dummy r2 authorized daemon-reload generator identity changed"
            )
    return json.loads(integration.canonical_json(dict(observed)))


def _live_context(
    *,
    runner: integration.CommandRunner = integration.run_command,
    manager_reader: integration.ManagerReader = (
        integration.collect_manager_generation
    ),
) -> dict[str, object]:
    chain = _sealed_original_chain()
    authorization = chain["authorization"]
    fragment = Path(str(authorization["unit_directory"])) / UNIT_NAME
    policy = realizer.freeze_user_unit_path_policy(
        UNIT_NAME,
        runner=runner,
        allowed_fragment=fragment,
    )
    policy = _validated_recovery_path_policy(
        authorization["unit_path_policy"],
        policy,
    )
    manager = manager_reader()
    integration._validate_manager_generation(manager)
    if manager != authorization["manager_generation"]:
        raise PermissionError("dummy r2 manager generation changed")
    rendered = authorization["rendered_fragment"]
    plan = realizer.build_realization_plan(
        unit_name=UNIT_NAME,
        unit_directory=Path(str(authorization["unit_directory"])),
        fragment_text=str(rendered["utf8_text"]),
        expected_fragment_sha256=str(rendered["sha256"]),
        execute_authorized=True,
        removal_authorized=True,
    )
    state = realizer.query_unit_properties(UNIT_NAME, runner=runner)
    realizer.validate_realized_static_unit(plan, state)
    fragment_identity = integration._observed_fragment_identity(fragment)
    if (
        fragment != EXPECTED_FRAGMENT_PATH
        or fragment_identity["fragment_path"] != str(EXPECTED_FRAGMENT_PATH)
        or fragment_identity["fragment_sha256"] != EXPECTED_FRAGMENT_SHA256
        or fragment_identity["fragment_sha256"] != rendered["sha256"]
        or fragment_identity["device"] != EXPECTED_FRAGMENT_DEVICE
        or fragment_identity["inode"] != EXPECTED_FRAGMENT_INODE
        or state
        != {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "static",
            "FragmentPath": str(fragment),
            "DropInPaths": "",
            "Transient": "no",
            "Restart": "no",
            "NRestarts": "0",
            "NeedDaemonReload": "no",
        }
    ):
        raise PermissionError("dummy r2 fragment is not exact inactive static state")
    artifacts = chain["spec"]["artifacts"]
    for key, value in artifacts.items():
        path = Path(str(value))
        if key in {"root", "heartbeat_dir", "systemd_invocation_dir"}:
            if (
                not path.is_dir()
                or path.is_symlink()
                or path.resolve(strict=True) != path
                or stat.S_IMODE(path.stat().st_mode) != 0o700
                or (
                    key in {"heartbeat_dir", "systemd_invocation_dir"}
                    and any(path.iterdir())
                )
            ):
                raise PermissionError("dummy r2 runtime directories changed")
        elif os.path.lexists(path):
            raise PermissionError("dummy r2 reached supervisor/child evidence")
    return {
        **chain,
        "manager_generation": manager,
        "unit_path_policy": policy,
        "fragment_identity": fragment_identity,
        "inactive_static_state": state,
        "plan": plan,
    }


def _remove_exact_authorized_fragment(
    plan: realizer.RealizationPlan,
    *,
    expected_identity: Mapping[str, object],
    authorization: Mapping[str, object],
    intent: Mapping[str, object],
    on_action_started: Callable[[str], None],
) -> None:
    """Unlink only the authorization-bound inode at a fresh action time."""

    realizer.validate_integration_unit_name(plan.unit_name)
    identity_keys = {
        "fragment_path",
        "fragment_sha256",
        "device",
        "inode",
        "owner_uid",
        "mode",
        "nlink",
    }
    if (
        set(expected_identity) != identity_keys
        or plan.unit_name != UNIT_NAME
        or plan.fragment_path != EXPECTED_FRAGMENT_PATH
        or expected_identity.get("fragment_path") != str(plan.fragment_path)
        or expected_identity.get("fragment_sha256") != plan.fragment_sha256
        or expected_identity.get("fragment_sha256") != EXPECTED_FRAGMENT_SHA256
        or expected_identity.get("device") != EXPECTED_FRAGMENT_DEVICE
        or expected_identity.get("inode") != EXPECTED_FRAGMENT_INODE
        or plan.execute_authorized is not True
        or plan.removal_authorized is not True
    ):
        raise PermissionError("dummy fragment removal identity is not exact")

    directory_fd = realizer._open_verified_directory(
        plan.unit_directory,
        owner_uid=plan.owner_uid,
    )
    fragment_fd: int | None = None
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fragment_fd = os.open(plan.unit_name, flags, dir_fd=directory_fd)
        opened = os.fstat(fragment_fd)
        linked = os.stat(
            plan.unit_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )

        def _same_authorized_inode(value: os.stat_result) -> bool:
            return (
                stat.S_ISREG(value.st_mode)
                and value.st_dev == expected_identity["device"]
                and value.st_ino == expected_identity["inode"]
                and value.st_uid == expected_identity["owner_uid"]
                and stat.S_IMODE(value.st_mode) == expected_identity["mode"]
                and value.st_nlink == expected_identity["nlink"] == 1
            )

        if (
            not _same_authorized_inode(opened)
            or not _same_authorized_inode(linked)
            or opened.st_dev != linked.st_dev
            or opened.st_ino != linked.st_ino
            or realizer._fd_sha256(fragment_fd)
            != expected_identity["fragment_sha256"]
        ):
            raise PermissionError("dummy fragment inode changed after authorization")
        linked_again = os.stat(
            plan.unit_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not _same_authorized_inode(linked_again)
            or linked_again.st_dev != opened.st_dev
            or linked_again.st_ino != opened.st_ino
        ):
            raise PermissionError("dummy fragment changed before unlink")

        action_started = datetime.now(timezone.utc)
        issued = _timestamp(
            authorization.get("issued_at_utc"),
            name="recovery issuance",
        )
        intent_created = _timestamp(
            intent.get("created_at_utc"),
            name="recovery intent",
        )
        expires = _timestamp(
            authorization.get("expires_at_utc"),
            name="recovery expiry",
        )
        if not issued <= intent_created <= action_started <= expires:
            raise PermissionError(
                "dummy recovery authorization expired before unlink"
            )
        on_action_started(
            action_started.isoformat().replace("+00:00", "Z")
        )

        linked_final = os.stat(
            plan.unit_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not _same_authorized_inode(linked_final)
            or linked_final.st_dev != opened.st_dev
            or linked_final.st_ino != opened.st_ino
        ):
            raise PermissionError("dummy fragment changed at unlink")
        os.unlink(plan.unit_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        if fragment_fd is not None:
            os.close(fragment_fd)
        os.close(directory_fd)


def create_recovery_authorization(
    *,
    validity_seconds: int = 300,
    runner: integration.CommandRunner = integration.run_command,
    manager_reader: integration.ManagerReader = (
        integration.collect_manager_generation
    ),
) -> dict[str, object]:
    if (
        isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("recovery authorization validity must be in [1,300]")
    if any(
        os.path.lexists(path)
        for path in (
            RECOVERY_AUTHORIZATION_PATH,
            RECOVERY_INTENT_PATH,
            RECOVERY_TERMINAL_PATH,
        )
    ):
        raise FileExistsError("dummy recovery identity is already consumed")
    context = _live_context(runner=runner, manager_reader=manager_reader)
    original = context["authorization"]
    terminal = context["terminal"]
    removal_state = context["removal_state"]
    spec = context["spec"]
    issued = datetime.now(timezone.utc)
    action = {
        "ordinal": 0,
        "action": "remove-exact-runtime-static-fragment",
        "unit_name": UNIT_NAME,
        "fragment_path": context["fragment_identity"]["fragment_path"],
        "then": ["daemon-reload", "verify-not-found"],
    }
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "scenario_id": SCENARIO_ID,
        "unit_name": UNIT_NAME,
        "instruction_id": INSTRUCTION_ID,
        "authorization_basis": AUTHORIZATION_BASIS,
        "authorized_uid": os.getuid(),
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (
            issued + timedelta(seconds=validity_seconds)
        ).isoformat().replace("+00:00", "Z"),
        "archived_roots": {
            "original_authorization": _root(
                ORIGINAL_AUTHORIZATION_PATH,
                original,
                fingerprint_field="authorization_fingerprint",
            ),
            "runtime_spec": _root(
                context["spec_path"],
                spec,
                fingerprint_field="runtime_spec_fingerprint",
            ),
            "integration_terminal": _root(
                ORIGINAL_TERMINAL_PATH,
                terminal,
                fingerprint_field="integration_terminal_fingerprint",
            ),
            "removal_state": _root(
                ORIGINAL_REMOVAL_STATE_PATH,
                removal_state,
                fingerprint_field="removal_state_fingerprint",
            ),
        },
        "archived_executable_bindings": context[
            "archived_executable_bindings"
        ],
        "current_recovery_tool_binding": integration._file_binding(
            Path(__file__).resolve()
        ),
        "current_required_executable_bindings": context[
            "current_required_executable_bindings"
        ],
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
    }
    return integration._write_sealed(
        RECOVERY_AUTHORIZATION_PATH,
        body,
        fingerprint_field="recovery_authorization_fingerprint",
    )


def load_recovery_authorization(
    *,
    require_fresh: bool,
    runner: integration.CommandRunner = integration.run_command,
    manager_reader: integration.ManagerReader = (
        integration.collect_manager_generation
    ),
) -> tuple[dict[str, object], dict[str, object]]:
    authorization = integration._read_sealed(
        RECOVERY_AUTHORIZATION_PATH,
        fingerprint_field="recovery_authorization_fingerprint",
        schema=AUTHORIZATION_SCHEMA,
    )
    if set(authorization) != _AUTHORIZATION_KEYS:
        raise PermissionError("dummy recovery authorization keys changed")
    _no_payload(authorization)
    issued = _timestamp(
        authorization.get("issued_at_utc"),
        name="recovery issuance",
    )
    expires = _timestamp(
        authorization.get("expires_at_utc"),
        name="recovery expiry",
    )
    now = datetime.now(timezone.utc)
    expected_action = {
        "ordinal": 0,
        "action": "remove-exact-runtime-static-fragment",
        "unit_name": UNIT_NAME,
        "fragment_path": authorization["fragment_identity"]["fragment_path"],
        "then": ["daemon-reload", "verify-not-found"],
    }
    if (
        authorization.get("scenario_id") != SCENARIO_ID
        or authorization.get("unit_name") != UNIT_NAME
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis") != AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != os.getuid()
        or expires <= issued
        or expires - issued > timedelta(seconds=300)
        or issued > now
        or (require_fresh and now > expires)
        or authorization.get("authorized_action") != expected_action
        or authorization.get("remove_authorized") is not True
        or authorization.get("daemon_reload_authorized") is not True
        or authorization.get("not_found_verification_authorized") is not True
        or authorization.get("start_authorized") is not False
        or authorization.get("stop_authorized") is not False
        or authorization.get("enable_authorized") is not False
        or authorization.get("automatic_retry_authorized") is not False
    ):
        raise PermissionError("dummy recovery authorization is stale or changed")
    roots = authorization.get("archived_roots")
    if not isinstance(roots, Mapping) or set(roots) != {
        "original_authorization",
        "runtime_spec",
        "integration_terminal",
        "removal_state",
    }:
        raise PermissionError("dummy recovery archived root set changed")
    for name, root in roots.items():
        _validate_root(root, name=name)
    context = _live_context(runner=runner, manager_reader=manager_reader)
    observed_roots = {
        "original_authorization": _root(
            ORIGINAL_AUTHORIZATION_PATH,
            context["authorization"],
            fingerprint_field="authorization_fingerprint",
        ),
        "runtime_spec": _root(
            context["spec_path"],
            context["spec"],
            fingerprint_field="runtime_spec_fingerprint",
        ),
        "integration_terminal": _root(
            ORIGINAL_TERMINAL_PATH,
            context["terminal"],
            fingerprint_field="integration_terminal_fingerprint",
        ),
        "removal_state": _root(
            ORIGINAL_REMOVAL_STATE_PATH,
            context["removal_state"],
            fingerprint_field="removal_state_fingerprint",
        ),
    }
    if (
        dict(roots) != observed_roots
        or authorization.get("archived_executable_bindings")
        != context["archived_executable_bindings"]
        or authorization.get("current_required_executable_bindings")
        != context["current_required_executable_bindings"]
        or authorization.get("current_recovery_tool_binding")
        != integration._file_binding(Path(__file__).resolve())
        or authorization.get("manager_generation")
        != context["manager_generation"]
        or authorization.get("unit_path_policy")
        != context["unit_path_policy"]
        or authorization.get("fragment_identity")
        != context["fragment_identity"]
        or authorization.get("inactive_static_state")
        != context["inactive_static_state"]
    ):
        raise PermissionError("dummy recovery authorization/live closure changed")
    return authorization, context


def execute_recovery(
    *,
    execute: bool,
    timeout_seconds: float = 10.0,
    runner: integration.CommandRunner = integration.run_command,
    manager_reader: integration.ManagerReader = (
        integration.collect_manager_generation
    ),
) -> dict[str, object]:
    if not execute:
        raise PermissionError("explicit dummy recovery execution is required")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or float(timeout_seconds) <= 0.0
    ):
        raise ValueError("dummy recovery timeout must be positive")
    if os.path.lexists(RECOVERY_INTENT_PATH) or os.path.lexists(
        RECOVERY_TERMINAL_PATH
    ):
        raise FileExistsError("dummy recovery execution identity is consumed")
    authorization, context = load_recovery_authorization(
        require_fresh=True,
        runner=runner,
        manager_reader=manager_reader,
    )
    action = authorization["authorized_action"]
    intent = integration._write_sealed(
        RECOVERY_INTENT_PATH,
        {
            "schema_version": INTENT_SCHEMA,
            "created_at_utc": integration._utc_now(),
            "scenario_id": SCENARIO_ID,
            "unit_name": UNIT_NAME,
            "recovery_authorization_path": str(
                RECOVERY_AUTHORIZATION_PATH
            ),
            "recovery_authorization_file_sha256": integration.file_sha256(
                RECOVERY_AUTHORIZATION_PATH
            ),
            "recovery_authorization_fingerprint": authorization[
                "recovery_authorization_fingerprint"
            ],
            "fragment_identity": context["fragment_identity"],
            "inactive_static_state": context["inactive_static_state"],
            "authorized_action": action,
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_accessed": False,
        },
        fingerprint_field="recovery_intent_fingerprint",
    )
    completed: list[str] = []
    post_state: dict[str, str] | None = None
    action_started_at_utc: str | None = None
    error: BaseException | None = None

    def _record_action_started(value: str) -> None:
        nonlocal action_started_at_utc
        if action_started_at_utc is not None:
            raise RuntimeError("dummy recovery action-start trace repeated")
        action_started_at_utc = value

    try:
        _remove_exact_authorized_fragment(
            context["plan"],
            expected_identity=context["fragment_identity"],
            authorization=authorization,
            intent=intent,
            on_action_started=_record_action_started,
        )
        completed.append("remove-exact-runtime-static-fragment")
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
    fragment_absent = not os.path.lexists(
        context["plan"].fragment_path
    )
    terminal = integration._write_sealed(
        RECOVERY_TERMINAL_PATH,
        {
            "schema_version": TERMINAL_SCHEMA,
            "created_at_utc": integration._utc_now(),
            "scenario_id": SCENARIO_ID,
            "unit_name": UNIT_NAME,
            "recovery_authorization_fingerprint": authorization[
                "recovery_authorization_fingerprint"
            ],
            "recovery_intent_fingerprint": intent[
                "recovery_intent_fingerprint"
            ],
            "action_started_at_utc": action_started_at_utc,
            "completed_actions": completed,
            "fragment_absent": fragment_absent,
            "post_removal_unit_state": post_state,
            "passed": error is None and fragment_absent,
            "error_type": (
                type(error).__name__ if error is not None else None
            ),
            "error_message": str(error) if error is not None else None,
            "payload_authority": "none",
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "gpu_accessed": False,
        },
        fingerprint_field="recovery_terminal_fingerprint",
    )
    if set(intent) != _INTENT_KEYS or set(terminal) != _RECOVERY_TERMINAL_KEYS:
        raise RuntimeError("dummy recovery evidence schema changed")
    if error is not None:
        raise error
    if terminal["passed"] is not True:
        raise RuntimeError("dummy recovery did not reach exact terminal PASS")
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    authorize = sub.add_parser("authorize")
    authorize.add_argument("--validity-seconds", type=int, default=300)
    apply = sub.add_parser("apply")
    apply.add_argument("--execute-authorized-removal", action="store_true")
    apply.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "authorize":
        result = create_recovery_authorization(
            validity_seconds=args.validity_seconds
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
        execute=args.execute_authorized_removal,
        timeout_seconds=args.timeout_seconds,
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
