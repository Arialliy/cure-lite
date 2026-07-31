#!/usr/bin/env python3
"""Close the one retained actual-unit realization failure without replay.

This tool is deliberately separate from the already sealed realization tool.
It never installs a fragment and never reloads, starts, enables, stops, or
removes a unit.  It can only attest the exact retained fragment produced by
the historical failed realization and the single systemd-v255 representation
difference that caused that realization to stop:

``SuccessExitStatus=0`` in the rendered unit is reported as ``0`` by the live
manager, while the historical authorization expected the redundant ``0 0``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Callable, Mapping, Sequence

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import cure_lite_v24_actual_unit_realization as legacy


EVIDENCE_ROOT = (
    REPOSITORY
    / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
)
ORIGINAL_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_unit_realization_authorization.json"
)
FAILURE_TERMINAL_PATH = EVIDENCE_ROOT / "r2_unit_realization_terminal.json"
NORMAL_RECEIPT_PATH = EVIDENCE_ROOT / "r2_unit_realization_receipt.json"
RECOVERY_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_unit_realization_recovery_authorization.json"
)
RECOVERY_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_unit_realization_recovery_receipt.json"
)
RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_spec.json"
)
RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_runtime_launch_authorization.json"
)
RELEASE_CONSUMER_PATH = (
    REPOSITORY / "tools/cure_lite_v24_actual_runtime_release.py"
)

RECOVERY_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-actual-unit-realization-recovery-authorization-v1"
)
RECOVERY_RECEIPT_SCHEMA = (
    "cure-lite-v24-actual-unit-realization-recovery-receipt-v1"
)
RECOVERY_KIND = (
    "retained_static_fragment_success_exit_status_representation_closure"
)
RECOVERY_ACTIONS = [
    "verify-sealed-failed-realization-lineage",
    "verify-retained-static-fragment",
    "verify-success-exit-status-zero-representation",
    "seal-recovery-receipt",
]
RECOVERY_AUTHORIZATION_BASIS = legacy.AUTHORIZATION_BASIS

_SHA = legacy._SHA
_TERMINAL_KEYS = {
    "schema_version",
    "candidate",
    "stage_id",
    "attempt_id",
    "unit_name",
    "created_at_utc",
    "authorization_path",
    "authorization_fingerprint",
    "completed_actions",
    "fragment_identity",
    "fragment_may_remain",
    "automatic_removal_performed",
    "daemon_reload_attempted",
    "enable_attempted",
    "start_attempted",
    "remove_attempted",
    "error_type",
    "error_message",
    "passed",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "terminal_fingerprint",
}
_FRAGMENT_IDENTITY_KEYS = {
    "path",
    "file_sha256",
    "device",
    "inode",
    "owner_uid",
    "mode",
    "nlink",
}
_EVIDENCE_ROOT_KEYS = {
    "path",
    "file_sha256",
    "device",
    "inode",
    "size",
    "mtime_ns",
    "ctime_ns",
    "owner_uid",
    "mode",
    "nlink",
    "parent_device",
    "parent_inode",
    "schema_version",
    "fingerprint_field",
    "fingerprint",
}
_MUTATION_AUTHORITY = {
    "install_fragment_authorized": False,
    "daemon_reload_authorized": False,
    "enable_authorized": False,
    "start_authorized": False,
    "stop_authorized": False,
    "remove_authorized": False,
    "reset_failed_authorized": False,
    "runtime_spec_creation_authorized": False,
    "runtime_launch_authorization_authorized": False,
    "evidence_receipt_creation_authorized": True,
    "automatic_retry_authorized": False,
}
_AUTHORIZATION_KEYS = {
    "schema_version",
    "candidate",
    "stage_id",
    "attempt_id",
    "unit_name",
    "instruction_id",
    "authorization_basis",
    "authorized_uid",
    "created_at_utc",
    "issued_at_utc",
    "expires_at_utc",
    "recovery_kind",
    "actions",
    "mutation_authority",
    "original_authorization",
    "failure_terminal",
    "normal_receipt_path",
    "normal_receipt_absent_at_authorization",
    "runtime_spec_path",
    "runtime_spec_absent_at_authorization",
    "runtime_launch_authorization_path",
    "runtime_launch_authorization_absent_at_authorization",
    "manager_generation",
    "unit_path_policy",
    "fragment_identity",
    "full_static_shadow",
    "compatibility_exception",
    "source_bindings",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "authorization_fingerprint",
}
_RECEIPT_KEYS = {
    "schema_version",
    "candidate",
    "stage_id",
    "attempt_id",
    "unit_name",
    "instruction_id",
    "created_at_utc",
    "recovery_kind",
    "recovery_authorization_path",
    "recovery_authorization_file_sha256",
    "recovery_authorization_fingerprint",
    "recovery_authorization_root",
    "original_authorization",
    "failure_terminal",
    "authorization_path",
    "authorization_file_sha256",
    "authorization_fingerprint",
    "manager_generation",
    "unit_path_policy",
    "template_binding",
    "rendered_fragment",
    "runtime_spec_binding",
    "expected_future_runtime_spec_path",
    "runtime_spec_absent_at_receipt",
    "runtime_launch_authorization_path",
    "runtime_launch_authorization_absent_at_receipt",
    "normal_receipt_path",
    "normal_receipt_absent_at_receipt",
    "executable_bindings",
    "fragment_identity",
    "full_static_shadow",
    "compatibility_exception",
    "source_bindings",
    "completed_actions",
    "static",
    "enabled",
    "started",
    "removed",
    "recovery_install_attempted",
    "recovery_daemon_reload_attempted",
    "recovery_enable_attempted",
    "recovery_start_attempted",
    "recovery_stop_attempted",
    "recovery_remove_attempted",
    "normal_realization_passed",
    "recovery_acceptance_passed",
    "historical_failure_preserved",
    "passed",
    "payload_authority",
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
    "receipt_fingerprint",
}

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ManagerReader = Callable[[], dict[str, object]]
ShadowReader = Callable[[], Mapping[str, str]]
UnitPathPolicyReader = Callable[[Path], Mapping[str, object]]


def canonical_json(value: object) -> str:
    return legacy.canonical_json(value)


def stable_fingerprint(value: object) -> str:
    return legacy.stable_fingerprint(value)


def _deep_exact_equal(left: object, right: object) -> bool:
    return legacy._deep_exact_equal(left, right)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: object, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise PermissionError(f"{name} is not an exact UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PermissionError(f"{name} is not an exact UTC timestamp") from error
    if parsed.tzinfo is None:
        raise PermissionError(f"{name} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _require_absent(path: Path, *, name: str) -> None:
    if os.path.lexists(path):
        raise PermissionError(f"{name} must remain absent")


def _require_fixed_path(
    supplied: Path,
    expected: Path,
    *,
    name: str,
) -> None:
    if supplied.absolute() != expected.absolute():
        raise PermissionError(f"{name} differs from the fixed recovery path")


def _require_fixed_lineage_paths(
    *,
    recovery_authorization_path: Path | None = None,
    recovery_receipt_path: Path | None = None,
    original_authorization_path: Path | None = None,
    failure_terminal_path: Path | None = None,
    normal_receipt_path: Path | None = None,
    runtime_spec_path: Path | None = None,
    runtime_launch_authorization_path: Path | None = None,
) -> None:
    pairs = (
        (
            recovery_authorization_path,
            RECOVERY_AUTHORIZATION_PATH,
            "recovery authorization",
        ),
        (
            recovery_receipt_path,
            RECOVERY_RECEIPT_PATH,
            "recovery receipt",
        ),
        (
            original_authorization_path,
            ORIGINAL_AUTHORIZATION_PATH,
            "historical authorization",
        ),
        (
            failure_terminal_path,
            FAILURE_TERMINAL_PATH,
            "historical failure terminal",
        ),
        (normal_receipt_path, NORMAL_RECEIPT_PATH, "normal receipt"),
        (runtime_spec_path, RUNTIME_SPEC_PATH, "runtime spec"),
        (
            runtime_launch_authorization_path,
            RUNTIME_LAUNCH_AUTHORIZATION_PATH,
            "runtime launch authorization",
        ),
    )
    for supplied, expected, name in pairs:
        if supplied is not None:
            _require_fixed_path(supplied, expected, name=name)


def _evidence_generation(path: Path) -> dict[str, int]:
    target = Path(path).absolute()
    linked = target.lstat()
    parent = target.parent.lstat()
    if (
        not stat.S_ISREG(linked.st_mode)
        or stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
    ):
        raise PermissionError(
            "recovery predecessor path generation is unsafe",
        )
    return {
        "device": linked.st_dev,
        "inode": linked.st_ino,
        "size": linked.st_size,
        "mtime_ns": linked.st_mtime_ns,
        "ctime_ns": linked.st_ctime_ns,
        "owner_uid": linked.st_uid,
        "mode": stat.S_IMODE(linked.st_mode),
        "nlink": linked.st_nlink,
        "parent_device": parent.st_dev,
        "parent_inode": parent.st_ino,
    }


def _load_sealed_json_bound(
    path: Path,
    fingerprint_field: str,
) -> tuple[dict[str, object], dict[str, object]]:
    target = Path(path).absolute()
    generation_before = _evidence_generation(target)
    payload, legacy_identity = legacy._load_sealed_json_bound(
        target,
        fingerprint_field,
    )
    generation_after = _evidence_generation(target)
    if (
        not _deep_exact_equal(generation_before, generation_after)
        or any(
            legacy_identity[key] != generation_after[key]
            for key in (
                "device",
                "inode",
                "owner_uid",
                "mode",
                "nlink",
            )
        )
    ):
        raise PermissionError("recovery evidence changed while loading")
    identity = dict(legacy_identity)
    identity.update(generation_after)
    return payload, identity


def _sealed_root(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    target = Path(path).absolute()
    payload, identity = _load_sealed_json_bound(
        target,
        fingerprint_field,
    )
    if payload.get("schema_version") != schema:
        raise PermissionError("recovery predecessor schema changed")
    root = {
        "path": str(target),
        "file_sha256": identity["file_sha256"],
        **{
            key: identity[key]
            for key in _EVIDENCE_ROOT_KEYS
            if key
            not in {
                "path",
                "file_sha256",
                "schema_version",
                "fingerprint_field",
                "fingerprint",
            }
        },
        "schema_version": schema,
        "fingerprint_field": fingerprint_field,
        "fingerprint": payload[fingerprint_field],
    }
    return payload, identity, root


def _validate_root(
    value: object,
    *,
    fingerprint_field: str,
    schema: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_ROOT_KEYS:
        raise PermissionError("recovery evidence root schema changed")
    root = dict(value)
    if (
        root.get("schema_version") != schema
        or root.get("fingerprint_field") != fingerprint_field
        or not isinstance(root.get("path"), str)
        or not Path(str(root["path"])).is_absolute()
        or not isinstance(root.get("file_sha256"), str)
        or _SHA.fullmatch(str(root["file_sha256"])) is None
        or not isinstance(root.get("fingerprint"), str)
        or _SHA.fullmatch(str(root["fingerprint"])) is None
    ):
        raise PermissionError("recovery evidence root identity changed")
    payload, identity, observed = _sealed_root(
        Path(str(root["path"])),
        fingerprint_field=fingerprint_field,
        schema=schema,
    )
    if not _deep_exact_equal(observed, root):
        raise PermissionError("recovery evidence root was replaced")
    return payload, identity


def _preflight_recovery_authorization_root(
    receipt: Mapping[str, object],
    *,
    recovery_authorization_path: Path,
) -> dict[str, object]:
    if set(receipt) != _RECEIPT_KEYS:
        raise PermissionError("recovery receipt keys changed")
    value = receipt.get("recovery_authorization_root")
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_ROOT_KEYS:
        raise PermissionError(
            "recovery authorization root schema changed",
        )
    root = dict(value)
    root_path = root.get("path")
    expected = str(recovery_authorization_path.absolute())
    if (
        not isinstance(root_path, str)
        or not Path(root_path).is_absolute()
        or root_path != expected
    ):
        raise PermissionError(
            "recovery authorization root differs from the fixed path",
        )
    return root


def _validate_fragment_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FRAGMENT_IDENTITY_KEYS:
        raise PermissionError("retained fragment identity schema changed")
    result = dict(value)
    numeric = (
        result.get("device"),
        result.get("inode"),
        result.get("owner_uid"),
        result.get("mode"),
        result.get("nlink"),
    )
    if (
        not isinstance(result.get("path"), str)
        or not Path(str(result["path"])).is_absolute()
        or not isinstance(result.get("file_sha256"), str)
        or _SHA.fullmatch(str(result["file_sha256"])) is None
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in numeric
        )
        or result["owner_uid"] != os.getuid()
        or result["mode"] != 0o600
        or result["nlink"] != 1
    ):
        raise PermissionError("retained fragment identity is unsafe")
    return result


def _validate_original_authorization(
    path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    authorization, identity, root = _sealed_root(
        path,
        fingerprint_field="authorization_fingerprint",
        schema=legacy.AUTHORIZATION_SCHEMA,
    )
    if set(authorization) != legacy._AUTH_KEYS:
        raise PermissionError("historical realization authorization keys changed")
    if (
        authorization.get("candidate") != legacy.CANDIDATE
        or authorization.get("stage_id") != legacy.STAGE_ID
        or authorization.get("attempt_id") != legacy.ATTEMPT_ID
        or authorization.get("unit_name") != legacy.ACTUAL_UNIT
        or authorization.get("instruction_id") != legacy.INSTRUCTION_ID
        or authorization.get("authorization_basis")
        != legacy.AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != os.getuid()
        or authorization.get("actions") != legacy._ACTIONS
        or authorization.get("persistent_install_authorized") is not False
        or authorization.get("enable_authorized") is not False
        or authorization.get("start_authorized") is not False
        or authorization.get("remove_authorized") is not False
        or authorization.get("payload_authority") != "none"
        or any(
            authorization.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError("historical realization authorization changed")
    issued = _timestamp(
        authorization.get("issued_at_utc"),
        name="historical authorization issued_at_utc",
    )
    created = _timestamp(
        authorization.get("created_at_utc"),
        name="historical authorization created_at_utc",
    )
    expires = _timestamp(
        authorization.get("expires_at_utc"),
        name="historical authorization expires_at_utc",
    )
    if (
        not issued <= created <= expires
        or expires - issued > timedelta(minutes=5)
    ):
        raise PermissionError("historical authorization chronology changed")
    executable_bindings = authorization.get("executable_bindings")
    if (
        not isinstance(executable_bindings, Mapping)
        or set(executable_bindings)
        != {
            "realization_tool",
            "python",
            "supervisor",
            "systemd_path",
            "systemd_analyze",
            "systemctl",
        }
    ):
        raise PermissionError("historical executable closure changed")
    for binding in executable_bindings.values():
        if not isinstance(binding, Mapping):
            raise PermissionError("historical executable binding is malformed")
        legacy._validate_binding(binding)
    legacy._validate_python_binding(executable_bindings["python"])
    template_binding = authorization.get("template_binding")
    if not isinstance(template_binding, Mapping):
        raise PermissionError("historical template binding is absent")
    template_raw = legacy._validate_binding(template_binding)
    try:
        template_text = template_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PermissionError("historical template is not UTF-8") from error
    runtime_binding = authorization.get("runtime_spec_binding")
    if (
        not isinstance(runtime_binding, Mapping)
        or set(runtime_binding)
        != {
            "kind",
            "runtime_spec_path",
            "runtime_spec_parent_identity",
            "absent_at_authorization",
            "required_schema",
        }
        or runtime_binding.get("kind") != "future-absent-runtime-spec-v2"
        or runtime_binding.get("required_schema")
        != legacy.SUPERVISOR_SPEC_SCHEMA
        or runtime_binding.get("absent_at_authorization") is not True
    ):
        raise PermissionError("historical runtime-spec binding changed")
    runtime_spec = Path(str(runtime_binding["runtime_spec_path"]))
    if (
        not runtime_spec.is_absolute()
        or not _deep_exact_equal(
            legacy._path_row(runtime_spec.parent),
            runtime_binding["runtime_spec_parent_identity"],
        )
    ):
        raise PermissionError("historical runtime-spec parent changed")
    rendered = legacy.render_fragment(
        template_text,
        python_path=Path(str(executable_bindings["python"]["path"])),
        supervisor_path=Path(str(executable_bindings["supervisor"]["path"])),
        runtime_spec_path=runtime_spec,
    )
    rendered_binding = {
        "utf8_text": rendered,
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }
    unit_directory = Path(str(authorization.get("unit_directory")))
    if (
        not unit_directory.is_absolute()
        or not _deep_exact_equal(
            authorization.get("rendered_fragment"),
            rendered_binding,
        )
        or not _deep_exact_equal(
            authorization.get("expected_static_shadow"),
            legacy._expected_static_shadow(
                unit_directory / legacy.ACTUAL_UNIT,
            ),
        )
    ):
        raise PermissionError("historical rendered/static closure changed")
    legacy._validate_manager_generation(authorization["manager_generation"])
    return authorization, identity, root


def _validate_failure_terminal(
    path: Path,
    *,
    original_authorization: Mapping[str, object],
    original_authorization_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    terminal, identity, root = _sealed_root(
        path,
        fingerprint_field="terminal_fingerprint",
        schema=legacy.TERMINAL_SCHEMA,
    )
    if set(terminal) != _TERMINAL_KEYS:
        raise PermissionError("historical realization terminal keys changed")
    fragment = _validate_fragment_identity(terminal.get("fragment_identity"))
    issued = _timestamp(
        original_authorization.get("issued_at_utc"),
        name="historical authorization issued_at_utc",
    )
    created = _timestamp(
        original_authorization.get("created_at_utc"),
        name="historical authorization created_at_utc",
    )
    expires = _timestamp(
        original_authorization.get("expires_at_utc"),
        name="historical authorization expires_at_utc",
    )
    terminal_created = _timestamp(
        terminal.get("created_at_utc"),
        name="historical terminal created_at_utc",
    )
    if (
        not issued <= created <= terminal_created <= expires
        or terminal.get("candidate") != legacy.CANDIDATE
        or terminal.get("stage_id") != legacy.STAGE_ID
        or terminal.get("attempt_id") != legacy.ATTEMPT_ID
        or terminal.get("unit_name") != legacy.ACTUAL_UNIT
        or terminal.get("authorization_path")
        != str(original_authorization_path.absolute())
        or terminal.get("authorization_fingerprint")
        != original_authorization["authorization_fingerprint"]
        or terminal.get("completed_actions") != legacy._ACTIONS[:2]
        or terminal.get("fragment_may_remain") is not True
        or terminal.get("automatic_removal_performed") is not False
        or terminal.get("daemon_reload_attempted") is not True
        or terminal.get("enable_attempted") is not False
        or terminal.get("start_attempted") is not False
        or terminal.get("remove_attempted") is not False
        or terminal.get("error_type") != "PermissionError"
        or terminal.get("error_message")
        != "installed actual unit shadow is not exact static"
        or terminal.get("passed") is not False
        or terminal.get("payload_authority") != "none"
        or any(
            terminal.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError("historical realization failure is not exact")
    if fragment["file_sha256"] != original_authorization[
        "rendered_fragment"
    ]["sha256"]:
        raise PermissionError("historical terminal fragment SHA changed")
    return terminal, identity, root


def _source_bindings() -> dict[str, object]:
    return {
        "recovery_tool": legacy._file_binding(Path(__file__).resolve()),
        "release_consumer": legacy._file_binding(
            RELEASE_CONSUMER_PATH.resolve(),
        ),
    }


def _validate_source_bindings(value: object) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"recovery_tool", "release_consumer"}
    ):
        raise PermissionError("recovery source binding closure changed")
    expected_paths = {
        "recovery_tool": Path(__file__).resolve(),
        "release_consumer": RELEASE_CONSUMER_PATH.resolve(),
    }
    for name, expected in expected_paths.items():
        binding = value.get(name)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != str(expected)
        ):
            raise PermissionError(f"recovery {name} path changed")
        legacy._validate_binding(binding)
    return dict(value)


def _compatibility_shadow(
    shadow: Mapping[str, str],
    *,
    fragment_identity: Mapping[str, object],
    authorization: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    live = dict(shadow)
    expected = authorization["expected_static_shadow"]
    if (
        set(live) != set(legacy._SHADOW_PROPERTIES)
        or not isinstance(expected, Mapping)
        or expected.get("SuccessExitStatus") != "0 0"
        or live.get("SuccessExitStatus") != "0"
    ):
        raise PermissionError(
            "live shadow is not the exact SuccessExitStatus recovery case"
        )
    rendered = authorization["rendered_fragment"]["utf8_text"]
    directives = legacy._directives(str(rendered))
    if directives.get("SuccessExitStatus") != ["0"]:
        raise PermissionError("rendered SuccessExitStatus directive changed")
    projected = dict(live)
    projected["SuccessExitStatus"] = "0 0"
    validated_projection = legacy.validate_installed_shadow(
        projected,
        fragment_identity=fragment_identity,
        authorization=authorization,
    )
    full_static_shadow = dict(validated_projection)
    full_static_shadow["SuccessExitStatus"] = "0"
    compatibility = {
        "property": "SuccessExitStatus",
        "classification": (
            "systemd_v255_explicit_zero_default_zero_deduplication"
        ),
        "authorized_expected_value": "0 0",
        "live_observed_value": "0",
        "rendered_directive_values": ["0"],
        "semantic_success_exit_statuses": [0],
        "all_other_static_properties_exact": True,
        "projected_legacy_shadow_fingerprint": stable_fingerprint(
            validated_projection,
        ),
        "live_validated_shadow_fingerprint": stable_fingerprint(
            full_static_shadow,
        ),
    }
    return full_static_shadow, compatibility


def _live_fragment_identity(
    expected: Mapping[str, object],
) -> dict[str, object]:
    fragment = _validate_fragment_identity(expected)
    _, observed = legacy._stable_read_file(Path(str(fragment["path"])))
    live = {
        key: observed[key]
        for key in _FRAGMENT_IDENTITY_KEYS
    }
    if not _deep_exact_equal(live, fragment):
        raise PermissionError("retained fragment identity changed")
    return live


def _observe_recovery_closure(
    *,
    original_authorization: Mapping[str, object],
    failure_terminal: Mapping[str, object],
    shadow_reader: ShadowReader,
    manager_reader: ManagerReader,
    unit_path_policy_reader: UnitPathPolicyReader,
) -> dict[str, object]:
    expected_manager = original_authorization["manager_generation"]

    def require_manager() -> dict[str, object]:
        current = dict(manager_reader())
        legacy._validate_manager_generation(current)
        if not _deep_exact_equal(current, expected_manager):
            raise PermissionError("historical user-manager generation changed")
        return current

    fragment = _live_fragment_identity(
        failure_terminal["fragment_identity"],
    )
    if (
        fragment["path"]
        != str(
            Path(str(original_authorization["unit_directory"]))
            / legacy.ACTUAL_UNIT
        )
        or fragment["file_sha256"]
        != original_authorization["rendered_fragment"]["sha256"]
    ):
        raise PermissionError("retained fragment left its authorized identity")
    require_manager()
    policy = dict(unit_path_policy_reader(Path(str(fragment["path"]))))
    transitioned = legacy._validate_daemon_reload_path_policy_transition(
        original_authorization["unit_path_policy"],
        policy,
        runtime_directory=Path(
            str(original_authorization["unit_directory"]),
        ),
        authorized_uid=int(original_authorization["authorized_uid"]),
    )
    if not _deep_exact_equal(transitioned, policy):
        raise PermissionError("retained unit path policy changed")
    require_manager()
    live_shadow = dict(shadow_reader())
    full_static_shadow, compatibility = _compatibility_shadow(
        live_shadow,
        fragment_identity=fragment,
        authorization=original_authorization,
    )
    if (
        live_shadow.get("LoadState") != "loaded"
        or live_shadow.get("ActiveState") != "inactive"
        or live_shadow.get("SubState") != "dead"
        or live_shadow.get("UnitFileState") != "static"
        or live_shadow.get("NRestarts") != "0"
        or live_shadow.get("NeedDaemonReload") != "no"
    ):
        raise PermissionError(
            "retained unit is not inactive static with live NRestarts=0",
        )
    require_manager()
    policy_after = dict(
        unit_path_policy_reader(Path(str(fragment["path"]))),
    )
    if not _deep_exact_equal(policy_after, policy):
        raise PermissionError("retained unit path policy rotated")
    require_manager()
    return {
        "manager_generation": expected_manager,
        "unit_path_policy": policy_after,
        "fragment_identity": fragment,
        "full_static_shadow": full_static_shadow,
        "compatibility_exception": compatibility,
    }


def _default_observers(
    runner: CommandRunner,
) -> tuple[ShadowReader, UnitPathPolicyReader]:
    def shadow_reader() -> Mapping[str, str]:
        return legacy.query_shadow(runner=runner)

    def policy_reader(fragment: Path) -> Mapping[str, object]:
        return legacy._observe_unit_path_policy(
            runner=runner,
            allowed_fragment=fragment.absolute(),
        )

    return shadow_reader, policy_reader


def _check_absent_outputs(
    *,
    normal_receipt_path: Path,
    runtime_spec_path: Path,
    runtime_launch_authorization_path: Path,
) -> None:
    _require_absent(
        normal_receipt_path,
        name="normal realization receipt",
    )
    _require_absent(runtime_spec_path, name="runtime spec")
    _require_absent(
        runtime_launch_authorization_path,
        name="runtime launch authorization",
    )


def _require_same_observation(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> None:
    if not _deep_exact_equal(dict(first), dict(second)):
        raise PermissionError("recovery live observation changed before commit")


def create_recovery_authorization(
    authorization_path: Path,
    *,
    original_authorization_path: Path,
    failure_terminal_path: Path,
    normal_receipt_path: Path,
    runtime_spec_path: Path,
    runtime_launch_authorization_path: Path,
    validity_seconds: int = 300,
    runner: CommandRunner = subprocess.run,
    manager_reader: ManagerReader = legacy.collect_manager_generation,
) -> dict[str, object]:
    paths = (
        authorization_path,
        original_authorization_path,
        failure_terminal_path,
        normal_receipt_path,
        runtime_spec_path,
        runtime_launch_authorization_path,
    )
    if any(not Path(path).is_absolute() for path in paths):
        raise ValueError("all recovery paths must be absolute")
    _require_fixed_lineage_paths(
        recovery_authorization_path=authorization_path,
        original_authorization_path=original_authorization_path,
        failure_terminal_path=failure_terminal_path,
        normal_receipt_path=normal_receipt_path,
        runtime_spec_path=runtime_spec_path,
        runtime_launch_authorization_path=(
            runtime_launch_authorization_path
        ),
    )
    if (
        isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("recovery authorization validity must be in [1,300]")
    _require_absent(authorization_path, name="recovery authorization")
    _require_absent(
        RECOVERY_RECEIPT_PATH,
        name="recovery receipt",
    )
    _check_absent_outputs(
        normal_receipt_path=normal_receipt_path,
        runtime_spec_path=runtime_spec_path,
        runtime_launch_authorization_path=(
            runtime_launch_authorization_path
        ),
    )
    original, _, original_root = _validate_original_authorization(
        original_authorization_path,
    )
    terminal, _, terminal_root = _validate_failure_terminal(
        failure_terminal_path,
        original_authorization=original,
        original_authorization_path=original_authorization_path,
    )
    if Path(str(original["runtime_spec_binding"]["runtime_spec_path"])) != (
        runtime_spec_path
    ):
        raise PermissionError("recovery runtime-spec path changed")
    shadow_reader, policy_reader = _default_observers(runner)
    observation = _observe_recovery_closure(
        original_authorization=original,
        failure_terminal=terminal,
        shadow_reader=shadow_reader,
        manager_reader=manager_reader,
        unit_path_policy_reader=policy_reader,
    )
    source_bindings = _source_bindings()
    issued = datetime.now(timezone.utc)
    body = {
        "schema_version": RECOVERY_AUTHORIZATION_SCHEMA,
        "candidate": legacy.CANDIDATE,
        "stage_id": legacy.STAGE_ID,
        "attempt_id": legacy.ATTEMPT_ID,
        "unit_name": legacy.ACTUAL_UNIT,
        "instruction_id": legacy.INSTRUCTION_ID,
        "authorization_basis": RECOVERY_AUTHORIZATION_BASIS,
        "authorized_uid": os.getuid(),
        "created_at_utc": _utc_now(),
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (
            issued + timedelta(seconds=validity_seconds)
        ).isoformat().replace("+00:00", "Z"),
        "recovery_kind": RECOVERY_KIND,
        "actions": list(RECOVERY_ACTIONS),
        "mutation_authority": dict(_MUTATION_AUTHORITY),
        "original_authorization": original_root,
        "failure_terminal": terminal_root,
        "normal_receipt_path": str(normal_receipt_path),
        "normal_receipt_absent_at_authorization": True,
        "runtime_spec_path": str(runtime_spec_path),
        "runtime_spec_absent_at_authorization": True,
        "runtime_launch_authorization_path": str(
            runtime_launch_authorization_path,
        ),
        "runtime_launch_authorization_absent_at_authorization": True,
        **observation,
        "source_bindings": source_bindings,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    # Recheck sources and predecessors immediately before the sole write.
    _validate_source_bindings(source_bindings)
    _validate_root(
        original_root,
        fingerprint_field="authorization_fingerprint",
        schema=legacy.AUTHORIZATION_SCHEMA,
    )
    _validate_root(
        terminal_root,
        fingerprint_field="terminal_fingerprint",
        schema=legacy.TERMINAL_SCHEMA,
    )
    _check_absent_outputs(
        normal_receipt_path=normal_receipt_path,
        runtime_spec_path=runtime_spec_path,
        runtime_launch_authorization_path=(
            runtime_launch_authorization_path
        ),
    )
    second_observation = _observe_recovery_closure(
        original_authorization=original,
        failure_terminal=terminal,
        shadow_reader=shadow_reader,
        manager_reader=manager_reader,
        unit_path_policy_reader=policy_reader,
    )
    _require_same_observation(observation, second_observation)
    _validate_source_bindings(source_bindings)
    _validate_root(
        original_root,
        fingerprint_field="authorization_fingerprint",
        schema=legacy.AUTHORIZATION_SCHEMA,
    )
    _validate_root(
        terminal_root,
        fingerprint_field="terminal_fingerprint",
        schema=legacy.TERMINAL_SCHEMA,
    )
    _check_absent_outputs(
        normal_receipt_path=normal_receipt_path,
        runtime_spec_path=runtime_spec_path,
        runtime_launch_authorization_path=(
            runtime_launch_authorization_path
        ),
    )
    _require_absent(authorization_path, name="recovery authorization")
    _require_absent(
        RECOVERY_RECEIPT_PATH,
        name="recovery receipt",
    )
    commit_now = datetime.now(timezone.utc)
    if not issued <= commit_now <= issued + timedelta(
        seconds=validity_seconds,
    ):
        raise PermissionError(
            "recovery authorization expired before commit",
        )
    return legacy.write_create_once_json(
        authorization_path,
        body,
        fingerprint_field="authorization_fingerprint",
    )


def _validate_recovery_authorization(
    path: Path,
    *,
    require_fresh: bool,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    authorization, identity = _load_sealed_json_bound(
        path,
        "authorization_fingerprint",
    )
    if (
        set(authorization) != _AUTHORIZATION_KEYS
        or authorization.get("schema_version")
        != RECOVERY_AUTHORIZATION_SCHEMA
        or authorization.get("candidate") != legacy.CANDIDATE
        or authorization.get("stage_id") != legacy.STAGE_ID
        or authorization.get("attempt_id") != legacy.ATTEMPT_ID
        or authorization.get("unit_name") != legacy.ACTUAL_UNIT
        or authorization.get("instruction_id") != legacy.INSTRUCTION_ID
        or authorization.get("authorization_basis")
        != RECOVERY_AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != os.getuid()
        or authorization.get("recovery_kind") != RECOVERY_KIND
        or authorization.get("actions") != RECOVERY_ACTIONS
        or not _deep_exact_equal(
            authorization.get("mutation_authority"),
            _MUTATION_AUTHORITY,
        )
        or authorization.get("normal_receipt_absent_at_authorization")
        is not True
        or authorization.get("runtime_spec_absent_at_authorization")
        is not True
        or authorization.get(
            "runtime_launch_authorization_absent_at_authorization"
        )
        is not True
        or authorization.get("payload_authority") != "none"
        or any(
            authorization.get(field) is not False
            for field in (
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
    ):
        raise PermissionError("recovery authorization identity changed")
    issued = _timestamp(
        authorization.get("issued_at_utc"),
        name="recovery issued_at_utc",
    )
    created = _timestamp(
        authorization.get("created_at_utc"),
        name="recovery created_at_utc",
    )
    expires = _timestamp(
        authorization.get("expires_at_utc"),
        name="recovery expires_at_utc",
    )
    now = datetime.now(timezone.utc)
    if (
        not issued <= created <= expires
        or not timedelta(seconds=1)
        <= expires - issued
        <= timedelta(seconds=300)
        or (require_fresh and not issued <= now <= expires)
    ):
        raise PermissionError("recovery authorization chronology changed")
    _validate_source_bindings(authorization.get("source_bindings"))
    _require_fixed_lineage_paths(
        original_authorization_path=Path(
            str(authorization["original_authorization"]["path"]),
        ),
        failure_terminal_path=Path(
            str(authorization["failure_terminal"]["path"]),
        ),
        normal_receipt_path=Path(
            str(authorization["normal_receipt_path"]),
        ),
        runtime_spec_path=Path(str(authorization["runtime_spec_path"])),
        runtime_launch_authorization_path=Path(
            str(authorization["runtime_launch_authorization_path"]),
        ),
    )
    original, _ = _validate_root(
        authorization.get("original_authorization"),
        fingerprint_field="authorization_fingerprint",
        schema=legacy.AUTHORIZATION_SCHEMA,
    )
    terminal, _ = _validate_root(
        authorization.get("failure_terminal"),
        fingerprint_field="terminal_fingerprint",
        schema=legacy.TERMINAL_SCHEMA,
    )
    _validate_original_authorization(
        Path(str(authorization["original_authorization"]["path"])),
    )
    _validate_failure_terminal(
        Path(str(authorization["failure_terminal"]["path"])),
        original_authorization=original,
        original_authorization_path=Path(
            str(authorization["original_authorization"]["path"]),
        ),
    )
    if (
        authorization.get("runtime_spec_path")
        != original["runtime_spec_binding"]["runtime_spec_path"]
    ):
        raise PermissionError("recovery runtime-spec binding changed")
    _check_absent_outputs(
        normal_receipt_path=Path(
            str(authorization["normal_receipt_path"]),
        ),
        runtime_spec_path=Path(str(authorization["runtime_spec_path"])),
        runtime_launch_authorization_path=Path(
            str(authorization["runtime_launch_authorization_path"]),
        ),
    )
    return authorization, identity, original, terminal


def seal_recovery_receipt(
    recovery_authorization_path: Path,
    *,
    receipt_path: Path,
    runner: CommandRunner = subprocess.run,
    manager_reader: ManagerReader = legacy.collect_manager_generation,
) -> dict[str, object]:
    if not recovery_authorization_path.is_absolute() or not receipt_path.is_absolute():
        raise ValueError("recovery evidence paths must be absolute")
    _require_fixed_lineage_paths(
        recovery_authorization_path=recovery_authorization_path,
        recovery_receipt_path=receipt_path,
    )
    _require_absent(receipt_path, name="recovery receipt")
    (
        recovery_authorization,
        recovery_authorization_identity,
        original,
        terminal,
    ) = _validate_recovery_authorization(
        recovery_authorization_path,
        require_fresh=True,
    )
    (
        rooted_recovery_authorization,
        rooted_recovery_authorization_identity,
        recovery_authorization_root,
    ) = _sealed_root(
        recovery_authorization_path,
        fingerprint_field="authorization_fingerprint",
        schema=RECOVERY_AUTHORIZATION_SCHEMA,
    )
    if (
        not _deep_exact_equal(
            recovery_authorization,
            rooted_recovery_authorization,
        )
        or not _deep_exact_equal(
            recovery_authorization_identity,
            rooted_recovery_authorization_identity,
        )
    ):
        raise PermissionError(
            "recovery authorization changed before receipt observation",
        )
    shadow_reader, policy_reader = _default_observers(runner)
    observation = _observe_recovery_closure(
        original_authorization=original,
        failure_terminal=terminal,
        shadow_reader=shadow_reader,
        manager_reader=manager_reader,
        unit_path_policy_reader=policy_reader,
    )
    for field in (
        "manager_generation",
        "unit_path_policy",
        "fragment_identity",
        "full_static_shadow",
        "compatibility_exception",
    ):
        if not _deep_exact_equal(observation[field], recovery_authorization[field]):
            raise PermissionError(
                f"recovery authorization live {field} changed",
            )
    _validate_source_bindings(recovery_authorization["source_bindings"])
    body = {
        "schema_version": RECOVERY_RECEIPT_SCHEMA,
        "candidate": legacy.CANDIDATE,
        "stage_id": legacy.STAGE_ID,
        "attempt_id": legacy.ATTEMPT_ID,
        "unit_name": legacy.ACTUAL_UNIT,
        "instruction_id": legacy.INSTRUCTION_ID,
        "created_at_utc": _utc_now(),
        "recovery_kind": RECOVERY_KIND,
        "recovery_authorization_path": str(
            recovery_authorization_path.absolute(),
        ),
        "recovery_authorization_file_sha256": (
            recovery_authorization_identity["file_sha256"]
        ),
        "recovery_authorization_fingerprint": (
            recovery_authorization["authorization_fingerprint"]
        ),
        "recovery_authorization_root": recovery_authorization_root,
        "original_authorization": recovery_authorization[
            "original_authorization"
        ],
        "failure_terminal": recovery_authorization["failure_terminal"],
        "authorization_path": recovery_authorization[
            "original_authorization"
        ]["path"],
        "authorization_file_sha256": recovery_authorization[
            "original_authorization"
        ]["file_sha256"],
        "authorization_fingerprint": original[
            "authorization_fingerprint"
        ],
        "manager_generation": observation["manager_generation"],
        "unit_path_policy": observation["unit_path_policy"],
        "template_binding": original["template_binding"],
        "rendered_fragment": original["rendered_fragment"],
        "runtime_spec_binding": original["runtime_spec_binding"],
        "expected_future_runtime_spec_path": recovery_authorization[
            "runtime_spec_path"
        ],
        "runtime_spec_absent_at_receipt": True,
        "runtime_launch_authorization_path": recovery_authorization[
            "runtime_launch_authorization_path"
        ],
        "runtime_launch_authorization_absent_at_receipt": True,
        "normal_receipt_path": recovery_authorization[
            "normal_receipt_path"
        ],
        "normal_receipt_absent_at_receipt": True,
        "executable_bindings": original["executable_bindings"],
        "fragment_identity": observation["fragment_identity"],
        "full_static_shadow": observation["full_static_shadow"],
        "compatibility_exception": observation[
            "compatibility_exception"
        ],
        "source_bindings": recovery_authorization["source_bindings"],
        "completed_actions": list(RECOVERY_ACTIONS),
        "static": True,
        "enabled": False,
        "started": False,
        "removed": False,
        "recovery_install_attempted": False,
        "recovery_daemon_reload_attempted": False,
        "recovery_enable_attempted": False,
        "recovery_start_attempted": False,
        "recovery_stop_attempted": False,
        "recovery_remove_attempted": False,
        "normal_realization_passed": False,
        "recovery_acceptance_passed": True,
        "historical_failure_preserved": True,
        "passed": True,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    _check_absent_outputs(
        normal_receipt_path=Path(str(body["normal_receipt_path"])),
        runtime_spec_path=Path(
            str(body["expected_future_runtime_spec_path"]),
        ),
        runtime_launch_authorization_path=Path(
            str(body["runtime_launch_authorization_path"]),
        ),
    )
    second_observation = _observe_recovery_closure(
        original_authorization=original,
        failure_terminal=terminal,
        shadow_reader=shadow_reader,
        manager_reader=manager_reader,
        unit_path_policy_reader=policy_reader,
    )
    _require_same_observation(observation, second_observation)
    _validate_source_bindings(recovery_authorization["source_bindings"])
    _validate_root(
        recovery_authorization["original_authorization"],
        fingerprint_field="authorization_fingerprint",
        schema=legacy.AUTHORIZATION_SCHEMA,
    )
    _validate_root(
        recovery_authorization["failure_terminal"],
        fingerprint_field="terminal_fingerprint",
        schema=legacy.TERMINAL_SCHEMA,
    )
    refreshed_authorization, refreshed_identity, _, _ = (
        _validate_recovery_authorization(
            recovery_authorization_path,
            require_fresh=True,
        )
    )
    if (
        not _deep_exact_equal(
            recovery_authorization,
            refreshed_authorization,
        )
        or not _deep_exact_equal(
            recovery_authorization_identity,
            refreshed_identity,
        )
    ):
        raise PermissionError("recovery authorization changed before commit")
    rooted_authorization, rooted_identity = _validate_root(
        recovery_authorization_root,
        fingerprint_field="authorization_fingerprint",
        schema=RECOVERY_AUTHORIZATION_SCHEMA,
    )
    if (
        not _deep_exact_equal(
            recovery_authorization,
            rooted_authorization,
        )
        or not _deep_exact_equal(
            recovery_authorization_identity,
            rooted_identity,
        )
    ):
        raise PermissionError(
            "recovery authorization root changed before commit",
        )
    _check_absent_outputs(
        normal_receipt_path=Path(str(body["normal_receipt_path"])),
        runtime_spec_path=Path(
            str(body["expected_future_runtime_spec_path"]),
        ),
        runtime_launch_authorization_path=Path(
            str(body["runtime_launch_authorization_path"]),
        ),
    )
    _require_absent(receipt_path, name="recovery receipt")
    final_rooted_authorization, final_rooted_identity = _validate_root(
        recovery_authorization_root,
        fingerprint_field="authorization_fingerprint",
        schema=RECOVERY_AUTHORIZATION_SCHEMA,
    )
    if (
        not _deep_exact_equal(
            recovery_authorization,
            final_rooted_authorization,
        )
        or not _deep_exact_equal(
            recovery_authorization_identity,
            final_rooted_identity,
        )
    ):
        raise PermissionError(
            "recovery authorization root changed at receipt commit",
        )
    commit_now = datetime.now(timezone.utc)
    if not _timestamp(
        recovery_authorization["issued_at_utc"],
        name="recovery issued_at_utc",
    ) <= commit_now <= _timestamp(
        recovery_authorization["expires_at_utc"],
        name="recovery expires_at_utc",
    ):
        raise PermissionError("recovery authorization expired before receipt")
    return legacy.write_create_once_json(
        receipt_path,
        body,
        fingerprint_field="receipt_fingerprint",
    )


def validate_release_recovery_chain(
    *,
    recovery_authorization_path: Path,
    recovery_receipt_path: Path,
    shadow_reader: ShadowReader,
    manager_reader: ManagerReader,
    unit_path_policy_reader: UnitPathPolicyReader,
) -> dict[str, object]:
    _require_fixed_lineage_paths(
        recovery_authorization_path=recovery_authorization_path,
        recovery_receipt_path=recovery_receipt_path,
    )
    (
        recovery_authorization,
        recovery_authorization_identity,
        original,
        terminal,
    ) = _validate_recovery_authorization(
        recovery_authorization_path,
        require_fresh=False,
    )
    receipt, receipt_identity = _load_sealed_json_bound(
        recovery_receipt_path,
        "receipt_fingerprint",
    )
    recovery_authorization_root = (
        _preflight_recovery_authorization_root(
            receipt,
            recovery_authorization_path=recovery_authorization_path,
        )
    )
    rooted_authorization, rooted_authorization_identity = _validate_root(
        recovery_authorization_root,
        fingerprint_field="authorization_fingerprint",
        schema=RECOVERY_AUTHORIZATION_SCHEMA,
    )
    if (
        set(receipt) != _RECEIPT_KEYS
        or receipt.get("schema_version") != RECOVERY_RECEIPT_SCHEMA
        or receipt.get("candidate") != legacy.CANDIDATE
        or receipt.get("stage_id") != legacy.STAGE_ID
        or receipt.get("attempt_id") != legacy.ATTEMPT_ID
        or receipt.get("unit_name") != legacy.ACTUAL_UNIT
        or receipt.get("instruction_id") != legacy.INSTRUCTION_ID
        or receipt.get("recovery_kind") != RECOVERY_KIND
        or receipt.get("recovery_authorization_path")
        != str(recovery_authorization_path.absolute())
        or receipt.get("recovery_authorization_file_sha256")
        != recovery_authorization_identity["file_sha256"]
        or receipt.get("recovery_authorization_fingerprint")
        != recovery_authorization["authorization_fingerprint"]
        or recovery_authorization_root["path"]
        != str(recovery_authorization_path.absolute())
        or not _deep_exact_equal(
            rooted_authorization,
            recovery_authorization,
        )
        or not _deep_exact_equal(
            rooted_authorization_identity,
            recovery_authorization_identity,
        )
        or not _deep_exact_equal(
            receipt.get("original_authorization"),
            recovery_authorization["original_authorization"],
        )
        or not _deep_exact_equal(
            receipt.get("failure_terminal"),
            recovery_authorization["failure_terminal"],
        )
        or receipt.get("authorization_path")
        != recovery_authorization["original_authorization"]["path"]
        or receipt.get("authorization_file_sha256")
        != recovery_authorization["original_authorization"]["file_sha256"]
        or receipt.get("authorization_fingerprint")
        != original["authorization_fingerprint"]
        or receipt.get("expected_future_runtime_spec_path")
        != recovery_authorization["runtime_spec_path"]
        or receipt.get("runtime_spec_absent_at_receipt") is not True
        or receipt.get("runtime_launch_authorization_path")
        != recovery_authorization["runtime_launch_authorization_path"]
        or receipt.get(
            "runtime_launch_authorization_absent_at_receipt"
        )
        is not True
        or receipt.get("normal_receipt_path")
        != recovery_authorization["normal_receipt_path"]
        or receipt.get("normal_receipt_absent_at_receipt") is not True
        or not _deep_exact_equal(
            receipt.get("template_binding"),
            original["template_binding"],
        )
        or not _deep_exact_equal(
            receipt.get("rendered_fragment"),
            original["rendered_fragment"],
        )
        or not _deep_exact_equal(
            receipt.get("runtime_spec_binding"),
            original["runtime_spec_binding"],
        )
        or not _deep_exact_equal(
            receipt.get("executable_bindings"),
            original["executable_bindings"],
        )
        or not _deep_exact_equal(
            receipt.get("source_bindings"),
            recovery_authorization["source_bindings"],
        )
        or receipt.get("completed_actions") != RECOVERY_ACTIONS
        or receipt.get("static") is not True
        or receipt.get("enabled") is not False
        or receipt.get("started") is not False
        or receipt.get("removed") is not False
        or receipt.get("normal_realization_passed") is not False
        or receipt.get("recovery_acceptance_passed") is not True
        or receipt.get("historical_failure_preserved") is not True
        or any(
            receipt.get(field) is not False
            for field in (
                "recovery_install_attempted",
                "recovery_daemon_reload_attempted",
                "recovery_enable_attempted",
                "recovery_start_attempted",
                "recovery_stop_attempted",
                "recovery_remove_attempted",
                "D_R_payload_accessed",
                "D_V_payload_accessed",
                "D_T_payload_accessed",
            )
        )
        or receipt.get("passed") is not True
        or receipt.get("payload_authority") != "none"
    ):
        raise PermissionError("recovery receipt identity changed")
    issued = _timestamp(
        recovery_authorization["issued_at_utc"],
        name="recovery issued_at_utc",
    )
    authorized_created = _timestamp(
        recovery_authorization["created_at_utc"],
        name="recovery authorization created_at_utc",
    )
    receipt_created = _timestamp(
        receipt.get("created_at_utc"),
        name="recovery receipt created_at_utc",
    )
    expires = _timestamp(
        recovery_authorization["expires_at_utc"],
        name="recovery expires_at_utc",
    )
    if not issued <= authorized_created <= receipt_created <= expires:
        raise PermissionError("recovery authorization/receipt chronology changed")
    observation = _observe_recovery_closure(
        original_authorization=original,
        failure_terminal=terminal,
        shadow_reader=shadow_reader,
        manager_reader=manager_reader,
        unit_path_policy_reader=unit_path_policy_reader,
    )
    for field in (
        "manager_generation",
        "unit_path_policy",
        "fragment_identity",
        "full_static_shadow",
        "compatibility_exception",
    ):
        if (
            not _deep_exact_equal(
                receipt.get(field),
                recovery_authorization.get(field),
            )
            or not _deep_exact_equal(receipt.get(field), observation[field])
        ):
            raise PermissionError(f"recovery live {field} changed")
    _validate_source_bindings(receipt["source_bindings"])
    _check_absent_outputs(
        normal_receipt_path=Path(str(receipt["normal_receipt_path"])),
        runtime_spec_path=Path(
            str(receipt["expected_future_runtime_spec_path"]),
        ),
        runtime_launch_authorization_path=Path(
            str(receipt["runtime_launch_authorization_path"]),
        ),
    )
    refreshed_receipt, refreshed_receipt_identity = (
        _load_sealed_json_bound(
            recovery_receipt_path,
            "receipt_fingerprint",
        )
    )
    if (
        not _deep_exact_equal(receipt, refreshed_receipt)
        or not _deep_exact_equal(
            receipt_identity,
            refreshed_receipt_identity,
        )
    ):
        raise PermissionError("recovery receipt changed during validation")
    final_rooted_authorization, final_rooted_identity = _validate_root(
        recovery_authorization_root,
        fingerprint_field="authorization_fingerprint",
        schema=RECOVERY_AUTHORIZATION_SCHEMA,
    )
    if (
        not _deep_exact_equal(
            recovery_authorization,
            final_rooted_authorization,
        )
        or not _deep_exact_equal(
            recovery_authorization_identity,
            final_rooted_identity,
        )
    ):
        raise PermissionError(
            "recovery authorization root changed before second observation",
        )
    second_observation = _observe_recovery_closure(
        original_authorization=original,
        failure_terminal=terminal,
        shadow_reader=shadow_reader,
        manager_reader=manager_reader,
        unit_path_policy_reader=unit_path_policy_reader,
    )
    _require_same_observation(observation, second_observation)
    _validate_source_bindings(receipt["source_bindings"])
    _validate_root(
        receipt["original_authorization"],
        fingerprint_field="authorization_fingerprint",
        schema=legacy.AUTHORIZATION_SCHEMA,
    )
    _validate_root(
        receipt["failure_terminal"],
        fingerprint_field="terminal_fingerprint",
        schema=legacy.TERMINAL_SCHEMA,
    )
    refreshed_authorization, refreshed_identity, _, _ = (
        _validate_recovery_authorization(
            recovery_authorization_path,
            require_fresh=False,
        )
    )
    if (
        not _deep_exact_equal(
            recovery_authorization,
            refreshed_authorization,
        )
        or not _deep_exact_equal(
            recovery_authorization_identity,
            refreshed_identity,
        )
    ):
        raise PermissionError("recovery authorization changed during validation")
    final_rooted_authorization, final_rooted_identity = _validate_root(
        recovery_authorization_root,
        fingerprint_field="authorization_fingerprint",
        schema=RECOVERY_AUTHORIZATION_SCHEMA,
    )
    if (
        not _deep_exact_equal(
            recovery_authorization,
            final_rooted_authorization,
        )
        or not _deep_exact_equal(
            recovery_authorization_identity,
            final_rooted_identity,
        )
    ):
        raise PermissionError(
            "recovery authorization root changed during validation",
        )
    _check_absent_outputs(
        normal_receipt_path=Path(str(receipt["normal_receipt_path"])),
        runtime_spec_path=Path(
            str(receipt["expected_future_runtime_spec_path"]),
        ),
        runtime_launch_authorization_path=Path(
            str(receipt["runtime_launch_authorization_path"]),
        ),
    )
    refreshed_receipt, refreshed_receipt_identity = (
        _load_sealed_json_bound(
            recovery_receipt_path,
            "receipt_fingerprint",
        )
    )
    if (
        not _deep_exact_equal(receipt, refreshed_receipt)
        or not _deep_exact_equal(
            receipt_identity,
            refreshed_receipt_identity,
        )
    ):
        raise PermissionError("recovery receipt changed during validation")
    committed_rooted_authorization, committed_rooted_identity = (
        _validate_root(
            recovery_authorization_root,
            fingerprint_field="authorization_fingerprint",
            schema=RECOVERY_AUTHORIZATION_SCHEMA,
        )
    )
    if (
        not _deep_exact_equal(
            recovery_authorization,
            committed_rooted_authorization,
        )
        or not _deep_exact_equal(
            recovery_authorization_identity,
            committed_rooted_identity,
        )
    ):
        raise PermissionError(
            "recovery authorization root changed at validation commit",
        )
    # The top-level shape is intentionally identical to the normal release
    # closure.  The old authorization remains the source of executable and
    # template identity, while the two environment input identities bind the
    # new recovery authorization and receipt paths passed to build-spec.
    return {
        "authorization": original,
        "receipt": receipt,
        "authorization_identity": recovery_authorization_identity,
        "receipt_identity": receipt_identity,
        "live_unit_path_policy": observation["unit_path_policy"],
        "live_shadow": {
            key: str(receipt["full_static_shadow"][key])
            for key in legacy._SHADOW_PROPERTIES
        },
        "validated_shadow": receipt["full_static_shadow"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize-recovery")
    authorize.add_argument("--validity-seconds", type=int, default=300)
    subparsers.add_parser("seal-receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "authorize-recovery":
        create_recovery_authorization(
            RECOVERY_AUTHORIZATION_PATH,
            original_authorization_path=ORIGINAL_AUTHORIZATION_PATH,
            failure_terminal_path=FAILURE_TERMINAL_PATH,
            normal_receipt_path=NORMAL_RECEIPT_PATH,
            runtime_spec_path=RUNTIME_SPEC_PATH,
            runtime_launch_authorization_path=(
                RUNTIME_LAUNCH_AUTHORIZATION_PATH
            ),
            validity_seconds=arguments.validity_seconds,
        )
    else:
        seal_recovery_receipt(
            RECOVERY_AUTHORIZATION_PATH,
            receipt_path=RECOVERY_RECEIPT_PATH,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
