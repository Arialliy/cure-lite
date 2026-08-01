#!/usr/bin/env python3
"""Runtime-supervisor wrapper for v24 compatibility generation c5.

The audited c1 supervisor remains the implementation authority.  This module
loads that exact source generation, replaces only runtime compatibility
identities, and puts a c5 source-generation guard in front of every inherited
runtime entry point.  Scientific attempt ordinal and scientific result paths
remain the original r2 identities.

``describe_compatibility_identity`` is deliberately metadata-only.  It is the
only identity boundary that remains usable while the c5 bridge hash is still a
placeholder.  Every production verifier and CLI path fails closed until the
bridge hash is frozen in this source and all three loaded source generations
(self, c1, and bridge) remain unchanged.
"""

from __future__ import annotations

from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType
from typing import Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()
RUNS_ROOT = (REPOSITORY / "runs/irstd1k_stage_a_seed42").resolve()
if str(REPOSITORY) not in sys.path:
    # Production invokes this file with Python -I -S.  Add only the exact
    # repository root needed by the sealed c5 bridge import.
    sys.path.insert(0, str(REPOSITORY))

FROZEN_COMPAT_C1_SUPERVISOR_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c1.py"
).resolve()
FROZEN_COMPAT_C1_SUPERVISOR_SHA256 = (
    "7e2182da4f818bda5567c677194a1daf4cf02ce6874754acbc6b42095bd77447"
)

COMPATIBILITY_GENERATION = "c5"
COMPAT_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c5.service"
)
COMPAT_RUNTIME_SPEC_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_preaccess_compat_c5_runtime_spec.json"
    ).resolve()
)
COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_preaccess_compat_c5_"
        "runtime_launch_authorization.json"
    ).resolve()
)
COMPAT_RUNTIME_ARTIFACT_ROOT = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c5_runtime_artifacts"
    ).resolve()
)
COMPAT_GPU_LEASE_ROOT = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c5_gpu_lease"
    ).resolve()
)
# Compatibility-shaped scientific output paths are forbidden aliases.  The
# frozen r2 adapter continues to own the original scientific paths below.
COMPAT_RUN_ROOT_ALIAS_PATH = str(
    (
        RUNS_ROOT
        / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c5"
    ).resolve()
)
COMPAT_RESULT_RECEIPT_ALIAS_PATH = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c5_receipt.json"
    ).resolve()
)
SCIENTIFIC_RUN_ROOT = str(
    (RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2").resolve()
)
SCIENTIFIC_RESULT_RECEIPT_PATH = str(
    (EVIDENCE_ROOT / "D_R_structural_attempt_r2_receipt.json").resolve()
)
COMPAT_ADAPTER_PATH = str(
    (
        REPOSITORY
        / "tools/"
        "run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c5.py"
    ).resolve()
)
COMPAT_SUPERVISOR_PATH = str(Path(__file__).resolve())
COMPATIBILITY_RECEIPT_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "r2_preaccess_schema_compat_c5_receipt.json"
    ).resolve()
)
COMPAT_POLICY_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c5.py"
).resolve()

# This value is replaced only after the c5 bridge and its tests are frozen.
# A fresh interpreter must load the final source; monkey-patching this sentinel
# in a live process never promotes that process to production-ready.
COMPAT_POLICY_SOURCE_SHA256 = (
    "388843b9b840db41610d57543f4982666cdf442ba81fa5acb208033de062319f"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FILE_IDENTITY_FIELDS = (
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
_PARENT_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
    "st_nlink",
)

_C1_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service"
)
_OLD_UNIT_NAME = "cure-lite-v24-gcr-pacre-dr-r2.service"
_C2_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service"
)
_C3_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c3.service"
)
_C4_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service"
)
_C1_RUNTIME_SPEC_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json"
    ).resolve()
)
_C2_RUNTIME_SPEC_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_spec.json"
    ).resolve()
)
_C3_RUNTIME_SPEC_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_preaccess_compat_c3_runtime_spec.json"
    ).resolve()
)
_C4_RUNTIME_SPEC_PATH = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_spec.json"
    ).resolve()
)
_C1_RUNTIME_LAUNCH_AUTHORIZATION_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_preaccess_compat_c1_"
        "runtime_launch_authorization.json"
    ).resolve()
)
_C2_RUNTIME_LAUNCH_AUTHORIZATION_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_preaccess_compat_c2_"
        "runtime_launch_authorization.json"
    ).resolve()
)
_C3_RUNTIME_LAUNCH_AUTHORIZATION_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_preaccess_compat_c3_"
        "runtime_launch_authorization.json"
    ).resolve()
)
_C4_RUNTIME_LAUNCH_AUTHORIZATION_PATH = str(
    (
        EVIDENCE_ROOT
        / (
            "D_R_structural_attempt_r2_preaccess_compat_c4_"
            "runtime_launch_authorization.json"
        )
    ).resolve()
)
_OLD_RUNTIME_SPEC_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_runtime_spec.json"
    ).resolve()
)
_OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_runtime_launch_authorization.json"
    ).resolve()
)
_OLD_RUNTIME_ARTIFACT_ROOT = str(
    (EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_artifacts").resolve()
)
_OLD_GPU_LEASE_ROOT = str(
    (EVIDENCE_ROOT / "D_R_structural_attempt_r2_gpu_lease").resolve()
)
_C1_RUNTIME_ARTIFACT_ROOT = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_artifacts"
    ).resolve()
)
_C2_RUNTIME_ARTIFACT_ROOT = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_artifacts"
    ).resolve()
)
_C3_RUNTIME_ARTIFACT_ROOT = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c3_runtime_artifacts"
    ).resolve()
)
_C4_RUNTIME_ARTIFACT_ROOT = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_artifacts"
    ).resolve()
)
_C1_GPU_LEASE_ROOT = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c1_gpu_lease"
    ).resolve()
)
_C2_GPU_LEASE_ROOT = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c2_gpu_lease"
    ).resolve()
)
_C3_GPU_LEASE_ROOT = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c3_gpu_lease"
    ).resolve()
)
_C4_GPU_LEASE_ROOT = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c4_gpu_lease"
    ).resolve()
)
_C1_RUN_ROOT_ALIAS_PATH = str(
    (
        RUNS_ROOT
        / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c1"
    ).resolve()
)
_C2_RUN_ROOT_ALIAS_PATH = str(
    (
        RUNS_ROOT
        / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c2"
    ).resolve()
)
_C3_RUN_ROOT_ALIAS_PATH = str(
    (
        RUNS_ROOT
        / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c3"
    ).resolve()
)
_C4_RUN_ROOT_ALIAS_PATH = str(
    (
        RUNS_ROOT
        / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c4"
    ).resolve()
)
_C1_RESULT_RECEIPT_ALIAS_PATH = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c1_receipt.json"
    ).resolve()
)
_C2_RESULT_RECEIPT_ALIAS_PATH = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c2_receipt.json"
    ).resolve()
)
_C3_RESULT_RECEIPT_ALIAS_PATH = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c3_receipt.json"
    ).resolve()
)
_C4_RESULT_RECEIPT_ALIAS_PATH = str(
    (
        EVIDENCE_ROOT
        / "D_R_structural_attempt_r2_preaccess_compat_c4_receipt.json"
    ).resolve()
)
_C1_RECEIPT_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "r2_preaccess_schema_compat_c1_receipt.json"
    ).resolve()
)


_C2_RECEIPT_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "r2_preaccess_schema_compat_c2_receipt.json"
    ).resolve()
)
_C3_RECEIPT_PATH = str(
    (
        REPOSITORY
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "r2_preaccess_schema_compat_c3_receipt.json"
    ).resolve()
)
_C4_RECEIPT_PATH = str(
    (EVIDENCE_ROOT / "r2_preaccess_schema_compat_c4_receipt.json").resolve()
)
_C1_ADAPTER_PATH = str(
    (
        REPOSITORY
        / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c1.py"
    ).resolve()
)
_C2_ADAPTER_PATH = str(
    (
        REPOSITORY
        / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c2.py"
    ).resolve()
)
_C3_ADAPTER_PATH = str(
    (
        REPOSITORY
        / "tools/"
        "run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c3.py"
    ).resolve()
)
_C4_ADAPTER_PATH = str(
    (
        REPOSITORY
        / "tools/"
        "run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c4.py"
    ).resolve()
)
_C1_SUPERVISOR_PATH = str(
    (
        REPOSITORY
        / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c1.py"
    ).resolve()
)
_C2_SUPERVISOR_PATH = str(
    (
        REPOSITORY
        / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c2.py"
    ).resolve()
)
_C3_SUPERVISOR_PATH = str(
    (
        REPOSITORY
        / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c3.py"
    ).resolve()
)
_C4_SUPERVISOR_PATH = str(
    (
        REPOSITORY
        / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c4.py"
    ).resolve()
)
_C1_POLICY_SOURCE_PATH = str(
    (
        REPOSITORY / "tools/cure_lite_v24_preaccess_schema_compatibility.py"
    ).resolve()
)
_C2_POLICY_SOURCE_PATH = str(
    (
        REPOSITORY / "tools/cure_lite_v24_preaccess_schema_compatibility_c2.py"
    ).resolve()
)
_C3_POLICY_SOURCE_PATH = str(
    (
        REPOSITORY
        / "tools/cure_lite_v24_preaccess_schema_compatibility_c3.py"
    ).resolve()
)
_C4_POLICY_SOURCE_PATH = str(
    (
        REPOSITORY / "tools/cure_lite_v24_preaccess_schema_compatibility_c4.py"
    ).resolve()
)

def _stable_source_bytes(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, int]]:
    """Read one canonical regular-file generation through no-follow fds."""

    target = Path(path).absolute()
    parent = target.parent
    parent_before = parent.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or parent.resolve(strict=True) != parent
        or target.resolve(strict=True) != target
    ):
        raise PermissionError("compatibility supervisor source is unsafe")
    directory_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        parent_opened = os.fstat(directory_fd)
        before = os.stat(
            target.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
        ):
            raise PermissionError(
                "compatibility supervisor source is unsafe"
            )
        descriptor = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            finished = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        linked = os.stat(
            target.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        parent_finished = os.fstat(directory_fd)
    finally:
        os.close(directory_fd)
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(finished, field)
        or getattr(finished, field) != getattr(linked, field)
        for field in _FILE_IDENTITY_FIELDS
    ) or any(
        getattr(parent_before, field)
        != getattr(parent_opened, field)
        or getattr(parent_opened, field)
        != getattr(parent_finished, field)
        for field in _PARENT_IDENTITY_FIELDS
    ):
        raise PermissionError(
            "compatibility supervisor source changed during read"
        )
    raw = b"".join(chunks)
    if (
        expected_sha256 is not None
        and hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise PermissionError("compatibility supervisor source hash changed")
    identity = {
        field: int(getattr(linked, field))
        for field in _FILE_IDENTITY_FIELDS
    }
    identity.update(
        {
            f"parent_{field}": int(getattr(parent_finished, field))
            for field in _PARENT_IDENTITY_FIELDS
        }
    )
    return raw, identity


def _load_frozen_c1() -> tuple[ModuleType, dict[str, int]]:
    raw, generation = _stable_source_bytes(
        FROZEN_COMPAT_C1_SUPERVISOR_PATH,
        expected_sha256=FROZEN_COMPAT_C1_SUPERVISOR_SHA256,
    )
    name = (
        "tools._cure_lite_v24_runtime_supervisor_preaccess_compat_c1_"
        "verified_for_compat_c5"
    )
    module = ModuleType(name)
    module.__file__ = str(FROZEN_COMPAT_C1_SUPERVISOR_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                raw,
                str(FROZEN_COMPAT_C1_SUPERVISOR_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    raw_after, generation_after = _stable_source_bytes(
        FROZEN_COMPAT_C1_SUPERVISOR_PATH,
        expected_sha256=FROZEN_COMPAT_C1_SUPERVISOR_SHA256,
    )
    if generation_after != generation or raw_after != raw:
        sys.modules.pop(name, None)
        raise PermissionError(
            "compatibility c1 supervisor generation changed while loading"
        )
    return module, generation


compat_c1, _C1_LOAD_GENERATION = _load_frozen_c1()
_SELF_LOAD_BYTES, _SELF_LOAD_GENERATION = _stable_source_bytes(
    Path(__file__).resolve()
)
_SELF_LOAD_SHA256 = hashlib.sha256(_SELF_LOAD_BYTES).hexdigest()


def _capture_bridge_load_generation() -> (
    tuple[dict[str, int], str] | None
):
    if _SHA256.fullmatch(COMPAT_POLICY_SOURCE_SHA256) is None:
        return None
    raw, generation = _stable_source_bytes(
        COMPAT_POLICY_SOURCE_PATH,
        expected_sha256=COMPAT_POLICY_SOURCE_SHA256,
    )
    return generation, hashlib.sha256(raw).hexdigest()


_BRIDGE_LOAD = _capture_bridge_load_generation()


def _require_self_and_c1_generations() -> None:
    raw_c1, observed_c1 = _stable_source_bytes(
        FROZEN_COMPAT_C1_SUPERVISOR_PATH,
        expected_sha256=FROZEN_COMPAT_C1_SUPERVISOR_SHA256,
    )
    raw_self, observed_self = _stable_source_bytes(
        Path(COMPAT_SUPERVISOR_PATH)
    )
    if (
        observed_c1 != _C1_LOAD_GENERATION
        or observed_self != _SELF_LOAD_GENERATION
        or hashlib.sha256(raw_c1).hexdigest()
        != FROZEN_COMPAT_C1_SUPERVISOR_SHA256
        or hashlib.sha256(raw_self).hexdigest() != _SELF_LOAD_SHA256
    ):
        raise PermissionError(
            "compatibility supervisor generation changed after load"
        )


def _require_source_generations() -> None:
    """Require the exact loaded self, c1, and frozen c5 bridge generations."""

    _require_self_and_c1_generations()
    if _SHA256.fullmatch(COMPAT_POLICY_SOURCE_SHA256) is None:
        raise PermissionError("c5 bridge source generation is not frozen")
    if _BRIDGE_LOAD is None:
        raise PermissionError(
            "c5 bridge source was not frozen at supervisor load"
        )
    expected_generation, expected_hash = _BRIDGE_LOAD
    raw_bridge, observed_bridge = _stable_source_bytes(
        COMPAT_POLICY_SOURCE_PATH,
        expected_sha256=COMPAT_POLICY_SOURCE_SHA256,
    )
    if (
        observed_bridge != expected_generation
        or hashlib.sha256(raw_bridge).hexdigest() != expected_hash
    ):
        raise PermissionError(
            "c5 bridge source generation changed after supervisor load"
        )


def _require_disjoint_c5_identity() -> None:
    expected = {
        "unit": (
            "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c5.service"
        ),
        "spec": str(
            (
                REPOSITORY
                / "protocols/IRSTD-1K/gcr_pacre_v24/"
                "runtime_evidence_r2/"
                "D_R_structural_attempt_r2_preaccess_compat_c5_"
                "runtime_spec.json"
            ).resolve()
        ),
        "launch": str(
            (
                REPOSITORY
                / "protocols/IRSTD-1K/gcr_pacre_v24/"
                "runtime_evidence_r2/"
                "D_R_structural_attempt_r2_preaccess_compat_c5_"
                "runtime_launch_authorization.json"
            ).resolve()
        ),
        "artifacts": str(
            (
                EVIDENCE_ROOT
                / (
                    "D_R_structural_attempt_r2_preaccess_compat_c5_"
                    "runtime_artifacts"
                )
            ).resolve()
        ),
        "gpu_lease": str(
            (
                EVIDENCE_ROOT
                / "D_R_structural_attempt_r2_preaccess_compat_c5_gpu_lease"
            ).resolve()
        ),
        "run_alias": str(
            (
                RUNS_ROOT
                / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c5"
            ).resolve()
        ),
        "result_alias": str(
            (
                EVIDENCE_ROOT
                / "D_R_structural_attempt_r2_preaccess_compat_c5_receipt.json"
            ).resolve()
        ),
        "adapter": str(
            (
                REPOSITORY
                / "tools/"
                "run_cure_lite_v24_gcr_pacre_dr_gate_r2_"
                "preaccess_compat_c5.py"
            ).resolve()
        ),
        "supervisor": str(
            (
                REPOSITORY
                / "tools/"
                "cure_lite_v24_runtime_supervisor_"
                "preaccess_compat_c5.py"
            ).resolve()
        ),
        "receipt": str(
            (
                REPOSITORY
                / "protocols/IRSTD-1K/gcr_pacre_v24/"
                "runtime_evidence_r2/"
                "r2_preaccess_schema_compat_c5_receipt.json"
            ).resolve()
        ),
        "policy": str(
            (
                REPOSITORY
                / "tools/"
                "cure_lite_v24_preaccess_schema_compatibility_c5.py"
            ).resolve()
        ),
    }
    c5_values = {
        COMPAT_UNIT_NAME,
        COMPAT_RUNTIME_SPEC_PATH,
        COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        COMPAT_RUNTIME_ARTIFACT_ROOT,
        COMPAT_GPU_LEASE_ROOT,
        COMPAT_RUN_ROOT_ALIAS_PATH,
        COMPAT_RESULT_RECEIPT_ALIAS_PATH,
        COMPAT_ADAPTER_PATH,
        COMPAT_SUPERVISOR_PATH,
        COMPATIBILITY_RECEIPT_PATH,
        str(COMPAT_POLICY_SOURCE_PATH),
    }
    forbidden = {
        _C1_UNIT_NAME,
        _OLD_UNIT_NAME,
        _C2_UNIT_NAME,
        _C3_UNIT_NAME,
        _C4_UNIT_NAME,
        _C1_RUNTIME_SPEC_PATH,
        _C2_RUNTIME_SPEC_PATH,
        _C3_RUNTIME_SPEC_PATH,
        _C4_RUNTIME_SPEC_PATH,
        _OLD_RUNTIME_SPEC_PATH,
        _C1_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        _C2_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        _C3_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        _C4_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        _OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        _OLD_RUNTIME_ARTIFACT_ROOT,
        _OLD_GPU_LEASE_ROOT,
        _C1_RUNTIME_ARTIFACT_ROOT,
        _C2_RUNTIME_ARTIFACT_ROOT,
        _C3_RUNTIME_ARTIFACT_ROOT,
        _C4_RUNTIME_ARTIFACT_ROOT,
        _C1_GPU_LEASE_ROOT,
        _C2_GPU_LEASE_ROOT,
        _C3_GPU_LEASE_ROOT,
        _C4_GPU_LEASE_ROOT,
        _C1_RUN_ROOT_ALIAS_PATH,
        _C2_RUN_ROOT_ALIAS_PATH,
        _C3_RUN_ROOT_ALIAS_PATH,
        _C4_RUN_ROOT_ALIAS_PATH,
        _C1_RESULT_RECEIPT_ALIAS_PATH,
        _C2_RESULT_RECEIPT_ALIAS_PATH,
        _C3_RESULT_RECEIPT_ALIAS_PATH,
        _C4_RESULT_RECEIPT_ALIAS_PATH,
        _C1_RECEIPT_PATH,
        _C2_RECEIPT_PATH,
        _C3_RECEIPT_PATH,
        _C4_RECEIPT_PATH,
        _C1_ADAPTER_PATH,
        _C2_ADAPTER_PATH,
        _C3_ADAPTER_PATH,
        _C4_ADAPTER_PATH,
        _C1_SUPERVISOR_PATH,
        _C2_SUPERVISOR_PATH,
        _C3_SUPERVISOR_PATH,
        _C4_SUPERVISOR_PATH,
        _C1_POLICY_SOURCE_PATH,
        _C2_POLICY_SOURCE_PATH,
        _C3_POLICY_SOURCE_PATH,
        _C4_POLICY_SOURCE_PATH,
    }
    if (
        COMPATIBILITY_GENERATION != "c5"
        or len(c5_values) != 11
        or c5_values & forbidden
        or COMPAT_UNIT_NAME != expected["unit"]
        or COMPAT_RUNTIME_SPEC_PATH != expected["spec"]
        or COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        != expected["launch"]
        or COMPAT_RUNTIME_ARTIFACT_ROOT != expected["artifacts"]
        or COMPAT_GPU_LEASE_ROOT != expected["gpu_lease"]
        or COMPAT_RUN_ROOT_ALIAS_PATH != expected["run_alias"]
        or COMPAT_RESULT_RECEIPT_ALIAS_PATH != expected["result_alias"]
        or COMPAT_RUN_ROOT_ALIAS_PATH == SCIENTIFIC_RUN_ROOT
        or COMPAT_RESULT_RECEIPT_ALIAS_PATH
        == SCIENTIFIC_RESULT_RECEIPT_PATH
        or COMPAT_ADAPTER_PATH != expected["adapter"]
        or COMPAT_SUPERVISOR_PATH != expected["supervisor"]
        or COMPATIBILITY_RECEIPT_PATH != expected["receipt"]
        or str(COMPAT_POLICY_SOURCE_PATH) != expected["policy"]
    ):
        raise PermissionError("compatibility c5 runtime identity changed")


def _require_compatibility_aliases_absent() -> None:
    aliases = (
        _C1_RUN_ROOT_ALIAS_PATH,
        _C2_RUN_ROOT_ALIAS_PATH,
        _C3_RUN_ROOT_ALIAS_PATH,
        _C4_RUN_ROOT_ALIAS_PATH,
        COMPAT_RUN_ROOT_ALIAS_PATH,
        _C1_RESULT_RECEIPT_ALIAS_PATH,
        _C2_RESULT_RECEIPT_ALIAS_PATH,
        _C3_RESULT_RECEIPT_ALIAS_PATH,
        _C4_RESULT_RECEIPT_ALIAS_PATH,
        COMPAT_RESULT_RECEIPT_ALIAS_PATH,
    )
    if any(os.path.lexists(path) for path in aliases):
        raise PermissionError(
            "compatibility c5 scientific output alias exists"
        )


def _configure_c5_identity() -> None:
    _require_disjoint_c5_identity()
    compat_c1.COMPATIBILITY_GENERATION = COMPATIBILITY_GENERATION
    compat_c1.COMPAT_UNIT_NAME = COMPAT_UNIT_NAME
    compat_c1.COMPAT_RUNTIME_SPEC_PATH = COMPAT_RUNTIME_SPEC_PATH
    compat_c1.COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
        COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
    )
    compat_c1.COMPAT_RUNTIME_ARTIFACT_ROOT = COMPAT_RUNTIME_ARTIFACT_ROOT
    compat_c1.COMPAT_GPU_LEASE_ROOT = COMPAT_GPU_LEASE_ROOT
    compat_c1.COMPAT_RUN_ROOT_ALIAS_PATH = COMPAT_RUN_ROOT_ALIAS_PATH
    compat_c1.COMPAT_RESULT_RECEIPT_ALIAS_PATH = (
        COMPAT_RESULT_RECEIPT_ALIAS_PATH
    )
    compat_c1.SCIENTIFIC_RUN_ROOT = SCIENTIFIC_RUN_ROOT
    compat_c1.SCIENTIFIC_RESULT_RECEIPT_PATH = (
        SCIENTIFIC_RESULT_RECEIPT_PATH
    )
    compat_c1.COMPAT_ADAPTER_PATH = COMPAT_ADAPTER_PATH
    compat_c1.COMPAT_SUPERVISOR_PATH = COMPAT_SUPERVISOR_PATH
    compat_c1.COMPATIBILITY_RECEIPT_PATH = COMPATIBILITY_RECEIPT_PATH
    compat_c1.COMPAT_POLICY_SOURCE_PATH = COMPAT_POLICY_SOURCE_PATH
    compat_c1.COMPAT_POLICY_SOURCE_SHA256 = COMPAT_POLICY_SOURCE_SHA256
    compat_c1._configure_frozen_supervisor()


def _prepare_production_execution() -> None:
    _require_source_generations()
    _configure_c5_identity()
    _require_compatibility_aliases_absent()


def _guarded_inherited_callable(
    function: Callable[..., object],
) -> Callable[..., object]:
    """Close over one inherited callable without publishing a raw alias.

    ``functools.wraps`` is intentionally not used: its ``__wrapped__`` escape
    hatch would expose the unguarded c1/legacy function to import callers.
    """

    def guarded(*args, **kwargs):
        _prepare_production_execution()
        return function(*args, **kwargs)

    guarded.__name__ = getattr(function, "__name__", "guarded_c5_callable")
    guarded.__doc__ = getattr(function, "__doc__", None)
    return guarded


_C1_VERIFY_COMPATIBILITY_IDENTITY = _guarded_inherited_callable(
    compat_c1.verify_compatibility_identity
)
_C1_VALIDATE_SPEC_STRUCTURE = _guarded_inherited_callable(
    compat_c1._validate_spec_structure
)
_C1_VALIDATE_PREWRITE_SPEC = _guarded_inherited_callable(
    compat_c1.validate_prewrite_spec
)
_C1_LOAD_VERIFIED_COMPATIBILITY_POLICY = _guarded_inherited_callable(
    compat_c1._load_verified_compatibility_policy
)
_C1_VALIDATE_POLICY_RECEIPT_CONTRACT = _guarded_inherited_callable(
    compat_c1._validate_policy_receipt_contract
)
_C1_VERIFY_POLICY_PREWRITE = _guarded_inherited_callable(
    compat_c1._verify_policy_compatibility_prewrite
)
_C1_VERIFY_CHILD_ATTESTATION = _guarded_inherited_callable(
    compat_c1.verify_child_runtime_attestation
)
_C1_MAIN = _guarded_inherited_callable(compat_c1.main)
_RUNTIME_PHASES = frozenset(
    {
        "preactivation",
        "commit",
        "claim",
        "verify",
        "run_once",
        "finalize_success",
        "finalize_failure",
    }
)
_ACTIVE_RUNTIME_PHASE: ContextVar[str] = ContextVar(
    "cure_lite_v24_c5_runtime_phase",
    default="preactivation",
)
# Deliberately remains empty after installation. Raw inherited callables live
# only inside guarded closures and cannot be reached through a module global.
_RAW_LEGACY_RUNTIME_ENTRYPOINTS: dict[str, Callable[..., object]] = {}

# The release producer needs this non-callable schema constant. Keep it as an
# explicit local export instead of reopening the whole c1 namespace.
RUNTIME_SPEC_SCHEMA = compat_c1.legacy.RUNTIME_SPEC_SCHEMA


def _exact_absolute_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be an absolute path")
    target = Path(value)
    canonical = Path(os.path.abspath(value))
    if not target.is_absolute() or target != canonical:
        raise ValueError(f"{name} must be an exact absolute path")
    return canonical


def _validate_c5_runtime_lane(
    payload: Mapping[str, object],
    *,
    loaded_spec_path: Path,
) -> None:
    """Bind only runtime-owned fields to c5; archival evidence stays foreign."""

    if Path(os.path.abspath(loaded_spec_path)) != Path(
        COMPAT_RUNTIME_SPEC_PATH
    ):
        raise ValueError("compatibility actual runtime spec path is not exact")
    if payload.get("execution_kind") != compat_c1.legacy.ACTUAL_EXECUTION_KIND:
        raise ValueError("compatibility c5 supervisor permits only actual D_R")
    if (
        payload.get("attempt_id")
        != "gcr_pacre_v24_D_R_zero_update_structural_r2"
        or payload.get("attempt_ordinal") != 2
        or payload.get("prior_attempt_count") != 1
    ):
        raise ValueError("compatibility c5 scientific attempt changed")

    authorization = payload.get("authorization")
    child = payload.get("child")
    artifacts = payload.get("artifacts")
    environment = payload.get("environment")
    runtime = payload.get("runtime")
    if any(
        not isinstance(value, Mapping)
        for value in (authorization, child, artifacts, environment, runtime)
    ):
        raise ValueError("compatibility c5 runtime lane is malformed")
    assert isinstance(authorization, Mapping)
    assert isinstance(child, Mapping)
    assert isinstance(artifacts, Mapping)
    assert isinstance(environment, Mapping)
    assert isinstance(runtime, Mapping)

    argv = child.get("argv")
    systemd_contract = runtime.get("systemd")
    if not isinstance(argv, list) or not isinstance(systemd_contract, Mapping):
        raise ValueError("compatibility c5 runtime command is malformed")
    if (
        _exact_absolute_path(
            authorization.get("path"),
            name="authorization.path",
        )
        != Path(COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH)
        or _exact_absolute_path(
            child.get("entrypoint_path"),
            name="child.entrypoint_path",
        )
        != Path(COMPAT_ADAPTER_PATH)
        or COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH not in argv
        or _exact_absolute_path(
            artifacts.get("root"),
            name="artifacts.root",
        )
        != Path(COMPAT_RUNTIME_ARTIFACT_ROOT)
        or _exact_absolute_path(
            environment.get("gpu_lease_path"),
            name="environment.gpu_lease_path",
        )
        != Path(COMPAT_GPU_LEASE_ROOT) / "active.json"
        or _exact_absolute_path(
            environment.get("gpu_lease_tombstone_path"),
            name="environment.gpu_lease_tombstone_path",
        )
        != Path(COMPAT_GPU_LEASE_ROOT) / "released.json"
        or systemd_contract.get("unit_name") != COMPAT_UNIT_NAME
        or runtime.get("launch_limit") != 1
        or runtime.get("automatic_retry_allowed") is not False
        or runtime.get("resume_allowed") is not False
        or runtime.get("restart") != "no"
    ):
        raise ValueError("compatibility c5 runtime lane changed")

    root = Path(COMPAT_RUNTIME_ARTIFACT_ROOT)
    for field, value in artifacts.items():
        path = _exact_absolute_path(value, name=f"artifacts.{field}")
        if field != "root" and path.parent != root and root not in path.parents:
            raise ValueError("compatibility c5 artifact escaped its root")


def describe_compatibility_identity() -> dict[str, object]:
    """Return fixed metadata without authorizing or reading runtime evidence."""

    _require_self_and_c1_generations()
    _require_disjoint_c5_identity()
    bridge_frozen = (
        _SHA256.fullmatch(COMPAT_POLICY_SOURCE_SHA256) is not None
        and _BRIDGE_LOAD is not None
    )
    return {
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_generation": COMPATIBILITY_GENERATION,
        "unit_name": COMPAT_UNIT_NAME,
        "runtime_spec_path": COMPAT_RUNTIME_SPEC_PATH,
        "runtime_launch_authorization_path": (
            COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "runtime_artifact_root": COMPAT_RUNTIME_ARTIFACT_ROOT,
        "gpu_lease_root": COMPAT_GPU_LEASE_ROOT,
        "forbidden_run_root_alias": COMPAT_RUN_ROOT_ALIAS_PATH,
        "forbidden_result_receipt_alias": (
            COMPAT_RESULT_RECEIPT_ALIAS_PATH
        ),
        "scientific_run_root": SCIENTIFIC_RUN_ROOT,
        "scientific_result_receipt_path": SCIENTIFIC_RESULT_RECEIPT_PATH,
        "supervisor_path": COMPAT_SUPERVISOR_PATH,
        "adapter_path": COMPAT_ADAPTER_PATH,
        "compatibility_receipt_path": COMPATIBILITY_RECEIPT_PATH,
        "compatibility_policy_source_path": str(
            COMPAT_POLICY_SOURCE_PATH
        ),
        "compatibility_policy_source_sha256": (
            COMPAT_POLICY_SOURCE_SHA256
        ),
        "bridge_source_frozen_at_load": bridge_frozen,
        "production_ready": bridge_frozen,
        "scientific_identity_changed": False,
        "scientific_output_paths_changed": False,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
        "parameter_updates_authorized": 0,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def verify_compatibility_identity() -> dict[str, object]:
    _prepare_production_execution()
    result = dict(_C1_VERIFY_COMPATIBILITY_IDENTITY())
    expected = {
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_generation": COMPATIBILITY_GENERATION,
        "unit_name": COMPAT_UNIT_NAME,
        "runtime_spec_path": COMPAT_RUNTIME_SPEC_PATH,
        "runtime_launch_authorization_path": (
            COMPAT_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "supervisor_path": COMPAT_SUPERVISOR_PATH,
        "adapter_path": COMPAT_ADAPTER_PATH,
        "compatibility_receipt_path": COMPATIBILITY_RECEIPT_PATH,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise PermissionError("compatibility c5 supervisor identity changed")
    forbidden_values = {
        _C1_UNIT_NAME,
        _OLD_UNIT_NAME,
        _C2_UNIT_NAME,
        _C3_UNIT_NAME,
        _C4_UNIT_NAME,
        _C1_RUNTIME_SPEC_PATH,
        _C2_RUNTIME_SPEC_PATH,
        _C3_RUNTIME_SPEC_PATH,
        _C4_RUNTIME_SPEC_PATH,
        _OLD_RUNTIME_SPEC_PATH,
        _C1_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        _C2_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        _C3_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        _C4_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        _OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        _OLD_RUNTIME_ARTIFACT_ROOT,
        _OLD_GPU_LEASE_ROOT,
        _C1_RUNTIME_ARTIFACT_ROOT,
        _C2_RUNTIME_ARTIFACT_ROOT,
        _C3_RUNTIME_ARTIFACT_ROOT,
        _C4_RUNTIME_ARTIFACT_ROOT,
        _C1_GPU_LEASE_ROOT,
        _C2_GPU_LEASE_ROOT,
        _C3_GPU_LEASE_ROOT,
        _C4_GPU_LEASE_ROOT,
        _C1_RUN_ROOT_ALIAS_PATH,
        _C2_RUN_ROOT_ALIAS_PATH,
        _C3_RUN_ROOT_ALIAS_PATH,
        _C4_RUN_ROOT_ALIAS_PATH,
        _C1_RESULT_RECEIPT_ALIAS_PATH,
        _C2_RESULT_RECEIPT_ALIAS_PATH,
        _C3_RESULT_RECEIPT_ALIAS_PATH,
        _C4_RESULT_RECEIPT_ALIAS_PATH,
        _C1_RECEIPT_PATH,
        _C2_RECEIPT_PATH,
        _C3_RECEIPT_PATH,
        _C4_RECEIPT_PATH,
        _C1_ADAPTER_PATH,
        _C2_ADAPTER_PATH,
        _C3_ADAPTER_PATH,
        _C4_ADAPTER_PATH,
        _C1_SUPERVISOR_PATH,
        _C2_SUPERVISOR_PATH,
        _C3_SUPERVISOR_PATH,
        _C4_SUPERVISOR_PATH,
        _C1_POLICY_SOURCE_PATH,
        _C2_POLICY_SOURCE_PATH,
        _C3_POLICY_SOURCE_PATH,
        _C4_POLICY_SOURCE_PATH,
    }
    if any(
        isinstance(value, str) and value in forbidden_values
        for value in result.values()
    ):
        raise PermissionError(
            "compatibility c5 identity retained an old runtime lane"
        )
    result.update(
        {
            "compatibility_policy_source_path": str(
                COMPAT_POLICY_SOURCE_PATH
            ),
            "compatibility_policy_source_sha256": (
                COMPAT_POLICY_SOURCE_SHA256
            ),
            "frozen_compat_c1_supervisor_path": str(
                FROZEN_COMPAT_C1_SUPERVISOR_PATH
            ),
            "frozen_compat_c1_supervisor_file_sha256": (
                FROZEN_COMPAT_C1_SUPERVISOR_SHA256
            ),
            "c5_supervisor_file_sha256": _SELF_LOAD_SHA256,
            "bridge_source_frozen_at_load": True,
            "production_ready": True,
            "scientific_identity_changed": False,
            "scientific_output_paths_changed": False,
            "runtime_artifact_root": COMPAT_RUNTIME_ARTIFACT_ROOT,
            "gpu_lease_root": COMPAT_GPU_LEASE_ROOT,
            "forbidden_run_root_alias": COMPAT_RUN_ROOT_ALIAS_PATH,
            "forbidden_result_receipt_alias": (
                COMPAT_RESULT_RECEIPT_ALIAS_PATH
            ),
            "scientific_run_root": SCIENTIFIC_RUN_ROOT,
            "scientific_result_receipt_path": (
                SCIENTIFIC_RESULT_RECEIPT_PATH
            ),
            "automatic_retry_allowed": False,
            "resume_allowed": False,
            "training_authorized": False,
            "optimizer_steps_authorized": 0,
            "parameter_updates_authorized": 0,
        }
    )
    return result


def _verify_policy_compatibility_receipt(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    _prepare_production_execution()
    runtime_phase = _ACTIVE_RUNTIME_PHASE.get()
    if runtime_phase not in _RUNTIME_PHASES:
        raise PermissionError("compatibility c5 runtime phase is invalid")
    policy, policy_generation = (
        _C1_LOAD_VERIFIED_COMPATIBILITY_POLICY()
    )
    policy_path = getattr(policy, "COMPAT_RECEIPT_PATH", None)
    if (
        policy_path is None
        or Path(os.path.abspath(policy_path))
        != Path(COMPATIBILITY_RECEIPT_PATH)
    ):
        raise PermissionError(
            "preaccess compatibility receipt path interface changed"
        )
    verifier = getattr(policy, "verify_compatibility_receipt", None)
    if not callable(verifier):
        raise PermissionError(
            "preaccess compatibility receipt verifier is unavailable"
        )
    result = verifier(
        Path(COMPATIBILITY_RECEIPT_PATH),
        expected_spec=dict(payload),
        require_spec_binding=True,
        allow_runtime_activation=(
            runtime_phase != "preactivation"
        ),
        runtime_phase=runtime_phase,
    )
    return _C1_VALIDATE_POLICY_RECEIPT_CONTRACT(
        result,
        policy=policy,
        policy_generation=policy_generation,
    )


def _verify_policy_compatibility_prewrite(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    _prepare_production_execution()
    return _C1_VERIFY_POLICY_PREWRITE(payload)


def _validate_spec_structure(
    payload: dict[str, object],
    *,
    loaded_spec_path: Path,
) -> None:
    _prepare_production_execution()
    _validate_c5_runtime_lane(payload, loaded_spec_path=loaded_spec_path)
    _C1_VALIDATE_SPEC_STRUCTURE(
        payload,
        loaded_spec_path=loaded_spec_path,
    )


def validate_prewrite_spec(payload: Mapping[str, object]) -> None:
    _prepare_production_execution()
    materialized = dict(payload)
    _validate_c5_runtime_lane(
        materialized,
        loaded_spec_path=Path(COMPAT_RUNTIME_SPEC_PATH),
    )
    _C1_VALIDATE_PREWRITE_SPEC(materialized)


def verify_child_runtime_attestation(*args, **kwargs):
    # This public entrypoint is reached by the already-spawned adapter child.
    # Keep the run_once phase active across the entire inherited attestation
    # and nested runtime-spec load so B5 observes runtime activation authority.
    # Generic load_runtime_spec calls intentionally remain preactivation.
    token = _ACTIVE_RUNTIME_PHASE.set("run_once")
    try:
        _prepare_production_execution()
        return _C1_VERIFY_CHILD_ATTESTATION(*args, **kwargs)
    finally:
        _ACTIVE_RUNTIME_PHASE.reset(token)


def _finalization_runtime_phase(
    _args: Sequence[object],
    kwargs: Mapping[str, object],
) -> str:
    environment = kwargs.get("environment")
    if environment is None:
        environment = os.environ
    if not isinstance(environment, Mapping):
        return "finalize_failure"
    success = (
        environment.get("SERVICE_RESULT") == "success"
        and environment.get("EXIT_CODE") == "exited"
        and environment.get("EXIT_STATUS") == "0"
    )
    return "finalize_success" if success else "finalize_failure"


def _phase_guarded_inherited_callable(
    function: Callable[..., object],
    phase: str
    | Callable[[Sequence[object], Mapping[str, object]], str],
) -> Callable[..., object]:
    def guarded(*args, **kwargs):
        selected = phase(args, kwargs) if callable(phase) else phase
        if selected not in _RUNTIME_PHASES - {"preactivation"}:
            raise PermissionError(
                "compatibility c5 runtime phase is invalid"
            )
        token = _ACTIVE_RUNTIME_PHASE.set(selected)
        try:
            _prepare_production_execution()
            return function(*args, **kwargs)
        finally:
            _ACTIVE_RUNTIME_PHASE.reset(token)

    guarded.__name__ = getattr(function, "__name__", "guarded_c5_phase")
    guarded.__doc__ = getattr(function, "__doc__", None)
    return guarded


def _install_inherited_runtime_guards() -> None:
    """Close the loaded c1/legacy module escape hatches.

    The frozen c1 module and its materialized legacy supervisor remain
    reachable for evidence inspection.  Their inherited runtime entry points
    must nevertheless cross the same c5 source-generation guard as this
    module's public wrappers before the inherited implementation can parse a
    path, inspect systemd, or create runtime evidence.
    """

    raw_entrypoints = {
        name: getattr(compat_c1.legacy, name)
        for name in (
            "commit_and_start",
            "claim_materialization",
            "verify_runtime_spec",
            "run_once",
            "finalize_systemd",
            "load_runtime_spec",
            "verify_child_runtime_attestation",
            "main",
        )
    }
    compat_c1.main = _C1_MAIN
    phases: dict[
        str,
        str
        | Callable[[Sequence[object], Mapping[str, object]], str],
    ] = {
        "commit_and_start": "commit",
        "claim_materialization": "claim",
        "verify_runtime_spec": "verify",
        "run_once": "run_once",
        "finalize_systemd": _finalization_runtime_phase,
    }
    try:
        for name, function in raw_entrypoints.items():
            if name in {
                "main",
                "load_runtime_spec",
                "verify_child_runtime_attestation",
            }:
                guarded = _guarded_inherited_callable(function)
            else:
                guarded = _phase_guarded_inherited_callable(
                    function,
                    phases[name],
                )
            setattr(compat_c1.legacy, name, guarded)
    finally:
        raw_entrypoints.clear()
        _RAW_LEGACY_RUNTIME_ENTRYPOINTS.clear()


def main(argv: Sequence[str] | None = None) -> int:
    _prepare_production_execution()
    return int(_C1_MAIN(argv))


# c1's own implementation resolves these names dynamically.  Replacing them
# ensures calls reached through the frozen supervisor also cross the c5 guard.
compat_c1.verify_compatibility_identity = verify_compatibility_identity
compat_c1._verify_policy_compatibility_receipt = (
    _verify_policy_compatibility_receipt
)
compat_c1._verify_policy_compatibility_prewrite = (
    _verify_policy_compatibility_prewrite
)
compat_c1._validate_spec_structure = _validate_spec_structure
compat_c1.validate_prewrite_spec = validate_prewrite_spec
compat_c1.verify_child_runtime_attestation = (
    verify_child_runtime_attestation
)
_install_inherited_runtime_guards()


# Explicit phase-aware exports used by tests and any in-process supervisor
# caller. All production systemd invocations continue to enter through main.
commit_and_start = compat_c1.legacy.commit_and_start
claim_materialization = compat_c1.legacy.claim_materialization
verify_runtime_spec = compat_c1.legacy.verify_runtime_spec
run_once = compat_c1.legacy.run_once
finalize_systemd = compat_c1.legacy.finalize_systemd
load_runtime_spec = compat_c1.legacy.load_runtime_spec


if __name__ == "__main__":
    raise SystemExit(main())
