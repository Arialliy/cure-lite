#!/usr/bin/env python3
"""Compatibility-c4 entry point for the frozen v24 r2 gate adapter.

The byte-bound r2 adapter remains authoritative.  Its source is materialized
in memory with exactly one import target changed: child runtime attestations
are verified by the compatibility-c4 supervisor.  All bootstrap, scientific
identity, receipt, and run-marker behavior is otherwise the frozen adapter's
own code.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN_ADAPTER_PATH = (
    REPOSITORY / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2.py"
).resolve()
FROZEN_ADAPTER_SHA256 = (
    "5cbfd073d7df8f4257079c71e6f05110d31b383c48abe6e0e9127ee154785495"
)
COMPAT_ADAPTER_PATH = str(Path(__file__).resolve())
COMPAT_SUPERVISOR_MODULE = (
    "tools.cure_lite_v24_runtime_supervisor_preaccess_compat_c4"
)
_FROZEN_VERIFIER_IMPORT = (
    b"from tools.cure_lite_v24_runtime_supervisor import ("
)
_COMPAT_VERIFIER_IMPORT = (
    b"from tools.cure_lite_v24_runtime_supervisor_preaccess_compat_c4 "
    b"import ("
)


def _verified_source_bytes(path: Path, expected_sha256: str) -> bytes:
    target = Path(path)
    before = target.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or target.resolve(strict=True) != target
    ):
        raise PermissionError("frozen r2 adapter source is unsafe")
    descriptor = os.open(
        target,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
    )
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(after, field)
        for field in identity_fields
    ):
        raise PermissionError("frozen r2 adapter source changed while reading")
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError("frozen r2 adapter source changed")
    return raw


def _verify_frozen_adapter_source() -> str:
    _verified_source_bytes(FROZEN_ADAPTER_PATH, FROZEN_ADAPTER_SHA256)
    return FROZEN_ADAPTER_SHA256


def _materialized_compat_source() -> bytes:
    raw = _verified_source_bytes(
        FROZEN_ADAPTER_PATH,
        FROZEN_ADAPTER_SHA256,
    )
    if (
        raw.count(_FROZEN_VERIFIER_IMPORT) != 1
        or _COMPAT_VERIFIER_IMPORT in raw
    ):
        raise PermissionError(
            "frozen r2 adapter attestation-verifier import changed"
        )
    transformed = raw.replace(
        _FROZEN_VERIFIER_IMPORT,
        _COMPAT_VERIFIER_IMPORT,
    )
    if (
        transformed.count(_FROZEN_VERIFIER_IMPORT) != 0
        or transformed.count(_COMPAT_VERIFIER_IMPORT) != 1
    ):
        raise PermissionError(
            "compatibility adapter verifier projection is not unique"
        )
    return transformed


def _load_materialized_adapter(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = COMPAT_ADAPTER_PATH
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                _materialized_compat_source(),
                COMPAT_ADAPTER_PATH,
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _execute_direct() -> None:
    namespace = {
        "__name__": "__main__",
        "__file__": COMPAT_ADAPTER_PATH,
        "__package__": "tools",
        "__builtins__": __builtins__,
    }
    exec(
        compile(
            _materialized_compat_source(),
            COMPAT_ADAPTER_PATH,
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )


if __name__ == "__main__":
    _execute_direct()
else:
    legacy = _load_materialized_adapter(
        "tools._run_cure_lite_v24_gcr_pacre_dr_gate_r2_"
        "materialized_for_preaccess_compat_c4"
    )

    def configure_r2_execution_identity() -> dict[str, object]:
        _verify_frozen_adapter_source()
        result = legacy.configure_r2_execution_identity()
        expected_repo_path = str(
            Path(COMPAT_ADAPTER_PATH).relative_to(REPOSITORY)
        )
        if result.get("adapter_repo_path") != expected_repo_path:
            raise PermissionError(
                "compatibility adapter identity summary path changed"
            )
        return result

    def _verify_runtime_launch(
        runtime_launch_authorization_path: str,
    ) -> dict[str, object]:
        _verify_frozen_adapter_source()
        return legacy._verify_runtime_launch(
            runtime_launch_authorization_path
        )

    def main(argv: Sequence[str] | None = None) -> int:
        _verify_frozen_adapter_source()
        return int(legacy.main(argv))

    def verify_compatibility_identity() -> dict[str, object]:
        raw = _materialized_compat_source()
        expected_repo_path = str(
            Path(COMPAT_ADAPTER_PATH).relative_to(REPOSITORY)
        )
        if (
            _FROZEN_VERIFIER_IMPORT in raw
            or raw.count(_COMPAT_VERIFIER_IMPORT) != 1
            or Path(legacy.__file__).resolve()
            != Path(COMPAT_ADAPTER_PATH).resolve()
        ):
            raise PermissionError("compatibility adapter identity changed")
        return {
            "scientific_attempt_ordinal": 2,
            "runtime_compatibility_generation": "c4",
            "adapter_path": COMPAT_ADAPTER_PATH,
            "adapter_repo_path": expected_repo_path,
            "attestation_verifier_module": COMPAT_SUPERVISOR_MODULE,
            "frozen_adapter_path": str(FROZEN_ADAPTER_PATH),
            "frozen_adapter_file_sha256": FROZEN_ADAPTER_SHA256,
            "transformed_import_count": 1,
            "scientific_identity_changed": False,
            "scientific_receipt_paths_changed": False,
            "scientific_run_marker_paths_changed": False,
            "automatic_retry_allowed": False,
            "resume_allowed": False,
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }

    def __getattr__(name: str):
        return getattr(legacy, name)
