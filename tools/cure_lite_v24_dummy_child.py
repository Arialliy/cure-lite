"""Dataset-free child used only by the user-systemd integration matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time


DUMMY_ARTIFACT_SCHEMA = "cure-lite-v24-user-systemd-dummy-child-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _open_safe_parent(path: Path) -> int:
    parent = path.parent
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise PermissionError("dummy artifact path must be absolute and direct")
    before = os.lstat(parent)
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o700
        or parent.resolve(strict=True) != parent
    ):
        raise PermissionError("dummy artifact parent must be an owned safe directory")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(parent, flags)
    current = os.fstat(descriptor)
    if (
        current.st_dev != before.st_dev
        or current.st_ino != before.st_ino
        or not stat.S_ISDIR(current.st_mode)
        or current.st_uid != os.getuid()
        or stat.S_IMODE(current.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise PermissionError("dummy artifact parent changed during verification")
    return descriptor


def _write_new(path: Path, payload: dict[str, object]) -> str:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent = _open_safe_parent(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        current = os.fstat(descriptor)
        linked = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_uid != os.getuid()
            or current.st_dev != linked.st_dev
            or current.st_ino != linked.st_ino
        ):
            raise PermissionError("dummy artifact identity is unsafe")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        sealed_link = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if (
            stat.S_IMODE(sealed.st_mode) != 0o444
            or sealed.st_dev != sealed_link.st_dev
            or sealed.st_ino != sealed_link.st_ino
            or sealed.st_nlink != 1
        ):
            raise PermissionError("dummy artifact did not seal exactly")
        os.fsync(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)
    if path.read_bytes() != encoded:
        raise PermissionError("dummy artifact changed after sealing")
    return hashlib.sha256(encoded).hexdigest()


def run_dummy_child(
    *,
    artifact: Path,
    scenario_id: str,
    wait_seconds: float,
) -> str:
    if wait_seconds < 0 or wait_seconds > 2:
        raise ValueError("dummy wait must lie in [0,2] seconds")
    body: dict[str, object] = {
        "schema_version": DUMMY_ARTIFACT_SCHEMA,
        "scenario_id": scenario_id,
        "dataset_accessed": False,
        "gpu_accessed": False,
        "torch_imported": False,
        "pid": os.getpid(),
    }
    payload = {
        **body,
        "dummy_artifact_fingerprint": hashlib.sha256(
            _canonical_json(body).encode("utf-8")
        ).hexdigest(),
    }
    digest = _write_new(artifact, payload)
    print("CURE-Lite v24 user-systemd dummy child", flush=True)
    if wait_seconds:
        time.sleep(wait_seconds)
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--wait-seconds", type=float, default=0.05)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    run_dummy_child(
        artifact=arguments.artifact,
        scenario_id=arguments.scenario_id,
        wait_seconds=arguments.wait_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
