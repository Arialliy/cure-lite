"""Strict append-only artifacts for CURE-Lite v23 PACRE-VC."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Final, Mapping

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)


PACRE_VC_SOURCE_CLOSURE_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-source-closure-v2"
)
PACRE_VC_SOURCE_ROOTS: Final = (
    "cure_lite",
    "cure_lite_v22",
    "cure_lite_v23",
)
PACRE_VC_PACKAGE_METADATA_PATHS: Final = (
    "pyproject.toml",
    "cure_lite_v22/pyproject.toml",
    "cure_lite_v23/pyproject.toml",
)
PACRE_VC_TOOL_PREFIX: Final = "run_cure_lite_v23_pacre_vc_"
PACRE_VC_EXTRA_TOOL_PATHS: Final = (
    "tools/__init__.py",
    "tools/run_with_gpu_temperature_control.py",
    "tools/verify_cure_lite_v23_pacre_vc_dr_receipt.py",
    "tools/verify_cure_lite_v23_pacre_vc_formal_800_receipt.py",
    (
        "tools/"
        "verify_cure_lite_v23_pacre_vc_formal_d_v_receipt.py"
    ),
)
PACRE_VC_CONTRACT_PATHS: Final = (
    (
        "protocols/IRSTD-1K/pacre_v23_verifier_corrected/"
        "verifier_design_preregistration.md"
    ),
    (
        "protocols/IRSTD-1K/pacre_v23_verifier_corrected/"
        "relative_performance_preregistration.md"
    ),
)
PACRE_VC_SOURCE_LAYER_NAMES: Final = (
    "root",
    "v22",
    "v23_and_runners",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _regular_repo_file(relative: str) -> Path:
    root = repository_root()
    path = root / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError(f"invalid repository file: {relative}")
    return path


def source_inventory() -> tuple[str, ...]:
    """Return the complete package, metadata, and runner source closure."""

    root = repository_root()
    rows: list[str] = []
    for source_root in PACRE_VC_SOURCE_ROOTS:
        directory = root / source_root
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeError(f"invalid source root: {source_root}")
        rows.extend(
            path.relative_to(root).as_posix()
            for path in directory.rglob("*.py")
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
        )
    tools = root / "tools"
    rows.extend(
        path.relative_to(root).as_posix()
        for path in tools.glob(f"{PACRE_VC_TOOL_PREFIX}*.py")
        if path.is_file() and not path.is_symlink()
    )
    rows.extend(
        relative
        for relative in PACRE_VC_EXTRA_TOOL_PATHS
    )
    rows.extend(PACRE_VC_PACKAGE_METADATA_PATHS)
    rows.extend(PACRE_VC_CONTRACT_PATHS)
    result = tuple(sorted(rows))
    if not result or len(result) != len(set(result)):
        raise RuntimeError("PACRE-VC source inventory is empty or duplicated")
    for relative in result:
        _regular_repo_file(relative)
    return result


def source_binding() -> tuple[tuple[str, str], ...]:
    return tuple(
        (relative, file_sha256(_regular_repo_file(relative)))
        for relative in source_inventory()
    )


def _source_layer(relative: str) -> str:
    if relative == "pyproject.toml" or relative.startswith("cure_lite/"):
        return "root"
    if relative.startswith("cure_lite_v22/"):
        return "v22"
    if (
        relative.startswith("cure_lite_v23/")
        or relative.startswith("tools/")
        or relative.startswith(
            "protocols/IRSTD-1K/pacre_v23_verifier_corrected/"
        )
    ):
        return "v23_and_runners"
    raise RuntimeError(f"unclassified PACRE-VC source: {relative}")


def source_closure_payload() -> dict[str, object]:
    binding = source_binding()
    layered: dict[str, list[dict[str, str]]] = {
        name: [] for name in PACRE_VC_SOURCE_LAYER_NAMES
    }
    for path, digest in binding:
        layered[_source_layer(path)].append(
            {"repo_path": path, "sha256": digest}
        )
    layers = {
        name: {
            "file_count": len(layered[name]),
            "files": layered[name],
            "binding_fingerprint": stable_fingerprint(
                {
                    row["repo_path"]: row["sha256"]
                    for row in layered[name]
                }
            ),
        }
        for name in PACRE_VC_SOURCE_LAYER_NAMES
    }
    if any(layer["file_count"] < 1 for layer in layers.values()):
        raise RuntimeError("PACRE-VC source layer is empty")
    body: dict[str, object] = {
        "schema_version": PACRE_VC_SOURCE_CLOSURE_SCHEMA,
        "source_roots": list(PACRE_VC_SOURCE_ROOTS),
        "package_metadata_paths": list(
            PACRE_VC_PACKAGE_METADATA_PATHS
        ),
        "tool_prefix": PACRE_VC_TOOL_PREFIX,
        "extra_tool_paths": list(PACRE_VC_EXTRA_TOOL_PATHS),
        "contract_paths": list(PACRE_VC_CONTRACT_PATHS),
        "file_count": len(binding),
        "files": [
            {"repo_path": path, "sha256": digest}
            for path, digest in binding
        ],
        "binding_fingerprint": stable_fingerprint(dict(binding)),
        "layer_order": list(PACRE_VC_SOURCE_LAYER_NAMES),
        "layers": layers,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    return {
        **body,
        "closure_fingerprint": stable_fingerprint(body),
    }


def verify_source_closure(payload: Mapping[str, object]) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError("source closure must be a mapping")
    current = source_closure_payload()
    if dict(payload) != current:
        raise RuntimeError("PACRE-VC source closure differs from live sources")
    fingerprint = current["closure_fingerprint"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise AssertionError("source closure fingerprint is malformed")
    return fingerprint


def fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    if field in payload:
        raise ValueError(f"payload already contains {field}")
    body = dict(payload)
    return {**body, field: stable_fingerprint(body)}


def verify_fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError("fingerprinted payload must be a mapping")
    body = dict(payload)
    value = body.pop(field, None)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or stable_fingerprint(body) != value
    ):
        raise ValueError("payload fingerprint is invalid")
    return value


def strict_json_bytes(payload: Mapping[str, object]) -> bytes:
    text = canonical_json(dict(payload))
    # Reparse with strict non-finite rejection.  canonical_json already rejects
    # non-finite Python floats; parse_constant closes the JSON reader boundary.
    decoded = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(decoded, dict) or canonical_json(decoded) != text:
        raise ValueError("JSON payload is not a canonical object")
    return (text + "\n").encode("utf-8")


def write_new_json(path: Path, payload: Mapping[str, object]) -> str:
    """Create one canonical JSON file without overwrite."""

    path = Path(path)
    if path.is_symlink():
        raise FileExistsError(f"refusing symlink output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = strict_json_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return sha256(data).hexdigest()


def read_strict_json(path: Path) -> dict[str, object]:
    path = Path(path)
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError(f"invalid JSON artifact: {path}")
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(
        raw,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if (
        not isinstance(payload, dict)
        or strict_json_bytes(payload).decode("utf-8") != raw
    ):
        raise ValueError(f"non-canonical JSON artifact: {path}")
    return payload


__all__ = [
    "PACRE_VC_CONTRACT_PATHS",
    "PACRE_VC_PACKAGE_METADATA_PATHS",
    "PACRE_VC_SOURCE_CLOSURE_SCHEMA",
    "PACRE_VC_SOURCE_LAYER_NAMES",
    "PACRE_VC_SOURCE_ROOTS",
    "PACRE_VC_EXTRA_TOOL_PATHS",
    "PACRE_VC_TOOL_PREFIX",
    "fingerprinted",
    "read_strict_json",
    "repository_root",
    "source_binding",
    "source_closure_payload",
    "source_inventory",
    "strict_json_bytes",
    "verify_fingerprinted",
    "verify_source_closure",
    "write_new_json",
]
