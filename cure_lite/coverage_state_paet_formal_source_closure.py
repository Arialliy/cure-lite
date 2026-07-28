"""Reproducible local-source closure for the PAET-BFA Formal800 runner.

This deliberately archives source, not a git revision.  The closure is a
fixed, exhaustive inventory of the installed ``cure_lite`` package plus the
small set of executable entrypoints which can affect Formal800.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping


SOURCE_CLOSURE_SCHEMA = "cure-lite-paet-bfa-v21-formal800-source-closure-v1"
RECEIPT_SCHEMA = "cure-lite-paet-bfa-v21-formal800-source-closure-receipt-v1"
ARCHIVE_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_paet_bfa_v21_pmope_formal800_source_closure.tar"
)
MANIFEST_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_paet_bfa_v21_pmope_formal800_source_closure.json"
)
_EXTRA_PATHS = (
    "pyproject.toml",
    "tools/__init__.py",
    "tools/run_coverage_state_cslf_ppce_support_oriented_bounded_400.py",
    "tools/run_coverage_state_cslf_support_oriented_bounded_400.py",
    "tools/run_coverage_state_cmif_pmope_bounded_400.py",
    "tools/run_coverage_state_paet_bfa_pmope_bounded_400.py",
    "tools/run_coverage_state_paet_bfa_pmope_formal_800.py",
    "tools/run_coverage_state_paet_bfa_pmope_formal_d_v.py",
    "tools/run_with_gpu_temperature_control.py",
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "archive",
        "content_fingerprint",
        "file_count",
        "files",
    }
)
_ARCHIVE_KEYS = frozenset({"repo_path", "sha256", "bytes"})
_FILE_KEYS = frozenset({"repo_path", "sha256", "bytes"})


class FormalSourceClosureError(RuntimeError):
    """Raised when a Formal800 source closure is absent or invalid."""


def _repository_root(repository_root: Path | None) -> Path:
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    absolute = Path(os.path.abspath(root))
    if absolute.is_symlink() or absolute.resolve(strict=True) != absolute:
        raise FormalSourceClosureError("repository root is not canonical")
    if not absolute.is_dir():
        raise FormalSourceClosureError("repository root is not a directory")
    return absolute


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_repo_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise FormalSourceClosureError("closure path must be non-empty")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise FormalSourceClosureError(f"closure path is unsafe: {value!r}")
    if path.as_posix() != value:
        raise FormalSourceClosureError(f"closure path is not canonical: {value!r}")
    return value


def _checked_file(root: Path, repo_path: str) -> Path:
    repo_path = _canonical_repo_path(repo_path)
    candidate = root.joinpath(*PurePosixPath(repo_path).parts)
    absolute = Path(os.path.abspath(candidate))
    try:
        resolved = absolute.resolve(strict=True)
    except FileNotFoundError as error:
        raise FormalSourceClosureError(
            f"closure file is missing: {repo_path}"
        ) from error
    if candidate.is_symlink() or resolved != absolute or not absolute.is_file():
        raise FormalSourceClosureError(
            f"closure file is not canonical regular file: {repo_path}"
        )
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise FormalSourceClosureError(
            f"closure file escapes repository: {repo_path}"
        ) from error
    return absolute


def _tool_import_closure(
    root: Path,
    seed_paths: set[str],
) -> set[str]:
    """Resolve repository-local ``tools`` imports transitively.

    Formal800 imports historical runner modules for frozen constants and
    runtime checks.  Archiving only the final CLI would therefore omit source
    that Python executes while importing it.
    """

    discovered = {
        repo_path
        for repo_path in seed_paths
        if repo_path.startswith("tools/") and repo_path.endswith(".py")
    }
    pending = list(sorted(discovered))
    while pending:
        repo_path = pending.pop()
        path = _checked_file(root, repo_path)
        try:
            tree = ast.parse(path.read_bytes(), filename=repo_path)
        except SyntaxError as error:
            raise FormalSourceClosureError(
                f"closure tool source cannot be parsed: {repo_path}"
            ) from error
        module_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level != 0 or node.module is None:
                    continue
                if node.module == "tools":
                    module_names.update(
                        f"tools.{alias.name}" for alias in node.names
                    )
                elif node.module.startswith("tools."):
                    module_names.add(node.module)
            elif isinstance(node, ast.Import):
                module_names.update(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("tools.")
                )
        for module_name in module_names:
            candidate = module_name.replace(".", "/") + ".py"
            candidate_path = root.joinpath(
                *PurePosixPath(candidate).parts
            )
            if not candidate_path.is_file():
                continue
            _checked_file(root, candidate)
            if candidate not in discovered:
                discovered.add(candidate)
                pending.append(candidate)
    return discovered


def formal_source_closure_paths(
    repository_root: Path | None = None,
) -> tuple[str, ...]:
    """Return the sorted exhaustive Formal800 source inventory."""

    root = _repository_root(repository_root)
    package = root / "cure_lite"
    if package.is_symlink() or not package.is_dir():
        raise FormalSourceClosureError("cure_lite package directory is invalid")
    paths: set[str] = set(_EXTRA_PATHS)
    paths.update(_tool_import_closure(root, paths))
    for candidate in package.rglob("*"):
        if candidate.is_symlink():
            raise FormalSourceClosureError(
                f"cure_lite source tree contains a non-canonical symbolic link: "
                f"{candidate.relative_to(root).as_posix()}"
            )
    for source in package.rglob("*.py"):
        relative = source.relative_to(root).as_posix()
        _checked_file(root, relative)
        paths.add(relative)
    if "cure_lite/__init__.py" not in paths:
        raise FormalSourceClosureError("cure_lite package initializer is missing")
    for repo_path in paths:
        _checked_file(root, repo_path)
    return tuple(sorted(paths))


def _content_fingerprint(files: list[dict[str, object]]) -> str:
    return _sha256_bytes(_canonical_json({"files": files}))


def _file_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repo_path in formal_source_closure_paths(root):
        path = _checked_file(root, repo_path)
        rows.append(
            {
                "repo_path": repo_path,
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def _archive_path(root: Path) -> Path:
    return _closure_destination(root, ARCHIVE_REPO_PATH)


def _manifest_path(root: Path) -> Path:
    return _closure_destination(root, MANIFEST_REPO_PATH)


def _closure_destination(root: Path, repo_path: str) -> Path:
    """Return an output path while rejecting symlinked artifact ancestors."""

    _canonical_repo_path(repo_path)
    current = root
    for part in PurePosixPath(repo_path).parts:
        current = current / part
        if current.is_symlink():
            raise FormalSourceClosureError(
                f"source closure destination is symlinked: {repo_path}"
            )
    return current


def _archive_sources(root: Path, archive: Path, files: list[dict[str, object]]) -> None:
    with archive.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tar:
            for row in files:
                repo_path = str(row["repo_path"])
                path = _checked_file(root, repo_path)
                info = tarfile.TarInfo(name=repo_path)
                info.size = int(row["bytes"])
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with path.open("rb") as handle:
                    tar.addfile(info, handle)


def build_coverage_state_paet_formal_source_closure(
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Create the fixed one-time Formal800 source tar and manifest.

    The destination is intentionally not configurable, and either existing
    destination (including a symlink) rejects creation rather than being
    replaced.
    """

    root = _repository_root(repository_root)
    archive = _archive_path(root)
    manifest_path = _manifest_path(root)
    if archive.exists() or archive.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("Formal800 source closure destination already exists")
    archive.parent.mkdir(parents=True, exist_ok=True)
    files = _file_rows(root)
    _archive_sources(root, archive, files)
    manifest = {
        "schema_version": SOURCE_CLOSURE_SCHEMA,
        "archive": {
            "repo_path": ARCHIVE_REPO_PATH,
            "sha256": _file_sha256(archive),
            "bytes": archive.stat().st_size,
        },
        "content_fingerprint": _content_fingerprint(files),
        "file_count": len(files),
        "files": files,
    }
    with manifest_path.open("xb") as handle:
        handle.write(_canonical_json(manifest))
    return verify_coverage_state_paet_formal_source_closure(root)


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormalSourceClosureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_manifest(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FormalSourceClosureError("Formal800 source closure manifest is missing")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalSourceClosureError("source closure manifest is not strict JSON") from error
    if not isinstance(value, dict) or _canonical_json(value) != raw:
        raise FormalSourceClosureError("source closure manifest is not canonical JSON")
    return value


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FormalSourceClosureError(f"{field} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise FormalSourceClosureError(f"{field} is not hexadecimal") from error
    return value


def _validate_manifest(manifest: Mapping[str, object], root: Path) -> list[dict[str, object]]:
    if set(manifest) != _MANIFEST_KEYS or manifest.get("schema_version") != SOURCE_CLOSURE_SCHEMA:
        raise FormalSourceClosureError("source closure manifest schema changed")
    archive = manifest.get("archive")
    if not isinstance(archive, dict) or set(archive) != _ARCHIVE_KEYS:
        raise FormalSourceClosureError("source closure archive record is invalid")
    if archive.get("repo_path") != ARCHIVE_REPO_PATH:
        raise FormalSourceClosureError("source closure archive path changed")
    _require_sha256(archive.get("sha256"), "archive.sha256")
    if type(archive.get("bytes")) is not int or int(archive["bytes"]) < 0:
        raise FormalSourceClosureError("archive bytes is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise FormalSourceClosureError("source closure file inventory is invalid")
    normalized: list[dict[str, object]] = []
    for row in files:
        if not isinstance(row, dict) or set(row) != _FILE_KEYS:
            raise FormalSourceClosureError("source closure file row is invalid")
        repo_path = _canonical_repo_path(row.get("repo_path"))
        digest = _require_sha256(row.get("sha256"), f"file {repo_path} SHA")
        size = row.get("bytes")
        if type(size) is not int or size < 0:
            raise FormalSourceClosureError(f"file {repo_path} byte count is invalid")
        normalized.append({"repo_path": repo_path, "sha256": digest, "bytes": size})
    names = [str(row["repo_path"]) for row in normalized]
    if names != sorted(names) or len(names) != len(set(names)):
        raise FormalSourceClosureError("source closure file order is not canonical")
    if names != list(formal_source_closure_paths(root)):
        raise FormalSourceClosureError("source closure inventory differs from Formal800 scope")
    if type(manifest.get("file_count")) is not int or manifest.get("file_count") != len(normalized):
        raise FormalSourceClosureError("source closure file count differs")
    if manifest.get("content_fingerprint") != _content_fingerprint(normalized):
        raise FormalSourceClosureError("source closure content fingerprint differs")
    return normalized


def verify_coverage_state_paet_formal_source_closure(
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Rehash the sealed archive, its members, and every live source file."""

    root = _repository_root(repository_root)
    manifest_path = _manifest_path(root)
    manifest = _read_manifest(manifest_path)
    files = _validate_manifest(manifest, root)
    archive_record = manifest["archive"]
    assert isinstance(archive_record, dict)
    archive = _archive_path(root)
    if archive.is_symlink() or not archive.is_file():
        raise FormalSourceClosureError("Formal800 source closure archive is missing")
    if archive.stat().st_size != archive_record["bytes"] or _file_sha256(archive) != archive_record["sha256"]:
        raise FormalSourceClosureError("Formal800 source closure archive hash differs")
    expected_names = [str(row["repo_path"]) for row in files]
    try:
        with tarfile.open(archive, mode="r:") as tar:
            members = tar.getmembers()
            names = [member.name for member in members]
            if names != expected_names:
                raise FormalSourceClosureError("source closure archive member inventory differs")
            for member, row in zip(members, files, strict=True):
                if (
                    not member.isreg()
                    or member.name != row["repo_path"]
                    or member.size != row["bytes"]
                    or member.mode != 0o644
                    or member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                ):
                    raise FormalSourceClosureError("source closure tar metadata differs")
                extracted = tar.extractfile(member)
                if extracted is None or _sha256_bytes(extracted.read()) != row["sha256"]:
                    raise FormalSourceClosureError("source closure tar member hash differs")
    except (tarfile.TarError, OSError) as error:
        if isinstance(error, FormalSourceClosureError):
            raise
        raise FormalSourceClosureError("source closure archive cannot be read") from error
    for row in files:
        path = _checked_file(root, str(row["repo_path"]))
        if path.stat().st_size != row["bytes"] or _file_sha256(path) != row["sha256"]:
            raise FormalSourceClosureError("current Formal800 source differs from closure")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "sealed": True,
        "manifest_repo_path": MANIFEST_REPO_PATH,
        "manifest_sha256": _file_sha256(manifest_path),
        "archive_repo_path": ARCHIVE_REPO_PATH,
        "archive_sha256": archive_record["sha256"],
        "archive_bytes": archive_record["bytes"],
        "content_fingerprint": manifest["content_fingerprint"],
        "file_count": manifest["file_count"],
    }


__all__ = [
    "ARCHIVE_REPO_PATH",
    "MANIFEST_REPO_PATH",
    "FormalSourceClosureError",
    "build_coverage_state_paet_formal_source_closure",
    "formal_source_closure_paths",
    "verify_coverage_state_paet_formal_source_closure",
]
