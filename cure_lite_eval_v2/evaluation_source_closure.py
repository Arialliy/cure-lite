"""Separate append-only source closure for the PAET D_V recovery wrapper."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping

from .formal800_schema_erratum import (
    ERRATUM_REPO_PATH,
    verify_formal800_schema_erratum,
)


EVALUATION_SOURCE_CLOSURE_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-evaluation-source-closure-v2"
)
EVALUATION_SOURCE_CLOSURE_RECEIPT_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-evaluation-source-closure-receipt-v2"
)
EVALUATION_SOURCE_CLOSURE_ARCHIVE_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_paet_bfa_v21_formal_d_v_evaluation_v2_source_closure.tar"
)
EVALUATION_SOURCE_CLOSURE_MANIFEST_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_paet_bfa_v21_formal_d_v_evaluation_v2_source_closure.json"
)
_SOURCE_PATHS = (
    "cure_lite_eval_v2/__init__.py",
    "cure_lite_eval_v2/evaluation_source_closure.py",
    "cure_lite_eval_v2/formal800_schema_erratum.py",
    "cure_lite_eval_v2/formal_d_v_runner_v2.py",
    ERRATUM_REPO_PATH,
    "tools/build_coverage_state_paet_formal_d_v_evaluation_v2_closure.py",
    "tools/run_coverage_state_paet_bfa_pmope_formal_d_v_evaluation_v2.py",
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "archive",
        "content_fingerprint",
        "file_count",
        "files",
        "original_formal_source_closure",
        "formal800_schema_erratum",
    }
)
_ARCHIVE_KEYS = frozenset({"repo_path", "sha256", "bytes"})
_FILE_KEYS = frozenset({"repo_path", "sha256", "bytes"})
_ORIGINAL_KEYS = frozenset(
    {
        "manifest_repo_path",
        "manifest_sha256",
        "archive_repo_path",
        "archive_sha256",
        "content_fingerprint",
        "file_count",
    }
)
_ERRATUM_KEYS = frozenset(
    {"repo_path", "sha256", "erratum_fingerprint"}
)


class EvaluationSourceClosureError(RuntimeError):
    """Raised when the v2 evaluation-code closure is absent or invalid."""


def _repository_root(repository_root: Path | None) -> Path:
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    absolute = Path(os.path.abspath(root))
    if (
        absolute.is_symlink()
        or not absolute.is_dir()
        or absolute.resolve(strict=True) != absolute
    ):
        raise EvaluationSourceClosureError(
            "repository root must be a canonical directory"
        )
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
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationSourceClosureError(
            f"{name} must be a lowercase SHA256 digest"
        )
    return value


def _canonical_repo_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationSourceClosureError(
            "evaluation closure path must be non-empty"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise EvaluationSourceClosureError(
            f"evaluation closure path is unsafe: {value!r}"
        )
    return value


def _checked_file(root: Path, repo_path: str) -> Path:
    canonical = _canonical_repo_path(repo_path)
    candidate = root.joinpath(*PurePosixPath(canonical).parts)
    absolute = Path(os.path.abspath(candidate))
    try:
        resolved = absolute.resolve(strict=True)
    except FileNotFoundError as error:
        raise EvaluationSourceClosureError(
            f"evaluation closure file is missing: {canonical}"
        ) from error
    if (
        candidate.is_symlink()
        or resolved != absolute
        or not absolute.is_file()
    ):
        raise EvaluationSourceClosureError(
            f"evaluation closure file is not canonical: {canonical}"
        )
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise EvaluationSourceClosureError(
            f"evaluation closure file escapes repository: {canonical}"
        ) from error
    return absolute


def evaluation_source_closure_paths(
    repository_root: Path | None = None,
) -> tuple[str, ...]:
    """Return the fixed, append-only evaluation-code inventory."""

    root = _repository_root(repository_root)
    paths = tuple(sorted(_SOURCE_PATHS))
    if (
        len(paths) != len(set(paths))
        or any(path.startswith("cure_lite/") for path in paths)
    ):
        raise EvaluationSourceClosureError(
            "evaluation closure must not absorb original cure_lite sources"
        )
    for repo_path in paths:
        _checked_file(root, repo_path)
    return paths


def _file_rows(root: Path) -> list[dict[str, object]]:
    return [
        {
            "repo_path": repo_path,
            "sha256": _file_sha256(_checked_file(root, repo_path)),
            "bytes": _checked_file(root, repo_path).stat().st_size,
        }
        for repo_path in evaluation_source_closure_paths(root)
    ]


def _content_fingerprint(files: list[dict[str, object]]) -> str:
    return _sha256_bytes(_canonical_json({"files": files}))


def _destination(root: Path, repo_path: str) -> Path:
    _canonical_repo_path(repo_path)
    current = root
    for part in PurePosixPath(repo_path).parts:
        current = current / part
        if current.is_symlink():
            raise EvaluationSourceClosureError(
                "evaluation closure destination has a symbolic-link ancestor"
            )
    return current


def _archive_sources(
    root: Path,
    archive: Path,
    files: list[dict[str, object]],
) -> None:
    with archive.open("xb") as raw:
        with tarfile.open(
            fileobj=raw,
            mode="w",
            format=tarfile.GNU_FORMAT,
        ) as tar:
            for row in files:
                repo_path = str(row["repo_path"])
                info = tarfile.TarInfo(name=repo_path)
                info.size = int(row["bytes"])
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with _checked_file(root, repo_path).open("rb") as stream:
                    tar.addfile(info, stream)


def _closure_bindings(
    repository_root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    erratum = verify_formal800_schema_erratum(repository_root)
    original = {
        "manifest_repo_path": (
            "artifacts/source_closures/"
            "cure_lite_paet_bfa_v21_pmope_formal800_source_closure.json"
        ),
        "manifest_sha256": erratum[
            "original_source_closure_manifest_sha256"
        ],
        "archive_repo_path": (
            "artifacts/source_closures/"
            "cure_lite_paet_bfa_v21_pmope_formal800_source_closure.tar"
        ),
        "archive_sha256": erratum[
            "original_source_closure_archive_sha256"
        ],
        "content_fingerprint": erratum[
            "original_source_closure_content_fingerprint"
        ],
        "file_count": erratum[
            "original_source_closure_file_count"
        ],
    }
    erratum_binding = {
        "repo_path": erratum["erratum_repo_path"],
        "sha256": erratum["erratum_sha256"],
        "erratum_fingerprint": erratum["erratum_fingerprint"],
    }
    return original, erratum_binding


def build_evaluation_source_closure(
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Create the evaluation-code closure once after source review finishes."""

    root = _repository_root(repository_root)
    archive = _destination(
        root,
        EVALUATION_SOURCE_CLOSURE_ARCHIVE_REPO_PATH,
    )
    manifest_path = _destination(
        root,
        EVALUATION_SOURCE_CLOSURE_MANIFEST_REPO_PATH,
    )
    if (
        archive.exists()
        or archive.is_symlink()
        or manifest_path.exists()
        or manifest_path.is_symlink()
    ):
        raise FileExistsError(
            "evaluation source closure destination already exists"
        )
    original, erratum = _closure_bindings(root)
    files = _file_rows(root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _archive_sources(root, archive, files)
    manifest = {
        "schema_version": EVALUATION_SOURCE_CLOSURE_SCHEMA,
        "archive": {
            "repo_path": (
                EVALUATION_SOURCE_CLOSURE_ARCHIVE_REPO_PATH
            ),
            "sha256": _file_sha256(archive),
            "bytes": archive.stat().st_size,
        },
        "content_fingerprint": _content_fingerprint(files),
        "file_count": len(files),
        "files": files,
        "original_formal_source_closure": original,
        "formal800_schema_erratum": erratum,
    }
    with manifest_path.open("xb") as stream:
        stream.write(_canonical_json(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    return verify_evaluation_source_closure(root)


def _strict_manifest(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise EvaluationSourceClosureError(
            "evaluation source closure manifest is missing"
        )

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationSourceClosureError(
                    "evaluation closure manifest has duplicate keys"
                )
            result[key] = value
        return result

    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationSourceClosureError(
            "evaluation source closure manifest is not strict JSON"
        ) from error
    if (
        not isinstance(value, dict)
        or _canonical_json(value) != raw
    ):
        raise EvaluationSourceClosureError(
            "evaluation source closure manifest is not canonical"
        )
    return value


def _validate_binding(
    actual: object,
    expected: Mapping[str, object],
    *,
    keys: frozenset[str],
    name: str,
) -> None:
    if (
        not isinstance(actual, dict)
        or set(actual) != keys
        or actual != expected
    ):
        raise EvaluationSourceClosureError(f"{name} binding changed")


def verify_evaluation_source_closure(
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Verify the separate archive and every current evaluation source byte."""

    root = _repository_root(repository_root)
    manifest_path = _destination(
        root,
        EVALUATION_SOURCE_CLOSURE_MANIFEST_REPO_PATH,
    )
    manifest = _strict_manifest(manifest_path)
    if (
        set(manifest) != _MANIFEST_KEYS
        or manifest.get("schema_version")
        != EVALUATION_SOURCE_CLOSURE_SCHEMA
    ):
        raise EvaluationSourceClosureError(
            "evaluation source closure schema changed"
        )
    original, erratum = _closure_bindings(root)
    _validate_binding(
        manifest.get("original_formal_source_closure"),
        original,
        keys=_ORIGINAL_KEYS,
        name="original source closure",
    )
    _validate_binding(
        manifest.get("formal800_schema_erratum"),
        erratum,
        keys=_ERRATUM_KEYS,
        name="Formal800 schema erratum",
    )

    archive_row = manifest.get("archive")
    if (
        not isinstance(archive_row, dict)
        or set(archive_row) != _ARCHIVE_KEYS
        or archive_row.get("repo_path")
        != EVALUATION_SOURCE_CLOSURE_ARCHIVE_REPO_PATH
    ):
        raise EvaluationSourceClosureError(
            "evaluation closure archive record changed"
        )
    archive_sha = _digest(
        archive_row.get("sha256"),
        name="evaluation closure archive SHA256",
    )
    archive_bytes = archive_row.get("bytes")
    if type(archive_bytes) is not int or archive_bytes < 1:
        raise EvaluationSourceClosureError(
            "evaluation closure archive byte count changed"
        )

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise EvaluationSourceClosureError(
            "evaluation closure file inventory is invalid"
        )
    files: list[dict[str, object]] = []
    for raw_row in raw_files:
        if (
            not isinstance(raw_row, dict)
            or set(raw_row) != _FILE_KEYS
        ):
            raise EvaluationSourceClosureError(
                "evaluation closure file row is invalid"
            )
        repo_path = _canonical_repo_path(raw_row.get("repo_path"))
        digest = _digest(
            raw_row.get("sha256"),
            name=f"{repo_path} SHA256",
        )
        size = raw_row.get("bytes")
        if type(size) is not int or size < 0:
            raise EvaluationSourceClosureError(
                f"{repo_path} byte count is invalid"
            )
        files.append(
            {"repo_path": repo_path, "sha256": digest, "bytes": size}
        )
    if (
        [row["repo_path"] for row in files]
        != list(evaluation_source_closure_paths(root))
        or manifest.get("file_count") != len(files)
        or manifest.get("content_fingerprint")
        != _content_fingerprint(files)
    ):
        raise EvaluationSourceClosureError(
            "evaluation closure source inventory changed"
        )

    archive = _destination(
        root,
        EVALUATION_SOURCE_CLOSURE_ARCHIVE_REPO_PATH,
    )
    if (
        archive.is_symlink()
        or not archive.is_file()
        or archive.stat().st_size != archive_bytes
        or _file_sha256(archive) != archive_sha
    ):
        raise EvaluationSourceClosureError(
            "evaluation source archive bytes changed"
        )
    try:
        with tarfile.open(archive, mode="r:") as tar:
            members = tar.getmembers()
            if [member.name for member in members] != [
                str(row["repo_path"]) for row in files
            ]:
                raise EvaluationSourceClosureError(
                    "evaluation source archive inventory changed"
                )
            for member, row in zip(members, files, strict=True):
                extracted = tar.extractfile(member)
                if (
                    not member.isreg()
                    or member.size != row["bytes"]
                    or member.mode != 0o644
                    or member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or extracted is None
                    or _sha256_bytes(extracted.read()) != row["sha256"]
                ):
                    raise EvaluationSourceClosureError(
                        "evaluation source archive member changed"
                    )
    except (tarfile.TarError, OSError) as error:
        if isinstance(error, EvaluationSourceClosureError):
            raise
        raise EvaluationSourceClosureError(
            "evaluation source archive cannot be read"
        ) from error

    for row in files:
        source = _checked_file(root, str(row["repo_path"]))
        if (
            source.stat().st_size != row["bytes"]
            or _file_sha256(source) != row["sha256"]
        ):
            raise EvaluationSourceClosureError(
                f"live evaluation source changed: {row['repo_path']}"
            )
    return {
        "schema_version": EVALUATION_SOURCE_CLOSURE_RECEIPT_SCHEMA,
        "sealed": True,
        "manifest_repo_path": (
            EVALUATION_SOURCE_CLOSURE_MANIFEST_REPO_PATH
        ),
        "manifest_sha256": _file_sha256(manifest_path),
        "archive_repo_path": (
            EVALUATION_SOURCE_CLOSURE_ARCHIVE_REPO_PATH
        ),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "content_fingerprint": manifest["content_fingerprint"],
        "file_count": len(files),
        "original_formal_source_closure": original,
        "formal800_schema_erratum": erratum,
    }


__all__ = [
    "EVALUATION_SOURCE_CLOSURE_ARCHIVE_REPO_PATH",
    "EVALUATION_SOURCE_CLOSURE_MANIFEST_REPO_PATH",
    "EvaluationSourceClosureError",
    "build_evaluation_source_closure",
    "evaluation_source_closure_paths",
    "verify_evaluation_source_closure",
]
