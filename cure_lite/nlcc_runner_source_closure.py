"""Deterministic local-source closure for the NLCC-v12 formal runner.

The closure is intentionally small in responsibility.  It starts from fixed
entrypoints, follows statically visible local Python imports, includes every
executed package ``__init__.py``, and hashes the resulting source graph.
Repository state is reported for context, but commit identity and unrelated
working-tree changes are deliberately excluded from the closure fingerprint.
"""

from __future__ import annotations

import ast
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Mapping, Sequence


SOURCE_CLOSURE_SCHEMA = "cure-lite.nlcc-v12.runner-source-closure.v1"

FORMAL_ROOT_PATHS = (
    "tools/evaluate_nlcc_development_regression.py",
    "tools/evaluate_nlcc_exposure_holdout.py",
)

# The builder is part of the evidence-producing implementation and therefore
# hashes itself alongside the two scientific entrypoints.
CLOSURE_BUILDER_ROOT_PATH = "cure_lite/nlcc_runner_source_closure.py"

REQUIRED_CLOSURE_PATHS = (
    "cure_lite/__init__.py",
    "cure_lite/cache/__init__.py",
    "cure_lite/train/__init__.py",
    "cure_lite/decoder.py",
    "cure_lite/factorized_config.py",
    "cure_lite/factorized_decoder.py",
    "cure_lite/nlcc_dataset_free_decision.py",
    "cure_lite/nlcc_dataset_free_inputs.py",
    "cure_lite/nlcc_dataset_free_runner.py",
    "cure_lite/nlcc_dataset_free_runner_config.py",
    "cure_lite/nlcc_development_inputs.py",
    "cure_lite/nlcc_holdout_inputs.py",
    "cure_lite/null_anchored_local_count_crossing_config.py",
    "cure_lite/null_anchored_local_count_crossing_decoder.py",
)

_FINGERPRINT_FIELDS = (
    "schema_version",
    "formal_roots",
    "analysis_roots",
    "required_paths",
    "nodes",
    "edges",
    "dynamic_imports",
    "external_modules",
    "unresolved_local_imports",
)


class SourceClosureError(RuntimeError):
    """Base error for source-closure construction or verification."""


class UnresolvedLocalImportError(SourceClosureError):
    """Raised when a local import cannot be mapped to repository source."""

    def __init__(self, unresolved: Sequence[Mapping[str, object]]) -> None:
        self.unresolved = tuple(dict(item) for item in unresolved)
        detail = _canonical_json(list(self.unresolved))
        super().__init__(f"unresolved local imports: {detail}")


class MissingRequiredClosurePathError(SourceClosureError):
    """Raised when a fixed expected source is absent from the closure."""

    def __init__(self, missing_paths: Sequence[str]) -> None:
        self.missing_paths = tuple(sorted(missing_paths))
        super().__init__(
            "required paths are not in the local import closure: "
            + ", ".join(self.missing_paths)
        )


class SourceClosureDriftError(SourceClosureError):
    """Raised when a recomputed in-scope source graph differs from a freeze."""

    def __init__(
        self,
        *,
        expected_fingerprint: str,
        actual_fingerprint: str,
        changed_paths: Sequence[str],
    ) -> None:
        self.expected_fingerprint = expected_fingerprint
        self.actual_fingerprint = actual_fingerprint
        self.changed_paths = tuple(sorted(changed_paths))
        super().__init__(
            "runner source closure drifted: "
            f"expected={expected_fingerprint}, actual={actual_fingerprint}, "
            f"changed_paths={list(self.changed_paths)}"
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_repo_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("repository paths must be non-empty strings")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe repository-relative path: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"repository path is not canonical: {value!r}")
    return normalized


def _checked_source_path(repository_root: Path, relative_path: str) -> Path:
    relative_path = _normalize_repo_path(relative_path)
    candidate = repository_root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise SourceClosureError(
            f"source path does not exist: {relative_path}"
        ) from error
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise SourceClosureError(
            f"source path escapes repository: {relative_path}"
        ) from error
    if not resolved.is_file() or resolved.suffix != ".py":
        raise SourceClosureError(
            f"source path must be a Python file: {relative_path}"
        )
    return resolved


def _module_index(
    repository_root: Path,
    local_package_names: Sequence[str],
) -> tuple[dict[str, str], dict[str, str]]:
    module_to_path: dict[str, str] = {}
    path_to_module: dict[str, str] = {}
    for raw_name in sorted(set(local_package_names)):
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or any(not part.isidentifier() for part in raw_name.split("."))
        ):
            raise ValueError(f"invalid local package name: {raw_name!r}")
        package_dir = repository_root.joinpath(*raw_name.split("."))
        initializer = package_dir / "__init__.py"
        if not initializer.is_file():
            raise SourceClosureError(
                f"local package has no initializer: {raw_name}"
            )
        for source_path in sorted(package_dir.rglob("*.py")):
            resolved = source_path.resolve(strict=True)
            try:
                relative = resolved.relative_to(repository_root).as_posix()
            except ValueError as error:
                raise SourceClosureError(
                    f"local package source escapes repository: {source_path}"
                ) from error
            module_parts = list(PurePosixPath(relative).with_suffix("").parts)
            if module_parts[-1] == "__init__":
                module_parts.pop()
            module_name = ".".join(module_parts)
            previous = module_to_path.get(module_name)
            if previous is not None and previous != relative:
                raise SourceClosureError(
                    f"duplicate local module {module_name!r}: "
                    f"{previous!r}, {relative!r}"
                )
            module_to_path[module_name] = relative
            path_to_module[relative] = module_name
    return module_to_path, path_to_module


def _is_package_path(path: str) -> bool:
    return PurePosixPath(path).name == "__init__.py"


def _package_ancestors(
    module_name: str,
    module_to_path: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    parts = module_name.split(".")
    ancestors: list[tuple[str, str]] = []
    for end in range(1, len(parts)):
        ancestor = ".".join(parts[:end])
        path = module_to_path.get(ancestor)
        if path is not None and _is_package_path(path):
            ancestors.append((ancestor, path))
    return tuple(ancestors)


def _current_package(
    source_path: str,
    path_to_module: Mapping[str, str],
) -> str | None:
    module_name = path_to_module.get(source_path)
    if module_name is None:
        return None
    if _is_package_path(source_path):
        return module_name
    package, _, _ = module_name.rpartition(".")
    return package or None


def _git_output(
    repository_root: Path,
    arguments: Sequence[str],
) -> bytes | None:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout


def _git_metadata(
    repository_root: Path,
    in_scope_paths: set[str],
) -> dict[str, object]:
    commit_raw = _git_output(repository_root, ("rev-parse", "HEAD"))
    tracked_raw = _git_output(
        repository_root,
        ("diff", "--name-only", "-z", "HEAD", "--"),
    )
    untracked_raw = _git_output(
        repository_root,
        ("ls-files", "--others", "--exclude-standard", "-z"),
    )
    if commit_raw is None or tracked_raw is None or untracked_raw is None:
        return {
            "repository_commit": None,
            "repository_dirty": None,
            "repository_dirty_paths": [],
            "in_scope_dirty_paths": [],
            "out_of_scope_dirty_paths": [],
        }

    def decode_paths(payload: bytes) -> list[str]:
        return [
            item.decode("utf-8", errors="surrogateescape")
            for item in payload.split(b"\0")
            if item
        ]

    dirty_paths = sorted(
        set(decode_paths(tracked_raw)) | set(decode_paths(untracked_raw))
    )
    return {
        "repository_commit": commit_raw.decode("ascii").strip(),
        "repository_dirty": bool(dirty_paths),
        "repository_dirty_paths": dirty_paths,
        "in_scope_dirty_paths": sorted(
            path for path in dirty_paths if path in in_scope_paths
        ),
        "out_of_scope_dirty_paths": sorted(
            path for path in dirty_paths if path not in in_scope_paths
        ),
    }


def _fingerprint_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    missing = [field for field in _FINGERPRINT_FIELDS if field not in manifest]
    if missing:
        raise SourceClosureError(
            "source closure manifest is missing fingerprint fields: "
            + ", ".join(missing)
        )
    return {field: manifest[field] for field in _FINGERPRINT_FIELDS}


def validate_source_closure_manifest(
    manifest: Mapping[str, object],
) -> str:
    """Validate and return the internally recomputed closure fingerprint."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    if manifest.get("schema_version") != SOURCE_CLOSURE_SCHEMA:
        raise SourceClosureError("unknown source closure schema")
    unresolved = manifest.get("unresolved_local_imports")
    if unresolved != []:
        raise SourceClosureError(
            "a valid source closure must contain no unresolved local imports"
        )
    fingerprint = manifest.get("closure_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise SourceClosureError("closure_fingerprint must be SHA256")
    recomputed = _sha256_bytes(
        _canonical_json(_fingerprint_payload(manifest)).encode("utf-8")
    )
    if recomputed != fingerprint:
        raise SourceClosureError(
            "closure_fingerprint does not match the manifest payload"
        )
    return recomputed


def build_local_import_closure(
    repository_root: str | Path,
    *,
    formal_roots: Sequence[str],
    analysis_roots: Sequence[str] | None = None,
    required_paths: Sequence[str] = (),
    local_package_names: Sequence[str] = ("cure_lite",),
) -> dict[str, object]:
    """Build a deterministic transitive closure of local Python imports.

    ``formal_roots`` identify the scientific entrypoints.  ``analysis_roots``
    may additionally bind the closure builder itself.  ``required_paths`` are
    expectations, not traversal seeds: every one must be reached through the
    roots and normal Python package initialization.
    """

    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    normalized_formal_roots = tuple(
        sorted({_normalize_repo_path(path) for path in formal_roots})
    )
    if not normalized_formal_roots:
        raise ValueError("formal_roots must not be empty")
    if analysis_roots is None:
        normalized_analysis_roots = normalized_formal_roots
    else:
        normalized_analysis_roots = tuple(
            sorted({_normalize_repo_path(path) for path in analysis_roots})
        )
        if not set(normalized_formal_roots).issubset(
            normalized_analysis_roots
        ):
            raise ValueError("analysis_roots must include every formal root")
    normalized_required = tuple(
        sorted({_normalize_repo_path(path) for path in required_paths})
    )
    module_to_path, path_to_module = _module_index(
        root,
        local_package_names,
    )
    local_prefixes = tuple(
        f"{name}." for name in sorted(set(local_package_names))
    )
    local_names = frozenset(local_package_names)

    queue: deque[str] = deque()
    reached: set[str] = set()
    visited: set[str] = set()
    edges: set[tuple[str, str, str, int, str]] = set()
    external_modules: set[str] = set()
    dynamic_imports: set[tuple[str, int, str, str]] = set()
    unresolved: list[dict[str, object]] = []

    def is_local_name(module_name: str) -> bool:
        return module_name in local_names or module_name.startswith(
            local_prefixes
        )

    def enqueue_path(path: str) -> None:
        _checked_source_path(root, path)
        if path not in reached:
            reached.add(path)
            queue.append(path)

    def add_target(
        *,
        source: str,
        target_module: str,
        kind: str,
        line: int,
        requested_module: str,
    ) -> bool:
        target_path = module_to_path.get(target_module)
        if target_path is None:
            if is_local_name(target_module):
                unresolved.append(
                    {
                        "source": source,
                        "line": line,
                        "kind": kind,
                        "module": requested_module,
                        "resolved_module": target_module,
                    }
                )
            return False
        for ancestor_module, ancestor_path in _package_ancestors(
            target_module,
            module_to_path,
        ):
            enqueue_path(ancestor_path)
            if source != ancestor_path:
                edges.add(
                    (
                        source,
                        ancestor_path,
                        "package_initializer",
                        line,
                        ancestor_module,
                    )
                )
        enqueue_path(target_path)
        if source != target_path:
            edges.add(
                (source, target_path, kind, line, requested_module)
            )
        return True

    for source in normalized_analysis_roots:
        enqueue_path(source)
        root_module = path_to_module.get(source)
        if root_module is not None:
            for ancestor_module, ancestor_path in _package_ancestors(
                root_module,
                module_to_path,
            ):
                enqueue_path(ancestor_path)
                if source != ancestor_path:
                    edges.add(
                        (
                            source,
                            ancestor_path,
                            "root_package_initializer",
                            0,
                            ancestor_module,
                        )
                    )

    while queue:
        source = queue.popleft()
        if source in visited:
            continue
        visited.add(source)
        source_path = _checked_source_path(root, source)
        try:
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=source,
            )
        except (SyntaxError, UnicodeError) as error:
            raise SourceClosureError(
                f"cannot parse Python source {source!r}: {error}"
            ) from error
        current_package = _current_package(source, path_to_module)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    requested = alias.name
                    if is_local_name(requested):
                        add_target(
                            source=source,
                            target_module=requested,
                            kind="absolute_import",
                            line=node.lineno,
                            requested_module=requested,
                        )
                    else:
                        external_modules.add(requested.split(".", 1)[0])
                continue

            if isinstance(node, ast.ImportFrom):
                requested = node.module or ""
                if node.level:
                    if current_package is None:
                        unresolved.append(
                            {
                                "source": source,
                                "line": node.lineno,
                                "kind": "relative_import",
                                "module": "." * node.level + requested,
                                "resolved_module": None,
                            }
                        )
                        continue
                    relative_name = "." * node.level + requested
                    try:
                        target_module = importlib.util.resolve_name(
                            relative_name,
                            current_package,
                        )
                    except (ImportError, ValueError) as error:
                        unresolved.append(
                            {
                                "source": source,
                                "line": node.lineno,
                                "kind": "relative_import",
                                "module": relative_name,
                                "resolved_module": None,
                                "error": str(error),
                            }
                        )
                        continue
                    kind = "relative_import"
                else:
                    target_module = requested
                    kind = "absolute_from_import"

                if not target_module:
                    unresolved.append(
                        {
                            "source": source,
                            "line": node.lineno,
                            "kind": kind,
                            "module": requested,
                            "resolved_module": None,
                        }
                    )
                    continue
                if not is_local_name(target_module):
                    external_modules.add(target_module.split(".", 1)[0])
                    continue
                base_found = add_target(
                    source=source,
                    target_module=target_module,
                    kind=kind,
                    line=node.lineno,
                    requested_module=(
                        "." * node.level + requested
                        if node.level
                        else requested
                    ),
                )
                if not base_found:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    child_module = f"{target_module}.{alias.name}"
                    if child_module in module_to_path:
                        add_target(
                            source=source,
                            target_module=child_module,
                            kind=f"{kind}_member_module",
                            line=node.lineno,
                            requested_module=child_module,
                        )
                continue

            if not isinstance(node, ast.Call):
                continue
            function = node.func
            dynamic_kind: str | None = None
            if isinstance(function, ast.Name) and function.id == "__import__":
                dynamic_kind = "__import__"
            elif (
                isinstance(function, ast.Attribute)
                and function.attr == "import_module"
                and isinstance(function.value, ast.Name)
                and function.value.id == "importlib"
            ):
                dynamic_kind = "importlib.import_module"
            if dynamic_kind is None:
                continue
            if (
                not node.args
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
            ):
                unresolved.append(
                    {
                        "source": source,
                        "line": node.lineno,
                        "kind": "dynamic_import",
                        "module": None,
                        "resolved_module": None,
                    }
                )
                continue
            requested = node.args[0].value
            if requested.startswith("."):
                unresolved.append(
                    {
                        "source": source,
                        "line": node.lineno,
                        "kind": "dynamic_import",
                        "module": requested,
                        "resolved_module": None,
                    }
                )
                continue
            local = is_local_name(requested)
            dynamic_imports.add(
                (
                    source,
                    node.lineno,
                    requested,
                    "local" if local else "external",
                )
            )
            if local:
                add_target(
                    source=source,
                    target_module=requested,
                    kind="dynamic_import",
                    line=node.lineno,
                    requested_module=requested,
                )
            else:
                external_modules.add(requested.split(".", 1)[0])

    if unresolved:
        ordered_unresolved = sorted(
            unresolved,
            key=lambda item: (
                str(item.get("source")),
                int(item.get("line", 0)),
                str(item.get("kind")),
                str(item.get("module")),
            ),
        )
        raise UnresolvedLocalImportError(ordered_unresolved)

    missing_required = sorted(set(normalized_required) - reached)
    if missing_required:
        raise MissingRequiredClosurePathError(missing_required)

    nodes = [
        {
            "path": path,
            "sha256": _file_sha256(_checked_source_path(root, path)),
        }
        for path in sorted(reached)
    ]
    edge_records = [
        {
            "from": source,
            "to": target,
            "kind": kind,
            "line": line,
            "module": module,
        }
        for source, target, kind, line, module in sorted(edges)
    ]
    dynamic_records = [
        {
            "source": source,
            "line": line,
            "module": module,
            "scope": scope,
        }
        for source, line, module, scope in sorted(dynamic_imports)
    ]
    manifest: dict[str, object] = {
        "schema_version": SOURCE_CLOSURE_SCHEMA,
        "formal_roots": list(normalized_formal_roots),
        "analysis_roots": list(normalized_analysis_roots),
        "required_paths": list(normalized_required),
        "nodes": nodes,
        "edges": edge_records,
        "dynamic_imports": dynamic_records,
        "external_modules": sorted(external_modules),
        "unresolved_local_imports": [],
    }
    manifest["closure_fingerprint"] = _sha256_bytes(
        _canonical_json(_fingerprint_payload(manifest)).encode("utf-8")
    )
    manifest.update(_git_metadata(root, set(reached)))
    validate_source_closure_manifest(manifest)
    return manifest


def build_nlcc_runner_source_closure(
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Build the fixed NLCC-v12 runner implementation closure."""

    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root)
    )
    return build_local_import_closure(
        root,
        formal_roots=FORMAL_ROOT_PATHS,
        analysis_roots=(
            *FORMAL_ROOT_PATHS,
            CLOSURE_BUILDER_ROOT_PATH,
        ),
        required_paths=REQUIRED_CLOSURE_PATHS,
        local_package_names=("cure_lite",),
    )


def verify_frozen_source_closure(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> dict[str, object]:
    """Verify source identity without treating unrelated dirtiness as drift."""

    expected_fingerprint = validate_source_closure_manifest(expected)
    actual_fingerprint = validate_source_closure_manifest(actual)
    expected_nodes = {
        str(node["path"]): str(node["sha256"])
        for node in expected["nodes"]  # type: ignore[index]
    }
    actual_nodes = {
        str(node["path"]): str(node["sha256"])
        for node in actual["nodes"]  # type: ignore[index]
    }
    changed_paths = sorted(
        path
        for path in set(expected_nodes) | set(actual_nodes)
        if expected_nodes.get(path) != actual_nodes.get(path)
    )
    if expected_fingerprint != actual_fingerprint:
        raise SourceClosureDriftError(
            expected_fingerprint=expected_fingerprint,
            actual_fingerprint=actual_fingerprint,
            changed_paths=changed_paths,
        )
    return {
        "schema_version": SOURCE_CLOSURE_SCHEMA,
        "closure_fingerprint": actual_fingerprint,
        "in_scope_closure_drift": False,
        "repository_dirty": actual.get("repository_dirty"),
        "in_scope_dirty_paths": list(
            actual.get("in_scope_dirty_paths", [])
        ),
        "out_of_scope_dirty_paths": list(
            actual.get("out_of_scope_dirty_paths", [])
        ),
        "all_pass": True,
    }
