from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

from cure_lite.nlcc_runner_source_closure import (
    FORMAL_ROOT_PATHS,
    REQUIRED_CLOSURE_PATHS,
    MissingRequiredClosurePathError,
    SourceClosureDriftError,
    UnresolvedLocalImportError,
    build_local_import_closure,
    build_nlcc_runner_source_closure,
    validate_source_closure_manifest,
    verify_frozen_source_closure,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_repository(root: Path) -> None:
    (root / "pkg/sub").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "pkg/__init__.py").write_text(
        "from . import api\n",
        encoding="utf-8",
    )
    (root / "pkg/api.py").write_text(
        "\n".join(
            (
                "import importlib",
                "import numpy",
                "from .sub.logic import VALUE",
                "if False:",
                "    from .conditional import CONDITIONAL",
                'importlib.import_module("pkg.dynamic")',
                "",
            )
        ),
        encoding="utf-8",
    )
    (root / "pkg/sub/__init__.py").write_text(
        "from .logic import VALUE\n",
        encoding="utf-8",
    )
    (root / "pkg/sub/logic.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (root / "pkg/conditional.py").write_text(
        "CONDITIONAL = 2\n",
        encoding="utf-8",
    )
    (root / "pkg/dynamic.py").write_text(
        "DYNAMIC = 3\n",
        encoding="utf-8",
    )
    (root / "tools/run.py").write_text(
        "from pkg.api import VALUE\n",
        encoding="utf-8",
    )


def _build_minimal(root: Path) -> dict[str, object]:
    return build_local_import_closure(
        root,
        formal_roots=("tools/run.py",),
        required_paths=(
            "pkg/__init__.py",
            "pkg/sub/__init__.py",
            "pkg/sub/logic.py",
            "pkg/conditional.py",
            "pkg/dynamic.py",
        ),
        local_package_names=("pkg",),
    )


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_fixed_nlcc_closure_is_complete_ordered_and_deterministic() -> None:
    first = build_nlcc_runner_source_closure(ROOT)
    second = build_nlcc_runner_source_closure(ROOT)

    assert first["formal_roots"] == sorted(FORMAL_ROOT_PATHS)
    assert first["closure_fingerprint"] == second["closure_fingerprint"]
    assert validate_source_closure_manifest(first) == (
        first["closure_fingerprint"]
    )
    node_paths = [node["path"] for node in first["nodes"]]
    assert node_paths == sorted(node_paths)
    assert set(REQUIRED_CLOSURE_PATHS).issubset(node_paths)
    assert "cure_lite/nlcc_runner_source_closure.py" in node_paths
    assert "cure_lite/__init__.py" in node_paths
    assert "cure_lite/cache/__init__.py" in node_paths
    assert "cure_lite/train/__init__.py" in node_paths
    assert "cure_lite/decoder.py" in node_paths
    assert "cure_lite/factorized_config.py" in node_paths
    assert "cure_lite/factorized_decoder.py" in node_paths
    edge_keys = [
        (
            edge["from"],
            edge["to"],
            edge["kind"],
            edge["line"],
            edge["module"],
        )
        for edge in first["edges"]
    ]
    assert edge_keys == sorted(edge_keys)
    assert first["unresolved_local_imports"] == []


def test_static_union_includes_conditioned_dynamic_and_initializers(
    tmp_path: Path,
) -> None:
    _write_minimal_repository(tmp_path)

    manifest = _build_minimal(tmp_path)

    node_paths = {node["path"] for node in manifest["nodes"]}
    assert node_paths == {
        "pkg/__init__.py",
        "pkg/api.py",
        "pkg/conditional.py",
        "pkg/dynamic.py",
        "pkg/sub/__init__.py",
        "pkg/sub/logic.py",
        "tools/run.py",
    }
    assert manifest["dynamic_imports"] == [
        {
            "source": "pkg/api.py",
            "line": 6,
            "module": "pkg.dynamic",
            "scope": "local",
        }
    ]
    assert "numpy" in manifest["external_modules"]
    assert any(
        edge["to"] == "pkg/sub/__init__.py"
        and edge["kind"] == "package_initializer"
        for edge in manifest["edges"]
    )


def test_unresolved_local_import_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "tools/run.py").write_text(
        "from pkg.missing import VALUE\n",
        encoding="utf-8",
    )

    with pytest.raises(UnresolvedLocalImportError) as captured:
        build_local_import_closure(
            tmp_path,
            formal_roots=("tools/run.py",),
            local_package_names=("pkg",),
        )

    assert captured.value.unresolved[0]["resolved_module"] == "pkg.missing"


def test_required_path_must_be_reached_not_merely_exist(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg/unused.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tools/run.py").write_text(
        "import pkg\n",
        encoding="utf-8",
    )

    with pytest.raises(MissingRequiredClosurePathError) as captured:
        build_local_import_closure(
            tmp_path,
            formal_roots=("tools/run.py",),
            required_paths=("pkg/unused.py",),
            local_package_names=("pkg",),
        )

    assert captured.value.missing_paths == ("pkg/unused.py",)


def test_only_closure_content_drift_blocks_verification(
    tmp_path: Path,
) -> None:
    _write_minimal_repository(tmp_path)
    (tmp_path / "notes.md").write_text("outside\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(
        tmp_path,
        "-c",
        "user.name=NLCC Test",
        "-c",
        "user.email=nlcc@example.invalid",
        "commit",
        "-qm",
        "freeze",
    )
    frozen = _build_minimal(tmp_path)
    assert frozen["repository_dirty"] is False

    (tmp_path / "notes.md").write_text("unrelated change\n", encoding="utf-8")
    unrelated_dirty = _build_minimal(tmp_path)
    assert unrelated_dirty["closure_fingerprint"] == (
        frozen["closure_fingerprint"]
    )
    assert unrelated_dirty["in_scope_dirty_paths"] == []
    assert unrelated_dirty["out_of_scope_dirty_paths"] == ["notes.md"]
    assert verify_frozen_source_closure(
        frozen,
        unrelated_dirty,
    )["all_pass"] is True

    (tmp_path / "pkg/sub/logic.py").write_text(
        "VALUE = 99\n",
        encoding="utf-8",
    )
    in_scope_dirty = _build_minimal(tmp_path)
    assert in_scope_dirty["in_scope_dirty_paths"] == ["pkg/sub/logic.py"]
    with pytest.raises(SourceClosureDriftError) as captured:
        verify_frozen_source_closure(frozen, in_scope_dirty)
    assert captured.value.changed_paths == ("pkg/sub/logic.py",)


def test_repository_metadata_is_not_part_of_closure_fingerprint() -> None:
    manifest = build_nlcc_runner_source_closure(ROOT)
    changed_metadata = deepcopy(manifest)
    changed_metadata["repository_commit"] = "0" * 40
    changed_metadata["repository_dirty"] = True
    changed_metadata["repository_dirty_paths"] = ["unrelated/report.md"]
    changed_metadata["in_scope_dirty_paths"] = []
    changed_metadata["out_of_scope_dirty_paths"] = [
        "unrelated/report.md"
    ]

    assert validate_source_closure_manifest(changed_metadata) == (
        manifest["closure_fingerprint"]
    )
    assert verify_frozen_source_closure(
        manifest,
        changed_metadata,
    )["all_pass"] is True
