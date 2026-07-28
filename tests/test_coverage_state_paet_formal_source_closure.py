from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from cure_lite.coverage_state_paet_formal_source_closure import (
    ARCHIVE_REPO_PATH,
    MANIFEST_REPO_PATH,
    FormalSourceClosureError,
    build_coverage_state_paet_formal_source_closure,
    formal_source_closure_paths,
    verify_coverage_state_paet_formal_source_closure,
)
from tools import build_coverage_state_paet_formal_source_closure as builder_cli


def _write(path: Path, text: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _root(tmp_path: Path) -> Path:
    _write(tmp_path / "pyproject.toml", "[project]\nname = 'closure-test'\n")
    _write(tmp_path / "cure_lite/__init__.py")
    _write(tmp_path / "cure_lite/experiment/__init__.py")
    _write(tmp_path / "cure_lite/experiment/runtime.py")
    _write(tmp_path / "tools/__init__.py")
    for name in (
        "run_coverage_state_cslf_ppce_support_oriented_bounded_400.py",
        "run_coverage_state_cslf_support_oriented_bounded_400.py",
        "run_coverage_state_cmif_pmope_bounded_400.py",
        "run_coverage_state_paet_bfa_pmope_bounded_400.py",
        "run_coverage_state_paet_bfa_pmope_formal_800.py",
        "run_coverage_state_paet_bfa_pmope_formal_d_v.py",
        "run_with_gpu_temperature_control.py",
    ):
        _write(tmp_path / "tools" / name)
    return tmp_path


def test_create_is_deterministic_and_validate_seals_live_tree(tmp_path: Path):
    left = _root(tmp_path / "left")
    right = _root(tmp_path / "right")
    left_receipt = build_coverage_state_paet_formal_source_closure(left)
    right_receipt = build_coverage_state_paet_formal_source_closure(right)

    assert left_receipt == verify_coverage_state_paet_formal_source_closure(left)
    assert left_receipt["content_fingerprint"] == right_receipt["content_fingerprint"]
    assert (left / ARCHIVE_REPO_PATH).read_bytes() == (right / ARCHIVE_REPO_PATH).read_bytes()
    assert (left / MANIFEST_REPO_PATH).read_bytes() == (right / MANIFEST_REPO_PATH).read_bytes()
    with pytest.raises(FileExistsError):
        build_coverage_state_paet_formal_source_closure(left)


def test_transitive_local_tool_imports_join_the_closure(tmp_path: Path):
    root = _root(tmp_path)
    dependency = root / "tools/formal_transitive_dependency.py"
    _write(dependency)
    formal_cli = root / "tools/run_coverage_state_paet_bfa_pmope_formal_800.py"
    _write(
        formal_cli,
        "from tools import formal_transitive_dependency\n",
    )
    assert (
        "tools/formal_transitive_dependency.py"
        in formal_source_closure_paths(root)
    )


def test_verifier_rejects_member_tampering_extra_member_and_live_drift(tmp_path: Path):
    root = _root(tmp_path)
    build_coverage_state_paet_formal_source_closure(root)
    archive = root / ARCHIVE_REPO_PATH
    with tarfile.open(archive, "a") as tar:
        info = tarfile.TarInfo("unexpected.py")
        info.size = 1
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(b"x"))
    manifest_path = root / MANIFEST_REPO_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["archive"]["bytes"] = archive.stat().st_size
    manifest["archive"]["sha256"] = hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(FormalSourceClosureError, match="member inventory"):
        verify_coverage_state_paet_formal_source_closure(root)

    # Recreate a separate sealed root to exercise current-worktree hashing.
    root = _root(tmp_path / "live")
    build_coverage_state_paet_formal_source_closure(root)
    _write(root / "cure_lite/experiment/runtime.py", "x = 2\n")
    with pytest.raises(FormalSourceClosureError, match="current Formal800 source differs"):
        verify_coverage_state_paet_formal_source_closure(root)


def test_builder_rejects_symlink_and_cli_has_only_create_or_validate(
    tmp_path: Path,
):
    root = _root(tmp_path)
    target = root / "cure_lite/experiment/runtime.py"
    target.unlink()
    target.symlink_to(root / "cure_lite/__init__.py")
    with pytest.raises(FormalSourceClosureError, match="canonical"):
        build_coverage_state_paet_formal_source_closure(root)
    assert builder_cli.parse_args(("--create-once",)).create_once
    assert builder_cli.parse_args(("--validate",)).validate
    with pytest.raises(SystemExit):
        builder_cli.parse_args(())
    with pytest.raises(SystemExit):
        builder_cli.parse_args(("--create-once", "--validate"))
