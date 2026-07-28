from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

import cure_lite.experiment.coverage_state_uscope_sealed_v18 as sealed_module
from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.coverage_state_uscope_sealed_v18 import (
    COVERAGE_STATE_USCOPE_V18_ARTIFACT_COUNT,
    COVERAGE_STATE_USCOPE_V18_COMPLETE_FINGERPRINT,
    COVERAGE_STATE_USCOPE_V18_COMPLETE_SHA256,
    COVERAGE_STATE_USCOPE_V18_DECISION_FINGERPRINT,
    COVERAGE_STATE_USCOPE_V18_RESULT_FINGERPRINT,
    COVERAGE_STATE_USCOPE_V18_RUN_REPO_PATH,
    COVERAGE_STATE_USCOPE_V18_SEALED_RECEIPT_FINGERPRINT,
    COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_REPO_PATH,
    COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_SHA256,
    COVERAGE_STATE_USCOPE_V18_SOURCE_FILE_COUNT,
    COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_REPO_PATH,
    COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_SHA256,
    CoverageStateUSCOPESealedV18Receipt,
    verify_coverage_state_uscope_sealed_v18_negative,
    verify_repository_coverage_state_uscope_sealed_v18,
)


_ROOT = Path(__file__).resolve().parents[1]
_RUN = _ROOT / COVERAGE_STATE_USCOPE_V18_RUN_REPO_PATH
_MANIFEST = (
    _ROOT / COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_REPO_PATH
)
_ARCHIVE = _ROOT / COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_REPO_PATH


@pytest.fixture(scope="module")
def receipt() -> CoverageStateUSCOPESealedV18Receipt:
    return verify_repository_coverage_state_uscope_sealed_v18(_ROOT)


def test_real_v18_negative_result_is_exactly_bound(
    receipt: CoverageStateUSCOPESealedV18Receipt,
) -> None:
    payload = receipt.canonical_payload()

    assert (
        receipt.complete_fingerprint
        == COVERAGE_STATE_USCOPE_V18_COMPLETE_FINGERPRINT
    )
    assert (
        receipt.complete_file_sha256
        == COVERAGE_STATE_USCOPE_V18_COMPLETE_SHA256
    )
    assert (
        receipt.decision_fingerprint
        == COVERAGE_STATE_USCOPE_V18_DECISION_FINGERPRINT
    )
    assert (
        receipt.bounded_result_fingerprint
        == COVERAGE_STATE_USCOPE_V18_RESULT_FINGERPRINT
    )
    assert (
        receipt.source_manifest_file_sha256
        == COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_SHA256
    )
    assert (
        receipt.source_archive_file_sha256
        == COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_SHA256
    )
    assert len(receipt.artifact_files) == (
        COVERAGE_STATE_USCOPE_V18_ARTIFACT_COUNT
    )
    assert len(receipt.source_members) == (
        COVERAGE_STATE_USCOPE_V18_SOURCE_FILE_COUNT
    )
    assert receipt.receipt_fingerprint == (
        COVERAGE_STATE_USCOPE_V18_SEALED_RECEIPT_FINGERPRINT
    )
    assert payload["negative_result"]["bounded_gate_passed"] is False
    assert payload["negative_result"]["seed"] == 42
    assert payload["negative_result"]["completed_updates"] == 400
    assert payload["historical_negative_result"] is True
    assert payload["contemporaneous_candidate_result"] is False
    assert payload["checkpoint_treated_as_opaque_bytes"] is True
    assert payload["model_deserialization_performed"] is False
    assert payload["evaluator_called"] is False
    assert payload["training_performed"] is False
    assert payload["D_R_cached_tensor_payload_accessed"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["runtime_splits"] == []
    assert all(payload["checks"].values())


def test_verifier_source_has_no_model_loading_or_training_imports() -> None:
    source_path = Path(sealed_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.partition(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.lstrip(".").partition(".")[0])

    assert "torch" not in imported_roots
    assert "safetensors" not in imported_roots
    assert "train" not in imported_roots
    assert "coverage_state_zero_level_evaluation" not in imported_roots


def test_replay_is_deterministic_and_read_only(
    receipt: CoverageStateUSCOPESealedV18Receipt,
) -> None:
    watched = tuple(
        sorted(
            (
                path,
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            for path in (*_RUN.rglob("*"), _MANIFEST, _ARCHIVE)
            if path.is_file()
        )
    )
    replay = verify_repository_coverage_state_uscope_sealed_v18(_ROOT)
    watched_after = tuple(
        sorted(
            (
                path,
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            for path in (*_RUN.rglob("*"), _MANIFEST, _ARCHIVE)
            if path.is_file()
        )
    )

    assert replay == receipt
    assert replay.receipt_fingerprint == receipt.receipt_fingerprint
    assert watched_after == watched
    receipt.verify_unchanged(_ROOT)


def _relocated_evidence(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    run = tmp_path / "sealed-v18-run"
    manifest = tmp_path / "source-closure.json"
    archive = tmp_path / "source-closure.tar"
    shutil.copytree(_RUN, run)
    shutil.copy2(_MANIFEST, manifest)
    shutil.copy2(_ARCHIVE, archive)
    return run, manifest, archive


def test_relocated_byte_identical_evidence_is_accepted(
    tmp_path: Path,
    receipt: CoverageStateUSCOPESealedV18Receipt,
) -> None:
    run, manifest, archive = _relocated_evidence(tmp_path)
    relocated = verify_coverage_state_uscope_sealed_v18_negative(
        run,
        source_manifest_path=manifest,
        source_archive_path=archive,
    )
    assert relocated == receipt
    assert relocated.receipt_fingerprint == receipt.receipt_fingerprint


def test_opaque_checkpoint_byte_change_is_rejected(
    tmp_path: Path,
) -> None:
    run, manifest, archive = _relocated_evidence(tmp_path)
    checkpoint = run / "checkpoints/pmope_joint.safetensors"
    content = bytearray(checkpoint.read_bytes())
    content[-1] ^= 1
    checkpoint.write_bytes(content)

    with pytest.raises(RuntimeError, match="artifact hash changed"):
        verify_coverage_state_uscope_sealed_v18_negative(
            run,
            source_manifest_path=manifest,
            source_archive_path=archive,
        )


def test_coherently_rewritten_complete_is_rejected(
    tmp_path: Path,
) -> None:
    run, manifest, archive = _relocated_evidence(tmp_path)
    path = run / "COMPLETE.json"
    complete = json.loads(path.read_text(encoding="utf-8"))
    complete["bounded_gate_passed"] = True
    unsigned = dict(complete)
    unsigned.pop("complete_fingerprint")
    complete["complete_fingerprint"] = stable_fingerprint(unsigned)
    path.write_text(
        json.dumps(
            complete,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="COMPLETE binding changed"):
        verify_coverage_state_uscope_sealed_v18_negative(
            run,
            source_manifest_path=manifest,
            source_archive_path=archive,
        )


def test_source_archive_change_is_rejected(tmp_path: Path) -> None:
    run, manifest, archive = _relocated_evidence(tmp_path)
    content = bytearray(archive.read_bytes())
    content[-1] ^= 1
    archive.write_bytes(content)

    with pytest.raises(
        RuntimeError,
        match="source closure manifest changed",
    ):
        verify_coverage_state_uscope_sealed_v18_negative(
            run,
            source_manifest_path=manifest,
            source_archive_path=archive,
        )


def test_unexpected_run_file_is_rejected(tmp_path: Path) -> None:
    run, manifest, archive = _relocated_evidence(tmp_path)
    (run / "unexpected.txt").write_text("not sealed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="run tree is not exact"):
        verify_coverage_state_uscope_sealed_v18_negative(
            run,
            source_manifest_path=manifest,
            source_archive_path=archive,
        )


def test_receipt_cannot_be_mutated_into_a_positive_result(
    receipt: CoverageStateUSCOPESealedV18Receipt,
) -> None:
    with pytest.raises(ValueError, match="binding changed"):
        replace(
            receipt,
            decision_fingerprint="0" * 64,
        )
