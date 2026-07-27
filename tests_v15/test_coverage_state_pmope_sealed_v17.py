from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from cure_lite.experiment.coverage_state_pmope_sealed_v17 import (
    COVERAGE_STATE_PMOPE_V17_ARTIFACT_COUNT,
    COVERAGE_STATE_PMOPE_V17_COMPLETE_FINGERPRINT,
    COVERAGE_STATE_PMOPE_V17_COMPLETE_SHA256,
    COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES,
    COVERAGE_STATE_PMOPE_V17_DECISION_FINGERPRINT,
    COVERAGE_STATE_PMOPE_V17_RESULT_FINGERPRINT,
    COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH,
    COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_REPO_PATH,
    COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_SHA256,
    COVERAGE_STATE_PMOPE_V17_SOURCE_FILE_COUNT,
    COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_REPO_PATH,
    COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_SHA256,
    CoverageStatePMOPESealedV17Receipt,
    verify_coverage_state_pmope_sealed_v17_controls,
)


_ROOT = Path(__file__).resolve().parents[1]
_RUN = _ROOT / COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH
_MANIFEST = (
    _ROOT / COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_REPO_PATH
)
_ARCHIVE = _ROOT / COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_REPO_PATH


@pytest.fixture(scope="module")
def receipt() -> CoverageStatePMOPESealedV17Receipt:
    return verify_coverage_state_pmope_sealed_v17_controls(
        _RUN,
        source_manifest_path=_MANIFEST,
        source_archive_path=_ARCHIVE,
    )


def test_real_sealed_v17_controls_are_strictly_bound(
    receipt: CoverageStatePMOPESealedV17Receipt,
) -> None:
    payload = receipt.canonical_payload()

    assert receipt.complete_fingerprint == (
        COVERAGE_STATE_PMOPE_V17_COMPLETE_FINGERPRINT
    )
    assert receipt.complete_file_sha256 == (
        COVERAGE_STATE_PMOPE_V17_COMPLETE_SHA256
    )
    assert receipt.decision_fingerprint == (
        COVERAGE_STATE_PMOPE_V17_DECISION_FINGERPRINT
    )
    assert receipt.bounded_result_fingerprint == (
        COVERAGE_STATE_PMOPE_V17_RESULT_FINGERPRINT
    )
    assert receipt.source_manifest_file_sha256 == (
        COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_SHA256
    )
    assert receipt.source_archive_file_sha256 == (
        COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_SHA256
    )
    assert len(receipt.artifact_files) == (
        COVERAGE_STATE_PMOPE_V17_ARTIFACT_COUNT
    )
    assert len(receipt.source_members) == (
        COVERAGE_STATE_PMOPE_V17_SOURCE_FILE_COUNT
    )
    assert tuple(value.objective for value in receipt.controls) == (
        COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES
    )
    assert payload["historical_frozen_controls"] is True
    assert payload["contemporaneous_controls"] is False
    assert payload["control_outcomes_are_not_candidate_gates"] is True
    assert payload["model_deserialization_performed"] is False
    assert payload["evaluator_called"] is False
    assert payload["training_performed"] is False
    assert payload["D_R_cached_tensor_payload_accessed"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["runtime_splits"] == []
    assert all(payload["checks"].values())


def test_repeated_verification_is_canonically_deterministic(
    receipt: CoverageStatePMOPESealedV17Receipt,
) -> None:
    replay = verify_coverage_state_pmope_sealed_v17_controls(
        _RUN,
        source_manifest_path=_MANIFEST,
        source_archive_path=_ARCHIVE,
    )
    assert replay.canonical_payload() == receipt.canonical_payload()
    assert replay.receipt_fingerprint == receipt.receipt_fingerprint


def _relocated_evidence(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    run = tmp_path / "sealed-run"
    manifest = tmp_path / "source-closure.json"
    archive = tmp_path / "source-closure.tar"
    shutil.copytree(_RUN, run)
    shutil.copy2(_MANIFEST, manifest)
    shutil.copy2(_ARCHIVE, archive)
    return run, manifest, archive


def test_caller_can_pass_a_relocated_fixed_historical_run(
    tmp_path: Path,
    receipt: CoverageStatePMOPESealedV17Receipt,
) -> None:
    run, manifest, archive = _relocated_evidence(tmp_path)
    relocated = verify_coverage_state_pmope_sealed_v17_controls(
        run,
        source_manifest_path=manifest,
        source_archive_path=archive,
    )
    assert relocated.canonical_payload() == receipt.canonical_payload()
    assert relocated.receipt_fingerprint == receipt.receipt_fingerprint


def test_checkpoint_byte_tampering_is_rejected(tmp_path: Path) -> None:
    run, manifest, archive = _relocated_evidence(tmp_path)
    checkpoint = run / "checkpoints/identity_joint.safetensors"
    content = bytearray(checkpoint.read_bytes())
    content[-1] ^= 1
    checkpoint.write_bytes(content)

    with pytest.raises(RuntimeError, match="artifact hash changed"):
        verify_coverage_state_pmope_sealed_v17_controls(
            run,
            source_manifest_path=manifest,
            source_archive_path=archive,
        )


def test_coherently_rewritten_complete_is_still_rejected(
    tmp_path: Path,
) -> None:
    run, manifest, archive = _relocated_evidence(tmp_path)
    path = run / "COMPLETE.json"
    complete = json.loads(path.read_text(encoding="utf-8"))
    complete["status"] = "changed"
    unsigned = dict(complete)
    unsigned.pop("complete_fingerprint")
    from cure_lite.cache.schema import stable_fingerprint

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
        verify_coverage_state_pmope_sealed_v17_controls(
            run,
            source_manifest_path=manifest,
            source_archive_path=archive,
        )


def test_source_archive_tampering_is_rejected(tmp_path: Path) -> None:
    run, manifest, archive = _relocated_evidence(tmp_path)
    content = bytearray(archive.read_bytes())
    content[-1] ^= 1
    archive.write_bytes(content)

    with pytest.raises(RuntimeError, match="source closure manifest changed"):
        verify_coverage_state_pmope_sealed_v17_controls(
            run,
            source_manifest_path=manifest,
            source_archive_path=archive,
        )


def test_unexpected_run_file_is_rejected(tmp_path: Path) -> None:
    run, manifest, archive = _relocated_evidence(tmp_path)
    (run / "unexpected.txt").write_text("not sealed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="run tree is not exact"):
        verify_coverage_state_pmope_sealed_v17_controls(
            run,
            source_manifest_path=manifest,
            source_archive_path=archive,
        )
