from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile

import pytest

from cure_lite.experiment import stage_a_runner
from cure_lite.experiment.artifacts import DECODER_ARTIFACT_SCHEMA_V2
from cure_lite.experiment.stage_a_m_extension import (
    load_stage_a_reference_snapshot,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REAL_RUNS = (
    (
        _PROJECT_ROOT
        / "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3",
        42,
    ),
    (
        _PROJECT_ROOT
        / "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3_s43",
        43,
    ),
)
_HISTORICAL_SOURCE_DIGEST = (
    "3954aae1b548c1c484b6490c436315961840705844b4c0e9a520ea1db633e042"
)


def _require_real_run(path: Path) -> None:
    if not (path / "COMPLETE.json").is_file():
        pytest.skip(f"historical Stage-A fixture is unavailable: {path}")


def _hardlink_clone(source: Path) -> tuple[Path, Path]:
    """Clone inside the same filesystem without duplicating the large caches."""

    temporary_root = Path(
        tempfile.mkdtemp(prefix=".stage-a-reference-test-", dir=source.parent)
    )
    clone = temporary_root / "snapshot"
    shutil.copytree(source, clone, copy_function=os.link)
    return temporary_root, clone


@pytest.mark.parametrize(("root", "seed"), _REAL_RUNS)
def test_loads_real_historical_snapshots_without_current_source_digest(
    root: Path,
    seed: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_real_run(root)

    def forbidden_current_digest() -> str:
        raise AssertionError("historical loader must not inspect current source digest")

    monkeypatch.setattr(stage_a_runner, "_source_tree_digest", forbidden_current_digest)
    snapshot = load_stage_a_reference_snapshot(root)

    assert snapshot.config.training.global_seed == seed
    assert snapshot.source_tree_digest == _HISTORICAL_SOURCE_DIGEST
    assert len(snapshot.snapshot_fingerprint) == 64
    assert snapshot.factual_artifact.config.schema_version == (
        DECODER_ARTIFACT_SCHEMA_V2
    )
    assert snapshot.factual_artifact.config.variant == "factual_only"
    assert (
        snapshot.factual_exposure_matched_artifact.config.variant
        == "factual_exposure_matched"
    )
    assert snapshot.uniform_artifact.config.variant == "uniform_legal"
    assert len(
        {
            snapshot.factual_artifact.config.initial_decoder_fingerprint,
            snapshot.factual_exposure_matched_artifact.config.initial_decoder_fingerprint,
            snapshot.uniform_artifact.config.initial_decoder_fingerprint,
        }
    ) == 1


def test_snapshot_is_sealed_and_verify_unchanged_rechecks_inventory() -> None:
    source, _ = _REAL_RUNS[0]
    _require_real_run(source)
    temporary_root, clone = _hardlink_clone(source)
    try:
        snapshot = load_stage_a_reference_snapshot(clone)
        with pytest.raises(TypeError, match="fields were replaced"):
            replace(snapshot, snapshot_fingerprint="f" * 64)

        added = clone / "unexpected.json"
        added.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="directory inventory changed|file inventory changed"):
            snapshot.verify_unchanged()
    finally:
        shutil.rmtree(temporary_root)


def test_tampered_reference_copy_is_rejected_by_actual_sha_inventory() -> None:
    source, _ = _REAL_RUNS[0]
    _require_real_run(source)
    temporary_root, clone = _hardlink_clone(source)
    try:
        support_path = clone / "receipts" / "support.json"
        payload = json.loads(support_path.read_text(encoding="utf-8"))
        payload["summary"]["source_images"] += 1

        # Break this one hardlink before writing so the historical source remains
        # byte-for-byte untouched.
        support_path.unlink()
        support_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="artifact file inventory changed"):
            load_stage_a_reference_snapshot(clone)
    finally:
        shutil.rmtree(temporary_root)


def test_reference_root_symlink_is_rejected(tmp_path: Path) -> None:
    source, _ = _REAL_RUNS[0]
    _require_real_run(source)
    link = tmp_path / "stage-a-link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_stage_a_reference_snapshot(link)
