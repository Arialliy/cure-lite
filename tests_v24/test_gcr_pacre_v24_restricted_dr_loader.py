from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

import cure_lite.experiment.cache_pipeline as cache_pipeline
from cure_lite.experiment.cache_pipeline import (
    cache_d_r_states,
    cache_manifest_split,
    load_d_r_cache_bundle,
)
from cure_lite.toy import ToyFrozenBaseAdapter
from tests.test_experiment_cache_pipeline import (
    _dataset,
    _manifest_dataset_root,
)


def test_restricted_loader_never_opens_or_hashes_holdout_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = _dataset(manifest_path, manifest, "D_R")
    adapter = ToyFrozenBaseAdapter()
    base_output = tmp_path / "base-output"
    cache_manifest_split(adapter, dataset, "D_R", base_output)
    state_output = tmp_path / "state-output"
    cache_d_r_states(
        base_output / "index.json",
        dataset,
        state_output,
        expected_base_fingerprint=adapter.fingerprint,
    )

    state_index = json.loads(
        (state_output / "index.json").read_text(encoding="utf-8")
    )
    base_index = json.loads(
        (base_output / "index.json").read_text(encoding="utf-8")
    )
    holdout_id = "z-dr"
    state_row = next(
        row
        for row in state_index["records"]
        if row["sample_id"] == holdout_id
    )
    base_row = next(
        row
        for row in base_index["records"]
        if row["sample_id"] == holdout_id
    )
    forbidden = {
        Path(state_row["image_path"]).resolve(),
        Path(state_row["mask_path"]).resolve(),
        (state_output / state_row["state_cache_path"]).resolve(),
        (base_output / base_row["cache_path"]).resolve(),
    }

    original_hash = cache_pipeline.file_sha256
    original_path_open = Path.open
    original_image_open = Image.open
    observed_forbidden: list[Path] = []

    def guarded_hash(path):
        resolved = Path(path).resolve()
        if resolved in forbidden:
            observed_forbidden.append(resolved)
            raise AssertionError("holdout payload was hashed")
        return original_hash(path)

    def guarded_path_open(self, *args, **kwargs):
        resolved = self.resolve()
        if resolved in forbidden:
            observed_forbidden.append(resolved)
            raise AssertionError("holdout payload was opened")
        return original_path_open(self, *args, **kwargs)

    def guarded_image_open(fp, *args, **kwargs):
        if isinstance(fp, (str, Path)):
            resolved = Path(fp).resolve()
            if resolved in forbidden:
                observed_forbidden.append(resolved)
                raise AssertionError("holdout image/mask was opened")
        return original_image_open(fp, *args, **kwargs)

    monkeypatch.setattr(cache_pipeline, "file_sha256", guarded_hash)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(Image, "open", guarded_image_open)

    bundle = load_d_r_cache_bundle(
        state_output / "index.json",
        dataset,
        expected_base_fingerprint=adapter.fingerprint,
        allowed_sample_ids=("a-dr",),
    )
    assert tuple(row.sample_id for row in bundle.rows) == ("a-dr",)
    assert observed_forbidden == []


def test_restricted_loader_defers_holdout_integrity_check_until_open(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = _dataset(manifest_path, manifest, "D_R")
    adapter = ToyFrozenBaseAdapter()
    base_output = tmp_path / "base-output"
    cache_manifest_split(adapter, dataset, "D_R", base_output)
    state_output = tmp_path / "state-output"
    cache_d_r_states(
        base_output / "index.json",
        dataset,
        state_output,
        expected_base_fingerprint=adapter.fingerprint,
    )
    index = json.loads(
        (state_output / "index.json").read_text(encoding="utf-8")
    )
    holdout_row = next(
        row for row in index["records"] if row["sample_id"] == "z-dr"
    )
    holdout_state = (
        state_output / holdout_row["state_cache_path"]
    ).resolve()
    holdout_state.write_bytes(holdout_state.read_bytes() + b"tampered")

    train = load_d_r_cache_bundle(
        state_output / "index.json",
        dataset,
        expected_base_fingerprint=adapter.fingerprint,
        allowed_sample_ids=("a-dr",),
    )
    assert tuple(row.sample_id for row in train.rows) == ("a-dr",)
    with pytest.raises(ValueError, match="state cache SHA256"):
        load_d_r_cache_bundle(
            state_output / "index.json",
            dataset,
            expected_base_fingerprint=adapter.fingerprint,
            allowed_sample_ids=("z-dr",),
        )
