from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from cure_lite.cache.base_cache import load_base_cache
from cure_lite.cache.schema import STATE_CACHE_SCHEMA, file_sha256, stable_fingerprint
from cure_lite.cache.state_cache import load_state_cache
from cure_lite.config import InterventionConfig, MatchConfig, OccupancyConfig
from cure_lite.data import ManifestImageDataset, PreprocessConfig
from cure_lite.experiment.cache_pipeline import (
    MANIFEST_BASE_CACHE_INDEX_SCHEMA,
    MANIFEST_STATE_CACHE_INDEX_SCHEMA,
    build_state_record,
    cache_d_r_states,
    cache_manifest_split,
    load_d_r_cache_bundle,
    load_d_v_cache_bundle,
)
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.toy import ToyFrozenBaseAdapter
from cure_lite.types import FrozenBaseOutput


def _base_output(occupancy: torch.Tensor) -> FrozenBaseOutput:
    probability = torch.where(
        occupancy,
        torch.tensor(0.9, dtype=torch.float32),
        torch.tensor(0.1, dtype=torch.float32),
    )[None, None]
    feature = torch.zeros(
        (1, 3, *occupancy.shape),
        dtype=torch.float32,
    )
    return FrozenBaseOutput(probability=probability, feature=feature)


def test_build_state_record_retains_complete_factual_reachability_catalog() -> None:
    gt_mask = torch.tensor(
        [
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    occupancy = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=torch.bool,
    )

    record = build_state_record(
        "multi-reachable",
        _base_output(occupancy),
        gt_mask,
    )

    assert torch.equal(record.occupancy, occupancy)
    assert record.real_miss_ids.tolist() == [1, 4]
    assert record.reachable_miss_ids.tolist() == [1, 4]
    assert record.base_match_pairs.ndim == 2
    assert record.base_match_pairs.shape[1] == 2
    assert record.legal_pairs.ndim == 2
    assert record.legal_pairs.shape[1] == 2


def test_build_state_record_retains_all_legal_deletion_pairs() -> None:
    occupancy = torch.zeros((8, 8), dtype=torch.bool)
    occupancy[1, 1] = True
    occupancy[6, 6] = True
    gt_mask = occupancy.clone()

    record = build_state_record(
        "two-covered-targets",
        _base_output(occupancy),
        gt_mask,
    )

    assert record.base_match_pairs.tolist() == [[1, 1], [2, 2]]
    assert record.real_miss_ids.numel() == 0
    assert record.reachable_miss_ids.numel() == 0
    assert record.legal_pairs.tolist() == [[1, 1], [2, 2]]


def test_build_state_record_masks_invalid_pixels_before_component_labeling() -> None:
    occupancy = torch.zeros((6, 6), dtype=torch.bool)
    occupancy[1, 1] = True
    occupancy[4, 4] = True
    gt_mask = occupancy.clone()
    valid = torch.zeros((1, 1, 6, 6), dtype=torch.bool)
    valid[..., :3, :3] = True

    record = build_state_record(
        "padded",
        _base_output(occupancy),
        gt_mask,
        image_valid_mask=valid,
    )

    assert record.pred_labels.max().item() == 1
    assert record.gt_labels.max().item() == 1
    assert record.base_match_pairs.tolist() == [[1, 1]]
    assert not torch.any(record.occupancy & ~record.image_valid_mask)
    assert not torch.any((record.gt_labels > 0) & ~record.image_valid_mask)


def test_build_state_record_rejects_batched_base_output() -> None:
    probability = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    feature = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="exactly one"):
        build_state_record(
            "batched",
            FrozenBaseOutput(probability=probability, feature=feature),
            torch.zeros((4, 4), dtype=torch.bool),
        )


def _manifest_dataset_root(tmp_path: Path) -> tuple[Path, SplitManifest]:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    identities = ("db", "z-dr", "a-dr", "z-dv", "a-dv", "dt")
    for index, sample_id in enumerate(identities):
        image = np.zeros((8, 8), dtype=np.uint8)
        image[1 + index % 4, 1 + index % 5] = 255
        mask = (image > 0).astype(np.uint8) * 255
        Image.fromarray(image, mode="L").save(images / f"{sample_id}.png")
        Image.fromarray(mask, mode="L").save(masks / f"{sample_id}.png")

    records = (
        SplitRecord("db", "D_B", "db", "images/db.png", "masks/db.png"),
        # Deliberately non-canonical row order: the cache index must sort IDs.
        SplitRecord(
            "z-dr",
            "D_R",
            "z-dr",
            "images/z-dr.png",
            "masks/z-dr.png",
        ),
        SplitRecord(
            "a-dr",
            "D_R",
            "a-dr",
            "images/a-dr.png",
            "masks/a-dr.png",
        ),
        SplitRecord(
            "z-dv",
            "D_V",
            "z-dv",
            "images/z-dv.png",
            "masks/z-dv.png",
        ),
        SplitRecord(
            "a-dv",
            "D_V",
            "a-dv",
            "images/a-dv.png",
            "masks/a-dv.png",
        ),
        SplitRecord("dt", "D_T", "dt", "images/dt.png", "masks/dt.png"),
    )
    manifest = SplitManifest(dataset="toy-cache", records=records)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.canonical_payload(), sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path, SplitManifest.load(manifest_path)


def _dataset(
    manifest_path: Path,
    manifest: SplitManifest,
    split: str,
) -> ManifestImageDataset:
    return ManifestImageDataset(
        manifest,
        split,  # type: ignore[arg-type]
        PreprocessConfig(
            height=8,
            width=8,
            color_mode="L",
            mean=(0.0,),
            std=(1.0,),
        ),
        manifest_path=manifest_path,
    )


def test_cache_manifest_split_writes_bound_per_sample_caches_and_strict_index(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = _dataset(manifest_path, manifest, "D_R")
    adapter = ToyFrozenBaseAdapter()
    output = tmp_path / "cache"

    returned = cache_manifest_split(adapter, dataset, "D_R", output)
    persisted = json.loads((output / "index.json").read_text(encoding="utf-8"))

    assert returned == persisted
    assert set(persisted) == {
        "schema_version",
        "base_cache_schema",
        "dataset",
        "split",
        "sample_count",
        "split_manifest_fingerprint",
        "split_manifest_file_sha256",
        "base_fingerprint",
        "preprocessing",
        "preprocessing_fingerprint",
        "records",
        "index_fingerprint",
    }
    assert persisted["schema_version"] == MANIFEST_BASE_CACHE_INDEX_SCHEMA
    assert persisted["split"] == "D_R"
    assert persisted["sample_count"] == 2
    assert persisted["split_manifest_fingerprint"] == manifest.fingerprint
    assert persisted["split_manifest_file_sha256"] == file_sha256(manifest_path)
    assert persisted["base_fingerprint"] == adapter.fingerprint
    assert persisted["preprocessing_fingerprint"] == stable_fingerprint(
        dataset.preprocess.fingerprint_payload()
    )
    assert [row["sample_id"] for row in persisted["records"]] == ["a-dr", "z-dr"]

    fingerprint_payload = dict(persisted)
    declared_index_fingerprint = fingerprint_payload.pop("index_fingerprint")
    assert declared_index_fingerprint == stable_fingerprint(fingerprint_payload)
    for row in persisted["records"]:
        assert set(row) == {
            "sample_id",
            "split",
            "image_path",
            "image_sha256",
            "cache_path",
            "cache_sha256",
            "probability_shape",
            "feature_shape",
        }
        cache_path = output / row["cache_path"]
        assert row["cache_sha256"] == file_sha256(cache_path)
        assert row["image_sha256"] == file_sha256(row["image_path"])
        cached = load_base_cache(
            cache_path,
            expected_fingerprint=adapter.fingerprint,
            expected_sample_id=row["sample_id"],
            expected_image_fingerprint=row["image_sha256"],
        )
        assert list(cached.probability.shape) == row["probability_shape"]
        assert list(cached.feature.shape) == row["feature_shape"]


def test_cache_manifest_split_rejects_adapter_preprocessing_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = ManifestImageDataset(
        manifest,
        "D_R",
        PreprocessConfig(
            height=8,
            width=8,
            color_mode="L",
            mean=(0.5,),
            std=(0.5,),
        ),
        manifest_path=manifest_path,
    )
    output = tmp_path / "cache"

    with pytest.raises(ValueError, match="toy base requires"):
        cache_manifest_split(
            ToyFrozenBaseAdapter(),
            dataset,
            "D_R",
            output,
        )
    assert not output.exists()


def test_cache_manifest_split_refuses_nonempty_output_and_forbidden_split(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    adapter = ToyFrozenBaseAdapter()
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("user-owned", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be empty"):
        cache_manifest_split(
            adapter,
            _dataset(manifest_path, manifest, "D_V"),
            "D_V",
            output,
        )
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user-owned"

    forbidden_output = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="only D_R or D_V"):
        cache_manifest_split(
            adapter,
            _dataset(manifest_path, manifest, "D_T"),
            "D_T",  # type: ignore[arg-type]
            forbidden_output,
        )
    assert not forbidden_output.exists()


def test_cache_manifest_split_rejects_unfingerprinted_transform(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = ManifestImageDataset(
        manifest,
        "D_V",
        PreprocessConfig(
            height=8,
            width=8,
            color_mode="L",
            mean=(0.0,),
            std=(1.0,),
        ),
        manifest_path=manifest_path,
        transform=lambda sample: sample,
    )
    output = tmp_path / "transform-cache"

    with pytest.raises(ValueError, match="forbids transforms"):
        cache_manifest_split(ToyFrozenBaseAdapter(), dataset, "D_V", output)
    assert not output.exists()


def test_cache_d_r_states_writes_complete_bound_state_index(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = _dataset(manifest_path, manifest, "D_R")
    adapter = ToyFrozenBaseAdapter()
    base_output = tmp_path / "base-output"
    cache_manifest_split(adapter, dataset, "D_R", base_output)
    state_output = tmp_path / "state-output"
    occupancy_config = OccupancyConfig(threshold=0.4)
    match_config = MatchConfig(max_distance=2.5)
    intervention_config = InterventionConfig()

    returned = cache_d_r_states(
        base_output / "index.json",
        dataset,
        state_output,
        expected_base_fingerprint=adapter.fingerprint,
        occupancy_config=occupancy_config,
        match_config=match_config,
        intervention_config=intervention_config,
    )
    persisted = json.loads(
        (state_output / "index.json").read_text(encoding="utf-8")
    )

    assert returned == persisted
    assert persisted["schema_version"] == MANIFEST_STATE_CACHE_INDEX_SCHEMA
    assert persisted["state_cache_schema"] == STATE_CACHE_SCHEMA
    assert persisted["split"] == "D_R"
    assert persisted["sample_count"] == 2
    assert persisted["base_fingerprint"] == adapter.fingerprint
    assert persisted["base_index"]["sha256"] == file_sha256(
        base_output / "index.json"
    )
    index_payload = dict(persisted)
    declared = index_payload.pop("index_fingerprint")
    assert declared == stable_fingerprint(index_payload)
    assert [row["sample_id"] for row in persisted["records"]] == ["a-dr", "z-dr"]
    for row in persisted["records"]:
        assert set(row["catalog_counts"]) == {
            "pred_components",
            "gt_components",
            "base_matches",
            "real_misses",
            "reachable_misses",
            "legal_pairs",
        }
        assert row["mask_sha256"] == file_sha256(row["mask_path"])
        assert row["base_cache_sha256"] == file_sha256(row["base_cache_path"])
        state_path = state_output / row["state_cache_path"]
        assert row["state_cache_sha256"] == file_sha256(state_path)
        state = load_state_cache(
            state_path,
            expected_fingerprint=persisted["state_fingerprint"],
            expected_sample_id=row["sample_id"],
        )
        assert state.real_miss_ids.numel() == row["catalog_counts"]["real_misses"]
        assert (
            state.reachable_miss_ids.numel()
            == row["catalog_counts"]["reachable_misses"]
        )
        assert state.legal_pairs.shape[0] == row["catalog_counts"]["legal_pairs"]

    bundle = load_d_r_cache_bundle(
        state_output / "index.json",
        dataset,
        expected_base_fingerprint=adapter.fingerprint,
    )
    assert bundle.split == "D_R"
    assert bundle.occupancy_config == occupancy_config
    assert bundle.match_config == match_config
    assert bundle.intervention_config == intervention_config
    assert bundle.manifest_path == manifest_path.resolve()
    assert bundle.base_index_path == (base_output / "index.json").resolve()
    assert bundle.state_index_path == (state_output / "index.json").resolve()
    assert bundle.base_fingerprint == adapter.fingerprint
    assert bundle.state_fingerprint == persisted["state_fingerprint"]
    assert bundle.state_index_fingerprint == persisted["index_fingerprint"]
    assert tuple(row.sample_id for row in bundle.rows) == ("a-dr", "z-dr")
    assert all(row.base_output.probability.shape[0] == 1 for row in bundle.rows)
    assert all(row.state.sample_id == row.sample_id for row in bundle.rows)
    bundle.verify_unchanged()


def test_cache_d_r_states_rejects_tampered_base_index_and_cache(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = _dataset(manifest_path, manifest, "D_R")
    adapter = ToyFrozenBaseAdapter()
    base_output = tmp_path / "base-output"
    base_index = cache_manifest_split(adapter, dataset, "D_R", base_output)
    index_path = base_output / "index.json"

    tampered_index = json.loads(index_path.read_text(encoding="utf-8"))
    tampered_index["sample_count"] = 999
    index_path.write_text(json.dumps(tampered_index), encoding="utf-8")
    rejected_output = tmp_path / "index-tamper-state"
    with pytest.raises(ValueError, match="fingerprint does not match"):
        cache_d_r_states(
            index_path,
            dataset,
            rejected_output,
            expected_base_fingerprint=adapter.fingerprint,
        )
    assert not rejected_output.exists()

    # Restore the exact index bytes, then mutate one referenced cache artifact.
    index_path.write_text(
        json.dumps(base_index, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    cache_path = base_output / base_index["records"][0]["cache_path"]
    cache_path.write_bytes(cache_path.read_bytes() + b"tampered")
    cache_rejected_output = tmp_path / "cache-tamper-state"
    with pytest.raises(ValueError, match="base cache file SHA256 mismatch"):
        cache_d_r_states(
            index_path,
            dataset,
            cache_rejected_output,
            expected_base_fingerprint=adapter.fingerprint,
        )
    assert not cache_rejected_output.exists()


@pytest.mark.parametrize("split", ("D_V", "D_T"))
def test_cache_d_r_states_rejects_non_d_r_split_before_output(
    tmp_path: Path,
    split: str,
) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    output = tmp_path / f"forbidden-{split}"
    with pytest.raises(ValueError, match="only exact D_R"):
        cache_d_r_states(
            tmp_path / "unused-index.json",
            _dataset(manifest_path, manifest, split),
            output,
            expected_base_fingerprint="a" * 64,
        )
    assert not output.exists()


def test_cache_d_r_states_rejects_symlink_output(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = _dataset(manifest_path, manifest, "D_R")
    adapter = ToyFrozenBaseAdapter()
    base_output = tmp_path / "base-output"
    cache_manifest_split(adapter, dataset, "D_R", base_output)
    occupied = tmp_path / "occupied-state"
    occupied.mkdir()
    marker = occupied / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        cache_d_r_states(
            base_output / "index.json",
            dataset,
            occupied,
            expected_base_fingerprint=adapter.fingerprint,
        )
    assert marker.read_text(encoding="utf-8") == "keep"

    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    output = tmp_path / "state-link"
    output.symlink_to(symlink_target, target_is_directory=True)

    with pytest.raises(ValueError, match="may not be a symlink"):
        cache_d_r_states(
            base_output / "index.json",
            dataset,
            output,
            expected_base_fingerprint=adapter.fingerprint,
        )
    assert not any(symlink_target.iterdir())


def test_load_d_r_cache_bundle_rejects_state_and_bound_base_tampering(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = _dataset(manifest_path, manifest, "D_R")
    adapter = ToyFrozenBaseAdapter()
    base_output = tmp_path / "base-output"
    base_index = cache_manifest_split(adapter, dataset, "D_R", base_output)
    state_output = tmp_path / "state-output"
    state_index = cache_d_r_states(
        base_output / "index.json",
        dataset,
        state_output,
        expected_base_fingerprint=adapter.fingerprint,
    )

    state_path = state_output / state_index["records"][0]["state_cache_path"]
    original_state = state_path.read_bytes()
    state_path.write_bytes(original_state + b"tampered")
    with pytest.raises(ValueError, match="state cache SHA256 binding mismatch"):
        load_d_r_cache_bundle(
            state_output / "index.json",
            dataset,
            expected_base_fingerprint=adapter.fingerprint,
        )
    state_path.write_bytes(original_state)

    base_index_path = base_output / "index.json"
    base_index_path.write_text(
        json.dumps(base_index, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    # Semantics are unchanged, but the state receipt binds the exact base-index
    # file bytes, so even reformatting is rejected.
    with pytest.raises(ValueError, match="bound base cache index SHA256 mismatch"):
        load_d_r_cache_bundle(
            state_output / "index.json",
            dataset,
            expected_base_fingerprint=adapter.fingerprint,
        )


def test_loaded_d_r_bundle_verify_unchanged_detects_file_and_memory_mutation(
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
    bundle = load_d_r_cache_bundle(
        state_output / "index.json",
        dataset,
        expected_base_fingerprint=adapter.fingerprint,
    )
    bundle.verify_unchanged()

    probability = bundle.rows[0].base_output.probability
    original_probability = probability.clone()
    probability[0, 0, 0, 0] += 0.001
    with pytest.raises(RuntimeError, match="tensors changed in memory"):
        bundle.verify_unchanged()
    probability.copy_(original_probability)
    bundle.verify_unchanged()

    image_path = bundle.rows[0].image_path
    image_path.write_bytes(image_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="image .* SHA256 changed"):
        bundle.verify_unchanged()


def test_load_d_v_cache_bundle_is_exact_sorted_and_fully_bound(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = _dataset(manifest_path, manifest, "D_V")
    adapter = ToyFrozenBaseAdapter()
    base_output = tmp_path / "d-v-base"
    base_index = cache_manifest_split(adapter, dataset, "D_V", base_output)

    bundle = load_d_v_cache_bundle(
        base_output / "index.json",
        dataset,
        expected_base_fingerprint=adapter.fingerprint,
    )

    assert bundle.split == "D_V"
    assert tuple(row.sample_id for row in bundle.rows) == ("a-dv", "z-dv")
    assert bundle.manifest_path == manifest_path.resolve()
    assert bundle.base_index_path == (base_output / "index.json").resolve()
    assert bundle.split_manifest_fingerprint == manifest.fingerprint
    assert bundle.base_index_fingerprint == base_index["index_fingerprint"]
    assert bundle.base_index_sha256 == file_sha256(base_output / "index.json")
    assert bundle.base_fingerprint == adapter.fingerprint
    assert len(bundle.d_v_image_fingerprint) == 64
    assert len(bundle.d_v_gt_fingerprint) == 64
    for row in bundle.rows:
        assert row.gt_mask.dtype == torch.bool
        assert row.gt_mask.shape == (1, 8, 8)
        assert row.base_output.probability.shape == (1, 1, 8, 8)
        assert row.image_sha256 == file_sha256(row.image_path)
        assert row.mask_sha256 == file_sha256(row.mask_path)
        assert row.base_cache_sha256 == file_sha256(row.base_cache_path)
    bundle.verify_unchanged()


@pytest.mark.parametrize("split", ("D_B", "D_R", "D_T"))
def test_load_d_v_cache_bundle_rejects_every_non_d_v_split(
    tmp_path: Path,
    split: str,
) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    with pytest.raises(ValueError, match="only exact D_V"):
        load_d_v_cache_bundle(
            tmp_path / "unused-index.json",
            _dataset(manifest_path, manifest, split),
            expected_base_fingerprint="a" * 64,
        )


def test_load_d_v_cache_bundle_rejects_index_symlink_and_cache_tamper(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = _dataset(manifest_path, manifest, "D_V")
    adapter = ToyFrozenBaseAdapter()
    base_output = tmp_path / "d-v-base"
    index = cache_manifest_split(adapter, dataset, "D_V", base_output)
    index_path = base_output / "index.json"
    index_link = tmp_path / "index-link.json"
    index_link.symlink_to(index_path)
    with pytest.raises(ValueError, match="may not be a symlink"):
        load_d_v_cache_bundle(
            index_link,
            dataset,
            expected_base_fingerprint=adapter.fingerprint,
        )

    cache_path = base_output / index["records"][0]["cache_path"]
    cache_path.write_bytes(cache_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="base cache file SHA256 mismatch"):
        load_d_v_cache_bundle(
            index_path,
            dataset,
            expected_base_fingerprint=adapter.fingerprint,
        )


def test_loaded_d_v_bundle_detects_memory_and_source_file_drift(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest_dataset_root(tmp_path)
    dataset = _dataset(manifest_path, manifest, "D_V")
    adapter = ToyFrozenBaseAdapter()
    base_output = tmp_path / "d-v-base"
    cache_manifest_split(adapter, dataset, "D_V", base_output)
    bundle = load_d_v_cache_bundle(
        base_output / "index.json",
        dataset,
        expected_base_fingerprint=adapter.fingerprint,
    )
    bundle.verify_unchanged()

    gt_mask = bundle.rows[0].gt_mask
    original_gt = gt_mask.clone()
    gt_mask[0, 0, 0] = ~gt_mask[0, 0, 0]
    with pytest.raises(RuntimeError, match="tensors changed in memory"):
        bundle.verify_unchanged()
    gt_mask.copy_(original_gt)
    bundle.verify_unchanged()

    mask_path = bundle.rows[0].mask_path
    mask_path.write_bytes(mask_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="mask .* SHA256 changed"):
        bundle.verify_unchanged()
