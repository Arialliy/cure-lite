from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from cure_lite.calibration import FalseAlarmBudget
from cure_lite.cache.schema import file_sha256
from cure_lite.config import DecoderConfig
from cure_lite.data import ManifestImageDataset, PreprocessConfig
from cure_lite.experiment.cache_pipeline import cache_manifest_split
from cure_lite.experiment.formal_training import PairedGate2TrainingConfig
from cure_lite.experiment.stage_a_runner import (
    StageARunConfig,
    load_stage_a_run,
    run_stage_a,
    run_stage_a_from_base_caches,
)
from cure_lite.experiment.training_pipeline import TrainingSupportRequirements
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.toy import ToyFrozenBaseAdapter


_UNUSED_D_T_ID = "unused-dt-not-accessed"
_UNUSED_D_T_IMAGE = "unused-assets/dt-image.png"
_UNUSED_D_T_MASK = "unused-assets/dt-mask.png"


def _write_scene(
    root: Path,
    sample_id: str,
    target_values: tuple[int, int],
) -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    for value, (top, left) in zip(
        target_values,
        ((6, 6), (22, 22)),
        strict=True,
    ):
        image[top : top + 3, left : left + 3] = value
        mask[top : top + 3, left : left + 3] = 255
    Image.fromarray(image, mode="L").save(root / "images" / f"{sample_id}.png")
    Image.fromarray(mask, mode="L").save(root / "masks" / f"{sample_id}.png")


def _development_datasets(
    tmp_path: Path,
) -> tuple[ManifestImageDataset, ManifestImageDataset]:
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "masks").mkdir()
    for sample_id, values in {
        "db": (255, 255),
        "dr-covered": (255, 255),
        # 89/255 maps to a base probability below the frozen tau_o=0.5.
        "dr-miss": (89, 255),
        "dv": (89, 255),
    }.items():
        _write_scene(tmp_path, sample_id, values)

    manifest = SplitManifest(
        dataset="formal-stage-a-toy",
        records=(
            SplitRecord("db", "D_B", "g-db", "images/db.png", "masks/db.png"),
            SplitRecord(
                "dr-miss",
                "D_R",
                "g-dr-miss",
                "images/dr-miss.png",
                "masks/dr-miss.png",
            ),
            SplitRecord(
                "dr-covered",
                "D_R",
                "g-dr-covered",
                "images/dr-covered.png",
                "masks/dr-covered.png",
            ),
            SplitRecord("dv", "D_V", "g-dv", "images/dv.png", "masks/dv.png"),
            # These paths deliberately do not exist.  Stage A must never touch
            # official-test assets even though membership is fixed in manifest.
            SplitRecord(
                _UNUSED_D_T_ID,
                "D_T",
                "g-dt-unused",
                _UNUSED_D_T_IMAGE,
                _UNUSED_D_T_MASK,
            ),
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.canonical_payload(), sort_keys=True),
        encoding="utf-8",
    )
    loaded = SplitManifest.load(manifest_path)
    preprocess = PreprocessConfig(
        height=32,
        width=32,
        color_mode="L",
        mean=(0.0,),
        std=(1.0,),
    )
    return (
        ManifestImageDataset(
            loaded,
            "D_R",
            preprocess,
            manifest_path=manifest_path,
        ),
        ManifestImageDataset(
            loaded,
            "D_V",
            preprocess,
            manifest_path=manifest_path,
        ),
    )


def _stage_a_config(*, feature_channels: int = 3) -> StageARunConfig:
    return StageARunConfig(
        training=PairedGate2TrainingConfig(
            decoder_config=DecoderConfig(feature_channels=feature_channels),
            optimizer="sgd",
            learning_rate=1e-3,
            epochs=1,
            steps_per_epoch=1,
            global_seed=17,
        ),
        anchor_thresholds=(0.5,),
        base_thresholds=(0.3, 0.5),
        residual_thresholds=(0.5, 0.9),
        budget=FalseAlarmBudget(
            pixel_fa_budget=1.0,
            component_fa_per_mp_budget=float("inf"),
            raw_background_fa_budget=1.0,
            minimum_retention=0.0,
        ),
        device="cpu",
    )


def _all_json_text(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.json"))
    )


def test_stage_a_single_entry_is_replayable_and_never_reads_d_t(
    tmp_path: Path,
) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    output = tmp_path / "stage-a-run"

    completed = run_stage_a(
        adapter,
        d_r_dataset,
        d_v_dataset,
        _stage_a_config(),
        output,
    )

    assert (output / "COMPLETE.json").is_file()
    assert not (output / ".incomplete").exists()
    for receipt_name in ("config", "anchor", "support", "calibration", "results"):
        assert (output / "receipts" / f"{receipt_name}.json").is_file()
    for variant in ("factual_only", "uniform_legal"):
        assert (output / "decoders" / variant / "receipt.json").is_file()
    assert {path.name for path in (output / "decoders").iterdir()} == {
        "factual_only",
        "uniform_legal",
    }

    completed.verify_unchanged()
    reloaded = load_stage_a_run(
        output,
        d_r_dataset,
        d_v_dataset,
        expected_base_fingerprint=adapter.fingerprint,
    )
    reloaded.verify_unchanged()
    assert reloaded.results == completed.results
    assert reloaded.calibration == completed.calibration
    assert reloaded.support_summary == completed.support_summary
    assert completed.support_summary.factual_miss_images >= 1
    assert completed.support_summary.factual_no_miss_images >= 1
    assert completed.support_summary.synthetic_images >= 1

    results_payload = json.loads(
        (output / "receipts" / "results.json").read_text(encoding="utf-8")
    )
    calibration_payload = json.loads(
        (output / "receipts" / "calibration.json").read_text(encoding="utf-8")
    )
    complete_payload = json.loads(
        (output / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert set(results_payload["methods"]) == {"A", "Base@B", "F", "U"}
    assert set(calibration_payload["methods"]) == {"A", "Base@B", "F", "U"}
    assert complete_payload["method_order"] == ["A", "Base@B", "F", "U"]

    persisted_json = _all_json_text(output)
    assert _UNUSED_D_T_ID not in persisted_json
    assert _UNUSED_D_T_IMAGE not in persisted_json
    assert _UNUSED_D_T_MASK not in persisted_json

    complete_bytes = (output / "COMPLETE.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_stage_a(
            adapter,
            d_r_dataset,
            d_v_dataset,
            _stage_a_config(),
            output,
        )
    assert (output / "COMPLETE.json").read_bytes() == complete_bytes


def test_stage_a_rejects_insufficient_real_support_before_decoder_training(
    tmp_path: Path,
) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    output = tmp_path / "insufficient-support"
    config = replace(
        _stage_a_config(),
        support_requirements=TrainingSupportRequirements(
            minimum_factual_miss_images=100,
        ),
    )

    with pytest.raises(RuntimeError, match="factual_miss_images"):
        run_stage_a(
            adapter,
            d_r_dataset,
            d_v_dataset,
            config,
            output,
        )

    assert (output / "receipts" / "anchor.json").is_file()
    assert not (output / "decoders").exists()
    assert not (output / "COMPLETE.json").exists()


def test_stage_a_failure_never_publishes_complete_marker(tmp_path: Path) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    output = tmp_path / "failed-stage-a-run"

    with pytest.raises((RuntimeError, ValueError)):
        run_stage_a(
            adapter,
            d_r_dataset,
            d_v_dataset,
            _stage_a_config(feature_channels=4),
            output,
        )

    assert not (output / "COMPLETE.json").exists()
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        load_stage_a_run(
            output,
            d_r_dataset,
            d_v_dataset,
            expected_base_fingerprint=adapter.fingerprint,
        )


def test_stage_a_from_generic_base_caches_is_self_contained(
    tmp_path: Path,
) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    source = tmp_path / "generic-base-cache"
    d_r_source = source / "d_r"
    d_v_source = source / "d_v"
    cache_manifest_split(adapter, d_r_dataset, "D_R", d_r_source)
    cache_manifest_split(adapter, d_v_dataset, "D_V", d_v_source)
    output = tmp_path / "stage-a-from-cache"

    completed = run_stage_a_from_base_caches(
        d_r_source / "index.json",
        d_v_source / "index.json",
        d_r_dataset,
        d_v_dataset,
        _stage_a_config(),
        output,
    )

    assert (output / "COMPLETE.json").is_file()
    assert file_sha256(output / "d_r" / "base_cache" / "index.json") == (
        file_sha256(d_r_source / "index.json")
    )
    assert file_sha256(output / "d_v" / "base_cache" / "index.json") == (
        file_sha256(d_v_source / "index.json")
    )
    completed.verify_unchanged()
    assert _UNUSED_D_T_ID not in _all_json_text(output)

    for path in source.rglob("*"):
        if path.is_file():
            path.unlink()
    for path in sorted(source.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    source.rmdir()
    completed.verify_unchanged()
