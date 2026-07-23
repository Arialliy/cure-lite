from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from cure_lite.data import PreprocessConfig
from cure_lite.experiment.cache_pipeline import load_base_cache_pair_contract
from cure_lite.reference_base import (
    ReferenceBaseAdapter,
    ReferenceBaseModelConfig,
    ReferenceBaseNetwork,
    ReferenceBaseTrainingConfig,
    build_d_b_partition,
    load_reference_base_run,
    load_verified_reference_base_run_identity,
    train_reference_base,
)
from cure_lite.splits import SplitManifest, SplitRecord
from tools import cache_reference_base as cache_cli
from tools import train_reference_base as train_cli


_D_T_ID = "unused-d-t"
_D_T_IMAGE = "missing/d-t-image.png"
_D_T_MASK = "missing/d-t-mask.png"


def _write_sample(root: Path, sample_id: str, offset: int) -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    image += np.arange(32, dtype=np.uint8)[None, :]
    mask = np.zeros((32, 32), dtype=np.uint8)
    top = 5 + offset
    left = 7 + offset
    image[top : top + 3, left : left + 3] = 230
    mask[top : top + 3, left : left + 3] = 255
    Image.fromarray(image, mode="L").save(root / "images" / f"{sample_id}.png")
    Image.fromarray(mask, mode="L").save(root / "masks" / f"{sample_id}.png")


def _manifest(tmp_path: Path) -> tuple[SplitManifest, Path]:
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "masks").mkdir()
    records: list[SplitRecord] = []
    for index in range(4):
        sample_id = f"db-{index}"
        _write_sample(tmp_path, sample_id, index)
        records.append(
            SplitRecord(
                sample_id,
                "D_B",
                f"g-db-{index}",
                f"images/{sample_id}.png",
                f"masks/{sample_id}.png",
            )
        )
    for split, sample_id in (("D_R", "dr"), ("D_V", "dv")):
        _write_sample(tmp_path, sample_id, 2)
        records.append(
            SplitRecord(
                sample_id,
                split,  # type: ignore[arg-type]
                f"g-{sample_id}",
                f"images/{sample_id}.png",
                f"masks/{sample_id}.png",
            )
        )
    records.append(
        SplitRecord(
            _D_T_ID,
            "D_T",
            "g-unused-d-t",
            _D_T_IMAGE,
            _D_T_MASK,
        )
    )
    manifest = SplitManifest(dataset="tiny-reference", records=tuple(records))
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(manifest.canonical_payload(), sort_keys=True),
        encoding="utf-8",
    )
    return SplitManifest.load(path), path


def _tiny_config() -> ReferenceBaseTrainingConfig:
    return ReferenceBaseTrainingConfig(
        dataset="tiny-reference",
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        positive_weight=10.0,
        training_seed=7,
        selection_seed=11,
        device="cpu",
        model=ReferenceBaseModelConfig(
            stem_channels=8,
            half_channels=8,
            feature_channels=8,
            eighth_channels=8,
            bottleneck_channels=8,
            norm_groups=8,
        ),
        preprocess=PreprocessConfig(
            height=32,
            width=32,
            color_mode="L",
            mean=(0.0,),
            std=(1.0,),
        ),
    )


def test_protocol_reference_base_is_800_epochs_and_not_mshnet() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "protocols/IRSTD-1K/stage_a_seed42/base_training_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = ReferenceBaseTrainingConfig.from_mapping(payload)
    assert config.epochs == 800
    assert config.model.feature_channels == 64
    assert "mshnet" not in path.read_text(encoding="utf-8").lower()


def test_reference_network_and_adapter_follow_generic_base_contract() -> None:
    config = _tiny_config()
    model = ReferenceBaseNetwork(config.model)
    output = model.forward_with_feature(torch.zeros(2, 1, 32, 32))
    assert output.logits.shape == (2, 1, 32, 32)
    assert output.feature.shape == (2, 8, 8, 8)
    adapter = ReferenceBaseAdapter(model, config.preprocess, "1" * 64)
    frozen = adapter(torch.zeros(2, 1, 32, 32))
    assert frozen.probability.shape == (2, 1, 32, 32)
    assert frozen.feature.shape == (2, 8, 8, 8)
    assert not any(parameter.requires_grad for parameter in adapter.parameters())


def test_d_b_partition_is_group_disjoint_and_exact(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    partition = build_d_b_partition(
        manifest,
        selection_fraction=0.5,
        seed=19,
    )
    assert set(partition.fit_sample_ids).isdisjoint(partition.select_sample_ids)
    assert set(partition.fit_group_ids).isdisjoint(partition.select_group_ids)
    assert set(partition.fit_sample_ids) | set(partition.select_sample_ids) == {
        record.sample_id for record in manifest.records_for("D_B")
    }
    assert _D_T_ID not in partition.fit_sample_ids + partition.select_sample_ids


def test_one_epoch_reference_run_and_cache_never_read_d_t(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path / "data")
    run_root = tmp_path / "reference-run"
    completed = train_reference_base(
        manifest,
        manifest_path,
        _tiny_config(),
        run_root,
    )
    assert completed.best_epoch == 1
    assert (run_root / "COMPLETE.json").is_file()
    assert len((run_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    loaded = load_reference_base_run(run_root, device="cpu")
    assert loaded.base_fingerprint == completed.base_fingerprint
    identity = load_verified_reference_base_run_identity(run_root)
    identity.verify_unchanged()
    assert identity.identity.base_fingerprint == loaded.base_fingerprint
    assert identity.identity.checkpoint_sha256 == loaded.checkpoint_sha256
    assert identity.identity.producer_schema == "cure-lite-reference-base-run-v1"

    cache_root = tmp_path / "base-caches"
    cache_cli.main(
        [
            "--manifest",
            str(manifest_path),
            "--reference-base-run",
            str(run_root),
            "--output",
            str(cache_root),
            "--device",
            "cpu",
        ]
    )
    pair = load_base_cache_pair_contract(
        cache_root / "D_R/index.json",
        cache_root / "D_V/index.json",
    )
    assert pair.base_fingerprint == loaded.base_fingerprint
    assert pair.base_state_fingerprint == identity.identity.base_state_fingerprint
    assert pair.feature_channels == 8
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in cache_root.rglob("*.json")
    )
    assert _D_T_ID not in persisted
    assert _D_T_IMAGE not in persisted
    assert _D_T_MASK not in persisted


def test_reference_clis_have_no_external_model_or_d_t_argument() -> None:
    for module in (train_cli, cache_cli):
        options = {
            option
            for action in module.build_parser()._actions
            for option in action.option_strings
        }
        assert not any("d-t" in option or "d_t" in option for option in options)
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        assert "mshnet" not in source
