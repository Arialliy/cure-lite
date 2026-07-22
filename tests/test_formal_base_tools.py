from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from cure_lite.cache.schema import file_sha256
from cure_lite.provenance import deterministic_base_checkpoint_selection
from cure_lite.splits import SplitManifest
from tools import train_mshnet_base


def _write_image(path: Path, value: int, *, rgb: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb:
        array = np.full((16, 16, 3), value, dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(path)
    else:
        array = np.full((16, 16), value, dtype=np.uint8)
        Image.fromarray(array, mode="L").save(path)


def _write_manifest(tmp_path: Path) -> Path:
    samples: list[dict[str, object]] = []
    for index in range(4):
        image = tmp_path / "assets" / f"db-{index}-image.png"
        mask = tmp_path / "assets" / f"db-{index}-mask.png"
        _write_image(image, 32 + index * 32, rgb=True)
        _write_image(mask, 255 if index % 2 else 0, rgb=False)
        samples.append(
            {
                "sample_id": f"db-{index}",
                "split": "D_B",
                "group_id": f"db-group-{index}",
                "image": str(image),
                "mask": str(mask),
            }
        )
    for split in ("D_R", "D_V", "D_T"):
        samples.append(
            {
                "sample_id": split.lower(),
                "split": split,
                "group_id": f"{split.lower()}-group",
                "image": str(tmp_path / "unused" / f"{split.lower()}-image.png"),
                "mask": str(tmp_path / "unused" / f"{split.lower()}-mask.png"),
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cure-lite-splits-v1",
                "dataset": "formal-tool-toy",
                "created_before_training": True,
                "samples": samples,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_fake_mshnet_repository(tmp_path: Path) -> tuple[Path, str, str]:
    repository = tmp_path / "fake_mshnet"
    (repository / "model").mkdir(parents=True)
    (repository / "utils").mkdir()
    (repository / "model" / "MSHNet.py").write_text(
        """\
import torch.nn as nn


class MSHNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.output = nn.Conv2d(input_channels, 1, kernel_size=1)

    def forward(self, image, warm_flag):
        del warm_flag
        return [], self.output(image)
""",
        encoding="utf-8",
    )
    (repository / "model" / "loss.py").write_text(
        """\
import torch.nn as nn
import torch.nn.functional as functional


class SLSIoULoss(nn.Module):
    def forward(self, logits, target, warm_epoch, epoch):
        del warm_epoch, epoch
        return functional.binary_cross_entropy_with_logits(logits, target)
""",
        encoding="utf-8",
    )
    (repository / "utils" / "data.py").write_text(
        """\
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class IRSTD_Dataset(Dataset):
    def __init__(self, args, mode="train"):
        self.root = Path(args.dataset_dir)
        split_file = "trainval.txt" if mode == "train" else "test.txt"
        self.names = (self.root / split_file).read_text(encoding="utf-8").splitlines()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        sample_id = self.names[index]
        image = np.asarray(Image.open(self.root / "images" / f"{sample_id}.png").convert("RGB"), dtype=np.float32)
        mask = np.asarray(Image.open(self.root / "masks" / f"{sample_id}.png").convert("L"), dtype=np.float32)
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0) / 255.0
        return image_tensor, mask_tensor
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "CURE-Lite Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fake sources"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit, tree


def test_d_b_view_and_owned_runner_contract_use_exact_roles(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = SplitManifest.load(manifest_path)
    selection = deterministic_base_checkpoint_selection(
        manifest,
        select_fraction=0.5,
        seed=7,
    )
    view_root = tmp_path / "view"
    view = train_mshnet_base.build_d_b_dataset_view(
        manifest,
        manifest_path,
        selection,
        view_root,
    )
    train_mshnet_base._verify_dataset_view(view_root, view)

    assert view["roles"] == {"train": "D_B-fit", "validation": "D_B-select"}
    assert {row["split"] for row in view["records"]} == {"D_B"}
    assert {row["sample_id"] for row in view["records"]} == {
        "db-0",
        "db-1",
        "db-2",
        "db-3",
    }
    assert len((view_root / "trainval.txt").read_text().splitlines()) == 2
    assert len((view_root / "test.txt").read_text().splitlines()) == 2
    assert not (tmp_path / "unused").exists()

    runner = Path(train_mshnet_base.__file__).with_name(
        "run_pinned_mshnet_train.py"
    )
    source_hashes = {
        "model/MSHNet.py": "a" * 64,
        "model/loss.py": "b" * 64,
        "utils/data.py": "c" * 64,
    }
    contract = train_mshnet_base._runner_contract(runner, source_hashes)
    assert contract == {
        "contract": "cure-lite-pinned-mshnet-runner-v1",
        "entrypoint": "run_pinned_mshnet_train.py",
        "runner_sha256": file_sha256(runner),
        "upstream_sources_sha256": source_hashes,
    }
    args = argparse.Namespace(
        batch_size=1,
        epochs=1,
        lr=0.05,
        warm_epoch=5,
        base_size=16,
        crop_size=16,
        num_workers=0,
        device="cpu",
        multi_gpus=False,
        seed=3,
    )
    command = train_mshnet_base._native_command(
        args,
        python_executable=Path(sys.executable),
        runner=runner,
        repository=tmp_path / "repo",
        commit="commit",
        tree="tree",
        source_hashes=source_hashes,
        view_root=view_root,
        native_output=tmp_path / "native",
    )
    for option in ("--mshnet-repo", "--dataset-dir", "--output-dir"):
        assert command.count(option) == 1
    option_names = {item for item in command if item.startswith("--")}
    assert not any(
        fragment in option
        for option in option_names
        for fragment in ("resume", "weight", "checkpoint")
    )


def test_formal_launcher_runs_one_epoch_and_self_validates(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    repository, commit, _ = _write_fake_mshnet_repository(tmp_path)
    output = tmp_path / "formal_run"
    launcher = Path(train_mshnet_base.__file__)
    completed = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--manifest",
            str(manifest_path),
            "--mshnet-repo",
            str(repository),
            "--expected-commit",
            commit,
            "--output",
            str(output),
            "--python-executable",
            sys.executable,
            "--selection-fraction",
            "0.5",
            "--selection-seed",
            "7",
            "--batch-size",
            "1",
            "--epochs",
            "1",
            "--warm-epoch",
            "5",
            "--base-size",
            "16",
            "--crop-size",
            "16",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--seed",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

    final = json.loads((output / "final_receipt.json").read_text(encoding="utf-8"))
    preflight = json.loads(
        (output / "preflight_receipt.json").read_text(encoding="utf-8")
    )
    assert final["native_trainer_contract"] == preflight[
        "native_trainer_contract"
    ]
    assert final["native_trainer_contract"]["contract"] == (
        "cure-lite-pinned-mshnet-runner-v1"
    )
    assert final["training_metrics"]["best_epoch"] == 0
    checkpoint = output / final["checkpoint"]["path"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(state) == {"output.weight", "output.bias"}
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert not tuple(repository.rglob("__pycache__"))
