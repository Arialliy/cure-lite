"""Manifest-backed deterministic image loading for frozen-base extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .splits import SplitManifest, SplitName, SplitRecord


@dataclass(frozen=True)
class PreprocessConfig:
    """Evaluation-grid preprocessing that is included in adapter fingerprints."""

    height: int = 256
    width: int = 256
    color_mode: str = "RGB"
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    image_interpolation: str = "bilinear"
    mask_interpolation: str = "nearest"

    def __post_init__(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError("evaluation-grid dimensions must be positive")
        if self.color_mode not in {"L", "RGB"}:
            raise ValueError("color_mode must be 'L' or 'RGB'")
        channels = 1 if self.color_mode == "L" else 3
        if len(self.mean) != channels or len(self.std) != channels:
            raise ValueError("normalization statistics do not match color channels")
        if any(value <= 0 for value in self.std):
            raise ValueError("normalization standard deviations must be positive")
        if self.image_interpolation != "bilinear" or self.mask_interpolation != "nearest":
            raise ValueError("CURE-Lite MSHNet preprocessing fixes bilinear/nearest resize")

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "height": self.height,
            "width": self.width,
            "color_mode": self.color_mode,
            "mean": list(self.mean),
            "std": list(self.std),
            "image_interpolation": self.image_interpolation,
            "mask_interpolation": self.mask_interpolation,
            "range": "float32-[0,1]-then-normalize",
        }


@dataclass(frozen=True)
class LoadedSample:
    sample_id: str
    image: Tensor
    gt_mask: Tensor
    split: SplitName
    image_path: str
    mask_path: str


def _resolve_path(raw_path: str, manifest_path: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and manifest_path is not None:
        path = manifest_path.parent / path
    return path.resolve(strict=True)


def load_record(
    record: SplitRecord,
    config: PreprocessConfig,
    *,
    manifest_path: Path | None = None,
) -> LoadedSample:
    if record.mask is None:
        raise ValueError(f"sample {record.sample_id!r} has no GT mask")
    image_path = _resolve_path(record.image, manifest_path)
    mask_path = _resolve_path(record.mask, manifest_path)
    image = Image.open(image_path).convert(config.color_mode)
    mask = Image.open(mask_path).convert("L")
    size = (config.width, config.height)
    image = image.resize(size, Image.Resampling.BILINEAR)
    mask = mask.resize(size, Image.Resampling.NEAREST)

    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 2:
        image_array = image_array[..., None]
    image_tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1) / 255.0
    mean = torch.tensor(config.mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(config.std, dtype=torch.float32)[:, None, None]
    image_tensor = (image_tensor - mean) / std

    mask_array = np.asarray(mask, dtype=np.uint8)
    mask_tensor = torch.from_numpy((mask_array > 0).copy()).unsqueeze(0)
    return LoadedSample(
        sample_id=record.sample_id,
        image=image_tensor.contiguous(),
        gt_mask=mask_tensor.contiguous(),
        split=record.split,
        image_path=str(image_path),
        mask_path=str(mask_path),
    )


class ManifestImageDataset(Dataset[LoadedSample]):
    """A split-specific loader with no random transforms or hidden split fallback."""

    def __init__(
        self,
        manifest: SplitManifest,
        split: SplitName,
        preprocess: PreprocessConfig = PreprocessConfig(),
        *,
        manifest_path: str | Path | None = None,
        transform: Callable[[LoadedSample], LoadedSample] | None = None,
    ) -> None:
        self.manifest = manifest
        self.split = split
        self.preprocess = preprocess
        if manifest_path:
            self.manifest_path = Path(manifest_path).resolve()
        elif manifest.manifest_directory is not None:
            self.manifest_path = manifest.manifest_directory / "manifest.yaml"
        else:
            self.manifest_path = None
        self.transform = transform
        self.records = manifest.records_for(split)
        if not self.records:
            raise ValueError(f"manifest split {split} is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> LoadedSample:
        sample = load_record(
            self.records[index], self.preprocess, manifest_path=self.manifest_path
        )
        return self.transform(sample) if self.transform is not None else sample


def collate_loaded_samples(samples: list[LoadedSample]) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    shape = samples[0].image.shape
    gt_shape = samples[0].gt_mask.shape
    if any(sample.image.shape != shape or sample.gt_mask.shape != gt_shape for sample in samples):
        raise ValueError("all samples in a batch must share the evaluation grid")
    return {
        "sample_ids": tuple(sample.sample_id for sample in samples),
        "image_paths": tuple(sample.image_path for sample in samples),
        "images": torch.stack([sample.image for sample in samples]),
        "gt_masks": torch.stack([sample.gt_mask for sample in samples]),
        "splits": tuple(sample.split for sample in samples),
    }
