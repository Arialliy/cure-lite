"""Strict catalog for the three IRSTD benchmarks used by CURE-Lite.

This module describes dataset identity and official membership only.  It does
not import an upstream repository, choose a model preprocessing recipe, or
turn the official test set into a development split.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Literal


BenchmarkStorageId = Literal["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"]
OfficialSplit = Literal["train", "test"]
SUPPORTED_BENCHMARKS: tuple[BenchmarkStorageId, ...] = (
    "NUAA-SIRST",
    "NUDT-SIRST",
    "IRSTD-1K",
)
RASTER_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Stable benchmark identity used consistently in paths and artifacts."""

    storage_id: BenchmarkStorageId

    @property
    def train_index_relative_path(self) -> Path:
        return Path("img_idx") / f"train_{self.storage_id}.txt"

    @property
    def test_index_relative_path(self) -> Path:
        return Path("img_idx") / f"test_{self.storage_id}.txt"


BENCHMARK_SPECS: dict[BenchmarkStorageId, BenchmarkSpec] = {
    "NUAA-SIRST": BenchmarkSpec("NUAA-SIRST"),
    "NUDT-SIRST": BenchmarkSpec("NUDT-SIRST"),
    "IRSTD-1K": BenchmarkSpec("IRSTD-1K"),
}


@dataclass(frozen=True, slots=True)
class BenchmarkAsset:
    sample_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkCatalog:
    """Index-driven official membership with exactly resolved image/mask pairs."""

    spec: BenchmarkSpec
    root: Path
    train_index_path: Path
    test_index_path: Path
    train_assets: tuple[BenchmarkAsset, ...]
    test_assets: tuple[BenchmarkAsset, ...]

    @property
    def train_sample_ids(self) -> tuple[str, ...]:
        return tuple(asset.sample_id for asset in self.train_assets)

    @property
    def test_sample_ids(self) -> tuple[str, ...]:
        return tuple(asset.sample_id for asset in self.test_assets)


def benchmark_spec(storage_id: str) -> BenchmarkSpec:
    """Resolve only an exact storage ID; display aliases are not path aliases."""

    try:
        return BENCHMARK_SPECS[storage_id]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(
            f"unsupported IRSTD benchmark {storage_id!r}; "
            f"expected one of {SUPPORTED_BENCHMARKS}"
        ) from error


def _absolute_without_symlink(path: str | Path, *, name: str) -> Path:
    candidate = Path(path).expanduser().absolute()
    chain = tuple(reversed(candidate.parents)) + (candidate,)
    for component in chain:
        if component.is_symlink():
            raise ValueError(f"{name} may not traverse a symlink: {component}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {name}: {candidate}") from error
    if resolved != candidate:
        raise ValueError(f"{name} must be an already-normalized absolute path")
    return resolved


def _regular_file(path: Path, *, name: str) -> Path:
    resolved = _absolute_without_symlink(path, name=name)
    mode = os.stat(resolved, follow_symlinks=False).st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _read_official_ids(path: Path, *, role: OfficialSplit) -> tuple[str, ...]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"official {role} index must be UTF-8 text") from error
    values: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        value = raw_line.strip()
        if not value:
            raise ValueError(
                f"official {role} index line {line_number} may not be empty"
            )
        candidate = Path(value)
        if (
            candidate.name != value
            or candidate.stem != value
            or value in {".", ".."}
        ):
            raise ValueError(
                f"official {role} index line {line_number} must contain one "
                "extension-free sample ID"
            )
        values.append(value)
    if not values:
        raise ValueError(f"official {role} index is empty")
    if len(values) != len(set(values)):
        raise ValueError(f"official {role} index contains duplicate sample IDs")
    return tuple(values)


def _index_raster_assets(directory: Path, *, kind: str) -> dict[str, Path]:
    resolved_directory = _absolute_without_symlink(directory, name=f"{kind} directory")
    if not resolved_directory.is_dir():
        raise ValueError(f"{kind} directory must be a directory")
    result: dict[str, Path] = {}
    for path in sorted(resolved_directory.iterdir(), key=lambda item: item.name):
        if path.suffix.lower() not in RASTER_EXTENSIONS:
            continue
        resolved = _regular_file(path, name=f"{kind} asset")
        if path.stem in result:
            raise ValueError(f"ambiguous {kind} assets for sample {path.stem!r}")
        result[path.stem] = resolved
    if not result:
        raise ValueError(f"{kind} directory contains no supported raster assets")
    return result


def catalog_benchmark_dataset(
    dataset_root: str | Path,
    *,
    dataset: str | None = None,
) -> BenchmarkCatalog:
    """Validate one benchmark without decoding or hashing image/mask content."""

    root = _absolute_without_symlink(dataset_root, name="dataset root")
    if not root.is_dir():
        raise ValueError("dataset root must be a directory")
    storage_id = root.name if dataset is None else dataset
    spec = benchmark_spec(storage_id)
    if root.name != spec.storage_id:
        raise ValueError(
            "dataset root leaf name does not match the declared benchmark storage ID"
        )

    train_index = _regular_file(
        root / spec.train_index_relative_path,
        name="official train index",
    )
    test_index = _regular_file(
        root / spec.test_index_relative_path,
        name="official test index",
    )
    train_ids = _read_official_ids(train_index, role="train")
    test_ids = _read_official_ids(test_index, role="test")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"official train/test sample IDs overlap: {overlap[:10]}")

    images = _index_raster_assets(root / "images", kind="image")
    masks = _index_raster_assets(root / "masks", kind="mask")
    official_ids = set(train_ids) | set(test_ids)
    for kind, assets in (("image", images), ("mask", masks)):
        actual_ids = set(assets)
        if actual_ids != official_ids:
            missing = sorted(official_ids - actual_ids)
            unlisted = sorted(actual_ids - official_ids)
            raise ValueError(
                f"official indexes/{kind} assets differ; "
                f"missing={missing[:10]}, unlisted={unlisted[:10]}"
            )

    def assets_for(sample_ids: tuple[str, ...]) -> tuple[BenchmarkAsset, ...]:
        return tuple(
            BenchmarkAsset(
                sample_id=sample_id,
                image_path=images[sample_id],
                mask_path=masks[sample_id],
            )
            for sample_id in sample_ids
        )

    return BenchmarkCatalog(
        spec=spec,
        root=root,
        train_index_path=train_index,
        test_index_path=test_index,
        train_assets=assets_for(train_ids),
        test_assets=assets_for(test_ids),
    )


__all__ = [
    "BENCHMARK_SPECS",
    "RASTER_EXTENSIONS",
    "SUPPORTED_BENCHMARKS",
    "BenchmarkAsset",
    "BenchmarkCatalog",
    "BenchmarkSpec",
    "benchmark_spec",
    "catalog_benchmark_dataset",
]
