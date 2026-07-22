"""Deterministic construction of experiment cache records.

This module contains orchestration that is shared by detector adapters while
keeping the CURE-Lite mechanism itself independent of any particular IRSTD
backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal, Mapping

import torch
from torch import Tensor

from ..cache.base_cache import load_base_cache, save_base_cache
from ..cache.schema import (
    BASE_CACHE_SCHEMA,
    STATE_CACHE_SCHEMA,
    build_state_fingerprint,
    file_sha256,
    stable_fingerprint,
)
from ..cache.state_cache import StateCacheRecord, load_state_cache, save_state_cache
from ..config import (
    InterventionConfig,
    MatchConfig,
    OccupancyConfig,
    config_to_dict,
)
from ..data import LoadedSample, ManifestImageDataset, PreprocessConfig
from ..frozen_base import FrozenBaseAdapter
from ..instances import instances_from_binary_mask
from ..intervention import enumerate_legal_deletions
from ..matching import match_components
from ..occupancy import build_occupancy
from ..splits import SplitManifest
from ..supervision import full_gt_recoverable
from ..types import FrozenBaseOutput


MANIFEST_BASE_CACHE_INDEX_SCHEMA = "cure-lite-manifest-base-cache-index-v1"
MANIFEST_STATE_CACHE_INDEX_SCHEMA = "cure-lite-manifest-state-cache-index-v1"

_BASE_INDEX_KEYS = {
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
_BASE_INDEX_RECORD_KEYS = {
    "sample_id",
    "split",
    "image_path",
    "image_sha256",
    "cache_path",
    "cache_sha256",
    "probability_shape",
    "feature_shape",
}
_STATE_INDEX_KEYS = {
    "schema_version",
    "state_cache_schema",
    "dataset",
    "split",
    "sample_count",
    "split_manifest_fingerprint",
    "split_manifest_file_sha256",
    "base_fingerprint",
    "base_index",
    "preprocessing",
    "preprocessing_fingerprint",
    "occupancy_config",
    "matching_config",
    "intervention_config",
    "gt_fingerprint",
    "state_fingerprint",
    "records",
    "index_fingerprint",
}
_STATE_INDEX_RECORD_KEYS = {
    "sample_id",
    "split",
    "image_path",
    "image_sha256",
    "mask_path",
    "mask_sha256",
    "base_cache_path",
    "base_cache_sha256",
    "state_cache_path",
    "state_cache_sha256",
    "catalog_counts",
}
_CATALOG_COUNT_KEYS = {
    "pred_components",
    "gt_components",
    "base_matches",
    "real_misses",
    "reachable_misses",
    "legal_pairs",
}


@dataclass(frozen=True)
class BaseCachePairContract:
    """Model-independent identity shared by exact D_R and D_V base caches."""

    dataset: str
    split_manifest_fingerprint: str
    split_manifest_file_sha256: str
    base_fingerprint: str
    preprocessing: PreprocessConfig
    preprocessing_fingerprint: str
    feature_channels: int
    feature_shape: tuple[int, int, int, int]
    d_r_index_path: Path
    d_r_index_sha256: str
    d_r_index_fingerprint: str
    d_r_sample_ids: tuple[str, ...]
    d_v_index_path: Path
    d_v_index_sha256: str
    d_v_index_fingerprint: str
    d_v_sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ParsedBaseIndex:
    path: Path
    sha256: str
    payload: dict[str, Any]
    preprocessing: PreprocessConfig
    feature_shape: tuple[int, int, int, int]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LoadedBundleSeal:
    kind: Literal["D_R", "D_V"]
    bound_objects: tuple[object, ...]
    bound_values: tuple[object, ...]


@dataclass(frozen=True)
class LoadedDRCacheRow:
    """One verified formal D_R row ready for training orchestration."""

    sample_id: str
    base_output: FrozenBaseOutput
    state: StateCacheRecord
    image_path: Path
    mask_path: Path
    base_cache_path: Path
    state_cache_path: Path
    image_sha256: str
    mask_sha256: str
    base_cache_sha256: str
    state_cache_sha256: str
    content_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("loaded D_R row sample_id must be non-empty")
        if not isinstance(self.base_output, FrozenBaseOutput):
            raise TypeError("loaded D_R row base_output has invalid type")
        if not isinstance(self.state, StateCacheRecord):
            raise TypeError("loaded D_R row state has invalid type")
        if self.state.sample_id != self.sample_id:
            raise ValueError("loaded D_R row state/sample identity mismatch")
        if self.base_output.probability.shape[0] != 1:
            raise ValueError("loaded D_R row base output must contain one sample")
        for name in (
            "image_path",
            "mask_path",
            "base_cache_path",
            "state_cache_path",
        ):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"loaded D_R row {name} must be an absolute Path")
        for name in (
            "image_sha256",
            "mask_sha256",
            "base_cache_sha256",
            "state_cache_sha256",
            "content_fingerprint",
        ):
            _canonical_sha256(getattr(self, name), name=name)
        expected = _loaded_row_content_fingerprint(
            self.sample_id,
            self.base_output,
            self.state,
        )
        if self.content_fingerprint != expected:
            raise ValueError("loaded D_R row memory fingerprint mismatch")


@dataclass(frozen=True)
class LoadedDRCacheBundle:
    """Fully verified D_R base/state tensors plus their experiment identities."""

    split: Literal["D_R"]
    rows: tuple[LoadedDRCacheRow, ...]
    occupancy_config: OccupancyConfig
    match_config: MatchConfig
    intervention_config: InterventionConfig
    manifest_path: Path
    base_index_path: Path
    state_index_path: Path
    split_manifest_fingerprint: str
    split_manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    state_fingerprint: str
    gt_fingerprint: str
    base_index_fingerprint: str
    base_index_sha256: str
    state_index_fingerprint: str
    state_index_sha256: str
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _LoadedBundleSeal or seal.kind != "D_R":
            raise TypeError("LoadedDRCacheBundle must be created by its strict loader")
        expected_objects = (
            self.rows,
            self.occupancy_config,
            self.match_config,
            self.intervention_config,
        )
        if len(seal.bound_objects) != len(expected_objects) or any(
            sealed is not current
            for sealed, current in zip(seal.bound_objects, expected_objects)
        ):
            raise TypeError("loaded D_R bundle source objects were replaced")
        if seal.bound_values != (
            self.split,
            self.manifest_path,
            self.base_index_path,
            self.state_index_path,
            self.split_manifest_fingerprint,
            self.split_manifest_file_sha256,
            self.preprocessing_fingerprint,
            self.base_fingerprint,
            self.state_fingerprint,
            self.gt_fingerprint,
            self.base_index_fingerprint,
            self.base_index_sha256,
            self.state_index_fingerprint,
            self.state_index_sha256,
        ):
            raise TypeError("loaded D_R bundle bound fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        if self.split != "D_R":
            raise ValueError("loaded cache bundle split must be exactly D_R")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("loaded D_R cache bundle rows must be a nonempty tuple")
        if any(not isinstance(row, LoadedDRCacheRow) for row in self.rows):
            raise TypeError("loaded D_R cache bundle contains an invalid row")
        sample_ids = tuple(row.sample_id for row in self.rows)
        if sample_ids != tuple(sorted(set(sample_ids))):
            raise ValueError("loaded D_R cache rows must have sorted unique sample IDs")
        if not isinstance(self.occupancy_config, OccupancyConfig):
            raise TypeError("bundle occupancy_config has invalid type")
        if not isinstance(self.match_config, MatchConfig):
            raise TypeError("bundle match_config has invalid type")
        if not isinstance(self.intervention_config, InterventionConfig):
            raise TypeError("bundle intervention_config has invalid type")
        for name in ("manifest_path", "base_index_path", "state_index_path"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"bundle {name} must be an absolute Path")
        for name in (
            "split_manifest_fingerprint",
            "split_manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "state_fingerprint",
            "gt_fingerprint",
            "base_index_fingerprint",
            "base_index_sha256",
            "state_index_fingerprint",
            "state_index_sha256",
        ):
            _canonical_sha256(getattr(self, name), name=name)

    def verify_unchanged(self) -> None:
        """Fail if any bound file or in-memory tensor changed after loading."""

        self._verify_source_seal()
        file_bindings = [
            (
                self.manifest_path,
                self.split_manifest_file_sha256,
                "split manifest",
            ),
            (self.base_index_path, self.base_index_sha256, "base cache index"),
            (self.state_index_path, self.state_index_sha256, "state cache index"),
        ]
        for row in self.rows:
            file_bindings.extend(
                (
                    (row.image_path, row.image_sha256, f"image {row.sample_id!r}"),
                    (row.mask_path, row.mask_sha256, f"mask {row.sample_id!r}"),
                    (
                        row.base_cache_path,
                        row.base_cache_sha256,
                        f"base cache {row.sample_id!r}",
                    ),
                    (
                        row.state_cache_path,
                        row.state_cache_sha256,
                        f"state cache {row.sample_id!r}",
                    ),
                )
            )
        for path, expected_sha256, label in file_bindings:
            if path.is_symlink():
                raise RuntimeError(f"bound {label} became a symlink")
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise RuntimeError(f"bound {label} is missing") from error
            if resolved != path or not resolved.is_file():
                raise RuntimeError(f"bound {label} changed path or file type")
            if file_sha256(resolved) != expected_sha256:
                raise RuntimeError(f"bound {label} SHA256 changed")
        for row in self.rows:
            current = _loaded_row_content_fingerprint(
                row.sample_id,
                row.base_output,
                row.state,
            )
            if current != row.content_fingerprint:
                raise RuntimeError(
                    f"loaded D_R row {row.sample_id!r} tensors changed in memory"
                )


@dataclass(frozen=True)
class LoadedDVCacheRow:
    """One verified D_V image, GT mask, and frozen-base output."""

    sample_id: str
    base_output: FrozenBaseOutput
    gt_mask: Tensor
    image_path: Path
    mask_path: Path
    base_cache_path: Path
    image_sha256: str
    mask_sha256: str
    base_cache_sha256: str
    content_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("loaded D_V row sample_id must be non-empty")
        if not isinstance(self.base_output, FrozenBaseOutput):
            raise TypeError("loaded D_V row base_output has invalid type")
        if self.base_output.probability.shape[0] != 1:
            raise ValueError("loaded D_V row base output must contain one sample")
        if (
            not isinstance(self.gt_mask, Tensor)
            or self.gt_mask.dtype != torch.bool
            or self.gt_mask.device.type != "cpu"
            or self.gt_mask.ndim != 3
            or self.gt_mask.shape[0] != 1
        ):
            raise ValueError("loaded D_V row gt_mask must be CPU bool [1,H,W]")
        if self.base_output.probability.shape[-2:] != self.gt_mask.shape[-2:]:
            raise ValueError("loaded D_V probability and GT grids differ")
        for name in ("image_path", "mask_path", "base_cache_path"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"loaded D_V row {name} must be an absolute Path")
        for name in (
            "image_sha256",
            "mask_sha256",
            "base_cache_sha256",
            "content_fingerprint",
        ):
            _canonical_sha256(getattr(self, name), name=name)
        if self.content_fingerprint != _loaded_d_v_content_fingerprint(
            self.sample_id,
            self.base_output,
            self.gt_mask,
        ):
            raise ValueError("loaded D_V row memory fingerprint mismatch")


@dataclass(frozen=True)
class LoadedDVCacheBundle:
    """Strict, read-only D_V inputs for downstream calibration."""

    split: Literal["D_V"]
    rows: tuple[LoadedDVCacheRow, ...]
    manifest_path: Path
    base_index_path: Path
    split_manifest_fingerprint: str
    split_manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    base_index_fingerprint: str
    base_index_sha256: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _LoadedBundleSeal or seal.kind != "D_V":
            raise TypeError("LoadedDVCacheBundle must be created by its strict loader")
        if len(seal.bound_objects) != 1 or seal.bound_objects[0] is not self.rows:
            raise TypeError("loaded D_V bundle rows were replaced")
        if seal.bound_values != (
            self.split,
            self.manifest_path,
            self.base_index_path,
            self.split_manifest_fingerprint,
            self.split_manifest_file_sha256,
            self.preprocessing_fingerprint,
            self.base_fingerprint,
            self.base_index_fingerprint,
            self.base_index_sha256,
            self.d_v_image_fingerprint,
            self.d_v_gt_fingerprint,
        ):
            raise TypeError("loaded D_V bundle bound fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        if self.split != "D_V":
            raise ValueError("loaded validation cache bundle split must be exactly D_V")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("loaded D_V cache rows must be a nonempty tuple")
        if any(not isinstance(row, LoadedDVCacheRow) for row in self.rows):
            raise TypeError("loaded D_V cache bundle contains an invalid row")
        sample_ids = tuple(row.sample_id for row in self.rows)
        if sample_ids != tuple(sorted(set(sample_ids))):
            raise ValueError("loaded D_V cache rows must have sorted unique sample IDs")
        for name in ("manifest_path", "base_index_path"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"D_V bundle {name} must be an absolute Path")
        for name in (
            "split_manifest_fingerprint",
            "split_manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "base_index_fingerprint",
            "base_index_sha256",
            "d_v_image_fingerprint",
            "d_v_gt_fingerprint",
        ):
            _canonical_sha256(getattr(self, name), name=name)

    def verify_unchanged(self) -> None:
        """Recheck every persisted D_V input and every in-memory tensor."""

        self._verify_source_seal()
        bindings = [
            (
                self.manifest_path,
                self.split_manifest_file_sha256,
                "split manifest",
            ),
            (self.base_index_path, self.base_index_sha256, "base cache index"),
        ]
        for row in self.rows:
            bindings.extend(
                (
                    (row.image_path, row.image_sha256, f"image {row.sample_id!r}"),
                    (row.mask_path, row.mask_sha256, f"mask {row.sample_id!r}"),
                    (
                        row.base_cache_path,
                        row.base_cache_sha256,
                        f"base cache {row.sample_id!r}",
                    ),
                )
            )
        for path, expected_sha256, label in bindings:
            if path.is_symlink():
                raise RuntimeError(f"bound {label} became a symlink")
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise RuntimeError(f"bound {label} is missing") from error
            if resolved != path or not resolved.is_file():
                raise RuntimeError(f"bound {label} changed path or file type")
            if file_sha256(resolved) != expected_sha256:
                raise RuntimeError(f"bound {label} SHA256 changed")
        for row in self.rows:
            if row.content_fingerprint != _loaded_d_v_content_fingerprint(
                row.sample_id,
                row.base_output,
                row.gt_mask,
            ):
                raise RuntimeError(
                    f"loaded D_V row {row.sample_id!r} tensors changed in memory"
                )


def _valid_mask_2d(
    value: Tensor | None,
    *,
    shape: tuple[int, int],
) -> Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool)
    if not isinstance(value, Tensor):
        raise TypeError("image_valid_mask must be a torch.Tensor or None")
    if value.is_complex():
        raise TypeError("image_valid_mask may not have a complex dtype")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("image_valid_mask contains non-finite values")
    result = value.detach().to(device="cpu", dtype=torch.bool)
    if result.ndim == 4 and result.shape[:2] == (1, 1):
        result = result[0, 0]
    elif result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 2 or tuple(result.shape) != shape:
        raise ValueError(
            "image_valid_mask must match the probability grid as [H,W], "
            "[1,H,W], or [1,1,H,W]"
        )
    if not torch.any(result):
        raise ValueError("image_valid_mask cannot be empty")
    return result.contiguous()


def _identity_pairs(rows: tuple[Any, ...]) -> Tensor:
    values = sorted((int(row.gt_id), int(row.pred_id)) for row in rows)
    if not values:
        return torch.empty((0, 2), dtype=torch.int64)
    return torch.tensor(values, dtype=torch.int64)


def build_state_record(
    sample_id: str,
    base_output: FrozenBaseOutput,
    gt_mask: Tensor,
    occupancy_config: OccupancyConfig = OccupancyConfig(),
    match_config: MatchConfig = MatchConfig(),
    intervention_config: InterventionConfig = InterventionConfig(),
    image_valid_mask: Tensor | None = None,
) -> StateCacheRecord:
    """Build one complete pre-sampling CURE-Lite state-cache record.

    ``base_output`` must contain exactly one sample.  Factual reachability is
    evaluated independently for every unmatched GT using the complete-GT
    recovery diagnostic.  No epoch-specific target is selected here.  Legal
    interventions likewise retain the complete sorted ``(gt_id, pred_id)``
    catalog.
    """

    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if not isinstance(base_output, FrozenBaseOutput):
        raise TypeError("base_output must be a FrozenBaseOutput")
    if base_output.probability.shape[0] != 1:
        raise ValueError("base_output must contain exactly one sample")
    if not isinstance(gt_mask, Tensor):
        raise TypeError("gt_mask must be a torch.Tensor")
    if not isinstance(occupancy_config, OccupancyConfig):
        raise TypeError("occupancy_config must be OccupancyConfig")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be MatchConfig")
    if not isinstance(intervention_config, InterventionConfig):
        raise TypeError("intervention_config must be InterventionConfig")

    raw_occupancy, _ = build_occupancy(
        base_output.probability,
        occupancy_config,
    )
    grid_shape = (int(raw_occupancy.shape[0]), int(raw_occupancy.shape[1]))
    valid_mask = _valid_mask_2d(image_valid_mask, shape=grid_shape)

    # Padding/invalid pixels are removed before connected-component labeling;
    # otherwise two valid components could be spuriously joined through an
    # invalid region.
    occupancy = (raw_occupancy & valid_mask).contiguous()
    pred = instances_from_binary_mask(
        occupancy,
        connectivity=occupancy_config.connectivity,
        min_area=occupancy_config.min_component_area,
    )

    raw_gt = instances_from_binary_mask(
        gt_mask,
        connectivity=occupancy_config.connectivity,
        min_area=occupancy_config.min_component_area,
    )
    if raw_gt.shape != grid_shape:
        raise ValueError("gt_mask must match the base probability grid")
    gt = instances_from_binary_mask(
        raw_gt.occupancy & valid_mask,
        connectivity=occupancy_config.connectivity,
        min_area=occupancy_config.min_component_area,
    )

    base_match = match_components(pred, gt, match_config)
    real_miss_ids = tuple(sorted(base_match.unmatched_gt_ids))
    reachable_miss_ids = tuple(
        gt_id
        for gt_id in real_miss_ids
        if full_gt_recoverable(
            occupancy,
            gt,
            gt_id,
            base_match,
            match_config,
        )
    )
    legal = enumerate_legal_deletions(
        pred,
        gt,
        base_match,
        occupancy,
        match_config=match_config,
        intervention_config=intervention_config,
    )

    return StateCacheRecord(
        sample_id=sample_id,
        occupancy=occupancy,
        pred_labels=pred.labels,
        gt_labels=gt.labels,
        base_match_pairs=_identity_pairs(base_match.pairs),
        real_miss_ids=torch.tensor(real_miss_ids, dtype=torch.int64),
        reachable_miss_ids=torch.tensor(
            reachable_miss_ids,
            dtype=torch.int64,
        ),
        legal_pairs=_identity_pairs(legal),
        image_valid_mask=valid_mask,
    ).normalized()


def _canonical_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    if len(value) != 64 or value != value.lower() or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _loaded_row_content_fingerprint(
    sample_id: str,
    base_output: FrozenBaseOutput,
    state: StateCacheRecord,
) -> str:
    digest = hashlib.sha256()
    digest.update(sample_id.encode("utf-8"))
    tensors = {
        "base.feature": base_output.feature,
        "base.probability": base_output.probability,
        "state.base_match_pairs": state.base_match_pairs,
        "state.gt_labels": state.gt_labels,
        "state.image_valid_mask": state.image_valid_mask,
        "state.legal_pairs": state.legal_pairs,
        "state.occupancy": state.occupancy,
        "state.pred_labels": state.pred_labels,
        "state.reachable_miss_ids": state.reachable_miss_ids,
        "state.real_miss_ids": state.real_miss_ids,
    }
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a tensor")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _loaded_d_v_content_fingerprint(
    sample_id: str,
    base_output: FrozenBaseOutput,
    gt_mask: Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(sample_id.encode("utf-8"))
    tensors = {
        "base.feature": base_output.feature,
        "base.probability": base_output.probability,
        "gt_mask": gt_mask,
    }
    for name in sorted(tensors):
        value = tensors[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _frozen_manifest_file(dataset: ManifestImageDataset) -> Path:
    raw_path = dataset.manifest_path
    if raw_path is None:
        raise ValueError(
            "cache extraction requires the exact split-manifest file path"
        )
    path = Path(raw_path)
    if path.is_symlink():
        raise ValueError("split-manifest file may not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("split-manifest path must identify a regular file")
    persisted = SplitManifest.load(resolved)
    if persisted.fingerprint != dataset.manifest.fingerprint:
        raise ValueError(
            "ManifestImageDataset does not match its declared split-manifest file"
        )
    return resolved


def _record_image_path(raw_path: str, manifest_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    if candidate.is_symlink():
        raise ValueError("manifest image may not be a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("manifest image must be a regular file")
    return resolved


def _adapter_device(adapter: FrozenBaseAdapter) -> torch.device:
    devices = {
        tensor.device
        for tensor in (*tuple(adapter.parameters()), *tuple(adapter.buffers()))
    }
    if not devices:
        return torch.device("cpu")
    if len(devices) != 1:
        raise ValueError("frozen adapter parameters/buffers span multiple devices")
    device = next(iter(devices))
    if device.type == "meta":
        raise ValueError("frozen adapter cannot extract caches from the meta device")
    return device


def _prepare_empty_output(path: str | Path) -> Path:
    output = Path(path).expanduser().absolute()
    if output.is_symlink():
        raise ValueError("cache output directory may not be a symlink")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError("cache output exists and is not a directory")
        if any(output.iterdir()):
            raise FileExistsError("cache output directory must be empty")
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is an atomic create-without-replacement operation.  It
        # preserves the explicit refusal to overwrite an index created by a
        # concurrent or previous run.
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite cache index {path}") from error
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def cache_manifest_split(
    adapter: FrozenBaseAdapter,
    dataset: ManifestImageDataset,
    split: Literal["D_R", "D_V"],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Extract one-sample base caches for an exact ``D_R`` or ``D_V`` view.

    The destination must be absent or empty.  The final strict JSON index binds
    the persisted manifest file, canonical manifest contents, frozen adapter,
    preprocessing contract, every source image, and every cache artifact.  An
    arbitrary dataset transform is rejected because its behavior would not be
    covered by the preprocessing fingerprint.
    """

    if not isinstance(adapter, FrozenBaseAdapter):
        raise TypeError("adapter must be a FrozenBaseAdapter")
    if not isinstance(dataset, ManifestImageDataset):
        raise TypeError("dataset must be a ManifestImageDataset")
    if split not in {"D_R", "D_V"}:
        raise ValueError("base cache extraction permits only D_R or D_V")
    if dataset.split != split:
        raise ValueError("requested split does not match ManifestImageDataset.split")
    if dataset.transform is not None:
        raise ValueError(
            "cache extraction forbids transforms outside the bound preprocessing"
        )
    adapter.validate_preprocessing(dataset.preprocess)
    purpose = "residual_state" if split == "D_R" else "validation"
    dataset.manifest.assert_purpose(purpose, dataset.records)
    if tuple(dataset.records) != dataset.manifest.records_for(split):
        raise ValueError("ManifestImageDataset records are not the exact manifest split")

    manifest_path = _frozen_manifest_file(dataset)
    manifest_file_sha256 = file_sha256(manifest_path)
    manifest_fingerprint = _canonical_sha256(
        dataset.manifest.fingerprint,
        name="split_manifest_fingerprint",
    )
    preprocessing = dataset.preprocess.fingerprint_payload()
    preprocessing_fingerprint = stable_fingerprint(preprocessing)
    base_fingerprint = _canonical_sha256(
        adapter.fingerprint,
        name="adapter.fingerprint",
    )
    device = _adapter_device(adapter)

    output = _prepare_empty_output(output_dir)
    cache_directory = output / "base"
    cache_directory.mkdir(exist_ok=False)
    rows: list[dict[str, Any]] = []
    ordered = tuple(
        sorted(
            enumerate(dataset.records),
            key=lambda item: item[1].sample_id,
        )
    )
    adapter.eval()
    for ordinal, (dataset_index, record) in enumerate(ordered):
        if adapter.fingerprint != base_fingerprint:
            raise RuntimeError("adapter fingerprint changed during cache extraction")
        image_path = _record_image_path(record.image, manifest_path)
        image_sha256 = file_sha256(image_path)
        sample = dataset[dataset_index]
        if not isinstance(sample, LoadedSample):
            raise TypeError("ManifestImageDataset must return LoadedSample records")
        if sample.sample_id != record.sample_id or sample.split != split:
            raise RuntimeError("loaded sample identity differs from the manifest record")
        if Path(sample.image_path).resolve(strict=True) != image_path:
            raise RuntimeError("loaded image path differs from the manifest record")
        if file_sha256(image_path) != image_sha256:
            raise RuntimeError("source image changed while it was being loaded")

        image_batch = sample.image.unsqueeze(0).to(device=device, dtype=torch.float32)
        base_output = adapter(image_batch)
        if base_output.probability.shape[0] != 1:
            raise RuntimeError("adapter returned a non-singleton cache record")
        if file_sha256(image_path) != image_sha256:
            raise RuntimeError("source image changed during base extraction")
        if adapter.fingerprint != base_fingerprint:
            raise RuntimeError("adapter fingerprint changed during cache extraction")

        cache_path = cache_directory / f"{ordinal:06d}.safetensors"
        if cache_path.exists() or cache_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite base cache {cache_path}")
        save_base_cache(
            cache_path,
            base_output,
            fingerprint=base_fingerprint,
            sample_id=sample.sample_id,
            image_fingerprint=image_sha256,
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "split": split,
                "image_path": str(image_path),
                "image_sha256": image_sha256,
                "cache_path": cache_path.relative_to(output).as_posix(),
                "cache_sha256": file_sha256(cache_path),
                "probability_shape": list(base_output.probability.shape),
                "feature_shape": list(base_output.feature.shape),
            }
        )

    if file_sha256(manifest_path) != manifest_file_sha256:
        raise RuntimeError("split-manifest file changed during cache extraction")
    if adapter.fingerprint != base_fingerprint:
        raise RuntimeError("adapter fingerprint changed during cache extraction")
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_BASE_CACHE_INDEX_SCHEMA,
        "base_cache_schema": BASE_CACHE_SCHEMA,
        "dataset": dataset.manifest.dataset,
        "split": split,
        "sample_count": len(rows),
        "split_manifest_fingerprint": manifest_fingerprint,
        "split_manifest_file_sha256": manifest_file_sha256,
        "base_fingerprint": base_fingerprint,
        "preprocessing": preprocessing,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "records": rows,
    }
    payload["index_fingerprint"] = stable_fingerprint(payload)
    _write_new_json(output / "index.json", payload)
    return payload


def _strict_json_object(path: Path, *, name: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON number {value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_nonfinite,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read strict {name}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return payload


def _bound_index_file(root: Path, raw_path: object, *, name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{name} must be a non-empty relative path")
    relative = Path(raw_path)
    if relative.is_absolute() or not relative.parts:
        raise ValueError(f"{name} must be relative to its base index")
    candidate = root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise ValueError(f"{name} must be a normalized child path")
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"{name} may not traverse a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{name} escapes its base-index directory") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must identify a regular file")
    return resolved


def _record_mask_path(raw_path: str | None, manifest_path: Path) -> Path:
    if raw_path is None:
        raise ValueError("D_R cache construction requires every GT mask")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    if candidate.is_symlink():
        raise ValueError("manifest mask may not be a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("manifest mask must be a regular file")
    return resolved


def _validated_shape(value: object, *, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in value
        )
    ):
        raise ValueError(f"{name} must be a non-empty positive-integer shape")
    return value


def _read_base_index_contract(
    index_path: str | Path,
    *,
    expected_split: Literal["D_R", "D_V"],
) -> _ParsedBaseIndex:
    raw_path = Path(index_path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("base cache index may not be a symlink")
    path = raw_path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("base cache index must be a regular file")
    index_sha256 = file_sha256(path)
    payload = _strict_json_object(path, name=f"{expected_split} base cache index")
    if set(payload) != _BASE_INDEX_KEYS:
        raise ValueError("base cache index has missing or unknown fields")
    if (
        payload["schema_version"] != MANIFEST_BASE_CACHE_INDEX_SCHEMA
        or payload["base_cache_schema"] != BASE_CACHE_SCHEMA
        or payload["split"] != expected_split
    ):
        raise ValueError(f"base cache index is not formal {expected_split}")

    fingerprint = _canonical_sha256(
        payload["index_fingerprint"], name="base index fingerprint"
    )
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("index_fingerprint")
    if stable_fingerprint(fingerprint_payload) != fingerprint:
        raise ValueError("base cache index fingerprint does not match its contents")
    _canonical_sha256(
        payload["split_manifest_fingerprint"],
        name="base index manifest fingerprint",
    )
    _canonical_sha256(
        payload["split_manifest_file_sha256"],
        name="base index manifest file SHA256",
    )
    base_fingerprint = _canonical_sha256(
        payload["base_fingerprint"], name="base fingerprint"
    )
    preprocessing = PreprocessConfig.from_fingerprint_payload(
        payload["preprocessing"]
    )
    preprocessing_fingerprint = _canonical_sha256(
        payload["preprocessing_fingerprint"],
        name="preprocessing fingerprint",
    )
    if preprocessing_fingerprint != stable_fingerprint(
        preprocessing.fingerprint_payload()
    ):
        raise ValueError("base index preprocessing fingerprint mismatch")
    dataset = payload["dataset"]
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("base cache dataset must be a non-empty string")

    rows = payload["records"]
    sample_count = payload["sample_count"]
    if (
        not isinstance(rows, list)
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or sample_count != len(rows)
    ):
        raise ValueError("base cache index sample_count mismatch")
    sample_ids: list[str] = []
    feature_shape: tuple[int, int, int, int] | None = None
    seen_cache_paths: set[Path] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _BASE_INDEX_RECORD_KEYS:
            raise ValueError("base cache index record has invalid fields")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("base cache sample_id must be non-empty")
        if row["split"] != expected_split:
            raise ValueError("base cache record split mismatch")
        image_path = Path(row["image_path"])
        if image_path.is_symlink():
            raise ValueError("base cache source image may not be a symlink")
        image_path = image_path.resolve(strict=True)
        if not image_path.is_file():
            raise ValueError("base cache source image must be a regular file")
        image_sha256 = _canonical_sha256(
            row["image_sha256"], name=f"base row {ordinal} image SHA256"
        )
        if file_sha256(image_path) != image_sha256:
            raise ValueError("base cache source image SHA256 mismatch")
        cache_path = _bound_index_file(
            path.parent,
            row["cache_path"],
            name=f"base cache for {sample_id!r}",
        )
        if cache_path in seen_cache_paths:
            raise ValueError("base cache index reuses one cache file")
        seen_cache_paths.add(cache_path)
        cache_sha256 = _canonical_sha256(
            row["cache_sha256"], name=f"base row {ordinal} cache SHA256"
        )
        if file_sha256(cache_path) != cache_sha256:
            raise ValueError("base cache file SHA256 mismatch")
        probability_shape = tuple(
            _validated_shape(
                row["probability_shape"], name="base probability shape"
            )
        )
        current_feature_shape = tuple(
            _validated_shape(row["feature_shape"], name="base feature shape")
        )
        if probability_shape != (
            1,
            1,
            preprocessing.height,
            preprocessing.width,
        ):
            raise ValueError(
                "base probability shape differs from the preprocessing grid"
            )
        if len(current_feature_shape) != 4 or current_feature_shape[0] != 1:
            raise ValueError("base feature shape must be [1,C,h,w]")
        if feature_shape is None:
            feature_shape = current_feature_shape
        elif current_feature_shape != feature_shape:
            raise ValueError("base feature shape must be fixed across a split")
        cached = load_base_cache(
            cache_path,
            expected_fingerprint=base_fingerprint,
            expected_sample_id=sample_id,
            expected_image_fingerprint=image_sha256,
        )
        if (
            tuple(cached.probability.shape) != probability_shape
            or tuple(cached.feature.shape) != current_feature_shape
        ):
            raise ValueError("base cache tensor shape differs from its index")
        sample_ids.append(sample_id)
    if sample_ids != sorted(set(sample_ids)):
        raise ValueError("base cache sample IDs must be sorted and unique")
    if file_sha256(path) != index_sha256:
        raise RuntimeError("base cache index changed during contract validation")
    assert feature_shape is not None
    return _ParsedBaseIndex(
        path=path,
        sha256=index_sha256,
        payload=payload,
        preprocessing=preprocessing,
        feature_shape=feature_shape,
        sample_ids=tuple(sample_ids),
    )


def load_base_cache_pair_contract(
    d_r_index_path: str | Path,
    d_v_index_path: str | Path,
) -> BaseCachePairContract:
    """Validate the model-independent D_R/D_V evidence interface."""

    d_r = _read_base_index_contract(d_r_index_path, expected_split="D_R")
    d_v = _read_base_index_contract(d_v_index_path, expected_split="D_V")
    common_fields = (
        "dataset",
        "split_manifest_fingerprint",
        "split_manifest_file_sha256",
        "base_fingerprint",
        "preprocessing",
        "preprocessing_fingerprint",
    )
    for field in common_fields:
        if d_r.payload[field] != d_v.payload[field]:
            raise ValueError(f"D_R/D_V base cache mismatch for {field}")
    if d_r.feature_shape != d_v.feature_shape:
        raise ValueError("D_R/D_V base feature shapes differ")
    if set(d_r.sample_ids) & set(d_v.sample_ids):
        raise ValueError("D_R and D_V base cache memberships overlap")
    return BaseCachePairContract(
        dataset=d_r.payload["dataset"],
        split_manifest_fingerprint=d_r.payload[
            "split_manifest_fingerprint"
        ],
        split_manifest_file_sha256=d_r.payload[
            "split_manifest_file_sha256"
        ],
        base_fingerprint=d_r.payload["base_fingerprint"],
        preprocessing=d_r.preprocessing,
        preprocessing_fingerprint=d_r.payload[
            "preprocessing_fingerprint"
        ],
        feature_channels=d_r.feature_shape[1],
        feature_shape=d_r.feature_shape,
        d_r_index_path=d_r.path,
        d_r_index_sha256=d_r.sha256,
        d_r_index_fingerprint=d_r.payload["index_fingerprint"],
        d_r_sample_ids=d_r.sample_ids,
        d_v_index_path=d_v.path,
        d_v_index_sha256=d_v.sha256,
        d_v_index_fingerprint=d_v.payload["index_fingerprint"],
        d_v_sample_ids=d_v.sample_ids,
    )


def materialize_base_cache_bundle(
    source_index_path: str | Path,
    output_dir: str | Path,
    *,
    expected_split: Literal["D_R", "D_V"],
    expected_base_fingerprint: str,
) -> dict[str, Any]:
    """Copy one verified generic base-cache bundle into a Stage-A run."""

    source = _read_base_index_contract(
        source_index_path,
        expected_split=expected_split,
    )
    expected = _canonical_sha256(
        expected_base_fingerprint, name="expected_base_fingerprint"
    )
    if source.payload["base_fingerprint"] != expected:
        raise ValueError("base cache bundle uses a different base fingerprint")
    output = _prepare_empty_output(output_dir)
    seen_destinations: set[Path] = set()
    for row in source.payload["records"]:
        relative = Path(row["cache_path"])
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "base"
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("base cache path is not a canonical bundle child")
        source_cache = _bound_index_file(
            source.path.parent,
            row["cache_path"],
            name=f"base cache for {row['sample_id']!r}",
        )
        destination = output / relative
        if destination in seen_destinations:
            raise ValueError("base cache bundle reuses one destination")
        seen_destinations.add(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite base cache {destination}")
        before = file_sha256(source_cache)
        if before != row["cache_sha256"]:
            raise RuntimeError("base cache changed before materialization")
        shutil.copyfile(source_cache, destination)
        if (
            file_sha256(source_cache) != before
            or file_sha256(destination) != before
        ):
            raise RuntimeError("materialized base cache differs from its source")
    destination_index = output / "index.json"
    shutil.copyfile(source.path, destination_index)
    if (
        file_sha256(source.path) != source.sha256
        or file_sha256(destination_index) != source.sha256
    ):
        raise RuntimeError("materialized base index differs from its source")
    copied = _read_base_index_contract(
        destination_index,
        expected_split=expected_split,
    )
    if copied.payload != source.payload:
        raise RuntimeError("materialized base index contents changed")
    return copied.payload


def _gt_catalog_fingerprint(
    manifest_fingerprint: str,
    gt_rows: list[dict[str, str]],
) -> str:
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-d-r-gt-catalog-v1",
            "split_manifest_fingerprint": manifest_fingerprint,
            "split": "D_R",
            "records": gt_rows,
        }
    )


def _catalog_counts(state: StateCacheRecord) -> dict[str, int]:
    return {
        "pred_components": int(
            torch.count_nonzero(torch.unique(state.pred_labels) > 0)
        ),
        "gt_components": int(
            torch.count_nonzero(torch.unique(state.gt_labels) > 0)
        ),
        "base_matches": int(state.base_match_pairs.shape[0]),
        "real_misses": int(state.real_miss_ids.numel()),
        "reachable_misses": int(state.reachable_miss_ids.numel()),
        "legal_pairs": int(state.legal_pairs.shape[0]),
    }


def cache_d_r_states(
    base_index_path: str | Path,
    dataset: ManifestImageDataset,
    output_dir: str | Path,
    *,
    expected_base_fingerprint: str,
    occupancy_config: OccupancyConfig = OccupancyConfig(),
    match_config: MatchConfig = MatchConfig(),
    intervention_config: InterventionConfig = InterventionConfig(),
) -> dict[str, Any]:
    """Build the exact ``D_R`` state cache from a verified base-cache index.

    Every source binding is checked before the destination is created.  State
    records use an explicit all-valid image mask because
    :class:`ManifestImageDataset` performs a dense rectangular resize without
    padding.  The complete factual and legal catalogs are produced by
    :func:`build_state_record` and are never sampled in this offline step.
    """

    if not isinstance(dataset, ManifestImageDataset):
        raise TypeError("dataset must be a ManifestImageDataset")
    if dataset.split != "D_R":
        raise ValueError("state-cache production permits only exact D_R data")
    if dataset.transform is not None:
        raise ValueError(
            "state-cache production forbids transforms outside preprocessing"
        )
    if not isinstance(occupancy_config, OccupancyConfig):
        raise TypeError("occupancy_config must be OccupancyConfig")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be MatchConfig")
    if not isinstance(intervention_config, InterventionConfig):
        raise TypeError("intervention_config must be InterventionConfig")
    dataset.manifest.assert_purpose("residual_state", dataset.records)
    if tuple(dataset.records) != dataset.manifest.records_for("D_R"):
        raise ValueError("ManifestImageDataset records are not the exact D_R split")

    declared_base_fingerprint = _canonical_sha256(
        expected_base_fingerprint,
        name="expected_base_fingerprint",
    )
    raw_base_index_path = Path(base_index_path).expanduser()
    if raw_base_index_path.is_symlink():
        raise ValueError("base cache index may not be a symlink")
    base_index_file = raw_base_index_path.resolve(strict=True)
    if not base_index_file.is_file():
        raise ValueError("base cache index must be a regular file")
    base_index_sha256 = file_sha256(base_index_file)
    base_index = _strict_json_object(base_index_file, name="base cache index")
    if set(base_index) != _BASE_INDEX_KEYS:
        raise ValueError("base cache index has missing or unknown fields")
    index_fingerprint = _canonical_sha256(
        base_index["index_fingerprint"],
        name="base index fingerprint",
    )
    fingerprint_payload = dict(base_index)
    fingerprint_payload.pop("index_fingerprint")
    if stable_fingerprint(fingerprint_payload) != index_fingerprint:
        raise ValueError("base cache index fingerprint does not match its contents")
    if (
        base_index["schema_version"] != MANIFEST_BASE_CACHE_INDEX_SCHEMA
        or base_index["base_cache_schema"] != BASE_CACHE_SCHEMA
        or base_index["split"] != "D_R"
    ):
        raise ValueError("base cache index schema or split is not formal D_R")
    if base_index["dataset"] != dataset.manifest.dataset:
        raise ValueError("base cache index dataset mismatch")
    if base_index["base_fingerprint"] != declared_base_fingerprint:
        raise ValueError("base cache index uses a different base fingerprint")

    manifest_path = _frozen_manifest_file(dataset)
    manifest_file_sha256 = file_sha256(manifest_path)
    manifest_fingerprint = dataset.manifest.fingerprint
    if (
        base_index["split_manifest_fingerprint"] != manifest_fingerprint
        or base_index["split_manifest_file_sha256"] != manifest_file_sha256
    ):
        raise ValueError("base cache index manifest binding mismatch")
    preprocessing = dataset.preprocess.fingerprint_payload()
    preprocessing_fingerprint = stable_fingerprint(preprocessing)
    if (
        base_index["preprocessing"] != preprocessing
        or base_index["preprocessing_fingerprint"]
        != preprocessing_fingerprint
    ):
        raise ValueError("base cache index preprocessing binding mismatch")

    raw_rows = base_index["records"]
    if not isinstance(raw_rows, list):
        raise ValueError("base cache index records must be a list")
    if (
        isinstance(base_index["sample_count"], bool)
        or not isinstance(base_index["sample_count"], int)
        or base_index["sample_count"] != len(raw_rows)
    ):
        raise ValueError("base cache index sample_count mismatch")
    ordered_records = tuple(
        sorted(
            enumerate(dataset.records),
            key=lambda item: item[1].sample_id,
        )
    )
    expected_ids = [record.sample_id for _, record in ordered_records]
    if len(raw_rows) != len(ordered_records):
        raise ValueError("base cache index is not the exact D_R membership")

    sources: list[dict[str, Any]] = []
    seen_cache_paths: set[Path] = set()
    gt_rows: list[dict[str, str]] = []
    for raw_row, (dataset_index, record) in zip(
        raw_rows,
        ordered_records,
        strict=True,
    ):
        if not isinstance(raw_row, dict) or set(raw_row) != _BASE_INDEX_RECORD_KEYS:
            raise ValueError("base cache index record has invalid fields")
        if raw_row["sample_id"] != record.sample_id or raw_row["split"] != "D_R":
            raise ValueError("base cache index is not the exact D_R membership")
        image_path = _record_image_path(record.image, manifest_path)
        image_sha256 = file_sha256(image_path)
        if (
            raw_row["image_path"] != str(image_path)
            or raw_row["image_sha256"] != image_sha256
        ):
            raise ValueError("base cache index image binding mismatch")
        cache_path = _bound_index_file(
            base_index_file.parent,
            raw_row["cache_path"],
            name=f"base cache for {record.sample_id!r}",
        )
        if cache_path in seen_cache_paths:
            raise ValueError("base cache index reuses one cache file")
        seen_cache_paths.add(cache_path)
        cache_sha256 = file_sha256(cache_path)
        if raw_row["cache_sha256"] != cache_sha256:
            raise ValueError("base cache file SHA256 mismatch")
        probability_shape = _validated_shape(
            raw_row["probability_shape"],
            name="base probability shape",
        )
        feature_shape = _validated_shape(
            raw_row["feature_shape"],
            name="base feature shape",
        )
        mask_path = _record_mask_path(record.mask, manifest_path)
        mask_sha256 = file_sha256(mask_path)
        gt_rows.append(
            {"sample_id": record.sample_id, "mask_sha256": mask_sha256}
        )
        sources.append(
            {
                "dataset_index": dataset_index,
                "record": record,
                "image_path": image_path,
                "image_sha256": image_sha256,
                "mask_path": mask_path,
                "mask_sha256": mask_sha256,
                "base_cache_path": cache_path,
                "base_cache_sha256": cache_sha256,
                "probability_shape": probability_shape,
                "feature_shape": feature_shape,
            }
        )
    if [row["sample_id"] for row in raw_rows] != expected_ids:
        raise ValueError("base cache index records must be sorted by sample_id")

    gt_fingerprint = _gt_catalog_fingerprint(manifest_fingerprint, gt_rows)
    occupancy_payload = config_to_dict(occupancy_config)
    matching_payload = config_to_dict(match_config)
    intervention_payload = config_to_dict(intervention_config)
    state_fingerprint = build_state_fingerprint(
        schema_version=STATE_CACHE_SCHEMA,
        base_fingerprint=declared_base_fingerprint,
        split_manifest_sha256=manifest_file_sha256,
        gt_fingerprint=gt_fingerprint,
        occupancy_config=occupancy_payload,
        matching_config=matching_payload,
        intervention_config=intervention_payload,
    )

    # All cheap provenance/content checks above complete before any state output
    # is created.  A failed or interrupted extraction leaves a nonempty output,
    # which cannot be silently reused by a later invocation.
    output = _prepare_empty_output(output_dir)
    state_directory = output / "state"
    state_directory.mkdir(exist_ok=False)
    output_rows: list[dict[str, Any]] = []
    for ordinal, source in enumerate(sources):
        if file_sha256(source["image_path"]) != source["image_sha256"]:
            raise RuntimeError("source image changed during state-cache production")
        if file_sha256(source["mask_path"]) != source["mask_sha256"]:
            raise RuntimeError("GT mask changed during state-cache production")
        if (
            file_sha256(source["base_cache_path"])
            != source["base_cache_sha256"]
        ):
            raise RuntimeError("base cache changed during state-cache production")

        sample = dataset[source["dataset_index"]]
        record = source["record"]
        if not isinstance(sample, LoadedSample):
            raise TypeError("ManifestImageDataset must return LoadedSample records")
        if sample.sample_id != record.sample_id or sample.split != "D_R":
            raise RuntimeError("loaded D_R sample differs from its manifest record")
        if (
            Path(sample.image_path).resolve(strict=True) != source["image_path"]
            or Path(sample.mask_path).resolve(strict=True) != source["mask_path"]
        ):
            raise RuntimeError("loaded D_R asset path differs from the manifest")

        base_output = load_base_cache(
            source["base_cache_path"],
            expected_fingerprint=declared_base_fingerprint,
            expected_sample_id=record.sample_id,
            expected_image_fingerprint=source["image_sha256"],
        )
        if (
            list(base_output.probability.shape) != source["probability_shape"]
            or list(base_output.feature.shape) != source["feature_shape"]
        ):
            raise ValueError("base cache tensor shape differs from its strict index")
        state = build_state_record(
            record.sample_id,
            base_output,
            sample.gt_mask,
            occupancy_config,
            match_config,
            intervention_config,
            image_valid_mask=torch.ones_like(sample.gt_mask, dtype=torch.bool),
        )
        state_path = state_directory / f"{ordinal:06d}.npz"
        if state_path.exists() or state_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite state cache {state_path}")
        save_state_cache(state_path, state, fingerprint=state_fingerprint)
        output_rows.append(
            {
                "sample_id": record.sample_id,
                "split": "D_R",
                "image_path": str(source["image_path"]),
                "image_sha256": source["image_sha256"],
                "mask_path": str(source["mask_path"]),
                "mask_sha256": source["mask_sha256"],
                "base_cache_path": str(source["base_cache_path"]),
                "base_cache_sha256": source["base_cache_sha256"],
                "state_cache_path": state_path.relative_to(output).as_posix(),
                "state_cache_sha256": file_sha256(state_path),
                "catalog_counts": _catalog_counts(state),
            }
        )

    if file_sha256(base_index_file) != base_index_sha256:
        raise RuntimeError("base cache index changed during state-cache production")
    if file_sha256(manifest_path) != manifest_file_sha256:
        raise RuntimeError("split manifest changed during state-cache production")
    for source in sources:
        for path_key, sha_key, label in (
            ("image_path", "image_sha256", "source image"),
            ("mask_path", "mask_sha256", "GT mask"),
            ("base_cache_path", "base_cache_sha256", "base cache"),
        ):
            if file_sha256(source[path_key]) != source[sha_key]:
                raise RuntimeError(f"{label} changed during state-cache production")

    payload = {
        "schema_version": MANIFEST_STATE_CACHE_INDEX_SCHEMA,
        "state_cache_schema": STATE_CACHE_SCHEMA,
        "dataset": dataset.manifest.dataset,
        "split": "D_R",
        "sample_count": len(output_rows),
        "split_manifest_fingerprint": manifest_fingerprint,
        "split_manifest_file_sha256": manifest_file_sha256,
        "base_fingerprint": declared_base_fingerprint,
        "base_index": {
            "path": str(base_index_file),
            "sha256": base_index_sha256,
            "index_fingerprint": index_fingerprint,
        },
        "preprocessing": preprocessing,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "occupancy_config": occupancy_payload,
        "matching_config": matching_payload,
        "intervention_config": intervention_payload,
        "gt_fingerprint": gt_fingerprint,
        "state_fingerprint": state_fingerprint,
        "records": output_rows,
    }
    payload["index_fingerprint"] = stable_fingerprint(payload)
    _write_new_json(output / "index.json", payload)
    return payload


def _same_state(left: StateCacheRecord, right: StateCacheRecord) -> bool:
    if left.sample_id != right.sample_id:
        return False
    return all(
        torch.equal(getattr(left, name), getattr(right, name))
        for name in (
            "occupancy",
            "pred_labels",
            "gt_labels",
            "base_match_pairs",
            "real_miss_ids",
            "reachable_miss_ids",
            "legal_pairs",
            "image_valid_mask",
        )
    )


def load_d_r_cache_bundle(
    state_index_path: str | Path,
    dataset: ManifestImageDataset,
    *,
    expected_base_fingerprint: str,
) -> LoadedDRCacheBundle:
    """Strictly reload and semantically verify one formal D_R cache bundle."""

    if not isinstance(dataset, ManifestImageDataset):
        raise TypeError("dataset must be a ManifestImageDataset")
    if dataset.split != "D_R":
        raise ValueError("D_R cache bundle loading permits only exact D_R data")
    if dataset.transform is not None:
        raise ValueError(
            "D_R cache bundle loading forbids transforms outside preprocessing"
        )
    dataset.manifest.assert_purpose("residual_state", dataset.records)
    if tuple(dataset.records) != dataset.manifest.records_for("D_R"):
        raise ValueError("ManifestImageDataset records are not the exact D_R split")
    base_fingerprint = _canonical_sha256(
        expected_base_fingerprint,
        name="expected_base_fingerprint",
    )

    raw_state_index_path = Path(state_index_path).expanduser()
    if raw_state_index_path.is_symlink():
        raise ValueError("state cache index may not be a symlink")
    state_index_file = raw_state_index_path.resolve(strict=True)
    if not state_index_file.is_file():
        raise ValueError("state cache index must be a regular file")
    state_index_sha256 = file_sha256(state_index_file)
    state_index = _strict_json_object(state_index_file, name="state cache index")
    if set(state_index) != _STATE_INDEX_KEYS:
        raise ValueError("state cache index has missing or unknown fields")
    state_index_fingerprint = _canonical_sha256(
        state_index["index_fingerprint"],
        name="state index fingerprint",
    )
    state_fingerprint_payload = dict(state_index)
    state_fingerprint_payload.pop("index_fingerprint")
    if stable_fingerprint(state_fingerprint_payload) != state_index_fingerprint:
        raise ValueError("state cache index fingerprint does not match its contents")
    if (
        state_index["schema_version"] != MANIFEST_STATE_CACHE_INDEX_SCHEMA
        or state_index["state_cache_schema"] != STATE_CACHE_SCHEMA
        or state_index["split"] != "D_R"
    ):
        raise ValueError("state cache index schema or split is not formal D_R")
    if state_index["dataset"] != dataset.manifest.dataset:
        raise ValueError("state cache index dataset mismatch")
    if state_index["base_fingerprint"] != base_fingerprint:
        raise ValueError("state cache index uses a different base fingerprint")

    manifest_path = _frozen_manifest_file(dataset)
    manifest_file_sha256 = file_sha256(manifest_path)
    manifest_fingerprint = dataset.manifest.fingerprint
    if (
        state_index["split_manifest_fingerprint"] != manifest_fingerprint
        or state_index["split_manifest_file_sha256"] != manifest_file_sha256
    ):
        raise ValueError("state cache index manifest binding mismatch")
    preprocessing = dataset.preprocess.fingerprint_payload()
    preprocessing_fingerprint = stable_fingerprint(preprocessing)
    if (
        state_index["preprocessing"] != preprocessing
        or state_index["preprocessing_fingerprint"]
        != preprocessing_fingerprint
    ):
        raise ValueError("state cache index preprocessing binding mismatch")
    for name in (
        "occupancy_config",
        "matching_config",
        "intervention_config",
    ):
        if not isinstance(state_index[name], dict):
            raise ValueError(f"state cache index {name} must be a mapping")
    try:
        occupancy_config = OccupancyConfig(**state_index["occupancy_config"])
        match_config = MatchConfig(**state_index["matching_config"])
        intervention_config = InterventionConfig(
            **state_index["intervention_config"]
        )
    except (TypeError, ValueError) as error:
        raise ValueError("state cache index contains an invalid bound config") from error
    occupancy_payload = config_to_dict(occupancy_config)
    matching_payload = config_to_dict(match_config)
    intervention_payload = config_to_dict(intervention_config)
    if (
        state_index["occupancy_config"] != occupancy_payload
        or state_index["matching_config"] != matching_payload
        or state_index["intervention_config"] != intervention_payload
    ):
        raise ValueError("state cache index config payload is not canonical")

    base_binding = state_index["base_index"]
    if not isinstance(base_binding, dict) or set(base_binding) != {
        "path",
        "sha256",
        "index_fingerprint",
    }:
        raise ValueError("state cache index has an invalid base-index binding")
    if not isinstance(base_binding["path"], str) or not Path(
        base_binding["path"]
    ).is_absolute():
        raise ValueError("bound base cache index path must be absolute")
    raw_base_index_path = Path(base_binding["path"])
    if raw_base_index_path.is_symlink():
        raise ValueError("bound base cache index may not be a symlink")
    base_index_file = raw_base_index_path.resolve(strict=True)
    if not base_index_file.is_file():
        raise ValueError("bound base cache index must be a regular file")
    base_index_sha256 = file_sha256(base_index_file)
    if base_binding["sha256"] != base_index_sha256:
        raise ValueError("bound base cache index SHA256 mismatch")
    base_index = _strict_json_object(base_index_file, name="base cache index")
    if set(base_index) != _BASE_INDEX_KEYS:
        raise ValueError("base cache index has missing or unknown fields")
    base_index_fingerprint = _canonical_sha256(
        base_index["index_fingerprint"],
        name="base index fingerprint",
    )
    base_fingerprint_payload = dict(base_index)
    base_fingerprint_payload.pop("index_fingerprint")
    if (
        stable_fingerprint(base_fingerprint_payload) != base_index_fingerprint
        or base_binding["index_fingerprint"] != base_index_fingerprint
    ):
        raise ValueError("base cache index fingerprint binding mismatch")
    expected_base_globals = {
        "schema_version": MANIFEST_BASE_CACHE_INDEX_SCHEMA,
        "base_cache_schema": BASE_CACHE_SCHEMA,
        "dataset": dataset.manifest.dataset,
        "split": "D_R",
        "split_manifest_fingerprint": manifest_fingerprint,
        "split_manifest_file_sha256": manifest_file_sha256,
        "base_fingerprint": base_fingerprint,
        "preprocessing": preprocessing,
        "preprocessing_fingerprint": preprocessing_fingerprint,
    }
    for key, expected in expected_base_globals.items():
        if base_index[key] != expected:
            raise ValueError(f"base cache index {key} mismatch")

    state_rows = state_index["records"]
    base_rows = base_index["records"]
    if not isinstance(state_rows, list) or not isinstance(base_rows, list):
        raise ValueError("cache index records must be lists")
    for index, (declared_count, rows) in enumerate(
        (
            (state_index["sample_count"], state_rows),
            (base_index["sample_count"], base_rows),
        )
    ):
        if (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count != len(rows)
        ):
            kind = "state" if index == 0 else "base"
            raise ValueError(f"{kind} cache index sample_count mismatch")
    ordered_records = tuple(
        sorted(
            enumerate(dataset.records),
            key=lambda item: item[1].sample_id,
        )
    )
    if len(state_rows) != len(ordered_records) or len(base_rows) != len(
        ordered_records
    ):
        raise ValueError("cache indexes are not the exact D_R membership")

    verified_sources: list[dict[str, Any]] = []
    gt_rows: list[dict[str, str]] = []
    seen_base_paths: set[Path] = set()
    seen_state_paths: set[Path] = set()
    for state_row, base_row, (dataset_index, record) in zip(
        state_rows,
        base_rows,
        ordered_records,
        strict=True,
    ):
        if not isinstance(base_row, dict) or set(base_row) != _BASE_INDEX_RECORD_KEYS:
            raise ValueError("base cache index record has invalid fields")
        if not isinstance(state_row, dict) or set(state_row) != _STATE_INDEX_RECORD_KEYS:
            raise ValueError("state cache index record has invalid fields")
        if (
            base_row["sample_id"] != record.sample_id
            or state_row["sample_id"] != record.sample_id
            or base_row["split"] != "D_R"
            or state_row["split"] != "D_R"
        ):
            raise ValueError("cache indexes are not the exact D_R membership")
        image_path = _record_image_path(record.image, manifest_path)
        image_sha256 = file_sha256(image_path)
        if any(
            row["image_path"] != str(image_path)
            or row["image_sha256"] != image_sha256
            for row in (base_row, state_row)
        ):
            raise ValueError("cache index image binding mismatch")
        mask_path = _record_mask_path(record.mask, manifest_path)
        mask_sha256 = file_sha256(mask_path)
        if (
            state_row["mask_path"] != str(mask_path)
            or state_row["mask_sha256"] != mask_sha256
        ):
            raise ValueError("state cache index mask binding mismatch")
        base_cache_path = _bound_index_file(
            base_index_file.parent,
            base_row["cache_path"],
            name=f"base cache for {record.sample_id!r}",
        )
        if (
            base_cache_path in seen_base_paths
            or state_row["base_cache_path"] != str(base_cache_path)
        ):
            raise ValueError("state/base index cache-path binding mismatch")
        seen_base_paths.add(base_cache_path)
        base_cache_sha256 = file_sha256(base_cache_path)
        if (
            base_row["cache_sha256"] != base_cache_sha256
            or state_row["base_cache_sha256"] != base_cache_sha256
        ):
            raise ValueError("base cache SHA256 binding mismatch")
        state_cache_path = _bound_index_file(
            state_index_file.parent,
            state_row["state_cache_path"],
            name=f"state cache for {record.sample_id!r}",
        )
        if state_cache_path in seen_state_paths:
            raise ValueError("state cache index reuses one state file")
        seen_state_paths.add(state_cache_path)
        state_cache_sha256 = file_sha256(state_cache_path)
        if state_row["state_cache_sha256"] != state_cache_sha256:
            raise ValueError("state cache SHA256 binding mismatch")
        probability_shape = _validated_shape(
            base_row["probability_shape"],
            name="base probability shape",
        )
        feature_shape = _validated_shape(
            base_row["feature_shape"],
            name="base feature shape",
        )
        counts = state_row["catalog_counts"]
        if (
            not isinstance(counts, dict)
            or set(counts) != _CATALOG_COUNT_KEYS
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in counts.values()
            )
        ):
            raise ValueError("state cache catalog_counts are invalid")
        gt_rows.append(
            {"sample_id": record.sample_id, "mask_sha256": mask_sha256}
        )
        verified_sources.append(
            {
                "dataset_index": dataset_index,
                "record": record,
                "image_path": image_path,
                "image_sha256": image_sha256,
                "mask_path": mask_path,
                "mask_sha256": mask_sha256,
                "base_cache_path": base_cache_path,
                "base_cache_sha256": base_cache_sha256,
                "state_cache_path": state_cache_path,
                "state_cache_sha256": state_cache_sha256,
                "probability_shape": probability_shape,
                "feature_shape": feature_shape,
                "catalog_counts": counts,
            }
        )

    gt_fingerprint = _gt_catalog_fingerprint(manifest_fingerprint, gt_rows)
    if state_index["gt_fingerprint"] != gt_fingerprint:
        raise ValueError("state cache index GT fingerprint mismatch")
    expected_state_fingerprint = build_state_fingerprint(
        schema_version=STATE_CACHE_SCHEMA,
        base_fingerprint=base_fingerprint,
        split_manifest_sha256=manifest_file_sha256,
        gt_fingerprint=gt_fingerprint,
        occupancy_config=occupancy_payload,
        matching_config=matching_payload,
        intervention_config=intervention_payload,
    )
    if state_index["state_fingerprint"] != expected_state_fingerprint:
        raise ValueError("state cache fingerprint does not match its dependencies")

    loaded_rows: list[LoadedDRCacheRow] = []
    for source in verified_sources:
        sample = dataset[source["dataset_index"]]
        record = source["record"]
        if not isinstance(sample, LoadedSample):
            raise TypeError("ManifestImageDataset must return LoadedSample records")
        if sample.sample_id != record.sample_id or sample.split != "D_R":
            raise RuntimeError("loaded D_R sample differs from its manifest record")
        if (
            Path(sample.image_path).resolve(strict=True) != source["image_path"]
            or Path(sample.mask_path).resolve(strict=True) != source["mask_path"]
        ):
            raise RuntimeError("loaded D_R asset path differs from the manifest")
        base_output = load_base_cache(
            source["base_cache_path"],
            expected_fingerprint=base_fingerprint,
            expected_sample_id=record.sample_id,
            expected_image_fingerprint=source["image_sha256"],
        )
        if (
            list(base_output.probability.shape) != source["probability_shape"]
            or list(base_output.feature.shape) != source["feature_shape"]
        ):
            raise ValueError("base cache tensor shape differs from its strict index")
        state = load_state_cache(
            source["state_cache_path"],
            expected_fingerprint=expected_state_fingerprint,
            expected_sample_id=record.sample_id,
        ).normalized()
        rebuilt = build_state_record(
            record.sample_id,
            base_output,
            sample.gt_mask,
            occupancy_config,
            match_config,
            intervention_config,
            image_valid_mask=torch.ones_like(sample.gt_mask, dtype=torch.bool),
        )
        if not _same_state(state, rebuilt):
            raise ValueError("state cache differs from normative D_R reconstruction")
        if _catalog_counts(state) != source["catalog_counts"]:
            raise ValueError("state cache catalog counts differ from the strict index")
        loaded_rows.append(
            LoadedDRCacheRow(
                sample_id=record.sample_id,
                base_output=base_output,
                state=state,
                image_path=source["image_path"],
                mask_path=source["mask_path"],
                base_cache_path=source["base_cache_path"],
                state_cache_path=source["state_cache_path"],
                image_sha256=source["image_sha256"],
                mask_sha256=source["mask_sha256"],
                base_cache_sha256=source["base_cache_sha256"],
                state_cache_sha256=source["state_cache_sha256"],
                content_fingerprint=_loaded_row_content_fingerprint(
                    record.sample_id,
                    base_output,
                    state,
                ),
            )
        )

    if file_sha256(state_index_file) != state_index_sha256:
        raise RuntimeError("state cache index changed while loading the bundle")
    if file_sha256(base_index_file) != base_index_sha256:
        raise RuntimeError("base cache index changed while loading the bundle")
    if file_sha256(manifest_path) != manifest_file_sha256:
        raise RuntimeError("split manifest changed while loading the bundle")
    for source in verified_sources:
        for path_key, sha_key, label in (
            ("image_path", "image_sha256", "source image"),
            ("mask_path", "mask_sha256", "GT mask"),
            ("base_cache_path", "base_cache_sha256", "base cache"),
            ("state_cache_path", "state_cache_sha256", "state cache"),
        ):
            if file_sha256(source[path_key]) != source[sha_key]:
                raise RuntimeError(f"{label} changed while loading the bundle")

    rows_tuple = tuple(loaded_rows)
    seal = _LoadedBundleSeal(
        kind="D_R",
        bound_objects=(
            rows_tuple,
            occupancy_config,
            match_config,
            intervention_config,
        ),
        bound_values=(
            "D_R",
            manifest_path,
            base_index_file,
            state_index_file,
            manifest_fingerprint,
            manifest_file_sha256,
            preprocessing_fingerprint,
            base_fingerprint,
            expected_state_fingerprint,
            gt_fingerprint,
            base_index_fingerprint,
            base_index_sha256,
            state_index_fingerprint,
            state_index_sha256,
        ),
    )
    bundle = LoadedDRCacheBundle(
        split="D_R",
        rows=rows_tuple,
        occupancy_config=occupancy_config,
        match_config=match_config,
        intervention_config=intervention_config,
        manifest_path=manifest_path,
        base_index_path=base_index_file,
        state_index_path=state_index_file,
        split_manifest_fingerprint=manifest_fingerprint,
        split_manifest_file_sha256=manifest_file_sha256,
        preprocessing_fingerprint=preprocessing_fingerprint,
        base_fingerprint=base_fingerprint,
        state_fingerprint=expected_state_fingerprint,
        gt_fingerprint=gt_fingerprint,
        base_index_fingerprint=base_index_fingerprint,
        base_index_sha256=base_index_sha256,
        state_index_fingerprint=state_index_fingerprint,
        state_index_sha256=state_index_sha256,
        _verification_token=seal,
    )
    bundle.verify_unchanged()
    return bundle


def load_d_v_cache_bundle(
    base_index_path: str | Path,
    dataset: ManifestImageDataset,
    *,
    expected_base_fingerprint: str,
) -> LoadedDVCacheBundle:
    """Strictly load only D_V GT masks and their frozen-base outputs."""

    if not isinstance(dataset, ManifestImageDataset):
        raise TypeError("dataset must be a ManifestImageDataset")
    if dataset.split != "D_V":
        raise ValueError("validation cache bundle loading permits only exact D_V")
    if dataset.transform is not None:
        raise ValueError(
            "validation cache bundle loading forbids unbound transforms"
        )
    dataset.manifest.assert_purpose("validation", dataset.records)
    if tuple(dataset.records) != dataset.manifest.records_for("D_V"):
        raise ValueError("ManifestImageDataset records are not the exact D_V split")
    base_fingerprint = _canonical_sha256(
        expected_base_fingerprint,
        name="expected_base_fingerprint",
    )

    raw_index_path = Path(base_index_path).expanduser()
    if raw_index_path.is_symlink():
        raise ValueError("D_V base cache index may not be a symlink")
    index_path = raw_index_path.resolve(strict=True)
    if not index_path.is_file():
        raise ValueError("D_V base cache index must be a regular file")
    index_sha256 = file_sha256(index_path)
    index = _strict_json_object(index_path, name="D_V base cache index")
    if set(index) != _BASE_INDEX_KEYS:
        raise ValueError("D_V base cache index has missing or unknown fields")
    index_fingerprint = _canonical_sha256(
        index["index_fingerprint"],
        name="D_V base index fingerprint",
    )
    fingerprint_payload = dict(index)
    fingerprint_payload.pop("index_fingerprint")
    if stable_fingerprint(fingerprint_payload) != index_fingerprint:
        raise ValueError("D_V base cache index fingerprint does not match its contents")
    if (
        index["schema_version"] != MANIFEST_BASE_CACHE_INDEX_SCHEMA
        or index["base_cache_schema"] != BASE_CACHE_SCHEMA
        or index["split"] != "D_V"
    ):
        raise ValueError("base cache index schema or split is not formal D_V")
    if index["dataset"] != dataset.manifest.dataset:
        raise ValueError("D_V base cache index dataset mismatch")
    if index["base_fingerprint"] != base_fingerprint:
        raise ValueError("D_V base cache index uses a different base fingerprint")

    manifest_path = _frozen_manifest_file(dataset)
    manifest_file_sha256 = file_sha256(manifest_path)
    manifest_fingerprint = dataset.manifest.fingerprint
    if (
        index["split_manifest_fingerprint"] != manifest_fingerprint
        or index["split_manifest_file_sha256"] != manifest_file_sha256
    ):
        raise ValueError("D_V base cache index manifest binding mismatch")
    preprocessing = dataset.preprocess.fingerprint_payload()
    preprocessing_fingerprint = stable_fingerprint(preprocessing)
    if (
        index["preprocessing"] != preprocessing
        or index["preprocessing_fingerprint"] != preprocessing_fingerprint
    ):
        raise ValueError("D_V base cache index preprocessing binding mismatch")

    raw_rows = index["records"]
    if not isinstance(raw_rows, list):
        raise ValueError("D_V base cache index records must be a list")
    if (
        isinstance(index["sample_count"], bool)
        or not isinstance(index["sample_count"], int)
        or index["sample_count"] != len(raw_rows)
    ):
        raise ValueError("D_V base cache index sample_count mismatch")
    ordered_records = tuple(
        sorted(
            enumerate(dataset.records),
            key=lambda item: item[1].sample_id,
        )
    )
    if len(raw_rows) != len(ordered_records):
        raise ValueError("D_V base cache index is not the exact D_V membership")

    rows: list[LoadedDVCacheRow] = []
    image_catalog: list[dict[str, str]] = []
    gt_catalog: list[dict[str, str]] = []
    seen_cache_paths: set[Path] = set()
    for raw_row, (dataset_index, record) in zip(
        raw_rows,
        ordered_records,
        strict=True,
    ):
        if not isinstance(raw_row, dict) or set(raw_row) != _BASE_INDEX_RECORD_KEYS:
            raise ValueError("D_V base cache index record has invalid fields")
        if raw_row["sample_id"] != record.sample_id or raw_row["split"] != "D_V":
            raise ValueError("D_V base cache index is not the exact D_V membership")
        image_path = _record_image_path(record.image, manifest_path)
        image_sha256 = file_sha256(image_path)
        if (
            raw_row["image_path"] != str(image_path)
            or raw_row["image_sha256"] != image_sha256
        ):
            raise ValueError("D_V base cache index image binding mismatch")
        mask_path = _record_mask_path(record.mask, manifest_path)
        mask_sha256 = file_sha256(mask_path)
        cache_path = _bound_index_file(
            index_path.parent,
            raw_row["cache_path"],
            name=f"D_V base cache for {record.sample_id!r}",
        )
        if cache_path in seen_cache_paths:
            raise ValueError("D_V base cache index reuses one cache file")
        seen_cache_paths.add(cache_path)
        cache_sha256 = file_sha256(cache_path)
        if raw_row["cache_sha256"] != cache_sha256:
            raise ValueError("D_V base cache file SHA256 mismatch")
        probability_shape = _validated_shape(
            raw_row["probability_shape"],
            name="D_V base probability shape",
        )
        feature_shape = _validated_shape(
            raw_row["feature_shape"],
            name="D_V base feature shape",
        )

        sample = dataset[dataset_index]
        if not isinstance(sample, LoadedSample):
            raise TypeError("ManifestImageDataset must return LoadedSample records")
        if sample.sample_id != record.sample_id or sample.split != "D_V":
            raise RuntimeError("loaded D_V sample differs from its manifest record")
        if (
            Path(sample.image_path).resolve(strict=True) != image_path
            or Path(sample.mask_path).resolve(strict=True) != mask_path
        ):
            raise RuntimeError("loaded D_V asset path differs from the manifest")
        if (
            file_sha256(image_path) != image_sha256
            or file_sha256(mask_path) != mask_sha256
            or file_sha256(cache_path) != cache_sha256
        ):
            raise RuntimeError("D_V source changed while it was being loaded")
        base_output = load_base_cache(
            cache_path,
            expected_fingerprint=base_fingerprint,
            expected_sample_id=record.sample_id,
            expected_image_fingerprint=image_sha256,
        )
        if (
            list(base_output.probability.shape) != probability_shape
            or list(base_output.feature.shape) != feature_shape
        ):
            raise ValueError("D_V base cache tensor shape differs from its index")
        gt_mask = sample.gt_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
        if (
            gt_mask.ndim != 3
            or gt_mask.shape[0] != 1
            or base_output.probability.shape[-2:] != gt_mask.shape[-2:]
        ):
            raise ValueError("D_V loaded GT and base probability grids differ")
        content_fingerprint = _loaded_d_v_content_fingerprint(
            record.sample_id,
            base_output,
            gt_mask,
        )
        rows.append(
            LoadedDVCacheRow(
                sample_id=record.sample_id,
                base_output=base_output,
                gt_mask=gt_mask,
                image_path=image_path,
                mask_path=mask_path,
                base_cache_path=cache_path,
                image_sha256=image_sha256,
                mask_sha256=mask_sha256,
                base_cache_sha256=cache_sha256,
                content_fingerprint=content_fingerprint,
            )
        )
        image_catalog.append(
            {"sample_id": record.sample_id, "image_sha256": image_sha256}
        )
        gt_catalog.append(
            {"sample_id": record.sample_id, "mask_sha256": mask_sha256}
        )

    d_v_image_fingerprint = stable_fingerprint(
        {
            "schema_version": "cure-lite-d-v-image-catalog-v1",
            "split_manifest_fingerprint": manifest_fingerprint,
            "split": "D_V",
            "records": image_catalog,
        }
    )
    d_v_gt_fingerprint = stable_fingerprint(
        {
            "schema_version": "cure-lite-d-v-gt-catalog-v1",
            "split_manifest_fingerprint": manifest_fingerprint,
            "split": "D_V",
            "records": gt_catalog,
        }
    )
    if file_sha256(index_path) != index_sha256:
        raise RuntimeError("D_V base cache index changed while loading")
    if file_sha256(manifest_path) != manifest_file_sha256:
        raise RuntimeError("split manifest changed while loading D_V")
    for row in rows:
        for path, expected_sha256, label in (
            (row.image_path, row.image_sha256, "D_V image"),
            (row.mask_path, row.mask_sha256, "D_V mask"),
            (row.base_cache_path, row.base_cache_sha256, "D_V base cache"),
        ):
            if file_sha256(path) != expected_sha256:
                raise RuntimeError(f"{label} changed while loading")

    rows_tuple = tuple(rows)
    seal = _LoadedBundleSeal(
        kind="D_V",
        bound_objects=(rows_tuple,),
        bound_values=(
            "D_V",
            manifest_path,
            index_path,
            manifest_fingerprint,
            manifest_file_sha256,
            preprocessing_fingerprint,
            base_fingerprint,
            index_fingerprint,
            index_sha256,
            d_v_image_fingerprint,
            d_v_gt_fingerprint,
        ),
    )
    bundle = LoadedDVCacheBundle(
        split="D_V",
        rows=rows_tuple,
        manifest_path=manifest_path,
        base_index_path=index_path,
        split_manifest_fingerprint=manifest_fingerprint,
        split_manifest_file_sha256=manifest_file_sha256,
        preprocessing_fingerprint=preprocessing_fingerprint,
        base_fingerprint=base_fingerprint,
        base_index_fingerprint=index_fingerprint,
        base_index_sha256=index_sha256,
        d_v_image_fingerprint=d_v_image_fingerprint,
        d_v_gt_fingerprint=d_v_gt_fingerprint,
        _verification_token=seal,
    )
    bundle.verify_unchanged()
    return bundle


__all__ = [
    "BaseCachePairContract",
    "cache_d_r_states",
    "cache_manifest_split",
    "load_base_cache_pair_contract",
    "load_d_r_cache_bundle",
    "load_d_v_cache_bundle",
    "materialize_base_cache_bundle",
]
