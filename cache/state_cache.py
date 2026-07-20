"""NPZ cache for deterministic, pre-sampling CURE-Lite method state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import Tensor

from .schema import STATE_CACHE_SCHEMA, CacheIntegrityError, require_fingerprint


def _labels(value: Tensor, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.int64, device="cpu")
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise CacheIntegrityError(f"{name} must be [H,W] or [1,H,W]")
    if torch.any(tensor < 0):
        raise CacheIntegrityError(f"{name} cannot contain negative IDs")
    return tensor.contiguous()


def _occupancy(value: Tensor) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.bool, device="cpu")
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise CacheIntegrityError("occupancy must be [H,W] or [1,H,W]")
    return tensor.contiguous()


def _image_valid_mask(value: Tensor) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.bool, device="cpu")
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise CacheIntegrityError(
            "image_valid_mask must be [H,W] or [1,H,W]"
        )
    if not torch.any(tensor):
        raise CacheIntegrityError("image_valid_mask cannot be empty")
    return tensor.contiguous()


def _pairs(value: Tensor, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.int64, device="cpu")
    if tensor.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64)
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise CacheIntegrityError(f"{name} must be [N,2] ordered as (gt_id,pred_id)")
    if torch.any(tensor <= 0):
        raise CacheIntegrityError(f"{name} IDs must be positive")
    rows = [tuple(int(item) for item in row) for row in tensor.tolist()]
    if len(rows) != len(set(rows)):
        raise CacheIntegrityError(f"{name} contains duplicate pairs")
    if rows != sorted(rows):
        raise CacheIntegrityError(f"{name} must be sorted by (gt_id,pred_id)")
    return tensor.contiguous()


def _ids(value: Tensor, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.int64, device="cpu").reshape(-1)
    if torch.any(tensor <= 0):
        raise CacheIntegrityError(f"{name} IDs must be positive")
    values = [int(item) for item in tensor.tolist()]
    if len(values) != len(set(values)):
        raise CacheIntegrityError(f"{name} contains duplicate IDs")
    if values != sorted(values):
        raise CacheIntegrityError(f"{name} IDs must be sorted")
    return tensor.contiguous()


@dataclass(frozen=True)
class StateCacheRecord:
    """Source state from which epoch-specific synthetic supervision is built.

    Legal candidates are cached, while the deterministic uniform choice remains
    an online function of ``sample_id``, epoch, and global seed.
    """

    sample_id: str
    occupancy: Tensor
    pred_labels: Tensor
    gt_labels: Tensor
    base_match_pairs: Tensor
    real_miss_ids: Tensor
    reachable_miss_ids: Tensor
    legal_pairs: Tensor
    image_valid_mask: Tensor

    @property
    def unmatched_gt_ids(self) -> Tensor:
        """Read-only semantic alias for callers that consume a match complement."""

        return self.real_miss_ids

    def normalized(self) -> "StateCacheRecord":
        if not self.sample_id:
            raise CacheIntegrityError("sample_id must be non-empty")
        occupancy = _occupancy(self.occupancy)
        image_valid_mask = _image_valid_mask(self.image_valid_mask)
        pred_labels = _labels(self.pred_labels, name="pred_labels")
        gt_labels = _labels(self.gt_labels, name="gt_labels")
        if not (
            occupancy.shape
            == image_valid_mask.shape
            == pred_labels.shape
            == gt_labels.shape
        ):
            raise CacheIntegrityError(
                "occupancy, image-valid, prediction, and GT grids must match"
            )
        if not torch.equal(occupancy, pred_labels > 0):
            raise CacheIntegrityError("occupancy must exactly equal pred_labels > 0")
        if torch.any(occupancy & ~image_valid_mask):
            raise CacheIntegrityError("occupancy extends outside image_valid_mask")
        if torch.any((pred_labels > 0) & ~image_valid_mask):
            raise CacheIntegrityError("pred_labels extend outside image_valid_mask")
        if torch.any((gt_labels > 0) & ~image_valid_mask):
            raise CacheIntegrityError("gt_labels extend outside image_valid_mask")

        match_pairs = _pairs(self.base_match_pairs, name="base_match_pairs")
        legal_pairs = _pairs(self.legal_pairs, name="legal_pairs")
        real_misses = _ids(self.real_miss_ids, name="real_miss_ids")
        reachable_misses = _ids(
            self.reachable_miss_ids, name="reachable_miss_ids"
        )
        pred_ids = set(int(item) for item in torch.unique(pred_labels).tolist()) - {0}
        gt_ids = set(int(item) for item in torch.unique(gt_labels).tolist()) - {0}
        match_rows = set(tuple(int(item) for item in row) for row in match_pairs.tolist())
        if len({row[0] for row in match_rows}) != len(match_rows) or len(
            {row[1] for row in match_rows}
        ) != len(match_rows):
            raise CacheIntegrityError("base_match_pairs are not one-to-one")
        if any(gt_id not in gt_ids or pred_id not in pred_ids for gt_id, pred_id in match_rows):
            raise CacheIntegrityError("base_match_pairs reference unknown component IDs")
        real_miss_set = set(int(item) for item in real_misses.tolist())
        reachable_miss_set = set(int(item) for item in reachable_misses.tolist())
        if not real_miss_set <= gt_ids:
            raise CacheIntegrityError("real_miss_ids reference unknown GT IDs")
        matched_gt_ids = {row[0] for row in match_rows}
        if real_miss_set != gt_ids - matched_gt_ids:
            raise CacheIntegrityError(
                "real_miss_ids must equal the GT complement of base matches"
            )
        if not reachable_miss_set <= real_miss_set:
            raise CacheIntegrityError(
                "reachable_miss_ids must be a subset of real_miss_ids"
            )
        legal_rows = set(tuple(int(item) for item in row) for row in legal_pairs.tolist())
        if not legal_rows <= match_rows:
            raise CacheIntegrityError("legal_pairs must be a subset of base match pairs")

        return StateCacheRecord(
            sample_id=self.sample_id,
            occupancy=occupancy,
            pred_labels=pred_labels,
            gt_labels=gt_labels,
            base_match_pairs=match_pairs,
            real_miss_ids=real_misses,
            reachable_miss_ids=reachable_misses,
            legal_pairs=legal_pairs,
            image_valid_mask=image_valid_mask,
        )


def _content_fingerprint(state: StateCacheRecord) -> str:
    digest = hashlib.sha256()
    digest.update(state.sample_id.encode("utf-8"))
    tensors = {
        "occupancy": state.occupancy.to(torch.uint8),
        "pred_labels": state.pred_labels,
        "gt_labels": state.gt_labels,
        "base_match_pairs": state.base_match_pairs,
        "real_miss_ids": state.real_miss_ids,
        "reachable_miss_ids": state.reachable_miss_ids,
        "legal_pairs": state.legal_pairs,
        "image_valid_mask": state.image_valid_mask.to(torch.uint8),
    }
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_state_cache(
    path: str | Path,
    state: StateCacheRecord,
    *,
    fingerprint: str,
) -> None:
    """Atomically save one source-state record with a frozen state hash."""

    require_fingerprint(fingerprint, fingerprint, cache_kind="state")
    normalized = state.normalized()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": STATE_CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "sample_id": normalized.sample_id,
        "content_fingerprint": _content_fingerprint(normalized),
    }
    _atomic_save_npz(
        target,
        occupancy=normalized.occupancy.to(torch.uint8).numpy(),
        pred_labels=normalized.pred_labels.numpy(),
        gt_labels=normalized.gt_labels.numpy(),
        base_match_pairs=normalized.base_match_pairs.numpy(),
        real_miss_ids=normalized.real_miss_ids.numpy(),
        reachable_miss_ids=normalized.reachable_miss_ids.numpy(),
        legal_pairs=normalized.legal_pairs.numpy(),
        image_valid_mask=normalized.image_valid_mask.to(torch.uint8).numpy(),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )


def load_state_cache(
    path: str | Path,
    *,
    expected_fingerprint: str,
    expected_sample_id: str | None = None,
) -> StateCacheRecord:
    """Load a source state and hard-fail on schema, ID, hash, or content mismatch."""

    expected_keys = {
        "occupancy",
        "pred_labels",
        "gt_labels",
        "base_match_pairs",
        "real_miss_ids",
        "reachable_miss_ids",
        "legal_pairs",
        "image_valid_mask",
        "metadata",
    }
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            if set(data.files) != expected_keys:
                raise CacheIntegrityError(
                    f"unexpected state-cache arrays: {sorted(data.files)}"
                )
            try:
                metadata = json.loads(str(data["metadata"].item()))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise CacheIntegrityError("invalid state-cache metadata") from error
            arrays = {name: np.array(data[name], copy=True) for name in expected_keys - {"metadata"}}
    except (OSError, ValueError) as error:
        if isinstance(error, CacheIntegrityError):
            raise
        raise CacheIntegrityError("unable to read state cache") from error

    if metadata.get("schema_version") != STATE_CACHE_SCHEMA:
        raise CacheIntegrityError("unsupported or missing state-cache schema")
    require_fingerprint(
        metadata.get("fingerprint"), expected_fingerprint, cache_kind="state"
    )
    sample_id = metadata.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise CacheIntegrityError("state cache has invalid sample_id")
    if expected_sample_id is not None and sample_id != expected_sample_id:
        raise CacheIntegrityError(
            f"state cache sample mismatch: expected {expected_sample_id!r}, got {sample_id!r}"
        )

    result = StateCacheRecord(
        sample_id=sample_id,
        occupancy=torch.from_numpy(arrays["occupancy"]).to(torch.bool),
        pred_labels=torch.from_numpy(arrays["pred_labels"]).to(torch.int64),
        gt_labels=torch.from_numpy(arrays["gt_labels"]).to(torch.int64),
        base_match_pairs=torch.from_numpy(arrays["base_match_pairs"]).to(torch.int64),
        real_miss_ids=torch.from_numpy(arrays["real_miss_ids"]).to(torch.int64),
        reachable_miss_ids=torch.from_numpy(arrays["reachable_miss_ids"]).to(
            torch.int64
        ),
        legal_pairs=torch.from_numpy(arrays["legal_pairs"]).to(torch.int64),
        image_valid_mask=torch.from_numpy(arrays["image_valid_mask"]).to(
            torch.bool
        ),
    ).normalized()
    if metadata.get("content_fingerprint") != _content_fingerprint(result):
        raise CacheIntegrityError("state-cache tensor content fingerprint mismatch")
    return result
