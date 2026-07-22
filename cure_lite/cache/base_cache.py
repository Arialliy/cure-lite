"""Safetensors cache for detached frozen-base probabilities and features."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile

import torch

from ..types import FrozenBaseOutput
from .schema import (
    BASE_CACHE_SCHEMA,
    CacheIntegrityError,
    require_fingerprint,
)


def _optional_sha256(value: str | None, *, name: str) -> str:
    if value is None:
        return ""
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a 64-character SHA256 digest")
    return normalized


def _validate_output(output: FrozenBaseOutput) -> None:
    probability = output.probability
    feature = output.feature
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise CacheIntegrityError("probability must be [B,1,H,W]")
    if feature.ndim != 4 or feature.shape[0] != probability.shape[0]:
        raise CacheIntegrityError("feature must be [B,C,h,w] with the same batch")
    if probability.dtype != torch.float32:
        raise CacheIntegrityError("probability must be float32")
    if not feature.is_floating_point():
        raise CacheIntegrityError("feature must be floating point")
    if probability.requires_grad or feature.requires_grad:
        raise CacheIntegrityError("base cache accepts detached tensors only")
    if not torch.isfinite(probability).all() or not torch.isfinite(feature).all():
        raise CacheIntegrityError("base output contains non-finite values")
    if torch.any((probability < 0) | (probability > 1)):
        raise CacheIntegrityError("probability must lie in [0,1]")


def _tensor_content_fingerprint(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_save_file(tensors: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - dependency failure is explicit
        raise RuntimeError("safetensors is required for base caches") from error

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_base_cache(
    path: str | Path,
    output: FrozenBaseOutput,
    *,
    fingerprint: str,
    sample_id: str,
    image_fingerprint: str | None = None,
) -> None:
    """Atomically persist one base-output record with its frozen fingerprint."""

    if not sample_id:
        raise ValueError("sample_id must be non-empty")
    require_fingerprint(fingerprint, fingerprint, cache_kind="base")
    _validate_output(output)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    probability = output.probability.detach().to(device="cpu", dtype=torch.float32).contiguous()
    feature = output.feature.detach().to(device="cpu").contiguous()
    metadata = {
        "schema_version": BASE_CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "sample_id": sample_id,
        "image_fingerprint": _optional_sha256(
            image_fingerprint, name="image_fingerprint"
        ),
        "probability_shape": json.dumps(list(probability.shape), separators=(",", ":")),
        "feature_shape": json.dumps(list(feature.shape), separators=(",", ":")),
        "content_fingerprint": _tensor_content_fingerprint(
            {"probability": probability, "feature": feature}
        ),
    }
    _atomic_save_file(
        {"probability": probability, "feature": feature}, target, metadata
    )


def load_base_cache(
    path: str | Path,
    *,
    expected_fingerprint: str,
    expected_sample_id: str | None = None,
    expected_image_fingerprint: str | None = None,
) -> FrozenBaseOutput:
    """Load a base record and hard-fail on any schema, ID, or hash mismatch."""

    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - dependency failure is explicit
        raise RuntimeError("safetensors is required for base caches") from error

    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = frozenset(handle.keys())
    if metadata.get("schema_version") != BASE_CACHE_SCHEMA:
        raise CacheIntegrityError("unsupported or missing base-cache schema")
    require_fingerprint(
        metadata.get("fingerprint"), expected_fingerprint, cache_kind="base"
    )
    if expected_sample_id is not None and metadata.get("sample_id") != expected_sample_id:
        raise CacheIntegrityError(
            f"base cache sample mismatch: expected {expected_sample_id!r}, "
            f"got {metadata.get('sample_id')!r}"
        )
    if (
        expected_image_fingerprint is not None
        and metadata.get("image_fingerprint")
        != _optional_sha256(
            expected_image_fingerprint, name="expected_image_fingerprint"
        )
    ):
        raise CacheIntegrityError("base cache image fingerprint mismatch")
    if keys != {"probability", "feature"}:
        raise CacheIntegrityError(f"unexpected base-cache tensors: {sorted(keys)}")

    tensors = load_file(str(source), device="cpu")
    output = FrozenBaseOutput(
        probability=tensors["probability"].to(torch.float32),
        feature=tensors["feature"],
    )
    _validate_output(output)
    if metadata.get("content_fingerprint") != _tensor_content_fingerprint(
        {"probability": output.probability, "feature": output.feature}
    ):
        raise CacheIntegrityError("base-cache tensor content fingerprint mismatch")
    try:
        probability_shape = tuple(json.loads(metadata["probability_shape"]))
        feature_shape = tuple(json.loads(metadata["feature_shape"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CacheIntegrityError("invalid base-cache shape metadata") from error
    if probability_shape != tuple(output.probability.shape):
        raise CacheIntegrityError("base-cache probability shape metadata mismatch")
    if feature_shape != tuple(output.feature.shape):
        raise CacheIntegrityError("base-cache feature shape metadata mismatch")
    return output
