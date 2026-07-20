"""Canonical fingerprints and hard cache-mismatch errors."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping


BASE_CACHE_SCHEMA = "cure-lite-base-cache-v2"
STATE_CACHE_SCHEMA = "cure-lite-state-cache-v2"


class CacheFingerprintError(RuntimeError):
    """Raised when cached data does not belong to the requested experiment."""


class CacheIntegrityError(RuntimeError):
    """Raised when cache metadata or tensor contents are malformed."""


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("fingerprint mapping keys must be strings")
            normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("non-finite floats are not canonical fingerprint values")
        return value
    raise TypeError(f"unsupported fingerprint value type: {type(value).__name__}")


def canonical_json(payload: Any) -> str:
    return json.dumps(
        _normalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def stable_fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: str, name: str) -> str:
    value = _required_text(value, name).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 64-character SHA256 digest")
    return value


def _git_object_id(value: str, name: str) -> str:
    value = _required_text(value, name).lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 40-character Git object ID")
    return value


def _schema_version(value: str, expected: str) -> str:
    value = _required_text(value, "schema_version")
    if value != expected:
        raise ValueError(
            f"schema_version must be {expected!r}, got {value!r}"
        )
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    normalized = _normalize(value)
    if not isinstance(normalized, dict):
        raise TypeError(f"{name} must be a mapping or dataclass")
    return normalized


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} must contain exactly {sorted(expected)}, got {sorted(value)}"
        )


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_preprocessing(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "preprocessing")
    expected = {
        "height",
        "width",
        "color_mode",
        "mean",
        "std",
        "image_interpolation",
        "mask_interpolation",
        "range",
    }
    _exact_keys(payload, expected, "preprocessing")
    height = _positive_integer(payload["height"], "preprocessing.height")
    width = _positive_integer(payload["width"], "preprocessing.width")
    color_mode = payload["color_mode"]
    if color_mode not in {"L", "RGB"}:
        raise ValueError("preprocessing.color_mode must be 'L' or 'RGB'")
    channels = 1 if color_mode == "L" else 3
    mean = payload["mean"]
    std = payload["std"]
    if not isinstance(mean, list) or not isinstance(std, list):
        raise TypeError("preprocessing mean/std must be lists")
    if len(mean) != channels or len(std) != channels:
        raise ValueError("preprocessing mean/std do not match color_mode")
    normalized_mean = [
        _finite_number(item, f"preprocessing.mean[{index}]")
        for index, item in enumerate(mean)
    ]
    normalized_std = [
        _finite_number(item, f"preprocessing.std[{index}]")
        for index, item in enumerate(std)
    ]
    if any(item <= 0 for item in normalized_std):
        raise ValueError("preprocessing std values must be positive")
    if payload["image_interpolation"] != "bilinear":
        raise ValueError("preprocessing.image_interpolation must be 'bilinear'")
    if payload["mask_interpolation"] != "nearest":
        raise ValueError("preprocessing.mask_interpolation must be 'nearest'")
    if payload["range"] != "float32-[0,1]-then-normalize":
        raise ValueError("preprocessing.range is not the canonical MSHNet pipeline")
    return {
        "height": height,
        "width": width,
        "color_mode": color_mode,
        "mean": normalized_mean,
        "std": normalized_std,
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "range": "float32-[0,1]-then-normalize",
    }


def _canonical_occupancy_config(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "occupancy_config")
    _exact_keys(
        payload,
        {"threshold", "connectivity", "min_component_area"},
        "occupancy_config",
    )
    threshold = _finite_number(payload["threshold"], "occupancy_config.threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("occupancy_config.threshold must lie in [0,1]")
    connectivity = _positive_integer(
        payload["connectivity"], "occupancy_config.connectivity"
    )
    min_area = _positive_integer(
        payload["min_component_area"], "occupancy_config.min_component_area"
    )
    if connectivity != 8 or min_area != 1:
        raise ValueError("CURE-Lite v0.1 fixes occupancy connectivity/area at 8/1")
    return {
        "threshold": threshold,
        "connectivity": connectivity,
        "min_component_area": min_area,
    }


def _canonical_matching_config(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "matching_config")
    _exact_keys(
        payload,
        {"max_distance", "distance_quantization", "iou_quantization"},
        "matching_config",
    )
    max_distance = _finite_number(
        payload["max_distance"], "matching_config.max_distance"
    )
    if max_distance <= 0:
        raise ValueError("matching_config.max_distance must be positive")
    return {
        "max_distance": max_distance,
        "distance_quantization": _positive_integer(
            payload["distance_quantization"],
            "matching_config.distance_quantization",
        ),
        "iou_quantization": _positive_integer(
            payload["iou_quantization"], "matching_config.iou_quantization"
        ),
    }


def _canonical_intervention_config(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "intervention_config")
    _exact_keys(payload, {"min_writable_pixels"}, "intervention_config")
    minimum = _positive_integer(
        payload["min_writable_pixels"], "intervention_config.min_writable_pixels"
    )
    if minimum != 1:
        raise ValueError("CURE-Lite v0.1 fixes min_writable_pixels at 1")
    return {"min_writable_pixels": minimum}


def build_base_fingerprint(
    *,
    schema_version: str,
    checkpoint_sha256: str,
    adapter_version: str,
    upstream_commit: str,
    upstream_tree: str,
    model_source_sha256: str,
    base_training_provenance_fingerprint: str,
    base_training_final_receipt_sha256: str,
    preprocessing: Any,
    preprocessing_fingerprint: str,
    feature_module_name: str,
    feature_channels: int,
    feature_stride: int,
    forward_kwargs: Mapping[str, Any],
    output_selector: str,
) -> str:
    """Build the base hash from every dependency required by the specification."""

    preprocessing_payload = _canonical_preprocessing(preprocessing)
    preprocessing_digest = _sha256(
        preprocessing_fingerprint, "preprocessing_fingerprint"
    )
    if preprocessing_digest != stable_fingerprint(preprocessing_payload):
        raise ValueError(
            "preprocessing_fingerprint does not match the canonical preprocessing"
        )
    payload = {
        "schema_version": _schema_version(schema_version, BASE_CACHE_SCHEMA),
        "checkpoint_sha256": _sha256(checkpoint_sha256, "checkpoint_sha256"),
        "adapter_version": _required_text(adapter_version, "adapter_version"),
        "upstream_commit": _git_object_id(upstream_commit, "upstream_commit"),
        "upstream_tree": _git_object_id(upstream_tree, "upstream_tree"),
        "model_source_sha256": _sha256(
            model_source_sha256, "model_source_sha256"
        ),
        "base_training_provenance_fingerprint": _sha256(
            base_training_provenance_fingerprint,
            "base_training_provenance_fingerprint",
        ),
        "base_training_final_receipt_sha256": _sha256(
            base_training_final_receipt_sha256,
            "base_training_final_receipt_sha256",
        ),
        "preprocessing": preprocessing_payload,
        "preprocessing_fingerprint": preprocessing_digest,
        "feature_module_name": _required_text(
            feature_module_name, "feature_module_name"
        ),
        "feature_channels": _positive_integer(
            feature_channels, "feature_channels"
        ),
        "feature_stride": _positive_integer(feature_stride, "feature_stride"),
        "forward_kwargs": _mapping(forward_kwargs, "forward_kwargs"),
        "output_selector": _required_text(output_selector, "output_selector"),
    }
    return stable_fingerprint(payload)


def build_state_fingerprint(
    *,
    schema_version: str,
    base_fingerprint: str,
    split_manifest_sha256: str,
    gt_fingerprint: str,
    occupancy_config: Any,
    matching_config: Any,
    intervention_config: Any,
) -> str:
    """Build the method-state hash from the frozen base, data, GT, and configs."""

    payload = {
        "schema_version": _schema_version(schema_version, STATE_CACHE_SCHEMA),
        "base_fingerprint": _sha256(base_fingerprint, "base_fingerprint"),
        "split_manifest_sha256": _sha256(
            split_manifest_sha256, "split_manifest_sha256"
        ),
        "gt_fingerprint": _sha256(gt_fingerprint, "gt_fingerprint"),
        "occupancy_config": _canonical_occupancy_config(occupancy_config),
        "matching_config": _canonical_matching_config(matching_config),
        "intervention_config": _canonical_intervention_config(
            intervention_config
        ),
    }
    return stable_fingerprint(payload)


def require_fingerprint(actual: str | None, expected: str, *, cache_kind: str) -> None:
    expected = _sha256(expected, "expected_fingerprint")
    if actual != expected:
        raise CacheFingerprintError(
            f"{cache_kind} cache fingerprint mismatch: expected {expected}, got {actual!r}"
        )
