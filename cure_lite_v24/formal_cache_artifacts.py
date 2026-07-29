"""Mechanical verification for physically independent Formal800 caches.

The protocol layer must not accept caller-authored ``is_reflink`` or tensor
storage booleans.  This module therefore verifies the regular file itself,
queries Linux FIEMAP, loads the neutral tensor envelope with
``weights_only=True`` and ``mmap=False``, and issues private-issuer tokens.
The pair verifier keeps both loads alive simultaneously while comparing real
storage addresses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
from typing import Final, Iterator, Mapping

import torch
from torch import Tensor

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.paired_types import tensor_content_fingerprint


FORMAL_CACHE_PAYLOAD_SCHEMA: Final = (
    "cure-lite-v24-formal-cache-neutral-tensor-envelope-v3"
)
FORMAL_CACHE_VERIFICATION_SCHEMA: Final = (
    "cure-lite-v24-formal-cache-artifact-verification-v1"
)
FORMAL_CACHE_PAIR_SCHEMA: Final = (
    "cure-lite-v24-formal-cache-physical-independence-v1"
)

_HEX = frozenset("0123456789abcdef")
_TOKEN_ISSUER = object()
_MATERIALIZATION_ORIGIN_ISSUER = object()
_TOKEN_REGISTRY: dict[int, object] = {}


def _register_token(value: object) -> object:
    """Keep a strong exact-instance record for every issued capability."""

    if getattr(value, "_issuer", None) is not _TOKEN_ISSUER:
        raise AssertionError("attempted to register an unsigned cache token")
    identity = id(value)
    previous = _TOKEN_REGISTRY.get(identity)
    if previous is not None and previous is not value:
        raise RuntimeError("Formal cache token identity collision")
    _TOKEN_REGISTRY[identity] = value
    return value

# Linux uapi values from linux/fs.h and linux/fiemap.h.
_FS_IOC_FIEMAP = 0xC020660B
_FIEMAP_FLAG_SYNC = 0x00000001
_FIEMAP_EXTENT_LAST = 0x00000001
_FIEMAP_ALLOWED_EXTENT_FLAGS = _FIEMAP_EXTENT_LAST
_FIEMAP_HEADER = struct.Struct("=QQIIII")
_FIEMAP_EXTENT = struct.Struct("=QQQQQIIII")
_FIEMAP_BATCH = 256


def _sha256_text(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path_value: str | Path) -> tuple[Path, os.stat_result]:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("Formal cache path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Formal cache must be a regular non-symlink file")
    path = path.resolve(strict=True)
    stat_result = path.stat()
    if stat_result.st_nlink != 1 or stat_result.st_size < 1:
        raise PermissionError(
            "Formal cache must be nonempty with exactly one hard link"
        )
    return path, stat_result


def _fiemap_flags(path: Path) -> tuple[int, ...]:
    """Return every allocated extent flag, failing closed on uncertainty."""

    open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    descriptor = os.open(path, open_flags)
    try:
        opened = os.fstat(descriptor)
        expected = path.stat()
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
        ) != (
            expected.st_dev,
            expected.st_ino,
            expected.st_nlink,
            expected.st_size,
        ):
            raise RuntimeError("Formal cache changed before FIEMAP audit")
        start = 0
        flags: list[int] = []
        saw_last = False
        while not saw_last:
            remaining = (1 << 64) - 1 - start
            buffer = bytearray(
                _FIEMAP_HEADER.size
                + _FIEMAP_BATCH * _FIEMAP_EXTENT.size
            )
            _FIEMAP_HEADER.pack_into(
                buffer,
                0,
                start,
                remaining,
                _FIEMAP_FLAG_SYNC,
                0,
                _FIEMAP_BATCH,
                0,
            )
            try:
                fcntl.ioctl(descriptor, _FS_IOC_FIEMAP, buffer, True)
            except OSError as error:
                raise RuntimeError(
                    "filesystem cannot provide a fail-closed FIEMAP audit"
                ) from error
            mapped = struct.unpack_from("=I", buffer, 20)[0]
            if mapped < 1 or mapped > _FIEMAP_BATCH:
                raise RuntimeError("FIEMAP returned an incomplete extent set")
            next_start = start
            for index in range(mapped):
                values = _FIEMAP_EXTENT.unpack_from(
                    buffer,
                    _FIEMAP_HEADER.size
                    + index * _FIEMAP_EXTENT.size,
                )
                logical = int(values[0])
                length = int(values[2])
                extent_flags = int(values[5])
                if (
                    length < 1
                    or extent_flags & ~_FIEMAP_ALLOWED_EXTENT_FLAGS
                ):
                    raise PermissionError(
                        "Formal cache has a physical extent flag outside "
                        "the fail-closed {0,LAST} whitelist"
                    )
                flags.append(extent_flags)
                next_start = max(next_start, logical + length)
                saw_last = bool(extent_flags & _FIEMAP_EXTENT_LAST)
                if saw_last and index != mapped - 1:
                    raise RuntimeError("FIEMAP LAST flag is not terminal")
            if not saw_last:
                if next_start <= start:
                    raise RuntimeError("FIEMAP extent traversal did not advance")
                start = next_start
        return tuple(flags)
    finally:
        os.close(descriptor)


def _walk_tensors(
    value: object,
    *,
    path: str,
) -> Iterator[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield path, value
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Formal cache envelope mapping keys must be strings")
        for key in sorted(value):
            yield from _walk_tensors(value[key], path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_tensors(item, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    raise TypeError(
        "Formal cache envelope contains a non-neutral object: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _normalized_envelope(value: object, *, path: str) -> object:
    if isinstance(value, Tensor):
        if (
            value.device.type != "cpu"
            or value.layout != torch.strided
            or value.requires_grad
            or not value.is_contiguous()
            or value.numel() < 1
        ):
            raise ValueError(
                "Formal cache tensors must be nonempty detached contiguous "
                "CPU strided tensors"
            )
        return {
            "tensor_path": path,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "content_fingerprint": tensor_content_fingerprint(value),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Formal cache envelope mapping keys must be strings")
        return {
            key: _normalized_envelope(value[key], path=f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [
            _normalized_envelope(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return {
            "tuple": [
                _normalized_envelope(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        }
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value)).item():
            raise ValueError("Formal cache envelope contains a non-finite float")
        return {"float_hex": value.hex()}
    raise TypeError(
        "Formal cache envelope contains a non-neutral object: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _cache_object_graph(
    value: object,
    *,
    tensor_paths: Mapping[int, str],
) -> object:
    """Encode a cache using primitives plus references into ``tensors``."""

    if isinstance(value, Tensor):
        logical_path = tensor_paths.get(id(value))
        if logical_path is None:
            raise RuntimeError("cache tensor lacks a neutral tensor reference")
        return {"kind": "tensor_reference", "logical_path": logical_path}
    if isinstance(value, Enum):
        type_name = (
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
        if type_name not in _cache_enum_types():
            raise TypeError(f"cache object graph rejects enum {type_name}")
        return {
            "kind": "enum",
            "type": type_name,
            "value": value.value,
        }
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value)).item():
            raise ValueError("cache object graph contains non-finite float")
        return {"kind": "finite_float_hex", "value": value.hex()}
    if isinstance(value, Path):
        return {"kind": "path_text", "value": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        type_name = (
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
        if type_name not in _cache_dataclass_types():
            raise TypeError(
                f"cache object graph rejects dataclass {type_name}"
            )
        return {
            "kind": "dataclass",
            "type": type_name,
            "fields": {
                definition.name: _cache_object_graph(
                    getattr(value, definition.name),
                    tensor_paths=tensor_paths,
                )
                for definition in fields(value)
            },
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("cache object graph mappings require text keys")
        return {
            "kind": "mapping",
            "items": {
                key: _cache_object_graph(
                    value[key],
                    tensor_paths=tensor_paths,
                )
                for key in sorted(value)
            },
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _cache_object_graph(item, tensor_paths=tensor_paths)
                for item in value
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _cache_object_graph(item, tensor_paths=tensor_paths)
                for item in value
            ],
        }
    raise TypeError(
        "cache object graph rejects "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _cache_dataclass_types() -> dict[str, type]:
    """Return the fixed constructor whitelist for scalar-cache rebuilding."""

    from cure_lite.coverage_state_observability import (
        CoverageStatePairObservabilityAudit,
        CoverageStatePopulationObservabilityReceipt,
        CoverageStateRepresentationAudit,
    )
    from cure_lite.coverage_state_precomputed_cache import (
        CoverageStateCachedNatural,
        CoverageStateCachedPair,
        CoverageStateScalarCache,
    )
    from cure_lite.coverage_state_raw_catalog import (
        CoverageStateNaturalRecord,
        CoverageStatePairRecord,
        CoverageStateRawCatalog,
    )
    from cure_lite.coverage_state_sobolev import (
        CoverageStateAbsoluteTargets,
        CoverageStatePairTargets,
        CoverageStateSobolevConfig,
    )

    values = (
        CoverageStatePairObservabilityAudit,
        CoverageStatePopulationObservabilityReceipt,
        CoverageStateRepresentationAudit,
        CoverageStateCachedNatural,
        CoverageStateCachedPair,
        CoverageStateScalarCache,
        CoverageStateNaturalRecord,
        CoverageStatePairRecord,
        CoverageStateRawCatalog,
        CoverageStateAbsoluteTargets,
        CoverageStatePairTargets,
        CoverageStateSobolevConfig,
    )
    return {
        f"{value.__module__}.{value.__qualname__}": value
        for value in values
    }


def _cache_enum_types() -> dict[str, type[Enum]]:
    from cure_lite.coverage_state_observability import (
        CoverageStateObservabilityDecision,
    )

    values = (CoverageStateObservabilityDecision,)
    return {
        f"{value.__module__}.{value.__qualname__}": value
        for value in values
    }


def _decode_cache_object_graph(
    value: object,
    *,
    tensors: Mapping[str, Tensor],
) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("cache object graph node must be a mapping")
    kind = value.get("kind")
    if kind == "enum":
        if set(value) != {"kind", "type", "value"}:
            raise ValueError("enum node schema changed")
        type_name = value.get("type")
        types = _cache_enum_types()
        if not isinstance(type_name, str) or type_name not in types:
            raise PermissionError("cache enum type is not whitelisted")
        return types[type_name](value.get("value"))
    if kind == "tensor_reference":
        if set(value) != {"kind", "logical_path"}:
            raise ValueError("tensor-reference node schema changed")
        logical_path = value.get("logical_path")
        if not isinstance(logical_path, str) or logical_path not in tensors:
            raise ValueError("tensor-reference path is absent")
        return tensors[logical_path]
    if kind == "finite_float_hex":
        if set(value) != {"kind", "value"}:
            raise ValueError("float node schema changed")
        raw = value.get("value")
        if not isinstance(raw, str):
            raise TypeError("float hex value must be text")
        result = float.fromhex(raw)
        if not torch.isfinite(torch.tensor(result)).item():
            raise ValueError("decoded cache float is non-finite")
        return result
    if kind == "path_text":
        if set(value) != {"kind", "value"}:
            raise ValueError("path node schema changed")
        raw = value.get("value")
        if not isinstance(raw, str):
            raise TypeError("path node value must be text")
        return Path(raw)
    if kind in {"tuple", "list"}:
        if set(value) != {"kind", "items"}:
            raise ValueError(f"{kind} node schema changed")
        raw_items = value.get("items")
        if not isinstance(raw_items, list):
            raise TypeError(f"{kind} items must be a list")
        decoded = [
            _decode_cache_object_graph(item, tensors=tensors)
            for item in raw_items
        ]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "mapping":
        if set(value) != {"kind", "items"}:
            raise ValueError("mapping node schema changed")
        raw_items = value.get("items")
        if (
            not isinstance(raw_items, Mapping)
            or any(not isinstance(key, str) for key in raw_items)
        ):
            raise TypeError("mapping node items require text keys")
        return {
            key: _decode_cache_object_graph(
                raw_items[key],
                tensors=tensors,
            )
            for key in sorted(raw_items)
        }
    if kind == "dataclass":
        if set(value) != {"kind", "type", "fields"}:
            raise ValueError("dataclass node schema changed")
        type_name = value.get("type")
        raw_fields = value.get("fields")
        types = _cache_dataclass_types()
        if (
            not isinstance(type_name, str)
            or type_name not in types
            or not isinstance(raw_fields, Mapping)
        ):
            raise PermissionError("cache dataclass type is not whitelisted")
        constructor = types[type_name]
        expected_fields = {definition.name for definition in fields(constructor)}
        if set(raw_fields) != expected_fields:
            raise ValueError("cache dataclass fields changed")
        return constructor(
            **{
                name: _decode_cache_object_graph(
                    raw_fields[name],
                    tensors=tensors,
                )
                for name in sorted(raw_fields)
            }
        )
    raise PermissionError(f"unknown cache object graph kind {kind!r}")


def _load_neutral_cache(
    path: Path,
    *,
    expected_semantic_cache_fingerprint: str,
) -> tuple[object, tuple[tuple[str, Tensor], ...], str]:
    try:
        value = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=False,
        )
    except Exception as error:
        raise RuntimeError(
            "Formal cache failed the fixed weights-only non-mmap loader"
        ) from error
    return _inspect_neutral_cache(
        value,
        expected_semantic_cache_fingerprint=(
            expected_semantic_cache_fingerprint
        ),
    )


def _inspect_neutral_cache(
    value: object,
    *,
    expected_semantic_cache_fingerprint: str,
) -> tuple[object, tuple[tuple[str, Tensor], ...], str]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "semantic_cache_fingerprint",
        "payload",
    }:
        raise ValueError("Formal cache neutral envelope schema changed")
    if (
        value.get("schema_version") != FORMAL_CACHE_PAYLOAD_SCHEMA
        or value.get("semantic_cache_fingerprint")
        != expected_semantic_cache_fingerprint
    ):
        raise ValueError("Formal cache semantic identity changed")
    semantic_fp = _sha256_text(
        value.get("semantic_cache_fingerprint"),
        name="Formal cache semantic_cache_fingerprint",
    )
    payload = value.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {
        "canonical_cache_payload_json",
        "canonical_cache_payload_fingerprint",
        "object_graph",
        "tensor_ledger",
        "tensors",
    }:
        raise ValueError("Formal cache neutral payload schema changed")
    canonical_payload_json = payload.get("canonical_cache_payload_json")
    if not isinstance(canonical_payload_json, str):
        raise TypeError("canonical_cache_payload_json must be text")
    try:
        canonical_payload = json.loads(canonical_payload_json)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "canonical_cache_payload_json is not valid JSON"
        ) from error
    if (
        not isinstance(canonical_payload, dict)
        or canonical_json(canonical_payload) != canonical_payload_json
        or stable_fingerprint(canonical_payload) != semantic_fp
        or _sha256_text(
            payload.get("canonical_cache_payload_fingerprint"),
            name="canonical_cache_payload_fingerprint",
        )
        != semantic_fp
    ):
        raise ValueError(
            "canonical cache payload is not a canonical self-fingerprint "
            "of the declared semantic cache"
        )
    tensor_map = payload.get("tensors")
    if (
        not isinstance(tensor_map, Mapping)
        or not tensor_map
        or any(
            not isinstance(logical_path, str)
            or not logical_path.startswith("cache.")
            or type(tensor) is not Tensor
            for logical_path, tensor in tensor_map.items()
        )
    ):
        raise ValueError(
            "Formal cache tensors must be a nonempty flat cache.* tensor map"
        )
    if not isinstance(payload.get("object_graph"), Mapping):
        raise TypeError("Formal cache object_graph must be a mapping")
    expected_ledger = [
        {
            "logical_path": logical_path,
            "dtype": str(tensor_map[logical_path].dtype),
            "shape": list(tensor_map[logical_path].shape),
            "stride": list(tensor_map[logical_path].stride()),
            "content_fingerprint": tensor_content_fingerprint(
                tensor_map[logical_path]
            ),
        }
        for logical_path in sorted(tensor_map)
    ]
    if payload.get("tensor_ledger") != expected_ledger:
        raise ValueError(
            "tensor_ledger paths/content do not exactly match actual tensors"
        )
    tensors = tuple(
        _walk_tensors(tensor_map, path="payload.tensors")
    )
    if not tensors:
        raise ValueError("Formal cache neutral envelope contains no tensors")
    normalized = _normalized_envelope(value, path="envelope")
    return value, tensors, stable_fingerprint(normalized)


def _cache_tensor_paths(
    value: object,
    *,
    path: str,
    seen_objects: set[int],
) -> Iterator[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield path, value
        return
    if value is None or isinstance(value, (str, int, float, bool, Path)):
        return
    identity = id(value)
    if identity in seen_objects:
        return
    seen_objects.add(identity)
    if is_dataclass(value) and not isinstance(value, type):
        for definition in fields(value):
            yield from _cache_tensor_paths(
                getattr(value, definition.name),
                path=f"{path}.{definition.name}",
                seen_objects=seen_objects,
            )
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("cache mappings must use string keys")
        for key in sorted(value):
            yield from _cache_tensor_paths(
                value[key],
                path=f"{path}.{key}",
                seen_objects=seen_objects,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _cache_tensor_paths(
                item,
                path=f"{path}[{index}]",
                seen_objects=seen_objects,
            )
        return
    # Non-tensor policy/config objects need not be pickled into the neutral
    # artifact: their exact public meaning is already in canonical_payload.
    return


def build_formal_cache_neutral_envelope(
    cache: object,
) -> dict[str, object]:
    """Clone one verified scalar cache into the fixed neutral tensor format."""

    from cure_lite.coverage_state_precomputed_cache import (
        CoverageStateScalarCache,
    )

    if type(cache) is not CoverageStateScalarCache:
        raise TypeError("cache must be exact CoverageStateScalarCache")
    cache.verify_unchanged()
    canonical_payload = cache.canonical_payload()
    semantic_fp = cache.cache_fingerprint
    if stable_fingerprint(canonical_payload) != semantic_fp:
        raise RuntimeError("scalar cache canonical fingerprint changed")
    tensor_rows = tuple(
        _cache_tensor_paths(
            cache,
            path="cache",
            seen_objects=set(),
        )
    )
    if not tensor_rows:
        raise ValueError("scalar cache contains no tensors")
    tensors = {
        logical_path: tensor.detach().cpu().clone().contiguous()
        for logical_path, tensor in tensor_rows
    }
    tensor_paths_by_identity = {
        id(tensor): logical_path
        for logical_path, tensor in tensor_rows
    }
    tensor_ledger = [
        {
            "logical_path": logical_path,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "content_fingerprint": tensor_content_fingerprint(tensor),
        }
        for logical_path, tensor in sorted(tensors.items())
    ]
    envelope = {
        "schema_version": FORMAL_CACHE_PAYLOAD_SCHEMA,
        "semantic_cache_fingerprint": semantic_fp,
        "payload": {
            "canonical_cache_payload_json": canonical_json(
                canonical_payload
            ),
            "canonical_cache_payload_fingerprint": semantic_fp,
            "object_graph": _cache_object_graph(
                cache,
                tensor_paths=tensor_paths_by_identity,
            ),
            "tensor_ledger": tensor_ledger,
            "tensors": tensors,
        },
    }
    _inspect_neutral_cache(
        envelope,
        expected_semantic_cache_fingerprint=semantic_fp,
    )
    return envelope


def formal_cache_neutral_payload_fingerprint(
    envelope: Mapping[str, object],
) -> str:
    """Recompute the composite metadata-and-tensor content fingerprint."""

    if not isinstance(envelope, Mapping):
        raise TypeError("envelope must be a mapping")
    semantic_fp = _sha256_text(
        envelope.get("semantic_cache_fingerprint"),
        name="envelope.semantic_cache_fingerprint",
    )
    _, _, neutral_fp = _inspect_neutral_cache(
        envelope,
        expected_semantic_cache_fingerprint=semantic_fp,
    )
    return neutral_fp


def rebuild_formal_scalar_cache_from_neutral_envelope(
    envelope: Mapping[str, object],
    *,
    expected_semantic_cache_fingerprint: str,
):
    """Rebuild an exact scalar cache from an already safely loaded envelope.

    This is the in-memory counterpart of
    :func:`load_formal_scalar_cache_artifact`.  It deliberately accepts only
    primitive/tensor mappings that have already crossed a
    ``torch.load(..., weights_only=True, mmap=False)`` boundary.  OOF cache
    artifacts reuse this one constructor whitelist instead of maintaining a
    second pickle-capable decoder.
    """

    from cure_lite.coverage_state_precomputed_cache import (
        CoverageStateScalarCache,
    )

    if not isinstance(envelope, Mapping):
        raise TypeError("neutral scalar-cache envelope must be a mapping")
    semantic_fp = _sha256_text(
        expected_semantic_cache_fingerprint,
        name="expected_semantic_cache_fingerprint",
    )
    _, _, _ = _inspect_neutral_cache(
        envelope,
        expected_semantic_cache_fingerprint=semantic_fp,
    )
    payload = envelope.get("payload")
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("tensors"), Mapping)
    ):
        raise TypeError("neutral scalar-cache payload is invalid")
    raw_tensors = payload["tensors"]
    tensors = {
        name: tensor
        for name, tensor in raw_tensors.items()
        if isinstance(name, str) and type(tensor) is Tensor
    }
    if len(tensors) != len(raw_tensors):
        raise TypeError("neutral scalar-cache tensors changed")
    cache = _decode_cache_object_graph(
        payload.get("object_graph"),
        tensors=tensors,
    )
    if (
        type(cache) is not CoverageStateScalarCache
        or cache.cache_fingerprint != semantic_fp
        or stable_fingerprint(cache.canonical_payload()) != semantic_fp
    ):
        raise RuntimeError("rebuilt scalar cache identity changed")
    cache.verify_unchanged()
    return cache


def load_formal_scalar_cache_artifact(
    artifact: VerifiedFormalCacheArtifact,
):
    """Rebuild one exact scalar cache through the neutral safe loader."""

    from cure_lite.coverage_state_precomputed_cache import (
        CoverageStateScalarCache,
    )

    token = require_verified_formal_cache_artifact(artifact)
    reverified = verify_formal_cache_artifact(
        token.path,
        cache_id=token.cache_id,
        expected_semantic_cache_fingerprint=(
            token.semantic_cache_fingerprint
        ),
        expected_neutral_payload_fingerprint=(
            token.neutral_payload_fingerprint
        ),
    )
    if reverified.receipt_fingerprint != token.receipt_fingerprint:
        raise PermissionError("Formal cache token changed before rebuild")
    path, _ = _regular_file(token.path)
    try:
        envelope = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=False,
        )
    except Exception as error:
        raise RuntimeError(
            "Formal cache failed its fixed reconstruction loader"
        ) from error
    if not isinstance(envelope, Mapping):
        raise TypeError("Formal cache reconstruction envelope is invalid")
    _, _, neutral_fp = _inspect_neutral_cache(
        envelope,
        expected_semantic_cache_fingerprint=(
            token.semantic_cache_fingerprint
        ),
    )
    if neutral_fp != token.neutral_payload_fingerprint:
        raise PermissionError("Formal cache reconstruction binding changed")
    return rebuild_formal_scalar_cache_from_neutral_envelope(
        envelope,
        expected_semantic_cache_fingerprint=(
            token.semantic_cache_fingerprint
        ),
    )


def save_formal_cache_neutral_artifact_new(
    cache: object,
    path: str | Path,
    *,
    cache_id: str,
) -> VerifiedFormalCacheArtifact:
    """Create one no-replace artifact, fsync it, then mechanically verify it."""

    envelope = build_formal_cache_neutral_envelope(cache)
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("Formal cache destination must be absolute")
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent or not parent.is_dir():
        raise RuntimeError("Formal cache destination parent is not canonical")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    created = True
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            torch.save(envelope, handle)
            handle.flush()
            os.fsync(handle.fileno())
        created = False
    except BaseException:
        if created:
            try:
                os.unlink(destination)
            except FileNotFoundError:
                pass
        raise
    return _verify_formal_cache_artifact(
        destination,
        cache_id=cache_id,
        expected_semantic_cache_fingerprint=str(
            envelope["semantic_cache_fingerprint"]
        ),
        expected_neutral_payload_fingerprint=None,
        materialization_origin=_MATERIALIZATION_ORIGIN_ISSUER,
    )


def _storage_identities(
    tensors: tuple[tuple[str, Tensor], ...],
) -> tuple[tuple[str, int, int], ...]:
    identities = {
        (
            str(tensor.device),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        )
        for _, tensor in tensors
    }
    if any(pointer < 1 or size < 1 for _, pointer, size in identities):
        raise RuntimeError("Formal cache tensor storage identity is invalid")
    return tuple(sorted(identities))


@dataclass(frozen=True, slots=True)
class VerifiedFormalCacheArtifact:
    payload_json: str
    cache_id: str
    semantic_cache_fingerprint: str
    path: str
    file_sha256: str
    device: int
    inode: int
    hardlink_count: int
    neutral_payload_fingerprint: str
    receipt_fingerprint: str
    _issuer: object = field(repr=False, compare=False)
    _materialization_origin: object | None = field(
        repr=False,
        compare=False,
    )

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise AssertionError("verified Formal cache payload changed")
        return value


@dataclass(frozen=True, slots=True)
class VerifiedFormalCachePair:
    payload_json: str
    pair_fingerprint: str
    seed42_cache_receipt_fingerprint: str
    seed43_cache_receipt_fingerprint: str
    _issuer: object = field(repr=False, compare=False)

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise AssertionError("verified Formal cache-pair payload changed")
        return value


def require_verified_formal_cache_artifact(
    value: object,
) -> VerifiedFormalCacheArtifact:
    if (
        type(value) is not VerifiedFormalCacheArtifact
        or value._issuer is not _TOKEN_ISSUER
        or _TOKEN_REGISTRY.get(id(value)) is not value
    ):
        raise TypeError(
            "cache_artifact must be issued by the fixed Formal cache verifier"
        )
    return value


def require_verified_formal_cache_origin_artifact(
    value: object,
) -> VerifiedFormalCacheArtifact:
    verified = require_verified_formal_cache_artifact(value)
    if verified._materialization_origin is not _MATERIALIZATION_ORIGIN_ISSUER:
        raise TypeError(
            "full_d_r_cache_artifact must be issued by "
            "save_formal_cache_neutral_artifact_new from an exact "
            "CoverageStateScalarCache"
        )
    return verified


def verify_formal_cache_artifact(
    path: str | Path,
    *,
    cache_id: str,
    expected_semantic_cache_fingerprint: str,
    expected_neutral_payload_fingerprint: str | None = None,
) -> VerifiedFormalCacheArtifact:
    """Mechanically verify one non-mmap, non-reflink neutral cache file."""

    return _verify_formal_cache_artifact(
        path,
        cache_id=cache_id,
        expected_semantic_cache_fingerprint=(
            expected_semantic_cache_fingerprint
        ),
        expected_neutral_payload_fingerprint=(
            expected_neutral_payload_fingerprint
        ),
        materialization_origin=None,
    )


def _verify_formal_cache_artifact(
    path: str | Path,
    *,
    cache_id: str,
    expected_semantic_cache_fingerprint: str,
    expected_neutral_payload_fingerprint: str | None,
    materialization_origin: object | None,
) -> VerifiedFormalCacheArtifact:
    """Internal issuer; only the exact-cache save path may set origin."""

    if materialization_origin not in {
        None,
        _MATERIALIZATION_ORIGIN_ISSUER,
    }:
        raise AssertionError("unknown Formal cache materialization origin")
    cache_identity = _text(cache_id, name="cache_id")
    semantic_fp = _sha256_text(
        expected_semantic_cache_fingerprint,
        name="expected_semantic_cache_fingerprint",
    )
    canonical_path, stat_result = _regular_file(path)
    extent_flags = _fiemap_flags(canonical_path)
    _, tensors, neutral_payload_fp = _load_neutral_cache(
        canonical_path,
        expected_semantic_cache_fingerprint=semantic_fp,
    )
    if expected_neutral_payload_fingerprint is not None:
        expected_neutral_fp = _sha256_text(
            expected_neutral_payload_fingerprint,
            name="expected_neutral_payload_fingerprint",
        )
        if neutral_payload_fp != expected_neutral_fp:
            raise PermissionError(
                "Formal cache neutral payload differs from its verified "
                "bounded predecessor"
            )
    storage_identities = _storage_identities(tensors)
    body = {
        "schema_version": FORMAL_CACHE_VERIFICATION_SCHEMA,
        "cache_id": cache_identity,
        "path": str(canonical_path),
        "size_bytes": stat_result.st_size,
        "file_sha256": _file_sha256(canonical_path),
        "semantic_cache_fingerprint": semantic_fp,
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "hardlink_count": stat_result.st_nlink,
        "fiemap": {
            "extent_count": len(extent_flags),
            "extent_flags": list(extent_flags),
            "shared_extent_count": 0,
            "unknown_extent_count": 0,
            "delalloc_extent_count": 0,
            "encoded_extent_count": 0,
        },
        "loader": {
            "implementation": "torch.load",
            "weights_only": True,
            "map_location": "cpu",
            "mmap_used": False,
            "tensor_count": len(tensors),
            "unique_storage_count": len(storage_identities),
            "neutral_payload_fingerprint": neutral_payload_fp,
        },
        "is_symlink": False,
        "is_reflink": False,
        "shared_tensor_storage_with_other_formal_cache": "PAIR_CHECK_PENDING",
        "process_cache_reused": False,
    }
    receipt_fp = stable_fingerprint(body)
    payload = {**body, "receipt_fingerprint": receipt_fp}
    return _register_token(VerifiedFormalCacheArtifact(
        payload_json=canonical_json(payload),
        cache_id=cache_identity,
        semantic_cache_fingerprint=semantic_fp,
        path=str(canonical_path),
        file_sha256=str(body["file_sha256"]),
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        hardlink_count=stat_result.st_nlink,
        neutral_payload_fingerprint=neutral_payload_fp,
        receipt_fingerprint=receipt_fp,
        _issuer=_TOKEN_ISSUER,
        _materialization_origin=materialization_origin,
    ))


def _reverify_token_file(
    value: VerifiedFormalCacheArtifact,
) -> tuple[Path, os.stat_result]:
    path, stat_result = _regular_file(value.path)
    if (
        _file_sha256(path) != value.file_sha256
        or stat_result.st_dev != value.device
        or stat_result.st_ino != value.inode
        or stat_result.st_nlink != value.hardlink_count
        or value.payload.get("receipt_fingerprint")
        != value.receipt_fingerprint
    ):
        raise RuntimeError("verified Formal cache artifact changed")
    _fiemap_flags(path)
    return path, stat_result


def verify_formal_cache_pair_independence(
    seed42_cache: VerifiedFormalCacheArtifact,
    seed43_cache: VerifiedFormalCacheArtifact,
) -> VerifiedFormalCachePair:
    """Load both caches together and compare their actual storage objects."""

    primary = require_verified_formal_cache_artifact(seed42_cache)
    integrity = require_verified_formal_cache_artifact(seed43_cache)
    primary_path, primary_stat = _reverify_token_file(primary)
    integrity_path, integrity_stat = _reverify_token_file(integrity)
    primary_value, primary_tensors, primary_payload_fp = _load_neutral_cache(
        primary_path,
        expected_semantic_cache_fingerprint=(
            primary.semantic_cache_fingerprint
        ),
    )
    integrity_value, integrity_tensors, integrity_payload_fp = (
        _load_neutral_cache(
            integrity_path,
            expected_semantic_cache_fingerprint=(
                integrity.semantic_cache_fingerprint
            ),
        )
    )
    # Keep both loaded object graphs alive through the pointer comparison.
    primary_storage = set(_storage_identities(primary_tensors))
    integrity_storage = set(_storage_identities(integrity_tensors))
    checks = {
        "same_semantic_cache_fingerprint": (
            primary.semantic_cache_fingerprint
            == integrity.semantic_cache_fingerprint
        ),
        "same_neutral_payload_fingerprint": (
            primary.neutral_payload_fingerprint
            == integrity.neutral_payload_fingerprint
            == primary_payload_fp
            == integrity_payload_fp
        ),
        "different_canonical_paths": primary_path != integrity_path,
        "different_device_inode": (
            (primary_stat.st_dev, primary_stat.st_ino)
            != (integrity_stat.st_dev, integrity_stat.st_ino)
        ),
        "both_single_link": (
            primary_stat.st_nlink == integrity_stat.st_nlink == 1
        ),
        "different_loaded_process_objects": (
            primary_value is not integrity_value
        ),
        "actual_loaded_tensor_storages_disjoint": not (
            primary_storage & integrity_storage
        ),
        "both_fixed_non_mmap_loads": True,
        "both_fiemap_no_shared_unknown_delalloc_encoded": True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise PermissionError(
            "Formal cache independence failed: " + ", ".join(failed)
        )
    body = {
        "schema_version": FORMAL_CACHE_PAIR_SCHEMA,
        "seed42_cache_receipt_fingerprint": primary.receipt_fingerprint,
        "seed43_cache_receipt_fingerprint": integrity.receipt_fingerprint,
        "semantic_cache_fingerprint": primary.semantic_cache_fingerprint,
        "checks": checks,
    }
    pair_fp = stable_fingerprint(body)
    payload = {**body, "pair_fingerprint": pair_fp}
    return _register_token(VerifiedFormalCachePair(
        payload_json=canonical_json(payload),
        pair_fingerprint=pair_fp,
        seed42_cache_receipt_fingerprint=primary.receipt_fingerprint,
        seed43_cache_receipt_fingerprint=integrity.receipt_fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


__all__ = [
    "FORMAL_CACHE_PAIR_SCHEMA",
    "FORMAL_CACHE_PAYLOAD_SCHEMA",
    "FORMAL_CACHE_VERIFICATION_SCHEMA",
    "VerifiedFormalCacheArtifact",
    "VerifiedFormalCachePair",
    "build_formal_cache_neutral_envelope",
    "formal_cache_neutral_payload_fingerprint",
    "load_formal_scalar_cache_artifact",
    "rebuild_formal_scalar_cache_from_neutral_envelope",
    "require_verified_formal_cache_artifact",
    "require_verified_formal_cache_origin_artifact",
    "save_formal_cache_neutral_artifact_new",
    "verify_formal_cache_artifact",
    "verify_formal_cache_pair_independence",
]
