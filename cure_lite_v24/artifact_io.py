"""Small create-only artifact primitives shared by v24 protected runners.

The helpers in this module deliberately know nothing about datasets, stage
authorization, or model selection.  They provide only:

* canonical JSON encoding and strict canonical reads;
* same-directory atomic, no-replace publication;
* final-state-only ``safetensors`` serialization; and
* recomputable regular-file metadata.

No pickle-capable loader is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
from hashlib import sha256
from typing import Mapping

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from cure_lite.cache.schema import canonical_json, file_sha256


def _regular_parent(path: Path) -> Path:
    parent = path.parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
    ):
        raise RuntimeError(
            f"artifact parent is not a regular resolved directory: {parent}"
        )
    return parent


def atomic_write_new_bytes(path: str | Path, payload: bytes) -> Path:
    """Publish ``payload`` atomically and fail if ``path`` already exists."""

    supplied = Path(path)
    parent = supplied.parent
    if not supplied.is_absolute():
        parent = (Path.cwd() / parent).resolve(strict=True)
    else:
        parent = parent.resolve(strict=True)
    target = parent / supplied.name
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not target.name or target.name in {".", ".."}:
        raise ValueError("artifact target must name one file")
    parent = _regular_parent(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)

    temporary = parent / (
        f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link commit has no replacement mode: an existing target
        # makes ``link`` fail, while a successful link is immediately visible
        # with complete bytes.  The temporary name is then removed.
        os.link(temporary, target, follow_symlinks=False)
        os.unlink(temporary)
        if file_sha256(target) != sha256(payload).hexdigest():
            raise RuntimeError("published artifact bytes changed")
        os.chmod(target, 0o444, follow_symlinks=False)
        published_descriptor = os.open(target, os.O_RDONLY)
        try:
            os.fsync(published_descriptor)
        finally:
            os.close(published_descriptor)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    if (
        not target.is_file()
        or target.is_symlink()
        or target.resolve(strict=True) != target
        or target.stat().st_nlink != 1
    ):
        raise RuntimeError("published artifact is not a unique regular file")
    return target


def atomic_write_new_canonical_json(
    path: str | Path,
    payload: Mapping[str, object],
) -> Path:
    """Publish exactly one newline-terminated canonical JSON object."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    encoded = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    return atomic_write_new_bytes(path, encoded)


def read_canonical_json(path: str | Path) -> dict[str, object]:
    """Read one strict newline-terminated canonical JSON object."""

    source = Path(path)
    if (
        not source.is_file()
        or source.is_symlink()
        or source.resolve(strict=True) != source
        or source.stat().st_nlink != 1
    ):
        raise RuntimeError("canonical JSON source is not a regular file")
    raw = source.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("canonical JSON must end with one newline")
    text = raw[:-1].decode("utf-8", errors="strict")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("artifact contains invalid JSON") from error
    if not isinstance(value, dict) or canonical_json(value) != text:
        raise ValueError("artifact JSON is not one canonical object")
    return value


def _cpu_finite_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    state = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    if not state:
        raise ValueError("model state cannot be empty")
    if any(
        not tensor.is_floating_point()
        or tensor.dtype != torch.float32
        or tensor.requires_grad
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor).all())
        for tensor in state.values()
    ):
        raise ValueError(
            "terminal model state must contain finite contiguous float32 tensors"
        )
    return state


def save_terminal_safetensors_new(
    path: str | Path,
    model: torch.nn.Module,
    *,
    metadata: Mapping[str, str],
) -> dict[str, object]:
    """Save only the terminal model state through a no-replace commit."""

    required_metadata = {
        "schema",
        "run",
        "seed",
        "role",
        "model_fingerprint",
    }
    if (
        not isinstance(metadata, Mapping)
        or not required_metadata.issubset(metadata)
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in metadata.items()
        )
    ):
        raise TypeError(
            "safetensors metadata must include non-empty "
            "schema/run/seed/role/model_fingerprint text"
        )
    if (
        len(str(metadata["model_fingerprint"])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(metadata["model_fingerprint"])
        )
    ):
        raise ValueError("safetensors model_fingerprint is malformed")
    state = _cpu_finite_state(model)
    encoded = save_safetensors(state, metadata=dict(metadata))
    target = atomic_write_new_bytes(path, encoded)
    stat_result = target.stat()
    return {
        "path": str(target.resolve(strict=True)),
        "size_bytes": stat_result.st_size,
        "file_sha256": file_sha256(target),
        "state_keys": list(state),
        "state_shapes": {
            name: list(tensor.shape) for name, tensor in state.items()
        },
        "state_dtypes": {
            name: str(tensor.dtype) for name, tensor in state.items()
        },
        "parameter_count": sum(tensor.numel() for tensor in state.values()),
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "hardlink_count": stat_result.st_nlink,
    }


def load_terminal_safetensors_strict(
    path: str | Path,
) -> dict[str, torch.Tensor]:
    """Load a regular safetensors file and reject non-finite/non-FP32 state."""

    source = Path(path)
    if (
        not source.is_file()
        or source.is_symlink()
        or source.resolve(strict=True) != source
        or source.stat().st_nlink != 1
    ):
        raise RuntimeError("terminal safetensors source is not regular")
    state = load_safetensors(source.read_bytes())
    if not isinstance(state, dict) or not state:
        raise ValueError("terminal safetensors state is empty")
    if any(
        not isinstance(name, str)
        or not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
        or tensor.requires_grad
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor).all())
        for name, tensor in state.items()
    ):
        raise ValueError("terminal safetensors state is invalid")
    return state


def regular_file_receipt(path: str | Path) -> dict[str, object]:
    """Return recomputable metadata for one unique regular file."""

    source = Path(path)
    if (
        not source.is_file()
        or source.is_symlink()
        or source.resolve(strict=True) != source
        or source.stat().st_nlink != 1
    ):
        raise RuntimeError("artifact is not a regular resolved file")
    stat_result = source.stat()
    return {
        "path": str(source.resolve(strict=True)),
        "size_bytes": stat_result.st_size,
        "file_sha256": file_sha256(source),
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "hardlink_count": stat_result.st_nlink,
    }


__all__ = [
    "atomic_write_new_bytes",
    "atomic_write_new_canonical_json",
    "load_terminal_safetensors_strict",
    "read_canonical_json",
    "regular_file_receipt",
    "save_terminal_safetensors_new",
]
