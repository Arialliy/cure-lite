from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite_v24.artifact_io import (
    atomic_write_new_bytes,
    atomic_write_new_canonical_json,
    load_terminal_safetensors_strict,
    read_canonical_json,
    regular_file_receipt,
    save_terminal_safetensors_new,
)


def _metadata(model: torch.nn.Module) -> dict[str, str]:
    model_fp = stable_fingerprint(
        {
            name: tensor.detach().cpu().tolist()
            for name, tensor in model.state_dict().items()
        }
    )
    return {
        "schema": "generated-terminal-v1",
        "run": "generated-only",
        "seed": "42",
        "role": "test",
        "model_fingerprint": model_fp,
    }


def test_atomic_terminal_is_absolute_read_only_and_no_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(3, 2, bias=True, dtype=torch.float32)
    monkeypatch.chdir(tmp_path)
    receipt = save_terminal_safetensors_new(
        "terminal.safetensors",
        model,
        metadata=_metadata(model),
    )
    path = Path(str(receipt["path"]))
    assert path.is_absolute()
    assert path.parent == tmp_path
    assert file_sha256(path) == receipt["file_sha256"]
    assert path.stat().st_mode & 0o777 == 0o444
    assert path.stat().st_nlink == 1
    state = load_terminal_safetensors_strict(path)
    assert set(state) == set(model.state_dict())

    with pytest.raises(FileExistsError):
        save_terminal_safetensors_new(
            path,
            model,
            metadata=_metadata(model),
        )


def test_atomic_writer_rejects_existing_symlink_and_hardlink_mutation(
    tmp_path: Path,
) -> None:
    source = atomic_write_new_bytes(tmp_path / "source.bin", b"source")
    alias = tmp_path / "alias.bin"
    os.symlink(source, alias)
    with pytest.raises(FileExistsError):
        atomic_write_new_bytes(alias, b"replacement")

    hardlink = tmp_path / "hardlink.bin"
    os.link(source, hardlink)
    with pytest.raises(RuntimeError, match="regular"):
        regular_file_receipt(source)
    with pytest.raises(RuntimeError, match="regular"):
        load_terminal_safetensors_strict(source)


def test_canonical_json_and_tampered_safetensors_fail_closed(
    tmp_path: Path,
) -> None:
    canonical_path = atomic_write_new_canonical_json(
        tmp_path / "canonical.json",
        {"b": 2, "a": 1},
    )
    assert read_canonical_json(canonical_path) == {"a": 1, "b": 2}

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text('{"b":2, "a":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        read_canonical_json(noncanonical)

    model = torch.nn.Linear(2, 1, dtype=torch.float32)
    receipt = save_terminal_safetensors_new(
        tmp_path / "tampered.safetensors",
        model,
        metadata=_metadata(model),
    )
    path = Path(str(receipt["path"]))
    os.chmod(path, 0o600)
    path.write_bytes(b"not-a-safetensors-file")
    with pytest.raises(Exception):
        load_terminal_safetensors_strict(path)


def test_terminal_metadata_requires_all_content_bindings(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1, dtype=torch.float32)
    for missing in ("schema", "run", "seed", "role", "model_fingerprint"):
        metadata = _metadata(model)
        metadata.pop(missing)
        with pytest.raises(TypeError, match="metadata"):
            save_terminal_safetensors_new(
                tmp_path / f"missing-{missing}.safetensors",
                model,
                metadata=metadata,
            )
