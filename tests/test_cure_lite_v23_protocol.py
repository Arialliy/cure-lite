from __future__ import annotations

import json
from pathlib import Path

import pytest

from cure_lite_v23.protocol import (
    PACRE_VC_PACKAGE_METADATA_PATHS,
    PACRE_VC_SOURCE_LAYER_NAMES,
    fingerprinted,
    read_strict_json,
    source_closure_payload,
    source_inventory,
    verify_fingerprinted,
    verify_source_closure,
    write_new_json,
)


def test_source_closure_covers_all_three_packages() -> None:
    inventory = source_inventory()
    closure = source_closure_payload()

    assert any(path.startswith("cure_lite/") for path in inventory)
    assert any(path.startswith("cure_lite_v22/") for path in inventory)
    assert any(path.startswith("cure_lite_v23/") for path in inventory)
    assert set(PACRE_VC_PACKAGE_METADATA_PATHS).issubset(inventory)
    assert closure["file_count"] == len(inventory)
    assert tuple(closure["layer_order"]) == PACRE_VC_SOURCE_LAYER_NAMES
    assert set(closure["layers"]) == set(PACRE_VC_SOURCE_LAYER_NAMES)
    assert sum(
        layer["file_count"] for layer in closure["layers"].values()
    ) == len(inventory)
    assert verify_source_closure(closure) == closure["closure_fingerprint"]


def test_fingerprinted_payload_recomputes_exactly() -> None:
    payload = fingerprinted({"schema_version": "test-v1", "passed": True})

    assert verify_fingerprinted(payload) == payload["receipt_fingerprint"]
    changed = dict(payload)
    changed["passed"] = False
    with pytest.raises(ValueError, match="fingerprint"):
        verify_fingerprinted(changed)


def test_json_artifact_is_create_only_and_canonical(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = fingerprinted(
        {"schema_version": "test-v1", "rows": [2, 1], "passed": True}
    )

    digest = write_new_json(path, payload)
    assert len(digest) == 64
    assert read_strict_json(path) == payload

    with pytest.raises(FileExistsError):
        write_new_json(path, payload)


def test_reader_rejects_noncanonical_or_nonfinite_json(
    tmp_path: Path,
) -> None:
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps({"b": 1, "a": 2}, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-canonical"):
        read_strict_json(noncanonical)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}\\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        read_strict_json(nonfinite)
