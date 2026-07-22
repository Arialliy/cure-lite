from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from cure_lite.splits import SplitManifest
from tools.build_irstd1k_manifest import (
    AUDIT_SCHEMA,
    build_irstd1k_manifest,
    write_artifacts,
)


def _write_image(path: Path, seed: int) -> None:
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, size=(24, 24), dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


@pytest.fixture
def tiny_irstd1k(tmp_path: Path) -> Path:
    root = tmp_path / "IRSTD-1K"
    (root / "img_idx").mkdir(parents=True)
    (root / "images").mkdir()
    (root / "masks").mkdir()
    train_ids = ("XDU0", "XDU1", "XDU2", "XDU3", "XDU4", "XDU5")
    test_ids = ("XDU6", "XDU7")
    (root / "img_idx" / "train_IRSTD-1K.txt").write_text(
        "\n".join(train_ids) + "\n", encoding="utf-8"
    )
    (root / "img_idx" / "test_IRSTD-1K.txt").write_text(
        "\n".join(test_ids) + "\n", encoding="utf-8"
    )
    for index, sample_id in enumerate(train_ids):
        _write_image(root / "images" / f"{sample_id}.png", index + 10)
        # Deliberately not an image: successful construction proves mask bytes
        # are not consumed by grouping.
        (root / "masks" / f"{sample_id}.png").write_bytes(b"not-a-mask")
    # Make one exact image duplicate so the test is independent of perceptual
    # threshold details.
    (root / "images" / "XDU1.png").write_bytes(
        (root / "images" / "XDU0.png").read_bytes()
    )
    for sample_id in test_ids:
        # Test bytes are deliberately undecodable: D_T sealing means only the
        # paths and official membership may be inspected by this tool.
        (root / "images" / f"{sample_id}.png").write_bytes(b"sealed-test-image")
        (root / "masks" / f"{sample_id}.png").write_bytes(b"sealed-test-mask")
    return root


def _build(root: Path):
    return build_irstd1k_manifest(
        root,
        d_b_fraction=0.5,
        d_r_fraction=0.25,
        d_v_fraction=0.25,
        seed=42,
    )


def test_builds_group_disjoint_manifest_and_keeps_official_test_sealed(
    tiny_irstd1k: Path,
) -> None:
    artifacts = _build(tiny_irstd1k)
    manifest = artifacts.manifest
    manifest.validate()
    by_id = {record.sample_id: record for record in manifest.records}

    assert by_id["XDU0"].group_id == by_id["XDU1"].group_id
    assert by_id["XDU0"].split == by_id["XDU1"].split
    assert {record.sample_id for record in manifest.records_for("D_T")} == {
        "XDU6",
        "XDU7",
    }
    assert all(
        record.near_duplicate_group is None
        for record in manifest.records_for("D_T")
    )
    assert all(manifest.records_for(split) for split in ("D_B", "D_R", "D_V"))

    audit = artifacts.audit
    assert audit["schema_version"] == AUDIT_SCHEMA
    assert audit["manifest_fingerprint"] == manifest.fingerprint
    assert audit["d_t_content_accessed"] is False
    assert audit["grouping"]["scope"] == "official_train_only"
    assert audit["grouping"]["labels_used"] is False
    assert audit["d_t_policy"]["d_t_content_accessed"] is False
    assert audit["d_t_policy"]["used_for_development_grouping"] is False
    assert audit["d_t_policy"]["used_for_development_allocation"] is False
    assert audit["content_access_policy"]["official_test_images"].endswith(
        "content_not_read"
    )
    assert {
        row["sample_id"] for row in audit["grouping"]["train_image_signatures"]
    } == {"XDU0", "XDU1", "XDU2", "XDU3", "XDU4", "XDU5"}


def test_output_is_deterministic_loadable_and_never_overwrites(
    tiny_irstd1k: Path, tmp_path: Path
) -> None:
    first = _build(tiny_irstd1k)
    second = _build(tiny_irstd1k)
    assert first.manifest.canonical_payload() == second.manifest.canonical_payload()
    assert first.audit == second.audit

    manifest_path = tmp_path / "outputs" / "manifest.json"
    audit_path = tmp_path / "outputs" / "audit.json"
    write_artifacts(
        first,
        manifest_out=manifest_path,
        audit_out=audit_path,
        dataset_root=tiny_irstd1k,
    )
    loaded = SplitManifest.load(manifest_path)
    assert loaded.fingerprint == first.manifest.fingerprint
    assert json.loads(audit_path.read_text(encoding="utf-8")) == first.audit

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_artifacts(
            first,
            manifest_out=manifest_path,
            audit_out=audit_path,
            dataset_root=tiny_irstd1k,
        )


def test_rejects_output_inside_read_only_dataset_scope(tiny_irstd1k: Path) -> None:
    artifacts = _build(tiny_irstd1k)
    with pytest.raises(ValueError, match="inside the dataset root"):
        write_artifacts(
            artifacts,
            manifest_out=tiny_irstd1k / "manifest.json",
            audit_out=tiny_irstd1k.parent / "audit.json",
            dataset_root=tiny_irstd1k,
        )


def test_rejects_official_membership_overlap_and_invalid_fractions(
    tiny_irstd1k: Path,
) -> None:
    test_index = tiny_irstd1k / "img_idx" / "test_IRSTD-1K.txt"
    test_index.write_text("XDU0\nXDU6\nXDU7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="official train/test sample IDs overlap"):
        _build(tiny_irstd1k)

    test_index.write_text("XDU6\nXDU7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fractions must sum to 1"):
        build_irstd1k_manifest(
            tiny_irstd1k,
            d_b_fraction=0.5,
            d_r_fraction=0.3,
            d_v_fraction=0.3,
            seed=42,
        )
