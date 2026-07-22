"""Frozen, group-disjoint data-split manifests for CURE-Lite experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import yaml


SplitName = Literal["D_B", "D_R", "D_V", "D_T"]
ALL_SPLITS: tuple[SplitName, ...] = ("D_B", "D_R", "D_V", "D_T")

_PURPOSE_TO_SPLIT: dict[str, SplitName] = {
    "base_train": "D_B",
    "residual_state": "D_R",
    "residual_train": "D_R",
    "calibration": "D_V",
    "validation": "D_V",
    "final_test": "D_T",
}


@dataclass(frozen=True)
class SplitRecord:
    """One sample plus all grouping keys that can reveal shared provenance."""

    sample_id: str
    split: SplitName
    group_id: str
    image: str
    mask: str | None = None
    scene_id: str | None = None
    sequence_id: str | None = None
    crop_source_id: str | None = None
    near_duplicate_group: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SplitRecord":
        known = {
            "sample_id",
            "split",
            "group_id",
            "image",
            "mask",
            "scene_id",
            "sequence_id",
            "crop_source_id",
            "near_duplicate_group",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown split-record fields: {sorted(unknown)}")
        missing = {"sample_id", "split", "group_id", "image"} - set(value)
        if missing:
            raise ValueError(f"missing split-record fields: {sorted(missing)}")
        return cls(**dict(value))

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id or not self.image:
            raise ValueError("sample_id, group_id, and image must be non-empty")
        if self.split not in ALL_SPLITS:
            raise ValueError(f"invalid split {self.split!r}")
        for name in (
            "scene_id",
            "sequence_id",
            "crop_source_id",
            "near_duplicate_group",
        ):
            if getattr(self, name) == "":
                raise ValueError(f"{name} must be null or non-empty")

    def grouping_keys(self) -> tuple[tuple[str, str], ...]:
        values = [("group_id", self.group_id)]
        for name in (
            "scene_id",
            "sequence_id",
            "crop_source_id",
            "near_duplicate_group",
        ):
            value = getattr(self, name)
            if value is not None:
                values.append((name, value))
        return tuple(values)


@dataclass(frozen=True)
class SplitManifest:
    """Validated immutable description of D_B, D_R, D_V, and D_T."""

    dataset: str
    records: tuple[SplitRecord, ...]
    schema_version: str = "cure-lite-splits-v1"
    created_before_training: bool = True
    manifest_directory: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != "cure-lite-splits-v1":
            raise ValueError(f"unsupported manifest schema {self.schema_version!r}")
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        if not self.created_before_training:
            raise ValueError("manifest must be frozen before training")
        if self.manifest_directory is not None and not self.manifest_directory.is_absolute():
            raise ValueError("manifest_directory must be absolute when provided")
        self.validate()

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                payload = json.load(handle)
            else:
                payload = yaml.safe_load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("manifest root must be a mapping")
        allowed = {"schema_version", "dataset", "created_before_training", "samples"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError("manifest samples must be a list")
        return cls(
            dataset=str(payload.get("dataset", "")),
            records=tuple(SplitRecord.from_mapping(item) for item in samples),
            schema_version=str(payload.get("schema_version", "")),
            created_before_training=payload.get("created_before_training") is True,
            manifest_directory=path.resolve().parent,
        )

    def validate(self, *, require_all_splits: bool = True) -> None:
        if not self.records:
            raise ValueError("manifest contains no samples")
        sample_ids: set[str] = set()
        asset_owners: dict[str, tuple[SplitName, str]] = {}
        group_owners: dict[tuple[str, str], SplitName] = {}
        counts = {split: 0 for split in ALL_SPLITS}
        for record in self.records:
            if record.sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id {record.sample_id!r}")
            sample_ids.add(record.sample_id)
            counts[record.split] += 1

            image_path = Path(record.image).expanduser()
            if not image_path.is_absolute() and self.manifest_directory is not None:
                image_path = self.manifest_directory / image_path
            normalized_image = str(image_path.resolve(strict=False))
            prior_image_split, prior_image_kind = asset_owners.setdefault(
                normalized_image, (record.split, "image")
            )
            if prior_image_split != record.split:
                raise ValueError(
                    f"image {record.image!r} crosses {prior_image_split}/{record.split} "
                    f"(previously used as {prior_image_kind})"
                )
            if record.mask is not None:
                mask_path = Path(record.mask).expanduser()
                if not mask_path.is_absolute() and self.manifest_directory is not None:
                    mask_path = self.manifest_directory / mask_path
                normalized_mask = str(mask_path.resolve(strict=False))
                prior_mask_split, prior_mask_kind = asset_owners.setdefault(
                    normalized_mask, (record.split, "mask")
                )
                if prior_mask_split != record.split:
                    raise ValueError(
                        f"mask {record.mask!r} crosses {prior_mask_split}/{record.split} "
                        f"(previously used as {prior_mask_kind})"
                    )

            for grouping_key in record.grouping_keys():
                prior_split = group_owners.setdefault(grouping_key, record.split)
                if prior_split != record.split:
                    kind, value = grouping_key
                    raise ValueError(
                        f"{kind}={value!r} crosses {prior_split}/{record.split}"
                    )
        if require_all_splits:
            empty = [split for split, count in counts.items() if count == 0]
            if empty:
                raise ValueError(f"formal manifest has empty splits: {empty}")

    def records_for(self, split: SplitName) -> tuple[SplitRecord, ...]:
        if split not in ALL_SPLITS:
            raise ValueError(f"invalid split {split!r}")
        return tuple(record for record in self.records if record.split == split)

    def assert_purpose(self, purpose: str, records: Iterable[SplitRecord]) -> None:
        try:
            expected = _PURPOSE_TO_SPLIT[purpose]
        except KeyError as error:
            raise ValueError(f"unknown experimental purpose {purpose!r}") from error
        wrong = sorted(record.sample_id for record in records if record.split != expected)
        if wrong:
            raise RuntimeError(
                f"purpose {purpose!r} may use only {expected}; invalid samples: {wrong}"
            )

    def canonical_payload(self) -> dict[str, Any]:
        rows = []
        for record in sorted(self.records, key=lambda item: item.sample_id):
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "split": record.split,
                    "group_id": record.group_id,
                    "image": record.image,
                    "mask": record.mask,
                    "scene_id": record.scene_id,
                    "sequence_id": record.sequence_id,
                    "crop_source_id": record.crop_source_id,
                    "near_duplicate_group": record.near_duplicate_group,
                }
            )
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "created_before_training": self.created_before_training,
            "samples": rows,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_and_validate_manifest(path: str | Path) -> SplitManifest:
    """Load a manifest and enforce every formal CURE-Lite split invariant."""

    return SplitManifest.load(path)
