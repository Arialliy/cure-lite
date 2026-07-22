#!/usr/bin/env python3
"""Build a frozen, group-disjoint CURE-Lite manifest for one IRSTD benchmark.

Only official-training image pixels are inspected.  Masks are resolved by
name but never opened, and official-test image/mask bytes are never read.  The
official test membership is copied unchanged to D_T; only official training
groups are allocated among D_B, D_R, and D_V.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.benchmarks import (
    SUPPORTED_BENCHMARKS,
    BenchmarkCatalog,
    catalog_benchmark_dataset,
)
from cure_lite.splits import SplitManifest, SplitName, SplitRecord


AUDIT_SCHEMA = "cure-lite-irstd-benchmark-manifest-audit-v1"
GROUPING_ALGORITHM = "train-only-phash64-plus-zrmse-single-linkage-v1"
DEVELOPMENT_SPLITS: tuple[SplitName, ...] = ("D_B", "D_R", "D_V")


@dataclass(frozen=True)
class BuildArtifacts:
    """In-memory outputs; writing is an explicit, separate operation."""

    manifest: SplitManifest
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class _ImageSignature:
    sample_id: str
    sha256: str
    phash: int
    normalized_thumbnail: np.ndarray
    width: int
    height: int
    mode: str


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        first, second = sorted((left_root, right_root))
        self._parent[second] = first
        return True


def _dct_basis(low_frequency_size: int = 8, input_size: int = 32) -> np.ndarray:
    positions = np.arange(input_size, dtype=np.float64) + 0.5
    frequencies = np.arange(low_frequency_size, dtype=np.float64)[:, None]
    basis = np.cos(math.pi * frequencies * positions / input_size)
    basis[0] *= math.sqrt(1.0 / input_size)
    basis[1:] *= math.sqrt(2.0 / input_size)
    return basis


_PHASH_DCT_BASIS = _dct_basis()


def _bits_to_integer(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return value


def _image_signature(sample_id: str, path: Path) -> _ImageSignature:
    """Read one official-training image and derive fixed label-free features."""

    digest = file_sha256(path)
    try:
        with Image.open(path) as source:
            source.load()
            width, height = source.size
            mode = source.mode
            gray = source.convert("L")
            phash_image = gray.resize((32, 32), Image.Resampling.LANCZOS)
            thumbnail_image = gray.resize((16, 16), Image.Resampling.BILINEAR)
    except Exception as error:
        raise ValueError(f"unable to decode official-training image {path}") from error

    phash_array = np.asarray(phash_image, dtype=np.float64) / 255.0
    coefficients = _PHASH_DCT_BASIS @ phash_array @ _PHASH_DCT_BASIS.T
    low_frequency = coefficients.reshape(-1)
    median = float(np.median(low_frequency[1:]))
    phash_bits = low_frequency >= median
    phash_bits[0] = False  # DC intensity must not drive similarity.

    thumbnail = np.asarray(thumbnail_image, dtype=np.float32).reshape(-1) / 255.0
    thumbnail = thumbnail - float(thumbnail.mean())
    scale = max(float(thumbnail.std()), 1.0 / 255.0)
    normalized = np.ascontiguousarray(thumbnail / scale, dtype=np.float32)
    normalized.setflags(write=False)
    return _ImageSignature(
        sample_id=sample_id,
        sha256=digest,
        phash=_bits_to_integer(phash_bits),
        normalized_thumbnail=normalized,
        width=width,
        height=height,
        mode=mode,
    )


def _dataset_token(dataset: str) -> str:
    return dataset.lower().replace("_", "-")


def _group_id(dataset: str, members: Iterable[str]) -> str:
    canonical = json.dumps(sorted(members), separators=(",", ":")).encode("utf-8")
    return f"{_dataset_token(dataset)}-ndg-" + hashlib.sha256(canonical).hexdigest()


def _official_test_group_id(dataset: str, sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return f"{_dataset_token(dataset)}-official-test-" + digest


def _build_train_groups(
    signatures: Mapping[str, _ImageSignature],
    *,
    dataset: str,
    phash_hamming_threshold: int,
    normalized_rmse_threshold: float,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    sample_ids = sorted(signatures)
    disjoint = _DisjointSet(sample_ids)
    accepted_pairs = 0
    spanning_edges: list[dict[str, Any]] = []
    compared_pairs = 0
    phash_candidates = 0

    for left_index, left_id in enumerate(sample_ids):
        left = signatures[left_id]
        for right_id in sample_ids[left_index + 1 :]:
            right = signatures[right_id]
            compared_pairs += 1
            exact = left.sha256 == right.sha256
            hamming = (left.phash ^ right.phash).bit_count()
            if not exact and hamming > phash_hamming_threshold:
                continue
            phash_candidates += 1
            rmse = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            left.normalized_thumbnail
                            - right.normalized_thumbnail,
                            dtype=np.float32,
                        ),
                        dtype=np.float64,
                    )
                )
            )
            if not exact and rmse > normalized_rmse_threshold:
                continue
            accepted_pairs += 1
            if disjoint.union(left_id, right_id):
                spanning_edges.append(
                    {
                        "left": left_id,
                        "right": right_id,
                        "reason": "exact_sha256" if exact else "perceptual_near_duplicate",
                        "phash_hamming": hamming,
                        "normalized_thumbnail_rmse": rmse,
                    }
                )

    components: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        components.setdefault(disjoint.find(sample_id), []).append(sample_id)
    groups: dict[str, tuple[str, ...]] = {}
    for members in components.values():
        canonical_members = tuple(sorted(members))
        identifier = _group_id(dataset, canonical_members)
        if identifier in groups:
            raise RuntimeError("near-duplicate group digest collision")
        groups[identifier] = canonical_members
    return groups, {
        "compared_pairs": compared_pairs,
        "phash_candidate_pairs": phash_candidates,
        "accepted_pairs": accepted_pairs,
        "spanning_edges": spanning_edges,
    }


def _validate_fractions(fractions: Mapping[SplitName, float]) -> None:
    if set(fractions) != set(DEVELOPMENT_SPLITS):
        raise ValueError("fractions must define exactly D_B, D_R, and D_V")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in fractions.values()
    ):
        raise ValueError("all development split fractions must be finite and positive")
    if not math.isclose(sum(float(value) for value in fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("D_B, D_R, and D_V fractions must sum to 1")


def _allocate_groups(
    groups: Mapping[str, tuple[str, ...]],
    *,
    fractions: Mapping[SplitName, float],
    seed: int,
) -> tuple[dict[str, SplitName], dict[SplitName, int]]:
    _validate_fractions(fractions)
    if len(groups) < len(DEVELOPMENT_SPLITS):
        raise ValueError("at least three independent training groups are required")
    total_samples = sum(len(members) for members in groups.values())
    targets = {
        split: float(fractions[split]) * total_samples for split in DEVELOPMENT_SPLITS
    }
    assigned_counts = {split: 0 for split in DEVELOPMENT_SPLITS}

    def seeded_rank(group_id: str) -> str:
        return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()

    ordered_groups = sorted(
        groups,
        key=lambda group_id: (-len(groups[group_id]), seeded_rank(group_id), group_id),
    )
    assignments: dict[str, SplitName] = {}
    for group_id in ordered_groups:
        # Largest normalized remaining deficit wins.  The fixed split index is
        # the final tie-break, so assignment is independent of input ordering.
        destination = max(
            DEVELOPMENT_SPLITS,
            key=lambda split: (
                (targets[split] - assigned_counts[split]) / targets[split],
                -DEVELOPMENT_SPLITS.index(split),
            ),
        )
        assignments[group_id] = destination
        assigned_counts[destination] += len(groups[group_id])
    empty = [split for split, count in assigned_counts.items() if count == 0]
    if empty:
        raise ValueError(
            f"group topology cannot realize non-empty development splits: {empty}"
        )
    return assignments, assigned_counts


def build_irstd_manifest(
    dataset_root: str | Path,
    *,
    dataset: str | None = None,
    d_b_fraction: float,
    d_r_fraction: float,
    d_v_fraction: float,
    seed: int,
    phash_hamming_threshold: int = 6,
    normalized_rmse_threshold: float = 0.30,
) -> BuildArtifacts:
    """Build deterministic outputs for one supported benchmark without writing."""

    catalog: BenchmarkCatalog = catalog_benchmark_dataset(
        dataset_root,
        dataset=dataset,
    )
    root = catalog.root
    dataset_id = catalog.spec.storage_id
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(phash_hamming_threshold, bool)
        or not isinstance(phash_hamming_threshold, int)
        or not 0 <= phash_hamming_threshold <= 63
    ):
        raise ValueError("phash hamming threshold must be an integer in [0,63]")
    if (
        isinstance(normalized_rmse_threshold, bool)
        or not isinstance(normalized_rmse_threshold, (int, float))
        or not math.isfinite(float(normalized_rmse_threshold))
        or normalized_rmse_threshold < 0
    ):
        raise ValueError("normalized RMSE threshold must be finite and nonnegative")

    train_index = catalog.train_index_path
    test_index = catalog.test_index_path
    train_ids = catalog.train_sample_ids
    test_ids = catalog.test_sample_ids
    images = {asset.sample_id: asset.image_path for asset in catalog.train_assets}
    images.update(
        {asset.sample_id: asset.image_path for asset in catalog.test_assets}
    )
    masks = {asset.sample_id: asset.mask_path for asset in catalog.train_assets}
    masks.update({asset.sample_id: asset.mask_path for asset in catalog.test_assets})

    # Deliberately restricted to official train.  D_T image and mask bytes are
    # not opened or hashed by this builder.
    signatures = {
        sample_id: _image_signature(sample_id, images[sample_id])
        for sample_id in sorted(train_ids)
    }
    groups, grouping_audit = _build_train_groups(
        signatures,
        dataset=dataset_id,
        phash_hamming_threshold=phash_hamming_threshold,
        normalized_rmse_threshold=float(normalized_rmse_threshold),
    )
    fractions: dict[SplitName, float] = {
        "D_B": float(d_b_fraction),
        "D_R": float(d_r_fraction),
        "D_V": float(d_v_fraction),
    }
    assignments, development_counts = _allocate_groups(
        groups,
        fractions=fractions,
        seed=seed,
    )
    train_group_by_sample = {
        sample_id: group_id
        for group_id, members in groups.items()
        for sample_id in members
    }

    records: list[SplitRecord] = []
    for sample_id in train_ids:
        group_id = train_group_by_sample[sample_id]
        records.append(
            SplitRecord(
                sample_id=sample_id,
                split=assignments[group_id],
                group_id=group_id,
                image=str(images[sample_id]),
                mask=str(masks[sample_id]),
                near_duplicate_group=group_id,
            )
        )
    for sample_id in test_ids:
        records.append(
            SplitRecord(
                sample_id=sample_id,
                split="D_T",
                group_id=_official_test_group_id(dataset_id, sample_id),
                image=str(images[sample_id]),
                mask=str(masks[sample_id]),
            )
        )
    manifest = SplitManifest(dataset=dataset_id, records=tuple(records))

    group_rows = []
    for group_id, members in sorted(groups.items()):
        group_rows.append(
            {
                "group_id": group_id,
                "split": assignments[group_id],
                "size": len(members),
                "members": list(members),
            }
        )
    signature_rows = [
        {
            "sample_id": signature.sample_id,
            "image_sha256": signature.sha256,
            "phash64": f"{signature.phash:016x}",
            "width": signature.width,
            "height": signature.height,
            "source_mode": signature.mode,
        }
        for signature in (signatures[sample_id] for sample_id in sorted(signatures))
    ]
    split_counts = {
        split: len(manifest.records_for(split))
        for split in ("D_B", "D_R", "D_V", "D_T")
    }
    group_counts = {
        split: sum(1 for value in assignments.values() if value == split)
        for split in DEVELOPMENT_SPLITS
    }
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "dataset": dataset_id,
        "dataset_root": str(root),
        "manifest_fingerprint": manifest.fingerprint,
        "d_t_content_accessed": False,
        "official_indexes": {
            "train": {
                "relative_path": str(catalog.spec.train_index_relative_path),
                "sha256": file_sha256(train_index),
                "samples": len(train_ids),
            },
            "test": {
                "relative_path": str(catalog.spec.test_index_relative_path),
                "sha256": file_sha256(test_index),
                "samples": len(test_ids),
            },
        },
        "content_access_policy": {
            "official_train_images": "decoded_for_label_free_grouping",
            "official_train_masks": "path_existence_only_content_not_read",
            "official_test_images": "path_existence_only_content_not_read",
            "official_test_masks": "path_existence_only_content_not_read",
        },
        "d_t_policy": {
            "source": "complete_official_test_index",
            "allocation": "immutable_D_T",
            "d_t_content_accessed": False,
            "used_for_development_grouping": False,
            "used_for_development_allocation": False,
        },
        "grouping": {
            "algorithm": GROUPING_ALGORITHM,
            "scope": "official_train_only",
            "labels_used": False,
            "phash_hamming_threshold": phash_hamming_threshold,
            "normalized_thumbnail_rmse_threshold": float(
                normalized_rmse_threshold
            ),
            "linkage": "single_linkage_transitive_components",
            "train_image_signatures": signature_rows,
            "groups": group_rows,
            **grouping_audit,
        },
        "allocation": {
            "unit": "near_duplicate_group",
            "algorithm": "largest-group-first-normalized-deficit-v1",
            "seed": seed,
            "requested_fractions": fractions,
            "target_sample_counts": {
                split: fractions[split] * len(train_ids)
                for split in DEVELOPMENT_SPLITS
            },
            "realized_sample_counts": development_counts,
            "realized_group_counts": group_counts,
        },
        "split_sample_counts": split_counts,
    }
    audit["audit_fingerprint"] = stable_fingerprint(audit)
    return BuildArtifacts(manifest=manifest, audit=audit)


def build_irstd1k_manifest(
    dataset_root: str | Path,
    *,
    d_b_fraction: float,
    d_r_fraction: float,
    d_v_fraction: float,
    seed: int,
    phash_hamming_threshold: int = 6,
    normalized_rmse_threshold: float = 0.30,
) -> BuildArtifacts:
    """Backward-compatible IRSTD-1K wrapper around the three-dataset builder."""

    return build_irstd_manifest(
        dataset_root,
        dataset="IRSTD-1K",
        d_b_fraction=d_b_fraction,
        d_r_fraction=d_r_fraction,
        d_v_fraction=d_v_fraction,
        seed=seed,
        phash_hamming_threshold=phash_hamming_threshold,
        normalized_rmse_threshold=normalized_rmse_threshold,
    )


def _resolved_output(path: str | Path, *, dataset_root: Path) -> Path:
    candidate = Path(path).expanduser().resolve(strict=False)
    try:
        candidate.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("outputs may not be written inside the dataset root")
    return candidate


def _encode_json(payload: Mapping[str, Any]) -> bytes:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    return (serialized + "\n").encode("utf-8")


def write_artifacts(
    artifacts: BuildArtifacts,
    *,
    manifest_out: str | Path,
    audit_out: str | Path,
    dataset_root: str | Path,
) -> None:
    """Publish both JSON files without overwriting either destination."""

    root = Path(dataset_root).expanduser().resolve(strict=True)
    manifest_path = _resolved_output(manifest_out, dataset_root=root)
    audit_path = _resolved_output(audit_out, dataset_root=root)
    if manifest_path == audit_path:
        raise ValueError("manifest and audit outputs must be different paths")
    if manifest_path.exists() or audit_path.exists():
        raise FileExistsError("refusing to overwrite an existing manifest or audit")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    payloads = (
        (manifest_path, _encode_json(artifacts.manifest.canonical_payload())),
        (audit_path, _encode_json(artifacts.audit)),
    )
    temporary_paths: list[Path] = []
    published: list[Path] = []
    try:
        for destination, content in payloads:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for (destination, _), temporary in zip(payloads, temporary_paths, strict=True):
            os.link(temporary, destination)
            published.append(destination)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)

    loaded = SplitManifest.load(manifest_path)
    if loaded.fingerprint != artifacts.manifest.fingerprint:
        raise RuntimeError("published manifest fingerprint verification failed")


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("fraction must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def _hamming_threshold(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 63:
        raise argparse.ArgumentTypeError("value must be in [0,63]")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_BENCHMARKS,
        help="exact benchmark storage ID; defaults to the dataset-root leaf name",
    )
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--d-b-fraction", type=_fraction, required=True)
    parser.add_argument("--d-r-fraction", type=_fraction, required=True)
    parser.add_argument("--d-v-fraction", type=_fraction, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--phash-hamming-threshold", type=_hamming_threshold, default=6
    )
    parser.add_argument(
        "--normalized-rmse-threshold", type=_nonnegative_float, default=0.30
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = build_irstd_manifest(
        args.dataset_root,
        dataset=args.dataset,
        d_b_fraction=args.d_b_fraction,
        d_r_fraction=args.d_r_fraction,
        d_v_fraction=args.d_v_fraction,
        seed=args.seed,
        phash_hamming_threshold=args.phash_hamming_threshold,
        normalized_rmse_threshold=args.normalized_rmse_threshold,
    )
    write_artifacts(
        artifacts,
        manifest_out=args.manifest_out,
        audit_out=args.audit_out,
        dataset_root=args.dataset_root,
    )
    print(f"manifest={Path(args.manifest_out).expanduser().resolve()}")
    print(f"audit={Path(args.audit_out).expanduser().resolve()}")
    print(f"manifest_fingerprint={artifacts.manifest.fingerprint}")
    for split in ("D_B", "D_R", "D_V", "D_T"):
        print(f"{split}={len(artifacts.manifest.records_for(split))}")


if __name__ == "__main__":
    main()
