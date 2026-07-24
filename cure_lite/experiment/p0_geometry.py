"""Native-to-evaluation-grid identity diagnostics for P0-A."""

from __future__ import annotations

from math import hypot
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from ..instances import instances_from_binary_mask
from .cache_pipeline import LoadedDRCacheBundle
from .p0_protocol import P0GeometryConfig
from .training_pipeline import PreparedTrainingCatalog


P0_A_SCHEMA = "cure-lite-p0-a-geometry-v1"


def _resize_mask(mask: Tensor, size: tuple[int, int]) -> Tensor:
    source = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
    if source.ndim != 2:
        raise ValueError("geometry mask must be two-dimensional")
    image = Image.fromarray(
        source.to(torch.uint8).numpy() * np.uint8(255),
        mode="L",
    )
    height, width = size
    resized = image.resize((width, height), Image.Resampling.NEAREST)
    return torch.from_numpy((np.asarray(resized, dtype=np.uint8) > 0).copy())


def _native_mask(path: Path) -> Tensor:
    with Image.open(path) as handle:
        value = np.asarray(handle.convert("L"), dtype=np.uint8)
    return torch.from_numpy((value > 0).copy())


def _quantized(value: float, scale: int) -> float:
    return round(float(value) * scale) / scale


def _scaled_centroid(
    centroid: tuple[float, float],
    native_size: tuple[int, int],
    evaluation_size: tuple[int, int],
) -> tuple[float, float]:
    return tuple(
        (coordinate + 0.5) * destination / source - 0.5
        for coordinate, source, destination in zip(
            centroid,
            native_size,
            evaluation_size,
            strict=True,
        )
    )


def build_p0_a_geometry(
    bundle: LoadedDRCacheBundle,
    catalog: PreparedTrainingCatalog,
    config: P0GeometryConfig,
) -> dict[str, object]:
    """Trace every native D_R GT component through the frozen nearest resize."""

    if not isinstance(bundle, LoadedDRCacheBundle):
        raise TypeError("bundle must be a LoadedDRCacheBundle")
    if not isinstance(catalog, PreparedTrainingCatalog):
        raise TypeError("catalog must be a PreparedTrainingCatalog")
    if not isinstance(config, P0GeometryConfig):
        raise TypeError("config must be a P0GeometryConfig")
    if catalog.source_ids != tuple(row.sample_id for row in bundle.rows):
        raise ValueError("catalog and D_R bundle sample identities differ")

    row_by_id = {row.sample_id: row for row in bundle.rows}
    sample_receipts: list[dict[str, object]] = []
    native_total = resized_total = legal_total = 0
    disappeared_total = merge_total = split_total = 0
    legal_untraceable_total = legal_area_fail_total = 0
    legal_centroid_fail_total = 0

    for entry in catalog.entries:
        row = row_by_id[entry.sample_id]
        native_mask = _native_mask(row.mask_path)
        if tuple(native_mask.shape) != config.expected_native_size:
            raise RuntimeError(
                f"{entry.sample_id!r} native mask shape differs from P0 freeze"
            )
        if tuple(row.state.gt_labels.shape) != config.expected_evaluation_size:
            raise RuntimeError(
                f"{entry.sample_id!r} evaluation mask shape differs from P0 freeze"
            )
        native = instances_from_binary_mask(
            native_mask,
            connectivity=config.connectivity,
            min_area=config.min_component_area,
        )
        resized = instances_from_binary_mask(
            _resize_mask(native_mask, config.expected_evaluation_size),
            connectivity=config.connectivity,
            min_area=config.min_component_area,
        )
        if not torch.equal(resized.labels, row.state.gt_labels):
            raise RuntimeError(
                f"{entry.sample_id!r} native resize differs from cached GT labels"
            )

        projected_by_native: dict[int, Tensor] = {}
        descendants: dict[int, tuple[int, ...]] = {}
        projected_union = torch.zeros(
            config.expected_evaluation_size,
            dtype=torch.bool,
        )
        for target in native.instances:
            projected = _resize_mask(
                target.mask,
                config.expected_evaluation_size,
            )
            projected_by_native[target.instance_id] = projected
            projected_union |= projected
            descendants[target.instance_id] = tuple(
                sorted(
                    int(value)
                    for value in torch.unique(
                        row.state.gt_labels[projected]
                    ).tolist()
                    if int(value) > 0
                )
            )
        if not torch.equal(projected_union, resized.occupancy):
            raise RuntimeError(
                f"{entry.sample_id!r} per-component resize is not union-preserving"
            )

        ancestors: dict[int, tuple[int, ...]] = {}
        for target in resized.instances:
            ancestors[target.instance_id] = tuple(
                native_id
                for native_id, projected in projected_by_native.items()
                if bool(torch.any(projected & target.mask))
            )

        disappeared = tuple(
            native_id
            for native_id, child_ids in descendants.items()
            if not child_ids
        )
        split = tuple(
            native_id
            for native_id, child_ids in descendants.items()
            if len(child_ids) > 1
        )
        merged = tuple(
            resized_id
            for resized_id, parent_ids in ancestors.items()
            if len(parent_ids) > 1
        )

        legal_rows: list[dict[str, object]] = []
        for candidate in entry.decoder_visible_legal_candidates:
            legal_total += 1
            parent_ids = ancestors.get(candidate.gt_id, ())
            one_to_one = (
                len(parent_ids) == 1
                and descendants.get(parent_ids[0], ()) == (candidate.gt_id,)
            )
            record: dict[str, Any] = {
                "gt_id": candidate.gt_id,
                "pred_id": candidate.pred_id,
                "native_ancestor_ids": list(parent_ids),
                "one_to_one_lineage": one_to_one,
                "area_ratio": None,
                "centroid_shift_px256": None,
                "area_within_gate": False,
                "centroid_within_gate": False,
            }
            if one_to_one:
                native_target = native.by_id(parent_ids[0])
                resized_target = resized.by_id(candidate.gt_id)
                native_height, native_width = config.expected_native_size
                resized_height, resized_width = config.expected_evaluation_size
                expected_area = native_target.area * (
                    resized_height / native_height
                ) * (resized_width / native_width)
                area_ratio = resized_target.area / expected_area
                expected_centroid = _scaled_centroid(
                    native_target.centroid,
                    config.expected_native_size,
                    config.expected_evaluation_size,
                )
                shift = hypot(
                    resized_target.centroid[0] - expected_centroid[0],
                    resized_target.centroid[1] - expected_centroid[1],
                )
                area_ok = (
                    config.legal_area_ratio_min_inclusive
                    <= area_ratio
                    <= config.legal_area_ratio_max_inclusive
                )
                centroid_ok = (
                    shift
                    <= config.legal_centroid_shift_max_px256_inclusive
                )
                record.update(
                    {
                        "native_ancestor_id": parent_ids[0],
                        "native_area": native_target.area,
                        "evaluation_area": resized_target.area,
                        "expected_scaled_area": _quantized(
                            expected_area,
                            config.geometry_quantization,
                        ),
                        "area_ratio": _quantized(
                            area_ratio,
                            config.geometry_quantization,
                        ),
                        "native_centroid": [
                            _quantized(value, config.geometry_quantization)
                            for value in native_target.centroid
                        ],
                        "expected_evaluation_centroid": [
                            _quantized(value, config.geometry_quantization)
                            for value in expected_centroid
                        ],
                        "evaluation_centroid": [
                            _quantized(value, config.geometry_quantization)
                            for value in resized_target.centroid
                        ],
                        "centroid_shift_px256": _quantized(
                            shift,
                            config.geometry_quantization,
                        ),
                        "area_within_gate": area_ok,
                        "centroid_within_gate": centroid_ok,
                    }
                )
                legal_area_fail_total += int(not area_ok)
                legal_centroid_fail_total += int(not centroid_ok)
            else:
                legal_untraceable_total += 1
            legal_rows.append(record)

        native_rows = [
            {
                "native_id": target.instance_id,
                "area": target.area,
                "centroid": [
                    _quantized(value, config.geometry_quantization)
                    for value in target.centroid
                ],
                "projected_area": int(
                    torch.count_nonzero(
                        projected_by_native[target.instance_id]
                    )
                ),
                "descendant_ids": list(descendants[target.instance_id]),
            }
            for target in native.instances
        ]
        resized_rows = [
            {
                "evaluation_id": target.instance_id,
                "area": target.area,
                "centroid": [
                    _quantized(value, config.geometry_quantization)
                    for value in target.centroid
                ],
                "ancestor_ids": list(ancestors[target.instance_id]),
            }
            for target in resized.instances
        ]
        sample_receipts.append(
            {
                "sample_id": entry.sample_id,
                "native_targets": native_rows,
                "evaluation_targets": resized_rows,
                "decoder_visible_legal_targets": legal_rows,
                "disappeared_native_ids": list(disappeared),
                "merged_evaluation_ids": list(merged),
                "split_native_ids": list(split),
            }
        )
        native_total += len(native.instances)
        resized_total += len(resized.instances)
        disappeared_total += len(disappeared)
        merge_total += len(merged)
        split_total += len(split)

    failed_rules: list[str] = []
    if disappeared_total:
        failed_rules.append("zero_native_disappearances")
    if merge_total:
        failed_rules.append("zero_resized_merges")
    if split_total:
        failed_rules.append("zero_native_splits")
    if legal_untraceable_total:
        failed_rules.append("one_to_one_legal_lineage")
    if legal_area_fail_total:
        failed_rules.append("legal_area_ratio")
    if legal_centroid_fail_total:
        failed_rules.append("legal_centroid_shift")
    return {
        "schema_version": P0_A_SCHEMA,
        "split": "D_R",
        "summary": {
            "source_images": len(bundle.rows),
            "native_targets": native_total,
            "evaluation_targets": resized_total,
            "decoder_visible_legal_targets": legal_total,
            "native_disappearances": disappeared_total,
            "evaluation_merges": merge_total,
            "native_splits": split_total,
            "untraceable_legal_targets": legal_untraceable_total,
            "legal_area_ratio_failures": legal_area_fail_total,
            "legal_centroid_shift_failures": legal_centroid_fail_total,
        },
        "failed_rules": failed_rules,
        "p0_a_pass": not failed_rules,
        "failure_decision": (
            None if not failed_rules else "rebuild_synthetic_target_extraction"
        ),
        "samples": sample_receipts,
    }


__all__ = ["P0_A_SCHEMA", "build_p0_a_geometry"]
