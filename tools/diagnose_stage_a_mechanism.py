#!/usr/bin/env python3
"""Diagnose completed CURE-Lite Stage-A F/U target behavior without training.

The command consumes independently replayed Stage-A results, runs only the
already selected F and U decoder points, and writes one create-only development
diagnostic outside every Stage-A directory.  It never evaluates a threshold
grid and exposes no D_T input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.config import MatchConfig, OccupancyConfig, config_to_dict  # noqa: E402
from cure_lite.data import ManifestImageDataset  # noqa: E402
from cure_lite.decoder import project_occupancy_to_feature_grid  # noqa: E402
from cure_lite.experiment.artifacts import (  # noqa: E402
    LoadedDecoderArtifact,
    load_decoder_artifact,
)
from cure_lite.experiment.cache_pipeline import (  # noqa: E402
    LoadedDRCacheBundle,
    LoadedDVCacheBundle,
    load_base_cache_pair_contract,
    load_d_r_cache_bundle,
    load_d_v_cache_bundle,
)
from cure_lite.experiment.formal_evaluation import (  # noqa: E402
    LoadedDVMethodRun,
    build_loaded_d_v_method_run,
)
from cure_lite.experiment.stage_a_runner import (  # noqa: E402
    STAGE_A_RUN_SCHEMA,
    StageARunConfig,
    _COMPLETE_NAME,
    _INCOMPLETE_NAME,
    _check_dataset_pair,
    _config_receipt,
    _require_same_base_cache_identity,
    _source_tree_digest,
    _strict_json,
    _tree_inventory,
    _verified_base_run_payload,
    _verify_artifact_training_binding,
    _write_new_json,
)
from cure_lite.instances import (  # noqa: E402
    centroid_distance,
    instances_from_binary_mask,
    mask_iou,
)
from cure_lite.matching import match_components  # noqa: E402
from cure_lite.occupancy import build_occupancy  # noqa: E402
from cure_lite.reference_base import (  # noqa: E402
    load_verified_reference_base_run_identity,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402
from cure_lite.types import Instance, InstanceMap, MatchPair, MatchResult  # noqa: E402
from tools.assess_stage_a import ASSESSMENT_SCHEMA  # noqa: E402
from tools.run_stage_a import load_stage_a_config  # noqa: E402


DIAGNOSTIC_SCHEMA = "cure-lite-stage-a-mechanism-diagnostic-v1"
DR_ALIGNMENT_SCHEMA = "cure-lite-d-r-state-alignment-v1"
DV_PARTITION_SCHEMA = "cure-lite-d-v-f-u-target-partition-v1"
METHOD_ORDER = ("A", "Base@B", "F", "F×", "U")
CATEGORIES = ("f_only", "u_only", "both", "neither")
SCALAR_ALIGNMENT_FIELDS = (
    "gt_area",
    "supervision_area",
    "supervision_fraction",
    "base_gt_mean",
    "base_gt_max",
    "base_supervision_mean",
    "base_supervision_max",
    "feature_embedding_l2",
    "feature_abs_mean",
    "feature_rms",
)
SCALAR_NEIGHBOUR_FIELDS = (
    "log1p_gt_area",
    "base_gt_mean",
    "base_gt_max",
    "supervision_fraction",
)
_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _RunMetadata:
    root: Path
    assessment_path: Path
    stage_config_path: Path
    complete: dict[str, Any]
    assessment: dict[str, Any]
    config: StageARunConfig
    calibration: dict[str, Any]
    results: dict[str, Any]
    support: dict[str, Any]
    efficiency: dict[str, Any]
    snapshot: tuple[list[str], dict[str, str], str]
    assessment_sha256: str
    stage_config_sha256: str

    @property
    def seed(self) -> int:
        return self.config.training.global_seed


@dataclass(frozen=True)
class _PreparedDVRow:
    sample_id: str
    base_probability: Tensor
    feature: Tensor
    occupancy: Tensor
    anchor_instances: InstanceMap
    gt_instances: InstanceMap
    anchor_match: MatchResult


@dataclass(frozen=True)
class _MethodImageOutcome:
    sample_id: str
    residual_probability: Tensor
    residual_mask: Tensor
    prediction: Tensor
    pred_instances: InstanceMap
    match: MatchResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-base-run", type=Path, required=True)
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        nargs=3,
        type=Path,
        required=True,
        metavar=("STAGE_RUN", "ASSESSMENT", "STAGE_CONFIG"),
        help=(
            "repeat for every seed: completed Stage-A directory, independent "
            "assessment JSON, and frozen Stage-A config JSON"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _canonical_path_without_links(path: Path, *, name: str, directory: bool) -> Path:
    raw = path.expanduser()
    absolute = Path(os.path.abspath(raw))
    if raw.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if resolved != absolute:
        raise ValueError(f"{name} may not traverse a symbolic link")
    if directory:
        if not resolved.is_dir() or resolved.is_symlink():
            raise ValueError(f"{name} must be a regular directory")
    elif not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _prepare_output(path: Path, stage_roots: Sequence[Path]) -> Path:
    raw = path.expanduser()
    absolute = Path(os.path.abspath(raw))
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(f"diagnostic output already exists: {absolute}")
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("diagnostic output may not traverse a symbolic link")
    resolved = absolute.resolve(strict=False)
    for root in stage_roots:
        if resolved == root or root in resolved.parents:
            raise ValueError("diagnostic output must be outside every Stage-A run")
    return absolute


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _complete_keys() -> set[str]:
    return {
        "schema_version",
        "status",
        "method",
        "stage",
        "method_order",
        "runtime_splits",
        "unused_split",
        "source_tree_digest",
        "run_config_fingerprint",
        "anchor_receipt_fingerprint",
        "support_receipt_fingerprint",
        "calibration_receipt_fingerprint",
        "results_fingerprint",
        "efficiency_receipt_fingerprint",
        "dataset",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "base_state_fingerprint",
        "base_run_identity",
        "d_r_base_index_fingerprint",
        "d_r_state_index_fingerprint",
        "d_v_base_index_fingerprint",
        "factual_decoder_artifact_fingerprint",
        "factual_exposure_matched_decoder_artifact_fingerprint",
        "uniform_decoder_artifact_fingerprint",
        "artifact_directories",
        "artifact_files",
        "complete_fingerprint",
    }


def _assessment_keys() -> set[str]:
    return {
        "schema_version",
        "dataset",
        "seed",
        "evaluation_split",
        "independent_generalization_result",
        "runtime_splits",
        "unused_split",
        "verified_full_replay",
        "manifest_fingerprint",
        "stage_complete_fingerprint",
        "stage_config_sha256",
        "decision_rule_sha256",
        "protocol_freeze_sha256",
        "training_support",
        "efficiency",
        "selected_thresholds",
        "method_order",
        "methods",
        "development_mechanism_screen",
        "conclusion",
        "scope_note",
    }


def _selected_thresholds(calibration: Mapping[str, Any]) -> dict[str, float | None]:
    methods = calibration.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(METHOD_ORDER):
        raise ValueError("Stage-A calibration methods are not canonical")
    result: dict[str, float | None] = {}
    for method in METHOD_ORDER:
        payload = methods[method]
        if not isinstance(payload, Mapping):
            raise TypeError(f"calibration method {method} must be a mapping")
        if method == "A":
            value = payload.get("selected_threshold")
        else:
            protocol = payload.get("protocol")
            if not isinstance(protocol, Mapping):
                raise TypeError(f"calibration method {method} lacks a protocol")
            value = protocol.get("selected_threshold")
        if value is None:
            if method in {"A", "Base@B"}:
                raise ValueError(f"{method} requires a numeric threshold")
            result[method] = None
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{method} threshold must be numeric or null")
            threshold = float(value)
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{method} threshold must lie in [0,1]")
            result[method] = threshold
    return result


def _verify_stage_tree(root: Path, complete: Mapping[str, Any]) -> tuple[list[str], dict[str, str], str]:
    directories, files = _tree_inventory(root)
    if complete.get("artifact_directories") != directories:
        raise RuntimeError("Stage-A directory inventory changed")
    if complete.get("artifact_files") != files:
        raise RuntimeError("Stage-A file inventory changed")
    return directories, files, file_sha256(root / _COMPLETE_NAME)


def _load_run_metadata(
    stage_root: Path,
    assessment_path: Path,
    stage_config_path: Path,
    *,
    manifest: object,
    manifest_path: Path,
    verified_base_payload: Mapping[str, str],
    source_digest: str,
) -> _RunMetadata:
    root = _canonical_path_without_links(stage_root, name="Stage-A run", directory=True)
    assessment_file = _canonical_path_without_links(
        assessment_path,
        name="Stage-A assessment",
        directory=False,
    )
    config_file = _canonical_path_without_links(
        stage_config_path,
        name="Stage-A config",
        directory=False,
    )
    if (root / _INCOMPLETE_NAME).exists() or (root / _INCOMPLETE_NAME).is_symlink():
        raise RuntimeError("Stage-A run is incomplete")

    complete = _strict_json(root / _COMPLETE_NAME, name="Stage-A COMPLETE")
    if set(complete) != _complete_keys():
        raise ValueError("Stage-A COMPLETE fields are not canonical")
    if (
        complete["schema_version"] != STAGE_A_RUN_SCHEMA
        or complete["status"] != "complete"
        or complete["method"] != "CURE-Lite"
        or complete["stage"] != "Stage-A"
        or complete["method_order"] != list(METHOD_ORDER)
        or complete["runtime_splits"] != ["D_R", "D_V"]
        or complete["unused_split"] != "D_T"
    ):
        raise ValueError("Stage-A COMPLETE protocol fields are invalid")
    complete_core = dict(complete)
    complete_fingerprint = _digest(
        complete_core.pop("complete_fingerprint", None),
        name="complete_fingerprint",
    )
    if stable_fingerprint(complete_core) != complete_fingerprint:
        raise ValueError("Stage-A COMPLETE fingerprint does not reproduce")
    if complete["source_tree_digest"] != source_digest:
        raise RuntimeError("CURE-Lite method sources differ from the completed run")
    if complete["base_run_identity"] != dict(verified_base_payload):
        raise RuntimeError("Stage-A run binds another reference Base")
    if (
        complete["dataset"] != manifest.dataset
        or complete["manifest_fingerprint"] != manifest.fingerprint
        or complete["manifest_file_sha256"] != file_sha256(manifest_path)
    ):
        raise RuntimeError("Stage-A run differs from the supplied manifest")
    snapshot = _verify_stage_tree(root, complete)

    config = load_stage_a_config(config_file)
    config_receipt = _strict_json(
        root / "receipts" / "config.json",
        name="Stage-A config receipt",
    )
    if config_receipt != _config_receipt(config, source_digest):
        raise RuntimeError("Stage-A config receipt differs from the frozen config")
    if complete["run_config_fingerprint"] != stable_fingerprint(
        config.canonical_payload()
    ):
        raise RuntimeError("Stage-A COMPLETE binds another config")

    calibration = _strict_json(
        root / "receipts" / "calibration.json",
        name="Stage-A calibration receipt",
    )
    results = _strict_json(
        root / "receipts" / "results.json",
        name="Stage-A results receipt",
    )
    support = _strict_json(
        root / "receipts" / "support.json",
        name="Stage-A support receipt",
    )
    efficiency = _strict_json(
        root / "receipts" / "efficiency.json",
        name="Stage-A efficiency receipt",
    )
    if calibration.get("method_order") != list(METHOD_ORDER):
        raise ValueError("Stage-A calibration method order changed")
    if results.get("method_order") != list(METHOD_ORDER):
        raise ValueError("Stage-A results method order changed")
    if calibration.get("receipt_fingerprint") != complete[
        "calibration_receipt_fingerprint"
    ]:
        raise RuntimeError("Stage-A calibration fingerprint differs from COMPLETE")
    if results.get("calibration_receipt_fingerprint") != calibration.get(
        "receipt_fingerprint"
    ):
        raise RuntimeError("Stage-A results bind another calibration receipt")
    if results.get("results_fingerprint") != complete["results_fingerprint"]:
        raise RuntimeError("Stage-A results fingerprint differs from COMPLETE")
    if support.get("support_fingerprint") != complete["support_receipt_fingerprint"]:
        raise RuntimeError("Stage-A support fingerprint differs from COMPLETE")
    if efficiency.get("receipt_fingerprint") != complete[
        "efficiency_receipt_fingerprint"
    ]:
        raise RuntimeError("Stage-A efficiency fingerprint differs from COMPLETE")

    assessment = _strict_json(assessment_file, name="Stage-A assessment")
    if set(assessment) != _assessment_keys():
        raise ValueError("Stage-A assessment fields are not canonical")
    if (
        assessment["schema_version"] != ASSESSMENT_SCHEMA
        or assessment["evaluation_split"] != "D_V"
        or assessment["runtime_splits"] != ["D_R", "D_V"]
        or assessment["unused_split"] != "D_T"
        or assessment["verified_full_replay"] is not True
        or assessment["independent_generalization_result"] is not False
    ):
        raise ValueError("Stage-A assessment protocol fields are invalid")
    if (
        assessment["dataset"] != manifest.dataset
        or assessment["manifest_fingerprint"] != manifest.fingerprint
        or assessment["seed"] != config.training.global_seed
        or assessment["stage_complete_fingerprint"] != complete_fingerprint
        or assessment["stage_config_sha256"] != file_sha256(config_file)
        or assessment["method_order"] != list(METHOD_ORDER)
        or assessment["methods"] != results.get("methods")
        or assessment["training_support"] != support.get("summary")
        or assessment["efficiency"] != efficiency
        or assessment["selected_thresholds"] != _selected_thresholds(calibration)
    ):
        raise RuntimeError("Stage-A assessment differs from its bound run")

    return _RunMetadata(
        root=root,
        assessment_path=assessment_file,
        stage_config_path=config_file,
        complete=complete,
        assessment=assessment,
        config=config,
        calibration=calibration,
        results=results,
        support=support,
        efficiency=efficiency,
        snapshot=snapshot,
        assessment_sha256=file_sha256(assessment_file),
        stage_config_sha256=file_sha256(config_file),
    )


def _tensor_stats(values: Tensor) -> dict[str, float]:
    data = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(-1)
    if data.numel() < 1 or not torch.isfinite(data).all():
        raise ValueError("diagnostic statistic input must be nonempty and finite")
    quantiles = torch.quantile(
        data,
        torch.tensor((0.25, 0.5, 0.75), dtype=torch.float64),
    )
    return {
        "min": float(data.min()),
        "q25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "population_std": float(data.std(unbiased=False)),
    }


def _probability_stats(probability: Tensor, mask: Tensor) -> dict[str, float]:
    selected = probability[mask]
    if selected.numel() < 1:
        raise ValueError("target diagnostic mask is empty")
    return _tensor_stats(selected)


def _feature_descriptor(feature: Tensor, mask: Tensor) -> dict[str, object]:
    source = torch.as_tensor(feature, dtype=torch.float32, device="cpu")
    if source.ndim != 4 or source.shape[0] != 1 or source.shape[1] < 1:
        raise ValueError("feature must have shape [1,C,h,w]")
    target = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
    if target.ndim != 2 or not torch.any(target):
        raise ValueError("feature diagnostic mask must be nonempty [H,W]")
    projected = project_occupancy_to_feature_grid(
        target.unsqueeze(0).unsqueeze(0),
        tuple(int(value) for value in source.shape[-2:]),
    )[0, 0]
    if not torch.any(projected):
        raise RuntimeError("target disappeared on the decoder feature grid")
    cells = source[0].to(torch.float64)
    weights = F.adaptive_avg_pool2d(
        target.to(torch.float64).unsqueeze(0).unsqueeze(0),
        tuple(int(value) for value in source.shape[-2:]),
    )[0, 0]
    weight_sum = weights.sum()
    if float(weight_sum) <= 0.0:
        raise RuntimeError("target has zero feature-grid weight")
    embedding = (cells * weights.unsqueeze(0)).sum(dim=(1, 2)) / weight_sum
    centered = cells - embedding[:, None, None]
    channel_std = torch.sqrt(
        (centered.square() * weights.unsqueeze(0)).sum(dim=(1, 2))
        / weight_sum
    )
    weighted_abs_mean = (
        cells.abs() * weights.unsqueeze(0)
    ).sum() / (weight_sum * cells.shape[0])
    weighted_rms = torch.sqrt(
        (cells.square() * weights.unsqueeze(0)).sum()
        / (weight_sum * cells.shape[0])
    )
    return {
        "feature_cells": int(torch.count_nonzero(projected)),
        "feature_weight_sum": float(weight_sum),
        "feature_embedding": [float(value) for value in embedding.tolist()],
        "feature_channel_weighted_std": [
            float(value) for value in channel_std.tolist()
        ],
        "feature_embedding_l2": float(torch.linalg.vector_norm(embedding)),
        "feature_abs_mean": float(weighted_abs_mean),
        "feature_rms": float(weighted_rms),
    }


def _target_descriptor(
    *,
    sample_id: str,
    gt_instance: Instance,
    base_probability: Tensor,
    feature: Tensor,
    original_occupancy: Tensor,
    conditioning_occupancy: Tensor,
    supervision_mask: Tensor,
) -> dict[str, object]:
    gt_mask = gt_instance.mask
    target = torch.as_tensor(supervision_mask, dtype=torch.bool, device="cpu")
    if target.shape != gt_mask.shape or not torch.any(target):
        raise ValueError("supervision target must be a nonempty subset of the GT grid")
    if torch.any(target & ~gt_mask):
        raise ValueError("supervision target must be contained in its GT component")
    descriptor = _feature_descriptor(feature, target)
    descriptor.update(
        {
            "sample_id": sample_id,
            "gt_id": gt_instance.instance_id,
            "gt_area": gt_instance.area,
            "bbox": list(gt_instance.bbox),
            "centroid": [float(value) for value in gt_instance.centroid],
            "supervision_area": int(torch.count_nonzero(target)),
            "supervision_fraction": (
                int(torch.count_nonzero(target)) / gt_instance.area
            ),
            "original_occupancy_fraction": float(
                torch.count_nonzero(original_occupancy & gt_mask).item()
                / gt_instance.area
            ),
            "conditioning_occupancy_fraction": float(
                torch.count_nonzero(conditioning_occupancy & gt_mask).item()
                / gt_instance.area
            ),
            "base_gt": _probability_stats(base_probability, gt_mask),
            "base_supervision": _probability_stats(base_probability, target),
        }
    )
    return descriptor


def _alignment_scalar(record: Mapping[str, Any], field: str) -> float:
    if field == "log1p_gt_area":
        return math.log1p(float(record["gt_area"]))
    if field == "base_gt_mean":
        return float(record["base_gt"]["mean"])
    if field == "base_gt_max":
        return float(record["base_gt"]["max"])
    if field == "base_supervision_mean":
        return float(record["base_supervision"]["mean"])
    if field == "base_supervision_max":
        return float(record["base_supervision"]["max"])
    return float(record[field])


def _group_scalar_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("alignment group cannot be empty")
    return {
        field: _tensor_stats(
            torch.tensor(
                [_alignment_scalar(record, field) for record in records],
                dtype=torch.float64,
            )
        )
        for field in SCALAR_ALIGNMENT_FIELDS
    }


def _standardized_mean_difference(
    factual: Sequence[Mapping[str, Any]],
    legal: Sequence[Mapping[str, Any]],
    field: str,
) -> float | None:
    first = torch.tensor(
        [_alignment_scalar(record, field) for record in factual],
        dtype=torch.float64,
    )
    second = torch.tensor(
        [_alignment_scalar(record, field) for record in legal],
        dtype=torch.float64,
    )
    pooled = torch.sqrt(
        (first.var(unbiased=False) + second.var(unbiased=False)) / 2.0
    )
    if float(pooled) == 0.0:
        return None
    return float((first.mean() - second.mean()) / pooled)


def _embedding(record: Mapping[str, Any]) -> Tensor:
    value = torch.tensor(record["feature_embedding"], dtype=torch.float64)
    if value.ndim != 1 or value.numel() < 1 or not torch.isfinite(value).all():
        raise ValueError("feature embedding is invalid")
    return value


def _cosine_distance(first: Tensor, second: Tensor) -> float:
    first_norm = float(torch.linalg.vector_norm(first))
    second_norm = float(torch.linalg.vector_norm(second))
    if first_norm == 0.0 and second_norm == 0.0:
        return 0.0
    if first_norm == 0.0 or second_norm == 0.0:
        return 1.0
    similarity = float(torch.dot(first, second) / (first_norm * second_norm))
    return 1.0 - max(-1.0, min(1.0, similarity))


def _candidate_identity(record: Mapping[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {
        "sample_id": record["sample_id"],
        "gt_id": record["gt_id"],
    }
    if record.get("pred_id") is not None:
        result["pred_id"] = record["pred_id"]
    return result


def _nearest_legal_targets(
    factual: Sequence[Mapping[str, Any]],
    legal: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    combined = tuple(factual) + tuple(legal)
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for field in SCALAR_NEIGHBOUR_FIELDS:
        values = torch.tensor(
            [_alignment_scalar(record, field) for record in combined],
            dtype=torch.float64,
        )
        means[field] = float(values.mean())
        scales[field] = float(values.std(unbiased=False))

    def scalar_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
        squared: list[float] = []
        for field in SCALAR_NEIGHBOUR_FIELDS:
            scale = scales[field]
            if scale == 0.0:
                continue
            difference = (
                _alignment_scalar(first, field)
                - _alignment_scalar(second, field)
            ) / scale
            squared.append(difference * difference)
        return math.sqrt(sum(squared)) if squared else 0.0

    rows: list[dict[str, object]] = []
    for source in factual:
        feature_ranked = sorted(
            (
                _cosine_distance(_embedding(source), _embedding(candidate)),
                str(candidate["sample_id"]),
                int(candidate["gt_id"]),
                int(candidate["pred_id"]),
                candidate,
            )
            for candidate in legal
        )
        scalar_ranked = sorted(
            (
                scalar_distance(source, candidate),
                str(candidate["sample_id"]),
                int(candidate["gt_id"]),
                int(candidate["pred_id"]),
                candidate,
            )
            for candidate in legal
        )
        rows.append(
            {
                "factual_target": _candidate_identity(source),
                "nearest_by_feature_cosine": [
                    {
                        "distance": float(item[0]),
                        "legal_target": _candidate_identity(item[4]),
                    }
                    for item in feature_ranked[:3]
                ],
                "nearest_by_scalar_z_distance": [
                    {
                        "distance": float(item[0]),
                        "legal_target": _candidate_identity(item[4]),
                    }
                    for item in scalar_ranked[:3]
                ],
            }
        )
    return rows


def _d_r_semantic_identity(bundle: LoadedDRCacheBundle) -> dict[str, object]:
    return {
        "manifest_fingerprint": bundle.split_manifest_fingerprint,
        "manifest_file_sha256": bundle.split_manifest_file_sha256,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "base_fingerprint": bundle.base_fingerprint,
        "base_state_fingerprint": bundle.base_state_fingerprint,
        "state_fingerprint": bundle.state_fingerprint,
        "gt_fingerprint": bundle.gt_fingerprint,
        "d_r_base_index_fingerprint": bundle.base_index_fingerprint,
        "occupancy_config": config_to_dict(bundle.occupancy_config),
        "match_config": config_to_dict(bundle.match_config),
        "intervention_config": config_to_dict(bundle.intervention_config),
        "ordered_sample_ids": [row.sample_id for row in bundle.rows],
    }


def _build_d_r_alignment(
    bundle: LoadedDRCacheBundle,
    support_receipt: Mapping[str, Any],
) -> dict[str, object]:
    factual: list[dict[str, object]] = []
    unreachable: list[dict[str, object]] = []
    legal: list[dict[str, object]] = []
    raw_legal_count = 0

    for row in bundle.rows:
        state = row.state
        probability = row.base_output.probability[0, 0]
        feature = row.base_output.feature
        occupancy = state.occupancy
        gt = instances_from_binary_mask(state.gt_labels > 0, connectivity=8, min_area=1)
        pred = instances_from_binary_mask(
            state.pred_labels > 0,
            connectivity=8,
            min_area=1,
        )
        before_small = project_occupancy_to_feature_grid(
            occupancy.unsqueeze(0).unsqueeze(0),
            tuple(int(value) for value in feature.shape[-2:]),
        )
        real_ids = tuple(int(value) for value in state.real_miss_ids.tolist())
        reachable_ids = tuple(
            int(value) for value in state.reachable_miss_ids.tolist()
        )
        for gt_id in real_ids:
            target = gt.by_id(gt_id)
            supervision = target.mask & ~occupancy
            record = _target_descriptor(
                sample_id=row.sample_id,
                gt_instance=target,
                base_probability=probability,
                feature=feature,
                original_occupancy=occupancy,
                conditioning_occupancy=occupancy,
                supervision_mask=supervision,
            )
            record.update(
                {
                    "role": (
                        "reachable_factual_miss"
                        if gt_id in reachable_ids
                        else "unreachable_factual_miss"
                    ),
                    "pred_id": None,
                    "deleted_feature_cells": 0,
                }
            )
            if gt_id in reachable_ids:
                factual.append(record)
            else:
                unreachable.append(record)

        for pair in state.legal_pairs.tolist():
            gt_id, pred_id = (int(pair[0]), int(pair[1]))
            raw_legal_count += 1
            target = gt.by_id(gt_id)
            component = pred.by_id(pred_id)
            after = occupancy & ~component.mask
            after_small = project_occupancy_to_feature_grid(
                after.unsqueeze(0).unsqueeze(0),
                tuple(int(value) for value in feature.shape[-2:]),
            )
            changed = before_small ^ after_small
            if not torch.any(changed):
                continue
            supervision = target.mask & ~after
            record = _target_descriptor(
                sample_id=row.sample_id,
                gt_instance=target,
                base_probability=probability,
                feature=feature,
                original_occupancy=occupancy,
                conditioning_occupancy=after,
                supervision_mask=supervision,
            )
            record.update(
                {
                    "role": "decoder_visible_legal_target",
                    "pred_id": pred_id,
                    "pred_area": component.area,
                    "pred_bbox": list(component.bbox),
                    "pred_base": _probability_stats(probability, component.mask),
                    "gt_pred_centroid_distance": centroid_distance(target, component),
                    "gt_pred_iou": mask_iou(target.mask, component.mask),
                    "deleted_feature_cells": int(torch.count_nonzero(changed)),
                }
            )
            legal.append(record)

    factual.sort(key=lambda item: (str(item["sample_id"]), int(item["gt_id"])))
    unreachable.sort(
        key=lambda item: (str(item["sample_id"]), int(item["gt_id"]))
    )
    legal.sort(
        key=lambda item: (
            str(item["sample_id"]),
            int(item["gt_id"]),
            int(item["pred_id"]),
        )
    )
    expected = {
        "real_miss_targets": len(factual) + len(unreachable),
        "reachable_miss_targets": len(factual),
        "legal_candidates": raw_legal_count,
        "decoder_visible_legal_candidates": len(legal),
    }
    for name, value in expected.items():
        if support_receipt.get(name) != value:
            raise RuntimeError(f"D_R diagnostic count differs from support receipt: {name}")
    if not factual or not legal:
        raise RuntimeError("D_R alignment requires both factual and legal targets")

    factual_embeddings = torch.stack([_embedding(record) for record in factual])
    legal_embeddings = torch.stack([_embedding(record) for record in legal])
    factual_centroid = factual_embeddings.mean(dim=0)
    legal_centroid = legal_embeddings.mean(dim=0)
    comparison = {
        field: {
            "factual_minus_legal_standardized_mean_difference": (
                _standardized_mean_difference(factual, legal, field)
            )
        }
        for field in SCALAR_ALIGNMENT_FIELDS
    }
    return {
        "schema_version": DR_ALIGNMENT_SCHEMA,
        "comparison_groups": {
            "factual": "reachable factual misses used by F/F×/U",
            "legal": "decoder-visible legal covered targets available to U",
        },
        "counts": {
            "reachable_factual_targets": len(factual),
            "unreachable_factual_targets": len(unreachable),
            "raw_legal_targets": raw_legal_count,
            "decoder_visible_legal_targets": len(legal),
            "images_with_both_primary_groups": len(
                {
                    str(record["sample_id"]) for record in factual
                }
                & {
                    str(record["sample_id"]) for record in legal
                }
            ),
        },
        "group_scalar_summaries": {
            "reachable_factual": _group_scalar_summary(factual),
            "decoder_visible_legal": _group_scalar_summary(legal),
        },
        "scalar_comparison": comparison,
        "feature_centroid_comparison": {
            "channels": int(factual_centroid.numel()),
            "factual_centroid": [
                float(value) for value in factual_centroid.tolist()
            ],
            "legal_centroid": [float(value) for value in legal_centroid.tolist()],
            "cosine_distance": _cosine_distance(
                factual_centroid,
                legal_centroid,
            ),
        },
        "nearest_legal_targets": _nearest_legal_targets(factual, legal),
        "reachable_factual_targets": factual,
        "unreachable_factual_targets": unreachable,
        "decoder_visible_legal_targets": legal,
    }


def _prepare_d_v_rows(
    bundle: LoadedDVCacheBundle,
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
) -> tuple[_PreparedDVRow, ...]:
    rows: list[_PreparedDVRow] = []
    for row in bundle.rows:
        base = row.base_output.probability[0, 0].contiguous()
        occupancy, anchor_instances = build_occupancy(
            row.base_output.probability,
            occupancy_config,
        )
        gt_instances = instances_from_binary_mask(
            row.gt_mask,
            connectivity=8,
            min_area=1,
        )
        rows.append(
            _PreparedDVRow(
                sample_id=row.sample_id,
                base_probability=base,
                feature=row.base_output.feature,
                occupancy=occupancy,
                anchor_instances=anchor_instances,
                gt_instances=gt_instances,
                anchor_match=match_components(
                    anchor_instances,
                    gt_instances,
                    match_config,
                ),
            )
        )
    return tuple(rows)


def _pair_payload(pair: MatchPair, pred: InstanceMap) -> dict[str, object]:
    component = pred.by_id(pair.pred_id)
    return {
        "pred_id": pair.pred_id,
        "pred_area": component.area,
        "distance": float(pair.distance),
        "iou": float(pair.iou),
    }


def _build_method_outcomes(
    prepared_rows: Sequence[_PreparedDVRow],
    method_run: LoadedDVMethodRun,
    *,
    threshold: float | None,
    match_config: MatchConfig,
    budget: object,
) -> tuple[
    tuple[_MethodImageOutcome, ...],
    dict[str, float | bool],
    list[dict[str, object]],
]:
    samples = {sample.sample_id: sample for sample in method_run.residual_samples}
    if set(samples) != {row.sample_id for row in prepared_rows}:
        raise RuntimeError("D_V method samples differ from prepared rows")
    outcomes: list[_MethodImageOutcome] = []
    unmatched_components: list[dict[str, object]] = []
    matched_gt = total_gt = unmatched_pixels = unmatched_count = 0
    raw_background = total_pixels = intersection = union = retained = covered = 0
    image_ious: list[float] = []

    for row in prepared_rows:
        base, residual, gt_mask = samples[row.sample_id].normalized()
        if not torch.equal(base, row.base_probability):
            raise RuntimeError("D_V method base probability changed")
        if threshold is None:
            residual_mask = torch.zeros_like(row.occupancy)
        else:
            residual_mask = (residual >= threshold) & ~row.occupancy
        prediction = row.occupancy | residual_mask
        pred = instances_from_binary_mask(
            prediction,
            connectivity=8,
            min_area=1,
        )
        match = match_components(pred, row.gt_instances, match_config)
        outcome = _MethodImageOutcome(
            sample_id=row.sample_id,
            residual_probability=residual,
            residual_mask=residual_mask,
            prediction=prediction,
            pred_instances=pred,
            match=match,
        )
        outcomes.append(outcome)

        matched_gt += match.cardinality
        total_gt += len(row.gt_instances.ids)
        anchor_covered = row.anchor_match.matched_gt_ids
        retained += len(match.matched_gt_ids & anchor_covered)
        covered += len(anchor_covered)
        for pred_id in sorted(match.unmatched_pred_ids):
            component = pred.by_id(pred_id)
            area = component.area
            unmatched_pixels += area
            unmatched_count += 1
            unmatched_components.append(
                {
                    "sample_id": row.sample_id,
                    "pred_id": pred_id,
                    "area": area,
                    "bbox": list(component.bbox),
                    "background_pixels": int(
                        torch.count_nonzero(component.mask & ~gt_mask)
                    ),
                    "residual_pixels": int(
                        torch.count_nonzero(component.mask & residual_mask)
                    ),
                }
            )
        raw_background += int(torch.count_nonzero(prediction & ~gt_mask))
        total_pixels += prediction.numel()
        image_intersection = int(torch.count_nonzero(prediction & gt_mask))
        image_union = int(torch.count_nonzero(prediction | gt_mask))
        intersection += image_intersection
        union += image_union
        image_ious.append(
            image_intersection / image_union if image_union else 1.0
        )

    metrics: dict[str, float | bool] = {
        "pd": matched_gt / total_gt if total_gt else 1.0,
        "miou": intersection / union if union else 1.0,
        "niou": sum(image_ious) / len(image_ious),
        "pixel_fa": unmatched_pixels / total_pixels if total_pixels else 0.0,
        "fp_components_per_mp": (
            unmatched_count / (total_pixels / 1_000_000.0)
            if total_pixels
            else 0.0
        ),
        "raw_background_fa": (
            raw_background / total_pixels if total_pixels else 0.0
        ),
        "retention": retained / covered if covered else 1.0,
        "budget_violation": False,
    }
    metrics["budget_violation"] = not (
        float(metrics["pixel_fa"]) <= float(budget.pixel_fa_budget)
        and float(metrics["fp_components_per_mp"])
        <= float(budget.component_fa_per_mp_budget)
        and float(metrics["raw_background_fa"])
        <= float(budget.raw_background_fa_budget)
        and float(metrics["retention"]) >= float(budget.minimum_retention)
    )
    unmatched_components.sort(
        key=lambda item: (str(item["sample_id"]), int(item["pred_id"]))
    )
    return tuple(outcomes), metrics, unmatched_components


def _residual_target_stats(
    residual: Tensor,
    gt_mask: Tensor,
    residual_mask: Tensor,
) -> dict[str, object]:
    result: dict[str, object] = _probability_stats(residual, gt_mask)
    result.update(
        {
            "active_pixels_in_gt": int(
                torch.count_nonzero(residual_mask & gt_mask)
            ),
            "active_fraction_in_gt": float(
                torch.count_nonzero(residual_mask & gt_mask).item()
                / int(torch.count_nonzero(gt_mask))
            ),
        }
    )
    return result


def _partition_summary(
    targets: Sequence[Mapping[str, Any]],
    *,
    anchor_misses_only: bool,
) -> dict[str, object]:
    selected = [
        target
        for target in targets
        if not anchor_misses_only or bool(target["anchor_miss"])
    ]
    counts = {
        category: sum(target["category"] == category for target in selected)
        for category in CATEGORIES
    }
    total = len(selected)
    if sum(counts.values()) != total:
        raise RuntimeError("D_V target categories do not partition the selected set")
    return {
        "target_scope": "anchor_misses" if anchor_misses_only else "all_gt",
        "total_targets": total,
        "counts": counts,
        "fractions": {
            category: counts[category] / total if total else 0.0
            for category in CATEGORIES
        },
        "f_matched_targets": counts["f_only"] + counts["both"],
        "u_matched_targets": counts["u_only"] + counts["both"],
    }


def _build_d_v_partition(
    prepared_rows: Sequence[_PreparedDVRow],
    f_outcomes: Sequence[_MethodImageOutcome],
    u_outcomes: Sequence[_MethodImageOutcome],
    *,
    f_threshold: float | None,
    u_threshold: float | None,
) -> dict[str, object]:
    if not (
        len(prepared_rows) == len(f_outcomes) == len(u_outcomes)
    ):
        raise RuntimeError("D_V prepared rows and method outcomes differ in length")
    targets: list[dict[str, object]] = []
    image_membership: list[dict[str, object]] = []
    for row, factual, uniform in zip(
        prepared_rows,
        f_outcomes,
        u_outcomes,
        strict=True,
    ):
        if row.sample_id != factual.sample_id or row.sample_id != uniform.sample_id:
            raise RuntimeError("D_V F/U sample order changed")
        f_pairs = {pair.gt_id: pair for pair in factual.match.pairs}
        u_pairs = {pair.gt_id: pair for pair in uniform.match.pairs}
        image_membership.append(
            {
                "sample_id": row.sample_id,
                "gt_ids": list(row.gt_instances.ids),
                "anchor_matched_gt_ids": sorted(row.anchor_match.matched_gt_ids),
                "f_matched_gt_ids": sorted(factual.match.matched_gt_ids),
                "u_matched_gt_ids": sorted(uniform.match.matched_gt_ids),
            }
        )
        for target in row.gt_instances.instances:
            f_hit = target.instance_id in factual.match.matched_gt_ids
            u_hit = target.instance_id in uniform.match.matched_gt_ids
            category = {
                (True, False): "f_only",
                (False, True): "u_only",
                (True, True): "both",
                (False, False): "neither",
            }[(f_hit, u_hit)]
            descriptor = _target_descriptor(
                sample_id=row.sample_id,
                gt_instance=target,
                base_probability=row.base_probability,
                feature=row.feature,
                original_occupancy=row.occupancy,
                conditioning_occupancy=row.occupancy,
                supervision_mask=target.mask,
            )
            descriptor.update(
                {
                    "category": category,
                    "anchor_miss": (
                        target.instance_id in row.anchor_match.unmatched_gt_ids
                    ),
                    "f_matched": f_hit,
                    "u_matched": u_hit,
                    "f_pair": (
                        _pair_payload(
                            f_pairs[target.instance_id],
                            factual.pred_instances,
                        )
                        if f_hit
                        else None
                    ),
                    "u_pair": (
                        _pair_payload(
                            u_pairs[target.instance_id],
                            uniform.pred_instances,
                        )
                        if u_hit
                        else None
                    ),
                    "f_residual": _residual_target_stats(
                        factual.residual_probability,
                        target.mask,
                        factual.residual_mask,
                    ),
                    "u_residual": _residual_target_stats(
                        uniform.residual_probability,
                        target.mask,
                        uniform.residual_mask,
                    ),
                }
            )
            targets.append(descriptor)
    targets.sort(key=lambda item: (str(item["sample_id"]), int(item["gt_id"])))
    image_membership.sort(key=lambda item: str(item["sample_id"]))
    return {
        "schema_version": DV_PARTITION_SCHEMA,
        "selected_thresholds": {"F": f_threshold, "U": u_threshold},
        "all_targets": _partition_summary(targets, anchor_misses_only=False),
        "anchor_miss_targets": _partition_summary(
            targets,
            anchor_misses_only=True,
        ),
        "image_membership": image_membership,
        "targets": targets,
    }


def _method_receipt(
    calibration: Mapping[str, Any],
    method: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    methods = calibration["methods"]
    receipt = methods[method]
    protocol = receipt["protocol"]
    if not isinstance(receipt, Mapping) or not isinstance(protocol, Mapping):
        raise TypeError(f"calibration {method} receipt is invalid")
    return receipt, protocol


def _verify_method_run(
    run: LoadedDVMethodRun,
    artifact: LoadedDecoderArtifact,
    calibration: Mapping[str, Any],
    method: str,
) -> None:
    receipt, protocol = _method_receipt(calibration, method)
    expected_variant = "factual_only" if method == "F" else "uniform_legal"
    if (
        artifact.config.variant != expected_variant
        or receipt.get("decoder_variant") != expected_variant
        or receipt.get("global_seed") != artifact.config.global_seed
        or receipt.get("decoder_artifact_fingerprint")
        != artifact.artifact_fingerprint
        or receipt.get("decoder_state_fingerprint")
        != artifact.decoder_state_fingerprint
        or receipt.get("decoder_receipt_sha256") != artifact.receipt_sha256
        or receipt.get("d_v_run_fingerprint") != run.run_fingerprint
        or protocol.get("sample_tensor_fingerprint")
        != run.residual_samples_fingerprint
    ):
        raise RuntimeError(f"{method} selected-point run differs from calibration")


def _run_payload(
    metadata: _RunMetadata,
    d_r_bundle: LoadedDRCacheBundle,
    d_v_bundle: LoadedDVCacheBundle,
    prepared_rows: Sequence[_PreparedDVRow],
) -> dict[str, object]:
    artifacts = {
        "F": load_decoder_artifact(metadata.root / "decoders" / "factual_only"),
        "U": load_decoder_artifact(metadata.root / "decoders" / "uniform_legal"),
    }
    _verify_artifact_training_binding(
        artifacts["F"],
        expected_variant="factual_only",
        bundle=d_r_bundle,
        config=metadata.config,
    )
    _verify_artifact_training_binding(
        artifacts["U"],
        expected_variant="uniform_legal",
        bundle=d_r_bundle,
        config=metadata.config,
    )
    if (
        artifacts["F"].config.initial_decoder_fingerprint
        != artifacts["U"].config.initial_decoder_fingerprint
    ):
        raise RuntimeError("F and U do not share their initial decoder")
    if (
        metadata.complete["factual_decoder_artifact_fingerprint"]
        != artifacts["F"].artifact_fingerprint
        or metadata.complete["uniform_decoder_artifact_fingerprint"]
        != artifacts["U"].artifact_fingerprint
    ):
        raise RuntimeError("Stage-A COMPLETE binds other F/U artifacts")

    print(
        json.dumps(
            {
                "event": "selected_point_inference_start",
                "seed": metadata.seed,
                "methods": ["F", "U"],
                "d_v_samples": len(d_v_bundle.rows),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    f_run = build_loaded_d_v_method_run(d_v_bundle, artifacts["F"])
    u_run = build_loaded_d_v_method_run(d_v_bundle, artifacts["U"])
    _verify_method_run(f_run, artifacts["F"], metadata.calibration, "F")
    _verify_method_run(u_run, artifacts["U"], metadata.calibration, "U")
    thresholds = _selected_thresholds(metadata.calibration)
    f_outcomes, f_metrics, f_unmatched = _build_method_outcomes(
        prepared_rows,
        f_run,
        threshold=thresholds["F"],
        match_config=metadata.config.match_config,
        budget=metadata.config.budget,
    )
    u_outcomes, u_metrics, u_unmatched = _build_method_outcomes(
        prepared_rows,
        u_run,
        threshold=thresholds["U"],
        match_config=metadata.config.match_config,
        budget=metadata.config.budget,
    )
    expected_methods = metadata.results["methods"]
    if f_metrics != expected_methods["F"]:
        raise RuntimeError("F selected-point metrics do not reproduce")
    if u_metrics != expected_methods["U"]:
        raise RuntimeError("U selected-point metrics do not reproduce")
    partition = _build_d_v_partition(
        prepared_rows,
        f_outcomes,
        u_outcomes,
        f_threshold=thresholds["F"],
        u_threshold=thresholds["U"],
    )
    all_targets = partition["all_targets"]
    if (
        all_targets["f_matched_targets"]
        != round(float(f_metrics["pd"]) * all_targets["total_targets"])
        or all_targets["u_matched_targets"]
        != round(float(u_metrics["pd"]) * all_targets["total_targets"])
    ):
        raise RuntimeError("D_V target partition disagrees with selected Pd")

    for artifact in artifacts.values():
        artifact.verify_unchanged()
    d_r_bundle.verify_unchanged()
    print(
        json.dumps(
            {
                "event": "selected_point_inference_complete",
                "seed": metadata.seed,
                "target_counts": partition["anchor_miss_targets"]["counts"],
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    return {
        "seed": metadata.seed,
        "stage_run": str(metadata.root),
        "assessment_path": str(metadata.assessment_path),
        "stage_config_path": str(metadata.stage_config_path),
        "stage_complete_fingerprint": metadata.complete["complete_fingerprint"],
        "stage_complete_file_sha256": metadata.snapshot[2],
        "assessment_file_sha256": metadata.assessment_sha256,
        "stage_config_file_sha256": metadata.stage_config_sha256,
        "run_config_fingerprint": metadata.complete["run_config_fingerprint"],
        "calibration_receipt_fingerprint": metadata.calibration[
            "receipt_fingerprint"
        ],
        "results_fingerprint": metadata.results["results_fingerprint"],
        "assessment_conclusion": metadata.assessment["conclusion"],
        "selected_thresholds": {"F": thresholds["F"], "U": thresholds["U"]},
        "decoder_bindings": {
            method: {
                "variant": artifacts[method].config.variant,
                "artifact_fingerprint": artifacts[method].artifact_fingerprint,
                "decoder_state_fingerprint": (
                    artifacts[method].decoder_state_fingerprint
                ),
                "receipt_sha256": artifacts[method].receipt_sha256,
                "d_v_run_fingerprint": (
                    f_run.run_fingerprint if method == "F" else u_run.run_fingerprint
                ),
                "residual_samples_fingerprint": (
                    f_run.residual_samples_fingerprint
                    if method == "F"
                    else u_run.residual_samples_fingerprint
                ),
            }
            for method in ("F", "U")
        },
        "selected_metrics": {"F": f_metrics, "U": u_metrics},
        "unmatched_components": {"F": f_unmatched, "U": u_unmatched},
        "d_v_partition": partition,
    }


def _cross_seed_payload(runs: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if not runs:
        raise ValueError("at least one diagnostic run is required")
    targets_by_seed: dict[int, dict[tuple[str, int], Mapping[str, Any]]] = {}
    for run in runs:
        seed = int(run["seed"])
        targets = run["d_v_partition"]["targets"]
        mapping = {
            (str(target["sample_id"]), int(target["gt_id"])): target
            for target in targets
        }
        if len(mapping) != len(targets):
            raise RuntimeError("D_V target identities are not unique")
        targets_by_seed[seed] = mapping
    ordered_seeds = sorted(targets_by_seed)
    identities = tuple(sorted(targets_by_seed[ordered_seeds[0]]))
    if any(tuple(sorted(targets_by_seed[seed])) != identities for seed in ordered_seeds):
        raise RuntimeError("D_V target identities differ across seeds")
    transitions: list[dict[str, object]] = []
    transition_counts: dict[str, int] = {}
    for sample_id, gt_id in identities:
        categories = {
            str(seed): targets_by_seed[seed][(sample_id, gt_id)]["category"]
            for seed in ordered_seeds
        }
        key = " -> ".join(categories[str(seed)] for seed in ordered_seeds)
        transition_counts[key] = transition_counts.get(key, 0) + 1
        transitions.append(
            {
                "sample_id": sample_id,
                "gt_id": gt_id,
                "anchor_miss": bool(
                    targets_by_seed[ordered_seeds[0]][(sample_id, gt_id)][
                        "anchor_miss"
                    ]
                ),
                "categories": categories,
            }
        )
    return {
        "ordered_seeds": ordered_seeds,
        "shared_target_count": len(identities),
        "transition_counts": dict(sorted(transition_counts.items())),
        "target_transitions": transitions,
    }


def _verify_snapshot(metadata: _RunMetadata) -> None:
    current = _verify_stage_tree(metadata.root, metadata.complete)
    if current != metadata.snapshot:
        raise RuntimeError("Stage-A run changed during diagnosis")
    if file_sha256(metadata.assessment_path) != metadata.assessment_sha256:
        raise RuntimeError("Stage-A assessment changed during diagnosis")
    if file_sha256(metadata.stage_config_path) != metadata.stage_config_sha256:
        raise RuntimeError("Stage-A config changed during diagnosis")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_arguments = tuple(tuple(group) for group in args.runs)
    stage_roots = tuple(
        _canonical_path_without_links(
            group[0],
            name="Stage-A run",
            directory=True,
        )
        for group in run_arguments
    )
    if len(set(stage_roots)) != len(stage_roots):
        raise ValueError("Stage-A run arguments must be unique")
    output = _prepare_output(args.output, stage_roots)
    manifest_path = _canonical_path_without_links(
        args.manifest,
        name="split manifest",
        directory=False,
    )
    reference_base_root = _canonical_path_without_links(
        args.reference_base_run,
        name="reference Base run",
        directory=True,
    )
    manifest = load_and_validate_manifest(manifest_path)
    verified_base = load_verified_reference_base_run_identity(reference_base_root)
    verified_base_payload = _verified_base_run_payload(verified_base)
    source_digest = _source_tree_digest()

    metadata = tuple(
        _load_run_metadata(
            group[0],
            group[1],
            group[2],
            manifest=manifest,
            manifest_path=manifest_path,
            verified_base_payload=verified_base_payload,
            source_digest=source_digest,
        )
        for group in run_arguments
    )
    metadata = tuple(sorted(metadata, key=lambda item: item.seed))
    if len({item.seed for item in metadata}) != len(metadata):
        raise ValueError("Stage-A diagnostic seeds must be unique")

    first = metadata[0]
    first_contract = load_base_cache_pair_contract(
        first.root / "d_r" / "base_cache" / "index.json",
        first.root / "d_v" / "base_cache" / "index.json",
    )
    d_r_dataset = ManifestImageDataset(
        manifest,
        "D_R",
        first_contract.preprocessing,
        manifest_path=manifest_path,
    )
    d_v_dataset = ManifestImageDataset(
        manifest,
        "D_V",
        first_contract.preprocessing,
        manifest_path=manifest_path,
    )
    _check_dataset_pair(d_r_dataset, d_v_dataset)
    shared_d_v = load_d_v_cache_bundle(
        first.root / "d_v" / "base_cache" / "index.json",
        d_v_dataset,
        expected_base_fingerprint=verified_base_payload["base_fingerprint"],
    )
    if (
        shared_d_v.base_index_fingerprint
        != first.complete["d_v_base_index_fingerprint"]
    ):
        raise RuntimeError("D_V cache differs from Stage-A COMPLETE")

    run_payloads: list[dict[str, object]] = []
    shared_d_r_identity: dict[str, object] | None = None
    d_r_alignment: dict[str, object] | None = None
    prepared_rows: tuple[_PreparedDVRow, ...] | None = None
    for index, item in enumerate(metadata):
        contract = load_base_cache_pair_contract(
            item.root / "d_r" / "base_cache" / "index.json",
            item.root / "d_v" / "base_cache" / "index.json",
        )
        if (
            contract.preprocessing != first_contract.preprocessing
            or contract.split_manifest_fingerprint
            != first_contract.split_manifest_fingerprint
            or contract.base_fingerprint != first_contract.base_fingerprint
            or contract.base_state_fingerprint
            != first_contract.base_state_fingerprint
            or contract.d_v_index_fingerprint
            != shared_d_v.base_index_fingerprint
        ):
            raise RuntimeError("Stage-A runs do not share one D_R/D_V contract")
        d_r_bundle = load_d_r_cache_bundle(
            item.root / "d_r" / "state_cache" / "index.json",
            d_r_dataset,
            expected_base_fingerprint=verified_base_payload["base_fingerprint"],
        )
        _require_same_base_cache_identity(d_r_bundle, shared_d_v)
        if (
            d_r_bundle.base_index_fingerprint
            != item.complete["d_r_base_index_fingerprint"]
            or d_r_bundle.state_index_fingerprint
            != item.complete["d_r_state_index_fingerprint"]
        ):
            raise RuntimeError("D_R cache differs from Stage-A COMPLETE")
        semantic_identity = _d_r_semantic_identity(d_r_bundle)
        if shared_d_r_identity is None:
            shared_d_r_identity = semantic_identity
            support_summary = item.support.get("summary")
            if not isinstance(support_summary, Mapping):
                raise TypeError("Stage-A support summary is invalid")
            d_r_alignment = _build_d_r_alignment(
                d_r_bundle,
                support_summary,
            )
        elif semantic_identity != shared_d_r_identity:
            raise RuntimeError("Stage-A runs use different D_R semantic states")
        if prepared_rows is None:
            prepared_rows = _prepare_d_v_rows(
                shared_d_v,
                d_r_bundle.occupancy_config,
                item.config.match_config,
            )
        run_payloads.append(
            _run_payload(
                item,
                d_r_bundle,
                shared_d_v,
                prepared_rows,
            )
        )
        print(
            json.dumps(
                {
                    "event": "diagnostic_run_complete",
                    "completed": index + 1,
                    "total": len(metadata),
                    "seed": item.seed,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    if shared_d_r_identity is None or d_r_alignment is None or prepared_rows is None:
        raise AssertionError("diagnostic inputs unexpectedly empty")
    for item in metadata:
        _verify_snapshot(item)
    shared_d_v.verify_unchanged()
    verified_base.verify_unchanged()
    if _source_tree_digest() != source_digest:
        raise RuntimeError("CURE-Lite method sources changed during diagnosis")

    core: dict[str, object] = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "method": "CURE-Lite",
        "stage": "Stage-A",
        "role": "development_mechanism_diagnosis",
        "runtime_splits": ["D_R", "D_V"],
        "unused_split": "D_T",
        "independent_generalization_result": False,
        "training_entrypoint_called": False,
        "optimizer_updates_during_this_operation": 0,
        "candidate_thresholds_evaluated_during_this_operation": 0,
        "selected_methods_recomputed": ["F", "U"],
        "manifest_path": str(manifest_path),
        "manifest_fingerprint": manifest.fingerprint,
        "manifest_file_sha256": file_sha256(manifest_path),
        "reference_base_run": str(reference_base_root),
        "reference_base_run_identity": dict(verified_base_payload),
        "method_source_tree_digest": source_digest,
        "diagnostic_tool_sha256": file_sha256(Path(__file__).resolve(strict=True)),
        "shared_d_v_identity": {
            "sample_count": len(shared_d_v.rows),
            "base_index_fingerprint": shared_d_v.base_index_fingerprint,
            "base_index_sha256": shared_d_v.base_index_sha256,
            "d_v_image_fingerprint": shared_d_v.d_v_image_fingerprint,
            "d_v_gt_fingerprint": shared_d_v.d_v_gt_fingerprint,
            "ordered_sample_ids": [row.sample_id for row in shared_d_v.rows],
        },
        "shared_d_r_semantic_identity": shared_d_r_identity,
        "d_r_alignment": d_r_alignment,
        "runs": run_payloads,
        "cross_seed": _cross_seed_payload(run_payloads),
    }
    payload = {**core, "diagnostic_fingerprint": stable_fingerprint(core)}
    _write_new_json(output, payload)
    print(
        json.dumps(
            {
                "diagnostic_fingerprint": payload["diagnostic_fingerprint"],
                "diagnostic_output": str(output),
                "d_r_target_counts": d_r_alignment["counts"],
                "runs": [
                    {
                        "seed": run["seed"],
                        "target_categories": run["d_v_partition"][
                            "anchor_miss_targets"
                        ]["counts"],
                    }
                    for run in run_payloads
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
