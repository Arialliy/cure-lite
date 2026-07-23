"""Decoder-free selection of the frozen CURE-Lite occupancy anchor.

The occupancy threshold ``tau_o`` is a property of the frozen base detector,
not of a trained residual decoder.  This module therefore binds anchor
selection directly to one verified D_V base-cache bundle.  The only permitted
selection rule maximizes dataset-level (global) base-only mIoU and resolves a
tie in favour of the higher threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Literal

import torch

from ..cache.schema import stable_fingerprint
from ..calibration import (
    CalibrationSample,
    evaluate_base_threshold,
    select_anchor_threshold_by_miou,
)
from ..config import MatchConfig, OccupancyConfig
from ..metrics import AggregateEvaluation, formal_stage_a_metrics_payload
from ..splits import SplitManifest
from .cache_pipeline import LoadedDVCacheBundle
from .evaluation_pipeline import (
    DevelopmentSplitAccess,
    calibration_samples_fingerprint,
)


ANCHOR_SELECTION_RULE = "max_global_miou_tie_higher_threshold"


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _canonical_threshold_grid(values: Iterable[float]) -> tuple[float, ...]:
    resolved: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("anchor threshold candidates may not be bool")
        try:
            threshold = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("anchor thresholds must be real numbers") from error
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("anchor thresholds must be finite and in [0,1]")
        resolved.append(threshold)
    grid = tuple(sorted(set(resolved)))
    if not grid:
        raise ValueError("anchor threshold candidates are empty")
    return grid


@dataclass(frozen=True, slots=True)
class _LoadedDVBaseRunSeal:
    bundle: LoadedDVCacheBundle
    access: DevelopmentSplitAccess
    base_samples: tuple[CalibrationSample, ...]
    base_samples_fingerprint: str
    ordered_d_v_sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class LoadedDVBaseRun:
    """Exact base-only tensors from one verified D_V cache bundle."""

    bundle: LoadedDVCacheBundle
    access: DevelopmentSplitAccess
    base_samples: tuple[CalibrationSample, ...]
    base_samples_fingerprint: str
    ordered_d_v_sample_ids: tuple[str, ...]
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _LoadedDVBaseRunSeal:
            raise TypeError("LoadedDVBaseRun must come from the strict base builder")
        if (
            seal.bundle is not self.bundle
            or seal.access is not self.access
            or seal.base_samples is not self.base_samples
            or seal.base_samples_fingerprint != self.base_samples_fingerprint
            or seal.ordered_d_v_sample_ids != self.ordered_d_v_sample_ids
        ):
            raise TypeError("loaded D_V base-run source fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        _digest(self.base_samples_fingerprint, name="base_samples_fingerprint")
        if not isinstance(self.bundle, LoadedDVCacheBundle):
            raise TypeError("bundle must be a LoadedDVCacheBundle")
        if not isinstance(self.access, DevelopmentSplitAccess):
            raise TypeError("access must be DevelopmentSplitAccess")
        if not isinstance(self.base_samples, tuple) or not self.base_samples:
            raise ValueError("base_samples must be a non-empty tuple")
        if any(
            not isinstance(sample, CalibrationSample) for sample in self.base_samples
        ):
            raise TypeError("base_samples contain an invalid item")
        if (
            not isinstance(self.ordered_d_v_sample_ids, tuple)
            or not self.ordered_d_v_sample_ids
            or len(set(self.ordered_d_v_sample_ids))
            != len(self.ordered_d_v_sample_ids)
        ):
            raise ValueError("ordered D_V sample IDs must be non-empty and unique")
        self.verify_unchanged()

    @property
    def run_fingerprint(self) -> str:
        self._verify_source_seal()
        return stable_fingerprint(
            {
                "schema_version": "cure-lite-loaded-d-v-base-run-v1",
                "manifest_fingerprint": self.bundle.split_manifest_fingerprint,
                "manifest_file_sha256": self.bundle.split_manifest_file_sha256,
                "base_index_fingerprint": self.bundle.base_index_fingerprint,
                "base_index_sha256": self.bundle.base_index_sha256,
                "d_v_image_fingerprint": self.bundle.d_v_image_fingerprint,
                "d_v_gt_fingerprint": self.bundle.d_v_gt_fingerprint,
                "preprocessing_fingerprint": self.bundle.preprocessing_fingerprint,
                "base_fingerprint": self.bundle.base_fingerprint,
                "ordered_d_v_sample_ids": self.ordered_d_v_sample_ids,
                "base_samples_fingerprint": self.base_samples_fingerprint,
            }
        )

    def verify_unchanged(self) -> None:
        """Recheck the persisted bundle and all base-only tensors."""

        self._verify_source_seal()
        self.bundle.verify_unchanged()
        if self.access.manifest.fingerprint != self.bundle.split_manifest_fingerprint:
            raise RuntimeError("D_V access manifest differs from the base bundle")
        expected_ids = tuple(
            record.sample_id for record in self.access.records_for("D_V")
        )
        if expected_ids != self.ordered_d_v_sample_ids:
            raise RuntimeError("ordered D_V sample membership changed")
        if tuple(sample.sample_id for sample in self.base_samples) != expected_ids:
            raise RuntimeError("base-only D_V sample order changed")
        if (
            calibration_samples_fingerprint(self.base_samples)
            != self.base_samples_fingerprint
        ):
            raise RuntimeError("base-only D_V tensors changed in memory")


def build_loaded_d_v_base_run(bundle: LoadedDVCacheBundle) -> LoadedDVBaseRun:
    """Build the decoder-free base-only D_V input used to select ``tau_o``."""

    if not isinstance(bundle, LoadedDVCacheBundle):
        raise TypeError("bundle must be a LoadedDVCacheBundle")
    bundle.verify_unchanged()
    manifest = SplitManifest.load(bundle.manifest_path)
    access = DevelopmentSplitAccess(manifest)
    records = access.records_for("D_V")
    ordered_ids = tuple(record.sample_id for record in records)
    rows_by_id = {row.sample_id: row for row in bundle.rows}
    if set(rows_by_id) != set(ordered_ids) or len(rows_by_id) != len(ordered_ids):
        raise RuntimeError("D_V bundle membership differs from its manifest")

    samples: list[CalibrationSample] = []
    for sample_id in ordered_ids:
        row = rows_by_id[sample_id]
        base = row.base_output.probability.detach().to(device="cpu").clone()
        gt = row.gt_mask.detach().to(device="cpu").clone()
        candidate = CalibrationSample(
            sample_id=sample_id,
            base_probability=base,
            residual_probability=torch.zeros_like(base),
            gt_mask=gt,
        )
        normalized_base, normalized_residual, normalized_gt = candidate.normalized()
        samples.append(
            CalibrationSample(
                sample_id,
                normalized_base,
                normalized_residual,
                normalized_gt,
            )
        )
    samples_tuple = tuple(samples)
    sample_fingerprint = calibration_samples_fingerprint(samples_tuple)
    seal = _LoadedDVBaseRunSeal(
        bundle=bundle,
        access=access,
        base_samples=samples_tuple,
        base_samples_fingerprint=sample_fingerprint,
        ordered_d_v_sample_ids=ordered_ids,
    )
    result = LoadedDVBaseRun(
        bundle=bundle,
        access=access,
        base_samples=samples_tuple,
        base_samples_fingerprint=sample_fingerprint,
        ordered_d_v_sample_ids=ordered_ids,
        _verification_token=seal,
    )
    bundle.verify_unchanged()
    return result


@dataclass(frozen=True, slots=True)
class _FrozenAnchorReceiptSeal:
    occupancy_config: OccupancyConfig
    match_config: MatchConfig
    selected_metrics: AggregateEvaluation
    bound_values: tuple[object, ...]


@dataclass(frozen=True)
class FrozenAnchorReceipt:
    """Immutable proof of decoder-free ``tau_o`` selection on exact D_V."""

    selection_rule: Literal["max_global_miou_tie_higher_threshold"]
    candidate_threshold_grid: tuple[float, ...]
    occupancy_config: OccupancyConfig
    match_config: MatchConfig
    selected_metrics: AggregateEvaluation
    d_v_run_fingerprint: str
    ordered_d_v_sample_ids: tuple[str, ...]
    base_samples_fingerprint: str
    manifest_fingerprint: str
    manifest_file_sha256: str
    base_index_fingerprint: str
    base_index_sha256: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _FrozenAnchorReceiptSeal:
            raise TypeError("FrozenAnchorReceipt must come from formal D_V selection")
        if (
            seal.occupancy_config is not self.occupancy_config
            or seal.match_config is not self.match_config
            or seal.selected_metrics is not self.selected_metrics
            or seal.bound_values
            != (
                self.selection_rule,
                self.candidate_threshold_grid,
                self.d_v_run_fingerprint,
                self.ordered_d_v_sample_ids,
                self.base_samples_fingerprint,
                self.manifest_fingerprint,
                self.manifest_file_sha256,
                self.base_index_fingerprint,
                self.base_index_sha256,
                self.d_v_image_fingerprint,
                self.d_v_gt_fingerprint,
                self.preprocessing_fingerprint,
                self.base_fingerprint,
            )
        ):
            raise TypeError("frozen anchor receipt fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        if self.selection_rule != ANCHOR_SELECTION_RULE:
            raise ValueError("unsupported frozen-anchor selection rule")
        if _canonical_threshold_grid(self.candidate_threshold_grid) != (
            self.candidate_threshold_grid
        ):
            raise ValueError("anchor threshold grid must be sorted and unique")
        if not isinstance(self.occupancy_config, OccupancyConfig):
            raise TypeError("occupancy_config must be an OccupancyConfig")
        if self.occupancy_config.threshold not in self.candidate_threshold_grid:
            raise ValueError("selected tau_o is absent from its frozen grid")
        if not isinstance(self.match_config, MatchConfig):
            raise TypeError("match_config must be a MatchConfig")
        if not isinstance(self.selected_metrics, AggregateEvaluation):
            raise TypeError("selected_metrics must be an AggregateEvaluation")
        if (
            not isinstance(self.ordered_d_v_sample_ids, tuple)
            or not self.ordered_d_v_sample_ids
            or len(set(self.ordered_d_v_sample_ids))
            != len(self.ordered_d_v_sample_ids)
        ):
            raise ValueError("ordered D_V sample IDs must be non-empty and unique")
        for name in (
            "d_v_run_fingerprint",
            "base_samples_fingerprint",
            "manifest_fingerprint",
            "manifest_file_sha256",
            "base_index_fingerprint",
            "base_index_sha256",
            "d_v_image_fingerprint",
            "d_v_gt_fingerprint",
            "preprocessing_fingerprint",
            "base_fingerprint",
        ):
            _digest(getattr(self, name), name=name)

    @property
    def selected_threshold(self) -> float:
        """The frozen occupancy threshold, provided as a receipt convenience."""

        self._verify_source_seal()
        return self.occupancy_config.threshold

    def canonical_payload(self) -> dict[str, object]:
        """Return the complete JSON-compatible receipt payload."""

        self._verify_source_seal()
        return {
            "schema_version": "cure-lite-frozen-anchor-receipt-v2",
            "selection_rule": self.selection_rule,
            "candidate_threshold_grid": list(self.candidate_threshold_grid),
            "occupancy_config": asdict(self.occupancy_config),
            "match_config": asdict(self.match_config),
            "selected_threshold": self.selected_threshold,
            "selected_metrics": formal_stage_a_metrics_payload(
                self.selected_metrics
            ),
            "d_v_run_fingerprint": self.d_v_run_fingerprint,
            "ordered_d_v_sample_ids": list(self.ordered_d_v_sample_ids),
            "base_samples_fingerprint": self.base_samples_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "manifest_file_sha256": self.manifest_file_sha256,
            "base_index_fingerprint": self.base_index_fingerprint,
            "base_index_sha256": self.base_index_sha256,
            "d_v_image_fingerprint": self.d_v_image_fingerprint,
            "d_v_gt_fingerprint": self.d_v_gt_fingerprint,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "base_fingerprint": self.base_fingerprint,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _anchor_receipt(
    run: LoadedDVBaseRun,
    grid: tuple[float, ...],
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    selected_metrics: AggregateEvaluation,
) -> FrozenAnchorReceipt:
    bundle = run.bundle
    bound_values: tuple[object, ...] = (
        ANCHOR_SELECTION_RULE,
        grid,
        run.run_fingerprint,
        run.ordered_d_v_sample_ids,
        run.base_samples_fingerprint,
        bundle.split_manifest_fingerprint,
        bundle.split_manifest_file_sha256,
        bundle.base_index_fingerprint,
        bundle.base_index_sha256,
        bundle.d_v_image_fingerprint,
        bundle.d_v_gt_fingerprint,
        bundle.preprocessing_fingerprint,
        bundle.base_fingerprint,
    )
    seal = _FrozenAnchorReceiptSeal(
        occupancy_config=occupancy_config,
        match_config=match_config,
        selected_metrics=selected_metrics,
        bound_values=bound_values,
    )
    return FrozenAnchorReceipt(
        selection_rule=ANCHOR_SELECTION_RULE,
        candidate_threshold_grid=grid,
        occupancy_config=occupancy_config,
        match_config=match_config,
        selected_metrics=selected_metrics,
        d_v_run_fingerprint=run.run_fingerprint,
        ordered_d_v_sample_ids=run.ordered_d_v_sample_ids,
        base_samples_fingerprint=run.base_samples_fingerprint,
        manifest_fingerprint=bundle.split_manifest_fingerprint,
        manifest_file_sha256=bundle.split_manifest_file_sha256,
        base_index_fingerprint=bundle.base_index_fingerprint,
        base_index_sha256=bundle.base_index_sha256,
        d_v_image_fingerprint=bundle.d_v_image_fingerprint,
        d_v_gt_fingerprint=bundle.d_v_gt_fingerprint,
        preprocessing_fingerprint=bundle.preprocessing_fingerprint,
        base_fingerprint=bundle.base_fingerprint,
        _verification_token=seal,
    )


def select_frozen_anchor(
    run: LoadedDVBaseRun,
    thresholds: Iterable[float],
    match_config: MatchConfig,
) -> FrozenAnchorReceipt:
    """Select ``tau_o`` by global base-only mIoU, tie-breaking upward."""

    if not isinstance(run, LoadedDVBaseRun):
        raise TypeError("run must be a LoadedDVBaseRun")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be a MatchConfig")
    run.verify_unchanged()
    grid = _canonical_threshold_grid(thresholds)
    normalized = tuple(sample.normalized() for sample in run.base_samples)
    threshold = select_anchor_threshold_by_miou(
        tuple(base for base, _, _ in normalized),
        tuple(gt for _, _, gt in normalized),
        grid,
        split_role="D_V",
    )
    occupancy_config = OccupancyConfig(threshold=threshold)
    metrics = evaluate_base_threshold(
        run.base_samples,
        threshold,
        occupancy_config,
        match_config,
    )
    receipt = _anchor_receipt(
        run,
        grid,
        occupancy_config,
        match_config,
        metrics,
    )
    run.verify_unchanged()
    return receipt


def evaluate_frozen_anchor(
    run: LoadedDVBaseRun,
    receipt: FrozenAnchorReceipt,
) -> AggregateEvaluation:
    """Replay the full frozen-grid anchor selection on the bound D_V cache."""

    if not isinstance(run, LoadedDVBaseRun):
        raise TypeError("run must be a LoadedDVBaseRun")
    if not isinstance(receipt, FrozenAnchorReceipt):
        raise TypeError("receipt must be a FrozenAnchorReceipt")
    receipt._verify_source_seal()
    run.verify_unchanged()
    if (
        run.run_fingerprint != receipt.d_v_run_fingerprint
        or run.ordered_d_v_sample_ids != receipt.ordered_d_v_sample_ids
        or run.base_samples_fingerprint != receipt.base_samples_fingerprint
    ):
        raise RuntimeError("D_V base run differs from the frozen anchor receipt")
    normalized = tuple(sample.normalized() for sample in run.base_samples)
    threshold = select_anchor_threshold_by_miou(
        tuple(base for base, _, _ in normalized),
        tuple(gt for _, _, gt in normalized),
        receipt.candidate_threshold_grid,
        split_role="D_V",
    )
    if threshold != receipt.selected_threshold:
        raise RuntimeError("frozen anchor is not the deterministic grid optimum")
    metrics = evaluate_base_threshold(
        run.base_samples,
        threshold,
        receipt.occupancy_config,
        receipt.match_config,
    )
    expected = _anchor_receipt(
        run,
        receipt.candidate_threshold_grid,
        receipt.occupancy_config,
        receipt.match_config,
        metrics,
    )
    if expected != receipt:
        raise RuntimeError("frozen anchor receipt does not reproduce on D_V")
    run.verify_unchanged()
    return metrics


__all__ = [
    "FrozenAnchorReceipt",
    "LoadedDVBaseRun",
    "build_loaded_d_v_base_run",
    "evaluate_frozen_anchor",
    "select_frozen_anchor",
]
