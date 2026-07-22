"""Leakage-resistant Gate-2 development calibration boundaries.

This module deliberately separates three operations:

* Gate-2 development access, which can never return ``D_B`` or ``D_T``;
* deterministic conversion of a frozen base cache plus decoder into a
  :class:`~cure_lite.calibration.CalibrationSample`;
* validation-only threshold selection versus fixed-threshold evaluation.

The current ICLR route explicitly stops before the final-test stage.  This
module exposes no registry, cache, or evaluation path for ``D_T``.  Its fixed
evaluation wrapper accepts only ``D_V`` and has no threshold-grid argument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Iterable, Literal, Sequence

import torch
from torch import Tensor

from ..calibration import (
    CalibrationSample,
    FalseAlarmBudget,
    evaluate_base_threshold as _evaluate_base_threshold,
    evaluate_residual_threshold as _evaluate_residual_threshold,
    select_base_threshold_at_budget as _select_base_threshold_at_budget,
    select_residual_threshold as _select_residual_threshold,
)
from ..cache.schema import stable_fingerprint
from ..config import MatchConfig, OccupancyConfig
from ..decoder import CURELiteDecoder
from ..metrics import AggregateEvaluation
from ..occupancy import build_occupancy_batch
from ..splits import SplitManifest, SplitName, SplitRecord
from ..types import FrozenBaseOutput


GATE_2_SPLITS: frozenset[str] = frozenset({"D_R", "D_V"})


class Gate2SplitAccessError(RuntimeError):
    """Raised when code attempts to cross a sealed Gate-2 split boundary."""


@dataclass(frozen=True)
class DevelopmentSplitAccess:
    """Expose only ``D_R`` and ``D_V`` during the current Gate-2 stage."""

    manifest: SplitManifest

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, SplitManifest):
            raise TypeError("manifest must be a SplitManifest")

    def records_for(self, split: SplitName | str) -> tuple[SplitRecord, ...]:
        if split == "D_T":
            raise Gate2SplitAccessError(
                "D_T is sealed and has no access path in the current Gate-2 stage"
            )
        if split not in GATE_2_SPLITS:
            raise Gate2SplitAccessError(
                "the evaluation pipeline permits only D_R or D_V"
            )
        return self.manifest.records_for(split)  # type: ignore[arg-type]


def development_records(
    manifest: SplitManifest,
    split: SplitName | str,
) -> tuple[SplitRecord, ...]:
    """Functional form of :class:`DevelopmentSplitAccess`."""

    return DevelopmentSplitAccess(manifest).records_for(split)


def _decoder_device_and_dtype(decoder: CURELiteDecoder) -> tuple[torch.device, torch.dtype]:
    parameters = tuple(decoder.parameters())
    if not parameters:
        raise RuntimeError("CURE-Lite decoder unexpectedly has no parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise RuntimeError("decoder parameters must share one device and dtype")
    device = next(iter(devices))
    dtype = next(iter(dtypes))
    if dtype != torch.float32:
        raise TypeError("CURE-Lite evaluation currently requires an FP32 decoder")
    return device, dtype


def calibration_sample_from_cached_base(
    record: SplitRecord,
    base_probability: Tensor,
    feature: Tensor,
    decoder: CURELiteDecoder,
    gt_mask: Tensor,
    occupancy_config: OccupancyConfig,
) -> CalibrationSample:
    """Run only the residual decoder over one detached frozen-base cache.

    The cached feature is never allowed to carry gradients into a base model.
    Decoder train/eval mode is restored before returning, and the resulting
    sample is canonical CPU data suitable for validation or fixed evaluation.
    """

    if not isinstance(record, SplitRecord):
        raise TypeError("record must be a SplitRecord")
    if record.split != "D_V":
        raise Gate2SplitAccessError(
            "calibration samples may be constructed only for D_V"
        )
    if not isinstance(decoder, CURELiteDecoder):
        raise TypeError("decoder must be a CURELiteDecoder")
    if not isinstance(occupancy_config, OccupancyConfig):
        raise TypeError("occupancy_config must be an OccupancyConfig")
    cached = FrozenBaseOutput(base_probability, feature)
    if cached.probability.shape[0] != 1:
        raise ValueError("one CalibrationSample requires a cache batch size of one")
    if cached.probability.device != cached.feature.device:
        raise ValueError("cached probability and feature must share a device")
    if not torch.isfinite(cached.feature).all():
        raise ValueError("cached feature contains non-finite values")
    if cached.feature.shape[1] != decoder.feature_channels:
        raise ValueError("cached feature channels do not match the decoder")
    if cached.feature.shape[-2] < 1 or cached.feature.shape[-1] < 1:
        raise ValueError("cached feature spatial dimensions must be non-empty")
    if cached.probability.shape[-2] < 1 or cached.probability.shape[-1] < 1:
        raise ValueError("cached probability spatial dimensions must be non-empty")

    device, dtype = _decoder_device_and_dtype(decoder)
    feature_for_decoder = cached.feature.detach().to(device=device, dtype=dtype)
    occupancy = build_occupancy_batch(cached.probability, occupancy_config)
    occupancy_for_decoder = occupancy.to(device=device)
    was_training = decoder.training
    try:
        decoder.eval()
        with torch.no_grad():
            logits = decoder(feature_for_decoder, occupancy_for_decoder)
            if logits.shape != occupancy_for_decoder.shape:
                raise ValueError("decoder output and base occupancy grids differ")
            if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                raise ValueError("decoder logits must be finite FP32 values")
            residual_probability = torch.sigmoid(logits).masked_fill(
                occupancy_for_decoder, 0.0
            )
    finally:
        decoder.train(was_training)

    candidate = CalibrationSample(
        sample_id=record.sample_id,
        base_probability=cached.probability.detach().to(device="cpu"),
        residual_probability=residual_probability.detach().to(device="cpu"),
        gt_mask=torch.as_tensor(gt_mask).detach().to(device="cpu"),
    )
    base, residual, gt = candidate.normalized()
    return CalibrationSample(record.sample_id, base, residual, gt)


def _require_exact_sample_membership(
    records: Sequence[SplitRecord],
    samples: Sequence[CalibrationSample],
) -> tuple[CalibrationSample, ...]:
    supplied = tuple(samples)
    if not supplied:
        raise ValueError("at least one evaluation sample is required")
    if any(not isinstance(item, CalibrationSample) for item in supplied):
        raise TypeError("samples must contain only CalibrationSample values")
    sample_ids = [item.sample_id for item in supplied]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("evaluation samples contain duplicate sample IDs")
    record_ids = [record.sample_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("D_V records contain duplicate sample IDs")
    if set(sample_ids) != set(record_ids):
        raise ValueError("evaluation samples do not exactly match the D_V records")
    by_id = {sample.sample_id: sample for sample in supplied}
    canonical: list[CalibrationSample] = []
    for sample_id in record_ids:
        base, residual, gt = by_id[sample_id].normalized()
        canonical.append(CalibrationSample(sample_id, base, residual, gt))
    return tuple(canonical)


def _canonical_threshold_grid(
    values: Iterable[float], *, allow_empty: bool
) -> tuple[float, ...]:
    resolved: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("threshold candidates must be real numbers, not bool")
        try:
            threshold = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("threshold candidates must be real numbers") from error
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold candidates must be finite and in [0,1]")
        resolved.append(threshold)
    result = tuple(sorted(set(resolved)))
    if not result and not allow_empty:
        raise ValueError("threshold candidates are empty")
    return result


def _normalize_budget(value: object) -> FalseAlarmBudget:
    try:
        raw_values = (
            getattr(value, "pixel_fa_budget"),
            getattr(value, "component_fa_per_mp_budget"),
            getattr(value, "raw_background_fa_budget", float("inf")),
            getattr(value, "minimum_retention", 0.0),
        )
        if any(isinstance(item, bool) for item in raw_values):
            raise TypeError("budget fields may not be bool")
        return FalseAlarmBudget(
            pixel_fa_budget=float(raw_values[0]),
            component_fa_per_mp_budget=float(raw_values[1]),
            raw_background_fa_budget=float(raw_values[2]),
            minimum_retention=float(raw_values[3]),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "budget must expose valid false-alarm and retention limits"
        ) from error


def _tensor_content_fingerprint(samples: Sequence[CalibrationSample]) -> str:
    digest = hashlib.sha256(b"cure-lite-d-v-calibration-samples-v1")
    for sample in samples:
        encoded_id = sample.sample_id.encode("utf-8")
        digest.update(len(encoded_id).to_bytes(8, "big"))
        digest.update(encoded_id)
        base, residual, gt = sample.normalized()
        for name, tensor in (
            ("base_probability", base),
            ("residual_probability", residual),
            ("gt_mask", gt),
        ):
            canonical = tensor.detach().to(device="cpu").contiguous()
            encoded_name = name.encode("ascii")
            encoded_dtype = str(canonical.dtype).encode("ascii")
            encoded_shape = json.dumps(
                list(canonical.shape), separators=(",", ":")
            ).encode("ascii")
            raw = canonical.reshape(-1).view(torch.uint8).numpy().tobytes()
            for block in (encoded_name, encoded_dtype, encoded_shape, raw):
                digest.update(len(block).to_bytes(8, "big"))
                digest.update(block)
    return digest.hexdigest()


def calibration_samples_fingerprint(
    samples: Sequence[CalibrationSample],
) -> str:
    """Return the canonical tensor-content identity used by D_V receipts."""

    return _tensor_content_fingerprint(tuple(samples))


def _digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _budget_payload(budget: FalseAlarmBudget) -> dict[str, float | None]:
    def finite_or_none(value: float) -> float | None:
        return None if math.isinf(value) else value

    return {
        "pixel_fa_budget": budget.pixel_fa_budget,
        "component_fa_per_mp_budget": finite_or_none(
            budget.component_fa_per_mp_budget
        ),
        "raw_background_fa_budget": finite_or_none(
            budget.raw_background_fa_budget
        ),
        "minimum_retention": budget.minimum_retention,
    }


@dataclass(frozen=True)
class BoundDVThresholdProtocol:
    """Immutable selection receipt and fixed D_V evaluation protocol."""

    variant: Literal["residual", "base_at_budget"]
    manifest_fingerprint: str
    ordered_d_v_sample_ids: tuple[str, ...]
    sample_tensor_fingerprint: str
    candidate_threshold_grid: tuple[float, ...]
    occupancy_config: OccupancyConfig
    match_config: MatchConfig
    budget: FalseAlarmBudget
    selected_threshold: float | None
    selected_metrics: AggregateEvaluation

    def __post_init__(self) -> None:
        if self.variant not in {"residual", "base_at_budget"}:
            raise ValueError("unsupported D_V threshold protocol variant")
        _digest(self.manifest_fingerprint, name="manifest_fingerprint")
        _digest(self.sample_tensor_fingerprint, name="sample_tensor_fingerprint")
        if (
            not isinstance(self.ordered_d_v_sample_ids, tuple)
            or not self.ordered_d_v_sample_ids
            or any(
                not isinstance(sample_id, str) or not sample_id
                for sample_id in self.ordered_d_v_sample_ids
            )
            or len(set(self.ordered_d_v_sample_ids))
            != len(self.ordered_d_v_sample_ids)
        ):
            raise ValueError("ordered D_V sample IDs must be a non-empty unique tuple")
        if not isinstance(self.candidate_threshold_grid, tuple):
            raise TypeError("candidate threshold grid must be a tuple")
        expected_grid = _canonical_threshold_grid(
            self.candidate_threshold_grid,
            allow_empty=self.variant == "residual",
        )
        if expected_grid != self.candidate_threshold_grid:
            raise ValueError("candidate threshold grid must be sorted and unique")
        if not isinstance(self.occupancy_config, OccupancyConfig):
            raise TypeError("occupancy_config must be an OccupancyConfig")
        if not isinstance(self.match_config, MatchConfig):
            raise TypeError("match_config must be a MatchConfig")
        if not isinstance(self.budget, FalseAlarmBudget):
            raise TypeError("budget must be a normalized FalseAlarmBudget")
        if not isinstance(self.selected_metrics, AggregateEvaluation):
            raise TypeError("selected_metrics must be an AggregateEvaluation")
        if self.selected_metrics.budget_violation or not self.budget.accepts(
            self.selected_metrics
        ):
            raise ValueError("selected D_V metrics violate the frozen budget")
        if self.selected_threshold is None:
            if self.variant != "residual":
                raise ValueError("Base@B must select a numeric threshold")
        else:
            if isinstance(self.selected_threshold, bool):
                raise TypeError("selected threshold may not be bool")
            threshold = float(self.selected_threshold)
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("selected threshold must be finite and in [0,1]")
            object.__setattr__(self, "selected_threshold", threshold)
            permitted = set(self.candidate_threshold_grid)
            if self.variant == "base_at_budget":
                permitted.add(self.occupancy_config.threshold)
                if threshold > self.occupancy_config.threshold:
                    raise ValueError("Base@B threshold may not exceed tau_o")
            if threshold not in permitted:
                raise ValueError("selected threshold is absent from its frozen grid")

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": "cure-lite-bound-d-v-threshold-protocol-v1",
                "variant": self.variant,
                "manifest_fingerprint": self.manifest_fingerprint,
                "ordered_d_v_sample_ids": self.ordered_d_v_sample_ids,
                "sample_tensor_fingerprint": self.sample_tensor_fingerprint,
                "candidate_threshold_grid": self.candidate_threshold_grid,
                "occupancy_config": self.occupancy_config,
                "match_config": self.match_config,
                "budget": _budget_payload(self.budget),
                "selected_threshold": self.selected_threshold,
                "selected_metrics": asdict(self.selected_metrics),
            }
        )


def _bound_d_v_inputs(
    access: DevelopmentSplitAccess,
    samples: Sequence[CalibrationSample],
) -> tuple[tuple[str, ...], tuple[CalibrationSample, ...], str]:
    if not isinstance(access, DevelopmentSplitAccess):
        raise TypeError("access must be DevelopmentSplitAccess")
    records = access.records_for("D_V")
    canonical_samples = _require_exact_sample_membership(records, samples)
    ordered_ids = tuple(record.sample_id for record in records)
    return (
        ordered_ids,
        canonical_samples,
        _tensor_content_fingerprint(canonical_samples),
    )


def _selection_receipt(
    *,
    variant: Literal["residual", "base_at_budget"],
    access: DevelopmentSplitAccess,
    ordered_ids: tuple[str, ...],
    sample_fingerprint: str,
    threshold_grid: tuple[float, ...],
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    budget: FalseAlarmBudget,
    selected_threshold: float | None,
    selected_metrics: AggregateEvaluation | None,
) -> BoundDVThresholdProtocol:
    if selected_metrics is None:
        raise RuntimeError("D_V threshold selection did not produce metrics")
    return BoundDVThresholdProtocol(
        variant=variant,
        manifest_fingerprint=access.manifest.fingerprint,
        ordered_d_v_sample_ids=ordered_ids,
        sample_tensor_fingerprint=sample_fingerprint,
        candidate_threshold_grid=threshold_grid,
        occupancy_config=occupancy_config,
        match_config=match_config,
        budget=budget,
        selected_threshold=selected_threshold,
        selected_metrics=selected_metrics,
    )


def select_residual_threshold_on_d_v(
    access: DevelopmentSplitAccess,
    samples: Sequence[CalibrationSample],
    thresholds: Iterable[float],
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    budget: object,
) -> BoundDVThresholdProtocol:
    """Select and freeze a provenance-bound residual protocol on exact D_V."""

    grid = _canonical_threshold_grid(thresholds, allow_empty=True)
    resolved_budget = _normalize_budget(budget)
    ordered_ids, canonical_samples, sample_fingerprint = _bound_d_v_inputs(
        access, samples
    )
    selection = _select_residual_threshold(
        canonical_samples,
        grid,
        occupancy_config,
        match_config,
        resolved_budget,
        split_role="D_V",
    )
    if not selection.feasible:
        raise RuntimeError(selection.reason or "D_V residual selection is infeasible")
    return _selection_receipt(
        variant="residual",
        access=access,
        ordered_ids=ordered_ids,
        sample_fingerprint=sample_fingerprint,
        threshold_grid=grid,
        occupancy_config=occupancy_config,
        match_config=match_config,
        budget=resolved_budget,
        selected_threshold=selection.threshold,
        selected_metrics=selection.metrics,
    )


def select_base_threshold_on_d_v(
    access: DevelopmentSplitAccess,
    samples: Sequence[CalibrationSample],
    thresholds: Iterable[float],
    anchor_occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    budget: object,
) -> BoundDVThresholdProtocol:
    """Select and freeze a provenance-bound Base@B protocol on exact D_V."""

    grid = _canonical_threshold_grid(thresholds, allow_empty=False)
    resolved_budget = _normalize_budget(budget)
    ordered_ids, canonical_samples, sample_fingerprint = _bound_d_v_inputs(
        access, samples
    )
    selection = _select_base_threshold_at_budget(
        canonical_samples,
        grid,
        anchor_occupancy_config,
        match_config,
        resolved_budget,
        split_role="D_V",
    )
    if not selection.feasible or selection.threshold is None:
        raise RuntimeError(selection.reason or "D_V Base@B selection is infeasible")
    return _selection_receipt(
        variant="base_at_budget",
        access=access,
        ordered_ids=ordered_ids,
        sample_fingerprint=sample_fingerprint,
        threshold_grid=grid,
        occupancy_config=anchor_occupancy_config,
        match_config=match_config,
        budget=resolved_budget,
        selected_threshold=selection.threshold,
        selected_metrics=selection.metrics,
    )


def _revalidate_bound_inputs(
    access: DevelopmentSplitAccess,
    samples: Sequence[CalibrationSample],
    protocol: BoundDVThresholdProtocol,
    *,
    expected_variant: Literal["residual", "base_at_budget"],
) -> tuple[CalibrationSample, ...]:
    if not isinstance(protocol, BoundDVThresholdProtocol):
        raise TypeError("protocol must be a BoundDVThresholdProtocol")
    if protocol.variant != expected_variant:
        raise TypeError(f"expected a {expected_variant} D_V protocol")
    ordered_ids, canonical_samples, sample_fingerprint = _bound_d_v_inputs(
        access, samples
    )
    if access.manifest.fingerprint != protocol.manifest_fingerprint:
        raise RuntimeError("D_V manifest fingerprint differs from the selection receipt")
    if ordered_ids != protocol.ordered_d_v_sample_ids:
        raise RuntimeError("ordered D_V sample IDs differ from the selection receipt")
    if sample_fingerprint != protocol.sample_tensor_fingerprint:
        raise RuntimeError("D_V sample tensor content differs from the selection receipt")
    # Re-run the complete frozen-grid selector.  Reproducing metrics at one
    # caller-supplied threshold is insufficient: this check proves the stored
    # threshold is exactly the deterministic optimum selected by the protocol.
    if expected_variant == "residual":
        selection = _select_residual_threshold(
            canonical_samples,
            protocol.candidate_threshold_grid,
            protocol.occupancy_config,
            protocol.match_config,
            protocol.budget,
            split_role="D_V",
        )
    else:
        selection = _select_base_threshold_at_budget(
            canonical_samples,
            protocol.candidate_threshold_grid,
            protocol.occupancy_config,
            protocol.match_config,
            protocol.budget,
            split_role="D_V",
        )
    if not selection.feasible or selection.metrics is None:
        raise RuntimeError("frozen D_V threshold grid is no longer feasible")
    expected = _selection_receipt(
        variant=expected_variant,
        access=access,
        ordered_ids=ordered_ids,
        sample_fingerprint=sample_fingerprint,
        threshold_grid=protocol.candidate_threshold_grid,
        occupancy_config=protocol.occupancy_config,
        match_config=protocol.match_config,
        budget=protocol.budget,
        selected_threshold=selection.threshold,
        selected_metrics=selection.metrics,
    )
    if expected != protocol:
        raise RuntimeError("D_V receipt is not the deterministic frozen-grid selection")
    return canonical_samples


def _checked_metrics(
    metrics: AggregateEvaluation,
    protocol: BoundDVThresholdProtocol,
) -> AggregateEvaluation:
    result = replace(
        metrics,
        budget_violation=not protocol.budget.accepts(metrics),
    )
    if result != protocol.selected_metrics:
        raise RuntimeError("fixed D_V metrics do not reproduce the selection receipt")
    return result


def evaluate_frozen_residual_threshold(
    access: DevelopmentSplitAccess,
    samples: Sequence[CalibrationSample],
    protocol: BoundDVThresholdProtocol,
) -> AggregateEvaluation:
    """Re-evaluate one provenance-bound residual selection on the same D_V."""

    canonical_samples = _revalidate_bound_inputs(
        access, samples, protocol, expected_variant="residual"
    )
    metrics = _evaluate_residual_threshold(
        canonical_samples,
        protocol.selected_threshold,
        protocol.occupancy_config,
        protocol.match_config,
    )
    return _checked_metrics(metrics, protocol)


def evaluate_frozen_base_threshold(
    access: DevelopmentSplitAccess,
    samples: Sequence[CalibrationSample],
    protocol: BoundDVThresholdProtocol,
) -> AggregateEvaluation:
    """Re-evaluate one provenance-bound anchor/Base@B selection on its D_V."""

    canonical_samples = _revalidate_bound_inputs(
        access, samples, protocol, expected_variant="base_at_budget"
    )
    if protocol.selected_threshold is None:  # defended by protocol validation
        raise RuntimeError("Base@B receipt unexpectedly contains a null threshold")
    metrics = _evaluate_base_threshold(
        canonical_samples,
        protocol.selected_threshold,
        protocol.occupancy_config,
        protocol.match_config,
    )
    return _checked_metrics(metrics, protocol)


__all__ = [
    "GATE_2_SPLITS",
    "BoundDVThresholdProtocol",
    "DevelopmentSplitAccess",
    "Gate2SplitAccessError",
    "calibration_sample_from_cached_base",
    "calibration_samples_fingerprint",
    "development_records",
    "evaluate_frozen_base_threshold",
    "evaluate_frozen_residual_threshold",
    "select_base_threshold_on_d_v",
    "select_residual_threshold_on_d_v",
]
