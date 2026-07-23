"""Strict D_V calibration over verified caches and immutable decoder artifacts.

This is the formal Gate-2 layer above the tensor-level calibration helpers.
Residual-method results are bound to their trained decoder artifacts.  The
``Base@B`` control instead has a distinct decoder-free receipt bound directly
to the verified base-only D_V run, cache provenance, threshold grid, and
false-alarm budget.  There is deliberately no ``D_T`` entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from multiprocessing.context import BaseContext
from typing import Iterable, Literal

import torch

from ..cache.schema import stable_fingerprint
from ..calibration import (
    CalibrationSample,
    FalseAlarmBudget,
    ThresholdSelection,
)
from ..calibration_ledger import (
    CalibrationCandidateLedger,
    ProgressCallback,
    evaluate_candidate_ledger,
    prepare_calibration_context,
)
from ..config import MatchConfig, OccupancyConfig
from ..metrics import AggregateEvaluation
from ..splits import SplitManifest
from .artifacts import DECODER_ARTIFACT_SCHEMA_V2, LoadedDecoderArtifact
from .cache_pipeline import LoadedDVCacheBundle
from .formal_anchor import (
    FrozenAnchorReceipt,
    LoadedDVBaseRun,
    evaluate_frozen_anchor,
)
from .evaluation_pipeline import (
    BoundDVThresholdProtocol,
    DevelopmentSplitAccess,
    calibration_sample_from_cached_base,
    calibration_samples_fingerprint,
    evaluate_frozen_base_threshold,
    evaluate_frozen_residual_threshold,
    select_base_threshold_on_d_v,
    select_residual_threshold_on_d_v,
)


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True, slots=True)
class _LoadedDVRunSeal:
    bundle: LoadedDVCacheBundle
    artifact: LoadedDecoderArtifact
    access: DevelopmentSplitAccess
    residual_samples: tuple[CalibrationSample, ...]
    base_samples: tuple[CalibrationSample, ...]
    residual_samples_fingerprint: str
    base_samples_fingerprint: str


@dataclass(frozen=True)
class LoadedDVMethodRun:
    """One decoder evaluated over one exact, verified D_V cache bundle."""

    bundle: LoadedDVCacheBundle
    artifact: LoadedDecoderArtifact
    access: DevelopmentSplitAccess
    residual_samples: tuple[CalibrationSample, ...]
    base_samples: tuple[CalibrationSample, ...]
    residual_samples_fingerprint: str
    base_samples_fingerprint: str
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _LoadedDVRunSeal:
            raise TypeError(
                "LoadedDVMethodRun must be created by the strict D_V builder"
            )
        if (
            seal.bundle is not self.bundle
            or seal.artifact is not self.artifact
            or seal.access is not self.access
            or seal.residual_samples is not self.residual_samples
            or seal.base_samples is not self.base_samples
        ):
            raise TypeError("loaded D_V method-run source objects were replaced")
        if (
            seal.residual_samples_fingerprint
            != self.residual_samples_fingerprint
            or seal.base_samples_fingerprint != self.base_samples_fingerprint
        ):
            raise TypeError("loaded D_V method-run fingerprints were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        _digest(
            self.residual_samples_fingerprint,
            name="residual_samples_fingerprint",
        )
        _digest(self.base_samples_fingerprint, name="base_samples_fingerprint")
        self.verify_unchanged()

    @property
    def run_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": "cure-lite-loaded-d-v-method-run-v1",
                "manifest_fingerprint": self.bundle.split_manifest_fingerprint,
                "manifest_file_sha256": self.bundle.split_manifest_file_sha256,
                "base_index_fingerprint": self.bundle.base_index_fingerprint,
                "base_index_sha256": self.bundle.base_index_sha256,
                "d_v_image_fingerprint": self.bundle.d_v_image_fingerprint,
                "d_v_gt_fingerprint": self.bundle.d_v_gt_fingerprint,
                "preprocessing_fingerprint": self.bundle.preprocessing_fingerprint,
                "base_fingerprint": self.bundle.base_fingerprint,
                "decoder_artifact_fingerprint": self.artifact.artifact_fingerprint,
                "decoder_receipt_sha256": self.artifact.receipt_sha256,
                "decoder_state_fingerprint": self.artifact.decoder_state_fingerprint,
                "decoder_variant": self.artifact.config.variant,
                "global_seed": self.artifact.config.global_seed,
                "residual_samples_fingerprint": self.residual_samples_fingerprint,
                "base_samples_fingerprint": self.base_samples_fingerprint,
            }
        )

    def verify_unchanged(self) -> None:
        self._verify_source_seal()
        if not isinstance(self.bundle, LoadedDVCacheBundle):
            raise TypeError("bundle must be a LoadedDVCacheBundle")
        if not isinstance(self.artifact, LoadedDecoderArtifact):
            raise TypeError("artifact must be a LoadedDecoderArtifact")
        if not isinstance(self.access, DevelopmentSplitAccess):
            raise TypeError("access must be DevelopmentSplitAccess")
        self.bundle.verify_unchanged()
        self.artifact.verify_unchanged()
        config = self.artifact.config
        if config.manifest_fingerprint != self.bundle.split_manifest_fingerprint:
            raise RuntimeError("decoder and D_V bundle use different manifests")
        if config.manifest_file_sha256 != self.bundle.split_manifest_file_sha256:
            raise RuntimeError("decoder and D_V bundle bind different manifest bytes")
        if config.base_fingerprint != self.bundle.base_fingerprint:
            raise RuntimeError("decoder and D_V bundle use different frozen bases")
        if config.preprocessing_fingerprint != self.bundle.preprocessing_fingerprint:
            raise RuntimeError("decoder and D_V bundle use different preprocessing")
        if self.access.manifest.fingerprint != self.bundle.split_manifest_fingerprint:
            raise RuntimeError("D_V access manifest differs from the cache bundle")
        if calibration_samples_fingerprint(
            self.residual_samples
        ) != self.residual_samples_fingerprint:
            raise RuntimeError("residual D_V samples changed in memory")
        if calibration_samples_fingerprint(
            self.base_samples
        ) != self.base_samples_fingerprint:
            raise RuntimeError("base-only D_V samples changed in memory")
        expected_ids = tuple(
            record.sample_id for record in self.access.records_for("D_V")
        )
        for name, samples in (
            ("residual", self.residual_samples),
            ("base-only", self.base_samples),
        ):
            sample_ids = tuple(sample.sample_id for sample in samples)
            if set(sample_ids) != set(expected_ids) or len(sample_ids) != len(
                expected_ids
            ):
                raise RuntimeError(f"{name} D_V sample membership changed")


def build_loaded_d_v_method_run(
    bundle: LoadedDVCacheBundle,
    artifact: LoadedDecoderArtifact,
) -> LoadedDVMethodRun:
    """Run one verified decoder over every row of one exact D_V bundle."""

    if not isinstance(bundle, LoadedDVCacheBundle):
        raise TypeError("bundle must be a LoadedDVCacheBundle")
    if not isinstance(artifact, LoadedDecoderArtifact):
        raise TypeError("artifact must be a LoadedDecoderArtifact")
    bundle.verify_unchanged()
    artifact.verify_unchanged()
    config = artifact.config
    if config.manifest_fingerprint != bundle.split_manifest_fingerprint:
        raise RuntimeError("decoder artifact belongs to another manifest")
    if config.manifest_file_sha256 != bundle.split_manifest_file_sha256:
        raise RuntimeError("decoder artifact binds different manifest bytes")
    if config.base_fingerprint != bundle.base_fingerprint:
        raise RuntimeError("decoder artifact belongs to another frozen base")
    if config.preprocessing_fingerprint != bundle.preprocessing_fingerprint:
        raise RuntimeError("decoder artifact uses different preprocessing")
    if config.decoder_config.feature_channels != artifact.decoder.feature_channels:
        raise RuntimeError("decoder artifact channel contract is inconsistent")
    feature_channels = {int(row.base_output.feature.shape[1]) for row in bundle.rows}
    if feature_channels != {config.decoder_config.feature_channels}:
        raise RuntimeError("D_V feature channels differ from the decoder artifact")

    manifest = SplitManifest.load(bundle.manifest_path)
    access = DevelopmentSplitAccess(manifest)
    records = {record.sample_id: record for record in access.records_for("D_V")}
    if set(records) != {row.sample_id for row in bundle.rows}:
        raise RuntimeError("D_V bundle membership differs from its manifest")
    residual_samples: list[CalibrationSample] = []
    base_samples: list[CalibrationSample] = []
    for row in bundle.rows:
        residual = calibration_sample_from_cached_base(
            records[row.sample_id],
            row.base_output.probability,
            row.base_output.feature,
            artifact.decoder,
            row.gt_mask,
            config.occupancy_config,
        )
        residual_samples.append(residual)
        base, _, gt = residual.normalized()
        base_samples.append(
            CalibrationSample(
                row.sample_id,
                base,
                torch.zeros_like(base),
                gt,
            )
        )
    residual_tuple = tuple(residual_samples)
    base_tuple = tuple(base_samples)
    residual_fingerprint = calibration_samples_fingerprint(residual_tuple)
    base_fingerprint = calibration_samples_fingerprint(base_tuple)
    seal = _LoadedDVRunSeal(
        bundle=bundle,
        artifact=artifact,
        access=access,
        residual_samples=residual_tuple,
        base_samples=base_tuple,
        residual_samples_fingerprint=residual_fingerprint,
        base_samples_fingerprint=base_fingerprint,
    )
    result = LoadedDVMethodRun(
        bundle=bundle,
        artifact=artifact,
        access=access,
        residual_samples=residual_tuple,
        base_samples=base_tuple,
        residual_samples_fingerprint=residual_fingerprint,
        base_samples_fingerprint=base_fingerprint,
        _verification_token=seal,
    )
    bundle.verify_unchanged()
    artifact.verify_unchanged()
    return result


@dataclass(frozen=True)
class FormalDVThresholdReceipt:
    """A residual threshold protocol bound to its cache and decoder source."""

    mode: Literal["residual"]
    protocol: BoundDVThresholdProtocol
    d_v_run_fingerprint: str
    manifest_file_sha256: str
    base_index_fingerprint: str
    base_index_sha256: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    decoder_artifact_fingerprint: str
    decoder_receipt_sha256: str
    decoder_state_fingerprint: str
    decoder_variant: str
    global_seed: int
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _FormalDVReceiptSeal:
            raise TypeError(
                "FormalDVThresholdReceipt must come from formal D_V selection"
            )
        if seal.protocol is not self.protocol:
            raise TypeError("formal D_V threshold protocol object was replaced")
        if seal.bound_values != (
            self.mode,
            self.d_v_run_fingerprint,
            self.manifest_file_sha256,
            self.base_index_fingerprint,
            self.base_index_sha256,
            self.d_v_image_fingerprint,
            self.d_v_gt_fingerprint,
            self.preprocessing_fingerprint,
            self.base_fingerprint,
            self.decoder_artifact_fingerprint,
            self.decoder_receipt_sha256,
            self.decoder_state_fingerprint,
            self.decoder_variant,
            self.global_seed,
        ):
            raise TypeError("formal D_V threshold receipt fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        if self.mode != "residual":
            raise ValueError("formal decoder receipt mode must be residual")
        if not isinstance(self.protocol, BoundDVThresholdProtocol):
            raise TypeError("protocol must be BoundDVThresholdProtocol")
        if self.protocol.variant != self.mode:
            raise ValueError("formal receipt mode and threshold protocol differ")
        for name in (
            "d_v_run_fingerprint",
            "manifest_file_sha256",
            "base_index_fingerprint",
            "base_index_sha256",
            "d_v_image_fingerprint",
            "d_v_gt_fingerprint",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "decoder_artifact_fingerprint",
            "decoder_receipt_sha256",
            "decoder_state_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        if not isinstance(self.decoder_variant, str) or not self.decoder_variant:
            raise ValueError("decoder_variant must be non-empty")
        if isinstance(self.global_seed, bool) or not isinstance(self.global_seed, int):
            raise TypeError("global_seed must be an integer")

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": "cure-lite-formal-d-v-threshold-receipt-v1",
                "mode": self.mode,
                "threshold_protocol_fingerprint": self.protocol.receipt_fingerprint,
                "d_v_run_fingerprint": self.d_v_run_fingerprint,
                "manifest_file_sha256": self.manifest_file_sha256,
                "base_index_fingerprint": self.base_index_fingerprint,
                "base_index_sha256": self.base_index_sha256,
                "d_v_image_fingerprint": self.d_v_image_fingerprint,
                "d_v_gt_fingerprint": self.d_v_gt_fingerprint,
                "preprocessing_fingerprint": self.preprocessing_fingerprint,
                "base_fingerprint": self.base_fingerprint,
                "decoder_artifact_fingerprint": self.decoder_artifact_fingerprint,
                "decoder_receipt_sha256": self.decoder_receipt_sha256,
                "decoder_state_fingerprint": self.decoder_state_fingerprint,
                "decoder_variant": self.decoder_variant,
                "global_seed": self.global_seed,
            }
        )


@dataclass(frozen=True, slots=True)
class _FormalDVReceiptSeal:
    protocol: BoundDVThresholdProtocol
    bound_values: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _FormalDVBaseReceiptSeal:
    protocol: BoundDVThresholdProtocol
    bound_values: tuple[object, ...]


@dataclass(frozen=True)
class FormalDVBaseThresholdReceipt:
    """Decoder-free Base@B selection bound to one verified D_V base run.

    This receipt intentionally has no decoder artifact, decoder state,
    decoder variant, or training seed.  Base@B is selected from cached base
    probabilities and GT only; introducing any decoder identity here would
    make the control depend on an unrelated trained method.
    """

    protocol: BoundDVThresholdProtocol
    d_v_base_run_fingerprint: str
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
        if type(seal) is not _FormalDVBaseReceiptSeal:
            raise TypeError(
                "FormalDVBaseThresholdReceipt must come from formal "
                "decoder-free D_V selection"
            )
        if seal.protocol is not self.protocol:
            raise TypeError("formal Base@B threshold protocol object was replaced")
        if seal.bound_values != (
            self.d_v_base_run_fingerprint,
            self.manifest_file_sha256,
            self.base_index_fingerprint,
            self.base_index_sha256,
            self.d_v_image_fingerprint,
            self.d_v_gt_fingerprint,
            self.preprocessing_fingerprint,
            self.base_fingerprint,
        ):
            raise TypeError("formal Base@B threshold receipt fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        if not isinstance(self.protocol, BoundDVThresholdProtocol):
            raise TypeError("protocol must be BoundDVThresholdProtocol")
        if self.protocol.variant != "base_at_budget":
            raise ValueError("Base@B receipt requires a base_at_budget protocol")
        if self.protocol.selected_threshold is None:
            raise ValueError("Base@B receipt requires a numeric threshold")
        for name in (
            "d_v_base_run_fingerprint",
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
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": (
                    "cure-lite-formal-d-v-base-threshold-receipt-v1"
                ),
                "method": "Base@B",
                "threshold_protocol_fingerprint": (
                    self.protocol.receipt_fingerprint
                ),
                "d_v_base_run_fingerprint": self.d_v_base_run_fingerprint,
                "manifest_file_sha256": self.manifest_file_sha256,
                "base_index_fingerprint": self.base_index_fingerprint,
                "base_index_sha256": self.base_index_sha256,
                "d_v_image_fingerprint": self.d_v_image_fingerprint,
                "d_v_gt_fingerprint": self.d_v_gt_fingerprint,
                "preprocessing_fingerprint": self.preprocessing_fingerprint,
                "base_fingerprint": self.base_fingerprint,
            }
        )


def _formal_receipt(
    run: LoadedDVMethodRun,
    protocol: BoundDVThresholdProtocol,
) -> FormalDVThresholdReceipt:
    if protocol.variant != "residual":
        raise ValueError("decoder-bound formal receipt requires residual protocol")
    bundle = run.bundle
    artifact = run.artifact
    bound_values = (
        protocol.variant,
        run.run_fingerprint,
        bundle.split_manifest_file_sha256,
        bundle.base_index_fingerprint,
        bundle.base_index_sha256,
        bundle.d_v_image_fingerprint,
        bundle.d_v_gt_fingerprint,
        bundle.preprocessing_fingerprint,
        bundle.base_fingerprint,
        artifact.artifact_fingerprint,
        artifact.receipt_sha256,
        artifact.decoder_state_fingerprint,
        artifact.config.variant,
        artifact.config.global_seed,
    )
    seal = _FormalDVReceiptSeal(protocol=protocol, bound_values=bound_values)
    return FormalDVThresholdReceipt(
        mode=bound_values[0],
        protocol=protocol,
        d_v_run_fingerprint=bound_values[1],
        manifest_file_sha256=bound_values[2],
        base_index_fingerprint=bound_values[3],
        base_index_sha256=bound_values[4],
        d_v_image_fingerprint=bound_values[5],
        d_v_gt_fingerprint=bound_values[6],
        preprocessing_fingerprint=bound_values[7],
        base_fingerprint=bound_values[8],
        decoder_artifact_fingerprint=bound_values[9],
        decoder_receipt_sha256=bound_values[10],
        decoder_state_fingerprint=bound_values[11],
        decoder_variant=bound_values[12],
        global_seed=bound_values[13],
        _verification_token=seal,
    )


def _formal_base_receipt(
    run: LoadedDVBaseRun,
    protocol: BoundDVThresholdProtocol,
) -> FormalDVBaseThresholdReceipt:
    """Bind Base@B to base-only cache provenance, never to a decoder."""

    if not isinstance(run, LoadedDVBaseRun):
        raise TypeError("run must be a LoadedDVBaseRun")
    if protocol.variant != "base_at_budget":
        raise ValueError("decoder-free Base@B receipt requires base_at_budget")
    bundle = run.bundle
    bound_values = (
        run.run_fingerprint,
        bundle.split_manifest_file_sha256,
        bundle.base_index_fingerprint,
        bundle.base_index_sha256,
        bundle.d_v_image_fingerprint,
        bundle.d_v_gt_fingerprint,
        bundle.preprocessing_fingerprint,
        bundle.base_fingerprint,
    )
    seal = _FormalDVBaseReceiptSeal(
        protocol=protocol,
        bound_values=bound_values,
    )
    return FormalDVBaseThresholdReceipt(
        protocol=protocol,
        d_v_base_run_fingerprint=bound_values[0],
        manifest_file_sha256=bound_values[1],
        base_index_fingerprint=bound_values[2],
        base_index_sha256=bound_values[3],
        d_v_image_fingerprint=bound_values[4],
        d_v_gt_fingerprint=bound_values[5],
        preprocessing_fingerprint=bound_values[6],
        base_fingerprint=bound_values[7],
        _verification_token=seal,
    )


def _canonical_candidate_grid(
    values: Iterable[float],
    *,
    allow_empty: bool,
) -> tuple[float, ...]:
    """Match the strict D_V protocol's threshold-grid normalization."""

    resolved: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("threshold candidates must be real numbers, not bool")
        try:
            threshold = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("threshold candidates must be real numbers") from error
        if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold candidates must be finite and in [0,1]")
        resolved.append(threshold)
    grid = tuple(sorted(set(resolved)))
    if not grid and not allow_empty:
        raise ValueError("threshold candidates are empty")
    return grid


def _protocol_from_ledger_selection(
    run: LoadedDVMethodRun,
    thresholds: Iterable[float],
    budget: FalseAlarmBudget,
    selection: ThresholdSelection,
    *,
    variant: Literal["residual"] = "residual",
) -> BoundDVThresholdProtocol:
    """Bind one exact residual ledger selection to its decoder run."""

    if not isinstance(selection, ThresholdSelection):
        raise TypeError("selection must be a ThresholdSelection")
    if not selection.feasible or selection.metrics is None:
        raise RuntimeError(
            selection.reason or f"D_V {variant} selection is infeasible"
        )
    samples = run.residual_samples
    sample_fingerprint = run.residual_samples_fingerprint
    ordered_ids = tuple(
        record.sample_id for record in run.access.records_for("D_V")
    )
    if tuple(sample.sample_id for sample in samples) != ordered_ids:
        raise RuntimeError("formal D_V samples are not in manifest order")
    return BoundDVThresholdProtocol(
        variant=variant,
        manifest_fingerprint=run.access.manifest.fingerprint,
        ordered_d_v_sample_ids=ordered_ids,
        sample_tensor_fingerprint=sample_fingerprint,
        candidate_threshold_grid=_canonical_candidate_grid(
            thresholds,
            allow_empty=True,
        ),
        occupancy_config=run.artifact.config.occupancy_config,
        match_config=run.artifact.config.match_config,
        budget=budget,
        selected_threshold=selection.threshold,
        selected_metrics=selection.metrics,
    )


def _base_protocol_from_ledger_selection(
    run: LoadedDVBaseRun,
    thresholds: Iterable[float],
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    budget: FalseAlarmBudget,
    selection: ThresholdSelection,
) -> BoundDVThresholdProtocol:
    """Bind one exact Base@B ledger selection to a decoder-free base run."""

    if not isinstance(selection, ThresholdSelection):
        raise TypeError("selection must be a ThresholdSelection")
    if not selection.feasible or selection.metrics is None:
        raise RuntimeError(selection.reason or "D_V Base@B selection is infeasible")
    if selection.threshold is None:
        raise RuntimeError("D_V Base@B selection must be numeric")
    if not isinstance(occupancy_config, OccupancyConfig):
        raise TypeError("occupancy_config must be an OccupancyConfig")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be a MatchConfig")
    if tuple(sample.sample_id for sample in run.base_samples) != (
        run.ordered_d_v_sample_ids
    ):
        raise RuntimeError("formal Base@B samples are not in manifest order")
    return BoundDVThresholdProtocol(
        variant="base_at_budget",
        manifest_fingerprint=run.access.manifest.fingerprint,
        ordered_d_v_sample_ids=run.ordered_d_v_sample_ids,
        sample_tensor_fingerprint=run.base_samples_fingerprint,
        candidate_threshold_grid=_canonical_candidate_grid(
            thresholds,
            allow_empty=False,
        ),
        occupancy_config=occupancy_config,
        match_config=match_config,
        budget=budget,
        selected_threshold=selection.threshold,
        selected_metrics=selection.metrics,
    )


def select_formal_residual_threshold(
    run: LoadedDVMethodRun,
    thresholds: Iterable[float],
    budget: object,
) -> FormalDVThresholdReceipt:
    run.verify_unchanged()
    protocol = select_residual_threshold_on_d_v(
        run.access,
        run.residual_samples,
        thresholds,
        run.artifact.config.occupancy_config,
        run.artifact.config.match_config,
        budget,
    )
    receipt = _formal_receipt(run, protocol)
    run.verify_unchanged()
    return receipt


def select_formal_residual_threshold_from_ledger(
    run: LoadedDVMethodRun,
    thresholds: Iterable[float],
    budget: FalseAlarmBudget,
    *,
    method_label: str = "M",
    max_workers: int = 1,
    mp_context: BaseContext | str | None = None,
    progress: ProgressCallback | None = None,
) -> FormalDVThresholdReceipt:
    """Select one residual method with the shared, parallel-capable ledger."""

    if not isinstance(run, LoadedDVMethodRun):
        raise TypeError("run must be a LoadedDVMethodRun")
    if not isinstance(budget, FalseAlarmBudget):
        raise TypeError("budget must be a FalseAlarmBudget")
    if not isinstance(method_label, str) or not method_label:
        raise ValueError("method_label must be a non-empty string")
    if method_label == "Base@B":
        raise ValueError("method_label must identify a residual method")
    run.verify_unchanged()
    grid = _canonical_candidate_grid(thresholds, allow_empty=True)
    occupancy_config = run.artifact.config.occupancy_config
    context = prepare_calibration_context(
        run.base_samples,
        occupancy_config,
        run.artifact.config.match_config,
    )
    ledger = evaluate_candidate_ledger(
        context,
        {method_label: run.residual_samples},
        base_thresholds=(occupancy_config.threshold,),
        residual_thresholds_by_method={method_label: grid},
        max_workers=max_workers,
        mp_context=mp_context,
        progress=progress,
    )
    protocol = _protocol_from_ledger_selection(
        run,
        grid,
        budget,
        ledger.select(method_label, budget),
    )
    receipt = _formal_receipt(run, protocol)
    run.verify_unchanged()
    return receipt


def select_formal_base_threshold(
    run: LoadedDVBaseRun,
    thresholds: Iterable[float],
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    budget: object,
) -> FormalDVBaseThresholdReceipt:
    """Select Base@B from a verified base run without any decoder artifact."""

    if not isinstance(run, LoadedDVBaseRun):
        raise TypeError("run must be a LoadedDVBaseRun")
    run.verify_unchanged()
    protocol = select_base_threshold_on_d_v(
        run.access,
        run.base_samples,
        thresholds,
        occupancy_config,
        match_config,
        budget,
    )
    receipt = _formal_base_receipt(run, protocol)
    run.verify_unchanged()
    return receipt


def _verify_receipt(
    run: LoadedDVMethodRun,
    receipt: FormalDVThresholdReceipt,
    *,
    mode: Literal["residual"] = "residual",
) -> None:
    if not isinstance(run, LoadedDVMethodRun):
        raise TypeError("run must be LoadedDVMethodRun")
    if not isinstance(receipt, FormalDVThresholdReceipt):
        raise TypeError("receipt must be FormalDVThresholdReceipt")
    receipt._verify_source_seal()
    if receipt.mode != mode:
        raise TypeError(f"expected a formal {mode} receipt")
    run.verify_unchanged()
    expected = _formal_receipt(run, receipt.protocol)
    if expected != receipt:
        raise RuntimeError("formal D_V source differs from the threshold receipt")


def _verify_base_receipt(
    run: LoadedDVBaseRun,
    receipt: FormalDVBaseThresholdReceipt,
) -> None:
    if not isinstance(run, LoadedDVBaseRun):
        raise TypeError("run must be LoadedDVBaseRun")
    if not isinstance(receipt, FormalDVBaseThresholdReceipt):
        raise TypeError("receipt must be FormalDVBaseThresholdReceipt")
    receipt._verify_source_seal()
    run.verify_unchanged()
    expected = _formal_base_receipt(run, receipt.protocol)
    if expected != receipt:
        raise RuntimeError("formal Base@B source differs from the threshold receipt")


def evaluate_formal_residual_threshold(
    run: LoadedDVMethodRun,
    receipt: FormalDVThresholdReceipt,
) -> AggregateEvaluation:
    _verify_receipt(run, receipt, mode="residual")
    result = evaluate_frozen_residual_threshold(
        run.access,
        run.residual_samples,
        receipt.protocol,
    )
    run.verify_unchanged()
    return result


def evaluate_formal_residual_fixed_point(
    run: LoadedDVMethodRun,
    threshold: float | None,
    *,
    method_label: str = "fixed",
) -> AggregateEvaluation:
    """Evaluate exactly one residual operating point without selecting it.

    This is the replay primitive for comparing a new method with an already
    frozen historical operating point.  It deliberately constructs no
    threshold-selection receipt and never considers any alternative threshold.
    ``None`` denotes the residual-off anchor candidate.
    """

    if not isinstance(run, LoadedDVMethodRun):
        raise TypeError("run must be a LoadedDVMethodRun")
    if not isinstance(method_label, str) or not method_label:
        raise ValueError("method_label must be a non-empty string")
    if method_label == "Base@B":
        raise ValueError("method_label must identify a residual method")
    if threshold is None:
        grid: tuple[float, ...] = ()
        resolved_threshold = None
    else:
        grid = _canonical_candidate_grid((threshold,), allow_empty=True)
        resolved_threshold = grid[0]

    run.verify_unchanged()
    occupancy_config = run.artifact.config.occupancy_config
    context = prepare_calibration_context(
        run.base_samples,
        occupancy_config,
        run.artifact.config.match_config,
    )
    ledger = evaluate_candidate_ledger(
        context,
        {method_label: run.residual_samples},
        base_thresholds=(occupancy_config.threshold,),
        residual_thresholds_by_method={method_label: grid},
        max_workers=1,
    )
    result = ledger.get(method_label, resolved_threshold).metrics
    run.verify_unchanged()
    return result


def evaluate_formal_base_threshold(
    run: LoadedDVBaseRun,
    receipt: FormalDVBaseThresholdReceipt,
) -> AggregateEvaluation:
    _verify_base_receipt(run, receipt)
    result = evaluate_frozen_base_threshold(
        run.access,
        run.base_samples,
        receipt.protocol,
    )
    run.verify_unchanged()
    return result


def _common_training_fingerprint(
    factual_run: LoadedDVMethodRun,
    exposure_matched_run: LoadedDVMethodRun,
    uniform_run: LoadedDVMethodRun,
) -> str:
    runs = (factual_run, exposure_matched_run, uniform_run)
    expected_variants = (
        "factual_only",
        "factual_exposure_matched",
        "uniform_legal",
    )
    common_payloads: list[dict[str, object]] = []
    for run, expected_variant in zip(runs, expected_variants, strict=True):
        if (
            run.artifact.config.schema_version
            != DECODER_ARTIFACT_SCHEMA_V2
        ):
            raise ValueError(
                "paired v0.1 calibration accepts only decoder artifact v2"
            )
        payload = run.artifact.config.canonical_payload()
        variant = payload.pop("variant")
        payload.pop("variant_contract")
        if variant != expected_variant:
            raise ValueError(
                "paired calibration requires factual_only, "
                "factual_exposure_matched, and uniform_legal"
            )
        common_payloads.append(payload)
    if any(payload != common_payloads[0] for payload in common_payloads[1:]):
        raise RuntimeError(
            "F/Fx/U training contracts differ outside decoder variant"
        )
    initial_fingerprints = {
        run.artifact.config.initial_decoder_fingerprint for run in runs
    }
    if len(initial_fingerprints) != 1:
        raise RuntimeError(
            "F/Fx/U decoder artifacts do not share one initialization"
        )
    if any(
        run.base_samples_fingerprint != factual_run.base_samples_fingerprint
        for run in runs[1:]
    ):
        raise RuntimeError("F/Fx/U runs do not use the same D_V base/GT tensors")
    bundle_fields = (
        "split_manifest_fingerprint",
        "split_manifest_file_sha256",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "base_index_fingerprint",
        "base_index_sha256",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
    )
    if any(
        getattr(factual_run.bundle, name) != getattr(run.bundle, name)
        for run in runs[1:]
        for name in bundle_fields
    ):
        raise RuntimeError("F/Fx/U runs do not use the same verified D_V bundle")
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-paired-gate2-training-contract-v2",
            "common_run_config": common_payloads[0],
            "initial_decoder_fingerprint": (
                factual_run.artifact.config.initial_decoder_fingerprint
            ),
            "d_v_base_samples_fingerprint": (
                factual_run.base_samples_fingerprint
            ),
        }
    )


def _verify_base_run_alignment(
    base_run: LoadedDVBaseRun,
    *method_runs: LoadedDVMethodRun,
) -> None:
    """Require decoder-free Base@B and residual methods to share exact D_V."""

    if not isinstance(base_run, LoadedDVBaseRun):
        raise TypeError("base_run must be a LoadedDVBaseRun")
    base_run.verify_unchanged()
    provenance_fields = (
        "split_manifest_fingerprint",
        "split_manifest_file_sha256",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "base_index_fingerprint",
        "base_index_sha256",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
    )
    for method_run in method_runs:
        if not isinstance(method_run, LoadedDVMethodRun):
            raise TypeError("method runs must be LoadedDVMethodRun")
        method_run.verify_unchanged()
        if any(
            getattr(method_run.bundle, name) != getattr(base_run.bundle, name)
            for name in provenance_fields
        ):
            raise RuntimeError("Base@B and residual methods use different D_V caches")
        if (
            method_run.base_samples_fingerprint
            != base_run.base_samples_fingerprint
            or tuple(sample.sample_id for sample in method_run.base_samples)
            != base_run.ordered_d_v_sample_ids
        ):
            raise RuntimeError(
                "Base@B and residual methods use different D_V base/GT tensors"
            )


@dataclass(frozen=True)
class PairedGate2Calibration:
    """The complete A/Base@B/F/Fx/U D_V selection with shared contracts."""

    anchor: FrozenAnchorReceipt
    base_at_budget: FormalDVBaseThresholdReceipt
    factual_only: FormalDVThresholdReceipt
    factual_exposure_matched: FormalDVThresholdReceipt
    uniform_legal: FormalDVThresholdReceipt
    common_training_fingerprint: str
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _PairedCalibrationSeal:
            raise TypeError(
                "PairedGate2Calibration must come from paired Gate-2 calibration"
            )
        if (
            seal.anchor is not self.anchor
            or seal.base_at_budget is not self.base_at_budget
            or seal.factual_only is not self.factual_only
            or seal.factual_exposure_matched is not self.factual_exposure_matched
            or seal.uniform_legal is not self.uniform_legal
            or seal.common_training_fingerprint
            != self.common_training_fingerprint
        ):
            raise TypeError("paired Gate-2 calibration fields were replaced")
        if not isinstance(seal.ledger, CalibrationCandidateLedger):
            raise TypeError("paired Gate-2 calibration ledger was replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        _digest(
            self.common_training_fingerprint,
            name="common_training_fingerprint",
        )
        if not isinstance(self.anchor, FrozenAnchorReceipt):
            raise TypeError("anchor must be a decoder-free FrozenAnchorReceipt")
        self.anchor._verify_source_seal()
        if not isinstance(self.base_at_budget, FormalDVBaseThresholdReceipt):
            raise TypeError("Base@B must use a decoder-free formal receipt")
        self.base_at_budget._verify_source_seal()
        if self.factual_only.mode != "residual":
            raise ValueError("F receipt has the wrong mode")
        if self.factual_exposure_matched.mode != "residual":
            raise ValueError("Fx receipt has the wrong mode")
        if self.uniform_legal.mode != "residual":
            raise ValueError("U receipt has the wrong mode")
        if self.factual_only.decoder_variant != "factual_only":
            raise ValueError("F receipt is not bound to the factual-only decoder")
        if (
            self.factual_exposure_matched.decoder_variant
            != "factual_exposure_matched"
        ):
            raise ValueError(
                "Fx receipt is not bound to the exposure-matched decoder"
            )
        if self.uniform_legal.decoder_variant != "uniform_legal":
            raise ValueError("U receipt is not bound to the Uniform-Legal decoder")
        common_protocols = (
            self.base_at_budget.protocol,
            self.factual_only.protocol,
            self.factual_exposure_matched.protocol,
            self.uniform_legal.protocol,
        )
        reference = common_protocols[0]
        if any(
            protocol.manifest_fingerprint != reference.manifest_fingerprint
            or protocol.ordered_d_v_sample_ids
            != reference.ordered_d_v_sample_ids
            or protocol.occupancy_config != reference.occupancy_config
            or protocol.match_config != reference.match_config
            or protocol.budget != reference.budget
            for protocol in common_protocols[1:]
        ):
            raise ValueError("Base@B/F/Fx/U do not share one D_V protocol")
        if (
            self.anchor.manifest_fingerprint != reference.manifest_fingerprint
            or self.anchor.ordered_d_v_sample_ids
            != reference.ordered_d_v_sample_ids
            or self.anchor.base_samples_fingerprint
            != self.base_at_budget.protocol.sample_tensor_fingerprint
            or self.anchor.occupancy_config != reference.occupancy_config
            or self.anchor.match_config != reference.match_config
        ):
            raise ValueError("A/Base@B/F/Fx/U do not share one D_V anchor protocol")
        residual_protocols = (
            self.factual_only.protocol,
            self.factual_exposure_matched.protocol,
            self.uniform_legal.protocol,
        )
        if any(
            protocol.candidate_threshold_grid
            != residual_protocols[0].candidate_threshold_grid
            for protocol in residual_protocols[1:]
        ):
            raise ValueError("F/Fx/U residual threshold grids differ")
        provenance_fields = (
            "manifest_file_sha256",
            "base_index_fingerprint",
            "base_index_sha256",
            "d_v_image_fingerprint",
            "d_v_gt_fingerprint",
            "preprocessing_fingerprint",
            "base_fingerprint",
        )
        if any(
            getattr(receipt, name) != getattr(self.anchor, name)
            for receipt in common_protocols_receipts(self)
            for name in provenance_fields
        ):
            raise ValueError("A/Base@B/F/Fx/U provenance differs")
        formal_receipts = (
            self.factual_only,
            self.factual_exposure_matched,
            self.uniform_legal,
        )
        if any(
            receipt.global_seed != formal_receipts[0].global_seed
            for receipt in formal_receipts[1:]
        ):
            raise ValueError("F/Fx/U global seeds differ")

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": "cure-lite-paired-gate2-calibration-v4",
                "anchor": self.anchor.receipt_fingerprint,
                "base_at_budget": self.base_at_budget.receipt_fingerprint,
                "factual_only": self.factual_only.receipt_fingerprint,
                "factual_exposure_matched": (
                    self.factual_exposure_matched.receipt_fingerprint
                ),
                "uniform_legal": self.uniform_legal.receipt_fingerprint,
                "common_training_fingerprint": self.common_training_fingerprint,
            }
        )


def common_protocols_receipts(
    calibration: PairedGate2Calibration,
) -> tuple[
    FormalDVBaseThresholdReceipt | FormalDVThresholdReceipt,
    ...,
]:
    return (
        calibration.base_at_budget,
        calibration.factual_only,
        calibration.factual_exposure_matched,
        calibration.uniform_legal,
    )


@dataclass(frozen=True, slots=True)
class _PairedCalibrationSeal:
    anchor: FrozenAnchorReceipt
    base_at_budget: FormalDVBaseThresholdReceipt
    factual_only: FormalDVThresholdReceipt
    factual_exposure_matched: FormalDVThresholdReceipt
    uniform_legal: FormalDVThresholdReceipt
    common_training_fingerprint: str
    ledger: CalibrationCandidateLedger


@dataclass(frozen=True)
class Gate2DVResults:
    anchor: AggregateEvaluation
    base_at_budget: AggregateEvaluation
    factual_only: AggregateEvaluation
    factual_exposure_matched: AggregateEvaluation
    uniform_legal: AggregateEvaluation


def calibrate_paired_gate2(
    base_run: LoadedDVBaseRun,
    factual_run: LoadedDVMethodRun,
    exposure_matched_run: LoadedDVMethodRun,
    uniform_run: LoadedDVMethodRun,
    *,
    anchor: FrozenAnchorReceipt,
    residual_thresholds: Iterable[float],
    base_thresholds: Iterable[float],
    budget: FalseAlarmBudget,
    max_workers: int = 1,
    mp_context: BaseContext | str | None = None,
    progress: ProgressCallback | None = None,
) -> PairedGate2Calibration:
    """Select A/Base@B/F/Fx/U from one exact shared candidate ledger."""

    if not isinstance(anchor, FrozenAnchorReceipt):
        raise TypeError("anchor must be a FrozenAnchorReceipt")
    if not isinstance(budget, FalseAlarmBudget):
        raise TypeError("budget must be a FalseAlarmBudget")
    base_run.verify_unchanged()
    factual_run.verify_unchanged()
    exposure_matched_run.verify_unchanged()
    uniform_run.verify_unchanged()
    common_fingerprint = _common_training_fingerprint(
        factual_run,
        exposure_matched_run,
        uniform_run,
    )
    _verify_base_run_alignment(
        base_run,
        factual_run,
        exposure_matched_run,
        uniform_run,
    )
    evaluate_frozen_anchor(base_run, anchor)
    if factual_run.artifact.config.occupancy_config != anchor.occupancy_config:
        raise RuntimeError("decoder occupancy config differs from frozen tau_o")
    if factual_run.artifact.config.match_config != anchor.match_config:
        raise RuntimeError("decoder matching config differs from frozen anchor")
    residual_grid = _canonical_candidate_grid(
        residual_thresholds,
        allow_empty=True,
    )
    base_grid = _canonical_candidate_grid(
        base_thresholds,
        allow_empty=False,
    )
    context = prepare_calibration_context(
        base_run.base_samples,
        anchor.occupancy_config,
        anchor.match_config,
    )
    if context.anchor_metrics != anchor.selected_metrics:
        raise RuntimeError("prepared calibration anchor differs from frozen A")
    ledger = evaluate_candidate_ledger(
        context,
        {
            "F": factual_run.residual_samples,
            "F×": exposure_matched_run.residual_samples,
            "U": uniform_run.residual_samples,
        },
        base_thresholds=base_grid,
        residual_thresholds_by_method={
            "F": residual_grid,
            "F×": residual_grid,
            "U": residual_grid,
        },
        max_workers=max_workers,
        mp_context=mp_context,
        progress=progress,
    )
    base_at_budget = _formal_base_receipt(
        base_run,
        _base_protocol_from_ledger_selection(
            base_run,
            base_grid,
            anchor.occupancy_config,
            anchor.match_config,
            budget,
            ledger.select("Base@B", budget),
        ),
    )
    factual_only = _formal_receipt(
        factual_run,
        _protocol_from_ledger_selection(
            factual_run,
            residual_grid,
            budget,
            ledger.select("F", budget),
            variant="residual",
        ),
    )
    factual_exposure_matched = _formal_receipt(
        exposure_matched_run,
        _protocol_from_ledger_selection(
            exposure_matched_run,
            residual_grid,
            budget,
            ledger.select("F×", budget),
            variant="residual",
        ),
    )
    uniform_legal = _formal_receipt(
        uniform_run,
        _protocol_from_ledger_selection(
            uniform_run,
            residual_grid,
            budget,
            ledger.select("U", budget),
            variant="residual",
        ),
    )
    seal = _PairedCalibrationSeal(
        anchor=anchor,
        base_at_budget=base_at_budget,
        factual_only=factual_only,
        factual_exposure_matched=factual_exposure_matched,
        uniform_legal=uniform_legal,
        common_training_fingerprint=common_fingerprint,
        ledger=ledger,
    )
    calibration = PairedGate2Calibration(
        anchor=anchor,
        base_at_budget=base_at_budget,
        factual_only=factual_only,
        factual_exposure_matched=factual_exposure_matched,
        uniform_legal=uniform_legal,
        common_training_fingerprint=common_fingerprint,
        _verification_token=seal,
    )
    base_run.verify_unchanged()
    factual_run.verify_unchanged()
    exposure_matched_run.verify_unchanged()
    uniform_run.verify_unchanged()
    return calibration


def evaluate_paired_gate2(
    base_run: LoadedDVBaseRun,
    factual_run: LoadedDVMethodRun,
    exposure_matched_run: LoadedDVMethodRun,
    uniform_run: LoadedDVMethodRun,
    calibration: PairedGate2Calibration,
) -> Gate2DVResults:
    """Materialize results already established by the shared-grid ledger.

    A serialized Stage-A replay calls :func:`calibrate_paired_gate2` again and
    therefore recomputes the complete ledger.  Re-running four independent
    legacy selectors here would duplicate that same grid inside one state
    build without adding an independent evidence boundary.
    """

    if not isinstance(calibration, PairedGate2Calibration):
        raise TypeError("calibration must be PairedGate2Calibration")
    calibration._verify_source_seal()
    seal = calibration._verification_token
    assert type(seal) is _PairedCalibrationSeal
    _verify_base_run_alignment(
        base_run,
        factual_run,
        exposure_matched_run,
        uniform_run,
    )
    if (
        _common_training_fingerprint(
            factual_run,
            exposure_matched_run,
            uniform_run,
        )
        != calibration.common_training_fingerprint
    ):
        raise RuntimeError("paired training contract differs from calibration")
    _verify_base_receipt(base_run, calibration.base_at_budget)
    receipts = (
        (factual_run, calibration.factual_only, "residual"),
        (
            exposure_matched_run,
            calibration.factual_exposure_matched,
            "residual",
        ),
        (uniform_run, calibration.uniform_legal, "residual"),
    )
    for run, receipt, mode in receipts:
        _verify_receipt(run, receipt, mode=mode)

    selected = {
        method: seal.ledger.select(method, calibration.base_at_budget.protocol.budget)
        for method in ("Base@B", "F", "F×", "U")
    }
    protocols = {
        "Base@B": calibration.base_at_budget.protocol,
        "F": calibration.factual_only.protocol,
        "F×": calibration.factual_exposure_matched.protocol,
        "U": calibration.uniform_legal.protocol,
    }
    if any(
        choice.threshold != protocols[method].selected_threshold
        or choice.metrics != protocols[method].selected_metrics
        for method, choice in selected.items()
    ):
        raise RuntimeError("paired candidate ledger differs from frozen receipts")

    # Independently rebuild fixed anchor/GT state and evaluate only the four
    # selected points.  The complete ledger establishes global selection; this
    # small second pass retains the independent fixed-threshold consistency
    # check without repeating every candidate grid four more times.
    verification_context = prepare_calibration_context(
        base_run.base_samples,
        calibration.anchor.occupancy_config,
        calibration.anchor.match_config,
    )
    if verification_context.anchor_metrics != calibration.anchor.selected_metrics:
        raise RuntimeError("fixed-point verification anchor differs from A")
    residual_verification_grids = {
        method: (
            ()
            if protocols[method].selected_threshold is None
            else (protocols[method].selected_threshold,)
        )
        for method in ("F", "F×", "U")
    }
    verification_ledger = evaluate_candidate_ledger(
        verification_context,
        {
            "F": factual_run.residual_samples,
            "F×": exposure_matched_run.residual_samples,
            "U": uniform_run.residual_samples,
        },
        base_thresholds=(protocols["Base@B"].selected_threshold,),
        residual_thresholds_by_method=residual_verification_grids,
        max_workers=1,
    )
    verified = {
        method: verification_ledger.select(
            method,
            calibration.base_at_budget.protocol.budget,
        )
        for method in ("Base@B", "F", "F×", "U")
    }
    if any(
        choice.threshold != protocols[method].selected_threshold
        or choice.metrics != protocols[method].selected_metrics
        for method, choice in verified.items()
    ):
        raise RuntimeError("selected-point verification differs from frozen receipts")

    result = Gate2DVResults(
        anchor=evaluate_frozen_anchor(base_run, calibration.anchor),
        base_at_budget=verified["Base@B"].metrics,
        factual_only=verified["F"].metrics,
        factual_exposure_matched=verified["F×"].metrics,
        uniform_legal=verified["U"].metrics,
    )
    base_run.verify_unchanged()
    factual_run.verify_unchanged()
    exposure_matched_run.verify_unchanged()
    uniform_run.verify_unchanged()
    return result


__all__ = [
    "FormalDVBaseThresholdReceipt",
    "FormalDVThresholdReceipt",
    "build_loaded_d_v_method_run",
    "calibrate_paired_gate2",
    "evaluate_formal_base_threshold",
    "evaluate_formal_residual_fixed_point",
    "evaluate_formal_residual_threshold",
    "evaluate_paired_gate2",
    "select_formal_base_threshold",
    "select_formal_residual_threshold",
    "select_formal_residual_threshold_from_ledger",
]
