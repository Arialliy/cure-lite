"""Strict D_V calibration over verified caches and immutable decoder artifacts.

This is the formal Gate-2 layer above the tensor-level calibration helpers.  It
binds every D_V result to the manifest, image/GT catalogs, base-cache index,
trained decoder artifact, common method configuration, threshold grid, and
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
from ..metrics import AggregateEvaluation
from ..splits import SplitManifest
from .artifacts import LoadedDecoderArtifact
from .cache_pipeline import LoadedDVCacheBundle
from .formal_anchor import (
    FrozenAnchorReceipt,
    build_loaded_d_v_base_run,
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
    """A tensor-level threshold protocol bound to its cache and decoder source."""

    mode: Literal["residual", "base_at_budget"]
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
        if self.mode not in {"residual", "base_at_budget"}:
            raise ValueError("unsupported formal D_V receipt mode")
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


def _formal_receipt(
    run: LoadedDVMethodRun,
    protocol: BoundDVThresholdProtocol,
) -> FormalDVThresholdReceipt:
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
    variant: Literal["residual", "base_at_budget"],
) -> BoundDVThresholdProtocol:
    """Bind one exact ledger selection to the existing formal receipt type."""

    if not isinstance(selection, ThresholdSelection):
        raise TypeError("selection must be a ThresholdSelection")
    if not selection.feasible or selection.metrics is None:
        raise RuntimeError(
            selection.reason or f"D_V {variant} selection is infeasible"
        )
    if variant == "base_at_budget" and selection.threshold is None:
        raise RuntimeError("D_V Base@B selection must be numeric")
    samples = run.base_samples if variant == "base_at_budget" else run.residual_samples
    sample_fingerprint = (
        run.base_samples_fingerprint
        if variant == "base_at_budget"
        else run.residual_samples_fingerprint
    )
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
            allow_empty=variant == "residual",
        ),
        occupancy_config=run.artifact.config.occupancy_config,
        match_config=run.artifact.config.match_config,
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


def select_formal_base_threshold(
    run: LoadedDVMethodRun,
    thresholds: Iterable[float],
    budget: object,
) -> FormalDVThresholdReceipt:
    run.verify_unchanged()
    protocol = select_base_threshold_on_d_v(
        run.access,
        run.base_samples,
        thresholds,
        run.artifact.config.occupancy_config,
        run.artifact.config.match_config,
        budget,
    )
    receipt = _formal_receipt(run, protocol)
    run.verify_unchanged()
    return receipt


def _verify_receipt(
    run: LoadedDVMethodRun,
    receipt: FormalDVThresholdReceipt,
    *,
    mode: Literal["residual", "base_at_budget"],
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


def evaluate_formal_base_threshold(
    run: LoadedDVMethodRun,
    receipt: FormalDVThresholdReceipt,
) -> AggregateEvaluation:
    _verify_receipt(run, receipt, mode="base_at_budget")
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


@dataclass(frozen=True)
class PairedGate2Calibration:
    """The complete A/Base@B/F/Fx/U D_V selection with shared contracts."""

    anchor: FrozenAnchorReceipt
    base_at_budget: FormalDVThresholdReceipt
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
        if self.base_at_budget.mode != "base_at_budget":
            raise ValueError("Base@B receipt has the wrong mode")
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
        formal_receipts = common_protocols_receipts(self)
        if any(
            receipt.global_seed != formal_receipts[0].global_seed
            for receipt in formal_receipts[1:]
        ):
            raise ValueError("Base@B/F/Fx/U global seeds differ")

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": "cure-lite-paired-gate2-calibration-v3",
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
) -> tuple[FormalDVThresholdReceipt, ...]:
    return (
        calibration.base_at_budget,
        calibration.factual_only,
        calibration.factual_exposure_matched,
        calibration.uniform_legal,
    )


@dataclass(frozen=True, slots=True)
class _PairedCalibrationSeal:
    anchor: FrozenAnchorReceipt
    base_at_budget: FormalDVThresholdReceipt
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
    factual_run.verify_unchanged()
    exposure_matched_run.verify_unchanged()
    uniform_run.verify_unchanged()
    common_fingerprint = _common_training_fingerprint(
        factual_run,
        exposure_matched_run,
        uniform_run,
    )
    anchor_run = build_loaded_d_v_base_run(factual_run.bundle)
    evaluate_frozen_anchor(anchor_run, anchor)
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
        factual_run.base_samples,
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
    base_at_budget = _formal_receipt(
        factual_run,
        _protocol_from_ledger_selection(
            factual_run,
            base_grid,
            budget,
            ledger.select("Base@B", budget),
            variant="base_at_budget",
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
    factual_run.verify_unchanged()
    exposure_matched_run.verify_unchanged()
    uniform_run.verify_unchanged()
    return calibration


def evaluate_paired_gate2(
    factual_run: LoadedDVMethodRun,
    exposure_matched_run: LoadedDVMethodRun,
    uniform_run: LoadedDVMethodRun,
    calibration: PairedGate2Calibration,
) -> Gate2DVResults:
    """Materialize results already proven by the sealed shared-grid ledger.

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
    if (
        _common_training_fingerprint(
            factual_run,
            exposure_matched_run,
            uniform_run,
        )
        != calibration.common_training_fingerprint
    ):
        raise RuntimeError("paired training contract differs from calibration")
    receipts = (
        (factual_run, calibration.base_at_budget, "base_at_budget"),
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
    # selected points.  The sealed full ledger proves global selection; this
    # small second pass preserves the former fail-closed fixed-threshold check
    # without repeating every candidate grid four more times.
    verification_context = prepare_calibration_context(
        factual_run.base_samples,
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

    anchor_run = build_loaded_d_v_base_run(factual_run.bundle)
    result = Gate2DVResults(
        anchor=evaluate_frozen_anchor(anchor_run, calibration.anchor),
        base_at_budget=verified["Base@B"].metrics,
        factual_only=verified["F"].metrics,
        factual_exposure_matched=verified["F×"].metrics,
        uniform_legal=verified["U"].metrics,
    )
    factual_run.verify_unchanged()
    exposure_matched_run.verify_unchanged()
    uniform_run.verify_unchanged()
    return result


__all__ = [
    "build_loaded_d_v_method_run",
    "calibrate_paired_gate2",
    "evaluate_paired_gate2",
]
