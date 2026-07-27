"""Strict formal D_V evaluation for completed real PFCR attempts.

This layer deliberately does not accept a decoder or a decoder artifact as
the evaluation source.  A PFCR decoder becomes eligible for D_V only through
an exact :class:`PublishedPFCRRealFormalAttempt`, which proves that the
predeclared 800 x 40 D_R training run completed and that its artifact is
strictly loadable.

The threshold grid, null candidate, occupancy/matching rules, false-addition
budget, D_V population and frozen cache provenance all come from the existing
``paired_formal_evaluation_v1`` comparison protocol.  Callers cannot override
any of them and this module exposes no D_T path.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Mapping

import torch

from ..cache.schema import stable_fingerprint
from ..calibration import CalibrationSample, FalseAlarmBudget
from ..metrics import AggregateEvaluation
from ..phase_resolved_real_cache import PFCRRealCacheAdapter
from ..phase_resolved_relation_decoder import (
    CURELitePhaseResolvedRelationDecoder,
)
from ..splits import load_and_validate_manifest
from .cache_pipeline import LoadedDVCacheBundle
from .evaluation_pipeline import (
    BoundDVThresholdProtocol,
    DevelopmentSplitAccess,
    calibration_samples_fingerprint,
    evaluate_frozen_residual_threshold,
    select_residual_threshold_on_d_v,
)
from .paired_formal_decision import FORMAL_SEEDS, FormalMethodEvidence
from .paired_formal_evaluation import (
    FORMAL_DV_ANCHOR_COVERED,
    FORMAL_DV_ANCHOR_MISSES,
    FORMAL_DV_IMAGES,
    FORMAL_DV_TOTAL_TARGETS,
    FrozenComparisonProtocol,
)
from .phase_resolved_real_artifacts import (
    LoadedPFCRRealDecoderArtifact,
)
from .phase_resolved_real_evaluation import (
    PFCRDVSamples,
    _build_pfcr_d_v_samples,
)
from .phase_resolved_real_formal_runner import (
    PublishedPFCRRealFormalAttempt,
    load_pfcr_real_formal_attempt,
)
from .phase_resolved_real_training import pfcr_model_state_fingerprint


PFCR_FORMAL_DV_RESULT_SCHEMA = "cure-lite-pfcr-formal-d-v-result-v1"
PFCR_FORMAL_DV_RUN_SCHEMA = "cure-lite-loaded-pfcr-formal-d-v-run-v1"
PFCR_FORMAL_METHOD = "PFCR"
PFCR_FORMAL_DV_BATCH_SIZE = 8
_HEX = frozenset("0123456789abcdef")
_AGGREGATE_FIELDS = tuple(field.name for field in fields(AggregateEvaluation))
_AGGREGATE_INTEGER_FIELDS = frozenset(
    {
        "images",
        "recovered_anchor_misses",
        "net_recovered_anchor_misses",
        "total_anchor_misses",
        "retained_anchor_covered",
        "total_anchor_covered",
        "recovered_reachable_anchor_misses",
        "total_reachable_anchor_misses",
    }
)
_AGGREGATE_BOOL_FIELDS = frozenset({"budget_violation"})


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _threshold(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    result = _finite(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _aggregate_payload(metrics: AggregateEvaluation) -> dict[str, object]:
    if not isinstance(metrics, AggregateEvaluation):
        raise TypeError("metrics must be AggregateEvaluation")
    return {name: getattr(metrics, name) for name in _AGGREGATE_FIELDS}


def _aggregate_from_payload(value: object) -> AggregateEvaluation:
    if not isinstance(value, Mapping) or set(value) != set(_AGGREGATE_FIELDS):
        raise ValueError("aggregate_evaluation fields are not canonical")
    normalized: dict[str, object] = {}
    for name in _AGGREGATE_FIELDS:
        item = value[name]
        if name in _AGGREGATE_BOOL_FIELDS:
            if not isinstance(item, bool):
                raise TypeError(f"aggregate_evaluation.{name} must be bool")
            normalized[name] = item
        elif name in _AGGREGATE_INTEGER_FIELDS:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"aggregate_evaluation.{name} must be an integer"
                )
            normalized[name] = item
        else:
            normalized[name] = _finite(
                item,
                name=f"aggregate_evaluation.{name}",
            )
    return AggregateEvaluation(**normalized)  # type: ignore[arg-type]


def _budget_payload(budget: FalseAlarmBudget) -> dict[str, float]:
    if not isinstance(budget, FalseAlarmBudget):
        raise TypeError("budget must be FalseAlarmBudget")
    values = {
        "pixel_fa_budget": budget.pixel_fa_budget,
        "component_fa_per_mp_budget": budget.component_fa_per_mp_budget,
        "raw_background_fa_budget": budget.raw_background_fa_budget,
        "minimum_retention": budget.minimum_retention,
    }
    if any(not isfinite(value) for value in values.values()):
        raise ValueError("formal PFCR budget fields must be finite")
    return values


def _budget_from_payload(value: object) -> FalseAlarmBudget:
    expected = {
        "pixel_fa_budget",
        "component_fa_per_mp_budget",
        "raw_background_fa_budget",
        "minimum_retention",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("false_alarm_budget fields are not canonical")
    return FalseAlarmBudget(
        pixel_fa_budget=_finite(
            value["pixel_fa_budget"],
            name="false_alarm_budget.pixel_fa_budget",
        ),
        component_fa_per_mp_budget=_finite(
            value["component_fa_per_mp_budget"],
            name="false_alarm_budget.component_fa_per_mp_budget",
        ),
        raw_background_fa_budget=_finite(
            value["raw_background_fa_budget"],
            name="false_alarm_budget.raw_background_fa_budget",
        ),
        minimum_retention=_finite(
            value["minimum_retention"],
            name="false_alarm_budget.minimum_retention",
        ),
    )


def _population_counts(
    metrics: AggregateEvaluation,
) -> tuple[int, int, int, int]:
    """Validate the exact frozen formal D_V population and count identities."""

    if not isinstance(metrics, AggregateEvaluation):
        raise TypeError("metrics must be AggregateEvaluation")
    if metrics.images != FORMAL_DV_IMAGES:
        raise ValueError(
            "formal PFCR D_V evaluation must contain exactly 120 images"
        )
    if metrics.total_anchor_misses != FORMAL_DV_ANCHOR_MISSES:
        raise ValueError(
            "formal PFCR D_V anchor-miss denominator must remain exactly 23"
        )
    if metrics.total_anchor_covered != FORMAL_DV_ANCHOR_COVERED:
        raise ValueError(
            "formal PFCR D_V anchor-covered denominator must remain exactly 147"
        )
    total_targets = (
        metrics.total_anchor_misses + metrics.total_anchor_covered
    )
    if total_targets != FORMAL_DV_TOTAL_TARGETS:
        raise ValueError(
            "formal PFCR D_V target count must remain exactly 170"
        )
    if not (
        0 <= metrics.recovered_anchor_misses <= metrics.total_anchor_misses
        and 0
        <= metrics.retained_anchor_covered
        <= metrics.total_anchor_covered
        and 0
        <= metrics.recovered_reachable_anchor_misses
        <= metrics.total_reachable_anchor_misses
        <= metrics.total_anchor_misses
        and metrics.recovered_reachable_anchor_misses
        <= metrics.recovered_anchor_misses
    ):
        raise ValueError("formal PFCR D_V recovery counts are inconsistent")
    true_targets = (
        metrics.retained_anchor_covered + metrics.recovered_anchor_misses
    )
    expected_pd = true_targets / total_targets
    expected_retention = (
        metrics.retained_anchor_covered / metrics.total_anchor_covered
    )
    expected_recovery = (
        metrics.recovered_anchor_misses / metrics.total_anchor_misses
    )
    expected_net = true_targets - metrics.total_anchor_covered
    if not isclose(metrics.pd, expected_pd, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("PFCR Pd is not the exact matched-target fraction")
    if not isclose(
        metrics.retention,
        expected_retention,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "PFCR retention is not the exact anchor-covered fraction"
        )
    if not (
        isclose(
            metrics.rmr,
            expected_recovery,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and isclose(
            metrics.gross_rmr,
            expected_recovery,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("PFCR gross recovery ratio is count-inconsistent")
    if (
        metrics.net_recovered_anchor_misses != expected_net
        or not isclose(
            metrics.net_rmr,
            expected_net / metrics.total_anchor_misses,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("PFCR net recovery is count-inconsistent")
    reachable = (
        metrics.recovered_reachable_anchor_misses
        / metrics.total_reachable_anchor_misses
        if metrics.total_reachable_anchor_misses
        else 0.0
    )
    if not isclose(
        metrics.reachable_rmr,
        reachable,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("PFCR reachable recovery is count-inconsistent")
    if not isclose(
        metrics.oracle_upper_bound,
        metrics.total_reachable_anchor_misses
        / metrics.total_anchor_misses,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("PFCR oracle upper bound is count-inconsistent")
    unit_fields = (
        "pd",
        "rmr",
        "gross_rmr",
        "retention",
        "reachable_rmr",
        "oracle_upper_bound",
        "overlap_supported_rmr",
        "miou",
        "niou",
    )
    if any(
        not 0.0 <= _finite(getattr(metrics, name), name=name) <= 1.0
        for name in unit_fields
    ):
        raise ValueError("formal PFCR D_V unit metrics must lie in [0,1]")
    if metrics.overlap_supported_rmr > metrics.gross_rmr:
        raise ValueError(
            "PFCR overlap-supported recovery exceeds gross recovery"
        )
    if any(
        _finite(getattr(metrics, name), name=name) < 0.0
        for name in (
            "pixel_fa",
            "raw_background_fa",
            "fp_components_per_mp",
        )
    ):
        raise ValueError("formal PFCR false-addition metrics are negative")
    if not isinstance(metrics.budget_violation, bool):
        raise TypeError("PFCR budget_violation must be bool")
    return (
        total_targets,
        true_targets,
        metrics.total_anchor_misses,
        metrics.recovered_anchor_misses,
    )


def _verify_published_attempt(
    attempt: PublishedPFCRRealFormalAttempt,
) -> None:
    """Re-authenticate the completed attempt and its immutable artifact."""

    if not isinstance(attempt, PublishedPFCRRealFormalAttempt):
        raise TypeError(
            "attempt must be PublishedPFCRRealFormalAttempt; "
            "bare PFCR artifacts are not eligible for formal D_V"
        )
    if not isinstance(attempt.artifact, LoadedPFCRRealDecoderArtifact):
        raise TypeError("published PFCR attempt has an invalid artifact")
    attempt.artifact.verify_unchanged()
    reloaded = load_pfcr_real_formal_attempt(attempt.root)
    if (
        reloaded.seed != attempt.seed
        or reloaded.run_receipt_fingerprint
        != attempt.run_receipt_fingerprint
        or reloaded.complete_fingerprint != attempt.complete_fingerprint
        or reloaded.artifact.artifact_fingerprint
        != attempt.artifact.artifact_fingerprint
        or reloaded.artifact.receipt_sha256
        != attempt.artifact.receipt_sha256
        or reloaded.artifact.decoder_state_fingerprint
        != attempt.artifact.decoder_state_fingerprint
    ):
        raise RuntimeError(
            "published PFCR formal attempt changed since strict loading"
        )
    config = attempt.artifact.config
    ledger = attempt.artifact.execution_ledger
    if (
        attempt.seed not in FORMAL_SEEDS
        or config.seed != attempt.seed
        or ledger.seed != attempt.seed
        or ledger.optimizer_updates != 32_000
        or ledger.minimum_adam_step != 32_000
        or ledger.maximum_adam_step != 32_000
    ):
        raise RuntimeError(
            "PFCR formal attempt does not prove the frozen 800x40 run"
        )


def _base_only_samples(
    samples: tuple[CalibrationSample, ...],
) -> tuple[CalibrationSample, ...]:
    result: list[CalibrationSample] = []
    for sample in samples:
        base, _, gt = sample.normalized()
        result.append(
            CalibrationSample(
                sample.sample_id,
                base,
                torch.zeros_like(base),
                gt,
            )
        )
    return tuple(result)


def _validated_execution_device(
    value: torch.device | str,
) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must be an explicit CPU or CUDA device") from error
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU device may not include an index")
        return device
    if device.type != "cuda" or device.index is None:
        raise ValueError("CUDA device must include an explicit index")
    if (
        not torch.cuda.is_available()
        or device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError("requested PFCR evaluation CUDA device unavailable")
    return device


def _frozen_decoder_clone(
    artifact: LoadedPFCRRealDecoderArtifact,
    *,
    device: torch.device,
) -> CURELitePhaseResolvedRelationDecoder:
    """Clone a sealed artifact onto an execution device without mutating it."""

    artifact.verify_unchanged()
    decoder = CURELitePhaseResolvedRelationDecoder(
        artifact.config.decoder_config
    )
    decoder.load_state_dict(artifact.decoder.state_dict(), strict=True)
    decoder.to(device=device, dtype=torch.float32)
    decoder.eval()
    decoder.requires_grad_(False)
    if (
        pfcr_model_state_fingerprint(decoder)
        != artifact.decoder_state_fingerprint
    ):
        raise RuntimeError("device clone differs from frozen PFCR artifact")
    artifact.verify_unchanged()
    return decoder


@dataclass(frozen=True, slots=True)
class _LoadedPFCRFormalDVRunSeal:
    bundle: LoadedDVCacheBundle
    d_r_cache: PFCRRealCacheAdapter
    attempt: PublishedPFCRRealFormalAttempt
    comparison_protocol: FrozenComparisonProtocol
    access: DevelopmentSplitAccess
    d_v_samples: PFCRDVSamples
    base_samples: tuple[CalibrationSample, ...]
    base_samples_fingerprint: str


@dataclass(frozen=True)
class LoadedPFCRFormalDVRun:
    """One completed PFCR attempt evaluated over one exact D_V bundle."""

    bundle: LoadedDVCacheBundle
    d_r_cache: PFCRRealCacheAdapter
    attempt: PublishedPFCRRealFormalAttempt
    comparison_protocol: FrozenComparisonProtocol
    access: DevelopmentSplitAccess
    d_v_samples: PFCRDVSamples
    base_samples: tuple[CalibrationSample, ...]
    base_samples_fingerprint: str
    _verification_token: object

    def _seal(self) -> _LoadedPFCRFormalDVRunSeal:
        seal = self._verification_token
        if type(seal) is not _LoadedPFCRFormalDVRunSeal:
            raise TypeError(
                "LoadedPFCRFormalDVRun must come from its strict builder"
            )
        if (
            seal.bundle is not self.bundle
            or seal.d_r_cache is not self.d_r_cache
            or seal.attempt is not self.attempt
            or seal.comparison_protocol is not self.comparison_protocol
            or seal.access is not self.access
            or seal.d_v_samples is not self.d_v_samples
            or seal.base_samples is not self.base_samples
            or seal.base_samples_fingerprint
            != self.base_samples_fingerprint
        ):
            raise TypeError("loaded PFCR D_V run source objects were replaced")
        return seal

    def __post_init__(self) -> None:
        self._seal()
        _digest(
            self.base_samples_fingerprint,
            name="base_samples_fingerprint",
        )
        self.verify_unchanged()

    @property
    def seed(self) -> int:
        return self.attempt.seed

    @property
    def residual_samples_fingerprint(self) -> str:
        return self.d_v_samples.sample_tensor_fingerprint

    @property
    def run_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(
            {
                "schema_version": PFCR_FORMAL_DV_RUN_SCHEMA,
                "runtime_split": "D_V",
                "method": PFCR_FORMAL_METHOD,
                "seed": self.seed,
                "comparison_protocol_fingerprint": (
                    self.comparison_protocol.comparison_protocol_fingerprint
                ),
                "manifest_fingerprint": (
                    self.bundle.split_manifest_fingerprint
                ),
                "manifest_file_sha256": (
                    self.bundle.split_manifest_file_sha256
                ),
                "preprocessing_fingerprint": (
                    self.bundle.preprocessing_fingerprint
                ),
                "base_fingerprint": self.bundle.base_fingerprint,
                "d_v_base_index_fingerprint": (
                    self.bundle.base_index_fingerprint
                ),
                "d_v_base_index_sha256": self.bundle.base_index_sha256,
                "d_v_image_fingerprint": self.bundle.d_v_image_fingerprint,
                "d_v_gt_fingerprint": self.bundle.d_v_gt_fingerprint,
                "base_samples_fingerprint": (
                    self.base_samples_fingerprint
                ),
                "residual_samples_fingerprint": (
                    self.residual_samples_fingerprint
                ),
                "sample_adapter_fingerprint": (
                    self.d_v_samples.adapter_fingerprint
                ),
                "cache_contract_fingerprint": (
                    self.d_r_cache.contract.contract_fingerprint
                ),
                "formal_attempt_run_receipt_fingerprint": (
                    self.attempt.run_receipt_fingerprint
                ),
                "formal_attempt_complete_fingerprint": (
                    self.attempt.complete_fingerprint
                ),
                "decoder_artifact_fingerprint": (
                    self.attempt.artifact.artifact_fingerprint
                ),
                "decoder_state_fingerprint": (
                    self.attempt.artifact.decoder_state_fingerprint
                ),
            }
        )

    def verify_unchanged(self) -> None:
        self._seal()
        if not isinstance(self.bundle, LoadedDVCacheBundle):
            raise TypeError("bundle must be LoadedDVCacheBundle")
        if not isinstance(self.d_r_cache, PFCRRealCacheAdapter):
            raise TypeError("d_r_cache must be PFCRRealCacheAdapter")
        if not isinstance(self.comparison_protocol, FrozenComparisonProtocol):
            raise TypeError(
                "comparison_protocol must be FrozenComparisonProtocol"
            )
        if not isinstance(self.access, DevelopmentSplitAccess):
            raise TypeError("access must be DevelopmentSplitAccess")
        if not isinstance(self.d_v_samples, PFCRDVSamples):
            raise TypeError("d_v_samples must be PFCRDVSamples")
        self.bundle.verify_unchanged()
        self.d_r_cache.verify_unchanged()
        _verify_published_attempt(self.attempt)
        self.comparison_protocol.verify_bundle(self.bundle)
        contract = self.d_r_cache.contract
        artifact = self.attempt.artifact
        if (
            artifact.config.cache_contract_fingerprint
            != contract.contract_fingerprint
            or contract.dataset != self.comparison_protocol.dataset
            or contract.split_manifest_fingerprint
            != self.bundle.split_manifest_fingerprint
            or contract.preprocessing_fingerprint
            != self.bundle.preprocessing_fingerprint
            or contract.base_fingerprint != self.bundle.base_fingerprint
            or contract.base_state_fingerprint
            != self.bundle.base_state_fingerprint
            or contract.occupancy_threshold
            != self.comparison_protocol.occupancy_config.threshold
        ):
            raise RuntimeError(
                "PFCR D_R contract, attempt and frozen D_V protocol differ"
            )
        expected_ids = tuple(row.sample_id for row in self.bundle.rows)
        if (
            tuple(
                record.sample_id
                for record in self.access.records_for("D_V")
            )
            != expected_ids
            or self.d_v_samples.ordered_sample_ids != expected_ids
            or tuple(sample.sample_id for sample in self.base_samples)
            != expected_ids
        ):
            raise RuntimeError("PFCR formal D_V sample ordering changed")
        residual_fingerprint = calibration_samples_fingerprint(
            self.d_v_samples.samples
        )
        if (
            residual_fingerprint
            != self.d_v_samples.sample_tensor_fingerprint
            or stable_fingerprint(self.d_v_samples.canonical_payload())
            != self.d_v_samples.adapter_fingerprint
            or calibration_samples_fingerprint(self.base_samples)
            != self.base_samples_fingerprint
            or self.base_samples_fingerprint
            != self.comparison_protocol.base_samples_fingerprint
        ):
            raise RuntimeError(
                "PFCR formal D_V sample fingerprints changed"
            )


def build_loaded_pfcr_formal_d_v_run(
    bundle: LoadedDVCacheBundle,
    d_r_cache: PFCRRealCacheAdapter,
    attempt: PublishedPFCRRealFormalAttempt,
    *,
    comparison_protocol: FrozenComparisonProtocol,
    device: torch.device | str,
) -> LoadedPFCRFormalDVRun:
    """Build the only formal PFCR D_V run accepted by this module."""

    # Check the decoder source first so a bare artifact is rejected before any
    # cache or D_V operation can be attempted.
    if not isinstance(attempt, PublishedPFCRRealFormalAttempt):
        raise TypeError(
            "attempt must be PublishedPFCRRealFormalAttempt; "
            "bare PFCR artifacts are not eligible for formal D_V"
        )
    if not isinstance(bundle, LoadedDVCacheBundle):
        raise TypeError("bundle must be LoadedDVCacheBundle")
    if not isinstance(d_r_cache, PFCRRealCacheAdapter):
        raise TypeError("d_r_cache must be PFCRRealCacheAdapter")
    if not isinstance(comparison_protocol, FrozenComparisonProtocol):
        raise TypeError(
            "comparison_protocol must be FrozenComparisonProtocol"
        )
    execution_device = _validated_execution_device(device)
    _verify_published_attempt(attempt)
    bundle.verify_unchanged()
    d_r_cache.verify_unchanged()
    comparison_protocol.verify_bundle(bundle)
    contract = d_r_cache.contract
    if (
        attempt.artifact.config.cache_contract_fingerprint
        != contract.contract_fingerprint
        or contract.dataset != comparison_protocol.dataset
        or contract.split_manifest_fingerprint
        != bundle.split_manifest_fingerprint
        or contract.preprocessing_fingerprint
        != bundle.preprocessing_fingerprint
        or contract.base_fingerprint != bundle.base_fingerprint
        or contract.base_state_fingerprint
        != bundle.base_state_fingerprint
        or contract.occupancy_threshold
        != comparison_protocol.occupancy_config.threshold
    ):
        raise RuntimeError(
            "PFCR D_R contract, attempt and frozen D_V protocol differ"
        )
    manifest = load_and_validate_manifest(bundle.manifest_path)
    access = DevelopmentSplitAccess(manifest)
    expected_ids = tuple(row.sample_id for row in bundle.rows)
    if tuple(
        record.sample_id for record in access.records_for("D_V")
    ) != expected_ids:
        raise RuntimeError("PFCR D_V bundle differs from its manifest order")
    evaluation_decoder = _frozen_decoder_clone(
        attempt.artifact,
        device=execution_device,
    )
    try:
        d_v_samples = _build_pfcr_d_v_samples(
            bundle,
            contract,
            evaluation_decoder,
            comparison_protocol.occupancy_config,
            batch_size=PFCR_FORMAL_DV_BATCH_SIZE,
        )
    finally:
        del evaluation_decoder
    attempt.artifact.verify_unchanged()
    base_samples = _base_only_samples(d_v_samples.samples)
    base_fingerprint = calibration_samples_fingerprint(base_samples)
    if base_fingerprint != comparison_protocol.base_samples_fingerprint:
        raise RuntimeError(
            "PFCR base-only D_V tensors differ from frozen Wave-A evidence"
        )
    seal = _LoadedPFCRFormalDVRunSeal(
        bundle=bundle,
        d_r_cache=d_r_cache,
        attempt=attempt,
        comparison_protocol=comparison_protocol,
        access=access,
        d_v_samples=d_v_samples,
        base_samples=base_samples,
        base_samples_fingerprint=base_fingerprint,
    )
    return LoadedPFCRFormalDVRun(
        bundle=bundle,
        d_r_cache=d_r_cache,
        attempt=attempt,
        comparison_protocol=comparison_protocol,
        access=access,
        d_v_samples=d_v_samples,
        base_samples=base_samples,
        base_samples_fingerprint=base_fingerprint,
        _verification_token=seal,
    )


@dataclass(frozen=True, slots=True)
class _PFCRFormalResultSeal:
    core_fingerprint: str


@dataclass(frozen=True)
class PFCRFormalDVResult:
    """One selected PFCR/seed result sealed to its complete training attempt."""

    seed: int
    execution_device: str
    comparison_protocol_fingerprint: str
    selected_threshold: float | None
    metrics: AggregateEvaluation
    budget: FalseAlarmBudget
    pfcr_d_v_run_fingerprint: str
    threshold_protocol_fingerprint: str
    manifest_fingerprint: str
    manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    d_v_base_index_fingerprint: str
    d_v_base_index_sha256: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    base_samples_fingerprint: str
    residual_samples_fingerprint: str
    sample_adapter_fingerprint: str
    cache_contract_fingerprint: str
    formal_attempt_run_receipt_fingerprint: str
    formal_attempt_complete_fingerprint: str
    decoder_artifact_fingerprint: str
    decoder_receipt_sha256: str
    decoder_state_fingerprint: str
    formal_schedule_fingerprint: str
    state_catalog_fingerprint: str
    lineage_allowlist_fingerprint: str
    preflight_result_fingerprint: str
    _verification_token: object

    @property
    def method(self) -> str:
        return PFCR_FORMAL_METHOD

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": PFCR_FORMAL_DV_RESULT_SCHEMA,
            "runtime_split": "D_V",
            "D_T_accessed": False,
            "method": PFCR_FORMAL_METHOD,
            "seed": self.seed,
            "execution_device": self.execution_device,
            "comparison_protocol_fingerprint": (
                self.comparison_protocol_fingerprint
            ),
            "selected_threshold": self.selected_threshold,
            "aggregate_evaluation": _aggregate_payload(self.metrics),
            "false_alarm_budget": _budget_payload(self.budget),
            "bindings": {
                "pfcr_d_v_run_fingerprint": (
                    self.pfcr_d_v_run_fingerprint
                ),
                "threshold_protocol_fingerprint": (
                    self.threshold_protocol_fingerprint
                ),
                "manifest_fingerprint": self.manifest_fingerprint,
                "manifest_file_sha256": self.manifest_file_sha256,
                "preprocessing_fingerprint": (
                    self.preprocessing_fingerprint
                ),
                "base_fingerprint": self.base_fingerprint,
                "d_v_base_index_fingerprint": (
                    self.d_v_base_index_fingerprint
                ),
                "d_v_base_index_sha256": self.d_v_base_index_sha256,
                "d_v_image_fingerprint": self.d_v_image_fingerprint,
                "d_v_gt_fingerprint": self.d_v_gt_fingerprint,
                "base_samples_fingerprint": (
                    self.base_samples_fingerprint
                ),
                "residual_samples_fingerprint": (
                    self.residual_samples_fingerprint
                ),
                "sample_adapter_fingerprint": (
                    self.sample_adapter_fingerprint
                ),
                "cache_contract_fingerprint": (
                    self.cache_contract_fingerprint
                ),
                "formal_attempt_run_receipt_fingerprint": (
                    self.formal_attempt_run_receipt_fingerprint
                ),
                "formal_attempt_complete_fingerprint": (
                    self.formal_attempt_complete_fingerprint
                ),
                "decoder_artifact_fingerprint": (
                    self.decoder_artifact_fingerprint
                ),
                "decoder_receipt_sha256": self.decoder_receipt_sha256,
                "decoder_state_fingerprint": (
                    self.decoder_state_fingerprint
                ),
                "formal_schedule_fingerprint": (
                    self.formal_schedule_fingerprint
                ),
                "state_catalog_fingerprint": (
                    self.state_catalog_fingerprint
                ),
                "lineage_allowlist_fingerprint": (
                    self.lineage_allowlist_fingerprint
                ),
                "preflight_result_fingerprint": (
                    self.preflight_result_fingerprint
                ),
            },
        }

    def __post_init__(self) -> None:
        seal = self._verification_token
        if type(seal) is not _PFCRFormalResultSeal:
            raise TypeError(
                "PFCRFormalDVResult must come from strict evaluation or loader"
            )
        if self.seed not in FORMAL_SEEDS:
            raise ValueError(f"seed must be one of {FORMAL_SEEDS}")
        try:
            result_device = torch.device(self.execution_device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("execution_device is invalid") from error
        if (
            str(result_device) != self.execution_device
            or result_device.type not in {"cpu", "cuda"}
            or (
                result_device.type == "cpu"
                and result_device.index is not None
            )
            or (
                result_device.type == "cuda"
                and result_device.index is None
            )
        ):
            raise ValueError("execution_device is not canonical")
        _digest(
            self.comparison_protocol_fingerprint,
            name="comparison_protocol_fingerprint",
        )
        object.__setattr__(
            self,
            "selected_threshold",
            _threshold(self.selected_threshold, name="selected_threshold"),
        )
        _population_counts(self.metrics)
        if not isinstance(self.budget, FalseAlarmBudget):
            raise TypeError("budget must be FalseAlarmBudget")
        if self.metrics.budget_violation or not self.budget.accepts(
            self.metrics
        ):
            raise ValueError("selected PFCR metrics violate the frozen budget")
        for name in (
            "pfcr_d_v_run_fingerprint",
            "threshold_protocol_fingerprint",
            "manifest_fingerprint",
            "manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "d_v_base_index_fingerprint",
            "d_v_base_index_sha256",
            "d_v_image_fingerprint",
            "d_v_gt_fingerprint",
            "base_samples_fingerprint",
            "residual_samples_fingerprint",
            "sample_adapter_fingerprint",
            "cache_contract_fingerprint",
            "formal_attempt_run_receipt_fingerprint",
            "formal_attempt_complete_fingerprint",
            "decoder_artifact_fingerprint",
            "decoder_receipt_sha256",
            "decoder_state_fingerprint",
            "formal_schedule_fingerprint",
            "state_catalog_fingerprint",
            "lineage_allowlist_fingerprint",
            "preflight_result_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        if seal.core_fingerprint != stable_fingerprint(
            self._core_payload()
        ):
            raise TypeError("PFCR formal D_V result fields were replaced")

    def verify_unchanged(self) -> None:
        seal = self._verification_token
        if (
            type(seal) is not _PFCRFormalResultSeal
            or seal.core_fingerprint
            != stable_fingerprint(self._core_payload())
        ):
            raise RuntimeError("PFCR formal D_V result changed in memory")
        _population_counts(self.metrics)
        if self.metrics.budget_violation or not self.budget.accepts(
            self.metrics
        ):
            raise RuntimeError(
                "PFCR formal D_V result budget binding changed"
            )

    @property
    def result_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(self._core_payload())

    def to_formal_method_evidence(self) -> FormalMethodEvidence:
        self.verify_unchanged()
        total, true, total_misses, recovered = _population_counts(
            self.metrics
        )
        return FormalMethodEvidence(
            method=PFCR_FORMAL_METHOD,
            seed=self.seed,
            total_targets=total,
            true_targets=true,
            pd=self.metrics.pd,
            total_anchor_misses=total_misses,
            recovered_anchor_misses=recovered,
            retention=self.metrics.retention,
            pixel_fa=self.metrics.pixel_fa,
            raw_background_fa=self.metrics.raw_background_fa,
            fp_components_per_mp=self.metrics.fp_components_per_mp,
            budget_violation=self.metrics.budget_violation,
            comparison_protocol_fingerprint=(
                self.comparison_protocol_fingerprint
            ),
            result_fingerprint=self.result_fingerprint,
        )

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        core = self._core_payload()
        receipt_core = {
            **core,
            "formal_method_evidence": (
                self.to_formal_method_evidence().canonical_payload()
            ),
            "result_fingerprint": self.result_fingerprint,
        }
        return {
            **receipt_core,
            "receipt_fingerprint": stable_fingerprint(receipt_core),
        }

    @property
    def receipt_fingerprint(self) -> str:
        return str(self.canonical_payload()["receipt_fingerprint"])


def _new_result(**values: object) -> PFCRFormalDVResult:
    temporary = object.__new__(PFCRFormalDVResult)
    for field in fields(PFCRFormalDVResult):
        if field.name == "_verification_token":
            continue
        object.__setattr__(temporary, field.name, values[field.name])
    core_fingerprint = stable_fingerprint(temporary._core_payload())
    return PFCRFormalDVResult(
        **values,
        _verification_token=_PFCRFormalResultSeal(core_fingerprint),
    )  # type: ignore[arg-type]


def _verify_selected_protocol(
    run: LoadedPFCRFormalDVRun,
    protocol: BoundDVThresholdProtocol,
) -> None:
    if not isinstance(protocol, BoundDVThresholdProtocol):
        raise TypeError("threshold protocol must be BoundDVThresholdProtocol")
    comparison = run.comparison_protocol
    if (
        protocol.variant != "residual"
        or protocol.manifest_fingerprint
        != comparison.manifest_fingerprint
        or tuple(protocol.ordered_d_v_sample_ids)
        != run.d_v_samples.ordered_sample_ids
        or protocol.sample_tensor_fingerprint
        != run.residual_samples_fingerprint
        or protocol.candidate_threshold_grid
        != comparison.residual_thresholds
        or protocol.occupancy_config != comparison.occupancy_config
        or protocol.match_config != comparison.match_config
        or protocol.budget != comparison.budget
    ):
        raise RuntimeError(
            "PFCR selected threshold differs from frozen comparison protocol"
        )


def select_and_evaluate_pfcr_formal_method(
    bundle: LoadedDVCacheBundle,
    d_r_cache: PFCRRealCacheAdapter,
    attempt: PublishedPFCRRealFormalAttempt,
    *,
    comparison_protocol: FrozenComparisonProtocol,
    device: torch.device | str,
) -> PFCRFormalDVResult:
    """Select and replay one PFCR operating point under the frozen protocol."""

    if not isinstance(attempt, PublishedPFCRRealFormalAttempt):
        raise TypeError(
            "attempt must be PublishedPFCRRealFormalAttempt; "
            "bare PFCR artifacts are not eligible for formal D_V"
        )
    execution_device = _validated_execution_device(device)
    run = build_loaded_pfcr_formal_d_v_run(
        bundle,
        d_r_cache,
        attempt,
        comparison_protocol=comparison_protocol,
        device=execution_device,
    )
    run.verify_unchanged()
    threshold_protocol = select_residual_threshold_on_d_v(
        run.access,
        run.d_v_samples.samples,
        comparison_protocol.residual_thresholds,
        comparison_protocol.occupancy_config,
        comparison_protocol.match_config,
        comparison_protocol.budget,
    )
    _verify_selected_protocol(run, threshold_protocol)
    metrics = evaluate_frozen_residual_threshold(
        run.access,
        run.d_v_samples.samples,
        threshold_protocol,
    )
    if metrics != threshold_protocol.selected_metrics:
        raise RuntimeError(
            "replayed PFCR metrics differ from selected D_V metrics"
        )
    _population_counts(metrics)
    if metrics.budget_violation or not comparison_protocol.budget.accepts(
        metrics
    ):
        raise RuntimeError("selected PFCR D_V result violates its budget")
    artifact = attempt.artifact
    config = artifact.config
    result = _new_result(
        seed=attempt.seed,
        execution_device=str(execution_device),
        comparison_protocol_fingerprint=(
            comparison_protocol.comparison_protocol_fingerprint
        ),
        selected_threshold=threshold_protocol.selected_threshold,
        metrics=metrics,
        budget=comparison_protocol.budget,
        pfcr_d_v_run_fingerprint=run.run_fingerprint,
        threshold_protocol_fingerprint=(
            threshold_protocol.receipt_fingerprint
        ),
        manifest_fingerprint=bundle.split_manifest_fingerprint,
        manifest_file_sha256=bundle.split_manifest_file_sha256,
        preprocessing_fingerprint=bundle.preprocessing_fingerprint,
        base_fingerprint=bundle.base_fingerprint,
        d_v_base_index_fingerprint=bundle.base_index_fingerprint,
        d_v_base_index_sha256=bundle.base_index_sha256,
        d_v_image_fingerprint=bundle.d_v_image_fingerprint,
        d_v_gt_fingerprint=bundle.d_v_gt_fingerprint,
        base_samples_fingerprint=run.base_samples_fingerprint,
        residual_samples_fingerprint=(
            run.residual_samples_fingerprint
        ),
        sample_adapter_fingerprint=(
            run.d_v_samples.adapter_fingerprint
        ),
        cache_contract_fingerprint=config.cache_contract_fingerprint,
        formal_attempt_run_receipt_fingerprint=(
            attempt.run_receipt_fingerprint
        ),
        formal_attempt_complete_fingerprint=(
            attempt.complete_fingerprint
        ),
        decoder_artifact_fingerprint=artifact.artifact_fingerprint,
        decoder_receipt_sha256=artifact.receipt_sha256,
        decoder_state_fingerprint=artifact.decoder_state_fingerprint,
        formal_schedule_fingerprint=config.formal_schedule_fingerprint,
        state_catalog_fingerprint=config.state_catalog_fingerprint,
        lineage_allowlist_fingerprint=(
            config.lineage_allowlist_fingerprint
        ),
        preflight_result_fingerprint=config.preflight_result_fingerprint,
    )
    run.verify_unchanged()
    _verify_selected_protocol(run, threshold_protocol)
    return result


def save_pfcr_formal_d_v_result(
    path: str | Path,
    result: PFCRFormalDVResult,
) -> str:
    """Create one canonical result receipt and refuse every overwrite."""

    if not isinstance(result, PFCRFormalDVResult):
        raise TypeError("result must be PFCRFormalDVResult")
    result.verify_unchanged()
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("PFCR formal D_V result target may not be a symlink")
    target = requested.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            result.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with target.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite PFCR formal D_V result {target}"
        ) from None
    return result.receipt_fingerprint


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("PFCR formal D_V result may not be a symlink")
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("PFCR formal D_V result must be a regular file")

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    "PFCR formal D_V result contains duplicate JSON keys"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(
            f"PFCR formal D_V result contains non-finite number {value}"
        )

    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_nonfinite,
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "PFCR formal D_V result is not valid UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("PFCR formal D_V result must contain one object")
    return value


def load_pfcr_formal_d_v_result(
    path: str | Path,
) -> PFCRFormalDVResult:
    """Strictly load and fingerprint-check one persisted PFCR result."""

    payload = _strict_json(Path(path).expanduser())
    expected_top = {
        "schema_version",
        "runtime_split",
        "D_T_accessed",
        "method",
        "seed",
        "execution_device",
        "comparison_protocol_fingerprint",
        "selected_threshold",
        "aggregate_evaluation",
        "false_alarm_budget",
        "bindings",
        "formal_method_evidence",
        "result_fingerprint",
        "receipt_fingerprint",
    }
    if set(payload) != expected_top:
        raise ValueError("PFCR formal D_V result fields are not canonical")
    if (
        payload["schema_version"] != PFCR_FORMAL_DV_RESULT_SCHEMA
        or payload["runtime_split"] != "D_V"
        or payload["D_T_accessed"] is not False
        or payload["method"] != PFCR_FORMAL_METHOD
    ):
        raise ValueError("PFCR formal D_V result identity changed")
    expected_bindings = {
        "pfcr_d_v_run_fingerprint",
        "threshold_protocol_fingerprint",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "d_v_base_index_fingerprint",
        "d_v_base_index_sha256",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
        "base_samples_fingerprint",
        "residual_samples_fingerprint",
        "sample_adapter_fingerprint",
        "cache_contract_fingerprint",
        "formal_attempt_run_receipt_fingerprint",
        "formal_attempt_complete_fingerprint",
        "decoder_artifact_fingerprint",
        "decoder_receipt_sha256",
        "decoder_state_fingerprint",
        "formal_schedule_fingerprint",
        "state_catalog_fingerprint",
        "lineage_allowlist_fingerprint",
        "preflight_result_fingerprint",
    }
    bindings = payload["bindings"]
    if not isinstance(bindings, Mapping) or set(bindings) != expected_bindings:
        raise ValueError(
            "PFCR formal D_V result bindings are not canonical"
        )
    result = _new_result(
        seed=payload["seed"],
        execution_device=payload["execution_device"],
        comparison_protocol_fingerprint=(
            payload["comparison_protocol_fingerprint"]
        ),
        selected_threshold=payload["selected_threshold"],
        metrics=_aggregate_from_payload(payload["aggregate_evaluation"]),
        budget=_budget_from_payload(payload["false_alarm_budget"]),
        **dict(bindings),
    )
    if payload != result.canonical_payload():
        raise ValueError(
            "PFCR formal D_V result fingerprint or evidence binding mismatch"
        )
    return result


__all__ = [
    "PFCR_FORMAL_DV_BATCH_SIZE",
    "PFCR_FORMAL_DV_RESULT_SCHEMA",
    "PFCR_FORMAL_DV_RUN_SCHEMA",
    "PFCR_FORMAL_METHOD",
    "LoadedPFCRFormalDVRun",
    "PFCRFormalDVResult",
    "build_loaded_pfcr_formal_d_v_run",
    "load_pfcr_formal_d_v_result",
    "save_pfcr_formal_d_v_result",
    "select_and_evaluate_pfcr_formal_method",
]
