"""Create-only formal runner for the CURE-Lite v0.2 M extension.

The runner extends one completed v0.1 Stage-A reference without retraining or
recalibrating F/Fx/U.  It reuses the exact historical D_R/D_V caches, trains
only ``miss_aligned_legal`` (M), selects only M on the already frozen residual
threshold grid, and evaluates U once at its historical selected operating
point to recover the diagnostic counts omitted from the v0.1 result receipt.

There is deliberately no D_T input and no resume path.  An interrupted output
directory remains marked incomplete and a later invocation must use a new
destination.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..config import MissAlignmentConfig, config_to_dict
from ..data import ManifestImageDataset
from ..metrics import AggregateEvaluation, formal_stage_a_metrics_payload
from .artifacts import LoadedDecoderArtifact, load_decoder_artifact
from .cache_pipeline import (
    LoadedDRCacheBundle,
    LoadedDVCacheBundle,
    load_d_r_cache_bundle,
    load_d_v_cache_bundle,
)
from .formal_evaluation import (
    FormalDVThresholdReceipt,
    LoadedDVMethodRun,
    build_loaded_d_v_method_run,
    evaluate_formal_residual_fixed_point,
    select_formal_residual_threshold_from_ledger,
)
from .formal_training import (
    MissAlignedGate2TrainingConfig,
    prepare_gate2_training,
    run_miss_aligned_gate2_extension,
    save_completed_decoder_run,
    summarize_gate2_training_support,
)
from .stage_a_m_extension import (
    StageAReferenceSnapshot,
    load_stage_a_reference_snapshot,
)
from .stage_a_runner import (
    _budget_payload,
    _calibration_worker_count,
    _check_dataset_pair,
    _preflight_stage_a_device,
    _protocol_payload,
    _require_same_base_cache_identity,
    _source_tree_digest,
    _strict_json,
    _training_payload,
    _tree_inventory,
    _write_new_json,
)
from .training_pipeline import FixedEpochTrainingLog


STAGE_A_M_EXTENSION_SCHEMA = "cure-lite-stage-a-m-extension-run-v1"
STAGE_A_M_CONFIG_SCHEMA = "cure-lite-stage-a-m-extension-config-v1"
STAGE_A_M_REFERENCE_SCHEMA = "cure-lite-stage-a-m-reference-v1"
STAGE_A_M_ALIGNMENT_SCHEMA = "cure-lite-stage-a-m-alignment-v1"
STAGE_A_M_CALIBRATION_SCHEMA = "cure-lite-stage-a-m-calibration-v1"
STAGE_A_M_RESULTS_SCHEMA = "cure-lite-stage-a-m-results-v1"

STAGE_A_M_METHOD_ORDER = ("A", "Base@B", "F", "F×", "U", "M")
_REFERENCE_METHOD_ORDER = STAGE_A_M_METHOD_ORDER[:-1]
_COMPLETE_NAME = "COMPLETE.json"
_INCOMPLETE_NAME = ".incomplete"
_FORMAL_METRIC_FIELDS = (
    "pd",
    "miou",
    "niou",
    "pixel_fa",
    "fp_components_per_mp",
    "raw_background_fa",
    "retention",
    "budget_violation",
)
_RECOVERY_FIELDS = (
    "rmr",
    "gross_rmr",
    "net_rmr",
    "reachable_rmr",
    "oracle_upper_bound",
    "overlap_supported_rmr",
    "recovered_anchor_misses",
    "net_recovered_anchor_misses",
    "total_anchor_misses",
    "retained_anchor_covered",
    "total_anchor_covered",
    "recovered_reachable_anchor_misses",
    "total_reachable_anchor_misses",
)

TrainingProgressCallback = Callable[[FixedEpochTrainingLog], None]


def _extension_tree_inventory(
    root: Path,
) -> tuple[list[str], dict[str, str]]:
    """Inventory an extension while omitting its transient marker."""

    directories, files = _tree_inventory(root)
    files.pop(_INCOMPLETE_NAME, None)
    return directories, files


def _exact_mapping(
    value: object,
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{name} fields are not canonical")
    return dict(value)


def _canonical_threshold(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number or null")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return result


def _formal_metrics_mapping(value: object, *, name: str) -> dict[str, Any]:
    payload = _exact_mapping(value, set(_FORMAL_METRIC_FIELDS), name=name)
    for field in _FORMAL_METRIC_FIELDS[:-1]:
        item = payload[field]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name}.{field} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}.{field} must be finite")
    if not isinstance(payload["budget_violation"], bool):
        raise TypeError(f"{name}.budget_violation must be bool")
    return payload


@dataclass(frozen=True, slots=True)
class _HistoricalReceipts:
    complete: dict[str, Any]
    calibration: dict[str, Any]
    results: dict[str, Any]
    complete_sha256: str
    calibration_sha256: str
    results_sha256: str
    uniform_threshold: float | None
    uniform_metrics: dict[str, Any]


def _load_historical_receipts(
    snapshot: StageAReferenceSnapshot,
) -> _HistoricalReceipts:
    """Read only fields already validated by the sealed historical loader."""

    complete_path = snapshot.root / _COMPLETE_NAME
    calibration_path = snapshot.root / "receipts" / "calibration.json"
    results_path = snapshot.root / "receipts" / "results.json"
    complete = _strict_json(complete_path, name="historical COMPLETE receipt")
    calibration = _strict_json(
        calibration_path,
        name="historical calibration receipt",
    )
    results = _strict_json(results_path, name="historical results receipt")

    if complete.get("complete_fingerprint") != snapshot.complete_fingerprint:
        raise RuntimeError("historical COMPLETE differs from the sealed snapshot")
    if complete.get("method_order") != list(_REFERENCE_METHOD_ORDER):
        raise ValueError("historical method order is not the v0.1 reference order")
    if calibration.get("method_order") != list(_REFERENCE_METHOD_ORDER):
        raise ValueError("historical calibration method order changed")
    if results.get("method_order") != list(_REFERENCE_METHOD_ORDER):
        raise ValueError("historical result method order changed")
    if (
        complete.get("calibration_receipt_fingerprint")
        != calibration.get("receipt_fingerprint")
        or complete.get("results_fingerprint")
        != results.get("results_fingerprint")
        or results.get("calibration_receipt_fingerprint")
        != calibration.get("receipt_fingerprint")
    ):
        raise RuntimeError("historical completion/calibration/results bindings differ")

    methods = results.get("methods")
    calibration_methods = calibration.get("methods")
    if (
        not isinstance(methods, Mapping)
        or tuple(methods) != _REFERENCE_METHOD_ORDER
        or not isinstance(calibration_methods, Mapping)
        or tuple(calibration_methods) != _REFERENCE_METHOD_ORDER
    ):
        raise ValueError("historical method mappings are not canonical")
    for method in _REFERENCE_METHOD_ORDER:
        _formal_metrics_mapping(
            methods[method],
            name=f"historical results {method}",
        )

    uniform = calibration_methods["U"]
    if not isinstance(uniform, Mapping) or not isinstance(
        uniform.get("protocol"),
        Mapping,
    ):
        raise ValueError("historical U calibration receipt is invalid")
    uniform_protocol = uniform["protocol"]
    uniform_threshold = _canonical_threshold(
        uniform_protocol.get("selected_threshold"),
        name="historical U threshold",
    )
    uniform_metrics = _formal_metrics_mapping(
        uniform_protocol.get("selected_metrics"),
        name="historical U selected metrics",
    )
    if uniform_metrics != dict(methods["U"]):
        raise RuntimeError("historical U calibration and result metrics differ")

    return _HistoricalReceipts(
        complete=complete,
        calibration=calibration,
        results=results,
        complete_sha256=file_sha256(complete_path),
        calibration_sha256=file_sha256(calibration_path),
        results_sha256=file_sha256(results_path),
        uniform_threshold=uniform_threshold,
        uniform_metrics=uniform_metrics,
    )


def _m_training_config(
    snapshot: StageAReferenceSnapshot,
) -> MissAlignedGate2TrainingConfig:
    reference = snapshot.config.training
    return MissAlignedGate2TrainingConfig(
        decoder_config=reference.decoder_config,
        loss_config=reference.loss_config,
        training_config=reference.training_config,
        optimizer=reference.optimizer,
        learning_rate=reference.learning_rate,
        weight_decay=reference.weight_decay,
        epochs=reference.epochs,
        steps_per_epoch=reference.steps_per_epoch,
        factual_miss_batch=reference.factual_miss_batch,
        factual_no_miss_batch=reference.factual_no_miss_batch,
        synthetic_batch=reference.synthetic_batch,
        global_seed=reference.global_seed,
        miss_alignment_config=MissAlignmentConfig(),
    )


def _load_reference_cache_bundles(
    snapshot: StageAReferenceSnapshot,
    d_r_dataset: ManifestImageDataset,
    d_v_dataset: ManifestImageDataset,
) -> tuple[LoadedDRCacheBundle, LoadedDVCacheBundle]:
    _check_dataset_pair(d_r_dataset, d_v_dataset)
    if (
        d_r_dataset.manifest.fingerprint
        != snapshot.factual_artifact.config.manifest_fingerprint
    ):
        raise ValueError("datasets differ from the historical Stage-A manifest")
    expected_base = snapshot.factual_artifact.config.base_fingerprint
    d_r_bundle = load_d_r_cache_bundle(
        snapshot.root / "d_r" / "state_cache" / "index.json",
        d_r_dataset,
        expected_base_fingerprint=expected_base,
    )
    d_v_bundle = load_d_v_cache_bundle(
        snapshot.root / "d_v" / "base_cache" / "index.json",
        d_v_dataset,
        expected_base_fingerprint=expected_base,
    )
    _require_same_base_cache_identity(d_r_bundle, d_v_bundle)
    return d_r_bundle, d_v_bundle


def _verify_uniform_d_v_binding(
    run: LoadedDVMethodRun,
    historical: _HistoricalReceipts,
) -> None:
    methods = historical.calibration["methods"]
    assert isinstance(methods, Mapping)
    value = methods["U"]
    assert isinstance(value, Mapping)
    protocol = value["protocol"]
    assert isinstance(protocol, Mapping)
    expected = {
        "d_v_run_fingerprint": run.run_fingerprint,
        "decoder_artifact_fingerprint": run.artifact.artifact_fingerprint,
        "decoder_receipt_sha256": run.artifact.receipt_sha256,
        "decoder_state_fingerprint": run.artifact.decoder_state_fingerprint,
        "decoder_variant": run.artifact.config.variant,
        "global_seed": run.artifact.config.global_seed,
    }
    for name, actual in expected.items():
        if value.get(name) != actual:
            raise RuntimeError(f"historical U D_V binding differs at {name}")
    if (
        protocol.get("sample_tensor_fingerprint")
        != run.residual_samples_fingerprint
        or protocol.get("ordered_d_v_sample_ids")
        != [sample.sample_id for sample in run.residual_samples]
    ):
        raise RuntimeError("historical U D_V sample binding differs")


def _recovery_payload(metrics: AggregateEvaluation) -> dict[str, object]:
    if not isinstance(metrics, AggregateEvaluation):
        raise TypeError("recovery diagnostics require AggregateEvaluation")
    return {field: getattr(metrics, field) for field in _RECOVERY_FIELDS}


def _development_gate_payload(
    historical_methods: Mapping[str, object],
    m_metrics: AggregateEvaluation,
    u_metrics: AggregateEvaluation,
    *,
    budget: object,
    reference_snapshot_fingerprint: str,
    results_fingerprint: str,
) -> dict[str, object]:
    old = {
        method: _formal_metrics_mapping(
            historical_methods[method],
            name=f"historical {method}",
        )
        for method in _REFERENCE_METHOD_ORDER
    }
    m_formal = formal_stage_a_metrics_payload(m_metrics)
    if not hasattr(budget, "accepts"):
        raise TypeError("budget must expose the frozen acceptance rule")
    all_within_constraints = not any(
        bool(old[method]["budget_violation"])
        for method in _REFERENCE_METHOD_ORDER
    ) and bool(budget.accepts(m_metrics))
    comparators = ("Base@B", "F", "F×", "U")
    deltas = {
        method: float(m_formal["pd"]) - float(old[method]["pd"])
        for method in comparators
    }
    strict_pd_rule = all(delta > 0.0 for delta in deltas.values())
    recovery_delta = (
        m_metrics.recovered_anchor_misses
        - u_metrics.recovered_anchor_misses
    )
    recovery_rule = recovery_delta > 0
    denominator_equality = {
        "total_anchor_misses": (
            m_metrics.total_anchor_misses == u_metrics.total_anchor_misses
        ),
        "total_reachable_anchor_misses": (
            m_metrics.total_reachable_anchor_misses
            == u_metrics.total_reachable_anchor_misses
        ),
        "total_anchor_covered": (
            m_metrics.total_anchor_covered == u_metrics.total_anchor_covered
        ),
    }
    denominators_match = all(denominator_equality.values())
    signal = (
        all_within_constraints
        and strict_pd_rule
        and recovery_rule
        and denominators_match
    )
    return {
        "schema_version": "cure-lite-stage-a-m-development-gate-v1",
        "reference_snapshot_fingerprint": reference_snapshot_fingerprint,
        "results_fingerprint": results_fingerprint,
        "primary_metric": "total_pd",
        "strict_pd_comparators": list(comparators),
        "m_minus_comparator_pd": deltas,
        "strict_pd_rule_met": strict_pd_rule,
        "recovery_comparator": "U_at_historical_selected_threshold",
        "m_minus_u_recovered_anchor_misses": recovery_delta,
        "recovery_rule_met": recovery_rule,
        "recovery_denominator_equality": denominator_equality,
        "recovery_denominators_match": denominators_match,
        "all_methods_within_constraints": all_within_constraints,
        "secondary_iou_non_degradation": (
            m_metrics.miou
            >= max(float(old[method]["miou"]) for method in comparators)
            and m_metrics.niou
            >= max(float(old[method]["niou"]) for method in comparators)
        ),
        "mechanism_signal": signal,
        "interpretation": (
            "supported_single_seed_development_signal"
            if signal
            else "not_supported_single_seed_development_signal"
        ),
        "independent_generalization_claim": False,
    }


def _receipt_with_fingerprint(
    schema_version: str,
    payload: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    core = {"schema_version": schema_version, **dict(payload)}
    return {
        **core,
        fingerprint_field: stable_fingerprint(core),
    }


@dataclass(frozen=True, slots=True)
class _PublishedSeal:
    root: Path
    reference: StageAReferenceSnapshot
    m_artifact: LoadedDecoderArtifact
    m_calibration: FormalDVThresholdReceipt
    m_metrics: AggregateEvaluation
    u_fixed_metrics: AggregateEvaluation
    alignment_catalog_fingerprint: str
    results_fingerprint: str
    complete_fingerprint: str
    mechanism_signal: bool


@dataclass(frozen=True, slots=True)
class PublishedStageAMExtension:
    """One newly published M result bound to an immutable v0.1 reference."""

    root: Path
    reference: StageAReferenceSnapshot
    m_artifact: LoadedDecoderArtifact
    m_calibration: FormalDVThresholdReceipt
    m_metrics: AggregateEvaluation
    u_fixed_metrics: AggregateEvaluation
    alignment_catalog_fingerprint: str
    results_fingerprint: str
    complete_fingerprint: str
    mechanism_signal: bool
    _verification_token: object

    def _seal(self) -> _PublishedSeal:
        seal = self._verification_token
        if type(seal) is not _PublishedSeal:
            raise TypeError(
                "PublishedStageAMExtension must come from the formal M runner"
            )
        if (
            self.root != seal.root
            or self.reference is not seal.reference
            or self.m_artifact is not seal.m_artifact
            or self.m_calibration is not seal.m_calibration
            or self.m_metrics is not seal.m_metrics
            or self.u_fixed_metrics is not seal.u_fixed_metrics
            or self.alignment_catalog_fingerprint
            != seal.alignment_catalog_fingerprint
            or self.results_fingerprint != seal.results_fingerprint
            or self.complete_fingerprint != seal.complete_fingerprint
            or self.mechanism_signal != seal.mechanism_signal
        ):
            raise TypeError("published M extension fields were replaced")
        return seal

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            raise ValueError("published M root must be absolute")
        self._seal()

    def verify_unchanged(self) -> None:
        """Recheck the historical snapshot and every published extension byte."""

        self._seal()
        self.reference.verify_unchanged()
        self.m_artifact.verify_unchanged()
        if (self.root / _INCOMPLETE_NAME).exists():
            raise RuntimeError("published M extension became incomplete")
        complete = _strict_json(
            self.root / _COMPLETE_NAME,
            name="M extension COMPLETE receipt",
        )
        complete_core = dict(complete)
        actual_fingerprint = complete_core.pop("complete_fingerprint", None)
        if (
            stable_fingerprint(complete_core) != actual_fingerprint
            or actual_fingerprint != self.complete_fingerprint
        ):
            raise RuntimeError("M extension COMPLETE fingerprint changed")
        directories, files = _tree_inventory(self.root)
        if (
            complete.get("artifact_directories") != directories
            or complete.get("artifact_files") != files
        ):
            raise RuntimeError("M extension artifact inventory changed")
        if complete.get("results_fingerprint") != self.results_fingerprint:
            raise RuntimeError("M extension results binding changed")

    def verify(self) -> None:
        self.verify_unchanged()


def _bind_published(
    *,
    root: Path,
    reference: StageAReferenceSnapshot,
    m_artifact: LoadedDecoderArtifact,
    m_calibration: FormalDVThresholdReceipt,
    m_metrics: AggregateEvaluation,
    u_fixed_metrics: AggregateEvaluation,
    alignment_catalog_fingerprint: str,
    results_fingerprint: str,
    complete_fingerprint: str,
    mechanism_signal: bool,
) -> PublishedStageAMExtension:
    seal = _PublishedSeal(
        root=root,
        reference=reference,
        m_artifact=m_artifact,
        m_calibration=m_calibration,
        m_metrics=m_metrics,
        u_fixed_metrics=u_fixed_metrics,
        alignment_catalog_fingerprint=alignment_catalog_fingerprint,
        results_fingerprint=results_fingerprint,
        complete_fingerprint=complete_fingerprint,
        mechanism_signal=mechanism_signal,
    )
    return PublishedStageAMExtension(
        root=root,
        reference=reference,
        m_artifact=m_artifact,
        m_calibration=m_calibration,
        m_metrics=m_metrics,
        u_fixed_metrics=u_fixed_metrics,
        alignment_catalog_fingerprint=alignment_catalog_fingerprint,
        results_fingerprint=results_fingerprint,
        complete_fingerprint=complete_fingerprint,
        mechanism_signal=mechanism_signal,
        _verification_token=seal,
    )


def run_stage_a_m_extension(
    reference_stage_a_dir: str | Path,
    d_r_dataset: ManifestImageDataset,
    d_v_dataset: ManifestImageDataset,
    output_dir: str | Path,
    *,
    device: torch.device | str,
    calibration_workers: int = 1,
    calibration_progress: Callable[[int, int], None] | None = None,
    training_progress: TrainingProgressCallback | None = None,
) -> PublishedStageAMExtension:
    """Train, calibrate, evaluate, and publish only M.

    All scientific choices except the fixed miss-alignment rule are inherited
    from the completed v0.1 reference.  The destination must not exist.
    """

    requested = Path(output_dir).expanduser()
    if requested.is_symlink() or requested.exists():
        raise FileExistsError(f"refusing to overwrite M extension {requested}")
    reference_root = Path(reference_stage_a_dir).expanduser()
    if reference_root.is_symlink():
        raise ValueError("historical Stage-A root may not be a symlink")
    resolved_reference_root = reference_root.resolve(strict=True)
    resolved_output = requested.resolve(strict=False)
    if (
        resolved_output == resolved_reference_root
        or resolved_reference_root in resolved_output.parents
    ):
        raise ValueError(
            "M extension output must be outside the historical Stage-A tree"
        )
    if training_progress is not None and not callable(training_progress):
        raise TypeError("training_progress must be callable or None")
    calibration_workers = _calibration_worker_count(calibration_workers)
    resolved_device = str(torch.device(device))
    _preflight_stage_a_device(resolved_device)

    snapshot = load_stage_a_reference_snapshot(reference_stage_a_dir)
    historical = _load_historical_receipts(snapshot)
    d_r_bundle, d_v_bundle = _load_reference_cache_bundles(
        snapshot,
        d_r_dataset,
        d_v_dataset,
    )
    prepared = prepare_gate2_training(d_r_bundle)
    support = summarize_gate2_training_support(
        d_r_bundle,
        prepared=prepared,
    )
    if support != snapshot.support_summary:
        raise RuntimeError("current preparation differs from historical D_R support")
    snapshot.config.support_requirements.require(support)
    m_training = _m_training_config(snapshot)
    source_digest = _source_tree_digest()

    root = resolved_output
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    incomplete = root / _INCOMPLETE_NAME
    incomplete.open("xb").close()
    receipts = root / "receipts"

    config_receipt = _receipt_with_fingerprint(
        STAGE_A_M_CONFIG_SCHEMA,
        {
            "method": "CURE-Lite",
            "stage": "Stage-A-M",
            "runtime_splits": ["D_R", "D_V"],
            "unused_split": "D_T",
            "method_order": list(STAGE_A_M_METHOD_ORDER),
            "source_tree_digest": source_digest,
            "reference_run_config": snapshot.config.canonical_payload(),
            "m_training": {
                **_training_payload(m_training),
                "miss_alignment_config": config_to_dict(
                    m_training.miss_alignment_config
                ),
            },
            "m_residual_thresholds": list(snapshot.config.residual_thresholds),
            "budget": _budget_payload(snapshot.config.budget),
            "device": resolved_device,
            "execution": {
                "calibration_workers": calibration_workers,
            },
            "mechanism_contract": {
                "changed_choice": "synthetic_target_selection_only",
                "selection_rule": (
                    "nearest_decoder_visible_legal_target_by_quantized_"
                    "log1p_positive_region_feature_rms"
                ),
                "unchanged": [
                    "frozen_base",
                    "legal_target_definition",
                    "decoder_architecture",
                    "supervision",
                    "loss",
                    "branch_weights",
                    "optimizer",
                    "training_horizon",
                    "inference_rule",
                ],
            },
        },
        fingerprint_field="config_fingerprint",
    )
    _write_new_json(receipts / "config.json", config_receipt)
    reference_receipt = _receipt_with_fingerprint(
        STAGE_A_M_REFERENCE_SCHEMA,
        {
            "snapshot_fingerprint": snapshot.snapshot_fingerprint,
            "artifact_inventory_fingerprint": (
                snapshot.artifact_inventory_fingerprint
            ),
            "historical_source_tree_digest": snapshot.source_tree_digest,
            "historical_complete_fingerprint": snapshot.complete_fingerprint,
            "historical_complete_sha256": historical.complete_sha256,
            "historical_calibration_receipt_fingerprint": (
                historical.calibration["receipt_fingerprint"]
            ),
            "historical_calibration_sha256": historical.calibration_sha256,
            "historical_results_fingerprint": (
                historical.results["results_fingerprint"]
            ),
            "historical_results_sha256": historical.results_sha256,
            "historical_common_training_fingerprint": (
                snapshot.common_training_fingerprint
            ),
            "decoder_artifact_fingerprints": {
                "F": snapshot.factual_artifact.artifact_fingerprint,
                "F×": (
                    snapshot.factual_exposure_matched_artifact.artifact_fingerprint
                ),
                "U": snapshot.uniform_artifact.artifact_fingerprint,
            },
        },
        fingerprint_field="reference_fingerprint",
    )
    _write_new_json(receipts / "reference.json", reference_receipt)

    alignment_receipt = _receipt_with_fingerprint(
        STAGE_A_M_ALIGNMENT_SCHEMA,
        {
            "config": config_to_dict(m_training.miss_alignment_config),
            "catalog_fingerprint": prepared.catalog.miss_alignment_fingerprint,
            "summary": prepared.catalog.miss_alignment_summary,
            "choices": [
                choice.canonical_payload()
                for choice in prepared.catalog.miss_aligned_choices
            ],
        },
        fingerprint_field="alignment_receipt_fingerprint",
    )
    _write_new_json(receipts / "alignment.json", alignment_receipt)

    extension = run_miss_aligned_gate2_extension(
        d_r_bundle,
        m_training,
        factual_only_reference=snapshot.factual_artifact,
        factual_exposure_matched_reference=(
            snapshot.factual_exposure_matched_artifact
        ),
        uniform_legal_reference=snapshot.uniform_artifact,
        device=resolved_device,
        prepared=prepared,
        training_progress=training_progress,
    )
    m_directory = root / "decoders" / "miss_aligned_legal"
    save_completed_decoder_run(
        m_directory,
        extension.miss_aligned_legal,
    )
    m_artifact = load_decoder_artifact(
        m_directory,
        expected_config=extension.miss_aligned_legal.config,
    )
    if (
        m_artifact.config.alignment_catalog_fingerprint
        != prepared.catalog.miss_alignment_fingerprint
    ):
        raise RuntimeError("saved M artifact differs from its alignment catalog")

    m_run = build_loaded_d_v_method_run(d_v_bundle, m_artifact)
    m_calibration = select_formal_residual_threshold_from_ledger(
        m_run,
        snapshot.config.residual_thresholds,
        snapshot.config.budget,
        method_label="M",
        max_workers=calibration_workers,
        progress=calibration_progress,
    )
    # The ledger receipt already contains the exact selected AggregateEvaluation.
    # Re-evaluating the frozen grid here would duplicate calibration.
    m_metrics = m_calibration.protocol.selected_metrics

    u_run = build_loaded_d_v_method_run(
        d_v_bundle,
        snapshot.uniform_artifact,
    )
    _verify_uniform_d_v_binding(u_run, historical)
    if m_run.base_samples_fingerprint != u_run.base_samples_fingerprint:
        raise RuntimeError("M and historical U use different D_V base samples")
    u_fixed_metrics = evaluate_formal_residual_fixed_point(
        u_run,
        historical.uniform_threshold,
        method_label="U-historical-fixed",
    )
    if (
        formal_stage_a_metrics_payload(u_fixed_metrics)
        != historical.uniform_metrics
    ):
        raise RuntimeError("fixed-point U evaluation differs from historical results")

    m_calibration_payload = _protocol_payload(m_calibration)
    calibration_receipt = _receipt_with_fingerprint(
        STAGE_A_M_CALIBRATION_SCHEMA,
        {
            "method": "M",
            "methods": {"M": m_calibration_payload},
            "reference_snapshot_fingerprint": snapshot.snapshot_fingerprint,
            "reference_calibration_receipt_fingerprint": (
                historical.calibration["receipt_fingerprint"]
            ),
            "historical_u_selected_threshold": historical.uniform_threshold,
        },
        fingerprint_field="calibration_receipt_fingerprint",
    )
    _write_new_json(receipts / "calibration.json", calibration_receipt)

    historical_methods = historical.results["methods"]
    assert isinstance(historical_methods, Mapping)
    result_methods = {
        method: dict(historical_methods[method]) for method in _REFERENCE_METHOD_ORDER
    }
    result_methods["M"] = formal_stage_a_metrics_payload(m_metrics)
    results_receipt = _receipt_with_fingerprint(
        STAGE_A_M_RESULTS_SCHEMA,
        {
            "method_order": list(STAGE_A_M_METHOD_ORDER),
            "methods": result_methods,
            "reference_results_fingerprint": (
                historical.results["results_fingerprint"]
            ),
            "m_calibration_receipt_fingerprint": (
                calibration_receipt["calibration_receipt_fingerprint"]
            ),
            "recovery_diagnostics": {
                "U@historical": _recovery_payload(u_fixed_metrics),
                "M": _recovery_payload(m_metrics),
            },
        },
        fingerprint_field="results_fingerprint",
    )
    _write_new_json(receipts / "results.json", results_receipt)
    gate = _development_gate_payload(
        historical_methods,
        m_metrics,
        u_fixed_metrics,
        budget=snapshot.config.budget,
        reference_snapshot_fingerprint=snapshot.snapshot_fingerprint,
        results_fingerprint=results_receipt["results_fingerprint"],
    )
    gate_fingerprint = stable_fingerprint(gate)
    gate_receipt = {**gate, "gate_fingerprint": gate_fingerprint}
    _write_new_json(receipts / "gate.json", gate_receipt)

    snapshot.verify_unchanged()
    d_r_bundle.verify_unchanged()
    d_v_bundle.verify_unchanged()
    m_artifact.verify_unchanged()
    if _source_tree_digest() != source_digest:
        raise RuntimeError("CURE-Lite sources changed during the M extension")

    # Keep the marker until COMPLETE has been atomically created and checked.
    # The extension inventory deliberately excludes this transient file.
    directories, files = _extension_tree_inventory(root)
    complete_core: dict[str, object] = {
        "schema_version": STAGE_A_M_EXTENSION_SCHEMA,
        "status": "complete",
        "method": "CURE-Lite",
        "stage": "Stage-A-M",
        "method_order": list(STAGE_A_M_METHOD_ORDER),
        "runtime_splits": ["D_R", "D_V"],
        "unused_split": "D_T",
        "source_tree_digest": source_digest,
        "reference_snapshot_fingerprint": snapshot.snapshot_fingerprint,
        "reference_complete_fingerprint": snapshot.complete_fingerprint,
        "reference_complete_sha256": historical.complete_sha256,
        "reference_source_tree_digest": snapshot.source_tree_digest,
        "reference_results_fingerprint": historical.results["results_fingerprint"],
        "reference_calibration_receipt_fingerprint": (
            historical.calibration["receipt_fingerprint"]
        ),
        "config_fingerprint": config_receipt["config_fingerprint"],
        "reference_fingerprint": reference_receipt["reference_fingerprint"],
        "alignment_catalog_fingerprint": (
            prepared.catalog.miss_alignment_fingerprint
        ),
        "alignment_receipt_fingerprint": (
            alignment_receipt["alignment_receipt_fingerprint"]
        ),
        "m_decoder_artifact_fingerprint": m_artifact.artifact_fingerprint,
        "m_decoder_state_fingerprint": m_artifact.decoder_state_fingerprint,
        "m_calibration_receipt_fingerprint": (
            calibration_receipt["calibration_receipt_fingerprint"]
        ),
        "results_fingerprint": results_receipt["results_fingerprint"],
        "gate_fingerprint": gate_fingerprint,
        "mechanism_signal": gate["mechanism_signal"],
        "dataset": d_v_dataset.manifest.dataset,
        "manifest_fingerprint": d_v_bundle.split_manifest_fingerprint,
        "manifest_file_sha256": d_v_bundle.split_manifest_file_sha256,
        "preprocessing_fingerprint": d_v_bundle.preprocessing_fingerprint,
        "base_fingerprint": d_v_bundle.base_fingerprint,
        "base_state_fingerprint": d_v_bundle.base_state_fingerprint,
        "d_r_base_index_fingerprint": d_r_bundle.base_index_fingerprint,
        "d_r_base_index_sha256": d_r_bundle.base_index_sha256,
        "d_r_state_index_fingerprint": d_r_bundle.state_index_fingerprint,
        "d_r_state_index_sha256": d_r_bundle.state_index_sha256,
        "d_v_base_index_fingerprint": d_v_bundle.base_index_fingerprint,
        "d_v_base_index_sha256": d_v_bundle.base_index_sha256,
        "reference_decoder_artifact_fingerprints": {
            "F": snapshot.factual_artifact.artifact_fingerprint,
            "F×": (
                snapshot.factual_exposure_matched_artifact.artifact_fingerprint
            ),
            "U": snapshot.uniform_artifact.artifact_fingerprint,
        },
        "artifact_directories": directories,
        "artifact_files": files,
        "m_decoder_receipt_sha256": m_artifact.receipt_sha256,
        "m_train_log_fingerprint": m_artifact.train_log_fingerprint,
    }
    complete = {
        **complete_core,
        "complete_fingerprint": stable_fingerprint(complete_core),
    }
    _write_new_json(root / _COMPLETE_NAME, complete)
    published_complete = _strict_json(
        root / _COMPLETE_NAME,
        name="published M extension COMPLETE receipt",
    )
    if published_complete != complete:
        raise RuntimeError("published M extension COMPLETE bytes differ")
    published_directories, published_files = _extension_tree_inventory(root)
    if (
        published_directories != directories
        or published_files != files
        or stable_fingerprint(
            {
                key: value
                for key, value in published_complete.items()
                if key != "complete_fingerprint"
            }
        )
        != published_complete["complete_fingerprint"]
    ):
        raise RuntimeError("published M extension inventory differs")
    incomplete.unlink()
    final_directories, final_files = _tree_inventory(root)
    if final_directories != directories or final_files != files:
        raise RuntimeError("final M extension inventory differs")
    return _bind_published(
        root=root,
        reference=snapshot,
        m_artifact=m_artifact,
        m_calibration=m_calibration,
        m_metrics=m_metrics,
        u_fixed_metrics=u_fixed_metrics,
        alignment_catalog_fingerprint=(
            prepared.catalog.miss_alignment_fingerprint
        ),
        results_fingerprint=results_receipt["results_fingerprint"],
        complete_fingerprint=complete["complete_fingerprint"],
        mechanism_signal=bool(gate["mechanism_signal"]),
    )


__all__ = [
    "PublishedStageAMExtension",
    "STAGE_A_M_METHOD_ORDER",
    "run_stage_a_m_extension",
]
