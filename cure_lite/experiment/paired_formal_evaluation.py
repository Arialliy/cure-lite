"""Strict D_V result contract for formal paired CURE-Lite methods.

This module is intentionally narrower than a training runner.  It composes the
already verified ``formal_evaluation`` build/select/evaluate path for
``LoadedPairedDecoderArtifact`` and turns one selected operating point into a
create-only, fingerprinted result receipt.  It exposes no split selector and no
``D_T`` entry point.

Historical Base@B/F/F×/U results are adapted from their already frozen metric
receipts.  The adapter performs integer-consistency checks against the fixed
IRSTD-1K D_V population; it never rebuilds cache samples, reselects a threshold,
or reevaluates a historical decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Mapping

from ..cache.schema import stable_fingerprint
from ..calibration import FalseAlarmBudget, THRESHOLD_SELECTION_RULE
from ..config import MatchConfig, OccupancyConfig, config_to_dict
from ..metrics import (
    AggregateEvaluation,
    FORMAL_STAGE_A_METRIC_FIELDS,
    formal_stage_a_metrics_payload,
)
from .cache_pipeline import LoadedDVCacheBundle
from .formal_evaluation import (
    FormalDVThresholdReceipt,
    LoadedDVMethodRun,
    build_loaded_d_v_method_run,
    evaluate_formal_residual_threshold,
    select_formal_residual_threshold_from_ledger,
)
from .paired_artifacts import PAIRED_METHODS, LoadedPairedDecoderArtifact
from .paired_formal_decision import (
    FORMAL_SEEDS,
    HISTORICAL_COMPARATORS,
    FormalMethodEvidence,
)


PAIRED_FORMAL_DV_RESULT_SCHEMA = "cure-lite-paired-formal-d-v-result-v1"
PAIRED_FORMAL_COMPARISON_PROTOCOL_SCHEMA = (
    "cure-lite-paired-formal-comparison-protocol-v1"
)
HISTORICAL_EVIDENCE_ADAPTER_SCHEMA = (
    "cure-lite-frozen-historical-formal-evidence-adapter-v1"
)
FORMAL_DV_TOTAL_TARGETS = 170
FORMAL_DV_ANCHOR_MISSES = 23
FORMAL_DV_ANCHOR_COVERED = (
    FORMAL_DV_TOTAL_TARGETS - FORMAL_DV_ANCHOR_MISSES
)
FORMAL_DV_IMAGES = 120
NULL_OPERATING_POINT_POLICY = (
    "include_exactly_one_null_residual_off_candidate_"
    "tie_ranked_above_numeric_thresholds-v1"
)
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


def _exact_integer_fraction(
    value: object,
    denominator: int,
    *,
    name: str,
) -> int:
    number = _finite(value, name=name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    scaled = number * denominator
    integer = int(round(scaled))
    if not isclose(scaled, integer, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            f"{name} is not an exact integer fraction of {denominator}"
        )
    return integer


def _aggregate_payload(metrics: AggregateEvaluation) -> dict[str, object]:
    if not isinstance(metrics, AggregateEvaluation):
        raise TypeError("metrics must be AggregateEvaluation")
    return {name: getattr(metrics, name) for name in _AGGREGATE_FIELDS}


def _aggregate_from_payload(value: object) -> AggregateEvaluation:
    if not isinstance(value, Mapping) or set(value) != set(_AGGREGATE_FIELDS):
        raise ValueError("aggregate_evaluation fields are not canonical")
    payload: dict[str, object] = {}
    for name in _AGGREGATE_FIELDS:
        item = value[name]
        if name in _AGGREGATE_BOOL_FIELDS:
            if not isinstance(item, bool):
                raise TypeError(f"aggregate_evaluation.{name} must be bool")
            payload[name] = item
        elif name in _AGGREGATE_INTEGER_FIELDS:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"aggregate_evaluation.{name} must be an integer"
                )
            payload[name] = item
        else:
            payload[name] = _finite(
                item,
                name=f"aggregate_evaluation.{name}",
            )
    return AggregateEvaluation(**payload)  # type: ignore[arg-type]


def _population_counts(
    metrics: AggregateEvaluation,
) -> tuple[int, int, int, int]:
    """Validate and return total, true, total-miss, recovered-miss counts."""

    if not isinstance(metrics, AggregateEvaluation):
        raise TypeError("metrics must be AggregateEvaluation")
    if metrics.images != FORMAL_DV_IMAGES:
        raise ValueError(
            "formal D_V evaluation must contain exactly 120 images"
        )
    if metrics.total_anchor_misses != FORMAL_DV_ANCHOR_MISSES:
        raise ValueError(
            "formal D_V anchor-miss denominator must remain exactly 23"
        )
    if metrics.total_anchor_covered != FORMAL_DV_ANCHOR_COVERED:
        raise ValueError(
            "formal D_V anchor-covered denominator must remain exactly 147"
        )
    total_targets = (
        metrics.total_anchor_misses + metrics.total_anchor_covered
    )
    if total_targets != FORMAL_DV_TOTAL_TARGETS:
        raise ValueError("formal D_V total target count must remain exactly 170")
    count_bounds = (
        0 <= metrics.recovered_anchor_misses <= metrics.total_anchor_misses,
        0 <= metrics.retained_anchor_covered <= metrics.total_anchor_covered,
        0
        <= metrics.recovered_reachable_anchor_misses
        <= metrics.total_reachable_anchor_misses
        <= metrics.total_anchor_misses,
        metrics.recovered_reachable_anchor_misses
        <= metrics.recovered_anchor_misses,
    )
    if not all(count_bounds):
        raise ValueError("formal D_V recovery counts are inconsistent")
    true_targets = (
        metrics.retained_anchor_covered + metrics.recovered_anchor_misses
    )
    expected_pd = true_targets / total_targets
    if not isclose(metrics.pd, expected_pd, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Pd is not the exact matched-target fraction")
    expected_retention = (
        metrics.retained_anchor_covered / metrics.total_anchor_covered
    )
    if not isclose(
        metrics.retention,
        expected_retention,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("retention is not the exact anchor-covered fraction")
    expected_rmr = (
        metrics.recovered_anchor_misses / metrics.total_anchor_misses
    )
    if (
        not isclose(metrics.rmr, expected_rmr, rel_tol=0.0, abs_tol=1e-12)
        or not isclose(
            metrics.gross_rmr,
            expected_rmr,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("gross recovery ratio is inconsistent with counts")
    expected_net = true_targets - metrics.total_anchor_covered
    if metrics.net_recovered_anchor_misses != expected_net:
        raise ValueError("net recovered-miss count is inconsistent with Pd")
    if not isclose(
        metrics.net_rmr,
        expected_net / metrics.total_anchor_misses,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("net recovery ratio is inconsistent with counts")
    reachable_ratio = (
        metrics.recovered_reachable_anchor_misses
        / metrics.total_reachable_anchor_misses
        if metrics.total_reachable_anchor_misses
        else 0.0
    )
    if not isclose(
        metrics.reachable_rmr,
        reachable_ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("reachable recovery ratio is inconsistent with counts")
    if not isclose(
        metrics.oracle_upper_bound,
        metrics.total_reachable_anchor_misses
        / metrics.total_anchor_misses,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("oracle upper bound is inconsistent with counts")
    bounded_unit_fields = (
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
        for name in bounded_unit_fields
    ):
        raise ValueError("formal D_V unit metrics must lie in [0,1]")
    if metrics.overlap_supported_rmr > metrics.gross_rmr:
        raise ValueError(
            "overlap-supported recovery cannot exceed gross recovery"
        )
    if any(
        _finite(getattr(metrics, name), name=name) < 0.0
        for name in (
            "pixel_fa",
            "raw_background_fa",
            "fp_components_per_mp",
        )
    ):
        raise ValueError("formal D_V false-addition metrics must be non-negative")
    if not isinstance(metrics.budget_violation, bool):
        raise TypeError("budget_violation must be bool")
    return (
        total_targets,
        true_targets,
        metrics.total_anchor_misses,
        metrics.recovered_anchor_misses,
    )


def _budget_payload(budget: FalseAlarmBudget) -> dict[str, float | None]:
    if not isinstance(budget, FalseAlarmBudget):
        raise TypeError("budget must be FalseAlarmBudget")
    return {
        "pixel_fa_budget": budget.pixel_fa_budget,
        "component_fa_per_mp_budget": (
            budget.component_fa_per_mp_budget
            if isfinite(budget.component_fa_per_mp_budget)
            else None
        ),
        "raw_background_fa_budget": (
            budget.raw_background_fa_budget
            if isfinite(budget.raw_background_fa_budget)
            else None
        ),
        "minimum_retention": budget.minimum_retention,
    }


def _budget_from_payload(value: object) -> FalseAlarmBudget:
    expected = {
        "pixel_fa_budget",
        "component_fa_per_mp_budget",
        "raw_background_fa_budget",
        "minimum_retention",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("false_alarm_budget fields are not canonical")

    def optional_limit(name: str) -> float:
        item = value[name]
        return (
            float("inf")
            if item is None
            else _finite(item, name=f"false_alarm_budget.{name}")
        )

    return FalseAlarmBudget(
        pixel_fa_budget=_finite(
            value["pixel_fa_budget"],
            name="false_alarm_budget.pixel_fa_budget",
        ),
        component_fa_per_mp_budget=optional_limit(
            "component_fa_per_mp_budget"
        ),
        raw_background_fa_budget=optional_limit(
            "raw_background_fa_budget"
        ),
        minimum_retention=_finite(
            value["minimum_retention"],
            name="false_alarm_budget.minimum_retention",
        ),
    )


@dataclass(frozen=True)
class HistoricalFXV3Binding:
    """Frozen seed-specific identities of one authoritative fx_v3 run."""

    seed: int
    run_config_fingerprint: str
    config_file_sha256: str
    calibration_receipt_fingerprint: str
    results_fingerprint: str
    complete_fingerprint: str
    base_at_budget_protocol_fingerprint: str
    factual_protocol_fingerprint: str
    factual_exposure_matched_protocol_fingerprint: str
    uniform_protocol_fingerprint: str
    base_at_budget_metrics_fingerprint: str
    factual_metrics_fingerprint: str
    factual_exposure_matched_metrics_fingerprint: str
    uniform_metrics_fingerprint: str

    def __post_init__(self) -> None:
        if self.seed not in FORMAL_SEEDS:
            raise ValueError(f"historical seed must be one of {FORMAL_SEEDS}")
        for name in (
            "run_config_fingerprint",
            "config_file_sha256",
            "calibration_receipt_fingerprint",
            "results_fingerprint",
            "complete_fingerprint",
            "base_at_budget_protocol_fingerprint",
            "factual_protocol_fingerprint",
            "factual_exposure_matched_protocol_fingerprint",
            "uniform_protocol_fingerprint",
            "base_at_budget_metrics_fingerprint",
            "factual_metrics_fingerprint",
            "factual_exposure_matched_metrics_fingerprint",
            "uniform_metrics_fingerprint",
        ):
            _digest(getattr(self, name), name=f"historical.{name}")

    def protocol_fingerprint_for(self, method: str) -> str:
        by_method = {
            "Base@B": self.base_at_budget_protocol_fingerprint,
            "F": self.factual_protocol_fingerprint,
            "F×": self.factual_exposure_matched_protocol_fingerprint,
            "U": self.uniform_protocol_fingerprint,
        }
        if method not in by_method:
            raise ValueError(
                f"historical method must be one of {HISTORICAL_COMPARATORS}"
            )
        return by_method[method]

    def metrics_fingerprint_for(self, method: str) -> str:
        by_method = {
            "Base@B": self.base_at_budget_metrics_fingerprint,
            "F": self.factual_metrics_fingerprint,
            "F×": self.factual_exposure_matched_metrics_fingerprint,
            "U": self.uniform_metrics_fingerprint,
        }
        if method not in by_method:
            raise ValueError(
                f"historical method must be one of {HISTORICAL_COMPARATORS}"
            )
        return by_method[method]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "run_config_fingerprint": self.run_config_fingerprint,
            "config_file_sha256": self.config_file_sha256,
            "calibration_receipt_fingerprint": (
                self.calibration_receipt_fingerprint
            ),
            "results_fingerprint": self.results_fingerprint,
            "complete_fingerprint": self.complete_fingerprint,
            "method_protocol_fingerprints": {
                "Base@B": self.base_at_budget_protocol_fingerprint,
                "F": self.factual_protocol_fingerprint,
                "F×": self.factual_exposure_matched_protocol_fingerprint,
                "U": self.uniform_protocol_fingerprint,
            },
            "method_metrics_fingerprints": {
                "Base@B": self.base_at_budget_metrics_fingerprint,
                "F": self.factual_metrics_fingerprint,
                "F×": self.factual_exposure_matched_metrics_fingerprint,
                "U": self.uniform_metrics_fingerprint,
            },
        }

    @classmethod
    def from_mapping(cls, value: object) -> HistoricalFXV3Binding:
        expected = {
            "seed",
            "run_config_fingerprint",
            "config_file_sha256",
            "calibration_receipt_fingerprint",
            "results_fingerprint",
            "complete_fingerprint",
            "method_protocol_fingerprints",
            "method_metrics_fingerprints",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("historical fx_v3 binding fields are not canonical")
        method_fingerprints = value["method_protocol_fingerprints"]
        if (
            not isinstance(method_fingerprints, Mapping)
            or set(method_fingerprints) != set(HISTORICAL_COMPARATORS)
        ):
            raise ValueError(
                "historical method protocol fingerprints are not canonical"
            )
        method_metrics_fingerprints = value["method_metrics_fingerprints"]
        if (
            not isinstance(method_metrics_fingerprints, Mapping)
            or set(method_metrics_fingerprints)
            != set(HISTORICAL_COMPARATORS)
        ):
            raise ValueError(
                "historical method metric fingerprints are not canonical"
            )
        return cls(
            seed=value["seed"],  # type: ignore[arg-type]
            run_config_fingerprint=value["run_config_fingerprint"],  # type: ignore[arg-type]
            config_file_sha256=value["config_file_sha256"],  # type: ignore[arg-type]
            calibration_receipt_fingerprint=value[
                "calibration_receipt_fingerprint"
            ],  # type: ignore[arg-type]
            results_fingerprint=value["results_fingerprint"],  # type: ignore[arg-type]
            complete_fingerprint=value["complete_fingerprint"],  # type: ignore[arg-type]
            base_at_budget_protocol_fingerprint=method_fingerprints[
                "Base@B"
            ],  # type: ignore[arg-type]
            factual_protocol_fingerprint=method_fingerprints["F"],  # type: ignore[arg-type]
            factual_exposure_matched_protocol_fingerprint=(
                method_fingerprints["F×"]  # type: ignore[arg-type]
            ),
            uniform_protocol_fingerprint=method_fingerprints["U"],  # type: ignore[arg-type]
            base_at_budget_metrics_fingerprint=(
                method_metrics_fingerprints["Base@B"]  # type: ignore[arg-type]
            ),
            factual_metrics_fingerprint=method_metrics_fingerprints[
                "F"
            ],  # type: ignore[arg-type]
            factual_exposure_matched_metrics_fingerprint=(
                method_metrics_fingerprints["F×"]  # type: ignore[arg-type]
            ),
            uniform_metrics_fingerprint=method_metrics_fingerprints[
                "U"
            ],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class _ComparisonProtocolSeal:
    core_fingerprint: str


@dataclass(frozen=True)
class FrozenComparisonProtocol:
    """Authoritative common D_V protocol for paired and historical methods."""

    dataset: str
    ordered_d_v_sample_ids_fingerprint: str
    manifest_fingerprint: str
    manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    d_v_base_index_fingerprint: str
    d_v_base_index_sha256: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    base_samples_fingerprint: str
    occupancy_config: OccupancyConfig
    match_config: MatchConfig
    residual_thresholds: tuple[float, ...]
    budget: FalseAlarmBudget
    historical_fx_v3: tuple[HistoricalFXV3Binding, ...]
    _verification_token: object

    def _core_payload(self) -> dict[str, object]:
        occupancy_payload = config_to_dict(self.occupancy_config)
        match_payload = config_to_dict(self.match_config)
        return {
            "schema_version": PAIRED_FORMAL_COMPARISON_PROTOCOL_SCHEMA,
            "dataset": self.dataset,
            "runtime_split": "D_V",
            "population": {
                "images": FORMAL_DV_IMAGES,
                "total_targets": FORMAL_DV_TOTAL_TARGETS,
                "anchor_covered": FORMAL_DV_ANCHOR_COVERED,
                "anchor_misses": FORMAL_DV_ANCHOR_MISSES,
            },
            "ordered_d_v_sample_ids_fingerprint": (
                self.ordered_d_v_sample_ids_fingerprint
            ),
            "bundle_binding": {
                "manifest_fingerprint": self.manifest_fingerprint,
                "manifest_file_sha256": self.manifest_file_sha256,
                "preprocessing_fingerprint": self.preprocessing_fingerprint,
                "base_fingerprint": self.base_fingerprint,
                "d_v_base_index_fingerprint": (
                    self.d_v_base_index_fingerprint
                ),
                "d_v_base_index_sha256": self.d_v_base_index_sha256,
                "d_v_image_fingerprint": self.d_v_image_fingerprint,
                "d_v_gt_fingerprint": self.d_v_gt_fingerprint,
                "base_samples_fingerprint": self.base_samples_fingerprint,
            },
            "occupancy_config": occupancy_payload,
            "occupancy_config_fingerprint": stable_fingerprint(
                occupancy_payload
            ),
            "match_config": match_payload,
            "match_config_fingerprint": stable_fingerprint(match_payload),
            "residual_threshold_grid": list(self.residual_thresholds),
            "false_alarm_budget": _budget_payload(self.budget),
            "selection_rule": THRESHOLD_SELECTION_RULE,
            "null_operating_point_policy": NULL_OPERATING_POINT_POLICY,
            "historical_fx_v3": [
                binding.canonical_payload()
                for binding in self.historical_fx_v3
            ],
        }

    def __post_init__(self) -> None:
        seal = self._verification_token
        if type(seal) is not _ComparisonProtocolSeal:
            raise TypeError(
                "FrozenComparisonProtocol must come from its strict loader"
            )
        if self.dataset != "IRSTD-1K":
            raise ValueError("formal comparison dataset must be IRSTD-1K")
        _digest(
            self.ordered_d_v_sample_ids_fingerprint,
            name="ordered_d_v_sample_ids_fingerprint",
        )
        for name in (
            "manifest_fingerprint",
            "manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "d_v_base_index_fingerprint",
            "d_v_base_index_sha256",
            "d_v_image_fingerprint",
            "d_v_gt_fingerprint",
            "base_samples_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        if not isinstance(self.occupancy_config, OccupancyConfig):
            raise TypeError("occupancy_config must be OccupancyConfig")
        if self.occupancy_config != OccupancyConfig(
            threshold=0.72,
            connectivity=8,
            min_component_area=1,
        ):
            raise ValueError("formal comparison occupancy config changed")
        if not isinstance(self.match_config, MatchConfig):
            raise TypeError("match_config must be MatchConfig")
        if self.match_config != MatchConfig(
            max_distance=3.0,
            distance_quantization=1_000_000,
            iou_quantization=1_000_000,
        ):
            raise ValueError("formal comparison matching config changed")
        expected_grid = tuple(index / 50 for index in range(51))
        if self.residual_thresholds != expected_grid:
            raise ValueError(
                "formal residual grid must remain 0..1 in steps of 0.02"
            )
        if self.budget != FalseAlarmBudget(
            pixel_fa_budget=1e-4,
            component_fa_per_mp_budget=100.0,
            raw_background_fa_budget=1e-4,
            minimum_retention=0.99,
        ):
            raise ValueError("formal comparison false-alarm budget changed")
        if (
            not isinstance(self.historical_fx_v3, tuple)
            or tuple(binding.seed for binding in self.historical_fx_v3)
            != FORMAL_SEEDS
            or any(
                not isinstance(binding, HistoricalFXV3Binding)
                for binding in self.historical_fx_v3
            )
        ):
            raise ValueError(
                "historical fx_v3 bindings must contain seeds 42 and 43"
            )
        if seal.core_fingerprint != stable_fingerprint(
            self._core_payload()
        ):
            raise TypeError("formal comparison protocol fields were replaced")

    @property
    def comparison_protocol_fingerprint(self) -> str:
        return stable_fingerprint(self._core_payload())

    def historical_binding(self, seed: int) -> HistoricalFXV3Binding:
        if seed not in FORMAL_SEEDS:
            raise ValueError(f"seed must be one of {FORMAL_SEEDS}")
        return next(
            binding
            for binding in self.historical_fx_v3
            if binding.seed == seed
        )

    def verify_bundle(self, bundle: LoadedDVCacheBundle) -> None:
        if not isinstance(bundle, LoadedDVCacheBundle):
            raise TypeError("bundle must be LoadedDVCacheBundle")
        bundle.verify_unchanged()
        expected = {
            "split_manifest_fingerprint": self.manifest_fingerprint,
            "split_manifest_file_sha256": self.manifest_file_sha256,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "base_fingerprint": self.base_fingerprint,
            "base_index_fingerprint": self.d_v_base_index_fingerprint,
            "base_index_sha256": self.d_v_base_index_sha256,
            "d_v_image_fingerprint": self.d_v_image_fingerprint,
            "d_v_gt_fingerprint": self.d_v_gt_fingerprint,
        }
        for name, value in expected.items():
            if getattr(bundle, name) != value:
                raise RuntimeError(
                    f"D_V bundle differs from comparison protocol at {name}"
                )
        sample_ids = tuple(row.sample_id for row in bundle.rows)
        if (
            len(sample_ids) != FORMAL_DV_IMAGES
            or len(set(sample_ids)) != FORMAL_DV_IMAGES
            or stable_fingerprint(list(sample_ids))
            != self.ordered_d_v_sample_ids_fingerprint
        ):
            raise RuntimeError(
                "D_V ordered sample identities differ from comparison protocol"
            )

    def verify_loaded_run(self, run: LoadedDVMethodRun) -> None:
        if not isinstance(run, LoadedDVMethodRun):
            raise TypeError("run must be LoadedDVMethodRun")
        self.verify_bundle(run.bundle)
        if run.base_samples_fingerprint != self.base_samples_fingerprint:
            raise RuntimeError(
                "D_V base tensor fingerprint differs from comparison protocol"
            )
        sample_ids = [
            sample.sample_id for sample in run.residual_samples
        ]
        if (
            len(sample_ids) != FORMAL_DV_IMAGES
            or stable_fingerprint(sample_ids)
            != self.ordered_d_v_sample_ids_fingerprint
        ):
            raise RuntimeError(
                "D_V method run order differs from comparison protocol"
            )
        config = run.artifact.config
        if (
            config.occupancy_config != self.occupancy_config
            or config.match_config != self.match_config
        ):
            raise RuntimeError(
                "decoder occupancy/matching differs from comparison protocol"
            )

    def verify_selected_receipt(
        self,
        run: LoadedDVMethodRun,
        receipt: FormalDVThresholdReceipt,
    ) -> None:
        self.verify_loaded_run(run)
        if not isinstance(receipt, FormalDVThresholdReceipt):
            raise TypeError("receipt must be FormalDVThresholdReceipt")
        protocol = receipt.protocol
        if (
            protocol.manifest_fingerprint != self.manifest_fingerprint
            or len(protocol.ordered_d_v_sample_ids) != FORMAL_DV_IMAGES
            or stable_fingerprint(list(protocol.ordered_d_v_sample_ids))
            != self.ordered_d_v_sample_ids_fingerprint
            or protocol.candidate_threshold_grid
            != self.residual_thresholds
            or protocol.occupancy_config != self.occupancy_config
            or protocol.match_config != self.match_config
            or protocol.budget != self.budget
            or protocol.selection_rule != THRESHOLD_SELECTION_RULE
        ):
            raise RuntimeError(
                "selected threshold receipt differs from comparison protocol"
            )


def _new_comparison_protocol(
    **values: object,
) -> FrozenComparisonProtocol:
    temporary = object.__new__(FrozenComparisonProtocol)
    for field in fields(FrozenComparisonProtocol):
        if field.name == "_verification_token":
            continue
        object.__setattr__(temporary, field.name, values[field.name])
    fingerprint = stable_fingerprint(temporary._core_payload())
    return FrozenComparisonProtocol(
        **values,
        _verification_token=_ComparisonProtocolSeal(fingerprint),
    )  # type: ignore[arg-type]


def _strict_json_mapping(path: Path, *, name: str) -> dict[str, object]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON object")
    return payload


def load_frozen_comparison_protocol(
    path: str | Path,
) -> FrozenComparisonProtocol:
    """Load the create-before-D_V common comparison protocol."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("comparison protocol may not be a symlink")
    source = requested.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("comparison protocol must be a regular file")
    payload = _strict_json_mapping(source, name="comparison protocol")
    expected_top = {
        "schema_version",
        "dataset",
        "runtime_split",
        "population",
        "ordered_d_v_sample_ids_fingerprint",
        "bundle_binding",
        "occupancy_config",
        "occupancy_config_fingerprint",
        "match_config",
        "match_config_fingerprint",
        "residual_threshold_grid",
        "false_alarm_budget",
        "selection_rule",
        "null_operating_point_policy",
        "historical_fx_v3",
        "comparison_protocol_fingerprint",
    }
    if set(payload) != expected_top:
        raise ValueError("comparison protocol fields are not canonical")
    population = payload["population"]
    if (
        payload["schema_version"]
        != PAIRED_FORMAL_COMPARISON_PROTOCOL_SCHEMA
        or payload["dataset"] != "IRSTD-1K"
        or payload["runtime_split"] != "D_V"
        or population
        != {
            "images": FORMAL_DV_IMAGES,
            "total_targets": FORMAL_DV_TOTAL_TARGETS,
            "anchor_covered": FORMAL_DV_ANCHOR_COVERED,
            "anchor_misses": FORMAL_DV_ANCHOR_MISSES,
        }
        or payload["selection_rule"] != THRESHOLD_SELECTION_RULE
        or payload["null_operating_point_policy"]
        != NULL_OPERATING_POINT_POLICY
    ):
        raise ValueError("comparison protocol constants changed")
    _digest(
        payload["ordered_d_v_sample_ids_fingerprint"],
        name="ordered_d_v_sample_ids_fingerprint",
    )
    binding_names = {
        "manifest_fingerprint",
        "manifest_file_sha256",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "d_v_base_index_fingerprint",
        "d_v_base_index_sha256",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
        "base_samples_fingerprint",
    }
    bundle_binding = payload["bundle_binding"]
    if (
        not isinstance(bundle_binding, Mapping)
        or set(bundle_binding) != binding_names
    ):
        raise ValueError("comparison bundle binding fields are not canonical")
    occupancy_payload = payload["occupancy_config"]
    match_payload = payload["match_config"]
    if not isinstance(occupancy_payload, Mapping) or set(
        occupancy_payload
    ) != {"threshold", "connectivity", "min_component_area"}:
        raise ValueError("comparison occupancy config is not canonical")
    if not isinstance(match_payload, Mapping) or set(match_payload) != {
        "max_distance",
        "distance_quantization",
        "iou_quantization",
    }:
        raise ValueError("comparison match config is not canonical")
    if payload["occupancy_config_fingerprint"] != stable_fingerprint(
        occupancy_payload
    ):
        raise ValueError("occupancy config fingerprint mismatch")
    if payload["match_config_fingerprint"] != stable_fingerprint(
        match_payload
    ):
        raise ValueError("match config fingerprint mismatch")
    history = payload["historical_fx_v3"]
    if not isinstance(history, list):
        raise TypeError("historical_fx_v3 must be a list")
    values = {
        "dataset": payload["dataset"],
        "ordered_d_v_sample_ids_fingerprint": payload[
            "ordered_d_v_sample_ids_fingerprint"
        ],
        **dict(bundle_binding),
        "occupancy_config": OccupancyConfig(
            threshold=occupancy_payload["threshold"],  # type: ignore[arg-type]
            connectivity=occupancy_payload["connectivity"],  # type: ignore[arg-type]
            min_component_area=occupancy_payload[
                "min_component_area"
            ],  # type: ignore[arg-type]
        ),
        "match_config": MatchConfig(
            max_distance=match_payload["max_distance"],  # type: ignore[arg-type]
            distance_quantization=match_payload[
                "distance_quantization"
            ],  # type: ignore[arg-type]
            iou_quantization=match_payload["iou_quantization"],  # type: ignore[arg-type]
        ),
        "residual_thresholds": tuple(
            _finite(value, name="residual_threshold_grid item")
            for value in payload["residual_threshold_grid"]  # type: ignore[union-attr]
        ),
        "budget": _budget_from_payload(payload["false_alarm_budget"]),
        "historical_fx_v3": tuple(
            HistoricalFXV3Binding.from_mapping(item) for item in history
        ),
    }
    protocol = _new_comparison_protocol(**values)
    fingerprint = _digest(
        payload["comparison_protocol_fingerprint"],
        name="comparison_protocol_fingerprint",
    )
    if (
        fingerprint != protocol.comparison_protocol_fingerprint
        or payload
        != {
            **protocol._core_payload(),
            "comparison_protocol_fingerprint": fingerprint,
        }
    ):
        raise ValueError("comparison protocol fingerprint or payload mismatch")
    return protocol


@dataclass(frozen=True, slots=True)
class _PairedFormalResultSeal:
    core_fingerprint: str


@dataclass(frozen=True)
class PairedFormalDVResult:
    """One selected paired method/seed result, sealed to D_V provenance."""

    method: str
    seed: int
    comparison_protocol_fingerprint: str
    selected_threshold: float | None
    metrics: AggregateEvaluation
    budget: FalseAlarmBudget
    d_v_run_fingerprint: str
    threshold_protocol_fingerprint: str
    manifest_fingerprint: str
    manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    d_v_base_index_fingerprint: str
    d_v_base_index_sha256: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    residual_samples_fingerprint: str
    decoder_artifact_fingerprint: str
    decoder_receipt_sha256: str
    decoder_state_fingerprint: str
    formal_protocol_fingerprint: str
    paired_objective_fingerprint: str
    pair_catalog_fingerprint: str
    paired_schedule_fingerprint: str
    formal_schedule_fingerprint: str
    runtime_input_fingerprint: str
    control_preflight_fingerprint: str
    control_provider_fingerprint: str | None
    method_contract_fingerprint: str
    paired_criterion_fingerprint: str
    method_objective_fingerprint: str
    _verification_token: object

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": PAIRED_FORMAL_DV_RESULT_SCHEMA,
            "runtime_split": "D_V",
            "D_T_accessed": False,
            "method": self.method,
            "seed": self.seed,
            "comparison_protocol_fingerprint": (
                self.comparison_protocol_fingerprint
            ),
            "selected_threshold": self.selected_threshold,
            "aggregate_evaluation": _aggregate_payload(self.metrics),
            "false_alarm_budget": _budget_payload(self.budget),
            "bindings": {
                "d_v_run_fingerprint": self.d_v_run_fingerprint,
                "threshold_protocol_fingerprint": (
                    self.threshold_protocol_fingerprint
                ),
                "manifest_fingerprint": self.manifest_fingerprint,
                "manifest_file_sha256": self.manifest_file_sha256,
                "preprocessing_fingerprint": self.preprocessing_fingerprint,
                "base_fingerprint": self.base_fingerprint,
                "d_v_base_index_fingerprint": (
                    self.d_v_base_index_fingerprint
                ),
                "d_v_base_index_sha256": self.d_v_base_index_sha256,
                "d_v_image_fingerprint": self.d_v_image_fingerprint,
                "d_v_gt_fingerprint": self.d_v_gt_fingerprint,
                "residual_samples_fingerprint": (
                    self.residual_samples_fingerprint
                ),
                "decoder_artifact_fingerprint": (
                    self.decoder_artifact_fingerprint
                ),
                "decoder_receipt_sha256": self.decoder_receipt_sha256,
                "decoder_state_fingerprint": (
                    self.decoder_state_fingerprint
                ),
                "formal_protocol_fingerprint": (
                    self.formal_protocol_fingerprint
                ),
                "paired_objective_fingerprint": (
                    self.paired_objective_fingerprint
                ),
                "pair_catalog_fingerprint": self.pair_catalog_fingerprint,
                "paired_schedule_fingerprint": (
                    self.paired_schedule_fingerprint
                ),
                "formal_schedule_fingerprint": (
                    self.formal_schedule_fingerprint
                ),
                "runtime_input_fingerprint": (
                    self.runtime_input_fingerprint
                ),
                "control_preflight_fingerprint": (
                    self.control_preflight_fingerprint
                ),
                "control_provider_fingerprint": (
                    self.control_provider_fingerprint
                ),
                "method_contract_fingerprint": (
                    self.method_contract_fingerprint
                ),
                "paired_criterion_fingerprint": (
                    self.paired_criterion_fingerprint
                ),
                "method_objective_fingerprint": (
                    self.method_objective_fingerprint
                ),
            },
        }

    def __post_init__(self) -> None:
        seal = self._verification_token
        if type(seal) is not _PairedFormalResultSeal:
            raise TypeError(
                "PairedFormalDVResult must come from strict evaluation or loader"
            )
        if self.method not in PAIRED_METHODS:
            raise ValueError(f"method must be one of {PAIRED_METHODS}")
        if self.seed not in FORMAL_SEEDS:
            raise ValueError(f"seed must be one of {FORMAL_SEEDS}")
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
            raise ValueError("selected metrics violate the frozen budget")
        for name in (
            "d_v_run_fingerprint",
            "threshold_protocol_fingerprint",
            "manifest_fingerprint",
            "manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "d_v_base_index_fingerprint",
            "d_v_base_index_sha256",
            "d_v_image_fingerprint",
            "d_v_gt_fingerprint",
            "residual_samples_fingerprint",
            "decoder_artifact_fingerprint",
            "decoder_receipt_sha256",
            "decoder_state_fingerprint",
            "formal_protocol_fingerprint",
            "paired_objective_fingerprint",
            "pair_catalog_fingerprint",
            "paired_schedule_fingerprint",
            "formal_schedule_fingerprint",
            "runtime_input_fingerprint",
            "control_preflight_fingerprint",
            "method_contract_fingerprint",
            "paired_criterion_fingerprint",
            "method_objective_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        if self.method == "paired_difference":
            if self.control_provider_fingerprint is not None:
                raise ValueError(
                    "paired_difference must not bind a control provider"
                )
        else:
            _digest(
                self.control_provider_fingerprint,
                name="control_provider_fingerprint",
            )
        if seal.core_fingerprint != stable_fingerprint(
            self._core_payload()
        ):
            raise TypeError("paired formal D_V result fields were replaced")

    def verify_unchanged(self) -> None:
        seal = self._verification_token
        if (
            type(seal) is not _PairedFormalResultSeal
            or seal.core_fingerprint != stable_fingerprint(self._core_payload())
        ):
            raise RuntimeError("paired formal D_V result changed in memory")
        _population_counts(self.metrics)
        if self.metrics.budget_violation or not self.budget.accepts(
            self.metrics
        ):
            raise RuntimeError("paired formal D_V result budget binding changed")

    @property
    def result_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(self._core_payload())

    def to_formal_method_evidence(self) -> FormalMethodEvidence:
        self.verify_unchanged()
        total, true, total_misses, recovered = _population_counts(self.metrics)
        return FormalMethodEvidence(
            method=self.method,
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
        evidence = self.to_formal_method_evidence().canonical_payload()
        result_fingerprint = self.result_fingerprint
        receipt_core = {
            **core,
            "formal_method_evidence": evidence,
            "result_fingerprint": result_fingerprint,
        }
        return {
            **receipt_core,
            "receipt_fingerprint": stable_fingerprint(receipt_core),
        }

    @property
    def receipt_fingerprint(self) -> str:
        return str(self.canonical_payload()["receipt_fingerprint"])


def _new_result(**values: object) -> PairedFormalDVResult:
    temporary = object.__new__(PairedFormalDVResult)
    for field in fields(PairedFormalDVResult):
        if field.name == "_verification_token":
            continue
        object.__setattr__(temporary, field.name, values[field.name])
    core_fingerprint = stable_fingerprint(temporary._core_payload())
    return PairedFormalDVResult(
        **values,
        _verification_token=_PairedFormalResultSeal(core_fingerprint),
    )  # type: ignore[arg-type]


def result_from_selected_paired_evaluation(
    run: LoadedDVMethodRun,
    receipt: FormalDVThresholdReceipt,
    comparison_protocol: FrozenComparisonProtocol,
) -> PairedFormalDVResult:
    """Evaluate one already selected paired D_V operating point strictly."""

    if not isinstance(run, LoadedDVMethodRun):
        raise TypeError("run must be LoadedDVMethodRun")
    if not isinstance(receipt, FormalDVThresholdReceipt):
        raise TypeError("receipt must be FormalDVThresholdReceipt")
    if not isinstance(comparison_protocol, FrozenComparisonProtocol):
        raise TypeError(
            "comparison_protocol must be FrozenComparisonProtocol"
        )
    if not isinstance(run.artifact, LoadedPairedDecoderArtifact):
        raise TypeError("run must bind a LoadedPairedDecoderArtifact")
    comparison_protocol.verify_selected_receipt(run, receipt)
    run.verify_unchanged()
    run.artifact.verify_unchanged()
    config = run.artifact.config
    if (
        receipt.decoder_variant != config.method
        or receipt.global_seed != config.seed
    ):
        raise RuntimeError("selected receipt method/seed differs from artifact")
    metrics = evaluate_formal_residual_threshold(run, receipt)
    if metrics != receipt.protocol.selected_metrics:
        raise RuntimeError(
            "replayed AggregateEvaluation differs from selected metrics"
        )
    _population_counts(metrics)
    if metrics.budget_violation or not receipt.protocol.budget.accepts(metrics):
        raise RuntimeError("selected paired D_V result violates its budget")
    config_payload = config.canonical_payload()
    result = _new_result(
        method=config.method,
        seed=config.seed,
        comparison_protocol_fingerprint=(
            comparison_protocol.comparison_protocol_fingerprint
        ),
        selected_threshold=receipt.protocol.selected_threshold,
        metrics=metrics,
        budget=receipt.protocol.budget,
        d_v_run_fingerprint=run.run_fingerprint,
        threshold_protocol_fingerprint=receipt.protocol.receipt_fingerprint,
        manifest_fingerprint=run.bundle.split_manifest_fingerprint,
        manifest_file_sha256=run.bundle.split_manifest_file_sha256,
        preprocessing_fingerprint=run.bundle.preprocessing_fingerprint,
        base_fingerprint=run.bundle.base_fingerprint,
        d_v_base_index_fingerprint=run.bundle.base_index_fingerprint,
        d_v_base_index_sha256=run.bundle.base_index_sha256,
        d_v_image_fingerprint=run.bundle.d_v_image_fingerprint,
        d_v_gt_fingerprint=run.bundle.d_v_gt_fingerprint,
        residual_samples_fingerprint=run.residual_samples_fingerprint,
        decoder_artifact_fingerprint=run.artifact.artifact_fingerprint,
        decoder_receipt_sha256=run.artifact.receipt_sha256,
        decoder_state_fingerprint=run.artifact.decoder_state_fingerprint,
        formal_protocol_fingerprint=config.formal_protocol_fingerprint,
        paired_objective_fingerprint=config.paired_objective_fingerprint,
        pair_catalog_fingerprint=config.pair_catalog_fingerprint,
        paired_schedule_fingerprint=config.paired_schedule_fingerprint,
        formal_schedule_fingerprint=config.formal_schedule_fingerprint,
        runtime_input_fingerprint=config.runtime_input_fingerprint,
        control_preflight_fingerprint=config.control_preflight_fingerprint,
        control_provider_fingerprint=config.control_provider_fingerprint,
        method_contract_fingerprint=config.method_contract_fingerprint,
        paired_criterion_fingerprint=stable_fingerprint(
            config_payload["paired_criterion"]
        ),
        method_objective_fingerprint=stable_fingerprint(
            config_payload["method_objective"]
        ),
    )
    run.verify_unchanged()
    run.artifact.verify_unchanged()
    comparison_protocol.verify_selected_receipt(run, receipt)
    return result


def select_and_evaluate_paired_formal_method(
    bundle: LoadedDVCacheBundle,
    artifact: LoadedPairedDecoderArtifact,
    *,
    comparison_protocol: FrozenComparisonProtocol,
) -> PairedFormalDVResult:
    """Compose the existing strict D_V build, select, and evaluate path."""

    if not isinstance(bundle, LoadedDVCacheBundle):
        raise TypeError("bundle must be LoadedDVCacheBundle")
    if not isinstance(artifact, LoadedPairedDecoderArtifact):
        raise TypeError("artifact must be LoadedPairedDecoderArtifact")
    if not isinstance(comparison_protocol, FrozenComparisonProtocol):
        raise TypeError(
            "comparison_protocol must be FrozenComparisonProtocol"
        )
    comparison_protocol.verify_bundle(bundle)
    run = build_loaded_d_v_method_run(bundle, artifact)
    comparison_protocol.verify_loaded_run(run)
    receipt = select_formal_residual_threshold_from_ledger(
        run,
        comparison_protocol.residual_thresholds,
        comparison_protocol.budget,
        method_label=artifact.config.method,
    )
    comparison_protocol.verify_selected_receipt(run, receipt)
    return result_from_selected_paired_evaluation(
        run,
        receipt,
        comparison_protocol,
    )


def adapt_frozen_historical_method_evidence(
    *,
    method: str,
    seed: int,
    selected_metrics: Mapping[str, object],
    comparison_protocol: FrozenComparisonProtocol,
    source_results_fingerprint: str,
    source_protocol_fingerprint: str,
    source_run_config_fingerprint: str,
    source_config_file_sha256: str,
    source_calibration_receipt_fingerprint: str,
    source_complete_fingerprint: str,
) -> FormalMethodEvidence:
    """Adapt frozen Base@B/F/F×/U values without reevaluation.

    The fixed 170-target D_V partition makes the integer true-target,
    retained-target, and recovered-miss counts identifiable from the persisted
    Pd and retention values.  Rejecting non-integral values prevents rounded
    table values from being passed off as authoritative receipts.
    """

    if method not in HISTORICAL_COMPARATORS:
        raise ValueError(
            f"historical method must be one of {HISTORICAL_COMPARATORS}"
        )
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"seed must be one of {FORMAL_SEEDS}")
    if not isinstance(comparison_protocol, FrozenComparisonProtocol):
        raise TypeError(
            "comparison_protocol must be FrozenComparisonProtocol"
        )
    if (
        not isinstance(selected_metrics, Mapping)
        or set(selected_metrics) != set(FORMAL_STAGE_A_METRIC_FIELDS)
    ):
        raise ValueError("historical selected_metrics fields are not canonical")
    normalized: dict[str, float | bool] = {}
    for name in FORMAL_STAGE_A_METRIC_FIELDS:
        value = selected_metrics[name]
        if name == "budget_violation":
            if not isinstance(value, bool):
                raise TypeError("historical budget_violation must be bool")
            normalized[name] = value
        else:
            normalized[name] = _finite(
                value,
                name=f"historical selected_metrics.{name}",
            )
    historical_binding = comparison_protocol.historical_binding(seed)
    if stable_fingerprint(normalized) != (
        historical_binding.metrics_fingerprint_for(method)
    ):
        raise RuntimeError(
            "historical selected metrics differ from the frozen fx_v3 result"
        )
    true_targets = _exact_integer_fraction(
        normalized["pd"],
        FORMAL_DV_TOTAL_TARGETS,
        name="historical Pd",
    )
    retained = _exact_integer_fraction(
        normalized["retention"],
        FORMAL_DV_ANCHOR_COVERED,
        name="historical retention",
    )
    recovered = true_targets - retained
    if not 0 <= recovered <= FORMAL_DV_ANCHOR_MISSES:
        raise ValueError(
            "historical Pd/retention imply an invalid recovered-miss count"
        )
    source_results_fingerprint = _digest(
        source_results_fingerprint,
        name="source_results_fingerprint",
    )
    source_protocol_fingerprint = _digest(
        source_protocol_fingerprint,
        name="source_protocol_fingerprint",
    )
    source_run_config_fingerprint = _digest(
        source_run_config_fingerprint,
        name="source_run_config_fingerprint",
    )
    source_config_file_sha256 = _digest(
        source_config_file_sha256,
        name="source_config_file_sha256",
    )
    source_calibration_receipt_fingerprint = _digest(
        source_calibration_receipt_fingerprint,
        name="source_calibration_receipt_fingerprint",
    )
    source_complete_fingerprint = _digest(
        source_complete_fingerprint,
        name="source_complete_fingerprint",
    )
    expected_sources = {
        "source_results_fingerprint": historical_binding.results_fingerprint,
        "source_protocol_fingerprint": (
            historical_binding.protocol_fingerprint_for(method)
        ),
        "source_run_config_fingerprint": (
            historical_binding.run_config_fingerprint
        ),
        "source_config_file_sha256": historical_binding.config_file_sha256,
        "source_calibration_receipt_fingerprint": (
            historical_binding.calibration_receipt_fingerprint
        ),
        "source_complete_fingerprint": (
            historical_binding.complete_fingerprint
        ),
    }
    actual_sources = {
        "source_results_fingerprint": source_results_fingerprint,
        "source_protocol_fingerprint": source_protocol_fingerprint,
        "source_run_config_fingerprint": source_run_config_fingerprint,
        "source_config_file_sha256": source_config_file_sha256,
        "source_calibration_receipt_fingerprint": (
            source_calibration_receipt_fingerprint
        ),
        "source_complete_fingerprint": source_complete_fingerprint,
    }
    if actual_sources != expected_sources:
        raise RuntimeError(
            "historical evidence differs from the frozen fx_v3 binding"
        )
    adapter_payload = {
        "schema_version": HISTORICAL_EVIDENCE_ADAPTER_SCHEMA,
        "source_is_frozen_receipt": True,
        "reevaluated": False,
        "method": method,
        "seed": seed,
        "selected_metrics": normalized,
        "fixed_population": {
            "total_targets": FORMAL_DV_TOTAL_TARGETS,
            "total_anchor_misses": FORMAL_DV_ANCHOR_MISSES,
            "total_anchor_covered": FORMAL_DV_ANCHOR_COVERED,
        },
        "derived_exact_counts": {
            "true_targets": true_targets,
            "retained_anchor_covered": retained,
            "recovered_anchor_misses": recovered,
        },
        "source_results_fingerprint": source_results_fingerprint,
        "source_protocol_fingerprint": source_protocol_fingerprint,
        "source_run_config_fingerprint": source_run_config_fingerprint,
        "source_config_file_sha256": source_config_file_sha256,
        "source_calibration_receipt_fingerprint": (
            source_calibration_receipt_fingerprint
        ),
        "source_complete_fingerprint": source_complete_fingerprint,
        "comparison_protocol_fingerprint": (
            comparison_protocol.comparison_protocol_fingerprint
        ),
    }
    return FormalMethodEvidence(
        method=method,
        seed=seed,
        total_targets=FORMAL_DV_TOTAL_TARGETS,
        true_targets=true_targets,
        pd=float(normalized["pd"]),
        total_anchor_misses=FORMAL_DV_ANCHOR_MISSES,
        recovered_anchor_misses=recovered,
        retention=float(normalized["retention"]),
        pixel_fa=float(normalized["pixel_fa"]),
        raw_background_fa=float(normalized["raw_background_fa"]),
        fp_components_per_mp=float(normalized["fp_components_per_mp"]),
        budget_violation=bool(normalized["budget_violation"]),
        comparison_protocol_fingerprint=(
            comparison_protocol.comparison_protocol_fingerprint
        ),
        result_fingerprint=stable_fingerprint(adapter_payload),
    )


def save_paired_formal_d_v_result(
    path: str | Path,
    result: PairedFormalDVResult,
) -> str:
    """Write one canonical JSON receipt without overwriting any path."""

    if not isinstance(result, PairedFormalDVResult):
        raise TypeError("result must be PairedFormalDVResult")
    result.verify_unchanged()
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("paired formal D_V result target may not be a symlink")
    target = requested.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            result.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with target.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite paired formal D_V result {target}"
        ) from None
    return result.receipt_fingerprint


def load_paired_formal_d_v_result(
    path: str | Path,
) -> PairedFormalDVResult:
    """Strictly load and fingerprint-check a persisted result receipt."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("paired formal D_V result may not be a symlink")
    source = requested.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("paired formal D_V result must be a regular file")
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    "paired formal D_V result contains duplicate JSON keys"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("paired formal D_V result is not valid UTF-8 JSON") from error
    expected_top = {
        "schema_version",
        "runtime_split",
        "D_T_accessed",
        "method",
        "seed",
        "comparison_protocol_fingerprint",
        "selected_threshold",
        "aggregate_evaluation",
        "false_alarm_budget",
        "bindings",
        "formal_method_evidence",
        "result_fingerprint",
        "receipt_fingerprint",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_top:
        raise ValueError("paired formal D_V result fields are not canonical")
    if (
        payload["schema_version"] != PAIRED_FORMAL_DV_RESULT_SCHEMA
        or payload["runtime_split"] != "D_V"
        or payload["D_T_accessed"] is not False
    ):
        raise ValueError("paired formal D_V result split/schema is invalid")
    expected_bindings = {
        "d_v_run_fingerprint",
        "threshold_protocol_fingerprint",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "d_v_base_index_fingerprint",
        "d_v_base_index_sha256",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
        "residual_samples_fingerprint",
        "decoder_artifact_fingerprint",
        "decoder_receipt_sha256",
        "decoder_state_fingerprint",
        "formal_protocol_fingerprint",
        "paired_objective_fingerprint",
        "pair_catalog_fingerprint",
        "paired_schedule_fingerprint",
        "formal_schedule_fingerprint",
        "runtime_input_fingerprint",
        "control_preflight_fingerprint",
        "control_provider_fingerprint",
        "method_contract_fingerprint",
        "paired_criterion_fingerprint",
        "method_objective_fingerprint",
    }
    bindings = payload["bindings"]
    if not isinstance(bindings, Mapping) or set(bindings) != expected_bindings:
        raise ValueError("paired formal D_V result bindings are not canonical")
    metrics = _aggregate_from_payload(payload["aggregate_evaluation"])
    budget = _budget_from_payload(payload["false_alarm_budget"])
    values = {
        "method": payload["method"],
        "seed": payload["seed"],
        "comparison_protocol_fingerprint": payload[
            "comparison_protocol_fingerprint"
        ],
        "selected_threshold": payload["selected_threshold"],
        "metrics": metrics,
        "budget": budget,
        **dict(bindings),
    }
    result = _new_result(**values)
    expected = result.canonical_payload()
    if dict(payload) != expected:
        raise ValueError(
            "paired formal D_V result fingerprint or evidence binding mismatch"
        )
    return result


__all__ = [
    "FORMAL_DV_ANCHOR_COVERED",
    "FORMAL_DV_ANCHOR_MISSES",
    "FORMAL_DV_IMAGES",
    "FORMAL_DV_TOTAL_TARGETS",
    "HISTORICAL_EVIDENCE_ADAPTER_SCHEMA",
    "NULL_OPERATING_POINT_POLICY",
    "PAIRED_FORMAL_COMPARISON_PROTOCOL_SCHEMA",
    "PAIRED_FORMAL_DV_RESULT_SCHEMA",
    "FrozenComparisonProtocol",
    "HistoricalFXV3Binding",
    "PairedFormalDVResult",
    "adapt_frozen_historical_method_evidence",
    "load_frozen_comparison_protocol",
    "load_paired_formal_d_v_result",
    "result_from_selected_paired_evaluation",
    "save_paired_formal_d_v_result",
    "select_and_evaluate_paired_formal_method",
]
