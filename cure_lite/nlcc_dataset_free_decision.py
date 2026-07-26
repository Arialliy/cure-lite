"""Independent decision recomputation for NLCC-v12 dataset-free results.

This module deliberately does not import the runner.  It consumes a completed
result mapping plus the frozen runner configuration, validates the
decision-relevant result schema, and recomputes every numeric and structural
gate from the recorded observations.

Embedded ``checks``, ``all_pass``, and ``decision`` fields are treated as
redundant assertions.  They must agree with recomputation; they are never used
as inputs to it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

from .cache.schema import stable_fingerprint
from .nlcc_dataset_free_runner_config import (
    DEVELOPMENT,
    NLCCDatasetFreeRunnerConfig,
)


RESULT_SCHEMA = "cure-lite.nlcc-v12.dataset-free-result.v2"
PASS = "PASS"
GATE_FAIL = "GATE_FAIL"

_GLOBAL_METRIC_KEYS = {
    "population_total_loss",
    "factual_miss_target_min",
    "factual_miss_background_max",
    "factual_no_miss_max",
    "factual_miss_loss",
    "factual_no_miss_loss",
    "pair_loss",
}
_GLOBAL_CHECK_KEYS = {
    "population_total_loss",
    "factual_miss_target",
    "factual_miss_background",
    "factual_no_miss",
}
_COMMON_METRIC_KEYS = {
    "row_count",
    "slot_count",
    "positive_anchor_min",
    "matched_anchor_null_max",
    "plus_background_max",
    "zero_H_max_abs",
    "zero_G_near_max_abs",
    "zero_G_norm_tail_max_abs",
    "matched_twin_gap",
}
_D_METRIC_KEYS = {
    "clean_D_pixel_count",
    "clean_D_delta_mean",
    "clean_D_plus_max",
    "clean_D_minus_min",
    "D_wrong_direction_pixel_count",
}
_COMMON_CHECK_KEYS = {
    "positive_anchor",
    "matched_anchor_null",
    "plus_background",
    "zero_H",
    "zero_G_near",
    "zero_G_norm_tail",
}
_D_CHECK_KEYS = {
    "clean_D_delta_mean",
    "clean_D_plus",
    "clean_D_minus",
    "D_wrong_direction",
}
_EXPECTED_GROUP_KINDS = {
    "clean_same_cell_1px": "clean_positive",
    "clean_same_cell_3px": "clean_positive",
    "clean_adjacent_cell_1px": "clean_positive",
    "clean_adjacent_cell_3px": "clean_positive",
    "clean_multicount_2to1": "clean_positive",
    "clean_multicount_3to2": "clean_positive",
    "component_null_block": "component_null",
    "component_null_sparse": "component_null",
}
_STRUCTURAL_KEYS = {
    "updates_executed",
    "expected_updates",
    "training_forward_call_count",
    "expected_training_forward_call_count",
    "training_forward_pattern_counts",
    "all_update_forward_patterns_4_4_4",
    "step_contract_failure_count",
    "gradient_failure_count",
    "finite_state_audit_count",
    "expected_finite_state_audit_count",
    "finite_state_nonfinite_element_count",
    "all_six_gradients_finite_nonzero_every_update",
    "feature_cache_grad_tensor_count",
    "feature_cache_leaves_remain_without_grad",
    "one_backward_and_one_step_per_update",
    "population_builder_reentry",
    "from_scratch_seed_42",
    "fresh_adam_state_before_first_update",
    "development_checkpoint_loaded",
    "development_optimizer_state_loaded",
    "all_pass",
}
_FINAL_FORWARD_KEYS = {
    "pair_endpoint_forward_calls",
    "pair_endpoint_states",
    "factual_miss_forward_calls",
    "factual_no_miss_forward_calls",
    "total_decoder_calls",
    "unique_pair_rows_equal_weight",
    "exposure_weighted",
    "repeated_group_forwards",
}
_OPERATOR_DIAGNOSTIC_KEYS = {
    "crossing_margin",
    "recovery_factor",
    "forward_fields_call_count",
    "forward_fields_batch_sizes",
    "field_tensor_count",
    "field_element_count",
    "field_nonfinite_element_count",
}
_FINITE_AUDIT_KEYS = {
    "phase",
    "update_index",
    "parameter_tensor_count",
    "buffer_tensor_count",
    "optimizer_state_tensor_count",
    "total_tensor_count",
    "total_element_count",
    "nonfinite_element_count",
    "nonfinite_tensor_paths",
    "maximum_absolute_value",
    "global_l2_norm",
    "all_finite",
}
_REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "method_id",
    "profile_kind",
    "profile_id",
    "decision",
    "all_pass",
    "config",
    "materialized_cache",
    "training",
    "final_evaluation",
    "result_fingerprint",
}
_FINAL_EVALUATION_KEYS = {
    "global_metrics",
    "global_checks",
    "groups",
    "numeric_gate_count",
    "structural_training_contract",
    "operator_field_diagnostics",
    "final_forward_contract",
    "all_pass",
}


class InvalidResultError(ValueError):
    """The result cannot be interpreted as a valid NLCC-v12 evidence object."""


@dataclass(frozen=True)
class GateDecision:
    """One independently recomputed gate."""

    gate_id: str
    value: int | float | bool
    operator: str
    threshold: int | float | bool
    passed: bool

    def manifest(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "value": self.value,
            "operator": self.operator,
            "threshold": self.threshold,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RecomputedDecision:
    """A valid result's decision, derived without trusting embedded decisions."""

    status: str
    all_pass: bool
    decision: str
    numeric_gate_count: int
    numeric_gates: tuple[GateDecision, ...]
    structural_gates: tuple[GateDecision, ...]

    def __post_init__(self) -> None:
        if self.status not in {PASS, GATE_FAIL}:
            raise ValueError("status must be PASS or GATE_FAIL")
        if self.all_pass != (self.status == PASS):
            raise ValueError("status and all_pass differ")

    def manifest(self) -> dict[str, object]:
        return {
            "status": self.status,
            "all_pass": self.all_pass,
            "decision": self.decision,
            "numeric_gate_count": self.numeric_gate_count,
            "numeric_gates": [gate.manifest() for gate in self.numeric_gates],
            "structural_gates": [
                gate.manifest() for gate in self.structural_gates
            ],
        }


def _reject_json_constant(value: str) -> None:
    raise InvalidResultError(f"non-standard JSON constant is forbidden: {value}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidResultError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _validate_finite_tree(value: object, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, Integral)):
        return
    if isinstance(value, Real):
        if not isfinite(float(value)):
            raise InvalidResultError(f"{path} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidResultError(f"{path} contains a non-string key")
            _validate_finite_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_tree(item, path=f"{path}[{index}]")
        return
    raise InvalidResultError(
        f"{path} contains a non-JSON value of type {type(value).__name__}"
    )


def strict_json_loads(text: str) -> object:
    """Parse strict JSON, rejecting duplicate keys and non-finite constants."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except InvalidResultError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise InvalidResultError(f"invalid JSON: {error}") from error
    _validate_finite_tree(value)
    return value


def load_strict_json_object(path: str | Path) -> dict[str, object]:
    """Load one strict, finite JSON object from ``path``."""

    value = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InvalidResultError("result JSON must contain one object")
    return value


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidResultError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise InvalidResultError(f"{path} contains a non-string key")
    return value


def _sequence(value: object, *, path: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise InvalidResultError(f"{path} must be an array")
    return value


def _require_keys(
    value: Mapping[str, object],
    required: set[str],
    *,
    path: str,
) -> None:
    missing = required - set(value)
    if missing:
        raise InvalidResultError(f"{path} is missing keys: {sorted(missing)}")


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    path: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise InvalidResultError(
            f"{path} keys differ: expected {sorted(expected)}, "
            f"got {sorted(observed)}"
        )


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidResultError(f"{path} must be boolean")
    return value


def _integer(
    value: object,
    *,
    path: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InvalidResultError(f"{path} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise InvalidResultError(f"{path} must be at least {minimum}")
    return result


def _real(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidResultError(f"{path} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise InvalidResultError(f"{path} must be finite")
    return result


def _text(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidResultError(f"{path} must be non-empty text")
    return value


def _assert_equal(observed: object, expected: object, *, path: str) -> None:
    if observed != expected:
        raise InvalidResultError(
            f"{path} differs from recomputation: "
            f"observed={observed!r}, expected={expected!r}"
        )


def _gate(
    *,
    gate_id: str,
    value: int | float | bool,
    operator: str,
    threshold: int | float | bool,
) -> GateDecision:
    if operator == "<":
        passed = value < threshold
    elif operator == ">":
        passed = value > threshold
    elif operator == "<=":
        passed = value <= threshold
    elif operator == ">=":
        passed = value >= threshold
    elif operator == "==":
        passed = value == threshold
    else:  # pragma: no cover - all callers use frozen literal operators.
        raise AssertionError(f"unknown gate operator: {operator}")
    return GateDecision(
        gate_id=gate_id,
        value=value,
        operator=operator,
        threshold=threshold,
        passed=bool(passed),
    )


def _validate_result_identity(
    result: Mapping[str, object],
    config: NLCCDatasetFreeRunnerConfig,
) -> None:
    _require_keys(result, _REQUIRED_TOP_LEVEL_KEYS, path="result")
    _assert_equal(
        result["schema_version"],
        RESULT_SCHEMA,
        path="result.schema_version",
    )
    _assert_equal(result["method_id"], config.method_id, path="result.method_id")
    _assert_equal(
        result["profile_kind"],
        config.profile.kind,
        path="result.profile_kind",
    )
    _assert_equal(
        result["profile_id"],
        config.profile.profile_id,
        path="result.profile_id",
    )
    _boolean(result["all_pass"], path="result.all_pass")
    _text(result["decision"], path="result.decision")
    config_payload = _mapping(result["config"], path="result.config")
    if stable_fingerprint(dict(config_payload)) != stable_fingerprint(
        config.manifest()
    ):
        raise InvalidResultError("result.config differs from recomputation")

    fingerprint = _text(
        result["result_fingerprint"],
        path="result.result_fingerprint",
    )
    unsigned = dict(result)
    unsigned.pop("result_fingerprint", None)
    try:
        expected_fingerprint = stable_fingerprint(unsigned)
    except (TypeError, ValueError) as error:
        raise InvalidResultError(
            f"result cannot be canonically fingerprinted: {error}"
        ) from error
    _assert_equal(
        fingerprint,
        expected_fingerprint,
        path="result.result_fingerprint",
    )


def _global_gates(
    final: Mapping[str, object],
    config: NLCCDatasetFreeRunnerConfig,
) -> list[GateDecision]:
    metrics = _mapping(final["global_metrics"], path="final.global_metrics")
    checks = _mapping(final["global_checks"], path="final.global_checks")
    _exact_keys(metrics, _GLOBAL_METRIC_KEYS, path="final.global_metrics")
    _exact_keys(checks, _GLOBAL_CHECK_KEYS, path="final.global_checks")
    for key in _GLOBAL_METRIC_KEYS:
        _real(metrics[key], path=f"final.global_metrics.{key}")

    thresholds = config.thresholds
    gates = [
        _gate(
            gate_id="global/population_total_loss",
            value=_real(
                metrics["population_total_loss"],
                path="final.global_metrics.population_total_loss",
            ),
            operator="<",
            threshold=thresholds.population_total_loss_max_exclusive,
        ),
        _gate(
            gate_id="global/factual_miss_target",
            value=_real(
                metrics["factual_miss_target_min"],
                path="final.global_metrics.factual_miss_target_min",
            ),
            operator=">",
            threshold=thresholds.factual_miss_target_min_exclusive,
        ),
        _gate(
            gate_id="global/factual_miss_background",
            value=_real(
                metrics["factual_miss_background_max"],
                path="final.global_metrics.factual_miss_background_max",
            ),
            operator="<",
            threshold=thresholds.factual_miss_background_max_exclusive,
        ),
        _gate(
            gate_id="global/factual_no_miss",
            value=_real(
                metrics["factual_no_miss_max"],
                path="final.global_metrics.factual_no_miss_max",
            ),
            operator="<",
            threshold=thresholds.factual_no_miss_max_exclusive,
        ),
    ]
    embedded = {
        "population_total_loss": gates[0].passed,
        "factual_miss_target": gates[1].passed,
        "factual_miss_background": gates[2].passed,
        "factual_no_miss": gates[3].passed,
    }
    for key, expected in embedded.items():
        observed = _boolean(checks[key], path=f"final.global_checks.{key}")
        _assert_equal(
            observed,
            expected,
            path=f"final.global_checks.{key}",
        )
    return gates


def _validate_twin_gap(
    value: object,
    *,
    group_id: str,
    row_count: int,
) -> None:
    path = f"final.groups[{group_id}].metrics.matched_twin_gap"
    payload = _mapping(value, path=path)
    _exact_keys(
        payload,
        {"matches", "minimum", "mean", "maximum", "is_gate"},
        path=path,
    )
    _assert_equal(
        _boolean(payload["is_gate"], path=f"{path}.is_gate"),
        False,
        path=f"{path}.is_gate",
    )
    matches = _sequence(payload["matches"], path=f"{path}.matches")
    if len(matches) != row_count // 2:
        raise InvalidResultError(
            f"{path}.matches length differs from row_count/2"
        )
    gaps: list[float] = []
    for index, item in enumerate(matches):
        item_path = f"{path}.matches[{index}]"
        row = _mapping(item, path=item_path)
        _exact_keys(
            row,
            {
                "match_id",
                "positive_plus_score",
                "null_plus_score",
                "gap",
            },
            path=item_path,
        )
        _text(row["match_id"], path=f"{item_path}.match_id")
        positive = _real(
            row["positive_plus_score"],
            path=f"{item_path}.positive_plus_score",
        )
        null = _real(
            row["null_plus_score"],
            path=f"{item_path}.null_plus_score",
        )
        gap = _real(row["gap"], path=f"{item_path}.gap")
        if abs(gap - (positive - null)) > 1e-7:
            raise InvalidResultError(
                f"{item_path}.gap differs from its endpoint scores"
            )
        gaps.append(gap)
    if not gaps:
        raise InvalidResultError(f"{path}.matches must be non-empty")
    recomputed = {
        "minimum": min(gaps),
        "mean": sum(gaps) / len(gaps),
        "maximum": max(gaps),
    }
    for key, expected in recomputed.items():
        observed = _real(payload[key], path=f"{path}.{key}")
        if abs(observed - expected) > 1e-7:
            raise InvalidResultError(
                f"{path}.{key} differs from matched rows"
            )


def _group_gates(
    final: Mapping[str, object],
    result: Mapping[str, object],
    config: NLCCDatasetFreeRunnerConfig,
) -> list[GateDecision]:
    groups = _sequence(final["groups"], path="final.groups")
    if len(groups) != len(_EXPECTED_GROUP_KINDS):
        raise InvalidResultError("final.groups must contain exactly eight groups")
    materialized = _mapping(
        result["materialized_cache"],
        path="result.materialized_cache",
    )
    group_rows = _mapping(
        materialized.get("group_rows"),
        path="result.materialized_cache.group_rows",
    )
    if set(group_rows) != set(_EXPECTED_GROUP_KINDS):
        raise InvalidResultError(
            "materialized_cache.group_rows does not contain the frozen groups"
        )

    thresholds = config.thresholds
    gates: list[GateDecision] = []
    seen: set[str] = set()
    for index, item in enumerate(groups):
        path = f"final.groups[{index}]"
        group = _mapping(item, path=path)
        _exact_keys(
            group,
            {
                "group_id",
                "pair_kind",
                "D_gate_status",
                "metrics",
                "checks",
                "all_pass",
            },
            path=path,
        )
        group_id = _text(group["group_id"], path=f"{path}.group_id")
        if group_id in seen or group_id not in _EXPECTED_GROUP_KINDS:
            raise InvalidResultError(f"{path}.group_id is unknown or duplicated")
        seen.add(group_id)
        pair_kind = _text(group["pair_kind"], path=f"{path}.pair_kind")
        _assert_equal(
            pair_kind,
            _EXPECTED_GROUP_KINDS[group_id],
            path=f"{path}.pair_kind",
        )
        metrics = _mapping(group["metrics"], path=f"{path}.metrics")
        checks = _mapping(group["checks"], path=f"{path}.checks")
        clean = pair_kind == "clean_positive"
        _exact_keys(
            metrics,
            _COMMON_METRIC_KEYS | _D_METRIC_KEYS,
            path=f"{path}.metrics",
        )
        _exact_keys(
            checks,
            _COMMON_CHECK_KEYS | (_D_CHECK_KEYS if clean else set()),
            path=f"{path}.checks",
        )
        row_count = _integer(
            metrics["row_count"],
            path=f"{path}.metrics.row_count",
            minimum=1,
        )
        expected_rows = _integer(
            group_rows[group_id],
            path=f"result.materialized_cache.group_rows.{group_id}",
            minimum=1,
        )
        _assert_equal(
            row_count,
            expected_rows,
            path=f"{path}.metrics.row_count",
        )
        if row_count % 2:
            raise InvalidResultError(f"{path}.metrics.row_count must be even")
        _integer(
            metrics["slot_count"],
            path=f"{path}.metrics.slot_count",
            minimum=1,
        )
        _validate_twin_gap(
            metrics["matched_twin_gap"],
            group_id=group_id,
            row_count=row_count,
        )

        group_gates = [
            _gate(
                gate_id=f"{group_id}/positive_anchor",
                value=_real(
                    metrics["positive_anchor_min"],
                    path=f"{path}.metrics.positive_anchor_min",
                ),
                operator=">",
                threshold=thresholds.positive_anchor_min_exclusive,
            ),
            _gate(
                gate_id=f"{group_id}/matched_anchor_null",
                value=_real(
                    metrics["matched_anchor_null_max"],
                    path=f"{path}.metrics.matched_anchor_null_max",
                ),
                operator="<",
                threshold=thresholds.matched_anchor_null_max_exclusive,
            ),
            _gate(
                gate_id=f"{group_id}/plus_background",
                value=_real(
                    metrics["plus_background_max"],
                    path=f"{path}.metrics.plus_background_max",
                ),
                operator="<",
                threshold=thresholds.plus_background_max_exclusive,
            ),
            _gate(
                gate_id=f"{group_id}/zero_H",
                value=_real(
                    metrics["zero_H_max_abs"],
                    path=f"{path}.metrics.zero_H_max_abs",
                ),
                operator="<=",
                threshold=thresholds.zero_H_max_abs_max_inclusive,
            ),
            _gate(
                gate_id=f"{group_id}/zero_G_near",
                value=_real(
                    metrics["zero_G_near_max_abs"],
                    path=f"{path}.metrics.zero_G_near_max_abs",
                ),
                operator="<=",
                threshold=thresholds.zero_G_near_max_abs_max_inclusive,
            ),
            _gate(
                gate_id=f"{group_id}/zero_G_norm_tail",
                value=_real(
                    metrics["zero_G_norm_tail_max_abs"],
                    path=f"{path}.metrics.zero_G_norm_tail_max_abs",
                ),
                operator="<=",
                threshold=thresholds.zero_G_norm_tail_max_abs_max_inclusive,
            ),
        ]
        expected_checks = {
            "positive_anchor": group_gates[0].passed,
            "matched_anchor_null": group_gates[1].passed,
            "plus_background": group_gates[2].passed,
            "zero_H": group_gates[3].passed,
            "zero_G_near": group_gates[4].passed,
            "zero_G_norm_tail": group_gates[5].passed,
        }
        if clean:
            _assert_equal(
                group["D_gate_status"],
                "APPLICABLE",
                path=f"{path}.D_gate_status",
            )
            _integer(
                metrics["clean_D_pixel_count"],
                path=f"{path}.metrics.clean_D_pixel_count",
                minimum=1,
            )
            D_gates = [
                _gate(
                    gate_id=f"{group_id}/clean_D_delta_mean",
                    value=_real(
                        metrics["clean_D_delta_mean"],
                        path=f"{path}.metrics.clean_D_delta_mean",
                    ),
                    operator=">=",
                    threshold=thresholds.clean_D_delta_mean_min_inclusive,
                ),
                _gate(
                    gate_id=f"{group_id}/clean_D_plus",
                    value=_real(
                        metrics["clean_D_plus_max"],
                        path=f"{path}.metrics.clean_D_plus_max",
                    ),
                    operator="<",
                    threshold=thresholds.clean_D_plus_max_exclusive,
                ),
                _gate(
                    gate_id=f"{group_id}/clean_D_minus",
                    value=_real(
                        metrics["clean_D_minus_min"],
                        path=f"{path}.metrics.clean_D_minus_min",
                    ),
                    operator=">",
                    threshold=thresholds.clean_D_minus_min_exclusive,
                ),
                _gate(
                    gate_id=f"{group_id}/D_wrong_direction",
                    value=_integer(
                        metrics["D_wrong_direction_pixel_count"],
                        path=(
                            f"{path}.metrics."
                            "D_wrong_direction_pixel_count"
                        ),
                        minimum=0,
                    ),
                    operator="<=",
                    threshold=(
                        thresholds
                        .D_wrong_direction_pixel_count_max_inclusive
                    ),
                ),
            ]
            expected_checks.update(
                {
                    "clean_D_delta_mean": D_gates[0].passed,
                    "clean_D_plus": D_gates[1].passed,
                    "clean_D_minus": D_gates[2].passed,
                    "D_wrong_direction": D_gates[3].passed,
                }
            )
            group_gates.extend(D_gates)
        else:
            _assert_equal(
                group["D_gate_status"],
                "NOT_APPLICABLE_EMPTY_D",
                path=f"{path}.D_gate_status",
            )
            _assert_equal(
                _integer(
                    metrics["clean_D_pixel_count"],
                    path=f"{path}.metrics.clean_D_pixel_count",
                    minimum=0,
                ),
                0,
                path=f"{path}.metrics.clean_D_pixel_count",
            )
            for key in (
                "clean_D_delta_mean",
                "clean_D_plus_max",
                "clean_D_minus_min",
                "D_wrong_direction_pixel_count",
            ):
                _assert_equal(
                    metrics[key],
                    None,
                    path=f"{path}.metrics.{key}",
                )

        for key, expected in expected_checks.items():
            observed = _boolean(checks[key], path=f"{path}.checks.{key}")
            _assert_equal(
                observed,
                expected,
                path=f"{path}.checks.{key}",
            )
        recomputed_group_pass = all(gate.passed for gate in group_gates)
        _assert_equal(
            _boolean(group["all_pass"], path=f"{path}.all_pass"),
            recomputed_group_pass,
            path=f"{path}.all_pass",
        )
        gates.extend(group_gates)
    if seen != set(_EXPECTED_GROUP_KINDS):
        raise InvalidResultError("final.groups does not cover the frozen groups")
    return gates


def _structural_gates(
    result: Mapping[str, object],
    final: Mapping[str, object],
    config: NLCCDatasetFreeRunnerConfig,
) -> list[GateDecision]:
    training = _mapping(result["training"], path="result.training")
    _require_keys(
        training,
        {
            "structural_contract",
            "gradient_failures",
            "step_contract_failures",
            "finite_state_audit",
        },
        path="result.training",
    )
    structural = _mapping(
        training["structural_contract"],
        path="result.training.structural_contract",
    )
    _exact_keys(
        structural,
        _STRUCTURAL_KEYS,
        path="result.training.structural_contract",
    )
    final_structural = _mapping(
        final["structural_training_contract"],
        path="final.structural_training_contract",
    )
    _assert_equal(
        dict(final_structural),
        dict(structural),
        path="final.structural_training_contract",
    )
    gradient_failures = _sequence(
        training["gradient_failures"],
        path="result.training.gradient_failures",
    )
    step_failures = _sequence(
        training["step_contract_failures"],
        path="result.training.step_contract_failures",
    )

    updates = _integer(
        structural["updates_executed"],
        path="structural.updates_executed",
        minimum=0,
    )
    expected_updates = _integer(
        structural["expected_updates"],
        path="structural.expected_updates",
        minimum=1,
    )
    forward_calls = _integer(
        structural["training_forward_call_count"],
        path="structural.training_forward_call_count",
        minimum=0,
    )
    expected_forward_calls = _integer(
        structural["expected_training_forward_call_count"],
        path="structural.expected_training_forward_call_count",
        minimum=1,
    )
    step_failure_count = _integer(
        structural["step_contract_failure_count"],
        path="structural.step_contract_failure_count",
        minimum=0,
    )
    gradient_failure_count = _integer(
        structural["gradient_failure_count"],
        path="structural.gradient_failure_count",
        minimum=0,
    )
    finite_audit_count = _integer(
        structural["finite_state_audit_count"],
        path="structural.finite_state_audit_count",
        minimum=0,
    )
    expected_finite_audit_count = _integer(
        structural["expected_finite_state_audit_count"],
        path="structural.expected_finite_state_audit_count",
        minimum=1,
    )
    finite_nonfinite_count = _integer(
        structural["finite_state_nonfinite_element_count"],
        path="structural.finite_state_nonfinite_element_count",
        minimum=0,
    )
    _assert_equal(
        step_failure_count,
        len(step_failures),
        path="structural.step_contract_failure_count",
    )
    _assert_equal(
        gradient_failure_count,
        len(gradient_failures),
        path="structural.gradient_failure_count",
    )
    pattern_counts = _mapping(
        structural["training_forward_pattern_counts"],
        path="structural.training_forward_pattern_counts",
    )
    normalized_pattern_counts = {
        _text(key, path="structural.training_forward_pattern_counts.key"):
        _integer(
            count,
            path=f"structural.training_forward_pattern_counts.{key}",
            minimum=1,
        )
        for key, count in pattern_counts.items()
    }
    if sum(normalized_pattern_counts.values()) != updates:
        raise InvalidResultError(
            "structural.training_forward_pattern_counts does not cover updates"
        )
    patterns_ok = normalized_pattern_counts == {
        "4,4,4": config.profile.updates
    }
    _assert_equal(
        _boolean(
        structural["all_update_forward_patterns_4_4_4"],
        path="structural.all_update_forward_patterns_4_4_4",
        ),
        patterns_ok,
        path="structural.all_update_forward_patterns_4_4_4",
    )
    feature_cache_grad_count = _integer(
        structural["feature_cache_grad_tensor_count"],
        path="structural.feature_cache_grad_tensor_count",
        minimum=0,
    )
    _assert_equal(
        _boolean(
            structural["feature_cache_leaves_remain_without_grad"],
            path="structural.feature_cache_leaves_remain_without_grad",
        ),
        feature_cache_grad_count == 0,
        path="structural.feature_cache_leaves_remain_without_grad",
    )
    gradients_ok = gradient_failure_count == 0
    steps_ok = step_failure_count == 0
    _assert_equal(
        _boolean(
            structural["all_six_gradients_finite_nonzero_every_update"],
            path=(
                "structural."
                "all_six_gradients_finite_nonzero_every_update"
            ),
        ),
        gradients_ok,
        path=(
            "structural."
            "all_six_gradients_finite_nonzero_every_update"
        ),
    )
    _assert_equal(
        _boolean(
            structural["one_backward_and_one_step_per_update"],
            path="structural.one_backward_and_one_step_per_update",
        ),
        steps_ok,
        path="structural.one_backward_and_one_step_per_update",
    )

    structural_gates = [
        _gate(
            gate_id="structural/updates_executed",
            value=updates,
            operator="==",
            threshold=config.profile.updates,
        ),
        _gate(
            gate_id="structural/expected_updates",
            value=expected_updates,
            operator="==",
            threshold=config.profile.updates,
        ),
        _gate(
            gate_id="structural/training_forward_call_count",
            value=forward_calls,
            operator="==",
            threshold=3 * config.profile.updates,
        ),
        _gate(
            gate_id="structural/expected_training_forward_call_count",
            value=expected_forward_calls,
            operator="==",
            threshold=3 * config.profile.updates,
        ),
        _gate(
            gate_id="structural/all_update_forward_patterns_4_4_4",
            value=patterns_ok,
            operator="==",
            threshold=True,
        ),
        _gate(
            gate_id="structural/step_contract_failure_count",
            value=step_failure_count,
            operator="==",
            threshold=0,
        ),
        _gate(
            gate_id="structural/gradient_failure_count",
            value=gradient_failure_count,
            operator="==",
            threshold=0,
        ),
        _gate(
            gate_id="structural/finite_state_audit_count",
            value=finite_audit_count,
            operator="==",
            threshold=config.profile.updates + 1,
        ),
        _gate(
            gate_id="structural/expected_finite_state_audit_count",
            value=expected_finite_audit_count,
            operator="==",
            threshold=config.profile.updates + 1,
        ),
        _gate(
            gate_id="structural/finite_state_nonfinite_element_count",
            value=finite_nonfinite_count,
            operator="==",
            threshold=0,
        ),
        _gate(
            gate_id="structural/feature_cache_grad_tensor_count",
            value=feature_cache_grad_count,
            operator="==",
            threshold=0,
        ),
        _gate(
            gate_id="structural/feature_cache_leaves_remain_without_grad",
            value=feature_cache_grad_count == 0,
            operator="==",
            threshold=True,
        ),
        _gate(
            gate_id="structural/population_builder_reentry",
            value=_boolean(
                structural["population_builder_reentry"],
                path="structural.population_builder_reentry",
            ),
            operator="==",
            threshold=False,
        ),
        _gate(
            gate_id="structural/from_scratch_seed_42",
            value=_boolean(
                structural["from_scratch_seed_42"],
                path="structural.from_scratch_seed_42",
            ),
            operator="==",
            threshold=True,
        ),
        _gate(
            gate_id="structural/fresh_adam_state_before_first_update",
            value=_boolean(
                structural["fresh_adam_state_before_first_update"],
                path="structural.fresh_adam_state_before_first_update",
            ),
            operator="==",
            threshold=True,
        ),
        _gate(
            gate_id="structural/development_checkpoint_loaded",
            value=_boolean(
                structural["development_checkpoint_loaded"],
                path="structural.development_checkpoint_loaded",
            ),
            operator="==",
            threshold=False,
        ),
        _gate(
            gate_id="structural/development_optimizer_state_loaded",
            value=_boolean(
                structural["development_optimizer_state_loaded"],
                path="structural.development_optimizer_state_loaded",
            ),
            operator="==",
            threshold=False,
        ),
    ]

    forward = _mapping(
        final["final_forward_contract"],
        path="final.final_forward_contract",
    )
    _exact_keys(
        forward,
        _FINAL_FORWARD_KEYS,
        path="final.final_forward_contract",
    )
    materialized = _mapping(
        result["materialized_cache"],
        path="result.materialized_cache",
    )
    pair_rows = _integer(
        materialized.get("pair_rows"),
        path="result.materialized_cache.pair_rows",
        minimum=1,
    )
    forward_expectations: tuple[tuple[str, int | bool], ...] = (
        ("pair_endpoint_forward_calls", 1),
        ("pair_endpoint_states", 2 * pair_rows),
        ("factual_miss_forward_calls", 1),
        ("factual_no_miss_forward_calls", 1),
        ("total_decoder_calls", 3),
        ("unique_pair_rows_equal_weight", True),
        ("exposure_weighted", False),
        ("repeated_group_forwards", False),
    )
    for key, expected in forward_expectations:
        observed: int | bool
        if isinstance(expected, bool):
            observed = _boolean(
                forward[key],
                path=f"final.final_forward_contract.{key}",
            )
        else:
            observed = _integer(
                forward[key],
                path=f"final.final_forward_contract.{key}",
                minimum=0,
            )
        structural_gates.append(
            _gate(
                gate_id=f"structural/final_forward/{key}",
                value=observed,
                operator="==",
                threshold=expected,
            )
        )

    _validate_finite_state_audit(
        training["finite_state_audit"],
        config=config,
        structural_count=finite_audit_count,
        structural_expected_count=expected_finite_audit_count,
        structural_nonfinite_count=finite_nonfinite_count,
    )
    recomputed_structural_pass = all(
        gate.passed for gate in structural_gates
    )
    _assert_equal(
        _boolean(structural["all_pass"], path="structural.all_pass"),
        recomputed_structural_pass,
        path="structural.all_pass",
    )
    return structural_gates


def _validate_one_finite_audit(
    value: object,
    *,
    path: str,
    expected_phase: str,
    expected_update_index: int,
) -> None:
    audit = _mapping(value, path=path)
    _exact_keys(audit, _FINITE_AUDIT_KEYS, path=path)
    _assert_equal(audit["phase"], expected_phase, path=f"{path}.phase")
    _assert_equal(
        _integer(audit["update_index"], path=f"{path}.update_index"),
        expected_update_index,
        path=f"{path}.update_index",
    )
    parameter_count = _integer(
        audit["parameter_tensor_count"],
        path=f"{path}.parameter_tensor_count",
        minimum=1,
    )
    buffer_count = _integer(
        audit["buffer_tensor_count"],
        path=f"{path}.buffer_tensor_count",
        minimum=0,
    )
    optimizer_count = _integer(
        audit["optimizer_state_tensor_count"],
        path=f"{path}.optimizer_state_tensor_count",
        minimum=0,
    )
    total_count = _integer(
        audit["total_tensor_count"],
        path=f"{path}.total_tensor_count",
        minimum=1,
    )
    _assert_equal(
        total_count,
        parameter_count + buffer_count + optimizer_count,
        path=f"{path}.total_tensor_count",
    )
    _integer(
        audit["total_element_count"],
        path=f"{path}.total_element_count",
        minimum=1,
    )
    nonfinite = _integer(
        audit["nonfinite_element_count"],
        path=f"{path}.nonfinite_element_count",
        minimum=0,
    )
    paths = _sequence(
        audit["nonfinite_tensor_paths"],
        path=f"{path}.nonfinite_tensor_paths",
    )
    if any(not isinstance(item, str) or not item for item in paths):
        raise InvalidResultError(
            f"{path}.nonfinite_tensor_paths must contain non-empty text"
        )
    if (nonfinite == 0) != (len(paths) == 0):
        raise InvalidResultError(
            f"{path}.nonfinite paths disagree with element count"
        )
    for field in ("maximum_absolute_value", "global_l2_norm"):
        if _real(audit[field], path=f"{path}.{field}") < 0.0:
            raise InvalidResultError(f"{path}.{field} must be nonnegative")
    _assert_equal(
        _boolean(audit["all_finite"], path=f"{path}.all_finite"),
        nonfinite == 0,
        path=f"{path}.all_finite",
    )


def _validate_finite_state_audit(
    value: object,
    *,
    config: NLCCDatasetFreeRunnerConfig,
    structural_count: int,
    structural_expected_count: int,
    structural_nonfinite_count: int,
) -> None:
    path = "result.training.finite_state_audit"
    audit = _mapping(value, path=path)
    _exact_keys(
        audit,
        {
            "call_count",
            "expected_call_count",
            "initial",
            "final",
            "nonfinite_element_count",
        },
        path=path,
    )
    call_count = _integer(
        audit["call_count"], path=f"{path}.call_count", minimum=0
    )
    expected = _integer(
        audit["expected_call_count"],
        path=f"{path}.expected_call_count",
        minimum=1,
    )
    nonfinite = _integer(
        audit["nonfinite_element_count"],
        path=f"{path}.nonfinite_element_count",
        minimum=0,
    )
    _assert_equal(call_count, structural_count, path=f"{path}.call_count")
    _assert_equal(
        expected,
        structural_expected_count,
        path=f"{path}.expected_call_count",
    )
    _assert_equal(
        expected,
        config.profile.updates + 1,
        path=f"{path}.expected_call_count",
    )
    _assert_equal(
        nonfinite,
        structural_nonfinite_count,
        path=f"{path}.nonfinite_element_count",
    )
    _validate_one_finite_audit(
        audit["initial"],
        path=f"{path}.initial",
        expected_phase="before_first_update",
        expected_update_index=-1,
    )
    _validate_one_finite_audit(
        audit["final"],
        path=f"{path}.final",
        expected_phase="after_optimizer_step",
        expected_update_index=config.profile.updates - 1,
    )


def _validate_operator_diagnostics(
    final: Mapping[str, object],
    result: Mapping[str, object],
) -> None:
    diagnostics = _mapping(
        final["operator_field_diagnostics"],
        path="final.operator_field_diagnostics",
    )
    _exact_keys(
        diagnostics,
        _OPERATOR_DIAGNOSTIC_KEYS,
        path="final.operator_field_diagnostics",
    )
    for field in ("crossing_margin", "recovery_factor"):
        bounds = _mapping(
            diagnostics[field],
            path=f"final.operator_field_diagnostics.{field}",
        )
        _exact_keys(
            bounds,
            {"minimum", "maximum"},
            path=f"final.operator_field_diagnostics.{field}",
        )
        minimum = _real(
            bounds["minimum"],
            path=f"final.operator_field_diagnostics.{field}.minimum",
        )
        maximum = _real(
            bounds["maximum"],
            path=f"final.operator_field_diagnostics.{field}.maximum",
        )
        if minimum > maximum:
            raise InvalidResultError(
                f"final.operator_field_diagnostics.{field} range is reversed"
            )
    _assert_equal(
        _integer(
            diagnostics["forward_fields_call_count"],
            path="final.operator_field_diagnostics.forward_fields_call_count",
            minimum=0,
        ),
        3,
        path="final.operator_field_diagnostics.forward_fields_call_count",
    )
    materialized = _mapping(
        result["materialized_cache"],
        path="result.materialized_cache",
    )
    pair_rows = _integer(
        materialized.get("pair_rows"),
        path="result.materialized_cache.pair_rows",
        minimum=1,
    )
    factual_rows = _mapping(
        materialized.get("factual_rows_per_branch"),
        path="result.materialized_cache.factual_rows_per_branch",
    )
    _exact_keys(
        factual_rows,
        {"factual_miss", "factual_no_miss"},
        path="result.materialized_cache.factual_rows_per_branch",
    )
    batch_sizes = [
        _integer(
            item,
            path=(
                "final.operator_field_diagnostics."
                f"forward_fields_batch_sizes[{index}]"
            ),
            minimum=1,
        )
        for index, item in enumerate(
            _sequence(
                diagnostics["forward_fields_batch_sizes"],
                path=(
                    "final.operator_field_diagnostics."
                    "forward_fields_batch_sizes"
                ),
            )
        )
    ]
    expected_batches = [
        2 * pair_rows,
        _integer(
            factual_rows["factual_miss"],
            path=(
                "result.materialized_cache."
                "factual_rows_per_branch.factual_miss"
            ),
            minimum=1,
        ),
        _integer(
            factual_rows["factual_no_miss"],
            path=(
                "result.materialized_cache."
                "factual_rows_per_branch.factual_no_miss"
            ),
            minimum=1,
        ),
    ]
    _assert_equal(
        batch_sizes,
        expected_batches,
        path=(
            "final.operator_field_diagnostics."
            "forward_fields_batch_sizes"
        ),
    )
    _assert_equal(
        _integer(
            diagnostics["field_tensor_count"],
            path="final.operator_field_diagnostics.field_tensor_count",
            minimum=1,
        ),
        45,
        path="final.operator_field_diagnostics.field_tensor_count",
    )
    _integer(
        diagnostics["field_element_count"],
        path="final.operator_field_diagnostics.field_element_count",
        minimum=1,
    )
    _assert_equal(
        _integer(
            diagnostics["field_nonfinite_element_count"],
            path=(
                "final.operator_field_diagnostics."
                "field_nonfinite_element_count"
            ),
            minimum=0,
        ),
        0,
        path=(
            "final.operator_field_diagnostics."
            "field_nonfinite_element_count"
        ),
    )


def recompute_result_decision(
    result: Mapping[str, object],
    config: NLCCDatasetFreeRunnerConfig,
) -> RecomputedDecision:
    """Validate and independently recompute one NLCC-v12 result decision.

    A well-formed scientific failure returns ``GATE_FAIL``.  Malformed
    evidence, non-finite values, or disagreement between embedded and
    recomputed derived fields raises :class:`InvalidResultError`.
    """

    if not isinstance(config, NLCCDatasetFreeRunnerConfig):
        raise TypeError("config must be NLCCDatasetFreeRunnerConfig")
    result_mapping = _mapping(result, path="result")
    _validate_finite_tree(result_mapping)
    _validate_result_identity(result_mapping, config)
    final = _mapping(
        result_mapping["final_evaluation"],
        path="result.final_evaluation",
    )
    _exact_keys(final, _FINAL_EVALUATION_KEYS, path="result.final_evaluation")

    numeric_gates = _global_gates(final, config)
    numeric_gates.extend(_group_gates(final, result_mapping, config))
    if len(numeric_gates) != 76:
        raise InvalidResultError(
            f"recomputed numeric gate count is {len(numeric_gates)}, not 76"
        )
    _assert_equal(
        _integer(
            final["numeric_gate_count"],
            path="final.numeric_gate_count",
            minimum=0,
        ),
        76,
        path="final.numeric_gate_count",
    )
    structural_gates = _structural_gates(
        result_mapping,
        final,
        config,
    )
    _validate_operator_diagnostics(final, result_mapping)

    all_pass = all(gate.passed for gate in numeric_gates) and all(
        gate.passed for gate in structural_gates
    )
    _assert_equal(
        _boolean(final["all_pass"], path="final.all_pass"),
        all_pass,
        path="final.all_pass",
    )
    _assert_equal(
        _boolean(result_mapping["all_pass"], path="result.all_pass"),
        all_pass,
        path="result.all_pass",
    )
    prefix = (
        "NLCC_V12_DEVELOPMENT"
        if config.profile.kind == DEVELOPMENT
        else "NLCC_V12_HOLDOUT"
    )
    decision = f"{prefix}_{'PASS' if all_pass else 'FAIL'}"
    _assert_equal(
        _text(result_mapping["decision"], path="result.decision"),
        decision,
        path="result.decision",
    )
    return RecomputedDecision(
        status=PASS if all_pass else GATE_FAIL,
        all_pass=all_pass,
        decision=decision,
        numeric_gate_count=len(numeric_gates),
        numeric_gates=tuple(numeric_gates),
        structural_gates=tuple(structural_gates),
    )


__all__ = [
    "GATE_FAIL",
    "InvalidResultError",
    "PASS",
    "GateDecision",
    "RecomputedDecision",
    "load_strict_json_object",
    "recompute_result_decision",
    "strict_json_loads",
]
