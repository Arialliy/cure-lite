"""Frozen, per-seed decisions for formal paired CURE-Lite waves.

This module consumes already selected D_V results.  It does not train,
calibrate, access a dataset, or average seeds.  Its only role is to apply the
predeclared performance and false-addition gates without discretionary
post-result interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Iterable

from ..cache.schema import stable_fingerprint


FORMAL_SEEDS = (42, 43)
PROPOSED_METHOD = "paired_difference"
HISTORICAL_COMPARATORS = ("Base@B", "F", "F×", "U")
FORMAL_WAVES = {
    "A": ("independent_endpoint",),
    "B": (
        "after_only",
        "target_permutation",
        "plus_detach",
        "minus_detach",
    ),
    "C": ("zero_feature", "coordinate_basis", "feature_only"),
}
_WAVE_ORDER = ("A", "B", "C")
_HEX = frozenset("0123456789abcdef")


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _finite(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


@dataclass(frozen=True)
class FormalMethodEvidence:
    """One method's frozen D_V result for one decoder seed."""

    method: str
    seed: int
    total_targets: int
    true_targets: int
    pd: float
    total_anchor_misses: int
    recovered_anchor_misses: int
    retention: float
    pixel_fa: float
    raw_background_fa: float
    fp_components_per_mp: float
    budget_violation: bool
    comparison_protocol_fingerprint: str
    result_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty string")
        if self.seed not in FORMAL_SEEDS:
            raise ValueError(f"seed must be one of {FORMAL_SEEDS}")
        for name in (
            "total_targets",
            "true_targets",
            "total_anchor_misses",
            "recovered_anchor_misses",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.total_targets < 1 or self.true_targets > self.total_targets:
            raise ValueError("target counts are inconsistent")
        if (
            self.total_anchor_misses < 1
            or self.recovered_anchor_misses > self.total_anchor_misses
        ):
            raise ValueError("anchor-miss counts are inconsistent")
        for name in (
            "pd",
            "retention",
            "pixel_fa",
            "raw_background_fa",
            "fp_components_per_mp",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name=name),
            )
        if self.pd > 1.0 or self.retention > 1.0:
            raise ValueError("pd and retention must lie in [0,1]")
        expected_pd = self.true_targets / self.total_targets
        if not isclose(self.pd, expected_pd, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("pd is not the exact true-target fraction")
        if not isinstance(self.budget_violation, bool):
            raise TypeError("budget_violation must be bool")
        _digest(
            self.comparison_protocol_fingerprint,
            name="comparison_protocol_fingerprint",
        )
        _digest(self.result_fingerprint, name="result_fingerprint")

    def constraints(self, *, proposed: bool) -> dict[str, bool]:
        checks = {
            "calibration_retention_ge_0_99": self.retention >= 0.99,
            "pixel_fa_le_1e-4": self.pixel_fa <= 1e-4,
            "raw_background_fa_le_1e-4": self.raw_background_fa <= 1e-4,
            "fp_components_per_mp_le_100": self.fp_components_per_mp <= 100.0,
            "budget_violation_false": self.budget_violation is False,
        }
        if proposed:
            checks["proposed_retention_equal_1"] = isclose(
                self.retention,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        return checks

    def canonical_payload(self) -> dict[str, object]:
        return {
            "method": self.method,
            "seed": self.seed,
            "total_targets": self.total_targets,
            "true_targets": self.true_targets,
            "pd": self.pd,
            "total_anchor_misses": self.total_anchor_misses,
            "recovered_anchor_misses": self.recovered_anchor_misses,
            "retention": self.retention,
            "pixel_fa": self.pixel_fa,
            "raw_background_fa": self.raw_background_fa,
            "fp_components_per_mp": self.fp_components_per_mp,
            "budget_violation": self.budget_violation,
            "comparison_protocol_fingerprint": (
                self.comparison_protocol_fingerprint
            ),
            "result_fingerprint": self.result_fingerprint,
        }


def expected_methods_for_wave(wave: str) -> tuple[str, ...]:
    """Return the exact cumulative evidence set for a formal reveal."""

    if wave not in _WAVE_ORDER:
        raise ValueError(f"wave must be one of {_WAVE_ORDER}")
    methods = [PROPOSED_METHOD, *HISTORICAL_COMPARATORS]
    for current in _WAVE_ORDER:
        methods.extend(FORMAL_WAVES[current])
        if current == wave:
            break
    return tuple(methods)


def assess_formal_wave(
    evidence: Iterable[FormalMethodEvidence],
    *,
    wave: str,
    protocol_fingerprint: str,
    comparison_protocol_fingerprint: str,
) -> dict[str, object]:
    """Apply the exact two-seed margin and constraint gate for one wave."""

    protocol_fingerprint = _digest(
        protocol_fingerprint,
        name="protocol_fingerprint",
    )
    comparison_protocol_fingerprint = _digest(
        comparison_protocol_fingerprint,
        name="comparison_protocol_fingerprint",
    )
    expected_methods = expected_methods_for_wave(wave)
    rows = tuple(evidence)
    if any(not isinstance(row, FormalMethodEvidence) for row in rows):
        raise TypeError("evidence must contain FormalMethodEvidence values")
    keys = tuple((row.seed, row.method) for row in rows)
    expected_keys = tuple(
        (seed, method)
        for seed in FORMAL_SEEDS
        for method in expected_methods
    )
    if len(set(keys)) != len(keys) or set(keys) != set(expected_keys):
        raise ValueError(
            "formal wave evidence must contain every expected method exactly "
            "once for both seeds and no extra methods"
        )
    comparison_fingerprints = {
        row.comparison_protocol_fingerprint for row in rows
    }
    if comparison_fingerprints != {comparison_protocol_fingerprint}:
        raise ValueError(
            "all formal evidence must bind the declared common comparison "
            "protocol fingerprint"
        )
    by_key = {(row.seed, row.method): row for row in rows}

    seed_decisions: list[dict[str, object]] = []
    for seed in FORMAL_SEEDS:
        proposed = by_key[(seed, PROPOSED_METHOD)]
        comparators = tuple(
            by_key[(seed, method)]
            for method in expected_methods
            if method != PROPOSED_METHOD
        )
        total_targets = {row.total_targets for row in (proposed, *comparators)}
        total_misses = {
            row.total_anchor_misses for row in (proposed, *comparators)
        }
        if len(total_targets) != 1 or len(total_misses) != 1:
            raise ValueError(
                f"seed {seed} methods do not share target/miss populations"
            )
        constraint_checks = {
            row.method: row.constraints(
                proposed=row.method == PROPOSED_METHOD
            )
            for row in (proposed, *comparators)
        }
        all_constraints = all(
            passed
            for checks in constraint_checks.values()
            for passed in checks.values()
        )
        best_tp = max(row.true_targets for row in comparators)
        best_recovered = max(
            row.recovered_anchor_misses for row in comparators
        )
        best_tp_methods = sorted(
            row.method for row in comparators if row.true_targets == best_tp
        )
        best_recovered_methods = sorted(
            row.method
            for row in comparators
            if row.recovered_anchor_misses == best_recovered
        )
        tp_margin = proposed.true_targets - best_tp
        recovered_margin = (
            proposed.recovered_anchor_misses - best_recovered
        )
        checks = {
            "all_methods_satisfy_constraints": all_constraints,
            "proposed_true_targets_margin_ge_2": tp_margin >= 2,
            "proposed_recovered_anchor_misses_margin_ge_2": (
                recovered_margin >= 2
            ),
        }
        seed_decisions.append(
            {
                "seed": seed,
                "pass": all(checks.values()),
                "checks": checks,
                "constraint_checks": constraint_checks,
                "proposed": proposed.canonical_payload(),
                "best_comparator_true_targets": best_tp,
                "best_comparator_true_target_methods": best_tp_methods,
                "true_target_margin": tp_margin,
                "best_comparator_recovered_anchor_misses": best_recovered,
                "best_comparator_recovered_methods": best_recovered_methods,
                "recovered_anchor_miss_margin": recovered_margin,
            }
        )

    passed = all(row["pass"] is True for row in seed_decisions)
    wave_index = _WAVE_ORDER.index(wave)
    final_wave = wave == _WAVE_ORDER[-1]
    if not passed:
        status = "PERFORMANCE_FAIL"
        next_action = "STOP_AND_PRESERVE_EVIDENCE"
    elif final_wave:
        status = "FORMAL_MATCHED_CONTROL_GATE_PASS"
        next_action = "FROZEN_CONFIRMATION_ONLY"
    else:
        status = "FORMAL_WAVE_PASS"
        next_action = f"RUN_PRE_FROZEN_WAVE_{_WAVE_ORDER[wave_index + 1]}"

    canonical_evidence = [
        by_key[key].canonical_payload() for key in expected_keys
    ]
    payload = {
        "schema_version": "cure-lite-paired-formal-wave-decision-v1",
        "protocol_fingerprint": protocol_fingerprint,
        "comparison_protocol_fingerprint": (
            comparison_protocol_fingerprint
        ),
        "wave": wave,
        "development_seeds": list(FORMAL_SEEDS),
        "per_seed_not_mean": True,
        "minimum_true_target_margin": 2,
        "minimum_recovered_anchor_miss_margin": 2,
        "expected_methods": list(expected_methods),
        "evidence": canonical_evidence,
        "seed_decisions": seed_decisions,
        "all_seeds_pass": passed,
        "status": status,
        "next_action": next_action,
        "D_T_accessed": False,
        "authorizes_full_cure": False,
        "authorizes_cross_backbone": False,
        "authorizes_only_frozen_confirmation": passed and final_wave,
    }
    return {**payload, "decision_fingerprint": stable_fingerprint(payload)}


__all__ = [
    "FORMAL_SEEDS",
    "FORMAL_WAVES",
    "HISTORICAL_COMPARATORS",
    "PROPOSED_METHOD",
    "FormalMethodEvidence",
    "assess_formal_wave",
    "expected_methods_for_wave",
]
