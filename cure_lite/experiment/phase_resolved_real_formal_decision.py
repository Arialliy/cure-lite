"""Frozen per-seed performance gate for formal PFCR CURE-Lite.

This module consumes only already sealed ``FormalMethodEvidence`` rows.  It
does not open a dataset, select a threshold, train a model, or average seeds.
The proposed PFCR result must beat the strongest existing method separately
for both true targets and recovered anchor misses on each seed.
"""

from __future__ import annotations

from math import isclose
from typing import Iterable

from ..cache.schema import stable_fingerprint
from .paired_formal_decision import FormalMethodEvidence


PFCR_FORMAL_SEEDS = (42, 43)
PFCR_PROPOSED_METHOD = "PFCR"
PFCR_FORMAL_COMPARATORS = (
    "Base@B",
    "F",
    "F×",
    "U",
    "paired_difference",
    "independent_endpoint",
)
PFCR_MINIMUM_TRUE_TARGET_MARGIN = 2
PFCR_MINIMUM_RECOVERED_MISS_MARGIN = 2
PFCR_FORMAL_DECISION_SCHEMA = (
    "cure-lite-pfcr-formal-d-v-decision-v1"
)
_HEX = frozenset("0123456789abcdef")


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _constraint_checks(
    row: FormalMethodEvidence,
    *,
    proposed: bool,
) -> dict[str, bool]:
    checks = {
        "retention_ge_0_99": row.retention >= 0.99,
        "pixel_fa_le_1e-4": row.pixel_fa <= 1.0e-4,
        "raw_background_fa_le_1e-4": (
            row.raw_background_fa <= 1.0e-4
        ),
        "fp_components_per_mp_le_100": (
            row.fp_components_per_mp <= 100.0
        ),
        "budget_violation_false": row.budget_violation is False,
    }
    if proposed:
        checks["pfcr_retention_equal_1"] = isclose(
            row.retention,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    return checks


def assess_pfcr_formal_d_v_gate(
    evidence: Iterable[FormalMethodEvidence],
    *,
    protocol_fingerprint: str,
    comparison_protocol_fingerprint: str,
) -> dict[str, object]:
    """Apply the frozen two-seed PFCR gate without seed averaging."""

    protocol_fingerprint = _digest(
        protocol_fingerprint,
        name="protocol_fingerprint",
    )
    comparison_protocol_fingerprint = _digest(
        comparison_protocol_fingerprint,
        name="comparison_protocol_fingerprint",
    )
    rows = tuple(evidence)
    if any(not isinstance(row, FormalMethodEvidence) for row in rows):
        raise TypeError(
            "evidence must contain FormalMethodEvidence values"
        )
    expected_methods = (
        PFCR_PROPOSED_METHOD,
        *PFCR_FORMAL_COMPARATORS,
    )
    expected_keys = tuple(
        (seed, method)
        for seed in PFCR_FORMAL_SEEDS
        for method in expected_methods
    )
    keys = tuple((row.seed, row.method) for row in rows)
    if len(set(keys)) != len(keys) or set(keys) != set(expected_keys):
        raise ValueError(
            "PFCR formal evidence must contain the proposed method and "
            "every frozen comparator exactly once for both seeds"
        )
    if {
        row.comparison_protocol_fingerprint for row in rows
    } != {comparison_protocol_fingerprint}:
        raise ValueError(
            "all evidence must bind the frozen common comparison protocol"
        )
    by_key = {(row.seed, row.method): row for row in rows}

    seed_decisions: list[dict[str, object]] = []
    for seed in PFCR_FORMAL_SEEDS:
        proposed = by_key[(seed, PFCR_PROPOSED_METHOD)]
        comparators = tuple(
            by_key[(seed, method)]
            for method in PFCR_FORMAL_COMPARATORS
        )
        population = (proposed, *comparators)
        if (
            {row.total_targets for row in population} != {170}
            or {row.total_anchor_misses for row in population} != {23}
        ):
            raise ValueError(
                f"seed {seed} evidence does not bind 170 targets/23 misses"
            )
        for row in population:
            retained_scaled = row.retention * 147
            retained_count = int(round(retained_scaled))
            if (
                not isclose(
                    retained_scaled,
                    retained_count,
                    rel_tol=0.0,
                    abs_tol=1.0e-10,
                )
                or not 0 <= retained_count <= 147
                or row.true_targets
                != retained_count + row.recovered_anchor_misses
            ):
                raise ValueError(
                    f"seed {seed} method {row.method} violates the "
                    "true-target = retained-covered + recovered-miss "
                    "integer identity"
                )
        constraint_checks = {
            row.method: _constraint_checks(
                row,
                proposed=row.method == PFCR_PROPOSED_METHOD,
            )
            for row in population
        }
        all_constraints = all(
            passed
            for checks in constraint_checks.values()
            for passed in checks.values()
        )
        best_true_targets = max(
            row.true_targets for row in comparators
        )
        best_recovered = max(
            row.recovered_anchor_misses for row in comparators
        )
        true_target_margin = (
            proposed.true_targets - best_true_targets
        )
        recovered_margin = (
            proposed.recovered_anchor_misses - best_recovered
        )
        checks = {
            "all_budget_and_retention_constraints_pass": (
                all_constraints
            ),
            "pfcr_true_target_margin_ge_2": (
                true_target_margin
                >= PFCR_MINIMUM_TRUE_TARGET_MARGIN
            ),
            "pfcr_recovered_anchor_miss_margin_ge_2": (
                recovered_margin
                >= PFCR_MINIMUM_RECOVERED_MISS_MARGIN
            ),
        }
        seed_decisions.append(
            {
                "seed": seed,
                "pass": all(checks.values()),
                "checks": checks,
                "constraint_checks": constraint_checks,
                "pfcr": proposed.canonical_payload(),
                "best_comparator_true_targets": best_true_targets,
                "best_comparator_true_target_methods": sorted(
                    row.method
                    for row in comparators
                    if row.true_targets == best_true_targets
                ),
                "true_target_margin": true_target_margin,
                "best_comparator_recovered_anchor_misses": (
                    best_recovered
                ),
                "best_comparator_recovered_methods": sorted(
                    row.method
                    for row in comparators
                    if row.recovered_anchor_misses == best_recovered
                ),
                "recovered_anchor_miss_margin": recovered_margin,
            }
        )

    all_seeds_pass = all(
        decision["pass"] is True for decision in seed_decisions
    )
    if all_seeds_pass:
        status = "PFCR_D_V_GATE_PASS"
        next_action = "FROZEN_CONFIRMATION_ONLY"
    else:
        status = "PFCR_D_V_GATE_FAIL"
        next_action = "STOP_AND_PRESERVE_EVIDENCE"
    canonical_evidence = [
        by_key[key].canonical_payload() for key in expected_keys
    ]
    payload: dict[str, object] = {
        "schema_version": PFCR_FORMAL_DECISION_SCHEMA,
        "protocol_fingerprint": protocol_fingerprint,
        "comparison_protocol_fingerprint": (
            comparison_protocol_fingerprint
        ),
        "development_seeds": list(PFCR_FORMAL_SEEDS),
        "per_seed_not_mean": True,
        "proposed_method": PFCR_PROPOSED_METHOD,
        "comparators": list(PFCR_FORMAL_COMPARATORS),
        "minimum_true_target_margin": (
            PFCR_MINIMUM_TRUE_TARGET_MARGIN
        ),
        "minimum_recovered_anchor_miss_margin": (
            PFCR_MINIMUM_RECOVERED_MISS_MARGIN
        ),
        "evidence": canonical_evidence,
        "seed_decisions": seed_decisions,
        "all_seeds_pass": all_seeds_pass,
        "status": status,
        "next_action": next_action,
        "D_T_accessed": False,
        "authorizes_frozen_confirmation": all_seeds_pass,
        "authorizes_full_cure": False,
        "authorizes_cross_backbone": False,
    }
    return {
        **payload,
        "decision_fingerprint": stable_fingerprint(payload),
    }


__all__ = [
    "PFCR_FORMAL_COMPARATORS",
    "PFCR_FORMAL_DECISION_SCHEMA",
    "PFCR_FORMAL_SEEDS",
    "PFCR_MINIMUM_RECOVERED_MISS_MARGIN",
    "PFCR_MINIMUM_TRUE_TARGET_MARGIN",
    "PFCR_PROPOSED_METHOD",
    "assess_pfcr_formal_d_v_gate",
]
