"""Bounded-400 structural decision for CURE-Lite v22 PACRE.

The numerical inequalities are deliberately inherited from the frozen v21
bounded protocol.  This module does not reuse a v21 model, training entry, or
model label: it reuses only the already declared, model-independent decision
function over a generic zero-level ``D_R`` diagnostic.  The resulting receipt
is relabelled and rebound to PACRE so that a v22 result cannot be mistaken for
historical PAET evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.coverage_state_paet_decision import (
    COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    COVERAGE_STATE_PAET_DECISION_SCHEMA,
    CoverageStatePAETBoundedDecision,
    decide_coverage_state_paet_bounded,
)
from cure_lite.experiment.coverage_state_zero_level_evaluation import (
    CoverageStateZeroLevelEvaluationResult,
)

from .pacre import (
    CSLF_PACRE_CENTERING_POLICY,
    CSLF_PACRE_EQUATION_POLICY,
    CSLF_PACRE_FIELD_POLICY,
)


PACRE_BOUNDED_RUN_ID = (
    "cure_lite_pacre_v22_pmope_bounded_400_seed42_r1"
)
PACRE_BOUNDED_DECISION_SCHEMA = (
    "cure-lite-pacre-v22-pmope-bounded-400-decision-v1"
)
PACRE_REFERENCE_DECISION_POLICY = (
    "exact_v21_predeclared_zero_level_inequalities_applied_to_"
    "pacre_diagnostic_v1"
)


def _sorted_int_items(
    values: dict[str, int],
) -> tuple[tuple[str, int], ...]:
    if any(
        not isinstance(name, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for name, value in values.items()
    ):
        raise ValueError("decision counts must be nonnegative integers")
    return tuple(sorted(values.items()))


@dataclass(frozen=True, eq=False)
class CoverageStatePACREBoundedDecision:
    """PACRE-labelled result of the frozen structural inequalities."""

    run_id: str
    diagnostic: CoverageStateZeroLevelEvaluationResult
    reference_decision_fingerprint: str
    checks: tuple[tuple[str, bool], ...]
    population: tuple[tuple[str, int], ...]
    observed: tuple[tuple[str, int], ...]
    response_sign_diagnostic: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.run_id != PACRE_BOUNDED_RUN_ID:
            raise ValueError("PACRE bounded run_id changed")
        if not isinstance(
            self.diagnostic,
            CoverageStateZeroLevelEvaluationResult,
        ):
            raise TypeError("diagnostic must be a zero-level result")
        if (
            not isinstance(self.reference_decision_fingerprint, str)
            or len(self.reference_decision_fingerprint) != 64
        ):
            raise ValueError("reference decision fingerprint is malformed")
        if (
            self.checks != tuple(sorted(self.checks))
            or len({name for name, _ in self.checks})
            != len(self.checks)
            or any(
                not isinstance(name, str)
                or not isinstance(value, bool)
                for name, value in self.checks
            )
        ):
            raise ValueError("PACRE decision checks are malformed")
        for name, values in (
            ("population", self.population),
            ("observed", self.observed),
            ("response", self.response_sign_diagnostic),
        ):
            if (
                values != tuple(sorted(values))
                or len({key for key, _ in values}) != len(values)
                or any(
                    not isinstance(key, str)
                    or isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for key, value in values
                )
            ):
                raise ValueError(f"PACRE {name} counts are malformed")

    @property
    def bounded_gate_passed(self) -> bool:
        return bool(self.checks) and all(
            value for _, value in self.checks
        )

    @property
    def formal800_eligible(self) -> bool:
        return self.bounded_gate_passed

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(
            name for name, passed in self.checks if not passed
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PACRE_BOUNDED_DECISION_SCHEMA,
            "run_id": self.run_id,
            "candidate": "PACRE-v22",
            "field_policy": CSLF_PACRE_FIELD_POLICY,
            "equation_policy": CSLF_PACRE_EQUATION_POLICY,
            "centering_policy": CSLF_PACRE_CENTERING_POLICY,
            "diagnostic_result_fingerprint": (
                self.diagnostic.result_fingerprint
            ),
            "reference_inequality_policy": {
                "policy": PACRE_REFERENCE_DECISION_POLICY,
                "source_schema": COVERAGE_STATE_PAET_DECISION_SCHEMA,
                "source_run_id": COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
                "source_decision_fingerprint": (
                    self.reference_decision_fingerprint
                ),
                "historical_model_reused": False,
                "historical_training_reused": False,
                "only_numeric_inequalities_reused": True,
            },
            "population": dict(self.population),
            "observed": dict(self.observed),
            "same_sign_response_diagnostic": {
                **dict(self.response_sign_diagnostic),
                "is_gate": False,
            },
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "bounded_gate_passed": self.bounded_gate_passed,
            "formal800_eligible": self.formal800_eligible,
            "formal_800_authorized": False,
            "threshold": 0.0,
            "threshold_search_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
        }

    @cached_property
    def decision_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _population(reference: CoverageStatePAETBoundedDecision) -> dict[str, int]:
    return {
        "factual_miss": reference.factual_miss_count,
        "factual_target_pixels": reference.factual_target_pixels,
        "factual_no_miss": reference.factual_no_miss_count,
        "clean_positive": reference.clean_pair_count,
        "clean_target_pixels": reference.clean_target_pixels,
        "component_null": reference.component_null_count,
        "identity_null": reference.identity_null_count,
        "diagnostic_null": reference.diagnostic_null_count,
    }


def _observed(reference: CoverageStatePAETBoundedDecision) -> dict[str, int]:
    return {
        "factual_target_negative_pixels": (
            reference.factual_target_negative_pixels
        ),
        "factual_strict": reference.factual_strict_count,
        "factual_recovered": reference.factual_recovered_count,
        "factual_no_miss_passed": (
            reference.factual_no_miss_passed_count
        ),
        "clean_target_negative_pixels": (
            reference.clean_target_negative_pixels
        ),
        "clean_outside_completion_pixels": (
            reference.clean_outside_completion_pixels
        ),
        "clean_compact_support_passed": (
            reference.clean_compact_support_passed_count
        ),
        "component_null_passed": (
            reference.component_null_passed_count
        ),
        "identity_null_passed": (
            reference.identity_null_passed_count
        ),
        "diagnostic_null_passed": (
            reference.diagnostic_null_passed_count
        ),
        "invalid_completion_pixels": (
            reference.invalid_completion_pixels
        ),
    }


def decide_coverage_state_pacre_bounded(
    diagnostic: CoverageStateZeroLevelEvaluationResult,
    *,
    run_id: str,
) -> CoverageStatePACREBoundedDecision:
    """Apply the frozen bounded inequalities to a PACRE checkpoint."""

    if run_id != PACRE_BOUNDED_RUN_ID:
        raise PermissionError("PACRE bounded decision run_id is not frozen")
    if not isinstance(
        diagnostic,
        CoverageStateZeroLevelEvaluationResult,
    ):
        raise TypeError("diagnostic must be a zero-level result")
    reference = decide_coverage_state_paet_bounded(
        diagnostic,
        run_id=COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    )
    return CoverageStatePACREBoundedDecision(
        run_id=run_id,
        diagnostic=diagnostic,
        reference_decision_fingerprint=reference.decision_fingerprint,
        checks=reference.checks,
        population=_sorted_int_items(_population(reference)),
        observed=_sorted_int_items(_observed(reference)),
        response_sign_diagnostic=_sorted_int_items(
            {
                "pixels": reference.response_sign_pixels,
                "correct_pixels": (
                    reference.response_sign_correct_pixels
                ),
                "all_correct_pairs": (
                    reference.response_sign_all_correct_pair_count
                ),
            }
        ),
    )


__all__ = [
    "PACRE_BOUNDED_DECISION_SCHEMA",
    "PACRE_BOUNDED_RUN_ID",
    "PACRE_REFERENCE_DECISION_POLICY",
    "CoverageStatePACREBoundedDecision",
    "decide_coverage_state_pacre_bounded",
]
