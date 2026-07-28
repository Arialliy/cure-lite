"""Predeclared bounded-400 decision for the v20 BFA-CMIF field.

The v20 decision is intentionally different from the historical aggregate
``bounded_gate_passed`` flag.  It asks whether the binary-flip
antisymmetrization improves the frozen v18 target--outside trade-off while
preserving the four non-negotiable zero-level invariants.

Every count is reconstructed from the immutable per-state diagnostics.  No
threshold is searched and neither ``D_V`` nor ``D_T`` is consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from ..cache.schema import stable_fingerprint
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationResult,
)


COVERAGE_STATE_BFA_DECISION_SCHEMA = (
    "cure-lite-bfa-cmif-v20-bounded-zero-level-decision-v1"
)
COVERAGE_STATE_BFA_ROLE_COUNT = 16
COVERAGE_STATE_BFA_CLEAN_TARGET_PIXELS = 149
COVERAGE_STATE_BFA_FACTUAL_STRICT_FLOOR = 11
COVERAGE_STATE_BFA_CLEAN_TARGET_FLOOR = 123
COVERAGE_STATE_BFA_CLEAN_OUTSIDE_CEILING = 47
COVERAGE_STATE_BFA_COMPONENT_NULL_FLOOR = 15


@dataclass(frozen=True)
class CoverageStateBFABoundedDecision:
    """Frozen v20 structural-advancement decision and its raw counts."""

    diagnostic: CoverageStateZeroLevelEvaluationResult
    checks: tuple[tuple[str, bool], ...]
    factual_miss_count: int
    factual_strict_count: int
    factual_recovered_count: int
    factual_no_miss_count: int
    factual_no_miss_passed_count: int
    clean_pair_count: int
    clean_target_pixels: int
    clean_target_negative_pixels: int
    clean_outside_completion_pixels: int
    clean_compact_support_passed_count: int
    component_null_count: int
    component_null_passed_count: int
    identity_null_count: int
    identity_null_passed_count: int
    diagnostic_null_count: int
    diagnostic_null_passed_count: int
    invalid_completion_pixels: int
    response_sign_pixels: int
    response_sign_correct_pixels: int
    response_sign_all_correct_pair_count: int

    def __post_init__(self) -> None:
        if not isinstance(
            self.diagnostic,
            CoverageStateZeroLevelEvaluationResult,
        ):
            raise TypeError("diagnostic must be a zero-level result")
        if (
            self.checks != tuple(sorted(self.checks))
            or len({name for name, _ in self.checks}) != len(self.checks)
            or any(
                not isinstance(name, str) or not isinstance(value, bool)
                for name, value in self.checks
            )
        ):
            raise ValueError("BFA decision checks are malformed")
        integers = (
            self.factual_miss_count,
            self.factual_strict_count,
            self.factual_recovered_count,
            self.factual_no_miss_count,
            self.factual_no_miss_passed_count,
            self.clean_pair_count,
            self.clean_target_pixels,
            self.clean_target_negative_pixels,
            self.clean_outside_completion_pixels,
            self.clean_compact_support_passed_count,
            self.component_null_count,
            self.component_null_passed_count,
            self.identity_null_count,
            self.identity_null_passed_count,
            self.diagnostic_null_count,
            self.diagnostic_null_passed_count,
            self.invalid_completion_pixels,
            self.response_sign_pixels,
            self.response_sign_correct_pixels,
            self.response_sign_all_correct_pair_count,
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in integers
            )
            or self.factual_strict_count > self.factual_miss_count
            or self.factual_recovered_count > self.factual_miss_count
            or self.factual_no_miss_passed_count
            > self.factual_no_miss_count
            or self.clean_target_negative_pixels
            > self.clean_target_pixels
            or self.clean_compact_support_passed_count
            > self.clean_pair_count
            or self.component_null_passed_count
            > self.component_null_count
            or self.identity_null_passed_count
            > self.identity_null_count
            or self.diagnostic_null_passed_count
            > self.diagnostic_null_count
            or self.response_sign_correct_pixels
            > self.response_sign_pixels
            or self.response_sign_all_correct_pair_count
            > self.clean_pair_count
        ):
            raise ValueError("BFA decision counts are inconsistent")

    @property
    def bounded_gate_passed(self) -> bool:
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def formal800_eligible(self) -> bool:
        return self.bounded_gate_passed

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks if not passed)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_BFA_DECISION_SCHEMA,
            "diagnostic_result_fingerprint": (
                self.diagnostic.result_fingerprint
            ),
            "frozen_reference": {
                "version": "v18_pmope_bounded_400",
                "factual_strict_floor_exclusive": (
                    COVERAGE_STATE_BFA_FACTUAL_STRICT_FLOOR
                ),
                "clean_target_negative_floor_exclusive": (
                    COVERAGE_STATE_BFA_CLEAN_TARGET_FLOOR
                ),
                "clean_outside_completion_ceiling_exclusive": (
                    COVERAGE_STATE_BFA_CLEAN_OUTSIDE_CEILING
                ),
                "component_null_floor_inclusive": (
                    COVERAGE_STATE_BFA_COMPONENT_NULL_FLOOR
                ),
            },
            "population": {
                "factual_miss": self.factual_miss_count,
                "factual_no_miss": self.factual_no_miss_count,
                "clean_positive": self.clean_pair_count,
                "clean_target_pixels": self.clean_target_pixels,
                "component_null": self.component_null_count,
                "identity_null": self.identity_null_count,
                "diagnostic_null": self.diagnostic_null_count,
            },
            "observed": {
                "factual_strict": self.factual_strict_count,
                "factual_recovered": self.factual_recovered_count,
                "factual_no_miss_passed": (
                    self.factual_no_miss_passed_count
                ),
                "clean_target_negative_pixels": (
                    self.clean_target_negative_pixels
                ),
                "clean_outside_completion_pixels": (
                    self.clean_outside_completion_pixels
                ),
                "clean_compact_support_passed": (
                    self.clean_compact_support_passed_count
                ),
                "component_null_passed": (
                    self.component_null_passed_count
                ),
                "identity_null_passed": (
                    self.identity_null_passed_count
                ),
                "diagnostic_null_passed": (
                    self.diagnostic_null_passed_count
                ),
                "invalid_completion_pixels": (
                    self.invalid_completion_pixels
                ),
            },
            "same_sign_response_diagnostic": {
                "pixels": self.response_sign_pixels,
                "correct_pixels": self.response_sign_correct_pixels,
                "all_correct_pairs": (
                    self.response_sign_all_correct_pair_count
                ),
                "is_gate": False,
            },
            "gates": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "bounded_gate_passed": self.bounded_gate_passed,
            "formal800_eligible": self.formal800_eligible,
            "threshold_search_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }

    @cached_property
    def decision_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def decide_coverage_state_bfa_bounded(
    diagnostic: CoverageStateZeroLevelEvaluationResult,
) -> CoverageStateBFABoundedDecision:
    """Apply the exact predeclared v20 bounded-400 inequalities."""

    if not isinstance(
        diagnostic,
        CoverageStateZeroLevelEvaluationResult,
    ):
        raise TypeError("diagnostic must be a zero-level result")

    factual_miss = tuple(
        value
        for value in diagnostic.natural_diagnostics
        if value.state_kind == "factual_miss"
    )
    factual_no_miss = tuple(
        value
        for value in diagnostic.natural_diagnostics
        if value.state_kind == "factual_no_miss"
    )
    clean = tuple(
        value
        for value in diagnostic.pair_diagnostics
        if value.pair_kind == "clean_positive"
        and value.optimizer_role == "clean_positive"
    )
    component = tuple(
        value
        for value in diagnostic.pair_diagnostics
        if value.pair_kind == "component_null"
        and value.optimizer_role == "component_null"
    )
    identity = tuple(
        value
        for value in diagnostic.pair_diagnostics
        if value.pair_kind == "identity_null"
    )
    diagnostic_null = tuple(
        value
        for value in diagnostic.pair_diagnostics
        if value.optimizer_role == "diagnostic_only"
    )

    factual_strict = sum(value.gate_passed for value in factual_miss)
    factual_recovered = sum(
        value.target_recovered is True for value in factual_miss
    )
    no_miss_passed = sum(
        value.gate_passed for value in factual_no_miss
    )
    clean_target_pixels = sum(value.added_target_pixels for value in clean)
    clean_target_negative = sum(
        value.minus_added_target_negative_pixels for value in clean
    )
    clean_outside = sum(
        value.new_completion_outside_added_target_pixels or 0
        for value in clean
    )
    clean_compact = sum(
        value.compact_support_passed is True for value in clean
    )
    component_passed = sum(value.gate_passed for value in component)
    identity_passed = sum(value.gate_passed for value in identity)
    diagnostic_passed = sum(
        value.gate_passed for value in diagnostic_null
    )
    invalid = sum(
        value.invalid_completion_pixels
        for value in diagnostic.natural_diagnostics
    ) + sum(
        value.invalid_completion_pixels_plus
        + value.invalid_completion_pixels_minus
        for value in diagnostic.pair_diagnostics
    )
    response_pixels = sum(value.response_sign_pixels for value in clean)
    response_correct = sum(
        value.response_sign_correct_pixels for value in clean
    )
    response_all = sum(
        value.response_sign_all_correct is True for value in clean
    )

    population_fixed = (
        len(diagnostic.natural_diagnostics)
        == 2 * COVERAGE_STATE_BFA_ROLE_COUNT
        and len(diagnostic.pair_diagnostics)
        == 3 * COVERAGE_STATE_BFA_ROLE_COUNT + 1
        and len(factual_miss) == COVERAGE_STATE_BFA_ROLE_COUNT
        and len(factual_no_miss) == COVERAGE_STATE_BFA_ROLE_COUNT
        and len(clean) == COVERAGE_STATE_BFA_ROLE_COUNT
        and all(
            value.minus_added_target_all_negative is not None
            and value.new_completion_outside_added_target_pixels
            is not None
            and value.compact_support_passed is not None
            for value in clean
        )
        and clean_target_pixels == COVERAGE_STATE_BFA_CLEAN_TARGET_PIXELS
        and len(component) == COVERAGE_STATE_BFA_ROLE_COUNT
        and len(identity) == COVERAGE_STATE_BFA_ROLE_COUNT
        and len(diagnostic_null) == 1
        and diagnostic_null[0].pair_kind == "component_null"
        and len(
            {
                value.record_id
                for value in diagnostic.natural_diagnostics
            }
        )
        == len(diagnostic.natural_diagnostics)
        and len(
            {
                value.pair_id
                for value in diagnostic.pair_diagnostics
            }
        )
        == len(diagnostic.pair_diagnostics)
    )
    checks = tuple(
        sorted(
            {
                "fixed_D_R_population": population_fixed,
                "factual_no_miss_16_of_16": (
                    len(factual_no_miss) == COVERAGE_STATE_BFA_ROLE_COUNT
                    and no_miss_passed == COVERAGE_STATE_BFA_ROLE_COUNT
                ),
                "identity_null_16_of_16": (
                    len(identity) == COVERAGE_STATE_BFA_ROLE_COUNT
                    and identity_passed == COVERAGE_STATE_BFA_ROLE_COUNT
                ),
                "diagnostic_null_pass": (
                    len(diagnostic_null) == 1
                    and diagnostic_passed == 1
                ),
                "invalid_completion_zero": invalid == 0,
                "factual_strict_gt_11_of_16": (
                    len(factual_miss) == COVERAGE_STATE_BFA_ROLE_COUNT
                    and factual_strict
                    > COVERAGE_STATE_BFA_FACTUAL_STRICT_FLOOR
                ),
                "factual_recovered_16_of_16": (
                    len(factual_miss) == COVERAGE_STATE_BFA_ROLE_COUNT
                    and factual_recovered == COVERAGE_STATE_BFA_ROLE_COUNT
                ),
                "clean_target_negative_gt_123_of_149": (
                    clean_target_pixels
                    == COVERAGE_STATE_BFA_CLEAN_TARGET_PIXELS
                    and clean_target_negative
                    > COVERAGE_STATE_BFA_CLEAN_TARGET_FLOOR
                ),
                "clean_outside_completion_lt_47": (
                    len(clean) == COVERAGE_STATE_BFA_ROLE_COUNT
                    and clean_outside
                    < COVERAGE_STATE_BFA_CLEAN_OUTSIDE_CEILING
                ),
                "clean_compact_support_gt_0_of_16": (
                    len(clean) == COVERAGE_STATE_BFA_ROLE_COUNT
                    and clean_compact > 0
                ),
                "component_null_ge_15_of_16": (
                    len(component) == COVERAGE_STATE_BFA_ROLE_COUNT
                    and component_passed
                    >= COVERAGE_STATE_BFA_COMPONENT_NULL_FLOOR
                ),
                "threshold_zero_without_search": (
                    diagnostic.config.residual_threshold == 0.0
                    and not diagnostic.config.threshold_search_performed
                    and diagnostic.config.input_representation
                    == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                ),
                "evaluation_read_only_D_R_only": (
                    diagnostic.split == "D_R"
                    and diagnostic.backward_calls == 0
                    and diagnostic.optimizer_steps == 0
                    and not diagnostic.config.d_v_accessed
                    and not diagnostic.config.d_t_accessed
                ),
            }.items()
        )
    )
    return CoverageStateBFABoundedDecision(
        diagnostic=diagnostic,
        checks=checks,
        factual_miss_count=len(factual_miss),
        factual_strict_count=factual_strict,
        factual_recovered_count=factual_recovered,
        factual_no_miss_count=len(factual_no_miss),
        factual_no_miss_passed_count=no_miss_passed,
        clean_pair_count=len(clean),
        clean_target_pixels=clean_target_pixels,
        clean_target_negative_pixels=clean_target_negative,
        clean_outside_completion_pixels=clean_outside,
        clean_compact_support_passed_count=clean_compact,
        component_null_count=len(component),
        component_null_passed_count=component_passed,
        identity_null_count=len(identity),
        identity_null_passed_count=identity_passed,
        diagnostic_null_count=len(diagnostic_null),
        diagnostic_null_passed_count=diagnostic_passed,
        invalid_completion_pixels=invalid,
        response_sign_pixels=response_pixels,
        response_sign_correct_pixels=response_correct,
        response_sign_all_correct_pair_count=response_all,
    )


__all__ = [
    "COVERAGE_STATE_BFA_CLEAN_OUTSIDE_CEILING",
    "COVERAGE_STATE_BFA_CLEAN_TARGET_FLOOR",
    "COVERAGE_STATE_BFA_CLEAN_TARGET_PIXELS",
    "COVERAGE_STATE_BFA_COMPONENT_NULL_FLOOR",
    "COVERAGE_STATE_BFA_DECISION_SCHEMA",
    "COVERAGE_STATE_BFA_FACTUAL_STRICT_FLOOR",
    "COVERAGE_STATE_BFA_ROLE_COUNT",
    "CoverageStateBFABoundedDecision",
    "decide_coverage_state_bfa_bounded",
]
