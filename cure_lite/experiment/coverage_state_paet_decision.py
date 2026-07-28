"""Predeclared bounded-400 decision for the v21 PAET-BFA field.

The decision is reconstructed only from the fixed zero-threshold ``D_R``
diagnostics.  It preserves every v20 invariant and requires both spatial
allocation improvements at the same time.  No threshold is searched and
neither ``D_V`` nor ``D_T`` is consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from ..cache.schema import stable_fingerprint
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationResult,
)


COVERAGE_STATE_PAET_DECISION_SCHEMA = (
    "cure-lite-paet-bfa-v21-bounded-zero-level-decision-v1"
)
COVERAGE_STATE_PAET_BOUNDED_RUN_ID = (
    "cure_lite_paet_bfa_v21_pmope_bounded_400_r1"
)
COVERAGE_STATE_PAET_ROLE_COUNT = 16
COVERAGE_STATE_PAET_FACTUAL_TARGET_PIXELS = 335
COVERAGE_STATE_PAET_CLEAN_TARGET_PIXELS = 149
COVERAGE_STATE_PAET_FACTUAL_RECOVERED_REQUIRED = 16
COVERAGE_STATE_PAET_FACTUAL_STRICT_MINIMUM = 14
COVERAGE_STATE_PAET_FACTUAL_NEGATIVE_MINIMUM = 310
COVERAGE_STATE_PAET_CLEAN_TARGET_NEGATIVE_MINIMUM = 124
COVERAGE_STATE_PAET_CLEAN_OUTSIDE_COMPLETION_MAXIMUM = 46
COVERAGE_STATE_PAET_COMPACT_SUPPORT_MINIMUM = 1
COVERAGE_STATE_PAET_COMPONENT_NULL_REQUIRED = 16


@dataclass(frozen=True)
class CoverageStatePAETBoundedDecision:
    """Frozen v21 structural-advancement decision and its raw counts."""

    run_id: str
    diagnostic: CoverageStateZeroLevelEvaluationResult
    checks: tuple[tuple[str, bool], ...]
    factual_miss_count: int
    factual_target_pixels: int
    factual_target_negative_pixels: int
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
        if self.run_id != COVERAGE_STATE_PAET_BOUNDED_RUN_ID:
            raise ValueError("PAET decision run_id changed")
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
            raise ValueError("PAET decision checks are malformed")
        integers = (
            self.factual_miss_count,
            self.factual_target_pixels,
            self.factual_target_negative_pixels,
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
            or self.factual_target_negative_pixels
            > self.factual_target_pixels
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
            raise ValueError("PAET decision counts are inconsistent")

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
            "schema_version": COVERAGE_STATE_PAET_DECISION_SCHEMA,
            "run_id": self.run_id,
            "diagnostic_result_fingerprint": (
                self.diagnostic.result_fingerprint
            ),
            "frozen_reference": {
                "version": "v20_bfa_cmif_pmope_bounded_400_r2",
                "v20_complete_fingerprint": (
                    "8908a8c1896951e46fd737aa6f7fef2c9935e6524632b3576b"
                    "8069faa026e2eb"
                ),
                "factual_recovered_required": (
                    COVERAGE_STATE_PAET_FACTUAL_RECOVERED_REQUIRED
                ),
                "factual_strict_minimum_inclusive": (
                    COVERAGE_STATE_PAET_FACTUAL_STRICT_MINIMUM
                ),
                "factual_target_negative_minimum_inclusive": (
                    COVERAGE_STATE_PAET_FACTUAL_NEGATIVE_MINIMUM
                ),
                "clean_target_negative_minimum_inclusive": (
                    COVERAGE_STATE_PAET_CLEAN_TARGET_NEGATIVE_MINIMUM
                ),
                "clean_outside_completion_maximum_inclusive": (
                    COVERAGE_STATE_PAET_CLEAN_OUTSIDE_COMPLETION_MAXIMUM
                ),
                "compact_support_minimum_inclusive": (
                    COVERAGE_STATE_PAET_COMPACT_SUPPORT_MINIMUM
                ),
                "component_null_required": (
                    COVERAGE_STATE_PAET_COMPONENT_NULL_REQUIRED
                ),
            },
            "population": {
                "factual_miss": self.factual_miss_count,
                "factual_target_pixels": self.factual_target_pixels,
                "factual_no_miss": self.factual_no_miss_count,
                "clean_positive": self.clean_pair_count,
                "clean_target_pixels": self.clean_target_pixels,
                "component_null": self.component_null_count,
                "identity_null": self.identity_null_count,
                "diagnostic_null": self.diagnostic_null_count,
            },
            "observed": {
                "factual_target_negative_pixels": (
                    self.factual_target_negative_pixels
                ),
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
            "formal800_eligible_by_performance": (
                self.formal800_eligible
            ),
            "formal_800_authorized": False,
            "threshold_search_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }

    @cached_property
    def decision_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def decide_coverage_state_paet_bounded(
    diagnostic: CoverageStateZeroLevelEvaluationResult,
    *,
    run_id: str,
) -> CoverageStatePAETBoundedDecision:
    """Apply the exact predeclared v21 bounded-400 inequalities."""

    if run_id != COVERAGE_STATE_PAET_BOUNDED_RUN_ID:
        raise PermissionError("PAET decision run_id is not frozen")
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

    factual_target_pixels = sum(
        value.focus_target_pixels for value in factual_miss
    )
    factual_target_negative = sum(
        value.focus_target_negative_pixels for value in factual_miss
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
        == 2 * COVERAGE_STATE_PAET_ROLE_COUNT
        and len(diagnostic.pair_diagnostics)
        == 3 * COVERAGE_STATE_PAET_ROLE_COUNT + 1
        and len(factual_miss) == COVERAGE_STATE_PAET_ROLE_COUNT
        and len(factual_no_miss) == COVERAGE_STATE_PAET_ROLE_COUNT
        and len(clean) == COVERAGE_STATE_PAET_ROLE_COUNT
        and all(
            value.target_recovered is not None
            for value in factual_miss
        )
        and all(
            value.minus_added_target_all_negative is not None
            and value.new_completion_outside_added_target_pixels
            is not None
            and value.compact_support_passed is not None
            for value in clean
        )
        and factual_target_pixels
        == COVERAGE_STATE_PAET_FACTUAL_TARGET_PIXELS
        and clean_target_pixels == COVERAGE_STATE_PAET_CLEAN_TARGET_PIXELS
        and len(component) == COVERAGE_STATE_PAET_ROLE_COUNT
        and len(identity) == COVERAGE_STATE_PAET_ROLE_COUNT
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
                    len(factual_no_miss)
                    == COVERAGE_STATE_PAET_ROLE_COUNT
                    and no_miss_passed
                    == COVERAGE_STATE_PAET_ROLE_COUNT
                ),
                "identity_null_16_of_16": (
                    len(identity) == COVERAGE_STATE_PAET_ROLE_COUNT
                    and identity_passed == COVERAGE_STATE_PAET_ROLE_COUNT
                ),
                "diagnostic_null_1_of_1": (
                    len(diagnostic_null) == 1
                    and diagnostic_passed == 1
                ),
                "invalid_completion_zero": invalid == 0,
                "factual_recovered_16_of_16": (
                    len(factual_miss) == COVERAGE_STATE_PAET_ROLE_COUNT
                    and factual_recovered
                    == COVERAGE_STATE_PAET_FACTUAL_RECOVERED_REQUIRED
                ),
                "factual_strict_ge_14_of_16": (
                    len(factual_miss) == COVERAGE_STATE_PAET_ROLE_COUNT
                    and factual_strict
                    >= COVERAGE_STATE_PAET_FACTUAL_STRICT_MINIMUM
                ),
                "factual_target_negative_ge_310_of_335": (
                    factual_target_pixels
                    == COVERAGE_STATE_PAET_FACTUAL_TARGET_PIXELS
                    and factual_target_negative
                    >= COVERAGE_STATE_PAET_FACTUAL_NEGATIVE_MINIMUM
                ),
                "clean_target_negative_ge_124_of_149": (
                    clean_target_pixels
                    == COVERAGE_STATE_PAET_CLEAN_TARGET_PIXELS
                    and clean_target_negative
                    >= COVERAGE_STATE_PAET_CLEAN_TARGET_NEGATIVE_MINIMUM
                ),
                "clean_outside_completion_le_46": (
                    len(clean) == COVERAGE_STATE_PAET_ROLE_COUNT
                    and clean_outside
                    <= COVERAGE_STATE_PAET_CLEAN_OUTSIDE_COMPLETION_MAXIMUM
                ),
                "clean_compact_support_ge_1_of_16": (
                    len(clean) == COVERAGE_STATE_PAET_ROLE_COUNT
                    and clean_compact
                    >= COVERAGE_STATE_PAET_COMPACT_SUPPORT_MINIMUM
                ),
                "component_null_16_of_16": (
                    len(component) == COVERAGE_STATE_PAET_ROLE_COUNT
                    and component_passed
                    == COVERAGE_STATE_PAET_COMPONENT_NULL_REQUIRED
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
    return CoverageStatePAETBoundedDecision(
        run_id=run_id,
        diagnostic=diagnostic,
        checks=checks,
        factual_miss_count=len(factual_miss),
        factual_target_pixels=factual_target_pixels,
        factual_target_negative_pixels=factual_target_negative,
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
    "COVERAGE_STATE_PAET_BOUNDED_RUN_ID",
    "COVERAGE_STATE_PAET_CLEAN_OUTSIDE_COMPLETION_MAXIMUM",
    "COVERAGE_STATE_PAET_CLEAN_TARGET_NEGATIVE_MINIMUM",
    "COVERAGE_STATE_PAET_CLEAN_TARGET_PIXELS",
    "COVERAGE_STATE_PAET_COMPACT_SUPPORT_MINIMUM",
    "COVERAGE_STATE_PAET_COMPONENT_NULL_REQUIRED",
    "COVERAGE_STATE_PAET_DECISION_SCHEMA",
    "COVERAGE_STATE_PAET_FACTUAL_NEGATIVE_MINIMUM",
    "COVERAGE_STATE_PAET_FACTUAL_RECOVERED_REQUIRED",
    "COVERAGE_STATE_PAET_FACTUAL_STRICT_MINIMUM",
    "COVERAGE_STATE_PAET_FACTUAL_TARGET_PIXELS",
    "COVERAGE_STATE_PAET_ROLE_COUNT",
    "CoverageStatePAETBoundedDecision",
    "decide_coverage_state_paet_bounded",
]
