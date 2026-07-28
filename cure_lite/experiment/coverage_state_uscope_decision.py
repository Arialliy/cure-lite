"""USCOPE interpretation of the frozen zero-level diagnostics.

The shared evaluator continues to report the historical continuous
same-sign response ordering.  USCOPE treats that quantity as a diagnostic,
not as a binary-completion gate: when both endpoint target fields have the
same sign, their continuous ordering does not change the zero-level output.
All detection-facing natural, compact-support, and null gates remain
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from ..cache.schema import stable_fingerprint
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationResult,
)


COVERAGE_STATE_USCOPE_DECISION_SCHEMA = (
    "cure-lite-cmif-v19-uscope-zero-level-decision-v1"
)


@dataclass(frozen=True)
class CoverageStateUSCOPEZeroLevelDecision:
    """Detection-facing bounded decision plus response-only diagnostics."""

    diagnostic: CoverageStateZeroLevelEvaluationResult
    checks: tuple[tuple[str, bool], ...]
    clean_pair_count: int
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
            raise ValueError("USCOPE decision checks are malformed")
        integers = (
            self.clean_pair_count,
            self.response_sign_pixels,
            self.response_sign_correct_pixels,
            self.response_sign_all_correct_pair_count,
        )
        if (
            any(isinstance(value, bool) or not isinstance(value, int)
                for value in integers)
            or any(value < 0 for value in integers)
            or self.response_sign_correct_pixels
            > self.response_sign_pixels
            or self.response_sign_all_correct_pair_count
            > self.clean_pair_count
        ):
            raise ValueError("USCOPE response diagnostics are invalid")

    @property
    def zero_level_gate_passed(self) -> bool:
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def bounded_gate_passed(self) -> bool:
        """Compatibility alias; the bounded runner must also gate gamma."""

        return self.zero_level_gate_passed

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks if not passed)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_USCOPE_DECISION_SCHEMA,
            "diagnostic_result_fingerprint": (
                self.diagnostic.result_fingerprint
            ),
            "gates": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "zero_level_gate_passed": self.zero_level_gate_passed,
            "bounded_gate_passed": self.bounded_gate_passed,
            "same_sign_response_policy": (
                "legacy_all_response_ordering_is_diagnostic_not_binary_gate"
            ),
            "response_diagnostic": {
                "clean_pair_count": self.clean_pair_count,
                "response_sign_pixels": self.response_sign_pixels,
                "response_sign_correct_pixels": (
                    self.response_sign_correct_pixels
                ),
                "response_sign_all_correct_pair_count": (
                    self.response_sign_all_correct_pair_count
                ),
                "available": (
                    self.clean_pair_count > 0
                    and self.response_sign_pixels > 0
                ),
            },
            "threshold_search_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }

    @cached_property
    def decision_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def decide_coverage_state_uscope_zero_level(
    diagnostic: CoverageStateZeroLevelEvaluationResult,
) -> CoverageStateUSCOPEZeroLevelDecision:
    """Apply the predeclared v19 binary-completion gate."""

    if not isinstance(
        diagnostic,
        CoverageStateZeroLevelEvaluationResult,
    ):
        raise TypeError("diagnostic must be a zero-level result")
    clean = tuple(
        value
        for value in diagnostic.pair_diagnostics
        if value.pair_kind == "clean_positive"
    )
    response_pixels = sum(value.response_sign_pixels for value in clean)
    response_correct = sum(
        value.response_sign_correct_pixels for value in clean
    )
    response_all_pairs = sum(
        value.response_sign_all_correct is True for value in clean
    )
    checks = tuple(
        sorted(
            {
                "factual_miss": diagnostic.factual_miss_gate_passed,
                "factual_no_miss": (
                    diagnostic.factual_no_miss_gate_passed
                ),
                "clean_compact_support": (
                    diagnostic.clean_compact_support_gate_passed
                ),
                "component_null": (
                    diagnostic.component_null_gate_passed
                ),
                "identity_null": diagnostic.identity_null_gate_passed,
                "diagnostic_null": (
                    diagnostic.diagnostic_null_gate_passed
                ),
                "threshold_zero_without_search": (
                    diagnostic.config.residual_threshold == 0.0
                    and not diagnostic.config.threshold_search_performed
                    and diagnostic.config.input_representation
                    == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                ),
                "evaluation_read_only": (
                    diagnostic.backward_calls == 0
                    and diagnostic.optimizer_steps == 0
                    and not diagnostic.config.d_v_accessed
                    and not diagnostic.config.d_t_accessed
                ),
            }.items()
        )
    )
    return CoverageStateUSCOPEZeroLevelDecision(
        diagnostic=diagnostic,
        checks=checks,
        clean_pair_count=len(clean),
        response_sign_pixels=response_pixels,
        response_sign_correct_pixels=response_correct,
        response_sign_all_correct_pair_count=response_all_pairs,
    )


__all__ = [
    "COVERAGE_STATE_USCOPE_DECISION_SCHEMA",
    "CoverageStateUSCOPEZeroLevelDecision",
    "decide_coverage_state_uscope_zero_level",
]
