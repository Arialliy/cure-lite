"""Frozen seed-42 D_V advancement gate for formal PAET-BFA v21.

This is a development gate only.  A pass authorizes one frozen D_T
confirmation; it does not establish final CURE-Lite success and does not
authorize Full CURE or cross-backbone experiments.
"""

from __future__ import annotations

from math import isclose, isfinite

from ..cache.schema import stable_fingerprint
from ..metrics import AggregateEvaluation
from .coverage_state_paet_formal_evaluation import (
    PAET_FORMAL_METHOD,
    PAET_FORMAL_SEED,
    PAETFormalDVEvaluationResult,
)
from .paired_formal_evaluation import (
    FORMAL_DV_ANCHOR_COVERED,
    FORMAL_DV_ANCHOR_MISSES,
    FORMAL_DV_IMAGES,
    FORMAL_DV_TOTAL_TARGETS,
)


PAET_FORMAL_DV_DECISION_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-decision-v1"
)
PAET_MINIMUM_TRUE_TARGET_MARGIN = 2
PAET_MINIMUM_RECOVERED_MISS_MARGIN = 2
PAET_MAXIMUM_PIXEL_FA = 1.0e-4
PAET_MAXIMUM_RAW_BACKGROUND_FA = 1.0e-4
PAET_MAXIMUM_FP_COMPONENTS_PER_MP = 100.0
_HEX = frozenset("0123456789abcdef")


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _validate_metrics(
    metrics: AggregateEvaluation,
    *,
    name: str,
) -> tuple[int, int]:
    if not isinstance(metrics, AggregateEvaluation):
        raise TypeError(f"{name} must be AggregateEvaluation")
    if (
        metrics.images != FORMAL_DV_IMAGES
        or metrics.total_anchor_misses != FORMAL_DV_ANCHOR_MISSES
        or metrics.total_anchor_covered != FORMAL_DV_ANCHOR_COVERED
        or (
            metrics.total_anchor_misses
            + metrics.total_anchor_covered
        )
        != FORMAL_DV_TOTAL_TARGETS
    ):
        raise ValueError(
            f"{name} does not bind the frozen 120/170/23/147 population"
        )
    if not (
        0
        <= metrics.recovered_anchor_misses
        <= metrics.total_anchor_misses
        and 0
        <= metrics.retained_anchor_covered
        <= metrics.total_anchor_covered
    ):
        raise ValueError(f"{name} target counts are inconsistent")
    true_targets = (
        metrics.retained_anchor_covered
        + metrics.recovered_anchor_misses
    )
    if not isclose(
        metrics.pd,
        true_targets / FORMAL_DV_TOTAL_TARGETS,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{name} Pd is inconsistent with target counts")
    if not isclose(
        metrics.retention,
        metrics.retained_anchor_covered
        / FORMAL_DV_ANCHOR_COVERED,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"{name} retention is inconsistent with covered-target counts"
        )
    unit_values = (
        metrics.pd,
        metrics.rmr,
        metrics.gross_rmr,
        metrics.retention,
        metrics.reachable_rmr,
        metrics.oracle_upper_bound,
        metrics.overlap_supported_rmr,
        metrics.miou,
        metrics.niou,
    )
    if any(
        not isfinite(value) or not 0.0 <= value <= 1.0
        for value in unit_values
    ):
        raise ValueError(f"{name} unit metrics must be finite and in [0,1]")
    false_addition_values = (
        metrics.pixel_fa,
        metrics.raw_background_fa,
        metrics.fp_components_per_mp,
    )
    if any(
        not isfinite(value) or value < 0.0
        for value in false_addition_values
    ):
        raise ValueError(
            f"{name} false-addition metrics must be finite/nonnegative"
        )
    if not isinstance(metrics.budget_violation, bool):
        raise TypeError(f"{name} budget_violation must be bool")
    return true_targets, metrics.recovered_anchor_misses


def _metrics_summary(
    metrics: AggregateEvaluation,
) -> dict[str, object]:
    true_targets = (
        metrics.retained_anchor_covered
        + metrics.recovered_anchor_misses
    )
    return {
        "true_targets": true_targets,
        "Pd": metrics.pd,
        "mIoU": metrics.miou,
        "nIoU": metrics.niou,
        "pixel_Fa": metrics.pixel_fa,
        "raw_background_Fa": metrics.raw_background_fa,
        "false_positive_components_per_megapixel": (
            metrics.fp_components_per_mp
        ),
        "recovered_anchor_misses": metrics.recovered_anchor_misses,
        "retention": metrics.retention,
        "budget_violation": metrics.budget_violation,
    }


def _assess_paet_formal_d_v_metrics(
    *,
    base_at_a: AggregateEvaluation,
    base_at_b: AggregateEvaluation,
    base_at_a_plus_cure: AggregateEvaluation,
    evaluation_result_fingerprint: str,
    artifact_binding_fingerprint: str,
    comparison_protocol_fingerprint: str,
    seed: int = PAET_FORMAL_SEED,
    d_t_accessed: bool = False,
) -> dict[str, object]:
    """Compute a non-authorizing metric diagnostic for sealed-result use.

    This is deliberately private: aggregate metric rows carry no provenance
    and therefore cannot themselves authorize access to D_T.
    """

    if seed != PAET_FORMAL_SEED:
        raise ValueError("PAET formal D_V decision fixes seed 42")
    if d_t_accessed is not False:
        raise PermissionError(
            "D_T must remain unread while making the D_V advancement decision"
        )
    evaluation_result_fingerprint = _digest(
        evaluation_result_fingerprint,
        name="evaluation_result_fingerprint",
    )
    artifact_binding_fingerprint = _digest(
        artifact_binding_fingerprint,
        name="artifact_binding_fingerprint",
    )
    comparison_protocol_fingerprint = _digest(
        comparison_protocol_fingerprint,
        name="comparison_protocol_fingerprint",
    )
    base_a_true, base_a_recovered = _validate_metrics(
        base_at_a,
        name="Base@A",
    )
    base_b_true, base_b_recovered = _validate_metrics(
        base_at_b,
        name="Base@B",
    )
    cure_true, cure_recovered = _validate_metrics(
        base_at_a_plus_cure,
        name="Base@A+CURE",
    )
    if (
        base_at_a.recovered_anchor_misses != 0
        or base_at_a.retained_anchor_covered
        != FORMAL_DV_ANCHOR_COVERED
    ):
        raise ValueError(
            "Base@A must be the unchanged 0.72 occupancy anchor"
        )
    for name, metrics in (
        ("Base@A", base_at_a),
        ("Base@B", base_at_b),
    ):
        if (
            metrics.budget_violation
            or metrics.retention < 0.99
            or metrics.pixel_fa > PAET_MAXIMUM_PIXEL_FA
            or metrics.raw_background_fa
            > PAET_MAXIMUM_RAW_BACKGROUND_FA
            or metrics.fp_components_per_mp
            > PAET_MAXIMUM_FP_COMPONENTS_PER_MP
        ):
            raise ValueError(
                f"{name} is not a valid comparator under the frozen budget"
            )

    best_base_true = max(base_a_true, base_b_true)
    best_base_recovered = max(
        base_a_recovered,
        base_b_recovered,
    )
    best_base_miou = max(base_at_a.miou, base_at_b.miou)
    best_base_niou = max(base_at_a.niou, base_at_b.niou)
    true_target_margin = cure_true - best_base_true
    recovered_margin = cure_recovered - best_base_recovered
    checks = {
        "CURE_true_targets_above_best_base_by_at_least_2": (
            true_target_margin >= PAET_MINIMUM_TRUE_TARGET_MARGIN
        ),
        "CURE_recovered_anchor_misses_above_best_base_by_at_least_2": (
            recovered_margin >= PAET_MINIMUM_RECOVERED_MISS_MARGIN
        ),
        "CURE_retention_equal_1": isclose(
            base_at_a_plus_cure.retention,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "CURE_mIoU_not_below_best_base": (
            base_at_a_plus_cure.miou >= best_base_miou
        ),
        "CURE_nIoU_not_below_best_base": (
            base_at_a_plus_cure.niou >= best_base_niou
        ),
        "CURE_pixel_Fa_le_1e-4": (
            base_at_a_plus_cure.pixel_fa <= PAET_MAXIMUM_PIXEL_FA
        ),
        "CURE_raw_background_Fa_le_1e-4": (
            base_at_a_plus_cure.raw_background_fa
            <= PAET_MAXIMUM_RAW_BACKGROUND_FA
        ),
        "CURE_false_positive_components_per_megapixel_le_100": (
            base_at_a_plus_cure.fp_components_per_mp
            <= PAET_MAXIMUM_FP_COMPONENTS_PER_MP
        ),
        "CURE_budget_violation_false": (
            base_at_a_plus_cure.budget_violation is False
        ),
        "D_T_not_accessed": True,
    }
    gate_passed = all(checks.values())
    payload: dict[str, object] = {
        "schema_version": PAET_FORMAL_DV_DECISION_SCHEMA,
        "method": PAET_FORMAL_METHOD,
        "seed": seed,
        "runtime_split": "D_V",
        "D_T_accessed": False,
        "development_gate_only": True,
        "gate_thresholds": {
            "minimum_true_target_margin": (
                PAET_MINIMUM_TRUE_TARGET_MARGIN
            ),
            "minimum_recovered_anchor_miss_margin": (
                PAET_MINIMUM_RECOVERED_MISS_MARGIN
            ),
            "required_retention": 1.0,
            "mIoU_floor": "max(Base@A,Base@B)",
            "nIoU_floor": "max(Base@A,Base@B)",
            "maximum_pixel_Fa": PAET_MAXIMUM_PIXEL_FA,
            "maximum_raw_background_Fa": (
                PAET_MAXIMUM_RAW_BACKGROUND_FA
            ),
            "maximum_false_positive_components_per_megapixel": (
                PAET_MAXIMUM_FP_COMPONENTS_PER_MP
            ),
        },
        "operating_points": {
            "Base@A": _metrics_summary(base_at_a),
            "Base@B": _metrics_summary(base_at_b),
            "Base@A+CURE": _metrics_summary(
                base_at_a_plus_cure
            ),
        },
        "best_base": {
            "true_targets": best_base_true,
            "recovered_anchor_misses": best_base_recovered,
            "mIoU": best_base_miou,
            "nIoU": best_base_niou,
        },
        "CURE_margins": {
            "true_targets": true_target_margin,
            "recovered_anchor_misses": recovered_margin,
        },
        "checks": checks,
        "gate_passed": gate_passed,
        "status": (
            "PAET_BFA_V21_FORMAL_D_V_GATE_PASS"
            if gate_passed
            else "PAET_BFA_V21_FORMAL_D_V_GATE_FAIL"
        ),
        "next_action": (
            "SEALED_RESULT_ELIGIBLE_FOR_ONE_FROZEN_D_T_CONFIRMATION"
            if gate_passed
            else "STOP_AND_PRESERVE_D_V_EVIDENCE"
        ),
        "eligible_for_D_T_confirmation": gate_passed,
        "authorizes_D_T": False,
        "final_model_success_established": False,
        "authorizes_full_CURE": False,
        "authorizes_cross_backbone": False,
        "bindings": {
            "evaluation_result_fingerprint": (
                evaluation_result_fingerprint
            ),
            "artifact_binding_fingerprint": (
                artifact_binding_fingerprint
            ),
            "comparison_protocol_fingerprint": (
                comparison_protocol_fingerprint
            ),
        },
    }
    return {
        **payload,
        "decision_fingerprint": stable_fingerprint(payload),
    }


def assess_paet_formal_d_v_result(
    result: PAETFormalDVEvaluationResult,
) -> dict[str, object]:
    """Assess one sealed PAET result without caller-supplied metric rows."""

    if type(result) is not PAETFormalDVEvaluationResult:
        raise TypeError("result must be PAETFormalDVEvaluationResult")
    result.verify_unchanged()
    diagnostic = _assess_paet_formal_d_v_metrics(
        base_at_a=result.base_at_a,
        base_at_b=result.base_at_b,
        base_at_a_plus_cure=result.base_at_a_plus_cure,
        evaluation_result_fingerprint=result.result_fingerprint,
        artifact_binding_fingerprint=(
            result.artifact_binding_fingerprint
        ),
        comparison_protocol_fingerprint=(
            result.comparison_protocol_fingerprint
        ),
        seed=result.seed,
        d_t_accessed=False,
    )
    # Provenance is checked above; this is the sole public authorization path.
    payload = {
        **{
            key: value
            for key, value in diagnostic.items()
            if key != "decision_fingerprint"
        },
        "next_action": (
            "AUTHORIZE_ONE_FROZEN_D_T_CONFIRMATION"
            if diagnostic["gate_passed"]
            else "STOP_AND_PRESERVE_D_V_EVIDENCE"
        ),
        "authorizes_D_T": diagnostic["gate_passed"],
        "sealed_evaluation_result_verified": True,
    }
    return {
        **payload,
        "decision_fingerprint": stable_fingerprint(payload),
    }


__all__ = [
    "PAET_FORMAL_DV_DECISION_SCHEMA",
    "PAET_MAXIMUM_FP_COMPONENTS_PER_MP",
    "PAET_MAXIMUM_PIXEL_FA",
    "PAET_MAXIMUM_RAW_BACKGROUND_FA",
    "PAET_MINIMUM_RECOVERED_MISS_MARGIN",
    "PAET_MINIMUM_TRUE_TARGET_MARGIN",
    "assess_paet_formal_d_v_result",
]
