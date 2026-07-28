"""Read-only curvature-sign audit for the frozen v20 BFA-CMIF checkpoint.

This module does not define a trainable candidate.  It decomposes the same
shared feature-presence energy already used by BFA-CMIF into

``delta = 0.5 * (H(U) - H(flip(U)))``

and the flip-even midpoint curvature

``e = 0.5 * (H0 + H1) - Hm``.

The sole frozen-checkpoint proxy is predeclared as

``delta_proxy = delta * (1 - tanh(e / 0.9))``.

No parameter is added or changed.  The proxy is evaluated at the unchanged
zero threshold and is only a read-only decision aid for the next field
equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import comb, isfinite
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..cache.schema import stable_fingerprint
from ..coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
)
from ..frozen_base import module_state_fingerprint
from ..paired_types import tensor_content_fingerprint
from .coverage_state_bounded_protocol import (
    CoverageStateBoundedPopulation,
)
from .coverage_state_real_dr_inputs import CoverageStateRealDRInputs
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
    evaluate_coverage_state_zero_level_checkpoint,
)


COVERAGE_STATE_BFA_CURVATURE_AUDIT_SCHEMA = (
    "cure-lite-bfa-cmif-v20-r2-checkpoint-curvature-sign-audit-v1"
)
COVERAGE_STATE_BFA_CURVATURE_PROXY_POLICY = (
    "delta_prime=delta*(1-tanh(e/0.9));zero_threshold;no_search-v1"
)
COVERAGE_STATE_BFA_CURVATURE_SCALE = 0.9
COVERAGE_STATE_BFA_CURVATURE_DECISION_ACCEPT = (
    "V21_FORMULA_ACCEPTED_FOR_IMPLEMENTATION"
)
COVERAGE_STATE_BFA_CURVATURE_DECISION_REJECT = "V21_FORMULA_REJECTED"

# Frozen v20 r2 evidence binding.  The audit must not rewrite any of these
# artifacts; the values only identify its immutable input evidence.
COVERAGE_STATE_BFA_V20_R2_COMPLETE_FINGERPRINT = (
    "8908a8c1896951e46fd737aa6f7fef2c9935e6524632b3576b8069faa026e2eb"
)
COVERAGE_STATE_BFA_V20_R2_ZERO_RECEIPT_FINGERPRINT = (
    "4301c8e9f3393c2bc64c28b20e3b6e16bdc98b974281b7d9c67d239a86c76219"
)
COVERAGE_STATE_BFA_V20_R2_DIAGNOSTIC_FINGERPRINT = (
    "50a92452a04d2a40f735c4e4cef75ce50df4ecf98338ce145f344bd6a76b3b77"
)
COVERAGE_STATE_BFA_V20_R2_CACHE_FINGERPRINT = (
    "c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca00754c206092aaf66d8"
)
COVERAGE_STATE_BFA_V20_R2_MODEL_FINGERPRINT = (
    "0393532f8ea62e790c120ca0c0b86bf04c67b88c863e333f3f7c640d865ab5c0"
)
COVERAGE_STATE_BFA_V20_R2_CHECKPOINT_SHA256 = (
    "040d2ca4ffa012c813e2c3e5dfa2c6f4877a91c8ff0b901bf8dc83df62026c42"
)
COVERAGE_STATE_BFA_V20_R2_POPULATION_FINGERPRINT = (
    "1a53467d57bea595afcc1edd3330708d1dda39e0e2d606325e552e8993e7841c"
)
COVERAGE_STATE_BFA_V20_R2_REAL_INPUTS_FINGERPRINT = (
    "ee717a7e13461fb86cacc65d33efd331abcf9b27611f254f981082d45eb7bfb4"
)

COVERAGE_STATE_BFA_V20_R2_CLEAN_TARGET_PIXELS = 149
COVERAGE_STATE_BFA_V20_R2_SPILL_PIXELS = 54
COVERAGE_STATE_BFA_V20_R2_FACTUAL_TARGET_PIXELS = 335
COVERAGE_STATE_BFA_V20_R2_COMPONENT_GROUPS = 16


@dataclass(frozen=True)
class CoverageStateBFAScalarOddCurvature:
    """Scalar output-grid decomposition of one shared BFA energy."""

    canonical_odd_delta: Tensor
    oriented_odd_delta: Tensor
    midpoint_curvature: Tensor
    proxy_multiplier: Tensor
    proxy_oriented_delta: Tensor
    bfa_field: Tensor
    proxy_field: Tensor


def curvature_gated_odd_delta(
    delta: Tensor,
    midpoint_curvature: Tensor,
) -> Tensor:
    """Apply the one predeclared curvature gate without broadcasting."""

    if (
        not isinstance(delta, Tensor)
        or not isinstance(midpoint_curvature, Tensor)
        or not delta.is_floating_point()
        or not midpoint_curvature.is_floating_point()
        or delta.shape != midpoint_curvature.shape
        or delta.dtype != midpoint_curvature.dtype
        or delta.device != midpoint_curvature.device
    ):
        raise TypeError("delta and curvature must be aligned floating tensors")
    if not bool(
        torch.isfinite(delta).all()
        & torch.isfinite(midpoint_curvature).all()
    ):
        raise ValueError("delta and curvature must be finite")
    scale = torch.full_like(delta, COVERAGE_STATE_BFA_CURVATURE_SCALE)
    return delta * (1.0 - torch.tanh(midpoint_curvature / scale))


def evaluate_bfa_scalar_odd_curvature(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
    feature: Tensor,
    occupancy: Tensor,
) -> CoverageStateBFAScalarOddCurvature:
    """Evaluate scalar odd delta and curvature from one shared energy."""

    if type(model) is not CURELiteBinaryFlipAntisymmetricLevelSet:
        raise TypeError("curvature audit requires the exact BFA-CMIF model")
    fields = model.forward_fields(feature, occupancy)
    phase = fields.phase_occupancy
    actual = fields.actual_feature_presence_energy.expand_as(
        fields.flipped_feature_presence_energy
    )
    flipped = fields.flipped_feature_presence_energy
    h0 = torch.where(phase, flipped, actual)
    h1 = torch.where(phase, actual, flipped)

    center = model.config.coarse_radius
    center_weight = model.occupancy_weight[
        :, :, center, center
    ].transpose(0, 1)
    midpoint_delta = (
        0.5 - phase.to(dtype=torch.float32)
    ).unsqueeze(2) * center_weight[None, :, :, None, None]
    midpoint_hidden = (
        F.silu(fields.joint_affine.unsqueeze(1) + midpoint_delta)
        - F.silu(fields.occupancy_affine.unsqueeze(1) + midpoint_delta)
    )
    hm = (
        midpoint_hidden
        * model.scalar_energy_weight[None, None, :, None, None]
    ).sum(dim=2)
    canonical_native = 0.5 * (h0 - h1)
    curvature_native = 0.5 * (h0 + h1) - hm
    oriented_native = fields.native_phase_interaction
    expected_oriented = torch.where(
        phase,
        -canonical_native,
        canonical_native,
    )
    if not torch.allclose(
        oriented_native,
        expected_oriented,
        rtol=2.0e-6,
        atol=2.0e-7,
    ):
        raise AssertionError("BFA scalar odd orientation changed")

    canonical = F.pixel_shuffle(
        canonical_native,
        model.config.feature_stride,
    ).contiguous()
    oriented = F.pixel_shuffle(
        oriented_native,
        model.config.feature_stride,
    ).contiguous()
    curvature = F.pixel_shuffle(
        curvature_native,
        model.config.feature_stride,
    ).contiguous()
    proxy = curvature_gated_odd_delta(oriented, curvature)
    multiplier = (
        1.0
        - torch.tanh(
            curvature
            / torch.full_like(curvature, COVERAGE_STATE_BFA_CURVATURE_SCALE)
        )
    ).contiguous()
    bfa_field = fields.field
    proxy_field = (model.config.field_amplitude + proxy).contiguous()
    result = CoverageStateBFAScalarOddCurvature(
        canonical_odd_delta=canonical,
        oriented_odd_delta=oriented,
        midpoint_curvature=curvature,
        proxy_multiplier=multiplier,
        proxy_oriented_delta=proxy,
        bfa_field=bfa_field,
        proxy_field=proxy_field,
    )
    expected_shape = occupancy.shape
    for value in result.__dict__.values():
        if (
            value.shape != expected_shape
            or value.dtype != torch.float32
            or value.device != occupancy.device
            or not bool(torch.isfinite(value).all())
        ):
            raise FloatingPointError("BFA scalar curvature field is invalid")
    if not torch.equal(result.bfa_field, fields.field):
        raise AssertionError("curvature audit changed the BFA field")
    return result


def _hex(value: float) -> str:
    result = float(value)
    if not isfinite(result):
        raise FloatingPointError("audit value is non-finite")
    return result.hex()


def _median(values: Tensor) -> float:
    """Return the deterministic midpoint-of-middle-values median."""

    value = values.detach().to("cpu", dtype=torch.float32).flatten()
    if value.numel() < 1 or not bool(torch.isfinite(value).all()):
        raise ValueError("audit median requires finite nonempty values")
    ordered = torch.sort(value).values.to(dtype=torch.float64)
    count = int(ordered.numel())
    middle = count // 2
    result = (
        float(ordered[middle])
        if count % 2
        else float(0.5 * (ordered[middle - 1] + ordered[middle]))
    )
    if not isfinite(result):
        raise FloatingPointError("audit median is non-finite")
    return result


def _distribution(values: Tensor) -> dict[str, object]:
    value = values.detach().to("cpu", dtype=torch.float32).flatten().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("audit distribution is non-finite")
    count = int(value.numel())
    if count == 0:
        return {
            "count": 0,
            "negative_count": 0,
            "zero_count": 0,
            "positive_count": 0,
            "mean_hex": None,
            "minimum_hex": None,
            "maximum_hex": None,
            "nearest_rank_quantiles": None,
            "ordered_value_fingerprint": tensor_content_fingerprint(value),
        }
    ordered = torch.sort(value).values
    indices = {
        "q000": 0,
        "q025": ((count - 1) * 25 + 50) // 100,
        "q050": ((count - 1) * 50 + 50) // 100,
        "q075": ((count - 1) * 75 + 50) // 100,
        "q100": count - 1,
    }
    return {
        "count": count,
        "negative_count": int(torch.count_nonzero(value < 0.0)),
        "zero_count": int(torch.count_nonzero(value == 0.0)),
        "positive_count": int(torch.count_nonzero(value > 0.0)),
        "mean_hex": _hex(float(value.to(torch.float64).mean())),
        "minimum_hex": _hex(float(ordered[0])),
        "maximum_hex": _hex(float(ordered[-1])),
        "nearest_rank_quantiles": {
            name: _hex(float(ordered[index]))
            for name, index in indices.items()
        },
        "ordered_value_fingerprint": tensor_content_fingerprint(ordered),
        "quantile_policy": (
            "fixed_nearest_rank_round_half_up_over_count_minus_one_v1"
        ),
    }


def one_sided_exact_sign_test(
    signed_values: tuple[float, ...],
    *,
    desired: str,
) -> dict[str, object]:
    """Return a fixed one-sided exact sign test, excluding exact ties."""

    if desired not in {"negative", "positive"}:
        raise ValueError("desired sign must be negative or positive")
    if any(not isfinite(float(value)) for value in signed_values):
        raise ValueError("sign-test values must be finite")
    if desired == "negative":
        wins = sum(value < 0.0 for value in signed_values)
        losses = sum(value > 0.0 for value in signed_values)
    else:
        wins = sum(value > 0.0 for value in signed_values)
        losses = sum(value < 0.0 for value in signed_values)
    ties = len(signed_values) - wins - losses
    n = wins + losses
    p = (
        1.0
        if n == 0
        else sum(comb(n, k) for k in range(wins, n + 1)) / (2**n)
    )
    return {
        "desired_sign": desired,
        "group_count": len(signed_values),
        "non_tied_count": n,
        "win_count": wins,
        "loss_count": losses,
        "tie_count": ties,
        "one_sided_exact_p_hex": _hex(p),
        "minimum_non_tied_count": 5,
        "maximum_p_hex": float(0.05).hex(),
        "passed": n >= 5 and p <= 0.05,
    }


def decide_coverage_state_bfa_curvature_audit(
    *,
    mask_counts: Mapping[str, int],
    sign_tests: Mapping[str, Mapping[str, object]],
    global_curvature_medians: Mapping[str, float],
    proxy_multiplier_summary: Mapping[str, object],
    proxy_summary: Mapping[str, object],
) -> tuple[str, tuple[tuple[str, bool], ...]]:
    """Apply the predeclared structural and zero-level proxy decision."""

    checks = {
        "clean_target_mask_exact": (
            mask_counts.get("clean_added_target")
            == COVERAGE_STATE_BFA_V20_R2_CLEAN_TARGET_PIXELS
        ),
        "v20_spill_mask_exact": (
            mask_counts.get("v20_new_completion_outside")
            == COVERAGE_STATE_BFA_V20_R2_SPILL_PIXELS
        ),
        "factual_target_mask_exact": (
            mask_counts.get("factual_target")
            == COVERAGE_STATE_BFA_V20_R2_FACTUAL_TARGET_PIXELS
        ),
        "component_group_count_exact": (
            mask_counts.get("component_null_groups")
            == COVERAGE_STATE_BFA_V20_R2_COMPONENT_GROUPS
        ),
        "full_clean_true_background_present": (
            int(mask_counts.get("clean_true_background", 0)) > 0
        ),
        "clean_target_curvature_negative": bool(
            sign_tests.get("clean_target_e_negative", {}).get("passed")
        ),
        "spill_curvature_positive": bool(
            sign_tests.get("spill_e_positive", {}).get("passed")
        ),
        "factual_curvature_negative": bool(
            sign_tests.get("factual_target_e_negative", {}).get("passed")
        ),
        "same_pair_target_below_spill": bool(
            sign_tests.get("same_pair_target_below_spill", {}).get(
                "passed"
            )
        ),
        "global_clean_target_curvature_negative": (
            float(
                global_curvature_medians.get(
                    "clean_added_target",
                    float("inf"),
                )
            )
            < 0.0
        ),
        "global_spill_curvature_positive": (
            float(
                global_curvature_medians.get(
                    "v20_new_completion_outside",
                    float("-inf"),
                )
            )
            > 0.0
        ),
        "global_factual_curvature_negative": (
            float(
                global_curvature_medians.get(
                    "factual_target",
                    float("inf"),
                )
            )
            < 0.0
        ),
        "proxy_multiplier_finite_strict_open_interval": (
            int(proxy_multiplier_summary.get("count", 0)) > 0
            and proxy_multiplier_summary.get("all_finite") is True
            and int(
                proxy_multiplier_summary.get(
                    "less_than_or_equal_zero_count",
                    -1,
                )
            )
            == 0
            and int(
                proxy_multiplier_summary.get(
                    "greater_than_or_equal_two_count",
                    -1,
                )
            )
            == 0
        ),
        "proxy_clean_target_improved": (
            int(proxy_summary.get("clean_target_negative_pixels", -1))
            >= 116
        ),
        "proxy_clean_outside_improved": (
            int(proxy_summary.get("clean_outside_pixels", 10**9)) <= 53
        ),
        "proxy_pair_pareto": bool(
            sign_tests.get("proxy_pair_pareto", {}).get("passed")
        ),
        "proxy_factual_recovered_preserved": (
            int(proxy_summary.get("factual_recovered", -1)) >= 16
        ),
        "proxy_factual_strict_preserved": (
            int(proxy_summary.get("factual_strict", -1)) >= 14
        ),
        "proxy_factual_negative_preserved": (
            int(proxy_summary.get("factual_target_negative_pixels", -1))
            >= 310
        ),
        "proxy_factual_no_miss_preserved": (
            int(proxy_summary.get("factual_no_miss_passed", -1)) >= 16
        ),
        "proxy_component_null_preserved": (
            int(proxy_summary.get("component_null_passed", -1)) >= 16
        ),
        "proxy_identity_null_preserved": (
            int(proxy_summary.get("identity_null_passed", -1)) >= 16
        ),
        "proxy_diagnostic_null_preserved": (
            proxy_summary.get("diagnostic_null_passed") is True
        ),
        "proxy_invalid_completion_preserved": (
            int(proxy_summary.get("invalid_completion_pixels", -1)) == 0
        ),
        "proxy_compact_support_not_worse": (
            int(proxy_summary.get("clean_compact_support_passed", -1))
            >= 1
        ),
    }
    ordered = tuple(sorted(checks.items()))
    decision = (
        COVERAGE_STATE_BFA_CURVATURE_DECISION_ACCEPT
        if all(checks.values())
        else COVERAGE_STATE_BFA_CURVATURE_DECISION_REJECT
    )
    return decision, ordered


class _CurvatureProxyModule(nn.Module):
    def __init__(
        self,
        source: CURELiteBinaryFlipAntisymmetricLevelSet,
    ) -> None:
        super().__init__()
        self.source = source
        self._multiplier_values: list[Tensor] = []

    @property
    def feature_stride(self) -> int:
        return self.source.feature_stride

    @property
    def feature_channels(self) -> int:
        return self.source.feature_channels

    @property
    def config(self) -> object:
        return self.source.config

    def forward(self, feature: Tensor, occupancy: Tensor) -> Tensor:
        result = evaluate_bfa_scalar_odd_curvature(
            self.source,
            feature,
            occupancy,
        )
        self._multiplier_values.append(
            result.proxy_multiplier.detach().to("cpu").flatten().contiguous()
        )
        return result.proxy_field

    def multiplier_summary(self) -> dict[str, object]:
        if not self._multiplier_values:
            raise RuntimeError("proxy multiplier was never evaluated")
        value = torch.cat(self._multiplier_values)
        finite = torch.isfinite(value)
        count = int(value.numel())
        finite_count = int(torch.count_nonzero(finite))
        return {
            "count": count,
            "finite_count": finite_count,
            "all_finite": finite_count == count,
            "minimum_hex": (
                _hex(float(value.amin())) if finite_count == count else None
            ),
            "maximum_hex": (
                _hex(float(value.amax())) if finite_count == count else None
            ),
            "exact_zero_count": int(torch.count_nonzero(value == 0.0)),
            "exact_two_count": int(torch.count_nonzero(value == 2.0)),
            "less_than_or_equal_zero_count": int(
                torch.count_nonzero(value <= 0.0)
            ),
            "greater_than_or_equal_two_count": int(
                torch.count_nonzero(value >= 2.0)
            ),
            "ordered_value_fingerprint": tensor_content_fingerprint(
                torch.sort(value).values
            ),
        }


def _gradient_fingerprint(model: nn.Module) -> str:
    return stable_fingerprint(
        {
            name: (
                None
                if parameter.grad is None
                else tensor_content_fingerprint(parameter.grad)
            )
            for name, parameter in sorted(model.named_parameters())
        }
    )


def _proxy_summary(
    result: CoverageStateZeroLevelEvaluationResult,
) -> dict[str, object]:
    factual = tuple(
        row for row in result.natural_diagnostics
        if row.state_kind == "factual_miss"
    )
    no_miss = tuple(
        row for row in result.natural_diagnostics
        if row.state_kind == "factual_no_miss"
    )
    clean = tuple(
        row for row in result.pair_diagnostics
        if row.optimizer_role == "clean_positive"
    )
    component = tuple(
        row for row in result.pair_diagnostics
        if row.optimizer_role == "component_null"
    )
    identity = tuple(
        row for row in result.pair_diagnostics
        if row.optimizer_role == "identity_diagnostic"
    )
    diagnostic = tuple(
        row for row in result.pair_diagnostics
        if row.optimizer_role == "diagnostic_only"
    )
    return {
        "factual_recovered": sum(row.target_recovered is True for row in factual),
        "factual_strict": sum(row.gate_passed for row in factual),
        "factual_target_negative_pixels": sum(
            row.focus_target_negative_pixels for row in factual
        ),
        "factual_no_miss_passed": sum(row.gate_passed for row in no_miss),
        "clean_target_negative_pixels": sum(
            row.minus_added_target_negative_pixels for row in clean
        ),
        "clean_outside_pixels": sum(
            int(row.new_completion_outside_added_target_pixels or 0)
            for row in clean
        ),
        "clean_compact_support_passed": sum(
            row.compact_support_passed is True for row in clean
        ),
        "component_null_passed": sum(row.gate_passed for row in component),
        "identity_null_passed": sum(row.gate_passed for row in identity),
        "diagnostic_null_passed": bool(diagnostic) and all(
            row.gate_passed for row in diagnostic
        ),
        "invalid_completion_pixels": sum(
            row.invalid_completion_pixels
            for row in result.natural_diagnostics
        ) + sum(
            row.invalid_completion_pixels_plus
            + row.invalid_completion_pixels_minus
            for row in result.pair_diagnostics
        ),
    }


@dataclass(frozen=True)
class CoverageStateBFACurvatureAuditReceipt:
    schema_version: str
    evidence_binding: tuple[tuple[str, str], ...]
    checkpoint_fingerprint_before: str
    checkpoint_fingerprint_after: str
    gradient_fingerprint_before: str
    gradient_fingerprint_after: str
    population_fingerprint: str
    cache_fingerprint: str
    group_distributions: tuple[tuple[str, dict[str, object]], ...]
    group_rows: tuple[dict[str, object], ...]
    mask_counts: tuple[tuple[str, int], ...]
    sign_tests: tuple[tuple[str, dict[str, object]], ...]
    global_curvature_medians: tuple[tuple[str, str], ...]
    proxy_multiplier_summary: dict[str, object]
    proxy_zero_level_fingerprint: str
    proxy_summary: dict[str, object]
    decision_checks: tuple[tuple[str, bool], ...]
    decision: str
    model_training_before: bool
    model_training_after: bool
    optimizer_constructed: bool
    backward_performed: bool
    training_performed: bool
    d_v_accessed: bool
    d_t_accessed: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_binding": dict(self.evidence_binding),
            "formula": {
                "policy": COVERAGE_STATE_BFA_CURVATURE_PROXY_POLICY,
                "curvature_scale_hex": (
                    COVERAGE_STATE_BFA_CURVATURE_SCALE.hex()
                ),
                "threshold_hex": float(0.0).hex(),
                "parameter_search_performed": False,
            },
            "checkpoint": {
                "before": self.checkpoint_fingerprint_before,
                "after": self.checkpoint_fingerprint_after,
                "gradient_before": self.gradient_fingerprint_before,
                "gradient_after": self.gradient_fingerprint_after,
                "training_before": self.model_training_before,
                "training_after": self.model_training_after,
            },
            "population_fingerprint": self.population_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "group_distributions": dict(self.group_distributions),
            "group_rows": list(self.group_rows),
            "group_statistic_policy": (
                "per-record-or-pair-midpoint-of-middle-values-median-curvature;"
                "exact-ties-excluded-one-sided-binomial-sign-test-v1"
            ),
            "mask_counts": dict(self.mask_counts),
            "sign_tests": dict(self.sign_tests),
            "global_curvature_medians_hex": dict(
                self.global_curvature_medians
            ),
            "proxy_multiplier_summary": self.proxy_multiplier_summary,
            "proxy_zero_level_fingerprint": (
                self.proxy_zero_level_fingerprint
            ),
            "proxy_summary": self.proxy_summary,
            "decision_checks": dict(self.decision_checks),
            "decision": self.decision,
            "execution": {
                "optimizer_constructed": self.optimizer_constructed,
                "backward_performed": self.backward_performed,
                "training_performed": self.training_performed,
                "D_V_accessed": self.d_v_accessed,
                "D_T_accessed": self.d_t_accessed,
            },
        }

    @cached_property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify(self) -> None:
        if (
            self.checkpoint_fingerprint_before
            != self.checkpoint_fingerprint_after
            or self.gradient_fingerprint_before
            != self.gradient_fingerprint_after
            or self.model_training_before != self.model_training_after
            or self.optimizer_constructed
            or self.backward_performed
            or self.training_performed
            or self.d_v_accessed
            or self.d_t_accessed
            or self.decision not in {
                COVERAGE_STATE_BFA_CURVATURE_DECISION_ACCEPT,
                COVERAGE_STATE_BFA_CURVATURE_DECISION_REJECT,
            }
        ):
            raise RuntimeError("BFA curvature audit read-only contract failed")


def audit_coverage_state_bfa_curvature_checkpoint(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device | str,
    evidence_binding: Mapping[str, str] | None = None,
) -> CoverageStateBFACurvatureAuditReceipt:
    """Run the read-only decomposition and fixed zero-threshold proxy."""

    if type(model) is not CURELiteBinaryFlipAntisymmetricLevelSet:
        raise TypeError("curvature audit requires the exact BFA-CMIF model")
    if not isinstance(population, CoverageStateBoundedPopulation):
        raise TypeError("population must be CoverageStateBoundedPopulation")
    if model.training:
        raise ValueError("curvature audit requires model.eval()")
    requested = torch.device(device)
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("curvature audit device must be CPU or CUDA")
    if requested.type == "cuda" and requested.index is None:
        raise ValueError("curvature audit CUDA device needs an explicit index")
    if any(value.device != requested for value in model.state_dict().values()):
        raise ValueError("model must already be on the requested device")
    population.verify_unchanged()
    cache = population.cache
    if cache.raw_catalog.split != "D_R":
        raise PermissionError("curvature audit permits only D_R")
    checkpoint_before = module_state_fingerprint(model)
    gradient_before = _gradient_fingerprint(model)
    training_before = model.training
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(requested)
        if requested.type == "cuda"
        else None
    )

    clean_target_values: list[Tensor] = []
    spill_values: list[Tensor] = []
    full_background_values: list[Tensor] = []
    factual_values: list[Tensor] = []
    component_values: list[Tensor] = []
    rows: list[dict[str, object]] = []
    baseline_pair_metrics: dict[str, tuple[int, int]] = {}
    clean_target_group_medians: dict[str, float] = {}
    spill_group_medians: dict[str, float] = {}
    factual_group_medians: list[float] = []
    metric_attributes = {
        "oriented_odd_delta": "oriented_odd_delta",
        "midpoint_curvature": "midpoint_curvature",
        "proxy_multiplier": "proxy_multiplier",
        "proxy_oriented_delta": "proxy_oriented_delta",
        "bfa_field": "bfa_field",
        "proxy_field": "proxy_field",
    }
    metric_values: dict[str, dict[str, list[Tensor]]] = {
        group: {name: [] for name in metric_attributes}
        for group in (
            "clean_added_target",
            "v20_new_completion_outside",
            "clean_true_background",
            "factual_target",
            "component_null",
        )
    }

    def collect(
        group: str,
        value: CoverageStateBFAScalarOddCurvature,
        mask: Tensor,
    ) -> None:
        for name, attribute in metric_attributes.items():
            metric_values[group][name].append(
                getattr(value, attribute)[mask].detach().to("cpu")
            )

    def evaluate(
        feature: Tensor,
        occupancy: Tensor,
    ) -> CoverageStateBFAScalarOddCurvature:
        with (
            torch.inference_mode(),
            torch.autocast(device_type=requested.type, enabled=False),
        ):
            return evaluate_bfa_scalar_odd_curvature(
                model,
                feature.to(device=requested, dtype=torch.float32),
                occupancy.to(device=requested),
            )

    factual_records = sorted(
        (
            row for row in cache.natural_records
            if row.record.state_kind == "factual_miss"
        ),
        key=lambda row: row.record.record_id,
    )
    for row in factual_records:
        value = evaluate(row.record.feature, row.record.occupancy)
        mask = (
            row.targets.focus_support
            & row.record.valid_mask
            & ~row.record.occupancy
        ).to(device=requested)
        e = value.midpoint_curvature[mask]
        collect("factual_target", value, mask)
        factual_values.append(e.detach().cpu())
        median = _median(e)
        factual_group_medians.append(median)
        rows.append({
            "group": "factual_target",
            "record_id": row.record.record_id,
            "coordinate_count": int(e.numel()),
            "mask_fingerprint": tensor_content_fingerprint(mask.to("cpu")),
            "curvature_median_hex": _hex(median),
        })

    clean_records = sorted(
        cache.clean_positive_records,
        key=lambda row: row.record.pair_id,
    )
    for row in clean_records:
        plus = evaluate(row.record.feature, row.record.occupancy_plus)
        minus = evaluate(row.record.feature, row.record.occupancy_minus)
        valid = row.record.valid_mask.to(device=requested)
        occupancy_plus = row.record.occupancy_plus.to(device=requested)
        occupancy_minus = row.record.occupancy_minus.to(device=requested)
        target_plus = row.record.target_plus.to(device=requested)
        target_minus = row.record.target_minus.to(device=requested)
        added = (
            target_minus & ~target_plus & valid & ~occupancy_minus
        )
        background = valid & ~occupancy_minus & ~target_minus
        completion_plus = (plus.bfa_field < 0.0) & ~occupancy_plus
        completion_minus = (minus.bfa_field < 0.0) & ~occupancy_minus
        new_completion = completion_minus & ~completion_plus & valid
        spill = new_completion & ~added
        target_e = minus.midpoint_curvature[added]
        background_e = minus.midpoint_curvature[background]
        spill_e = minus.midpoint_curvature[spill]
        collect("clean_added_target", minus, added)
        collect("clean_true_background", minus, background)
        collect("v20_new_completion_outside", minus, spill)
        clean_target_values.append(target_e.detach().cpu())
        full_background_values.append(background_e.detach().cpu())
        spill_values.append(spill_e.detach().cpu())
        target_median = _median(target_e)
        clean_target_group_medians[row.record.pair_id] = target_median
        if spill_e.numel():
            spill_group_medians[row.record.pair_id] = _median(spill_e)
        baseline_pair_metrics[row.record.pair_id] = (
            int(torch.count_nonzero(completion_minus & added)),
            int(torch.count_nonzero(spill)),
        )
        rows.append({
            "group": "clean_pair",
            "pair_id": row.record.pair_id,
            "target_count": int(target_e.numel()),
            "spill_count": int(spill_e.numel()),
            "full_background_count": int(background_e.numel()),
            "added_target_mask_fingerprint": tensor_content_fingerprint(
                added.to("cpu")
            ),
            "spill_mask_fingerprint": tensor_content_fingerprint(
                spill.to("cpu")
            ),
            "full_background_mask_fingerprint": tensor_content_fingerprint(
                background.to("cpu")
            ),
            "target_curvature_median_hex": _hex(target_median),
            "spill_curvature_median_hex": (
                None
                if not spill_e.numel()
                else _hex(spill_group_medians[row.record.pair_id])
            ),
        })

    component_records = sorted(
        cache.component_null_records,
        key=lambda row: row.record.pair_id,
    )
    for row in component_records:
        value = evaluate(row.record.feature, row.record.occupancy_minus)
        mask = (
            row.record.removed_component
            & row.record.valid_mask
            & ~row.record.occupancy_minus
            & ~row.record.target_minus
        ).to(device=requested)
        e = value.midpoint_curvature[mask]
        collect("component_null", value, mask)
        component_values.append(e.detach().cpu())
        rows.append({
            "group": "component_null",
            "pair_id": row.record.pair_id,
            "coordinate_count": int(e.numel()),
            "mask_fingerprint": tensor_content_fingerprint(mask.to("cpu")),
            "curvature_median_hex": (
                None
                if not e.numel()
                else _hex(_median(e))
            ),
        })

    group_tensors = {
        "clean_added_target": torch.cat(clean_target_values),
        "v20_new_completion_outside": torch.cat(spill_values),
        "clean_true_background": torch.cat(full_background_values),
        "factual_target": torch.cat(factual_values),
        "component_null": torch.cat(component_values),
    }
    group_distributions = tuple(
        (
            group,
            {
                metric: _distribution(torch.cat(values))
                for metric, values in sorted(metrics.items())
            },
        )
        for group, metrics in sorted(metric_values.items())
    )
    mask_counts = {
        name: int(value.numel())
        for name, value in group_tensors.items()
    }
    mask_counts["component_null_groups"] = len(component_records)
    global_curvature_medians = {
        group: _median(value)
        for group, value in group_tensors.items()
        if value.numel()
    }

    proxy_module = _CurvatureProxyModule(model).eval()
    proxy_result = evaluate_coverage_state_zero_level_checkpoint(
        proxy_module,
        cache,
        device=requested,
        config=CoverageStateZeroLevelEvaluationConfig(
            input_representation=COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
        ),
    )
    proxy_multiplier_summary = proxy_module.multiplier_summary()
    proxy_summary = _proxy_summary(proxy_result)
    proxy_clean = {
        row.pair_id: (
            row.minus_added_target_negative_pixels,
            int(row.new_completion_outside_added_target_pixels or 0),
        )
        for row in proxy_result.pair_diagnostics
        if row.optimizer_role == "clean_positive"
    }
    pareto_values: list[float] = []
    for pair_id in sorted(baseline_pair_metrics):
        before_target, before_spill = baseline_pair_metrics[pair_id]
        after_target, after_spill = proxy_clean[pair_id]
        if (
            after_target >= before_target
            and after_spill <= before_spill
            and (
                after_target > before_target
                or after_spill < before_spill
            )
        ):
            pareto_values.append(1.0)
            pareto_class = "win"
        elif (
            after_target <= before_target
            and after_spill >= before_spill
            and (
                after_target < before_target
                or after_spill > before_spill
            )
        ):
            pareto_values.append(-1.0)
            pareto_class = "loss"
        else:
            pareto_values.append(0.0)
            pareto_class = "tie_or_mixed"
        rows.append({
            "group": "proxy_pair_pareto",
            "pair_id": pair_id,
            "v20_target_negative_pixels": before_target,
            "v20_outside_pixels": before_spill,
            "proxy_target_negative_pixels": after_target,
            "proxy_outside_pixels": after_spill,
            "pareto_class": pareto_class,
        })

    paired_curvature = tuple(
        clean_target_group_medians[pair_id]
        - spill_group_medians[pair_id]
        for pair_id in sorted(spill_group_medians)
    )
    sign_tests = {
        "clean_target_e_negative": one_sided_exact_sign_test(
            tuple(
                clean_target_group_medians[key]
                for key in sorted(clean_target_group_medians)
            ),
            desired="negative",
        ),
        "spill_e_positive": one_sided_exact_sign_test(
            tuple(
                spill_group_medians[key]
                for key in sorted(spill_group_medians)
            ),
            desired="positive",
        ),
        "factual_target_e_negative": one_sided_exact_sign_test(
            tuple(factual_group_medians),
            desired="negative",
        ),
        "same_pair_target_below_spill": one_sided_exact_sign_test(
            paired_curvature,
            desired="negative",
        ),
        "proxy_pair_pareto": one_sided_exact_sign_test(
            tuple(pareto_values),
            desired="positive",
        ),
    }
    decision, decision_checks = decide_coverage_state_bfa_curvature_audit(
        mask_counts=mask_counts,
        sign_tests=sign_tests,
        global_curvature_medians=global_curvature_medians,
        proxy_multiplier_summary=proxy_multiplier_summary,
        proxy_summary=proxy_summary,
    )

    checkpoint_after = module_state_fingerprint(model)
    gradient_after = _gradient_fingerprint(model)
    training_after = model.training
    population.verify_unchanged()
    if not torch.equal(cpu_rng_before, torch.random.get_rng_state()):
        raise RuntimeError("CPU RNG changed during curvature audit")
    if (
        cuda_rng_before is not None
        and not torch.equal(
            cuda_rng_before,
            torch.cuda.get_rng_state(requested),
        )
    ):
        raise RuntimeError("CUDA RNG changed during curvature audit")
    receipt = CoverageStateBFACurvatureAuditReceipt(
        schema_version=COVERAGE_STATE_BFA_CURVATURE_AUDIT_SCHEMA,
        evidence_binding=tuple(sorted((evidence_binding or {}).items())),
        checkpoint_fingerprint_before=checkpoint_before,
        checkpoint_fingerprint_after=checkpoint_after,
        gradient_fingerprint_before=gradient_before,
        gradient_fingerprint_after=gradient_after,
        population_fingerprint=population.population_fingerprint,
        cache_fingerprint=cache.cache_fingerprint,
        group_distributions=group_distributions,
        group_rows=tuple(rows),
        mask_counts=tuple(sorted(mask_counts.items())),
        sign_tests=tuple(sorted(sign_tests.items())),
        global_curvature_medians=tuple(
            (name, _hex(value))
            for name, value in sorted(global_curvature_medians.items())
        ),
        proxy_multiplier_summary=proxy_multiplier_summary,
        proxy_zero_level_fingerprint=proxy_result.result_fingerprint,
        proxy_summary=proxy_summary,
        decision_checks=decision_checks,
        decision=decision,
        model_training_before=training_before,
        model_training_after=training_after,
        optimizer_constructed=False,
        backward_performed=False,
        training_performed=False,
        d_v_accessed=False,
        d_t_accessed=False,
    )
    receipt.verify()
    return receipt


def audit_frozen_coverage_state_bfa_v20_r2_curvature_checkpoint(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
    real_inputs: CoverageStateRealDRInputs,
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device | str,
    complete_fingerprint: str,
    zero_receipt_fingerprint: str,
    diagnostic_fingerprint: str,
    checkpoint_file_sha256: str,
) -> CoverageStateBFACurvatureAuditReceipt:
    """Strict r2-bound wrapper around the generic read-only audit."""

    if not isinstance(real_inputs, CoverageStateRealDRInputs):
        raise TypeError("real_inputs must be CoverageStateRealDRInputs")
    real_inputs.verify_unchanged()
    binding = {
        "complete_fingerprint": complete_fingerprint,
        "zero_receipt_fingerprint": zero_receipt_fingerprint,
        "diagnostic_fingerprint": diagnostic_fingerprint,
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "real_inputs_fingerprint": real_inputs.build_fingerprint,
        "population_fingerprint": population.population_fingerprint,
        "cache_fingerprint": population.cache.cache_fingerprint,
        "model_fingerprint": module_state_fingerprint(model),
    }
    expected = {
        "complete_fingerprint": (
            COVERAGE_STATE_BFA_V20_R2_COMPLETE_FINGERPRINT
        ),
        "zero_receipt_fingerprint": (
            COVERAGE_STATE_BFA_V20_R2_ZERO_RECEIPT_FINGERPRINT
        ),
        "diagnostic_fingerprint": (
            COVERAGE_STATE_BFA_V20_R2_DIAGNOSTIC_FINGERPRINT
        ),
        "checkpoint_file_sha256": (
            COVERAGE_STATE_BFA_V20_R2_CHECKPOINT_SHA256
        ),
        "real_inputs_fingerprint": (
            COVERAGE_STATE_BFA_V20_R2_REAL_INPUTS_FINGERPRINT
        ),
        "population_fingerprint": (
            COVERAGE_STATE_BFA_V20_R2_POPULATION_FINGERPRINT
        ),
        "cache_fingerprint": COVERAGE_STATE_BFA_V20_R2_CACHE_FINGERPRINT,
        "model_fingerprint": COVERAGE_STATE_BFA_V20_R2_MODEL_FINGERPRINT,
    }
    if binding != expected:
        raise RuntimeError("v20 r2 curvature-audit evidence binding differs")
    receipt = audit_coverage_state_bfa_curvature_checkpoint(
        model,
        population,
        device=device,
        evidence_binding=binding,
    )
    real_inputs.verify_unchanged()
    return receipt


__all__ = [
    "COVERAGE_STATE_BFA_CURVATURE_AUDIT_SCHEMA",
    "COVERAGE_STATE_BFA_CURVATURE_DECISION_ACCEPT",
    "COVERAGE_STATE_BFA_CURVATURE_DECISION_REJECT",
    "COVERAGE_STATE_BFA_CURVATURE_PROXY_POLICY",
    "COVERAGE_STATE_BFA_CURVATURE_SCALE",
    "CoverageStateBFACurvatureAuditReceipt",
    "CoverageStateBFAScalarOddCurvature",
    "audit_coverage_state_bfa_curvature_checkpoint",
    "audit_frozen_coverage_state_bfa_v20_r2_curvature_checkpoint",
    "curvature_gated_odd_delta",
    "decide_coverage_state_bfa_curvature_audit",
    "evaluate_bfa_scalar_odd_curvature",
    "one_sided_exact_sign_test",
]
