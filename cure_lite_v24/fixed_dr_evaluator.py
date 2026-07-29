"""Fixed, cache-only D_R evaluator for v24 bounded and Formal training.

The evaluator has no path, callback, loader, dataset, or split parameter.  It
can consume only the exact in-memory ``CoverageStateScalarCache`` supplied by
the already verified runner authorization.  Consequently it cannot open D_V
or D_T.  It reuses the frozen zero-level evaluator and PMOPE geometry, and
reports cache-state safety/optimization diagnostics rather than unseen-data
generalization evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final

import torch
from torch import Tensor, nn

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.coverage_state_device_cache import (
    prepare_coverage_state_device_cache,
)
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
)
from cure_lite.coverage_state_sobolev import (
    CoverageStateAbsoluteTargets,
    CoverageStatePairTargets,
    coverage_state_absolute_sobolev_loss_from_targets,
    coverage_state_pmope_pair_loss_from_targets,
)
from cure_lite.experiment.coverage_state_training import (
    coverage_state_model_fingerprint,
)
from cure_lite.experiment.coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    evaluate_coverage_state_zero_level_checkpoint,
)
from cure_lite.instances import instances_from_binary_mask
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
)
from tools.gcr_pacre_v24_protocol import (
    FP_COMPONENTS_PER_MP_LIMIT,
    PIXEL_FA_LIMIT,
    RAW_BACKGROUND_FA_LIMIT,
)

from .bounded_runner import (
    GCR_PACRE_CANDIDATE_ARM,
    GCR_PACRE_CONTROL_ARM,
    GCR_PACRE_FORCED_G1_MODE,
    GCR_PACRE_NATIVE_MODE,
    GCRPACREBoundedEvaluation,
    GCRPACREBoundedEvaluator,
)
from .formal_training import GCRPACREFormalTerminalEvaluator
from .gcr_pacre import (
    CURELiteGatedCommonResidualPACRELevelSet,
    CoverageStateGCRPACREFields,
)


FROZEN_GCR_PACRE_D_R_EVALUATOR_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-fixed-cache-only-D_R-evaluator-v1"
)
FROZEN_GCR_PACRE_D_R_METRIC_POLICY: Final = (
    "full-cache-state-zero-level-safety-and-population-PMOPE-v1"
)


class _ExactForwardAdapter(nn.Module):
    """Present one frozen native/G1 forward to the zero-level evaluator."""

    def __init__(self, model: nn.Module, forward_mode: str) -> None:
        super().__init__()
        self.model = model
        self.forward_mode = forward_mode

    @property
    def config(self):
        return self.model.config

    @property
    def feature_stride(self) -> int:
        return int(self.model.feature_stride)

    @property
    def feature_channels(self) -> int:
        return int(self.model.feature_channels)

    def forward(self, feature: Tensor, occupancy: Tensor) -> Tensor:
        if self.forward_mode == GCR_PACRE_NATIVE_MODE:
            return self.model(feature, occupancy)
        if (
            self.forward_mode == GCR_PACRE_FORCED_G1_MODE
            and type(self.model)
            is CURELiteGatedCommonResidualPACRELevelSet
        ):
            return self.model.forward_forced_unit_gate(feature, occupancy)
        raise TypeError("fixed evaluator received an invalid forward mode")


@dataclass(frozen=True)
class _PopulationEvaluation:
    field: Tensor
    completion: Tensor
    final: Tensor
    pmope: float
    target_role_violation: float
    background_role_violation: float
    gate_summary: dict[str, object] | None


def _absolute_targets(store) -> CoverageStateAbsoluteTargets:
    return CoverageStateAbsoluteTargets(
        target_field=store.target_field,
        integration_measure=store.integration_measure,
        field_valid_mask=store.field_valid_mask,
        loss_valid_mask=store.loss_valid_mask,
        focus_support=store.focus_support,
        focus_support_field=store.focus_support_field,
    )


def _pair_targets(store) -> CoverageStatePairTargets:
    return CoverageStatePairTargets(
        target_field_plus=store.joint_target_field_plus,
        target_field_minus=store.joint_target_field_minus,
        focus_support=store.joint_focus_support,
        focus_support_field=store.joint_focus_support_field,
        integration_measure=store.joint_integration_measure,
        valid_mask=store.joint_valid_mask,
    )


def _weighted_role_mean(
    violation: Tensor,
    target_field: Tensor,
    integration_measure: Tensor,
    valid_mask: Tensor,
    *,
    target_role: bool,
) -> float:
    role = (
        (target_field < 0.0)
        if target_role
        else (target_field > 0.0)
    ) & valid_mask
    weight = integration_measure * role.to(integration_measure.dtype)
    denominator = weight.sum()
    if float(denominator.item()) <= 0.0:
        raise ValueError("fixed PMOPE evaluator lacks a required role")
    value = float(((violation * weight).sum() / denominator).item())
    if not isfinite(value):
        raise FloatingPointError("fixed PMOPE role violation is non-finite")
    return value


def _gate_statistics(
    values: tuple[tuple[Tensor, Tensor, Tensor, Tensor], ...],
) -> dict[str, object]:
    target_gates: list[Tensor] = []
    background_gates: list[Tensor] = []
    target_energies: list[Tensor] = []
    background_energies: list[Tensor] = []
    gates: list[Tensor] = []
    for gate, energy, target_field, valid_mask in values:
        if (
            tuple(gate.shape) != tuple(target_field.shape)
            or tuple(energy.shape) != tuple(gate.shape)
            or tuple(valid_mask.shape) != tuple(gate.shape)
        ):
            raise ValueError("gate role grids are not aligned")
        target = (target_field < 0.0) & valid_mask
        background = (target_field > 0.0) & valid_mask
        if bool(target.any()):
            target_gates.append(gate[target])
            target_energies.append(energy[target])
        if bool(background.any()):
            background_gates.append(gate[background])
            background_energies.append(energy[background])
        gates.append(gate[valid_mask])
    if (
        not target_gates
        or not background_gates
        or any(
            not bool(torch.isfinite(value).all())
            for value in (
                *target_gates,
                *background_gates,
                *target_energies,
                *background_energies,
            )
        )
    ):
        raise ValueError("gate target/background distributions are incomplete")
    all_gate = torch.cat(gates)
    target_gate = torch.cat(target_gates)
    background_gate = torch.cat(background_gates)
    target_energy = torch.cat(target_energies)
    background_energy = torch.cat(background_energies)
    if bool(torch.any((all_gate < 0.0) | (all_gate > 2.0))):
        raise ValueError("GCR gate left its closed machine interval")

    def summary(value: Tensor) -> dict[str, object]:
        return {
            "count": value.numel(),
            "minimum": float(value.amin().item()),
            "maximum": float(value.amax().item()),
            "mean": float(value.mean().item()),
        }

    return {
        "schema_version": "cure-lite-v24-gcr-pacre-gate-role-summary-v1",
        "endpoint_counts": {
            "G_equal_0": int(torch.count_nonzero(all_gate == 0.0).item()),
            "G_equal_2": int(torch.count_nonzero(all_gate == 2.0).item()),
            "G_strict_interior": int(
                torch.count_nonzero(
                    (all_gate > 0.0) & (all_gate < 2.0)
                ).item()
            ),
        },
        "target_G": summary(target_gate),
        "background_G": summary(background_gate),
        "target_E": summary(target_energy),
        "background_E": summary(background_energy),
    }


def _forward_with_gate(
    model: nn.Module,
    feature: Tensor,
    occupancy: Tensor,
    *,
    forward_mode: str,
) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
    if type(model) is CURELitePACREVerifierCorrectedLevelSet:
        if forward_mode != GCR_PACRE_NATIVE_MODE:
            raise ValueError("v23 control has no forced-G1 mode")
        field = model(feature, occupancy)
        return field, None
    if type(model) is not CURELiteGatedCommonResidualPACRELevelSet:
        raise TypeError("fixed evaluator accepts only exact v23/v24 models")
    fields = model.forward_fields(feature, occupancy)
    if type(fields) is not CoverageStateGCRPACREFields:
        raise TypeError("GCR-PACRE returned the wrong fields object")
    field = (
        fields.field
        if forward_mode == GCR_PACRE_NATIVE_MODE
        else model.pixel_shuffle(
            model.config.field_amplitude
            + fields.residual_odd_interaction
        ).contiguous()
        if forward_mode == GCR_PACRE_FORCED_G1_MODE
        else None
    )
    if field is None:
        raise ValueError("fixed evaluator received an unknown forward mode")
    effective_gate = (
        fields.common_gate
        if forward_mode == GCR_PACRE_NATIVE_MODE
        else torch.ones_like(fields.common_gate)
    )
    gate = model.pixel_shuffle(effective_gate).contiguous()
    energy = model.pixel_shuffle(fields.common_even_energy).contiguous()
    return field, (gate, energy)


def _population_evaluation(
    model: nn.Module,
    cache: CoverageStateScalarCache,
    *,
    forward_mode: str,
) -> _PopulationEvaluation:
    device = next(model.parameters()).device
    packed = prepare_coverage_state_device_cache(cache, device=device)
    packed.verify_unchanged()
    natural = packed.natural
    pairs = packed.pairs
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=False),
    ):
        natural_field, natural_gate = _forward_with_gate(
            model,
            natural.feature,
            natural.occupancy,
            forward_mode=forward_mode,
        )
        pair_count = len(pairs.pair_ids)
        pair_field, pair_gate = _forward_with_gate(
            model,
            torch.cat((pairs.feature, pairs.feature), dim=0).contiguous(),
            torch.cat(
                (pairs.occupancy_plus, pairs.occupancy_minus),
                dim=0,
            ).contiguous(),
            forward_mode=forward_mode,
        )
        field_plus, field_minus = torch.split(
            pair_field,
            (pair_count, pair_count),
            dim=0,
        )
        natural_loss = coverage_state_absolute_sobolev_loss_from_targets(
            natural_field,
            _absolute_targets(natural),
            config=cache.sobolev_config,
            validate=False,
        )
        pair_targets = _pair_targets(pairs)
        pair_loss = coverage_state_pmope_pair_loss_from_targets(
            field_plus,
            field_minus,
            pair_targets,
            config=cache.sobolev_config,
            validate=False,
        )

    miss = torch.tensor(
        [kind == "factual_miss" for kind in natural.state_kinds],
        dtype=torch.bool,
        device=device,
    )
    no_miss = ~miss
    clean = torch.tensor(
        [role == "clean_positive" for role in pairs.optimizer_roles],
        dtype=torch.bool,
        device=device,
    )
    component = torch.tensor(
        [role == "component_null" for role in pairs.optimizer_roles],
        dtype=torch.bool,
        device=device,
    )
    if not all(
        bool(value.any()) for value in (miss, no_miss, clean, component)
    ):
        raise ValueError("fixed evaluator lacks a frozen optimizer role")
    total = (
        natural_loss.per_state_loss[miss].mean()
        + natural_loss.per_state_loss[no_miss].mean()
        + 0.5
        * (
            pair_loss.per_state_loss[clean].mean()
            + pair_loss.per_state_loss[component].mean()
        )
    )
    pmope = float(total.item())
    if not isfinite(pmope):
        raise FloatingPointError("full-population PMOPE is non-finite")

    violations = torch.cat(
        (pair_loss.violation_plus, pair_loss.violation_minus),
        dim=0,
    )
    pair_target_fields = torch.cat(
        (
            pair_targets.target_field_plus,
            pair_targets.target_field_minus,
        ),
        dim=0,
    )
    pair_measure = torch.cat(
        (
            pair_targets.integration_measure,
            pair_targets.integration_measure,
        ),
        dim=0,
    )
    pair_valid = torch.cat(
        (pair_targets.valid_mask, pair_targets.valid_mask),
        dim=0,
    )

    gate_summary = None
    if natural_gate is not None and pair_gate is not None:
        natural_gate_value, natural_energy = natural_gate
        pair_gate_value, pair_energy = pair_gate
        plus_gate, minus_gate = torch.split(
            pair_gate_value,
            (pair_count, pair_count),
            dim=0,
        )
        plus_energy, minus_energy = torch.split(
            pair_energy,
            (pair_count, pair_count),
            dim=0,
        )
        gate_summary = _gate_statistics(
            (
                (
                    natural_gate_value,
                    natural_energy,
                    natural.target_field,
                    natural.field_valid_mask,
                ),
                (
                    plus_gate,
                    plus_energy,
                    pair_targets.target_field_plus,
                    pair_targets.valid_mask,
                ),
                (
                    minus_gate,
                    minus_energy,
                    pair_targets.target_field_minus,
                    pair_targets.valid_mask,
                ),
            )
        )

    completion = (
        (natural_field < 0.0) & ~natural.occupancy
    ).contiguous()
    final = (natural.occupancy | completion).contiguous()
    return _PopulationEvaluation(
        field=natural_field.detach().to("cpu").contiguous(),
        completion=completion.detach().to("cpu").contiguous(),
        final=final.detach().to("cpu").contiguous(),
        pmope=pmope,
        target_role_violation=_weighted_role_mean(
            violations,
            pair_target_fields,
            pair_measure,
            pair_valid,
            target_role=True,
        ),
        background_role_violation=_weighted_role_mean(
            violations,
            pair_target_fields,
            pair_measure,
            pair_valid,
            target_role=False,
        ),
        gate_summary=gate_summary,
    )


def _cache_state_metrics(
    cache: CoverageStateScalarCache,
    population: _PopulationEvaluation,
) -> tuple[int, int, float, float, float, float, float, float, bool]:
    occupancy = torch.cat(
        tuple(value.record.occupancy for value in cache.natural_records),
        dim=0,
    )
    target = torch.cat(
        tuple(value.targets.focus_support for value in cache.natural_records),
        dim=0,
    )
    valid = torch.cat(
        tuple(value.record.valid_mask for value in cache.natural_records),
        dim=0,
    )
    desired = (occupancy | target) & valid
    final = population.final & valid
    intersection = (final & desired).flatten(1).sum(dim=1)
    union = (final | desired).flatten(1).sum(dim=1)
    per_iou = torch.where(
        union > 0,
        intersection.to(torch.float64) / union.to(torch.float64),
        torch.ones_like(union, dtype=torch.float64),
    )
    global_union = int(union.sum().item())
    miou = (
        int(intersection.sum().item()) / global_union
        if global_union
        else 1.0
    )
    niou = float(per_iou.mean().item())
    miss = torch.tensor(
        [
            value.record.state_kind == "factual_miss"
            for value in cache.natural_records
        ],
        dtype=torch.bool,
    )
    recovered_rows = (
        (population.completion & target).flatten(1).any(dim=1) & miss
    )
    recovered = int(torch.count_nonzero(recovered_rows).item())
    miss_count = int(torch.count_nonzero(miss).item())
    pd = recovered / miss_count if miss_count else 1.0
    writable_background = valid & ~occupancy & ~target
    false = population.completion & writable_background
    false_pixels = int(torch.count_nonzero(false).item())
    total_pixels = int(torch.count_nonzero(valid).item())
    writable_background_pixels = int(
        torch.count_nonzero(writable_background).item()
    )
    false_components = sum(
        len(
            instances_from_binary_mask(
                row[0],
                connectivity=8,
                min_area=1,
            ).instances
        )
        for row in false
    )
    pixel_fa = false_pixels / total_pixels if total_pixels else 0.0
    # This D_R scalar cache does not carry a full-image GT evaluator.  The
    # only honest raw-background quantity available here is therefore the
    # completion rate on the exact writable cache background.  It is never
    # presented as Formal D_V/D_T generalization evidence.
    raw_background_fa = (
        false_pixels / writable_background_pixels
        if writable_background_pixels
        else 0.0
    )
    fp_components_per_mp = (
        false_components / (total_pixels / 1_000_000.0)
        if total_pixels
        else 0.0
    )
    retention = 1.0
    budget_violation = not (
        retention == 1.0
        and pixel_fa <= PIXEL_FA_LIMIT
        and raw_background_fa <= RAW_BACKGROUND_FA_LIMIT
        and fp_components_per_mp <= FP_COMPONENTS_PER_MP_LIMIT
    )
    return (
        recovered,
        recovered,
        miou,
        niou,
        pd,
        pixel_fa,
        raw_background_fa,
        fp_components_per_mp,
        budget_violation,
    )


class FrozenGCRPACREDREvaluator(
    GCRPACREBoundedEvaluator,
    GCRPACREFormalTerminalEvaluator,
):
    """Exact stateless evaluator accepted by the real bounded/Formal CLIs."""

    __slots__ = ()

    @property
    def evaluator_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": FROZEN_GCR_PACRE_D_R_EVALUATOR_SCHEMA,
                "metric_policy": FROZEN_GCR_PACRE_D_R_METRIC_POLICY,
                "split": "D_R",
                "input_representation": (
                    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                ),
                "zero_level_threshold": 0.0,
                "cache_only": True,
                "threshold_search_performed": False,
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            }
        )

    def evaluate(
        self,
        model: nn.Module,
        cache: CoverageStateScalarCache,
        *,
        arm: str,
        checkpoint: str,
        forward_mode: str,
    ) -> GCRPACREBoundedEvaluation:
        if type(cache) is not CoverageStateScalarCache:
            raise TypeError("fixed evaluator requires exact scalar cache")
        cache.verify_unchanged()
        if cache.raw_catalog.split != "D_R":
            raise PermissionError("fixed evaluator permits only D_R")
        if (
            arm not in {GCR_PACRE_CONTROL_ARM, GCR_PACRE_CANDIDATE_ARM}
            or checkpoint not in {"initial", "terminal"}
        ):
            raise ValueError("fixed evaluator identity changed")
        expected_type = (
            CURELitePACREVerifierCorrectedLevelSet
            if arm == GCR_PACRE_CONTROL_ARM
            else CURELiteGatedCommonResidualPACRELevelSet
        )
        if type(model) is not expected_type:
            raise TypeError("fixed evaluator arm/model type differs")
        if (
            arm == GCR_PACRE_CONTROL_ARM
            and forward_mode != GCR_PACRE_NATIVE_MODE
        ):
            raise ValueError("v23 control cannot use forced G1")
        before = coverage_state_model_fingerprint(model)
        was_training = model.training
        try:
            model.eval()
            adapter = _ExactForwardAdapter(model, forward_mode).eval()
            zero = evaluate_coverage_state_zero_level_checkpoint(
                adapter,
                cache,
                device=next(model.parameters()).device,
                config=CoverageStateZeroLevelEvaluationConfig(
                    input_representation=(
                        COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                    )
                ),
            )
            population = _population_evaluation(
                model,
                cache,
                forward_mode=forward_mode,
            )
        finally:
            model.train(was_training)
        if coverage_state_model_fingerprint(model) != before:
            raise RuntimeError("fixed evaluator mutated model state")
        (
            true_targets,
            recovered,
            miou,
            niou,
            pd,
            pixel_fa,
            raw_background_fa,
            fp_components_per_mp,
            budget_violation,
        ) = _cache_state_metrics(cache, population)
        zero_crossed = sum(
            value.target_recovered is True
            for value in zero.natural_diagnostics
        ) + sum(
            value.pair_kind == "clean_positive"
            and value.minus_added_target_negative_pixels > 0
            for value in zero.pair_diagnostics
        )
        false_completion = sum(
            (
                value.negative_pixels
                > value.focus_target_negative_pixels
            )
            for value in zero.natural_diagnostics
        ) + sum(
            (
                value.invalid_completion_pixels_plus > 0
                or value.invalid_completion_pixels_minus > 0
                or (
                    value.new_completion_outside_added_target_pixels
                    not in {None, 0}
                )
                or (
                    value.pair_kind != "clean_positive"
                    and value.new_completion_pixels > 0
                )
            )
            for value in zero.pair_diagnostics
        )
        field_fingerprint = stable_fingerprint(
            {
                "forward_mode": forward_mode,
                "zero_level_fields": [
                    {
                        "state_id": value.state_id,
                        "field_fingerprint": value.field_fingerprint,
                    }
                    for value in zero.state_ledger
                ],
                "gate_summary": population.gate_summary,
            }
        )
        role_prediction_fingerprint = stable_fingerprint(
            {
                "forward_mode": forward_mode,
                "zero_level_predictions": [
                    {
                        "state_id": value.state_id,
                        "completion_fingerprint": (
                            value.completion_fingerprint
                        ),
                        "final_fingerprint": value.final_fingerprint,
                    }
                    for value in zero.state_ledger
                ],
            }
        )
        return GCRPACREBoundedEvaluation(
            true_targets=true_targets,
            recovered_anchor_misses=recovered,
            mIoU=miou,
            nIoU=niou,
            pd=pd,
            retention=1.0,
            pixel_fa=pixel_fa,
            raw_background_fa=raw_background_fa,
            fp_components_per_mp=fp_components_per_mp,
            budget_violation=budget_violation,
            PMOPE=population.pmope,
            target_role_violation=population.target_role_violation,
            background_role_violation=(
                population.background_role_violation
            ),
            zero_crossed_target_states=zero_crossed,
            false_completion_states=false_completion,
            gate_role_distributions_present=(
                population.gate_summary is not None
            ),
            gate_role_distribution_json=(
                None
                if population.gate_summary is None
                else canonical_json(population.gate_summary)
            ),
            field_fingerprint=field_fingerprint,
            role_prediction_fingerprint=role_prediction_fingerprint,
        )

    def evaluate_terminal_d_r(
        self,
        model: nn.Module,
        cache: CoverageStateScalarCache,
        *,
        seed: int,
        role: str,
    ) -> dict[str, object]:
        if (seed, role) not in {
            (42, "primary"),
            (43, "training_integrity_only"),
        }:
            raise ValueError("fixed Formal evaluator seed/role changed")
        value = self.evaluate(
            model,
            cache,
            arm=GCR_PACRE_CANDIDATE_ARM,
            checkpoint="terminal",
            forward_mode=GCR_PACRE_NATIVE_MODE,
        )
        return {
            "schema_version": (
                "cure-lite-v24-gcr-pacre-formal800-cache-only-D_R-"
                "terminal-metrics-v1"
            ),
            "seed": seed,
            "role": role,
            "evidence_scope": (
                "D_R_full_cache_training_integrity_not_generalization"
            ),
            "metrics": value.canonical_payload(),
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }


__all__ = [
    "FROZEN_GCR_PACRE_D_R_EVALUATOR_SCHEMA",
    "FROZEN_GCR_PACRE_D_R_METRIC_POLICY",
    "FrozenGCRPACREDREvaluator",
]
