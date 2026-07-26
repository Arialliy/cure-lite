#!/usr/bin/env python3
"""Run the dataset-free CURE-Lite CC-SEA v8 toy model gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import (  # noqa: E402
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.config import LossConfig  # noqa: E402
from cure_lite.conservative_factorized_decoder import (  # noqa: E402
    CURELiteConservativeFactorizedDecoder,
    coverage_conserving_phase_evidence,
)
from cure_lite.crossing_factorized_decoder import (  # noqa: E402
    crossing_recoverable_evidence,
)
from cure_lite.experiment.conservative_toy_inputs import (  # noqa: E402
    CONSERVATIVE_TOY_CASES,
    LEGACY_FAMILY,
    build_conservative_toy_case,
)
from cure_lite.losses import CURELiteLoss  # noqa: E402
from cure_lite.paired_outcome_losses import (  # noqa: E402
    OutcomeCompleteTransitionLoss,
)
from cure_lite.train.paired_outcome_step import (  # noqa: E402
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import (  # noqa: E402
    _paired_endpoint_logits,
)


SCHEMA_VERSION = "cure-lite-cc-sea-v8-toy-gate-result-v1"
METHOD_ID = "cc_sea_v8"
FROZEN_SEED = 7817
FROZEN_UPDATES = 320
FROZEN_LEARNING_RATE = 0.004
EXPECTED_PARAMETER_TENSORS = 6
EXPECTED_PARAMETER_COUNT = 2593
THRESHOLDS = {
    "total_loss_max_exclusive": 0.10,
    "plus_completion_min_exclusive": 0.95,
    "plus_background_max_exclusive": 0.05,
    "factual_miss_target_min_exclusive": 0.95,
    "factual_miss_background_max_exclusive": 0.05,
    "factual_no_miss_max_exclusive": 0.05,
    "clean_D_mean_min_inclusive": 0.80,
    "clean_H_max_abs_max_inclusive": 0.05,
    "clean_G_max_abs_max_inclusive": 0.05,
    "component_H_max_abs_max_inclusive": 0.05,
    "component_G_max_abs_max_inclusive": 0.05,
}
_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conserving_subpixel_evidence_allocation_v8"
)
_TOY_CONFIG = _PROTOCOL / "toy_config.json"


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _verify_internal_fingerprint(
    value: dict[str, object],
    *,
    field: str,
    name: str,
) -> str:
    unsigned = dict(value)
    observed = unsigned.pop(field, None)
    if not isinstance(observed, str):
        raise TypeError(f"{name}.{field} must be a string")
    if stable_fingerprint(unsigned) != observed:
        raise RuntimeError(f"{name} fingerprint differs")
    return observed


def _load_protocol_binding() -> dict[str, object]:
    """Strictly bind the frozen proposal, config, and v7 predecessor."""

    config = _load_object(_TOY_CONFIG, name="v8 toy config")
    config_fingerprint = _verify_internal_fingerprint(
        config,
        field="config_fingerprint",
        name="v8 toy config",
    )
    if config.get("schema_version") != (
        "cure-lite-cc-sea-v8-toy-config-v1"
    ):
        raise RuntimeError("v8 toy config schema differs")
    if config.get("method_id") != METHOD_ID:
        raise RuntimeError("v8 toy config method differs")
    expected_cases = [
        {
            "family_id": family,
            "case_id": case_id,
            "clean_pixels": [list(pixel) for pixel in pixels],
        }
        for family, case_id, pixels in CONSERVATIVE_TOY_CASES
    ]
    if config.get("cases") != expected_cases:
        raise RuntimeError("v8 toy cases differ")
    if config.get("decoder") != {
        "feature_channels": 8,
        "feature_stride": 4,
        "width": 32,
        "groups": 8,
        "trunk_residual_scale": 0.5,
        "baseline_probability": 0.1,
        "vacancy_kernel_size": 3,
        "expected_parameter_tensors": EXPECTED_PARAMETER_TENSORS,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
    }:
        raise RuntimeError("v8 toy decoder contract differs")
    if config.get("optimization") != {
        "seed": FROZEN_SEED,
        "optimizer": "adam",
        "updates_per_case": FROZEN_UPDATES,
        "learning_rate": FROZEN_LEARNING_RATE,
        "weight_decay": 0.0,
        "loss": "unchanged_outcome_complete_transition_plus_absolute_factual",
        "training_step": "unchanged_outcome_complete_train_step",
        "automatic_retry_allowed": False,
    }:
        raise RuntimeError("v8 toy optimization contract differs")
    if config.get("thresholds") != THRESHOLDS:
        raise RuntimeError("v8 toy thresholds differ")
    if config.get("operator_thresholds") != {
        "float32_allocation_sum_max_abs_error": 1.0e-6,
        "float32_phase_contrast_sum_max_abs_error": 1.0e-5,
        "float32_mass_conservation_max_relative_error": 1.0e-6,
        "float64_coordinate_max_abs_error": 1.0e-12,
        "float64_allocation_change_min_exclusive": 1.0e-3,
    }:
        raise RuntimeError("v8 toy operator thresholds differ")
    if config.get("decision_rule") != {
        "required_passed_case_count": 6,
        "required_passed_family_count": 2,
        "per_case_all_checks_required": True,
        "mean_cannot_override_case_failure": True,
        "coordinate_audit_required": True,
        "numerical_audit_required": True,
        "pass_decision": "CC_SEA_V8_TOY_GATE_PASS",
        "fail_decision": "CC_SEA_V8_TOY_GATE_FAIL",
    }:
        raise RuntimeError("v8 toy decision rule differs")
    if config.get("execution_boundary") != {
        "D_R_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
        "detection_performance_allowed": False,
        "real_bounded_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "cross_detector_authorized": False,
    }:
        raise RuntimeError("v8 toy execution boundary differs")

    proposal_binding = config.get("proposal_binding")
    if not isinstance(proposal_binding, dict):
        raise TypeError("v8 proposal_binding must be an object")
    proposal_path = _ROOT / str(proposal_binding.get("repo_path"))
    if file_sha256(proposal_path) != proposal_binding.get("file_sha256"):
        raise RuntimeError("v8 proposal file hash differs")
    proposal = _load_object(proposal_path, name="v8 proposal")
    proposal_fingerprint = _verify_internal_fingerprint(
        proposal,
        field="proposal_fingerprint",
        name="v8 proposal",
    )
    if proposal_fingerprint != proposal_binding.get(
        "proposal_fingerprint"
    ):
        raise RuntimeError("v8 proposal fingerprint binding differs")
    if proposal.get("method_id") != METHOD_ID:
        raise RuntimeError("v8 proposal method differs")
    if config.get("operator") != proposal.get("single_mechanism"):
        raise RuntimeError("v8 proposal/config operator differs")
    inherited = config.get("inherited_primitive_binding")
    if inherited != proposal.get("inherited_primitive"):
        raise RuntimeError("v8 inherited primitive binding differs")
    if not isinstance(inherited, dict):
        raise TypeError("v8 inherited primitive binding must be an object")
    primitive_path = _ROOT / str(inherited.get("repo_path"))
    if file_sha256(primitive_path) != inherited.get("file_sha256"):
        raise RuntimeError("v8 inherited primitive source differs")

    design = proposal.get("design_document")
    if not isinstance(design, dict):
        raise TypeError("v8 design binding must be an object")
    design_path = _ROOT / str(design.get("repo_path"))
    if file_sha256(design_path) != design.get("file_sha256"):
        raise RuntimeError("v8 design document differs")
    predecessor = proposal.get("predecessor_v7")
    if not isinstance(predecessor, dict):
        raise TypeError("v8 predecessor binding must be an object")
    predecessor_path = _ROOT / str(predecessor.get("repo_path"))
    if file_sha256(predecessor_path) != predecessor.get("file_sha256"):
        raise RuntimeError("v8 predecessor closure hash differs")
    predecessor_value = _load_object(
        predecessor_path,
        name="v7 negative closure",
    )
    if (
        predecessor_value.get("receipt_fingerprint")
        != predecessor.get("receipt_fingerprint")
        or predecessor_value.get("decision") != predecessor.get("decision")
        or predecessor_value.get("decision")
        != "CR_LVEC_V7_BOUNDED_MODEL_CODE_GATE_FAIL"
    ):
        raise RuntimeError("v8 predecessor closure binding differs")
    return {
        "toy_config_repo_path": str(_TOY_CONFIG.relative_to(_ROOT)),
        "toy_config_file_sha256": file_sha256(_TOY_CONFIG),
        "toy_config_fingerprint": config_fingerprint,
        "proposal_repo_path": str(proposal_path.relative_to(_ROOT)),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_fingerprint": proposal_fingerprint,
        "design_document_file_sha256": design["file_sha256"],
        "predecessor_v7_closure_file_sha256": predecessor[
            "file_sha256"
        ],
        "predecessor_v7_closure_fingerprint": predecessor[
            "receipt_fingerprint"
        ],
        "inherited_primitive_file_sha256": inherited["file_sha256"],
    }


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return (
        str(tensor.dtype).encode("utf-8")
        + repr(tuple(tensor.shape)).encode("utf-8")
        + tensor.numpy().tobytes()
    )


def _decoder_fingerprint(
    decoder: CURELiteConservativeFactorizedDecoder,
) -> str:
    digest = hashlib.sha256()
    for name, value in decoder.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _inputs_fingerprint(outcome: object, factual: dict[str, object]) -> str:
    digest = hashlib.sha256()
    tensors = (
        outcome.pair_batch.feature,
        outcome.pair_batch.occupancy_plus,
        outcome.pair_batch.occupancy_minus,
        outcome.pair_batch.label_increment,
        outcome.pair_batch.image_valid_mask,
        outcome.completion_plus,
        outcome.completion_minus,
        outcome.gt_union,
        outcome.intervention_footprint,
        factual["factual_miss"].feature,
        factual["factual_miss"].occupancy,
        factual["factual_miss"].target,
        factual["factual_no_miss"].feature,
        factual["factual_no_miss"].occupancy,
        factual["factual_no_miss"].target,
    )
    for value in tensors:
        digest.update(_tensor_bytes(value))
    for values in (
        outcome.pair_batch.pair_ids,
        outcome.pair_batch.sample_ids,
        outcome.pair_batch.group_ids,
        outcome.pair_batch.pair_kinds,
    ):
        digest.update(canonical_json(list(values)).encode("utf-8"))
    return digest.hexdigest()


def _maximum(value: Tensor) -> float:
    return 0.0 if value.numel() == 0 else float(value.max().detach())


def _minimum(value: Tensor) -> float:
    return 0.0 if value.numel() == 0 else float(value.min().detach())


def _operator_audit(
    decoder: CURELiteConservativeFactorizedDecoder,
    outcome: object,
) -> dict[str, object]:
    batch = outcome.pair_batch
    with torch.no_grad():
        plus = decoder.forward_fields(batch.feature, batch.occupancy_plus)
        minus = decoder.forward_fields(batch.feature, batch.occupancy_minus)
        identity = decoder(batch.feature, batch.occupancy_plus)
        identity_again = decoder(
            batch.feature,
            batch.occupancy_plus.clone(),
        )
        count_release = (
            plus.local_occupancy_count
            - minus.local_occupancy_count
        )
        phase_count = int(plus.raw_phase_evidence.shape[1])
        stride = decoder.feature_stride
        changed = count_release > 0.0
        changed_output = F.pixel_shuffle(
            changed.expand(-1, phase_count, -1, -1).to(torch.float32),
            stride,
        ).to(torch.bool)
        if plus.field_resize_applied:
            changed_output = F.interpolate(
                changed_output.to(torch.float32),
                size=plus.output_size,
                mode="nearest",
            ).to(torch.bool)
        delta = minus.logits - plus.logits
        probability_delta = (
            torch.sigmoid(minus.logits)
            - torch.sigmoid(plus.logits)
        )
        outside = ~changed_output
        allocation_delta = (
            plus.phase_allocation - minus.phase_allocation
        ).abs()

        conservation_absolute_errors = []
        conservation_relative_errors = []
        allocation_errors = []
        state_equations = []
        for fields in (plus, minus):
            phase_contrast = (
                fields.raw_phase_evidence
                - fields.common_mode_phase_evidence
            )
            conservation_error = (
                fields.allocated_phase_evidence.sum(
                    dim=1,
                    keepdim=True,
                )
                - fields.evidence_budget
            ).abs()
            conservation_absolute_errors.append(conservation_error.max())
            conservation_relative_errors.append(
                (
                    conservation_error
                    / fields.evidence_budget.abs().clamp_min(1.0)
                ).max()
            )
            allocation_errors.append(
                (
                    fields.phase_allocation.sum(dim=1, keepdim=True)
                    - 1.0
                ).abs().max()
            )
            expected_budget = crossing_recoverable_evidence(
                fields.budget_margin
            )
            state_equations.append(
                {
                    "common_mode_max_abs_error": float(
                        (
                            fields.raw_phase_evidence.mean(
                                dim=1,
                                keepdim=True,
                            )
                            - fields.common_mode_phase_evidence
                        ).abs().max()
                    ),
                    "phase_contrast_sum_max_abs_error": float(
                        phase_contrast.sum(dim=1, keepdim=True).abs().max()
                    ),
                    "allocation_sum_max_abs_error": float(
                        (
                            fields.phase_allocation.sum(
                                dim=1,
                                keepdim=True,
                            )
                            - 1.0
                        ).abs().max()
                    ),
                    "allocation_min": float(
                        fields.phase_allocation.min()
                    ),
                    "evidence_min": float(
                        fields.allocated_phase_evidence.min()
                    ),
                    "mass_conservation_max_abs_error": float(
                        conservation_error.max()
                    ),
                    "mass_conservation_max_relative_error": float(
                        (
                            conservation_error
                            / fields.evidence_budget.abs().clamp_min(1.0)
                        ).max()
                    ),
                    "budget_forward_exact": torch.equal(
                        fields.evidence_budget,
                        expected_budget,
                    ),
                    "logit_composition_exact": torch.equal(
                        fields.logits,
                        fields.baseline_logits + fields.evidence,
                    ),
                }
            )

        response = outcome.response_stratum
        response_outside_support = response & ~changed_output
        budget_strict_by_pair = (
            (
                minus.evidence_budget > plus.evidence_budget
            )
            .flatten(1)
            .any(dim=1)
        )
        phase_mass_delta = (
            minus.allocated_phase_evidence.sum(dim=1, keepdim=True)
            - plus.allocated_phase_evidence.sum(dim=1, keepdim=True)
        )
        budget_delta = minus.evidence_budget - plus.evidence_budget
        delta_mass_error = (phase_mass_delta - budget_delta).abs()
        delta_mass_relative_error = (
            delta_mass_error / budget_delta.abs().clamp_min(1.0)
        )

    checks = {
        "identity_exact": torch.equal(identity, identity_again),
        "allocation_is_occupancy_invariant": (
            float(allocation_delta.max()) == 0.0
        ),
        "budget_release_is_monotone": bool(
            torch.all(minus.evidence_budget >= plus.evidence_budget)
        ),
        "budget_release_is_strict_per_pair": bool(
            torch.all(budget_strict_by_pair)
        ),
        "allocated_release_is_monotone": bool(
            torch.all(
                minus.allocated_phase_evidence
                >= plus.allocated_phase_evidence
            )
        ),
        "outside_count_support_exact": torch.equal(
            delta[outside],
            torch.zeros_like(delta[outside]),
        ),
        "outside_count_support_probability_exact": torch.equal(
            probability_delta[outside],
            torch.zeros_like(probability_delta[outside]),
        ),
        "response_is_inside_count_support": not bool(
            response_outside_support.any()
        ),
        "phase_allocation_sums_to_one": (
            max(float(value) for value in allocation_errors) <= 1.0e-6
        ),
        "phase_evidence_sums_to_budget": (
            max(
                float(value) for value in conservation_relative_errors
            )
            <= 1.0e-6
        ),
        "phase_mass_delta_equals_budget_delta": (
            float(delta_mass_relative_error.max()) <= 1.0e-6
        ),
        "state_equations_exact": all(
            state["common_mode_max_abs_error"] == 0.0
            and state["phase_contrast_sum_max_abs_error"] <= 1.0e-5
            and state["allocation_sum_max_abs_error"] <= 1.0e-6
            and state["allocation_min"] >= 0.0
            and state["evidence_min"] >= 0.0
            and state["mass_conservation_max_relative_error"] <= 1.0e-6
            and state["budget_forward_exact"] is True
            and state["logit_composition_exact"] is True
            for state in state_equations
        ),
        "all_fields_finite": all(
            bool(torch.isfinite(value).all())
            for value in (
                plus.common_mode_phase_evidence,
                plus.budget_margin,
                plus.evidence_budget,
                plus.phase_allocation,
                plus.evidence,
                plus.logits,
                minus.common_mode_phase_evidence,
                minus.budget_margin,
                minus.evidence_budget,
                minus.phase_allocation,
                minus.evidence,
                minus.logits,
            )
        ),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "identity_max_abs_delta": _maximum((identity - identity_again).abs()),
        "allocation_occupancy_delta_max": float(allocation_delta.max()),
        "conservation_error_max": max(
            float(value) for value in conservation_absolute_errors
        ),
        "conservation_relative_error_max": max(
            float(value) for value in conservation_relative_errors
        ),
        "allocation_sum_error_max": max(
            float(value) for value in allocation_errors
        ),
        "outside_count_support_max_abs_delta": _maximum(
            delta[outside].abs()
        ),
        "outside_count_support_probability_max_abs_delta": _maximum(
            probability_delta[outside].abs()
        ),
        "phase_mass_delta_budget_delta_max_relative_error": float(
            delta_mass_relative_error.max()
        ),
        "state_equation_contract": {
            "plus": state_equations[0],
            "minus": state_equations[1],
        },
        "changed_support_pixels": int(changed_output.sum()),
        "unchanged_support_pixels": int(outside.sum()),
        "maximum_absolute_budget_margin": max(
            float(plus.budget_margin.abs().max()),
            float(minus.budget_margin.abs().max()),
        ),
    }


def _orthogonality_audit() -> dict[str, object]:
    raw = torch.tensor(
        [[[[1.0]], [[-1.0]], [[0.5]], [[-0.5]]]],
        dtype=torch.float64,
    )
    burden = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    base = coverage_conserving_phase_evidence(raw, burden)
    shifted = coverage_conserving_phase_evidence(raw + 2.0, burden)
    zero_mean = torch.tensor(
        [[[[1.5]], [[-1.5]], [[0.5]], [[-0.5]]]],
        dtype=torch.float64,
    )
    contrasted = coverage_conserving_phase_evidence(
        raw + zero_mean,
        burden,
    )
    released_raw = raw + 0.5
    occupied = coverage_conserving_phase_evidence(
        released_raw,
        torch.full_like(burden, float(torch.log(torch.tensor(2.0)))),
    )
    released = coverage_conserving_phase_evidence(
        released_raw,
        burden,
    )
    shifted_contrast_error = (
        (raw + 2.0 - shifted[0]) - (raw - base[0])
    ).abs().max()
    contrasted_allocation_change = (contrasted[3] - base[3]).abs().max()
    simplex_error = max(
        float((state[3].sum(dim=1, keepdim=True) - 1.0).abs().max())
        for state in (base, shifted, contrasted, occupied, released)
    )
    mass_error = max(
        float((state[4].sum(dim=1, keepdim=True) - state[2]).abs().max())
        for state in (base, shifted, contrasted, occupied, released)
    )
    checks = {
        "common_shift_changes_only_common_mode_and_budget": (
            torch.equal(shifted[0], base[0] + 2.0)
            and torch.allclose(shifted[3], base[3], rtol=0.0, atol=1.0e-15)
            and float(shifted_contrast_error) <= 1.0e-12
            and bool(torch.all(shifted[2] > base[2]))
        ),
        "zero_mean_contrast_preserves_common_mode": torch.equal(
            contrasted[0], base[0]
        ),
        "zero_mean_contrast_preserves_budget": torch.equal(
            contrasted[2], base[2]
        ),
        "zero_mean_contrast_changes_allocation": (
            float(contrasted_allocation_change) > 1.0e-3
        ),
        "occupancy_release_preserves_allocation": torch.equal(
            occupied[3], released[3]
        ),
        "occupancy_release_strictly_increases_budget": bool(
            torch.all(released[2] > occupied[2])
        ),
        "simplex_is_conserved": simplex_error <= 1.0e-12,
        "mass_is_conserved": mass_error <= 1.0e-12,
    }
    return {
        "checks": checks,
        "common_shift": {
            "common_mode_delta": float((shifted[0] - base[0]).item()),
            "phase_contrast_max_abs_error": float(shifted_contrast_error),
            "allocation_max_abs_error": float(
                (shifted[3] - base[3]).abs().max()
            ),
        },
        "zero_mean_contrast": {
            "common_mode_max_abs_error": float(
                (contrasted[0] - base[0]).abs().max()
            ),
            "budget_max_abs_error": float(
                (contrasted[2] - base[2]).abs().max()
            ),
            "allocation_max_abs_change": float(
                contrasted_allocation_change
            ),
        },
        "occupancy_release": {
            "allocation_max_abs_error": float(
                (occupied[3] - released[3]).abs().max()
            ),
            "budget_delta": float((released[2] - occupied[2]).item()),
        },
        "simplex_max_abs_error": simplex_error,
        "mass_conservation_max_abs_error": mass_error,
        "all_pass": all(checks.values()),
    }


def _numerical_audit() -> dict[str, object]:
    negative = torch.tensor([-80.0], requires_grad=True)
    negative_value = crossing_recoverable_evidence(negative)
    negative_value.sum().backward()
    first_zero_failed = False
    try:
        crossing_recoverable_evidence(torch.tensor([-104.0]))
    except ValueError:
        first_zero_failed = True
    largest = torch.tensor([88.0], requires_grad=True)
    largest_value = crossing_recoverable_evidence(largest)
    largest_value.sum().backward()
    first_nonfinite_failed = False
    try:
        crossing_recoverable_evidence(torch.tensor([89.0]))
    except ValueError:
        first_nonfinite_failed = True
    checks = {
        "negative_recovery_finite_nonzero": (
            negative.grad is not None
            and bool(torch.isfinite(negative.grad).all())
            and bool(torch.all(negative.grad > 0.0))
        ),
        "zero_recovery_fails_fast": first_zero_failed,
        "largest_supported_positive_is_finite": (
            largest.grad is not None
            and bool(torch.isfinite(largest_value).all())
            and bool(torch.isfinite(largest.grad).all())
        ),
        "first_nonfinite_positive_fails_fast": first_nonfinite_failed,
    }
    return {"checks": checks, "all_pass": all(checks.values())}


def _case(
    family_id: str,
    case_id: str,
    clean_pixels: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FROZEN_SEED)
        outcome, factual = build_conservative_toy_case(
            family_id,
            clean_pixels,
        )
        decoder = CURELiteConservativeFactorizedDecoder(
            feature_channels=8,
            feature_stride=4,
        )
        initial_decoder_fingerprint = _decoder_fingerprint(decoder)
        initial_operator = _operator_audit(decoder, outcome)
        named_parameters = tuple(decoder.named_parameters())
        if len(named_parameters) != EXPECTED_PARAMETER_TENSORS:
            raise AssertionError("v8 toy parameter tensor count differs")
        if sum(value.numel() for _, value in named_parameters) != (
            EXPECTED_PARAMETER_COUNT
        ):
            raise AssertionError("v8 toy parameter count differs")

        absolute = CURELiteLoss()
        criterion = OutcomeCompleteTransitionLoss(LossConfig())
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=FROZEN_LEARNING_RATE,
        )
        forward_batches: list[int] = []

        def observe_batch(_module: object, args: tuple[object, ...]) -> None:
            forward_batches.append(int(args[0].shape[0]))

        handle = decoder.register_forward_pre_hook(observe_batch)
        try:
            initial_plus, initial_minus = _paired_endpoint_logits(
                decoder,
                feature=outcome.pair_batch.feature,
                occupancy_plus=outcome.pair_batch.occupancy_plus,
                occupancy_minus=outcome.pair_batch.occupancy_minus,
            )
            initial_loss = criterion(
                initial_plus,
                initial_minus,
                outcome.completion_plus,
                outcome.pair_batch.occupancy_plus,
                outcome.gt_union,
                outcome.pair_batch.label_increment,
                outcome.pair_batch.image_valid_mask,
                outcome.intervention_footprint,
            )["total"]
            plus_gradient, minus_gradient = torch.autograd.grad(
                initial_loss,
                (initial_plus, initial_minus),
            )
            endpoint_gradient = {
                "plus_finite_nonzero": (
                    bool(torch.isfinite(plus_gradient).all())
                    and int(torch.count_nonzero(plus_gradient)) > 0
                ),
                "minus_finite_nonzero": (
                    bool(torch.isfinite(minus_gradient).all())
                    and int(torch.count_nonzero(minus_gradient)) > 0
                ),
            }

            gradient_failures: list[dict[str, object]] = []
            gradient_minimum = float("inf")
            gradient_maximum = 0.0
            first_logs: dict[str, float | int] | None = None
            last_logs: dict[str, float | int] | None = None
            for update in range(FROZEN_UPDATES):
                logs = outcome_complete_train_step(
                    decoder,
                    absolute,
                    criterion,
                    optimizer,
                    factual,
                    outcome,
                )
                if update == 0:
                    first_logs = dict(logs)
                if update == FROZEN_UPDATES - 1:
                    last_logs = dict(logs)
                for name, parameter in named_parameters:
                    gradient = parameter.grad
                    finite = gradient is not None and bool(
                        torch.isfinite(gradient).all()
                    )
                    norm = (
                        0.0
                        if gradient is None
                        else float(gradient.detach().double().norm())
                    )
                    gradient_minimum = min(gradient_minimum, norm)
                    gradient_maximum = max(gradient_maximum, norm)
                    if not finite or norm <= 0.0:
                        gradient_failures.append(
                            {
                                "update": update,
                                "parameter": name,
                                "finite": finite,
                                "l2_norm": norm,
                            }
                        )

            decoder.eval()
            with torch.no_grad():
                logits_plus, logits_minus = _paired_endpoint_logits(
                    decoder,
                    feature=outcome.pair_batch.feature,
                    occupancy_plus=outcome.pair_batch.occupancy_plus,
                    occupancy_minus=outcome.pair_batch.occupancy_minus,
                )
                score_plus = torch.sigmoid(logits_plus)
                score_minus = torch.sigmoid(logits_minus)
                delta = score_minus - score_plus
                miss_score = torch.sigmoid(
                    decoder(
                        factual["factual_miss"].feature,
                        factual["factual_miss"].occupancy,
                    )
                )
                no_miss_score = torch.sigmoid(
                    decoder(
                        factual["factual_no_miss"].feature,
                        factual["factual_no_miss"].occupancy,
                    )
                )
        finally:
            handle.remove()

    clean = slice(0, 1)
    component = slice(1, 2)
    clean_D = outcome.response_stratum[clean]
    clean_H = outcome.local_zero_stratum[clean]
    clean_G = outcome.global_zero_stratum[clean]
    component_H = outcome.local_zero_stratum[component]
    component_G = outcome.global_zero_stratum[component]
    anchor_background = (
        outcome.pair_batch.image_valid_mask
        & ~outcome.pair_batch.occupancy_plus
        & ~outcome.gt_union
    )
    miss_target = factual["factual_miss"].target > 0.5
    miss_background = factual["factual_miss"].valid_mask & ~miss_target
    metrics = {
        "total_loss": float(last_logs["total"]),
        "plus_completion_min": _minimum(score_plus[outcome.completion_plus]),
        "plus_background_max": _maximum(score_plus[anchor_background]),
        "factual_miss_target_min": _minimum(miss_score[miss_target]),
        "factual_miss_background_max": _maximum(
            miss_score[miss_background]
        ),
        "factual_no_miss_max": _maximum(no_miss_score),
        "clean_D_mean": float(delta[clean][clean_D].mean()),
        "clean_H_max_abs": _maximum(delta[clean][clean_H].abs()),
        "clean_G_max_abs": _maximum(delta[clean][clean_G].abs()),
        "component_H_max_abs": _maximum(
            delta[component][component_H].abs()
        ),
        "component_G_max_abs": _maximum(
            delta[component][component_G].abs()
        ),
    }
    checks = {
        "total_loss": metrics["total_loss"] < THRESHOLDS[
            "total_loss_max_exclusive"
        ],
        "plus_completion": metrics["plus_completion_min"] > THRESHOLDS[
            "plus_completion_min_exclusive"
        ],
        "plus_background": metrics["plus_background_max"] < THRESHOLDS[
            "plus_background_max_exclusive"
        ],
        "factual_miss_target": metrics[
            "factual_miss_target_min"
        ] > THRESHOLDS["factual_miss_target_min_exclusive"],
        "factual_miss_background": metrics[
            "factual_miss_background_max"
        ] < THRESHOLDS["factual_miss_background_max_exclusive"],
        "factual_no_miss": metrics["factual_no_miss_max"] < THRESHOLDS[
            "factual_no_miss_max_exclusive"
        ],
        "clean_D": metrics["clean_D_mean"] >= THRESHOLDS[
            "clean_D_mean_min_inclusive"
        ],
        "clean_H": metrics["clean_H_max_abs"] <= THRESHOLDS[
            "clean_H_max_abs_max_inclusive"
        ],
        "clean_G": metrics["clean_G_max_abs"] <= THRESHOLDS[
            "clean_G_max_abs_max_inclusive"
        ],
        "component_H": metrics["component_H_max_abs"] <= THRESHOLDS[
            "component_H_max_abs_max_inclusive"
        ],
        "component_G": metrics["component_G_max_abs"] <= THRESHOLDS[
            "component_G_max_abs_max_inclusive"
        ],
        "dual_endpoint_gradients": all(endpoint_gradient.values()),
        "all_parameter_gradients_finite_nonzero": not gradient_failures,
        "pair_endpoints_use_one_2B_forward": forward_batches[0] == 4,
    }
    operator = _operator_audit(decoder, outcome)
    checks["initial_operator_contract"] = (
        initial_operator["all_pass"] is True
    )
    checks["operator_contract"] = operator["all_pass"] is True
    geometry_checks = {
        "response_inside_count_support": (
            operator["changed_support_pixels"] > 0
        ),
        "support_family_response_outside_removed_component": (
            family_id == LEGACY_FAMILY
            or not bool(
                (
                    outcome.response_stratum[clean]
                    & outcome.removed_component[clean]
                ).any()
            )
        ),
    }
    checks["geometry_contract"] = all(geometry_checks.values())
    return {
        "family_id": family_id,
        "case_id": case_id,
        "clean_pixels": [list(pixel) for pixel in clean_pixels],
        "input_fingerprint": _inputs_fingerprint(outcome, factual),
        "initial_decoder_fingerprint": initial_decoder_fingerprint,
        "final_decoder_fingerprint": _decoder_fingerprint(decoder),
        "metrics": metrics,
        "checks": checks,
        "all_pass": all(checks.values()),
        "operator_audit": operator,
        "initial_operator_audit": initial_operator,
        "geometry_checks": geometry_checks,
        "endpoint_gradient": endpoint_gradient,
        "gradient_contract": {
            "parameter_tensors": len(named_parameters),
            "parameters": sum(value.numel() for _, value in named_parameters),
            "updates_checked": FROZEN_UPDATES,
            "failure_count": len(gradient_failures),
            "failures": gradient_failures,
            "minimum_l2_norm": gradient_minimum,
            "maximum_l2_norm": gradient_maximum,
        },
        "forward_contract": {
            "first_call_batch_size": forward_batches[0],
            "paired_batch_size": 2,
            "endpoint_state_count": 4,
            "uses_one_2B_endpoint_forward": forward_batches[0] == 4,
            "training_step_decoder_calls": int(
                first_logs["decoder_forward_calls_per_update"]
            ),
            "training_step_decoder_states": int(
                first_logs["decoder_states_per_update"]
            ),
        },
        "first_update_logs": first_logs,
        "last_update_logs": last_logs,
    }


def evaluate() -> dict[str, object]:
    """Return the deterministic six-case v8 toy result."""

    protocol_binding = _load_protocol_binding()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(min(previous_threads, 2))
        torch.use_deterministic_algorithms(True)
        cases = [_case(*case) for case in CONSERVATIVE_TOY_CASES]
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    orthogonality = _orthogonality_audit()
    numerical = _numerical_audit()
    all_pass = (
        all(case["all_pass"] is True for case in cases)
        and orthogonality["all_pass"] is True
        and numerical["all_pass"] is True
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "protocol_binding": protocol_binding,
        "decision": (
            "CC_SEA_V8_TOY_GATE_PASS"
            if all_pass
            else "CC_SEA_V8_TOY_GATE_FAIL"
        ),
        "all_pass": all_pass,
        "contract": {
            "seed": FROZEN_SEED,
            "updates_per_case": FROZEN_UPDATES,
            "learning_rate": FROZEN_LEARNING_RATE,
            "case_count": len(CONSERVATIVE_TOY_CASES),
            "family_count": 2,
            "parameter_tensors": EXPECTED_PARAMETER_TENSORS,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "thresholds": THRESHOLDS,
            "pre_mask_paired_loss": True,
            "topology_changed": False,
            "loss_changed": False,
            "inference_changed": False,
        },
        "passed_case_count": sum(case["all_pass"] is True for case in cases),
        "failed_case_count": sum(case["all_pass"] is not True for case in cases),
        "cases": cases,
        "orthogonality_audit": orthogonality,
        "numerical_contract_audit": numerical,
        "execution_boundary": {
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "detection_performance_evaluated": False,
            "real_bounded_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_detector_authorized": False,
        },
        "interpretation": (
            "dataset_free_model_code_gate_not_detection_performance"
        ),
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def _write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate()
    _write_result(args.output, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "result_fingerprint": result["result_fingerprint"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["all_pass"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
