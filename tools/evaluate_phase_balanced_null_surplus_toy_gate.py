#!/usr/bin/env python3
"""Run the dataset-free CURE-Lite PB-NAES v9 toy model gate."""

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
from cure_lite.experiment.conservative_toy_inputs import (  # noqa: E402
    CONSERVATIVE_TOY_CASES,
    LEGACY_FAMILY,
    build_conservative_toy_case,
)
from cure_lite.losses import CURELiteLoss  # noqa: E402
from cure_lite.paired_outcome_losses import (  # noqa: E402
    OutcomeCompleteTransitionLoss,
)
from cure_lite.phase_balanced_null_surplus_factorized_decoder import (  # noqa: E402
    CURELitePhaseBalancedNullSurplusFactorizedDecoder,
    phase_balanced_null_surplus_evidence,
)
from cure_lite.train.paired_outcome_step import (  # noqa: E402
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import (  # noqa: E402
    _paired_endpoint_logits,
)


SCHEMA_VERSION = "cure-lite-pb-naes-v9-toy-gate-result-v1"
METHOD_ID = "pb_naes_v9"
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
OPERATOR_THRESHOLDS = {
    "float64_formula_max_abs_error": 1.0e-12,
    "float32_capacity_max_relative_error": 1.0e-6,
    "inactive_coordinate_gradient_min_exclusive": 1.0e-12,
    "count_support_outside_max_abs_delta": 0.0,
    "uniform_zero_max_abs_evidence": 0.0,
    "uniform_negative_max_abs_evidence": 0.0,
}
_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "phase_balanced_null_anchored_evidence_surplus_v9"
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
    """Strictly bind the predeclared v9 proposal and toy contract."""

    config = _load_object(_TOY_CONFIG, name="v9 toy config")
    config_fingerprint = _verify_internal_fingerprint(
        config,
        field="config_fingerprint",
        name="v9 toy config",
    )
    if (
        config.get("schema_version")
        != "cure-lite-pb-naes-v9-toy-config-v1"
    ):
        raise RuntimeError("v9 toy config schema differs")
    if config.get("method_id") != METHOD_ID:
        raise RuntimeError("v9 toy config method differs")
    expected_cases = [
        {
            "family_id": family,
            "case_id": case_id,
            "clean_pixels": [list(pixel) for pixel in pixels],
        }
        for family, case_id, pixels in CONSERVATIVE_TOY_CASES
    ]
    if config.get("cases") != expected_cases:
        raise RuntimeError("v9 toy cases differ")
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
        raise RuntimeError("v9 toy decoder contract differs")
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
        raise RuntimeError("v9 toy optimization contract differs")
    if config.get("thresholds") != THRESHOLDS:
        raise RuntimeError("v9 toy thresholds differ")
    if config.get("operator_thresholds") != OPERATOR_THRESHOLDS:
        raise RuntimeError("v9 toy operator thresholds differ")
    if config.get("decision_rule") != {
        "required_passed_case_count": 6,
        "required_passed_family_count": 2,
        "per_case_all_checks_required": True,
        "mean_cannot_override_case_failure": True,
        "counterexample_audit_required": True,
        "numerical_audit_required": True,
        "pass_decision": "PB_NAES_V9_TOY_GATE_PASS",
        "fail_decision": "PB_NAES_V9_TOY_GATE_FAIL",
    }:
        raise RuntimeError("v9 toy decision rule differs")
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
        raise RuntimeError("v9 toy execution boundary differs")

    proposal_binding = config.get("proposal_binding")
    if not isinstance(proposal_binding, dict):
        raise TypeError("v9 proposal_binding must be an object")
    proposal_path = _ROOT / str(proposal_binding.get("repo_path"))
    if file_sha256(proposal_path) != proposal_binding.get("file_sha256"):
        raise RuntimeError("v9 proposal file hash differs")
    proposal = _load_object(proposal_path, name="v9 proposal")
    proposal_fingerprint = _verify_internal_fingerprint(
        proposal,
        field="proposal_fingerprint",
        name="v9 proposal",
    )
    if (
        proposal_fingerprint
        != proposal_binding.get("proposal_fingerprint")
    ):
        raise RuntimeError("v9 proposal fingerprint binding differs")
    if proposal.get("method_id") != METHOD_ID:
        raise RuntimeError("v9 proposal method differs")
    if config.get("operator") != proposal.get("single_mechanism"):
        raise RuntimeError("v9 proposal/config operator differs")

    design = proposal.get("design_document")
    if not isinstance(design, dict):
        raise TypeError("v9 design binding must be an object")
    design_path = _ROOT / str(design.get("repo_path"))
    if file_sha256(design_path) != design.get("file_sha256"):
        raise RuntimeError("v9 design document differs")
    predecessor = proposal.get("predecessor_v8")
    if not isinstance(predecessor, dict):
        raise TypeError("v9 predecessor binding must be an object")
    predecessor_path = _ROOT / str(predecessor.get("repo_path"))
    if file_sha256(predecessor_path) != predecessor.get("file_sha256"):
        raise RuntimeError("v9 predecessor closure hash differs")
    predecessor_value = _load_object(
        predecessor_path,
        name="v8 negative closure",
    )
    if (
        predecessor_value.get("receipt_fingerprint")
        != predecessor.get("receipt_fingerprint")
        or predecessor_value.get("decision") != predecessor.get("decision")
        or predecessor_value.get("decision")
        != "CC_SEA_V8_R2_BOUNDED_NEGATIVE_CLOSED"
    ):
        raise RuntimeError("v9 predecessor closure binding differs")
    return {
        "toy_config_repo_path": str(_TOY_CONFIG.relative_to(_ROOT)),
        "toy_config_file_sha256": file_sha256(_TOY_CONFIG),
        "toy_config_fingerprint": config_fingerprint,
        "proposal_repo_path": str(proposal_path.relative_to(_ROOT)),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_fingerprint": proposal_fingerprint,
        "design_document_file_sha256": design["file_sha256"],
        "predecessor_v8_closure_file_sha256": predecessor["file_sha256"],
        "predecessor_v8_closure_fingerprint": predecessor[
            "receipt_fingerprint"
        ],
    }


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return (
        str(tensor.dtype).encode("utf-8")
        + repr(tuple(tensor.shape)).encode("utf-8")
        + tensor.numpy().tobytes()
    )


def _decoder_fingerprint(
    decoder: CURELitePhaseBalancedNullSurplusFactorizedDecoder,
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
    decoder: CURELitePhaseBalancedNullSurplusFactorizedDecoder,
    outcome: object,
) -> dict[str, object]:
    """Audit the frozen equation on one complete paired toy state."""

    batch = outcome.pair_batch
    with torch.no_grad():
        plus = decoder.forward_fields(batch.feature, batch.occupancy_plus)
        minus = decoder.forward_fields(batch.feature, batch.occupancy_minus)
        identity = decoder(batch.feature, batch.occupancy_plus)
        identity_again = decoder(
            batch.feature.clone(),
            batch.occupancy_plus.clone(),
        )
        phase_count = int(plus.raw_phase_evidence.shape[1])
        stride = decoder.feature_stride
        count_release = (
            plus.local_occupancy_count - minus.local_occupancy_count
        )
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
            torch.sigmoid(minus.logits) - torch.sigmoid(plus.logits)
        )
        outside = ~changed_output

        formula_errors: list[Tensor] = []
        capacity_relative_violations: list[Tensor] = []
        state_equations: list[dict[str, object]] = []
        for fields in (plus, minus):
            expected_intensity = torch.exp(fields.raw_phase_evidence)
            expected_threshold = (
                expected_intensity.sum(dim=1, keepdim=True) + phase_count
            ) / (2 * phase_count)
            expected_signed = (
                expected_intensity - expected_threshold
            ) / (1.0 + fields.local_occupancy_count)
            expected_forward = expected_signed.clamp_min(0.0)
            errors = (
                (fields.phase_intensity - expected_intensity).abs().max(),
                (
                    fields.implicit_null_threshold - expected_threshold
                ).abs().max(),
                (
                    fields.signed_phase_surplus - expected_signed
                ).abs().max(),
                (
                    fields.native_phase_evidence - expected_forward
                ).abs().max(),
            )
            formula_errors.extend(errors)
            capacity = (
                fields.phase_intensity.sum(dim=1, keepdim=True)
                / (1.0 + fields.local_occupancy_count)
            )
            capacity_violation = (
                fields.native_phase_evidence.sum(dim=1, keepdim=True)
                - capacity
            ).clamp_min(0.0)
            capacity_relative = (
                capacity_violation / capacity.clamp_min(1.0)
            )
            capacity_relative_violations.append(capacity_relative.max())
            state_equations.append(
                {
                    "formula_max_abs_error": max(
                        float(value) for value in errors
                    ),
                    "active_mask_exact": torch.equal(
                        fields.active_phase_mask,
                        expected_signed > 0.0,
                    ),
                    "forward_nonnegative": bool(
                        torch.all(fields.native_phase_evidence >= 0.0)
                    ),
                    "capacity_max_relative_violation": float(
                        capacity_relative.max()
                    ),
                    "logit_composition_exact": torch.equal(
                        fields.logits,
                        fields.baseline_logits + fields.evidence,
                    ),
                }
            )

        response_outside_support = (
            outcome.response_stratum & ~changed_output
        )
        checks = {
            "identity_exact": torch.equal(identity, identity_again),
            "formula_exact": max(
                float(value) for value in formula_errors
            ) <= OPERATOR_THRESHOLDS[
                "float64_formula_max_abs_error"
            ],
            "phase_active_set_is_occupancy_invariant": torch.equal(
                plus.active_phase_mask,
                minus.active_phase_mask,
            ),
            "deletion_is_phasewise_monotone": bool(
                torch.all(
                    minus.native_phase_evidence
                    >= plus.native_phase_evidence
                )
            ),
            "deletion_is_output_monotone": bool(torch.all(delta >= 0.0)),
            "outside_count_support_exact": (
                _maximum(delta[outside].abs())
                <= OPERATOR_THRESHOLDS[
                    "count_support_outside_max_abs_delta"
                ]
            ),
            "outside_count_support_probability_exact": (
                _maximum(probability_delta[outside].abs())
                <= OPERATOR_THRESHOLDS[
                    "count_support_outside_max_abs_delta"
                ]
            ),
            "response_is_inside_count_support": not bool(
                response_outside_support.any()
            ),
            "capacity_bound": max(
                float(value) for value in capacity_relative_violations
            ) <= OPERATOR_THRESHOLDS[
                "float32_capacity_max_relative_error"
            ],
            "state_equations_exact": all(
                state["formula_max_abs_error"]
                <= OPERATOR_THRESHOLDS[
                    "float64_formula_max_abs_error"
                ]
                and state["active_mask_exact"] is True
                and state["forward_nonnegative"] is True
                and state["capacity_max_relative_violation"]
                <= OPERATOR_THRESHOLDS[
                    "float32_capacity_max_relative_error"
                ]
                and state["logit_composition_exact"] is True
                for state in state_equations
            ),
            "all_fields_finite": all(
                bool(torch.isfinite(value).all())
                for value in (
                    plus.phase_intensity,
                    plus.implicit_null_threshold,
                    plus.signed_phase_surplus,
                    plus.native_phase_evidence,
                    plus.evidence,
                    plus.logits,
                    minus.phase_intensity,
                    minus.implicit_null_threshold,
                    minus.signed_phase_surplus,
                    minus.native_phase_evidence,
                    minus.evidence,
                    minus.logits,
                )
            ),
        }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "formula_max_abs_error": max(
            float(value) for value in formula_errors
        ),
        "capacity_max_relative_violation": max(
            float(value) for value in capacity_relative_violations
        ),
        "outside_count_support_max_abs_delta": _maximum(
            delta[outside].abs()
        ),
        "outside_count_support_probability_max_abs_delta": _maximum(
            probability_delta[outside].abs()
        ),
        "active_set_occupancy_delta_count": int(
            torch.count_nonzero(
                plus.active_phase_mask != minus.active_phase_mask
            )
        ),
        "changed_support_pixels": int(changed_output.sum()),
        "unchanged_support_pixels": int(outside.sum()),
        "state_equation_contract": {
            "plus": state_equations[0],
            "minus": state_equations[1],
        },
    }


def _counterexample_audit() -> dict[str, object]:
    """Evaluate every predeclared algebraic counterexample."""

    count0 = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    formula_raw = torch.tensor(
        [[[[1.2]], [[0.1]], [[-0.4]], [[-1.3]]]],
        dtype=torch.float64,
    )
    formula_count = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    formula = phase_balanced_null_surplus_evidence(
        formula_raw,
        formula_count,
    )
    expected_intensity = formula_raw.exp()
    expected_threshold = (
        4.0 + expected_intensity.sum(dim=1, keepdim=True)
    ) / 8.0
    expected_signed = (
        expected_intensity - expected_threshold
    ) / (1.0 + formula_count)
    formula_error = max(
        float((formula[0] - expected_intensity).abs().max()),
        float((formula[1] - expected_threshold).abs().max()),
        float((formula[2] - expected_signed).abs().max()),
        float((formula[4] - expected_signed.clamp_min(0.0)).abs().max()),
    )

    uniform_zero = phase_balanced_null_surplus_evidence(
        torch.zeros((2, 16, 2, 3), dtype=torch.float64),
        torch.tensor(
            [[[[0.0]]], [[[3.0]]]],
            dtype=torch.float64,
        ).expand(2, 1, 2, 3),
    )
    uniform_negative = phase_balanced_null_surplus_evidence(
        torch.full((1, 16, 2, 3), -0.4, dtype=torch.float64),
        torch.zeros((1, 1, 2, 3), dtype=torch.float64),
    )
    positive_raw = torch.full(
        (1, 16, 2, 3),
        0.4,
        dtype=torch.float64,
    )
    uniform_positive = phase_balanced_null_surplus_evidence(
        positive_raw,
        torch.full((1, 1, 2, 3), 2.0, dtype=torch.float64),
    )
    p1_positive = phase_balanced_null_surplus_evidence(
        torch.tensor([[[[0.7]]]], dtype=torch.float64),
        count0,
    )
    p1_negative = phase_balanced_null_surplus_evidence(
        torch.tensor([[[[-0.7]]]], dtype=torch.float64),
        count0,
    )

    inactive_raw = torch.full(
        (1, 4, 1, 1),
        -2.0,
        dtype=torch.float64,
        requires_grad=True,
    )
    inactive_evidence = phase_balanced_null_surplus_evidence(
        inactive_raw,
        count0,
    )[4]
    inactive_evidence.sum().backward()
    inactive_gradient = inactive_raw.grad.detach().clone()

    wrong_raw = torch.tensor(
        [[[[2.0]], [[-1.0]], [[-2.0]], [[-3.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    wrong_evidence = phase_balanced_null_surplus_evidence(
        wrong_raw,
        count0,
    )[4]
    (
        wrong_evidence[0, 0, 0, 0]
        - wrong_evidence[0, 1, 0, 0]
    ).backward()
    wrong_gradient = wrong_raw.grad.detach().clone()

    multi_phase_active_counts: list[int] = []
    for active_count in (1, 2, 3):
        raw_values = torch.full(
            (1, 4, 1, 1),
            -2.0,
            dtype=torch.float64,
        )
        raw_values[:, :active_count] = 1.0
        active = phase_balanced_null_surplus_evidence(
            raw_values,
            count0,
        )[3]
        multi_phase_active_counts.append(int(active.sum()))

    occupancy_raw = torch.tensor(
        [[[[1.4]], [[0.3]], [[-0.2]], [[-1.1]]]],
        dtype=torch.float64,
    )
    occupied = phase_balanced_null_surplus_evidence(
        occupancy_raw,
        torch.full_like(count0, 4.0),
    )
    deleted = phase_balanced_null_surplus_evidence(
        occupancy_raw,
        count0,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9031)
        capacity_raw = (
            torch.randn(3, 16, 7, 5, dtype=torch.float32) * 2.5
        )
        capacity_count = torch.randint(
            0,
            10,
            (3, 1, 7, 5),
        ).to(torch.float32)
    capacity_state = phase_balanced_null_surplus_evidence(
        capacity_raw,
        capacity_count,
    )
    capacity = (
        capacity_state[0].sum(dim=1, keepdim=True)
        / (1.0 + capacity_count)
    )
    capacity_violation = (
        capacity_state[4].sum(dim=1, keepdim=True) - capacity
    ).clamp_min(0.0)
    capacity_relative = float(
        (capacity_violation / capacity.clamp_min(1.0)).max()
    )

    checks = {
        "formula_exact": (
            formula_error
            <= OPERATOR_THRESHOLDS["float64_formula_max_abs_error"]
        ),
        "uniform_zero_is_exact_null": (
            _maximum(uniform_zero[4].abs())
            <= OPERATOR_THRESHOLDS["uniform_zero_max_abs_evidence"]
            and not bool(uniform_zero[3].any())
        ),
        "uniform_negative_is_finite_null": (
            _maximum(uniform_negative[4].abs())
            <= OPERATOR_THRESHOLDS[
                "uniform_negative_max_abs_evidence"
            ]
            and not bool(uniform_negative[3].any())
            and all(
                bool(torch.isfinite(value).all())
                for value in uniform_negative[:3]
            )
        ),
        "uniform_positive_activates_full_phase_capacity": (
            bool(uniform_positive[3].all())
            and bool(torch.all(uniform_positive[4] > 0.0))
        ),
        "stride_one_is_nondegenerate": (
            bool(p1_positive[3].item())
            and float(p1_positive[4].item()) > 0.0
            and not bool(p1_negative[3].item())
            and float(p1_negative[4].item()) == 0.0
        ),
        "inactive_phase_recovery_gradient": (
            bool(torch.isfinite(inactive_gradient).all())
            and bool(
                torch.all(
                    inactive_gradient
                    > OPERATOR_THRESHOLDS[
                        "inactive_coordinate_gradient_min_exclusive"
                    ]
                )
            )
        ),
        "wrong_winner_recovery_direction": (
            float(wrong_evidence[0, 0].detach()) > 0.0
            and float(wrong_evidence[0, 1].detach()) == 0.0
            and float(wrong_gradient[0, 0]) > 0.0
            and float(wrong_gradient[0, 1]) < 0.0
        ),
        "multi_phase_capacity_1_2_3": (
            multi_phase_active_counts == [1, 2, 3]
        ),
        "active_set_is_occupancy_invariant": torch.equal(
            occupied[3],
            deleted[3],
        ),
        "occupancy_deletion_is_phasewise_monotone": (
            bool(torch.all(deleted[4] >= occupied[4]))
            and bool(torch.any(deleted[4] > occupied[4]))
        ),
        "positive_surplus_obeys_capacity_bound": (
            capacity_relative
            <= OPERATOR_THRESHOLDS[
                "float32_capacity_max_relative_error"
            ]
        ),
        "all_counterexample_fields_finite": all(
            bool(torch.isfinite(value).all())
            for state in (
                formula,
                uniform_zero,
                uniform_negative,
                uniform_positive,
                p1_positive,
                p1_negative,
                occupied,
                deleted,
                capacity_state,
            )
            for value in (state[0], state[1], state[2], state[4])
        ),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "formula_max_abs_error": formula_error,
        "uniform_zero_max_abs_evidence": _maximum(
            uniform_zero[4].abs()
        ),
        "uniform_negative_max_abs_evidence": _maximum(
            uniform_negative[4].abs()
        ),
        "uniform_positive_min_evidence": _minimum(uniform_positive[4]),
        "inactive_gradient_min": _minimum(inactive_gradient),
        "wrong_winner_gradient": {
            "wrong_phase": float(wrong_gradient[0, 0]),
            "target_phase": float(wrong_gradient[0, 1]),
        },
        "multi_phase_active_counts": multi_phase_active_counts,
        "occupancy_release_min_delta": _minimum(
            deleted[4] - occupied[4]
        ),
        "capacity_max_relative_violation": capacity_relative,
    }


def _numerical_audit() -> dict[str, object]:
    negative = torch.full(
        (1, 4, 1, 1),
        -2.0,
        dtype=torch.float32,
        requires_grad=True,
    )
    negative_value = phase_balanced_null_surplus_evidence(
        negative,
        torch.zeros((1, 1, 1, 1), dtype=torch.float32),
    )[4]
    negative_value.sum().backward()
    largest = torch.full(
        (1, 4, 1, 1),
        87.0,
        dtype=torch.float32,
        requires_grad=True,
    )
    largest_value = phase_balanced_null_surplus_evidence(
        largest,
        torch.zeros((1, 1, 1, 1), dtype=torch.float32),
    )[4]
    largest_value.sum().backward()
    first_nonfinite_failed = False
    try:
        phase_balanced_null_surplus_evidence(
            torch.full((1, 4, 1, 1), 88.0, dtype=torch.float32),
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        )
    except ValueError:
        first_nonfinite_failed = True
    nan_failed = False
    try:
        phase_balanced_null_surplus_evidence(
            torch.full(
                (1, 4, 1, 1),
                float("nan"),
                dtype=torch.float32,
            ),
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        )
    except ValueError:
        nan_failed = True
    checks = {
        "inactive_recovery_is_finite_nonzero": (
            negative.grad is not None
            and bool(torch.isfinite(negative.grad).all())
            and bool(
                torch.all(
                    negative.grad
                    > OPERATOR_THRESHOLDS[
                        "inactive_coordinate_gradient_min_exclusive"
                    ]
                )
            )
        ),
        "largest_supported_positive_is_finite": (
            largest.grad is not None
            and bool(torch.isfinite(largest_value).all())
            and bool(torch.isfinite(largest.grad).all())
        ),
        "first_nonfinite_positive_fails_fast": first_nonfinite_failed,
        "nonfinite_input_fails_fast": nan_failed,
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
        decoder = CURELitePhaseBalancedNullSurplusFactorizedDecoder(
            feature_channels=8,
            feature_stride=4,
        )
        initial_decoder_fingerprint = _decoder_fingerprint(decoder)
        initial_operator = _operator_audit(decoder, outcome)
        named_parameters = tuple(decoder.named_parameters())
        if len(named_parameters) != EXPECTED_PARAMETER_TENSORS:
            raise AssertionError("v9 toy parameter tensor count differs")
        if sum(value.numel() for _, value in named_parameters) != (
            EXPECTED_PARAMETER_COUNT
        ):
            raise AssertionError("v9 toy parameter count differs")

        feature_probe = (
            outcome.pair_batch.feature.detach().clone().requires_grad_(True)
        )
        feature_probe_logits = decoder(
            feature_probe,
            outcome.pair_batch.occupancy_plus,
        )
        feature_probe_gradient = torch.autograd.grad(
            feature_probe_logits.sum(),
            feature_probe,
            allow_unused=True,
        )[0]
        feature_detached = feature_probe_gradient is None

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
        "total_loss": metrics["total_loss"]
        < THRESHOLDS["total_loss_max_exclusive"],
        "plus_completion": metrics["plus_completion_min"]
        > THRESHOLDS["plus_completion_min_exclusive"],
        "plus_background": metrics["plus_background_max"]
        < THRESHOLDS["plus_background_max_exclusive"],
        "factual_miss_target": metrics["factual_miss_target_min"]
        > THRESHOLDS["factual_miss_target_min_exclusive"],
        "factual_miss_background": metrics["factual_miss_background_max"]
        < THRESHOLDS["factual_miss_background_max_exclusive"],
        "factual_no_miss": metrics["factual_no_miss_max"]
        < THRESHOLDS["factual_no_miss_max_exclusive"],
        "clean_D": metrics["clean_D_mean"]
        >= THRESHOLDS["clean_D_mean_min_inclusive"],
        "clean_H": metrics["clean_H_max_abs"]
        <= THRESHOLDS["clean_H_max_abs_max_inclusive"],
        "clean_G": metrics["clean_G_max_abs"]
        <= THRESHOLDS["clean_G_max_abs_max_inclusive"],
        "component_H": metrics["component_H_max_abs"]
        <= THRESHOLDS["component_H_max_abs_max_inclusive"],
        "component_G": metrics["component_G_max_abs"]
        <= THRESHOLDS["component_G_max_abs_max_inclusive"],
        "dual_endpoint_gradients": all(endpoint_gradient.values()),
        "all_parameter_gradients_finite_nonzero": not gradient_failures,
        "pair_endpoints_use_one_2B_forward": forward_batches[0] == 4,
        "feature_is_detached": feature_detached,
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
        "feature_detach_contract": {
            "input_requires_grad": feature_probe.requires_grad,
            "input_gradient_is_none": feature_probe_gradient is None,
            "passed": feature_detached,
        },
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
    """Return the deterministic six-case v9 toy result in memory."""

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

    counterexamples = _counterexample_audit()
    numerical = _numerical_audit()
    passed_families = {
        case["family_id"] for case in cases if case["all_pass"] is True
    }
    all_pass = (
        all(case["all_pass"] is True for case in cases)
        and len(passed_families) == 2
        and counterexamples["all_pass"] is True
        and numerical["all_pass"] is True
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "protocol_binding": protocol_binding,
        "decision": (
            "PB_NAES_V9_TOY_GATE_PASS"
            if all_pass
            else "PB_NAES_V9_TOY_GATE_FAIL"
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
            "operator_thresholds": OPERATOR_THRESHOLDS,
            "pre_mask_paired_loss": True,
            "topology_changed": False,
            "loss_changed": False,
            "inference_changed": False,
        },
        "passed_case_count": sum(
            case["all_pass"] is True for case in cases
        ),
        "failed_case_count": sum(
            case["all_pass"] is not True for case in cases
        ),
        "passed_family_count": len(passed_families),
        "cases": cases,
        "counterexample_audit": counterexamples,
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
