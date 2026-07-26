#!/usr/bin/env python3
"""Run the dataset-free PECO v10 six-case development regression.

The six cases in this evaluator were used during candidate selection.  A
passing result is therefore an implementation regression only; it is not an
independent confirmation and has no detection-performance authority.
"""

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
)
from cure_lite.experiment.conservative_toy_inputs import (  # noqa: E402
    CONSERVATIVE_TOY_CASES,
    LEGACY_FAMILY,
    build_conservative_toy_case,
)
from cure_lite.losses import CURELiteLoss  # noqa: E402
from cure_lite.paired_endpoint_crossing_losses import (  # noqa: E402
    PairedEndpointCrossingLoss,
)
from cure_lite.train.paired_outcome_step import (  # noqa: E402
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import (  # noqa: E402
    _paired_endpoint_logits,
)


SCHEMA_VERSION = (
    "cure-lite-peco-v10-development-regression-r2-result-v1"
)
METHOD_ID = "peco_v10"
STAGE_ID = "dataset_free_development_regression"
CORRECTION_ID = "restore_frozen_v8_relative_roundoff_r2"
EVIDENTIARY_STATUS = (
    "candidate_selection_development_regression_"
    "not_independent_confirmation"
)
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
    / "paired_endpoint_crossing_objective_v10"
)
_CONFIG = _PROTOCOL / "development_regression_config.json"
_IMPLEMENTATION_CLOSURE = (
    _PROTOCOL / "development_regression_r2_implementation_closure.json"
)
_R1_INVALIDATION = (
    _PROTOCOL
    / "development_regression_r1_verifier_invalidation_receipt.json"
)


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


def _expected_cases() -> list[dict[str, object]]:
    return [
        {
            "family_id": family_id,
            "case_id": case_id,
            "clean_pixels": [list(pixel) for pixel in clean_pixels],
        }
        for family_id, case_id, clean_pixels in CONSERVATIVE_TOY_CASES
    ]


def _load_protocol_binding() -> dict[str, object]:
    """Validate and return the frozen development-regression bindings."""

    config = _load_object(_CONFIG, name="PECO development config")
    config_fingerprint = _verify_internal_fingerprint(
        config,
        field="config_fingerprint",
        name="PECO development config",
    )
    if config.get("schema_version") != (
        "cure-lite-peco-v10-development-regression-config-v1"
    ):
        raise RuntimeError("PECO development config schema differs")
    if config.get("method_id") != METHOD_ID:
        raise RuntimeError("PECO development method differs")
    if config.get("stage_id") != STAGE_ID:
        raise RuntimeError("PECO development stage differs")
    if config.get("status") != (
        "FROZEN_BEFORE_DEVELOPMENT_REGRESSION_EXECUTION"
    ):
        raise RuntimeError("PECO development status differs")
    if config.get("evidentiary_status") != EVIDENTIARY_STATUS:
        raise RuntimeError("PECO development evidence scope differs")
    if config.get("cases") != _expected_cases():
        raise RuntimeError("PECO development cases differ")
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
        "topology_changed": False,
    }:
        raise RuntimeError("PECO development decoder contract differs")
    if config.get("optimization") != {
        "seed": FROZEN_SEED,
        "optimizer": "adam",
        "updates_per_case": FROZEN_UPDATES,
        "learning_rate": FROZEN_LEARNING_RATE,
        "weight_decay": 0.0,
        "loss": (
            "paired_endpoint_crossing_response_plus_unchanged_"
            "zero_anchor_and_factual"
        ),
        "training_step": "unchanged_outcome_complete_train_step",
        "automatic_retry_allowed": False,
    }:
        raise RuntimeError("PECO development optimization differs")
    if config.get("thresholds") != THRESHOLDS:
        raise RuntimeError("PECO development thresholds differ")
    if config.get("objective") != {
        "response_stratum": (
            "D=bool(label_increment)&image_valid_mask"
        ),
        "response_risk": (
            "0.5*(softplus(logits_plus)+softplus(-logits_minus))"
        ),
        "covered_plus_target": 0,
        "uncovered_minus_target": 1,
        "local_zero_risk": "unchanged_probability_delta_squared",
        "global_zero_risk": "unchanged_probability_delta_squared",
        "plus_anchor": "unchanged",
        "hierarchical_active_means": "unchanged",
        "factual_branches": "unchanged",
        "pair_kind_dispatch": False,
    }:
        raise RuntimeError("PECO development objective differs")
    if config.get("audit_contract") != {
        "pre_mask_paired_loss": True,
        "one_2B_endpoint_forward": True,
        "every_update_exact_three_calls_with_batch_sizes_4_4_4": True,
        "dual_endpoint_gradients_finite_nonzero": True,
        "all_six_parameter_gradients_finite_nonzero_every_update": True,
        "input_feature_detached_by_unchanged_step": True,
        "input_and_decoder_fingerprints_recorded": True,
        "frozen_v8_operator_contract_rechecked": True,
        "create_only_result": True,
        "runtime_scope_recorded_without_cross_environment_claim": True,
    }:
        raise RuntimeError("PECO development audit contract differs")
    if config.get("decision_rule") != {
        "required_passed_case_count": 6,
        "required_passed_family_count": 2,
        "per_case_all_checks_required": True,
        "mean_cannot_override_case_failure": True,
        "pass_decision": "PECO_V10_DEVELOPMENT_REGRESSION_PASS",
        "fail_decision": "PECO_V10_DEVELOPMENT_REGRESSION_FAIL",
        "pass_scope": (
            "implementation_regression_only_not_independent_confirmation"
        ),
    }:
        raise RuntimeError("PECO development decision rule differs")
    boundary = {
        "dataset_access_allowed": False,
        "D_R_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
        "detection_performance_allowed": False,
        "exposure_matched_confirmation_run_authorized": False,
        "real_bounded_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "cross_detector_authorized": False,
    }
    if config.get("execution_boundary") != boundary:
        raise RuntimeError("PECO development execution boundary differs")

    proposal_binding = config.get("proposal_binding")
    if not isinstance(proposal_binding, dict):
        raise TypeError("PECO proposal binding must be an object")
    proposal_path = _ROOT / str(proposal_binding.get("repo_path"))
    if file_sha256(proposal_path) != proposal_binding.get("file_sha256"):
        raise RuntimeError("PECO proposal file hash differs")
    proposal = _load_object(proposal_path, name="PECO proposal")
    proposal_fingerprint = _verify_internal_fingerprint(
        proposal,
        field="proposal_fingerprint",
        name="PECO proposal",
    )
    if proposal_fingerprint != proposal_binding.get(
        "proposal_fingerprint"
    ):
        raise RuntimeError("PECO proposal fingerprint binding differs")
    if proposal.get("method_id") != METHOD_ID:
        raise RuntimeError("PECO proposal method differs")
    disclosure = proposal.get("development_screen_disclosure")
    if not isinstance(disclosure, dict):
        raise TypeError("PECO development disclosure must be an object")
    if disclosure.get("evidentiary_status") != (
        "candidate_selection_only_not_independent_confirmation"
    ):
        raise RuntimeError("PECO proposal evidence disclosure differs")

    source_bindings = config.get("frozen_source_bindings")
    if not isinstance(source_bindings, dict):
        raise TypeError("PECO source bindings must be an object")
    observed_source_hashes: dict[str, str] = {}
    for name in (
        "decoder",
        "peco_loss",
        "predecessor_loss",
        "training_step",
        "toy_inputs",
        "absolute_loss",
        "loss_config",
    ):
        binding = source_bindings.get(name)
        if not isinstance(binding, dict):
            raise TypeError(f"PECO {name} binding must be an object")
        source_path = _ROOT / str(binding.get("repo_path"))
        observed_hash = file_sha256(source_path)
        if observed_hash != binding.get("file_sha256"):
            raise RuntimeError(f"PECO {name} source hash differs")
        observed_source_hashes[name] = observed_hash
    if source_bindings["decoder"] != proposal.get(
        "frozen_decoder_binding"
    ):
        raise RuntimeError("PECO frozen decoder proposal binding differs")
    if source_bindings["predecessor_loss"] != {
        **proposal["predecessor_loss_binding"],
    }:
        raise RuntimeError("PECO predecessor loss proposal binding differs")
    proposal_step = proposal.get("training_step_binding")
    if not isinstance(proposal_step, dict):
        raise TypeError("PECO proposal step binding must be an object")
    configured_step = source_bindings["training_step"]
    if (
        configured_step.get("repo_path")
        != proposal_step.get("repo_path")
        or configured_step.get("file_sha256")
        != proposal_step.get("file_sha256")
        or configured_step.get("policy") != proposal_step.get("policy")
    ):
        raise RuntimeError("PECO training-step proposal binding differs")
    return {
        "development_config_repo_path": str(
            _CONFIG.relative_to(_ROOT)
        ),
        "development_config_file_sha256": file_sha256(_CONFIG),
        "development_config_fingerprint": config_fingerprint,
        "proposal_repo_path": str(proposal_path.relative_to(_ROOT)),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_fingerprint": proposal_fingerprint,
        "source_file_sha256": observed_source_hashes,
    }


def _load_implementation_closure(
    protocol_binding: dict[str, object],
) -> dict[str, object]:
    """Verify the signed evaluator and its frozen transitive source closure."""

    closure = _load_object(
        _IMPLEMENTATION_CLOSURE,
        name="PECO development implementation closure",
    )
    closure_fingerprint = _verify_internal_fingerprint(
        closure,
        field="receipt_fingerprint",
        name="PECO development implementation closure",
    )
    if closure.get("schema_version") != (
        "cure-lite-peco-v10-development-r2-implementation-closure-v1"
    ):
        raise RuntimeError("PECO development closure schema differs")
    if closure.get("method_id") != METHOD_ID:
        raise RuntimeError("PECO development closure method differs")
    if closure.get("stage_id") != STAGE_ID:
        raise RuntimeError("PECO development closure stage differs")
    if closure.get("status") != (
        "R2_CORRECTION_CLOSED_BEFORE_SINGLE_AUTHORITY_RUN"
    ):
        raise RuntimeError("PECO development closure status differs")
    if closure.get("decision") != (
        "PECO_V10_DEVELOPMENT_R2_IMPLEMENTATION_CLOSED"
    ):
        raise RuntimeError("PECO development closure decision differs")
    if closure.get("config_binding") != {
        "repo_path": protocol_binding[
            "development_config_repo_path"
        ],
        "file_sha256": protocol_binding[
            "development_config_file_sha256"
        ],
        "config_fingerprint": protocol_binding[
            "development_config_fingerprint"
        ],
    }:
        raise RuntimeError("PECO development closure config binding differs")
    if closure.get("proposal_binding") != {
        "repo_path": protocol_binding["proposal_repo_path"],
        "file_sha256": protocol_binding["proposal_file_sha256"],
        "proposal_fingerprint": protocol_binding[
            "proposal_fingerprint"
        ],
    }:
        raise RuntimeError("PECO development closure proposal binding differs")
    invalidation = _load_object(
        _R1_INVALIDATION,
        name="PECO development r1 verifier invalidation",
    )
    invalidation_fingerprint = _verify_internal_fingerprint(
        invalidation,
        field="receipt_fingerprint",
        name="PECO development r1 verifier invalidation",
    )
    if closure.get("r1_invalidation_binding") != {
        "repo_path": str(_R1_INVALIDATION.relative_to(_ROOT)),
        "file_sha256": file_sha256(_R1_INVALIDATION),
        "receipt_fingerprint": invalidation_fingerprint,
        "decision": "PECO_V10_DEVELOPMENT_R1_VERIFIER_INVALIDATED",
    }:
        raise RuntimeError("PECO development r1 invalidation binding differs")

    source_bindings = closure.get("source_bindings")
    if not isinstance(source_bindings, list) or not source_bindings:
        raise TypeError("PECO development closure sources must be a list")
    observed_paths: set[str] = set()
    for index, binding in enumerate(source_bindings):
        if not isinstance(binding, dict):
            raise TypeError(
                f"PECO development closure source {index} must be an object"
            )
        repo_path = binding.get("repo_path")
        expected_sha = binding.get("file_sha256")
        if not isinstance(repo_path, str) or not repo_path:
            raise TypeError("PECO development closure source path is invalid")
        if repo_path in observed_paths:
            raise RuntimeError(
                "PECO development closure contains a duplicate source path"
            )
        observed_paths.add(repo_path)
        source_path = _ROOT / repo_path
        if file_sha256(source_path) != expected_sha:
            raise RuntimeError(
                f"PECO development closure source differs: {repo_path}"
            )
    required_paths = {
        "CURE_Lite_PECO_v10_模型与代码设计.md",
        "cure_lite/__init__.py",
        "cure_lite/cache/schema.py",
        "cure_lite/config.py",
        "cure_lite/conservative_factorized_config.py",
        "cure_lite/conservative_factorized_decoder.py",
        "cure_lite/crossing_factorized_decoder.py",
        "cure_lite/decoder.py",
        "cure_lite/experiment/conservative_toy_inputs.py",
        "cure_lite/factorized_decoder.py",
        "cure_lite/losses.py",
        "cure_lite/paired_endpoint_crossing_losses.py",
        "cure_lite/paired_outcome_losses.py",
        "cure_lite/paired_outcome_types.py",
        "cure_lite/paired_types.py",
        "cure_lite/train/__init__.py",
        "cure_lite/train/paired_outcome_step.py",
        "cure_lite/train/paired_step.py",
        "cure_lite/train/step.py",
        "protocols/IRSTD-1K/paired_endpoint_crossing_objective_v10/"
        "development_regression_config.json",
        "protocols/IRSTD-1K/paired_endpoint_crossing_objective_v10/"
        "development_regression_r1_verifier_invalidation_receipt.json",
        "protocols/IRSTD-1K/paired_endpoint_crossing_objective_v10/"
        "proposal_receipt.json",
        "tests_v10/test_paired_endpoint_crossing_development_regression.py",
        "tests_v10/test_paired_endpoint_crossing_losses.py",
        "tests_v10/test_paired_endpoint_crossing_protocol.py",
        "tools/evaluate_paired_endpoint_crossing_development_regression.py",
    }
    if observed_paths != required_paths:
        missing = sorted(required_paths - observed_paths)
        extra = sorted(observed_paths - required_paths)
        raise RuntimeError(
            "PECO development closure source set differs: "
            f"missing={missing}, extra={extra}"
        )

    if closure.get("semantic_scope") != {
        "component_null_transition_group": (
            "zero_risk_only_when_D_is_empty"
        ),
        "component_null_complete_pair_loss": (
            "unchanged_plus_anchor_plus_zero_transition"
        ),
        "crossing_boundary": (
            "training_residual_logit_zero_not_frozen_calibration_threshold"
        ),
        "evidentiary_scope": (
            "candidate_selection_development_regression_only"
        ),
        "correction_scope": (
            "restore_frozen_v8_relative_roundoff_audit_only"
        ),
    }:
        raise RuntimeError("PECO development closure semantics differ")
    if closure.get("authorization") != {
        "dataset_free_development_authority_run_count": 1,
        "canonical_output_repo_path": (
            "protocols/IRSTD-1K/"
            "paired_endpoint_crossing_objective_v10/"
            "development_regression_result_r2.json"
        ),
        "automatic_retry_allowed": False,
        "D_R_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
        "detection_performance_allowed": False,
        "exposure_matched_confirmation_run_authorized": False,
        "real_bounded_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "cross_detector_authorized": False,
    }:
        raise RuntimeError("PECO development closure authorization differs")
    return {
        "repo_path": str(_IMPLEMENTATION_CLOSURE.relative_to(_ROOT)),
        "file_sha256": file_sha256(_IMPLEMENTATION_CLOSURE),
        "receipt_fingerprint": closure_fingerprint,
        "r1_invalidation": {
            "repo_path": str(_R1_INVALIDATION.relative_to(_ROOT)),
            "file_sha256": file_sha256(_R1_INVALIDATION),
            "receipt_fingerprint": invalidation_fingerprint,
        },
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


def _adam_contract(
    optimizer: torch.optim.Adam,
) -> dict[str, object]:
    """Serialize the effective Adam defaults used by the frozen run."""

    defaults = optimizer.defaults
    return {
        "class": "torch.optim.Adam",
        "lr": float(defaults["lr"]),
        "betas": [float(value) for value in defaults["betas"]],
        "eps": float(defaults["eps"]),
        "weight_decay": float(defaults["weight_decay"]),
        "amsgrad": bool(defaults["amsgrad"]),
        "maximize": bool(defaults["maximize"]),
        "foreach": defaults["foreach"],
        "capturable": bool(defaults["capturable"]),
        "differentiable": bool(defaults["differentiable"]),
        "fused": defaults["fused"],
        "decoupled_weight_decay": bool(
            defaults["decoupled_weight_decay"]
        ),
    }


def _inputs_fingerprint(
    outcome: object,
    factual: dict[str, object],
) -> str:
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


def _mass_conservation_errors(
    allocated_phase_evidence: Tensor,
    evidence_budget: Tensor,
) -> tuple[float, float]:
    """Return absolute and frozen-v8 scale-aware conservation errors."""

    absolute = (
        allocated_phase_evidence.sum(dim=1, keepdim=True)
        - evidence_budget
    ).abs()
    relative = absolute / evidence_budget.abs().clamp_min(1.0)
    return (
        float(absolute.max().detach()),
        float(relative.max().detach()),
    )


def _operator_audit(
    decoder: CURELiteConservativeFactorizedDecoder,
    outcome: object,
) -> dict[str, object]:
    """Recheck the frozen v8 forward invariants used by all six cases."""

    batch = outcome.pair_batch
    with torch.no_grad():
        plus = decoder.forward_fields(
            batch.feature,
            batch.occupancy_plus,
        )
        minus = decoder.forward_fields(
            batch.feature,
            batch.occupancy_minus,
        )
        first = decoder(batch.feature, batch.occupancy_plus)
        second = decoder(
            batch.feature.clone(),
            batch.occupancy_plus.clone(),
        )
        phase_count = int(plus.phase_allocation.shape[1])
        stride = int(decoder.feature_stride)
        count_release = (
            plus.local_occupancy_count
            - minus.local_occupancy_count
        ) > 0.0
        changed_output = F.pixel_shuffle(
            count_release.expand(
                -1,
                phase_count,
                -1,
                -1,
            ).to(torch.float32),
            stride,
        ).to(torch.bool)
        if plus.field_resize_applied:
            changed_output = F.interpolate(
                changed_output.to(torch.float32),
                size=plus.output_size,
                mode="nearest",
            ).to(torch.bool)
        probability_delta = (
            torch.sigmoid(minus.logits)
            - torch.sigmoid(plus.logits)
        )
        outside = ~changed_output
        states = (plus, minus)
        allocation_error = max(
            float(
                (
                    state.phase_allocation.sum(dim=1, keepdim=True)
                    - 1.0
                ).abs().max()
            )
            for state in states
        )
        conservation_errors = [
            _mass_conservation_errors(
                state.allocated_phase_evidence,
                state.evidence_budget,
            )
            for state in states
        ]
        conservation_abs_error = max(
            absolute for absolute, _ in conservation_errors
        )
        conservation_relative_error = max(
            relative for _, relative in conservation_errors
        )
        finite = all(
            bool(torch.isfinite(value).all())
            for state in states
            for value in (
                state.baseline_logits,
                state.raw_phase_evidence,
                state.occupancy_burden,
                state.evidence_budget,
                state.phase_allocation,
                state.allocated_phase_evidence,
                state.logits,
            )
        )
        response = outcome.response_stratum
        checks = {
            "repeat_forward_exact": torch.equal(first, second),
            "baseline_is_endpoint_invariant": torch.equal(
                plus.baseline_logits,
                minus.baseline_logits,
            ),
            "raw_phase_is_endpoint_invariant": torch.equal(
                plus.raw_phase_evidence,
                minus.raw_phase_evidence,
            ),
            "allocation_is_endpoint_invariant": torch.equal(
                plus.phase_allocation,
                minus.phase_allocation,
            ),
            "occupancy_release_is_budget_monotone": bool(
                torch.all(
                    minus.evidence_budget >= plus.evidence_budget
                )
            ),
            "response_is_inside_count_support": not bool(
                (response & ~changed_output).any()
            ),
            "outside_count_support_probability_exact": torch.equal(
                probability_delta[outside],
                torch.zeros_like(probability_delta[outside]),
            ),
            "allocation_sums_to_one": allocation_error <= 1.0e-6,
            "allocated_evidence_sums_to_budget": (
                conservation_relative_error <= 1.0e-6
            ),
            "all_fields_finite": finite,
        }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "allocation_sum_max_abs_error": allocation_error,
        "mass_conservation_max_abs_error": conservation_abs_error,
        "mass_conservation_max_relative_error": (
            conservation_relative_error
        ),
        "changed_support_pixels": int(changed_output.sum()),
        "outside_probability_max_abs_delta": _maximum(
            probability_delta[outside].abs()
        ),
    }


def _objective_contract_audit() -> dict[str, object]:
    plus = torch.tensor(
        [-8.0, 8.0, -8.0, 8.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    minus = torch.tensor(
        [8.0, 8.0, -8.0, -8.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    observed = 0.5 * (
        F.softplus(plus) + F.softplus(-minus)
    )
    expected = 0.5 * (
        F.binary_cross_entropy_with_logits(
            plus,
            torch.zeros_like(plus),
            reduction="none",
        )
        + F.binary_cross_entropy_with_logits(
            minus,
            torch.ones_like(minus),
            reduction="none",
        )
    )
    plus_gradient, minus_gradient = torch.autograd.grad(
        observed.sum(),
        (plus, minus),
    )
    formula_max_abs_error = float(
        (observed - expected).detach().abs().max()
    )
    checks = {
        "formula_matches_bce_to_float64": (
            formula_max_abs_error <= 1.0e-15
        ),
        "plus_gradient_strictly_positive": bool(
            torch.all(plus_gradient > 0.0)
        ),
        "minus_gradient_strictly_negative": bool(
            torch.all(minus_gradient < 0.0)
        ),
        "both_high_retains_plus_correction": (
            float(plus_gradient[1]) > 0.49
        ),
        "both_low_retains_minus_correction": (
            float(minus_gradient[2]) < -0.49
        ),
        "all_values_and_gradients_finite": all(
            bool(torch.isfinite(value).all())
            for value in (
                observed,
                plus_gradient,
                minus_gradient,
            )
        ),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "formula_max_abs_error": formula_max_abs_error,
        "plus_gradient": [
            float(value) for value in plus_gradient.detach()
        ],
        "minus_gradient": [
            float(value) for value in minus_gradient.detach()
        ],
    }


def _case(
    family_id: str,
    case_id: str,
    clean_pixels: tuple[tuple[int, int], ...],
    *,
    updates: int = FROZEN_UPDATES,
) -> dict[str, object]:
    if updates <= 0:
        raise ValueError("updates must be positive")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FROZEN_SEED)
        outcome, factual = build_conservative_toy_case(
            family_id,
            clean_pixels,
        )
        outcome.pair_batch.feature.requires_grad_()
        for branch in factual.values():
            branch.feature.requires_grad_()

        decoder = CURELiteConservativeFactorizedDecoder(
            feature_channels=8,
            feature_stride=4,
        )
        initial_decoder_fingerprint = _decoder_fingerprint(decoder)
        initial_operator = _operator_audit(decoder, outcome)
        named_parameters = tuple(decoder.named_parameters())
        if len(named_parameters) != EXPECTED_PARAMETER_TENSORS:
            raise AssertionError("PECO parameter tensor count differs")
        if sum(value.numel() for _, value in named_parameters) != (
            EXPECTED_PARAMETER_COUNT
        ):
            raise AssertionError("PECO parameter count differs")

        absolute = CURELiteLoss()
        criterion = PairedEndpointCrossingLoss(LossConfig())
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=FROZEN_LEARNING_RATE,
        )
        optimizer_contract = _adam_contract(optimizer)
        forward_batches: list[int] = []

        def observe_batch(
            _module: object,
            args: tuple[object, ...],
        ) -> None:
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
            initial_forward_batches = tuple(forward_batches)
            training_forward_start = len(forward_batches)

            gradient_failures: list[dict[str, object]] = []
            gradient_minimum = float("inf")
            gradient_maximum = 0.0
            first_logs: dict[str, float | int] | None = None
            last_logs: dict[str, float | int] | None = None
            for update in range(updates):
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
                if update == updates - 1:
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

            training_forward_end = len(forward_batches)
            training_forward_batches = tuple(
                forward_batches[
                    training_forward_start:training_forward_end
                ]
            )
            training_forward_patterns = tuple(
                training_forward_batches[index : index + 3]
                for index in range(
                    0,
                    len(training_forward_batches),
                    3,
                )
            )
            training_forward_contract = (
                initial_forward_batches == (4,)
                and len(training_forward_batches) == updates * 3
                and len(training_forward_patterns) == updates
                and all(
                    pattern == (4, 4, 4)
                    for pattern in training_forward_patterns
                )
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

        operator = _operator_audit(decoder, outcome)

    if first_logs is None or last_logs is None:
        raise AssertionError("PECO case produced no training logs")
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
        "plus_completion_min": _minimum(
            score_plus[outcome.completion_plus]
        ),
        "plus_background_max": _maximum(
            score_plus[anchor_background]
        ),
        "factual_miss_target_min": _minimum(
            miss_score[miss_target]
        ),
        "factual_miss_background_max": _maximum(
            miss_score[miss_background]
        ),
        "factual_no_miss_max": _maximum(no_miss_score),
        "clean_D_mean": float(delta[clean][clean_D].mean()),
        "clean_D_plus_mean": float(
            score_plus[clean][clean_D].mean()
        ),
        "clean_D_minus_mean": float(
            score_minus[clean][clean_D].mean()
        ),
        "clean_H_max_abs": _maximum(
            delta[clean][clean_H].abs()
        ),
        "clean_G_max_abs": _maximum(
            delta[clean][clean_G].abs()
        ),
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
        "plus_completion": metrics[
            "plus_completion_min"
        ] > THRESHOLDS["plus_completion_min_exclusive"],
        "plus_background": metrics[
            "plus_background_max"
        ] < THRESHOLDS["plus_background_max_exclusive"],
        "factual_miss_target": metrics[
            "factual_miss_target_min"
        ] > THRESHOLDS["factual_miss_target_min_exclusive"],
        "factual_miss_background": metrics[
            "factual_miss_background_max"
        ] < THRESHOLDS["factual_miss_background_max_exclusive"],
        "factual_no_miss": metrics[
            "factual_no_miss_max"
        ] < THRESHOLDS["factual_no_miss_max_exclusive"],
        "clean_D": metrics["clean_D_mean"] >= THRESHOLDS[
            "clean_D_mean_min_inclusive"
        ],
        "clean_H": metrics["clean_H_max_abs"] <= THRESHOLDS[
            "clean_H_max_abs_max_inclusive"
        ],
        "clean_G": metrics["clean_G_max_abs"] <= THRESHOLDS[
            "clean_G_max_abs_max_inclusive"
        ],
        "component_H": metrics[
            "component_H_max_abs"
        ] <= THRESHOLDS["component_H_max_abs_max_inclusive"],
        "component_G": metrics[
            "component_G_max_abs"
        ] <= THRESHOLDS["component_G_max_abs_max_inclusive"],
        "dual_endpoint_gradients": all(endpoint_gradient.values()),
        "all_parameter_gradients_finite_nonzero": not gradient_failures,
        "pair_endpoints_use_one_2B_forward": (
            initial_forward_batches == (4,)
            and training_forward_contract
        ),
        "every_update_exact_three_4_state_calls": (
            training_forward_contract
        ),
        "feature_detach": (
            outcome.pair_batch.feature.grad is None
            and all(
                branch.feature.grad is None
                for branch in factual.values()
            )
        ),
        "initial_operator_contract": (
            initial_operator["all_pass"] is True
        ),
        "operator_contract": operator["all_pass"] is True,
        "geometry_contract": (
            operator["changed_support_pixels"] > 0
            and (
                family_id == LEGACY_FAMILY
                or not bool(
                    (
                        outcome.response_stratum[clean]
                        & outcome.removed_component[clean]
                    ).any()
                )
            )
        ),
    }
    return {
        "family_id": family_id,
        "case_id": case_id,
        "clean_pixels": [list(pixel) for pixel in clean_pixels],
        "input_fingerprint": _inputs_fingerprint(outcome, factual),
        "initial_decoder_fingerprint": initial_decoder_fingerprint,
        "final_decoder_fingerprint": _decoder_fingerprint(decoder),
        "updates_executed": updates,
        "metrics": metrics,
        "checks": checks,
        "all_pass": all(checks.values()),
        "endpoint_gradient": endpoint_gradient,
        "feature_detach_contract": {
            "pair_feature_requires_grad": (
                outcome.pair_batch.feature.requires_grad
            ),
            "pair_feature_gradient_is_none": (
                outcome.pair_batch.feature.grad is None
            ),
            "factual_feature_gradients_are_none": all(
                branch.feature.grad is None
                for branch in factual.values()
            ),
            "passed": checks["feature_detach"],
        },
        "gradient_contract": {
            "parameter_tensors": len(named_parameters),
            "parameters": sum(
                value.numel() for _, value in named_parameters
            ),
            "updates_checked": updates,
            "failure_count": len(gradient_failures),
            "failures": gradient_failures,
            "minimum_l2_norm": gradient_minimum,
            "maximum_l2_norm": gradient_maximum,
        },
        "optimizer_contract": optimizer_contract,
        "forward_contract": {
            "initial_paired_call_batch_sizes": list(
                initial_forward_batches
            ),
            "paired_batch_size": 2,
            "endpoint_state_count": 4,
            "uses_one_2B_endpoint_forward": (
                initial_forward_batches == (4,)
                and training_forward_contract
            ),
            "training_step_decoder_calls": int(
                first_logs["decoder_forward_calls_per_update"]
            ),
            "training_step_decoder_states": int(
                first_logs["decoder_states_per_update"]
            ),
            "training_call_count": len(training_forward_batches),
            "expected_training_call_count": updates * 3,
            "per_update_batch_sizes_expected": [4, 4, 4],
            "first_update_batch_sizes": list(
                training_forward_patterns[0]
            ),
            "last_update_batch_sizes": list(
                training_forward_patterns[-1]
            ),
            "all_updates_exact_three_4_state_calls": (
                training_forward_contract
            ),
        },
        "initial_operator_audit": initial_operator,
        "operator_audit": operator,
        "first_update_logs": first_logs,
        "last_update_logs": last_logs,
    }


def evaluate() -> dict[str, object]:
    """Return the deterministic six-case PECO development regression."""

    protocol_binding = _load_protocol_binding()
    protocol_binding["implementation_closure"] = (
        _load_implementation_closure(protocol_binding)
    )
    previous_threads = torch.get_num_threads()
    previous_deterministic = (
        torch.are_deterministic_algorithms_enabled()
    )
    try:
        case_threads = min(previous_threads, 2)
        torch.set_num_threads(case_threads)
        torch.use_deterministic_algorithms(True)
        cases = [_case(*case) for case in CONSERVATIVE_TOY_CASES]
        objective_audit = _objective_contract_audit()
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    passed_families = {
        family_id
        for family_id, _, _ in CONSERVATIVE_TOY_CASES
        if all(
            case["all_pass"] is True
            for case in cases
            if case["family_id"] == family_id
        )
    }
    all_pass = (
        all(case["all_pass"] is True for case in cases)
        and len(passed_families) == 2
        and objective_audit["all_pass"] is True
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "stage_id": STAGE_ID,
        "correction_id": CORRECTION_ID,
        "evidentiary_status": EVIDENTIARY_STATUS,
        "protocol_binding": protocol_binding,
        "decision": (
            "PECO_V10_DEVELOPMENT_REGRESSION_PASS"
            if all_pass
            else "PECO_V10_DEVELOPMENT_REGRESSION_FAIL"
        ),
        "correction_decision": (
            "PECO_V10_DEVELOPMENT_R2_CORRECTION_PASS"
            if all_pass
            else "PECO_V10_DEVELOPMENT_R2_CORRECTION_FAIL"
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
            "optimizer_contract": cases[0]["optimizer_contract"],
            "pre_mask_paired_loss": True,
            "topology_changed": False,
            "response_loss_changed": True,
            "zero_anchor_factual_changed": False,
            "training_step_changed": False,
            "inference_changed": False,
            "component_null_transition_group": (
                "zero_risk_only_when_D_is_empty"
            ),
            "component_null_complete_pair_loss": (
                "unchanged_plus_anchor_plus_zero_transition"
            ),
            "crossing_boundary": (
                "training_residual_logit_zero_"
                "not_frozen_calibration_threshold"
            ),
            "correction_scope": (
                "restore_frozen_v8_relative_roundoff_audit_only"
            ),
            "mass_conservation_rule": (
                "max(abs(sum_phase(e)-M)/"
                "clamp_min(abs(M),1))<=1e-6"
            ),
        },
        "passed_case_count": sum(
            case["all_pass"] is True for case in cases
        ),
        "failed_case_count": sum(
            case["all_pass"] is not True for case in cases
        ),
        "passed_family_count": len(passed_families),
        "cases": cases,
        "objective_contract_audit": objective_audit,
        "runtime": {
            "torch_version": str(torch.__version__),
            "device": "cpu",
            "deterministic_algorithms": True,
            "maximum_torch_threads": 2,
            "observed_case_torch_threads": case_threads,
            "reproducibility_scope": (
                "same_runtime_deterministic_execution_"
                "not_cross_environment_byte_identity"
            ),
        },
        "execution_boundary": {
            "dataset_accessed": False,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "detection_performance_evaluated": False,
            "independent_confirmation_established": False,
            "exposure_matched_confirmation_authorized": False,
            "real_bounded_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_detector_authorized": False,
        },
        "interpretation": (
            "candidate_selection_development_regression_"
            "not_independent_confirmation_or_detection_performance"
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
    if args.output.exists():
        raise FileExistsError(
            f"refusing to run because output already exists: {args.output}"
        )
    result = evaluate()
    _write_result(args.output, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "evidentiary_status": result["evidentiary_status"],
                "result_fingerprint": result["result_fingerprint"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["all_pass"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
