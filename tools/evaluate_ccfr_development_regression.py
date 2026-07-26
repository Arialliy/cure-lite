#!/usr/bin/env python3
"""Run the dataset-free CCFR-v11 six-case development regression.

These cases were already used by earlier candidates.  A pass is therefore an
implementation/learnability regression only, never an independent
confirmation or a detection-performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
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
from cure_lite.coverage_feature_release_decoder import (  # noqa: E402
    CURELiteCoverageFeatureReleaseDecoder,
)
from cure_lite.ccfr_development_inputs import (  # noqa: E402
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


SCHEMA_VERSION = "cure-lite-ccfr-v11-development-result-v1"
METHOD_ID = "ccfr_v11"
STAGE_ID = "dataset_free_development_regression"
EVIDENTIARY_STATUS = (
    "reused_case_implementation_and_learnability_regression_"
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
    "clean_D_plus_max_exclusive": 0.05,
    "clean_D_minus_min_exclusive": 0.95,
    "D_wrong_direction_pixel_count_max_inclusive": 0,
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
    / "coverage_conditioned_feature_release_v11"
)
_CONFIG = _PROTOCOL / "development_regression_config.json"
_HOLDOUT_RECEIPT = _PROTOCOL / "exposure_holdout_design_receipt.json"
_CANONICAL_RESULT = _PROTOCOL / "development_regression_result_r1.json"
_CANONICAL_ATTEMPT = _PROTOCOL / "development_regression_attempt_r1.json"
_CANONICAL_COMPLETE = _PROTOCOL / (
    "development_regression_result_r1.COMPLETE.sha256"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_SOURCE_BINDINGS = (
    "CURE_Lite_CCFR_v11_模型与代码设计.md",
    "cure_lite/__init__.py",
    "cure_lite/cache/__init__.py",
    "cure_lite/cache/base_cache.py",
    "cure_lite/cache/schema.py",
    "cure_lite/cache/state_cache.py",
    "cure_lite/calibration.py",
    "cure_lite/ccfr_development_inputs.py",
    "cure_lite/ccfr_holdout_inputs.py",
    "cure_lite/config.py",
    "cure_lite/conservative_factorized_config.py",
    "cure_lite/coverage_feature_release_config.py",
    "cure_lite/coverage_feature_release_decoder.py",
    "cure_lite/decoder.py",
    "cure_lite/factorized_config.py",
    "cure_lite/factorized_decoder.py",
    "cure_lite/frozen_base.py",
    "cure_lite/instances.py",
    "cure_lite/intervention.py",
    "cure_lite/losses.py",
    "cure_lite/matching.py",
    "cure_lite/metrics.py",
    "cure_lite/model.py",
    "cure_lite/occupancy.py",
    "cure_lite/paired_endpoint_crossing_losses.py",
    "cure_lite/paired_losses.py",
    "cure_lite/paired_outcome_losses.py",
    "cure_lite/paired_outcome_types.py",
    "cure_lite/paired_types.py",
    "cure_lite/sampling.py",
    "cure_lite/supervision.py",
    "cure_lite/train/__init__.py",
    "cure_lite/train/engine.py",
    "cure_lite/train/paired_outcome_step.py",
    "cure_lite/train/paired_step.py",
    "cure_lite/train/pools.py",
    "cure_lite/train/step.py",
    "cure_lite/types.py",
    (
        "protocols/IRSTD-1K/coverage_conditioned_feature_release_v11/"
        "exposure_holdout_design_receipt.json"
    ),
    "tests_v11/test_ccfr_development_regression.py",
    "tests_v11/test_ccfr_exposure_holdout.py",
    "tests_v11/test_ccfr_holdout_inputs.py",
    "tests_v11/test_coverage_feature_release_config.py",
    "tests_v11/test_coverage_feature_release_decoder.py",
    "tests_v11/test_coverage_feature_release_model.py",
    "tests_v11/test_coverage_feature_release_paired_integration.py",
    "tools/evaluate_ccfr_development_regression.py",
    "tools/evaluate_ccfr_exposure_holdout.py",
    "tools/__init__.py",
)

_FORBIDDEN_RUNTIME_MODULES = {
    "cure_lite.experiment.cache_pipeline",
    "cure_lite.experiment.stage_a_runner",
    "cure_lite.experiment.stage_a_m_runner",
    "cure_lite.experiment.training_pipeline",
}


def _runtime_import_boundary() -> dict[str, object]:
    """Reject real-pipeline and dataset imports before synthetic training."""

    forbidden = sorted(
        name
        for name in sys.modules
        if name in _FORBIDDEN_RUNTIME_MODULES
        or name == "datasets"
        or name.startswith("datasets.")
    )
    if forbidden:
        raise RuntimeError(
            "CCFR development runtime import boundary was crossed: "
            f"{forbidden}"
        )
    return {
        "forbidden_exact_modules": sorted(_FORBIDDEN_RUNTIME_MODULES),
        "datasets_prefix_forbidden": True,
        "observed_forbidden_modules": forbidden,
        "all_pass": True,
    }


def _local_imported_python_paths() -> set[str]:
    """Return existing repository Python sources imported in this process."""

    root = _ROOT.resolve()
    paths: set[str] = set()
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            continue
        path = Path(source)
        if path.suffix in {".pyc", ".pyo"}:
            candidate = (
                path.parent.parent / f"{path.stem.split('.')[0]}.py"
            )
            if candidate.is_file():
                path = candidate
        if not path.is_file() or path.suffix != ".py":
            continue
        try:
            relative = path.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        paths.add(str(relative))
    return paths


def _runtime_source_closure(
    source_bindings: object,
) -> dict[str, object]:
    """Require every actually imported local source to be pre-bound."""

    if not isinstance(source_bindings, dict):
        raise TypeError("runtime source_bindings must be an object")
    imported = _local_imported_python_paths()
    unbound = sorted(imported - set(source_bindings))
    if unbound:
        raise RuntimeError(
            "CCFR development runtime has unbound local imports: "
            f"{unbound}"
        )
    return {
        "imported_local_python_paths": sorted(imported),
        "imported_local_python_path_count": len(imported),
        "imported_local_python_path_fingerprint": stable_fingerprint(
            sorted(imported)
        ),
        "unbound_local_imports": unbound,
        "all_pass": True,
    }


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return (
        str(tensor.dtype).encode("utf-8")
        + repr(tuple(tensor.shape)).encode("utf-8")
        + tensor.numpy().tobytes()
    )


def _decoder_fingerprint(
    decoder: CURELiteCoverageFeatureReleaseDecoder,
) -> str:
    digest = hashlib.sha256()
    for name, value in decoder.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


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


def _adam_contract(
    optimizer: torch.optim.Adam,
) -> dict[str, object]:
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
    formula_error = float(
        (observed - expected).detach().abs().max()
    )
    objective_case = CONSERVATIVE_TOY_CASES[0]
    outcome, _ = build_conservative_toy_case(
        objective_case[0],
        objective_case[2],
    )
    shape = outcome.pair_batch.occupancy_plus.shape
    actual_plus = torch.full(
        shape,
        -0.75,
        dtype=torch.float32,
        requires_grad=True,
    )
    actual_minus = torch.full(
        shape,
        0.5,
        dtype=torch.float32,
        requires_grad=True,
    )
    actual_result = PairedEndpointCrossingLoss(LossConfig())(
        actual_plus,
        actual_minus,
        outcome.completion_plus,
        outcome.pair_batch.occupancy_plus,
        outcome.gt_union,
        outcome.pair_batch.label_increment,
        outcome.pair_batch.image_valid_mask,
        outcome.intervention_footprint,
    )
    response = actual_result["response_stratum"]
    zero_response = (
        actual_result["local_zero_stratum"]
        | actual_result["global_zero_stratum"]
    )
    actual_response_error = 0.5 * (
        F.softplus(actual_plus) + F.softplus(-actual_minus)
    )
    active_response_means = torch.stack(
        tuple(
            actual_response_error[index][response[index]].mean()
            for index in range(shape[0])
            if bool(response[index].any())
        )
    ).mean()
    actual_formula_error = float(
        (
            actual_result["response_stratum_loss"]
            - active_response_means
        )
        .detach()
        .abs()
    )
    actual_response_gradients = torch.autograd.grad(
        actual_result["response_stratum_loss"],
        (actual_plus, actual_minus),
        retain_graph=True,
    )
    actual_zero_gradients = torch.autograd.grad(
        actual_result["zero_risk"],
        (actual_plus, actual_minus),
    )
    checks = {
        "formula_matches_bce_to_float64": formula_error <= 1.0e-15,
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
        "actual_criterion_response_formula_matches": (
            actual_formula_error <= 1.0e-7
        ),
        "actual_criterion_D_plus_gradient_positive": bool(
            torch.all(actual_response_gradients[0][response] > 0.0)
        ),
        "actual_criterion_D_minus_gradient_negative": bool(
            torch.all(actual_response_gradients[1][response] < 0.0)
        ),
        "actual_criterion_zero_plus_gradient_aligns": bool(
            torch.all(actual_zero_gradients[0][zero_response] < 0.0)
        ),
        "actual_criterion_zero_minus_gradient_aligns": bool(
            torch.all(actual_zero_gradients[1][zero_response] > 0.0)
        ),
        "actual_criterion_values_and_gradients_finite": all(
            bool(torch.isfinite(value).all())
            for value in (
                actual_result["total"],
                actual_result["response_stratum_loss"],
                actual_result["zero_risk"],
                *actual_response_gradients,
                *actual_zero_gradients,
            )
        ),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "formula_max_abs_error": formula_error,
        "actual_criterion_response_formula_max_abs_error": (
            actual_formula_error
        ),
        "actual_response_pixel_count": int(response.sum()),
        "actual_zero_response_pixel_count": int(zero_response.sum()),
        "plus_gradient": [
            float(value) for value in plus_gradient.detach()
        ],
        "minus_gradient": [
            float(value) for value in minus_gradient.detach()
        ],
    }


def _expected_cases() -> list[dict[str, object]]:
    return [
        {
            "family_id": family_id,
            "case_id": case_id,
            "clean_pixels": [list(pixel) for pixel in clean_pixels],
        }
        for family_id, case_id, clean_pixels in CONSERVATIVE_TOY_CASES
    ]


def _load_holdout_receipt() -> dict[str, object]:
    receipt = json.loads(_HOLDOUT_RECEIPT.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise TypeError("CCFR holdout receipt must be an object")
    unsigned = dict(receipt)
    fingerprint = unsigned.pop("receipt_fingerprint", None)
    if (
        not isinstance(fingerprint, str)
        or not _SHA256_PATTERN.fullmatch(fingerprint)
        or stable_fingerprint(unsigned) != fingerprint
    ):
        raise RuntimeError("CCFR holdout receipt fingerprint differs")
    if receipt.get("method_id") != METHOD_ID:
        raise RuntimeError("CCFR holdout method differs")
    if receipt.get("stage_id") != (
        "dataset_free_exposure_holdout_confirmation"
    ):
        raise RuntimeError("CCFR holdout stage differs")
    if receipt.get("status") != (
        "FROZEN_BEFORE_DEVELOPMENT_REGRESSION_RESULT"
    ):
        raise RuntimeError("CCFR holdout was not frozen before development")
    order = receipt.get("execution_order")
    if not isinstance(order, dict) or order.get(
        "generator_and_source_closure_must_be_frozen_before_development_result"
    ) is not True:
        raise RuntimeError("CCFR holdout execution order differs")
    return {
        "repo_path": str(_HOLDOUT_RECEIPT.relative_to(_ROOT)),
        "file_sha256": file_sha256(_HOLDOUT_RECEIPT),
        "receipt_fingerprint": fingerprint,
        "status": receipt["status"],
        "design_seed": receipt.get("design_seed"),
    }


def _load_protocol() -> dict[str, object]:
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("CCFR development config must be an object")
    unsigned = dict(config)
    observed_fingerprint = unsigned.pop("config_fingerprint", None)
    if (
        not isinstance(observed_fingerprint, str)
        or stable_fingerprint(unsigned) != observed_fingerprint
    ):
        raise RuntimeError("CCFR development config fingerprint differs")

    exact = {
        "schema_version": (
            "cure-lite-ccfr-v11-development-config-v1"
        ),
        "method_id": METHOD_ID,
        "stage_id": STAGE_ID,
        "status": "FROZEN_BEFORE_SINGLE_DEVELOPMENT_RUN",
        "evidentiary_status": EVIDENTIARY_STATUS,
        "cases": _expected_cases(),
        "optimization": {
            "seed": FROZEN_SEED,
            "optimizer": "adam",
            "updates_per_case": FROZEN_UPDATES,
            "learning_rate": FROZEN_LEARNING_RATE,
            "weight_decay": 0.0,
            "device": "cpu",
            "torch_threads": 2,
            "automatic_retry_allowed": False,
        },
        "decoder": {
            "feature_channels": 8,
            "feature_stride": 4,
            "expected_parameter_tensors": EXPECTED_PARAMETER_TENSORS,
            "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
            "topology_changed": False,
            "joint_state_changed": True,
            "extra_module_added": False,
        },
        "thresholds": THRESHOLDS,
        "decision_rule": {
            "required_passed_case_count": 6,
            "required_passed_family_count": 2,
            "per_case_all_checks_required": True,
            "mean_cannot_override_case_failure": True,
            "pass_decision": "CCFR_V11_DEVELOPMENT_REGRESSION_PASS",
            "fail_decision": "CCFR_V11_DEVELOPMENT_REGRESSION_FAIL",
            "pass_scope": (
                "implementation_and_learnability_regression_only"
            ),
        },
        "execution_boundary": {
            "dataset_access_allowed": False,
            "D_R_access_allowed": False,
            "D_V_access_allowed": False,
            "D_T_access_allowed": False,
            "detection_performance_allowed": False,
            "independent_confirmation_authorized": False,
            "real_bounded_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_detector_authorized": False,
        },
    }
    for name, expected in exact.items():
        if config.get(name) != expected:
            raise RuntimeError(f"CCFR development {name} differs")

    source_bindings = config.get("source_bindings")
    if not isinstance(source_bindings, dict):
        raise TypeError("CCFR source_bindings must be an object")
    if set(source_bindings) != set(REQUIRED_SOURCE_BINDINGS):
        raise RuntimeError("CCFR exact source binding set differs")
    root = _ROOT.resolve()
    for repo_path, expected_hash in source_bindings.items():
        if not isinstance(repo_path, str) or not isinstance(
            expected_hash,
            str,
        ):
            raise TypeError("CCFR source binding is invalid")
        relative = Path(repo_path)
        if (
            not repo_path
            or "\\" in repo_path
            or relative.is_absolute()
            or relative.as_posix() != repo_path
            or any(part in {"", ".", ".."} for part in relative.parts)
            or not _SHA256_PATTERN.fullmatch(expected_hash)
        ):
            raise RuntimeError("CCFR source binding path/hash is invalid")
        resolved = (_ROOT / relative).resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError("CCFR source binding escapes the repository")
        if file_sha256(resolved) != expected_hash:
            raise RuntimeError(
                f"CCFR bound source differs: {repo_path}"
            )
    holdout_binding = _load_holdout_receipt()
    return {
        "repo_path": str(_CONFIG.relative_to(_ROOT)),
        "file_sha256": file_sha256(_CONFIG),
        "config_fingerprint": observed_fingerprint,
        "source_bindings": source_bindings,
        "pre_frozen_holdout_binding": holdout_binding,
    }


def _operator_audit(
    decoder: CURELiteCoverageFeatureReleaseDecoder,
    outcome: object,
) -> dict[str, object]:
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
        repeated = decoder(
            batch.feature.clone(),
            batch.occupancy_plus.clone(),
        )
        zero_feature = torch.zeros_like(batch.feature)
        zero_plus = decoder(zero_feature, batch.occupancy_plus)
        zero_minus = decoder(zero_feature, batch.occupancy_minus)

        count_release = (
            plus.local_occupancy_count
            - minus.local_occupancy_count
        ) > 0.0
        release_delta = minus.feature_release - plus.feature_release
        phase_count = decoder.feature_stride**2
        changed_output = F.pixel_shuffle(
            count_release.expand(
                -1,
                phase_count,
                -1,
                -1,
            ).to(torch.float32),
            decoder.feature_stride,
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
        response = outcome.response_stratum
        release_identity_error = max(
            float(
                (
                    state.released_stem_feature
                    - state.stem_feature * state.feature_release
                ).abs().max()
            )
            for state in (plus, minus)
        )
        finite = all(
            bool(torch.isfinite(value).all())
            for state in (plus, minus)
            for value in (
                state.stem_feature,
                state.feature_release,
                state.released_stem_feature,
                state.trunk_feature,
                state.baseline_logits,
                state.raw_phase_evidence,
                state.common_mode_phase_evidence,
                state.budget_margin,
                state.evidence_budget,
                state.phase_allocation,
                state.allocated_phase_evidence,
                state.evidence,
                state.logits,
            )
        )
        changed_release = count_release.expand_as(
            plus.released_stem_feature
        )
        changed_latent_magnitude = _maximum(
            (
                minus.released_stem_feature
                - plus.released_stem_feature
            )[changed_release].abs()
        )
        outside = ~changed_output
        checks = {
            "repeat_forward_exact": torch.equal(
                repeated,
                plus.logits,
            ),
            "stem_feature_is_endpoint_invariant": torch.equal(
                plus.stem_feature,
                minus.stem_feature,
            ),
            "release_is_deletion_monotone": bool(
                torch.all(release_delta >= 0.0)
            ),
            "release_changes_only_with_count": torch.equal(
                release_delta[~count_release],
                torch.zeros_like(release_delta[~count_release]),
            ),
            "released_stem_equation_exact": (
                release_identity_error == 0.0
            ),
            "joint_latent_state_changes": (
                changed_latent_magnitude > 0.0
            ),
            "zero_feature_occupancy_control_exact": torch.equal(
                zero_plus,
                zero_minus,
            ),
            "response_is_inside_count_support": not bool(
                (response & ~changed_output).any()
            ),
            "all_fields_finite": finite,
        }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "release_identity_max_abs_error": release_identity_error,
        "changed_native_count_cells": int(count_release.sum()),
        "changed_output_support_pixels": int(changed_output.sum()),
        "changed_latent_max_abs": changed_latent_magnitude,
        "outside_count_support_probability_max_abs_delta": _maximum(
            probability_delta[outside].abs()
        ),
        "zero_feature_control_max_abs_delta": _maximum(
            (zero_minus - zero_plus).abs()
        ),
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

        decoder = CURELiteCoverageFeatureReleaseDecoder(
            feature_channels=8,
            feature_stride=4,
        )
        initial_decoder_fingerprint = _decoder_fingerprint(decoder)
        initial_operator = _operator_audit(decoder, outcome)
        named_parameters = tuple(decoder.named_parameters())
        if len(named_parameters) != EXPECTED_PARAMETER_TENSORS:
            raise AssertionError("CCFR parameter tensor count differs")
        if sum(value.numel() for _, value in named_parameters) != (
            EXPECTED_PARAMETER_COUNT
        ):
            raise AssertionError("CCFR parameter count differs")

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
            step_contract_failures: list[dict[str, object]] = []
            expected_step_logs = {
                "factual_miss/states": 4,
                "factual_no_miss/states": 4,
                "outcome/pairs": 2,
                "outcome/endpoints": 4,
                "outcome/clean_pairs": 1,
                "outcome/component_null_pairs": 1,
                "decoder_forward_calls_per_update": 3,
                "decoder_states_per_update": 12,
                "backward_calls": 1,
                "optimizer_steps": 1,
            }
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
                observed_step_logs = {
                    name: logs.get(name) for name in expected_step_logs
                }
                if observed_step_logs != expected_step_logs:
                    step_contract_failures.append(
                        {
                            "update": update,
                            "expected": expected_step_logs,
                            "observed": observed_step_logs,
                        }
                    )
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

            training_forward_batches = tuple(
                forward_batches[training_forward_start:]
            )
            patterns = tuple(
                training_forward_batches[index : index + 3]
                for index in range(
                    0,
                    len(training_forward_batches),
                    3,
                )
            )
            forward_contract = (
                initial_forward_batches == (4,)
                and len(training_forward_batches) == updates * 3
                and len(patterns) == updates
                and all(pattern == (4, 4, 4) for pattern in patterns)
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
                miss_logits = decoder(
                    factual["factual_miss"].feature,
                    factual["factual_miss"].occupancy,
                )
                no_miss_logits = decoder(
                    factual["factual_no_miss"].feature,
                    factual["factual_no_miss"].occupancy,
                )
                miss_score = torch.sigmoid(miss_logits)
                no_miss_score = torch.sigmoid(no_miss_logits)
                final_factual_miss_loss = absolute(
                    miss_logits,
                    factual["factual_miss"].target,
                    factual["factual_miss"].valid_mask,
                )["total"]
                final_factual_no_miss_loss = absolute(
                    no_miss_logits,
                    factual["factual_no_miss"].target,
                    factual["factual_no_miss"].valid_mask,
                )["total"]
                final_pair_loss = criterion(
                    logits_plus,
                    logits_minus,
                    outcome.completion_plus,
                    outcome.pair_batch.occupancy_plus,
                    outcome.gt_union,
                    outcome.pair_batch.label_increment,
                    outcome.pair_batch.image_valid_mask,
                    outcome.intervention_footprint,
                )["total"]
                final_total_loss = (
                    final_factual_miss_loss
                    + final_factual_no_miss_loss
                    + final_pair_loss
                )
        finally:
            handle.remove()

        operator = _operator_audit(decoder, outcome)

    if first_logs is None or last_logs is None:
        raise AssertionError("CCFR case produced no training logs")
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
    no_miss_valid = factual["factual_no_miss"].valid_mask
    stratum_counts = {
        "plus_completion": int(outcome.completion_plus.sum()),
        "plus_anchor_background": int(anchor_background.sum()),
        "factual_miss_target": int(miss_target.sum()),
        "factual_miss_background": int(miss_background.sum()),
        "factual_no_miss_valid": int(no_miss_valid.sum()),
        "clean_D": int(clean_D.sum()),
        "clean_H": int(clean_H.sum()),
        "clean_G": int(clean_G.sum()),
        "component_H": int(component_H.sum()),
        "component_G": int(component_G.sum()),
    }
    empty_strata = sorted(
        name for name, count in stratum_counts.items() if count <= 0
    )
    if empty_strata:
        raise AssertionError(
            "CCFR required evaluation strata are empty: "
            + ",".join(empty_strata)
        )
    metrics = {
        "total_loss": float(final_total_loss),
        "last_update_pre_step_total_loss": float(last_logs["total"]),
        "final_factual_miss_loss": float(final_factual_miss_loss),
        "final_factual_no_miss_loss": float(
            final_factual_no_miss_loss
        ),
        "final_pair_loss": float(final_pair_loss),
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
        "factual_no_miss_max": _maximum(
            no_miss_score[no_miss_valid]
        ),
        "clean_D_mean": float(delta[clean][clean_D].mean()),
        "clean_D_plus_max": _maximum(score_plus[clean][clean_D]),
        "clean_D_minus_min": _minimum(score_minus[clean][clean_D]),
        "D_wrong_direction_pixel_count": int(
            (delta[clean][clean_D] < 0.0).sum()
        ),
        "clean_H_max_abs": _maximum(
            delta[clean][clean_H].abs()
        ),
        "clean_H_positive_max": _maximum(
            torch.relu(delta[clean][clean_H])
        ),
        "clean_H_negative_max": _maximum(
            torch.relu(-delta[clean][clean_H])
        ),
        "clean_G_max_abs": _maximum(
            delta[clean][clean_G].abs()
        ),
        "clean_G_positive_max": _maximum(
            torch.relu(delta[clean][clean_G])
        ),
        "clean_G_negative_max": _maximum(
            torch.relu(-delta[clean][clean_G])
        ),
        "component_H_max_abs": _maximum(
            delta[component][component_H].abs()
        ),
        "component_H_positive_max": _maximum(
            torch.relu(delta[component][component_H])
        ),
        "component_H_negative_max": _maximum(
            torch.relu(-delta[component][component_H])
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
        "clean_D_plus": metrics[
            "clean_D_plus_max"
        ] < THRESHOLDS["clean_D_plus_max_exclusive"],
        "clean_D_minus": metrics[
            "clean_D_minus_min"
        ] > THRESHOLDS["clean_D_minus_min_exclusive"],
        "D_direction": metrics[
            "D_wrong_direction_pixel_count"
        ] <= THRESHOLDS[
            "D_wrong_direction_pixel_count_max_inclusive"
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
        "pair_endpoints_use_one_2B_forward": forward_contract,
        "every_update_exact_three_4_state_calls": forward_contract,
        "every_update_exact_4_4_2_step_contract": (
            not step_contract_failures
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
            operator["changed_output_support_pixels"] > 0
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
        "stratum_counts": stratum_counts,
        "checks": checks,
        "all_pass": all(checks.values()),
        "endpoint_gradient": endpoint_gradient,
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
            "training_call_count": len(training_forward_batches),
            "expected_training_call_count": updates * 3,
            "per_update_batch_sizes_expected": [4, 4, 4],
            "first_update_batch_sizes": list(patterns[0]),
            "last_update_batch_sizes": list(patterns[-1]),
            "all_updates_exact_three_4_state_calls": forward_contract,
            "step_log_contract_expected": expected_step_logs,
            "step_log_contract_failure_count": len(
                step_contract_failures
            ),
            "step_log_contract_failures": step_contract_failures,
        },
        "initial_operator_audit": initial_operator,
        "operator_audit": operator,
        "first_update_logs": first_logs,
        "last_update_logs": last_logs,
    }


def evaluate(
    *,
    attempt_binding: dict[str, object] | None = None,
) -> dict[str, object]:
    protocol_binding = _load_protocol()
    runtime_import_boundary = _runtime_import_boundary()
    runtime_import_boundary["local_source_closure"] = (
        _runtime_source_closure(protocol_binding["source_bindings"])
    )
    previous_threads = torch.get_num_threads()
    previous_deterministic = (
        torch.are_deterministic_algorithms_enabled()
    )
    try:
        case_threads = 2
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
        "evidentiary_status": EVIDENTIARY_STATUS,
        "protocol_binding": protocol_binding,
        "attempt_binding": (
            {
                "execution_mode": "IN_MEMORY_TEST_ONLY",
                "canonical_attempt_consumed": False,
            }
            if attempt_binding is None
            else attempt_binding
        ),
        "decision": (
            "CCFR_V11_DEVELOPMENT_REGRESSION_PASS"
            if all_pass
            else "CCFR_V11_DEVELOPMENT_REGRESSION_FAIL"
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
            "joint_state_changed": True,
            "loss_changed_from_v10": False,
            "training_step_changed": False,
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
        "objective_contract_audit": objective_audit,
        "runtime_import_boundary": runtime_import_boundary,
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
            "real_bounded_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_detector_authorized": False,
        },
        "interpretation": (
            "reused_case_implementation_and_learnability_regression_"
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


def _write_text_create_only(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _attempt_payload(
    protocol_binding: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "cure-lite-ccfr-v11-development-attempt-v1",
        "method_id": METHOD_ID,
        "stage_id": STAGE_ID,
        "attempt_id": "r1",
        "status": "STARTED_CREATE_ONLY",
        "canonical_result": str(_CANONICAL_RESULT.relative_to(_ROOT)),
        "canonical_complete": str(
            _CANONICAL_COMPLETE.relative_to(_ROOT)
        ),
        "automatic_retry_allowed": False,
        "protocol_binding": protocol_binding,
    }
    payload["attempt_fingerprint"] = stable_fingerprint(payload)
    return payload


def _attempt_binding(attempt: dict[str, object]) -> dict[str, object]:
    return {
        "repo_path": str(_CANONICAL_ATTEMPT.relative_to(_ROOT)),
        "file_sha256": file_sha256(_CANONICAL_ATTEMPT),
        "attempt_fingerprint": attempt["attempt_fingerprint"],
        "attempt_id": "r1",
    }


def _load_attempt(
    protocol_binding: dict[str, object],
) -> dict[str, object]:
    attempt = json.loads(_CANONICAL_ATTEMPT.read_text(encoding="utf-8"))
    if not isinstance(attempt, dict):
        raise TypeError("CCFR development attempt must be an object")
    unsigned = dict(attempt)
    observed = unsigned.pop("attempt_fingerprint", None)
    if (
        not isinstance(observed, str)
        or not _SHA256_PATTERN.fullmatch(observed)
        or stable_fingerprint(unsigned) != observed
        or attempt != _attempt_payload(protocol_binding)
    ):
        raise RuntimeError("CCFR development attempt differs")
    return _attempt_binding(attempt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=_CANONICAL_RESULT,
    )
    args = parser.parse_args(argv)
    if args.output.resolve() != _CANONICAL_RESULT.resolve():
        raise ValueError(
            "CCFR development output must use the canonical r1 path"
        )
    existing = tuple(
        path
        for path in (
            _CANONICAL_ATTEMPT,
            _CANONICAL_RESULT,
            _CANONICAL_COMPLETE,
        )
        if path.exists()
    )
    if existing:
        raise FileExistsError(
            "refusing to run because a canonical r1 artifact exists: "
            + ",".join(str(path) for path in existing)
        )
    protocol_binding = _load_protocol()
    _write_result(
        _CANONICAL_ATTEMPT,
        _attempt_payload(protocol_binding),
    )
    attempt_binding = _load_attempt(protocol_binding)
    result = evaluate(attempt_binding=attempt_binding)
    _write_result(_CANONICAL_RESULT, result)
    result_sha256 = file_sha256(_CANONICAL_RESULT)
    complete_payload = (
        f"{result_sha256}  {_CANONICAL_RESULT.name}\n"
        f"attempt_sha256={attempt_binding['file_sha256']}\n"
        "attempt_fingerprint="
        f"{attempt_binding['attempt_fingerprint']}\n"
        f"config_sha256={protocol_binding['file_sha256']}\n"
        "config_fingerprint="
        f"{protocol_binding['config_fingerprint']}\n"
        f"result_fingerprint={result['result_fingerprint']}\n"
    )
    _write_text_create_only(_CANONICAL_COMPLETE, complete_payload)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "evidentiary_status": result["evidentiary_status"],
                "result_fingerprint": result["result_fingerprint"],
                "output": str(_CANONICAL_RESULT),
                "complete": str(_CANONICAL_COMPLETE),
                "file_sha256": result_sha256,
            },
            sort_keys=True,
        )
    )
    return 0 if result["all_pass"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
