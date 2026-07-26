#!/usr/bin/env python3
"""Run the frozen dataset-free CCFR-v11 exposure holdout confirmation.

The holdout is independent of the six reused development cases.  It uses the
new 5x5/20x20 population and schedule frozen in
``exposure_holdout_design_receipt.json``.  The canonical command consumes one
attempt before optimization, writes one create-only result, and seals that
result with a GNU-sha256-style COMPLETE file.

This evaluator never imports a dataset, detector, cache pipeline, Stage-A
runner, or detection metric.  A pass is model-code confirmation only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.config import LossConfig  # noqa: E402
from cure_lite.conservative_factorized_decoder import (  # noqa: E402
    CURELiteConservativeFactorizedDecoder,
)
from cure_lite.coverage_feature_release_decoder import (  # noqa: E402
    CURELiteCoverageFeatureReleaseDecoder,
)
from cure_lite.ccfr_holdout_inputs import (  # noqa: E402
    CLEAN_PAIR_COUNT,
    CLEAN_SLOT_COUNT,
    COMPONENT_NULL_PAIR_COUNT,
    COMPONENT_NULL_SLOT_COUNT,
    DESIGN_SEED,
    FACTUAL_BATCH_SIZE,
    FACTUAL_EXPOSURES_PER_STATE,
    FACTUAL_POPULATION_SIZE,
    FEATURE_CHANNELS,
    FEATURE_HEIGHT,
    FEATURE_STRIDE,
    FEATURE_WIDTH,
    GROUP_COUNTS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PAIR_BATCH_SIZE,
    TOTAL_PAIR_SLOTS,
    UPDATE_COUNT,
    CCFRHoldoutPairSpec,
    CCFRHoldoutUpdate,
    build_ccfr_holdout_factual_batches,
    build_ccfr_holdout_factual_population,
    build_ccfr_holdout_outcome_batch,
    build_ccfr_holdout_pair_specs,
    build_ccfr_holdout_schedule,
    build_ccfr_holdout_strata,
    catalog_fingerprint,
    factual_indices_for_update,
    factual_population_fingerprint,
    factual_schedule_fingerprint,
    holdout_fingerprint,
    schedule_fingerprint,
)
from cure_lite.losses import CURELiteLoss  # noqa: E402
from cure_lite.paired_endpoint_crossing_losses import (  # noqa: E402
    PairedEndpointCrossingLoss,
)
from cure_lite.train.paired_outcome_step import (  # noqa: E402
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import _paired_endpoint_logits  # noqa: E402


SCHEMA_VERSION = "cure-lite-ccfr-v11-exposure-holdout-result-r1-v1"
ATTEMPT_SCHEMA_VERSION = (
    "cure-lite-ccfr-v11-exposure-holdout-attempt-r1-v1"
)
METHOD_ID = "ccfr_v11"
COMPARATOR_ID = "cc_sea_v8"
STAGE_ID = "dataset_free_exposure_holdout_confirmation"
EXPECTED_PARAMETER_TENSORS = 6
EXPECTED_PARAMETER_COUNT = 2593
EXPECTED_IMPLEMENTATION_FINGERPRINTS = {
    "holdout": "3b81cc8cfd4d156ff6b711b1f8163dbb7ee2395d0d7934c425be33bee96ef1e2",
    "catalog": "ae4f853d588b7ebf514541775388443cc88216427d93362a0af2444a673a7a8e",
    "schedule": "bb631cb68ff0d4baeb68151e3aab089728873af32920118c97f0cb922c0a010a",
    "factual_population": "1521453889fc38954f39c6629b3894d0e665ca6e7fa4f1ca0d5999bcbca70792",
    "factual_schedule": "18d1a702d62299d072328c407233cfc5f65e94ebd0eaa71eb2c7004af1764073",
}
FROZEN_DECODER_SEED = 42
FROZEN_LEARNING_RATE = 0.001
FROZEN_BETAS = (0.9, 0.999)
FROZEN_EPSILON = 1.0e-8
FROZEN_WEIGHT_DECAY = 0.0
FROZEN_AMSGRAD = False
FROZEN_MAXIMIZE = False
FROZEN_FOREACH = None
FROZEN_CAPTURABLE = False
FROZEN_DIFFERENTIABLE = False
FROZEN_FUSED = None
FROZEN_DECOUPLED_WEIGHT_DECAY = False
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

THRESHOLDS = {
    "population_total_loss_max_exclusive": 0.1,
    "plus_completion_min_exclusive": 0.95,
    "plus_background_max_exclusive": 0.05,
    "factual_miss_target_min_exclusive": 0.95,
    "factual_miss_background_max_exclusive": 0.05,
    "factual_no_miss_max_exclusive": 0.05,
    "clean_D_delta_mean_min_inclusive": 0.8,
    "clean_D_plus_max_exclusive": 0.05,
    "clean_D_minus_min_exclusive": 0.95,
    "D_wrong_direction_pixel_count_max_inclusive": 0,
    "zero_H_max_abs_max_inclusive": 0.05,
    "zero_G_near_max_abs_max_inclusive": 0.05,
    "zero_G_norm_tail_max_abs_max_inclusive": 0.05,
}

_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conditioned_feature_release_v11"
)
_DESIGN_RECEIPT = _PROTOCOL / "exposure_holdout_design_receipt.json"
_DEVELOPMENT_ATTEMPT = _PROTOCOL / "development_regression_attempt_r1.json"
_DEVELOPMENT_RESULT = _PROTOCOL / "development_regression_result_r1.json"
_DEVELOPMENT_COMPLETE = _PROTOCOL / (
    "development_regression_result_r1.COMPLETE.sha256"
)
_CANONICAL_ATTEMPT = _PROTOCOL / "exposure_holdout_attempt_r1.json"
_CANONICAL_RESULT = _PROTOCOL / "exposure_holdout_result_r1.json"
_CANONICAL_COMPLETE = _PROTOCOL / (
    "exposure_holdout_result_r1.COMPLETE.sha256"
)

_SOURCE_PATHS = (
    "CURE_Lite_CCFR_v11_模型与代码设计.md",
    "cure_lite/__init__.py",
    "cure_lite/cache/__init__.py",
    "cure_lite/cache/base_cache.py",
    "cure_lite/cache/schema.py",
    "cure_lite/cache/state_cache.py",
    "cure_lite/calibration.py",
    "cure_lite/ccfr_development_inputs.py",
    "cure_lite/config.py",
    "cure_lite/conservative_factorized_config.py",
    "cure_lite/conservative_factorized_decoder.py",
    "cure_lite/coverage_feature_release_config.py",
    "cure_lite/coverage_feature_release_decoder.py",
    "cure_lite/crossing_factorized_config.py",
    "cure_lite/crossing_factorized_decoder.py",
    "cure_lite/decoder.py",
    "cure_lite/ccfr_holdout_inputs.py",
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
    "tests_v11/test_ccfr_development_regression.py",
    "tests_v11/test_ccfr_exposure_holdout.py",
    "tests_v11/test_ccfr_holdout_inputs.py",
    "tools/evaluate_ccfr_development_regression.py",
    "tools/__init__.py",
    "tools/evaluate_ccfr_exposure_holdout.py",
)

_EXPECTED_GROUPS = {
    "clean_adjacent_cell_1px": 35,
    "clean_adjacent_cell_3px": 34,
    "clean_multicount_2to1": 34,
    "clean_multicount_3to2": 34,
    "clean_same_cell_1px": 35,
    "clean_same_cell_3px": 34,
    "component_null_block": 8,
    "component_null_sparse": 8,
}

_FORBIDDEN_RUNTIME_MODULES = {
    "cure_lite.experiment.cache_pipeline",
    "cure_lite.experiment.stage_a_runner",
    "cure_lite.experiment.stage_a_m_runner",
    "cure_lite.experiment.training_pipeline",
}


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _verify_fingerprint(
    value: dict[str, object],
    *,
    field: str,
    name: str,
) -> str:
    unsigned = dict(value)
    observed = unsigned.pop(field, None)
    if (
        not isinstance(observed, str)
        or not _SHA256_PATTERN.fullmatch(observed)
        or stable_fingerprint(unsigned) != observed
    ):
        raise RuntimeError(f"{name} fingerprint differs")
    return observed


def _validate_design_contract(receipt: dict[str, object]) -> None:
    """Reject any scientific-contract drift independently of its hash."""

    expected = {
        "schema_version": (
            "cure-lite-ccfr-v11-exposure-holdout-design-v1"
        ),
        "method_id": METHOD_ID,
        "stage_id": STAGE_ID,
        "status": "FROZEN_BEFORE_DEVELOPMENT_REGRESSION_RESULT",
        "design_seed": DESIGN_SEED,
        "seed_derivation": {
            "text": "ccfr_v11|dataset_free_exposure_confirmation_v1",
            "sha256_prefix_hex": "edad8c9c",
            "conversion": "unsigned_big_endian_uint32",
        },
        "evidence_scope": (
            "pre_frozen_dataset_free_holdout_with_replayable_one_update_"
            "wiring_smoke_not_detection_performance"
        ),
        "preformal_execution_disclosure": {
            "wiring_smoke_uses_first_schedule_update": True,
            "wiring_smoke_may_be_replayed_by_test_suite": True,
            "maximum_updates_per_smoke_invocation": 1,
            "smoke_checks_only_execution_gradients_and_4_4_2_contract": True,
            "post_update_performance_gate_or_candidate_selection_allowed": (
                False
            ),
            "full_400_update_result_unobserved_at_freeze": True,
        },
        "input_contract": {
            "feature_channels": FEATURE_CHANNELS,
            "feature_height": FEATURE_HEIGHT,
            "feature_width": FEATURE_WIDTH,
            "feature_stride": FEATURE_STRIDE,
            "output_height": OUTPUT_HEIGHT,
            "output_width": OUTPUT_WIDTH,
            "same_feature_at_both_pair_endpoints": True,
            "pre_mask_endpoint_evaluation": True,
            "pair_kind_is_model_input": False,
            "pair_kind_is_loss_dispatch": False,
        },
        "population": {
            "clean_pair_count": CLEAN_PAIR_COUNT,
            "component_null_pair_count": COMPONENT_NULL_PAIR_COUNT,
            "total_pair_count": CLEAN_PAIR_COUNT + COMPONENT_NULL_PAIR_COUNT,
            "groups": _EXPECTED_GROUPS,
            "old_six_case_tensor_reuse_allowed": False,
            "old_v10_222_role_tensor_reuse_allowed": False,
            "old_v10_schedule_reuse_allowed": False,
        },
        "geometry_rules": {
            "interior_feature_cells_only": True,
            "interior_scope": (
                "component_response_and_fixed_occupancy_cells_all_have_"
                "row_and_column_in_[1,3]"
            ),
            "feature_cell_index_rule": {
                "row": "1+((3*within_group_index+group_offset)%3)",
                "column": (
                    "1+((2*within_group_index+1+group_offset)%3)"
                ),
            },
            "group_offset_mapping": {
                "clean_same_cell_1px": 0,
                "clean_same_cell_3px": 1,
                "clean_adjacent_cell_1px": 2,
                "clean_adjacent_cell_3px": 3,
                "clean_multicount_2to1": 4,
                "clean_multicount_3to2": 5,
                "component_null_block": 6,
                "component_null_sparse": 7,
            },
            "phase_patterns": {
                "one_pixel": [[0, 3]],
                "three_pixels": [[0, 3], [1, 2], [3, 0]],
            },
            "multicount_phase_pattern_rule": (
                "even_within_group_index_uses_one_pixel_"
                "odd_uses_three_pixels"
            ),
            "same_cell": (
                "response phases and removed occupancy component "
                "project to the same feature cell"
            ),
            "adjacent_cell": (
                "response projects to a horizontal or vertical neighbor "
                "of the removed component and remains inside changed 3x3 "
                "count support"
            ),
            "multicount_2to1": (
                "fixed neighboring occupancy leaves count one after the "
                "removed component changes count two to one at every "
                "response cell"
            ),
            "multicount_3to2": (
                "two fixed neighboring occupancies leave count two after "
                "the removed component changes count three to two at "
                "every response cell"
            ),
            "component_null": (
                "occupancy endpoint changes and projected count changes "
                "but label increment is empty"
            ),
            "projection_visible_required": True,
            "response_inside_changed_count_support_required": True,
            "nonempty_norm_tail_required": True,
        },
        "strata_rules": {
            "D": "label_increment_and_image_valid_mask",
            "H": "direct_projected_intervention_footprint_and_not_D",
            "G": "image_valid_mask_and_not_D_and_not_H",
            "changed_count_support": (
                "nearest_lift_of_3x3_local_projected_occupancy_"
                "count_inequality"
            ),
            "G_near": "G_and_changed_count_support",
            "G_norm_tail": "G_and_not_changed_count_support",
            "partition_required": (
                "D_union_H_union_G_near_union_G_norm_tail_"
                "equals_image_valid_mask"
            ),
            "interpretation": (
                "G_norm_tail_is_a_conservative_far_field_audit_not_a_"
                "GroupNorm_only_causal_attribution"
            ),
        },
        "feature_rules": {
            "generator": "sha256_indexed_sparse_signal_lattice_v1",
            "base_value_formula": (
                "((uint16(sha256(seed|group|index|channel|row|column)"
                "[0:2])%257)-128)/64"
            ),
            "uint16_byte_order": "big_endian",
            "sparsity_rule": (
                "only_the_two_listed_signal_channels_at_the_designated_"
                "signal_cell_are_assigned_lattice_values_all_other_"
                "entries_are_exact_zero"
            ),
            "signal_cell_rule": (
                "clean_pair_uses_response_cell_component_null_uses_"
                "component_cell_factual_uses_its_frozen_signal_cell"
            ),
            "clean_target_signal_channels": [0, 1],
            "component_null_signal_channels": [2, 3],
            "factual_miss_signal_channels": [0, 1],
            "factual_no_miss_signal_channels": [4, 5],
            "role_associated_channel_groups_are_intentional": True,
            "zero_lattice_values_are_allowed": True,
            "feature_values_do_not_depend_on_observed_model_or_"
            "training_results": True,
        },
        "schedule": {
            "updates": UPDATE_COUNT,
            "pair_batch_size": PAIR_BATCH_SIZE,
            "total_pair_slots": TOTAL_PAIR_SLOTS,
            "clean_slots": CLEAN_SLOT_COUNT,
            "component_null_slots": COMPONENT_NULL_SLOT_COUNT,
            "clean_exposure_histogram": {"3": 85, "4": 121},
            "component_null_exposure_histogram": {"3": 3, "4": 13},
            "four_exposure_selection": (
                "per_group_stable_sha256_rank_clean_group_quotas_"
                "[21,20,20,20,20,20]_component_group_quotas_[7,6]"
            ),
            "slot_order": (
                "stable_sha256_rank_by_design_seed_role_round_and_"
                "pair_id_then_earliest_distinct_pair_source_conflict_"
                "resolution_v1"
            ),
            "conflict_resolution": (
                "pop_first_ranked_slot_then_pair_with_earliest_remaining_"
                "slot_having_distinct_pair_id_and_source_id"
            ),
            "same_pair_twice_in_update_allowed": False,
            "same_source_twice_in_update_allowed": False,
        },
        "factual_population": {
            "factual_miss_states": FACTUAL_POPULATION_SIZE,
            "factual_no_miss_states": FACTUAL_POPULATION_SIZE,
            "batch_size_per_branch": FACTUAL_BATCH_SIZE,
            "exposures_per_state": FACTUAL_EXPOSURES_PER_STATE,
            "schedule": "four_contiguous_indices_rotated_every_update_v1",
        },
        "optimization": {
            "decoder_seed": FROZEN_DECODER_SEED,
            "device": "cpu",
            "torch_threads": 2,
            "optimizer": "Adam",
            "learning_rate": FROZEN_LEARNING_RATE,
            "betas": list(FROZEN_BETAS),
            "epsilon": FROZEN_EPSILON,
            "weight_decay": FROZEN_WEIGHT_DECAY,
            "updates": UPDATE_COUNT,
            "automatic_retry_allowed": False,
        },
        "thresholds": THRESHOLDS,
        "decision_rule": {
            "all_eight_groups_required": True,
            "all_absolute_and_directional_thresholds_required": True,
            "all_400_updates_require_six_finite_nonzero_parameter_gradients": (
                True
            ),
            "mean_cannot_override_group_failure": True,
            "v8_matched_comparator_is_report_only": True,
            "pass_decision": (
                "CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_PASS"
            ),
            "fail_decision": (
                "CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_FAIL"
            ),
        },
        "execution_order": {
            "generator_and_source_closure_must_be_frozen_before_development_result": (
                True
            ),
            "development_regression_must_pass_before_confirmation_execution": (
                True
            ),
            "confirmation_may_run_once": True,
            "real_D_R_authorized_only_after_confirmation_pass": True,
        },
        "execution_boundary": {
            "dataset_access_allowed": False,
            "D_R_access_allowed": False,
            "D_V_access_allowed": False,
            "D_T_access_allowed": False,
            "detection_performance_allowed": False,
            "real_bounded_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_detector_authorized": False,
        },
        "implementation_fingerprints": EXPECTED_IMPLEMENTATION_FINGERPRINTS,
    }
    for name, expected_value in expected.items():
        if receipt.get(name) != expected_value:
            raise RuntimeError(f"CCFR holdout design differs: {name}")
    expected_top_level = set(expected) | {
        "source_bindings",
        "receipt_fingerprint",
    }
    if set(receipt) != expected_top_level:
        raise RuntimeError("CCFR holdout design top-level fields differ")


def _validate_source_bindings(
    value: object,
) -> dict[str, str]:
    """Validate the pre-frozen, receipt-owned implementation closure."""

    if not isinstance(value, dict):
        raise TypeError("CCFR holdout source_bindings must be an object")
    if set(value) != set(_SOURCE_PATHS):
        raise RuntimeError("CCFR holdout exact source binding set differs")
    root = _ROOT.resolve()
    bindings: dict[str, str] = {}
    for repo_path in _SOURCE_PATHS:
        expected_hash = value.get(repo_path)
        if not isinstance(expected_hash, str):
            raise TypeError("CCFR holdout source hash must be a string")
        relative = Path(repo_path)
        if (
            not repo_path
            or "\\" in repo_path
            or relative.is_absolute()
            or relative.as_posix() != repo_path
            or any(part in {"", ".", ".."} for part in relative.parts)
            or not _SHA256_PATTERN.fullmatch(expected_hash)
        ):
            raise RuntimeError(
                f"CCFR holdout source path/hash is invalid: {repo_path}"
            )
        resolved = (_ROOT / relative).resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError(
                f"CCFR holdout source escapes repository: {repo_path}"
            )
        if not resolved.is_file():
            raise RuntimeError(
                f"CCFR holdout bound source is absent: {repo_path}"
            )
        if file_sha256(resolved) != expected_hash:
            raise RuntimeError(
                f"CCFR holdout bound source differs: {repo_path}"
            )
        bindings[repo_path] = expected_hash
    return bindings


def _load_design_receipt() -> dict[str, object]:
    receipt = _load_object(
        _DESIGN_RECEIPT,
        name="CCFR exposure holdout design receipt",
    )
    _validate_design_contract(receipt)
    source_bindings = _validate_source_bindings(
        receipt.get("source_bindings")
    )
    fingerprint = _verify_fingerprint(
        receipt,
        field="receipt_fingerprint",
        name="CCFR exposure holdout design receipt",
    )
    return {
        "repo_path": str(_DESIGN_RECEIPT.relative_to(_ROOT)),
        "file_sha256": file_sha256(_DESIGN_RECEIPT),
        "receipt_fingerprint": fingerprint,
        "status": receipt["status"],
        "design_seed": receipt["design_seed"],
        "implementation_fingerprints": receipt[
            "implementation_fingerprints"
        ],
        "source_bindings": source_bindings,
    }


def _load_development_pass() -> dict[str, object]:
    missing = [
        path
        for path in (
            _DEVELOPMENT_ATTEMPT,
            _DEVELOPMENT_RESULT,
            _DEVELOPMENT_COMPLETE,
        )
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "CCFR development authority chain is incomplete; holdout is "
            f"not authorized: {missing}"
        )
    attempt = _load_object(
        _DEVELOPMENT_ATTEMPT,
        name="CCFR development attempt",
    )
    attempt_fingerprint = _verify_fingerprint(
        attempt,
        field="attempt_fingerprint",
        name="CCFR development attempt",
    )
    if attempt.get("schema_version") != (
        "cure-lite-ccfr-v11-development-attempt-v1"
    ):
        raise RuntimeError("CCFR development attempt schema differs")
    if (
        attempt.get("method_id") != METHOD_ID
        or attempt.get("stage_id") != "dataset_free_development_regression"
        or attempt.get("attempt_id") != "r1"
        or attempt.get("status") != "STARTED_CREATE_ONLY"
        or attempt.get("automatic_retry_allowed") is not False
        or attempt.get("canonical_result")
        != str(_DEVELOPMENT_RESULT.relative_to(_ROOT))
        or attempt.get("canonical_complete")
        != str(_DEVELOPMENT_COMPLETE.relative_to(_ROOT))
    ):
        raise RuntimeError("CCFR development attempt contract differs")
    attempt_protocol = attempt.get("protocol_binding")
    if not isinstance(attempt_protocol, dict):
        raise TypeError("CCFR development attempt protocol must be an object")

    result = _load_object(
        _DEVELOPMENT_RESULT,
        name="CCFR development result",
    )
    fingerprint = _verify_fingerprint(
        result,
        field="result_fingerprint",
        name="CCFR development result",
    )
    if (
        result.get("schema_version")
        != "cure-lite-ccfr-v11-development-result-v1"
        or
        result.get("method_id") != METHOD_ID
        or result.get("stage_id") != "dataset_free_development_regression"
        or result.get("decision")
        != "CCFR_V11_DEVELOPMENT_REGRESSION_PASS"
        or result.get("all_pass") is not True
    ):
        raise RuntimeError(
            "CCFR development did not pass; holdout is not authorized"
        )
    result_protocol = result.get("protocol_binding")
    if result_protocol != attempt_protocol:
        raise RuntimeError(
            "CCFR development result/attempt protocol bindings differ"
        )
    expected_attempt_binding = {
        "repo_path": str(_DEVELOPMENT_ATTEMPT.relative_to(_ROOT)),
        "file_sha256": file_sha256(_DEVELOPMENT_ATTEMPT),
        "attempt_fingerprint": attempt_fingerprint,
        "attempt_id": "r1",
    }
    if result.get("attempt_binding") != expected_attempt_binding:
        raise RuntimeError("CCFR development result attempt binding differs")
    cases = result.get("cases")
    if (
        result.get("passed_case_count") != 6
        or result.get("failed_case_count") != 0
        or result.get("passed_family_count") != 2
        or not isinstance(cases, list)
        or len(cases) != 6
        or any(
            not isinstance(case, dict) or case.get("all_pass") is not True
            for case in cases
        )
    ):
        raise RuntimeError("CCFR development case/family gates did not pass")
    objective_audit = result.get("objective_contract_audit")
    if (
        not isinstance(objective_audit, dict)
        or objective_audit.get("all_pass") is not True
    ):
        raise RuntimeError("CCFR development objective audit did not pass")
    expected_execution_boundary = {
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
    }
    if result.get("execution_boundary") != expected_execution_boundary:
        raise RuntimeError("CCFR development execution boundary differs")

    complete_lines = _DEVELOPMENT_COMPLETE.read_text(
        encoding="utf-8"
    ).splitlines()
    if len(complete_lines) != 6:
        raise RuntimeError("CCFR development COMPLETE shape differs")
    result_sha256 = file_sha256(_DEVELOPMENT_RESULT)
    expected_lines = [
        f"{result_sha256}  {_DEVELOPMENT_RESULT.name}",
        f"attempt_sha256={expected_attempt_binding['file_sha256']}",
        "attempt_fingerprint="
        f"{expected_attempt_binding['attempt_fingerprint']}",
        f"config_sha256={attempt_protocol.get('file_sha256')}",
        "config_fingerprint="
        f"{attempt_protocol.get('config_fingerprint')}",
        f"result_fingerprint={fingerprint}",
    ]
    if complete_lines != expected_lines:
        raise RuntimeError("CCFR development COMPLETE content differs")
    return {
        "attempt": {
            "repo_path": str(_DEVELOPMENT_ATTEMPT.relative_to(_ROOT)),
            "file_sha256": file_sha256(_DEVELOPMENT_ATTEMPT),
            "attempt_fingerprint": attempt_fingerprint,
        },
        "result": {
            "repo_path": str(_DEVELOPMENT_RESULT.relative_to(_ROOT)),
            "file_sha256": result_sha256,
            "result_fingerprint": fingerprint,
            "decision": result["decision"],
        },
        "complete": {
            "repo_path": str(_DEVELOPMENT_COMPLETE.relative_to(_ROOT)),
            "file_sha256": file_sha256(_DEVELOPMENT_COMPLETE),
            "result_file_sha256": result_sha256,
        },
        "protocol_binding": attempt_protocol,
    }


def _holdout_contract() -> dict[str, object]:
    """Rebuild and audit the new tensors and both frozen schedules."""

    specs = build_ccfr_holdout_pair_specs()
    schedule = build_ccfr_holdout_schedule(specs)
    if len(specs) != CLEAN_PAIR_COUNT + COMPONENT_NULL_PAIR_COUNT:
        raise AssertionError("CCFR holdout pair count differs")
    if Counter(spec.group_id for spec in specs) != Counter(_EXPECTED_GROUPS):
        raise AssertionError("CCFR holdout group population differs")
    if dict(GROUP_COUNTS) != _EXPECTED_GROUPS:
        raise AssertionError("CCFR exported group contract differs")
    if len(schedule) != UPDATE_COUNT:
        raise AssertionError("CCFR holdout update count differs")

    observed_exposure = Counter(
        index
        for update in schedule
        for index in update.population_indices
    )
    expected_exposure = Counter(
        {spec.population_index: spec.exposure_count for spec in specs}
    )
    if observed_exposure != expected_exposure:
        raise AssertionError("CCFR holdout pair exposures differ")
    if any(
        len(set(update.pair_ids)) != PAIR_BATCH_SIZE
        or len(set(update.sample_ids)) != PAIR_BATCH_SIZE
        for update in schedule
    ):
        raise AssertionError("CCFR holdout update repeats a pair/source")

    factual_exposure = Counter(
        state_index
        for update_index in range(UPDATE_COUNT)
        for state_index in factual_indices_for_update(update_index)
    )
    if factual_exposure != Counter(
        {
            state_index: FACTUAL_EXPOSURES_PER_STATE
            for state_index in range(FACTUAL_POPULATION_SIZE)
        }
    ):
        raise AssertionError("CCFR factual exposure schedule differs")

    strata_counts: dict[str, dict[str, int]] = {}
    for group_id in sorted(_EXPECTED_GROUPS):
        group_specs = tuple(
            spec for spec in specs if spec.group_id == group_id
        )
        outcome = build_ccfr_holdout_outcome_batch(group_specs)
        strata = build_ccfr_holdout_strata(outcome)
        if bool((strata.D & ~strata.changed_count_support).any()):
            raise AssertionError(
                f"CCFR response escapes count support: {group_id}"
            )
        if not bool(strata.G_norm_tail.flatten(1).any(dim=1).all()):
            raise AssertionError(
                f"CCFR normalization tail is empty: {group_id}"
            )
        strata_counts[group_id] = {
            "D": int(strata.D.sum()),
            "H": int(strata.H.sum()),
            "G_near": int(strata.G_near.sum()),
            "G_norm_tail": int(strata.G_norm_tail.sum()),
        }

    observed_fingerprints = {
        "holdout": holdout_fingerprint(),
        "catalog": catalog_fingerprint(specs),
        "schedule": schedule_fingerprint(specs),
        "factual_population": factual_population_fingerprint(),
        "factual_schedule": factual_schedule_fingerprint(),
    }
    if observed_fingerprints != EXPECTED_IMPLEMENTATION_FINGERPRINTS:
        raise RuntimeError("CCFR holdout implementation fingerprints differ")
    return {
        "implementation_fingerprints": observed_fingerprints,
        "catalog_pair_count": len(specs),
        "pair_slots": sum(observed_exposure.values()),
        "factual_exposures_per_state": dict(
            sorted(factual_exposure.items())
        ),
        "strata_counts": strata_counts,
        "old_six_case_tensor_reused": False,
        "old_v10_222_role_tensor_reused": False,
        "old_v10_schedule_reused": False,
    }


def _load_prerequisites() -> dict[str, object]:
    runtime_import_boundary = _runtime_import_boundary()
    design_receipt = _load_design_receipt()
    runtime_import_boundary["local_source_closure"] = (
        _runtime_source_closure(design_receipt["source_bindings"])
    )
    development_pass = _load_development_pass()
    development_protocol = development_pass.get("protocol_binding")
    if not isinstance(development_protocol, dict):
        raise TypeError("CCFR development protocol binding must be an object")
    expected_holdout_binding = {
        name: design_receipt[name]
        for name in (
            "repo_path",
            "file_sha256",
            "receipt_fingerprint",
            "status",
            "design_seed",
        )
    }
    if development_protocol.get("pre_frozen_holdout_binding") != (
        expected_holdout_binding
    ):
        raise RuntimeError(
            "CCFR development result does not bind the current pre-frozen "
            "holdout receipt"
        )
    development_sources = development_protocol.get("source_bindings")
    if (
        not isinstance(development_sources, dict)
        or development_sources.get(design_receipt["repo_path"])
        != design_receipt["file_sha256"]
    ):
        raise RuntimeError(
            "CCFR development source closure does not bind the current "
            "holdout receipt file"
        )
    holdout_contract = _holdout_contract()
    if holdout_contract["implementation_fingerprints"] != (
        design_receipt["implementation_fingerprints"]
    ):
        raise RuntimeError(
            "CCFR holdout implementation differs from its frozen receipt"
        )
    return {
        "design_receipt": design_receipt,
        "development_pass": development_pass,
        "holdout_contract": holdout_contract,
        "runtime_import_boundary": runtime_import_boundary,
        "source_bindings": design_receipt["source_bindings"],
    }


def _runtime_import_boundary() -> dict[str, object]:
    """Reject any imported real-pipeline or dataset package before training."""

    forbidden = sorted(
        name
        for name in sys.modules
        if name in _FORBIDDEN_RUNTIME_MODULES
        or name == "datasets"
        or name.startswith("datasets.")
    )
    if forbidden:
        raise RuntimeError(
            "CCFR dataset-free runtime import boundary was crossed: "
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
            "CCFR runtime has unbound local imports: " f"{unbound}"
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


def _repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(_ROOT.resolve()))


def _attempt_payload(
    prerequisites: dict[str, object],
) -> dict[str, object]:
    attempt: dict[str, object] = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "stage_id": STAGE_ID,
        "attempt_number": 1,
        "status": "SINGLE_CANONICAL_ATTEMPT_CONSUMED_BEFORE_OPTIMIZATION",
        "prerequisites": prerequisites,
        "canonical_artifacts": {
            "attempt_repo_path": _repo_path(_CANONICAL_ATTEMPT),
            "result_repo_path": _repo_path(_CANONICAL_RESULT),
            "complete_repo_path": _repo_path(_CANONICAL_COMPLETE),
        },
        "execution": {
            "updates": UPDATE_COUNT,
            "device": "cpu",
            "torch_threads": 2,
            "automatic_retry_allowed": False,
            "dataset_access_allowed": False,
            "D_R_access_allowed": False,
            "D_V_access_allowed": False,
            "D_T_access_allowed": False,
        },
    }
    attempt["attempt_fingerprint"] = stable_fingerprint(attempt)
    return attempt


def _attempt_binding(attempt: dict[str, object]) -> dict[str, object]:
    return {
        "repo_path": _repo_path(_CANONICAL_ATTEMPT),
        "file_sha256": file_sha256(_CANONICAL_ATTEMPT),
        "attempt_fingerprint": attempt["attempt_fingerprint"],
        "attempt_number": 1,
    }


def _load_attempt(
    prerequisites: dict[str, object],
) -> dict[str, object]:
    attempt = _load_object(
        _CANONICAL_ATTEMPT,
        name="CCFR holdout canonical attempt",
    )
    _verify_fingerprint(
        attempt,
        field="attempt_fingerprint",
        name="CCFR holdout canonical attempt",
    )
    if attempt != _attempt_payload(prerequisites):
        raise RuntimeError("CCFR holdout attempt contract differs")
    return _attempt_binding(attempt)


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return (
        str(tensor.dtype).encode("utf-8")
        + repr(tuple(tensor.shape)).encode("utf-8")
        + tensor.numpy().tobytes()
    )


def _decoder_fingerprint(decoder: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in decoder.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _optimizer_contract(
    optimizer: torch.optim.Adam,
) -> dict[str, object]:
    defaults = optimizer.defaults
    return {
        "name": "Adam",
        "learning_rate": float(defaults["lr"]),
        "betas": [float(value) for value in defaults["betas"]],
        "epsilon": float(defaults["eps"]),
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


def _build_optimizer(
    decoder: nn.Module,
) -> tuple[torch.optim.Adam, dict[str, object]]:
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=FROZEN_LEARNING_RATE,
        betas=FROZEN_BETAS,
        eps=FROZEN_EPSILON,
        weight_decay=FROZEN_WEIGHT_DECAY,
        amsgrad=FROZEN_AMSGRAD,
        maximize=FROZEN_MAXIMIZE,
        foreach=FROZEN_FOREACH,
        capturable=FROZEN_CAPTURABLE,
        differentiable=FROZEN_DIFFERENTIABLE,
        fused=FROZEN_FUSED,
        decoupled_weight_decay=FROZEN_DECOUPLED_WEIGHT_DECAY,
    )
    contract = _optimizer_contract(optimizer)
    expected = {
        "name": "Adam",
        "learning_rate": FROZEN_LEARNING_RATE,
        "betas": list(FROZEN_BETAS),
        "epsilon": FROZEN_EPSILON,
        "weight_decay": FROZEN_WEIGHT_DECAY,
        "amsgrad": FROZEN_AMSGRAD,
        "maximize": FROZEN_MAXIMIZE,
        "foreach": FROZEN_FOREACH,
        "capturable": FROZEN_CAPTURABLE,
        "differentiable": FROZEN_DIFFERENTIABLE,
        "fused": FROZEN_FUSED,
        "decoupled_weight_decay": FROZEN_DECOUPLED_WEIGHT_DECAY,
    }
    if contract != expected:
        raise RuntimeError("CCFR frozen Adam contract differs")
    return optimizer, contract


def _minimum(value: Tensor) -> float:
    if value.numel() == 0:
        raise ValueError("minimum requires a non-empty tensor")
    return float(value.min().detach().cpu())


def _maximum(value: Tensor) -> float:
    if value.numel() == 0:
        raise ValueError("maximum requires a non-empty tensor")
    return float(value.max().detach().cpu())


def _mean(value: Tensor) -> float:
    if value.numel() == 0:
        raise ValueError("mean requires a non-empty tensor")
    return float(value.mean().detach().cpu())


def _build_decoder(decoder_class: type[nn.Module]) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FROZEN_DECODER_SEED)
        decoder = decoder_class(
            feature_channels=FEATURE_CHANNELS,
            feature_stride=FEATURE_STRIDE,
        )
    named = tuple(decoder.named_parameters())
    if len(named) != EXPECTED_PARAMETER_TENSORS:
        raise AssertionError("CCFR holdout parameter tensor count differs")
    if sum(value.numel() for _, value in named) != EXPECTED_PARAMETER_COUNT:
        raise AssertionError("CCFR holdout parameter count differs")
    return decoder


def _optimize(
    decoder: nn.Module,
    *,
    specs: tuple[CCFRHoldoutPairSpec, ...],
    schedule: tuple[CCFRHoldoutUpdate, ...],
    update_count: int,
) -> dict[str, object]:
    """Run a prefix of the frozen schedule; formal evaluation passes 400."""

    if (
        isinstance(update_count, bool)
        or not isinstance(update_count, int)
        or not 1 <= update_count <= UPDATE_COUNT
    ):
        raise ValueError("update_count is outside the frozen schedule")
    if len(schedule) != UPDATE_COUNT:
        raise ValueError("formal CCFR schedule must contain 400 updates")

    optimizer, optimizer_contract = _build_optimizer(decoder)
    absolute = CURELiteLoss()
    criterion = PairedEndpointCrossingLoss(LossConfig())
    by_index = {spec.population_index: spec for spec in specs}
    named_parameters = tuple(decoder.named_parameters())
    forward_batch_sizes: list[int] = []

    def observe(_module: object, args: tuple[object, ...]) -> None:
        forward_batch_sizes.append(int(args[0].shape[0]))

    gradient_failures: list[dict[str, object]] = []
    step_failures: list[dict[str, object]] = []
    minimum_gradient_norm = float("inf")
    maximum_gradient_norm = 0.0
    first_logs: dict[str, float | int] | None = None
    last_logs: dict[str, float | int] | None = None
    handle = decoder.register_forward_pre_hook(observe)
    try:
        for update in schedule[:update_count]:
            selected = tuple(
                by_index[index] for index in update.population_indices
            )
            outcome = build_ccfr_holdout_outcome_batch(selected)
            factual = build_ccfr_holdout_factual_batches(
                update_index=update.update_index
            )
            logs = outcome_complete_train_step(
                decoder,
                absolute,
                criterion,
                optimizer,
                factual,
                outcome,
            )
            if first_logs is None:
                first_logs = dict(logs)
            last_logs = dict(logs)
            expected_logs = {
                "factual_miss/states": FACTUAL_BATCH_SIZE,
                "factual_no_miss/states": FACTUAL_BATCH_SIZE,
                "outcome/pairs": PAIR_BATCH_SIZE,
                "outcome/endpoints": 2 * PAIR_BATCH_SIZE,
                "outcome/clean_pairs": sum(
                    spec.pair_kind == "clean_positive"
                    for spec in selected
                ),
                "outcome/component_null_pairs": sum(
                    spec.pair_kind == "component_null"
                    for spec in selected
                ),
                "decoder_forward_calls_per_update": 3,
                "decoder_states_per_update": 12,
                "backward_calls": 1,
                "optimizer_steps": 1,
            }
            observed_logs = {
                name: logs.get(name) for name in expected_logs
            }
            if observed_logs != expected_logs:
                step_failures.append(
                    {
                        "update_index": update.update_index,
                        "expected": expected_logs,
                        "observed": observed_logs,
                    }
                )

            for name, parameter in named_parameters:
                gradient = parameter.grad
                finite = (
                    gradient is not None
                    and bool(torch.isfinite(gradient).all())
                )
                norm = (
                    0.0
                    if gradient is None
                    else float(gradient.detach().double().norm().cpu())
                )
                minimum_gradient_norm = min(minimum_gradient_norm, norm)
                maximum_gradient_norm = max(maximum_gradient_norm, norm)
                if not finite or norm <= 0.0:
                    gradient_failures.append(
                        {
                            "update_index": update.update_index,
                            "parameter": name,
                            "finite": finite,
                            "l2_norm": norm,
                        }
                    )
    finally:
        handle.remove()
    if first_logs is None or last_logs is None:
        raise AssertionError("CCFR holdout produced no optimizer logs")

    patterns = tuple(
        tuple(forward_batch_sizes[index : index + 3])
        for index in range(0, len(forward_batch_sizes), 3)
    )
    forward_pass = (
        len(forward_batch_sizes) == update_count * 3
        and len(patterns) == update_count
        and all(pattern == (4, 4, 4) for pattern in patterns)
    )
    return {
        "optimizer_contract": optimizer_contract,
        "updates_checked": update_count,
        "gradient_contract": {
            "parameter_tensors_per_update": len(named_parameters),
            "parameters": sum(
                value.numel() for _, value in named_parameters
            ),
            "gradient_observations": (
                update_count * len(named_parameters)
            ),
            "failure_count": len(gradient_failures),
            "failures": gradient_failures,
            "minimum_l2_norm": minimum_gradient_norm,
            "maximum_l2_norm": maximum_gradient_norm,
            "all_finite_nonzero_every_update": not gradient_failures,
        },
        "step_contract": {
            "failure_count": len(step_failures),
            "failures": step_failures,
            "all_updates_exact_4_4_2": not step_failures,
        },
        "forward_contract": {
            "call_count": len(forward_batch_sizes),
            "expected_call_count": update_count * 3,
            "per_update_expected_batch_sizes": [4, 4, 4],
            "first_update_batch_sizes": list(patterns[0]),
            "last_update_batch_sizes": list(patterns[-1]),
            "all_updates_exact_three_4_state_calls": forward_pass,
        },
        "first_update_logs": first_logs,
        "last_update_logs": last_logs,
        "all_pass": (
            not gradient_failures
            and not step_failures
            and forward_pass
            and update_count == UPDATE_COUNT
        ),
    }


def _factual_evaluation(decoder: nn.Module) -> dict[str, object]:
    factual = build_ccfr_holdout_factual_population()
    absolute = CURELiteLoss()
    with torch.no_grad():
        miss = factual["factual_miss"]
        no_miss = factual["factual_no_miss"]
        miss_logits = decoder(miss.feature, miss.occupancy)
        no_miss_logits = decoder(no_miss.feature, no_miss.occupancy)
        miss_score = torch.sigmoid(miss_logits)
        no_miss_score = torch.sigmoid(no_miss_logits)
        miss_loss = absolute(
            miss_logits,
            miss.target,
            miss.valid_mask,
        )["total"]
        no_miss_loss = absolute(
            no_miss_logits,
            no_miss.target,
            no_miss.valid_mask,
        )["total"]
    target = miss.target > 0.5
    background = miss.valid_mask & ~target
    metrics = {
        "factual_miss_target_min": _minimum(miss_score[target]),
        "factual_miss_background_max": _maximum(
            miss_score[background]
        ),
        "factual_no_miss_max": _maximum(
            no_miss_score[no_miss.valid_mask]
        ),
        "factual_miss_loss": float(miss_loss.detach().cpu()),
        "factual_no_miss_loss": float(no_miss_loss.detach().cpu()),
    }
    checks = {
        "factual_miss_target": metrics["factual_miss_target_min"]
        > THRESHOLDS["factual_miss_target_min_exclusive"],
        "factual_miss_background": metrics["factual_miss_background_max"]
        < THRESHOLDS["factual_miss_background_max_exclusive"],
        "factual_no_miss": metrics["factual_no_miss_max"]
        < THRESHOLDS["factual_no_miss_max_exclusive"],
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "all_pass": all(checks.values()),
        "_loss_total": miss_loss + no_miss_loss,
    }


def _group_evaluation(
    decoder: nn.Module,
    specs: tuple[CCFRHoldoutPairSpec, ...],
) -> dict[str, object]:
    group_id = specs[0].group_id
    if any(spec.group_id != group_id for spec in specs):
        raise ValueError("group evaluation requires one exact group")
    pair_kind = specs[0].pair_kind
    if any(spec.pair_kind != pair_kind for spec in specs):
        raise ValueError("group evaluation mixes pair roles")

    outcome = build_ccfr_holdout_outcome_batch(specs)
    strata = build_ccfr_holdout_strata(outcome)
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

    anchor_background = (
        outcome.pair_batch.image_valid_mask
        & ~outcome.pair_batch.occupancy_plus
        & ~outcome.gt_union
    )
    required = {
        "plus_completion": outcome.completion_plus,
        "plus_background": anchor_background,
        "H": strata.H,
        "G_near": strata.G_near,
        "G_norm_tail": strata.G_norm_tail,
    }
    empty = sorted(name for name, mask in required.items() if not bool(mask.any()))
    if empty:
        raise AssertionError(
            f"CCFR group {group_id} has empty strata: {empty}"
        )
    metrics: dict[str, float | int] = {
        "pair_count": len(specs),
        "slot_count": sum(spec.exposure_count for spec in specs),
        "plus_completion_min": _minimum(
            score_plus[outcome.completion_plus]
        ),
        "plus_background_max": _maximum(
            score_plus[anchor_background]
        ),
        "H_max_abs": _maximum(delta[strata.H].abs()),
        "H_positive_max": _maximum(torch.relu(delta[strata.H])),
        "H_negative_max": _maximum(torch.relu(-delta[strata.H])),
        "G_near_max_abs": _maximum(delta[strata.G_near].abs()),
        "G_near_positive_max": _maximum(
            torch.relu(delta[strata.G_near])
        ),
        "G_near_negative_max": _maximum(
            torch.relu(-delta[strata.G_near])
        ),
        "G_norm_tail_max_abs": _maximum(
            delta[strata.G_norm_tail].abs()
        ),
        "G_norm_tail_positive_max": _maximum(
            torch.relu(delta[strata.G_norm_tail])
        ),
        "G_norm_tail_negative_max": _maximum(
            torch.relu(-delta[strata.G_norm_tail])
        ),
    }
    checks = {
        "plus_completion": metrics["plus_completion_min"]
        > THRESHOLDS["plus_completion_min_exclusive"],
        "plus_background": metrics["plus_background_max"]
        < THRESHOLDS["plus_background_max_exclusive"],
        "H": metrics["H_max_abs"]
        <= THRESHOLDS["zero_H_max_abs_max_inclusive"],
        "G_near": metrics["G_near_max_abs"]
        <= THRESHOLDS["zero_G_near_max_abs_max_inclusive"],
        "G_norm_tail": metrics["G_norm_tail_max_abs"]
        <= THRESHOLDS["zero_G_norm_tail_max_abs_max_inclusive"],
    }
    if pair_kind == "clean_positive":
        if not bool(strata.D.any()):
            raise AssertionError(f"CCFR clean group {group_id} has empty D")
        metrics.update(
            {
                "D_delta_mean": _mean(delta[strata.D]),
                "D_plus_max": _maximum(score_plus[strata.D]),
                "D_minus_min": _minimum(score_minus[strata.D]),
                "D_wrong_direction_pixel_count": int(
                    (delta[strata.D] < 0.0).sum()
                ),
            }
        )
        checks.update(
            {
                "D_delta": metrics["D_delta_mean"]
                >= THRESHOLDS["clean_D_delta_mean_min_inclusive"],
                "D_plus": metrics["D_plus_max"]
                < THRESHOLDS["clean_D_plus_max_exclusive"],
                "D_minus": metrics["D_minus_min"]
                > THRESHOLDS["clean_D_minus_min_exclusive"],
                "D_direction": metrics["D_wrong_direction_pixel_count"]
                <= THRESHOLDS[
                    "D_wrong_direction_pixel_count_max_inclusive"
                ],
            }
        )
    elif bool(strata.D.any()):
        raise AssertionError(f"CCFR component-null group {group_id} has D")

    return {
        "group_id": group_id,
        "pair_kind": pair_kind,
        "metrics": metrics,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def _final_gate_summary(
    *,
    population_objective: float,
    factual: dict[str, object],
    groups: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the frozen population/factual/eight-group conjunction."""

    observed_counts = {
        str(group["group_id"]): int(group["metrics"]["pair_count"])
        for group in groups
    }
    exact_group_contract = (
        observed_counts == _EXPECTED_GROUPS
        and len(groups) == len(_EXPECTED_GROUPS)
    )
    checks = {
        "population_objective": population_objective
        < THRESHOLDS["population_total_loss_max_exclusive"],
        "factual": factual.get("all_pass") is True,
        "all_eight_groups": (
            exact_group_contract
            and all(group.get("all_pass") is True for group in groups)
        ),
    }
    return {
        "passed_group_count": sum(
            group.get("all_pass") is True for group in groups
        ),
        "failed_group_count": sum(
            group.get("all_pass") is not True for group in groups
        ),
        "exact_group_contract": {
            "expected": _EXPECTED_GROUPS,
            "observed": observed_counts,
            "all_pass": exact_group_contract,
        },
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def _final_evaluation(
    decoder: nn.Module,
    specs: tuple[CCFRHoldoutPairSpec, ...],
) -> dict[str, object]:
    decoder.eval()
    criterion = PairedEndpointCrossingLoss(LossConfig())
    outcome = build_ccfr_holdout_outcome_batch(specs)
    with torch.no_grad():
        logits_plus, logits_minus = _paired_endpoint_logits(
            decoder,
            feature=outcome.pair_batch.feature,
            occupancy_plus=outcome.pair_batch.occupancy_plus,
            occupancy_minus=outcome.pair_batch.occupancy_minus,
        )
        pair_loss = criterion(
            logits_plus,
            logits_minus,
            outcome.completion_plus,
            outcome.pair_batch.occupancy_plus,
            outcome.gt_union,
            outcome.pair_batch.label_increment,
            outcome.pair_batch.image_valid_mask,
            outcome.intervention_footprint,
        )["total"]
    factual = _factual_evaluation(decoder)
    total_loss = factual["_loss_total"] + pair_loss
    del factual["_loss_total"]
    groups = [
        _group_evaluation(
            decoder,
            tuple(spec for spec in specs if spec.group_id == group_id),
        )
        for group_id in sorted(_EXPECTED_GROUPS)
    ]
    population_objective = float(total_loss.detach().cpu())
    gate = _final_gate_summary(
        population_objective=population_objective,
        factual=factual,
        groups=groups,
    )
    return {
        "population_objective": population_objective,
        "population_pair_loss": float(pair_loss.detach().cpu()),
        "factual": factual,
        "groups": groups,
        **gate,
    }


def _train_and_evaluate(
    *,
    objective_id: str,
    decoder_class: type[nn.Module],
    specs: tuple[CCFRHoldoutPairSpec, ...],
    schedule: tuple[CCFRHoldoutUpdate, ...],
) -> dict[str, object]:
    decoder = _build_decoder(decoder_class)
    initial = _decoder_fingerprint(decoder)
    training = _optimize(
        decoder,
        specs=specs,
        schedule=schedule,
        update_count=UPDATE_COUNT,
    )
    final = _final_evaluation(decoder, specs)
    checks = {
        "training": training["all_pass"] is True,
        "final_population_and_groups": final["all_pass"] is True,
    }
    return {
        "objective_id": objective_id,
        "execution_status": "COMPLETED",
        "initial_decoder_fingerprint": initial,
        "final_decoder_fingerprint": _decoder_fingerprint(decoder),
        "training": training,
        "final": final,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def _assemble_result(
    *,
    prerequisites: dict[str, object],
    attempt_binding: dict[str, object],
    ccfr: dict[str, object],
    comparator: dict[str, object],
    runtime: dict[str, object],
) -> dict[str, object]:
    comparator_completed = (
        comparator.get("execution_status") == "COMPLETED"
    )
    same_initialization: bool | None = None
    same_optimizer: bool | None = None
    if comparator_completed:
        same_initialization = (
            ccfr["initial_decoder_fingerprint"]
            == comparator["initial_decoder_fingerprint"]
        )
        same_optimizer = (
            ccfr["training"]["optimizer_contract"]
            == comparator["training"]["optimizer_contract"]
        )
    # The receipt defines v8 as report-only.  Neither a worse result, a
    # mismatched implementation report, nor a comparator execution error may
    # relax or veto the complete CCFR absolute decision.
    ccfr_pass = ccfr["all_pass"] is True
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "stage_id": STAGE_ID,
        "evidence_scope": (
            "pre_frozen_dataset_free_holdout_with_replayable_one_update_"
            "wiring_smoke_not_detection_performance"
        ),
        "prerequisites": prerequisites,
        "attempt_binding": attempt_binding,
        "decision": (
            "CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_PASS"
            if ccfr_pass
            else "CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_FAIL"
        ),
        "all_pass": ccfr_pass,
        "contract": {
            "design_seed": DESIGN_SEED,
            "decoder_seed": FROZEN_DECODER_SEED,
            "updates": UPDATE_COUNT,
            "device": "cpu",
            "torch_threads": 2,
            "pair_batch_size": PAIR_BATCH_SIZE,
            "factual_batch_size_per_branch": FACTUAL_BATCH_SIZE,
            "factual_exposures_per_state": FACTUAL_EXPOSURES_PER_STATE,
            "feature_shape": [
                FEATURE_CHANNELS,
                FEATURE_HEIGHT,
                FEATURE_WIDTH,
            ],
            "output_shape": [1, OUTPUT_HEIGHT, OUTPUT_WIDTH],
            "parameter_tensors": EXPECTED_PARAMETER_TENSORS,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "thresholds": THRESHOLDS,
            "pre_mask_endpoint_evaluation": True,
            "pair_kind_is_model_input": False,
            "pair_kind_is_loss_dispatch": False,
        },
        "same_initialization_verified": same_initialization,
        "same_optimizer_verified": same_optimizer,
        "ccfr": ccfr,
        "matched_v8_comparator": comparator,
        "matched_v8_comparator_execution_status": comparator.get(
            "execution_status"
        ),
        "matched_v8_comparator_affects_decision": False,
        "runtime": runtime,
        "execution_boundary": {
            "dataset_free_confirmation_performed": True,
            "dataset_accessed": False,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "detection_performance_evaluated": False,
            "real_D_R_authorized": ccfr_pass,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_detector_authorized": False,
        },
        "interpretation": (
            "dataset_free_exposure_holdout_model_code_confirmation_only;"
            "the_preformal_one_update_wiring_smoke_did_not_read_or_select_"
            "on_final_performance"
        ),
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def evaluate() -> dict[str, object]:
    """Run the consumed canonical attempt on the frozen CPU protocol."""

    prerequisites = _load_prerequisites()
    attempt_binding = _load_attempt(prerequisites)
    specs = build_ccfr_holdout_pair_specs()
    schedule = build_ccfr_holdout_schedule(specs)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    confirmation_threads = 2
    try:
        torch.set_num_threads(confirmation_threads)
        torch.use_deterministic_algorithms(True)
        ccfr = _train_and_evaluate(
            objective_id=METHOD_ID,
            decoder_class=CURELiteCoverageFeatureReleaseDecoder,
            specs=specs,
            schedule=schedule,
        )
        try:
            comparator = _train_and_evaluate(
                objective_id=COMPARATOR_ID,
                decoder_class=CURELiteConservativeFactorizedDecoder,
                specs=specs,
                schedule=schedule,
            )
        except Exception as error:  # report-only boundary
            comparator = {
                "objective_id": COMPARATOR_ID,
                "execution_status": "ERROR",
                "comparator_execution_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "all_pass": False,
            }
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)
    runtime = {
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "threads_before": previous_threads,
        "threads_during_confirmation": confirmation_threads,
        "deterministic_algorithms_before": previous_deterministic,
        "deterministic_algorithms_during_confirmation": True,
        "threads_restored": torch.get_num_threads() == previous_threads,
        "deterministic_algorithms_restored": (
            torch.are_deterministic_algorithms_enabled()
            == previous_deterministic
        ),
    }
    return _assemble_result(
        prerequisites=prerequisites,
        attempt_binding=attempt_binding,
        ccfr=ccfr,
        comparator=comparator,
        runtime=runtime,
    )


def _write_json_create_only(
    path: Path,
    value: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _write_complete_create_only(
    path: Path,
    *,
    result_path: Path,
) -> str:
    digest = file_sha256(result_path)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {result_path.name}\n")
    return digest


def _assert_fresh_canonical_artifacts(output: Path) -> None:
    if output.resolve() != _CANONICAL_RESULT.resolve():
        raise ValueError(
            "CCFR authority output must be the frozen canonical path: "
            f"{_CANONICAL_RESULT}"
        )
    existing = [
        path
        for path in (
            _CANONICAL_ATTEMPT,
            _CANONICAL_RESULT,
            _CANONICAL_COMPLETE,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "CCFR single canonical attempt is unavailable because an "
            f"authority artifact exists: {existing}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _assert_fresh_canonical_artifacts(args.output)

    prerequisites = _load_prerequisites()
    attempt = _attempt_payload(prerequisites)
    _write_json_create_only(_CANONICAL_ATTEMPT, attempt)
    result = evaluate()
    _write_json_create_only(_CANONICAL_RESULT, result)
    complete_sha256 = _write_complete_create_only(
        _CANONICAL_COMPLETE,
        result_path=_CANONICAL_RESULT,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "all_pass": result["all_pass"],
                "result_fingerprint": result["result_fingerprint"],
                "complete_sha256": complete_sha256,
                "output": str(_CANONICAL_RESULT),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["all_pass"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
