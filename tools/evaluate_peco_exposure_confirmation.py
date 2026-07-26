#!/usr/bin/env python3
"""Run the corrected frozen dataset-free PECO v10 confirmation.

The command trains PECO and the frozen predecessor objective from identical
initialization on the same 222-pair population and 400-update schedule.  The
predecessor is a matched comparator only.  PECO is decided exclusively by
predeclared absolute gates. Revision r3 only corrects canonical result
serialization after the r2 authority attempt produced no result artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import Tensor, nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import (  # noqa: E402
    file_sha256,
    stable_fingerprint,
)
from cure_lite.config import LossConfig  # noqa: E402
from cure_lite.conservative_factorized_decoder import (  # noqa: E402
    CURELiteConservativeFactorizedDecoder,
)
from cure_lite.experiment.peco_exposure_confirmation import (  # noqa: E402
    CLEAN_PAIR_COUNT,
    CLEAN_SLOT_COUNT,
    COMPONENT_PAIR_COUNT,
    COMPONENT_SLOT_COUNT,
    CONFIRMATION_SEED,
    FACTUAL_BATCH_SIZE,
    FACTUAL_EXPOSURES_PER_STATE,
    FACTUAL_POPULATION_SIZE,
    FACTUAL_SLOTS_PER_BRANCH,
    ConfirmationPairSpec,
    build_confirmation_factual_batches,
    build_confirmation_factual_population,
    build_confirmation_outcome_batch,
    build_confirmation_pair_specs,
    build_confirmation_schedule,
    build_identical_input_conflict_control,
    catalog_fingerprint,
    factual_schedule_fingerprint,
    schedule_fingerprint,
)
from cure_lite.losses import CURELiteLoss  # noqa: E402
from cure_lite.paired_endpoint_crossing_losses import (  # noqa: E402
    PairedEndpointCrossingLoss,
)
from cure_lite.paired_outcome_losses import (  # noqa: E402
    OutcomeCompleteTransitionLoss,
)
from cure_lite.train.paired_outcome_step import (  # noqa: E402
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import (  # noqa: E402
    _paired_endpoint_logits,
)


SCHEMA_VERSION = "cure-lite-peco-v10-exposure-confirmation-r3-result-v1"
METHOD_ID = "peco_v10"
PREDECESSOR_METHOD_ID = "outcome_complete_transition_v3"
EXPECTED_PARAMETER_TENSORS = 6
EXPECTED_PARAMETER_COUNT = 2593
FROZEN_UPDATES = 400
FROZEN_LEARNING_RATE = 0.001
FROZEN_ADAM_BETAS = (0.9, 0.999)
FROZEN_ADAM_EPSILON = 1.0e-8
FROZEN_ADAM_WEIGHT_DECAY = 0.0
FROZEN_ADAM_AMSGRAD = False
FROZEN_ADAM_MAXIMIZE = False
FROZEN_ADAM_FOREACH = None
FROZEN_ADAM_CAPTURABLE = False
FROZEN_ADAM_DIFFERENTIABLE = False
FROZEN_ADAM_FUSED = None
FROZEN_ADAM_DECOUPLED_WEIGHT_DECAY = False
EXPECTED_GROUP_CONTRACT = {
    "clean_contains_1px": {"pair_count": 35, "slot_count": 126},
    "clean_contains_2px": {"pair_count": 34, "slot_count": 122},
    "clean_contains_3px": {"pair_count": 34, "slot_count": 122},
    "clean_outside_1px": {"pair_count": 35, "slot_count": 125},
    "clean_outside_2px": {"pair_count": 34, "slot_count": 122},
    "clean_outside_3px": {"pair_count": 34, "slot_count": 122},
    "component_null_block": {"pair_count": 8, "slot_count": 31},
    "component_null_sparse": {"pair_count": 8, "slot_count": 30},
}

_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_endpoint_crossing_objective_v10"
)
_CONFIG = _PROTOCOL / "exposure_confirmation_config_r2.json"
_DESIGN_RECEIPT = (
    _PROTOCOL / "exposure_confirmation_design_receipt_r2.json"
)
_R1_INVALIDATION = (
    _PROTOCOL
    / "exposure_confirmation_design_r1_invalidation_receipt.json"
)
_R2_IMPLEMENTATION_CLOSURE = (
    _PROTOCOL / "exposure_confirmation_implementation_closure_r2.json"
)
_R2_PRE_RUN_RECEIPT = (
    _PROTOCOL / "exposure_confirmation_r2_pre_run_verification_receipt.json"
)
_R2_EXECUTION_FAILURE = (
    _PROTOCOL / "exposure_confirmation_r2_execution_failure_receipt.json"
)
_R2_CANONICAL_OUTPUT = (
    _PROTOCOL / "exposure_confirmation_result_r2.json"
)
_IMPLEMENTATION_CLOSURE = (
    _PROTOCOL / "exposure_confirmation_implementation_closure_r3.json"
)
_PRE_RUN_RECEIPT = (
    _PROTOCOL / "exposure_confirmation_r3_pre_run_verification_receipt.json"
)
_DEVELOPMENT_R2_RESULT = (
    _PROTOCOL / "development_regression_result_r2.json"
)
_DEVELOPMENT_R2_RESULT_RECEIPT = (
    _PROTOCOL / "development_regression_r2_result_receipt.json"
)
_CANONICAL_OUTPUT = (
    _PROTOCOL / "exposure_confirmation_result_r3.json"
)


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
    if not isinstance(observed, str):
        raise TypeError(f"{name}.{field} must be a string")
    if stable_fingerprint(unsigned) != observed:
        raise RuntimeError(f"{name} fingerprint differs")
    return observed


def _load_protocol() -> tuple[dict[str, object], dict[str, object]]:
    config = _load_object(_CONFIG, name="exposure confirmation config")
    receipt = _load_object(
        _DESIGN_RECEIPT,
        name="exposure confirmation design receipt",
    )
    invalidation = _load_object(
        _R1_INVALIDATION,
        name="exposure confirmation r1 design invalidation",
    )
    r2_closure = _load_object(
        _R2_IMPLEMENTATION_CLOSURE,
        name="exposure confirmation r2 implementation closure",
    )
    r2_pre_run = _load_object(
        _R2_PRE_RUN_RECEIPT,
        name="exposure confirmation r2 pre-run receipt",
    )
    r2_failure = _load_object(
        _R2_EXECUTION_FAILURE,
        name="exposure confirmation r2 execution failure",
    )
    config_fingerprint = _verify_fingerprint(
        config,
        field="config_fingerprint",
        name="exposure confirmation config",
    )
    receipt_fingerprint = _verify_fingerprint(
        receipt,
        field="receipt_fingerprint",
        name="exposure confirmation design receipt",
    )
    invalidation_fingerprint = _verify_fingerprint(
        invalidation,
        field="receipt_fingerprint",
        name="exposure confirmation r1 design invalidation",
    )
    r2_closure_fingerprint = _verify_fingerprint(
        r2_closure,
        field="receipt_fingerprint",
        name="exposure confirmation r2 implementation closure",
    )
    r2_pre_run_fingerprint = _verify_fingerprint(
        r2_pre_run,
        field="receipt_fingerprint",
        name="exposure confirmation r2 pre-run receipt",
    )
    r2_failure_fingerprint = _verify_fingerprint(
        r2_failure,
        field="receipt_fingerprint",
        name="exposure confirmation r2 execution failure",
    )
    if config.get("schema_version") != (
        "cure-lite-peco-v10-exposure-confirmation-r2-config-v1"
    ):
        raise RuntimeError("exposure confirmation config schema differs")
    if config.get("method_id") != METHOD_ID:
        raise RuntimeError("exposure confirmation method differs")
    if config.get("status") != "R2_FROZEN_BEFORE_AUTHORITY_RUN":
        raise RuntimeError("exposure confirmation config status differs")
    expected_invalidation_binding = {
        "repo_path": str(_R1_INVALIDATION.relative_to(_ROOT)),
        "file_sha256": file_sha256(_R1_INVALIDATION),
        "receipt_fingerprint": invalidation_fingerprint,
        "decision": (
            "PECO_V10_EXPOSURE_CONFIRMATION_R1_DESIGN_INVALIDATED"
        ),
    }
    if config.get("r1_invalidation_binding") != (
        expected_invalidation_binding
    ):
        raise RuntimeError(
            "exposure confirmation r1 invalidation binding differs"
        )

    for binding_name in (
        "proposal_binding",
        "decoder_binding",
        "peco_loss_binding",
        "predecessor_loss_binding",
        "training_step_binding",
        "population_source_binding",
    ):
        binding = config.get(binding_name)
        if not isinstance(binding, dict):
            raise TypeError(f"{binding_name} must be an object")
        path = _ROOT / str(binding.get("repo_path"))
        if file_sha256(path) != binding.get("file_sha256"):
            raise RuntimeError(f"{binding_name} source hash differs")
    evaluator_binding = config.get("evaluator_binding")
    if not isinstance(evaluator_binding, dict):
        raise TypeError("evaluator_binding must be an object")
    historical_sources = r2_closure.get("source_bindings")
    if not isinstance(historical_sources, list):
        raise TypeError("r2 closure sources must be a list")
    historical_evaluator = next(
        (
            binding
            for binding in historical_sources
            if isinstance(binding, dict)
            and binding.get("repo_path")
            == "tools/evaluate_peco_exposure_confirmation.py"
        ),
        None,
    )
    if historical_evaluator != {
        "repo_path": evaluator_binding.get("repo_path"),
        "file_sha256": evaluator_binding.get("file_sha256"),
    }:
        raise RuntimeError(
            "frozen r2 evaluator binding differs from r2 closure"
        )
    dependencies = config.get("transitive_dependency_bindings")
    if not isinstance(dependencies, list) or not dependencies:
        raise TypeError(
            "transitive_dependency_bindings must be a non-empty list"
        )
    dependency_paths: list[str] = []
    for binding in dependencies:
        if not isinstance(binding, dict):
            raise TypeError("every transitive dependency must be an object")
        repo_path = binding.get("repo_path")
        if not isinstance(repo_path, str):
            raise TypeError("transitive dependency repo_path must be a string")
        dependency_paths.append(repo_path)
        if file_sha256(_ROOT / repo_path) != binding.get("file_sha256"):
            raise RuntimeError(
                f"transitive dependency hash differs: {repo_path}"
            )
    if dependency_paths != sorted(dependency_paths):
        raise RuntimeError(
            "transitive dependency bindings must be canonically sorted"
        )
    if len(set(dependency_paths)) != len(dependency_paths):
        raise RuntimeError("transitive dependency bindings contain duplicates")

    population = config.get("population")
    if not isinstance(population, dict):
        raise TypeError("population contract must be an object")
    specs = build_confirmation_pair_specs()
    if population.get("catalog_fingerprint") != catalog_fingerprint(specs):
        raise RuntimeError("confirmation catalog fingerprint differs")
    if population != {
        "pair_count": 222,
        "clean_pair_count": CLEAN_PAIR_COUNT,
        "component_null_pair_count": COMPONENT_PAIR_COUNT,
        "clean_four_exposure_pair_count": 121,
        "clean_three_exposure_pair_count": 85,
        "component_four_exposure_pair_count": 13,
        "component_three_exposure_pair_count": 3,
        "catalog_fingerprint": catalog_fingerprint(specs),
    }:
        raise RuntimeError("confirmation population contract differs")
    schedule = config.get("schedule")
    if not isinstance(schedule, dict):
        raise TypeError("schedule contract must be an object")
    if schedule != {
        "seed": CONFIRMATION_SEED,
        "updates": FROZEN_UPDATES,
        "pair_batch_size": 2,
        "pair_slots": 800,
        "clean_slots": CLEAN_SLOT_COUNT,
        "component_null_slots": COMPONENT_SLOT_COUNT,
        "updates_with_zero_component": 340,
        "updates_with_one_component": 59,
        "updates_with_two_components": 1,
        "factual_miss_population_size": FACTUAL_POPULATION_SIZE,
        "factual_no_miss_population_size": FACTUAL_POPULATION_SIZE,
        "factual_miss_batch_size": FACTUAL_BATCH_SIZE,
        "factual_no_miss_batch_size": FACTUAL_BATCH_SIZE,
        "factual_miss_slots": FACTUAL_SLOTS_PER_BRANCH,
        "factual_no_miss_slots": FACTUAL_SLOTS_PER_BRANCH,
        "factual_exposures_per_state": FACTUAL_EXPOSURES_PER_STATE,
        "factual_schedule": "(4*update_index+i)%16_for_i_in_0_to_3",
        "factual_schedule_fingerprint": factual_schedule_fingerprint(),
        "schedule_fingerprint": schedule_fingerprint(specs),
    }:
        raise RuntimeError("confirmation schedule contract differs")
    if config.get("optimization") != {
        "device": "cpu",
        "seed": CONFIRMATION_SEED,
        "optimizer": "adam",
        "learning_rate": FROZEN_LEARNING_RATE,
        "weight_decay": FROZEN_ADAM_WEIGHT_DECAY,
        "adam_betas": list(FROZEN_ADAM_BETAS),
        "adam_epsilon": FROZEN_ADAM_EPSILON,
        "adam_amsgrad": FROZEN_ADAM_AMSGRAD,
        "adam_maximize": FROZEN_ADAM_MAXIMIZE,
        "adam_foreach": FROZEN_ADAM_FOREACH,
        "adam_capturable": FROZEN_ADAM_CAPTURABLE,
        "adam_differentiable": FROZEN_ADAM_DIFFERENTIABLE,
        "adam_fused": FROZEN_ADAM_FUSED,
        "adam_decoupled_weight_decay": (
            FROZEN_ADAM_DECOUPLED_WEIGHT_DECAY
        ),
        "all_fields_explicitly_passed": True,
        "exact_defaults_checked_before_training": True,
        "updates": FROZEN_UPDATES,
        "automatic_retry_allowed": False,
        "hyperparameter_search_allowed": False,
    }:
        raise RuntimeError("confirmation optimization contract differs")
    if config.get("exact_group_contract") != EXPECTED_GROUP_CONTRACT:
        raise RuntimeError("confirmation exact group contract differs")
    if config.get("matched_comparator") != {
        "method_id": PREDECESSOR_METHOD_ID,
        "same_population": True,
        "same_schedule": True,
        "same_initialization": True,
        "same_optimizer": True,
        "decision_use": "reported_only_not_used_to_relax_absolute_peco_gate",
    }:
        raise RuntimeError("matched comparator contract differs")
    if config.get("crossing_semantics") != {
        "boundary": "training_logit_zero",
        "equivalent_probability": 0.5,
        "is_final_frozen_calibration_threshold": False,
        "detection_threshold_claim_allowed": False,
    }:
        raise RuntimeError("endpoint crossing semantics differ")
    if config.get("decision_rule") != {
        "pass_decision": (
            "PECO_V10_EXPOSURE_MATCHED_CONFIRMATION_PASS"
        ),
        "fail_decision": (
            "PECO_V10_EXPOSURE_MATCHED_CONFIRMATION_FAIL"
        ),
        "all_clean_groups_must_pass": True,
        "all_component_groups_must_pass": True,
        "exact_eight_group_set_and_counts_must_pass": True,
        "factual_gates_must_pass": True,
        "gradient_and_2B_contracts_must_pass": True,
        "every_update_all_six_gradients_finite_nonzero": True,
        "frozen_optimizer_contract_must_match": True,
        "identical_input_conflict_control_must_pass": True,
        "mean_cannot_override_group_failure": True,
        "matched_comparator_cannot_change_peco_decision": True,
        "pass_authorizes_only_deterministic_dry_run": True,
        "runtime_contract_must_be_reported": True,
    }:
        raise RuntimeError("confirmation decision rule differs")
    if config.get("authority_contract") != {
        "development_r2_pass_required": True,
        "implementation_closure_required": True,
        "pre_run_receipt_required": True,
        "authority_run_count": 1,
        "canonical_output_repo_path": str(
            _R2_CANONICAL_OUTPUT.relative_to(_ROOT)
        ),
        "automatic_retry_allowed": False,
    }:
        raise RuntimeError("confirmation authority contract differs")
    if config.get("execution_boundary") != {
        "dataset_free_confirmation_run_authorized": True,
        "dry_run_authorized": False,
        "D_R_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
        "detection_performance_allowed": False,
        "real_bounded_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "cross_detector_authorized": False,
    }:
        raise RuntimeError("confirmation execution boundary differs")

    receipt_binding = receipt.get("config_binding")
    if receipt_binding != {
        "repo_path": str(_CONFIG.relative_to(_ROOT)),
        "file_sha256": file_sha256(_CONFIG),
        "config_fingerprint": config_fingerprint,
    }:
        raise RuntimeError("design receipt config binding differs")
    receipt_sources = receipt.get("source_bindings")
    expected_sources = {
        name: config[name]
        for name in (
            "decoder_binding",
            "peco_loss_binding",
            "predecessor_loss_binding",
            "training_step_binding",
            "population_source_binding",
            "evaluator_binding",
        )
    }
    if receipt_sources != expected_sources:
        raise RuntimeError("design receipt source bindings differ")
    if receipt.get("transitive_dependency_bindings") != dependencies:
        raise RuntimeError(
            "design receipt transitive dependency bindings differ"
        )
    if receipt.get("status") != "R2_FROZEN_BEFORE_AUTHORITY_RUN":
        raise RuntimeError("design receipt status differs")
    if receipt.get("decision") != (
        "PECO_V10_EXPOSURE_CONFIRMATION_R2_DESIGN_FROZEN"
    ):
        raise RuntimeError("design receipt decision differs")
    if receipt.get("r1_invalidation_binding") != (
        expected_invalidation_binding
    ):
        raise RuntimeError(
            "design receipt r1 invalidation binding differs"
        )
    expected_r2_closure_binding = {
        "repo_path": str(_R2_IMPLEMENTATION_CLOSURE.relative_to(_ROOT)),
        "file_sha256": file_sha256(_R2_IMPLEMENTATION_CLOSURE),
        "receipt_fingerprint": r2_closure_fingerprint,
    }
    expected_r2_pre_run_binding = {
        "repo_path": str(_R2_PRE_RUN_RECEIPT.relative_to(_ROOT)),
        "file_sha256": file_sha256(_R2_PRE_RUN_RECEIPT),
        "receipt_fingerprint": r2_pre_run_fingerprint,
    }
    if r2_failure.get("schema_version") != (
        "cure-lite-peco-v10-exposure-confirmation-r2-"
        "execution-failure-v1"
    ):
        raise RuntimeError("r2 execution failure schema differs")
    if r2_failure.get("status") != (
        "R2_AUTHORITY_EXECUTION_REACHED_RESULT_ASSEMBLY_"
        "WITHOUT_RESULT_ARTIFACT"
    ):
        raise RuntimeError("r2 execution failure status differs")
    if r2_failure.get("decision") != (
        "PECO_V10_EXPOSURE_CONFIRMATION_R2_RESULT_NOT_PRODUCED"
    ):
        raise RuntimeError("r2 execution failure decision differs")
    if r2_failure.get("implementation_closure_binding") != (
        expected_r2_closure_binding
    ):
        raise RuntimeError("r2 execution failure closure binding differs")
    if r2_failure.get("pre_run_verification_binding") != (
        expected_r2_pre_run_binding
    ):
        raise RuntimeError("r2 execution failure pre-run binding differs")
    authority_invocation = r2_failure.get("authority_invocation")
    if not isinstance(authority_invocation, dict):
        raise TypeError("r2 authority invocation must be an object")
    if authority_invocation.get("canonical_output_repo_path") != str(
        _R2_CANONICAL_OUTPUT.relative_to(_ROOT)
    ):
        raise RuntimeError("r2 failed output path differs")
    if authority_invocation.get("canonical_output_exists") is not False:
        raise RuntimeError("r2 result artifact unexpectedly exists")
    if r2_failure.get("allowed_r3_correction") != {
        "canonicalize_component_histogram_keys_as_strings": True,
        "add_a_unit_test_rejecting_non_string_result_mapping_keys": True,
        "version_result_schema_and_canonical_output_as_r3": True,
        "create_a_new_r3_implementation_closure_and_pre_run_receipt": True,
        "model_change": False,
        "decoder_change": False,
        "loss_change": False,
        "population_change": False,
        "schedule_change": False,
        "optimizer_change": False,
        "seed_change": False,
        "learning_rate_change": False,
        "updates_change": False,
        "threshold_change": False,
        "automatic_retry": False,
    }:
        raise RuntimeError("r2-authorized r3 correction scope differs")
    if _R2_CANONICAL_OUTPUT.exists():
        raise RuntimeError("r2 result artifact must remain absent")
    return config, {
        "config_repo_path": str(_CONFIG.relative_to(_ROOT)),
        "config_file_sha256": file_sha256(_CONFIG),
        "config_fingerprint": config_fingerprint,
        "design_receipt_repo_path": str(
            _DESIGN_RECEIPT.relative_to(_ROOT)
        ),
        "design_receipt_file_sha256": file_sha256(_DESIGN_RECEIPT),
        "design_receipt_fingerprint": receipt_fingerprint,
        "r1_invalidation_repo_path": str(
            _R1_INVALIDATION.relative_to(_ROOT)
        ),
        "r1_invalidation_file_sha256": file_sha256(_R1_INVALIDATION),
        "r1_invalidation_fingerprint": invalidation_fingerprint,
        "r2_implementation_closure_binding": (
            expected_r2_closure_binding
        ),
        "r2_pre_run_verification_binding": (
            expected_r2_pre_run_binding
        ),
        "r2_execution_failure_repo_path": str(
            _R2_EXECUTION_FAILURE.relative_to(_ROOT)
        ),
        "r2_execution_failure_file_sha256": file_sha256(
            _R2_EXECUTION_FAILURE
        ),
        "r2_execution_failure_fingerprint": r2_failure_fingerprint,
    }


def _local_imported_python_paths() -> set[str]:
    """Return imported local Python sources visible in this process."""

    root = _ROOT.resolve()
    paths: set[str] = set()
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            continue
        path = Path(source)
        if path.suffix in {".pyc", ".pyo"}:
            candidate = path.parent.parent / f"{path.stem.split('.')[0]}.py"
            if candidate.is_file():
                path = candidate
        if not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        if path.suffix == ".py":
            paths.add(str(relative))
    return paths


def _load_implementation_closure(
    protocol_binding: dict[str, object],
) -> dict[str, object]:
    """Validate the bounded r3 correction and inherited source closure."""

    closure = _load_object(
        _IMPLEMENTATION_CLOSURE,
        name="exposure confirmation implementation closure",
    )
    closure_fingerprint = _verify_fingerprint(
        closure,
        field="receipt_fingerprint",
        name="exposure confirmation implementation closure",
    )
    if closure.get("schema_version") != (
        "cure-lite-peco-v10-exposure-confirmation-r3-"
        "implementation-closure-v1"
    ):
        raise RuntimeError("confirmation closure schema differs")
    if closure.get("method_id") != METHOD_ID:
        raise RuntimeError("confirmation closure method differs")
    if closure.get("status") != (
        "R3_SERIALIZATION_CORRECTION_CLOSED_BEFORE_SINGLE_AUTHORITY_RUN"
    ):
        raise RuntimeError("confirmation closure status differs")
    if closure.get("decision") != (
        "PECO_V10_EXPOSURE_CONFIRMATION_R3_IMPLEMENTATION_CLOSED"
    ):
        raise RuntimeError("confirmation closure decision differs")
    if closure.get("config_binding") != {
        "repo_path": protocol_binding["config_repo_path"],
        "file_sha256": protocol_binding["config_file_sha256"],
        "config_fingerprint": protocol_binding["config_fingerprint"],
    }:
        raise RuntimeError("confirmation closure config binding differs")
    if closure.get("design_receipt_binding") != {
        "repo_path": protocol_binding["design_receipt_repo_path"],
        "file_sha256": protocol_binding[
            "design_receipt_file_sha256"
        ],
        "receipt_fingerprint": protocol_binding[
            "design_receipt_fingerprint"
        ],
    }:
        raise RuntimeError(
            "confirmation closure design receipt binding differs"
        )
    if closure.get("r1_invalidation_binding") != {
        "repo_path": protocol_binding["r1_invalidation_repo_path"],
        "file_sha256": protocol_binding[
            "r1_invalidation_file_sha256"
        ],
        "receipt_fingerprint": protocol_binding[
            "r1_invalidation_fingerprint"
        ],
        "decision": (
            "PECO_V10_EXPOSURE_CONFIRMATION_R1_DESIGN_INVALIDATED"
        ),
    }:
        raise RuntimeError(
            "confirmation closure r1 invalidation binding differs"
        )
    expected_r2_failure_binding = {
        "repo_path": protocol_binding[
            "r2_execution_failure_repo_path"
        ],
        "file_sha256": protocol_binding[
            "r2_execution_failure_file_sha256"
        ],
        "receipt_fingerprint": protocol_binding[
            "r2_execution_failure_fingerprint"
        ],
        "decision": (
            "PECO_V10_EXPOSURE_CONFIRMATION_R2_RESULT_NOT_PRODUCED"
        ),
    }
    if closure.get("r2_execution_failure_binding") != (
        expected_r2_failure_binding
    ):
        raise RuntimeError(
            "confirmation closure r2 failure binding differs"
        )
    if closure.get("inherited_r2_implementation_closure_binding") != (
        protocol_binding["r2_implementation_closure_binding"]
    ):
        raise RuntimeError(
            "confirmation inherited r2 closure binding differs"
        )

    development_receipt = _load_object(
        _DEVELOPMENT_R2_RESULT_RECEIPT,
        name="PECO development r2 result receipt",
    )
    development_receipt_fingerprint = _verify_fingerprint(
        development_receipt,
        field="receipt_fingerprint",
        name="PECO development r2 result receipt",
    )
    development_result = _load_object(
        _DEVELOPMENT_R2_RESULT,
        name="PECO development r2 result",
    )
    development_result_fingerprint = _verify_fingerprint(
        development_result,
        field="result_fingerprint",
        name="PECO development r2 result",
    )
    if development_receipt.get("decision") != (
        "PECO_V10_DEVELOPMENT_R2_CORRECTION_PASS"
    ):
        raise RuntimeError("development r2 receipt did not pass")
    if development_result.get("decision") != (
        "PECO_V10_DEVELOPMENT_REGRESSION_PASS"
    ):
        raise RuntimeError("development r2 result did not pass")
    if closure.get("development_r2_prerequisite_binding") != {
        "result_repo_path": str(
            _DEVELOPMENT_R2_RESULT.relative_to(_ROOT)
        ),
        "result_file_sha256": file_sha256(_DEVELOPMENT_R2_RESULT),
        "result_fingerprint": development_result_fingerprint,
        "result_decision": "PECO_V10_DEVELOPMENT_REGRESSION_PASS",
        "receipt_repo_path": str(
            _DEVELOPMENT_R2_RESULT_RECEIPT.relative_to(_ROOT)
        ),
        "receipt_file_sha256": file_sha256(
            _DEVELOPMENT_R2_RESULT_RECEIPT
        ),
        "receipt_fingerprint": development_receipt_fingerprint,
        "receipt_decision": (
            "PECO_V10_DEVELOPMENT_R2_CORRECTION_PASS"
        ),
    }:
        raise RuntimeError(
            "confirmation closure development prerequisite differs"
        )

    r2_closure = _load_object(
        _R2_IMPLEMENTATION_CLOSURE,
        name="exposure confirmation r2 implementation closure",
    )
    r2_source_bindings = r2_closure.get("source_bindings")
    if not isinstance(r2_source_bindings, list) or not r2_source_bindings:
        raise TypeError("r2 confirmation closure sources must be non-empty")
    corrected_bindings = closure.get("corrected_source_bindings")
    if not isinstance(corrected_bindings, list):
        raise TypeError("r3 corrected sources must be a list")
    if len(corrected_bindings) != 2 or not all(
        isinstance(binding, dict) for binding in corrected_bindings
    ):
        raise RuntimeError("r3 must bind exactly two corrected sources")
    corrected_paths = {
        "tests_v10/test_peco_exposure_confirmation.py",
        "tools/evaluate_peco_exposure_confirmation.py",
    }
    ordered_corrected_paths = [
        str(binding.get("repo_path")) for binding in corrected_bindings
    ]
    observed_corrected_paths = set(ordered_corrected_paths)
    if observed_corrected_paths != corrected_paths:
        raise RuntimeError("r3 corrected source set differs")
    if ordered_corrected_paths != sorted(ordered_corrected_paths):
        raise RuntimeError("r3 corrected sources are not canonically sorted")
    replacements = {
        str(binding["repo_path"]): binding
        for binding in corrected_bindings
        if isinstance(binding, dict)
    }
    source_bindings = [
        replacements.get(str(binding.get("repo_path")), binding)
        for binding in r2_source_bindings
        if isinstance(binding, dict)
    ]
    if len(source_bindings) != len(r2_source_bindings):
        raise RuntimeError("r2 source closure contains invalid entries")
    bound_paths: set[str] = set()
    ordered_paths: list[str] = []
    for binding in source_bindings:
        if not isinstance(binding, dict):
            raise TypeError("confirmation closure source must be an object")
        repo_path = binding.get("repo_path")
        expected_sha = binding.get("file_sha256")
        if not isinstance(repo_path, str) or not repo_path:
            raise TypeError("confirmation closure source path is invalid")
        if repo_path in bound_paths:
            raise RuntimeError(
                "confirmation closure source paths contain duplicates"
            )
        bound_paths.add(repo_path)
        ordered_paths.append(repo_path)
        if file_sha256(_ROOT / repo_path) != expected_sha:
            raise RuntimeError(
                f"confirmation closure source differs: {repo_path}"
            )
    if ordered_paths != sorted(ordered_paths):
        raise RuntimeError(
            "confirmation closure sources are not canonically sorted"
        )
    required_paths = {
        "cure_lite/conservative_factorized_decoder.py",
        "cure_lite/experiment/peco_exposure_confirmation.py",
        "cure_lite/losses.py",
        "cure_lite/paired_endpoint_crossing_losses.py",
        "cure_lite/paired_outcome_losses.py",
        "cure_lite/train/paired_outcome_step.py",
        "tests_v10/test_peco_exposure_confirmation.py",
        "tools/evaluate_peco_exposure_confirmation.py",
    }
    if not required_paths.issubset(bound_paths):
        raise RuntimeError(
            "confirmation closure is missing required implementation sources"
        )
    imported_paths = _local_imported_python_paths()
    unbound_imports = sorted(imported_paths - bound_paths)
    if unbound_imports:
        raise RuntimeError(
            "confirmation closure has unbound local imports: "
            f"{unbound_imports}"
        )
    if closure.get("authorization") != {
        "authority_run_count": 1,
        "canonical_output_repo_path": str(
            _CANONICAL_OUTPUT.relative_to(_ROOT)
        ),
        "automatic_retry_allowed": False,
        "deterministic_dry_run_currently_authorized": False,
        "D_R_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
        "detection_performance_allowed": False,
        "real_bounded_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "cross_detector_authorized": False,
    }:
        raise RuntimeError("confirmation closure authorization differs")
    if closure.get("correction_scope") != {
        "component_histogram_keys_canonicalized_as_strings": True,
        "non_string_result_mapping_key_regression_test_added": True,
        "result_schema_and_canonical_output_versioned_to_r3": True,
        "r3_closure_and_pre_run_chain_required": True,
        "model_changed": False,
        "decoder_changed": False,
        "loss_changed": False,
        "population_changed": False,
        "schedule_changed": False,
        "optimizer_changed": False,
        "seed_changed": False,
        "learning_rate_changed": False,
        "updates_changed": False,
        "thresholds_changed": False,
    }:
        raise RuntimeError("confirmation r3 correction scope differs")
    if closure.get("effective_source_closure") != {
        "inherited_source_count": 62,
        "corrected_source_count": 2,
        "effective_bound_source_count": 64,
        "all_observed_local_imports_must_be_bound": True,
        "unchanged_sources_must_retain_r2_sha256": True,
    }:
        raise RuntimeError("confirmation effective source closure differs")
    if closure.get("scope_limitations") != {
        "exact_v8_update_order_replayed": False,
        "exact_v8_source_structure_replayed": False,
        "synthetic_role_features_intentionally_separable": True,
        "pass_scope": (
            "dataset_free_exposure_matched_constructive_"
            "learnability_only"
        ),
    }:
        raise RuntimeError("confirmation r3 scope limitations differ")
    return {
        "repo_path": str(_IMPLEMENTATION_CLOSURE.relative_to(_ROOT)),
        "file_sha256": file_sha256(_IMPLEMENTATION_CLOSURE),
        "receipt_fingerprint": closure_fingerprint,
        "bound_source_count": len(bound_paths),
        "observed_local_import_count": len(imported_paths),
        "observed_local_import_fingerprint": stable_fingerprint(
            sorted(imported_paths)
        ),
    }


def _load_pre_run_receipt(
    protocol_binding: dict[str, object],
    implementation_binding: dict[str, object],
) -> dict[str, object]:
    """Validate the r3 pre-run gate before any confirmation update."""

    receipt = _load_object(
        _PRE_RUN_RECEIPT,
        name="exposure confirmation r3 pre-run receipt",
    )
    receipt_fingerprint = _verify_fingerprint(
        receipt,
        field="receipt_fingerprint",
        name="exposure confirmation r3 pre-run receipt",
    )
    if receipt.get("schema_version") != (
        "cure-lite-peco-v10-exposure-confirmation-r3-"
        "pre-run-verification-v1"
    ):
        raise RuntimeError("confirmation r3 pre-run schema differs")
    if receipt.get("method_id") != METHOD_ID:
        raise RuntimeError("confirmation r3 pre-run method differs")
    if receipt.get("status") != (
        "VERIFIED_BEFORE_SINGLE_R3_AUTHORITY_RUN"
    ):
        raise RuntimeError("confirmation r3 pre-run status differs")
    if receipt.get("decision") != (
        "PECO_V10_EXPOSURE_CONFIRMATION_R3_PRE_RUN_PASS"
    ):
        raise RuntimeError("confirmation r3 pre-run decision differs")
    if receipt.get("implementation_closure_binding") != {
        "repo_path": implementation_binding["repo_path"],
        "file_sha256": implementation_binding["file_sha256"],
        "receipt_fingerprint": implementation_binding[
            "receipt_fingerprint"
        ],
        "decision": (
            "PECO_V10_EXPOSURE_CONFIRMATION_R3_IMPLEMENTATION_CLOSED"
        ),
    }:
        raise RuntimeError("confirmation r3 pre-run closure binding differs")
    if receipt.get("r2_execution_failure_binding") != {
        "repo_path": protocol_binding[
            "r2_execution_failure_repo_path"
        ],
        "file_sha256": protocol_binding[
            "r2_execution_failure_file_sha256"
        ],
        "receipt_fingerprint": protocol_binding[
            "r2_execution_failure_fingerprint"
        ],
        "decision": (
            "PECO_V10_EXPOSURE_CONFIRMATION_R2_RESULT_NOT_PRODUCED"
        ),
    }:
        raise RuntimeError("confirmation r3 pre-run failure binding differs")
    test_evidence = receipt.get("test_evidence")
    if not isinstance(test_evidence, list) or len(test_evidence) != 2:
        raise RuntimeError("confirmation r3 pre-run test evidence differs")
    expected_test_evidence = (
        {
            "command": (
                "/home/md0/ly/MSHNet/.venv/bin/python -m pytest -q "
                "tests_v10"
            ),
            "passed": 36,
        },
        {
            "command": (
                "/home/md0/ly/MSHNet/.venv/bin/python -m pytest -q "
                "tests_v8/test_conservative_factorized_config.py "
                "tests_v8/test_conservative_factorized_decoder.py "
                "tests_v8/test_conservative_factorized_model.py "
                "tests_v8/test_conservative_factorized_support_toy.py "
                "tests_v8/test_conservative_factorized_toy_overfit.py "
                "tests/test_paired_outcome_losses.py "
                "tests/test_paired_outcome_step.py"
            ),
            "passed": 49,
        },
    )
    for evidence, expected in zip(
        test_evidence,
        expected_test_evidence,
        strict=True,
    ):
        if not isinstance(evidence, dict):
            raise TypeError("pre-run test evidence must be an object")
        if (
            evidence.get("command") != expected["command"]
            or evidence.get("passed") != expected["passed"]
            or evidence.get("failed") != 0
            or evidence.get("exit_code") != 0
        ):
            raise RuntimeError("confirmation r3 pre-run tests did not pass")
    if receipt.get("clean_import_preflight") != {
        "effective_bound_source_count": 64,
        "observed_local_import_count": 63,
        "observed_local_import_fingerprint": (
            "bb693eb72f41290a500b77e852663b6f525d32b1339025ea40bd12f38d5a5b6e"
        ),
        "unbound_local_import_count": 0,
        "passed": True,
    }:
        raise RuntimeError("confirmation r3 clean import evidence differs")
    if receipt.get("correction_verification") != {
        "component_update_histogram": {"0": 340, "1": 59, "2": 1},
        "all_histogram_keys_are_strings": True,
        "stable_fingerprint_accepts_corrected_mapping": True,
        "r2_result_remains_absent": True,
        "r3_result_absent_before_authority_run": True,
        "full_receipt_chain_test_required_before_authority_run": True,
    }:
        raise RuntimeError("confirmation r3 correction evidence differs")
    if receipt.get("unchanged_scientific_contract") != {
        "model_changed": False,
        "decoder_changed": False,
        "loss_changed": False,
        "population_changed": False,
        "schedule_changed": False,
        "seed_changed": False,
        "optimizer_changed": False,
        "learning_rate_changed": False,
        "updates_changed": False,
        "thresholds_changed": False,
    }:
        raise RuntimeError("confirmation scientific contract changed")
    authorization = receipt.get("authorization")
    if not isinstance(authorization, dict):
        raise TypeError("confirmation r3 authorization must be an object")
    if authorization != {
        "canonical_authority_run": True,
        "authority_run_count": 1,
        "canonical_output_repo_path": str(
            _CANONICAL_OUTPUT.relative_to(_ROOT)
        ),
        "automatic_retry_allowed": False,
        "deterministic_dry_run_currently_authorized": False,
        "D_R": False,
        "D_V": False,
        "D_T": False,
        "formal_800": False,
        "full_CURE": False,
        "cross_detector": False,
    }:
        raise RuntimeError("confirmation r3 pre-run authorization differs")
    return {
        "repo_path": str(_PRE_RUN_RECEIPT.relative_to(_ROOT)),
        "file_sha256": file_sha256(_PRE_RUN_RECEIPT),
        "receipt_fingerprint": receipt_fingerprint,
    }


def _component_update_histogram(
    schedule: Sequence[object],
) -> dict[str, int]:
    """Return a JSON-canonical component-count histogram."""

    counts = Counter(
        int(getattr(update, "component_count")) for update in schedule
    )
    return {
        str(component_count): int(count)
        for component_count, count in sorted(counts.items())
    }


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


def _frozen_adam_contract(
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


def _build_frozen_adam(
    parameters: object,
) -> tuple[torch.optim.Adam, dict[str, object]]:
    optimizer = torch.optim.Adam(
        parameters,
        lr=FROZEN_LEARNING_RATE,
        betas=FROZEN_ADAM_BETAS,
        eps=FROZEN_ADAM_EPSILON,
        weight_decay=FROZEN_ADAM_WEIGHT_DECAY,
        amsgrad=FROZEN_ADAM_AMSGRAD,
        maximize=FROZEN_ADAM_MAXIMIZE,
        foreach=FROZEN_ADAM_FOREACH,
        capturable=FROZEN_ADAM_CAPTURABLE,
        differentiable=FROZEN_ADAM_DIFFERENTIABLE,
        fused=FROZEN_ADAM_FUSED,
        decoupled_weight_decay=(
            FROZEN_ADAM_DECOUPLED_WEIGHT_DECAY
        ),
    )
    contract = _frozen_adam_contract(optimizer)
    expected = {
        "name": "Adam",
        "learning_rate": FROZEN_LEARNING_RATE,
        "betas": list(FROZEN_ADAM_BETAS),
        "epsilon": FROZEN_ADAM_EPSILON,
        "weight_decay": FROZEN_ADAM_WEIGHT_DECAY,
        "amsgrad": FROZEN_ADAM_AMSGRAD,
        "maximize": FROZEN_ADAM_MAXIMIZE,
        "foreach": FROZEN_ADAM_FOREACH,
        "capturable": FROZEN_ADAM_CAPTURABLE,
        "differentiable": FROZEN_ADAM_DIFFERENTIABLE,
        "fused": FROZEN_ADAM_FUSED,
        "decoupled_weight_decay": (
            FROZEN_ADAM_DECOUPLED_WEIGHT_DECAY
        ),
    }
    if contract != expected:
        raise RuntimeError("frozen Adam contract differs before training")
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


def _evaluate_pair_rows(
    decoder: nn.Module,
    specs: tuple[ConfirmationPairSpec, ConfirmationPairSpec],
) -> list[dict[str, object]]:
    outcome = build_confirmation_outcome_batch(specs)
    with torch.no_grad():
        logits_plus, logits_minus = _paired_endpoint_logits(
            decoder,
            feature=outcome.pair_batch.feature,
            occupancy_plus=outcome.pair_batch.occupancy_plus,
            occupancy_minus=outcome.pair_batch.occupancy_minus,
        )
        plus = torch.sigmoid(logits_plus)
        minus = torch.sigmoid(logits_minus)
        delta = minus - plus
    anchor_background = (
        outcome.pair_batch.image_valid_mask
        & ~outcome.pair_batch.occupancy_plus
        & ~outcome.gt_union
    )
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        response = outcome.response_stratum[index]
        local = outcome.local_zero_stratum[index]
        global_context = outcome.global_zero_stratum[index]
        completion = outcome.completion_plus[index]
        row: dict[str, object] = {
            "pair_id": spec.pair_id,
            "group_id": spec.group_id,
            "pair_kind": spec.pair_kind,
            "exposure_count": spec.exposure_count,
            "plus_completion_min": _minimum(plus[index][completion]),
            "plus_background_max": _maximum(
                plus[index][anchor_background[index]]
            ),
            "H_max_abs": _maximum(delta[index][local].abs()),
            "G_max_abs": _maximum(
                delta[index][global_context].abs()
            ),
        }
        if spec.pair_kind == "clean_positive":
            row.update(
                {
                    "D_plus_max": _maximum(plus[index][response]),
                    "D_minus_min": _minimum(minus[index][response]),
                    "D_delta_mean": _mean(delta[index][response]),
                }
            )
        rows.append(row)
    return rows


def _group_results(
    rows: list[dict[str, object]],
    thresholds: dict[str, object],
) -> tuple[list[dict[str, object]], bool]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    results: list[dict[str, object]] = []
    for group_id in sorted(grouped):
        values = grouped[group_id]
        pair_kind = str(values[0]["pair_kind"])
        metrics: dict[str, float | int] = {
            "pair_count": len(values),
            "slot_count": sum(int(row["exposure_count"]) for row in values),
            "plus_completion_min": min(
                float(row["plus_completion_min"]) for row in values
            ),
            "plus_background_max": max(
                float(row["plus_background_max"]) for row in values
            ),
            "H_max_abs": max(float(row["H_max_abs"]) for row in values),
            "G_max_abs": max(float(row["G_max_abs"]) for row in values),
        }
        checks = {
            "plus_completion": metrics["plus_completion_min"]
            > float(thresholds["plus_completion_min_exclusive"]),
            "plus_background": metrics["plus_background_max"]
            < float(thresholds["plus_background_max_exclusive"]),
            "H": metrics["H_max_abs"]
            <= float(thresholds["zero_H_max_abs_max_inclusive"]),
            "G": metrics["G_max_abs"]
            <= float(thresholds["zero_G_max_abs_max_inclusive"]),
        }
        if pair_kind == "clean_positive":
            metrics.update(
                {
                    "D_plus_max": max(
                        float(row["D_plus_max"]) for row in values
                    ),
                    "D_minus_min": min(
                        float(row["D_minus_min"]) for row in values
                    ),
                    "D_delta_mean_min": min(
                        float(row["D_delta_mean"]) for row in values
                    ),
                }
            )
            checks.update(
                {
                    "D_plus_endpoint": metrics["D_plus_max"]
                    < float(
                        thresholds["clean_D_plus_max_exclusive"]
                    ),
                    "D_minus_endpoint": metrics["D_minus_min"]
                    > float(
                        thresholds["clean_D_minus_min_exclusive"]
                    ),
                    "D_delta": metrics["D_delta_mean_min"]
                    >= float(
                        thresholds["clean_D_delta_mean_min_inclusive"]
                    ),
                }
            )
        results.append(
            {
                "group_id": group_id,
                "pair_kind": pair_kind,
                "metrics": metrics,
                "checks": checks,
                "all_pass": all(checks.values()),
            }
        )
    return results, all(result["all_pass"] for result in results)


def _exact_group_contract(
    groups: list[dict[str, object]],
) -> dict[str, object]:
    observed: dict[str, dict[str, int]] = {}
    for group in groups:
        metrics = group.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError("group metrics must be an object")
        observed[str(group["group_id"])] = {
            "pair_count": int(metrics["pair_count"]),
            "slot_count": int(metrics["slot_count"]),
        }
    checks = {
        "exact_eight_group_set": set(observed) == set(
            EXPECTED_GROUP_CONTRACT
        ),
        "exact_pair_and_slot_counts": observed
        == EXPECTED_GROUP_CONTRACT,
    }
    return {
        "expected": EXPECTED_GROUP_CONTRACT,
        "observed": observed,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def _factual_metrics(decoder: nn.Module) -> dict[str, float]:
    factual = build_confirmation_factual_population()
    with torch.no_grad():
        miss = factual["factual_miss"]
        no_miss = factual["factual_no_miss"]
        miss_score = torch.sigmoid(
            decoder(miss.feature, miss.occupancy)
        )
        no_miss_score = torch.sigmoid(
            decoder(no_miss.feature, no_miss.occupancy)
        )
    target = miss.target > 0.5
    background = miss.valid_mask & ~target
    return {
        "factual_miss_target_min": _minimum(miss_score[target]),
        "factual_miss_background_max": _maximum(
            miss_score[background]
        ),
        "factual_no_miss_max": _maximum(
            no_miss_score[no_miss.valid_mask]
        ),
    }


def _population_objective(
    decoder: nn.Module,
    criterion: OutcomeCompleteTransitionLoss,
    specs: tuple[ConfirmationPairSpec, ...],
) -> float:
    absolute = CURELiteLoss()
    factual = build_confirmation_factual_population()
    with torch.no_grad():
        factual_losses = []
        for name in ("factual_miss", "factual_no_miss"):
            batch = factual[name]
            factual_losses.append(
                absolute(
                    decoder(batch.feature, batch.occupancy),
                    batch.target,
                    batch.valid_mask,
                )["total"]
            )
        paired_losses = []
        for offset in range(0, len(specs), 2):
            outcome = build_confirmation_outcome_batch(
                (specs[offset], specs[offset + 1])
            )
            logits_plus, logits_minus = _paired_endpoint_logits(
                decoder,
                feature=outcome.pair_batch.feature,
                occupancy_plus=outcome.pair_batch.occupancy_plus,
                occupancy_minus=outcome.pair_batch.occupancy_minus,
            )
            paired_losses.append(
                criterion(
                    logits_plus,
                    logits_minus,
                    outcome.completion_plus,
                    outcome.pair_batch.occupancy_plus,
                    outcome.gt_union,
                    outcome.pair_batch.label_increment,
                    outcome.pair_batch.image_valid_mask,
                    outcome.intervention_footprint,
                )["total"]
            )
        total = (
            factual_losses[0]
            + factual_losses[1]
            + torch.stack(paired_losses).mean()
        )
    return float(total.cpu())


def _train_objective(
    *,
    objective_id: str,
    criterion: OutcomeCompleteTransitionLoss,
    specs: tuple[ConfirmationPairSpec, ...],
    thresholds: dict[str, object],
) -> dict[str, object]:
    torch.manual_seed(CONFIRMATION_SEED)
    decoder = CURELiteConservativeFactorizedDecoder(
        feature_channels=8,
        feature_stride=4,
    )
    named_parameters = tuple(decoder.named_parameters())
    if len(named_parameters) != EXPECTED_PARAMETER_TENSORS:
        raise AssertionError("confirmation parameter tensor count differs")
    if sum(parameter.numel() for _, parameter in named_parameters) != (
        EXPECTED_PARAMETER_COUNT
    ):
        raise AssertionError("confirmation parameter count differs")
    initial_fingerprint = _decoder_fingerprint(decoder)
    optimizer, optimizer_contract = _build_frozen_adam(
        decoder.parameters()
    )
    absolute = CURELiteLoss()
    schedule = build_confirmation_schedule(specs)
    by_index = {spec.population_index: spec for spec in specs}

    forward_batch_sizes: list[int] = []

    def observe(_module: object, args: tuple[object, ...]) -> None:
        forward_batch_sizes.append(int(args[0].shape[0]))

    gradient_failures: list[dict[str, object]] = []
    gradient_minimum = float("inf")
    gradient_maximum = 0.0
    first_logs: dict[str, float | int] | None = None
    last_logs: dict[str, float | int] | None = None
    handle = decoder.register_forward_pre_hook(observe)
    try:
        for update in schedule:
            factual = build_confirmation_factual_batches(
                update_index=update.update_index,
            )
            outcome = build_confirmation_outcome_batch(
                tuple(by_index[index] for index in update.population_indices)
            )
            logs = outcome_complete_train_step(
                decoder,
                absolute,
                criterion,
                optimizer,
                factual,
                outcome,
            )
            if update.update_index == 0:
                first_logs = dict(logs)
            if update.update_index == FROZEN_UPDATES - 1:
                last_logs = dict(logs)
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
                gradient_minimum = min(gradient_minimum, norm)
                gradient_maximum = max(gradient_maximum, norm)
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
        raise AssertionError("confirmation training logs are missing")

    decoder.eval()
    rows: list[dict[str, object]] = []
    for offset in range(0, len(specs), 2):
        rows.extend(
            _evaluate_pair_rows(
                decoder,
                (specs[offset], specs[offset + 1]),
            )
        )
    groups, groups_pass = _group_results(rows, thresholds)
    exact_group_contract = _exact_group_contract(groups)
    factual_metrics = _factual_metrics(decoder)
    factual_checks = {
        "factual_miss_target": factual_metrics[
            "factual_miss_target_min"
        ]
        > float(thresholds["factual_miss_target_min_exclusive"]),
        "factual_miss_background": factual_metrics[
            "factual_miss_background_max"
        ]
        < float(thresholds["factual_miss_background_max_exclusive"]),
        "factual_no_miss": factual_metrics["factual_no_miss_max"]
        < float(thresholds["factual_no_miss_max_exclusive"]),
    }
    population_objective = _population_objective(
        decoder,
        criterion,
        specs,
    )
    training_forward_sizes = forward_batch_sizes[: 3 * FROZEN_UPDATES]
    forward_contract = {
        "training_forward_call_count": len(training_forward_sizes),
        "expected_training_forward_call_count": 3 * FROZEN_UPDATES,
        "all_training_calls_batch4": all(
            size == 4 for size in training_forward_sizes
        ),
        "paired_endpoint_forward_is_2B": all(
            training_forward_sizes[index] == 4
            for index in range(2, len(training_forward_sizes), 3)
        ),
        "training_step_decoder_calls": int(
            first_logs["decoder_forward_calls_per_update"]
        ),
        "training_step_decoder_states": int(
            first_logs["decoder_states_per_update"]
        ),
    }
    checks = {
        "all_groups": (
            groups_pass and exact_group_contract["all_pass"] is True
        ),
        "factual": all(factual_checks.values()),
        "population_objective": population_objective
        < float(thresholds["population_total_loss_max_exclusive"]),
        "all_parameter_gradients_finite_nonzero_every_update": (
            not gradient_failures
        ),
        "2B_batched_forward": (
            forward_contract["training_forward_call_count"]
            == forward_contract["expected_training_forward_call_count"]
            and forward_contract["all_training_calls_batch4"]
            and forward_contract["paired_endpoint_forward_is_2B"]
            and forward_contract["training_step_decoder_calls"] == 3
            and forward_contract["training_step_decoder_states"] == 12
        ),
    }
    return {
        "objective_id": objective_id,
        "optimizer_contract": optimizer_contract,
        "initial_decoder_fingerprint": initial_fingerprint,
        "final_decoder_fingerprint": _decoder_fingerprint(decoder),
        "population_objective": population_objective,
        "factual_metrics": factual_metrics,
        "factual_checks": factual_checks,
        "groups": groups,
        "exact_group_contract": exact_group_contract,
        "passed_group_count": sum(
            group["all_pass"] is True for group in groups
        ),
        "failed_group_count": sum(
            group["all_pass"] is not True for group in groups
        ),
        "gradient_contract": {
            "updates_checked": FROZEN_UPDATES,
            "parameter_tensors_per_update": EXPECTED_PARAMETER_TENSORS,
            "gradient_observations": (
                FROZEN_UPDATES * EXPECTED_PARAMETER_TENSORS
            ),
            "failure_count": len(gradient_failures),
            "failures": gradient_failures,
            "minimum_l2_norm": gradient_minimum,
            "maximum_l2_norm": gradient_maximum,
            "all_finite_nonzero_every_update": not gradient_failures,
        },
        "forward_contract": forward_contract,
        "first_update_logs": first_logs,
        "last_update_logs": last_logs,
        "checks": checks,
        "all_pass": all(checks.values()),
        "_trained_decoder": decoder,
    }


def _identical_input_conflict_control(
    decoder: nn.Module,
    thresholds: dict[str, object],
) -> dict[str, object]:
    outcome = build_identical_input_conflict_control()
    with torch.no_grad():
        logits_plus, logits_minus = _paired_endpoint_logits(
            decoder,
            feature=outcome.pair_batch.feature,
            occupancy_plus=outcome.pair_batch.occupancy_plus,
            occupancy_minus=outcome.pair_batch.occupancy_minus,
        )
        delta = (
            torch.sigmoid(logits_minus)
            - torch.sigmoid(logits_plus)
        )
    response_pixel = outcome.response_stratum[0]
    if int(response_pixel.sum()) != 1:
        raise AssertionError("conflict control must contain one response pixel")
    null_same_pixel = response_pixel
    clean_value = _mean(delta[0][response_pixel])
    null_value = _mean(delta[1][null_same_pixel])
    clean_gate = clean_value >= float(
        thresholds["clean_D_delta_mean_min_inclusive"]
    )
    null_gate = abs(null_value) <= float(
        thresholds["zero_H_max_abs_max_inclusive"]
    )

    loss_signature = inspect.signature(
        PairedEndpointCrossingLoss.forward
    )
    decoder_signature = inspect.signature(
        CURELiteConservativeFactorizedDecoder.forward
    )
    loss_source = inspect.getsource(PairedEndpointCrossingLoss.forward)
    checks = {
        "feature_rows_bitwise_identical": torch.equal(
            outcome.pair_batch.feature[0],
            outcome.pair_batch.feature[1],
        ),
        "plus_occupancy_rows_bitwise_identical": torch.equal(
            outcome.pair_batch.occupancy_plus[0],
            outcome.pair_batch.occupancy_plus[1],
        ),
        "minus_occupancy_rows_bitwise_identical": torch.equal(
            outcome.pair_batch.occupancy_minus[0],
            outcome.pair_batch.occupancy_minus[1],
        ),
        "plus_logits_rows_bitwise_identical": torch.equal(
            logits_plus[0],
            logits_plus[1],
        ),
        "minus_logits_rows_bitwise_identical": torch.equal(
            logits_minus[0],
            logits_minus[1],
        ),
        "same_pixel_delta_bitwise_identical": clean_value == null_value,
        "outcome_truth_conflicts": (
            bool(outcome.response_stratum[0][response_pixel].all())
            and not bool(outcome.response_stratum[1][null_same_pixel].any())
            and bool(
                outcome.local_zero_stratum[1][null_same_pixel].all()
            )
        ),
        "clean_and_null_gates_are_mutually_exclusive": not (
            clean_gate and null_gate
        ),
        "loss_signature_has_no_pair_kind": all(
            "pair_kind" not in name for name in loss_signature.parameters
        ),
        "decoder_signature_has_no_pair_kind": all(
            "pair_kind" not in name for name in decoder_signature.parameters
        ),
        "loss_source_has_no_pair_kind_dispatch": (
            "pair_kind" not in loss_source
        ),
    }
    return {
        "clean_response_delta": clean_value,
        "component_same_pixel_delta": null_value,
        "clean_absolute_gate_pass": clean_gate,
        "component_absolute_gate_pass": null_gate,
        "checks": checks,
        "all_pass": all(checks.values()),
        "interpretation": (
            "identical_model_inputs_cannot_receive_role_specific_outputs"
        ),
    }


def evaluate() -> dict[str, object]:
    """Return the deterministic matched confirmation result."""

    config, protocol_binding = _load_protocol()
    implementation_binding = _load_implementation_closure(
        protocol_binding
    )
    protocol_binding["implementation_closure"] = implementation_binding
    protocol_binding["pre_run_verification"] = _load_pre_run_receipt(
        protocol_binding,
        implementation_binding,
    )
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict):
        raise TypeError("confirmation thresholds must be an object")
    specs = build_confirmation_pair_specs()
    schedule = build_confirmation_schedule(specs)

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    confirmation_threads = min(previous_threads, 2)
    try:
        torch.set_num_threads(confirmation_threads)
        torch.use_deterministic_algorithms(True)
        peco = _train_objective(
            objective_id=METHOD_ID,
            criterion=PairedEndpointCrossingLoss(LossConfig()),
            specs=specs,
            thresholds=thresholds,
        )
        predecessor = _train_objective(
            objective_id=PREDECESSOR_METHOD_ID,
            criterion=OutcomeCompleteTransitionLoss(LossConfig()),
            specs=specs,
            thresholds=thresholds,
        )
        same_initialization = (
            peco["initial_decoder_fingerprint"]
            == predecessor["initial_decoder_fingerprint"]
        )
        same_optimizer = (
            peco["optimizer_contract"]
            == predecessor["optimizer_contract"]
        )
        decoder_contract = config.get("decoder")
        if not isinstance(decoder_contract, dict):
            raise TypeError("confirmation decoder contract must be an object")
        expected_initial_fingerprint = decoder_contract.get(
            "expected_seed42_initial_fingerprint"
        )
        frozen_initialization = (
            isinstance(expected_initial_fingerprint, str)
            and peco["initial_decoder_fingerprint"]
            == expected_initial_fingerprint
            and predecessor["initial_decoder_fingerprint"]
            == expected_initial_fingerprint
        )
        negative_control = _identical_input_conflict_control(
            peco["_trained_decoder"],
            thresholds,
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    del peco["_trained_decoder"]
    del predecessor["_trained_decoder"]
    peco_pass = (
        peco["all_pass"] is True
        and same_initialization
        and same_optimizer
        and frozen_initialization
        and negative_control["all_pass"] is True
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "correction_id": "canonical_string_histogram_keys_r3",
        "protocol_binding": protocol_binding,
        "decision": (
            "PECO_V10_EXPOSURE_MATCHED_CONFIRMATION_PASS"
            if peco_pass
            else "PECO_V10_EXPOSURE_MATCHED_CONFIRMATION_FAIL"
        ),
        "all_pass": peco_pass,
        "contract": {
            "seed": CONFIRMATION_SEED,
            "optimizer": "adam",
            "learning_rate": FROZEN_LEARNING_RATE,
            "updates": FROZEN_UPDATES,
            "pair_batch_size": 2,
            "factual_miss_population_size": FACTUAL_POPULATION_SIZE,
            "factual_no_miss_population_size": FACTUAL_POPULATION_SIZE,
            "factual_miss_batch_size": FACTUAL_BATCH_SIZE,
            "factual_no_miss_batch_size": FACTUAL_BATCH_SIZE,
            "factual_miss_slots": FACTUAL_SLOTS_PER_BRANCH,
            "factual_no_miss_slots": FACTUAL_SLOTS_PER_BRANCH,
            "factual_exposures_per_state": FACTUAL_EXPOSURES_PER_STATE,
            "factual_schedule_fingerprint": (
                factual_schedule_fingerprint()
            ),
            "pair_count": len(specs),
            "clean_pair_count": CLEAN_PAIR_COUNT,
            "component_pair_count": COMPONENT_PAIR_COUNT,
            "clean_slots": CLEAN_SLOT_COUNT,
            "component_slots": COMPONENT_SLOT_COUNT,
            "component_update_histogram": (
                _component_update_histogram(schedule)
            ),
            "catalog_fingerprint": catalog_fingerprint(specs),
            "schedule_fingerprint": schedule_fingerprint(specs),
            "parameter_tensors": EXPECTED_PARAMETER_TENSORS,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "thresholds": thresholds,
            "decoder_topology_changed": False,
            "inference_changed": False,
            "pair_kind_is_model_input": False,
            "pair_kind_is_loss_dispatch": False,
        },
        "same_initialization_verified": same_initialization,
        "same_frozen_optimizer_verified": same_optimizer,
        "frozen_seed42_initialization_verified": frozen_initialization,
        "peco": peco,
        "matched_predecessor": predecessor,
        "matched_predecessor_decision": (
            "MATCHED_PREDECESSOR_ABSOLUTE_GATE_PASS"
            if predecessor["all_pass"] is True
            else "MATCHED_PREDECESSOR_ABSOLUTE_GATE_FAIL"
        ),
        "matched_predecessor_affects_peco_decision": False,
        "identical_input_conflict_control": negative_control,
        "runtime": {
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
        },
        "execution_boundary": {
            "dataset_free_confirmation_performed": True,
            "dry_run_authorized": peco_pass,
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
            "dataset_free_exposure_matched_model_code_confirmation"
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
    if args.output.resolve() != _CANONICAL_OUTPUT.resolve():
        raise ValueError(
            "authority output must be the single frozen canonical path: "
            f"{_CANONICAL_OUTPUT}"
        )
    result = evaluate()
    _write_result(args.output, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "all_pass": result["all_pass"],
                "result_fingerprint": result["result_fingerprint"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["all_pass"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
