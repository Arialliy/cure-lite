from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.nlcc_dataset_free_decision import (
    GATE_FAIL,
    PASS,
    InvalidResultError,
    recompute_result_decision,
    strict_json_loads,
)
from cure_lite.nlcc_dataset_free_runner import (
    RESULT_SCHEMA,
    evaluate_cached_logits,
    materialize_profile,
)
from cure_lite.nlcc_dataset_free_runner_config import (
    development_runner_config,
)


@pytest.fixture(scope="module")
def passing_result_and_config():
    config = development_runner_config()
    cache = materialize_profile(config)
    pair_shape = cache.pair_population.pair_batch.label_increment.shape
    plus = torch.full(pair_shape, -20.0)
    minus = plus.clone()
    positive_anchor = cache.pair_population.completion_plus
    plus[positive_anchor] = 20.0
    minus[positive_anchor] = 20.0
    D = cache.strata.D
    plus[D] = -20.0
    minus[D] = 20.0

    miss = cache.factual_population["factual_miss"]
    miss_logits = torch.full_like(miss.target, -20.0)
    miss_logits[miss.target > 0.5] = 20.0
    no_miss = cache.factual_population["factual_no_miss"]
    no_miss_logits = torch.full_like(no_miss.target, -20.0)

    structural = {
        "updates_executed": config.profile.updates,
        "expected_updates": config.profile.updates,
        "training_forward_call_count": 3 * config.profile.updates,
        "expected_training_forward_call_count": 3 * config.profile.updates,
        "training_forward_pattern_counts": {"4,4,4": config.profile.updates},
        "all_update_forward_patterns_4_4_4": True,
        "step_contract_failure_count": 0,
        "gradient_failure_count": 0,
        "finite_state_audit_count": config.profile.updates + 1,
        "expected_finite_state_audit_count": config.profile.updates + 1,
        "finite_state_nonfinite_element_count": 0,
        "all_six_gradients_finite_nonzero_every_update": True,
        "feature_cache_grad_tensor_count": 0,
        "feature_cache_leaves_remain_without_grad": True,
        "one_backward_and_one_step_per_update": True,
        "population_builder_reentry": False,
        "from_scratch_seed_42": True,
        "fresh_adam_state_before_first_update": True,
        "development_checkpoint_loaded": False,
        "development_optimizer_state_loaded": False,
        "all_pass": True,
    }
    final = evaluate_cached_logits(
        cache,
        logits_plus=plus,
        logits_minus=minus,
        factual_miss_logits=miss_logits,
        factual_no_miss_logits=no_miss_logits,
        structural_training_contract=structural,
        operator_field_diagnostics={
            "crossing_margin": {"minimum": -1.0, "maximum": 1.0},
            "recovery_factor": {"minimum": 0.1, "maximum": 2.0},
            "forward_fields_call_count": 3,
            "forward_fields_batch_sizes": [
                2 * len(cache.specs),
                int(miss.feature.shape[0]),
                int(no_miss.feature.shape[0]),
            ],
            "field_tensor_count": 45,
            "field_element_count": 1,
            "field_nonfinite_element_count": 0,
        },
    )
    initial_audit = {
        "phase": "before_first_update",
        "update_index": -1,
        "parameter_tensor_count": 6,
        "buffer_tensor_count": 0,
        "optimizer_state_tensor_count": 0,
        "total_tensor_count": 6,
        "total_element_count": 2593,
        "nonfinite_element_count": 0,
        "nonfinite_tensor_paths": [],
        "maximum_absolute_value": 1.0,
        "global_l2_norm": 1.0,
        "all_finite": True,
    }
    final_audit = {
        **initial_audit,
        "phase": "after_optimizer_step",
        "update_index": config.profile.updates - 1,
        "optimizer_state_tensor_count": 18,
        "total_tensor_count": 24,
        "total_element_count": 3 * 2593 + 6,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "method_id": config.method_id,
        "profile_kind": config.profile.kind,
        "profile_id": config.profile.profile_id,
        "evidentiary_role": config.profile.evidentiary_role,
        "decision": "NLCC_V12_DEVELOPMENT_PASS",
        "all_pass": True,
        "attempt_binding": {},
        "config": config.manifest(),
        "materialized_cache": cache.manifest(),
        "optimizer_contract": {},
        "initial_decoder_fingerprint": "1" * 64,
        "final_decoder_fingerprint": "2" * 64,
        "training": {
            "structural_contract": structural,
            "gradient_minimum_l2": 1.0,
            "gradient_maximum_l2": 2.0,
            "gradient_failures": [],
            "step_contract_failures": [],
            "finite_state_audit": {
                "call_count": config.profile.updates + 1,
                "expected_call_count": config.profile.updates + 1,
                "initial": initial_audit,
                "final": final_audit,
                "nonfinite_element_count": 0,
            },
            "first_update_logs": {},
            "last_update_logs": {},
        },
        "final_evaluation": final,
        "profile_independence": {},
        "runtime": {},
        "execution_boundary": {},
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result, config


def _resign(result: dict[str, object]) -> None:
    result.pop("result_fingerprint", None)
    result["result_fingerprint"] = stable_fingerprint(result)


def test_strict_json_rejects_nonfinite_constants_and_duplicate_keys() -> None:
    for constant in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(InvalidResultError, match="forbidden"):
            strict_json_loads('{"metric":' + constant + "}")
    with pytest.raises(InvalidResultError, match="duplicate"):
        strict_json_loads('{"all_pass":true,"all_pass":false}')


def test_valid_pass_is_recomputed_from_76_numeric_gates(
    passing_result_and_config,
) -> None:
    result, config = passing_result_and_config
    recomputed = recompute_result_decision(deepcopy(result), config)
    assert recomputed.status == PASS
    assert recomputed.all_pass is True
    assert recomputed.decision == "NLCC_V12_DEVELOPMENT_PASS"
    assert recomputed.numeric_gate_count == 76
    assert len(recomputed.numeric_gates) == 76
    assert all(gate.passed for gate in recomputed.numeric_gates)
    assert all(gate.passed for gate in recomputed.structural_gates)


def test_skeletal_pass_is_rejected(passing_result_and_config) -> None:
    _, config = passing_result_and_config
    skeletal = {
        "schema_version": RESULT_SCHEMA,
        "method_id": config.method_id,
        "profile_kind": config.profile.kind,
        "profile_id": config.profile.profile_id,
        "decision": "NLCC_V12_DEVELOPMENT_PASS",
        "all_pass": True,
    }
    skeletal["result_fingerprint"] = stable_fingerprint(skeletal)
    with pytest.raises(InvalidResultError, match="missing keys"):
        recompute_result_decision(skeletal, config)


def test_raw_metric_failure_cannot_hide_behind_embedded_pass(
    passing_result_and_config,
) -> None:
    source, config = passing_result_and_config
    result = deepcopy(source)
    result["final_evaluation"]["global_metrics"][
        "population_total_loss"
    ] = 0.1
    _resign(result)
    with pytest.raises(
        InvalidResultError,
        match="global_checks.population_total_loss differs",
    ):
        recompute_result_decision(result, config)


def test_derived_group_or_structural_mismatch_is_rejected(
    passing_result_and_config,
) -> None:
    source, config = passing_result_and_config
    group_result = deepcopy(source)
    group_result["final_evaluation"]["groups"][0]["all_pass"] = False
    _resign(group_result)
    with pytest.raises(InvalidResultError, match=r"groups\[0\].all_pass"):
        recompute_result_decision(group_result, config)

    structural_result = deepcopy(source)
    structural_result["training"]["structural_contract"][
        "gradient_failure_count"
    ] = 1
    structural_result["final_evaluation"]["structural_training_contract"][
        "gradient_failure_count"
    ] = 1
    _resign(structural_result)
    with pytest.raises(
        InvalidResultError,
        match="gradient_failure_count differs",
    ):
        recompute_result_decision(structural_result, config)


def test_nonfinite_value_in_mapping_is_rejected_before_fingerprint_use(
    passing_result_and_config,
) -> None:
    source, config = passing_result_and_config
    result = deepcopy(source)
    result["final_evaluation"]["global_metrics"][
        "population_total_loss"
    ] = float("nan")
    with pytest.raises(InvalidResultError, match="must be finite"):
        recompute_result_decision(result, config)


def test_valid_scientific_gate_failure_returns_gate_fail(
    passing_result_and_config,
) -> None:
    source, config = passing_result_and_config
    result = deepcopy(source)
    result["final_evaluation"]["global_metrics"][
        "population_total_loss"
    ] = 0.1
    result["final_evaluation"]["global_checks"][
        "population_total_loss"
    ] = False
    result["final_evaluation"]["all_pass"] = False
    result["all_pass"] = False
    result["decision"] = "NLCC_V12_DEVELOPMENT_FAIL"
    _resign(result)

    recomputed = recompute_result_decision(result, config)
    assert recomputed.status == GATE_FAIL
    assert recomputed.all_pass is False
    assert recomputed.decision == "NLCC_V12_DEVELOPMENT_FAIL"
    failed = [gate.gate_id for gate in recomputed.numeric_gates if not gate.passed]
    assert failed == ["global/population_total_loss"]


def test_unknown_gate_and_wrong_numeric_gate_count_are_rejected(
    passing_result_and_config,
) -> None:
    source, config = passing_result_and_config
    extra_gate = deepcopy(source)
    extra_gate["final_evaluation"]["global_checks"]["unknown"] = True
    _resign(extra_gate)
    with pytest.raises(InvalidResultError, match="global_checks keys differ"):
        recompute_result_decision(extra_gate, config)

    wrong_count = deepcopy(source)
    wrong_count["final_evaluation"]["numeric_gate_count"] = 75
    _resign(wrong_count)
    with pytest.raises(InvalidResultError, match="numeric_gate_count differs"):
        recompute_result_decision(wrong_count, config)


def test_raw_finite_and_operator_counts_are_enforced(
    passing_result_and_config,
) -> None:
    source, config = passing_result_and_config
    finite_count = deepcopy(source)
    finite_count["training"]["structural_contract"][
        "finite_state_audit_count"
    ] -= 1
    finite_count["final_evaluation"]["structural_training_contract"][
        "finite_state_audit_count"
    ] -= 1
    finite_count["training"]["finite_state_audit"]["call_count"] -= 1
    _resign(finite_count)
    with pytest.raises(InvalidResultError, match="structural.all_pass differs"):
        recompute_result_decision(finite_count, config)

    operator_count = deepcopy(source)
    operator_count["final_evaluation"]["operator_field_diagnostics"][
        "forward_fields_call_count"
    ] = 2
    _resign(operator_count)
    with pytest.raises(InvalidResultError, match="forward_fields_call_count"):
        recompute_result_decision(operator_count, config)
