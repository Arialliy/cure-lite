"""Strict publication verification for CC-SEA v8 bounded core results.

The first publication runner treated each computational-gate record as a
boolean.  The executor has always emitted structured records containing the
observed value, direction, threshold, status, and final boolean decision.  This
module validates that real schema directly and keeps model-code success equal
to the conjunction of structural and computational success.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping

from ..cache.schema import stable_fingerprint
from ..config import LossConfig
from ..conservative_factorized_config import (
    ConservativeFactorizedDecoderConfig,
)
from .conservative_factorized_outcome_bounded import (
    CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
    CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
)
from .factorized_outcome_bounded import FACTORIZED_JOINT_THRESHOLD
from .paired_outcome_bounded import COMPUTATIONAL_THRESHOLDS


_GATE_RULES = {
    "factual_miss_anchor_final_over_initial": (
        "max",
        "factual_miss_anchor_final_over_initial_max",
    ),
    "factual_no_miss_anchor_final_over_initial": (
        "max",
        "factual_no_miss_anchor_final_over_initial_max",
    ),
    "plus_baseline_final_over_initial": (
        "max",
        "plus_baseline_final_over_initial_max",
    ),
    "clean_transition_final_over_initial": (
        "max",
        "clean_transition_final_over_initial_max",
    ),
    "clean_mean_delta_on_D": ("min", "clean_mean_delta_on_D_min"),
    "clean_pairs_delta_at_least_0_25_fraction": (
        "min",
        "clean_pairs_delta_at_least_0_25_fraction_min",
    ),
    "clean_zero_macro_mean_abs_delta": (
        "max",
        "clean_zero_macro_mean_abs_delta_max",
    ),
    "component_null_footprint_macro_mean_abs_delta": (
        "max",
        "component_null_footprint_macro_mean_abs_delta_max",
    ),
    "component_null_footprint_global_max_abs_delta": (
        "max",
        "component_null_footprint_global_max_abs_delta_max",
    ),
    "component_null_context_macro_mean_abs_delta": (
        "max",
        "component_null_context_macro_mean_abs_delta_max",
    ),
    "identity_null_max_abs_delta": (
        "max",
        "identity_null_max_abs_delta_max",
    ),
    "clean_joint_D_ge_0_25_and_H_le_0_05_fraction": (
        "min",
        "clean_joint_D_ge_0_25_and_H_le_0_05_fraction_min",
    ),
}
_GATE_RECORD_KEYS = {
    "value",
    "direction",
    "threshold",
    "applicable",
    "status",
    "pass",
}
_EXPECTED_THRESHOLDS = {
    **dict(COMPUTATIONAL_THRESHOLDS),
    "clean_joint_D_ge_0_25_and_H_le_0_05_fraction_min": (
        FACTORIZED_JOINT_THRESHOLD
    ),
}


@dataclass(frozen=True)
class ConservativeBoundedResultContract:
    """Frozen identities and dimensions for one bounded core result."""

    device: str
    population_fingerprint: str
    materializer_fingerprint: str
    factual_schedule_fingerprint: str
    outcome_schedule_fingerprint: str
    feature_channels: int = 64
    feature_stride: int = 4
    trainable_parameter_count: int = 4385
    trainable_parameter_tensors: int = 6
    evaluation_chunk_size: int = 32
    pair_count: int = 222
    clean_pair_count: int = 206
    component_null_pair_count: int = 16
    optimizer_updates: int = 400
    steps_per_epoch: int = 40
    factual_miss_states_per_update: int = 4
    factual_no_miss_states_per_update: int = 4
    outcome_pairs_per_update: int = 2
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0


FROZEN_REAL_RESULT_CONTRACT = ConservativeBoundedResultContract(
    device="cuda:0",
    population_fingerprint=(
        "d251ed9061dd373aa0bf0e4ceeebbafc7ca32a4bab72c2f24601a20868d6d1cd"
    ),
    materializer_fingerprint=(
        "8cc4eac43ad708265d8639c4b577b37bd81be8ccde73e79993ba18c65dca10ff"
    ),
    factual_schedule_fingerprint=(
        "57264042879d9850aa538e01563496a8d3de7b82556d2b5ef15ca7f32b66fac3"
    ),
    outcome_schedule_fingerprint=(
        "747123867c88fd1444a514bf70e51013b739f39df2857e5ed021239e4847ec93"
    ),
)


def _finite_number(value: object, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
        and (not positive or float(value) > 0.0)
    )


def _expected_budget(
    contract: ConservativeBoundedResultContract,
) -> dict[str, object]:
    return {
        "seed": 42,
        "optimizer_updates": contract.optimizer_updates,
        "steps_per_epoch": contract.steps_per_epoch,
        "factual_miss_states_per_update": (
            contract.factual_miss_states_per_update
        ),
        "factual_no_miss_states_per_update": (
            contract.factual_no_miss_states_per_update
        ),
        "outcome_pairs_per_update": contract.outcome_pairs_per_update,
        "learning_rate": contract.learning_rate,
        "weight_decay": contract.weight_decay,
    }


def verify_computational_gate_records(
    computational: object,
    *,
    structural_execution_pass: bool,
    model_code_gate_pass: bool,
) -> bool:
    """Validate the twelve records and return their aggregate decision."""

    if not isinstance(computational, Mapping):
        raise RuntimeError("CC-SEA v8 computational gates are missing")
    checks = computational.get("checks")
    observed = computational.get("observed")
    thresholds = computational.get("thresholds")
    if (
        computational.get("scope")
        != "bounded_D_R_full_outcome_CC_SEA_v8_model_code_gate"
        or computational.get("not_detection_performance") is not True
        or computational.get("thresholds_unchanged_from_v4_v6_v7")
        is not True
        or not isinstance(checks, Mapping)
        or set(checks) != set(_GATE_RULES)
        or not isinstance(observed, Mapping)
        or set(observed) != set(_GATE_RULES)
        or thresholds != _EXPECTED_THRESHOLDS
    ):
        raise RuntimeError("CC-SEA v8 computational gate schema changed")

    decisions: list[bool] = []
    for name, (expected_direction, threshold_name) in _GATE_RULES.items():
        record = checks[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != _GATE_RECORD_KEYS
            or not _finite_number(record.get("value"))
            or not _finite_number(record.get("threshold"))
            or record.get("direction") != expected_direction
            or record.get("threshold") != thresholds[threshold_name]
            or record.get("applicable") is not True
            or record.get("status") != "EVALUATED"
            or not isinstance(record.get("pass"), bool)
            or not _finite_number(observed.get(name))
            or float(observed[name]) != float(record["value"])
        ):
            raise RuntimeError(
                f"CC-SEA v8 computational gate record changed: {name}"
            )
        expected_pass = (
            float(record["value"]) >= float(record["threshold"])
            if expected_direction == "min"
            else float(record["value"]) <= float(record["threshold"])
        )
        if record["pass"] is not expected_pass:
            raise RuntimeError(
                f"CC-SEA v8 computational gate decision changed: {name}"
            )
        decisions.append(expected_pass)

    aggregate = all(decisions)
    if (
        computational.get("all_pass") is not aggregate
        or model_code_gate_pass
        is not (structural_execution_pass and aggregate)
    ):
        raise RuntimeError("CC-SEA v8 computational decision algebra changed")
    return aggregate


def _verify_trace_and_exposure(
    trace: object,
    exposure: object,
    *,
    contract: ConservativeBoundedResultContract,
) -> None:
    if (
        not isinstance(trace, list)
        or len(trace) != contract.optimizer_updates
        or not isinstance(exposure, Mapping)
    ):
        raise RuntimeError("CC-SEA v8 trace or exposure ledger is missing")

    pair_counts: Counter[str] = Counter()
    miss_counts: Counter[str] = Counter()
    no_miss_counts: Counter[str] = Counter()
    for update, row in enumerate(trace):
        if not isinstance(row, Mapping):
            raise RuntimeError("CC-SEA v8 trace row is malformed")
        pair_ids = row.get("outcome_pair_ids")
        pair_kinds = row.get("outcome_pair_kinds")
        miss_ids = row.get("factual_miss_ids")
        no_miss_ids = row.get("factual_no_miss_ids")
        losses = row.get("losses")
        if (
            row.get("update") != update
            or row.get("epoch") != update // contract.steps_per_epoch
            or row.get("step") != update % contract.steps_per_epoch
            or not isinstance(pair_ids, list)
            or len(pair_ids) != contract.outcome_pairs_per_update
            or len(set(pair_ids)) != contract.outcome_pairs_per_update
            or not all(isinstance(value, str) for value in pair_ids)
            or not isinstance(pair_kinds, list)
            or len(pair_kinds) != contract.outcome_pairs_per_update
            or any(
                value not in {"clean_positive", "component_null"}
                for value in pair_kinds
            )
            or not isinstance(miss_ids, list)
            or len(miss_ids)
            != contract.factual_miss_states_per_update
            or not isinstance(no_miss_ids, list)
            or len(no_miss_ids)
            != contract.factual_no_miss_states_per_update
            or row.get("decoder_forward_calls") != 3
            or row.get("decoder_state_evaluations") != 12
            or row.get("decoder_input_requires_grad_violations") != 0
            or row.get("parameter_gradient_tensors")
            != contract.trainable_parameter_tensors
            or row.get("missing_parameter_gradients") != 0
            or row.get("nonfinite_parameter_gradients") != 0
            or not _finite_number(
                row.get("gradient_l2_norm"),
                positive=True,
            )
            or not isinstance(losses, Mapping)
            or losses.get("decoder_forward_calls_per_update") != 3
            or losses.get("decoder_states_per_update") != 12
        ):
            raise RuntimeError("CC-SEA v8 exact update trace changed")
        pair_counts.update(pair_ids)
        miss_counts.update(str(value) for value in miss_ids)
        no_miss_counts.update(str(value) for value in no_miss_ids)

    pair_rows = exposure.get("outcome_pairs")
    miss_rows = exposure.get("factual_miss")
    no_miss_rows = exposure.get("factual_no_miss")
    source_rows = exposure.get("source_images")
    if (
        not isinstance(pair_rows, list)
        or len(pair_rows) != contract.pair_count
        or not isinstance(miss_rows, list)
        or len(miss_rows) != 16
        or not isinstance(no_miss_rows, list)
        or len(no_miss_rows) != 16
        or not isinstance(source_rows, list)
        or exposure.get("outcome_pair_exposure_values") != [3, 4]
        or exposure.get("identity_null_optimizer_exposure") != 0
    ):
        raise RuntimeError("CC-SEA v8 exposure population changed")

    declared_pair_counts = {
        str(row.get("pair_id")): row.get("count")
        for row in pair_rows
        if isinstance(row, Mapping)
    }
    declared_miss_counts = {
        str(row.get("anchor_id")): row.get("count")
        for row in miss_rows
        if isinstance(row, Mapping)
    }
    declared_no_miss_counts = {
        str(row.get("anchor_id")): row.get("count")
        for row in no_miss_rows
        if isinstance(row, Mapping)
    }
    declared_source_counts = {
        str(row.get("sample_id")): row.get("count")
        for row in source_rows
        if isinstance(row, Mapping)
    }
    recomputed_source_counts: Counter[str] = Counter()
    for row in pair_rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("CC-SEA v8 pair exposure row is malformed")
        sample_id = row.get("sample_id")
        count = row.get("count")
        if not isinstance(sample_id, str) or not isinstance(count, int):
            raise RuntimeError("CC-SEA v8 pair exposure row changed")
        recomputed_source_counts[sample_id] += count
    expected_pair_slots = (
        contract.optimizer_updates * contract.outcome_pairs_per_update
    )
    expected_factual_slots = (
        contract.optimizer_updates
        * contract.factual_miss_states_per_update
    )
    if (
        declared_pair_counts != dict(pair_counts)
        or declared_miss_counts != dict(miss_counts)
        or declared_no_miss_counts != dict(no_miss_counts)
        or sum(pair_counts.values()) != expected_pair_slots
        or sum(miss_counts.values()) != expected_factual_slots
        or sum(no_miss_counts.values()) != expected_factual_slots
        or set(pair_counts.values()) != {3, 4}
        or declared_source_counts != dict(recomputed_source_counts)
        or sum(recomputed_source_counts.values()) != expected_pair_slots
        or len(declared_source_counts) != len(source_rows)
    ):
        raise RuntimeError("CC-SEA v8 exposure ledgers do not reproduce")


def verify_conservative_factorized_core_result(
    result: Mapping[str, Any],
    *,
    contract: ConservativeBoundedResultContract = (
        FROZEN_REAL_RESULT_CONTRACT
    ),
) -> None:
    """Validate one unsigned result emitted by the frozen v8 executor."""

    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint", None)
    structural = result.get("structural_execution_pass")
    model_pass = result.get("computational_model_code_gate_pass")
    audit = result.get("pretraining_structural_audit")
    structural_checks = result.get("structural_checks")
    interpretation = result.get("interpretation")
    decoder_config = ConservativeFactorizedDecoderConfig(
        feature_channels=contract.feature_channels,
        feature_stride=contract.feature_stride,
    )
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
        or result.get("schema_version")
        != CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
        or result.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or result.get("execution_status") != "completed"
        or result.get("device") != contract.device
        or not isinstance(structural, bool)
        or not isinstance(model_pass, bool)
        or result.get("population_fingerprint")
        != contract.population_fingerprint
        or result.get("materializer_fingerprint")
        != contract.materializer_fingerprint
        or result.get("factual_schedule_fingerprint")
        != contract.factual_schedule_fingerprint
        or result.get("outcome_schedule_fingerprint")
        != contract.outcome_schedule_fingerprint
        or result.get("decoder_config") != asdict(decoder_config)
        or result.get("loss_config") != asdict(LossConfig())
        or result.get("optimization_budget")
        != _expected_budget(contract)
        or result.get("evaluation_chunk_size")
        != contract.evaluation_chunk_size
        or not isinstance(audit, Mapping)
        or not isinstance(audit.get("checks"), Mapping)
        or not all(
            isinstance(value, bool)
            for value in audit["checks"].values()
        )
        or audit.get("all_pass") is not all(audit["checks"].values())
        or audit.get("pair_count") != contract.pair_count
        or audit.get("clean_pair_count") != contract.clean_pair_count
        or audit.get("component_null_pair_count")
        != contract.component_null_pair_count
        or audit.get("training_performed") is not False
        or audit.get("D_V_accessed") is not False
        or audit.get("D_T_accessed") is not False
        or not isinstance(structural_checks, Mapping)
        or not structural_checks
        or not all(
            isinstance(value, bool)
            for value in structural_checks.values()
        )
        or not isinstance(interpretation, Mapping)
        or interpretation.get("not_detection_performance_evidence")
        is not True
        or interpretation.get("does_not_establish_Pd_or_FA") is not True
        or interpretation.get("does_not_authorize_formal_training")
        is not True
        or interpretation.get("does_not_directly_authorize_formal_800")
        is not True
        or interpretation.get("eligible_for_frozen_review")
        is not model_pass
        or interpretation.get("D_V_accessed") is not False
        or interpretation.get("D_T_accessed") is not False
        or interpretation.get("calibration_performed") is not False
        or interpretation.get("inference_performed") is not False
        or interpretation.get("base_or_backbone_updated") is not False
        or interpretation.get("identity_null_optimizer_exposure") != 0
    ):
        raise RuntimeError(
            "CC-SEA v8 result violates its frozen bounded boundary"
        )

    expected_decision = (
        "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_PASS"
        if model_pass
        else (
            "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_FAIL"
            if structural
            else "CC_SEA_V8_STRUCTURAL_EXECUTION_FAIL"
        )
    )
    if result.get("decision") != expected_decision:
        raise RuntimeError("CC-SEA v8 core decision is inconsistent")

    if audit.get("all_pass") is not True:
        if (
            structural
            or model_pass
            or result.get("optimizer_updates_completed") != 0
            or result.get("training_performed") is not False
            or result.get("trace") != []
            or result.get("computational_gates")
            != {
                "status": "NOT_EVALUATED_BY_STRUCTURAL_STOP_RULE",
                "all_pass": None,
            }
            or not any(
                value is False for value in structural_checks.values()
            )
            or result.get("forward_budget", {}).get("training")
            != {"calls": 0, "state_evaluations": 0}
        ):
            raise RuntimeError(
                "CC-SEA v8 structural zero-update stop rule changed"
            )
        return

    initial = result.get("initial")
    final = result.get("final")
    parameters = result.get("parameters")
    gradients = result.get("gradients")
    ledger = result.get("execution_ledger")
    forward = result.get("forward_budget")
    deterministic = result.get("deterministic_runtime")
    observation = result.get("state_equation_observation")
    computational = result.get("computational_gates")
    if not isinstance(computational, Mapping):
        raise RuntimeError("CC-SEA v8 computational gates are missing")
    verify_computational_gate_records(
        computational,
        structural_execution_pass=structural,
        model_code_gate_pass=model_pass,
    )
    expected_training_forward = {
        "calls": 3 * contract.optimizer_updates,
        "state_evaluations": 12 * contract.optimizer_updates,
    }
    if (
        not isinstance(initial, Mapping)
        or not isinstance(final, Mapping)
        or structural is not all(structural_checks.values())
        or result.get("optimizer_updates_completed")
        != contract.optimizer_updates
        or result.get("training_performed") is not True
        or not isinstance(parameters, Mapping)
        or parameters.get("trainable_parameter_count")
        != contract.trainable_parameter_count
        or parameters.get("trainable_parameter_tensors")
        != contract.trainable_parameter_tensors
        or parameters.get("expected_parameter_count")
        != contract.trainable_parameter_count
        or any(
            not isinstance(parameters.get(name), str)
            or len(str(parameters.get(name))) != 64
            for name in (
                "initial_decoder_fingerprint",
                "final_decoder_fingerprint",
            )
        )
        or parameters.get("initial_decoder_fingerprint")
        == parameters.get("final_decoder_fingerprint")
        or not _finite_number(
            parameters.get("initial_l2_norm"),
            positive=True,
        )
        or not _finite_number(
            parameters.get("final_l2_norm"),
            positive=True,
        )
        or not isinstance(gradients, Mapping)
        or not _finite_number(
            gradients.get("minimum_update_l2_norm"),
            positive=True,
        )
        or not _finite_number(
            gradients.get("maximum_update_l2_norm"),
            positive=True,
        )
        or float(gradients["minimum_update_l2_norm"])
        > float(gradients["maximum_update_l2_norm"])
        or gradients.get("nonfinite_updates") != 0
        or gradients.get("zero_norm_updates") != 0
        or gradients.get("missing_parameter_gradient_updates") != 0
        or gradients.get("nonfinite_parameter_gradient_updates") != 0
        or not isinstance(ledger, Mapping)
        or dict(ledger)
        != {
            "backward_calls": contract.optimizer_updates,
            "optimizer_steps": contract.optimizer_updates,
            "expected_backward_calls": contract.optimizer_updates,
            "expected_optimizer_steps": contract.optimizer_updates,
        }
        or not isinstance(forward, Mapping)
        or forward.get("training") != expected_training_forward
        or forward.get("expected_training") != expected_training_forward
        or not isinstance(deterministic, Mapping)
        or deterministic.get("contract_satisfied") is not True
        or deterministic.get("flags_restored_after_execution") is not True
        or not isinstance(observation, Mapping)
        or observation.get("additional_decoder_forward_calls") != 0
        or observation.get("all_observed_fields_finite") is not True
        or observation.get("all_observed_allocations_nonnegative")
        is not True
        or observation.get("all_observed_evidence_nonnegative")
        is not True
        or not _finite_number(
            observation.get("maximum_observed_simplex_error")
        )
        or float(observation["maximum_observed_simplex_error"]) > 1.0e-6
        or not _finite_number(
            observation.get(
                "maximum_observed_mass_conservation_relative_error"
            )
        )
        or float(
            observation[
                "maximum_observed_mass_conservation_relative_error"
            ]
        )
        > 1.0e-6
    ):
        raise RuntimeError("CC-SEA v8 full bounded evidence changed")

    _verify_trace_and_exposure(
        result.get("trace"),
        result.get("exposure"),
        contract=contract,
    )


__all__ = [
    "ConservativeBoundedResultContract",
    "FROZEN_REAL_RESULT_CONTRACT",
    "verify_computational_gate_records",
    "verify_conservative_factorized_core_result",
]
