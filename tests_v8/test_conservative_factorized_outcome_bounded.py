from __future__ import annotations

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.config import LossConfig
from cure_lite.conservative_factorized_config import (
    ConservativeFactorizedDecoderConfig,
)
from cure_lite.conservative_factorized_decoder import (
    CURELiteConservativeFactorizedDecoder,
)
from cure_lite.crossing_factorized_config import (
    CrossingFactorizedDecoderConfig,
)
from cure_lite.crossing_factorized_decoder import (
    CURELiteCrossingFactorizedDecoder,
)
from cure_lite.experiment import (
    conservative_factorized_outcome_bounded as conservative_module,
)
from cure_lite.experiment.conservative_factorized_outcome_bounded import (
    CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
    CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
    CONSERVATIVE_OPERATOR_STRUCTURAL_CHECKS,
    conservative_factorized_decoder_state_fingerprint,
    execute_conservative_factorized_outcome_bounded,
)
from cure_lite.experiment.factorized_outcome_bounded import (
    FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    FACTORIZED_FROZEN_LEARNING_RATE,
    FACTORIZED_FROZEN_SEED,
    FACTORIZED_FROZEN_WEIGHT_DECAY,
    FACTORIZED_JOINT_THRESHOLD,
)
from cure_lite.experiment.paired_outcome_bounded import (
    COMPUTATIONAL_THRESHOLDS,
)
from cure_lite.experiment.paired_outcome_inputs import (
    PairedOutcomeInputMaterializer,
)
from tests.test_paired_outcome_bounded import _budget, _inputs


def _conservative_budget() -> dict[str, object]:
    budget = _budget()
    budget.update(
        {
            "seed": FACTORIZED_FROZEN_SEED,
            "learning_rate": FACTORIZED_FROZEN_LEARNING_RATE,
            "weight_decay": FACTORIZED_FROZEN_WEIGHT_DECAY,
        }
    )
    return budget


def test_conservative_state_fingerprint_binds_v8_class_and_tensor_state() -> None:
    config = ConservativeFactorizedDecoderConfig(3, 2)
    first = CURELiteConservativeFactorizedDecoder(config)
    second = CURELiteConservativeFactorizedDecoder(config)
    second.load_state_dict(first.state_dict())

    fingerprint = conservative_factorized_decoder_state_fingerprint(first)
    assert len(fingerprint) == 64
    assert fingerprint == conservative_factorized_decoder_state_fingerprint(
        second
    )

    with torch.no_grad():
        next(second.parameters()).reshape(-1)[0].add_(0.125)
    assert conservative_factorized_decoder_state_fingerprint(second) != (
        fingerprint
    )

    crossing = CURELiteCrossingFactorizedDecoder(
        CrossingFactorizedDecoderConfig(3, 2)
    )
    crossing.load_state_dict(first.state_dict())
    with pytest.raises(
        TypeError,
        match="CURELiteConservativeFactorizedDecoder",
    ):
        conservative_factorized_decoder_state_fingerprint(  # type: ignore[arg-type]
            crossing
        )


def test_conservative_bounded_executes_frozen_synthetic_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete 400-update core without any dataset loader."""

    population, factual_schedule, schedule, materializer = _inputs()
    verify_calls = 0
    construction_calls: list[ConservativeFactorizedDecoderConfig] = []
    original_verify = PairedOutcomeInputMaterializer.verify_unchanged
    original_init = CURELiteConservativeFactorizedDecoder.__init__

    def counted_verify(self: PairedOutcomeInputMaterializer) -> None:
        nonlocal verify_calls
        verify_calls += 1
        original_verify(self)

    def counted_init(
        self: CURELiteConservativeFactorizedDecoder,
        config: ConservativeFactorizedDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        assert isinstance(config, ConservativeFactorizedDecoderConfig)
        construction_calls.append(config)
        original_init(
            self,
            config,
            feature_channels=feature_channels,
            feature_stride=feature_stride,
        )

    monkeypatch.setattr(
        PairedOutcomeInputMaterializer,
        "verify_unchanged",
        counted_verify,
    )
    monkeypatch.setattr(
        CURELiteConservativeFactorizedDecoder,
        "__init__",
        counted_init,
    )
    config = ConservativeFactorizedDecoderConfig(
        feature_channels=3,
        feature_stride=2,
    )
    result = execute_conservative_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        config,
        LossConfig(),
        _conservative_budget(),
        device="cpu",
        evaluation_chunk_size=FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    )

    # The executor's trainable decoder must be the requested v8 instance.
    # Operator-reference construction is deliberately not coupled to this
    # test: it is already bound by the operator-contract assertions below.
    assert construction_calls == [config]
    assert verify_calls == 2
    assert result["schema_version"] == (
        CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
    )
    assert result["method_id"] == CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
    assert result["execution_status"] == "completed"
    assert result["optimizer_updates_completed"] == 400
    assert result["training_performed"] is True
    assert result["structural_execution_pass"] is True
    assert all(result["structural_checks"].values())
    assert result["decision"] in {
        "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_PASS",
        "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_FAIL",
    }

    structural = result["pretraining_structural_audit"]
    assert structural["all_pass"] is True
    assert structural["population_audit_scope"] == (
        "pretraining_D_R_full_population_CC_SEA_v8_state_equation"
    )
    assert structural["pair_count"] == 222
    assert structural["clean_pair_count"] == 206
    assert structural["component_null_pair_count"] == 16
    assert structural["clean_full_D_reachable_pairs"] == 206
    assert structural["clean_D_reachable_pixels"] == (
        structural["clean_D_total_pixels"]
    )
    assert structural["component_positive_count_support_pairs"] == 16
    assert structural["factual_full_target_reachable_anchors"] == 16
    assert structural["factual_target_reachable_pixels"] == (
        structural["factual_target_total_pixels"]
    )

    # CC-SEA is one state equation: common/contrast coordinates, one simplex
    # allocation, one conserved budget, and occupancy-invariant allocation.
    assert structural["common_mode_max_abs_error"] == 0.0
    assert structural["phase_contrast_sum_max_abs_error"] <= 1.0e-5
    assert structural["phase_allocation_max_abs_occupancy_delta"] == 0.0
    assert structural["phase_allocation_sum_max_abs_error"] <= 1.0e-6
    assert structural["minimum_phase_allocation"] >= 0.0
    assert structural["minimum_evidence_budget"] >= 0.0
    assert structural["minimum_allocated_phase_evidence"] >= 0.0
    assert structural["mass_conservation_max_relative_error"] <= 1.0e-6
    assert (
        structural[
            "phase_mass_delta_budget_delta_max_relative_error"
        ]
        <= 1.0e-6
    )
    assert structural["budget_forward_mismatch_endpoint_batches"] == 0
    assert structural["logit_composition_mismatch_endpoint_batches"] == 0
    assert structural["count_burden_support_mismatch_pixels"] == 0
    assert structural["outside_count_support_max_abs_logit_delta"] == 0.0
    assert (
        structural["outside_count_support_max_abs_probability_delta"]
        == 0.0
    )
    assert structural["evidence_budget_deletion_monotonicity_violations"] == 0
    assert structural["allocated_evidence_deletion_monotonicity_violations"] == 0
    assert structural["deletion_logit_monotonicity_violations"] == 0
    assert structural["deletion_probability_monotonicity_violations"] == 0
    assert structural["field_resize_endpoint_count"] == 0

    operator = structural["operator_contract"]
    assert tuple(operator["checks"]) == CONSERVATIVE_OPERATOR_STRUCTURAL_CHECKS
    assert operator["all_pass"] is True
    assert operator["autograd_gradient_calls"] == 3
    assert operator["D_R_accessed"] is False
    assert operator["D_V_accessed"] is False
    assert operator["D_T_accessed"] is False
    assert all(
        result["structural_checks"][name] is True
        for name in CONSERVATIVE_OPERATOR_STRUCTURAL_CHECKS
    )
    equation = operator["equation_errors"]
    assert equation["phase_contrast_sum_max_abs_error"] <= 1.0e-12
    assert equation["simplex_max_abs_error"] <= 1.0e-12
    assert equation["mass_conservation_max_abs_error"] <= 1.0e-12
    assert equation["zero_mean_contrast_allocation_max_abs_change"] > 1.0e-3
    assert equation["occupancy_release_budget_delta"] > 0.0

    locality = structural["phase_aware_locality_probe"]
    assert locality["all_pass"] is True
    assert locality["changed_support_pixels"] > 0
    assert locality["unchanged_support_pixels"] > 0
    assert locality["feature_gradient_is_none"] is True
    assert locality["controlled_outside_max_abs_delta"] == 0.0
    assert locality["actual_outside_max_abs_logit_delta"] == 0.0
    assert locality["actual_outside_max_abs_probability_delta"] == 0.0
    assert all(locality["checks"].values())

    endpoints = structural["dual_endpoint_gradient_audit"]
    assert endpoints["all_pass"] is True
    assert endpoints["pair_kinds"] == [
        "clean_positive",
        "component_null",
    ]
    assert endpoints["observed_decoder_batch_sizes"] == [4]
    assert endpoints["decoder_calls"] == 1
    assert endpoints["decoder_state_evaluations"] == 4
    assert all(endpoints["checks"].values())
    assert all(
        row["plus_gradient_finite"] is True
        and row["plus_gradient_nonzero_count"] > 0
        and row["minus_gradient_finite"] is True
        and row["minus_gradient_nonzero_count"] > 0
        for row in endpoints["records"]
    )

    assert structural["compute_budget"] == {
        "decoder_calls": 33,
        "decoder_state_evaluations": 911,
        "expected_decoder_calls": 33,
        "expected_decoder_state_evaluations": 911,
        "population_chunk_decoder_calls": 28,
        "population_chunk_decoder_state_evaluations": 888,
        "phase_aware_locality_decoder_calls": 3,
        "phase_aware_locality_decoder_state_evaluations": 3,
        "dual_endpoint_decoder_calls": 1,
        "dual_endpoint_decoder_state_evaluations": 4,
        "factual_forward_fields_calls": 1,
        "factual_forward_fields_states": 16,
    }
    assert result["forward_budget"]["training"] == {
        "calls": 1200,
        "state_evaluations": 4800,
    }
    assert result["forward_budget"]["initial_evaluation"] == {
        "calls": 10,
        "state_evaluations": 508,
    }
    assert result["forward_budget"]["final_evaluation"] == {
        "calls": 10,
        "state_evaluations": 508,
    }
    assert result["forward_budget"]["total_excluding_structural_audit"] == {
        "calls": 1220,
        "state_evaluations": 5816,
    }
    input_detach = result["forward_budget"]["input_detach"]
    assert input_detach["training"] == {
        "calls": 1200,
        "state_evaluations": 4800,
        "requires_grad_violations": 0,
    }
    assert input_detach["total_excluding_structural_audit"] == {
        "calls": 1220,
        "state_evaluations": 5816,
        "requires_grad_violations": 0,
    }

    observation = result["state_equation_observation"]
    assert observation["observed_forward_fields_calls"] == 1253
    assert observation["expected_forward_fields_calls"] == 1253
    assert observation["additional_decoder_forward_calls"] == 0
    assert observation["all_observed_fields_finite"] is True
    assert observation["all_observed_allocations_nonnegative"] is True
    assert observation["all_observed_evidence_nonnegative"] is True
    assert observation["maximum_observed_simplex_error"] <= 1.0e-6
    assert (
        observation[
            "maximum_observed_mass_conservation_relative_error"
        ]
        <= 1.0e-6
    )

    assert result["execution_ledger"] == {
        "backward_calls": 400,
        "optimizer_steps": 400,
        "expected_backward_calls": 400,
        "expected_optimizer_steps": 400,
    }
    assert len(result["trace"]) == 400
    assert all(
        row["decoder_forward_calls"] == 3
        and row["decoder_state_evaluations"] == 12
        and row["decoder_input_requires_grad_violations"] == 0
        and row["parameter_gradient_tensors"] == 6
        and row["missing_parameter_gradients"] == 0
        and row["nonfinite_parameter_gradients"] == 0
        for row in result["trace"]
    )
    assert {
        row["count"] for row in result["exposure"]["outcome_pairs"]
    } == {3, 4}
    assert len(result["exposure"]["outcome_pairs"]) == 222
    assert result["parameters"]["trainable_parameter_count"] == (
        config.expected_parameter_count
    )
    assert result["parameters"]["trainable_parameter_tensors"] == 6
    assert result["parameters"]["initial_decoder_fingerprint"] != (
        result["parameters"]["final_decoder_fingerprint"]
    )
    assert result["gradients"]["nonfinite_updates"] == 0
    assert result["gradients"]["zero_norm_updates"] == 0
    assert result["gradients"]["missing_parameter_gradient_updates"] == 0
    assert result["gradients"]["nonfinite_parameter_gradient_updates"] == 0
    assert result["gradients"]["minimum_update_l2_norm"] > 0.0

    gates = result["computational_gates"]
    expected_thresholds = {
        **dict(COMPUTATIONAL_THRESHOLDS),
        "clean_joint_D_ge_0_25_and_H_le_0_05_fraction_min": (
            FACTORIZED_JOINT_THRESHOLD
        ),
    }
    assert gates["scope"] == (
        "bounded_D_R_full_outcome_CC_SEA_v8_model_code_gate"
    )
    assert gates["thresholds"] == expected_thresholds
    assert gates["thresholds_unchanged_from_v4_v6_v7"] is True
    assert len(gates["checks"]) == 12
    assert result["interpretation"]["D_V_accessed"] is False
    assert result["interpretation"]["D_T_accessed"] is False
    assert result["interpretation"]["base_or_backbone_updated"] is False
    assert result["interpretation"]["eligible_for_frozen_review"] is (
        result["computational_model_code_gate_pass"]
    )

    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)


@pytest.mark.parametrize(
    "config",
    (
        CrossingFactorizedDecoderConfig(3, 2),
        object(),
    ),
)
def test_conservative_bounded_rejects_non_v8_config_types(
    config: object,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    with pytest.raises(
        TypeError,
        match="ConservativeFactorizedDecoderConfig",
    ):
        execute_conservative_factorized_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            config,  # type: ignore[arg-type]
            LossConfig(),
            _conservative_budget(),
            device="cpu",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("seed", 43, "seed"),
        ("learning_rate", 2.0e-3, "learning_rate"),
        ("weight_decay", 1.0e-4, "weight_decay"),
        ("optimizer_updates", 399, "400 optimizer updates"),
        ("steps_per_epoch", 20, "40 steps per epoch"),
        (
            "factual_miss_states_per_update",
            3,
            "four factual-miss",
        ),
        (
            "factual_no_miss_states_per_update",
            3,
            "four factual-no-miss",
        ),
        ("outcome_pairs_per_update", 1, "two outcome pairs"),
    ),
)
def test_conservative_bounded_rejects_any_frozen_budget_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    budget = _conservative_budget()
    budget[field] = value
    with pytest.raises(ValueError, match=message):
        execute_conservative_factorized_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            ConservativeFactorizedDecoderConfig(3, 2),
            LossConfig(),
            budget,
            device="cpu",
        )


def test_conservative_bounded_rejects_loss_chunk_and_stride_drift() -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    common = (
        population,
        factual_schedule,
        schedule,
        materializer,
    )
    with pytest.raises(ValueError, match="LossConfig"):
        execute_conservative_factorized_outcome_bounded(
            *common,
            ConservativeFactorizedDecoderConfig(3, 2),
            LossConfig(dice_weight=0.5),
            _conservative_budget(),
            device="cpu",
        )
    with pytest.raises(ValueError, match="evaluation_chunk_size"):
        execute_conservative_factorized_outcome_bounded(
            *common,
            ConservativeFactorizedDecoderConfig(3, 2),
            LossConfig(),
            _conservative_budget(),
            device="cpu",
            evaluation_chunk_size=64,
        )
    with pytest.raises(ValueError, match="native subpixel path"):
        execute_conservative_factorized_outcome_bounded(
            *common,
            ConservativeFactorizedDecoderConfig(3, 1),
            LossConfig(),
            _conservative_budget(),
            device="cpu",
        )


def test_conservative_population_structural_failure_is_zero_update_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    failed_audit = {
        "scope": "test-v8-population-audit",
        "all_pass": False,
        "checks": {
            "common_mode_equals_phase_mean": False,
            "phase_allocation_nonnegative_simplex": True,
        },
        "compute_budget": {
            "decoder_calls": 0,
            "decoder_state_evaluations": 0,
        },
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    monkeypatch.setattr(
        conservative_module,
        "audit_conservative_outcome_population",
        lambda *args, **kwargs: dict(failed_audit),
    )

    def forbidden_train(*args: object, **kwargs: object) -> None:
        raise AssertionError("training must not run after structural failure")

    monkeypatch.setattr(
        conservative_module,
        "outcome_complete_train_step",
        forbidden_train,
    )
    result = execute_conservative_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        ConservativeFactorizedDecoderConfig(3, 2),
        LossConfig(),
        _conservative_budget(),
        device="cpu",
    )

    assert result["decision"] == "CC_SEA_V8_STRUCTURAL_EXECUTION_FAIL"
    assert result["optimizer_updates_completed"] == 0
    assert result["training_performed"] is False
    assert result["trace"] == []
    assert result["forward_budget"]["training"] == {
        "calls": 0,
        "state_evaluations": 0,
    }
    assert result["structural_execution_pass"] is False
    assert result["computational_model_code_gate_pass"] is False
    assert result["computational_gates"] == {
        "status": "NOT_EVALUATED_BY_STRUCTURAL_STOP_RULE",
        "all_pass": None,
    }
    assert result["state_equation_observation"][
        "observed_forward_fields_calls"
    ] == 0
    assert result["interpretation"]["eligible_for_frozen_review"] is False
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)


def test_conservative_operator_structural_failure_is_zero_update_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    passing_population = {
        "scope": "test-v8-passing-population-audit",
        "all_pass": True,
        "checks": {"population_contract": True},
        "compute_budget": {
            "decoder_calls": 0,
            "decoder_state_evaluations": 0,
        },
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    failed_name = "v8_phase_mass_equals_single_budget"
    checks = {
        name: name != failed_name
        for name in CONSERVATIVE_OPERATOR_STRUCTURAL_CHECKS
    }
    failed_operator_audit = {
        "scope": "test-v8-operator-audit",
        "checks": checks,
        "all_pass": False,
        "autograd_gradient_calls": 3,
        "training_performed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    monkeypatch.setattr(
        conservative_module,
        "audit_conservative_outcome_population",
        lambda *args, **kwargs: dict(passing_population),
    )
    monkeypatch.setattr(
        conservative_module,
        "_audit_conservative_operator_contract",
        lambda **kwargs: dict(failed_operator_audit),
    )

    def forbidden_train(*args: object, **kwargs: object) -> None:
        raise AssertionError("training must not run after operator failure")

    monkeypatch.setattr(
        conservative_module,
        "outcome_complete_train_step",
        forbidden_train,
    )
    result = execute_conservative_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        ConservativeFactorizedDecoderConfig(3, 2),
        LossConfig(),
        _conservative_budget(),
        device="cpu",
    )

    assert result["decision"] == "CC_SEA_V8_STRUCTURAL_EXECUTION_FAIL"
    assert result["optimizer_updates_completed"] == 0
    assert result["training_performed"] is False
    assert result["trace"] == []
    assert result["structural_checks"][failed_name] is False
    assert result["pretraining_structural_audit"][
        "operator_contract"
    ] == failed_operator_audit
    assert result["state_equation_observation"][
        "observed_forward_fields_calls"
    ] == 0
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)
