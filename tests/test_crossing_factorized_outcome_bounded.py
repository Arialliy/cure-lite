from __future__ import annotations

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.config import LossConfig
from cure_lite.crossing_factorized_config import (
    CrossingFactorizedDecoderConfig,
)
from cure_lite.crossing_factorized_decoder import (
    CURELiteCrossingFactorizedDecoder,
)
from cure_lite.experiment import (
    crossing_factorized_outcome_bounded as crossing_module,
)
from cure_lite.experiment.crossing_factorized_outcome_bounded import (
    CROSSING_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
    CROSSING_FACTORIZED_OUTCOME_METHOD_ID,
    CROSSING_OPERATOR_STRUCTURAL_CHECKS,
    crossing_factorized_decoder_state_fingerprint,
    execute_crossing_factorized_outcome_bounded,
)
from cure_lite.experiment.factorized_outcome_bounded import (
    FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    FACTORIZED_FROZEN_LEARNING_RATE,
    FACTORIZED_FROZEN_SEED,
    FACTORIZED_FROZEN_WEIGHT_DECAY,
)
from cure_lite.experiment.paired_outcome_inputs import (
    PairedOutcomeInputMaterializer,
)
from cure_lite.recoverable_factorized_config import (
    RecoverableFactorizedDecoderConfig,
)
from cure_lite.recoverable_factorized_decoder import (
    CURELiteRecoverableFactorizedDecoder,
)
from tests.test_paired_outcome_bounded import _budget, _inputs


def _crossing_budget() -> dict[str, object]:
    budget = _budget()
    budget.update(
        {
            "seed": FACTORIZED_FROZEN_SEED,
            "learning_rate": FACTORIZED_FROZEN_LEARNING_RATE,
            "weight_decay": FACTORIZED_FROZEN_WEIGHT_DECAY,
        }
    )
    return budget


def test_crossing_state_fingerprint_binds_v7_class_and_tensor_state() -> None:
    config = CrossingFactorizedDecoderConfig(3, 2)
    first = CURELiteCrossingFactorizedDecoder(config)
    second = CURELiteCrossingFactorizedDecoder(config)
    second.load_state_dict(first.state_dict())

    first_fingerprint = crossing_factorized_decoder_state_fingerprint(first)
    assert first_fingerprint == (
        crossing_factorized_decoder_state_fingerprint(second)
    )
    assert len(first_fingerprint) == 64

    with torch.no_grad():
        next(second.parameters()).reshape(-1)[0].add_(0.125)
    assert crossing_factorized_decoder_state_fingerprint(second) != (
        first_fingerprint
    )

    recoverable = CURELiteRecoverableFactorizedDecoder(
        RecoverableFactorizedDecoderConfig(3, 2)
    )
    with pytest.raises(
        TypeError,
        match="CURELiteCrossingFactorizedDecoder",
    ):
        crossing_factorized_decoder_state_fingerprint(  # type: ignore[arg-type]
            recoverable
        )


def test_crossing_bounded_executes_frozen_synthetic_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    verify_calls = 0
    construction_calls: list[CrossingFactorizedDecoderConfig] = []
    original_verify = PairedOutcomeInputMaterializer.verify_unchanged
    original_init = CURELiteCrossingFactorizedDecoder.__init__

    def counted_verify(self: PairedOutcomeInputMaterializer) -> None:
        nonlocal verify_calls
        verify_calls += 1
        original_verify(self)

    def counted_init(
        self: CURELiteCrossingFactorizedDecoder,
        config: CrossingFactorizedDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        assert isinstance(config, CrossingFactorizedDecoderConfig)
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
        CURELiteCrossingFactorizedDecoder,
        "__init__",
        counted_init,
    )
    config = CrossingFactorizedDecoderConfig(
        feature_channels=3,
        feature_stride=2,
    )
    result = execute_crossing_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        config,
        LossConfig(),
        _crossing_budget(),
        device="cpu",
        evaluation_chunk_size=FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    )

    assert construction_calls == [config]
    assert verify_calls == 2
    assert (
        result["schema_version"]
        == CROSSING_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
    )
    assert result["method_id"] == CROSSING_FACTORIZED_OUTCOME_METHOD_ID
    assert result["execution_status"] == "completed"
    assert result["optimizer_updates_completed"] == 400
    assert result["structural_execution_pass"] is True
    assert all(result["structural_checks"].values())

    structural = result["pretraining_structural_audit"]
    assert structural["all_pass"] is True
    assert structural["population_audit_scope"] == (
        "pretraining_D_R_full_population_CR_LVEC_v7_structure"
    )
    assert structural["count_burden_support_mismatch_pixels"] == 0
    assert structural["local_count_deletion_monotonicity_violations"] == 0
    assert (
        structural[
            "occupancy_burden_deletion_monotonicity_violations"
        ]
        == 0
    )
    assert structural["clean_full_D_reachable_pairs"] == 206
    assert structural["component_positive_count_support_pairs"] == 16
    assert structural["factual_full_target_recoverable_anchors"] == 16
    assert structural["factual_target_recoverable_pixels"] == (
        structural["factual_target_total_pixels"]
    )

    operator = structural["operator_contract"]
    assert tuple(operator["checks"]) == CROSSING_OPERATOR_STRUCTURAL_CHECKS
    assert operator["all_pass"] is True
    assert operator["autograd_backward_calls"] == 2
    assert operator["D_R_accessed"] is False
    assert operator["numeric_probes"]["negative_104_rejected"] is True
    assert operator["numeric_probes"]["positive_89_rejected"] is True
    assert all(
        result["structural_checks"][name] is True
        for name in CROSSING_OPERATOR_STRUCTURAL_CHECKS
    )

    assert result["forward_budget"]["training"] == {
        "calls": 1200,
        "state_evaluations": 4800,
    }
    assert result["forward_budget"]["initial_evaluation"] == {
        "calls": 10,
        "state_evaluations": 508,
    }
    assert result["forward_budget"]["total_excluding_structural_audit"] == {
        "calls": 1220,
        "state_evaluations": 5816,
    }
    locality = structural["independent_nonvacuous_locality_probe"]
    assert locality["all_pass"] is True
    assert locality["changed_support_pixels"] > 0
    assert locality["unchanged_support_pixels"] > 0
    assert structural["compute_budget"] == {
        "decoder_calls": 31,
        "decoder_state_evaluations": 906,
        "expected_decoder_calls": 31,
        "expected_decoder_state_evaluations": 906,
        "factual_forward_fields_calls": 1,
        "factual_forward_fields_states": 16,
        "independent_locality_decoder_calls": 2,
        "independent_locality_decoder_state_evaluations": 2,
    }
    assert result["margin_observation"] == {
        "maximum_observed_absolute_margin": pytest.approx(
            result["margin_observation"][
                "maximum_observed_absolute_margin"
            ]
        ),
        "observed_forward_fields_calls": 1251,
        "all_observed_margins_finite": True,
        "scope": (
            "all_existing_decoder_forward_fields_calls_without_"
            "additional_decoder_computation"
        ),
        "additional_decoder_forward_calls": 0,
        "expected_forward_fields_calls": 1251,
    }
    assert (
        result["margin_observation"][
            "maximum_observed_absolute_margin"
        ]
        > 0.0
    )
    assert result["execution_ledger"] == {
        "backward_calls": 400,
        "optimizer_steps": 400,
        "expected_backward_calls": 400,
        "expected_optimizer_steps": 400,
    }
    assert {
        row["count"] for row in result["exposure"]["outcome_pairs"]
    } == {3, 4}
    assert len(result["exposure"]["outcome_pairs"]) == 222
    assert result["parameters"]["trainable_parameter_count"] == (
        config.expected_parameter_count
    )
    assert result["parameters"]["initial_decoder_fingerprint"] != (
        result["parameters"]["final_decoder_fingerprint"]
    )
    assert result["gradients"]["nonfinite_updates"] == 0
    assert result["gradients"]["zero_norm_updates"] == 0
    assert result["gradients"]["minimum_update_l2_norm"] > 0.0

    gates = result["computational_gates"]
    assert gates["scope"] == (
        "bounded_D_R_full_outcome_CR_LVEC_v7_model_code_gate"
    )
    assert gates["thresholds_unchanged_from_v4_v6"] is True
    assert len(gates["checks"]) == 12
    assert result["interpretation"]["D_V_accessed"] is False
    assert result["interpretation"]["D_T_accessed"] is False
    assert result["interpretation"]["eligible_for_frozen_review"] is (
        result["computational_model_code_gate_pass"]
    )

    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (
            RecoverableFactorizedDecoderConfig(3, 2),
            "CrossingFactorizedDecoderConfig",
        ),
        (object(), "CrossingFactorizedDecoderConfig"),
    ),
)
def test_crossing_bounded_rejects_non_v7_config_types(
    config: object,
    message: str,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    with pytest.raises(TypeError, match=message):
        execute_crossing_factorized_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            config,  # type: ignore[arg-type]
            LossConfig(),
            _crossing_budget(),
            device="cpu",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("seed", 43, "seed"),
        ("learning_rate", 2.0e-3, "learning_rate"),
        ("weight_decay", 1.0e-4, "weight_decay"),
        ("optimizer_updates", 399, "400 optimizer updates"),
    ),
)
def test_crossing_bounded_rejects_any_frozen_budget_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    budget = _crossing_budget()
    budget[field] = value
    with pytest.raises(ValueError, match=message):
        execute_crossing_factorized_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            CrossingFactorizedDecoderConfig(3, 2),
            LossConfig(),
            budget,
            device="cpu",
        )


def test_crossing_bounded_rejects_loss_chunk_and_stride_drift() -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    common = (
        population,
        factual_schedule,
        schedule,
        materializer,
    )
    with pytest.raises(ValueError, match="LossConfig"):
        execute_crossing_factorized_outcome_bounded(
            *common,
            CrossingFactorizedDecoderConfig(3, 2),
            LossConfig(dice_weight=0.5),
            _crossing_budget(),
            device="cpu",
        )
    with pytest.raises(ValueError, match="evaluation_chunk_size"):
        execute_crossing_factorized_outcome_bounded(
            *common,
            CrossingFactorizedDecoderConfig(3, 2),
            LossConfig(),
            _crossing_budget(),
            device="cpu",
            evaluation_chunk_size=64,
        )
    with pytest.raises(ValueError, match="native subpixel path"):
        execute_crossing_factorized_outcome_bounded(
            *common,
            CrossingFactorizedDecoderConfig(3, 1),
            LossConfig(),
            _crossing_budget(),
            device="cpu",
        )


def test_crossing_population_structural_failure_is_zero_update_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    failed_audit = {
        "scope": "test-crossing-population-audit",
        "all_pass": False,
        "checks": {
            "all_audited_fields_finite": False,
            "all_clean_pairs_have_nonempty_H": True,
        },
        "compute_budget": {
            "decoder_calls": 29,
            "decoder_state_evaluations": 904,
        },
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    monkeypatch.setattr(
        crossing_module,
        "audit_crossing_outcome_population",
        lambda *args, **kwargs: dict(failed_audit),
    )

    def forbidden_train(*args, **kwargs):
        raise AssertionError("training must not run after structural failure")

    monkeypatch.setattr(
        crossing_module,
        "outcome_complete_train_step",
        forbidden_train,
    )
    result = execute_crossing_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        CrossingFactorizedDecoderConfig(3, 2),
        LossConfig(),
        _crossing_budget(),
        device="cpu",
    )

    assert result["method_id"] == CROSSING_FACTORIZED_OUTCOME_METHOD_ID
    assert result["decision"] == "CR_LVEC_STRUCTURAL_EXECUTION_FAIL"
    assert result["structural_execution_pass"] is False
    assert result["computational_model_code_gate_pass"] is False
    assert result["optimizer_updates_completed"] == 0
    assert result["training_performed"] is False
    assert result["margin_observation"] == {
        "maximum_observed_absolute_margin": None,
        "observed_forward_fields_calls": 0,
        "all_observed_margins_finite": True,
        "scope": "no_decoder_forward_observed_before_structural_stop",
        "additional_decoder_forward_calls": 0,
    }
    composed = result["pretraining_structural_audit"]
    assert composed["population_audit_scope"] == failed_audit["scope"]
    assert composed["all_pass"] is False
    assert composed["compute_budget"] == failed_audit["compute_budget"]
    assert tuple(composed["operator_contract"]["checks"]) == (
        CROSSING_OPERATOR_STRUCTURAL_CHECKS
    )
    assert result["computational_gates"] == {
        "status": "NOT_EVALUATED_BY_STRUCTURAL_STOP_RULE",
        "all_pass": None,
    }
    assert result["interpretation"]["eligible_for_frozen_review"] is False
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)


def test_crossing_operator_failure_is_zero_update_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    failed_name = "v7_zero_boundary_gradient_equals_one"
    checks = {
        name: name != failed_name
        for name in CROSSING_OPERATOR_STRUCTURAL_CHECKS
    }
    failed_operator_audit = {
        "scope": "test-v7-operator-audit",
        "checks": checks,
        "all_pass": False,
        "autograd_backward_calls": 2,
        "training_performed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    monkeypatch.setattr(
        crossing_module,
        "_audit_crossing_operator_contract",
        lambda **kwargs: dict(failed_operator_audit),
    )

    def forbidden_train(*args, **kwargs):
        raise AssertionError("operator failure must stop before training")

    monkeypatch.setattr(
        crossing_module,
        "outcome_complete_train_step",
        forbidden_train,
    )
    result = execute_crossing_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        CrossingFactorizedDecoderConfig(3, 2),
        LossConfig(),
        _crossing_budget(),
        device="cpu",
    )

    assert result["decision"] == "CR_LVEC_STRUCTURAL_EXECUTION_FAIL"
    assert result["optimizer_updates_completed"] == 0
    assert result["training_performed"] is False
    assert result["structural_execution_pass"] is False
    assert result["structural_checks"][failed_name] is False
    assert result["pretraining_structural_audit"][
        "operator_contract"
    ] == failed_operator_audit
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)
