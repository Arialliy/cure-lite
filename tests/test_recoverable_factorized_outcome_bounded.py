from __future__ import annotations

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.config import LossConfig
from cure_lite.directed_factorized_config import (
    DirectedFactorizedDecoderConfig,
)
from cure_lite.directed_factorized_decoder import (
    CURELiteDirectedFactorizedDecoder,
)
from cure_lite.experiment import (
    recoverable_factorized_outcome_bounded as recoverable_module,
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
from cure_lite.experiment.recoverable_factorized_outcome_bounded import (
    RECOVERABLE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
    RECOVERABLE_FACTORIZED_OUTCOME_METHOD_ID,
    RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS,
    execute_recoverable_factorized_outcome_bounded,
    recoverable_factorized_decoder_state_fingerprint,
)
from cure_lite.recoverable_factorized_config import (
    RecoverableFactorizedDecoderConfig,
)
from cure_lite.recoverable_factorized_decoder import (
    CURELiteRecoverableFactorizedDecoder,
)
from tests.test_paired_outcome_bounded import _budget, _inputs


def _recoverable_budget() -> dict[str, object]:
    budget = _budget()
    budget.update(
        {
            "seed": FACTORIZED_FROZEN_SEED,
            "learning_rate": FACTORIZED_FROZEN_LEARNING_RATE,
            "weight_decay": FACTORIZED_FROZEN_WEIGHT_DECAY,
        }
    )
    return budget


def test_recoverable_state_fingerprint_binds_v6_class_and_tensor_state() -> None:
    config = RecoverableFactorizedDecoderConfig(3, 2)
    first = CURELiteRecoverableFactorizedDecoder(config)
    second = CURELiteRecoverableFactorizedDecoder(config)
    second.load_state_dict(first.state_dict())

    first_fingerprint = recoverable_factorized_decoder_state_fingerprint(first)
    assert first_fingerprint == (
        recoverable_factorized_decoder_state_fingerprint(second)
    )
    assert len(first_fingerprint) == 64

    with torch.no_grad():
        next(second.parameters()).reshape(-1)[0].add_(0.125)
    assert recoverable_factorized_decoder_state_fingerprint(second) != (
        first_fingerprint
    )

    directed = CURELiteDirectedFactorizedDecoder(
        DirectedFactorizedDecoderConfig(3, 2)
    )
    with pytest.raises(
        TypeError,
        match="CURELiteRecoverableFactorizedDecoder",
    ):
        recoverable_factorized_decoder_state_fingerprint(  # type: ignore[arg-type]
            directed
        )


def test_recoverable_bounded_constructs_v6_and_executes_frozen_toy_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    verify_calls = 0
    construction_calls: list[RecoverableFactorizedDecoderConfig] = []
    original_verify = PairedOutcomeInputMaterializer.verify_unchanged
    original_init = CURELiteRecoverableFactorizedDecoder.__init__

    def counted_verify(self: PairedOutcomeInputMaterializer) -> None:
        nonlocal verify_calls
        verify_calls += 1
        original_verify(self)

    def counted_init(
        self: CURELiteRecoverableFactorizedDecoder,
        config: RecoverableFactorizedDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        assert isinstance(config, RecoverableFactorizedDecoderConfig)
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
        CURELiteRecoverableFactorizedDecoder,
        "__init__",
        counted_init,
    )
    config = RecoverableFactorizedDecoderConfig(
        feature_channels=3,
        feature_stride=2,
    )
    result = execute_recoverable_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        config,
        LossConfig(),
        _recoverable_budget(),
        device="cpu",
        evaluation_chunk_size=FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    )

    assert construction_calls == [config]
    assert verify_calls == 2
    assert (
        result["schema_version"]
        == RECOVERABLE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
    )
    assert result["method_id"] == RECOVERABLE_FACTORIZED_OUTCOME_METHOD_ID
    assert result["execution_status"] == "completed"
    assert result["optimizer_updates_completed"] == 400
    assert result["structural_execution_pass"] is True
    assert all(result["structural_checks"].values())
    assert result["pretraining_structural_audit"]["all_pass"] is True
    operator_audit = result["pretraining_structural_audit"][
        "operator_contract"
    ]
    assert tuple(operator_audit["checks"]) == (
        RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS
    )
    assert operator_audit["all_pass"] is True
    assert operator_audit["autograd_backward_calls"] == 1
    assert operator_audit["D_R_accessed"] is False
    assert all(
        result["structural_checks"][name] is True
        for name in RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS
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
    assert result["interpretation"]["D_V_accessed"] is False
    assert result["interpretation"]["D_T_accessed"] is False
    assert result["interpretation"]["eligible_for_frozen_review"] is (
        result["computational_model_code_gate_pass"]
    )
    assert sum(
        row["pair_count"]
        for row in result["computational_gates"][
            "tiny_target_strata"
        ].values()
    ) == 206

    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (
            DirectedFactorizedDecoderConfig(3, 2),
            "RecoverableFactorizedDecoderConfig",
        ),
        (object(), "RecoverableFactorizedDecoderConfig"),
    ),
)
def test_recoverable_bounded_rejects_non_v6_config_types(
    config: object,
    message: str,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    with pytest.raises(TypeError, match=message):
        execute_recoverable_factorized_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            config,  # type: ignore[arg-type]
            LossConfig(),
            _recoverable_budget(),
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
def test_recoverable_bounded_rejects_any_frozen_budget_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    budget = _recoverable_budget()
    budget[field] = value
    with pytest.raises(ValueError, match=message):
        execute_recoverable_factorized_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            RecoverableFactorizedDecoderConfig(3, 2),
            LossConfig(),
            budget,
            device="cpu",
        )


def test_recoverable_bounded_rejects_loss_chunk_and_stride_drift() -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    common = (
        population,
        factual_schedule,
        schedule,
        materializer,
    )
    with pytest.raises(ValueError, match="LossConfig"):
        execute_recoverable_factorized_outcome_bounded(
            *common,
            RecoverableFactorizedDecoderConfig(3, 2),
            LossConfig(dice_weight=0.5),
            _recoverable_budget(),
            device="cpu",
        )
    with pytest.raises(ValueError, match="evaluation_chunk_size"):
        execute_recoverable_factorized_outcome_bounded(
            *common,
            RecoverableFactorizedDecoderConfig(3, 2),
            LossConfig(),
            _recoverable_budget(),
            device="cpu",
            evaluation_chunk_size=64,
        )
    with pytest.raises(ValueError, match="native subpixel path"):
        execute_recoverable_factorized_outcome_bounded(
            *common,
            RecoverableFactorizedDecoderConfig(3, 1),
            LossConfig(),
            _recoverable_budget(),
            device="cpu",
        )


def test_recoverable_structural_failure_is_fingerprinted_zero_training_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    failed_audit = {
        "scope": "test-recoverable-structural-audit",
        "all_pass": False,
        "checks": {
            "all_audited_fields_finite": False,
            "all_clean_pairs_have_nonempty_H": True,
        },
        "compute_budget": {
            "decoder_calls": 28,
            "decoder_state_evaluations": 888,
        },
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    monkeypatch.setattr(
        recoverable_module,
        "audit_factorized_outcome_population",
        lambda *args, **kwargs: dict(failed_audit),
    )

    def forbidden_train(*args, **kwargs):
        raise AssertionError("training must not run after structural failure")

    monkeypatch.setattr(
        recoverable_module,
        "outcome_complete_train_step",
        forbidden_train,
    )
    result = execute_recoverable_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        RecoverableFactorizedDecoderConfig(3, 2),
        LossConfig(),
        _recoverable_budget(),
        device="cpu",
    )

    assert result["method_id"] == RECOVERABLE_FACTORIZED_OUTCOME_METHOD_ID
    assert result["decision"] == "PR_SVEF_STRUCTURAL_EXECUTION_FAIL"
    assert result["structural_execution_pass"] is False
    assert result["computational_model_code_gate_pass"] is False
    assert result["optimizer_updates_completed"] == 0
    assert result["training_performed"] is False
    composed = result["pretraining_structural_audit"]
    assert composed["population_audit_scope"] == failed_audit["scope"]
    assert composed["all_pass"] is False
    assert composed["compute_budget"] == failed_audit["compute_budget"]
    assert tuple(composed["operator_contract"]["checks"]) == (
        RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS
    )
    assert composed["operator_contract"]["all_pass"] is True
    assert all(
        name in result["structural_checks"]
        for name in RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS
    )
    assert result["computational_gates"] == {
        "status": "NOT_EVALUATED_BY_STRUCTURAL_STOP_RULE",
        "all_pass": None,
    }
    assert result["interpretation"]["eligible_for_frozen_review"] is False
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)


def test_recoverable_operator_audit_failure_triggers_pretraining_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    checks = {
        name: name != "v6_zero_boundary_gradient_equals_half"
        for name in RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS
    }
    failed_operator_audit = {
        "scope": "test-v6-operator-audit",
        "checks": checks,
        "all_pass": False,
        "autograd_backward_calls": 1,
        "training_performed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    monkeypatch.setattr(
        recoverable_module,
        "_audit_recoverable_operator_contract",
        lambda **kwargs: dict(failed_operator_audit),
    )

    def forbidden_train(*args, **kwargs):
        raise AssertionError("operator failure must stop before training")

    monkeypatch.setattr(
        recoverable_module,
        "outcome_complete_train_step",
        forbidden_train,
    )
    result = execute_recoverable_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        RecoverableFactorizedDecoderConfig(3, 2),
        LossConfig(),
        _recoverable_budget(),
        device="cpu",
    )

    assert result["decision"] == "PR_SVEF_STRUCTURAL_EXECUTION_FAIL"
    assert result["optimizer_updates_completed"] == 0
    assert result["training_performed"] is False
    assert result["structural_execution_pass"] is False
    assert (
        result["structural_checks"][
            "v6_zero_boundary_gradient_equals_half"
        ]
        is False
    )
    assert result["pretraining_structural_audit"][
        "operator_contract"
    ] == failed_operator_audit
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)
