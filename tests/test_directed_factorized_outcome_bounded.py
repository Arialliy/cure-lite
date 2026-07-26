from __future__ import annotations

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.config import LossConfig
from cure_lite.directed_factorized_config import (
    DirectedFactorizedDecoderConfig,
)
from cure_lite.experiment import (
    directed_factorized_outcome_bounded as directed_module,
)
from cure_lite.experiment.directed_factorized_outcome_bounded import (
    DIRECTED_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
    execute_directed_factorized_outcome_bounded,
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
from tests.test_paired_outcome_bounded import _budget, _inputs


def _directed_budget() -> dict[str, object]:
    budget = _budget()
    budget.update(
        {
            "seed": FACTORIZED_FROZEN_SEED,
            "learning_rate": FACTORIZED_FROZEN_LEARNING_RATE,
            "weight_decay": FACTORIZED_FROZEN_WEIGHT_DECAY,
        }
    )
    return budget


def test_directed_bounded_executes_exact_frozen_population_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    verify_calls = 0
    original_verify = PairedOutcomeInputMaterializer.verify_unchanged

    def counted_verify(self: PairedOutcomeInputMaterializer) -> None:
        nonlocal verify_calls
        verify_calls += 1
        original_verify(self)

    monkeypatch.setattr(
        PairedOutcomeInputMaterializer,
        "verify_unchanged",
        counted_verify,
    )
    config = DirectedFactorizedDecoderConfig(
        feature_channels=3,
        feature_stride=2,
    )
    result = execute_directed_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        config,
        LossConfig(),
        _directed_budget(),
        device="cpu",
        evaluation_chunk_size=FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    )

    assert verify_calls == 2
    assert (
        result["schema_version"]
        == DIRECTED_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
    )
    assert result["execution_status"] == "completed"
    assert result["optimizer_updates_completed"] == 400
    assert result["structural_execution_pass"] is True
    assert all(result["structural_checks"].values())
    assert result["pretraining_structural_audit"]["all_pass"] is True
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
    ("field", "value", "message"),
    (
        ("seed", 43, "seed"),
        ("learning_rate", 2.0e-3, "learning_rate"),
        ("weight_decay", 1.0e-4, "weight_decay"),
        ("optimizer_updates", 399, "400 optimizer updates"),
    ),
)
def test_directed_bounded_rejects_any_frozen_budget_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    budget = _directed_budget()
    budget[field] = value
    with pytest.raises(ValueError, match=message):
        execute_directed_factorized_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            DirectedFactorizedDecoderConfig(3, 2),
            LossConfig(),
            budget,
            device="cpu",
        )


def test_directed_bounded_rejects_loss_chunk_and_stride_drift() -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    common = (
        population,
        factual_schedule,
        schedule,
        materializer,
    )
    with pytest.raises(ValueError, match="LossConfig"):
        execute_directed_factorized_outcome_bounded(
            *common,
            DirectedFactorizedDecoderConfig(3, 2),
            LossConfig(dice_weight=0.5),
            _directed_budget(),
            device="cpu",
        )
    with pytest.raises(ValueError, match="evaluation_chunk_size"):
        execute_directed_factorized_outcome_bounded(
            *common,
            DirectedFactorizedDecoderConfig(3, 2),
            LossConfig(),
            _directed_budget(),
            device="cpu",
            evaluation_chunk_size=64,
        )
    with pytest.raises(ValueError, match="native subpixel path"):
        execute_directed_factorized_outcome_bounded(
            *common,
            DirectedFactorizedDecoderConfig(3, 1),
            LossConfig(),
            _directed_budget(),
            device="cpu",
        )


def test_directed_structural_failure_is_auditable_zero_training_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    failed_audit = {
        "scope": "test-directed-structural-audit",
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
        directed_module,
        "audit_factorized_outcome_population",
        lambda *args, **kwargs: dict(failed_audit),
    )

    def forbidden_train(*args, **kwargs):
        raise AssertionError("training must not run after structural failure")

    monkeypatch.setattr(
        directed_module,
        "outcome_complete_train_step",
        forbidden_train,
    )
    result = execute_directed_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        DirectedFactorizedDecoderConfig(3, 2),
        LossConfig(),
        _directed_budget(),
        device="cpu",
    )

    assert result["decision"] == "D_SVEF_STRUCTURAL_EXECUTION_FAIL"
    assert result["structural_execution_pass"] is False
    assert result["computational_model_code_gate_pass"] is False
    assert result["optimizer_updates_completed"] == 0
    assert result["training_performed"] is False
    assert result["pretraining_structural_audit"] == failed_audit
    assert result["computational_gates"] == {
        "status": "NOT_EVALUATED_BY_STRUCTURAL_STOP_RULE",
        "all_pass": None,
    }
    assert result["interpretation"]["eligible_for_frozen_review"] is False
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)
