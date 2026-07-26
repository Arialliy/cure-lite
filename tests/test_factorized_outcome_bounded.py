from __future__ import annotations

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.config import LossConfig
from cure_lite.experiment import factorized_outcome_bounded as factorized_module
from cure_lite.experiment.factorized_outcome_bounded import (
    FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    FACTORIZED_FROZEN_LEARNING_RATE,
    FACTORIZED_FROZEN_SEED,
    FACTORIZED_FROZEN_WEIGHT_DECAY,
    FACTORIZED_JOINT_D_THRESHOLD,
    FACTORIZED_JOINT_H_THRESHOLD,
    FACTORIZED_JOINT_THRESHOLD,
    FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
    execute_factorized_outcome_bounded,
)
from cure_lite.experiment.paired_outcome_inputs import (
    PairedOutcomeInputMaterializer,
)
from cure_lite.factorized_config import FactorizedDecoderConfig
from tests.test_paired_outcome_bounded import _budget, _inputs


def _factorized_budget() -> dict[str, object]:
    budget = _budget()
    budget.update(
        {
            "seed": FACTORIZED_FROZEN_SEED,
            "learning_rate": FACTORIZED_FROZEN_LEARNING_RATE,
            "weight_decay": FACTORIZED_FROZEN_WEIGHT_DECAY,
        }
    )
    return budget


def test_structural_maximum_abs_is_json_finite_with_nonfinite_inputs() -> None:
    assert factorized_module._maximum_abs(
        torch.tensor([float("nan"), float("inf"), -2.0])
    ) == 2.0
    assert factorized_module._maximum_abs(
        torch.tensor([float("nan"), float("-inf")])
    ) == 0.0


def test_factorized_bounded_executes_frozen_population_and_budget(
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
    config = FactorizedDecoderConfig(
        feature_channels=3,
        feature_stride=2,
    )
    result = execute_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        config,
        LossConfig(),
        _factorized_budget(),
        device="cpu",
        evaluation_chunk_size=FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    )

    assert verify_calls == 2
    assert result["schema_version"] == FACTORIZED_OUTCOME_BOUNDED_SCHEMA
    assert result["execution_status"] == "completed"
    assert result["optimizer_updates_completed"] == 400
    assert result["structural_execution_pass"] is True
    assert all(result["structural_checks"].values())

    audit = result["pretraining_structural_audit"]
    assert audit["pair_count"] == 222
    assert audit["clean_pair_count"] == 206
    assert audit["component_null_pair_count"] == 16
    assert audit["zero_feature_max_abs_occupancy_delta"] == 0.0
    assert audit["outside_gate_max_abs_logit_delta"] == 0.0
    assert audit["outside_gate_max_abs_probability_delta"] == 0.0
    assert audit["nonfinite_audited_field_values"] == 0
    assert audit["vacancy_deletion_monotonicity_violations"] == 0
    assert audit["deletion_logit_monotonicity_violations"] == 0
    assert audit["deletion_probability_monotonicity_violations"] == 0
    assert audit["field_resize_endpoint_count"] == 0
    assert audit["clean_full_D_reachable_pairs"] == 206
    assert audit["clean_nonempty_H_pairs"] == 206
    assert audit["clean_D_reachable_pixels"] == audit["clean_D_total_pixels"]
    assert audit["component_positive_gate_support_pairs"] == 16
    assert audit["factual_full_target_reachable_anchors"] == 16
    assert audit["compute_budget"] == {
        "decoder_calls": 28,
        "decoder_state_evaluations": 888,
        "expected_decoder_calls": 28,
        "expected_decoder_state_evaluations": 888,
        "factual_vacancy_field_calls": 1,
        "factual_vacancy_field_states": 16,
    }
    assert audit["all_pass"] is True
    assert all(audit["checks"].values())

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
    assert result["forward_budget"]["pretraining_structural_audit"] == (
        audit["compute_budget"]
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

    gates = result["computational_gates"]
    joint_name = "clean_joint_D_ge_0_25_and_H_le_0_05_fraction"
    assert gates["checks"][joint_name]["threshold"] == (
        FACTORIZED_JOINT_THRESHOLD
    )
    assert joint_name in gates["observed"]
    assert set(gates["tiny_target_strata"]) == {
        "1_to_3",
        "4_to_7",
        "8_to_15",
        "16_plus",
    }
    assert sum(
        row["pair_count"]
        for row in gates["tiny_target_strata"].values()
    ) == 206

    assert result["parameters"]["trainable_parameter_count"] == (
        config.expected_parameter_count
    )
    assert result["parameters"]["initial_decoder_fingerprint"] != (
        result["parameters"]["final_decoder_fingerprint"]
    )
    assert result["gradients"]["nonfinite_updates"] == 0
    assert result["gradients"]["zero_norm_updates"] == 0
    assert result["interpretation"]["not_detection_performance_evidence"] is True
    assert result["interpretation"]["does_not_authorize_formal_training"] is True
    assert (
        result["interpretation"]["does_not_directly_authorize_formal_800"]
        is True
    )
    assert result["interpretation"]["eligible_for_frozen_review"] is (
        result["computational_model_code_gate_pass"]
    )
    assert result["interpretation"]["D_V_accessed"] is False
    assert result["interpretation"]["D_T_accessed"] is False

    payload = dict(result)
    fingerprint = payload.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(payload)


def test_factorized_bounded_rejects_non_native_stride() -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    with pytest.raises(ValueError, match="native subpixel path"):
        execute_factorized_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            FactorizedDecoderConfig(
                feature_channels=3,
                feature_stride=1,
            ),
            LossConfig(),
            _factorized_budget(),
            device="cpu",
            evaluation_chunk_size=FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
        )


def test_factorized_bounded_rejects_budget_drift_before_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()

    def fail_verify(self: PairedOutcomeInputMaterializer) -> None:
        del self
        raise AssertionError("verification must not run after budget rejection")

    monkeypatch.setattr(
        PairedOutcomeInputMaterializer,
        "verify_unchanged",
        fail_verify,
    )
    changed = _factorized_budget()
    changed["optimizer_updates"] = 399
    with pytest.raises(ValueError, match="fixes 400 optimizer updates"):
        execute_factorized_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            FactorizedDecoderConfig(
                feature_channels=3,
                feature_stride=2,
            ),
            LossConfig(),
            changed,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("seed", 43, "seed"),
        ("learning_rate", 2.0e-3, "learning_rate"),
        ("weight_decay", 1.0e-4, "weight_decay"),
    ),
)
def test_factorized_bounded_rejects_frozen_optimizer_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()

    def fail_verify(self: PairedOutcomeInputMaterializer) -> None:
        del self
        raise AssertionError("verification must not run after drift rejection")

    monkeypatch.setattr(
        PairedOutcomeInputMaterializer,
        "verify_unchanged",
        fail_verify,
    )
    changed = _factorized_budget()
    changed[field] = value
    with pytest.raises(ValueError, match=message):
        execute_factorized_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            FactorizedDecoderConfig(
                feature_channels=3,
                feature_stride=2,
            ),
            LossConfig(),
            changed,
            device="cpu",
        )


def test_factorized_bounded_rejects_loss_and_chunk_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()

    def fail_verify(self: PairedOutcomeInputMaterializer) -> None:
        del self
        raise AssertionError("verification must not run after drift rejection")

    monkeypatch.setattr(
        PairedOutcomeInputMaterializer,
        "verify_unchanged",
        fail_verify,
    )
    common = (
        population,
        factual_schedule,
        schedule,
        materializer,
        FactorizedDecoderConfig(
            feature_channels=3,
            feature_stride=2,
        ),
    )
    with pytest.raises(ValueError, match="LossConfig"):
        execute_factorized_outcome_bounded(
            *common,
            LossConfig(dice_weight=0.5),
            _factorized_budget(),
            device="cpu",
        )
    with pytest.raises(ValueError, match="evaluation_chunk_size"):
        execute_factorized_outcome_bounded(
            *common,
            LossConfig(),
            _factorized_budget(),
            device="cpu",
            evaluation_chunk_size=64,
        )


def test_factorized_structural_failure_returns_auditable_stop_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    failed_audit = {
        "scope": "test-structural-audit",
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
        factorized_module,
        "audit_factorized_outcome_population",
        lambda *args, **kwargs: dict(failed_audit),
    )

    def forbidden_train(*args, **kwargs):
        raise AssertionError("training must not run after structural failure")

    monkeypatch.setattr(
        factorized_module,
        "outcome_complete_train_step",
        forbidden_train,
    )
    result = execute_factorized_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        FactorizedDecoderConfig(
            feature_channels=3,
            feature_stride=2,
        ),
        LossConfig(),
        _factorized_budget(),
        device="cpu",
    )
    assert result["decision"] == "SVEF_STRUCTURAL_EXECUTION_FAIL"
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


def _joint_gate_records() -> list[dict[str, object]]:
    boundary_sizes = [1, 3, 4, 7, 8, 15, 16, 20]
    sizes = boundary_sizes + [1] * (206 - len(boundary_sizes))
    clean = [
        {
            "pair_id": f"clean-{index:03d}",
            "pair_kind": "clean_positive",
            "D_pixels": size,
            "H_pixels": 1,
            "D_mean_delta": FACTORIZED_JOINT_D_THRESHOLD,
            "H_mean_abs_delta": FACTORIZED_JOINT_H_THRESHOLD,
        }
        for index, size in enumerate(sizes)
    ]
    component = [
        {
            "pair_id": f"component-{index:03d}",
            "pair_kind": "component_null",
        }
        for index in range(16)
    ]
    return clean + component


def test_factorized_joint_gate_includes_threshold_and_tiny_bin_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factorized_module,
        "_computational_gates",
        lambda initial, final: {
            "checks": {"base": {"pass": True}},
            "observed": {},
            "thresholds": {},
        },
    )
    gates = factorized_module.factorized_computational_gates(
        {},
        {"outcome_population": {"per_pair": _joint_gate_records()}},
    )
    joint_name = "clean_joint_D_ge_0_25_and_H_le_0_05_fraction"
    assert gates["observed"][joint_name] == 1.0
    assert gates["checks"][joint_name]["pass"] is True
    assert {
        name: row["pair_count"]
        for name, row in gates["tiny_target_strata"].items()
    } == {
        "1_to_3": 200,
        "4_to_7": 2,
        "8_to_15": 2,
        "16_plus": 2,
    }


def test_factorized_joint_gate_rejects_empty_clean_H(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factorized_module,
        "_computational_gates",
        lambda initial, final: {
            "checks": {"base": {"pass": True}},
            "observed": {},
            "thresholds": {},
        },
    )
    records = _joint_gate_records()
    records[0]["H_pixels"] = 0
    records[0]["H_mean_abs_delta"] = None
    with pytest.raises(RuntimeError, match="non-empty H"):
        factorized_module.factorized_computational_gates(
            {},
            {"outcome_population": {"per_pair": records}},
        )
