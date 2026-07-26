from __future__ import annotations

from copy import deepcopy

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.config import LossConfig
from cure_lite.conservative_factorized_config import (
    ConservativeFactorizedDecoderConfig,
)
from cure_lite.experiment.conservative_factorized_outcome_bounded import (
    execute_conservative_factorized_outcome_bounded,
)
from cure_lite.experiment.conservative_factorized_result_verifier import (
    ConservativeBoundedResultContract,
    verify_computational_gate_records,
    verify_conservative_factorized_core_result,
)
from cure_lite.experiment.factorized_outcome_bounded import (
    FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    FACTORIZED_FROZEN_LEARNING_RATE,
    FACTORIZED_FROZEN_SEED,
    FACTORIZED_FROZEN_WEIGHT_DECAY,
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


def _resign(result: dict[str, object]) -> dict[str, object]:
    result.pop("result_fingerprint", None)
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


@pytest.fixture(scope="module")
def real_core_result_and_contract() -> tuple[
    dict[str, object],
    ConservativeBoundedResultContract,
]:
    """Run the actual 400-update core entirely on synthetic in-memory inputs."""

    population, factual_schedule, schedule, materializer = _inputs()
    config = ConservativeFactorizedDecoderConfig(3, 2)
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
    contract = ConservativeBoundedResultContract(
        device="cpu",
        population_fingerprint=population.population_fingerprint,
        materializer_fingerprint=materializer.materializer_fingerprint,
        factual_schedule_fingerprint=factual_schedule.schedule_fingerprint,
        outcome_schedule_fingerprint=schedule.schedule_fingerprint,
        feature_channels=3,
        feature_stride=2,
        trainable_parameter_count=config.expected_parameter_count,
        trainable_parameter_tensors=6,
    )
    return result, contract


def _set_gate_outcome(
    original: dict[str, object],
    *,
    all_pass: bool,
) -> dict[str, object]:
    result = deepcopy(original)
    computational = result["computational_gates"]
    assert isinstance(computational, dict)
    checks = computational["checks"]
    observed = computational["observed"]
    assert isinstance(checks, dict)
    assert isinstance(observed, dict)
    for record in checks.values():
        assert isinstance(record, dict)
        threshold = float(record["threshold"])
        if record["direction"] == "min":
            record["value"] = threshold + 1.0
        else:
            record["value"] = threshold - 1.0
        record["pass"] = True
    if not all_pass:
        name = next(iter(checks))
        record = checks[name]
        assert isinstance(record, dict)
        threshold = float(record["threshold"])
        if record["direction"] == "min":
            record["value"] = threshold - 1.0
        else:
            record["value"] = threshold + 1.0
        record["pass"] = False
    for name, record in checks.items():
        assert isinstance(record, dict)
        observed[name] = record["value"]
    computational["all_pass"] = all_pass
    structural = result["structural_execution_pass"] is True
    model_pass = structural and all_pass
    result["computational_model_code_gate_pass"] = model_pass
    result["decision"] = (
        "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_PASS"
        if model_pass
        else (
            "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_FAIL"
            if structural
            else "CC_SEA_V8_STRUCTURAL_EXECUTION_FAIL"
        )
    )
    interpretation = result["interpretation"]
    assert isinstance(interpretation, dict)
    interpretation["eligible_for_frozen_review"] = model_pass
    return _resign(result)


def test_actual_core_result_flows_into_actual_corrected_verifier(
    real_core_result_and_contract: tuple[
        dict[str, object],
        ConservativeBoundedResultContract,
    ],
) -> None:
    result, contract = real_core_result_and_contract
    verify_conservative_factorized_core_result(result, contract=contract)


@pytest.mark.parametrize("all_pass", (True, False))
def test_corrected_verifier_accepts_consistent_pass_and_nonpass(
    real_core_result_and_contract: tuple[
        dict[str, object],
        ConservativeBoundedResultContract,
    ],
    all_pass: bool,
) -> None:
    original, contract = real_core_result_and_contract
    result = _set_gate_outcome(original, all_pass=all_pass)
    verify_conservative_factorized_core_result(result, contract=contract)
    assert result["computational_model_code_gate_pass"] is all_pass


def test_model_pass_is_structural_and_computational(
    real_core_result_and_contract: tuple[
        dict[str, object],
        ConservativeBoundedResultContract,
    ],
) -> None:
    original, contract = real_core_result_and_contract
    result = _set_gate_outcome(original, all_pass=True)
    structural_checks = result["structural_checks"]
    assert isinstance(structural_checks, dict)
    structural_checks[next(iter(structural_checks))] = False
    result["structural_execution_pass"] = False
    result["computational_model_code_gate_pass"] = False
    result["decision"] = "CC_SEA_V8_STRUCTURAL_EXECUTION_FAIL"
    interpretation = result["interpretation"]
    assert isinstance(interpretation, dict)
    interpretation["eligible_for_frozen_review"] = False
    _resign(result)

    verify_conservative_factorized_core_result(result, contract=contract)
    computational = result["computational_gates"]
    assert isinstance(computational, dict)
    assert computational["all_pass"] is True


def test_malformed_gate_record_is_rejected(
    real_core_result_and_contract: tuple[
        dict[str, object],
        ConservativeBoundedResultContract,
    ],
) -> None:
    original, contract = real_core_result_and_contract
    result = _set_gate_outcome(original, all_pass=True)
    computational = result["computational_gates"]
    assert isinstance(computational, dict)
    checks = computational["checks"]
    assert isinstance(checks, dict)
    first = checks[next(iter(checks))]
    assert isinstance(first, dict)
    first.pop("pass")
    _resign(result)

    with pytest.raises(RuntimeError, match="gate record changed"):
        verify_conservative_factorized_core_result(
            result,
            contract=contract,
        )


def test_gate_record_aggregator_rejects_old_boolean_assumption() -> None:
    with pytest.raises(RuntimeError, match="gate schema changed"):
        verify_computational_gate_records(
            {
                "scope": (
                    "bounded_D_R_full_outcome_CC_SEA_v8_model_code_gate"
                ),
                "not_detection_performance": True,
                "thresholds_unchanged_from_v4_v6_v7": True,
                "thresholds": {},
                "observed": {},
                "checks": {f"gate_{index}": True for index in range(12)},
                "all_pass": True,
            },
            structural_execution_pass=True,
            model_code_gate_pass=True,
        )
