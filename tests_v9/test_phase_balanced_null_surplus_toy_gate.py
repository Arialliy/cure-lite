from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from tools.evaluate_phase_balanced_null_surplus_toy_gate import (
    CONSERVATIVE_TOY_CASES,
    _write_result,
    evaluate,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = (
    ROOT / "tools" / "evaluate_phase_balanced_null_surplus_toy_gate.py"
)
PROTOCOL = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "phase_balanced_null_anchored_evidence_surplus_v9"
)
EXPECTED_PROPOSAL_SHA256 = (
    "7bb563907f8b037ca2feac5ae356c5b3e3ed52cf945dfdaa2d7fa503131e02c5"
)
EXPECTED_PROPOSAL_FINGERPRINT = (
    "90dd2931d2e8aa5ed76e86165b8d68363d7735e522b0a7cca228cc04269de5e6"
)
EXPECTED_CONFIG_SHA256 = (
    "1089829e88d54b314bb94ba1f536130c37baa5ed2d75a0a8eff7f87d6336fddc"
)
EXPECTED_CONFIG_FINGERPRINT = (
    "8af91f01925c258d1be9d9ec590d7aedabbc17d7ad3028e7f581731fa3925406"
)


@pytest.fixture(scope="session")
def toy_result() -> dict[str, object]:
    return evaluate()


def test_protocol_chain_is_frozen_and_exact(
    toy_result: dict[str, object],
) -> None:
    proposal = PROTOCOL / "proposal_receipt.json"
    config = PROTOCOL / "toy_config.json"

    assert file_sha256(proposal) == EXPECTED_PROPOSAL_SHA256
    assert file_sha256(config) == EXPECTED_CONFIG_SHA256
    assert (
        toy_result["protocol_binding"]["proposal_fingerprint"]
        == EXPECTED_PROPOSAL_FINGERPRINT
    )
    assert (
        toy_result["protocol_binding"]["toy_config_fingerprint"]
        == EXPECTED_CONFIG_FINGERPRINT
    )


def test_frozen_six_case_gate_is_reported_without_mean_override(
    toy_result: dict[str, object],
) -> None:
    assert toy_result["decision"] == "PB_NAES_V9_TOY_GATE_FAIL"
    assert toy_result["all_pass"] is False
    assert toy_result["passed_case_count"] == 0
    assert toy_result["failed_case_count"] == 6
    assert toy_result["passed_family_count"] == 0
    assert len(toy_result["cases"]) == len(CONSERVATIVE_TOY_CASES)
    assert [case["case_id"] for case in toy_result["cases"]] == [
        case[1] for case in CONSERVATIVE_TOY_CASES
    ]
    for case in toy_result["cases"]:
        assert case["all_pass"] is False
        assert case["checks"]["clean_D"] is False


def test_operator_counterexamples_and_numerics_pass(
    toy_result: dict[str, object],
) -> None:
    counterexamples = toy_result["counterexample_audit"]
    numerical = toy_result["numerical_contract_audit"]

    assert counterexamples["all_pass"] is True
    assert all(counterexamples["checks"].values())
    assert counterexamples["formula_max_abs_error"] <= 1.0e-12
    assert counterexamples["multi_phase_active_counts"] == [1, 2, 3]
    assert counterexamples["capacity_max_relative_violation"] <= 1.0e-6
    assert numerical["all_pass"] is True
    assert all(numerical["checks"].values())


def test_every_case_keeps_the_frozen_training_contract(
    toy_result: dict[str, object],
) -> None:
    for case in toy_result["cases"]:
        assert case["initial_operator_audit"]["all_pass"] is True
        assert case["operator_audit"]["all_pass"] is True
        assert case["gradient_contract"]["failure_count"] == 0
        assert case["gradient_contract"]["updates_checked"] == 320
        assert case["gradient_contract"]["parameter_tensors"] == 6
        assert case["gradient_contract"]["parameters"] == 2593
        assert case["gradient_contract"]["minimum_l2_norm"] > 0.0
        assert case["endpoint_gradient"] == {
            "plus_finite_nonzero": True,
            "minus_finite_nonzero": True,
        }
        assert case["feature_detach_contract"] == {
            "input_requires_grad": True,
            "input_gradient_is_none": True,
            "passed": True,
        }
        assert case["forward_contract"] == {
            "first_call_batch_size": 4,
            "paired_batch_size": 2,
            "endpoint_state_count": 4,
            "uses_one_2B_endpoint_forward": True,
            "training_step_decoder_calls": 3,
            "training_step_decoder_states": 12,
        }


def test_result_has_no_real_data_or_performance_authority(
    toy_result: dict[str, object],
) -> None:
    boundary = toy_result["execution_boundary"]

    assert all(value is False for value in boundary.values())
    assert toy_result["interpretation"] == (
        "dataset_free_model_code_gate_not_detection_performance"
    )
    unsigned = dict(toy_result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)


def test_evaluator_is_version_local_and_output_is_create_only(
    tmp_path: Path,
) -> None:
    modules: set[str] = set()
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(
        module == "tests" or module.startswith("tests.")
        for module in modules
    )
    assert not any(
        "evaluate_conservative_factorized_toy_gate" in module
        for module in modules
    )
    assert not any("stage_a" in module for module in modules)
    assert not any("datasets" in module for module in modules)

    output = tmp_path / "create-only.json"
    _write_result(output, {"sentinel": True})
    with pytest.raises(FileExistsError):
        _write_result(output, {"sentinel": False})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "sentinel": True
    }
