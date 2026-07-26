from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from tools.evaluate_conservative_factorized_toy_gate import (
    CONSERVATIVE_TOY_CASES,
    _write_result,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools" / "evaluate_conservative_factorized_toy_gate.py"
PROTOCOL = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conserving_subpixel_evidence_allocation_v8"
)
EXPECTED_PROPOSAL_SHA256 = (
    "4590a681990a5332de233262510e0918f1d08d7b01ea6ad5e3c4ed7b8749c9bc"
)
EXPECTED_PROPOSAL_FINGERPRINT = (
    "14bb96e03598a613c5c201e891d7c5b690f8cc38dbf83d380aa3dbc17e82370b"
)
EXPECTED_CONFIG_SHA256 = (
    "979fedfef36c7fe77b6066d8d4cea99ec216f839412eeb1a38b6c96f55e83ba4"
)
EXPECTED_CONFIG_FINGERPRINT = (
    "3de5142832fc9b3dcd7a87832daf545588011a35a3388d7fefaaac8341898f29"
)


@pytest.fixture(scope="session")
def toy_replays(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path, Path]:
    directory = tmp_path_factory.mktemp("cc-sea-v8-toy-replays")
    first = directory / "r1.json"
    second = directory / "r2.json"
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for output in (first, second):
        completed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
    assert first.read_bytes() == second.read_bytes()
    value = json.loads(first.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value, first, second


def test_protocol_chain_is_frozen_and_exact(
    toy_replays: tuple[dict[str, object], Path, Path],
) -> None:
    result, first, second = toy_replays
    proposal = PROTOCOL / "proposal_receipt.json"
    config = PROTOCOL / "toy_config.json"

    assert file_sha256(proposal) == EXPECTED_PROPOSAL_SHA256
    assert file_sha256(config) == EXPECTED_CONFIG_SHA256
    assert (
        result["protocol_binding"]["proposal_fingerprint"]
        == EXPECTED_PROPOSAL_FINGERPRINT
    )
    assert (
        result["protocol_binding"]["toy_config_fingerprint"]
        == EXPECTED_CONFIG_FINGERPRINT
    )
    assert file_sha256(first) == file_sha256(second)


def test_six_cases_pass_individually_without_mean_override(
    toy_replays: tuple[dict[str, object], Path, Path],
) -> None:
    result, _, _ = toy_replays

    assert result["decision"] == "CC_SEA_V8_TOY_GATE_PASS"
    assert result["all_pass"] is True
    assert result["passed_case_count"] == 6
    assert result["failed_case_count"] == 0
    assert len(result["cases"]) == len(CONSERVATIVE_TOY_CASES)
    assert [case["case_id"] for case in result["cases"]] == [
        case[1] for case in CONSERVATIVE_TOY_CASES
    ]
    for case in result["cases"]:
        assert case["all_pass"] is True
        assert all(case["checks"].values())
        assert case["initial_operator_audit"]["all_pass"] is True
        assert case["operator_audit"]["all_pass"] is True
        assert case["gradient_contract"]["failure_count"] == 0
        assert case["gradient_contract"]["updates_checked"] == 320
        assert case["gradient_contract"]["minimum_l2_norm"] > 0.0
        assert case["endpoint_gradient"] == {
            "plus_finite_nonzero": True,
            "minus_finite_nonzero": True,
        }
        assert case["forward_contract"] == {
            "first_call_batch_size": 4,
            "paired_batch_size": 2,
            "endpoint_state_count": 4,
            "uses_one_2B_endpoint_forward": True,
            "training_step_decoder_calls": 3,
            "training_step_decoder_states": 12,
        }


def test_cc_sea_coordinate_and_numerical_contracts_pass(
    toy_replays: tuple[dict[str, object], Path, Path],
) -> None:
    result, _, _ = toy_replays
    coordinate = result["orthogonality_audit"]
    numerical = result["numerical_contract_audit"]

    assert coordinate["all_pass"] is True
    assert all(coordinate["checks"].values())
    assert coordinate["simplex_max_abs_error"] <= 1.0e-12
    assert coordinate["mass_conservation_max_abs_error"] <= 1.0e-12
    assert (
        coordinate["zero_mean_contrast"]["allocation_max_abs_change"]
        > 1.0e-3
    )
    assert numerical["all_pass"] is True
    assert all(numerical["checks"].values())


def test_result_has_no_real_data_or_performance_authority(
    toy_replays: tuple[dict[str, object], Path, Path],
) -> None:
    result, _, _ = toy_replays
    boundary = result["execution_boundary"]

    assert all(value is False for value in boundary.values())
    assert result["interpretation"] == (
        "dataset_free_model_code_gate_not_detection_performance"
    )
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)


def test_evaluator_is_version_local_and_output_is_create_only(
    tmp_path: Path,
) -> None:
    modules: set[str] = set()
    for path in (
        EVALUATOR,
        ROOT / "cure_lite" / "experiment" / "conservative_toy_inputs.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
    assert not any(module == "tests" or module.startswith("tests.") for module in modules)
    assert not any("evaluate_crossing_factorized_toy_gate" in module for module in modules)
    assert not any("stage_a" in module for module in modules)
    assert not any("dataset" in module for module in modules)

    output = tmp_path / "create-only.json"
    _write_result(output, {"sentinel": True})
    with pytest.raises(FileExistsError):
        _write_result(output, {"sentinel": False})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "sentinel": True
    }
