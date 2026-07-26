from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.ccfr_development_inputs import (
    CONSERVATIVE_TOY_CASES,
)
from tools import evaluate_ccfr_development_regression as dev


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conditioned_feature_release_v11"
)
CONFIG = PROTOCOL / "development_regression_config.json"
EVALUATOR = ROOT / "tools" / "evaluate_ccfr_development_regression.py"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_config_is_frozen_before_learning_regression() -> None:
    config = _load(CONFIG)
    unsigned = dict(config)
    fingerprint = unsigned.pop("config_fingerprint")

    assert stable_fingerprint(unsigned) == fingerprint
    assert config["method_id"] == "ccfr_v11"
    assert config["stage_id"] == "dataset_free_development_regression"
    assert config["status"] == "FROZEN_BEFORE_SINGLE_DEVELOPMENT_RUN"
    assert config["cases"] == [
        {
            "family_id": family_id,
            "case_id": case_id,
            "clean_pixels": [list(pixel) for pixel in pixels],
        }
        for family_id, case_id, pixels in CONSERVATIVE_TOY_CASES
    ]
    assert config["optimization"] == {
        "seed": 7817,
        "optimizer": "adam",
        "updates_per_case": 320,
        "learning_rate": 0.004,
        "weight_decay": 0.0,
        "device": "cpu",
        "torch_threads": 2,
        "automatic_retry_allowed": False,
    }
    assert config["decision_rule"][
        "required_passed_case_count"
    ] == 6
    assert config["decision_rule"][
        "mean_cannot_override_case_failure"
    ] is True
    assert all(
        value is False
        for value in config["execution_boundary"].values()
    )
    binding = dev._load_protocol()
    assert binding["config_fingerprint"] == fingerprint


def test_evaluator_is_dataset_free_and_defaults_to_frozen_320_updates() -> None:
    modules: set[str] = set()
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    assert not any(
        module == "datasets" or module.startswith("datasets.")
        for module in modules
    )
    assert not any("stage_a" in module for module in modules)
    assert "cure_lite.coverage_feature_release_decoder" in modules
    assert "cure_lite.ccfr_development_inputs" in modules
    assert "cure_lite.experiment.conservative_toy_inputs" not in modules
    assert "cure_lite.paired_endpoint_crossing_losses" in modules
    assert "cure_lite.train.paired_outcome_step" in modules
    assert inspect.signature(dev._case).parameters[
        "updates"
    ].default == 320

    command = """
import json
import sys
import tools.evaluate_ccfr_development_regression as evaluator

report = evaluator._runtime_import_boundary()
source_report = evaluator._runtime_source_closure(
    {path: '0' * 64 for path in evaluator.REQUIRED_SOURCE_BINDINGS}
)
print(json.dumps({
    'report': report,
    'source_report': source_report,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    isolated = json.loads(completed.stdout)
    assert isolated["report"]["observed_forbidden_modules"] == []
    assert isolated["report"]["all_pass"] is True
    assert isolated["source_report"]["unbound_local_imports"] == []
    assert isolated["source_report"]["all_pass"] is True


def test_one_update_smoke_has_joint_state_2b_detach_and_six_gradients() -> None:
    result = dev._case(*CONSERVATIVE_TOY_CASES[0], updates=1)

    assert result["updates_executed"] == 1
    assert result["endpoint_gradient"] == {
        "plus_finite_nonzero": True,
        "minus_finite_nonzero": True,
    }
    assert result["gradient_contract"]["parameter_tensors"] == 6
    assert result["gradient_contract"]["parameters"] == 2593
    assert result["gradient_contract"]["failure_count"] == 0
    assert result["gradient_contract"]["minimum_l2_norm"] > 0.0
    assert result["forward_contract"][
        "all_updates_exact_three_4_state_calls"
    ] is True
    assert result["initial_operator_audit"]["all_pass"] is True
    assert result["operator_audit"]["all_pass"] is True
    assert result["operator_audit"]["checks"][
        "joint_latent_state_changes"
    ] is True
    assert result["operator_audit"]["checks"][
        "zero_feature_occupancy_control_exact"
    ] is True
    assert result["forward_contract"][
        "step_log_contract_failure_count"
    ] == 0
    assert all(count > 0 for count in result["stratum_counts"].values())
    assert result["metrics"]["total_loss"] == pytest.approx(sum(
        result["metrics"][name]
        for name in (
            "final_factual_miss_loss",
            "final_factual_no_miss_loss",
            "final_pair_loss",
        )
    ), abs=1.0e-6)


def test_objective_audit_calls_actual_criterion_and_checks_directions() -> None:
    audit = dev._objective_contract_audit()

    assert audit["all_pass"] is True
    assert audit["actual_criterion_response_formula_max_abs_error"] == 0.0
    assert audit["actual_response_pixel_count"] > 0
    assert audit["actual_zero_response_pixel_count"] > 0
    assert audit["checks"][
        "actual_criterion_D_plus_gradient_positive"
    ] is True
    assert audit["checks"][
        "actual_criterion_D_minus_gradient_negative"
    ] is True


def test_decision_requires_every_case_and_never_claims_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dev,
        "_load_protocol",
        lambda: {"sentinel": True, "source_bindings": {}},
    )
    monkeypatch.setattr(
        dev,
        "_runtime_import_boundary",
        lambda: {"observed_forbidden_modules": [], "all_pass": True},
    )
    monkeypatch.setattr(
        dev,
        "_runtime_source_closure",
        lambda _bindings: {"unbound_local_imports": [], "all_pass": True},
    )

    def passing_case(*_args: object) -> dict[str, object]:
        return {
            "family_id": str(_args[0]),
            "all_pass": True,
            "optimizer_contract": {"sentinel": True},
        }

    monkeypatch.setattr(dev, "_case", passing_case)
    result = dev.evaluate()

    assert result["all_pass"] is True
    assert result["passed_case_count"] == 6
    assert result["passed_family_count"] == 2
    assert result["decision"] == (
        "CCFR_V11_DEVELOPMENT_REGRESSION_PASS"
    )
    assert result["execution_boundary"][
        "independent_confirmation_established"
    ] is False
    assert result["execution_boundary"]["real_bounded_authorized"] is False
    assert result["attempt_binding"] == {
        "execution_mode": "IN_MEMORY_TEST_ONLY",
        "canonical_attempt_consumed": False,
    }
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert stable_fingerprint(unsigned) == fingerprint


def test_one_failed_case_cannot_be_hidden_by_family_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dev,
        "_load_protocol",
        lambda: {"sentinel": True, "source_bindings": {}},
    )
    monkeypatch.setattr(
        dev,
        "_runtime_import_boundary",
        lambda: {"observed_forbidden_modules": [], "all_pass": True},
    )
    monkeypatch.setattr(
        dev,
        "_runtime_source_closure",
        lambda _bindings: {"unbound_local_imports": [], "all_pass": True},
    )
    call = 0

    def one_failure(*_args: object) -> dict[str, object]:
        nonlocal call
        call += 1
        return {
            "family_id": str(_args[0]),
            "all_pass": call != 1,
            "optimizer_contract": {"sentinel": True},
        }

    monkeypatch.setattr(dev, "_case", one_failure)
    result = dev.evaluate()

    assert result["all_pass"] is False
    assert result["passed_case_count"] == 5
    assert result["passed_family_count"] == 1
    assert result["decision"] == (
        "CCFR_V11_DEVELOPMENT_REGRESSION_FAIL"
    )


def test_result_writer_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "ccfr-development.json"
    dev._write_result(output, {"sentinel": True})
    with pytest.raises(FileExistsError):
        dev._write_result(output, {"sentinel": False})
    assert _load(output) == {"sentinel": True}


def test_attempt_receipt_and_complete_writer_are_create_only(
    tmp_path: Path,
) -> None:
    attempt = dev._attempt_payload({"sentinel": True})
    unsigned = dict(attempt)
    fingerprint = unsigned.pop("attempt_fingerprint")
    assert stable_fingerprint(unsigned) == fingerprint
    assert attempt["automatic_retry_allowed"] is False

    complete = tmp_path / "result.COMPLETE.sha256"
    dev._write_text_create_only(complete, "abc  result.json\n")
    with pytest.raises(FileExistsError):
        dev._write_text_create_only(complete, "changed\n")
    assert complete.read_text(encoding="utf-8") == "abc  result.json\n"


def test_canonical_attempt_is_loaded_and_bound_to_its_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_path = tmp_path / "development-attempt.json"
    monkeypatch.setattr(dev, "_ROOT", tmp_path)
    monkeypatch.setattr(dev, "_CANONICAL_ATTEMPT", attempt_path)
    monkeypatch.setattr(
        dev,
        "_CANONICAL_RESULT",
        tmp_path / "development-result.json",
    )
    monkeypatch.setattr(
        dev,
        "_CANONICAL_COMPLETE",
        tmp_path / "development-result.COMPLETE.sha256",
    )
    protocol = {"config_fingerprint": "a" * 64}
    payload = dev._attempt_payload(protocol)
    dev._write_result(attempt_path, payload)

    binding = dev._load_attempt(protocol)

    assert binding["repo_path"].endswith("development-attempt.json")
    assert binding["file_sha256"] == dev.file_sha256(attempt_path)
    assert binding["attempt_fingerprint"] == payload[
        "attempt_fingerprint"
    ]


def test_cli_rejects_noncanonical_output_before_execution(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="canonical r1 path"):
        dev.main(["--output", str(tmp_path / "alternate.json")])


def test_exact_required_source_binding_set_is_frozen() -> None:
    config = _load(CONFIG)
    assert set(config["source_bindings"]) == set(
        dev.REQUIRED_SOURCE_BINDINGS
    )
    holdout = dev._load_holdout_receipt()
    assert holdout["status"] == (
        "FROZEN_BEFORE_DEVELOPMENT_REGRESSION_RESULT"
    )


def test_bound_source_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = dev.file_sha256
    target = ROOT / "cure_lite" / "coverage_feature_release_decoder.py"

    def changed(path: Path) -> str:
        if Path(path) == target:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(dev, "file_sha256", changed)
    with pytest.raises(RuntimeError, match="bound source differs"):
        dev._load_protocol()
