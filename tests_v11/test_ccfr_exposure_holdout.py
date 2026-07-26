from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_feature_release_decoder import (
    CURELiteCoverageFeatureReleaseDecoder,
)
from cure_lite.conservative_factorized_decoder import (
    CURELiteConservativeFactorizedDecoder,
)
from cure_lite.ccfr_holdout_inputs import (
    UPDATE_COUNT,
    build_ccfr_holdout_pair_specs,
    build_ccfr_holdout_schedule,
)
from tools import evaluate_ccfr_exposure_holdout as holdout


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conditioned_feature_release_v11"
    / "exposure_holdout_design_receipt.json"
)
EVALUATOR = ROOT / "tools" / "evaluate_ccfr_exposure_holdout.py"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_evaluator_uses_only_new_holdout_inputs_and_is_dataset_free() -> None:
    modules: set[str] = set()
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    assert "cure_lite.ccfr_holdout_inputs" in modules
    assert "cure_lite.experiment.peco_exposure_confirmation" not in modules
    assert "cure_lite.experiment.conservative_toy_inputs" not in modules
    assert not any(
        module == "datasets" or module.startswith("datasets.")
        for module in modules
    )
    assert not any("stage_a" in module.lower() for module in modules)
    assert not any("training_pipeline" in module for module in modules)

    command = """
import json
import sys
import tools.evaluate_ccfr_exposure_holdout as evaluator

forbidden_exact = {
    'cure_lite.experiment.cache_pipeline',
    'cure_lite.experiment.stage_a_runner',
    'cure_lite.experiment.stage_a_m_runner',
    'cure_lite.experiment.training_pipeline',
}
report = evaluator._runtime_import_boundary()
source_report = evaluator._runtime_source_closure(
    {path: '0' * 64 for path in evaluator._SOURCE_PATHS}
)
observed = sorted(
    name for name in sys.modules
    if name in forbidden_exact or name == 'datasets' or name.startswith('datasets.')
)
print(json.dumps({
    'report': report,
    'source_report': source_report,
    'observed': observed,
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
    assert isolated["observed"] == []
    assert isolated["report"]["observed_forbidden_modules"] == []
    assert isolated["report"]["all_pass"] is True
    assert isolated["source_report"]["unbound_local_imports"] == []
    assert isolated["source_report"]["all_pass"] is True


def test_design_contract_is_exact_before_fingerprint_freeze() -> None:
    receipt = _load(RECEIPT)
    holdout._validate_design_contract(receipt)
    assert receipt["population"]["groups"] == holdout._EXPECTED_GROUPS
    assert receipt["thresholds"] == holdout.THRESHOLDS
    assert receipt["optimization"]["updates"] == UPDATE_COUNT == 400
    assert receipt["optimization"]["device"] == "cpu"
    assert receipt["optimization"]["torch_threads"] == 2
    assert receipt["evidence_scope"] == (
        "pre_frozen_dataset_free_holdout_with_replayable_one_update_"
        "wiring_smoke_not_detection_performance"
    )
    assert receipt["preformal_execution_disclosure"] == {
        "wiring_smoke_uses_first_schedule_update": True,
        "wiring_smoke_may_be_replayed_by_test_suite": True,
        "maximum_updates_per_smoke_invocation": 1,
        "smoke_checks_only_execution_gradients_and_4_4_2_contract": True,
        "post_update_performance_gate_or_candidate_selection_allowed": False,
        "full_400_update_result_unobserved_at_freeze": True,
    }
    assert receipt["decision_rule"][
        "all_400_updates_require_six_finite_nonzero_parameter_gradients"
    ] is True

    changed = json.loads(json.dumps(receipt))
    changed["thresholds"]["clean_D_delta_mean_min_inclusive"] = 0.79
    with pytest.raises(RuntimeError, match="thresholds"):
        holdout._validate_design_contract(changed)


def test_receipt_owns_the_exact_pre_frozen_source_closure() -> None:
    receipt = _load(RECEIPT)
    bindings = receipt["source_bindings"]

    assert "cure_lite/ccfr_development_inputs.py" in holdout._SOURCE_PATHS
    assert (
        "cure_lite/experiment/conservative_toy_inputs.py"
        not in holdout._SOURCE_PATHS
    )
    assert set(bindings) == set(holdout._SOURCE_PATHS)
    assert holdout._validate_source_bindings(bindings) == bindings

    missing = dict(bindings)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="exact source binding set"):
        holdout._validate_source_bindings(missing)

    changed = dict(bindings)
    changed["cure_lite/coverage_feature_release_decoder.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="bound source differs"):
        holdout._validate_source_bindings(changed)


def test_new_holdout_contract_rebuilds_all_eight_groups_and_schedules() -> None:
    contract = holdout._holdout_contract()

    assert contract["implementation_fingerprints"] == (
        holdout.EXPECTED_IMPLEMENTATION_FINGERPRINTS
    )
    assert contract["catalog_pair_count"] == 222
    assert contract["pair_slots"] == 800
    assert set(contract["strata_counts"]) == set(holdout._EXPECTED_GROUPS)
    assert set(contract["factual_exposures_per_state"].values()) == {100}
    assert contract["old_six_case_tensor_reused"] is False
    assert contract["old_v10_222_role_tensor_reused"] is False
    assert contract["old_v10_schedule_reused"] is False
    assert all(
        counts["G_norm_tail"] > 0
        for counts in contract["strata_counts"].values()
    )


def test_one_update_smoke_checks_six_gradients_and_exact_4_4_2() -> None:
    specs = build_ccfr_holdout_pair_specs()
    schedule = build_ccfr_holdout_schedule(specs)
    decoder = holdout._build_decoder(
        CURELiteCoverageFeatureReleaseDecoder
    )

    audit = holdout._optimize(
        decoder,
        specs=specs,
        schedule=schedule,
        update_count=1,
    )

    assert audit["updates_checked"] == 1
    assert audit["gradient_contract"]["parameter_tensors_per_update"] == 6
    assert audit["gradient_contract"]["gradient_observations"] == 6
    assert audit["gradient_contract"]["failure_count"] == 0
    assert audit["gradient_contract"]["minimum_l2_norm"] > 0.0
    assert audit["step_contract"]["all_updates_exact_4_4_2"] is True
    assert audit["forward_contract"][
        "all_updates_exact_three_4_state_calls"
    ] is True
    assert audit["all_pass"] is False


def test_ccfr_and_report_only_v8_start_from_identical_parameters() -> None:
    ccfr = holdout._build_decoder(
        CURELiteCoverageFeatureReleaseDecoder
    )
    comparator = holdout._build_decoder(
        CURELiteConservativeFactorizedDecoder
    )

    assert holdout._decoder_fingerprint(ccfr) == (
        holdout._decoder_fingerprint(comparator)
    )
    assert tuple(ccfr.state_dict()) == tuple(comparator.state_dict())


def test_formal_train_path_has_no_update_override() -> None:
    assert inspect.signature(holdout._optimize).parameters[
        "update_count"
    ].default is inspect.Parameter.empty
    source = inspect.getsource(holdout._train_and_evaluate)
    assert "update_count=UPDATE_COUNT" in source
    assert holdout.UPDATE_COUNT == 400
    evaluate_source = inspect.getsource(holdout.evaluate)
    assert "confirmation_threads = 2" in evaluate_source
    assert "min(previous_threads, 2)" not in evaluate_source


def test_group_gates_connect_clean_D_component_null_and_all_zero_strata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = build_ccfr_holdout_pair_specs()

    def run_group(group_id: str, *, corrupt_H: bool = False) -> dict[str, object]:
        specs = tuple(
            spec for spec in population if spec.group_id == group_id
        )[:2]
        outcome = holdout.build_ccfr_holdout_outcome_batch(specs)
        strata = holdout.build_ccfr_holdout_strata(outcome)
        plus = torch.full_like(
            outcome.pair_batch.label_increment,
            -10.0,
        )
        plus[outcome.completion_plus] = 10.0
        minus = plus.clone()
        minus[strata.D] = 10.0
        if corrupt_H:
            minus[strata.H] = 10.0

        def paired_logits(
            _decoder: object,
            *,
            feature: torch.Tensor,
            occupancy_plus: torch.Tensor,
            occupancy_minus: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert feature.shape[0] == len(specs)
            assert torch.equal(
                occupancy_plus,
                outcome.pair_batch.occupancy_plus,
            )
            assert torch.equal(
                occupancy_minus,
                outcome.pair_batch.occupancy_minus,
            )
            return plus, minus

        monkeypatch.setattr(holdout, "_paired_endpoint_logits", paired_logits)
        return holdout._group_evaluation(torch.nn.Identity(), specs)

    clean = run_group("clean_same_cell_1px")
    assert clean["all_pass"] is True
    assert {"D_delta", "D_plus", "D_minus", "D_direction"}.issubset(
        clean["checks"]
    )
    assert {"H", "G_near", "G_norm_tail"}.issubset(clean["checks"])

    component = run_group("component_null_block")
    assert component["all_pass"] is True
    assert not any(name.startswith("D_") for name in component["checks"])
    assert {"H", "G_near", "G_norm_tail"}.issubset(
        component["checks"]
    )

    corrupted = run_group("clean_same_cell_1px", corrupt_H=True)
    assert corrupted["checks"]["H"] is False
    assert corrupted["all_pass"] is False


def test_final_gate_is_strict_population_factual_and_eight_group_conjunction() -> None:
    groups = [
        {
            "group_id": group_id,
            "metrics": {"pair_count": pair_count},
            "all_pass": True,
        }
        for group_id, pair_count in holdout._EXPECTED_GROUPS.items()
    ]
    passing = holdout._final_gate_summary(
        population_objective=0.01,
        factual={"all_pass": True},
        groups=groups,
    )
    assert passing["all_pass"] is True
    assert passing["passed_group_count"] == 8

    one_group_fails = [dict(group) for group in groups]
    one_group_fails[0]["all_pass"] = False
    group_failure = holdout._final_gate_summary(
        population_objective=0.0,
        factual={"all_pass": True},
        groups=one_group_fails,
    )
    assert group_failure["all_pass"] is False
    assert group_failure["failed_group_count"] == 1

    factual_failure = holdout._final_gate_summary(
        population_objective=0.0,
        factual={"all_pass": False},
        groups=groups,
    )
    assert factual_failure["checks"]["factual"] is False
    assert factual_failure["all_pass"] is False

    population_failure = holdout._final_gate_summary(
        population_objective=0.1,
        factual={"all_pass": True},
        groups=groups,
    )
    assert population_failure["checks"]["population_objective"] is False
    assert population_failure["all_pass"] is False


def test_v8_comparator_cannot_change_ccfr_decision() -> None:
    ccfr = {
        "initial_decoder_fingerprint": "a" * 64,
        "training": {"optimizer_contract": {"ccfr": True}},
        "all_pass": True,
    }
    comparator = {
        "execution_status": "COMPLETED",
        "initial_decoder_fingerprint": "b" * 64,
        "training": {"optimizer_contract": {"v8": True}},
        "all_pass": False,
    }
    result = holdout._assemble_result(
        prerequisites={"sentinel": True},
        attempt_binding={"sentinel": True},
        ccfr=ccfr,
        comparator=comparator,
        runtime={"sentinel": True},
    )

    assert result["all_pass"] is True
    assert result["decision"] == (
        "CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_PASS"
    )
    assert result["matched_v8_comparator_affects_decision"] is False
    assert result["same_initialization_verified"] is False
    assert result["same_optimizer_verified"] is False
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert stable_fingerprint(unsigned) == fingerprint

    comparator_error = {
        "execution_status": "ERROR",
        "comparator_execution_error": {
            "type": "RuntimeError",
            "message": "report-only failure",
        },
        "all_pass": False,
    }
    error_result = holdout._assemble_result(
        prerequisites={"sentinel": True},
        attempt_binding={"sentinel": True},
        ccfr=ccfr,
        comparator=comparator_error,
        runtime={"sentinel": True},
    )
    assert error_result["all_pass"] is True
    assert error_result["matched_v8_comparator_execution_status"] == "ERROR"
    assert error_result["same_initialization_verified"] is None
    assert error_result["same_optimizer_verified"] is None

    failed_ccfr = {
        **ccfr,
        "all_pass": False,
    }
    for report in (comparator, comparator_error):
        failed_result = holdout._assemble_result(
            prerequisites={"sentinel": True},
            attempt_binding={"sentinel": True},
            ccfr=failed_ccfr,
            comparator=report,
            runtime={"sentinel": True},
        )
        assert failed_result["all_pass"] is False
        assert failed_result["decision"] == (
            "CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_FAIL"
        )


def test_attempt_and_result_writers_are_create_only(tmp_path: Path) -> None:
    attempt_path = tmp_path / "attempt.json"
    result_path = tmp_path / "result.json"
    complete_path = tmp_path / "result.COMPLETE.sha256"
    payload = {"sentinel": True}

    holdout._write_json_create_only(attempt_path, payload)
    holdout._write_json_create_only(result_path, payload)
    with pytest.raises(FileExistsError):
        holdout._write_json_create_only(attempt_path, {"sentinel": False})
    with pytest.raises(FileExistsError):
        holdout._write_json_create_only(result_path, {"sentinel": False})

    digest = holdout._write_complete_create_only(
        complete_path,
        result_path=result_path,
    )
    assert digest == file_sha256(result_path)
    assert complete_path.read_text(encoding="utf-8") == (
        f"{digest}  {result_path.name}\n"
    )
    with pytest.raises(FileExistsError):
        holdout._write_complete_create_only(
            complete_path,
            result_path=result_path,
        )


def test_attempt_payload_consumes_one_dataset_free_canonical_run() -> None:
    payload = holdout._attempt_payload({"sentinel": True})
    assert payload["attempt_number"] == 1
    assert payload["execution"] == {
        "updates": 400,
        "device": "cpu",
        "torch_threads": 2,
        "automatic_retry_allowed": False,
        "dataset_access_allowed": False,
        "D_R_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
    }
    unsigned = dict(payload)
    fingerprint = unsigned.pop("attempt_fingerprint")
    assert stable_fingerprint(unsigned) == fingerprint


def test_development_chain_must_bind_the_same_frozen_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = {
        "repo_path": "protocols/frozen-receipt.json",
        "file_sha256": "a" * 64,
        "receipt_fingerprint": "b" * 64,
        "status": "FROZEN_BEFORE_DEVELOPMENT_REGRESSION_RESULT",
        "design_seed": 3987573916,
        "implementation_fingerprints": (
            holdout.EXPECTED_IMPLEMENTATION_FINGERPRINTS
        ),
        "source_bindings": {"sentinel.py": "c" * 64},
    }
    expected_binding = {
        name: design[name]
        for name in (
            "repo_path",
            "file_sha256",
            "receipt_fingerprint",
            "status",
            "design_seed",
        )
    }
    development = {
        "protocol_binding": {
            "pre_frozen_holdout_binding": expected_binding,
            "source_bindings": {
                design["repo_path"]: design["file_sha256"]
            },
        }
    }
    monkeypatch.setattr(holdout, "_load_design_receipt", lambda: design)
    monkeypatch.setattr(holdout, "_load_development_pass", lambda: development)
    monkeypatch.setattr(
        holdout,
        "_holdout_contract",
        lambda: {
            "implementation_fingerprints": (
                holdout.EXPECTED_IMPLEMENTATION_FINGERPRINTS
            )
        },
    )
    monkeypatch.setattr(
        holdout,
        "_runtime_import_boundary",
        lambda: {"observed_forbidden_modules": [], "all_pass": True},
    )
    monkeypatch.setattr(
        holdout,
        "_runtime_source_closure",
        lambda _bindings: {"unbound_local_imports": [], "all_pass": True},
    )

    assert holdout._load_prerequisites()["development_pass"] is development
    development["protocol_binding"]["pre_frozen_holdout_binding"] = {
        **expected_binding,
        "file_sha256": "d" * 64,
    }
    with pytest.raises(RuntimeError, match="current pre-frozen"):
        holdout._load_prerequisites()


def test_noncanonical_output_is_rejected_without_writing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="canonical path"):
        holdout._assert_fresh_canonical_artifacts(
            tmp_path / "not-canonical.json"
        )
