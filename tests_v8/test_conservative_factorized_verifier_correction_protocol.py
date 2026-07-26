from __future__ import annotations

from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from tools import run_conservative_factorized_outcome_bounded_r2 as runner


def test_prefrozen_proposal_and_config_are_exact() -> None:
    proposal, proposal_path = runner._load_proposal()
    config_path = runner._repo_file(
        runner.CONFIG_REPO_PATH,
        name="r2 config",
    )
    config = runner._load_config(config_path)

    assert file_sha256(proposal_path) == runner.PROPOSAL_FILE_SHA256
    assert proposal["proposal_fingerprint"] == runner.PROPOSAL_FINGERPRINT
    assert file_sha256(config_path) == runner.CONFIG_FILE_SHA256
    assert config["config_fingerprint"] == runner.CONFIG_FINGERPRINT
    assert proposal["r1_failure_attribution"] == {
        "classification": "EXECUTOR_RESULT_TO_PUBLICATION_CONTRACT_ERROR",
        "model_gate_status": "NOT_EVALUATED_OR_NOT_PUBLISHED",
        "performance_status": "NOT_EVALUATED",
        "root_cause": (
            "runner_expected_boolean_gate_values_but_executor_"
            "returns_structured_gate_records"
        ),
        "r1_remains_immutable": True,
        "r1_is_not_a_model_nonpass": True,
        "r1_run_claim_consumed": True,
    }
    assert config["scientific_contract"] == {
        "source": "exact_v1_bounded_config_binding",
        "model_change_allowed": False,
        "core_executor_change_allowed": False,
        "loss_change_allowed": False,
        "budget_change_allowed": False,
        "threshold_change_allowed": False,
        "population_or_schedule_change_allowed": False,
    }


def test_r1_and_frozen_scientific_runtime_remain_unchanged() -> None:
    v1_config, v1_config_path = runner._load_v1_config()
    r1 = runner._load_r1()
    assert file_sha256(v1_config_path) == runner.V1_CONFIG_FILE_SHA256
    assert v1_config["config_fingerprint"] == runner.V1_CONFIG_FINGERPRINT
    assert r1.decision == "CC_SEA_V8_BOUNDED_EXECUTION_ERROR"
    assert r1.complete_fingerprint == runner.R1_COMPLETE_FINGERPRINT
    assert (
        file_sha256(runner.ROOT / runner.V1_RUNNER_REPO_PATH)
        == runner.V1_RUNNER_FILE_SHA256
    )
    assert (
        file_sha256(runner.ROOT / runner.CORE_REPO_PATH)
        == runner.CORE_FILE_SHA256
    )


def test_additive_runtime_inventory_is_complete_and_exact() -> None:
    implementation = runner._implementation_binding()
    runner._verify_implementation_files(implementation)
    inherited = implementation["v1_runtime_files"]
    additive = implementation["additive_runtime_files"]
    complete = implementation["all_runtime_files"]
    assert isinstance(inherited, dict)
    assert isinstance(additive, dict)
    assert isinstance(complete, dict)
    assert len(inherited) == 83
    assert set(additive) == {
        (
            "cure_lite/experiment/"
            "conservative_factorized_result_verifier.py"
        ),
        "tools/run_conservative_factorized_outcome_bounded_r2.py",
    }
    assert len(complete) == 85
    assert not set(inherited) & set(additive)
    assert implementation["model_or_core_file_changed"] is False
    assert len(stable_fingerprint(implementation)) == 64


def test_closure_and_authorization_are_phase_stable() -> None:
    closure_path = runner.ROOT / runner.CLOSURE_REPO_PATH
    authorization_path = runner.ROOT / runner.AUTHORIZATION_REPO_PATH
    implementation = runner._implementation_binding()
    config = runner._load_config(
        runner.ROOT / runner.CONFIG_REPO_PATH
    )

    if not closure_path.exists():
        assert not authorization_path.exists()
        assert config["boundary"]["r2_execution_authorized"] is False
        return

    closure, loaded_closure_path = runner._load_closure(implementation)
    assert loaded_closure_path == closure_path
    assert closure["boundary"]["r2_execution_authorized"] is False
    if not authorization_path.exists():
        return

    authorization, loaded_authorization_path = runner._load_authorization(
        config,
        closure,
        closure_path,
        implementation,
    )
    assert loaded_authorization_path == authorization_path
    assert authorization["authorization"]["exact_r2_run_count"] == 1
    assert (
        authorization["authorization"]["output_repo_path"]
        == runner.OUTPUT_REPO_PATH
    )
