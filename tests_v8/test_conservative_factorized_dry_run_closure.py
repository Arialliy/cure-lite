from __future__ import annotations

import ast
import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conserving_subpixel_evidence_allocation_v8"
)
CLOSURE = PROTOCOL / "bounded_dry_run_closure_receipt.json"
RUNNER = ROOT / "tools/dry_run_conservative_factorized_outcome_bounded.py"

EXPECTED_SOURCE_FILES = frozenset(
    {
        "cure_lite/__init__.py",
        "cure_lite/base_identity.py",
        "cure_lite/cache/__init__.py",
        "cure_lite/cache/base_cache.py",
        "cure_lite/cache/schema.py",
        "cure_lite/cache/state_cache.py",
        "cure_lite/calibration.py",
        "cure_lite/calibration_ledger.py",
        "cure_lite/config.py",
        "cure_lite/conservative_factorized_config.py",
        "cure_lite/conservative_factorized_decoder.py",
        "cure_lite/crossing_factorized_config.py",
        "cure_lite/crossing_factorized_decoder.py",
        "cure_lite/data.py",
        "cure_lite/decoder.py",
        "cure_lite/efficiency.py",
        "cure_lite/experiment/__init__.py",
        "cure_lite/experiment/artifacts.py",
        "cure_lite/experiment/cache_pipeline.py",
        "cure_lite/experiment/conservative_toy_inputs.py",
        "cure_lite/experiment/deployment.py",
        "cure_lite/experiment/efficiency_evidence.py",
        "cure_lite/experiment/evaluation_pipeline.py",
        "cure_lite/experiment/formal_anchor.py",
        "cure_lite/experiment/formal_evaluation.py",
        "cure_lite/experiment/formal_training.py",
        "cure_lite/experiment/paired_artifacts.py",
        "cure_lite/experiment/seed_registry.py",
        "cure_lite/experiment/stage_a_m_extension.py",
        "cure_lite/experiment/stage_a_m_runner.py",
        "cure_lite/experiment/stage_a_runner.py",
        "cure_lite/experiment/training_pipeline.py",
        "cure_lite/factorized_config.py",
        "cure_lite/factorized_decoder.py",
        "cure_lite/frozen_base.py",
        "cure_lite/instances.py",
        "cure_lite/intervention.py",
        "cure_lite/losses.py",
        "cure_lite/matching.py",
        "cure_lite/metrics.py",
        "cure_lite/model.py",
        "cure_lite/occupancy.py",
        "cure_lite/paired_control_inputs.py",
        "cure_lite/paired_control_losses.py",
        "cure_lite/paired_losses.py",
        "cure_lite/paired_outcome_losses.py",
        "cure_lite/paired_outcome_types.py",
        "cure_lite/paired_types.py",
        "cure_lite/sampling.py",
        "cure_lite/splits.py",
        "cure_lite/stage_a.py",
        "cure_lite/supervision.py",
        "cure_lite/train/__init__.py",
        "cure_lite/train/engine.py",
        "cure_lite/train/paired_control_step.py",
        "cure_lite/train/paired_outcome_step.py",
        "cure_lite/train/paired_step.py",
        "cure_lite/train/pools.py",
        "cure_lite/train/step.py",
        "cure_lite/types.py",
        "tools/__init__.py",
        "tools/dry_run_conservative_factorized_outcome_bounded.py",
    }
)
EXPECTED_TEST_FILES = frozenset(
    {
        "tests_v8/test_dry_run_conservative_factorized_outcome_bounded.py",
        "tests_v8/test_conservative_factorized_dry_run_closure.py",
    }
)


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _verified_fingerprint(
    value: dict[str, object],
    field: str,
) -> str:
    unsigned = dict(value)
    observed = unsigned.pop(field)
    assert isinstance(observed, str)
    assert observed == stable_fingerprint(unsigned)
    return observed


def test_v8_bounded_dry_run_closure_is_exact_and_narrow() -> None:
    assert CLOSURE.is_file()
    assert not CLOSURE.is_symlink()
    closure = _object(CLOSURE)
    fingerprint = _verified_fingerprint(
        closure,
        "receipt_fingerprint",
    )
    assert len(fingerprint) == 64
    assert closure["schema_version"] == (
        "cure-lite-cc-sea-v8-bounded-dry-run-closure-v1"
    )
    assert closure["artifact_kind"] == "bounded_dry_run_closure"
    assert closure["method_id"] == "cc_sea_v8"
    assert closure["phase_status"] == "FROZEN_BOUNDED_DRY_RUN_PASS"
    assert closure["decision"] == (
        "CC_SEA_V8_DRY_RUN_CLOSURE_PASS_AND_"
        "REAL_BOUNDED_CODE_AUTHORIZED"
    )

    for binding in closure["protocol_bindings"].values():
        path = ROOT / binding["repo_path"]
        assert path.is_file() and not path.is_symlink()
        assert file_sha256(path) == binding["file_sha256"]
        payload = _object(path)
        assert _verified_fingerprint(
            payload,
            binding["fingerprint_field"],
        ) == binding["fingerprint"]

    result_binding = closure["single_process_result_binding"]
    result_path = ROOT / result_binding["repo_path"]
    result = _object(result_path)
    assert file_sha256(result_path) == result_binding["file_sha256"]
    assert _verified_fingerprint(result, "result_fingerprint") == (
        result_binding["result_fingerprint"]
    )
    assert result["decision"] == (
        "CC_SEA_V8_DRY_RUN_SINGLE_PROCESS_PASS"
    )
    assert result["all_pass"] is True
    assert result["closure_eligible_after_replay"] is True
    assert result["real_D_R_bounded_code_creation_authorized"] is False

    source_binding = closure["source_bindings"]
    assert source_binding["policy"] == (
        "explicit_loaded_source_allowlist_v1"
    )
    assert set(source_binding["files"]) == EXPECTED_SOURCE_FILES
    assert source_binding["source_file_count"] == len(EXPECTED_SOURCE_FILES)
    assert source_binding["future_unlisted_files_affect_closure_validity"] is False
    assert source_binding["absence_of_other_files_claimed"] is False
    for repo_path, expected in source_binding["files"].items():
        path = ROOT / repo_path
        assert path.is_file() and not path.is_symlink()
        assert file_sha256(path) == expected

    test_binding = closure["test_bindings"]
    assert test_binding["policy"] == "explicit_test_allowlist_v1"
    assert set(test_binding["files"]) == EXPECTED_TEST_FILES
    assert test_binding["future_unlisted_tests_affect_closure_validity"] is False
    for repo_path, expected in test_binding["files"].items():
        path = ROOT / repo_path
        assert path.is_file() and not path.is_symlink()
        assert file_sha256(path) == expected


def test_v8_bounded_dry_run_replay_and_isolation_are_closed() -> None:
    closure = _object(CLOSURE)
    replay = closure["independent_process_replay"]
    assert replay["process_count"] == 3
    assert replay["distinct_processes"] is True
    assert replay["byte_identical"] is True
    assert replay["all_match_canonical_result"] is True
    assert replay["exit_codes"] == [0, 0, 0]
    assert len(set(replay["output_sha256"])) == 1
    assert len(set(replay["output_byte_counts"])) == 1
    assert len(set(replay["result_fingerprints"])) == 1

    create_only = closure["create_only_audit"]
    assert create_only == {
        "first_write_to_absent_path_succeeded": True,
        "second_write_to_same_path_rejected": True,
        "second_write_exit_code_nonzero": True,
        "existing_bytes_unchanged_after_rejection": True,
        "existing_sha256_unchanged_after_rejection": True,
        "evaluate_function_has_no_write_side_effect": True,
        "canonical_created_only_after_replays_match": True,
        "passed": True,
    }

    isolation = closure["payload_isolation_audit"]
    assert isolation["entrypoint_direct_import_audit_passed"] is True
    assert isolation["package_initialization_imports_disclosed"] is True
    assert isolation["zero_real_loader_calls"] is True
    assert isolation["zero_dataset_or_cache_payload_accesses"] is True
    assert isolation["actual_file_access_allowlist_test_passed"] is True
    assert isolation["D_R_payload_accessed"] is False
    assert isolation["D_V_accessed"] is False
    assert isolation["D_T_accessed"] is False

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    direct_imports: set[str] = set()
    forbidden_dynamic: list[str] = []
    forbidden_discovery: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            direct_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            direct_imports.add(node.module or "")
        elif isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if (
                isinstance(node.func, ast.Name)
                and name in {"__import__", "eval", "exec", "compile"}
            ) or (
                isinstance(node.func, ast.Attribute)
                and name == "import_module"
            ):
                forbidden_dynamic.append(name)
            if name in {"glob", "rglob", "iterdir", "walk"}:
                forbidden_discovery.append(name)
    assert not any(
        module in {
            "cure_lite.data",
            "cure_lite.cache.base_cache",
            "cure_lite.cache.state_cache",
            "cure_lite.experiment.cache_pipeline",
            "cure_lite.experiment.training_pipeline",
        }
        for module in direct_imports
    )
    assert forbidden_dynamic == []
    assert forbidden_discovery == []


def test_v8_bounded_dry_run_closure_authorizes_only_code_creation() -> None:
    closure = _object(CLOSURE)
    assert all(closure["gate_summary"].values())
    assert closure["authorization_scope"] == {
        "real_D_R_bounded_code_creation_authorized": True,
        "real_D_R_bounded_mock_tests_authorized": True,
        "real_D_R_bounded_executor_creation_authorized": True,
        "real_D_R_bounded_runner_creation_authorized": True,
        "real_D_R_payload_access_authorized": False,
        "real_D_R_bounded_execution_authorized": False,
        "real_run_authorization_receipt_created": False,
        "implementation_closure_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "other_detector_integration_authorized": False,
    }
    assert closure["boundary"] == {
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "calibration_performed": False,
        "detection_performance_evaluated": False,
        "real_D_R_bounded_execution_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "other_detector_integration_authorized": False,
    }
