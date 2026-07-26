from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.experiment import (
    recoverable_factorized_outcome_bounded as recoverable_core,
)
from cure_lite.experiment.recoverable_factorized_outcome_bounded import (
    RECOVERABLE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
    RECOVERABLE_FACTORIZED_OUTCOME_METHOD_ID,
    RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS,
)
from tools import run_recoverable_factorized_outcome_bounded as runner


PAIR_CATALOG_FINGERPRINT = (
    "4886e52d2cfb3392d0f4fdda376159d6e7f694fd449dc809cf8874793febde76"
)
PREPARED_CATALOG_FINGERPRINT = (
    "4955e5b4f1749b5f267db0ac1f031335a16cc48a470d6446ca6c99d04a5e85ed"
)
V6_CONFIG_FILE_SHA256 = (
    "c12c5a4d98123a7f39e38a2ebeaff696807ad215bdf3d231eec17c9797b42c35"
)


def _internally_fingerprinted(
    payload: dict[str, object],
    *,
    field: str,
) -> dict[str, object]:
    result = dict(payload)
    result[field] = stable_fingerprint(payload)
    return result


def _resign(
    payload: dict[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    unsigned[field] = stable_fingerprint(unsigned)
    return unsigned


class _UnchangedBundle:
    def __init__(self) -> None:
        self.verify_calls = 0

    def verify_unchanged(self) -> None:
        self.verify_calls += 1


def _fake_catalog() -> SimpleNamespace:
    return SimpleNamespace(
        catalog_fingerprint=PAIR_CATALOG_FINGERPRINT,
        split="D_R",
        clean_positive=tuple(range(206)),
        component_null=tuple(range(16)),
    )


def _fake_population() -> SimpleNamespace:
    receipt = _internally_fingerprinted(
        {
            "schema_version": "test-anchor-population",
            "pair_catalog_fingerprint": PAIR_CATALOG_FINGERPRINT,
            "prepared_catalog_fingerprint": PREPARED_CATALOG_FINGERPRINT,
            "factual_miss": [],
            "factual_no_miss": [],
            "identity_null": [],
        },
        field="population_fingerprint",
    )
    return SimpleNamespace(
        pair_catalog_fingerprint=PAIR_CATALOG_FINGERPRINT,
        prepared_catalog_fingerprint=PREPARED_CATALOG_FINGERPRINT,
        population_fingerprint=receipt["population_fingerprint"],
        canonical_receipt=lambda: dict(receipt),
    )


def _fake_factual_schedule(population: SimpleNamespace) -> SimpleNamespace:
    receipt = _internally_fingerprinted(
        {
            "schema_version": "test-factual-schedule",
            "population_fingerprint": population.population_fingerprint,
            "optimizer_updates": 400,
        },
        field="schedule_fingerprint",
    )
    return SimpleNamespace(
        schedule_fingerprint=receipt["schedule_fingerprint"],
        canonical_receipt=lambda: dict(receipt),
    )


def _fake_materializer() -> SimpleNamespace:
    receipt = _internally_fingerprinted(
        {
            "schema_version": "test-outcome-inputs",
            "pair_catalog_fingerprint": PAIR_CATALOG_FINGERPRINT,
            "prepared_catalog_fingerprint": PREPARED_CATALOG_FINGERPRINT,
            "outcome_pairs": 222,
        },
        field="materializer_fingerprint",
    )
    return SimpleNamespace(
        pair_catalog_fingerprint=PAIR_CATALOG_FINGERPRINT,
        prepared_catalog_fingerprint=PREPARED_CATALOG_FINGERPRINT,
        materializer_fingerprint=receipt["materializer_fingerprint"],
        canonical_receipt=lambda: dict(receipt),
    )


def _fake_outcome_schedule() -> SimpleNamespace:
    receipt = _internally_fingerprinted(
        {
            "schema_version": "test-outcome-schedule",
            "catalog_fingerprint": PAIR_CATALOG_FINGERPRINT,
            "optimizer_updates": 400,
        },
        field="schedule_fingerprint",
    )
    return SimpleNamespace(
        catalog_fingerprint=PAIR_CATALOG_FINGERPRINT,
        schedule_fingerprint=receipt["schedule_fingerprint"],
        canonical_receipt=lambda: dict(receipt),
    )


def _fake_authorization() -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": runner.AUTHORIZATION_SCHEMA,
        "method_id": "pr_svef_v6",
        "split": "D_R",
        "phase_status": "FROZEN_BOUNDED_CODE_GATE_PASS",
        "decision": "PRSVEF_V6_ONE_REAL_D_R_BOUNDED_RUN_AUTHORIZED",
        "authorization": {
            "real_D_R_bounded_execution": True,
            "exact_run_count": 1,
            "device": "cpu",
            "output_repo_path": runner.OUTPUT_REPO_PATH,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_V_access_allowed": False,
            "D_T_access_allowed": False,
            "formal_800_allowed": False,
        },
    }
    return {
        **unsigned,
        "receipt_fingerprint": stable_fingerprint(unsigned),
    }


def _exact_authorization_receipt(
    implementation: dict[str, object],
) -> dict[str, object]:
    focused_files = runner._test_file_binding(
        runner.FOCUSED_TEST_REPO_PATHS
    )
    full_files = runner._full_test_inventory()
    wrapper_test = (
        Path.cwd() / runner.TEMPERATURE_TEST_REPO_PATH
    ).resolve()
    unsigned: dict[str, object] = {
        "schema_version": runner.AUTHORIZATION_SCHEMA,
        "method_id": "pr_svef_v6",
        "split": "D_R",
        "phase_status": "FROZEN_BOUNDED_CODE_GATE_PASS",
        "decision": "PRSVEF_V6_ONE_REAL_D_R_BOUNDED_RUN_AUTHORIZED",
        "receipt_fingerprint_scope": (
            "all-fields-except-receipt_fingerprint"
        ),
        "authorization": {
            "real_D_R_bounded_execution": True,
            "exact_run_count": 1,
            "device": runner.FROZEN_DEVICE,
            "output_repo_path": runner.OUTPUT_REPO_PATH,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_V_access_allowed": False,
            "D_T_access_allowed": False,
            "formal_800_allowed": False,
        },
        "bounded_config_binding": {
            "repo_path": runner.CONFIG_REPO_PATH,
            "file_sha256": runner.CONFIG_FILE_SHA256,
            "config_fingerprint": runner.CONFIG_FINGERPRINT,
        },
        "proposal_binding": {
            "repo_path": runner.PROPOSAL_REPO_PATH,
            "file_sha256": runner.PROPOSAL_FILE_SHA256,
            "proposal_fingerprint": runner.PROPOSAL_FINGERPRINT,
        },
        "toy_gate_closure_binding": {
            "repo_path": runner.TOY_CLOSURE_REPO_PATH,
            "file_sha256": runner.TOY_CLOSURE_FILE_SHA256,
            "receipt_fingerprint": runner.TOY_CLOSURE_FINGERPRINT,
        },
        "runtime_implementation_binding": {
            "implementation_fingerprint": stable_fingerprint(
                implementation
            ),
            "all_runtime_files": implementation["all_runtime_files"],
        },
        "focused_tests_binding": {
            "passed": True,
            "command": list(runner.FOCUSED_TEST_COMMAND),
            "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
            "exit_code": 0,
            "passed_test_count": runner.FOCUSED_TEST_PASSED_COUNT,
            "failed_test_count": 0,
            "skipped_test_count": 0,
            "files": focused_files,
        },
        "full_regression_binding": {
            "passed": True,
            "command": list(runner.FULL_REGRESSION_COMMAND),
            "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
            "exit_code": 0,
            "passed_test_count": runner.FULL_REGRESSION_PASSED_COUNT,
            "failed_test_count": 0,
            "skipped_test_count": (
                runner.FULL_REGRESSION_SKIPPED_COUNT
            ),
            "files": full_files,
        },
        "execution_control_binding": {
            "wrapper_repo_path": (
                runner.TEMPERATURE_WRAPPER_REPO_PATH
            ),
            "wrapper_file_sha256": (
                runner.TEMPERATURE_WRAPPER_FILE_SHA256
            ),
            "gpu_index": 0,
            "pause_temperature_celsius": 82,
            "resume_temperature_celsius": 75,
            "wrapped_command": list(runner.TEMPERATURE_WRAPPED_COMMAND),
            "wrapper_test_repo_path": (
                runner.TEMPERATURE_TEST_REPO_PATH
            ),
            "wrapper_test_file_sha256": file_sha256(wrapper_test),
            "wrapper_test_command": list(
                runner.TEMPERATURE_TEST_COMMAND
            ),
            "wrapper_test_passed": True,
            "wrapper_test_passed_count": (
                runner.TEMPERATURE_TEST_PASSED_COUNT
            ),
            "wrapper_test_failed_count": 0,
            "wrapper_test_skipped_count": 0,
        },
    }
    return {
        **unsigned,
        "receipt_fingerprint": stable_fingerprint(unsigned),
    }


def _load_temporary_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    receipt: dict[str, object],
    implementation: dict[str, object],
) -> tuple[dict[str, object], Path]:
    original_root = runner._ROOT
    expected_full = runner._full_test_inventory()
    temporary_root = tmp_path / "temporary-repository"
    authorization_path = (
        temporary_root / runner.AUTHORIZATION_REPO_PATH
    )
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_text(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def routed_repo_file(path_text: object, *, name: str) -> Path:
        if path_text == runner.AUTHORIZATION_REPO_PATH:
            return authorization_path.resolve(strict=True)
        if not isinstance(path_text, str):
            raise ValueError(f"{name} must be repo-relative")
        candidate = (original_root / path_text).resolve(strict=True)
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"{name} must be a regular file")
        return candidate

    monkeypatch.setattr(runner, "_ROOT", temporary_root)
    monkeypatch.setattr(runner, "_repo_file", routed_repo_file)
    monkeypatch.setattr(
        runner,
        "_full_test_inventory",
        lambda: dict(expected_full),
    )
    config = json.loads(
        (original_root / runner.CONFIG_REPO_PATH).read_text(
            encoding="utf-8"
        )
    )
    return runner._load_authorization(config, implementation)


def _fake_structural_audit(
    *,
    all_pass: bool = True,
) -> dict[str, object]:
    budget = {
        "decoder_calls": 28,
        "decoder_state_evaluations": 888,
        "expected_decoder_calls": 28,
        "expected_decoder_state_evaluations": 888,
        "factual_vacancy_field_calls": 1,
        "factual_vacancy_field_states": 16,
    }
    check_names = (
        "zero_feature_occupancy_delta_exact_zero",
        "gate_support_outside_logit_delta_exact_zero",
        "gate_support_outside_probability_delta_exact_zero",
        "all_audited_fields_finite",
        "vacancy_deletion_monotonicity_exact",
        "deletion_logit_monotonicity_exact",
        "deletion_probability_monotonicity_exact",
        "native_subpixel_path_without_resize",
        "all_clean_D_pixels_structurally_reachable",
        "all_clean_pairs_have_nonempty_H",
        "all_component_null_pairs_have_positive_gate_support",
        "all_factual_targets_have_positive_vacancy",
        "structural_audit_decoder_budget_exact",
    )
    population_checks = {name: True for name in check_names}
    nonfinite = 0
    if not all_pass:
        population_checks["all_audited_fields_finite"] = False
        nonfinite = 1
    operator = recoverable_core._audit_recoverable_operator_contract(
        device=torch.device("cpu"),
    )
    operator_checks = dict(operator["checks"])
    checks = {**population_checks, **operator_checks}
    clean_records = [
        {
            "pair_id": f"clean-{index:03d}",
            "pair_kind": "clean_positive",
            "D_pixels": 13 if index < 79 else 12,
        }
        for index in range(206)
    ]
    component_records = [
        {
            "pair_id": f"component-{index:03d}",
            "pair_kind": "component_null",
            "D_pixels": 0,
        }
        for index in range(16)
    ]
    return {
        "scope": (
            "pretraining_D_R_full_population_SVEF_structure_plus_"
            "PR_SVEF_v6_operator"
        ),
        "population_audit_scope": (
            "pretraining_D_R_full_population_SVEF_structure"
        ),
        "operator_contract": operator,
        "all_pass": all_pass,
        "checks": checks,
        "pair_count": 222,
        "clean_pair_count": 206,
        "component_null_pair_count": 16,
        "zero_feature_max_abs_occupancy_delta": 0.0,
        "outside_gate_max_abs_logit_delta": 0.0,
        "outside_gate_max_abs_probability_delta": 0.0,
        "nonfinite_audited_field_values": nonfinite,
        "vacancy_deletion_monotonicity_violations": 0,
        "deletion_logit_monotonicity_violations": 0,
        "deletion_probability_monotonicity_violations": 0,
        "field_resize_endpoint_count": 0,
        "clean_full_D_reachable_pairs": 206,
        "clean_D_reachable_pixels": 2551,
        "clean_D_total_pixels": 2551,
        "clean_nonempty_H_pairs": 206,
        "component_positive_gate_support_pairs": 16,
        "factual_full_target_reachable_anchors": 16,
        "factual_target_reachable_pixels": 150,
        "factual_target_total_pixels": 150,
        "compute_budget": budget,
        "per_pair": clean_records + component_records,
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def _fake_snapshots(
    *,
    all_pass: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    d_value = 0.80 if all_pass else 0.40
    clean_records = [
        {
            "pair_kind": "clean_positive",
            "D_pixels": 8,
            "H_pixels": 1,
            "D_mean_delta": d_value,
            "H_mean_abs_delta": 0.01,
        }
        for _ in range(206)
    ]
    component_records = [
        {"pair_kind": "component_null"} for _ in range(16)
    ]

    def snapshot(
        *,
        factual_loss: float,
        factual_no_miss_loss: float,
        plus_loss: float,
        transition_loss: float,
    ) -> dict[str, object]:
        return {
            "factual_anchors": {
                "factual_miss": {"loss": factual_loss},
                "factual_no_miss": {"loss": factual_no_miss_loss},
            },
            "outcome_population": {
                "plus_baseline_loss": plus_loss,
                "clean": {
                    "transition_loss": transition_loss,
                    "D_pair_macro_mean_delta": d_value,
                    "D_pair_fraction_mean_delta_ge_0_25": 1.0,
                    "zero_strata_pair_macro_mean_abs_delta": 0.01,
                },
                "component_null": {
                    "footprint_pair_macro_mean_abs_delta": 0.01,
                    "footprint_global_max_abs_delta": 0.10,
                    "context_pair_macro_mean_abs_delta": 0.01,
                },
                "per_pair": deepcopy(
                    clean_records + component_records
                ),
            },
            "identity_null": {"maximum_abs_delta": 0.0},
        }

    return (
        snapshot(
            factual_loss=1.0,
            factual_no_miss_loss=1.0,
            plus_loss=1.0,
            transition_loss=1.0,
        ),
        snapshot(
            factual_loss=0.50,
            factual_no_miss_loss=0.50,
            plus_loss=0.50,
            transition_loss=0.40,
        ),
    )


def _fake_computational_gates(
    *,
    all_pass: bool = True,
) -> dict[str, object]:
    initial, final = _fake_snapshots(all_pass=all_pass)
    return runner.factorized_computational_gates(initial, final)


def _full_structural_checks(
    *,
    all_pass: bool = True,
) -> dict[str, bool]:
    names = (
        "deterministic_runtime_contract_satisfied",
        "PR_SVEF_pretraining_structural_audit_passed",
        *RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS,
        "factual_anchor_and_identity_counts_exact",
        "all_222_outcome_pairs_bound",
        "all_222_outcome_pairs_evaluated_initial",
        "all_222_outcome_pairs_evaluated_final",
        "all_optimizer_updates_completed",
        "one_backward_per_update",
        "one_optimizer_step_per_update",
        "all_gradients_finite",
        "every_update_total_gradient_norm_positive",
        "decoder_parameters_changed",
        "training_forward_budget_exact",
        "evaluation_forward_budget_exact",
        "total_forward_budget_exact",
        "pair_exposure_ledger_exact",
        "source_exposure_ledger_exact",
        "factual_exposure_ledgers_exact",
        "identity_null_excluded_from_optimizer",
        "identity_null_diagnosed_without_autograd",
    )
    checks = {name: True for name in names}
    if not all_pass:
        checks["every_update_total_gradient_norm_positive"] = False
    return checks


def _fake_core_result() -> dict[str, object]:
    audit_budget = {
        "decoder_calls": 28,
        "decoder_state_evaluations": 888,
        "expected_decoder_calls": 28,
        "expected_decoder_state_evaluations": 888,
        "factual_vacancy_field_calls": 1,
        "factual_vacancy_field_states": 16,
    }
    initial, final = _fake_snapshots()
    result: dict[str, object] = {
        "schema_version": RECOVERABLE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
        "method_id": RECOVERABLE_FACTORIZED_OUTCOME_METHOD_ID,
        "execution_status": "completed",
        "device": runner.FROZEN_DEVICE,
        "decision": "PR_SVEF_BOUNDED_MODEL_CODE_GATE_PASS",
        "structural_execution_pass": True,
        "computational_model_code_gate_pass": True,
        "population_fingerprint": runner.ANCHOR_POPULATION_FINGERPRINT,
        "materializer_fingerprint": runner.MATERIALIZER_FINGERPRINT,
        "factual_schedule_fingerprint": (
            runner.FACTUAL_SCHEDULE_FINGERPRINT
        ),
        "outcome_schedule_fingerprint": (
            runner.OUTCOME_SCHEDULE_FINGERPRINT
        ),
        "decoder_config": {
            "feature_channels": 64,
            "feature_stride": 4,
            "width": 32,
            "groups": 8,
            "trunk_residual_scale": 0.5,
            "baseline_probability": 0.1,
            "vacancy_kernel_size": 3,
            "forward_evidence_transform": (
                "one_sided_zero_anchored_squared_softplus_v2"
            ),
            "backward_surrogate_policy": (
                "negative_half_axis_softplus_recovery_v1"
            ),
            "zero_boundary_policy": (
                "recovery_branch_gradient_half_v1"
            ),
            "resize_policy": "separate_fields_then_final_gate_v1",
        },
        "loss_config": {"dice_weight": 1.0, "epsilon": 0.000001},
        "optimization_budget": {
            "seed": 42,
            "optimizer_updates": 400,
            "steps_per_epoch": 40,
            "factual_miss_states_per_update": 4,
            "factual_no_miss_states_per_update": 4,
            "outcome_pairs_per_update": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "evaluation_chunk_size": 32,
        "optimizer_updates_completed": 400,
        "pretraining_structural_audit": _fake_structural_audit(),
        "structural_checks": _full_structural_checks(),
        "computational_gates": _fake_computational_gates(),
        "initial": initial,
        "final": final,
        "parameters": {
            "trainable_parameter_count": 4385,
            "expected_parameter_count": 4385,
        },
        "forward_budget": {
            "pretraining_structural_audit": audit_budget,
            "initial_evaluation": {
                "calls": 10,
                "state_evaluations": 508,
            },
            "training": {
                "calls": 1200,
                "state_evaluations": 4800,
            },
            "final_evaluation": {
                "calls": 10,
                "state_evaluations": 508,
            },
            "total_excluding_structural_audit": {
                "calls": 1220,
                "state_evaluations": 5816,
            },
            "expected_initial_evaluation": {
                "calls": 10,
                "state_evaluations": 508,
            },
            "expected_training": {
                "calls": 1200,
                "state_evaluations": 4800,
            },
            "expected_final_evaluation": {
                "calls": 10,
                "state_evaluations": 508,
            },
            "expected_total_excluding_structural_audit": {
                "calls": 1220,
                "state_evaluations": 5816,
            },
        },
        "execution_ledger": {
            "backward_calls": 400,
            "optimizer_steps": 400,
        },
        "gradients": {
            "nonfinite_updates": 0,
            "zero_norm_updates": 0,
        },
        "exposure": {
            "identity_null_optimizer_exposure": 0,
            "outcome_pairs": [{} for _ in range(222)],
        },
        "trace": [{} for _ in range(400)],
        "deterministic_runtime": {
            "contract_satisfied": True,
            "flags_restored_after_execution": True,
        },
        "interpretation": {
            "not_detection_performance_evidence": True,
            "does_not_authorize_formal_training": True,
            "does_not_directly_authorize_formal_800": True,
            "eligible_for_frozen_review": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "base_or_backbone_updated": False,
        },
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def _fake_computational_fail_result() -> dict[str, object]:
    result = _fake_core_result()
    initial, final = _fake_snapshots(all_pass=False)
    result["decision"] = "PR_SVEF_BOUNDED_MODEL_CODE_GATE_FAIL"
    result["computational_model_code_gate_pass"] = False
    result["initial"] = initial
    result["final"] = final
    result["computational_gates"] = _fake_computational_gates(
        all_pass=False
    )
    result["interpretation"]["eligible_for_frozen_review"] = False
    return _resign(result, field="result_fingerprint")


def _fake_post_training_structural_fail_result() -> dict[str, object]:
    result = _fake_core_result()
    result["decision"] = "PR_SVEF_STRUCTURAL_EXECUTION_FAIL"
    result["structural_execution_pass"] = False
    result["computational_model_code_gate_pass"] = False
    result["structural_checks"] = _full_structural_checks(all_pass=False)
    result["gradients"]["zero_norm_updates"] = 1
    result["interpretation"]["eligible_for_frozen_review"] = False
    return _resign(result, field="result_fingerprint")


def _fake_pretraining_structural_fail_result() -> dict[str, object]:
    result = _fake_core_result()
    audit = _fake_structural_audit(all_pass=False)
    result.update(
        {
            "decision": "PR_SVEF_STRUCTURAL_EXECUTION_FAIL",
            "structural_execution_pass": False,
            "computational_model_code_gate_pass": False,
            "optimizer_updates_completed": 0,
            "pretraining_structural_audit": audit,
            "structural_checks": dict(audit["checks"]),
            "computational_gates": {
                "status": "NOT_EVALUATED_BY_STRUCTURAL_STOP_RULE",
                "all_pass": None,
            },
            "training_performed": False,
            "forward_budget": {
                "pretraining_structural_audit": audit["compute_budget"],
                "training": {"calls": 0, "state_evaluations": 0},
            },
            "trace": [],
        }
    )
    for field in (
        "initial",
        "final",
        "execution_ledger",
        "exposure",
        "gradients",
        "deterministic_runtime",
    ):
        result.pop(field, None)
    result["interpretation"]["eligible_for_frozen_review"] = False
    return _resign(result, field="result_fingerprint")


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    execute,
    output: Path,
) -> _UnchangedBundle:
    catalog = _fake_catalog()
    prepared = object()
    bundle = _UnchangedBundle()
    population = _fake_population()
    factual_schedule = _fake_factual_schedule(population)
    materializer = _fake_materializer()
    outcome_schedule = _fake_outcome_schedule()
    monkeypatch.setattr(runner, "FROZEN_DEVICE", "cpu")
    monkeypatch.setattr(
        runner,
        "_frozen_output_path",
        lambda: output.resolve(),
    )
    monkeypatch.setattr(
        runner,
        "ANCHOR_POPULATION_FINGERPRINT",
        population.population_fingerprint,
    )
    monkeypatch.setattr(
        runner,
        "MATERIALIZER_FINGERPRINT",
        materializer.materializer_fingerprint,
    )
    monkeypatch.setattr(
        runner,
        "FACTUAL_SCHEDULE_FINGERPRINT",
        factual_schedule.schedule_fingerprint,
    )
    monkeypatch.setattr(
        runner,
        "OUTCOME_SCHEDULE_FINGERPRINT",
        outcome_schedule.schedule_fingerprint,
    )
    authorization = _fake_authorization()
    authorization_path = (
        Path.cwd() / runner.CONFIG_REPO_PATH
    ).resolve()
    monkeypatch.setattr(
        runner,
        "_load_authorization",
        lambda config, implementation: (
            dict(authorization),
            authorization_path,
        ),
    )

    monkeypatch.setattr(
        runner.v3_runner.legacy_runner,
        "_load_real_catalog",
        lambda config: (catalog, prepared, bundle, {}),
    )
    monkeypatch.setattr(
        runner.v3_runner,
        "build_outcome_bounded_anchor_population",
        lambda pair_catalog, prepared_catalog, specification: population,
    )
    monkeypatch.setattr(
        runner.v3_runner,
        "build_outcome_factual_anchor_schedule",
        lambda selected, **kwargs: factual_schedule,
    )
    monkeypatch.setattr(
        runner.v3_runner,
        "build_paired_outcome_input_materializer",
        lambda pair_catalog, prepared_catalog: materializer,
    )
    monkeypatch.setattr(
        runner.v3_runner,
        "build_outcome_pair_schedule",
        lambda pair_catalog, **kwargs: outcome_schedule,
    )
    monkeypatch.setattr(
        runner,
        "execute_recoverable_factorized_outcome_bounded",
        execute,
    )
    return bundle


def _run_args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=(Path.cwd() / runner.CONFIG_REPO_PATH).resolve(),
        device="cpu",
        output=output,
    )


def test_frozen_config_and_proposal_bindings_validate() -> None:
    config_path = (Path.cwd() / runner.CONFIG_REPO_PATH).resolve()
    config = runner._load_config(config_path)
    proposal, proposal_path, design_path = runner._load_proposal(config)

    assert runner.CONFIG_FILE_SHA256 == V6_CONFIG_FILE_SHA256
    assert config["config_fingerprint"] == runner.CONFIG_FINGERPRINT
    assert config["method_id"] == "pr_svef_v6"
    assert config["optimization"]["decoder"] == {
        "feature_channels": 64,
        "feature_stride": 4,
        "width": 32,
        "groups": 8,
        "trunk_residual_scale": 0.5,
        "baseline_probability": 0.1,
        "vacancy_kernel_size": 3,
        "forward_evidence_transform": (
            "one_sided_zero_anchored_squared_softplus_v2"
        ),
        "backward_surrogate_policy": (
            "negative_half_axis_softplus_recovery_v1"
        ),
        "zero_boundary_policy": "recovery_branch_gradient_half_v1",
        "resize_policy": "separate_fields_then_final_gate_v1",
    }
    assert config["pre_run_authorization_contract"] == {
        "required": True,
        "repo_path": runner.AUTHORIZATION_REPO_PATH,
        "schema_version": runner.AUTHORIZATION_SCHEMA,
        "must_bind_bounded_config": True,
        "must_bind_runtime_implementation": True,
        "must_bind_focused_tests": True,
        "must_authorize_exactly_one_real_D_R_run": True,
        "may_authorize_D_V_or_D_T": False,
        "may_authorize_formal_800": False,
    }
    assert proposal["proposal_fingerprint"] == runner.PROPOSAL_FINGERPRINT
    assert proposal["method_id"] == "pr_svef_v6"
    assert proposal_path == (Path.cwd() / runner.PROPOSAL_REPO_PATH).resolve()
    assert design_path.is_file()


def test_config_and_proposal_semantic_tampering_is_rejected() -> None:
    config_path = (Path.cwd() / runner.CONFIG_REPO_PATH).resolve()
    config = runner._load_config(config_path)
    proposal, _, _ = runner._load_proposal(config)

    changed_config = deepcopy(config)
    changed_config["execution_policy"]["D_V_access_allowed"] = True
    with pytest.raises(RuntimeError, match="fingerprint"):
        runner._validate_config_payload(changed_config)

    changed_proposal = deepcopy(proposal)
    changed_proposal["stage_decision"][
        "real_D_R_run_authorized_at_proposal_time"
    ] = True
    with pytest.raises(RuntimeError, match="proposal contract"):
        runner._validate_proposal_payload(changed_proposal, config)


@pytest.mark.parametrize(
    ("message", "error_type"),
    (
        ("authorization receipt missing", FileNotFoundError),
        ("authorization contract changed", RuntimeError),
    ),
)
def test_authorization_failure_precedes_real_catalog_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    message: str,
    error_type: type[Exception],
) -> None:
    output = tmp_path / "not-created"
    real_catalog_calls = 0

    def fail_authorization(*args, **kwargs):
        raise error_type(message)

    def forbidden_real_catalog(*args, **kwargs):
        nonlocal real_catalog_calls
        real_catalog_calls += 1
        raise AssertionError("real catalog must not be loaded")

    monkeypatch.setattr(runner, "FROZEN_DEVICE", "cpu")
    monkeypatch.setattr(
        runner,
        "_frozen_output_path",
        lambda: output.resolve(),
    )
    monkeypatch.setattr(runner, "_load_authorization", fail_authorization)
    monkeypatch.setattr(
        runner.v3_runner.legacy_runner,
        "_load_real_catalog",
        forbidden_real_catalog,
    )

    with pytest.raises(error_type, match=message):
        runner.run(_run_args(output))
    assert real_catalog_calls == 0
    assert not output.exists()


def test_exact_authorization_receipt_validates_without_data_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    implementation = runner._implementation_binding()
    receipt = _exact_authorization_receipt(implementation)
    loaded, path = _load_temporary_authorization(
        monkeypatch,
        tmp_path,
        receipt,
        implementation,
    )

    assert loaded == receipt
    assert path.name == "bounded_code_authorization_receipt.json"
    assert len(
        loaded["runtime_implementation_binding"]["all_runtime_files"]
    ) == 49
    assert set(loaded["focused_tests_binding"]["files"]) == set(
        runner.FOCUSED_TEST_REPO_PATHS
    )
    assert (
        loaded["authorization"]["real_D_R_bounded_execution"]
        is True
    )
    assert loaded["authorization"]["D_V_access_allowed"] is False
    assert loaded["authorization"]["D_T_access_allowed"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "focused_missing_file",
        "focused_command",
        "full_file",
        "wrapper_temperature",
        "wrapped_command",
    ),
)
def test_exact_authorization_rejects_coherently_resigned_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    implementation = runner._implementation_binding()
    receipt = _exact_authorization_receipt(implementation)
    if mutation == "focused_missing_file":
        files = receipt["focused_tests_binding"]["files"]
        files.pop(next(iter(files)))
    elif mutation == "focused_command":
        receipt["focused_tests_binding"]["command"][-1] = (
            "tests/test_wrong.py"
        )
    elif mutation == "full_file":
        files = receipt["full_regression_binding"]["files"]
        first = next(iter(files))
        files[first] = "f" * 64
    elif mutation == "wrapper_temperature":
        receipt["execution_control_binding"][
            "pause_temperature_celsius"
        ] = 83
    elif mutation == "wrapped_command":
        receipt["execution_control_binding"]["wrapped_command"][-1] = (
            "runs/not-the-frozen-output"
        )
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)
    receipt = _resign(receipt)

    with pytest.raises(
        RuntimeError,
        match="authorization contract changed",
    ):
        _load_temporary_authorization(
            monkeypatch,
            tmp_path,
            receipt,
            implementation,
        )


def test_create_only_output_rejects_existing_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(
        runner,
        "_frozen_output_path",
        lambda: existing.resolve(),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        runner._prepare_output(existing)


def test_output_path_single_version_and_device_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    monkeypatch.setattr(
        runner,
        "_frozen_output_path",
        lambda: expected.resolve(),
    )
    with pytest.raises(ValueError, match="frozen r1 output path"):
        runner._prepare_output(tmp_path / "different")

    prior = tmp_path / f"{runner.OUTPUT_VERSION_PREFIX}legacy"
    prior.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        runner._prepare_output(expected)

    assert runner._validate_device("cuda:0") == "cuda:0"
    with pytest.raises(ValueError, match="cuda:0"):
        runner._validate_device("cuda:1")


def test_core_decision_and_review_eligibility_are_fail_closed() -> None:
    wrong_status = _fake_core_result()
    wrong_status["decision"] = "BOUNDED_MODEL_CODE_GATE_PASS"
    with pytest.raises(RuntimeError, match="core result and decision"):
        runner._decision(
            wrong_status,
            failure=None,
            evidence_receipt_fingerprint="a" * 64,
        )

    wrong_review = _fake_core_result()
    wrong_review["interpretation"]["eligible_for_frozen_review"] = False
    unsigned = dict(wrong_review)
    unsigned.pop("result_fingerprint")
    wrong_review["result_fingerprint"] = stable_fingerprint(unsigned)
    with pytest.raises(RuntimeError, match="frozen boundary"):
        runner._verify_core_result(wrong_review)


def test_mocked_success_publishes_all_receipts_and_round_trips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"
    bundle = _patch_runtime(
        monkeypatch,
        execute=lambda *args, **kwargs: _fake_core_result(),
        output=output,
    )
    result = runner.run(_run_args(output))
    published = runner.load_recoverable_factorized_outcome_bounded_artifact(output)

    assert bundle.verify_calls == 1
    assert result["decision"] == "BOUNDED_MODEL_CODE_GATE_PASS"
    assert result["bounded_model_code_gate_pass"] is True
    assert result["directly_authorizes_formal_800"] is False
    assert result["D_V_accessed"] is False
    assert result["D_T_accessed"] is False
    assert published.bounded_model_code_gate_pass is True
    assert published.pair_catalog_fingerprint == PAIR_CATALOG_FINGERPRINT
    assert not (output / ".incomplete").exists()

    names = {path.name for path in (output / "receipts").iterdir()}
    assert names == {
        "anchor_population.json",
        "authorization_binding.json",
        "config_binding.json",
        "decision.json",
        "factual_schedule.json",
        "implementation_binding.json",
        "outcome_inputs.json",
        "outcome_schedule.json",
        "proposal_binding.json",
        "result.json",
        "source_reconstruction.json",
        "toy_gate_binding.json",
    }
    complete = runner._strict_json(
        output / "COMPLETE.json",
        name="test COMPLETE",
    )
    assert complete["artifact_file_count"] == 12
    assert complete["toy_gate_closure_fingerprint"] == (
        runner.TOY_CLOSURE_FINGERPRINT
    )
    assert isinstance(
        complete["authorization_receipt_fingerprint"],
        str,
    )
    assert complete["resume_used"] is False
    assert complete["automatic_retry_performed"] is False
    assert complete["real_D_R_run_count"] == 1
    assert complete["formal_800_training_performed"] is False
    assert complete["performance_evaluation_performed"] is False
    assert complete["D_V_accessed"] is False
    assert complete["D_T_accessed"] is False


@pytest.mark.parametrize(
    ("name", "factory", "expected_decision"),
    (
        (
            "computational-fail",
            _fake_computational_fail_result,
            "BOUNDED_MODEL_CODE_GATE_FAIL",
        ),
        (
            "post-training-structural-fail",
            _fake_post_training_structural_fail_result,
            "STRUCTURAL_EXECUTION_FAIL",
        ),
        (
            "pretraining-structural-fail",
            _fake_pretraining_structural_fail_result,
            "STRUCTURAL_EXECUTION_FAIL",
        ),
    ),
)
def test_completed_nonpass_results_round_trip_without_reclassification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    factory,
    expected_decision: str,
) -> None:
    output = tmp_path / name
    _patch_runtime(
        monkeypatch,
        execute=lambda *args, **kwargs: factory(),
        output=output,
    )
    result = runner.run(_run_args(output))
    published = runner.load_recoverable_factorized_outcome_bounded_artifact(output)
    assert result["decision"] == expected_decision
    assert published.decision == expected_decision
    assert (output / "receipts" / "result.json").is_file()
    assert not (output / "receipts" / "failure.json").exists()


def test_pass_without_exact_gate_evidence_is_rejected() -> None:
    result = _fake_core_result()
    result["computational_gates"] = {
        "all_pass": True,
        "tiny_target_strata": {
            "1_to_3": {},
            "4_to_7": {},
            "8_to_15": {},
            "16_plus": {},
        },
    }
    result = _resign(result, field="result_fingerprint")
    with pytest.raises(RuntimeError, match="computational gates"):
        runner._verify_core_result(result)


def test_coordinated_observed_and_check_change_is_rejected() -> None:
    result = _fake_core_result()
    gates = deepcopy(result["computational_gates"])
    name = "clean_mean_delta_on_D"
    gates["observed"][name] = 0.90
    gates["checks"][name]["value"] = 0.90
    result["computational_gates"] = gates
    result = _resign(result, field="result_fingerprint")

    with pytest.raises(
        RuntimeError,
        match="not exactly bound to snapshots",
    ):
        runner._verify_core_result(result)


def test_execution_error_is_published_as_failure_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic bounded failure")

    output = tmp_path / "failed"
    bundle = _patch_runtime(
        monkeypatch,
        execute=fail,
        output=output,
    )
    result = runner.run(_run_args(output))
    published = runner.load_recoverable_factorized_outcome_bounded_artifact(output)

    assert bundle.verify_calls == 1
    assert result["decision"] == "STRUCTURAL_EXECUTION_ERROR"
    assert result["structural_execution_pass"] is False
    assert result["bounded_model_code_gate_pass"] is False
    assert published.decision == "STRUCTURAL_EXECUTION_ERROR"
    assert (output / "receipts" / "failure.json").is_file()
    assert not (output / "receipts" / "result.json").exists()
    failure = runner._strict_json(
        output / "receipts" / "failure.json",
        name="test failure",
    )
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "synthetic bounded failure"
    assert failure["D_V_accessed"] is False
    assert failure["D_T_accessed"] is False


def test_failed_final_validation_preserves_incomplete_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "invalid-final"
    _patch_runtime(
        monkeypatch,
        execute=lambda *args, **kwargs: _fake_core_result(),
        output=output,
    )

    def fail_final_validation(*args, **kwargs):
        raise RuntimeError("synthetic final validation failure")

    monkeypatch.setattr(
        runner,
        "load_recoverable_factorized_outcome_bounded_artifact",
        fail_final_validation,
    )
    with pytest.raises(RuntimeError, match="final validation failure"):
        runner.run(_run_args(output))
    assert (output / ".incomplete").is_file()
    assert (output / "COMPLETE.json").is_file()


def test_loader_rejects_coherently_resigned_decision_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "coherent-tamper"
    _patch_runtime(
        monkeypatch,
        execute=lambda *args, **kwargs: _fake_core_result(),
        output=output,
    )
    runner.run(_run_args(output))

    decision_path = output / "receipts" / "decision.json"
    complete_path = output / "COMPLETE.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision.update(
        {
            "status": "BOUNDED_MODEL_CODE_GATE_FAIL",
            "bounded_model_code_gate_pass": False,
            "next_action": (
                "preserve_failure_and_revise_model_code_before_training"
            ),
        }
    )
    decision = _resign(decision)
    decision_path.write_text(
        json.dumps(decision, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete.update(
        {
            "decision": "BOUNDED_MODEL_CODE_GATE_FAIL",
            "bounded_model_code_gate_pass": False,
            "decision_fingerprint": decision["receipt_fingerprint"],
        }
    )
    complete["artifact_files"]["receipts/decision.json"] = file_sha256(
        decision_path
    )
    complete = _resign(complete, field="complete_fingerprint")
    complete_path.write_text(
        json.dumps(complete, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="result and publication decision"):
        runner.load_recoverable_factorized_outcome_bounded_artifact(output)


def test_loader_rejects_coherently_resigned_catalog_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "catalog-tamper"
    _patch_runtime(
        monkeypatch,
        execute=lambda *args, **kwargs: _fake_core_result(),
        output=output,
    )
    runner.run(_run_args(output))
    complete_path = output / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["pair_catalog_fingerprint"] = "f" * 64
    complete = _resign(complete, field="complete_fingerprint")
    complete_path.write_text(
        json.dumps(complete, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="decision or evidence binding"):
        runner.load_recoverable_factorized_outcome_bounded_artifact(output)


def test_loader_rejects_a_tampered_published_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "tampered"
    _patch_runtime(
        monkeypatch,
        execute=lambda *args, **kwargs: _fake_core_result(),
        output=output,
    )
    runner.run(_run_args(output))
    decision_path = output / "receipts" / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["status"] = "TAMPERED"
    decision_path.write_text(
        json.dumps(decision, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="fingerprint|hashes"):
        runner.load_recoverable_factorized_outcome_bounded_artifact(output)
