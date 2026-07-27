from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from tools import run_coverage_state_cmif_pmope_bounded_400 as cli


def _sealed_v17_stub() -> SimpleNamespace:
    payload = {
        "historical_frozen_controls": True,
        "contemporaneous_controls": False,
        "control_outcomes_are_not_candidate_gates": True,
        "model_deserialization_performed": False,
        "evaluator_called": False,
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    return SimpleNamespace(
        receipt_fingerprint="8" * 64,
        canonical_payload=lambda: payload,
    )


def _dataset_free_stub() -> SimpleNamespace:
    return SimpleNamespace(
        all_pass=True,
        receipt_fingerprint="9" * 64,
        canonical_payload=lambda: {
            "all_pass": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )


def test_constants_and_static_config_are_singleton_pmope() -> None:
    assert cli.RUN_ID == "cure_lite_cmif_v18_pmope_bounded_400_r1"
    assert cli.FROZEN_FEATURE_CHANNELS == 64
    assert cli.FROZEN_FEATURE_STRIDE == 4
    assert cli.FROZEN_MODEL_WIDTH == 32
    assert cli.FROZEN_PARAMETER_COUNT == 64064
    assert cli.FROZEN_SEED == 42
    assert cli.FROZEN_EPOCHS == 10
    assert cli.FROZEN_STEPS_PER_EPOCH == 40
    assert cli.FROZEN_UPDATES_PER_OBJECTIVE == 400
    assert cli.FROZEN_ARTIFACT_FILE_COUNT == 15
    config = cli._static_config_payload(
        source_paths={},
        implementation=(("implementation.py", "1" * 64),),
        dataset_free_receipt_fingerprint="2" * 64,
        sealed_v17_receipt_fingerprint="3" * 64,
    )
    assert config["runtime_splits"] == ["D_R"]
    assert config["model"]["objective_suite"] == ["pmope_joint"]
    assert config["model"]["candidate_objective"] == "pmope_joint"
    assert config["model"]["candidate_objective_policy"] == (
        cli.CSLF_PMOPE_POLICY
    )
    assert config["model"]["fixed_margin_hex"] == float(0.225).hex()
    assert config["model"]["parameter_count"] == 64064
    assert config["budget"] == {
        "seed": 42,
        "epochs": 10,
        "steps_per_epoch": 40,
        "updates_per_objective": 400,
        "objectives": 1,
    }
    assert config["real_D_R_gate"]["status"] == (
        "not_run_in_static_config"
    )
    assert config["evidence_scope"]["bounded_400_authorized"] is False
    assert config["evidence_scope"]["formal_800_authorized"] is False


def test_create_only_never_loads_or_runs_real_d_r(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "OUTPUT_PATH", tmp_path / "unclaimed")
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: (("implementation.py", "1" * 64),),
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_pmope_dataset_free_gate",
        _dataset_free_stub,
    )
    monkeypatch.setattr(
        cli,
        "verify_current_sealed_v17_controls",
        _sealed_v17_stub,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("create-only entered real D_R or run path")

    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        forbidden,
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_pmope_dr_gate",
        forbidden,
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_cmif_pmope_bounded_400",
        forbidden,
    )
    monkeypatch.setattr(cli._v15b_cli, "_claim_output", forbidden)
    receipt = cli.validate_create_only()
    assert not cli.OUTPUT_PATH.exists()
    assert receipt["static_contract_valid"] is True
    assert receipt["D_R_gate_status"] == "not_run"
    assert receipt["D_R_gate_performed"] is False
    assert receipt["bounded_400_authorized"] is False
    assert receipt["authorization_created"] is False
    assert receipt["training_performed"] is False
    assert receipt["output_claimed"] is False
    assert receipt["D_V_accessed"] is False
    assert receipt["D_T_accessed"] is False


def test_cli_has_only_create_only_and_run_once_modes() -> None:
    assert cli.parse_args(
        ("--validate-create-only",)
    ).validate_create_only
    assert cli.parse_args(("--run-once",)).run_once
    with pytest.raises(SystemExit):
        cli.parse_args(())
    with pytest.raises(SystemExit):
        cli.parse_args(("--run-once", "--output", "/tmp/forbidden"))
    with pytest.raises(SystemExit):
        cli.parse_args(("--run-once", "--seed", "43"))


def test_implementation_binding_is_complete_v18_closure() -> None:
    binding = dict(cli._implementation_binding())
    required = {
        "cure_lite/experiment/coverage_state_bounded_runner.py",
        "cure_lite/coverage_state_phase_preserving.py",
        "cure_lite/coverage_state_centered_mixed_interaction.py",
        "cure_lite/experiment/coverage_state_pmope_dataset_free.py",
        "cure_lite/experiment/coverage_state_pmope_dr_gate.py",
        "cure_lite/experiment/coverage_state_pmope_sealed_v17.py",
        "cure_lite/experiment/coverage_state_pmope_bounded_runner.py",
        "tools/audit_coverage_state_cmif_pmope_v18.py",
        "tools/run_coverage_state_cmif_pmope_bounded_400.py",
        (
            "tools/"
            "run_coverage_state_cslf_ppce_support_oriented_bounded_400.py"
        ),
        "tools/run_coverage_state_cslf_support_oriented_bounded_400.py",
        "tools/run_with_gpu_temperature_control.py",
    }
    assert required <= set(binding)
    assert len(binding) == 41
    assert all(len(value) == 64 for value in binding.values())


def test_checkpoint_is_one_tensor_only_pmope_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=4,
        feature_stride=2,
        width=8,
    )
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    receipt = cli._write_checkpoint_new(
        directory,
        objective="pmope_joint",
        objective_policy=cli.CSLF_PMOPE_POLICY,
        model=model,
    )
    assert receipt["objective"] == "pmope_joint"
    assert receipt["tensor_only_state_dict"] is True
    assert receipt["weights_only_roundtrip_verified"] is True
    assert receipt["model_config"]["fixed_margin_hex"] == (
        float(0.225).hex()
    )
    assert tuple(directory.iterdir())
    with pytest.raises(ValueError, match="singleton PMOPE"):
        cli._write_checkpoint_new(
            directory,
            objective="identity_joint",
            objective_policy="wrong",
            model=model,
        )
    with pytest.raises(TypeError, match="exact model class"):
        cli._write_checkpoint_new(
            directory,
            objective="pmope_joint",
            objective_policy=cli.CSLF_PMOPE_POLICY,
            model=object(),  # type: ignore[arg-type]
        )


def test_memory_preflight_uses_one_model_and_frozen_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=4,
        feature_stride=2,
        width=8,
    )
    projected_payload = 1234
    projected = SimpleNamespace(
        resident_tensor_bytes=projected_payload,
        device_cache_fingerprint=(
            cli.COVERAGE_STATE_PMOPE_HISTORICAL_DEVICE_CACHE_FINGERPRINT
        ),
        source_cache_fingerprint="7" * 64,
        verify_unchanged=lambda **kwargs: None,
        memory_report=lambda: {"resident_tensor_bytes": projected_payload},
    )
    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_device_cache",
        lambda cache, *, device: projected,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda index: (32 * 1024**3, 48 * 1024**3),
    )
    receipt = cli._device_memory_preflight(object(), config)
    parameter_bytes = sum(
        value.numel() * value.element_size()
        for value in CURELiteCenteredMixedInteractionLevelSet(
            config
        ).parameters()
    )
    buffer_bytes = sum(
        value.numel() * value.element_size()
        for value in CURELiteCenteredMixedInteractionLevelSet(
            config
        ).buffers()
    )
    assert receipt["model_optimizer_retention_bytes"] == (
        4 * parameter_bytes + buffer_bytes
    )
    assert receipt["checks"]["device_cache_fingerprint_exact"] is True
    assert receipt["checks"]["optimizer_fingerprint_exact"] is True
    assert receipt["all_pass"] is True


def _diagnostic(passed: bool) -> SimpleNamespace:
    return SimpleNamespace(
        factual_miss_gate_passed=passed,
        factual_no_miss_gate_passed=passed,
        clean_defined_metrics_passed=passed,
        clean_compact_support_gate_passed=passed,
        component_null_gate_passed=passed,
        identity_null_gate_passed=passed,
        diagnostic_null_gate_passed=passed,
        bounded_gate_passed=passed,
        canonical_payload=lambda: {
            "bounded_gate_passed": passed,
            "input_representation": "phase_preserving",
        },
    )


def _decision_result(passed: bool) -> SimpleNamespace:
    authorization = SimpleNamespace(
        candidate_objective="pmope_joint",
        sealed_v17_receipt_fingerprint="8" * 64,
    )
    return SimpleNamespace(
        authorization=authorization,
        diagnostic=_diagnostic(passed),
        bounded_gate_passed=passed,
        failed_checks=() if passed else ("candidate_seven_zero_level_gates",),
        result_fingerprint="6" * 64,
        checks=(("candidate_seven_zero_level_gates", passed),),
    )


@pytest.mark.parametrize("passed", [True, False])
def test_decision_is_single_candidate_and_never_authorizes_formal800(
    passed: bool,
) -> None:
    result = _decision_result(passed)
    receipt = cli._decision_payload(
        result,
        (
            {
                "objective": "pmope_joint",
                "receipt_fingerprint": "3" * 64,
            },
        ),
    )
    assert receipt["status"] == (
        "PMOPE_V18_BOUNDED_400_GATE_PASS"
        if passed
        else "PMOPE_V18_BOUNDED_400_GATE_FAIL"
    )
    assert receipt["formal800_eligible"] is passed
    assert receipt["formal_800_authorized"] is False
    assert receipt[
        "historical_control_outcomes_are_candidate_gates"
    ] is False
    with pytest.raises(ValueError, match="one candidate checkpoint"):
        cli._decision_payload(result, ())


def test_existing_output_stops_before_any_prerequisite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-claimed"
    output.mkdir()
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("prerequisite ran after output existed")

    monkeypatch.setattr(cli, "_verify_frozen_sources", forbidden)
    monkeypatch.setattr(
        cli,
        "run_coverage_state_pmope_dataset_free_gate",
        forbidden,
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_pmope_dr_gate",
        forbidden,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        cli.run_once()


def _patch_run_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_PATH", tmp_path / "bounded")
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: (("implementation.py", "1" * 64),),
    )
    monkeypatch.setattr(
        cli._v15b_cli,
        "_verify_runtime_contract",
        lambda: {
            "device": "cuda:0",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )
    dataset_free = _dataset_free_stub()
    sealed = _sealed_v17_stub()
    monkeypatch.setattr(
        cli,
        "run_coverage_state_pmope_dataset_free_gate",
        lambda: dataset_free,
    )
    monkeypatch.setattr(
        cli,
        "verify_current_sealed_v17_controls",
        lambda: sealed,
    )
    monkeypatch.setattr(
        cli,
        "_static_config_payload",
        lambda **kwargs: {
            "schema_version": cli.RUN_SCHEMA,
            "objective_suite": ["pmope_joint"],
        },
    )
    return dataset_free, sealed


def _patch_real_dr_objects(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gate_passed: bool,
) -> tuple[SimpleNamespace, SimpleNamespace, dict[str, int]]:
    cache = SimpleNamespace()
    real_inputs = SimpleNamespace(
        scalar_cache=cache,
        source_binding=SimpleNamespace(
            canonical_payload=lambda: {"binding": "stub"}
        ),
        canonical_payload=lambda: {"real_inputs": "stub"},
        verify_unchanged=lambda: None,
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        lambda **kwargs: real_inputs,
    )
    population = SimpleNamespace(
        cache=cache,
        population_fingerprint="2" * 64,
        canonical_payload=lambda: {"population": "stub"},
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_bounded_population",
        lambda value, *, seed: population,
    )
    schedule = SimpleNamespace(
        selections=(),
        canonical_payload=lambda: {"schedule": "stub"},
    )
    preflight = SimpleNamespace(
        population=population,
        schedule=schedule,
        training_authorized=True,
        canonical_payload=lambda: {"preflight": "stub"},
    )
    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_bounded_preflight",
        lambda value: preflight,
    )
    monkeypatch.setattr(
        cli,
        "coverage_state_schedule_exposure_report",
        lambda *args: {"exposure": "stub"},
    )
    counters = {"D_R_gate": 0, "training": 0}

    def run_gate(**kwargs: object) -> SimpleNamespace:
        counters["D_R_gate"] += 1
        return SimpleNamespace(
            checks=(("real_D_R_gate", gate_passed),),
            receipt_fingerprint="4" * 64,
            all_pass=gate_passed,
            canonical_payload=lambda: {
                "all_pass": gate_passed,
                "runtime_splits": ["D_R"],
            },
            verify_unchanged=lambda: None,
        )

    monkeypatch.setattr(
        cli,
        "run_coverage_state_pmope_dr_gate",
        run_gate,
    )
    return preflight, real_inputs, counters


def test_mocked_run_once_writes_complete_singleton_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, sealed = _patch_run_prerequisites(tmp_path, monkeypatch)
    preflight, _, counters = _patch_real_dr_objects(
        monkeypatch,
        gate_passed=True,
    )
    authorization = SimpleNamespace(
        candidate_objective="pmope_joint",
        sealed_v17_receipt=sealed,
        sealed_v17_receipt_fingerprint=sealed.receipt_fingerprint,
        authorization_fingerprint="3" * 64,
        training_authorized=True,
        canonical_payload=lambda: {"authorization": "stub"},
        verify_unchanged=lambda: None,
    )
    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_pmope_bounded_run_authorization",
        lambda *args, **kwargs: authorization,
    )
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    monkeypatch.setattr(
        cli,
        "expected_coverage_state_pmope_config",
        lambda value: config,
    )
    monkeypatch.setattr(
        cli,
        "_device_memory_preflight",
        lambda *args: {
            "schema_version": "memory",
            "receipt_fingerprint": "5" * 64,
            "all_pass": True,
        },
    )
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    training = SimpleNamespace(
        results=(
            SimpleNamespace(
                objective="pmope_joint",
                objective_policy=cli.CSLF_PMOPE_POLICY,
            ),
        ),
        models=(("pmope_joint", model),),
        result_fingerprint="6" * 64,
        canonical_payload=lambda: {"training": "stub"},
    )
    result = SimpleNamespace(
        authorization=authorization,
        training=training,
        diagnostic=_diagnostic(True),
        checks=(("candidate_seven_zero_level_gates", True),),
        bounded_gate_passed=True,
        failed_checks=(),
        result_fingerprint="7" * 64,
        canonical_payload=lambda: {"result": "stub"},
        verify_unchanged=lambda: None,
    )

    def run_candidate(*args: object, **kwargs: object) -> object:
        counters["training"] += 1
        return result

    monkeypatch.setattr(
        cli,
        "run_coverage_state_cmif_pmope_bounded_400",
        run_candidate,
    )
    terminal = cli.run_once()
    assert counters == {"D_R_gate": 1, "training": 1}
    assert terminal["decision"] == "PMOPE_V18_BOUNDED_400_GATE_PASS"
    assert terminal["bounded_gate_passed"] is True
    assert not (cli.OUTPUT_PATH / ".incomplete").exists()
    assert not (cli.OUTPUT_PATH / "FAILURE.json").exists()
    complete = json.loads(
        (cli.OUTPUT_PATH / "COMPLETE.json").read_text()
    )
    assert complete["artifact_file_count"] == 15
    assert len(complete["artifact_files"]) == 15
    assert complete["formal800_eligible"] is True
    assert complete["formal_800_authorized"] is False
    assert set(
        path.name
        for path in (cli.OUTPUT_PATH / "checkpoints").iterdir()
    ) == {
        "pmope_joint.safetensors",
        "pmope_joint.checkpoint.json",
    }
    assert set(
        path.name
        for path in (cli.OUTPUT_PATH / "receipts").iterdir()
    ) == {
        "authorization.json",
        "bounded_result.json",
        "config.json",
        "dataset_free.json",
        "decision.json",
        "device_memory_preflight.json",
        "dr_gate.json",
        "inputs.json",
        "preflight.json",
        "sealed_v17_controls.json",
        "training.json",
        "zero_level.json",
    }
    assert preflight.training_authorized is True


def test_real_dr_gate_fail_is_complete_stop_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_prerequisites(tmp_path, monkeypatch)
    _, _, counters = _patch_real_dr_objects(
        monkeypatch,
        gate_passed=False,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("D_R gate failure entered training path")

    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_pmope_bounded_run_authorization",
        forbidden,
    )
    monkeypatch.setattr(cli, "_device_memory_preflight", forbidden)
    monkeypatch.setattr(
        cli,
        "run_coverage_state_cmif_pmope_bounded_400",
        forbidden,
    )
    terminal = cli.run_once()
    assert counters == {"D_R_gate": 1, "training": 0}
    assert terminal["decision"] == "PMOPE_V18_DR_GATE_FAIL"
    assert terminal["bounded_gate_passed"] is False
    assert not (cli.OUTPUT_PATH / ".incomplete").exists()
    assert not (cli.OUTPUT_PATH / "FAILURE.json").exists()
    complete = json.loads(
        (cli.OUTPUT_PATH / "COMPLETE.json").read_text()
    )
    assert complete["artifact_file_count"] == 8
    assert complete["authorization_created"] is False
    assert complete["bounded_training_performed"] is False
    assert complete["checkpoint_count"] == 0
    assert complete["formal_800_authorized"] is False
    assert not any((cli.OUTPUT_PATH / "checkpoints").iterdir())


def test_execution_exception_writes_nonresumable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_prerequisites(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected D_R construction error")
        ),
    )
    with pytest.raises(RuntimeError, match="injected"):
        cli.run_once()
    assert (cli.OUTPUT_PATH / ".incomplete").is_file()
    assert not (cli.OUTPUT_PATH / "COMPLETE.json").exists()
    failure = json.loads(
        (cli.OUTPUT_PATH / "FAILURE.json").read_text()
    )
    assert failure["status"] == "failed_incomplete_attempt"
    assert failure["resume_allowed"] is False
    assert failure["automatic_retry_allowed"] is False
    assert failure["formal_800_authorized"] is False
    assert failure["D_V_accessed"] is False
    assert failure["D_T_accessed"] is False
