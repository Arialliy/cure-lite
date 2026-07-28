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
from tools import run_coverage_state_cmif_uscope_bounded_400 as cli


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


def _sealed_v18_stub() -> SimpleNamespace:
    payload = {
        "historical_negative_result": True,
        "contemporaneous_candidate_result": False,
        "checkpoint_treated_as_opaque_bytes": True,
        "model_deserialization_performed": False,
        "evaluator_called": False,
        "training_performed": False,
        "D_R_cached_tensor_payload_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "runtime_splits": [],
    }
    return SimpleNamespace(
        receipt_fingerprint="8" * 64,
        canonical_payload=lambda: payload,
        verify_unchanged=lambda root: None,
    )


def test_constants_and_static_config_are_singleton_uscope() -> None:
    assert cli.RUN_ID == "cure_lite_cmif_v19_uscope_bounded_400_r1"
    assert cli.FROZEN_FEATURE_CHANNELS == 64
    assert cli.FROZEN_FEATURE_STRIDE == 4
    assert cli.FROZEN_MODEL_WIDTH == 32
    assert cli.FROZEN_PARAMETER_COUNT == 64064
    assert cli.FROZEN_SEED == 42
    assert cli.FROZEN_EPOCHS == 10
    assert cli.FROZEN_STEPS_PER_EPOCH == 40
    assert cli.FROZEN_UPDATES_PER_OBJECTIVE == 400
    assert cli.FROZEN_ARTIFACT_FILE_COUNT == 16
    assert cli.FROZEN_DEVICE == "cuda:0"
    assert cli.FROZEN_VISIBLE_GPU == "0"
    assert cli.FROZEN_PAUSE_TEMPERATURE_C == 82
    assert cli.FROZEN_RESUME_TEMPERATURE_C == 75
    config = cli._static_config_payload(
        source_paths={},
        implementation=(("implementation.py", "1" * 64),),
        dataset_free_receipt_fingerprint="2" * 64,
        sealed_v18_receipt_fingerprint="3" * 64,
    )
    assert config["runtime_splits"] == ["D_R"]
    assert config["model"]["objective_suite"] == ["uscope_joint"]
    assert config["model"]["candidate_objective"] == "uscope_joint"
    assert config["model"]["candidate_objective_policy"] == (
        cli.CSLF_USCOPE_POLICY
    )
    assert config["model"]["fixed_margin_hex"] == float(0.225).hex()
    assert config["model"]["same_sign_response_is_gate"] is False
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
    assert config["post_training_certificate"]["run_once_only"] is True
    assert config["evidence_scope"]["bounded_400_authorized"] is False
    assert config["evidence_scope"]["formal_800_authorized"] is False


def test_create_only_never_enters_real_dr_or_run_path(
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
        "run_coverage_state_uscope_dataset_free_gate",
        _dataset_free_stub,
    )
    monkeypatch.setattr(
        cli,
        "verify_repository_coverage_state_uscope_sealed_v18",
        lambda root: _sealed_v18_stub(),
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
        "run_coverage_state_uscope_dr_gate",
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
    assert receipt["post_training_certificate_performed"] is False
    assert receipt["zero_level_evaluation_performed"] is False
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
    for arguments in (
        ("--run-once", "--output", "/tmp/forbidden"),
        ("--run-once", "--seed", "43"),
        ("--run-once", "--updates", "401"),
        ("--run-once", "--resume"),
        ("--run-once", "--retry"),
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(arguments)


def test_implementation_binding_is_complete_v19_closure() -> None:
    binding = dict(cli._implementation_binding())
    required = {
        "cure_lite/coverage_state_centered_mixed_interaction.py",
        "cure_lite/coverage_state_supremal_projection.py",
        "cure_lite/train/coverage_state_fused_step.py",
        "cure_lite/experiment/coverage_state_training.py",
        "cure_lite/experiment/coverage_state_uscope_dataset_free.py",
        "cure_lite/experiment/coverage_state_uscope_dr_gate.py",
        "cure_lite/experiment/coverage_state_uscope_sealed_v18.py",
        "cure_lite/experiment/coverage_state_uscope_certificate.py",
        "cure_lite/experiment/coverage_state_uscope_decision.py",
        "cure_lite/experiment/coverage_state_uscope_bounded_runner.py",
        "tools/run_coverage_state_cmif_pmope_bounded_400.py",
        "tools/run_coverage_state_cmif_uscope_bounded_400.py",
        (
            "tools/"
            "run_coverage_state_cslf_ppce_support_oriented_bounded_400.py"
        ),
        "tools/run_coverage_state_cslf_support_oriented_bounded_400.py",
        "tools/run_with_gpu_temperature_control.py",
    }
    assert required <= set(binding)
    assert len(binding) == 44
    assert all(len(value) == 64 for value in binding.values())


def test_existing_output_stops_before_dataset_free_or_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-claimed"
    output.mkdir()
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("prerequisite ran after output existed")

    monkeypatch.setattr(
        cli,
        "run_coverage_state_uscope_dataset_free_gate",
        forbidden,
    )
    monkeypatch.setattr(cli, "_verify_frozen_sources", forbidden)
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        forbidden,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        cli.run_once()


def test_checkpoint_is_one_tensor_only_uscope_model(
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
        objective="uscope_joint",
        objective_policy=cli.CSLF_USCOPE_POLICY,
        model=model,
    )
    assert receipt["objective"] == "uscope_joint"
    assert receipt["tensor_only_state_dict"] is True
    assert receipt["weights_only_roundtrip_verified"] is True
    assert receipt["model_config"]["fixed_margin_hex"] == (
        float(0.225).hex()
    )
    assert set(path.name for path in directory.iterdir()) == {
        "uscope_joint.safetensors",
        "uscope_joint.checkpoint.json",
    }
    with pytest.raises(ValueError, match="singleton USCOPE"):
        cli._write_checkpoint_new(
            directory,
            objective="pmope_joint",
            objective_policy="wrong",
            model=model,
        )
    with pytest.raises(TypeError, match="exact model class"):
        cli._write_checkpoint_new(
            directory,
            objective="uscope_joint",
            objective_policy=cli.CSLF_USCOPE_POLICY,
            model=object(),  # type: ignore[arg-type]
        )


def test_memory_preflight_uses_sealed_static_budget_without_repack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    cache = SimpleNamespace(
        cache_fingerprint=cli._FROZEN_SOURCE_CACHE_FINGERPRINT
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

    receipt = cli._device_memory_preflight(cache, config)

    assert receipt["projected_device_cache"] == {
        "resident_tensor_bytes": 205_521_408,
        "binding_mode": "sealed_v18_static_budget_no_runtime_repack",
        "runtime_pack_count": 0,
    }
    assert receipt["model_parameter_bytes"] == 64064 * 4
    assert receipt["model_buffer_bytes"] == 0
    assert receipt["checks"]["source_cache_fingerprint_exact"] is True
    assert receipt["checks"]["device_cache_fingerprint_exact"] is True
    assert receipt["checks"]["optimizer_fingerprint_exact"] is True
    assert receipt["all_pass"] is True


def _patch_run_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
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
    sealed = _sealed_v18_stub()

    def dataset_free_once() -> SimpleNamespace:
        events.append("dataset_free")
        return dataset_free

    monkeypatch.setattr(
        cli,
        "run_coverage_state_uscope_dataset_free_gate",
        dataset_free_once,
    )
    monkeypatch.setattr(
        cli,
        "verify_repository_coverage_state_uscope_sealed_v18",
        lambda root: sealed,
    )
    monkeypatch.setattr(
        cli,
        "_static_config_payload",
        lambda **kwargs: {
            "schema_version": cli.RUN_SCHEMA,
            "objective_suite": ["uscope_joint"],
        },
    )
    return dataset_free, sealed


def _patch_real_dr_objects(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    gate_passed: bool,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    cache = SimpleNamespace(cache_fingerprint="2" * 64)
    real_inputs = SimpleNamespace(
        scalar_cache=cache,
        source_binding=SimpleNamespace(
            canonical_payload=lambda: {"binding": "stub"}
        ),
        canonical_payload=lambda: {"real_inputs": "stub"},
        verify_unchanged=lambda: None,
    )

    def build_real(**kwargs: object) -> SimpleNamespace:
        events.append("real_inputs")
        return real_inputs

    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        build_real,
    )
    population = SimpleNamespace(
        cache=cache,
        population_fingerprint="2" * 64,
        canonical_payload=lambda: {"population": "stub"},
    )

    def build_population(value: object, *, seed: int) -> SimpleNamespace:
        events.append("population")
        assert seed == 42
        return population

    monkeypatch.setattr(
        cli,
        "build_coverage_state_bounded_population",
        build_population,
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

    def build_preflight(value: object) -> SimpleNamespace:
        events.append("preflight")
        return preflight

    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_bounded_preflight",
        build_preflight,
    )
    monkeypatch.setattr(
        cli,
        "coverage_state_schedule_exposure_report",
        lambda *args: {"exposure": "stub"},
    )

    def run_gate(**kwargs: object) -> SimpleNamespace:
        events.append("D_R_gate")
        return SimpleNamespace(
            all_pass=gate_passed,
            evidence_fingerprint="4" * 64,
            canonical_payload=lambda: {
                "all_pass": gate_passed,
                "runtime_splits": ["D_R"],
            },
        )

    monkeypatch.setattr(
        cli,
        "run_coverage_state_uscope_dr_gate",
        run_gate,
    )
    return preflight, real_inputs


def _mock_bounded_result(
    config: CoverageStateCenteredMixedInteractionConfig,
    authorization: SimpleNamespace,
    *,
    passed: bool,
) -> SimpleNamespace:
    model = CURELiteCenteredMixedInteractionLevelSet(config)
    training = SimpleNamespace(
        results=(
            SimpleNamespace(
                objective="uscope_joint",
                objective_policy=cli.CSLF_USCOPE_POLICY,
            ),
        ),
        models=(("uscope_joint", model),),
        result_fingerprint="6" * 64,
        canonical_payload=lambda: {"training": "stub"},
    )
    certificate = SimpleNamespace(
        gate_passed=passed,
        all_pass=passed,
        receipt_fingerprint="a" * 64,
        verify=lambda: None,
        canonical_payload=lambda: {
            "certificate": "stub",
            "all_pass": passed,
        },
    )
    diagnostic = SimpleNamespace(
        result_fingerprint="b" * 64,
        canonical_payload=lambda: {"diagnostic": "stub"},
    )
    zero_decision = SimpleNamespace(
        zero_level_gate_passed=passed,
        decision_fingerprint="c" * 64,
        canonical_payload=lambda: {
            "zero_level_gate_passed": passed,
            "same_sign_response_is_gate": False,
        },
    )
    return SimpleNamespace(
        authorization=authorization,
        training=training,
        certificate=certificate,
        diagnostic=diagnostic,
        decision=zero_decision,
        training_invocations=1,
        certificate_invocations=1,
        zero_level_evaluation_invocations=1,
        bounded_gate_passed=passed,
        failed_checks=() if passed else ("post_training_certificate",),
        result_fingerprint="7" * 64,
        canonical_payload=lambda: {
            "result": "stub",
            "bounded_gate_passed": passed,
        },
        verify_unchanged=lambda: None,
    )


@pytest.mark.parametrize("passed", [True, False])
def test_terminal_decision_uses_certificate_and_zero_level_without_formal800(
    passed: bool,
) -> None:
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    authorization = SimpleNamespace(
        sealed_v18_receipt_fingerprint="8" * 64,
    )
    result = _mock_bounded_result(
        config,
        authorization,
        passed=passed,
    )
    receipt = cli._decision_payload(
        result,
        (
            {
                "objective": "uscope_joint",
                "receipt_fingerprint": "3" * 64,
            },
        ),
        result_fingerprint="7" * 64,
    )
    assert receipt["status"] == (
        "USCOPE_V19_BOUNDED_400_GATE_PASS"
        if passed
        else "USCOPE_V19_BOUNDED_400_GATE_FAIL"
    )
    assert receipt["bounded_gate_passed"] is passed
    assert receipt["post_training_certificate_passed"] is passed
    assert receipt["zero_level_gate_passed"] is passed
    assert receipt["same_sign_response_is_gate"] is False
    assert receipt["training_invocations"] == 1
    assert receipt["certificate_invocations"] == 1
    assert receipt["zero_level_evaluation_invocations"] == 1
    assert receipt["formal800_eligible"] is passed
    assert receipt["formal_800_authorized"] is False
    with pytest.raises(ValueError, match="one candidate checkpoint"):
        cli._decision_payload(
            result,
            (),
            result_fingerprint="7" * 64,
        )


def test_mocked_run_once_writes_complete_singleton_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _, sealed = _patch_run_prerequisites(
        tmp_path,
        monkeypatch,
        events,
    )
    preflight, _ = _patch_real_dr_objects(
        monkeypatch,
        events,
        gate_passed=True,
    )
    authorization = SimpleNamespace(
        sealed_v18_receipt_fingerprint=sealed.receipt_fingerprint,
        authorization_fingerprint="3" * 64,
        training_authorized=True,
        canonical_payload=lambda: {"authorization": "stub"},
    )

    def authorize(*args: object, **kwargs: object) -> SimpleNamespace:
        events.append("authorization")
        return authorization

    monkeypatch.setattr(cli, "_prepare_uscope_authorization", authorize)
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    monkeypatch.setattr(
        cli,
        "_expected_uscope_config",
        lambda value: config,
    )

    def memory(*args: object) -> dict[str, object]:
        events.append("memory_preflight")
        return {
            "schema_version": "memory",
            "receipt_fingerprint": "5" * 64,
            "all_pass": True,
        }

    monkeypatch.setattr(cli, "_device_memory_preflight", memory)
    result = _mock_bounded_result(
        config,
        authorization,
        passed=True,
    )

    def run_candidate(*args: object, **kwargs: object) -> object:
        events.extend(("training", "certificate", "zero_level"))
        return result

    monkeypatch.setattr(cli, "_run_uscope_bounded", run_candidate)
    terminal = cli.run_once()

    assert events == [
        "dataset_free",
        "real_inputs",
        "population",
        "preflight",
        "D_R_gate",
        "authorization",
        "memory_preflight",
        "training",
        "certificate",
        "zero_level",
    ]
    assert terminal["decision"] == (
        "USCOPE_V19_BOUNDED_400_GATE_PASS"
    )
    assert terminal["bounded_gate_passed"] is True
    assert not (cli.OUTPUT_PATH / ".incomplete").exists()
    assert not (cli.OUTPUT_PATH / "FAILURE.json").exists()
    complete = json.loads(
        (cli.OUTPUT_PATH / "COMPLETE.json").read_text()
    )
    assert complete["artifact_file_count"] == 16
    assert len(complete["artifact_files"]) == 16
    assert complete["dataset_free_invocations"] == 1
    assert complete["real_inputs_construction_invocations"] == 1
    assert complete["population_construction_invocations"] == 1
    assert complete["preflight_invocations"] == 1
    assert complete["D_R_gate_invocations"] == 1
    assert complete["training_invocations"] == 1
    assert complete["post_training_certificate_invocations"] == 1
    assert complete["zero_level_evaluation_invocations"] == 1
    assert complete["formal800_eligible"] is True
    assert complete["formal_800_authorized"] is False
    assert set(
        path.name
        for path in (cli.OUTPUT_PATH / "checkpoints").iterdir()
    ) == {
        "uscope_joint.safetensors",
        "uscope_joint.checkpoint.json",
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
        "post_training_certificate.json",
        "preflight.json",
        "sealed_v18_negative_result.json",
        "training.json",
        "zero_level.json",
    }
    assert preflight.training_authorized is True


def test_real_dr_gate_fail_is_complete_stop_without_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _patch_run_prerequisites(tmp_path, monkeypatch, events)
    _patch_real_dr_objects(
        monkeypatch,
        events,
        gate_passed=False,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("D_R gate failure entered later path")

    monkeypatch.setattr(cli, "_prepare_uscope_authorization", forbidden)
    monkeypatch.setattr(cli, "_device_memory_preflight", forbidden)
    monkeypatch.setattr(cli, "_run_uscope_bounded", forbidden)
    terminal = cli.run_once()

    assert events == [
        "dataset_free",
        "real_inputs",
        "population",
        "preflight",
        "D_R_gate",
    ]
    assert terminal["decision"] == "USCOPE_V19_DR_GATE_FAIL"
    assert terminal["bounded_gate_passed"] is False
    assert not (cli.OUTPUT_PATH / ".incomplete").exists()
    assert not (cli.OUTPUT_PATH / "FAILURE.json").exists()
    complete = json.loads(
        (cli.OUTPUT_PATH / "COMPLETE.json").read_text()
    )
    assert complete["artifact_file_count"] == 8
    assert complete["authorization_created"] is False
    assert complete["bounded_training_performed"] is False
    assert complete["post_training_certificate_performed"] is False
    assert complete["zero_level_evaluation_performed"] is False
    assert complete["checkpoint_count"] == 0
    assert complete["formal_800_authorized"] is False
    assert not any((cli.OUTPUT_PATH / "checkpoints").iterdir())


def test_execution_exception_writes_nonresumable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _patch_run_prerequisites(tmp_path, monkeypatch, events)
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


def test_main_emits_one_canonical_json_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "validate_create_only",
        lambda: {"status": "validated"},
    )
    assert cli.main(("--validate-create-only",)) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "validated"
    }
