from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from tools import (
    run_coverage_state_bfa_cmif_pmope_bounded_400 as cli,
)


def _dataset_free_stub() -> SimpleNamespace:
    return SimpleNamespace(
        all_pass=True,
        receipt_fingerprint="9" * 64,
        canonical_payload=lambda: {
            "all_pass": True,
            "D_R_accessed": False,
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
        verify_unchanged=lambda root=None: None,
    )


def test_constants_and_static_config_are_singleton_bfa() -> None:
    assert cli.RUN_ID == (
        "cure_lite_bfa_cmif_v20_pmope_bounded_400_r2"
    )
    assert cli.OUTPUT_REPO_PATH == (
        "runs/irstd1k_stage_a_seed42/"
        "cure_lite_bfa_cmif_v20_pmope_bounded_400_r2"
    )
    assert cli.INVALID_R1_RUN_ID.endswith("_r1")
    assert len(cli.INVALID_R1_COMPLETE_FINGERPRINT) == 64
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
    model = config["model"]
    assert config["runtime_splits"] == ["D_R"]
    assert model["class"] == (
        "CURELiteBinaryFlipAntisymmetricLevelSet"
    )
    assert model["input_interface"] == ["F_b", "O"]
    assert model["input_representation"] == "phase_preserving"
    assert model["interaction_policy"] == cli.BFA_INTERACTION_POLICY
    assert model["energy_policy"] == cli.BFA_ENERGY_POLICY
    assert model["flip_policy"] == cli.BFA_FLIP_POLICY
    assert model["objective_suite"] == ["pmope_joint"]
    assert model["candidate_objective"] == "pmope_joint"
    assert model["candidate_objective_policy"] == cli.CSLF_PMOPE_POLICY
    assert model["fixed_margin_hex"] == float(0.225).hex()
    assert model["parameter_count"] == 64064
    assert model["single_completion_field"] is True
    assert model["additional_learned_components"] == 0
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
    correction = config["protocol_correction"]
    assert correction["correction_id"] == (
        "component_null_endpoint_target_exclusion_v1"
    )
    assert correction["predecessor_run_id"] == (
        "cure_lite_bfa_cmif_v20_pmope_bounded_400_r1"
    )
    assert correction["predecessor_scientific_interpretation"] == (
        "invalid_gate_annotation_not_method_result"
    )
    assert correction["model_equation_changed"] is False
    assert correction["objective_changed"] is False
    assert correction["data_population_changed"] is False
    assert correction["seed_changed"] is False
    assert correction["budget_changed"] is False
    assert correction["gate_thresholds_changed"] is False
    assert (
        config["post_training_certificate"][
            "pair_result_is_bounded_gate"
        ]
        is False
    )
    assert config["evidence_scope"]["bounded_400_authorized"] is False
    assert config["evidence_scope"]["formal_800_authorized"] is False


def test_create_only_never_claims_output_or_enters_real_dr(
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
        "run_coverage_state_bfa_dataset_free_gate",
        _dataset_free_stub,
    )
    monkeypatch.setattr(
        cli,
        "verify_repository_coverage_state_uscope_sealed_v18",
        lambda root: _sealed_v18_stub(),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("create-only entered a run-only path")

    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        forbidden,
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_bfa_dr_gate",
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
    assert receipt["D_R_cached_tensor_payload_accessed"] is False
    assert receipt["D_V_accessed"] is False
    assert receipt["D_T_accessed"] is False


def test_cli_exposes_only_create_only_and_run_once() -> None:
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
        ("--validate-create-only", "--run-once"),
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(arguments)


def test_implementation_binding_contains_bfa_and_execution_closure() -> None:
    binding = dict(cli._implementation_binding())
    required = {
        "cure_lite/coverage_state_binary_flip_antisymmetric.py",
        "cure_lite/coverage_state_sobolev.py",
        "cure_lite/train/coverage_state_fused_step.py",
        "cure_lite/experiment/coverage_state_training.py",
        "cure_lite/experiment/coverage_state_bfa_dataset_free.py",
        "cure_lite/experiment/coverage_state_bfa_dr_gate.py",
        "cure_lite/experiment/coverage_state_uscope_sealed_v18.py",
        "cure_lite/experiment/coverage_state_bfa_certificate.py",
        "cure_lite/experiment/coverage_state_bfa_decision.py",
        "cure_lite/experiment/coverage_state_bfa_bounded_runner.py",
        "tools/run_coverage_state_bfa_cmif_pmope_bounded_400.py",
        "tools/run_coverage_state_cmif_pmope_bounded_400.py",
        (
            "tools/"
            "run_coverage_state_cslf_ppce_support_oriented_bounded_400.py"
        ),
        "tools/run_coverage_state_cslf_support_oriented_bounded_400.py",
        "tools/run_with_gpu_temperature_control.py",
    }
    assert required <= set(binding)
    assert all(len(value) == 64 for value in binding.values())


def test_existing_output_stops_before_any_prerequisite(
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
        "run_coverage_state_bfa_dataset_free_gate",
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


def test_checkpoint_is_one_tensor_only_exact_bfa_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=4,
        feature_stride=2,
        width=8,
    )
    model = CURELiteBinaryFlipAntisymmetricLevelSet(config)
    receipt = cli._write_checkpoint_new(
        directory,
        objective="pmope_joint",
        objective_policy=cli.CSLF_PMOPE_POLICY,
        model=model,
    )
    assert receipt["objective"] == "pmope_joint"
    assert receipt["model_class"] == (
        "CURELiteBinaryFlipAntisymmetricLevelSet"
    )
    assert receipt["tensor_only_state_dict"] is True
    assert receipt["weights_only_roundtrip_verified"] is True
    assert receipt["model_config"]["flip_policy"] == (
        cli.BFA_FLIP_POLICY
    )
    assert receipt["model_config"]["fixed_margin_hex"] == (
        float(0.225).hex()
    )
    assert set(path.name for path in directory.iterdir()) == {
        "pmope_joint.safetensors",
        "pmope_joint.checkpoint.json",
    }
    with pytest.raises(ValueError, match="singleton BFA"):
        cli._write_checkpoint_new(
            directory,
            objective="uscope_joint",
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


def test_memory_preflight_uses_sealed_coordinates_without_repack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    cache = SimpleNamespace(
        cache_fingerprint=(
            cli.COVERAGE_STATE_BFA_HISTORICAL_CACHE_FINGERPRINT
        )
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

    def sealed_once(root: Path) -> SimpleNamespace:
        events.append("sealed_v18")
        return sealed

    monkeypatch.setattr(
        cli,
        "run_coverage_state_bfa_dataset_free_gate",
        dataset_free_once,
    )
    monkeypatch.setattr(
        cli,
        "verify_repository_coverage_state_uscope_sealed_v18",
        sealed_once,
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
    build_count = 0

    def build_real(**kwargs: object) -> SimpleNamespace:
        nonlocal build_count
        build_count += 1
        assert build_count == 1
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
        assert value is cache
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
        assert value is population
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
        assert kwargs == {
            "dataset_free_receipt": kwargs[
                "dataset_free_receipt"
            ],
            "real_inputs": real_inputs,
            "bounded_population": population,
            "device": "cuda:0",
        }
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
        "run_coverage_state_bfa_dr_gate",
        run_gate,
    )
    return preflight, real_inputs


def _mock_bounded_result(
    config: CoverageStateBinaryFlipAntisymmetricConfig,
    authorization: SimpleNamespace,
    *,
    passed: bool,
) -> SimpleNamespace:
    model = CURELiteBinaryFlipAntisymmetricLevelSet(config)
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
    certificate = SimpleNamespace(
        integrity_passed=True,
        all_pairs_passed=False,
        receipt_fingerprint="a" * 64,
        verify=lambda: None,
        canonical_payload=lambda: {
            "certificate": "stub",
            "integrity_passed": True,
            "diagnostic_summary": {
                "all_pairs_passed": False,
                "pair_result_is_bounded_gate": False,
            },
        },
    )
    diagnostic = SimpleNamespace(
        result_fingerprint="b" * 64,
        canonical_payload=lambda: {"diagnostic": "stub"},
    )
    zero_decision = SimpleNamespace(
        bounded_gate_passed=passed,
        formal800_eligible=passed,
        failed_checks=() if passed else ("clean_target",),
        decision_fingerprint="c" * 64,
        canonical_payload=lambda: {
            "bounded_gate_passed": passed,
            "same_sign_response_diagnostic": {"is_gate": False},
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
        failed_checks=() if passed else (
            "predeclared_structural_advancement_gate",
        ),
        result_fingerprint="7" * 64,
        canonical_payload=lambda: {
            "result": "stub",
            "bounded_gate_passed": passed,
        },
        verify_unchanged=lambda: None,
    )


@pytest.mark.parametrize("passed", [True, False])
def test_terminal_decision_uses_v20_structure_not_pair_certificate(
    passed: bool,
) -> None:
    config = CoverageStateBinaryFlipAntisymmetricConfig(
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
                "objective": "pmope_joint",
                "receipt_fingerprint": "3" * 64,
            },
        ),
        result_fingerprint="7" * 64,
    )
    assert receipt["status"] == (
        "BFA_CMIF_V20_BOUNDED_400_GATE_PASS"
        if passed
        else "BFA_CMIF_V20_BOUNDED_400_GATE_FAIL"
    )
    assert receipt["bounded_gate_passed"] is passed
    assert receipt[
        "post_training_certificate_integrity_passed"
    ] is True
    assert receipt["pair_certificate_result_is_bounded_gate"] is False
    assert receipt["zero_level_gate_passed"] is passed
    assert receipt["same_sign_response_is_gate"] is False
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
        assert kwargs["sealed_v18_receipt"] is sealed
        return authorization

    monkeypatch.setattr(cli, "_prepare_bfa_authorization", authorize)
    config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    monkeypatch.setattr(
        cli,
        "_expected_bfa_config",
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

    monkeypatch.setattr(cli, "_run_bfa_bounded", run_candidate)
    terminal = cli.run_once()

    assert events == [
        "dataset_free",
        "sealed_v18",
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
        "BFA_CMIF_V20_BOUNDED_400_GATE_PASS"
    )
    assert terminal["bounded_gate_passed"] is True
    assert not (cli.OUTPUT_PATH / ".incomplete").exists()
    assert not (cli.OUTPUT_PATH / "FAILURE.json").exists()
    complete = json.loads(
        (cli.OUTPUT_PATH / "COMPLETE.json").read_text()
    )
    attempt = json.loads(
        (cli.OUTPUT_PATH / "attempt.json").read_text()
    )
    assert attempt["protocol_correction_id"] == (
        cli.PROTOCOL_CORRECTION_ID
    )
    assert attempt["predecessor_run_id"] == cli.INVALID_R1_RUN_ID
    assert attempt["predecessor_is_method_result"] is False
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
    assert complete["pair_certificate_result_is_bounded_gate"] is False
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
        "post_training_certificate.json",
        "preflight.json",
        "sealed_v18_negative_result.json",
        "training.json",
        "zero_level.json",
    }
    assert preflight.training_authorized is True


def test_real_dr_gate_fail_is_complete_stop_without_training(
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
        raise AssertionError("D_R failure entered a later path")

    monkeypatch.setattr(cli, "_prepare_bfa_authorization", forbidden)
    monkeypatch.setattr(cli, "_device_memory_preflight", forbidden)
    monkeypatch.setattr(cli, "_run_bfa_bounded", forbidden)
    terminal = cli.run_once()

    assert events == [
        "dataset_free",
        "sealed_v18",
        "real_inputs",
        "population",
        "preflight",
        "D_R_gate",
    ]
    assert terminal["decision"] == "BFA_CMIF_V20_DR_GATE_FAIL"
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
