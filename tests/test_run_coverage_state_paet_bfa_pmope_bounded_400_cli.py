from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from tools import (
    run_coverage_state_paet_bfa_pmope_bounded_400 as cli,
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


def _sealed_v20_stub() -> SimpleNamespace:
    payload = {
        "run_id": "historical-v20-r2",
        "read_only": True,
        "training_performed": False,
        "reevaluated": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "resource_reference": {
            "measured_reference_available": False,
        },
    }
    return SimpleNamespace(
        reference_fingerprint="8" * 64,
        canonical_payload=lambda: payload,
    )


def test_constants_and_static_config_are_singleton_paet() -> None:
    assert cli.RUN_ID == (
        "cure_lite_paet_bfa_v21_pmope_bounded_400_r1"
    )
    assert cli.OUTPUT_REPO_PATH.endswith(cli.RUN_ID)
    assert cli.FROZEN_FEATURE_CHANNELS == 64
    assert cli.FROZEN_FEATURE_STRIDE == 4
    assert cli.FROZEN_MODEL_WIDTH == 32
    assert cli.FROZEN_PARAMETER_COUNT == 64064
    assert cli.FROZEN_SEED == 42
    assert cli.FROZEN_EPOCHS == 10
    assert cli.FROZEN_STEPS_PER_EPOCH == 40
    assert cli.FROZEN_UPDATES_PER_OBJECTIVE == 400
    assert cli.FROZEN_ARTIFACT_FILE_COUNT == 17
    assert cli.FROZEN_DEVICE == "cuda:0"
    assert cli.FROZEN_VISIBLE_GPU == "0"
    assert cli.FROZEN_PAUSE_TEMPERATURE_C == 82
    assert cli.FROZEN_RESUME_TEMPERATURE_C == 75

    config = cli._static_config_payload(
        source_paths={},
        implementation=(("implementation.py", "1" * 64),),
        dataset_free_receipt_fingerprint="2" * 64,
        sealed_v20_reference_fingerprint="3" * 64,
    )
    model = config["model"]
    assert config["run_id"] == cli.RUN_ID
    assert config["runtime_splits"] == ["D_R"]
    assert model["class"] == (
        "CURELitePhaseAlignedEvidenceTransportLevelSet"
    )
    assert model["candidate"] == "PAET-BFA"
    assert model["input_interface"] == ["F_b", "O"]
    assert model["input_representation"] == "phase_preserving"
    assert model["field_policy"] == cli.CSLF_PAET_FIELD_POLICY
    assert model["equation_policy"] == cli.CSLF_PAET_EQUATION_POLICY
    assert model["flip_policy"] == cli.CSLF_PAET_FLIP_POLICY
    assert model["transport_policy"] == cli.CSLF_PAET_TRANSPORT_POLICY
    assert model["objective_suite"] == ["pmope_joint"]
    assert model["candidate_objective_policy"] == cli.CSLF_PMOPE_POLICY
    assert model["fixed_margin_hex"] == float(0.225).hex()
    assert model["parameter_count"] == 64064
    assert model["parameter_tensor_count"] == 3
    assert config["budget"] == {
        "seed": 42,
        "epochs": 10,
        "steps_per_epoch": 40,
        "updates_per_objective": 400,
        "objectives": 1,
    }
    decision = config["bounded_decision"]
    assert decision["clean_target_negative_minimum"] == [124, 149]
    assert decision["clean_outside_completion_maximum"] == 46
    assert decision["factual_recovered_required"] == [16, 16]
    assert decision["factual_strict_minimum"] == [14, 16]
    assert decision["factual_target_negative_minimum"] == [310, 335]
    assert decision["component_null_required"] == [16, 16]
    resource = config["resource_measurement"]
    assert resource["v20_measured_reference_available"] is False
    assert resource["v20_comparison_status"] == (
        "NOT_EVALUATED_NO_MATCHED_V20_MEASUREMENT"
    )
    assert resource["ratio_is_scientific_gate"] is False
    assert config["evidence_scope"]["formal_800_authorized"] is False


def test_create_only_never_claims_or_enters_real_dr(
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
        "run_coverage_state_paet_dataset_free_gate",
        _dataset_free_stub,
    )
    monkeypatch.setattr(
        cli,
        "verify_repository_coverage_state_bfa_v20_reference",
        lambda root: _sealed_v20_stub(),
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
        "run_coverage_state_paet_dr_gate",
        forbidden,
    )
    monkeypatch.setattr(cli._v15b_cli, "_claim_output", forbidden)
    receipt = cli.validate_create_only()
    replay = cli.validate_create_only()

    assert not cli.OUTPUT_PATH.exists()
    assert replay == receipt
    assert replay["receipt_fingerprint"] == receipt["receipt_fingerprint"]
    assert receipt["run_id"] == cli.RUN_ID
    assert receipt["static_contract_valid"] is True
    assert receipt["D_R_gate_status"] == "not_run"
    assert receipt["D_R_gate_performed"] is False
    assert receipt["authorization_created"] is False
    assert receipt["training_performed"] is False
    assert receipt["resource_measurement_performed"] is False
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


def test_implementation_binding_contains_paet_execution_closure() -> None:
    binding = dict(cli._implementation_binding())
    required = {
        "cure_lite/coverage_state_phase_aligned_evidence_transport.py",
        "cure_lite/coverage_state_sobolev.py",
        "cure_lite/train/coverage_state_fused_step.py",
        "cure_lite/experiment/coverage_state_training.py",
        "cure_lite/experiment/coverage_state_paet_dataset_free.py",
        "cure_lite/experiment/coverage_state_paet_dr_gate.py",
        "cure_lite/experiment/coverage_state_paet_certificate.py",
        "cure_lite/experiment/coverage_state_paet_decision.py",
        "cure_lite/experiment/coverage_state_paet_bounded_runner.py",
        "tools/run_coverage_state_paet_bfa_pmope_bounded_400.py",
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
        "run_coverage_state_paet_dataset_free_gate",
        forbidden,
    )
    monkeypatch.setattr(cli, "_verify_frozen_sources", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.run_once()


def test_checkpoint_is_tensor_only_exact_paet_and_run_id_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(
        CoverageStatePhaseAlignedEvidenceTransportConfig(
            feature_channels=4,
            feature_stride=2,
            width=8,
        )
    )
    receipt = cli._write_checkpoint_new(
        directory,
        objective="pmope_joint",
        objective_policy=cli.CSLF_PMOPE_POLICY,
        model=model,
    )
    assert receipt["run_id"] == cli.RUN_ID
    assert receipt["model_class"] == (
        "CURELitePhaseAlignedEvidenceTransportLevelSet"
    )
    assert receipt["tensor_only_state_dict"] is True
    assert receipt["weights_only_roundtrip_verified"] is True
    assert receipt["model_config"]["transport_policy"] == (
        cli.CSLF_PAET_TRANSPORT_POLICY
    )
    assert set(path.name for path in directory.iterdir()) == {
        "pmope_joint.safetensors",
        "pmope_joint.checkpoint.json",
    }
    with pytest.raises(ValueError, match="singleton PAET"):
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


def test_memory_preflight_is_minimum_only_and_peak_is_deferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    cache = SimpleNamespace(
        cache_fingerprint=(
            cli.COVERAGE_STATE_PAET_HISTORICAL_CACHE_FINGERPRINT
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

    assert receipt["run_id"] == cli.RUN_ID
    assert receipt["model_parameter_count"] == 64064
    assert receipt["preflight_is_not_peak_measurement"] is True
    assert receipt["actual_peak_recorded_by_runner"] is True
    assert receipt["checks"]["source_cache_fingerprint_exact"] is True
    assert receipt["all_pass"] is True


def test_wrapper_functions_pass_the_same_explicit_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    authorization = SimpleNamespace(run_id=cli.RUN_ID)

    def prepare(*args, **kwargs):
        calls.append(("prepare", kwargs["run_id"]))
        return authorization

    def run(*args, **kwargs):
        calls.append(("run", kwargs["run_id"]))
        return SimpleNamespace(run_id=kwargs["run_id"])

    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_paet_bounded_run_authorization",
        prepare,
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_paet_bfa_pmope_bounded_400",
        run,
    )
    assert (
        cli._prepare_paet_authorization(
            object(),
            object(),
            object(),
            sealed_v20_reference=object(),
        )
        is authorization
    )
    result = cli._run_paet_bounded(authorization, object())

    assert result.run_id == cli.RUN_ID
    assert calls == [
        ("prepare", cli.RUN_ID),
        ("run", cli.RUN_ID),
    ]


def test_dr_gate_failure_completes_without_authorization_or_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / cli.RUN_ID
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "OUTPUT_REPO_PATH", f"runs/{cli.RUN_ID}")
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: (("implementation.py", "1" * 64),),
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_paet_dataset_free_gate",
        _dataset_free_stub,
    )
    monkeypatch.setattr(
        cli,
        "verify_repository_coverage_state_bfa_v20_reference",
        lambda root: _sealed_v20_stub(),
    )
    monkeypatch.setattr(
        cli._v15b_cli,
        "_verify_runtime_contract",
        lambda: {"device": "cuda:0"},
    )
    monkeypatch.setattr(
        cli._v15b_cli,
        "_claim_output",
        cli._v15b_cli._claim_output,
    )
    source_binding = SimpleNamespace(
        canonical_payload=lambda: {"split": "D_R"}
    )
    real_inputs = SimpleNamespace(
        scalar_cache=object(),
        source_binding=source_binding,
        canonical_payload=lambda: {"split": "D_R"},
    )
    population = SimpleNamespace(
        cache=object(),
        population_fingerprint="5" * 64,
        canonical_payload=lambda: {"seed": 42},
    )
    schedule = SimpleNamespace(
        selections=(),
        canonical_payload=lambda: {"updates": 400},
    )
    preflight = SimpleNamespace(
        population=population,
        schedule=schedule,
        training_authorized=True,
        canonical_payload=lambda: {"training_authorized": True},
    )
    dr_gate = SimpleNamespace(
        all_pass=False,
        evidence_fingerprint="6" * 64,
        canonical_payload=lambda: {"all_pass": False},
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        lambda **kwargs: real_inputs,
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_bounded_population",
        lambda cache, seed: population,
    )
    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_bounded_preflight",
        lambda value: preflight,
    )
    monkeypatch.setattr(
        cli,
        "coverage_state_schedule_exposure_report",
        lambda cache, value: {"updates": 400},
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_paet_dr_gate",
        lambda **kwargs: dr_gate,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("DR failure reached training")

    monkeypatch.setattr(cli, "_prepare_paet_authorization", forbidden)
    monkeypatch.setattr(cli, "_run_paet_bounded", forbidden)
    result = cli.run_once()

    assert result["run_id"] == cli.RUN_ID
    assert result["decision"] == "PAET_BFA_V21_DR_GATE_FAIL"
    assert result["bounded_gate_passed"] is False
    assert (output / "COMPLETE.json").is_file()
    assert not (output / ".incomplete").exists()
    assert not (output / "FAILURE.json").exists()
    complete = json.loads(
        (output / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert complete["run_id"] == cli.RUN_ID
    assert complete["authorization_created"] is False
    assert complete["bounded_training_performed"] is False
    assert complete["resource_measurement_performed"] is False
    assert complete["formal_800_authorized"] is False
    assert complete["D_V_accessed"] is False
    assert complete["D_T_accessed"] is False


def test_mocked_pass_writes_one_run_id_consistent_terminal_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / cli.RUN_ID
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "OUTPUT_REPO_PATH", f"runs/{cli.RUN_ID}")
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: (("implementation.py", "1" * 64),),
    )
    dataset_free = _dataset_free_stub()
    sealed_v20 = _sealed_v20_stub()
    monkeypatch.setattr(
        cli,
        "run_coverage_state_paet_dataset_free_gate",
        lambda: dataset_free,
    )
    monkeypatch.setattr(
        cli,
        "verify_repository_coverage_state_bfa_v20_reference",
        lambda root: sealed_v20,
    )
    monkeypatch.setattr(
        cli._v15b_cli,
        "_verify_runtime_contract",
        lambda: {"device": "cuda:0"},
    )
    source_binding = SimpleNamespace(
        canonical_payload=lambda: {"split": "D_R"}
    )
    real_inputs = SimpleNamespace(
        scalar_cache=object(),
        source_binding=source_binding,
        canonical_payload=lambda: {"split": "D_R"},
    )
    population = SimpleNamespace(
        cache=object(),
        population_fingerprint="5" * 64,
        canonical_payload=lambda: {"seed": 42},
    )
    schedule = SimpleNamespace(
        selections=(),
        canonical_payload=lambda: {"updates": 400},
    )
    preflight = SimpleNamespace(
        population=population,
        schedule=schedule,
        training_authorized=True,
        canonical_payload=lambda: {"training_authorized": True},
    )
    dr_gate = SimpleNamespace(
        all_pass=True,
        evidence_fingerprint="6" * 64,
        canonical_payload=lambda: {"all_pass": True},
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        lambda **kwargs: real_inputs,
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_bounded_population",
        lambda cache, seed: population,
    )
    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_bounded_preflight",
        lambda value: preflight,
    )
    monkeypatch.setattr(
        cli,
        "coverage_state_schedule_exposure_report",
        lambda cache, value: {"updates": 400},
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_paet_dr_gate",
        lambda **kwargs: dr_gate,
    )
    authorization = SimpleNamespace(
        run_id=cli.RUN_ID,
        authorization_fingerprint="a" * 64,
        sealed_v20_reference_fingerprint=(
            sealed_v20.reference_fingerprint
        ),
        training_authorized=True,
        canonical_payload=lambda: {"run_id": cli.RUN_ID},
    )
    monkeypatch.setattr(
        cli,
        "_prepare_paet_authorization",
        lambda *args, **kwargs: authorization,
    )
    model_config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(model_config)
    monkeypatch.setattr(
        cli,
        "_expected_paet_config",
        lambda value: model_config,
    )
    monkeypatch.setattr(
        cli,
        "_device_memory_preflight",
        lambda cache, config: cli._fingerprinted(
            {
                "run_id": cli.RUN_ID,
                "all_pass": True,
                "preflight_is_not_peak_measurement": True,
            }
        ),
    )
    training_row = SimpleNamespace(
        objective="pmope_joint",
        objective_policy=cli.CSLF_PMOPE_POLICY,
    )
    training = SimpleNamespace(
        results=(training_row,),
        models=(("pmope_joint", model),),
        canonical_payload=lambda: {
            "objective_suite": ["pmope_joint"],
        },
    )
    resource = SimpleNamespace(
        measurement_fingerprint="b" * 64,
        canonical_payload=lambda: {
            "device": "cuda:0",
            "updates": 400,
            "parameter_count": 64064,
            "v20_comparison": {
                "status": (
                    "NOT_EVALUATED_NO_MATCHED_V20_MEASUREMENT"
                ),
                "working_memory_ratio": None,
                "step_time_ratio": None,
                "not_a_scientific_gate": True,
            },
        },
    )
    certificate = SimpleNamespace(
        integrity_passed=True,
        all_pairs_passed=False,
        verify=lambda: None,
        canonical_payload=lambda: {
            "diagnostic_summary": {
                "pair_result_is_bounded_gate": False,
            }
        },
    )
    diagnostic = SimpleNamespace(
        canonical_payload=lambda: {"split": "D_R"}
    )
    decision = SimpleNamespace(
        run_id=cli.RUN_ID,
        bounded_gate_passed=True,
        failed_checks=(),
        canonical_payload=lambda: {
            "run_id": cli.RUN_ID,
            "bounded_gate_passed": True,
            "same_sign_response_diagnostic": {"is_gate": False},
        },
    )
    result = SimpleNamespace(
        run_id=cli.RUN_ID,
        authorization=authorization,
        training=training,
        resource_measurement=resource,
        certificate=certificate,
        diagnostic=diagnostic,
        decision=decision,
        training_invocations=1,
        certificate_invocations=1,
        zero_level_evaluation_invocations=1,
        failed_checks=(),
        bounded_gate_passed=True,
        formal800_eligible=True,
        canonical_payload=lambda: {
            "run_id": cli.RUN_ID,
            "bounded_gate_passed": True,
            "formal800_eligible": True,
            "formal_800_authorized": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_paet_bounded",
        lambda actual, config: result,
    )

    terminal = cli.run_once()

    assert terminal["run_id"] == cli.RUN_ID
    assert terminal["bounded_gate_passed"] is True
    assert terminal["formal800_eligible"] is True
    assert terminal["formal_800_authorized"] is False
    assert (output / "COMPLETE.json").is_file()
    assert not (output / ".incomplete").exists()
    complete = json.loads(
        (output / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert complete["run_id"] == cli.RUN_ID
    assert complete["artifact_file_count"] == 17
    assert complete["formal800_eligible"] is True
    assert complete["formal_800_authorized"] is False
    assert complete["formal_800_executed"] is False
    for relative in (
        "attempt.json",
        "receipts/config.json",
        "receipts/dataset_free.json",
        "receipts/sealed_v20_reference.json",
        "receipts/inputs.json",
        "receipts/preflight.json",
        "receipts/dr_gate.json",
        "receipts/authorization.json",
        "receipts/device_memory_preflight.json",
        "receipts/training_resource_measurement.json",
        "receipts/training.json",
        "receipts/post_training_certificate.json",
        "receipts/zero_level.json",
        "receipts/bounded_result.json",
        "receipts/decision.json",
        "checkpoints/pmope_joint.checkpoint.json",
    ):
        payload = json.loads(
            (output / relative).read_text(encoding="utf-8")
        )
        assert payload["run_id"] == cli.RUN_ID
    bounded = json.loads(
        (output / "receipts/bounded_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert bounded["result"]["run_id"] == cli.RUN_ID


def test_execution_exception_is_nonresumable_incomplete_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / cli.RUN_ID
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "OUTPUT_REPO_PATH", f"runs/{cli.RUN_ID}")
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: (("implementation.py", "1" * 64),),
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_paet_dataset_free_gate",
        _dataset_free_stub,
    )
    monkeypatch.setattr(
        cli,
        "verify_repository_coverage_state_bfa_v20_reference",
        lambda root: _sealed_v20_stub(),
    )
    monkeypatch.setattr(
        cli._v15b_cli,
        "_verify_runtime_contract",
        lambda: {"device": "cuda:0"},
    )

    def fail_inputs(**kwargs):
        raise RuntimeError("synthetic execution failure")

    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        fail_inputs,
    )
    with pytest.raises(RuntimeError, match="synthetic execution failure"):
        cli.run_once()

    assert (output / ".incomplete").is_file()
    assert (output / "FAILURE.json").is_file()
    assert not (output / "COMPLETE.json").exists()
    failure = json.loads(
        (output / "FAILURE.json").read_text(encoding="utf-8")
    )
    assert failure["run_id"] == cli.RUN_ID
    assert failure["status"] == "failed_incomplete_attempt"
    assert failure["resume_allowed"] is False
    assert failure["automatic_retry_allowed"] is False
    assert failure["formal_800_authorized"] is False
    assert failure["D_V_accessed"] is False
    assert failure["D_T_accessed"] is False


def test_main_prints_one_canonical_json_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"run_id": cli.RUN_ID, "ok": True}
    monkeypatch.setattr(cli, "validate_create_only", lambda: payload)
    assert cli.main(("--validate-create-only",)) == 0
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert json.loads(output) == payload
