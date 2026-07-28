from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
)
from tools import (
    run_coverage_state_paet_bfa_pmope_formal_800 as cli,
)


@pytest.fixture(autouse=True)
def _sealed_source_closure(monkeypatch: pytest.MonkeyPatch):
    """Unit tests do not create the irreversible repository closure."""

    monkeypatch.setattr(
        cli,
        "verify_coverage_state_paet_formal_source_closure",
        lambda: {
            "sealed": True,
            "manifest_sha256": "c" * 64,
            "archive_sha256": "d" * 64,
            "content_fingerprint": "e" * 64,
            "file_count": 7,
        },
    )


class _BoundedSeal:
    audit_fingerprint = "a" * 64
    structural_advancement_passed = True
    generic_population_gate_passed = False

    def verify_unchanged(self) -> None:
        return None


def test_constants_and_static_config_freeze_unique_formal800():
    assert cli.RUN_ID == (
        "cure_lite_paet_bfa_v21_pmope_formal_800_seed42_r1"
    )
    assert cli.OUTPUT_REPO_PATH.endswith(cli.RUN_ID)
    assert cli.FROZEN_SEED == 42
    assert cli.FROZEN_EPOCHS == 800
    assert cli.FROZEN_STEPS_PER_EPOCH == 40
    assert cli.FROZEN_UPDATES == 32_000
    assert cli.FROZEN_FEATURE_CHANNELS == 64
    assert cli.FROZEN_FEATURE_STRIDE == 4
    assert cli.FROZEN_MODEL_WIDTH == 32
    assert cli.FROZEN_PARAMETER_COUNT == 64_064
    assert cli.FROZEN_DEVICE == "cuda:0"
    assert cli.FROZEN_VISIBLE_GPU == "0"
    assert cli.FROZEN_CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert cli.FROZEN_PAUSE_TEMPERATURE_C == 82
    assert cli.FROZEN_RESUME_TEMPERATURE_C == 75

    config = cli._static_config_payload(
        source_paths={},
        implementation=(("implementation.py", "b" * 64),),
        bounded_artifact_seal_fingerprint="a" * 64,
    )
    assert config["run_id"] == cli.RUN_ID
    assert config["runtime_splits"] == ["D_R"]
    assert config["budget"] == {
        "seed": 42,
        "epochs": 800,
        "steps_per_epoch": 40,
        "updates": 32_000,
        "objectives": 1,
        "from_scratch": True,
    }
    assert config["model"]["parameter_count"] == 64_064
    assert config["final_artifact"]["checkpoint_policy"] == (
        "final_model_only"
    )
    assert config["final_artifact"]["optimizer_state_saved"] is False
    assert (
        config["post_training_structural_replay"][
            "D_V_authorized_only_if_structural_retention_passes"
        ]
        is True
    )
    assert (
        config["post_training_structural_replay"][
            "generic_population_gate_reported_separately"
        ]
        is True
    )
    assert config["evidence_scope"]["D_V_accessed"] is False
    assert config["evidence_scope"]["D_T_accessed"] is False


def test_create_only_is_exact_and_never_enters_D_R_or_claims_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cli, "OUTPUT_PATH", tmp_path / "unclaimed")
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: (("implementation.py", "b" * 64),),
    )
    monkeypatch.setattr(
        cli,
        "load_repository_coverage_state_paet_bounded_artifact_seal",
        _BoundedSeal,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("create-only entered a run-only path")

    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        forbidden,
    )
    monkeypatch.setattr(cli, "_claim_output", forbidden)
    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_paet_formal_800_authorization",
        forbidden,
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_paet_bfa_pmope_formal_800",
        forbidden,
    )
    first = cli.validate_create_only()
    second = cli.validate_create_only()

    assert first == second
    assert first["receipt_fingerprint"] == second["receipt_fingerprint"]
    assert first["D_R_cached_tensor_payload_accessed"] is False
    assert first["real_inputs_constructed"] is False
    assert first["authorization_created"] is False
    assert first["training_performed"] is False
    assert first["output_claimed"] is False
    assert first["D_V_accessed"] is False
    assert first["D_T_accessed"] is False
    assert not cli.OUTPUT_PATH.exists()


def test_create_only_verifies_the_sealed_closure_before_any_d_r_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cli, "OUTPUT_PATH", tmp_path / "unclaimed")
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "verify_coverage_state_paet_formal_source_closure",
        lambda: calls.append("closure") or {
            "sealed": True,
            "manifest_sha256": "c" * 64,
            "archive_sha256": "d" * 64,
            "content_fingerprint": "e" * 64,
            "file_count": 7,
        },
    )
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli, "_implementation_binding", lambda: (("impl.py", "b" * 64),)
    )
    monkeypatch.setattr(
        cli, "load_repository_coverage_state_paet_bounded_artifact_seal", _BoundedSeal
    )

    result = cli.validate_create_only()

    assert calls == ["closure"]
    assert result["source_closure_manifest_sha256"] == "c" * 64
    assert result["D_R_cached_tensor_payload_accessed"] is False


def test_run_once_claims_then_verifies_closure_before_any_d_r_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "claimed-before-closure-failure"
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    calls: list[str] = []

    def reject_closure():
        calls.append("closure")
        raise RuntimeError("closure missing")

    def forbidden(*args, **kwargs):
        raise AssertionError("run-once reached a D_R path")

    monkeypatch.setattr(
        cli, "verify_coverage_state_paet_formal_source_closure", reject_closure
    )
    monkeypatch.setattr(cli, "_verify_frozen_sources", forbidden)

    with pytest.raises(RuntimeError, match="closure missing"):
        cli.run_once()
    assert calls == ["closure"]
    assert output.is_dir()
    assert (output / ".incomplete").is_file()
    assert not (output / "attempt.json").exists()


def test_create_only_stdout_is_byte_deterministic_and_d_r_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    monkeypatch.setattr(cli, "OUTPUT_PATH", tmp_path / "unclaimed")
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: (("implementation.py", "b" * 64),),
    )
    monkeypatch.setattr(
        cli,
        "load_repository_coverage_state_paet_bounded_artifact_seal",
        _BoundedSeal,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("create-only accessed a run-only D_R path")

    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        forbidden,
    )
    assert cli.main(("--validate-create-only",)) == 0
    first = capsys.readouterr().out.encode("utf-8")
    assert cli.main(("--validate-create-only",)) == 0
    second = capsys.readouterr().out.encode("utf-8")
    assert first == second
    assert not cli.OUTPUT_PATH.exists()


def test_cli_exposes_only_two_modes():
    assert cli.parse_args(
        ("--validate-create-only",)
    ).validate_create_only
    assert cli.parse_args(("--run-once",)).run_once
    with pytest.raises(SystemExit):
        cli.parse_args(())
    for arguments in (
        ("--run-once", "--seed", "43"),
        ("--run-once", "--epochs", "10"),
        ("--run-once", "--output", "/tmp/other"),
        ("--run-once", "--resume"),
        ("--run-once", "--retry"),
        ("--validate-create-only", "--run-once"),
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(arguments)


def test_implementation_binding_contains_complete_execution_closure():
    binding = dict(cli._implementation_binding())
    required = {
        "cure_lite/experiment/coverage_state_paet_formal_training.py",
        "cure_lite/experiment/coverage_state_paet_formal_artifacts.py",
        "cure_lite/experiment/coverage_state_paet_formal_structural.py",
        "cure_lite/experiment/coverage_state_bounded_protocol.py",
        "cure_lite/experiment/coverage_state_zero_level_evaluation.py",
        "tools/run_coverage_state_paet_bfa_pmope_formal_800.py",
        "tools/run_with_gpu_temperature_control.py",
    }
    assert required <= set(binding)
    assert all(len(value) == 64 for value in binding.values())


def test_epoch_progress_records_all_800_rows(tmp_path: Path, capsys):
    recorder = cli._EpochProgressRecorder(tmp_path / "progress.jsonl")
    rows = []
    for epoch in range(800):
        row = {
            "epoch": epoch,
            "completed_updates": (epoch + 1) * 40,
            "objective": "pmope_joint",
            "mean_total": 1.0,
        }
        rows.append(row)
        recorder("pmope_joint", row)
    recorder.close_and_verify(rows)

    lines = (tmp_path / "progress.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 800
    assert json.loads(lines[0])["epoch_result"]["epoch"] == 0
    assert json.loads(lines[-1])["epoch_result"]["epoch"] == 799
    assert "formal800_epoch_complete" in capsys.readouterr().err


def test_resource_measurement_wraps_exactly_one_training_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    recorder = cli._EpochProgressRecorder(tmp_path / "progress.jsonl")
    returned = object()
    calls = []

    def run(authorization, config, **kwargs):
        calls.append((authorization, config, kwargs))
        return returned

    monkeypatch.setattr(
        cli,
        "run_coverage_state_paet_bfa_pmope_formal_800",
        run,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: 100,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: 200,
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: None,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: 150,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda device: 260,
    )
    ticks = iter((1_000, 33_001_000))
    monkeypatch.setattr(
        cli.time,
        "perf_counter_ns",
        lambda: next(ticks),
    )

    result, resource = cli._measure_formal_training(
        "authorization",
        cli._formal_model_config(),
        epoch_callback=recorder,
    )
    recorder.close_after_failure()
    assert result is returned
    assert len(calls) == 1
    assert calls[0][2]["device"] == "cuda:0"
    assert calls[0][2]["epoch_callback"] is recorder
    assert resource["elapsed_ns"] == 33_000_000
    assert resource["incremental_peak_allocated_bytes"] == 50
    assert resource["incremental_peak_reserved_bytes"] == 60
    assert resource["training_invocations"] == 1


def _decision_inputs(*, structural_passed: bool, generic_passed: bool):
    authorization = SimpleNamespace(
        authorization_fingerprint="1" * 64
    )
    row = SimpleNamespace(final_model_fingerprint="2" * 64)
    training = SimpleNamespace(results=(row,))
    formal = SimpleNamespace(
        training_complete=True,
        result_fingerprint="3" * 64,
        authorization=authorization,
        training=training,
    )
    loaded = SimpleNamespace(
        formal_result_fingerprint="3" * 64,
        authorization_fingerprint="1" * 64,
        training_model_fingerprint="2" * 64,
        module_state_fingerprint="4" * 64,
        artifact_fingerprint="5" * 64,
    )
    structural = SimpleNamespace(
        post_formal_structural_retention_passed=structural_passed,
        generic_population_gate_passed=generic_passed,
        evaluation_invocations=1,
        final_model_fingerprint="4" * 64,
        result_fingerprint="6" * 64,
    )
    return formal, loaded, structural


@pytest.mark.parametrize(
    ("structural_passed", "generic_passed", "authorized"),
    (
        (True, False, True),
        (True, True, True),
        (False, True, False),
        (False, False, False),
    ),
)
def test_decision_separates_structural_and_generic_gates(
    structural_passed: bool,
    generic_passed: bool,
    authorized: bool,
):
    formal, loaded, structural = _decision_inputs(
        structural_passed=structural_passed,
        generic_passed=generic_passed,
    )
    decision = cli._structural_decision_payload(
        formal_result=formal,
        loaded_artifact=loaded,
        structural=structural,
    )
    assert decision["paet_structural_retention_gate_passed"] is (
        structural_passed
    )
    assert decision["generic_zero_level_population_gate_passed"] is (
        generic_passed
    )
    assert decision["generic_gate_is_D_V_prerequisite"] is False
    assert decision["D_V_authorized"] is authorized
    assert decision["performance_gate_passed"] is None
    assert decision["performance_evaluation_performed"] is False


def _patch_successful_run(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    *,
    structural_passed: bool,
    generic_passed: bool,
):
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    # The production path is rooted under the repository.  Keep that
    # invariant in the synthetic run while redirecting its bytes to tmp_path.
    monkeypatch.setattr(cli, "_ROOT", output.parent)
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    binding = (("implementation.py", "b" * 64),)
    monkeypatch.setattr(cli, "_implementation_binding", lambda: binding)
    monkeypatch.setattr(
        cli,
        "load_repository_coverage_state_paet_bounded_artifact_seal",
        _BoundedSeal,
    )
    monkeypatch.setattr(
        cli._base_cli,
        "_verify_runtime_contract",
        lambda: {
            "device": "cuda:0",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
    )
    scalar_cache = SimpleNamespace(
        cache_fingerprint=(
            cli.COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        )
    )
    real_inputs = SimpleNamespace(
        scalar_cache=scalar_cache,
        build_fingerprint="7" * 64,
        verify_unchanged=lambda: None,
        canonical_payload=lambda: {
            "dataset": "IRSTD-1K",
            "split": "D_R",
            "scalar_cache_fingerprint": scalar_cache.cache_fingerprint,
        },
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        lambda **kwargs: real_inputs,
    )
    config = cli._formal_model_config()
    monkeypatch.setattr(
        cli,
        "expected_coverage_state_paet_formal_config",
        lambda inputs: config,
    )
    authorization = SimpleNamespace(
        audit_fingerprint="8" * 64,
        authorization_fingerprint="1" * 64,
        formal_training_authorized=True,
        verify_unchanged=lambda: None,
        canonical_payload=lambda: {
            "run_id": cli.RUN_ID,
            "formal_training_authorized": True,
        },
    )
    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_paet_formal_800_authorization",
        lambda *args, **kwargs: authorization,
    )
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(config)
    epoch_rows = tuple(
        {
            "epoch": epoch,
            "completed_updates": (epoch + 1) * 40,
            "objective": "pmope_joint",
        }
        for epoch in range(800)
    )
    row = SimpleNamespace(
        final_model_fingerprint="2" * 64,
        epoch_logs=epoch_rows,
    )
    training = SimpleNamespace(
        results=(row,),
        models=(("pmope_joint", model),),
    )
    formal_result = SimpleNamespace(
        training_complete=True,
        training_invocations=1,
        training=training,
        result_fingerprint="3" * 64,
        authorization=authorization,
        canonical_payload=lambda: {
            "run_id": cli.RUN_ID,
            "training_complete": True,
        },
    )
    resource = {
        "receipt_fingerprint": "9" * 64,
        "elapsed_ns": 1,
    }

    def measure(auth, model_config, *, epoch_callback):
        for row_value in epoch_rows:
            epoch_callback("pmope_joint", row_value)
        return formal_result, resource

    monkeypatch.setattr(cli, "_measure_formal_training", measure)

    def save(directory, result):
        directory.mkdir()
        for name in (
            "model.safetensors",
            "formal_result.json",
            "training.json",
            "epoch_log.json",
            "receipt.json",
        ):
            (directory / name).write_bytes(name.encode("utf-8"))
        return "5" * 64

    monkeypatch.setattr(
        cli,
        "save_coverage_state_paet_formal_artifact",
        save,
    )
    loaded = SimpleNamespace(
        model=model,
        formal_result_fingerprint="3" * 64,
        authorization_fingerprint="1" * 64,
        training_model_fingerprint="2" * 64,
        module_state_fingerprint="4" * 64,
        artifact_fingerprint="5" * 64,
        receipt_sha256="6" * 64,
        verify_unchanged=lambda: None,
    )
    monkeypatch.setattr(
        cli,
        "load_coverage_state_paet_formal_artifact",
        lambda *args, **kwargs: loaded,
    )
    population = SimpleNamespace(
        cache=object(),
        population_fingerprint="a" * 64,
        verify_unchanged=lambda: None,
        canonical_payload=lambda: {"seed": 42},
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_bounded_population",
        lambda cache, seed: population,
    )
    structural = SimpleNamespace(
        post_formal_structural_retention_passed=structural_passed,
        generic_population_gate_passed=generic_passed,
        evaluation_invocations=1,
        final_model_fingerprint="4" * 64,
        result_fingerprint="b" * 64,
        canonical_payload=lambda: {
            "post_formal_structural_retention_passed": structural_passed,
            "generic_population_gate_passed": generic_passed,
        },
    )
    monkeypatch.setattr(
        cli,
        "evaluate_coverage_state_paet_formal_structural_retention",
        lambda *args, **kwargs: structural,
    )
    return formal_result, loaded, structural


@pytest.mark.parametrize(
    ("structural_passed", "generic_passed", "authorized"),
    ((True, False, True), (False, True, False)),
)
def test_run_once_closes_complete_with_separate_structural_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    structural_passed: bool,
    generic_passed: bool,
    authorized: bool,
):
    output = tmp_path / f"run-{int(structural_passed)}"
    _patch_successful_run(
        monkeypatch,
        output,
        structural_passed=structural_passed,
        generic_passed=generic_passed,
    )
    result = cli.run_once()

    assert result["formal_training_complete"] is True
    assert result["paet_structural_retention_gate_passed"] is (
        structural_passed
    )
    assert result["generic_zero_level_population_gate_passed"] is (
        generic_passed
    )
    assert result["D_V_authorized"] is authorized
    assert result["D_V_accessed"] is False
    assert result["D_T_accessed"] is False
    assert (output / "STARTED.json").is_file()
    assert (output / "COMPLETE.json").is_file()
    assert not (output / "FAILURE.json").exists()
    assert not (output / ".incomplete").exists()
    complete = json.loads(
        (output / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert complete["artifact_file_count"] == 14
    assert len(complete["artifact_files"]) == 14
    assert not {
        "attempt.json",
        "STARTED.json",
        "FAILURE.json",
        ".incomplete",
    } & set(complete["artifact_files"])
    assert complete["D_V_authorized"] is authorized
    assert (
        complete["generic_zero_level_population_gate_passed"]
        is generic_passed
    )
    assert complete["performance_evaluation_performed"] is False
    assert complete["attempt_fingerprint"] == json.loads(
        (output / "attempt.json").read_text(encoding="utf-8")
    )["receipt_fingerprint"]
    assert complete["started_fingerprint"] == json.loads(
        (output / "STARTED.json").read_text(encoding="utf-8")
    )["receipt_fingerprint"]
    assert complete["source_closure_file_count"] == 7
    assert "formal800_epoch_complete" in capsys.readouterr().err


def test_run_once_refuses_complete_when_source_closure_drifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "source-drift"
    _patch_successful_run(
        monkeypatch,
        output,
        structural_passed=True,
        generic_passed=False,
    )
    initial = {
        "sealed": True,
        "manifest_sha256": "c" * 64,
        "archive_sha256": "d" * 64,
        "content_fingerprint": "e" * 64,
        "file_count": 7,
    }
    changed = {
        **initial,
        "content_fingerprint": "f" * 64,
    }
    receipts = iter((initial, changed))
    monkeypatch.setattr(
        cli,
        "verify_coverage_state_paet_formal_source_closure",
        lambda: next(receipts),
    )

    with pytest.raises(
        RuntimeError,
        match="source closure changed during execution",
    ):
        cli.run_once()
    assert (output / "FAILURE.json").is_file()
    assert (output / ".incomplete").is_file()
    assert not (output / "COMPLETE.json").exists()


def test_failure_is_terminal_and_second_attempt_stops_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "failed"
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: (("implementation.py", "b" * 64),),
    )
    monkeypatch.setattr(
        cli,
        "load_repository_coverage_state_paet_bounded_artifact_seal",
        _BoundedSeal,
    )
    monkeypatch.setattr(
        cli._base_cli,
        "_verify_runtime_contract",
        lambda: {"device": "cuda:0"},
    )
    calls = 0

    def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("input build failed")

    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        fail,
    )
    with pytest.raises(RuntimeError, match="input build failed"):
        cli.run_once()
    assert calls == 1
    assert (output / "STARTED.json").is_file()
    assert (output / "FAILURE.json").is_file()
    assert (output / ".incomplete").is_file()
    assert not (output / "COMPLETE.json").exists()

    def forbidden(*args, **kwargs):
        raise AssertionError("second attempt entered prerequisites")

    monkeypatch.setattr(cli, "_verify_frozen_sources", forbidden)
    with pytest.raises(FileExistsError, match="File exists"):
        cli.run_once()
    assert calls == 1


def test_existing_output_is_rejected_before_any_d_r_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "already-claimed"
    output.mkdir()
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)

    def forbidden(*args, **kwargs):
        raise AssertionError("existing output inspected prerequisites")

    monkeypatch.setattr(cli, "_verify_frozen_sources", forbidden)
    with pytest.raises(FileExistsError, match="File exists"):
        cli.run_once()


def test_cli_source_has_no_formal_d_v_or_d_t_execution_import():
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "coverage_state_paet_formal_evaluation" not in source
    assert "coverage_state_paet_formal_decision" not in source
    assert "build_paet_fixed_d_v_samples" not in source
    assert "evaluate_paet_formal_d_v" not in source
