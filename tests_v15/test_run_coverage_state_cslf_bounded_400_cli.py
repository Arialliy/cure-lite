from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.cache.schema import file_sha256
from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
)
from tools import run_coverage_state_cslf_bounded_400 as runner


def test_frozen_real_inputs_wrapper_and_output_contract_are_exact() -> None:
    assert runner.OUTPUT_REPO_PATH == (
        "runs/irstd1k_stage_a_seed42/"
        "cure_lite_cslf_v15_bounded_400_r1"
    )
    assert runner.FROZEN_DEVICE == "cuda:0"
    assert runner.FROZEN_VISIBLE_GPU == "0"
    assert runner.FROZEN_CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert runner.FROZEN_PAUSE_TEMPERATURE_C == 82
    assert runner.FROZEN_RESUME_TEMPERATURE_C == 75
    assert len(runner.FROZEN_REAL_DR_INPUTS) == 5
    paths = runner._verify_frozen_sources()
    for name, relative, digest in runner.FROZEN_REAL_DR_INPUTS:
        assert paths[name] == (runner._ROOT / relative).resolve()
        assert file_sha256(paths[name]) == digest
    wrapper = (
        runner._ROOT / runner.TEMPERATURE_WRAPPER_REPO_PATH
    ).resolve()
    assert file_sha256(wrapper) == (
        runner.TEMPERATURE_WRAPPER_FILE_SHA256
    )


def test_create_only_validation_never_claims_or_loads_cached_payload() -> None:
    existed_before = (
        runner.OUTPUT_PATH.exists() or runner.OUTPUT_PATH.is_symlink()
    )
    receipt = runner.validate_create_only()
    existed_after = (
        runner.OUTPUT_PATH.exists() or runner.OUTPUT_PATH.is_symlink()
    )
    assert existed_after is existed_before
    assert receipt["static_contract_valid"] is True
    assert receipt["not_a_formal_result"] is True
    assert receipt["output_claimed"] is False
    assert receipt["D_R_cached_tensor_payload_accessed"] is False
    assert receipt["dataset_free_gate_executed"] is False
    assert receipt["authorization_created"] is False
    assert receipt["training_performed"] is False
    assert receipt["formal_800_authorized"] is False
    assert receipt["D_V_accessed"] is False
    assert receipt["D_T_accessed"] is False


def test_wrapper_command_requires_exact_gpu_and_hysteresis() -> None:
    wrapper = str(
        (runner._ROOT / runner.TEMPERATURE_WRAPPER_REPO_PATH).resolve()
    )
    valid = (
        "/usr/bin/python3",
        wrapper,
        "--gpu",
        "0",
        "--pause-temp=82",
        "--resume-temp",
        "75",
        "--",
        "/python",
        "tools/run_coverage_state_cslf_bounded_400.py",
        "--run-once",
    )
    runner._validate_wrapper_command(valid)
    for changed, message in (
        (
            tuple("1" if value == "0" else value for value in valid),
            "--gpu",
        ),
        (
            tuple("83" if value == "--pause-temp=82" else value for value in valid),
            "--pause-temp",
        ),
    ):
        with pytest.raises(RuntimeError, match=message):
            runner._validate_wrapper_command(changed)
    with pytest.raises(RuntimeError, match="bound wrapper"):
        runner._validate_wrapper_command(("/usr/bin/python3", "other.py"))


def test_runtime_contract_requires_wrapper_env_and_single_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = str(
        (runner._ROOT / runner.TEMPERATURE_WRAPPER_REPO_PATH).resolve()
    )
    parent = (
        "/python",
        wrapper,
        "--gpu",
        "0",
        "--pause-temp",
        "82",
        "--resume-temp",
        "75",
        "--",
        "/python",
        "tools/run_coverage_state_cslf_bounded_400.py",
        "--run-once",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(runner, "_read_parent_command", lambda: parent)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(name="test-gpu", total_memory=8),
    )
    receipt = runner._verify_runtime_contract()
    assert receipt["device"] == "cuda:0"
    assert receipt["visible_device_count"] == 1
    assert receipt["pause_temperature_c"] == 82
    assert receipt["resume_temperature_c"] == 75

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        runner._verify_runtime_contract()


def test_output_claim_is_exclusive_and_marks_incomplete(
    tmp_path: Path,
) -> None:
    output = tmp_path / "single"
    attempt = runner._fingerprinted(
        {"schema_version": runner.ATTEMPT_SCHEMA}
    )
    receipts, checkpoints = runner._claim_output(
        output,
        attempt=attempt,
    )
    assert (output / ".incomplete").is_file()
    assert (output / "attempt.json").is_file()
    assert receipts.is_dir()
    assert checkpoints.is_dir()
    with pytest.raises(FileExistsError):
        runner._claim_output(output, attempt=attempt)


def test_checkpoint_is_tensor_only_exclusive_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    model = CURELiteCoverageStateLevelSet(
        CoverageStateLevelSetConfig(
            feature_channels=4,
            feature_stride=4,
            width=8,
        )
    )
    receipt = runner._write_checkpoint_new(
        directory,
        objective="response_joint",
        model=model,
    )
    assert receipt["tensor_only_state_dict"] is True
    assert receipt["weights_only_roundtrip_verified"] is True
    assert receipt["serialization"] in {
        "safetensors",
        "torch_tensor_only_state_dict",
    }
    checkpoint = next(
        value
        for value in directory.iterdir()
        if value.suffix in {".pt", ".safetensors"}
    )
    assert file_sha256(checkpoint) == receipt["checkpoint_file_sha256"]
    with pytest.raises(FileExistsError):
        runner._write_checkpoint_new(
            directory,
            objective="response_joint",
            model=model,
        )


def test_existing_single_use_output_stops_before_any_input_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "already-claimed"
    existing.mkdir()
    monkeypatch.setattr(runner, "OUTPUT_PATH", existing)

    def forbidden() -> object:
        raise AssertionError("input validation must not run after claim exists")

    monkeypatch.setattr(runner, "_verify_frozen_sources", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_once()


def test_implementation_binding_includes_cli_device_cache_and_core() -> None:
    binding = dict(runner._implementation_binding())
    required = {
        "tools/run_coverage_state_cslf_bounded_400.py",
        "tools/run_with_gpu_temperature_control.py",
        "cure_lite/coverage_state_device_cache.py",
        "cure_lite/experiment/coverage_state_real_dr_inputs.py",
        "cure_lite/experiment/coverage_state_bounded_protocol.py",
        "cure_lite/experiment/coverage_state_bounded_runner.py",
        "cure_lite/experiment/coverage_state_training.py",
        "cure_lite/experiment/coverage_state_zero_level_evaluation.py",
    }
    assert required <= set(binding)
    assert all(len(value) == 64 for value in binding.values())


def test_cli_requires_explicit_nonconfigurable_mode() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(())
    assert runner.parse_args(
        ("--validate-create-only",)
    ).validate_create_only
    assert runner.parse_args(("--run-once",)).run_once
    with pytest.raises(SystemExit):
        runner.parse_args(
            ("--run-once", "--output", "/tmp/not-authorized")
        )
