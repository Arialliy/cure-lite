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
from tools import (
    run_coverage_state_cslf_completion_rooted_bounded_400 as runner,
)


def test_frozen_real_inputs_wrapper_and_output_contract_are_exact() -> None:
    assert runner.RUN_ID == (
        "cure_lite_cslf_v15a_completion_rooted_bounded_400_r1"
    )
    assert runner.OUTPUT_REPO_PATH == (
        "runs/irstd1k_stage_a_seed42/"
        "cure_lite_cslf_v15a_completion_rooted_bounded_400_r1"
    )
    assert runner.OUTPUT_REPO_PATH != runner.PARENT_V15_RUN_REPO_PATH
    assert runner.FROZEN_DEVICE == "cuda:0"
    assert runner.FROZEN_VISIBLE_GPU == "0"
    assert runner.FROZEN_CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert runner.FROZEN_PAUSE_TEMPERATURE_C == 82
    assert runner.FROZEN_RESUME_TEMPERATURE_C == 75
    assert runner.FROZEN_CHECKPOINT_SERIALIZATION == "safetensors"
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


def test_completion_rooted_objective_suite_and_policy_are_exact() -> None:
    suite = tuple(
        value.value
        for value in runner.COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES
    )
    assert suite == (
        "completion_rooted_response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    assert runner.coverage_state_pair_objective_policy(suite[0]) == (
        "completion_endpoint_absolute_root_with_finite_coverage_response_v1"
    )
    assert runner.coverage_state_pair_objective_policy(suite[1]) == suite[1]
    assert runner.coverage_state_pair_objective_policy(suite[2]) == suite[2]


def test_parent_v15_negative_result_and_source_closure_are_exact() -> None:
    closure = runner._verify_parent_v15_closure()
    assert closure == {
        "run_repo_path": runner.PARENT_V15_RUN_REPO_PATH,
        "complete_file_sha256": runner.PARENT_V15_COMPLETE_SHA256,
        "complete_fingerprint": (
            runner.PARENT_V15_COMPLETE_FINGERPRINT
        ),
        "artifact_file_count": 17,
        "source_archive_repo_path": (
            runner.PARENT_V15_SOURCE_ARCHIVE_REPO_PATH
        ),
        "source_archive_sha256": (
            runner.PARENT_V15_SOURCE_ARCHIVE_SHA256
        ),
        "source_manifest_repo_path": (
            runner.PARENT_V15_SOURCE_MANIFEST_REPO_PATH
        ),
        "source_manifest_sha256": (
            runner.PARENT_V15_SOURCE_MANIFEST_SHA256
        ),
    }
    archive = (
        runner._ROOT / runner.PARENT_V15_SOURCE_ARCHIVE_REPO_PATH
    ).resolve()
    manifest = (
        runner._ROOT / runner.PARENT_V15_SOURCE_MANIFEST_REPO_PATH
    ).resolve()
    assert file_sha256(archive) == runner.PARENT_V15_SOURCE_ARCHIVE_SHA256
    assert file_sha256(manifest) == runner.PARENT_V15_SOURCE_MANIFEST_SHA256


def test_parent_complete_sha_mismatch_rejects_without_modifying_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = (
        runner._ROOT
        / runner.PARENT_V15_RUN_REPO_PATH
        / "COMPLETE.json"
    ).resolve()
    before = file_sha256(complete)
    monkeypatch.setattr(runner, "PARENT_V15_COMPLETE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="parent v15 COMPLETE changed"):
        runner._verify_parent_v15_closure()
    assert file_sha256(complete) == before


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
    assert receipt["bounded_output_exists"] is existed_before
    assert "formal_output_exists" not in receipt
    assert receipt["D_V_accessed"] is False
    assert receipt["D_T_accessed"] is False
    assert receipt["parent_v15_negative_result"][
        "complete_fingerprint"
    ] == runner.PARENT_V15_COMPLETE_FINGERPRINT
    assert receipt["parent_v15_negative_result"][
        "source_archive_sha256"
    ] == runner.PARENT_V15_SOURCE_ARCHIVE_SHA256


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
        "tools/run_coverage_state_cslf_completion_rooted_bounded_400.py",
        "--run-once",
    )
    runner._validate_wrapper_command(valid)
    for changed, message in (
        (
            tuple("1" if value == "0" else value for value in valid),
            "--gpu",
        ),
        (
            tuple(
                "83" if value == "--pause-temp=82" else value
                for value in valid
            ),
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
        "tools/run_coverage_state_cslf_completion_rooted_bounded_400.py",
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


def test_checkpoint_is_policy_bound_tensor_only_and_round_trips(
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
    objective = "completion_rooted_response_joint"
    objective_policy = runner.coverage_state_pair_objective_policy(
        objective
    )
    receipt = runner._write_checkpoint_new(
        directory,
        objective=objective,
        objective_policy=objective_policy,
        model=model,
    )
    assert receipt["objective"] == objective
    assert receipt["objective_policy"] == objective_policy
    assert receipt["tensor_only_state_dict"] is True
    assert receipt["weights_only_roundtrip_verified"] is True
    assert receipt["serialization"] == "safetensors"
    checkpoint = next(
        value
        for value in directory.iterdir()
        if value.suffix == ".safetensors"
    )
    assert file_sha256(checkpoint) == receipt["checkpoint_file_sha256"]
    with pytest.raises(FileExistsError):
        runner._write_checkpoint_new(
            directory,
            objective=objective,
            objective_policy=objective_policy,
            model=model,
        )


def test_static_config_freezes_checkpoint_serialization() -> None:
    source_paths = runner._verify_frozen_sources()
    config = runner._static_config_payload(
        source_paths=source_paths,
        implementation=runner._implementation_binding(),
    )
    assert config["execution"]["checkpoint_serialization"] == (
        runner.FROZEN_CHECKPOINT_SERIALIZATION
    )


def test_complete_receipt_graph_binds_bounded_result_directly() -> None:
    names = (
        "config",
        "input",
        "preflight",
        "dataset_free",
        "authorization",
        "training",
        "zero",
        "bounded",
        "decision",
    )
    receipts = {
        name: {"receipt_fingerprint": name * 8}
        for name in names
    }
    graph = runner._complete_receipt_fingerprints(
        config=receipts["config"],
        input_receipt=receipts["input"],
        preflight_receipt=receipts["preflight"],
        dataset_free_receipt=receipts["dataset_free"],
        authorization_receipt=receipts["authorization"],
        training_receipt=receipts["training"],
        zero_receipt=receipts["zero"],
        bounded_receipt=receipts["bounded"],
        decision=receipts["decision"],
    )
    assert graph == {
        "config_fingerprint": receipts["config"][
            "receipt_fingerprint"
        ],
        "input_receipt_fingerprint": receipts["input"][
            "receipt_fingerprint"
        ],
        "preflight_receipt_fingerprint": receipts["preflight"][
            "receipt_fingerprint"
        ],
        "dataset_free_receipt_fingerprint": receipts["dataset_free"][
            "receipt_fingerprint"
        ],
        "authorization_receipt_fingerprint": receipts["authorization"][
            "receipt_fingerprint"
        ],
        "training_receipt_fingerprint": receipts["training"][
            "receipt_fingerprint"
        ],
        "zero_level_receipt_fingerprint": receipts["zero"][
            "receipt_fingerprint"
        ],
        "bounded_result_receipt_fingerprint": receipts["bounded"][
            "receipt_fingerprint"
        ],
        "decision_fingerprint": receipts["decision"][
            "receipt_fingerprint"
        ],
    }


def test_existing_single_use_output_stops_before_any_input_or_parent_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "already-claimed"
    existing.mkdir()
    monkeypatch.setattr(runner, "OUTPUT_PATH", existing)

    def forbidden() -> object:
        raise AssertionError(
            "input and parent validation must not run after claim exists"
        )

    monkeypatch.setattr(runner, "_verify_frozen_sources", forbidden)
    monkeypatch.setattr(runner, "_verify_parent_v15_closure", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_once()


def test_implementation_binding_includes_new_cli_device_cache_and_core() -> None:
    binding = dict(runner._implementation_binding())
    required = {
        "tools/run_coverage_state_cslf_completion_rooted_bounded_400.py",
        "tools/run_with_gpu_temperature_control.py",
        "cure_lite/coverage_state_device_cache.py",
        "cure_lite/experiment/coverage_state_real_dr_inputs.py",
        "cure_lite/experiment/coverage_state_bounded_protocol.py",
        "cure_lite/experiment/coverage_state_bounded_runner.py",
        "cure_lite/experiment/coverage_state_training.py",
        "cure_lite/experiment/coverage_state_zero_level_evaluation.py",
    }
    assert required <= set(binding)
    assert "tools/run_coverage_state_cslf_bounded_400.py" not in binding
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
