from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.cache.schema import file_sha256
from cure_lite.coverage_state_phase_preserving import (
    CURELitePhasePreservingCoverageStateLevelSet,
    CoverageStatePhasePreservingConfig,
)
from tools import (
    run_coverage_state_cslf_ppce_support_oriented_bounded_400 as cli,
)


def test_ppce_create_only_constants_and_parent_closure_are_exact() -> None:
    assert cli.RUN_ID == (
        "cure_lite_cslf_v16_ppce_support_oriented_bounded_400_r1"
    )
    assert cli.OUTPUT_REPO_PATH != cli.PARENT_V15B_RUN_REPO_PATH
    assert cli.FROZEN_FEATURE_CHANNELS == 64
    assert cli.FROZEN_FEATURE_STRIDE == 4
    assert cli.FROZEN_MODEL_WIDTH == 32
    assert cli.FROZEN_PARAMETER_COUNT == 23856
    closure = cli._verify_parent_v15b_closure()
    assert closure["complete_fingerprint"] == (
        "13cc94f4f5140031fc050ac8d1726e13f9e5e1bbfa8a433bda28783088121f95"
    )
    assert closure["complete_sha256"] == (
        "58460fde25d08123231e2ab1ae5767f46ae3e40896b605b9e77c144413f6a896"
    )
    assert closure["source_manifest_sha256"] == (
        "d5d5df197eab3bf4423777a4192f7d1bc0781518a54d9e56d37a8dbb48d9da8f"
    )
    assert closure["source_archive_sha256"] == (
        "e6ced21bef5926cb4fd6b9c79181980614eef3bf0fd7c14ac1cead63815cc069"
    )
    assert closure["artifact_file_count"] == 17


def test_ppce_parent_hash_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = (
        cli._ROOT / cli.PARENT_V15B_RUN_REPO_PATH / "COMPLETE.json"
    ).resolve()
    before = file_sha256(complete)
    monkeypatch.setattr(
        cli,
        "COVERAGE_STATE_V15B_PARENT_COMPLETE_SHA256",
        "0" * 64,
    )
    with pytest.raises(RuntimeError, match="COMPLETE changed"):
        cli._verify_parent_v15b_closure()
    assert file_sha256(complete) == before


def test_ppce_static_config_binds_actual_dataset_free_digest() -> None:
    sources = cli._verify_frozen_sources()
    implementation = cli._implementation_binding()
    actual_digest = (
        cli.COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT
    )
    config = cli._static_config_payload(
        source_paths=sources,
        implementation=implementation,
        dataset_free_receipt_fingerprint=actual_digest,
    )
    assert config["runtime_splits"] == ["D_R"]
    assert config["model"]["class"] == (
        "CURELitePhasePreservingCoverageStateLevelSet"
    )
    assert config["model"]["phase_occupancy_channels"] == 16
    assert config["model"]["parameter_count"] == 23856
    assert config["model"]["objective_suite"] == [
        "support_oriented_response_joint",
        "identity_joint",
        "separable_endpoint",
    ]
    assert config["dataset_free_gate"]["receipt_fingerprint"] == (
        actual_digest
    )
    assert config["dataset_free_gate"]["binding_mode"] == (
        "actual_runtime_receipt_fingerprint"
    )
    assert config["evidence_scope"]["training_performed"] is False
    assert config["evidence_scope"]["formal_800_authorized"] is False
    assert config["evidence_scope"]["full_CURE_authorized"] is False
    assert config["evidence_scope"]["cross_backbone_authorized"] is False
    with pytest.raises(ValueError, match="frozen PPCE gate"):
        cli._static_config_payload(
            source_paths=sources,
            implementation=implementation,
            dataset_free_receipt_fingerprint="not-a-digest",
        )


def test_create_only_uses_receipt_fingerprint_without_claiming_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_digest = (
        cli.COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_phase_preserving_dataset_free_gate",
        lambda: SimpleNamespace(
            all_pass=True,
            receipt_fingerprint=actual_digest,
        ),
    )
    existed_before = (
        cli.OUTPUT_PATH.exists() or cli.OUTPUT_PATH.is_symlink()
    )
    receipt = cli.validate_create_only()
    existed_after = (
        cli.OUTPUT_PATH.exists() or cli.OUTPUT_PATH.is_symlink()
    )
    assert existed_after is existed_before
    assert receipt["static_contract_valid"] is True
    assert receipt["dataset_free_receipt_fingerprint"] == actual_digest
    assert receipt["dataset_free_gate_passed"] is True
    assert receipt["run_once_implemented"] is True
    assert receipt["output_claimed"] is False
    assert receipt["D_R_cached_tensor_payload_accessed"] is False
    assert receipt["D_V_accessed"] is False
    assert receipt["D_T_accessed"] is False
    assert receipt["training_performed"] is False
    assert receipt["formal_800_authorized"] is False
    assert receipt["full_CURE_authorized"] is False
    assert receipt["cross_backbone_authorized"] is False
    assert receipt["not_a_formal_result"] is True


def test_create_only_rejects_dataset_free_receipt_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_coverage_state_phase_preserving_dataset_free_gate",
        lambda: SimpleNamespace(
            all_pass=True,
            receipt_fingerprint="0" * 64,
        ),
    )
    with pytest.raises(RuntimeError, match="receipt changed"):
        cli.validate_create_only()


def test_create_only_cli_has_no_training_or_output_override_surface() -> None:
    assert cli.parse_args(
        ("--validate-create-only",)
    ).validate_create_only
    with pytest.raises(SystemExit):
        cli.parse_args(())
    assert cli.parse_args(("--run-once",)).run_once
    with pytest.raises(SystemExit):
        cli.parse_args(
            (
                "--validate-create-only",
                "--output",
                "/tmp/not-authorized",
            )
        )


def test_implementation_binding_contains_ppce_core_runner_and_cli() -> None:
    binding = dict(cli._implementation_binding())
    required = {
        "cure_lite/coverage_state_phase_preserving.py",
        "cure_lite/experiment/coverage_state_ppce_bounded_runner.py",
        (
            "tools/"
            "run_coverage_state_cslf_ppce_support_oriented_bounded_400.py"
        ),
        "tools/run_coverage_state_cslf_support_oriented_bounded_400.py",
        "tools/run_with_gpu_temperature_control.py",
    }
    assert required <= set(binding)
    assert all(len(value) == 64 for value in binding.values())


def test_ppce_checkpoint_is_tensor_only_and_class_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    model = CURELitePhasePreservingCoverageStateLevelSet(
        CoverageStatePhasePreservingConfig(
            feature_channels=4,
            feature_stride=2,
            width=8,
        )
    )
    receipt = cli._write_checkpoint_new(
        directory,
        objective="support_oriented_response_joint",
        objective_policy=(
            "added_target_support_oriented_absolute_root_"
            "with_finite_coverage_response_v1"
        ),
        model=model,
    )
    assert receipt["model_class"] == (
        "CURELitePhasePreservingCoverageStateLevelSet"
    )
    assert receipt["model_config"]["parameter_count"] == (
        model.config.expected_parameter_count
    )
    assert receipt["tensor_only_state_dict"] is True
    assert receipt["weights_only_roundtrip_verified"] is True


def test_existing_output_stops_before_any_d_r_or_parent_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-claimed"
    output.mkdir()
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)

    def forbidden() -> object:
        raise AssertionError("no prerequisite may run after an existing claim")

    monkeypatch.setattr(cli, "_verify_frozen_sources", forbidden)
    monkeypatch.setattr(cli, "_verify_parent_v15b_closure", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.run_once()


def _patch_terminal_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    output = tmp_path / "bounded"
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    parent = {
        "complete_fingerprint": (
            cli.COVERAGE_STATE_V15B_PARENT_COMPLETE_FINGERPRINT
        )
    }
    monkeypatch.setattr(
        cli,
        "_verify_parent_v15b_closure",
        lambda: parent,
    )
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: (("implementation.py", "1" * 64),),
    )
    monkeypatch.setattr(
        cli._v15b_cli,
        "_verify_runtime_contract",
        lambda: {"device": "cuda:0"},
    )
    dataset_free = SimpleNamespace(
        all_pass=True,
        receipt_fingerprint=(
            cli.COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT
        ),
        canonical_payload=lambda: {"all_pass": True},
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_phase_preserving_dataset_free_gate",
        lambda: dataset_free,
    )
    monkeypatch.setattr(
        cli,
        "_static_config_payload",
        lambda **kwargs: {
            "schema_version": cli.RUN_SCHEMA,
            "dataset_free_receipt_fingerprint": (
                kwargs["dataset_free_receipt_fingerprint"]
            ),
        },
    )
    return dataset_free


def test_mocked_run_once_writes_complete_terminal_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_free = _patch_terminal_prerequisites(
        monkeypatch,
        tmp_path,
    )
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
        lambda value: population,
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
    authorization = SimpleNamespace(
        candidate_objective="support_oriented_response_joint",
        parent_v15b_complete_fingerprint=(
            cli.COVERAGE_STATE_V15B_PARENT_COMPLETE_FINGERPRINT
        ),
        authorization_fingerprint="3" * 64,
        training_authorized=True,
        canonical_payload=lambda: {"authorization": "stub"},
        verify_unchanged=lambda: None,
    )
    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_ppce_bounded_run_authorization",
        lambda *args: authorization,
    )
    config = CoverageStatePhasePreservingConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    monkeypatch.setattr(
        cli,
        "expected_coverage_state_ppce_config",
        lambda value: config,
    )
    monkeypatch.setattr(
        cli,
        "_device_memory_preflight",
        lambda *args: {
            "schema_version": "memory",
            "receipt_fingerprint": "4" * 64,
        },
    )
    names = (
        "support_oriented_response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    models = tuple(
        (
            name,
            CURELitePhasePreservingCoverageStateLevelSet(config),
        )
        for name in names
    )
    training = SimpleNamespace(
        results=tuple(
            SimpleNamespace(
                objective=name,
                objective_policy=(
                    cli.coverage_state_pair_objective_policy(name)
                ),
            )
            for name in names
        ),
        models=models,
        result_fingerprint="5" * 64,
        canonical_payload=lambda: {"training": "stub"},
    )

    def diagnostic(name: str, passed: bool) -> SimpleNamespace:
        return SimpleNamespace(
            bounded_gate_passed=passed,
            canonical_payload=lambda: {
                "objective": name,
                "bounded_gate_passed": passed,
                "input_representation": "phase_preserving",
            },
        )

    result = SimpleNamespace(
        authorization=authorization,
        training=training,
        diagnostics=(
            (names[0], diagnostic(names[0], True)),
            (names[1], diagnostic(names[1], False)),
            (names[2], diagnostic(names[2], False)),
        ),
        checks=(
            ("candidate_original_zero_level_gates", True),
            ("control_diagnostics_complete", True),
        ),
        bounded_gate_passed=True,
        failed_checks=(),
        result_fingerprint="6" * 64,
        canonical_payload=lambda: {"result": "stub"},
        verify_unchanged=lambda: None,
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_ppce_support_oriented_bounded_400",
        lambda *args, **kwargs: result,
    )
    terminal = cli.run_once()
    assert terminal["bounded_gate_passed"] is True
    assert not (cli.OUTPUT_PATH / ".incomplete").exists()
    assert not (cli.OUTPUT_PATH / "FAILURE.json").exists()
    complete = json.loads(
        (cli.OUTPUT_PATH / "COMPLETE.json").read_text()
    )
    assert complete["artifact_file_count"] == 17
    assert len(complete["artifact_files"]) == 17
    assert complete["bounded_gate_passed"] is True
    assert complete["resume_allowed"] is False
    zero = json.loads(
        (cli.OUTPUT_PATH / "receipts" / "zero_level.json").read_text()
    )
    assert zero["input_representation"] == "phase_preserving"
    assert zero["candidate_bounded_gate_passed"] is True
    assert dataset_free.receipt_fingerprint == (
        cli.COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT
    )


def test_mocked_run_once_failure_writes_nonresumable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_terminal_prerequisites(monkeypatch, tmp_path)

    def fail(**kwargs):
        raise RuntimeError("injected D_R construction failure")

    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        fail,
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
