from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from tools import (
    run_coverage_state_cmif_support_oriented_bounded_400 as cli,
)


def test_cmif_constants_and_persisted_p0_are_exact() -> None:
    assert cli.RUN_ID == (
        "cure_lite_cmif_v17_support_oriented_bounded_400_r1"
    )
    assert cli.FROZEN_FEATURE_CHANNELS == 64
    assert cli.FROZEN_FEATURE_STRIDE == 4
    assert cli.FROZEN_MODEL_WIDTH == 32
    assert cli.FROZEN_PARAMETER_COUNT == 64064
    assert cli.FROZEN_SEED == 42
    assert cli.FROZEN_UPDATES_PER_OBJECTIVE == 400
    p0 = cli._verify_cmif_p0_authorization()
    assert p0["r2_complete_fingerprint"] == (
        cli.COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT
    )
    assert p0["p0_core_receipt_fingerprint"] == (
        cli.COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT
    )
    assert p0["bounded_population_fingerprint"] == (
        cli.COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT
    )
    assert p0["training_authorized"] is True
    assert p0["D_V_accessed"] is False
    assert p0["D_T_accessed"] is False
    assert p0["training_performed"] is False


def test_p0_must_authorize_bounded_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "_verify_persisted_cmif_p0_authorization",
        lambda: {
            "training_authorized": False,
            "r2_complete_fingerprint": (
                cli.COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT
            ),
            "p0_core_receipt_fingerprint": (
                cli.COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT
            ),
            "bounded_population_fingerprint": (
                cli.COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT
            ),
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
        },
    )
    with pytest.raises(PermissionError, match="did not authorize"):
        cli._verify_cmif_p0_authorization()


def test_static_config_binds_cmif_and_four_layer_p0() -> None:
    sources = cli._verify_frozen_sources()
    implementation = cli._implementation_binding()
    p0 = cli._verify_cmif_p0_authorization()
    config = cli._static_config_payload(
        source_paths=sources,
        implementation=implementation,
        dataset_free_receipt_fingerprint=str(
            p0["dataset_free_receipt_fingerprint"]
        ),
        p0_evidence=p0,
    )
    assert config["runtime_splits"] == ["D_R"]
    assert config["model"]["class"] == (
        "CURELiteCenteredMixedInteractionLevelSet"
    )
    assert config["model"]["parameter_count"] == 64064
    assert config["model"]["feature_channels"] == 64
    assert config["model"]["feature_stride"] == 4
    assert config["model"]["width"] == 32
    assert config["model"]["objective_suite"] == [
        "support_oriented_response_joint",
        "identity_joint",
        "separable_endpoint",
    ]
    assert config["persisted_p0_authorization"] == p0
    assert config["evidence_scope"]["bounded_400_authorized"] is True
    assert config["evidence_scope"]["formal_800_authorized"] is False
    assert config["evidence_scope"]["full_CURE_authorized"] is False
    assert config["evidence_scope"]["cross_backbone_authorized"] is False
    with pytest.raises(ValueError, match="binding changed"):
        cli._static_config_payload(
            source_paths=sources,
            implementation=implementation,
            dataset_free_receipt_fingerprint="0" * 64,
            p0_evidence=p0,
        )


def test_create_only_does_not_claim_or_load_d_r(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p0 = cli._verify_cmif_p0_authorization()
    digest = str(p0["dataset_free_receipt_fingerprint"])
    monkeypatch.setattr(
        cli,
        "run_coverage_state_cmif_dataset_free_gate",
        lambda: SimpleNamespace(
            all_pass=True,
            receipt_fingerprint=digest,
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("create-only loaded D_R tensors")
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
    assert receipt["bounded_400_authorized"] is True
    assert receipt["output_claimed"] is False
    assert receipt["D_R_cached_tensor_payload_accessed"] is False
    assert receipt["D_V_accessed"] is False
    assert receipt["D_T_accessed"] is False
    assert receipt["training_performed"] is False
    assert receipt["formal_800_authorized"] is False
    assert receipt["not_a_formal_result"] is True


def test_cli_has_only_create_and_single_run_modes() -> None:
    assert cli.parse_args(
        ("--validate-create-only",)
    ).validate_create_only
    assert cli.parse_args(("--run-once",)).run_once
    with pytest.raises(SystemExit):
        cli.parse_args(())
    with pytest.raises(SystemExit):
        cli.parse_args(
            (
                "--validate-create-only",
                "--output",
                "/tmp/not-authorized",
            )
        )


def test_implementation_binding_contains_cmif_core_cli_and_wrapper() -> None:
    binding = dict(cli._implementation_binding())
    required = {
        "cure_lite/coverage_state_centered_mixed_interaction.py",
        "cure_lite/experiment/coverage_state_cmif_dataset_free.py",
        "cure_lite/experiment/coverage_state_cmif_p0.py",
        "cure_lite/experiment/coverage_state_cmif_bounded_runner.py",
        (
            "tools/"
            "run_coverage_state_cmif_support_oriented_bounded_400.py"
        ),
        (
            "tools/"
            "run_coverage_state_cslf_ppce_support_oriented_bounded_400.py"
        ),
        "tools/run_with_gpu_temperature_control.py",
    }
    assert required <= set(binding)
    assert all(len(value) == 64 for value in binding.values())


def test_cmif_checkpoint_is_tensor_only_and_exact_class_bound(
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
        objective="support_oriented_response_joint",
        objective_policy=(
            "added_target_support_oriented_absolute_root_"
            "with_finite_coverage_response_v1"
        ),
        model=model,
    )
    assert receipt["model_class"] == (
        "CURELiteCenteredMixedInteractionLevelSet"
    )
    assert receipt["model_config"]["parameter_count"] == (
        config.expected_parameter_count
    )
    assert receipt["model_config"]["interaction_policy"] == (
        config.interaction_policy
    )
    assert receipt["tensor_only_state_dict"] is True
    assert receipt["weights_only_roundtrip_verified"] is True
    with pytest.raises(TypeError, match="exact model class"):
        cli._write_checkpoint_new(
            directory,
            objective="wrong",
            objective_policy="wrong",
            model=object(),  # type: ignore[arg-type]
        )


def test_decision_uses_candidate_only_and_never_authorizes_formal800() -> None:
    candidate = SimpleNamespace(
        factual_miss_gate_passed=True,
        factual_no_miss_gate_passed=True,
        clean_defined_metrics_passed=True,
        clean_compact_support_gate_passed=True,
        component_null_gate_passed=True,
        identity_null_gate_passed=True,
        diagnostic_null_gate_passed=True,
    )
    authorization = SimpleNamespace(
        candidate_objective="support_oriented_response_joint",
        p0_evidence_fingerprint="1" * 64,
    )
    result = SimpleNamespace(
        authorization=authorization,
        bounded_gate_passed=True,
        failed_checks=(),
        result_fingerprint="2" * 64,
        checks=(
            ("candidate_original_zero_level_gates", True),
            ("control_diagnostics_complete", True),
        ),
        diagnostics=(
            ("support_oriented_response_joint", candidate),
        ),
    )
    receipt = cli._decision_payload(
        result,
        (
            {
                "objective": "support_oriented_response_joint",
                "receipt_fingerprint": "3" * 64,
            },
        ),
    )
    assert receipt["status"] == (
        "CMIF_V17_BOUNDED_400_GATE_PASS"
    )
    assert receipt["candidate_gate_passed"] is True
    assert receipt["control_outcomes_are_not_candidate_gates"] is True
    assert receipt["formal_800_authorized"] is False
    assert receipt["performance_claim_supported"] is False


def test_compact_only_failure_is_frozen_for_independent_v18() -> None:
    candidate = SimpleNamespace(
        factual_miss_gate_passed=True,
        factual_no_miss_gate_passed=True,
        clean_defined_metrics_passed=True,
        clean_compact_support_gate_passed=False,
        component_null_gate_passed=True,
        identity_null_gate_passed=True,
        diagnostic_null_gate_passed=True,
    )
    authorization = SimpleNamespace(
        candidate_objective="support_oriented_response_joint",
        p0_evidence_fingerprint="1" * 64,
    )
    result = SimpleNamespace(
        authorization=authorization,
        bounded_gate_passed=False,
        failed_checks=("candidate_original_zero_level_gates",),
        result_fingerprint="2" * 64,
        checks=(("candidate_original_zero_level_gates", False),),
        diagnostics=(
            ("support_oriented_response_joint", candidate),
        ),
    )
    receipt = cli._decision_payload(result, ())
    assert receipt["status"] == "CMIF_V17_BOUNDED_400_GATE_FAIL"
    assert receipt["compact_gate_only_failure"] is True
    assert receipt["next_action"] == (
        "freeze_v17_and_review_objective_only_in_independent_v18"
    )
    assert receipt["formal_800_authorized"] is False


def test_existing_output_stops_before_p0_or_d_r_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-claimed"
    output.mkdir()
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)

    def forbidden() -> object:
        raise AssertionError("no prerequisite may run after existing output")

    monkeypatch.setattr(cli, "_verify_frozen_sources", forbidden)
    monkeypatch.setattr(cli, "_verify_cmif_p0_authorization", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.run_once()


def test_p0_failure_stops_before_output_claim_and_d_r_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "bounded"
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_verify_cmif_p0_authorization",
        lambda: (_ for _ in ()).throw(
            PermissionError("injected P0 failure")
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("D_R must not load")
        ),
    )
    monkeypatch.setattr(
        cli._v15b_cli,
        "_claim_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("output must not be claimed")
        ),
    )
    with pytest.raises(PermissionError, match="injected P0 failure"):
        cli.run_once()
    assert not output.exists()


def _patch_terminal_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[SimpleNamespace, dict[str, object]]:
    output = tmp_path / "bounded"
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    p0 = {
        "evidence_fingerprint": "0" * 64,
        "r2_complete_fingerprint": (
            cli.COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT
        ),
        "p0_core_receipt_fingerprint": (
            cli.COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT
        ),
        "bounded_population_fingerprint": (
            cli.COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT
        ),
        "dataset_free_receipt_fingerprint": "9" * 64,
        "training_authorized": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    monkeypatch.setattr(
        cli,
        "_verify_cmif_p0_authorization",
        lambda: p0,
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
        receipt_fingerprint="9" * 64,
        canonical_payload=lambda: {"all_pass": True},
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_state_cmif_dataset_free_gate",
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
    return dataset_free, p0


def test_mocked_run_once_writes_complete_terminal_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_free, p0 = _patch_terminal_prerequisites(
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
        p0_evidence_fingerprint=str(p0["evidence_fingerprint"]),
        authorization_fingerprint="3" * 64,
        training_authorized=True,
        canonical_payload=lambda: {"authorization": "stub"},
        verify_unchanged=lambda: None,
    )
    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_cmif_bounded_run_authorization",
        lambda *args: authorization,
    )
    config = CoverageStateCenteredMixedInteractionConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    monkeypatch.setattr(
        cli,
        "expected_coverage_state_cmif_config",
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
            CURELiteCenteredMixedInteractionLevelSet(config),
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
            factual_miss_gate_passed=passed,
            factual_no_miss_gate_passed=passed,
            clean_defined_metrics_passed=passed,
            clean_compact_support_gate_passed=passed,
            component_null_gate_passed=passed,
            identity_null_gate_passed=passed,
            diagnostic_null_gate_passed=passed,
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
        "run_coverage_state_cmif_support_oriented_bounded_400",
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
    assert complete["persisted_p0_authorization"] == p0
    assert complete["formal_800_authorized"] is False
    zero = json.loads(
        (cli.OUTPUT_PATH / "receipts" / "zero_level.json").read_text()
    )
    assert zero["input_representation"] == "phase_preserving"
    assert zero["candidate_bounded_gate_passed"] is True
    assert dataset_free.receipt_fingerprint == "9" * 64


def test_mocked_run_once_failure_writes_nonresumable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_terminal_prerequisites(monkeypatch, tmp_path)

    def fail(**kwargs: object) -> object:
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
